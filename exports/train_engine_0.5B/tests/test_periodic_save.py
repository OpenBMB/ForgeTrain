"""Unit tests for the periodic resume-checkpoint schedule.

Covers two related things:

  1. The pure ``should_save_after_step`` policy used by both the
     ``run_training_loop`` integration and the ``train`` / ``pretrain``
     CLI entry points. Boundary semantics matter — an off-by-one here
     would silently change which step number ends up in the saved
     ``training_state.pt`` and cause ``resume-gate-200`` to look like a
     correctness regression.

  2. A simulated periodic-save loop that drives the same
     ``save_checkpoint(async_io=True)`` + ``wait_for_async_save()``
     handshake the train loop performs, so that the final on-disk
     checkpoint always reflects the *last* save boundary even when
     consecutive saves happen back-to-back.

These tests deliberately avoid the GPU-bound parts of
``run_training_loop``; the goal is to catch scheduler / async-handshake
regressions in ``harness run unit`` (CPU-only) instead of waiting for a
multi-GPU cctl job.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestShouldSaveAfterStep(unittest.TestCase):
    """Boundary semantics of the pure save-policy helper."""

    def test_disabled_when_save_interval_is_none(self) -> None:
        from training_engine_tensor.parameters import should_save_after_step

        for s in (0, 1, 50, 99, 100, 199):
            self.assertFalse(
                should_save_after_step(completed_step=s, save_interval=None),
                f"None save_interval must never trigger a save (step={s})",
            )

    def test_disabled_when_save_interval_non_positive(self) -> None:
        from training_engine_tensor.parameters import should_save_after_step

        for interval in (0, -1, -100):
            for s in (0, 1, 50, 99, 100):
                self.assertFalse(
                    should_save_after_step(
                        completed_step=s, save_interval=interval,
                    ),
                    f"non-positive interval={interval} must not trigger "
                    f"(step={s})",
                )

    def test_save_at_canonical_resume_boundary_step_99(self) -> None:
        """``save_interval=100`` must fire at completed_step ∈ {99, 199, ...}.

        This mirrors the ``RESUME_SAVE_STEP=100`` semantics in
        ``test_resume_train.py`` — after step 99 finishes, the checkpoint
        is persisted with ``step=100`` (the next step to run).
        """
        from training_engine_tensor.parameters import should_save_after_step

        for s in (99, 199, 299, 999):
            self.assertTrue(
                should_save_after_step(completed_step=s, save_interval=100),
                f"interval=100 must fire at completed_step={s}",
            )
        for s in (0, 1, 50, 98, 100, 101, 198, 200):
            self.assertFalse(
                should_save_after_step(completed_step=s, save_interval=100),
                f"interval=100 must NOT fire at completed_step={s}",
            )

    def test_save_every_step_when_interval_is_one(self) -> None:
        from training_engine_tensor.parameters import should_save_after_step

        for s in range(0, 25):
            self.assertTrue(
                should_save_after_step(completed_step=s, save_interval=1),
                f"interval=1 must fire every step (step={s})",
            )

    def test_save_at_first_boundary_for_small_interval(self) -> None:
        """interval=N → first save at completed_step=N-1 (i.e. step #N)."""
        from training_engine_tensor.parameters import should_save_after_step

        for interval in (2, 3, 5, 7, 100):
            self.assertTrue(
                should_save_after_step(
                    completed_step=interval - 1, save_interval=interval,
                ),
                f"interval={interval} first save must be at "
                f"completed_step={interval - 1}",
            )
            self.assertFalse(
                should_save_after_step(
                    completed_step=interval - 2, save_interval=interval,
                ),
                f"interval={interval} must NOT fire one step early",
            )


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestPeriodicSaveSimulation(unittest.TestCase):
    """Simulate the periodic-save loop without the GPU training body.

    The mini loop iterates ``step_indices``, queries the same
    ``should_save_after_step`` policy the real loop uses, and dispatches
    ``save_checkpoint(async_io=True)`` at each boundary. After the loop
    a final ``wait_for_async_save()`` drains any in-flight write — this
    is the exact handshake ``run_training_loop`` performs at the end.
    """

    def setUp(self) -> None:
        import torch

        from training_engine_tensor.engine_config import EngineConfig, set_global_config
        set_global_config(EngineConfig())

        self.torch = torch
        self.tmp_dir = tempfile.mkdtemp(prefix="periodic_save_")
        self.addCleanup(self._cleanup)
        self.param_names = ["norm.weight", "tok_embeddings.weight"]

    def _cleanup(self) -> None:
        import contextlib
        import shutil

        from training_engine_tensor.parameters import wait_for_async_save
        with contextlib.suppress(Exception):
            wait_for_async_save()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build_state(self, step_offset: int):
        """Build a synthetic params + AdamState whose values depend on step.

        Mutating per-step lets us tell different saves apart on disk.
        """
        torch = self.torch
        from training_engine_tensor.optimizer import AdamState

        params = {
            "norm.weight": torch.full((4,), 1.0 + step_offset, dtype=torch.float32),
            "tok_embeddings.weight": (
                torch.arange(12, dtype=torch.float32).reshape(3, 4)
                + float(step_offset)
            ),
        }
        opt = AdamState(self.param_names, params, device="cpu")
        opt.step_count = step_offset
        opt.num_samples = 160 * step_offset
        for name in self.param_names:
            opt.exp_avg[name].fill_(0.1 * step_offset)
            opt.exp_avg_sq[name].fill_(0.01 * step_offset)
        return params, opt

    def _drive_loop(
        self,
        *,
        num_steps: int,
        save_interval: int | None,
        save_dir: str | None,
    ) -> list[int]:
        """Run the simulated loop. Returns the list of saved step values."""
        from training_engine_tensor.parameters import (
            save_checkpoint,
            should_save_after_step,
            wait_for_async_save,
        )

        saved_steps: list[int] = []
        for step_idx in range(num_steps):
            params, opt = self._build_state(step_offset=step_idx + 1)
            if save_dir and should_save_after_step(
                completed_step=step_idx, save_interval=save_interval,
            ):
                save_checkpoint(
                    checkpoint_dir=save_dir,
                    params=params,
                    optimizer_state=opt,
                    step=step_idx + 1,
                    async_io=True,
                )
                saved_steps.append(step_idx + 1)
        wait_for_async_save()
        return saved_steps

    def test_disabled_save_interval_writes_nothing(self) -> None:
        from training_engine_tensor.parameters import _CHECKPOINT_FILENAME

        saved = self._drive_loop(
            num_steps=10, save_interval=None, save_dir=self.tmp_dir,
        )
        self.assertEqual(saved, [])
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)),
            "no save_interval → no checkpoint must be written",
        )

    def test_no_save_dir_writes_nothing(self) -> None:
        """Even with a positive interval, no save_dir → no save call."""
        from training_engine_tensor.parameters import _CHECKPOINT_FILENAME

        saved = self._drive_loop(
            num_steps=10, save_interval=2, save_dir=None,
        )
        self.assertEqual(saved, [])
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)),
        )

    def test_periodic_saves_land_at_expected_boundaries(self) -> None:
        saved = self._drive_loop(
            num_steps=25, save_interval=5, save_dir=self.tmp_dir,
        )
        self.assertEqual(saved, [5, 10, 15, 20, 25])

    def test_final_on_disk_state_reflects_last_save(self) -> None:
        """Back-to-back async saves: file must reflect the *last* boundary."""
        from training_engine_tensor.parameters import load_resume_checkpoint

        saved = self._drive_loop(
            num_steps=20, save_interval=4, save_dir=self.tmp_dir,
        )
        self.assertEqual(saved, [4, 8, 12, 16, 20])

        ckpt = load_resume_checkpoint(self.tmp_dir, device="cpu")
        self.assertEqual(ckpt["step"], 20, "final file must be the last save")
        # Synthetic state was built so every field encodes its own step.
        # load_resume_checkpoint converts params to compute_dtype (BF16).
        loaded_dtype = ckpt["params"]["norm.weight"].dtype
        self.torch.testing.assert_close(
            ckpt["params"]["norm.weight"],
            self.torch.full((4,), 21.0, dtype=loaded_dtype),
            atol=0.0, rtol=0.0,
        )
        self.assertEqual(ckpt["num_samples"], 160 * 20)

    def test_wait_for_async_save_returns_false_after_drain(self) -> None:
        """Final wait inside _drive_loop must leave no in-flight save."""
        from training_engine_tensor.parameters import wait_for_async_save

        self._drive_loop(
            num_steps=10, save_interval=3, save_dir=self.tmp_dir,
        )
        self.assertFalse(
            wait_for_async_save(),
            "after the loop's terminal wait there must be no pending save",
        )


if __name__ == "__main__":
    unittest.main()
