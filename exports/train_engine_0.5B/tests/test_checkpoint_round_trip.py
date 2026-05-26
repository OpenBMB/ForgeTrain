"""CPU round-trip tests for resume checkpoint save/load.

These tests exercise the on-disk checkpoint format and the
``AdamState.state_dict() / load_state_dict()`` helpers without
touching CUDA, the dataloader, or external training frameworks.
They lock the contract that the resume-gate-200 suite depends on
at the format level, so a regression that breaks the disk shape
(e.g. renaming a key, forgetting to detach to CPU, dropping a
field) is caught by the local ``harness run unit`` invocation
instead of waiting for a multi-GPU cctl job.

The tests are skipped automatically when ``torch`` isn't importable
(e.g. on a developer Mac), so they cost nothing locally; on the
remote / CI machines that do have torch they run for free.
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
class TestCheckpointRoundTrip(unittest.TestCase):
    """Verify save_checkpoint → load_resume_checkpoint preserves all fields."""

    def setUp(self) -> None:
        import torch

        from training_engine_tensor.engine_config import EngineConfig, set_global_config
        set_global_config(EngineConfig())

        self.torch = torch
        self.tmp_dir = tempfile.mkdtemp(prefix="ckpt_rt_")
        self.addCleanup(self._cleanup)

        # Tiny synthetic param + optimizer state on CPU. Using ``norm.weight``
        # (1D) and one matrix param so we exercise both wd / no-wd buckets
        # without standing up the full 24-layer model.
        self.param_names = ["norm.weight", "tok_embeddings.weight"]
        self.params = {
            "norm.weight": torch.full((4,), 1.5, dtype=torch.float32),
            "tok_embeddings.weight": torch.arange(
                12, dtype=torch.float32
            ).reshape(3, 4),
        }

    def _cleanup(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build_opt(self):
        from training_engine_tensor.optimizer import AdamState

        opt = AdamState(self.param_names, self.params, device="cpu")
        # Mutate state to non-default values so a round-trip that drops
        # a field would surface as a comparison failure.
        opt.step_count = 7
        opt.num_samples = 160 * 7
        for name in self.param_names:
            opt.master_weights[name].mul_(1.25)
            opt.exp_avg[name].fill_(0.5)
            opt.exp_avg_sq[name].fill_(0.25)
        return opt

    def test_round_trip_preserves_all_fields(self) -> None:
        from training_engine_tensor.optimizer import AdamState
        from training_engine_tensor.parameters import (
            load_resume_checkpoint,
            save_checkpoint,
        )

        opt = self._build_opt()
        save_checkpoint(
            checkpoint_dir=self.tmp_dir,
            params=self.params,
            optimizer_state=opt,
            step=42,
        )

        ckpt = load_resume_checkpoint(self.tmp_dir, device="cpu")

        self.assertEqual(ckpt["step"], 42)
        self.assertEqual(ckpt["num_samples"], 160 * 7)
        self.assertEqual(ckpt["param_names"], self.param_names)
        self.assertIsNone(ckpt["dataloader_state"])

        # Params and optimizer-state tensors must round-trip bitwise.
        for name in self.param_names:
            self.torch.testing.assert_close(
                ckpt["params"][name], self.params[name].to(ckpt["params"][name].dtype),
                atol=0.0, rtol=0.0,
            )

        # Apply via load_state_dict and confirm fields land where expected.
        opt2 = AdamState(self.param_names, self.params, device="cpu")
        opt2.load_state_dict(ckpt["optimizer_state"])

        self.assertEqual(opt2.step_count, opt.step_count)
        self.assertEqual(opt2.num_samples, opt.num_samples)
        for name in self.param_names:
            self.torch.testing.assert_close(
                opt2.master_weights[name], opt.master_weights[name],
                atol=0.0, rtol=0.0,
            )
            self.torch.testing.assert_close(
                opt2.exp_avg[name], opt.exp_avg[name], atol=0.0, rtol=0.0,
            )
            self.torch.testing.assert_close(
                opt2.exp_avg_sq[name], opt.exp_avg_sq[name], atol=0.0, rtol=0.0,
            )

    def test_save_default_num_samples_uses_optimizer(self) -> None:
        """When num_samples kwarg is omitted, the value comes from opt."""
        from training_engine_tensor.parameters import (
            load_resume_checkpoint,
            save_checkpoint,
        )

        opt = self._build_opt()
        save_checkpoint(
            checkpoint_dir=self.tmp_dir,
            params=self.params,
            optimizer_state=opt,
            step=10,
        )
        ckpt = load_resume_checkpoint(self.tmp_dir, device="cpu")
        self.assertEqual(ckpt["num_samples"], opt.num_samples)
        self.assertEqual(ckpt["optimizer_state"]["num_samples"], opt.num_samples)

    def test_save_rejects_mismatched_num_samples(self) -> None:
        from training_engine_tensor.parameters import save_checkpoint

        opt = self._build_opt()
        with self.assertRaises(ValueError):
            save_checkpoint(
                checkpoint_dir=self.tmp_dir,
                params=self.params,
                optimizer_state=opt,
                step=10,
                num_samples=opt.num_samples + 1,  # caller mistake
            )

    def test_load_missing_file_raises_filenotfound(self) -> None:
        from training_engine_tensor.parameters import load_resume_checkpoint

        with self.assertRaises(FileNotFoundError):
            load_resume_checkpoint(
                os.path.join(self.tmp_dir, "no_such_subdir"), device="cpu",
            )

    def test_load_rejects_format_version_mismatch(self) -> None:
        from training_engine_tensor.parameters import (
            _CHECKPOINT_FILENAME,
            load_resume_checkpoint,
        )

        bad = {"format_version": 999, "step": 0, "num_samples": 0,
               "params": {}, "optimizer_state": {}}
        path = os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)
        self.torch.save(bad, path)

        with self.assertRaises(ValueError):
            load_resume_checkpoint(self.tmp_dir, device="cpu")

    def test_load_rejects_missing_required_keys(self) -> None:
        from training_engine_tensor.parameters import (
            _CHECKPOINT_FILENAME,
            _CHECKPOINT_FORMAT_VERSION,
            load_resume_checkpoint,
        )

        bad = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "step": 0,
            # ``num_samples`` and ``params`` deliberately missing.
            "optimizer_state": {},
        }
        path = os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)
        self.torch.save(bad, path)

        with self.assertRaises(ValueError):
            load_resume_checkpoint(self.tmp_dir, device="cpu")

    def test_save_atomic_no_orphan_tmp_on_success(self) -> None:
        from training_engine_tensor.parameters import (
            _CHECKPOINT_FILENAME,
            save_checkpoint,
        )

        opt = self._build_opt()
        save_checkpoint(
            checkpoint_dir=self.tmp_dir,
            params=self.params,
            optimizer_state=opt,
            step=1,
        )
        path = os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)
        tmp = path + ".tmp"
        self.assertTrue(os.path.exists(path))
        self.assertFalse(
            os.path.exists(tmp),
            "save_checkpoint must atomically rename tmp → final and leave no .tmp behind",
        )

    def test_save_cleans_tmp_on_failure(self) -> None:
        """If torch.save fails, the .tmp file is removed."""
        from training_engine_tensor.parameters import (
            _CHECKPOINT_FILENAME,
            save_checkpoint,
        )

        opt = self._build_opt()
        path = os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)
        tmp = path + ".tmp"

        from training_engine_tensor import parameters as params_mod

        original_save = params_mod.torch.save

        def _failing_save(payload, file_path):
            # Touch the tmp file so we can verify cleanup, then raise.
            with open(file_path, "wb") as fh:
                fh.write(b"\x00")
            raise RuntimeError("simulated I/O failure")

        params_mod.torch.save = _failing_save
        try:
            with self.assertRaises(RuntimeError):
                save_checkpoint(
                    checkpoint_dir=self.tmp_dir,
                    params=self.params,
                    optimizer_state=opt,
                    step=2,
                )
        finally:
            params_mod.torch.save = original_save

        self.assertFalse(os.path.exists(path), "no final file should be created on failure")
        self.assertFalse(os.path.exists(tmp), "tmp file must be cleaned up on failure")

    def test_save_non_zero_rank_is_noop(self) -> None:
        """A non-rank-0 caller must not write to disk."""
        from unittest.mock import patch

        from training_engine_tensor.parameters import (
            _CHECKPOINT_FILENAME,
            save_checkpoint,
        )

        opt = self._build_opt()
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=3):
            save_checkpoint(
                checkpoint_dir=self.tmp_dir,
                params=self.params,
                optimizer_state=opt,
                step=5,
            )

        path = os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)
        self.assertFalse(os.path.exists(path))


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestAsyncCheckpointSave(unittest.TestCase):
    """Verify save_checkpoint(async_io=True) plus wait_for_async_save."""

    def setUp(self) -> None:
        import torch

        from training_engine_tensor.engine_config import EngineConfig, set_global_config
        set_global_config(EngineConfig())

        self.torch = torch
        self.tmp_dir = tempfile.mkdtemp(prefix="ckpt_async_")
        self.addCleanup(self._cleanup)

        self.param_names = ["norm.weight", "tok_embeddings.weight"]
        self.params = {
            "norm.weight": torch.full((4,), 1.5, dtype=torch.float32),
            "tok_embeddings.weight": torch.arange(
                12, dtype=torch.float32
            ).reshape(3, 4),
        }
        # Drain any pending save left over from a previous test (defensive;
        # _pending_save is module-level state).
        from training_engine_tensor.parameters import wait_for_async_save
        wait_for_async_save()

    def _cleanup(self) -> None:
        import contextlib
        import shutil

        from training_engine_tensor.parameters import wait_for_async_save
        with contextlib.suppress(Exception):
            wait_for_async_save()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build_opt(self):
        from training_engine_tensor.optimizer import AdamState

        opt = AdamState(self.param_names, self.params, device="cpu")
        opt.step_count = 11
        opt.num_samples = 160 * 11
        for name in self.param_names:
            opt.master_weights[name].mul_(1.5)
            opt.exp_avg[name].fill_(0.7)
            opt.exp_avg_sq[name].fill_(0.49)
        return opt

    def test_async_save_round_trip(self) -> None:
        """async_io=True path must produce a file readable by load_resume_checkpoint."""
        from training_engine_tensor.parameters import (
            load_resume_checkpoint,
            save_checkpoint,
            wait_for_async_save,
        )

        opt = self._build_opt()
        save_checkpoint(
            checkpoint_dir=self.tmp_dir,
            params=self.params,
            optimizer_state=opt,
            step=99,
            async_io=True,
        )
        # File may not exist yet — that's the whole point of async.
        # wait_for_async_save() blocks until it does.
        completed = wait_for_async_save()
        self.assertTrue(completed, "wait_for_async_save() should return True when a save was pending")

        ckpt = load_resume_checkpoint(self.tmp_dir, device="cpu")
        self.assertEqual(ckpt["step"], 99)
        self.assertEqual(ckpt["num_samples"], 160 * 11)
        for name in self.param_names:
            self.torch.testing.assert_close(
                ckpt["params"][name],
                self.params[name].to(ckpt["params"][name].dtype),
                atol=0.0, rtol=0.0,
            )
            self.torch.testing.assert_close(
                ckpt["optimizer_state"]["master_weights"][name],
                opt.master_weights[name],
                atol=0.0, rtol=0.0,
            )

    def test_wait_returns_false_when_no_pending_save(self) -> None:
        from training_engine_tensor.parameters import wait_for_async_save

        # Nothing in flight → wait returns False, does not raise.
        self.assertFalse(wait_for_async_save())

    def test_async_save_propagates_errors(self) -> None:
        """An exception in the bg thread must surface on wait_for_async_save."""
        from training_engine_tensor import parameters as params_mod
        from training_engine_tensor.parameters import (
            save_checkpoint,
            wait_for_async_save,
        )

        opt = self._build_opt()
        original_save = params_mod.torch.save

        def _failing_save(payload, file_path):
            raise RuntimeError("simulated bg I/O failure")

        params_mod.torch.save = _failing_save
        try:
            save_checkpoint(
                checkpoint_dir=self.tmp_dir,
                params=self.params,
                optimizer_state=opt,
                step=1,
                async_io=True,
            )
            with self.assertRaises(RuntimeError):
                wait_for_async_save()
        finally:
            params_mod.torch.save = original_save

        # And after the failure surfaced, the pending slot is cleared so a
        # subsequent wait does not re-raise the same error.
        self.assertFalse(wait_for_async_save())

    def test_async_save_auto_waits_on_subsequent_save(self) -> None:
        """A second save call must transparently wait for the first to finish."""
        import threading
        import time

        from training_engine_tensor import parameters as params_mod
        from training_engine_tensor.parameters import (
            _CHECKPOINT_FILENAME,
            load_resume_checkpoint,
            save_checkpoint,
            wait_for_async_save,
        )

        opt = self._build_opt()
        original_save = params_mod.torch.save
        gate = threading.Event()
        invocation_count = [0]

        def _slow_then_fast_save(payload, file_path):
            invocation_count[0] += 1
            if invocation_count[0] == 1:
                # First call: stall until the gate releases so we can prove
                # the second save_checkpoint had to wait for us.
                gate.wait(timeout=5.0)
            return original_save(payload, file_path)

        params_mod.torch.save = _slow_then_fast_save
        try:
            save_checkpoint(
                checkpoint_dir=self.tmp_dir,
                params=self.params,
                optimizer_state=opt,
                step=1,
                async_io=True,
            )

            # Schedule the gate release so the second save_checkpoint —
            # which must internally wait for the first — eventually
            # progresses. A small sleep makes the ordering deterministic.
            def _release_after():
                time.sleep(0.05)
                gate.set()
            threading.Thread(target=_release_after, daemon=True).start()

            t0 = time.time()
            save_checkpoint(
                checkpoint_dir=self.tmp_dir,
                params=self.params,
                optimizer_state=opt,
                step=2,
                async_io=True,
            )
            elapsed = time.time() - t0

            wait_for_async_save()
        finally:
            params_mod.torch.save = original_save
            gate.set()

        # The second save had to wait at least until the gate fired.
        self.assertGreaterEqual(elapsed, 0.04)

        # File contents reflect the *second* save (step=2).
        path = os.path.join(self.tmp_dir, _CHECKPOINT_FILENAME)
        self.assertTrue(os.path.exists(path))
        ckpt = load_resume_checkpoint(self.tmp_dir, device="cpu")
        self.assertEqual(ckpt["step"], 2)

    def test_async_save_then_sync_save_is_safe(self) -> None:
        """Mixing async + sync save must not corrupt the on-disk state."""
        from training_engine_tensor.parameters import (
            load_resume_checkpoint,
            save_checkpoint,
            wait_for_async_save,
        )

        opt = self._build_opt()
        save_checkpoint(
            checkpoint_dir=self.tmp_dir,
            params=self.params,
            optimizer_state=opt,
            step=10,
            async_io=True,
        )
        # Synchronous save must wait for the async one to finish before
        # writing, so the final on-disk file always reflects step=11.
        save_checkpoint(
            checkpoint_dir=self.tmp_dir,
            params=self.params,
            optimizer_state=opt,
            step=11,
        )
        # No outstanding async work after a sync save.
        self.assertFalse(wait_for_async_save())
        ckpt = load_resume_checkpoint(self.tmp_dir, device="cpu")
        self.assertEqual(ckpt["step"], 11)


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestAdamStateLoadStateDict(unittest.TestCase):
    """Verify the AdamState helpers themselves (independent of disk IO)."""

    def setUp(self) -> None:
        import torch

        self.torch = torch
        self.param_names = ["a", "b"]
        self.bf16_params = {
            "a": torch.zeros(3, dtype=torch.bfloat16),
            "b": torch.zeros((2, 2), dtype=torch.bfloat16),
        }

    def test_state_dict_round_trip_in_memory(self) -> None:
        from training_engine_tensor.optimizer import AdamState

        opt = AdamState(self.param_names, self.bf16_params, device="cpu")
        opt.step_count = 5
        opt.num_samples = 1234
        opt.master_weights["a"].fill_(7.0)
        opt.exp_avg["b"].fill_(0.1)

        snapshot = opt.state_dict()

        opt2 = AdamState(self.param_names, self.bf16_params, device="cpu")
        opt2.load_state_dict(snapshot)

        self.assertEqual(opt2.step_count, 5)
        self.assertEqual(opt2.num_samples, 1234)
        self.torch.testing.assert_close(
            opt2.master_weights["a"], opt.master_weights["a"], atol=0.0, rtol=0.0,
        )
        self.torch.testing.assert_close(
            opt2.exp_avg["b"], opt.exp_avg["b"], atol=0.0, rtol=0.0,
        )

    def test_load_state_dict_strict_missing_field(self) -> None:
        from training_engine_tensor.optimizer import AdamState

        opt = AdamState(self.param_names, self.bf16_params, device="cpu")
        partial = opt.state_dict()
        partial.pop("exp_avg_sq")
        with self.assertRaises(KeyError):
            opt.load_state_dict(partial)

    def test_load_state_dict_strict_missing_param_key(self) -> None:
        from training_engine_tensor.optimizer import AdamState

        opt = AdamState(self.param_names, self.bf16_params, device="cpu")
        snap = opt.state_dict()
        snap["master_weights"].pop("b")  # corrupt: key for 'b' missing
        with self.assertRaises(KeyError):
            opt.load_state_dict(snap)

    def test_load_state_dict_non_strict_tolerates_missing_field(self) -> None:
        from training_engine_tensor.optimizer import AdamState

        opt = AdamState(self.param_names, self.bf16_params, device="cpu")
        opt.step_count = 3
        snap = opt.state_dict()
        snap.pop("num_samples")
        opt.num_samples = 42
        opt.load_state_dict(snap, strict=False)
        # num_samples preserved (because the field was absent in the snap).
        self.assertEqual(opt.num_samples, 42)
        # step_count restored from the snap.
        self.assertEqual(opt.step_count, 3)

    def test_load_state_dict_rejects_non_dict(self) -> None:
        from training_engine_tensor.optimizer import AdamState

        opt = AdamState(self.param_names, self.bf16_params, device="cpu")
        with self.assertRaises(TypeError):
            opt.load_state_dict("not a dict")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
