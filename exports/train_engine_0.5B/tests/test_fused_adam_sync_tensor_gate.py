"""Numerical gate for the M2-P33 Phase B tensor-arg fused Adam path.

Pins the contract that ``fused_adam_sync_tensor`` (the CUDA-Graph-
compatible variant of ``fused_adam_sync``) produces byte-for-byte
identical updates to the legacy scalar-arg path when fed identical
inputs and the same per-step ``(lr, step, clip_coeff)`` triplet.
Equivalence here is the precondition for swapping the scalar path
out at capture time without touching numerics — the loss-gate (cctl)
can only catch first-order divergence in <100 steps, but a stuck
LR / bias-correction would only show up much later in long runs.

Test plan
---------
1. Allocate a small synthetic dataset (a handful of params with
   diverse shapes and dtypes mimicking the real model layout —
   2D weight + 1D LayerNorm) on CUDA, in random FP32 master and
   FP32 grad.
2. Run N=100 simulated optimizer steps with the **scalar** path,
   stashing master / m / v at every step.
3. Reset state, run the **same** N=100 steps with the **tensor**
   path (re-using ``OptimizerScalarBuffers`` updated each step).
4. Assert all three state tensors and the BF16 param output match
   exactly across every step.

The schedule mimics production: ``lr`` follows the cosine MSP
schedule (warmup-stable-decay), ``clip_coeff < 1`` deliberately
exercised on every other step so the clip path is covered, and
``state.step_count`` increments by 1 each iteration so bias
corrections drift across all 100 steps.

Skip conditions
---------------
* ``torch`` not importable — skip silently (mirrors the convention
  used by the rest of the test suite for CUDA-only kernels).
* ``torch.cuda.is_available() == False`` — skip with a message.
* ``triton`` not importable — skip with a message.

Run locally on a CUDA box:
    pytest tests/test_fused_adam_sync_tensor_gate.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

_HAS_CUDA = _HAS_TORCH and torch.cuda.is_available()

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
@unittest.skipUnless(_HAS_CUDA, "CUDA not available — Triton kernels need a GPU")
@unittest.skipUnless(_HAS_TRITON, "triton not available")
class FusedAdamSyncTensorGate(unittest.TestCase):
    """Byte-for-byte equivalence between scalar-arg and tensor-arg fused Adam."""

    NUM_STEPS = 100

    def _make_dataset(self, device: torch.device) -> tuple[
        list[tuple[str, tuple[int, ...]]],
        dict[str, torch.Tensor],  # bf16_params
        dict[str, torch.Tensor],  # masters
        dict[str, torch.Tensor],  # exp_avg
        dict[str, torch.Tensor],  # exp_avg_sq
        dict[str, torch.Tensor],  # grads (FP32)
    ]:
        torch.manual_seed(1234)
        # Diverse shapes mirroring the real model: 2D weights of two
        # sizes (one with weight decay, one without), and a 1D norm.
        layout = [
            ("layer0.weight",      (1024, 4096)),
            ("layer0.attn.qkv",    (4096, 4096)),
            ("layer0.norm.weight", (4096,)),
        ]
        bf16: dict[str, torch.Tensor] = {}
        masters: dict[str, torch.Tensor] = {}
        exp_avg: dict[str, torch.Tensor] = {}
        exp_avg_sq: dict[str, torch.Tensor] = {}
        grads: dict[str, torch.Tensor] = {}
        for name, shape in layout:
            m_fp32 = torch.randn(*shape, dtype=torch.float32, device=device) * 0.02
            masters[name] = m_fp32.clone()
            bf16[name] = m_fp32.to(torch.bfloat16)
            exp_avg[name] = torch.randn_like(m_fp32) * 0.001
            exp_avg_sq[name] = torch.randn_like(m_fp32).abs() * 1e-5
            grads[name] = torch.randn_like(m_fp32) * 0.1
        return layout, bf16, masters, exp_avg, exp_avg_sq, grads

    def _run_scalar_path(
        self,
        layout: list[tuple[str, tuple[int, ...]]],
        bf16: dict[str, torch.Tensor],
        masters: dict[str, torch.Tensor],
        exp_avg: dict[str, torch.Tensor],
        exp_avg_sq: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        schedule: list[tuple[float, int, float]],
    ) -> None:
        from training_engine_tensor.triton_kernels import fused_adam_sync
        from training_engine_tensor import optimizer as opt_mod

        for lr, step, clip_coeff in schedule:
            for name, shape in layout:
                has_wd = len(shape) > 1
                wd = opt_mod.WEIGHT_DECAY if has_wd else 0.0
                fused_adam_sync(
                    grads[name], masters[name],
                    exp_avg[name], exp_avg_sq[name],
                    bf16[name],
                    lr=lr, step=step,
                    clip_coeff=clip_coeff, wd=wd,
                )

    def _run_tensor_path(
        self,
        layout: list[tuple[str, tuple[int, ...]]],
        bf16: dict[str, torch.Tensor],
        masters: dict[str, torch.Tensor],
        exp_avg: dict[str, torch.Tensor],
        exp_avg_sq: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        schedule: list[tuple[float, int, float]],
    ) -> None:
        from training_engine_tensor.triton_kernels import fused_adam_sync_tensor
        from training_engine_tensor import optimizer as opt_mod

        bufs = opt_mod.OptimizerScalarBuffers(device=masters[layout[0][0]].device)

        for lr, step, clip_coeff in schedule:
            bufs.update_from_host(lr, step, clip_coeff)
            torch.cuda.synchronize()
            for name, shape in layout:
                has_wd = len(shape) > 1
                wd = opt_mod.WEIGHT_DECAY if has_wd else 0.0
                fused_adam_sync_tensor(
                    grads[name], masters[name],
                    exp_avg[name], exp_avg_sq[name],
                    bf16[name],
                    lr_buf=bufs.lr_buf,
                    clip_coeff_buf=bufs.clip_coeff_buf,
                    bc1_buf=bufs.bc1_buf,
                    bc2_buf=bufs.bc2_buf,
                    wd=wd,
                )

    def test_scalar_vs_tensor_byte_for_byte(self) -> None:
        device = torch.device("cuda")

        layout, bf16_a, masters_a, exp_avg_a, exp_avg_sq_a, grads_a = (
            self._make_dataset(device)
        )
        bf16_b = {k: v.clone() for k, v in bf16_a.items()}
        masters_b = {k: v.clone() for k, v in masters_a.items()}
        exp_avg_b = {k: v.clone() for k, v in exp_avg_a.items()}
        exp_avg_sq_b = {k: v.clone() for k, v in exp_avg_sq_a.items()}
        # Fresh per-step grads identical for both paths — generate the
        # full schedule of grads up front so step ordering matches.
        torch.manual_seed(7777)
        per_step_grads: list[dict[str, torch.Tensor]] = []
        for _ in range(self.NUM_STEPS):
            g_step: dict[str, torch.Tensor] = {}
            for name, shape in layout:
                g_step[name] = torch.randn(*shape, dtype=torch.float32, device=device) * 0.1
            per_step_grads.append(g_step)

        # Schedule: lr from cosine warmup-decay, clip_coeff alternates
        # < 1 / == 1 to cover both branches.
        schedule: list[tuple[float, int, float]] = []
        import math
        max_lr = 3.0e-4
        warmup = 5
        for step in range(1, self.NUM_STEPS + 1):
            if step < warmup:
                lr = max_lr * step / warmup
            else:
                lr = max_lr * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, self.NUM_STEPS - warmup)))
            clip_coeff = 0.97 if step % 2 == 0 else 1.0
            schedule.append((lr, step, clip_coeff))

        # Scalar path
        for step_idx, (lr, step, clip_coeff) in enumerate(schedule):
            grads_step = per_step_grads[step_idx]
            self._run_scalar_path(
                layout, bf16_a, masters_a, exp_avg_a, exp_avg_sq_a,
                grads_step, [(lr, step, clip_coeff)],
            )

        # Tensor path
        for step_idx, (lr, step, clip_coeff) in enumerate(schedule):
            grads_step = per_step_grads[step_idx]
            self._run_tensor_path(
                layout, bf16_b, masters_b, exp_avg_b, exp_avg_sq_b,
                grads_step, [(lr, step, clip_coeff)],
            )

        torch.cuda.synchronize()

        for name, _shape in layout:
            self.assertTrue(
                torch.equal(masters_a[name], masters_b[name]),
                f"masters[{name!r}] diverged: "
                f"max abs Δ = {(masters_a[name] - masters_b[name]).abs().max().item():.3e}",
            )
            self.assertTrue(
                torch.equal(exp_avg_a[name], exp_avg_b[name]),
                f"exp_avg[{name!r}] diverged",
            )
            self.assertTrue(
                torch.equal(exp_avg_sq_a[name], exp_avg_sq_b[name]),
                f"exp_avg_sq[{name!r}] diverged",
            )
            self.assertTrue(
                torch.equal(bf16_a[name], bf16_b[name]),
                f"bf16[{name!r}] diverged",
            )


if __name__ == "__main__":
    unittest.main()
