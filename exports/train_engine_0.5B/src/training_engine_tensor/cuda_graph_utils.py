"""Shared utilities for CUDA Graph capture modules.

Provides snapshot/restore helpers used by step_graph_full, step_graph_nccl_opt,
and step_graph_optimizer to save and restore optimizer + param state across
warmup/capture passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .optimizer import AdamState

import torch

__all__ = ["restore_state", "snapshot_state"]


def snapshot_state(
    params: dict[str, torch.Tensor],
    opt_state: AdamState,
) -> dict[str, Any]:
    """Snapshot every tensor that warmup/capture would mutate.

    Captures params (BF16 weights), master_weights (FP32),
    exp_avg, exp_avg_sq, step_count, and num_samples so they
    can be rolled back after warmup + capture.
    """
    return {
        "master": {n: t.clone() for n, t in opt_state.master_weights.items()},
        "exp_avg": {n: t.clone() for n, t in opt_state.exp_avg.items()},
        "exp_avg_sq": {n: t.clone() for n, t in opt_state.exp_avg_sq.items()},
        "params": {n: t.clone() for n, t in params.items()},
        "step_count": int(opt_state.step_count),
        "num_samples": int(getattr(opt_state, "num_samples", 0)),
    }


def restore_state(
    snap: dict[str, Any],
    params: dict[str, torch.Tensor],
    opt_state: AdamState,
) -> None:
    """Restore optimizer + param state from a snapshot."""
    for n, t in snap["master"].items():
        opt_state.master_weights[n].copy_(t)
    for n, t in snap["exp_avg"].items():
        opt_state.exp_avg[n].copy_(t)
    for n, t in snap["exp_avg_sq"].items():
        opt_state.exp_avg_sq[n].copy_(t)
    for n, t in snap["params"].items():
        params[n].copy_(t)
    opt_state.step_count = snap["step_count"]
    if hasattr(opt_state, "num_samples"):
        opt_state.num_samples = snap["num_samples"]
