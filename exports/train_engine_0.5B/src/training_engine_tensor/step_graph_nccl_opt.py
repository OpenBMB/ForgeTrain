"""NCCL + optimizer CUDA Graph (M2-P33 Phase C).

Captures the **entire** post-compute tail into a single CUDA Graph:

    Graph A (M2-P32)                  existing ``step_graph.py``
        forward + CE + backward → grads dict

    Graph C (this module)
        pack grads → fp32_flat
        reduce_scatter_tensor        (NCCL)
        compute sum-of-squares       (local sq-norm)
        all_reduce grad_norm_sq_buf  (NCCL)
        all_gather_into_tensor       (NCCL)
        compute_clip_coeff_device
        fused_clip_adam_sync_tensor   (Triton, per-param)

Strategy difference from Phase B+C full-step graph
---------------------------------------------------
The earlier ``step_graph_full.py`` tried to capture fwd + CE + bwd +
NCCL + opt *all in one graph*. That approach reproducibly crashed with
``CUDA error: an illegal memory access`` on the first ``replay()``
under PyTorch 2.4 / NCCL 2.20+, even with ``capture_error_mode=
"thread_local"`` and ``TORCH_NCCL_USE_COMM_NONBLOCKING=1``.

This module takes a different approach: the compute graph (Graph A)
and the NCCL + optimizer graph (Graph C) are **separate** captures.
Graph A replays first (on the default stream), then Graph C replays
(also on the default stream). CUDA stream ordering guarantees that
Graph C's kernels run after Graph A's, so the grad tensors are valid.
Isolating the NCCL capture from the autograd / TE / Triton-autotune
kernels avoids the suspected source of the earlier crash.

Capture-safety contract
-----------------------
* **Grad tensors from Graph A**: the ``grads`` dict returned by
  ``graphed_compute_step`` aliases Graph A's internal mempool. These
  tensors have stable ``data_ptr`` s across replays (guaranteed by
  ``_GraphedComputeStep``). The packing loop in
  ``reduce_scatter_grads_persistent`` copies from these pointers into
  ``nccl_bufs.fp32_flat``, so the same source ``data_ptr`` s are
  recorded at capture time and valid on every replay.

* **nccl_bufs**: ``fp32_flat``, ``shard_view``, ``fp32_avg``,
  ``grad_norm_sq_buf`` — all allocated once in the constructor with
  stable ``data_ptr`` s.

* **opt_bufs**: ``lr_buf``, ``clip_coeff_buf``, ``bc1_buf``,
  ``bc2_buf`` — 1-element FP32 device tensors, stable ``data_ptr``.
  Host refreshes them via ``update_from_host`` before each replay.

* **params, master_weights, exp_avg, exp_avg_sq**: all have stable
  ``data_ptr`` (the Triton kernel writes back via ``tl.store``).

NCCL capture requirements
-------------------------
* ``TORCH_NCCL_USE_COMM_NONBLOCKING=1`` should be set in the env.
* ``capture_error_mode="thread_local"`` is passed to
  ``torch.cuda.graph()``.
* Warmup runs on a side stream, capture on the default stream. NCCL
  ProcessGroup supports capture on either, but default stream is the
  simpler path.
* The warmup calls real NCCL collectives so the communicator is
  initialised before capture begins.

Opt-in via ``STEP_CUDA_GRAPH_NCCL_OPT=1`` (default 0). Mutually
exclusive with ``STEP_CUDA_GRAPH_OPTIMIZER=1`` (Phase B) — when both
are set, Phase C (this module) takes precedence.
"""

from __future__ import annotations

__all__ = ["graphed_nccl_opt_step", "is_nccl_opt_graph_enabled"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .optimizer import AdamState

import torch

from .cuda_graph_utils import restore_state, snapshot_state
from .engine_config import get_config
from .nccl import (
    NcclStaticBuffers,
    allgather_grads_persistent,
    compute_distributed_grad_norm_tensor,
    reduce_scatter_grads_persistent,
)
from .optimizer import (
    OptimizerScalarBuffers,
    compute_clip_coeff_device,
    fused_clip_adam_sync_tensor,
)


def _run_eager_nccl_opt(
    params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
    *,
    nccl_bufs: NcclStaticBuffers,
    opt_bufs: OptimizerScalarBuffers,
    opt_state: AdamState,
) -> None:
    """Eager NCCL + optimizer pipeline captured verbatim by Graph C."""
    shard = reduce_scatter_grads_persistent(nccl_bufs, grads)
    compute_distributed_grad_norm_tensor(shard, nccl_bufs)
    allgather_grads_persistent(nccl_bufs, shard, params)

    compute_clip_coeff_device(
        nccl_bufs.grad_norm_sq_buf, opt_bufs.clip_coeff_buf,
    )
    fp32_grads = nccl_bufs.cached_grad_views
    fused_clip_adam_sync_tensor(opt_state, fp32_grads, params, opt_bufs)


class _GraphedNcclOpt:
    """Captured CUDAGraph: NCCL reduce_scatter + grad_norm + all_gather +
    clip-coeff + fused Adam."""

    def __init__(self) -> None:
        self._captured: bool = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._params_ref: dict[str, torch.Tensor] | None = None
        self._grads_ref: dict[str, torch.Tensor] | None = None
        self._nccl_bufs_ref: NcclStaticBuffers | None = None
        self._opt_bufs_ref: OptimizerScalarBuffers | None = None
        self._opt_state_ref: Any | None = None

    def _warmup_and_capture(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        *,
        nccl_bufs: NcclStaticBuffers,
        opt_bufs: OptimizerScalarBuffers,
        opt_state: AdamState,
        lr: float,
    ) -> None:
        cfg = get_config()
        pre_snap = snapshot_state(params, opt_state)

        # Prime the cached grad-views dict before capture so the
        # identity is stable from the very first call.
        allgather_grads_persistent(nccl_bufs, nccl_bufs.shard_view, params)

        # Warmup on a side stream so NCCL communicator is initialised.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(cfg.step_cuda_graph_nccl_opt_warmup):
                opt_state.step_count += 1
                opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
                _run_eager_nccl_opt(
                    params, grads,
                    nccl_bufs=nccl_bufs, opt_bufs=opt_bufs,
                    opt_state=opt_state,
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Capture on the DEFAULT stream. The NCCL ProcessGroup tracks
        # communicator state per-stream; warmup on ``side`` initialised
        # the communicator there, and PyTorch ≥ 2.1 reuses the same
        # communicator on the default stream. Using the default stream
        # for capture matches the canonical PyTorch CUDA-graph + NCCL
        # pattern and avoids the stream-mismatch crash we saw when
        # capturing on ``side`` in step_graph_full.py.
        self._graph = torch.cuda.CUDAGraph()
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        try:
            graph_ctx = torch.cuda.graph(
                self._graph, capture_error_mode="thread_local",
            )
        except TypeError:
            graph_ctx = torch.cuda.graph(self._graph)
        with graph_ctx:
            _run_eager_nccl_opt(
                params, grads,
                nccl_bufs=nccl_bufs, opt_bufs=opt_bufs,
                opt_state=opt_state,
            )

        self._params_ref = params
        self._grads_ref = grads
        self._nccl_bufs_ref = nccl_bufs
        self._opt_bufs_ref = opt_bufs
        self._opt_state_ref = opt_state
        self._captured = True

        restore_state(pre_snap, params, opt_state)
        torch.cuda.synchronize()

    def replay(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        *,
        nccl_bufs: NcclStaticBuffers,
        opt_bufs: OptimizerScalarBuffers,
        opt_state: AdamState,
        lr: float,
    ) -> None:
        if self._params_ref is not params:
            raise RuntimeError(
                "[step_graph_nccl_opt] params identity changed since capture."
            )
        if self._grads_ref is not grads:
            raise RuntimeError(
                "[step_graph_nccl_opt] grads dict identity changed since "
                "capture — the grads must be the SAME dict from "
                "graphed_compute_step (Graph A's static_grads)."
            )
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        self._graph.replay()


_NCCL_OPT_GRAPH_CACHE: dict[int, _GraphedNcclOpt] = {}


def graphed_nccl_opt_step(
    params: dict[str, torch.Tensor],
    grads: dict[str, torch.Tensor],
    *,
    nccl_bufs: NcclStaticBuffers,
    opt_bufs: OptimizerScalarBuffers,
    opt_state: AdamState,
    lr: float,
) -> None:
    """Run NCCL + optimizer in one captured CUDA Graph (Phase C).

    Eager passthrough when ``STEP_CUDA_GRAPH_NCCL_OPT=0``.
    """
    cfg = get_config()
    if not cfg.step_cuda_graph_nccl_opt:
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        _run_eager_nccl_opt(
            params, grads,
            nccl_bufs=nccl_bufs, opt_bufs=opt_bufs,
            opt_state=opt_state,
        )
        return

    key = id(params)
    cell = _NCCL_OPT_GRAPH_CACHE.get(key)
    if cell is None:
        cell = _GraphedNcclOpt()
        cell._warmup_and_capture(
            params, grads,
            nccl_bufs=nccl_bufs, opt_bufs=opt_bufs,
            opt_state=opt_state, lr=lr,
        )
        _NCCL_OPT_GRAPH_CACHE[key] = cell
    cell.replay(
        params, grads,
        nccl_bufs=nccl_bufs, opt_bufs=opt_bufs,
        opt_state=opt_state, lr=lr,
    )


def is_nccl_opt_graph_enabled() -> bool:
    cfg = get_config()
    return cfg.step_cuda_graph_nccl_opt


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402

_register_reset_hook(_NCCL_OPT_GRAPH_CACHE.clear)
