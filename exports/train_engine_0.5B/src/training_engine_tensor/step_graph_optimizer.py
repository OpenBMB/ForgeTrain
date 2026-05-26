"""Optimizer-only CUDA Graph (M2-P33 Phase B).

Captures the per-step optimizer body — clip-coeff + fused Adam + bf16
master sync — into a CUDA Graph that the host replays after the eager
NCCL collectives complete. Pairs with :mod:`step_graph` (M2-P32
Phase A — fwd + CE + bwd) so the train loop runs:

    Graph A (compute)              Phase A capture
        forward_pass_with_save
        cross_entropy_loss_backward
        backward_pass(grad_sink=None)

    Eager seam (NCCL only)         capture-incompatible
        reduce_scatter_grads_persistent(...)
        local_sq → grad_norm_sq_buf
        all_reduce(grad_norm_sq_buf)
        all_gather → fp32_avg

    Graph B (optimizer)            this module
        compute_clip_coeff_device(grad_norm_sq_buf → clip_coeff_buf)
        fused_clip_adam_sync_tensor(state, fp32_avg-views, params)

Why a *separate* graph?
-----------------------
Phase C — capturing NCCL collectives in the same graph — turned out to
be unstable in practice (PyTorch 2.4 + ProcessGroupNCCL still reproducibly
crashes with ``CUDA error: an illegal memory access`` on the first
``replay()`` even with ``capture_error_mode="thread_local"`` and
``TORCH_NCCL_USE_COMM_NONBLOCKING=1``). The two-graph split keeps every
launch overhead the captured Adam kernels were designed to hide while
leaving the NCCL collectives on the well-trodden eager path.

Capture-safety contract
-----------------------
* ``params``, ``state.master_weights/exp_avg/exp_avg_sq`` and the
  ``nccl_bufs.fp32_avg`` slot all have stable ``data_ptr`` s for the
  whole training run (the optimiser writes back via ``copy_`` /
  ``tl.store``); each is recorded into the graph exactly once at
  capture time.
* ``opt_bufs`` holds 1-element FP32 buffers for ``lr`` /
  ``clip_coeff`` / ``bias_correction{1,2}``. The host refreshes the
  three host-known scalars (``lr``, ``bc1``, ``bc2``) BEFORE every
  ``replay()`` via :meth:`OptimizerScalarBuffers.update_from_host`;
  ``clip_coeff_buf`` is computed *inside* the captured graph from
  ``nccl_bufs.grad_norm_sq_buf`` (which the eager NCCL path
  populates).
* ``nccl_bufs.grad_norm_sq_buf`` is a 1-element FP32 tensor whose
  ``data_ptr`` is fixed at construction; the eager path's
  ``compute_distributed_grad_norm_tensor`` writes into it directly so
  the captured ``compute_clip_coeff_device`` reads the freshest value
  on each replay.

Numerical contract
------------------
End-to-end correctness gate: the cctl loss-gate (≤200/1000 steps,
``avg_rel_diff < 1%``). The Triton kernel byte-for-byte equivalence
between scalar-arg and tensor-arg fused Adam paths is gated by
``tests/test_fused_adam_sync_tensor_gate.py``; this module reuses that
kernel verbatim, so capture only adds graph-level launch elision.

Opt-in via env ``STEP_CUDA_GRAPH_OPTIMIZER=1`` (default 0). When set,
``test_long_train`` routes the optimizer step through this module and
expects ``STEP_CUDA_GRAPH=1`` to also be active for Phase A. Activation
also requires ``WGRAD_OVERLAP=0``, ``SHARDED_OPTIMIZER=0``,
``num_local_microbatches=1`` and ``FUSE_ADAM_SYNC=1`` for the same
reasons :mod:`step_graph` documents (the bucketed reducer's host-side
callbacks are not capture-safe).
"""

from __future__ import annotations

__all__ = ["graphed_optimizer_step", "is_step_cuda_graph_optimizer_enabled"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .optimizer import AdamState

import torch

from .cuda_graph_utils import restore_state, snapshot_state
from .engine_config import get_config
from .nccl import NcclStaticBuffers
from .optimizer import (
    OptimizerScalarBuffers,
    compute_clip_coeff_device,
    fused_clip_adam_sync_tensor,
)


def _run_eager_optimizer_step(
    params: dict[str, torch.Tensor],
    fp32_grads: dict[str, torch.Tensor],
    *,
    opt_state: AdamState,
    opt_bufs: OptimizerScalarBuffers,
    grad_norm_sq_buf: torch.Tensor,
) -> None:
    """Eager optimizer body that capture / replay reproduce 1:1.

    The clip-coefficient is computed *inside* this body so the captured
    graph picks up the freshest ``grad_norm_sq_buf`` value — populated
    by the upstream eager NCCL ``all_reduce`` — on every replay. The
    body never reads or writes Python-side state on ``opt_state``;
    ``step_count`` is bumped by the caller so the host can refresh
    ``opt_bufs.bc{1,2}_buf`` before each replay.
    """
    compute_clip_coeff_device(grad_norm_sq_buf, opt_bufs.clip_coeff_buf)
    fused_clip_adam_sync_tensor(opt_state, fp32_grads, params, opt_bufs)


class _GraphedOptimizerStep:
    """One captured CUDAGraph spanning compute_clip_coeff + fused Adam."""

    def __init__(self) -> None:
        self._captured: bool = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._params_ref: dict[str, torch.Tensor] | None = None
        self._fp32_grads_ref: dict[str, torch.Tensor] | None = None
        self._opt_state_ref: Any | None = None
        self._opt_bufs_ref: OptimizerScalarBuffers | None = None
        self._nccl_bufs_ref: NcclStaticBuffers | None = None

    def _warmup_and_capture(
        self,
        params: dict[str, torch.Tensor],
        fp32_grads: dict[str, torch.Tensor],
        *,
        opt_state: AdamState,
        opt_bufs: OptimizerScalarBuffers,
        nccl_bufs: NcclStaticBuffers,
        lr: float,
    ) -> None:
        cfg = get_config()
        # Roll back warmup + capture mutations of master / exp_avg /
        # exp_avg_sq / params so the first ``replay()`` is the user's
        # first real optimiser step (matches the behaviour of
        # ``step_graph_full._GraphedFullStep`` minus NCCL).
        pre_snap = snapshot_state(params, opt_state)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(cfg.step_cuda_graph_optimizer_warmup):
                opt_state.step_count += 1
                opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
                _run_eager_optimizer_step(
                    params, fp32_grads,
                    opt_state=opt_state, opt_bufs=opt_bufs,
                    grad_norm_sq_buf=nccl_bufs.grad_norm_sq_buf,
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        self._graph = torch.cuda.CUDAGraph()
        # Update host scalars before capture so the kernel's recorded
        # tensor reads see a sane initial value at capture time. Host
        # then refreshes again before every replay.
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        with torch.cuda.graph(self._graph):
            _run_eager_optimizer_step(
                params, fp32_grads,
                opt_state=opt_state, opt_bufs=opt_bufs,
                grad_norm_sq_buf=nccl_bufs.grad_norm_sq_buf,
            )

        self._params_ref = params
        self._fp32_grads_ref = fp32_grads
        self._opt_state_ref = opt_state
        self._opt_bufs_ref = opt_bufs
        self._nccl_bufs_ref = nccl_bufs
        self._captured = True

        # Restore so the very next ``replay()`` is the first real step.
        restore_state(pre_snap, params, opt_state)
        torch.cuda.synchronize()

    def replay(
        self,
        params: dict[str, torch.Tensor],
        fp32_grads: dict[str, torch.Tensor],
        *,
        opt_state: AdamState,
        opt_bufs: OptimizerScalarBuffers,
        nccl_bufs: NcclStaticBuffers,
        lr: float,
    ) -> None:
        if self._params_ref is not params:
            raise RuntimeError(
                "[step_graph_optimizer] params dict identity changed since "
                "capture — would invalidate recorded data_ptrs. "
                "Disable STEP_CUDA_GRAPH_OPTIMIZER or recapture."
            )
        if self._fp32_grads_ref is not fp32_grads:
            # NcclStaticBuffers caches the per-param view dict so
            # ``allgather_grads_persistent`` returns the same dict each
            # call. Identity drift here means the caller forgot to thread
            # ``nccl_bufs`` through, which would invalidate every recorded
            # ``data_ptr`` in the captured graph.
            raise RuntimeError(
                "[step_graph_optimizer] fp32_grads dict identity changed "
                "since capture — pass the same nccl_bufs "
                "(and therefore the same cached grad-views dict) on every "
                "call. Disable STEP_CUDA_GRAPH_OPTIMIZER or recapture."
            )
        if self._opt_state_ref is not opt_state:
            raise RuntimeError(
                "[step_graph_optimizer] opt_state identity changed since "
                "capture."
            )
        if self._opt_bufs_ref is not opt_bufs:
            raise RuntimeError(
                "[step_graph_optimizer] opt_bufs identity changed since "
                "capture."
            )
        if self._nccl_bufs_ref is not nccl_bufs:
            raise RuntimeError(
                "[step_graph_optimizer] nccl_bufs identity changed since "
                "capture."
            )
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        self._graph.replay()


_OPTIMIZER_GRAPH_CACHE: dict[tuple, _GraphedOptimizerStep] = {}


def graphed_optimizer_step(
    params: dict[str, torch.Tensor],
    fp32_grads: dict[str, torch.Tensor],
    *,
    opt_state: AdamState,
    opt_bufs: OptimizerScalarBuffers,
    nccl_bufs: NcclStaticBuffers,
    lr: float,
) -> None:
    """Run compute_clip_coeff + fused Adam in one captured CUDA Graph.

    Eager passthrough when ``STEP_CUDA_GRAPH_OPTIMIZER=0``. Each unique
    ``params`` / ``fp32_grads`` identity gets its own cached graph, but
    the production path holds these dicts steady for the entire run, so
    cache size is exactly 1.
    """
    cfg = get_config()
    if not cfg.step_cuda_graph_optimizer:
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        _run_eager_optimizer_step(
            params, fp32_grads,
            opt_state=opt_state, opt_bufs=opt_bufs,
            grad_norm_sq_buf=nccl_bufs.grad_norm_sq_buf,
        )
        return

    key = (id(params), id(fp32_grads), id(opt_state), id(opt_bufs))
    cell = _OPTIMIZER_GRAPH_CACHE.get(key)
    if cell is None:
        cell = _GraphedOptimizerStep()
        cell._warmup_and_capture(
            params, fp32_grads,
            opt_state=opt_state, opt_bufs=opt_bufs,
            nccl_bufs=nccl_bufs, lr=lr,
        )
        _OPTIMIZER_GRAPH_CACHE[key] = cell
    cell.replay(
        params, fp32_grads,
        opt_state=opt_state, opt_bufs=opt_bufs,
        nccl_bufs=nccl_bufs, lr=lr,
    )


def is_step_cuda_graph_optimizer_enabled() -> bool:
    cfg = get_config()
    return cfg.step_cuda_graph_optimizer


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402

_register_reset_hook(_OPTIMIZER_GRAPH_CACHE.clear)
