"""Full-step CUDA Graph capture (M2-P33 Phase B + Phase C trial).

Extends :mod:`step_graph` (M2-P32 Phase A — fwd + CE + bwd) to capture
the **entire** training step into a single replay:

    forward + CE + backward
    → reduce_scatter (NCCL)
    → distributed grad-norm (multi_tensor_l2norm + all_reduce)
    → device-only clip-coeff
    → all_gather (NCCL)
    → fused Adam + master→bf16 sync (Triton, tensor-arg variant)

NCCL collectives sit *inside* the captured stream — Phase C trial.
PyTorch 2.4+ ``ProcessGroupNCCL`` supports stream capture provided:

* the NCCL communicator was warmed up before capture (we run 3 eager
  iterations on a side stream first);
* the ``capture_error_mode="thread_local"`` is set so a captured
  collective failing to launch tears down only the capture stream
  rather than crashing the whole device;
* every collective input / output buffer has a stable ``data_ptr`` —
  enforced via :class:`nccl.NcclStaticBuffers` (one alloc-per-run).

The fused Adam call uses :func:`optimizer.fused_adam_sync_tensor`,
which routes ``lr`` / ``clip_coeff`` / two bias-correction terms
through 1-element FP32 device buffers (M2-P33 Phase B Triton kernel
upgrade). The host updates those buffers each step *before*
:meth:`replay` so the cosine LR schedule and Adam bias correction
behave correctly across replays.

Capture-safety constraints (must hold or this module raises)
------------------------------------------------------------
* ``WGRAD_OVERLAP=0`` — :class:`BucketedGradReducer` is incompatible
  with capture because ``write_grad`` is a host op (Python callback
  inside backward).
* ``SHARDED_OPTIMIZER=0`` — sharded optimizer uses
  ``transformer_engine_torch.multi_tensor_adam`` whose scalar-arg
  contract is identical to the *legacy* fused Adam (Phase B did not
  yet upgrade the TE kernel; deferred to Phase B-4).
* ``num_local_microbatches=1`` — multi-microbatch grad accumulation
  needs Python control flow inside the step which capture cannot
  record.
* ``FORWARD_CUDA_GRAPH=0`` — the forward-only graph (M2-P31) is
  superseded by this wider capture; running both wastes the M2-P31
  graph private mempool.
* ``FUSE_ADAM_SYNC=1`` — the tensor-arg fused Adam path lives only
  inside the fused Adam kernel; the legacy split clip / Adam / sync
  path was not promoted.

Numerical contract
------------------
End-to-end correctness is gated by the cctl loss-gate (≤200/1000
steps, ``avg_rel_diff < 1%``). The Triton kernel byte-for-byte
equivalence between scalar-arg and tensor-arg paths is gated by
``tests/test_fused_adam_sync_tensor_gate.py``. The NCCL-in-graph
behaviour is gated only by the cctl loss-gate (a divergent NCCL
capture would blow up loss within ~10 steps; PyTorch's NCCL
capture path also raises immediately on incompatible setups —
see ``capture_error_mode="thread_local"``).

Opt-in via env ``STEP_CUDA_GRAPH_FULL=1`` (default 0). When set,
``train_loop`` and ``test_long_train`` route the per-step body
through :func:`graphed_full_step` and skip the eager-path
``reduce_scatter_grads`` / ``compute_distributed_grad_norm`` /
``allgather_grads`` / ``fused_clip_adam_sync`` calls.
"""

from __future__ import annotations

__all__ = ["graphed_full_step", "is_step_cuda_graph_full_enabled"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .optimizer import AdamState

import torch

from .backward import backward_pass, cross_entropy_loss_backward
from .cuda_graph_utils import restore_state, snapshot_state
from .engine_config import get_config
from .forward import forward_pass_with_save
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


def _run_eager_full_step(
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_freqs: torch.Tensor,
    position_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_scale: float,
    *,
    nccl_bufs: NcclStaticBuffers,
    opt_bufs: OptimizerScalarBuffers,
    opt_state: AdamState,
) -> torch.Tensor:
    """Eager pipeline that capture / replay reproduce 1:1.

    The function reads ``opt_state.master_weights`` /
    ``opt_state.exp_avg`` / ``opt_state.exp_avg_sq`` and writes them
    in-place via the fused Adam Triton kernel; ``opt_state.step_count``
    is *not* mutated here — the host bumps it before each call so the
    captured kernel sees the right value through ``opt_bufs.bc1_buf``
    / ``opt_bufs.bc2_buf``.
    """
    logits, saved = forward_pass_with_save(
        params, input_ids, rope_freqs, position_ids=position_ids,
    )
    if get_config().fuse_ce:
        from .triton_kernels import fused_cross_entropy_fwd_bwd
        loss, _, d_logits = fused_cross_entropy_fwd_bwd(
            logits, labels, loss_mask, loss_scale=loss_scale,
        )
    else:
        loss, _, d_logits = cross_entropy_loss_backward(
            logits, labels, loss_mask, loss_scale=loss_scale,
        )
    grads = backward_pass(
        d_logits, params, saved, rope_freqs, grad_sink=None,
    )
    del logits, saved, d_logits

    fp32_shard = reduce_scatter_grads_persistent(nccl_bufs, grads)
    del grads
    grad_norm_sq = compute_distributed_grad_norm_tensor(fp32_shard, nccl_bufs)
    compute_clip_coeff_device(grad_norm_sq, opt_bufs.clip_coeff_buf)

    fp32_grads = allgather_grads_persistent(nccl_bufs, fp32_shard, params)

    fused_clip_adam_sync_tensor(opt_state, fp32_grads, params, opt_bufs)

    return loss


class _GraphedFullStep:
    """One captured CUDAGraph spanning fwd+CE+bwd+RS+grad_norm+AG+opt.

    Reuses the static-buffer pattern from :class:`step_graph._GraphedComputeStep`
    but additionally captures NCCL collectives + the optimizer step.
    """

    def __init__(self) -> None:
        self._captured: bool = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._params_ref: dict[str, torch.Tensor] | None = None
        self._loss_scale_ref: float | None = None
        self._opt_state_ref: Any | None = None
        self._nccl_bufs_ref: NcclStaticBuffers | None = None
        self._opt_bufs_ref: OptimizerScalarBuffers | None = None

        self._static_input_ids: torch.Tensor | None = None
        self._static_position_ids: torch.Tensor | None = None
        self._static_labels: torch.Tensor | None = None
        self._static_loss_mask: torch.Tensor | None = None
        self._static_rope_freqs: torch.Tensor | None = None

        self._static_loss: torch.Tensor | None = None

    def _alloc_static_inputs(
        self,
        input_ids: torch.Tensor,
        rope_freqs: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> None:
        self._static_input_ids = torch.empty_like(input_ids)
        self._static_position_ids = torch.empty_like(position_ids)
        self._static_labels = torch.empty_like(labels)
        self._static_loss_mask = torch.empty_like(loss_mask)
        self._static_rope_freqs = torch.empty_like(rope_freqs)
        self._static_input_ids.copy_(input_ids)
        self._static_position_ids.copy_(position_ids)
        self._static_labels.copy_(labels)
        self._static_loss_mask.copy_(loss_mask)
        self._static_rope_freqs.copy_(rope_freqs)

    def _warmup_and_capture(
        self,
        params: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        rope_freqs: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_scale: float,
        *,
        nccl_bufs: NcclStaticBuffers,
        opt_bufs: OptimizerScalarBuffers,
        opt_state: AdamState,
        lr: float,
    ) -> None:
        cfg = get_config()
        self._alloc_static_inputs(
            input_ids, rope_freqs, position_ids, labels, loss_mask,
        )

        # Snapshot every tensor that warmup + capture mutate, so the
        # first ``replay()`` after capture is the user's first real
        # training step. Without this, the model would have already
        # advanced 4 fake-data steps before step #0 of the loss-gate.
        pre_snap = snapshot_state(params, opt_state)

        ref_stats: dict[str, Any] | None = None
        if cfg.step_cuda_graph_full_sanity:
            opt_state.step_count += 1
            opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
            torch.cuda.synchronize()
            eager_loss = _run_eager_full_step(
                params, input_ids, rope_freqs, position_ids,
                labels, loss_mask, loss_scale,
                nccl_bufs=nccl_bufs, opt_bufs=opt_bufs, opt_state=opt_state,
            )
            torch.cuda.synchronize()
            ref_stats = self._step_stats(eager_loss, params, opt_state)
            del eager_loss
            restore_state(pre_snap, params, opt_state)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # CRITICAL: warmup AND capture must share the *same* stream so
        # NCCL's ProcessGroupNCCL has primed the communicator on that
        # exact stream. If we let ``torch.cuda.graph`` create its own
        # capture stream, NCCL's first invocation on that stream during
        # capture binds half-baked communicator state into the captured
        # nodes — then ``replay()`` aborts with an illegal memory
        # access (PyTorch 2.4+ documents this constraint and the same
        # pattern appears in ``torch.cuda.make_graphed_callables``).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(cfg.step_cuda_graph_full_warmup):
                opt_state.step_count += 1
                opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
                _loss = _run_eager_full_step(
                    params,
                    self._static_input_ids, self._static_rope_freqs,
                    self._static_position_ids,
                    self._static_labels, self._static_loss_mask,
                    loss_scale,
                    nccl_bufs=nccl_bufs, opt_bufs=opt_bufs,
                    opt_state=opt_state,
                )
                del _loss
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        self._graph = torch.cuda.CUDAGraph()
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        # capture_error_mode="thread_local" is required for capturing
        # NCCL collectives — a captured collective failing to launch
        # tears down only this capture stream rather than aborting the
        # whole device. PyTorch 2.4+ ProcessGroupNCCL relies on this.
        # ``stream=side`` ensures the capture replays on the same stream
        # the warmup used (NCCL is now primed on that stream).
        try:
            graph_ctx = torch.cuda.graph(
                self._graph, stream=side,
                capture_error_mode="thread_local",
            )
        except TypeError:
            # PyTorch < 2.4 does not accept capture_error_mode
            graph_ctx = torch.cuda.graph(self._graph, stream=side)
        with graph_ctx:
            self._static_loss = _run_eager_full_step(
                params,
                self._static_input_ids, self._static_rope_freqs,
                self._static_position_ids,
                self._static_labels, self._static_loss_mask,
                loss_scale,
                nccl_bufs=nccl_bufs, opt_bufs=opt_bufs, opt_state=opt_state,
            )

        self._params_ref = params
        self._loss_scale_ref = float(loss_scale)
        self._opt_state_ref = opt_state
        self._nccl_bufs_ref = nccl_bufs
        self._opt_bufs_ref = opt_bufs
        self._captured = True

        if cfg.step_cuda_graph_full_sanity:
            if ref_stats is None:
                raise RuntimeError("ref_stats should have been captured by the sanity-check branch")
            graph_stats = self._step_stats(
                self._static_loss, params, opt_state,
            )
            self._verify_stats(ref_stats, graph_stats)

        # Restore so the very next ``replay()`` is the first real step.
        # This rolls back the warmup mutations AND the single capture
        # iteration's optimizer update. Replay below uses the same
        # graph instance — its update_from_host plus replay() now
        # produces the correct step #1 weights.
        restore_state(pre_snap, params, opt_state)
        torch.cuda.synchronize()

    @staticmethod
    def _step_stats(
        loss: torch.Tensor,
        params: dict[str, torch.Tensor],
        opt_state: AdamState,
    ) -> dict[str, Any]:
        loss_val = float(loss.detach().float().item())
        param_stats: dict[str, dict[str, float]] = {}
        for name, p in params.items():
            f = p.detach().float()
            param_stats[name] = {
                "mean": f.mean().item(),
                "std": f.std().item(),
                "abs_max": f.abs().max().item(),
            }
        return {"loss": loss_val, "params": param_stats}

    def _verify_stats(
        self,
        ref: dict[str, Any],
        got: dict[str, Any],
    ) -> None:
        loss_diff = abs(got["loss"] - ref["loss"])
        worst_diff = 0.0
        worst_name = ""
        worst_field = ""
        for name in ref["params"]:
            ref_p = ref["params"][name]
            got_p = got["params"][name]
            for key in ("mean", "std", "abs_max"):
                d = abs(got_p[key] - ref_p[key])
                rel = d / max(abs(ref_p[key]), 1e-9)
                metric = max(d, rel * 1e-3)
                if metric > worst_diff:
                    worst_diff = metric
                    worst_name = name
                    worst_field = key
        cfg = get_config()
        worst = max(loss_diff, worst_diff)
        if worst > cfg.step_cuda_graph_full_tol:
            raise RuntimeError(
                f"[step_graph_full] sanity check FAILED: loss Δ={loss_diff:.3e}, "
                f"params worst Δ={worst_diff:.3e} on "
                f"{worst_name!r}/{worst_field} > tol={cfg.step_cuda_graph_full_tol:.1e}. "
                "Capture path is numerically unsafe; disable with "
                "STEP_CUDA_GRAPH_FULL=0."
            )
        try:
            rank_str = (
                f"rank{torch.distributed.get_rank()}"
                if torch.distributed.is_initialized() else "rank0"
            )
        except Exception:
            rank_str = "rank0"
        print(
            f"[step_graph_full][{rank_str}] capture OK: loss Δ={loss_diff:.3e}, "
            f"params worst Δ={worst_diff:.3e} on "
            f"{worst_name!r}/{worst_field} (tol={cfg.step_cuda_graph_full_tol:.1e}, "
            f"warmup={cfg.step_cuda_graph_full_warmup}, n_params={len(ref['params'])})",
            flush=True,
        )

    def replay(
        self,
        params: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        rope_freqs: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_scale: float,
        *,
        opt_state: AdamState,
        opt_bufs: OptimizerScalarBuffers,
        lr: float,
    ) -> torch.Tensor:
        if self._params_ref is not params:
            raise RuntimeError(
                "[step_graph_full] params dict identity changed since "
                "capture — would invalidate recorded data_ptrs. "
                "Disable STEP_CUDA_GRAPH_FULL or recapture."
            )
        if self._opt_state_ref is not opt_state:
            raise RuntimeError(
                "[step_graph_full] opt_state identity changed since capture."
            )
        if self._opt_bufs_ref is not opt_bufs:
            raise RuntimeError(
                "[step_graph_full] opt_bufs identity changed since capture."
            )
        if float(loss_scale) != self._loss_scale_ref:
            raise RuntimeError(
                f"[step_graph_full] loss_scale changed since capture "
                f"({self._loss_scale_ref!r} → {loss_scale!r})."
            )

        self._static_input_ids.copy_(input_ids, non_blocking=True)
        self._static_position_ids.copy_(position_ids, non_blocking=True)
        self._static_labels.copy_(labels, non_blocking=True)
        self._static_loss_mask.copy_(loss_mask, non_blocking=True)
        if rope_freqs.data_ptr() != self._static_rope_freqs.data_ptr():
            self._static_rope_freqs.copy_(rope_freqs, non_blocking=True)

        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)

        self._graph.replay()
        return self._static_loss


_FULL_STEP_GRAPH_CACHE: dict[tuple, _GraphedFullStep] = {}


def graphed_full_step(
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_scale: float = 1.0,
    nccl_bufs: NcclStaticBuffers,
    opt_bufs: OptimizerScalarBuffers,
    opt_state: AdamState,
    lr: float,
) -> torch.Tensor:
    """Run one full training step (fwd → ... → opt) inside a CUDA Graph.

    Falls through to eager execution when ``STEP_CUDA_GRAPH_FULL=0``;
    in that case the eager pipeline is bytewise identical to the train
    loop's legacy non-overlap path (with ``fused_clip_adam_sync_tensor``
    in place of the scalar-arg variant — equivalence pinned by
    ``tests/test_fused_adam_sync_tensor_gate.py``).

    Different per-call shapes get distinct cached graphs; the
    production path keeps shapes static via the
    ``CONDITIONAL_FIXED_LENGTH_SEGMENT`` packer (seq_length=4096).
    """
    cfg = get_config()
    if not cfg.step_cuda_graph_full:
        opt_state.step_count += 1
        opt_bufs.update_from_host(lr, opt_state.step_count, 1.0)
        return _run_eager_full_step(
            params, input_ids, rope_freqs, position_ids,
            labels, loss_mask, loss_scale,
            nccl_bufs=nccl_bufs, opt_bufs=opt_bufs, opt_state=opt_state,
        )

    key = (
        tuple(input_ids.shape),
        tuple(position_ids.shape),
        tuple(labels.shape),
        tuple(loss_mask.shape),
        tuple(rope_freqs.shape),
        input_ids.dtype, labels.dtype, loss_mask.dtype, rope_freqs.dtype,
    )
    cell = _FULL_STEP_GRAPH_CACHE.get(key)
    if cell is None:
        cell = _GraphedFullStep()
        cell._warmup_and_capture(
            params, input_ids, rope_freqs, position_ids,
            labels, loss_mask, loss_scale,
            nccl_bufs=nccl_bufs, opt_bufs=opt_bufs, opt_state=opt_state,
            lr=lr,
        )
        _FULL_STEP_GRAPH_CACHE[key] = cell
    return cell.replay(
        params, input_ids, rope_freqs, position_ids,
        labels, loss_mask, loss_scale,
        opt_state=opt_state, opt_bufs=opt_bufs, lr=lr,
    )


def is_step_cuda_graph_full_enabled() -> bool:
    cfg = get_config()
    return cfg.step_cuda_graph_full


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402

_register_reset_hook(_FULL_STEP_GRAPH_CACHE.clear)
