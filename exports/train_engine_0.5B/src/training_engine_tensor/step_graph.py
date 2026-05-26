"""Full-step CUDA Graph capture: forward + CE + backward (M2-P32 Phase A).

Phase A of full-step CUDA Graph: collapses the entire ``fwd → CE → bwd``
chain (~78% of step time, dominated by short kernel launches) into a
single ``graph.replay()`` call. NCCL collectives (reduce_scatter /
all_reduce / allgather) and the optimizer step stay eager because:

* NCCL collectives are not capture-safe under the legacy PyTorch path
  used by this codebase (``dist.reduce_scatter_tensor`` etc. dispatch
  on a NCCL-internal stream which capture cannot record consistently);
* :func:`compute_distributed_grad_norm` calls ``.item()`` (host sync,
  see ``nccl.py:210``);
* :func:`fused_adam_sync` (Triton) takes ``lr`` / ``step`` /
  ``clip_coeff`` as Python scalar kernel args. Capture would burn the
  capture-time values into the graph's launch nodes, making ``lr``
  cosine schedule and per-step bias correction silently wrong on
  replay. Tensor-arg promotion is deferred to Phase B.

This wrapper extends :mod:`forward_graph` (M2-P31): when both
``FORWARD_CUDA_GRAPH=1`` and ``STEP_CUDA_GRAPH=1`` are set, the wider
step-graph supersedes the forward-only one, which becomes a no-op
because forward is already inside the step graph.

Opt-in via env ``STEP_CUDA_GRAPH=1`` (default 0). Requires
``WGRAD_OVERLAP=0`` because the bucketed reducer dispatches a Python
callback (:meth:`BucketedGradReducer.write_grad`) per gradient inside
backward — host execution is not capture-safe. When ``STEP_CUDA_GRAPH=1``
the train loop must therefore route ``grad_sink=None`` (which this
wrapper enforces by construction); the callsite is responsible for
selecting the legacy non-overlap reduce_scatter / grad_norm / allgather
path after replay.

Capture safety contract
-----------------------
* ``saved`` and ``d_logits`` are graph-internal temporaries; their
  storage is reused across replays, callers must NOT reference them
  after :meth:`replay` returns.
* ``loss`` and every tensor in ``grads`` reference graph-internal
  buffers. The immediate downstream consumers (``mb_losses.append``
  + ``reduce_scatter_grads(grads, device)``) read these tensors
  strictly before the next ``replay()``, which the train loop
  enforces by structure (one replay per microbatch, NCCL/optimizer
  serial after).
* ``params`` dict entries are bound by ``data_ptr`` at capture time;
  ``optimizer.fused_adam_sync`` writes back via ``copy_`` so storage
  is stable across steps. Verified at ``optimizer.py:L301``.
* ``rope_freqs`` is constant after :func:`precompute_rope_freqs`
  (called once at startup), so we skip the per-step ``copy_`` to save
  one D2D copy.
* ``loss_scale`` is a Python float that lands in the
  ``fused_cross_entropy`` Triton kernel scalar args. We snapshot it at
  capture time and require the same value on every replay;
  ``num_local_microbatches`` is fixed at init so this is a no-op for
  the production path (loss_scale=1.0).

Numerical sanity check
----------------------
Optional, OFF by default (``STEP_CUDA_GRAPH_SANITY=0``). The cctl
loss-gate is the canonical end-to-end correctness signal — a divergent
graph would blow up ``avg_rel_diff`` within the first 50 steps. When
opt-in, the wrapper records reduced-footprint reference statistics
(loss scalar + per-grad mean/std/abs_max) from a single eager step
BEFORE warmup (so the graph private mempool has not yet pinned the
saved-tensor + grads-dict footprint), then replays the captured graph
and compares; aborts if any scalar deviates by more than
``STEP_CUDA_GRAPH_TOL`` (default 1e-3 BF16). This avoids the
~3 GB grads-dict full-clone cost and the 5.7 GB d_logits clone that
would OOM the H100 SXM (80 GB) at typical per-step working sets.

Phase B handoff
---------------
The captured graph here covers fwd+CE+bwd; merging the optimizer step
into a second graph (with a NCCL-eager seam in between) is Phase B and
requires:
1. Promoting ``lr`` / ``step`` / ``clip_coeff`` from Python scalar to
   GPU tensor args in :func:`_fused_adam_sync_kernel`.
2. A persistent host-pinned ``lr_buf`` that the optimizer wrapper
   ``copy_`` s the per-step values into before each replay.
3. A separate Triton-kernel numerical gate to validate the tensor-arg
   variant matches the scalar-arg path bytewise.
"""

from __future__ import annotations

__all__ = ["graphed_compute_step", "is_step_cuda_graph_enabled"]

from collections.abc import Callable
from typing import Any

import torch

from .backward import backward_pass, cross_entropy_loss_backward
from .engine_config import get_config
from .forward import forward_pass_with_save

GradSink = Callable[[str, torch.Tensor], None]


def _run_eager_compute_step(
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_freqs: torch.Tensor,
    position_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_scale: float,
    *,
    grad_sink: GradSink | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Eager fwd → CE → bwd, identical to the train loop's structure.

    ``grad_sink``:
      * ``None`` (default) — backward returns a full ``grads`` dict
        of BF16 tensors referencing graph-internal buffers. This is
        the M2-P32 Phase A path that the train loop uses with
        ``grad_reducer=None``.
      * a capture-safe callable (typically
        ``BucketedGradReducer.make_device_grad_sink()``) — backward
        writes each gradient directly into the reducer's bucket
        buffer via a single in-place ``add_``. The returned dict is
        empty. This is the path-1 (Direction-C) coexistence between
        step CUDA Graph and the bucketed reducer (wgrad_overlap=1).
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
        d_logits, params, saved, rope_freqs, grad_sink=grad_sink,
    )
    del logits, saved, d_logits
    return loss, grads


class _GraphedComputeStep:
    """One captured CUDAGraph for fwd + CE + bwd at fixed shapes.

    Optionally captures the backward pass with a device-side
    ``grad_sink`` (path-1, Direction-C), in which case the static
    grads dict is empty — gradients flow into the reducer's bucket
    buffers in place.
    """

    def __init__(self, grad_sink: GradSink | None = None) -> None:
        self._captured: bool = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._params_ref: dict[str, torch.Tensor] | None = None
        self._loss_scale_ref: float | None = None
        self._grad_sink = grad_sink

        self._static_input_ids: torch.Tensor | None = None
        self._static_position_ids: torch.Tensor | None = None
        self._static_labels: torch.Tensor | None = None
        self._static_loss_mask: torch.Tensor | None = None
        self._static_rope_freqs: torch.Tensor | None = None

        self._static_loss: torch.Tensor | None = None
        self._static_grads: dict[str, torch.Tensor] | None = None

    def _warmup_and_capture(
        self,
        params: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        rope_freqs: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_scale: float,
    ) -> None:
        cfg = get_config()
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

        # 1. (optional) reference statistics from a single eager step.
        # Done BEFORE warmup/capture so the graph private mempool has
        # not yet pinned the saved-tensor (~22 GB) + grads (~3 GB) +
        # d_logits (~5.7 GB) footprint, leaving room for the eager
        # working set. We capture only reduced-footprint stats
        # (loss scalar + per-grad mean/std/abs_max, ~10 KB total) so
        # they survive across capture + replay without OOM.
        ref_stats: dict[str, Any] | None = None
        if cfg.step_cuda_graph_sanity:
            # Sanity-check the eager path with NO grad_sink so we get
            # a full grads dict back to compare against the captured
            # path's stats. The captured path may run with a device
            # sink (path-1) in which case the captured grads dict is
            # empty, but the eager reference still validates
            # fwd+CE+bwd numerics.
            eager_loss, eager_grads = _run_eager_compute_step(
                params, input_ids, rope_freqs, position_ids,
                labels, loss_mask, loss_scale,
                grad_sink=None,
            )
            torch.cuda.synchronize()
            ref_stats = self._step_stats(eager_loss, eager_grads)
            del eager_loss, eager_grads
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # 2. Warmup on a side stream so cuBLAS / TE / Triton autotune
        # completes BEFORE capture (capture-time autotune would corrupt
        # the recorded launch sequence). ``torch.cuda.graph`` requires
        # warmup outside the captured stream.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(cfg.step_cuda_graph_warmup):
                _loss, _grads = _run_eager_compute_step(
                    params,
                    self._static_input_ids, self._static_rope_freqs,
                    self._static_position_ids,
                    self._static_labels, self._static_loss_mask,
                    loss_scale,
                    grad_sink=self._grad_sink,
                )
                del _loss, _grads
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 3. Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_loss, self._static_grads = _run_eager_compute_step(
                params,
                self._static_input_ids, self._static_rope_freqs,
                self._static_position_ids,
                self._static_labels, self._static_loss_mask,
                loss_scale,
                grad_sink=self._grad_sink,
            )

        self._params_ref = params
        self._loss_scale_ref = float(loss_scale)
        self._captured = True

        # 4. (optional) replay once and compare statistics.
        if cfg.step_cuda_graph_sanity:
            if ref_stats is None:
                raise RuntimeError("ref_stats should have been captured by the sanity-check branch")
            self._graph.replay()
            torch.cuda.synchronize()
            graph_stats = self._step_stats(
                self._static_loss, self._static_grads,
            )
            self._verify_stats(ref_stats, graph_stats)

    @staticmethod
    def _step_stats(
        loss: torch.Tensor,
        grads: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        loss_val = float(loss.detach().float().item())
        grad_stats: dict[str, dict[str, float]] = {}
        for name, g in grads.items():
            f = g.detach().float()
            grad_stats[name] = {
                "mean": f.mean().item(),
                "std": f.std().item(),
                "abs_max": f.abs().max().item(),
            }
        return {"loss": loss_val, "grads": grad_stats}

    def _verify_stats(
        self,
        ref: dict[str, Any],
        got: dict[str, Any],
    ) -> None:
        loss_diff = abs(got["loss"] - ref["loss"])
        worst_grad_diff = 0.0
        worst_grad_name = ""
        worst_grad_field = ""
        # When a device-side grad_sink is bound (path-1), the captured
        # path's ``got["grads"]`` is empty (gradients land in the
        # reducer's bucket buffer, not in the dict). The eager
        # reference still has grad stats from the no-sink fwd+CE+bwd
        # path, but cross-comparison only makes sense for the loss
        # scalar; skip the grad sub-comparison in that case.
        if got["grads"]:
            for name in ref["grads"]:
                ref_g = ref["grads"][name]
                got_g = got["grads"][name]
                for key in ("mean", "std", "abs_max"):
                    d = abs(got_g[key] - ref_g[key])
                    if d > worst_grad_diff:
                        worst_grad_diff = d
                        worst_grad_name = name
                        worst_grad_field = key
        cfg = get_config()
        worst = max(loss_diff, worst_grad_diff)
        if worst > cfg.step_cuda_graph_tol:
            raise RuntimeError(
                f"[step_graph] sanity check FAILED: loss Δ={loss_diff:.3e}, "
                f"grads worst Δ={worst_grad_diff:.3e} on "
                f"{worst_grad_name!r}/{worst_grad_field} "
                f"> tol={cfg.step_cuda_graph_tol:.1e}. Capture path is numerically "
                "unsafe; disable with STEP_CUDA_GRAPH=0."
            )
        try:
            rank_str = (
                f"rank{torch.distributed.get_rank()}"
                if torch.distributed.is_initialized() else "rank0"
            )
        except Exception:
            rank_str = "rank0"
        print(
            f"[step_graph][{rank_str}] capture OK: loss Δ={loss_diff:.3e}, "
            f"grads worst Δ={worst_grad_diff:.3e} on "
            f"{worst_grad_name!r}/{worst_grad_field} "
            f"(tol={cfg.step_cuda_graph_tol:.1e}, warmup={cfg.step_cuda_graph_warmup}, "
            f"n_grads={len(self._static_grads)})",
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._params_ref is not params:
            raise RuntimeError(
                "[step_graph] params dict identity changed since "
                "capture — would invalidate recorded data_ptrs. "
                "Disable STEP_CUDA_GRAPH or recapture."
            )
        if float(loss_scale) != self._loss_scale_ref:
            raise RuntimeError(
                f"[step_graph] loss_scale changed since capture "
                f"({self._loss_scale_ref!r} → {loss_scale!r}). "
                "fused CE Triton kernel scalar arg is burnt-in at "
                "capture; recapture or disable STEP_CUDA_GRAPH."
            )
        self._static_input_ids.copy_(input_ids, non_blocking=True)
        self._static_position_ids.copy_(position_ids, non_blocking=True)
        self._static_labels.copy_(labels, non_blocking=True)
        self._static_loss_mask.copy_(loss_mask, non_blocking=True)
        if rope_freqs.data_ptr() != self._static_rope_freqs.data_ptr():
            self._static_rope_freqs.copy_(rope_freqs, non_blocking=True)
        self._graph.replay()
        return self._static_loss, self._static_grads


_COMPUTE_STEP_GRAPH_CACHE: dict[tuple, _GraphedComputeStep] = {}


def graphed_compute_step(
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_scale: float = 1.0,
    grad_sink: GradSink | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward + CE + backward in one captured CUDA Graph (M2-P32 Phase A).

    Falls through to eager execution unless ``STEP_CUDA_GRAPH=1`` is set.

    ``grad_sink`` selects the backward write protocol the graph
    captures:

    * ``None`` (Phase A default) — grads land in a dict; caller must
      have ``grad_reducer is None``.
    * a capture-safe callable (typically
      ``BucketedGradReducer.make_device_grad_sink()``) — grads land
      in the reducer's bucket buffers via in-place ``add_``; the
      returned dict is empty. This is path-1 (Direction-C) and
      requires the train loop to:

      1. Call ``reducer.zero_input_bufs()`` once per step before the
         first MB's replay.
      2. Call ``reducer.flush_all_buckets()`` after the final MB's
         replay to dispatch the reduce_scatters.

    Different per-call shapes get distinct cached graphs; cache key
    also includes the sink identity so eager-fallthrough captures
    can coexist with reducer captures if ever co-trained.
    """
    cfg = get_config()
    if not cfg.step_cuda_graph:
        return _run_eager_compute_step(
            params, input_ids, rope_freqs, position_ids,
            labels, loss_mask, loss_scale,
            grad_sink=grad_sink,
        )

    key = (
        tuple(input_ids.shape),
        tuple(position_ids.shape),
        tuple(labels.shape),
        tuple(loss_mask.shape),
        tuple(rope_freqs.shape),
        input_ids.dtype, labels.dtype, loss_mask.dtype, rope_freqs.dtype,
        id(grad_sink) if grad_sink is not None else None,
    )
    cell = _COMPUTE_STEP_GRAPH_CACHE.get(key)
    if cell is None:
        cell = _GraphedComputeStep(grad_sink=grad_sink)
        cell._warmup_and_capture(
            params, input_ids, rope_freqs, position_ids,
            labels, loss_mask, loss_scale,
        )
        _COMPUTE_STEP_GRAPH_CACHE[key] = cell
    return cell.replay(
        params, input_ids, rope_freqs, position_ids,
        labels, loss_mask, loss_scale,
    )


def is_step_cuda_graph_enabled() -> bool:
    cfg = get_config()
    return cfg.step_cuda_graph


def _clear_step_graph_cache() -> None:
    _COMPUTE_STEP_GRAPH_CACHE.clear()


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402, I001
_register_reset_hook(_clear_step_graph_cache)
