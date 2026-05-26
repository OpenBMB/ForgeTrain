"""CUDA Graph capture for ``forward_pass_with_save`` (M2-P31).

Wraps :func:`training_engine_tensor.forward.forward_pass_with_save` in a
captured CUDA Graph so the ~300 kernel launches per forward pass
collapse into a single ``graph.replay()`` call. Targets the launch-bound
fraction of forward-pass step time (RoPE, RMSNorm, SwiGLU and other
short Triton/elementwise kernels interleaved with cuBLAS GEMM and TE
fused attention).

Opt-in via env ``FORWARD_CUDA_GRAPH=1`` (default 0).  Falls back to
eager execution unless ALL of the following hold:
  * ``FORWARD_CUDA_GRAPH=1``;
  * ``position_ids is not None`` (doc-aware fused RoPE branch — the
    non-doc-aware branch in ``forward.py`` calls ``requires_grad_(True)
    + q_rot.backward(retain_graph=True)`` which is not capture-safe);
  * ``need_backward=True`` (training path; we capture the
    ``saved`` dict shape established by this branch).

Capture safety contract
-----------------------
* ``saved`` returned from ``replay`` references graph-internal buffers.
  The immediate consumer (CE + backward) MUST finish reading every
  saved tensor before the next ``replay()`` is invoked. The current
  ``train_loop._train_one_step`` enforces this naturally because
  forward → CE → backward runs strictly serial within a microbatch.
* ``params`` dict entries are bound by ``data_ptr`` at capture time.
  ``optimizer.sync_params_from_master`` (and the M2-P15 ``fused_adam_sync``
  path) updates parameters with ``copy_`` (in-place), so ``data_ptr``
  is stable across steps. Verified at ``optimizer.py:L301``.
* ``rope_freqs`` is constant after ``precompute_rope_freqs`` (called
  once at startup), so we skip the per-step ``copy_`` to save one D2D
  copy. ``input_ids`` and ``position_ids`` are copied per step because
  their values change while shapes stay fixed (the
  ``CONDITIONAL_FIXED_LENGTH_SEGMENT`` packer pads every microbatch to
  ``seq_length=4096``, so shapes are static).

Numerical sanity check
----------------------
Optional, OFF by default (``FORWARD_CUDA_GRAPH_SANITY=0``). The cctl
loss-gate suite is the canonical end-to-end correctness signal — a
divergent graph would blow up ``avg_rel_diff`` within the first 50
steps. When opt-in, the wrapper runs ONE eager forward BEFORE warmup
(so the graph private mempool has not yet been allocated and the
~22 GB of saved-tensor footprint is briefly free), captures
reduced-footprint reference statistics (mean/std/max + a 1/1024
strided sample of the logits, total ~5 MB), then replays the captured
graph and compares: aborts if ``max|Δ stats| > FORWARD_CUDA_GRAPH_TOL``
(default 1e-3 BF16). This deliberately avoids cloning the 5.7 GB
full logits tensor — at capture time the H100 SXM (80 GB) is already
~78 GB used (params + master + reduce_scatter buckets + sharded
optimizer + per-step activations) so a full clone would OOM.
"""

from __future__ import annotations

__all__ = ["graphed_forward_pass_with_save", "is_forward_cuda_graph_enabled"]

from typing import Any

import torch

from .engine_config import get_config
from .forward import forward_pass_with_save


class _GraphedForward:
    """One captured CUDAGraph for a fixed (input shape × params dict) pair."""

    def __init__(self) -> None:
        self._captured: bool = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._params_ref: dict[str, torch.Tensor] | None = None
        self._static_input_ids: torch.Tensor | None = None
        self._static_rope_freqs: torch.Tensor | None = None
        self._static_position_ids: torch.Tensor | None = None
        self._static_logits: torch.Tensor | None = None
        self._static_saved: dict[str, Any] | None = None

    def _warmup_and_capture(
        self,
        params: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        rope_freqs: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        cfg = get_config()
        # Static buffers — graph captures their data_ptr; on replay we
        # ``copy_`` real per-step inputs into them.
        self._static_input_ids = torch.empty_like(input_ids)
        self._static_rope_freqs = torch.empty_like(rope_freqs)
        self._static_position_ids = torch.empty_like(position_ids)
        self._static_input_ids.copy_(input_ids)
        self._static_rope_freqs.copy_(rope_freqs)
        self._static_position_ids.copy_(position_ids)

        # 1. (optional) reference statistics from a single eager forward.
        # Done BEFORE warmup/capture so the graph private mempool has
        # not yet pinned the ~22 GB of saved-tensor footprint, leaving
        # room for the ~640 MiB eager intermediates. We capture only
        # reduced-footprint stats (~5 MB) so they survive across the
        # capture + replay phases without OOM. See module docstring.
        ref_stats: dict[str, torch.Tensor | float] | None = None
        if cfg.forward_cuda_graph_sanity:
            eager_logits, eager_saved = forward_pass_with_save(
                params, input_ids, rope_freqs, position_ids=position_ids,
            )
            torch.cuda.synchronize()
            ref_stats = self._logits_stats(eager_logits)
            del eager_logits, eager_saved
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # 2. Warmup on a side stream so cuBLAS / TE / Triton autotune
        # completes BEFORE capture (capture-time autotune would corrupt
        # the recorded launch sequence). ``torch.cuda.graph`` requires
        # warmup outside the captured stream.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(cfg.forward_cuda_graph_warmup):
                logits, saved = forward_pass_with_save(
                    params,
                    self._static_input_ids,
                    self._static_rope_freqs,
                    position_ids=self._static_position_ids,
                )
                del logits, saved
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 3. Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_logits, self._static_saved = forward_pass_with_save(
                params,
                self._static_input_ids,
                self._static_rope_freqs,
                position_ids=self._static_position_ids,
            )

        self._params_ref = params
        self._captured = True

        # 4. (optional) replay once and compare statistics.
        if cfg.forward_cuda_graph_sanity:
            if ref_stats is None:
                raise RuntimeError("ref_stats should have been captured by the sanity-check branch")
            self._graph.replay()
            torch.cuda.synchronize()
            graph_stats = self._logits_stats(self._static_logits)
            self._verify_stats(ref_stats, graph_stats)

    @staticmethod
    def _logits_stats(logits: torch.Tensor) -> dict[str, torch.Tensor | float]:
        """Reduced-footprint reference stats (~5 MB). Includes scalar
        moments + a strided sample of the flattened logits — enough to
        detect any non-trivial graph-vs-eager divergence without paying
        the 5.7 GB full-clone cost.
        """
        f = logits.detach().float()
        flat = f.view(-1)
        cfg = get_config()
        return {
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "abs_max": flat.abs().max().item(),
            "sample": flat[::cfg.forward_cuda_graph_sample_stride].clone(),
        }

    def _verify_stats(
        self,
        ref: dict[str, torch.Tensor | float],
        got: dict[str, torch.Tensor | float],
    ) -> None:
        scalar_diffs = {
            k: abs(float(got[k]) - float(ref[k]))
            for k in ("mean", "std", "abs_max")
        }
        sample_diff = (
            (got["sample"] - ref["sample"]).abs().max().item()  # type: ignore[union-attr]
        )
        cfg = get_config()
        worst = max(max(scalar_diffs.values()), sample_diff)
        if worst > cfg.forward_cuda_graph_tol:
            raise RuntimeError(
                f"[forward_graph] sanity check FAILED: "
                f"scalar Δ={scalar_diffs}, sample max|Δ|={sample_diff:.3e} "
                f"> tol={cfg.forward_cuda_graph_tol:.1e}. Capture path is numerically "
                "unsafe; disable with FORWARD_CUDA_GRAPH=0."
            )
        try:
            rank_str = (
                f"rank{torch.distributed.get_rank()}"
                if torch.distributed.is_initialized()
                else "rank0"
            )
        except Exception:
            rank_str = "rank0"
        print(
            f"[forward_graph][{rank_str}] capture OK: scalar Δ={scalar_diffs}, "
            f"sample max|Δ|={sample_diff:.3e} (tol={cfg.forward_cuda_graph_tol:.1e}, "
            f"warmup={cfg.forward_cuda_graph_warmup}, saved_keys={len(self._static_saved)})",
            flush=True,
        )

    def replay(
        self,
        params: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        rope_freqs: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._params_ref is not params:
            raise RuntimeError(
                "[forward_graph] params dict identity changed since "
                "capture — this would break the recorded data_ptrs. "
                "Disable FORWARD_CUDA_GRAPH or recapture."
            )
        self._static_input_ids.copy_(input_ids, non_blocking=True)
        self._static_position_ids.copy_(position_ids, non_blocking=True)
        if rope_freqs.data_ptr() != self._static_rope_freqs.data_ptr():
            self._static_rope_freqs.copy_(rope_freqs, non_blocking=True)
        self._graph.replay()
        return self._static_logits, self._static_saved


_FORWARD_GRAPH_CACHE: dict[tuple, _GraphedForward] = {}


def graphed_forward_pass_with_save(
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    need_backward: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Drop-in replacement for :func:`forward_pass_with_save` with
    optional CUDA Graph capture (M2-P31).

    Falls through to eager execution unless ``FORWARD_CUDA_GRAPH=1``
    AND the doc-aware path is active (``position_ids is not None``)
    AND the call is on the training path (``need_backward=True``).
    Different ``input_ids`` / ``position_ids`` shapes get distinct
    cached graphs.
    """
    cfg = get_config()
    if not cfg.forward_cuda_graph:
        return forward_pass_with_save(
            params, input_ids, rope_freqs,
            position_ids=position_ids, need_backward=need_backward,
        )
    if position_ids is None or not need_backward:
        return forward_pass_with_save(
            params, input_ids, rope_freqs,
            position_ids=position_ids, need_backward=need_backward,
        )

    key = (
        tuple(input_ids.shape),
        tuple(rope_freqs.shape),
        tuple(position_ids.shape),
        input_ids.dtype,
        rope_freqs.dtype,
    )
    cell = _FORWARD_GRAPH_CACHE.get(key)
    if cell is None:
        cell = _GraphedForward()
        cell._warmup_and_capture(params, input_ids, rope_freqs, position_ids)
        _FORWARD_GRAPH_CACHE[key] = cell
    return cell.replay(params, input_ids, rope_freqs, position_ids)


def is_forward_cuda_graph_enabled() -> bool:
    cfg = get_config()
    return cfg.forward_cuda_graph


def _clear_forward_graph_cache() -> None:
    _FORWARD_GRAPH_CACHE.clear()


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402, I001
_register_reset_hook(_clear_forward_graph_cache)
