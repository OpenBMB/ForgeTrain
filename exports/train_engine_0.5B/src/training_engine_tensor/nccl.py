"""NCCL communication primitives for MiniCPM4 0.5B.

Uses torch.distributed as the backend — provides allreduce, reduce_scatter,
all_gather, and baseline-compatible gradient buffer management for DP training.

Two complementary code paths live here:

- The legacy single-buffer path (``reduce_scatter_grads`` /
  ``compute_distributed_grad_norm`` / ``allgather_grads``) packs the
  entire model gradient into one FP32 flat tensor, does ONE
  reduce_scatter and ONE all_gather. This is the default path used by
  the production CLI and every gate today; it leaves the ~18 ms of
  collective wall-clock fully exposed in the step time (see
  ``perf_log.md`` Round M2-P0.5).
- The bucketed overlap path (``BucketedGradReducer`` plus the
  ``get_comm_stream`` / ``compute_grad_buckets`` helpers) shards the
  buffer into a per-layer set of small FP32 buffers and dispatches the
  reduce_scatter on a dedicated communication stream as soon as each
  bucket fills during backward. Backward layers continue to run on the
  default stream so the collective overlaps with the next layers'
  ``wgrad`` GEMMs. This path is **opt-in**: hot-path callers must set
  ``WGRAD_OVERLAP=1`` AND wire the reducer in (``train_loop.py``).

Both paths produce per-parameter FP32 gradient dicts with the same
shape / dtype contract, so the downstream optimizer / grad-clip /
sync code is identical regardless of which path produced the grads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "BucketedGradReducer",
    "NcclStaticBuffers",
    "allgather_grads",
    "allgather_grads_persistent",
    "barrier",
    "baseline_buffer_layout",
    "compute_distributed_grad_norm",
    "compute_distributed_grad_norm_tensor",
    "get_comm_stream",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "reduce_scatter_grads",
    "reduce_scatter_grads_persistent",
]

from . import config
from .engine_config import get_config


def is_distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def init_distributed(
    backend: str = "nccl",
    rank: int | None = None,
    world_size: int | None = None,
    timeout_seconds: int = 120,
) -> None:
    """Initialize torch.distributed if not already done."""
    if dist.is_initialized():
        return
    import datetime
    kwargs: dict[str, Any] = {"backend": backend}
    if rank is not None:
        kwargs["rank"] = rank
    if world_size is not None:
        kwargs["world_size"] = world_size
    kwargs["timeout"] = datetime.timedelta(seconds=timeout_seconds)
    dist.init_process_group(**kwargs)


def baseline_buffer_layout() -> list[tuple[str, int]]:
    """Return (param_name, numel) in baseline reversed-registration order.

    Derived from ``parameters.all_param_specs()`` — the single source of
    truth.  The forward-registration order produced by ``all_param_specs``
    is reversed to match Megatron's DistributedOptimizer _ParamAndGradBuffer
    layout, with the following twist for main-decoder layers: within each
    layer the reverse order swaps ``attention.wo`` to *after*
    ``attention_norm`` (Megatron registers ``linear_proj`` before
    ``pre_mlp_layernorm``, so in reverse they swap).  The forward-order
    sub-sequence for a main layer is:
        attention_norm → wqkv → wo → ffn_norm → wfc1 → w2
    The expected reversed sub-sequence in the buffer is:
        w2 → wfc1 → ffn_norm → wqkv → attention_norm → wo

    MTP layers follow standard reverse order (no swap).
    """
    from .parameters import all_param_specs  # noqa: I001 — lazy to avoid nccl→parameters top-level dep
    spec_map = {s.name: s.numel for s in all_param_specs()}

    entries: list[tuple[str, int]] = []

    def _emit(name: str) -> None:
        entries.append((name, spec_map[name]))

    _emit("output.weight")

    mtp_n = get_config().mtp_num_layers
    for k in range(mtp_n - 1, -1, -1):
        p = f"mtp.layers.{k}"
        _emit(f"{p}.feed_forward.w2.weight")
        _emit(f"{p}.feed_forward.wfc1.weight")
        _emit(f"{p}.ffn_norm.weight")
        _emit(f"{p}.attention.wo.weight")
        _emit(f"{p}.attention.wqkv.weight")
        _emit(f"{p}.attention_norm.weight")
        _emit(f"{p}.eh_proj.weight")
        _emit(f"{p}.hnorm.weight")
        _emit(f"{p}.enorm.weight")

    _emit("norm.weight")

    for i in range(config.NUM_LAYERS - 1, -1, -1):
        p = f"layers.{i}"
        _emit(f"{p}.feed_forward.w2.weight")
        _emit(f"{p}.feed_forward.wfc1.weight")
        _emit(f"{p}.ffn_norm.weight")
        _emit(f"{p}.attention.wqkv.weight")
        _emit(f"{p}.attention_norm.weight")
        _emit(f"{p}.attention.wo.weight")

    _emit("tok_embeddings.weight")
    return entries


def reduce_scatter_grads(
    param_grads: dict[str, torch.Tensor],
    device: str | torch.device,
) -> tuple[torch.Tensor, int, list[tuple[str, int]]]:
    """Pack gradients into a flat FP32 buffer and reduce_scatter across DP ranks.

    Returns (fp32_shard, shard_size, buf_entries) where fp32_shard is this
    rank's portion of the averaged gradient buffer.
    """
    world_size = get_world_size()
    buf_entries = baseline_buffer_layout()
    buf_total = sum(sz for _, sz in buf_entries)
    bucket_align = math.lcm(world_size, 128)
    buf_size = int(math.ceil(buf_total / bucket_align) * bucket_align)
    shard_size = buf_size // world_size

    fp32_flat = torch.zeros(buf_size, dtype=torch.float32, device=device)
    off = 0
    for name, mg_numel in buf_entries:
        g = param_grads[name].reshape(-1).float()
        fp32_flat[off:off + mg_numel] = g
        off += mg_numel

    fp32_flat.mul_(1.0 / world_size)

    # `reduce_scatter_tensor` output must not alias the input tensor. Using a
    # view into `fp32_flat` here can corrupt the reduction on some NCCL/PyTorch
    # combinations and inflate the reported grad norm.
    fp32_shard = torch.empty(shard_size, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(fp32_shard, fp32_flat, op=dist.ReduceOp.SUM)

    return fp32_shard, shard_size, buf_entries


# ---------------------------------------------------------------------
# M2-P33 Phase B-2: persistent flat buffers (CUDA Graph compatible)
# ---------------------------------------------------------------------
#
# The legacy ``reduce_scatter_grads`` / ``allgather_grads`` pair allocates
# two giant FP32 flat tensors every step (~5.7 GB combined for the
# 0.5 B model on FP32). Inside ``torch.cuda.graph(...)`` capture, every
# allocation is recorded as a node — the graph then locks down the
# resulting ``data_ptr`` and replay would hit a use-after-free if the
# allocator re-uses the address. Persistent module-level buffers
# eliminate the per-step alloc + free, so each capture sees the same
# storage on every replay.
#
# We piggy-back on the persistent ``fp32_flat`` for the all-gather
# output too — after the grad-norm tensor is computed, the buffer is
# free for the all-gather to overwrite (the down-stream consumer reads
# from the **same** buffer slot via per-param views into ``fp32_flat``).


class NcclStaticBuffers:
    """Module-level persistent buffers for the non-bucketed NCCL path.

    Holds a single FP32 flat buffer sized to ``world_size``-aligned
    total grad numel. The same buffer is used for:

    * ``reduce_scatter_tensor`` output  → owns ``[rank * shard_size,
      (rank+1) * shard_size)``;
    * ``all_gather_into_tensor`` output → owns the entire range.

    The shard-size and ``buf_entries`` are derived once from the
    baseline layout and stored on the instance so every per-step path
    can pull them without re-running the layout helper.

    Lifecycle: one ``NcclStaticBuffers`` per training run, owned by
    ``train_loop`` (or whichever driver). The same instance MUST flow
    through every ``reduce_scatter_grads_persistent`` /
    ``compute_distributed_grad_norm_tensor`` /
    ``allgather_grads_persistent`` call so the captured graph's
    ``data_ptr`` remains valid forever.
    """

    def __init__(self, device: str | torch.device) -> None:
        self.device = device
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.buf_entries = baseline_buffer_layout()
        buf_total = sum(sz for _, sz in self.buf_entries)
        bucket_align = math.lcm(self.world_size, 128)
        self.buf_size = int(math.ceil(buf_total / bucket_align) * bucket_align)
        self.shard_size = self.buf_size // self.world_size

        self.fp32_flat = torch.zeros(
            self.buf_size, dtype=torch.float32, device=device,
        )
        self.fp32_avg = torch.empty(
            self.buf_size, dtype=torch.float32, device=device,
        )
        # Persistent buffer for the squared distributed grad norm — the
        # captured ``compute_distributed_grad_norm_tensor`` writes into
        # this slot so the host can ``.item()`` the value at the end of
        # the step without breaking capture (the read happens AFTER the
        # graph replay completes, outside the captured stream).
        self.grad_norm_sq_buf = torch.zeros(
            1, dtype=torch.float32, device=device,
        )

        self.shard_view = self.fp32_flat[
            self.rank * self.shard_size:(self.rank + 1) * self.shard_size
        ]

        self._param_offsets: dict[str, tuple[int, int]] = {}
        off = 0
        for name, numel in self.buf_entries:
            self._param_offsets[name] = (off, numel)
            off += numel

        # Cache of param-name → view-into-fp32_avg so callers
        # (allgather_grads_persistent) can return the SAME dict /
        # tensor instances every step — required by the Phase B
        # optimizer capture which records data_ptrs at capture time.
        self._cached_grad_views: dict[str, torch.Tensor] | None = None
        self._cached_grad_views_params: dict[str, torch.Tensor] | None = None

    @property
    def cached_grad_views(self) -> dict[str, torch.Tensor] | None:
        """Per-param FP32 gradient views, populated after allgather_grads_persistent."""
        return self._cached_grad_views

    def param_view(
        self, name: str, params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return a typed view into ``fp32_avg`` for a single param.

        Aliases the persistent ``fp32_avg`` buffer at the param's slot
        in baseline order. The caller MUST consume the view before the
        next step's all-gather overwrites the slot — guaranteed by the
        per-step ``torch.cuda.synchronize`` boundary plus the optimizer
        contract that grads are read-only after consumption.
        """
        offset, numel = self._param_offsets[name]
        shape = params[name].shape
        return self.fp32_avg[offset:offset + numel].view(shape)


def reduce_scatter_grads_persistent(
    bufs: NcclStaticBuffers,
    param_grads: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pack grads into ``bufs.fp32_flat`` and reduce_scatter (no fresh alloc).

    The pre-allocated shard view ``bufs.shard_view`` receives this
    rank's portion of the averaged gradient. Returns the same view for
    convenience; callers should NOT free or reassign it.
    """
    fp32_flat = bufs.fp32_flat
    off = 0
    for name, mg_numel in bufs.buf_entries:
        # ``copy_`` does the BF16/FP16→FP32 dtype cast in-place into the
        # persistent slot, no fresh ``.float()`` allocation per param.
        fp32_flat[off:off + mg_numel].copy_(
            param_grads[name].reshape(-1),
        )
        off += mg_numel
    fp32_flat.mul_(1.0 / bufs.world_size)
    if is_distributed():
        dist.reduce_scatter_tensor(
            bufs.shard_view, fp32_flat, op=dist.ReduceOp.SUM,
        )
    return bufs.shard_view


def compute_distributed_grad_norm_tensor(
    fp32_shard: torch.Tensor,
    bufs: NcclStaticBuffers,
) -> torch.Tensor:
    """Tensor-only variant of :func:`compute_distributed_grad_norm`.

    Returns a 1-element FP32 device tensor holding the squared L2 norm
    of the **distributed** gradients (i.e. the norm summed across all
    DP ranks). The caller can either:

    * read ``.item() ** 0.5`` to recover the legacy float (this
      stays compatible with the eager scalar-arg optimizer); or
    * pass it directly to :func:`compute_clip_coeff_device` to feed
      the tensor-arg optimizer without crossing the host boundary.

    Implementation notes (CUDA Graph safety, M2-P33):

    * All math runs in pure ``aten`` ops — we deliberately avoid
      :func:`multi_tensor_applier` / ``multi_tensor_l2norm`` because
      the Apex-style chunked kernel allocates per-chunk scratch on
      every call, which is incompatible with graph capture (the
      private mempool would record a different ``data_ptr`` each
      replay).
    * The squared-norm value is written directly into
      ``bufs.grad_norm_sq_buf`` (a 1-element FP32 device tensor that
      lives for the entire training run). All-reduce then runs
      in-place on that same buffer, so no fresh tensor is allocated
      inside the captured region.
    * The sum-of-squares is over the *entire* ``fp32_shard``
      (``shard_size`` elements). Padding slots beyond the last param
      stay zero throughout training (they are zero-initialised in
      :class:`NcclStaticBuffers` and never written), so they do not
      contribute to the squared norm. This matches the legacy
      ``multi_tensor_l2norm`` value to within FP32 rounding order
      (the loss-gate's 1% rel-diff tolerance subsumes the order-
      sensitive rounding difference).
    """
    if fp32_shard.numel() > 0:
        local_sq = (fp32_shard.float() ** 2).sum()
    else:
        local_sq = torch.zeros((), dtype=torch.float32, device=bufs.device)
    bufs.grad_norm_sq_buf.view(()).copy_(local_sq.view(()))
    if is_distributed():
        dist.all_reduce(bufs.grad_norm_sq_buf, op=dist.ReduceOp.SUM)
    return bufs.grad_norm_sq_buf


def allgather_grads_persistent(
    bufs: NcclStaticBuffers,
    fp32_shard: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """All-gather ``fp32_shard`` into the persistent ``bufs.fp32_avg`` and
    return per-param views (no fresh dict-of-clones).

    The returned dict aliases ``bufs.fp32_avg``; consumers must read
    grads before the next step's :func:`reduce_scatter_grads_persistent`
    overwrites the slots. The per-step ``torch.cuda.synchronize``
    boundary plus the optimizer's read-only contract make that safe.

    The dict object is cached on ``bufs`` so the same dict instance is
    returned on every call (with the same view tensors aliasing the same
    storage). Stable identity is required by the Phase B optimizer
    capture (``step_graph_optimizer``) which reuses the recorded
    ``data_ptr`` s on every replay — recreating the dict each step
    would force capture-cache invalidation.
    """
    if is_distributed():
        dist.all_gather_into_tensor(bufs.fp32_avg, fp32_shard.contiguous())
    else:
        bufs.fp32_avg.copy_(bufs.fp32_flat)
    if bufs._cached_grad_views is None or bufs._cached_grad_views_params is not params:
        bufs._cached_grad_views = {
            name: bufs.param_view(name, params)
            for name, _ in bufs.buf_entries
        }
        bufs._cached_grad_views_params = params
    return bufs._cached_grad_views


_norm_param_names_cache: frozenset[str] | None = None


def _clear_norm_param_names_cache() -> None:
    """Clear cached norm param names. Called when global config is reset."""
    global _norm_param_names_cache
    _norm_param_names_cache = None


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: I001
_register_reset_hook(_clear_norm_param_names_cache)


def _norm_param_names() -> frozenset[str]:
    """Norm param names derived from ``ParamSpec.is_norm`` (SSOT)."""
    global _norm_param_names_cache
    if _norm_param_names_cache is None:
        from .parameters import all_param_specs  # noqa: I001 — lazy to avoid nccl→parameters top-level dep
        _norm_param_names_cache = frozenset(
            s.name for s in all_param_specs() if s.is_norm
        )
    return _norm_param_names_cache


def _is_norm_param(name: str) -> bool:
    return name in _norm_param_names()


def compute_distributed_grad_norm(
    fp32_shard: torch.Tensor,
    shard_size: int,
    buf_entries: list[tuple[str, int]],
    device: str | torch.device,
) -> float:
    """Compute L2 grad norm from sharded gradients via all_reduce.

    Matches the baseline DistributedOptimizer grad norm computation:
    weight grads first, then norm grads, each shard's local l2norm
    squared then all_reduced.
    """
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
    )

    rank = get_rank()
    shard_start = rank * shard_size

    weight_grads = []
    norm_grads = []
    off = 0
    for name, mg_numel in buf_entries:
        p_start = off
        p_end = off + mg_numel
        ov_start = max(p_start, shard_start)
        ov_end = min(p_end, shard_start + shard_size)
        if ov_start < ov_end:
            local_s = ov_start - shard_start
            local_e = ov_end - shard_start
            g = fp32_shard[local_s:local_e]
            if _is_norm_param(name):
                norm_grads.append(g)
            else:
                weight_grads.append(g)
        off += mg_numel

    shard_grads = weight_grads + norm_grads

    dummy = torch.zeros(1, dtype=torch.int, device=device)
    if shard_grads:
        local_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy, [shard_grads], False,
        )
    else:
        local_norm = torch.zeros(1, dtype=torch.float32, device=device)

    total_norm_sq = local_norm ** 2
    dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
    return total_norm_sq.item() ** 0.5


def allgather_grads(
    fp32_shard: torch.Tensor,
    buf_size: int,
    params: dict[str, torch.Tensor],
    buf_entries: list[tuple[str, int]],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """All_gather shards and unpack into per-parameter gradient dict."""
    fp32_avg = torch.empty(buf_size, dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(fp32_avg, fp32_shard.contiguous())

    fp32_grads: dict[str, torch.Tensor] = {}
    off = 0
    for name, mg_numel in buf_entries:
        shape = params[name].shape
        fp32_grads[name] = fp32_avg[off:off + mg_numel].view(shape).clone()
        off += mg_numel

    return fp32_grads



# ===========================================================================
# Bucketed grad-reduce overlap (opt-in via ``WGRAD_OVERLAP=1``)
# ===========================================================================
#
# Design rationale (see ``recommendation.md`` Round M2-P1.6 §M2-P2 for the
# full motivation):
#
# Each backward layer produces a few FP32 weight gradients. The legacy path
# accumulates ALL gradients into a single flat buffer, then issues exactly one
# ``reduce_scatter`` after backward finishes — leaving ~18 ms of collective
# wall-clock fully exposed in the 346 ms step (~5.3% headroom on DP=16).
#
# The bucketed overlap path partitions the same flat layout into small
# per-layer buckets and issues an async ``reduce_scatter`` on a dedicated
# communication stream as soon as each bucket fills during backward. The
# default stream keeps running the next layer's backward GEMMs while the
# previous layer's gradients are reduced. ``wait_all`` then drains the
# comm-stream events into the default stream so the optimizer can read the
# averaged gradients safely.
#
# The reducer follows three hard invariants:
#
# 1. **FP32 buffers.** Reduce-scatter input/output and the gathered grad
#    buffer are FP32 throughout, matching the FP32 9-environment rule
#    (``minicpm5_kernel_framework_rules.md`` §FP32 §6 — AllReduce buffer).
# 2. **Bucket size is a multiple of ``world_size``.** Each bucket pads the
#    raw param numel up to ``world_size`` so reduce_scatter can shard it
#    evenly. The trailing padding slots are zero-initialised once at
#    construction and are never written to thereafter, so the reduced
#    output of the padding region is also zero (no contamination of the
#    grad-norm or optimizer math).
# 3. **Single comm stream, ordered events.** All buckets use the same
#    dedicated communication stream so NCCL operations cannot reorder.
#    Each bucket records a CUDA event upon completion so ``wait_all`` can
#    insert fine-grained waits into the default stream without a global
#    ``synchronize``.
#
# The reducer is constructed once per training run and reused across steps
# (the buffers are persistent; only the per-bucket pending counters and
# completion events are reset each step).

_comm_stream: torch.cuda.Stream | None = None


def get_comm_stream(device: str | torch.device) -> torch.cuda.Stream:
    """Return (lazily-init) the dedicated communication stream.

    The stream lives on the supplied device. Tests should call
    :func:`reset_comm_stream` between scenarios to drop the singleton.
    """
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream(device=device)
    return _comm_stream


def reset_comm_stream() -> None:
    """Drop the cached communication stream (for tests)."""
    global _comm_stream
    _comm_stream = None


@dataclass(frozen=True)
class BucketSpec:
    """Specification of a single gradient bucket within the reducer.

    Attributes:
        params: Ordered ``(name, numel)`` entries assigned to this bucket.
            Order MUST match ``baseline_buffer_layout`` so the post-allgather
            unpacking maps each param back to its baseline-buffer slot.
        param_offsets: 0-indexed offsets (in element units) of each param
            within the bucket buffer; ``param_offsets[i]`` is where
            ``params[i]`` starts. Always parallel to ``params``.
        raw_size: Sum of all params' numel (without padding).
        size: Padded bucket size — multiple of ``world_size``. Equals
            ``raw_size`` plus zero or more padding slots at the tail.
    """

    params: tuple[tuple[str, int], ...]
    param_offsets: tuple[int, ...]
    raw_size: int
    size: int


def compute_grad_buckets(
    buf_entries: list[tuple[str, int]],
    world_size: int,
    *,
    bucket_target_elems: int = 25 * 1024 * 1024,
) -> list[BucketSpec]:
    """Partition ``buf_entries`` into ``world_size``-aligned buckets.

    Greedy strategy: walk ``buf_entries`` in order; accumulate the current
    bucket's raw size until it reaches ``bucket_target_elems`` (or the
    next param would push it past), then close the bucket and round up to
    the next multiple of ``world_size``. A bucket always contains at
    least one param even if that single param exceeds the target — large
    layers (e.g. the embedding) get their own bucket rather than being
    split across two.

    The first param of a layer typically completes shortly after the
    layer's backward GEMMs, so per-layer bucketing maximises the
    backward / reduce_scatter overlap. With the 0.5 B-param model and a
    25 M-element target, this produces ~26 buckets (head + 24 layers +
    embedding), each with predictable shard alignment.

    Raises:
        ValueError: if ``world_size`` is non-positive or ``bucket_target_elems``
            is non-positive (fail-fast — silently sliding past these is
            how a 0-byte bucket would later trigger a NCCL hang).
    """
    if world_size <= 0:
        raise ValueError(
            f"compute_grad_buckets: world_size must be positive, got {world_size}"
        )
    if bucket_target_elems <= 0:
        raise ValueError(
            f"compute_grad_buckets: bucket_target_elems must be positive, "
            f"got {bucket_target_elems}"
        )

    buckets: list[BucketSpec] = []
    cur_params: list[tuple[str, int]] = []
    cur_offsets: list[int] = []
    cur_raw = 0

    def _close_current() -> None:
        nonlocal cur_params, cur_offsets, cur_raw
        if not cur_params:
            return
        padded = ((cur_raw + world_size - 1) // world_size) * world_size
        buckets.append(
            BucketSpec(
                params=tuple(cur_params),
                param_offsets=tuple(cur_offsets),
                raw_size=cur_raw,
                size=padded,
            )
        )
        cur_params = []
        cur_offsets = []
        cur_raw = 0

    for name, numel in buf_entries:
        if cur_raw > 0 and cur_raw + numel > bucket_target_elems:
            _close_current()
        cur_offsets.append(cur_raw)
        cur_params.append((name, numel))
        cur_raw += numel
    _close_current()

    return buckets


class BucketedGradReducer:
    """Per-bucket async reduce_scatter manager for wgrad-allreduce overlap.

    Hot-path lifecycle (driven by ``train_loop.py`` when
    ``WGRAD_OVERLAP=1``):

    1. ``__init__(buf_entries, world_size, rank, device, …)`` — allocates
       per-bucket FP32 input + shard buffers, computes the bucket layout,
       initialises the comm stream.
    2. ``start_step()`` at the top of each training step — resets the
       per-bucket pending counters and clears the previous step's
       completion events. Persistent buffers are NOT zeroed (they will be
       overwritten by ``write_grad`` calls; padding regions stay zero).
    3. ``write_grad(name, grad)`` from the backward callback — copies
       ``grad.float()`` into the bucket's slot on the default stream.
       When the bucket's last param arrives, records a default-stream
       event, switches to the comm stream, scales the bucket by
       ``1/world_size`` and issues ``reduce_scatter_tensor``. The
       corresponding completion event is recorded on the comm stream.
    4. ``wait_all()`` after backward returns — inserts a per-bucket
       ``wait_event`` on the default stream so subsequent default-stream
       work (grad_norm / optimizer / allgather unpack) sees the reduced
       gradients in deterministic order.
    5. ``allgather_and_unpack(params)`` — issues per-bucket
       ``all_gather_into_tensor`` (in-place onto the input buffer) and
       returns a per-parameter FP32 grad dict with the original shapes.
    6. ``compute_distributed_grad_norm(device)`` — fused per-bucket L2
       grad norm path that walks each bucket's local shard, classifies
       weight vs norm grads matching the legacy ordering, and all_reduces
       the squared norm. Provides the same numeric contract as
       :func:`compute_distributed_grad_norm` so downstream clip / Adam
       behave identically.

    Multi-microbatch (gradient accumulation) extension
    --------------------------------------------------
    When ``num_local_microbatches > 1``, the reducer accumulates each
    bucket's input buffer across ``N`` microbatches *before* issuing the
    reduce_scatter. The state machine adds a per-bucket ``mb_remaining``
    counter (separate from ``_bucket_pending`` which tracks per-MB
    "params still to arrive in this MB"):

    * Microbatch 1 — first arrival of every bucket: ``copy_`` into the
      FP32 slot (matches the single-MB behaviour, padding stays zero).
    * Microbatches 2..N-1 — ``add_`` into the same slot. No
      reduce_scatter yet.
    * Microbatch N — ``add_`` into the slot, then issue reduce_scatter
      exactly as the single-MB path does.

    Effect: the dict-of-FP32 accumulation in ``train_loop`` is replaced
    by an in-place add on the persistent bucket buffer (zero extra
    allocations, zero per-MB cast of every wgrad), and wgrad-allreduce
    overlap (reduce_scatter on the comm stream after the *final* MB)
    becomes legal in the gradient-accumulation regime. Per-MB writes
    are bytewise identical to the legacy "MB==1" path when N==1.
    """

    def __init__(
        self,
        buf_entries: list[tuple[str, int]],
        world_size: int,
        rank: int,
        device: str | torch.device,
        *,
        bucket_target_elems: int = 25 * 1024 * 1024,
        num_local_microbatches: int = 1,
        accum_stream: "torch.cuda.Stream | None" = None,
    ) -> None:
        if world_size <= 0:
            raise ValueError(
                f"BucketedGradReducer: world_size must be positive, "
                f"got {world_size}"
            )
        if not 0 <= rank < world_size:
            raise ValueError(
                f"BucketedGradReducer: rank {rank} out of range "
                f"[0, {world_size})"
            )
        if num_local_microbatches < 1:
            raise ValueError(
                f"BucketedGradReducer: num_local_microbatches must be >= 1, "
                f"got {num_local_microbatches}"
            )

        self._world_size = world_size
        self._rank = rank
        self._device = device
        self._num_mb = int(num_local_microbatches)
        # P2-B grad-accum overlap: optional side stream where the
        # per-microbatch ``copy_``/``add_`` runs. ``None`` keeps the
        # P2-A behaviour (all accumulation on the default stream).
        self._accum_stream = accum_stream
        # Event recorded on ``_accum_stream`` after the *final* MB's
        # bucket fill, so the comm-stream reduce_scatter (and any
        # default-stream non-dist fallback) can wait for the side
        # stream to finish writing the bucket before reading it.
        self._bucket_accum_done_events: list[
            "torch.cuda.Event | None"
        ] = []
        self._buckets: list[BucketSpec] = compute_grad_buckets(
            buf_entries, world_size, bucket_target_elems=bucket_target_elems,
        )

        self._input_bufs: list[torch.Tensor] = []
        self._shard_bufs: list[torch.Tensor] = []
        for bucket in self._buckets:
            input_buf = torch.zeros(
                bucket.size, dtype=torch.float32, device=device,
            )
            shard_size = bucket.size // world_size
            shard_buf = torch.empty(
                shard_size, dtype=torch.float32, device=device,
            )
            self._input_bufs.append(input_buf)
            self._shard_bufs.append(shard_buf)

        self._param_loc: dict[str, tuple[int, int, int]] = {}
        for bucket_idx, bucket in enumerate(self._buckets):
            for (name, numel), offset in zip(bucket.params, bucket.param_offsets):
                self._param_loc[name] = (bucket_idx, offset, numel)

        self._bucket_pending: list[int] = [
            len(b.params) for b in self._buckets
        ]
        self._bucket_done_events: list[torch.cuda.Event | None] = [
            None for _ in self._buckets
        ]
        # M2-P15 batch flush: deferred grad references per bucket.
        # Instead of copying each grad immediately in write_grad, we
        # store a (offset, numel, grad_tensor) reference and flush all
        # of a bucket's deferred grads in a single burst when the
        # bucket fills.  This reduces host-side overhead from ~147
        # Python→CUDA dispatches interleaved with backward GEMMs to
        # ~5 burst-flushes (one per bucket), keeping the GPU busier.
        self._deferred: list[list[tuple[int, int, torch.Tensor]]] = [
            [] for _ in self._buckets
        ]
        # P2-A grad-accum overlap: per-bucket remaining-microbatch
        # counter and "first arrival" flag. When ``num_local_microbatches
        # == 1`` these collapse to the legacy behaviour (copy_, then
        # immediately issue reduce_scatter). When N > 1 the first MB's
        # fill of a bucket uses ``copy_`` and every subsequent MB uses
        # ``add_``; the reduce_scatter is deferred until the Nth MB
        # completes the bucket. This makes wgrad-allreduce overlap and
        # ZeRO-1 sharded optimizer legal in the grad_accum > 1 regime.
        self._bucket_mb_remaining: list[int] = [
            self._num_mb for _ in self._buckets
        ]
        self._bucket_is_first_mb: list[bool] = [
            True for _ in self._buckets
        ]
        self._bucket_accum_done_events = [None for _ in self._buckets]

    @property
    def buckets(self) -> list[BucketSpec]:
        """Read-only view of the bucket layout (for inspection / tests)."""
        return list(self._buckets)

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def num_buckets(self) -> int:
        return len(self._buckets)

    def start_step(self) -> None:
        """Reset per-step counters / events. Buffer values are not zeroed.

        Padding slots stay zero across steps because they are never
        written to. Per-param slots in ``[0, raw_size)`` are overwritten
        by the next step's ``write_grad`` calls before the bucket can
        complete (the pending counter starts at ``len(params)`` and only
        triggers reduce_scatter when it hits zero).

        When ``num_local_microbatches > 1`` the per-bucket
        ``mb_remaining`` counter is also reset to ``N`` and the
        ``is_first_mb`` flag is set so the first arrival of every
        bucket performs ``copy_`` (overwriting last step's data)
        rather than ``add_`` (which would mix step boundaries).
        """
        for i in range(len(self._buckets)):
            self._bucket_pending[i] = len(self._buckets[i].params)
            self._bucket_done_events[i] = None
            self._deferred[i].clear()
            self._bucket_mb_remaining[i] = self._num_mb
            self._bucket_is_first_mb[i] = True
            self._bucket_accum_done_events[i] = None

    # ------------------------------------------------------------------
    # Direction-C path-1 (step CUDA Graph + reducer coexistence)
    #
    # The trio below — ``make_device_grad_sink`` / ``zero_input_bufs`` /
    # ``flush_all_buckets`` — bypasses the ``write_grad`` host-state
    # machine entirely. Backward (running inside a captured CUDA Graph)
    # calls the device sink once per parameter; the sink performs a
    # single in-place ``add_`` into the bucket's preallocated input
    # buffer. No host conditional, no Python branch, no per-bucket
    # pending counter — all three would be impossible to capture.
    # The trade-off:
    #   - ``zero_input_bufs`` must run on the host between steps so
    #     ``add_`` semantics work on the first MB (~0.5 ms HBM cost).
    #   - ``flush_all_buckets`` issues all RS sequentially after the
    #     final MB, losing the per-bucket "fire as it fills" overlap;
    #     graph capture would freeze the schedule anyway so this is
    #     a wash on average.
    # The two paths (eager wgrad-overlap vs step-graph + reducer) are
    # mutually exclusive at construction time — controlled by the
    # ``STEP_CUDA_GRAPH_REDUCER`` env in the train loop.
    # ------------------------------------------------------------------

    def make_device_grad_sink(self) -> Callable[[str, torch.Tensor], None]:
        """Return a capture-safe ``grad_sink`` callable for backward.

        The returned closure performs the following CUDA ops only:

        * ``g.view(-1)`` — view, not a kernel.
        * ``flat.float()`` — cast (one kernel) when grad isn't already
          FP32.
        * ``input_buf.narrow(...).add_(flat)`` — single in-place add
          on the default stream.

        Python lookups (``param_loc[name]``, attribute access) execute
        at capture time and resolve to constants by replay. The
        callable is therefore safe to embed inside
        ``torch.cuda.graph`` capture.

        Caller contract:

        * ``zero_input_bufs()`` must be called once per step BEFORE
          the first MB's replay, so the buffer starts at zero and
          ``add_`` accumulates correctly.
        * ``flush_all_buckets()`` must be called AFTER the final MB's
          replay to schedule the reduce_scatters.
        * ``write_grad`` must NOT be invoked while this sink path is
          live; the two state machines are mutually exclusive.
        """
        param_loc = self._param_loc
        input_bufs = self._input_bufs

        def device_grad_sink(name: str, g: torch.Tensor) -> None:
            bucket_idx, offset, numel = param_loc[name]
            flat = g.view(-1)
            if flat.dtype != torch.float32:
                flat = flat.float()
            input_bufs[bucket_idx].narrow(0, offset, numel).add_(flat)

        return device_grad_sink

    def zero_input_bufs(self) -> None:
        """Host-side zero of every bucket's input buffer.

        Used by the step_graph + reducer coexistence path:
        ``device_grad_sink`` uses ``add_`` for all MBs (not the
        ``copy_``-then-``add_`` mix of ``write_grad``), so each step
        must start with zeroed buffers. Padding tail stays zero
        across steps because backward never writes to it.
        """
        for buf in self._input_bufs:
            buf.zero_()

    def flush_all_buckets(self) -> None:
        """Issue ``reduce_scatter`` on every bucket.

        Used after the final MB's graph replay completes, when all
        gradients have been baked into ``_input_bufs`` by the device
        sink but no bucket-pending counter has fired (the device sink
        bypasses ``write_grad``). The buckets are issued sequentially
        on the comm stream; ``wait_all`` then blocks the default
        stream on every per-bucket done event before downstream
        ops (grad_norm / allgather / optimizer) consume the shards.

        Updates ``_bucket_pending`` to 0 so the standard ``wait_all``
        invariants hold after the call.
        """
        for bucket_idx in range(len(self._buckets)):
            self._issue_bucket_reduce(bucket_idx)
            self._bucket_pending[bucket_idx] = 0

    def write_grad(self, name: str, grad: torch.Tensor) -> None:
        """Register ``grad`` for deferred copy; trigger reduce_scatter when full.

        The grad tensor reference is stored in a per-bucket deferred
        list. When the bucket's last param arrives (pending counter
        hits zero), all deferred grads are flushed into the bucket
        buffer in a single burst. This batch-flush strategy replaces
        the former per-param immediate copy, cutting host-side
        Python→CUDA dispatch overhead from ~N_params individual copies
        to ~N_buckets burst-flushes (see ``perf_log.md`` Round M2-P15).

        Single-microbatch (``num_local_microbatches == 1``)
            On bucket fill the deferred grads are ``copy_``-ed into the
            FP32 input buffer and ``reduce_scatter`` is dispatched on
            the comm stream — bytewise identical to the M2-P15 path.

        Multi-microbatch (``num_local_microbatches == N > 1``)
            On the first MB's bucket fill: ``copy_`` (overwriting last
            step's data, padding stays zero). On MB 2..N-1: ``add_``.
            On MB N: ``add_`` then dispatch ``reduce_scatter``. After
            each non-final MB the per-bucket pending counter is reset
            to ``len(params)`` so the next MB's writes are accepted.

        The grad tensor is up-cast to FP32 during the flush (matches
        the FP32 AllReduce buffer rule). The deferred reference is
        safe to hold because: (a) the grad is a freshly allocated
        cuBLAS / TE output, not a view into a shared buffer; (b) we
        hold a Python reference so the allocator cannot reclaim it;
        (c) the flush runs on the same default stream as the backward
        GEMMs, so ordering is correct.
        """
        if name not in self._param_loc:
            raise KeyError(
                f"BucketedGradReducer.write_grad: unknown param {name!r}"
            )
        bucket_idx, offset, numel = self._param_loc[name]
        if grad.numel() != numel:
            raise ValueError(
                f"BucketedGradReducer.write_grad: param {name!r} expected "
                f"numel {numel}, got {grad.numel()}"
            )
        if self._bucket_pending[bucket_idx] <= 0:
            raise RuntimeError(
                f"BucketedGradReducer.write_grad: bucket {bucket_idx} already "
                f"complete — duplicate write for {name!r} (did you forget "
                f"to call start_step()?)"
            )

        self._deferred[bucket_idx].append((offset, numel, grad))

        self._bucket_pending[bucket_idx] -= 1
        if self._bucket_pending[bucket_idx] == 0:
            is_first_mb = self._bucket_is_first_mb[bucket_idx]
            self._bucket_mb_remaining[bucket_idx] -= 1
            is_final_mb = self._bucket_mb_remaining[bucket_idx] == 0
            self._flush_for_microbatch(
                bucket_idx,
                is_first_mb=is_first_mb,
                is_final_mb=is_final_mb,
            )
            self._bucket_is_first_mb[bucket_idx] = False
            if not is_final_mb:
                self._bucket_pending[bucket_idx] = len(
                    self._buckets[bucket_idx].params
                )

    def _flush_for_microbatch(
        self,
        bucket_idx: int,
        *,
        is_first_mb: bool,
        is_final_mb: bool,
    ) -> None:
        """Flush deferred grads into the bucket; reduce only on final MB.

        Two execution profiles share the same arithmetic:

        Default stream (``self._accum_stream is None``)
            Bunched copies/adds at bucket-fill time on the default
            stream (same ordering guarantee as the M2-P15 single-MB
            path and the P2-A multi-MB path). ``copy_`` on the first
            MB (overwrites last step's residue; padding stays zero),
            ``add_`` on subsequent MBs.

        P2-B accum stream (``self._accum_stream is not None``)
            Bunched copies/adds bake on a dedicated side stream.
            Before any work, ``accum_stream`` waits an event recorded
            on the default stream so the wgrad tensors are visible
            (without this, the side-stream ``copy_`` could observe
            half-written grads). We also call ``record_stream`` on the
            held grad tensors so the caching allocator does not
            reclaim them while the side-stream add is in flight. The
            default stream is therefore free to start the next MB's
            forward immediately after backward — the accumulation is
            entirely off the critical path.

        ``reduce_scatter`` is only dispatched on the final MB — for
        single-MB callers (N=1) the first-and-final MB collapse to
        the legacy ``copy_ + issue_bucket_reduce`` sequence.
        """
        accum_stream = self._accum_stream
        deferred = self._deferred[bucket_idx]
        input_buf = self._input_bufs[bucket_idx]

        if accum_stream is None:
            if is_first_mb:
                for offset, numel, grad in deferred:
                    flat = grad.reshape(-1)
                    if flat.dtype != torch.float32:
                        flat = flat.float()
                    input_buf[offset:offset + numel].copy_(flat)
            else:
                for offset, numel, grad in deferred:
                    flat = grad.reshape(-1)
                    if flat.dtype != torch.float32:
                        flat = flat.float()
                    input_buf[offset:offset + numel].add_(flat)
            deferred.clear()
            if is_final_mb:
                self._issue_bucket_reduce(bucket_idx)
            return

        # P2-B side-stream path. The default stream must signal that
        # every wgrad in ``deferred`` is committed before the accum
        # stream may read it.
        grad_ready = torch.cuda.Event()
        grad_ready.record()
        accum_stream.wait_event(grad_ready)
        with torch.cuda.stream(accum_stream):
            if is_first_mb:
                for offset, numel, grad in deferred:
                    flat = grad.reshape(-1)
                    if flat.dtype != torch.float32:
                        flat = flat.float()
                    input_buf[offset:offset + numel].copy_(flat)
            else:
                for offset, numel, grad in deferred:
                    flat = grad.reshape(-1)
                    if flat.dtype != torch.float32:
                        flat = flat.float()
                    input_buf[offset:offset + numel].add_(flat)
        # Keep the grad tensors alive on accum_stream so the allocator
        # cannot recycle them until the side-stream add completes.
        # ``record_stream`` is a no-op on CPU tensors / mock objects,
        # making the call safe for unit tests too.
        for _, _, grad in deferred:
            rs = getattr(grad, "record_stream", None)
            if rs is not None:
                rs(accum_stream)
        deferred.clear()

        if is_final_mb:
            accum_done = torch.cuda.Event()
            accum_done.record(accum_stream)
            self._bucket_accum_done_events[bucket_idx] = accum_done
            self._issue_bucket_reduce(bucket_idx)

    def _issue_bucket_reduce(self, bucket_idx: int) -> None:
        """Schedule the bucket's reduce_scatter on the comm stream.

        When ``self._accum_stream`` is active the accumulation work
        lived on a side stream; ``ready_event`` is taken from the
        ``accum_done`` event stashed by ``_flush_for_microbatch`` so
        the comm stream waits for the side-stream ``add_`` to finish
        before reading the input buffer.
        """
        accum_done = self._bucket_accum_done_events[bucket_idx]
        if not is_distributed():
            # CPU / single-rank fallback: serialise so the test reads
            # the post-add buffer. ``Event.synchronize()`` is safe on
            # CPU (the event recorded on the accum_stream is already
            # marked complete because the side-stream ops were issued
            # synchronously in the calling thread under the unit-test
            # mock; on real CUDA it actually waits).
            if accum_done is not None:
                accum_done.synchronize()
            self._input_bufs[bucket_idx].mul_(1.0 / self._world_size)
            shard_size = self._buckets[bucket_idx].size // self._world_size
            shard_view = self._input_bufs[bucket_idx][
                self._rank * shard_size:(self._rank + 1) * shard_size
            ]
            self._shard_bufs[bucket_idx].copy_(shard_view)
            return

        if accum_done is not None:
            ready_event = accum_done
        else:
            ready_event = torch.cuda.Event()
            ready_event.record()

        comm_stream = get_comm_stream(self._device)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready_event)
            input_buf = self._input_bufs[bucket_idx]
            shard_buf = self._shard_bufs[bucket_idx]
            input_buf.mul_(1.0 / self._world_size)
            dist.reduce_scatter_tensor(
                shard_buf, input_buf, op=dist.ReduceOp.SUM,
            )
            done_event = torch.cuda.Event()
            done_event.record()
            self._bucket_done_events[bucket_idx] = done_event

    def wait_all(self) -> None:
        """Synchronise the default stream with all in-flight bucket reduces.

        Inserts a per-bucket ``wait_event`` so subsequent default-stream
        ops see the reduced shards. No-op if not distributed (in that
        case ``_issue_bucket_reduce`` already wrote the local shard
        synchronously).
        """
        if not is_distributed():
            return
        default_stream = torch.cuda.current_stream(self._device)
        for ev in self._bucket_done_events:
            if ev is not None:
                default_stream.wait_event(ev)

    def allgather_and_unpack(
        self,
        params: dict[str, torch.Tensor],
        *,
        clone: bool = True,
    ) -> dict[str, torch.Tensor]:
        """All_gather each bucket's shard back into a full bucket and unpack.

        The all_gather output is written **in place** over the bucket's
        input buffer (which is no longer needed after reduce_scatter).
        Per-param FP32 grad views are then sliced + reshaped into a
        fresh dict with the original parameter shapes.

        When ``clone=True`` (default), each view is cloned so the
        consumer can mutate gradients without aliasing the reducer's
        persistent buffer. This is required by the legacy optimizer
        path (``clip_gradients_fp32`` mutates grads in-place).

        When ``clone=False``, the returned tensors are non-owning
        views into the bucket's ``input_buf``. This is safe when the
        consumer only reads the gradients (e.g. the fused Adam+sync
        Triton kernel, which loads from ``grad_ptr`` but never stores
        to it). The views remain valid until the next step's
        ``write_grad`` overwrites the buffer, which is strictly after
        the ``torch.cuda.synchronize()`` at step end — so no race.
        Skipping 147 clone operations saves ~2 GB of transient
        allocations and the associated memcpy bandwidth.
        """
        out: dict[str, torch.Tensor] = {}
        for i, bucket in enumerate(self._buckets):
            input_buf = self._input_bufs[i]
            shard_buf = self._shard_bufs[i]
            if is_distributed():
                dist.all_gather_into_tensor(input_buf, shard_buf.contiguous())
            for (name, numel), offset in zip(bucket.params, bucket.param_offsets):
                shape = params[name].shape
                view = input_buf[offset:offset + numel].view(shape)
                out[name] = view.clone() if clone else view
        return out

    # ------------------------------------------------------------------
    # M2-P6 P1.c: per-bucket allgather + optimizer pipeline
    # ------------------------------------------------------------------
    #
    # The legacy ``allgather_and_unpack`` issues every bucket's
    # ``all_gather_into_tensor`` back-to-back on the default stream and
    # then returns a dict for the whole-model optimizer step — so
    # allgather (~9.10 ms wall on the M2-P5 100M-bucket profile) and the
    # optimizer (~5.60 ms wall) run strictly serially. The methods below
    # split that into a producer/consumer pair so the comm stream can
    # keep issuing the next bucket's ``all_gather_into_tensor`` while
    # the default stream runs ``fused_adam_sync`` for the previous
    # bucket's params, hiding the comm wall-clock under the optimizer.
    #
    # Stream contract (matches the ``_issue_bucket_reduce`` invariants
    # so reduce_scatter and allgather share the same NCCL ordering on
    # ``comm_stream``):
    #
    # * ``start_bucket_allgather`` issues the per-bucket allgather on
    #   ``comm_stream`` and records a completion event. The shard buffer
    #   read inside the call is naturally ordered after the prior
    #   ``reduce_scatter`` because both run on ``comm_stream``.
    # * ``wait_bucket_allgather`` inserts a ``wait_event`` so the
    #   default stream (or the supplied consumer stream) blocks until
    #   the bucket's input buffer is fully populated before reading it.
    # * ``unpack_bucket`` returns FP32 views into the bucket's
    #   ``input_buf`` (NO clone — the optimizer reads grads but does
    #   not mutate them, and the views stay valid until the next step's
    #   ``write_grad`` overwrites the slots, which is strictly after
    #   the consumer's ``fused_adam_sync`` returns thanks to the
    #   per-step ``torch.cuda.synchronize`` in the train loop).
    #
    # The non-distributed fallback path is intentionally identical to
    # the legacy ``allgather_and_unpack`` for buckets — the input buffer
    # already holds the full averaged data after ``_issue_bucket_reduce``
    # so allgather is a no-op and ``unpack_bucket`` reads it directly.

    def bucket_param_names(self, bucket_idx: int) -> tuple[str, ...]:
        """Return the param names assigned to a single bucket, in
        baseline-buffer order (the same order the reducer copies grads
        into its input buffer and the same order the optimizer must
        consume them in).
        """
        if not 0 <= bucket_idx < len(self._buckets):
            raise IndexError(
                f"BucketedGradReducer.bucket_param_names: bucket_idx "
                f"{bucket_idx} out of range [0, {len(self._buckets)})"
            )
        return tuple(name for name, _ in self._buckets[bucket_idx].params)

    def start_bucket_allgather(self, bucket_idx: int) -> None:
        """Issue the per-bucket ``all_gather_into_tensor`` on the comm
        stream and record a completion event.

        Must be called AFTER ``wait_all`` (so the prior step's
        ``reduce_scatter`` is guaranteed to have populated
        ``shard_buf``) and before any consumer reads ``input_buf`` for
        this bucket. Re-records the bucket's done event so a subsequent
        ``wait_bucket_allgather`` synchronises against the allgather
        rather than the earlier reduce_scatter.

        Non-dist fallback: no-op — ``input_buf`` already contains the
        full averaged data from ``_issue_bucket_reduce``, and the
        completion event slot is cleared so wait is also a no-op.
        """
        if not 0 <= bucket_idx < len(self._buckets):
            raise IndexError(
                f"BucketedGradReducer.start_bucket_allgather: bucket_idx "
                f"{bucket_idx} out of range [0, {len(self._buckets)})"
            )
        if not is_distributed():
            self._bucket_done_events[bucket_idx] = None
            return

        comm_stream = get_comm_stream(self._device)
        with torch.cuda.stream(comm_stream):
            input_buf = self._input_bufs[bucket_idx]
            shard_buf = self._shard_bufs[bucket_idx]
            dist.all_gather_into_tensor(input_buf, shard_buf.contiguous())
            done_event = torch.cuda.Event()
            done_event.record()
            self._bucket_done_events[bucket_idx] = done_event

    def wait_bucket_allgather(
        self,
        bucket_idx: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Block ``stream`` (default: current default stream) until the
        bucket's pending allgather completes.

        Inserts a ``wait_event`` rather than a host sync so the consumer
        stream's pending ops can keep being issued while the wait is
        only enforced at the moment the comm completes. Idempotent /
        no-op when the bucket has no pending event (non-dist fallback,
        or the prior ``start_bucket_allgather`` was skipped).
        """
        if not 0 <= bucket_idx < len(self._buckets):
            raise IndexError(
                f"BucketedGradReducer.wait_bucket_allgather: bucket_idx "
                f"{bucket_idx} out of range [0, {len(self._buckets)})"
            )
        ev = self._bucket_done_events[bucket_idx]
        if ev is None:
            return
        consumer = stream if stream is not None else torch.cuda.current_stream(
            self._device,
        )
        consumer.wait_event(ev)

    def unpack_bucket(
        self,
        bucket_idx: int,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return per-param FP32 grad views for a single bucket.

        Unlike :meth:`allgather_and_unpack`, this does NOT clone the
        per-param view — the returned tensors alias the bucket's
        ``input_buf`` directly. The caller MUST consume them before the
        next step's ``write_grad`` overwrites the slots (which is
        guaranteed by the train loop's per-step
        ``torch.cuda.synchronize`` boundary plus the contract that the
        optimizer only reads grads, never mutates them).

        Combined with ``start_bucket_allgather`` /
        ``wait_bucket_allgather`` this lets the optimizer step process
        bucket ``i`` while bucket ``i+1``'s allgather is still in
        flight on the comm stream — the M2-P6 P1.c overlap design.
        """
        if not 0 <= bucket_idx < len(self._buckets):
            raise IndexError(
                f"BucketedGradReducer.unpack_bucket: bucket_idx "
                f"{bucket_idx} out of range [0, {len(self._buckets)})"
            )
        bucket = self._buckets[bucket_idx]
        input_buf = self._input_bufs[bucket_idx]
        out: dict[str, torch.Tensor] = {}
        for (name, numel), offset in zip(bucket.params, bucket.param_offsets):
            shape = params[name].shape
            out[name] = input_buf[offset:offset + numel].view(shape)
        return out

    def get_grad_shard(self, bucket_idx: int) -> torch.Tensor:
        """Return the FP32 gradient shard for a given bucket.

        After ``wait_all()``, this shard contains the reduce-scattered
        averaged gradients for this rank's portion of the bucket. The
        returned tensor is a reference to the reducer's persistent
        buffer — callers must not modify it (the sharded optimizer
        reads from it but never writes).

        Used by :func:`optimizer.sharded_fused_clip_adam_sync` to run
        the per-shard Adam update without allgathering the full FP32
        gradients first.
        """
        if not 0 <= bucket_idx < len(self._buckets):
            raise IndexError(
                f"BucketedGradReducer.get_grad_shard: bucket_idx "
                f"{bucket_idx} out of range [0, {len(self._buckets)})"
            )
        return self._shard_bufs[bucket_idx]

    def allgather_bf16_and_unpack(
        self,
        bf16_shards: list[torch.Tensor],
        bf16_full_buf: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> None:
        """All_gather BF16 shards and unpack into per-param BF16 tensors.

        Each rank contributes its BF16 shard for every bucket; the
        all_gather output is written into ``bf16_full_buf`` (reused
        across buckets) and immediately unpacked into ``params``.

        This replaces the FP32 ``allgather_and_unpack`` in the sharded
        optimizer path: since the optimizer has already produced the
        BF16 output per-shard, allgathering BF16 instead of FP32 halves
        the data volume.

        Args:
            bf16_shards: Per-bucket BF16 shard tensors (one per bucket,
                each of length ``bucket.size // world_size``).
            bf16_full_buf: Reusable BF16 buffer of length
                ``max(bucket.size for bucket in buckets)``.
            params: BF16 parameter dict to update in-place.
        """
        for i, bucket in enumerate(self._buckets):
            bf16_shard = bf16_shards[i]
            out = bf16_full_buf[:bucket.size]
            if is_distributed():
                dist.all_gather_into_tensor(out, bf16_shard.contiguous())
            else:
                out[:bf16_shard.numel()] = bf16_shard
            for (name, numel), offset in zip(bucket.params, bucket.param_offsets):
                shape = params[name].shape
                params[name].copy_(out[offset:offset + numel].view(shape))

    def allgather_optimizer_state(self, state: Any) -> None:
        """Allgather sharded optimizer state so all ranks hold the full FP32 copy.

        After :func:`sharded_fused_clip_adam_sync`, each rank only has
        up-to-date values for its 1/world_size shard of
        ``state.master_weights``, ``state.exp_avg``, and
        ``state.exp_avg_sq``. The remaining elements are stale (from
        the previous step or from initialisation).

        This method allgathers each of the three FP32 state dicts
        bucket-by-bucket so every rank ends up with the complete,
        current optimizer state. Must be called **before**
        ``save_checkpoint`` when ``SHARDED_OPTIMIZER=1``.

        Reuses the reducer's persistent ``_shard_bufs`` and
        ``_input_bufs`` for packing / allgather — these are safe to
        overwrite at save-time because the gradient data they held
        during the step is no longer needed.

        Non-distributed fallback: early return (single rank owns
        everything, so the state is already complete).
        """
        if not is_distributed():
            return
        for field in ("master_weights", "exp_avg", "exp_avg_sq"):
            self._allgather_state_field(getattr(state, field))

    def _allgather_state_field(
        self, state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Allgather one FP32 optimizer-state field across DP ranks."""
        for bucket_idx, bucket in enumerate(self._buckets):
            shard_size = bucket.size // self._world_size
            rank_start = self._rank * shard_size

            shard_buf = self._shard_bufs[bucket_idx]
            shard_buf.zero_()
            for (name, numel), offset in zip(
                bucket.params, bucket.param_offsets,
            ):
                ov_start = max(offset, rank_start)
                ov_end = min(offset + numel, rank_start + shard_size)
                if ov_start >= ov_end:
                    continue
                local_s = ov_start - rank_start
                local_e = ov_end - rank_start
                param_s = ov_start - offset
                param_e = ov_end - offset
                shard_buf[local_s:local_e].copy_(
                    state_dict[name].view(-1)[param_s:param_e]
                )

            input_buf = self._input_bufs[bucket_idx]
            dist.all_gather_into_tensor(input_buf, shard_buf.contiguous())

            for (name, numel), offset in zip(
                bucket.params, bucket.param_offsets,
            ):
                state_dict[name].view(-1)[:numel].copy_(
                    input_buf[offset:offset + numel]
                )

    def compute_distributed_grad_norm(
        self,
        device: str | torch.device,
    ) -> float:
        """L2 grad norm from per-bucket shards, all_reduced across DP.

        Walks each bucket's local shard (the rank's ``shard_buf``),
        finds the slice each param contributes, classifies as weight vs
        norm grad (matching the legacy ordering convention), and accumulates
        a single squared L2 norm via ``multi_tensor_l2norm``. The squared
        norm is then ``all_reduce``-summed across DP ranks and the
        square root is returned — same numeric contract as the legacy
        :func:`compute_distributed_grad_norm`.

        Padding slots at the tail of each bucket are zero (initialised
        once, never written) so they contribute zero to the norm even
        when included in the local shard.
        """
        from transformer_engine.pytorch.optimizers import (
            multi_tensor_applier,
            multi_tensor_l2norm,
        )

        weight_grads: list[torch.Tensor] = []
        norm_grads: list[torch.Tensor] = []

        for bucket_idx, bucket in enumerate(self._buckets):
            shard_size = bucket.size // self._world_size
            rank_start = self._rank * shard_size
            rank_end = rank_start + shard_size
            shard_buf = self._shard_bufs[bucket_idx]

            for (name, numel), offset in zip(bucket.params, bucket.param_offsets):
                p_start = offset
                p_end = offset + numel
                ov_start = max(p_start, rank_start)
                ov_end = min(p_end, rank_end)
                if ov_start < ov_end:
                    local_s = ov_start - rank_start
                    local_e = ov_end - rank_start
                    g = shard_buf[local_s:local_e]
                    if _is_norm_param(name):
                        norm_grads.append(g)
                    else:
                        weight_grads.append(g)

        shard_grads = weight_grads + norm_grads

        dummy = torch.zeros(1, dtype=torch.int, device=device)
        if shard_grads:
            local_norm, _ = multi_tensor_applier(
                multi_tensor_l2norm, dummy, [shard_grads], False,
            )
        else:
            local_norm = torch.zeros(1, dtype=torch.float32, device=device)

        total_norm_sq = local_norm ** 2
        if is_distributed():
            dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
        return total_norm_sq.item() ** 0.5


_register_reset_hook(reset_comm_stream)
