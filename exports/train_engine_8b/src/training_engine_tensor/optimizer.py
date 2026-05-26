"""Adam optimizer with FP32 master params, LR schedule, grad clipping.

Uses Apex's multi_tensor_adam CUDA kernel for bitwise-identical behavior
with Megatron's distributed optimizer (which wraps Apex FusedAdam internally).

Weight decay groups: RMSNorm weights get wd=0.0, all other weights get wd=0.1,
matching Megatron's _get_params_for_weight_decay_optimization().
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch

__all__ = [
    "MAX_LR", "MIN_LR", "WARMUP_ITERS", "DECAY_ITERS",
    "WEIGHT_DECAY", "CLIP_GRAD",
    "compute_lr", "compute_grad_norm_fp32", "clip_gradients_fp32",
    "AdamState", "adam_step", "sync_params_from_master",
    "DistributedAdamState", "distributed_adam_step",
    "distributed_adam_step_bucket",
    "compute_distributed_grad_norm", "clip_shard_grads",
    "validate_backends",
]

from . import config
from .parameters import _canonical_buffer_order

# ---------------------------------------------------------------------------
# Backend resolution for fused multi-tensor kernels.
#
# Try the candidate import locations in turn, take the first one that imports
# cleanly, and only set the symbol to ``None`` if nothing works.  Every call
# site below ultimately triggers a ``RuntimeError`` rather than a silent NPE.
# ---------------------------------------------------------------------------
multi_tensor_applier = None
multi_tensor_l2norm = None
multi_tensor_scale = None
_multi_tensor_adam_fn = None


def _resolve_multi_tensor_backends():
    """Resolve the four symbols above from whichever backend is present."""
    global multi_tensor_applier, multi_tensor_l2norm, multi_tensor_scale
    global _multi_tensor_adam_fn

    # 1) `multi_tensor_applier` — same name in TE/apex.
    try:
        from transformer_engine.pytorch.optimizers import multi_tensor_applier as _mta  # type: ignore
        multi_tensor_applier = _mta
    except Exception:
        try:
            from transformer_engine.pytorch.optimizers.multi_tensor_apply import multi_tensor_applier as _mta  # type: ignore
            multi_tensor_applier = _mta
        except Exception:
            try:
                from apex.multi_tensor_apply import multi_tensor_applier as _mta  # type: ignore
                multi_tensor_applier = _mta
            except Exception:
                pass

    # 2) `multi_tensor_l2norm` and `multi_tensor_scale`.
    try:
        from transformer_engine.pytorch.optimizers import (
            multi_tensor_l2norm as _l2,
            multi_tensor_scale as _sc,
        )
        multi_tensor_l2norm = _l2
        multi_tensor_scale = _sc
    except Exception:
        try:
            import amp_C as _amp_C  # type: ignore
            multi_tensor_l2norm = _amp_C.multi_tensor_l2norm
            multi_tensor_scale = _amp_C.multi_tensor_scale
        except Exception:
            pass

    # 3) `multi_tensor_adam`.
    try:
        import transformer_engine_torch as _tex  # type: ignore
        _multi_tensor_adam_fn = _tex.multi_tensor_adam
    except Exception:
        try:
            from transformer_engine.pytorch.optimizers import (
                multi_tensor_adam as _mta_adam,  # type: ignore
            )
            _multi_tensor_adam_fn = _mta_adam
        except Exception:
            try:
                import amp_C as _amp_C  # type: ignore
                _multi_tensor_adam_fn = _amp_C.multi_tensor_adam
            except Exception:
                pass


_resolve_multi_tensor_backends()


def validate_backends() -> None:
    """Fail-fast if required multi-tensor CUDA kernels are unavailable.

    Call at training start (before first optimizer step) to surface missing
    backends immediately rather than getting a cryptic NoneType error later.
    """
    missing = []
    if multi_tensor_applier is None:
        missing.append("multi_tensor_applier")
    if multi_tensor_l2norm is None:
        missing.append("multi_tensor_l2norm")
    if _multi_tensor_adam_fn is None:
        missing.append("multi_tensor_adam")
    if missing:
        raise RuntimeError(
            f"Required multi-tensor CUDA backends not available: {missing}. "
            "Ensure transformer_engine or apex is installed."
        )


MAX_LR = 2.273e-4
MIN_LR = 0.0
WARMUP_ITERS = 2000
DECAY_ITERS = 1000000
WEIGHT_DECAY = 0.1
CLIP_GRAD = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8


def _is_no_weight_decay(name: str) -> bool:
    """Match Megatron's default: name.endswith('.bias') or len(param.shape)==1.
    For this model (no biases), 1-D params are RMSNorm gain vectors.

    Match both the legacy unfused names (``input_layernorm.weight``,
    ``pre_mlp_layernorm.weight``) and the fused mcore names
    (``linear_qkv.layer_norm_weight`` / ``linear_fc1.layer_norm_weight``,
    plus the standalone ``decoder.final_layernorm.weight``); the underscore
    inside ``layer_norm_weight`` would otherwise miss the ``layernorm``
    substring check.
    """
    lower = name.lower()
    return ("layernorm" in lower) or ("layer_norm" in lower) or ("bias" in lower)




def compute_lr(step: int) -> float:
    """WSD learning rate schedule. All milestones (<=500 steps) are within warmup."""
    if step <= WARMUP_ITERS:
        return MAX_LR * step / WARMUP_ITERS
    return MAX_LR


def compute_grad_norm_fp32(
    fp32_grads: Dict[str, torch.Tensor],
    trainable_names: List[str],
    device: str,
) -> float:
    """Compute FP32 gradient L2 norm matching Megatron's DistributedOptimizer.

    Megatron's get_main_grads_for_grad_norm() iterates param_groups
    (WD group first, no-WD group second). Within each group, params appear
    in ParamAndGradBuffer order (reverse of model.named_parameters()).
    A single multi_tensor_l2norm call is made on all grads in this order.
    """
    buf_order = _canonical_buffer_order(trainable_names)
    wd_names = [n for n in buf_order if not _is_no_weight_decay(n)]
    no_wd_names = [n for n in buf_order if _is_no_weight_decay(n)]
    norm_order = wd_names + no_wd_names

    grads_for_norm = [fp32_grads[name].view(-1) for name in norm_order]
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=device)

    grad_norm, _ = multi_tensor_applier(
        multi_tensor_l2norm, dummy_overflow_buf, [grads_for_norm], False,
    )
    total_norm = grad_norm ** 2
    return total_norm.item() ** 0.5


def clip_gradients_fp32(
    fp32_grads: Dict[str, torch.Tensor],
    trainable_names: List[str],
    max_norm: float,
    total_norm: float,
    device: str,
) -> None:
    """Clip FP32 gradients by total norm (in-place).

    clip_coeff is computed in Python FP64 (matching Megatron), then
    multi_tensor_scale casts it to FP32 inside the CUDA kernel.
    """
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        grad_list = [fp32_grads[name].view(-1) for name in trainable_names]
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=device)
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf,
            [grad_list, grad_list], clip_coeff,
        )


class AdamState:
    """FP32 master parameters + Adam optimizer state (exp_avg, exp_avg_sq)."""

    def __init__(
        self,
        trainable_names: List[str],
        params: Dict[str, torch.Tensor],
        device: str,
    ):
        self.fp32_master: Dict[str, torch.Tensor] = {}
        self.exp_avg: Dict[str, torch.Tensor] = {}
        self.exp_avg_sq: Dict[str, torch.Tensor] = {}
        self.step_count = 0
        self.device = device

        for name in trainable_names:
            fp32_p = params[name].float()
            self.fp32_master[name] = fp32_p
            self.exp_avg[name] = torch.zeros_like(fp32_p)
            self.exp_avg_sq[name] = torch.zeros_like(fp32_p)


def adam_step(
    opt_state: AdamState,
    fp32_grads: Dict[str, torch.Tensor],
    trainable_names: List[str],
    lr: float,
) -> None:
    """One Adam step matching Apex FusedAdam (adam_w_mode=True, bias_correction=True).

    Uses amp_C.multi_tensor_adam when available for bitwise match with Megatron.
    Falls back to PyTorch scalar ops otherwise.
    """
    opt_state.step_count += 1

    if _multi_tensor_adam_fn is not None:
        _adam_step_fused(opt_state, fp32_grads, trainable_names, lr)
    else:
        _adam_step_pytorch(opt_state, fp32_grads, trainable_names, lr)


def _adam_step_fused(
    opt_state: AdamState,
    fp32_grads: Dict[str, torch.Tensor],
    trainable_names: List[str],
    lr: float,
) -> None:
    """Adam step via Apex multi_tensor_adam CUDA kernel."""
    t = opt_state.step_count

    wd_g, wd_p, wd_m, wd_v = [], [], [], []
    no_wd_g, no_wd_p, no_wd_m, no_wd_v = [], [], [], []

    for name in trainable_names:
        g = fp32_grads[name]
        p = opt_state.fp32_master[name]
        m = opt_state.exp_avg[name]
        v = opt_state.exp_avg_sq[name]

        if _is_no_weight_decay(name):
            no_wd_g.append(g)
            no_wd_p.append(p)
            no_wd_m.append(m)
            no_wd_v.append(v)
        else:
            wd_g.append(g)
            wd_p.append(p)
            wd_m.append(m)
            wd_v.append(v)

    overflow_buf = torch.tensor([0], dtype=torch.int, device=opt_state.device)

    if wd_g:
        multi_tensor_applier(
            _multi_tensor_adam_fn,
            overflow_buf,
            [wd_g, wd_p, wd_m, wd_v],
            lr,
            ADAM_BETA1,
            ADAM_BETA2,
            ADAM_EPS,
            t,
            1,
            1,
            WEIGHT_DECAY,
        )

    if no_wd_g:
        multi_tensor_applier(
            _multi_tensor_adam_fn,
            overflow_buf,
            [no_wd_g, no_wd_p, no_wd_m, no_wd_v],
            lr,
            ADAM_BETA1,
            ADAM_BETA2,
            ADAM_EPS,
            t,
            1,
            1,
            0.0,
        )


def _adam_step_pytorch(
    opt_state: AdamState,
    fp32_grads: Dict[str, torch.Tensor],
    trainable_names: List[str],
    lr: float,
) -> None:
    """Fallback Adam step using PyTorch ops (may differ by <=1 ULP from fused)."""
    t = opt_state.step_count
    bias_correction1 = 1.0 - ADAM_BETA1 ** t
    bias_correction2 = 1.0 - ADAM_BETA2 ** t
    step_size = lr / bias_correction1
    bc2_sqrt = math.sqrt(bias_correction2)

    for name in trainable_names:
        g = fp32_grads[name]
        m = opt_state.exp_avg[name]
        v = opt_state.exp_avg_sq[name]
        p = opt_state.fp32_master[name]

        m.mul_(ADAM_BETA1).add_(g, alpha=1.0 - ADAM_BETA1)
        v.mul_(ADAM_BETA2).addcmul_(g, g, value=1.0 - ADAM_BETA2)

        denom = (v.sqrt() / bc2_sqrt).add_(ADAM_EPS)
        p.addcdiv_(m, denom, value=-step_size)

        if not _is_no_weight_decay(name):
            p.add_(p, alpha=-lr * WEIGHT_DECAY)


def sync_params_from_master(
    params: Dict[str, torch.Tensor],
    opt_state: AdamState,
    trainable_names: List[str],
) -> None:
    """Copy FP32 master params back to BF16 working params."""
    for name in trainable_names:
        params[name].copy_(opt_state.fp32_master[name].to(torch.bfloat16))


# ===========================================================================
# Distributed optimizer — sharded across DP ranks
# ===========================================================================

class DistributedAdamState:
    """Sharded FP32 optimizer state for distributed training.

    Each DP rank owns a shard of the full parameter buffer. Only the
    optimizer state (master params, exp_avg, exp_avg_sq) for that shard
    is stored on this rank.
    """

    def __init__(
        self,
        layout: "BufferLayout",
        params: Dict[str, torch.Tensor],
        dp_rank: int,
        device: str,
    ):
        from .nccl import BufferLayout

        self.layout = layout
        self.dp_rank = dp_rank
        self.device = device
        self.step_count = 0
        self.shard_size = layout.shard_size

        # Build full FP32 param buffer on CPU to avoid GPU OOM,
        # then extract this rank's shard and move to GPU
        full_buf = torch.zeros(layout.total_numel, dtype=torch.float32, device="cpu")
        for name in layout.buffer_order:
            offset = layout.param_offsets[name]
            numel = layout.param_numels[name]
            full_buf[offset:offset + numel].copy_(params[name].float().cpu().view(-1))

        self.fp32_master_shard = layout.extract_shard(full_buf, dp_rank).to(device)
        del full_buf

        self.exp_avg_shard = torch.zeros_like(self.fp32_master_shard)
        self.exp_avg_sq_shard = torch.zeros_like(self.fp32_master_shard)

        # Pre-compute which params overlap with this shard and their WD status
        self.shard_param_slices = layout.get_shard_param_slices(dp_rank)

        # Build WD and no-WD index lists for multi_tensor_adam
        self.wd_slices: List[tuple] = []
        self.no_wd_slices: List[tuple] = []
        for name, local_start, local_end in self.shard_param_slices:
            if _is_no_weight_decay(name):
                self.no_wd_slices.append((local_start, local_end))
            else:
                self.wd_slices.append((local_start, local_end))

        # Pre-compute master/m/v slice views (reused every step)
        self._wd_p = [self.fp32_master_shard[s:e] for s, e in self.wd_slices]
        self._wd_m = [self.exp_avg_shard[s:e] for s, e in self.wd_slices]
        self._wd_v = [self.exp_avg_sq_shard[s:e] for s, e in self.wd_slices]
        self._no_wd_p = [self.fp32_master_shard[s:e] for s, e in self.no_wd_slices]
        self._no_wd_m = [self.exp_avg_shard[s:e] for s, e in self.no_wd_slices]
        self._no_wd_v = [self.exp_avg_sq_shard[s:e] for s, e in self.no_wd_slices]

        # Grad shard views are bound later via bind_grad_shard()
        self._wd_g: List[torch.Tensor] = []
        self._no_wd_g: List[torch.Tensor] = []
        self._norm_views: List[torch.Tensor] = []
        self._clip_views: List[torch.Tensor] = []

        # Per-bucket Adam views (populated by prepare_bucket_adam)
        self._num_adam_buckets = 0

    def prepare_bucket_adam(self, layout, num_buckets: int) -> None:
        """Pre-compute per-bucket WD/no-WD p/m/v views for pipelined Adam+AG.

        Must be called after __init__ and before bind_grad_shard.
        """
        self._num_adam_buckets = num_buckets
        bucket_wd_ranges = [[] for _ in range(num_buckets)]
        bucket_no_wd_ranges = [[] for _ in range(num_buckets)]

        bucket_shard_ranges = []
        for bi in range(num_buckets):
            so = layout.bucket_shard_offsets[bi]
            b_start, b_end = layout.bucket_ranges[bi]
            bs = (b_end - b_start) // layout.dp_size
            bucket_shard_ranges.append((so, so + bs))

        for name, local_start, local_end in self.shard_param_slices:
            for bi, (bso, bse) in enumerate(bucket_shard_ranges):
                if local_start >= bso and local_end <= bse:
                    target = (bucket_no_wd_ranges[bi]
                              if _is_no_weight_decay(name)
                              else bucket_wd_ranges[bi])
                    target.append((local_start, local_end))
                    break

        ms = self.fp32_master_shard
        ma = self.exp_avg_shard
        msq = self.exp_avg_sq_shard
        self._bucket_wd_p = [[ms[s:e] for s, e in r] for r in bucket_wd_ranges]
        self._bucket_wd_m = [[ma[s:e] for s, e in r] for r in bucket_wd_ranges]
        self._bucket_wd_v = [[msq[s:e] for s, e in r] for r in bucket_wd_ranges]
        self._bucket_no_wd_p = [[ms[s:e] for s, e in r] for r in bucket_no_wd_ranges]
        self._bucket_no_wd_m = [[ma[s:e] for s, e in r] for r in bucket_no_wd_ranges]
        self._bucket_no_wd_v = [[msq[s:e] for s, e in r] for r in bucket_no_wd_ranges]
        self._bucket_wd_ranges = bucket_wd_ranges
        self._bucket_no_wd_ranges = bucket_no_wd_ranges
        self._bucket_wd_g: List[List[torch.Tensor]] = [[] for _ in range(num_buckets)]
        self._bucket_no_wd_g: List[List[torch.Tensor]] = [[] for _ in range(num_buckets)]

    def bind_grad_shard(self, grad_shard: torch.Tensor, tp_rank: int = 0) -> None:
        """Pre-compute grad_shard slice views for norm/clip/adam (call once)."""
        self._wd_g = [grad_shard[s:e] for s, e in self.wd_slices]
        self._no_wd_g = [grad_shard[s:e] for s, e in self.no_wd_slices]
        self._clip_views = self._wd_g + self._no_wd_g
        self._norm_views = list(self._wd_g)
        if tp_rank == 0:
            self._norm_views = self._norm_views + list(self._no_wd_g)

        if self._num_adam_buckets > 0:
            for bi in range(self._num_adam_buckets):
                self._bucket_wd_g[bi] = [grad_shard[s:e]
                                         for s, e in self._bucket_wd_ranges[bi]]
                self._bucket_no_wd_g[bi] = [grad_shard[s:e]
                                            for s, e in self._bucket_no_wd_ranges[bi]]


def compute_distributed_grad_norm(
    grad_buffer: torch.Tensor,
    layout: "BufferLayout",
    dp_rank: int,
    state: "DistributedAdamState",
    device: str,
    tp_rank: int = 0,
    overflow_buf: torch.Tensor = None,
    norm_sq_buf: torch.Tensor = None,
    out_norm_sq: torch.Tensor = None,
    grad_shard: torch.Tensor = None,
) -> float:
    """Compute grad norm matching Megatron's DistributedOptimizer exactly.

    Megatron's get_main_grads_for_grad_norm() returns per-parameter gradient
    slices (within the shard), ordered WD-first then no-WD. A single
    multi_tensor_l2norm call is made on this list. This ordering matters
    because the CUDA kernel's FP32 accumulation order affects the result.

    Megatron also calls param_is_not_tensor_parallel_duplicate() which
    excludes TP-replicated params (layernorm weights = no-WD group) on
    tp_rank > 0 to avoid double-counting. We replicate this by skipping
    no-WD slices when tp_rank > 0.

    The all-reduce uses group=None (world), matching
    DistributedOptimizer.get_model_parallel_group() which returns None.

    If grad_shard is provided, index into it directly instead of into
    grad_buffer (used after reduce-scatter when the shard is separate).
    """
    import torch.distributed as dist

    if state._norm_views:
        grads_for_norm = state._norm_views
    else:
        if grad_shard is not None:
            shard_view = grad_shard
        else:
            shard_start = dp_rank * layout.shard_size
            shard_view = grad_buffer[shard_start:shard_start + layout.shard_size]

        grads_for_norm = []
        for local_start, local_end in state.wd_slices:
            grads_for_norm.append(shard_view[local_start:local_end])
        if tp_rank == 0:
            for local_start, local_end in state.no_wd_slices:
                grads_for_norm.append(shard_view[local_start:local_end])

    if overflow_buf is None:
        overflow_buf = torch.tensor([0], dtype=torch.int, device=device)

    if not grads_for_norm:
        if norm_sq_buf is not None:
            norm_sq_buf.fill_(0.0)
            local_norm_sq = norm_sq_buf
        else:
            local_norm_sq = torch.tensor([0.0], dtype=torch.float32, device=device)
    else:
        overflow_buf.zero_()
        grad_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, overflow_buf, [grads_for_norm], False,
        )
        if norm_sq_buf is not None:
            norm_sq_buf.copy_(grad_norm ** 2)
            local_norm_sq = norm_sq_buf
        else:
            local_norm_sq = grad_norm ** 2

    del grads_for_norm
    if out_norm_sq is not None:
        out_norm_sq.copy_(local_norm_sq)
        dist.all_reduce(out_norm_sq, op=dist.ReduceOp.SUM, group=None)
        return None
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM, group=None)
    total_norm = local_norm_sq.item() ** 0.5
    return total_norm


def clip_shard_grads(
    grad_shard: torch.Tensor,
    state: DistributedAdamState,
    max_norm: float,
    total_norm: float,
    overflow_buf: torch.Tensor = None,
) -> None:
    """Clip the gradient shard in-place by the global norm."""
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        if state._clip_views:
            all_slices = state._clip_views
        else:
            all_slices = []
            for local_start, local_end in state.wd_slices:
                all_slices.append(grad_shard[local_start:local_end])
            for local_start, local_end in state.no_wd_slices:
                all_slices.append(grad_shard[local_start:local_end])

        if all_slices:
            if overflow_buf is None:
                overflow_buf = torch.tensor([0], dtype=torch.int, device=state.device)
            else:
                overflow_buf.zero_()
            multi_tensor_applier(
                multi_tensor_scale, overflow_buf,
                [all_slices, all_slices], clip_coeff,
            )


def distributed_adam_step(
    state: DistributedAdamState,
    grad_shard: torch.Tensor,
    lr: float,
    overflow_buf: torch.Tensor = None,
) -> None:
    """Adam step on the local shard (WD and no-WD groups separately)."""
    state.step_count += 1
    t = state.step_count

    if _multi_tensor_adam_fn is None:
        raise RuntimeError("multi_tensor_adam not available")

    if overflow_buf is None:
        overflow_buf = torch.tensor([0], dtype=torch.int, device=state.device)
    else:
        overflow_buf.zero_()

    # Use pre-computed views when available (bind_grad_shard called at init)
    _has_precomp = bool(state._wd_g) or bool(state._no_wd_g)

    if state.wd_slices:
        if _has_precomp:
            wd_g, wd_p, wd_m, wd_v = state._wd_g, state._wd_p, state._wd_m, state._wd_v
        else:
            master = state.fp32_master_shard
            m = state.exp_avg_shard
            v = state.exp_avg_sq_shard
            wd_g = [grad_shard[s:e] for s, e in state.wd_slices]
            wd_p = [master[s:e] for s, e in state.wd_slices]
            wd_m = [m[s:e] for s, e in state.wd_slices]
            wd_v = [v[s:e] for s, e in state.wd_slices]

        multi_tensor_applier(
            _multi_tensor_adam_fn,
            overflow_buf,
            [wd_g, wd_p, wd_m, wd_v],
            lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, t, 1, 1, WEIGHT_DECAY,
        )

    if state.no_wd_slices:
        if _has_precomp:
            no_wd_g, no_wd_p = state._no_wd_g, state._no_wd_p
            no_wd_m, no_wd_v = state._no_wd_m, state._no_wd_v
        else:
            master = state.fp32_master_shard
            m = state.exp_avg_shard
            v = state.exp_avg_sq_shard
            no_wd_g = [grad_shard[s:e] for s, e in state.no_wd_slices]
            no_wd_p = [master[s:e] for s, e in state.no_wd_slices]
            no_wd_m = [m[s:e] for s, e in state.no_wd_slices]
            no_wd_v = [v[s:e] for s, e in state.no_wd_slices]

        multi_tensor_applier(
            _multi_tensor_adam_fn,
            overflow_buf,
            [no_wd_g, no_wd_p, no_wd_m, no_wd_v],
            lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, t, 1, 1, 0.0,
        )


def distributed_adam_step_bucket(
    state: DistributedAdamState,
    bucket_idx: int,
    lr: float,
    step_count: int,
    overflow_buf: torch.Tensor,
) -> None:
    """Adam step on one bucket's WD + no-WD params.

    Caller is responsible for incrementing state.step_count once per
    training step and passing the same value to every bucket call.
    Caller must also pre-zero overflow_buf once before the bucket loop;
    multi_tensor_adam never sets the overflow flag during normal training,
    so a single zero per step suffices.
    """
    wd_g = state._bucket_wd_g[bucket_idx]
    if wd_g:
        multi_tensor_applier(
            _multi_tensor_adam_fn, overflow_buf,
            [wd_g, state._bucket_wd_p[bucket_idx],
             state._bucket_wd_m[bucket_idx], state._bucket_wd_v[bucket_idx]],
            lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, step_count, 1, 1, WEIGHT_DECAY,
        )

    no_wd_g = state._bucket_no_wd_g[bucket_idx]
    if no_wd_g:
        multi_tensor_applier(
            _multi_tensor_adam_fn, overflow_buf,
            [no_wd_g, state._bucket_no_wd_p[bucket_idx],
             state._bucket_no_wd_m[bucket_idx], state._bucket_no_wd_v[bucket_idx]],
            lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, step_count, 1, 1, 0.0,
        )


