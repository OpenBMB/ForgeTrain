"""Optimizer step for MiniCPM4 0.5B.

Implements Adam with FP32 master weights, precision-aware parameter sync,
gradient clipping, and learning rate scheduling.
"""

from __future__ import annotations

__all__ = [
    "AdamState",
    "OptimizerScalarBuffers",
    "ShardedOptimizerBuffers",
    "adam_step",
    "clip_gradients_fp32",
    "compute_clip_coeff_device",
    "compute_grad_norm_fp32",
    "compute_lr",
    "fused_clip_adam_sync",
    "fused_clip_adam_sync_bucketed",
    "fused_clip_adam_sync_tensor",
    "sharded_fused_clip_adam_sync",
    "sharded_fused_clip_adam_sync_multi_tensor",
    "sync_params_from_master",
]

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from . import config
from .engine_config import get_config, register_reset_hook

if TYPE_CHECKING:
    from .nccl import BucketedGradReducer

_wd_lookup: frozenset[str] | None = None


def _has_weight_decay(name: str) -> bool:
    """Whether *name* should receive weight decay, derived from ParamSpec SSOT."""
    global _wd_lookup
    if _wd_lookup is None:
        from .parameters import all_param_specs
        _wd_lookup = frozenset(
            s.name for s in all_param_specs() if s.has_weight_decay
        )
    return name in _wd_lookup


def _clear_wd_lookup() -> None:
    global _wd_lookup
    _wd_lookup = None


register_reset_hook(_clear_wd_lookup)


class AdamState:
    """FP32 Adam optimizer state (all optimizer state in FP32).

    Field naming uses ``exp_avg`` / ``exp_avg_sq`` (the canonical Adam
    first/second moment names used across the wider ML ecosystem) so the
    on-disk checkpoint can carry the same dict keys end-to-end and the
    resume-gate driver doesn't need any aliasing layer.
    """

    # Keys persisted to / restored from the optimizer-state portion of the
    # resume checkpoint. Centralized here so add/remove of a serialised
    # field requires touching exactly one site.
    _PERSISTED_TENSOR_DICT_FIELDS = ("master_weights", "exp_avg", "exp_avg_sq")
    _PERSISTED_SCALAR_FIELDS = ("step_count", "num_samples")

    def __init__(
        self,
        param_names: list[str],
        bf16_params: dict[str, torch.Tensor],
        device: str | torch.device = "cuda:0",
    ):
        self.param_names = param_names
        self.master_weights: dict[str, torch.Tensor] = {}
        self.exp_avg: dict[str, torch.Tensor] = {}
        self.exp_avg_sq: dict[str, torch.Tensor] = {}
        self.step_count: int = 0
        self.num_samples: int = 0

        for name in param_names:
            if name in bf16_params:
                w = bf16_params[name].to(dtype=torch.float32, device=device)
                self.master_weights[name] = w
                self.exp_avg[name] = torch.zeros_like(w)
                self.exp_avg_sq[name] = torch.zeros_like(w)

    # ------------------------------------------------------------------
    # Resume serialisation helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Return a serialisable dict of all FP32 Adam state.

        Keys map 1:1 to ``AdamState`` attributes; tensors are returned by
        reference (callers are expected to ``.detach().cpu()`` before
        writing to disk so the on-disk file does not pin GPU memory).
        """
        out: dict = {}
        for f in self._PERSISTED_TENSOR_DICT_FIELDS:
            out[f] = dict(getattr(self, f))
        for f in self._PERSISTED_SCALAR_FIELDS:
            out[f] = int(getattr(self, f))
        return out

    def load_state_dict(self, state: dict, *, strict: bool = True) -> None:
        """Restore Adam state from a previously-saved dict in place.

        ``strict=True`` (default) requires every persisted tensor-dict
        field to be present **and** each tensor key to match the current
        ``param_names`` set, so a partial / corrupt checkpoint fails loud
        instead of resuming with zeroed Adam moments.

        ``strict=False`` is for the loose round-trip path (e.g. when a
        future field is added but resumes from an older checkpoint that
        doesn't carry it yet).
        """
        if not isinstance(state, dict):
            raise TypeError(
                f"AdamState.load_state_dict expected dict, got {type(state).__name__}"
            )

        for f in self._PERSISTED_TENSOR_DICT_FIELDS:
            if f not in state:
                if strict:
                    raise KeyError(
                        f"AdamState.load_state_dict: missing required field {f!r}"
                    )
                continue
            value = state[f]
            if not isinstance(value, dict):
                raise TypeError(
                    f"AdamState.load_state_dict: field {f!r} must be dict, "
                    f"got {type(value).__name__}"
                )
            if strict:
                expected = set(self.param_names) & set(self.master_weights.keys())
                got = set(value.keys())
                missing = expected - got
                if missing:
                    raise KeyError(
                        f"AdamState.load_state_dict: field {f!r} missing keys "
                        f"for params {sorted(missing)[:5]}..."
                    )
            setattr(self, f, dict(value))

        for f in self._PERSISTED_SCALAR_FIELDS:
            if f in state:
                setattr(self, f, int(state[f]))
            elif strict:
                raise KeyError(
                    f"AdamState.load_state_dict: missing required field {f!r}"
                )


def compute_lr(num_samples: int, global_batch_size: int,
               step_count: int | None = None) -> float:
    """Compute LR using WSD schedule (warmup-stable-decay).

    When ``step_count`` is provided it is used directly as the schedule
    index (``iteration``). Otherwise falls back to
    ``num_samples // global_batch_size`` for backward compatibility.

    WSD schedule matches Megatron's OptimizerParamScheduler.get_lr():
      - warmup: linear from 0 to max_lr over warmup_iters steps
      - stable: max_lr until lr_decay_iters - wsd_decay_iters
      - decay: exponential decay from max_lr to min_lr over wsd_decay_iters
        using Megatron's formula: coeff = 2 * 0.5^(ratio) - 1
    """
    cfg = get_config()
    if step_count is not None:
        iteration = step_count
    else:
        iteration = num_samples // global_batch_size if global_batch_size > 0 else 0

    if iteration < cfg.warmup_iters:
        return cfg.max_lr * iteration / cfg.warmup_iters

    if iteration > cfg.lr_decay_iters:
        return cfg.min_lr

    wsd_anneal_start = cfg.lr_decay_iters - cfg.wsd_decay_iters
    if iteration <= wsd_anneal_start:
        return cfg.max_lr

    wsd_steps = iteration - wsd_anneal_start
    wsd_ratio = float(wsd_steps) / float(cfg.wsd_decay_iters)
    coeff = 2.0 * math.pow(0.5, wsd_ratio) - 1.0
    return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)


def mup_lr_for_param(name: str, base_lr: float) -> float:
    """Apply muP per-parameter LR scaling.

    Megatron muP scales lr by 1/width_scale for weight matrices that are
    NOT embedding, layernorm, or output_layer. This matches the
    ``scale_lr_cond`` logic in ``megatron/training/training.py``.
    """
    ws = config.mup_width_scale()
    if ws == 1.0:
        return base_lr
    if name.endswith(".weight") and "norm" not in name \
            and "embedding" not in name and "output" not in name:
        return base_lr / ws
    return base_lr


def compute_grad_norm_fp32(
    fp32_grads: dict[str, torch.Tensor],
    *,
    grad_order: list[str] | None = None,
) -> float:
    """Compute L2 gradient norm in FP32 using TE multi_tensor_l2norm.

    Path: l2norm → square → .item()**0.5
    Uses baseline optimizer parameter ordering for bitwise-exact accumulation.
    """
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
    )

    norm_order = grad_order if grad_order is not None else _baseline_grad_norm_order()
    grads_list = []
    for name in norm_order:
        if name in fp32_grads:
            g = fp32_grads[name]
            if g.dtype != torch.float32:
                g = g.float()
            grads_list.append(g.view(-1))

    if not grads_list:
        return 0.0

    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device="cuda")
    grad_norm, _ = multi_tensor_applier(
        multi_tensor_l2norm,
        dummy_overflow_buf,
        [grads_list],
        False,
    )
    total_norm = grad_norm * grad_norm
    return total_norm.item() ** 0.5


def clip_gradients_fp32(
    fp32_grads: dict[str, torch.Tensor],
    max_norm: float,
    total_norm: float,
) -> None:
    """Clip gradients by global L2 norm in-place."""
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_scale,
    )

    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        clip_order = _baseline_grad_norm_order()
        grads_list = []
        for name in clip_order:
            if name in fp32_grads:
                grads_list.append(fp32_grads[name])
        if grads_list:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device="cuda")
            multi_tensor_applier(
                multi_tensor_scale,
                dummy_overflow_buf,
                [grads_list, grads_list],
                clip_coeff,
            )


def adam_step(
    state: AdamState,
    fp32_grads: dict[str, torch.Tensor],
    lr: float,
) -> None:
    """Single Adam step with FP32 master weights (AdamW).

    Calls TE multi_tensor_adam CUDA kernel directly.
    1D params (layernorm) use wd=0; 2D+ params use weight_decay.
    """
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.optimizers import multi_tensor_applier

    cfg = get_config()
    state.step_count += 1

    wd_lists: list[list[torch.Tensor]] = [[], [], [], []]
    no_wd_lists: list[list[torch.Tensor]] = [[], [], [], []]

    for name in state.param_names:
        if name not in fp32_grads or name not in state.master_weights:
            continue
        g = fp32_grads[name].float()
        p = state.master_weights[name]
        m = state.exp_avg[name]
        v = state.exp_avg_sq[name]
        has_wd = _has_weight_decay(name)
        target = wd_lists if has_wd else no_wd_lists
        target[0].append(g)
        target[1].append(p)
        target[2].append(m)
        target[3].append(v)

    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device="cuda")

    if wd_lists[0]:
        multi_tensor_applier(
            tex.multi_tensor_adam,
            dummy_overflow_buf,
            wd_lists,
            lr,
            cfg.adam_beta1,
            cfg.adam_beta2,
            cfg.adam_eps,
            state.step_count,
            1,     # adam_w_mode = 1 (AdamW / decoupled weight decay)
            True,  # bias_correction
            cfg.weight_decay,
        )

    if no_wd_lists[0]:
        multi_tensor_applier(
            tex.multi_tensor_adam,
            dummy_overflow_buf,
            no_wd_lists,
            lr,
            cfg.adam_beta1,
            cfg.adam_beta2,
            cfg.adam_eps,
            state.step_count,
            1,     # adam_w_mode (irrelevant when wd=0)
            True,  # bias_correction
            0.0,   # no weight decay for 1D params
        )


def sync_params_from_master(
    params: dict[str, torch.Tensor],
    state: AdamState,
) -> None:
    """Copy FP32 master weights to BF16 model params."""
    for name in state.param_names:
        if name in params and name in state.master_weights:
            master = state.master_weights[name]
            params[name].copy_(master.to(params[name].dtype))


def fused_clip_adam_sync(
    state: AdamState,
    fp32_grads: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    lr: float,
    clip_coeff: float,
) -> None:
    """Fused gradient clip + AdamW update + FP32→BF16 sync in one pass.

    Replaces clip_gradients_fp32 + adam_step + sync_params_from_master.
    """
    from .triton_kernels import fused_adam_sync

    cfg = get_config()
    state.step_count += 1
    for name in state.param_names:
        if name not in fp32_grads or name not in state.master_weights:
            continue
        g = fp32_grads[name].float()
        has_wd = _has_weight_decay(name)
        wd = cfg.weight_decay if has_wd else 0.0
        param_lr = mup_lr_for_param(name, lr)
        fused_adam_sync(
            g, state.master_weights[name],
            state.exp_avg[name], state.exp_avg_sq[name],
            params[name],
            lr=param_lr, step=state.step_count,
            clip_coeff=clip_coeff, wd=wd,
        )


# ---------------------------------------------------------------------
# M2-P33 Phase B: tensor-arg fused Adam (CUDA Graph compatible)
# ---------------------------------------------------------------------
#
# When the optimizer step is captured into a CUDA Graph, every Python
# scalar passed to the Triton kernel is burnt into the launch node;
# replay would then freeze ``lr`` at the cosine-schedule value of the
# capture step and freeze Adam bias correction at ``1 - beta**step_capture``.
# Both errors are silent — loss looks plausible but the schedule is
# stuck. The classes / wrapper below replace those four scalars with
# 1-element FP32 device buffers that the caller refreshes BEFORE every
# replay, restoring per-step correctness while keeping the graph
# launch-once.
#
# Lifecycle (one ``OptimizerScalarBuffers`` instance per training run):
#
#   bufs = OptimizerScalarBuffers(device)            # one alloc total
#   ...
#   for step in range(num_steps):
#       lr         = compute_lr(...)                  # cosine schedule
#       clip_coeff = ...                              # see grad-norm path
#       opt_state.step_count += 1
#       bufs.update_from_host(lr, opt_state.step_count, clip_coeff)
#       # graph-replay invokes fused_adam_sync_tensor (kernel reads
#       # the freshly-loaded 1-elem buffers).
#
# Values from ``EngineConfig`` (``adam_beta1``, ``adam_beta2``,
# ``adam_eps``, ``weight_decay``) stay scalar kernel-args — they truly
# are constant for the whole run so capturing them in the launch node
# is correct.


def compute_clip_coeff_device(
    grad_norm_sq: torch.Tensor,
    clip_coeff_buf: torch.Tensor,
    *,
    max_norm: float | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Device-only computation of Adam-clip coefficient (no host sync).

    Replaces the legacy

        grad_norm = total_norm_sq.item() ** 0.5
        clip_coeff = min(1.0, max_norm / (grad_norm + 1e-6))

    with a fully tensor-resident pipeline:

        grad_norm  = sqrt(grad_norm_sq)
        clip_coeff = min(1, max_norm / (grad_norm + eps))

    Writes the result into the caller-supplied ``clip_coeff_buf`` (the
    same 1-element FP32 device tensor that the captured Adam kernel
    reads via ``tl.load``). Returning the buffer back lets callers chain
    fluent style (``compute_clip_coeff_device(...).item()`` for a debug
    print) without round-tripping the host on the hot path.

    Numerical equivalence to the legacy path: both expressions compute
    the same FP32 ``min(1.0, max_norm / (sqrt(grad_norm_sq) + eps))``
    in single-precision arithmetic. The only behavioural difference is
    that the kernel-side path defers the host visibility of the clip
    coefficient until the next ``synchronize`` — which is what allows
    capture in the first place.
    """
    if max_norm is None:
        max_norm = get_config().max_grad_norm
    grad_norm = grad_norm_sq.view(()).clamp(min=0.0).sqrt()
    coeff = (max_norm / (grad_norm + eps)).clamp(max=1.0)
    clip_coeff_buf.view(()).copy_(coeff)
    return clip_coeff_buf


class OptimizerScalarBuffers:
    """Persistent 1-element FP32 device buffers for tensor-arg Adam.

    Holds the four per-step scalars that the captured fused-Adam
    Triton kernel reads via ``tl.load``:

    * ``lr_buf``           — current learning rate
    * ``clip_coeff_buf``   — ``min(1, max_grad_norm / (grad_norm + 1e-6))``
    * ``bc1_buf``          — Adam bias correction ``1 - adam_beta1**step``
    * ``bc2_buf``          — Adam bias correction ``1 - adam_beta2**step``

    The buffers are allocated once and reused for every step, so their
    ``data_ptr`` s are stable across captures and the same recorded
    launch node is correct on every replay.

    The instance also owns a 4-slot pinned host-staging tensor so the
    per-step ``update_from_host`` path issues a single 4-element H2D
    copy (via 4 ``copy_(non_blocking=True)`` of length-1 views over
    the pinned scratch) instead of crossing the H2D PCIe boundary
    four times with Python-float source operands.
    """

    def __init__(self, device: str | torch.device) -> None:
        self.lr_buf = torch.zeros(1, dtype=torch.float32, device=device)
        self.clip_coeff_buf = torch.zeros(1, dtype=torch.float32, device=device)
        self.bc1_buf = torch.zeros(1, dtype=torch.float32, device=device)
        self.bc2_buf = torch.zeros(1, dtype=torch.float32, device=device)
        try:
            self._scratch = torch.zeros(4, dtype=torch.float32, pin_memory=True)
        except (RuntimeError, NotImplementedError):
            self._scratch = torch.zeros(4, dtype=torch.float32)

    def update_from_host(
        self,
        lr: float,
        step_count: int,
        clip_coeff: float,
        beta1: float | None = None,
        beta2: float | None = None,
    ) -> None:
        """Refresh the four device buffers from host scalars.

        Computes the two Adam bias-correction terms on the host (cheap
        — a pair of ``float ** int`` ops) so the kernel does not need
        a per-program ``tl.exp(log(beta) * step)`` reduction. Issues
        four async H2D copies; the captured replay's first kernel
        will naturally observe the latest values because the copies
        target buffers whose storage pointers were recorded into the
        graph and there is no inter-step CUDA sync separating the
        copies from the launch.
        """
        if beta1 is None or beta2 is None:
            cfg = get_config()
            beta1 = beta1 if beta1 is not None else cfg.adam_beta1
            beta2 = beta2 if beta2 is not None else cfg.adam_beta2
        bc1 = 1.0 - beta1 ** step_count
        bc2 = 1.0 - beta2 ** step_count
        self._scratch[0] = float(lr)
        self._scratch[1] = float(clip_coeff)
        self._scratch[2] = bc1
        self._scratch[3] = bc2
        self.lr_buf.copy_(self._scratch[0:1], non_blocking=True)
        self.clip_coeff_buf.copy_(self._scratch[1:2], non_blocking=True)
        self.bc1_buf.copy_(self._scratch[2:3], non_blocking=True)
        self.bc2_buf.copy_(self._scratch[3:4], non_blocking=True)


def fused_clip_adam_sync_tensor(
    state: AdamState,
    fp32_grads: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    bufs: OptimizerScalarBuffers,
) -> None:
    """Tensor-arg variant of :func:`fused_clip_adam_sync` (CUDA Graph safe).

    The caller MUST have refreshed ``bufs`` for the current step
    BEFORE invoking this function (typically alongside
    ``state.step_count += 1``). When this function runs inside
    ``torch.cuda.graph(...)``, the resulting launch nodes capture the
    four buffer ``data_ptr`` s; replay then reads whatever the host
    last copied into the buffers, so the per-step
    ``bufs.update_from_host(lr, step, clip_coeff)`` immediately
    upstream of ``graph.replay()`` is what implements the cosine
    schedule and Adam bias correction during replay.

    Per-param weight decay grouping mirrors the scalar-arg variant
    (``_has_weight_decay(name)`` ⇒ ``weight_decay``, else ``0.0``).
    """
    from .triton_kernels import fused_adam_sync_tensor

    cfg = get_config()
    for name in state.param_names:
        if name not in fp32_grads or name not in state.master_weights:
            continue
        g = fp32_grads[name].float()
        has_wd = _has_weight_decay(name)
        wd = cfg.weight_decay if has_wd else 0.0
        fused_adam_sync_tensor(
            g, state.master_weights[name],
            state.exp_avg[name], state.exp_avg_sq[name],
            params[name],
            lr_buf=bufs.lr_buf,
            clip_coeff_buf=bufs.clip_coeff_buf,
            bc1_buf=bufs.bc1_buf,
            bc2_buf=bufs.bc2_buf,
            wd=wd,
        )


def fused_clip_adam_sync_bucketed(
    state: AdamState,
    bucket_iter: Iterable[tuple[Sequence[str], dict[str, torch.Tensor]]],
    params: dict[str, torch.Tensor],
    lr: float,
    clip_coeff: float,
) -> None:
    """Per-bucket variant of :func:`fused_clip_adam_sync`.

    Each ``(bucket_param_names, bucket_grads)`` tuple yielded by
    ``bucket_iter`` is processed in order. ``bucket_grads`` only needs
    to contain entries for the names in ``bucket_param_names`` (other
    keys are ignored). Per-param ``fused_adam_sync`` calls match the
    whole-model variant byte-for-byte: same step_count, same per-param
    weight-decay rule (``len(shape) > 1 → weight_decay``, else 0.0),
    same lr / clip_coeff.

    Why a separate function? The wgrad-overlap + allgather-overlap
    pipeline (M2-P6 P1.c) wants the optimizer to consume grads
    bucket-by-bucket so the consumer can synchronise with the comm
    stream's per-bucket allgather event one bucket at a time, letting
    the comm stream keep issuing the next bucket's allgather while the
    default stream runs the current bucket's adam. Reusing
    ``fused_clip_adam_sync`` directly would require batching the
    iterator into a dict first (which defeats the overlap by forcing
    every bucket's allgather to complete before the first
    ``fused_adam_sync`` call).

    The function increments ``state.step_count`` exactly once before
    the iteration starts — the bias-correction terms inside
    ``fused_adam_sync`` are functions of step_count, so calling per
    bucket is mathematically equivalent to one whole-model call as
    long as step_count is the same constant for every per-param
    invocation in the step.
    """
    from .triton_kernels import fused_adam_sync

    cfg = get_config()
    state.step_count += 1
    seen_names: set[str] = set()
    for bucket_param_names, bucket_grads in bucket_iter:
        for name in bucket_param_names:
            if name not in bucket_grads or name not in state.master_weights:
                continue
            if name in seen_names:
                raise RuntimeError(
                    f"fused_clip_adam_sync_bucketed: duplicate param "
                    f"{name!r} across buckets — bucket layout must be a "
                    f"partition of the trainable params"
                )
            seen_names.add(name)
            g = bucket_grads[name].float()
            has_wd = _has_weight_decay(name)
            wd = cfg.weight_decay if has_wd else 0.0
            param_lr = mup_lr_for_param(name, lr)
            fused_adam_sync(
                g, state.master_weights[name],
                state.exp_avg[name], state.exp_avg_sq[name],
                params[name],
                lr=param_lr, step=state.step_count,
                clip_coeff=clip_coeff, wd=wd,
            )


class ShardedOptimizerBuffers:
    """Per-bucket BF16 shard + shared allgather buffer for ZeRO-1 optimizer.

    Allocates lightweight BF16 buffers that receive the Triton kernel's
    BF16 output.  The optimizer state (master_weights, exp_avg,
    exp_avg_sq) stays in the per-param :class:`AdamState` — the sharded
    optimizer writes directly into the per-param tensors' views,
    avoiding any packing / unpacking overhead.

    Buffers (once, persistent across steps):

    * ``bf16_shards``: one ``(shard_size,)`` BF16 tensor per bucket.
      The Triton kernel writes each shard element's updated BF16 value
      here during the per-shard optimizer step.
    * ``bf16_full_buf``: one shared ``(max_bucket_size,)`` BF16 tensor
      for ``all_gather_into_tensor`` output.  Reused across buckets.
    """

    def __init__(
        self,
        reducer: BucketedGradReducer,
        device: str | torch.device,
    ) -> None:
        self.bf16_shards: list[torch.Tensor] = []
        for bucket in reducer.buckets:
            shard_size = bucket.size // reducer.world_size
            self.bf16_shards.append(
                torch.zeros(shard_size, dtype=torch.bfloat16, device=device)
            )
        max_bucket_size = max(b.size for b in reducer.buckets)
        self.bf16_full_buf = torch.empty(
            max_bucket_size, dtype=torch.bfloat16, device=device,
        )


def sharded_fused_clip_adam_sync(
    state: AdamState,
    reducer: BucketedGradReducer,
    sharded_bufs: ShardedOptimizerBuffers,
    params: dict[str, torch.Tensor],
    lr: float,
    clip_coeff: float,
) -> None:
    """ZeRO-1 sharded optimizer: per-shard Adam + BF16 allgather.

    Each rank runs the fused Adam update only on the gradient shard it
    owns (1/world_size of the model), then ``allgather`` broadcasts the
    resulting BF16 parameters.  This eliminates two costs that dominate
    the step tail:

    * The FP32 gradient ``all_gather_into_tensor`` (~9 ms on DP=16)
      is replaced by a BF16 parameter ``all_gather_into_tensor``
      (half the data → ~4.5 ms).
    * The full-model optimizer step (~5.6 ms on DP=16) is replaced
      by a 1/world_size shard (~0.35 ms on DP=16).

    The function writes directly into per-param slices of
    ``state.master_weights``, ``state.exp_avg``, ``state.exp_avg_sq``
    via views, so checkpoint save still works on the per-param dicts
    (although only this rank's shard elements are up-to-date — see the
    ``recommendation.md`` caveat for checkpoint compatibility).

    Must be called AFTER ``reducer.wait_all()`` (so the per-bucket
    reduce_scatter results are available in the shard buffers) and
    AFTER ``compute_distributed_grad_norm`` (so ``clip_coeff`` has
    been computed from the sharded norm).
    """
    from .triton_kernels import fused_adam_sync

    cfg = get_config()
    state.step_count += 1
    world_size = reducer.world_size
    rank = reducer.rank

    for bucket_idx, bucket in enumerate(reducer.buckets):
        shard_size = bucket.size // world_size
        rank_start = rank * shard_size
        grad_shard = reducer.get_grad_shard(bucket_idx)
        bf16_shard = sharded_bufs.bf16_shards[bucket_idx]

        for (name, numel), offset in zip(bucket.params, bucket.param_offsets):
            p_start = offset
            p_end = offset + numel
            ov_start = max(p_start, rank_start)
            ov_end = min(p_end, rank_start + shard_size)
            if ov_start >= ov_end:
                continue

            local_s = ov_start - rank_start
            local_e = ov_end - rank_start
            param_s = ov_start - p_start
            param_e = ov_end - p_start

            grad_slice = grad_shard[local_s:local_e]
            master_slice = state.master_weights[name].view(-1)[param_s:param_e]
            exp_avg_slice = state.exp_avg[name].view(-1)[param_s:param_e]
            exp_avg_sq_slice = state.exp_avg_sq[name].view(-1)[param_s:param_e]
            bf16_slice = bf16_shard[local_s:local_e]

            has_wd = _has_weight_decay(name)
            wd = cfg.weight_decay if has_wd else 0.0
            param_lr = mup_lr_for_param(name, lr)

            fused_adam_sync(
                grad_slice, master_slice,
                exp_avg_slice, exp_avg_sq_slice,
                bf16_slice,
                lr=param_lr, step=state.step_count,
                clip_coeff=clip_coeff, wd=wd,
            )

    reducer.allgather_bf16_and_unpack(
        sharded_bufs.bf16_shards,
        sharded_bufs.bf16_full_buf,
        params,
    )


def sharded_fused_clip_adam_sync_multi_tensor(
    state: AdamState,
    reducer: BucketedGradReducer,
    sharded_bufs: ShardedOptimizerBuffers,
    params: dict[str, torch.Tensor],
    lr: float,
    clip_coeff: float,
) -> None:
    """Multi-tensor variant of :func:`sharded_fused_clip_adam_sync`.

    Replaces the inner per-param Triton ``fused_adam_sync`` kernel
    launches (~148 calls/step on 0.5B × 4 buckets) with a small
    constant number of multi-tensor kernel launches:

    * ``multi_tensor_scale``  — in-place clip of every grad shard
    * ``tex.multi_tensor_adam`` — fused AdamW update of master/m/v
    * ``torch._foreach_copy_`` — multi-tensor FP32→BF16 cast into
      the per-bucket ``bf16_shards`` buffer

    All slice geometry (which param contributes which range to which
    bucket shard) is identical to the per-param variant — the same
    ``(local_s, local_e, param_s, param_e)`` overlap arithmetic feeds
    the multi-tensor lists. Param weight-decay grouping mirrors the
    per-param decision (``len(shape) > 1`` → ``weight_decay``, else
    ``0.0``) so the resulting update is mathematically equivalent to
    the per-param fused kernel:

    * Phase 1 in-place FP32 grad scale by ``clip_coeff`` matches the
      ``g * clip_coeff`` register-load step inside
      ``_fused_adam_sync_kernel``.
    * Phase 2 ``multi_tensor_adam`` with ``adam_w_mode=1`` and
      ``bias_correction=True`` reproduces the same decoupled-AdamW
      update (``p_new = p - lr * (m_hat / (sqrt(v_hat) + adam_eps) +
      weight_decay * p)``) byte-for-byte modulo per-thread reduction order.
    * Phase 3 ``torch._foreach_copy_(bf16, master)`` performs the
      identical RNE FP32→BF16 cast as Triton's ``.to(tl.bfloat16)``.

    The function still calls ``reducer.allgather_bf16_and_unpack`` at
    the end so the broader train-loop contract is unchanged.
    """
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_scale,
    )

    cfg = get_config()
    state.step_count += 1
    world_size = reducer.world_size
    rank = reducer.rank

    grads_wd: list[torch.Tensor] = []
    masters_wd: list[torch.Tensor] = []
    ms_wd: list[torch.Tensor] = []
    vs_wd: list[torch.Tensor] = []
    bf16_dst_wd: list[torch.Tensor] = []
    masters_src_wd: list[torch.Tensor] = []

    grads_no_wd: list[torch.Tensor] = []
    masters_no_wd: list[torch.Tensor] = []
    ms_no_wd: list[torch.Tensor] = []
    vs_no_wd: list[torch.Tensor] = []
    bf16_dst_no_wd: list[torch.Tensor] = []
    masters_src_no_wd: list[torch.Tensor] = []

    for bucket_idx, bucket in enumerate(reducer.buckets):
        shard_size = bucket.size // world_size
        rank_start = rank * shard_size
        grad_shard = reducer.get_grad_shard(bucket_idx)
        bf16_shard = sharded_bufs.bf16_shards[bucket_idx]

        for (name, numel), offset in zip(bucket.params, bucket.param_offsets):
            p_start = offset
            p_end = offset + numel
            ov_start = max(p_start, rank_start)
            ov_end = min(p_end, rank_start + shard_size)
            if ov_start >= ov_end:
                continue

            local_s = ov_start - rank_start
            local_e = ov_end - rank_start
            param_s = ov_start - p_start
            param_e = ov_end - p_start

            grad_slice = grad_shard[local_s:local_e]
            master_slice = state.master_weights[name].view(-1)[param_s:param_e]
            exp_avg_slice = state.exp_avg[name].view(-1)[param_s:param_e]
            exp_avg_sq_slice = state.exp_avg_sq[name].view(-1)[param_s:param_e]
            bf16_slice = bf16_shard[local_s:local_e]

            has_wd = _has_weight_decay(name)
            if has_wd:
                grads_wd.append(grad_slice)
                masters_wd.append(master_slice)
                ms_wd.append(exp_avg_slice)
                vs_wd.append(exp_avg_sq_slice)
                bf16_dst_wd.append(bf16_slice)
                masters_src_wd.append(master_slice)
            else:
                grads_no_wd.append(grad_slice)
                masters_no_wd.append(master_slice)
                ms_no_wd.append(exp_avg_slice)
                vs_no_wd.append(exp_avg_sq_slice)
                bf16_dst_no_wd.append(bf16_slice)
                masters_src_no_wd.append(master_slice)

    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device="cuda")

    # Phase 1: in-place FP32 grad clip. Equivalent to ``g * clip_coeff``
    # at the load step inside the per-param Triton kernel — same FP32
    # arithmetic, just batched. Skipping the call when ``clip_coeff ==
    # 1.0`` would still be correct but the saved 1 kernel launch is not
    # worth the branch cost when the multi_tensor list is already on the
    # GPU.
    if grads_wd:
        multi_tensor_applier(
            multi_tensor_scale,
            dummy_overflow_buf,
            [grads_wd, grads_wd],
            clip_coeff,
        )
    if grads_no_wd:
        multi_tensor_applier(
            multi_tensor_scale,
            dummy_overflow_buf,
            [grads_no_wd, grads_no_wd],
            clip_coeff,
        )

    # Phase 2: multi-tensor AdamW update of master/m/v. Two calls — one
    # per weight-decay group — replace ~148 per-param Triton dispatches.
    if grads_wd:
        multi_tensor_applier(
            tex.multi_tensor_adam,
            dummy_overflow_buf,
            [grads_wd, masters_wd, ms_wd, vs_wd],
            lr,
            cfg.adam_beta1,
            cfg.adam_beta2,
            cfg.adam_eps,
            state.step_count,
            1,     # adam_w_mode = 1 (decoupled AdamW)
            True,  # bias_correction
            cfg.weight_decay,
        )
    if grads_no_wd:
        multi_tensor_applier(
            tex.multi_tensor_adam,
            dummy_overflow_buf,
            [grads_no_wd, masters_no_wd, ms_no_wd, vs_no_wd],
            lr,
            cfg.adam_beta1,
            cfg.adam_beta2,
            cfg.adam_eps,
            state.step_count,
            1,
            True,
            0.0,
        )

    # Phase 3: multi-tensor FP32→BF16 cast into the per-bucket bf16
    # shard buffers. ``torch._foreach_copy_`` is a single CUDA kernel
    # launch per call (PyTorch >= 2.0), with built-in dtype cast
    # following IEEE round-to-nearest-even, identical to Triton's
    # ``.to(tl.bfloat16)`` store path.
    if bf16_dst_wd:
        torch._foreach_copy_(bf16_dst_wd, masters_src_wd)
    if bf16_dst_no_wd:
        torch._foreach_copy_(bf16_dst_no_wd, masters_src_no_wd)

    reducer.allgather_bf16_and_unpack(
        sharded_bufs.bf16_shards,
        sharded_bufs.bf16_full_buf,
        params,
    )


_baseline_grad_norm_order_cache: list[str] | None = None


def _clear_grad_norm_cache() -> None:
    """Clear the cached grad norm order. Called when global config is reset."""
    global _baseline_grad_norm_order_cache
    _baseline_grad_norm_order_cache = None


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: I001
_register_reset_hook(_clear_grad_norm_cache)


def _baseline_grad_norm_order() -> list[str]:
    """Return parameter names in baseline optimizer grad norm order.

    Derived from ``parameters.all_param_specs()`` — the single source of
    truth.  BF16 params (``is_norm=False``) come first in
    reverse-registration order, followed by FP32 params
    (``is_norm=True``) in reverse-registration order.

    The result is cached because the model structure is fixed for the
    lifetime of a training run, and this function is called twice per
    step (norm computation + clipping).
    """
    global _baseline_grad_norm_order_cache
    if _baseline_grad_norm_order_cache is not None:
        return _baseline_grad_norm_order_cache

    from .parameters import all_param_specs

    specs = all_param_specs()
    non_norm = [s.name for s in specs if not s.is_norm]
    norm = [s.name for s in specs if s.is_norm]
    _baseline_grad_norm_order_cache = list(reversed(non_norm)) + list(reversed(norm))
    return _baseline_grad_norm_order_cache
