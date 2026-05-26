"""Top-level entry points for the MiniCPM4-8B training engine.

Three callables form the public surface:

  run_forward(checkpoint_root, batch, device)
    → (loss: float, per_token_losses: Tensor)

  run_forward_backward(checkpoint_root, batch, device)
    → (loss: float, gradients: dict[str, Tensor])

  run_training(checkpoint_root, batches, num_steps, device,
               world_size=1, rank=0, tp_size=1,
               params_override=None)
    → list[dict]  # {loss, grad_norm, step_time}

When ``params_override`` is supplied, ``run_training`` consumes the
pre-built parameter dict directly instead of reading the checkpoint
file — useful for from-scratch initialisation paths that want to seed
parameters via :func:`parameters.self_init_params` without ever
touching disk.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

__all__ = ["run_forward", "run_forward_backward", "run_training"]

from . import config
from .engine_config import get_config, print_config_summary
from .parameters import (
    load_megatron_checkpoint, trainable_param_names,
    build_fwd_layer_params, build_bwd_precomputed,
)
from .kernels import precompute_rope_freqs
from .forward import forward_pass

# (Fused multi-tensor kernels are imported and used inside .optimizer; this
# module no longer needs to reference them directly after the migration.)


# ---------------------------------------------------------------------------
# Forward-only entry point
# ---------------------------------------------------------------------------
def run_forward(
    checkpoint_root: str,
    batch: dict,
    device: str,
) -> Tuple[float, torch.Tensor]:
    """Forward-only entry: 32-layer, single process.

    Returns (scalar_loss, per_token_losses).
    per_token_losses shape/dtype must match baseline reference.
    """
    params = load_megatron_checkpoint(checkpoint_root, device)
    rope_cos, rope_sin = precompute_rope_freqs(config.MAX_SEQ_LENGTH, device)

    input_ids = batch["tokens"]      # [B, S]
    labels = batch["labels"]         # [B, S]
    loss_mask = batch["loss_mask"]   # [B, S]

    with torch.no_grad():
        logits, _ = forward_pass(params, input_ids, rope_cos, rope_sin)
    from .backward import compute_ce_loss_forward_only
    loss_val, per_token = compute_ce_loss_forward_only(logits, labels, loss_mask)

    return loss_val, per_token


# ---------------------------------------------------------------------------
# Forward + manual backward entry point
# ---------------------------------------------------------------------------
def run_forward_backward(
    checkpoint_root: str,
    batch: dict,
    device: str,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Forward+backward entry: 32-layer.

    Uses a fully manual backward pass (no autograd graph for d_hidden
    propagation) to match Megatron's backward exactly.  Attention backward
    uses local autograd through TE DPA.

    Returns (scalar_loss, {megatron_param_name: FP32_gradient}).
    """
    from .backward import compute_ce_loss_and_grad, manual_backward, to_canonical_grad_name, has_fused_wgrad

    params = load_megatron_checkpoint(checkpoint_root, device)
    rope_cos, rope_sin = precompute_rope_freqs(config.MAX_SEQ_LENGTH, device)

    input_ids = batch["tokens"]
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]

    with torch.no_grad():
        logits, saved = forward_pass(
            params, input_ids, rope_cos, rope_sin,
            save_for_backward=True,
        )

    loss_t, d_logits = compute_ce_loss_and_grad(logits, labels, loss_mask)
    loss_val = loss_t.item()

    fp32_grads = manual_backward(params, saved, d_logits, rope_cos, rope_sin)

    from .optimizer import compute_grad_norm_fp32, clip_gradients_fp32, CLIP_GRAD

    names = trainable_param_names()
    total_norm = compute_grad_norm_fp32(fp32_grads, names, device)

    clip_gradients_fp32(fp32_grads, names, CLIP_GRAD, total_norm, device)

    gradients: Dict[str, torch.Tensor] = {}
    for name, grad in fp32_grads.items():
        canonical_name = to_canonical_grad_name(name)
        gradients[canonical_name] = grad.float()

    return loss_val, gradients


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def run_training(
    checkpoint_root: str,
    batches,
    num_steps: int,
    device: str,
    world_size: int = 1,
    rank: int = 0,
    tp_size: int = 1,
    *,
    params_override: Optional[Dict[str, torch.Tensor]] = None,
    optimizer_state_in: Optional[Dict[str, object]] = None,
    state_out: Optional[Dict[str, object]] = None,
    start_step: int = 0,
    grad_accum_steps: int = 1,
) -> List[dict]:
    """Full training loop: forward → CE loss → backward → grad norm →
    clip → Adam → param sync, for num_steps steps.

    Returns list of {loss: float, grad_norm: float, step_time: float}.

    ``batches`` may be either a ``list[dict]`` (gate-style pre-pump,
    keeps backward compat with all existing entry scripts) OR any
    iterable / generator yielding ``num_steps`` batch dicts on demand
    (production-style streaming dataloader, no GPU-resident pre-pump).
    Internally both paths route through ``iter()`` + ``next()``.

    ``params_override`` (kw-only): if not None, uses this pre-built param
    dict instead of calling ``load_megatron_checkpoint(checkpoint_root,
    ...)``.  The caller is responsible for passing a dict with the right
    keys (matching ``trainable_param_names()`` for the configured TP) and
    bf16 dtype on ``device``.  Default behavior — load from the canonical
    checkpoint — is unchanged.

    ``optimizer_state_in`` (kw-only): if not None, restore Adam state from
    this dict (keys: fp32_master_shard, exp_avg_shard, exp_avg_sq_shard,
    step_count).  Used for resume training (M1).

    ``state_out`` (kw-only): if not None, populated at the end of training
    with ``{"optimizer_state": {...}}`` so the caller can persist it via
    ``save_checkpoint``.

    ``start_step`` (kw-only): global step offset for the LR scheduler.
    Phase B of resume training passes ``save_step`` here so that
    ``compute_lr(start_step + local_step)`` produces the correct LR
    for the resumed portion of the trajectory.

    ``grad_accum_steps`` (kw-only): number of micro-batches per effective
    optimizer step.  When > 1, the iterator must yield
    ``num_steps * grad_accum_steps`` batch dicts.  Each micro-batch's
    ``d_logits`` is scaled by ``1/grad_accum_steps`` before backward so
    that the accumulated gradients equal the mean over micro-batches
    (matching the standard ``loss / num_micro_batches`` reference).
    """
    if world_size > 1:
        return _run_training_distributed(
            checkpoint_root, batches, num_steps, device,
            world_size, rank, tp_size,
            params_override=params_override,
            optimizer_state_in=optimizer_state_in,
            state_out=state_out,
            start_step=start_step,
            grad_accum_steps=grad_accum_steps,
        )
    return _run_training_single(
        checkpoint_root, batches, num_steps, device,
        params_override=params_override,
        start_step=start_step,
        grad_accum_steps=grad_accum_steps,
    )


def _run_training_single(
    checkpoint_root: str,
    batches,
    num_steps: int,
    device: str,
    *,
    params_override: Optional[Dict[str, torch.Tensor]] = None,
    start_step: int = 0,
    grad_accum_steps: int = 1,
) -> List[dict]:
    """Single-GPU training loop (M3).

    ``batches`` accepts either a list (gate-style pre-pump) OR any
    iterable / generator that yields one per-step batch dict (production-
    style streaming live dataloader).  Both are routed through ``iter()``
    + ``next()`` so the inner loop is unchanged.
    """
    from .backward import compute_ce_loss_and_grad, manual_backward
    from .optimizer import (
        AdamState,
        adam_step,
        clip_gradients_fp32,
        compute_grad_norm_fp32,
        compute_lr,
        sync_params_from_master,
        CLIP_GRAD,
    )

    gas = grad_accum_steps
    if params_override is not None:
        params = params_override
    else:
        params = load_megatron_checkpoint(checkpoint_root, device)
    rope_cos, rope_sin = precompute_rope_freqs(config.MAX_SEQ_LENGTH, device)

    names = trainable_param_names()
    opt_state = AdamState(names, params, device)

    results: List[dict] = []
    _batch_iter = iter(batches)

    for local_step in range(num_steps):
        t0 = time.time()
        global_step = start_step + local_step
        lr = compute_lr(global_step)

        loss_acc = 0.0
        fp32_grads = None

        for micro_step in range(gas):
            batch = next(_batch_iter)

            with torch.no_grad():
                logits, saved = forward_pass(
                    params, batch["tokens"], rope_cos, rope_sin,
                    save_for_backward=True,
                )

            loss_t, d_logits = compute_ce_loss_and_grad(
                logits, batch["labels"], batch["loss_mask"],
            )
            loss_acc += loss_t.item()

            micro_grads = manual_backward(
                params, saved, d_logits, rope_cos, rope_sin,
            )
            del logits, saved, d_logits

            if fp32_grads is None:
                fp32_grads = micro_grads
            else:
                for name in fp32_grads:
                    fp32_grads[name].add_(micro_grads[name])
                del micro_grads

        if gas > 1:
            for name in fp32_grads:
                fp32_grads[name].div_(gas)

        loss_val = loss_acc / gas
        total_norm = compute_grad_norm_fp32(fp32_grads, names, device)

        clip_gradients_fp32(fp32_grads, names, CLIP_GRAD, total_norm, device)
        adam_step(opt_state, fp32_grads, names, lr)
        sync_params_from_master(params, opt_state, names)

        dt = time.time() - t0
        results.append({
            "loss": loss_val,
            "grad_norm": total_norm,
            "step_time": dt,
        })

        print(
            f"  [engine] step {global_step}: loss={loss_val:.15e} "
            f"grad_norm={total_norm:.15f} lr={lr:.6e} ({dt:.2f}s)"
        )

    return results


def _run_training_distributed(
    checkpoint_root: str,
    batches,
    num_steps: int,
    device: str,
    world_size: int,
    rank: int,
    tp_size: int,
    *,
    params_override: Optional[Dict[str, torch.Tensor]] = None,
    optimizer_state_in: Optional[Dict[str, object]] = None,
    state_out: Optional[Dict[str, object]] = None,
    start_step: int = 0,
    grad_accum_steps: int = 1,
) -> List[dict]:
    """Multi-GPU distributed training loop.

    ``batches`` accepts either a list (gate-style pre-pump) OR any
    iterable / generator that yields one per-step batch dict (production-
    style streaming live dataloader).  Both are routed through ``iter()``
    + ``next()`` so the inner loop is unchanged.

    DP gradient communication via reduce-scatter / all-gather.
    Distributed optimizer: each DP rank owns a shard of the params.
    For TP>1: TP-sharded checkpoint, TP-parallel forward/backward,
    vocab-parallel CE loss.
    Optimizations: saved autograd for attention (no DPA re-computation in
    backward), direct wgrad accumulation into contiguous buffer.
    """
    from .backward import (
        compute_ce_loss_and_grad,
        compute_ce_loss_and_grad_tp,
        fused_ce_loss_and_grad_tp,
        manual_backward,
    )
    from .optimizer import (
        DistributedAdamState,
        compute_lr,
        distributed_adam_step,
        distributed_adam_step_bucket,
        compute_distributed_grad_norm,
        clip_shard_grads,
        multi_tensor_applier,
        multi_tensor_l2norm,
        validate_backends,
        CLIP_GRAD,
    )
    from .nccl import (
        BufferLayout,
        init_process_groups,
        get_dp_group,
        get_dp_rank,
        get_dp_size,
        get_tp_group,
        get_tp_rank,
    )

    init_process_groups(rank, world_size, tp_size)
    validate_backends()
    dp_group = get_dp_group()
    dp_rank = get_dp_rank()
    dp_size = get_dp_size()
    tp_group_val = get_tp_group() if tp_size > 1 else None
    tp_rank_val = get_tp_rank()

    gas = grad_accum_steps

    if rank == 0:
        print_config_summary()
        print(f"  [engine] Distributed: world={world_size} TP={tp_size} "
              f"DP={dp_size} dp_rank={dp_rank} tp_rank={tp_rank_val}"
              f"{f' GAS={gas}' if gas > 1 else ''}")

    if params_override is not None:
        params = params_override
        if rank == 0:
            print(
                f"  [engine] Using params_override (skipping "
                f"load_megatron_checkpoint); {len(params)} keys",
                flush=True,
            )
    else:
        params = load_megatron_checkpoint(
            checkpoint_root, device,
            tp_rank=tp_rank_val, tp_size=tp_size,
        )
    rope_cos, rope_sin = precompute_rope_freqs(config.MAX_SEQ_LENGTH, device)

    cfg = get_config()
    names = trainable_param_names()
    _NUM_RS_BUCKETS = cfg.rs_buckets
    layout = BufferLayout(names, params, dp_size, num_rs_buckets=_NUM_RS_BUCKETS)
    opt_state = DistributedAdamState(layout, params, dp_rank, device)
    if cfg.adam_ag_pipeline:
        opt_state.prepare_bucket_adam(layout, _NUM_RS_BUCKETS)

    if optimizer_state_in is not None:
        opt_state.fp32_master_shard.copy_(optimizer_state_in["fp32_master_shard"])
        opt_state.exp_avg_shard.copy_(optimizer_state_in["exp_avg_shard"])
        opt_state.exp_avg_sq_shard.copy_(optimizer_state_in["exp_avg_sq_shard"])
        opt_state.step_count = int(optimizer_state_in["step_count"])
        del optimizer_state_in
        if rank == 0:
            print(f"  [engine] Restored optimizer state (step_count={opt_state.step_count})")

    if rank == 0:
        print(f"  [engine] Buffer: total={layout.total_numel} "
              f"shard={layout.shard_size} params_in_shard="
              f"{len(opt_state.shard_param_slices)}")

    overflow_buf = torch.tensor([0], dtype=torch.int, device=device)
    norm_sq_buf = torch.zeros(1, dtype=torch.float32, device=device)
    loss_tensor = torch.zeros(1, dtype=torch.float32, device=device)
    grad_shard = torch.empty(layout.shard_size, dtype=torch.float32, device=device)

    # Bind grad_shard to opt_state for pre-computed slice views
    opt_state.bind_grad_shard(grad_shard, tp_rank=tp_rank_val)

    shard_start = dp_rank * layout.shard_size
    use_tp = tp_size > 1

    # Pre-allocate grad buffer for direct wgrad accumulation
    grad_buffer = torch.zeros(layout.total_numel, dtype=torch.float32, device=device)

    # Pre-allocate contiguous BF16 param buffer for fused sync
    bf16_param_buf = torch.empty(layout.total_numel, dtype=torch.bfloat16, device=device)
    # Reuse grad_buffer storage for BF16 shard conversion (safe: only used after
    # reduce_scatter is done with grad_buffer, before next step's zero_())
    bf16_shard_view = grad_buffer.view(torch.bfloat16)[:layout.shard_size]
    for name in layout.buffer_order:
        offset = layout.param_offsets[name]
        numel = layout.param_numels[name]
        bf16_param_buf[offset:offset + numel].copy_(params[name].view(-1))
        params[name] = bf16_param_buf[offset:offset + numel].view(params[name].shape)

    # Streams for overlapping zero_() with CE and reusing wgrad stream.
    import os as _ablate_os
    _force_wgrad = _ablate_os.environ.get("STREAM_FORCE_WGRAD_ON_MAIN", "0") == "1"
    _force_rs    = _ablate_os.environ.get("STREAM_FORCE_RS_ON_MAIN",    "0") == "1"
    _force_ag    = _ablate_os.environ.get("STREAM_FORCE_AG_ON_MAIN",    "0") == "1"
    _main = torch.cuda.current_stream()
    zero_stream  = torch.cuda.Stream()
    wgrad_stream = _main if _force_wgrad else torch.cuda.Stream()
    rs_stream    = _main if _force_rs    else torch.cuda.Stream()
    ag_stream    = _main if _force_ag    else torch.cuda.Stream()
    if _force_wgrad or _force_rs or _force_ag:
        print(
            f"[stream-ablate] WGRAD_ON_MAIN={int(_force_wgrad)} "
            f"RS_ON_MAIN={int(_force_rs)} AG_ON_MAIN={int(_force_ag)}",
            flush=True,
        )
    _layers_per_bucket = config.NUM_LAYERS // _NUM_RS_BUCKETS

    # Pre-allocate CUDA events and buffer views for RS/AG to avoid
    # per-step object creation overhead.
    _rs_main_events = [torch.cuda.Event() for _ in range(_NUM_RS_BUCKETS)]
    _rs_grad_views = []
    _rs_shard_views = []
    for _bi in range(_NUM_RS_BUCKETS):
        _bs, _be = layout.bucket_ranges[_bi]
        _bsh = (_be - _bs) // dp_size
        _so = layout.bucket_shard_offsets[_bi]
        _rs_grad_views.append(grad_buffer[_bs:_be])
        _rs_shard_views.append(grad_shard[_so:_so + _bsh])

    # Pre-allocate wgrad events for backward bucket callbacks
    _wgrad_events = [torch.cuda.Event() for _ in range(_NUM_RS_BUCKETS)]

    # Pre-allocate a single reusable event for wgrad_stream synchronization.
    # Replaces per-call wait_stream() which internally creates+destroys a
    # CUDA event every invocation (~2µs overhead × 1024 calls/step = ~2ms).
    _wgrad_sync_event = torch.cuda.Event()

    _ag_events_pool = [torch.cuda.Event() for _ in range(_NUM_RS_BUCKETS)]
    _ag_copy_event = torch.cuda.Event()
    _ag_param_views = []
    _ag_shard_views = []
    _ag_fp32_shard_views = []
    for _bi in range(_NUM_RS_BUCKETS):
        _bs, _be = layout.bucket_ranges[_bi]
        _bsh = (_be - _bs) // dp_size
        _so = layout.bucket_shard_offsets[_bi]
        _ag_param_views.append(bf16_param_buf[_bs:_be])
        _ag_shard_views.append(bf16_shard_view[_so:_so + _bsh])
        _ag_fp32_shard_views.append(opt_state.fp32_master_shard[_so:_so + _bsh])

    _rs_divisor = float(dp_size)

    def _on_bucket_ready(bucket_idx, wgrad_event):
        """Launch async reduce-scatter for one bucket on rs_stream.

        The DP-average division is deferred to a single grad_shard.div_()
        after all RS completes, removing 16 per-bucket FP32 div kernels
        from the RS stream critical path.
        """
        _rs_main_events[bucket_idx].record()
        rs_stream.wait_event(_rs_main_events[bucket_idx])
        if wgrad_event is not None:
            rs_stream.wait_event(wgrad_event)
        with torch.cuda.stream(rs_stream):
            torch.distributed.reduce_scatter_tensor(
                _rs_shard_views[bucket_idx],
                _rs_grad_views[bucket_idx],
                op=torch.distributed.ReduceOp.SUM,
                group=dp_group,
            )

    # Pre-compute forward-consumption AG order: forward needs the last
    # backward bucket first (embedding + layer 0..k), so AG in reverse.
    _ag_fwd_order = list(range(_NUM_RS_BUCKETS - 1, -1, -1))

    # Pre-compute per-layer data once (avoids rebuilding 32-layer param
    # tuples, buffer views, and name strings on every micro-batch call).
    _fwd_layer_params = build_fwd_layer_params(params, config.NUM_LAYERS)
    _bwd_precomputed = build_bwd_precomputed(
        params, config.NUM_LAYERS, grad_buffer=grad_buffer, layout=layout,
    )

    results: List[dict] = []
    ag_bucket_events: dict = {}
    _batch_iter = iter(batches)

    _use_ag_fwd_overlap = cfg.ag_fwd_overlap
    _use_adam_ag_pipeline = cfg.adam_ag_pipeline
    _use_gpu_norm_clip = cfg.gpu_norm_clip
    _use_fused_ce = cfg.fused_ce and use_tp
    _use_defer_wgrad = cfg.defer_wgrad_sync and gas > 1
    _ce_grad_scale = (1.0 / gas) if gas > 1 else 1.0

    for local_step in range(num_steps):
        t0 = time.time()
        global_step = start_step + local_step
        lr = compute_lr(global_step)
        do_profile = (rank == 0 and global_step in (52, 53, 54, 200, 400))

        # GPU-side loss accumulation: avoid per-micro-batch .item() syncs.
        # loss_tensor is reused as accumulator, zeroed at the start of each
        # effective step.  Final .item() is deferred to after Adam+AG.
        loss_tensor.zero_()
        _profile_micros = [] if do_profile else None

        for micro_step in range(gas):
            batch = next(_batch_iter)
            _is_first_micro = (micro_step == 0)
            _is_last_micro = (micro_step == gas - 1)

            if do_profile:
                torch.cuda.synchronize()
                _tp = time.time()

            _ag_ev_for_fwd = (
                ag_bucket_events
                if _use_ag_fwd_overlap and _is_first_micro
                else None
            )

            with torch.no_grad():
                logits, saved = forward_pass(
                    params, batch["tokens"], rope_cos, rope_sin,
                    save_for_backward=True,
                    tp_group=tp_group_val, tp_rank=tp_rank_val, tp_size=tp_size,
                    ag_events=_ag_ev_for_fwd,
                    ag_layers_per_bucket=_layers_per_bucket,
                    ag_num_buckets=_NUM_RS_BUCKETS,
                    _precomputed_layer_params=_fwd_layer_params,
                )

            if do_profile:
                torch.cuda.synchronize()
                _t_fwd_i = time.time() - _tp; _tp = time.time()

            if _is_first_micro:
                if not _use_ag_fwd_overlap and ag_bucket_events is not None:
                    torch.cuda.current_stream().wait_stream(ag_stream)

                zero_stream.wait_stream(torch.cuda.current_stream())
                zero_stream.wait_stream(ag_stream)
                with torch.cuda.stream(zero_stream):
                    grad_buffer.zero_()

            if _use_fused_ce:
                loss_t_local, d_logits = fused_ce_loss_and_grad_tp(
                    logits, batch["labels"], batch["loss_mask"],
                    tp_group_val, tp_rank_val, tp_size,
                    grad_scale=_ce_grad_scale,
                )
            elif use_tp:
                loss_t_local, d_logits = compute_ce_loss_and_grad_tp(
                    logits, batch["labels"], batch["loss_mask"],
                    tp_group_val, tp_rank_val, tp_size,
                    grad_scale=_ce_grad_scale,
                )
            else:
                loss_t_local, d_logits = compute_ce_loss_and_grad(
                    logits, batch["labels"], batch["loss_mask"],
                    grad_scale=_ce_grad_scale,
                )
            del logits
            loss_tensor.add_(loss_t_local)

            if do_profile:
                torch.cuda.synchronize()
                _t_ce_i = time.time() - _tp; _tp = time.time()

            if _is_first_micro:
                torch.cuda.current_stream().wait_stream(zero_stream)

            manual_backward(
                params, saved, d_logits, rope_cos, rope_sin,
                tp_group=tp_group_val, tp_rank=tp_rank_val, tp_size=tp_size,
                grad_buffer=grad_buffer, layout=layout,
                wgrad_stream_ext=wgrad_stream,
                bucket_ready_fn=_on_bucket_ready if _is_last_micro else None,
                layers_per_bucket=_layers_per_bucket,
                wgrad_events=_wgrad_events if _is_last_micro else None,
                defer_embedding_bwd=_use_defer_wgrad,
                _precomputed=_bwd_precomputed,
                _wgrad_sync_event=_wgrad_sync_event,
            )
            del saved, d_logits, loss_t_local

            if do_profile:
                torch.cuda.synchronize()
                _t_bwd_i = time.time() - _tp
                _profile_micros.append((_t_fwd_i, _t_ce_i, _t_bwd_i))

        # ── Post-accumulation: RS sync → norm+clip → Adam → AG ──

        if do_profile:
            _tp = time.time()

        # Effective-step loss: average over micro-batches, then AR+avg
        # across DP ranks.
        loss_tensor.div_(gas)
        _loss_handle = torch.distributed.all_reduce(
            loss_tensor, op=torch.distributed.ReduceOp.SUM,
            group=dp_group, async_op=True,
        )

        torch.cuda.current_stream().wait_stream(rs_stream)

        if _use_gpu_norm_clip:
            # Compute L2 norm on the raw (undivided) shard, fold DP-average
            # division into the clip multiplier.  The actual grad_shard scaling
            # is deferred to per-bucket inside the Adam+AG pipeline loop so
            # each bucket's clip is hidden behind the previous bucket's AG
            # communication (eliminates one full-shard elementwise pass).
            if opt_state._norm_views:
                overflow_buf.zero_()
                _gn, _ = multi_tensor_applier(
                    multi_tensor_l2norm, overflow_buf,
                    [opt_state._norm_views], False,
                )
                norm_sq_buf.copy_(_gn.square())
            else:
                norm_sq_buf.fill_(0.0)
            torch.distributed.all_reduce(
                norm_sq_buf, op=torch.distributed.ReduceOp.SUM, group=None,
            )
            norm_sq_buf.sqrt_()
            norm_sq_buf.div_(_rs_divisor)
            _clip_scale = (CLIP_GRAD / (norm_sq_buf + 1e-6)).clamp_(max=1.0)
            _clip_scale.div_(_rs_divisor)
            if not _use_adam_ag_pipeline:
                grad_shard.mul_(_clip_scale)
        else:
            grad_shard.div_(_rs_divisor)
            total_norm = compute_distributed_grad_norm(
                grad_buffer, layout, dp_rank, opt_state, device,
                tp_rank=tp_rank_val, overflow_buf=overflow_buf,
                grad_shard=grad_shard,
            )
            clip_shard_grads(
                grad_shard, opt_state, CLIP_GRAD, total_norm, overflow_buf,
            )

        _loss_handle.wait()
        loss_tensor.div_(dp_size)

        if do_profile:
            torch.cuda.synchronize()
            _t_nc = time.time() - _tp; _tp = time.time()

        if _use_adam_ag_pipeline:
            opt_state.step_count += 1
            _step_count = opt_state.step_count
            overflow_buf.zero_()
            for _bi in _ag_fwd_order:
                if _use_gpu_norm_clip:
                    _rs_shard_views[_bi].mul_(_clip_scale)
                distributed_adam_step_bucket(
                    opt_state, _bi, lr, _step_count, overflow_buf,
                )
                _ag_shard_views[_bi].copy_(_ag_fp32_shard_views[_bi])
                _ag_copy_event.record()
                ag_stream.wait_event(_ag_copy_event)
                with torch.cuda.stream(ag_stream):
                    torch.distributed.all_gather_into_tensor(
                        _ag_param_views[_bi],
                        _ag_shard_views[_bi],
                        group=dp_group,
                    )
                    _ag_events_pool[_bi].record()
                    ag_bucket_events[_bi] = _ag_events_pool[_bi]
        else:
            distributed_adam_step(opt_state, grad_shard, lr, overflow_buf)
            bf16_shard_view.copy_(opt_state.fp32_master_shard)
            for _bi in _ag_fwd_order:
                torch.distributed.all_gather_into_tensor(
                    _ag_param_views[_bi],
                    _ag_shard_views[_bi],
                    group=dp_group,
                )

        if _use_gpu_norm_clip:
            loss_val = loss_tensor.item()
            total_norm = norm_sq_buf.item()
        else:
            loss_val = loss_tensor.item()

        if do_profile:
            torch.cuda.synchronize()
            _t_aa = time.time() - _tp

        dt = time.time() - t0
        results.append({
            "loss": loss_val,
            "grad_norm": total_norm,
            "step_time": dt,
        })

        if do_profile:
            if gas > 1:
                _totals = [f + c + b for f, c, b in _profile_micros]
                _fwds = [t[0] for t in _profile_micros]
                _ces = [t[1] for t in _profile_micros]
                _bwds = [t[2] for t in _profile_micros]
                print(
                    f"  [profile] step {global_step} per_micro: "
                    + " ".join(f"{t:.4f}" for t in _totals)
                )
                print(
                    f"  [profile] step {global_step} micro0: "
                    f"fwd={_fwds[0]:.4f} ce={_ces[0]:.4f} bwd={_bwds[0]:.4f}"
                )
                if gas > 2:
                    print(
                        f"  [profile] step {global_step} micro{gas-1}: "
                        f"fwd={_fwds[-1]:.4f} ce={_ces[-1]:.4f} bwd={_bwds[-1]:.4f}"
                    )
                _avg_f = sum(_fwds) / gas
                _avg_c = sum(_ces) / gas
                _avg_b = sum(_bwds) / gas
                print(
                    f"  [profile] step {global_step}: "
                    f"micro_avg[fwd={_avg_f:.4f} ce={_avg_c:.4f} bwd={_avg_b:.4f}={_avg_f+_avg_c+_avg_b:.4f}] "
                    f"norm_clip={_t_nc:.4f} adam_ag={_t_aa:.4f} total={dt:.4f} "
                    f"(GAS={gas})"
                )
            else:
                _f, _c, _b = _profile_micros[0]
                print(
                    f"  [profile] step {global_step}: fwd={_f:.4f} "
                    f"ce={_c:.4f} bwd_rs={_b:.4f} "
                    f"norm_clip={_t_nc:.4f} "
                    f"adam_ag={_t_aa:.4f} total={dt:.4f}"
                )

        if rank == 0:
            print(
                f"  [engine] step {global_step}: loss={loss_val:.15e} "
                f"grad_norm={total_norm:.15f} lr={lr:.6e} ({dt:.2f}s)"
            )

    if state_out is not None:
        state_out["optimizer_state"] = {
            "fp32_master_shard": opt_state.fp32_master_shard.cpu().clone(),
            "exp_avg_shard": opt_state.exp_avg_shard.cpu().clone(),
            "exp_avg_sq_shard": opt_state.exp_avg_sq_shard.cpu().clone(),
            "step_count": opt_state.step_count,
        }

    return results


