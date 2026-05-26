"""Training loop for MiniCPM4 0.5B.

Implements the core training loop that drives forward, backward, optimizer
step, and communication.  Called by the harness evaluation suites.

Supports both single-card (DP=1) and multi-card (DP>1) training.
DP communication uses torch.distributed via the nccl module.
"""

from __future__ import annotations

__all__ = ["run_training_loop"]

import math
import os
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from . import config
from .backward import backward_pass, cross_entropy_loss_backward, mtp_backward_pass
from .engine_config import EngineConfig, get_config
from .forward import mtp_forward_pass_with_save
from .forward_graph import (
    graphed_forward_pass_with_save,
    is_forward_cuda_graph_enabled,
)
from .kernels import precompute_rope_freqs
from .nccl import (
    BucketedGradReducer,
    allgather_grads,
    baseline_buffer_layout,
    compute_distributed_grad_norm,
    get_comm_stream,
    get_rank,
    is_distributed,
    reduce_scatter_grads,
)
from .optimizer import (
    AdamState,
    ShardedOptimizerBuffers,
    adam_step,
    clip_gradients_fp32,
    compute_grad_norm_fp32,
    compute_lr,
    fused_clip_adam_sync,
    fused_clip_adam_sync_bucketed,
    sharded_fused_clip_adam_sync,
    sharded_fused_clip_adam_sync_multi_tensor,
    sync_params_from_master,
)
from .parameters import (
    load_megatron_checkpoint,
    load_megatron_optimizer_state,
    save_checkpoint,
    should_save_after_step,
    trainable_param_names,
    wait_for_async_save,
)
from .profiling import (
    DEEP_PHASE_ACCUM,
    DEEP_PHASE_BWD,
    DEEP_PHASE_CE,
    DEEP_PHASE_DATA,
    DEEP_PHASE_FWD,
    SEG_ALLGATHER,
    SEG_GRAD_NORM,
    SEG_LOSS_ALLREDUCE,
    SEG_MICROBATCHES,
    SEG_OPTIMIZER,
    SEG_REDUCE_SCATTER,
    SEG_STEP_START,
    StepProfiler,
)
from .step_graph import (
    graphed_compute_step,
    is_step_cuda_graph_enabled,
)


def _load_or_resume_params(
    *,
    resume_state: dict | None,
    checkpoint_root: str,
    device: str,
    random_init: bool,
    dp_rank: int,
    t_names: list[str],
) -> tuple[dict[str, torch.Tensor], AdamState]:
    """Load model params and optimizer state from checkpoint or resume dict."""
    if resume_state is not None:
        for required_key in ("params", "optimizer_state"):
            if required_key not in resume_state:
                raise KeyError(
                    f"run_training_loop: resume_state missing required key "
                    f"{required_key!r}"
                )
        params = resume_state["params"]
        opt_state = AdamState(t_names, params, device=device)
        opt_state.load_state_dict(resume_state["optimizer_state"])
        sync_params_from_master(params, opt_state)
        if dp_rank == 0:
            print(
                f"  [run_training_loop] resumed from "
                f"step={int(resume_state.get('step', 0))} "
                f"num_samples={opt_state.num_samples} "
                f"step_count={opt_state.step_count} "
                f"params={len(params)}",
                flush=True,
            )
    else:
        params = load_megatron_checkpoint(
            checkpoint_root, device=device, random_init=random_init,
        )
        opt_state = AdamState(t_names, params, device=device)
        mg_opt_state = load_megatron_optimizer_state(
            checkpoint_root, device=device,
        )
        if mg_opt_state is not None:
            opt_state.load_state_dict(mg_opt_state, strict=False)
            sync_params_from_master(params, opt_state)
            if dp_rank == 0:
                print(
                    f"  [run_training_loop] rehydrated Adam state from "
                    f"Megatron distcp: step_count={opt_state.step_count}, "
                    f"num_samples={opt_state.num_samples}",
                    flush=True,
                )
    return params, opt_state


def _setup_grad_communication(
    *,
    cfg: EngineConfig,
    use_dp: bool,
    world_size: int,
    dp_rank: int,
    device: str,
    num_local_microbatches: int,
) -> tuple[
    list[tuple[str, int]] | None,
    int,
    BucketedGradReducer | None,
    bool,
    ShardedOptimizerBuffers | None,
    bool,
]:
    """Set up gradient communication infrastructure.

    Returns (buf_entries, buf_size, grad_reducer, use_sharded_opt,
    sharded_opt_bufs, use_step_graph, use_step_graph_reducer).
    """
    buf_entries: list[tuple[str, int]] | None = None
    buf_size = 0
    if use_dp:
        buf_entries = baseline_buffer_layout()
        buf_total = sum(sz for _, sz in buf_entries)
        bucket_align = math.lcm(world_size, 128)
        buf_size = int(math.ceil(buf_total / bucket_align) * bucket_align)

    # P2-A: wgrad-allreduce overlap now legal across all grad-accum
    # regimes. When ``num_local_microbatches > 1`` the reducer
    # ``copy_``-s the first MB into the bucket buffer and ``add_``-s
    # subsequent MBs, deferring ``reduce_scatter`` until the final MB
    # completes each bucket. This wipes out the per-MB dict-of-FP32
    # accumulation (no extra allocs, no per-param FP32 cast on the host)
    # AND keeps the comm-stream overlap that the single-MB path
    # already enjoyed. Step CUDA Graph remains gated on
    # ``num_local_microbatches == 1`` separately (host op inside the
    # write_grad callback is not capture-safe).
    use_wgrad_overlap = cfg.wgrad_overlap and use_dp
    grad_reducer: BucketedGradReducer | None = None
    if use_wgrad_overlap:
        # P2-B grad-accum overlap: only meaningful when there are
        # actually >1 microbatches AND the user opted in via the
        # ``ACCUM_STREAM_OVERLAP`` env (cfg.accum_stream_overlap).
        # Default = None preserves the P2-A behaviour bytewise.
        accum_stream: "torch.cuda.Stream | None" = None
        if cfg.accum_stream_overlap and num_local_microbatches > 1:
            accum_stream = torch.cuda.Stream(device=device)
        grad_reducer = BucketedGradReducer(
            buf_entries=buf_entries,
            world_size=world_size,
            rank=get_rank(),
            device=device,
            bucket_target_elems=cfg.wgrad_bucket_target_elems,
            num_local_microbatches=num_local_microbatches,
            accum_stream=accum_stream,
        )
        if dp_rank == 0:
            print(
                f"  [run_training_loop] wgrad-allreduce overlap: ENABLED "
                f"(buckets={grad_reducer.num_buckets}, "
                f"bucket_target_elems={cfg.wgrad_bucket_target_elems}, "
                f"num_local_microbatches={num_local_microbatches}, "
                f"accum_stream_overlap="
                f"{'ENABLED' if accum_stream is not None else 'disabled'})",
                flush=True,
            )

    use_sharded_opt = (
        cfg.sharded_optimizer
        and grad_reducer is not None
        and cfg.fuse_adam_sync
    )
    sharded_opt_bufs: ShardedOptimizerBuffers | None = None
    if use_sharded_opt:
        sharded_opt_bufs = ShardedOptimizerBuffers(grad_reducer, device=device)
        if dp_rank == 0:
            print(
                f"  [run_training_loop] sharded optimizer: ENABLED "
                f"(bf16_shards={len(sharded_opt_bufs.bf16_shards)}, "
                f"bf16_full_buf={sharded_opt_bufs.bf16_full_buf.numel()})",
                flush=True,
            )

    if dp_rank == 0:
        print(
            f"  [run_training_loop] forward CUDA Graph: "
            f"{'ENABLED' if is_forward_cuda_graph_enabled() else 'disabled'}",
            flush=True,
        )

    # Direction-C path-1 (step_cuda_graph + reducer coexistence).
    # Highest-priority path: when both ``step_cuda_graph`` and
    # ``step_cuda_graph_reducer`` are on AND the reducer is alive,
    # the train loop captures backward with the reducer's device-side
    # grad_sink (single in-place ``add_`` per param). This requires
    # neither the accum>1 dict bypass nor the legacy ``write_grad``
    # state machine — bucket buffers are zeroed at the start of each
    # step and ``reduce_scatter`` is dispatched once after the final
    # MB. Mutually exclusive with ``use_step_graph`` below.
    use_step_graph_reducer = (
        is_step_cuda_graph_enabled()
        and cfg.step_cuda_graph_reducer
        and grad_reducer is not None
        and not config.mtp_enabled()
    )

    # Direction-C path-0 (PoC, dict-accum coexistence). The legacy
    # accum>1 gate originally rejected this configuration because each
    # ``cell.replay()`` returns ``grads_dict`` referring to graph
    # buffers which get overwritten on the next replay. With
    # ``cfg.step_cuda_graph_accum=1`` the train loop explicitly
    # snapshots each MB's grad into the FP32 ``accumulated_grads``
    # dict between replays (BF16 → FP32 → add_), making accum>1 safe.
    # Requires ``grad_reducer is None`` (no wgrad_overlap).
    accum_ok = (
        num_local_microbatches == 1
        or cfg.step_cuda_graph_accum
    )
    use_step_graph = (
        is_step_cuda_graph_enabled()
        and not use_step_graph_reducer
        and grad_reducer is None
        and accum_ok
        and not config.mtp_enabled()
    )

    if use_step_graph_reducer and dp_rank == 0:
        print(
            "  [run_training_loop] step CUDA Graph + reducer coexistence "
            f"(M2-P32 path-1): ENABLED (accum={num_local_microbatches}; "
            f"fwd+CE+bwd captured with device-side grad_sink; bucket "
            f"buffers zeroed per step; reduce_scatter flushed after "
            f"final MB; NCCL/optimizer remain eager)",
            flush=True,
        )
    elif use_step_graph and dp_rank == 0:
        mode = "accum=1" if num_local_microbatches == 1 else (
            f"accum={num_local_microbatches} via STEP_CUDA_GRAPH_ACCUM "
            f"(host-side FP32 dict accumulation between replays)"
        )
        print(
            "  [run_training_loop] step CUDA Graph (M2-P32 Phase A): ENABLED "
            f"({mode}; fwd+CE+bwd captured; NCCL/optimizer remain eager)",
            flush=True,
        )
    elif is_step_cuda_graph_enabled() and dp_rank == 0:
        gate_msg = ""
        if grad_reducer is not None and not cfg.step_cuda_graph_reducer:
            gate_msg = (
                f"reducer present but STEP_CUDA_GRAPH_REDUCER=0; either "
                f"set STEP_CUDA_GRAPH_REDUCER=1 (path-1) or set "
                f"WGRAD_OVERLAP=0 (path-0/PoC)"
            )
        elif grad_reducer is None and cfg.step_cuda_graph_reducer:
            gate_msg = (
                f"STEP_CUDA_GRAPH_REDUCER=1 requires a live reducer "
                f"(set WGRAD_OVERLAP=1)"
            )
        else:
            gate_msg = (
                f"requires WGRAD_OVERLAP=0 (got grad_reducer="
                f"{'present' if grad_reducer is not None else 'None'})"
            )
            if num_local_microbatches > 1 and not cfg.step_cuda_graph_accum:
                gate_msg += (
                    f" and num_local_microbatches==1 "
                    f"(got {num_local_microbatches}; set "
                    f"STEP_CUDA_GRAPH_ACCUM=1 to opt into the accum>1 path)"
                )
        gate_msg += f"; MTP must be disabled (got mtp_num_layers={cfg.mtp_num_layers})"
        print(
            f"  [run_training_loop] step CUDA Graph requested but DISABLED: "
            f"{gate_msg}. Falling through to eager.",
            flush=True,
        )

    if config.mtp_enabled() and dp_rank == 0:
        print(
            f"  [run_training_loop] MTP enabled: "
            f"num_layers={cfg.mtp_num_layers}, "
            f"loss_scaling_factor={cfg.mtp_loss_scaling_factor} "
            f"(per-layer scale={config.mtp_per_layer_loss_scale():.6f})",
            flush=True,
        )

    return (buf_entries, buf_size, grad_reducer, use_sharded_opt,
            sharded_opt_bufs, use_step_graph, use_step_graph_reducer)


def _aggregate_mtp_losses(
    mb_mtp_losses: list[list[torch.Tensor]],
    *,
    cfg: EngineConfig,
    use_dp: bool,
    world_size: int,
    device: str,
) -> tuple[list[float], float]:
    """Average MTP losses across microbatches and DP ranks.

    Returns (mtp_loss_per_layer_global, mtp_loss_total).
    """
    mtp_loss_per_layer_local: list[float] = []
    mtp_loss_total_local = 0.0
    if config.mtp_enabled() and mb_mtp_losses:
        num_layers = cfg.mtp_num_layers
        for layer_idx in range(num_layers):
            vals = [
                mb[layer_idx].item() for mb in mb_mtp_losses
                if layer_idx < len(mb)
            ]
            if vals:
                mtp_loss_per_layer_local.append(sum(vals) / len(vals))
        if mtp_loss_per_layer_local:
            mtp_loss_total_local = (
                sum(mtp_loss_per_layer_local) / len(mtp_loss_per_layer_local)
            )
    mtp_loss_per_layer_global: list[float] = list(mtp_loss_per_layer_local)
    mtp_loss_total = mtp_loss_total_local
    if config.mtp_enabled() and use_dp and mtp_loss_per_layer_local:
        _mtp_t = torch.tensor(
            mtp_loss_per_layer_local,
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(_mtp_t, op=dist.ReduceOp.SUM)
        _mtp_t /= world_size
        mtp_loss_per_layer_global = _mtp_t.tolist()
        mtp_loss_total = sum(mtp_loss_per_layer_global) / len(mtp_loss_per_layer_global)
    return mtp_loss_per_layer_global, mtp_loss_total


def _maybe_save_checkpoint(
    *,
    step_idx: int,
    params: dict[str, torch.Tensor],
    opt_state: AdamState,
    save_dir: str | None,
    save_interval: int | None,
    use_sharded_opt: bool,
    grad_reducer: BucketedGradReducer | None,
    dataloader_state_provider: Callable[[str], dict | None] | None,
) -> None:
    """Persist a resume checkpoint if we are on a save boundary."""
    if not save_dir or not should_save_after_step(
        completed_step=step_idx, save_interval=save_interval,
    ):
        return
    if use_sharded_opt and grad_reducer is not None:
        grad_reducer.allgather_optimizer_state(opt_state)
    save_step = step_idx + 1
    step_save_dir = os.path.join(save_dir, f"step_{save_step:07d}")
    dl_state: dict | None = None
    if dataloader_state_provider is not None:
        dl_state = dataloader_state_provider(step_save_dir)
    save_checkpoint(
        checkpoint_dir=step_save_dir,
        params=params,
        optimizer_state=opt_state,
        step=save_step,
        dataloader_state=dl_state,
        async_io=True,
    )


def run_training_loop(
    *,
    checkpoint_root: str,
    step_indices: list[int],
    dp_rank: int = 0,
    world_size: int = 1,
    num_local_microbatches: int = 1,
    seed: int = 1234,
    device: str = "cuda:0",
    tokenized_inputs_path: str | None = None,
    data_iterator: Any | None = None,
    dump_inputs_dir: str | None = None,
    save_interval: int | None = None,
    save_dir: str | None = None,
    dataloader_state_provider: Callable[[str], dict | None] | None = None,
    resume_state: dict | None = None,
    random_init: bool = False,
    micro_batch_size: int = 10,
    profiler: StepProfiler | None = None,
) -> dict[str, Any]:
    """Run the MiniCPM4 0.5B training loop and return structured results.

    For DP>1, torch.distributed must be initialized before calling this.
    Gradient reduction uses the same buffer layout as the baseline's
    DistributedOptimizer for bitwise-exact all_reduce results when
    sharing the same NCCL communicator.

    Periodic resume checkpointing
    -----------------------------
    When ``save_interval`` is a positive integer **and** ``save_dir`` is
    set, the loop persists a resume checkpoint after every
    ``save_interval`` global steps via
    :func:`parameters.save_checkpoint` with ``async_io=True``. The
    background writer is single-writer (a second save inside the same
    process auto-waits for the first to finish), and a final
    :func:`parameters.wait_for_async_save` call right before returning
    guarantees the bg thread does not outlive ``run_training_loop`` —
    so callers never observe a half-written ``training_state.pt`` when
    the trainer process exits. Non-rank-0 ranks are no-ops on the save
    path so this is safe to call collectively from every rank.

    ``dataloader_state_provider``, when set, is invoked on each save
    boundary; its return value is opaque-passed into ``save_checkpoint``
    so dataloader cursor persistence can be wired in once the underlying
    SDK exposes a ``state_dict`` (currently a follow-up — see
    ``recommendation.md``).

    Cross-process resume
    --------------------
    When ``resume_state`` is supplied (typically the dict returned by
    :func:`parameters.load_resume_checkpoint`), the loop **skips** the
    canonical-checkpoint load and the fresh ``AdamState`` initialisation,
    and instead rehydrates ``params`` / ``optimizer_state`` directly from
    the snapshot. The caller is responsible for choosing
    ``step_indices`` consistent with ``resume_state['step']`` (typically
    ``range(resume_state['step'], num_steps)``); the loop honours
    whatever step indices are passed in so it stays a pure executor and
    does not silently truncate the trajectory the caller asked for.

    The dataloader cursor (``resume_state['dataloader_state']``) must be
    applied **before** the data iterator is wrapped into a prefetcher,
    so the train-loop assumes ``data_iterator`` already yields the next
    unconsumed batch on first call. This matches the call site in
    ``__main__.py`` where ``apply_dataloader_state`` runs prior to
    ``DataPrefetcher`` construction.
    """
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cfg = get_config()
    if not cfg.fused_ops:
        torch.use_deterministic_algorithms(True)

    use_dp = is_distributed() and world_size > 1

    t_names = trainable_param_names()
    params, opt_state = _load_or_resume_params(
        resume_state=resume_state,
        checkpoint_root=checkpoint_root,
        device=device,
        random_init=random_init,
        dp_rank=dp_rank,
        t_names=t_names,
    )
    rope_freqs = precompute_rope_freqs(config.MAX_SEQ_LENGTH, device=device)

    static_position_ids = None
    if data_iterator is not None:
        pass
    else:
        if tokenized_inputs_path is None:
            raise ValueError("tokenized_inputs_path is required when data_iterator is None")
        batch_data = torch.load(tokenized_inputs_path, map_location="cpu", weights_only=False)
        if use_dp:
            start = dp_rank * micro_batch_size
            end = start + micro_batch_size
            input_ids = batch_data["tokens"][start:end].to(device)
            labels = batch_data["labels"][start:end].to(device)
            loss_mask = batch_data["loss_mask"][start:end].float().to(device)
            if "position_ids" in batch_data:
                static_position_ids = batch_data["position_ids"][start:end].to(device)
        else:
            input_ids = batch_data["tokens"].to(device)
            labels = batch_data["labels"].to(device)
            loss_mask = batch_data["loss_mask"].float().to(device)
            if "position_ids" in batch_data:
                static_position_ids = batch_data["position_ids"].to(device)

    global_batch_size = num_local_microbatches * micro_batch_size * world_size

    steps_result: list[dict[str, Any]] = []

    (buf_entries, buf_size, grad_reducer, use_sharded_opt,
     sharded_opt_bufs, use_step_graph,
     use_step_graph_reducer) = _setup_grad_communication(
        cfg=cfg,
        use_dp=use_dp,
        world_size=world_size,
        dp_rank=dp_rank,
        device=device,
        num_local_microbatches=num_local_microbatches,
    )

    # Path-1 device-side grad sink: built once, after the reducer is
    # created; passed to ``graphed_compute_step`` so the captured
    # graph routes every parameter gradient directly into the
    # reducer's bucket buffer via a single in-place ``add_``.
    device_grad_sink: "Callable[[str, torch.Tensor], None] | None" = None
    if use_step_graph_reducer:
        assert grad_reducer is not None
        device_grad_sink = grad_reducer.make_device_grad_sink()

    loss_scale = 1.0 / num_local_microbatches

    if profiler is None:
        profiler = StepProfiler.disabled()
    host_step_ms_by_step: dict[int, float] = {}

    for step_idx in step_indices:
        torch.cuda.synchronize()
        t0 = time.time()
        profiler.begin_step(step_idx)
        profiler.mark(SEG_STEP_START)

        # When using the bucketed reducer, reset per-step pending counters
        # BEFORE backward so write_grad calls trigger reduce_scatter only
        # when the current step's slots are full (and not when a stale
        # counter already happens to be at zero). Buffer values are kept
        # across steps; only counters and completion events get reset.
        if grad_reducer is not None:
            grad_reducer.start_step()
            # Path-1: device-side grad sink uses ``add_`` on all MBs
            # (no copy_+add_ mix), so the bucket buffers must start
            # at zero each step. This is the only piece of state that
            # ``write_grad``'s legacy code relied on starting "dirty"
            # but the new sink doesn't tolerate.
            if use_step_graph_reducer:
                grad_reducer.zero_input_bufs()

        accumulated_grads: dict[str, torch.Tensor] | None = None
        mb_losses: list[torch.Tensor] = []
        # Per-microbatch MTP losses, indexed [microbatch][layer]. Used for
        # alignment-checking against Megatron's ``eagle_ce_loss`` /
        # MTPLossLoggingHelper.tracker (which averages MTP CE across layers,
        # microbatches, and DP ranks).
        mb_mtp_losses: list[list[torch.Tensor]] = []

        for _mb in range(num_local_microbatches):
            position_ids = None
            if data_iterator is not None:
                batch = next(data_iterator)
                if isinstance(batch, dict):
                    input_ids = batch["tokens"]
                    labels = batch["labels"]
                    loss_mask = batch["loss_mask"].float()
                    position_ids = batch.get("position_ids")
                elif isinstance(batch, (tuple, list)) and len(batch) >= 4:
                    input_ids, labels, loss_mask = batch[0], batch[1], batch[2].float()
                    position_ids = batch[3]
                else:
                    input_ids, labels = batch[0], batch[1]
                    loss_mask = batch[2].float() if len(batch) > 2 else torch.ones_like(batch[0], dtype=torch.float32)
            else:
                position_ids = static_position_ids
            profiler.mark_deep(DEEP_PHASE_DATA)
            profiler.mark_host(DEEP_PHASE_DATA)

            if dump_inputs_dir:
                os.makedirs(dump_inputs_dir, exist_ok=True)
                dump_payload = {
                    "tokens": input_ids.detach().cpu(),
                    "labels": labels.detach().cpu(),
                    "loss_mask": loss_mask.detach().cpu(),
                    "position_ids": (
                        position_ids.detach().cpu()
                        if position_ids is not None else None
                    ),
                    "step": int(step_idx),
                    "microbatch": int(_mb),
                    "rank": int(dp_rank),
                    "world_size": int(world_size),
                }
                torch.save(
                    dump_payload,
                    os.path.join(
                        dump_inputs_dir,
                        f"rank{dp_rank:05d}_step{int(step_idx):06d}_mb{_mb:03d}.pt",
                    ),
                )

            if use_step_graph or use_step_graph_reducer:
                # M2-P32 Phase A: forward + CE + backward collapsed into
                # a single captured graph.replay(). Two flavours:
                # * ``use_step_graph`` (path-0 / Phase A original):
                #   grad_sink=None; backward returns a grads dict; the
                #   train loop's accum logic snapshots BF16→FP32 between
                #   replays. Requires grad_reducer=None.
                # * ``use_step_graph_reducer`` (path-1 / Direction-C):
                #   grad_sink=device_grad_sink; gradients land directly
                #   in the reducer's bucket buffers via a single
                #   in-place ``add_``. ``param_grads`` returned dict is
                #   empty (no host-side accum work below). Requires
                #   grad_reducer != None and bucket buffers zeroed by
                #   ``grad_reducer.zero_input_bufs()`` above.
                # Profiler marks below collapse fwd/CE/bwd into the FWD
                # segment so the timeline schema stays a strict subset
                # of the eager path.
                if loss_mask is None:
                    raise RuntimeError(
                        "[run_training_loop] step CUDA Graph requires a "
                        "loss_mask in the batch; got None."
                    )
                if position_ids is None:
                    raise RuntimeError(
                        "[run_training_loop] step CUDA Graph requires "
                        "position_ids (doc-aware fused-RoPE branch); got None."
                    )
                loss, param_grads = graphed_compute_step(
                    params, input_ids, rope_freqs, position_ids=position_ids,
                    labels=labels, loss_mask=loss_mask, loss_scale=loss_scale,
                    grad_sink=device_grad_sink,
                )
                mb_losses.append(loss.detach())
                profiler.mark_deep(DEEP_PHASE_FWD)
                profiler.mark_host(DEEP_PHASE_FWD)
                profiler.mark_deep(DEEP_PHASE_CE)
                profiler.mark_host(DEEP_PHASE_CE)
                profiler.mark_deep(DEEP_PHASE_BWD)
                profiler.mark_host(DEEP_PHASE_BWD)
            else:
                logits, saved = graphed_forward_pass_with_save(
                    params, input_ids, rope_freqs, position_ids=position_ids,
                )

                # MTP forward (no-op when MTP_NUM_LAYERS == 0). Reuses
                # the main forward's cached ``normed_final`` (the
                # post-final-layernorm hidden state) as the MTP input —
                # exactly what Megatron's GPTModel passes to its
                # MultiTokenPredictionBlock.
                mtp_outputs: dict | None = None
                if config.mtp_enabled():
                    mtp_outputs = mtp_forward_pass_with_save(
                        params, input_ids, labels, loss_mask,
                        hidden_input=saved["normed_final"],
                        rope_freqs=rope_freqs,
                        position_ids=position_ids,
                        need_backward=True,
                    )
                profiler.mark_deep(DEEP_PHASE_FWD)
                profiler.mark_host(DEEP_PHASE_FWD)

                if cfg.fuse_ce:
                    from .triton_kernels import fused_cross_entropy_fwd_bwd
                    loss, _, d_logits = fused_cross_entropy_fwd_bwd(
                        logits, labels, loss_mask, loss_scale=loss_scale,
                    )
                else:
                    loss, _, d_logits = cross_entropy_loss_backward(
                        logits, labels, loss_mask, loss_scale=loss_scale,
                    )

                # MTP backward runs FIRST (before the main backward) so
                # we can fold its shared-weight contributions
                # (output.weight, tok_embeddings.weight) and its
                # ``d_normed_final`` into the main backward in a single
                # _emit per parameter — keeping the bucketed reducer's
                # single-write contract intact.
                extra_d_normed_final = None
                extra_param_grads: dict[str, torch.Tensor] | None = None
                if mtp_outputs is not None:
                    mtp_loss_scale = (
                        config.mtp_per_layer_loss_scale() * loss_scale
                    )
                    mtp_grads = mtp_backward_pass(
                        mtp_outputs, params, rope_freqs,
                        loss_scale=mtp_loss_scale,
                        grad_sink=(
                            grad_reducer.write_grad
                            if grad_reducer is not None else None
                        ),
                    )
                    extra_d_normed_final = mtp_grads["d_normed_final"]
                    extra_param_grads = mtp_grads["shared_grads"]
                    # Per-microbatch MTP losses (already scalar tensors,
                    # un-scaled mean CE per token — same convention
                    # Megatron uses for ``mtp_X loss`` tensorboard scalars
                    # via MTPLossLoggingHelper.save_loss_to_tracker).
                    mb_mtp_losses.append(list(mtp_grads["mtp_losses"]))

                mb_losses.append(loss.detach())
                profiler.mark_deep(DEEP_PHASE_CE)
                profiler.mark_host(DEEP_PHASE_CE)

                param_grads = backward_pass(
                    d_logits, params, saved, rope_freqs,
                    grad_sink=(
                        grad_reducer.write_grad if grad_reducer is not None else None
                    ),
                    extra_d_normed_final=extra_d_normed_final,
                    extra_param_grads=extra_param_grads,
                )
                # Fold MTP-only param grads back into the main grads
                # dict for the legacy (non-overlap) path. With the
                # bucketed reducer they were already written via
                # grad_sink so ``mtp_grads['param_grads']`` is empty.
                if mtp_outputs is not None and grad_reducer is None:
                    for n, g in mtp_grads["param_grads"].items():
                        if n in param_grads:
                            raise RuntimeError(
                                f"MTP backward emitted a grad for {n!r} that "
                                "was also produced by the main backward — "
                                "shared params should flow through "
                                "extra_param_grads, not param_grads."
                            )
                        param_grads[n] = g
                del saved, logits, d_logits
                if mtp_outputs is not None:
                    del mtp_outputs, mtp_grads
                profiler.mark_deep(DEEP_PHASE_BWD)
                profiler.mark_host(DEEP_PHASE_BWD)

            if grad_reducer is not None:
                # When the bucketed reducer is active, backward already
                # forwarded every gradient into the bucket buffers via
                # ``grad_sink``; ``param_grads`` is an empty dict and the
                # legacy "accumulate per-MB" code path must be skipped.
                accumulated_grads = None
            elif num_local_microbatches == 1:
                accumulated_grads = param_grads
            else:
                if accumulated_grads is None:
                    accumulated_grads = {n: param_grads[n].float() for n in param_grads}
                else:
                    for n in param_grads:
                        accumulated_grads[n].add_(param_grads[n].float())
                del param_grads
            profiler.mark_deep(DEEP_PHASE_ACCUM)
            profiler.mark_host(DEEP_PHASE_ACCUM)

        if len(mb_losses) == 1:
            _local_loss_t = mb_losses[0]
        else:
            _local_loss_t = torch.stack(mb_losses).to(torch.float64).mean().float()
        profiler.mark(SEG_MICROBATCHES)
        profiler.mark_host(SEG_MICROBATCHES)

        if use_dp:
            loss_tensor = _local_loss_t.to(dtype=torch.float64).reshape(1)
            if cfg.async_loss_ar:
                pass
            else:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                reported_loss = loss_tensor.item() / world_size
                profiler.mark(SEG_LOSS_ALLREDUCE)
                profiler.mark_host(SEG_LOSS_ALLREDUCE)

            if grad_reducer is not None:
                # Path-1 (step_graph + reducer): backward (now inside a
                # captured graph) wrote grads via ``device_grad_sink``;
                # it bypasses ``write_grad`` so no bucket-pending
                # counter ever fired. Dispatch all RS now, after the
                # final MB's replay completes, before ``wait_all``
                # records the done events.
                if use_step_graph_reducer:
                    grad_reducer.flush_all_buckets()
                # M2-P2 wgrad-allreduce overlap path. All per-bucket
                # ``reduce_scatter`` calls were already issued on the comm
                # stream the moment each bucket filled inside backward.
                # ``wait_all`` inserts the per-bucket completion events
                # into the default stream so subsequent default-stream ops
                # observe the reduced shards in deterministic order.
                grad_reducer.wait_all()
                profiler.mark(SEG_REDUCE_SCATTER)
                profiler.mark_host(SEG_REDUCE_SCATTER)

                grad_norm_full = grad_reducer.compute_distributed_grad_norm(
                    device,
                )
                profiler.mark(SEG_GRAD_NORM)
                profiler.mark_host(SEG_GRAD_NORM)

                if use_sharded_opt:
                    # M2-P18 ZeRO-1: skip FP32 allgather entirely. The
                    # per-shard optimizer will run on the reduce_scatter
                    # shards directly and issue a BF16 allgather instead.
                    # SEG_ALLGATHER is marked as a placeholder (0 ms);
                    # the actual BF16 allgather is included in
                    # SEG_OPTIMIZER timing.
                    fp32_grads = None
                    profiler.mark(SEG_ALLGATHER)
                    profiler.mark_host(SEG_ALLGATHER)
                elif cfg.opt_allgather_overlap and cfg.fuse_adam_sync:
                    # M2-P6 P1.c: per-bucket allgather + adam pipeline.
                    for _bidx in range(grad_reducer.num_buckets):
                        grad_reducer.start_bucket_allgather(_bidx)
                    profiler.mark(SEG_ALLGATHER)
                    profiler.mark_host(SEG_ALLGATHER)
                    fp32_grads = None
                else:
                    # Skip cloning when the fused Adam kernel is active:
                    # the Triton kernel only reads grad_ptr and never
                    # writes to it, so non-owning views are safe and
                    # avoid ~2 GB of transient allocation + memcpy.
                    fp32_grads = grad_reducer.allgather_and_unpack(
                        params, clone=not cfg.fuse_adam_sync,
                    )
                    profiler.mark(SEG_ALLGATHER)
                    profiler.mark_host(SEG_ALLGATHER)
            else:
                if cfg.grad_norm_diag and dp_rank == 0:
                    _diag_grads = {
                        n: (accumulated_grads[n].float()
                            if accumulated_grads[n].dtype != torch.float32
                            else accumulated_grads[n])
                        for n in t_names
                        if n in accumulated_grads
                    }
                    _diag_norm = compute_grad_norm_fp32(_diag_grads)
                    _diag_total = 0.0
                    _diag_rows = []
                    for _n, _g in _diag_grads.items():
                        _gn = _g.detach().float().norm().item()
                        _diag_total += _gn * _gn
                        _diag_rows.append((_gn, _n))
                    _diag_rows.sort(reverse=True)
                    print(
                        f"[grad-norm-diag] pre_reduce_rank0_full={_diag_norm:.10f} "
                        f"torch_norm={_diag_total ** 0.5:.10f}",
                        flush=True,
                    )
                    print(
                        "[grad-norm-top] "
                        + " | ".join(
                            f"{_gn:.6f}:{_n}" for _gn, _n in _diag_rows[:12]
                        ),
                        flush=True,
                    )
                fp32_shard, shard_size, _be = reduce_scatter_grads(
                    accumulated_grads, device,
                )
                del accumulated_grads
                profiler.mark(SEG_REDUCE_SCATTER)
                profiler.mark_host(SEG_REDUCE_SCATTER)

                grad_norm_full = compute_distributed_grad_norm(
                    fp32_shard, shard_size, _be, device,
                )
                profiler.mark(SEG_GRAD_NORM)
                profiler.mark_host(SEG_GRAD_NORM)

                fp32_grads = allgather_grads(
                    fp32_shard, buf_size, params, buf_entries, device,
                )
                del fp32_shard
                profiler.mark(SEG_ALLGATHER)
                profiler.mark_host(SEG_ALLGATHER)

            if cfg.async_loss_ar:
                _loss_ready_ev = torch.cuda.Event()
                _loss_ready_ev.record()
                _loss_comm = get_comm_stream(device)
                with torch.cuda.stream(_loss_comm):
                    _loss_comm.wait_event(_loss_ready_ev)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                profiler.mark(SEG_LOSS_ALLREDUCE)
                profiler.mark_host(SEG_LOSS_ALLREDUCE)

            grad_norm = grad_norm_full
        else:
            fp32_grads: dict[str, torch.Tensor] = {}
            for name in t_names:
                if name in accumulated_grads:
                    g = accumulated_grads[name]
                    fp32_grads[name] = g.float() if g.dtype != torch.float32 else g
            del accumulated_grads

            grad_norm_full = compute_grad_norm_fp32(fp32_grads)
            grad_norm = grad_norm_full

        lr = compute_lr(opt_state.num_samples, global_batch_size,
                        step_count=opt_state.step_count)

        if cfg.fuse_adam_sync:
            clip_coeff = min(1.0, cfg.max_grad_norm / (grad_norm_full + 1.0e-6))
            if use_sharded_opt and grad_reducer is not None:
                # M2-P18 ZeRO-1: per-shard Adam + BF16 allgather.
                # FORK-only (MULTI_TENSOR_ADAM=1): batch the per-param
                # Triton fused_adam_sync calls into a handful of
                # multi-tensor kernels (TE multi_tensor_scale +
                # multi_tensor_adam + torch._foreach_copy_) to slash
                # per-launch host overhead while preserving the same
                # AdamW + RNE-cast arithmetic.
                if cfg.multi_tensor_adam:
                    sharded_fused_clip_adam_sync_multi_tensor(
                        opt_state, grad_reducer, sharded_opt_bufs,
                        params, lr, clip_coeff,
                    )
                else:
                    sharded_fused_clip_adam_sync(
                        opt_state, grad_reducer, sharded_opt_bufs,
                        params, lr, clip_coeff,
                    )
            elif (
                cfg.opt_allgather_overlap
                and grad_reducer is not None
                and fp32_grads is None
            ):
                # M2-P6 P1.c: per-bucket allgather + adam pipeline.
                def _bucket_iter():
                    for _bidx in range(grad_reducer.num_buckets):
                        grad_reducer.wait_bucket_allgather(_bidx)
                        names = grad_reducer.bucket_param_names(_bidx)
                        grads = grad_reducer.unpack_bucket(_bidx, params)
                        yield names, grads
                fused_clip_adam_sync_bucketed(
                    opt_state, _bucket_iter(), params, lr, clip_coeff,
                )
            else:
                fused_clip_adam_sync(opt_state, fp32_grads, params, lr, clip_coeff)
            opt_state.num_samples += global_batch_size
        else:
            clip_gradients_fp32(fp32_grads, cfg.max_grad_norm, grad_norm_full)
            adam_step(opt_state, fp32_grads, lr)
            opt_state.num_samples += global_batch_size
            sync_params_from_master(params, opt_state)
        profiler.mark(SEG_OPTIMIZER)
        profiler.mark_host(SEG_OPTIMIZER)

        torch.cuda.synchronize()
        step_time = time.time() - t0
        # Resolve CUDA-event timings AFTER the sync that closes the step:
        # all profiler events on the default stream are now guaranteed
        # complete, so ``elapsed_time`` reads do not introduce additional
        # GPU sync. Disabled / out-of-range steps no-op.
        profiler.flush(step_idx)
        if profiler.enabled:
            host_step_ms_by_step[step_idx] = step_time * 1000.0

        local_loss = _local_loss_t.item()
        if use_dp and cfg.async_loss_ar:
            reported_loss = loss_tensor.item() / world_size
        elif not use_dp:
            reported_loss = local_loss

        mtp_loss_per_layer_global, mtp_loss_total = _aggregate_mtp_losses(
            mb_mtp_losses, cfg=cfg, use_dp=use_dp,
            world_size=world_size, device=device,
        )

        tokens_per_gpu = num_local_microbatches * micro_batch_size * config.MAX_SEQ_LENGTH
        flops = config.compute_training_flops(tokens_per_gpu)
        flops_std = config.compute_training_flops_standard(tokens_per_gpu)
        mfu = flops / (step_time * config.H100_PEAK_TFLOPS_BF16 * 1e12) * 100
        mfu_std = flops_std / (step_time * config.H100_PEAK_TFLOPS_BF16 * 1e12) * 100

        lr_next = compute_lr(opt_state.num_samples, global_batch_size,
                             step_count=opt_state.step_count)

        steps_result.append({
            "step": step_idx,
            "loss": reported_loss,
            "local_loss": local_loss,
            "grad_norm": grad_norm,
            "lr": lr_next,
            "total_step_time": step_time,
            "mfu_e2e": mfu,
            "mfu_e2e_standard": mfu_std,
            "mtp_loss": mtp_loss_total,
            "mtp_loss_per_layer": mtp_loss_per_layer_global,
        })

        if dp_rank == 0:
            mtp_suffix = ""
            if config.mtp_enabled() and mtp_loss_per_layer_global:
                mtp_suffix = (
                    " | mtp loss: "
                    + ", ".join(f"{v:.6e}" for v in mtp_loss_per_layer_global)
                )
            print(
                f" iteration {step_idx + 1:>6} | "
                f"loss: {reported_loss:.6e} | "
                f"local_loss: {local_loss:.6e} | "
                f"grad norm: {grad_norm:.3f} | "
                f"lr: {lr_next:.6e} | "
                f"time(ms): {step_time*1000:.1f}"
                f"{mtp_suffix}",
                flush=True,
            )
            # Gate-mode wire format. The default training run does NOT pay
            # the extra print (zero-cost off-path) — only the resume-mainloop
            # gate orchestrator opts in by exporting EMIT_LOSS_LINES=1, so
            # the relative-loss orchestrator in suites.py can parse a uniform
            # [LOSS] trajectory across the production CLI and the existing
            # baseline gate scripts. Format MUST stay in lockstep with
            # ``test_long_train.py`` and the ``_parse_loss_lines`` regex in
            # ``workload/infra/suites.py`` — drift would silently turn the
            # gate into a no-op.
            if cfg.emit_loss_lines:
                _mtp_loss_str = (
                    f" mtp_loss={mtp_loss_total:.10f}"
                    if config.mtp_enabled() else ""
                )
                print(
                    f"[LOSS] step={step_idx} global_loss={reported_loss:.10f} "
                    f"grad_norm={grad_norm:.10f} time_s={step_time:.6f} "
                    f"mfu_e2e_standard={mfu_std:.6f}{_mtp_loss_str}",
                    flush=True,
                )

        del fp32_grads

        _maybe_save_checkpoint(
            step_idx=step_idx,
            params=params,
            opt_state=opt_state,
            save_dir=save_dir,
            save_interval=save_interval,
            use_sharded_opt=use_sharded_opt,
            grad_reducer=grad_reducer,
            dataloader_state_provider=dataloader_state_provider,
        )

    # Drain any in-flight async save before returning so the file is on
    # disk before the trainer process can exit. Cheap no-op if there is
    # nothing pending (e.g. save_interval was None or this is not rank 0).
    wait_for_async_save()

    # Persist the per-segment profile (rank-0 only — all DP ranks see
    # identical step shapes, so a single file is sufficient and saves
    # ``world_size`` × disk I/O on the cctl shared volume). ``write`` is
    # a no-op when the profiler is disabled OR no profiled steps actually
    # ran.
    if profiler.enabled and dp_rank == 0:
        out_path = profiler.write(step_time_ms_by_step=host_step_ms_by_step)
        if out_path is not None:
            print(
                f"  [run_training_loop] wrote step profile to {out_path} "
                f"({len(profiler.records)} step(s))",
                flush=True,
            )

    return {"steps": steps_result}
