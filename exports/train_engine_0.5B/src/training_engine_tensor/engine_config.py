"""Explicit configuration dataclass for the training engine.

All framework-internal behavioral switches are collected here as typed,
documented fields — replacing the previous pattern of reading
``os.environ.get(...)`` at module-import time.

The ``EngineConfig`` is constructed once at CLI entry-point time
(``__main__.py``) and threaded through the call graph via function
parameters. No module in ``training_engine_tensor/`` should read
``os.environ`` directly for behavioral flags; the only exception is
the process-external contract variables listed in ``ENV_WHITELIST``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

__all__ = [
    "ENV_WHITELIST",
    "EngineConfig",
    "from_env",
    "get_config",
    "register_reset_hook",
    "set_global_config",
]

ENV_WHITELIST: frozenset[str] = frozenset({
    # ── Process-external contracts (torchrun / cctl / CUDA runtime) ───
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_UNIQUE_ID_FILE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NVTE_FUSED_ATTN",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG",
    "TORCH_NCCL_USE_COMM_NONBLOCKING",
    "MEGATRON_ROOT",
    "PYTHONPATH",
})

_DEPRECATED_ENV_BRIDGE: frozenset[str] = frozenset({
    # Observability — read by profiling.from_config / op_dispatcher.init,
    # not by EngineConfig, but historically lumped with the whitelist.
    "PROFILE_RANGE",
    "PROFILE_OUTPUT",
    "PROFILE_DEEP",
    "HOST_TIMER",
    # Operator version contracts (op_dispatcher.py, frozen at load)
    "OP_GEMM_FC1",
    "OP_GEMM_FC2",
    "OP_GEMM_QKV_PROJ",
    "OP_GEMM_ATTN_OUT_PROJ",
    "OP_GEMM_OUTPUT",
    "OP_ATTENTION",
    # Legacy migration bridge (from_env() → EngineConfig).
    # These env vars are read ONCE by from_env() at process startup and
    # converted to EngineConfig fields. They exist for backward compat
    # with cctl job specs that pass config via envVars. New deployments
    # should migrate to CLI arguments (--fused-ops, --wgrad-overlap, etc).
    "FUSED_OPS",
    "FUSE_CE",
    "FUSE_SWIGLU",
    "FUSE_RMSNORM_RESIDUAL",
    "FUSE_ROPE",
    "FUSE_ROPE_DOC_AWARE",
    "FUSE_ADAM_SYNC",
    "FUSE_GEMM_PAD",
    "FUSE_DIRECT_ATTN",
    "FUSE_INPLACE_RESIDUAL",
    "FUSE_INDEX_ADD_EMB_BWD",
    "USE_SPAN_BASED_ATTN",
    "FP32_MODE",
    "UNFUSED_MODE",
    "TE_ROPE",
    "FP32_ALLREDUCE",
    "FP32_GRAD_ACCUM",
    "NO_GRAD_CLIP",
    "GRAD_DIAG",
    "WGRAD_OVERLAP",
    "WGRAD_BUCKET_TARGET_ELEMS",
    "OPT_ALLGATHER_OVERLAP",
    "ASYNC_LOSS_AR",
    "SHARDED_OPTIMIZER",
    "MULTI_TENSOR_ADAM",
    "CUSTOM_GEMM",
    "FORWARD_CUDA_GRAPH",
    "FORWARD_CUDA_GRAPH_WARMUP",
    "FORWARD_CUDA_GRAPH_TOL",
    "FORWARD_CUDA_GRAPH_SANITY",
    "FORWARD_CUDA_GRAPH_SAMPLE_STRIDE",
    "STEP_CUDA_GRAPH",
    "STEP_CUDA_GRAPH_WARMUP",
    "STEP_CUDA_GRAPH_TOL",
    "STEP_CUDA_GRAPH_SANITY",
    "STEP_CUDA_GRAPH_FULL",
    "STEP_CUDA_GRAPH_FULL_WARMUP",
    "STEP_CUDA_GRAPH_FULL_TOL",
    "STEP_CUDA_GRAPH_FULL_SANITY",
    "STEP_CUDA_GRAPH_NCCL_OPT",
    "STEP_CUDA_GRAPH_NCCL_OPT_WARMUP",
    "STEP_CUDA_GRAPH_OPTIMIZER",
    "STEP_CUDA_GRAPH_OPTIMIZER_WARMUP",
    "USE_FUSED_NORM_BWD",
    "NORM_BWD_NAN_DIAG",
    "EMIT_LOSS_LINES",
    "GRAD_NORM_DIAG",
    "MUP_EMB_SCALE",
    "MUP_DEPTH_SCALE",
    "VOCAB_SIZE",
    "MTP_NUM_LAYERS",
    "MTP_LOSS_SCALING_FACTOR",
    "MAX_LR",
    "MIN_LR",
    "WARMUP_ITERS",
    "LR_DECAY_ITERS",
    "WSD_DECAY_ITERS",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPS",
    "WEIGHT_DECAY",
    "MAX_GRAD_NORM",
    "RANDOM_INIT",
})


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for a single training run.

    All fields have sensible defaults matching the previous env-var
    defaults, so constructing ``EngineConfig()`` is equivalent to
    running with no env-var overrides.
    """

    # ── Fusion switches ──────────────────────────────────────────────
    fused_ops: bool = False
    fuse_ce: bool = False
    fuse_swiglu: bool = False
    fuse_rmsnorm_residual: bool = False
    fuse_rope: bool = False
    fuse_rope_doc_aware: bool = False
    fuse_adam_sync: bool = False
    fuse_gemm_pad: bool = False
    fuse_direct_attn: bool = False
    fuse_inplace_residual: bool = False
    fuse_index_add_emb_bwd: bool = False
    use_span_based_attn: bool = False

    # ── Precision / debug ────────────────────────────────────────────
    fp32_mode: bool = False
    unfused_mode: bool = False
    te_rope: bool = False
    fp32_allreduce: bool = False
    fp32_grad_accum: bool = False
    no_grad_clip: bool = False
    grad_diag: bool = False

    # ── Communication strategy ───────────────────────────────────────
    wgrad_overlap: bool = True
    wgrad_bucket_target_elems: int = 100 * 1024 * 1024
    opt_allgather_overlap: bool = False
    async_loss_ar: bool = True
    sharded_optimizer: bool = True
    # P2-B grad-accum overlap: when ``num_local_microbatches > 1`` AND
    # this is set, ``BucketedGradReducer`` runs its per-microbatch
    # accumulation (the ``copy_`` of the first MB and the ``add_`` of
    # subsequent MBs) on a dedicated CUDA stream rather than the
    # default stream. This lets the next MB's forward begin the moment
    # backward finishes, without waiting for the accumulation kernel.
    # Default off so the change is opt-in and trivially reversible
    # (set ``ACCUM_STREAM_OVERLAP=0`` to fall back to the P2-A path).
    accum_stream_overlap: bool = False
    # P1-E data prefetch pipeline depth. ``1`` keeps the legacy
    # single-slot prefetcher (one batch in flight; bytewise identical
    # to the original behaviour). ``>= 2`` switches to a multi-slot
    # queue that keeps N batches in flight on the copy stream so the
    # host-side ``next(iter)`` cost is decoupled from the GPU's
    # forward critical path. Capped at a small constant to avoid
    # unbounded host memory growth.
    data_prefetch_depth: int = 1

    # ── Optimizer ────────────────────────────────────────────────────
    multi_tensor_adam: bool = False

    # ── Custom GEMM ─────────────────────────────────────────────────
    custom_gemm: bool = False

    # ── CUDA Graph ──────────────────────────────────────────────────
    forward_cuda_graph: bool = False
    forward_cuda_graph_warmup: int = 3
    forward_cuda_graph_tol: float = 1e-3
    forward_cuda_graph_sanity: bool = False
    forward_cuda_graph_sample_stride: int = 1024

    step_cuda_graph: bool = False
    step_cuda_graph_warmup: int = 3
    step_cuda_graph_tol: float = 1e-3
    step_cuda_graph_sanity: bool = False
    # Direction-C PoC switch: when ``step_cuda_graph`` is also on AND
    # ``num_local_microbatches > 1``, allow the train loop to keep
    # using step CUDA Graph (each MB replays one captured fwd+CE+bwd;
    # grads accumulate in the legacy FP32 dict in host code between
    # replays). Forces ``wgrad_overlap`` off as the underlying graph
    # cannot record the BucketedGradReducer host callback. Default off
    # so the legacy "step graph requires accum=1" gate stays the
    # explicit, documented contract; flip via ``STEP_CUDA_GRAPH_ACCUM=1``
    # for opt-in experiments.
    step_cuda_graph_accum: bool = False
    # Direction-C path-1: device-side grad emit, allowing step CUDA
    # Graph to coexist with BucketedGradReducer. When enabled, the
    # graph captures the backward pass with ``grad_sink`` bound to a
    # capture-safe callable that writes each parameter gradient
    # directly into its bucket slot (single in-place ``add_`` op, no
    # host conditional). Requires ``step_cuda_graph=1`` AND
    # ``wgrad_overlap=1`` (the reducer must be alive). The train loop
    # zeros bucket buffers at the start of each step (so ``add_``
    # semantics are correct even for the first MB) and triggers the
    # per-bucket reduce_scatters after the final MB's replay. Default
    # off; flip via ``STEP_CUDA_GRAPH_REDUCER=1``.
    step_cuda_graph_reducer: bool = False

    step_cuda_graph_full: bool = False
    step_cuda_graph_full_warmup: int = 3
    step_cuda_graph_full_tol: float = 1e-3
    step_cuda_graph_full_sanity: bool = False

    step_cuda_graph_nccl_opt: bool = False
    step_cuda_graph_nccl_opt_warmup: int = 3

    step_cuda_graph_optimizer: bool = False
    step_cuda_graph_optimizer_warmup: int = 3

    # ── Backward ────────────────────────────────────────────────────
    use_fused_norm_bwd: bool = False

    # ── Diagnostics ─────────────────────────────────────────────────
    norm_bwd_nan_diag: int = 0
    emit_loss_lines: bool = False
    grad_norm_diag: bool = False

    # ── muP (maximal update parameterization) ──────────────────────
    mup_emb_scale: float = 0.0
    mup_depth_scale: float = 0.0

    # ── MTP ──────────────────────────────────────────────────────────
    vocab_size: int = 0
    mtp_num_layers: int = 0
    mtp_loss_scaling_factor: float = 0.1

    # ── LR schedule (WSD) ────────────────────────────────────────────
    max_lr: float = 3.0e-4
    min_lr: float = 0.0
    warmup_iters: int = 2000
    lr_decay_iters: int = 240000
    wsd_decay_iters: int = 3000

    # ── Optimizer hyper-parameters ───────────────────────────────────
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # ── Random init ─────────────────────────────────────────────────
    random_init: bool = False

    # ── Profiling ────────────────────────────────────────────────────
    profile_range: tuple[int, int] | None = None
    profile_output: str | None = None
    profile_deep: bool = False
    host_timer: bool = False


def _parse_profile_range(raw: str) -> tuple[int, int] | None:
    """Parse ``"START,END"`` into a tuple or return None if empty."""
    raw = raw.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"PROFILE_RANGE must be 'start,end' (got {raw!r})")
    return (int(parts[0]), int(parts[1]))


def _model_spec_vocab_size() -> int:
    """Load vocab_size from model_spec.toml (the L1 SSOT)."""
    from training_engine_tensor.config import _VOCAB_SIZE  # noqa: PLC0415
    return _VOCAB_SIZE


def from_env() -> EngineConfig:
    """Construct an ``EngineConfig`` by reading the current ``os.environ``.

    This is the **migration bridge**: call once at program start, then
    pass the result through function parameters. Once all callers are
    migrated, this function should be deleted and config should be
    built from CLI args / TOML only.
    """
    active_deprecated = _DEPRECATED_ENV_BRIDGE & set(os.environ)
    if active_deprecated:
        warnings.warn(
            "Deprecated env vars detected — migrate to CLI args: "
            + ", ".join(sorted(active_deprecated)),
            DeprecationWarning,
            stacklevel=2,
        )

    def _bool(key: str, default: str = "0") -> bool:
        return os.environ.get(key, default) == "1"

    def _int(key: str, default: str) -> int:
        return int(os.environ.get(key, default))

    def _float(key: str, default: str) -> float:
        return float(os.environ.get(key, default))

    fused_ops = _bool("FUSED_OPS")

    def _fuse_flag(name: str) -> bool:
        return os.environ.get(name, "1" if fused_ops else "0") == "1"

    fuse_rope = _fuse_flag("FUSE_ROPE")

    return EngineConfig(
        fused_ops=fused_ops,
        fuse_ce=_fuse_flag("FUSE_CE"),
        fuse_swiglu=_fuse_flag("FUSE_SWIGLU"),
        fuse_rmsnorm_residual=_fuse_flag("FUSE_RMSNORM_RESIDUAL"),
        fuse_rope=fuse_rope,
        fuse_rope_doc_aware=(
            os.environ.get("FUSE_ROPE_DOC_AWARE", "1" if fuse_rope else "0") == "1"
        ),
        fuse_adam_sync=_fuse_flag("FUSE_ADAM_SYNC"),
        fuse_gemm_pad=_fuse_flag("FUSE_GEMM_PAD"),
        fuse_direct_attn=_bool("FUSE_DIRECT_ATTN"),
        fuse_inplace_residual=_fuse_flag("FUSE_INPLACE_RESIDUAL"),
        fuse_index_add_emb_bwd=_fuse_flag("FUSE_INDEX_ADD_EMB_BWD"),
        use_span_based_attn=_bool("USE_SPAN_BASED_ATTN"),
        fp32_mode=_bool("FP32_MODE"),
        unfused_mode=_bool("UNFUSED_MODE"),
        te_rope=_bool("TE_ROPE"),
        fp32_allreduce=_bool("FP32_ALLREDUCE"),
        fp32_grad_accum=_bool("FP32_GRAD_ACCUM"),
        no_grad_clip=_bool("NO_GRAD_CLIP"),
        grad_diag=_bool("GRAD_DIAG"),
        wgrad_overlap=_bool("WGRAD_OVERLAP", "1"),
        wgrad_bucket_target_elems=_int("WGRAD_BUCKET_TARGET_ELEMS", str(100 * 1024 * 1024)),
        opt_allgather_overlap=_bool("OPT_ALLGATHER_OVERLAP"),
        async_loss_ar=_bool("ASYNC_LOSS_AR", "1"),
        sharded_optimizer=_bool("SHARDED_OPTIMIZER", "1"),
        accum_stream_overlap=_bool("ACCUM_STREAM_OVERLAP"),
        data_prefetch_depth=_int("DATA_PREFETCH_DEPTH", "1"),
        multi_tensor_adam=_bool("MULTI_TENSOR_ADAM"),
        custom_gemm=_bool("CUSTOM_GEMM"),
        forward_cuda_graph=_bool("FORWARD_CUDA_GRAPH"),
        forward_cuda_graph_warmup=_int("FORWARD_CUDA_GRAPH_WARMUP", "3"),
        forward_cuda_graph_tol=_float("FORWARD_CUDA_GRAPH_TOL", "1e-3"),
        forward_cuda_graph_sanity=_bool("FORWARD_CUDA_GRAPH_SANITY"),
        forward_cuda_graph_sample_stride=_int("FORWARD_CUDA_GRAPH_SAMPLE_STRIDE", "1024"),
        step_cuda_graph=_bool("STEP_CUDA_GRAPH"),
        step_cuda_graph_warmup=_int("STEP_CUDA_GRAPH_WARMUP", "3"),
        step_cuda_graph_tol=_float("STEP_CUDA_GRAPH_TOL", "1e-3"),
        step_cuda_graph_sanity=_bool("STEP_CUDA_GRAPH_SANITY"),
        step_cuda_graph_accum=_bool("STEP_CUDA_GRAPH_ACCUM"),
        step_cuda_graph_reducer=_bool("STEP_CUDA_GRAPH_REDUCER"),
        step_cuda_graph_full=_bool("STEP_CUDA_GRAPH_FULL"),
        step_cuda_graph_full_warmup=_int("STEP_CUDA_GRAPH_FULL_WARMUP", "3"),
        step_cuda_graph_full_tol=_float("STEP_CUDA_GRAPH_FULL_TOL", "1e-3"),
        step_cuda_graph_full_sanity=_bool("STEP_CUDA_GRAPH_FULL_SANITY"),
        step_cuda_graph_nccl_opt=_bool("STEP_CUDA_GRAPH_NCCL_OPT"),
        step_cuda_graph_nccl_opt_warmup=_int("STEP_CUDA_GRAPH_NCCL_OPT_WARMUP", "3"),
        step_cuda_graph_optimizer=_bool("STEP_CUDA_GRAPH_OPTIMIZER"),
        step_cuda_graph_optimizer_warmup=_int("STEP_CUDA_GRAPH_OPTIMIZER_WARMUP", "3"),
        use_fused_norm_bwd=_bool("USE_FUSED_NORM_BWD"),
        norm_bwd_nan_diag=_int("NORM_BWD_NAN_DIAG", "0"),
        emit_loss_lines=_bool("EMIT_LOSS_LINES"),
        grad_norm_diag=_bool("GRAD_NORM_DIAG"),
        mup_emb_scale=_float("MUP_EMB_SCALE", "0.0"),
        mup_depth_scale=_float("MUP_DEPTH_SCALE", "0.0"),
        vocab_size=_int("VOCAB_SIZE", str(_model_spec_vocab_size())),
        mtp_num_layers=_int("MTP_NUM_LAYERS", "0"),
        mtp_loss_scaling_factor=_float("MTP_LOSS_SCALING_FACTOR", "0.1"),
        max_lr=_float("MAX_LR", "3e-4"),
        min_lr=_float("MIN_LR", "0.0"),
        warmup_iters=_int("WARMUP_ITERS", "2000"),
        lr_decay_iters=_int("LR_DECAY_ITERS", "240000"),
        wsd_decay_iters=_int("WSD_DECAY_ITERS", "3000"),
        adam_beta1=_float("ADAM_BETA1", "0.9"),
        adam_beta2=_float("ADAM_BETA2", "0.95"),
        adam_eps=_float("ADAM_EPS", "1e-8"),
        weight_decay=_float("WEIGHT_DECAY", "0.1"),
        max_grad_norm=_float("MAX_GRAD_NORM", "1.0"),
        random_init=_bool("RANDOM_INIT"),
        profile_range=_parse_profile_range(os.environ.get("PROFILE_RANGE", "")),
        profile_output=os.environ.get("PROFILE_OUTPUT") or None,
        profile_deep=_bool("PROFILE_DEEP"),
        host_timer=_bool("HOST_TIMER"),
    )


# ---------------------------------------------------------------------------
# Global config singleton
# ---------------------------------------------------------------------------

_GLOBAL_CONFIG: EngineConfig | None = None


def set_global_config(cfg: EngineConfig) -> None:
    """Set the process-wide EngineConfig singleton.

    Must be called exactly once, at CLI entry-point time, before any
    module reads the config. Raises ``RuntimeError`` on double-set.
    """
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is not None:
        raise RuntimeError(
            "set_global_config() called twice — EngineConfig is already set. "
            "This indicates a double-init bug in the entry point."
        )
    _GLOBAL_CONFIG = cfg


def get_config() -> EngineConfig:
    """Return the process-wide EngineConfig singleton.

    Raises ``RuntimeError`` if ``set_global_config()`` has not been
    called yet — every entry point must explicitly initialise the config.
    """
    if _GLOBAL_CONFIG is None:
        raise RuntimeError(
            "get_config(): set_global_config() has not been called yet. "
            "Call set_global_config(from_env()) or set_global_config(EngineConfig(...)) "
            "at the CLI entry point before importing framework modules."
        )
    return _GLOBAL_CONFIG


_reset_hooks: list = []


def register_reset_hook(hook) -> None:
    """Register a callable to be invoked when global config is reset.

    Modules with config-dependent caches call this at import time
    so ``_reset_global_config()`` can clear them automatically.
    """
    _reset_hooks.append(hook)


def _reset_global_config() -> None:
    """Reset the singleton and fire all registered cache-clearing hooks (for testing only)."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None
    for hook in _reset_hooks:
        hook()
