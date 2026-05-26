"""Single source of truth for runtime configuration.

Every framework-internal behavioural switch lives here as a typed,
documented dataclass field.  Modules in :mod:`training_engine_tensor`
read configuration through :func:`get_config`, which returns the
process-wide :class:`EngineConfig` singleton initialised once at
:mod:`training_engine_tensor.__main__` entry time.

The only direct ``os.environ`` reads permitted in the rest of the
package are the values listed in :data:`ENV_WHITELIST` — these are
either process-external contracts (``RANK`` / ``LOCAL_RANK`` set by
``torchrun``), runtime-provider knobs (``CUDA_DEVICE_MAX_CONNECTIONS``,
``NVTE_FUSED_ATTN``, ``CUTLASS_DSL_PATH``), or operator-level dispatch
keys consumed by :mod:`training_engine_tensor.op_dispatcher`
(``OP_GEMM_*``).  Everything else flows through ``EngineConfig``.

The ``OURS_*`` environment variables that the framework historically
read at module-import time still work: :func:`from_env` provides a
backward-compatible bridge that materialises an ``EngineConfig`` from
those env vars.  New deployments should pass values explicitly via the
CLI (every dataclass field is mirrored as ``--field-name`` on the
``__main__`` parser).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "ENV_WHITELIST",
    "EngineConfig",
    "from_env",
    "get_config",
    "print_config_summary",
    "register_reset_hook",
    "set_global_config",
]


# ---------------------------------------------------------------------------
# Process-external contracts.
# ---------------------------------------------------------------------------
# These env vars are READ from os.environ at runtime by the framework
# (or by libraries the framework calls into).  They are NOT mirrored
# onto EngineConfig fields because they are owned by an outside actor
# (torchrun, the CUDA runtime, or the per-operator dispatcher) and
# providing a Python-level override would be misleading.

ENV_WHITELIST: frozenset[str] = frozenset({
    # torchrun / distributed launcher
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    # CUDA / NCCL runtime
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVTE_FUSED_ATTN",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "TORCH_NCCL_USE_COMM_NONBLOCKING",
    # CuTeDSL bootstrap (optional sys.path hint for cutlass-dsl wheel)
    "CUTLASS_DSL_PATH",
    # Python module search path
    "PYTHONPATH",
    # Per-operator version dispatch (consumed by op_dispatcher only)
    "OP_GEMM_FC1",
    "OP_GEMM_FC2",
    "OP_GEMM_OUTPUT",
    "OP_GEMM_QKV_PROJ",
    "OP_GEMM_ATTN_OUT_PROJ",
})


# ---------------------------------------------------------------------------
# Backward-compatibility bridge.
# ---------------------------------------------------------------------------
# These env vars are converted into EngineConfig fields by
# :func:`from_env` once at process startup, so external job specs that
# already pass values through envVars continue to work without code
# changes on the consumer side.

_FUSED_KERNEL_ENV_BRIDGE: tuple[tuple[str, str, bool], ...] = (
    # (env name, dataclass field, default-on)
    ("OURS_FUSED_ROPE",                  "fused_rope",                  True),
    ("OURS_FUSED_ROPE_PACK",             "fused_rope_pack",             True),
    ("OURS_FUSED_RESIDUAL_RMSNORM",      "fused_residual_rmsnorm",      True),
    ("OURS_FUSED_ROPE_QKV",              "fused_rope_qkv",              True),
    ("OURS_FUSED_ATTN_RMSNORM_BWD",      "fused_attn_rmsnorm_bwd",      True),
    ("OURS_FUSED_MLP_RESIDUAL_RMSNORM",  "fused_mlp_residual_rmsnorm",  True),
    ("OURS_FUSED_DW_REDUCE",             "fused_dw_reduce",             True),
    ("OURS_FUSED_ROPE_FP32COS",          "fused_rope_fp32cos",          True),
    ("OURS_AG_FWD_OVERLAP",              "ag_fwd_overlap",              True),
    ("OURS_ADAM_AG_PIPELINE",            "adam_ag_pipeline",            True),
    ("OURS_GPU_NORM_CLIP",               "gpu_norm_clip",               True),
    ("OURS_FUSED_CE",                    "fused_ce",                    True),
    ("OURS_DEFER_WGRAD_SYNC",            "defer_wgrad_sync",            True),
    ("OURS_FUSED_SWIGLU",                "fused_swiglu",                True),
    ("OURS_DIRECT_ATTN",                 "direct_attn",                 True),
    ("OURS_DSL_ATTN_FWD",                "dsl_attn_fwd",                True),
    ("OURS_ROPE_BHSD",                   "rope_bhsd",                   True),
)


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for a single training run.

    Every field has a sensible default that matches the production
    deployment, so ``EngineConfig()`` produces a runnable, fast,
    bit-deterministic configuration.  Override individual fields via
    the ``__main__`` CLI flags or by constructing a new instance.
    """

    # -- Fused Triton kernel switches (16 flags) ------------------------
    # These select between the framework's fused implementation and the
    # naive PyTorch path.  Setting any flag to False is a kill-switch
    # that recovers the unoptimised path without a code change.
    fused_rope: bool = True
    fused_rope_pack: bool = True
    fused_residual_rmsnorm: bool = True
    fused_rope_qkv: bool = True
    fused_attn_rmsnorm_bwd: bool = True
    fused_mlp_residual_rmsnorm: bool = True
    fused_dw_reduce: bool = True
    fused_rope_fp32cos: bool = True
    ag_fwd_overlap: bool = True
    adam_ag_pipeline: bool = True
    gpu_norm_clip: bool = True
    fused_ce: bool = True
    defer_wgrad_sync: bool = True
    fused_swiglu: bool = True

    # -- Tunable kernel-shape constants ---------------------------------
    # Reduce-scatter bucket count for backward overlap; 1 reverts to a
    # single post-backward RS (no overlap, naive path).
    rs_buckets: int = 32
    # Rows-per-block hyperparameter for the fused rmsnorm-bwd dW reduce.
    dw_reduce_rows_per_block: int = 32

    # -- Attention forward strategy -------------------------------------
    # ``direct_attn`` calls cuDNN's ``_scaled_dot_product_cudnn_attention``
    # ATen op directly, bypassing autograd dispatch overhead while still
    # using cuDNN under the hood.  ``dsl_attn_fwd`` additionally swaps
    # the cuDNN forward for a self-developed Hopper SM90a flash-attention
    # kernel from :mod:`flash_attn_dsl`; backward stays on cuDNN.  Both
    # default on — the engine ships the production-ready attention path.
    # Set ``OURS_DIRECT_ATTN=0`` / ``OURS_DSL_ATTN_FWD=0`` to fall back
    # to the cuDNN autograd path (kill-switch).
    direct_attn: bool = True
    dsl_attn_fwd: bool = True

    # -- RoPE BHSD layout bridge ----------------------------------------
    # When True the RoPE forward writes q/k directly in [B, H, S, D]
    # contiguous layout and the backward kernels consume dq/dk/dv in
    # the same layout, eliminating a permute().contiguous() at the
    # attention boundary.  Requires direct_attn=True and the fused-rope
    # qkv/pack flags (BHSD kernels reuse those code paths); when
    # ``direct_attn`` is off this flag is a no-op (forward and backward
    # AND it with ``direct_attn`` at the call site).
    rope_bhsd: bool = True

    # -- Custom GEMM master switch --------------------------------------
    # Off by default so the engine is bit-identical to the torch
    # baseline.  When True, every operator in
    # ``training_engine_tensor.ops`` whose ``register.toml`` selects a
    # non-baseline version is dispatched to its self-developed kernel.
    custom_gemm: bool = False

    # -- Vocabulary geometry --------------------------------------------
    # ``padded_vocab_size`` is the lm_head output dimension after
    # ``make_vocab_size_divisible_by * tensor_model_parallel_size``
    # rounding.  Default matches the HF tokenizer (73448, already a
    # multiple of 4 so no padding rows are added).  Override if
    # bringing up an alternative tokenizer.
    padded_vocab_size: int = 73448

    # -- LR schedule (WSD) ----------------------------------------------
    max_lr: float = 2.273e-4
    min_lr: float = 0.0
    lr_warmup_iters: int = 2000
    lr_decay_iters: int = 1_000_000
    lr_wsd_decay_iters: int = 0

    # -- Optimizer hyper-parameters -------------------------------------
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    clip_grad: float = 1.0


# ---------------------------------------------------------------------------
# Env-bridge constructor.
# ---------------------------------------------------------------------------


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    return float(raw) if raw else default


def from_env() -> EngineConfig:
    """Construct an ``EngineConfig`` from the current ``os.environ``.

    Reads the legacy ``OURS_*`` fused-kernel kill-switch env vars plus
    a handful of LR / optimizer overrides, falling back to the
    dataclass defaults when an env var is unset.  Call once at process
    startup, then thread the result through the call graph via
    :func:`set_global_config`.
    """
    overrides: dict[str, object] = {}
    for env_name, field_name, default_on in _FUSED_KERNEL_ENV_BRIDGE:
        overrides[field_name] = _env_bool(env_name, default_on)

    overrides["rs_buckets"] = _env_int("OURS_RS_BUCKETS", 32)
    overrides["dw_reduce_rows_per_block"] = _env_int("OURS_DW_REDUCE_RPB", 32)
    overrides["custom_gemm"] = _env_bool("CUSTOM_GEMM", False)

    if "PADDED_VOCAB_SIZE" in os.environ:
        overrides["padded_vocab_size"] = _env_int("PADDED_VOCAB_SIZE", 73448)

    return EngineConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Process-wide singleton.
# ---------------------------------------------------------------------------


_GLOBAL_CONFIG: EngineConfig | None = None
_RESET_HOOKS: list = []


def set_global_config(cfg: EngineConfig) -> None:
    """Install the process-wide :class:`EngineConfig`.

    Must be called exactly once, at CLI entry-point time, before any
    other module reads configuration.  Raises :class:`RuntimeError` on
    double-set so a duplicate init is loud rather than silent.
    """
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is not None:
        raise RuntimeError(
            "set_global_config() called twice — EngineConfig is already set. "
            "This indicates a double-init bug in the entry point.",
        )
    _GLOBAL_CONFIG = cfg


def get_config() -> EngineConfig:
    """Return the process-wide :class:`EngineConfig`.

    Raises :class:`RuntimeError` if :func:`set_global_config` has not
    been called yet — every entry point is required to install the
    config explicitly.
    """
    if _GLOBAL_CONFIG is None:
        raise RuntimeError(
            "get_config(): set_global_config() has not been called yet. "
            "Call set_global_config(from_env()) or "
            "set_global_config(EngineConfig(...)) at the CLI entry point.",
        )
    return _GLOBAL_CONFIG


def register_reset_hook(hook) -> None:
    """Register a callable invoked when the global config is reset.

    Modules with config-derived caches call this at import time so
    :func:`_reset_global_config` can clear them automatically.
    """
    _RESET_HOOKS.append(hook)


def _reset_global_config() -> None:
    """Clear the singleton and fire reset hooks; for tests only."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None
    for hook in _RESET_HOOKS:
        hook()


# ---------------------------------------------------------------------------
# Single-line summary for log scraping.
# ---------------------------------------------------------------------------


def print_config_summary(cfg: EngineConfig | None = None) -> None:
    """Emit one ``[CONFIG] key=value ...`` line for log filters.

    Use ``grep '\\[CONFIG\\]' <log>`` to pull the active configuration
    out of a job's stdout.  When ``cfg`` is omitted the process-wide
    singleton (see :func:`get_config`) is used.
    """
    cfg = cfg if cfg is not None else get_config()
    parts = [
        f"fused_rope={cfg.fused_rope}",
        f"fused_rope_pack={cfg.fused_rope_pack}",
        f"fused_residual_rmsnorm={cfg.fused_residual_rmsnorm}",
        f"fused_rope_qkv={cfg.fused_rope_qkv}",
        f"fused_attn_rmsnorm_bwd={cfg.fused_attn_rmsnorm_bwd}",
        f"fused_mlp_residual_rmsnorm={cfg.fused_mlp_residual_rmsnorm}",
        f"fused_dw_reduce={cfg.fused_dw_reduce}",
        f"fused_rope_fp32cos={cfg.fused_rope_fp32cos}",
        f"rs_buckets={cfg.rs_buckets}",
        f"dw_reduce_rpb={cfg.dw_reduce_rows_per_block}",
        f"ag_fwd_overlap={cfg.ag_fwd_overlap}",
        f"adam_ag_pipeline={cfg.adam_ag_pipeline}",
        f"gpu_norm_clip={cfg.gpu_norm_clip}",
        f"fused_ce={cfg.fused_ce}",
        f"defer_wgrad_sync={cfg.defer_wgrad_sync}",
        f"fused_swiglu={cfg.fused_swiglu}",
        f"direct_attn={cfg.direct_attn}",
        f"dsl_attn_fwd={cfg.dsl_attn_fwd}",
        f"rope_bhsd={cfg.rope_bhsd}",
        f"custom_gemm={cfg.custom_gemm}",
        f"padded_vocab_size={cfg.padded_vocab_size}",
    ]
    print(f"[CONFIG] {' '.join(parts)}", flush=True)
