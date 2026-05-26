"""Architecture constants for MiniCPM4 0.5B.

Model geometry is loaded from ``model_spec.toml`` (L1 source of truth)
at module-import time. Behavioral flags live in ``EngineConfig`` and
are accessed via ``get_config()`` — this module only holds architecture
constants and muP/MTP runtime state.
"""

from pathlib import Path

__all__ = [
    "DTYPE",
    "FFN_HIDDEN_SIZE",
    "GEMM_PAD_TO",
    "H100_PEAK_TFLOPS_BF16",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "INIT_METHOD_STD",
    "MAX_SEQ_LENGTH",
    "MUP_BASE_HIDDEN_SIZE",
    "NORM_EPS",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "NUM_LAYERS",
    "ROPE_THETA",
    "compute_dtype",
    "compute_training_flops",
    "compute_training_flops_standard",
    "get_effective_vocab_size",
    "mtp_enabled",
    "mtp_per_layer_loss_scale",
    "mup_residual_scale",
    "mup_width_scale",
    "use_unfused",
]

import torch

from .engine_config import get_config

# ── Load model architecture from model_spec.toml ────────────────────
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "model_spec.toml"
with open(_SPEC_PATH, "rb") as _f:
    _SPEC = tomllib.load(_f)
_MODEL = _SPEC["model"]
_TRAINING = _SPEC["training"]

NUM_LAYERS: int = int(_MODEL["num_layers"])
HIDDEN_SIZE: int = int(_MODEL["hidden_size"])
NUM_HEADS: int = int(_MODEL["num_attention_heads"])
NUM_KV_HEADS: int = int(_MODEL["num_query_groups"])
HEAD_DIM: int = HIDDEN_SIZE // NUM_HEADS
FFN_HIDDEN_SIZE: int = int(_MODEL["ffn_hidden_size"])
_VOCAB_SIZE: int = int(_MODEL["vocab_size"])
MAX_SEQ_LENGTH: int = int(_MODEL["seq_length"])
NORM_EPS: float = float(_MODEL["norm_epsilon"])
ROPE_THETA: float = float(_MODEL["rotary_base"])
DTYPE: torch.dtype = torch.bfloat16

MUP_BASE_HIDDEN_SIZE: int = 256
INIT_METHOD_STD: float = float(_MODEL.get("init_method_std", 0.1))


def mup_width_scale() -> float:
    cfg = get_config()
    if cfg.mup_emb_scale <= 0:
        return 1.0
    return HIDDEN_SIZE / MUP_BASE_HIDDEN_SIZE


def mup_residual_scale() -> float:
    """Pre-computed mup_depth_scale / sqrt(num_layers) for residual connections.

    Matches Megatron's ``mup_bias_dropout_add_unfused``:
        depth_scale = mup_depth_scale / math.sqrt(num_layers)
        out = residual + out * depth_scale
    """
    cfg = get_config()
    if cfg.mup_depth_scale <= 0:
        return 1.0
    return cfg.mup_depth_scale / NUM_LAYERS ** 0.5


def mtp_enabled() -> bool:
    return get_config().mtp_num_layers > 0


def mtp_per_layer_loss_scale() -> float:
    """Per-MTP-layer loss weight = ``mtp_loss_scaling_factor / mtp_num_layers``.

    Matches ``mtp_loss_scale = self.mtp_loss_scaling_factor / mtp_num_layers``
    in ``megatron/core/transformer/multi_token_prediction.py``.
    """
    cfg = get_config()
    if cfg.mtp_num_layers <= 0:
        return 0.0
    return cfg.mtp_loss_scaling_factor / cfg.mtp_num_layers


def get_effective_vocab_size() -> int:
    """Return the effective vocab size from ``EngineConfig``.

    Raises ``RuntimeError`` if the global config has not been initialised.
    """
    return get_config().vocab_size


def use_unfused() -> bool:
    cfg = get_config()
    return cfg.fp32_mode or cfg.unfused_mode


def compute_dtype() -> torch.dtype:
    return torch.float32 if get_config().fp32_mode else DTYPE


# ── Hardware ─────────────────────────────────────────────────────────
GEMM_PAD_TO: int = 128
H100_PEAK_TFLOPS_BF16: float = 989.4

# ── FLOP counts (GQA-aware) ─────────────────────────────────────────
_Q_PROJ_SIZE: int = NUM_HEADS * HEAD_DIM      # 16 * 64 = 1024
_KV_PROJ_SIZE: int = NUM_KV_HEADS * HEAD_DIM  # 2 * 64 = 128

_SHARED_LAYER_FLOPS_PER_TOKEN: int = (
    2 * HIDDEN_SIZE * _Q_PROJ_SIZE           # Q projection
    + 2 * HIDDEN_SIZE * _KV_PROJ_SIZE        # K projection
    + 2 * HIDDEN_SIZE * _KV_PROJ_SIZE        # V projection
    + 2 * _Q_PROJ_SIZE * HIDDEN_SIZE         # O projection
    + 2 * HIDDEN_SIZE * FFN_HIDDEN_SIZE      # gate proj (SwiGLU)
    + 2 * HIDDEN_SIZE * FFN_HIDDEN_SIZE      # up proj (SwiGLU)
    + 2 * FFN_HIDDEN_SIZE * HIDDEN_SIZE      # down proj
)
_GLOBAL_FLOPS_PER_TOKEN: int = 2 * HIDDEN_SIZE * _VOCAB_SIZE

_LAYER_FLOPS_PER_TOKEN: int = (
    _SHARED_LAYER_FLOPS_PER_TOKEN + 4 * MAX_SEQ_LENGTH * _Q_PROJ_SIZE
)
_STANDARD_LAYER_FLOPS_PER_TOKEN: int = (
    _SHARED_LAYER_FLOPS_PER_TOKEN + 2 * MAX_SEQ_LENGTH * _Q_PROJ_SIZE
)

_FORWARD_FLOPS_PER_TOKEN: int = (
    NUM_LAYERS * _LAYER_FLOPS_PER_TOKEN + _GLOBAL_FLOPS_PER_TOKEN
)
_STANDARD_FORWARD_FLOPS_PER_TOKEN: int = (
    NUM_LAYERS * _STANDARD_LAYER_FLOPS_PER_TOKEN + _GLOBAL_FLOPS_PER_TOKEN
)

# ── MTP (EAGLE) per-layer extras ────────────────────────────────────
# Per MTP layer, the forward pass adds (mirrors
# ``harness/ref/reference/train_pure_mup_mtp.py`` MFU enumeration and
# ``train_engine/src/training_engine_tensor/forward.py:mtp_forward_pass_with_save``):
#   * one full transformer block (same shape as a main layer)
#   * one extra LM-head call (tied weight, independent compute = 2 H V)
#   * one eh_proj Linear(2H -> H, no bias)            = 2 * (2H) * H = 4 H^2
# The transformer block is counted under the corresponding ``_LAYER_*``
# convention so the standard/hardware split stays consistent with the
# main-model accounting.
_MTP_EH_PROJ_FLOPS_PER_TOKEN: int = 2 * (2 * HIDDEN_SIZE) * HIDDEN_SIZE
_MTP_LAYER_FLOPS_PER_TOKEN: int = (
    _LAYER_FLOPS_PER_TOKEN + _GLOBAL_FLOPS_PER_TOKEN + _MTP_EH_PROJ_FLOPS_PER_TOKEN
)
_STANDARD_MTP_LAYER_FLOPS_PER_TOKEN: int = (
    _STANDARD_LAYER_FLOPS_PER_TOKEN
    + _GLOBAL_FLOPS_PER_TOKEN
    + _MTP_EH_PROJ_FLOPS_PER_TOKEN
)


def compute_training_flops(tokens_per_step: int) -> int:
    """Hardware-realistic training FLOPs (no causal-mask half on attention).

    Adds ``mtp_num_layers`` MTP extras when MTP is enabled — otherwise the
    achieved-flops MFU silently under-counts the MTP forward/backward work
    that already shows up in ``step_time``.
    """
    mtp_n = get_config().mtp_num_layers
    fwd_per_token = _FORWARD_FLOPS_PER_TOKEN + mtp_n * _MTP_LAYER_FLOPS_PER_TOKEN
    return 3 * fwd_per_token * tokens_per_step


def compute_training_flops_standard(tokens_per_step: int) -> int:
    """Standard training FLOPs (causal mask halves attention) used by
    ``mfu_e2e_standard``.

    Adds ``mtp_num_layers`` MTP extras when MTP is enabled. Must stay
    bitwise-equivalent to the reference enumeration in
    ``harness/ref/reference/train_pure_mup_mtp.py`` so the gate's
    ``mfu_e2e_standard`` is comparable across the production CLI and the
    Megatron reference.
    """
    mtp_n = get_config().mtp_num_layers
    fwd_per_token = (
        _STANDARD_FORWARD_FLOPS_PER_TOKEN + mtp_n * _STANDARD_MTP_LAYER_FLOPS_PER_TOKEN
    )
    return 3 * fwd_per_token * tokens_per_step


