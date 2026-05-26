"""MiniCPM4-8B architectural constants and FLOPs accounting.

This module reads the L1 single source of truth at
``train_engine/model_spec.toml`` once at import time and exposes the
geometry as Python module-level constants.  Downstream modules
(``forward.py``, ``backward.py``, ``parameters.py``, ...) import the
constants from here rather than hardcoding the same values, so a change
to the spec file is the only edit required to retarget the engine to a
different geometry.

The runtime padded vocab size is read from :class:`EngineConfig` rather
than from this module's constant, so the bare-metal CLI can override it
for alternative tokenizers without re-importing the package.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

__all__ = [
    "NUM_LAYERS",
    "HIDDEN_SIZE",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "FFN_HIDDEN_SIZE",
    "MAX_SEQ_LENGTH",
    "NORM_EPSILON",
    "ROTARY_BASE",
    "INIT_METHOD_STD",
    "MAKE_VOCAB_SIZE_DIVISIBLE_BY",
    "VOCAB_SIZE",
    "DEFAULT_PADDED_VOCAB_SIZE",
    "H100_PEAK_TFLOPS_BF16",
    "compute_training_flops",
    "padded_vocab_size",
]


def _locate_model_spec() -> Path:
    """Find ``model_spec.toml`` whether the package is editable or installed.

    Layout when developing in-tree::

        train_engine/
          model_spec.toml
          src/training_engine_tensor/config.py   <-- this file

    Layout after ``pip install train_engine/``::

        site-packages/training_engine_tensor/config.py
        site-packages/training_engine_tensor/model_spec.toml   (package_data)

    Both layouts are supported by walking up from this file's parent
    until a ``model_spec.toml`` is found.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        spec_path = candidate / "model_spec.toml"
        if spec_path.is_file():
            return spec_path
    raise FileNotFoundError(
        "model_spec.toml not found near training_engine_tensor.config; "
        "expected to discover it next to this module or in train_engine/.",
    )


def _load_spec() -> dict:
    spec_path = _locate_model_spec()
    with spec_path.open("rb") as fh:
        return tomllib.load(fh)


_SPEC = _load_spec()
_MODEL = _SPEC["model"]


NUM_LAYERS: int = int(_MODEL["num_layers"])
HIDDEN_SIZE: int = int(_MODEL["hidden_size"])
NUM_HEADS: int = int(_MODEL["num_attention_heads"])
NUM_KV_HEADS: int = int(_MODEL["num_query_groups"])
HEAD_DIM: int = int(_MODEL["head_dim"])
FFN_HIDDEN_SIZE: int = int(_MODEL["ffn_hidden_size"])
MAX_SEQ_LENGTH: int = int(_MODEL["seq_length"])
NORM_EPSILON: float = float(_MODEL["norm_epsilon"])
ROTARY_BASE: float = float(_MODEL["rotary_base"])
INIT_METHOD_STD: float = float(_MODEL["init_method_std"])
MAKE_VOCAB_SIZE_DIVISIBLE_BY: int = int(_MODEL["make_vocab_size_divisible_by"])
VOCAB_SIZE: int = int(_MODEL["vocab_size"])
DEFAULT_PADDED_VOCAB_SIZE: int = int(_MODEL["padded_vocab_size"])

# H100 SXM5 80GB BF16 Tensor Core peak (dense, no sparsity).
H100_PEAK_TFLOPS_BF16: float = 989.0


def padded_vocab_size() -> int:
    """Return the active padded vocab size from :class:`EngineConfig`.

    Falls back to :data:`DEFAULT_PADDED_VOCAB_SIZE` when called before
    ``set_global_config`` (i.e. at very early import time before the
    CLI has bootstrapped the engine).  The fallback path lets utility
    scripts that only need geometry constants import this module
    without first wiring up an EngineConfig.
    """
    if "training_engine_tensor.engine_config" not in sys.modules:
        return DEFAULT_PADDED_VOCAB_SIZE
    from training_engine_tensor import engine_config  # noqa: PLC0415

    if engine_config._GLOBAL_CONFIG is None:  # noqa: SLF001
        return DEFAULT_PADDED_VOCAB_SIZE
    return engine_config.get_config().padded_vocab_size


def compute_training_flops(tokens_per_step: int) -> float:
    """Return the per-step training FLOPs across all GPUs combined.
    """
    L = NUM_LAYERS
    h = HIDDEN_SIZE
    s = MAX_SEQ_LENGTH
    V = padded_vocab_size()

    qkv_dim = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    o_dim = NUM_HEADS * HEAD_DIM
    layer_params = (
        qkv_dim * h
        + o_dim * h
        + 2 * FFN_HIDDEN_SIZE * h
        + FFN_HIDDEN_SIZE * h
    )

    # Only lm_head contributes a real GEMM at the vocab boundary; the
    # input embedding is gather/scatter and contributes 0 FLOPs.
    total_params = L * layer_params + V * h

    flops = 6 * total_params * tokens_per_step
    # Causal attention activation matmuls: 6 * s * h per token per layer.
    flops += 6 * L * h * s * tokens_per_step
    return float(flops)
