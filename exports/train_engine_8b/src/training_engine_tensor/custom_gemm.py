"""Wiring layer between the engine and the per-operator kernels.

Each of the five GEMM call sites (qkv_proj, attn_out_proj, fc1, fc2,
output) exposes three directions in production: fwd, dgrad, wgrad.
:mod:`forward` and :mod:`backward` call the ``custom_gemm_<op>_<dir>``
helpers in this module *before* the baseline ``torch.matmul`` /
``torch.mm`` / fused-wgrad path.  When the helper returns ``None`` the
caller falls back to the baseline.

Dispatch logic — ``None`` is returned when the legitimate baseline
path is selected:

  1. ``EngineConfig.custom_gemm`` (or the legacy ``CUSTOM_GEMM=1`` env
     var via :func:`engine_config.from_env`) is off — every helper
     returns ``None`` and the run is bit-identical to torch baseline.
  2. The per-op ``register.toml`` selects ``"baseline"`` (either as the
     default or via the operator's ``OP_GEMM_<NAME>`` env var override).
     Three operators (``gemm_fc2``, ``gemm_qkv_proj``,
     ``gemm_attn_out_proj``) ship as baseline-only stubs.

When the helper does dispatch into a self-developed kernel, every
exception (``ImportError`` from the kernel module, ``AttributeError``
on the entry-point symbol, runtime error inside the kernel)
propagates — there is no silent fallback to torch.  The first failing
step crashes loudly and gets bisected.

Tensor rank conventions:

  fwd  (x[*, I],  w[O, I])           → y[*, O]      (rank-preserving)
  dgrad(d_y[T, O], w[O, I], out=...) → out[T, I]    (or a fresh tensor
                                                     for the lm-head)
  wgrad(d_y[*, O], x[*, I], out_buf=Tensor) → FP32 [O, I]
        — strict ABI: ``out_buf`` MUST be a non-None FP32 ``[O, I]``
          tensor that the kernel accumulates into.

The wiring layer owns any 3D↔2D reshape needed at the boundary; the
kernel ABI is 2D for dgrad and rank-preserving for fwd / wgrad.
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch

from .engine_config import get_config
from .op_dispatcher import get_op_version


def _custom_gemm_enabled() -> bool:
    """Check whether custom GEMM dispatching is active for this run."""
    return get_config().custom_gemm


def _resolve(op_name: str, kind: str):
    """Return ``<op_name>_<kind>`` from the operator's ``kernel`` module.

    Returns ``None`` for the two legitimate baseline cases:
      * ``EngineConfig.custom_gemm`` is off.
      * ``register.toml`` (or ``OP_GEMM_<NAME>``) selects ``"baseline"``.

    Anything else — kernel module import error, missing symbol on the
    kernel module, runtime error inside the kernel — propagates.

    ``kind`` is one of ``"fwd"``, ``"dgrad"``, ``"wgrad"``.
    """
    if not _custom_gemm_enabled():
        return None
    if get_op_version(op_name) == "baseline":
        return None
    mod = importlib.import_module(
        f"training_engine_tensor.ops.{op_name}.kernel",
    )
    return getattr(mod, f"{op_name}_{kind}")


# ──────────────────────────────────────────────────────────────────────────
# gemm_qkv_proj  (column-parallel: O = (heads + 2*kv_heads) * head_dim / TP)
# ──────────────────────────────────────────────────────────────────────────
def custom_gemm_qkv_proj_fwd(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_qkv_proj", "fwd")
    if fn is None:
        return None
    return fn(x, w)


def custom_gemm_qkv_proj_dgrad(
    d_qkv: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_qkv_proj", "dgrad")
    if fn is None:
        return None
    return fn(d_qkv, w, out=out)


def custom_gemm_qkv_proj_wgrad(
    d_qkv: torch.Tensor, x: torch.Tensor, *, out_buf: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_qkv_proj", "wgrad")
    if fn is None:
        return None
    return fn(d_qkv, x, out_buf=out_buf)


# ──────────────────────────────────────────────────────────────────────────
# gemm_attn_out_proj  (row-parallel: I = heads * head_dim / TP)
# ──────────────────────────────────────────────────────────────────────────
def custom_gemm_attn_out_proj_fwd(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_attn_out_proj", "fwd")
    if fn is None:
        return None
    return fn(x, w)


def custom_gemm_attn_out_proj_dgrad(
    d_hidden: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_attn_out_proj", "dgrad")
    if fn is None:
        return None
    return fn(d_hidden, w, out=out)


def custom_gemm_attn_out_proj_wgrad(
    d_hidden: torch.Tensor, x: torch.Tensor, *, out_buf: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_attn_out_proj", "wgrad")
    if fn is None:
        return None
    return fn(d_hidden, x, out_buf=out_buf)


# ──────────────────────────────────────────────────────────────────────────
# gemm_fc1  (column-parallel SwiGLU input: O = 2 * ffn / TP — fused gate+up)
# ──────────────────────────────────────────────────────────────────────────
def custom_gemm_fc1_fwd(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_fc1", "fwd")
    if fn is None:
        return None
    return fn(x, w)


def custom_gemm_fc1_dgrad(
    d_fc1_out: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_fc1", "dgrad")
    if fn is None:
        return None
    return fn(d_fc1_out, w, out=out)


def custom_gemm_fc1_wgrad(
    d_fc1_out: torch.Tensor, x: torch.Tensor, *, out_buf: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_fc1", "wgrad")
    if fn is None:
        return None
    return fn(d_fc1_out, x, out_buf=out_buf)


# ──────────────────────────────────────────────────────────────────────────
# gemm_fc2  (row-parallel SwiGLU output: I = ffn/TP/2 after SwiGLU halving)
# ──────────────────────────────────────────────────────────────────────────
def custom_gemm_fc2_fwd(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_fc2", "fwd")
    if fn is None:
        return None
    return fn(x, w)


def custom_gemm_fc2_dgrad(
    d_hidden: torch.Tensor, w: torch.Tensor, *, out: torch.Tensor,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_fc2", "dgrad")
    if fn is None:
        return None
    return fn(d_hidden, w, out=out)


def custom_gemm_fc2_wgrad(
    d_hidden: torch.Tensor, x: torch.Tensor, *, out_buf: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_fc2", "wgrad")
    if fn is None:
        return None
    return fn(d_hidden, x, out_buf=out_buf)


# ──────────────────────────────────────────────────────────────────────────
# gemm_output  (column-parallel LM head: O = padded_vocab / TP)
#
# Note: dgrad has no ``out=`` buffer.  ``d_hidden = d_logits @ output_w``
# allocates a fresh tensor each step (the LM head has no per-layer
# reusable buffer because there is only one of it).  The kernel ABI
# mirrors that by accepting ``out=None``; the wiring helper drops the
# kwarg before the call.
# ──────────────────────────────────────────────────────────────────────────


def custom_gemm_output_fwd(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_output", "fwd")
    if fn is None:
        return None
    return fn(x, w)


def custom_gemm_output_dgrad(
    d_logits: torch.Tensor, output_w: torch.Tensor,
) -> Optional[torch.Tensor]:
    """d_hidden = d_logits @ output_w (returns fresh tensor; no out= buffer)."""
    fn = _resolve("gemm_output", "dgrad")
    if fn is None:
        return None
    return fn(d_logits, output_w, out=None)


def custom_gemm_output_wgrad(
    d_logits: torch.Tensor, hidden_final: torch.Tensor,
    *, out_buf: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    fn = _resolve("gemm_output", "wgrad")
    if fn is None:
        return None
    return fn(d_logits, hidden_final, out_buf=out_buf)


__all__ = [
    "custom_gemm_qkv_proj_fwd", "custom_gemm_qkv_proj_dgrad", "custom_gemm_qkv_proj_wgrad",
    "custom_gemm_attn_out_proj_fwd", "custom_gemm_attn_out_proj_dgrad", "custom_gemm_attn_out_proj_wgrad",
    "custom_gemm_fc1_fwd", "custom_gemm_fc1_dgrad", "custom_gemm_fc1_wgrad",
    "custom_gemm_fc2_fwd", "custom_gemm_fc2_dgrad", "custom_gemm_fc2_wgrad",
    "custom_gemm_output_fwd", "custom_gemm_output_dgrad", "custom_gemm_output_wgrad",
]
