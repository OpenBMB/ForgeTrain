"""Unified loader for Stage 2 custom GEMM operators.

Provides thin wrappers around the 5 CuTeDSL GEMM kernels (gemm_fc1,
gemm_fc2, gemm_qkv_proj, gemm_attn_out_proj, gemm_output).  Each
wrapper checks the op_dispatcher version and, if the op is not
"baseline", lazily imports and invokes the custom kernel.

Fallback behaviour:
  - If CUSTOM_GEMM=0 (default), all wrappers return None immediately.
  - If the custom kernel import fails (CuTeDSL not installed, JIT
    compilation error, missing CUDA arch, etc.), a warning is printed
    once per op and the wrapper returns None so the caller can fall
    back to the default (e.g. Transformer Engine) implementation.
  - If op_dispatcher reports "baseline" for an op, None is returned.

Usage in forward.py / backward.py::

    from training_engine_tensor.custom_gemm import custom_gemm_fc1_fwd
    result = custom_gemm_fc1_fwd(x, weight)
    if result is None:
        result = default_linear_forward(x, weight)  # TE / cuBLAS
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import torch

__all__ = [
    "custom_gemm_attn_out_proj_bwd",
    "custom_gemm_attn_out_proj_fwd",
    "custom_gemm_fc1_bwd",
    "custom_gemm_fc1_fwd",
    "custom_gemm_fc2_bwd",
    "custom_gemm_fc2_fwd",
    "custom_gemm_output_bwd",
    "custom_gemm_output_fwd",
    "custom_gemm_qkv_proj_bwd",
    "custom_gemm_qkv_proj_fwd",
]

_PACKAGE_DIR = str(Path(__file__).resolve().parent)

from . import op_dispatcher
from .engine_config import get_config

# ── Registry ────────────────────────────────────────────────────────────

_GEMM_OPS: dict[str, tuple[str, str, str]] = {
    "gemm_fc1":          ("training_engine_tensor.ops.gemm_fc1.kernel",          "gemm_fc1_fwd",          "gemm_fc1_bwd"),
    "gemm_fc2":          ("training_engine_tensor.ops.gemm_fc2.kernel",          "gemm_fc2_fwd",          "gemm_fc2_bwd"),
    "gemm_qkv_proj":     ("training_engine_tensor.ops.gemm_qkv_proj.kernel",     "gemm_qkv_proj_fwd",     "gemm_qkv_proj_bwd"),
    "gemm_attn_out_proj":("training_engine_tensor.ops.gemm_attn_out_proj.kernel", "gemm_attn_out_proj_fwd","gemm_attn_out_proj_bwd"),
    "gemm_output":       ("training_engine_tensor.ops.gemm_output.kernel",        "gemm_output_fwd",       "gemm_output_bwd"),
}

_warned: set[str] = set()
_kernel_cache: dict[str, tuple[Callable | None, Callable | None]] = {}


def _load_op(op_name: str) -> tuple[Callable | None, Callable | None]:
    """Lazily import the fwd/bwd kernel for *op_name*, caching the result."""
    cached = _kernel_cache.get(op_name)
    if cached is not None:
        return cached
    module_path, fwd_name, bwd_name = _GEMM_OPS[op_name]
    try:
        file_path = Path(_PACKAGE_DIR) / module_path.removeprefix("training_engine_tensor.").replace(".", "/")
        file_path = file_path.with_suffix(".py")
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        pair = (getattr(mod, fwd_name), getattr(mod, bwd_name))
    except Exception as exc:
        if op_name not in _warned:
            _warned.add(op_name)
            print(
                f"[custom_gemm] {op_name}: custom kernel unavailable, "
                f"falling back to baseline ({type(exc).__name__}: {exc})",
                file=sys.stderr, flush=True,
            )
        pair = (None, None)
    _kernel_cache[op_name] = pair
    return pair


def _make_fwd(op_name: str) -> Callable[..., torch.Tensor | None]:
    """Create a forward wrapper for *op_name*."""
    def _fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
        cfg = get_config()
        if not cfg.custom_gemm:
            return None
        if op_dispatcher.get_op_version(op_name) == "baseline":
            return None
        fwd_fn, _ = _load_op(op_name)
        if fwd_fn is None:
            return None
        return fwd_fn(x, weight)
    _fwd.__name__ = f"custom_{op_name}_fwd"
    _fwd.__qualname__ = f"custom_{op_name}_fwd"
    return _fwd


def _make_bwd(op_name: str) -> Callable[..., tuple[torch.Tensor, torch.Tensor] | None]:
    """Create a backward wrapper for *op_name*."""
    def _bwd(
        d_output: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        te_wgrad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        cfg = get_config()
        if not cfg.custom_gemm:
            return None
        if op_dispatcher.get_op_version(op_name) == "baseline":
            return None
        _, bwd_fn = _load_op(op_name)
        if bwd_fn is None:
            return None
        return bwd_fn(d_output, x, weight, te_wgrad=te_wgrad)
    _bwd.__name__ = f"custom_{op_name}_bwd"
    _bwd.__qualname__ = f"custom_{op_name}_bwd"
    return _bwd


# ── Public API (same names as before refactoring) ───────────────────────

custom_gemm_fc1_fwd = _make_fwd("gemm_fc1")
custom_gemm_fc1_bwd = _make_bwd("gemm_fc1")

custom_gemm_fc2_fwd = _make_fwd("gemm_fc2")
custom_gemm_fc2_bwd = _make_bwd("gemm_fc2")

custom_gemm_qkv_proj_fwd = _make_fwd("gemm_qkv_proj")
custom_gemm_qkv_proj_bwd = _make_bwd("gemm_qkv_proj")

custom_gemm_attn_out_proj_fwd = _make_fwd("gemm_attn_out_proj")
custom_gemm_attn_out_proj_bwd = _make_bwd("gemm_attn_out_proj")

custom_gemm_output_fwd = _make_fwd("gemm_output")
custom_gemm_output_bwd = _make_bwd("gemm_output")


def _clear_custom_gemm_cache() -> None:
    _warned.clear()
    _kernel_cache.clear()


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402, I001
_register_reset_hook(_clear_custom_gemm_cache)
