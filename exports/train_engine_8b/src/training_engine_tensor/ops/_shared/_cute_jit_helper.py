"""CuTeDSL JIT scaffolding shared by the self-developed GEMM kernels.

This module wraps the boilerplate around ``cute.compile`` /
``@cute.jit`` so each per-operator ``kernel.py`` does not have to
reinvent the SDK glue:

* Cached canonical import of ``cutlass`` / ``cutlass.cute`` /
  hopper helpers / ``cutlass.pipeline`` / ``cutlass.torch``.
* ``torch.Tensor`` → CuTe-tensor conversion via
  ``cutlass.torch.cute_tensor_like`` with a ``layout_hint`` parameter
  that documents the kernel's intent (row-major / column-major).
* A small wrapper around ``cute.compile`` with a process-wide
  in-memory cache, so a per-op fwd/dgrad/wgrad pair only re-traces
  through ``cute.compile`` once per Python process.  CuTeDSL itself
  already disk-caches the lowered MLIR / SASS through
  ``CUTE_DSL_CACHE_DIR``; this in-process cache only saves the
  graph-trace cost on the second call.

Typical ``kernel.py`` usage::

    import torch
    from training_engine_tensor.ops._shared._cute_jit_helper import (
        import_cute_modules,
        torch_to_cute,
        make_cuda_stream,
        get_max_active_clusters,
        compile_jit,
    )

    cute, cutlass, sm90_utils, pipeline, cutlass_torch = import_cute_modules()

    @cute.jit
    def _fwd_v1(a, b, c, max_active, stream):
        ...

    def gemm_<op>_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape[0], x.shape[1], w.shape[0],
                          dtype=torch.bfloat16, device=x.device)
        a_t = torch_to_cute(x, layout_hint="row_major")
        b_t = torch_to_cute(w, layout_hint="row_major")
        c_t = torch_to_cute(out, layout_hint="row_major")
        stream = make_cuda_stream()
        cluster_mn = (1, 1)
        max_active = get_max_active_clusters(cluster_mn)
        compiled = compile_jit(
            _fwd_v1, a_t, b_t, c_t, max_active, stream,
            cache_key=("gemm_<op>_fwd_v1", x.shape, w.shape),
        )
        compiled(a_t, b_t, c_t, max_active, stream)
        return out

The kernel body, tile / cluster / occupancy choices, and pipeline
wiring all live in ``kernel.py``; this module only removes the
SDK-discovery friction.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Hashable, Optional


__all__ = [
    "BF16",
    "FP32",
    "compile_jit",
    "get_max_active_clusters",
    "import_cute_modules",
    "make_cuda_stream",
    "reset_jit_cache",
    "torch_to_cute",
]


# ---------------------------------------------------------------------------
# Cached canonical CuTeDSL imports.
# ---------------------------------------------------------------------------

_IMPORT_RESULT: Optional[tuple] = None


def import_cute_modules() -> tuple[Any, Any, Any, Any, Any]:
    """Return ``(cute, cutlass, sm90_utils, pipeline, cutlass_torch)``.

    Idempotent — first call imports + memoises the result, subsequent
    calls return the cached tuple.  Raises a precise
    :class:`RuntimeError` when the cutlass-dsl wheel is not importable.
    """
    global _IMPORT_RESULT
    if _IMPORT_RESULT is not None:
        return _IMPORT_RESULT
    try:
        import cutlass  # type: ignore
        import cutlass.cute as cute  # type: ignore
        import cutlass.utils.hopper_helpers as sm90_utils  # type: ignore
        import cutlass.pipeline as pipeline  # type: ignore
        import cutlass.torch as cutlass_torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CuTeDSL import failed.  The host needs the NVIDIA "
            "cutlass-dsl wheel; install it via pip or point "
            "CUTLASS_DSL_PATH at an unpacked tree before importing "
            "training_engine_tensor.ops.gemm_*.kernel.\n"
            f"  underlying error: {type(exc).__name__}: {exc}",
        ) from exc
    _IMPORT_RESULT = (cute, cutlass, sm90_utils, pipeline, cutlass_torch)
    return _IMPORT_RESULT


# ---------------------------------------------------------------------------
# Convenience dtype shortcuts.
# ---------------------------------------------------------------------------

class _LazyCutlassDtype:
    """Wrapper that resolves to ``cutlass.<attr>`` on first access.

    Allows ``BF16`` / ``FP32`` to be imported from this module without
    forcing CuTeDSL to be present at import time (useful for static
    analysis / linting on hosts without the wheel installed).
    """

    def __init__(self, attr: str):
        self._attr = attr
        self._resolved: Any = None

    def _resolve(self) -> Any:
        if self._resolved is None:
            import cutlass  # type: ignore
            self._resolved = getattr(cutlass, self._attr)
        return self._resolved

    def __call__(self, *a, **kw):
        return self._resolve()(*a, **kw)

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self):
        return f"<lazy cutlass.{self._attr}>"


BF16 = _LazyCutlassDtype("BFloat16")
FP32 = _LazyCutlassDtype("Float32")


# ---------------------------------------------------------------------------
# torch.Tensor ↔ cute.Tensor glue.
# ---------------------------------------------------------------------------

def torch_to_cute(
    t: Any,
    *,
    layout_hint: str = "row_major",
    is_dynamic_layout: bool = True,
    assumed_align: int = 16,
):
    """Convert a ``torch.Tensor`` into the ``(cute_tensor, gpu_tensor)``
    pair that a compiled CuTeDSL kernel expects.

    Parameters
    ----------
    t : torch.Tensor
        The producer tensor (BF16 / FP32, on CUDA).  Must already be
        contiguous in the dimension implied by ``layout_hint`` (the
        caller is responsible for the right ``.contiguous()`` /
        ``.t()`` / ``.view(...)`` reshape).
    layout_hint : ``"row_major"`` or ``"col_major"``
        Documents the caller's intent.  Not enforced at runtime —
        CuTeDSL derives the actual layout from the tensor's stride.
    is_dynamic_layout : bool
        Forwarded to ``cutlass.torch.cute_tensor_like``.  Use
        ``False`` for static-shape forward / dgrad paths and ``True``
        for column-major transpose views in wgrad (where strides are
        not compile-time constants).
    assumed_align : int
        Tensor base-pointer alignment in bytes.  ``16`` matches the
        WGMMA / TMA requirement for BF16 / FP32 operands.
    """
    _, _, _, _, cutlass_torch = import_cute_modules()
    pair = cutlass_torch.cute_tensor_like(
        t,
        t.dtype_cutlass if hasattr(t, "dtype_cutlass") else None,
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=assumed_align,
    ) if hasattr(t, "dtype_cutlass") else cutlass_torch.cute_tensor_like(
        t,
        _torch_dtype_to_cutlass(t.dtype),
        is_dynamic_layout=is_dynamic_layout,
        assumed_align=assumed_align,
    )
    return pair


def _torch_dtype_to_cutlass(dtype: Any) -> Any:
    """Map ``torch.dtype`` to the matching ``cutlass`` dtype class."""
    import cutlass  # type: ignore
    import torch  # type: ignore

    return {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]


def make_cuda_stream(torch_stream=None):
    """Wrap a ``torch.cuda.Stream`` into the ``cuda.CUstream`` handle.

    Compiled CuTeDSL kernels accept a ``CUstream`` as their last
    positional argument.  When ``torch_stream`` is ``None``, a fresh
    ``torch.cuda.Stream()`` is created on the current device.
    """
    import cuda.bindings.driver as cuda  # type: ignore
    import torch  # type: ignore

    if torch_stream is None:
        torch_stream = torch.cuda.Stream()
    return cuda.CUstream(torch_stream.cuda_stream)


def get_max_active_clusters(cluster_shape_mn: tuple[int, int]) -> int:
    """Resolve ``max_active_clusters`` for a given cluster shape.

    Equivalent to::

        hw = cutlass.utils.HardwareInfo()
        max_active = hw.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

    Pinned in a helper so kernels can write a single function call
    instead of repeating the API.
    """
    import cutlass  # type: ignore

    hw = cutlass.utils.HardwareInfo()
    return hw.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])


# ---------------------------------------------------------------------------
# cute.compile wrapper with in-process memoisation.
# ---------------------------------------------------------------------------

_JIT_CACHE: dict[Hashable, Any] = {}


def compile_jit(
    kernel: Callable,
    *args,
    cache_key: Optional[Hashable] = None,
    verbose: bool = True,
) -> Any:
    """One-call ``cute.compile`` wrapper.

    Parameters
    ----------
    kernel : Callable
        The ``@cute.jit``-decorated callable to compile.
    *args :
        Compile-time-erased arguments forwarded to ``cute.compile``.
        These are the same tensors / scalars the runtime invocation
        will pass — CuTeDSL inspects their shapes / dtypes to
        specialise.
    cache_key : Hashable, optional
        Process-wide in-memory cache key.  When non-None, repeat
        calls with the same key return the cached compiled object
        instead of re-invoking ``cute.compile``.
    verbose : bool
        Print a ``[compile_jit] <kernel> compiled in Xms`` line so
        startup logs show the JIT cost.  Set ``False`` inside hot
        paths.

    Returns
    -------
    The callable returned by ``cute.compile``; invoke with the same
    runtime args as the JITed callable.

    Raises
    ------
    RuntimeError
        With the original CuTeDSL traceback chained.
    """
    cute, _, _, _, _ = import_cute_modules()

    if cache_key is not None and cache_key in _JIT_CACHE:
        if verbose:
            print(f"[compile_jit] {cache_key!r} cached (in-process)", flush=True)
        return _JIT_CACHE[cache_key]

    t0 = time.time()
    try:
        compiled = cute.compile(kernel, *args)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"cute.compile failed for "
            f"kernel={getattr(kernel, '__name__', repr(kernel))} "
            f"cache_key={cache_key!r}\n"
            f"  original: {type(exc).__name__}: {exc}",
        ) from exc
    elapsed_ms = (time.time() - t0) * 1000.0
    if verbose:
        name = getattr(kernel, "__name__", repr(kernel))
        print(
            f"[compile_jit] {name} ({cache_key!r}) compiled in "
            f"{elapsed_ms:.1f} ms",
            flush=True,
        )
    if cache_key is not None:
        _JIT_CACHE[cache_key] = compiled
    return compiled


def reset_jit_cache() -> None:
    """Drop the in-process compile cache.

    Tests / sweeps that JIT-compile many configurations can call this
    between configurations to bound memory.
    """
    _JIT_CACHE.clear()
