"""JIT compilation backend for in-house persistent GEMM kernels.

Uses memref descriptor pointer rebinding for zero-copy data passing:
the cute tensor's _memref_desc.aligned field is set to the framework
tensor's data pointer before each kernel launch.

Compiled kernels are cached by (cfg_name, M, N, K, L, c_dtype).
"""
from __future__ import annotations

import threading
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

from training_engine_tensor.ops._gemm_inhouse_kernel import (
    M1PersistentGemmKernel,
    PERSISTENT_CONFIGS,
)

_lock = threading.Lock()
_compiled_cache: dict[str, tuple] = {}
_hw_info = None
_logged: set[str] = set()


def _get_hw():
    global _hw_info
    if _hw_info is None:
        _hw_info = cutlass.utils.HardwareInfo()
    return _hw_info


import ctypes

class _MemrefDesc3D(ctypes.Structure):
    """MLIR memref descriptor for a 3D tensor — matches the layout emitted by
    CuTeDSL for cute_tensor_like(matrix(L=1,M,K,...), is_dynamic_layout=False).
    """
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned",   ctypes.c_void_p),
        ("offset",    ctypes.c_int64),
        ("sizes",     ctypes.c_int64 * 3),
        ("strides",   ctypes.c_int64 * 3),
    ]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]

_capsule_name_cache: bytes | None = None


def _rebind(ct, ptr: int):
    """Rebind a cute tensor's memref to a different GPU data pointer.

    Directly patches the MLIR memref descriptor (allocated + aligned fields)
    inside the PyCapsule so the compiled kernel reads from / writes to the
    new GPU address.  Zero-copy — no data movement.
    """
    global _capsule_name_cache
    capsule = ct._memref_desc
    if _capsule_name_cache is None:
        _capsule_name_cache = ctypes.pythonapi.PyCapsule_GetName(capsule)
    raw = ctypes.pythonapi.PyCapsule_GetPointer(capsule, _capsule_name_cache)
    desc = _MemrefDesc3D.from_address(raw)
    desc.allocated = ptr
    desc.aligned = ptr


def _compile_cached(cfg_name: str, M: int, N: int, K: int,
                    a_major: str, b_major: str, c_major: str,
                    c_dtype, L: int = 1):
    """Return (compiled, a_t, b_t, c_t)."""
    cache_key = f"{cfg_name}_{M}_{N}_{K}_{L}_{c_dtype}"
    cached = _compiled_cache.get(cache_key)
    if cached is not None:
        return cached
    with _lock:
        cached = _compiled_cache.get(cache_key)
        if cached is not None:
            return cached

        cfg = PERSISTENT_CONFIGS[cfg_name]
        tile = cfg["tile_mn"]
        cluster = cfg["cluster_mn"]
        swizzle = cfg.get("swizzle", 1)
        raster_m = cfg.get("raster_m", False)

        a_cpu = cutlass_torch.matrix(L, M, K, a_major == "m", cutlass.BFloat16)
        b_cpu = cutlass_torch.matrix(L, N, K, b_major == "n", cutlass.BFloat16)
        c_cpu = cutlass_torch.matrix(L, M, N, c_major == "m", c_dtype)

        a_t, _ = cutlass_torch.cute_tensor_like(
            a_cpu, cutlass.BFloat16, is_dynamic_layout=False, assumed_align=16)
        b_t, _ = cutlass_torch.cute_tensor_like(
            b_cpu, cutlass.BFloat16, is_dynamic_layout=False, assumed_align=16)
        c_t, _ = cutlass_torch.cute_tensor_like(
            c_cpu, c_dtype, is_dynamic_layout=False, assumed_align=16)

        gemm = M1PersistentGemmKernel(cutlass.Float32, tile, cluster,
                                       swizzle_size=swizzle, raster_along_m=raster_m)
        hw = _get_hw()
        max_active = hw.get_max_active_clusters(cluster[0] * cluster[1])
        s = torch.cuda.Stream()
        cs = cuda.CUstream(s.cuda_stream)
        compiled = cute.compile(gemm, a_t, b_t, c_t, max_active, cs)
        with torch.cuda.stream(s):
            compiled(a_t, b_t, c_t, cs)
        torch.cuda.synchronize()

        entry = (compiled, a_t, b_t, c_t)
        _compiled_cache[cache_key] = entry
        if cfg_name not in _logged:
            _logged.add(cfg_name)
            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            if rank == 0:
                print(f"[inhouse_jit] compiled {cfg_name}: M={M} N={N} K={K} "
                      f"tile={tile} cluster={cluster} sw={swizzle}", flush=True)
        return entry


def _launch(compiled, a_t, b_t, c_t):
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(a_t, b_t, c_t, stream)


# ── fc1 ──────────────────────────────────────────────────────────────

def jit_gemm_fc1_fwd(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    M, K = x.shape; N = w.shape[0]
    compiled, a_t, b_t, c_t = _compile_cached(
        "fc1_fwd", M, N, K, "k", "k", "n", cutlass.BFloat16)
    _rebind(a_t, x.data_ptr())
    _rebind(b_t, w.data_ptr())
    _rebind(c_t, out.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


def jit_gemm_fc1_dgrad(dy: torch.Tensor, w: torch.Tensor, dx: torch.Tensor):
    M, N_dy = dy.shape; K_out = w.shape[1]
    compiled, a_t, b_t, c_t = _compile_cached(
        "fc1_dgrad", M, K_out, N_dy, "k", "n", "n", cutlass.BFloat16)
    _rebind(a_t, dy.data_ptr())
    _rebind(b_t, w.data_ptr())
    _rebind(c_t, dx.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


def jit_gemm_fc1_wgrad(dy: torch.Tensor, x: torch.Tensor, dw: torch.Tensor):
    M_orig, N, K = dy.shape[0], dy.shape[1], x.shape[1]
    compiled, a_t, b_t, c_t = _compile_cached(
        "fc1_wgrad", N, K, M_orig, "m", "n", "n", cutlass.Float32)
    _rebind(a_t, dy.data_ptr())
    _rebind(b_t, x.data_ptr())
    _rebind(c_t, dw.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


# ── fc2 ──────────────────────────────────────────────────────────────

def jit_gemm_fc2_fwd(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    M, K = x.shape; N = w.shape[0]
    compiled, a_t, b_t, c_t = _compile_cached(
        "fc2_fwd", M, N, K, "k", "k", "n", cutlass.BFloat16)
    _rebind(a_t, x.data_ptr())
    _rebind(b_t, w.data_ptr())
    _rebind(c_t, out.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


def jit_gemm_fc2_dgrad(dy: torch.Tensor, w: torch.Tensor, dx: torch.Tensor):
    M, N_dy = dy.shape; K_out = w.shape[1]
    compiled, a_t, b_t, c_t = _compile_cached(
        "fc2_dgrad", M, K_out, N_dy, "k", "n", "n", cutlass.BFloat16)
    _rebind(a_t, dy.data_ptr())
    _rebind(b_t, w.data_ptr())
    _rebind(c_t, dx.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


def jit_gemm_fc2_wgrad(dy: torch.Tensor, x: torch.Tensor, dw: torch.Tensor):
    M_orig, N, K = dy.shape[0], dy.shape[1], x.shape[1]
    compiled, a_t, b_t, c_t = _compile_cached(
        "fc2_wgrad", N, K, M_orig, "m", "n", "n", cutlass.Float32)
    _rebind(a_t, dy.data_ptr())
    _rebind(b_t, x.data_ptr())
    _rebind(c_t, dw.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


# ── aop ──────────────────────────────────────────────────────────────

def jit_gemm_aop_fwd(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    M, K = x.shape; N = w.shape[0]
    compiled, a_t, b_t, c_t = _compile_cached(
        "aop_fwd", M, N, K, "k", "k", "n", cutlass.BFloat16)
    _rebind(a_t, x.data_ptr())
    _rebind(b_t, w.data_ptr())
    _rebind(c_t, out.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


def jit_gemm_aop_dgrad(dy: torch.Tensor, w: torch.Tensor, dx: torch.Tensor):
    M, N_dy = dy.shape; K_out = w.shape[1]
    compiled, a_t, b_t, c_t = _compile_cached(
        "aop_dgrad", M, K_out, N_dy, "k", "n", "n", cutlass.BFloat16)
    _rebind(a_t, dy.data_ptr())
    _rebind(b_t, w.data_ptr())
    _rebind(c_t, dx.data_ptr())
    _launch(compiled, a_t, b_t, c_t)


_wgrad_partials_cache: dict[str, torch.Tensor] = {}
_wgrad_dw_cache: dict[str, torch.Tensor] = {}


def jit_gemm_aop_wgrad(dy: torch.Tensor, x: torch.Tensor, dw: torch.Tensor,
                        num_splits: int = 2):
    """aop wgrad with split-K: partition K across L batches, then reduce."""
    K_full = dy.shape[0]     # token dim = 40960
    M_wgrad = dy.shape[1]    # d_model = 1024
    N_wgrad = x.shape[1]     # d_model = 1024
    chunk_K = K_full // num_splits

    cfg_name = f"aop_wgrad_sk{num_splits}"
    compiled, a_t, b_t, c_t = _compile_cached(
        cfg_name, M_wgrad, N_wgrad, chunk_K, "m", "n", "n",
        cutlass.Float32, L=num_splits)

    # Batched views: partition K dimension into num_splits chunks
    # A = dy^T reshaped: (M_wgrad, chunk_K, num_splits), strides: (1, stride_0, chunk_K*stride_0)
    dy_stride0 = dy.stride(0)
    a_batched = torch.as_strided(
        dy, (M_wgrad, chunk_K, num_splits),
        (1, dy_stride0, chunk_K * dy_stride0))
    x_stride0 = x.stride(0)
    b_batched = torch.as_strided(
        x, (N_wgrad, chunk_K, num_splits),
        (1, x_stride0, chunk_K * x_stride0))

    # Allocate/cache partials: (num_splits, M_wgrad, N_wgrad) FP32
    dev_key = f"{x.device}_{num_splits}_{M_wgrad}_{N_wgrad}"
    partials = _wgrad_partials_cache.get(dev_key)
    if partials is None:
        partials = torch.empty(num_splits, M_wgrad, N_wgrad,
                               dtype=torch.float32, device=x.device)
        _wgrad_partials_cache[dev_key] = partials

    _rebind(a_t, a_batched.data_ptr())
    _rebind(b_t, b_batched.data_ptr())
    _rebind(c_t, partials.data_ptr())
    _launch(compiled, a_t, b_t, c_t)

    # Reduce partials → dw
    if num_splits == 2:
        torch.add(partials[0], partials[1], out=dw)
    else:
        torch.sum(partials, dim=0, out=dw)
