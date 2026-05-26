"""GEMM kernel for the ``gemm_output`` operator (LM head vocab projection).

Two-phase build (preferred):
  1. export_kernels.py compiles CuTeDSL kernels → .h + .o files (MLIR path,
     no C7510 WGMMA serialization)
  2. gemm_cutedsl.cpp links .o files into a shared library

Fallback:
  torch.utils.cpp_extension JIT-compiles gemm_sm90.cu (CUTLASS 3.x C++).
  This path has C7510 serialization (~2x slower) but always works.

Three GEMM directions:
  fwd:   logits[M,N] = X[M,K] * W[N,K]^T        BF16→BF16
  dgrad: dX[M,N]     = dY[M,K] * W_t[N,K]^T     BF16→BF16
  wgrad: dW[M,N]     = dY^T[M,K] * X_t[N,K]^T   BF16→FP32

Shapes (with vocab pad):
  fwd:   M=40960, K=1024, N=73472
  dgrad: M=40960, K=73472, N=1024
  wgrad: M=73472, K=40960, N=1024
"""

import fcntl
import hashlib
import os
import subprocess
import sys
import time
from typing import Tuple

import torch

__all__ = ["gemm_output_fwd", "gemm_output_bwd"]


from training_engine_tensor.op_dispatcher import get_build_env as _get_build_env
from training_engine_tensor.op_dispatcher import get_frozen_env as _get_frozen_env

# ── Weight-padding buffer (allocation cached, contents refreshed every call) ──

GEMM_PAD_TO: int = 128

# Persistent buffer for padded weight. The dgrad path (Round 37) consumes
# `_wt_padded` *directly* as a col-major B view (zero-copy via w_padded.t()),
# so the previous side-stream transpose into a separate `_wt_padded_t`
# buffer is no longer needed (saves 143 MB GPU memory and ~0.6 ms / step
# of side-stream DRAM traffic that previously contended with the fwd GEMM).
#
# 2026-05-08 root-cause fix for the OP_GEMM_OUTPUT=v1 end-to-end divergence
# (see ``workload/notes/stable_alignment_findings.md`` §11):
# The previous ``_ensure_w_cache`` skipped the ``weight → _wt_padded`` copy
# whenever ``weight._version`` was unchanged across calls. That is correct
# for paths whose optimizer step bumps the version (e.g. legacy
# ``sync_params_from_master`` which calls ``params[name].copy_(master)`` —
# an autograd-aware in-place op). It is **wrong** for the production path
# (``FUSE_ADAM_SYNC=1``) where the optimizer is the Triton kernel
# ``fused_adam_sync(_tensor)`` writing into ``params[name]`` via raw
# ``tl.store``: the storage ``_version`` is **not** bumped, so the cache
# hit returned the iter-0 padded weights for every subsequent forward /
# dgrad. The op-unit tests (``test_op.py``) never exercise an optimizer
# step between fwd and bwd so they could not catch this; in end-to-end
# stable from-scratch the live ``output.weight`` and the cached
# ``_wt_padded`` drifted apart, the forward used progressively staler
# weights, and lm_loss fell behind Megatron baseline starting iter ~25.
#
# Fix: keep the persistent buffer to avoid allocations, but drop the
# version check and always refresh the contents — same semantics as the
# baseline ``kernels.py::_get_padded_weight`` (which has been
# bit-exactly aligned to Megatron in production for >1000 steps). The
# extra ``copy_`` is ~150 MB / call (output GEMM is the largest weight),
# i.e. ≈0.05 ms / step on H100 HBM3 — well below noise vs the 30-50%
# loss inflation it was masking.
_wt_padded: torch.Tensor | None = None    # [V_pad, K] zero-padded weight
_wt_orig_n: int = 0

_wgrad_b_buf: torch.Tensor | None = None


def _ensure_w_cache(weight: torch.Tensor):
    """Return (w_padded, orig_n).

    The padded BUFFER is cached across calls (saves ~150 MB / step alloc
    and the zero-fill of the pad rows). The CONTENTS are refreshed on
    every call from the live ``weight`` — mirroring the baseline
    ``_get_padded_weight`` semantics, since ``weight._version`` is not
    a reliable change indicator under the Triton-based fused Adam path
    (see module-level note above).
    """
    global _wt_padded, _wt_orig_n

    orig_n = weight.shape[0]
    K = weight.shape[1]
    padded_n = (orig_n + GEMM_PAD_TO - 1) // GEMM_PAD_TO * GEMM_PAD_TO

    if (_wt_padded is None
            or _wt_padded.shape != (padded_n, K)
            or _wt_padded.dtype != weight.dtype
            or _wt_padded.device != weight.device):
        # Initial allocation — the pad rows (orig_n:padded_n) stay zero
        # for the lifetime of the buffer; we only ever overwrite
        # ``_wt_padded[:orig_n]``.
        _wt_padded = torch.zeros(
            padded_n, K, dtype=weight.dtype, device=weight.device)

    if padded_n == orig_n:
        _wt_padded.copy_(weight)
    else:
        _wt_padded[:orig_n].copy_(weight)

    _wt_orig_n = orig_n
    return _wt_padded, orig_n


# ── CuTeDSL C-export infrastructure ───────────────────────────────────

# Shared-cache convention (see workload/ops/gemm_fc1/kernel.py).
EXPORT_DIR = os.path.join(
    _get_build_env().cutedsl_cache_root,
    "cutedsl_export_gemm_output",
)
_LOCK_PATH = os.path.join(EXPORT_DIR, ".export.lock")
_DIRECTIONS = ("gemm_output_fwd", "gemm_output_dgrad", "gemm_output_wgrad")
_MIN_OBJ_SIZE = 4096


def _try_cutedsl_export():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(src_dir, "export_kernels.py")
    if not os.path.exists(export_script):
        return False

    os.makedirs(EXPORT_DIR, exist_ok=True)

    needed = [os.path.join(EXPORT_DIR, d, f"{d}.o") for d in _DIRECTIONS]
    headers = [os.path.join(EXPORT_DIR, d, f"{d}.h") for d in _DIRECTIONS]
    all_files = needed + headers

    h = hashlib.md5()
    with open(export_script, "rb") as f:
        h.update(f.read())
    current_hash = h.hexdigest()
    config_hash_path = os.path.join(EXPORT_DIR, ".config_hash")

    def _files_valid():
        for p in all_files:
            if not os.path.exists(p):
                return False
        for p in needed:
            if os.path.getsize(p) < _MIN_OBJ_SIZE:
                return False
        return True

    def _hash_matches():
        if not os.path.exists(config_hash_path):
            return False
        try:
            with open(config_hash_path) as f:
                return f.read().strip() == current_hash
        except OSError:
            return False

    if _hash_matches() and _files_valid():
        return True

    rank = _get_build_env().local_rank

    if rank == 0:
        lock_fd = open(_LOCK_PATH, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if _hash_matches() and _files_valid():
                return True

            result = subprocess.run(
                [sys.executable, export_script],
                cwd=src_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print(f"CuTeDSL export failed (rc={result.returncode}): "
                      f"{result.stderr[:500]}", file=sys.stderr)
                return False
            if not _files_valid():
                print("CuTeDSL export produced incomplete files",
                      file=sys.stderr)
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"CuTeDSL export error: {e}", file=sys.stderr)
            return False
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    else:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if _hash_matches() and _files_valid():
                return True
            time.sleep(2)
        print(f"CuTeDSL export: rank {rank} timed out waiting for rank 0",
              file=sys.stderr)
        return False


_CUDART_PRELOAD = "/usr/local/cuda/lib64/libcudart.so.12"


def _cutedsl_preload_ok() -> bool:
    """CuTeDSL C-export needs the system's libcudart to be loaded before
    PyTorch's bundled one (which may lack CUDA 12.4+ Library APIs).
    Returns True if LD_PRELOAD contains the system libcudart."""
    return _CUDART_PRELOAD in _get_frozen_env().get("LD_PRELOAD", "")


def _load_cutedsl_ext():
    from torch.utils.cpp_extension import load

    cuda_home = _get_build_env().cuda_home

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(src_dir, "gemm_cutedsl.cpp")
    obj_files = [os.path.join(EXPORT_DIR, d, f"{d}.o") for d in _DIRECTIONS]
    header_files = [os.path.join(EXPORT_DIR, d, f"{d}.h") for d in _DIRECTIONS]

    h = hashlib.md5()
    for p in header_files + obj_files:
        with open(p, "rb") as f:
            h.update(f.read())
    build_hash = h.hexdigest()[:8]
    ext_name = f"gemm_output_cutedsl_{build_hash}"

    include_dirs = [
        os.path.join(EXPORT_DIR, d) for d in _DIRECTIONS
    ] + [os.path.join(cuda_home, "include")]

    runtime_lib = "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a"
    if not os.path.exists(runtime_lib):
        import importlib.util
        spec = importlib.util.find_spec("nvidia_cutlass_dsl")
        if spec and spec.submodule_search_locations:
            alt = os.path.join(spec.submodule_search_locations[0],
                               "lib", "libcuda_dialect_runtime_static.a")
            if os.path.exists(alt):
                runtime_lib = alt
    if not os.path.exists(runtime_lib):
        _candidates = [
            "/opt/cutlass_dsl/nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a",
            "/usr/local/lib/python3.12/dist-packages/cutlass_dsl/lib/libcuda_dialect_runtime_static.a",
        ]
        # Shared-FS layout: caller exports ``CUTLASS_DSL_FALLBACK_DIR``
        # pointing at the package root that holds ``lib/`` and
        # ``python_packages/`` (extracted manually on devspaces where
        # nvidia-cutlass-dsl-libs-base is not pip-installed by default).
        _fb_dir = os.environ.get("CUTLASS_DSL_FALLBACK_DIR", "")
        if _fb_dir:
            _candidates.append(os.path.join(_fb_dir, "lib",
                                            "libcuda_dialect_runtime_static.a"))
        for cand in _candidates:
            if os.path.exists(cand):
                runtime_lib = cand
                break

    cuda_lib64 = os.path.join(cuda_home, "lib64")
    ext = load(
        name=ext_name,
        sources=[cpp_src],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3"],
        extra_ldflags=obj_files + [
            "-Wl,--whole-archive", runtime_lib, "-Wl,--no-whole-archive",
            "-lcuda",
            "-L" + cuda_lib64, "-lcudart",
            "-Wl,-rpath," + cuda_lib64,
        ],
        verbose=True,
    )
    return ext


def _load_cutlass_cpp_ext():
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_src = os.path.join(src_dir, "gemm_sm90.cu")
    cuda_home = _get_build_env().cuda_home

    h = hashlib.md5()
    with open(cuda_src, "rb") as f:
        h.update(f.read())
    build_hash = h.hexdigest()[:8]

    ext = load(
        name=f"gemm_output_sm90_{build_hash}",
        sources=[cuda_src],
        extra_include_paths=[
            "/usr/include",
            os.path.join(cuda_home, "include"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-Xptxas=--allow-expensive-optimizations=true",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )
    return ext


# ── CuTeDSL Python JIT backend ────────────────────────────────────────

class _CuTeDSLJITBackend:
    """MLIR-compiled GEMM via CuTeDSL Python JIT (zero-copy, no nvcc)."""

    def __init__(self):
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from cuda.bindings import driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack
        from cutlass.torch import get_leading_dim

        self._from_dlpack = from_dlpack
        self._get_leading_dim = get_leading_dim
        self._cuda_driver = cuda_driver

        ts = torch.cuda.Stream()
        stream = cuda_driver.CUstream(ts.cuda_stream)

        def _dummy(M, K, col_major, dtype):
            cpu = cutlass_torch.matrix(1, M, K, col_major, dtype)
            t, _ = cutlass_torch.cute_tensor_like(
                cpu, dtype, is_dynamic_layout=True, assumed_align=16)
            return t

        a_bf = _dummy(256, 256, False, cutlass.BFloat16)
        b_bf = _dummy(256, 256, False, cutlass.BFloat16)
        c_bf = _dummy(256, 256, False, cutlass.BFloat16)

        from .export_kernels import HopperGemmKernel

        self._fwd = cute.compile(
            HopperGemmKernel(cutlass.Float32, (128, 256), (2, 1)),
            a_bf, b_bf, c_bf, stream)

        # Round 37: dgrad now uses col-major B (B = w_padded.t() view).
        b_col = _dummy(256, 256, True, cutlass.BFloat16)
        self._dgrad = cute.compile(
            HopperGemmKernel(cutlass.Float32, (128, 256), (2, 1)),
            a_bf, b_col, c_bf, stream)

        a_col = _dummy(256, 256, True, cutlass.BFloat16)
        c_fp32 = _dummy(256, 256, False, cutlass.Float32)
        self._wgrad = cute.compile(
            HopperGemmKernel(cutlass.Float32, (128, 256), (1, 1)),
            a_col, b_col, c_fp32, stream)

        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _to_cute(self, tensor_2d):
        t3 = tensor_2d.unsqueeze(-1)
        ct = self._from_dlpack(t3, assumed_align=16)
        ld = self._get_leading_dim(t3)
        return ct.mark_layout_dynamic(leading_dim=ld)

    def _stream(self):
        return self._cuda_driver.CUstream(
            torch.cuda.current_stream().cuda_stream)

    def gemm_fwd(self, x, w, out):
        self._fwd(self._to_cute(x), self._to_cute(w),
                  self._to_cute(out), self._stream())

    def gemm_bwd(self, dy, w_padded, dx, x, dw):
        s = self._stream()
        # Round 37: dgrad uses col-major B view of w_padded (no pre-transpose).
        # w_padded is [V_pad=K, K_w=N] row-major; w_padded.t() is the same
        # memory exposed as B[N, K] col-major (leading_dim = N).
        w_col = w_padded.t().unsqueeze(-1)
        ct_b_dg = self._from_dlpack(w_col, assumed_align=16)
        ct_b_dg = ct_b_dg.mark_layout_dynamic(
            leading_dim=self._get_leading_dim(w_col),
        )
        self._dgrad(self._to_cute(dy), ct_b_dg, self._to_cute(dx), s)
        # wgrad: A is dY^T (col-major view of dy); B is X^T (col-major view of x).
        # No copies — both are zero-cost views.
        dy_col = dy.t().unsqueeze(-1)
        ct_a = self._from_dlpack(dy_col, assumed_align=16)
        ct_a = ct_a.mark_layout_dynamic(leading_dim=self._get_leading_dim(dy_col))
        x_col = x.t().unsqueeze(-1)
        ct_b = self._from_dlpack(x_col, assumed_align=16)
        ct_b = ct_b.mark_layout_dynamic(leading_dim=self._get_leading_dim(x_col))
        self._wgrad(ct_a, ct_b, self._to_cute(dw), s)


def _try_cutedsl_jit():
    try:
        return _CuTeDSLJITBackend()
    except Exception:
        return None


# ── Extension loading ──────────────────────────────────────────────────

_ext = None
_ext_type = None


def _get_ext():
    global _ext, _ext_type
    if _ext is not None:
        return _ext

    rank = _get_build_env().local_rank

    if _cutedsl_preload_ok() and _try_cutedsl_export():
        try:
            _ext = _load_cutedsl_ext()
            _ext_type = "cutedsl"
            if rank == 0:
                print("[gemm_output] loaded CuTeDSL C-export backend",
                      flush=True)
            return _ext
        except Exception as e:
            if rank == 0:
                print(f"[gemm_output] CuTeDSL C-export load failed: {e}",
                      file=sys.stderr, flush=True)
    elif rank == 0 and _try_cutedsl_export() and not _cutedsl_preload_ok():
        print("[gemm_output] CuTeDSL exports available but LD_PRELOAD not set",
              flush=True)

    jit_backend = _try_cutedsl_jit()
    if jit_backend is not None:
        _ext = jit_backend
        _ext_type = "cutedsl_jit"
        if rank == 0:
            print("[gemm_output] loaded CuTeDSL Python JIT backend",
                  flush=True)
        return _ext

    _ext = _load_cutlass_cpp_ext()
    _ext_type = "cutlass_cpp"
    if rank == 0:
        print("[gemm_output] loaded CUTLASS C++ fallback", flush=True)
    return _ext


# ── Public API ─────────────────────────────────────────────────────────


# Reduce per-call overhead: cache the ext reference + the cutedsl-vs-cpp
# dispatch flag so both fwd and bwd can skip the _get_ext() dict lookup
# and the _ext_type string compare on the hot path.
_cached_ext = None
_cached_ext_is_cpp = False


def _hot_ext():
    global _cached_ext, _cached_ext_is_cpp
    if _cached_ext is None:
        _cached_ext = _get_ext()
        _cached_ext_is_cpp = (_ext_type == "cutlass_cpp")
    return _cached_ext


# Round 45: fast-path callables that fold weight-padding / contiguous /
# reshape / alloc / as_strided into a single pybind11 C++ call, mirroring
# gemm_fc1 R37/R39.  Set on first call; None if the C-export backend
# doesn't expose the _fast bindings (graceful fallback to the default path).
_BOUND_FWD_FAST = None
_BOUND_BWD_FAST = None
_FAST_PATHS_TRIED = False


def _try_bind_fast_paths():
    global _BOUND_FWD_FAST, _BOUND_BWD_FAST, _FAST_PATHS_TRIED
    if _FAST_PATHS_TRIED:
        return
    _FAST_PATHS_TRIED = True
    ext = _hot_ext()
    if _ext_type == "cutedsl":
        try:
            _BOUND_FWD_FAST = ext.gemm_fwd_fast
            _BOUND_BWD_FAST = ext.gemm_bwd_fast
        except AttributeError:
            pass


def gemm_output_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Forward: logits = x @ weight.T with vocab padding.

    x:      [S, B, 1024] BF16
    weight: [73449, 1024] BF16
    Returns [S, B, 73449] BF16 (as-strided over a [S*B, 73472] storage).
    """
    if not _FAST_PATHS_TRIED:
        _try_bind_fast_paths()
    fn = _BOUND_FWD_FAST
    if fn is not None:
        return fn(x, weight)

    shape = x.shape
    K = shape[-1]
    if x.is_contiguous():
        x_2d = x.view(-1, K)
    else:
        x_2d = x.reshape(-1, K).contiguous()
    w_padded, orig_n = _ensure_w_cache(weight)

    M = x_2d.shape[0]
    N = w_padded.shape[0]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)

    _hot_ext().gemm_fwd(x_2d, w_padded, out)

    if len(shape) <= 2:
        return out[:, :orig_n]
    row_stride = out.stride(0)
    return torch.as_strided(
        out, (*shape[:-1], orig_n),
        (row_stride * shape[1], row_stride, 1),
        storage_offset=0,
    )


def gemm_output_bwd(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward: dgrad + wgrad.

    d_output: [S, B, 73472] BF16  (padded by CE kernel)
    x:        [S, B, 1024]  BF16
    weight:   [73449, 1024]  BF16
    Returns (d_input [S,B,1024] BF16, d_weight [73449,1024] FP32).
    """
    if not _FAST_PATHS_TRIED:
        _try_bind_fast_paths()
    fn = _BOUND_BWD_FAST
    if fn is not None:
        return fn(d_output, x, weight)

    if d_output.is_contiguous():
        d_out_2d = d_output.view(-1, d_output.shape[-1])
    else:
        d_out_2d = d_output.reshape(-1, d_output.shape[-1]).contiguous()
    if x.is_contiguous():
        x_2d = x.view(-1, x.shape[-1])
    else:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

    w_padded, orig_n = _ensure_w_cache(weight)

    d_input_2d = torch.empty(
        d_out_2d.shape[0], w_padded.shape[1],
        dtype=x.dtype, device=x.device,
    )
    d_weight_full = torch.empty(
        w_padded.shape[0], x_2d.shape[1],
        dtype=torch.float32, device=x.device,
    )

    ext = _hot_ext()
    # CuTeDSL backends (C-export & JIT) consume w_padded + x_2d directly:
    #  - dgrad's B is compiled with b_col_major=True (Round 37) so it
    #    interprets w_padded's row-major storage as a col-major B[N, K]
    #    view (no transpose copy, no `_wt_padded_t` buffer).
    #  - wgrad's B is also compiled with b_col_major=True so it
    #    interprets x_2d's row-major storage as col-major B (no copy).
    # The CUTLASS C++ fallback still expects pre-transposed row-major Bs
    # and needs explicit transpose copies.
    if _cached_ext_is_cpp:
        global _wgrad_b_buf
        wgrad_b_shape = (x_2d.shape[1], x_2d.shape[0])
        if (_wgrad_b_buf is None
                or _wgrad_b_buf.shape != wgrad_b_shape
                or _wgrad_b_buf.dtype != x_2d.dtype
                or _wgrad_b_buf.device != x_2d.device):
            _wgrad_b_buf = torch.empty(
                *wgrad_b_shape, dtype=x_2d.dtype, device=x_2d.device,
            )
        _wgrad_b_buf.copy_(x_2d.t())
        b_dgrad_cpp = w_padded.t().contiguous()
        ext.gemm_bwd(d_out_2d, b_dgrad_cpp, d_input_2d,
                     _wgrad_b_buf, d_weight_full)
    else:
        ext.gemm_bwd(d_out_2d, w_padded, d_input_2d, x_2d, d_weight_full)

    d_input = d_input_2d.view_as(x)
    d_weight = d_weight_full[:orig_n]

    return d_input, d_weight
