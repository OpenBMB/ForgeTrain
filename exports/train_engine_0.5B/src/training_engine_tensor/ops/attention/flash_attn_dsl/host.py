"""Python host-side wrapper that drives the DSL kernel.

Responsibilities:
- Bridge PyTorch tensors to CuTe tensors via `from_dlpack`.
- JIT-compile the kernel once per (dtype, is_causal) combination and cache it.
- Resolve `softmax_scale` default and forward the current CUDA stream.

Exported names: run_flash_fwd, run_flash_bwd.
"""

import importlib
import math
import os
import re
import sys
from typing import Optional

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _ensure_cutlass_dsl_cuda_env() -> None:
    """Hint SM target for CuTe DSL JIT (clearer diagnostics than an unknown arch).

    Do not prepend ``/usr/local/cuda/bin`` here: on some DevSpace images that shadows the
    toolchain paired with PyTorch and DSL, triggering ``CUDA_ERROR_LAUNCH_FAILED`` (719).
    """
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")


_ensure_cutlass_dsl_cuda_env()

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def _patch_swizzle_type():
    """Add num_bits/num_base/num_shift to SwizzleType if missing (cutlass-dsl <4.5 compat)."""
    try:
        from cutlass._mlir.dialects import cute as _cute_dialect
        SwzT = _cute_dialect.SwizzleType
        if hasattr(SwzT, 'num_bits'):
            return
        _swz_re = re.compile(r'<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>')

        def _params(self):
            m = _swz_re.search(str(self))
            return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)

        SwzT.num_bits = property(lambda self: _params(self)[0])
        SwzT.num_base = property(lambda self: _params(self)[1])
        SwzT.num_shift = property(lambda self: _params(self)[2])
    except Exception:
        pass

_patch_swizzle_type()


_TORCH_TO_CUTE_DTYPE = None


def _dtype_map():
    global _TORCH_TO_CUTE_DTYPE
    if _TORCH_TO_CUTE_DTYPE is None:
        import torch
        _TORCH_TO_CUTE_DTYPE = {
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
            torch.float32: cutlass.Float32,
        }
    return _TORCH_TO_CUTE_DTYPE


_compile_cache: dict[tuple, object] = {}


def _compile_key(dtype, head_dim, is_causal, num_stages, n_block, qhead_per_kvhead):
    return (dtype, head_dim, is_causal, num_stages, n_block, qhead_per_kvhead)


def _to_cute_tensor(t):
    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)


def _to_cute_lse(t):
    return from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim - 1)


def run_flash_fwd(
    q,
    k,
    v,
    out,
    lse,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    num_stages: int = 2,
    n_block: int = 128,
):
    """Run FA3 forward kernel.

    q/out: [B, H_q, N, D], k/v: [B, H_kv, N, D] (native GQA) or [B, H_q, N, D].
    lse: [B, H_q, N].
    """
    import torch

    if q.is_cuda:
        torch.cuda.set_device(q.device)

    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    qhead_per_kvhead = H_q // H_kv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    cute_dtype = _dtype_map()[q.dtype]
    key = _compile_key(cute_dtype, D, bool(is_causal), int(num_stages), int(n_block), qhead_per_kvhead)

    # FA3 expects (B, S, H, D) layout; PyTorch convention is (B, H, N, D).
    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2)
    out_bshd = out.transpose(1, 2)

    qT = _to_cute_tensor(q_bshd)
    kT = _to_cute_tensor(k_bshd)
    vT = _to_cute_tensor(v_bshd)
    oT = _to_cute_tensor(out_bshd)
    lT = _to_cute_lse(lse) if lse is not None else None

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # FA4 (flash_attn.cute.flash_fwd_sm90) is untracked → not available on
    # cctl `git clone` builds. Fall back to in-tree DSL fwd kernel when the
    # FA4 module isn't importable. The fallback expects GQA-expanded K/V
    # (H_q heads), so the caller is responsible for that on the GQA path.
    try:
        from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
        _HAS_FA4 = True
    except ImportError:
        _HAS_FA4 = False

    if key not in _compile_cache:
        import time as _t
        _c0 = _t.time()
        if _HAS_FA4:
            fa = FlashAttentionForwardSm90(
                dtype=cute_dtype,
                head_dim=D,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=bool(is_causal),
                pack_gqa=(qhead_per_kvhead > 1),
                tile_m=128,
                tile_n=n_block,
                num_stages=num_stages,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
            )
            print(f"[fwd-fa4] cute.compile start (causal={is_causal}, gqa={qhead_per_kvhead})", flush=True)
            _compile_cache[key] = cute.compile(
                fa, qT, kT, vT, oT, lT, cutlass.Float32(softmax_scale),
                None, None, None, None,
                None,
                None, None,
                None,
                None,
                None,
                stream,
            )
            print(f"[fwd-fa4] cute.compile done ({_t.time()-_c0:.1f}s)", flush=True)
        else:
            # DSL fwd path expects (B, H, N, D) layout (not transposed).
            qT_bhnd = _to_cute_tensor(q)
            kT_bhnd = _to_cute_tensor(k)
            vT_bhnd = _to_cute_tensor(v)
            oT_bhnd = _to_cute_tensor(out)
            from .flash_fwd import FlashAttnFwdSm90
            fa_dsl = FlashAttnFwdSm90(
                dtype=cute_dtype,
                head_dim=D,
                m_block_size=64,
                n_block_size=n_block,
                num_stages=num_stages,
                is_causal=bool(is_causal),
            )
            print(f"[fwd-dsl] cute.compile start (causal={is_causal}, gqa={qhead_per_kvhead})", flush=True)
            _compile_cache[key] = cute.compile(
                fa_dsl, qT_bhnd, kT_bhnd, vT_bhnd, oT_bhnd, lT,
                cutlass.Float32(softmax_scale), stream,
            )
            print(f"[fwd-dsl] cute.compile done ({_t.time()-_c0:.1f}s)", flush=True)

    if _HAS_FA4:
        _compile_cache[key](
            qT, kT, vT, oT, lT, cutlass.Float32(softmax_scale),
            None, None, None, None,
            None,
            None, None,
            None,
            None,
            None,
            stream,
        )
    else:
        qT_bhnd = _to_cute_tensor(q)
        kT_bhnd = _to_cute_tensor(k)
        vT_bhnd = _to_cute_tensor(v)
        oT_bhnd = _to_cute_tensor(out)
        _compile_cache[key](
            qT_bhnd, kT_bhnd, vT_bhnd, oT_bhnd, lT,
            cutlass.Float32(softmax_scale), stream,
        )
    return out


_bwd_cpp_module = None


def _get_bwd_cpp_module():
    """JIT-compile the CUDA C++ backward kernel on first use."""
    global _bwd_cpp_module
    if _bwd_cpp_module is not None:
        return _bwd_cpp_module

    import torch

    csrc_dir = os.path.join(_PKG_ROOT, "csrc")
    cutlass_include = os.path.join(_PKG_ROOT, "third_party", "cutlass", "include")
    cu_src = os.path.join(csrc_dir, "flash_bwd_cutlass.cu")

    if not os.path.isfile(cu_src):
        raise RuntimeError(f"C++ backward source not found: {cu_src}")

    from torch.utils.cpp_extension import load
    import torch.utils.cpp_extension as _ext

    _props = torch.cuda.get_device_properties(0)
    _arch = _props.major * 10 + _props.minor
    _suf = "a" if _arch >= 90 else ""
    _gencode = f"-gencode=arch=compute_{_arch}{_suf},code=sm_{_arch}{_suf}"
    _ext._get_cuda_arch_flags = lambda: [_gencode]

    print(f"[bwd-cpp] JIT compiling {cu_src}...", flush=True)
    _bwd_cpp_module = load(
        name="flash_bwd_cutlass",
        sources=[cu_src],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-std=c++17",
            f"-I{cutlass_include}",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )
    print("[bwd-cpp] JIT compile done.", flush=True)
    return _bwd_cpp_module


def run_flash_bwd(
    q, k, v, out, dout, lse, dpsum, dqaccum, dk, dv,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
):
    """Run backward kernel via the CuTe DSL path.

    Delegates to flash_attn_dsl.flash_bwd.run_flash_bwd_dsl which now
    drives a fully cute-DSL backward (3 kernels: prep + main + post).
    The legacy C++ csrc/flash_bwd_cutlass.cu is no longer the
    benchmark target.  `dpsum` and `dqaccum` arguments are kept for
    interface compatibility but ignored — cute prep recomputes dpsum
    and zero-inits dq_accum internally.
    """
    import torch
    from .flash_bwd import run_flash_bwd_dsl

    if q.is_cuda:
        torch.cuda.set_device(q.device)

    B, H_q, N, D = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    dq = torch.empty_like(q)
    run_flash_bwd_dsl(
        q, k, v, out, dout, lse, dq, dk, dv,
        softmax_scale, bool(is_causal),
    )
    dqaccum.copy_(dq.float())


_BWD_PREWARM_TIMEOUT_S = int(os.environ.get("BWD_PREWARM_TIMEOUT", "120"))


def prewarm(device=None, is_causal_both=True):
    """Compile fwd+bwd kernels for GQA workload (matches tests: B=10, H_q=16, H_kv=2).

    FA3 forward uses native GQA (no K/V expansion needed).
    Safe to call once per process."""
    import threading
    import torch

    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    try:
        _fa_mod = importlib.import_module("python.flash_attention")
        FlashAttentionFunction = (
            _fa_mod.FlashAttentionFunction if getattr(_fa_mod, "HAS_DSL", False) else None
        )
    except ImportError:
        FlashAttentionFunction = None
    if device is None:
        device = torch.device("cuda")
    if device.type == "cuda":
        didx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(didx)
        _ = torch.zeros(1, device=device, dtype=torch.float16)
        torch.cuda.synchronize(device)
    B, H_q, H_kv, N, D = 10, 16, 2, 4096, 64
    scale = 1.0 / math.sqrt(D)
    q = torch.zeros(B, H_q, N, D, device=device, dtype=torch.float16)
    k = torch.zeros(B, H_kv, N, D, device=device, dtype=torch.float16)
    v = torch.zeros(B, H_kv, N, D, device=device, dtype=torch.float16)
    o = torch.zeros_like(q)
    lse = torch.zeros(B, H_q, N, device=device, dtype=torch.float32)
    for c in ((False, True) if is_causal_both else (False,)):
        # FA3 native GQA: pass K/V with H_kv heads directly
        run_flash_fwd(q, k, v, o, lse, scale, is_causal=c)
        if FlashAttentionFunction is not None:
            q_ = q.clone().detach().requires_grad_(True)
            k_ = k.clone().detach().requires_grad_(True)
            v_ = v.clone().detach().requires_grad_(True)
            fout = FlashAttentionFunction.apply(q_, k_, v_, scale, c)
            dout = torch.zeros_like(fout)

            bwd_err: list = []
            bwd_done = threading.Event()

            def _run_bwd(f=fout, d=dout):
                try:
                    f.backward(d)
                except Exception as e:
                    bwd_err.append(e)
                finally:
                    bwd_done.set()

            t = threading.Thread(target=_run_bwd, daemon=True)
            t.start()
            if not bwd_done.wait(timeout=_BWD_PREWARM_TIMEOUT_S):
                print(
                    f"[prewarm] BWD kernel timed out after {_BWD_PREWARM_TIMEOUT_S}s "
                    f"(likely barrier deadlock, causal={c}), skipping remaining prewarm"
                )
                break
            if bwd_err:
                raise bwd_err[0]
    torch.cuda.synchronize()
