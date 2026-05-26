"""Python host-side wrapper that drives the SM90a flash-attention forward DSL kernel.

Responsibilities:
- Bridge PyTorch tensors to CuTe tensors via ``from_dlpack``.
- JIT-compile the kernel once per (dtype, is_causal, n_block, ...) key and cache it.
- Resolve the ``softmax_scale`` default and forward the current CUDA stream.

Exported names: :func:`run_flash_fwd`, :func:`prewarm`.

Native GQA: K/V are accepted at their native ``[B, H_kv, N, D]`` shape and the
kernel is responsible for sharing each K/V tile across ``H_q // H_kv`` query
heads in the same group.  The host MUST NOT broadcast (e.g. via
``repeat_interleave``) K/V before invoking the kernel.
"""

import math
from typing import Optional

try:
    import cuda.bindings.driver as cuda
except ModuleNotFoundError:
    import cuda.cuda as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .flash_fwd import FlashAttnFwdSm90


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


def _compile_key(dtype, head_dim, is_causal, num_stages, n_block, qhead_per_kvhead=1, has_lse=True):
    return (dtype, head_dim, is_causal, num_stages, n_block, qhead_per_kvhead, has_lse)


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
    num_stages: int = 3,
    n_block: int = 128,
):
    """Run forward. q/out are [B, H_q, N, D], k/v are [B, H_kv, N, D], lse is [B, H_q, N].

    Supports native GQA: H_q can be a multiple of H_kv (e.g. H_q=16, H_kv=1).
    Default num_stages=3 / n_block=128 are placeholder safe choices for D=128
    BF16 — the kernel author is expected to override via its own __init__.
    """
    import torch

    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    assert k.shape == (B, H_kv, N, D) and v.shape == (B, H_kv, N, D)
    assert out.shape == q.shape
    assert H_q % H_kv == 0, f"H_q={H_q} must be a multiple of H_kv={H_kv}"
    qhead_per_kvhead = H_q // H_kv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    cute_dtype = _dtype_map()[q.dtype]
    has_lse = lse is not None
    key = _compile_key(cute_dtype, D, bool(is_causal), int(num_stages), int(n_block), qhead_per_kvhead, has_lse)

    qT = _to_cute_tensor(q)
    kT = _to_cute_tensor(k)
    vT = _to_cute_tensor(v)
    oT = _to_cute_tensor(out)
    lT = _to_cute_lse(lse) if lse is not None else None

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if key not in _compile_cache:
        fa = FlashAttnFwdSm90(
            dtype=cute_dtype,
            head_dim=D,
            m_block_size=64,
            n_block_size=n_block,
            num_stages=num_stages,
            is_causal=bool(is_causal),
            qhead_per_kvhead=qhead_per_kvhead,
        )
        _compile_cache[key] = cute.compile(
            fa, qT, kT, vT, oT, lT, cutlass.Float32(softmax_scale), stream
        )
    _compile_cache[key](qT, kT, vT, oT, lT, cutlass.Float32(softmax_scale), stream)
    return out


def prewarm(device=None, is_causal_both=False):
    """Compile the kernel for the canonical training shape to amortise JIT
    cost outside the hot training loop.  Safe to call once per process.

    Canonical shape: ``B=2, H_q=16, H_kv=1, S=4096, D=128, BF16``.
    Training is causal-only, so by default this only warms the causal
    code path.  Pass ``is_causal_both=True`` to also warm the
    non-causal path when the kernel emits different code for it.
    """
    import torch
    if device is None:
        device = torch.device("cuda")
    B, H_q, H_kv, S, D = 2, 16, 1, 4096, 128
    q = torch.zeros(B, H_q, S, D, device=device, dtype=torch.bfloat16)
    k = torch.zeros(B, H_kv, S, D, device=device, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    o = torch.zeros_like(q)
    lse = torch.zeros(B, H_q, S, device=device, dtype=torch.float32)
    modes = (False, True) if is_causal_both else (True,)
    for c in modes:
        try:
            run_flash_fwd(q, k, v, o, lse, 1.0 / math.sqrt(D), is_causal=c)
        except NotImplementedError:
            # Kernel is still a stub: prewarm is best-effort, skip silently.
            return
    torch.cuda.synchronize()
