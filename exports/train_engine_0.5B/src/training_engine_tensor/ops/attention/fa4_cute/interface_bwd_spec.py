# SM90-only FlashAttention backward: dense, fixed-tile (FA3 C++-aligned) path for the
# production shape: GQA 8:1, D=64, SplitKV/MLA/Blackwell removed from call chain.
# Tensors: FA4 / PyTorch (batch, seqlen, heads, headdim).

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor as fake_tensor

from .cache_utils import get_jit_cache
from .cute_dsl_utils import to_cute_tensor
from .flash_bwd_postprocess_spec import FlashAttentionBackwardPostprocessSpec
from .flash_bwd_preprocess_spec import FlashAttentionBackwardPreprocessSpec
from .flash_bwd_sm90_spec import FlashAttentionBackwardSm90Spec
from . import softmax_spec as _softmax_spec
from .testing import is_fake_mode
from . import utils

from training_engine_tensor.op_dispatcher import get_frozen_env as _get_frozen_env

# --- Fixed spec (B, N, Hq, D) with GQA; K/V (B, N, Hkv, D) -----------------
SPEC_B, SPEC_N, SPEC_HQ, SPEC_HKV, SPEC_D = 10, 4096, 16, 2, 64
SPEC_SOFTMAX_SCALE: float = _softmax_spec.SPEC_SOFTMAX_SCALE_LINEAR

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _parse_arch_str(arch_str: str) -> int:
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch() -> int:
    arch_override = _get_frozen_env().get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def maybe_contiguous(t):
    return t.contiguous() if t is not None and t.stride(-1) != 1 else t


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2
    dQ_single_wg: bool = False


def _tile_size_bwd_sm90_fixed() -> BwdConfig:
    """C++ FA3 / interface _tile_size_bwd_sm90 for head_dim <= 64 (causal)."""
    return BwdConfig(
        m_block_size=128,
        n_block_size=128,
        num_stages_Q=2,
        num_stages_dO=2,
        num_stages_PdS=2,
        SdP_swapAB=True,
        dKV_swapAB=False,
        dQ_swapAB=False,
        AtomLayoutMSdP=1,
        AtomLayoutNdKV=2,
        AtomLayoutMdQ=2,
    )


# --- Fake-tensor pre/post compile (from generic interface) -------------------
def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k):
    sym = cute.sym_int
    div = 128 // dtype.width
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    if not varlen_q:
        mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    else:
        mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (h_q, total_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum


def _compile_bwd_preprocess(dtype):
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, _, _ = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=False, varlen_k=False
    )
    fa_bwd_pre = FlashAttentionBackwardPreprocessSpec(dtype)
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, None, None, None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    cu_seqlens_q, seqused_q, dlse,
    dtype, head_dim, head_dim_v, m_block_size,
):
    _ = (head_dim, head_dim_v, m_block_size, cu_seqlens_q, seqused_q, dlse)
    compile_key = (dtype,)
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(dtype)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, None, None, None
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre_spec")


def _compile_bwd_postprocess(dtype):
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, _, _ = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=False, varlen_k=False
    )
    fa_bwd_post = FlashAttentionBackwardPostprocessSpec(dtype)
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), None, None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    cu_seqlens, seqused,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
):
    _ = (cu_seqlens, seqused, arch, hdim, block_size, num_threads, atom_layout, swap_ab, use_2cta_instrs, cluster_size)
    compile_key = (dtype,)
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(dtype)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](
            accum, output, scale, None, None,
        )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post_spec")


def _flash_attn_bwd_sm90_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = _get_device_arch()
    assert arch // 10 == 9, "fa4 bwd spec: SM90 (Hopper) only"

    block_sparse_tensors = None
    cu_seqlens_q = None
    cu_seqlens_k = None
    seqused_k = None
    score_mod = None
    score_mod_bwd = None
    mask_mod = None
    aux_tensors = None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod=None
    )

    cfg = _tile_size_bwd_sm90_fixed()
    m_block_size = cfg.m_block_size
    n_block_size = cfg.n_block_size
    num_stages_Q = cfg.num_stages_Q
    num_stages_dO = cfg.num_stages_dO
    num_stages_PdS = cfg.num_stages_PdS
    SdP_swapAB = cfg.SdP_swapAB
    dKV_swapAB = cfg.dKV_swapAB
    dQ_swapAB = cfg.dQ_swapAB
    AtomLayoutMSdP = cfg.AtomLayoutMSdP
    AtomLayoutNdKV = cfg.AtomLayoutNdKV
    AtomLayoutMdQ = cfg.AtomLayoutMdQ
    num_threads = (cfg.num_wg + 1) * 128
    dQ_single_wg = cfg.dQ_single_wg
    cluster_size = 1
    use_2cta_instrs = False
    V_in_regs = False
    pack_gqa = False

    q, k, v, out, dout, lse, _, _, seqused_q, _ = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]

    batch_size, seqlen_q = q.shape[:2]
    _, seqlen_k = k.shape[:2]
    num_head = q.shape[2]
    num_head_kv = k.shape[2]
    head_dim = q.shape[3]
    head_dim_v = v.shape[3]
    use_block_sparsity = False
    subtile_factor = 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size

    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
    assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
    assert lse.shape == (batch_size, num_head, seqlen_q)
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    if not is_fake_mode():
        assert q.is_cuda
    assert num_head % num_head_kv == 0
    assert softmax_scale is not None
    qhead_per_kvhead = num_head // num_head_kv

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    dq_accum = torch.empty(
        batch_size, num_head, seqlen_q_rounded * head_dim_rounded,
        dtype=torch.float32, device=q.device,
    )
    dpsum = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=q.device)
    lse_log2 = torch.empty(
        batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=q.device
    )

    dKV_postprocess = qhead_per_kvhead > 1
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        dk_accum = torch.zeros(
            batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded,
            dtype=torch.float32, device=q.device,
        )
        dv_accum = torch.zeros(
            batch_size, num_head_kv, seqlen_k_rounded * head_dim_v_rounded,
            dtype=torch.float32, device=q.device,
        )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    dQ_semaphore = dK_semaphore = dV_semaphore = None
    if deterministic:
        dQ_semaphore = torch.zeros(
            batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size,
            dtype=torch.int32, device=q.device,
        )
    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(
            batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=q.device,
        )
        dV_semaphore = torch.zeros(
            batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=q.device,
        )

    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        cu_seqlens_q, seqused_q, dlse,
        dtype, head_dim, head_dim_v, m_block_size,
    )

    cute_aux_tensors = None
    compile_key = (dtype, deterministic)

    if compile_key not in _flash_attn_bwd_sm90_dense.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [to_cute_tensor(t) for t in (dk_accum, dv_accum)]
        cu_seqlens_q_tensor = cu_seqlens_k_tensor = seqused_q_tensor = seqused_k_tensor = None
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(
                t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order()
            ) if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        fa_bwd_obj = FlashAttentionBackwardSm90Spec(
            dtype,
            deterministic=deterministic,
        )
        _flash_attn_bwd_sm90_dense.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj, q_tensor, k_tensor, v_tensor, do_tensor, lse_log2_tensor, dpsum_tensor, dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor,
            window_size_left, window_size_right, dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor,
            cute_aux_tensors, None, current_stream, options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        _flash_attn_bwd_sm90_dense.compile_cache[compile_key](
            q.detach(), k.detach(), v.detach(), dout, lse_log2, dpsum, dq_accum,
            dk if not dKV_postprocess else dk_accum, dv if not dKV_postprocess else dv_accum,
            softmax_scale, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
            window_size_left, window_size_right, dQ_semaphore, dK_semaphore, dV_semaphore,
            aux_tensors, None,
        )

    num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
    num_threads_post_dKV = cfg.num_wg * 128

    _bwd_postprocess_convert(
        dq_accum, dq, softmax_scale, cu_seqlens_q, seqused_q, arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
        AtomLayoutMdQ, dQ_swapAB, use_2cta_instrs=use_2cta_instrs, cluster_size=1,
    )
    if dKV_postprocess:
        _bwd_postprocess_convert(
            dk_accum, dk, softmax_scale, cu_seqlens_k, seqused_k, arch, dtype, head_dim, n_block_size, num_threads_post_dKV,
            AtomLayoutNdKV, dKV_swapAB, cluster_size=cluster_size,
        )
        _bwd_postprocess_convert(
            dv_accum, dv, 1.0, cu_seqlens_k, seqused_k, arch, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
            AtomLayoutNdKV, dKV_swapAB, cluster_size=cluster_size,
        )

    return dq, dk, dv


_flash_attn_bwd_sm90_dense.compile_cache = get_jit_cache("bwd_sm90_spec")


def fa4_spec_shape_ok(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """True if (B, N, H, D) matches the specialized GQA 8:1 + fixed N/B/D product shape."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    b, n, hq, d = q.shape
    _, nk, hkv, dk = k.shape
    if (b, n, d) != (SPEC_B, SPEC_N, SPEC_D) or nk != SPEC_N or dk != SPEC_D:
        return False
    if (hq, hkv) != (SPEC_HQ, SPEC_HKV):
        return False
    return v.shape == (b, n, hkv, d)


def run_flash_bwd_fa4_spec(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float = SPEC_SOFTMAX_SCALE,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FA4 layout (B, N, H, D): SM90 dense backward, fixed tile 128/128, no sparse/varlen/FP8."""
    assert fa4_spec_shape_ok(q, k, v), f"expected spec shape {(SPEC_B, SPEC_N, SPEC_HQ, SPEC_D)} + K/V H={SPEC_HKV}, got {q.shape}, {k.shape}, {v.shape}"
    return _flash_attn_bwd_sm90_dense(
        q, k, v, out, dout, lse, softmax_scale, is_causal, deterministic=False,
    )
