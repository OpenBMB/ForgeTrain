"""Low-level compute kernels: RMSNorm, RoPE, attention, activations.

  - RMSNorm fwd/bwd are implemented as standalone Triton kernels with
    FP32 accumulation, avoiding any dependency on
    TransformerEngine's private ``cpp_extensions`` API (whose signature
    is not stable across TE major versions).
  - Causal attention dispatches through PyTorch's native scaled-dot-
    product attention, which picks Flash-Attention-2 / cuDNN on H100.
    The optional :mod:`flash_attn_dsl` forward kernel is selected when
    ``EngineConfig.dsl_attn_fwd`` is enabled (backward stays on the
    cuDNN ATen op for a numerically stable round-trip).

The fused Triton paths in this file are numerically equivalent to the
naive PyTorch implementation up to BF16 rounding noise.
"""
from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "rmsnorm_forward", "precompute_rope_freqs", "apply_rope",
    "fused_rope_backward", "fused_rope_backward_pack", "fused_rope_from_qkv",
    "fused_rope_from_qkv_bhsd", "fused_rope_backward_pack_bhsd",
    "causal_attention",
    "causal_attention_fwd_direct", "causal_attention_bwd_direct",
    "causal_attention_fwd_dsl",
    "swiglu", "swiglu_backward",
    "fused_residual_add_rmsnorm_fwd",
    "fused_residual_add_rmsnorm_bwd_add",
    "fused_residual_add_rmsnorm_bwd_add_dw_reduce",
    "fused_rmsnorm_bwd_add", "fused_rmsnorm_bwd_add_dw_reduce",
    "fused_ce_max_sum_pred", "fused_ce_correction", "fused_ce_bwd",
]

import torch
import torch.nn.functional as F

from . import config
from .engine_config import get_config

# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------
# Triton drives the standalone CUDA RMSNorm fwd/bwd path.
_triton_available = False
try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    pass



# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
def rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm forward.  Returns (output, rsigma).

    Uses TE fused kernel if available; manual FP32 accumulation otherwise.
    TE requires 2D input [tokens, hidden], so we flatten/unflatten as needed.
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    if _triton_available and x.is_cuda:
        out, rsigma = _te_rmsnorm_forward(x, weight, eps)
    else:
        out, rsigma = _manual_rmsnorm(x, weight, eps)
    if out.shape != orig_shape:
        out = out.view(orig_shape)
    return out, rsigma


def _te_rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone RMSNorm forward (Triton on CUDA, PyTorch on CPU).

    FP32 variance + FP32 weight multiply, BF16 output; returns (out, rsigma).
    """
    if _triton_available and x.is_cuda:
        return _triton_rmsnorm_forward(x, weight, eps)
    return _manual_rmsnorm(x, weight, eps)


def _manual_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manual RMSNorm with FP32 variance accumulation (for local testing)."""
    orig_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    rsigma = torch.rsqrt(variance + eps)
    out = (x_fp32 * rsigma * weight.float()).to(orig_dtype)
    return out, rsigma.squeeze(-1)


def _manual_rmsnorm_backward(
    d_out: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manual RMSNorm backward (CPU / non-Triton fallback).

    Math identical to TE's `rmsnorm_bwd`:
        c   = sum(d_out * weight * x)        (over last dim)
        dx  = rsigma * (d_out * weight - (rsigma^2 / H) * x * c)
        dw  = sum(d_out * x * rsigma)        (over rows, FP32)
    """
    H = x.shape[-1]
    d_out_fp32 = d_out.reshape(-1, H).float()
    x_fp32 = x.reshape(-1, H).float()
    if rsigma.dim() == 1:
        rs = rsigma.float().unsqueeze(-1)  # [T, 1]
    else:
        rs = rsigma.float().reshape(-1, 1)
    w_fp32 = weight.float().unsqueeze(0)  # [1, H]
    c = (d_out_fp32 * w_fp32 * x_fp32).sum(-1, keepdim=True)
    dx = rs * (d_out_fp32 * w_fp32 - (rs * rs / H) * x_fp32 * c)
    dw = (d_out_fp32 * x_fp32 * rs).sum(0)
    return dx.to(d_out.dtype).view_as(d_out), dw


# ---------------------------------------------------------------------------
# Standalone Triton RMSNorm kernels (replaces TE cpp_extensions on CUDA)
#
# The fwd/bwd math here mirrors the existing `_residual_add_rmsnorm_*`
# kernels (lines below) — only the residual-add step is dropped.  This gives
# numerics identical (up to BF16 rounding) to TE's legacy RMSNorm, with no
# dependency on TE private signatures.
# ---------------------------------------------------------------------------
if _triton_available:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr, weight_ptr, out_ptr, rsigma_ptr,
        N,
        H: tl.constexpr,
        eps: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return
        offs = tl.arange(0, H)
        row_base = pid * H
        x = tl.load(x_ptr + row_base + offs).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / H
        rs = 1.0 / tl.sqrt(variance + eps)
        w = tl.load(weight_ptr + offs).to(tl.float32)
        y = x * rs * w
        tl.store(out_ptr + row_base + offs, y.to(out_ptr.dtype.element_ty))
        tl.store(rsigma_ptr + pid, rs)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        d_out_ptr, x_ptr, rsigma_ptr, weight_ptr,
        dx_ptr, dw_partial_ptr,
        N,
        H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return
        offs = tl.arange(0, H)
        row_base = pid * H
        d_out = tl.load(d_out_ptr + row_base + offs).to(tl.float32)
        x = tl.load(x_ptr + row_base + offs).to(tl.float32)
        rs = tl.load(rsigma_ptr + pid).to(tl.float32)
        w = tl.load(weight_ptr + offs).to(tl.float32)
        c = tl.sum(d_out * w * x, axis=0)
        dx = rs * (d_out * w - (rs * rs / H) * x * c)
        tl.store(dx_ptr + row_base + offs, dx.to(d_out_ptr.dtype.element_ty))
        dw_row = d_out * x * rs
        tl.store(dw_partial_ptr + row_base + offs, dw_row)


def _triton_rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone Triton RMSNorm forward, returning (out_bf16, rsigma_fp32[T])."""
    H = x.shape[-1]
    x_2d = x.reshape(-1, H).contiguous()
    T = x_2d.shape[0]
    out_2d = torch.empty(T, H, dtype=x.dtype, device=x.device)
    rsigma = torch.empty(T, dtype=torch.float32, device=x.device)
    _rmsnorm_fwd_kernel[(T,)](
        x_2d, weight, out_2d, rsigma,
        T, H=H, eps=eps,
    )
    return out_2d.view_as(x), rsigma


def _triton_rmsnorm_backward(
    d_out: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone Triton RMSNorm backward, returning (dx [same shape as d_out], dw [H] FP32)."""
    H = d_out.shape[-1]
    d_out_2d = d_out.reshape(-1, H).contiguous()
    x_2d = x.reshape(-1, H).contiguous()
    T = d_out_2d.shape[0]
    dx_2d = torch.empty(T, H, dtype=d_out.dtype, device=d_out.device)
    dw_partial = torch.empty(T, H, dtype=torch.float32, device=d_out.device)
    _rmsnorm_bwd_kernel[(T,)](
        d_out_2d, x_2d, rsigma, weight,
        dx_2d, dw_partial,
        T, H=H,
    )
    dw = dw_partial.sum(dim=0)
    return dx_2d.view_as(d_out), dw


# ---------------------------------------------------------------------------
# RMSNorm — autograd-compatible wrappers for backward pass
# ---------------------------------------------------------------------------
def _te_rmsnorm_backward(
    d_out: torch.Tensor,
    x: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone RMSNorm backward (Triton on CUDA, PyTorch on CPU).

    Returns (dx, dw_fp32) with FP32 weight gradient accumulation.
    """
    if _triton_available and d_out.is_cuda:
        return _triton_rmsnorm_backward(d_out, x, rsigma, weight)
    return _manual_rmsnorm_backward(d_out, x, rsigma, weight)


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------
def precompute_rope_freqs(
    seq_len: int,
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables in FP32.

    Returns (cos, sin) each of shape [seq_len, head_dim].
    Matches baseline r0.8.0 RotaryEmbedding: freq = 1/(base^(2i/d)),
    emb = cat(freqs, freqs).
    """
    dim = config.HEAD_DIM
    base = float(config.ROTARY_BASE)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [seq, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
    return emb.cos(), emb.sin()


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    q: [S, B, Hq, D]   k: [S, B, Hkv, D]
    cos, sin: [S, D] (FP32)
    """
    if get_config().fused_rope and q.is_cuda:
        return _fused_rope_apply(q, k, cos, sin, is_forward=True)
    seq_len = q.shape[0]
    cos_bf16 = cos[:seq_len, None, None, :].to(q.dtype)
    sin_bf16 = sin[:seq_len, None, None, :].to(q.dtype)
    q_rot = q * cos_bf16 + _rotate_half(q) * sin_bf16
    k_rot = k * cos_bf16 + _rotate_half(k) * sin_bf16
    return q_rot, k_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: [-x2, x1] where x1 = first half, x2 = second half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# ---------------------------------------------------------------------------
# Fused RoPE via Triton
# ---------------------------------------------------------------------------
# (`_triton_available` and the `triton` / `tl` modules are imported at the
# top of this file so the standalone RMSNorm kernels below can reference
# them.)

if _triton_available:
    @triton.jit
    def _rope_triton_kernel(
        x_ptr, cos_ptr, sin_ptr, out_ptr,
        BH,
        N,
        D: tl.constexpr, HALF_D: tl.constexpr,
        IS_FORWARD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        offs_first = tl.arange(0, HALF_D)
        offs_second = offs_first + HALF_D

        x_base = pid * D
        x_first = tl.load(x_ptr + x_base + offs_first).to(tl.float32)
        x_second = tl.load(x_ptr + x_base + offs_second).to(tl.float32)

        # Broadcast: cos/sin are [S, D], index by s = pid // BH
        s_idx = pid // BH
        cos_base = s_idx * D
        cos_first = tl.load(cos_ptr + cos_base + offs_first).to(tl.float32)
        sin_first = tl.load(sin_ptr + cos_base + offs_first).to(tl.float32)
        cos_second = tl.load(cos_ptr + cos_base + offs_second).to(tl.float32)
        sin_second = tl.load(sin_ptr + cos_base + offs_second).to(tl.float32)

        if IS_FORWARD:
            out_first = x_first * cos_first - x_second * sin_first
            out_second = x_second * cos_second + x_first * sin_second
        else:
            out_first = x_first * cos_first + x_second * sin_first
            out_second = x_second * cos_second - x_first * sin_second

        tl.store(out_ptr + x_base + offs_first, out_first.to(x_ptr.dtype.element_ty))
        tl.store(out_ptr + x_base + offs_second, out_second.to(x_ptr.dtype.element_ty))


def _fused_rope_single(
    x: torch.Tensor,
    cos_bf16: torch.Tensor,
    sin_bf16: torch.Tensor,
    BH: int,
    is_forward: bool,
) -> torch.Tensor:
    """Apply fused RoPE to a single tensor (q or k).

    x: [S, B, H, D] BF16 (may be non-contiguous)
    cos_bf16: [S, D] BF16 (compact, broadcast inside kernel)
    sin_bf16: [S, D] BF16
    BH: B * H (for computing s = pid // BH)
    """
    x_in = x.contiguous()
    S, B, H, D = x_in.shape
    HALF_D = D // 2
    N = S * B * H

    out = torch.empty(N, D, dtype=x.dtype, device=x.device)

    if _triton_available and x.is_cuda:
        grid = (N,)
        _rope_triton_kernel[grid](
            x_in.view(N, D), cos_bf16, sin_bf16, out,
            BH,
            N,
            D=D, HALF_D=HALF_D,
            IS_FORWARD=is_forward,
        )
    else:
        raise RuntimeError("Fused RoPE requires Triton + CUDA")

    return out.view(S, B, H, D)


def _fused_rope_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_forward: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE with in-kernel cos/sin broadcast.

    cos, sin: [S, D] FP32 tables from precompute_rope_freqs.
    The kernel indexes cos/sin by s = pid // (B*H), avoiding
    the expensive .expand().contiguous() that materializes [S*B*H, D].
    """
    S = q.shape[0]
    B = q.shape[1]
    Hq = q.shape[2]
    Hkv = k.shape[2]
    D = q.shape[3]

    cos_s = cos[:S]  # [S, D] FP32
    sin_s = sin[:S]  # [S, D] FP32

    if get_config().fused_rope_fp32cos:
        cos_data = cos_s
        sin_data = sin_s
    else:
        cos_data = cos_s.to(q.dtype)
        sin_data = sin_s.to(q.dtype)

    q_rot = _fused_rope_single(q, cos_data, sin_data, B * Hq, is_forward)
    k_rot = _fused_rope_single(k, cos_data, sin_data, B * Hkv, is_forward)
    return q_rot, k_rot


def fused_rope_backward(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE backward: d_x_pre = d_x * cos ∓ rotate_half(d_x) * sin."""
    return _fused_rope_apply(d_q, d_k, cos, sin, is_forward=False)


# ---------------------------------------------------------------------------
# Fused RoPE backward + QKV gradient packing
# ---------------------------------------------------------------------------
if _triton_available:
    @triton.jit
    def _rope_bwd_pack_kernel(
        x_ptr, cos_ptr, sin_ptr, out_ptr,
        BH,
        N,
        H_LOCAL,
        OUT_ROW_STRIDE,
        COL_OFFSET,
        D: tl.constexpr, HALF_D: tl.constexpr,
    ):
        """RoPE backward writing directly into packed QKV grad buffer.

        For each program (one row of the input tensor), computes:
          out_first  = x_first * cos_first + x_second * sin_first
          out_second = x_second * cos_second - x_first * sin_second
        and writes to the packed output at:
          row = pid // H_LOCAL
          col = (pid % H_LOCAL) * D + COL_OFFSET
        """
        pid = tl.program_id(0)
        if pid >= N:
            return

        offs_first = tl.arange(0, HALF_D)
        offs_second = offs_first + HALF_D

        x_base = pid * D
        x_first = tl.load(x_ptr + x_base + offs_first).to(tl.float32)
        x_second = tl.load(x_ptr + x_base + offs_second).to(tl.float32)

        s_idx = pid // BH
        cos_base = s_idx * D
        cos_first = tl.load(cos_ptr + cos_base + offs_first).to(tl.float32)
        sin_first = tl.load(sin_ptr + cos_base + offs_first).to(tl.float32)
        cos_second = tl.load(cos_ptr + cos_base + offs_second).to(tl.float32)
        sin_second = tl.load(sin_ptr + cos_base + offs_second).to(tl.float32)

        out_first = x_first * cos_first + x_second * sin_first
        out_second = x_second * cos_second - x_first * sin_second

        row = pid // H_LOCAL
        col = (pid % H_LOCAL) * D + COL_OFFSET
        out_base = row * OUT_ROW_STRIDE + col

        tl.store(out_ptr + out_base + offs_first, out_first.to(x_ptr.dtype.element_ty))
        tl.store(out_ptr + out_base + offs_second, out_second.to(x_ptr.dtype.element_ty))


def fused_rope_backward_pack(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    """Fused RoPE backward + QKV gradient packing.

    Instead of:
      d_q_pre, d_k_pre = rope_backward(d_q, d_k, cos, sin)
      d_qkv = cat([d_q_pre.reshape(...), d_k_pre, d_v], dim=-1).reshape(...)

    This writes RoPE backward output directly into a packed buffer,
    eliminating the cat and intermediate allocations.

    d_q: [S, B, Hq, D]   d_k: [S, B, Hkv, D]   d_v: [S, B, Hkv, D]
    cos, sin: [S, D] FP32 tables
    Returns: d_qkv [S, B, qkv_dim] BF16
    """
    S, B, Hq, D = d_q.shape
    Hkv = d_k.shape[2]
    assert Hkv == num_kv_heads, \
        f"Kernel indexing requires Hkv({Hkv})==num_kv_heads({num_kv_heads})"
    q_per_group = heads_per_group * head_dim
    group_size = q_per_group + 2 * head_dim
    qkv_dim = num_kv_heads * group_size

    if get_config().fused_rope_fp32cos:
        cos_s = cos[:S]
        sin_s = sin[:S]
    else:
        cos_s = cos[:S].to(d_q.dtype)
        sin_s = sin[:S].to(d_q.dtype)

    SB = S * B
    d_qkv_flat = torch.empty(SB * Hkv, group_size, dtype=d_q.dtype, device=d_q.device)

    d_q_in = d_q.contiguous()
    Nq = S * B * Hq
    _rope_bwd_pack_kernel[(Nq,)](
        d_q_in.view(Nq, D), cos_s, sin_s, d_qkv_flat,
        B * Hq,
        Nq,
        Hq,
        group_size,
        0,
        D=D, HALF_D=D // 2,
    )

    d_k_in = d_k.contiguous()
    Nk = S * B * Hkv
    _rope_bwd_pack_kernel[(Nk,)](
        d_k_in.view(Nk, D), cos_s, sin_s, d_qkv_flat,
        B * Hkv,
        Nk,
        Hkv,
        group_size,
        q_per_group,
        D=D, HALF_D=D // 2,
    )

    d_v_in = d_v.contiguous()
    d_qkv_flat[:, q_per_group + head_dim:].copy_(d_v_in.reshape(Nk, D))

    return d_qkv_flat.view(S, B, qkv_dim)


# ---------------------------------------------------------------------------
# Fused RoPE backward + QKV pack from [B, H, S, D] inputs
# ---------------------------------------------------------------------------
# Consuming dq/dk in BHSD contig layout (no permute().contiguous() at the attention bwd exit), apply
# RoPE-inverse, and write directly into the packed [S, B, qkv_dim] grad
# buffer.  d_v is handled by `_dv_pack_bhsd_kernel` below — same idea but
# without the RoPE math.
if _triton_available:
    @triton.jit
    def _rope_bwd_pack_kernel_bhsd(
        x_ptr,
        cos_ptr, sin_ptr,
        out_ptr,
        H, S, B,
        OUT_ROW_STRIDE,
        COL_OFFSET,
        D: tl.constexpr, HALF_D: tl.constexpr,
        TILE_S: tl.constexpr,
    ):
        """RoPE bwd: [B, H, S, D] contig input → packed [S, B, qkv_dim] grad.

        Grid: (BH * (S // TILE_S),) — same pid scheme as the BHSD fwd
        kernel.  Input reads are sequential within (b, h) (BHSD contig);
        output writes are strided in the packed buffer (row = s*B + b,
        col = h*D + COL_OFFSET) but each program writes TILE_S × D as a
        2-D strided store, which Triton lowers reasonably well since the
        d-axis is contiguous within each output row.
        """
        pid = tl.program_id(0)
        s_blocks = S // TILE_S
        bh = pid // s_blocks
        s_blk = pid % s_blocks

        b_idx = bh // H
        h_idx = bh % H

        offs_s = s_blk * TILE_S + tl.arange(0, TILE_S)   # [TILE_S]
        offs_first = tl.arange(0, HALF_D)                # [HALF_D]
        offs_second = offs_first + HALF_D                # [HALF_D]

        # Input read: BHSD contig at (b, h, s, d) → bh*S*D + s*D + d
        in_bh_base = bh * S * D
        x_first = tl.load(
            x_ptr + in_bh_base + offs_s[:, None] * D + offs_first[None, :]
        ).to(tl.float32)
        x_second = tl.load(
            x_ptr + in_bh_base + offs_s[:, None] * D + offs_second[None, :]
        ).to(tl.float32)

        cos_first = tl.load(cos_ptr + offs_s[:, None] * D + offs_first[None, :]).to(tl.float32)
        sin_first = tl.load(sin_ptr + offs_s[:, None] * D + offs_first[None, :]).to(tl.float32)
        cos_second = tl.load(cos_ptr + offs_s[:, None] * D + offs_second[None, :]).to(tl.float32)
        sin_second = tl.load(sin_ptr + offs_s[:, None] * D + offs_second[None, :]).to(tl.float32)

        # RoPE inverse (matches fused_rope_backward_pack math)
        out_first = x_first * cos_first + x_second * sin_first
        out_second = x_second * cos_second - x_first * sin_second

        # Packed output: row = s * B + b, col = h * D + COL_OFFSET
        rows = offs_s * B + b_idx                        # [TILE_S]
        col_base = h_idx * D + COL_OFFSET
        tl.store(
            out_ptr + rows[:, None] * OUT_ROW_STRIDE + col_base + offs_first[None, :],
            out_first.to(x_ptr.dtype.element_ty),
        )
        tl.store(
            out_ptr + rows[:, None] * OUT_ROW_STRIDE + col_base + offs_second[None, :],
            out_second.to(x_ptr.dtype.element_ty),
        )


if _triton_available:
    @triton.jit
    def _dv_pack_bhsd_kernel(
        v_ptr, out_ptr,
        H, S, B,
        OUT_ROW_STRIDE,
        COL_OFFSET,
        D: tl.constexpr,
        TILE_S: tl.constexpr,
    ):
        """Pure copy: d_v [B, Hkv, S, D] contig → packed [S, B, qkv_dim] slot.

        Same tile layout as the RoPE bwd BHSD kernel but skips the RoPE math.
        ``out_ptr`` is the packed [S*B, group_size] flat buffer; ``COL_OFFSET``
        points to the v slot (q_per_group + head_dim).
        """
        pid = tl.program_id(0)
        s_blocks = S // TILE_S
        bh = pid // s_blocks
        s_blk = pid % s_blocks

        b_idx = bh // H
        h_idx = bh % H

        offs_s = s_blk * TILE_S + tl.arange(0, TILE_S)   # [TILE_S]
        offs_d = tl.arange(0, D)                         # [D]

        in_bh_base = bh * S * D
        x = tl.load(v_ptr + in_bh_base + offs_s[:, None] * D + offs_d[None, :])

        rows = offs_s * B + b_idx
        col_base = h_idx * D + COL_OFFSET
        tl.store(
            out_ptr + rows[:, None] * OUT_ROW_STRIDE + col_base + offs_d[None, :],
            x,
        )


def fused_rope_backward_pack_bhsd(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_kv_heads: int,
    heads_per_group: int,
    head_dim: int,
    tile_s: int = 64,
) -> torch.Tensor:
    """BHSD-input variant of :func:`fused_rope_backward_pack`.

    d_q: [B, Hq,  S, D] contig BF16
    d_k: [B, Hkv, S, D] contig BF16
    d_v: [B, Hkv, S, D] contig BF16
    cos, sin: [S, D] FP32 tables
    Returns: d_qkv [S, B, qkv_dim] BF16 (same layout as fused_rope_backward_pack)

    Skipping the upstream ``dq.permute(2,0,1,3).contiguous()`` chain (3 calls,
    ~36 MB) is the whole point of this path; here we just stream the BHSD
    inputs through Triton tile-by-tile and scatter into the packed grad
    buffer.  Caller is responsible for ensuring d_q/d_k/d_v are BHSD contig.
    """
    B, Hq, S, D = d_q.shape
    Hkv = d_k.shape[1]
    assert d_k.shape == (B, Hkv, S, D), f"d_k shape {d_k.shape} != (B,Hkv,S,D)"
    assert d_v.shape == (B, Hkv, S, D), f"d_v shape {d_v.shape} != (B,Hkv,S,D)"
    assert d_q.is_contiguous(), "d_q must be BHSD contig"
    assert d_k.is_contiguous(), "d_k must be BHSD contig"
    assert d_v.is_contiguous(), "d_v must be BHSD contig"
    assert Hkv == num_kv_heads, \
        f"Kernel indexing requires Hkv({Hkv})==num_kv_heads({num_kv_heads})"
    assert Hq == heads_per_group * Hkv, \
        f"Hq({Hq}) must equal heads_per_group({heads_per_group})*Hkv({Hkv})"
    assert S % tile_s == 0, f"S({S}) must be divisible by tile_s({tile_s})"

    q_per_group = heads_per_group * head_dim
    group_size = q_per_group + 2 * head_dim
    qkv_dim = num_kv_heads * group_size

    if get_config().fused_rope_fp32cos:
        cos_s = cos[:S]
        sin_s = sin[:S]
    else:
        cos_s = cos[:S].to(d_q.dtype)
        sin_s = sin[:S].to(d_q.dtype)

    SB = S * B
    d_qkv_flat = torch.empty(SB * Hkv, group_size, dtype=d_q.dtype, device=d_q.device)

    s_blocks = S // tile_s

    _rope_bwd_pack_kernel_bhsd[(B * Hq * s_blocks,)](
        d_q, cos_s, sin_s, d_qkv_flat,
        Hq, S, B,
        group_size,
        0,
        D=D, HALF_D=D // 2,
        TILE_S=tile_s,
    )

    _rope_bwd_pack_kernel_bhsd[(B * Hkv * s_blocks,)](
        d_k, cos_s, sin_s, d_qkv_flat,
        Hkv, S, B,
        group_size,
        q_per_group,
        D=D, HALF_D=D // 2,
        TILE_S=tile_s,
    )

    _dv_pack_bhsd_kernel[(B * Hkv * s_blocks,)](
        d_v, d_qkv_flat,
        Hkv, S, B,
        group_size,
        q_per_group + head_dim,
        D=D,
        TILE_S=tile_s,
    )

    return d_qkv_flat.view(S, B, qkv_dim)


# ---------------------------------------------------------------------------
# Fused RoPE from interleaved QKV buffer (stride-aware forward)
# ---------------------------------------------------------------------------
if _triton_available:
    @triton.jit
    def _rope_qkv_fwd_kernel(
        qkv_ptr,
        cos_ptr, sin_ptr,
        out_ptr,
        QKV_STRIDE_S,
        QKV_STRIDE_B,
        COL_OFFSET,
        HEAD_STRIDE,
        N,
        H,
        BH,
        D: tl.constexpr, HALF_D: tl.constexpr,
    ):
        """RoPE forward reading directly from interleaved QKV buffer.

        Each program handles one (s, b, h) head. Reads from qkv buffer at
        qkv_ptr + s*QKV_STRIDE_S + b*QKV_STRIDE_B + COL_OFFSET + h*HEAD_STRIDE,
        writes to contiguous output at out_ptr + pid*D.
        """
        pid = tl.program_id(0)
        if pid >= N:
            return

        s_idx = pid // BH
        bh = pid % BH
        b_idx = bh // H
        h_idx = bh % H

        offs_first = tl.arange(0, HALF_D)
        offs_second = offs_first + HALF_D

        qkv_base = s_idx * QKV_STRIDE_S + b_idx * QKV_STRIDE_B + COL_OFFSET + h_idx * HEAD_STRIDE
        x_first = tl.load(qkv_ptr + qkv_base + offs_first).to(tl.float32)
        x_second = tl.load(qkv_ptr + qkv_base + offs_second).to(tl.float32)

        cos_base = s_idx * D
        cos_first = tl.load(cos_ptr + cos_base + offs_first).to(tl.float32)
        sin_first = tl.load(sin_ptr + cos_base + offs_first).to(tl.float32)
        cos_second = tl.load(cos_ptr + cos_base + offs_second).to(tl.float32)
        sin_second = tl.load(sin_ptr + cos_base + offs_second).to(tl.float32)

        out_first = x_first * cos_first - x_second * sin_first
        out_second = x_second * cos_second + x_first * sin_second

        out_base = pid * D
        tl.store(out_ptr + out_base + offs_first, out_first.to(qkv_ptr.dtype.element_ty))
        tl.store(out_ptr + out_base + offs_second, out_second.to(qkv_ptr.dtype.element_ty))


def fused_rope_from_qkv(
    qkv_grouped: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE reading q/k directly from the interleaved QKV buffer.

    Eliminates the q.reshape (contiguous copy) and k.contiguous() calls by
    having the Triton kernel read from the QKV buffer with explicit strides.

    qkv_grouped: [S, B, num_kv_heads, group_size] BF16 (view of QKV GEMM output)
    cos, sin: [S, D] FP32 tables
    Returns: (q_rot [S, B, Hq, D], k_rot [S, B, Hkv, D]) both contiguous BF16
    """
    S, B, Hkv, gs = qkv_grouped.shape
    Hq = num_heads
    D = head_dim
    q_per_group = (Hq // Hkv) * D

    if get_config().fused_rope_fp32cos:
        cos_data = cos[:S]
        sin_data = sin[:S]
    else:
        cos_data = cos[:S].to(qkv_grouped.dtype)
        sin_data = sin[:S].to(qkv_grouped.dtype)

    qkv_stride_s = qkv_grouped.stride(0)
    qkv_stride_b = qkv_grouped.stride(1)

    N_q = S * B * Hq
    q_out = torch.empty(N_q, D, dtype=qkv_grouped.dtype, device=qkv_grouped.device)
    _rope_qkv_fwd_kernel[(N_q,)](
        qkv_grouped, cos_data, sin_data, q_out,
        qkv_stride_s, qkv_stride_b,
        0, D,
        N_q, Hq, B * Hq,
        D=D, HALF_D=D // 2,
    )

    N_k = S * B * Hkv
    k_out = torch.empty(N_k, D, dtype=qkv_grouped.dtype, device=qkv_grouped.device)
    _rope_qkv_fwd_kernel[(N_k,)](
        qkv_grouped, cos_data, sin_data, k_out,
        qkv_stride_s, qkv_stride_b,
        q_per_group, D,
        N_k, Hkv, B * Hkv,
        D=D, HALF_D=D // 2,
    )

    return q_out.view(S, B, Hq, D), k_out.view(S, B, Hkv, D)


# ---------------------------------------------------------------------------
# Fused RoPE from QKV → [B, H, S, D] output
# ---------------------------------------------------------------------------
# Tiled kernel: each program handles (1 b, 1 h, TILE_S consecutive s) so
# writes to BHSD output are fully contiguous within a program (TILE_S × D
# sequential), and reads from the [S, B, num_kv_heads, group_size] QKV view
# are strided across s but contig within the d dimension.  The 2D-tile
# launch keeps occupancy reasonable (≈ BH × S/TILE_S programs) and amortises
# kernel-launch overhead vs. the original "one program per element" design.
#
# Output buffer is allocated as [B, H, S, D] contig so downstream cuDNN /
# DSL attention can consume it directly without permute().contiguous().
if _triton_available:
    @triton.jit
    def _rope_qkv_fwd_kernel_bhsd(
        qkv_ptr,
        cos_ptr, sin_ptr,
        out_ptr,
        QKV_STRIDE_S,
        QKV_STRIDE_B,
        COL_OFFSET,
        HEAD_STRIDE,
        H, S,
        D: tl.constexpr, HALF_D: tl.constexpr,
        TILE_S: tl.constexpr,
    ):
        """RoPE fwd: QKV stride view → [B, H, S, D] contig output (tiled).

        Grid: (BH * (S // TILE_S),) with s_blk innermost so consecutive
        programs advance the s-tile within the same (b, h), keeping the
        BHSD output write strictly sequential.

        Program pid = bh * (S // TILE_S) + s_blk
        ⇒ (b_idx, h_idx)  =  divmod(bh, H)
        Output write offset (BHSD contig): bh*S*D + s_blk*TILE_S*D + s*D
        Input read offset (QKV stride):    s*QKV_STRIDE_S + b*QKV_STRIDE_B
                                           + COL_OFFSET + h*HEAD_STRIDE
        """
        pid = tl.program_id(0)
        s_blocks = S // TILE_S
        bh = pid // s_blocks
        s_blk = pid % s_blocks

        b_idx = bh // H
        h_idx = bh % H

        offs_s = s_blk * TILE_S + tl.arange(0, TILE_S)  # [TILE_S]
        offs_first = tl.arange(0, HALF_D)               # [HALF_D]
        offs_second = offs_first + HALF_D               # [HALF_D]

        qkv_bh_base = b_idx * QKV_STRIDE_B + h_idx * HEAD_STRIDE + COL_OFFSET

        x_first = tl.load(
            qkv_ptr + offs_s[:, None] * QKV_STRIDE_S
            + qkv_bh_base + offs_first[None, :]
        ).to(tl.float32)
        x_second = tl.load(
            qkv_ptr + offs_s[:, None] * QKV_STRIDE_S
            + qkv_bh_base + offs_second[None, :]
        ).to(tl.float32)

        cos_first = tl.load(cos_ptr + offs_s[:, None] * D + offs_first[None, :]).to(tl.float32)
        sin_first = tl.load(sin_ptr + offs_s[:, None] * D + offs_first[None, :]).to(tl.float32)
        cos_second = tl.load(cos_ptr + offs_s[:, None] * D + offs_second[None, :]).to(tl.float32)
        sin_second = tl.load(sin_ptr + offs_s[:, None] * D + offs_second[None, :]).to(tl.float32)

        out_first = x_first * cos_first - x_second * sin_first
        out_second = x_second * cos_second + x_first * sin_second

        # BHSD contig: out[b, h, s, d] = out_ptr + bh*S*D + s*D + d
        out_bh_base = bh * S * D
        tl.store(
            out_ptr + out_bh_base + offs_s[:, None] * D + offs_first[None, :],
            out_first.to(qkv_ptr.dtype.element_ty),
        )
        tl.store(
            out_ptr + out_bh_base + offs_s[:, None] * D + offs_second[None, :],
            out_second.to(qkv_ptr.dtype.element_ty),
        )


def fused_rope_from_qkv_bhsd(
    qkv_grouped: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tile_s: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE fwd from QKV view, outputs in [B, H, S, D] contig layout.

    Returns (q_rot, k_rot, v_bhsd) where:
      q_rot:   [B, Hq,  S, D] contig BF16
      k_rot:   [B, Hkv, S, D] contig BF16
      v_bhsd:  [B, Hkv, S, D] contig BF16 (v slice extracted via permute+contig)

    Compared to :func:`fused_rope_from_qkv` (which returns [S,B,H,D] contig),
    this lets ``causal_attention_fwd_direct`` / ``causal_attention_fwd_dsl``
    skip the ``q.permute(1,2,0,3).contiguous()`` and ``k.permute(...)`` calls
    that account for ~34 MB of device-to-device traffic per layer per call.
    The kernel switches from "(s,b,h) with h innermost in pid" to a 2-D tile
    layout (bh, s_blk) so the BHSD output write stays sequential.

    The v slice is still copied via a torch permute().contiguous() because
    extracting it into BHSD costs the same ~2 MB memcpy either way; keeping
    it in torch avoids a third Triton kernel that wouldn't save bytes.

    Assumes ``S % tile_s == 0`` (caller's responsibility — the trainer uses
    S=4096 and tile_s=64 always divides cleanly).
    """
    S, B, Hkv, gs = qkv_grouped.shape
    Hq = num_heads
    D = head_dim
    q_per_group = (Hq // Hkv) * D
    assert S % tile_s == 0, f"S({S}) must be divisible by tile_s({tile_s})"

    if get_config().fused_rope_fp32cos:
        cos_data = cos[:S]
        sin_data = sin[:S]
    else:
        cos_data = cos[:S].to(qkv_grouped.dtype)
        sin_data = sin[:S].to(qkv_grouped.dtype)

    qkv_stride_s = qkv_grouped.stride(0)
    qkv_stride_b = qkv_grouped.stride(1)

    s_blocks = S // tile_s

    q_out = torch.empty(B, Hq, S, D, dtype=qkv_grouped.dtype, device=qkv_grouped.device)
    _rope_qkv_fwd_kernel_bhsd[(B * Hq * s_blocks,)](
        qkv_grouped, cos_data, sin_data, q_out,
        qkv_stride_s, qkv_stride_b,
        0, D,
        Hq, S,
        D=D, HALF_D=D // 2,
        TILE_S=tile_s,
    )

    k_out = torch.empty(B, Hkv, S, D, dtype=qkv_grouped.dtype, device=qkv_grouped.device)
    _rope_qkv_fwd_kernel_bhsd[(B * Hkv * s_blocks,)](
        qkv_grouped, cos_data, sin_data, k_out,
        qkv_stride_s, qkv_stride_b,
        q_per_group, D,
        Hkv, S,
        D=D, HALF_D=D // 2,
        TILE_S=tile_s,
    )

    # v is at offset q_per_group + head_dim within each kv-group; it does
    # not need RoPE.  Extract via stride view then permute+contig to BHSD.
    v_sbhd = qkv_grouped[..., q_per_group + D : q_per_group + 2 * D]  # [S,B,Hkv,D] stride view
    v_bhsd = v_sbhd.permute(1, 2, 0, 3).contiguous()                   # [B,Hkv,S,D] contig

    return q_out, k_out, v_bhsd


# ---------------------------------------------------------------------------
# Fused Residual Add + RMSNorm (forward + backward)
# ---------------------------------------------------------------------------
if _triton_available:
    @triton.jit
    def _residual_add_rmsnorm_fwd_kernel(
        residual_ptr, x_ptr, weight_ptr,
        hidden_ptr, norm_ptr, rsigma_ptr,
        N,
        H: tl.constexpr,
        eps: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        offs = tl.arange(0, H)
        row_base = pid * H

        residual = tl.load(residual_ptr + row_base + offs).to(tl.float32)
        x = tl.load(x_ptr + row_base + offs).to(tl.float32)

        out_dtype = residual_ptr.dtype.element_ty
        hidden_bf16 = (residual + x).to(out_dtype)
        hidden = hidden_bf16.to(tl.float32)

        variance = tl.sum(hidden * hidden, axis=0) / H
        rsigma_val = 1.0 / tl.sqrt(variance + eps)

        weight = tl.load(weight_ptr + offs).to(tl.float32)
        norm_out = hidden * rsigma_val * weight

        tl.store(hidden_ptr + row_base + offs, hidden_bf16)
        tl.store(norm_ptr + row_base + offs, norm_out.to(out_dtype))
        tl.store(rsigma_ptr + pid, rsigma_val)

def fused_residual_add_rmsnorm_fwd(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm forward.

    Computes hidden = residual + x, then norm_out = rmsnorm(hidden, weight, eps).
    Eliminates intermediate global memory round-trip for the hidden tensor.

    residual: [S, B, H] or [T, H]   x: [S, B, H] or [T, H]
    weight: [H]   eps: float
    Returns: (hidden [same shape], norm_out [same shape], rsigma [T])
    """
    orig_shape = residual.shape
    H = residual.shape[-1]
    residual_2d = residual.reshape(-1, H).contiguous()
    x_2d = x.reshape(-1, H).contiguous()
    T = residual_2d.shape[0]

    hidden_2d = torch.empty(T, H, dtype=residual.dtype, device=residual.device)
    norm_2d = torch.empty(T, H, dtype=residual.dtype, device=residual.device)
    rsigma = torch.empty(T, dtype=torch.float32, device=residual.device)

    if _triton_available and residual.is_cuda:
        _residual_add_rmsnorm_fwd_kernel[(T,)](
            residual_2d, x_2d, weight, hidden_2d, norm_2d, rsigma,
            T, H=H, eps=eps,
        )
    else:
        raise RuntimeError("Fused residual_add_rmsnorm requires Triton + CUDA")

    return hidden_2d.view(orig_shape), norm_2d.view(orig_shape), rsigma


if _triton_available:
    @triton.jit
    def _residual_add_rmsnorm_bwd_add_kernel(
        d_norm_ptr, hidden_ptr, rsigma_ptr, weight_ptr,
        d_hidden_ptr, dw_partial_ptr,
        N,
        H: tl.constexpr,
    ):
        """RMSNorm backward with in-place d_hidden += dx accumulation.

        Same math as _residual_add_rmsnorm_bwd_kernel but instead of writing
        dx to a separate buffer, loads d_hidden, adds dx, and stores back.
        Eliminates the separate vectorized_elementwise_kernel for d_hidden += dx.
        """
        pid = tl.program_id(0)
        if pid >= N:
            return

        offs = tl.arange(0, H)
        row_base = pid * H

        d_norm = tl.load(d_norm_ptr + row_base + offs).to(tl.float32)
        hidden = tl.load(hidden_ptr + row_base + offs).to(tl.float32)
        rsigma_val = tl.load(rsigma_ptr + pid).to(tl.float32)
        weight = tl.load(weight_ptr + offs).to(tl.float32)

        c = tl.sum(d_norm * weight * hidden, axis=0)
        dx = rsigma_val * (d_norm * weight - (rsigma_val * rsigma_val / H) * hidden * c)

        d_hidden_old = tl.load(d_hidden_ptr + row_base + offs).to(tl.float32)
        d_hidden_new = d_hidden_old + dx
        tl.store(d_hidden_ptr + row_base + offs, d_hidden_new.to(d_norm_ptr.dtype.element_ty))

        dw_row = d_norm * hidden * rsigma_val
        tl.store(dw_partial_ptr + row_base + offs, dw_row)


def fused_residual_add_rmsnorm_bwd_add(
    d_norm: torch.Tensor,
    hidden: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
    d_hidden: torch.Tensor,
) -> torch.Tensor:
    """Fused RMSNorm backward + in-place d_hidden += dx for MLP pre-norm.

    Same as fused_residual_add_rmsnorm_bwd but accumulates dx directly into
    d_hidden instead of returning a separate dx tensor. Eliminates the
    vectorized_elementwise_kernel that computes d_hidden = d_hidden + dx_mlp.

    d_norm: [T, H] BF16   hidden: [T, H] BF16   rsigma: [T] FP32
    weight: [H] BF16   d_hidden: [S, B, H] or [T, H] BF16 (modified in-place)
    Returns: dw [H] FP32
    """
    H = d_norm.shape[-1]
    d_norm_2d = d_norm.reshape(-1, H).contiguous()
    hidden_2d = hidden.reshape(-1, H).contiguous()
    T = d_norm_2d.shape[0]

    d_hidden_2d = d_hidden.reshape(-1, H)
    if not d_hidden_2d.is_contiguous():
        raise RuntimeError("d_hidden must be contiguous for in-place update")

    dw_partial = torch.empty(T, H, dtype=torch.float32, device=d_norm.device)

    if _triton_available and d_norm.is_cuda:
        _residual_add_rmsnorm_bwd_add_kernel[(T,)](
            d_norm_2d, hidden_2d, rsigma, weight,
            d_hidden_2d, dw_partial,
            T, H=H,
        )
    else:
        raise RuntimeError("fused_residual_add_rmsnorm_bwd_add requires Triton + CUDA")

    dw = dw_partial.sum(dim=0)
    return dw


# ---------------------------------------------------------------------------
# Fused RMSNorm backward + in-place d_hidden accumulation
# ---------------------------------------------------------------------------
if _triton_available:
    @triton.jit
    def _rmsnorm_bwd_add_kernel(
        d_norm_ptr, hidden_ptr, rsigma_ptr, weight_ptr,
        d_hidden_ptr, dw_partial_ptr,
        N,
        H: tl.constexpr,
    ):
        """RMSNorm backward with in-place d_hidden += dx accumulation.

        Fuses _te_rmsnorm_backward + d_hidden += dx into a single kernel,
        eliminating the intermediate dx tensor write and the subsequent
        vectorized_elementwise_kernel for the addition.
        """
        pid = tl.program_id(0)
        if pid >= N:
            return

        offs = tl.arange(0, H)
        row_base = pid * H

        d_norm = tl.load(d_norm_ptr + row_base + offs).to(tl.float32)
        hidden = tl.load(hidden_ptr + row_base + offs).to(tl.float32)
        rsigma_val = tl.load(rsigma_ptr + pid).to(tl.float32)
        weight = tl.load(weight_ptr + offs).to(tl.float32)

        c = tl.sum(d_norm * weight * hidden, axis=0)
        dx = rsigma_val * (d_norm * weight - (rsigma_val * rsigma_val / H) * hidden * c)

        d_hidden_old = tl.load(d_hidden_ptr + row_base + offs).to(tl.float32)
        d_hidden_new = d_hidden_old + dx
        tl.store(d_hidden_ptr + row_base + offs, d_hidden_new.to(d_norm_ptr.dtype.element_ty))

        dw_row = d_norm * hidden * rsigma_val
        tl.store(dw_partial_ptr + row_base + offs, dw_row)


def fused_rmsnorm_bwd_add(
    d_norm: torch.Tensor,
    hidden: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
    d_hidden: torch.Tensor,
) -> torch.Tensor:
    """Fused RMSNorm backward + in-place d_hidden += dx.

    Computes dx = rmsnorm_bwd(d_norm, hidden, rsigma, weight), then
    d_hidden += dx in a single kernel. Also computes dw via partial buffer.

    d_norm: [T, H] BF16   hidden: [T, H] BF16   rsigma: [T] FP32
    weight: [H] BF16   d_hidden: [S, B, H] or [T, H] BF16 (modified in-place)
    Returns: dw [H] FP32
    """
    H = d_norm.shape[-1]
    d_norm_2d = d_norm.reshape(-1, H).contiguous()
    hidden_2d = hidden.reshape(-1, H).contiguous()
    T = d_norm_2d.shape[0]

    d_hidden_2d = d_hidden.reshape(-1, H)
    if not d_hidden_2d.is_contiguous():
        raise RuntimeError("d_hidden must be contiguous for in-place update")

    dw_partial = torch.empty(T, H, dtype=torch.float32, device=d_norm.device)

    if _triton_available and d_norm.is_cuda:
        _rmsnorm_bwd_add_kernel[(T,)](
            d_norm_2d, hidden_2d, rsigma, weight,
            d_hidden_2d, dw_partial,
            T, H=H,
        )
    else:
        raise RuntimeError("fused_rmsnorm_bwd_add requires Triton + CUDA")

    dw = dw_partial.sum(dim=0)
    return dw


# ---------------------------------------------------------------------------
# Fused dw reduction variants — process ROWS_PER_BLOCK rows per program,
# accumulate dw in-register, write to a smaller partial buffer.
# Eliminates most of the separate reduce_kernel cost.  ROWS_PER_BLOCK is
# read from EngineConfig at launch time (see callers below) so the value
# is settable per run from the CLI without re-importing the module.
# ---------------------------------------------------------------------------

if _triton_available:
    @triton.jit
    def _residual_add_rmsnorm_bwd_add_dw_reduce_kernel(
        d_norm_ptr, hidden_ptr, rsigma_ptr, weight_ptr,
        d_hidden_ptr, dw_partial_ptr,
        H: tl.constexpr,
        ROWS_PER_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_start = pid * ROWS_PER_BLOCK
        offs = tl.arange(0, H)

        weight = tl.load(weight_ptr + offs).to(tl.float32)
        dw_acc = tl.zeros([H], dtype=tl.float32)

        for row_offset in range(ROWS_PER_BLOCK):
            row = row_start + row_offset
            row_base = row * H

            d_norm = tl.load(d_norm_ptr + row_base + offs).to(tl.float32)
            hidden = tl.load(hidden_ptr + row_base + offs).to(tl.float32)
            rsigma_val = tl.load(rsigma_ptr + row).to(tl.float32)

            c = tl.sum(d_norm * weight * hidden, axis=0)
            dx = rsigma_val * (d_norm * weight - (rsigma_val * rsigma_val / H) * hidden * c)

            d_hidden_old = tl.load(d_hidden_ptr + row_base + offs).to(tl.float32)
            d_hidden_new = d_hidden_old + dx
            tl.store(d_hidden_ptr + row_base + offs, d_hidden_new.to(d_norm_ptr.dtype.element_ty))

            dw_acc += d_norm * hidden * rsigma_val

        tl.store(dw_partial_ptr + pid * H + offs, dw_acc)

    @triton.jit
    def _rmsnorm_bwd_add_dw_reduce_kernel(
        d_norm_ptr, hidden_ptr, rsigma_ptr, weight_ptr,
        d_hidden_ptr, dw_partial_ptr,
        H: tl.constexpr,
        ROWS_PER_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_start = pid * ROWS_PER_BLOCK
        offs = tl.arange(0, H)

        weight = tl.load(weight_ptr + offs).to(tl.float32)
        dw_acc = tl.zeros([H], dtype=tl.float32)

        for row_offset in range(ROWS_PER_BLOCK):
            row = row_start + row_offset
            row_base = row * H

            d_norm = tl.load(d_norm_ptr + row_base + offs).to(tl.float32)
            hidden = tl.load(hidden_ptr + row_base + offs).to(tl.float32)
            rsigma_val = tl.load(rsigma_ptr + row).to(tl.float32)

            c = tl.sum(d_norm * weight * hidden, axis=0)
            dx = rsigma_val * (d_norm * weight - (rsigma_val * rsigma_val / H) * hidden * c)

            d_hidden_old = tl.load(d_hidden_ptr + row_base + offs).to(tl.float32)
            d_hidden_new = d_hidden_old + dx
            tl.store(d_hidden_ptr + row_base + offs, d_hidden_new.to(d_norm_ptr.dtype.element_ty))

            dw_acc += d_norm * hidden * rsigma_val

        tl.store(dw_partial_ptr + pid * H + offs, dw_acc)


def fused_residual_add_rmsnorm_bwd_add_dw_reduce(
    d_norm: torch.Tensor,
    hidden: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
    d_hidden: torch.Tensor,
) -> torch.Tensor:
    """Like fused_residual_add_rmsnorm_bwd_add but with in-kernel dw partial
    reduction: processes ROWS_PER_BLOCK rows per program, accumulates dw in
    registers, writes to a smaller [T//RPB, H] buffer. Eliminates most of the
    separate reduce_kernel cost."""
    H = d_norm.shape[-1]
    d_norm_2d = d_norm.reshape(-1, H).contiguous()
    hidden_2d = hidden.reshape(-1, H).contiguous()
    T = d_norm_2d.shape[0]

    d_hidden_2d = d_hidden.reshape(-1, H)
    if not d_hidden_2d.is_contiguous():
        raise RuntimeError("d_hidden must be contiguous for in-place update")

    RPB = get_config().dw_reduce_rows_per_block
    if T % RPB != 0:
        raise RuntimeError(
            f"T={T} must be divisible by ROWS_PER_BLOCK={RPB} for dw_reduce kernel"
        )
    num_groups = T // RPB
    dw_partial = torch.empty(num_groups, H, dtype=torch.float32, device=d_norm.device)

    _residual_add_rmsnorm_bwd_add_dw_reduce_kernel[(num_groups,)](
        d_norm_2d, hidden_2d, rsigma, weight,
        d_hidden_2d, dw_partial,
        H=H, ROWS_PER_BLOCK=RPB,
    )

    dw = dw_partial.sum(dim=0)
    return dw


def fused_rmsnorm_bwd_add_dw_reduce(
    d_norm: torch.Tensor,
    hidden: torch.Tensor,
    rsigma: torch.Tensor,
    weight: torch.Tensor,
    d_hidden: torch.Tensor,
) -> torch.Tensor:
    """Like fused_rmsnorm_bwd_add but with in-kernel dw partial reduction:
    processes ROWS_PER_BLOCK rows per program, accumulates dw in registers,
    writes to a smaller [T//RPB, H] buffer. Eliminates most of the separate
    reduce_kernel cost."""
    H = d_norm.shape[-1]
    d_norm_2d = d_norm.reshape(-1, H).contiguous()
    hidden_2d = hidden.reshape(-1, H).contiguous()
    T = d_norm_2d.shape[0]

    d_hidden_2d = d_hidden.reshape(-1, H)
    if not d_hidden_2d.is_contiguous():
        raise RuntimeError("d_hidden must be contiguous for in-place update")

    RPB = get_config().dw_reduce_rows_per_block
    if T % RPB != 0:
        raise RuntimeError(
            f"T={T} must be divisible by ROWS_PER_BLOCK={RPB} for dw_reduce kernel"
        )
    num_groups = T // RPB
    dw_partial = torch.empty(num_groups, H, dtype=torch.float32, device=d_norm.device)

    _rmsnorm_bwd_add_dw_reduce_kernel[(num_groups,)](
        d_norm_2d, hidden_2d, rsigma, weight,
        d_hidden_2d, dw_partial,
        H=H, ROWS_PER_BLOCK=RPB,
    )

    dw = dw_partial.sum(dim=0)
    return dw


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
# Causal attention is routed through PyTorch's native
# ``F.scaled_dot_product_attention``, which dispatches to FlashAttention-2 /
# cuDNN-attn on H100 with no extra setup.


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_number: int = 1,
) -> torch.Tensor:
    """Scaled dot-product attention with causal mask.

    q: [S, B, Hq, D], k: [S, B, Hkv, D], v: [S, B, Hkv, D] in BF16.
    Returns: [S, B, Hq*D].
    """
    if q.is_cuda:
        return _sdpa_causal_attention(q, k, v)
    return _manual_causal_attention(q, k, v)


def _sdpa_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Causal attention via PyTorch native scaled-dot-product attention.

    Inputs are [S, B, H, D] sequence-first (matching the rest of the
    forward).  PyTorch SDPA wants [B, H, S, D]; GQA is handled natively
    by SDPA's enable_gqa flag (FlashAttention handles head broadcasting
    internally without materializing expanded KV tensors).
    """
    S, B, Hq, D = q.shape

    q_ = q.permute(1, 2, 0, 3).contiguous()  # [B, Hq, S, D]
    k_ = k.permute(1, 2, 0, 3).contiguous()  # [B, Hkv, S, D]
    v_ = v.permute(1, 2, 0, 3).contiguous()  # [B, Hkv, S, D]

    out = F.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=True,
        enable_gqa=True,
    )  # [B, Hq, S, D]
    return out.permute(2, 0, 1, 3).reshape(S, B, Hq * D)



# ---------------------------------------------------------------------------
# Direct cuDNN attention ATen op (no autograd) — DIRECT_ATTN path
# ---------------------------------------------------------------------------
# Replaces the F.scaled_dot_product_attention + autograd.grad pair with
# explicit aten::_scaled_dot_product_cudnn_attention[_backward] calls.
# Same kernel (cuDNN attention via cuDNN >= 9.10), drops the autograd
# dispatch overhead.
#
# Verified contract on PyTorch 2.8.0a0+nv25.06 / cuDNN 9.10.2:
#   fwd 9-tuple = (out, lse, cum_seq_q=None, cum_seq_k=None,
#                  max_q:int, max_k:int, philox_seed, philox_offset,
#                  debug_attn_mask=None)
#   bwd kwargs  = {grad_out, query, key, value, out, logsumexp,
#                  philox_seed, philox_offset, attn_bias=None,
#                  cum_seq_q=None, cum_seq_k=None, max_q, max_k,
#                  dropout_p=0.0, is_causal=True, scale=None}
#                 → (dq, dk, dv) native GQA; dk/dv bitwise == autograd.


def causal_attention_fwd_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    inputs_already_bhsd: bool = False,
) -> Tuple[torch.Tensor, Tuple]:
    """Direct cuDNN forward for causal-attention; returns (output, saved).

    q: [S, B, Hq, D],  k/v: [S, B, Hkv, D]  in BF16, q.is_cuda=True.
        If ``inputs_already_bhsd=True``, q/k/v are instead [B, H, S, D]
        contig (produced by ``fused_rope_from_qkv_bhsd``) and the
        per-call permute().contiguous() is skipped (~34 MB saved).
    output: [S, B, Hq*D]  matching ``_sdpa_causal_attention``.

    The returned ``saved`` tuple is opaque — pass it back into
    :func:`causal_attention_bwd_direct` to recover (dq, dk, dv).  The
    saved layout is the cuDNN-permuted [B, H, S, D] view of q/k/v plus
    everything cuDNN's _backward op needs (out, lse, philox seed/offset,
    max_q, max_k); the bwd helper handles the un-permute back to the
    [S, B, H, D] layout the rest of the framework expects.
    """
    if not q.is_cuda:
        raise RuntimeError(
            "causal_attention_fwd_direct requires CUDA tensors; "
            "fall back to _manual_causal_attention on CPU."
        )

    if inputs_already_bhsd:
        q_p, k_p, v_p = q, k, v        # [B, H, S, D] contig from RoPE BHSD path
        B, Hq, S, D = q_p.shape
    else:
        S, B, Hq, D = q.shape
        q_p = q.permute(1, 2, 0, 3).contiguous()  # [B, Hq, S, D]
        k_p = k.permute(1, 2, 0, 3).contiguous()  # [B, Hkv, S, D]
        v_p = v.permute(1, 2, 0, 3).contiguous()  # [B, Hkv, S, D]

    # Returns (out, lse, cum_seq_q, cum_seq_k, max_q, max_k,
    #          philox_seed, philox_offset, debug_attn_mask).
    fwd = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q_p, k_p, v_p,
        attn_bias=None, compute_log_sumexp=True,
        dropout_p=0.0, is_causal=True, return_debug_mask=False,
    )
    out_BHqSD = fwd[0]
    lse = fwd[1]
    max_q = fwd[4]
    max_k = fwd[5]
    philox_seed = fwd[6]
    philox_offset = fwd[7]

    # [B, Hq, S, D] → [S, B, Hq*D]   (matches _sdpa_causal_attention out).
    # This 32 MB cliff is structural — output projection consumes [S,B,H]
    # layout; can't be avoided without restructuring the whole model's
    # layout convention.
    out = out_BHqSD.permute(2, 0, 1, 3).reshape(S, B, Hq * D)

    saved = (
        q_p, k_p, v_p, out_BHqSD,
        lse, philox_seed, philox_offset,
        max_q, max_k,
    )
    return out, saved


def causal_attention_bwd_direct(
    d_out: torch.Tensor,
    saved: Tuple,
    return_bhsd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct cuDNN backward; returns (dq, dk, dv).

    ``d_out`` has shape [S, B, Hq*D] matching ``causal_attention_fwd_direct``'s
    output.  ``saved`` is the opaque tuple produced by the fwd helper.

    If ``return_bhsd=False`` (default), outputs are permuted back to
    [S, B, H, D] contig so the SBHD-input RoPE bwd path can consume them.

    If ``return_bhsd=True``, outputs are returned as [B, H, S, D] contig
    straight from cuDNN — caller is expected to feed them to
    ``fused_rope_backward_pack_bhsd`` (saves ~36 MB of dq/dk/dv permute
    contig traffic per layer per call).
    """
    q_p, k_p, v_p, out_BHqSD, lse, philox_seed, philox_offset, max_q, max_k = saved
    S, B, _ = d_out.shape
    Hq, D = q_p.shape[1], q_p.shape[3]

    # [S, B, Hq*D] → [B, Hq, S, D].  This 32 MB cliff is the boundary
    # between output projection (SBH layout) and cuDNN (BHSD layout) —
    # unavoidable unless we restructure the linear's input layout.
    d_out_BHqSD = (
        d_out.view(S, B, Hq, D).permute(1, 2, 0, 3).contiguous()
    )

    dq_BHqSD, dk_BHkvSD, dv_BHkvSD = (
        torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            grad_out=d_out_BHqSD,
            query=q_p, key=k_p, value=v_p, out=out_BHqSD,
            logsumexp=lse,
            philox_seed=philox_seed, philox_offset=philox_offset,
            attn_bias=None,
            cum_seq_q=None, cum_seq_k=None,
            max_q=max_q, max_k=max_k,
            dropout_p=0.0, is_causal=True, scale=None,
        )
    )

    if return_bhsd:
        return dq_BHqSD, dk_BHkvSD, dv_BHkvSD

    # [B, H, S, D] → [S, B, H, D]
    dq = dq_BHqSD.permute(2, 0, 1, 3).contiguous()
    dk = dk_BHkvSD.permute(2, 0, 1, 3).contiguous()
    dv = dv_BHkvSD.permute(2, 0, 1, 3).contiguous()
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Self-developed CuTeDSL flash-attention forward kernel.
# ---------------------------------------------------------------------------
# Bridges :mod:`flash_attn_dsl.host` into the engine.  ``FlashAttnFwdSm90``
# (1D persistent grid, 3 warpgroups, BM=BN=128) is a drop-in for cuDNN's
# ``_scaled_dot_product_cudnn_attention`` when ``EngineConfig.dsl_attn_fwd``
# is enabled: same input layout (``q [B, Hq, S, D]``, ``k/v [B, Hkv, S, D]``,
# native GQA), same return contract (``out`` + ``lse``).  Backward stays
# on the cuDNN ATen op and consumes the DSL ``(out, lse)`` as its saved
# state.
#
# Numerical compatibility relies on cuDNN's ``_backward`` recomputing
# attention probabilities from ``(q, k, scale, lse)`` and fusing with
# ``(out, dout, v)`` for ``dq``/``dk``/``dv``: as long as our ``lse`` is
# in the same natural-log convention as cuDNN's, swapping
# ``(out_dsl, lse_dsl)`` into the backward op gives BF16-equivalent
# gradients to the autograd cuDNN fwd+bwd path.

_DSL_BOOTSTRAPPED = False


def _ensure_dsl_imports() -> "module":  # noqa: F722
    """Lazy import of ``flash_attn_dsl.host`` with optional sys.path bootstrap.

    The vendored :mod:`flash_attn_dsl` and :mod:`quack` packages ship
    inside this repo and are picked up by the standard package
    discovery.  :mod:`cutlass.cute` (NVIDIA's CuTeDSL Python frontend)
    is the external dependency: when installed as a wheel it is
    importable directly; when materialised as an unpacked tree at a
    custom location, point ``CUTLASS_DSL_PATH`` at it (single
    directory) and we will prepend it to ``sys.path`` here.  Caches
    the bootstrap in a module-level flag so subsequent calls are O(1).
    """
    global _DSL_BOOTSTRAPPED
    if not _DSL_BOOTSTRAPPED:
        import os
        import sys
        _hint = os.environ.get("CUTLASS_DSL_PATH", "").strip()
        if _hint and os.path.isdir(_hint) and _hint not in sys.path:
            sys.path.insert(0, _hint)
        _DSL_BOOTSTRAPPED = True
    from flash_attn_dsl import host as _dsl_host
    return _dsl_host


def causal_attention_fwd_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    inputs_already_bhsd: bool = False,
) -> Tuple[torch.Tensor, Tuple]:
    """DSL forward + cuDNN-compatible saved state for the bwd ATen op.

    Same signature as :func:`causal_attention_fwd_direct`.  Backward uses
    :func:`causal_attention_bwd_direct` (cuDNN ATen op) with DSL's
    (out, lse) as saved state; philox seed/offset are zero scalars
    because dropout_p=0 (cuDNN bwd ignores them in that case).
    """
    if not q.is_cuda:
        raise RuntimeError(
            "causal_attention_fwd_dsl requires CUDA tensors; "
            "fall back to _manual_causal_attention on CPU."
        )

    _dsl_host = _ensure_dsl_imports()

    if inputs_already_bhsd:
        q_p, k_p, v_p = q, k, v        # [B, H, S, D] contig from RoPE BHSD path
        B, Hq, S, D = q_p.shape
        Hkv = k_p.shape[1]
    else:
        S, B, Hq, D = q.shape
        Hkv = k.shape[2]
        q_p = q.permute(1, 2, 0, 3).contiguous()  # [B, Hq, S, D]
        k_p = k.permute(1, 2, 0, 3).contiguous()  # [B, Hkv, S, D]
        v_p = v.permute(1, 2, 0, 3).contiguous()  # [B, Hkv, S, D]

    out_BHqSD = torch.empty_like(q_p)
    lse = torch.empty(
        B, Hq, S, device=q.device, dtype=torch.float32,
    )

    scale = 1.0 / math.sqrt(D)
    _dsl_host.run_flash_fwd(
        q_p, k_p, v_p, out_BHqSD, lse,
        softmax_scale=scale, is_causal=True,
    )

    out = out_BHqSD.permute(2, 0, 1, 3).reshape(S, B, Hq * D)

    # cuDNN bwd ATen op needs philox_seed / philox_offset tensors.  With
    # dropout_p=0 they are unused but the op still accepts only Tensor
    # values for these slots, so build matching zero scalars.
    philox_seed = torch.zeros((), dtype=torch.int64, device=q.device)
    philox_offset = torch.zeros((), dtype=torch.int64, device=q.device)

    saved = (
        q_p, k_p, v_p, out_BHqSD,
        lse, philox_seed, philox_offset,
        S, S,  # max_q, max_k
    )
    return out, saved


def _manual_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Manual unfused attention (CPU / non-TE fallback)."""
    S, B, Hq, D = q.shape
    Hkv = k.shape[2]
    heads_per_group = Hq // Hkv
    scale = 1.0 / math.sqrt(D)

    q_ = q.permute(1, 2, 0, 3)  # [B, Hq, S, D]
    k_ = k.permute(1, 2, 0, 3)  # [B, Hkv, S, D]
    v_ = v.permute(1, 2, 0, 3)  # [B, Hkv, S, D]

    if heads_per_group > 1:
        k_ = k_.repeat_interleave(heads_per_group, dim=1)
        v_ = v_.repeat_interleave(heads_per_group, dim=1)

    q_ = q_ * scale
    scores = torch.matmul(q_, k_.transpose(-2, -1))  # [B, Hq, S, S]

    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
    )
    scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)

    ctx = torch.matmul(probs, v_)  # [B, Hq, S, D]
    ctx = ctx.permute(2, 0, 1, 3).reshape(S, B, Hq * D)
    return ctx


# ---------------------------------------------------------------------------
# SwiGLU activation
# ---------------------------------------------------------------------------
@torch.compile
def _swiglu_compiled(y: torch.Tensor) -> torch.Tensor:
    """Megatron's exact @torch.compile SwiGLU forward.

    Triton auto-promotes BF16→FP32 for silu and multiply, then stores as BF16.
    Must use @torch.compile to get bitwise-identical Triton codegen.
    """
    y = torch.chunk(y, 2, -1)
    return torch.ops.aten.silu(y[0]) * y[1]


if _triton_available:
    @triton.jit
    def _swiglu_fwd_kernel(
        y_ptr, out_ptr,
        T,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """SwiGLU forward: out = silu(y_1) * y_2 where y = [y_1, y_2]."""
        pid = tl.program_id(0)
        if pid >= T:
            return
        row_base = pid * D * 2
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            mask = d_offs < D
            y_1 = tl.load(y_ptr + row_base + d_offs, mask=mask).to(tl.float32)
            y_2 = tl.load(y_ptr + row_base + D + d_offs, mask=mask).to(tl.float32)
            sig = 1.0 / (1.0 + tl.exp(-y_1))
            silu = y_1 * sig
            out_val = silu * y_2
            tl.store(out_ptr + pid * D + d_offs, out_val.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        g_ptr, y_ptr, out_ptr,
        T,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """SwiGLU backward: computes d_gate and d_up, writes to packed output.

        g: [T, D] gradient of SwiGLU output
        y: [T, 2*D] saved fc1_out (gate || up)
        out: [T, 2*D] gradient of fc1_out (d_gate || d_up)
        """
        pid = tl.program_id(0)
        if pid >= T:
            return
        y_base = pid * D * 2
        g_base = pid * D
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            mask = d_offs < D
            g = tl.load(g_ptr + g_base + d_offs, mask=mask).to(tl.float32)
            y_1 = tl.load(y_ptr + y_base + d_offs, mask=mask).to(tl.float32)
            y_2 = tl.load(y_ptr + y_base + D + d_offs, mask=mask).to(tl.float32)
            sig = 1.0 / (1.0 + tl.exp(-y_1))
            d_gate = g * sig * (1.0 + y_1 * (1.0 - sig)) * y_2
            d_up = g * y_1 * sig
            tl.store(out_ptr + y_base + d_offs, d_gate.to(g_ptr.dtype.element_ty), mask=mask)
            tl.store(out_ptr + y_base + D + d_offs, d_up.to(g_ptr.dtype.element_ty), mask=mask)


def _triton_swiglu_forward(y_2d: torch.Tensor) -> torch.Tensor:
    """SwiGLU forward via Triton. y_2d: [T, 2*D] → [T, D]."""
    T = y_2d.shape[0]
    D = y_2d.shape[1] // 2
    out = torch.empty(T, D, dtype=y_2d.dtype, device=y_2d.device)
    BLOCK_D = min(4096, triton.next_power_of_2(D))
    _swiglu_fwd_kernel[(T,)](y_2d, out, T, D=D, BLOCK_D=BLOCK_D)
    return out


def _triton_swiglu_backward(g_2d: torch.Tensor, y_2d: torch.Tensor) -> torch.Tensor:
    """SwiGLU backward via Triton. g_2d: [T, D], y_2d: [T, 2*D] → [T, 2*D]."""
    T = g_2d.shape[0]
    D = g_2d.shape[1]
    out = torch.empty(T, D * 2, dtype=g_2d.dtype, device=g_2d.device)
    BLOCK_D = min(4096, triton.next_power_of_2(D))
    _swiglu_bwd_kernel[(T,)](g_2d, y_2d, out, T, D=D, BLOCK_D=BLOCK_D)
    return out


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(gate) * up.  x shape [..., 2*FFN] → [..., FFN]."""
    x_2d = x.reshape(-1, x.shape[-1])
    if get_config().fused_swiglu and _triton_available and x.is_cuda:
        out_2d = _triton_swiglu_forward(x_2d)
    else:
        out_2d = _swiglu_compiled(x_2d)
    return out_2d.view(*x.shape[:-1], out_2d.shape[-1])


@torch.compile
def _swiglu_back_compiled(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Megatron's swiglu_back formula under @torch.compile.

    Triton auto-promotes BF16→FP32 for all arithmetic; result stored as BF16.
    Mirrors megatron.core.fusions.fused_bias_swiglu.swiglu_back exactly.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2,
         g * torch.ops.aten.silu(y_1)),
        -1,
    )


def swiglu_backward(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SwiGLU backward SSOT — dispatches Triton vs @torch.compile."""
    g_2d = g.reshape(-1, g.shape[-1])
    y_2d = y.reshape(-1, y.shape[-1])
    if get_config().fused_swiglu and _triton_available and g.is_cuda:
        out_2d = _triton_swiglu_backward(g_2d, y_2d)
    else:
        out_2d = _swiglu_back_compiled(g_2d, y_2d)
    return out_2d.view(*y.shape[:-1], out_2d.shape[-1])


# ---------------------------------------------------------------------------
# Fused CE loss Triton kernels (vocab-parallel cross-entropy)
#
# Three kernels replace the ~12 PyTorch ops in compute_ce_loss_and_grad_tp:
#   Kernel A (_ce_local_max): per-row max of BF16 logits → local_max [T] FP32
#   Kernel B (_ce_sum_pred):  given global_max, compute local sum_exp +
#                             predicted logit → [T] FP32 each
#   Kernel C (_ce_bwd):       given global stats, compute d_logits BF16
#
# Benefits: eliminates 6+ large FP32 intermediate tensors (logits_fp32,
# exp_logits, softmax, one_hot, d_logits_fp32), reducing HBM traffic from
# ~11 GB to ~2.4 GB per CE call.
# ---------------------------------------------------------------------------
if _triton_available:
    @triton.jit
    def _ce_online_max_sum_pred_kernel(
        logits_ptr,
        local_max_ptr,
        local_sum_ptr,
        predicted_raw_ptr,
        labels_ptr,
        vocab_start,
        T,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Fused single-pass max + sum_exp + predicted using online softmax.

        Reads each logit row once.  Maintains a running max and corrects the
        running sum_exp when a new block-max exceeds the current max.  This
        is numerically equivalent to the two-pass (max then sum) approach.
        """
        pid = tl.program_id(0)
        if pid >= T:
            return

        row_base = pid * V
        label = tl.load(labels_ptr + pid)
        target_local = label - vocab_start
        in_range = (target_local >= 0) & (target_local < V)
        target_clamped = tl.where(in_range, target_local, 0)

        max_val = float('-inf')
        sum_exp = 0.0
        pred_raw = 0.0

        for v_start in range(0, V, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            mask = v_offs < V
            logit = tl.load(
                logits_ptr + row_base + v_offs, mask=mask, other=float('-inf'),
            ).to(tl.float32)

            block_max = tl.max(logit, axis=0)
            new_max = tl.where(block_max > max_val, block_max, max_val)
            correction = tl.exp(max_val - new_max)
            sum_exp = sum_exp * correction + tl.sum(
                tl.where(mask, tl.exp(logit - new_max), 0.0), axis=0,
            )
            max_val = new_max

            target_mask = (v_offs == target_clamped) & in_range
            pred_raw += tl.sum(
                tl.where(target_mask, logit, 0.0), axis=0,
            )

        tl.store(local_max_ptr + pid, max_val)
        tl.store(local_sum_ptr + pid, sum_exp)
        tl.store(predicted_raw_ptr + pid, pred_raw)

    @triton.jit
    def _ce_bwd_kernel(
        logits_ptr,
        d_logits_ptr,
        global_max_ptr,
        global_sum_ptr,
        labels_ptr,
        upstream_ptr,
        vocab_start,
        T,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Compute d_logits_bf16 = (softmax_local - one_hot) * upstream."""
        pid = tl.program_id(0)
        if pid >= T:
            return
        row_base = pid * V
        gmax = tl.load(global_max_ptr + pid)
        gsum = tl.load(global_sum_ptr + pid)
        upstream = tl.load(upstream_ptr + pid)
        label = tl.load(labels_ptr + pid)
        target_local = label - vocab_start
        in_range = (target_local >= 0) & (target_local < V)
        target_clamped = tl.where(in_range, target_local, 0)

        inv_sum = 1.0 / gsum
        for v_start in range(0, V, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            mask = v_offs < V
            logit = tl.load(
                logits_ptr + row_base + v_offs, mask=mask, other=float('-inf'),
            ).to(tl.float32)
            shifted = logit - gmax
            softmax_val = tl.exp(shifted) * inv_sum
            one_hot = ((v_offs == target_clamped) & in_range).to(tl.float32)
            d_val = (softmax_val - one_hot) * upstream
            tl.store(
                d_logits_ptr + row_base + v_offs,
                d_val.to(tl.bfloat16),
                mask=mask,
            )


def fused_ce_max_sum_pred(
    logits_bf16: torch.Tensor,
    labels_flat: torch.Tensor,
    vocab_start: int,
) -> tuple:
    """Single-pass local_max + local_sum_exp + predicted_raw via online softmax.

    Merges kernels A (local_max) and B (sum_pred) into one Triton kernel that
    reads logits only once, using the online softmax trick to compute max and
    sum_exp simultaneously.  Saves one full logits read (~573 MB for 8B model).

    Returns (local_max [T] FP32, local_sum [T] FP32, predicted_raw [T] FP32).
    local_sum is computed using local_max (not global_max) — caller must apply
    correction: corrected_sum = local_sum * exp(local_max - global_max).
    predicted_raw is the unshifted logit at the target position (0 if target
    not in this TP rank's vocab shard).
    """
    T, V = logits_bf16.shape[0], logits_bf16.shape[-1]
    logits_2d = logits_bf16.reshape(T, V) if logits_bf16.dim() > 2 else logits_bf16
    T = logits_2d.shape[0]
    V = logits_2d.shape[1]
    local_max = torch.empty(T, dtype=torch.float32, device=logits_bf16.device)
    local_sum = torch.empty(T, dtype=torch.float32, device=logits_bf16.device)
    predicted_raw = torch.empty(T, dtype=torch.float32, device=logits_bf16.device)
    BLOCK_V = min(4096, triton.next_power_of_2(V))
    _ce_online_max_sum_pred_kernel[(T,)](
        logits_2d, local_max, local_sum, predicted_raw,
        labels_flat, vocab_start,
        T, V=V, BLOCK_V=BLOCK_V,
    )
    return local_max, local_sum, predicted_raw


if _triton_available:
    @triton.jit
    def _ce_correction_kernel(
        local_max_saved_ptr, global_max_ptr, local_sum_ptr, predicted_raw_ptr,
        labels_ptr, ar_buf_ptr,
        vocab_start, V_local, T: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T

        lm_saved = tl.load(local_max_saved_ptr + offs, mask=mask)
        gm = tl.load(global_max_ptr + offs, mask=mask)
        ls = tl.load(local_sum_ptr + offs, mask=mask)
        pr = tl.load(predicted_raw_ptr + offs, mask=mask)
        lab = tl.load(labels_ptr + offs, mask=mask)

        corrected_sum = ls * tl.exp(lm_saved - gm)

        target_local = lab - vocab_start
        in_range = (target_local >= 0) & (target_local < V_local)
        predicted_local = tl.where(in_range, pr - gm, 0.0)

        tl.store(ar_buf_ptr + offs, corrected_sum, mask=mask)
        tl.store(ar_buf_ptr + T + offs, predicted_local, mask=mask)


def fused_ce_correction(
    local_max_saved: torch.Tensor,
    global_max: torch.Tensor,
    local_sum: torch.Tensor,
    predicted_raw: torch.Tensor,
    labels_flat: torch.Tensor,
    vocab_start: int,
    V_local: int,
) -> torch.Tensor:
    """Fused correction between TP all-reduce rounds in CE forward.

    Replaces ~11 PyTorch kernel launches with 1 Triton kernel.
    Returns ar_buf [2, T] FP32: row 0 = corrected local_sum, row 1 = predicted_local.
    """
    T = local_max_saved.shape[0]
    ar_buf = torch.empty(2, T, dtype=torch.float32, device=local_max_saved.device)
    BLOCK = 1024
    grid = ((T + BLOCK - 1) // BLOCK,)
    _ce_correction_kernel[grid](
        local_max_saved, global_max, local_sum, predicted_raw,
        labels_flat, ar_buf,
        vocab_start, V_local, T, BLOCK=BLOCK,
    )
    return ar_buf


def fused_ce_bwd(
    logits_bf16: torch.Tensor,
    global_max: torch.Tensor,
    global_sum: torch.Tensor,
    labels_flat: torch.Tensor,
    upstream: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    """Compute d_logits BF16 via Triton. Returns [T, V_local] BF16."""
    orig_shape = logits_bf16.shape
    T, V = logits_bf16.shape[0], logits_bf16.shape[-1]
    logits_2d = logits_bf16.reshape(T, V) if logits_bf16.dim() > 2 else logits_bf16
    T = logits_2d.shape[0]
    V = logits_2d.shape[1]
    d_logits = torch.empty_like(logits_2d)
    BLOCK_V = min(4096, triton.next_power_of_2(V))
    _ce_bwd_kernel[(T,)](
        logits_2d, d_logits,
        global_max, global_sum, labels_flat, upstream,
        vocab_start, T, V=V, BLOCK_V=BLOCK_V,
    )
    return d_logits.view(orig_shape)
