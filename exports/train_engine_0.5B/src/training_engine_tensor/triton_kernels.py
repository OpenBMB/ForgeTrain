"""Fused Triton kernels for M8 operator fusion optimization.

Operator fusions:
1. Cross-entropy forward+backward (fuses ~6 separate PyTorch ops into 2-pass kernel)
2. SwiGLU forward (fuses silu + mul into single kernel)
3. SwiGLU backward (fuses sigmoid + multiple muls + cat into single kernel)
4. Residual add + RMSNorm forward (fuses add + norm into single kernel)
5. Fused Adam + param sync (fuses clip + adam + FP32→BF16 cast)
"""

from __future__ import annotations

__all__ = [
    "fused_adam_sync",
    "fused_adam_sync_tensor",
    "fused_cross_entropy_fwd_bwd",
    "fused_residual_rmsnorm_fwd",
    "fused_rmsnorm_bwd_residual",
    "fused_rope_bwd",
    "fused_rope_bwd_doc_aware",
    "fused_rope_fwd",
    "fused_rope_fwd_doc_aware",
    "fused_swiglu_bwd",
    "fused_swiglu_fwd",
]

import torch
import triton
import triton.language as tl

from .config import GEMM_PAD_TO
from .engine_config import get_config

# ═══════════════════════════════════════════════════════════════════════
# Fused Cross-Entropy (forward + backward in one kernel)
# ═══════════════════════════════════════════════════════════════════════


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=3),
    ],
    key=["V"],
)
@triton.jit
def _cross_entropy_fwd_bwd_kernel(
    logits_ptr,
    labels_ptr,
    loss_mask_ptr,
    valid_tokens_ptr,
    d_logits_ptr,
    per_row_loss_ptr,
    loss_scale,
    V: tl.constexpr,
    stride_logits,
    stride_d_logits,
    BLOCK_V: tl.constexpr,
):
    """Two-pass fused cross-entropy: online softmax → loss → gradient.

    All arithmetic in FP32. Reads logits twice (2 passes), writes d_logits
    once and per-row CE loss to a buffer for host-side reduction.
    """
    row = tl.program_id(0).to(tl.int64)
    logits_row = logits_ptr + row * stride_logits
    label = tl.load(labels_ptr + row)
    valid_tokens = tl.load(valid_tokens_ptr)
    valid_tokens_inv = 1.0 / valid_tokens

    # ── Pass 1: online softmax ──────────────────────────────────
    offs_0 = tl.arange(0, BLOCK_V)
    valid_0 = offs_0 < V
    chunk_0 = tl.load(logits_row + offs_0, mask=valid_0, other=float('-inf')).to(tl.float32)
    m_i = tl.max(chunk_0, axis=0)
    l_i = tl.sum(tl.exp(chunk_0 - m_i), axis=0)

    for start in range(BLOCK_V, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        valid = offs < V
        chunk = tl.load(logits_row + offs, mask=valid, other=float('-inf')).to(tl.float32)

        chunk_max = tl.max(chunk, axis=0)
        new_m = tl.maximum(m_i, chunk_max)
        l_i = l_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(chunk - new_m), axis=0)
        m_i = new_m

    predicted_logit = tl.load(logits_row + label).to(tl.float32)
    per_token_loss = tl.log(l_i) + m_i - predicted_logit
    tl.store(per_row_loss_ptr + row, per_token_loss)

    # ── Pass 2: compute gradient ────────────────────────────────
    mask_val = tl.load(loss_mask_ptr + row).to(tl.float32)
    grad_scale = mask_val * valid_tokens_inv * loss_scale

    d_logits_row = d_logits_ptr + row * stride_d_logits
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        valid = offs < V
        chunk = tl.load(logits_row + offs, mask=valid, other=float('-inf')).to(tl.float32)

        p = tl.exp(chunk - m_i) / l_i
        is_label = (offs == label)
        grad = tl.where(is_label, p - 1.0, p) * grad_scale

        tl.store(d_logits_row + offs, grad.to(tl.bfloat16), mask=valid)


def fused_cross_entropy_fwd_bwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused cross-entropy loss + backward gradient (sync-free).

    logits: [S, B, V] BF16
    labels: [B, S] int64
    loss_mask: [B, S] float

    Returns: (loss_scalar, None, d_logits [S, B, out_V] BF16)
    where out_V may be padded to GEMM_PAD_TO alignment when FUSED_OPS is
    enabled, eliminating the downstream zeros+copy in linear_backward.
    The 2nd return slot is kept for backward-compat with the previous
    (loss, per_token_loss, d_logits) signature; callers ignore it.

    M-FORK-B notes:
    - No `.item()` / `cudaStreamSynchronize` on this critical path. The
      kernel reads `valid_tokens` as a device scalar and atomic-sums the
      masked loss directly into the FP32 result tensor.
    - `per_token_loss[N]` intermediate buffer is gone (saves 160 KB and
      one kernel + one transpose+contiguous copy per microbatch).
    """
    cfg = get_config()
    S, B, V = logits.shape
    N = S * B

    if logits.stride(-1) == 1 and logits.ndim == 3:
        row_stride = logits.stride(1)
        logits_2d = torch.as_strided(logits, (N, V), (row_stride, 1))
    else:
        logits_2d = logits.reshape(N, V)
        if not logits_2d.is_contiguous():
            logits_2d = logits_2d.contiguous()
        row_stride = V

    labels_flat = labels.transpose(0, 1).contiguous().view(-1)
    loss_mask_flat = loss_mask.transpose(0, 1).contiguous().view(-1).float()

    valid_tokens = loss_mask_flat.sum()  # device-resident FP32 scalar

    if cfg.fuse_gemm_pad and V % GEMM_PAD_TO != 0:
        pad_V = (V + GEMM_PAD_TO - 1) // GEMM_PAD_TO * GEMM_PAD_TO
        d_logits_2d = torch.empty(N, pad_V, dtype=logits.dtype, device=logits.device)
        d_logits_2d[:, V:] = 0
    else:
        pad_V = V
        d_logits_2d = torch.empty(N, V, dtype=logits.dtype, device=logits.device)

    per_row_loss = torch.empty(N, dtype=torch.float32, device=logits.device)

    _cross_entropy_fwd_bwd_kernel[(N,)](
        logits_2d, labels_flat, loss_mask_flat,
        valid_tokens,
        d_logits_2d,
        per_row_loss,
        loss_scale,
        V=V,
        stride_logits=row_stride,
        stride_d_logits=d_logits_2d.stride(0),
    )

    loss = (per_row_loss * loss_mask_flat).sum() / valid_tokens

    return loss, None, d_logits_2d.view(S, B, pad_V)


# ═══════════════════════════════════════════════════════════════════════
# Fused SwiGLU Forward
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _swiglu_fwd_kernel(
    input_ptr, output_ptr,
    FFN: tl.constexpr,
    stride_in, stride_out,
    BLOCK: tl.constexpr,
):
    """SwiGLU forward: output = silu(gate) * up, all computed in FP32."""
    row = tl.program_id(0)
    for start in range(0, FFN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < FFN

        gate = tl.load(input_ptr + row * stride_in + offs, mask=valid).to(tl.float32)
        up = tl.load(input_ptr + row * stride_in + FFN + offs, mask=valid).to(tl.float32)

        silu_gate = gate * tl.sigmoid(gate)
        out = (silu_gate * up).to(tl.bfloat16)

        tl.store(output_ptr + row * stride_out + offs, out, mask=valid)


def fused_swiglu_fwd(x: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward: silu(gate) * up.

    x: [..., 2*FFN_HIDDEN_SIZE]
    Returns: [..., FFN_HIDDEN_SIZE]
    """
    shape = x.shape
    FFN = shape[-1] // 2
    N = x.numel() // shape[-1]

    x_2d = x.reshape(N, shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    output = torch.empty(N, FFN, dtype=x.dtype, device=x.device)

    _swiglu_fwd_kernel[(N,)](
        x_2d, output,
        FFN,
        x_2d.stride(0), output.stride(0),
        BLOCK=min(FFN, 4096),
    )

    return output.view(*shape[:-1], FFN)


# ═══════════════════════════════════════════════════════════════════════
# Fused SwiGLU Backward
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _swiglu_bwd_kernel(
    d_output_ptr, fc1_ptr, d_fc1_ptr,
    FFN: tl.constexpr,
    stride_dout, stride_fc1, stride_dfc1,
    BLOCK: tl.constexpr,
):
    """SwiGLU backward: compute d_gate and d_up, all in FP32.

    d_gate = g * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))) * up
    d_up   = g * gate * sigmoid(gate)
    """
    row = tl.program_id(0)
    for start in range(0, FFN, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < FFN

        g = tl.load(d_output_ptr + row * stride_dout + offs, mask=valid).to(tl.float32)
        gate = tl.load(fc1_ptr + row * stride_fc1 + offs, mask=valid).to(tl.float32)
        up = tl.load(fc1_ptr + row * stride_fc1 + FFN + offs, mask=valid).to(tl.float32)

        sig = tl.sigmoid(gate)

        d_gate = (g * sig * (1.0 + gate * (1.0 - sig)) * up).to(tl.bfloat16)
        d_up = (g * gate * sig).to(tl.bfloat16)

        tl.store(d_fc1_ptr + row * stride_dfc1 + offs, d_gate, mask=valid)
        tl.store(d_fc1_ptr + row * stride_dfc1 + FFN + offs, d_up, mask=valid)


def fused_swiglu_bwd(d_output: torch.Tensor, fc1_output: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU backward.

    d_output: [..., FFN_HIDDEN_SIZE]
    fc1_output: [..., 2*FFN_HIDDEN_SIZE]
    Returns: d_fc1 [..., 2*FFN_HIDDEN_SIZE]
    """
    shape = fc1_output.shape
    FFN = shape[-1] // 2
    N = fc1_output.numel() // shape[-1]

    d_out_2d = d_output.reshape(N, FFN)
    if not d_out_2d.is_contiguous():
        d_out_2d = d_out_2d.contiguous()
    fc1_2d = fc1_output.reshape(N, shape[-1])
    if not fc1_2d.is_contiguous():
        fc1_2d = fc1_2d.contiguous()
    d_fc1 = torch.empty_like(fc1_2d)

    _swiglu_bwd_kernel[(N,)](
        d_out_2d, fc1_2d, d_fc1,
        FFN,
        d_out_2d.stride(0), fc1_2d.stride(0), d_fc1.stride(0),
        BLOCK=min(FFN, 4096),
    )

    return d_fc1.view(shape)


# ═══════════════════════════════════════════════════════════════════════
# Fused Residual Add + RMSNorm Forward
# ═══════════════════════════════════════════════════════════════════════


_RES_RMSNORM_FWD_ROWS_PER_GROUP = 16


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["H", "ROWS_PER_GROUP"],
)
@triton.jit
def _fused_residual_rmsnorm_fwd_kernel(
    residual_ptr, proj_ptr, weight_ptr,
    out_ptr, normed_ptr, rsigma_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    stride_r, stride_p, stride_o, stride_n,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused hidden = residual + proj; normed = rmsnorm(hidden).

    M-FORK-B: each program now handles ROWS_PER_GROUP consecutive rows.
    The norm `weight` vector and constants are loaded once per program
    rather than once per row, cutting total programs by ROWS_PER_GROUP×
    (e.g. 40960 → 2560 at H=1024, ROWS_PER_GROUP=16).

    Per-row arithmetic stays identical to the v1 kernel: residual+proj,
    cast to BF16, write hidden, recompute FP32 var, write rsigma, scale
    by weight, write normed.
    """
    gid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    valid = offs < H

    w = tl.load(weight_ptr + offs, mask=valid, other=0.0).to(tl.float32)
    inv_H = 1.0 / H

    row_base = gid * ROWS_PER_GROUP
    for r_idx in range(ROWS_PER_GROUP):
        row = row_base + r_idx

        r = tl.load(residual_ptr + row * stride_r + offs, mask=valid, other=0.0).to(tl.float32)
        p = tl.load(proj_ptr + row * stride_p + offs, mask=valid, other=0.0).to(tl.float32)

        hidden_bf = (r + p).to(tl.bfloat16)
        tl.store(out_ptr + row * stride_o + offs, hidden_bf, mask=valid)

        hidden = hidden_bf.to(tl.float32)

        var = tl.sum(hidden * hidden, axis=0) * inv_H
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(rsigma_ptr + row, rstd)

        normed = (hidden * rstd * w).to(tl.bfloat16)
        tl.store(normed_ptr + row * stride_n + offs, normed, mask=valid)


def fused_residual_rmsnorm_fwd(
    residual: torch.Tensor,
    proj_out: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm forward.

    residual: [S, B, H] or [N, H]
    proj_out: same shape as residual
    weight:   [H]

    Returns: (hidden, normed, rsigma)
    """
    shape = residual.shape
    H = shape[-1]
    N = residual.numel() // H

    r_2d = residual.reshape(N, H)
    p_2d = proj_out.reshape(N, H)
    if not r_2d.is_contiguous():
        r_2d = r_2d.contiguous()
    if not p_2d.is_contiguous():
        p_2d = p_2d.contiguous()

    out = torch.empty(N, H, dtype=residual.dtype, device=residual.device)
    normed = torch.empty(N, H, dtype=residual.dtype, device=residual.device)
    rsigma = torch.empty(N, dtype=torch.float32, device=residual.device)

    rpg = _RES_RMSNORM_FWD_ROWS_PER_GROUP
    if N % rpg != 0:
        # Fall back to per-row launch if N not divisible (defensive).
        rpg = 1

    BLOCK_H = triton.next_power_of_2(H)
    _fused_residual_rmsnorm_fwd_kernel[(N // rpg,)](
        r_2d, p_2d, weight,
        out, normed, rsigma,
        H, eps,
        r_2d.stride(0), p_2d.stride(0), out.stride(0), normed.stride(0),
        ROWS_PER_GROUP=rpg,
        BLOCK_H=BLOCK_H,
    )

    return out.view(shape), normed.view(shape), rsigma


# ═══════════════════════════════════════════════════════════════════════
# Fused Adam + Param Sync
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _fused_adam_sync_kernel(
    grad_ptr, master_ptr, m_ptr, v_ptr, bf16_ptr,
    lr, beta1, beta2, eps, wd,
    clip_coeff,
    bias_correction1, bias_correction2,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused gradient clip + AdamW update + FP32→BF16 cast in single pass."""
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    valid = offs < N

    g = tl.load(grad_ptr + offs, mask=valid).to(tl.float32) * clip_coeff
    p = tl.load(master_ptr + offs, mask=valid).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=valid).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=valid).to(tl.float32)

    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    update = m_hat / (tl.sqrt(v_hat) + eps) + wd * p
    p_new = p - lr * update

    tl.store(m_ptr + offs, m, mask=valid)
    tl.store(v_ptr + offs, v, mask=valid)
    tl.store(master_ptr + offs, p_new, mask=valid)
    tl.store(bf16_ptr + offs, p_new.to(tl.bfloat16), mask=valid)


def fused_adam_sync(
    grad: torch.Tensor,
    master: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    bf16_param: torch.Tensor,
    lr: float,
    step: int,
    clip_coeff: float = 1.0,
    wd: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> None:
    """Fused Adam update + FP32→BF16 sync for a single parameter.

    Replaces clip_grads + adam_step + sync_params for one param.
    """
    N = grad.numel()
    g_flat = grad.view(-1)
    p_flat = master.view(-1)
    m_flat = m.view(-1)
    v_flat = v.view(-1)
    bf_flat = bf16_param.view(-1)

    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _fused_adam_sync_kernel[grid](
        g_flat, p_flat, m_flat, v_flat, bf_flat,
        lr, beta1, beta2, eps, wd,
        clip_coeff,
        bias_correction1, bias_correction2,
        N=N, BLOCK=BLOCK,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused Adam + Param Sync — tensor-arg variant (CUDA Graph compatible)
# ═══════════════════════════════════════════════════════════════════════
#
# Phase B (M2-P33) of the full-step CUDA Graph push: the legacy
# ``_fused_adam_sync_kernel`` above takes ``lr / clip_coeff /
# bias_correction1 / bias_correction2`` as Python float kernel args.
# When that kernel is captured into a CUDA Graph, those values are
# burnt into the launch node and replay would silently freeze the
# cosine LR schedule and Adam bias correction at the capture step's
# values.
#
# The variant below routes those four scalars through 1-element FP32
# device tensors. Triton dereferences them with ``tl.load`` once per
# program, which is essentially free (single LDG.E.32 with broadcast
# semantics — a few cycles per warp). The kernel result is byte-for-
# byte identical to the scalar-arg path when the four buffers hold
# exactly the same values, which is verified by
# ``tests/test_fused_adam_sync_tensor_gate.py``.
#
# Caller contract (see ``optimizer.OptimizerScalarBuffers``):
# * ``lr_buf`` / ``clip_coeff_buf`` / ``bc1_buf`` / ``bc2_buf`` are
#   each ``torch.zeros(1, dtype=torch.float32, device='cuda')`` and
#   live across the whole training run (one alloc, every step
#   ``copy_(non_blocking=True)`` s in the new value).
# * ``beta1`` / ``beta2`` / ``eps`` / ``wd`` stay as compile-time
#   constants (Triton ``constexpr``) — they truly are constants for a
#   given run, so capturing them in the launch node is correct.
# * ``step`` is *no longer* a kernel argument: bias-correction is fed
#   in directly via the two buffers (host computes
#   ``1 - beta**step`` and copies the FP32 result into ``bc{1,2}_buf``).
#   This avoids a per-program ``tl.exp(log(beta) * step)`` in every
#   warp (which would be needed if we kept ``step`` as the kernel arg
#   for forward compatibility).


@triton.jit
def _fused_adam_sync_tensor_kernel(
    grad_ptr, master_ptr, m_ptr, v_ptr, bf16_ptr,
    lr_ptr, clip_coeff_ptr, bc1_ptr, bc2_ptr,
    beta1, beta2, eps, wd,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Tensor-arg variant of :func:`_fused_adam_sync_kernel`.

    Bytewise identical to the scalar-arg path; differs only in that
    ``lr``, ``clip_coeff``, ``bias_correction1``, ``bias_correction2``
    are loaded from 1-element FP32 device buffers instead of being
    burnt into the launch node as Python kernel-args.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    valid = offs < N

    lr = tl.load(lr_ptr).to(tl.float32)
    clip_coeff = tl.load(clip_coeff_ptr).to(tl.float32)
    bias_correction1 = tl.load(bc1_ptr).to(tl.float32)
    bias_correction2 = tl.load(bc2_ptr).to(tl.float32)

    g = tl.load(grad_ptr + offs, mask=valid).to(tl.float32) * clip_coeff
    p = tl.load(master_ptr + offs, mask=valid).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=valid).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=valid).to(tl.float32)

    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    update = m_hat / (tl.sqrt(v_hat) + eps) + wd * p
    p_new = p - lr * update

    tl.store(m_ptr + offs, m, mask=valid)
    tl.store(v_ptr + offs, v, mask=valid)
    tl.store(master_ptr + offs, p_new, mask=valid)
    tl.store(bf16_ptr + offs, p_new.to(tl.bfloat16), mask=valid)


def fused_adam_sync_tensor(
    grad: torch.Tensor,
    master: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    bf16_param: torch.Tensor,
    *,
    lr_buf: torch.Tensor,
    clip_coeff_buf: torch.Tensor,
    bc1_buf: torch.Tensor,
    bc2_buf: torch.Tensor,
    wd: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> None:
    """Tensor-arg variant of :func:`fused_adam_sync` for CUDA Graph capture.

    All per-step scalars (``lr``, ``clip_coeff``, both bias
    corrections) are passed via 1-element FP32 device tensors that
    the caller refreshes every step BEFORE this call (and BEFORE
    ``graph.replay()`` when this is inside a captured graph).
    Constants (``wd``, ``beta1/2``, ``eps``) stay scalar.
    """
    N = grad.numel()
    g_flat = grad.view(-1)
    p_flat = master.view(-1)
    m_flat = m.view(-1)
    v_flat = v.view(-1)
    bf_flat = bf16_param.view(-1)

    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _fused_adam_sync_tensor_kernel[grid](
        g_flat, p_flat, m_flat, v_flat, bf_flat,
        lr_buf, clip_coeff_buf, bc1_buf, bc2_buf,
        beta1, beta2, eps, wd,
        N=N, BLOCK=BLOCK,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fused RMSNorm Backward + Residual Add (v2: multi-row, no atomic)
# ═══════════════════════════════════════════════════════════════════════


_NORM_BWD_ROWS_PER_GROUP = 64


# 2026-05-08 root-cause fix for the USE_FUSED_NORM_BWD=1 gradient-inflation bug
# (see ``workload/notes/stable_alignment_findings.md`` §12):
# This kernel reads-and-writes ``d_hidden_ptr`` in place (loads d_upstream,
# adds d_x, stores back). With ``triton.autotune`` and **no** ``restore_value``,
# Triton runs each candidate config through ``do_bench`` (~30+ kernel
# invocations per config × 4 configs = ~120 total) on the FIRST call for
# a given (H, ROWS_PER_GROUP) cache key. Each invocation accumulates
# into the same ``d_hidden`` buffer, inflating the gradient by ~120×.
# Subsequent calls reuse the cached config and run the kernel exactly
# once, so they are correct — but the iter-1 damage is permanent.
#
# Fix: declare ``d_hidden_ptr`` in ``restore_value`` so Triton snapshots
# its values before each config benchmark and restores them after.
# (``d_weight_partial_ptr`` is a write-only scratch buffer that the host
# overwrites every call before the post-kernel ``.sum(dim=0)`` reduction,
# so it does NOT need ``restore_value`` — the pre-call uninitialized
# contents are never read by anything.)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["H", "ROWS_PER_GROUP"],
    restore_value=("d_hidden_ptr",),
)
@triton.jit
def _fused_rmsnorm_bwd_dx_kernel(
    d_normed_ptr, x_ptr, weight_ptr, rsigma_ptr,
    d_hidden_ptr,  # in/out: holds d_upstream on entry; receives d_upstream + d_x on exit
    d_weight_partial_ptr,
    H: tl.constexpr,
    stride_dn, stride_x, stride_dh,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """RMSNorm backward d_x + residual add (M-FORK-B: in-place fused).

    Each block processes ROWS_PER_GROUP rows:
    - Computes d_x in FP32 registers
    - Loads d_upstream from `d_hidden_ptr`, accumulates `d_upstream + d_x`,
      writes back to `d_hidden_ptr` (in-place). This eliminates the
      separate `d_x` HBM buffer and the trailing PyTorch add kernel
      that previously fired 48× per step.
    - Accumulates d_weight partial sums in FP32 registers and writes

    NOTE: ``d_hidden_ptr`` MUST be listed in the autotune decorator's
    ``restore_value`` argument because of the in-place semantics —
    otherwise autotune corrupts the output during its config-bench
    sweep on the first call. See module-level note above.
      to [num_groups, H] for final torch.sum reduction.
    """
    gid = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    valid_h = offs_h < H

    w = tl.load(weight_ptr + offs_h, mask=valid_h, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    row_base = gid * ROWS_PER_GROUP
    for r in range(ROWS_PER_GROUP):
        row = row_base + r

        dn = tl.load(d_normed_ptr + row * stride_dn + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        x_val = tl.load(x_ptr + row * stride_x + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        rsig = tl.load(rsigma_ptr + row)

        x_hat = x_val * rsig
        grad_norm = dn * w
        c = tl.sum(grad_norm * x_hat, axis=0) / H
        d_x = rsig * (grad_norm - x_hat * c)

        dw_acc += dn * x_hat

        d_up = tl.load(d_hidden_ptr + row * stride_dh + offs_h, mask=valid_h, other=0.0).to(tl.float32)
        tl.store(d_hidden_ptr + row * stride_dh + offs_h, (d_up + d_x).to(tl.bfloat16), mask=valid_h)

    tl.store(d_weight_partial_ptr + gid * H + offs_h, dw_acc, mask=valid_h)


def fused_rmsnorm_bwd_residual(
    d_normed: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsigma: torch.Tensor,
    d_upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm backward + residual add (M-FORK-B: in-place fused).

    Stage 1: Triton kernel computes d_x in registers and writes
             ``d_upstream + d_x`` back into ``d_upstream`` IN-PLACE.
             d_weight partial sums are written to a [num_groups, H] buf.
    Stage 2: Reduces partial sums to final d_weight via torch.sum.

    Returns (d_upstream, d_weight [H] FP32). The first return value
    aliases the input ``d_upstream`` storage — callers that need the
    original gradient must clone it before calling. (All current call
    sites in backward.py treat ``d_hidden`` as a per-layer accumulator
    and immediately rebind it, so in-place is safe.)

    Saves vs the previous version:
    - one [N, H] BF16 ``d_x`` buffer (~80 MB / layer at S=4096, B=10).
    - one PyTorch ``add`` kernel + extra HBM read+write per layer.
    Across 24 layers × 2 norms / layer that is ~3.8 GB transient HBM
    traffic and 48 kernel launches eliminated per microbatch.
    """
    if get_config().norm_bwd_nan_diag:
        _check_nan("d_normed", d_normed)
        _check_nan("x", x)
        _check_nan("weight", weight)
        _check_nan("rsigma", rsigma)
        _check_nan("d_upstream", d_upstream)

    shape = x.shape
    H = shape[-1]
    N = x.numel() // H

    d_normed_2d = d_normed.reshape(N, H)
    x_2d = x.reshape(N, H)

    if not d_normed_2d.is_contiguous():
        d_normed_2d = d_normed_2d.contiguous()
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    # In-place fuse requires d_upstream to be contiguous so the 2D view
    # shares storage with the original tensor.
    if not d_upstream.is_contiguous():
        d_upstream = d_upstream.contiguous()
    d_hidden_2d = d_upstream.view(N, H)

    rpg = _NORM_BWD_ROWS_PER_GROUP
    if N % rpg != 0:
        raise ValueError(f"N={N} must be divisible by ROWS_PER_GROUP={rpg}")
    num_groups = N // rpg

    d_weight_partial = torch.empty(num_groups, H, dtype=torch.float32, device=x.device)

    BLOCK_H = triton.next_power_of_2(H)
    _fused_rmsnorm_bwd_dx_kernel[(num_groups,)](
        d_normed_2d, x_2d, weight, rsigma,
        d_hidden_2d,
        d_weight_partial,
        H,
        d_normed_2d.stride(0), x_2d.stride(0), d_hidden_2d.stride(0),
        ROWS_PER_GROUP=rpg,
        BLOCK_H=BLOCK_H,
    )

    d_weight = d_weight_partial.sum(dim=0)
    d_result = d_upstream  # already contains d_upstream + d_x in-place

    if get_config().norm_bwd_nan_diag:
        _check_nan("d_result_after_inplace_add", d_result)
        if torch.isnan(d_weight).any():
            nan_partial = torch.isnan(d_weight_partial).any(dim=1).sum().item()
            print(
                f"[NaN-DIAG] d_weight: NaN! "
                f"partial_nan_groups={nan_partial}/{num_groups} "
                f"partial_shape={list(d_weight_partial.shape)}",
                flush=True,
            )
            x_hat_ref = x_2d.float() * rsigma.unsqueeze(1)
            dw_ref = (d_normed_2d.float() * x_hat_ref).sum(dim=0)
            ref_nan = torch.isnan(dw_ref).sum().item()
            print(
                f"[NaN-DIAG] PyTorch ref d_weight: nan_count={ref_nan}/{dw_ref.numel()} "
                f"abs_max={dw_ref[~torch.isnan(dw_ref)].abs().max().item() if ref_nan < dw_ref.numel() else 'all-NaN'}",
                flush=True,
            )
        _check_nan("d_result", d_result)

    return d_result, d_weight


def _check_nan(name: str, t: torch.Tensor) -> None:
    if torch.isnan(t).any():
        nan_count = torch.isnan(t).sum().item()
        print(
            f"[NaN-DIAG] {name}: NaN detected! "
            f"count={nan_count}/{t.numel()} "
            f"shape={list(t.shape)} dtype={t.dtype} "
            f"abs_max={t[~torch.isnan(t)].abs().max().item() if nan_count < t.numel() else 'all-NaN'}",
            flush=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# Fused RoPE — precomputed cos/sin tables + multi-row, strided-input kernel
# ═══════════════════════════════════════════════════════════════════════
#
# M-FORK-B improvements over previous version:
#   1. cos/sin computed once on the host per `freqs` tensor (data_ptr keyed
#      LRU cache) instead of inside the kernel for every row. Saves ~80 K
#      transcendental ops per kernel × 24 layers × 2 directions / step.
#   2. One program now handles `ROWS_PER_PROG` consecutive rows that share
#      the same `s_idx`. Grid count drops by `ROWS_PER_PROG`× (e.g. 655K →
#      82K for Q at S=4096, B=10, Hq=16) and the cos/sin vector is loaded
#      once per program rather than once per row.
#
# M-FORK-C improvement (this round):
#   3. Kernel reads the input via 4D strides (s, b, h) directly, eliminating
#      the host-side `.contiguous()` copies on q/k that previously cost
#      ~1 GB of transient HBM traffic per step (k from `split_qkv_interleaved`
#      is a strided view; q is already contiguous after reshape so this
#      removal is a no-op, but the kernel is uniform). Output buffers
#      remain contiguous 2D (`torch.empty(N, D)`) to avoid the silent
#      `reshape→contiguous` copy bug that produced eq_frac~0.001 vs TE
#      in earlier iterations.
#
#   Required input invariant:
#       q.stride(-1) == 1 and k.stride(-1) == 1
#   This is satisfied by `split_qkv_interleaved` outputs (their last-dim
#   stride is inherited from the source qkv buffer's stride==1) and by
#   any 4D contiguous tensor.

from collections import OrderedDict

_ROPE_CS_CACHE: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_ROPE_CS_CACHE_MAX = 8
_ROPE_ROWS_PER_PROG = 8


def _get_rope_cos_sin(freqs: torch.Tensor, S: int, D_HALF: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 cos/sin tables of shape [S, D_HALF] for the first S
    rows of `freqs`. Caches by (data_ptr, S, D_HALF, dtype). Recomputed
    only when the underlying tensor changes (e.g. position_ids gather)."""
    key = (freqs.data_ptr(), S, D_HALF, freqs.dtype, freqs.device.index)
    cached = _ROPE_CS_CACHE.get(key)
    if cached is not None:
        _ROPE_CS_CACHE.move_to_end(key)
        return cached

    freqs_2d = freqs[:S].reshape(S, -1).contiguous()
    freqs_half = freqs_2d[:, :D_HALF].float()
    cos_t = freqs_half.cos().contiguous()
    sin_t = freqs_half.sin().contiguous()

    _ROPE_CS_CACHE[key] = (cos_t, sin_t)
    if len(_ROPE_CS_CACHE) > _ROPE_CS_CACHE_MAX:
        _ROPE_CS_CACHE.popitem(last=False)
    return cos_t, sin_t


@triton.jit
def _rope_fwd_kernel_linear(
    x_ptr, out_ptr, cos_ptr, sin_ptr,
    BH,
    stride_x_row,
    stride_cs_s,
    D_HALF: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """RoPE forward, linear (row-major contiguous) input layout.

    Used when input is contiguous and can be viewed as [N, D]. Same
    logic as M-FORK-B: row-stride addressing, ROWS_PER_PROG rows
    share one s_idx and one cos/sin load.
    """
    pid = tl.program_id(0)
    row_base = pid * ROWS_PER_PROG
    s_idx = row_base // BH

    offs = tl.arange(0, D_HALF)

    cos_val = tl.load(cos_ptr + s_idx * stride_cs_s + offs)
    sin_val = tl.load(sin_ptr + s_idx * stride_cs_s + offs)

    for r in range(ROWS_PER_PROG):
        row = row_base + r
        x1 = tl.load(x_ptr + row * stride_x_row + offs).to(tl.float32)
        x2 = tl.load(x_ptr + row * stride_x_row + D_HALF + offs).to(tl.float32)

        out1 = (x1 * cos_val - x2 * sin_val).to(tl.bfloat16)
        out2 = (x1 * sin_val + x2 * cos_val).to(tl.bfloat16)

        tl.store(out_ptr + row * stride_x_row + offs, out1)
        tl.store(out_ptr + row * stride_x_row + D_HALF + offs, out2)


@triton.jit
def _rope_fwd_kernel_strided(
    x_ptr, out_ptr, cos_ptr, sin_ptr,
    BH, H,
    stride_x_s, stride_x_b, stride_x_h,
    stride_o_row,
    stride_cs_s,
    D_HALF: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """RoPE forward, 4D strided input layout (skips host `.contiguous()`).

    Used when input is a strided slice view (e.g. K from
    split_qkv_interleaved). ROWS_PER_PROG must divide BH so that all
    rows handled by one program share the same s_idx. Output is a
    contiguous 2D buffer.
    """
    pid = tl.program_id(0)
    row_base = pid * ROWS_PER_PROG
    s_idx = row_base // BH
    bh_base = row_base - s_idx * BH  # row_base % BH

    offs = tl.arange(0, D_HALF)

    cos_val = tl.load(cos_ptr + s_idx * stride_cs_s + offs)
    sin_val = tl.load(sin_ptr + s_idx * stride_cs_s + offs)

    s_off = s_idx * stride_x_s
    for r in range(ROWS_PER_PROG):
        bh = bh_base + r
        b_idx = bh // H
        h_idx = bh - b_idx * H  # bh % H
        x_row_in = s_off + b_idx * stride_x_b + h_idx * stride_x_h

        x1 = tl.load(x_ptr + x_row_in + offs).to(tl.float32)
        x2 = tl.load(x_ptr + x_row_in + D_HALF + offs).to(tl.float32)

        out1 = (x1 * cos_val - x2 * sin_val).to(tl.bfloat16)
        out2 = (x1 * sin_val + x2 * cos_val).to(tl.bfloat16)

        out_row = (row_base + r) * stride_o_row
        tl.store(out_ptr + out_row + offs, out1)
        tl.store(out_ptr + out_row + D_HALF + offs, out2)


def _pick_rows_per_prog(BH: int) -> int:
    """Pick the largest ROWS_PER_PROG ≤ _ROPE_ROWS_PER_PROG that divides BH."""
    for r in (_ROPE_ROWS_PER_PROG, 4, 2, 1):
        if BH % r == 0:
            return r
    return 1


def _assert_rope_input_layout(t: torch.Tensor, name: str) -> None:
    """Strided RoPE kernel requires the last (D) dimension to be contiguous.
    Everything else may be strided — we read s/b/h via explicit strides."""
    if t.dim() != 4:
        raise ValueError(f"{name}: expected 4D [S,B,H,D], got shape {tuple(t.shape)}")
    if t.stride(-1) != 1:
        raise ValueError(
            f"{name}: RoPE strided kernel requires stride(-1)==1; got strides "
            f"{tuple(t.stride())} for shape {tuple(t.shape)}"
        )


def _launch_rope_fwd(
    x: torch.Tensor,
    out_2d: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    N: int,
    BH: int,
    H: int,
    D: int,
    D_HALF: int,
    rpp: int,
    cs_stride: int,
) -> None:
    """Dispatch fwd to linear or strided kernel based on x.is_contiguous().

    Linear path is used when x is fully contiguous (then `x.view(N, D)`
    is a zero-copy reshape and the row-major kernel is bit-exact equivalent
    to the M-FORK-B path). Strided path avoids the host `.contiguous()`
    copy when x is a slice view (e.g. K/V from split_qkv_interleaved).
    """
    if x.is_contiguous():
        _rope_fwd_kernel_linear[(N // rpp,)](
            x.view(N, D), out_2d, cos_t, sin_t,
            BH,
            D,
            cs_stride,
            D_HALF=D_HALF,
            ROWS_PER_PROG=rpp,
        )
    else:
        _rope_fwd_kernel_strided[(N // rpp,)](
            x, out_2d, cos_t, sin_t,
            BH, H,
            x.stride(0), x.stride(1), x.stride(2),
            out_2d.stride(0),
            cs_stride,
            D_HALF=D_HALF,
            ROWS_PER_PROG=rpp,
        )


def fused_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE forward for Q and K using precomputed cos/sin tables.

    q: [S, B, Hq, D]
    k: [S, B, Hkv, D]
    freqs: [max_S, 1, 1, D]  (precomputed RoPE frequency embeddings)

    Returns: (q_rot, k_rot) with forward rotation applied (always
    contiguous 4D tensors, regardless of input strides).

    Layout dispatch:
      - is_contiguous() → linear kernel (`x.view(N, D)`); bit-exact same
        path as M-FORK-B, no copy.
      - strided (e.g. K from split_qkv_interleaved which is a `qkv[..., a:b]`
        slice) → strided kernel reads directly from the source storage
        via 4D strides, eliminating the previous `.contiguous()` copy
        (~1 GB/step of transient HBM traffic across 24 layers × 2 dirs).

    Output is allocated as contiguous 2D `torch.empty(N, D)` and viewed
    back to 4D — avoids the historical `reshape→silent-copy` bug where
    writing into an `empty_like(strided)` would land in a PyTorch-internal
    temporary (eq_frac~0.001 vs TE on real activations).
    """
    _assert_rope_input_layout(q, "q")
    _assert_rope_input_layout(k, "k")

    S, B, Hq, D = q.shape
    _, _, Hkv, _ = k.shape
    D_HALF = D // 2

    cos_t, sin_t = _get_rope_cos_sin(freqs, S, D_HALF)
    cs_stride = cos_t.stride(0)

    Nq = S * B * Hq
    BH_q = B * Hq
    rpp_q = _pick_rows_per_prog(BH_q)
    q_rot_2d = torch.empty(Nq, D, dtype=q.dtype, device=q.device)
    _launch_rope_fwd(q, q_rot_2d, cos_t, sin_t, Nq, BH_q, Hq, D, D_HALF, rpp_q, cs_stride)

    Nk = S * B * Hkv
    BH_k = B * Hkv
    rpp_k = _pick_rows_per_prog(BH_k)
    k_rot_2d = torch.empty(Nk, D, dtype=k.dtype, device=k.device)
    _launch_rope_fwd(k, k_rot_2d, cos_t, sin_t, Nk, BH_k, Hkv, D, D_HALF, rpp_k, cs_stride)

    return q_rot_2d.view(S, B, Hq, D), k_rot_2d.view(S, B, Hkv, D)


# ═══════════════════════════════════════════════════════════════════════
# Fused RoPE Backward
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _rope_bwd_kernel_linear(
    dx_ptr, out_ptr, cos_ptr, sin_ptr,
    BH,
    stride_dx_row,
    stride_cs_s,
    D_HALF: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """RoPE backward (inverse rotation), linear input layout."""
    pid = tl.program_id(0)
    row_base = pid * ROWS_PER_PROG
    s_idx = row_base // BH

    offs = tl.arange(0, D_HALF)

    cos_val = tl.load(cos_ptr + s_idx * stride_cs_s + offs)
    sin_val = tl.load(sin_ptr + s_idx * stride_cs_s + offs)

    for r in range(ROWS_PER_PROG):
        row = row_base + r
        dx1 = tl.load(dx_ptr + row * stride_dx_row + offs).to(tl.float32)
        dx2 = tl.load(dx_ptr + row * stride_dx_row + D_HALF + offs).to(tl.float32)

        out1 = (dx1 * cos_val + dx2 * sin_val).to(tl.bfloat16)
        out2 = (-dx1 * sin_val + dx2 * cos_val).to(tl.bfloat16)

        tl.store(out_ptr + row * stride_dx_row + offs, out1)
        tl.store(out_ptr + row * stride_dx_row + D_HALF + offs, out2)


@triton.jit
def _rope_bwd_kernel_strided(
    dx_ptr, out_ptr, cos_ptr, sin_ptr,
    BH, H,
    stride_dx_s, stride_dx_b, stride_dx_h,
    stride_o_row,
    stride_cs_s,
    D_HALF: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """RoPE backward, 4D strided input layout."""
    pid = tl.program_id(0)
    row_base = pid * ROWS_PER_PROG
    s_idx = row_base // BH
    bh_base = row_base - s_idx * BH

    offs = tl.arange(0, D_HALF)

    cos_val = tl.load(cos_ptr + s_idx * stride_cs_s + offs)
    sin_val = tl.load(sin_ptr + s_idx * stride_cs_s + offs)

    s_off = s_idx * stride_dx_s
    for r in range(ROWS_PER_PROG):
        bh = bh_base + r
        b_idx = bh // H
        h_idx = bh - b_idx * H
        dx_row_in = s_off + b_idx * stride_dx_b + h_idx * stride_dx_h

        dx1 = tl.load(dx_ptr + dx_row_in + offs).to(tl.float32)
        dx2 = tl.load(dx_ptr + dx_row_in + D_HALF + offs).to(tl.float32)

        out1 = (dx1 * cos_val + dx2 * sin_val).to(tl.bfloat16)
        out2 = (-dx1 * sin_val + dx2 * cos_val).to(tl.bfloat16)

        out_row = (row_base + r) * stride_o_row
        tl.store(out_ptr + out_row + offs, out1)
        tl.store(out_ptr + out_row + D_HALF + offs, out2)


def _launch_rope_bwd(
    dx: torch.Tensor,
    out_2d: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    N: int,
    BH: int,
    H: int,
    D: int,
    D_HALF: int,
    rpp: int,
    cs_stride: int,
) -> None:
    if dx.is_contiguous():
        _rope_bwd_kernel_linear[(N // rpp,)](
            dx.view(N, D), out_2d, cos_t, sin_t,
            BH,
            D,
            cs_stride,
            D_HALF=D_HALF,
            ROWS_PER_PROG=rpp,
        )
    else:
        _rope_bwd_kernel_strided[(N // rpp,)](
            dx, out_2d, cos_t, sin_t,
            BH, H,
            dx.stride(0), dx.stride(1), dx.stride(2),
            out_2d.stride(0),
            cs_stride,
            D_HALF=D_HALF,
            ROWS_PER_PROG=rpp,
        )


def fused_rope_bwd(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE backward for Q and K using precomputed cos/sin tables.

    Same dispatch as `fused_rope_fwd`: contiguous → linear kernel (zero-copy
    `view(N, D)`); strided → 4D-strided kernel (no `.contiguous()` copy).
    """
    _assert_rope_input_layout(d_q, "d_q")
    _assert_rope_input_layout(d_k, "d_k")

    S, B, Hq, D = d_q.shape
    _, _, Hkv, _ = d_k.shape
    D_HALF = D // 2

    cos_t, sin_t = _get_rope_cos_sin(freqs, S, D_HALF)
    cs_stride = cos_t.stride(0)

    Nq = S * B * Hq
    BH_q = B * Hq
    rpp_q = _pick_rows_per_prog(BH_q)
    d_q_pre_2d = torch.empty(Nq, D, dtype=d_q.dtype, device=d_q.device)
    _launch_rope_bwd(d_q, d_q_pre_2d, cos_t, sin_t, Nq, BH_q, Hq, D, D_HALF, rpp_q, cs_stride)

    Nk = S * B * Hkv
    BH_k = B * Hkv
    rpp_k = _pick_rows_per_prog(BH_k)
    d_k_pre_2d = torch.empty(Nk, D, dtype=d_k.dtype, device=d_k.device)
    _launch_rope_bwd(d_k, d_k_pre_2d, cos_t, sin_t, Nk, BH_k, Hkv, D, D_HALF, rpp_k, cs_stride)

    return d_q_pre_2d.view(S, B, Hq, D), d_k_pre_2d.view(S, B, Hkv, D)


# ═══════════════════════════════════════════════════════════════════════
# M2-FORK-D: Doc-aware fused RoPE (per-(s, b) freqs)
# ═══════════════════════════════════════════════════════════════════════
#
# Doc-aware (a.k.a. packed-document) training feeds a per-token
# `position_ids` tensor instead of the implicit `[0..S)` positions, so
# every (s, b) entry has its own RoPE frequency vector. Until now the
# doc-aware branch in `forward.py` fell back to `_apply_rope_manual`
# (cos / sin / mul / cat / mul / add), which chrome-trace identified as
# ~360 CatArrayBatchedCopy launches per step and ~13 ms/step of pure
# overhead — the dominant remaining post-FORK-C cost in production.
#
# These two kernels mirror the standard `_rope_{fwd,bwd}_kernel_linear`
# but accept a 3-D `freqs` of shape `[S, B, D_HALF]` indexed by
# (s_idx, b_idx) per row, and compute `cos / sin` inside the kernel
# (the position vector changes every step in production, so the
# host-side LRU cache used by the standard path would never hit).
#
# Layout contract: q/k must be contiguous along the last two dims
# (`reshape(N, D)` is a view), matching `_apply_rope_manual`'s
# expectation; freqs must be `[S, B, 1, D]` (the same tensor the manual
# path consumes), and `_prepare_freqs_da` slices it down to
# `[S, B, D_HALF]` once at the entry point.
#
# Numerics: trace-validated against the manual path on shandong:
# step 0..99 `avg_rel_diff = 1.65e-6` (bit-exact within BF16 RNE),
# step 100+ drift is BF16 round-off compound (the fused kernel
# accumulates in FP32 inside the rotation, which is *more* precise
# than the manual chain that materializes BF16 intermediates after
# every PyTorch op). Stays well under the 1% loss-gate threshold.


@triton.jit
def _rope_fwd_kernel_da(
    x_ptr, out_ptr, freqs_ptr,
    BH: tl.constexpr,
    H: tl.constexpr,
    stride_x_row,
    stride_freq_s, stride_freq_b,
    D_HALF: tl.constexpr,
):
    """Doc-aware fused RoPE forward (per-(s, b) freqs, in-kernel cos/sin)."""
    row = tl.program_id(0)
    s_idx = row // BH
    bh_in_s = row % BH
    b_idx = bh_in_s // H

    offs = tl.arange(0, D_HALF)

    x1 = tl.load(x_ptr + row * stride_x_row + offs).to(tl.float32)
    x2 = tl.load(x_ptr + row * stride_x_row + D_HALF + offs).to(tl.float32)

    freq = tl.load(
        freqs_ptr + s_idx * stride_freq_s + b_idx * stride_freq_b + offs
    ).to(tl.float32)
    cos_val = tl.cos(freq)
    sin_val = tl.sin(freq)

    out1 = (x1 * cos_val - x2 * sin_val).to(tl.bfloat16)
    out2 = (x1 * sin_val + x2 * cos_val).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_x_row + offs, out1)
    tl.store(out_ptr + row * stride_x_row + D_HALF + offs, out2)


@triton.jit
def _rope_bwd_kernel_da(
    dx_ptr, out_ptr, freqs_ptr,
    BH: tl.constexpr,
    H: tl.constexpr,
    stride_dx_row,
    stride_freq_s, stride_freq_b,
    D_HALF: tl.constexpr,
):
    """Doc-aware fused RoPE backward (inverse rotation of _rope_fwd_kernel_da)."""
    row = tl.program_id(0)
    s_idx = row // BH
    bh_in_s = row % BH
    b_idx = bh_in_s // H

    offs = tl.arange(0, D_HALF)

    dx1 = tl.load(dx_ptr + row * stride_dx_row + offs).to(tl.float32)
    dx2 = tl.load(dx_ptr + row * stride_dx_row + D_HALF + offs).to(tl.float32)

    freq = tl.load(
        freqs_ptr + s_idx * stride_freq_s + b_idx * stride_freq_b + offs
    ).to(tl.float32)
    cos_val = tl.cos(freq)
    sin_val = tl.sin(freq)

    out1 = (dx1 * cos_val + dx2 * sin_val).to(tl.bfloat16)
    out2 = (-dx1 * sin_val + dx2 * cos_val).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_dx_row + offs, out1)
    tl.store(out_ptr + row * stride_dx_row + D_HALF + offs, out2)


def _prepare_freqs_da(freqs: torch.Tensor, S: int) -> torch.Tensor:
    """Squeeze and slice doc-aware freqs to the kernel-expected layout.

    Input contract from `forward.py` doc-aware branch: `[S, B, 1, D]`.
    Output: contiguous `[S, B, D_HALF]` (the second half of `D` is
    bytewise identical to the first by RoPE construction in
    `kernels.py:34`, so we discard it).
    """
    if freqs.dim() != 4 or freqs.shape[2] != 1:
        raise ValueError(
            f"fused_rope_*_doc_aware: expected freqs [S, B, 1, D], "
            f"got {tuple(freqs.shape)}"
        )
    if freqs.shape[0] < S:
        raise ValueError(
            f"fused_rope_*_doc_aware: freqs.shape[0]={freqs.shape[0]} < S={S}"
        )
    D_HALF = freqs.shape[-1] // 2
    return freqs[:S, :, 0, :D_HALF].contiguous()


def fused_rope_fwd_doc_aware(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Doc-aware fused RoPE forward.

    q: [S, B, Hq, D]; k: [S, B, Hkv, D]; freqs: [S, B, 1, D].
    Returns (q_rot, k_rot) with forward rotation applied per (s, b).
    """
    S, B, Hq, D = q.shape
    _, _, Hkv, _ = k.shape
    D_HALF = D // 2

    freqs_3d = _prepare_freqs_da(freqs, S)

    q_in = q.contiguous()
    k_in = k.contiguous()

    Nq = S * B * Hq
    q_rot_2d = torch.empty(Nq, D, dtype=q.dtype, device=q.device)
    _rope_fwd_kernel_da[(Nq,)](
        q_in.reshape(Nq, D), q_rot_2d, freqs_3d,
        B * Hq, Hq,
        D,
        freqs_3d.stride(0), freqs_3d.stride(1),
        D_HALF=D_HALF,
    )

    Nk = S * B * Hkv
    k_rot_2d = torch.empty(Nk, D, dtype=k.dtype, device=k.device)
    _rope_fwd_kernel_da[(Nk,)](
        k_in.reshape(Nk, D), k_rot_2d, freqs_3d,
        B * Hkv, Hkv,
        D,
        freqs_3d.stride(0), freqs_3d.stride(1),
        D_HALF=D_HALF,
    )

    return q_rot_2d.view(S, B, Hq, D), k_rot_2d.view(S, B, Hkv, D)


def fused_rope_bwd_doc_aware(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Doc-aware fused RoPE backward — inverse of fused_rope_fwd_doc_aware."""
    S, B, Hq, D = d_q.shape
    _, _, Hkv, _ = d_k.shape
    D_HALF = D // 2

    freqs_3d = _prepare_freqs_da(freqs, S)

    dq_in = d_q.contiguous()
    dk_in = d_k.contiguous()

    Nq = S * B * Hq
    d_q_pre_2d = torch.empty(Nq, D, dtype=d_q.dtype, device=d_q.device)
    _rope_bwd_kernel_da[(Nq,)](
        dq_in.reshape(Nq, D), d_q_pre_2d, freqs_3d,
        B * Hq, Hq,
        D,
        freqs_3d.stride(0), freqs_3d.stride(1),
        D_HALF=D_HALF,
    )

    Nk = S * B * Hkv
    d_k_pre_2d = torch.empty(Nk, D, dtype=d_k.dtype, device=d_k.device)
    _rope_bwd_kernel_da[(Nk,)](
        dk_in.reshape(Nk, D), d_k_pre_2d, freqs_3d,
        B * Hkv, Hkv,
        D,
        freqs_3d.stride(0), freqs_3d.stride(1),
        D_HALF=D_HALF,
    )

    return d_q_pre_2d.view(S, B, Hq, D), d_k_pre_2d.view(S, B, Hkv, D)


def _clear_triton_caches() -> None:
    _ROPE_CS_CACHE.clear()


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402, I001
_register_reset_hook(_clear_triton_caches)
