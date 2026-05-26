"""Kernel primitives for MiniCPM4 0.5B.

Wraps cuBLAS GEMM, TransformerEngine fused kernels (RMSNorm, SwiGLU, fused
attention), and any custom Triton kernels.  GQA-aware attention with
num_kv_heads=2 requires appropriate head grouping.
"""

from __future__ import annotations

__all__ = [
    "apply_rotary_embeddings_te",
    "attention_backward_te",
    "attention_forward_te",
    "clear_te_attn_cache",
    "combine_qkv_interleaved",
    "linear",
    "linear_backward",
    "precompute_rope_freqs",
    "rmsnorm_te",
    "rmsnorm_te_backward",
    "rmsnorm_te_with_rsigma",
    "rope_backward_te",
    "split_qkv_interleaved",
    "swiglu",
    "swiglu_back",
]

import os

import torch
import transformer_engine  # noqa: F401  — registers transformer_engine_torch C extension

from . import config
from .engine_config import get_config

# ── RoPE Frequencies ─────────────────────────────────────────────────


def precompute_rope_freqs(
    max_seq_len: int,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    """Precompute RoPE frequency embeddings in FP32.

    Returns [max_seq_len, 1, 1, head_dim] matching the baseline
    RotaryEmbedding output format.
    """
    dim = config.HEAD_DIM
    inv_freq = 1.0 / (
        config.ROPE_THETA
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    seq = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(seq, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb[:, None, None, :]


# ── RMSNorm ──────────────────────────────────────────────────────────


def _te_dtype(dt: torch.dtype):
    """Convert torch dtype to TE DType enum."""
    import transformer_engine_torch as tex
    _map = {
        torch.float32: tex.DType.kFloat32,
        torch.float16: tex.DType.kFloat16,
        torch.bfloat16: tex.DType.kBFloat16,
    }
    return _map[dt]


def _rmsnorm_fwd_2d(x_2d, weight):
    """Call TE rmsnorm_fwd on a 2D [tokens, hidden] tensor.

    Returns (output, rsigma). TE rmsnorm_fwd returns a variable-length list;
    rsigma is the last element (index varies across TE versions).
    """
    import transformer_engine_torch as tex

    try:
        result = tex.rmsnorm_fwd(
            x_2d, weight, config.NORM_EPS,
            None, None, _te_dtype(x_2d.dtype), 0, False,
        )
    except TypeError:
        result = tex.rmsnorm_fwd(x_2d, weight, config.NORM_EPS, 0, False)
    return result[0], result[-1]


def rmsnorm_te(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """RMSNorm forward using TransformerEngine's fused kernel.

    Handles 2D or 3D input by reshaping to [tokens, hidden] internally.
    """
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    out_2d, _ = _rmsnorm_fwd_2d(x_2d, weight)
    return out_2d.view(shape)


def rmsnorm_te_with_rsigma(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm forward returning output and rsigma for backward."""
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    out_2d, rsigma = _rmsnorm_fwd_2d(x_2d, weight)
    return out_2d.view(shape), rsigma


def rmsnorm_te_backward(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsigma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm backward.  Returns (d_input, d_weight)."""
    import transformer_engine_torch as tex

    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    d_out_2d = d_output.reshape(-1, d_output.shape[-1])

    if rsigma is None:
        _, rsigma = _rmsnorm_fwd_2d(x_2d, weight)
    result = tex.rmsnorm_bwd(
        d_out_2d, x_2d, rsigma, weight, 0, False,
    )
    return result[0].view(shape), result[1]


# ── Linear (cuBLAS GEMM) ────────────────────────────────────────────


_padded_weight_cache: dict[int, torch.Tensor] = {}

_GEMM_PAD_TO: int = config.GEMM_PAD_TO


def _get_padded_weight(weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad weight's output dim to multiple of GEMM_PAD_TO for efficient cuBLAS tiling.

    The cached padded buffer is a *separate storage* from `weight`; we therefore
    must copy the live weight contents into the buffer on every call, otherwise
    the forward path would silently use a stale snapshot once the optimizer
    updates `weight` in place. The pad rows (orig_n:padded_n) stay zero.
    """
    orig_n = weight.shape[0]
    padded_n = (orig_n + _GEMM_PAD_TO - 1) // _GEMM_PAD_TO * _GEMM_PAD_TO
    if padded_n == orig_n:
        return weight, orig_n
    wid = id(weight)
    cached = _padded_weight_cache.get(wid)
    if (cached is None
            or cached.shape != (padded_n, weight.shape[1])
            or cached.dtype != weight.dtype
            or cached.device != weight.device):
        cached = torch.zeros(
            padded_n, weight.shape[1],
            dtype=weight.dtype, device=weight.device,
        )
        _padded_weight_cache[wid] = cached
    cached[:orig_n].copy_(weight)
    return cached, orig_n


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x @ weight.T via a single cuBLAS GEMM (2D reshape)."""
    cfg = get_config()
    shape = x.shape
    if cfg.fuse_gemm_pad and weight.shape[0] % _GEMM_PAD_TO != 0:
        w_padded, orig_n = _get_padded_weight(weight)
        out = torch.mm(x.reshape(-1, shape[-1]), w_padded.t())
        if len(shape) <= 2:
            return out[:, :orig_n]
        row_stride = out.stride(0)
        return torch.as_strided(
            out, (*shape[:-1], orig_n),
            (row_stride * shape[1], row_stride, 1),
            storage_offset=0,
        )
    out = torch.mm(x.reshape(-1, shape[-1]), weight.t())
    return out.view(*shape[:-1], -1)


_te_gemm_workspace: torch.Tensor | None = None


def _get_te_gemm_workspace(device: torch.device) -> torch.Tensor:
    global _te_gemm_workspace
    if _te_gemm_workspace is None or _te_gemm_workspace.device != device:
        _te_gemm_workspace = torch.zeros(
            33_554_432, dtype=torch.int8, device=device,
        )
    return _te_gemm_workspace


def linear_backward(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear backward.  d_weight computed in FP32.

    te_wgrad: use TE general_gemm with split accumulator for wgrad,
    matching TE's TEColumnParallelLinear / TERowParallelLinear backward.
    """
    cfg = get_config()
    d_out_2d = d_output.reshape(-1, d_output.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    orig_n = weight.shape[0]
    _weight_needs_pad = cfg.fuse_gemm_pad and orig_n % _GEMM_PAD_TO != 0

    d_out_padded = None
    if _weight_needs_pad:
        w_padded, _ = _get_padded_weight(weight)
        pad_n = w_padded.shape[0]
        if d_out_2d.shape[-1] == pad_n:
            d_out_padded = d_out_2d
        else:
            d_out_padded = torch.zeros(
                d_out_2d.shape[0], pad_n,
                dtype=d_out_2d.dtype, device=d_out_2d.device,
            )
            d_out_padded[:, :d_out_2d.shape[-1]] = d_out_2d
        d_input = torch.mm(d_out_padded, w_padded).view_as(x)
    else:
        d_input = torch.mm(d_out_2d, weight).view_as(x)

    if te_wgrad:
        from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

        wgrad_d_out = d_out_padded if d_out_padded is not None else d_out_2d
        result = general_gemm(
            x_2d,
            wgrad_d_out,
            _get_te_gemm_workspace(x.device),
            layout="NT",
            grad=True,
            out_dtype=torch.float32,
            use_split_accumulator=True,
            accumulate=False,
        )
        d_weight = result[0]
        if d_out_padded is not None:
            d_weight = d_weight[:orig_n]
    else:
        if _weight_needs_pad:
            d_weight = torch.mm(d_out_padded.t().float(), x_2d.float())[:orig_n]
        else:
            d_weight = torch.mm(d_out_2d.t().float(), x_2d.float())
    return d_input, d_weight


# ── QKV Split / Combine (GQA interleaved) ───────────────────────────


def split_qkv_interleaved(
    qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split interleaved QKV for GQA.

    Layout per group: [Q_heads..., K_head, V_head]
      Group 0: Q[0:8]*128 + K[0]*128 + V[0]*128 = 1280
      Group 1: Q[8:16]*128 + K[1]*128 + V[1]*128 = 1280
    Total = 2560

    qkv: [S, B, 2560]
    Returns: Q [S,B,Hq,D], K [S,B,Hkv,D], V [S,B,Hkv,D]
    """
    S, B, _ = qkv.shape
    nkv = config.NUM_KV_HEADS
    nq_per_kv = config.NUM_HEADS // nkv
    d = config.HEAD_DIM
    group_w = (nq_per_kv + 2) * d

    qkv = qkv.view(S, B, nkv, group_w)
    q_w = nq_per_kv * d
    q = qkv[..., :q_w].reshape(S, B, config.NUM_HEADS, d)
    k = qkv[..., q_w : q_w + d]
    v = qkv[..., q_w + d :]
    return q, k, v


def combine_qkv_interleaved(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    d_v: torch.Tensor,
) -> torch.Tensor:
    """Inverse of split_qkv_interleaved."""
    S, B = d_q.shape[:2]
    nkv = config.NUM_KV_HEADS
    nq_per_kv = config.NUM_HEADS // nkv
    d = config.HEAD_DIM
    d_q_grouped = d_q.reshape(S, B, nkv, nq_per_kv * d)
    return torch.cat([d_q_grouped, d_k, d_v], dim=-1).reshape(S, B, -1)


# ── RoPE Application ────────────────────────────────────────────────


def apply_rotary_embeddings_te(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE via TE fused kernel (TE_ROPE=1) or manual.

    q: [S,B,Hq,D], k: [S,B,Hkv,D], freqs: [S,1,1,D]
    """
    cfg = get_config()
    if cfg.te_rope:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

        q_rot = apply_rotary_pos_emb(q, freqs, tensor_format="sbhd", fused=True)
        k_rot = apply_rotary_pos_emb(k, freqs, tensor_format="sbhd", fused=True)
        return q_rot, k_rot

    return _apply_rope_manual(q, freqs), _apply_rope_manual(k, freqs)


def _apply_rope_manual(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    x1 = t[..., : t.shape[-1] // 2]
    x2 = t[..., t.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return t * cos_ + rotated * sin_


def rope_backward_te(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE backward (inverse rotation).

    freqs may be precomputed for MAX_SEQ_LENGTH; slice to actual seq_len.
    """
    seq_len = d_q.shape[0]
    freqs_sliced = freqs[:seq_len]
    cos_ = torch.cos(freqs_sliced).to(d_q.dtype)
    sin_ = torch.sin(freqs_sliced).to(d_q.dtype)

    def _inv_rotate(dx: torch.Tensor) -> torch.Tensor:
        x1 = dx[..., : dx.shape[-1] // 2]
        x2 = dx[..., dx.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return dx * cos_ + rotated * (-sin_)

    return _inv_rotate(d_q), _inv_rotate(d_k)


# ── Fused Attention (TransformerEngine) ──────────────────────────────

_te_attn_modules: dict = {}


def clear_te_attn_cache() -> None:
    """Clear cached TE attention modules (useful between test phases)."""
    _te_attn_modules.clear()


def _get_te_attn(device: torch.device | str, layer_number: int = 1):
    """Lazily create TE DotProductAttention for the given device/layer."""
    key = (str(device), layer_number)
    if key not in _te_attn_modules:
        from transformer_engine.pytorch import DotProductAttention

        mod = DotProductAttention(
            num_attention_heads=config.NUM_HEADS,
            kv_channels=config.HEAD_DIM,
            num_gqa_groups=config.NUM_KV_HEADS,
            attention_dropout=0.0,
            attn_mask_type="causal",
            sequence_parallel=False,
            tp_size=1,
            layer_number=layer_number,
        )
        mod = mod.to(device)
        mod.eval()
        _te_attn_modules[key] = mod
    return _te_attn_modules[key]


def attention_forward_te(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    layer_number: int = 1,
    need_backward: bool = False,
) -> tuple[torch.Tensor, object]:
    """Fused attention via TE DotProductAttention.

    q: [S,B,Hq,D]  k: [S,B,Hkv,D]  v: [S,B,Hkv,D]
    Returns (output [S,B,Hq*D], aux_ctx_for_backward)
    """
    from .op_dispatcher import get_op_version

    cfg = get_config()
    if get_op_version("attention") == "baseline":
        if cfg.fuse_direct_attn and need_backward:
            return _attention_forward_direct(q, k, v, layer_number)

        attn_mod = _get_te_attn(q.device, layer_number)

        if need_backward:
            q_g = q.detach().requires_grad_(True)
            k_g = k.detach().requires_grad_(True)
            v_g = v.detach().requires_grad_(True)
            attn_mod.train()
            output = attn_mod(q_g, k_g, v_g, attention_mask=None)
            if output.ndim == 4:
                output = output.reshape(*output.shape[:2], -1)
            aux_ctx = (q_g, k_g, v_g, output)
            return output.detach(), aux_ctx

        output = attn_mod(q, k, v, attention_mask=None)
        if output.ndim == 4:
            output = output.reshape(*output.shape[:2], -1)
        return output, None

    from training_engine_tensor.ops.attention.kernel import attention_fwd as _attn_fwd_opt
    return _attn_fwd_opt(
        q, k, v,
        causal=causal,
        layer_number=layer_number,
        need_backward=need_backward,
    )


_direct_attn_backend = None


def _get_direct_attn_backend():
    """Lazily determine the TE fused attention backend."""
    global _direct_attn_backend
    if _direct_attn_backend is None:
        import transformer_engine_torch as tex
        _direct_attn_backend = tex.get_fused_attn_backend(
            tex.DType.kBFloat16, tex.DType.kBFloat16,
            tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD,
            tex.NVTE_Bias_Type.NVTE_NO_BIAS,
            tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
            0.0, config.NUM_HEADS, config.NUM_KV_HEADS,
            config.MAX_SEQ_LENGTH, config.MAX_SEQ_LENGTH,
            config.HEAD_DIM, config.HEAD_DIM, -1, -1,
        )
    return _direct_attn_backend


_cu_seqlens_cache: dict = {}


def _get_cu_seqlens(B: int, S: int, device: torch.device) -> torch.Tensor:
    key = (B, S, device)
    if key not in _cu_seqlens_cache:
        _cu_seqlens_cache[key] = torch.arange(
            0, (B + 1) * S, S, dtype=torch.int32, device=device,
        )
    return _cu_seqlens_cache[key]


def _attention_forward_direct(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, _layer_number: int,
) -> tuple[torch.Tensor, object]:
    """Direct TE fused attention forward in SBHD layout, bypassing autograd.

    NOTE: k and v are typically non-contiguous slice views of the QKV tensor
    (from `split_qkv_interleaved`). `fused_attn_fwd` requires contiguous
    inputs for the SBHD_SBHD_SBHD layout; passing slice views may silently
    read wrong memory. We force contiguous copies for safety - same defense
    that was required for `fused_rope_fwd`.
    """
    from transformer_engine.pytorch.cpp_extensions.fused_attn import fused_attn_fwd

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    S, B = q.shape[0], q.shape[1]
    backend = _get_direct_attn_backend()
    cu_seqlens = _get_cu_seqlens(B, S, q.device)

    out, aux_tensors = fused_attn_fwd(
        is_training=True, max_seqlen_q=S, max_seqlen_kv=S,
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        q=q, k=k, v=v,
        fake_dtype=torch.bfloat16, fused_attention_backend=backend,
        attn_scale=1.0 / (config.HEAD_DIM ** 0.5), dropout=0.0,
        qkv_layout="sbhd_sbhd_sbhd",
        attn_bias_type="no_bias", attn_mask_type="causal",
    )

    out_flat = out.reshape(S, B, -1)
    aux_ctx = ("direct", q, k, v, out, aux_tensors, cu_seqlens)
    return out_flat, aux_ctx


def attention_backward_te(
    d_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    aux_ctx: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused attention backward via TE.

    Returns (d_q, d_k, d_v) in original head-split shapes.
    """
    from .op_dispatcher import get_op_version

    if aux_ctx is None:
        raise RuntimeError("aux_ctx is None — forward not called with need_backward=True")

    if get_op_version("attention") == "baseline":
        bwd_override = os.environ.get("ATTN_BWD_OVERRIDE", "")
        if bwd_override == "flash_dsl" and isinstance(aux_ctx, tuple) and len(aux_ctx) > 0 and aux_ctx[0] == "direct":
            return _attention_backward_flash_dsl_override(d_output, aux_ctx)

        if isinstance(aux_ctx, tuple) and len(aux_ctx) > 0 and aux_ctx[0] == "direct":
            return _attention_backward_direct(d_output, aux_ctx)

        cfg = get_config()
        q_g, k_g, v_g, fwd_output = aux_ctx
        d_out = d_output
        if fwd_output.shape != d_out.shape:
            d_out = d_out.reshape_as(fwd_output)
        fwd_output.backward(d_out, retain_graph=False)
        if cfg.fuse_direct_attn:
            dq = q_g.grad
            dk = k_g.grad
            dv = v_g.grad
        else:
            dq = q_g.grad.clone()
            dk = k_g.grad.clone()
            dv = v_g.grad.clone()
        q_g.grad = k_g.grad = v_g.grad = None
        return dq, dk, dv

    from training_engine_tensor.ops.attention.kernel import attention_bwd as _attn_bwd_opt
    return _attn_bwd_opt(d_output, q, k, v, output, aux_ctx)


def _attention_backward_flash_dsl_override(
    d_output: torch.Tensor, aux_ctx: tuple,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Override baseline backward with flash_dsl CuTe DSL backward.

    Uses the same TE forward output (SBHD layout) but routes the backward
    through the optimized 3-kernel CuTe DSL pipeline. Extracts softmax LSE
    from TE's aux_tensors (first element, shape [B, H, S, 1] or [B, H, S]).
    """
    import math
    from training_engine_tensor.ops.attention.flash_attn_dsl.flash_bwd import run_flash_bwd_dsl

    _, q, k, v, out, aux_tensors, cu_seqlens = aux_ctx
    S, B, Hq, D = q.shape
    Hkv = k.shape[2]
    scale = 1.0 / math.sqrt(D)

    # TE's aux_tensors[0] is softmax_lse, typically [B, H, S, 1] or [B, H, S]
    lse_raw = aux_tensors[0]
    if lse_raw.ndim == 4:
        lse = lse_raw.squeeze(-1)  # [B, H, S]
    elif lse_raw.ndim == 3:
        lse = lse_raw
    else:
        raise RuntimeError(f"Unexpected LSE shape from TE: {lse_raw.shape}")
    lse = lse.contiguous().float()

    # SBHD [S,B,H,D] → BHND [B,H,S,D]: strided views only, no memcpy.
    # CuTe TMA handles non-contiguous outer strides (all ≡0 mod 8 elems
    # for BF16) via mark_layout_dynamic; see flash_bwd.py _to_cute_tensor4.
    q_bhnd = q.permute(1, 2, 0, 3)
    k_bhnd = k.permute(1, 2, 0, 3)
    v_bhnd = v.permute(1, 2, 0, 3)
    out_bhnd = out.permute(1, 2, 0, 3)
    d_out_bhnd = d_output.reshape(S, B, Hq, D).permute(1, 2, 0, 3)

    # Allocate outputs directly in SBHD; kernel writes via strided BHND views.
    dq = torch.empty(S, B, Hq, D, dtype=q.dtype, device=q.device)
    dk = torch.empty(S, B, Hkv, D, dtype=q.dtype, device=q.device)
    dv = torch.empty(S, B, Hkv, D, dtype=q.dtype, device=q.device)

    run_flash_bwd_dsl(
        q_bhnd, k_bhnd, v_bhnd, out_bhnd, d_out_bhnd, lse,
        dq.permute(1, 2, 0, 3), dk.permute(1, 2, 0, 3), dv.permute(1, 2, 0, 3),
        softmax_scale=scale,
        is_causal=True,
    )

    return dq, dk, dv


def _attention_backward_direct(
    d_output: torch.Tensor, aux_ctx: tuple,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct TE fused attention backward in SBHD layout, bypassing autograd."""
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.fused_attn import fused_attn_bwd

    _, q, k, v, out, aux_tensors, cu_seqlens = aux_ctx
    S, B = d_output.shape[0], d_output.shape[1]
    backend = _get_direct_attn_backend()

    d_out_4d = d_output.reshape(S, B, config.NUM_HEADS, config.HEAD_DIM)
    aux_list = [t for t in aux_tensors if t is not None]

    result = fused_attn_bwd(
        max_seqlen_q=S, max_seqlen_kv=S,
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        q=q, k=k, v=v, o=out, d_o=d_out_4d,
        fake_dtype=torch.bfloat16,
        dqkv_dtype=tex.DType.kBFloat16,
        aux_ctx_tensors=aux_list,
        fused_attention_backend=backend,
        attn_scale=1.0 / (config.HEAD_DIM ** 0.5), dropout=0.0,
        qkv_layout="sbhd_sbhd_sbhd",
        attn_bias_type="no_bias", attn_mask_type="causal",
    )

    return result[0], result[1], result[2]


# ── SwiGLU Activation ────────────────────────────────────────────────


@torch.compile
def _swiglu_fused(y: torch.Tensor) -> torch.Tensor:
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.ops.aten.silu(y_1) * y_2


@torch.compile
def _swiglu_back_compiled(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SwiGLU backward matching baseline's fused_bias_swiglu.swiglu_back.

    Sigmoid computed in FP32 then cast to input dtype.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (
            g
            * torch.sigmoid(y_1.float()).to(y_1.dtype)
            * (1 + y_1 * (1 - torch.sigmoid(y_1.float()).to(y_1.dtype)))
            * y_2,
            g * torch.ops.aten.silu(y_1),
        ),
        -1,
    )


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(gate) * up.

    x: [..., 2*ffn_hidden_size]
    Returns: [..., ffn_hidden_size]
    """
    cfg = get_config()
    if cfg.fuse_swiglu:
        from .triton_kernels import fused_swiglu_fwd
        return fused_swiglu_fwd(x)
    return _swiglu_fused(x)


def swiglu_back(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SwiGLU backward. Dispatches to Triton fused kernel or torch.compile."""
    cfg = get_config()
    if cfg.fuse_swiglu:
        from .triton_kernels import fused_swiglu_bwd
        return fused_swiglu_bwd(g, y)
    return _swiglu_back_compiled(g, y)


def _clear_kernels_cache() -> None:
    global _te_gemm_workspace, _direct_attn_backend
    _padded_weight_cache.clear()
    _te_gemm_workspace = None
    _te_attn_modules.clear()
    _direct_attn_backend = None
    _cu_seqlens_cache.clear()


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402, I001
_register_reset_hook(_clear_kernels_cache)
