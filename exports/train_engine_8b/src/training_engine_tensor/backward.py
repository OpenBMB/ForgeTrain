"""Backward pass utilities for M2+.

Provides:
  - CE loss + manual gradient matching Megatron's VocabParallelCrossEntropy
  - Vocab-parallel CE loss for TP>1
  - Fully manual backward pass (no autograd graph for d_hidden propagation)
  - Parameter-name mapping from our checkpoint keys to Megatron named_parameters

TP backward communication (column-parallel dgrad requires all-reduce):
  - output_layer dgrad: all-reduce across TP
  - QKV dgrad: all-reduce across TP
  - FC1 dgrad: all-reduce across TP
  - Proj/FC2 dgrad: NO all-reduce (row-parallel backward)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

__all__ = [
    "compute_ce_loss_forward_only", "compute_ce_loss_and_grad",
    "compute_ce_loss_and_grad_tp", "fused_ce_loss_and_grad_tp",
    "manual_backward", "has_fused_wgrad", "to_canonical_grad_name",
]
import torch.distributed as dist

from . import config
from .engine_config import get_config
from .parameters import build_bwd_precomputed
from .custom_gemm import (
    custom_gemm_attn_out_proj_dgrad,
    custom_gemm_attn_out_proj_wgrad,
    custom_gemm_fc1_dgrad,
    custom_gemm_fc1_wgrad,
    custom_gemm_fc2_dgrad,
    custom_gemm_fc2_wgrad,
    custom_gemm_output_dgrad,
    custom_gemm_output_wgrad,
    custom_gemm_qkv_proj_dgrad,
    custom_gemm_qkv_proj_wgrad,
)
from .kernels import (
    _te_rmsnorm_backward,
    _triton_available,
    _rotate_half,
    causal_attention,
    fused_ce_max_sum_pred,
    fused_ce_correction,
    fused_ce_bwd,
    fused_residual_add_rmsnorm_bwd_add,
    fused_residual_add_rmsnorm_bwd_add_dw_reduce,
    fused_rmsnorm_bwd_add,
    fused_rmsnorm_bwd_add_dw_reduce,
    fused_rope_backward,
    fused_rope_backward_pack,
    rmsnorm_forward,
    swiglu,
    swiglu_backward,
)

# Try importing Megatron's fused wgrad GEMM for BF16×BF16→FP32 accumulation.
_wgrad_gemm_fn = None
try:
    import fused_weight_gradient_mlp_cuda as _fwg
    _wgrad_gemm_fn = _fwg.wgrad_gemm_accum_fp32
except ImportError:
    pass


def has_fused_wgrad() -> bool:
    """Whether the fused BF16×BF16→FP32 wgrad kernel is available."""
    return _wgrad_gemm_fn is not None


def _wgrad(d_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute weight gradient: d_W = d_out^T @ x  in FP32.

    d_out: [*, O] BF16  —  upstream gradient
    x:     [*, I] BF16  —  input from forward

    Uses fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32 when available
    (same kernel as Megatron). Falls back to FP32 matmul.
    """
    d_2d = d_out.reshape(-1, d_out.shape[-1])   # [T, O]
    x_2d = x.reshape(-1, x.shape[-1])           # [T, I]
    if _wgrad_gemm_fn is not None:
        grad_w = torch.zeros(
            d_2d.shape[1], x_2d.shape[1],
            dtype=torch.float32, device=d_2d.device,
        )
        _wgrad_gemm_fn(x_2d, d_2d, grad_w)
        return grad_w
    return torch.matmul(d_2d.t().float(), x_2d.float())


def _wgrad_into(d_out: torch.Tensor, x: torch.Tensor, out_flat: torch.Tensor):
    """Write wgrad directly into a pre-allocated buffer slice (1D FP32).

    The buffer slice must be pre-zeroed by the caller.
    """
    d_2d = d_out.reshape(-1, d_out.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = out_flat.view(d_2d.shape[1], x_2d.shape[1])
    if _wgrad_gemm_fn is not None:
        _wgrad_gemm_fn(x_2d, d_2d, out_2d)
    else:
        torch.matmul(d_2d.t().float(), x_2d.float(), out=out_2d)


# ---------------------------------------------------------------------------
# CE loss forward-only (SSOT for per-token CE; used by entry.run_forward)
# ---------------------------------------------------------------------------
def compute_ce_loss_forward_only(
    logits_bf16: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    """Compute scalar CE loss and per-token losses (no gradient).

    Same math as compute_ce_loss_and_grad but skips the backward part.

    Args:
        logits_bf16: [S, B, V] BF16
        labels:      [B, S] long
        loss_mask:   [B, S]

    Returns:
        loss_val:      float scalar
        per_token_bs:  [B, S] FP32 per-token CE losses
    """
    logits_fp32 = logits_bf16.float()
    labels_sb = labels.transpose(0, 1).contiguous()

    logits_max = logits_fp32.max(dim=-1)[0]
    logits_shifted = logits_fp32 - logits_max.unsqueeze(-1)
    exp_logits = logits_shifted.exp()
    sum_exp = exp_logits.sum(dim=-1)

    predicted = logits_shifted.gather(-1, labels_sb.unsqueeze(-1)).squeeze(-1)
    per_token_sb = torch.log(sum_exp) - predicted
    per_token_bs = per_token_sb.transpose(0, 1).contiguous()

    flat_losses = per_token_bs.view(-1)
    flat_mask = loss_mask.view(-1).float()
    loss_val = (flat_losses * flat_mask).sum() / flat_mask.sum()

    return loss_val.item(), per_token_bs


# ---------------------------------------------------------------------------
# CE loss forward + manual backward 
# ---------------------------------------------------------------------------
def compute_ce_loss_and_grad(
    logits_bf16: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    grad_scale: float = 1.0,
) -> Tuple[float, torch.Tensor]:
    """Compute scalar CE loss and d_logits matching baseline exactly.

    Args:
        logits_bf16: [S, B, V] BF16
        labels:      [B, S] long
        loss_mask:   [B, S]

    Returns:
        loss_val:      float scalar
        d_logits_bf16: [S, B, V] BF16
    """
    logits_fp32 = logits_bf16.float()
    labels_sb = labels.transpose(0, 1).contiguous()  # [S, B]

    logits_max = logits_fp32.max(dim=-1)[0]
    logits_shifted = logits_fp32 - logits_max.unsqueeze(-1)
    exp_logits = logits_shifted.exp()
    sum_exp = exp_logits.sum(dim=-1)
    softmax = exp_logits / sum_exp.unsqueeze(-1)

    predicted = logits_shifted.gather(-1, labels_sb.unsqueeze(-1)).squeeze(-1)
    per_token_sb = torch.log(sum_exp) - predicted
    per_token_bs = per_token_sb.transpose(0, 1).contiguous()

    flat_losses = per_token_bs.view(-1)
    flat_mask = loss_mask.view(-1).float()
    mask_sum = flat_mask.sum()
    loss_val = (flat_losses * flat_mask).sum() / mask_sum

    d_per_token_bs = loss_mask.float() / mask_sum
    if grad_scale != 1.0:
        d_per_token_bs = d_per_token_bs * grad_scale
    d_per_token_sb = d_per_token_bs.transpose(0, 1).contiguous()

    one_hot = torch.zeros_like(softmax)
    one_hot.scatter_(-1, labels_sb.unsqueeze(-1), 1.0)
    d_logits_fp32 = (softmax - one_hot) * d_per_token_sb.unsqueeze(-1)

    d_logits_bf16 = d_logits_fp32.to(torch.bfloat16)
    return loss_val, d_logits_bf16


def compute_ce_loss_and_grad_tp(
    logits_bf16: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    tp_group,
    tp_rank: int,
    tp_size: int,
    *,
    grad_scale: float = 1.0,
) -> Tuple[float, torch.Tensor]:
    """Vocab-parallel CE loss matching Megatron's VocabParallelCrossEntropy.

    Each TP rank holds logits for its vocab shard. Global max and sum are
    computed via all-reduce across the TP group.

    Args:
        logits_bf16: [S, B, V_local] BF16 (V_local = V / tp_size)
        labels:      [B, S] long
        loss_mask:   [B, S]
        tp_group:    TP process group
        tp_rank:     rank within TP group
        tp_size:     TP world size

    Returns:
        loss_val:      float scalar
        d_logits_bf16: [S, B, V_local] BF16
    """
    V_local = logits_bf16.shape[-1]
    vocab_start = tp_rank * V_local

    logits_fp32 = logits_bf16.float()
    labels_sb = labels.transpose(0, 1).contiguous()  # [S, B]

    # Global max for numerical stability
    local_max = logits_fp32.max(dim=-1)[0]  # [S, B]
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)

    logits_shifted = logits_fp32 - local_max.unsqueeze(-1)
    exp_logits = logits_shifted.exp()

    # Global sum of exp
    local_sum = exp_logits.sum(dim=-1)  # [S, B]
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=tp_group)

    # Predicted logit: only the rank owning the target token contributes
    target_local = labels_sb - vocab_start
    in_range = (target_local >= 0) & (target_local < V_local)
    target_clamped = target_local.clamp(0, V_local - 1)
    predicted_local = logits_shifted.gather(-1, target_clamped.unsqueeze(-1)).squeeze(-1)
    predicted_local = predicted_local * in_range.float()
    dist.all_reduce(predicted_local, op=dist.ReduceOp.SUM, group=tp_group)

    per_token_sb = torch.log(local_sum) - predicted_local
    per_token_bs = per_token_sb.transpose(0, 1).contiguous()

    flat_losses = per_token_bs.view(-1)
    flat_mask = loss_mask.view(-1).float()
    mask_sum = flat_mask.sum()
    loss_val = (flat_losses * flat_mask).sum() / mask_sum

    # Backward: local softmax - local one_hot, scaled by upstream gradient
    d_per_token_bs = loss_mask.float() / mask_sum
    if grad_scale != 1.0:
        d_per_token_bs = d_per_token_bs * grad_scale
    d_per_token_sb = d_per_token_bs.transpose(0, 1).contiguous()

    softmax_local = exp_logits / local_sum.unsqueeze(-1)
    one_hot_local = torch.zeros_like(softmax_local)
    one_hot_local.scatter_(
        -1, target_clamped.unsqueeze(-1), in_range.float().unsqueeze(-1),
    )

    d_logits_fp32 = (softmax_local - one_hot_local) * d_per_token_sb.unsqueeze(-1)
    d_logits_bf16 = d_logits_fp32.to(torch.bfloat16)

    return loss_val, d_logits_bf16


def fused_ce_loss_and_grad_tp(
    logits_bf16: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    tp_group,
    tp_rank: int,
    tp_size: int,
    *,
    grad_scale: float = 1.0,
) -> Tuple[float, torch.Tensor]:
    """Vocab-parallel CE loss using fused Triton kernels.

    Same semantics as compute_ce_loss_and_grad_tp but replaces PyTorch ops
    with 3 Triton kernels + 2 NCCL all-reduces:
      1. online-softmax (local_max + local_sum + predicted_raw, 1 logits read)
      2. correction (local→global basis shift + predicted masking, 1 kernel)
      3. backward (d_logits, 1 logits read)
    """
    S, B, V_local = logits_bf16.shape
    T = S * B
    vocab_start = tp_rank * V_local

    labels_sb = labels.transpose(0, 1).contiguous()  # [S, B]
    labels_flat = labels_sb.reshape(T)

    # Fused kernel A+B: single-pass local_max + local_sum + predicted_raw
    logits_2d = logits_bf16.reshape(T, V_local)
    local_max, local_sum, predicted_raw = fused_ce_max_sum_pred(
        logits_2d, labels_flat, vocab_start,
    )

    # Global max across TP (local_max updated in-place).
    local_max_saved = local_max.clone()
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)

    # Fused correction: local_sum basis shift + predicted masking → ar_buf [2,T]
    ar_buf = fused_ce_correction(
        local_max_saved, local_max, local_sum, predicted_raw,
        labels_flat, vocab_start, V_local,
    )

    # Batch sum_exp and predicted into one all_reduce across TP.
    dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=tp_group)
    global_sum = ar_buf[0]      # [T]
    global_predicted = ar_buf[1]  # [T]

    # Loss computation
    per_token_sb = torch.log(global_sum) - global_predicted  # [T] in SB order
    per_token_bs = per_token_sb.view(S, B).transpose(0, 1).contiguous()
    flat_losses = per_token_bs.view(-1)
    flat_mask = loss_mask.view(-1).float()
    mask_sum = flat_mask.sum()
    loss_val = (flat_losses * flat_mask).sum() / mask_sum

    # Upstream gradient for backward kernel
    d_per_token_bs = loss_mask.float() / mask_sum
    if grad_scale != 1.0:
        d_per_token_bs = d_per_token_bs * grad_scale
    upstream = d_per_token_bs.transpose(0, 1).contiguous().reshape(T)  # [T] FP32

    # d_logits (uses the global-max stored in local_max).
    d_logits_bf16 = fused_ce_bwd(
        logits_2d, local_max, global_sum, labels_flat, upstream, vocab_start,
    )  # [T, V_local] → [S, B, V_local]

    return loss_val, d_logits_bf16.view(S, B, V_local)


# ---------------------------------------------------------------------------
# Manual backward pass (no autograd for d_hidden propagation)
# ---------------------------------------------------------------------------
def manual_backward(
    params: Dict[str, torch.Tensor],
    saved: dict,
    d_logits: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    tp_group=None,
    tp_rank: int = 0,
    tp_size: int = 1,
    grad_buffer=None,
    layout=None,
    wgrad_stream_ext=None,
    bucket_ready_fn=None,
    layers_per_bucket: int = 0,
    wgrad_events=None,
    defer_embedding_bwd: bool = False,
    _precomputed=None,
    _wgrad_sync_event=None,
) -> Dict[str, torch.Tensor]:
    """Compute per-parameter FP32 gradients with fully manual backward.

    All d_hidden propagation is explicit (no autograd graph).
    Attention backward uses saved autograd graph when available (from forward
    pass with save_for_backward=True), otherwise re-runs DPA forward.
    Weight gradients are FP32 (matching Megatron's wgrad_gemm_accum_fp32).

    When grad_buffer and layout are provided, wgrads are written directly into
    the buffer (buffer must be pre-zeroed). Returns None in this mode.
    Otherwise returns a dict of per-parameter FP32 gradients.
    """
    cfg = get_config()
    num_layers = config.NUM_LAYERS
    num_heads = config.NUM_HEADS // tp_size
    num_kv_heads = config.NUM_KV_HEADS // tp_size
    head_dim = config.HEAD_DIM
    heads_per_group = num_heads // num_kv_heads
    eps = config.NORM_EPSILON

    use_buffer = grad_buffer is not None
    fp32_grads: Dict[str, torch.Tensor] = {} if not use_buffer else None

    if use_buffer:
        main_stream = torch.cuda.current_stream()
        wgrad_stream = wgrad_stream_ext or torch.cuda.Stream()
    else:
        main_stream = wgrad_stream = None

    _ws_event = _wgrad_sync_event

    def _wgrad_wait_main():
        """Make wgrad_stream wait for main_stream's latest work.

        Uses a pre-allocated CUDA event when available, avoiding the
        per-call cudaEventCreate+Destroy overhead of wait_stream().
        """
        if _ws_event is not None:
            _ws_event.record()
            wgrad_stream.wait_event(_ws_event)
        else:
            wgrad_stream.wait_stream(main_stream)

    def _store_wgrad_dict(name, d_out, x):
        fp32_grads[name] = _wgrad(d_out, x)

    def _store_ln_grad_dict(name, dw):
        fp32_grads[name] = dw.float()

    # --- Pre-compute per-layer data to avoid string ops in hot loop ---
    # When _precomputed is provided (from the training loop), skip all the
    # per-call dict lookups, f-string formatting, and tensor view creation.
    if _precomputed is not None:
        _layer_params = _precomputed['layer_params']
        _layer_wbufs = _precomputed.get('layer_wbufs')
        _layer_ln_bufs = _precomputed.get('layer_ln_bufs')
        _layer_names = _precomputed['layer_names']
    else:
        _fallback = build_bwd_precomputed(
            params, num_layers,
            grad_buffer=grad_buffer if use_buffer else None,
            layout=layout if use_buffer else None,
        )
        _layer_params = _fallback['layer_params']
        _layer_wbufs = _fallback.get('layer_wbufs')
        _layer_ln_bufs = _fallback.get('layer_ln_bufs')
        _layer_names = _fallback['layer_names']

    use_tp = tp_size > 1
    q_per_group = heads_per_group * head_dim
    qkv_dim = num_kv_heads * (q_per_group + 2 * head_dim)

    # --- Output layer backward (column-parallel) ---
    hidden_final = saved["hidden_final"]
    output_w = params["output_layer.weight"]

    _d_logits_2d = d_logits.reshape(-1, d_logits.shape[-1])
    _ret_dhidden_2d = custom_gemm_output_dgrad(_d_logits_2d, output_w)
    if _ret_dhidden_2d is not None:
        d_hidden = _ret_dhidden_2d.view(*d_logits.shape[:-1], output_w.shape[-1])
    else:
        d_hidden = torch.matmul(d_logits, output_w)
    if use_tp:
        ar_handle = dist.all_reduce(d_hidden, group=tp_group, async_op=True)
    if use_buffer:
        d_logits.record_stream(wgrad_stream)
        _wgrad_wait_main()
        with torch.cuda.stream(wgrad_stream):
            _out_n = "output_layer.weight"
            _out_wbuf = grad_buffer[layout.param_offsets[_out_n]:layout.param_offsets[_out_n]+layout.param_numels[_out_n]].view(output_w.shape)
            _ret = custom_gemm_output_wgrad(d_logits, hidden_final, out_buf=_out_wbuf)
            if _ret is None:
                _wgrad_into(d_logits, hidden_final, _out_wbuf.view(-1))
    else:
        _ret = custom_gemm_output_wgrad(d_logits, hidden_final, out_buf=None)
        if _ret is None:
            _store_wgrad_dict("output_layer.weight", d_logits, hidden_final)
        else:
            fp32_grads["output_layer.weight"] = _ret
    if use_tp:
        ar_handle.wait()

    # --- Final LayerNorm backward ---
    hidden_pre_final = saved["hidden_pre_final"]
    rsigma_final = saved["rsigma_final"]
    final_ln_w = params["decoder.final_layernorm.weight"]

    S_full = d_hidden.shape[0]
    B_batch = d_hidden.shape[1]
    H = d_hidden.shape[2]

    orig_shape = d_hidden.shape
    T = S_full * B_batch

    d_hidden_2d = d_hidden.reshape(T, H)
    hp_2d = hidden_pre_final.reshape(T, H)
    dx, dw = _te_rmsnorm_backward(d_hidden_2d.contiguous(), hp_2d, rsigma_final, final_ln_w)
    d_hidden = dx.view(orig_shape)
    if use_buffer:
        _fln_n = "decoder.final_layernorm.weight"
        grad_buffer[layout.param_offsets[_fln_n]:layout.param_offsets[_fln_n]+layout.param_numels[_fln_n]].add_(dw.float().view(-1))
    else:
        _store_ln_grad_dict("decoder.final_layernorm.weight", dw)

    # Pre-slice RoPE tables (constant across layers).  BF16 copies are only
    # needed by the non-fused RoPE backward path; skip when fused kernels
    # handle the conversion internally (default config).
    cos_fp32 = rope_cos[:S_full, None, None, :]
    sin_fp32 = rope_sin[:S_full, None, None, :]
    if not (cfg.fused_rope_pack or cfg.fused_rope):
        cos_bf16 = cos_fp32.to(torch.bfloat16)
        sin_bf16 = sin_fp32.to(torch.bfloat16)

    # Pre-allocate shared dgrad output buffers (reused across layers)
    T_full = S_full * B_batch
    _ffn_dim = config.FFN_HIDDEN_SIZE // tp_size
    _proj_dim = num_heads * head_dim
    _buf_dact = torch.empty(T_full, _ffn_dim, dtype=torch.bfloat16, device=d_hidden.device)
    _buf_dnorm = torch.empty(T_full, H, dtype=torch.bfloat16, device=d_hidden.device)
    _buf_dproj = torch.empty(T_full, _proj_dim, dtype=torch.bfloat16, device=d_hidden.device)
    _wgrad_fn = _wgrad_gemm_fn

    # --- Transformer layers (reverse order) ---
    # Layer saved activations are tuples (see forward.py for layout):
    #  0: hidden_pre_attn  1: norm_out_attn   2: rsigma_attn
    #  3: proj_input       4: hidden_pre_mlp   5: rsigma_mlp
    #  6: fc1_out|None     7: act_out|None     8: norm_out_mlp|None
    #  9: q_ag|q          10: k_ag|k          11: v_ag|v
    # 12: attn_out_ag|None (tensor ⇒ autograd path, None ⇒ recompute)
    layers_saved = saved["layers"]
    for i in range(num_layers - 1, -1, -1):
        _ls = layers_saved[i]
        fc2_w, fc1_w, mlp_ln_w, proj_w, qkv_w, ln_w = _layer_params[i]
        n_fc2, n_fc1, n_mlp_ln, n_proj, n_qkv, n_ln = _layer_names[i]

        # ── MLP sublayer backward ──
        hidden_pre_mlp = _ls[4]
        rsigma_mlp = _ls[5]

        if _ls[6] is not None:
            fc1_out = _ls[6]
            act_out = _ls[7]
            norm_out_mlp = _ls[8]
        else:
            norm_out_mlp, _ = rmsnorm_forward(
                hidden_pre_mlp, mlp_ln_w, config.NORM_EPSILON,
            )
            _s = norm_out_mlp.shape
            fc1_out = torch.mm(norm_out_mlp.reshape(-1, _s[-1]), fc1_w.t()).view(*_s[:-1], -1)
            act_out = swiglu(fc1_out)

        d_hidden_for_fc2 = d_hidden

        # FC2 dgrad (baseline fallback when custom kernel disabled)
        _d2d_fc2 = d_hidden_for_fc2.view(T_full, H)
        _ret_dact = custom_gemm_fc2_dgrad(_d2d_fc2, fc2_w, out=_buf_dact)
        if _ret_dact is None:
            torch.mm(_d2d_fc2, fc2_w, out=_buf_dact)
        d_act_out = _buf_dact.view(S_full, B_batch, _ffn_dim)

        # FC2 wgrad (baseline fallback when custom kernel disabled).
        # IMPORTANT: _layer_wbufs[i][0] is an FP32 view into grad_buffer,
        # so we cannot use torch.mm(bf16, bf16, out=fp32).  Use Megatron's
        # fused BF16xBF16→FP32 wgrad path via _wgrad_into.
        if use_buffer:
            d_hidden_fc2 = d_hidden_for_fc2
            d_hidden_fc2.record_stream(wgrad_stream)
            _wgrad_wait_main()
            with torch.cuda.stream(wgrad_stream):
                _wret_fc2 = custom_gemm_fc2_wgrad(
                    d_hidden_fc2, act_out, out_buf=_layer_wbufs[i][0],
                )
                if _wret_fc2 is None:
                    _wgrad_into(d_hidden_fc2, act_out, _layer_wbufs[i][0].view(-1))
        else:
            _ret = custom_gemm_fc2_wgrad(d_hidden_for_fc2, act_out, out_buf=None)
            if _ret is None:
                _ret = _wgrad(d_hidden_for_fc2, act_out)
            fp32_grads[n_fc2] = _ret

        # SwiGLU backward
        d_fc1_out = swiglu_backward(d_act_out, fc1_out).view(S_full, B_batch, -1)

        _d2d_fc1 = d_fc1_out.reshape(T_full, -1)
        _ret = custom_gemm_fc1_dgrad(_d2d_fc1, fc1_w, out=_buf_dnorm)
        if _ret is None:
            torch.mm(_d2d_fc1, fc1_w, out=_buf_dnorm)
        if use_tp:
            d_norm_mlp = _buf_dnorm.view(S_full, B_batch, H)
            ar_handle = dist.all_reduce(d_norm_mlp, group=tp_group, async_op=True)
        else:
            d_norm_mlp = _buf_dnorm.view(S_full, B_batch, H)

        if use_buffer:
            d_fc1_out.record_stream(wgrad_stream)
            _wgrad_wait_main()
            with torch.cuda.stream(wgrad_stream):
                _wret = custom_gemm_fc1_wgrad(
                    d_fc1_out, norm_out_mlp, out_buf=_layer_wbufs[i][1],
                )
                if _wret is None:
                    # _layer_wbufs[i][1] is an FP32 view into grad_buffer;
                    # use the BF16xBF16→FP32 fused wgrad helper.
                    _wgrad_into(d_fc1_out, norm_out_mlp, _layer_wbufs[i][1].view(-1))
        else:
            _ret = custom_gemm_fc1_wgrad(d_fc1_out, norm_out_mlp, out_buf=None)
            if _ret is None:
                _ret = _wgrad(d_fc1_out, norm_out_mlp)
            fp32_grads[n_fc1] = _ret
        if use_tp:
            ar_handle.wait()

        _dnorm_for_mlp = _buf_dnorm
        if cfg.fused_residual_rmsnorm:
            if cfg.fused_dw_reduce:
                dw_mlp = fused_residual_add_rmsnorm_bwd_add_dw_reduce(
                    _dnorm_for_mlp, hidden_pre_mlp.reshape(T, H), rsigma_mlp, mlp_ln_w,
                    d_hidden.view(T, H),
                )
            else:
                dw_mlp = fused_residual_add_rmsnorm_bwd_add(
                    _dnorm_for_mlp, hidden_pre_mlp.reshape(T, H), rsigma_mlp, mlp_ln_w,
                    d_hidden.view(T, H),
                )
        else:
            dx_mlp, dw_mlp = _te_rmsnorm_backward(
                _dnorm_for_mlp, hidden_pre_mlp.reshape(T, H), rsigma_mlp, mlp_ln_w,
            )
            d_hidden = d_hidden + dx_mlp.view(orig_shape)
        if use_buffer:
            _layer_ln_bufs[i][0].add_(dw_mlp.float().view(-1))
        else:
            _store_ln_grad_dict(n_mlp_ln, dw_mlp)

        # ── Attention sublayer backward ──
        proj_input = _ls[3]
        norm_out_attn = _ls[1]
        hidden_pre_attn = _ls[0]
        rsigma_attn = _ls[2]

        d_hidden_for_proj = d_hidden

        # Proj dgrad
        _d2d_proj = d_hidden_for_proj.view(T_full, H)
        _ret = custom_gemm_attn_out_proj_dgrad(_d2d_proj, proj_w, out=_buf_dproj)
        if _ret is None:
            torch.mm(_d2d_proj, proj_w, out=_buf_dproj)
        d_proj_input = _buf_dproj.view(S_full, B_batch, _proj_dim)

        # Proj wgrad
        if use_buffer:
            d_hidden_proj = d_hidden_for_proj
            d_hidden_proj.record_stream(wgrad_stream)
            _wgrad_wait_main()
            with torch.cuda.stream(wgrad_stream):
                _ret = custom_gemm_attn_out_proj_wgrad(
                    d_hidden_proj, proj_input, out_buf=_layer_wbufs[i][2],
                )
                if _ret is None:
                    _wgrad_fn(
                        proj_input.reshape(T_full, -1), d_hidden_proj.reshape(T_full, H),
                        _layer_wbufs[i][2],
                    )
        else:
            _ret = custom_gemm_attn_out_proj_wgrad(d_hidden_for_proj, proj_input, out_buf=None)
            if _ret is None:
                _store_wgrad_dict(n_proj, d_hidden_for_proj, proj_input)
            else:
                fp32_grads[n_proj] = _ret

        # Attention backward — three paths (mirrors forward.py):
        #   tuple (slot 12) : DIRECT_ATTN / DSL_ATTN_FWD path; saved
        #                     state holds q/k/v + cuDNN fwd outputs, dq/
        #                     dk/dv come from aten cuDNN _backward.
        #   tensor (slot 12): autograd-saved attn_out_ag; default path.
        #   None  (slot 12) : recompute path — re-run sdpa under
        #                     enable_grad and autograd.grad on the spot.
        _slot12 = _ls[12]
        _bhsd_grad = cfg.rope_bhsd and cfg.direct_attn
        if isinstance(_slot12, tuple):
            from .kernels import causal_attention_bwd_direct
            d_q, d_k, d_v = causal_attention_bwd_direct(
                d_proj_input, _slot12, return_bhsd=_bhsd_grad,
            )
        elif _slot12 is not None:
            d_q, d_k, d_v = torch.autograd.grad(
                _slot12, [_ls[9], _ls[10], _ls[11]],
                grad_outputs=d_proj_input,
            )
        else:
            q = _ls[9]
            k = _ls[10]
            v = _ls[11]
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                q_ag = q.detach().requires_grad_(True)
                k_ag = k.detach().requires_grad_(True)
                v_ag = v.detach().requires_grad_(True)
                attn_out_ag = causal_attention(q_ag, k_ag, v_ag, layer_number=i + 1)
                d_q, d_k, d_v = torch.autograd.grad(
                    attn_out_ag, [q_ag, k_ag, v_ag], grad_outputs=d_proj_input,
                )

        # RoPE backward + QKV gradient packing
        if cfg.fused_rope_pack and _bhsd_grad and isinstance(_slot12, tuple):
            from .kernels import fused_rope_backward_pack_bhsd
            d_qkv = fused_rope_backward_pack_bhsd(
                d_q, d_k, d_v, rope_cos, rope_sin,
                num_kv_heads, heads_per_group, head_dim,
            )
        elif cfg.fused_rope_pack:
            d_qkv = fused_rope_backward_pack(
                d_q, d_k, d_v, rope_cos, rope_sin,
                num_kv_heads, heads_per_group, head_dim,
            )
        else:
            if cfg.fused_rope:
                d_q_pre, d_k_pre = fused_rope_backward(
                    d_q, d_k, rope_cos, rope_sin,
                )
            else:
                d_q_pre = d_q * cos_bf16 - _rotate_half(d_q) * sin_bf16
                d_k_pre = d_k * cos_bf16 - _rotate_half(d_k) * sin_bf16

            d_q_grouped = d_q_pre.reshape(S_full, -1, num_kv_heads, q_per_group)
            d_qkv_packed = torch.cat([d_q_grouped, d_k_pre, d_v], dim=-1)
            d_qkv = d_qkv_packed.reshape(S_full, -1, qkv_dim)

        # QKV dgrad (reuse _buf_dnorm)
        _d2d_qkv = d_qkv.view(T_full, qkv_dim)
        _ret = custom_gemm_qkv_proj_dgrad(_d2d_qkv, qkv_w, out=_buf_dnorm)
        if _ret is None:
            torch.mm(_d2d_qkv, qkv_w, out=_buf_dnorm)
        if use_tp:
            d_norm_attn = _buf_dnorm.view(S_full, B_batch, H)
            ar_handle = dist.all_reduce(d_norm_attn, group=tp_group, async_op=True)
        else:
            d_norm_attn = _buf_dnorm.view(S_full, B_batch, H)

        # QKV wgrad
        if use_buffer:
            d_qkv.record_stream(wgrad_stream)
            _wgrad_wait_main()
            with torch.cuda.stream(wgrad_stream):
                _ret = custom_gemm_qkv_proj_wgrad(
                    d_qkv, norm_out_attn, out_buf=_layer_wbufs[i][3],
                )
                if _ret is None:
                    _wgrad_fn(
                        norm_out_attn.reshape(T_full, H), d_qkv.reshape(T_full, qkv_dim),
                        _layer_wbufs[i][3],
                    )
        else:
            _ret = custom_gemm_qkv_proj_wgrad(d_qkv, norm_out_attn, out_buf=None)
            if _ret is None:
                _store_wgrad_dict(n_qkv, d_qkv, norm_out_attn)
            else:
                fp32_grads[n_qkv] = _ret
        if use_tp:
            ar_handle.wait()

        _dnorm_for_attn = _buf_dnorm
        if cfg.fused_attn_rmsnorm_bwd:
            if cfg.fused_dw_reduce:
                dw_attn = fused_rmsnorm_bwd_add_dw_reduce(
                    _dnorm_for_attn, hidden_pre_attn.reshape(T, H), rsigma_attn, ln_w,
                    d_hidden.view(T, H),
                )
            else:
                dw_attn = fused_rmsnorm_bwd_add(
                    _dnorm_for_attn, hidden_pre_attn.reshape(T, H), rsigma_attn, ln_w,
                    d_hidden.view(T, H),
                )
        else:
            dx_attn, dw_attn = _te_rmsnorm_backward(
                _dnorm_for_attn, hidden_pre_attn.reshape(T, H), rsigma_attn, ln_w,
            )
            d_hidden = d_hidden + dx_attn.view(orig_shape)
        if use_buffer:
            _layer_ln_bufs[i][1].add_(dw_attn.float().view(-1))
        else:
            _store_ln_grad_dict(n_ln, dw_attn)

        # Bucketed RS: signal when a bucket's grads are all written
        if bucket_ready_fn is not None and layers_per_bucket > 0:
            _layers_done = num_layers - i
            if _layers_done % layers_per_bucket == 0:
                _b_idx = _layers_done // layers_per_bucket - 1
                _n_buckets = num_layers // layers_per_bucket
                if _b_idx < _n_buckets - 1:
                    _wg_ev = wgrad_events[_b_idx] if wgrad_events is not None else torch.cuda.Event()
                    wgrad_stream.record_event(_wg_ev)
                    bucket_ready_fn(_b_idx, _wg_ev)

    # --- Embedding backward ---
    # For GAS>1 intermediate micro-batches with defer_embedding_bwd=True,
    # move embedding backward to wgrad_stream and skip the main→wgrad
    # sync.  This lets the next micro-batch's forward start immediately
    # on main_stream while wgrad_stream finishes remaining wgrads +
    # embedding grad.  record_stream on d_hidden and input_ids ensures
    # the allocator keeps them alive even after the caller does `del saved`.
    _defer = defer_embedding_bwd and use_buffer
    if _defer:
        d_hidden.record_stream(wgrad_stream)
        saved["embedding_input"].record_stream(wgrad_stream)
        _wgrad_wait_main()
    elif use_buffer:
        main_stream.wait_stream(wgrad_stream)

    emb_weight = params["embedding.word_embeddings.weight"]
    input_ids = saved["embedding_input"]

    def _do_embedding_bwd():
        nonlocal d_hidden
        d_hidden_bs = d_hidden.transpose(0, 1).contiguous()

        if tp_size > 1:
            V_local = emb_weight.shape[0]
            vocab_start = tp_rank * V_local
            flat_ids = input_ids.reshape(-1)
            masked_ids = (flat_ids - vocab_start).clone()
            input_mask = (masked_ids < 0) | (masked_ids >= V_local)
            masked_ids[input_mask] = 0

            flat_d = d_hidden_bs.reshape(-1, d_hidden_bs.shape[-1])
            flat_d_masked = flat_d.clone()
            flat_d_masked[input_mask] = 0.0

            d_emb = torch.zeros(
                V_local, emb_weight.shape[1],
                dtype=d_hidden.dtype, device=d_hidden.device,
            )
            d_emb.index_put_((masked_ids,), flat_d_masked, accumulate=True)
        else:
            d_emb = torch.zeros(
                emb_weight.shape[0], emb_weight.shape[1],
                dtype=d_hidden.dtype, device=d_hidden.device,
            )
            d_emb.index_put_(
                (input_ids.reshape(-1),),
                d_hidden_bs.reshape(-1, d_hidden_bs.shape[-1]),
                accumulate=True,
            )
        return d_emb

    if _defer:
        with torch.cuda.stream(wgrad_stream):
            d_emb = _do_embedding_bwd()
            _emb_n = "embedding.word_embeddings.weight"
            grad_buffer[layout.param_offsets[_emb_n]:layout.param_offsets[_emb_n]+layout.param_numels[_emb_n]].add_(d_emb.float().view(-1))
            if bucket_ready_fn is not None and layers_per_bucket > 0:
                _n_buckets = num_layers // layers_per_bucket
                bucket_ready_fn(_n_buckets - 1, None)
        # Cross-stream sync at the micro-batch boundary: make main_stream
        # wait for wgrad_stream to fully drain before manual_backward
        # returns, so the next micro-batch's forward does not overlap with
        # in-flight persistent wgrad kernels on the same SMs.
        main_stream.wait_stream(wgrad_stream)
        return None
    else:
        d_emb = _do_embedding_bwd()
        if use_buffer:
            _emb_n = "embedding.word_embeddings.weight"
            grad_buffer[layout.param_offsets[_emb_n]:layout.param_offsets[_emb_n]+layout.param_numels[_emb_n]].add_(d_emb.float().view(-1))
            if bucket_ready_fn is not None and layers_per_bucket > 0:
                _n_buckets = num_layers // layers_per_bucket
                bucket_ready_fn(_n_buckets - 1, None)
            return None
        else:
            fp32_grads["embedding.word_embeddings.weight"] = d_emb.float()
            return fp32_grads


# ---------------------------------------------------------------------------
# Parameter name mapping: engine state-dict key → reference impl
# ``named_parameters()`` key.  Identity under the fused mcore layout
# adopted by this engine; kept as an abstraction layer so a future
# layout change only needs to touch this one table.
# ---------------------------------------------------------------------------
def to_canonical_grad_name(name: str) -> str:
    """Map an engine state-dict key onto the canonical mcore parameter name.

    Identity under the fused mcore layout this engine ships with;
    callers should keep using this shim so future layout drift can be
    handled in one place.
    """
    return name


