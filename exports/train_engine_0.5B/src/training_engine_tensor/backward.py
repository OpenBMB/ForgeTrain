"""Backward pass for MiniCPM4 0.5B.

Implements the static backward execution graph, mirroring the forward pass
in reverse order.  Backward order is determined at compile time.
"""

from __future__ import annotations

__all__ = ["backward_pass", "cross_entropy_loss_backward", "mtp_backward_pass"]

from collections.abc import Callable

import torch

from . import config
from .custom_gemm import (
    custom_gemm_attn_out_proj_bwd,
    custom_gemm_fc1_bwd,
    custom_gemm_fc2_bwd,
    custom_gemm_output_bwd,
    custom_gemm_qkv_proj_bwd,
)
from .engine_config import get_config

# Type alias for the optional bucketed-overlap sink. Receives (param_name,
# fp32_or_bf16_grad). Used by ``train_loop.py`` when ``WGRAD_OVERLAP=1``
# so the BucketedGradReducer can copy each gradient into its bucket slot
# the moment it is produced — and dispatch the per-bucket reduce_scatter
# on the comm stream as soon as the bucket fills, while the next layer's
# backward keeps running on the default stream.
GradSink = Callable[[str, torch.Tensor], None]
from .kernels import (
    attention_backward_te,
    combine_qkv_interleaved,
    linear_backward,
    rmsnorm_te_backward,
    swiglu_back,
)


def _apply_rope_bwd(
    d_q: torch.Tensor,
    d_k: torch.Tensor,
    prefix: str,
    saved: dict[str, torch.Tensor],
    rope_freqs: torch.Tensor,
    *,
    fuse_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Four-way RoPE backward dispatch shared by main backward and MTP backward."""
    rope_freqs_da = saved.get(f"{prefix}.rope_freqs_doc_aware")
    rope_ctx = saved.get(f"{prefix}.rope_ctx")
    if rope_freqs_da is not None:
        from .triton_kernels import fused_rope_bwd_doc_aware
        d_q_pre, d_k_pre = fused_rope_bwd_doc_aware(d_q, d_k, rope_freqs_da)
    elif rope_ctx is not None:
        q_rope_in, k_rope_in, q_rot, k_rot = rope_ctx
        q_rot.backward(d_q, retain_graph=True)
        d_q_pre = q_rope_in.grad.clone()
        q_rope_in.grad = None
        k_rot.backward(d_k, retain_graph=False)
        d_k_pre = k_rope_in.grad.clone()
        k_rope_in.grad = None
    elif fuse_rope:
        from .triton_kernels import fused_rope_bwd
        d_q_pre, d_k_pre = fused_rope_bwd(d_q, d_k, rope_freqs)
    else:
        from .kernels import rope_backward_te
        d_q_pre, d_k_pre = rope_backward_te(d_q, d_k, rope_freqs)
    return d_q_pre, d_k_pre


def cross_entropy_loss_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy loss + backward in FP32, matching the baseline's
    VocabParallelCrossEntropy exactly.

    logits: [S, B, V] in BF16
    labels: [B, S] (int64)
    loss_mask: [B, S] (float)

    Returns: (loss_scalar, per_token_losses, d_logits [S, B, V])
    """
    S, B, V = logits.shape
    logits_f32 = logits.float()

    logits_2d = logits_f32.view(-1, V)
    labels_flat = labels.transpose(0, 1).contiguous().view(-1)
    loss_mask_flat = loss_mask.transpose(0, 1).contiguous().view(-1).float()

    logits_max = logits_2d.max(dim=-1, keepdim=True)[0]
    logits_shifted = logits_2d - logits_max

    exp_logits = torch.exp(logits_shifted)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)

    arange_1d = torch.arange(S * B, device=logits.device)
    predicted_logits = logits_shifted[arange_1d, labels_flat].clone()

    per_token_loss = torch.log(sum_exp.squeeze(-1)) - predicted_logits

    # Aggregate loss in B-major order to match baseline pipeline:
    # VocabParallelCrossEntropy returns [S,B] → transpose to [B,S] → view(-1) → sum
    loss_bs = per_token_loss.view(S, B).transpose(0, 1).contiguous().view(-1)
    mask_bs = loss_mask.view(-1).float()
    valid_tokens = mask_bs.sum()
    loss = (loss_bs * mask_bs).sum() / valid_tokens

    exp_logits.div_(sum_exp)
    softmax = exp_logits

    softmax[arange_1d, labels_flat] -= 1.0

    grad_output = (loss_mask_flat / valid_tokens * loss_scale).unsqueeze(-1)
    softmax *= grad_output

    d_logits = softmax.view(S, B, V).to(logits.dtype)
    return loss, per_token_loss.view(S, B), d_logits


def backward_pass(
    d_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
    saved: dict[str, object],
    rope_freqs: torch.Tensor,
    *,
    diagnostics: dict | None = None,
    grad_sink: GradSink | None = None,
    extra_d_normed_final: torch.Tensor | None = None,
    extra_param_grads: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all parameter gradients via static backward graph.

    ``grad_sink`` (M2-P2 wgrad-allreduce overlap; opt-in via
    ``WGRAD_OVERLAP=1`` in ``train_loop.py``): when supplied, every
    parameter gradient is forwarded to ``grad_sink(name, grad)`` the
    instant the corresponding wgrad GEMM completes, and the returned
    dict is empty. This lets the bucketed reducer dispatch
    ``reduce_scatter`` per-bucket on a side stream as soon as each
    bucket fills, overlapping the collective with the next layers'
    backward GEMMs. When ``grad_sink`` is ``None`` (the default and
    every existing call site today), the function preserves the legacy
    "return a dict of all grads" contract bytewise.

    MTP integration (``config.MTP_NUM_LAYERS > 0``):
    * ``extra_d_normed_final`` — extra gradient flowing INTO
      ``saved["normed_final"]`` from the MTP block (the gradient on
      ``hidden_input_list[0]`` returned by :func:`mtp_backward_pass`).
      Added BEFORE we run ``rmsnorm_te_backward`` for the final norm,
      so the full d_normed_final is consumed in a single pass.
    * ``extra_param_grads`` — per-parameter pre-accumulated gradient
      contributions from MTP that share weights with the main model
      (``output.weight`` and ``tok_embeddings.weight``). Added INSIDE
      ``_emit`` so the bucketed reducer still observes a single write
      per parameter.
    """
    cfg = get_config()
    grads: dict[str, torch.Tensor] = {}
    extra_param_grads = extra_param_grads or {}
    _consumed_extras: set[str] = set()

    def _emit(name: str, g: torch.Tensor) -> None:
        extra = extra_param_grads.get(name)
        if extra is not None:
            # Cast to FP32 to match accumulation precision used by both
            # main and MTP wgrads. Avoid touching the main grad's dtype
            # otherwise (BF16 wgrads stay BF16 like the existing path).
            if g.dtype != torch.float32 and extra.dtype == torch.float32:
                g = g.float() + extra
            else:
                g = g + extra.to(dtype=g.dtype)
            _consumed_extras.add(name)
        if grad_sink is not None:
            grad_sink(name, g)
        else:
            grads[name] = g

    _mup_ws = config.mup_width_scale()
    _output_input = saved["normed_final"]
    if _mup_ws != 1.0:
        _output_input = saved["normed_final"] / _mup_ws
    _custom_out_bwd = custom_gemm_output_bwd(
        d_logits, _output_input, params["output.weight"], te_wgrad=True,
    )
    if _custom_out_bwd is not None:
        d_normed_final, d_output_w = _custom_out_bwd
    else:
        d_normed_final, d_output_w = linear_backward(
            d_logits, _output_input, params["output.weight"],
            te_wgrad=True,
        )
    if _mup_ws != 1.0:
        d_normed_final = d_normed_final / _mup_ws
    _emit("output.weight", d_output_w)

    if extra_d_normed_final is not None:
        # MTP backward returns the gradient flowing INTO the MTP input
        # (which is the main model's normed_final). Accumulate it in
        # FP32 to keep the precision contract with the layernorm
        # backward kernel.
        if d_normed_final.dtype == torch.float32 or extra_d_normed_final.dtype == torch.float32:
            d_normed_final = d_normed_final.float() + extra_d_normed_final.float()
            d_normed_final = d_normed_final.to(saved["hidden_final"].dtype)
        else:
            d_normed_final = d_normed_final + extra_d_normed_final.to(d_normed_final.dtype)

    if diagnostics is not None:
        diagnostics["d_normed_final"] = d_normed_final.detach().clone()

    d_hidden, d_norm_w = rmsnorm_te_backward(
        d_normed_final, saved["hidden_final"], params["norm.weight"],
        rsigma=saved.get("rsigma_final"),
    )
    _emit("norm.weight", d_norm_w)

    if diagnostics is not None:
        diagnostics["d_hidden_after_norm"] = d_hidden.detach().clone()

    _mup_res = config.mup_residual_scale()

    for i in reversed(range(config.NUM_LAYERS)):
        p = f"layers.{i}"

        d_hidden_for_down = d_hidden * _mup_res if _mup_res != 1.0 else d_hidden
        _custom_fc2 = custom_gemm_fc2_bwd(
            d_hidden_for_down, saved[f"{p}.intermediate"],
            params[f"{p}.feed_forward.w2.weight"], te_wgrad=True,
        )
        if _custom_fc2 is not None:
            d_intermediate, d_w2 = _custom_fc2
        else:
            d_intermediate, d_w2 = linear_backward(
                d_hidden_for_down, saved[f"{p}.intermediate"],
                params[f"{p}.feed_forward.w2.weight"],
                te_wgrad=True,
            )
        _emit(f"{p}.feed_forward.w2.weight", d_w2)

        _diag_last = diagnostics is not None and i == config.NUM_LAYERS - 1

        if _diag_last:
            diagnostics["d_hidden_into_layer23"] = d_hidden.detach().clone()
            diagnostics["d_intermediate_23"] = d_intermediate.detach().clone()

        d_fc1 = swiglu_back(d_intermediate, saved[f"{p}.fc1_output"])

        if _diag_last:
            diagnostics["d_fc1_23"] = d_fc1.detach().clone()

        _custom_fc1 = custom_gemm_fc1_bwd(
            d_fc1, saved[f"{p}.ffn_normed"],
            params[f"{p}.feed_forward.wfc1.weight"], te_wgrad=True,
        )
        if _custom_fc1 is not None:
            d_ffn_normed, d_wfc1 = _custom_fc1
        else:
            d_ffn_normed, d_wfc1 = linear_backward(
                d_fc1, saved[f"{p}.ffn_normed"],
                params[f"{p}.feed_forward.wfc1.weight"],
                te_wgrad=True,
            )
        _emit(f"{p}.feed_forward.wfc1.weight", d_wfc1)

        if cfg.use_fused_norm_bwd:
            from .triton_kernels import fused_rmsnorm_bwd_residual
            d_hidden, d_ffn_norm_w = fused_rmsnorm_bwd_residual(
                d_ffn_normed, saved[f"{p}.hidden_after_attn"],
                params[f"{p}.ffn_norm.weight"],
                saved.get(f"{p}.rsigma_ffn"),
                d_hidden,
            )
        else:
            d_hidden_mlp, d_ffn_norm_w = rmsnorm_te_backward(
                d_ffn_normed, saved[f"{p}.hidden_after_attn"],
                params[f"{p}.ffn_norm.weight"],
                rsigma=saved.get(f"{p}.rsigma_ffn"),
            )
            if cfg.fuse_inplace_residual:
                d_hidden.add_(d_hidden_mlp)
            else:
                d_hidden = d_hidden + d_hidden_mlp
        _emit(f"{p}.ffn_norm.weight", d_ffn_norm_w)

        if _diag_last:
            diagnostics["d_hidden_after_mlp_23"] = d_hidden.detach().clone()

        d_hidden_for_attn = d_hidden * _mup_res if _mup_res != 1.0 else d_hidden
        _custom_aop = custom_gemm_attn_out_proj_bwd(
            d_hidden_for_attn, saved[f"{p}.attn_out"],
            params[f"{p}.attention.wo.weight"], te_wgrad=True,
        )
        if _custom_aop is not None:
            d_attn_out, d_wo = _custom_aop
        else:
            d_attn_out, d_wo = linear_backward(
                d_hidden_for_attn, saved[f"{p}.attn_out"],
                params[f"{p}.attention.wo.weight"],
                te_wgrad=True,
            )
        _emit(f"{p}.attention.wo.weight", d_wo)

        if _diag_last:
            diagnostics["d_attn_out_23"] = d_attn_out.detach().clone()

        d_q, d_k, d_v = attention_backward_te(
            d_attn_out,
            saved[f"{p}.q_rotary"],
            saved[f"{p}.k_rotary"],
            saved[f"{p}.v_heads"],
            saved[f"{p}.attn_out"],
            saved[f"{p}.attn_aux_ctx"],
        )

        d_q_pre, d_k_pre = _apply_rope_bwd(
            d_q, d_k, p, saved, rope_freqs, fuse_rope=cfg.fuse_rope,
        )

        d_qkv = combine_qkv_interleaved(d_q_pre, d_k_pre, d_v)

        if _diag_last:
            diagnostics["d_qkv_23"] = d_qkv.detach().clone()

        _custom_qkv = custom_gemm_qkv_proj_bwd(
            d_qkv, saved[f"{p}.normed_attn"],
            params[f"{p}.attention.wqkv.weight"], te_wgrad=True,
        )
        if _custom_qkv is not None:
            d_normed, d_wqkv = _custom_qkv
        else:
            d_normed, d_wqkv = linear_backward(
                d_qkv, saved[f"{p}.normed_attn"],
                params[f"{p}.attention.wqkv.weight"],
                te_wgrad=True,
            )
        _emit(f"{p}.attention.wqkv.weight", d_wqkv)

        if cfg.use_fused_norm_bwd:
            d_hidden, d_attn_norm_w = fused_rmsnorm_bwd_residual(
                d_normed, saved[f"{p}.hidden_input"],
                params[f"{p}.attention_norm.weight"],
                saved.get(f"{p}.rsigma_attn"),
                d_hidden,
            )
        else:
            d_hidden_attn, d_attn_norm_w = rmsnorm_te_backward(
                d_normed, saved[f"{p}.hidden_input"],
                params[f"{p}.attention_norm.weight"],
                rsigma=saved.get(f"{p}.rsigma_attn"),
            )
            if cfg.fuse_inplace_residual:
                d_hidden.add_(d_hidden_attn)
            else:
                d_hidden = d_hidden + d_hidden_attn
        _emit(f"{p}.attention_norm.weight", d_attn_norm_w)

    if cfg.mup_emb_scale > 0:
        d_hidden = d_hidden * cfg.mup_emb_scale
    input_ids = saved["input_ids"]
    if cfg.fuse_index_add_emb_bwd:
        d_hidden_bs = d_hidden.transpose(0, 1).reshape(-1, config.HIDDEN_SIZE)
        ids_flat = input_ids.reshape(-1)
        vocab_size = params["tok_embeddings.weight"].shape[0]
        d_emb = torch.zeros(
            vocab_size, config.HIDDEN_SIZE,
            dtype=torch.float32, device=d_hidden.device,
        )
        d_emb.index_add_(0, ids_flat, d_hidden_bs.float())
        _emit("tok_embeddings.weight", d_emb)
    else:
        weight_emb = params["tok_embeddings.weight"].detach().clone().requires_grad_(True)
        emb = torch.ops.aten.embedding(weight_emb, input_ids, -1, False, False)
        emb.backward(d_hidden.transpose(0, 1))
        _emit("tok_embeddings.weight", weight_emb.grad.float())

    # Sanity guard: every key the MTP path passed in via
    # ``extra_param_grads`` MUST have been consumed by an _emit call
    # for a shared parameter. Anything left over indicates a bug
    # (typically a typo in the MTP backward output dict).
    if extra_param_grads:
        leftover = set(extra_param_grads.keys()) - _consumed_extras
        if leftover:
            raise RuntimeError(
                "backward_pass: extra_param_grads contained keys with no "
                "matching _emit call; either the MTP backward emitted them "
                "directly (then they should not appear here) or the name is "
                f"misspelled. Leftover: {sorted(leftover)}"
            )

    return grads


# ---------------------------------------------------------------------------
# Multi-Token Prediction (MTP) backward
# ---------------------------------------------------------------------------


def _mtp_transformer_layer_backward(
    d_hidden: torch.Tensor,
    params: dict[str, torch.Tensor],
    saved: dict[str, object],
    prefix: str,
    rope_freqs: torch.Tensor,
    *,
    grad_emit,
) -> torch.Tensor:
    """Backward for one MTP transformer layer (mirror of
    :func:`forward._mtp_transformer_layer_forward`).

    Takes the gradient on the layer's OUTPUT (post-FFN-residual hidden
    state) and returns the gradient on the layer's INPUT (the
    ``proj`` tensor produced by ``eh_proj``). MTP-only weight
    gradients (qkv / wo / wfc1 / w2 + the two layernorms) are emitted
    via ``grad_emit(name, g)`` so the caller controls whether they go
    into the bucketed grad sink or a return dict.

    No FUSE_RMSNORM_RESIDUAL fast-path here — the MTP layer is
    standalone (no cross-layer fusion), so we always run the unfused
    rmsnorm + add residual sequence.
    """
    cfg = get_config()
    _mup_res = config.mup_residual_scale()

    d_hidden_for_down = d_hidden * _mup_res if _mup_res != 1.0 else d_hidden
    _custom_fc2 = custom_gemm_fc2_bwd(
        d_hidden_for_down, saved[f"{prefix}.intermediate"],
        params[f"{prefix}.feed_forward.w2.weight"], te_wgrad=True,
    )
    if _custom_fc2 is not None:
        d_intermediate, d_w2 = _custom_fc2
    else:
        d_intermediate, d_w2 = linear_backward(
            d_hidden_for_down, saved[f"{prefix}.intermediate"],
            params[f"{prefix}.feed_forward.w2.weight"],
            te_wgrad=True,
        )
    grad_emit(f"{prefix}.feed_forward.w2.weight", d_w2)

    d_fc1 = swiglu_back(d_intermediate, saved[f"{prefix}.fc1_output"])

    _custom_fc1 = custom_gemm_fc1_bwd(
        d_fc1, saved[f"{prefix}.ffn_normed"],
        params[f"{prefix}.feed_forward.wfc1.weight"], te_wgrad=True,
    )
    if _custom_fc1 is not None:
        d_ffn_normed, d_wfc1 = _custom_fc1
    else:
        d_ffn_normed, d_wfc1 = linear_backward(
            d_fc1, saved[f"{prefix}.ffn_normed"],
            params[f"{prefix}.feed_forward.wfc1.weight"],
            te_wgrad=True,
        )
    grad_emit(f"{prefix}.feed_forward.wfc1.weight", d_wfc1)

    d_hidden_mlp, d_ffn_norm_w = rmsnorm_te_backward(
        d_ffn_normed, saved[f"{prefix}.hidden_after_attn"],
        params[f"{prefix}.ffn_norm.weight"],
        rsigma=saved.get(f"{prefix}.rsigma_ffn"),
    )
    d_hidden = d_hidden + d_hidden_mlp
    grad_emit(f"{prefix}.ffn_norm.weight", d_ffn_norm_w)

    d_hidden_for_attn = d_hidden * _mup_res if _mup_res != 1.0 else d_hidden
    _custom_aop = custom_gemm_attn_out_proj_bwd(
        d_hidden_for_attn, saved[f"{prefix}.attn_out"],
        params[f"{prefix}.attention.wo.weight"], te_wgrad=True,
    )
    if _custom_aop is not None:
        d_attn_out, d_wo = _custom_aop
    else:
        d_attn_out, d_wo = linear_backward(
            d_hidden_for_attn, saved[f"{prefix}.attn_out"],
            params[f"{prefix}.attention.wo.weight"],
            te_wgrad=True,
        )
    grad_emit(f"{prefix}.attention.wo.weight", d_wo)

    d_q, d_k, d_v = attention_backward_te(
        d_attn_out,
        saved[f"{prefix}.q_rotary"],
        saved[f"{prefix}.k_rotary"],
        saved[f"{prefix}.v_heads"],
        saved[f"{prefix}.attn_out"],
        saved[f"{prefix}.attn_aux_ctx"],
    )

    d_q_pre, d_k_pre = _apply_rope_bwd(
        d_q, d_k, prefix, saved, rope_freqs, fuse_rope=cfg.fuse_rope,
    )

    d_qkv = combine_qkv_interleaved(d_q_pre, d_k_pre, d_v)

    _custom_qkv = custom_gemm_qkv_proj_bwd(
        d_qkv, saved[f"{prefix}.normed_attn"],
        params[f"{prefix}.attention.wqkv.weight"], te_wgrad=True,
    )
    if _custom_qkv is not None:
        d_normed, d_wqkv = _custom_qkv
    else:
        d_normed, d_wqkv = linear_backward(
            d_qkv, saved[f"{prefix}.normed_attn"],
            params[f"{prefix}.attention.wqkv.weight"],
            te_wgrad=True,
        )
    grad_emit(f"{prefix}.attention.wqkv.weight", d_wqkv)

    d_hidden_attn, d_attn_norm_w = rmsnorm_te_backward(
        d_normed, saved[f"{prefix}.hidden_input"],
        params[f"{prefix}.attention_norm.weight"],
        rsigma=saved.get(f"{prefix}.rsigma_attn"),
    )
    d_hidden = d_hidden + d_hidden_attn
    grad_emit(f"{prefix}.attention_norm.weight", d_attn_norm_w)

    return d_hidden


def mtp_backward_pass(
    mtp_outputs: dict,
    params: dict[str, torch.Tensor],
    rope_freqs: torch.Tensor,
    *,
    loss_scale: float,
    grad_sink: GradSink | None = None,
) -> dict[str, object]:
    """Backward for the entire MTP block.

    Takes the dict produced by :func:`forward.mtp_forward_pass_with_save`
    and:

    * For each MTP layer k = K-1 down to 0, runs CE backward to get
      ``d_mtp_logits_k`` (with ``loss_scale = mtp_loss_scaling_factor /
      mtp_num_layers / num_local_microbatches``), then walks the layer
      backward through ``output_layer → final_layernorm →
      transformer_layer → eh_proj → split → enorm/hnorm`` while
      accumulating shared-weight gradients (``output.weight`` and
      ``tok_embeddings.weight``).
    * Emits MTP-only param gradients via ``grad_sink`` (or stores them
      in the returned ``param_grads`` dict when ``grad_sink`` is None
      so the caller can fold them into a single param-grad bag).

    Returns a dict with:
      * ``mtp_losses``: list[Tensor scalar] per MTP layer (FP64 for log)
      * ``d_normed_final``: contribution to be added to main backward's
        d_normed_final (the gradient on ``hidden_input_list[0]`` =
        the MTP block's input from the main model).
      * ``shared_grads``: dict[str, Tensor] containing the accumulated
        grads for ``output.weight`` and ``tok_embeddings.weight`` —
        these are passed to ``backward_pass(extra_param_grads=...)`` so
        the bucketed reducer still observes a single write per
        parameter.
      * ``param_grads``: dict[str, Tensor] of MTP-only grads when
        ``grad_sink`` is None (otherwise empty).
    """
    cfg = get_config()
    if cfg.mtp_num_layers <= 0:
        raise RuntimeError("mtp_backward_pass called with MTP_NUM_LAYERS=0")

    saved = mtp_outputs["saved"]
    logits_list: list[torch.Tensor] = mtp_outputs["logits_list"]
    rolled_labels_list: list[torch.Tensor] = mtp_outputs["rolled_labels_list"]
    rolled_mask_list: list[torch.Tensor] = mtp_outputs["rolled_mask_list"]
    rolled_input_ids_list: list[torch.Tensor] = mtp_outputs["rolled_input_ids_list"]
    hidden_input_list: list[torch.Tensor] = mtp_outputs["hidden_input_list"]

    K = len(logits_list)
    if K != cfg.mtp_num_layers:
        raise ValueError(
            f"mtp_backward_pass: got {K} logits but MTP_NUM_LAYERS={cfg.mtp_num_layers}"
        )

    param_grads: dict[str, torch.Tensor] = {}

    def _emit(name: str, g: torch.Tensor) -> None:
        if grad_sink is not None:
            grad_sink(name, g)
        else:
            param_grads[name] = g

    # Shared-weight accumulators (output.weight is FP32 from
    # linear_backward; tok_embeddings is built FP32 via index_add).
    h = config.HIDDEN_SIZE
    vocab_size = params["tok_embeddings.weight"].shape[0]
    device = hidden_input_list[0].device
    d_output_w_total = torch.zeros_like(
        params["output.weight"], dtype=torch.float32, device=device
    )
    d_tok_emb_total = torch.zeros(
        vocab_size, h, dtype=torch.float32, device=device,
    )

    mtp_losses: list[torch.Tensor] = []

    # Carry-over: the gradient flowing INTO the input of MTP layer (k+1)
    # is added to the gradient of the OUTPUT of MTP layer k's
    # transformer_layer (because hidden_input_list[k+1] is exactly
    # the post-transformer hidden tensor of layer k). For k = K-1
    # there is no "next" iteration so the carry starts at None.
    d_carry: torch.Tensor | None = None

    for k in range(K - 1, -1, -1):
        prefix = f"mtp.layers.{k}"

        # 1) CE backward → d_mtp_logits + scalar loss for logging.
        # Use the same fused-CE Triton kernel the main path uses when
        # FUSE_CE is on, otherwise fall back to the precision-matched
        # ``cross_entropy_loss_backward`` reference impl.
        if cfg.fuse_ce:
            from .triton_kernels import fused_cross_entropy_fwd_bwd
            mtp_loss_k, _, d_mtp_logits = fused_cross_entropy_fwd_bwd(
                logits_list[k],
                rolled_labels_list[k],
                rolled_mask_list[k],
                loss_scale=loss_scale,
            )
        else:
            mtp_loss_k, _, d_mtp_logits = cross_entropy_loss_backward(
                logits_list[k],
                rolled_labels_list[k],
                rolled_mask_list[k],
                loss_scale=loss_scale,
            )
        mtp_losses.append(mtp_loss_k.detach())

        # 2) output_layer backward — MTP applies the SAME muP width-scale
        # divisor as the main head (EAGLE-style), so we must run the
        # linear backward on the muP-scaled hidden and then divide the
        # input gradient by ``mup_ws`` once more (chain rule through
        # the division). d_output_w is FP32 and accumulates across MTP
        # layers.
        hidden_after_layer = saved[f"{prefix}.hidden_after_layer"]
        _mup_ws = config.mup_width_scale()
        if _mup_ws != 1.0:
            output_input = hidden_after_layer / _mup_ws
        else:
            output_input = hidden_after_layer
        _custom_out = custom_gemm_output_bwd(
            d_mtp_logits, output_input, params["output.weight"], te_wgrad=True,
        )
        if _custom_out is not None:
            d_output_input, d_output_w_k = _custom_out
        else:
            d_output_input, d_output_w_k = linear_backward(
                d_mtp_logits, output_input, params["output.weight"],
                te_wgrad=True,
            )
        d_output_w_total = d_output_w_total + d_output_w_k.float()
        if _mup_ws != 1.0:
            d_hidden_after_layer = d_output_input / _mup_ws
        else:
            d_hidden_after_layer = d_output_input

        # 3) (No final_layernorm in the EAGLE-style MTP — see
        # MTPTransformerBlock.forward; the transformer-layer output
        # is fed straight into the shared output_layer.)

        # 4) Add the carry from MTP layer (k+1)'s backward (gradient on
        # this layer's transformer_layer output, which equals the input
        # of MTP layer k+1).
        if d_carry is not None:
            d_hidden_after_layer = d_hidden_after_layer + d_carry

        # 5) Transformer layer backward → gradient on the layer INPUT
        # (the ``proj`` tensor produced by eh_proj).
        d_proj = _mtp_transformer_layer_backward(
            d_hidden_after_layer,
            params,
            saved,
            prefix,
            rope_freqs,
            grad_emit=_emit,
        )

        # 6) eh_proj backward (linear). Cast inputs through the regular
        # ``linear_backward`` helper since this is a one-off Linear and
        # no custom_gemm path is registered for it.
        combined = saved[f"{prefix}.eh_proj_input"]  # [S, B, 2H]
        d_combined, d_eh_proj_w = linear_backward(
            d_proj, combined, params[f"{prefix}.eh_proj.weight"],
            te_wgrad=True,
        )
        _emit(f"{prefix}.eh_proj.weight", d_eh_proj_w)

        # 7) Split d_combined into d_decoder_input_normed [S,B,H] and
        # d_hidden_normed [S,B,H] (concat was [decoder_input, hidden]
        # along the last dim).
        d_decoder_input_normed = d_combined[..., :h].contiguous()
        d_hidden_normed = d_combined[..., h:].contiguous()

        # 8) enorm backward.
        d_decoder_input, d_enorm_w = rmsnorm_te_backward(
            d_decoder_input_normed,
            saved[f"{prefix}.decoder_input"],
            params[f"{prefix}.enorm.weight"],
            rsigma=saved.get(f"{prefix}.enorm_rsigma"),
        )
        _emit(f"{prefix}.enorm.weight", d_enorm_w)

        # 9) hnorm backward — gradient on the MTP layer INPUT
        # (``hidden_input_list[k]``). For k=0 this is the contribution
        # back to the main model's ``normed_final``; for k>0 it becomes
        # the carry into iteration k-1.
        d_hidden_input, d_hnorm_w = rmsnorm_te_backward(
            d_hidden_normed,
            hidden_input_list[k],
            params[f"{prefix}.hnorm.weight"],
            rsigma=saved.get(f"{prefix}.hnorm_rsigma"),
        )
        _emit(f"{prefix}.hnorm.weight", d_hnorm_w)

        # 10) Shared embedding backward. Use FP32 index_add to mirror
        # the FUSE_INDEX_ADD_EMB_BWD path in the main backward (always
        # active here because we want a single FP32 buffer to combine
        # with the main embedding grad in ``backward_pass``).
        if cfg.mup_emb_scale > 0:
            d_dec = d_decoder_input * cfg.mup_emb_scale
        else:
            d_dec = d_decoder_input
        d_dec_bs = d_dec.transpose(0, 1).reshape(-1, h)
        ids_flat = rolled_input_ids_list[k].reshape(-1)
        d_tok_emb_total.index_add_(0, ids_flat, d_dec_bs.float())

        # Prepare carry for the next (smaller k) iteration.
        d_carry = d_hidden_input

    # After the k=0 iteration, d_carry IS the gradient on the MTP
    # input = main model's normed_final. Keep it FP32 for additive
    # combination inside backward_pass.
    d_normed_final_contribution = d_carry

    return {
        "mtp_losses": list(reversed(mtp_losses)),
        "d_normed_final": d_normed_final_contribution,
        "shared_grads": {
            "output.weight": d_output_w_total,
            "tok_embeddings.weight": d_tok_emb_total,
        },
        "param_grads": param_grads,
    }
