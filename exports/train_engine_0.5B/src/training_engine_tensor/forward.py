"""Forward pass for MiniCPM4 0.5B.

Implements the static forward execution graph: embedding -> 24 transformer
layers (with GQA attention + SwiGLU FFN) -> final norm -> output projection.
Optionally also runs Multi-Token Prediction (MTP, DeepSeek-V3 style): one
or more extra heads that predict the (k+1)-th future token, sharing the
main model's embedding and output_layer.
"""

from __future__ import annotations

__all__ = ["forward_pass_with_save", "mtp_forward_pass_with_save"]

import torch

from . import config
from .custom_gemm import (
    custom_gemm_attn_out_proj_fwd,
    custom_gemm_fc1_fwd,
    custom_gemm_fc2_fwd,
    custom_gemm_output_fwd,
    custom_gemm_qkv_proj_fwd,
)
from .engine_config import get_config
from .kernels import (
    apply_rotary_embeddings_te,
    attention_forward_te,
    linear,
    rmsnorm_te_with_rsigma,
    split_qkv_interleaved,
    swiglu,
)


def _apply_rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_freqs: torch.Tensor,
    prefix: str,
    saved: dict[str, torch.Tensor],
    *,
    fuse_rope: bool,
    fuse_rope_doc_aware: bool,
    doc_aware_rope: bool,
    need_backward: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Four-way RoPE dispatch shared by main forward and MTP forward."""
    if fuse_rope and not doc_aware_rope:
        from .triton_kernels import fused_rope_fwd
        q, k = fused_rope_fwd(q, k, rope_freqs)
    elif doc_aware_rope and fuse_rope and fuse_rope_doc_aware:
        from .triton_kernels import fused_rope_fwd_doc_aware
        if need_backward:
            saved[f"{prefix}.q_pre_rope"] = q
            saved[f"{prefix}.k_pre_rope"] = k
            saved[f"{prefix}.rope_freqs_doc_aware"] = rope_freqs
        q, k = fused_rope_fwd_doc_aware(q, k, rope_freqs)
    elif doc_aware_rope:
        from .kernels import _apply_rope_manual
        saved[f"{prefix}.q_pre_rope"] = q
        saved[f"{prefix}.k_pre_rope"] = k
        q_rope_in = q.detach().requires_grad_(True)
        k_rope_in = k.detach().requires_grad_(True)
        q_rot = _apply_rope_manual(q_rope_in, rope_freqs)
        k_rot = _apply_rope_manual(k_rope_in, rope_freqs)
        saved[f"{prefix}.rope_ctx"] = (q_rope_in, k_rope_in, q_rot, k_rot)
        q, k = q_rot.detach(), k_rot.detach()
    elif need_backward:
        saved[f"{prefix}.q_pre_rope"] = q
        saved[f"{prefix}.k_pre_rope"] = k
        q_rope_in = q.detach().requires_grad_(True)
        k_rope_in = k.detach().requires_grad_(True)
        q_rot, k_rot = apply_rotary_embeddings_te(q_rope_in, k_rope_in, rope_freqs)
        saved[f"{prefix}.rope_ctx"] = (q_rope_in, k_rope_in, q_rot, k_rot)
        q, k = q_rot.detach(), k_rot.detach()
    else:
        q, k = apply_rotary_embeddings_te(q, k, rope_freqs)
    return q, k


def forward_pass_with_save(
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    need_backward: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run forward pass, saving intermediates for backward.

    Returns (logits, saved_dict).
    """
    cfg = get_config()
    saved: dict[str, object] = {}
    saved["input_ids"] = input_ids

    _doc_aware_rope = position_ids is not None and cfg.use_span_based_attn
    if _doc_aware_rope:
        freqs_table = rope_freqs.squeeze(1).squeeze(1)  # [max_seq, D]
        pos_sb = position_ids.transpose(0, 1).contiguous()  # [S, B]
        rope_freqs = freqs_table[pos_sb.long()][:, :, None, :]  # [S, B, 1, D]

    hidden = params["tok_embeddings.weight"][input_ids]
    if cfg.mup_emb_scale > 0:
        hidden = hidden * cfg.mup_emb_scale
    hidden = hidden.transpose(0, 1).contiguous()

    _mup_res = config.mup_residual_scale()
    _pre_normed = None
    normed_final = None
    rsigma_final = None

    for i in range(config.NUM_LAYERS):
        p = f"layers.{i}"
        saved[f"{p}.hidden_input"] = hidden

        if _pre_normed is not None:
            normed, rsigma_attn = _pre_normed
            _pre_normed = None
        else:
            normed, rsigma_attn = rmsnorm_te_with_rsigma(hidden, params[f"{p}.attention_norm.weight"])
        saved[f"{p}.normed_attn"] = normed
        saved[f"{p}.rsigma_attn"] = rsigma_attn

        qkv = custom_gemm_qkv_proj_fwd(normed, params[f"{p}.attention.wqkv.weight"])
        if qkv is None:
            qkv = linear(normed, params[f"{p}.attention.wqkv.weight"])
        q, k, v = split_qkv_interleaved(qkv)

        q, k = _apply_rope_fwd(
            q, k, rope_freqs, p, saved,
            fuse_rope=cfg.fuse_rope,
            fuse_rope_doc_aware=cfg.fuse_rope_doc_aware,
            doc_aware_rope=_doc_aware_rope,
            need_backward=need_backward,
        )
        saved[f"{p}.q_rotary"] = q
        saved[f"{p}.k_rotary"] = k
        saved[f"{p}.v_heads"] = v

        attn_out, aux_ctx = attention_forward_te(
            q, k, v,
            causal=True,
            layer_number=i + 1,
            need_backward=need_backward,
        )
        saved[f"{p}.attn_out"] = attn_out
        saved[f"{p}.attn_aux_ctx"] = aux_ctx

        o_out = custom_gemm_attn_out_proj_fwd(attn_out, params[f"{p}.attention.wo.weight"])
        if o_out is None:
            o_out = linear(attn_out, params[f"{p}.attention.wo.weight"])

        if _mup_res != 1.0:
            o_out = o_out * _mup_res
        if cfg.fuse_rmsnorm_residual:
            from .triton_kernels import fused_residual_rmsnorm_fwd
            hidden, normed2, rsigma_ffn = fused_residual_rmsnorm_fwd(
                hidden, o_out, params[f"{p}.ffn_norm.weight"], config.NORM_EPS,
            )
        else:
            hidden = hidden + o_out
            normed2, rsigma_ffn = rmsnorm_te_with_rsigma(hidden, params[f"{p}.ffn_norm.weight"])
        saved[f"{p}.hidden_after_attn"] = hidden
        saved[f"{p}.ffn_normed"] = normed2
        saved[f"{p}.rsigma_ffn"] = rsigma_ffn

        fc1_out = custom_gemm_fc1_fwd(normed2, params[f"{p}.feed_forward.wfc1.weight"])
        if fc1_out is None:
            fc1_out = linear(normed2, params[f"{p}.feed_forward.wfc1.weight"])
        saved[f"{p}.fc1_output"] = fc1_out

        intermediate = swiglu(fc1_out)
        saved[f"{p}.intermediate"] = intermediate

        down = custom_gemm_fc2_fwd(intermediate, params[f"{p}.feed_forward.w2.weight"])
        if down is None:
            down = linear(intermediate, params[f"{p}.feed_forward.w2.weight"])

        if _mup_res != 1.0:
            down = down * _mup_res
        if cfg.fuse_rmsnorm_residual and i < config.NUM_LAYERS - 1:
            next_p = f"layers.{i + 1}"
            hidden, next_normed, next_rsigma = fused_residual_rmsnorm_fwd(
                hidden, down, params[f"{next_p}.attention_norm.weight"], config.NORM_EPS,
            )
            _pre_normed = (next_normed, next_rsigma)
        elif cfg.fuse_rmsnorm_residual and i == config.NUM_LAYERS - 1:
            hidden, normed_final, rsigma_final = fused_residual_rmsnorm_fwd(
                hidden, down, params["norm.weight"], config.NORM_EPS,
            )
        else:
            hidden = hidden + down

    saved["hidden_final"] = hidden

    if cfg.fuse_rmsnorm_residual:
        if normed_final is None:
            raise RuntimeError("normed_final unset after fused residual loop")
        if rsigma_final is None:
            raise RuntimeError("rsigma_final unset after fused residual loop")
        saved["normed_final"] = normed_final
        saved["rsigma_final"] = rsigma_final
    else:
        normed_final, rsigma_final = rmsnorm_te_with_rsigma(hidden, params["norm.weight"])
        saved["normed_final"] = normed_final
        saved["rsigma_final"] = rsigma_final

    output_input = normed_final
    _mup_ws = config.mup_width_scale()
    if _mup_ws != 1.0:
        output_input = normed_final / _mup_ws

    logits = custom_gemm_output_fwd(output_input, params["output.weight"])
    if logits is None:
        logits = linear(output_input, params["output.weight"])
    return logits, saved


# ---------------------------------------------------------------------------
# Multi-Token Prediction (MTP, DeepSeek-V3 style)
# ---------------------------------------------------------------------------


def _roll_left_one(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shift ``t`` left by 1 along ``dim`` and zero the trailing slot.

    Mirrors Megatron's
    ``rolled = torch.roll(t, -1, dim); rolled.select(dim, -1).fill_(0)``
    used by ``MultiTokenPredictionBlock.forward`` for ``input_ids`` /
    ``labels`` / ``loss_mask``.
    """
    rolled = torch.roll(t, shifts=-1, dims=dim)
    rolled.select(dim, -1).fill_(0)
    return rolled


def _mtp_transformer_layer_forward(
    hidden: torch.Tensor,
    params: dict[str, torch.Tensor],
    prefix: str,
    saved: dict[str, object],
    rope_freqs: torch.Tensor,
    *,
    layer_number: int,
    doc_aware_rope: bool,
    need_backward: bool,
) -> torch.Tensor:
    """Run ONE transformer layer at the given parameter ``prefix``.

    Mirrors the per-layer body of :func:`forward_pass_with_save` for the
    standalone (non-chained) case: no ``_pre_normed`` carry-over from a
    previous layer, no FUSE_RMSNORM_RESIDUAL into a NEXT layer's
    attention norm. Used by the MTP forward where each MTP layer is
    structurally a regular transformer layer (we deliberately reuse the
    same parameter sub-naming so this single helper drives both the MTP
    layer and — at the kernel-call level — the main loop).

    All intermediate tensors are stashed into ``saved`` under
    ``f"{prefix}.<key>"`` so the matching backward helper can pull them
    out symmetrically.
    """
    cfg = get_config()
    saved[f"{prefix}.hidden_input"] = hidden

    normed, rsigma_attn = rmsnorm_te_with_rsigma(
        hidden, params[f"{prefix}.attention_norm.weight"]
    )
    saved[f"{prefix}.normed_attn"] = normed
    saved[f"{prefix}.rsigma_attn"] = rsigma_attn

    qkv = custom_gemm_qkv_proj_fwd(normed, params[f"{prefix}.attention.wqkv.weight"])
    if qkv is None:
        qkv = linear(normed, params[f"{prefix}.attention.wqkv.weight"])
    q, k, v = split_qkv_interleaved(qkv)

    q, k = _apply_rope_fwd(
        q, k, rope_freqs, prefix, saved,
        fuse_rope=cfg.fuse_rope,
        fuse_rope_doc_aware=cfg.fuse_rope_doc_aware,
        doc_aware_rope=doc_aware_rope,
        need_backward=need_backward,
    )
    saved[f"{prefix}.q_rotary"] = q
    saved[f"{prefix}.k_rotary"] = k
    saved[f"{prefix}.v_heads"] = v

    attn_out, aux_ctx = attention_forward_te(
        q, k, v,
        causal=True,
        layer_number=layer_number,
        need_backward=need_backward,
    )
    saved[f"{prefix}.attn_out"] = attn_out
    saved[f"{prefix}.attn_aux_ctx"] = aux_ctx

    o_out = custom_gemm_attn_out_proj_fwd(attn_out, params[f"{prefix}.attention.wo.weight"])
    if o_out is None:
        o_out = linear(attn_out, params[f"{prefix}.attention.wo.weight"])

    _mup_res = config.mup_residual_scale()
    if _mup_res != 1.0:
        o_out = o_out * _mup_res
    hidden = hidden + o_out
    saved[f"{prefix}.hidden_after_attn"] = hidden

    normed2, rsigma_ffn = rmsnorm_te_with_rsigma(
        hidden, params[f"{prefix}.ffn_norm.weight"]
    )
    saved[f"{prefix}.ffn_normed"] = normed2
    saved[f"{prefix}.rsigma_ffn"] = rsigma_ffn

    fc1_out = custom_gemm_fc1_fwd(normed2, params[f"{prefix}.feed_forward.wfc1.weight"])
    if fc1_out is None:
        fc1_out = linear(normed2, params[f"{prefix}.feed_forward.wfc1.weight"])
    saved[f"{prefix}.fc1_output"] = fc1_out

    intermediate = swiglu(fc1_out)
    saved[f"{prefix}.intermediate"] = intermediate

    down = custom_gemm_fc2_fwd(intermediate, params[f"{prefix}.feed_forward.w2.weight"])
    if down is None:
        down = linear(intermediate, params[f"{prefix}.feed_forward.w2.weight"])
    if _mup_res != 1.0:
        down = down * _mup_res
    hidden = hidden + down
    return hidden


def mtp_forward_pass_with_save(
    params: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    hidden_input: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    need_backward: bool = True,
) -> dict[str, object]:
    """Run the EAGLE-style MTP block (K = ``config.MTP_NUM_LAYERS`` extra heads).

    Mirrors deliver-Megatron's
    ``Megatron-LM_4_deliver/megatron/core/transformer/mtp_block.py``
    + the ``num_eagle_layers != 0`` branch of ``gpt_model.py``. Per
    MTP layer k = 0..K-1:

        input_ids = roll_left_one(input_ids)           # next-token shift
        labels    = roll_left_one(labels)
        loss_mask = roll_left_one(loss_mask)
        decoder_input = embedding(input_ids)           # SHARED with main
        decoder_input = enorm(decoder_input)
        hidden_normed = hnorm(hidden_input)
        hidden = eh_proj(cat([decoder_input_normed, hidden_normed], -1))
        hidden = transformer_layer(hidden)             # one full layer
        # NB: NO final_layernorm — see MTPTransformerBlock.forward()
        if MUP: hidden = hidden / mup_width_scale       # match main muP
        mtp_logits = output_layer(hidden)              # SHARED with main

    NB on the rolling: deliver-Megatron's EAGLE batch packer rolls
    ``input_ids`` / ``labels`` / ``loss_mask`` PER-DOCUMENT inside
    ``EagleBatchPacker._create_mtp`` and ships them as
    ``mtp_tokens`` / ``mtp_labels`` / ``mtp_loss_mask``. Until our
    dataloader is upgraded to surface those, we approximate with a
    sequence-wide ``roll(-1)`` — exact for single-document micro-batches
    and a near-match when the per-document boundary tokens are masked
    out by ``loss_mask`` (which the batch packer already enforces).

    The returned dict carries everything the matching backward needs:
        - ``logits_list``: [Tensor[S,B,V]] per MTP layer
        - ``rolled_labels_list``: [Tensor[B,S]] per MTP layer
        - ``rolled_mask_list``: [Tensor[B,S]] per MTP layer
        - ``rolled_input_ids_list``: [Tensor[B,S]] per MTP layer
        - ``hidden_input_list``: [Tensor[S,B,H]] per layer (input to that
          MTP layer; hidden_input_list[0] == ``hidden_input`` arg, for
          k>0 it's the previous MTP layer's output)
        - ``saved``: per-layer intermediate dict, prefix
          ``mtp.layers.{k}.<sub>``
    """
    cfg = get_config()
    if cfg.mtp_num_layers <= 0:
        raise RuntimeError("mtp_forward_pass_with_save called with MTP_NUM_LAYERS=0")
    saved: dict[str, object] = {}
    logits_list: list[torch.Tensor] = []
    rolled_labels_list: list[torch.Tensor] = []
    rolled_mask_list: list[torch.Tensor] = []
    rolled_input_ids_list: list[torch.Tensor] = []
    hidden_input_list: list[torch.Tensor] = []

    # MTP rolls input_ids along the last dim (which is the sequence
    # axis: input_ids has shape [B, S]). labels and loss_mask follow
    # the same convention.
    cur_ids = input_ids
    cur_labels = labels
    cur_mask = loss_mask
    hidden = hidden_input

    _doc_aware_rope = position_ids is not None and cfg.use_span_based_attn
    if _doc_aware_rope:
        # Reshape rope_freqs the same way forward_pass_with_save does
        # so the per-layer helper sees a sequence-aligned freqs tensor.
        freqs_table = rope_freqs.squeeze(1).squeeze(1)  # [max_seq, D]
        pos_sb = position_ids.transpose(0, 1).contiguous()  # [S, B]
        rope_freqs_layer = freqs_table[pos_sb.long()][:, :, None, :]  # [S,B,1,D]
    else:
        rope_freqs_layer = rope_freqs

    _mup_emb = cfg.mup_emb_scale

    for k in range(cfg.mtp_num_layers):
        prefix = f"mtp.layers.{k}"

        # 1) Shift input_ids / labels / loss_mask. Each iteration shifts
        # again on top of the previous shift, so MTP layer k effectively
        # predicts the (k+1)-th future token.
        cur_ids = _roll_left_one(cur_ids, dim=-1)
        cur_labels = _roll_left_one(cur_labels, dim=-1)
        cur_mask = _roll_left_one(cur_mask, dim=-1)

        rolled_input_ids_list.append(cur_ids)
        rolled_labels_list.append(cur_labels)
        rolled_mask_list.append(cur_mask)
        hidden_input_list.append(hidden)

        # 2) Shared embedding lookup → decoder_input [S, B, H]. We do
        # not save the embedding output itself; the embedding's wgrad
        # will be reconstructed in backward via index_add over the
        # rolled input_ids (mirroring main backward's path).
        decoder_input = params["tok_embeddings.weight"][cur_ids]
        if _mup_emb > 0:
            decoder_input = decoder_input * _mup_emb
        decoder_input = decoder_input.transpose(0, 1).contiguous()  # [S,B,H]
        saved[f"{prefix}.decoder_input"] = decoder_input

        # 3) enorm / hnorm. Both are RMSNorm with weight only, returning
        # rsigma so the same ``rmsnorm_te_backward`` helper used by main
        # layers can drive their backward.
        d_normed, rsigma_e = rmsnorm_te_with_rsigma(
            decoder_input, params[f"{prefix}.enorm.weight"]
        )
        h_normed, rsigma_h = rmsnorm_te_with_rsigma(
            hidden, params[f"{prefix}.hnorm.weight"]
        )
        saved[f"{prefix}.enorm_out"] = d_normed
        saved[f"{prefix}.enorm_rsigma"] = rsigma_e
        saved[f"{prefix}.hnorm_out"] = h_normed
        saved[f"{prefix}.hnorm_rsigma"] = rsigma_h

        # 4) Concatenate and project 2H → H. We use the plain ``linear``
        # path (no custom_gemm fast-path — eh_proj is one-off and tiny
        # compared to the main-loop GEMMs).
        combined = torch.cat([d_normed, h_normed], dim=-1)  # [S,B,2H]
        saved[f"{prefix}.eh_proj_input"] = combined
        proj = linear(combined, params[f"{prefix}.eh_proj.weight"])  # [S,B,H]

        # 5) Transformer layer (reuses the per-layer helper). Layer
        # number is offset by config.NUM_LAYERS so the TE attention
        # cache key does not collide with main-model layers.
        hidden = _mtp_transformer_layer_forward(
            proj,
            params,
            prefix,
            saved,
            rope_freqs_layer,
            layer_number=config.NUM_LAYERS + k + 1,
            doc_aware_rope=_doc_aware_rope,
            need_backward=need_backward,
        )

        # 6) Shared output head. EAGLE-style MTP feeds the transformer
        # output DIRECTLY into the shared output_layer (NO MTP-side
        # final_layernorm), with the same muP width-scale division the
        # main path applies on its own ``output_input``. This matches
        # ``gpt_model.py``:
        #     mtp_hidden = mtp_hidden / (hidden_size / mup_base_hidden_size)
        #     mtp_logits, _ = self.output_layer(mtp_hidden, weight=output_weight)
        saved[f"{prefix}.hidden_after_layer"] = hidden  # pre-muP, post-transformer
        _mup_ws = config.mup_width_scale()
        if _mup_ws != 1.0:
            output_input = hidden / _mup_ws
        else:
            output_input = hidden
        mtp_logits = custom_gemm_output_fwd(output_input, params["output.weight"])
        if mtp_logits is None:
            mtp_logits = linear(output_input, params["output.weight"])
        logits_list.append(mtp_logits)

    return {
        "logits_list": logits_list,
        "rolled_labels_list": rolled_labels_list,
        "rolled_mask_list": rolled_mask_list,
        "rolled_input_ids_list": rolled_input_ids_list,
        "hidden_input_list": hidden_input_list,
        "saved": saved,
    }
