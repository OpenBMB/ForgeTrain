"""Forward pass through the MiniCPM5 8B model.

Architecture per layer:
  residual ─┬─ RMSNorm → QKV linear → RoPE → GQA Attention → Output proj ─┬─ add
            └──────────────────────────────────────────────────────────────┘
  residual ─┬─ RMSNorm → FC1 linear → SwiGLU → FC2 linear ───────────────┬─ add
            └──────────────────────────────────────────────────────────────┘
  → Final RMSNorm → Output layer

All linear layers: weight @ input (no bias, --disable-bias-linear).
Internal layout: [S, B, H] (sequence-first, matching baseline).

TP communication (no sequence-parallel):
  - Embedding: vocab-parallel lookup → all-reduce across TP
  - QKV: column-parallel (no communication)
  - Proj: row-parallel → all-reduce across TP
  - FC1: column-parallel (no communication)
  - FC2: row-parallel → all-reduce across TP
  - Output: column-parallel, vocab-split logits (no gather)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

__all__ = ["forward_pass", "forward_pass_with_save"]
import torch.distributed as dist

from . import config
from .engine_config import get_config
from .parameters import build_fwd_layer_params
from .kernels import (
    apply_rope,
    causal_attention,
    fused_residual_add_rmsnorm_fwd,
    fused_rope_from_qkv,
    rmsnorm_forward,
    swiglu,
)
from .custom_gemm import (
    custom_gemm_attn_out_proj_fwd,
    custom_gemm_fc1_fwd,
    custom_gemm_fc2_fwd,
    custom_gemm_output_fwd,
    custom_gemm_qkv_proj_fwd,
)


def _linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, w.t())


def _embed(w: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return w[ids]


def forward_pass(
    params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    save_for_backward: bool = False,
    recompute_attn: bool = False,
    recompute_mlp: bool = False,
    tp_group: object = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    ag_events: Optional[Dict[int, "torch.cuda.Event"]] = None,
    ag_layers_per_bucket: int = 0,
    ag_num_buckets: int = 0,
    _precomputed_layer_params=None,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """Full transformer forward pass.

    Args:
        params: BF16 parameter dict (from load_megatron_checkpoint).
        input_ids: [B, S] token ids.
        rope_cos, rope_sin: [S, D] FP32 RoPE tables.
        save_for_backward: if True, store activations for backward.
        recompute_attn: if True (and save_for_backward=True), save q/k/v
            instead of autograd attention graphs, trading compute for memory.
        recompute_mlp: if True (and save_for_backward=True), skip saving
            fc1_out/act_out/norm_out_mlp; backward recomputes them.
        tp_group: tensor-parallel process group (None for TP=1).
        tp_rank: this rank's position within the TP group.
        tp_size: tensor-parallel world size (1 = no TP).
        ag_events: per-bucket CUDA events from a previous step's async
            all-gather.  Forward waits for the relevant bucket before
            consuming its params, enabling AG-forward overlap.
        ag_layers_per_bucket: layers per AG bucket (for bucket index calc).
        ag_num_buckets: total number of AG buckets.

    Returns:
        logits: [S, B, V_local] BF16 logits (V_local = V/tp_size when TP>1).
        saved: dict of saved tensors (or None if save_for_backward=False).
    """
    cfg = get_config()
    num_layers = config.NUM_LAYERS
    num_heads = config.NUM_HEADS // tp_size
    num_kv_heads = config.NUM_KV_HEADS // tp_size
    head_dim = config.HEAD_DIM
    heads_per_group = num_heads // num_kv_heads
    eps = config.NORM_EPSILON
    norm_fn = rmsnorm_forward

    saved = {} if save_for_backward else None

    # --- Wait for previous step's AG (embedding bucket) ---
    if ag_events is not None and ag_num_buckets > 0:
        _emb_bucket = ag_num_buckets - 1
        if _emb_bucket in ag_events:
            torch.cuda.current_stream().wait_event(ag_events[_emb_bucket])

    # --- Embedding ---
    emb_weight = params["embedding.word_embeddings.weight"]  # [V_local, H]
    if tp_size > 1:
        V_local = emb_weight.shape[0]
        vocab_start = tp_rank * V_local
        input_mask = (input_ids < vocab_start) | (input_ids >= vocab_start + V_local)
        masked_ids = (input_ids - vocab_start).clone()
        masked_ids[input_mask] = 0
        hidden = _embed(emb_weight, masked_ids)  # [B, S, H]
        hidden[input_mask] = 0.0
        hidden = hidden.transpose(0, 1).contiguous()  # [S, B, H]
        dist.all_reduce(hidden, group=tp_group)
    else:
        hidden = _embed(emb_weight, input_ids)  # [B, S, H]
        hidden = hidden.transpose(0, 1).contiguous()  # [S, B, H]

    if saved is not None:
        saved["embedding_input"] = input_ids
        saved["layers"] = []

    if _precomputed_layer_params is not None:
        _fwd_layer_params = _precomputed_layer_params
    else:
        _fwd_layer_params = build_fwd_layer_params(params, num_layers)
    use_tp = tp_size > 1

    # Pre-compute layer 0's attention pre-norm when MLP residual fusion is active
    # (layer 0 has no preceding MLP residual add to fuse with)
    _mlp_res_norm_out = None
    _mlp_res_rsigma = None
    _mlp_res_hidden_pre = None
    if cfg.fused_mlp_residual_rmsnorm:
        _mlp_res_hidden_pre = hidden
        _mlp_res_norm_out, _mlp_res_rsigma = norm_fn(hidden, _fwd_layer_params[0][0], eps)

    # --- Transformer layers ---
    for i in range(num_layers):
        # Wait for previous step's AG when crossing a bucket boundary
        if (ag_events is not None and ag_layers_per_bucket > 0
                and i > 0 and i % ag_layers_per_bucket == 0):
            _bucket_idx = ag_num_buckets - 1 - i // ag_layers_per_bucket
            if _bucket_idx in ag_events:
                torch.cuda.current_stream().wait_event(ag_events[_bucket_idx])

        ln_w, qkv_w, proj_w, mlp_ln_w, fc1_w, fc2_w = _fwd_layer_params[i]

        # ── Self-attention sub-layer ──

        if cfg.fused_mlp_residual_rmsnorm:
            hidden_pre_attn = _mlp_res_hidden_pre
            norm_out = _mlp_res_norm_out
            rsigma_attn = _mlp_res_rsigma
        else:
            hidden_pre_attn = hidden
            norm_out, rsigma_attn = norm_fn(hidden, ln_w, eps)

        norm_out_full = norm_out

        _custom = custom_gemm_qkv_proj_fwd(norm_out_full, qkv_w)
        qkv = _custom if _custom is not None else _linear(norm_out_full, qkv_w)  # [S, B, Dqkv_local]

        S, B, _ = qkv.shape
        group_size = (heads_per_group + 2) * head_dim
        qkv = qkv.view(S, B, num_kv_heads, group_size)
        q_per_group = heads_per_group * head_dim

        # ROPE_BHSD path: RoPE writes q/k directly in [B,H,S,D] contig
        # and extracts v into [B,Hkv,S,D] contig.  Skips the q/k permute()
        # .contiguous() in the attention fwd entry (~34 MB / call).  Only
        # active when DIRECT_ATTN is also on — the default autograd path
        # always consumes [S,B,H,D] via F.sdpa.
        _use_bhsd = (
            cfg.rope_bhsd and cfg.direct_attn
            and cfg.fused_rope_qkv and qkv.is_cuda
            and save_for_backward and not recompute_attn
        )
        if _use_bhsd:
            from .kernels import fused_rope_from_qkv_bhsd
            q, k, v = fused_rope_from_qkv_bhsd(
                qkv, rope_cos, rope_sin,
                num_heads, num_kv_heads, head_dim,
            )  # [B,Hq,S,D], [B,Hkv,S,D], [B,Hkv,S,D] all contig
        elif cfg.fused_rope_qkv and qkv.is_cuda:
            q, k = fused_rope_from_qkv(
                qkv, rope_cos, rope_sin,
                num_heads, num_kv_heads, head_dim,
            )
            v = qkv[..., q_per_group + head_dim :]
        else:
            q = qkv[..., :q_per_group]
            k = qkv[..., q_per_group : q_per_group + head_dim]
            v = qkv[..., q_per_group + head_dim :]
            q = q.reshape(S, B, num_heads, head_dim)
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Attention forward.  Three paths:
        #   (a) DIRECT_ATTN + DSL_ATTN_FWD : self-developed CuTeDSL fwd
        #       (cuDNN bwd via aten op).  Requires save_for_backward.
        #   (b) DIRECT_ATTN only           : aten cuDNN fwd + aten cuDNN
        #       bwd, no autograd dispatch.
        #   (c) default                    : F.sdpa under enable_grad,
        #       backward via the autograd engine on the saved attn_out.
        # Recompute path stays on the F.sdpa kernel (no save).
        _attn_saved_state = None
        if save_for_backward and not recompute_attn and cfg.direct_attn:
            if cfg.dsl_attn_fwd:
                from .kernels import causal_attention_fwd_dsl  # late import
                proj_input, _attn_saved_state = causal_attention_fwd_dsl(
                    q, k, v, inputs_already_bhsd=_use_bhsd,
                )
            else:
                from .kernels import causal_attention_fwd_direct
                proj_input, _attn_saved_state = causal_attention_fwd_direct(
                    q, k, v, inputs_already_bhsd=_use_bhsd,
                )
        elif save_for_backward and not recompute_attn:
            with torch.enable_grad():
                q_ag = q.detach().requires_grad_(True)
                k_ag = k.detach().requires_grad_(True)
                v_ag = v.detach().requires_grad_(True)
                attn_out_ag = causal_attention(
                    q_ag, k_ag, v_ag, layer_number=i + 1,
                )
            proj_input = attn_out_ag.detach()
        else:
            proj_input = causal_attention(q, k, v, layer_number=i + 1)

        _custom = custom_gemm_attn_out_proj_fwd(proj_input, proj_w)
        attn_out = _custom if _custom is not None else _linear(proj_input, proj_w)  # row-parallel: [S, B, H]
        if use_tp:
            dist.all_reduce(attn_out, group=tp_group)

        # ── MLP sub-layer ──
        if cfg.fused_residual_rmsnorm:
            hidden, norm_out_mlp, rsigma_mlp = fused_residual_add_rmsnorm_fwd(
                hidden, attn_out, mlp_ln_w, eps,
            )
            hidden_pre_mlp = hidden
        else:
            hidden = hidden + attn_out
            hidden_pre_mlp = hidden
            norm_out_mlp, rsigma_mlp = norm_fn(hidden, mlp_ln_w, eps)

        norm_out_mlp_full = norm_out_mlp

        _custom_fc1 = custom_gemm_fc1_fwd(norm_out_mlp_full, fc1_w)
        fc1_out = _custom_fc1 if _custom_fc1 is not None else _linear(norm_out_mlp_full, fc1_w)
        act_out = swiglu(fc1_out)
        _custom_fc2 = custom_gemm_fc2_fwd(act_out, fc2_w)  # row-parallel: [S, B, H]
        mlp_out = _custom_fc2 if _custom_fc2 is not None else _linear(act_out, fc2_w)
        if use_tp:
            dist.all_reduce(mlp_out, group=tp_group)

        if cfg.fused_mlp_residual_rmsnorm and i < num_layers - 1:
            next_ln_w = _fwd_layer_params[i + 1][0]
            hidden, _mlp_res_norm_out, _mlp_res_rsigma = fused_residual_add_rmsnorm_fwd(
                hidden, mlp_out, next_ln_w, eps,
            )
            _mlp_res_hidden_pre = hidden
        else:
            hidden = hidden + mlp_out

        if saved is not None:
            # Tuple layout (13 elements) — consumed by backward.py:
            #  0: hidden_pre_attn  1: norm_out_attn   2: rsigma_attn
            #  3: proj_input       4: hidden_pre_mlp   5: rsigma_mlp
            #  6: fc1_out|None     7: act_out|None     8: norm_out_mlp|None
            #  9: q_ag|q          10: k_ag|k          11: v_ag|v
            # 12: attn_out_ag | None | tuple
            #     - tensor (attn_out_ag): autograd path
            #     - None: recompute attn path
            #     - tuple: DIRECT_ATTN / DSL_ATTN_FWD path; opaque saved
            #              state for kernels.causal_attention_bwd_direct.
            _save_ag = save_for_backward and not recompute_attn
            if _save_ag and _attn_saved_state is not None:
                # DIRECT_ATTN / DSL fwd path — slot 12 is the saved-state
                # tuple; q/k/v slots are unused (bwd reads from the tuple).
                _slot12 = _attn_saved_state
                _slot9, _slot10, _slot11 = None, None, None
            elif _save_ag:
                _slot12 = attn_out_ag
                _slot9, _slot10, _slot11 = q_ag, k_ag, v_ag
            else:
                _slot12 = None
                _slot9, _slot10, _slot11 = q, k, v
            saved["layers"].append((
                hidden_pre_attn, norm_out_full, rsigma_attn, proj_input,
                hidden_pre_mlp, rsigma_mlp,
                fc1_out if not recompute_mlp else None,
                act_out if not recompute_mlp else None,
                norm_out_mlp_full if not recompute_mlp else None,
                _slot9, _slot10, _slot11,
                _slot12,
            ))

    # Wait for bucket 0 (output_layer + final_ln + last layers in backward order)
    if ag_events is not None and 0 in ag_events:
        torch.cuda.current_stream().wait_event(ag_events[0])

    # --- Final LayerNorm ---
    final_ln_w = params["decoder.final_layernorm.weight"]
    hidden_pre_final = hidden
    hidden, rsigma_final = norm_fn(hidden, final_ln_w, eps)

    hidden_final_full = hidden

    if saved is not None:
        saved["hidden_pre_final"] = hidden_pre_final
        saved["hidden_final"] = hidden_final_full
        saved["rsigma_final"] = rsigma_final

    # --- Output layer (column-parallel, vocab-split logits) ---
    output_w = params["output_layer.weight"]  # [V_local, H]
    _custom = custom_gemm_output_fwd(hidden_final_full, output_w)
    logits = _custom if _custom is not None else _linear(hidden_final_full, output_w)  # [S, B, V_local]

    return logits, saved


def forward_pass_with_save(
    params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    rope_freqs: Tuple[torch.Tensor, torch.Tensor],
    *,
    tp_group: object = None,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """Wrapper matching the documented API: receives rope_freqs tuple."""
    rope_cos, rope_sin = rope_freqs
    return forward_pass(
        params, input_ids, rope_cos, rope_sin,
        save_for_backward=True, tp_group=tp_group,
        tp_rank=tp_rank, tp_size=tp_size,
    )
