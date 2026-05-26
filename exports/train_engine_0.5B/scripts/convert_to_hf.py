#!/usr/bin/env python3
"""Convert a self-developed framework training_state.pt to HuggingFace format.

Usage:
    python convert_to_hf.py --input /path/to/step_0005000 --output /path/to/hf_model
    python convert_to_hf.py --input /path/to/training_state.pt --output /path/to/hf_model --fp32

The script:
  1. Loads the checkpoint (BF16 params by default, FP32 master weights with --fp32)
  2. Splits fused QKV and gate/up projections
  3. Remaps parameter names to match HuggingFace MiniCPMForCausalLM
  4. Writes model.safetensors + config.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

# ── Architecture constants from model_spec.toml (L1 SSOT) ─────────────
_SPEC_PATH = Path(__file__).resolve().parent.parent / "model_spec.toml"
with open(_SPEC_PATH, "rb") as _f:
    _SPEC = tomllib.load(_f)
_M = _SPEC["model"]

NUM_LAYERS: int = int(_M["num_layers"])
HIDDEN_SIZE: int = int(_M["hidden_size"])
NUM_HEADS: int = int(_M["num_attention_heads"])
NUM_KV_HEADS: int = int(_M["num_query_groups"])
HEAD_DIM: int = HIDDEN_SIZE // NUM_HEADS
FFN_HIDDEN_SIZE: int = int(_M["ffn_hidden_size"])
VOCAB_SIZE: int = int(_M["vocab_size"])
NORM_EPS: float = float(_M["norm_epsilon"])
ROPE_THETA: float = float(_M["rotary_base"])
MAX_POSITION_EMBEDDINGS: int = 32768


def _resolve_checkpoint_path(input_path: str) -> str:
    if os.path.isfile(input_path):
        return input_path
    candidate = os.path.join(input_path, "training_state.pt")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"Cannot find training_state.pt at {input_path} or {candidate}"
    )


def _split_qkv_interleaved(
    wqkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split interleaved GQA QKV weight [Q_dim + 2*KV_dim, H] → q, k, v.

    Interleave layout per group: [Q_heads_in_group..., K_head, V_head]
      nq_per_kv = NUM_HEADS // NUM_KV_HEADS = 8
      group_width = (8 + 2) * 64 = 640
      Group 0 rows 0..639:   Q[0:8] (512) | K[0] (64) | V[0] (64)
      Group 1 rows 640..1279: Q[8:16] (512) | K[1] (64) | V[1] (64)
    """
    nq_per_kv = NUM_HEADS // NUM_KV_HEADS
    group_w = (nq_per_kv + 2) * HEAD_DIM
    grouped = wqkv.view(NUM_KV_HEADS, group_w, HIDDEN_SIZE)

    q_w = nq_per_kv * HEAD_DIM
    q = grouped[:, :q_w, :].reshape(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)
    k = grouped[:, q_w : q_w + HEAD_DIM, :].reshape(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
    v = grouped[:, q_w + HEAD_DIM :, :].reshape(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
    return q, k, v


def _split_gate_up(wfc1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split fused gate+up [2*ffn, H] → gate [ffn, H], up [ffn, H]."""
    gate, up = wfc1.chunk(2, dim=0)
    return gate, up


def convert(
    ckpt_path: str,
    output_dir: str,
    *,
    use_fp32: bool = False,
    dtype_override: torch.dtype | None = None,
) -> None:
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", "?")
    print(f"  Checkpoint step: {step}")

    if use_fp32 and "optimizer_state" in ckpt:
        print("  Using FP32 master weights from optimizer_state")
        src = ckpt["optimizer_state"]["master_weights"]
    else:
        if use_fp32:
            print("  WARNING: --fp32 requested but no optimizer_state found; "
                  "falling back to params")
        src = ckpt["params"]

    out_dtype = dtype_override or torch.bfloat16

    hf_state: dict[str, torch.Tensor] = {}

    # ── Global tensors ──────────────────────────────────────────────────
    embed_w = src["tok_embeddings.weight"].to(out_dtype).contiguous()
    hf_state["model.embed_tokens.weight"] = embed_w
    # MiniCPM4-0.5B ties lm_head ↔ embed_tokens (434M params confirms this),
    # but we save lm_head explicitly from the trained output.weight so the
    # values reflect any drift that may have happened during training.
    hf_state["lm_head.weight"] = src["output.weight"].to(out_dtype).contiguous()
    hf_state["model.norm.weight"] = src["norm.weight"].to(out_dtype).contiguous()

    # ── Per-layer tensors ───────────────────────────────────────────────
    for i in range(NUM_LAYERS):
        p_src = f"layers.{i}"
        p_hf = f"model.layers.{i}"

        hf_state[f"{p_hf}.input_layernorm.weight"] = (
            src[f"{p_src}.attention_norm.weight"].to(out_dtype).contiguous()
        )
        hf_state[f"{p_hf}.post_attention_layernorm.weight"] = (
            src[f"{p_src}.ffn_norm.weight"].to(out_dtype).contiguous()
        )

        # QKV split
        wqkv = src[f"{p_src}.attention.wqkv.weight"]
        q, k, v = _split_qkv_interleaved(wqkv)
        hf_state[f"{p_hf}.self_attn.q_proj.weight"] = q.to(out_dtype).contiguous()
        hf_state[f"{p_hf}.self_attn.k_proj.weight"] = k.to(out_dtype).contiguous()
        hf_state[f"{p_hf}.self_attn.v_proj.weight"] = v.to(out_dtype).contiguous()

        # O proj (direct rename)
        hf_state[f"{p_hf}.self_attn.o_proj.weight"] = (
            src[f"{p_src}.attention.wo.weight"].to(out_dtype).contiguous()
        )

        # gate/up split
        wfc1 = src[f"{p_src}.feed_forward.wfc1.weight"]
        gate, up = _split_gate_up(wfc1)
        hf_state[f"{p_hf}.mlp.gate_proj.weight"] = gate.to(out_dtype).contiguous()
        hf_state[f"{p_hf}.mlp.up_proj.weight"] = up.to(out_dtype).contiguous()

        # down proj (direct rename)
        hf_state[f"{p_hf}.mlp.down_proj.weight"] = (
            src[f"{p_src}.feed_forward.w2.weight"].to(out_dtype).contiguous()
        )

    # ── Check whether embed and lm_head are identical ───────────────────
    tied = torch.equal(hf_state["model.embed_tokens.weight"],
                       hf_state["lm_head.weight"])
    print(f"  embed_tokens == lm_head (tied): {tied}")

    # ── Save weights ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    total_params = sum(t.numel() for t in hf_state.values())
    print(f"  Total parameters: {total_params:,} ({out_dtype})")

    try:
        from safetensors.torch import save_file
        safetensors_path = os.path.join(output_dir, "model.safetensors")
        save_file(hf_state, safetensors_path)
        print(f"  Saved: {safetensors_path}")
    except ImportError:
        print("  safetensors not installed; falling back to pytorch_model.bin")
        bin_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(hf_state, bin_path)
        print(f"  Saved: {bin_path}")

    # ── Write config.json ───────────────────────────────────────────────
    # Mirrors https://huggingface.co/openbmb/MiniCPM4-0.5B/blob/main/config.json
    vocab_size = hf_state["model.embed_tokens.weight"].shape[0]
    config = {
        "_name_or_path": "minicpm4-0.5b-converted",
        "architectures": ["MiniCPMForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_minicpm.MiniCPMConfig",
            "AutoModel": "modeling_minicpm.MiniCPMModel",
            "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
        },
        "bos_token_id": 1,
        "eos_token_id": [2, 73440],
        "hidden_act": "silu",
        "hidden_size": HIDDEN_SIZE,
        "initializer_range": 0.1,
        "intermediate_size": FFN_HIDDEN_SIZE,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "model_type": "minicpm",
        "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": NUM_LAYERS,
        "num_key_value_heads": NUM_KV_HEADS,
        "rms_norm_eps": NORM_EPS,
        "rope_theta": ROPE_THETA,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": tied,
        "use_cache": True,
        "vocab_size": vocab_size,
        "scale_emb": 12,
        "dim_model_base": 256,
        "scale_depth": 1.4,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {config_path}")

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\nConversion complete → {output_dir}")
    print(f"  Step:       {step}")
    print(f"  Params:     {total_params:,}")
    print(f"  Dtype:      {out_dtype}")
    print(f"  Tied:       {tied}")
    print(f"  Vocab size: {vocab_size}")
    if not tied:
        print("  NOTE: lm_head.weight ≠ embed_tokens.weight — "
              "set tie_word_embeddings=false in config.json (already done).")
    print("\nTo load in HuggingFace (requires openbmb/MiniCPM4-0.5B modeling code):")
    print('  from transformers import AutoModelForCausalLM')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}", '
          f'trust_remote_code=True)')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert training_state.pt → HuggingFace MiniCPM4 0.5B"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to training_state.pt or its parent step directory",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for HuggingFace checkpoint",
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Use FP32 master weights from optimizer state instead of BF16 params",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
        help="Output tensor dtype (default: bf16)",
    )
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    ckpt_path = _resolve_checkpoint_path(args.input)
    convert(
        ckpt_path,
        args.output,
        use_fp32=args.fp32,
        dtype_override=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
