"""Parameter loading, self-init, and checkpoint save/resume.

Naming convention: **fused mcore layout**, matching Megatron 0.16's
``--use-mcore-models`` default — the per-block RMSNorm gain is registered
inside the column-parallel linear it precedes (TELayerNormColumnParallelLinear),
so the state_dict keys look like:

  decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight  → attn RMSNorm gain
  decoder.layers.{i}.self_attention.linear_qkv.weight             → fused QKV [Dqkv, H]
  decoder.layers.{i}.self_attention.linear_proj.weight            → output proj [H, H]
  decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight             → MLP RMSNorm gain
  decoder.layers.{i}.mlp.linear_fc1.weight                        → fused gate+up [2*FFN, H]
  decoder.layers.{i}.mlp.linear_fc2.weight                        → down proj [H, FFN]
  embedding.word_embeddings.weight                                 → [V, H]
  decoder.final_layernorm.weight                                   → [H]
  output_layer.weight                                              → [V, H]

This lets engine-side state-dicts interchange directly with Megatron-0.16-mcore
mcore checkpoints (no key translation needed) and is the prerequisite for
``self_init_params(...)`` to bitwise-match a fresh Megatron 0.16 mcore init
from seed=1234 (the init order for fused layer_norm_weight + linear weight
within ``TELayerNormColumnParallelLinear`` matters for global RNG state
evolution).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "load_megatron_checkpoint", "trainable_param_names",
    "self_init_params", "save_checkpoint", "load_resume_checkpoint",
    "build_fwd_layer_params", "build_bwd_precomputed",
]

from . import config


# TP sharding: column-parallel (axis 0) and row-parallel (axis 1)
_TP_COLUMN_SUFFIXES = [
    "embedding.word_embeddings.weight",
    "output_layer.weight",
    "self_attention.linear_qkv.weight",
    "mlp.linear_fc1.weight",
]

_TP_ROW_SUFFIXES = [
    "self_attention.linear_proj.weight",
    "mlp.linear_fc2.weight",
]


def _tp_shard_axis(key: str) -> Optional[int]:
    """Return the axis to shard for TP, or None if not sharded."""
    for suffix in _TP_COLUMN_SUFFIXES:
        if key.endswith(suffix):
            return 0
    for suffix in _TP_ROW_SUFFIXES:
        if key.endswith(suffix):
            return 1
    return None


def _tp_shard_weight(
    w: torch.Tensor, key: str, tp_rank: int, tp_size: int,
) -> torch.Tensor:
    """Shard a weight tensor for TP, matching Megatron's axis-based split."""
    axis = _tp_shard_axis(key)
    if axis is None:
        return w
    chunk = w.shape[axis] // tp_size
    return w.narrow(axis, tp_rank * chunk, chunk).contiguous()


# ---------------------------------------------------------------------------
# Per-layer key suffixes (FUSED layout).  Order matters — it's mirrored by
# ``trainable_param_names()`` and ``_canonical_buffer_order`` (below), which
# reverses + interleaves these to reconstruct Megatron's ParamAndGradBuffer
# layout.  Keep the fc1.layer_norm_weight EARLIER than fc1.weight, and
# qkv.layer_norm_weight EARLIER than qkv.weight, because inside
# ``TELayerNormColumnParallelLinear`` the layer_norm_weight is registered
# first via ``register_parameter`` and so it appears first in
# ``named_parameters()``.
# ---------------------------------------------------------------------------
_LAYER_KEYS = [
    "self_attention.linear_qkv.layer_norm_weight",
    "self_attention.linear_qkv.weight",
    "self_attention.linear_proj.weight",
    "mlp.linear_fc1.layer_norm_weight",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc2.weight",
]


# Legacy ``canonical_state_fp32.pt`` (unfused layout, where
# input_layernorm.weight / pre_mlp_layernorm.weight are stored as their
# own state_dict entries) → fused layout the rest of this codebase now
# uses.  Drop this remap when the legacy ckpt is no longer needed.
_LEGACY_KEY_REMAP: Dict[str, str] = {
    "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
    "pre_mlp_layernorm.weight": "mlp.linear_fc1.layer_norm_weight",
}


def load_megatron_checkpoint(
    checkpoint_root: str,
    device: str,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Load ``canonical_state_fp32.pt`` and return a BF16 parameter dict.

    The on-disk file uses the unfused layout — we translate to the fused
    layout this engine now uses (see ``_LEGACY_KEY_REMAP``).  Only layers
    ``0 .. NUM_LAYERS-1`` are loaded.  For ``TP>1``, weights are sharded
    along the appropriate axis.  New code should prefer
    ``self_init_params(...)``, which doesn't depend on any on-disk file
    and gives a fresh Megatron-0.16-mcore-equivalent init from seed alone.
    """
    fp32_path = os.path.join(checkpoint_root, "canonical_state_fp32.pt")
    ckpt = torch.load(fp32_path, map_location="cpu", weights_only=False)

    num_layers = config.NUM_LAYERS
    params: Dict[str, torch.Tensor] = {}

    _GLOBAL_KEYS = [
        "embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "output_layer.weight",
    ]

    for gk in _GLOBAL_KEYS:
        if gk not in ckpt:
            raise KeyError(f"Missing global key in checkpoint: {gk}")
        w = ckpt[gk]
        if tp_size > 1:
            w = _tp_shard_weight(w, gk, tp_rank, tp_size)
        params[gk] = w.to(torch.bfloat16).to(device)

    for i in range(num_layers):
        prefix = f"decoder.layers.{i}"
        for lk in _LAYER_KEYS:
            # The on-disk ckpt uses the LEGACY (unfused) suffix for the two
            # layer-norm gains; translate the suffix back when reading.
            legacy_suffix = next(
                (legacy for legacy, fused in _LEGACY_KEY_REMAP.items()
                 if fused == lk),
                lk,
            )
            ckpt_key = f"{prefix}.{legacy_suffix}"
            fused_key = f"{prefix}.{lk}"
            if ckpt_key not in ckpt:
                raise KeyError(
                    f"Missing layer key in checkpoint: {ckpt_key} "
                    f"(target fused key: {fused_key})"
                )
            w = ckpt[ckpt_key]
            if tp_size > 1:
                # _tp_shard_weight axis-classifies by KEY name, so feed it the
                # fused key (which is what its suffix table is keyed on).
                w = _tp_shard_weight(w, fused_key, tp_rank, tp_size)
            params[fused_key] = w.to(torch.bfloat16).to(device)

    del ckpt
    return params


def trainable_param_names() -> List[str]:
    """Return all trainable parameter names (for NUM_LAYERS layers).

    Order: embedding → per-layer (fused suffixes in `_LAYER_KEYS` order) →
    final_layernorm → output_layer.  This is the ``forward-pass natural``
    order; ``_canonical_buffer_order`` further reorders it to match the
    Megatron ParamAndGradBuffer slot order.
    """
    names: List[str] = ["embedding.word_embeddings.weight"]
    for i in range(config.NUM_LAYERS):
        for suffix in _LAYER_KEYS:
            names.append(f"decoder.layers.{i}.{suffix}")
    names.append("decoder.final_layernorm.weight")
    names.append("output_layer.weight")
    return names


def _canonical_buffer_order(trainable_names: List[str]) -> List[str]:
    """Return trainable names in Megatron ParamAndGradBuffer order.

    Megatron's buffer is the reverse of ``model.named_parameters()``.
    Within each layer, Megatron's forward order is:
      proj, qkv_layer_norm, qkv, fc1_layer_norm, fc1, fc2
    (due to TELayerNormColumnParallelLinear registering ``layer_norm_weight``
    BEFORE ``weight`` inside the same fused module).  We now match this
    naming bit-for-bit since the engine-side layout switched to the fused
    mcore convention.
    """
    _MEG_LAYER_SUFFIXES = [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.layer_norm_weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.layer_norm_weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
    ]
    num_layers = config.NUM_LAYERS
    megatron_fwd = []
    megatron_fwd.append("embedding.word_embeddings.weight")
    for i in range(num_layers):
        for suffix in _MEG_LAYER_SUFFIXES:
            megatron_fwd.append(f"decoder.layers.{i}.{suffix}")
    megatron_fwd.append("decoder.final_layernorm.weight")
    megatron_fwd.append("output_layer.weight")
    return list(reversed(megatron_fwd))


# ---------------------------------------------------------------------------
# Self-init (no-checkpoint) path — replaces ``load_megatron_checkpoint`` for
# fresh-init runs.  The init recipe matches the established Llama-style
# mcore reference implementation:
#
#   * Linear weights drawn from N(0, init_method_std) in BF16 directly on
#     GPU using CUDA RNG.
#   * The two RESIDUAL projections per layer — ``linear_proj`` (attention
#     output) and ``linear_fc2`` (MLP down) — use a SCALED stddev,
#     std = init_method_std / sqrt(2 * num_layers).
#   * ``output_layer.weight`` (the LM head) uses the UNSCALED init.
#   * ``embedding.word_embeddings.weight`` uses the same UNSCALED init.
#   * Layer-norm gains are initialized to 1.0 (no RNG draw).
#
# Determinism / TP semantics:  This replicates Megatron's default GPU-init
# path (``use_cpu_initialization=False``).  Each TP rank seeds its own CUDA
# RNG with ``seed + 2718 + tp_rank`` (the model-parallel seed from
# ``model_parallel_cuda_manual_seed``), then draws **partition-local**
# tensors directly in BF16 on GPU.  This produces byte-identical weights
# to a fresh Megatron 0.16 mcore init under ``--seed 1234``.
#
# Init order mirrors mcore's ``TransformerLayer`` constructor:
#   embedding → per-layer(linear_proj → linear_qkv → linear_fc1
#   → linear_fc2) → output_layer.
# ---------------------------------------------------------------------------

_RESIDUAL_SUBLAYERS_PER_BLOCK = 2.0


def _residual_init_std(init_method_std: float, num_layers: int) -> float:
    """Mcore's residual scaling: σ_res = σ / sqrt(2 * num_layers)."""
    return float(init_method_std) / math.sqrt(_RESIDUAL_SUBLAYERS_PER_BLOCK * num_layers)


def self_init_params(
    *,
    device: str,
    seed: int = 1234,
    init_method_std: float = 0.02,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Generate a full BF16 parameter dict from scratch — no on-disk file.

    Replicates Megatron 0.16 mcore's GPU-init path: each TP rank seeds
    CUDA RNG with ``seed + 2718 + tp_rank`` and draws partition-local
    BF16 tensors directly on GPU.  This produces byte-identical weights
    to a fresh Megatron 0.16 init under the same seed.

    Returns the same key set as ``trainable_param_names()`` (fused mcore
    layout), with each tensor in bf16 on ``device``.
    """
    H = config.HIDDEN_SIZE
    V = config.padded_vocab_size()
    L = config.NUM_LAYERS
    head_dim = config.HEAD_DIM
    num_heads = config.NUM_HEADS
    num_kv_heads = config.NUM_KV_HEADS
    ffn = config.FFN_HIDDEN_SIZE

    sigma_main = float(init_method_std)
    sigma_res = _residual_init_std(init_method_std, L)

    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim   # 4608 for our config
    o_dim = num_heads * head_dim                          # 4096

    # Partition-local sizes (each TP rank generates only its own shard)
    V_part = V // tp_size
    qkv_part = qkv_dim // tp_size
    o_part = o_dim // tp_size
    ffn_part = ffn // tp_size
    fc1_part = (2 * ffn) // tp_size

    # Replicate Megatron's model-parallel CUDA RNG seeding:
    #   model_parallel_cuda_manual_seed(seed) →
    #     tensor_model_parallel_seed = seed + 2718 + tp_rank
    #   _CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", tensor_model_parallel_seed)
    # Each _initialize_affine_weight_gpu call forks this state.
    # We replicate by setting CUDA seed once and drawing sequentially.
    model_parallel_seed = seed + 2718 + tp_rank
    orig_cuda_rng_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(model_parallel_seed)

    def _draw_gpu(shape: tuple, std: float) -> torch.Tensor:
        """Draw BF16 tensor on GPU using current CUDA RNG (model-parallel)."""
        t = torch.empty(*shape, dtype=torch.bfloat16, device=device)
        t.normal_(mean=0.0, std=std)
        return t

    params: Dict[str, torch.Tensor] = {}

    # ── 1. Embedding [V/tp, H] ─────────────────────────────────────────────
    params["embedding.word_embeddings.weight"] = _draw_gpu((V_part, H), sigma_main)

    # ── 2. Per-transformer-layer ───────────────────────────────────────────
    for i in range(L):
        prefix = f"decoder.layers.{i}"

        # 2a. linear_proj (row-parallel) [H, o_dim/tp]
        params[f"{prefix}.self_attention.linear_proj.weight"] = (
            _draw_gpu((H, o_part), sigma_res)
        )

        # 2b. linear_qkv.layer_norm_weight — RMSNorm gain, ones [H]
        params[f"{prefix}.self_attention.linear_qkv.layer_norm_weight"] = (
            torch.ones(H, dtype=torch.bfloat16, device=device)
        )

        # 2c. linear_qkv (column-parallel) [qkv_dim/tp, H]
        params[f"{prefix}.self_attention.linear_qkv.weight"] = (
            _draw_gpu((qkv_part, H), sigma_main)
        )

        # 2d. mlp.linear_fc1.layer_norm_weight — RMSNorm gain, ones [H]
        params[f"{prefix}.mlp.linear_fc1.layer_norm_weight"] = (
            torch.ones(H, dtype=torch.bfloat16, device=device)
        )

        # 2e. mlp.linear_fc1 (column-parallel) [2*ffn/tp, H]
        params[f"{prefix}.mlp.linear_fc1.weight"] = (
            _draw_gpu((fc1_part, H), sigma_main)
        )

        # 2f. mlp.linear_fc2 (row-parallel) [H, ffn/tp], SCALED residual init
        params[f"{prefix}.mlp.linear_fc2.weight"] = (
            _draw_gpu((H, ffn_part), sigma_res)
        )

    # ── 3. Final RMSNorm ───────────────────────────────────────────────────
    params["decoder.final_layernorm.weight"] = torch.ones(
        H, dtype=torch.bfloat16, device=device,
    )

    # ── 4. Output layer (LM head) [V/tp, H], column-parallel ──────────────
    params["output_layer.weight"] = _draw_gpu((V_part, H), sigma_main)

    # Restore original CUDA RNG state so callers are unaffected.
    torch.cuda.set_rng_state(orig_cuda_rng_state)

    return params


# ---------------------------------------------------------------------------
# M1 Resume — checkpoint save/load.
#
# Each rank saves its own file ``rank_<global_rank>.pt`` containing:
#   - BF16 params (TP-sharded, same across DP ranks after all-gather)
#   - FP32 optimizer shard (per DP rank — fp32_master, exp_avg, exp_avg_sq)
#   - Adam step count, training step, num_samples
#   - (Optional) dataloader sampler state
#
# On resume, each rank loads its own file, reconstructs the param dict
# and optimizer state, and training continues bitwise-identically.
# ---------------------------------------------------------------------------


def save_checkpoint(
    checkpoint_dir: str,
    *,
    params: Dict[str, torch.Tensor],
    optimizer_state: object,
    step: int,
    num_samples: int,
    dataloader_state: object | None = None,
) -> None:
    """Persist a resume-able training state to ``checkpoint_dir``.

    ``optimizer_state`` must be a dict with keys:
        fp32_master_shard, exp_avg_shard, exp_avg_sq_shard, step_count
    (the format produced by ``entry.run_training`` via ``state_out``).
    """
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    state: Dict[str, Any] = {
        "step": step,
        "num_samples": num_samples,
        "params": {k: v.cpu() for k, v in params.items()},
    }
    if optimizer_state is not None:
        opt_cpu: Dict[str, Any] = {}
        for k, v in optimizer_state.items():
            opt_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        state["optimizer_state"] = opt_cpu
    if dataloader_state is not None:
        state["dataloader_state"] = dataloader_state

    path = os.path.join(checkpoint_dir, f"rank_{rank}.pt")
    torch.save(state, path)


def load_resume_checkpoint(
    checkpoint_dir: str,
    *,
    device: str,
) -> Dict[str, object]:
    """Load a resume-able training state from ``checkpoint_dir``.

    Returns::

        {
            "params":           Dict[str, BF16 tensor on device, TP-sharded],
            "optimizer_state":  dict with fp32_master_shard / exp_avg_shard /
                                exp_avg_sq_shard / step_count (on device),
            "step":             int,
            "num_samples":      int,
            "dataloader_state": opaque object or None,
        }
    """
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    path = os.path.join(checkpoint_dir, f"rank_{rank}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu", weights_only=False)

    params = {k: v.to(device) for k, v in state["params"].items()}

    opt_state = state.get("optimizer_state")

    return {
        "params": params,
        "optimizer_state": opt_state,
        "step": int(state["step"]),
        "num_samples": int(state["num_samples"]),
        "dataloader_state": state.get("dataloader_state"),
    }


# ---------------------------------------------------------------------------
# Per-layer parameter packing helpers (SSOT for forward / backward / entry)
# ---------------------------------------------------------------------------

def build_fwd_layer_params(
    params: Dict[str, torch.Tensor],
    num_layers: int,
) -> list:
    """Build per-layer weight tuples for forward_pass.

    Returns list of (ln_w, qkv_w, proj_w, mlp_ln_w, fc1_w, fc2_w) tuples.
    """
    result = []
    for i in range(num_layers):
        p = f"decoder.layers.{i}"
        result.append((
            params[f"{p}.self_attention.linear_qkv.layer_norm_weight"],
            params[f"{p}.self_attention.linear_qkv.weight"],
            params[f"{p}.self_attention.linear_proj.weight"],
            params[f"{p}.mlp.linear_fc1.layer_norm_weight"],
            params[f"{p}.mlp.linear_fc1.weight"],
            params[f"{p}.mlp.linear_fc2.weight"],
        ))
    return result


def build_bwd_precomputed(
    params: Dict[str, torch.Tensor],
    num_layers: int,
    grad_buffer: Optional[torch.Tensor] = None,
    layout=None,
) -> dict:
    """Build per-layer precomputed data for manual_backward.

    Returns dict with keys: layer_params, layer_names,
    and optionally layer_wbufs, layer_ln_bufs (when grad_buffer/layout given).
    """
    layer_params = []
    layer_names = []
    layer_wbufs = [] if grad_buffer is not None else None
    layer_ln_bufs = [] if grad_buffer is not None else None

    for i in range(num_layers):
        p = f"decoder.layers.{i}"
        fc2_n = f"{p}.mlp.linear_fc2.weight"
        fc1_n = f"{p}.mlp.linear_fc1.weight"
        proj_n = f"{p}.self_attention.linear_proj.weight"
        qkv_n = f"{p}.self_attention.linear_qkv.weight"
        mln_n = f"{p}.mlp.linear_fc1.layer_norm_weight"
        aln_n = f"{p}.self_attention.linear_qkv.layer_norm_weight"

        layer_params.append((
            params[fc2_n], params[fc1_n], params[mln_n],
            params[proj_n], params[qkv_n], params[aln_n],
        ))
        layer_names.append((fc2_n, fc1_n, mln_n, proj_n, qkv_n, aln_n))

        if grad_buffer is not None and layout is not None:
            _po = layout.param_offsets
            _pn = layout.param_numels
            layer_wbufs.append((
                grad_buffer[_po[fc2_n]:_po[fc2_n]+_pn[fc2_n]].view(params[fc2_n].shape),
                grad_buffer[_po[fc1_n]:_po[fc1_n]+_pn[fc1_n]].view(params[fc1_n].shape),
                grad_buffer[_po[proj_n]:_po[proj_n]+_pn[proj_n]].view(params[proj_n].shape),
                grad_buffer[_po[qkv_n]:_po[qkv_n]+_pn[qkv_n]].view(params[qkv_n].shape),
            ))
            layer_ln_bufs.append((
                grad_buffer[_po[mln_n]:_po[mln_n]+_pn[mln_n]],
                grad_buffer[_po[aln_n]:_po[aln_n]+_pn[aln_n]],
            ))

    return {
        'layer_params': layer_params,
        'layer_wbufs': layer_wbufs,
        'layer_ln_bufs': layer_ln_bufs,
        'layer_names': layer_names,
    }
