"""Parameter management for MiniCPM4 0.5B.

Handles static parameter layout, weight loading from canonical state,
and GQA-aware projection shapes.
"""

from __future__ import annotations

__all__ = [
    "ParamSpec",
    "all_param_specs",
    "load_megatron_checkpoint",
    "load_megatron_optimizer_state",
    "load_resume_checkpoint",
    "mtp_layer_prefix",
    "save_checkpoint",
    "should_save_after_step",
    "trainable_param_names",
    "wait_for_async_save",
]

import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import optimizer as optimizer_mod

from . import config
from .engine_config import get_config

_PARAM_NAME_MAP: dict[str, str] = {
    "embedding.word_embeddings.weight": "tok_embeddings.weight",
    "output_layer.weight": "output.weight",
    "decoder.final_layernorm.weight": "norm.weight",
}
for _i in range(config.NUM_LAYERS):
    _mg = f"decoder.layers.{_i}"
    _ours = f"layers.{_i}"
    _PARAM_NAME_MAP.update({
        f"{_mg}.input_layernorm.weight": f"{_ours}.attention_norm.weight",
        f"{_mg}.self_attention.linear_qkv.weight": f"{_ours}.attention.wqkv.weight",
        f"{_mg}.self_attention.linear_proj.weight": f"{_ours}.attention.wo.weight",
        f"{_mg}.pre_mlp_layernorm.weight": f"{_ours}.ffn_norm.weight",
        f"{_mg}.mlp.linear_fc1.weight": f"{_ours}.feed_forward.wfc1.weight",
        f"{_mg}.mlp.linear_fc2.weight": f"{_ours}.feed_forward.w2.weight",
    })


# ---------------------------------------------------------------------------
# Multi-Token Prediction (MTP) parameter naming
# ---------------------------------------------------------------------------
#
# In Megatron the MTP block is registered as ``self.mtp`` on the GPTModel
# (after ``self.decoder``), so its leaf parameter keys are
# ``mtp.layers.{k}.<sub>.weight``. The MTP transformer_layer is a regular
# Megatron TransformerLayer, so its sub-keys mirror the main decoder
# (``input_layernorm`` / ``self_attention.linear_qkv`` / ...).
#
# We mirror the same prefix on our side: ``mtp.layers.{k}.<sub>.weight``,
# but reuse the SAME local sub-names as our main layers
# (``attention_norm`` / ``attention.wqkv`` / ``attention.wo`` /
# ``ffn_norm`` / ``feed_forward.wfc1`` / ``feed_forward.w2``). That lets
# the per-layer forward/backward helpers in ``forward.py`` / ``backward.py``
# drive an MTP transformer layer with nothing more than a different
# prefix string — no new variables, no parallel code path. The MTP-only
# pieces (``enorm`` / ``hnorm`` / ``eh_proj`` / ``final_layernorm``)
# stay outside that helper because they have no analogue in the main
# block.
def mtp_layer_prefix(k: int) -> str:
    """Return our local prefix for MTP layer ``k`` (zero-indexed)."""
    return f"mtp.layers.{k}"


def _main_layer_param_shapes(i: int) -> dict[str, tuple[int, ...]]:
    """SSOT for parameter shapes of main decoder layer ``i``.

    Used by ``_random_init_params`` and ``all_param_specs`` to avoid
    shape duplication across the codebase.
    """
    h = config.HIDDEN_SIZE
    qkv_dim = (config.NUM_HEADS * config.HEAD_DIM
                + 2 * config.NUM_KV_HEADS * config.HEAD_DIM)
    p = f"layers.{i}"
    return {
        f"{p}.attention_norm.weight": (h,),
        f"{p}.attention.wqkv.weight": (qkv_dim, h),
        f"{p}.attention.wo.weight": (h, config.NUM_HEADS * config.HEAD_DIM),
        f"{p}.ffn_norm.weight": (h,),
        f"{p}.feed_forward.wfc1.weight": (2 * config.FFN_HIDDEN_SIZE, h),
        f"{p}.feed_forward.w2.weight": (h, config.FFN_HIDDEN_SIZE),
    }


def _mtp_layer_param_shapes(k: int) -> dict[str, tuple[int, ...]]:
    """SSOT for parameter shapes of a single MTP layer.

    Used by both ``_random_init_params`` and the cold-start branch of
    ``load_megatron_checkpoint`` to avoid shape duplication.
    """
    h = config.HIDDEN_SIZE
    qkv_dim = (config.NUM_HEADS * config.HEAD_DIM
                + 2 * config.NUM_KV_HEADS * config.HEAD_DIM)
    p = mtp_layer_prefix(k)
    return {
        f"{p}.enorm.weight": (h,),
        f"{p}.hnorm.weight": (h,),
        f"{p}.eh_proj.weight": (h, 2 * h),
        f"{p}.attention_norm.weight": (h,),
        f"{p}.attention.wqkv.weight": (qkv_dim, h),
        f"{p}.attention.wo.weight": (h, config.NUM_HEADS * config.HEAD_DIM),
        f"{p}.ffn_norm.weight": (h,),
        f"{p}.feed_forward.wfc1.weight": (2 * config.FFN_HIDDEN_SIZE, h),
        f"{p}.feed_forward.w2.weight": (h, config.FFN_HIDDEN_SIZE),
    }


def _mtp_megatron_to_ours(k: int) -> dict[str, str]:
    """Megatron (EAGLE-style) → ours name map for MTP layer ``k``.

    The deliver-Megatron MTP block is the EAGLE-style speculative-head
    layout (``MTPTransformerBlock`` in
    ``Megatron-LM_4_deliver/megatron/core/transformer/mtp_block.py``),
    NOT the upstream DeepSeek-V3 ``MultiTokenPredictionLayer``. Two key
    differences vs the upstream MTP that the deliver checkpoint forces
    us to mirror here:

    * The checkpoint uses ``mtp.layers.<sub>`` (no integer index) for
      the single transformer layer. K > 1 is not supported by the
      deliver checkpoint we map onto, so we only emit a name for k == 0.
    * The transformer layer's input_layernorm / pre_mlp_layernorm are
      stored as ``layer_norm_weight`` ATTRIBUTES of the linear modules
      (TE fused-layernorm-linear), not as separate top-level keys —
      numerically they are still independent RMSNorm weights, so the
      mapping below pulls them out as separate tensors on our side.
    * There is NO ``final_layernorm`` after the transformer layer in
      the EAGLE block — see ``MTPTransformerBlock.forward()``.
    """
    mg = "mtp"
    ours = mtp_layer_prefix(k)
    if k != 0:
        # The deliver checkpoint contains only one MTP layer. K>1 would
        # require a separate per-k naming convention which the upstream
        # checkpoint we load from cannot satisfy.
        return {}
    return {
        f"{mg}.emb_input_layernorm.weight":
            f"{ours}.enorm.weight",
        f"{mg}.hidden_input_layernorm.weight":
            f"{ours}.hnorm.weight",
        f"{mg}.eagle_fc.weight":
            f"{ours}.eh_proj.weight",
        f"{mg}.layers.self_attention.linear_qkv.layer_norm_weight":
            f"{ours}.attention_norm.weight",
        f"{mg}.layers.self_attention.linear_qkv.weight":
            f"{ours}.attention.wqkv.weight",
        f"{mg}.layers.self_attention.linear_proj.weight":
            f"{ours}.attention.wo.weight",
        f"{mg}.layers.mlp.linear_fc1.layer_norm_weight":
            f"{ours}.ffn_norm.weight",
        f"{mg}.layers.mlp.linear_fc1.weight":
            f"{ours}.feed_forward.wfc1.weight",
        f"{mg}.layers.mlp.linear_fc2.weight":
            f"{ours}.feed_forward.w2.weight",
    }


def mtp_param_name_map() -> dict[str, str]:
    """Build the Megatron→ours name map for all MTP layers in the current
    runtime config (empty dict when MTP is disabled).
    """
    out: dict[str, str] = {}
    for k in range(get_config().mtp_num_layers):
        out.update(_mtp_megatron_to_ours(k))
    return out


def _random_init_params(
    device: str | torch.device = "cuda:0",
    init_std: float | None = None,
    vocab_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Create randomly initialized parameters matching MiniCPM4 0.5B architecture.

    muP-aware init mirrors Megatron's ``transformer_config.py`` +
    ``gpt_model.py`` semantics for ``--use-mup --init-method-std S``:

    * **embeddings + output_layer** use Megatron's ``standard_init_method``
      (init_method_normal(S))  → std = ``init_std``.
      - ``language_model_embedding.py:51``:
        ``init_method=self.config.init_method if not self.config.use_mup
        else self.config.standard_init_method``
      - ``gpt_model.py:138``:
        ``init_method=config.init_method if not config.use_mup
        else config.standard_init_method``
    * **all other linear weights** (qkv / wo / fc1 / fc2 / mtp eh_proj)
      use ``mup_init_method_normal(S, hidden/base)``  →
      std = ``init_std / sqrt(hidden / mup_base_hidden_size)``.

    For MiniCPM4-0.5B (hidden=1024, base=256, scale=4):
      - mup-init std = init_std / 2
      - standard std = init_std

    With baseline ``--init-method-std 0.1`` this gives 0.05 (linear) +
    0.1 (emb / output). Without muP (MUP_EMB_SCALE<=0), all weights fall
    back to N(0, init_std).

    The earlier vanilla ``init_std=0.02`` everywhere was off by 2.5x-5x
    relative to baseline, which compounded into a ~70% lm-loss gap by
    step 1000 (verified against Megatron task 99584's stable log).
    """
    import math

    v = vocab_size if vocab_size is not None else config.get_effective_vocab_size()
    dtype = config.compute_dtype()
    h = config.HIDDEN_SIZE
    params: dict[str, torch.Tensor] = {}

    if init_std is None:
        init_std = config.INIT_METHOD_STD

    cfg = get_config()
    use_mup = cfg.mup_emb_scale > 0
    if use_mup:
        width_scale = h / config.MUP_BASE_HIDDEN_SIZE  # 4.0 for MiniCPM4 0.5B
        mup_std = init_std / math.sqrt(width_scale)
    else:
        mup_std = init_std

    def _rand_mup(shape: tuple[int, ...]) -> torch.Tensor:
        """Standard linear weight: muP-aware std (or vanilla when use_mup=False)."""
        return torch.randn(shape, dtype=torch.float32).mul_(mup_std).to(dtype=dtype, device=device)

    def _rand_std(shape: tuple[int, ...]) -> torch.Tensor:
        """Standard-init weight (embeddings + output_layer when use_mup=True)."""
        return torch.randn(shape, dtype=torch.float32).mul_(init_std).to(dtype=dtype, device=device)

    params["tok_embeddings.weight"] = _rand_std((v, h))
    for i in range(config.NUM_LAYERS):
        for name, shape in _main_layer_param_shapes(i).items():
            if name.endswith("norm.weight"):
                params[name] = torch.ones(shape[0], dtype=dtype, device=device)
            else:
                params[name] = _rand_mup(shape)
    params["norm.weight"] = torch.ones(h, dtype=dtype, device=device)
    params["output.weight"] = _rand_std((v, h))

    for k in range(cfg.mtp_num_layers):
        for name, shape in _mtp_layer_param_shapes(k).items():
            if name.endswith("norm.weight"):
                params[name] = torch.ones(shape[0], dtype=dtype, device=device)
            else:
                params[name] = _rand_mup(shape)

    print(f"  Random init: {len(params)} parameters ({dtype}), "
          f"vocab={v}, init_std={init_std} (use_mup={use_mup}, "
          f"mup_std={mup_std:.4f}, standard_std={init_std:.4f}), "
          f"mtp_layers={cfg.mtp_num_layers}")
    return params


def load_megatron_checkpoint(
    checkpoint_root: str,
    device: str | torch.device = "cuda:0",
    *,
    random_init: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Load canonical_state_fp32.pt and return parameter dict in compute dtype.

    When ``random_init`` is True, creates randomly initialized parameters
    instead. Falls back to ``get_config().random_init`` when not explicitly passed.
    """
    cfg = get_config()
    do_random = random_init if random_init is not None else cfg.random_init
    if do_random:
        return _random_init_params(device=device, vocab_size=cfg.vocab_size)

    fp32_path = os.path.join(checkpoint_root, "canonical_state_fp32.pt")
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"No checkpoint at {fp32_path}")

    ckpt = torch.load(fp32_path, map_location="cpu", weights_only=False)
    dtype = config.compute_dtype()
    params: dict[str, torch.Tensor] = {}

    full_map = dict(_PARAM_NAME_MAP)
    full_map.update(mtp_param_name_map())

    missing_mtp: list[str] = []
    for ckpt_key, our_name in full_map.items():
        if ckpt_key not in ckpt:
            if our_name.startswith("mtp."):
                missing_mtp.append(ckpt_key)
            continue
        w = ckpt[ckpt_key]
        if w is None or not isinstance(w, torch.Tensor):
            continue
        params[our_name] = w.to(dtype=dtype, device=device)

    if "output.weight" in params:
        v = params["output.weight"].shape[0]
        expected = config.get_effective_vocab_size()
        if v != expected:
            raise ValueError(
                f"Checkpoint vocab_size ({v}) != configured vocab_size ({expected}). "
                f"Set EngineConfig(vocab_size={v}) if this is intentional."
            )

    # Cold-start MTP from a checkpoint that does not contain MTP weights:
    # init enorm/hnorm/final_layernorm to ones, eh_proj from N(0, 0.02),
    # and the transformer_layer to a zero-overhead random init that lets
    # the loss start at the same place the baseline does. We deliberately
    # log this so resuming from a non-MTP checkpoint is not silent.
    _mtp_n = cfg.mtp_num_layers
    if _mtp_n > 0 and missing_mtp:
        def _rand_like(shape: tuple[int, ...], std: float = 0.02) -> torch.Tensor:
            return (
                torch.randn(shape, dtype=torch.float32)
                .mul_(std)
                .to(dtype=dtype, device=device)
            )

        cold_initialised = 0
        for k in range(_mtp_n):
            for name, shape in _mtp_layer_param_shapes(k).items():
                if name not in params:
                    if name.endswith("norm.weight"):
                        params[name] = torch.ones(shape[0], dtype=dtype, device=device)
                    else:
                        params[name] = _rand_like(shape)
                    cold_initialised += 1
        print(
            f"  [load_megatron_checkpoint] MTP weights NOT in checkpoint — "
            f"cold-initialised {cold_initialised} MTP tensors "
            f"(checkpoint missing {len(missing_mtp)} mtp.* keys; "
            f"first={missing_mtp[0]!r})",
            flush=True,
        )

    print(f"  Loaded {len(params)} parameters from checkpoint ({dtype})")
    return params



_OPTIM_STATE_FILENAME = "canonical_optimizer_state_fp32.pt"


def load_megatron_optimizer_state(
    checkpoint_root: str,
    device: str | torch.device = "cuda:0",
) -> dict | None:
    """Load the FP32 Adam optimizer state extracted from a Megatron distcp.

    Looks for ``<checkpoint_root>/canonical_optimizer_state_fp32.pt``
    (produced by ``workload/scripts/extract_optimizer_state_from_distcp.py``).
    Returns a dict ready to feed into :meth:`AdamState.load_state_dict`:

        {
            "master_weights": dict[str, FP32 Tensor on ``device``],
            "exp_avg":        dict[str, FP32 Tensor on ``device``],
            "exp_avg_sq":     dict[str, FP32 Tensor on ``device``],
            "step_count":     int,
            "num_samples":    int,   # 0 when unknown / forced to fresh
        }

    Returns ``None`` when the file is absent — callers should fall back
    to a fresh AdamState. This is the single biggest fix for the
    "step 1 loss explosion" we hit when starting from a Megatron
    mid-training checkpoint that ships m/v moments in its distcp:
    fresh Adam at step 0 has bias correction ``bc1=0.1, bc2=0.05``,
    which inflates the first per-param update by ~22× compared to
    a mature Adam state — the loss diverges in step 1 and never
    recovers in the warmup window.
    """
    path = os.path.join(checkpoint_root, _OPTIM_STATE_FILENAME)
    if not os.path.exists(path):
        return None

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{_OPTIM_STATE_FILENAME} at {path} is not a dict "
            f"(got {type(payload).__name__})"
        )

    required = {"master_weights", "exp_avg", "exp_avg_sq", "step_count"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(
            f"{_OPTIM_STATE_FILENAME} at {path} missing required keys: "
            f"{sorted(missing)}"
        )

    def _move(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            k: v.to(dtype=torch.float32, device=device).contiguous()
            for k, v in d.items()
        }

    out: dict = {
        "master_weights": _move(payload["master_weights"]),
        "exp_avg": _move(payload["exp_avg"]),
        "exp_avg_sq": _move(payload["exp_avg_sq"]),
        "step_count": int(payload["step_count"]),
        # ``num_samples`` is optional in the on-disk file (the converter
        # writes -1 when unknown). Treat any non-positive value as 0
        # (fresh sample counter) so the LR scheduler still advances
        # correctly via ``step_count``.
        "num_samples": max(0, int(payload.get("num_samples", -1))),
    }
    print(
        f"  [load_megatron_optimizer_state] step_count={out['step_count']} "
        f"num_samples={out['num_samples']} master={len(out['master_weights'])} "
        f"exp_avg={len(out['exp_avg'])} exp_avg_sq={len(out['exp_avg_sq'])} "
        f"← {path}",
        flush=True,
    )
    return out


_CHECKPOINT_FILENAME = "training_state.pt"
_CHECKPOINT_FORMAT_VERSION = 1


def _detach_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu") for k, v in state.items()}


def should_save_after_step(
    *,
    completed_step: int,
    save_interval: int | None,
) -> bool:
    """Return ``True`` iff a resume checkpoint should be saved after
    ``completed_step`` (zero-based) finishes.

    Convention: ``completed_step + 1`` is the next step the trainer will
    run. A save fires whenever the next-step index is a positive multiple
    of ``save_interval`` — i.e. saves land *between* step boundaries,
    matching the ``RESUME_SAVE_STEP`` semantics in
    ``test_resume_train.py`` (``save_step=100`` means "after completing
    step 99, persist next-step=100"). Centralising the rule here keeps
    train-loop and CLI entry points from drifting on the +1 / modulo
    boundary.

    Args:
        completed_step: zero-based index of the step that just finished.
        save_interval: save frequency in global steps; ``None`` or any
            non-positive value disables saves (returns ``False``).

    Returns:
        ``True`` when a save should fire after ``completed_step``,
        ``False`` otherwise.
    """
    if save_interval is None or int(save_interval) <= 0:
        return False
    return ((int(completed_step) + 1) % int(save_interval)) == 0


# ---------------------------------------------------------------------------
# Async checkpoint save plumbing
# ---------------------------------------------------------------------------
#
# When ``save_checkpoint(..., async_io=True)`` is used the GPU→CPU copy still
# happens on the main thread (so the caller can immediately tear down GPU
# state), but the actual ``torch.save`` write is dispatched to a background
# thread. The caller must invoke ``wait_for_async_save()`` before any
# downstream consumer (typically ``load_resume_checkpoint`` on another rank)
# expects the file to exist on disk.
#
# Only one async save can be in flight at a time per process. ``save_checkpoint``
# auto-waits for any pending save before starting a new one to keep the
# protocol single-writer. Errors raised inside the background thread are
# captured and re-raised the next time the caller waits, so failures are
# never silently swallowed.

_pending_save_lock = threading.Lock()
_pending_save: dict | None = None


def _save_payload_sync(payload: dict, path: str, tmp_path: str) -> None:
    """Synchronously persist ``payload`` to ``path`` via ``tmp_path`` rename.

    Cleans up ``tmp_path`` if anything goes wrong so a partial file cannot be
    picked up by a later resume.
    """
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _async_save_target(payload: dict, path: str, tmp_path: str, slot: dict) -> None:
    """Background-thread entry: run the disk write and stash any exception."""
    try:
        _save_payload_sync(payload, path, tmp_path)
    except BaseException as exc:
        slot["error"] = exc


def wait_for_async_save(timeout: float | None = None) -> bool:
    """Block until the in-flight async save (if any) completes.

    Returns:
        ``True`` if a save was pending and has completed; ``False`` if no
        async save was in flight at the time of the call.

    Raises:
        Re-raises any exception that the background save thread captured,
        so the caller learns about disk errors loudly.
        ``TimeoutError`` if ``timeout`` is given and the save did not
        complete in time (the save is left running in the background and
        a subsequent ``wait_for_async_save`` call must be made).
    """
    global _pending_save
    with _pending_save_lock:
        slot = _pending_save
    if slot is None:
        return False

    thread: threading.Thread = slot["thread"]
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise TimeoutError(
            f"async checkpoint save did not finish within {timeout}s "
            f"(target={slot['path']})"
        )

    with _pending_save_lock:
        _pending_save = None

    err = slot.get("error")
    if err is not None:
        raise err
    return True


def save_checkpoint(
    checkpoint_dir: str,
    params: dict[str, torch.Tensor],
    optimizer_state: optimizer_mod.AdamState,
    step: int,
    num_samples: int | None = None,
    dataloader_state: dict | None = None,
    *,
    async_io: bool = False,
) -> None:
    """Save training checkpoint to ``<checkpoint_dir>/training_state.pt``.

    Persists everything required for an identical-trajectory resume in a
    DP-replicated setting (every DP rank holds the same parameters and
    optimizer state because grads are reduce_scatter→all_gathered and
    Adam is applied identically on each rank):

      - BF16 model parameters (``params``)
      - FP32 optimizer state via ``AdamState.state_dict()``:
          * ``master_weights`` (FP32 deep copy of the master)
          * ``exp_avg`` / ``exp_avg_sq`` (Adam first/second moments)
          * ``step_count`` (number of optimizer.step() calls)
          * ``num_samples`` (samples consumed so far)
      - Training progress (``step``, top-level ``num_samples``)
      - Dataloader state (opaque; ``None`` is allowed when the resume is
        in-process and the live dataloader is reused)

    ``num_samples`` defaults to ``optimizer_state.num_samples`` so callers
    don't have to pass it explicitly. If both the kwarg and the optimizer
    state carry a value and they disagree, the call fails fast (mismatch
    is almost always a bug — e.g. forgetting to update the optimizer
    counter after a microbatch reduce).

    All tensors are moved to CPU before serialization to keep checkpoints
    portable across GPU layouts and to avoid carrying CUDA storage handles
    in the on-disk file. The temp file is cleaned up on any write failure
    so a half-finished checkpoint cannot be picked up by a later resume.
    Only rank 0 writes to disk; other ranks are no-ops so this function
    is safe to call collectively from every rank.

    Callers are expected to choose ``checkpoint_dir``. The production
    train loop now passes a per-step subdirectory such as
    ``<save_dir>/step_0005000`` so multiple checkpoints are retained
    instead of overwriting a single file in place.

    ``async_io=True`` snapshots the full payload to CPU on the calling
    thread (so the caller can immediately ``del`` the on-device tensors
    and free GPU memory) and dispatches the actual ``torch.save`` write to
    a background thread. The caller must call :func:`wait_for_async_save`
    before any downstream code expects the file to exist on disk
    (typically just before the cross-rank barrier that gates the resume
    load). If a previous async save is still in flight when this is
    called, this function transparently waits for it to finish first
    (single-writer protocol) so two concurrent writes never race for the
    same path.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank != 0:
        return

    if num_samples is None:
        num_samples = int(optimizer_state.num_samples)
    elif int(num_samples) != int(optimizer_state.num_samples):
        raise ValueError(
            "save_checkpoint: num_samples kwarg "
            f"({int(num_samples)}) disagrees with optimizer_state.num_samples "
            f"({int(optimizer_state.num_samples)}); callers should not pass "
            "stale snapshots."
        )

    # Single-writer guarantee: a previous async save must complete before
    # we touch the same target directory. We do this *before* building the
    # next payload so the GPU→CPU copy isn't wasted if the prior save
    # failed.
    wait_for_async_save()

    os.makedirs(checkpoint_dir, exist_ok=True)
    opt_dict = optimizer_state.state_dict()
    payload = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "num_samples": int(num_samples),
        "param_names": list(optimizer_state.param_names),
        "params": _detach_to_cpu(params),
        "optimizer_state": {
            "master_weights": _detach_to_cpu(opt_dict["master_weights"]),
            "exp_avg": _detach_to_cpu(opt_dict["exp_avg"]),
            "exp_avg_sq": _detach_to_cpu(opt_dict["exp_avg_sq"]),
            "step_count": int(opt_dict["step_count"]),
            "num_samples": int(opt_dict["num_samples"]),
        },
        "dataloader_state": dataloader_state,
    }
    path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)
    tmp_path = path + ".tmp"

    if async_io:
        global _pending_save
        slot: dict = {"path": path, "error": None}
        thread = threading.Thread(
            target=_async_save_target,
            args=(payload, path, tmp_path, slot),
            name=f"ckpt-save-step{step}",
            daemon=True,
        )
        slot["thread"] = thread
        with _pending_save_lock:
            _pending_save = slot
        thread.start()
        print(
            f"  [save_checkpoint async] step={step} num_samples={num_samples} "
            f"params={len(payload['params'])} → {path} (background)",
            flush=True,
        )
        return

    _save_payload_sync(payload, path, tmp_path)
    print(
        f"  [save_checkpoint] step={step} num_samples={num_samples} "
        f"params={len(payload['params'])} → {path}",
        flush=True,
    )


def load_resume_checkpoint(
    checkpoint_dir: str,
    device: str = "cuda:0",
) -> dict:
    """Load a previously-saved training checkpoint for resume.

    Returns a dict with the keys required by the resume-gate driver:

      - ``params``:           dict[str, Tensor] in compute dtype on ``device``
      - ``optimizer_state``:
          * ``master_weights``: dict[str, FP32 Tensor on ``device``]
          * ``exp_avg``:        dict[str, FP32 Tensor on ``device``]
          * ``exp_avg_sq``:     dict[str, FP32 Tensor on ``device``]
          * ``step_count``:     int
      - ``step``:             int
      - ``num_samples``:      int
      - ``dataloader_state``: opaque dict | None
      - ``param_names``:      list[str] in optimizer iteration order

    All ranks load the same on-disk file (DP-replicated state); per-rank
    sharding is reconstructed by the optimizer at first use.
    """
    path = os.path.join(checkpoint_dir, _CHECKPOINT_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No resume checkpoint at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Resume checkpoint at {path} is not a dict (got {type(payload).__name__})"
        )

    fmt = int(payload.get("format_version", 0))
    if fmt != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Resume checkpoint format mismatch: file={fmt}, "
            f"expected={_CHECKPOINT_FORMAT_VERSION}"
        )

    required = {"step", "num_samples", "params", "optimizer_state"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(
            f"Resume checkpoint at {path} missing required keys: {sorted(missing)}"
        )

    dtype = config.compute_dtype()
    params: dict[str, torch.Tensor] = {
        k: v.to(dtype=dtype, device=device).contiguous()
        for k, v in payload["params"].items()
    }

    raw_opt = payload["optimizer_state"]
    for needed in ("master_weights", "exp_avg", "exp_avg_sq", "step_count"):
        if needed not in raw_opt:
            raise ValueError(
                f"Resume checkpoint optimizer_state missing key: {needed!r}"
            )

    optimizer_state: dict = {
        "master_weights": {
            k: v.to(dtype=torch.float32, device=device).contiguous()
            for k, v in raw_opt["master_weights"].items()
        },
        "exp_avg": {
            k: v.to(dtype=torch.float32, device=device).contiguous()
            for k, v in raw_opt["exp_avg"].items()
        },
        "exp_avg_sq": {
            k: v.to(dtype=torch.float32, device=device).contiguous()
            for k, v in raw_opt["exp_avg_sq"].items()
        },
        "step_count": int(raw_opt["step_count"]),
        # ``num_samples`` lived only at top level in format_version=1
        # initial release; mirror it inside optimizer_state so callers
        # can pass this dict straight to ``AdamState.load_state_dict``.
        "num_samples": int(
            raw_opt.get("num_samples", payload["num_samples"])
        ),
    }

    if "output.weight" in params:
        v = params["output.weight"].shape[0]
        expected = config.get_effective_vocab_size()
        if v != expected:
            raise ValueError(
                f"Resume checkpoint vocab_size ({v}) != configured vocab_size ({expected}). "
                f"Set EngineConfig(vocab_size={v}) if this is intentional."
            )

    print(
        f"  [load_resume_checkpoint] step={int(payload['step'])} "
        f"num_samples={int(payload['num_samples'])} params={len(params)} "
        f"({dtype}) ← {path}",
        flush=True,
    )

    return {
        "params": params,
        "optimizer_state": optimizer_state,
        "step": int(payload["step"]),
        "num_samples": int(payload["num_samples"]),
        "dataloader_state": payload.get("dataloader_state"),
        "param_names": payload.get(
            "param_names", list(optimizer_state["master_weights"].keys())
        ),
    }


@dataclass(frozen=True)
class ParamSpec:
    """Metadata for a single trainable parameter.

    ``all_param_specs()`` is the **single source of truth** for the
    complete set of trainable parameters.  Every consumer that needs an
    ordered list of names, numels, or a norm/non-norm partition derives
    its view from this registry instead of hand-enumerating layers.
    """

    name: str
    numel: int
    is_norm: bool
    has_weight_decay: bool


def _specs_from_shapes(
    shapes: dict[str, tuple[int, ...]], is_norm_fn=None,
) -> list[ParamSpec]:
    """Convert a shape dict into ParamSpec list, preserving insertion order."""
    import math
    if is_norm_fn is None:
        is_norm_fn = lambda name: name.endswith("norm.weight")
    return [
        ParamSpec(name, math.prod(shape), is_norm_fn(name),
                  has_weight_decay=len(shape) > 1)
        for name, shape in shapes.items()
    ]


def all_param_specs() -> list[ParamSpec]:
    """Return specs for every trainable parameter in forward-registration order.

    This is the single authoritative enumeration of model parameters.
    ``trainable_param_names()``, ``nccl.baseline_buffer_layout()``, and
    ``optimizer._baseline_grad_norm_order()`` all derive from this list.

    Shape knowledge is derived from ``_main_layer_param_shapes`` and
    ``_mtp_layer_param_shapes`` — the single sources of truth for layer shapes.
    """
    h = config.HIDDEN_SIZE
    specs: list[ParamSpec] = []

    specs.append(ParamSpec("tok_embeddings.weight", config.get_effective_vocab_size() * h, False, has_weight_decay=True))
    for i in range(config.NUM_LAYERS):
        specs.extend(_specs_from_shapes(_main_layer_param_shapes(i)))
    specs.append(ParamSpec("norm.weight", h, True, has_weight_decay=False))
    for k in range(get_config().mtp_num_layers):
        specs.extend(_specs_from_shapes(_mtp_layer_param_shapes(k)))
    specs.append(ParamSpec("output.weight", config.get_effective_vocab_size() * h, False, has_weight_decay=True))
    return specs


def trainable_param_names() -> list[str]:
    """Return ordered list of trainable parameter names.

    Derived from ``all_param_specs()`` — the single source of truth.
    """
    return [s.name for s in all_param_specs()]
