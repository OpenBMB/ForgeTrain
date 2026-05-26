#!/bin/bash
set -euo pipefail
# 64-GPU self-developed training engine entry for MiniCPM4 0.5B decay phase.
# Uses `python -m training_engine_tensor train` CLI with checkpoint save/resume.
# Loads pre-trained weights from stable_2 (converted to canonical_state_fp32.pt).

# ── Resolve ENGINE_ROOT from this script's location ──────────────────
# Avoid hard-coding any user-specific install path: derive it from
# ``BASH_SOURCE`` so the same script works for any user / any checkout
# without surgery. Falls back to caller's value if explicitly exported.
ENGINE_ROOT="${ENGINE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ENGINE_ROOT

# Default to Tsinghua (public mirror) because pypi.hs1.paratera.com
# (the legacy in-cluster mirror) frequently times out on jobs launched
# from outside the harness pipeline. Override with PYPI_INDEX_URL in
# the spec envVars when running inside the closed paratera network.
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

install_pkg() {
    python -m pip install --no-cache-dir --retries 1 --timeout 15 "$@" \
        -i "$PYPI_INDEX_URL" 2>/dev/null || \
    python -m pip install --no-cache-dir --retries 1 --timeout 15 "$@" 2>/dev/null || true
}

python -c "import sentencepiece" 2>/dev/null || install_pkg sentencepiece
python -c "import modelbest_sdk" 2>/dev/null || install_pkg modelbest_sdk

# ── HuggingFace datasets/transformers ──
# Required since zz/no_harness retired the modelbest_sdk ``--data-path``
# backend in favor of the new ``--hf-dataset`` CLI surface. Install into
# an in-repo --target dir so the install follows the checkout (also lets
# entry_hf_pretrain.sh and entry_train.sh share the same cache).
#
# Index selection: PYPI_INDEX_URL above defaults to the (unreliable)
# paratera-internal mirror; for HF deps we use the public Tsinghua
# mirror by default because the paratera mirror often times out on
# datasets/transformers (task 401905 hit this). Override via
# PYPI_INDEX_URL_HF when needed.
HF_SITE="${HF_SITE:-${ENGINE_ROOT}/.hf_site}"
export PYTHONPATH="${HF_SITE}:${PYTHONPATH:-}"
if ! python -c "import datasets, transformers" 2>/dev/null; then
    _HF_INDEX="${PYPI_INDEX_URL_HF:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    pip install --target "$HF_SITE" -i "$_HF_INDEX" --timeout 30 --retries 3 \
        datasets transformers || {
        echo "[entry_train] WARN: HF deps install via $_HF_INDEX failed; retrying via pypi.org" >&2
        pip install --target "$HF_SITE" --timeout 30 --retries 3 datasets transformers
    }
fi

# Custom-GEMM (CUTLASS DSL) wheels live on the shared FS — only install when
# the operator gates request them (CUSTOM_GEMM=1). Five DSL wheels (mirror of
# workload/scripts/run_16gpu_mfu.sh) PLUS the standalone ``nvidia_cutlass``
# package — the gemm_fc1/gemm_fc2/... export scripts do ``import cutlass``
# at compile time, which the DSL bundle alone does not satisfy. With only the
# DSL installed, CuTeDSL export silently fails, the loader falls back to the
# CUTLASS C++ kernel which only exposes ``gemm_fwd`` (no ``gemm_fwd_fast``),
# and ``_bind_fast_paths()`` crashes with AttributeError mid-step (see
# task 329370 master log around line 1722).
# All wheels use ``--break-system-packages --no-deps`` to avoid touching the
# image's torch / cuda metadata.
# Cutlass is needed both by CUSTOM_GEMM=1 (the 5 GEMM ops) AND by
# OP_ATTENTION=v1 (the FA4 cutedsl forward + backward specs); guard on
# either gate. Forgetting the OP_ATTENTION=v1 case caused task 330319 to
# hit the strict fail-fast in attention.kernel.py::_resolve_fa4_bwd
# because cutlass had not been installed, fa4_cute could not import,
# and we (correctly) refuse to fall back to the O(S^2) manual SDPA path.
if [[ "${CUSTOM_GEMM:-0}" == "1" || "${OP_ATTENTION:-baseline}" == "v1" ]]; then
    # ── Wheel install (idempotent — skipped when ``import cutlass`` works) ──
    if ! python -c "import cutlass" 2>/dev/null; then
        # Wheel directory holds the 5 CUTLASS-DSL wheels (see for-loop
        # below). Set CUTLASS_WHEELS_DIR explicitly in the job spec
        # envVars; no individual-user default is provided here.
        _CUTLASS_WHEELS="${CUTLASS_WHEELS_DIR:-}"
        if [[ -n "$_CUTLASS_WHEELS" && -d "$_CUTLASS_WHEELS" ]]; then
            # Install one wheel at a time + ``--no-index`` so pip never
            # touches the (offline) public index — task 329418 hit
            # ``ResolutionImpossible`` when pip tried to resolve a pip
            # self-update against pypi.org. The DSL libs_base wheel
            # ships ``nvidia_cutlass_dsl/python_packages/cutlass/`` plus a
            # ``.pth`` that injects it onto sys.path, so ``import cutlass``
            # works after install.
            for _whl in \
                "nvidia_cutlass_dsl-4.4.2-py3-none-any.whl" \
                "nvidia_cutlass_dsl_libs_base-4.4.2-cp312-cp312-manylinux_2_28_x86_64.whl" \
                "cuda_bindings-12.9.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl" \
                "cuda_pathfinder-1.5.4-py3-none-any.whl" \
                "apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"; do
                echo "[cutlass] installing $_whl"
                pip3 install --break-system-packages --no-deps --no-index \
                    "$_CUTLASS_WHEELS/$_whl" 2>&1 | tail -2 \
                    || echo "[cutlass] FAILED: $_whl"
            done
        else
            echo "[cutlass] FATAL: CUSTOM_GEMM=1 but no usable wheel directory."
            echo "  Set CUTLASS_WHEELS_DIR=<path> to the dir containing the"
            echo "  CUTLASS-DSL wheels listed above (currently '${_CUTLASS_WHEELS:-<unset>}')."
            exit 11
        fi
    fi
    if ! python -c "import cutlass" 2>/dev/null; then
        echo "[cutlass] FATAL: import cutlass still fails after install — refusing to continue with CUSTOM_GEMM=1 (no fallback)"
        exit 12
    fi
    echo "[cutlass] OK: import cutlass works"

    # ── Per-pod ephemeral compile cache ──────────────────────────────
    # Earlier zz/no_harness versions of this script pinned
    # TORCH_EXTENSIONS_DIR / CUTEDSL_CACHE_ROOT under the shared-FS
    # ${ENGINE_ROOT}/.persist_cache to share build artefacts across
    # pods. That breaks first-cold-start runs from a brand-new
    # checkout: multiple ranks race on the same TORCH_EXTENSIONS_DIR
    # path (per-rank lockfiles only cover *one* extension at a time)
    # and 7 of 8 local ranks fall through to the CUTLASS C++ fallback
    # extension (``gemm_fc1_sm90``) which is missing the
    # ``gemm_fwd_fast`` symbol; _bind_fast_paths() then crashes with
    # ``AttributeError: module 'gemm_fc1_sm90' has no attribute
    # 'gemm_fwd_fast'`` (task 401923).
    #
    # Until a proper rendezvous/precompile job is plumbed in for
    # zz/no_harness, leave the cache as torch's default (per-pod
    # ``/tmp/.../torch_extensions`` + per-pod CUTEDSL JIT). This is
    # the same behaviour as ``entry_hf_pretrain.sh`` which has been
    # passing under the same image with multi-pod CUSTOM_GEMM=1 runs
    # (tasks 401827/401829/401832/401834/401835/401927). Re-enable
    # the shared cache later by exporting TORCH_EXTENSIONS_DIR /
    # CUTEDSL_CACHE_ROOT / CUTE_DSL_CACHE_DIR / OP_GEMM_ATTN_OUT_PROJ_BACKEND
    # explicitly in the spec envVars when the precompile flow is back.
    if [[ -n "${USE_SHARED_OPS_CACHE:-}" ]]; then
        _CACHE_ROOT="${CUSTOM_OPS_CACHE_ROOT:-${ENGINE_ROOT}/.persist_cache}"
        _IMG_TAG="${OPS_CACHE_TAG:-ngc_47165eea_py312}"
        _PERSIST="$_CACHE_ROOT/$_IMG_TAG"
        mkdir -p "$_PERSIST/torch_extensions" "$_PERSIST/cutedsl_jit_cache"
        export TORCH_EXTENSIONS_DIR="$_PERSIST/torch_extensions"
        export CUTEDSL_CACHE_ROOT="$_PERSIST"
        export CUTE_DSL_CACHE_DIR="$_PERSIST/cutedsl_jit_cache"
        export OP_GEMM_ATTN_OUT_PROJ_BACKEND="${OP_GEMM_ATTN_OUT_PROJ_BACKEND:-cexport}"
        echo "[cutlass] USE_SHARED_OPS_CACHE=1 → TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
    else
        echo "[cutlass] per-pod ephemeral cache (USE_SHARED_OPS_CACHE unset)"
    fi

    # ── LD_PRELOAD: gemm_output's CuTeDSL backend needs the system
    #    libcudart loaded BEFORE the torch-bundled one (otherwise it
    #    misses a CUDA 12.4+ Library API). Detected at runtime via
    #    ``_cutedsl_preload_ok()``; without it gemm_output silently
    #    falls back to its (slow) Python JIT path. We also give the
    #    other gemm ops their LD_PRELOAD just in case.
    if [[ -f /usr/local/cuda/lib64/libcudart.so.12 ]]; then
        export LD_PRELOAD="${LD_PRELOAD:-}${LD_PRELOAD:+:}/usr/local/cuda/lib64/libcudart.so.12"
        echo "[cutlass] LD_PRELOAD=$LD_PRELOAD"
    fi
fi

cd "$ENGINE_ROOT"

export PYTHONPATH="${ENGINE_ROOT}/src:${PYTHONPATH:-}"
# MEGATRON_ROOT is a harness-side contract (whitelisted in
# ``src/training_engine_tensor/engine_config.py``); the engine itself does
# not import Megatron at runtime. Leave unset by default so this checkout
# does not pin any individual user's Megatron path; jobs that need it must
# set MEGATRON_ROOT explicitly in spec envVars.
export MEGATRON_ROOT="${MEGATRON_ROOT:-}"

# Env vars consumed by training_engine_tensor internals
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export FP32_GRAD_ACCUM=1
export TE_ROPE=1
export FUSED_OPS=1
export FUSE_DIRECT_ATTN=1
export GRAD_DIAG=1
# ── Custom op defaults (bugs fixed 2026-05-09, verified in tasks 333305/333646/335574) ──
#
# USE_FUSED_NORM_BWD: fused Triton RMSNorm backward. Root cause was autotune
#   corrupting the in-place d_hidden buffer (missing restore_value); fixed by
#   adding restore_value=("d_hidden_ptr",) to the autotune decorator.
# OP_GEMM_OUTPUT: CuTeDSL custom output-layer GEMM. Root cause was _ensure_w_cache
#   using weight._version as cache key, which Triton fused Adam (tl.store) does not
#   bump; fixed by dropping the version check and always refreshing from live weight.
export USE_FUSED_NORM_BWD="${USE_FUSED_NORM_BWD:-1}"
export OP_GEMM_OUTPUT="${OP_GEMM_OUTPUT:-v1}"
# Safe defaults for the remaining custom ops (override per-spec when
# needed; see ``workload/notes/decay_phase_training.md`` §4.2 for the
# verified-safe combination).
export OP_ATTENTION="${OP_ATTENTION:-baseline}"
export OP_GEMM_QKV_PROJ="${OP_GEMM_QKV_PROJ:-baseline}"
export OP_GEMM_ATTN_OUT_PROJ="${OP_GEMM_ATTN_OUT_PROJ:-baseline}"
export OP_GEMM_FC1="${OP_GEMM_FC1:-baseline}"
export OP_GEMM_FC2="${OP_GEMM_FC2:-baseline}"
export FUSE_GEMM_PAD="${FUSE_GEMM_PAD:-0}"
# muP per-parameter LR scaling requires per-param optimizer path
export MULTI_TENSOR_ADAM="${MULTI_TENSOR_ADAM:-0}"
export RANDOM_INIT="${RANDOM_INIT:-0}"
export VOCAB_SIZE=73448
# TOKENIZER_TYPE / TOKENIZER_MODEL: pass-through env vars for downstream
# Megatron-compatible tooling launched from the same environment.
# 注意 entry_train.sh 走的是 modelbest_sdk + ``--data-path`` 路径，数据流
# 已经预 tokenize，所以 ``training_engine_tensor`` 在这个 entry 下不会读
# 这两个变量；但 HF dataset 路径（``entry_hf_pretrain.sh`` →
# ``--tokenizer-path``）下 ``hf_dataloader._load_tokenizer`` 会真正用
# transformers.AutoTokenizer 加载 tokenizer，那条路径请通过该入口的
# ``TOKENIZER_PATH`` env / ``--tokenizer-path`` CLI 提供具体路径，而不是
# 把 cluster 路径写回到本脚本。
export TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export CLEAR_SAMPLER_STATE="${CLEAR_SAMPLER_STATE:-1}"

# Performance optimizations
export WGRAD_OVERLAP="${WGRAD_OVERLAP:-1}"
export WGRAD_BUCKET_TARGET_ELEMS="${WGRAD_BUCKET_TARGET_ELEMS:-419430400}"
export OPT_ALLGATHER_OVERLAP="${OPT_ALLGATHER_OVERLAP:-0}"
export ASYNC_LOSS_AR="${ASYNC_LOSS_AR:-1}"
export SHARDED_OPTIMIZER="${SHARDED_OPTIMIZER:-1}"
# P2-B grad-accum overlap (opt-in, defaults off → P2-A behaviour).
# Only takes effect when num_local_microbatches > 1 AND WGRAD_OVERLAP=1.
export ACCUM_STREAM_OVERLAP="${ACCUM_STREAM_OVERLAP:-0}"
# P1-E data prefetch queue depth. 1 = legacy single-slot. >=2 keeps
# N batches in flight on the H2D copy stream to absorb host-side
# `next(iter)` jitter. Capped at 4 in code to avoid unbounded memory.
export DATA_PREFETCH_DEPTH="${DATA_PREFETCH_DEPTH:-1}"
export FORWARD_CUDA_GRAPH="${FORWARD_CUDA_GRAPH:-1}"
export STEP_CUDA_GRAPH="${STEP_CUDA_GRAPH:-0}"
# Direction-C PoC: opt-in switch to relax the legacy
# ``num_local_microbatches == 1`` gate on step CUDA Graph. Only
# meaningful with STEP_CUDA_GRAPH=1 + WGRAD_OVERLAP=0 + accum > 1.
export STEP_CUDA_GRAPH_ACCUM="${STEP_CUDA_GRAPH_ACCUM:-0}"
# Direction-C path-1: device-side grad emit allows step CUDA Graph
# to coexist with BucketedGradReducer (wgrad_overlap=1). Requires
# STEP_CUDA_GRAPH=1 + WGRAD_OVERLAP=1. Default off.
export STEP_CUDA_GRAPH_REDUCER="${STEP_CUDA_GRAPH_REDUCER:-0}"
export STEP_CUDA_GRAPH_FULL="${STEP_CUDA_GRAPH_FULL:-0}"
export STEP_CUDA_GRAPH_OPTIMIZER="${STEP_CUDA_GRAPH_OPTIMIZER:-0}"
export STEP_CUDA_GRAPH_NCCL_OPT="${STEP_CUDA_GRAPH_NCCL_OPT:-0}"
export CROSS_STEP_PIPELINE="${CROSS_STEP_PIPELINE:-0}"
export CUSTOM_GEMM="${CUSTOM_GEMM:-0}"
export EMIT_LOSS_LINES="${EMIT_LOSS_LINES:-0}"
export NCCL_ALGO="${NCCL_ALGO:-}"
export NCCL_PROTO="${NCCL_PROTO:-}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-}"

# STAGE selects which CLI subcommand + parameter shape to use:
#   - stable               : from-scratch random init (no --checkpoint-root,
#                            no --resume). Maps to
#                            ``training_engine_tensor pretrain``. Mirrors the
#                            Megatron "stable.sh" phase (iter 0 → 150000).
#   - stable_2             : continue from a previous self-built save dir
#                            (RESUME_DIR is required). Maps to
#                            ``training_engine_tensor train --resume <RESUME_DIR>``.
#                            --checkpoint-root is still required by argparse
#                            but ignored in the resume code path. Mirrors the
#                            Megatron "stable.0501.sh" phase
#                            (iter 150000 → 250000).
#   - decay_from_megatron  : decay phase (iter 250000 → 300000) bootstrapped
#                            from a *Megatron* distcp ckpt that was extracted
#                            into ``canonical_state_fp32.pt + canonical_optimizer_state_fp32.pt``
#                            via ``workload/scripts/extract_optimizer_state_from_distcp.py``.
#                            Maps to ``train`` with --checkpoint-root pointing
#                            at the canonical dir. Default (backward-compatible
#                            with existing ``no_mtp_custom_ops_64gpu.json``);
#                            also alias-accessible as ``decay``.
#   - decay_from_ours      : decay phase but continue from our self-built
#                            stable_2 save dir (RESUME_DIR required). Same
#                            mechanic as stable_2 — train + --resume — with a
#                            dummy --checkpoint-root the resume code path
#                            ignores. Use this when you already ran stable +
#                            stable_2 in this framework and want to skip the
#                            Megatron extract step.
STAGE="${STAGE:-decay_from_megatron}"
# Alias: ``decay`` (legacy) = ``decay_from_megatron`` (current explicit name).
if [[ "$STAGE" == "decay" ]]; then
    STAGE="decay_from_megatron"
fi

# Training parameters (configurable via env). All path-typed defaults are
# either repo-relative (so the checkout is self-contained) or empty (with
# a STAGE-aware fail-fast below) — no individual-user paths are baked in.
#
# CHECKPOINT_ROOT
#   - decay_from_megatron : init source dir holding canonical_state_fp32.pt
#                           + canonical_optimizer_state_fp32.pt; required.
#   - stable_2 / decay_from_ours : argparse-required but ignored on the
#                           --resume code path; an empty placeholder is OK.
#   - stable             : not passed to CLI.
# SAVE_DIR
#   Where ``training_state.pt`` is written every SAVE_INTERVAL steps.
#   Defaults under the repo (``.checkpoints/<stage>``) so a brand-new
#   checkout works without external configuration; override per-spec for
#   shared-FS production runs.
# RESUME_DIR
#   Path to a previous-phase save dir; required for stable_2 /
#   decay_from_ours, optional for decay_from_megatron, ignored for stable.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
NUM_STEPS="${NUM_STEPS:-300000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-640}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SEED="${SEED:-1234}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
SAVE_DIR="${SAVE_DIR:-${ENGINE_ROOT}/.checkpoints/${STAGE}}"
RESUME_DIR="${RESUME_DIR:-}"

# STAGE-specific validation + side effects
case "$STAGE" in
    stable)
        # ``pretrain`` subcommand sets RANDOM_INIT=1 internally when
        # --resume is absent; we don't want to also pass a stale resume
        # path picked up from an env default.
        RESUME_DIR=""
        SUBCOMMAND="pretrain"
        ;;
    stable_2|decay_from_ours)
        # Both stages bootstrap from a self-built save dir; the only
        # difference is the value of RESUME_DIR (stable end vs stable_2
        # end) and the LR scheduler shape (configured per-spec via
        # WSD_DECAY_ITERS). The entry script handles both the same way.
        if [[ -z "${RESUME_DIR:-}" ]]; then
            echo "ERROR: STAGE=$STAGE requires RESUME_DIR (path to the previous phase save dir, e.g. <SAVE_DIR>/step_<NNNNNNN> from the prior run)" >&2
            exit 2
        fi
        # CHECKPOINT_ROOT is required by argparse but ignored on the
        # resume code path; substitute a harmless placeholder if the
        # spec did not set one, so argparse does not get an empty value.
        if [[ -z "${CHECKPOINT_ROOT:-}" ]]; then
            CHECKPOINT_ROOT="${ENGINE_ROOT}/.checkpoints/_unused_for_resume"
        fi
        SUBCOMMAND="train"
        ;;
    decay_from_megatron)
        # Train from a Megatron-extracted canonical_*.pt at CHECKPOINT_ROOT.
        # RESUME_DIR is optional — only set it when restarting after a
        # crash, in which case it points at our own decay save dir.
        if [[ -z "${CHECKPOINT_ROOT:-}" ]]; then
            echo "ERROR: STAGE=$STAGE requires CHECKPOINT_ROOT (dir holding canonical_state_fp32.pt extracted from a Megatron distcp checkpoint)" >&2
            exit 2
        fi
        SUBCOMMAND="train"
        ;;
    *)
        echo "ERROR: unknown STAGE=$STAGE (allowed: stable, stable_2, decay_from_megatron, decay_from_ours, decay)" >&2
        exit 2
        ;;
esac

# Multi-Token Prediction (MTP / DeepSeek-V3 style). Defaults to disabled.
# When MTP_NUM_LAYERS > 0, forwards both flags to the CLI parser; we
# also export them so the env-time defaults inside config.py see them
# before the CLI parser takes effect.
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
MTP_LOSS_SCALING_FACTOR="${MTP_LOSS_SCALING_FACTOR:-0.1}"
export MTP_NUM_LAYERS MTP_LOSS_SCALING_FACTOR

# Data path — historically this script wired ``--data-path`` (modelbest_sdk
# mmap shards listed in DATA_PATH_FILE) into the CLI. zz/no_harness retired
# that backend; the engine now only accepts ``--hf-dataset`` (single local
# or Hub HF dataset) for both ``train`` and ``pretrain``. DATA_PATH /
# DATA_PATH_FILE are still echoed below for compatibility with older spec
# envVars but are NOT forwarded to the CLI any more. Set HF_DATASET (+
# HF_DATASET_CONFIG/HF_DATASET_SPLIT/HF_TEXT_FIELD/HF_TEXT_TEMPLATE/
# TOKENIZER_PATH) in spec envVars to provide data to the new path.
DATA_PATH_FILE="${DATA_PATH_FILE:-}"
if [[ -n "$DATA_PATH_FILE" && -f "$DATA_PATH_FILE" ]]; then
    echo "  DATA_PATH_FILE=$DATA_PATH_FILE  (informational; engine reads HF_DATASET, not --data-path)"
fi
if [[ -z "${HF_DATASET:-}" ]]; then
    echo "ERROR: HF_DATASET must be set (engine CLI no longer accepts --data-path / modelbest_sdk shards on zz/no_harness)." >&2
    exit 2
fi

# pytorchjob sets WORLD_SIZE=num_nodes, RANK=node_rank
NNODES=${WORLD_SIZE:-8}
NODE_RANK=${RANK:-0}

echo "=== Custom Engine 64-GPU Run (train CLI) ==="
echo "  STAGE=$STAGE SUBCOMMAND=$SUBCOMMAND"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29500}"
echo "  GBS=$GLOBAL_BATCH_SIZE MBS=$MICRO_BATCH_SIZE NUM_STEPS=$NUM_STEPS"
if [[ "$STAGE" == "decay_from_megatron" ]]; then
    echo "  CHECKPOINT_ROOT=$CHECKPOINT_ROOT  (used as init source)"
elif [[ "$STAGE" == "stable_2" || "$STAGE" == "decay_from_ours" ]]; then
    echo "  CHECKPOINT_ROOT=$CHECKPOINT_ROOT  (passed for argparse, ignored on resume path)"
fi
echo "  SAVE_DIR=$SAVE_DIR SAVE_INTERVAL=$SAVE_INTERVAL"
echo "  RANDOM_INIT=$RANDOM_INIT"
if [[ -n "$RESUME_DIR" ]]; then
    echo "  RESUME_DIR=$RESUME_DIR"
fi
echo "  MTP_NUM_LAYERS=$MTP_NUM_LAYERS MTP_LOSS_SCALING_FACTOR=$MTP_LOSS_SCALING_FACTOR"
echo "  HF_DATASET=$HF_DATASET${HF_DATASET_CONFIG:+ (cfg=$HF_DATASET_CONFIG)}${HF_DATASET_SPLIT:+ (split=$HF_DATASET_SPLIT)}"

# Build CLI args. The ``pretrain`` subcommand does not accept
# --checkpoint-root (it generates random params), so omit it for STAGE=stable.
TRAIN_ARGS=(
    --num-steps "$NUM_STEPS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --seq-length "$SEQ_LENGTH"
    --seed "$SEED"
    --save-interval "$SAVE_INTERVAL"
    --save-dir "$SAVE_DIR"
    --mup-emb-scale "${MUP_EMB_SCALE:-12}"
    --mup-depth-scale "${MUP_DEPTH_SCALE:-1.4}"
    --max-lr "${MAX_LR:-1e-2}"
    --min-lr "${MIN_LR:-5e-4}"
    --warmup-iters "${WARMUP_ITERS:-2000}"
    --lr-decay-iters "${LR_DECAY_ITERS:-300000}"
    --wsd-decay-iters "${WSD_DECAY_ITERS:-50000}"
    --hf-dataset "$HF_DATASET"
)
if [[ -n "${HF_DATASET_CONFIG:-}" ]]; then
    TRAIN_ARGS+=(--hf-dataset-config "$HF_DATASET_CONFIG")
fi
if [[ -n "${HF_DATASET_SPLIT:-}" ]]; then
    TRAIN_ARGS+=(--hf-dataset-split "$HF_DATASET_SPLIT")
fi
if [[ -n "${HF_TEXT_FIELD:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word-split: multi-column case
    TRAIN_ARGS+=(--hf-text-field ${HF_TEXT_FIELD})
fi
if [[ -n "${HF_TEXT_TEMPLATE:-}" ]]; then
    TRAIN_ARGS+=(--hf-text-template "$HF_TEXT_TEMPLATE")
fi
if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    TRAIN_ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ "$STAGE" != "stable" ]]; then
    TRAIN_ARGS=(--checkpoint-root "$CHECKPOINT_ROOT" "${TRAIN_ARGS[@]}")
fi
if [[ -n "$RESUME_DIR" ]]; then
    TRAIN_ARGS+=(--resume "$RESUME_DIR")
fi
if [[ -n "${DUMP_INPUTS_DIR:-}" ]]; then
    TRAIN_ARGS+=(--dump-inputs-dir "$DUMP_INPUTS_DIR")
fi
if [[ "$MTP_NUM_LAYERS" -gt 0 ]]; then
    TRAIN_ARGS+=(
        --mtp-num-layers "$MTP_NUM_LAYERS"
        --mtp-loss-scaling-factor "$MTP_LOSS_SCALING_FACTOR"
    )
fi

torchrun \
    --nproc_per_node=8 \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="${MASTER_ADDR:-localhost}" \
    --master_port="${MASTER_PORT:-29500}" \
    -m training_engine_tensor "$SUBCOMMAND" \
    "${TRAIN_ARGS[@]}"
