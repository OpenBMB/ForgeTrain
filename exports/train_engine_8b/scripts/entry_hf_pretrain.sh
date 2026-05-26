#!/bin/bash
# ----------------------------------------------------------------------------
# HuggingFace dataloader pretraining entry for the MiniCPM4-8B engine.
#
# Single-host launcher: torchrun fans the process out across
# nproc_per_node ranks (typically 8 on a single H100 node).  Multi-node
# is supported through the standard NNODES / NODE_RANK / MASTER_ADDR /
# MASTER_PORT environment variables.
#
# Required env vars:
#   HF_DATASET            HuggingFace dataset path (local dir or Hub name).
#   TOKENIZER_PATH        HuggingFace tokenizer path.
#
# Optional env vars:
#   HF_DATASET_CONFIG     Subset name passed as the ``--config`` arg.
#   HF_DATASET_SPLIT      Split name (default: ``train``).
#   HF_TEXT_TEMPLATE      Python format string applied to row dicts.
#   HF_TEXT_FIELD         Whitespace-separated text column names.
#   HF_DATA_FORMAT        ``parquet`` / ``json`` / ... (use with HF_DATA_FILES).
#   HF_DATA_FILES         Whitespace-separated globs for HF_DATA_FORMAT.
#
#   NUM_STEPS             Default 100.
#   GLOBAL_BATCH_SIZE     Default 64 (= MBS=2 * GAS=8 * DP=4).
#   MICRO_BATCH_SIZE      Default 2.
#   GRAD_ACCUM_STEPS      Default 8 (set to 1 for a single-step path).
#   SEQ_LENGTH            Default 4096.
#   SEED                  Default 1234.
#
#   NPROC_PER_NODE        Default 8.
#   NNODES                Default 1 (or take WORLD_SIZE if set).
#   NODE_RANK             Default 0 (or take RANK if set).
#   MASTER_ADDR           Default localhost.
#   MASTER_PORT           Default 29500.
#
#   CUSTOM_GEMM           Default 0.  Set to 1 to enable engine GEMMs that
#                         have a non-baseline kernel (gemm_fc1, gemm_output).
#   OP_GEMM_FC1           ``baseline`` or ``v1`` (default ``v1`` when
#                         CUSTOM_GEMM=1, else ``baseline``).
#   OP_GEMM_OUTPUT        ``baseline`` or ``v1``.
#   OP_ATTENTION          ``v1`` (engine DSL fwd, default) or ``baseline``
#                         (cuDNN through autograd).  ``v1`` matches the
#                         dataclass defaults for
#                         ``OURS_DIRECT_ATTN`` / ``OURS_DSL_ATTN_FWD`` /
#                         ``OURS_ROPE_BHSD`` — together they remove the
#                         attention-boundary permute().contiguous() copy
#                         and switch RoPE to the [B,H,S,D] BHSD layout.
#                         ``baseline`` is a kill-switch: this script then
#                         exports the three OURS_* flags to ``0`` so the
#                         engine falls back to the cuDNN autograd path.
#   CUTLASS_DSL_PATH      Optional; sys.path hint for the CuTeDSL wheel.
#
#   EXPORT_REPORT         Optional path to write a per-step JSON report.
#   SAVE_CHECKPOINT_DIR   Optional directory to persist a resume-able
#                         checkpoint (one ``rank_<R>.pt`` per rank) after the
#                         final step.
# ----------------------------------------------------------------------------

set -euo pipefail

# ── Resolve ENGINE_ROOT from this script's location ──────────────────────────
ENGINE_ROOT="${ENGINE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ENGINE_ROOT

# ── CUDA / NCCL runtime knobs (safe defaults) ────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"

# ── Engine fused-kernel switches (safe defaults; override per-run) ───────────
# These flip on the fused, fp32-grad-accum, RoPE-fused, direct-attention paths
# inside the engine.  
export FP32_GRAD_ACCUM="${FP32_GRAD_ACCUM:-1}"
export TE_ROPE="${TE_ROPE:-1}"
export FUSED_OPS="${FUSED_OPS:-1}"
export FUSE_DIRECT_ATTN="${FUSE_DIRECT_ATTN:-1}"
export GRAD_DIAG="${GRAD_DIAG:-1}"

# ── Engine source on PYTHONPATH ──────────────────────────────────────────────
export PYTHONPATH="${ENGINE_ROOT}/src:${PYTHONPATH:-}"

# ── Required dataset args ────────────────────────────────────────────────────
HF_DATASET="${HF_DATASET:?HF_DATASET must be set (local path or Hub name)}"
TOKENIZER_PATH="${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-train}"
HF_TEXT_TEMPLATE="${HF_TEXT_TEMPLATE:-}"
HF_TEXT_FIELD="${HF_TEXT_FIELD:-}"
HF_DATA_FORMAT="${HF_DATA_FORMAT:-}"
HF_DATA_FILES="${HF_DATA_FILES:-}"

# ── Training shape ───────────────────────────────────────────────────────────
NUM_STEPS="${NUM_STEPS:-100}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SEED="${SEED:-1234}"

# ── Operator dispatch ────────────────────────────────────────────────────────
CUSTOM_GEMM="${CUSTOM_GEMM:-0}"
export CUSTOM_GEMM
if [[ "${CUSTOM_GEMM}" == "1" ]]; then
    export OP_GEMM_FC1="${OP_GEMM_FC1:-v1}"
    export OP_GEMM_OUTPUT="${OP_GEMM_OUTPUT:-v1}"
fi
export OP_GEMM_FC1="${OP_GEMM_FC1:-baseline}"
export OP_GEMM_OUTPUT="${OP_GEMM_OUTPUT:-baseline}"
# Operators below have only a baseline implementation:
export OP_GEMM_FC2=baseline
export OP_GEMM_QKV_PROJ=baseline
export OP_GEMM_ATTN_OUT_PROJ=baseline

OP_ATTENTION="${OP_ATTENTION:-v1}"
if [[ "${OP_ATTENTION}" == "baseline" ]]; then
    # Kill-switch: revert to the cuDNN autograd attention path.  The
    # engine's dataclass defaults are on (direct_attn / dsl_attn_fwd /
    # rope_bhsd), so we MUST export them as 0 here to override
    # ``EngineConfig.from_env``'s env-bridge defaults.
    export OURS_DSL_ATTN_FWD=0
    export OURS_DIRECT_ATTN=0
    export OURS_ROPE_BHSD=0
fi

# ── Distributed launcher topology ────────────────────────────────────────────
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

EXPORT_REPORT="${EXPORT_REPORT:-}"
SAVE_CHECKPOINT_DIR="${SAVE_CHECKPOINT_DIR:-}"

echo "=== HF dataloader pretraining ==="
echo "  ENGINE_ROOT=${ENGINE_ROOT}"
echo "  HF_DATASET=${HF_DATASET}"
echo "  TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "  HF_DATASET_CONFIG=${HF_DATASET_CONFIG:-<unset>}"
echo "  HF_DATASET_SPLIT=${HF_DATASET_SPLIT}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE} NNODES=${NNODES} NODE_RANK=${NODE_RANK}"
echo "  GBS=${GLOBAL_BATCH_SIZE} MBS=${MICRO_BATCH_SIZE} GAS=${GRAD_ACCUM_STEPS} SEQ=${SEQ_LENGTH}"
echo "  NUM_STEPS=${NUM_STEPS} SEED=${SEED}"
echo "  CUSTOM_GEMM=${CUSTOM_GEMM} OP_ATTENTION=${OP_ATTENTION}"

TRAIN_ARGS=(
    --num-steps "${NUM_STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --seq-length "${SEQ_LENGTH}"
    --seed "${SEED}"
    --hf-dataset "${HF_DATASET}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --hf-dataset-split "${HF_DATASET_SPLIT}"
)
[[ -n "${HF_DATASET_CONFIG}" ]] && TRAIN_ARGS+=(--hf-dataset-config "${HF_DATASET_CONFIG}")
[[ -n "${HF_TEXT_TEMPLATE}" ]] && TRAIN_ARGS+=(--hf-text-template "${HF_TEXT_TEMPLATE}")
# shellcheck disable=SC2206
[[ -n "${HF_TEXT_FIELD}" ]]    && TRAIN_ARGS+=(--hf-text-field ${HF_TEXT_FIELD})
[[ -n "${HF_DATA_FORMAT}" ]]   && TRAIN_ARGS+=(--hf-data-format "${HF_DATA_FORMAT}")
# shellcheck disable=SC2206
[[ -n "${HF_DATA_FILES}" ]]    && TRAIN_ARGS+=(--hf-data-files ${HF_DATA_FILES})
[[ -n "${EXPORT_REPORT}" ]]        && TRAIN_ARGS+=(--export-report "${EXPORT_REPORT}")
[[ -n "${SAVE_CHECKPOINT_DIR}" ]]  && TRAIN_ARGS+=(--save-checkpoint-dir "${SAVE_CHECKPOINT_DIR}")

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m training_engine_tensor pretrain \
    "${TRAIN_ARGS[@]}"
