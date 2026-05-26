#!/bin/bash
set -euo pipefail
# HuggingFace dataset pretraining entry for self-developed training engine.
# Uses --hf-dataset instead of modelbest_sdk --data-path.
#
# Required env vars:
#   HF_DATASET        — local path or HuggingFace Hub name
#   TOKENIZER_PATH    — HuggingFace tokenizer path
# Optional env vars:
#   HF_DATASET_CONFIG, HF_DATASET_SPLIT, HF_TEXT_TEMPLATE, HF_TEXT_FIELD
#   GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, SEQ_LENGTH, NUM_STEPS, SEED
#   CUSTOM_GEMM, OP_ATTENTION, OP_GEMM_*

# ── Resolve ENGINE_ROOT from this script's location ──────────────────
# Avoid hard-coding an individual-user path: derive from BASH_SOURCE so
# the script works for any checkout. Override only if absolutely needed.
ENGINE_ROOT="${ENGINE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ENGINE_ROOT

# ── CUDA memory ──
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# ── HuggingFace deps ──
# Optional user-supplied environment unset hook (e.g. PYTHONPATH wipe
# for a sealed venv). Skipped silently when not provided.
if [[ -n "${USER_ENV_UNSET_HOOK:-}" && -f "${USER_ENV_UNSET_HOOK}" ]]; then
    # shellcheck disable=SC1090
    source "${USER_ENV_UNSET_HOOK}" 2>/dev/null || true
fi

# HF_SITE: where datasets/transformers/sentencepiece live (target dir
# for ``pip install --target``). Defaults to in-repo so the install
# follows the checkout and does not write to an individual-user dir.
HF_SITE="${HF_SITE:-${ENGINE_ROOT}/.hf_site}"
if ! python -c "import datasets, transformers, sentencepiece" 2>/dev/null; then
    pip install --target "$HF_SITE" -i "${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
        datasets transformers sentencepiece
fi
export PYTHONPATH="${HF_SITE}:${PYTHONPATH:-}"

# ── Custom GEMM / FA4 wheel install ──
if [[ "${CUSTOM_GEMM:-0}" == "1" || "${OP_ATTENTION:-baseline}" == "v1" || "${ATTN_BWD_OVERRIDE:-}" == "flash_dsl" ]]; then
    # Wheel directory: caller exports CUTLASS_WHEELS_DIR in the job
    # spec envVars; no individual-user default is provided here.
    _CUTLASS_WHEELS="${CUTLASS_WHEELS_DIR:-}"
    if [[ -n "$_CUTLASS_WHEELS" && -d "$_CUTLASS_WHEELS" ]]; then
        for _whl in \
            "cuda_pathfinder-1.5.4-py3-none-any.whl" \
            "apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
            "cuda_bindings-12.9.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl" \
            "nvidia_cutlass_dsl-4.4.2-py3-none-any.whl" \
            "nvidia_cutlass_dsl_libs_base-4.4.2-cp312-cp312-manylinux_2_28_x86_64.whl"; do
            [[ -f "$_CUTLASS_WHEELS/$_whl" ]] && \
                pip install --break-system-packages --no-deps "$_CUTLASS_WHEELS/$_whl" 2>/dev/null || true
        done
    fi
fi

# ── Engine source ──
export PYTHONPATH="${ENGINE_ROOT}/src:${PYTHONPATH:-}"

# ── Dataset config ──
HF_DATASET="${HF_DATASET:?HF_DATASET must be set (local path or Hub name)}"
TOKENIZER_PATH="${TOKENIZER_PATH:?TOKENIZER_PATH must be set}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-train}"
HF_TEXT_TEMPLATE="${HF_TEXT_TEMPLATE:-}"
HF_TEXT_FIELD="${HF_TEXT_FIELD:-}"

# ── Training config ──
NUM_STEPS="${NUM_STEPS:-200}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1280}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
SEED="${SEED:-1234}"

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

echo "=== HF Dataset Pretraining ==="
echo "  HF_DATASET=$HF_DATASET"
echo "  TOKENIZER_PATH=$TOKENIZER_PATH"
echo "  HF_DATASET_CONFIG=${HF_DATASET_CONFIG:-auto}"
echo "  HF_DATASET_SPLIT=$HF_DATASET_SPLIT"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK"
echo "  GBS=$GLOBAL_BATCH_SIZE MBS=$MICRO_BATCH_SIZE SEQ_LEN=$SEQ_LENGTH"
echo "  NUM_STEPS=$NUM_STEPS"

TRAIN_ARGS=(
    --num-steps "$NUM_STEPS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --seq-length "$SEQ_LENGTH"
    --seed "$SEED"
    --hf-dataset "$HF_DATASET"
    --tokenizer-path "$TOKENIZER_PATH"
    --hf-dataset-split "$HF_DATASET_SPLIT"
)
[[ -n "$HF_DATASET_CONFIG" ]] && TRAIN_ARGS+=(--hf-dataset-config "$HF_DATASET_CONFIG")
[[ -n "$HF_TEXT_TEMPLATE" ]] && TRAIN_ARGS+=(--hf-text-template "$HF_TEXT_TEMPLATE")
[[ -n "$HF_TEXT_FIELD" ]]    && TRAIN_ARGS+=(--hf-text-field $HF_TEXT_FIELD)

torchrun \
    --nproc_per_node=8 \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="${MASTER_ADDR:-localhost}" \
    --master_port="${MASTER_PORT:-29500}" \
    -m training_engine_tensor pretrain \
    "${TRAIN_ARGS[@]}"
