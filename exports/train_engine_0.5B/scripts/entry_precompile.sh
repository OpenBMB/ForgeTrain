#!/bin/bash
set -euo pipefail
# One-shot: install cutlass + populate the persistent kernel cache used
# by ``entry_minicpm4_0_5b_new_sst_64gpu.sh``. Submit this as a 1-GPU
# BATCH job once per (image, cache_root) tuple; subsequent training
# jobs that point at the same TORCH_EXTENSIONS_DIR + symlinks will
# skip both AOT export and C++ link.

# Derive ENGINE_ROOT from script location (no hard-coded user path).
ENGINE_ROOT="${ENGINE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ENGINE_ROOT

PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.hs1.paratera.com/root/pypi/+simple}"

install_pkg() {
    python -m pip install --no-cache-dir --retries 1 --timeout 15 "$@" \
        -i "$PYPI_INDEX_URL" 2>/dev/null || \
    python -m pip install --no-cache-dir --retries 1 --timeout 15 "$@" 2>/dev/null || true
}

python -c "import sentencepiece" 2>/dev/null || install_pkg sentencepiece
python -c "import modelbest_sdk" 2>/dev/null || install_pkg modelbest_sdk

cd "$ENGINE_ROOT"

export PYTHONPATH="${ENGINE_ROOT}/src:${PYTHONPATH:-}"

# Match the env vars in entry_minicpm4_0_5b_new_sst_64gpu.sh that affect
# operator selection so every kernel that production touches is also
# touched here.
export CUSTOM_GEMM=1
export OP_ATTENTION="${OP_ATTENTION:-v1}"
export OP_GEMM_QKV_PROJ="${OP_GEMM_QKV_PROJ:-v1}"
export OP_GEMM_ATTN_OUT_PROJ="${OP_GEMM_ATTN_OUT_PROJ:-v1}"
export OP_GEMM_FC1="${OP_GEMM_FC1:-v1}"
export OP_GEMM_FC2="${OP_GEMM_FC2:-v1}"
export OP_GEMM_OUTPUT="${OP_GEMM_OUTPUT:-v1}"
export FUSE_GEMM_PAD="${FUSE_GEMM_PAD:-0}"
# Tell attn_out_proj to use the AOT C-export backend so its .so + .o get
# baked into the persistent torch_extensions cache. Production training
# jobs must mirror this env to actually hit the cache instead of falling
# back to the JIT path. See precompile_custom_ops.py for context.
export OP_GEMM_ATTN_OUT_PROJ_BACKEND="${OP_GEMM_ATTN_OUT_PROJ_BACKEND:-cexport}"

# ── Cutlass install ──────────────────────────────────────────────────
if ! python -c "import cutlass" 2>/dev/null; then
    # Wheel directory holds the 5 CUTLASS-DSL wheels below. Set
    # CUTLASS_WHEELS_DIR explicitly in the job spec envVars; no
    # individual-user default is provided here.
    _CUTLASS_WHEELS="${CUTLASS_WHEELS_DIR:-}"
    if [[ -z "$_CUTLASS_WHEELS" || ! -d "$_CUTLASS_WHEELS" ]]; then
        echo "[cutlass] FATAL: CUTLASS_WHEELS_DIR not set or missing"
        echo "  point it at the dir containing the cutlass-dsl wheels."
        exit 11
    fi
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
fi
python -c "import cutlass" || { echo "[cutlass] FATAL: import failed"; exit 11; }
echo "[cutlass] OK"

# ── Persistent cache layout (matches entry_minicpm4_0_5b_new_sst_64gpu.sh) ──
# Default cache root lives in-repo (.persist_cache/) so the kernel
# cache follows the checkout and never depends on a user-specific
# absolute path. Override with CUSTOM_OPS_CACHE_ROOT for shared caches.
_CACHE_ROOT="${CUSTOM_OPS_CACHE_ROOT:-${ENGINE_ROOT}/.persist_cache}"
_IMG_TAG="${OPS_CACHE_TAG:-ngc_47165eea_py312}"
_PERSIST="$_CACHE_ROOT/$_IMG_TAG"
mkdir -p "$_PERSIST/torch_extensions" "$_PERSIST/cutedsl_jit_cache"
export TORCH_EXTENSIONS_DIR="$_PERSIST/torch_extensions"
export CUTEDSL_CACHE_ROOT="$_PERSIST"
# Mirror the production env so any cute.compile JIT calls touched here
# (currently NONE in precompile_custom_ops.py — gemm_attn_out_proj
# warmup not yet wired in — but kept symmetric for future-proofing) also
# write into the persistent JIT IR cache.
export CUTE_DSL_CACHE_DIR="$_PERSIST/cutedsl_jit_cache"
echo "[precompile] TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "[precompile] CUTEDSL_CACHE_ROOT=$CUTEDSL_CACHE_ROOT"
echo "[precompile] CUTE_DSL_CACHE_DIR=$CUTE_DSL_CACHE_DIR"

if [[ -f /usr/local/cuda/lib64/libcudart.so.12 ]]; then
    export LD_PRELOAD="${LD_PRELOAD:-}${LD_PRELOAD:+:}/usr/local/cuda/lib64/libcudart.so.12"
fi

# ── Run the warmup ───────────────────────────────────────────────────
python "$ENGINE_ROOT/scripts/precompile_ops.py"

# Sanity — list resulting cache contents so we can spot empty dirs.
echo "[precompile] cache contents after warmup:"
du -sh "$_CACHE_ROOT/$_IMG_TAG"/{torch_extensions,cutedsl_export_*} 2>&1 | head -20
