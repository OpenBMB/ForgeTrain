"""Pre-compile every custom GEMM + attention kernel into the persistent cache.

Run this ONCE on a 1-GPU pod after a base-image change or after a fresh
``CUSTOM_OPS_CACHE_ROOT``. Subsequent training jobs that point at the
same ``TORCH_EXTENSIONS_DIR`` and ``/tmp/cutedsl_export_<op>`` symlinks
will skip both the CuTeDSL AOT export and the C++ link step, going from
~40 minutes of cold-start build to ~5 seconds of warm ``dlopen``.

Triggers compilation by:

  1. ``import``-ing each ``ops.<op>.kernel`` module (causes the lazy
     ``_get_ext()`` path to fire on first call).
  2. Running ONE forward + backward call per op on small tensors so
     ``_bind_fast_paths()`` reaches its happy path and the .so is
     loaded from disk (so torch's cpp_extension sees a hit on the
     SECOND run).

Only the GEMM ops actually compile C++ on first call. The attention
kernel uses pure Python CuTeDSL JIT and benefits less from caching, but
we still touch it so the FA4 spec import side-effects (which take
~10s of import-time work) are dropped onto the persistent torch
extension cache.

Env vars consumed (must match the entry script):
  CUSTOM_OPS_CACHE_ROOT  default ``${ENGINE_ROOT}/.persist_cache``
  OPS_CACHE_TAG          default ``ngc_47165eea_py312``

This script must be invoked AFTER the entry-script preamble (cutlass
install + symlinks + LD_PRELOAD) so it sees the same paths the
training pods will see.
"""

from __future__ import annotations

import os
import sys
import time

import torch

# ── op_dispatcher must be initialised BEFORE any ops kernel module is
# imported. Several kernel.py files (gemm_fc1, gemm_fc2, gemm_qkv_proj,
# gemm_output) compute ``EXPORT_DIR = _get_build_env().cutedsl_cache_root``
# at module-import time, and ``get_build_env()`` raises RuntimeError if
# ``init()`` has not run. ``__main__.py`` calls init() before any kernel
# import in real training jobs, but precompile_ops.py is a standalone
# script that bypasses ``__main__.py``, so we replicate the init here.
# Without this, the first ``from training_engine_tensor.ops.gemm_fc1.kernel
# import ...`` below raises and the persistent AOT cache never gets
# warmed — every training pod then pays the full ~40 min cold-start cost.
from training_engine_tensor.op_dispatcher import init as _init_op_dispatcher

_init_op_dispatcher(env=dict(os.environ))


def _shape(name: str) -> tuple[int, int, int]:
    """Production (M, N, K) per op.

    The CuTeDSL AOT-exported C++ backend bakes M / N / K as compile-time
    constants and TORCH_CHECKs the runtime tensor sizes against them, so
    the precompile MUST exercise the exact production shape — calling
    with a smaller M (e.g. 1024) immediately throws
    ``fwd_fast: expected M=40960 ... got M=1024`` and the persistent
    cache never picks up the .so.

    Production shape comes from MiniCPM4-0.5B at MICRO_BATCH_SIZE=10 ×
    MAX_SEQ_LENGTH=4096 = 40960 tokens per microbatch on every rank;
    K=1024 hidden, FC intermediate=8192 (gated), FC unfused output
    intermediate=4096, head_dim*num_heads=1280 for QKV proj, vocab=73448
    (padded inside ``gemm_output`` to 73472). Keep these in lockstep
    with the per-op ``gemm_cutedsl.cpp`` ``"expected M=..."`` constants.
    """
    return {
        "gemm_fc1": (40960, 8192, 1024),
        "gemm_fc2": (40960, 1024, 4096),
        "gemm_qkv_proj": (40960, 1280, 1024),
        "gemm_attn_out_proj": (40960, 1024, 1024),
        "gemm_output": (40960, 73448, 1024),
    }[name]


def _exercise(op_name: str, fwd_fn, bwd_fn) -> None:
    M, N, K = _shape(op_name)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    t0 = time.time()
    y = fwd_fn(x, w)
    if y is None:
        raise RuntimeError(f"{op_name}: fwd_fn returned None — kernel disabled?")
    torch.cuda.synchronize()
    t_fwd = time.time() - t0
    print(f"  {op_name} fwd OK ({M}x{N}x{K}) in {t_fwd*1000:.1f} ms")

    if bwd_fn is None:
        print(f"  {op_name} bwd skipped (no bwd_fn)")
        return
    g = torch.randn_like(y)
    t0 = time.time()
    out = bwd_fn(g, x, w, te_wgrad=True)
    if out is None:
        raise RuntimeError(f"{op_name}: bwd_fn returned None")
    torch.cuda.synchronize()
    t_bwd = time.time() - t0
    print(f"  {op_name} bwd OK in {t_bwd*1000:.1f} ms")


def main() -> int:
    print("=" * 70)
    print("precompile_custom_ops: warming the persistent kernel cache")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}, device={torch.cuda.get_device_name(0)}")
    print(f"  TORCH_EXTENSIONS_DIR={os.environ.get('TORCH_EXTENSIONS_DIR', '<unset>')}")
    print(f"  CUTEDSL_CACHE_ROOT={os.environ.get('CUTEDSL_CACHE_ROOT', '<unset>')}")
    print(f"  LD_PRELOAD={os.environ.get('LD_PRELOAD', '<unset>')}")

    print("\n[1/5] gemm_fc1 ...")
    from training_engine_tensor.ops.gemm_fc1.kernel import gemm_fc1_fwd, gemm_fc1_bwd
    _exercise("gemm_fc1", gemm_fc1_fwd, gemm_fc1_bwd)

    print("\n[2/5] gemm_fc2 ...")
    from training_engine_tensor.ops.gemm_fc2.kernel import gemm_fc2_fwd, gemm_fc2_bwd
    _exercise("gemm_fc2", gemm_fc2_fwd, gemm_fc2_bwd)

    print("\n[3/5] gemm_qkv_proj ...")
    from training_engine_tensor.ops.gemm_qkv_proj.kernel import gemm_qkv_proj_fwd, gemm_qkv_proj_bwd
    _exercise("gemm_qkv_proj", gemm_qkv_proj_fwd, gemm_qkv_proj_bwd)

    print("\n[4/5] gemm_attn_out_proj ...")
    # Force the C-export AOT backend for attn_out_proj. By default this op
    # uses the Python JIT path (Round 26 bench: JIT wgrad 1.10x cuBLAS vs
    # C-export 1.18x cuBLAS — JIT is ~7% faster on wgrad due to TMA-
    # descriptor reuse), but JIT compilation outputs are NOT persisted to
    # disk, forcing every training pod to re-run ~100 cute.compile calls
    # on cold start (~9 min × 16 ranks). Switching to C-export here makes
    # the precompile job populate the same TORCH_EXTENSIONS_DIR the other
    # 4 GEMMs already use, so subsequent training pods dlopen the cached
    # .so in ~ms instead of re-JITing. The steady-state perf cost is
    # ~2.5 ms / step (0.09% of a 2.8 s step), well below noise.
    # Production training jobs MUST also set
    # OP_GEMM_ATTN_OUT_PROJ_BACKEND=cexport for the cache hit to land —
    # see workload/scripts/mtp_custom_ops_200steps.json.
    os.environ["OP_GEMM_ATTN_OUT_PROJ_BACKEND"] = "cexport"
    from training_engine_tensor.ops.gemm_attn_out_proj.kernel import (
        gemm_attn_out_proj_fwd, gemm_attn_out_proj_bwd,
    )
    _exercise("gemm_attn_out_proj", gemm_attn_out_proj_fwd, gemm_attn_out_proj_bwd)

    print("\n[5/5] gemm_output ...")
    from training_engine_tensor.ops.gemm_output.kernel import gemm_output_fwd, gemm_output_bwd
    _exercise("gemm_output", gemm_output_fwd, gemm_output_bwd)

    print("\n" + "=" * 70)
    print("All custom GEMM kernels compiled and cached. Done.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
