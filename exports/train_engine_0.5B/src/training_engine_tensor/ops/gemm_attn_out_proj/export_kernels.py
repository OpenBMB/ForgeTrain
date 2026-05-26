"""Export CuTeDSL persistent GEMM kernels for gemm_attn_out_proj.

Three directions, all M=40960 N=1024 K=1024 except wgrad:
  fwd:   Y[M=40960,N=1024]  = X[M,K=1024]  @ W^T[K,N]   BF16->BF16
  dgrad: dX[M=40960,N=1024] = dY[M,K=1024] @ W [K,N]    BF16->BF16
  wgrad: dW[M=1024,N=1024]  = dY^T[M,K=40960] @ X[K,N]  BF16->FP32 (split-K=2)

Run on remote GPU (single-shot). After running, .h/.o files appear in
EXPORT_DIR (defined here + in kernel.py).
"""
import os
import shutil
import subprocess
import sys
import torch
import cuda.bindings.driver as cuda

_fb = os.environ.get("CUTLASS_DSL_FALLBACK_DIR", "")
if _fb:
    _pkg = os.path.join(_fb, "python_packages")
    if os.path.isdir(_pkg) and _pkg not in sys.path:
        sys.path.insert(0, _pkg)

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import _HopperGemmPersistent  # noqa: E402

BF16 = cutlass.BFloat16
FP32 = cutlass.Float32


def compile_and_export(name, M, N, K, *, a_major, b_major, c_major,
                       c_dtype, tile, cluster, swizzle, raster_along_m,
                       mma_inst_tile_k, epi_stage=4, batch=1, export_dir,
                       is_dynamic_layout=True):
    print(f"\n=== {name}: M={M} N={N} K={K} batch={batch} ===")
    print(f"  a_major={a_major} b_major={b_major} c_major={c_major} c_dtype={c_dtype}")
    print(f"  tile={tile} cluster={cluster} swizzle={swizzle} raster_along_m={raster_along_m}")
    print(f"  mma_inst_tile_k={mma_inst_tile_k} epi_stage={epi_stage}")
    print(f"  is_dynamic_layout={is_dynamic_layout}")

    a_cpu = cutlass_torch.matrix(batch, M, K, a_major == "m", BF16)
    b_cpu = cutlass_torch.matrix(batch, N, K, b_major == "n", BF16)
    c_cpu = cutlass_torch.matrix(batch, M, N, c_major == "m", c_dtype)

    a_t, _ = cutlass_torch.cute_tensor_like(
        a_cpu, BF16, is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    b_t, _ = cutlass_torch.cute_tensor_like(
        b_cpu, BF16, is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    c_t, _ = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype, is_dynamic_layout=is_dynamic_layout, assumed_align=16)

    print(f"  a_cute stride={a_t.stride}")
    print(f"  b_cute stride={b_t.stride}")
    print(f"  c_cute stride={c_t.stride}")

    gemm = _HopperGemmPersistent(
        FP32, tile, cluster, swizzle_size=swizzle,
        raster_along_m=raster_along_m, mma_inst_tile_k=mma_inst_tile_k,
        epi_stage=epi_stage,
    )

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    hw = cutlass.utils.HardwareInfo()
    mac = hw.get_max_active_clusters(cluster[0] * cluster[1])

    compiled = cute.compile(gemm, a_t, b_t, c_t, mac, stream)
    print(f"  compiled OK; runtime args: {compiled.args_spec.args_spec.args}")

    out_dir = os.path.join(export_dir, name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    compiled.export_to_c(out_dir, name)
    for f in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"  exported: {f}  ({sz} bytes)")

    return out_dir


def main():
    # ``AOP_EXPORT_DIR`` was the original op-specific knob; ``CUTEDSL_CACHE_ROOT``
    # is the cross-op knob (matches kernel.py L1949 + the four other gemm
    # ops). When both are set, the op-specific one wins for backward
    # compat.
    export_dir = os.environ.get("AOP_EXPORT_DIR")
    if export_dir is None:
        export_dir = os.path.join(
            os.environ.get("CUTEDSL_CACHE_ROOT", "/tmp"),
            "cutedsl_export_gemm_attn_out_proj",
        )
    os.makedirs(export_dir, exist_ok=True)
    for child in os.listdir(export_dir):
        p = os.path.join(export_dir, child)
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
        else:
            shutil.rmtree(p, ignore_errors=True)

    # Round 41 update: read the SAME env-var names as kernel.py so a single
    # config controls both JIT and AOT-export builds. The lock defaults match
    # round 40 winners (sw=1 for all three, c=(1,1) for fwd/dgrad, c=(1,2)
    # for wgrad, raster=N for all, epi=8 for fwd/dgrad, epi=4 for wgrad).

    # fwd: Y = X @ W^T;  C[M,N] = A[M,K] * B[N,K]; A=X, B=W
    fwd_tile = (
        int(os.environ.get("AOP_FWD_TILE_M", 128)),
        int(os.environ.get("AOP_FWD_TILE_N", 256)),
    )
    fwd_cluster = (
        # R60: c(1,2) best (-10.0% vs cuBLAS)
        int(os.environ.get("AOP_FWD_CLU_M",
            os.environ.get("AOP_FWD_CLUSTER_M", 1))),
        int(os.environ.get("AOP_FWD_CLU_N",
            os.environ.get("AOP_FWD_CLUSTER_N", 2))),
    )
    fwd_sw = int(os.environ.get("AOP_FWD_SW", 8))  # R60: sw8 best
    fwd_ras_m = bool(int(os.environ.get("AOP_FWD_RAS_M", 0)))
    fwd_mma_k = int(os.environ.get("AOP_FWD_MMA_K", 4))
    fwd_epi = int(os.environ.get("AOP_FWD_EPI", 4))  # R60: auto/4 beats 8
    # R44: is_dynamic_layout=False bakes shapes into the kernel binary.
    # Descriptor becomes { void *data } — eliminates host-side fill overhead.
    compile_and_export("aop_fwd",
        M=40960, N=1024, K=1024,
        a_major="k", b_major="k", c_major="n",
        c_dtype=BF16,
        tile=fwd_tile, cluster=fwd_cluster, swizzle=fwd_sw,
        raster_along_m=fwd_ras_m, mma_inst_tile_k=fwd_mma_k,
        epi_stage=fwd_epi,
        export_dir=export_dir,
        is_dynamic_layout=False)

    # dgrad: dX = dY @ W;  C[M,N] = A[M,K] * B[N,K]
    # dY[M, K] row-major; W viewed as B[N, K] is col-major NK -> b_major=n.
    dgrad_tile = (
        int(os.environ.get("AOP_DGRAD_TILE_M", 128)),
        int(os.environ.get("AOP_DGRAD_TILE_N", 256)),
    )
    dgrad_cluster = (
        int(os.environ.get("AOP_DGRAD_CLU_M",
            os.environ.get("AOP_DGRAD_CLUSTER_M", 1))),
        int(os.environ.get("AOP_DGRAD_CLU_N",
            os.environ.get("AOP_DGRAD_CLUSTER_N", 1))),
    )
    dgrad_sw = int(os.environ.get("AOP_DGRAD_SW", 2))  # R60: sw2 best (-9.4% vs cuBLAS)
    dgrad_ras_m = bool(int(os.environ.get("AOP_DGRAD_RAS_M", 0)))
    dgrad_mma_k = int(os.environ.get("AOP_DGRAD_MMA_K", 4))
    dgrad_epi = int(os.environ.get("AOP_DGRAD_EPI", 4))  # R60: auto/4 beats 8
    compile_and_export("aop_dgrad",
        M=40960, N=1024, K=1024,
        a_major="k", b_major="n", c_major="n",
        c_dtype=BF16,
        tile=dgrad_tile, cluster=dgrad_cluster, swizzle=dgrad_sw,
        raster_along_m=dgrad_ras_m, mma_inst_tile_k=dgrad_mma_k,
        epi_stage=dgrad_epi,
        export_dir=export_dir,
        is_dynamic_layout=False)

    # wgrad: dW = dY^T @ X; batched split-K=2
    wgrad_tile = (
        int(os.environ.get("AOP_WGRAD_TILE_M", 128)),
        int(os.environ.get("AOP_WGRAD_TILE_N", 128)),
    )
    wgrad_cluster = (
        int(os.environ.get("AOP_WGRAD_CLU_M",
            os.environ.get("AOP_WGRAD_CLUSTER_M", 1))),
        int(os.environ.get("AOP_WGRAD_CLU_N",
            os.environ.get("AOP_WGRAD_CLUSTER_N", 2))),
    )
    wgrad_sw = int(os.environ.get("AOP_WGRAD_SW", 1))
    wgrad_ras_m = bool(int(os.environ.get(
        "AOP_WGRAD_RAS_M", os.environ.get("AOP_WGRAD_RASTER_M", 0))))
    wgrad_mma_k = int(os.environ.get("AOP_WGRAD_MMA_K", 4))
    wgrad_epi = int(os.environ.get("AOP_WGRAD_EPI", 4))
    wgrad_splits = int(os.environ.get("AOP_WGRAD_SPLITS", 2))
    wgrad_K_total = int(os.environ.get("AOP_WGRAD_K", 40960))
    wgrad_K_chunk = wgrad_K_total // wgrad_splits
    # R44: wgrad also uses static layout. gemm_output R48 proved that
    # static layout is net-positive for large descriptor-fill overhead.
    # For batched wgrad (L=2), the three descriptors go from
    # {ptr, shapes[3], strides[2]} to just {ptr}.
    compile_and_export("aop_wgrad",
        M=1024, N=1024, K=wgrad_K_chunk,
        a_major="m", b_major="n", c_major="n",
        c_dtype=FP32,
        tile=wgrad_tile, cluster=wgrad_cluster, swizzle=wgrad_sw,
        raster_along_m=wgrad_ras_m, mma_inst_tile_k=wgrad_mma_k,
        epi_stage=wgrad_epi,
        batch=wgrad_splits,
        export_dir=export_dir,
        is_dynamic_layout=False)

    print(f"\n=== All exports complete. Files under {export_dir} ===")
    for root, dirs, files in os.walk(export_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            print(f"  {os.path.relpath(fp, export_dir)} ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
