"""Export CuTeDSL GEMM kernels to C object files for gemm_fc2 (MLP down projection).

Shape-specialized for MiniCPM4 0.5B:
  fwd:   Y[40960,1024] = X[40960,4096] @ W^T   BF16→BF16   static layout
  dgrad: dX[40960,4096] = dY[40960,1024] @ W    BF16→BF16   static layout
  wgrad: dW[1024,4096] = dY^T @ X               BF16→FP32   dynamic layout (col-major views)

Uses HopperWgmmaGemmPersistentKernel from gemm_fc1/bench_cutedsl.py for proper
cutlass.Constexpr annotations needed by export_to_c().
"""
import os
import sys
import shutil
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

from training_engine_tensor.ops.gemm_fc1.bench_cutedsl import HopperWgmmaGemmPersistentKernel
from training_engine_tensor.op_dispatcher import get_build_env as _get_build_env
from training_engine_tensor.op_dispatcher import init as _init_build_env


def compile_and_export(name, M, N, K, a_major, b_major, c_major, c_dtype,
                       tile, cluster, swizzle, export_dir,
                       raster_along_m=False, mma_inst_tile_k=4,
                       is_dynamic_layout=True):
    print(f"\n=== Compiling {name}: M={M} N={N} K={K} ===")
    print(f"  a_major={a_major}, b_major={b_major}, c_major={c_major}, c_dtype={c_dtype}")
    print(f"  tile={tile}, cluster={cluster}, swizzle={swizzle}, "
          f"raster_along_m={raster_along_m}, is_dynamic_layout={is_dynamic_layout}")

    a_cpu = cutlass_torch.matrix(1, M, K, a_major == "m", cutlass.BFloat16)
    b_cpu = cutlass_torch.matrix(1, N, K, b_major == "n", cutlass.BFloat16)
    c_cpu = cutlass_torch.matrix(1, M, N, c_major == "m", c_dtype)

    a_t, a_gpu = cutlass_torch.cute_tensor_like(
        a_cpu, cutlass.BFloat16,
        is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    b_t, b_gpu = cutlass_torch.cute_tensor_like(
        b_cpu, cutlass.BFloat16,
        is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    c_t, c_gpu = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype,
        is_dynamic_layout=is_dynamic_layout, assumed_align=16)

    print(f"  a_cute: shape={a_t.shape}, stride={a_t.stride}")
    print(f"  b_cute: shape={b_t.shape}, stride={b_t.stride}")
    print(f"  c_cute: shape={c_t.shape}, stride={c_t.stride}")

    gemm = HopperWgmmaGemmPersistentKernel(
        cutlass.Float32, tile, cluster,
        swizzle_size=swizzle, raster_along_m=raster_along_m,
        mma_inst_tile_k=mma_inst_tile_k,
    )
    hw = cutlass.utils.HardwareInfo()
    max_active = hw.get_max_active_clusters(cluster[0] * cluster[1])
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled = cute.compile(gemm, a_t, b_t, c_t, max_active, stream)

    compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()

    a_2d = a_gpu.squeeze(-1)
    b_2d = b_gpu.squeeze(-1)
    c_2d = c_gpu.squeeze(-1)
    ref = a_2d.float() @ b_2d.float().t()
    if c_dtype == cutlass.BFloat16:
        ref = ref.bfloat16()
    max_diff = (c_2d - ref).abs().max().item()
    print(f"  correctness: max_diff={max_diff}")

    out_dir = os.path.join(export_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    compiled.export_to_c(out_dir, name)

    for f in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"  exported: {f} ({sz} bytes)")

    h_path = os.path.join(out_dir, f"{name}.h")
    with open(h_path) as f:
        print(f"\n--- {name}.h ---")
        print(f.read())

    for _ in range(10):
        compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()

    import time
    torch.cuda.synchronize()
    t0 = time.time()
    N_iter = 200
    for _ in range(N_iter):
        compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()
    t1 = time.time()
    ms = (t1 - t0) / N_iter * 1000
    tflops = 2 * M * N * K / (ms / 1000) / 1e12
    print(f"  benchmark: {ms:.3f} ms/call, {tflops:.1f} TFLOPS")

    return compiled


def main():
    _init_build_env(dict(os.environ))
    # See gemm_fc1/export_kernels.py for the rationale on
    # ``CUTEDSL_CACHE_ROOT`` + the rmtree-on-symlink-safe purge.
    export_dir = os.path.join(
        _get_build_env().cutedsl_cache_root,
        "cutedsl_export_gemm_fc2",
    )
    os.makedirs(export_dir, exist_ok=True)
    for child in os.listdir(export_dir):
        p = os.path.join(export_dir, child)
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
        else:
            shutil.rmtree(p, ignore_errors=True)

    # v12 production: tile=(128,256), cluster=(2,1), sw=1, raster=N for all 3

    # fwd: Y[M,N] = X[M,K] @ W^T  where X=[40960,4096], W=[1024,4096]
    # a_major=k (row-major), b_major=k (row-major), c_major=n (row-major)
    compile_and_export("gemm_fwd",
        M=40960, N=1024, K=4096,
        a_major="k", b_major="k", c_major="n",
        c_dtype=cutlass.BFloat16,
        tile=(128, 256), cluster=(2, 1), swizzle=1,
        raster_along_m=False,
        is_dynamic_layout=False,
        export_dir=export_dir)

    # dgrad: dX[M,N] = dY[M,K] @ W  where dY=[40960,1024], W=[1024,4096]
    # B = W^T viewed as [N=4096, K=1024] col-major → b_major=n
    compile_and_export("gemm_dgrad",
        M=40960, N=4096, K=1024,
        a_major="k", b_major="n", c_major="n",
        c_dtype=cutlass.BFloat16,
        tile=(128, 256), cluster=(2, 1), swizzle=1,
        raster_along_m=False,
        is_dynamic_layout=False,
        export_dir=export_dir)

    # wgrad: dW[M,N] = dY^T[M,K] @ X^T[N,K]
    # A = dY^T: [M=1024, K=40960] col-major (a_major=m)
    # B = X^T:  [N=4096, K=40960] col-major (b_major=n)
    # C = dW:   [M=1024, N=4096] row-major FP32 (c_major=n)
    # Dynamic layout for col-major views with non-trivial strides
    compile_and_export("gemm_wgrad",
        M=1024, N=4096, K=40960,
        a_major="m", b_major="n", c_major="n",
        c_dtype=cutlass.Float32,
        tile=(128, 256), cluster=(2, 1), swizzle=1,
        raster_along_m=False,
        is_dynamic_layout=True,
        export_dir=export_dir)

    print(f"\n=== All exports complete ===")
    for root, dirs, files in os.walk(export_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            print(f"  {os.path.relpath(fp, export_dir)} ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
