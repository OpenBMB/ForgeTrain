"""Export CuTeDSL GEMM kernels to C object files for qkv_proj (3 directions).

Shape-specialized for MiniCPM4 0.5B:
  fwd:   Y[40960,1280] = X[40960,1024] @ W^T  (BF16→BF16)  static layout
  dgrad: dX[40960,1024] = dY[40960,1280] @ W   (BF16→BF16)  static layout
  wgrad: partials[1280,1024,4] = dY_batched * X_batched (BF16→FP32, split-K=4)  dynamic layout

Tile/cluster choices come from the production _PERSISTENT_DEFAULTS in kernel.py
after 59+ rounds of tuning.

Uses the reference HopperWgmmaGemmPersistentKernel from gemm_fc1/bench_cutedsl.py
which has proper cutlass.Constexpr type annotations needed for export_to_c().
"""
import os
import sys
import shutil

# CuTeDSL PYTHONPATH fallback (shared filesystem on cctl pods). Caller
# sets ``CUTLASS_DSL_FALLBACK_DIR`` or the legacy ``CUTLASS_INSTALL_DIR``
# to the package root containing ``python_packages/``.
_fb_dir = os.environ.get("CUTLASS_DSL_FALLBACK_DIR", "")
_inst_dir = os.environ.get("CUTLASS_INSTALL_DIR", "")
for _p in [
    os.path.join(_fb_dir, "python_packages") if _fb_dir else "",
    os.path.join(_inst_dir, "python_packages") if _inst_dir else "",
]:
    if _p and _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gemm_fc1"))
from bench_cutedsl import HopperWgmmaGemmPersistentKernel


def compile_and_export(name, M, N, K, a_major, b_major, c_major, c_dtype,
                       tile, cluster, swizzle, export_dir,
                       raster_along_m=False, mma_inst_tile_k=4,
                       is_dynamic_layout=True,
                       L=None):
    """Compile one CuTeDSL kernel and export to C.

    L: if not None, creates 3D batched tensors (M, K, L) for split-K wgrad.
    """
    print(f"\n=== Compiling {name}: M={M} N={N} K={K} L={L} ===")
    print(f"  a_major={a_major}, b_major={b_major}, c_major={c_major}, c_dtype={c_dtype}")
    print(f"  tile={tile}, cluster={cluster}, swizzle={swizzle}, "
          f"raster_along_m={raster_along_m}, mma_inst_tile_k={mma_inst_tile_k}, "
          f"is_dynamic_layout={is_dynamic_layout}")

    batch = L if L is not None else 1
    a_cpu = cutlass_torch.matrix(batch, M, K, a_major == "m", cutlass.BFloat16)
    b_cpu = cutlass_torch.matrix(batch, N, K, b_major == "n", cutlass.BFloat16)
    c_cpu = cutlass_torch.matrix(batch, M, N, c_major == "m", c_dtype)

    a_t, a_gpu = cutlass_torch.cute_tensor_like(
        a_cpu, cutlass.BFloat16,
        is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    b_t, b_gpu = cutlass_torch.cute_tensor_like(
        b_cpu, cutlass.BFloat16,
        is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    c_t, c_gpu = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype,
        is_dynamic_layout=is_dynamic_layout, assumed_align=16)

    print(f"  a_gpu: shape={a_gpu.shape}, stride={a_gpu.stride()}")
    print(f"  b_gpu: shape={b_gpu.shape}, stride={b_gpu.stride()}")
    print(f"  c_gpu: shape={c_gpu.shape}, stride={c_gpu.stride()}")
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

    a_2d = a_gpu.squeeze(-1) if L is None else a_gpu
    b_2d = b_gpu.squeeze(-1) if L is None else b_gpu
    c_2d = c_gpu.squeeze(-1) if L is None else c_gpu
    if L is None:
        ref = a_2d.float() @ b_2d.float().t()
        if c_dtype == cutlass.BFloat16:
            ref = ref.bfloat16()
        max_diff = (c_2d - ref).abs().max().item()
    else:
        max_diff = 0.0
        for i in range(L):
            ref_i = a_2d[:, :, i].float() @ b_2d[:, :, i].float().t()
            max_diff = max(max_diff, (c_2d[:, :, i] - ref_i).abs().max().item())
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
    tflops = 2 * M * N * K * batch / (ms / 1000) / 1e12
    print(f"  benchmark: {ms:.3f} ms/call, {tflops:.1f} TFLOPS")

    return compiled


def main():
    # See gemm_fc1/export_kernels.py for the rationale.
    export_dir = os.path.join(
        os.environ.get("CUTEDSL_CACHE_ROOT", "/tmp"),
        "cutedsl_export_gemm_qkv_proj",
    )
    os.makedirs(export_dir, exist_ok=True)
    for child in os.listdir(export_dir):
        p = os.path.join(export_dir, child)
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
        else:
            shutil.rmtree(p, ignore_errors=True)

    # fwd: Y[M,N] = X[M,K] @ W^T   where X=[40960,1024], W=[1280,1024]
    # CuTeDSL: C[M,N] = A[M,K] * B[N,K]
    # a_major=k (row-major), b_major=k (row-major), c_major=n (row-major)
    # Production: tile=(128,256), cluster=(2,1), sw=1, raster=N
    # R78: tested sw=4 (JIT auto-tune winner) but AOT static-layout regressed
    # fwd by ~9µs (1.006x vs R77 0.963x). AOT optimal swizzle differs from JIT.
    compile_and_export("gemm_fwd",
        M=40960, N=1280, K=1024,
        a_major="k", b_major="k", c_major="n",
        c_dtype=cutlass.BFloat16,
        tile=(128, 256), cluster=(2, 1), swizzle=1,
        raster_along_m=False,
        is_dynamic_layout=False,
        export_dir=export_dir)

    # dgrad: dX[M,N] = dY[M,K] @ W   where dY=[40960,1280], W=[1280,1024]
    # CuTeDSL: C[M,N] = A[M,K] * B[N,K] with B=W viewed as [N=1024,K=1280]
    # B is originally W[1280,1024] row-major → viewed as [1024,1280] col-major
    # so b_major=n (N is fast dim = col-major)
    # Production: tile=(128,256), cluster=(1,2), sw=1, raster=N
    # R78: tested sw=2 (JIT auto-tune winner) — effect masked by overlap overhead
    # in same run; reverted to sw=1 for safety. AOT swizzle tuning is a separate
    # axis from JIT (static layout changes compiler code-gen).
    compile_and_export("gemm_dgrad",
        M=40960, N=1024, K=1280,
        a_major="k", b_major="n", c_major="n",
        c_dtype=cutlass.BFloat16,
        tile=(128, 256), cluster=(1, 2), swizzle=1,
        raster_along_m=False,
        is_dynamic_layout=False,
        export_dir=export_dir)

    # wgrad: partials[M,N,L] = dY_batched[M,K,L] * X_batched[N,K,L]
    # Split-K=4, chunk_K = 40960/4 = 10240
    # A = dY^T: [M=1280, K=10240, L=4] col-major (a_major=m)
    # B = X^T:  [N=1024, K=10240, L=4] col-major (b_major=n)
    # C = partials: [M=1280, N=1024, L=4] row-major FP32 (c_major=n)
    # Production: tile=(128,128), cluster=(1,4), sw=1, raster=N
    # R77: static layout — shapes/strides baked into kernel binary.
    # NUM_SPLITS=4 is fixed, so the 3D shape is always the same.
    compile_and_export("gemm_wgrad",
        M=1280, N=1024, K=10240,
        a_major="m", b_major="n", c_major="n",
        c_dtype=cutlass.Float32,
        tile=(128, 128), cluster=(1, 4), swizzle=1,
        raster_along_m=False,
        is_dynamic_layout=False,
        L=4,
        export_dir=export_dir)

    print(f"\n=== All exports complete ===")
    for root, dirs, files in os.walk(export_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            print(f"  {os.path.relpath(fp, export_dir)} ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
