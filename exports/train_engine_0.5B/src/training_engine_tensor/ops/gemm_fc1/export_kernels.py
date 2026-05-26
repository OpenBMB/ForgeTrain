"""Export CuTeDSL GEMM kernels to C object files for all 3 directions."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from bench_cutedsl import HopperWgmmaGemmPersistentKernel
import os
import shutil

from training_engine_tensor.op_dispatcher import get_build_env as _get_build_env
from training_engine_tensor.op_dispatcher import init as _init_build_env

def compile_and_export(name, M, N, K, a_major, b_major, c_major, c_dtype,
                       tile, cluster, swizzle, export_dir,
                       raster_along_m=True, mma_inst_tile_k=4,
                       is_dynamic_layout=True):
    """Compile one CuTeDSL kernel and export to C.

    Round 46: `is_dynamic_layout=False` bakes the strides into the kernel as
    compile-time constants. For our fixed shapes this saves ~70µs/call on fwd
    (8% faster) and ~10µs on dgrad (1% faster).  wgrad regresses (+10µs) so it
    stays dynamic.  When static, the exported tensor descriptor is just a void*.
    """
    print(f"\n=== Compiling {name}: M={M} N={N} K={K} ===")
    print(f"  a_major={a_major}, b_major={b_major}, c_major={c_major}, c_dtype={c_dtype}")
    print(f"  tile={tile}, cluster={cluster}, swizzle={swizzle}, "
          f"raster_along_m={raster_along_m}, mma_inst_tile_k={mma_inst_tile_k}, "
          f"is_dynamic_layout={is_dynamic_layout}")
    
    a_cpu = cutlass_torch.matrix(1, M, K, a_major == "m", cutlass.BFloat16)
    b_cpu = cutlass_torch.matrix(1, N, K, b_major == "n", cutlass.BFloat16)
    c_cpu = cutlass_torch.matrix(1, M, N, c_major == "m", c_dtype)

    a_t, a_gpu = cutlass_torch.cute_tensor_like(a_cpu, cutlass.BFloat16, is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    b_t, b_gpu = cutlass_torch.cute_tensor_like(b_cpu, cutlass.BFloat16, is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    c_t, c_gpu = cutlass_torch.cute_tensor_like(c_cpu, c_dtype, is_dynamic_layout=is_dynamic_layout, assumed_align=16)

    print(f"  a_gpu: shape={a_gpu.shape}, stride={a_gpu.stride()}")
    print(f"  b_gpu: shape={b_gpu.shape}, stride={b_gpu.stride()}")
    print(f"  c_gpu: shape={c_gpu.shape}, stride={c_gpu.stride()}")
    print(f"  a_cute: shape={a_t.shape}, stride={a_t.stride}")
    print(f"  b_cute: shape={b_t.shape}, stride={b_t.stride}")
    print(f"  c_cute: shape={c_t.shape}, stride={c_t.stride}")

    gemm = HopperWgmmaGemmPersistentKernel(
        cutlass.Float32, tile, cluster, swizzle_size=swizzle,
        raster_along_m=raster_along_m, mma_inst_tile_k=mma_inst_tile_k,
    )
    hw = cutlass.utils.HardwareInfo()
    mac = hw.get_max_active_clusters(cluster[0] * cluster[1])
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    
    compiled = cute.compile(gemm, a_t, b_t, c_t, mac, stream)
    
    # Verify correctness
    compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()
    
    # Check vs torch reference
    a_2d = a_gpu.squeeze(-1)
    b_2d = b_gpu.squeeze(-1)
    c_2d = c_gpu.squeeze(-1)
    ref = a_2d.float() @ b_2d.float().t()
    if c_dtype == cutlass.BFloat16:
        ref = ref.bfloat16()
    max_diff = (c_2d - ref).abs().max().item()
    print(f"  correctness: max_diff={max_diff}")
    
    # Export
    out_dir = os.path.join(export_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    compiled.export_to_c(out_dir, name)
    
    for f in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"  exported: {f} ({sz} bytes)")
    
    # Print header
    h_path = os.path.join(out_dir, f"{name}.h")
    with open(h_path) as f:
        print(f"\n--- {name}.h ---")
        print(f.read())
    
    # Benchmark
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
    ms = (t1-t0)/N_iter*1000
    tflops = 2*M*N*K / (ms/1000) / 1e12
    print(f"  benchmark: {ms:.3f} ms/call, {tflops:.0f} TFLOPS")
    
    return compiled

def main():
    _init_build_env(dict(os.environ))
    # Honor ``CUTEDSL_CACHE_ROOT`` (matches the loader's resolver in
    # ``kernel.py::EXPORT_DIR``) so a single env var routes both the
    # export and load to the persistent shared FS path. Default ``/tmp``
    # keeps local-dev behaviour unchanged.
    export_dir = os.path.join(
        _get_build_env().cutedsl_cache_root,
        "cutedsl_export_gemm_fc1",
    )
    os.makedirs(export_dir, exist_ok=True)
    # Use ``os.unlink`` for symlinks + ``shutil.rmtree`` for plain dirs;
    # Python 3.12's rmtree refuses to recurse into symlinked dirs.
    for child in os.listdir(export_dir):
        p = os.path.join(export_dir, child)
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
        else:
            shutil.rmtree(p, ignore_errors=True)
    
    # fwd: Y[M,N] = X[M,K] @ W^T   where W is [N,K]
    # In CuTeDSL convention: C[M,N] = A[M,K] * B[N,K]
    # a_major=k (K is fast dim → row-major), b_major=k, c_major=n
    # Round 46 — KEY CHANGE: is_dynamic_layout=False (static strides baked into
    # kernel as compile-time constants).  probe_static_layout.py on devspace
    # 293058 GPU 2:
    #   dynamic + sw=4 : 0.8982 ms (former production)
    #   static  + sw=4 : 0.8294 ms  → −69µs (7.7% faster)
    #   static  + sw=8 : 0.8269 ms  → −71µs vs dynamic+sw=4, 25µs vs static+sw=4
    # We pick sw=8 + static.  cuBLAS reference for this shape ≈0.92ms event_med,
    # so v1 should now beat cuBLAS by ~90µs/call (about 10%).
    compile_and_export("gemm_fwd",
        M=40960, N=8192, K=1024,
        a_major="k", b_major="k", c_major="n",
        c_dtype=cutlass.BFloat16,
        tile=(128, 256), cluster=(2, 1), swizzle=8,  # Reverted: auto-tuned sw=4 was -2.0% vs cuBLAS; keep R46 sw=8
        is_dynamic_layout=False,
        export_dir=export_dir)
    
    # dgrad: dX[M,N] = dY[M,K] @ W   where W is [K_orig, N_orig] but we pass W[N,K]
    # In BLAS terms: dX = dY @ W^T^T = dY @ W
    # dY is [M=40960, K=8192], W is originally [8192, 1024] row-major
    # In CuTeDSL: C[M,N] = A[M,K] * B[N,K]  where A=dY, B=W treated as [N=1024, K=8192]
    # a_major=k, b_major=n (N is fast dim for W[1024,8192] because original W is [8192,1024] row-major)
    # Round 46 — is_dynamic_layout=False:
    #   dynamic: 0.7720 ms
    #   static : 0.7630 ms → −9µs (1.2% faster), no regression
    compile_and_export("gemm_dgrad",
        M=40960, N=1024, K=8192,
        a_major="k", b_major="n", c_major="n",
        c_dtype=cutlass.BFloat16,
        tile=(128, 256), cluster=(1, 2), swizzle=2,  # R-autotune: c(1,2) sw2 > c(2,1) sw4
        is_dynamic_layout=False,
        export_dir=export_dir)
    
    # wgrad: dW[M,N] = dY^T @ X
    # dY is [40960, 8192] row-major, X is [40960, 1024] row-major
    # In CuTeDSL: C[M,N] = A[M,K] * B[N,K]
    # A = dY^T: [M=8192, K=40960] with a_major="m" (M contiguous = col-major)
    # B = X^T: [N=1024, K=40960] with b_major="n" (N contiguous = col-major)
    # c_dtype = Float32
    # Round 47 — re-baselined on devspace 293801 GPU 2 with sweep_round47.py wgrad:
    #   cluster=(2,1) sw=4 : 1.9645 ms bwd (1.0534× cuBLAS+TE)  ← Round 46 default
    #   cluster=(1,1) sw=4 : 1.8630 ms bwd (0.9989× cuBLAS+TE)  ← new best
    # Reverting cluster (2,1) → (1,1) recovers ~100µs/call, putting us at parity
    # with cuBLAS+TE on bwd (vs 5% slower in Round 46 default on this machine).
    # Round 46's "5µs gain from cluster=(2,1)" was hardware-specific (293058);
    # on 293801 the multicast overhead dominates the 8192-only M dim that already
    # fits 64 tiles×132 SMs without help.
    # Round 49 — DO NOT use mma_inst_tile_k=2.  sweep_round49 reported mma_k=2
    # would beat mma_k=4 by ~70µs/bwd, but that was a min-of-N selection bias
    # (deeper pipeline has higher variance, lowest run skews low).  Re-measured
    # on bench_all_gemm.py (median over 200 iters, 5 fresh-process runs):
    #   mma_k=4 (Round 48): v1/bl total = 1.020-1.025× (median) ← keep
    #   mma_k=2 (Round 49 trial): v1/bl total = 1.040-1.050× (median) — regression
    # The `min` bench in sweep_round49 sampled the right-tail of mma_k=2 but
    # the typical case is worse, hurting the steady-state median that drives
    # actual training step time.
    compile_and_export("gemm_wgrad",
        M=8192, N=1024, K=40960,
        a_major="m", b_major="n", c_major="n",
        c_dtype=cutlass.Float32,
        tile=(128, 256), cluster=(1, 2), swizzle=2,  # R-autotune: c(1,2) sw2 > c(1,1) sw4
        is_dynamic_layout=True,
        export_dir=export_dir)
    
    print(f"\n=== All exports complete ===")
    for root, dirs, files in os.walk(export_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            print(f"  {os.path.relpath(fp, export_dir)} ({os.path.getsize(fp)} bytes)")

if __name__ == "__main__":
    main()
