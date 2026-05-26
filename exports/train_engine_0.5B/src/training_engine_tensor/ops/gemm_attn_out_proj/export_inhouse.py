"""AOT C-export for in-house persistent GEMM kernel — attn_out_proj fwd/dgrad only.

Static layouts, MiniCPM4-0.5B dimensions (M=40960).
wgrad uses the original CuTeDSL operator (better MFU for this shape).
"""
import os
import sys
import shutil
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch

from training_engine_tensor.ops._gemm_inhouse_kernel import M1PersistentGemmKernel, PERSISTENT_CONFIGS


def compile_and_export(name, M, N, K, a_major, b_major, c_major, c_dtype,
                       cfg, export_dir, L=1):
    tile = cfg["tile_mn"]
    cluster = cfg["cluster_mn"]
    swizzle = cfg.get("swizzle", 1)
    raster_m = cfg.get("raster_m", False)

    print(f"\n=== [{name}] M={M} N={N} K={K} L={L} tile={tile} cluster={cluster} sw={swizzle} ===")

    a_cpu = cutlass_torch.matrix(L, M, K, a_major == "m", cutlass.BFloat16)
    b_cpu = cutlass_torch.matrix(L, N, K, b_major == "n", cutlass.BFloat16)
    c_cpu = cutlass_torch.matrix(L, M, N, c_major == "m", c_dtype)

    a_t, a_gpu = cutlass_torch.cute_tensor_like(a_cpu, cutlass.BFloat16, is_dynamic_layout=False, assumed_align=16)
    b_t, b_gpu = cutlass_torch.cute_tensor_like(b_cpu, cutlass.BFloat16, is_dynamic_layout=False, assumed_align=16)
    c_t, c_gpu = cutlass_torch.cute_tensor_like(c_cpu, c_dtype, is_dynamic_layout=False, assumed_align=16)

    gemm = M1PersistentGemmKernel(cutlass.Float32, tile, cluster, swizzle_size=swizzle, raster_along_m=raster_m)
    hw = cutlass.utils.HardwareInfo()
    max_active = hw.get_max_active_clusters(cluster[0] * cluster[1])
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled = cute.compile(gemm, a_t, b_t, c_t, max_active, stream)
    compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()

    if L == 1:
        ref = a_gpu.squeeze(-1).float() @ b_gpu.squeeze(-1).float().t()
        if c_dtype == cutlass.BFloat16:
            ref = ref.bfloat16()
        print(f"  max_diff={(c_gpu.squeeze(-1) - ref).abs().max().item()}")
    else:
        max_diff = 0.0
        for i in range(L):
            ref_i = a_gpu[:, :, i].float() @ b_gpu[:, :, i].float().t()
            max_diff = max(max_diff, (c_gpu[:, :, i] - ref_i).abs().max().item())
        print(f"  max_diff={max_diff} (batched L={L})")

    out_dir = os.path.join(export_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    compiled.export_to_c(out_dir, name)
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f} ({os.path.getsize(os.path.join(out_dir, f))} bytes)")


def get_export_dir():
    return os.path.join(os.environ.get("CUTEDSL_CACHE_ROOT", "/tmp"), "inhouse_aot_aop")


def main():
    export_dir = get_export_dir()
    os.makedirs(export_dir, exist_ok=True)
    for child in os.listdir(export_dir):
        p = os.path.join(export_dir, child)
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
        else:
            shutil.rmtree(p, ignore_errors=True)

    compile_and_export("inhouse_aop_fwd",
        M=40960, N=1024, K=1024, a_major="k", b_major="k", c_major="n",
        c_dtype=cutlass.BFloat16, cfg=PERSISTENT_CONFIGS["aop_fwd"], export_dir=export_dir)

    compile_and_export("inhouse_aop_dgrad",
        M=40960, N=1024, K=1024, a_major="k", b_major="n", c_major="n",
        c_dtype=cutlass.BFloat16, cfg=PERSISTENT_CONFIGS["aop_dgrad"], export_dir=export_dir)

    print(f"\n=== aop inhouse export done (fwd+dgrad only) → {export_dir} ===")


if __name__ == "__main__":
    main()
