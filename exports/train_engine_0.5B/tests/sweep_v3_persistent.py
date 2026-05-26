"""Parameter sweep for M1PersistentGemmKernel on SXM H100.

Directly compiles and benchmarks via CuTeDSL JIT (no AOT export).
Tests fc1_fwd, fc2_dgrad (lagging shapes), all wgrad, and baseline shapes.
"""
import os, sys, time
os.environ.setdefault("LD_PRELOAD", "/usr/local/cuda/lib64/libcudart.so.12")

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
GEMM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workload", "src", "gemm"))
for p in [SRC, GEMM]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from gemm_kernel import M1PersistentGemmKernel, PERSISTENT_CONFIGS

DEVICE = "cuda"
BF16 = cutlass.BFloat16
FP32 = cutlass.Float32
WARMUP = 5
REPEAT = 20


def _tflops(M, N, K, ms):
    return 2 * M * N * K / (ms / 1000) / 1e12


def bench_persistent(name, M, N, K, a_major, b_major, c_dtype,
                     tile, cluster, swizzle, raster_m, L=1, **kw):
    a_m_major = (a_major == "m")
    b_m_major = (b_major == "n")
    c_m_major = False

    a_cpu = cutlass_torch.matrix(L, M, K, a_m_major, BF16)
    b_cpu = cutlass_torch.matrix(L, N, K, b_m_major, BF16)
    c_cpu = cutlass_torch.matrix(L, M, N, c_m_major, c_dtype)

    a_t, a_gpu = cutlass_torch.cute_tensor_like(a_cpu, BF16, is_dynamic_layout=False, assumed_align=16)
    b_t, b_gpu = cutlass_torch.cute_tensor_like(b_cpu, BF16, is_dynamic_layout=False, assumed_align=16)
    c_t, c_gpu = cutlass_torch.cute_tensor_like(c_cpu, c_dtype, is_dynamic_layout=False, assumed_align=16)

    occ = kw.get("occupancy", 1)
    kpm = kw.get("k_pipe_mmas", 1)
    gemm = M1PersistentGemmKernel(FP32, tile, cluster, swizzle, raster_m,
                                   occupancy=occ, k_pipe_mmas=kpm)

    hwinfo = cutlass.utils.HardwareInfo()
    mac = hwinfo.get_max_active_clusters(cluster[0] * cluster[1])

    ts = torch.cuda.Stream()
    stream = cuda.CUstream(ts.cuda_stream)

    compiled = cute.compile(gemm, a_t, b_t, c_t, mac, stream)

    compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()

    # correctness check
    if L == 1:
        a_2d = a_gpu.squeeze(-1)
        b_2d = b_gpu.squeeze(-1)
        c_2d = c_gpu.squeeze(-1)
        ref = a_2d.float() @ b_2d.float().t()
        if c_dtype == BF16:
            ref = ref.bfloat16()
        diff = (c_2d - ref).abs().max().item()
    else:
        diff = -1  # skip for batched

    # benchmark
    for _ in range(WARMUP):
        compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / REPEAT * 1000
    tflops = _tflops(M, N, K * L, ms)  # total FLOPs across splits

    return ms, tflops, diff


def run_original(name, M, N, K, a_major, b_major):
    """Run original operator's fwd/dgrad for comparison."""
    S, B = 80, 512
    assert S * B == M
    DTYPE = torch.bfloat16

    if "fc1_fwd" in name:
        os.environ.setdefault("CUSTOM_GEMM", "1")
        os.environ.setdefault("OP_GEMM_FC1", "v1")
        os.environ.setdefault("OP_ATTENTION", "v1")
        os.environ.setdefault("VOCAB_SIZE", "73448")
        os.environ.setdefault("FUSE_GEMM_PAD", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        from training_engine_tensor import op_dispatcher
        if not hasattr(op_dispatcher, '_init_done'):
            op_dispatcher.init(env=dict(os.environ))
            op_dispatcher._init_done = True
        from training_engine_tensor.ops.gemm_fc1.kernel import gemm_fc1_fwd
        x = torch.randn(S, B, K, dtype=DTYPE, device=DEVICE)
        w = torch.randn(N, K, dtype=DTYPE, device=DEVICE)
        for _ in range(WARMUP):
            gemm_fc1_fwd(x, w)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            gemm_fc1_fwd(x, w)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / REPEAT * 1000
        return ms, _tflops(M, N, K, ms)

    if "fc2_dgrad" in name:
        os.environ.setdefault("CUSTOM_GEMM", "1")
        os.environ.setdefault("OP_GEMM_FC2", "v1")
        os.environ.setdefault("OP_ATTENTION", "v1")
        os.environ.setdefault("VOCAB_SIZE", "73448")
        os.environ.setdefault("FUSE_GEMM_PAD", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        from training_engine_tensor import op_dispatcher
        if not hasattr(op_dispatcher, '_init_done'):
            op_dispatcher.init(env=dict(os.environ))
            op_dispatcher._init_done = True
        from training_engine_tensor.ops.gemm_fc2.kernel import gemm_fc2_fwd, gemm_fc2_bwd
        x = torch.randn(S, B, 4096, dtype=DTYPE, device=DEVICE)
        w = torch.randn(1024, 4096, dtype=DTYPE, device=DEVICE)
        dy = torch.randn(S, B, 1024, dtype=DTYPE, device=DEVICE)
        # dgrad only
        from training_engine_tensor.ops.gemm_fc2.kernel import _get_aot_ext
        ext = _get_aot_ext()
        if ext and hasattr(ext, 'gemm_dgrad'):
            for _ in range(WARMUP):
                ext.gemm_dgrad(dy.view(-1, 1024), w, torch.empty(M, 4096, dtype=DTYPE, device=DEVICE))
            torch.cuda.synchronize()
            d_buf = torch.empty(M, 4096, dtype=DTYPE, device=DEVICE)
            t0 = time.perf_counter()
            for _ in range(REPEAT):
                ext.gemm_dgrad(dy.view(-1, 1024), w, d_buf)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / REPEAT * 1000
            return ms, _tflops(M, 4096, 1024, ms)

    return None, None


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Warmup={WARMUP}, Repeat={REPEAT}")
    print()

    # ── Sweep fc1_fwd: M=40960, N=8192, K=1024 ──
    print("=" * 80)
    print("fc1_fwd sweep: M=40960, N=8192, K=1024")
    print(f"{'Config':<45} {'ms':>8} {'TFLOPS':>8} {'diff':>10}")
    print("-" * 80)

    fc1_fwd_configs = [
        {"cluster": (2,1), "swizzle": 8, "raster_m": True,  "tag": "R1 baseline (2,1) sw8 rM"},
        {"cluster": (2,1), "swizzle": 4, "raster_m": True,  "tag": "(2,1) sw4 rM"},
        {"cluster": (2,1), "swizzle": 2, "raster_m": True,  "tag": "(2,1) sw2 rM"},
        {"cluster": (2,1), "swizzle": 8, "raster_m": False, "tag": "(2,1) sw8 rN"},
        {"cluster": (1,2), "swizzle": 8, "raster_m": True,  "tag": "(1,2) sw8 rM"},
        {"cluster": (1,2), "swizzle": 8, "raster_m": False, "tag": "(1,2) sw8 rN"},
        {"cluster": (1,2), "swizzle": 4, "raster_m": False, "tag": "(1,2) sw4 rN"},
        {"cluster": (1,1), "swizzle": 1, "raster_m": True,  "tag": "(1,1) sw1 rM"},
    ]
    for c in fc1_fwd_configs:
        ms, tf, diff = bench_persistent("fc1_fwd", 40960, 8192, 1024, "k", "k", BF16,
                                         (128, 256), c["cluster"], c["swizzle"], c["raster_m"])
        print(f"  {c['tag']:<43} {ms:>8.3f} {tf:>8.1f} {diff:>10.4f}")

    orig_ms, orig_tf = run_original("fc1_fwd", 40960, 8192, 1024, "k", "k")
    if orig_ms:
        print(f"  {'ORIGINAL (CuTeDSL AOT)':<43} {orig_ms:>8.3f} {orig_tf:>8.1f}")

    # ── Sweep fc2_dgrad: M=40960, N=4096, K=1024 ──
    print()
    print("=" * 80)
    print("fc2_dgrad sweep: M=40960, N=4096, K=1024 (B col-major)")
    print(f"{'Config':<45} {'ms':>8} {'TFLOPS':>8} {'diff':>10}")
    print("-" * 80)

    fc2_dgrad_configs = [
        {"cluster": (2,1), "swizzle": 1, "raster_m": False, "tag": "R1 baseline (2,1) sw1 rN"},
        {"cluster": (2,1), "swizzle": 8, "raster_m": False, "tag": "(2,1) sw8 rN"},
        {"cluster": (2,1), "swizzle": 4, "raster_m": False, "tag": "(2,1) sw4 rN"},
        {"cluster": (2,1), "swizzle": 2, "raster_m": False, "tag": "(2,1) sw2 rN"},
        {"cluster": (2,1), "swizzle": 8, "raster_m": True,  "tag": "(2,1) sw8 rM"},
        {"cluster": (1,2), "swizzle": 8, "raster_m": False, "tag": "(1,2) sw8 rN"},
        {"cluster": (1,2), "swizzle": 4, "raster_m": False, "tag": "(1,2) sw4 rN"},
        {"cluster": (1,1), "swizzle": 1, "raster_m": False, "tag": "(1,1) sw1 rN"},
    ]
    for c in fc2_dgrad_configs:
        ms, tf, diff = bench_persistent("fc2_dgrad", 40960, 4096, 1024, "k", "n", BF16,
                                         (128, 256), c["cluster"], c["swizzle"], c["raster_m"])
        print(f"  {c['tag']:<43} {ms:>8.3f} {tf:>8.1f} {diff:>10.4f}")

    orig_ms, orig_tf = run_original("fc2_dgrad", 40960, 4096, 1024, "k", "n")
    if orig_ms:
        print(f"  {'ORIGINAL (CuTeDSL AOT)':<43} {orig_ms:>8.3f} {orig_tf:>8.1f}")

    # ── Wgrad: FP32 output, split-K L=2 ──
    print()
    print("=" * 80)
    print("wgrad benchmark (FP32 output, L=2 split-K)")
    print(f"{'Name':<30} {'ms':>8} {'TFLOPS':>8}")
    print("-" * 80)

    K_total = 40960
    K_chunk = K_total // 2

    wgrad_shapes = [
        ("fc1_wgrad", 8192, 1024, K_chunk, PERSISTENT_CONFIGS["fc1_wgrad"]),
        ("fc2_wgrad", 1024, 4096, K_chunk, PERSISTENT_CONFIGS["fc2_wgrad"]),
        ("aop_wgrad", 1024, 1024, K_chunk, PERSISTENT_CONFIGS["aop_wgrad"]),
    ]
    for name, M, N, K, cfg in wgrad_shapes:
        try:
            ms, tf, _ = bench_persistent(name, M, N, K, "m", "n", FP32,
                                          cfg["tile_mn"], cfg["cluster_mn"],
                                          cfg["swizzle"], cfg["raster_m"], L=2)
            print(f"  {name:<28} {ms:>8.3f} {tf:>8.1f}")
        except Exception as e:
            print(f"  {name:<28} ERROR: {e}")

    print()
    print("Done.")
