"""Benchmark: attention backward — FA4 CuTe DSL (v1) vs Transformer Engine baseline.

Usage (on H100):
    python scripts/bench_attention_bwd.py [--warmup 10] [--iters 50]

Measures GPU kernel time via CUDA events and computes MFU against H100 BF16 TC peak.
"""

import argparse
import math
import os
import sys
import time
import traceback

import torch

# ── Shapes (MiniCPM4 0.5B production) ───────────────────────────────
B = 10          # micro_batch_size
N = 4096        # seq_length
HQ = 16         # num_attention_heads
HKV = 2         # num_query_groups (GQA 8:1)
D = 64          # head_dim
CAUSAL = True
DTYPE = torch.bfloat16
SCALE = 1.0 / math.sqrt(D)

H100_PEAK_TFLOPS = 989.4  # BF16 Tensor Core peak

# FlashAttention backward has 5 matmuls per Q-head (recompute QK^T, dV, dP, dQ, dK).
# Each matmul: 2*N*N*D FLOPs (non-causal) per batch per head.
# Causal ≈ half.
CAUSAL_FACTOR = 0.5
BWD_FLOPS = 5 * 2 * B * HQ * N * N * D * CAUSAL_FACTOR


def _cuda_event_bench(fn, warmup, iters):
    """Return median GPU kernel time in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


# ── FA4 CuTe DSL backward ──────────────────────────────────────────

def bench_fa4_bwd(warmup, iters):
    from training_engine_tensor.ops.attention.fa4_cute.interface_bwd_spec import (
        _flash_attn_bwd_sm90_dense,
    )

    q = torch.randn(B, N, HQ, D, dtype=DTYPE, device="cuda")
    k = torch.randn(B, N, HKV, D, dtype=DTYPE, device="cuda")
    v = torch.randn(B, N, HKV, D, dtype=DTYPE, device="cuda")
    out = torch.randn(B, N, HQ, D, dtype=DTYPE, device="cuda")
    d_out = torch.randn(B, N, HQ, D, dtype=DTYPE, device="cuda")
    lse = torch.randn(B, HQ, N, dtype=torch.float32, device="cuda")

    def run():
        _flash_attn_bwd_sm90_dense(
            q, k, v, out, d_out, lse,
            softmax_scale=SCALE, causal=CAUSAL,
        )

    print("  [FA4 CuTe DSL] JIT compiling ... ", end="", flush=True)
    t0 = time.time()
    run()
    torch.cuda.synchronize()
    print(f"done ({time.time() - t0:.1f}s)")

    ms = _cuda_event_bench(run, warmup, iters)
    return ms


# ── Transformer Engine fused_attn backward ──────────────────────────

def bench_te_bwd(warmup, iters):
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        fused_attn_fwd,
        fused_attn_bwd,
    )

    q = torch.randn(N, B, HQ, D, dtype=DTYPE, device="cuda").contiguous()
    k = torch.randn(N, B, HKV, D, dtype=DTYPE, device="cuda").contiguous()
    v = torch.randn(N, B, HKV, D, dtype=DTYPE, device="cuda").contiguous()

    cu_seqlens = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device="cuda")
    backend = tex.get_fused_attn_backend(
        tex.DType.kBFloat16, tex.DType.kBFloat16,
        tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD,
        tex.NVTE_Bias_Type.NVTE_NO_BIAS,
        tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK,
        0.0, HQ, HKV, N, N, D, D, -1, -1,
    )
    print(f"  [TE] backend = {backend}")

    out, aux_tensors = fused_attn_fwd(
        is_training=True, max_seqlen_q=N, max_seqlen_kv=N,
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        q=q, k=k, v=v,
        fake_dtype=torch.bfloat16, fused_attention_backend=backend,
        attn_scale=SCALE, dropout=0.0,
        qkv_layout="sbhd_sbhd_sbhd",
        attn_bias_type="no_bias", attn_mask_type="causal",
    )
    torch.cuda.synchronize()

    d_out = torch.randn_like(out)
    aux_list = [t for t in aux_tensors if t is not None]

    def run():
        fused_attn_bwd(
            max_seqlen_q=N, max_seqlen_kv=N,
            cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
            q=q, k=k, v=v, o=out, d_o=d_out,
            fake_dtype=torch.bfloat16,
            dqkv_dtype=tex.DType.kBFloat16,
            aux_ctx_tensors=aux_list,
            fused_attention_backend=backend,
            attn_scale=SCALE, dropout=0.0,
            qkv_layout="sbhd_sbhd_sbhd",
            attn_bias_type="no_bias", attn_mask_type="causal",
        )

    ms = _cuda_event_bench(run, warmup, iters)
    return ms


# ── flash_attn pip package backward (fallback baseline) ─────────────

def bench_flash_attn_bwd(warmup, iters):
    from flash_attn import flash_attn_func

    q = torch.randn(B, N, HQ, D, dtype=DTYPE, device="cuda", requires_grad=True)
    k = torch.randn(B, N, HKV, D, dtype=DTYPE, device="cuda", requires_grad=True)
    v = torch.randn(B, N, HKV, D, dtype=DTYPE, device="cuda", requires_grad=True)

    out = flash_attn_func(q, k, v, causal=CAUSAL, softmax_scale=SCALE)
    d_out = torch.randn_like(out)
    out.backward(d_out, retain_graph=True)
    q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    def run():
        out2 = flash_attn_func(q, k, v, causal=CAUSAL, softmax_scale=SCALE)
        out2.backward(d_out, retain_graph=False)
        q.grad = k.grad = v.grad = None

    ms_fwd_bwd = _cuda_event_bench(run, warmup, iters)

    # Measure fwd only to subtract
    q2 = q.detach().requires_grad_(False)
    k2 = k.detach().requires_grad_(False)
    v2 = v.detach().requires_grad_(False)

    def run_fwd():
        flash_attn_func(q2, k2, v2, causal=CAUSAL, softmax_scale=SCALE)

    ms_fwd = _cuda_event_bench(run_fwd, warmup, iters)

    ms_bwd = ms_fwd_bwd - ms_fwd
    return ms_bwd, ms_fwd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    dev = torch.cuda.get_device_properties(0)
    print(f"Device: {dev.name}  (SM {dev.major}{dev.minor})")
    print(f"Shape:  B={B}, N={N}, Hq={HQ}, Hkv={HKV}, D={D}, causal={CAUSAL}")
    print(f"BWD FLOPs (causal): {BWD_FLOPS / 1e12:.4f} TFLOP")
    print(f"H100 BF16 TC peak:  {H100_PEAK_TFLOPS} TFLOPS")
    print(f"Warmup={args.warmup}, Iters={args.iters}")
    print()

    results = {}

    # ── TE baseline ─────────────────────────────────────────────────
    print("[1/3] Transformer Engine (baseline) attention bwd")
    try:
        ms_te = bench_te_bwd(args.warmup, args.iters)
        tflops_te = BWD_FLOPS / (ms_te / 1000) / 1e12
        mfu_te = tflops_te / H100_PEAK_TFLOPS * 100
        results["TE_fused"] = (ms_te, tflops_te, mfu_te)
        print(f"  median = {ms_te:.3f} ms  |  {tflops_te:.1f} TFLOPS  |  MFU = {mfu_te:.2f}%")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    print()

    # ── flash_attn pip package (FA2) ────────────────────────────────
    print("[2/3] flash_attn 2.x (pip) attention bwd")
    try:
        ms_fa2, ms_fa2_fwd = bench_flash_attn_bwd(args.warmup, args.iters)
        tflops_fa2 = BWD_FLOPS / (ms_fa2 / 1000) / 1e12
        mfu_fa2 = tflops_fa2 / H100_PEAK_TFLOPS * 100
        results["flash_attn"] = (ms_fa2, tflops_fa2, mfu_fa2)
        print(f"  fwd = {ms_fa2_fwd:.3f} ms  |  bwd = {ms_fa2:.3f} ms")
        print(f"  median bwd = {ms_fa2:.3f} ms  |  {tflops_fa2:.1f} TFLOPS  |  MFU = {mfu_fa2:.2f}%")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    print()

    # ── FA4 CuTe DSL (v1) ──────────────────────────────────────────
    print("[3/3] FA4 CuTe DSL (v1) attention bwd")
    try:
        ms_fa4 = bench_fa4_bwd(args.warmup, args.iters)
        tflops_fa4 = BWD_FLOPS / (ms_fa4 / 1000) / 1e12
        mfu_fa4 = tflops_fa4 / H100_PEAK_TFLOPS * 100
        results["FA4_DSL"] = (ms_fa4, tflops_fa4, mfu_fa4)
        print(f"  median = {ms_fa4:.3f} ms  |  {tflops_fa4:.1f} TFLOPS  |  MFU = {mfu_fa4:.2f}%")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
    print()

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 64)
    print(f"{'Backend':<20} {'ms':>8} {'TFLOPS':>10} {'MFU':>8}")
    print("-" * 64)
    for name, (ms, tf, mfu) in results.items():
        print(f"{name:<20} {ms:>8.3f} {tf:>10.1f} {mfu:>7.2f}%")
    print("=" * 64)

    baselines = [n for n in ("TE_fused", "flash_attn") if n in results]
    if "FA4_DSL" in results and baselines:
        for bl in baselines:
            speedup = results[bl][0] / results["FA4_DSL"][0]
            print(f"  FA4_DSL vs {bl}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
