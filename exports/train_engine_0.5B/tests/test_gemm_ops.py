"""Single-operator correctness + perf test for all 5 GEMM kernels.

Usage (on a GPU node):
    cd /path/to/train_engine
    PYTHONPATH=src CUSTOM_GEMM=1 \
    OP_GEMM_QKV_PROJ=v1 OP_GEMM_ATTN_OUT_PROJ=v1 \
    OP_GEMM_FC1=v1 OP_GEMM_FC2=v1 OP_GEMM_OUTPUT=v1 \
    python tests/test_gemm_ops.py
"""
import os
import sys
import time

import torch

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + ":" + os.environ.get("PYTHONPATH", "")

os.environ.setdefault("CUSTOM_GEMM", "1")
os.environ.setdefault("OP_GEMM_QKV_PROJ", "v1")
os.environ.setdefault("OP_GEMM_ATTN_OUT_PROJ", "v1")
os.environ.setdefault("OP_GEMM_FC1", "v1")
os.environ.setdefault("OP_GEMM_FC2", "v1")
os.environ.setdefault("OP_GEMM_OUTPUT", "v1")
os.environ.setdefault("OP_ATTENTION", "v1")
os.environ.setdefault("VOCAB_SIZE", "73448")
os.environ.setdefault("FUSE_GEMM_PAD", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from training_engine_tensor import op_dispatcher
op_dispatcher.init(env=dict(os.environ))

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = 5
REPEAT = 20

SHAPES = {
    "gemm_qkv_proj": {
        "fwd": {"S": 80, "B": 512, "K": 1024, "N": 1280},
        "bwd": {"S": 80, "B": 512, "K": 1024, "N": 1280},
    },
    "gemm_attn_out_proj": {
        "fwd": {"S": 80, "B": 512, "K": 1024, "N": 1024},
        "bwd": {"S": 80, "B": 512, "K": 1024, "N": 1024},
    },
    "gemm_fc1": {
        "fwd": {"S": 80, "B": 512, "K": 1024, "N": 8192},
        "bwd": {"S": 80, "B": 512, "K": 1024, "N": 8192},
    },
    "gemm_fc2": {
        "fwd": {"S": 80, "B": 512, "K": 4096, "N": 1024},
        "bwd": {"S": 80, "B": 512, "K": 4096, "N": 1024},
    },
    "gemm_output": {
        "fwd": {"S": 80, "B": 512, "K": 1024, "N": 73449},
        "bwd": {"S": 80, "B": 512, "K": 1024, "N": 73449},
    },
}


def _torch_ref_fwd(x_2d, w):
    return x_2d @ w.T


def _torch_ref_bwd(dy_2d, x_2d, w):
    d_input = dy_2d @ w
    d_weight = dy_2d.float().T @ x_2d.float()
    return d_input, d_weight


def _tflops(M, N, K, elapsed_ms):
    return 2 * M * N * K / (elapsed_ms / 1000) / 1e12


def _bench(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000
    return elapsed


def _max_rel_err(a, b):
    a_f, b_f = a.float(), b.float()
    denom = b_f.abs().clamp(min=1e-6)
    return (a_f - b_f).abs().div(denom).max().item()


def _max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def test_gemm_qkv_proj():
    from training_engine_tensor.ops.gemm_qkv_proj.kernel import (
        gemm_qkv_proj_fwd,
        gemm_qkv_proj_bwd,
    )
    s = SHAPES["gemm_qkv_proj"]
    S, B, K, N = s["fwd"]["S"], s["fwd"]["B"], s["fwd"]["K"], s["fwd"]["N"]
    M = S * B

    x = torch.randn(S, B, K, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    out = gemm_qkv_proj_fwd(x, w)
    ref = _torch_ref_fwd(x.view(-1, K), w).view(S, B, N)
    fwd_err = _max_rel_err(out, ref)

    dy = torch.randn_like(out)
    di, dw = gemm_qkv_proj_bwd(dy, x, w)
    ref_di, ref_dw = _torch_ref_bwd(dy.view(-1, N), x.view(-1, K), w)

    di_err = _max_rel_err(di.view(-1, K), ref_di)
    dw_err = _max_rel_err(dw, ref_dw[:N, :K])

    fwd_ms = _bench(lambda: gemm_qkv_proj_fwd(x, w))
    bwd_ms = _bench(lambda: gemm_qkv_proj_bwd(dy, x, w))

    fwd_tflops = _tflops(M, N, K, fwd_ms)
    bwd_tflops_dgrad = _tflops(M, K, N, bwd_ms)

    print(f"  [gemm_qkv_proj]")
    print(f"    fwd: {fwd_ms:.3f} ms  {fwd_tflops:.1f} TFLOPS  max_rel_err={fwd_err:.2e}")
    print(f"    bwd: {bwd_ms:.3f} ms  di_err={di_err:.2e}  dw_err={dw_err:.2e}")
    return fwd_err < 0.05 and di_err < 0.05


def test_gemm_attn_out_proj():
    from training_engine_tensor.ops.gemm_attn_out_proj.kernel import (
        gemm_attn_out_proj_fwd,
        gemm_attn_out_proj_bwd,
    )
    s = SHAPES["gemm_attn_out_proj"]
    S, B, K, N = s["fwd"]["S"], s["fwd"]["B"], s["fwd"]["K"], s["fwd"]["N"]
    M = S * B

    x = torch.randn(S, B, K, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    out = gemm_attn_out_proj_fwd(x, w)
    ref = _torch_ref_fwd(x.view(-1, K), w).view(S, B, N)
    fwd_err = _max_rel_err(out, ref)

    dy = torch.randn_like(out)
    di, dw = gemm_attn_out_proj_bwd(dy, x, w)
    ref_di, ref_dw = _torch_ref_bwd(dy.view(-1, N), x.view(-1, K), w)

    di_err = _max_rel_err(di.view(-1, K), ref_di)
    dw_err = _max_rel_err(dw, ref_dw[:N, :K])

    fwd_ms = _bench(lambda: gemm_attn_out_proj_fwd(x, w))
    bwd_ms = _bench(lambda: gemm_attn_out_proj_bwd(dy, x, w))

    fwd_tflops = _tflops(M, N, K, fwd_ms)

    print(f"  [gemm_attn_out_proj]")
    print(f"    fwd: {fwd_ms:.3f} ms  {fwd_tflops:.1f} TFLOPS  max_rel_err={fwd_err:.2e}")
    print(f"    bwd: {bwd_ms:.3f} ms  di_err={di_err:.2e}  dw_err={dw_err:.2e}")
    return fwd_err < 0.05 and di_err < 0.05


def test_gemm_fc1():
    from training_engine_tensor.ops.gemm_fc1.kernel import (
        gemm_fc1_fwd,
        gemm_fc1_bwd,
    )
    s = SHAPES["gemm_fc1"]
    S, B, K, N = s["fwd"]["S"], s["fwd"]["B"], s["fwd"]["K"], s["fwd"]["N"]
    M = S * B

    x = torch.randn(S, B, K, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    out = gemm_fc1_fwd(x, w)
    ref = _torch_ref_fwd(x.view(-1, K), w).view(S, B, N)
    fwd_err = _max_rel_err(out, ref)

    dy = torch.randn(S, B, N, dtype=DTYPE, device=DEVICE)
    di, dw = gemm_fc1_bwd(dy, x, w)
    ref_di, ref_dw = _torch_ref_bwd(dy.view(-1, N), x.view(-1, K), w)

    di_err = _max_rel_err(di.view(-1, K), ref_di)
    dw_err = _max_rel_err(dw, ref_dw[:N, :K])

    fwd_ms = _bench(lambda: gemm_fc1_fwd(x, w))
    bwd_ms = _bench(lambda: gemm_fc1_bwd(dy, x, w))

    fwd_tflops = _tflops(M, N, K, fwd_ms)

    print(f"  [gemm_fc1]")
    print(f"    fwd: {fwd_ms:.3f} ms  {fwd_tflops:.1f} TFLOPS  max_rel_err={fwd_err:.2e}")
    print(f"    bwd: {bwd_ms:.3f} ms  di_err={di_err:.2e}  dw_err={dw_err:.2e}")
    return fwd_err < 0.05 and di_err < 0.05


def test_gemm_fc2():
    from training_engine_tensor.ops.gemm_fc2.kernel import (
        gemm_fc2_fwd,
        gemm_fc2_bwd,
    )
    s = SHAPES["gemm_fc2"]
    S, B, K, N = s["fwd"]["S"], s["fwd"]["B"], s["fwd"]["K"], s["fwd"]["N"]
    M = S * B

    x = torch.randn(S, B, K, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    out = gemm_fc2_fwd(x, w)
    ref = _torch_ref_fwd(x.view(-1, K), w).view(S, B, N)
    fwd_err = _max_rel_err(out, ref)

    dy = torch.randn(S, B, N, dtype=DTYPE, device=DEVICE)
    di, dw = gemm_fc2_bwd(dy, x, w)
    ref_di, ref_dw = _torch_ref_bwd(dy.view(-1, N), x.view(-1, K), w)

    di_err = _max_rel_err(di.view(-1, K), ref_di)
    dw_err = _max_rel_err(dw, ref_dw[:N, :K])

    fwd_ms = _bench(lambda: gemm_fc2_fwd(x, w))
    bwd_ms = _bench(lambda: gemm_fc2_bwd(dy, x, w))

    fwd_tflops = _tflops(M, N, K, fwd_ms)

    print(f"  [gemm_fc2]")
    print(f"    fwd: {fwd_ms:.3f} ms  {fwd_tflops:.1f} TFLOPS  max_rel_err={fwd_err:.2e}")
    print(f"    bwd: {bwd_ms:.3f} ms  di_err={di_err:.2e}  dw_err={dw_err:.2e}")
    return fwd_err < 0.05 and di_err < 0.05


def test_gemm_output():
    from training_engine_tensor.ops.gemm_output.kernel import (
        gemm_output_fwd,
        gemm_output_bwd,
    )
    s = SHAPES["gemm_output"]
    S, B, K, N = s["fwd"]["S"], s["fwd"]["B"], s["fwd"]["K"], s["fwd"]["N"]
    M = S * B

    x = torch.randn(S, B, K, dtype=DTYPE, device=DEVICE)
    w = torch.randn(N, K, dtype=DTYPE, device=DEVICE)

    out = gemm_output_fwd(x, w)
    ref = _torch_ref_fwd(x.view(-1, K), w).view(S, B, N)
    fwd_err = _max_rel_err(out, ref)

    from training_engine_tensor.ops.gemm_output.kernel import GEMM_PAD_TO
    padded_n = (N + GEMM_PAD_TO - 1) // GEMM_PAD_TO * GEMM_PAD_TO
    dy = torch.randn(S, B, padded_n, dtype=DTYPE, device=DEVICE)
    di, dw = gemm_output_bwd(dy, x, w)
    w_padded = torch.zeros(padded_n, K, dtype=DTYPE, device=DEVICE)
    w_padded[:N].copy_(w)
    ref_di = dy.view(-1, padded_n).float() @ w_padded.float()
    ref_dw = dy.view(-1, padded_n).float().T @ x.view(-1, K).float()

    di_err = _max_rel_err(di.view(-1, K), ref_di.bfloat16())
    dw_err = _max_rel_err(dw, ref_dw[:N, :K])

    fwd_ms = _bench(lambda: gemm_output_fwd(x, w))
    bwd_ms = _bench(lambda: gemm_output_bwd(dy, x, w))

    fwd_tflops = _tflops(M, padded_n, K, fwd_ms)

    print(f"  [gemm_output]")
    print(f"    fwd: {fwd_ms:.3f} ms  {fwd_tflops:.1f} TFLOPS  max_rel_err={fwd_err:.2e}")
    print(f"    bwd: {bwd_ms:.3f} ms  di_err={di_err:.2e}  dw_err={dw_err:.2e}")
    print(f"    orig_n={N}  padded_n={padded_n}")
    return fwd_err < 0.05 and di_err < 0.05


ALL_TESTS = [
    ("gemm_qkv_proj",      test_gemm_qkv_proj),
    ("gemm_attn_out_proj",  test_gemm_attn_out_proj),
    ("gemm_fc1",            test_gemm_fc1),
    ("gemm_fc2",            test_gemm_fc2),
    ("gemm_output",         test_gemm_output),
]


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Shape: S=80, B=512  (M=40960)")
    print()

    results = {}
    for name, fn in ALL_TESTS:
        print(f"{'='*60}")
        print(f"Testing {name} ...")
        try:
            ok = fn()
            results[name] = "PASS" if ok else "FAIL (precision)"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()
        print()

    print(f"{'='*60}")
    print("Summary:")
    for name, status in results.items():
        tag = "✓" if status == "PASS" else "✗"
        print(f"  {tag} {name}: {status}")
