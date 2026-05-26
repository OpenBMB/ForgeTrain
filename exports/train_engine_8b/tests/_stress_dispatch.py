"""Per-operator dispatch for the bare-metal stress runner.

The runner (``_stress_runner.run_stress``) is op-agnostic: it owns the
worker pool, the watchdog, the in-loop NaN guard, and the post-loop
oracle comparison.  Everything that depends on the candidate op's ABI
lives here, packaged into a small dataclass returned by
:func:`build_dispatch`.

Supported ops (all on MiniCPM4-8B / TP=2 / MBS=2 / SEQ=4096):

    gemm_fc1          SwiGLU column-parallel GEMM
                        S=4096, B=2, I=4096, O=16384
    gemm_output       LM-head column-parallel GEMM
                        S=4096, B=2, I=4096, O=36724
    attention_fwd     SM90a flash-attention forward
                        B=2, H_q=16, H_kv=1, N=4096, D=128

GEMM ABIs (gemm_fc1, gemm_output) are identical:

    op_fwd(x:BF16[S,B,I], w:BF16[O,I]) -> BF16[S,B,O]
    op_dgrad(d_y:BF16[T,O], w:BF16[O,I], *, out:BF16[T,I]) -> BF16[T,I]
    op_wgrad(d_y:BF16[S,B,O], x:BF16[S,B,I], *, out_buf:FP32[O,I]) -> FP32

Attention shape mirrors the engine's production attn-fwd call site
(``training_engine_tensor.config.NUM_KV_HEADS=2`` and TP=2 → H_kv=1
per rank; H_q=32/TP=16).

Cross-op invariant: every dispatch returns the same :class:`OpDispatch`
dataclass so the runner only has to call ``candidate_call``,
``oracle_call``, ``get_candidate_out``, and ``make_w6_call``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


__all__ = [
    "OpDispatch",
    "SUPPORTED_OPS",
    "build_dispatch",
]


SUPPORTED_OPS: tuple[str, ...] = ("gemm_fc1", "gemm_output", "attention_fwd")


# ──────────────────────────────────────────────────────────────────────────
# Shape specs (MiniCPM4-8B, TP=2, MBS=2, SEQ=4096)
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _GemmShape:
    op_name: str
    S: int
    B: int
    I: int
    O: int

    @property
    def T(self) -> int:
        return self.S * self.B


_GEMM_SHAPES: dict[str, _GemmShape] = {
    # SwiGLU column-parallel; O = 2 * ffn_hidden_size / TP = 2*16384/2.
    "gemm_fc1": _GemmShape("gemm_fc1", S=4096, B=2, I=4096, O=16384),
    # LM-head column-parallel; O = padded_vocab_size / TP = 73448 / 2.
    "gemm_output": _GemmShape("gemm_output", S=4096, B=2, I=4096, O=36724),
}

# Attention shape (per rank, TP=2): H_q = 32/2, H_kv = 2/2 = 1 (GQA).
_ATTN_SHAPE = dict(B=2, H_q=16, H_kv=1, N=4096, D=128)


# ──────────────────────────────────────────────────────────────────────────
# Dispatch result type
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class OpDispatch:
    """Everything :mod:`_stress_runner` needs that's op-specific."""

    op_name: str
    mode: str
    candidate_call: Callable[[], None]
    oracle_call: Callable[[], torch.Tensor]
    get_candidate_out: Callable[[], torch.Tensor]
    make_w6_call: Callable[[], Callable[[], None]]
    shape_info: dict = field(default_factory=dict)


def build_dispatch(op_name: str, kernel: str, mode: str,
                   device: str = "cuda:0") -> OpDispatch:
    """Build a dispatch for ``op_name``.

    ``kernel``:
      * ``"active"``  — production self-developed CuTeDSL kernel.
      * ``"cublas"``  — ``torch.matmul`` reference (GEMM ops only).
      * ``"sdpa"``    — ``F.scaled_dot_product_attention`` reference
                        (``attention_fwd`` only).

    ``mode`` (GEMM ops):
      * ``"fwd"``     — single forward pass per iter.
      * ``"dgrad"``   — single dgrad per iter.
      * ``"wgrad"``   — single wgrad per iter (RMW; W6 must be disabled).
      * ``"all"``     — production cadence: fwd → dgrad → wgrad per iter.

    ``mode`` (attention_fwd): only ``"fwd"`` is supported.
    """
    if op_name in ("gemm_fc1", "gemm_output"):
        return _build_gemm_dispatch(op_name, kernel, mode, device)
    if op_name == "attention_fwd":
        if mode != "fwd":
            raise ValueError(
                f"attention_fwd only supports mode='fwd' (got {mode!r}); "
                "the engine's flash_attn_dsl ships forward only."
            )
        return _build_attention_fwd_dispatch(kernel, device)
    raise ValueError(f"unknown op {op_name!r}; supported: {SUPPORTED_OPS}")


# ──────────────────────────────────────────────────────────────────────────
# GEMM dispatch (gemm_fc1, gemm_output)
# ──────────────────────────────────────────────────────────────────────────
def _gemm_load_active(op_name: str):
    """Resolve (fwd, dgrad, wgrad) for the active CuTeDSL kernel."""
    if op_name == "gemm_fc1":
        from training_engine_tensor.ops.gemm_fc1.kernel import (
            gemm_fc1_fwd, gemm_fc1_dgrad, gemm_fc1_wgrad,
        )
        return gemm_fc1_fwd, gemm_fc1_dgrad, gemm_fc1_wgrad
    if op_name == "gemm_output":
        from training_engine_tensor.ops.gemm_output.kernel import (
            gemm_output_fwd, gemm_output_dgrad, gemm_output_wgrad,
        )
        return gemm_output_fwd, gemm_output_dgrad, gemm_output_wgrad
    raise ValueError(f"unknown GEMM op {op_name!r}")


def _gemm_make_cublas_triplet():
    """``torch.matmul`` reference for the (fwd, dgrad, wgrad) ABI."""

    def fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        S, B, I = x.shape
        return torch.matmul(x.view(S * B, I), w.t()).view(S, B, w.shape[0])

    def dgrad(d_y: torch.Tensor, w: torch.Tensor, *,
              out: torch.Tensor) -> torch.Tensor:
        torch.matmul(d_y, w, out=out)
        return out

    def wgrad(d_y: torch.Tensor, x: torch.Tensor, *,
              out_buf: torch.Tensor) -> torch.Tensor:
        S, B, O = d_y.shape
        I = x.shape[-1]
        out_buf.add_(torch.matmul(
            d_y.view(S * B, O).t().float(),
            x.view(S * B, I).float(),
        ))
        return out_buf

    return fwd, dgrad, wgrad


def _build_gemm_dispatch(op_name: str, kernel: str, mode: str,
                         device: str) -> OpDispatch:
    shape = _GEMM_SHAPES[op_name]
    S, B, I, O, T = shape.S, shape.B, shape.I, shape.O, shape.T

    if kernel == "active":
        fwd, dgrad, wgrad = _gemm_load_active(op_name)
    elif kernel == "cublas":
        fwd, dgrad, wgrad = _gemm_make_cublas_triplet()
    else:
        raise ValueError(
            f"GEMM kernel must be 'active' or 'cublas'; got {kernel!r}"
        )

    x = torch.randn(S, B, I, dtype=torch.bfloat16, device=device)
    w = torch.randn(O, I, dtype=torch.bfloat16, device=device)
    d_y = torch.randn(T, O, dtype=torch.bfloat16, device=device)
    d_y_3d = d_y.view(S, B, O)
    dx_buf = torch.empty(T, I, dtype=torch.bfloat16, device=device)
    out_fwd = torch.empty(S, B, O, dtype=torch.bfloat16, device=device)
    wgrad_buf = torch.zeros(O, I, dtype=torch.float32, device=device)

    if mode == "fwd":
        def candidate_call() -> None:
            y = fwd(x, w)
            out_fwd.copy_(y)

        def oracle_call() -> torch.Tensor:
            return torch.matmul(x.view(T, I), w.t()).view(S, B, O)

        def get_candidate_out() -> torch.Tensor:
            return out_fwd

    elif mode == "dgrad":
        def candidate_call() -> None:
            dgrad(d_y, w, out=dx_buf)

        def oracle_call() -> torch.Tensor:
            return torch.matmul(d_y, w)

        def get_candidate_out() -> torch.Tensor:
            return dx_buf

    elif mode == "wgrad":
        def candidate_call() -> None:
            wgrad_buf.zero_()
            wgrad(d_y_3d, x, out_buf=wgrad_buf)

        def oracle_call() -> torch.Tensor:
            return torch.matmul(
                d_y_3d.view(T, O).t().float(),
                x.view(T, I).float(),
            )

        def get_candidate_out() -> torch.Tensor:
            return wgrad_buf

    elif mode == "all":
        # Production cadence: fwd → dgrad → wgrad per iter, mirroring one
        # micro-batch of an LM training step.  Real training crashes hit
        # during long runs of this exact pattern, so exercising all three
        # directions on the candidate stream is the closest reproduction
        # of the crash conditions.  The oracle compares the cheapest
        # output (dgrad); the in-loop NaN guard catches divergence on
        # dx_buf which is written last every iter.
        def candidate_call() -> None:
            y = fwd(x, w)
            out_fwd.copy_(y)
            dgrad(d_y, w, out=dx_buf)
            wgrad_buf.zero_()
            wgrad(d_y_3d, x, out_buf=wgrad_buf)

        def oracle_call() -> torch.Tensor:
            return torch.matmul(d_y, w)

        def get_candidate_out() -> torch.Tensor:
            return dx_buf

    else:
        raise ValueError(
            f"GEMM mode must be 'fwd' | 'dgrad' | 'wgrad' | 'all'; "
            f"got {mode!r}"
        )

    def make_w6_call() -> Callable[[], None]:
        # W6 graph-replay needs PRIVATE buffers so the captured graph
        # never aliases the main thread's tensors.  wgrad is RMW so
        # graph-replay is unsafe — callers must drop W6 in that mode.
        if mode == "wgrad":
            def call() -> None:  # noqa: D401
                pass
            return call
        if mode in ("fwd", "all"):
            _x = torch.randn(S, B, I, dtype=torch.bfloat16, device=device)
            _w = torch.randn(O, I, dtype=torch.bfloat16, device=device)
            _y = torch.empty(S, B, O, dtype=torch.bfloat16, device=device)

            def call() -> None:
                y = fwd(_x, _w)
                _y.copy_(y)
            return call
        # dgrad
        _dy = torch.randn(T, O, dtype=torch.bfloat16, device=device)
        _w = torch.randn(O, I, dtype=torch.bfloat16, device=device)
        _dx = torch.empty(T, I, dtype=torch.bfloat16, device=device)

        def call() -> None:
            dgrad(_dy, _w, out=_dx)
        return call

    return OpDispatch(
        op_name=op_name, mode=mode,
        candidate_call=candidate_call, oracle_call=oracle_call,
        get_candidate_out=get_candidate_out, make_w6_call=make_w6_call,
        shape_info={"S": S, "B": B, "I": I, "O": O, "T": T, "kernel": kernel},
    )


# ──────────────────────────────────────────────────────────────────────────
# attention_fwd dispatch
# ──────────────────────────────────────────────────────────────────────────
def _build_attention_fwd_dispatch(kernel: str, device: str) -> OpDispatch:
    B = _ATTN_SHAPE["B"]
    H_q = _ATTN_SHAPE["H_q"]
    H_kv = _ATTN_SHAPE["H_kv"]
    N = _ATTN_SHAPE["N"]
    D = _ATTN_SHAPE["D"]
    softmax_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H_q, N, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device=device)
    out = torch.empty(B, H_q, N, D, dtype=torch.bfloat16, device=device)
    lse = torch.empty(B, H_q, N, dtype=torch.float32, device=device)

    if kernel == "active":
        from flash_attn_dsl.host import run_flash_fwd
        try:
            from flash_attn_dsl.host import prewarm
            prewarm(device=torch.device(device), is_causal_both=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[attn] prewarm failed (will JIT in warmup): {exc}",
                  flush=True)

        def candidate_call() -> None:
            run_flash_fwd(q, k, v, out, lse,
                          softmax_scale=softmax_scale, is_causal=True)
    elif kernel == "sdpa":
        def candidate_call() -> None:
            k_exp = k.expand(B, H_q, N, D)
            v_exp = v.expand(B, H_q, N, D)
            y = F.scaled_dot_product_attention(
                q, k_exp, v_exp, is_causal=True, scale=softmax_scale)
            out.copy_(y)
    else:
        raise ValueError(
            f"attention_fwd kernel must be 'active' or 'sdpa'; got {kernel!r}"
        )

    def oracle_call() -> torch.Tensor:
        # Always use SDPA as the oracle — verifies the candidate output
        # stays self-consistent under worker pressure.  GQA expand is
        # zero-copy.
        k_exp = k.expand(B, H_q, N, D)
        v_exp = v.expand(B, H_q, N, D)
        return F.scaled_dot_product_attention(
            q, k_exp, v_exp, is_causal=True, scale=softmax_scale)

    def get_candidate_out() -> torch.Tensor:
        return out

    def make_w6_call() -> Callable[[], None]:
        # PRIVATE buffer set so graph-capture's pinned addrs don't alias
        # the main thread's q/k/v.
        _q = torch.randn(B, H_q, N, D, dtype=torch.bfloat16, device=device)
        _k = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device=device)
        _v = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device=device)
        _o = torch.empty(B, H_q, N, D, dtype=torch.bfloat16, device=device)
        _l = torch.empty(B, H_q, N, dtype=torch.float32, device=device)

        if kernel == "active":
            from flash_attn_dsl.host import run_flash_fwd

            def call() -> None:
                run_flash_fwd(_q, _k, _v, _o, _l,
                              softmax_scale=softmax_scale, is_causal=True)
        else:
            def call() -> None:
                k_exp = _k.expand(B, H_q, N, D)
                v_exp = _v.expand(B, H_q, N, D)
                y = F.scaled_dot_product_attention(
                    _q, k_exp, v_exp, is_causal=True,
                    scale=softmax_scale)
                _o.copy_(y)
        return call

    return OpDispatch(
        op_name="attention_fwd", mode="fwd",
        candidate_call=candidate_call, oracle_call=oracle_call,
        get_candidate_out=get_candidate_out, make_w6_call=make_w6_call,
        shape_info={**_ATTN_SHAPE, "causal": True,
                    "dtype": "bfloat16", "kernel": kernel},
    )
