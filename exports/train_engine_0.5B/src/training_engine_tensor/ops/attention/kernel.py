"""Attention operator kernel — FA4 CuTe DSL (SM90 Shape Specialization).

Forward : Selectable via ATTN_V1_FWD_BACKEND:
          - "flash_dsl" (default): in-tree CuTe DSL SM90 fwd, GQA-native.
          - "fa4": FA4 DSL SM90 forward spec (pack_gqa).
Backward: Selectable via ATTN_V1_BWD_BACKEND:
          - "flash_dsl" (default): 3-kernel CuTe DSL pipeline (~1.9ms/45.7% MFU on H100).
          - "fa4": FA4 DSL SM90 backward spec (interface_bwd_spec).
          - "manual": PyTorch autograd fallback (debug only).

Layout convention
  Framework (SBHD): q [S, B, Hq, D],  k/v [S, B, Hkv, D]
  FA4      (BHSD):  q [B, S, Hq, D],  k/v [B, S, Hkv, D]
  DSL      (BHND):  q [B, Hq, N, D],  k/v [B, Hkv, N, D]
  Transposes at the boundary: SBHD ↔ BHSD via permute(1,0,2,3),
  BHSD ↔ BHND via transpose(1,2).
"""
from __future__ import annotations

__all__ = ["attention_fwd", "attention_bwd"]

import math
import os
import sys

import torch

from training_engine_tensor.op_dispatcher import get_frozen_env as _get_frozen_env

_OP_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _manual_fwd_bf16(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manual BF16 forward returning (out, lse).

    Uses BF16 matmul (matching Tensor Core precision used in backward)
    with float32 reduction for logsumexp.

    Returns
    -------
    out : [B, S, Hq, D]  (same dtype as input)
    lse : [B, Hq, S]     (float32)
    """
    B, S, Hq, D = q_bhsd.shape
    Hkv = k_bhsd.shape[2]
    g = Hq // Hkv

    k_exp = k_bhsd.repeat_interleave(g, dim=2)
    v_exp = v_bhsd.repeat_interleave(g, dim=2)

    qt = q_bhsd.permute(0, 2, 1, 3)          # [B, Hq, S, D]
    kt = k_exp.permute(0, 2, 1, 3)
    vt = v_exp.permute(0, 2, 1, 3)

    scores = torch.matmul(qt, kt.transpose(-2, -1)) * softmax_scale

    if causal:
        mask = torch.triu(
            torch.ones(S, S, device=q_bhsd.device, dtype=torch.bool), diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    scores_f32 = scores.float()
    lse = torch.logsumexp(scores_f32, dim=-1)  # [B, Hq, S]
    attn = torch.softmax(scores_f32, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    out = torch.matmul(attn, vt.float()).to(q_bhsd.dtype)
    out = out.permute(0, 2, 1, 3).contiguous()  # [B, S, Hq, D]
    return out, lse.contiguous()


# ---------------------------------------------------------------------------
# Backward helpers
# ---------------------------------------------------------------------------

def _manual_bwd_autograd(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    d_out_bhsd: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual backward via PyTorch autograd on the BF16 forward path.

    Re-runs the exact same computation as _manual_fwd_bf16 with grad tracking,
    then calls .backward() to get gradients for q, k, v.

    Returns (dq, dk, dv) in BHSD layout, same dtype as input.
    """
    q = q_bhsd.detach().requires_grad_(True)
    k = k_bhsd.detach().requires_grad_(True)
    v = v_bhsd.detach().requires_grad_(True)

    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    g = Hq // Hkv

    k_exp = k.repeat_interleave(g, dim=2)
    v_exp = v.repeat_interleave(g, dim=2)

    qt = q.permute(0, 2, 1, 3)
    kt = k_exp.permute(0, 2, 1, 3)
    vt = v_exp.permute(0, 2, 1, 3)

    scores = torch.matmul(qt, kt.transpose(-2, -1)) * softmax_scale

    if causal:
        mask = torch.triu(
            torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    scores_f32 = scores.float()
    attn = torch.softmax(scores_f32, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    out = torch.matmul(attn, vt.float()).to(q.dtype)
    out = out.permute(0, 2, 1, 3).contiguous()

    out.backward(d_out_bhsd)

    return q.grad, k.grad, v.grad


_fa4_dsl_fwd_resolved: object = None  # None = not yet probed
_fa4_fwd_resolved: object = None  # None = not yet probed
_fa4_bwd_resolved: object = None  # None = not yet probed
_flash_dsl_fwd_resolved: object = None  # None = not yet probed
_flash_dsl_bwd_resolved: object = None  # None = not yet probed


def _resolve_flash_dsl_fwd():
    """Resolve the in-tree flash_attn_dsl host.run_flash_fwd if available.

    This is the alternative SM90 forward written from scratch in CuTe DSL
    (see workload notes). The kernel is fully GQA-native end to end:
      * the host wrapper accepts K/V at ``[B, H_kv, N, D]`` directly,
      * the kernel resolves the K/V head via ``head_idx // FastDivmod(g)``
        inside the producer warp,
      * num_stages=3 K/V pipeline + dual K/V pipelines + sQ/sO smem reuse
        are kept for throughput,
      * LSE is a plain (non-atomic) bounds-checked store, so zero-init is
        no longer required — we keep ``torch.zeros`` defensively to match
        the historical contract.
    """
    global _flash_dsl_fwd_resolved
    if _flash_dsl_fwd_resolved is not None:
        return _flash_dsl_fwd_resolved if _flash_dsl_fwd_resolved is not False else None
    try:
        from .flash_attn_dsl import host as _flash_host
        _flash_dsl_fwd_resolved = _flash_host
        return _flash_host
    except Exception as e:
        import traceback
        print(f"[kernel.py] flash_attn_dsl host import failed: {e}", flush=True)
        traceback.print_exc()
        _flash_dsl_fwd_resolved = False
        return None


def _resolve_fa4_dsl_fwd():
    """Strict resolver: never fall back. Re-raises the import failure
    so callers crash loudly with a clear root cause instead of silently
    routing through ``flash_attn`` v2 (different numerics, different
    memory profile) or the O(S^2) manual SDPA path.

    Set ``ATTN_V1_ALLOW_NON_FA4_FWD=1`` ONLY for local debugging where
    you accept the substitute kernel; production must keep this strict.
    """
    global _fa4_dsl_fwd_resolved
    if _fa4_dsl_fwd_resolved is not None:
        return _fa4_dsl_fwd_resolved if _fa4_dsl_fwd_resolved is not False else None
    try:
        from .fa4_cute.interface_fwd_spec import _flash_attn_fwd_sm90_spec
        _fa4_dsl_fwd_resolved = _flash_attn_fwd_sm90_spec
        return _flash_attn_fwd_sm90_spec
    except Exception as e:
        if _get_frozen_env().get("ATTN_V1_ALLOW_NON_FA4_FWD", "0") == "1":
            import traceback
            print(f"[kernel.py] FA4 DSL fwd spec not available, allowing "
                  f"flash_attn / manual fallback (DEBUG ONLY): {e}",
                  flush=True)
            traceback.print_exc()
            _fa4_dsl_fwd_resolved = False
            return None
        raise RuntimeError(
            "attention v1: FA4 forward kernel could not be imported "
            "(fa4_cute.interface_fwd_spec._flash_attn_fwd_sm90_spec). "
            "Refusing to fall back to flash_attn v2 / manual SDPA — "
            "different numerics + memory profile would invalidate "
            "loss-alignment tests against the baseline. The most "
            "common root cause is that the entry script did not install "
            "the cutlass DSL wheels (which fa4_cute depends on); make "
            "sure cutlass install runs whenever OP_ATTENTION=v1, not "
            "only when CUSTOM_GEMM=1.\n\nRoot-cause import error: "
            + repr(e)
        ) from e


def _resolve_fa4_fwd():
    """Secondary forward (flash_attn v2). Disabled by default — only
    surfaced when the user explicitly opts in via
    ``ATTN_V1_ALLOW_NON_FA4_FWD=1`` AND the FA4 DSL fwd is also
    unavailable. See _resolve_fa4_dsl_fwd for the rationale."""
    global _fa4_fwd_resolved
    if _fa4_fwd_resolved is not None:
        return _fa4_fwd_resolved if _fa4_fwd_resolved is not False else None
    if _get_frozen_env().get("ATTN_V1_ALLOW_NON_FA4_FWD", "0") != "1":
        _fa4_fwd_resolved = False
        return None
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        _fa4_fwd_resolved = flash_attn_func
        return flash_attn_func
    except Exception:
        _fa4_fwd_resolved = False
        return None


def _resolve_fa4_bwd():
    """Strict resolver: never fall back. Re-raises the import failure so
    callers crash loudly with a clear root cause instead of silently
    routing through the O(S^2) manual SDPA path (which can OOM at
    seq=4096 and produces gradients identical to the FA4 path only
    within numerical noise — but at ~100x memory).

    Set ``ATTN_V1_ALLOW_MANUAL_BWD_FALLBACK=1`` ONLY for local debugging
    where you accept the memory hit; production must keep this strict.
    """
    global _fa4_bwd_resolved
    if _fa4_bwd_resolved is not None:
        return _fa4_bwd_resolved if _fa4_bwd_resolved is not False else None
    try:
        from .fa4_cute.interface_bwd_spec import _flash_attn_bwd_sm90_dense
        _fa4_bwd_resolved = _flash_attn_bwd_sm90_dense
        return _flash_attn_bwd_sm90_dense
    except Exception as e:
        if _get_frozen_env().get("ATTN_V1_ALLOW_MANUAL_BWD_FALLBACK", "0") == "1":
            import traceback
            print(f"[kernel.py] FA4 bwd spec not available, falling back "
                  f"to manual SDPA (DEBUG ONLY): {e}", flush=True)
            traceback.print_exc()
            _fa4_bwd_resolved = False
            return None
        raise RuntimeError(
            "attention v1: FA4 backward kernel could not be imported "
            "(fa4_cute.interface_bwd_spec._flash_attn_bwd_sm90_dense). "
            "Refusing to fall back to the O(S^2) manual SDPA bwd path "
            "(see ops/attention/kernel.py::_manual_bwd_autograd) — at "
            "seq_len=4096 it materializes a (B,H,S,S)=GBs softmax "
            "tensor and OOMs an 80 GB H100 inside the MTP layer's "
            "attention bwd.\n\nRoot-cause import error: " + repr(e)
        ) from e


def _resolve_flash_dsl_bwd():
    """Resolve the in-tree flash_attn_dsl backward (3-kernel CuTe DSL pipeline).

    This is the optimized SM90 backward written from scratch in CuTe DSL,
    achieving ~1.9ms / 45.7% MFU on H100 for the MiniCPM4-0.5B shape. It uses
    a 3-kernel architecture: preprocess (dPsum + lse_log2 + zero dQ_accum),
    main (5-GEMM WGMMA with GQA via cp.reduce.async.bulk.add), and postprocess
    (FP32 accum → FP16 scale).

    Input layout: [B, H, N, D] (BHND). GQA-native: K/V at H_kv heads.
    """
    global _flash_dsl_bwd_resolved
    if _flash_dsl_bwd_resolved is not None:
        return _flash_dsl_bwd_resolved if _flash_dsl_bwd_resolved is not False else None
    try:
        from .flash_attn_dsl.flash_bwd import run_flash_bwd_dsl
        _flash_dsl_bwd_resolved = run_flash_bwd_dsl
        return run_flash_bwd_dsl
    except Exception as e:
        import traceback
        print(f"[kernel.py] flash_attn_dsl bwd import failed: {e}", flush=True)
        traceback.print_exc()
        _flash_dsl_bwd_resolved = False
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    layer_number: int = 1,
    need_backward: bool = False,
) -> tuple[torch.Tensor, object]:
    """Attention forward — drop-in replacement for attention_forward_te.

    q: [S, B, Hq, D]  k: [S, B, Hkv, D]  v: [S, B, Hkv, D]
    Returns: (output [S, B, Hq*D], aux_ctx)
    """
    S, B, Hq, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Forward backend selector — defaults to "flash_dsl" (the in-tree
    # flash_attn_dsl SM90 forward written from scratch in CuTe DSL). It is
    # fully GQA-native at the host boundary: K/V are passed in their
    # natural [B, Hkv, S, D] layout and the producer warp resolves the K/V
    # head via FastDivmodDivisor(g). Set ATTN_V1_FWD_BACKEND=fa4 to fall
    # back to the FA4 DSL spec forward (pack_gqa-based). Backward always
    # stays on the FA4 spec backward (this op only swaps the forward kernel
    # — see attention/notes for the rationale).
    fwd_backend = _get_frozen_env().get("ATTN_V1_FWD_BACKEND", "flash_dsl")

    # DSL forward: skip .contiguous() for inputs; TMA handles strided layout.
    # Pre-allocate output in SBHD so TMA writes directly to target layout.
    # Fallback paths still need contiguous inputs and output copy.

    if fwd_backend == "flash_dsl":
        flash_host = _resolve_flash_dsl_fwd()
        if flash_host is None:
            raise RuntimeError(
                "ATTN_V1_FWD_BACKEND=flash_dsl requested but flash_attn_dsl "
                "could not be imported. See traceback above for the root cause."
            )

        # Layout convention reminder (see file docstring):
        #   q (input)    : SBHD = [S, B, Hq, D]
        #   q_bhsd (FA4) : BSHD = [B, S, Hq, D]  (FA4 bwd / aux_ctx convention)
        #   q_bnhd (DSL) : BHSD = [B, Hq, S, D]  (flash_attn_dsl input convention)
        # All three views share the same physical storage; only strides differ
        # and stride(-1) stays 1 across all permutations, which is what the
        # CuTe TMA descriptors require.
        q_bhsd = q.permute(1, 0, 2, 3)  # [B, S, Hq, D]
        k_bhsd = k.permute(1, 0, 2, 3)  # [B, S, Hkv, D]
        v_bhsd = v.permute(1, 0, 2, 3)  # [B, S, Hkv, D]

        # flash_attn_dsl forward (FlashAttnFwdSm90 in flash_fwd.py) does NOT
        # handle GQA internally — it indexes K/V at head_idx directly. We must
        # expand K/V to H_q heads here so TMA reads are in-bounds.
        Hkv = k.shape[2]
        g = Hq // Hkv
        q_bnhd = q_bhsd.transpose(1, 2)              # [B, Hq, S, D]
        k_bnhd = k_bhsd.transpose(1, 2).contiguous()  # [B, Hkv, S, D]
        v_bnhd = v_bhsd.transpose(1, 2).contiguous()  # [B, Hkv, S, D]
        if g > 1:
            k_bnhd = k_bnhd.repeat_interleave(g, dim=1)  # [B, Hq, S, D]
            v_bnhd = v_bnhd.repeat_interleave(g, dim=1)  # [B, Hq, S, D]

        # LSE in this revision is a plain bounds-checked store (no atomic
        # add), so torch.empty would technically suffice — keep zero-init
        # for defense in depth.
        out_bnhd = torch.empty(B, Hq, S, D, dtype=q.dtype, device=q.device)
        lse = torch.zeros(B, Hq, S, dtype=torch.float32, device=q.device)

        flash_host.run_flash_fwd(
            q_bnhd, k_bnhd, v_bnhd, out_bnhd, lse,
            softmax_scale=scale,
            is_causal=causal,
        )

        # Reshape output: back to BSHD for FA4 bwd compatibility, then to
        # the SBH·D flat layout the caller expects.
        out_bhsd = out_bnhd.transpose(1, 2).contiguous()  # [B, S, Hq, D]
        out_flat = (
            out_bhsd
            .permute(1, 0, 2, 3)             # [S, B, Hq, D]
            .reshape(S, B, Hq * D)
            .contiguous()
        )
    elif (dsl_fwd := _resolve_fa4_dsl_fwd()) is not None:
        q_bhsd = q.permute(1, 0, 2, 3)
        k_bhsd = k.permute(1, 0, 2, 3)
        v_bhsd = v.permute(1, 0, 2, 3)

        out_sbhd = torch.empty(S, B, Hq, D, dtype=q.dtype, device=q.device)
        out_bhsd = out_sbhd.permute(1, 0, 2, 3)
        _, lse = dsl_fwd(
            q_bhsd, k_bhsd, v_bhsd,
            softmax_scale=scale,
            causal=causal,
            out=out_bhsd,
        )
        out_flat = out_sbhd.reshape(S, B, Hq * D)
    else:
        q_bhsd = q.permute(1, 0, 2, 3).contiguous()
        k_bhsd = k.permute(1, 0, 2, 3).contiguous()
        v_bhsd = v.permute(1, 0, 2, 3).contiguous()
        fa4_fwd = _resolve_fa4_fwd()
        if fa4_fwd is not None:
            out_bhsd, lse, _ = fa4_fwd(
                q_bhsd, k_bhsd, v_bhsd,
                softmax_scale=scale,
                causal=causal,
                return_attn_probs=True,
            )
        else:
            out_bhsd, lse = _manual_fwd_bf16(q_bhsd, k_bhsd, v_bhsd, scale, causal)
        out_flat = out_bhsd.permute(1, 0, 2, 3).contiguous().reshape(S, B, Hq * D)

    aux_ctx = None
    if need_backward:
        aux_ctx = (q_bhsd, k_bhsd, v_bhsd, out_bhsd, lse, scale)

    return out_flat, aux_ctx


def attention_bwd(
    d_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    aux_ctx: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention backward.

    Backend selection via ATTN_V1_BWD_BACKEND env var:
      - "flash_dsl" (default): 3-kernel CuTe DSL pipeline (optimized, ~1.9ms on H100)
      - "fa4": FA4 DSL SM90 backward spec (interface_bwd_spec)
      - "manual": PyTorch autograd fallback (debug only)

    Returns: (d_q, d_k, d_v) in original head-split shapes [S, B, H, D].
    """
    if aux_ctx is None:
        raise RuntimeError("aux_ctx is None — forward not called with need_backward=True")

    S, B = d_output.shape[:2]
    Hq = q.shape[2]
    D = q.shape[3]

    q_bhsd, k_bhsd, v_bhsd, out_bhsd, lse, scale = aux_ctx

    Hkv = k.shape[2]

    bwd_backend = _get_frozen_env().get("ATTN_V1_BWD_BACKEND", "flash_dsl")

    if bwd_backend == "flash_dsl":
        flash_dsl_bwd = _resolve_flash_dsl_bwd()
        if flash_dsl_bwd is None:
            raise RuntimeError(
                "ATTN_V1_BWD_BACKEND=flash_dsl requested but flash_attn_dsl "
                "backward could not be imported. See traceback above."
            )

        # BSHD [B,S,H,D] → BHND [B,H,S,D]: strided views only, no memcpy.
        # CuTe TMA handles non-contiguous outer strides via mark_layout_dynamic.
        q_bhnd = q_bhsd.transpose(1, 2)
        k_bhnd = k_bhsd.transpose(1, 2)
        v_bhnd = v_bhsd.transpose(1, 2)
        out_bhnd = out_bhsd.transpose(1, 2)
        d_out_bhnd = d_output.reshape(S, B, Hq, D).permute(1, 2, 0, 3)

        # Allocate outputs in SBHD; kernel writes via strided BHND views.
        dq_sbhd = torch.empty(S, B, Hq, D, dtype=q.dtype, device=q.device)
        dk_sbhd = torch.empty(S, B, Hkv, D, dtype=q.dtype, device=q.device)
        dv_sbhd = torch.empty(S, B, Hkv, D, dtype=q.dtype, device=q.device)

        flash_dsl_bwd(
            q_bhnd, k_bhnd, v_bhnd, out_bhnd, d_out_bhnd, lse,
            dq_sbhd.permute(1, 2, 0, 3),
            dk_sbhd.permute(1, 2, 0, 3),
            dv_sbhd.permute(1, 2, 0, 3),
            softmax_scale=scale,
            is_causal=True,
        )

        return dq_sbhd, dk_sbhd, dv_sbhd

    if bwd_backend == "fa4":
        fa4_bwd = _resolve_fa4_bwd()
        if fa4_bwd is None:
            raise RuntimeError(
                "ATTN_V1_BWD_BACKEND=fa4 requested but FA4 backward spec "
                "could not be imported. See traceback above."
            )

        d_out_bhsd = (
            d_output
            .reshape(S, B, Hq, D)
            .permute(1, 0, 2, 3)
        )

        dq_sbhd = torch.empty(S, B, Hq, D, dtype=q.dtype, device=q.device)
        dk_sbhd = torch.empty(S, B, Hkv, D, dtype=q.dtype, device=q.device)
        dv_sbhd = torch.empty(S, B, Hkv, D, dtype=q.dtype, device=q.device)

        fa4_bwd(
            q_bhsd, k_bhsd, v_bhsd, out_bhsd, d_out_bhsd, lse,
            softmax_scale=scale,
            causal=True,
            dq=dq_sbhd.permute(1, 0, 2, 3),
            dk=dk_sbhd.permute(1, 0, 2, 3),
            dv=dv_sbhd.permute(1, 0, 2, 3),
        )

        return dq_sbhd, dk_sbhd, dv_sbhd

    # Fallback: manual autograd (ATTN_V1_BWD_BACKEND=manual or unrecognized)
    d_out_bhsd = (
        d_output
        .reshape(S, B, Hq, D)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    dq_bhsd, dk_bhsd, dv_bhsd = _manual_bwd_autograd(
        q_bhsd, k_bhsd, v_bhsd, d_out_bhsd, scale, True,
    )

    return (
        dq_bhsd.permute(1, 0, 2, 3).contiguous(),
        dk_bhsd.permute(1, 0, 2, 3).contiguous(),
        dv_bhsd.permute(1, 0, 2, 3).contiguous(),
    )
