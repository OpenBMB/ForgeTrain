"""Hopper (SM90a) Flash-Attention backward — 3-kernel architecture.

Kernel 1: FlashAttnBwdPreprocess  — dPsum, lse_log2, zero dQaccum
Kernel 2: FlashAttnBwdSm90        — main backward (384 threads, 3 WG)
Kernel 3: FlashAttnBwdPostprocess  — FP32 accum → FP16 with scale

Host dispatch: run_flash_bwd_dsl() orchestrates all 3 kernels.

Signature spec frozen per rules/milestones.md §Kernel Signature Spec.
"""

import math
import os
import torch

LOG2E = math.log2(math.e)

# ---------------------------------------------------------------------------
# CuTe DSL imports (conditional — fall back to PyTorch ref when unavailable)
# ---------------------------------------------------------------------------

try:
    import cuda.bindings.driver as cuda

    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, const_expr
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.cute.nvgpu import cpasync, warp, warpgroup
    import cutlass.pipeline as cutlass_pipeline
    import cutlass.utils.hopper_helpers as sm90_utils_basic
    from cutlass.cute.runtime import from_dlpack

    # Phase 1.7: layout helpers from vendored quack 0.3.11.
    #   - reshape_acc_to_mn(acc, transpose=...): transpose acc_S layout
    #     under SdP_swapAB so we can index it as P[m, n] in softmax.
    #   - reshape_acc_to_frgA(acc): convert wgmma C-output layout into
    #     the wgmma A-operand layout for back-to-back register-source.
    from quack.layout_utils import reshape_acc_to_mn, reshape_acc_to_frgA

    from . import utils, hopper_helpers
    from . import pipeline as fa_pipeline

    _HAS_CUTE = True
except ImportError:
    _HAS_CUTE = False


# ---------------------------------------------------------------------------
# Kernel class skeletons (FROZEN signatures)
# ---------------------------------------------------------------------------


class FlashAttnBwdPreprocess:
    """Preprocess kernel: dPsum, lse_log2, zero dQaccum.

    Phase 0: implemented via PyTorch ref.
    """

    def __init__(self, dtype=None, head_dim=64, tile_m=128, num_threads=256):
        self.dtype = dtype
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.num_threads = num_threads


class FlashAttnBwdSm90:
    """Main backward kernel: 384 threads, 3 WG, FA4 warp specialization."""

    def __init__(
        self,
        dtype=None,
        head_dim: int = 64,
        qhead_per_kvhead: int = 1,
        is_causal: bool = True,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        dQ_single_wg: bool = False,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_threads = num_threads
        self.num_wg_mma = (num_threads // 128) - 1
        self.dQ_single_wg = dQ_single_wg


class FlashAttnBwdPostprocess:
    """Postprocess kernel: FP32 accum * scale → FP16 output.

    Phase 0: implemented via PyTorch ref.
    """

    def __init__(self, dtype=None, head_dim=64, tile_m=128, num_threads=256,
                 AtomLayoutMdQ=1, dQ_swapAB=False):
        self.dtype = dtype
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.num_threads = num_threads
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.dQ_swapAB = dQ_swapAB


# ---------------------------------------------------------------------------
# PyTorch reference implementations (match kernel data flow exactly)
# ---------------------------------------------------------------------------


def _preprocess_ref(out, dout, dpsum, lse, lse_log2, dq_accum):
    """PyTorch ref for FlashAttnBwdPreprocess."""
    dpsum.copy_((out.float() * dout.float()).sum(dim=-1))
    lse_log2.copy_(lse * LOG2E)
    dq_accum.zero_()


def _postprocess_ref(accum, output, scale):
    """PyTorch ref for FlashAttnBwdPostprocess.

    Handles both legacy 4D `(B, H, N, D)` accum (used by dK/dV
    accumulators) and the new flat 3D `(B, H, N*D)` dq_accum.  The
    latter is the FA4-aligned layout where each (B, H_q) row is
    laid out as `(slot_size, num_wg_dQ)` per m_block — but the
    PyTorch ref path runs a tiled fallback that wrote in row-major
    order, so for ref-correctness the flat path just reshapes.
    """
    if accum.ndim == 3 and output.ndim == 4:
        accum_view = accum.view_as(output)
    else:
        accum_view = accum
    output.copy_((accum_view * scale).to(output.dtype))


# ---------------------------------------------------------------------------
# Tiled backward algorithm (PyTorch, mirrors FA4 CTA-level structure)
# ---------------------------------------------------------------------------


def _main_bwd_tiled(q, k, v, dout, lse_log2, dpsum, dq_accum, dk_accum,
                    dv_accum, softmax_scale, is_causal, qhead_per_kvhead):
    """Tiled backward pass — same block structure as CuTe DSL kernel."""
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    tile_m = 64
    tile_n = 128
    m_blocks = N // tile_m
    n_blocks = N // tile_n
    scale_log2 = softmax_scale * LOG2E

    for bidb in range(B):
        for bidh_kv in range(H_kv):
            for n_block in range(n_blocks):
                n_start = n_block * tile_n
                n_end = n_start + tile_n

                k_tile = k[bidb, bidh_kv, n_start:n_end, :].float()
                v_tile = v[bidb, bidh_kv, n_start:n_end, :].float()

                acc_dk = torch.zeros(tile_n, D, device=q.device,
                                     dtype=torch.float32)
                acc_dv = torch.zeros(tile_n, D, device=q.device,
                                     dtype=torch.float32)

                for gqa_idx in range(qhead_per_kvhead):
                    bidh_q = bidh_kv * qhead_per_kvhead + gqa_idx

                    if is_causal:
                        m_block_min = n_start // tile_m
                    else:
                        m_block_min = 0
                    m_block_max = m_blocks

                    for m_block in range(m_block_min, m_block_max):
                        m_start = m_block * tile_m
                        m_end = m_start + tile_m

                        q_tile = q[bidb, bidh_q, m_start:m_end, :].float()
                        do_tile = dout[bidb, bidh_q, m_start:m_end, :].float()
                        lse_tile = lse_log2[bidb, bidh_q, m_start:m_end]
                        dp_tile = dpsum[bidb, bidh_q, m_start:m_end]

                        s_tile = q_tile @ k_tile.T
                        s_scaled = s_tile * scale_log2 - lse_tile.unsqueeze(1)

                        if is_causal:
                            row_idx = torch.arange(
                                m_start, m_end, device=q.device
                            ).unsqueeze(1)
                            col_idx = torch.arange(
                                n_start, n_end, device=q.device
                            ).unsqueeze(0)
                            causal_mask = row_idx < col_idx
                            s_scaled = s_scaled.masked_fill(
                                causal_mask, float("-inf")
                            )

                        p_tile = torch.pow(2.0, s_scaled)
                        p_tile = torch.nan_to_num(p_tile, nan=0.0)

                        dp_tile_full = do_tile @ v_tile.T
                        ds_tile = p_tile * (
                            dp_tile_full - dp_tile.unsqueeze(1)
                        )

                        acc_dv += p_tile.T @ do_tile
                        dq_accum[bidb, bidh_q, m_start:m_end, :] += (
                            ds_tile @ k_tile
                        )
                        acc_dk += ds_tile.T @ q_tile

                dk_accum[bidb, bidh_kv, n_start:n_end, :] += acc_dk
                dv_accum[bidb, bidh_kv, n_start:n_end, :] += acc_dv


# ---------------------------------------------------------------------------
# CuTe DSL main backward kernel implementation
# ---------------------------------------------------------------------------

if _HAS_CUTE:

    @dsl_user_op
    def _ld_global_f32(
        gmem_ptr: cute.Pointer, *, loc=None, ip=None,
    ) -> Float32:
        """Load FP32 from global memory via PTX."""
        from cutlass.cute.arch import llvm
        result = llvm.inline_asm(
            T.f32(),
            [gmem_ptr.llvm_ptr],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
        )
        return Float32(result)

    class _BwdMainKernel:
        """CuTe DSL FA4-style backward main kernel.

        Phase 0 simplifications (correctness-first, optimize in Phase 2):
        - LSE/dPsum read from global memory (not via TMA to smem)
        - dQ via atomicAdd to global (not via warp 1 bulk_reduce)
        - dK/dV via store_global_fp32 (not via TMA S2G)
        """

        arch = 90

        def __init__(
            self,
            dtype,
            head_dim,
            qhead_per_kvhead,
            is_causal,
            # Tile shape (Phase 1.3)
            tile_m: int = 64,
            tile_n: int = 128,
            # Pipeline depth (Phase 1.5) — all >= 2 for proper overlap.
            Q_stage: int = 2,
            dO_stage: int = 2,
            PdS_stage: int = 2,
            sdQacc_stage: int = 2,
            # When True + sdQacc_stage==1 + dQ_single_wg, use FA4-style
            # named-barrier consumer↔warp-1 handshake instead of the
            # mbarrier path.  Setting to False forces mbarrier even
            # with single-buffer staging (used to isolate single-
            # buffer cost from sync-type cost in A/B/C ablation).
            use_named_dq_barrier: bool = True,
            # Layout shape (Phase 1.3 / 1.7) — formerly hardcoded; now
            # exposed to match FA4's signature so users can experiment
            # with SwapAB and atom-layout sweeps without editing kernel
            # source. Defaults reproduce the previous baseline exactly.
            SdP_swapAB: bool = False,
            dKV_swapAB: bool = False,
            dQ_swapAB: bool = False,
            AtomLayoutMSdP: int = 1,
            AtomLayoutNdKV: int = 2,
            AtomLayoutMdQ: int = 1,
            # Inter-iter overlap (Phase 1.7) — placeholder; full
            # `dQ_single_wg=True` path lands with K-2.
            dQ_single_wg: bool = False,
            # Threading
            num_threads: int = 384,
            # ── Per-GEMM early-exit instrumentation ─────────────
            # 0 = run the full iter; 1..5 = inside each iter,
            # drain wgmma right after GEMMi and skip the rest of
            # the body before releasing pipelines. Used to measure
            # cumulative critical-path time per GEMM cut and
            # compare against an FA4 build with the same cuts.
            # Correctness is broken when != 0 (acc_dV/dK/dQ
            # incomplete) — only call from perf benchmarks.
            early_exit_after_gemm: int = 0,
        ):
            # Note on Phase 1.7 (SwapAB+RS): in our cute-DSL setup the
            # RS path costs ~0.9 ms vs the SMEM-source baseline
            # (3.77 → 4.69 ms on H100), so SdP_swapAB defaults to False.
            # Full RS infrastructure (cvt_f16x2_f32, get_smem_store_C
            # with transpose+position-independent partition, ported
            # quack swizzle helpers, RS GEMM3/5 wiring) lives in
            # utils.py for reuse; opt in via SdP_swapAB=True.
            self.dtype = dtype
            self.head_dim = head_dim
            self.qhead_per_kvhead = qhead_per_kvhead
            self.is_causal = is_causal
            self.tile_m = tile_m
            self.tile_n = tile_n
            self.Q_stage = Q_stage
            self.dO_stage = dO_stage
            self.PdS_stage = PdS_stage
            # FA4 invariant (line 97-98): when PdS smem rotates with the
            # Q pipeline state, the two stage counts must agree (or PdS
            # collapses to single buffer).  Same for dO_stage.
            assert dO_stage in (1, Q_stage), (
                f"dO_stage must be 1 or {Q_stage}, got {dO_stage}")
            assert PdS_stage in (1, Q_stage), (
                f"PdS_stage must be 1 or {Q_stage}, got {PdS_stage} "
                "(otherwise smem_idx_PdS = q_state.index would index "
                "out of the sP/sdS staging buffer).")
            # Phase 1.5: consumer ↔ warp 1 staging. >= 2 to avoid the dQ
            # ping-pong bottleneck.
            self.sdQacc_stage = sdQacc_stage
            assert sdQacc_stage >= 1, (
                "sdQacc_stage must be >= 1.")
            self.use_named_dq_barrier = use_named_dq_barrier
            self.SdP_swapAB = SdP_swapAB
            self.dKV_swapAB = dKV_swapAB
            self.dQ_swapAB = dQ_swapAB
            self.AtomLayoutMSdP = AtomLayoutMSdP
            self.AtomLayoutNdKV = AtomLayoutNdKV
            self.AtomLayoutMdQ = AtomLayoutMdQ
            # Threading — match FA4: warp groups 0=producer, 1..=consumer.
            self.num_threads = num_threads
            assert num_threads % 128 == 0, "num_threads must be multiple of 128"
            self.num_wg_mma = (num_threads // 128) - 1
            assert self.num_wg_mma >= 1, "need >= 1 consumer warp group"
            # K-2: dQ_single_wg=True lets WG0 own the entire mma_dQ
            # (atom_layout=(1,1,1), tiler=(tile_m, D)) while WG1 skips
            # GEMM4 / dQ R2S and continues directly to GEMM5(dK) +
            # release(Q).  WG1 thus reaches the next iter's consumer_wait
            # ~one dQ-GEMM ahead of WG0, exposing inter-iter overlap.
            # Only valid for num_wg_mma == 2.
            self.dQ_single_wg = dQ_single_wg
            if dQ_single_wg:
                assert self.num_wg_mma == 2, (
                    "dQ_single_wg requires num_wg_mma == 2 (got "
                    f"{self.num_wg_mma}); the path slots WG0 onto a "
                    "(1,1,1) mma_dQ and WG1 idles GEMM4 to early-"
                    "start the next iter's GEMM1.")
            self.num_wg_dQ = 1 if dQ_single_wg else self.num_wg_mma
            # Auto-derived: register-source dKV gates on (matches FA4).
            self.mma_dkv_is_rs = (
                self.AtomLayoutMSdP == 1
                and self.AtomLayoutNdKV == self.num_wg_mma
                and self.SdP_swapAB
                and not self.dKV_swapAB
            )
            # GQA invariant inherited from FA4 (line 116):
            if qhead_per_kvhead > 1:
                assert self.num_wg_mma == 2, (
                    "GQA backward assumes 2 consumer warp groups")
            self.num_mma_threads = 128 * self.num_wg_mma
            self.num_producer_regs = 24
            self.num_mma_regs = 240
            self.BARRIER_PDS = 5
            self.BARRIER_INIT = 6
            # dQ R2S sync — FA4-style PER-WG named barrier pairs.
            # Empty[wg]: warp 1 → consumer WG[wg] (slot[wg] ready to
            # write).  Full[wg]: consumer WG[wg] → warp 1 (slot[wg]
            # has data).  Number of threads per barrier = 1 consumer
            # WG + warp 1 = 128 + 32 = 160.
            #
            # Layout: BARRIER_DQ_EMPTY_BASE + wg_idx (wg_idx ∈
            # [0..num_wg_dQ)).  Same for FULL.  This matches FA4
            # `dQEmptyWG0 + warp_group_idx` (mainloop_bwd_sm90 line
            # 1587 / 1845).
            self.BARRIER_DQ_EMPTY_BASE = 7
            self.BARRIER_DQ_FULL_BASE = 9
            self.num_dq_sync_threads = 128 + 32
            assert 0 <= early_exit_after_gemm <= 5, (
                f"early_exit_after_gemm must be 0..5, got {early_exit_after_gemm}")
            self.early_exit_after_gemm = early_exit_after_gemm

        def _make_smem_layouts(self):
            dtype = self.dtype
            D = self.head_dim

            q_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    cutlass.utils.LayoutEnum.ROW_MAJOR, dtype, D,
                ), dtype,
            )
            n_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    cutlass.utils.LayoutEnum.ROW_MAJOR, dtype, D,
                ), dtype,
            )
            # sP / sdS atom — both D=64 and tile_n=128 yield SW128
            # in get_smem_layout_atom (1024 / 2048 bits both
            # multiples of 1024).  We pass tile_n to match FA4
            # (quack.sm90_utils.make_smem_layout line 234, where
            # major_mode_size=gcd(tile_n//wg_n_SdP, ...)) — same
            # SW128 atom, no functional change but documents intent.
            p_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    cutlass.utils.LayoutEnum.ROW_MAJOR, dtype,
                    self.tile_n,
                ), dtype,
            )

            sQ_l = cute.tile_to_shape(
                q_atom, (self.tile_m, D, self.Q_stage), (0, 1, 2))
            sdO_l = cute.tile_to_shape(
                q_atom, (self.tile_m, D, self.dO_stage), (0, 1, 2))
            sK_l = cute.tile_to_shape(n_atom, (self.tile_n, D), (0, 1))
            sV_l = cute.tile_to_shape(n_atom, (self.tile_n, D), (0, 1))
            sP_l = cute.tile_to_shape(
                p_atom, (self.tile_m, self.tile_n, self.PdS_stage), (0, 1, 2))
            sdS_l = cute.tile_to_shape(
                p_atom, (self.tile_m, self.tile_n, self.PdS_stage), (0, 1, 2))
            # D-1: LSE / dPsum staging — per-stage tile_m FP32 vector.
            # cp.async.bulk loaded by warp 0 inside the Q / dO TMA
            # mbarrier so consumer reads sLSE[r, q_state.index] from
            # smem instead of the per-row gmem load (~ saves several
            # hundred cycles of long_scoreboard per softmax row).
            sLSE_l = cute.make_layout(
                (self.tile_m, self.Q_stage), stride=(1, self.tile_m))
            sdPsum_l = cute.make_layout(
                (self.tile_m, self.dO_stage), stride=(1, self.tile_m))
            # FA4-style flat sdQaccum layout (mainloop_bwd_sm90.py
            # line 236):  (slot_size, num_wg_dQ, sdQacc_stage)
            # where slot_size = tile_m * D / num_wg_dQ.
            #
            # The R2S into this flat buffer uses a flat TV copy
            # (make_tiled_copy_tv((128, num_wg_dQ), val=4)) so each
            # thread writes 4 contiguous fp32 (128-bit STS) at
            # bank-friendly offsets — eliminates the 51.8 % bank
            # conflict that the WGMMA-frag-pattern partition_D into
            # row-major (slot_m, slot_d) suffered.  The mdQaccum
            # tensor is correspondingly reshaped to (B, H_q, N*D)
            # 1D-per-(B,H_q) so the round-trip layout matches.
            #
            # `_dQ_M_split` retained for warp-1 store-side gdQ
            # tiling: M-split tiles along outer m, D-split tiles
            # along WG.
            self._dQ_M_split = (
                self.AtomLayoutMdQ == self.num_wg_mma
                and self.num_wg_dQ == self.num_wg_mma
            )
            slot_size = self.tile_m * D // self.num_wg_dQ
            wg_size = slot_size * self.num_wg_dQ
            sdQacc_l = cute.make_layout(
                (slot_size, self.num_wg_dQ, self.sdQacc_stage),
                stride=(1, slot_size, wg_size),
            )

            return (sQ_l, sdO_l, sK_l, sV_l, sP_l, sdS_l, sdQacc_l,
                    sLSE_l, sdPsum_l)

        def _make_tiled_mmas(self):
            """Phase 1.7: SwapAB + register-source dV/dK.

            * mma_SdP no-swap: A=Q (K-major), B=K (K-major), out=S(M,N).
              SwapAB:           A=K (K-major), B=Q (K-major), out=S^T(N,M).
              The atom_layout swaps M↔N too so each WG owns a contiguous
              stripe in the *M* axis (semantically tile_n) of acc_S^T.
            * mma_dV / mma_dK: A=P^T / dS^T. With register-source the A
              operand comes straight out of the SwapAB acc_S/acc_dP
              fragment (no R2S to sP/sdS).  A leading mode flips to K
              when RS (FA4 uses K-major register A for back-to-back gemm).
            """
            dtype = self.dtype
            D = self.head_dim
            K = warpgroup.OperandMajorMode.K
            MN = warpgroup.OperandMajorMode.MN
            RMEM = warpgroup.OperandSource.RMEM
            SMEM = warpgroup.OperandSource.SMEM

            def _maybe_swap(s, swap):
                return (s[1], s[0], *s[2:]) if swap else s

            atom_layout_SdP = (
                self.AtomLayoutMSdP,
                self.num_wg_mma // self.AtomLayoutMSdP,
                1,
            )
            tiler_orig_SdP = (
                self.tile_m // atom_layout_SdP[0],
                self.tile_n // atom_layout_SdP[1],
            )
            tiler_SdP = (
                tiler_orig_SdP[1] if self.SdP_swapAB else tiler_orig_SdP[0],
                tiler_orig_SdP[0] if self.SdP_swapAB else tiler_orig_SdP[1],
            )
            mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
                dtype, dtype, K, K, Float32,
                atom_layout_mnk=_maybe_swap(atom_layout_SdP, self.SdP_swapAB),
                tiler_mn=tiler_SdP,
            )

            atom_layout_dKV = (
                self.AtomLayoutNdKV,
                self.num_wg_mma // self.AtomLayoutNdKV,
                1,
            )
            tiler_dKV = (
                self.tile_n // atom_layout_dKV[0],
                D // atom_layout_dKV[1],
            )
            a_lead_dKV = K if self.mma_dkv_is_rs else MN
            a_source_dKV = RMEM if self.mma_dkv_is_rs else SMEM
            mma_dV = sm90_utils_basic.make_trivial_tiled_mma(
                dtype, dtype, a_lead_dKV, MN, Float32,
                atom_layout_mnk=atom_layout_dKV,
                tiler_mn=tiler_dKV,
                a_source=a_source_dKV,
            )
            mma_dK = sm90_utils_basic.make_trivial_tiled_mma(
                dtype, dtype, a_lead_dKV, MN, Float32,
                atom_layout_mnk=atom_layout_dKV,
                tiler_mn=tiler_dKV,
                a_source=a_source_dKV,
            )

            # K-2: when dQ_single_wg, only WG0 issues mma_dQ — its
            # atom_layout collapses to (AtomLayoutMdQ, 1, 1) (instead
            # of splitting D between two WGs as (1, num_wg_mma, 1)).
            # Tiler still covers the full (tile_m, D) per WG.
            atom_layout_dQ = (
                self.AtomLayoutMdQ,
                self.num_wg_dQ // self.AtomLayoutMdQ,
                1,
            )
            tiler_dQ = (
                self.tile_m // atom_layout_dQ[0],
                D // atom_layout_dQ[1],
            )
            mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
                dtype, dtype, K, MN, Float32,
                atom_layout_mnk=atom_layout_dQ,
                tiler_mn=tiler_dQ,
            )
            return mma_SdP, mma_dV, mma_dQ, mma_dK

        def _make_shared_storage_cls(self, sQ_l, sdO_l, sK_l, sV_l,
                                     sP_l, sdS_l, sdQacc_l,
                                     sLSE_l, sdPsum_l):
            dtype = self.dtype
            sQ_sz = cute.cosize(sQ_l)
            sdO_sz = cute.cosize(sdO_l)
            sK_sz = cute.cosize(sK_l)
            sV_sz = cute.cosize(sV_l)
            # When mma_dkv_is_rs, sP smem is unused (P stays in
            # registers as the WGMMA acc output of GEMM1+softmax).
            # Skip allocation to save 64 KB at tile_m=128.
            sP_sz = (0 if self.mma_dkv_is_rs else cute.cosize(sP_l))
            sdS_sz = cute.cosize(sdS_l)
            sdQacc_sz = cute.cosize(sdQacc_l)
            sLSE_sz = cute.cosize(sLSE_l)
            sdPsum_sz = cute.cosize(sdPsum_l)

            @cute.struct
            class SharedStorage:
                mbar_Q: cute.struct.MemRange[cutlass.Int64,
                                             self.Q_stage * 2]
                mbar_dO: cute.struct.MemRange[cutlass.Int64,
                                              self.dO_stage * 2]
                # Phase 1.5 + K-2: per-WG_dQ per-stage dQ mbarriers.
                # Index = wg_dQ + num_wg_dQ * stage.
                mbar_dq_full: cute.struct.MemRange[
                    cutlass.Int64,
                    self.num_wg_dQ * self.sdQacc_stage]
                mbar_dq_empty: cute.struct.MemRange[
                    cutlass.Int64,
                    self.num_wg_dQ * self.sdQacc_stage]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[dtype, sQ_sz], 128]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[dtype, sdO_sz], 128]
                sK: cute.struct.Align[
                    cute.struct.MemRange[dtype, sK_sz], 128]
                sV: cute.struct.Align[
                    cute.struct.MemRange[dtype, sV_sz], 128]
                sP: cute.struct.Align[
                    cute.struct.MemRange[dtype, sP_sz], 128]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[dtype, sdS_sz], 128]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[Float32, sdQacc_sz], 128]
                # D-1: LSE / dPsum staging buffers.
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[Float32, sLSE_sz], 128]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[Float32, sdPsum_sz], 128]

            return SharedStorage

        # ── Host launcher ──

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
            mdO: cute.Tensor,
            mLSElog2: cute.Tensor, mdPsum: cute.Tensor,
            mdQaccum: cute.Tensor,
            mdK: cute.Tensor, mdV: cute.Tensor,
            softmax_scale: Float32,
            stream: cuda.CUstream = None,
        ):
            new_stride4 = lambda t: (
                *(cute.assume(s, divby=128 // t.element_type.width)
                  for s in t.stride[:-1]),
                t.stride[-1],
            )
            new_stride3 = lambda t: (
                *(cute.assume(s, divby=4) for s in t.stride[:-1]),
                t.stride[-1],
            )

            mQ = cute.make_tensor(mQ.iterator,
                cute.make_layout(mQ.shape, stride=new_stride4(mQ)))
            mK = cute.make_tensor(mK.iterator,
                cute.make_layout(mK.shape, stride=new_stride4(mK)))
            mV = cute.make_tensor(mV.iterator,
                cute.make_layout(mV.shape, stride=new_stride4(mV)))
            mdO = cute.make_tensor(mdO.iterator,
                cute.make_layout(mdO.shape, stride=new_stride4(mdO)))
            # FA4-style flat dq_accum: 3D (B, H_q, N*D). Same stride
            # treatment as 3D mLSElog2 / mdPsum (last dim contig).
            mdQaccum = cute.make_tensor(mdQaccum.iterator,
                cute.make_layout(mdQaccum.shape,
                                 stride=new_stride3(mdQaccum)))
            mdK = cute.make_tensor(mdK.iterator,
                cute.make_layout(mdK.shape, stride=new_stride4(mdK)))
            mdV = cute.make_tensor(mdV.iterator,
                cute.make_layout(mdV.shape, stride=new_stride4(mdV)))
            mLSElog2 = cute.make_tensor(mLSElog2.iterator,
                cute.make_layout(mLSElog2.shape,
                                 stride=new_stride3(mLSElog2)))
            mdPsum = cute.make_tensor(mdPsum.iterator,
                cute.make_layout(mdPsum.shape,
                                 stride=new_stride3(mdPsum)))

            mQ, mK, mV, mdO, mdK, mdV = [
                cute.make_tensor(
                    t.iterator, cute.select(t.layout, mode=[2, 3, 1, 0])
                ) for t in (mQ, mK, mV, mdO, mdK, mdV)
            ]
            # mdQaccum 3D is reordered to (N*D, H_q, B) — last dim
            # is now batch, middle is head, first is the flat
            # per-block bytes index.  Matches FA4 mdQaccum_cur =
            # mdQaccum[None, head_idx, batch_idx] which slices to
            # 1D (N*D,).
            mdQaccum = cute.make_tensor(
                mdQaccum.iterator,
                cute.select(mdQaccum.layout, mode=[2, 1, 0]),
            )
            mLSElog2, mdPsum = [
                cute.make_tensor(
                    t.iterator, cute.select(t.layout, mode=[2, 1, 0])
                ) for t in (mLSElog2, mdPsum)
            ]

            (sQ_l, sdO_l, sK_l, sV_l, sP_l, sdS_l, sdQacc_l,
             sLSE_l, sdPsum_l) = self._make_smem_layouts()
            mma_SdP, mma_dV, mma_dQ, mma_dK = self._make_tiled_mmas()

            tma_G2S = cpasync.CopyBulkTensorTileG2SOp()

            tma_q_bytes = cute.size_in_bytes(
                self.dtype, cute.select(sQ_l, mode=[0, 1]))
            tma_do_bytes = cute.size_in_bytes(
                self.dtype, cute.select(sdO_l, mode=[0, 1]))
            tma_k_bytes = cute.size_in_bytes(self.dtype, sK_l)
            tma_v_bytes = cute.size_in_bytes(self.dtype, sV_l)
            # D-1: extra mbar tx_count for LSE / dPsum cp.async.bulk
            # piggyback. tile_m FP32 vector loaded inside the same Q /
            # dO mbarrier as Q / dO themselves (FA4 tx_count line 690 /
            # 698).
            tma_lse_bytes = self.tile_m * 4   # FP32
            tma_dpsum_bytes = self.tile_m * 4

            tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
                tma_G2S, mQ,
                cute.select(sQ_l, mode=[0, 1]),
                (self.tile_m, self.head_dim),
            )
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                tma_G2S, mK, sK_l,
                (self.tile_n, self.head_dim),
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                tma_G2S, mV, sV_l,
                (self.tile_n, self.head_dim),
            )
            tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
                tma_G2S, mdO,
                cute.select(sdO_l, mode=[0, 1]),
                (self.tile_m, self.head_dim),
            )
            # FA4 path: warp 1 issues raw cp.reduce.async.bulk.add
            # (PTX `cp.reduce.async.bulk.shared::cluster.global.add`)
            # against (smem_ptr, gmem_ptr, bytes).  No TMA descriptor
            # needed because the source/dest are already 1D contiguous
            # bytes and there's no swizzle to undo.  Matches
            # FA4 mainloop_bwd_sm90.py line 1872+
            # `copy_utils.cpasync_reduce_bulk_add_f32`.

            # FA4-grid path: GQA cross-CTA dK/dV accumulation via
            # cp.reduce.async.bulk.add to FP32 mdKaccum/mdVaccum.
            # We use the raw PTX wrapper (utils.cpasync_reduce_bulk_add
            # _f32) inside epilogue_dKV — no TMA descriptor needed
            # (just smem ptr + gmem ptr + bytes).  Final FP32 → FP16
            # cast by the dKV postprocess kernel.

            SharedStorage = self._make_shared_storage_cls(
                sQ_l, sdO_l, sK_l, sV_l, sP_l, sdS_l, sdQacc_l,
                sLSE_l, sdPsum_l,
            )

            N_dim = cute.size(mQ.shape[0])
            H_q = cute.size(mQ.shape[2])
            B_dim = cute.size(mQ.shape[3])

            n_n_blocks = N_dim // self.tile_n
            # FA4 grid: (n_blocks, H_q, B).  Each (n_block, h_q, batch)
            # tile owned by one CTA; multiple CTAs share the same H_kv
            # slot of dK/dV via cp.reduce.bulk.add atomic accumulation.
            grid = (n_n_blocks, H_q, B_dim)

            scale_log2 = softmax_scale * Float32(LOG2E)

            self.kernel(
                tma_tensor_Q, tma_tensor_K, tma_tensor_V, tma_tensor_dO,
                mLSElog2, mdPsum, mdQaccum,
                mdK, mdV,
                tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO,
                sQ_l, sdO_l, sK_l, sV_l, sP_l, sdS_l, sdQacc_l,
                sLSE_l, sdPsum_l,
                mma_SdP, mma_dV, mma_dQ, mma_dK,
                SharedStorage,
                tma_q_bytes, tma_do_bytes, tma_k_bytes, tma_v_bytes,
                tma_lse_bytes, tma_dpsum_bytes,
                scale_log2, softmax_scale,
            ).launch(
                grid=grid,
                block=[self.num_threads, 1, 1],
                smem=SharedStorage.size_in_bytes(),
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=True,
            )

        # ── Device kernel entry ──

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
            mdO: cute.Tensor,
            mLSElog2: cute.Tensor, mdPsum: cute.Tensor,
            mdQaccum: cute.Tensor, mdK: cute.Tensor, mdV: cute.Tensor,
            tma_atom_Q: cute.CopyAtom, tma_atom_K: cute.CopyAtom,
            tma_atom_V: cute.CopyAtom, tma_atom_dO: cute.CopyAtom,
            sQ_layout: cute.ComposedLayout, sdO_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout,
            sP_layout: cute.ComposedLayout, sdS_layout: cute.ComposedLayout,
            sdQacc_layout: cute.Layout,
            sLSE_layout: cute.Layout, sdPsum_layout: cute.Layout,
            mma_SdP: cute.TiledMma, mma_dV: cute.TiledMma,
            mma_dQ: cute.TiledMma, mma_dK: cute.TiledMma,
            SharedStorage: cutlass.Constexpr,
            tma_q_bytes: cutlass.Constexpr, tma_do_bytes: cutlass.Constexpr,
            tma_k_bytes: cutlass.Constexpr, tma_v_bytes: cutlass.Constexpr,
            tma_lse_bytes: cutlass.Constexpr,
            tma_dpsum_bytes: cutlass.Constexpr,
            scale_log2: Float32, softmax_scale: Float32,
        ):
            # ── Warp dispatch ──
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)

            # FA4-grid: griddepcontrol_wait deferred into producer
            # warp 0 (before LSE/dPsum TMA fire) and warp 1
            # (before cp.reduce.bulk.add).  K/V/Q/dO TMA loads
            # don't depend on prep, so let them overlap with prep
            # tail.  With 38.8 waves the wave-level pipelining
            # hides any prep+main HBM contention.

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            sQ = storage.sQ.get_tensor(sQ_layout.outer,
                                       swizzle=sQ_layout.inner)
            sdO = storage.sdO.get_tensor(sdO_layout.outer,
                                         swizzle=sdO_layout.inner)
            sK = storage.sK.get_tensor(sK_layout.outer,
                                       swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer,
                                       swizzle=sV_layout.inner)
            sdS = storage.sdS.get_tensor(sdS_layout.outer,
                                         swizzle=sdS_layout.inner)
            if const_expr(self.mma_dkv_is_rs):
                # sP smem skipped (RS path).  Use sdS as a typed
                # placeholder so downstream paths that branch on
                # mma_dkv_is_rs can keep a valid handle (e.g. epi).
                sP = sdS
            else:
                sP = storage.sP.get_tensor(sP_layout.outer,
                                           swizzle=sP_layout.inner)
            # Phase 1.1: per-WG dQ accumulator slot in smem (FP32).
            sdQacc = storage.sdQaccum.get_tensor(sdQacc_layout)
            # D-1: LSE / dPsum staging buffers (FP32, tile_m × stage).
            sLSE = storage.sLSE.get_tensor(sLSE_layout)
            sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)

            prod_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread)
            cons_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread, self.num_wg_mma)

            pipeline_Q = fa_pipeline.TmaPipelineNoCluster.create(
                barrier_storage=storage.mbar_Q.data_ptr(),
                num_stages=self.Q_stage,
                producer_group=prod_group,
                consumer_group=cons_group,
                # D-1: piggyback LSE bytes on the Q mbarrier.
                tx_count=tma_q_bytes + tma_lse_bytes,
                init_wait=False,
            )
            pipeline_dO = fa_pipeline.TmaPipelineNoCluster.create(
                barrier_storage=storage.mbar_dO.data_ptr(),
                num_stages=self.dO_stage,
                producer_group=prod_group,
                consumer_group=cons_group,
                # D-1: piggyback dPsum bytes on the dO mbarrier.
                tx_count=tma_do_bytes + tma_dpsum_bytes,
            )

            # Phase 1.5 + K-2: init per-WG_dQ per-stage dQ mbarriers.
            #   index = wg_dQ + num_wg_dQ * stage  (num_wg_dQ * stage)
            #   mbar_dq_full:  arrive_count = 128 (one consumer WG)
            #   mbar_dq_empty: arrive_count = 32  (warp 1)
            mbar_dq_full_ptr = storage.mbar_dq_full.data_ptr()
            mbar_dq_empty_ptr = storage.mbar_dq_empty.data_ptr()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    for s in cutlass.range_constexpr(
                        self.sdQacc_stage
                    ):
                        for wg in cutlass.range_constexpr(
                            self.num_wg_dQ
                        ):
                            idx = wg + self.num_wg_dQ * s
                            cute.arch.mbarrier_init(
                                mbar_dq_full_ptr + idx, 128)
                            cute.arch.mbarrier_init(
                                mbar_dq_empty_ptr + idx, 32)
                cute.arch.mbarrier_init_fence()
            # Sync all 384 threads so mbarrier init is visible before any use.
            cute.arch.barrier_arrive(
                barrier_id=self.BARRIER_INIT,
                number_of_threads=self.num_threads)
            cute.arch.barrier(
                barrier_id=self.BARRIER_INIT,
                number_of_threads=self.num_threads)

            # FA4 grid: (n_block, h_q, batch).  Compute kv_head_idx
            # from h_q via integer division (qhead_per_kvhead is
            # constexpr, so this folds at compile time).
            n_block, h_q_idx, batch_idx = cute.arch.block_idx()
            kv_head_idx = h_q_idx // self.qhead_per_kvhead

            N_dim = cute.size(mQ.shape[0])
            m_blocks_total = N_dim // self.tile_m

            if warp_idx < 4:
                # Producer WG (warps 0-3): TMA loads + warp-1 dQ
                # store.  Hand most of the per-thread register file
                # to consumer WGs by deallocating to 24 regs (FA4
                # mainloop_bwd_sm90 line 756).  This is the
                # difference between 168-reg/thread (uniform alloc,
                # spills) and 240-reg/thread for consumer WGs
                # (no spill at tile_m=128).
                #
                # MUST be the FIRST instruction in the branch — any
                # work before setmaxnreg corrupts registers.
                cute.arch.warpgroup_reg_dealloc(
                    self.num_producer_regs)
                if warp_idx == 0:
                    self.load(
                        mQ, mK, mV, mdO, mLSElog2, mdPsum,
                        sQ, sdO, sK, sV, sLSE, sdPsum,
                        tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO,
                        pipeline_Q, pipeline_dO,
                        tma_k_bytes, tma_v_bytes,
                        n_block, h_q_idx, kv_head_idx, batch_idx,
                        m_blocks_total,
                    )
                elif warp_idx == 1:
                    self.dQaccum_store(
                        mdQaccum, sdQacc,
                        mbar_dq_full_ptr, mbar_dq_empty_ptr,
                        n_block, h_q_idx, batch_idx,
                        m_blocks_total,
                    )
            else:
                # Consumer WG: take 240 regs/thread (FA4 line 834).
                cute.arch.warpgroup_reg_alloc(
                    self.num_mma_regs)
                tidx = cute.arch.thread_idx()[0] - Int32(128)
                wg_idx = (warp_idx - 4) // 4

                self.mma(
                    mdQaccum, mdK, mdV,
                    sQ, sdO, sK, sV, sP, sdS, sdQacc, sLSE, sdPsum,
                    mbar_dq_full_ptr, mbar_dq_empty_ptr,
                    mma_SdP, mma_dV, mma_dQ, mma_dK,
                    pipeline_Q, pipeline_dO,
                    n_block, h_q_idx, kv_head_idx, batch_idx,
                    m_blocks_total,
                    tidx, wg_idx,
                    scale_log2, softmax_scale,
                )

            # PDL: signal that downstream kernels (post) can start
            # their prologue.  Issued by every warp; runtime de-dups.
            cute.arch.griddepcontrol_launch_dependents()

        # ── Warp 0: TMA producer ──

        @cute.jit
        def load(
            self,
            mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
            mdO: cute.Tensor,
            mLSElog2: cute.Tensor, mdPsum: cute.Tensor,
            sQ: cute.Tensor, sdO: cute.Tensor,
            sK: cute.Tensor, sV: cute.Tensor,
            sLSE: cute.Tensor, sdPsum: cute.Tensor,
            tma_atom_Q: cute.CopyAtom, tma_atom_K: cute.CopyAtom,
            tma_atom_V: cute.CopyAtom, tma_atom_dO: cute.CopyAtom,
            pipeline_Q: fa_pipeline.TmaPipelineNoCluster,
            pipeline_dO: fa_pipeline.TmaPipelineNoCluster,
            tma_k_bytes: cutlass.Constexpr, tma_v_bytes: cutlass.Constexpr,
            n_block: Int32, h_q_idx: Int32, kv_head_idx: Int32,
            batch_idx: Int32,
            m_blocks_total: Int32,
        ):
            # FA4-grid path: one CTA owns one (n_block, h_q, batch)
            # tile, no gqa loop.  K/V indexed via kv_head_idx, Q/dO/
            # LSE/dPsum indexed via h_q_idx.
            gK = mK[None, None, kv_head_idx, batch_idx]
            gV = mV[None, None, kv_head_idx, batch_idx]
            gK_tile = cute.local_tile(
                gK, (self.tile_n, self.head_dim), (n_block, 0))
            gV_tile = cute.local_tile(
                gV, (self.tile_n, self.head_dim), (n_block, 0))

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K, 0, cute.make_layout(1),
                cute.group_modes(sK, 0, 2),
                cute.group_modes(gK_tile, 0, 2),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V, 0, cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gV_tile, 0, 2),
            )

            # D-1: cp.async.bulk atom for LSE / dPsum (FP32 1D copies).
            # Loaded into the same Q / dO mbarrier so the consumer sees
            # them ready at the same time as Q / dO via consumer_wait.
            bulk_atom_lse = cute.make_copy_atom(
                cpasync.CopyBulkG2SOp(), Float32)

            n_start = n_block * self.tile_n

            q_state = fa_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Producer, self.Q_stage)
            do_state = fa_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Producer, self.dO_stage)

            # PDL: producer waits before LSE/dPsum cp.async.bulk.
            # K/V/Q/dO TMA can fire freely (mbarrier sync covers
            # consumer access).  FA4 line 957 same idea.
            cute.arch.griddepcontrol_wait()

            gQ_h = mQ[None, None, h_q_idx, batch_idx]
            gdO_h = mdO[None, None, h_q_idx, batch_idx]
            gLSE_h = mLSElog2[None, h_q_idx, batch_idx]
            gdPsum_h = mdPsum[None, h_q_idx, batch_idx]
            gQ_all = cute.local_tile(
                gQ_h, (self.tile_m, self.head_dim), (None, 0))
            gdO_all = cute.local_tile(
                gdO_h, (self.tile_m, self.head_dim), (None, 0))
            gLSE_all = cute.local_tile(
                gLSE_h, (self.tile_m,), (None,))
            gdPsum_all = cute.local_tile(
                gdPsum_h, (self.tile_m,), (None,))

            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q, 0, cute.make_layout(1),
                cute.group_modes(sQ, 0, 2),
                cute.group_modes(gQ_all, 0, 2),
            )
            tOsO, tOgO = cpasync.tma_partition(
                tma_atom_dO, 0, cute.make_layout(1),
                cute.group_modes(sdO, 0, 2),
                cute.group_modes(gdO_all, 0, 2),
            )

            if const_expr(self.is_causal):
                m_block_min = n_start // self.tile_m
            else:
                m_block_min = Int32(0)
            m_block_max = m_blocks_total

            for m_idx in cutlass.range(
                m_block_max - m_block_min, unroll=1
            ):
                m_block = m_block_min + m_idx
                is_first_iter = (m_idx == 0)

                pipeline_Q.producer_acquire(q_state)
                if is_first_iter:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(
                            pipeline_Q.producer_get_barrier(q_state),
                            tma_k_bytes,
                        )
                    cute.copy(tma_atom_K, tKgK, tKsK,
                              tma_bar_ptr=(
                                  pipeline_Q.producer_get_barrier(
                                      q_state)))

                cute.copy(tma_atom_Q,
                          tQgQ[None, m_block],
                          tQsQ[None, q_state.index],
                          tma_bar_ptr=(
                              pipeline_Q.producer_get_barrier(q_state)))
                # D-1: piggyback LSE chunk on the Q mbarrier — its
                # bytes were included in pipeline_Q.tx_count.
                with cute.arch.elect_one():
                    cute.copy(
                        bulk_atom_lse,
                        gLSE_all[None, m_block],
                        sLSE[None, q_state.index],
                        mbar_ptr=pipeline_Q.producer_get_barrier(q_state),
                    )
                q_state.advance()

                pipeline_dO.producer_acquire(do_state)
                if is_first_iter:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(
                            pipeline_dO.producer_get_barrier(do_state),
                            tma_v_bytes,
                        )
                    cute.copy(tma_atom_V, tVgV, tVsV,
                              tma_bar_ptr=(
                                  pipeline_dO.producer_get_barrier(
                                      do_state)))

                cute.copy(tma_atom_dO,
                          tOgO[None, m_block],
                          tOsO[None, do_state.index],
                          tma_bar_ptr=(
                              pipeline_dO.producer_get_barrier(do_state)))
                # D-1: piggyback dPsum chunk on the dO mbarrier.
                with cute.arch.elect_one():
                    cute.copy(
                        bulk_atom_lse,
                        gdPsum_all[None, m_block],
                        sdPsum[None, do_state.index],
                        mbar_ptr=pipeline_dO.producer_get_barrier(
                            do_state),
                    )
                do_state.advance()

        # ── Consumer WG0+WG1: MMA ──

        @cute.jit
        def mma(
            self,
            mdQaccum: cute.Tensor, mdK: cute.Tensor, mdV: cute.Tensor,
            sQ: cute.Tensor, sdO: cute.Tensor,
            sK: cute.Tensor, sV: cute.Tensor,
            sP: cute.Tensor, sdS: cute.Tensor,
            sdQacc: cute.Tensor,
            sLSE: cute.Tensor, sdPsum: cute.Tensor,
            mbar_dq_full_ptr: cute.Pointer,
            mbar_dq_empty_ptr: cute.Pointer,
            mma_SdP: cute.TiledMma, mma_dV: cute.TiledMma,
            mma_dQ: cute.TiledMma, mma_dK: cute.TiledMma,
            pipeline_Q: fa_pipeline.TmaPipelineNoCluster,
            pipeline_dO: fa_pipeline.TmaPipelineNoCluster,
            n_block: Int32, h_q_idx: Int32, kv_head_idx: Int32,
            batch_idx: Int32,
            m_blocks_total: Int32,
            tidx: Int32, wg_idx: Int32,
            scale_log2: Float32, softmax_scale: Float32,
        ):
            n_start = n_block * self.tile_n

            wg_layout = cute.make_layout(self.num_wg_mma, stride=128)

            wg_slice_SdP = mma_SdP.get_slice(wg_layout(wg_idx))
            wg_slice_dV = mma_dV.get_slice(wg_layout(wg_idx))
            wg_slice_dQ = mma_dQ.get_slice(wg_layout(wg_idx))
            wg_slice_dK = mma_dK.get_slice(wg_layout(wg_idx))

            sKt = utils.transpose_smem_view(sK)
            sVt = utils.transpose_smem_view(sV)
            sPt = utils.transpose_smem_view(sP)
            sdSt = utils.transpose_smem_view(sdS)
            sdOt = utils.transpose_smem_view(sdO)
            sQt = utils.transpose_smem_view(sQ)

            # ── SwapAB partitions for mma_SdP ──────────────────────
            # No-swap: A=Q, B=K. Swap: A=K, B=Q (so output is S^T).
            if const_expr(self.SdP_swapAB):
                tSrK = mma_SdP.make_fragment_A(
                    wg_slice_SdP.partition_A(sK))
                tSrQ = mma_SdP.make_fragment_B(
                    wg_slice_SdP.partition_B(sQ))
                tSrV = mma_SdP.make_fragment_A(
                    wg_slice_SdP.partition_A(sV))
                tSrdO = mma_SdP.make_fragment_B(
                    wg_slice_SdP.partition_B(sdO))
            else:
                tSrQ = mma_SdP.make_fragment_A(
                    wg_slice_SdP.partition_A(sQ))
                tSrK = mma_SdP.make_fragment_B(
                    wg_slice_SdP.partition_B(sK))
                tSrdO = mma_SdP.make_fragment_A(
                    wg_slice_SdP.partition_A(sdO))
                tSrV = mma_SdP.make_fragment_B(
                    wg_slice_SdP.partition_B(sV))

            # ── mma_dV / mma_dK partitions ─────────────────────────
            # In RS mode A comes from acc_S/acc_dP register (no smem
            # partition_A); only B is partitioned from smem.
            tdVrdOt = mma_dV.make_fragment_B(
                wg_slice_dV.partition_B(sdOt))
            tdKrQt = mma_dK.make_fragment_B(
                wg_slice_dK.partition_B(sQt))
            if const_expr(not self.mma_dkv_is_rs):
                # SMEM-source path: A=P^T / dS^T → always read via the
                # transposed smem view (regardless of SdP_swapAB).
                tdVrPt = mma_dV.make_fragment_A(
                    wg_slice_dV.partition_A(sPt))
                tdKrdSt = mma_dK.make_fragment_A(
                    wg_slice_dK.partition_A(sdSt))

            # mma_dQ unchanged (smem-source: A=sdS, B=sK^T).
            tdQrdS = mma_dQ.make_fragment_A(wg_slice_dQ.partition_A(sdS))
            tdQrKt = mma_dQ.make_fragment_B(wg_slice_dQ.partition_B(sKt))

            acc_dV = cute.make_fragment(
                mma_dV.partition_shape_C((self.tile_n, self.head_dim)),
                Float32)
            acc_dK = cute.make_fragment(
                mma_dK.partition_shape_C((self.tile_n, self.head_dim)),
                Float32)
            acc_dV.fill(0.0)
            acc_dK.fill(0.0)

            # SwapAB: mma_SdP outputs (tile_n, tile_m), so the C-shape
            # passed to partition_shape_C must match.
            acc_S_shape = mma_SdP.partition_shape_C(
                (self.tile_n, self.tile_m) if self.SdP_swapAB
                else (self.tile_m, self.tile_n))
            acc_dQ_shape = mma_dQ.partition_shape_C(
                (self.tile_m, self.head_dim))

            thr_SdP = mma_SdP.get_slice(tidx)
            thr_dQ = mma_dQ.get_slice(tidx)
            thr_dK = mma_dK.get_slice(tidx)
            thr_dV = mma_dV.get_slice(tidx)

            # ── LSE / dPsum partition into mma_SdP C-frag layout ──
            # FA4 line 1234-1239: instead of N scalar `sLSE[row_local]`
            # smem reads inside the softmax row loop, partition the
            # smem vector once into the per-thread mma C-frag layout.
            # Each thread now owns the rows it'll write in acc_S, so
            # a single batch `load_s2r` per m_iter replaces nrows_S
            # scalar LDS.32 ops.  Targets the long_scoreboard /
            # mio_throttle elevation we saw in NCU.
            #
            # FA4 doc line 119-121: with SdP_swapAB each thread holds
            # tile_m/4 rows of LSE/dPsum in registers — at tile_m=128
            # that's 32 fp32 values per stat = 64 reg total, which
            # blows the 168-reg budget and causes spill (35M LDL/STL
            # in NCU).  shuffle_LSE/shuffle_dPsum cuts each thread to
            # ceil(32/8)=4 rows by spreading the load across 8 quads
            # of 4 threads and broadcasting via shfl.sync within each
            # quad in the softmax row loop.  Activated when SdP_swapAB
            # and tile_hdim<=64.
            shuffle_LSE = self.SdP_swapAB and self.head_dim <= 64
            shuffle_dPsum = self.SdP_swapAB and self.head_dim <= 64
            tLSEsLSE = utils.mma_partition_C_vec(
                sLSE, thr_SdP,
                expand_shape=self.tile_n,
                is_colvec=not self.SdP_swapAB,
            )
            tLSEsdPsum = utils.mma_partition_C_vec(
                sdPsum, thr_SdP,
                expand_shape=self.tile_n,
                is_colvec=not self.SdP_swapAB,
            )
            shfl_copy = utils.tiled_copy_1d(
                sLSE.element_type, num_threads=8, num_copy_elems=2)
            if const_expr(shuffle_LSE):
                tLSEsLSE = (
                    shfl_copy.get_slice(
                        cute.arch.lane_idx() // 4)
                    .partition_S(tLSEsLSE))
                tLSEsLSE = cute.group_modes(tLSEsLSE, 0, 2)
            if const_expr(shuffle_dPsum):
                tLSEsdPsum = (
                    shfl_copy.get_slice(
                        cute.arch.lane_idx() // 4)
                    .partition_S(tLSEsdPsum))
                tLSEsdPsum = cute.group_modes(tLSEsdPsum, 0, 2)
            self._shuffle_LSE = shuffle_LSE
            self._shuffle_dPsum = shuffle_dPsum


            # Identity tensor for tScS — match acc_S layout, then transpose
            # back to (m, n) view so softmax can index as P[m, n].
            cS = cute.make_identity_tensor(
                (self.tile_n, self.tile_m) if self.SdP_swapAB
                else (self.tile_m, self.tile_n))
            tScS = thr_SdP.partition_C(cS)
            tScS_mn = reshape_acc_to_mn(
                tScS, transpose=self.SdP_swapAB)
            # FA4-style mask helper (mask.py:250-359): build the
            # `t0ScS_mn` view from thread 0's slice — its (r, c)
            # coords are constexpr-known per (r, c) and shared by
            # every thread in the warpgroup (the per-thread offset
            # is folded into causal_row_offset once per iter), so
            # the inner-loop cmp becomes one constexpr compare per
            # cell instead of three runtime computes.  Cuts the
            # softmax-phase ALU/FMA we measured at 56 / 53 % over
            # FA4 in ncu (cut2->cut3 = 1.96 ms at tile_m=128).
            thr_SdP_t0 = mma_SdP.get_slice(0)
            t0ScS = thr_SdP_t0.partition_C(cS)
            t0ScS_mn = reshape_acc_to_mn(
                t0ScS, transpose=self.SdP_swapAB)
            # Coord index conventions: after reshape_acc_to_mn,
            # the (m, n) view's underlying tuple is (tile_m, tile_n)
            # without swap and (tile_n, tile_m) with swap (because
            # cS was built with the swapped shape).  So [0]/[1]
            # mean different things in the two paths.
            ROW_IDX = 1 if self.SdP_swapAB else 0
            COL_IDX = 0 if self.SdP_swapAB else 1
            thr_row_offset = tScS_mn[0, 0][ROW_IDX]
            thr_col_offset = tScS_mn[0, 0][COL_IDX]

            cDQ = cute.make_identity_tensor(
                (self.tile_m, self.head_dim))
            tScDQ = thr_dQ.partition_C(cDQ)
            tScDQ_mn = utils.make_acc_mn_view(tScDQ)

            cDK = cute.make_identity_tensor(
                (self.tile_n, self.head_dim))
            tScDK = thr_dK.partition_C(cDK)
            tScDK_mn = utils.make_acc_mn_view(tScDK)

            cDV = cute.make_identity_tensor(
                (self.tile_n, self.head_dim))
            tScDV = thr_dV.partition_C(cDV)
            tScDV_mn = utils.make_acc_mn_view(tScDV)

            nrows_S = cute.size(tScS_mn.shape[0])
            ncols_S = cute.size(tScS_mn.shape[1])
            nrows_dQ = cute.size(tScDQ_mn.shape[0])
            ncols_dQ = cute.size(tScDQ_mn.shape[1])
            nrows_dK = cute.size(tScDK_mn.shape[0])
            ncols_dK = cute.size(tScDK_mn.shape[1])
            nrows_dV = cute.size(tScDV_mn.shape[0])
            ncols_dV = cute.size(tScDV_mn.shape[1])

            # ── Phase 1.7: register→smem copy for dS / P ─────────
            # SwapAB path (mma C is (tile_n, tile_m)): need stmatrix
            # .trans + position-independent partition to land into the
            # row-major (tile_m, tile_n) sdS/sP smem. Use the new
            # `get_smem_store_C` helper (ported quack utilities).
            #
            # No-swap path: keep the simpler original construction
            # (plain stmatrix.x4 atom + make_tiled_copy_C) — bypassing
            # the helper avoids a small (~0.25 ms) overhead we measured
            # on H100 with the same workload.
            if const_expr(self.SdP_swapAB):
                copy_dS_r2s, _, _ = utils.get_smem_store_C(
                    mma_SdP, sdSt, tidx,
                    transpose=True, position_independent=True,
                )
                if const_expr(not self.mma_dkv_is_rs):
                    copy_P_r2s, _, _ = utils.get_smem_store_C(
                        mma_SdP, sPt, tidx,
                        transpose=True, position_independent=True,
                    )
            else:
                # D-3: partition_D the full (..., PdS_stage) tensor up
                # front, then slice the runtime stage index per iter.
                # (Original code did `partition_D(sP[None, None, 0])` —
                # a constexpr stage 0 slice that worked but blocked
                # PdS_stage>1 from rotating; with dynamic stage idx we
                # must partition the 4D layout once and select stage
                # last, matching FA4/quack convention.)
                smem_copy_atom = utils.get_smem_store_atom_sm90(self.dtype)
                smem_thr_copy_dS = cute.make_tiled_copy_C(
                    smem_copy_atom, mma_SdP).get_slice(tidx)
                tRS_sdS = smem_thr_copy_dS.partition_D(sdS)
                if const_expr(not self.mma_dkv_is_rs):
                    smem_thr_copy_P = cute.make_tiled_copy_C(
                        smem_copy_atom, mma_SdP).get_slice(tidx)
                    tRS_sP = smem_thr_copy_P.partition_D(sP)

            # ── FA4 flat dQ R2S TV layout (mainloop_bwd_sm90:1252) ──
            # `make_tiled_copy_tv((128, num_wg_dQ), val=4)` gives
            # each thread a CONTIGUOUS 4-fp32 (128-bit) STS into a
            # (slot_size, num_wg_dQ) flat smem buffer.  Threads in
            # the same warp hit 32 distinct banks → no conflict.
            # Postprocess uses the same TV layout to S2R back into
            # WGMMA-frag register order, then rebroadcasts via
            # tiled_mma's partition_C to recover the (m, d) view.
            #
            # Pre-refactor: `make_tiled_copy_C(atom, mma_dQ)` over a
            # row-major (slot_m, slot_d) buffer gave WGMMA-frag
            # scattered writes → 11.6-way bank conflict (NCU
            # measured 51.8 % of 12.1 M STS).
            universal_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32, num_bits_per_copy=128)
            r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
                universal_atom,
                cute.make_layout(
                    (128, self.num_wg_dQ)),
                cute.make_layout(128 // Float32.width),
            )
            r2s_thr_copy_dQaccum = (
                r2s_tiled_copy_dQaccum.get_slice(tidx))

            # Hoist invariant partition_D / make_tensor for dQ R2S.
            # sdQacc layout: (slot_size, num_wg_dQ, stage).  One
            # `partition_D(sdQacc[..., dq_stage])` slice gives a
            # (slot_size, num_wg_dQ) buffer; the flat TV maps each
            # thread to its 4-cell consecutive run.
            sdQacc_slot_const = sdQacc[None, None, 0]
            tdQsdQaccum_const = r2s_thr_copy_dQaccum.partition_D(
                sdQacc_slot_const)
            tdQ_layout_const = cute.make_layout(
                tdQsdQaccum_const.shape)

            q_state = fa_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Consumer, self.Q_stage)
            do_state = fa_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Consumer, self.dO_stage)

            # Phase 1.5: monotonic dq iter counter (replaces dq_empty_phase).
            # stage = dq_iter_idx % sdQacc_stage
            # phase = (dq_iter_idx // sdQacc_stage) & 1
            # Double-buffers consumer ↔ warp 1 so iter K can R2S into
            # stage K%S while warp 1 still drains stage (K-1)%S.
            dq_iter_idx = Int32(0)

            # FA4-grid path: each CTA owns 1 H_q tile (no gqa loop).
            # Keep an outer 1-iter loop to preserve indentation of
            # the m_loop body — the compiler folds it away.
            for gqa_idx in cutlass.range_constexpr(1):
                if const_expr(self.is_causal):
                    m_block_min = n_start // self.tile_m
                else:
                    m_block_min = Int32(0)
                m_block_max = m_blocks_total

                for m_idx in cutlass.range(
                    m_block_max - m_block_min, unroll=1
                ):
                    m_block = m_block_min + m_idx
                    m_start = m_block * self.tile_m

                    # Diff-#6: dispersed consumer_wait — wait(Q) →
                    # fire GEMM1 → wait(dO) → fire GEMM2.
                    pipeline_Q.consumer_wait(q_state)

                    smem_idx_PdS = (
                        q_state.index
                        if const_expr(self.PdS_stage > 1) else Int32(0)
                    )

                    # ── GEMM1: SwapAB → S^T = K @ Q^T ───────────────
                    acc_S = cute.make_fragment(acc_S_shape, Float32)
                    if const_expr(self.SdP_swapAB):
                        hopper_helpers.wgmma_gemm(
                            mma_SdP, acc_S,
                            tSrK,
                            tSrQ[None, None, None, q_state.index],
                            zero_init=True, wg_wait=-1,
                        )
                    else:
                        hopper_helpers.wgmma_gemm(
                            mma_SdP, acc_S,
                            tSrQ[None, None, None, q_state.index],
                            tSrK,
                            zero_init=True, wg_wait=-1,
                        )

                    pipeline_dO.consumer_wait(do_state)

                    # ────────────────────────────────────────────────
                    # Cutpoint 1: drain GEMM1 only, release, advance.
                    # The wait(dO) above is needed even at cut 1 so
                    # the producer can advance dO state at next iter.
                    # ────────────────────────────────────────────────
                    if const_expr(self.early_exit_after_gemm == 1):
                        warpgroup.wait_group(0)
                        pipeline_dO.consumer_release(do_state)
                        pipeline_Q.consumer_release(q_state)
                    else:
                        # ── GEMM2: SwapAB → dP^T = V @ dO^T ─────
                        acc_dP = cute.make_fragment(acc_S_shape, Float32)
                        if const_expr(self.SdP_swapAB):
                            hopper_helpers.wgmma_gemm(
                                mma_SdP, acc_dP,
                                tSrV,
                                tSrdO[None, None, None, do_state.index],
                                zero_init=True, wg_wait=1,
                            )
                        else:
                            hopper_helpers.wgmma_gemm(
                                mma_SdP, acc_dP,
                                tSrdO[None, None, None, do_state.index],
                                tSrV,
                                zero_init=True, wg_wait=1,
                            )

                        if const_expr(self.early_exit_after_gemm == 2):
                            warpgroup.wait_group(0)
                            pipeline_dO.consumer_release(do_state)
                            pipeline_Q.consumer_release(q_state)
                        else:
                            # softmax phase 1
                            acc_S_mn = reshape_acc_to_mn(
                                acc_S, transpose=self.SdP_swapAB)
                            # Batch register-load of LSE / dPsum via
                            # the partitioned mma C-frag view (FA4
                            # line 1497 / 1522).  One LDS per thread
                            # replaces nrows_S scalar smem reads.
                            rLSE = utils.load_s2r(
                                tLSEsLSE[None, q_state.index])
                            rDPsum = utils.load_s2r(
                                tLSEsdPsum[None, do_state.index])
                            # FA4-style causal mask pre-pass
                            # (mask.py:251-275 swap_AB branch).
                            # c-then-r loop; constexpr-known
                            # `t0ScS_mn[r, 0][ROW_IDX]` (per r) and
                            # `t0ScS_mn[0, c][COL_IDX]` (per c) plus
                            # one runtime scalar `causal_row_offset`
                            # (per iter).  Inner cmp uses two
                            # registers — no per-cell FMA for
                            # row_global / col_global.  Saves ~50 %
                            # of mask ALU vs the previous fused r-c
                            # loop (256 → 64 ops/thread/iter at
                            # tile_m=128).
                            NEG_INF = Float32(-1e30)
                            lane_idx = cute.arch.lane_idx()
                            # FA4 mask skip: with tile_m == tile_n,
                            # only the diagonal m_block (m_idx == 0
                            # since m_block_min = n_block in causal)
                            # contains masked cells.  All later
                            # m_idx have m_start >= n_start +
                            # tile_n so every cell is past the
                            # causal cutoff.  Skipping the mask
                            # pre-pass here saves ~42 M ALU /
                            # warp-iter (1 cmp / cell × tile_m *
                            # tile_n).  ~40 % of total ALU on the
                            # critical path.
                            if const_expr(self.is_causal):
                                if m_idx == 0:
                                    causal_row_offset = (
                                        m_start - n_start
                                        + thr_row_offset
                                        - thr_col_offset)
                                    for c in cutlass.range_constexpr(
                                        ncols_S
                                    ):
                                        col0 = t0ScS_mn[0, c][COL_IDX]
                                        row_limit_top = (
                                            col0 - causal_row_offset)
                                        for r in cutlass.range_constexpr(
                                            nrows_S
                                        ):
                                            t0_row = (
                                                t0ScS_mn[r, 0][ROW_IDX])
                                            if t0_row < row_limit_top:
                                                acc_S_mn[r, c] = NEG_INF

                            # Softmax phase 1 — vector-store row.
                            for r in cutlass.range_constexpr(nrows_S):
                                lse_val = utils.get_stat(
                                    rLSE, r, lane_idx,
                                    shuffle=self._shuffle_LSE)
                                acc_S_mn[r, None].store(
                                    cute.math.exp2(
                                        acc_S_mn[r, None].load()
                                        * scale_log2
                                        - lse_val,
                                        fastmath=True,
                                    )
                                )

                            # softmax phase 2 — vector-store row.
                            warpgroup.wait_group(0)
                            acc_dP_mn = reshape_acc_to_mn(
                                acc_dP, transpose=self.SdP_swapAB)
                            for r in cutlass.range_constexpr(nrows_S):
                                dpsum_val = utils.get_stat(
                                    rDPsum, r, lane_idx,
                                    shuffle=self._shuffle_dPsum)
                                acc_dP_mn[r, None].store(
                                    acc_S_mn[r, None].load()
                                    * (acc_dP_mn[r, None].load()
                                       - dpsum_val)
                                )

                            # cvt + R2S(dS)
                            if const_expr(self.mma_dkv_is_rs):
                                tdVrP = utils.cvt_f16(
                                    reshape_acc_to_frgA(acc_S),
                                    self.dtype)
                                tdKrdS = utils.cvt_f16(
                                    reshape_acc_to_frgA(acc_dP),
                                    self.dtype)
                            if const_expr(self.SdP_swapAB):
                                if const_expr(not self.mma_dkv_is_rs):
                                    rP = utils.cvt_f16(
                                        acc_S, self.dtype)
                                    copy_P_r2s(rP, dst_idx=smem_idx_PdS)
                                if const_expr(self.mma_dkv_is_rs):
                                    # FA4 line 1545+1558: reuse the
                                    # frgA-shaped FP16 fragment we
                                    # just cvt'd for the GEMM5 dK RS
                                    # source — copy_dS_r2s retiles
                                    # internally, so an extra
                                    # cvt(acc_dP→FP16) + register
                                    # spill is just dead work.
                                    # Saves ~8192 cvt PTX ops + ~16
                                    # registers per consumer WG on the
                                    # cut2→cut3 critical path.
                                    copy_dS_r2s(
                                        tdKrdS, dst_idx=smem_idx_PdS)
                                else:
                                    rDS = utils.cvt_f16(
                                        acc_dP, self.dtype)
                                    copy_dS_r2s(
                                        rDS, dst_idx=smem_idx_PdS)
                            else:
                                if const_expr(not self.mma_dkv_is_rs):
                                    rP = cute.make_fragment_like(
                                        acc_S, self.dtype)
                                    rP.store(
                                        acc_S.load().to(self.dtype))
                                    taccPrP = smem_thr_copy_P.retile(rP)
                                    cute.copy(
                                        smem_copy_atom, taccPrP,
                                        tRS_sP[
                                            None, None, None,
                                            smem_idx_PdS])
                                rDS = cute.make_fragment_like(
                                    acc_dP, self.dtype)
                                rDS.store(acc_dP.load().to(self.dtype))
                                taccDSrDS = (
                                    smem_thr_copy_dS.retile(rDS))
                                cute.copy(
                                    smem_copy_atom, taccDSrDS,
                                    tRS_sdS[
                                        None, None, None, smem_idx_PdS])

                            # ── Action A: skip first PdS_barrier in
                            # RS + PdS_stage>=2 mode ───────────────
                            # FA4 line 1553 gating:
                            #   need_first_barrier = !mma_dkv_is_rs OR
                            #                        (mma_dkv_is_rs &
                            #                         PdS_stage == 1)
                            # When the gate is False, R2S(dS) and
                            # GEMM3 dV proceed without a 256-thread
                            # rendezvous — GEMM3 doesn't read sdS
                            # (RS A) and PdS double-buffer keeps prev
                            # iter's sdS un-overwritten.  Saves the
                            # full barrier-arrival + LSE-stall cycle.
                            if const_expr(
                                (not self.mma_dkv_is_rs)
                                or (self.PdS_stage == 1
                                    and self.mma_dkv_is_rs)
                            ):
                                cute.arch.fence_proxy(
                                    cute.arch.ProxyKind.async_shared,
                                    space=(
                                        cute.arch.SharedSpace
                                        .shared_cta),
                                )
                                cute.arch.barrier(
                                    barrier_id=self.BARRIER_PDS,
                                    number_of_threads=256,
                                )

                            # GEMM3 (dV)
                            if const_expr(self.mma_dkv_is_rs):
                                hopper_helpers.wgmma_gemm(
                                    mma_dV, acc_dV,
                                    tdVrP,
                                    tdVrdOt[
                                        None, None, None,
                                        do_state.index],
                                    zero_init=False, wg_wait=-1,
                                )
                            else:
                                hopper_helpers.wgmma_gemm(
                                    mma_dV, acc_dV,
                                    tdVrPt[
                                        None, None, None,
                                        smem_idx_PdS],
                                    tdVrdOt[
                                        None, None, None,
                                        do_state.index],
                                    zero_init=False, wg_wait=-1,
                                )

                            if const_expr(self.early_exit_after_gemm
                                          == 3):
                                warpgroup.wait_group(0)
                                pipeline_dO.consumer_release(do_state)
                                pipeline_Q.consumer_release(q_state)
                            else:
                                # FA4 line 1569-1570: second PdS_barrier
                                # — needed regardless of RS mode because
                                # GEMM4 dQ reads sdS smem.
                                cute.arch.fence_proxy(
                                    cute.arch.ProxyKind.async_shared,
                                    space=(
                                        cute.arch.SharedSpace
                                        .shared_cta),
                                )
                                cute.arch.barrier(
                                    barrier_id=self.BARRIER_PDS,
                                    number_of_threads=256,
                                )
                                # is_dQ_wg dispatch
                                if const_expr(self.dQ_single_wg):
                                    is_dQ_wg = wg_idx == Int32(0)
                                else:
                                    is_dQ_wg = cutlass.Boolean(True)

                                if is_dQ_wg:
                                    # GEMM4 (dQ)
                                    acc_dQ = cute.make_fragment(
                                        acc_dQ_shape, Float32)
                                    hopper_helpers.wgmma_gemm(
                                        mma_dQ, acc_dQ,
                                        tdQrdS[
                                            None, None, None,
                                            smem_idx_PdS],
                                        tdQrKt,
                                        zero_init=True, wg_wait=1,
                                    )

                                    if const_expr(
                                        self.early_exit_after_gemm == 4
                                    ):
                                        warpgroup.wait_group(0)
                                        pipeline_dO.consumer_release(
                                            do_state)
                                        pipeline_Q.consumer_release(
                                            q_state)
                                    else:
                                        pipeline_dO.consumer_release(
                                            do_state)
                                        # GEMM5 (dK)
                                        if const_expr(
                                            self.mma_dkv_is_rs
                                        ):
                                            hopper_helpers.wgmma_gemm(
                                                mma_dK, acc_dK,
                                                tdKrdS,
                                                tdKrQt[
                                                    None, None, None,
                                                    q_state.index],
                                                zero_init=False,
                                                wg_wait=1,
                                            )
                                        else:
                                            hopper_helpers.wgmma_gemm(
                                                mma_dK, acc_dK,
                                                tdKrdSt[
                                                    None, None, None,
                                                    smem_idx_PdS],
                                                tdKrQt[
                                                    None, None, None,
                                                    q_state.index],
                                                zero_init=False,
                                                wg_wait=1,
                                            )

                                        if const_expr(
                                            self.early_exit_after_gemm
                                            == 5
                                        ):
                                            warpgroup.wait_group(0)
                                            pipeline_Q.consumer_release(
                                                q_state)
                                        else:
                                            # ── full path: dQ R2S + sync ──
                                            # Named-barrier path (FA4
                                            # line 1587-1599) for the
                                            # single-buffer single-WG
                                            # dQ pipeline: consumer
                                            # arrive+wait on EMPTY (warp
                                            # 1 must have released the
                                            # slot), R2S, fence, then
                                            # arrive-only on FULL
                                            # (signal warp 1 to TMA-store).
                                            # Eliminates the per-iter
                                            # mbarrier phase tracking
                                            # overhead — bar.sync is
                                            # one cycle vs ~10 for
                                            # mbarrier.try_wait.
                                            # FA4 flat TV: partition_D
                                            # over (slot_size,
                                            # num_wg_dQ) maps each WG's
                                            # threads to its own column
                                            # automatically; no manual
                                            # per-WG slicing needed.
                                            if const_expr(
                                                self.use_named_dq_barrier
                                                and self.dQ_single_wg
                                                and self.sdQacc_stage == 1
                                            ):
                                                cute.arch.barrier(
                                                    barrier_id=(
                                                        self.BARRIER_DQ_EMPTY_BASE),
                                                    number_of_threads=(
                                                        self
                                                        .num_dq_sync_threads),
                                                )
                                                tdQsdQaccum_iter = (
                                                    tdQsdQaccum_const)
                                            elif const_expr(
                                                self.use_named_dq_barrier
                                                and self._dQ_M_split
                                                and self.sdQacc_stage == 1
                                            ):
                                                # FA4 line 1587: wait
                                                # EMPTY[wg_idx] — warp
                                                # 1 must have released
                                                # this WG's slot before
                                                # we start R2S.
                                                cute.arch.barrier(
                                                    barrier_id=(
                                                        self.BARRIER_DQ_EMPTY_BASE
                                                        + wg_idx),
                                                    number_of_threads=(
                                                        self
                                                        .num_dq_sync_threads),
                                                )
                                                tdQsdQaccum_iter = (
                                                    tdQsdQaccum_const)
                                            else:
                                                dq_stage = (
                                                    dq_iter_idx
                                                    % self.sdQacc_stage)
                                                dq_phase = (
                                                    (dq_iter_idx
                                                     // self.sdQacc_stage)
                                                    & Int32(1))
                                                wg_idx_dQ = wg_idx
                                                # mbar_full / empty
                                                # still per-(wg, stage)
                                                # — each WG signals its
                                                # own done-flag so warp
                                                # 1 can pipeline TMAs.
                                                mbar_idx = (
                                                    wg_idx_dQ
                                                    + self.num_wg_dQ
                                                    * dq_stage)
                                                cute.arch.mbarrier_wait(
                                                    mbar_dq_empty_ptr
                                                    + mbar_idx, dq_phase)
                                                sdQacc_slot = sdQacc[
                                                    None, None,
                                                    dq_stage]
                                                tdQsdQaccum_iter = (
                                                    r2s_thr_copy_dQaccum
                                                    .partition_D(
                                                        sdQacc_slot))
                                            tdQrdQaccum_flat = (
                                                cute.make_tensor(
                                                    acc_dQ.iterator,
                                                    tdQ_layout_const))
                                            cute.autovec_copy(
                                                tdQrdQaccum_flat,
                                                tdQsdQaccum_iter)
                                            cute.arch.fence_proxy(
                                                cute.arch.ProxyKind
                                                .async_shared,
                                                space=cute.arch
                                                .SharedSpace
                                                .shared_cta,
                                            )
                                            if const_expr(
                                                self.use_named_dq_barrier
                                                and self.dQ_single_wg
                                                and self.sdQacc_stage == 1
                                            ):
                                                cute.arch.barrier_arrive(
                                                    barrier_id=(
                                                        self.BARRIER_DQ_FULL_BASE),
                                                    number_of_threads=(
                                                        self
                                                        .num_dq_sync_threads),
                                                )
                                            elif const_expr(
                                                self.use_named_dq_barrier
                                                and self._dQ_M_split
                                                and self.sdQacc_stage == 1
                                            ):
                                                # FA4 line 1599: arrive
                                                # FULL[wg_idx] to release
                                                # warp 1's TMA on this
                                                # WG's slot.
                                                cute.arch.barrier_arrive(
                                                    barrier_id=(
                                                        self.BARRIER_DQ_FULL_BASE
                                                        + wg_idx),
                                                    number_of_threads=(
                                                        self
                                                        .num_dq_sync_threads),
                                                )
                                            else:
                                                # Legacy mbarrier path:
                                                # each WG arrives its
                                                # own mbar_full[wg,
                                                # stage] (mbar_idx
                                                # carries wg).
                                                cute.arch.mbarrier_arrive(
                                                    mbar_dq_full_ptr
                                                    + mbar_idx)
                                            dq_iter_idx = (
                                                dq_iter_idx
                                                + Int32(1))
                                            warpgroup.wait_group(0)
                                            pipeline_Q.consumer_release(
                                                q_state)
                                else:
                                    # not is_dQ_wg (WG1 in dQ_single_wg)
                                    if const_expr(
                                        self.early_exit_after_gemm == 4
                                    ):
                                        warpgroup.wait_group(0)
                                        pipeline_dO.consumer_release(
                                            do_state)
                                        pipeline_Q.consumer_release(
                                            q_state)
                                    else:
                                        if const_expr(
                                            self.mma_dkv_is_rs
                                        ):
                                            hopper_helpers.wgmma_gemm(
                                                mma_dK, acc_dK,
                                                tdKrdS,
                                                tdKrQt[
                                                    None, None, None,
                                                    q_state.index],
                                                zero_init=False,
                                                wg_wait=1,
                                            )
                                        else:
                                            hopper_helpers.wgmma_gemm(
                                                mma_dK, acc_dK,
                                                tdKrdSt[
                                                    None, None, None,
                                                    smem_idx_PdS],
                                                tdKrQt[
                                                    None, None, None,
                                                    q_state.index],
                                                zero_init=False,
                                                wg_wait=1,
                                            )
                                        if const_expr(
                                            self.early_exit_after_gemm
                                            == 5
                                        ):
                                            warpgroup.wait_group(0)
                                            pipeline_dO.consumer_release(
                                                do_state)
                                            pipeline_Q.consumer_release(
                                                q_state)
                                        else:
                                            pipeline_dO.consumer_release(
                                                do_state)
                                            warpgroup.wait_group(0)
                                            pipeline_Q.consumer_release(
                                                q_state)

                    q_state.advance()
                    do_state.advance()

            # ── Epilogue: dK*scale + dV store to global ──
            self._epilogue_dKV(
                acc_dK, acc_dV,
                mdK, mdV,
                mma_dK, mma_dV,
                tScDK_mn, tScDV_mn,
                nrows_dK, ncols_dK, nrows_dV, ncols_dV,
                n_block, kv_head_idx, batch_idx,
                softmax_scale,
                sP, sdS, tidx,
            )

        # ── Warp 1: dQaccum store via mbarrier signal ──

        @cute.jit
        def dQaccum_store(
            self,
            mdQaccum: cute.Tensor,
            sdQacc: cute.Tensor,
            mbar_dq_full_ptr: cute.Pointer,
            mbar_dq_empty_ptr: cute.Pointer,
            n_block: Int32, h_q_idx: Int32, batch_idx: Int32,
            m_blocks_total: Int32,
        ):
            """FA4-style flat dQ store via raw cp.reduce.async.bulk.add.

            mdQaccum is 1D-per-(B, H_q) flat FP32 buffer.  sdQacc is
            (slot_size, num_wg_dQ, stage) 2D-per-stage flat.  Per
            iter we issue `num_wg_dQ` independent `slot_size *
            sizeof(fp32)`-byte cp.reduce ops, one per WG slot.
            """
            # Cutpoint instrumentation: when early_exit_after_gemm > 0
            # the consumer skips dQ R2S entirely, so warp 1 has no work
            # and would otherwise hang on the first mbarrier_wait.
            if const_expr(self.early_exit_after_gemm != 0):
                return  # cute.jit accepts early return when constexpr-true.

            # PDL: warp 1 reads dq_accum via cp.reduce.async.bulk.add
            # (RMW); must wait for prep to finish 0-init before the
            # first reduce-add.
            cute.arch.griddepcontrol_wait()

            n_start = n_block * self.tile_n
            # FA4 flat dQ store path (mainloop_bwd_sm90:1872+).
            # Each m_block writes (tile_m * D) FP32 = 32 KB total
            # to gmem, organized as (slot_size, num_wg_dQ) flat.
            # Per WG slot = slot_size * 4 = 16 KB.  Issued via raw
            # cp.reduce.async.bulk.add (no TMA descriptor).
            slot_size = self.tile_m * self.head_dim // self.num_wg_dQ
            per_wg_bytes = slot_size * 4

            # Sync init.  Two paths:
            #   (1) named-barrier M-split (sdQacc_stage=1): pre-arrive
            #       EMPTY[wg] for each WG so consumer iter 0 sees the
            #       slot as immediately empty (FA4 mainloop_bwd_sm90
            #       line 1845 barrier_arrive once before loop).
            #   (2) legacy mbarrier (any stage): pre-arrive every
            #       (wg, stage) mbar_empty so consumer iter K=0..S-1
            #       wait (phase=0) returns immediately.
            use_named_path = (
                self.use_named_dq_barrier
                and self._dQ_M_split
                and self.sdQacc_stage == 1)
            if const_expr(use_named_path):
                for wg in cutlass.range_constexpr(self.num_wg_dQ):
                    cute.arch.barrier_arrive(
                        barrier_id=(
                            self.BARRIER_DQ_EMPTY_BASE + wg),
                        number_of_threads=self.num_dq_sync_threads,
                    )
            else:
                for s in cutlass.range_constexpr(self.sdQacc_stage):
                    for wg in cutlass.range_constexpr(self.num_wg_dQ):
                        idx = wg + self.num_wg_dQ * s
                        cute.arch.mbarrier_arrive(
                            mbar_dq_empty_ptr + idx)

            dq_iter_idx = Int32(0)

            if const_expr(self.is_causal):
                m_block_min = n_start // self.tile_m
            else:
                m_block_min = Int32(0)
            m_block_max = m_blocks_total

            for gqa_idx in cutlass.range_constexpr(1):
                # Slice mdQaccum[reordered (N*D, H_q, B)] to 1D
                # `(N*D,)` for this (h_q, batch).  Reshape to
                # 2D (slot_size, num_wg_dQ * n_blocks) so we can
                # address (None, wg + num_wg * m_block) → 1D
                # (slot_size,) per (m_block, wg) slot.
                gdQ_h_1d = mdQaccum[None, h_q_idx, batch_idx]
                # gdQ_h_1d shape: (N*D,) — base ptr + element index.
                # Reshape to (slot_size, num_wg_dQ, n_blocks)
                # column-major: stride (1, slot_size, slot_size *
                # num_wg) so consecutive bytes match
                # main-kernel cp.reduce-add expectations.
                n_blocks_local = (
                    cute.size(gdQ_h_1d.shape) // (
                        slot_size * self.num_wg_dQ))
                gdQ_tiled = cute.make_tensor(
                    gdQ_h_1d.iterator,
                    cute.make_layout(
                        (slot_size, self.num_wg_dQ, n_blocks_local),
                        stride=(
                            1,
                            slot_size,
                            slot_size * self.num_wg_dQ,
                        ),
                    ),
                )

                num_dQ_chunks = self.num_wg_dQ
                for m_idx in cutlass.range(
                    m_block_max - m_block_min, unroll=1
                ):
                    m_block = m_block_min + m_idx
                    if const_expr(use_named_path):
                        # FA4-pipelined named-barrier path
                        # (mainloop_bwd_sm90 line 1840-1869).  Three
                        # phases:
                        #   1. For each prior WG[i], wait until at most
                        #      `num_dQ_chunks - 1 - i` cp.async groups
                        #      are still in flight from older iters.
                        #      Then `barrier_arrive(EMPTY[i])` to
                        #      release consumer WG[i] — it can now
                        #      R2S into slot i because warp 1 has
                        #      finished reading slot i.
                        #   2. For each WG, `barrier(FULL[wg])` wait
                        #      for consumer R2S done, then issue the
                        #      cp.reduce.async.bulk + commit_group.
                        # Replaces the legacy mbarrier double-buffer
                        # — bar.sync is 1 cycle vs ~10 for
                        # mbarrier.try_wait, and per-iter cp.async
                        # ops can pipeline across iters (up to
                        # `num_dQ_chunks - 1` in flight).
                        for wg_i in cutlass.range_constexpr(
                            num_dQ_chunks
                        ):
                            cute.arch.cp_async_bulk_wait_group(
                                num_dQ_chunks - 1 - wg_i,
                                read=True,
                            )
                            cute.arch.barrier_arrive(
                                barrier_id=(
                                    self.BARRIER_DQ_EMPTY_BASE
                                    + wg_i),
                                number_of_threads=(
                                    self.num_dq_sync_threads),
                            )
                        for wg_i in cutlass.range_constexpr(
                            num_dQ_chunks
                        ):
                            cute.arch.barrier(
                                barrier_id=(
                                    self.BARRIER_DQ_FULL_BASE
                                    + wg_i),
                                number_of_threads=(
                                    self.num_dq_sync_threads),
                            )
                            sdQacc_slot = sdQacc[None, wg_i, 0]
                            gdQ_slot = gdQ_tiled[None, wg_i, m_block]
                            with cute.arch.elect_one():
                                utils.cpasync_reduce_bulk_add_f32(
                                    sdQacc_slot.iterator,
                                    gdQ_slot.iterator,
                                    per_wg_bytes,
                                )
                            cute.arch.cp_async_bulk_commit_group()
                    else:
                        # Issue all per-wg cp.reduce.async.bulk for
                        # this m_block back-to-back, then drain
                        # ONCE.  Saves the inner per-wg
                        # `cp_async_bulk_wait_group(0)` between WG
                        # slots (~1 LSU drain saved per m_block).
                        # FA4 line 1840+ pipelines further across
                        # m_block iters (named-barrier path) — that
                        # was attempted and hung on this devspace
                        # so we keep mbarrier double-buffer here.
                        dq_stage = dq_iter_idx % self.sdQacc_stage
                        dq_phase = (
                            (dq_iter_idx // self.sdQacc_stage)
                            & Int32(1))
                        for wg in cutlass.range_constexpr(
                            num_dQ_chunks
                        ):
                            mbar_idx = wg + num_dQ_chunks * dq_stage
                            cute.arch.mbarrier_wait(
                                mbar_dq_full_ptr + mbar_idx,
                                dq_phase)
                            sdQacc_slot = sdQacc[None, wg, dq_stage]
                            gdQ_slot = gdQ_tiled[None, wg, m_block]
                            with cute.arch.elect_one():
                                utils.cpasync_reduce_bulk_add_f32(
                                    sdQacc_slot.iterator,
                                    gdQ_slot.iterator,
                                    per_wg_bytes,
                                )
                            cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            0, read=True)
                        cute.arch.sync_warp()
                        for wg in cutlass.range_constexpr(
                            num_dQ_chunks
                        ):
                            mbar_idx = wg + num_dQ_chunks * dq_stage
                            cute.arch.mbarrier_arrive(
                                mbar_dq_empty_ptr + mbar_idx)

                    dq_iter_idx = dq_iter_idx + Int32(1)

                # Drain remaining cp.async ops at end of loop so
                # warp 1 doesn't return while in-flight bulk reduce-
                # adds are still reading sdQacc smem.
                if const_expr(use_named_path):
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

            # PDL: signal post-dQ can start its prologue as soon as
            # warp 1 is done writing dq_accum.  Warp 1 typically
            # finishes ~30 µs ahead of the consumer's dKV epilogue
            # (consumer then runs cp.reduce.async.bulk for dk/dv,
            # which post-dual reads — but post-dual gates on
            # post-dQ's launch_dependents, not main's directly, and
            # stream serialisation enforces dk/dv visibility before
            # post-dual begins reading).  Net: post-dQ prologue
            # can overlap with consumer's dKV epilogue.
            cute.arch.griddepcontrol_launch_dependents()

        # ── dKV epilogue: cp.reduce.async.bulk.add (FP32 accum) ──
        #
        # FA4-grid path (line 1694-1769): multiple CTAs share the
        # same (n_block, h_kv, batch) tile of dK/dV (one CTA per
        # h_q).  Use cp.reduce.async.bulk.add to FP32 accum buffers
        # in HBM (mdKaccum / mdVaccum); a separate post kernel casts
        # to FP16.
        #
        # Layout: alias sP/sdS smem as FP32 (tile_n, D/num_wg_mma,
        # num_wg_mma) chunks, R2S acc_dK/acc_dV per WG, then warp 4
        # issues per-WG bulk-reduce-add.

        @cute.jit
        def _epilogue_dKV(
            self,
            acc_dK: cute.Tensor, acc_dV: cute.Tensor,
            mdK: cute.Tensor, mdV: cute.Tensor,
            mma_dK: cute.TiledMma, mma_dV: cute.TiledMma,
            tScDK_mn: cute.Tensor, tScDV_mn: cute.Tensor,
            nrows_dK: cutlass.Constexpr, ncols_dK: cutlass.Constexpr,
            nrows_dV: cutlass.Constexpr, ncols_dV: cutlass.Constexpr,
            n_block: Int32, kv_head_idx: Int32, batch_idx: Int32,
            softmax_scale: Float32,
            sP: cute.Tensor, sdS: cute.Tensor,
            tidx: Int32,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            D = self.head_dim
            chunk_size = self.tile_n * D // self.num_wg_mma  # FP32 elem
            chunk_bytes = chunk_size * 4

            # ── alias sP / sdS as FP32 sdKaccum / sdVaccum ───────
            # Layout (chunk_size, num_wg_mma) col-major — matches
            # FA4 line 1698.  Both alias same underlying smem (sP +
            # sdS together = 32 KB which fits both fp32 chunks).
            # Alias sdS smem region as FP32 buffer for both sdK/sdV
            # epilogue.  When mma_dkv_is_rs=True, sP smem is 0
            # bytes so we cannot use it.  sdS is 64 KB at tile_m=128
            # which is ample for one (tile_n, D)=128*64 FP32 = 32 KB
            # tile.  sdK and sdV epilogue run sequentially (TMA
            # bulk-reduce wait between them) so they may share smem.
            alias_layout = cute.make_layout(
                (chunk_size, self.num_wg_mma),
                stride=(1, chunk_size),
            )
            sdS_fp32_ptr = cute.recast_ptr(sdS.iterator, dtype=Float32)
            sdKaccum = cute.make_tensor(sdS_fp32_ptr, alias_layout)
            sdVaccum = cute.make_tensor(sdS_fp32_ptr, alias_layout)

            sdKaccum_2d = cute.make_tensor(
                sdS_fp32_ptr,
                cute.make_layout(
                    (self.tile_n, D), stride=(D, 1)),
            )
            sdVaccum_2d = cute.make_tensor(
                sdS_fp32_ptr,
                cute.make_layout(
                    (self.tile_n, D), stride=(D, 1)),
            )

            universal_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32, num_bits_per_copy=128)
            r2s_tiled_dK = cute.make_tiled_copy_C(universal_atom, mma_dK)
            r2s_tiled_dV = cute.make_tiled_copy_C(universal_atom, mma_dV)
            r2s_thr_dK = r2s_tiled_dK.get_slice(tidx)
            r2s_thr_dV = r2s_tiled_dV.get_slice(tidx)

            # No scale apply here — postprocess kernel multiplies by
            # softmax_scale on FP32 → FP16 cast.  Same for acc_dV
            # (post scale = 1.0).

            tdKsDK = r2s_thr_dK.partition_D(sdKaccum_2d)
            tdKrDK = cute.make_tensor(
                acc_dK.iterator,
                cute.make_layout(tdKsDK.shape))
            cute.autovec_copy(tdKrDK, tdKsDK)

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.barrier(
                barrier_id=self.BARRIER_PDS,
                number_of_threads=self.num_mma_threads,
            )

            # ── warp 4 lane 0 issues cp.reduce.bulk.add f32 (per WG)
            # FA4-grid: mdK is (B, H_kv, N, D) FP32 accum buffer.
            gdK_h = mdK[None, None, kv_head_idx, batch_idx]
            gdK_tile = cute.local_tile(
                gdK_h, (self.tile_n, D), (n_block, 0))
            if warp_idx == 4:
                with cute.arch.elect_one():
                    rows_per_wg = self.tile_n // self.num_wg_mma
                    for wg in cutlass.range_constexpr(self.num_wg_mma):
                        smem_chunk = sdKaccum[None, wg]
                        gmem_chunk = utils.elem_pointer(
                            gdK_tile, (wg * rows_per_wg, 0))
                        utils.cpasync_reduce_bulk_add_f32(
                            smem_chunk.iterator,
                            gmem_chunk,
                            chunk_bytes,
                        )
                    cute.arch.cp_async_bulk_commit_group()

            # sdK and sdV alias same sdS smem.  Wait for dK
            # cp.reduce.bulk.add to finish reading sdS before R2S
            # overwrites it with sdV.
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(
                barrier_id=self.BARRIER_PDS,
                number_of_threads=self.num_mma_threads,
            )

            # ── R2S acc_dV → sdV + reduce-bulk to mdV ────────────
            tdVsDV = r2s_thr_dV.partition_D(sdVaccum_2d)
            tdVrDV = cute.make_tensor(
                acc_dV.iterator,
                cute.make_layout(tdVsDV.shape))
            cute.autovec_copy(tdVrDV, tdVsDV)

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            cute.arch.barrier(
                barrier_id=self.BARRIER_PDS,
                number_of_threads=self.num_mma_threads,
            )

            gdV_h = mdV[None, None, kv_head_idx, batch_idx]
            gdV_tile = cute.local_tile(
                gdV_h, (self.tile_n, D), (n_block, 0))
            if warp_idx == 4:
                with cute.arch.elect_one():
                    rows_per_wg = self.tile_n // self.num_wg_mma
                    for wg in cutlass.range_constexpr(self.num_wg_mma):
                        smem_chunk = sdVaccum[None, wg]
                        gmem_chunk = utils.elem_pointer(
                            gdV_tile, (wg * rows_per_wg, 0))
                        utils.cpasync_reduce_bulk_add_f32(
                            smem_chunk.iterator,
                            gmem_chunk,
                            chunk_bytes,
                        )
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    # ─────────────────────────────────────────────────────────────────
    # Cute kernels for preprocess / postprocess (replace PyTorch ref).
    # Each is a standalone elementwise kernel.  Critical path numbers:
    #   PyTorch ref prep + 3× post  ≈ 1.11 ms (cuda event)
    #   cute prep + 3× cute post    ≈ 0.15 ms (target, FA4 same range)
    # ─────────────────────────────────────────────────────────────────

    class _FlashAttnBwdPreprocess:
        """Compute dpsum = sum(O*dO) + lse_log2 = lse * log2(e) +
        zero dq_accum.  Each CTA processes `tile_m` rows; threads
        within the CTA cooperate on the D axis (`threads_per_row`)
        and loop `tile_m / rows_per_block` times to cover all rows."""

        def __init__(self, dtype, head_dim,
                     tile_m: int = 128, num_threads: int = 256):
            self.dtype = dtype
            self.head_dim = head_dim
            self.tile_m = tile_m
            # FA4 default 256 threads — exactly one thread per row of
            # tile_m=128 + spare for the col loop.  Halves rounds vs
            # 128-thread version and lets every thread directly write
            # 1 row of LSE/dpsum without a `col_chunk == 0` predicate.
            self.num_threads = num_threads
            self.elems_per_load_O = 128 // dtype.width  # 8 fp16
            self.threads_per_row = head_dim // self.elems_per_load_O
            self.rows_per_block = num_threads // self.threads_per_row
            assert tile_m % self.rows_per_block == 0
            self.rounds = tile_m // self.rows_per_block

        @cute.jit
        def __call__(
            self,
            mO: cute.Tensor, mdO: cute.Tensor,
            mPdPsum: cute.Tensor, mLSE: cute.Tensor,
            mLSElog2: cute.Tensor, mdQaccum: cute.Tensor,
            stream: cuda.CUstream = None,
        ):
            N = cute.size(mO.shape[2])
            H_q = cute.size(mO.shape[1])
            B = cute.size(mO.shape[0])
            n_blocks = N // self.tile_m
            grid = (n_blocks, H_q, B)
            self.kernel(
                mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum,
            ).launch(
                grid=grid, block=[self.num_threads, 1, 1],
                stream=stream, use_pdl=True)

        @cute.kernel
        def kernel(
            self,
            mO: cute.Tensor, mdO: cute.Tensor,
            mPdPsum: cute.Tensor, mLSE: cute.Tensor,
            mLSElog2: cute.Tensor, mdQaccum: cute.Tensor,
        ):
            # FA4-aligned preprocess (flash_bwd_preprocess.py kernel).
            # Issue all LDGs upfront via tiled_copy_2d's per-thread
            # frag → bulk reduce → bulk STG so LDG latency overlaps
            # with reduce ALU instead of serializing per-row.
            cute.arch.griddepcontrol_wait()
            m_block, h_idx, b_idx = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            EPL = self.elems_per_load_O  # 8 fp16
            EPL_dQ = 128 // Float32.width  # 4 fp32

            # Build TiledCopy for O / dO LDG and for dQaccum STG.
            # FA4-style: thr_layout via make_ordered_layout(order=(1,
            # 0)) so col dim is fast-varying — coalesced 128-bit LDG.
            o_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype, num_bits_per_copy=128)
            o_thr_per_row = self.head_dim // EPL
            o_rows_per_iter = self.num_threads // o_thr_per_row
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(
                o_atom,
                cute.make_ordered_layout(
                    (o_rows_per_iter, o_thr_per_row), order=(1, 0)),
                cute.make_layout((1, EPL)),
            )
            # 1D tiled_copy for dQaccum zero-fill (flat tile_m*D fp32).
            dq_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32, num_bits_per_copy=128)
            gmem_tiled_copy_dQacc = cute.make_tiled_copy_tv(
                dq_atom,
                cute.make_layout(self.num_threads),
                cute.make_layout(EPL_dQ),
            )

            # cute.assume on outer strides (B, H_q, N) of mO/mdO so
            # the 128-bit LDG atom is accepted — strides are
            # H_q*N*D, N*D, D — all multiples of 8 fp16 (D=64).
            mO_a = cute.make_tensor(
                mO.iterator,
                cute.make_layout(
                    mO.shape,
                    stride=(*(cute.assume(s, divby=EPL)
                              for s in mO.stride[:-1]),
                            mO.stride[-1]),
                ),
            )
            mdO_a = cute.make_tensor(
                mdO.iterator,
                cute.make_layout(
                    mdO.shape,
                    stride=(*(cute.assume(s, divby=EPL)
                              for s in mdO.stride[:-1]),
                            mdO.stride[-1]),
                ),
            )
            # Same for mdQaccum (3D fp32, last-dim contiguous, outer
            # strides multiples of 4 fp32 = 128-bit).
            mdQ_a = cute.make_tensor(
                mdQaccum.iterator,
                cute.make_layout(
                    mdQaccum.shape,
                    stride=(*(cute.assume(s, divby=EPL_dQ)
                              for s in mdQaccum.stride[:-1]),
                            mdQaccum.stride[-1]),
                ),
            )
            # Tile O/dO into per-(b, h, m_block) (tile_m, head_dim).
            mO_cur = mO_a[b_idx, h_idx, None, None]
            mdO_cur = mdO_a[b_idx, h_idx, None, None]
            gO = cute.local_tile(
                mO_cur, (self.tile_m, self.head_dim), (m_block, 0))
            gdO = cute.local_tile(
                mdO_cur, (self.tile_m, self.head_dim), (m_block, 0))
            thr_O = gmem_tiled_copy_O.get_slice(tidx)
            tOgO = thr_O.partition_S(gO)
            tOgdO = thr_O.partition_S(gdO)
            tOrO = cute.make_fragment_like(tOgO, self.dtype)
            tOrdO = cute.make_fragment_like(tOgdO, self.dtype)
            # FA4 line 308-310: bulk LDG via cute.copy with unroll_full
            # — issues all LDGs in one shot so subsequent ALU/STG
            # overlap with the in-flight LDG group (no per-row stall).
            cute.copy(gmem_tiled_copy_O, tOgO, tOrO)
            cute.copy(gmem_tiled_copy_O, tOgdO, tOrdO)

            # PDL: signal downstream (main bwd) prologue can start.
            # FA4 mainloop_bwd_preprocess line 316 fires it AFTER the
            # LDGs but BEFORE the reduce + STGs.  Main bwd's consumer
            # warps only read pdpsum / lse_log2 / dq_accum via TMA
            # deep inside the m-loop (long after prologue), so the
            # ~30-50 µs of overlap saved by an early launch_dependents
            # wins versus deferring to kernel end.  Stream + PDL
            # semantics still ensure the actual reads in main wait
            # for our stores to drain.
            cute.arch.griddepcontrol_launch_dependents()

            # Pointwise (tOrO * tOrdO) and reduce along the col axis
            # to a single fp32 per row.  reduction_profile = (0, None,
            # 1) sums modes 0 and 2 of the per-thread frag (which are
            # the inner-cells × head_dim_iter), keeping mode 1 (rows).
            # FA4 line 318.
            pdpsum_per_row = (
                tOrO.load().to(Float32)
                * tOrdO.load().to(Float32)
            ).reduce(
                cute.ReductionOp.ADD, init_val=0.0,
                reduction_profile=(0, None, 1),
            )
            # warp-level reduce across `threads_per_row` lanes.
            pdpsum_per_row = utils.warp_shuffle_reduce(
                pdpsum_per_row, lambda a, b: a + b,
                width=o_thr_per_row,
            )

            # Per-thread frag covers CPY_M rows per thread.  With
            # thr_layout=(rows_per_iter, threads_per_row) order=(1,0),
            # threads with same `tidx // threads_per_row` share the
            # same row group; iter m in [0, CPY_M) hops by
            # rows_per_iter rows.  Only col_chunk == 0 lane writes.
            col_chunk = tidx % o_thr_per_row
            row_in_block = tidx // o_thr_per_row
            CPY_M = cute.size(tOrO.shape[1])
            if col_chunk == 0:
                for m in cutlass.range_constexpr(CPY_M):
                    row_global = (m_block * self.tile_m
                                  + row_in_block
                                  + m * o_rows_per_iter)
                    mPdPsum[b_idx, h_idx, row_global] = (
                        pdpsum_per_row[m])
                    lse_val = mLSE[b_idx, h_idx, row_global]
                    mLSElog2[b_idx, h_idx, row_global] = (
                        lse_val * Float32(LOG2E))

            # Zero dQaccum: one bulk STG covering tile_m*D fp32 cells.
            mdQ_cur = mdQ_a[b_idx, h_idx, None]
            gdQ = cute.local_tile(
                mdQ_cur, (self.tile_m * self.head_dim,), (m_block,))
            thr_dQ = gmem_tiled_copy_dQacc.get_slice(tidx)
            tdQgdQ = thr_dQ.partition_S(gdQ)
            zero = cute.make_fragment_like(tdQgdQ, Float32)
            zero.fill(0.0)
            cute.copy(gmem_tiled_copy_dQacc, zero, tdQgdQ)

    class _FlashAttnBwdPostprocess:
        """FA4-aligned dQ postprocess for flat dq_accum layout.

        mdAccum is the FP32 dq_accum buffer in (B, H_q, N*D) flat
        per-(B, H_q) layout — written by the main kernel via the
        `make_tiled_copy_tv((128, num_wg_dQ), val=4)` TV pattern.
        Each per-(B, H_q, m_block) chunk holds tile_m*D fp32
        organized as (slot_size, num_wg_dQ) flat where slot_size =
        tile_m*D/num_wg_dQ.

        Pipeline (matches FA4 flash_bwd_postprocess.py:493+):
          1. G2S — copy gdQaccum 1D `(tile_m*D,)` → sdQaccum smem
             via the SAME flat TV layout as main kernel's R2S, so
             smem ends up with bytes in (slot_size, num_wg_dQ)
             order.
          2. S2R — load smem → register frag in TV order; alias
             register cells via `tiled_mma.partition_C` so the
             same regs are now interpreted as the WGMMA-frag for
             (tile_m, D) in (m, d) coords (the inverse of the
             main kernel's "register → flat smem" mangling).
          3. cvt FP32 → FP16 with scale.
          4. R2S to row-major sdQ via stmatrix
             (`make_tiled_copy_C(stmatrix, tiled_mma)`).
          5. G2S — coalesced 128-bit STG sdQ → mdQ[(B, H_q, N, D)].
        """

        def __init__(self, dtype, head_dim,
                     tile_m: int = 128, num_threads: int = 256,
                     num_wg_dQ: int = 2,
                     AtomLayoutMdQ: int = 2,
                     dQ_swapAB: bool = False):
            self.dtype = dtype
            self.head_dim = head_dim
            self.tile_m = tile_m
            self.num_threads = num_threads
            self.num_wg_dQ = num_wg_dQ
            self.AtomLayoutMdQ = AtomLayoutMdQ
            self.dQ_swapAB = dQ_swapAB
            assert num_threads == 128 * num_wg_dQ, (
                "num_threads must equal 128 * num_wg_dQ to match "
                "the main kernel's TV layout (128 thr × num_wg)")

        def _get_tiled_mma(self):
            num_wg_mma = self.num_threads // 128
            atom_layout_dQ = (
                self.AtomLayoutMdQ,
                num_wg_mma // self.AtomLayoutMdQ,
                1,
            )
            tiler_mn_dQ = (
                self.tile_m // atom_layout_dQ[0],
                self.head_dim // atom_layout_dQ[1],
            )
            return sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype, self.dtype,
                warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=atom_layout_dQ,
                tiler_mn=tiler_mn_dQ,
            )

        @cute.jit
        def __call__(
            self,
            mAccum: cute.Tensor, mOut: cute.Tensor,
            scale: Float32,
            stream: cuda.CUstream = None,
        ):
            N = cute.size(mOut.shape[2])
            H = cute.size(mOut.shape[1])
            B = cute.size(mOut.shape[0])
            n_blocks = N // self.tile_m

            # Mark dq_accum strides as 4-fp32 (16-byte) aligned —
            # both outer strides are N*D and H_q*N*D which are
            # multiples of fp32×4.  Required for 128-bit cp atom.
            mAccum = cute.make_tensor(
                mAccum.iterator,
                cute.make_layout(
                    mAccum.shape,
                    stride=(
                        *(cute.assume(s, divby=4)
                          for s in mAccum.stride[:-1]),
                        mAccum.stride[-1],
                    ),
                ),
            )
            # mOut strides are H_q*N*D, N*D, D — all multiples of
            # 8 fp16 (= 128-bit aligned) since D=head_dim=64.
            # cute.assume so the 128-bit STG atom is accepted.
            elems_per_st = 128 // self.dtype.width
            mOut = cute.make_tensor(
                mOut.iterator,
                cute.make_layout(
                    mOut.shape,
                    stride=(
                        *(cute.assume(s, divby=elems_per_st)
                          for s in mOut.stride[:-1]),
                        mOut.stride[-1],
                    ),
                ),
            )

            tiled_mma = self._get_tiled_mma()

            # Flat sdQaccum smem layout (slot_size, num_wg_dQ).
            slot_size = self.tile_m * self.head_dim // self.num_wg_dQ
            sdQaccum_layout = cute.make_layout(
                (slot_size, self.num_wg_dQ))
            # Row-major sdQ smem with FA4 swizzle atom.
            sdQ_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    cutlass.utils.LayoutEnum.ROW_MAJOR,
                    self.dtype, self.head_dim,
                ), self.dtype,
            )
            sdQ_layout = cute.tile_to_shape(
                sdQ_atom, (self.tile_m, self.head_dim), (0, 1))

            # G2S = same TV as main R2S so the round-trip layout
            # matches.  Use 128-bit cp atom matching the main
            # kernel's R2S; align-16 input ptr is set via
            # `_to_cute_tensor3_align16`.
            g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    Float32, num_bits_per_copy=128),
                cute.make_layout((128, self.num_wg_dQ)),
                cute.make_layout(128 // Float32.width),
            )
            # gmem dQ store: 128-bit per thread per copy.
            # FA4 copy_utils.tiled_copy_2d uses make_ordered_layout
            # with order=(1, 0) so the col dim is FAST-VARYING.
            # That way threads in the same warp span consecutive
            # gmem cols → coalesced 128-bit STG.  Without
            # make_ordered_layout (using plain make_layout), mode
            # 0 (rows) becomes fast and warp-31 spread across rows
            # produces uncoalesced STGs (NCU reported ~47 % speedup
            # opportunity from non-coalesced global stores).
            elems_per_st = 128 // self.dtype.width  # 8 fp16
            threads_per_row = self.head_dim // elems_per_st
            rows_per_iter = self.num_threads // threads_per_row
            gmem_tiled_copy_dQ = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.dtype,
                    num_bits_per_copy=elems_per_st * self.dtype.width),
                cute.make_ordered_layout(
                    (rows_per_iter, threads_per_row),
                    order=(1, 0),
                ),
                cute.make_layout((1, elems_per_st)),
            )

            # SMEM size: sdQaccum (FP32, slot_size*num_wg) and
            # sdQ (FP16, tile_m*D) are aliased on the same region;
            # take the max byte size so the SM CTA carveout is the
            # real footprint, not the default 228 KB max — which
            # collapses occupancy to 1 CTA / SM (FA4 uses the same
            # max() pattern in flash_bwd_postprocess line 240).
            smem_size = max(
                cute.size_in_bytes(Float32, sdQaccum_layout),
                cute.size_in_bytes(self.dtype, sdQ_layout),
            )

            grid = (n_blocks, H, B)
            self.kernel(
                mAccum, mOut, scale, tiled_mma,
                sdQaccum_layout, sdQ_layout,
                g2s_tiled_copy_dQaccum, gmem_tiled_copy_dQ,
            ).launch(
                grid=grid,
                block=[self.num_threads, 1, 1],
                smem=smem_size,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mAccum: cute.Tensor, mOut: cute.Tensor,
            scale: Float32,
            tiled_mma: cute.TiledMma,
            sdQaccum_layout: cute.Layout,
            sdQ_layout: cute.ComposedLayout,
            g2s_tiled_copy_dQaccum: cute.TiledCopy,
            gmem_tiled_copy_dQ: cute.TiledCopy,
        ):
            # No PDL gate — post-dQ relies on stream serialisation
            # for main kernel's dq_accum visibility (FA4 pattern in
            # flash_bwd_postprocess.py: launch without use_pdl, no
            # griddepcontrol_wait inside).  This avoids the 1-2 µs
            # mbarrier wait latency on every CTA's prologue.
            m_block, h_idx, b_idx = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()

            # SMEM allocation: sdQaccum (FP32, full m_block worth)
            # aliased with sdQ (FP16 row-major swizzled output).
            smem = cutlass.utils.SmemAllocator()
            sdQaccum = smem.allocate_tensor(
                Float32, sdQaccum_layout, byte_alignment=1024)
            # Alias same smem region as FP16 sdQ (separate
            # lifetime: G2S→S2R→cvt→R2S→G2G; sdQaccum dies before
            # sdQ is written via stmatrix).
            sdQ = cute.make_tensor(
                cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype),
                sdQ_layout,
            )

            # gdQaccum 1D (tile_m*D,) for this (b, h, m_block),
            # reshaped to (slot_size, num_wg_dQ) so the flat TV
            # layout's partition_S rank-matches.  The stride-4
            # cute.assume on the outer mode is required so the
            # CuTe DSL recognises the 16-byte alignment for
            # 128-bit cp.async G2S.
            mdQaccum_cur = mAccum[b_idx, h_idx, None]
            slot_size_local = (
                self.tile_m * self.head_dim // self.num_wg_dQ)
            gdQaccum_1d = cute.local_tile(
                mdQaccum_cur,
                (self.tile_m * self.head_dim,),
                (m_block,),
            )
            gdQaccum = cute.make_tensor(
                gdQaccum_1d.iterator,
                cute.make_layout(
                    (slot_size_local, self.num_wg_dQ),
                    stride=(1, slot_size_local),
                ),
            )
            # gdQ row-major (tile_m, D) for output.
            gdQ = cute.local_tile(
                mOut[b_idx, h_idx, None, None],
                (self.tile_m, self.head_dim),
                (m_block, 0),
            )

            # Step 1: G2S — gdQaccum (1D) → sdQaccum (2D flat).
            g2s_thr = g2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQgdQaccum = g2s_thr.partition_S(gdQaccum)
            tdQsdQaccumg2s = g2s_thr.partition_D(sdQaccum)
            cute.copy(g2s_tiled_copy_dQaccum,
                      tdQgdQaccum, tdQsdQaccumg2s)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: S2R via SAME TV layout — registers receive
            # cells in TV order (FA4 line 519+).  Crucially, we
            # allocate `acc` directly with WGMMA partition shape,
            # then use `acc.iterator` aliased with TV layout so the
            # same register file holds values in TV order.
            s2r_thr = g2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = s2r_thr.partition_S(sdQaccum)
            acc_shape = tiled_mma.partition_shape_C(
                (self.tile_m, self.head_dim)
                if not self.dQ_swapAB
                else (self.head_dim, self.tile_m))
            acc = cute.make_fragment(acc_shape, Float32)
            assert cute.size(acc) == cute.size(tdQsdQaccum)
            tdQrdQaccum = cute.make_tensor(
                acc.iterator,
                cute.make_layout(tdQsdQaccum.shape))
            cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)

            # Step 3: cvt FP32→FP16 with scale.  Both `acc` (FP32)
            # and `rdQ` (FP16) have WGMMA partition_C layout — the
            # same physical register file is read via WGMMA frag
            # offsets.  This is FA4's exact pattern (line 533).
            rdQ = cute.make_fragment_like(acc, self.dtype)
            rdQ.store((acc.load() * scale).to(self.dtype))

            # Step 4: R2S to row-major sdQ via stmatrix.
            cute.arch.barrier()  # sdQaccum no longer needed.
            stmatrix_atom = utils.get_smem_store_atom_sm90(self.dtype)
            r2s_tiled_copy_dQ = cute.make_tiled_copy_C(
                stmatrix_atom, tiled_mma)
            r2s_thr = r2s_tiled_copy_dQ.get_slice(tidx)
            taccdQrdQ = r2s_thr.retile(rdQ)
            taccdQsdQ = r2s_thr.partition_D(sdQ)
            cute.copy(r2s_tiled_copy_dQ, taccdQrdQ, taccdQsdQ)

            # Step 5: S → R → G (FA4 line 569+).  Use the gmem
            # TV layout to partition sdQ for the smem load too,
            # so the per-thread frag aligns with how we'll write
            # gdQ.  cute.copy with the gmem_tiled_copy_dQ atom
            # emits 128-bit STG (one per thread), matching FA4 —
            # autovec_copy on a register→gmem path may fall back
            # to per-element STG and explode mio_throttle.
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()
            gmem_thr = gmem_tiled_copy_dQ.get_slice(tidx)
            tdQgdQ = gmem_thr.partition_D(gdQ)
            tdQsdQ = gmem_thr.partition_S(sdQ)
            tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
            cute.autovec_copy(tdQsdQ, tdQrdQ)
            cute.copy(gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ)
            # launch_dependents already fired at top — see PDL note.

    class _FlashAttnBwdPostprocessDual:
        """Fused dK + dV postprocess (same shape, different scale).

        Saves one kernel launch overhead per bwd call by processing
        both accumulators in the same CTA.  Both input/output tensors
        must share (B, H_kv, N, D) — typical for GQA backward.
        """

        def __init__(self, dtype, head_dim,
                     tile_m: int = 128, num_threads: int = 128):
            self.dtype = dtype
            self.head_dim = head_dim
            self.tile_m = tile_m
            self.num_threads = num_threads
            self.elems_per_load_acc = 128 // Float32.width
            self.threads_per_row = head_dim // self.elems_per_load_acc
            self.rows_per_block = num_threads // self.threads_per_row
            assert tile_m % self.rows_per_block == 0
            self.rounds = tile_m // self.rows_per_block

        @cute.jit
        def __call__(
            self,
            mAccumK: cute.Tensor, mOutK: cute.Tensor,
            mAccumV: cute.Tensor, mOutV: cute.Tensor,
            scaleK: Float32, scaleV: Float32,
            stream: cuda.CUstream = None,
        ):
            N = cute.size(mAccumK.shape[2])
            H = cute.size(mAccumK.shape[1])
            B = cute.size(mAccumK.shape[0])
            n_blocks = N // self.tile_m
            grid = (n_blocks, H, B)
            self.kernel(
                mAccumK, mOutK, mAccumV, mOutV, scaleK, scaleV,
            ).launch(
                grid=grid,
                block=[self.num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mAccumK: cute.Tensor, mOutK: cute.Tensor,
            mAccumV: cute.Tensor, mOutV: cute.Tensor,
            scaleK: Float32, scaleV: Float32,
        ):
            m_block, h_idx, b_idx = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            EPL_acc = 128 // Float32.width  # 4 fp32
            EPL_out = 128 // self.dtype.width  # 8 fp16

            # cute.assume on outer strides for 128-bit LDG/STG.
            mAccumK_a = cute.make_tensor(
                mAccumK.iterator,
                cute.make_layout(mAccumK.shape, stride=(
                    *(cute.assume(s, divby=EPL_acc)
                      for s in mAccumK.stride[:-1]),
                    mAccumK.stride[-1])),
            )
            mAccumV_a = cute.make_tensor(
                mAccumV.iterator,
                cute.make_layout(mAccumV.shape, stride=(
                    *(cute.assume(s, divby=EPL_acc)
                      for s in mAccumV.stride[:-1]),
                    mAccumV.stride[-1])),
            )
            mOutK_a = cute.make_tensor(
                mOutK.iterator,
                cute.make_layout(mOutK.shape, stride=(
                    *(cute.assume(s, divby=EPL_out)
                      for s in mOutK.stride[:-1]),
                    mOutK.stride[-1])),
            )
            mOutV_a = cute.make_tensor(
                mOutV.iterator,
                cute.make_layout(mOutV.shape, stride=(
                    *(cute.assume(s, divby=EPL_out)
                      for s in mOutV.stride[:-1]),
                    mOutV.stride[-1])),
            )

            # Use SAME thr_layout for LDG and STG so per-thread row /
            # col mapping matches across the load → cvt → store
            # round-trip.  Pick the stg-side chunk (8 fp16 = 8 fp32 in
            # element count, 32 vs 16 bytes — atom-bits differs but
            # element count is what TV partitioning sees).  cute
            # decomposes the 8-fp32 LDG into 2× 128-bit universal_copy
            # atoms internally.
            ldg_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32, num_bits_per_copy=128)
            stg_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype, num_bits_per_copy=128)
            thr_per_row = self.head_dim // EPL_out
            rows_per_iter = self.num_threads // thr_per_row
            shared_thr = cute.make_ordered_layout(
                (rows_per_iter, thr_per_row), order=(1, 0))
            ldg_copy = cute.make_tiled_copy_tv(
                ldg_atom, shared_thr,
                cute.make_layout((1, EPL_out)),
            )
            stg_copy = cute.make_tiled_copy_tv(
                stg_atom, shared_thr,
                cute.make_layout((1, EPL_out)),
            )

            # Tile (tile_m, head_dim) for this (b, h, m_block).
            gAccK = cute.local_tile(
                mAccumK_a[b_idx, h_idx, None, None],
                (self.tile_m, self.head_dim), (m_block, 0))
            gAccV = cute.local_tile(
                mAccumV_a[b_idx, h_idx, None, None],
                (self.tile_m, self.head_dim), (m_block, 0))
            gOutK = cute.local_tile(
                mOutK_a[b_idx, h_idx, None, None],
                (self.tile_m, self.head_dim), (m_block, 0))
            gOutV = cute.local_tile(
                mOutV_a[b_idx, h_idx, None, None],
                (self.tile_m, self.head_dim), (m_block, 0))

            ldg_thr = ldg_copy.get_slice(tidx)
            stg_thr = stg_copy.get_slice(tidx)
            tAgAccK = ldg_thr.partition_S(gAccK)
            tAgAccV = ldg_thr.partition_S(gAccV)
            tOgOutK = stg_thr.partition_D(gOutK)
            tOgOutV = stg_thr.partition_D(gOutV)
            tArAccK = cute.make_fragment_like(tAgAccK, Float32)
            tArAccV = cute.make_fragment_like(tAgAccV, Float32)

            # Bulk LDG of both K and V accumulators.
            cute.copy(ldg_copy, tAgAccK, tArAccK)
            cute.copy(ldg_copy, tAgAccV, tArAccV)

            # cvt + scale (register-only).
            tOrOutK = cute.make_fragment_like(tOgOutK, self.dtype)
            tOrOutV = cute.make_fragment_like(tOgOutV, self.dtype)
            tOrOutK.store((tArAccK.load() * scaleK).to(self.dtype))
            tOrOutV.store((tArAccV.load() * scaleV).to(self.dtype))

            # Bulk STG.
            cute.copy(stg_copy, tOrOutK, tOgOutK)
            cute.copy(stg_copy, tOrOutV, tOgOutV)

    # ── Compilation cache and runner ──

    _bwd_compile_cache: dict = {}
    _prep_compile_cache: dict = {}
    _post_compile_cache: dict = {}

    def _to_cute_tensor4(t):
        ct = from_dlpack(t.detach(), assumed_align=16)
        return ct.mark_layout_dynamic(leading_dim=t.ndim - 1)

    def _to_cute_tensor3(t):
        ct = from_dlpack(t.detach(), assumed_align=4)
        return ct.mark_layout_dynamic(leading_dim=t.ndim - 1)

    def _to_cute_tensor3_align16(t):
        # For tensors whose strides are guaranteed 16-byte aligned
        # (e.g. dq_accum (B, H_q, N*D) — last-dim stride 1, outer
        # strides are N*D / H_q*N*D both divisible by 16 for fp32).
        # Needed by FA4-style flat dq_accum so the postprocess
        # G2S can use 128-bit LDG.
        ct = from_dlpack(t.detach(), assumed_align=16)
        return ct.mark_layout_dynamic(leading_dim=t.ndim - 1)

    _fa4_main_compile_cache: dict = {}

    def _run_main_bwd_fa4(q, k, v, dout, lse_log2, dpsum, dq_accum,
                          dk_accum, dv_accum, softmax_scale, is_causal,
                          qhead_per_kvhead):
        """Diagnostic: route the dq_accum write through FA4's main
        kernel directly.  Used to isolate whether our postprocess
        (read flat dq_accum + WGMMA-frag round-trip) is correct: if
        FA4 main + our postprocess produces a valid dQ, our main
        kernel R2S pattern needs fixing; otherwise our postprocess
        is also buggy.
        """
        import time as _t
        import sys
        if "/user/chenyaojian/flash-attention" not in sys.path:
            sys.path.insert(0, "/user/chenyaojian/flash-attention")
        from flash_attn.cute.flash_bwd_sm90 import (
            FlashAttentionBackwardSm90,
        )

        B, H_q, N, D = q.shape
        H_kv = H_q // qhead_per_kvhead
        device = q.device
        cute_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16

        # FA4 expects QKV/dO in (B, N, H, D) layout (b, s, n, h).
        q_fa4 = q.transpose(1, 2).contiguous()
        k_fa4 = k.transpose(1, 2).contiguous()
        v_fa4 = v.transpose(1, 2).contiguous()
        dout_fa4 = dout.transpose(1, 2).contiguous()
        # FA4 dk_accum/dv_accum for GQA: (B, H_kv, N*D) flat fp32.
        dk_fa4 = torch.zeros(
            B, H_kv, N * D, dtype=torch.float32, device=device)
        dv_fa4 = torch.zeros(
            B, H_kv, N * D, dtype=torch.float32, device=device)

        cQ = from_dlpack(q_fa4.detach(), assumed_align=16)
        cQ = cQ.mark_layout_dynamic(leading_dim=q_fa4.ndim - 1)
        cK = from_dlpack(k_fa4.detach(), assumed_align=16)
        cK = cK.mark_layout_dynamic(leading_dim=k_fa4.ndim - 1)
        cV = from_dlpack(v_fa4.detach(), assumed_align=16)
        cV = cV.mark_layout_dynamic(leading_dim=v_fa4.ndim - 1)
        cdO = from_dlpack(dout_fa4.detach(), assumed_align=16)
        cdO = cdO.mark_layout_dynamic(leading_dim=dout_fa4.ndim - 1)
        cLSE = from_dlpack(lse_log2.detach(), assumed_align=4)
        cLSE = cLSE.mark_layout_dynamic(leading_dim=2)
        cdPsum = from_dlpack(dpsum.detach(), assumed_align=4)
        cdPsum = cdPsum.mark_layout_dynamic(leading_dim=2)
        cdQaccum = from_dlpack(dq_accum.detach(), assumed_align=16)
        cdQaccum = cdQaccum.mark_layout_dynamic(leading_dim=2)
        cdK = from_dlpack(dk_fa4.detach(), assumed_align=16)
        cdK = cdK.mark_layout_dynamic(leading_dim=2)
        cdV = from_dlpack(dv_fa4.detach(), assumed_align=16)
        cdV = cdV.mark_layout_dynamic(leading_dim=2)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        scale = cutlass.Float32(softmax_scale)

        # FA4 hdim≤64 default config (interface.py line 178):
        #   m=128, n=128, num_stages=2, SdP_swapAB=True,
        #   AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2.
        key = (cute_dtype, D, bool(is_causal), qhead_per_kvhead)
        if key not in _fa4_main_compile_cache:
            t0 = _t.time()
            fa4_bwd = FlashAttentionBackwardSm90(
                dtype=cute_dtype,
                head_dim=D,
                head_dim_v=D,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=bool(is_causal),
                tile_m=128, tile_n=128,
                Q_stage=2, dO_stage=2, PdS_stage=2,
                SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
                num_threads=384,
            )
            print("[fa4-main] cute.compile start", flush=True)
            _fa4_main_compile_cache[key] = cute.compile(
                fa4_bwd,
                cQ, cK, cV, cdO, cLSE, cdPsum, cdQaccum,
                cdK, cdV,
                scale,
                None, None, None, None,  # cu_seqlens / seqused
                None, None,              # window left/right
                None, None, None,        # semaphores
                None, None,              # aux, blocksparse
                stream,
            )
            print(f"[fa4-main] compile done ({_t.time()-t0:.1f}s)",
                  flush=True)

        _fa4_main_compile_cache[key](
            cQ, cK, cV, cdO, cLSE, cdPsum, cdQaccum, cdK, cdV,
            scale,
            None, None, None, None, None, None,
            None, None, None, None, None, stream,
        )
        torch.cuda.synchronize()


    def _run_main_bwd_cute(q, k, v, dout, lse_log2, dpsum, dq_accum,
                           dk_accum, dv_accum, softmax_scale, is_causal,
                           qhead_per_kvhead, compile_only=False):
        """Compile and run the CuTe DSL main backward kernel."""
        # Diagnostic flag — bypass our main and use FA4's so we can
        # test our postprocess in isolation.  When set, dk/dv from
        # FA4 are discarded (allocated separately); our outer dispatch
        # reuses our existing dk/dv accums (zero) for postprocess —
        # only dQ is validated.
        if int(os.environ.get("BWD_USE_FA4_MAIN", "0")):
            return _run_main_bwd_fa4(
                q, k, v, dout, lse_log2, dpsum, dq_accum,
                dk_accum, dv_accum, softmax_scale, is_causal,
                qhead_per_kvhead)
        import time as _t

        B, H_q, N, D = q.shape
        cute_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16

        # ── Pipeline cutpoint instrumentation ──────────────────
        # When BWD_EARLY_EXIT_GEMM=N (1..5), the kernel is built to
        # drain wgmma right after GEMMi and skip the rest of each
        # iter (warp 1 also early-returns).  Used by perf bench to
        # measure cumulative critical-path time per GEMM cut and
        # compare against an FA4 build with the same cuts.  The
        # output is intentionally garbage in this mode — only call
        # from latency benchmarks, never from correctness tests.
        early_exit = int(os.environ.get("BWD_EARLY_EXIT_GEMM", "0"))
        assert 0 <= early_exit <= 5, (
            f"BWD_EARLY_EXIT_GEMM must be 0..5, got {early_exit}")
        # Default = best-perf tile_m=64 + dQ_single_wg=True path
        # (~2.94 ms).  Opt into FA4-aligned tile_m=128 + M-split via:
        #   BWD_TILE_M=128 BWD_ATOMLAYOUT_MDQ=2 BWD_DQ_SINGLE_WG=0
        # — currently 3.96 ms (correctness PASS) due to shared-buffer
        # 256-thread sync overhead; see notes/tile_m128_perf.md for
        # the planned per-WG buffer optimization to reach FA4 parity.
        _tile_m = int(os.environ.get("BWD_TILE_M", "64"))
        _tile_n = int(os.environ.get("BWD_TILE_N", "128"))
        _atom_mdq = int(os.environ.get("BWD_ATOMLAYOUT_MDQ", "1"))
        _dq_swg = int(os.environ.get("BWD_DQ_SINGLE_WG", "1"))
        _sdp_swapab = int(os.environ.get("BWD_SDP_SWAPAB", "1"))
        key = (cute_dtype, D, bool(is_causal),
               qhead_per_kvhead, early_exit,
               _tile_m, _tile_n, _atom_mdq, _dq_swg,
               _sdp_swapab)

        cQ = _to_cute_tensor4(q)
        cK = _to_cute_tensor4(k)
        cV = _to_cute_tensor4(v)
        cdO = _to_cute_tensor4(dout)
        cLSE = _to_cute_tensor3(lse_log2)
        cdPsum = _to_cute_tensor3(dpsum)
        # FA4-style flat dq_accum: 3D (B, H_q, N*D) instead of 4D.
        # Use align=16 so 128-bit LDG/STG ops work; outer strides
        # = N*D, H_q*N*D are both fp32 16-byte aligned.
        cdQaccum = _to_cute_tensor3_align16(dq_accum)
        cdK = _to_cute_tensor4(dk_accum)
        cdV = _to_cute_tensor4(dv_accum)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        scale = cutlass.Float32(softmax_scale)

        if key not in _bwd_compile_cache:
            t0 = _t.time()
            print(f"[bwd-dsl] cute.compile start (causal={is_causal}, gqa={qhead_per_kvhead})",
                  flush=True)
            bwd = _BwdMainKernel(
                dtype=cute_dtype,
                head_dim=D,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=bool(is_causal),
                tile_m=_tile_m, tile_n=_tile_n,
                AtomLayoutMdQ=_atom_mdq,
                # D-3 invariant: Q_stage = PdS_stage so smem_idx_PdS
                # rotates correctly with q_state.index.
                Q_stage=2, dO_stage=2, PdS_stage=2,
                # FA4-style named-barrier dQ R2S sync.
                # tile_m=128: stage=2 mbarrier double-buffer is the
                # default — keeps overlap between warp 1's TMA store
                # and consumer's R2S of the next iter.  Named-barrier
                # path was investigated and it hangs in `bench_dsl_
                # only.py` (works in correctness due to different
                # input distribution) — kept gated behind
                # BWD_SDQACC_STAGE=1 for further debugging.
                sdQacc_stage=int(os.environ.get(
                    "BWD_SDQACC_STAGE",
                    "2" if _tile_m == 128 else "1")),
                use_named_dq_barrier=True,
                # K-2: dQ_single_wg=True puts mma_dQ on WG0 only and
                # frees WG1 to early-start next iter's GEMM1 (FA4
                # line 1603).  At tile_m=128 we override to False so
                # both WGs share the dQ tile via AtomLayoutMdQ=2.
                dQ_single_wg=bool(_dq_swg),
                # mma_dkv_is_rs: SwapAB on mma_SdP makes acc_S land
                # in (tile_n, tile_m) layout — i.e. P^T after softmax —
                # which is exactly the A operand of mma_dV.  With
                # a_source=RMEM, GEMM3 dV / GEMM5 dK take A straight
                # out of registers, eliminating R2S(P) entirely and
                # letting R2S(dS) overlap with GEMM3.  The previous
                # RS attempt regressed because PdS staging was
                # silently single-buffer (D-3) and LSE was per-row
                # gmem load (D-1) — both fixed now.
                SdP_swapAB=bool(_sdp_swapab),
                early_exit_after_gemm=early_exit,
            )
            print(f"[bwd-dsl] cute.compile start "
                  f"(causal={is_causal}, gqa={qhead_per_kvhead})",
                  flush=True)
            _bwd_compile_cache[key] = cute.compile(
                bwd,
                cQ, cK, cV, cdO, cLSE, cdPsum, cdQaccum, cdK, cdV,
                scale, stream,
            )
            print(f"[bwd-dsl] cute.compile done ({_t.time()-t0:.1f}s)",
                  flush=True)

        if compile_only:
            return
        _bwd_compile_cache[key](
            cQ, cK, cV, cdO, cLSE, cdPsum, cdQaccum, cdK, cdV,
            scale, stream,
        )


    def _run_preprocess_cute(out, dout, dpsum, lse, lse_log2, dq_accum,
                             compile_only=False):
        """Compile and run the CuTe DSL preprocess kernel."""
        import time as _t
        B, H_q, N, D = out.shape
        cute_dtype = cutlass.BFloat16 if out.dtype == torch.bfloat16 else cutlass.Float16
        key = (cute_dtype, D)
        cO = _to_cute_tensor4(out)
        cdO = _to_cute_tensor4(dout)
        cPdPsum = _to_cute_tensor3(dpsum)
        cLSE = _to_cute_tensor3(lse)
        cLSElog2 = _to_cute_tensor3(lse_log2)
        cdQaccum = _to_cute_tensor4(dq_accum)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if key not in _prep_compile_cache:
            t0 = _t.time()
            prep = _FlashAttnBwdPreprocess(
                dtype=cute_dtype, head_dim=D)
            print(f"[bwd-dsl] prep cute.compile start", flush=True)
            _prep_compile_cache[key] = cute.compile(
                prep, cO, cdO, cPdPsum, cLSE, cLSElog2, cdQaccum,
                stream)
            print(f"[bwd-dsl] prep done ({_t.time()-t0:.1f}s)",
                  flush=True)
        if compile_only:
            return
        _prep_compile_cache[key](
            cO, cdO, cPdPsum, cLSE, cLSElog2, cdQaccum, stream)


    def _run_postprocess_cute(accum, output, scale, compile_only=False):
        """Compile and run the CuTe DSL dQ postprocess kernel.

        accum: 3D flat (B, H_q, N*D) FP32 — the FA4-style mangled
        dQ accum from the main kernel.
        output: 4D (B, H_q, N, D) FP16 — the final dQ output.
        """
        import time as _t
        B, H, _ = accum.shape
        _, _, N, D = output.shape
        cute_dtype = cutlass.BFloat16 if output.dtype == torch.bfloat16 else cutlass.Float16
        _tile_m = int(os.environ.get("BWD_TILE_M", "64"))
        _atom_mdq = int(os.environ.get("BWD_ATOMLAYOUT_MDQ", "1"))
        _dq_swg = int(os.environ.get("BWD_DQ_SINGLE_WG", "1"))
        num_wg_dQ = 1 if _dq_swg else 2
        num_threads_post = 128 * num_wg_dQ
        key = (cute_dtype, D, _tile_m, _atom_mdq, _dq_swg)
        cAccum = _to_cute_tensor3_align16(accum)
        cOut = _to_cute_tensor4(output)
        scale_f32 = cutlass.Float32(scale)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if key not in _post_compile_cache:
            t0 = _t.time()
            post = _FlashAttnBwdPostprocess(
                dtype=cute_dtype, head_dim=D,
                tile_m=_tile_m,
                num_threads=num_threads_post,
                num_wg_dQ=num_wg_dQ,
                AtomLayoutMdQ=_atom_mdq,
            )
            print(f"[bwd-dsl] post cute.compile start", flush=True)
            _post_compile_cache[key] = cute.compile(
                post, cAccum, cOut, scale_f32, stream)
            print(f"[bwd-dsl] post done ({_t.time()-t0:.1f}s)",
                  flush=True)
        if compile_only:
            return
        _post_compile_cache[key](cAccum, cOut, scale_f32, stream)


    _post_dual_compile_cache: dict = {}

    def _run_postprocess_dual_cute(accum_k, out_k, accum_v, out_v,
                                   scale_k, scale_v, compile_only=False):
        """Compile and run the fused dK+dV postprocess kernel.
        Saves 1 kernel launch overhead vs 2 single-call posts."""
        import time as _t
        B, H, N, D = accum_k.shape
        cute_dtype = cutlass.BFloat16 if out_k.dtype == torch.bfloat16 else cutlass.Float16
        key = (cute_dtype, D)
        cAK = _to_cute_tensor4(accum_k)
        cOK = _to_cute_tensor4(out_k)
        cAV = _to_cute_tensor4(accum_v)
        cOV = _to_cute_tensor4(out_v)
        sK = cutlass.Float32(scale_k)
        sV = cutlass.Float32(scale_v)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if key not in _post_dual_compile_cache:
            t0 = _t.time()
            post = _FlashAttnBwdPostprocessDual(
                dtype=cute_dtype, head_dim=D)
            print(f"[bwd-dsl] post-dual cute.compile start", flush=True)
            _post_dual_compile_cache[key] = cute.compile(
                post, cAK, cOK, cAV, cOV, sK, sV, stream)
            print(f"[bwd-dsl] post-dual done "
                  f"({_t.time()-t0:.1f}s)", flush=True)
        if compile_only:
            return
        _post_dual_compile_cache[key](
            cAK, cOK, cAV, cOV, sK, sV, stream)


# ---------------------------------------------------------------------------
# Host dispatch (FROZEN signature)
# ---------------------------------------------------------------------------


def run_flash_bwd_dsl(q, k, v, out, dout, lse, dq, dk, dv,
                      softmax_scale, is_causal):
    """3-kernel backward pass — host dispatch."""
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    qhead_per_kvhead = H_q // H_kv
    device = q.device

    dpsum = torch.empty(B, H_q, N, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(B, H_q, N, dtype=torch.float32, device=device)
    # FA4-style flat dQ accumulator: shape (B, H_q, N * D) flat 1D
    # per (B, H_q).  Inside each (B, H_q), m_block layout is
    # (slot_size_per_wg, num_wg_dQ) flat, where slot_size_per_wg =
    # tile_m * D / num_wg_dQ.  This avoids the WGMMA-frag → row-major
    # bank conflicts that 4D (B, H_q, N, D) layout produced (NCU
    # measured 51.8 % SMEM-store conflicts before the refactor).
    # Postprocess reads with the matching make_tiled_copy_tv((128,
    # num_wg_dQ), val=4) and reinterprets registers as the WGMMA
    # frag for (tile_m, D).  Reference: FA4 interface.py line 1458.
    # prep cute kernel zero-inits dq_accum.
    dq_accum = torch.empty(B, H_q, N * D, dtype=torch.float32, device=device)
    # FA4-grid path: cross-CTA dK/dV accum requires FP32 buffers
    # (cp.reduce.async.bulk.add accumulates atomically across H_q
    # CTAs sharing the same H_kv tile).  Must zero-init since the
    # first cp.reduce reads existing value.
    dk_accum = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device=device)
    dv_accum = torch.zeros(B, H_kv, N, D, dtype=torch.float32, device=device)

    if not (_HAS_CUTE and q.is_cuda):
        raise RuntimeError(
            "run_flash_bwd_dsl requires CuTe DSL on CUDA. "
            "PyTorch tiled fallback removed because it silently masked "
            "cute.compile failures and made all perf measurements bogus.")

    # Match the standalone benchmark exactly: no torch.cuda.synchronize()
    # anywhere.  Each _run_* compiles (if needed) and executes in one shot.
    # PDL chain prep→main survives because cute.compile() does NOT issue
    # use_pdl=True kernels that would consume the parent signal.
    use_pytorch_prepost = int(
        os.environ.get("BWD_PYTORCH_PREPOST", "0"))

    if use_pytorch_prepost:
        _preprocess_ref(out, dout, dpsum, lse, lse_log2, dq_accum)
    else:
        _run_preprocess_cute(out, dout, dpsum, lse, lse_log2, dq_accum)

    if not int(os.environ.get("BWD_SKIP_MAIN", "0")):
        _run_main_bwd_cute(
            q, k, v, dout, lse_log2, dpsum, dq_accum,
            dk_accum, dv_accum,
            softmax_scale, is_causal, qhead_per_kvhead,
        )

    if use_pytorch_prepost:
        _postprocess_ref(dq_accum, dq, softmax_scale)
        _postprocess_ref(dk_accum, dk, softmax_scale)
        _postprocess_ref(dv_accum, dv, 1.0)
    else:
        _run_postprocess_cute(dq_accum, dq, softmax_scale)
        _run_postprocess_dual_cute(
            dk_accum, dk, dv_accum, dv,
            softmax_scale, 1.0,
        )

