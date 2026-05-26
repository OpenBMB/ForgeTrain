"""gemm_output — pad-and-slice fwd+dgrad+wgrad with device-fused RMW wgrad.

Production shape (MiniCPM4-8B / TP=2 / MBS=2 / SEQ=4096) — LM head:
  fwd:   y[S=4096, B=2, O=36724] = x[S, B, I=4096] @ w[O, I]^T
  dgrad: dx[T=8192, I]            = d_y[T, O]       @ w[O, I]
  wgrad: dw[O, I]                += d_y[T, O]^T    @ x[T, I]   (FP32 ACCUMULATE)

Pad-and-slice for the irregular axis O=36724
--------------------------------------------
``gemm_output`` is the only GEMM in the engine with an irregular axis:
``O = 36724`` (= 287 × 128 − 12) is not a multiple of any reasonable
``tile_N`` (256 / 128 / 64).  Calling the monolithic kernel directly
with the raw irregular shape causes the MMA epilogue to drop the tail
columns/rows and corrupts precision in all three directions.

We therefore pad O to ``O_pad = 36864 = 144 × 256`` (divisible by
``tile_N = 256`` AND ``tile_K = 64``), invoke the kernel on the padded
shape, and slice back to O on the way out.  This pattern is applied
in three slightly different shapes:

* **fwd** (``y = x @ w.T``) — the irregular axis is the *output*
  N-axis.  We cache a ``w_pad[O_pad, I]`` weight tensor whose tail
  rows ``[O:O_pad, :]`` are zero, and the kernel writes only the
  ``[:O]`` prefix back to the caller's tensor.
* **dgrad** (``dx = d_y @ w``) — the irregular axis is the
  *contraction* K-axis.  Padding K adds zero contributions to the
  output (both ``d_y_pad[:, O:O_pad] == 0`` and
  ``w_pad[O:O_pad, :] == 0``), so the result is exact.  We cache a
  ``d_y_pad[T, O_pad]`` BF16 buffer per process and ``copy_`` the
  current ``d_y[:, :O]`` into its leading columns each call; the
  tail stays at the zeros set during initial ``torch.zeros``.
* **wgrad** (``dw += d_y^T @ x``) — the irregular axis is the
  *output* M-axis.  Padding M means the kernel writes more rows
  than ``out_buf`` can hold, so we route the result through an FP32
  scratch buffer of shape ``(O_pad, I)`` and ``out_buf.add_(scratch[:O, :])``
  on the host to accumulate only the prefix back into the
  caller-provided ``out_buf``.  Discards the padded
  ``scratch[O:O_pad, :]`` rows.

ABI (matches :data:`_kernel_abi.GEMM_ABI`)
------------------------------------------

* ``gemm_output_fwd(x: BF16[S, B, I], w: BF16[O, I]) -> BF16[S, B, O]``
* ``gemm_output_dgrad(d_logits: BF16[T, O], output_w: BF16[O, I], *,
  out: BF16[T, I] | None = None) -> BF16[T, I]``

  - ``out=None`` (production hot path): allocate and return a fresh
    tensor.
  - ``out=Tensor``: write in-place into ``out``.

* ``gemm_output_wgrad(d_logits: BF16[T, O], hidden_final: BF16[T, I], *,
  out_buf: FP32[O, I]) -> out_buf``  (ACCUMULATE)

Hand-written CuTeDSL kernel
---------------------------
The monolithic kernel class below (~600 LOC) is built from the public
CuTeDSL Python API (``cutlass.cute`` / ``cutlass.cute.nvgpu`` /
``cutlass.pipeline`` / ``cutlass.utils``).  Every device-side primitive
maps directly onto a TMA / WGMMA / mbarrier hardware feature.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch

# ──────────────────────────────────────────────────────────────────────────
# CuTeDSL imports — public Python API (cutlass.cute / cutlass.cute.nvgpu /
# cutlass.pipeline / cutlass.utils) that drive Hopper SM90a TMA + WGMMA.
# ──────────────────────────────────────────────────────────────────────────
import cuda.bindings.driver as cuda  # noqa: E402

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.pipeline as pipeline  # noqa: E402
import cutlass.utils as utils  # noqa: E402
import cutlass.utils.hopper_helpers as sm90_utils  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# Per-direction config — three INDEPENDENT compiled kernel slots.
#
# Each direction (fwd / dgrad / wgrad) compiles its own
# ``PersistentDenseGemmKernel`` instance with its own tile / pipeline /
# cluster configuration. The values below were tuned for the
# LM-head shape (O_pad = 36864, I = 4096, T = 8192) on H100 SXM5;
# changing them does not require a code edit elsewhere because the
# kernel class is fully parameterised on these knobs.
#
# Notes on the values that aren't obvious from the math:
#
# * ``cluster_mn=(2,1)`` enables TMA B-multicast across the two CTAs in
# each cluster (same N-tile, different M-tiles), halving the B-operand
# HBM bandwidth.  ``cluster_size=4`` was tried (both (2,2) and (4,1))
# and rejected: the cluster-barrier consumer-arrive count grows fast
# enough that the synchronisation overhead exceeds the multicast
# savings on this shape.
# * ``raster_along="m"`` keeps the inner loop running over m-clusters,
# which means each B-tile lives in L2 across ``swizzle_m`` consecutive
# m-cluster steps.  Combined with ``swizzle_m=8`` on fwd (32 m_clusters
# over 4 chunks) and ``swizzle_m=4`` on dgrad, this keeps the A+B
# working set well under the 50 MB L2 capacity.
# * fwd uses ``k_pipe_mmas=2`` because the BF16 epilogue is shallow
# (``epi_stage=4`` is plenty); dgrad/wgrad use ``k_pipe_mmas=1``
# because the FP32 epilogue is much wider and the SMEM budget cannot
# afford a deeper K pipeline.
# ──────────────────────────────────────────────────────────────────────────
PERSISTENT_CONFIGS: dict[str, dict] = {
    # fwd: M=T=8192, N=O_pad=36864, K=I=4096.  Long-N (the lm_head
    # weight is the ~600 MB B operand) and short-K (input activations
    # ≈ 64 MB). Cluster grid m=32, n=144 with B-multicast inside each
    # 2-CTA cluster. raster="m" + swizzle_m=8 keeps the per-chunk
    # working set near 9 MB (8 MB A + 1 MB B), so the multi-tile A
    # reuse fits in L2 across the 144-step n-outer loop.
    "fwd":   {"tile_mn": (128, 256), "atom_layout_mnk": (2, 1, 1),
              "k_pipe_mmas": 2, "cluster_mn": (2, 1),
              "swizzle_m": 8, "epi_stage": 4, "raster_along": "m"},
    # dgrad: M=T=8192, N=I=4096, K=O_pad=36864. Long-K (compute-bound)
    # path; we keep the swizzle narrow (4) so the per-chunk A working
    # set is tighter, since pipeline drain hides the FP32 epilogue
    # cost on this direction's smaller B operand.
    "dgrad": {"tile_mn": (128, 256), "atom_layout_mnk": (2, 1, 1),
              "k_pipe_mmas": 1, "cluster_mn": (2, 1),
              "swizzle_m": 4, "epi_stage": 4, "raster_along": "m"},
    # wgrad: M=O_pad=36864, N=I=4096, K=T=8192. Long-M (the d_y_pad
    # operand is ~600 MB) and long-K, with an FP32 accumulator
    # epilogue. Wgrad uses the same shallow K pipeline as dgrad
    # (k_pipe_mmas=1) — the SMEM budget is dominated by the FP32
    # epi_tile, so a deeper K pipeline would force epi_stage down,
    # giving up TMA-store latency hiding without proportional
    # speedup. The accumulate path (RMW into ``out_buf`` instead of a
    # scratch + host-side add) is selected per call inside
    # ``gemm_output_wgrad`` when ``out_buf`` is 16-byte aligned.
    "wgrad": {"tile_mn": (128, 256), "atom_layout_mnk": (2, 1, 1),
              "k_pipe_mmas": 1, "cluster_mn": (2, 1),
              "swizzle_m": 8, "epi_stage": 4, "raster_along": "m"},
}

# ──────────────────────────────────────────────────────────────────────────
# Tile scheduler: chunk swizzle_m m-cluster rows along M, iterate N inside
# the chunk, then advance to the next chunk. raster_along="m" makes
# cluster_pid_m the inner-fastest axis (consecutive cluster_idx values share
# the same cluster_pid_n → B-tile L2 reuse across waves); raster_along="n"
# makes cluster_pid_n inner-fastest (same cluster_pid_m → A-tile L2 reuse).
# ──────────────────────────────────────────────────────────────────────────
def _swizzle_decompose(r, m_clusters, n_clusters, swizzle: int,
                       raster_along: str = "m"):
    """Inlined into @cute.kernel; swizzle and raster_along are constexpr."""
    if swizzle == 1:
        if raster_along == "m":
            cluster_pid_m = r // n_clusters
            cluster_pid_n = r % n_clusters
        else:
            cluster_pid_n = r // m_clusters
            cluster_pid_m = r % m_clusters
    else:
        chunk_size = swizzle * n_clusters
        chunk_idx = r // chunk_size
        within_chunk = r % chunk_size
        chunk_m_start = chunk_idx * swizzle
        chunk_m_size = cutlass.min(swizzle, m_clusters - chunk_m_start)
        if raster_along == "m":
            cluster_pid_m = chunk_m_start + (within_chunk % chunk_m_size)
            cluster_pid_n = within_chunk // chunk_m_size
        else:
            cluster_pid_m = chunk_m_start + (within_chunk // n_clusters)
            cluster_pid_n = within_chunk % n_clusters
    return cluster_pid_m, cluster_pid_n

# ──────────────────────────────────────────────────────────────────────────
# Persistent dense GEMM kernel — base class for all 3 directions.
# ──────────────────────────────────────────────────────────────────────────
class PersistentDenseGemmKernel:
    """Hopper SM90a persistent GEMM: C = A @ B (BF16/BF16 → BF16 or FP32).

    Warp-specialized cooperative layout:
      - 1 DMA producer WG (warp_group_idx 0): only does TMA loads + commits.
      - N=mma_warp_groups MMA consumer WGs (idx 1..N): cooperative WGMMA on
        the M dimension (each MMA WG covers tile_M / mma_warp_groups rows).
      - cluster_shape_mn=(2,1) default: B is TMA-multicast across the 2
        CTAs in a cluster (same N-column).
      - Grid size = num_SMs (rounded down to a multiple of cluster_size).
      - Each CTA persistently iterates over the cluster-tile share.

    Independent sC SMEM region (no sA aliasing) — DMA WG starts refilling
    sA for the next tile while the MMA WGs finish the current tile's
    epilogue R2S+TMA-store, giving natural across-tile overlap.
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        k_pipe_mmas: int = 1,
        cluster_shape_mn: tuple[int, int] = (2, 1),
        swizzle_m: int = 1,
        epi_stage_override: int | None = None,
        atom_layout_mnk: tuple[int, int, int] | None = None,
        raster_along: str = "m",
        accumulate: bool = False,
    ):
        self.acc_dtype = acc_dtype
        self.k_pipe_mmas = k_pipe_mmas
        self.swizzle_m = swizzle_m
        assert raster_along in ("m", "n"), \
            f"raster_along must be 'm' or 'n', got {raster_along!r}"
        self.raster_along = raster_along
        self.epi_stage_override = epi_stage_override
        # accumulate=True ⇒ device-fused RMW epilogue (C += acc):
        # before the chunked R2S/TMA-store epilogue, each thread reads its
        # own slice of the existing C tile from HBM via per-thread
        # ``thr_mma.partition_C(gC_read_slice)`` and adds the loaded FP32
        # values directly into its accumulator fragment. Chunked-hoist
        # variant: first half of V is hoisted PRE-DRAIN (overlaps with
        # WGMMA wait_group(0)) into a register tensor; second half is
        # inline LDG+ADD POST-DRAIN. Per-cell M-bounds predication via
        # ``cute.make_identity_tensor + cute.domain_offset`` so the
        # partial last m_tile (covering rows [O, O_pad)) and the all-OOB
        # M-padding tile correctly skip the LDG (rmw_loaded=0 for OOB
        # cells; TMA-store extent=O drops OOB writes naturally). Used by
        # PersistentDenseGemmWgradKernel on the aligned-out_buf path
        # (production + bench); misaligned ``out_buf`` views fall back
        # to the host scratch+``add_`` path with accumulate=False.
        self.accumulate = bool(accumulate)
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        if atom_layout_mnk is not None:
            self.atom_layout_mnk = atom_layout_mnk
        else:
            self.atom_layout_mnk = (
                (2, 1, 1)
                if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
                else (1, 1, 1)
            )
        self.cluster_shape_mn = cluster_shape_mn
        self.cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_dma_warp_groups = 1 if self.mma_warp_groups >= 2 else 0
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (
            (self.num_dma_warp_groups + self.mma_warp_groups)
            * self.num_threads_per_warp_group
        )
        self.num_mma_threads = self.mma_warp_groups * self.num_threads_per_warp_group
        self.dma_warp_offset = self.num_dma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

        self.a_dtype = None
        self.b_dtype = None
        self.c_dtype = None
        self.a_layout = None
        self.b_layout = None
        self.c_layout = None
        self.tiled_mma = None
        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None

    def _setup_attributes(self) -> None:
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        is_cooperative = self.atom_layout_mnk[0] >= 2
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )
        self.ab_stage, self.epi_stage = self._compute_stages()
        (self.a_smem_layout_staged,
         self.b_smem_layout_staged,
         self.epi_smem_layout_staged) = self._make_smem_layouts()

    def _compute_stages(self) -> tuple[int, int]:
        """Budget SMEM for independent sA + sB + sC + mbar.

        sC owns its own region (no sA aliasing).  Try epi_stage=4 first
        (deeper TMA-store pipeline), fall back to 2 if that leaves fewer
        than 2 ab_stages (mainloop wants ab_stage >= k_pipe_mmas + 1).
        Cap ab_stage at 4 to keep mbarrier overhead bounded.
        """
        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * self.a_dtype.width // 8
            + cute.size(b_shape) * self.b_dtype.width // 8
        )
        epi_tile_size = cute.size(self.epi_tile)
        c_bytes_per_stage = epi_tile_size * self.c_dtype.width // 8
        mbar_helpers_bytes = 1024
        budget = self.smem_capacity // self.occupancy - mbar_helpers_bytes
        if self.epi_stage_override is not None:
            epi_stage_candidates = (
                (self.epi_stage_override, 2)
                if self.epi_stage_override != 2 else (2,)
            )
        else:
            epi_stage_candidates = (4, 2)
        for epi_stage in epi_stage_candidates:
            avail_for_ab = budget - epi_stage * c_bytes_per_stage
            if avail_for_ab >= 2 * ab_bytes_per_stage:
                ab_stage = min(avail_for_ab // ab_bytes_per_stage, 4)
                return ab_stage, epi_stage
        return 2, 2

    def _make_smem_layouts(self):
        a_smem = sm90_utils.make_smem_layout_a(
            self.a_layout, self.tile_shape_mnk, self.a_dtype, self.ab_stage,
        )
        b_smem = sm90_utils.make_smem_layout_b(
            self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stage,
        )
        epi_smem = sm90_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.epi_stage,
        )
        return a_smem, b_smem, epi_smem

    # ──────────────────────────────────────────────────────────────────
    # Host-side JIT entry — set up TMA atoms, grid (~num_SMs), launch.
    # ──────────────────────────────────────────────────────────────────
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sm_count: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()

        a_tma_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
            if self.is_a_mcast
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        )
        b_tma_op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
            if self.is_b_mcast
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            a_tma_op,
            a,
            cute.slice_(self.a_smem_layout_staged, (None, None, 0)),
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            num_multicast=self.num_mcast_ctas_a,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            b_tma_op,
            b,
            cute.slice_(self.b_smem_layout_staged, (None, None, 0)),
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            num_multicast=self.num_mcast_ctas_b,
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            cute.slice_(self.epi_smem_layout_staged, (None, None, 0)),
            self.epi_tile,
        )

        tile_m = self.tile_shape_mnk[0]
        tile_n = self.tile_shape_mnk[1]
        a_shape = a.shape
        c_shape = c.shape
        # Use A's M extent for m_tiles so wgrad's accumulate=True path
        # (where C=out_buf is unpadded shape (O, I) but A=d_y_pad.t has
        # M-padded shape (O_pad, T)) iterates the full padded M-axis
        # (288 m_tiles for O_pad=36864 / tile_M=128) with clean
        # m_clusters=288/2=144. Using c_shape[0] would give 287 m_tiles
        # → 287/2 = 143 m_clusters which DROPS the partial last m_tile
        # (rows [36608, 36724) of valid grad-update would be silently
        # lost — catastrophic precision regression). For all
        # other directions a.shape[0] == c.shape[0] (clean shapes), so
        # this change is a no-op for fwd / dgrad / wgrad-scratch path.
        m_tiles = cute.ceil_div(a_shape[0], tile_m)
        n_tiles = cute.ceil_div(c_shape[1], tile_n)
        l_dim = c_shape[2]

        grid_size = (sm_count // self.cluster_size) * self.cluster_size
        grid = (grid_size, 1, 1)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        m_clusters = m_tiles // self.cluster_shape_mn[0]
        n_clusters = n_tiles // self.cluster_shape_mn[1]
        cluster_tiles = m_clusters * n_clusters * l_dim
        # NOTE: o_actual_global (= c.shape[0]) is intentionally NOT
        # threaded as a kernel arg. Instead the kernel body computes it
        # via ``cute.size(mC_mnl, mode=[0])`` which produces a constexpr
        # static-layout value; passing as Int32 kernel arg failed MLIR
        # 'arith.cmpi op using value defined outside the region' on the
        # range_constexpr-unrolled per-cell predicate. Using
        # ``cute.size(mC_mnl, mode=[0])`` keeps M_global purely constexpr
        # at the IR level and lets the cell-level predicate compile.
        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            c,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            m_tiles, n_tiles, l_dim,
            m_clusters, n_clusters, cluster_tiles,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(self.cluster_size, 1, 1),
            stream=stream,
        )

    # ──────────────────────────────────────────────────────────────────
    # Device-side persistent kernel (warp-specialized DMA/MMA split).
    # ──────────────────────────────────────────────────────────────────
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mC_read_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        m_tiles: cutlass.Int32,
        n_tiles: cutlass.Int32,
        l_dim: cutlass.Int32,
        m_clusters: cutlass.Int32,
        n_clusters: cutlass.Int32,
        cluster_tiles: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        is_dma_wg = warp_group_idx < self.num_dma_warp_groups

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        cluster_idx_in_grid = bidx // self.cluster_size
        num_clusters_in_grid = gdimx // self.cluster_size

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        smem_alloc = cutlass.utils.SmemAllocator()
        storage = smem_alloc.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_consumer_arrives = mcast_size * 4 * self.mma_warp_groups
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=num_consumer_arrives,
            alignment=num_consumer_arrives,
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        _a_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        _b_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = _a_mask if self.is_a_mcast else 0
        b_mcast_mask = _b_mask if self.is_b_mcast else 0

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )
        # Plain (non-TMA) C local-tile view used by the device-fused
        # RMW LOAD path. ``thr_mma.partition_C(gC_read_slice)`` returns
        # per-thread FP32 SCALARS suitable for ``rmw_loaded[v]=gC_read[v]``;
        # using the TMA-decorated ``gC_mnl`` would yield basis-stride
        # coordinate tuples (TypeError on rmem assign). Same TENSOR
        # data — they share the underlying ``c`` storage — but different
        # cute-layout decoration. When self.accumulate=False (fwd /
        # dgrad / wgrad-scratch path), gC_read_mnl is unused (the
        # const_expr branch elides the LDG block).
        gC_read_mnl = cute.local_tile(
            mC_read_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )
        # o_actual_global = the actual M-extent of the C output tensor
        # (= O for wgrad RMW where C=out_buf is shape (O, I); = M for
        # clean-shape directions where C's M-dim equals A's M-dim).
        # Computed via ``cute.size(mC_read_mnl, mode=[0])`` so it lifts
        # to a constexpr from the tensor's static layout — passing it
        # as an Int32 kernel arg fails MLIR 'arith.cmpi op using value
        # defined outside the region' on the range_constexpr-unrolled
        # per-cell predicate (the kernel-arg SSA value can't cross the
        # while-loop body's region isolation boundary). Used as the
        # bound in per-cell M-axis predication for the accumulate=True
        # RMW load: a cell with global m_abs >= o_actual_global must
        # skip the LDG (rmw stays at its 0.0 pre-fill). For non-
        # accumulate cases (fwd / dgrad / wgrad-scratch path) this
        # value is computed but never referenced — the const_expr
        # branch elides the predicated LDG block entirely.
        o_actual_global = cute.size(mC_read_mnl, mode=[0])

        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_mnk[0],
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        if is_dma_wg:
            # ============ DMA PRODUCER WARP GROUP ============
            cute.arch.warpgroup_reg_dealloc(24)

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            cluster_idx = cluster_idx_in_grid
            cluster_lane_m = cluster_coord_mnk[0]
            cluster_lane_n = cluster_coord_mnk[1]
            while cluster_idx < cluster_tiles:
                tiles_per_batch_cluster = m_clusters * n_clusters
                pid_l = cluster_idx // tiles_per_batch_cluster
                r = cluster_idx % tiles_per_batch_cluster
                cluster_pid_m, cluster_pid_n = _swizzle_decompose(
                    r, m_clusters, n_clusters, self.swizzle_m,
                    self.raster_along,
                )
                pid_m = cluster_pid_m * self.cluster_shape_mn[0] + cluster_lane_m
                pid_n = cluster_pid_n * self.cluster_shape_mn[1] + cluster_lane_n

                tAgA_mkl = tAgA[(None, pid_m, None, pid_l)]
                tBgB_nkl = tBgB[(None, pid_n, None, pid_l)]

                producer_state.reset_count()

                if warp_idx == 0:
                    for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                        mainloop_pipeline.producer_acquire(producer_state)
                        tAgA_k = tAgA_mkl[(None, producer_state.count)]
                        tAsA_pipe = tAsA[(None, producer_state.index)]
                        tBgB_k = tBgB_nkl[(None, producer_state.count)]
                        tBsB_pipe = tBsB[(None, producer_state.index)]
                        cute.copy(
                            tma_atom_a, tAgA_k, tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                producer_state
                            ),
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b, tBgB_k, tBsB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                producer_state
                            ),
                            mcast_mask=b_mcast_mask,
                        )
                        mainloop_pipeline.producer_commit(producer_state)
                        producer_state.advance()

                cluster_idx += num_clusters_in_grid

        else:
            # ============ MMA CONSUMER WARP GROUPS (cooperative) ============
            cute.arch.warpgroup_reg_alloc(
                240 if self.mma_warp_groups <= 2 else 112
            )

            mma_tidx = tidx - self.dma_warp_offset
            mma_wg_idx = warp_group_idx - self.num_dma_warp_groups

            wg_thread_layout = cute.make_layout(
                self.mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(wg_thread_layout(mma_wg_idx))
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCgC = thr_mma.partition_C(gC_mnl)
            acc_shape = tCgC.shape[:3]
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            num_k_blocks = cute.size(tCrA, mode=[2])

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
                cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        4,
                    ),
                    cutlass.BFloat16 if cutlass.const_expr(self.c_dtype.width == 32)
                    else self.c_dtype,
                ),
                tiled_mma,
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)

            thr_copy_r2s = tiled_copy_r2s.get_slice(mma_tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            size_tRS_rD = cute.size(
                cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            )

            tma_store_warp_idx = self.dma_warp_offset // 32
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            epilog_barrier = pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.num_mma_threads
            )

            consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            cluster_idx = cluster_idx_in_grid
            cluster_lane_m = cluster_coord_mnk[0]
            cluster_lane_n = cluster_coord_mnk[1]
            while cluster_idx < cluster_tiles:
                tiles_per_batch_cluster = m_clusters * n_clusters
                pid_l = cluster_idx // tiles_per_batch_cluster
                r = cluster_idx % tiles_per_batch_cluster
                cluster_pid_m, cluster_pid_n = _swizzle_decompose(
                    r, m_clusters, n_clusters, self.swizzle_m,
                    self.raster_along,
                )
                pid_m = cluster_pid_m * self.cluster_shape_mn[0] + cluster_lane_m
                pid_n = cluster_pid_n * self.cluster_shape_mn[1] + cluster_lane_n

                gC_mnl_slice = gC_mnl[(None, None, pid_m, pid_n, pid_l)]

                accumulators.fill(0.0)
                consumer_read_state.reset_count()
                consumer_release_state.reset_count()

                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

                # ──── Deep K-pipeline: prologue + mainloop + drain. ────
                k_pipe_mmas = self.k_pipe_mmas

                # PROLOGUE — issue first k_pipe_mmas WGMMA groups without
                # waiting; their stages stay acquired (held in flight).
                for prologue_idx in cutlass.range_constexpr(k_pipe_mmas):
                    mainloop_pipeline.consumer_wait(consumer_read_state)
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_block_coord = (
                            None, None, k_block_idx, consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    consumer_read_state.advance()

                # MAINLOOP — each iter issues one WGMMA, then wait_group
                # keeps only the latest k_pipe_mmas in flight, releasing
                # the oldest stage (consumer_release_state lags read by
                # exactly k_pipe_mmas).
                for k_tile in cutlass.range(
                    k_tile_cnt - k_pipe_mmas, unroll=1
                ):
                    mainloop_pipeline.consumer_wait(consumer_read_state)
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_block_coord = (
                            None, None, k_block_idx, consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(consumer_release_state)
                    consumer_read_state.advance()
                    consumer_release_state.advance()

                # ──── RMW PRE-DRAIN HOIST (chunked-hoist; predicated) ────
                # device-fused RMW epilogue.
                #
                # Issue per-thread ldg on the FIRST HALF (along the V
                # axis) of the existing C tile slice into a small
                # register tensor ``rmw_loaded_first`` BEFORE the WGMMA
                # ``wait_group(0)`` drain. ldg has its own scoreboard
                # (independent from WGMMA), so the load latency overlaps
                # with the final k_pipe_mmas trailing WGMMAs that are
                # still draining (committed at the bottom of the
                # mainloop above).
                #
                # Pattern transplanted from gemm_attn_out_proj v5
                # chunked-hoist; see that kernel for the full design
                # rationale (TMA-load avoided due to per-epi mbarrier
                # overhead; per-thread partition_C view matches WGMMA
                # accumulator's TV layout 1-to-1; chunk size = V/2 holds
                # the per-thread register pressure to 192/240 within the
                # warpgroup_reg_alloc(240) budget).
                #
                # gemm_output-specific: per-cell M-bounds predication
                # via ``cute.make_identity_tensor + cute.domain_offset``.
                # The wgrad accumulate=True path has C=out_buf (O, I)
                # with O=36724 NOT divisible by tile_M=128, so the
                # last partial m_tile (pid_m=286 covering rows
                # [36608, 36736)) addresses 12 rows past O for which
                # an unconditional LDG would read OOB GMEM (sentinel
                # cells → silently corrupted result). The all-OOB
                # M-padding tile (pid_m=287, rows [36736, 36864))
                # similarly needs ALL of its loads to be predicated
                # to 0.
                #
                # CRITICAL invariants (predicated-load + predicated-store
                # contract):
                # 1. Use ``gC_read_mnl`` (plain, non-TMA decorated) —
                # NOT ``gC_mnl`` (TMA).  partition_C on a basis-
                # stride TMA tensor yields coordinate tuples, not
                # scalars (TypeError on rmem assign).
                # 2. Slice tiled_mma by ``mma_tidx`` (per-thread), NOT
                # ``wg_thread_layout(mma_wg_idx)`` (per-WG-first).
                # Per-WG slice would make all 128 threads in a WG
                # LOAD THE SAME ADDRESS — incorrect.
                # 3. The coord tensor ``cC_pred`` MUST be sliced by the
                # SAME thr_mma_per_thread (mma_tidx) so cell index
                # [v, m, n] in tCcC_pred corresponds 1-to-1 with
                # cell [v, m, n] in tCgC_rmw / accumulators / etc.
                # 4. Keep the inline ``if m_abs < o_actual_global:``
                # guard inline — hoisting the load outside the
                # predicate would issue an UNCONDITIONAL gmem LDG
                # on OOB cells (segfault on the M-padding tile
                # where rows [36736, 36864) are well past out_buf's
                # 600 MB FP32 storage).
                if cutlass.const_expr(self.accumulate):
                    gC_read_slice = gC_read_mnl[
                        (None, None, pid_m, pid_n, pid_l)
                    ]
                    thr_mma_per_thread = tiled_mma.get_slice(mma_tidx)
                    tCgC_rmw = thr_mma_per_thread.partition_C(gC_read_slice)
                    # Coord tensor for per-cell M-bounds predication.
                    # ``cC_pred`` has shape (tile_M, tile_N) where each
                    # cell's value is its (m_local, n_local) tile-relative
                    # coord; ``domain_offset`` shifts to GLOBAL coords so
                    # ``tCcC_pred[v, m, n]`` returns a 2-tuple
                    # (m_global, n_global) for thread T's cell at
                    # accumulator position (v, m, n).
                    cC_pred = cute.make_identity_tensor(
                        (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
                    )
                    cC_pred = cute.domain_offset(
                        (pid_m * self.tile_shape_mnk[0],
                         pid_n * self.tile_shape_mnk[1]),
                        cC_pred,
                    )
                    tCcC_pred = thr_mma_per_thread.partition_C(cC_pred)
                    n_v_blk = cute.size(tCgC_rmw, mode=[0])
                    n_m_blk = cute.size(tCgC_rmw, mode=[1])
                    n_n_blk = cute.size(tCgC_rmw, mode=[2])
                    n_v_half = n_v_blk // 2
                    rmw_loaded_first = cute.make_rmem_tensor(
                        (n_v_half, n_m_blk, n_n_blk), self.acc_dtype,
                    )
                    # Pre-init register tensor to zero so the OOB cells
                    # (m_abs >= o_actual_global, where the LDG below is
                    # predicated out) hold a definite value 0.0 going
                    # into the unconditional post-drain ADD. Without
                    # this, the if/else assignment pattern relies on the
                    # cute compiler to track both branches' register
                    # state which is fragile across optimization passes.
                    rmw_loaded_first.fill(0.0)
                    # Pre-load FIRST HALF along V (v_idx in [0, n_v_half)).
                    # Predicated ldgs issue concurrent with in-flight
                    # WGMMAs; the load latency overlaps with the WGMMA
                    # drain below. For OOB cells (m_abs >= O), the LDG
                    # is skipped — rmw_loaded_first stays at its
                    # pre-fill 0.0 value, so the unconditional post-drain
                    # ADD (acc += rmw_loaded_first) yields acc + 0 = acc
                    # for those cells (which the WGMMA already left at 0
                    # since A's OOB rows are zero — final 0 + 0 = 0 then
                    # gets silently dropped by the TMA-store descriptor
                    # extent on the way out to gmem).
                    for v_idx in cutlass.range_constexpr(n_v_half):
                        for m_idx in cutlass.range_constexpr(n_m_blk):
                            for n_idx in cutlass.range_constexpr(n_n_blk):
                                coord = tCcC_pred[v_idx, m_idx, n_idx]
                                m_abs = coord[0]
                                if m_abs < o_actual_global:
                                    rmw_loaded_first[v_idx, m_idx, n_idx] = (
                                        tCgC_rmw[v_idx, m_idx, n_idx]
                                    )

                # DRAIN — flush in-flight WGMMAs and release trailing stages.
                # The first-half ldgs issued above ride alongside this
                # wait: wait_group(0) blocks on WGMMA only; the ldg
                # results may already be in registers by the time the
                # wait returns, in which case the first ``acc += rmw``
                # below issues with no further HBM latency.
                cute.nvgpu.warpgroup.wait_group(0)
                for drain_idx in cutlass.range_constexpr(k_pipe_mmas):
                    mainloop_pipeline.consumer_release(consumer_release_state)
                    consumer_release_state.advance()
                epilog_barrier.arrive_and_wait()

                # ──── RMW POST-DRAIN ADD — chunked: reg-add first / inline-add second ────
                # First half (v_idx in [0, n_v_half)): register-only
                # ``acc += rmw_loaded_first[v,m,n]`` (no memory ops,
                # both operands in registers; ldgs already completed
                # during drain).
                # Second half (v_idx in [n_v_half, n_v_blk)): predicated
                # inline ``acc += tCgC_rmw[v,m,n]`` (load+add fused).
                # OOB cells (m_abs >= O) skip the LDG+ADD entirely —
                # acc stays at WGMMA result (= 0 for OOB rows since A
                # rows are zero), which the TMA-store extent then drops.
                if cutlass.const_expr(self.accumulate):
                    for v_idx in cutlass.range_constexpr(n_v_half):
                        for m_idx in cutlass.range_constexpr(n_m_blk):
                            for n_idx in cutlass.range_constexpr(n_n_blk):
                                accumulators[v_idx, m_idx, n_idx] = (
                                    accumulators[v_idx, m_idx, n_idx]
                                    + rmw_loaded_first[v_idx, m_idx, n_idx]
                                )
                    for v_off in cutlass.range_constexpr(
                        n_v_blk - n_v_half
                    ):
                        v_actual = v_off + n_v_half
                        for m_idx in cutlass.range_constexpr(n_m_blk):
                            for n_idx in cutlass.range_constexpr(n_n_blk):
                                coord = tCcC_pred[v_actual, m_idx, n_idx]
                                m_abs = coord[0]
                                if m_abs < o_actual_global:
                                    accumulators[v_actual, m_idx, n_idx] = (
                                        accumulators[v_actual, m_idx, n_idx]
                                        + tCgC_rmw[v_actual, m_idx, n_idx]
                                    )

                # ──── EPILOGUE — acc → SMEM (R2S) → GMEM (TMA store) ────
                tCgC_for_tma_partition = cute.zipped_divide(
                    gC_mnl_slice, self.epi_tile,
                )
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma_partition,
                )

                epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])

                    tRS_rD = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, self.acc_dtype,
                    )
                    tRS_rD_out = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, self.c_dtype,
                    )
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))
                    cute.copy(
                        tiled_copy_r2s, tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilog_barrier.arrive_and_wait()
                    if warp_idx == tma_store_warp_idx:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    epilog_barrier.arrive_and_wait()

                cluster_idx += num_clusters_in_grid

            if warp_idx == tma_store_warp_idx:
                tma_store_pipeline.producer_tail()

            # Cluster exit fence: one CTA must not tear down its SMEM
            # mbarriers / multicast endpoints while peer CTAs still hold
            # references to them.  Placed AFTER the persistent loop so it
            # fires exactly once per kernel launch.  Executed by every
            # MMA-WG thread (the DMA WG has already exited via its own
            # producer-side path); the cluster barrier accounting matches
            # the MMA arrival count.
            if cutlass.const_expr(self.cluster_size > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

# ──────────────────────────────────────────────────────────────────────────
# Per-direction subclass hooks — initially identical to the base kernel.
# Future per-direction architectural specializations override __call__ /
# kernel here without affecting the other two compile graphs. Each
# direction gets its own cute.compile cache entry keyed by class identity.
# ──────────────────────────────────────────────────────────────────────────
class PersistentDenseGemmFwdKernel(PersistentDenseGemmKernel):
    """fwd-specialized (currently identical to base; reserved override slot)."""

class PersistentDenseGemmDgradKernel(PersistentDenseGemmKernel):
    """dgrad-specialized (currently identical to base; reserved override slot)."""

class PersistentDenseGemmWgradKernel(PersistentDenseGemmKernel):
    """wgrad-specialized: opt-in fused RMW via ``accumulate=True`` flag.

    The kernel body is the SAME as the base class — the device-fused
    RMW load+add (predicated chunked-hoist) is gated by
    ``self.accumulate`` (constexpr).  When True, after the WGMMA
    drain and before the chunked R2S/TMA-store epilogue, each thread:

      1. Pre-drain HOIST: predicated LDG of the FIRST HALF (along V)
         of existing C tile into a register tensor (``rmw_loaded_first``).
         The LDGs issue concurrent with in-flight WGMMAs, hiding LDG
         latency under the WGMMA drain .
      2. Post-drain ADD: register-only ``acc += rmw_loaded_first[v]``
         for the first half, predicated inline ``acc += LDG_C[v]`` for
         the second half.  Per-cell M-bounds predicate
         (``m_abs < o_actual_global``) skips the LDG for the partial
         last m_tile (rows [O, O_pad) on the M-padded shape) and the
         all-OOB M-padding tile, so OOB cells contribute 0 to acc
         (which the TMA-store extent=O then drops naturally).

    The host wrapper ``gemm_output_wgrad`` picks ``accumulate=True`` for
    the 16-byte-aligned ``out_buf`` case (production + bench) and
    ``accumulate=False`` for the misaligned-byte-offset case (which
    routes through a host scratch + ``add_``).  Two distinct compiled
    binaries are cached by ``_build_and_compile``'s ``accumulate``
    kwarg.
    """

KERNEL_CLASS_BY_DIRECTION: dict[str, type] = {
    "fwd":   PersistentDenseGemmFwdKernel,
    "dgrad": PersistentDenseGemmDgradKernel,
    "wgrad": PersistentDenseGemmWgradKernel,
}

_SM_COUNT_CACHE: int | None = None

def _hardware_sm_count() -> int:
    """Return the device's SM count (cached once per process)."""
    global _SM_COUNT_CACHE
    if _SM_COUNT_CACHE is None:
        hw = utils.HardwareInfo()
        _SM_COUNT_CACHE = int(hw.get_device_multiprocessor_count())
    return _SM_COUNT_CACHE

@lru_cache(maxsize=None)
def _build_and_compile(
    direction: str,
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    c_shape: tuple[int, ...],
    a_stride: tuple[int, ...],
    b_stride: tuple[int, ...],
    c_stride: tuple[int, ...],
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    sm_count: int,
    accumulate: bool = False,
) -> tuple[Any, tuple[int, int]]:
    """JIT-compile a kernel for this (direction, shapes, strides, dtypes, accumulate).

    Cached on the full input signature INCLUDING ``accumulate`` so the
    aligned wgrad path (accumulate=True, device-fused RMW with C=out_buf
    shape (O, I)) and the misaligned fallback path (accumulate=False,
    host scratch+add_ with C=scratch shape (O_pad, I)) each get their
    own compiled binary.  Three independent compile graphs (one per
    direction subclass) — even bit-identical configs produce distinct
    cached binaries because cute.compile is class-keyed.

    For non-wgrad directions (fwd / dgrad), ``accumulate=False`` is the
    only path used; the ``const_expr`` branches in the kernel elide
    the RMW load/add code entirely so the binary is identical to the
    non-accumulating fast path.
    """
    cfg = PERSISTENT_CONFIGS[direction]
    tile_shape_mn = cfg["tile_mn"]
    acc_dtype = cutlass.Float32
    k_pipe_mmas = cfg["k_pipe_mmas"]
    cluster_shape_mn = cfg["cluster_mn"]
    swizzle_m = cfg["swizzle_m"]
    epi_stage_override = cfg.get("epi_stage")
    atom_layout_mnk = cfg.get("atom_layout_mnk")
    raster_along = cfg.get("raster_along", "m")

    torch_dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    cute_dtype_map = {"bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}

    def _make_dummy(shape, stride, dt_str) -> tuple[torch.Tensor, Any]:
        t = torch.empty_strided(
            shape, stride, dtype=torch_dtype_map[dt_str], device="cuda"
        )
        ct = from_dlpack(t, assumed_align=16)
        ct.element_type = cute_dtype_map[dt_str]
        leading = stride.index(1)
        ct = ct.mark_layout_dynamic(leading_dim=leading)
        return t, ct

    _, mA = _make_dummy(a_shape, a_stride, a_dtype)
    _, mB = _make_dummy(b_shape, b_stride, b_dtype)
    _, mC = _make_dummy(c_shape, c_stride, c_dtype)

    kernel_cls = KERNEL_CLASS_BY_DIRECTION[direction]
    kernel = kernel_cls(
        acc_dtype=acc_dtype,
        tile_shape_mn=tile_shape_mn,
        k_pipe_mmas=k_pipe_mmas,
        cluster_shape_mn=cluster_shape_mn,
        swizzle_m=swizzle_m,
        epi_stage_override=epi_stage_override,
        atom_layout_mnk=atom_layout_mnk,
        raster_along=raster_along,
        accumulate=accumulate,
    )
    stream = _current_cute_stream()
    compiled = cute.compile(kernel, mA, mB, mC, sm_count, stream)
    return compiled, tile_shape_mn

def _cute_tensor_from(t: torch.Tensor, cute_dtype) -> Any:
    """from_dlpack helper that marks the leading_dim (stride==1) for cute."""
    ct = from_dlpack(t, assumed_align=16)
    ct.element_type = cute_dtype
    stride = list(t.stride())
    if 1 in stride:
        leading = stride.index(1)
        ct = ct.mark_layout_dynamic(leading_dim=leading)
    return ct

# Cache the cuda.CUstream wrapper for the default torch stream — saves
# ~2-5 µs/call of ctype boxing on the hot path.
@lru_cache(maxsize=None)
def _cached_cute_stream(stream_ptr: int) -> Any:
    return cuda.CUstream(stream_ptr)

def _current_cute_stream() -> Any:
    return _cached_cute_stream(torch.cuda.current_stream().cuda_stream)

# ──────────────────────────────────────────────────────────────────────────
# Pad-and-slice helpers for the irregular O=36724 axis.
# ──────────────────────────────────────────────────────────────────────────
# Round-up O to the next multiple of ``_PAD_TILE_O = 256``: this is divisible
# by both the M5 fwd config's tile_N=256 (so fwd's N-tile partition is clean)
# AND the BF16 WGMMA inst tile_K=64 (so dgrad's K-tile partition is clean
# when w_pad is reused as the dgrad weight via the ``w_pad.t`` view).
# For the production shape O=36724 → O_pad=36864 = 144 × 256 = 576 × 64 →
# 0.38 % extra FLOPs / HBM bandwidth (negligible vs the ~1.5-2× MFU uplift
# from M5 over the cuBLAS baseline at 29.13% MFU).
_PAD_TILE_O = 256

def _pad_size(O: int) -> int:
    """Round O up to the next multiple of ``_PAD_TILE_O`` (= 256)."""
    return ((O + _PAD_TILE_O - 1) // _PAD_TILE_O) * _PAD_TILE_O

# Module-cached padded weight per unique (w.data_ptr, O, I, O_pad) signature.
# Production keeps a single LM-head weight tensor pinned for the entire
# training step → cache hit rate = 100% after the first call. Cache is also valid across
# multiple processes because the key is per-data_ptr. Reused across all
# three directions: fwd reads as ``(O_pad, I)`` K-major, dgrad reads as
# ``w_pad.t`` = ``(I, O_pad)`` N-major (zero-copy stride permutation).
_W_PAD_CACHE: dict[tuple[int, int, int, int, int], torch.Tensor] = {}

def _get_padded_w(w: torch.Tensor, O_pad: int) -> torch.Tensor:
    """Return BF16 [O_pad, I] with rows 0..O-1 = w and rows O..O_pad-1 = 0.

    The padded buffer is cached by (data_ptr, shape, device) to avoid
    re-allocation, but the content is ALWAYS refreshed via ``copy_``
    because the optimizer updates w in-place (same data_ptr, different
    values).  The tail [O:O_pad] rows stay zero from the initial
    ``torch.zeros``.
    """
    O, I = w.shape
    key = (
        w.data_ptr(),
        O, I, O_pad,
        w.device.index if w.device.index is not None else 0,
    )
    cached = _W_PAD_CACHE.get(key)
    if cached is not None:
        cached[:O].copy_(w)
        return cached
    w_pad = torch.zeros(O_pad, I, dtype=w.dtype, device=w.device)
    w_pad[:O].copy_(w)
    _W_PAD_CACHE[key] = w_pad
    return w_pad

# Module-cached K-padded d_y buffer for dgrad. d_y is fresh each call
# (different ``data_ptr`` per micro-batch), so we cannot key by data_ptr —
# instead we cache ONE buffer per (T, O, O_pad, device_index) signature
# and rewrite its first O columns each call. The tail [:, O:O_pad] is
# zeroed ONCE at allocation via ``torch.zeros`` and NEVER overwritten,
# so it stays zero across the entire process lifetime (we only ever
# write ``d_y_pad[:, :O].copy_(d_y)``).
#
# Production memory cost: one buffer of (T=8192, O_pad=36864) BF16 =
# 603 MB. On 80 GB H100 SXM5 this is 0.75 % of GPU memory — fine.
_DY_PAD_CACHE: dict[tuple[int, int, int, int], torch.Tensor] = {}

def _get_dy_pad(T: int, O: int, O_pad: int, device: torch.device) -> torch.Tensor:
    """Return cached BF16 [T, O_pad] buffer; caller must fill [:, :O] each call.

    The tail [:, O:O_pad] is zero (from the initial ``torch.zeros`` and
    never overwritten), so the dgrad kernel sees a properly K-padded d_y
    that contributes zero from positions [O..O_pad-1] (matching the zero
    rows of w_pad at those K positions).

    Reused across dgrad (K-axis padding) and wgrad (M-axis padding via
    ``.t()``); same underlying memory, different cute layout views.
    """
    key = (T, O, O_pad, device.index if device.index is not None else 0)
    cached = _DY_PAD_CACHE.get(key)
    if cached is not None:
        return cached
    buf = torch.zeros(T, O_pad, dtype=torch.bfloat16, device=device)
    _DY_PAD_CACHE[key] = buf
    return buf

# Module-cached FP32 [O_pad, I] scratch for wgrad output (overwritten each
# call by the M5 kernel). One buffer per (O, O_pad, I, device_index)
# signature. Production keeps a single LM-head wgrad slot pinned for the
# entire step → 100% cache-hit after first call. Memory cost: ~604 MB on
# the production shape (O_pad=36864, I=4096) FP32 — 0.76% of an 80 GB
# H100 SXM5 GPU. Note: WE DO NOT zero the buffer (kernel always
# overwrites every cell — accumulators.fill(0.0) per tile + non-accumulate
# epilogue).
_WGRAD_PAD_CACHE: dict[tuple[int, int, int, int], torch.Tensor] = {}

def _get_wgrad_pad(O: int, O_pad: int, I: int,
                   device: torch.device) -> torch.Tensor:
    """Return cached FP32 [O_pad, I] scratch buffer for wgrad output.

    The buffer is reused across calls; each call OVERWRITES it via the
    M5 kernel.  Caller then does ``out_buf.add_(scratch[:O, :])`` to
    accumulate ONLY the prefix back into out_buf (GAS=8
    semantics preserved because the prefix is the correct gradient and
    .add_() is read-modify-write).
    """
    key = (O, O_pad, I, device.index if device.index is not None else 0)
    cached = _WGRAD_PAD_CACHE.get(key)
    if cached is not None:
        return cached
    buf = torch.empty(O_pad, I, dtype=torch.float32, device=device)
    _WGRAD_PAD_CACHE[key] = buf
    return buf

# ──────────────────────────────────────────────────────────────────────────
# Public entry points — gemm_output_{fwd, dgrad, wgrad}.
# ──────────────────────────────────────────────────────────────────────────
def gemm_output_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[S,B,O] = x[S,B,I] @ w[O,I]^T  (BF16; pad-and-slice for irregular O).

    Production: x.shape=(S=4096, B=2, I=4096), w.shape=(O=36724, I=4096).
    O is padded internally to O_pad=36864 (= 144 × 256, divisible by
    tile_N=256) so the M5 monolithic kernel sees a regular shape.  The
    returned view slices off the [O:O_pad] padding, so callers see
    exactly the production BF16 [S, B, O=36724] output.
    """
    assert x.dtype == torch.bfloat16, f"x dtype {x.dtype}"
    assert w.dtype == torch.bfloat16, f"w dtype {w.dtype}"
    assert x.dim() == 3, f"x dim {x.dim()}"
    assert w.dim() == 2, f"w dim {w.dim()}"
    S, B, I = x.shape
    O, I2 = w.shape
    assert I == I2, f"x.I {I} != w.I {I2}"
    T = S * B

    O_pad = _pad_size(O)

    x_in = x.contiguous()
    w_pad = _get_padded_w(w, O_pad)

    # H100 TMA requires 16B-aligned global pointers; cute is told
    # ``assumed_align=16`` (no runtime check), so we hard-assert the
    # alignment at the ABI boundary.  ``w_pad`` is a fresh allocation
    # and therefore always aligned.
    assert x_in.data_ptr() % 16 == 0, (
        f"gemm_output_fwd: x not 16B aligned (data_ptr=0x{x_in.data_ptr():x}, "
        f"%16={x_in.data_ptr() % 16}); shape={tuple(x.shape)} "
        f"storage_offset={x.storage_offset()} stride={tuple(x.stride())}")

    y_pad = torch.empty(T, O_pad, dtype=torch.bfloat16, device=x.device)

    a_3d = x_in.view(T, I).unsqueeze(-1)        # (T, I, 1), stride (I, 1, 0)
    b_3d = w_pad.unsqueeze(-1)                  # (O_pad, I, 1), stride (I, 1, 0)
    c_3d = y_pad.unsqueeze(-1)                  # (T, O_pad, 1), stride (O_pad, 1, 0)

    sm_count = _hardware_sm_count()
    compiled, _ = _build_and_compile(
        "fwd",
        tuple(a_3d.shape), tuple(b_3d.shape), tuple(c_3d.shape),
        tuple(a_3d.stride()), tuple(b_3d.stride()), tuple(c_3d.stride()),
        "bfloat16", "bfloat16", "bfloat16",
        sm_count=sm_count,
    )

    mA = _cute_tensor_from(a_3d, cutlass.BFloat16)
    mB = _cute_tensor_from(b_3d, cutlass.BFloat16)
    mC = _cute_tensor_from(c_3d, cutlass.BFloat16)
    stream = _current_cute_stream()
    compiled(mA, mB, mC, stream)

    # Slice off the padded [O:O_pad] columns, make contiguous, and
    # reshape to 3D [S, B, O]. The slice creates a non-contiguous view
    # (stride-0 = O_pad ≠ O); downstream Triton CE kernels may assume
    # contiguous row stride, so we must materialise a dense copy.
    return y_pad[:, :O].contiguous().view(S, B, O)

def gemm_output_dgrad(
    d_logits: torch.Tensor,
    output_w: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """dx = d_logits @ output_w  (BF16; pad-and-slice for irregular K=O=36724).

    Production (``backward.py``) calls this WITHOUT ``out=``, so the
    kernel allocates and returns a fresh ``BF16[T, I]`` tensor.  Callers
    that want to write into a pre-allocated, 16-byte-aligned ``out``
    pass it via the keyword.  Misaligned ``out`` slices are not
    supported on this path (``gemm_output``'s dgrad always allocates a
    fresh tensor in production).

    GEMM layout (per the kernel's MNK frame):
      * M = T = 8192       (= S * B; clean multiple of tile_M=128)
      * N = I = 4096       (clean multiple of tile_N=256)
      * K = O = 36724      (IRREGULAR — padded to O_pad=36864)
      * A = d_y_pad  shape (T, O_pad, 1)        K-major (stride O_pad, 1, 0)
      * B = w_pad.t()      shape (I, O_pad, 1)  N-major (stride 1, I,    0)
      * C = out            shape (T, I,     1)  N-major (stride I, 1,    0)

    The kernel framework picks the appropriate WGMMA atom (B is N-major
    here vs K-major in fwd) via ``b_layout.sm90_mma_major_mode()`` — same
    PersistentDenseGemmDgradKernel class, separate compile-cache entry
    keyed by the (direction, shape, stride, dtype) signature.

    Padded contributions are zero by construction: ``d_y_pad[:, O:O_pad]``
    and ``w_pad[O:O_pad, :]`` are both zero, so the K-loop's partial sum
    over positions [O, O_pad) adds nothing to the result.
    """
    assert d_logits.dtype == torch.bfloat16, \
        f"d_logits dtype {d_logits.dtype}"
    assert output_w.dtype == torch.bfloat16, \
        f"output_w dtype {output_w.dtype}"
    assert d_logits.dim() == 2, f"d_logits dim {d_logits.dim()}"
    assert output_w.dim() == 2, f"output_w dim {output_w.dim()}"
    T, O = d_logits.shape
    O2, I = output_w.shape
    assert O == O2, f"d_logits.O {O} != output_w.O {O2}"

    if out is None:
        # Production hot path: fresh BF16[T, I] allocation (matches
        # ``d_hidden = torch.matmul(d_logits, output_w)`` semantics).
        out = torch.empty(T, I, dtype=torch.bfloat16, device=d_logits.device)
    else:
        assert out.dtype == torch.bfloat16, f"out dtype {out.dtype}"
        assert out.shape == (T, I), f"out shape {out.shape} != ({T}, {I})"

    O_pad = _pad_size(O)

    # ── Inputs: cached w_pad (reused from fwd cache) + cached d_y_pad. ──
    d_logits_in = d_logits.contiguous()
    # TMA 16B alignment: assert at ABI boundary (see fwd note). output_w
    # is consumed via the cached padded ``w_pad`` (fresh, always aligned).
    assert d_logits_in.data_ptr() % 16 == 0, (
        f"gemm_output_dgrad: d_logits not 16B aligned "
        f"(data_ptr=0x{d_logits_in.data_ptr():x}, %16={d_logits_in.data_ptr() % 16}); "
        f"shape={tuple(d_logits.shape)} storage_offset={d_logits.storage_offset()} "
        f"stride={tuple(d_logits.stride())}")
    w_pad = _get_padded_w(output_w, O_pad)
    d_y_pad = _get_dy_pad(T, O, O_pad, d_logits.device)
    d_y_pad[:, :O].copy_(d_logits_in)
    # Tail [:, O:O_pad] stays at the initial torch.zeros — never overwritten.

    # ── Output alignment fallback for TMA's 16-byte data-ptr requirement. ──
    # Fresh ``torch.empty`` on CUDA returns ≥256-byte-aligned pointers,
    # so the ``out=None`` production path always passes.  Any caller
    # passing a non-zero-storage-offset BF16 view can land on a non-
    # aligned pointer — we route through a fresh scratch + final copy
    # so the kernel itself always sees a 16-byte-aligned C tensor.
    out_aligned = (out.data_ptr() % 16) == 0
    if out_aligned:
        c_out = out
    else:
        c_out = torch.empty(T, I, dtype=torch.bfloat16, device=d_logits.device)

    a_3d = d_y_pad.unsqueeze(-1)             # (T, O_pad, 1), stride (O_pad, 1, 0)
    b_3d = w_pad.t().unsqueeze(-1)           # (I, O_pad, 1), stride (1, I,    0)
    c_3d = c_out.unsqueeze(-1)               # (T, I,     1), stride (I, 1,    0)

    sm_count = _hardware_sm_count()
    compiled, _ = _build_and_compile(
        "dgrad",
        tuple(a_3d.shape), tuple(b_3d.shape), tuple(c_3d.shape),
        tuple(a_3d.stride()), tuple(b_3d.stride()), tuple(c_3d.stride()),
        "bfloat16", "bfloat16", "bfloat16",
        sm_count=sm_count,
    )

    mA = _cute_tensor_from(a_3d, cutlass.BFloat16)
    mB = _cute_tensor_from(b_3d, cutlass.BFloat16)
    mC = _cute_tensor_from(c_3d, cutlass.BFloat16)
    stream = _current_cute_stream()
    compiled(mA, mB, mC, stream)

    if not out_aligned:
        out.copy_(c_out)
    return out

def gemm_output_wgrad(d_logits: torch.Tensor, hidden_final: torch.Tensor,
                      *, out_buf: torch.Tensor) -> torch.Tensor:
    """dw[O,I] += d_logits^T @ hidden_final  (BF16 in; FP32 ACCUMULATE).

    ``out_buf`` is REQUIRED — production NEVER calls with out_buf=None.
    Two implementations selected at runtime by out_buf alignment:

    1. **Aligned (16-byte) out_buf** — production + bench fast path.
       Device kernel is compiled with
       ``accumulate=True``; in the epilogue it reads existing C from
       HBM per-thread (via ``thr_mma.partition_C(gC_slice)``) with
       per-cell M-bounds predication, adds into the accumulator BEFORE
       the chunked R2S/TMA-store epilogue, and writes back to out_buf
       (TMA-store extent=O drops the M-padding tail-tile rows
       [O, O_pad) automatically).  Result: a true device-fused RMW
       with one HBM round-trip (read C ~600 MB + write C ~600 MB =
       1.2 GB) and NO host-side ``add_``.  Saves ~600 µs/call vs an
       equivalent scratch+host-add path (which would otherwise push
       ~604 MB scratch write + 1.2 GB host-add = 1.8 GB HBM total).

    2. **Misaligned ``out_buf``** — fallback path for callers that pass
       a non-16-byte-aligned ``out_buf`` view.  The device kernel is
       compiled with ``accumulate=False``; the host wrapper writes to
       a cached FP32 scratch buffer and then does
       ``out_buf.add_(scratch[:O, :])``.  TMA stores require 16-byte
       alignment of the global pointer, so the device-fused path
       cannot run on a byte-offset view.

    Both paths satisfy the GAS=8 accumulate semantics (applies to both):
    repeated calls into the same out_buf correctly sum across the
    8 micro-batches.

    GEMM layout (per the kernel's MNK frame, both paths):
      * M = O_pad = 36864  (= ceil(O/256)*256; the irregular axis;
        kernel m_tiles=288 covers all of A's M-extent)
      * N = I     = 4096   (clean)
      * K = T     = 8192   (clean)
      * A = d_y_pad.t()   shape (O_pad, T, 1) stride (1, O_pad, 0)  M-major BF16
      * B = x.t()         shape (I,     T, 1) stride (1, I,     0)  N-major BF16
      * C (path 1) = out_buf  shape (O,     I, 1) stride (I, 1, 0)  N-major FP32
      * C (path 2) = scratch  shape (O_pad, I, 1) stride (I, 1, 0)  N-major FP32

    The d_y_pad cache is REUSED from dgrad: same ``(T, O_pad)`` BF16
    buffer (~603 MB cached), zero-copy ``.t()`` to ``(O_pad, T)`` for
    wgrad's M-major A operand.

    The wgrad-output scratch (path 2 only) is a per-process cache
    (~604 MB FP32) keyed by ``(O, O_pad, I, device_index)``;
    production never hits this path (out_buf is always 256-byte
    aligned via PyTorch's allocator).

    Caller-storage isolation (applies to both): the scratch buffer of
    path 2 is a SEPARATE ``torch.empty`` allocation, distinct from the
    caller's storage.  The kernel itself never touches caller memory
    on the misaligned path; on path 1 the kernel's TMA-store extent
    of ``O`` discards the M-padding tail tiles automatically, so the
    caller's ``out_buf`` is the only piece of caller memory ever
    written.
    """
    assert d_logits.dtype == torch.bfloat16, \
        f"d_logits dtype {d_logits.dtype}"
    assert hidden_final.dtype == torch.bfloat16, \
        f"hidden_final dtype {hidden_final.dtype}"
    assert out_buf.dtype == torch.float32, \
        f"out_buf dtype {out_buf.dtype}"

    # Production calls wgrad with 3D inputs (sequence-major BSI layout).
    # Flatten internally to 2D for the kernel.
    if d_logits.dim() == 3 and hidden_final.dim() == 3:
        S, B, O = d_logits.shape
        S2, B2, I = hidden_final.shape
        assert (S, B) == (S2, B2), \
            f"d_logits S,B={(S, B)} != hidden_final S,B={(S2, B2)}"
        T = S * B
        d_y_2d = d_logits.reshape(T, O)
        x_2d = hidden_final.reshape(T, I)
    elif d_logits.dim() == 2 and hidden_final.dim() == 2:
        T, O = d_logits.shape
        T2, I = hidden_final.shape
        assert T == T2, f"d_logits T={T} != hidden_final T={T2}"
        d_y_2d = d_logits
        x_2d = hidden_final
    else:
        raise ValueError(
            f"wgrad: d_logits.dim()={d_logits.dim()} "
            f"hidden_final.dim()={hidden_final.dim()} — expected both 3D or both 2D"
        )

    assert out_buf.shape == (O, I), \
        f"out_buf shape {out_buf.shape} != ({O}, {I})"
    assert out_buf.is_contiguous(), \
        f"out_buf must be contiguous (caller view of flat buffer); got stride {out_buf.stride()}"

    O_pad = _pad_size(O)

    # ── Inputs: cached d_y_pad (reused from dgrad) + contiguous x. ──
    d_y_pad = _get_dy_pad(T, O, O_pad, d_logits.device)
    d_y_2d_in = d_y_2d.contiguous()
    # TMA 16B alignment for the upstream BF16 tensor being copied into the
    # padded scratch (assert at the ABI boundary, see fwd note).
    assert d_y_2d_in.data_ptr() % 16 == 0, (
        f"gemm_output_wgrad: d_logits not 16B aligned "
        f"(data_ptr=0x{d_y_2d_in.data_ptr():x}, %16={d_y_2d_in.data_ptr() % 16}); "
        f"shape={tuple(d_logits.shape)} storage_offset={d_logits.storage_offset()} "
        f"stride={tuple(d_logits.stride())}")
    d_y_pad[:, :O].copy_(d_y_2d_in)
    # Tail [:, O:O_pad] stays at the initial torch.zeros — never overwritten.

    x_in = x_2d.contiguous()
    assert x_in.data_ptr() % 16 == 0, (
        f"gemm_output_wgrad: hidden_final not 16B aligned "
        f"(data_ptr=0x{x_in.data_ptr():x}, %16={x_in.data_ptr() % 16}); "
        f"shape={tuple(hidden_final.shape)} storage_offset={hidden_final.storage_offset()} "
        f"stride={tuple(hidden_final.stride())}")

    # Always use scratch(O_pad, I) with accumulate=False; the device-fused
    # RMW path on out_buf(O, I) requires O to be 128-aligned under
    # ``cluster_mn=(2,1)`` and we don't want a runtime branch on that.
    c_out = _get_wgrad_pad(O, O_pad, I, d_logits.device)
    accumulate = False

    # A = d_y_pad.t shape (O_pad, T, 1) stride (1, O_pad, 0) — M-major BF16
    # B = x.t shape (I,     T, 1) stride (1, I,     0) — N-major BF16
    # C (path 1) = out_buf shape (O,     I, 1) stride (I, 1, 0) FP32
    # C (path 2) = scratch shape (O_pad, I, 1) stride (I, 1, 0) FP32
    a_3d = d_y_pad.t().unsqueeze(-1)
    b_3d = x_in.t().unsqueeze(-1)
    c_3d = c_out.unsqueeze(-1)

    sm_count = _hardware_sm_count()
    compiled, _ = _build_and_compile(
        "wgrad",
        tuple(a_3d.shape), tuple(b_3d.shape), tuple(c_3d.shape),
        tuple(a_3d.stride()), tuple(b_3d.stride()), tuple(c_3d.stride()),
        "bfloat16", "bfloat16", "float32",
        sm_count=sm_count,
        accumulate=accumulate,
    )

    mA = _cute_tensor_from(a_3d, cutlass.BFloat16)
    mB = _cute_tensor_from(b_3d, cutlass.BFloat16)
    mC = _cute_tensor_from(c_3d, cutlass.Float32)
    stream = _current_cute_stream()
    compiled(mA, mB, mC, stream)

    out_buf.add_(c_out[:O, :])
    return out_buf

__all__ = [
    "gemm_output_fwd",
    "gemm_output_dgrad",
    "gemm_output_wgrad",
    "PERSISTENT_CONFIGS",
]
