"""Hopper (SM90a) Flash-Attention forward kernel in Python CuTe DSL.

Fixed shape: B=10, H=16, N=4096, D=64, FP16 in / FP32 accumulate.

Architecture (Phase-1 target — all four techniques present):
  T1 WGMMA        -- async tensor-core MMA for QK and PV
  T2 TMA          -- cpasync tiled TMA for Q / K / V / O
  T3 Warp-spec    -- 1 producer warp-group (warps 0..3) + 1 consumer WG (4..7)
  T4 Multi-stage  -- PipelineTmaAsync with num_stages >= 2 for K and V

Correctness target: both is_causal=False and is_causal=True pass.
"""
import math
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.pipeline as cutlass_pipeline
import cutlass.utils.hopper_helpers as sm90_utils_basic

from . import utils
from . import hopper_helpers
from . import pipeline as fa_pipeline
from .softmax import Softmax


class FlashAttnFwdSm90:
    """SM90-targeted flash-attention forward kernel.

    One CTA computes one (m_block, head_idx, batch_idx) output tile. Grid =
    (n_m_blocks, num_heads, num_batches). Each CTA has 2 warpgroups:
    producer (TMA issuer) and consumer (WGMMA + softmax + TMA store).
    """

    arch: int = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric] = cutlass.Float16,
        head_dim: int = 64,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_stages: int = 2,
        is_causal: bool = False,
        num_producer_regs: int = 24,
        num_mma_regs: int = 232,
    ):
        assert head_dim % 8 == 0
        assert m_block_size == 64, "WGMMA M-tile on SM90 is fixed to 64"
        assert n_block_size % 16 == 0
        assert num_stages >= 2
        self.dtype = dtype
        self.head_dim = head_dim
        self.head_dim_v = head_dim
        self.m_block = m_block_size
        self.n_block = n_block_size
        self.num_stages = num_stages
        self.is_causal = is_causal
        self.num_producer_regs = num_producer_regs
        self.num_mma_regs = num_mma_regs
        # Thread layout: 1 producer WG + 1 consumer WG = 256 threads.
        self.num_threads = 256
        self.num_mma_threads = 128
        self.num_producer_threads = 128

    # ------------------------------------------------------------------
    # Layouts, MMAs, shared storage
    # ------------------------------------------------------------------

    def _make_smem_layouts(self):
        """GMMA-swizzled smem layouts for Q, K, V (stage axis), and O."""
        q_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim
            ),
            self.dtype,
        )
        v_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_v
            ),
            self.dtype,
        )
        sQ = cute.tile_to_shape(q_atom, (self.m_block, self.head_dim), (0, 1))
        sK = cute.tile_to_shape(
            q_atom, (self.n_block, self.head_dim, self.num_stages), (0, 1, 2)
        )
        sV = cute.tile_to_shape(
            v_atom, (self.n_block, self.head_dim_v, self.num_stages), (0, 1, 2)
        )
        sO = cute.tile_to_shape(v_atom, (self.m_block, self.head_dim_v), (0, 1))
        return sQ, sK, sV, sO

    def _make_tiled_mmas(self):
        """Two TiledMMAs: QK (SS, K-major/K-major) and PV (RS, K-major/MN-major)."""
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.m_block, self.n_block),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.m_block, self.head_dim_v),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _make_shared_storage_cls(self, sQ_layout, sK_layout, sV_layout):
        dtype = self.dtype

        sQ_bytes = cute.cosize(sQ_layout)
        sK_bytes = cute.cosize(sK_layout)
        sV_bytes = cute.cosize(sV_layout)

        sQ_struct = cute.struct.Align[cute.struct.MemRange[dtype, sQ_bytes], 1024]
        sK_struct = cute.struct.Align[cute.struct.MemRange[dtype, sK_bytes], 128]
        sV_struct = cute.struct.Align[cute.struct.MemRange[dtype, sV_bytes], 128]

        mbar_Q_struct = cute.struct.MemRange[cutlass.Int64, 1]
        mbar_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorage:
            mbar_Q: mbar_Q_struct
            mbar_K: mbar_K_struct
            mbar_V: mbar_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        return SharedStorage

    # ------------------------------------------------------------------
    # Host launcher
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        # Re-layout tensors: torch gives (B, H, N, D), kernel wants
        # last-dim = D (contig). We transpose into (N, D, H, B) so that
        # tensor[..., head, batch] slices out a (N, D) per-head matrix.
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mK, mV, mO)
        ]
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[2, 3, 1, 0]))
            for t in (mQ, mK, mV, mO)
        ]
        # mLSE from (B, H, N) -> (N, H, B)
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
            if const_expr(mLSE is not None)
            else None
        )

        sQ_layout, sK_layout, sV_layout, sO_layout = self._make_smem_layouts()
        tiled_mma_qk, tiled_mma_pv = self._make_tiled_mmas()

        tma_G2S = cpasync.CopyBulkTensorTileG2SOp()
        tma_S2G = cpasync.CopyBulkTensorTileS2GOp()

        self.tma_q_bytes = cute.size_in_bytes(self.dtype, cute.select(sQ_layout, mode=[0, 1]))
        self.tma_k_bytes = cute.size_in_bytes(self.dtype, cute.select(sK_layout, mode=[0, 1]))
        self.tma_v_bytes = cute.size_in_bytes(self.dtype, cute.select(sV_layout, mode=[0, 1]))

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_G2S, mQ, sQ_layout, (self.m_block, self.head_dim)
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_G2S,
            mK,
            cute.select(sK_layout, mode=[0, 1]),
            (self.n_block, self.head_dim),
            1,  # no multicast
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            tma_G2S,
            mV,
            cute.select(sV_layout, mode=[0, 1]),
            (self.n_block, self.head_dim_v),
            1,
        )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            tma_S2G, mO, sO_layout, (self.m_block, self.head_dim_v)
        )

        SharedStorage = self._make_shared_storage_cls(sQ_layout, sK_layout, sV_layout)

        n_m_blocks = cute.ceil_div(mQ.shape[0], self.m_block)
        grid = (n_m_blocks, cute.size(mQ.shape[2]), cute.size(mQ.shape[3]))

        LOG2_E = math.log2(math.e)
        scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            scale_log2,
            sQ_layout,
            sK_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------
    # Kernel body
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors from a single warp.
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Q mbarrier — Q is loaded once per CTA, single-slot transaction barrier.
        mbar_ptr_Q = storage.mbar_Q.data_ptr()
        if warp_idx == 1:
            cute.arch.mbarrier_init(mbar_ptr_Q, 1)

        prod_group = cutlass_pipeline.CooperativeGroup(cutlass_pipeline.Agent.Thread)
        cons_group = cutlass_pipeline.CooperativeGroup(
            cutlass_pipeline.Agent.Thread, self.num_mma_threads // 128
        )
        pipeline_k = fa_pipeline.TmaPipelineNoCluster.create(
            barrier_storage=storage.mbar_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=self.tma_k_bytes,
            init_wait=False,
        )
        pipeline_v = fa_pipeline.TmaPipelineNoCluster.create(
            barrier_storage=storage.mbar_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=prod_group,
            consumer_group=cons_group,
            tx_count=self.tma_v_bytes,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = utils.transpose_smem_view(sV)
        # sO re-uses sQ's smem slab — sQ is dead by epilogue.
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sO_layout.inner, dtype=sQ.element_type),
            sO_layout.outer,
        )

        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(
                mQ, mK, mV,
                sQ, sK, sV,
                tma_atom_Q, tma_atom_K, tma_atom_V,
                pipeline_k, pipeline_v,
                mbar_ptr_Q,
            )
        else:
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128  # consumer-local thread index in [0, 128)
            self.mma(
                tiled_mma_qk, tiled_mma_pv,
                mQ, mK, mO, mLSE,
                sQ, sK, sVt, sO,
                tma_atom_O,
                pipeline_k, pipeline_v,
                mbar_ptr_Q,
                tidx,
                scale_log2,
            )

    # ------------------------------------------------------------------
    # Producer: issue Q once, then stream K/V via pipeline.
    # ------------------------------------------------------------------

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: fa_pipeline.TmaPipelineNoCluster,
        pipeline_v: fa_pipeline.TmaPipelineNoCluster,
        mbar_ptr_Q: cutlass.Pointer,
    ):
        warp_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_in_wg == 0:
            m_block, head_idx, batch_idx = cute.arch.block_idx()

            mQ_c = mQ[None, None, head_idx, batch_idx]
            mK_c = mK[None, None, head_idx, batch_idx]
            mV_c = mV[None, None, head_idx, batch_idx]

            gQ = cute.local_tile(mQ_c, (self.m_block, self.head_dim), (m_block, 0))
            gK = cute.local_tile(mK_c, (self.n_block, self.head_dim), (None, 0))
            gV = cute.local_tile(mV_c, (self.n_block, self.head_dim_v), (None, 0))

            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q, 0, cute.make_layout(1),
                cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K, 0, cute.make_layout(1),
                cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V, 0, cute.make_layout(1),
                cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2),
            )

            # Kick off the Q TMA. expect_tx arms the mbarrier with the byte count.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_Q, self.tma_q_bytes)
            cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=mbar_ptr_Q)

            n_block_max = cute.size(mK.shape[0]) // self.n_block
            if const_expr(self.is_causal):
                n_block_max = ((m_block + 1) * self.m_block + self.n_block - 1) // self.n_block

            kv_state = fa_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Producer, self.num_stages
            )

            # Walk N blocks in reverse so the causal-masked tile is first.
            for i in cutlass.range(n_block_max, unroll=2):
                n_block = n_block_max - i - 1
                pipeline_k.producer_acquire(kv_state)
                cute.copy(
                    tma_atom_K,
                    tKgK[None, n_block], tKsK[None, kv_state.index],
                    tma_bar_ptr=pipeline_k.producer_get_barrier(kv_state),
                )
                pipeline_v.producer_acquire(kv_state)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, n_block], tVsV[None, kv_state.index],
                    tma_bar_ptr=pipeline_v.producer_get_barrier(kv_state),
                )
                kv_state.advance()

    # ------------------------------------------------------------------
    # Consumer: WGMMA QK, softmax, WGMMA PV, epilogue.
    # ------------------------------------------------------------------

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_k: fa_pipeline.TmaPipelineNoCluster,
        pipeline_v: fa_pipeline.TmaPipelineNoCluster,
        mbar_ptr_Q: cutlass.Pointer,
        tidx: Int32,
        scale_log2: Float32,
    ):
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        # Consumer-wide view (single WG → warp-group slice at thread 0 covers
        # the operand partitioning; accumulator is per-thread via tidx).
        wg_layout = cute.make_layout(1, stride=128)
        wg_slice_qk = tiled_mma_qk.get_slice(wg_layout(0))
        wg_slice_pv = tiled_mma_pv.get_slice(wg_layout(0))
        thr_slice_qk = tiled_mma_qk.get_slice(tidx)

        tSrQ = tiled_mma_qk.make_fragment_A(wg_slice_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_slice_qk.partition_B(sK))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_slice_pv.partition_B(sVt))

        acc_S_shape = tiled_mma_qk.partition_shape_C((self.m_block, self.n_block))
        acc_O_shape = tiled_mma_pv.partition_shape_C((self.m_block, self.head_dim_v))
        tOrP = cute.make_fragment(
            utils.make_rs_frgA_layout(cute.make_layout(acc_S_shape)), self.dtype
        )
        acc_O = cute.make_fragment(acc_O_shape, Float32)

        softmax = Softmax(scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1])
        softmax.reset()

        # Wait for Q to land in smem.
        cute.arch.mbarrier_wait(mbar_ptr_Q, phase=0)

        n_block_max = cute.size(mK.shape[0]) // self.n_block
        if const_expr(self.is_causal):
            n_block_max = ((m_block + 1) * self.m_block + self.n_block - 1) // self.n_block

        kv_state = fa_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Consumer, self.num_stages
        )

        # -------- prologue: first tile (QK -> mask -> softmax init) --------
        acc_S = cute.make_fragment(acc_S_shape, Float32)
        pipeline_k.consumer_wait(kv_state)
        hopper_helpers.wgmma_gemm(
            tiled_mma_qk, acc_S, tSrQ, tSrK[None, None, None, kv_state.index],
            zero_init=True, wg_wait=0,
        )
        pipeline_k.consumer_release(kv_state)
        if const_expr(self.is_causal):
            self._apply_causal_mask(acc_S, m_block, n_block_max - 1, thr_slice_qk)
        softmax.online_softmax(acc_S, is_first=True, check_inf=True)
        tOrP_src = cute.make_tensor(acc_S.iterator, utils.make_rs_frgA_layout(acc_S.layout))
        tOrP.store(tOrP_src.load().to(self.dtype))
        # Initialize acc_O to zero so every PV uses zero_init=False; this keeps
        # the WGMMA ACCUMULATE flag static (and lets cutlass.range trace cleanly).
        acc_O.fill(0.0)

        # -------- body: QK(i) + PV(i-1) overlapped via wait_group(1) ------
        # Order MATTERS: issue QK (new tile) first, then PV (prev tile); wait_group(1)
        # then drains QK (older group) while PV stays in flight so softmax can run
        # on acc_S in parallel.
        n_body = n_block_max - 1
        for n_tile in cutlass.range(n_body, unroll=1):
            kv_state = self._body_one_n_block(
                tiled_mma_qk, tiled_mma_pv,
                tSrQ, tSrK, tOrVt,
                acc_O, tOrP, softmax,
                pipeline_k, pipeline_v,
                kv_state, acc_S_shape,
            )

        # -------- epilogue PV for the last tile --------
        pipeline_v.consumer_wait(kv_state)
        hopper_helpers.wgmma_gemm(
            tiled_mma_pv, acc_O, tOrP,
            tOrVt[None, None, None, kv_state.index],
            zero_init=False, wg_wait=0,
        )
        pipeline_v.consumer_release(kv_state)

        # Final softmax normalization.
        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        # Export per-row LSE to global memory (softmax.row_sum contains LSE
        # in natural log after finalize). Pre-zeroed buffer + atomicAdd from
        # one lane per quad ensures correct single-write semantics.
        if const_expr(mLSE is not None):
            gLSE = mLSE[None, head_idx, batch_idx]
            cS_lse = cute.make_identity_tensor((self.m_block, self.n_block))
            tScS_lse = thr_slice_qk.partition_C(cS_lse)
            tScS_lse_mn = utils.make_acc_mn_view(tScS_lse)
            nrows_lse = cute.size(softmax.row_sum)
            if tidx % 4 == 0:
                for r in cutlass.range_constexpr(nrows_lse):
                    row_idx = m_block * self.m_block + tScS_lse_mn[r, 0][0]
                    ptr = utils.elem_pointer(gLSE, row_idx)
                    utils.atomic_add_fp32(softmax.row_sum[r], ptr)

        # Epilogue: registers -> sO (via stmatrix) -> gmem (via TMA store).
        self._epilogue(acc_O, mO, sO, tma_atom_O, tiled_mma_pv, tidx, m_block, head_idx, batch_idx)

    # ------------------------------------------------------------------

    @cute.jit
    def _body_one_n_block(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrVt: cute.Tensor,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        softmax: Softmax,
        pipeline_k: fa_pipeline.TmaPipelineNoCluster,
        pipeline_v: fa_pipeline.TmaPipelineNoCluster,
        kv_state: fa_pipeline.PipelineStateSimple,
        acc_S_shape: cutlass.Constexpr,
    ):
        """One body iteration of the intra-warpgroup-overlapped main loop.

        Layout of work per call:
            QK (new tile) -> PV (prev tile) -> wait_group(1) -> softmax(acc_S)
            -> wait_group(0) -> rescale_O -> cast P.

        Returns the advanced consumer pipeline state.
        """
        kv_state_v = kv_state.clone()
        kv_state.advance()
        # Issue new QK.
        pipeline_k.consumer_wait(kv_state)
        acc_S_next = cute.make_fragment(acc_S_shape, Float32)
        hopper_helpers.wgmma_gemm(
            tiled_mma_qk, acc_S_next, tSrQ,
            tSrK[None, None, None, kv_state.index],
            zero_init=True, wg_wait=-1,
        )
        # Issue PV for previous tile.
        pipeline_v.consumer_wait(kv_state_v)
        hopper_helpers.wgmma_gemm(
            tiled_mma_pv, acc_O, tOrP,
            tOrVt[None, None, None, kv_state_v.index],
            zero_init=False, wg_wait=-1,
        )
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(kv_state)
        row_scale = softmax.online_softmax(acc_S_next, is_first=False, check_inf=False)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(kv_state_v)
        softmax.rescale_O(acc_O, row_scale)
        tOrP_src = cute.make_tensor(
            acc_S_next.iterator, utils.make_rs_frgA_layout(acc_S_next.layout)
        )
        tOrP.store(tOrP_src.load().to(self.dtype))
        return kv_state

    @cute.jit
    def _apply_causal_mask(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_slice_qk,
    ):
        cS = cute.make_identity_tensor((self.m_block, self.n_block))
        tScS = thr_slice_qk.partition_C(cS)
        tScS_mn = utils.make_acc_mn_view(tScS)
        acc_mn = utils.make_acc_mn_view(acc_S)
        nrows = cute.size(acc_mn.shape[0])
        ncols = cute.size(acc_mn.shape[1])
        for r in cutlass.range_constexpr(nrows):
            for c in cutlass.range_constexpr(ncols):
                row_idx = m_block * self.m_block + tScS_mn[r, c][0]
                col_idx = n_block * self.n_block + tScS_mn[r, c][1]
                if col_idx > row_idx:
                    acc_mn[r, c] = -Float32.inf

    # ------------------------------------------------------------------

    @cute.jit
    def _epilogue(
        self,
        acc_O: cute.Tensor,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # rO = (FP16) acc_O
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))

        # acc_O -> sO via stmatrix
        smem_copy_atom = utils.get_smem_store_atom_sm90(self.dtype)
        smem_copy = cute.make_tiled_copy_C(smem_copy_atom, tiled_mma_pv).get_slice(tidx)
        taccOrO = smem_copy.retile(rO)
        taccOsO = smem_copy.partition_D(sO)
        cute.copy(smem_copy_atom, taccOrO, taccOsO)

        # Ensure smem stores are visible to TMA, then one warp issues the TMA store.
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )
        cute.arch.barrier_arrive(barrier_id=0, number_of_threads=self.num_mma_threads + 32)

        mO_c = mO[None, None, head_idx, batch_idx]
        gO = cute.local_tile(mO_c, (self.m_block, self.head_dim_v), (m_block, 0))
        tOsO, tOgO = cpasync.tma_partition(
            tma_atom_O, 0, cute.make_layout(1),
            cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2),
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(barrier_id=0, number_of_threads=self.num_mma_threads + 32)
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
