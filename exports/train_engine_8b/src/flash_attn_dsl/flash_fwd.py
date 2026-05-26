"""Flash Attention Forward (SM90) — Python CuTe DSL kernel.

M0.6 1D persistent grid (3 warpgroups, 384 threads) for:
    B=2, H_q=16, H_kv=1 (GQA 16:1), S=4096, D=128, BF16, causal

Architecture (M0.6 — 1 producer WG + 2 MMA WGs, BM=BN=128, persistent):
- 3 warpgroups per CTA (384 threads):
    * producer WG (warp_group_idx 0, 128 thr): warp 0 streams TMA loads,
      warps 1-3 reg_dealloc(24) and fall through to exit.
    * consumer WGs (warp_group_idx 1 and 2, 256 thr total): reg_alloc(232),
      each MMA WG owns one 64-row M-slice of the BM=128 tile and runs the
      full MMA + softmax + epilogue path in parallel. No cross-WG named
      barrier in the body.
- WGMMA SS-mode for QK; WGMMA RS-mode for PV.
  TiledMma atom_layout_mnk=(2,1,1): 2 atoms along M, one per MMA WG.
- TMA bulk-tensor G2S for K, V via two independent
  PipelineTmaAsyncNoCluster instances (separate mbarrier rings).
  Empty-barrier arrival count = 2 (one per MMA WG via tidx % 128 == 0).
- Q via cp.async, issued by the 256 consumer threads.
- num_stages=2 ring buffer on K/V.
- sO ≡ sV smem alias (sV is last read by the final PV WGMMA in the
  body loop; epilogue R2S-then-TMA writes sO into the same region).
- Online softmax (exp2 space, FP32 accumulate). Each MMA WG runs its own
  per-thread softmax over its 64-row slice; row state never crosses WGs.
- Causal masking (n_block_max clamp + per-tile mask).
- Epilogue: each MMA WG R2S-writes its 64 rows of sO, then a single
  consumer warp (warp_idx == 4) issues the TMA bulk store for the full
  128-row sO tile.
- LSE write via quad-gated scalar fp32 stores (each MMA WG writes the
  64 LSE entries owned by its M-slice).
- 1D persistent grid: 132 CTAs (one per H100 SM), tile-stride loop in
  kernel decodes tile_id → (b, h_q, m_block) in h_q-major order so the
  16 q-heads sharing the same (b, h_kv) appear on consecutive tile_ids
  (L2 reuse of K/V across q-heads). CTA-invariant setup (smem alloc,
  TMA descriptor prefetch, pipeline create + mbarrier init, WGMMA
  partitions, fragments, R2S copy atom, gs_copy for Q) is hoisted ABOVE
  the tile loop; only per-tile gmem views, TMA closures, n_block_max,
  Q-load partition, body, and epilogue stay inside.
- Synchronization: producer ↔ consumer only via K/V mbarrier rings;
  WG-local named barrier (id=1, 256 threads) syncs the 2 MMA WGs.
  CTA-wide tile-end barrier (id=2, 384 threads) at the bottom of every
  tile-iter prevents the next tile's producer from refilling sV slot 0
  (which aliases sO) while the current tile's bulk-store-O is still
  draining shared memory.

Smem footprint: sQ(32 KB) + sK(64 KB) + sV≡sO(64 KB) = 160 KB / CTA → 1 CTA/SM
Register footprint: 24*128 (producer) + 232*256 (2× consumer) ≈ 62 KB / CTA

class FlashAttnFwdSm90:
    __init__(dtype, head_dim, m_block_size, n_block_size, num_stages,
             is_causal, qhead_per_kvhead)
    __call__(mQ, mK, mV, mO, mLSE, softmax_scale, stream)
"""
import math
from typing import Optional, Type

try:
    import cuda.bindings.driver as cuda
except ModuleNotFoundError:
    import cuda.cuda as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from quack import sm90_utils, copy_utils, layout_utils
from flash_attn_dsl import utils as fa_utils
from flash_attn_dsl.pipeline import (
    PipelineTmaAsyncNoCluster,
    make_pipeline_state as make_simple_pipeline_state,
)
from flash_attn_dsl.softmax import Softmax
from flash_attn_dsl.hopper_helpers import gemm as wgmma_gemm


M_BLOCK = 128
N_BLOCK = 128
NUM_STAGES = 2
# Warp-spec: 1 producer WG (128 thr) + 2 MMA WGs (256 thr) = 384 thr / CTA.
# Producer = warp_group_idx 0 (warp 0 issues TMA, warps 1-3 reg_dealloc and
# fall through); Consumer = warp_group_idx 1 and 2 (each reg_alloc(232) and
# handles one 64-row M-slice of BM=128 in parallel via atom_layout (2,1,1)).
NUM_THREADS = 384
PRODUCER_THREADS = 128
CONSUMER_THREADS = 256
PRODUCER_REG_COUNT = 24
CONSUMER_REG_COUNT = 232
# Persistent grid: H100 has 132 SMs and we run 1 CTA/SM.
NUM_PERSISTENT_CTAS = 132


class FlashAttnFwdSm90:
    """SM90 FA forward kernel — naive per-tile grid baseline."""

    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        head_dim: int = 128,
        m_block_size: int = M_BLOCK,
        n_block_size: int = N_BLOCK,
        num_stages: int = NUM_STAGES,
        is_causal: bool = False,
        qhead_per_kvhead: int = 1,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.m_block_size = M_BLOCK
        self.n_block_size = N_BLOCK
        self.num_stages = NUM_STAGES
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.num_threads = NUM_THREADS
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.sQ_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR,
            (self.m_block_size, self.tile_hdim),
        )
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR,
            (self.n_block_size, self.tile_hdim),
            stage=self.num_stages,
        )
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR,
            (self.n_block_size, self.tile_hdim),
            stage=self.num_stages,
        )
        self.sO_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR,
            (self.m_block_size, self.tile_hdim),
        )

    def _get_tiled_mma(self):
        # WGMMA atom is fixed at M=64; for BM>64 with a single warpgroup we
        # leave atom_layout_mnk = (1,1,1) (1 WG) and let `permutation_mnk`
        # logical_divide the M-tile into multiple atom-sized slices that
        # cute.gemm iterates serially. Each thread then holds
        # (BM / 32) rows of acc along M. PV uses the same trick on hdim.
        qk_atom = cute.make_mma_atom(
            warpgroup.MmaF16BF16Op(
                self.dtype, Float32,
                (64, self.n_block_size, 16),
                warpgroup.OperandSource.SMEM,
                warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            )
        )
        tiled_mma_qk = cute.make_tiled_mma(
            qk_atom,
            atom_layout_mnk=(2, 1, 1),
            permutation_mnk=(self.m_block_size, self.n_block_size, 16),
        )
        pv_atom = cute.make_mma_atom(
            warpgroup.MmaF16BF16Op(
                self.dtype, Float32,
                (64, self.tile_hdim, 16),
                warpgroup.OperandSource.RMEM,
                warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            )
        )
        tiled_mma_pv = cute.make_tiled_mma(
            pv_atom,
            atom_layout_mnk=(2, 1, 1),
            permutation_mnk=(self.m_block_size, self.tile_hdim, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_size = cute.cosize(self.sQ_layout)
        sK_size = cute.cosize(self.sK_layout)
        # sV holds a 2-stage K-block-shaped ring (cosize = 2 × BN × D);
        # sO (single buffer, cosize = BM × D) aliases the same region.
        # sO_cosize ≤ sV_cosize is guaranteed because BM = BN here.
        sV_size = max(cute.cosize(self.sV_layout), cute.cosize(self.sO_layout))
        dtype = self.dtype
        align = self.buffer_align_bytes
        from cutlass import Int64
        mbar_pairs_K = 2 * self.num_stages
        mbar_pairs_V = 2 * self.num_stages

        @cute.struct
        class SharedStorage:
            mbar_ptr_K: cute.struct.MemRange[Int64, mbar_pairs_K]
            mbar_ptr_V: cute.struct.MemRange[Int64, mbar_pairs_V]
            sQ: cute.struct.Align[cute.struct.MemRange[dtype, sQ_size], align]
            sK: cute.struct.Align[cute.struct.MemRange[dtype, sK_size], align]
            sV: cute.struct.Align[cute.struct.MemRange[dtype, sV_size], align]

        return SharedStorage

    # ------------------------------------------------------------------
    # Host entry
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
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()

        mQ = fa_utils.assume_tensor_aligned(mQ)
        mK = fa_utils.assume_tensor_aligned(mK)
        mV = fa_utils.assume_tensor_aligned(mV)
        mO = fa_utils.assume_tensor_aligned(mO)
        mLSE = fa_utils.assume_tensor_aligned(mLSE)

        # Permute mK/mV from [B, H_kv, S, D] to [S, D, H_kv, B] for TMA
        mK_tma = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[2], mK.shape[3], mK.shape[1], mK.shape[0]),
                stride=(mK.stride[2], mK.stride[3], mK.stride[1], mK.stride[0]),
            ),
        )
        mV_tma = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (mV.shape[2], mV.shape[3], mV.shape[1], mV.shape[0]),
                stride=(mV.stride[2], mV.stride[3], mV.stride[1], mV.stride[0]),
            ),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK_tma,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.n_block_size, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV_tma,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.n_block_size, self.tile_hdim),
        )

        # Permute mO from [B, H_q, S, D] to [S, D, H_q, B] for TMA S2G
        mO_tma = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (mO.shape[2], mO.shape[3], mO.shape[1], mO.shape[0]),
                stride=(mO.stride[2], mO.stride[3], mO.stride[1], mO.stride[0]),
            ),
        )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mO_tma,
            self.sO_layout,
            (self.m_block_size, self.tile_hdim),
        )

        k_bytes = self.n_block_size * self.tile_hdim * (self.dtype.width // 8)
        v_bytes = self.n_block_size * self.tile_hdim * (self.dtype.width // 8)

        B = cute.size(mQ.shape[0])
        H_q = cute.size(mQ.shape[1])
        seqlen_q = cute.size(mQ.shape[2])
        n_m_blocks = cute.ceil_div(seqlen_q, self.m_block_size)
        total_tiles = B * H_q * n_m_blocks

        # 1D persistent grid: one CTA per SM, tile-stride loop in kernel.
        grid_dim = (NUM_PERSISTENT_CTAS, 1, 1)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            mQ, tma_tensor_K, tma_tensor_V, tma_tensor_O, mLSE,
            tma_atom_K, tma_atom_V, tma_atom_O, k_bytes, v_bytes,
            n_m_blocks, H_q, total_tiles,
            self.sQ_layout, self.sK_layout, self.sV_layout,
            self.sO_layout,
            tiled_mma_qk, tiled_mma_pv,
            softmax_scale_log2,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        k_bytes: cutlass.Constexpr[int],
        v_bytes: cutlass.Constexpr[int],
        n_m_blocks: Int32,
        num_head_q: Int32,
        total_tiles: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        softmax_scale_log2: Float32,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Warp-spec dispatch: WG 0 = producer (TMA); WG 1, 2 = consumer (MMA).
        warp_group_idx = warp_idx // 4

        # Persistent grid: 1D, 1 CTA per SM. Tile-stride loop below decodes
        # tile_id → (b, h_q, m_block) in h_q-major order so the 16 q-heads
        # sharing the same (b, h_kv) appear on consecutive tile_ids and reuse
        # K/V from L2 across CTAs that land on the same SM-cluster.
        cta_id = cute.arch.block_idx()[0]

        # =========================================================
        # CTA-invariant setup (hoisted ABOVE the tile loop)
        # =========================================================

        # ---- Smem allocation (sO aliases sV) ----
        # sO reuses the sV storage region: sV is last read by the final PV
        # WGMMA in the body (wg_wait=0 ensures completion); after the
        # consumer-WG epilogue R2S+TMA writes sO, a CTA-wide tile-end
        # barrier prevents the next tile's producer from refilling sV slot 0
        # while the bulk-store-O is still draining smem.
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sO = storage.sV.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)

        # ---- Pipeline setup (mbarriers init exactly once) ----
        producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 1
        )
        consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 2
        )
        pipeline_K = PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=k_bytes,
            init_wait=False,
        )
        pipeline_V = PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=v_bytes,
            init_wait=True,
        )
        # init_wait=True above issues a CTA-wide barrier on all 384 threads.

        # ---- n_block_total (CTA-invariant; same for every tile) ----
        # mK_tma layout is (S, D, H_kv, B), so mK.shape[0] == seqlen_k.
        n_block_total = cute.ceil_div(
            cute.size(mK.shape[0]), self.n_block_size
        )

        # Per-CTA tile iteration count. tile_id = tile_iter*132 + cta_id.
        # For the target shape (1024 tiles / 132 CTAs), num_iters ∈ {7,8}.
        num_iters = cute.ceil_div(total_tiles - cta_id, NUM_PERSISTENT_CTAS)
        H_q_x_n_m = num_head_q * n_m_blocks

        # =========================================================
        # Warp-spec branch (regs allocated once, before tile loop)
        # =========================================================
        if warp_group_idx == 0:
            # ---- Producer WG (128 threads) ----
            cute.arch.warpgroup_reg_dealloc(PRODUCER_REG_COUNT)
            producer_state = make_simple_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer,
                self.num_stages,
            )

            for tile_iter in cutlass.range(0, num_iters, 1, unroll=1):
                tile_id = tile_iter * NUM_PERSISTENT_CTAS + cta_id
                # M8 m_block-major decode: same (b, h_q) → 32 m_blocks consecutive
                batch_idx = tile_id // H_q_x_n_m
                h_q_idx = (tile_id // n_m_blocks) % num_head_q
                m_block_idx = tile_id % n_m_blocks
                h_kv_idx = h_q_idx // self.qhead_per_kvhead

                # Per-tile K/V gmem views + TMA closures (gK/gV depend on tile)
                mK_cur = mK[None, None, h_kv_idx, batch_idx]
                mV_cur = mV[None, None, h_kv_idx, batch_idx]
                gK = cute.local_tile(
                    mK_cur, (self.n_block_size, self.tile_hdim), (None, 0),
                )
                gV = cute.local_tile(
                    mV_cur, (self.n_block_size, self.tile_hdim), (None, 0),
                )
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK,
                )
                load_K_p = copy_utils.tma_producer_copy_fn(load_K, pipeline_K)
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV,
                )
                load_V_p = copy_utils.tma_producer_copy_fn(load_V, pipeline_V)

                if const_expr(self.is_causal):
                    n_block_max_causal = cute.ceil_div(
                        (m_block_idx + 1) * self.m_block_size,
                        self.n_block_size,
                    )
                    n_block_max = cutlass.min(
                        n_block_total, n_block_max_causal
                    )
                else:
                    n_block_max = n_block_total

                if warp_idx == 0:
                    for n_block in cutlass.range(
                        0, n_block_max, 1, unroll=1
                    ):
                        pipeline_K.producer_acquire(producer_state)
                        load_K_p(n_block, producer_state)
                        pipeline_V.producer_acquire(producer_state)
                        load_V_p(n_block, producer_state)
                        producer_state.advance()
                # Warps 1-3 fall through to the tile-end barrier each iter.

                # Tile-end CTA-wide barrier: ensures consumer's bulk-store-O
                # has drained smem before producer refills sV slot 0 next tile.
                cute.arch.barrier(
                    barrier_id=2, number_of_threads=NUM_THREADS
                )
        else:
            # ---- Consumer WGs (2× 128 threads, MMA + softmax + epilogue) ----
            cute.arch.warpgroup_reg_alloc(CONSUMER_REG_COUNT)
            consumer_tidx = tidx - PRODUCER_THREADS

            # ---- Consumer-side hoisted partitions/fragments ----
            gs_copy_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=128,
            )
            threads_per_row = self.tile_hdim // 8
            thr_layout = cute.make_ordered_layout(
                (CONSUMER_THREADS // threads_per_row, threads_per_row),
                order=(1, 0),
            )
            val_layout = cute.make_layout((1, 8))
            gs_copy = cute.make_tiled_copy_tv(
                gs_copy_atom, thr_layout, val_layout
            )
            thr_gs = gs_copy.get_slice(consumer_tidx)
            tQsQ = thr_gs.partition_D(sQ)

            thr_mma_qk = tiled_mma_qk.get_slice(consumer_tidx)
            thr_mma_pv = tiled_mma_pv.get_slice(consumer_tidx)
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
            sVt = layout_utils.transpose_view(sV)
            tOrV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))

            acc_S_partition_shape = thr_mma_qk.partition_shape_C(
                (self.m_block_size, self.n_block_size)
            )
            acc_O_partition_shape = thr_mma_pv.partition_shape_C(
                (self.m_block_size, self.tile_hdim)
            )
            acc_O = cute.make_rmem_tensor(acc_O_partition_shape, Float32)
            rO = cute.make_rmem_tensor_like(acc_O, self.dtype)
            tOrP = cute.make_fragment(
                fa_utils.convert_layout_acc_frgA(
                    cute.make_layout(acc_S_partition_shape)
                ),
                self.dtype,
            )

            num_rows = const_expr(self.m_block_size // 64)
            softmax = Softmax(
                scale_log2=softmax_scale_log2, num_rows=num_rows
            )

            cS = cute.make_identity_tensor(
                (self.m_block_size, self.n_block_size)
            )
            tScS = thr_mma_qk.partition_C(cS)
            tScS_mn = layout_utils.make_acc_tensor_mn_view(tScS)

            copy_O_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_pv, sO, consumer_tidx, arch=self.arch,
            )

            consumer_state = make_simple_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer,
                self.num_stages,
            )

            for tile_iter in cutlass.range(0, num_iters, 1, unroll=1):
                tile_id = tile_iter * NUM_PERSISTENT_CTAS + cta_id
                # M8 m_block-major decode (must match producer side)
                batch_idx = tile_id // H_q_x_n_m
                h_q_idx = (tile_id // n_m_blocks) % num_head_q
                m_block_idx = tile_id % n_m_blocks

                # Per-tile Q/O gmem views (depend on b, h_q, m_block)
                mQ_cur = mQ[batch_idx, h_q_idx, None, None]
                mO_cur = mO[None, None, h_q_idx, batch_idx]
                gQ = cute.local_tile(
                    mQ_cur,
                    (self.m_block_size, self.tile_hdim),
                    (m_block_idx, 0),
                )
                gO = cute.local_tile(
                    mO_cur,
                    (self.m_block_size, self.tile_hdim),
                    (m_block_idx, 0),
                )
                tQgQ = thr_gs.partition_S(gQ)

                if const_expr(self.is_causal):
                    n_block_max_causal = cute.ceil_div(
                        (m_block_idx + 1) * self.m_block_size,
                        self.n_block_size,
                    )
                    n_block_max = cutlass.min(
                        n_block_total, n_block_max_causal
                    )
                else:
                    n_block_max = n_block_total

                # ---- Q load (cp.async, 256 consumer threads) ----
                cute.copy(gs_copy, tQgQ, tQsQ)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier(
                    barrier_id=1, number_of_threads=CONSUMER_THREADS
                )

                # Per-tile reset of accumulators / softmax state
                acc_O.fill(Float32.zero)
                softmax.reset()

                # ============================================================
                # M4 — Intra-WG WGMMA overlap: PV uses prev_P (1-iter shift)
                # ============================================================
                # In each body iter k, PV adds prev_P * V[k-1] to acc_O
                # using the P computed in the previous iter, so the QK[k]
                # WGMMA and PV[k-1] WGMMA both fly on the tensor pipe
                # while softmax runs on the CUDA pipe in parallel.
                # Pattern per body iter k (k = 1..n_block_max-1):
                #   wait K[k]; issue QK[k] (wg_wait=-1)
                #   wait V[k-1]; issue PV(prev_P, V[k-1]) (wg_wait=-1)
                #   wait_group(1) → drain QK[k], PV still in flight
                #   release K[k]
                #   [mask if diagonal] + softmax → row_scale, new tOrP
                #   wait_group(0) → drain PV
                #   release V[k-1]
                #   cvt acc_S → tOrP  (for next iter's PV)
                #   rescale_O(acc_O, row_scale)
                # Peel iter 0: QK[0] + mask (if m_block_idx==0) + softmax
                # only — NO PV here; state stays at slot 0 so V[0] is the
                # next PV's operand. Tail iter: PV(last tOrP, V[last]).

                # ---- Peel iter 0 (no PV; produce P_0 → tOrP) ----
                pipeline_K.consumer_wait(consumer_state)
                acc_S = cute.make_rmem_tensor(
                    acc_S_partition_shape, Float32
                )
                wgmma_gemm(
                    tiled_mma_qk, acc_S, tSrQ,
                    tSrK[None, None, None, consumer_state.index],
                    zero_init=True, wg_wait=0,
                )
                pipeline_K.consumer_release(consumer_state)

                if const_expr(self.is_causal):
                    m_off = m_block_idx * self.m_block_size
                    if self.n_block_size > m_off:
                        acc_S_mn = layout_utils.make_acc_tensor_mn_view(
                            acc_S
                        )
                        for r in cutlass.range_constexpr(
                            cute.size(acc_S_mn.shape[0])
                        ):
                            for c in cutlass.range_constexpr(
                                cute.size(acc_S_mn.shape[1])
                            ):
                                crd = tScS_mn[r, c]
                                global_row = m_off + crd[0]
                                global_col = crd[1]
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if global_col > global_row
                                    else acc_S_mn[r, c]
                                )

                softmax.online_softmax(
                    acc_S, is_first=True, check_inf=True,
                )

                tOrP_acc = cute.make_tensor(
                    acc_S.iterator,
                    fa_utils.convert_layout_acc_frgA(acc_S.layout),
                )
                fa_utils.cvt_f16(tOrP_acc, tOrP)
                # NOTE: do NOT advance consumer_state — V[0] slot must
                # remain visible to the next iter's PV (or to the tail
                # iter when n_block_max == 1). acc_O stays at zero.

                if const_expr(self.is_causal):
                    n_main_end = n_block_max - 1
                else:
                    n_main_end = n_block_max

                # ---- Unmasked main body iters [1, n_main_end) (M4 overlap) ----
                for n_block in cutlass.range(
                    1, n_main_end, 1, unroll=1
                ):
                    v_state = consumer_state.clone()
                    consumer_state.advance()
                    pipeline_K.consumer_wait(consumer_state)
                    acc_S = cute.make_rmem_tensor(
                        acc_S_partition_shape, Float32
                    )
                    wgmma_gemm(
                        tiled_mma_qk, acc_S, tSrQ,
                        tSrK[None, None, None, consumer_state.index],
                        zero_init=True, wg_wait=-1,
                    )
                    pipeline_V.consumer_wait(v_state)
                    wgmma_gemm(
                        tiled_mma_pv, acc_O, tOrP,
                        tOrV[None, None, None, v_state.index],
                        zero_init=False, wg_wait=-1,
                    )
                    warpgroup.wait_group(1)
                    pipeline_K.consumer_release(consumer_state)

                    row_scale = softmax.online_softmax(
                        acc_S, is_first=False, check_inf=False,
                    )

                    warpgroup.wait_group(0)
                    pipeline_V.consumer_release(v_state)

                    tOrP_acc = cute.make_tensor(
                        acc_S.iterator,
                        fa_utils.convert_layout_acc_frgA(acc_S.layout),
                    )
                    fa_utils.cvt_f16(tOrP_acc, tOrP)
                    softmax.rescale_O(acc_O, row_scale)

                # ---- Masked diagonal iter (causal, n_block_max > 1) ----
                # Same M4 overlap shape; mask applied to acc_S before
                # softmax. Skipped when n_block_max <= 1 (peel iter 0 IS
                # the diagonal for m_block_idx == 0 and masked itself).
                if const_expr(self.is_causal):
                    if n_block_max > 1:
                        n_block = n_block_max - 1
                        v_state = consumer_state.clone()
                        consumer_state.advance()
                        pipeline_K.consumer_wait(consumer_state)
                        acc_S = cute.make_rmem_tensor(
                            acc_S_partition_shape, Float32
                        )
                        wgmma_gemm(
                            tiled_mma_qk, acc_S, tSrQ,
                            tSrK[None, None, None, consumer_state.index],
                            zero_init=True, wg_wait=-1,
                        )
                        pipeline_V.consumer_wait(v_state)
                        wgmma_gemm(
                            tiled_mma_pv, acc_O, tOrP,
                            tOrV[None, None, None, v_state.index],
                            zero_init=False, wg_wait=-1,
                        )
                        warpgroup.wait_group(1)
                        pipeline_K.consumer_release(consumer_state)

                        m_off = m_block_idx * self.m_block_size
                        n_off = n_block * self.n_block_size
                        acc_S_mn = layout_utils.make_acc_tensor_mn_view(
                            acc_S
                        )
                        for r in cutlass.range_constexpr(
                            cute.size(acc_S_mn.shape[0])
                        ):
                            for c in cutlass.range_constexpr(
                                cute.size(acc_S_mn.shape[1])
                            ):
                                crd = tScS_mn[r, c]
                                global_row = m_off + crd[0]
                                global_col = n_off + crd[1]
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if global_col > global_row
                                    else acc_S_mn[r, c]
                                )

                        row_scale = softmax.online_softmax(
                            acc_S, is_first=False, check_inf=False,
                        )

                        warpgroup.wait_group(0)
                        pipeline_V.consumer_release(v_state)

                        tOrP_acc = cute.make_tensor(
                            acc_S.iterator,
                            fa_utils.convert_layout_acc_frgA(acc_S.layout),
                        )
                        fa_utils.cvt_f16(tOrP_acc, tOrP)
                        softmax.rescale_O(acc_O, row_scale)

                # ---- Tail iter: PV(last tOrP, V[n_block_max-1]) ----
                # consumer_state.index points at V[n_block_max-1]'s slot
                # (which is also where the last K was waited).
                pipeline_V.consumer_wait(consumer_state)
                wgmma_gemm(
                    tiled_mma_pv, acc_O, tOrP,
                    tOrV[None, None, None, consumer_state.index],
                    zero_init=False, wg_wait=0,
                )
                pipeline_V.consumer_release(consumer_state)
                consumer_state.advance()

                # ---- Epilogue ----
                final_row_scale = softmax.finalize()

                if const_expr(mLSE is not None):
                    mLSE_cur = mLSE[batch_idx, h_q_idx, None]
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (m_block_idx,)
                    )
                    lane_in_warp = cute.arch.lane_idx()
                    if lane_in_warp % 4 == 0:
                        for r in cutlass.range_constexpr(num_rows):
                            m_in_tile = tScS_mn[r, 0][0]
                            fa_utils.store_global_fp32(
                                softmax.row_sum[r],
                                fa_utils.elem_pointer(gLSE, m_in_tile),
                            )

                softmax.rescale_O(acc_O, final_row_scale)
                fa_utils.cvt_f16(acc_O, rO)

                # Consumer-WG-local barrier before R2S (sO aliases sV).
                cute.arch.barrier(
                    barrier_id=1, number_of_threads=CONSUMER_THREADS
                )

                copy_O_r2s(rO)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=1, number_of_threads=CONSUMER_THREADS
                )

                # TMA S2G for O — first warp of consumer WG 0 issues it.
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1),
                    sO, gO, single_stage=True,
                )
                if warp_idx == 4:
                    store_O()
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

                # Drain TMA-store smem reads before tile-end CTA barrier.
                cute.arch.barrier(
                    barrier_id=1, number_of_threads=CONSUMER_THREADS
                )

                # Tile-end CTA-wide barrier (matches producer side); ensures
                # bulk-store-O is fully drained before next tile's producer
                # writes sV slot 0 (which aliases sO).
                cute.arch.barrier(
                    barrier_id=2, number_of_threads=NUM_THREADS
                )
