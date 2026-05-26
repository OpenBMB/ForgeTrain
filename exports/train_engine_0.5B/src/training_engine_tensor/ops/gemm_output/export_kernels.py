"""Export CuTeDSL non-persistent GEMM kernels to C object files for gemm_output.

This script runs in a subprocess (not during training) to compile CuTeDSL kernels
via the MLIR path, which avoids the nvcc C7510 WGMMA serialization issue.

Three directions:
  fwd:   C[M,N] = A[M,K] * B[N,K]     BF16→BF16   (M=40960, K=1024, N=73472)
  dgrad: C[M,N] = A[M,K] * B[N,K]     BF16→BF16   (M=40960, K=73472, N=1024)
  wgrad: C[M,N] = A[M,K] * B[N,K]     BF16→FP32   (M=73472, K=40960, N=1024)
"""
import gc
import math
import os
import shutil
import sys
from typing import Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

from training_engine_tensor.op_dispatcher import get_build_env as _get_build_env
from training_engine_tensor.op_dispatcher import get_frozen_env as _get_frozen_env
from training_engine_tensor.op_dispatcher import init as _init_build_env


# ── HopperGemmKernel (self-contained, only used for export) ────────────

class HopperGemmKernel:
    """Batched GEMM D[m,n,l] = A[m,k,l] * B[n,k,l] on Hopper.

    SM90 WGMMA + TMA, FP32 accumulation. Output can be BF16 or FP32.
    """

    def __init__(
        self,
        acc_dtype: type,
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        epi_stage_override: int = 4,
    ):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.epi_stage_override = epi_stage_override
        self.mma_inst_shape_mn = None
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        self.atom_layout_mnk = (
            (2, 1, 1)
            if tile_shape_mn[0] > 64 and tile_shape_mn[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (
            self.mma_warp_groups * self.num_threads_per_warp_group
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.a_dtype = None
        self.b_dtype = None
        self.c_dtype = None
        self.a_layout = None
        self.b_layout = None
        self.c_layout = None

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")

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
            self.tile_shape_mnk[0], self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative,
        )
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk, self.a_dtype, self.b_dtype,
            self.smem_capacity, self.occupancy,
            self.epi_tile, self.c_dtype,
            self.epi_stage_override,
        )
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk, self.epi_tile,
            self.a_dtype, self.a_layout,
            self.b_dtype, self.b_layout,
            self.ab_stage,
            self.c_dtype, self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor,
                 c: cute.Tensor, stream: cuda.CUstream):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b, self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c, self.epi_smem_layout_staged, self.epi_tile,
        )
        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mn)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            self.tiled_mma, self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl,
               tma_atom_c, mC_mnl, tiled_mma, cta_layout_mnk,
               a_smem_layout_staged, b_smem_layout_staged,
               epi_smem_layout_staged):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cdimx * cidy

        group_size_m = 8
        s_shape = ((group_size_m, cdimx // group_size_m), cdimy)
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)
        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m),
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]
        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1,
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0,
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = (
            storage.mainloop_pipeline_array_ptr.data_ptr()
        )
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt,
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner,
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer,
            swizzle=epi_smem_layout_staged.inner,
        )
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1),
        )
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None),
        )
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group,
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group,
        )
        thr_mma = tiled_mma.get_slice(
            warp_group_thread_layout(warp_group_idx),
        )
        tCgC = thr_mma.partition_C(gC_mnl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape,
        )
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma = cute.group_modes(sA, 0, 2)
        gA_for_tma = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, a_cta_crd, a_cta_layout, sA_for_tma, gA_for_tma,
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape,
        )
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma = cute.group_modes(sB, 0, 2)
        gB_for_tma = cute.group_modes(gB_nkl, 0, 2)
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, b_cta_crd, b_cta_layout, sB_for_tma, gB_for_tma,
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(
            cutlass.min(self.ab_stage, k_tile_cnt), 0,
        )
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage,
        )
        if warp_idx == 0:
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                cute.copy(
                    tma_atom_a, tAgA_k, tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state,
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b, tBgB_k, tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state,
                    ),
                    mcast_mask=b_mcast_mask,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        k_pipe_mmas = 1
        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage,
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage,
        )
        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state,
            )
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])

        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status,
            )
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None, None, k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]
                cute.gemm(
                    tiled_mma, accumulators,
                    tCrA_1phase, tCrB_1phase, accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state,
                )

        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status,
            )
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None, None, k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]
                cute.gemm(
                    tiled_mma, accumulators,
                    tCrA_1phase, tCrB_1phase, accumulators,
                )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
            mainloop_pipeline.consumer_release(
                mainloop_consumer_release_state,
            )
            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state,
                )
            if (
                warp_idx == 0
                and mainloop_producer_state.count < k_tile_cnt
            ):
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                cute.copy(
                    tma_atom_a, tAgA_k, tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state,
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b, tBgB_k, tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state,
                    ),
                    mcast_mask=b_mcast_mask,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        cute.nvgpu.warpgroup.wait_group(0)
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout, elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(), 4,
            ),
            self.c_dtype,
        )
        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)
        sepi_for_tma = cute.group_modes(sC, 0, 2)
        tCgC_for_tma = cute.zipped_divide(gC_mnl, self.epi_tile)
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            sepi_for_tma, tCgC_for_tma,
        )
        epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
        epi_tile_shape = tCgC_for_tma.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1),
        )
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta,
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
            tRS_rD_out = cute.make_rmem_tensor_like(
                tRS_rD_layout, self.c_dtype,
            )
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out,
                tRS_sD[(None, None, None, epi_buffer)],
            )
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            pipeline.sync(barrier_id=1)
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            pipeline.sync(barrier_id=1)
        if warp_idx == 0:
            c_pipeline.producer_tail()

    @staticmethod
    def _compute_stages(tile_shape_mnk, a_dtype, b_dtype,
                        smem_capacity, occupancy,
                        epi_tile=None, c_dtype=None,
                        epi_stage_override=4):
        epi_stage = epi_stage_override
        if epi_tile is not None and c_dtype is not None:
            epi_bytes = (epi_tile[0] * epi_tile[1]
                         * c_dtype.width // 8 * epi_stage)
        else:
            epi_bytes = 0
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        print(f"  [compute_stages] tile={tile_shape_mnk} epi_stage={epi_stage} "
              f"epi_bytes={epi_bytes/1024:.1f}KB ab_stage={ab_stage} "
              f"ab_bytes={ab_bytes_per_stage/1024:.1f}KB/stage")
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk, epi_tile,
        a_dtype, a_layout, b_dtype, b_layout, ab_stage,
        c_dtype, c_layout, epi_stage,
    ):
        a_staged = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, ab_stage,
        )
        b_staged = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, ab_stage,
        )
        epi_staged = sm90_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, epi_stage,
        )
        return a_staged, b_staged, epi_staged

    @staticmethod
    def _compute_grid(c, tile_shape_mnk, cluster_shape_mn):
        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        clusters = cute.ceil_div(
            cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl,
        )
        return tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_c, epi_smem_staged, epi_tile):
        epi_smem = cute.slice_(epi_smem_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c, epi_smem, epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem = cute.slice_(smem_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem, smem_tile, num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


def _make_forced_persistent_kernel_class(forced_ab, forced_epi):
    """Return a subclass of HopperWgmmaGemmPersistentKernel that forces
    `_compute_stages` to return `(forced_ab, forced_epi)`.

    Falls back to the default heuristic if the requested stages exceed the
    H100 SMEM budget for the given (tile, dtype) — this keeps the export
    script safe to call with any (ab, epi) without manually proving it
    fits.
    """
    from training_engine_tensor.ops.gemm_fc1.bench_cutedsl import HopperWgmmaGemmPersistentKernel as _Base

    class _ForcedStages(_Base):
        @staticmethod
        def _compute_stages(tile_shape_mnk, a_dtype, b_dtype, epi_tile,
                            c_dtype, smem_capacity, occupancy):
            a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
            b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
            ab_bytes = (
                cute.size(a_shape) * a_dtype.width // 8
                + cute.size(b_shape) * b_dtype.width // 8
            )
            c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
            mbar = 1024
            need = mbar + c_bytes_per_stage * forced_epi + ab_bytes * forced_ab
            budget = smem_capacity // occupancy
            if need > budget:
                return _Base._compute_stages(
                    tile_shape_mnk, a_dtype, b_dtype, epi_tile, c_dtype,
                    smem_capacity, occupancy,
                )
            return forced_ab, forced_epi

    _ForcedStages.__name__ = (
        f"HopperWgmmaPersistentKernel_ab{forced_ab}_epi{forced_epi}"
    )
    return _ForcedStages


def compile_persistent_and_export(name, M, N, K, a_col_major, b_col_major,
                                  c_col_major, c_dtype, tile, cluster,
                                  swizzle, raster_along_m, export_dir,
                                  mma_inst_tile_k=4,
                                  ab_stage_override=None,
                                  epi_stage_override=None,
                                  is_dynamic_layout=True):
    """Compile HopperWgmmaGemmPersistentKernel and export to C object file.

    Persistent + warp-specialized variant: dedicated DMA warp group + 2 MMA
    warp groups; max_active_clusters baked in as constexpr at compile time.

    `mma_inst_tile_k` controls the WGMMA K-instruction count per mainloop
    iteration. CTA K-tile = mma_inst_shape_k * mma_inst_tile_k (16 * k for
    BF16). Default 4 (K-tile=64). Try 2 (32) or 8 (128) for shape-specific
    tuning — small K shapes (e.g. fwd K=1024) benefit from larger K-tiles
    that reduce TMA-burst count.

    `ab_stage_override` / `epi_stage_override`: when both are provided, the
    kernel will force `(ab_stage, epi_stage) = (ab_stage_override,
    epi_stage_override)`. If the resulting SMEM exceeds the H100 budget,
    silently falls back to the default heuristic. Used by Round 35: fwd
    benefits from (3, 3) instead of the default (4, 4) — the extra ab/epi
    buffers don't pay off at K=1024 (only 16 K-iters) and the saved SMEM
    apparently relieves either occupancy or register-spill pressure.

    Round 46: `is_dynamic_layout=False` bakes shapes/strides into the kernel
    as compile-time constants.  The exported tensor descriptor becomes just
    ``{ void *data; }`` instead of ``{ void *data; int32_t shapes[3];
    int64_t strides[2]; }``.  For fixed training shapes this saves host-side
    descriptor-fill overhead per kernel launch.  When static, proxy tensors
    must use the actual (M, N, K) — not min(M, 256) — because the shapes
    become part of the kernel binary.
    """
    from training_engine_tensor.ops.gemm_fc1.bench_cutedsl import HopperWgmmaGemmPersistentKernel

    print(f"\n=== Compiling {name} (persistent): M={M} N={N} K={K} ===")
    print(f"  a_col={a_col_major}, b_col={b_col_major}, c_col={c_col_major}, "
          f"c_dtype={c_dtype}, tile={tile}, cluster={cluster}, "
          f"swizzle={swizzle}, raster_along_m={raster_along_m}, "
          f"mma_inst_tile_k={mma_inst_tile_k}, "
          f"ab_stage={ab_stage_override}, epi_stage={epi_stage_override}, "
          f"is_dynamic_layout={is_dynamic_layout}")

    if is_dynamic_layout:
        small_M = min(M, 256)
        small_N = min(N, 256)
        small_K = min(K, 256)
    else:
        small_M, small_N, small_K = M, N, K

    a_cpu = cutlass_torch.matrix(1, small_M, small_K, a_col_major, cutlass.BFloat16)
    b_cpu = cutlass_torch.matrix(1, small_N, small_K, b_col_major, cutlass.BFloat16)
    c_cpu = cutlass_torch.matrix(1, small_M, small_N, c_col_major, c_dtype)

    a_t, a_gpu = cutlass_torch.cute_tensor_like(
        a_cpu, cutlass.BFloat16, is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    b_t, b_gpu = cutlass_torch.cute_tensor_like(
        b_cpu, cutlass.BFloat16, is_dynamic_layout=is_dynamic_layout, assumed_align=16)
    c_t, c_gpu = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype, is_dynamic_layout=is_dynamic_layout, assumed_align=16)

    print(f"  a_cute: shape={a_t.shape}, stride={a_t.stride}")
    print(f"  b_cute: shape={b_t.shape}, stride={b_t.stride}")
    print(f"  c_cute: shape={c_t.shape}, stride={c_t.stride}")

    if ab_stage_override is not None and epi_stage_override is not None:
        KernelCls = _make_forced_persistent_kernel_class(
            ab_stage_override, epi_stage_override,
        )
    else:
        KernelCls = HopperWgmmaGemmPersistentKernel
    gemm = KernelCls(
        cutlass.Float32, tile, cluster,
        swizzle_size=swizzle, raster_along_m=raster_along_m,
        mma_inst_tile_k=mma_inst_tile_k)
    hw = utils.HardwareInfo()
    mac = hw.get_max_active_clusters(cluster[0] * cluster[1])
    print(f"  max_active_clusters={mac}")

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(gemm, a_t, b_t, c_t, mac, stream)

    compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()

    out_dir = os.path.join(export_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(compiled, "export_to_c"):
        compiled.export_to_c(out_dir, name)
    else:
        from cutlass.cute.export import export_to_c as _export_to_c_fn
        _export_to_c_fn(compiled, out_dir, name)

    for f in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"  exported: {f} ({sz} bytes)")

    h_path = os.path.join(out_dir, f"{name}.h")
    with open(h_path) as f:
        print(f"\n--- {name}.h ---")
        print(f.read())

    del a_t, a_gpu, b_t, b_gpu, c_t, c_gpu, compiled, gemm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory freed after {name} export")


def compile_and_export(name, M, N, K, a_col_major, b_col_major, c_col_major,
                       c_dtype, tile, cluster, export_dir, epi_stage_override=4):
    """Compile a single GEMM kernel and export to C object file.

    Uses small dummy sizes for compilation (shape is dynamic), then
    aggressively frees GPU memory.
    """
    print(f"\n=== Compiling {name}: M={M} N={N} K={K} ===")
    print(f"  a_col={a_col_major}, b_col={b_col_major}, c_col={c_col_major}, "
          f"c_dtype={c_dtype}, tile={tile}, cluster={cluster}, "
          f"epi_stage={epi_stage_override}")

    small_M = min(M, 256)
    small_N = min(N, 256)
    small_K = min(K, 256)

    a_cpu = cutlass_torch.matrix(1, small_M, small_K, a_col_major, cutlass.BFloat16)
    b_cpu = cutlass_torch.matrix(1, small_N, small_K, b_col_major, cutlass.BFloat16)
    c_cpu = cutlass_torch.matrix(1, small_M, small_N, c_col_major, c_dtype)

    a_t, a_gpu = cutlass_torch.cute_tensor_like(
        a_cpu, cutlass.BFloat16, is_dynamic_layout=True, assumed_align=16)
    b_t, b_gpu = cutlass_torch.cute_tensor_like(
        b_cpu, cutlass.BFloat16, is_dynamic_layout=True, assumed_align=16)
    c_t, c_gpu = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16)

    print(f"  a_cute: shape={a_t.shape}, stride={a_t.stride}")
    print(f"  b_cute: shape={b_t.shape}, stride={b_t.stride}")
    print(f"  c_cute: shape={c_t.shape}, stride={c_t.stride}")

    gemm = HopperGemmKernel(cutlass.Float32, tile, cluster,
                            epi_stage_override=epi_stage_override)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled = cute.compile(gemm, a_t, b_t, c_t, stream)

    compiled(a_t, b_t, c_t, stream)
    torch.cuda.synchronize()

    out_dir = os.path.join(export_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(compiled, "export_to_c"):
        compiled.export_to_c(out_dir, name)
    else:
        from cutlass.cute.export import export_to_c as _export_to_c_fn
        _export_to_c_fn(compiled, out_dir, name)

    for f in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"  exported: {f} ({sz} bytes)")

    h_path = os.path.join(out_dir, f"{name}.h")
    with open(h_path) as f:
        print(f"\n--- {name}.h ---")
        print(f.read())

    del a_t, a_gpu, b_t, b_gpu, c_t, c_gpu, compiled, gemm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory freed after {name} export")


def main():
    # See gemm_fc1/export_kernels.py for the rationale.
    _init_build_env(dict(os.environ))
    export_dir = os.path.join(
        _get_build_env().cutedsl_cache_root,
        "cutedsl_export_gemm_output",
    )
    os.makedirs(export_dir, exist_ok=True)
    for child in os.listdir(export_dir):
        p = os.path.join(export_dir, child)
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
        else:
            shutil.rmtree(p, ignore_errors=True)

    # fwd: logits[M,N] = X[M,K] @ W_padded[N,K]^T
    # All row-major (K contiguous for A/B, N contiguous for C).
    # Round 29.7 fresh sweep (M=4096, N=73472, K=1024, tile=(128,256), p10):
    #   persistent (1,1) sw=4 : 0.899 ms (1.053x cuBLAS)
    #   persistent (1,1) sw=8 : 0.915 ms (1.072x cuBLAS)
    #   persistent (2,1) sw=4 : 0.898 ms (1.052x cuBLAS)  ← NEW WINNER
    #   persistent (2,1) sw=8 : 0.938 ms (1.099x cuBLAS)  ← Round 29.4 picked
    #   persistent (4,1) sw=4 : 0.947 ms (1.110x)
    #   persistent (4,1) sw=8 : 0.927 ms (1.086x)
    #   persistent (1,2) sw=4 : 0.962 ms (1.128x)         ← was Round 29.4 "winner"
    #   persistent (1,2) sw=8 : 0.946 ms (1.109x)
    #   persistent (2,2) sw=4 : 0.948 ms (1.111x)
    # cuBLAS p10 = 0.853 ms at this sweep. (2,1) sw=4 saves ~40us / call vs sw=8.
    # Round 29.4's "winner" (1,2) sw=4 is no longer optimal — different pre-sweep
    # warm-up state probably influenced the original ranking.
    #
    # Round 30 fresh fwd sweep (M=4096 — per-DP rank — N=73472 K=1024):
    #   Top 4 by p50 (cuBLAS p50 = 0.916 in same run):
    #     persistent (2,1) sw=8 ras_m=True : p10=0.913 p50=0.967 (1.055x)
    #     persistent (2,2) sw=4 ras_m=True : p10=0.929 p50=0.971 (1.060x)
    #     persistent (2,1) sw=4 ras_m=True : p10=0.911 p50=0.975 (1.064x)  ← keep
    #     persistent (2,1) sw=4 ras_m=False: p10=0.918 p50=0.979 (1.068x)
    # NOTE: sw=8 won the standalone sweep but in bench_all_gemm.py (where the
    # weight pad / transpose / output-alloc wrapper overhead is included)
    # sw=4 was empirically as good or slightly better, so keep sw=4.
    # Round 35 staging override: default heuristic gives ab_stage=4 epi_stage=4
    # for fwd. Forced (3, 3) is 2.7% faster on p50 in REPEATS=7 ITERS=400
    # re-validation (long-form) and 2.1% faster in the original sweep —
    # passes R30.2 variance discipline (≥ 2% in two repeats). The shallower
    # pipeline frees ~80 KiB SMEM (1 ab + 1 epi stage); plausible cause is
    # better register-allocation / occupancy at K=1024 (only 16 K-iterations,
    # so deep pipelines don't pay off).
    #
    # Round 38 attempted to swap (2,1)/sw=4 → (1,1)/sw=1 based on sweep
    # results from sweep_round38_revalidate.py (paired t-test on raw
    # cute.compile path showed B=(1,1)/sw=1 was -3.0% faster, t=-3.44).
    # However, when re-exported through the C-export AOT path and
    # measured via bench_all_gemm.py wrapper:
    #     PROD (2,1)/sw=4 : v1 fwd median ≈ 0.969 ms
    #     NEW  (1,1)/sw=1 : v1 fwd median ≈ 0.993 ms (8% slower than baseline)
    # 5-run repeats consistently showed regression; reverted to PROD.
    # Lesson: dynamic cute.compile p50 timings do NOT translate 1:1 to
    # AOT C-export wall times, likely because the C-export wrapper path
    # has its own dispatch costs that interact differently with cluster
    # scheduling (cluster=(1,1) makes scheduler issue 2x as many launch
    # batches at the wrapper level even though kernel runs faster).
    # Round 42 sweep hook (parallel to R39 wgrad hooks, lesson from R38:
    # never trust dynamic-compile sweeps — must validate every fwd candidate
    # through the C-export wrapper path used by bench_all_gemm.py). Set
    # GEMM_OUT_FWD_{TILE_M,TILE_N,CL_M,CL_N,SW,RASM,AB,EP,MMAK} env vars to
    # override the production fwd config for one run. Defaults match the
    # R35 production config (cluster (2,1), sw=4, ras_m=True, ab/epi=(3,3)).
    # After picking a winning config, drop the env hook and hard-code below.
    _fw_tm = int(os.environ.get("GEMM_OUT_FWD_TILE_M", "128"))
    _fw_tn = int(os.environ.get("GEMM_OUT_FWD_TILE_N", "256"))
    _fw_cm = int(os.environ.get("GEMM_OUT_FWD_CL_M",   "2"))
    _fw_cn = int(os.environ.get("GEMM_OUT_FWD_CL_N",   "1"))
    _fw_sw = int(os.environ.get("GEMM_OUT_FWD_SW",     "8"))
    _fw_rm = int(os.environ.get("GEMM_OUT_FWD_RASM",   "0")) != 0
    _fw_ab = os.environ.get("GEMM_OUT_FWD_AB", "3")
    _fw_ep = os.environ.get("GEMM_OUT_FWD_EP", "3")
    _fw_mk = os.environ.get("GEMM_OUT_FWD_MMAK", "")
    _fw_kw = {}
    if _fw_ab and _fw_ep:
        _fw_kw["ab_stage_override"] = int(_fw_ab)
        _fw_kw["epi_stage_override"] = int(_fw_ep)
    if _fw_mk:
        _fw_kw["mma_inst_tile_k"] = int(_fw_mk)
    # Round 46: is_dynamic_layout=False bakes shapes into the kernel binary.
    # Descriptor becomes { void *data } — saves host-side fill overhead.
    # gemm_fc1 R46 showed ~70µs/call savings on fwd with static layout.
    compile_persistent_and_export("gemm_output_fwd",
        M=40960, N=73472, K=1024,
        a_col_major=False, b_col_major=False, c_col_major=False,
        c_dtype=cutlass.BFloat16,
        tile=(_fw_tm, _fw_tn), cluster=(_fw_cm, _fw_cn),
        swizzle=_fw_sw, raster_along_m=_fw_rm,
        export_dir=export_dir, is_dynamic_layout=False, **_fw_kw)

    # dgrad: dX[M,N] = dY[M,K] @ W (where W is [N=K_w, K_inner=V_pad])
    # Round 37 switch: b_col_major=True
    #
    # Old path (b_col_major=False):
    #   - B was `_wt_padded_t = w_padded.t().contiguous()` — a 143 MB
    #     row-major buffer requiring a side-stream transpose every Adam
    #     step (cache invalidated when weight._version bumps).
    #   - Standalone transpose cost ≈ 0.60 ms (measured R37: HBM3 bw bound)
    #   - In real training the side-stream copy *competed* with fwd's L2/HBM
    #     traffic (143 MB write to DRAM during fwd's 0.93 ms compute).
    #
    # New path (b_col_major=True):
    #   - B is `w_padded.t()` viewed as col-major B[N=1024, K=73472]
    #     (zero-copy, leading_dim = N).
    #   - No side-stream transpose, no _wt_padded_t buffer (frees 143 MB).
    #   - Kernel itself is ~1% slower in steady-state bench (R37 sweep:
    #     0.812 ms vs 0.803 ms p50, both within ~1.0x cuBLAS), but the
    #     real-training step saves the entire 0.60 ms side-stream copy.
    #
    # Tile / cluster sweep with b_col_major=True (R37, REPEATS=3 ITERS=200):
    #   (128,256) (1,1) sw=4 : p50=0.819 (1.012x cuBLAS) ← clean
    #   (128,256) (2,1) sw=4 : p50=0.812 (1.003x cuBLAS) ← FASTEST col-major
    #   (128,256) (1,1) sw=8 : p50=2.42  catastrophic
    #   (128,128) / (64,*)   : 1.14-1.92x  much worse
    # Picked (128,256) cluster=(2,1) swizzle=4 raster_along_m=True for Round 37.
    # Same tile/cluster shape as fwd; the col-major B makes (2,1) cluster usable
    # because the K=73472 reduction has plenty of work to amortize sync.
    # Round 46: is_dynamic_layout=False — same rationale as fwd.
    # dgrad's B is col-major view of w_padded; with static layout the
    # col-major strides are baked in, so the C++ side passes { void* } only.
    compile_persistent_and_export("gemm_output_dgrad",
        M=40960, N=1024, K=73472,
        a_col_major=False, b_col_major=True, c_col_major=False,
        c_dtype=cutlass.BFloat16,
        tile=(128, 256), cluster=(2, 1),
        swizzle=1, raster_along_m=False,  # R-autotune: sw=1 raster=F +28.9% vs cuBLAS
        export_dir=export_dir, is_dynamic_layout=False)

    # wgrad: dW[M,N] = dY^T[M,K] @ X (B is col-major view of X)
    # A col-major (dY^T viewed from dY row-major), B col-major view of X,
    # C row-major. Round 29.5 persistent sweep (M=73472 N=1024 K=4096):
    #   dense      (1,1) sw=4 : ~0.90 ms (0.79x cuBLAS)  ← previous
    #   persistent (1,1) sw=4 : 0.868 ms (0.760x)
    #   persistent (2,1) sw=4 : 0.855 ms (0.750x)  ← winner — beats cuBLAS 25%
    # Persistent wins via dedicated DMA warp group + tail-wave balancing on
    # the long M=73472 axis (287 M-tiles × 4 N-tiles).
    #
    # Round 29.7: switched B from row-major (which required a 40 MB
    # x.t().contiguous() pre-copy per call) to a col-major view of x's
    # actual storage. Saves the explicit BF16 copy (≈40us per call) and
    # the persistent _wgrad_b_buf allocation (40 MB) without changing the
    # MMA layout. The kernel still consumes B with the same K-major SMEM
    # layout via TMA's gmem→smem swizzle.
    #
    # Round 30 fresh wgrad sweep (M=73472 N=1024 K=4096):
    #   Top 4 by p50:
    #     persistent (1,1) sw=4 ras_m=False: p10=0.847 p50=0.886            ← NEW
    #     persistent (1,1) sw=4 ras_m=True : p10=0.822 p50=0.890 (best p10)
    #     persistent (2,1) sw=4 ras_m=True : p10=0.826 p50=0.907            ← old
    #     persistent (1,1) sw=8 ras_m=False: p10=0.844 p50=0.915
    # Same pattern as dgrad: small N=1024 (4 N-tiles) means cluster (2,1)
    # along M just adds tail-wave imbalance with no multicast benefit. Use
    # ras_m=True since p10 is much better (and p50 is essentially tied).
    # Round 39 sweep hook (R38 taught us: never trust dynamic-compile sweeps;
    # MUST validate every candidate through the C-export wrapper path used
    # by bench_all_gemm.py). Set GEMM_OUT_WGRAD_TILE_M / TILE_N / CL_M / CL_N
    # / SW / RASM in the environment to override the production wgrad config
    # for one run. Defaults match the R30 winner. After picking a final
    # config, drop the env hook and hard-code the winning numbers below.
    _wg_tm = int(os.environ.get("GEMM_OUT_WGRAD_TILE_M", "128"))
    _wg_tn = int(os.environ.get("GEMM_OUT_WGRAD_TILE_N", "256"))
    _wg_cm = int(os.environ.get("GEMM_OUT_WGRAD_CL_M",   "1"))
    _wg_cn = int(os.environ.get("GEMM_OUT_WGRAD_CL_N",   "1"))
    _wg_sw = int(os.environ.get("GEMM_OUT_WGRAD_SW",     "1"))  # R-autotune: sw=1 +46.7% vs cuBLAS
    _wg_rm = int(os.environ.get("GEMM_OUT_WGRAD_RASM",   "0")) != 0  # R-autotune: raster=F
    _wg_ab = os.environ.get("GEMM_OUT_WGRAD_AB", "")
    _wg_ep = os.environ.get("GEMM_OUT_WGRAD_EP", "")
    _wg_mk = os.environ.get("GEMM_OUT_WGRAD_MMAK", "")
    _wg_kw = {}
    if _wg_ab and _wg_ep:
        _wg_kw["ab_stage_override"] = int(_wg_ab)
        _wg_kw["epi_stage_override"] = int(_wg_ep)
    if _wg_mk:
        _wg_kw["mma_inst_tile_k"] = int(_wg_mk)
    # Round 48: is_dynamic_layout=False bakes shapes into the kernel binary.
    # gemm_fc1 R46 found wgrad regressed with static layout (M=4096), but
    # gemm_output wgrad has M=73472 (much larger) — the larger descriptor
    # fill overhead at runtime may make static layout net-positive here.
    # Descriptor becomes { void *data } (same as fwd/dgrad since R46).
    compile_persistent_and_export("gemm_output_wgrad",
        M=73472, N=1024, K=40960,
        a_col_major=True, b_col_major=True, c_col_major=False,
        c_dtype=cutlass.Float32,
        tile=(_wg_tm, _wg_tn), cluster=(_wg_cm, _wg_cn),
        swizzle=_wg_sw, raster_along_m=_wg_rm,
        export_dir=export_dir, is_dynamic_layout=False, **_wg_kw)

    print(f"\n=== All exports complete ===")
    for root, dirs, files in os.walk(export_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            print(f"  {os.path.relpath(fp, export_dir)} ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
