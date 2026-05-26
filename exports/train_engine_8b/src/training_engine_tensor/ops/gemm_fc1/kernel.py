"""gemm_fc1 — host-side dispatcher for the persistent CuTeDSL SS-WGMMA GEMM.

Builds the per-direction tile / cluster / multicast configuration and forwards each call into :mod:`._cute_kernel`, which holds the device-side body.  The dispatcher uses the in-tree
``training_engine_tensor.op_dispatcher`` to honour the
``OP_GEMM_FC1`` selection (``v1`` for the engine kernel,
``baseline`` for the eager PyTorch fallback).
"""

import torch

from training_engine_tensor.ops._shared._cute_jit_helper import (
    compile_jit,
    import_cute_modules,
    make_cuda_stream,
)

# Module-level import — required so ``inspect.signature(..., eval_str=True)``
# in the @cute.kernel decorator can resolve ``cutlass.Constexpr`` /
# ``cutlass.Int32`` annotations against module globals. Wrapped in try/
# except so the file can still be parsed on the macOS host where the
# nvidia-cutlass-dsl wheel is absent.
try:
    import cutlass  # type: ignore
except ImportError:
    cutlass = None  # type: ignore

# Super-tile CTA swizzle.  Default row-major CTA scheduling makes
# adjacent CTAs work on different N columns of B, killing L2 reuse of
# the B matrix.  The super-tile schedule groups ``swizzle_m``
# consecutive m-tiles into a "pile" that walks all N_TILES before
# moving to the next pile, so the same (n_tile, k_tile) of B is reused
# by ``swizzle_m`` consecutive CTAs in flight.  Empirically worth
# ~+2-5 pp MFU on all three directions for this shape.
#
# Concurrent-stability swizzle_m caps (per direction):
#   * fwd   grid=8192 → swizzle_m ≤ 16
#   * dgrad grid=2048 → swizzle_m ≤ 16
#   * wgrad grid=4096 → swizzle_m ≤ 8
#
# Mainloop: tile_mn=(128, 256), atom_layout=(2,1,1), threads_per_cta=256;
# SMEM/CTA ≈ 96 KB → 2 CTAs/SM fit in the 228 KB SMEM/SM budget.
PERSISTENT_CONFIGS: dict[str, dict] = {
    "fwd":   {"tile_mn": (128, 256), "ab_stage": 3, "swizzle_m": 16, "k_pipe_mmas": 1, "cluster_mn": (2, 1), "epi_tma_store": True, "epi_smem_overlay": False, "epi_stage": 1},
    "dgrad": {"tile_mn": (128, 256), "ab_stage": 3, "swizzle_m": 16, "k_pipe_mmas": 1, "cluster_mn": (2, 1), "epi_tma_store": True, "epi_smem_overlay": False, "epi_stage": 1},
    # wgrad uses cluster=(2,1) with B-side multicast.  cluster=(2,1)
    # provides not just B-multicast HBM savings but also halved SMEM
    # coordination cost per CTA (peer barriers, single multicast atom
    # issuance), which latency-bound runs see as a steady-state win.
    "wgrad": {"tile_mn": (128, 256), "ab_stage": 4, "swizzle_m": 8,  "k_pipe_mmas": 2, "cluster_mn": (2, 1), "epi_tma_store": False},
}

_CUTE_MODS: tuple | None = None

def _cute():
    global _CUTE_MODS
    if _CUTE_MODS is None:
        _CUTE_MODS = import_cute_modules()
    return _CUTE_MODS

def _torch_to_cutlass(cutlass_mod, dtype: torch.dtype):
    """Map torch dtype → cutlass dtype class for the in-place
    from_dlpack-based input wrapping path used by ``_launch``."""
    if dtype == torch.bfloat16:
        return cutlass_mod.BFloat16
    if dtype == torch.float32:
        return cutlass_mod.Float32
    if dtype == torch.float16:
        return cutlass_mod.Float16
    raise TypeError(f"unsupported torch dtype for CuTe from_dlpack: {dtype}")

def _wrap_for_cute(t: torch.Tensor, cute_mod, cutlass_mod, *,
                   writable: bool, prime: bool = False):
    """Zero-copy wrap a torch tensor as a CuTe tensor when possible.

    Returns (cute_tensor, bounce_tensor).  If the input is 16-byte aligned
    (the TMA / WGMMA requirement), we wrap in-place with ``from_dlpack``
    and ``bounce_tensor`` is ``None`` (no copy needed).  Otherwise we
    allocate a fresh aligned buffer; for inputs we prime it from the
    caller's data, for outputs (``writable=True``) we return the bounce
    so the caller can copy results back after the kernel launch.

    ``prime=True`` (only meaningful with ``writable=True``): also copy
    the caller's existing data INTO the bounce before the kernel runs.
    Required for the in-kernel ACCUMULATE_C path, where the kernel must
    see the previous out_buf contents (so its += semantics are correct)
    even when out_buf isn't 16-byte aligned and we have to detour
    through a bounce buffer.
    """
    if (t.data_ptr() % 16) == 0:
        cute_t = cute_mod.runtime.from_dlpack(t, assumed_align=16)
        cute_t.element_type = _torch_to_cutlass(cutlass_mod, t.dtype)
        return cute_t, None

    bounce = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    if (not writable) or prime:
        bounce.copy_(t)
    cute_t = cute_mod.runtime.from_dlpack(bounce, assumed_align=16)
    cute_t.element_type = _torch_to_cutlass(cutlass_mod, t.dtype)
    return cute_t, (bounce if writable else None)

def _build_kernel(
    M: int, N: int, K: int,
    *,
    c_dtype_torch: torch.dtype,
    BM: int, BN: int,
    AB_STAGE: int,
    K_PIPE_MMAS: int,
    SWIZZLE_M: int = 1,
    ACCUMULATE_C: bool = False,
    CLUSTER_M: int = 1,
    CLUSTER_N: int = 1,
    EPI_TMA_STORE: bool = False,
    EPI_SMEM_OVERLAY: bool = False,
    MMA_INST_TILE_K: int = 4,
    EPI_STAGE_OVERRIDE: int | None = None,
):
    """Construct a JIT-compileable WGMMA SS GEMM for the given shape."""
    cute, cutlass, sm90_utils, pipeline, _ = _cute()
    from cutlass.cute.nvgpu import (  # type: ignore
        cpasync as cpa,
        warpgroup as wg,
    )

    a_dtype = cutlass.BFloat16
    b_dtype = cutlass.BFloat16
    if c_dtype_torch == torch.bfloat16:
        c_dtype = cutlass.BFloat16
    elif c_dtype_torch == torch.float32:
        c_dtype = cutlass.Float32
    else:
        raise TypeError(f"unsupported c_dtype: {c_dtype_torch}")
    acc_dtype = cutlass.Float32

    # In-kernel ACCUMULATE_C requires c_dtype == acc_dtype (FP32) so that the
    # existing C tile can be loaded into a register fragment of the same
    # element type as the WGMMA accumulator, summed, and stored back without
    # round-trip cast. Only the FP32-output direction (wgrad) uses it.
    if ACCUMULATE_C:
        assert c_dtype_torch == torch.float32, (
            "ACCUMULATE_C requires FP32 C buffer (wgrad-only path)"
        )

    # M6 — SMEM-staged StMatrix R2S + TMA store epilogue. Only viable when:
    # - C output is BF16 (cast from FP32 acc), AND
    # - we are NOT accumulating (overwrite-only path; fwd/dgrad).
    # For wgrad (ACCUMULATE_C=True, FP32 c_dtype), M6 is unsupported because
    # StMatrix 8x8x16b expects 16-bit destination and the accumulate semantics
    # would require an SMEM-staged read-modify-write of the existing FP32 C
    # tile (much more complex, deferred). Other paths const-fold M6 out.
    USE_EPI_TMA = bool(EPI_TMA_STORE) and (not ACCUMULATE_C) and (c_dtype is not acc_dtype)
    USE_EPI_TMA_OVERLAY = USE_EPI_TMA and bool(EPI_SMEM_OVERLAY)
    EPI_STAGE = int(EPI_STAGE_OVERRIDE) if EPI_STAGE_OVERRIDE is not None else (2 if USE_EPI_TMA else 1)
    assert EPI_STAGE >= 1, f"EPI_STAGE must be >=1, got {EPI_STAGE}"

    MMA_INST_K = 16
    BK = MMA_INST_K * MMA_INST_TILE_K  # 64 (default) or 32 (fwd shrink)
    assert K % BK == 0, f"K={K} must be divisible by BK={BK}"
    assert MMA_INST_TILE_K in (1, 2, 4, 8), (
        f"MMA_INST_TILE_K must be a small power of two (got {MMA_INST_TILE_K}); "
        "values > 8 risk SMEM swizzle pattern misalignment, values < 1 invalid"
    )
    assert M % BM == 0, f"M={M} must be divisible by BM={BM}"
    assert N % BN == 0, f"N={N} must be divisible by BN={BN}"

    K_TILES = K // BK
    M_TILES = M // BM
    N_TILES = N // BN

    NUM_TOTAL_TILES = M_TILES * N_TILES
    cluster_size = CLUSTER_M * CLUSTER_N
    _target = min(128, NUM_TOTAL_TILES)
    N_PERSISTENT_CTAS = _target
    while N_PERSISTENT_CTAS > 0 and (
        NUM_TOTAL_TILES % N_PERSISTENT_CTAS != 0
        or N_PERSISTENT_CTAS % cluster_size != 0
    ):
        N_PERSISTENT_CTAS -= 1
    assert N_PERSISTENT_CTAS > 0, (
        f"could not find any divisor of NUM_TOTAL_TILES={NUM_TOTAL_TILES} "
        f"that is also divisible by cluster_size={cluster_size}"
    )
    MAX_TILES_PER_CTA = NUM_TOTAL_TILES // N_PERSISTENT_CTAS

    # M4 super-tile swizzle: walk a 1D launch grid of M_TILES * N_TILES
    # CTAs through a pile of GROUP_M consecutive m-tiles × all N_TILES
    # n-tiles before advancing to the next pile. This forces GROUP_M
    # consecutive in-flight CTAs to share the same n_tile of B, hugely
    # improving L2 reuse of B (≈ 2-5 pp MFU per the rules document M4).
    # With M1 persistent CTA the swizzle is applied to the GLOBAL tile id
    # (bidx + tile_iter * N_PERSISTENT_CTAS), so each persistent CTA's
    # successive tiles walk the same pile-then-cross-pile sequence the
    # original non-persistent grid did.
    GROUP_M = max(1, int(SWIZZLE_M))
    # All five GEMM directions in this op have M_TILES divisible by 8,
    # so for swizzle_m ∈ {1, 2, 4, 8} the regular pile covers everything;
    # the const-expr-folded tail branch below is skipped at JIT time.
    M_TILES_REG = (M_TILES // GROUP_M) * GROUP_M
    M_TILES_TAIL = M_TILES - M_TILES_REG
    NUM_REG_CIDS = M_TILES_REG * N_TILES
    PILE_SIZE = GROUP_M * N_TILES

    assert CLUSTER_M >= 1 and CLUSTER_N >= 1
    assert CLUSTER_N == 1, "only cluster_n=1 supported for now"
    assert GROUP_M % CLUSTER_M == 0, (
        f"GROUP_M={GROUP_M} must be a multiple of CLUSTER_M={CLUSTER_M}"
    )
    assert M_TILES % CLUSTER_M == 0, (
        f"M_TILES={M_TILES} must be a multiple of CLUSTER_M={CLUSTER_M}"
    )
    # MCAST_SIZE counts the number of CTAs receiving each pipeline-stage
    # arrival. Each empty-barrier release fires from every CTA in the
    # cluster (via is_signalling_thread under cluster), so consumer
    # arrive_count = MCAST_SIZE * NUM_MMA_WARPS_PER_CTA (MMA warps only;
    # the DMA WG never calls consumer_release in M2).
    # mcast_a = 1 (A unicast), mcast_b = CLUSTER_M (B multicast along M).
    MCAST_SIZE = 1 + CLUSTER_M - 1  # = CLUSTER_M for our (mcast_b=2, mcast_a=1)

    assert BM % 64 == 0, f"BM={BM} must be divisible by 64 (WGMMA m64)"
    NUM_MMA_WG = BM // 64
    assert NUM_MMA_WG in (1, 2), f"only 1 or 2 MMA warpgroups supported, got {NUM_MMA_WG}"
    NUM_DMA_WG = 1
    NUM_WG = NUM_DMA_WG + NUM_MMA_WG  # = 3 for BM=128 (the standard config)
    THREADS_PER_WG = 128
    threads_per_cta = NUM_WG * THREADS_PER_WG  # = 384
    NUM_DMA_THREADS = NUM_DMA_WG * THREADS_PER_WG  # = 128
    NUM_MMA_THREADS = NUM_MMA_WG * THREADS_PER_WG  # = 256
    # Only MMA warps call consumer_release; consumer_group.count counts MMA
    # warp arrivals, not total CTA warps. Under cluster=(1,1,1,1) one
    # signalling thread per warp arrives, so consumer count = MCAST_SIZE *
    # NUM_MMA_WARPS_PER_CTA = MCAST_SIZE * 8 (same number as 's
    # cooperative-MMA setup; only the "what gets counted" semantic shifted).
    NUM_MMA_WARPS_PER_CTA = NUM_MMA_THREADS // 32  # = 8
    # Reg dealloc/alloc targets — Hopper canonical values for DMA / MMA WG split.
    DMA_REG_COUNT = 40
    MMA_REG_COUNT = 232

    @cute.kernel
    def gemm_device(
        a_tma_atom, a_tma_tensor,
        b_tma_atom, b_tma_tensor,
        c_gmem,
        tiled_mma,
        a_smem_layout_staged, b_smem_layout_staged,
        SharedStorage: cutlass.Constexpr,
        tx_count: cutlass.Constexpr,
        # M6 args. c_tma_atom / c_tma_tensor are runtime values (their
        # internal coord_tensor is captured by the host-side @cute.jit
        # frame and must be passed through normal runtime args so it
        # lives inside the device-kernel region — marking them Constexpr
        # leaks the outer-region SSA into the launch body and trips the
        # 'cute.get_layout op using value defined outside the region'
        # IR verification error. Only the Python-level None placeholders
        # (set by _build_kernel when USE_EPI_TMA=False) need to flow
        # through, and runtime args happily accept None — CuTeDSL only
        # introspects the value when it's actually used, which the
        # USE_EPI_TMA const_expr branch prevents on the non-M6 paths.
        c_tma_atom,
        c_tma_tensor,
        epi_smem_layout_staged,
        # ``c_layout_enum`` is a plain Python enum value (LayoutEnum) — fully
        # constexpr-safe. We deliberately do NOT pass the StMatrix CopyAtom
        # constructed host-side, because that atom carries an outer-region
        # SSA value which would trip 'cute.make_tiled_copy op using value
        # defined outside the region' inside @cute.kernel. Instead we
        # re-construct the StMatrix op INSIDE this device region using the
        # constexpr enum + dtypes (sm90_get_smem_store_op is JIT-traceable).
        c_layout_enum: cutlass.Constexpr,
        epi_tile_mn: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # M2 warp-specialization: split CTA warps into DMA producer + MMA
        # consumer cohorts using `warp_idx` directly (matches the canonical
        # flash_bwd / cute_kernel pattern). NUM_DMA_WG=1 → DMA WG is
        # warps 0-3; NUM_MMA_WG=2 → MMA WGs are warps 4-11 (for BM=128).
        #
        # NOTE: deliberately NOT computing warp_group_idx via
        # `tidx // THREADS_PER_WG` — that intermediate variable proved
        # to make the CuTeDSL 4.5.0 IR-flatten pass mis-classify the
        # downstream `if is_dma_wg:` predicate (errors with "if
        # statement encountered a user-defined Python object").
        # `warp_idx < 4` is the warp-uniform compare CuTeDSL handles
        # cleanly (flash_bwd.py line 1026 uses identical pattern).
        #
        # NUM_DMA_WARPS = NUM_DMA_WG * 4 is the warp-index cutoff
        # between DMA and MMA cohorts.
        NUM_DMA_WARPS = NUM_DMA_WG * 4  # = 4 for single DMA WG

        # M1 : PERSISTENT CTA loop. The 1D launch grid has
        # N_PERSISTENT_CTAS CTAs (compile-time const = 128 chosen to
        # divide all 3 production grids). Each CTA walks
        # MAX_TILES_PER_CTA = NUM_TOTAL_TILES / N_PERSISTENT_CTAS output
        # tiles, computing per-tile (m_tile, n_tile) via the existing
        # super-tile swizzle applied to the GLOBAL tile id:
        #
        # tile_id_global = bidx + tile_iter * N_PERSISTENT_CTAS
        #
        # The swizzle is now inside each warp-spec branch (DMA / MMA),
        # since both branches need it per tile_iter and CuTeDSL keeps
        # iter_args properly tracked inside `cutlass.range` loops.

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ``make_smem_layout_a`` returns a ``ComposedLayout`` (swizzle outer
        # composed with a plain inner layout); split it so the resulting
        # ``sA`` tensor has an AFFINE layout, which is what
        # ``make_fragment_A`` requires for WGMMA SS. See fmha.py Q/K/V smem
        # construction for the canonical pattern.
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner,
        )
        if cutlass.const_expr(USE_EPI_TMA):
            if cutlass.const_expr(USE_EPI_TMA_OVERLAY):
                sC = storage.sA.get_tensor(
                    epi_smem_layout_staged.outer,
                    swizzle=epi_smem_layout_staged.inner,
                    dtype=c_dtype,
                )
            else:
                sC = storage.sC.get_tensor(
                    epi_smem_layout_staged.outer,
                    swizzle=epi_smem_layout_staged.inner,
                )
            sC0 = sC[(None, None, 0)]
        else:
            sC = None
            sC0 = None

        # Producer = DMA WG warp 0, single TMA-issuing thread via
        # elect_one_sync (M2 canonical) — count=1.
        # Consumer = MCAST_SIZE * NUM_MMA_WARPS_PER_CTA signalling MMA
        # threads (one per MMA warp per cluster-CTA, per
        # PipelineTmaAsync.init_empty_barrier_arrive_signal which gates
        # the empty-barrier arrive on tidx % 32 < cluster_size). DMA WG
        # does not call consumer_release, so it does not contribute to
        # the arrive_count.
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, MCAST_SIZE * NUM_MMA_WARPS_PER_CTA,
        )

        if cutlass.const_expr(CLUSTER_M > 1 or CLUSTER_N > 1):
            cta_layout_vmnk = cute.make_layout((1, CLUSTER_M, CLUSTER_N, 1))
            ab_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar.data_ptr(),
                num_stages=AB_STAGE,
                producer_group=producer_group,
                consumer_group=consumer_group,
                tx_count=tx_count,
                cta_layout_vmnk=cta_layout_vmnk,
            )
        else:
            ab_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar.data_ptr(),
                num_stages=AB_STAGE,
                producer_group=producer_group,
                consumer_group=consumer_group,
                tx_count=tx_count,
            )

        # cta_tiler and B-multicast mask are tile-invariant; build once
        # and reuse across the persistent loop. The per-tile gA/gB/gC
        # views (and their tma_partition slices) are built INSIDE the
        # persistent loop because they depend on the runtime tile_coord.
        cta_tiler = (BM, BN, BK)
        if cutlass.const_expr(CLUSTER_M > 1):
            block_in_cluster_x, _, _ = cute.arch.block_in_cluster_idx()
            cta_layout_mnk = cute.make_layout((CLUSTER_M, CLUSTER_N, 1))
            cluster_coord_mnk = (block_in_cluster_x, 0, 0)
            b_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord_mnk, mode=0,
            )
        else:
            block_in_cluster_x = cutlass.Int32(0)
            b_mcast_mask = cutlass.Int32(0)
        cta_layout_v = cute.make_layout(1)
        if cutlass.const_expr(CLUSTER_M > 1):
            b_cta_layout_mc = cute.make_layout(CLUSTER_M)
        else:
            b_cta_layout_mc = cta_layout_v

        # NOTE: MMA partition (thr_mma / tCsA / tCsB / tCrA / tCrB) is
        # deferred into the MMA branch below — only MMA threads need the
        # per-thread WGMMA fragments.

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, AB_STAGE,
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, AB_STAGE,
        )
        # release_state lags cons_state by K_PIPE_MMAS+1 iterations: each
        # consumer_release flips its (idx, phase) for the stage whose WGMMA
        # group just retired (via wait_group(K_PIPE_MMAS)), allowing the
        # producer to refill that slot.
        release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, AB_STAGE,
        )

        # Pipeline layout:
        #
        # AB_STAGE = PREFETCH + K_PIPE_MMAS + 1
        #
        # where PREFETCH = number of TMA's prologue-issued ahead of consumer.
        # Iter k issues TMA for tile (k + PREFETCH). In steady state
        # K_PIPE_MMAS WGMMA commits are in flight; the oldest retires at
        # iter k via wait_group(K_PIPE_MMAS), freeing slot (k - K_PIPE_MMAS)
        # so producer can refill it with tile (k + PREFETCH) on the same iter.
        PREFETCH = AB_STAGE - K_PIPE_MMAS - 1
        # Cluster init sync: ensure all cluster CTAs have created the
        # pipeline barriers before any of them begins issuing TMA loads
        # (otherwise the multicast peer barrier may be uninitialised
        # when we sign it). No-op for cluster=1.
        if cutlass.const_expr(CLUSTER_M > 1 or CLUSTER_N > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        if warp_idx < NUM_DMA_WARPS:
            # ============ DMA PRODUCER WARP GROUP (warps 0..NUM_DMA_WARPS-1) ============
            cute.arch.warpgroup_reg_dealloc(DMA_REG_COUNT)

            if warp_idx == 0:
                # TMA descriptor prefetch — gated on cluster=(1,1) only;
                if cutlass.const_expr(CLUSTER_M == 1 and CLUSTER_N == 1):
                    cpa.prefetch_descriptor(a_tma_atom)
                    cpa.prefetch_descriptor(b_tma_atom)

                # ===== M1 PERSISTENT LOOP (DMA producer) =====
                # Each persistent CTA walks MAX_TILES_PER_CTA output tiles.
                # tile_iter is constexpr-bounded; tile_id_global is
                # runtime (depends on bidx). Per tile, we resolve
                # (m_tile, n_tile) via the same super-tile swizzle the
                # non-persistent kernel used, rebuild the per-tile
                # gA/gB views + tma_partition slices, then issue K_TILES
                # TMA loads. Pipeline state (prod_state) flows across
                # tile boundaries — it never resets.
                for tile_iter in cutlass.range(MAX_TILES_PER_CTA, unroll=1):
                    tile_id_global = bidx + tile_iter * N_PERSISTENT_CTAS

                    # Resolve (m_tile, n_tile) via super-tile swizzle.
                    if cutlass.const_expr(GROUP_M > 1):
                        pile_idx = tile_id_global // PILE_SIZE
                        in_pile = tile_id_global % PILE_SIZE
                        m_tile = pile_idx * GROUP_M + (in_pile % GROUP_M)
                        n_tile = in_pile // GROUP_M
                    else:
                        m_tile = tile_id_global % M_TILES
                        n_tile = tile_id_global // M_TILES

                    # Per-tile gA/gB and tma_partition (cheap — pure
                    # layout math, no SMEM/register state changes).
                    tile_coord = (m_tile, n_tile, None)
                    gA = cute.local_tile(
                        a_tma_tensor, tiler=cta_tiler, coord=tile_coord,
                        proj=(1, None, 1),
                    )
                    gB = cute.local_tile(
                        b_tma_tensor, tiler=cta_tiler, coord=tile_coord,
                        proj=(None, 1, 1),
                    )
                    tma_a_dst, tma_a_src = cpa.tma_partition(
                        a_tma_atom, 0, cta_layout_v,
                        cute.group_modes(sA, 0, 2),
                        cute.group_modes(gA, 0, 2),
                    )
                    if cutlass.const_expr(CLUSTER_M > 1):
                        tma_b_dst, tma_b_src = cpa.tma_partition(
                            b_tma_atom, block_in_cluster_x, b_cta_layout_mc,
                            cute.group_modes(sB, 0, 2),
                            cute.group_modes(gB, 0, 2),
                        )
                    else:
                        tma_b_dst, tma_b_src = cpa.tma_partition(
                            b_tma_atom, 0, cta_layout_v,
                            cute.group_modes(sB, 0, 2),
                            cute.group_modes(gB, 0, 2),
                        )

                    # Inner K-loop: K_TILES TMA loads. producer_acquire
                    # naturally throttles to MMA cadence; pipeline state
                    # carries across tile boundaries.
                    for k_load in cutlass.range(K_TILES, unroll=1):
                        ab_pipeline.producer_acquire(prod_state)
                        cute.copy(
                            a_tma_atom,
                            tma_a_src[(None, k_load)],
                            tma_a_dst[(None, prod_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(prod_state),
                        )
                        if cutlass.const_expr(CLUSTER_M > 1):
                            cute.copy(
                                b_tma_atom,
                                tma_b_src[(None, k_load)],
                                tma_b_dst[(None, prod_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(prod_state),
                                mcast_mask=b_mcast_mask,
                            )
                        else:
                            cute.copy(
                                b_tma_atom,
                                tma_b_src[(None, k_load)],
                                tma_b_dst[(None, prod_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(prod_state),
                            )
                        prod_state.advance()
            # DMA WG done — falls through past the MMA branch.
        else:
            # ============ MMA CONSUMER WARP GROUPS (warps NUM_DMA_WARPS..) ============
            cute.arch.warpgroup_reg_alloc(MMA_REG_COUNT)

            # MMA-cohort-local thread/warp indices (tile-invariant).
            mma_tidx = tidx - NUM_DMA_THREADS
            mma_warp_idx = cute.arch.make_warp_uniform(
                warp_idx - NUM_DMA_WARPS
            )

            # Tile-invariant MMA partitions: tCsA / tCsB / tCrA / tCrB
            # operate on sA/sB (which are the SMEM ring buffer, NOT
            # tile-dependent). Built once, reused across all tiles.
            thr_mma = tiled_mma.get_slice(mma_tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tiled_mma.set(wg.Field.ACCUMULATE, True)

            # M6 epilogue: r2s atom + tiled_copy_r2s + per-thread (sC dest,
            # acc source) partitions. sC0 / tRS_sC / rD are tile-invariant
            # (sC is a fixed SMEM region). Built once outside the persistent
            # loop.
            c_dtype_bits = 32 if c_dtype is acc_dtype else 16
            epi_vec_bits = 2 * c_dtype_bits
            copy_atom_c = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), c_dtype,
                num_bits_per_copy=epi_vec_bits,
            )
            if cutlass.const_expr(USE_EPI_TMA):
                r2s_atom = sm90_utils.sm90_get_smem_store_op(
                    c_layout_enum, c_dtype, acc_dtype,
                )
                tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
                    r2s_atom, tiled_mma,
                )
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    r2s_atom, tiled_copy_C_atom,
                )
                thr_copy_r2s = tiled_copy_r2s.get_slice(mma_tidx)
                tRS_sC_staged = thr_copy_r2s.partition_D(sC)
                rD = cute.make_fragment_like(
                    thr_copy_r2s.partition_D(sC0), c_dtype,
                )
                epi_m_tiles = BM // epi_tile_mn[0]
                epi_n_tiles = BN // epi_tile_mn[1]

                epilog_barrier = pipeline.NamedBarrier(
                    barrier_id=1, num_threads=NUM_MMA_THREADS,
                )

                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, NUM_MMA_THREADS,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=EPI_STAGE,
                    producer_group=tma_store_producer_group,
                )

            # ===== M1 PERSISTENT LOOP (MMA consumer) =====
            # Walk MAX_TILES_PER_CTA output tiles. Each tile: rebuild
            # tile-coord-dependent partitions (tCgC, gC_tma, gC_epi,
            # tma_c_smem/gmem, tCrC), reset tCrC accumulator, run K_TILES
            # WGMMA inner loop, drain pipeline, cluster fence, epilogue.
            # Pipeline state (cons_state, release_state) carries across
            # tile boundaries.
            for tile_iter in cutlass.range(MAX_TILES_PER_CTA, unroll=1):
                tile_id_global = bidx + tile_iter * N_PERSISTENT_CTAS

                # Same swizzle as DMA branch (lock-step tile sequence).
                if cutlass.const_expr(GROUP_M > 1):
                    pile_idx = tile_id_global // PILE_SIZE
                    in_pile = tile_id_global % PILE_SIZE
                    m_tile = pile_idx * GROUP_M + (in_pile % GROUP_M)
                    n_tile = in_pile // GROUP_M
                else:
                    m_tile = tile_id_global % M_TILES
                    n_tile = tile_id_global // M_TILES

                tile_coord = (m_tile, n_tile, None)
                gC = cute.local_tile(
                    c_gmem, tiler=cta_tiler, coord=tile_coord,
                    proj=(1, 1, None),
                )
                tCgC = thr_mma.partition_C(gC)
                tCrC = tiled_mma.make_fragment_C(tCgC)
                tCrC.fill(0.0)

                # ---- WGMMA mainloop (consumes pipeline stages from DMA WG) ----
                for k_iter in cutlass.range(K_TILES, unroll=1):
                    if k_iter >= K_PIPE_MMAS + 1:
                        wg.wait_group(K_PIPE_MMAS)
                        ab_pipeline.consumer_release(release_state)
                        release_state.advance()

                    ab_pipeline.consumer_wait(cons_state)
                    wg.fence()
                    cute.gemm(
                        tiled_mma,
                        tCrC,
                        tCrA[(None, None, None, cons_state.index)],
                        tCrB[(None, None, None, cons_state.index)],
                        tCrC,
                    )
                    wg.commit_group()
                    cons_state.advance()

                # Drain remaining K_PIPE_MMAS+1 WGMMA commits + release
                # their stages. Pipeline state continues to next tile.
                wg.wait_group(0)
                for _ in cutlass.range(K_PIPE_MMAS + 1, unroll=1):
                    ab_pipeline.consumer_release(release_state)
                    release_state.advance()

                # NOTE: per-tile cluster_arrive/wait was moved OUT of the
                # persistent loop and placed AFTER all tiles complete (see
                # below).  Per-tile cluster fences were observed to deadlock
                # wgrad cluster=(2,1) under high concurrency; a single
                # end-of-kernel cluster_arrive is the verified-safe pattern.
                # Persistent CTAs only need ONE arrival per CTA per kernel
                # launch to coordinate exit (mbarriers in SMEM must stay
                # valid until cluster peers finish their last multicast
                # TMA; one barrier at the end suffices because all tiles
                # share the same SMEM ring buffer and the MMA WG finishes
                # after the DMA WG).

                # ---- Epilogue (per tile) ----
                if cutlass.const_expr(USE_EPI_TMA):
                    # M6 + M5 — SMEM-staged StMatrix R2S + PIPELINED
                    # TMA store across 2 sC slots. Each tile's R2S
                    # alternates between sC[0] and sC[1] indexed by
                    # ``epi_buffer = tile_iter % EPI_STAGE``; the TMA
                    # store of THIS tile's slot is issued, then the
                    # producer_acquire for the OTHER slot waits for THAT
                    # slot's prior store (from 1 tile ago) to land in
                    # GMEM. Net effect: the per-tile epilogue no
                    # longer stalls the MMA WG on cp_async_bulk_wait_
                    # group(0); the wait is hidden under the NEXT tile's
                    # WGMMA mainloop.
                    tRS_rAcc = thr_copy_r2s.retile(tCrC)

                    # tile_iter is the cutlass.range loop var (runtime
                    # Int32); EPI_STAGE is a compile-time const = 2 so
                    # ``% EPI_STAGE`` const-folds to a bitwise AND on
                    # the IR side.
                    epi_buffer = tile_iter % EPI_STAGE

                    gC_tma = cute.local_tile(
                        c_tma_tensor, tiler=cta_tiler, coord=tile_coord,
                        proj=(1, 1, None),
                    )
                    gC_epi = cute.zipped_divide(gC_tma, epi_tile_mn)
                    # Partition the FULL sC (with stage dim) so the
                    # per-tile cute.copy can slot-index into the correct
                    # epi_buffer. Matches gemm_fc2's tma_partition
                    # pattern (kernel.py L1989).
                    tma_c_smem, tma_c_gmem = cpa.tma_partition(
                        c_tma_atom, 0, cta_layout_v,
                        cute.group_modes(sC, 0, 2),
                        gC_epi,
                    )

                    for epi_n_idx in cutlass.range_constexpr(epi_n_tiles):
                        for epi_m_idx in cutlass.range_constexpr(epi_m_tiles):
                            rD.store(
                                tRS_rAcc[(None, epi_m_idx, epi_n_idx)].load().to(c_dtype)
                            )
                            # Slot-aware R2S — sC[epi_buffer] is the
                            # current tile's destination; the OTHER slot
                            # may still be hosting an in-flight TMA store.
                            cute.copy(
                                r2s_atom, rD,
                                tRS_sC_staged[(None, None, None, epi_buffer)],
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            # MMA-only barrier (NOT CTA-wide __syncthreads):
                            # awaits all 256 MMA threads finishing R2S into
                            # sC[epi_buffer] before mma_warp_idx==0 issues
                            # the TMA store.
                            epilog_barrier.arrive_and_wait()

                            if mma_warp_idx == 0:
                                cute.copy(
                                    c_tma_atom,
                                    tma_c_smem[(None, epi_buffer)],
                                    tma_c_gmem[(None, (epi_m_idx, epi_n_idx))],
                                )
                                # producer_commit signals "store of
                                # epi_buffer has been ISSUED" (not
                                # completed); producer_acquire then
                                # waits for the OTHER slot's prior
                                # store to drain before the next tile
                                # may overwrite it.
                                tma_store_pipeline.producer_commit()
                                tma_store_pipeline.producer_acquire()

                            # NOTE: no second epilog_barrier here — the
                            # next tile's mainloop is allowed to start
                            # IMMEDIATELY (the only ordering needed is
                            # "this tile's R2S finished before this
                            # tile's TMA store reads sC[epi_buffer]",
                            # which is already enforced by the barrier
                            # ABOVE the cute.copy). The 'is this slot
                            # safe to reuse' guarantee comes from
                            # producer_acquire on a SUBSEQUENT tile that
                            # cycles back to the same slot.
                elif cutlass.const_expr(ACCUMULATE_C):
                    tCrC_existing = cute.make_fragment_like(tCrC, c_dtype)
                    cute.copy(copy_atom_c, tCgC, tCrC_existing)
                    tCrC.store(tCrC.load() + tCrC_existing.load())
                    cute.copy(copy_atom_c, tCrC, tCgC)
                elif cutlass.const_expr(c_dtype != acc_dtype):
                    tCrC_c = cute.make_fragment_like(tCrC, c_dtype)
                    tCrC_c.store(tCrC.load().to(c_dtype))
                    cute.copy(copy_atom_c, tCrC_c, tCgC)
                else:
                    cute.copy(copy_atom_c, tCrC, tCgC)

            if cutlass.const_expr(USE_EPI_TMA):
                if mma_warp_idx == 0:
                    tma_store_pipeline.producer_tail()

            # M5 cluster fence (single end-of-kernel; no-op for cluster=1).
            if cutlass.const_expr(CLUSTER_M > 1 or CLUSTER_N > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

    @cute.jit
    def gemm_kernel(
        a_gmem,
        b_gmem,
        c_gmem,
        stream,
    ):
        a_layout_enum = cutlass.utils.LayoutEnum.from_tensor(a_gmem)
        b_layout_enum = cutlass.utils.LayoutEnum.from_tensor(b_gmem)
        c_layout_enum = cutlass.utils.LayoutEnum.from_tensor(c_gmem)

        # M3 cooperative MMA across the MMA warp groups: atom_layout=
        # (NUM_MMA_WG, 1, 1) splits BM across NUM_MMA_WG MMA warpgroups via
        # the M atom replica. The WGMMA atom itself is m64×BN×k16, so the
        # tiler_mn passed to ``make_trivial_tiled_mma`` is the per-atom
        # shape (64, BN), NOT the full CTA tile. The DMA WG is NOT
        # included in atom_layout — only the MMA WGs execute WGMMA.
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype,
            a_layout_enum.sm90_mma_major_mode(),
            b_layout_enum.sm90_mma_major_mode(),
            acc_dtype,
            atom_layout_mnk=(NUM_MMA_WG, 1, 1),
            tiler_mn=(64, BN),
        )

        tile_shape_mnk = (BM, BN, BK)
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout_enum, tile_shape_mnk, a_dtype, AB_STAGE,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout_enum, tile_shape_mnk, b_dtype, AB_STAGE,
        )
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        a_tma_atom, a_tma_tensor = cpa.make_tiled_tma_atom(
            cpa.CopyBulkTensorTileG2SOp(),
            a_gmem, a_smem_layout, (BM, BK),
        )
        # M5 cluster: B is multicast along M (each cluster has CLUSTER_M
        # CTAs sharing the same n_tile/k_tile of B → one HBM read fans
        # out to all of them). num_multicast=CLUSTER_M; when CLUSTER_M=1
        # this const-folds to plain unicast.
        if cutlass.const_expr(CLUSTER_M > 1):
            b_tma_atom, b_tma_tensor = cpa.make_tiled_tma_atom(
                cpa.CopyBulkTensorTileG2SMulticastOp(),
                b_gmem, b_smem_layout, (BN, BK),
                num_multicast=CLUSTER_M,
            )
        else:
            b_tma_atom, b_tma_tensor = cpa.make_tiled_tma_atom(
                cpa.CopyBulkTensorTileG2SOp(),
                b_gmem, b_smem_layout, (BN, BK),
            )

        # M6 — epi_tile + epi_smem_layout + TMA store atom. Built only when
        # USE_EPI_TMA (gated above in _build_kernel); other paths get None
        # placeholders that const-fold out in gemm_device.
        if cutlass.const_expr(USE_EPI_TMA):
            assert EPI_STAGE >= 1
            # Use a SINGLE epi sub-tile = full CTA tile. This collapses
            # the (EPI_M_TILES, EPI_N_TILES) iteration to (1, 1) and makes
            # the per-thread partition shapes of (sC dest, acc source)
            # both span the full tile, eliminating shape-mismatch between
            # the WGMMA-blocked accumulator retile (whose trailing modes
            # are MMA_M × MMA_N, NOT EPI_M_TILES × EPI_N_TILES) and the
            # StMatrix-partitioned SMEM destination. SMEM cost is just
            # BM·BN·sizeof(c_dtype) = 128·128·2 = 32 KiB, well within
            # the SM90 228 KiB budget (sA+sB stages occupy ~144 KiB).
            epi_tile = (BM, BN)
            epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
                c_dtype, c_layout_enum, epi_tile, EPI_STAGE,
            )
            epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
            c_tma_atom, c_tma_tensor = cpa.make_tiled_tma_atom(
                cpa.CopyBulkTensorTileS2GOp(),
                c_gmem, epi_smem_layout, epi_tile,
            )
            # NOTE: the StMatrix CopyAtom itself is constructed INSIDE the
            # device kernel (see gemm_device USE_EPI_TMA branch). Building
            # it here on the host side and passing it down leaks an outer-
            # region SSA into the launch body and trips region isolation.

            if cutlass.const_expr(USE_EPI_TMA_OVERLAY):
                @cute.struct
                class SharedStorage:
                    mbar: cute.struct.MemRange[cutlass.Int64, 2 * AB_STAGE]
                    sA: cute.struct.Align[
                        cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                        1024,
                    ]
                    sB: cute.struct.Align[
                        cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                        1024,
                    ]
            else:
                @cute.struct
                class SharedStorage:
                    mbar: cute.struct.MemRange[cutlass.Int64, 2 * AB_STAGE]
                    sA: cute.struct.Align[
                        cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                        1024,
                    ]
                    sB: cute.struct.Align[
                        cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                        1024,
                    ]
                    sC: cute.struct.Align[
                        cute.struct.MemRange[c_dtype, cute.cosize(epi_smem_layout_staged)],
                        1024,
                    ]
        else:
            # Non-M6 path: keep SharedStorage byte-identical to the prior
            # build so dgrad/wgrad cache keys produce the same SMEM layout.
            epi_smem_layout_staged = None
            c_tma_atom = None
            c_tma_tensor = None
            epi_tile = (0, 0)

            @cute.struct
            class SharedStorage:
                mbar: cute.struct.MemRange[cutlass.Int64, 2 * AB_STAGE]
                sA: cute.struct.Align[
                    cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                    1024,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                    1024,
                ]

        a_bytes = cute.size_in_bytes(a_dtype, a_smem_layout)
        b_bytes = cute.size_in_bytes(b_dtype, b_smem_layout)
        tx_count = a_bytes + b_bytes

        grid_mn = (N_PERSISTENT_CTAS, 1, 1)

        gemm_device(
            a_tma_atom, a_tma_tensor,
            b_tma_atom, b_tma_tensor,
            c_gmem,
            tiled_mma,
            a_smem_layout_staged, b_smem_layout_staged,
            SharedStorage,
            tx_count,
            c_tma_atom, c_tma_tensor,
            epi_smem_layout_staged,
            c_layout_enum,
            epi_tile,
        ).launch(
            grid=grid_mn,
            block=[threads_per_cta, 1, 1],
            cluster=(CLUSTER_M, CLUSTER_N, 1),
            stream=stream,
        )

    return gemm_kernel

_KERNEL_CACHE: dict = {}

# dgrad calls ``w.t().contiguous`` which allocates + copies a 128MB BF16
# tensor per call (w is 4096×16384 BF16). Since `w` is the SAME tensor
# across all GAS=8 micro-batches within a step (and across many steps in
# the ProdPool bench), cache by (data_ptr, shape) to amortise the copy.
_W_T_CACHE: dict = {}

def _w_t_cached(w: torch.Tensor) -> torch.Tensor:
    """Return a contiguous ``w.t()`` tensor, cached per underlying buffer.

    Production passes the same ``w`` repeatedly; allocating + copying a
    transposed contiguous view on every dgrad call costs ~85 µs/call.
    Keying on ``data_ptr() + shape`` is safe because torch's caching
    allocator never recycles a live tensor's pointer; the cached entry
    naturally invalidates when ``w`` itself is freed (the cached
    transpose holds its own buffer so this dict can outlive ``w``, but
    next call with a fresh ``w`` lands on a different key).
    """
    key = (w.data_ptr(), w.shape, w.dtype, w.device.index)
    cached = _W_T_CACHE.get(key)
    if cached is not None:
        return cached
    wt = w.t().contiguous()
    _W_T_CACHE[key] = wt
    return wt

def _get_kernel(M, N, K, c_dtype_torch, BM, BN, AB_STAGE, K_PIPE_MMAS,
                SWIZZLE_M=1, ACCUMULATE_C=False, LAYOUT_TAG=None,
                CLUSTER_M=1, CLUSTER_N=1, EPI_TMA_STORE=False,
                MMA_INST_TILE_K=4, EPI_SMEM_OVERLAY=False,
                EPI_STAGE_OVERRIDE=None):
    # MMA_INST_TILE_K participates in the cache key so a fwd path with
    # ``mma_inst_tile_k=2`` (BK=32) gets a distinct compiled kernel
    # from dgrad/wgrad's default ``mma_inst_tile_k=4`` (BK=64).
    key = (M, N, K, c_dtype_torch, BM, BN, AB_STAGE, K_PIPE_MMAS,
           SWIZZLE_M, ACCUMULATE_C, LAYOUT_TAG, CLUSTER_M, CLUSTER_N,
           EPI_TMA_STORE, MMA_INST_TILE_K, EPI_SMEM_OVERLAY,
           EPI_STAGE_OVERRIDE)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    k = _build_kernel(
        M, N, K,
        c_dtype_torch=c_dtype_torch,
        BM=BM, BN=BN, AB_STAGE=AB_STAGE, K_PIPE_MMAS=K_PIPE_MMAS,
        SWIZZLE_M=SWIZZLE_M,
        ACCUMULATE_C=ACCUMULATE_C,
        CLUSTER_M=CLUSTER_M,
        CLUSTER_N=CLUSTER_N,
        EPI_TMA_STORE=EPI_TMA_STORE,
        MMA_INST_TILE_K=MMA_INST_TILE_K,
        EPI_SMEM_OVERLAY=EPI_SMEM_OVERLAY,
        EPI_STAGE_OVERRIDE=EPI_STAGE_OVERRIDE,
    )
    _KERNEL_CACHE[key] = k
    return k

def _launch(a_t: torch.Tensor, b_t: torch.Tensor, c_t: torch.Tensor,
            cfg: dict, *, accumulate_c: bool = False):
    """Launch the generic K-major × K-major GEMM kernel.

    a_t / b_t: 2D BF16, K-major (last dim = K, stride 1).
    c_t:       2D N-major (last dim = N, stride 1).  BF16 or FP32.

    ``accumulate_c=True``: kernel does ``c_t += A @ B`` (load existing
    c_t, add to WGMMA acc, store back).  Requires c_t.dtype == FP32.
    """
    M, K = a_t.shape
    N, K_b = b_t.shape
    assert K == K_b, f"K mismatch: A.K={K} vs B.K={K_b}"
    assert c_t.shape == (M, N), f"C shape {c_t.shape} != ({M},{N})"
    BM, BN = cfg["tile_mn"]
    AB_STAGE = cfg["ab_stage"]
    K_PIPE_MMAS = cfg["k_pipe_mmas"]
    SWIZZLE_M = int(cfg.get("swizzle_m", 1))
    CLUSTER_M, CLUSTER_N = cfg.get("cluster_mn", (1, 1))
    EPI_TMA_STORE = bool(cfg.get("epi_tma_store", False))
    EPI_SMEM_OVERLAY = bool(cfg.get("epi_smem_overlay", False))
    # BK = MMA_INST_K(=16) * MMA_INST_TILE_K (default 4 → BK=64).
    MMA_INST_TILE_K = int(cfg.get("mma_inst_tile_k", 4))
    EPI_STAGE_OVERRIDE = cfg.get("epi_stage", None)
    if N % BN != 0 and BN > 128 and N % 128 == 0:
        BN = 128
    if M % BM != 0 and BM > 64 and M % 64 == 0:
        BM = 64

    # Input majorness (K-major vs MN-major) is baked into the compiled
    # kernel at compile_jit time (via LayoutEnum.from_tensor), so we
    # tag the cache key with each operand's "K-major bit": a future
    # caller hitting the same (M, N, K, ...) with a different layout
    # gets a fresh build instead of stomping the cached one.
    # K-major ≡ stride[-1] == 1; MN-major ≡ stride[0] == 1.
    a_k_major = int(a_t.stride(-1) == 1)
    b_k_major = int(b_t.stride(-1) == 1)
    c_n_major = int(c_t.stride(-1) == 1)
    layout_tag = (a_k_major, b_k_major, c_n_major)

    kernel = _get_kernel(
        M, N, K, c_t.dtype, BM, BN, AB_STAGE, K_PIPE_MMAS,
        SWIZZLE_M=SWIZZLE_M,
        ACCUMULATE_C=accumulate_c,
        LAYOUT_TAG=layout_tag,
        CLUSTER_M=CLUSTER_M,
        CLUSTER_N=CLUSTER_N,
        EPI_TMA_STORE=EPI_TMA_STORE,
        MMA_INST_TILE_K=MMA_INST_TILE_K,
        EPI_SMEM_OVERLAY=EPI_SMEM_OVERLAY,
        EPI_STAGE_OVERRIDE=EPI_STAGE_OVERRIDE,
    )

    # Wrap torch tensors with from_dlpack so the resulting CuTe tensors
    # ALIAS the underlying torch storage (zero-copy fast path). TMA
    # descriptors require the data pointer to be 16-byte aligned; when
    # the caller hands us a non-aligned slice (e.g. a misaligned
    # wgrad / dgrad output view), we transparently fall back to a
    # bounce buffer and copy results back after launch. For the
    # accumulate path we also prime the bounce with the existing C
    # contents so the kernel's += has the right base value.
    cute, cutlass, _, _, _ = _cute()
    a_cute, _a_bounce = _wrap_for_cute(a_t, cute, cutlass, writable=False)
    b_cute, _b_bounce = _wrap_for_cute(b_t, cute, cutlass, writable=False)
    c_cute, c_bounce = _wrap_for_cute(
        c_t, cute, cutlass, writable=True, prime=accumulate_c,
    )

    stream = make_cuda_stream(torch.cuda.current_stream())

    compiled = compile_jit(
        kernel, a_cute, b_cute, c_cute, stream,
        cache_key=("gemm_fc1_kernel", M, N, K, str(c_t.dtype),
                   BM, BN, AB_STAGE, K_PIPE_MMAS, SWIZZLE_M,
                   accumulate_c, layout_tag, CLUSTER_M, CLUSTER_N,
                   EPI_TMA_STORE, MMA_INST_TILE_K, EPI_SMEM_OVERLAY,
                   EPI_STAGE_OVERRIDE),
    )
    compiled(a_cute, b_cute, c_cute, stream)
    if c_bounce is not None:
        c_t.copy_(c_bounce)

def gemm_fc1_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[S,B,O] = x[S,B,I] @ w[O,I].t()  (BF16 → BF16)."""
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, \
        f"BF16 inputs required; got x.dtype={x.dtype} w.dtype={w.dtype}"
    assert x.dim() == 3 and w.dim() == 2, \
        f"3D x, 2D w required; got x.dim()={x.dim()} w.dim()={w.dim()}"

    S, B, I = x.shape
    O = w.shape[0]
    T = S * B
    cfg = PERSISTENT_CONFIGS["fwd"]

    a = x.reshape(T, I)
    b = w
    y = torch.empty(T, O, dtype=torch.bfloat16, device=x.device)
    _launch(a, b, y, cfg)
    return y.view(S, B, O)

def gemm_fc1_dgrad(d_fc1_out: torch.Tensor, w: torch.Tensor,
                   *, out: torch.Tensor) -> torch.Tensor:
    """dx[T,I] = d_fc1_out[T,O] @ w[O,I]  (BF16, OVERWRITE into ``out``).

    Zero-transpose dgrad.  The kernel template supports MN-major B
    operands so dgrad can pass ``w.t()`` directly: shape ``[I, O]``
    stride ``(1, I)`` is N-major from the kernel's ``[N, K]=[I, O]``
    perspective.  Avoiding the materialised ``w.t().contiguous()`` step
    saves both the alloc and the ~85 µs copy on first call.
    """
    assert d_fc1_out.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, \
        f"BF16 inputs required; got d_fc1_out.dtype={d_fc1_out.dtype} w.dtype={w.dtype}"
    assert out.dtype == torch.bfloat16, f"BF16 out required; got {out.dtype}"

    T, O = d_fc1_out.shape
    O_w, I = w.shape
    assert O == O_w, f"O mismatch: {O} vs {O_w}"
    cfg = PERSISTENT_CONFIGS["dgrad"]

    a = d_fc1_out
    b = w.t()
    _launch(a, b, out, cfg)
    return out

def gemm_fc1_wgrad(d_fc1_out: torch.Tensor, x: torch.Tensor,
                   *, out_buf: torch.Tensor = None) -> torch.Tensor:
    """dw[O,I] += d_fc1_out^T @ x  (BF16 in, FP32 ACCUMULATE into out_buf).

    Zero-transpose wgrad.  Earlier versions of this op materialised
    ``.t().contiguous()`` views of both ``d_fc1_out`` and ``x`` to feed
    a K-major × K-major kernel template, paying ~0.2 ms / call in alloc
    + copy (≈96 MB ``d_y`` transpose + ≈32 MB ``x`` transpose at H100
    DRAM ~1.5 TB/s).  This path drops both ``.contiguous()`` calls and
    passes the stride-1-in-M / stride-1-in-N views directly to
    ``_launch``.

    The kernel template auto-detects operand majorness via
    ``LayoutEnum.from_tensor`` and the SM90 WGMMA SS atom supports both
    K-major and MN-major operands; ``make_smem_layout_a/b`` builds the
    right SMEM swizzle for whichever majorness is detected, and the
    TMA bulk-tensor descriptor handles the stride pattern transparently.
    Since the wgrad cache key (``accumulate_c=True`` + ``c_dtype=FP32``)
    is disjoint from the fwd/dgrad keys, the MN-major compiled kernel
    does not collide with the K-major × K-major fwd/dgrad cache.

    The in-kernel ``ACCUMULATE_C`` path (vectorised 64-bit FP32 gmem
    load on the epilogue's critical path → add → store) is unchanged.
    """
    assert d_fc1_out.dtype == torch.bfloat16 and x.dtype == torch.bfloat16, \
        f"BF16 inputs required; got d_fc1_out.dtype={d_fc1_out.dtype} x.dtype={x.dtype}"

    S, B, O = d_fc1_out.shape
    S_x, B_x, I = x.shape
    assert (S, B) == (S_x, B_x), f"shape mismatch: d_y={d_fc1_out.shape} x={x.shape}"
    T = S * B
    cfg = PERSISTENT_CONFIGS["wgrad"]

    d_y_2d = d_fc1_out.reshape(T, O)
    x_2d = x.reshape(T, I)

    if out_buf is None:
        out_buf = torch.zeros(O, I, dtype=torch.float32, device=d_fc1_out.device)
        accumulate = False
    else:
        assert out_buf.dtype == torch.float32, \
            f"FP32 out_buf required; got {out_buf.dtype}"
        accumulate = True

    a = d_y_2d.t()
    b = x_2d.t()

    _launch(a, b, out_buf, cfg, accumulate_c=accumulate)
    return out_buf
