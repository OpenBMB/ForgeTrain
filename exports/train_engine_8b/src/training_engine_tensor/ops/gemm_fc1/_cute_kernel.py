"""gemm_fc1 — persistent CuTeDSL SS-WGMMA GEMM device kernel.

The class below (``_Sm90GemmFC1``) is the device-side body of the
fused-SwiGLU input-projection GEMM.  It is built from public CuTeDSL
primitives (``cute``, ``cutlass.utils.hopper_helpers``,
``cutlass.pipeline``); every compute path in this file ends in
``cute.gemm`` issuing an ``m64nNk16`` WGMMA atom against the engine's
own SS-WGMMA pipeline.

Architectural levers in place
-----------------------------

* **Persistent CTA loop**: ``while cl_idx < total_cluster_tiles: ...``
  with ``grid == num_SMs`` (rounded to a cluster-aligned count).
* **Warp specialisation**: each CTA = 1 DMA producer warpgroup
  (128 threads) + N MMA consumer warpgroups (atom-layout dependent),
  with register re-partitioning via
  ``warpgroup_reg_dealloc(40)`` /  ``warpgroup_reg_alloc(<MMA_REG>)``.
* **Deep K pipeline**: ``PipelineTmaAsync`` mbarrier ring with
  ``ab_stage = 3-5`` and ``k_pipe_mmas = 1-2`` (per direction),
  keeping one or two WGMMA groups in flight across
  ``consumer_release`` / ``consumer_wait``.
* **Cluster + TMA multicast**: per-direction ``cluster_shape_mn``;
  BF16 fwd/dgrad use cluster ``(2, 1)`` with B-side multicast,
  FP32 wgrad uses cluster ``(2, 1)`` with B-side multicast.
* **Pipelined epilogue**: StMatrix8x8x16b (BF16) /
  ``CopyUniversalOp(32)`` (FP32) register-to-shared atom + epi-staged
  ``sC`` SMEM ring + TMA bulk-store (``PipelineTmaStore``); a
  sentinel-safe scalar fallback handles non-16-byte-aligned output
  slices.
* **Accumulate path** for wgrad gradient accumulation: each MMA
  thread reads its 4 ``gC[(m, n)]`` cells before the ``sC`` stage,
  casts to FP32, and folds them into the accumulator fragment, so
  the kernel is a single-launch RMW into the shared FP32 weight-grad
  buffer.

Per-direction tile config (FC1-tuned)
-------------------------------------
FC1's shapes are large (M=8192, N=16384, K=4096 for fwd; FP32 wgrad
M=16384, N=4096, K=8192).  The active tile config is:

  * BF16 fwd  (M=8192,  N=16384, K=4096):
      BM=128 BN=256 BK=64 atom=(2,1,1) ab_stages=4
  * BF16 dgrad (M=8192, N=4096,  K=16384):
      BM=128 BN=256 BK=64 atom=(2,1,1) ab_stages=4
  * FP32 wgrad (M=16384, N=4096, K=8192):
      BM=128 BN=256 BK=64 atom=(2,1,1) ab_stages=4

SMEM budget (sm_90 ≤ 228 KB):
  * BF16 fwd / dgrad : sA=64 KB sB=128 KB sC=32 KB + pipe ≈ 226 KB ✓
  * FP32 wgrad       : sA=64 KB sB=128 KB sC=32 KB + pipe ≈ 226 KB ✓

Compile cache: keyed on
``(M, N, K, a_layout, b_layout, c_layout, ab_dtype, c_dtype, accumulate)``
— each unique combination JIT-compiles once per process.  The MLIR
cache is rooted at ``$CUTE_DSL_CACHE_DIR`` (set by
``training_engine_tensor.ops._shared._cute_jit_helper``) so warm
starts are sub-second.
"""
from __future__ import annotations

import functools
from typing import Tuple

import torch

import cuda.bindings.driver as cuda  # type: ignore[import-not-found]
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

# ──────────────────────────────────────────────────────────────────────────
# Module-level constants.
# ──────────────────────────────────────────────────────────────────────────
_BK: int = 64                  # WGMMA m64nNk16 → BK = 16 × 4 = 64 for BF16.
_K_PIPE_MMAS: int = 1
_THREADS_PER_WG: int = 128
_NUM_DMA_WG: int = 1           # 1 DMA warpgroup (128 threads)
_LOAD_REG_REQUIREMENT: int = 40
_MMA_REG_REQUIREMENT: int = 232            # BF16 fwd/dgrad MMA-WG cap
_MMA_REG_REQUIREMENT_FP32: int = 128       # FP32 wgrad MMA-WG cap


def _resolve_tile_config(c_dtype_name: str):
    """[FC1-DIFF] Per-direction tile config.

    Returns ``(bm, bn, bk, atom_layout_mnk, ab_stages, raster_along_m,
    cluster_shape_mn)``.

    Active configurations:
      * BF16 fwd / dgrad: BM=128 BN=256 BK=64 atom_layout=(2,1,1)
        ab_stages=3, raster_along_m=True (M-fastest), cluster=(2,1).
      * FP32 wgrad      : BM=64  BN=256 BK=64 atom_layout=(1,1,1)
        ab_stages=3, raster_along_m=False, cluster=(2,1).
    """
    if c_dtype_name == "bfloat16":
        return 128, 256, _BK, (2, 1, 1), 3, True, (2, 1)
    if c_dtype_name == "float32":
        return 64, 256, _BK, (1, 1, 1), 3, False, (2, 1)
    raise ValueError(f"unsupported C dtype: {c_dtype_name!r}")

# ──────────────────────────────────────────────────────────────────────────
# Helpers — torch <-> cute tensor conversion + alignment for sentinel-safe
# slice buffers. Small (≤ ~20 LOC each) primitive utilities; the actual
# kernel body below is authored here.
# ──────────────────────────────────────────────────────────────────────────
def _layout_enum_from_strides(shape, strides) -> utils.LayoutEnum:
    """Resolve row/col-majorness of a 2D torch tensor from its strides.

    A 2D tensor with stride-1 on the LAST dim is ROW_MAJOR
    (k-major-A, k-major-B, n-major-C).
    A 2D tensor with stride-1 on the FIRST dim is COL_MAJOR
    (m-major-A, n-major-B, m-major-C).
    """
    assert len(shape) == 2 and len(strides) == 2
    if strides[1] == 1:
        return utils.LayoutEnum.ROW_MAJOR
    if strides[0] == 1:
        return utils.LayoutEnum.COL_MAJOR
    raise ValueError(
        f"non-contig 2D tensor (shape={shape}, strides={strides}); "
        f"wrapper must `.contiguous()` or `.t()` so one dim has stride 1"
    )

def _layout_str(role: str, layout: utils.LayoutEnum) -> str:
    """(role, LayoutEnum) → canonical label used for cache key + branch."""
    if role == "a":
        return "k_major_a" if layout == utils.LayoutEnum.ROW_MAJOR else "m_major_a"
    if role == "b":
        return "k_major_b" if layout == utils.LayoutEnum.ROW_MAJOR else "n_major_b"
    if role == "c":
        return "n_major_c" if layout == utils.LayoutEnum.ROW_MAJOR else "m_major_c"
    raise ValueError(role)

def _shape_and_strides_for_layout(d0: int, d1: int, layout_str: str):
    """``(shape, strides)`` for a ``torch.empty_strided`` placeholder
    matching the given layout label."""
    if layout_str in {"k_major_a", "k_major_b", "n_major_c"}:
        return (d0, d1), (d1, 1)
    if layout_str in {"m_major_a", "n_major_b", "m_major_c"}:
        return (d0, d1), (1, d0)
    raise ValueError(f"unknown layout_str: {layout_str!r}")

def _safe_align(t: torch.Tensor) -> int:
    """Largest power-of-two alignment (≤ 256 B) the tensor's data_ptr satisfies.

    Slice views into flat parameter-grad buffers may sit at
    element-aligned but not 16-B-aligned offsets, so we can't assume
    the 16-B default of fresh ``torch.empty``.

    The cap is extended to 256 so the IR alignment annotation matches
    the wider atoms.  H100 LDG.E.256 / STG.E.256 require
    ``align<32>`` on the global tensor; ``torch.zeros`` / fresh
    ``torch.empty`` allocations on CUDA are typically 256-B aligned,
    so production wgrad (``out_buf = fresh torch.zeros``) hits the
    top of the ladder and the kernel can emit STG.E.256.
    Misaligned slice views land on 4-B (FP32) or 2-B (BF16)
    alignment and stay on the scalar fallback path
    (``epi_vec_path == "scalar"`` via the existing ``>= 16`` gate).
    """
    addr = int(t.data_ptr())
    for align in (256, 128, 64, 32, 16, 8, 4, 2):
        if addr % align == 0 and align >= t.element_size():
            return align
    return max(1, t.element_size())

def _with_outer_l(t: torch.Tensor) -> torch.Tensor:
    """Add a trailing L=1 dim with OUTER stride (= numel of the 2D slab).

    An L stride of 1 collides with the inner K/N stride (also 1) and
    makes the TMA descriptor read garbage.  Use outer stride d0*d1.
    """
    d0, d1 = t.shape
    s0, s1 = t.stride()
    return t.as_strided((d0, d1, 1), (s0, s1, d0 * d1))

def _to_cute_tensor_3d(torch_t: torch.Tensor, dsl_dtype, layout_str: str,
                       *, align: int):
    """Wrap a 3D (last dim=1) torch tensor as a cute.Tensor.

    The ``leading_dim`` argument tells CuTeDSL which dim is contiguous
    (stride 1).  Major-mode for the WGMMA atom is derived from this.
    """
    leading_dim = (
        1 if layout_str in {"k_major_a", "k_major_b", "n_major_c"} else 0
    )
    t = from_dlpack(torch_t, assumed_align=align)
    t.element_type = dsl_dtype
    return t.mark_layout_dynamic(leading_dim=leading_dim)

_HW_SM_CACHE: int | None = None

def _hardware_sm_count() -> int:
    global _HW_SM_CACHE
    if _HW_SM_CACHE is None:
        hw = utils.HardwareInfo()
        _HW_SM_CACHE = int(hw.get_device_multiprocessor_count())
    return _HW_SM_CACHE

def _build_tma_atom_a(mA, smem_layout_single, tile_bm_bk, cluster_n: int):
    """Build the A-side TMA atom — multicast iff cluster_n > 1.

    Mirror of ``_build_tma_atom_b`` for the A-side.  At
    cluster_shape_mn=(M, N) cluster_n>1 multicasts the A tile along
    the cluster's N-mode, fanning out one HBM A fetch to cluster_N
    cluster-mate CTAs that share the same M-stripe (different
    N-stripes).  Halves A's HBM reads when cluster_N=2; quarters
    them when cluster_N=4.
    """
    if cluster_n > 1:
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(),
            mA, smem_layout_single, tile_bm_bk,
            num_multicast=cluster_n,
        )
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mA, smem_layout_single, tile_bm_bk,
    )

def _build_tma_atom_b(mB, smem_layout_single, tile_bn_bk, cluster_m: int):
    """Build the B-side TMA atom — multicast iff cluster_m > 1.

    Branch evaluates at Python compile-time (cluster_m is a Python int),
    so the @cute.jit JIT trace sees only one TMA op.  When cluster_m=2,
    the same B tile is fetched once from HBM and broadcast to the 2 CTAs
    in the cluster (M direction), halving B HBM bandwidth.
    """
    if cluster_m > 1:
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(),
            mB, smem_layout_single, tile_bn_bk,
            num_multicast=cluster_m,
        )
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mB, smem_layout_single, tile_bn_bk,
    )

# ──────────────────────────────────────────────────────────────────────────
# Persistent SS-WGMMA GEMM device kernel — gemm_fc1 specific.
# ──────────────────────────────────────────────────────────────────────────
class _Sm90GemmFC1:
    """Persistent Hopper SS-WGMMA GEMM, gemm_fc1-specific.

    Single instance per (shape, layout, dtype, accumulate) combo; cached
    via ``functools.lru_cache`` in ``_compile_for``.  Each ``__call__``
    invocation is a ``@cute.jit`` host function that:
      1. Resolves dtypes / majors from the placeholder cute tensors,
      2. Builds the SS-WGMMA tiled MMA atom + multi-stage SMEM layouts,
      3. Builds the TMA atoms for A, B (unicast at cluster=(1,1)),
      4. Defines the SharedStorage struct,
      5. Launches the device kernel on the persistent grid.

    The device kernel itself (``device_kernel`` below) implements the
    persistent CTA loop, warp-spec DMA / MMA split, K-loop with
    PipelineTmaAsync, and the scalar SMEM-staged epilogue with optional
    accumulate fold-in.
    """

    def __init__(self, *, acc_dtype, c_dtype, accumulate: bool,
                 M: int, N: int, K: int, sm_count: int,
                 ab_stages: int,
                 bm: int, bn: int, bk: int,
                 atom_layout_mnk: Tuple[int, int, int],
                 raster_along_m: bool,
                 cluster_shape_mn: Tuple[int, int],
                 epi_vec_path: str = "scalar",
                 use_stmatrix_epi: bool = False,
                 epi_stage: int = 2,
                 k_pipe_mmas: int = 1,
                 mma_reg_requirement: int = _MMA_REG_REQUIREMENT,
                 epi_vec_bits: int = 128,
                 epi_fp32_rs_bits: int = 32,
                 epi_tile_n_override: int | None = None,
                 swizzle_size: int = 1):
        self.acc_dtype = acc_dtype
        self.c_dtype_static = c_dtype
        self.accumulate = bool(accumulate)
        self.ab_stages = int(ab_stages)
        self.k_pipe_mmas = int(k_pipe_mmas)
        assert self.ab_stages >= self.k_pipe_mmas + 1, (
            f"ab_stages={ab_stages} < k_pipe_mmas+1={self.k_pipe_mmas + 1} "
            f"(the K pipeline needs at least one stage ahead of the in-flight MMAs)"
        )
        self.mma_reg_requirement = int(mma_reg_requirement)
        self.use_stmatrix_epi = bool(use_stmatrix_epi)
        self.epi_stage = int(epi_stage)
        assert epi_vec_path in {"vec", "scalar"}, (
            f"epi_vec_path={epi_vec_path!r} (must be 'vec' or 'scalar')"
        )
        self.epi_vec_path = epi_vec_path

        self.bm = int(bm)
        self.bn = int(bn)
        self.bk = int(bk)
        self.atom_layout_mnk = tuple(atom_layout_mnk)
        self.num_mma_wg = (
            self.atom_layout_mnk[0]
            * self.atom_layout_mnk[1]
            * self.atom_layout_mnk[2]
        )
        self.threads_per_cta = (
            (_NUM_DMA_WG + self.num_mma_wg) * _THREADS_PER_WG
        )
        self.mma_threads = self.num_mma_wg * _THREADS_PER_WG
        self.tile_shape_mnk = (self.bm, self.bn, self.bk)

        c_byte_size_bits = c_dtype.width        # BFloat16=16, Float32=32
        self.epi_vec_bits = int(epi_vec_bits)
        self.epi_fp32_rs_bits = int(epi_fp32_rs_bits)
        assert self.epi_fp32_rs_bits in (32, 64, 128), (
            f"epi_fp32_rs_bits={epi_fp32_rs_bits} "
            f"(must be 32, 64, or 128 — i.e. 1, 2, or 4 FP32 elements per R2S vector)"
        )
        if epi_tile_n_override is not None:
            assert isinstance(epi_tile_n_override, int)
            assert epi_tile_n_override > 0
        self.epi_tile_n_override = epi_tile_n_override
        self.epi_atom_n = self.epi_vec_bits // c_byte_size_bits   # 8 BF16 / 8 FP32(256) / 4 FP32(128)
        assert self.bn % self.epi_atom_n == 0, (
            f"BN={self.bn} not divisible by epi_atom_n={self.epi_atom_n}"
        )
        # Thread tile (M_t, N_t) = how the mma_threads are distributed over (M, N).
        # We want one warp per M-stripe (best coalescing, no warp interleaving):
        # warp covers (1, BN) per atom call (32 lanes × atom_n cells = BN
        # for BF16, BN/2 for FP32 — i.e. each warp does either 1 or 2 atom
        # calls per row).  N_t = num_lanes_per_warp = 32; M_t = mma_warps.
        # That gives:
        # BF16 BM=128 BN=256 mma_threads=256 (8 warps): thr=(8, 32) val=(16, 8)
        # FP32 BM=64  BN=256 mma_threads=128 (4 warps): thr=(4, 32) val=(16, 8)
        if self.bn // 32 >= self.epi_atom_n:
            self.epi_thr_n = 32
        elif self.bn // 16 >= self.epi_atom_n:
            self.epi_thr_n = 16
        else:
            self.epi_thr_n = 8
        self.epi_thr_m = self.mma_threads // self.epi_thr_n
        assert self.epi_thr_m * self.epi_thr_n == self.mma_threads
        self.epi_val_m = self.bm // self.epi_thr_m
        self.epi_val_n = self.bn // self.epi_thr_n
        assert self.epi_val_m * self.epi_thr_m == self.bm
        assert self.epi_val_n * self.epi_thr_n == self.bn
        assert self.epi_val_n % self.epi_atom_n == 0, (
            f"epi_val_n={self.epi_val_n} not divisible by atom_n={self.epi_atom_n} "
            f"(bn={self.bn} mma_threads={self.mma_threads} epi_thr_n={self.epi_thr_n})"
        )
        self.cluster_shape_mn = tuple(cluster_shape_mn)
        self.cluster_size = (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )
        self.raster_along_m = bool(raster_along_m)
        self.swizzle_size = int(swizzle_size)
        assert self.swizzle_size >= 1, (
            f"swizzle_size={swizzle_size} (must be >= 1)"
        )

        self.M = int(M)
        self.N = int(N)
        self.K = int(K)
        assert self.M % self.bm == 0, f"M={M} % BM={self.bm} ≠ 0"
        assert self.N % self.bn == 0, f"N={N} % BN={self.bn} ≠ 0"
        assert self.K % self.bk == 0, f"K={K} % BK={self.bk} ≠ 0"
        self.tiles_m = self.M // self.bm
        self.tiles_n = self.N // self.bn
        assert self.tiles_m % self.cluster_shape_mn[0] == 0
        assert self.tiles_n % self.cluster_shape_mn[1] == 0
        self.cluster_tiles_m = self.tiles_m // self.cluster_shape_mn[0]
        self.cluster_tiles_n = self.tiles_n // self.cluster_shape_mn[1]
        self.total_cluster_tiles = (
            self.cluster_tiles_m * self.cluster_tiles_n
        )
        if self.swizzle_size > 1:
            assert self.cluster_tiles_m % self.swizzle_size == 0, (
                f"cluster_tiles_m={self.cluster_tiles_m} not divisible by "
                f"swizzle_size={self.swizzle_size} — partial-group tail "
                f"would underflow M-axis decode"
            )

        self.sm_count = int(sm_count)
        num_clusters_in_grid = self.sm_count // self.cluster_size
        assert num_clusters_in_grid > 0
        self.num_clusters_in_grid = num_clusters_in_grid
        self.iters_per_cluster = (
            (self.total_cluster_tiles + num_clusters_in_grid - 1)
            // num_clusters_in_grid
        )

        # Filled at compile-time inside ``__call__``:
        self.a_dtype = None
        self.b_dtype = None
        self.c_dtype = None
        self.a_layout = None
        self.b_layout = None
        self.c_layout = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_tile = None
        self.epi_smem_layout_staged = None
        self.shared_storage = None
        self.tiled_mma = None

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Host entry — build TMA atoms, grid, shared storage, launch."""
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(mA)
        self.b_layout = utils.LayoutEnum.from_tensor(mB)
        self.c_layout = utils.LayoutEnum.from_tensor(mC)

        # WGMMA atom: m64nNk16 family. At atom_layout=(2,1,1) for
        # BF16 fwd/dgrad, atom_M=64 (BM=128 / 2 WGs), atom_N=256.
        # At atom_layout=(1,1,1) for FP32 wgrad, atom_M=64 (BM=64),
        # atom_N=256.
        atom_m = self.bm // self.atom_layout_mnk[0]
        atom_n = self.bn // self.atom_layout_mnk[1]
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(atom_m, atom_n),
        )

        # Hopper helpers build swizzled multi-stage SMEM layouts for us.
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout, self.tile_shape_mnk, self.a_dtype, self.ab_stages,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stages,
        )

        if cutlass.const_expr(self.use_stmatrix_epi):
            if cutlass.const_expr(self.epi_tile_n_override is not None):
                epi_tile_override = (self.bm, int(self.epi_tile_n_override))
            else:
                epi_tile_override = None
            self.epi_tile = sm90_utils.compute_tile_shape_or_override(
                self.tile_shape_mnk, self.c_dtype,
                is_cooperative=(self.num_mma_wg > 1),
                epi_tile_override=epi_tile_override,
            )
            if cutlass.const_expr(self.c_dtype.width == 32):
                _epi_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                    cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
                    self.c_dtype,
                )
                _epi_order = (
                    (1, 0, 2) if self.c_layout.is_m_major_c() else (0, 1, 2)
                )
                self.epi_smem_layout_staged = cute.tile_to_shape(
                    _epi_atom,
                    cute.append(self.epi_tile, self.epi_stage),
                    _epi_order,
                )
            else:
                self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
                    self.c_dtype, self.c_layout, self.epi_tile, self.epi_stage,
                )

        # TMA atoms. A multicast added (was always unicast pre-).
        # * A unicast at cluster_N=1, multicast at cluster_N>=2 (cluster-mates
        # along N-mode share the same M-stripe but different N-stripes).
        # * B unicast at cluster_M=1, multicast at cluster_M>=2 (cluster-mates
        # along M-mode share the same N-stripe but different M-stripes).
        # Both helpers branch at Python compile-time so the JIT trace sees
        # only one TMA op variant per JIT specialisation. At cluster=(2,2)
        # both A and B multicast → each HBM A/B fetch fans out to 2 CTAs.
        tma_atom_a, tma_tensor_a = _build_tma_atom_a(
            mA,
            cute.slice_(self.a_smem_layout_staged, (None, None, 0)),
            (self.bm, self.bk),
            cluster_n=self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = _build_tma_atom_b(
            mB,
            cute.slice_(self.b_smem_layout_staged, (None, None, 0)),
            (self.bn, self.bk),
            cluster_m=self.cluster_shape_mn[0],
        )

        if cutlass.const_expr(self.use_stmatrix_epi):
            tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
                mC,
                cute.slice_(self.epi_smem_layout_staged, (None, None, 0)),
                self.epi_tile,
            )
        else:
            tma_atom_c = tma_atom_a
            tma_tensor_c = mC

        # Persistent grid. At cluster=(1,1) grid = sm_count CTAs.  Tail
        # tiles (when total_cluster_tiles isn't a multiple of grid_x)
        # are handled by the ``while cl_idx < total_cluster_tiles``
        # check inside the device loop.
        grid_x = self.num_clusters_in_grid * self.cluster_shape_mn[0]
        grid = (grid_x, self.cluster_shape_mn[1], 1)
        cluster_shape_mnl = (
            self.cluster_shape_mn[0], self.cluster_shape_mn[1], 1,
        )

        # SharedStorage struct: pipeline mbarriers + sA + sB + sC.
        # sC sizing depends on ``use_stmatrix_epi``:
        # * False (legacy scalar / vec epilogue): single-stage
        # un-swizzled ``(BM, BN)`` row-major.  Cosize = BM × BN.
        # * True  (M5 stmatrix epilogue, +): epi-staged 3D layout
        # ``(epi_tile_M, epi_tile_N, epi_stage)`` with K_SW128
        # swizzle.  Cosize derived from
        # ``cute.cosize(epi_smem_layout_staged)`` (includes
        # swizzle padding).  For BF16 BM=128 epi_tile_N=32
        # epi_stage=2, this is 128×32×2 × 2 = 16 KB (vs 64 KB
        # unchunked).  Frees 48 KB of SMEM for future ab_stages
        # bumps.
        a_smem = self.a_smem_layout_staged
        b_smem = self.b_smem_layout_staged
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        c_dtype = self.c_dtype
        pipe_mbar_count = self.ab_stages * 2
        if cutlass.const_expr(self.use_stmatrix_epi):
            sc_cosize = cute.cosize(self.epi_smem_layout_staged)
        else:
            sc_cosize = self.bm * self.bn

        @cute.struct
        class _Shared:
            pipe_mbar: cute.struct.MemRange[cutlass.Int64, pipe_mbar_count]
            sA: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.cosize(a_smem)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.cosize(b_smem)],
                1024,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[c_dtype, sc_cosize],
                1024,
            ]

        self.shared_storage = _Shared

        if cutlass.const_expr(self.use_stmatrix_epi):
            epi_smem_layout_arg = self.epi_smem_layout_staged
        else:
            epi_smem_layout_arg = self.a_smem_layout_staged

        self.device_kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            mC,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            epi_smem_layout_arg,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=cluster_shape_mnl,
            stream=stream,
        )

    @cute.kernel
    def device_kernel(
        self,
        tma_atom_a: cute.CopyAtom, mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB: cute.Tensor,
        tma_atom_c: cute.CopyAtom, mC_tma: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """Persistent CTA loop with DMA / MMA warp-group split.

        Outer structure (CTA-uniform setup hoisted out of the tile loop):
          1. Resolve thread / warp / WG indices, prefetch TMA descriptors
          2. Allocate SMEM (sA + sB + sC) + pipeline mbarrier ring buffer
          3. Build TMA partitions (group_modes folds inner BM*BK into one
             mode so tma_partition can slice by tile)
          4. Build MMA partitions (per-thread tCrA / tCrB / acc fragment)
          5. Pipeline init arrive + wait (cluster-wide rendezvous)
          6. Per-WG register repartition (DMA: 40, MMA: 232)

        Per-tile body (inside the persistent loop):
          * Decode linear cl_idx → (pid_m, pid_n)
          * If DMA WG: issue K-loop TMA bulk copies (warp 0 issues)
          * Else (MMA WG): K-loop WGMMA + epilogue (acc → sC → gC)
            with optional gC fold-in for accumulate=True.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        # CTA-wide WG index: WG 0 = DMA, WG 1+ = MMA WGs.
        wg_idx_cta = cute.arch.make_warp_uniform(
            warp_idx // (_THREADS_PER_WG // 32)
        )
        is_dma_wg = wg_idx_cta < _NUM_DMA_WG
        tid_in_wg = tidx % _THREADS_PER_WG
        mma_wg_idx_local = wg_idx_cta - _NUM_DMA_WG

        # Warp 0 (= DMA WG's first warp) prefetches TMA descriptors —
        # one-shot cost, hides L1 cache fill latency.
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ──── SMEM allocation (CTA-wide, ONCE) ────
        smem = cutlass.utils.SmemAllocator()
        shared = smem.allocate(self.shared_storage)

        sA = shared.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner,
        )
        sB = shared.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner,
        )
        # sC: two layouts based on ``use_stmatrix_epi``:
        # * False (legacy scalar / vec epi): single-stage
        # row-major ``(BM, BN)`` un-swizzled.
        # * True  : epi-staged 3D layout
        # ``(epi_tile_M, epi_tile_N, epi_stage)`` with K_SW128
        # swizzle baked in by ``make_smem_layout_epi`` — the
        # bank-conflict-cancelling XOR pairs with stmatrix.x4's
        # 32-cell contiguous warp write pattern.
        # Hoist OUT of the tile loop's if-body to avoid CuTeDSL JIT
        # DSLTreeFlattenError (scoping issue with const-expr branches
        # inside the tile loop).
        if cutlass.const_expr(self.use_stmatrix_epi):
            sC = shared.sC.get_tensor(
                epi_smem_layout_staged.outer,
                swizzle=epi_smem_layout_staged.inner,
            )
        else:
            sC = shared.sC.get_tensor(
                cute.make_layout((self.bm, self.bn), stride=(self.bn, 1)),
            )

        # ──── Mainloop pipeline (PipelineTmaAsync mbarrier ring) ────
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tx_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Consumer arrive count. Each TMA-load via multicast can broadcast
        # to (cluster_M along M for B) + (cluster_N along N for A) − 1
        # other CTAs (the −1 because the issuing CTA is counted once). At
        # cluster=(1,1) → mcast_size=1 → arrive_cnt = num_mma_warps as in
        # the pre-M4 path. At cluster=(2,1) → mcast_size=2 → arrive_cnt =
        # 2 * num_mma_warps, because the local mbarrier needs to see the
        # extra "image" from the cluster-mate's multicast.
        num_mma_warps = self.mma_threads // 32
        num_mcast_ctas_a = self.cluster_shape_mn[1]
        num_mcast_ctas_b = self.cluster_shape_mn[0]
        mcast_size = num_mcast_ctas_a + num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * num_mma_warps
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt,
        )
        cta_layout_vmnk = cute.make_layout(
            (1, self.cluster_shape_mn[0], self.cluster_shape_mn[1], 1)
        )
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=shared.pipe_mbar.data_ptr(),
            num_stages=self.ab_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mn, is_relaxed=False,
        )

        # ──── MMA partition — per MMA-WG, per-thread fragments ────
        mma_wg_thread_layout = cute.make_layout(
            self.num_mma_wg, stride=_THREADS_PER_WG,
        )
        thr_mma = tiled_mma.get_slice(
            mma_wg_thread_layout(mma_wg_idx_local)
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        num_k_blocks = cute.size(tCrA, mode=[2])

        # ──── Multi-tile partition of GMEM tensors (CTA-uniform) ────
        # local_tile + group_modes once per CTA on the full (M,K,L) /
        # (N,K,L) / (M,N,L) tensor; per-tile we just slice into the
        # pre-partitioned tensor.
        gA_mkl = cute.local_tile(
            mA, self.tile_shape_mnk, (None, None, None, 0),
            proj=(1, None, 1),
        )
        gB_nkl = cute.local_tile(
            mB, self.tile_shape_mnk, (None, None, None, 0),
            proj=(None, 1, 1),
        )
        gC_mnl = cute.local_tile(
            mC, self.tile_shape_mnk, (None, None, None, 0),
            proj=(1, 1, None),
        )
        if cutlass.const_expr(self.use_stmatrix_epi):
            gC_mnl_tma = cute.local_tile(
                mC_tma, self.tile_shape_mnk, (None, None, None, 0),
                proj=(1, 1, None),
            )
        k_tile_count = cute.size(gA_mkl, mode=[3])

        tCgC_mnl = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC_mnl.shape[:3]
        acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # CTA coords within cluster (for TMA partitioning + per-cluster
        # tile offset). At cluster=(1,1) both are 0.
        m_in_cluster, n_in_cluster, _ = cute.arch.block_in_cluster_idx()

        sA_group = cute.group_modes(sA, 0, 2)
        sB_group = cute.group_modes(sB, 0, 2)
        gA_grouped = cute.group_modes(gA_mkl, 0, 2)
        gB_grouped = cute.group_modes(gB_nkl, 0, 2)
        a_cta_layout = cute.make_layout((self.cluster_shape_mn[1],))
        b_cta_layout = cute.make_layout((self.cluster_shape_mn[0],))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, n_in_cluster, a_cta_layout, sA_group, gA_grouped,
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, m_in_cluster, b_cta_layout, sB_group, gB_grouped,
        )

        # M4 multicast masks: B along cluster_M axis (mcast_mode=1
        # = M-mode of cluster), A along cluster_N axis (mcast_mode=2
        # = N-mode of cluster). Computed unconditionally (a couple
        # of bitops); only used in the producer-side const_expr
        # branches when the corresponding cluster dim > 1.
        cta_coord_vmnk = (0, m_in_cluster, n_in_cluster, 0)
        b_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_coord_vmnk, mcast_mode=1,
        )
        a_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_coord_vmnk, mcast_mode=2,
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ──── Per-thread epilogue (m, n) coords inside the MMA tile ────
        # Per Hopper PTX 8.0 §9.7.13.4, for wgmma m64nNk16 each warp covers
        # a 16-row M-stripe of the 64-row tile; lanes split into (group,
        # tig) inside the stripe.
        #
        # warp w (0..3) of an MMA WG: covers M rows [16w, 16w+16)
        # lane l (0..31): group=l//4 (offset 0..8), tig=l%4 (offset 0..4)
        # For each n_seg in [0, atom_N/8):
        # acc[n_seg*4 + 0] ↔ (16w + group,     n_seg*8 + tig*2    )
        # acc[n_seg*4 + 1] ↔ (16w + group,     n_seg*8 + tig*2 + 1)
        # acc[n_seg*4 + 2] ↔ (16w + group + 8, n_seg*8 + tig*2    )
        # acc[n_seg*4 + 3] ↔ (16w + group + 8, n_seg*8 + tig*2 + 1)
        #
        # atom_layout encodes how WGs split the tile :
        # * (2,1,1) — M-split.  E.g. BF16 fwd/dgrad: 2 MMA WGs cover
        # M rows [0, 64) and [64, 128).  ``m_offset_wg`` shifts
        # m_lo by ``BM / M_wgs``; ``n_seg_per_wg = BN / 8`` (full
        # BN per WG).
        # * (1,2,1) — N-split.  E.g. FP32 wgrad: 2 MMA WGs cover
        # N cols [0, 128) and [128, 256).  ``n_offset_wg`` shifts
        # n_lo by ``BN / N_wgs``; ``n_seg_per_wg = BN / N_wgs / 8``
        # (atom_N/8 segments per WG).
        # * (1,1,1) — no split.  Single WG covers the whole tile.
        warp_id_local = tid_in_wg // 32
        lane = tid_in_wg % 32
        group = lane // 4
        tig = lane % 4
        if cutlass.const_expr(self.atom_layout_mnk[0] > 1):
            # M-split path (BF16 fwd/dgrad and accumulate=True wgrad).
            m_offset_wg = mma_wg_idx_local * (
                self.bm // self.atom_layout_mnk[0]
            )
            n_offset_wg = 0
            n_seg_per_wg = self.bn // 8
        elif cutlass.const_expr(self.atom_layout_mnk[1] > 1):
            # N-split path. Each WG's atom covers
            # ``atom_n = BN / N_wgs`` cells along N.
            m_offset_wg = 0
            n_offset_wg = mma_wg_idx_local * (
                self.bn // self.atom_layout_mnk[1]
            )
            n_seg_per_wg = (self.bn // self.atom_layout_mnk[1]) // 8
        else:
            # No-split (single MMA WG, e.g. atom=(1,1,1)).
            m_offset_wg = 0
            n_offset_wg = 0
            n_seg_per_wg = self.bn // 8

        # MMA-only NamedBarrier for the epilogue sync (DMA WG skips it).
        nbar_epi = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.mma_threads,
        )

        # ──── Pipeline state machines — initialised ONCE per CTA ────
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stages,
        )
        read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stages,
        )
        # M3 uses a separate release_state lagging read_state by
        # k_pipe_mmas. At M2 (k_pipe_mmas=0), they're identical.
        release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stages,
        )

        cluster_idx_x, _, _ = cute.arch.cluster_idx()
        iters_per_cluster = self.iters_per_cluster
        cluster_tiles_m = self.cluster_tiles_m
        cluster_tiles_n = self.cluster_tiles_n
        total_cluster_tiles = self.total_cluster_tiles
        num_clusters_in_grid = self.num_clusters_in_grid

        # ──── Per-WG register repartition ────
        if is_dma_wg:
            cute.arch.warpgroup_reg_dealloc(_LOAD_REG_REQUIREMENT)
        else:
            cute.arch.warpgroup_reg_alloc(self.mma_reg_requirement)

        if is_dma_wg:
            # ═══════════ DMA Producer Warp Group ═══════════
            # Only warp 0 issues TMA bulk copies (single producer
            # arrives per acquire/commit pair).
            if warp_idx == 0:
                for tile_it in cutlass.range(iters_per_cluster, unroll=1):
                    cl_idx = (
                        cluster_idx_x + tile_it * num_clusters_in_grid
                    )
                    if cl_idx < total_cluster_tiles:
                        # Linear → (pid_m, pid_n) decode. Three modes:
                        # * swizzle (``swizzle_size > 1``): M4
                        # super-block N-fastest raster — group every
                        # ``sw`` cluster_pid_m rows together, walk
                        # N-fastest within group, advance group.
                        # Tighter M-axis compaction at the cost of
                        # looser N-axis compaction.  See __init__'s
                        # swizzle_size docstring for L2 trade-off.
                        # * raster_along_m=True (M-fastest): walks M
                        # then N — best A-side L2 reuse across
                        # concurrent CTAs at high cluster_tiles_n.
                        # * raster_along_m=False (N-fastest, default
                        # when sw=1): walks N then M.
                        if cutlass.const_expr(self.swizzle_size > 1):
                            sw = self.swizzle_size
                            tiles_per_group = sw * cluster_tiles_n
                            group_idx = cl_idx // tiles_per_group
                            in_group = cl_idx % tiles_per_group
                            cluster_pid_m = (
                                group_idx * sw + (in_group // cluster_tiles_n)
                            )
                            cluster_pid_n = in_group % cluster_tiles_n
                        elif cutlass.const_expr(self.raster_along_m):
                            cluster_pid_m = cl_idx % cluster_tiles_m
                            cluster_pid_n = cl_idx // cluster_tiles_m
                        else:
                            cluster_pid_m = cl_idx // cluster_tiles_n
                            cluster_pid_n = cl_idx % cluster_tiles_n
                        pid_m = (
                            cluster_pid_m * self.cluster_shape_mn[0]
                            + m_in_cluster
                        )
                        pid_n = (
                            cluster_pid_n * self.cluster_shape_mn[1]
                            + n_in_cluster
                        )
                        tAgA_tile = tAgA[(None, pid_m, None)]
                        tBgB_tile = tBgB[(None, pid_n, None)]
                        for k_idx in cutlass.range(k_tile_count, unroll=1):
                            pipe.producer_acquire(producer_state)
                            # A-side TMA : multicast when cluster_N
                            # > 1 so one HBM A fetch fans out to
                            # ``cluster_N`` cluster-mate CTAs.
                            if cutlass.const_expr(
                                self.cluster_shape_mn[1] > 1
                            ):
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_tile[(None, k_idx)],
                                    tAsA[(None, producer_state.index)],
                                    tma_bar_ptr=pipe.producer_get_barrier(
                                        producer_state,
                                    ),
                                    mcast_mask=a_mcast_mask,
                                )
                            else:
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_tile[(None, k_idx)],
                                    tAsA[(None, producer_state.index)],
                                    tma_bar_ptr=pipe.producer_get_barrier(
                                        producer_state,
                                    ),
                                )
                            # B-side TMA: multicast when cluster_M > 1 so
                            # one HBM fetch fans out to ``cluster_M`` CTAs.
                            # The const_expr branch evaluates at JIT
                            # specialisation time, so the trace sees only
                            # one cute.copy variant per JIT.
                            if cutlass.const_expr(
                                self.cluster_shape_mn[0] > 1
                            ):
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_tile[(None, k_idx)],
                                    tBsB[(None, producer_state.index)],
                                    tma_bar_ptr=pipe.producer_get_barrier(
                                        producer_state,
                                    ),
                                    mcast_mask=b_mcast_mask,
                                )
                            else:
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_tile[(None, k_idx)],
                                    tBsB[(None, producer_state.index)],
                                    tma_bar_ptr=pipe.producer_get_barrier(
                                        producer_state,
                                    ),
                                )
                            pipe.producer_commit(producer_state)
                            producer_state.advance()
        else:
            # ═══════════ MMA Consumer Warp Group ═══════════
            # ────────────── M5 CTA-uniform setup ──────────────
            # Build r2s (StMatrix) + S2G (TMA bulk store) atoms +
            # TMA store pipeline state machine. Only the MMA-WG
            # threads reach this code; partitions are per-MMA-thread
            # (offset by ``_NUM_DMA_WG * _THREADS_PER_WG`` = 128).
            #
            # All construction (CopyAtom / TiledCopy /
            # make_rmem_tensor / PipelineTmaStore.create) is JIT-time
            # — they produce compile-time layouts and runtime sync
            # objects that the per-tile epilogue loop reuses without
            # re-building.
            if cutlass.const_expr(self.use_stmatrix_epi):
                if cutlass.const_expr(self.c_dtype.width == 16):
                    copy_atom_r2s = sm90_utils.get_smem_store_op(
                        self.c_layout,
                        elem_ty_d=self.c_dtype,
                        elem_ty_acc=self.acc_dtype,
                    )
                    copy_atom_C = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(
                            self.c_layout.is_m_major_c(), 4,
                        ),
                        self.c_dtype,
                    )
                else:
                    # FP32 — CopyUniversalOp(32) = STS.32 = 1 FP32
                    # cell along N (was STS.64 ).
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                        num_bits_per_copy=self.epi_fp32_rs_bits,
                    )
                    copy_atom_C = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                        num_bits_per_copy=self.epi_fp32_rs_bits,
                    )
                tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
                    copy_atom_C, tiled_mma,
                )
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_C_atom,
                )
                # MMA-WG-local thread idx (0..mma_threads-1). Both
                # MMA WGs (for fc1's atom_layout=(2,1,1)) share a
                # 0..256 range here; the tiled_copy_C_atom's MMA C
                # partition handles the per-WG M-stripe offset
                # internally via the WGMMA acc fragment layout.
                mma_local_tidx = tidx - _NUM_DMA_WG * _THREADS_PER_WG
                thr_copy_r2s = tiled_copy_r2s.get_slice(mma_local_tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(acc)
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, self.acc_dtype,
                )
                tRS_rD_out = cute.make_rmem_tensor(
                    tRS_rD_layout.shape, self.c_dtype,
                )
                size_tRS_rD = cute.size(tRS_rD)
                # TMA store pipeline (one MMA WG group is the producer).
                # Only the dedicated "store warp" calls
                # producer_acquire/commit — remaining MMA threads
                # sync via the nbar_epi NamedBarrier between phases.
                # The CooperativeGroup arrive count matches
                # mma_threads so PipelineTmaStore's sync object size
                # matches the broadcast pattern.
                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.mma_threads,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )
                # Designated store warp = first warp of the first MMA
                # WG. warp_idx ∈ {4..(4+mma_threads/32-1)} since DMA
                # WG occupies warps 0..3. Pick warp 4 (= first MMA
                # warp).
                epi_store_warp_id = _NUM_DMA_WG * 4

            for tile_it in cutlass.range(iters_per_cluster, unroll=1):
                cl_idx = cluster_idx_x + tile_it * num_clusters_in_grid
                if cl_idx < total_cluster_tiles:
                    if cutlass.const_expr(self.swizzle_size > 1):
                        sw = self.swizzle_size
                        tiles_per_group = sw * cluster_tiles_n
                        group_idx = cl_idx // tiles_per_group
                        in_group = cl_idx % tiles_per_group
                        cluster_pid_m = (
                            group_idx * sw + (in_group // cluster_tiles_n)
                        )
                        cluster_pid_n = in_group % cluster_tiles_n
                    elif cutlass.const_expr(self.raster_along_m):
                        cluster_pid_m = cl_idx % cluster_tiles_m
                        cluster_pid_n = cl_idx // cluster_tiles_m
                    else:
                        cluster_pid_m = cl_idx // cluster_tiles_n
                        cluster_pid_n = cl_idx % cluster_tiles_n
                    pid_m = (
                        cluster_pid_m * self.cluster_shape_mn[0]
                        + m_in_cluster
                    )
                    pid_n = (
                        cluster_pid_n * self.cluster_shape_mn[1]
                        + n_in_cluster
                    )
                    gC = gC_mnl[(None, None, pid_m, pid_n)]

                    # ACCUMULATE=False on the FIRST WGMMA of each tile so
                    # the previous tile's residual ``acc`` is overwritten.
                    # Flipped to True after the first cute.gemm call.
                    tiled_mma.set(
                        cute.nvgpu.warpgroup.Field.ACCUMULATE, False,
                    )

                    # ──── M3 K-loop: deep pipeline with k_pipe_mmas in-flight ────
                    # earlier tuning: ``wait_group(k_pipe_mmas)`` keeps that many
                    # WGMMA groups in-flight at all times, giving the
                    # consumer slack to hide the WGMMA tail latency behind
                    # the next K-tile's consumer_wait + fence.
                    #
                    # Structure:
                    # * Prologue (k_pipe_mmas iters): issue WGMMA, NO
                    # wait, NO release.  Grows in-flight count from 0
                    # to k_pipe_mmas.
                    # * Mainloop (k_tile_count - k_pipe_mmas iters):
                    # issue WGMMA + wait_group(k_pipe_mmas) +
                    # release.  Keeps in-flight count at k_pipe_mmas.
                    # * Drain: wait_group(0) + k_pipe_mmas extra
                    # consumer_release calls to free the remaining
                    # prefetched stages.
                    #
                    # Constraint: ab_stages >= k_pipe_mmas + 1; enforced
                    # at __init__. bumps to ab_stages=4 +
                    # k_pipe_mmas=2 for M5 path (BF16 fwd/dgrad).

                    # Prologue: k_pipe_mmas iters (compile-time-unrolled).
                    for k_idx in cutlass.range_constexpr(self.k_pipe_mmas):
                        pipe.consumer_wait(read_state)
                        cute.nvgpu.warpgroup.fence()
                        for kb in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                acc,
                                tCrA[(None, None, kb, read_state.index)],
                                tCrB[(None, None, kb, read_state.index)],
                                acc,
                            )
                            tiled_mma.set(
                                cute.nvgpu.warpgroup.Field.ACCUMULATE, True,
                            )
                        cute.nvgpu.warpgroup.commit_group()
                        read_state.advance()

                    # Mainloop: k_tile_count - k_pipe_mmas iters.
                    for k_idx in cutlass.range(
                        k_tile_count - self.k_pipe_mmas, unroll=1,
                    ):
                        pipe.consumer_wait(read_state)
                        cute.nvgpu.warpgroup.fence()
                        for kb in cutlass.range_constexpr(num_k_blocks):
                            cute.gemm(
                                tiled_mma,
                                acc,
                                tCrA[(None, None, kb, read_state.index)],
                                tCrB[(None, None, kb, read_state.index)],
                                acc,
                            )
                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(self.k_pipe_mmas)
                        pipe.consumer_release(release_state)
                        read_state.advance()
                        release_state.advance()

                    # Drain: wait for the remaining k_pipe_mmas WGMMA
                    # groups + release their stages.
                    cute.nvgpu.warpgroup.wait_group(0)
                    for _drain in cutlass.range_constexpr(self.k_pipe_mmas):
                        pipe.consumer_release(release_state)
                        release_state.advance()

                    # ════════════════════════════════════════════════
                    if cutlass.const_expr(self.use_stmatrix_epi):
                        # ────────────────────────────────────────────
                        if cutlass.const_expr(self.accumulate):
                            for n_seg in cutlass.range_constexpr(
                                n_seg_per_wg,
                            ):
                                base = n_seg * 4
                                n_lo = (
                                    n_offset_wg
                                    + n_seg * 8
                                    + tig * 2
                                )
                                m_lo = (
                                    m_offset_wg
                                    + warp_id_local * 16
                                    + group
                                )
                                acc[base + 0] = (
                                    acc[base + 0]
                                    + self.acc_dtype(
                                        gC[(m_lo,     n_lo    )]
                                    )
                                )
                                acc[base + 1] = (
                                    acc[base + 1]
                                    + self.acc_dtype(
                                        gC[(m_lo,     n_lo + 1)]
                                    )
                                )
                                acc[base + 2] = (
                                    acc[base + 2]
                                    + self.acc_dtype(
                                        gC[(m_lo + 8, n_lo    )]
                                    )
                                )
                                acc[base + 3] = (
                                    acc[base + 3]
                                    + self.acc_dtype(
                                        gC[(m_lo + 8, n_lo + 1)]
                                    )
                                )

                        # Per-tile partition of gC (via ``mC_tma``) into
                        # epi-sized sub-blocks. ``zipped_divide``
                        # gives mode-0 = epi_tile shape (BM, 32) and
                        # mode-1 = sub-block coords (BM/BM, BN/32) =
                        # (1, 8) for fc1's BF16 fwd/dgrad tile.
                        # ``tma_partition`` converts mode-0 (per-sub-
                        # block elements) into a TMA-friendly grouped
                        # mode 0, matching the sC side's
                        # ``cute.group_modes(sC, 0, 2)``.
                        gC_tma = gC_mnl_tma[(None, None, pid_m, pid_n)]
                        tCgC_for_tma = cute.zipped_divide(
                            gC_tma, self.epi_tile,
                        )
                        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_c, 0, cute.make_layout(1),
                            cute.group_modes(sC, 0, 2),
                            tCgC_for_tma,
                        )
                        epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
                        epi_tile_shape = tCgC_for_tma.shape[1]
                        epi_tile_layout = cute.make_layout(
                            epi_tile_shape,
                            stride=(epi_tile_shape[1], 1),
                        )
                        # Running counter of epi-tiles processed by
                        # this CTA so far — feeds the
                        # ``num_prev_epi_tiles + epi_idx`` ring index
                        # into ``self.epi_stage``. ``tile_it *
                        # epi_tile_num`` is consistent with the TMA
                        # store pipeline's internal commit / acquire
                        # count because the per-tile epi loop is
                        # contiguous (no interleaving across tiles).
                        num_prev_epi_tiles = tile_it * epi_tile_num
                        for epi_idx in cutlass.range_constexpr(epi_tile_num):
                            # acc → tRS_rD : slice this epi-sub-block's
                            # cells out of the full WGMMA accumulator
                            # fragment. ``tRS_rAcc`` is the retiled
                            # acc with shape ``(atom_size, M_div,
                            # N_div * epi_tile_num)``;
                            # ``size_tRS_rD = atom_size * M_div *
                            # N_div`` is the per-sub-block cell count.
                            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                                tRS_rD[epi_v] = tRS_rAcc[
                                    epi_idx * size_tRS_rD + epi_v
                                ]
                            # FP32 → c_dtype (BF16) cast in register.
                            acc_vec = tRS_rD.load()
                            tRS_rD_out.store(acc_vec.to(self.c_dtype))
                            # rD → sC via stmatrix.x4 (8x8x16b atom).
                            # 32 lanes per warp collectively write a
                            # contiguous 32-cell strip per atom call —
                            # the K_SW128 swizzle baked into sC's
                            # layout exactly cancels the bank-collision
                            # XOR so the warp's stores hit 32 distinct
                            # SMEM banks.
                            epi_buffer = (
                                (num_prev_epi_tiles + epi_idx)
                                % self.epi_stage
                            )
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD_out,
                                tRS_sD[(None, None, None, epi_buffer)],
                            )
                            cute.arch.fence_proxy(
                                "async.shared", space="cta",
                            )
                            nbar_epi.arrive_and_wait()

                            # sC → gC via TMA bulk store (single warp
                            # issues; other MMA threads wait at nbar).
                            if warp_idx == epi_store_warp_id:
                                gmem_coord = (
                                    epi_tile_layout.get_hier_coord(
                                        epi_idx,
                                    )
                                )
                                cute.copy(
                                    tma_atom_c,
                                    bSG_sD[(None, epi_buffer)],
                                    bSG_gD[(None, gmem_coord)],
                                )
                                tma_store_pipeline.producer_commit()
                                tma_store_pipeline.producer_acquire()

                            # B3-equivalent: serialise sC slot reuse
                            # across epi-iters. Without this, the
                            # NEXT epi_idx's r2s store to this
                            # epi_buffer slot could race the TMA bulk
                            # store's read.
                            nbar_epi.arrive_and_wait()
                    else:
                        # ── Legacy scalar / vec path (kept for ──
                        # FP32 wgrad + BF16 accumulate=True +
                        # non-16-B-aligned slice writes).
                        # ──── Optional accumulate=True fold-in (wgrad GAS) ────
                        # Read old gC cells, cast to FP32, add to acc.
                        # The subsequent acc → sC stage writes the
                        # sum back. No cross-thread hazard: each
                        # thread reads only its own 4 (m, n) cells
                        # per n_seg.
                        #
                        if cutlass.const_expr(self.accumulate):
                            mma_tidx_full = (
                                tidx - _NUM_DMA_WG * _THREADS_PER_WG
                            )
                            if cutlass.const_expr(
                                self.epi_vec_path == "vec"
                            ):
                                copy_atom_g2s = cute.make_copy_atom(
                                    cute.nvgpu.CopyUniversalOp(),
                                    self.c_dtype,
                                    num_bits_per_copy=self.epi_vec_bits,
                                )
                                thr_layout_g2s = cute.make_layout(
                                    (self.epi_thr_m, self.epi_thr_n),
                                    stride=(self.epi_thr_n, 1),
                                )
                                val_layout_g2s = cute.make_layout(
                                    (self.epi_val_m, self.epi_val_n),
                                    stride=(self.epi_val_n, 1),
                                )
                                tiled_copy_g2s = (
                                    cute.make_tiled_copy_tv(
                                        copy_atom_g2s,
                                        thr_layout_g2s,
                                        val_layout_g2s,
                                    )
                                )
                                thr_copy_g2s = (
                                    tiled_copy_g2s.get_slice(
                                        mma_tidx_full,
                                    )
                                )
                                gC_par_g2s = (
                                    thr_copy_g2s.partition_S(gC)
                                )
                                sC_par_g2s = (
                                    thr_copy_g2s.partition_D(sC)
                                )
                                cute.copy(
                                    copy_atom_g2s,
                                    gC_par_g2s,
                                    sC_par_g2s,
                                )
                                # nbar to make the cooperative
                                # cute.copy's sC writes visible to
                                # the per-thread sC reads below
                                # (cross-thread visibility — each
                                # MMA thread reads cells written by
                                # different cooperative threads).
                                nbar_epi.arrive_and_wait()
                                for n_seg in cutlass.range_constexpr(
                                    n_seg_per_wg
                                ):
                                    base = n_seg * 4
                                    n_lo = (
                                        n_offset_wg
                                        + n_seg * 8
                                        + tig * 2
                                    )
                                    m_lo = (
                                        m_offset_wg
                                        + warp_id_local * 16
                                        + group
                                    )
                                    acc[base + 0] = (
                                        acc[base + 0]
                                        + self.acc_dtype(
                                            sC[(m_lo,     n_lo    )]
                                        )
                                    )
                                    acc[base + 1] = (
                                        acc[base + 1]
                                        + self.acc_dtype(
                                            sC[(m_lo,     n_lo + 1)]
                                        )
                                    )
                                    acc[base + 2] = (
                                        acc[base + 2]
                                        + self.acc_dtype(
                                            sC[(m_lo + 8, n_lo    )]
                                        )
                                    )
                                    acc[base + 3] = (
                                        acc[base + 3]
                                        + self.acc_dtype(
                                            sC[(m_lo + 8, n_lo + 1)]
                                        )
                                    )
                            else:
                                # Scalar fallback for non-16-B-aligned
                                # gC slices.  Keeps the per-cell gC
                                # reads, which are slow but provably
                                # safe against any slice alignment.
                                for n_seg in cutlass.range_constexpr(
                                    n_seg_per_wg
                                ):
                                    base = n_seg * 4
                                    n_lo = (
                                        n_offset_wg
                                        + n_seg * 8
                                        + tig * 2
                                    )
                                    m_lo = (
                                        m_offset_wg
                                        + warp_id_local * 16
                                        + group
                                    )
                                    acc[base + 0] = (
                                        acc[base + 0]
                                        + self.acc_dtype(
                                            gC[(m_lo,     n_lo    )]
                                        )
                                    )
                                    acc[base + 1] = (
                                        acc[base + 1]
                                        + self.acc_dtype(
                                            gC[(m_lo,     n_lo + 1)]
                                        )
                                    )
                                    acc[base + 2] = (
                                        acc[base + 2]
                                        + self.acc_dtype(
                                            gC[(m_lo + 8, n_lo    )]
                                        )
                                    )
                                    acc[base + 3] = (
                                        acc[base + 3]
                                        + self.acc_dtype(
                                            gC[(m_lo + 8, n_lo + 1)]
                                        )
                                    )

                        # ──── acc → sC (FP32 → c_dtype cast) ────
                        for n_seg in cutlass.range_constexpr(n_seg_per_wg):
                            base = n_seg * 4
                            n_lo = n_offset_wg + n_seg * 8 + tig * 2
                            m_lo = m_offset_wg + warp_id_local * 16 + group
                            sC[(m_lo,     n_lo    )] = self.c_dtype(acc[base + 0])
                            sC[(m_lo,     n_lo + 1)] = self.c_dtype(acc[base + 1])
                            sC[(m_lo + 8, n_lo    )] = self.c_dtype(acc[base + 2])
                            sC[(m_lo + 8, n_lo + 1)] = self.c_dtype(acc[base + 3])

                        # MMA-only B2: all sC writes done before SMEM→GMEM read.
                        nbar_epi.arrive_and_wait()

                        # ──── Cooperative sC → gC ────
                        mma_tidx_full = tidx - _NUM_DMA_WG * _THREADS_PER_WG
                        if cutlass.const_expr(self.epi_vec_path == "vec"):
                            # ── vec path: cute.copy + CopyUniversalOp(128) ──
                            copy_atom_s2g = cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                self.c_dtype,
                                num_bits_per_copy=self.epi_vec_bits,
                            )
                            thr_layout = cute.make_layout(
                                (self.epi_thr_m, self.epi_thr_n),
                                stride=(self.epi_thr_n, 1),
                            )
                            val_layout = cute.make_layout(
                                (self.epi_val_m, self.epi_val_n),
                                stride=(self.epi_val_n, 1),
                            )
                            tiled_copy_s2g = cute.make_tiled_copy_tv(
                                copy_atom_s2g, thr_layout, val_layout,
                            )
                            thr_copy_s2g = tiled_copy_s2g.get_slice(mma_tidx_full)
                            sC_par = thr_copy_s2g.partition_S(sC)
                            gC_par = thr_copy_s2g.partition_D(gC)
                            cute.copy(copy_atom_s2g, sC_par, gC_par)
                        else:
                            # ── Scalar fallback (sentinel-safe) ──────────
                            total_elems = self.bm * self.bn
                            elems_per_thread = total_elems // self.mma_threads
                            for it in cutlass.range_constexpr(elems_per_thread):
                                flat = it * self.mma_threads + mma_tidx_full
                                m = flat // self.bn
                                n = flat % self.bn
                                gC[(m, n)] = sC[(m, n)]

                        # MMA-only B3: serialize sC re-use across tiles.
                        nbar_epi.arrive_and_wait()

            # ──────────── M5 — drain TMA store pipeline ────────────
            # After all tiles processed, wait for any still-in-flight
            # TMA bulk stores to complete before the CTA exits. Only
            # the designated store warp calls producer_tail (the
            # remaining MMA threads have completed their nbar_epi
            # sync at the end of the final tile's epilogue, no
            # further work).
            if cutlass.const_expr(self.use_stmatrix_epi):
                if warp_idx == epi_store_warp_id:
                    tma_store_pipeline.producer_tail()

# ──────────────────────────────────────────────────────────────────────────
# Host-side compile cache + dispatch.
# ──────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _compile_for(
    M: int, N: int, K: int,
    a_layout_str: str, b_layout_str: str, c_layout_str: str,
    ab_dtype_name: str, c_dtype_name: str,
    accumulate: bool,
    ab_stages: int,
    bm: int, bn: int, bk: int, atom_layout_mnk: tuple,
    raster_along_m: bool,
    cluster_shape_mn: tuple,
    epi_vec_path: str,
    use_stmatrix_epi: bool,
    epi_stage: int,
    k_pipe_mmas: int,
    mma_reg_requirement: int = _MMA_REG_REQUIREMENT,
    epi_vec_bits: int = 128,
    epi_fp32_rs_bits: int = 32,
    epi_tile_n_override: int | None = None,
    swizzle_size: int = 1,
):
    """Compile + cache a ``_Sm90GemmFC1`` for this (shape, layout, dtype) combo.

    ``epi_vec_path`` is part of the LRU key so the aligned 128-bit-store
    specialisation and the sentinel-safe scalar specialisation each compile
    once — see ``_Sm90GemmFC1.__init__`` for the per-path semantics.

    ``use_stmatrix_epi`` + ``epi_stage`` added to the cache key.
    The pipelined ``stmatrix.x4`` + TMA bulk-store epilogue compiles
    once for the BF16 fwd/dgrad aligned fast path; the legacy scalar /
    vec epilogue compiles once for FP32 wgrad + BF16 accumulate=True +
    any misaligned-slice fall-back paths.
    """
    ab_dtype_map = {
        "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32,
    }
    c_dtype_map = {
        "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32,
    }
    torch_ab_dtype_map = {
        "bfloat16": torch.bfloat16, "float32": torch.float32,
    }
    torch_c_dtype_map = {
        "bfloat16": torch.bfloat16, "float32": torch.float32,
    }
    ab_dtype = ab_dtype_map[ab_dtype_name]
    c_dtype = c_dtype_map[c_dtype_name]
    acc_dtype = cutlass.Float32

    a_torch_shape, a_torch_strides = (
        _shape_and_strides_for_layout(M, K, a_layout_str)
    )
    b_torch_shape, b_torch_strides = (
        _shape_and_strides_for_layout(N, K, b_layout_str)
    )
    c_torch_shape, c_torch_strides = (
        _shape_and_strides_for_layout(M, N, c_layout_str)
    )
    torch_ab_dtype = torch_ab_dtype_map[ab_dtype_name]
    torch_c_dtype = torch_c_dtype_map[c_dtype_name]

    a_placeholder = _with_outer_l(torch.empty_strided(
        a_torch_shape, a_torch_strides,
        dtype=torch_ab_dtype, device="cuda",
    ))
    b_placeholder = _with_outer_l(torch.empty_strided(
        b_torch_shape, b_torch_strides,
        dtype=torch_ab_dtype, device="cuda",
    ))
    c_placeholder = _with_outer_l(torch.empty_strided(
        c_torch_shape, c_torch_strides,
        dtype=torch_c_dtype, device="cuda",
    ))

    c_compile_align = max(16, epi_vec_bits // 8)
    mA = _to_cute_tensor_3d(a_placeholder, ab_dtype, a_layout_str, align=16)
    mB = _to_cute_tensor_3d(b_placeholder, ab_dtype, b_layout_str, align=16)
    mC = _to_cute_tensor_3d(c_placeholder, c_dtype, c_layout_str, align=c_compile_align)

    sm_count = _hardware_sm_count()
    gemm = _Sm90GemmFC1(
        acc_dtype=acc_dtype, c_dtype=c_dtype, accumulate=accumulate,
        M=M, N=N, K=K, sm_count=sm_count,
        ab_stages=ab_stages,
        bm=bm, bn=bn, bk=bk,
        atom_layout_mnk=atom_layout_mnk,
        raster_along_m=raster_along_m,
        cluster_shape_mn=cluster_shape_mn,
        epi_vec_path=epi_vec_path,
        use_stmatrix_epi=use_stmatrix_epi,
        epi_stage=epi_stage,
        k_pipe_mmas=k_pipe_mmas,
        mma_reg_requirement=mma_reg_requirement,
        epi_vec_bits=epi_vec_bits,
        epi_fp32_rs_bits=epi_fp32_rs_bits,
        epi_tile_n_override=epi_tile_n_override,
        swizzle_size=swizzle_size,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(gemm, mA, mB, mC, stream)

    def run(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        cute_a = _to_cute_tensor_3d(
            _with_outer_l(a), ab_dtype, a_layout_str,
            align=_safe_align(a),
        )
        cute_b = _to_cute_tensor_3d(
            _with_outer_l(b), ab_dtype, b_layout_str,
            align=_safe_align(b),
        )
        cute_c = _to_cute_tensor_3d(
            _with_outer_l(c), c_dtype, c_layout_str,
            align=_safe_align(c),
        )
        s = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(cute_a, cute_b, cute_c, s)

    return run

def run_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    accumulate: bool = False,
) -> torch.Tensor:
    """Public entry — writes ``c = a @ b^T`` (or ``c += a @ b^T``).

    Inputs are 2D torch tensors:
      - a: (M, K) bf16, k-major-A (row) or m-major-A (col)
      - b: (N, K) bf16, k-major-B (row) or n-major-B (col)
      - c: (M, N) bf16 or fp32, n-major-C (row) or m-major-C (col)
    """
    assert a.is_cuda and b.is_cuda and c.is_cuda
    assert a.dim() == 2 and b.dim() == 2 and c.dim() == 2

    M, Ka = a.shape
    N, Kb = b.shape
    M2, N2 = c.shape
    assert Ka == Kb, f"K mismatch: a.K={Ka} vs b.K={Kb}"
    assert M == M2 and N == N2, (
        f"C shape mismatch: ({M},{N}) vs ({M2},{N2})"
    )

    a_layout = _layout_enum_from_strides(tuple(a.shape), tuple(a.stride()))
    b_layout = _layout_enum_from_strides(tuple(b.shape), tuple(b.stride()))
    c_layout = _layout_enum_from_strides(tuple(c.shape), tuple(c.stride()))

    a_layout_str = _layout_str("a", a_layout)
    b_layout_str = _layout_str("b", b_layout)
    c_layout_str = _layout_str("c", c_layout)

    ab_dtype_name = str(a.dtype).split(".")[-1]
    c_dtype_name = str(c.dtype).split(".")[-1]
    (
        bm, bn, bk, atom_layout_mnk, ab_stages, raster_along_m,
        cluster_shape_mn,
    ) = _resolve_tile_config(c_dtype_name)

    if c_dtype_name == "bfloat16":
        if int(Ka) <= 8192:
            # BF16 fwd (K=4096): B-multicast probe (cluster=(2, 1)).
            cluster_shape_mn = (2, 1)
        else:
            # BF16 dgrad (K=16384): floor (cluster=(1, 1)).
            cluster_shape_mn = (1, 1)

    epi_vec_path = "vec" if _safe_align(c) >= 16 else "scalar"

    use_stmatrix_epi = _safe_align(c) >= 16
    if use_stmatrix_epi:
        epi_stage = 2  # 2-stage SMEM ping-pong for the stmatrix epilogue
    else:
        epi_stage = 0

    if use_stmatrix_epi:
        if c_dtype_name == "float32":
            bm = 128
            atom_layout_mnk = (2, 1, 1)
            ab_stages = 4
            k_pipe_mmas = 2
        else:
            ab_stages = 4  # BF16 stmatrix path: 4-stage K pipeline
            if int(Ka) <= 8192:
                k_pipe_mmas = 1   # BF16 fwd (K=4096): single-stage WGMMA pipe
            else:
                k_pipe_mmas = 2   # BF16 dgrad (K=16384): 2-stage deep WGMMA pipe
    elif c_dtype_name == "float32":
        # FP32 wgrad accumulate=True path — legacy scalar epilogue
        # (full BM × BN × 4 B sC = 64 KB at BM=64).  Keeps the IR
        # BM=64 atom=(1,1,1) ab=4 k_pipe=2.  At BM=128 the scalar
        # sC would be 128 KB → overflow, so this path stays on the
        # smaller tile shape.
        ab_stages = 4
        k_pipe_mmas = 2
    else:
        k_pipe_mmas = 1

    if c_dtype_name == "float32":
        mma_reg_requirement = _MMA_REG_REQUIREMENT_FP32  # 128 reg/thread
    else:
        # BF16 fwd / dgrad share a single MMA register cap.
        #
        # NCU at the BM=128 atom=(2,1,1) IR shows the compiler
        # naturally allocates ~168 reg/thread for both directions, so
        # 168 is the natural cap.  At 232 the extra 64 reg/thread are
        # reserved-but-unused (per-block reg budget 232×256 + 40×128
        # = 64,512 vs 168×256 + 40×128 = 48,128).  Still SMEM-bound
        # at 227 KB / 228 KB cap, so neither cap fits 2 blocks/SM.
        mma_reg_requirement = 168

    if c_dtype_name == "float32" and _safe_align(c) >= 32:
        epi_vec_bits = 256
    else:
        epi_vec_bits = 128

    epi_fp32_rs_bits = 64  # 64-bit R2S vectors (2 FP32 elements per store)

    if use_stmatrix_epi and c_dtype_name == "bfloat16":
        epi_tile_n_override = 64  # BF16 stmatrix path: 64-cell N epi-tile
    else:
        epi_tile_n_override = None  # FP32 wgrad M5: canonical 32 (SMEM-bound)

    swizzle_size = 1

    runner = _compile_for(
        M=int(M), N=int(N), K=int(Ka),
        a_layout_str=a_layout_str,
        b_layout_str=b_layout_str,
        c_layout_str=c_layout_str,
        ab_dtype_name=ab_dtype_name,
        c_dtype_name=c_dtype_name,
        accumulate=bool(accumulate),
        ab_stages=ab_stages,
        bm=bm, bn=bn, bk=bk,
        atom_layout_mnk=atom_layout_mnk,
        raster_along_m=raster_along_m,
        cluster_shape_mn=cluster_shape_mn,
        epi_vec_path=epi_vec_path,
        use_stmatrix_epi=use_stmatrix_epi,
        epi_stage=epi_stage,
        k_pipe_mmas=k_pipe_mmas,
        mma_reg_requirement=mma_reg_requirement,
        epi_vec_bits=epi_vec_bits,
        epi_fp32_rs_bits=epi_fp32_rs_bits,
        epi_tile_n_override=epi_tile_n_override,
        swizzle_size=swizzle_size,
    )
    runner(a, b, c)
    return c

__all__ = ["run_gemm"]
