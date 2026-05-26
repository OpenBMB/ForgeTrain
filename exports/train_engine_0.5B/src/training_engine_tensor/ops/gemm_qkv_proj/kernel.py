"""CuTeDSL SM90 WGMMA+TMA kernel for gemm_qkv_proj (QKV projection GEMM).

Three directions:
  fwd:   Y[M,N] = X[M,K] @ W^T      M=40960 N=1280 K=1024  BF16→BF16
  dgrad: dX[M,N] = dY[M,K] @ W      M=40960 N=1024 K=1280  BF16→BF16
  wgrad: dW[M,N] = dY^T[M,K] @ X    M=1280  N=1024 K=40960 BF16→FP32

v1 (default-after-Round-7): non-persistent WGMMA+TMA, tile (128,256).
   wgrad: batched split-K=8 (precision needs ≥ 8 chunks vs FP64 ref).

v2 (Round 29): persistent kernel with warp specialization (DMA + MMA).
   Reference: gemm_fc2 v10 / CuTeDSL dense_gemm_persistent.py.
   Selected per-direction via env vars:
     QKV_FWD_PERSISTENT=1      use persistent for fwd
     QKV_DGRAD_PERSISTENT=1    use persistent for dgrad
     QKV_WGRAD_PERSISTENT=1    use persistent for wgrad (still split-K=8)
   Cluster shape and tile size for the persistent path are also tunable
   via env vars (see _persistent_cfg below).
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import subprocess
import sys
import time
import torch

# CuTeDSL PYTHONPATH fallback: shared filesystem paths where cutlass-dsl
# may live on cctl pods (not pip-installed in ephemeral images).
# Caller sets ``CUTLASS_DSL_FALLBACK_DIR`` to the package root containing
# ``python_packages/`` + ``lib/``; ``CUTLASS_INSTALL_DIR`` continues to
# work for legacy callers that point at the same root.
_fb_dir = os.environ.get("CUTLASS_DSL_FALLBACK_DIR", "")
_inst_dir = os.environ.get("CUTLASS_INSTALL_DIR", "")
for _p in [
    os.path.join(_fb_dir, "python_packages") if _fb_dir else "",
    os.path.join(_inst_dir, "python_packages") if _inst_dir else "",
]:
    if _p and _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline

BF16 = cutlass.BFloat16
FP32 = cutlass.Float32

# ---------------------------------------------------------------------------
#  AOT C-export backend (shape-specialized CuTeDSL → .o → pybind11)
# ---------------------------------------------------------------------------

_aot_ext = None
_aot_ext_loaded = False

# Shared-cache convention (see workload/ops/gemm_fc1/kernel.py).
EXPORT_DIR = os.path.join(
    os.environ.get("CUTEDSL_CACHE_ROOT", "/tmp"),
    "cutedsl_export_gemm_qkv_proj",
)
_LOCK_PATH = os.path.join(EXPORT_DIR, ".export.lock")
_AOT_DIRECTIONS = ("gemm_fwd", "gemm_dgrad", "gemm_wgrad")
_MIN_OBJ_SIZE = 4096


def _try_cutedsl_export():
    """Run CuTeDSL export if .o files don't exist or config has changed."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(src_dir, "export_kernels.py")
    if not os.path.exists(export_script):
        return False

    os.makedirs(EXPORT_DIR, exist_ok=True)

    needed = [os.path.join(EXPORT_DIR, d, f"{d}.o") for d in _AOT_DIRECTIONS]
    headers = [os.path.join(EXPORT_DIR, d, f"{d}.h") for d in _AOT_DIRECTIONS]
    all_files = needed + headers

    with open(export_script, "rb") as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    config_hash_path = os.path.join(EXPORT_DIR, ".config_hash")

    def _files_valid():
        for p in all_files:
            if not os.path.exists(p):
                return False
        for p in needed:
            if os.path.getsize(p) < _MIN_OBJ_SIZE:
                return False
        return True

    def _hash_matches():
        if not os.path.exists(config_hash_path):
            return False
        try:
            with open(config_hash_path) as f:
                return f.read().strip() == current_hash
        except OSError:
            return False

    if _hash_matches() and _files_valid():
        return True

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    if rank == 0:
        lock_fd = open(_LOCK_PATH, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if _hash_matches() and _files_valid():
                return True

            result = subprocess.run(
                [sys.executable, export_script],
                cwd=src_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f"[gemm_qkv_proj] CuTeDSL export failed (rc={result.returncode}): "
                      f"{result.stderr[:500]}", file=sys.stderr)
                return False
            if not _files_valid():
                print("[gemm_qkv_proj] CuTeDSL export produced incomplete files",
                      file=sys.stderr)
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"[gemm_qkv_proj] CuTeDSL export error: {e}", file=sys.stderr)
            return False
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    else:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if _hash_matches() and _files_valid():
                return True
            time.sleep(2)
        print(f"[gemm_qkv_proj] CuTeDSL export: rank {rank} timed out waiting for rank 0",
              file=sys.stderr)
        return False


def _load_cutedsl_ext():
    """Load CuTeDSL C++ extension linking exported .o files."""
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(src_dir, "gemm_cutedsl.cpp")
    reduce_cu_src = os.path.join(src_dir, "_aot_reduce.cu")

    obj_files = [os.path.join(EXPORT_DIR, d, f"{d}.o") for d in _AOT_DIRECTIONS]
    header_files = [os.path.join(EXPORT_DIR, d, f"{d}.h") for d in _AOT_DIRECTIONS]

    h = hashlib.md5()
    for p in header_files + obj_files + [reduce_cu_src]:
        with open(p, "rb") as f:
            h.update(f.read())
    build_hash = h.hexdigest()[:8]
    ext_name = f"gemm_qkv_proj_cutedsl_{build_hash}"

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    include_dirs = [
        os.path.join(EXPORT_DIR, d) for d in _AOT_DIRECTIONS
    ] + [os.path.join(cuda_home, "include")]

    runtime_lib = "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a"
    if not os.path.exists(runtime_lib):
        import importlib.util
        spec = importlib.util.find_spec("nvidia_cutlass_dsl")
        if spec and spec.submodule_search_locations:
            alt = os.path.join(spec.submodule_search_locations[0],
                               "lib", "libcuda_dialect_runtime_static.a")
            if os.path.exists(alt):
                runtime_lib = alt
    if not os.path.exists(runtime_lib):
        _candidates = [
            "/opt/cutlass_dsl/nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a",
            "/usr/local/lib/python3.12/dist-packages/cutlass_dsl/lib/libcuda_dialect_runtime_static.a",
        ]
        _fb_dir = os.environ.get("CUTLASS_DSL_FALLBACK_DIR", "")
        if _fb_dir:
            _candidates.append(os.path.join(_fb_dir, "lib",
                                            "libcuda_dialect_runtime_static.a"))
        for cand in _candidates:
            if os.path.exists(cand):
                runtime_lib = cand
                break
    if not os.path.exists(runtime_lib):
        cutlass_install_dir = os.environ.get("CUTLASS_INSTALL_DIR")
        if cutlass_install_dir:
            alt = os.path.join(cutlass_install_dir,
                               "lib", "libcuda_dialect_runtime_static.a")
            if os.path.exists(alt):
                runtime_lib = alt

    ext = load(
        name=ext_name,
        sources=[cpp_src, reduce_cu_src],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=obj_files + [
            "-Wl,--whole-archive", runtime_lib, "-Wl,--no-whole-archive",
            "-lcuda",
            "-L" + os.path.join(cuda_home, "lib64"), "-lcudart",
            "-Wl,-rpath," + os.path.join(cuda_home, "lib64"),
        ],
        verbose=True,
    )
    return ext


def _get_aot_ext():
    """Lazy-load AOT C++ extension; returns None on failure."""
    global _aot_ext, _aot_ext_loaded
    if _aot_ext_loaded:
        return _aot_ext
    _aot_ext_loaded = True

    if os.environ.get("QKV_AOT", "1") == "0":
        return None

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    try:
        if _try_cutedsl_export():
            _aot_ext = _load_cutedsl_ext()
            if rank == 0:
                print("[gemm_qkv_proj] loaded AOT CuTeDSL C++ backend", flush=True)
    except Exception as e:
        if rank == 0:
            print(f"[gemm_qkv_proj] AOT C++ backend failed, falling back to JIT: {e}",
                  file=sys.stderr, flush=True)
        _aot_ext = None
    return _aot_ext


class _HopperGemm:
    """Hopper WGMMA+TMA GEMM for BF16 input, FP32 accumulation.

    Supports BF16 or FP32 output (determined by c tensor element type).
    Adapted from CuTeDSL dense_gemm.py, single-batch only.
    """

    def __init__(self, acc_dtype, tile_shape_mn, cluster_shape_mn=(1, 1),
                 force_atom_layout=None, occupancy=1, k_pipe_mmas=1,
                 mma_inst_tile_k=4):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        if force_atom_layout is not None:
            self.atom_layout_mnk = force_atom_layout
        else:
            self.atom_layout_mnk = (
                (2, 1, 1)
                if tile_shape_mn[0] > 64 and tile_shape_mn[1] > 128
                else (1, 1, 1)
            )
        self.mma_inst_tile_k = mma_inst_tile_k
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = occupancy
        self.k_pipe_mmas = k_pipe_mmas
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile N must be 64/128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * self.mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(self, a, b, c, stream):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mn)

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

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a,
        mA_mkl,
        tma_atom_b,
        mB_nkl,
        tma_atom_c,
        mC_mnl,
        tiled_mma,
        cta_layout_mnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        epi_smem_layout_staged,
    ):
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
        s_shape = (
            (group_size_m, cdimx // group_size_m),
            cdimy,
        )
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
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
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
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
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)
        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma_partition,
            gB_for_tma_partition,
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
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if warp_idx == 0:
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # Prologue MMAs
        k_pipe_mmas = self.k_pipe_mmas

        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # Mainloop
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )

            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # Epilogue
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)

        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
        )

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
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

        return

    @staticmethod
    def _compute_stages(tile_shape_mnk, a_dtype, b_dtype, smem_capacity, occupancy):
        epi_stage = 4
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
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk, epi_tile,
        a_dtype, a_layout, b_dtype, b_layout, ab_stage,
        c_dtype, c_layout, epi_stage,
    ):
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, ab_stage,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, ab_stage,
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, epi_stage,
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(c, tile_shape_mnk, cluster_shape_mn):
        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        clusters = cute.ceil_div(
            cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl
        )
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))
        return grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_c, epi_smem_layout_staged, epi_tile):
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


# ---------------------------------------------------------------------------
#  Persistent SM90 WGMMA + warp-specialization (v2)
# ---------------------------------------------------------------------------
#
# Adapted from gemm_fc2 v10 / CuTeDSL dense_gemm_persistent.py.
# Supports L >= 1 batches (wgrad with split-K uses L=NUM_SPLITS).
#
class _HopperGemmPersistent:
    """Persistent Hopper WGMMA+TMA GEMM with warp specialization.

    DMA warp group does TMA loads, MMA warp group(s) do WGMMA + epilogue.
    CTAs stay resident and process multiple tiles via
    StaticPersistentTileScheduler.
    """

    def __init__(self, acc_dtype, tile_shape_mn, cluster_shape_mn=(1, 1),
                 swizzle_size=1, raster_along_m=True, mma_inst_tile_k=4,
                 epi_stage_override: int | None = None):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.mma_inst_tile_k = mma_inst_tile_k
        # Round 54: epi_stage tunable. None → use the historical default of 4.
        self.epi_stage_override = epi_stage_override
        # Round 55 verified: k_pipe_mmas=1 is the optimum for this kernel.
        # Tried 2x2x2 grid over (kp_fwd, kp_dgrad, kp_wgrad) ∈ {1,2}^3;
        # every kp>1 config matched or regressed vs all-1 (bwd worse by
        # 1-4% for kp_wgrad/kp_dgrad=2). Not exposed as a knob.
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
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            (self.num_dma_warp_groups + self.num_mma_warp_groups)
            * self.num_threads_per_warp_group
        )
        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile N must be 64/128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype, self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0], self.tile_shape_mnk[1],
            mma_inst_shape_k * self.mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * self.a_dtype.width // 8
            + cute.size(b_shape) * self.b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(self.epi_tile) * self.c_dtype.width // 8
        # Round 54: epi_stage tunable. Larger -> more in-flight epilogue
        # tiles (helps if epilogue is bottleneck); smaller -> more SMEM for
        # `ab_stage` (deeper mainloop pipeline).
        self.epi_stage = (
            self.epi_stage_override if self.epi_stage_override is not None else 4
        )
        epi_bytes = c_bytes_per_stage * self.epi_stage
        mbar_helpers_bytes = 1024
        self.ab_stage = (
            self.smem_capacity // self.occupancy
            - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout, self.tile_shape_mnk, self.a_dtype, self.ab_stage,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stage,
        )
        self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.epi_stage,
        )

    @cute.jit
    def __call__(self, a, b, c, max_active_clusters, stream):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = _HopperGemm._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = _HopperGemm._make_tma_atoms_and_tensors(
            b, self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )
        tma_atom_c, tma_tensor_c = _HopperGemm._make_tma_store_atoms_and_tensors(
            c, self.epi_smem_layout_staged, self.epi_tile,
        )

        c_shape = cute.slice_(self.tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*self.cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl,
            self.swizzle_size, self.raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

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
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl,
        tma_atom_c, mC_mnl, tiled_mma, cta_layout_mnk,
        a_smem_layout_staged, b_smem_layout_staged,
        epi_smem_layout_staged, tile_sched_params,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
        )

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

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

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, a_cta_crd, a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, b_cta_crd, b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups

        if is_dma_warp_group:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                    cute.copy(
                        tma_atom_a, tAgA_k, tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b, tBgB_k, tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        if not is_dma_warp_group:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(), 4,
                ),
                self.c_dtype,
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]

                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None, None, k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, accumulators,
                            tCrA[k_block_coord], tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()

                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None, None, k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma, accumulators,
                            tCrA[k_block_coord], tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                tCgC_for_tma_partition = cute.zipped_divide(
                    gC_mnl_slice, self.epi_tile
                )
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c, 0, cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma_partition,
                )

                epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )

                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                        tRS_sD, mode=[3]
                    )
                    cute.copy(
                        tiled_copy_r2s, tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                    self.epilog_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()

        return


# ---------------------------------------------------------------------------
#  Compiled kernel cache
# ---------------------------------------------------------------------------
_compiled_cache: dict[str, object] = {}

_H100_SM_COUNT = 132


# ---------------------------------------------------------------------------
#  Custom CUDA C++ extension for FP32 N-way reduction (wgrad split-K)
# ---------------------------------------------------------------------------
_reduce_ext = None
_reduce_ext_load_failed = False
_reduce_ext_v48 = None
_reduce_ext_v48_load_failed = False


def _load_reduce_ext():
    """Load the float4-vectorized 8-way FP32 reducer (lazy, cached, single-shot).

    Falls back to torch.sum if the build fails (e.g. nvcc unavailable).
    Compiles once per process; cached on disk by torch.utils.cpp_extension.
    """
    global _reduce_ext, _reduce_ext_load_failed
    if _reduce_ext is not None or _reduce_ext_load_failed:
        return _reduce_ext

    if os.environ.get("QKV_DISABLE_CUSTOM_REDUCE", "0") == "1":
        _reduce_ext_load_failed = True
        return None

    try:
        from torch.utils.cpp_extension import load
        # Round 40 committed the reducer source as `_reduce_kernel.cpp` but
        # the file actually contains CUDA __global__ + <<<>>> launch syntax.
        # torch.utils.cpp_extension dispatches compilers by file extension:
        # .cpp goes to plain g++ (cannot parse CUDA), .cu goes to nvcc. So
        # we feed `_reduce_kernel.cu` (a one-line shim that #includes the
        # .cpp) to the loader. The .cpp remains the source of truth.
        # (Round 40 originally referenced `reduce_partials.cu`, which never
        #  existed in the committed tree → JIT silently fell back to
        #  torch.sum on every clean checkout.)
        op_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(op_dir, "_reduce_kernel.cu")
        cpp_path = os.path.join(op_dir, "_reduce_kernel.cpp")
        if not os.path.exists(src_path) or not os.path.exists(cpp_path):
            _reduce_ext_load_failed = True
            return None
        # Suppress "TORCH_CUDA_ARCH_LIST not set" warning by setting it.
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        build_dir = os.environ.get(
            "QKV_REDUCE_BUILD_DIR",
            "/tmp/qkv_proj_reduce_build_r39_into",
        )
        os.makedirs(build_dir, exist_ok=True)
        _reduce_ext = load(
            name="qkv_proj_reduce_r39_into",
            sources=[src_path],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        _reduce_ext_load_failed = True
        _reduce_ext = None
        if os.environ.get("QKV_REDUCE_LOAD_VERBOSE", "0") == "1":
            import traceback as _tb
            print("[qkv_reduce] JIT build failed:", repr(e))
            _tb.print_exc()
    return _reduce_ext


def _v48_selftest(ext) -> bool:
    """Round 49: tiny on-GPU correctness gate for the v48 reducer.

    Round 48 staged the v48 reducer / NUM_SPLITS={2,4,8} dispatch but the
    project-target devspace fell out of Teleport inventory before any GPU
    verification could run, so the kernel's correctness is unknown — even
    though it looks right by inspection (per-element sum order is the same
    sequential `0 + a0 + a1 + ... + a_{NSPLIT-1}` as the round-40 reducer).

    To make the v48 path safe to commit *with default-on* despite that gap,
    we run a small self-test on first JIT load: for each NSPLIT ∈ {2, 4, 8},
    we exercise both the fast (n_vec_4 % 4 == 0) and slow (n_vec_4 % 4 != 0)
    code paths against a reference reduction that sums sequentially p=0..L-1
    (which both v40 and v48 *also* do).  Reference is computed on-GPU using
    the exact same sequential sum order so any pass-through (bit-identical)
    kernel matches; tolerance is set wide enough that float associativity
    differences inside torch's underlying reduction are still acceptable.

    Returns True on PASS.  Any exception or numerical mismatch returns False.
    Callers treat False as "fall back to v40 / torch.sum permanently".
    """
    if not torch.cuda.is_available():
        return False
    try:
        import torch as _t
        for nsplit, alloc_name, into_name in [
            (8, "reduce_8way_v48", "reduce_8way_v48_into"),
            (4, "reduce_4way_v48", "reduce_4way_v48_into"),
            (2, "reduce_2way_v48", "reduce_2way_v48_into"),
        ]:
            alloc_fn = getattr(ext, alloc_name, None)
            into_fn = getattr(ext, into_name, None)
            if alloc_fn is None or into_fn is None:
                return False
            for shape in [(nsplit, 4, 16), (nsplit, 1, 12)]:
                # Use small but spread-out values to surface any wrong-stride or
                # off-by-one bug clearly while keeping compare tolerances tight.
                x = _t.randn(*shape, dtype=_t.float32, device='cuda')
                # Reference: sum sequentially p=0..L-1 (same order as v48 kernel).
                ref = _t.zeros_like(x[0])
                for p in range(nsplit):
                    ref = ref + x[p]
                # alloc variant
                y_alloc = alloc_fn(x.contiguous())
                if y_alloc.shape != ref.shape:
                    return False
                if not _t.allclose(y_alloc, ref, rtol=1e-4, atol=1e-5):
                    return False
                # _into variant
                y_into = _t.empty_like(ref)
                into_fn(x.contiguous(), y_into)
                if not _t.allclose(y_into, ref, rtol=1e-4, atol=1e-5):
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_reduce_ext_v48():
    """Round 48: faster reducer (4-float4 ILP per thread, supports 4/8-way).

    Built from `_reduce_kernel_v2.cu`.  Lazy, single-shot.  Returns None on
    build failure or self-test failure; callers fall back to the round-40
    8-way reducer or `torch.sum`.

    Round 49 added an on-GPU self-test (`_v48_selftest`) that runs once per
    process after the JIT build succeeds.  If the self-test fails (or CUDA is
    unavailable, or the kernel symbols are missing), the v48 path is marked
    permanently broken for this process and the caller silently falls back.
    Set `QKV_DISABLE_REDUCE_V48=1` to skip v48 entirely (force v40 / torch.sum).
    Set `QKV_REDUCE_SKIP_SELFTEST=1` to bypass the self-test (debug only).
    """
    global _reduce_ext_v48, _reduce_ext_v48_load_failed
    if _reduce_ext_v48 is not None or _reduce_ext_v48_load_failed:
        return _reduce_ext_v48

    if os.environ.get("QKV_DISABLE_REDUCE_V48", "0") == "1":
        _reduce_ext_v48_load_failed = True
        return None

    try:
        from torch.utils.cpp_extension import load
        op_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(op_dir, "_reduce_kernel_v2.cu")
        if not os.path.exists(src_path):
            _reduce_ext_v48_load_failed = True
            return None
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        build_dir = os.environ.get(
            "QKV_REDUCE_V48_BUILD_DIR",
            "/tmp/qkv_proj_reduce_build_r48",
        )
        os.makedirs(build_dir, exist_ok=True)
        ext = load(
            name="qkv_proj_reduce_v48",
            sources=[src_path],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001
        _reduce_ext_v48_load_failed = True
        _reduce_ext_v48 = None
        if os.environ.get("QKV_REDUCE_LOAD_VERBOSE", "0") == "1":
            import traceback as _tb
            print("[qkv_reduce_v48] JIT build failed:", repr(e))
            _tb.print_exc()
        return None

    if os.environ.get("QKV_REDUCE_SKIP_SELFTEST", "0") == "1":
        _reduce_ext_v48 = ext
        return _reduce_ext_v48

    if not _v48_selftest(ext):
        _reduce_ext_v48_load_failed = True
        _reduce_ext_v48 = None
        if os.environ.get("QKV_REDUCE_LOAD_VERBOSE", "0") == "1":
            print("[qkv_reduce_v48] self-test FAILED — falling back to v40")
        return None

    _reduce_ext_v48 = ext
    return _reduce_ext_v48


# Round 53 cont: cache resolved (into_fn, alloc_fn) per num_splits so the
# fast path avoids 6 attribute lookups + branch testing per call.
_REDUCE_FN_CACHE: dict[int, tuple] = {}


def _resolve_reduce_fns(num_splits: int) -> tuple:
    """Return (into_fn, alloc_fn, prefers_v48) for a given num_splits, cached."""
    cached = _REDUCE_FN_CACHE.get(num_splits)
    if cached is not None:
        return cached
    variant = _reduce_variant()
    ext_v48 = _load_reduce_ext_v48() if variant == "v48" else None
    ext_v40 = _load_reduce_ext()

    if ext_v48 is not None and num_splits in (2, 4, 8):
        suffix = {2: "2way_v48", 4: "4way_v48", 8: "8way_v48"}[num_splits]
        into_fn = getattr(ext_v48, f"reduce_{suffix}_into", None)
        alloc_fn = getattr(ext_v48, f"reduce_{suffix}", None)
        if into_fn is not None or alloc_fn is not None:
            res = (into_fn, alloc_fn, True)
            _REDUCE_FN_CACHE[num_splits] = res
            return res

    if ext_v40 is not None and num_splits == 8:
        into_fn = getattr(ext_v40, "reduce_8way_fp32_into", None)
        alloc_fn = getattr(ext_v40, "reduce_8way_fp32", None)
        res = (into_fn, alloc_fn, False)
        _REDUCE_FN_CACHE[num_splits] = res
        return res

    res = (None, None, False)
    _REDUCE_FN_CACHE[num_splits] = res
    return res


def _reduce_split_partials(partials_flat: torch.Tensor,
                           M: int, N: int,
                           num_splits: int,
                           out: torch.Tensor | None = None) -> torch.Tensor:
    """Reduce partials_flat[L, M, N] FP32 -> [M, N] FP32 (alloc or fill `out`).

    When `out` is provided, write into it and return it (used by the overlap
    path so the destination is allocated on the *main* stream's allocator
    context rather than inside the side-stream context — without this, the
    C++-side `torch::empty` causes an allocator/stream interaction that
    doubles the bwd latency, see Round 42 notes).

    Round 53: dispatch via cached `_REDUCE_FN_CACHE` keyed by num_splits to
    skip 6 attribute lookups + branch testing per call.
    """
    total = M * N
    into_fn, alloc_fn, _ = _resolve_reduce_fns(num_splits)

    if into_fn is not None and out is not None and (total % 4) == 0:
        into_fn(partials_flat, out)
        return out
    if alloc_fn is not None and out is None and (total % 4) == 0:
        return alloc_fn(partials_flat)

    # Generic torch fallback.
    if out is not None:
        torch.sum(partials_flat.view(num_splits, -1), dim=0,
                  out=out.view(-1))
        return out
    return partials_flat.view(num_splits, -1).sum(dim=0).view(M, N)

# Persistent partials buffer for wgrad split-K reduction.
# Allocating ~40MB per call (catching allocator should be near-free, but keeping
# it stable also lets the kernel cache reuse the same TMA descriptors).
# Stored as (partials_flat, partials_perm) where:
#   - partials_flat [L, M, N]  : contig storage; reducer reads this directly
#   - partials_perm [M, N, L]  : permuted view passed to the wgrad GEMM kernel
_wgrad_partials_cache: dict[tuple, tuple] = {}

# Side stream for overlapping wgrad with dgrad (independent kernels).
_wgrad_stream: dict[int, torch.cuda.Stream] = {}


def _get_wgrad_stream(device_index: int) -> torch.cuda.Stream:
    s = _wgrad_stream.get(device_index)
    if s is None:
        # Round 45: high-priority side stream so the GPU scheduler picks
        # wgrad CTAs ahead of subsequent dgrad CTAs when both have
        # in-flight blocks. Empirically helps the dgrad/wgrad kernels
        # share SMs more evenly when both are persistent.
        priority = int(os.environ.get("QKV_WGRAD_STREAM_PRIORITY", "-1"))
        s = torch.cuda.Stream(device=device_index, priority=priority)
        _wgrad_stream[device_index] = s
    return s

# Cache cute.Tensor descriptors keyed by torch tensor data_ptr.
# Saves the from_dlpack + mark_layout_dynamic overhead (~5us) per call.
# NB: the cute.Tensor wraps a static pointer, so when the underlying torch
# tensor is replaced (data_ptr changes) we must rebuild.
#
# Stable tensors (weight, partials buffer): Adam keeps weight storage stable
# across steps, partials is allocated once and cached. Hits ~100% of the time.
#
# Round 73: extended to non-stable tensors (x, dy, out, d_input, etc.) too.
# Microprobe (`_round73_descprobe.py`) measures uncached=5.68us vs cached=0.53us
# per call — a 5.15us/call saving when the same data_ptr repeats. PyTorch's
# caching allocator reuses a small pool of buffers per shape during steady
# state, so even in real training a small data_ptr cycle hits the cache
# after the first few calls.  In isolation bench (R52 ab_bench) the same
# tensor is reused across iters → 100% hit rate after warmup.
#
# A bounded FIFO eviction policy (max ``_CUTE_TENSOR_CACHE_MAX`` entries)
# prevents unbounded growth if the allocator hands out many distinct
# data_ptrs over time. The cute.Tensor entries are small (~few KB each) so
# the cap doubles as a memory ceiling.
_cute_tensor_cache: dict[tuple, "cute.Tensor"] = {}
_CUTE_TENSOR_CACHE_MAX = 256


def _to_cute_3d_cached(t: torch.Tensor, cutlass_dtype) -> "cute.Tensor":
    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), id(cutlass_dtype))
    cached = _cute_tensor_cache.get(key)
    if cached is not None:
        return cached
    ct = _to_cute_3d(t, cutlass_dtype)
    if len(_cute_tensor_cache) >= _CUTE_TENSOR_CACHE_MAX:
        # FIFO eviction; preserves ordering since dict is insertion-ordered
        # in Python 3.7+. Cheap (~0.3us) and bounds the cache.
        _cute_tensor_cache.pop(next(iter(_cute_tensor_cache)))
    _cute_tensor_cache[key] = ct
    return ct

def _to_cute_3d(t: torch.Tensor, cutlass_dtype) -> cute.Tensor:
    """Convert a 2D or 3D PyTorch tensor to a 3D CuTe tensor.

    2D inputs get unsqueezed to (M, K, 1).
    3D inputs are used directly as (M, K, L) for batched GEMM.
    Row-major (K-major): stride(1)==1, leading_dim=1.
    Col-major (M-major): stride(0)==1, leading_dim=0.
    """
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    assert t.ndim == 3, f"Expected 2D or 3D, got shape={t.shape}"
    ct = from_dlpack(t, assumed_align=16)
    ct.element_type = cutlass_dtype
    leading_dim = 1 if t.stride(1) == 1 else 0
    ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


# Round 57: cache CUstream wrappers keyed by raw cuda_stream handle.
# The hot path calls torch.cuda.current_stream() (returns a Python wrapper
# allocated each call ~1.5us), then constructs cuda.CUstream from the raw
# handle (~1.5us). Both the main and side stream handles are stable for the
# process lifetime, so a 2-entry dict keyed by the int handle returns the
# pre-built CUstream in ~0.3us.
_stream_cache: dict[int, "cuda.CUstream"] = {}


def _get_stream():
    raw = torch.cuda.current_stream().cuda_stream
    cached = _stream_cache.get(raw)
    if cached is None:
        cached = cuda.CUstream(raw)
        _stream_cache[raw] = cached
    return cached


# Round 70: cache torch.cuda.Stream wrappers for the *main* stream of a
# given device, keyed by raw cuda_stream handle. `torch.cuda.current_stream`
# allocates a fresh Python `Stream` object every call (~3.94us) — when the
# caller hasn't switched streams (the common case in training) this is
# pure waste because we already have a Stream wrapper for that same raw
# handle from the previous call. The cache cost is ~0.14us per call.
_main_stream_cache: dict[int, torch.cuda.Stream] = {}
_get_current_stream_raw = torch._C._cuda_getCurrentStream
_get_current_device = torch.cuda.current_device


def _get_main_stream(device: torch.device) -> torch.cuda.Stream:
    """Cached `torch.cuda.current_stream(device)` keyed by raw handle.

    Falls back to the slow path (allocating a fresh `Stream` wrapper) if
    the caller has switched the current stream since the last cache hit.
    """
    dev_idx = device.index if device.index is not None else _get_current_device()
    raw = _get_current_stream_raw(dev_idx)[0]
    cached = _main_stream_cache.get(raw)
    if cached is not None:
        return cached
    s = torch.cuda.current_stream(device)
    _main_stream_cache[raw] = s
    return s


def _run_gemm(
    key: str,
    a_torch: torch.Tensor,
    b_torch: torch.Tensor,
    c_torch: torch.Tensor,
    a_cutlass_dtype,
    b_cutlass_dtype,
    c_cutlass_dtype,
    tile_mn: tuple[int, int] = (128, 128),
    cluster_mn: tuple[int, int] = (1, 1),
    force_atom_layout=None,
    occupancy: int = 1,
    k_pipe_mmas: int = 1,
    mma_inst_tile_k: int = 4,
    a_stable: bool = False,
    b_stable: bool = False,
    c_stable: bool = False,
):
    """Run a non-persistent GEMM using a cached compiled kernel.

    `*_stable` flags hint that the tensor's underlying storage is reused
    across calls (e.g. weight, cached partials buffer). When True, the
    cute.Tensor descriptor is cached to avoid the per-call from_dlpack +
    mark_layout_dynamic overhead (~10us per descriptor).
    """
    # Round 57 hot-path slim (same scheme as _run_gemm_persistent).
    # Round 73: always use _to_cute_3d_cached. The cache is keyed by
    # (data_ptr, shape, stride, dtype); cache miss costs the same as
    # _to_cute_3d (~5.7us). Bench reuses the same tensor → 100% hit.
    # `*_stable` flags are kept for backward compat / documentation but
    # no longer affect the code path.
    stream = _get_stream()
    if _DESC_CACHE_ENABLED:
        mA = _to_cute_3d_cached(a_torch, a_cutlass_dtype)
        mB = _to_cute_3d_cached(b_torch, b_cutlass_dtype)
        mC = _to_cute_3d_cached(c_torch, c_cutlass_dtype)
    else:
        mA = _to_cute_3d(a_torch, a_cutlass_dtype)
        mB = _to_cute_3d(b_torch, b_cutlass_dtype)
        mC = _to_cute_3d(c_torch, c_cutlass_dtype)

    cached_kernel = _compiled_cache.get(key)
    if cached_kernel is None:
        gemm = _HopperGemm(FP32, tile_mn, cluster_shape_mn=cluster_mn,
                            force_atom_layout=force_atom_layout,
                            occupancy=occupancy, k_pipe_mmas=k_pipe_mmas,
                            mma_inst_tile_k=mma_inst_tile_k)
        cached_kernel = cute.compile(gemm, mA, mB, mC, stream)
        _compiled_cache[key] = cached_kernel

    cached_kernel(mA, mB, mC, stream)


def _run_gemm_persistent(
    key: str,
    a_torch: torch.Tensor,
    b_torch: torch.Tensor,
    c_torch: torch.Tensor,
    a_cutlass_dtype,
    b_cutlass_dtype,
    c_cutlass_dtype,
    tile_mn: tuple[int, int] = (128, 256),
    cluster_mn: tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    max_active: int = _H100_SM_COUNT,
    mma_inst_tile_k: int = 4,
    epi_stage: int | None = None,
    a_stable: bool = False,
    b_stable: bool = False,
    c_stable: bool = False,
):
    """Run a persistent GEMM with warp specialization (DMA+MMA overlap).

    max_active controls how many CTA-clusters get spawned. Reducing it
    below 132 leaves SMs available for a concurrent kernel on a
    different stream (used in the bwd dgrad/wgrad overlap path).

    mma_inst_tile_k controls the K-pipeline depth (multiplier on the
    WGMMA instruction K-shape). Default 4 matches the round-44 winner;
    smaller values trade DMA pipeline depth for larger ab_stage count.

    Round 51: ``*_stable`` flags mirror ``_run_gemm`` — when the
    underlying torch tensor's data_ptr is reused across calls (typically
    weights from the optimizer or the cached wgrad partials buffer),
    the cute.Tensor descriptor is cached to skip the per-call
    ``from_dlpack + mark_layout_dynamic`` overhead (~10-15 µs each).
    Set ``QKV_PERSISTENT_DISABLE_CACHE=1`` to bypass for diagnostics.
    """
    # Round 57 hot-path slim:
    #   (a) `_DESC_CACHE_ENABLED` is a module-level bool refreshed by
    #       `_invalidate_all_env_caches`; skips the per-call
    #       `_persistent_use_descriptor_cache()` function-call cost.
    #   (b) Single `.get(key)` lookup saves a second hash vs `in` + `[]`.
    # Round 73: always cache (cache key includes data_ptr, miss ≡ no-cache).
    # `*_stable` flags are kept for documentation / backward compat but no
    # longer gate the cache path. See _to_cute_3d_cached header.
    stream = _get_stream()
    if _DESC_CACHE_ENABLED:
        mA = _to_cute_3d_cached(a_torch, a_cutlass_dtype)
        mB = _to_cute_3d_cached(b_torch, b_cutlass_dtype)
        mC = _to_cute_3d_cached(c_torch, c_cutlass_dtype)
    else:
        mA = _to_cute_3d(a_torch, a_cutlass_dtype)
        mB = _to_cute_3d(b_torch, b_cutlass_dtype)
        mC = _to_cute_3d(c_torch, c_cutlass_dtype)

    cached_kernel = _compiled_cache.get(key)
    if cached_kernel is None:
        gemm = _HopperGemmPersistent(
            FP32, tile_mn, cluster_shape_mn=cluster_mn,
            swizzle_size=swizzle_size, raster_along_m=raster_along_m,
            mma_inst_tile_k=mma_inst_tile_k,
            epi_stage_override=epi_stage,
        )
        cached_kernel = cute.compile(
            gemm, mA, mB, mC, max_active, stream,
        )
        _compiled_cache[key] = cached_kernel

    cached_kernel(mA, mB, mC, max_active, stream)


# ---------------------------------------------------------------------------
#  v2 persistent kernel configuration (env-var tunable)
# ---------------------------------------------------------------------------

def _env_int_pair(name: str, default: tuple[int, int]) -> tuple[int, int]:
    val = os.environ.get(name)
    if not val:
        return default
    parts = val.replace("x", ",").split(",")
    return (int(parts[0]), int(parts[1]))


# Round 43–44 sweep (132x6 grid: tile, cluster, swizzle, raster) — top configs
# per direction.  Each entry was both op-unit verified and isolation-bench
# faster than the previous default.
#   fwd   : (128,256) c(2,1) sw=1 raster=N  → 0.1601 ms (vs prev 0.1684, -5%)
#   dgrad : (128,256) c(1,2) sw=1 raster=N  → 0.1532 ms (Round 44 sweep; also
#                                              c(2,1) ties to ~0.1532)
#   wgrad : (128,128) c(1,2) sw=2 raster=N  → 0.1602 ms (isolation winner).
#           Round 45 enabled this as the default after the high-priority side
#           stream patch (_get_wgrad_stream priority=-1) made the dgrad/wgrad
#           overlap path actually overlap on the GPU CTA scheduler. With the
#           previous default-priority side stream, two persistent kernels
#           both reserving 132 SMs serialized; with priority -1, the wgrad
#           CTAs get scheduling preference and fill SMs released by dgrad
#           CTAs as they retire. Round-45 maxact sweep showed (132,132)
#           wins (bwd 0.3195 ms; cuBLAS 0.288 ms ≈ 1.11×) with no benefit
#           from trimming max_active for either side.
_PERSISTENT_DEFAULTS: dict[str, dict] = {
    "fwd": {
        "tile_mn": (128, 256),
        "cluster_mn": (2, 1),
        "swizzle_size": 4,  # R-autotune: sw4 > sw1 (+2.3%)
        "raster_along_m": False,
        "default_on": True,
    },
    "dgrad": {
        "tile_mn": (128, 256),
        "cluster_mn": (1, 2),
        "swizzle_size": 2,  # R-autotune: sw2 > sw1 (+1.6%)
        "raster_along_m": False,
        "default_on": True,
    },
    "wgrad": {
        "tile_mn": (128, 128),
        # Round 59: cluster_mn = (1, 4) replaces the historical (1, 2).
        # Under epi_stage=6 (R54), 4-way TMA multicast along N reduces
        # the A-tensor (dY^T) read traffic 4x vs (1, 2)'s 2x. R47 had
        # ruled (1, 4) out under epi=4 because the deeper 4-way
        # broadcast latency wasn't masked by the shallower epilogue
        # pipeline; epi=6 changed that. Isolation wgrad: 167.2us
        # @c(1,2)sw=2 → 161.9us @c(1,4)sw=1 (-5.3us). bwd ab_bench:
        # 378.7us → 374.2us (-4.5us). Note: requires N_tiles % 4 == 0
        # which holds for N_QKV=1280 / N=1024 → 8 N-tiles at tile_n=128.
        "cluster_mn": (1, 4),
        # Round 59: sw=1 wins under c(1,4) (sw=2 ties at 162.7 vs sw=1 161.9).
        "swizzle_size": 1,  # Reverted: auto-tuned sw=4 was -7.8% vs cuBLAS; keep R59 sw=1
        "raster_along_m": False,
        # Round 54: epi_stage=6 wins +4.4us in isolation wgrad sweep
        # (171.8us @ epi=4 → 167.4us @ epi=6). FP32 epilogue benefits
        # from deeper TMA-store pipeline; ab_stage stays sufficient.
        "epi_stage": 4,  # Fixed: epi=6 SMEM overflow on cutlass 4.4.2 + c(1,4)
        "default_on": True,
    },
}


def _persistent_cfg(direction: str) -> dict:
    """Return persistent-kernel config for a direction ('fwd'|'dgrad'|'wgrad')."""
    upper = direction.upper()
    d = _PERSISTENT_DEFAULTS.get(direction, {})
    default_raster = "M" if d.get("raster_along_m", True) else "N"
    epi_stage_env = os.environ.get(f"QKV_{upper}_EPISTAGE")
    epi_stage_default = d.get("epi_stage")
    if epi_stage_env is not None:
        epi_stage = int(epi_stage_env)
    elif epi_stage_default is not None:
        epi_stage = int(epi_stage_default)
    else:
        epi_stage = None
    return {
        "tile_mn": _env_int_pair(f"QKV_{upper}_TILE", d.get("tile_mn", (128, 256))),
        "cluster_mn": _env_int_pair(f"QKV_{upper}_CLUSTER", d.get("cluster_mn", (1, 1))),
        "swizzle_size": int(os.environ.get(f"QKV_{upper}_SWIZZLE", str(d.get("swizzle_size", 1)))),
        "raster_along_m": os.environ.get(f"QKV_{upper}_RASTER", default_raster).upper() == "M",
        "max_active": int(os.environ.get(f"QKV_{upper}_MAXACT", str(d.get("max_active", _H100_SM_COUNT)))),
        "mma_inst_tile_k": int(os.environ.get(f"QKV_{upper}_INSTK", str(d.get("mma_inst_tile_k", 4)))),
        "epi_stage": epi_stage,
    }


def _use_persistent(direction: str) -> bool:
    default = "1" if _PERSISTENT_DEFAULTS.get(direction, {}).get("default_on") else "0"
    return os.environ.get(f"QKV_{direction.upper()}_PERSISTENT", default) == "1"


def _persistent_key(prefix: str, direction: str) -> str:
    cfg = _persistent_cfg(direction)
    tile = f"{cfg['tile_mn'][0]}x{cfg['tile_mn'][1]}"
    cl = f"{cfg['cluster_mn'][0]}x{cfg['cluster_mn'][1]}"
    sw = cfg["swizzle_size"]
    rm = "M" if cfg["raster_along_m"] else "N"
    ik = cfg["mma_inst_tile_k"]
    es = cfg.get("epi_stage")
    es_part = f"_es{es}" if es is not None else ""
    return f"{prefix}_{tile}_c{cl}_s{sw}_r{rm}_ik{ik}{es_part}"


# ---------------------------------------------------------------------------
#  Round 52: hot-path config cache.
#
#  The original `_persistent_cfg` does ~6 `os.environ.get` lookups per call
#  and `_persistent_key` calls it again — totalling ~13-25 env lookups per
#  fwd/bwd. Round 52's overhead probe measured 65us / 173us of *Python-side*
#  per-call work for fwd / bwd respectively (vs cuBLAS 13us / 31us). The
#  GPU still dominated the tight-loop wallclock for our shape, but Python
#  eats wallclock on cold-cache fwd, on smaller GEMMs, and during real
#  training when the launch queue empties between layers.
#
#  This cache reads env once per (prefix, direction) pair on first use, then
#  serves a frozen dict + key string. Use `_invalidate_persistent_cache()`
#  for the bench harnesses that toggle env vars at runtime.
# ---------------------------------------------------------------------------

_PERSISTENT_HOT_CACHE: dict[str, tuple[dict, str, bool]] = {}
# (cfg_dict, kernel_key_string, use_persistent_bool) per direction.

# Per-direction prefix passed to _run_gemm_persistent's compile cache.
_PERSISTENT_PREFIX = {
    "fwd": "qkv_fwd_p",
    "dgrad": "qkv_dgrad_p",
    "wgrad": "qkv_wgrad_p",
}


def _persistent_state(direction: str) -> tuple[dict, str, bool]:
    """Hot-path cached (cfg, key, use_persistent) for a direction.

    First call per direction reads env; subsequent calls return the cached
    values. The cache is invalidated by `_invalidate_persistent_cache()` —
    call this if you toggle env vars at runtime (bench harnesses).
    """
    cached = _PERSISTENT_HOT_CACHE.get(direction)
    if cached is not None:
        return cached
    cfg = _persistent_cfg(direction)
    key = _persistent_key(_PERSISTENT_PREFIX[direction], direction)
    use_p = _use_persistent(direction)
    state = (cfg, key, use_p)
    _PERSISTENT_HOT_CACHE[direction] = state
    return state


def _invalidate_persistent_cache() -> None:
    """Force the next call to re-read env. Used by bench sweeps."""
    _PERSISTENT_HOT_CACHE.clear()


# Cached env lookups for non-persistent (v1) wgrad tile and overlap mode.
_NP_WGRAD_TILE_CACHE: tuple[int, int] | None = None
_NUM_SPLITS_CACHE: int | None = None
_USE_OVERLAP_CACHE: bool | None = None
_REDUCE_VARIANT_CACHE: str | None = None
_PERSISTENT_DESC_CACHE_FLAG: bool | None = None
# Round 57: module-level mirror of _persistent_use_descriptor_cache, set on
# first import / after _invalidate_all_env_caches; readable from hot path
# without a function call.
_DESC_CACHE_ENABLED: bool = (
    os.environ.get("QKV_PERSISTENT_DISABLE_CACHE", "0") != "1"
)



def _reduce_variant() -> str:
    global _REDUCE_VARIANT_CACHE
    if _REDUCE_VARIANT_CACHE is None:
        _REDUCE_VARIANT_CACHE = os.environ.get("QKV_REDUCE_VARIANT", "v48")
    return _REDUCE_VARIANT_CACHE


def _persistent_use_descriptor_cache() -> bool:
    global _PERSISTENT_DESC_CACHE_FLAG, _DESC_CACHE_ENABLED
    if _PERSISTENT_DESC_CACHE_FLAG is None:
        _PERSISTENT_DESC_CACHE_FLAG = (
            os.environ.get("QKV_PERSISTENT_DISABLE_CACHE", "0") != "1"
        )
        _DESC_CACHE_ENABLED = _PERSISTENT_DESC_CACHE_FLAG
    return _PERSISTENT_DESC_CACHE_FLAG


def _np_wgrad_tile() -> tuple[int, int]:
    global _NP_WGRAD_TILE_CACHE
    if _NP_WGRAD_TILE_CACHE is None:
        _NP_WGRAD_TILE_CACHE = _env_int_pair("QKV_WGRAD_NP_TILE", (64, 256))
    return _NP_WGRAD_TILE_CACHE


def _num_splits() -> int:
    global _NUM_SPLITS_CACHE
    if _NUM_SPLITS_CACHE is None:
        _NUM_SPLITS_CACHE = int(os.environ.get("QKV_WGRAD_SPLITS", "4"))
    return _NUM_SPLITS_CACHE


def _use_overlap() -> bool:
    global _USE_OVERLAP_CACHE
    if _USE_OVERLAP_CACHE is None:
        _USE_OVERLAP_CACHE = os.environ.get("QKV_BWD_OVERLAP", "1") == "1"
    return _USE_OVERLAP_CACHE


def _invalidate_all_env_caches() -> None:
    """Used by bench harnesses that toggle env vars at runtime."""
    global _NP_WGRAD_TILE_CACHE, _NUM_SPLITS_CACHE, _USE_OVERLAP_CACHE
    global _REDUCE_VARIANT_CACHE, _PERSISTENT_DESC_CACHE_FLAG
    global _DESC_CACHE_ENABLED
    _NP_WGRAD_TILE_CACHE = None
    _NUM_SPLITS_CACHE = None
    _USE_OVERLAP_CACHE = None
    _REDUCE_VARIANT_CACHE = None
    _PERSISTENT_DESC_CACHE_FLAG = None
    # Re-read env immediately so subsequent hot-path calls see the new value
    # without going through `_persistent_use_descriptor_cache()`.
    _DESC_CACHE_ENABLED = (
        os.environ.get("QKV_PERSISTENT_DISABLE_CACHE", "0") != "1"
    )
    _REDUCE_FN_CACHE.clear()
    _invalidate_persistent_cache()


# ---------------------------------------------------------------------------
#  AOT bwd helper: dgrad (C++) + wgrad (C++) + Python reduce, with overlap.
# ---------------------------------------------------------------------------

_aot_wgrad_partials_cache: dict = {}


def _gemm_qkv_proj_bwd_aot(
    ext, d_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward using AOT C++ kernels.

    R78: bwd_serial (single C++ call for dgrad+wgrad+reduce) is the
    preferred path — lowest overhead (~3µs pybind11 round-trip).
    R78 bench confirmed that Python-side stream overlap adds ~5µs of
    stream management that persistent kernels cannot amortize (they
    saturate 132 SMs and serialize regardless of stream assignment).

    Overlap fallback (when bwd_serial symbol is unavailable) is kept
    for backward compat with older AOT builds.
    """
    shape_x = x.shape
    M_wgrad = d_output.shape[-1]   # 1280
    N_wgrad = x.shape[-1]          # 1024
    NUM_SPLITS = _num_splits()     # 4

    cache_key = (NUM_SPLITS, M_wgrad, N_wgrad, x.device.index)
    cached = _aot_wgrad_partials_cache.get(cache_key)
    if cached is None:
        partials_flat = torch.empty(NUM_SPLITS, M_wgrad, N_wgrad,
                                    dtype=torch.float32, device=x.device)
        _aot_wgrad_partials_cache[cache_key] = partials_flat
    else:
        partials_flat = cached

    d_weight = torch.empty(M_wgrad, N_wgrad, dtype=torch.float32,
                           device=x.device)

    bwd_serial_fn = getattr(ext, "bwd_serial", None)
    if bwd_serial_fn is not None:
        d_input, d_weight = bwd_serial_fn(
            d_output, x, weight, partials_flat, d_weight)
        return d_input.view(shape_x), d_weight

    # Fallback: individual C++ calls when bwd_serial is unavailable.
    d_input = ext.gemm_dgrad_fast(d_output, weight)

    dy_2d = d_output.contiguous().reshape(-1, M_wgrad)
    x_2d = x.contiguous().reshape(-1, N_wgrad)
    ext.gemm_wgrad(dy_2d, x_2d, partials_flat)

    aot_reduce_fn = getattr(ext, "aot_reduce", None)
    if aot_reduce_fn is not None:
        aot_reduce_fn(partials_flat, d_weight)
    else:
        _reduce_split_partials(partials_flat, M_wgrad, N_wgrad, NUM_SPLITS,
                               out=d_weight)

    return d_input.view(shape_x), d_weight


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def gemm_qkv_proj_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Forward: out = x @ weight.T

    x:      [S, B, 1024] BF16
    weight: [1280, 1024] BF16
    Returns [S, B, 1280] BF16.

    CUTLASS convention: D[m,n] = A[m,k] * B[n,k]
    A = X (M×K, K-major), B = W (N×K, K-major), C = Y (M×N, N-major)
    """
    ext = _get_aot_ext()
    if ext is not None:
        return ext.gemm_fwd_fast(x, weight)

    shape = x.shape
    x_2d = x.contiguous().reshape(-1, shape[-1])
    w = weight.contiguous()
    M, K = x_2d.shape
    N = w.shape[0]

    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    cfg, key, use_p = _persistent_state("fwd")
    if use_p:
        _run_gemm_persistent(
            key,
            x_2d, w, out, BF16, BF16, BF16,
            b_stable=True,
            **cfg,
        )
    else:
        _run_gemm("qkv_fwd_256", x_2d, w, out, BF16, BF16, BF16,
                  tile_mn=(128, 256), b_stable=True)

    return out.view(*shape[:-1], N)


def gemm_qkv_proj_bwd(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward: returns (d_input, d_weight_fp32).

    d_output: [S, B, 1280] BF16
    x:        [S, B, 1024] BF16
    weight:   [1280, 1024] BF16
    Returns:  d_input [S, B, 1024] BF16, d_weight [1280, 1024] FP32.

    dgrad: D[M,N] = A[M,K]*B[N,K]  where A=dY, B=W.T (view, M-major)
    wgrad: D[M,N] = A[M,K]*B[N,K]  where A=dY.T (view, M-major), B=X.T (view, M-major)
           split-K via L=NUM_SPLITS batched view (precision needs ≥ 8 chunks).
    """
    ext = _get_aot_ext()
    if ext is not None:
        return _gemm_qkv_proj_bwd_aot(ext, d_output, x, weight)

    shape_dy = d_output.shape
    shape_x = x.shape

    dy_2d = d_output.contiguous().reshape(-1, shape_dy[-1])
    x_2d = x.contiguous().reshape(-1, shape_x[-1])
    w = weight.contiguous()

    M_dgrad = dy_2d.shape[0]
    N_dgrad = w.shape[1]

    d_input = torch.empty(M_dgrad, N_dgrad, dtype=torch.bfloat16, device=x.device)

    M_wgrad = dy_2d.shape[1]   # N_QKV = 1280
    N_wgrad = x_2d.shape[1]    # HIDDEN = 1024
    K_wgrad = dy_2d.shape[0]   # S*B = 40960

    # Round 50: NUM_SPLITS default lowered from 8 to 4.
    # Pareto-optimal pick across two interleaved A/B sweeps over {2,4,8}:
    #   - splits=4 preserves 2x-better-than-baseline wgrad precision
    #     (mean 5.4e-5 vs baseline 1.1e-4) — vs splits=2 which only matches
    #     baseline bitwise — while delivering the same ~4-7µs steady-state
    #     bwd speedup as splits=2.
    #   - splits=4 reduces partials traffic 2x (~10MB vs 20MB) and halves
    #     reducer work vs splits=8.
    #   - Interleaved A/B (8 measurements per config across two GPU thermal
    #     states) gives splits=4 mean ≈ 379.7µs, splits=2 mean ≈ 380.1µs,
    #     splits=8 steady-state mean ≈ 387µs (cold-cache rep 1 pulls naive
    #     mean lower). splits=4 wins by ~7µs over splits=8 with no precision
    #     regression.
    NUM_SPLITS = _num_splits()
    chunk_K = K_wgrad // NUM_SPLITS

    # Zero-copy batched split-K views via as_strided (avoids 190MB copy)
    # a_batched[m, k, s] = dy_2d[s*chunk_K + k, m]  (col-major per batch)
    dy_stride0 = dy_2d.stride(0)
    x_stride0 = x_2d.stride(0)
    a_batched = torch.as_strided(
        dy_2d, (M_wgrad, chunk_K, NUM_SPLITS),
        (1, dy_stride0, chunk_K * dy_stride0))
    b_batched = torch.as_strided(
        x_2d, (N_wgrad, chunk_K, NUM_SPLITS),
        (1, x_stride0, chunk_K * x_stride0))

    cache_key = (NUM_SPLITS, M_wgrad, N_wgrad, x.device.index)
    cached = _wgrad_partials_cache.get(cache_key)
    if cached is None:
        partials_flat = torch.empty(NUM_SPLITS, M_wgrad, N_wgrad,
                                    dtype=torch.float32, device=x.device)
        partials = partials_flat.permute(1, 2, 0)
        cached = (partials_flat, partials)
        _wgrad_partials_cache[cache_key] = cached
    partials_flat, partials = cached

    # Stream-overlap dgrad and wgrad: they're independent kernels.
    # Both kernels can use the persistent variant; the side stream runs
    # wgrad concurrently with dgrad on the main stream, then the main
    # stream waits at the end. Overlap is on by default; set
    # QKV_BWD_OVERLAP=0 for the serial (debug) path.
    dg_cfg, dg_key, dg_use_p = _persistent_state("dgrad")
    wg_cfg, wg_key, wg_use_p = _persistent_state("wgrad")

    if _use_overlap():
        # Round 70: cached current_stream lookup. Saves ~1.9us per bwd
        # (1.94us → 0.14us) when the caller hasn't switched streams.
        main_stream = _get_main_stream(x.device)
        side_stream = _get_wgrad_stream(x.device.index)

        # Round 42: pre-allocate d_weight on the *main* stream's allocator
        # context. If we let the C++ reducer call `torch::empty` from inside
        # `with torch.cuda.stream(side_stream)`, the caching allocator's
        # cross-stream bookkeeping doubles the bwd latency (~0.36 ms ->
        # ~0.72 ms). Pre-allocating sidesteps this entirely.
        d_weight = torch.empty(M_wgrad, N_wgrad, dtype=torch.float32,
                               device=x.device)

        # Side stream must wait for any prior producer of d_output / x on
        # the main stream (caller side, rare in our flow but cheap).
        side_stream.wait_stream(main_stream)

        # dgrad on main stream (persistent if requested, else non-persistent)
        if dg_use_p:
            _run_gemm_persistent(
                dg_key,
                dy_2d, w.T, d_input, BF16, BF16, BF16,
                b_stable=True,  # w.T view shares storage with weight
                **dg_cfg,
            )
        else:
            _run_gemm("qkv_dgrad_cm_256", dy_2d, w.T, d_input,
                      BF16, BF16, BF16, tile_mn=(128, 256), b_stable=True)

        # wgrad + reduce on side stream (writes into pre-allocated d_weight).
        # Round 69: replaced `with torch.cuda.stream(side_stream):` (which
        # costs ~7us per call from save-current-stream + try/finally) with
        # explicit `torch.cuda.set_stream(side_stream)` + restore. Since we
        # already know `main_stream` is the target to restore to (no need to
        # query it again), the round-trip is ~1us instead of ~7us. Net
        # savings: ~6us per bwd call. The wrapped block is small and pure
        # PyTorch / C++ — no Python exceptions inside; an explicit
        # try/finally provides matching error semantics.
        torch.cuda.set_stream(side_stream)
        try:
            if wg_use_p:
                _run_gemm_persistent(
                    wg_key,
                    a_batched, b_batched, partials, BF16, BF16, FP32,
                    c_stable=True,  # partials buffer is cached across calls
                    **wg_cfg,
                )
            else:
                wgrad_tile = _np_wgrad_tile()
                _run_gemm(
                    f"qkv_wgrad_np_{wgrad_tile[0]}x{wgrad_tile[1]}",
                    a_batched, b_batched, partials, BF16, BF16, FP32,
                    tile_mn=wgrad_tile,
                    c_stable=True,
                )
            _reduce_split_partials(
                partials_flat, M_wgrad, N_wgrad, NUM_SPLITS, out=d_weight
            )
            # The d_weight buffer was allocated on the main stream; tag it
            # with the side stream so the caching allocator preserves
            # ordering when the buffer is freed later.
            d_weight.record_stream(side_stream)
        finally:
            torch.cuda.set_stream(main_stream)

        # Main stream waits for wgrad to finish before returning d_weight
        main_stream.wait_stream(side_stream)
        return d_input.view_as(x), d_weight

    # ---- non-overlap path (serial) ----
    if dg_use_p:
        _run_gemm_persistent(
            dg_key,
            dy_2d, w.T, d_input, BF16, BF16, BF16,
            b_stable=True,
            **dg_cfg,
        )
    else:
        _run_gemm("qkv_dgrad_cm_256", dy_2d, w.T, d_input, BF16, BF16, BF16,
                  tile_mn=(128, 256), b_stable=True)

    if wg_use_p:
        _run_gemm_persistent(
            wg_key,
            a_batched, b_batched, partials, BF16, BF16, FP32,
            c_stable=True,
            **wg_cfg,
        )
    else:
        wgrad_tile = _np_wgrad_tile()
        _run_gemm(
            f"qkv_wgrad_np_{wgrad_tile[0]}x{wgrad_tile[1]}",
            a_batched, b_batched, partials, BF16, BF16, FP32,
            tile_mn=wgrad_tile,
            c_stable=True,
        )

    # See overlap-path comment above.
    d_weight = _reduce_split_partials(
        partials_flat, M_wgrad, N_wgrad, NUM_SPLITS
    )

    return d_input.view_as(x), d_weight
