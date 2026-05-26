"""CuTeDSL SM90 WGMMA+TMA kernel for gemm_fc2 (MLP down projection).

Three directions:
  fwd:   Y[M,N] = X[M,K] @ W^T      M=40960 N=1024 K=4096  BF16→BF16
  dgrad: dX[M,N] = dY[M,K] @ W      M=40960 N=4096 K=1024  BF16→BF16
  wgrad: dW[M,N] = dY^T[M,K] @ X    M=1024  N=4096 K=40960 BF16→FP32

Based on CuTeDSL dense_gemm_persistent.py (CUTLASS 4.4.2), stripped to BF16+FP32.
Uses warp specialization: dedicated DMA warp for TMA, MMA warps for WGMMA.
"""

from __future__ import annotations

__all__ = ["gemm_fc2_fwd", "gemm_fc2_bwd"]

import fcntl
import hashlib
import math
import os
import subprocess
import sys
import time
import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
from cutlass.cute.nvgpu.common import CopyUniversalOp

from training_engine_tensor.op_dispatcher import get_build_env as _get_build_env
from training_engine_tensor.op_dispatcher import get_frozen_env as _get_frozen_env

BF16 = cutlass.BFloat16
FP32 = cutlass.Float32

_FC2_BACKEND = _get_frozen_env().get("GEMM_FC2_BACKEND", "cutedsl")
_USE_INHOUSE = _FC2_BACKEND == "inhouse"
_USE_INHOUSE_JIT = _FC2_BACKEND == "inhouse_jit"
_inhouse_logged_fc2 = False

# ---------------------------------------------------------------------------
#  inhouse persistent AOT backend (self-contained export + load)
# ---------------------------------------------------------------------------
_inhouse_ext = None
_inhouse_ext_load_failed = False

_INHOUSE_EXPORT_DIR = os.path.join(
    _get_build_env().cutedsl_cache_root,
    "inhouse_aot_fc2",
)
_INHOUSE_LOCK_PATH = os.path.join(_INHOUSE_EXPORT_DIR, ".export.lock")
_INHOUSE_DIRECTIONS = ("inhouse_fc2_fwd", "inhouse_fc2_dgrad", "inhouse_fc2_wgrad")
_INHOUSE_MIN_OBJ_SIZE = 4096


def _try_inhouse_export():
    """Run export_inhouse.py if .h/.o files don't exist or script has changed."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(src_dir, "export_inhouse.py")
    if not os.path.exists(export_script):
        return False

    os.makedirs(_INHOUSE_EXPORT_DIR, exist_ok=True)

    needed = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.o") for d in _INHOUSE_DIRECTIONS]
    headers = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.h") for d in _INHOUSE_DIRECTIONS]
    all_files = needed + headers

    with open(export_script, "rb") as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    config_hash_path = os.path.join(_INHOUSE_EXPORT_DIR, ".config_hash")

    def _files_valid():
        for p in all_files:
            if not os.path.exists(p):
                return False
        for p in needed:
            if os.path.getsize(p) < _INHOUSE_MIN_OBJ_SIZE:
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

    rank = _get_build_env().local_rank

    if rank == 0:
        lock_fd = open(_INHOUSE_LOCK_PATH, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if _hash_matches() and _files_valid():
                return True
            result = subprocess.run(
                [sys.executable, export_script],
                cwd=src_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print(f"[gemm_fc2] inhouse export failed (rc={result.returncode}): "
                      f"{result.stderr[:500]}", file=sys.stderr)
                return False
            if not _files_valid():
                print("[gemm_fc2] inhouse export produced incomplete files",
                      file=sys.stderr)
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"[gemm_fc2] inhouse export error: {e}", file=sys.stderr)
            return False
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    else:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if _hash_matches() and _files_valid():
                return True
            time.sleep(2)
        print(f"[gemm_fc2] inhouse export: rank {rank} timed out waiting for rank 0",
              file=sys.stderr)
        return False


def _load_inhouse_ext():
    """Load inhouse AOT C++ extension linking exported .o files."""
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(src_dir, "gemm_inhouse.cpp")

    obj_files = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.o") for d in _INHOUSE_DIRECTIONS]
    header_files = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.h") for d in _INHOUSE_DIRECTIONS]

    h = hashlib.md5()
    for p in header_files + obj_files:
        with open(p, "rb") as f:
            h.update(f.read())
    build_hash = h.hexdigest()[:8]
    ext_name = f"gemm_fc2_inhouse_{build_hash}"

    cuda_home = _get_build_env().cuda_home

    include_dirs = [
        os.path.join(_INHOUSE_EXPORT_DIR, d) for d in _INHOUSE_DIRECTIONS
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
        cutlass_install_dir = _get_build_env().cutlass_install_dir or None
        if cutlass_install_dir:
            alt = os.path.join(cutlass_install_dir,
                               "lib", "libcuda_dialect_runtime_static.a")
            if os.path.exists(alt):
                runtime_lib = alt

    ext = load(
        name=ext_name,
        sources=[cpp_src],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3"],
        extra_ldflags=obj_files + [
            "-Wl,--whole-archive", runtime_lib, "-Wl,--no-whole-archive",
            "-lcuda", "-lcudart",
            "-L" + os.path.join(cuda_home, "lib64"),
        ],
        verbose=True,
    )
    return ext


def _get_inhouse_ext():
    """Lazy load inhouse AOT extension; cache success/failure."""
    global _inhouse_ext, _inhouse_ext_load_failed
    if _inhouse_ext is not None:
        return _inhouse_ext
    if _inhouse_ext_load_failed:
        return None
    rank = _get_build_env().local_rank
    try:
        if not _try_inhouse_export():
            _inhouse_ext_load_failed = True
            return None
        _inhouse_ext = _load_inhouse_ext()
        if rank == 0:
            print("[gemm_fc2] loaded inhouse persistent AOT backend", flush=True)
        return _inhouse_ext
    except Exception as exc:  # noqa: BLE001
        _inhouse_ext_load_failed = True
        if rank == 0:
            print(f"[gemm_fc2] inhouse AOT load failed: {exc}",
                  file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------------
#  AOT C-export backend
# ---------------------------------------------------------------------------
_aot_ext = None
_aot_ext_loaded = False
# See workload/ops/gemm_fc1/kernel.py: same shared-cache convention so a
# one-shot prebuild populates the AOT export and every training pod
# reuses it instead of paying the JIT cost again.
EXPORT_DIR = os.path.join(
    _get_build_env().cutedsl_cache_root,
    "cutedsl_export_gemm_fc2",
)
_LOCK_PATH = os.path.join(EXPORT_DIR, ".export.lock")
_AOT_DIRECTIONS = ("gemm_fwd", "gemm_dgrad", "gemm_wgrad")
_MIN_OBJ_SIZE = 4096


def _try_cutedsl_export():
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
    rank = _get_build_env().local_rank
    if rank == 0:
        lock_fd = open(_LOCK_PATH, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if _hash_matches() and _files_valid():
                return True
            result = subprocess.run(
                [sys.executable, export_script], cwd=src_dir,
                capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"[gemm_fc2] export failed: {result.stderr[:500]}", file=sys.stderr)
                return False
            if not _files_valid():
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"[gemm_fc2] export error: {e}", file=sys.stderr)
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
        return False


def _load_cutedsl_ext():
    from torch.utils.cpp_extension import load
    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(src_dir, "gemm_cutedsl.cpp")
    obj_files = [os.path.join(EXPORT_DIR, d, f"{d}.o") for d in _AOT_DIRECTIONS]
    header_files = [os.path.join(EXPORT_DIR, d, f"{d}.h") for d in _AOT_DIRECTIONS]
    h = hashlib.md5()
    for p in header_files + obj_files:
        with open(p, "rb") as f:
            h.update(f.read())
    ext_name = f"gemm_fc2_cutedsl_{h.hexdigest()[:8]}"
    cuda_home = _get_build_env().cuda_home
    include_dirs = [os.path.join(EXPORT_DIR, d) for d in _AOT_DIRECTIONS] + \
                   [os.path.join(cuda_home, "include")]
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
        cutlass_install_dir = _get_build_env().cutlass_install_dir or None
        if cutlass_install_dir:
            alt = os.path.join(cutlass_install_dir,
                               "lib", "libcuda_dialect_runtime_static.a")
            if os.path.exists(alt):
                runtime_lib = alt
    return load(
        name=ext_name, sources=[cpp_src],
        extra_include_paths=include_dirs, extra_cflags=["-O3"],
        extra_ldflags=obj_files + [
            "-Wl,--whole-archive", runtime_lib, "-Wl,--no-whole-archive",
            "-lcuda", "-lcudart", "-L" + os.path.join(cuda_home, "lib64"),
        ], verbose=True)


def _get_aot_ext():
    global _aot_ext, _aot_ext_loaded
    if _aot_ext_loaded:
        return _aot_ext
    _aot_ext_loaded = True
    if _get_frozen_env().get("FC2_AOT", "1") == "0":
        return None
    rank = _get_build_env().local_rank
    try:
        if _try_cutedsl_export():
            _aot_ext = _load_cutedsl_ext()
            if rank == 0:
                print("[gemm_fc2] loaded AOT CuTeDSL C++ backend", flush=True)
    except Exception as e:
        if rank == 0:
            print(f"[gemm_fc2] AOT failed, falling back to JIT: {e}",
                  file=sys.stderr, flush=True)
        _aot_ext = None
    return _aot_ext


class _HopperGemm:
    """Hopper WGMMA+TMA GEMM for BF16 input, FP32 accumulation.

    Supports BF16 or FP32 output (determined by c tensor element type).
    Adapted from CuTeDSL dense_gemm.py, single-batch only.
    """

    def __init__(self, acc_dtype, tile_shape_mn, cluster_shape_mn=(1, 1),
                 force_atom_layout=None, occupancy=1):
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
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = occupancy
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
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
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
        k_pipe_mmas = 1

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


class _HopperGemmPersistent:
    """Persistent Hopper WGMMA+TMA GEMM with warp specialization.

    DMA warp group handles TMA loads while MMA warp group runs WGMMA,
    enabling overlap for higher throughput. CTAs stay resident and process
    multiple tiles via StaticPersistentTileScheduler.
    """

    def __init__(self, acc_dtype, tile_shape_mn, cluster_shape_mn=(1, 1),
                 swizzle_size=1, raster_along_m=True, k_pipe_mmas=1,
                 mma_inst_tile_k=4):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.k_pipe_mmas = k_pipe_mmas
        self.mma_inst_tile_k = mma_inst_tile_k
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
        self.epi_stage = 4
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

        # --- DMA warp group: TMA loads ---
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

        # --- MMA warp group: WGMMA + epilogue ---
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

            k_pipe_mmas = self.k_pipe_mmas
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

                # Mainloop
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

                # Epilogue
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


def _to_cute_3d(t: torch.Tensor, cutlass_dtype) -> cute.Tensor:
    """Convert a 2D PyTorch tensor to a 3D CuTe tensor (L=1 batch).

    Supports row-major (contiguous) and column-major (transposed) layouts.
    Row-major: stride(1)==1, leading_dim=1.
    Col-major: stride(0)==1, leading_dim=0.
    """
    assert t.ndim == 2, f"Expected 2D, got shape={t.shape}"
    t3d = t.unsqueeze(-1)
    ct = from_dlpack(t3d, assumed_align=16)
    ct.element_type = cutlass_dtype
    leading_dim = 1 if t.stride(1) == 1 else 0
    ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def _get_stream():
    torch_stream = torch.cuda.current_stream()
    return cuda.CUstream(torch_stream.cuda_stream)


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
):
    """Run a GEMM using a cached compiled kernel."""
    stream = _get_stream()
    mA = _to_cute_3d(a_torch, a_cutlass_dtype)
    mB = _to_cute_3d(b_torch, b_cutlass_dtype)
    mC = _to_cute_3d(c_torch, c_cutlass_dtype)

    if key not in _compiled_cache:
        gemm = _HopperGemm(FP32, tile_mn, cluster_shape_mn=cluster_mn,
                            force_atom_layout=force_atom_layout,
                            occupancy=occupancy)
        compiled = cute.compile(gemm, mA, mB, mC, stream)
        _compiled_cache[key] = compiled

    _compiled_cache[key](mA, mB, mC, stream)


_H100_SM_COUNT = 132


def _run_gemm_persistent(
    key: str,
    a_torch: torch.Tensor,
    b_torch: torch.Tensor,
    c_torch: torch.Tensor,
    a_cutlass_dtype,
    b_cutlass_dtype,
    c_cutlass_dtype,
    tile_mn: tuple[int, int] = (128, 128),
    cluster_mn: tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    k_pipe_mmas: int = 1,
    mma_inst_tile_k: int = 4,
):
    """Run a persistent GEMM with warp-specialization (DMA+MMA overlap)."""
    stream = _get_stream()
    mA = _to_cute_3d(a_torch, a_cutlass_dtype)
    mB = _to_cute_3d(b_torch, b_cutlass_dtype)
    mC = _to_cute_3d(c_torch, c_cutlass_dtype)

    if key not in _compiled_cache:
        gemm = _HopperGemmPersistent(
            FP32, tile_mn, cluster_shape_mn=cluster_mn,
            swizzle_size=swizzle_size, raster_along_m=raster_along_m,
            k_pipe_mmas=k_pipe_mmas, mma_inst_tile_k=mma_inst_tile_k,
        )
        compiled = cute.compile(
            gemm, mA, mB, mC, _H100_SM_COUNT, stream,
        )
        _compiled_cache[key] = compiled

    _compiled_cache[key](mA, mB, mC, _H100_SM_COUNT, stream)


# ---------------------------------------------------------------------------
#  High-performance BF16 matrix transpose via shared memory
# ---------------------------------------------------------------------------
_transpose_module = None


def _get_transpose_fn():
    global _transpose_module
    if _transpose_module is not None:
        return _transpose_module.transpose_bf16

    from torch.utils.cpp_extension import load_inline

    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define TILE 32
#define BLOCK_ROWS 8

__global__ void transpose_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int rows, int cols)
{
    __shared__ __nv_bfloat16 tile[TILE][TILE + 1];

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;

    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        int r = by + threadIdx.y + j;
        int c = bx + threadIdx.x;
        if (r < rows && c < cols)
            tile[threadIdx.y + j][threadIdx.x] = in[r * cols + c];
    }

    __syncthreads();

    int bx2 = blockIdx.y * TILE;
    int by2 = blockIdx.x * TILE;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        int r = by2 + threadIdx.y + j;
        int c = bx2 + threadIdx.x;
        if (r < cols && c < rows)
            out[r * rows + c] = tile[threadIdx.x][threadIdx.y + j];
    }
}

torch::Tensor transpose_bf16(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Expected 2D tensor");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "Expected BFloat16");
    TORCH_CHECK(input.is_contiguous(), "Expected contiguous");

    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty({cols, rows}, input.options());

    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    dim3 block(TILE, BLOCK_ROWS);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    transpose_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        rows, cols);

    return output;
}
"""

    cpp_src = "torch::Tensor transpose_bf16(torch::Tensor input);"

    _transpose_module = load_inline(
        name="fast_transpose_bf16",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["transpose_bf16"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return _transpose_module.transpose_bf16


# ---------------------------------------------------------------------------
#  Async helper: overlap wgrad transposes with dgrad GEMM
# ---------------------------------------------------------------------------
_wgrad_stream: torch.cuda.Stream | None = None


def _get_wgrad_stream():
    global _wgrad_stream
    if _wgrad_stream is None:
        _wgrad_stream = torch.cuda.Stream()
    return _wgrad_stream


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def gemm_fc2_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Forward: out = x @ weight.T

    x:      [S, B, 4096] BF16
    weight: [1024, 4096] BF16
    Returns [S, B, 1024] BF16.

    CUTLASS convention: D[m,n] = A[m,k] * B[n,k]
    A = X (M×K, K-major), B = W (N×K, K-major), C = Y (M×N, N-major)
    """
    global _inhouse_logged_fc2
    if _USE_INHOUSE_JIT:
        if not _inhouse_logged_fc2:
            print("[gemm_fc2] using inhouse JIT for fwd/dgrad/wgrad")
            _inhouse_logged_fc2 = True
        from training_engine_tensor.ops._gemm_inhouse_jit import jit_gemm_fc2_fwd
        shape = x.shape
        x_2d = x.contiguous().reshape(-1, shape[-1])
        w = weight.contiguous()
        M, K = x_2d.shape
        N = w.shape[0]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        jit_gemm_fc2_fwd(x_2d, w, out)
        return out.view(*shape[:-1], N)

    ext = _get_aot_ext()
    if ext is not None:
        return ext.gemm_fwd_fast(x, weight)

    if _USE_INHOUSE:
        inhouse = _get_inhouse_ext()
        if inhouse is None:
            raise RuntimeError("[gemm_fc2] GEMM_FC2_BACKEND=inhouse but inhouse AOT "
                               "extension failed to load")
        if not _inhouse_logged_fc2:
            print("[gemm_fc2] using inhouse persistent AOT for fwd/dgrad")
            _inhouse_logged_fc2 = True
        shape = x.shape
        x_2d = x.contiguous().reshape(-1, shape[-1])
        w = weight.contiguous()
        M, K = x_2d.shape
        N = w.shape[0]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        inhouse.gemm_fwd_fast(x_2d, w, out)
        return out.view(*shape[:-1], N)

    shape = x.shape
    x_2d = x.contiguous().reshape(-1, shape[-1])
    w = weight.contiguous()
    M, K = x_2d.shape
    N = w.shape[0]

    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    _run_gemm_persistent("fwd_c12_rN_sw4", x_2d, w, out, BF16, BF16, BF16,
                         tile_mn=(128, 256), cluster_mn=(1, 2),
                         swizzle_size=4, raster_along_m=False)  # R-autotune

    return out.view(*shape[:-1], N)


def gemm_fc2_bwd(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward: returns (d_input, d_weight_fp32).

    d_output: [S, B, 1024] BF16
    x:        [S, B, 4096] BF16
    weight:   [1024, 4096] BF16
    Returns:  d_input [S, B, 4096] BF16, d_weight [1024, 4096] FP32.

    dgrad: dX = dY @ W.  CuTeDSL: C[M,N]=A[M,K]*B[N,K]
           A=dY[40960,1024], B=W^T[4096,1024] (col-major view), C=dX[40960,4096]
    wgrad: dW = dY^T @ X.  CuTeDSL: C[M,N]=A[M,K]*B[N,K]
           A=dY^T[1024,40960] (col-major view), B=X^T[4096,40960] (col-major view),
           C=dW[1024,4096] FP32

    Zero-copy: col-major views avoid explicit transpose memory copies.
    """
    global _inhouse_logged_fc2
    if _USE_INHOUSE_JIT:
        from training_engine_tensor.ops._gemm_inhouse_jit import (
            jit_gemm_fc2_dgrad, jit_gemm_fc2_wgrad)
        shape_dy = d_output.shape
        shape_x = x.shape
        dy_2d = d_output.contiguous().reshape(-1, shape_dy[-1])
        x_2d = x.contiguous().reshape(-1, shape_x[-1])
        w = weight.contiguous()
        M_dgrad = dy_2d.shape[0]
        N_dgrad = w.shape[1]
        d_input = torch.empty(M_dgrad, N_dgrad, dtype=torch.bfloat16, device=x.device)
        jit_gemm_fc2_dgrad(dy_2d, w, d_input)
        M_w, N_w = w.shape
        d_weight = torch.empty(M_w, N_w, dtype=torch.float32, device=x.device)
        jit_gemm_fc2_wgrad(dy_2d, x_2d, d_weight)
        return d_input.view_as(x), d_weight

    ext = _get_aot_ext()
    if ext is not None:
        if _USE_INHOUSE:
            inhouse = _get_inhouse_ext()
            if inhouse is None:
                raise RuntimeError("[gemm_fc2] GEMM_FC2_BACKEND=inhouse but inhouse AOT "
                                   "extension failed to load")
            if not _inhouse_logged_fc2:
                print("[gemm_fc2] using inhouse persistent AOT for fwd/dgrad/wgrad")
                _inhouse_logged_fc2 = True
            shape_dy = d_output.shape
            shape_x = x.shape
            dy_2d = d_output.contiguous().reshape(-1, shape_dy[-1])
            x_2d = x.contiguous().reshape(-1, shape_x[-1])
            w = weight.contiguous()
            M_dgrad = dy_2d.shape[0]
            N_dgrad = w.shape[1]
            d_input = torch.empty(M_dgrad, N_dgrad, dtype=torch.bfloat16, device=x.device)
            inhouse.gemm_dgrad_fast(dy_2d, w, d_input)
            M_w, N_w = w.shape
            d_weight = torch.empty(M_w, N_w, dtype=torch.float32, device=x.device)
            inhouse.gemm_wgrad(dy_2d, x_2d, d_weight)
            return d_input.view_as(x), d_weight
        d_input = ext.gemm_dgrad_fast(d_output, weight)
        d_weight = ext.gemm_wgrad_fast(d_output, x)
        return d_input, d_weight

    shape_dy = d_output.shape
    shape_x = x.shape

    dy_2d = d_output.contiguous().reshape(-1, shape_dy[-1])
    x_2d = x.contiguous().reshape(-1, shape_x[-1])
    w = weight.contiguous()

    M_dgrad = dy_2d.shape[0]
    N_dgrad = w.shape[1]

    if _USE_INHOUSE:
        inhouse = _get_inhouse_ext()
        if inhouse is None:
            raise RuntimeError("[gemm_fc2] GEMM_FC2_BACKEND=inhouse but inhouse AOT "
                               "extension failed to load")
        if not _inhouse_logged_fc2:
            print("[gemm_fc2] using inhouse persistent AOT for fwd/dgrad/wgrad")
            _inhouse_logged_fc2 = True
        d_input = torch.empty(M_dgrad, N_dgrad, dtype=torch.bfloat16, device=x.device)
        inhouse.gemm_dgrad_fast(dy_2d, w, d_input)

        M_w, N_w = w.shape
        d_weight = torch.empty(M_w, N_w, dtype=torch.float32, device=x.device)
        inhouse.gemm_wgrad(dy_2d, x_2d, d_weight)
    elif False:  # dead code — preserved for reference
        _run_gemm_persistent("wgrad_fp32_c21_sw2_rM_ep4", None, None, None, BF16, BF16, FP32,
                             tile_mn=(128, 256), cluster_mn=(2, 1),
                             swizzle_size=2, raster_along_m=True)
    else:
        wt_view = w.t()
        d_input = torch.empty(M_dgrad, N_dgrad, dtype=torch.bfloat16, device=x.device)
        _run_gemm_persistent("dgrad_colB_c21_rN_sw8", dy_2d, wt_view, d_input,
                             BF16, BF16, BF16,
                             tile_mn=(128, 256), cluster_mn=(2, 1),
                             swizzle_size=8, raster_along_m=False)

        a_wgrad = dy_2d.t()
        b_wgrad = x_2d.t()
        d_weight = torch.empty(a_wgrad.shape[0], b_wgrad.shape[0],
                               dtype=torch.float32, device=x.device)
        _run_gemm_persistent("wgrad_fp32_c21_sw2_rM_ep4", a_wgrad, b_wgrad, d_weight, BF16, BF16, FP32,
                             tile_mn=(128, 256), cluster_mn=(2, 1),
                             swizzle_size=2, raster_along_m=True)

    return d_input.view_as(x), d_weight
