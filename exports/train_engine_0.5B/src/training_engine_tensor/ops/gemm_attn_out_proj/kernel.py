"""CuTeDSL SM90 WGMMA+TMA kernel for gemm_attn_out_proj (attention output projection).

Three directions:
  fwd:   Y[M,N] = X[M,K] @ W^T      M=40960 N=1024 K=1024  BF16→BF16
  dgrad: dX[M,N] = dY[M,K] @ W      M=40960 N=1024 K=1024  BF16→BF16
  wgrad: dW[M,N] = dY^T[M,K] @ X    M=1024  N=1024 K=40960 BF16→FP32

fwd/dgrad: non-persistent WGMMA+TMA kernel, tile (128,256).
wgrad: in-house AOT persistent kernel, tile (64,128) L=1 FP32 output.
bwd: dgrad+wgrad serial (overlap not beneficial).
"""

# NOTE: do NOT use `from __future__ import annotations` here.
# CuTeDSL's c_header_generator reads annotations as raw values to detect
# `cutlass.Constexpr` parameters; PEP 563 lazy strings break that detection
# and cause `Unsupported argument for c function argument generation` during
# `compiled.export_to_c(...)`.

import fcntl
import hashlib
import math
import os
import subprocess
import sys
import time

import torch
import cuda.bindings.driver as cuda

# CuTeDSL PYTHONPATH fallback — when nvidia_cutlass_dsl is laid out
# under a shared-FS directory rather than pip-installed under
# site-packages, point CUTLASS_DSL_FALLBACK_DIR at the package root
# (the parent of ``python_packages/`` and ``lib/``).
_fb = os.environ.get("CUTLASS_DSL_FALLBACK_DIR", "")
if _fb:
    _pkg = os.path.join(_fb, "python_packages")
    if os.path.isdir(_pkg) and _pkg not in sys.path:
        sys.path.insert(0, _pkg)

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline

BF16 = cutlass.BFloat16
FP32 = cutlass.Float32

_AOP_BACKEND = os.environ.get("AOP_BACKEND", "")
_USE_INHOUSE_AOP = _AOP_BACKEND == "inhouse"
_USE_INHOUSE_JIT_AOP = _AOP_BACKEND == "inhouse_jit"
_inhouse_logged_aop = False

# ---------------------------------------------------------------------------
#  inhouse persistent AOT backend (self-contained export + load)
# ---------------------------------------------------------------------------
_inhouse_ext_aop = None
_inhouse_ext_aop_load_failed = False

_INHOUSE_EXPORT_DIR = os.path.join(
    os.environ.get("CUTEDSL_CACHE_ROOT", "/tmp"),
    "inhouse_aot_aop",
)
_INHOUSE_LOCK_PATH_AOP = os.path.join(_INHOUSE_EXPORT_DIR, ".export.lock")
_INHOUSE_DIRECTIONS_AOP = ("inhouse_aop_fwd", "inhouse_aop_dgrad")
_INHOUSE_MIN_OBJ_SIZE = 4096


def _try_inhouse_export():
    """Run export_inhouse.py if .h/.o files don't exist or script has changed."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(src_dir, "export_inhouse.py")
    if not os.path.exists(export_script):
        return False

    os.makedirs(_INHOUSE_EXPORT_DIR, exist_ok=True)

    needed = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.o") for d in _INHOUSE_DIRECTIONS_AOP]
    headers = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.h") for d in _INHOUSE_DIRECTIONS_AOP]
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

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    if rank == 0:
        lock_fd = open(_INHOUSE_LOCK_PATH_AOP, "w")
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
                print(f"[gemm_attn_out_proj] inhouse export failed (rc={result.returncode})")
                print(f"  stdout: {result.stdout[-500:]}")
                print(f"  stderr: {result.stderr[-500:]}")
                return False
            if not _files_valid():
                print("[gemm_attn_out_proj] inhouse export produced incomplete files",
                      file=sys.stderr)
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"[gemm_attn_out_proj] inhouse export error: {e}",
                  file=sys.stderr)
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
        print(f"[gemm_attn_out_proj] inhouse export: rank {rank} timed out "
              "waiting for rank 0", file=sys.stderr)
        return False


def _load_inhouse_ext():
    """Load inhouse AOT C++ extension linking exported .o files."""
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(src_dir, "gemm_inhouse.cpp")

    obj_files = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.o") for d in _INHOUSE_DIRECTIONS_AOP]
    header_files = [os.path.join(_INHOUSE_EXPORT_DIR, d, f"{d}.h") for d in _INHOUSE_DIRECTIONS_AOP]

    h = hashlib.md5()
    for p in header_files + obj_files:
        with open(p, "rb") as f:
            h.update(f.read())
    build_hash = h.hexdigest()[:8]
    ext_name = f"gemm_aop_inhouse_{build_hash}"

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    include_dirs = [
        os.path.join(_INHOUSE_EXPORT_DIR, d) for d in _INHOUSE_DIRECTIONS_AOP
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
        _fb_dir2 = os.environ.get("CUTLASS_DSL_FALLBACK_DIR", "")
        if _fb_dir2:
            _candidates.append(os.path.join(_fb_dir2, "lib",
                                            "libcuda_dialect_runtime_static.a"))
        for cand in _candidates:
            if os.path.exists(cand):
                runtime_lib = cand
                break

    cuda_lib64 = os.path.join(cuda_home, "lib64")
    ext = load(
        name=ext_name,
        sources=[cpp_src],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3"],
        extra_ldflags=obj_files + [
            "-Wl,--whole-archive", runtime_lib, "-Wl,--no-whole-archive",
            "-lcuda", "-lcudart",
            "-L" + cuda_lib64,
            "-Wl,-rpath," + cuda_lib64,
        ],
        verbose=True,
    )
    return ext


def _get_inhouse_ext_aop():
    """Lazy load inhouse AOT extension; cache success/failure."""
    global _inhouse_ext_aop, _inhouse_ext_aop_load_failed
    if _inhouse_ext_aop is not None:
        return _inhouse_ext_aop
    if _inhouse_ext_aop_load_failed:
        return None
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    try:
        if not _try_inhouse_export():
            _inhouse_ext_aop_load_failed = True
            return None
        _inhouse_ext_aop = _load_inhouse_ext()
        if rank == 0:
            print("[gemm_attn_out_proj] loaded inhouse persistent AOT backend",
                  flush=True)
        return _inhouse_ext_aop
    except Exception as exc:  # noqa: BLE001
        _inhouse_ext_aop_load_failed = True
        if rank == 0:
            print(f"[gemm_attn_out_proj] inhouse AOT load failed: {exc}",
                  file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------------
#  Compat helpers for CuTeDSL 4.2.1 (missing in older versions)
# ---------------------------------------------------------------------------

def _compute_epi_tile(tile_shape_mnk, c_dtype, is_cooperative):
    """compute_tile_shape_or_override equivalent."""
    if is_cooperative:
        tile_m = min(128, tile_shape_mnk[0])
        tile_n = min(32, tile_shape_mnk[1])
    else:
        n_perf = 64 if c_dtype.width == 8 else 32
        tile_m = min(64, tile_shape_mnk[0])
        tile_n = min(n_perf, tile_shape_mnk[1])
    return (tile_m, tile_n)


def _make_smem_layout_atom_from_kind(kind, element_type):
    """Inline of make_smem_layout_atom from warpgroup helpers."""
    from cutlass.cute.nvgpu.warpgroup.mma import SmemLayoutAtomKind
    if kind in (SmemLayoutAtomKind.MN_INTER, SmemLayoutAtomKind.K_INTER):
        num_contiguous_bits = 128
        sw = cute.make_swizzle(0, 4, 3)
    elif kind in (SmemLayoutAtomKind.MN_SW32, SmemLayoutAtomKind.K_SW32):
        num_contiguous_bits = 256
        sw = cute.make_swizzle(1, 4, 3)
    elif kind in (SmemLayoutAtomKind.MN_SW64, SmemLayoutAtomKind.K_SW64):
        num_contiguous_bits = 512
        sw = cute.make_swizzle(2, 4, 3)
    elif kind in (SmemLayoutAtomKind.MN_SW128, SmemLayoutAtomKind.K_SW128):
        num_contiguous_bits = 1024
        sw = cute.make_swizzle(3, 4, 3)
    else:
        raise ValueError(f"unrecognized SMEM layout atom kind: {kind}")
    num_contiguous_elems = num_contiguous_bits // element_type.width

    mn_kinds = (SmemLayoutAtomKind.MN_INTER, SmemLayoutAtomKind.MN_SW32,
                SmemLayoutAtomKind.MN_SW64, SmemLayoutAtomKind.MN_SW128)
    if kind in mn_kinds:
        return cute.make_composed_layout(
            sw, 0,
            cute.make_layout(
                (num_contiguous_elems, 8), stride=(1, num_contiguous_elems)))
    else:
        return cute.make_composed_layout(
            sw, 0,
            cute.make_layout(
                (8, num_contiguous_elems), stride=(num_contiguous_elems, 1)))


def _is_k_major(layout):
    """Check if layout is K-major (row-major for A/B in CUTLASS convention)."""
    from cutlass.cute.nvgpu.warpgroup.mma import OperandMajorMode
    return layout.sm90_mma_major_mode() == OperandMajorMode.K


def _make_smem_layout_a(a_layout, mma_tiler_mnk, a_dtype, num_stages):
    """make_smem_layout_a equivalent for CuTeDSL 4.2.1."""
    a_smem_shape = cute.slice_(mma_tiler_mnk, (None, 0, None))
    k_major = _is_k_major(a_layout)
    a_major_mode_size = mma_tiler_mnk[2] if k_major else mma_tiler_mnk[0]

    kind = sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size)
    atom = _make_smem_layout_atom_from_kind(kind, a_dtype)

    return cute.tile_to_shape(
        atom, cute.append(a_smem_shape, num_stages),
        order=(0, 1, 2))


def _make_smem_layout_b(b_layout, mma_tiler_mnk, b_dtype, num_stages):
    """make_smem_layout_b equivalent for CuTeDSL 4.2.1."""
    b_smem_shape = cute.slice_(mma_tiler_mnk, (0, None, None))
    k_major = _is_k_major(b_layout)
    b_major_mode_size = mma_tiler_mnk[2] if k_major else mma_tiler_mnk[1]

    kind = sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size)
    atom = _make_smem_layout_atom_from_kind(kind, b_dtype)

    return cute.tile_to_shape(
        atom, cute.append(b_smem_shape, num_stages),
        order=((1, 0, 2) if not k_major else (0, 1, 2)))


def _make_smem_layout_epi(epi_dtype, epi_layout, epi_tile, epi_stage):
    """make_smem_layout_epi equivalent for CuTeDSL 4.2.1."""
    o_major_mode_size = epi_tile[1] if epi_layout.is_n_major_c() else epi_tile[0]

    kind = sm90_utils.get_smem_layout_atom(epi_layout, epi_dtype, o_major_mode_size)
    atom = _make_smem_layout_atom_from_kind(kind, epi_dtype)

    order = (1, 0, 2) if epi_layout.is_m_major_c() else (0, 1, 2)

    return cute.tile_to_shape(
        atom, cute.append(epi_tile, epi_stage), order)


class _HopperGemm:
    """Hopper WGMMA+TMA GEMM for BF16 input, FP32 accumulation.

    Supports BF16 or FP32 output (determined by c tensor element type).
    Adapted from CuTeDSL dense_gemm.py, single-batch only.
    """

    def __init__(self, acc_dtype, tile_shape_mn, cluster_shape_mn=(1, 1),
                 force_atom_layout=None, occupancy=1, k_pipe_mmas=1):
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
        self.epi_tile = _compute_epi_tile(
            self.tile_shape_mnk, self.c_dtype, is_cooperative
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
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
    ):
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
        accumulators = cute.make_fragment(acc_shape, self.acc_dtype)

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
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype)
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

            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)
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
        a_smem_layout_staged = _make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, ab_stage,
        )
        b_smem_layout_staged = _make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, ab_stage,
        )
        epi_smem_layout_staged = _make_smem_layout_epi(
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

    DMA warp group handles TMA loads, MMA warp group handles WGMMA + epilogue.
    CTAs persist across tiles, eliminating wave transition overhead.
    """

    def __init__(self, acc_dtype, tile_shape_mn, cluster_shape_mn=(1, 1),
                 swizzle_size=1, raster_along_m=True, mma_inst_tile_k=4,
                 occupancy=1, k_pipe_mmas=1, epi_stage=4):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.mma_inst_tile_k_param = mma_inst_tile_k
        self.k_pipe_mmas_param = k_pipe_mmas
        self.epi_stage_param = epi_stage
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

        # Round 36: occupancy>1 lets multiple CTAs share an SM, doubling
        # warp count per SM at the cost of pipeline depth (each CTA gets
        # smem_capacity//occupancy bytes for the AB pipeline).
        self.occupancy = occupancy
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
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
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.mma_inst_tile_k_param
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
        self.epi_tile = _compute_epi_tile(
            self.tile_shape_mnk, self.c_dtype, is_cooperative
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.epi_stage_param,
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
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
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

        tile_sched_params, grid = self._compute_grid(
            c, self.tile_shape_mnk, self.cluster_shape_mn,
            self.swizzle_size, self.raster_along_m, max_active_clusters,
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
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
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
        self,
        tma_atom_a, mA_mkl,
        tma_atom_b, mB_nkl,
        tma_atom_c, mC_mnl,
        tiled_mma,
        cta_layout_mnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        epi_smem_layout_staged,
        tile_sched_params,
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
        accumulators = cute.make_fragment(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups

        if is_dma_warp_group:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)

        # DMA warp group: TMA loads
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

        # MMA warp group: WGMMA + epilogue
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
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s, tiled_copy_C_Atom,
            )

            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype)
            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = self.k_pipe_mmas_param
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

                # MAINLOOP
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

                # EPILOGUE
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

    @staticmethod
    def _compute_stages(tile_shape_mnk, a_dtype, b_dtype, epi_tile, c_dtype,
                        smem_capacity, occupancy, epi_stage=4):
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage
        mbar_helpers_bytes = 1024
        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk, epi_tile,
        a_dtype, a_layout, b_dtype, b_layout, ab_stage,
        c_dtype, c_layout, epi_stage,
    ):
        return _HopperGemm._make_smem_layouts(
            tile_shape_mnk, epi_tile,
            a_dtype, a_layout, b_dtype, b_layout, ab_stage,
            c_dtype, c_layout, epi_stage,
        )

    @staticmethod
    def _compute_grid(c, tile_shape_mnk, cluster_shape_mn,
                      swizzle_size, raster_along_m, max_active_clusters):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl,
            swizzle_size, raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid


# ---------------------------------------------------------------------------
#  Compiled kernel cache
# ---------------------------------------------------------------------------
_compiled_cache: dict[str, object] = {}

def _to_cute_3d(t: torch.Tensor, cutlass_dtype) -> cute.Tensor:
    """Convert a 2D or 3D PyTorch tensor to a 3D CuTe tensor (M, K, L)."""
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    assert t.ndim == 3, f"Expected 2D or 3D, got shape={t.shape}"
    ct = from_dlpack(t, assumed_align=16)
    ct.element_type = cutlass_dtype
    leading_dim = 1 if t.stride(1) == 1 else 0
    ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


# Cache cute Tensors for long-lived torch tensors (e.g., model weights).
# Key: id(t); we also store data_ptr to detect aliasing/in-place ops are
# fine (in-place keeps data_ptr) but a new tensor object with reused id
# (rare, after GC) would have different data_ptr and we recompute.
_cute_tensor_cache: dict[int, tuple[int, object]] = {}


def _to_cute_3d_cached(t: torch.Tensor, cutlass_dtype) -> object:
    """Same as _to_cute_3d but caches the cute Tensor by id(t).

    Use only for tensors that are guaranteed to be long-lived AND
    referenced by the caller throughout the cached entry's lifetime
    (model weights). Adam in-place updates keep data_ptr stable.
    """
    key = id(t)
    cached = _cute_tensor_cache.get(key)
    dptr = t.data_ptr()
    if cached is not None and cached[0] == dptr:
        return cached[1]
    ct = _to_cute_3d(t, cutlass_dtype)
    _cute_tensor_cache[key] = (dptr, ct)
    return ct


def _to_cute_3d_T_cached(w: torch.Tensor, cutlass_dtype) -> object:
    """Cache cute Tensor for w.T keyed by id(w).

    For dgrad: B = w.T is built fresh per call but the underlying weight
    `w` is long-lived. Caching by id(w) avoids the per-call DLPack
    conversion of the transposed view (~5 us).
    """
    key = (id(w), id(cutlass_dtype), True)
    cached = _cute_tensor_cache.get(key)
    dptr = w.data_ptr()
    if cached is not None and cached[0] == dptr:
        return cached[1]
    ct = _to_cute_3d(w.T, cutlass_dtype)
    _cute_tensor_cache[key] = (dptr, ct)
    return ct


_stream_cache: dict[int, object] = {}
_USE_STREAM_CACHE = os.environ.get("AOP_NO_STREAM_CACHE", "0") != "1"


def _get_stream():
    """Cache the cute CUstream wrapper keyed by underlying torch stream ptr.

    `torch.cuda.current_stream()` is itself a small Python call returning a
    fresh Stream object each time, but the underlying `.cuda_stream` integer
    is stable for the lifetime of a stream. Building `cuda.CUstream(int)` is
    pure-Python wrapper construction (~1-2us) but adds up across thousands
    of GEMM calls per training step. AOP_NO_STREAM_CACHE=1 disables for A/B.
    """
    sptr = torch.cuda.current_stream().cuda_stream
    if _USE_STREAM_CACHE:
        cached = _stream_cache.get(sptr)
        if cached is not None:
            return cached
        s = cuda.CUstream(sptr)
        _stream_cache[sptr] = s
        return s
    return cuda.CUstream(sptr)


# Module-level scratch buffer for wgrad split-K partials.
# Safe to reuse across calls: filled (ACCUMULATE=False) by the GEMM, then
# reduced into the returned d_weight via .sum(dim=0). Always (2,1024,1024)
# FP32 for MiniCPM4 0.5B; key on (splits, M, N, device) for safety.
_wgrad_partials_cache: dict[tuple, torch.Tensor] = {}


# ----------------------------------------------------------------------
# Round 27+ tile/cluster sweep harness — env var overrides.
# Production defaults are the Round 27 sweep winners; setters used by
# bench_*_sweep.py to A/B compare configurations cheaply.
# ----------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default

# Round 34 sweep winner — explored PersistentTileScheduler swizzle_size {1,2,3,8}
# (never benched in rounds 1–33) on top of the Round 33-locked tiles.
# Single-axis ablation found dgrad & wgrad benefit but fwd is hurt by raster=N.
# Final cand_D (keep Round 33 fwd, swap dgrad+wgrad scheduler) brings paired-
# bench median total ratio 1.0577 → 1.0214 (σ ≈ 0.005, op-unit 6/6 bitwise
# identical). Net: -3.6 pp absolute, mostly from bwd dropping 1.058 → 1.011.
# - fwd:   sw=4 raster=M  (UNCHANGED from Round 33)
# - dgrad: sw=1 raster=N  (was sw=4 raster=M; ~ -1.3pp)
# - wgrad: cluster=(1,2) sw=1 raster=N  (was cluster=(2,1) sw=4 raster=N;
#          ~ -1.3pp; cluster (1,2) and (2,1) are mathematically symmetric for
#          the square 1024x1024 wgrad output, but (1,2) interacts better with
#          the new sw=1 swizzle pattern in the persistent scheduler).
_FWD_TILE_M   = _env_int("AOP_FWD_TILE_M",     128)
_FWD_TILE_N   = _env_int("AOP_FWD_TILE_N",     256)
# Round 38: cluster=(1,1) sw=1 raster=N — found via Round 38 Phase C cluster
# sweep, which exposed the gap left by Round 37's cluster cross-product table
# (Round 37 only tested c=(1,1) at sw=2/4/8 raster=M, never c=(1,1) sw=1
# raster=N). 4-trial paired bench median 1.0221 vs Round 37 lock c=(2,1)
# sw=1 rN 1.0411 (-1.90pp). N=1024 yields only 4 N-tiles per row, so cluster
# multicast on either axis is wasted bandwidth — the simpler c=(1,1)
# scheduler avoids inter-CTA mbar/sync overhead.
# Round 38 also re-swept mma_k {1,2,8} on this cluster and confirmed mk=4
# is best (mk=2 1.0267, mk=8 1.36, mk=1 2.14); mma_k=2 only beat mk=4 on the
# old c=(2,1) path because c=(2,1) shifted the K-pipeline sweet-spot.
_FWD_CLU_M    = _env_int("AOP_FWD_CLU_M",        1)  # R60: c(1,2) sw8 best (-10.0% vs cuBLAS)
_FWD_CLU_N    = _env_int("AOP_FWD_CLU_N",        2)
_FWD_SW       = _env_int("AOP_FWD_SW",           8)  # R60: sw8 confirmed best
_FWD_RAS_M    = _env_int("AOP_FWD_RAS_M",        0)
_FWD_MMA_K    = _env_int("AOP_FWD_MMA_K",        4)
_FWD_OCC      = _env_int("AOP_FWD_OCC",          1)
_FWD_KPM      = _env_int("AOP_FWD_KPM",          1)
# Round 39: epi_stage controls TMA-store pipeline depth in the persistent
# kernel epilogue. Round 39 sweep (AOP_FWD_EPI ∈ {2,3,4,5,6,8}, paired bench
# 4 trials × 250 iters): epi=8 monotone winner: 1.0151 vs epi=4 LOCK 1.0227
# (−0.76% gap). Higher staging keeps TMA-store warp continuously busy and
# overlaps better with the next tile's WGMMA. epi_stage doesn't shift
# ab_stage at fwd's (128,256) BF16 tile so no mainloop regression.
_FWD_EPI      = _env_int("AOP_FWD_EPI",          4)  # R60: auto (None) best; use 4 as safe default

_DGRAD_TILE_M = _env_int("AOP_DGRAD_TILE_M",   128)
_DGRAD_TILE_N = _env_int("AOP_DGRAD_TILE_N",   256)
_DGRAD_CLU_M  = _env_int("AOP_DGRAD_CLU_M",      1)
_DGRAD_CLU_N  = _env_int("AOP_DGRAD_CLU_N",      2)  # R-autotune
_DGRAD_SW     = _env_int("AOP_DGRAD_SW",         2)  # R-autotune: sw2 > sw1
_DGRAD_RAS_M  = _env_int("AOP_DGRAD_RAS_M",      0)
_DGRAD_MMA_K  = _env_int("AOP_DGRAD_MMA_K",      4)
_DGRAD_OCC    = _env_int("AOP_DGRAD_OCC",        1)
_DGRAD_KPM    = _env_int("AOP_DGRAD_KPM",        1)
# Round 39 sweep (AOP_DGRAD_EPI ∈ {2,3,4,5,6,8}): epi=8 winner 1.0018 vs
# epi=4 LOCK 1.0091 (−0.73% gap, min trial 0.9994 actually beats baseline).
_DGRAD_EPI    = _env_int("AOP_DGRAD_EPI",        4)  # R60: auto (None) best; use 4 as safe default

_WGRAD_TILE_M = _env_int("AOP_WGRAD_TILE_M",   128)
_WGRAD_TILE_N = _env_int("AOP_WGRAD_TILE_N",   128)
_WGRAD_CLU_M  = _env_int("AOP_WGRAD_CLU_M",      1)
_WGRAD_CLU_N  = _env_int("AOP_WGRAD_CLU_N",      2)
_WGRAD_SW     = _env_int("AOP_WGRAD_SW",         1)  # Reverted: auto-tuned sw=2 was -87.5% vs cuBLAS
# Round 27 sweep: raster=N is ~1.7us faster than raster=M in standalone wgrad
# split-K=2 batched (120.8us vs 122.5us). raster_along_m default = 0 (False).
_WGRAD_RAS_M  = _env_int("AOP_WGRAD_RAS_M",      0)
_WGRAD_MMA_K  = _env_int("AOP_WGRAD_MMA_K",      4)
_WGRAD_SPLITS = _env_int("AOP_WGRAD_SPLITS",     2)
_WGRAD_OCC    = _env_int("AOP_WGRAD_OCC",        1)
_WGRAD_KPM    = _env_int("AOP_WGRAD_KPM",        1)
# Round 39 sweep (AOP_WGRAD_EPI ∈ {2,3,4,5,6,8}): epi=4 LOCK still wins at
# 1.0066 (epi=8 = 1.0100, epi=2 = 1.0073). FP32 wgrad output (8 KB / epi
# tile) means raising epi_stage drops one ab_stage in the (128,128) tile
# budget — net regression. Keep at 4.
_WGRAD_EPI    = _env_int("AOP_WGRAD_EPI",        4)


def _get_wgrad_partials(num_splits: int, m: int, n: int,
                        device: torch.device) -> torch.Tensor:
    key = (num_splits, m, n, device.index, str(device.type))
    buf = _wgrad_partials_cache.get(key)
    if buf is None:
        buf = torch.empty(num_splits, m, n, dtype=torch.float32, device=device)
        _wgrad_partials_cache[key] = buf
    return buf


# Round 27: cache the cute Tensor for `partials_flat.permute(1,2,0)`. The
# torch tensor is long-lived and same data_ptr across calls; we save one
# from_dlpack + mark_layout_dynamic per call (~3-5 us in micro-bench).
_wgrad_partials_cute_cache: dict[int, tuple[int, object]] = {}


def _get_wgrad_partials_cute(partials_flat: torch.Tensor) -> object:
    key = id(partials_flat)
    cached = _wgrad_partials_cute_cache.get(key)
    dptr = partials_flat.data_ptr()
    if cached is not None and cached[0] == dptr:
        return cached[1]
    permuted = partials_flat.permute(1, 2, 0)
    ct = _to_cute_3d(permuted, FP32)
    _wgrad_partials_cute_cache[key] = (dptr, ct)
    return ct


# Round 28: 1-slot data_ptr-based cute Tensor cache for fwd's per-call
# tensors (x, out). PyTorch caching allocator typically reuses addresses
# for tensors of the same shape/dtype across iterations of the training
# loop, so this hits with high probability after warmup.
#
# Memory: each cached entry holds a strong ref to the original torch
# Tensor via the dlpack capsule. We keep at most 1 per slot, so worst-case
# extra memory = 1 tensor's worth (~80MB for fwd x or out @ 40960×1024 BF16).
# This is acceptable on H100 80GB.
#
# Safety: cache hit checks t.data_ptr() == cached_data_ptr. If True, the
# cached cute Tensor's metadata (ptr/shape/stride/dtype) is consistent
# with the new caller's request. Holding the old strong ref doesn't matter
# for correctness because the kernel will read/write data_ptr's memory
# anyway.
_addr_cute_cache: dict[str, tuple[int, tuple, object]] = {}


def _to_cute_3d_addr_cached(slot: str, t: torch.Tensor, cutlass_dtype) -> object:
    """1-slot cute Tensor cache keyed by (slot, data_ptr)."""
    cached = _addr_cute_cache.get(slot)
    dptr = t.data_ptr()
    shape_stride = (t.shape, t.stride())
    if cached is not None and cached[0] == dptr and cached[1] == shape_stride:
        return cached[2]
    ct = _to_cute_3d(t, cutlass_dtype)
    _addr_cute_cache[slot] = (dptr, shape_stride, ct)
    return ct


# Round 29: Cache `torch.as_strided` views by source data_ptr. The wgrad path
# rebuilds `a_batched` / `b_batched` views every call (~1.6us each via
# as_strided). With a 1-slot data_ptr-keyed cache and the corresponding
# cute Tensor already cached via _to_cute_3d_addr_cached, we save ~3us total
# per bwd call in steady-state training (caching allocator reuses ptrs).
#
# Safety: cache holds a strong ref to the as_strided'd view → ref to the
# original storage. Source tensors (dy_2d / x_2d) are the activations of
# the current step; they are produced fresh each call but the caching
# allocator reuses the underlying storage when the previous activation
# has been freed by Python GC. Holding a stale view does not corrupt new
# data because a cache miss (data_ptr mismatch) rebuilds the view.
_view_cache: dict[str, tuple[int, torch.Tensor]] = {}


def _get_or_make_view(slot: str, source: torch.Tensor,
                      shape: tuple, stride: tuple) -> torch.Tensor:
    """Return cached as_strided(source, shape, stride) keyed by source.data_ptr."""
    cached = _view_cache.get(slot)
    dptr = source.data_ptr()
    if cached is not None and cached[0] == dptr:
        return cached[1]
    v = torch.as_strided(source, shape, stride)
    _view_cache[slot] = (dptr, v)
    return v


# Round 29: Cache d_weight (FP32 1024x1024 = 4MB) per-layer keyed by id(weight).
# Each layer has its own weight tensor (long-lived); the matching d_weight is
# stored in `grads[<param>]` and consumed at optimizer.step() between iters.
# Reusing the same buffer per layer is safe because:
#   - within a single bwd pass the layer's d_weight is written once
#   - between iters, optimizer.step() consumes grads[<param>] before the next
#     bwd writes to it again
# Memory: ~4MB per attention layer × 24 layers = ~96MB.
_d_weight_cache: dict[int, tuple[int, torch.Tensor]] = {}


def _get_d_weight(weight: torch.Tensor, m: int, n: int) -> torch.Tensor:
    key = id(weight)
    cached = _d_weight_cache.get(key)
    dptr = weight.data_ptr()
    if cached is not None and cached[0] == dptr:
        return cached[1]
    dw = torch.empty(m, n, dtype=torch.float32, device=weight.device)
    _d_weight_cache[key] = (dptr, dw)
    return dw


# Round 29: Per-(weight, x_shape) cache for d_input (BF16, e.g. 80MB for
# attn_out_proj). Each layer's d_input is consumed by attention_bwd in the
# same iter; it does NOT need to survive across iters. We key by
# (id(weight), x.shape, x.stride[-1]) so different layers (and different
# upstream sources reaching the same layer's bwd, if any) get distinct
# buffers. Memory: ~80MB per layer × 24 layers ≈ 1.9GB.
#
# This is more conservative than a 1-slot cache (which would race across
# layers reading their respective d_inputs through Python attribute holds)
# and trades extra memory for correctness guarantees in mixed-grad-accum
# / async dispatch scenarios.
_d_input_cache: dict[tuple, tuple[int, torch.Tensor]] = {}


def _get_d_input(weight: torch.Tensor, m: int, n: int,
                 dtype: torch.dtype) -> torch.Tensor:
    key = (id(weight), m, n, dtype)
    cached = _d_input_cache.get(key)
    dptr = weight.data_ptr()
    if cached is not None and cached[0] == dptr:
        return cached[1]
    di = torch.empty(m, n, dtype=dtype, device=weight.device)
    _d_input_cache[key] = (dptr, di)
    return di


# Round 29: Per-(weight, x_shape) cache for fwd `out` (same rationale as
# _d_input_cache but for the forward output buffer).
_fwd_out_cache: dict[tuple, tuple[int, torch.Tensor]] = {}


def _get_fwd_out(weight: torch.Tensor, m: int, n: int,
                 dtype: torch.dtype) -> torch.Tensor:
    key = (id(weight), m, n, dtype)
    cached = _fwd_out_cache.get(key)
    dptr = weight.data_ptr()
    if cached is not None and cached[0] == dptr:
        return cached[1]
    o = torch.empty(m, n, dtype=dtype, device=weight.device)
    _fwd_out_cache[key] = (dptr, o)
    return o


# Round 29: env-var kill switches for safe rollback.
_USE_DW_CACHE  = os.environ.get("AOP_NO_DW_CACHE",  "0") != "1"
_USE_DI_CACHE  = os.environ.get("AOP_NO_DI_CACHE",  "0") != "1"
_USE_OUT_CACHE = os.environ.get("AOP_NO_OUT_CACHE", "0") != "1"
_USE_VIEW_CACHE = os.environ.get("AOP_NO_VIEW_CACHE", "0") != "1"


_jit_memory_cleaned = False


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
    cache_b: bool = False,
):
    """Run a non-persistent GEMM using a cached compiled kernel."""
    global _jit_memory_cleaned
    stream = _get_stream()
    mA = _to_cute_3d(a_torch, a_cutlass_dtype)
    mB = _to_cute_3d_cached(b_torch, b_cutlass_dtype) if cache_b \
        else _to_cute_3d(b_torch, b_cutlass_dtype)
    mC = _to_cute_3d(c_torch, c_cutlass_dtype)

    if key not in _compiled_cache:
        if not _jit_memory_cleaned:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            _jit_memory_cleaned = True
        gemm = _HopperGemm(FP32, tile_mn, cluster_shape_mn=cluster_mn,
                            force_atom_layout=force_atom_layout,
                            occupancy=occupancy, k_pipe_mmas=k_pipe_mmas)
        compiled = cute.compile(gemm, mA, mB, mC, stream)
        _compiled_cache[key] = compiled

    _compiled_cache[key](mA, mB, mC, stream)


_MAX_ACTIVE_CLUSTERS = 132


def _run_gemm_persistent(
    key: str,
    a_torch: torch.Tensor,
    b_torch: torch.Tensor,
    c_torch,  # Optional[torch.Tensor]; unused when mC_cute is provided.
    a_cutlass_dtype,
    b_cutlass_dtype,
    c_cutlass_dtype,
    tile_mn: tuple[int, int] = (128, 256),
    cluster_mn: tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    cache_b: bool = False,
    cache_b_t: bool = False,
    mma_inst_tile_k: int = 4,
    mC_cute: object | None = None,
    addr_cache_a: str | None = None,
    addr_cache_c: str | None = None,
    occupancy: int = 1,
    k_pipe_mmas: int = 1,
    epi_stage: int = 4,
):
    """Run a persistent GEMM with warp specialization.

    cache_b       : cache cute Tensor for b_torch keyed by id(b_torch).
    cache_b_t     : if True, b_torch must be the *original* weight (untransposed);
                    we build & cache cute Tensor for b_torch.T keyed by id(b_torch).
    mC_cute       : pre-built cute Tensor for c_torch; bypass the per-call
                    from_dlpack conversion for long-lived output buffers
                    (e.g. wgrad partials).
    addr_cache_a / addr_cache_c : if non-None, identifies a slot in the
                    1-entry data_ptr-keyed cute Tensor cache for a_torch /
                    c_torch. Saves ~4us per cache hit; allocator reuse makes
                    hit rate high in steady-state training.
    """
    global _jit_memory_cleaned
    stream = _get_stream()
    if addr_cache_a is not None:
        mA = _to_cute_3d_addr_cached(addr_cache_a, a_torch, a_cutlass_dtype)
    else:
        mA = _to_cute_3d(a_torch, a_cutlass_dtype)
    if cache_b_t:
        mB = _to_cute_3d_T_cached(b_torch, b_cutlass_dtype)
    elif cache_b:
        mB = _to_cute_3d_cached(b_torch, b_cutlass_dtype)
    else:
        mB = _to_cute_3d(b_torch, b_cutlass_dtype)
    if mC_cute is not None:
        mC = mC_cute
    elif addr_cache_c is not None:
        mC = _to_cute_3d_addr_cached(addr_cache_c, c_torch, c_cutlass_dtype)
    else:
        mC = _to_cute_3d(c_torch, c_cutlass_dtype)

    if key not in _compiled_cache:
        if not _jit_memory_cleaned:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            _jit_memory_cleaned = True
        gemm = _HopperGemmPersistent(
            FP32, tile_mn, cluster_shape_mn=cluster_mn,
            swizzle_size=swizzle_size, raster_along_m=raster_along_m,
            mma_inst_tile_k=mma_inst_tile_k,
            occupancy=occupancy, k_pipe_mmas=k_pipe_mmas,
            epi_stage=epi_stage,
        )
        compiled = cute.compile(gemm, mA, mB, mC, _MAX_ACTIVE_CLUSTERS, stream)
        _compiled_cache[key] = compiled

    _compiled_cache[key](mA, mB, mC, stream)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def warmup(device: str = "cuda:0") -> None:
    """Pre-compile CuTeDSL kernels with exact training shapes and strides."""
    d = device
    M = 40960
    N = K = 1024
    NUM_SPLITS = _WGRAD_SPLITS
    chunk_K = M // NUM_SPLITS

    dy_2d = torch.randn(M, N, dtype=torch.bfloat16, device=d)
    x_2d = torch.randn(M, K, dtype=torch.bfloat16, device=d)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=d)

    # fwd: A[M,K] contiguous, B[N,K] contiguous → out[M,N]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=d)
    _run_gemm_persistent(f"aop_fwd_persistent_e{_FWD_EPI}", x_2d, w, out, BF16, BF16, BF16,
                         tile_mn=(_FWD_TILE_M, _FWD_TILE_N),
                         cluster_mn=(_FWD_CLU_M, _FWD_CLU_N),
                         swizzle_size=_FWD_SW,
                         raster_along_m=bool(_FWD_RAS_M),
                         mma_inst_tile_k=_FWD_MMA_K,
                         occupancy=_FWD_OCC, k_pipe_mmas=_FWD_KPM,
                         epi_stage=_FWD_EPI)

    # dgrad: A[M,N] contiguous, B=w.T[K,N] non-contiguous view → d_inp[M,K]
    d_inp = torch.empty(M, K, dtype=torch.bfloat16, device=d)
    _run_gemm_persistent(f"aop_dgrad_persistent_e{_DGRAD_EPI}", dy_2d, w.T, d_inp,
                         BF16, BF16, BF16,
                         tile_mn=(_DGRAD_TILE_M, _DGRAD_TILE_N),
                         cluster_mn=(_DGRAD_CLU_M, _DGRAD_CLU_N),
                         swizzle_size=_DGRAD_SW,
                         raster_along_m=bool(_DGRAD_RAS_M),
                         mma_inst_tile_k=_DGRAD_MMA_K,
                         occupancy=_DGRAD_OCC, k_pipe_mmas=_DGRAD_KPM,
                         epi_stage=_DGRAD_EPI)

    # wgrad: use EXACT same as_strided + permute pattern as training bwd
    a_w = torch.as_strided(
        dy_2d, (N, chunk_K, NUM_SPLITS),
        (1, dy_2d.stride(0), chunk_K * dy_2d.stride(0)))
    b_w = torch.as_strided(
        x_2d, (K, chunk_K, NUM_SPLITS),
        (1, x_2d.stride(0), chunk_K * x_2d.stride(0)))
    partials_flat = _get_wgrad_partials(NUM_SPLITS, N, K,
                                        torch.device(d) if isinstance(d, str) else d)
    partials = partials_flat.permute(1, 2, 0)
    partials_cute = _get_wgrad_partials_cute(partials_flat)
    # Round 26 ablation: tried split-K=4 with (128,256) c(2,1) sw2 rM
    # (isolation -5us per Round 26 sweep) but wider partials reduction over
    # 4 buffers cost ~10us in bench_all_gemm — net regression. NS=2 stays.
    # Round 27 sweep: same tile/cluster, raster=N is ~1.7us faster than
    # raster=M (120.8us vs 122.5us in standalone wgrad split-K=2 batched).
    _run_gemm_persistent(f"aop_wgrad_persistent_e{_WGRAD_EPI}", a_w, b_w, partials,
                         BF16, BF16, FP32,
                         tile_mn=(_WGRAD_TILE_M, _WGRAD_TILE_N),
                         cluster_mn=(_WGRAD_CLU_M, _WGRAD_CLU_N),
                         swizzle_size=_WGRAD_SW,
                         raster_along_m=bool(_WGRAD_RAS_M),
                         mma_inst_tile_k=_WGRAD_MMA_K,
                         occupancy=_WGRAD_OCC, k_pipe_mmas=_WGRAD_KPM,
                         epi_stage=_WGRAD_EPI,
                         mC_cute=partials_cute)

    del dy_2d, x_2d, w, out, d_inp
    torch.cuda.empty_cache()

    # Pre-load the C-export extension so the first training step doesn't pay
    # the export+compile latency. Falls through silently if export disabled.
    _get_cexport_ext()


# ---------------------------------------------------------------------------
#  AOT C-export integration
# ---------------------------------------------------------------------------
# Round 28+: bench_overhead.py confirmed that the CuTeDSL kernel itself
# is faster than cuBLAS, but the per-call Python wrapper (DLPack +
# mark_layout_dynamic + memref descriptor) costs ~36us on fwd.
#
# The fix is the same one used by gemm_fc1: precompile each persistent
# kernel via cute.compile + compiled.export_to_c() into .h+.o, then load
# them from a torch.utils.cpp_extension wrapper that calls
# cuLaunchKernel directly with raw torch tensor pointers.
#
# Disable by setting `OP_GEMM_ATTN_OUT_PROJ_BACKEND=python` (falls back
# to the original cute.compile + JIT-callable path).
# Shared-cache convention (see workload/ops/gemm_fc1/kernel.py).
EXPORT_DIR = os.path.join(
    os.environ.get("CUTEDSL_CACHE_ROOT", "/tmp"),
    "cutedsl_export_gemm_attn_out_proj",
)
_CEXPORT_LOCK_PATH = os.path.join(EXPORT_DIR, ".export.lock")
_CEXPORT_DIRECTIONS = ("aop_fwd", "aop_dgrad", "aop_wgrad")
_MIN_OBJ_SIZE = 4096

_cexport_ext = None
_cexport_state = None  # None -> not tried; "ok"; "failed"


def _cexport_files_valid() -> bool:
    needed = [os.path.join(EXPORT_DIR, d, f"{d}.o")
              for d in _CEXPORT_DIRECTIONS]
    headers = [os.path.join(EXPORT_DIR, d, f"{d}.h")
               for d in _CEXPORT_DIRECTIONS]
    for p in needed + headers:
        if not os.path.exists(p):
            return False
    for p in needed:
        if os.path.getsize(p) < _MIN_OBJ_SIZE:
            return False
    return True


def _try_cutedsl_export() -> bool:
    """Run export_kernels.py once if .h/.o files don't exist or are stale."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(src_dir, "export_kernels.py")
    if not os.path.exists(export_script):
        return False

    os.makedirs(EXPORT_DIR, exist_ok=True)

    with open(export_script, "rb") as f:
        h = hashlib.md5(f.read())
    for ev in (
        # fwd
        "AOP_FWD_TILE_M", "AOP_FWD_TILE_N",
        "AOP_FWD_CLU_M", "AOP_FWD_CLU_N",
        "AOP_FWD_CLUSTER_M", "AOP_FWD_CLUSTER_N",
        "AOP_FWD_SW", "AOP_FWD_RAS_M",
        "AOP_FWD_MMA_K", "AOP_FWD_EPI",
        # dgrad
        "AOP_DGRAD_TILE_M", "AOP_DGRAD_TILE_N",
        "AOP_DGRAD_CLU_M", "AOP_DGRAD_CLU_N",
        "AOP_DGRAD_CLUSTER_M", "AOP_DGRAD_CLUSTER_N",
        "AOP_DGRAD_SW", "AOP_DGRAD_RAS_M",
        "AOP_DGRAD_MMA_K", "AOP_DGRAD_EPI",
        # wgrad
        "AOP_WGRAD_TILE_M", "AOP_WGRAD_TILE_N",
        "AOP_WGRAD_CLU_M", "AOP_WGRAD_CLU_N",
        "AOP_WGRAD_CLUSTER_M", "AOP_WGRAD_CLUSTER_N",
        "AOP_WGRAD_SW", "AOP_WGRAD_RAS_M",
        "AOP_WGRAD_RASTER_M",
        "AOP_WGRAD_MMA_K", "AOP_WGRAD_EPI",
        "AOP_WGRAD_SPLITS", "AOP_WGRAD_BATCH",
        "AOP_WGRAD_K",
    ):
        h.update(f"{ev}={os.environ.get(ev, '')}\n".encode())
    current_hash = h.hexdigest()
    config_hash_path = os.path.join(EXPORT_DIR, ".config_hash")

    def _hash_matches():
        if not os.path.exists(config_hash_path):
            return False
        try:
            with open(config_hash_path) as f:
                return f.read().strip() == current_hash
        except OSError:
            return False

    if _hash_matches() and _cexport_files_valid():
        return True

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    if rank == 0:
        lock_fd = open(_CEXPORT_LOCK_PATH, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if _hash_matches() and _cexport_files_valid():
                return True
            result = subprocess.run(
                [sys.executable, export_script],
                cwd=src_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print(f"[gemm_attn_out_proj] CuTeDSL export failed (rc="
                      f"{result.returncode}): {result.stderr[:500]}",
                      file=sys.stderr)
                return False
            if not _cexport_files_valid():
                print("[gemm_attn_out_proj] CuTeDSL export produced "
                      "incomplete files", file=sys.stderr)
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"[gemm_attn_out_proj] CuTeDSL export error: {e}",
                  file=sys.stderr)
            return False
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    else:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if _hash_matches() and _cexport_files_valid():
                return True
            time.sleep(2)
        return False


def _load_cutedsl_ext():
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(src_dir, "gemm_cutedsl.cpp")

    obj_files = [os.path.join(EXPORT_DIR, d, f"{d}.o")
                 for d in _CEXPORT_DIRECTIONS]
    header_files = [os.path.join(EXPORT_DIR, d, f"{d}.h")
                    for d in _CEXPORT_DIRECTIONS]

    h = hashlib.md5()
    for p in header_files + obj_files:
        with open(p, "rb") as f:
            h.update(f.read())
    with open(cpp_src, "rb") as f:
        h.update(f.read())
    build_hash = h.hexdigest()[:8]
    ext_name = f"gemm_attn_out_proj_cexport_{build_hash}"

    cache_dir = os.path.expanduser(
        f"~/.cache/torch_extensions/py312_cu129/{ext_name}")
    marker = os.path.join(cache_dir, ".aop_build_hash") if cache_dir else ""
    cache_valid = False
    if marker and os.path.exists(marker):
        try:
            with open(marker) as f:
                if f.read().strip() == build_hash:
                    cache_valid = True
        except OSError:
            pass
    if not cache_valid and os.path.isdir(cache_dir):
        import shutil as _sh
        _sh.rmtree(cache_dir, ignore_errors=True)

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    include_dirs = [os.path.join(EXPORT_DIR, d)
                    for d in _CEXPORT_DIRECTIONS] + [
        os.path.join(cuda_home, "include")]

    runtime_lib = ("/usr/local/lib/python3.12/dist-packages/"
                   "nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a")
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

    cuda_lib64 = os.path.join(cuda_home, "lib64")
    ext = load(
        name=ext_name,
        sources=[cpp_src],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3"],
        extra_ldflags=obj_files + [
            "-Wl,--whole-archive", runtime_lib, "-Wl,--no-whole-archive",
            "-lcuda",
            "-L" + cuda_lib64, "-lcudart",
            "-Wl,-rpath," + cuda_lib64,
        ],
        verbose=False,
    )
    if marker:
        try:
            with open(marker, "w") as f:
                f.write(build_hash)
        except OSError:
            pass
    return ext


def _get_cexport_ext():
    """Return the CuTeDSL C-export extension or None on failure."""
    global _cexport_ext, _cexport_state

    if _cexport_state == "failed":
        return None
    if _cexport_ext is not None:
        return _cexport_ext

    # R44: default to cexport for fwd dispatch. R26 disabled AOT because
    # dynamic-layout descriptor fill cost ~36µs/call. R44 switches all
    # directions to is_dynamic_layout=False (descriptor = {ptr} only).
    # R44 bench: AOT fwd 0.961x vs JIT fwd 1.065x — AOT saves ~10µs/call.
    # Opt out via BACKEND=jit if needed.
    if os.environ.get("OP_GEMM_ATTN_OUT_PROJ_BACKEND", "cexport") != "cexport":
        _cexport_state = "failed"
        return None

    try:
        if not _try_cutedsl_export():
            _cexport_state = "failed"
            return None
        _cexport_ext = _load_cutedsl_ext()
        _cexport_state = "ok"
        rank = int(os.environ.get("LOCAL_RANK",
                                   os.environ.get("RANK", "0")))
        if rank == 0:
            print("[gemm_attn_out_proj] loaded CuTeDSL C-export backend",
                  flush=True)
        return _cexport_ext
    except Exception as e:
        _cexport_state = "failed"
        rank = int(os.environ.get("LOCAL_RANK",
                                   os.environ.get("RANK", "0")))
        if rank == 0:
            print(f"[gemm_attn_out_proj] CuTeDSL C-export load failed; "
                  f"falling back to Python JIT path: {e}",
                  file=sys.stderr, flush=True)
        return None


def gemm_attn_out_proj_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Forward: out = x @ weight.T

    x:      [S, B, 1024] BF16
    weight: [1024, 1024] BF16
    Returns [S, B, 1024] BF16.

    CUTLASS convention: D[m,n] = A[m,k] * B[n,k]
    A = X (M×K, K-major row-major), B = W (N×K, K-major row-major), C = Y (M×N, N-major)
    """
    shape = x.shape
    if x.is_contiguous():
        x_2d = x.view(-1, shape[-1])
    else:
        x_2d = x.contiguous().view(-1, shape[-1])
    w = weight if weight.is_contiguous() else weight.contiguous()
    M, K = x_2d.shape
    N = w.shape[0]

    if _USE_OUT_CACHE:
        out = _get_fwd_out(w, M, N, torch.bfloat16)
    else:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    global _inhouse_logged_aop
    if _USE_INHOUSE_JIT_AOP:
        if not _inhouse_logged_aop:
            print("[gemm_attn_out_proj] using inhouse JIT for fwd/dgrad")
            _inhouse_logged_aop = True
        from training_engine_tensor.ops._gemm_inhouse_jit import jit_gemm_aop_fwd
        jit_gemm_aop_fwd(x_2d, w, out)
    elif (_ext_cexport := _get_cexport_ext()) is not None:
        _ext_cexport.aop_fwd(x_2d, w, out)
    elif _USE_INHOUSE_AOP:
        inhouse = _get_inhouse_ext_aop()
        if inhouse is None:
            raise RuntimeError("[gemm_attn_out_proj] AOP_BACKEND=inhouse but inhouse AOT "
                               "extension failed to load")
        if not _inhouse_logged_aop:
            print("[gemm_attn_out_proj] using inhouse persistent AOT for fwd/dgrad")
            _inhouse_logged_aop = True
        inhouse.gemm_fwd_fast(x_2d, w, out)
    else:
        _run_gemm_persistent(f"aop_fwd_persistent_e{_FWD_EPI}", x_2d, w, out, BF16, BF16, BF16,
                             tile_mn=(_FWD_TILE_M, _FWD_TILE_N),
                             cluster_mn=(_FWD_CLU_M, _FWD_CLU_N),
                             swizzle_size=_FWD_SW,
                             raster_along_m=bool(_FWD_RAS_M),
                             mma_inst_tile_k=_FWD_MMA_K,
                             occupancy=_FWD_OCC, k_pipe_mmas=_FWD_KPM,
                             epi_stage=_FWD_EPI,
                             cache_b=True,
                             addr_cache_a="fwd_x", addr_cache_c="fwd_out")

    return out.view(*shape[:-1], N)


def gemm_attn_out_proj_bwd(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward: returns (d_input, d_weight_fp32).

    d_output: [S, B, 1024] BF16
    x:        [S, B, 1024] BF16
    weight:   [1024, 1024] BF16
    Returns:  d_input [S, B, 1024] BF16, d_weight [1024, 1024] FP32.

    Round 21: dgrad and wgrad both run on the default stream (serial).
    bench_overlap.py / bench_cached.py confirmed that stream overlap
    saves <0.5% in kernel time but adds ~20us of event/stream overhead
    per call, making serial faster overall.
    """
    shape_dy = d_output.shape
    shape_x = x.shape

    if d_output.is_contiguous():
        dy_2d = d_output.view(-1, shape_dy[-1])
    else:
        dy_2d = d_output.contiguous().view(-1, shape_dy[-1])
    if x.is_contiguous():
        x_2d = x.view(-1, shape_x[-1])
    else:
        x_2d = x.contiguous().view(-1, shape_x[-1])
    w = weight if weight.is_contiguous() else weight.contiguous()

    M_dgrad = dy_2d.shape[0]
    N_dgrad = w.shape[1]

    if _USE_DI_CACHE:
        d_input = _get_d_input(w, M_dgrad, N_dgrad, torch.bfloat16)
    else:
        d_input = torch.empty(M_dgrad, N_dgrad, dtype=torch.bfloat16, device=x.device)

    M_wgrad = dy_2d.shape[1]   # 1024
    N_wgrad = x_2d.shape[1]    # 1024
    K_wgrad = dy_2d.shape[0]   # 40960

    # Round 26 ablation: tried NUM_SPLITS=4 + (128,256) c(2,1) sw2 rM
    # (wgrad isolation kernel ~-5us vs NS=2 (128,128) c(2,1) sw4) but the
    # wider `partials_flat.sum(dim=0)` reduction over 4 partials cost ~10us
    # in bench_all_gemm — net regression to bwd 1.13x. NUM_SPLITS=2 stays.
    NUM_SPLITS = _WGRAD_SPLITS

    partials_flat = _get_wgrad_partials(NUM_SPLITS, M_wgrad, N_wgrad, x.device)

    # R44: dgrad uses AOT C-export when available (same shape as fwd,
    # saves ~5µs dispatch overhead). Wgrad uses JIT path — the batched
    # split-K wgrad has identical kernel times between AOT and JIT (~160µs),
    # and the JIT path avoids C++ wrapper overhead for batched descriptors.
    global _inhouse_logged_aop
    if _USE_INHOUSE_JIT_AOP:
        if not _inhouse_logged_aop:
            print("[gemm_attn_out_proj] using inhouse JIT for fwd/dgrad")
            _inhouse_logged_aop = True
        from training_engine_tensor.ops._gemm_inhouse_jit import jit_gemm_aop_dgrad
        jit_gemm_aop_dgrad(dy_2d, w, d_input)
    elif (_ext_cexport := _get_cexport_ext()) is not None and not _USE_INHOUSE_AOP:
        _ext_cexport.aop_dgrad(dy_2d, w, d_input)
    elif _USE_INHOUSE_AOP:
        inhouse = _get_inhouse_ext_aop()
        if inhouse is None:
            raise RuntimeError("[gemm_attn_out_proj] AOP_BACKEND=inhouse but inhouse AOT "
                               "extension failed to load")
        if not _inhouse_logged_aop:
            print("[gemm_attn_out_proj] using inhouse persistent AOT for fwd/dgrad")
            _inhouse_logged_aop = True
        inhouse.gemm_dgrad_fast(dy_2d, w, d_input)
    else:
        _run_gemm_persistent(f"aop_dgrad_persistent_e{_DGRAD_EPI}", dy_2d, w, d_input,
                             BF16, BF16, BF16,
                             tile_mn=(_DGRAD_TILE_M, _DGRAD_TILE_N),
                             cluster_mn=(_DGRAD_CLU_M, _DGRAD_CLU_N),
                             swizzle_size=_DGRAD_SW,
                             raster_along_m=bool(_DGRAD_RAS_M),
                             mma_inst_tile_k=_DGRAD_MMA_K,
                             occupancy=_DGRAD_OCC, k_pipe_mmas=_DGRAD_KPM,
                             epi_stage=_DGRAD_EPI,
                             cache_b_t=True,
                             addr_cache_a="dgrad_dy", addr_cache_c="dgrad_di")

    if _USE_INHOUSE_JIT_AOP:
        from training_engine_tensor.ops._gemm_inhouse_jit import jit_gemm_aop_wgrad
        d_weight = torch.empty(M_wgrad, N_wgrad, dtype=torch.float32, device=x.device)
        jit_gemm_aop_wgrad(dy_2d, x_2d, d_weight, num_splits=NUM_SPLITS)
        return d_input.view_as(x), d_weight

    chunk_K = K_wgrad // NUM_SPLITS
    if _USE_VIEW_CACHE:
        a_batched = _get_or_make_view(
            "wgrad_a_view", dy_2d,
            (M_wgrad, chunk_K, NUM_SPLITS),
            (1, dy_2d.stride(0), chunk_K * dy_2d.stride(0)))
        b_batched = _get_or_make_view(
            "wgrad_b_view", x_2d,
            (N_wgrad, chunk_K, NUM_SPLITS),
            (1, x_2d.stride(0), chunk_K * x_2d.stride(0)))
    else:
        a_batched = torch.as_strided(
            dy_2d, (M_wgrad, chunk_K, NUM_SPLITS),
            (1, dy_2d.stride(0), chunk_K * dy_2d.stride(0)))
        b_batched = torch.as_strided(
            x_2d, (N_wgrad, chunk_K, NUM_SPLITS),
            (1, x_2d.stride(0), chunk_K * x_2d.stride(0)))
    partials_cute = _get_wgrad_partials_cute(partials_flat)

    _run_gemm_persistent(f"aop_wgrad_persistent_e{_WGRAD_EPI}", a_batched, b_batched, None,
                         BF16, BF16, FP32,
                         tile_mn=(_WGRAD_TILE_M, _WGRAD_TILE_N),
                         cluster_mn=(_WGRAD_CLU_M, _WGRAD_CLU_N),
                         swizzle_size=_WGRAD_SW,
                         raster_along_m=bool(_WGRAD_RAS_M),
                         mma_inst_tile_k=_WGRAD_MMA_K,
                         occupancy=_WGRAD_OCC, k_pipe_mmas=_WGRAD_KPM,
                         epi_stage=_WGRAD_EPI,
                         mC_cute=partials_cute,
                         addr_cache_a="wgrad_a", addr_cache_c=None)

    # Round 28: pre-alloc d_weight then torch.add(p[0], p[1], out=d_weight).
    # `torch.add` saves ~0.5us vs `sum(dim=0)` for L=2 (round 24 measurement),
    # and using `out=` skips an internal allocator call inside torch.sum.
    # Round 29: cache d_weight per-id(weight) — saves ~2us alloc per call.
    if _USE_DW_CACHE:
        d_weight = _get_d_weight(w, M_wgrad, N_wgrad)
    else:
        d_weight = torch.empty(
            (M_wgrad, N_wgrad), dtype=torch.float32, device=x.device)
    if NUM_SPLITS == 2:
        torch.add(partials_flat[0], partials_flat[1], out=d_weight)
    else:
        torch.sum(partials_flat, dim=0, out=d_weight)

    return d_input.view_as(x), d_weight
