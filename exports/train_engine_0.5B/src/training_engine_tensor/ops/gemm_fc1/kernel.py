"""CuTeDSL-exported SM90 WGMMA+TMA kernel for gemm_fc1 (MLP up+gate projection).

Two-phase build:
  1. CuTeDSL compiles 3 persistent GEMM kernels → exports .h + .o (C ABI)
  2. torch.utils.cpp_extension links .o files into a shared library

This eliminates CuTeDSL Python dispatch overhead while keeping kernel performance
(~19% faster than CUTLASS 3.5.1 C++, which suffers from C7510 ptxas serialization).

Fallback: if CuTeDSL export fails, falls back to CUTLASS 3.x C++ kernels.
"""

from __future__ import annotations

__all__ = ["gemm_fc1_fwd", "gemm_fc1_bwd"]

import fcntl
import hashlib
import os
import sys
import subprocess
import time
import torch

from training_engine_tensor.op_dispatcher import get_build_env as _get_build_env
from training_engine_tensor.op_dispatcher import get_frozen_env as _get_frozen_env

_ext = None
_ext_type = None  # "cutedsl" or "cutlass"

# Cooperative + canonical-tile wgrad extension (Round 34 architectural
# data point #5 — only surviving CUTLASS 3.x native wgrad).  Tile
# (128,256,64) cluster=(1,1,1) matches the CuTeDSL default; the only
# knob exposed is `max_swizzle_size` via `GEMM_FC1_WGRAD_SWIZZLE` ∈
# {1,2,4,8}.  Default backend is still "cutedsl"; this extension is only
# built/loaded when `GEMM_FC1_WGRAD_BACKEND=coop_canonical` is set.  Kept
# in-tree (vs archived in `archive_r30_r35_negative/` like the other 5
# negative-result wgrad variants from R30/R31/R32/R33/R35) because it
# is statistically tied with CuTeDSL and serves as the canonical CUTLASS
# native skeleton for any future epilogue / mainloop experiment that
# needs a one-to-one comparison against the CuTeDSL builder default.
_wgrad_coop_canonical_ext = None
_wgrad_coop_canonical_load_failed = False

# Optional side stream for parallel dgrad+wgrad in bwd path.
# Set GEMM_FC1_PARALLEL_BWD=0 to disable (default = 1, on).
_bwd_side_stream = None

# Round 37: env-var cache (read once at import, not per-call).  All three
# knobs are intended to be set before training starts; flipping them at
# runtime was never supported anyway.  Reading them once saves a
# `_get_frozen_env().get(...)` lookup per `gemm_fc1_bwd` invocation (~1-2 µs each
# in CPython's PyMapping_GetItemString path).
_WGRAD_BACKEND = _get_frozen_env().get("GEMM_FC1_WGRAD_BACKEND", "cutedsl")
_USE_COOP_CANONICAL_WGRAD = (_WGRAD_BACKEND == "coop_canonical")
_PARALLEL_BWD = _get_frozen_env().get("GEMM_FC1_PARALLEL_BWD", "0") == "1"
try:
    _WGRAD_SWIZZLE = int(_get_frozen_env().get("GEMM_FC1_WGRAD_SWIZZLE", "4"))
except ValueError:
    _WGRAD_SWIZZLE = 4

_FC1_BACKEND = _get_frozen_env().get("GEMM_FC1_BACKEND", "cutedsl")
_USE_INHOUSE = _FC1_BACKEND == "inhouse"
_USE_INHOUSE_JIT = _FC1_BACKEND == "inhouse_jit"
_inhouse_logged = False

# ---------------------------------------------------------------------------
#  inhouse persistent AOT backend (self-contained export + load)
# ---------------------------------------------------------------------------
_inhouse_ext = None
_inhouse_ext_load_failed = False

_INHOUSE_EXPORT_DIR = os.path.join(
    _get_build_env().cutedsl_cache_root,
    "inhouse_aot_fc1",
)
_INHOUSE_LOCK_PATH = os.path.join(_INHOUSE_EXPORT_DIR, ".export.lock")
_INHOUSE_DIRECTIONS = ("inhouse_fc1_fwd", "inhouse_fc1_dgrad", "inhouse_fc1_wgrad")
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
                print(f"[gemm_fc1] inhouse export failed (rc={result.returncode}): "
                      f"{result.stderr[:500]}", file=sys.stderr)
                return False
            if not _files_valid():
                print("[gemm_fc1] inhouse export produced incomplete files",
                      file=sys.stderr)
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"[gemm_fc1] inhouse export error: {e}", file=sys.stderr)
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
        print(f"[gemm_fc1] inhouse export: rank {rank} timed out waiting for rank 0",
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
    ext_name = f"gemm_fc1_inhouse_{build_hash}"

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
            print("[gemm_fc1] loaded inhouse persistent AOT backend", flush=True)
        return _inhouse_ext
    except Exception as exc:  # noqa: BLE001
        _inhouse_ext_load_failed = True
        if rank == 0:
            print(f"[gemm_fc1] inhouse AOT load failed: {exc}",
                  file=sys.stderr, flush=True)
        return None


# Honor ``CUTEDSL_CACHE_ROOT`` so a one-shot prebuild on the cluster's
# shared FS (workload/scripts/prebuild_custom_ops.py) can populate the
# AOT-exported ``.o`` / ``.h`` files once and let every subsequent
# training pod skip the ~minutes-long CuTeDSL JIT export. Defaults to
# ``/tmp`` so the env-less local-dev path is unchanged.
EXPORT_DIR = os.path.join(
    _get_build_env().cutedsl_cache_root,
    "cutedsl_export_gemm_fc1",
)
_LOCK_PATH = os.path.join(EXPORT_DIR, ".export.lock")
_DIRECTIONS = ("gemm_fwd", "gemm_dgrad", "gemm_wgrad")
_MIN_OBJ_SIZE = 4096


def _try_cutedsl_export():
    """Run CuTeDSL export if .o files don't exist or config has changed.

    Uses a file lock to prevent races when multiple torchrun ranks call
    this simultaneously.  Only the rank that acquires the lock exports;
    others wait and then validate the files.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    export_script = os.path.join(src_dir, "export_kernels.py")
    if not os.path.exists(export_script):
        return False

    os.makedirs(EXPORT_DIR, exist_ok=True)

    needed = [os.path.join(EXPORT_DIR, d, f"{d}.o") for d in _DIRECTIONS]
    headers = [os.path.join(EXPORT_DIR, d, f"{d}.h") for d in _DIRECTIONS]
    all_files = needed + headers

    with open(export_script, "rb") as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    config_hash_path = os.path.join(EXPORT_DIR, ".config_hash")

    def _files_valid():
        """Check that all expected files exist and .o files are large enough."""
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
                [sys.executable, export_script],
                cwd=src_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f"CuTeDSL export failed (rc={result.returncode}): "
                      f"{result.stderr[:500]}", file=sys.stderr)
                return False
            if not _files_valid():
                print("CuTeDSL export produced incomplete files", file=sys.stderr)
                return False
            with open(config_hash_path, "w") as f:
                f.write(current_hash)
            return True
        except Exception as e:
            print(f"CuTeDSL export error: {e}", file=sys.stderr)
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
        print(f"CuTeDSL export: rank {rank} timed out waiting for rank 0",
              file=sys.stderr)
        return False


def _load_cutedsl_ext():
    """Load CuTeDSL C++ extension linking exported .o files.

    Uses a content hash of the .h/.o files in the extension name so that
    stale torch-extension caches from a previous CuTeDSL export (which
    embeds Python-object memory addresses in symbol names) are never reused.
    """
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_src = os.path.join(src_dir, "gemm_cutedsl.cpp")

    obj_files = [os.path.join(EXPORT_DIR, d, f"{d}.o") for d in _DIRECTIONS]
    header_files = [os.path.join(EXPORT_DIR, d, f"{d}.h") for d in _DIRECTIONS]

    h = hashlib.md5()
    for p in header_files + obj_files:
        with open(p, "rb") as f:
            h.update(f.read())
    build_hash = h.hexdigest()[:8]
    ext_name = f"gemm_fc1_cutedsl_{build_hash}"

    cuda_home = _get_build_env().cuda_home

    include_dirs = [
        os.path.join(EXPORT_DIR, d) for d in _DIRECTIONS
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
        # Round 41: some devspaces (e.g. shandongdev-297538) have the DSL
        # package laid out under a shared-FS root with the canonical
        # ``{lib,python_packages}/`` layout rather than pip-installed.
        # Honor ``CUTLASS_INSTALL_DIR`` (or the more recent
        # ``CUTLASS_DSL_FALLBACK_DIR``) so the link step finds
        # libcuda_dialect_runtime_static.a there.
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


def _load_cutlass_ext():
    """Fallback: load CUTLASS 3.x C++ extension."""
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_src = os.path.join(src_dir, "gemm_sm90.cu")

    ext = load(
        name="gemm_fc1_sm90",
        sources=[cuda_src],
        extra_include_paths=["/usr/include"],
        extra_cuda_cflags=[
            "-std=c++17", "-O3",
            "--generate-code=arch=compute_90a,code=sm_90a",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "--expt-relaxed-constexpr",
            "-Xptxas", "--suppress-stack-size-warning",
            "-Xptxas", "-O3",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )
    return ext


def _load_wgrad_coop_canonical_ext():
    """Round 34 — CUTLASS 3.x SM90 cooperative wgrad with canonical tile
    (128,256,64) cluster=(1,1,1) and runtime-configurable swizzle.

    Built only when `GEMM_FC1_WGRAD_BACKEND=coop_canonical` is requested.
    Lives in parallel with the CuTeDSL fwd/dgrad path so we can ablate
    "wgrad backend" independently.  The swizzle knob is passed at call
    time (not compile time) so a single .so can sweep `max_swizzle_size`
    across {1,2,4,8} without recompiling.
    """
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_src = os.path.join(src_dir, "gemm_wgrad_coop_canonical.cu")

    ext = load(
        name="gemm_fc1_wgrad_coop_canonical",
        sources=[cuda_src],
        extra_include_paths=["/usr/include"],
        extra_cuda_cflags=[
            "-std=c++17", "-O3",
            "--generate-code=arch=compute_90a,code=sm_90a",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            # Same CUTLASS 3.5.1 barrier.h assert workaround as the other
            # gemm_wgrad_*.cu variants (streamk / pingpong / coop_rot /
            # coop_2x2).
            "-DNDEBUG",
            "--expt-relaxed-constexpr",
            "-Xptxas", "--suppress-stack-size-warning",
            "-Xptxas", "-O3",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )
    return ext


def _get_wgrad_coop_canonical_ext():
    """Lazy load the cooperative+canonical wgrad extension; cache success/failure."""
    global _wgrad_coop_canonical_ext, _wgrad_coop_canonical_load_failed
    if _wgrad_coop_canonical_ext is not None:
        return _wgrad_coop_canonical_ext
    if _wgrad_coop_canonical_load_failed:
        return None
    rank = _get_build_env().local_rank
    try:
        _wgrad_coop_canonical_ext = _load_wgrad_coop_canonical_ext()
        if rank == 0:
            print("[gemm_fc1] loaded cooperative+canonical wgrad CUTLASS backend",
                  flush=True)
        return _wgrad_coop_canonical_ext
    except Exception as exc:  # noqa: BLE001
        _wgrad_coop_canonical_load_failed = True
        if rank == 0:
            print(f"[gemm_fc1] cooperative+canonical wgrad load failed, "
                  f"falling back to CuTeDSL: {exc}",
                  file=sys.stderr, flush=True)
        return None


def _get_ext():
    global _ext, _ext_type
    if _ext is not None:
        return _ext

    rank = _get_build_env().local_rank
    force_backend = _get_frozen_env().get("GEMM_FC1_BACKEND", "cutedsl")

    if force_backend == "cutedsl":
        if _try_cutedsl_export():
            try:
                _ext = _load_cutedsl_ext()
                _ext_type = "cutedsl"
                if rank == 0:
                    print(f"[gemm_fc1] loaded CuTeDSL C++ backend", flush=True)
                return _ext
            except Exception as e:
                if rank == 0:
                    print(f"[gemm_fc1] CuTeDSL load failed, falling back: {e}",
                          file=sys.stderr, flush=True)
        elif rank == 0:
            print("[gemm_fc1] CuTeDSL export unavailable, falling back to CUTLASS C++",
                  file=sys.stderr, flush=True)

    _ext = _load_cutlass_ext()
    _ext_type = "cutlass"
    if rank == 0:
        print(f"[gemm_fc1] loaded CUTLASS C++ backend", flush=True)
    return _ext


# Round 40: module-level cached callables for the default fast path.
# Bound lazily on first call to avoid touching the extension at import time
# (matters when the module is imported by tooling that never actually runs
# a kernel — e.g. `inspect.signature` consumers).  Once bound, every
# subsequent `gemm_fc1_fwd` / `gemm_fc1_bwd` call skips the `_get_ext()`
# global lookup + attribute lookup of `.gemm_*_fast` (~1-2 µs / call total
# under CPython 3.12, mostly LOAD_GLOBAL into a dict).  At 24 layers /
# step that buys back ~50-100 µs of host-side budget which directly
# trades against the tight ≤0.985× margin the bwd kernel leaves us.
#
# The `te_wgrad` keyword on `gemm_fc1_bwd` stays a no-op and remains
# accepted for drop-in parity with `linear_backward(..., te_wgrad=...)`.
_BOUND_FWD_FAST = None
_BOUND_BWD_FAST = None


def _bind_fast_paths():
    """Bind module-level callables to the loaded extension's fast entries.

    Idempotent: every successful call rebinds against the current `_ext`.
    """
    global _BOUND_FWD_FAST, _BOUND_BWD_FAST
    ext = _get_ext()
    _BOUND_FWD_FAST = ext.gemm_fwd_fast
    _BOUND_BWD_FAST = ext.gemm_bwd_fast


def gemm_fc1_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Forward: out = x @ weight.T"""
    global _inhouse_logged
    if _USE_INHOUSE_JIT:
        if not _inhouse_logged:
            print("[gemm_fc1] using inhouse JIT for fwd/dgrad/wgrad")
            _inhouse_logged = True
        from training_engine_tensor.ops._gemm_inhouse_jit import jit_gemm_fc1_fwd
        shape = x.shape
        x_2d = x.contiguous().reshape(-1, shape[-1])
        w = weight.contiguous()
        M, K = x_2d.shape
        N = w.shape[0]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        jit_gemm_fc1_fwd(x_2d, w, out)
        return out.view(*shape[:-1], N)
    if _USE_INHOUSE:
        inhouse = _get_inhouse_ext()
        if inhouse is None:
            raise RuntimeError("[gemm_fc1] GEMM_FC1_BACKEND=inhouse but inhouse AOT "
                               "extension failed to load")
        if not _inhouse_logged:
            print("[gemm_fc1] using inhouse persistent AOT for fwd/dgrad")
            _inhouse_logged = True
        shape = x.shape
        x_2d = x.contiguous().reshape(-1, shape[-1])
        w = weight.contiguous()
        M, K = x_2d.shape
        N = w.shape[0]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        inhouse.gemm_fwd_fast(x_2d, w, out)
        return out.view(*shape[:-1], N)
    fn = _BOUND_FWD_FAST
    if fn is None:
        _bind_fast_paths()
        fn = _BOUND_FWD_FAST
    return fn(x, weight)


# ---------------------------------------------------------------------------
# Backward dispatch.  At import time we read the env vars once (`_USE_COOP_
# CANONICAL_WGRAD`, `_PARALLEL_BWD`) and pick which `gemm_fc1_bwd`
# implementation to expose.  This eliminates the per-call branching that
# Round 39 was carrying:
#
#     if not _USE_COOP_CANONICAL_WGRAD and not _PARALLEL_BWD:
#         return ext.gemm_bwd_fast(...)
#
# Flipping `GEMM_FC1_WGRAD_BACKEND` / `GEMM_FC1_PARALLEL_BWD` mid-process
# was never supported — they were already cached in Round 37 — so this
# lift is a pure performance refinement with the same observable behaviour.
# ---------------------------------------------------------------------------


def _gemm_fc1_bwd_fast(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default fast path (Round 39 + Round 40).

    A single pybind11 call dispatches both dgrad and wgrad inside the
    extension; the cached callable removes the `_get_ext()` indirection
    that the original wrapper had.

    R39 fast-path semantics preserved:
      * `dy.contiguous()` + 3-D→2-D reshape happen **once** in C++.
      * One Python→C++ round trip per `gemm_fc1_bwd` invocation.
      * `d_input` is allocated with `x`'s 3-D shape directly inside C++,
        so the trailing `.view_as(x)` Python call is gone.

    Round 37 fast paths (`gemm_dgrad_fast` / `gemm_wgrad_fast`) are still
    exported individually for instrumentation; they are just no longer on
    the production hot path.
    """
    fn = _BOUND_BWD_FAST
    if fn is None:
        _bind_fast_paths()
        fn = _BOUND_BWD_FAST
    return fn(d_output, x, weight)


def _gemm_fc1_bwd_slow(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward fallback for opt-in dispatcher branches.

    Engaged only when `GEMM_FC1_WGRAD_BACKEND=coop_canonical` (R34) or
    `GEMM_FC1_PARALLEL_BWD=1` (R48 — kept for ablation, default off).
    The control flow mirrors the pre-R40 dispatcher exactly so any
    historical experiment that relied on these env knobs still works.
    """
    ext = _get_ext()

    shape_dy = d_output.shape
    shape_x = x.shape

    dy_2d = d_output.contiguous().reshape(-1, shape_dy[-1])
    x_2d = x.contiguous().reshape(-1, shape_x[-1])
    w = weight.contiguous()

    M = dy_2d.shape[0]
    N_dgrad = w.shape[1]

    d_input = torch.empty(M, N_dgrad, dtype=torch.bfloat16, device=x.device)
    M_wgrad = w.shape[0]
    N_wgrad = w.shape[1]
    d_weight = torch.empty(M_wgrad, N_wgrad, dtype=torch.float32, device=x.device)

    # Optional CUTLASS 3.x native wgrad backend (`coop_canonical`, R34).
    # 5 other backends were archived in R36 (`archive_r30_r35_negative/`).
    coop_canonical_ext = None
    if _USE_COOP_CANONICAL_WGRAD:
        coop_canonical_ext = _get_wgrad_coop_canonical_ext()

    def _do_wgrad():
        if coop_canonical_ext is not None:
            coop_canonical_ext.gemm_wgrad_coop_canonical(
                dy_2d, x_2d, d_weight, _WGRAD_SWIZZLE)
        else:
            ext.gemm_wgrad(dy_2d, x_2d, d_weight)

    if _PARALLEL_BWD:
        global _bwd_side_stream
        if _bwd_side_stream is None or _bwd_side_stream.device != x.device:
            _bwd_side_stream = torch.cuda.Stream(device=x.device)
        cur = torch.cuda.current_stream(x.device)
        _bwd_side_stream.wait_stream(cur)
        with torch.cuda.stream(_bwd_side_stream):
            _do_wgrad()
        ext.gemm_dgrad(dy_2d, w, d_input)
        cur.wait_stream(_bwd_side_stream)
    else:
        ext.gemm_dgrad(dy_2d, w, d_input)
        _do_wgrad()

    return d_input.view_as(x), d_weight


def _gemm_fc1_bwd_inhouse(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward with in-house persistent AOT kernel."""
    global _inhouse_logged
    inhouse = _get_inhouse_ext()
    if inhouse is None:
        raise RuntimeError("[gemm_fc1] GEMM_FC1_BACKEND=inhouse but inhouse AOT "
                           "extension failed to load")
    if not _inhouse_logged:
        print("[gemm_fc1] using inhouse persistent AOT for fwd/dgrad/wgrad")
        _inhouse_logged = True

    shape_dy = d_output.shape
    shape_x = x.shape
    dy_2d = d_output.contiguous().reshape(-1, shape_dy[-1])
    x_2d = x.contiguous().reshape(-1, shape_x[-1])
    w = weight.contiguous()

    M = dy_2d.shape[0]
    N_dgrad = w.shape[1]
    d_input = torch.empty(M, N_dgrad, dtype=torch.bfloat16, device=x.device)
    inhouse.gemm_dgrad_fast(dy_2d, w, d_input)

    M_wgrad = w.shape[0]
    N_wgrad = w.shape[1]
    d_weight = torch.empty(M_wgrad, N_wgrad, dtype=torch.float32, device=x.device)
    inhouse.gemm_wgrad(dy_2d, x_2d, d_weight)

    return d_input.view_as(x), d_weight


def _gemm_fc1_bwd_inhouse_jit(
    d_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    te_wgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward with in-house JIT kernel (all directions)."""
    from training_engine_tensor.ops._gemm_inhouse_jit import (
        jit_gemm_fc1_dgrad, jit_gemm_fc1_wgrad)

    shape_dy = d_output.shape
    shape_x = x.shape
    dy_2d = d_output.contiguous().reshape(-1, shape_dy[-1])
    x_2d = x.contiguous().reshape(-1, shape_x[-1])
    w = weight.contiguous()

    M = dy_2d.shape[0]
    N_dgrad = w.shape[1]
    d_input = torch.empty(M, N_dgrad, dtype=torch.bfloat16, device=x.device)
    jit_gemm_fc1_dgrad(dy_2d, w, d_input)

    M_wgrad = w.shape[0]
    N_wgrad = w.shape[1]
    d_weight = torch.empty(M_wgrad, N_wgrad, dtype=torch.float32, device=x.device)
    jit_gemm_fc1_wgrad(dy_2d, x_2d, d_weight)

    return d_input.view_as(x), d_weight


# Pick which `gemm_fc1_bwd` to expose based on the env state captured at
# import time.  The cached path is the production default; the slow path
# is selected only if either opt-in dispatcher knob is on.
if _USE_INHOUSE_JIT:
    gemm_fc1_bwd = _gemm_fc1_bwd_inhouse_jit
elif _USE_INHOUSE:
    gemm_fc1_bwd = _gemm_fc1_bwd_inhouse
elif _USE_COOP_CANONICAL_WGRAD or _PARALLEL_BWD:
    gemm_fc1_bwd = _gemm_fc1_bwd_slow
else:
    gemm_fc1_bwd = _gemm_fc1_bwd_fast
