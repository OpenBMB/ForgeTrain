"""Small DSL utilities used by the Hopper flash-attention kernel.

These helpers paraphrase common shapes/ops needed across softmax, epilogue
and the main loop. Kept deliberately short so that `flash_fwd.py` only has
structural code.
"""

import math
import operator as _operator
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.arch import llvm, nvvm
from typing import Optional, Tuple


@cute.jit
def warp_shuffle_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """XOR-shuffle reduction over a lane group of `width` lanes."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        frag = cute.make_fragment(val.shape, val.dtype)
        frag.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            frag[i] = warp_shuffle_reduce(frag[i], op, width)
        return frag.load()
    num_steps = int(math.log2(width))
    for s in cutlass.range_constexpr(num_steps):
        offset = 1 << s
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=offset))
    return val


@cute.jit
def fmax_reduce(x: cute.TensorSSA, init_val=None) -> Float32:
    """Per-thread max over a vector, optionally initialized with `init_val`."""
    frag = cute.make_fragment(x.shape, Float32)
    frag.store(x)
    n = cute.size(x.shape)
    acc = frag[0] if cutlass.const_expr(init_val is None) else cute.arch.fmax(frag[0], init_val)
    for i in cutlass.range_constexpr(1, n):
        acc = cute.arch.fmax(acc, frag[i])
    return acc


@cute.jit
def fadd_reduce(x: cute.TensorSSA, init_val=None) -> Float32:
    """Per-thread sum over a vector. `init_val` is added as a seed when given."""
    if cutlass.const_expr(init_val is None):
        seed = Float32.zero
    else:
        seed = Float32(init_val)
    return x.reduce(cute.ReductionOp.ADD, seed, 0)


@cute.jit
def exp2_vec(x: cute.TensorSSA) -> cute.TensorSSA:
    """Element-wise exp2 over a per-thread register tensor, returned as SSA."""
    frag = cute.make_fragment(x.shape, Float32)
    frag.store(x)
    for i in cutlass.range_constexpr(cute.size(x.shape)):
        frag[i] = cute.arch.exp2(frag[i])
    return frag.load()


@cute.jit
def log2_approx(x: Float32) -> Float32:
    # Float32.log2 doesn't exist in the DSL, but cute.arch.log2f does via the
    # math module. Use that — it's a straight wrapper over lg2.approx.ftz.f32.
    return cute.math.log2(x, fastmath=True)


def make_mn_view_layout(acc_layout: cute.Layout) -> cute.Layout:
    """Reshape SM90 WGMMA accumulator layout ((2,2,V), M_tile, N_tile)
    into a per-thread (row, col) view:
        ((2, M_tile), (2, V, N_tile)).
    """
    cm = cute.make_layout(acc_layout.shape)
    new_shape = (
        (cm.shape[0][1], cm.shape[1]),
        (cm.shape[0][0], *cm.shape[0][2:], cm.shape[2]),
        *cm.shape[3:],
    )
    new_stride = (
        (cm.stride[0][1], cm.stride[1]),
        (cm.stride[0][0], *cm.stride[0][2:], cm.stride[2]),
        *cm.stride[3:],
    )
    return cute.composition(acc_layout, cute.make_layout(new_shape, stride=new_stride))


def make_acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, make_mn_view_layout(acc.layout))


def tiled_copy_1d(
    dtype, num_threads: int, num_copy_elems: int = 1,
) -> cute.TiledCopy:
    """Make a 1-D TiledCopy with `num_threads` cooperatively loading
    `num_copy_elems` per thread.  Used to spread sLSE / sdPsum across
    a small thread group (8 quads) so each thread only does a single
    LDS, with values broadcast across the quad via warp shuffle.

    Ported from quack/copy_utils.tiled_copy_1d.
    """
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(
        copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def shuffle_sync(
    value, offset, width: cutlass.Constexpr[int] = 32,
):
    """Warp-level shuffle.  Broadcasts `value` from lane
    `(self_lane ^ offset)` (or similar pattern depending on the
    underlying PTX).  Works on any 32-bit-multiple type.

    Ported from fa4_cute/utils.shuffle_sync.
    """
    assert value.width % 32 == 0, (
        "value type must be a multiple of 32 bits")
    mask = 32 - width
    clamp = 32 - 1
    mask_and_clamp = (mask << 8) | clamp
    val = cute.make_rmem_tensor(
        cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@cute.jit
def get_stat(
    tSrS: cute.Tensor, row: cutlass.Constexpr,
    lane: Int32, shuffle: cutlass.Constexpr[bool],
):
    """Return acc_S row r's broadcast statistic (LSE / dPsum).

    `shuffle=False`: tSrS is a flat (rows_per_thread,) fragment from
    `mma_partition_C_vec` + `load_s2r`.  Direct register index.

    `shuffle=True`: tSrS is the (((vec, 1), 1),) fragment after
    `tiled_copy_1d.partition_S` + `group_modes`.  Each row's value
    lives in lane (off*4 + (self_lane % 4)) of the 8-quad shuffle
    group, register slot `idx0 + idx1*vecsize`.

    Ported from fa4_cute/flash_bwd_sm90._get_stat.
    """
    if const_expr(not shuffle):
        return tSrS[row]
    vecsize = cute.size(tSrS, mode=[0, 0])
    idx0, off, idx1 = cute.idx2crd(
        row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
    return shuffle_sync(
        tSrS[idx0 + idx1 * vecsize],
        offset=off * 4 + (lane % 4),
    )


@cute.jit
def load_s2r(src: cute.Tensor) -> cute.Tensor:
    """Vectorized smem→register copy.

    Allocates an rmem fragment matching `src`'s shape + element type
    and runs `cute.autovec_copy` so the compiler issues LDS.128 /
    LDS.64 wide loads where alignment allows.  Used in the bwd softmax
    path to batch-load LSE / dPsum vectors before the row loop, so
    the LDS chain starts early and overlaps with prior wgmma work.
    """
    dst = cute.make_fragment(src.shape, src.element_type)
    cute.autovec_copy(src, dst)
    return dst


@cute.jit
def mma_partition_C_vec(
    sVec: cute.Tensor,
    thr_mma,
    expand_shape: cutlass.Constexpr,
    is_colvec: cutlass.Constexpr,
) -> cute.Tensor:
    """Partition a (tile_dim, stage) smem vector into the per-thread
    register layout dictated by `thr_mma`'s C operand.

    Ported from quack/layout_utils.mma_partition_C_vec.

    The trick:
      1. Expand the vector into a 2-D tile via a 0-stride broadcast
         on the orthogonal axis (so smem footprint stays the same
         but the layout matches the mma_C tile shape).
      2. Run `thr_mma.partition_C` to figure out which row indices
         each thread owns under the wgmma C-frag distribution.
      3. mn-view the result and drop the broadcast axis.

    The returned (rows_per_thread, stage) view feeds straight into
    `load_s2r` for a single LDS per thread, replacing N scalar
    `sLSE_iter[row_local]` reads.

    `is_colvec=True` for normal layout (LSE is a column vector along
    the M axis).  `is_colvec=False` under SdP_swapAB (M and N axes
    are swapped, LSE becomes a row vector).
    """
    assert cute.rank(sVec) == 2, "sVec must be (tile_dim, stage)"
    assert sVec.stride[0] == 1, "sVec must be unit-strided in dim 0"
    stage = sVec.shape[1]
    if const_expr(is_colvec):
        shape = (sVec.shape[0], expand_shape, stage)
        stride = (1, 0, sVec.stride[1])
    else:
        shape = (expand_shape, sVec.shape[0], stage)
        stride = (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(
        sVec.iterator, cute.make_layout(shape, stride=stride)
    )
    tC_sVec = make_acc_mn_view(thr_mma.partition_C(sVec_mma))
    if const_expr(is_colvec):
        return tC_sVec[None, 0, None]
    return tC_sVec[0, None, None]


@cute.jit
def make_rs_frgA_layout(acc_layout: cute.Layout) -> cute.Layout:
    """Convert an FP32 QK WGMMA accumulator layout into a layout acceptable as
    the A operand of the PV WGMMA (RS mode) after cast to FP16.

    For SM90 FP16/BF16: ((2,2,N/8), MMA_M, MMA_N) →
                         ((2,2,2), MMA_M, (N/16, MMA_N)).
    """
    l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
    return cute.make_layout(
        (
            (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
            l.shape[1],
            (l.shape[0][2][1], l.shape[2]),
        ),
        stride=(
            (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
            l.stride[1],
            (l.stride[0][2][1], l.stride[2]),
        ),
    )


def transpose_smem_view(sT: cute.Tensor) -> cute.Tensor:
    """Return a `V^T`-style view of a (N, D[, stages]) smem tensor."""
    new_shape = (sT.shape[1], sT.shape[0], *sT.shape[2:])
    order = (1, 0, *range(2, cute.rank(sT)))
    return cute.composition(sT, cute.make_ordered_layout(new_shape, order=order))


def get_smem_store_atom_sm90(dtype, transpose=False, major_mode_size=None):
    """Epilogue register→smem copy atom (uses stmatrix for FP16/BF16).

    `transpose=True` issues `stmatrix.trans` so a register fragment whose
    layout matches a SwapAB mma C-output (i.e. transposed relative to the
    smem destination) can be stored without a register transpose pass.

    `major_mode_size` controls the atom's `num_matrices` (4 / 2 / 1) — must
    align with the smem destination's contiguous extent in the
    transpose-major direction. None → 4 (the common 16-wide case).
    """
    if major_mode_size is None or major_mode_size % 16 == 0:
        num_matrices = 4
    elif major_mode_size % 8 == 0:
        num_matrices = 2
    else:
        num_matrices = 1
    return cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(
            transpose=transpose, num_matrices=num_matrices),
        dtype,
    )


# ── Phase 1.7 component 1: register-fragment FP32 → FP16/BF16 cvt ─────
# The 4x faster way to convert wgmma FP32 acc fragments to FP16/BF16
# than `acc.load().to(fp16)`. Uses the dedicated `cvt.rn.f16x2.f32` PTX
# (or its bf16 cousin) which packs 2 f32 → 2 f16 in a single 32-bit reg.

@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32,
    to_dtype, *, loc=None, ip=None,
):
    """Pack two FP32 values into one Int32 holding two FP16/BF16."""
    assert to_dtype in (cutlass.Float16, cutlass.BFloat16), (
        "to_dtype must be Float16 or BFloat16")
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip),
             Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}"
            f".f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def cvt_f16(src: cute.Tensor, dtype):
    """Convert an FP32 cute.Tensor to a new FP16/BF16 fragment via cvt.rn.

    Faster than `dst.store(src.load().to(dtype))` for wgmma-shaped
    fragments: produces ~N/2 PTX cvt.rn.f16x2.f32 ops instead of N
    individual cvt + reg-shuffle.
    """
    assert src.element_type is Float32, "src must be Float32"
    assert dtype in (cutlass.Float16, cutlass.BFloat16), (
        "dst dtype must be Float16 or BFloat16")
    dst = cute.make_fragment(src.shape, dtype)
    assert cute.size(src.shape) % 2 == 0, "src element count must be even"
    dst_i32 = cute.recast_tensor(dst, Int32)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
        dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dtype)
    return dst


# ── Phase 1.7 component 2: swizzle / position-independent partition ──
# These are required so we can build a transpose-aware register→smem
# copy that targets a smem tile whose declared shape doesn't match the
# mma's C output (the SwapAB case: mma C is (tile_n, tile_m), smem is
# (tile_m, tile_n)).  Ported from quack.copy_utils so we don't have to
# pull in the full quack stack at runtime.

def swizzle_int(ptr_int: Int32, b: int, m: int, s: int) -> Int32:
    bit_msk = (1 << b) - 1
    yyy_msk = bit_msk << (m + s)
    return ptr_int ^ ((ptr_int & yyy_msk) >> s)


def swizzle_ptr(ptr: cute.Pointer):
    """Apply the pointer's static swizzle to its base address.

    The resulting pointer is "position-independent": swizzling has been
    folded into the address, so subsequent partition_D/partition_S calls
    produce a layout that no longer carries the swizzle composition.
    """
    swz = ptr.type.swizzle_type
    ptr_int = swizzle_int(
        ptr.toint(), swz.num_bits, swz.num_base, swz.num_shift)
    return cute.make_ptr(
        ptr.dtype, ptr_int, ptr.memspace, assumed_align=ptr.alignment)


def as_position_independent_swizzle_tensor(
    tensor: cute.Tensor,
) -> cute.Tensor:
    """Re-cast `tensor` so its swizzle lives at element granularity.

    The resulting tensor has the same element type and addressable
    layout, but the swizzle lives in the layout (not the iterator),
    matching what `partition_D` returns once we strip the iterator's
    swizzle via `swizzle_ptr`.
    """
    outer = tensor.layout
    width = tensor.element_type.width
    swizzle_type = tensor.iterator.type.swizzle_type
    inner = cute.make_swizzle(
        swizzle_type.num_bits, swizzle_type.num_base, swizzle_type.num_shift)
    new_layout = cute.recast_layout(
        width, 8, cute.make_composed_layout(
            inner, 0, cute.recast_layout(8, width, outer)),
    )
    return cute.make_tensor(
        cute.recast_ptr(tensor.iterator, dtype=tensor.element_type),
        new_layout,
    )


def partition_D_position_independent(
    thr_copy, tensor: cute.Tensor,
) -> cute.Tensor:
    """`thr_copy.partition_D(tensor)` but stripped of swizzle composition.

    Use this when partitioning a smem destination whose shape is the
    transpose of the mma C output: the standard `partition_D` path
    raises a size mismatch because the swizzled layout still carries
    the original mma C shape; this version uses the
    swizzle-on-the-pointer trick so the resulting partition has the
    correct shape and addresses the same physical smem cells.
    """
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_D(tensor).iterator),
        thr_copy.partition_D(
            as_position_independent_swizzle_tensor(tensor)).layout,
    )


def partition_S_position_independent(
    thr_copy, tensor: cute.Tensor,
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_S(tensor).iterator),
        thr_copy.partition_S(
            as_position_independent_swizzle_tensor(tensor)).layout,
    )


@dsl_user_op
def cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    """`cute.copy` with optional dtype cast and per-thread retile.

    Mirrors quack.copy_utils.cvt_copy: if `src.element_type` differs
    from `dst.element_type`, a fresh rmem fragment of the dst dtype is
    materialised first (using cvt_f16 for FP32→FP16/BF16, otherwise
    the generic `.load().to(...)` path).
    """
    assert isinstance(src.iterator, cute.Pointer) and (
        src.memspace == cute.AddressSpace.rmem)
    if const_expr(src.element_type != dst.element_type):
        if (src.element_type is Float32 and
                dst.element_type in (cutlass.Float16, cutlass.BFloat16)):
            src = cvt_f16(src, dst.element_type)
        else:
            src_cvt = cute.make_rmem_tensor_like(src, dst.element_type)
            src_cvt.store(src.load().to(dst.element_type))
            src = src_cvt
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, loc=loc, ip=ip, **kwargs)


def get_smem_store_C(
    tiled_mma: cute.TiledMma,
    sC: cute.Tensor,
    tidx: Int32,
    *,
    transpose: bool = False,
    position_independent: bool = False,
    major_mode_size: Optional[int] = None,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    """Build a register→smem copy callable matched to a tiled mma's C.

    Returns `(copy_fn, thr_copy, tRS_sC)` where `copy_fn(src, dst_idx)`
    issues a transpose-aware stmatrix store of `src` into the
    `dst_idx`-th stage of `sC`.

    * `transpose=True` issues `stmatrix.trans` so a SwapAB acc fragment
      can land in a shape-unswapped smem tile in one PTX op per warp.
    * `position_independent=True` uses partition_D_position_independent
      to bypass the size-mismatch check when sC is the transpose of the
      mma C-shape (e.g. mma C = (tile_n, tile_m) but sC = (tile_m,
      tile_n)).
    """
    dtype = sC.element_type
    copy_atom = get_smem_store_atom_sm90(
        dtype, transpose=transpose, major_mode_size=major_mode_size)
    tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if not position_independent:
        tRS_sC = thr_copy.partition_D(sC)
    else:
        tRS_sC = partition_D_position_independent(thr_copy, sC)

    def copy_fn(src: cute.Tensor, dst_idx: Optional[Int32] = None,
                **new_kwargs):
        dst_tensor = tRS_sC if const_expr(dst_idx is None) else (
            tRS_sC[None, None, None, dst_idx])
        cvt_copy(tiled_copy, src, dst_tensor, retile=True, **new_kwargs)

    return copy_fn, thr_copy, tRS_sC


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer, gmem_ptr: cute.Pointer, store_bytes: int,
    *, loc=None, ip=None,
) -> None:
    """Issue cp.reduce.async.bulk (smem→gmem FP32 atomic add) via inline PTX."""
    from cutlass import Int32
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """FP32 global-memory atomic add using nvvm.atomicrmw FADD."""
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )


@dsl_user_op
def store_global_fp32(val: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """FP32 global-memory store via PTX st.global."""
    llvm.inline_asm(
        T.f32(),
        [gmem_ptr.llvm_ptr, Float32(val).ir_value()],
        "st.global.f32 [$1], $2;\n mov.f32 $0, $2;",
        "=f,l,f",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get a pointer to a specific element of a tensor by coordinate."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


# Re-export stdlib ops so the softmax module can refer to them.
add = _operator.add
