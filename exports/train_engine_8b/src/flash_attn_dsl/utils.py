"""Utility functions for the SM90a flash-attention forward DSL kernel.

Helpers used by the SM90 forward kernel (``transpose_smem_view``,
``make_acc_mn_view``, ``make_rs_frgA_layout``, ``get_smem_store_atom_sm90``,
``atomic_add_fp32``, ``store_global_fp32``, ``elem_pointer``) alongside the
simpler primitives (``make_acc_tensor_mn_view`` etc.) used by the rest of the
package.
"""
import math
import operator as _operator
from typing import Type, Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """SM90: convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...)."""
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    """SM90 FP16/BF16: ((2, 2, N/8), MMA_M, MMA_N) -> ((2, 2, 2), MMA_M, (N/16, MMA_N))."""
    if cutlass.const_expr(cute.rank(acc_layout.shape[0]) == 3):
        l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
        rA_mma_view = cute.make_layout(
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
    else:
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            ((l.shape[0], l.shape[2][0]), l.shape[1], l.shape[2][1]),
            stride=((l.stride[0], l.stride[2][0]), l.stride[1], l.stride[2][1]),
        )
    return rA_mma_view


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a smem tensor."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def get_smem_store_atom(dtype) -> cute.CopyAtom:
    """SM90 stmatrix atom for writing from registers to shared memory."""
    return cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
        dtype,
    )


@cute.jit
def warp_reduce(
    val,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
):
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@dsl_user_op
def fmax_op(a, b, *, loc=None, ip=None):
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc, ip=ip,
        )
    )


@cute.jit
def fmax_reduce(x, init_val=None):
    res = cute.make_fragment(x.shape, Float32)
    res.store(x)
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = fmax_op(local_max[0], res[i + 0])
        local_max[1] = fmax_op(local_max[1], res[i + 1])
        local_max[2] = fmax_op(local_max[2], res[i + 2])
        local_max[3] = fmax_op(local_max[3], res[i + 3])
    local_max[0] = fmax_op(local_max[0], local_max[1])
    local_max[2] = fmax_op(local_max[2], local_max[3])
    local_max[0] = fmax_op(local_max[0], local_max[2])
    return local_max[0] if cutlass.const_expr(init_val is None) else fmax_op(local_max[0], init_val)


@cute.jit
def fadd_reduce(x, init_val=None):
    if cutlass.const_expr(init_val is None):
        init_val = Float32.zero
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


@cute.jit
def exp2f(x):
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)


@dsl_user_op
def log2f(a, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_f16x2_f32(a, b, to_dtype: Type, *, loc=None, ip=None):
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16]
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_f16(src: cute.Tensor, dst: cute.Tensor):
    assert cute.size(dst.shape) == cute.size(src.shape)
    assert cute.size(src.shape) % 2 == 0
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
        dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


# Aliases: shorter names used at the kernel call site.
make_acc_mn_view = make_acc_tensor_mn_view
make_rs_frgA_layout = convert_layout_acc_frgA
warp_shuffle_reduce = warp_reduce


# `operator.add` re-exported so softmax can refer to `utils.add` symmetrically
# with `utils.mul` / `utils.sub` used elsewhere in this package.
add = _operator.add


@cute.jit
def exp2_vec(x):
    """Vectorized exp2 — alias of `exp2f` accepting a TensorSSA input."""
    return exp2f(x)


@cute.jit
def log2_approx(x: Float32) -> Float32:
    """log2 via lg2.approx.ftz.f32 — alias of `log2f`."""
    return log2f(x)


def transpose_smem_view(sT: cute.Tensor) -> cute.Tensor:
    """Return a transposed view of a (N, D[, stages]) smem tensor.

    Explicitly named to emphasize that the caller is producing a V^T-style
    view (B-operand MN-major) for the PV WGMMA.
    """
    return transpose_view(sT)


def assume_tensor_aligned(t: cute.Tensor) -> cute.Tensor:
    """Rebuild `t` so each dynamic non-trailing stride is asserted as
    divisible by 128 bits (= 8 elements for bf16).

    Background. `flash_attn_dsl/host.py` constructs the gmem tensors via
    `from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=-1)`
    — this declares the base pointer 16-byte aligned but marks all
    non-trailing strides as fully dynamic (no divisibility metadata).
    When the kernel then slices per-CTA (`mQ[batch_idx, h_q_idx, :, :]`
    and friends), the DSL cannot prove that the resulting base pointer
    is still 16-byte aligned, so it conservatively annotates "16-bit"
    (one-element) alignment on the sliced pointer. The subsequent
    `cpasync.CopyG2SOp(num_bits_per_copy=128)` atom then fails IR
    verification with:

        'cute.copy' op src ptr alignment (16 bits) does not meet
        requirement (128 bits) of atom simt_async_copy<bf16, 128 b>

    Fix. Re-build the tensor with `cute.assume(s, divby=8)` on every
    dynamic stride; this re-asserts the divisibility that's actually
    true for any contiguous `[..., D=128]` BF16 layout (every outer
    stride is a multiple of `D * sizeof(bf16) = 256` bytes, so trivially
    a multiple of 16 bytes).

    No-op for fully static tensors (Python int strides pass through).
    Pure host-side metadata rewrite; emits no device code.
    """
    if t is None:
        return None
    divby = 128 // t.element_type.width  # 8 for bf16, 4 for fp32, 16 for int8
    strides = tuple(
        s if isinstance(s, int) else cute.assume(s, divby=divby)
        for s in t.stride[:-1]
    )
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(t.shape, stride=(*strides, t.stride[-1])),
    )


def get_smem_store_atom_sm90(dtype) -> cute.CopyAtom:
    """Epilogue register->smem copy atom (stmatrix)."""
    return get_smem_store_atom(dtype)


@dsl_user_op
def atomic_add_fp32(a, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """FP32 global-memory atomic add using nvvm.atomicrmw FADD.

    Used by the FWD epilogue to scatter per-row LSE values.  The
    kernel grid is ``(N // M_BLOCK, H_kv, B)`` with an in-CTA q-head
    loop, so each ``(batch_idx, q_head_idx, row_idx)`` tuple is touched
    by exactly one lane in one CTA — the atomic acts as a plain store
    today, but is retained so any future grid that emits multiple
    writers (e.g. split-K) still produces correct LSE.
    """
    nvvm.atomicrmw(
        res=T.f32(),
        op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(a).ir_value(loc=loc, ip=ip),
    )


@dsl_user_op
def store_global_fp32(val, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """FP32 non-atomic global store.

    Escape hatch for the LSE-store knob when atomic-LSU pressure
    starts to show up in NCU.
    """
    llvm.store(Float32(val).ir_value(loc=loc, ip=ip), gmem_ptr.llvm_ptr)


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Pointer to a specific element of `x` by coordinate. Convenience
    wrapper around `x.iterator + cute.crd2idx(coord, x.layout)`.
    """
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


