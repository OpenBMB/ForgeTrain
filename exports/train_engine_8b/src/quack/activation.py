"""Stub for ``quack.activation`` — provides ``sub_packed_f32x2`` used by
the FP32 packed-pair softmax path.

Only the fast_exp2_pair path (SM100) calls this; the SM90 backward kernel
does not.
"""

import cutlass.cute as cute


@cute.jit
def sub_packed_f32x2(a, b):
    return (a[0] - b[0], a[1] - b[1])
