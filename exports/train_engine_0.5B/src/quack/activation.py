"""Stub for quack.activation — provides sub_packed_f32x2 used by fa4_cute/utils.py.

Only the import needs to succeed; the backward kernel never calls exp2_packed
which is the sole consumer of sub_packed_f32x2.
"""

import cutlass.cute as cute


def sub_packed_f32x2(a, b, *, rnd=None):
    """Packed f32x2 subtraction: (a0-b0, a1-b1).

    Thin wrapper around cute.arch.add_packed_f32x2 with negated operands.
    """
    neg_b = (cute.arch.neg(b[0]), cute.arch.neg(b[1]))
    kwargs = {"rnd": rnd} if rnd is not None else {}
    return cute.arch.add_packed_f32x2(a, neg_b, **kwargs)
