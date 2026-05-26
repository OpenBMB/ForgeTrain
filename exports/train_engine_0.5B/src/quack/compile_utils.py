"""Compatibility shim for quack.compile_utils.make_fake_tensor."""
import cutlass.cute as cute

def make_fake_tensor(dtype, shape, divisibility=1):
    processed_shape = []
    for i, s in enumerate(shape):
        if isinstance(s, int):
            processed_shape.append(s)
        elif i == len(shape) - 1:
            processed_shape.append(cute.sym_int(divisibility=divisibility))
        else:
            processed_shape.append(cute.sym_int())
    align_bytes = max(16, divisibility * (dtype.width // 8))
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        tuple(processed_shape),
        stride_order=tuple(range(len(shape) - 1, -1, -1)),
        assumed_align=align_bytes,
    )
