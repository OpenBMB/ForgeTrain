"""Thin wrappers around warpgroup.wgmma / commit / wait used by the main loop."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup


@cute.jit
def wgmma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
) -> None:
    """Issue all K-iters of a WGMMA matmul, commit the group, and (optionally)
    wait for `wg_wait` in-flight wgmma groups to drain.

    `wg_wait = 0`  -> wait for complete drain (synchronous issue).
    `wg_wait = -1` -> don't wait (caller handles wait_group).
    `wg_wait > 0`  -> keep up to `wg_wait` groups in flight.

    We recreate the mma_atom each call so the ACCUMULATE flag toggles safely
    across multiple WGMMA issues without the MLIR "does not dominate this use"
    diagnostic.
    """
    warpgroup.fence()
    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    nk = cute.size(a.shape[2])
    for k in cutlass.range_constexpr(nk):
        cute.gemm(atom, acc, a[None, None, k], b[None, None, k], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if cutlass.const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)
