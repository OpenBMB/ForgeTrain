"""
SM90 WGMMA gemm helper — standalone implementation for forward kernel.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
) -> None:
    """Issue all K-iters of a WGMMA matmul, commit, optionally wait.

    `wg_wait =  0` -> drain all in-flight WGMMA groups (synchronous).
    `wg_wait = -1` -> don't wait (caller handles `warpgroup.wait_group`).
    `wg_wait >  0` -> keep up to `wg_wait` groups in flight.
    """
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if cutlass.const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)


wgmma_gemm = gemm
