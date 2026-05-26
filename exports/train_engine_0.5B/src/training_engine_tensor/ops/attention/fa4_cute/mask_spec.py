# Causal + seqlen mask for bwd SdP swap_AB=True (M/N transposed acc). No local window / flex / sparse.
from dataclasses import dataclass
from typing import Optional, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import layout_utils
from .seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class AttentionMaskBwdCausal:
    """Fixed-layout causal mask: swap_AB must match SdP_swapAB in the bwd kernel (True for spec)."""

    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    swap_AB: cutlass.Constexpr[bool] = True

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool] = True,
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
    ) -> None:
        _ = (mask_mod, mask_local, batch_idx, head_idx)  # spec: causal only; no flex/local
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape[::-1])
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 1
        COL = 0
        thr_col_offset = tScS_mn[0][COL]
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        assert self.swap_AB
        thr_row_offset = tScS_mn[0][ROW]
        causal_row_offset = (
            seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
        )
        if const_expr(mask_causal):
            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                col0 = t0ScS_mn[0, c][COL]
                row_limit_top = (
                    self.tile_m
                    if col0 >= seqlenk_col_limit and mask_seqlen
                    else col0 - causal_row_offset
                )
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    acc_S_mn[r, c] = (
                        -Float32.inf
                        if t0ScS_mn[r, 0][ROW] < row_limit_top
                        else acc_S_mn[r, c]
                    )
