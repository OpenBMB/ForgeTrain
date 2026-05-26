"""Smem layout atoms for pre-Hopper style tile_to_shape (used by bwd postprocess / SM80 path).

Delegates to Cutlass hopper WGMMA smem layout selection + warpgroup atom wrapper
(aligns with flash_fwd DSL in this repo).
"""
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as hop
from cutlass.utils import LayoutEnum


def get_smem_layout_atom(dtype, major_mode_size):
    return warpgroup.make_smem_layout_atom(
        hop.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, dtype, major_mode_size),
        dtype,
    )
