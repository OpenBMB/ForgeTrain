"""Online softmax that runs on CUDA cores, parallel to WGMMA on tensor cores.

Data model: each consumer thread owns `num_rows` rows of the `acc_S`/`acc_O`
accumulators. Within a row the four columns of the WGMMA atom are spread
across 4 lanes of a quad, so every row reduction is a warp-shuffle of width 4.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from . import utils


@dataclass
class Softmax:
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: object = None
    row_sum: object = None

    def __init__(self, scale_log2: Float32, num_rows: cutlass.Constexpr[int]):
        self.scale_log2 = scale_log2
        self.num_rows = num_rows
        self.row_max = cute.make_fragment(num_rows, Float32)
        self.row_sum = cute.make_fragment_like(self.row_max)

    def reset(self):
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Fold one tile of scores into the running softmax statistics.

        Returns a per-row `row_scale` fragment the caller applies to `acc_O`.
        For `is_first=True` the row_scale is just 1.0 (acc_O is re-initialized
        by the PV WGMMA's `zero_init` flag instead).
        """
        acc_mn = utils.make_acc_mn_view(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        s = self.scale_log2
        for r in cutlass.range_constexpr(cute.size(self.row_max)):
            row_vals = acc_mn[r, None].load()
            # 1) per-thread partial max, seeded with the running max.
            if cutlass.const_expr(is_first):
                m = utils.fmax_reduce(row_vals, init_val=None)
            else:
                m = utils.fmax_reduce(row_vals, init_val=self.row_max[r])
            # 2) reduce across the 4 lanes that own columns of this row.
            m = utils.warp_shuffle_reduce(m, cute.arch.fmax, width=4)
            if cutlass.const_expr(check_inf):
                # Clamp `-inf` rows to a large negative finite value so that
                # `exp2(row_vals*s - m*s)` does not produce NaN. Any score in
                # a real row is bounded well above -1e30, so this is a no-op
                # for non-degenerate rows.
                m = cute.arch.fmax(m, Float32(-1.0e30))
            m_scaled = m * s
            # 3) exponentiate (scale-subtract in the same FMA via exp2 input).
            exp_vals = utils.exp2_vec(row_vals * s - m_scaled)
            if cutlass.const_expr(is_first):
                row_scale[r] = 1.0
                new_sum = utils.fadd_reduce(exp_vals, init_val=None)
            else:
                # correction factor for the previous partial sum
                row_scale[r] = cute.arch.exp2((self.row_max[r] - m) * s)
                new_sum = utils.fadd_reduce(
                    exp_vals, init_val=self.row_sum[r] * row_scale[r]
                )
            self.row_max[r] = m
            self.row_sum[r] = new_sum
            acc_mn[r, None].store(exp_vals)
        return row_scale

    @cute.jit
    def finalize(self, final_scale: Float32 = 1.0):
        """Finish the running sum across the quad, return `1 / row_sum` scaled
        by `final_scale` as the final `row_scale` to apply to `acc_O`.
        Also replaces `row_sum` with LSE (in natural log) for backward.

        For the Phase-1 fixed shape (B=4,H=16,N=4096,D=64, causal or not) the
        construction guarantees every output row has at least one un-masked
        attention score, so `row_sum > 0` on every row. No degenerate-row
        branch is needed here — keeping this branch-free also keeps the
        tracer happy.
        """
        # reduce across 4 lanes
        self.row_sum.store(utils.warp_shuffle_reduce(self.row_sum.load(), utils.add, width=4))
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        LN2 = 0.6931471805599453
        s = self.scale_log2
        for r in cutlass.range_constexpr(cute.size(self.row_sum)):
            rs = self.row_sum[r]
            row_scale[r] = cute.arch.rcp_approx(rs) * final_scale
            # Store LSE (natural log) back into row_sum for backward / lse export.
            self.row_sum[r] = (self.row_max[r] * s + utils.log2_approx(rs)) * LN2
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor):
        acc_mn = utils.make_acc_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_mn, mode=[0])
        for r in cutlass.range_constexpr(cute.size(row_scale)):
            acc_mn[r, None].store(acc_mn[r, None].load() * row_scale[r])
