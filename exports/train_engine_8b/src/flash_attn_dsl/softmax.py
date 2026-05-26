"""Online softmax for the SM90 flash-attention forward kernel."""
import math
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from flash_attn_dsl import utils


@dataclass
class Softmax:
    scale_log2: Float32 = None
    num_rows: cutlass.Constexpr[int] = None
    row_max: object = None
    row_sum: object = None
    prev_row_scale: object = None  # kept for body-loop deferred rescale_O

    def __init__(self, scale_log2, num_rows, row_max=None, row_sum=None):
        self.scale_log2 = scale_log2
        self.num_rows = num_rows
        if row_max is not None:
            self.row_max = row_max
            self.row_sum = row_sum
        else:
            self.row_max = cute.make_fragment(num_rows, Float32)
            self.row_sum = cute.make_fragment_like(self.row_max)
        # Extra fragment caches the previous online_softmax row_scale so
        # the FWD kernel's body loop can defer `rescale_O(acc_O, row_scale)`
        # to AFTER the next QK WGMMA has been issued (the rescale then
        # runs in parallel with the PV WGMMA on the tensor pipe rather
        # than being blocked behind softmax).  On is_first iterations
        # row_scale is 1.0 anyway, so the extra storage costs `num_rows`
        # FP32 registers per consumer thread.
        self.prev_row_scale = cute.make_fragment_like(self.row_max)

    def reset(self):
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)
        self.prev_row_scale.fill(1.0)

    @cute.jit
    def online_softmax(self, acc_S, is_first=False, check_inf=True):
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
            )
            row_max_cur = utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur
            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None)
                row_scale[r] = 1.0
            else:
                row_max_prev = row_max[r]
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
                row_scale[r] = utils.exp2f((row_max_prev - row_max_cur) * scale_log2)
                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r]
                )
            row_max[r] = row_max_cur
            row_sum[r] = acc_S_row_sum
            self.prev_row_scale[r] = row_scale[r]
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale

    @cute.jit
    def finalize(self, final_scale=1.0):
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)
        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            zn = (row_sum[r] == 0.0 or row_sum[r] != row_sum[r])
            row_scale[r] = cute.arch.rcp_approx(row_sum[r] if not zn else 1.0) * final_scale
            rs = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (row_max[r] * scale_log2 + utils.log2f(rs)) * LN2 if not zn else -Float32.inf
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O, row_scale):
        acc_O_mn = utils.make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
