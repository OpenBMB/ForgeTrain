"""Producer/consumer pipeline helpers.

For the fixed (B=4, H=16, N=4096, D=64) shape we use one K pipeline and one V
pipeline, each a TMA-async pipeline with `num_stages` slots, driven by a
single producer thread inside the producer warpgroup. The consumer arrives at
the empty-barrier from exactly one of its 128 threads — matching the
"NoCluster" customization used in FA3/FA4.
"""

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import Boolean, if_generate
from cutlass.pipeline import (
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    PipelineState,
    PipelineUserType,
    pipeline_init_wait,
)


class PipelineStateSimple:
    """Single-counter encoding of (phase, index) for power-of-two and non-power
    stage counts. We don't rely on bit-twiddling — just use divmod, which the
    compiler folds when stages is a Python int constant.
    """

    def __init__(self, stages: int, counter: Int32):
        self._stages = stages
        self._counter = counter

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self._stages, self._counter)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        return self._counter % self._stages

    @property
    def phase(self) -> Int32:
        return self._counter // self._stages

    def advance(self):
        self._counter += 1

    # MLIR <-> Python plumbing so the state can cross `cute.jit` boundaries.
    def __extract_mlir_values__(self):
        return [self._counter.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self._stages, Int32(values[0]))


def make_pipeline_state(kind: PipelineUserType, stages: int) -> PipelineStateSimple:
    if kind is PipelineUserType.Producer:
        # Producer begins with an "empty" buffer that needs no wait but the
        # phase bit already flipped, so first wait succeeds instantly.
        return PipelineStateSimple(stages, Int32(stages))
    elif kind is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    raise ValueError(f"unsupported PipelineUserType: {kind}")


@dataclass(frozen=True)
class TmaPipelineNoCluster(PipelineAsync):
    """PipelineTmaAsync variant that skips the multicast/cluster machinery.
    Only one thread per warp group signals the empty barrier on release, which
    avoids a register/latency hotspot that otherwise costs 2-3% on H100."""

    @staticmethod
    def create(
        barrier_storage: cute.Pointer,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        init_wait: cutlass.Constexpr[bool] = True,
    ):
        full_sync = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8),
            num_stages,
            (PipelineOp.TmaLoad, producer_group),
            tx_count,
        )
        empty_sync = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages,
            num_stages,
            (PipelineOp.AsyncThread, consumer_group),
        )
        if cutlass.const_expr(init_wait):
            pipeline_init_wait()
        return TmaPipelineNoCluster(
            full_sync,
            empty_sync,
            num_stages,
            None,   # producer_mask (no multicast)
            None,   # dst_rank
        )

    def producer_acquire(self, state: PipelineState, try_token: Optional[Boolean] = None):
        """Block until the slot is drained by the consumer, then arm the
        transaction barrier for the TMA."""
        if_generate(
            try_token is None or try_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        self.sync_object_full.arrive(state.index, self.producer_mask)

    def producer_commit(self, state: PipelineState):  # TMA barrier auto-commits
        pass

    def consumer_release(self, state: PipelineState):
        if_generate(
            cute.arch.thread_idx()[0] % 128 == 0,
            lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask),
        )
