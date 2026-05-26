"""
PipelineTmaAsyncNoCluster and PipelineStateSimple.

Only 1 thread per warp group signals the empty barrier during consumer_release,
avoiding perf regression when ClusterShape == 1.
"""

from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, if_generate
from cutlass.pipeline import (
    PipelineAsync, PipelineState, CooperativeGroup,
    pipeline_init_wait, PipelineUserType, PipelineOp,
)


class PipelineStateSimple:
    """Circular buffer state using a single Int32 for both index and phase."""

    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        return self._phase_index // self._stages

    def advance(self):
        self._phase_index += 1

    def __extract_mlir_values__(self):
        return [self._phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    if type is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "invalid PipelineUserType"


@dataclass(frozen=True)
class PipelineTmaAsyncNoCluster(PipelineAsync):
    """
    PipelineTmaAsync variant where only 1 out of 128 threads signals the
    empty barrier during consumer_release. Avoids perf regression when
    ClusterShape == 1.
    """

    @staticmethod
    def create(
        barrier_storage: cute.Pointer,
        num_stages: Int32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        init_wait: cutlass.Constexpr[bool] = True,
    ):
        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.AsyncThread
        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)
        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        dst_rank = None
        producer_mask = None
        if cutlass.const_expr(init_wait):
            pipeline_init_wait()
        return PipelineTmaAsyncNoCluster(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            dst_rank,
        )

    def producer_acquire(self, state, try_acquire_token=None, extra_tx_count=0):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if cutlass.const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count)

    def producer_commit(self, state):
        pass

    def consumer_release(self, state):
        if_generate(
            cute.arch.thread_idx()[0] % 128 == 0,
            lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask),
        )


# Alias: the engine's FWD kernel uses the shorter name
# `TmaPipelineNoCluster`.  Both names refer to the same frozen dataclass;
# importing either works.
TmaPipelineNoCluster = PipelineTmaAsyncNoCluster
