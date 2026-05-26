"""NCCL / distributed communication for DP and TP parallelism.

Provides process group management and collective operations matching
Megatron's DistributedOptimizer behavior:
- Reduce-scatter for gradient aggregation across DP ranks
- All-gather for param reconstruction after Adam
- All-reduce for grad norm aggregation and loss averaging
"""
from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

__all__ = [
    "init_process_groups",
    "get_dp_group", "get_dp_rank", "get_dp_size",
    "get_tp_group", "get_tp_rank", "get_tp_size",
    "BufferLayout", "allreduce_loss",
]

from . import config
from .parameters import _canonical_buffer_order

_dp_group: Optional[dist.ProcessGroup] = None
_tp_group: Optional[dist.ProcessGroup] = None
_dp_rank: int = 0
_dp_size: int = 1
_tp_rank: int = 0
_tp_size: int = 1
_initialized: bool = False


def init_process_groups(rank: int, world_size: int, tp_size: int = 1) -> None:
    """Initialize DP and TP process groups.

    For LIGHT 8B (TP=2, DP=4): one TP group per 2 consecutive ranks, one DP group spanning TP-local rank 0s.
    For M5+ (TP=2, DP=4): TP groups {0,1},{2,3},{4,5},{6,7};
                           DP groups {0,2,4,6},{1,3,5,7}.
    """
    global _dp_group, _tp_group, _dp_rank, _dp_size, _tp_rank, _tp_size
    global _initialized

    _tp_size = tp_size
    dp_size = world_size // tp_size
    _dp_size = dp_size
    _group_timeout = datetime.timedelta(minutes=60)

    # TP groups: consecutive ranks of size tp_size
    for start in range(0, world_size, tp_size):
        ranks = list(range(start, start + tp_size))
        group = dist.new_group(ranks, timeout=_group_timeout)
        if rank in ranks:
            _tp_group = group
            _tp_rank = rank - start

    # DP groups: ranks with same tp_rank across all TP groups
    for tp_idx in range(tp_size):
        ranks = list(range(tp_idx, world_size, tp_size))
        group = dist.new_group(ranks, timeout=_group_timeout)
        if rank in ranks:
            _dp_group = group
            _dp_rank = ranks.index(rank)

    _initialized = True


def _check_initialized() -> None:
    if not _initialized:
        raise RuntimeError(
            "nccl process groups not initialized; call init_process_groups() first"
        )


def get_dp_group() -> Optional[dist.ProcessGroup]:
    _check_initialized()
    return _dp_group


def get_dp_rank() -> int:
    _check_initialized()
    return _dp_rank


def get_dp_size() -> int:
    _check_initialized()
    return _dp_size


def get_tp_group() -> Optional[dist.ProcessGroup]:
    _check_initialized()
    return _tp_group


def get_tp_rank() -> int:
    _check_initialized()
    return _tp_rank


def get_tp_size() -> int:
    _check_initialized()
    return _tp_size


# ---------------------------------------------------------------------------
# Buffer layout — matches Megatron's ParamAndGradBuffer
# ---------------------------------------------------------------------------

class BufferLayout:
    """Describes the contiguous gradient/param buffer layout.

    Buffer order = reverse of model.named_parameters() (same as
    _canonical_buffer_order). Each param occupies a contiguous slice.
    """

    def __init__(
        self,
        trainable_names: List[str],
        params: Dict[str, torch.Tensor],
        dp_size: int,
        num_rs_buckets: int = 1,
    ):
        self.buffer_order = _canonical_buffer_order(trainable_names)
        self.param_numels: Dict[str, int] = {}
        self.param_offsets: Dict[str, int] = {}

        offset = 0
        for name in self.buffer_order:
            numel = params[name].numel()
            self.param_offsets[name] = offset
            self.param_numels[name] = numel
            offset += numel

        self.total_numel = offset
        # Pad to multiple of dp_size for even reduce-scatter
        remainder = self.total_numel % dp_size
        if remainder != 0:
            self.total_numel += dp_size - remainder

        self.dp_size = dp_size
        self.shard_size = self.total_numel // dp_size

        self.num_rs_buckets = num_rs_buckets
        if num_rs_buckets > 1:
            self._init_rs_buckets(num_rs_buckets)
        else:
            self.bucket_ranges: List[Tuple[int, int]] = [(0, self.total_numel)]
            self.bucket_shard_offsets: List[int] = [0]

    def _init_rs_buckets(self, num_rs_buckets: int) -> None:
        """Split buffer into buckets aligned with backward layer boundaries.

        Bucket 0: output_layer + final_ln + first group of layers (backward order).
        Last bucket also includes embedding + padding.
        """
        num_layers = config.NUM_LAYERS
        layers_per_bucket = num_layers // num_rs_buckets

        bucket_ranges: List[Tuple[int, int]] = []
        prev_end = 0
        for b in range(num_rs_buckets - 1):
            last_layer = num_layers - (b + 1) * layers_per_bucket
            end_name = (f"decoder.layers.{last_layer}"
                        f".self_attention.linear_proj.weight")
            end = self.param_offsets[end_name] + self.param_numels[end_name]
            bucket_ranges.append((prev_end, end))
            prev_end = end
        bucket_ranges.append((prev_end, self.total_numel))

        self.bucket_ranges = bucket_ranges
        self.bucket_shard_offsets: List[int] = []
        offset = 0
        for b_start, b_end in bucket_ranges:
            b_size = b_end - b_start
            assert b_size % self.dp_size == 0, (
                f"Bucket [{b_start}:{b_end}] size {b_size} "
                f"not divisible by dp_size {self.dp_size}"
            )
            self.bucket_shard_offsets.append(offset)
            offset += b_size // self.dp_size
        assert offset == self.shard_size

    def extract_shard(
        self, full_buf: torch.Tensor, dp_rank: int,
    ) -> torch.Tensor:
        """Extract this rank's shard from a full-buffer tensor.

        For bucketed layout, the shard is a concatenation of per-bucket
        mini-shards (rank's portion of each bucket).
        """
        if self.num_rs_buckets <= 1:
            start, end = self.get_shard_range(dp_rank)
            return full_buf[start:end].clone()
        shard = torch.empty(
            self.shard_size, dtype=full_buf.dtype, device=full_buf.device,
        )
        for b_idx, (b_start, b_end) in enumerate(self.bucket_ranges):
            b_shard = (b_end - b_start) // self.dp_size
            buf_off = b_start + dp_rank * b_shard
            sh_off = self.bucket_shard_offsets[b_idx]
            shard[sh_off:sh_off + b_shard].copy_(
                full_buf[buf_off:buf_off + b_shard]
            )
        return shard

    def get_shard_range(self, dp_rank: int) -> Tuple[int, int]:
        """Return (start, end) indices into the buffer for this rank's shard."""
        start = dp_rank * self.shard_size
        end = start + self.shard_size
        return start, end

    def get_shard_param_slices(
        self, dp_rank: int
    ) -> List[Tuple[str, int, int]]:
        """Return list of (param_name, local_start, local_end) for params
        overlapping this rank's shard.

        local_start/local_end are offsets within the shard tensor.
        Dispatches to contiguous or bucketed implementation.
        """
        if self.num_rs_buckets <= 1:
            return self._get_shard_param_slices_contiguous(dp_rank)
        return self._get_shard_param_slices_bucketed(dp_rank)

    def _get_shard_param_slices_contiguous(
        self, dp_rank: int,
    ) -> List[Tuple[str, int, int]]:
        shard_start, shard_end = self.get_shard_range(dp_rank)
        slices: List[Tuple[str, int, int]] = []
        for name in self.buffer_order:
            p_start = self.param_offsets[name]
            p_end = p_start + self.param_numels[name]
            if p_end <= shard_start or p_start >= shard_end:
                continue
            local_start = max(p_start, shard_start) - shard_start
            local_end = min(p_end, shard_end) - shard_start
            slices.append((name, local_start, local_end))
        return slices

    def _get_shard_param_slices_bucketed(
        self, dp_rank: int,
    ) -> List[Tuple[str, int, int]]:
        slices: List[Tuple[str, int, int]] = []
        for b_idx, (b_start, b_end) in enumerate(self.bucket_ranges):
            b_shard = (b_end - b_start) // self.dp_size
            own_start = b_start + dp_rank * b_shard
            own_end = own_start + b_shard
            for name in self.buffer_order:
                p_start = self.param_offsets[name]
                p_end = p_start + self.param_numels[name]
                ov_start = max(p_start, own_start)
                ov_end = min(p_end, own_end)
                if ov_start >= ov_end:
                    continue
                local_start = (self.bucket_shard_offsets[b_idx]
                               + ov_start - own_start)
                local_end = local_start + (ov_end - ov_start)
                slices.append((name, local_start, local_end))
        return slices


# ---------------------------------------------------------------------------
# Distributed collectives
# ---------------------------------------------------------------------------

def allreduce_loss(
    local_loss: torch.Tensor,
    dp_group: dist.ProcessGroup,
    dp_size: int,
) -> float:
    """All-reduce loss across DP ranks and divide by dp_size.

    Matches reference: dist.all_reduce(loss_t, SUM, group=dp_group); loss_t.div_(dp_sz)
    Operates in-place on the input tensor.
    """
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM, group=dp_group)
    local_loss.div_(dp_size)
    return local_loss.item()
