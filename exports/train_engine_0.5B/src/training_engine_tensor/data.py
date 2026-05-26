"""Data prefetch utilities for MiniCPM4 0.5B.

Implements async H2D copy on a dedicated CUDA stream for overlapping
data transfer with GPU computation.

:class:`DataPrefetcher` supports a configurable pipeline depth:

* ``depth == 1`` — legacy single-slot behaviour. Exactly one batch is
  in flight on the copy stream at any time; bytewise identical to the
  pre-P1-E implementation.
* ``depth >= 2`` — multi-slot queue. ``depth`` batches are kept in
  flight on the copy stream so that host-side ``next(iter)`` jitter
  (sstable seeks, tokenization, etc.) is decoupled from the GPU's
  forward critical path. The ``__next__`` consumer always pops the
  oldest batch (FIFO) and immediately refills the tail of the queue.

Each in-flight batch carries its own ``torch.cuda.Event`` so the
default stream's ``wait_event`` synchronises on exactly the batch
being consumed, not on the whole copy stream — leaving subsequent
queue entries' H2D copies free to keep racing ahead.

Dicts are returned with keys: tokens, labels, loss_mask, position_ids.
"""

from __future__ import annotations

__all__ = ["DataPrefetcher"]

from collections import deque
from typing import Any

import torch

_MAX_PREFETCH_DEPTH = 4


class DataPrefetcher:
    """Prefetch batches from an iterator to GPU via async H2D copy.

    Uses a dedicated CUDA stream for host-to-device transfers, allowing
    data loading to overlap with computation on the main stream.

    Args:
        data_iter: Source iterator. Yields either a dict (with keys
            ``tokens``/``input_ids``, ``labels``, optional
            ``loss_mask``, ``position_ids``) or a positional tuple
            ``(tokens, labels[, loss_mask])``. Optional ``indexes`` /
            ``last_sample`` keys on dicts are forwarded back to the
            iterator's ``update(...)`` callback for cursor advance.
        device: Target CUDA device (default ``cuda:0``).
        depth: Number of batches to keep simultaneously in flight on
            the H2D copy stream. ``1`` matches the legacy behaviour;
            ``>= 2`` decouples host iteration from GPU forward.
            Capped at ``_MAX_PREFETCH_DEPTH`` to avoid unbounded
            pinned-memory growth.
    """

    def __init__(
        self,
        data_iter: Any,
        device: str = "cuda:0",
        depth: int = 1,
    ) -> None:
        if depth < 1:
            raise ValueError(
                f"DataPrefetcher: depth must be >= 1, got {depth}"
            )
        if depth > _MAX_PREFETCH_DEPTH:
            raise ValueError(
                f"DataPrefetcher: depth {depth} exceeds cap "
                f"{_MAX_PREFETCH_DEPTH}"
            )

        self._iter = iter(data_iter)
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._depth = depth
        # Queue entries: (event_or_None, batch_dict). On depth==1 the
        # event slot is always None — we keep the legacy "wait whole
        # copy stream" semantics. On depth>=2 every entry carries its
        # own event so the consumer syncs on exactly that batch's H2D.
        self._queue: deque[tuple[torch.cuda.Event | None, dict[str, torch.Tensor | None]]] = deque()
        self._exhausted = False

        for _ in range(depth):
            if not self._prefetch_one():
                break

    @staticmethod
    def _extract(
        raw: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Normalize batch to (input_ids, labels, loss_mask, position_ids)."""
        if isinstance(raw, dict):
            tokens = raw.get("tokens", raw.get("input_ids"))
            labels = raw.get("labels")
            if tokens is None or labels is None:
                raise ValueError(
                    f"Batch dict missing tokens/labels, keys={list(raw.keys())}"
                )
            loss_mask = raw.get("loss_mask")
            if loss_mask is None:
                loss_mask = torch.ones(tokens.shape, dtype=torch.float32)
            elif loss_mask.dtype != torch.float32:
                loss_mask = loss_mask.float()
            position_ids = raw.get("position_ids")
            return tokens, labels, loss_mask, position_ids

        tokens = raw[0]
        labels = raw[1]
        loss_mask = (
            raw[2].float()
            if len(raw) > 2
            else torch.ones(tokens.shape, dtype=torch.float32)
        )
        return tokens, labels, loss_mask, None

    def _prefetch_one(self) -> bool:
        """Pull one batch from the source iterator and enqueue its H2D copy.

        Returns ``True`` if a batch was enqueued, ``False`` if the
        source iterator was exhausted. On exhaustion sets the
        ``_exhausted`` flag so subsequent calls are no-ops.
        """
        if self._exhausted:
            return False

        try:
            raw = next(self._iter)
            while (
                isinstance(raw, dict)
                and (raw.get("loss_mask", None) is not None)
                and (raw["loss_mask"] == 0).all().item()
            ):
                raw = next(self._iter)
            if isinstance(raw, dict) and hasattr(self._iter, "update"):
                self._iter.update(raw.get("indexes"), raw.get("last_sample"))
        except StopIteration:
            self._exhausted = True
            return False

        tokens, labels, loss_mask, position_ids = self._extract(raw)
        with torch.cuda.stream(self._stream):
            batch: dict[str, torch.Tensor | None] = {
                "tokens": tokens.to(self._device, non_blocking=True),
                "labels": labels.to(self._device, non_blocking=True),
                "loss_mask": loss_mask.to(self._device, non_blocking=True),
                "position_ids": (
                    position_ids.to(self._device, non_blocking=True)
                    if position_ids is not None
                    else None
                ),
            }
            # depth==1: skip the per-batch event; the consumer falls
            # back to ``wait_stream`` (legacy semantics) so behaviour
            # stays bytewise identical to the pre-P1-E path.
            if self._depth == 1:
                event: torch.cuda.Event | None = None
            else:
                event = torch.cuda.Event()
                event.record(self._stream)
        self._queue.append((event, batch))
        return True

    def __iter__(self) -> "DataPrefetcher":
        return self

    def __next__(self) -> dict[str, torch.Tensor | None]:
        """Return the oldest prefetched batch; immediately refill the tail."""
        if not self._queue:
            raise StopIteration("Data iterator exhausted")

        event, batch = self._queue.popleft()
        default_stream = torch.cuda.current_stream(self._device)
        if event is None:
            # Legacy depth==1 path: wait the whole copy stream (matches
            # the single-slot pre-P1-E implementation bytewise).
            default_stream.wait_stream(self._stream)
        else:
            default_stream.wait_event(event)

        # Refill the tail so the queue stays at ``depth`` until the
        # source iterator is exhausted, after which it drains.
        self._prefetch_one()
        return batch
