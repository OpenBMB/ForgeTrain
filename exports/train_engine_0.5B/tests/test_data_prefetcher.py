"""CPU unit tests for the P1-E multi-slot ``DataPrefetcher`` pipeline.

The real prefetcher relies on ``torch.cuda.Stream`` / ``torch.cuda.Event``
for the async H2D copy, which is unavailable on the macOS dev sandbox
without CUDA. To exercise the queue / depth state machine on CPU we
install lightweight context-manager stubs in ``sys.modules`` so the
class methods that touch ``torch.cuda`` run without raising.

These tests pin the following contracts:

* ``depth == 1`` keeps a single in-flight batch and uses
  ``wait_stream`` (legacy bytewise behaviour). The per-batch event
  slot stays ``None``.
* ``depth >= 2`` enqueues ``depth`` batches at construction time and
  records a per-batch event so the consumer can ``wait_event`` on
  exactly the slot being popped.
* ``__next__`` pops FIFO order: the i-th call returns the i-th batch
  produced by the source iterator.
* The queue refills lazily after each ``__next__`` — if the iterator
  is exhausted mid-stream, the consumer drains the remaining batches
  and then raises ``StopIteration``.
* Invalid depths (``< 1`` or ``> _MAX_PREFETCH_DEPTH``) raise
  ``ValueError`` at construction (fail-fast).
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_HAS_TORCH = _can_import("torch")
_NEEDS_TORCH = unittest.skipUnless(_HAS_TORCH, "requires torch")

if _HAS_TORCH:
    import torch  # noqa: E402


class _FakeStream:
    """Drop-in for ``torch.cuda.Stream`` that supports the context
    manager protocol but does nothing on enter/exit."""

    def __init__(self, *_a: object, **_k: object) -> None:
        pass

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *_a: object) -> None:
        return None


class _FakeEvent:
    """Drop-in for ``torch.cuda.Event`` with a no-op ``record``."""

    def __init__(self) -> None:
        self.recorded_on: object | None = None

    def record(self, stream: object | None = None) -> None:
        self.recorded_on = stream


class _FakeDefaultStream:
    """Stand-in for the default CUDA stream used by ``__next__`` to
    assert sync calls (``wait_event`` / ``wait_stream``)."""

    def __init__(self) -> None:
        self.waited_events: list[object] = []
        self.waited_streams: list[object] = []

    def wait_event(self, ev: object) -> None:
        self.waited_events.append(ev)

    def wait_stream(self, st: object) -> None:
        self.waited_streams.append(st)


class _CudaShim:
    """Monkey-patched ``torch.cuda`` namespace exposing just enough
    surface for ``DataPrefetcher`` to run on CPU."""

    def __init__(self, default_stream: _FakeDefaultStream) -> None:
        self._default = default_stream

    def Stream(self, *a, **k) -> _FakeStream:  # noqa: N802
        return _FakeStream(*a, **k)

    def Event(self) -> _FakeEvent:  # noqa: N802
        return _FakeEvent()

    def current_stream(self, _device: str) -> _FakeDefaultStream:
        return self._default

    def stream(self, _stream: object):
        # Re-export the context manager so ``with torch.cuda.stream(...):
        # `` keeps working.
        return _FakeStream()


def _install_cuda_shim() -> tuple[object, _FakeDefaultStream]:
    """Replace ``torch.cuda`` with the shim for the duration of one
    test. Returns ``(saved_cuda, default_stream)`` so the test can
    restore the original and inspect sync activity."""
    saved = torch.cuda
    default = _FakeDefaultStream()
    torch.cuda = _CudaShim(default)  # type: ignore[assignment]
    return saved, default


def _restore_cuda(saved: object) -> None:
    torch.cuda = saved  # type: ignore[assignment]


@_NEEDS_TORCH
class TestDataPrefetcherSingleSlot(unittest.TestCase):
    """Depth=1 keeps legacy single-slot semantics bytewise."""

    def setUp(self) -> None:
        self._saved_cuda, self._default = _install_cuda_shim()

    def tearDown(self) -> None:
        _restore_cuda(self._saved_cuda)

    def _make_batch(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.full((2, 4), i, dtype=torch.long),
            "labels": torch.full((2, 4), i + 100, dtype=torch.long),
            "loss_mask": torch.ones(2, 4),
            "position_ids": torch.arange(4).unsqueeze(0).expand(2, 4),
        }

    def test_depth1_keeps_one_in_flight(self) -> None:
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(3)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=1)
        self.assertEqual(len(pf._queue), 1)  # noqa: SLF001
        # Depth=1 means no per-batch event; we still use wait_stream.
        event, _ = pf._queue[0]  # noqa: SLF001
        self.assertIsNone(event)

    def test_depth1_uses_wait_stream(self) -> None:
        """``__next__`` on depth=1 must call ``wait_stream`` (legacy
        behaviour). ``wait_event`` must NOT be touched."""
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(3)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=1)
        next(pf)
        self.assertEqual(len(self._default.waited_streams), 1)
        self.assertEqual(len(self._default.waited_events), 0)


@_NEEDS_TORCH
class TestDataPrefetcherMultiSlot(unittest.TestCase):
    """Depth>=2 enqueues ``depth`` batches and uses per-batch events."""

    def setUp(self) -> None:
        self._saved_cuda, self._default = _install_cuda_shim()

    def tearDown(self) -> None:
        _restore_cuda(self._saved_cuda)

    def _make_batch(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.full((2, 4), i, dtype=torch.long),
            "labels": torch.full((2, 4), i + 100, dtype=torch.long),
            "loss_mask": torch.ones(2, 4),
        }

    def test_invalid_depth_zero_raises(self) -> None:
        from training_engine_tensor.data import DataPrefetcher

        with self.assertRaises(ValueError):
            DataPrefetcher(iter([self._make_batch(0)]), depth=0)

    def test_invalid_depth_negative_raises(self) -> None:
        from training_engine_tensor.data import DataPrefetcher

        with self.assertRaises(ValueError):
            DataPrefetcher(iter([self._make_batch(0)]), depth=-1)

    def test_depth_above_cap_raises(self) -> None:
        """Depth above ``_MAX_PREFETCH_DEPTH`` rejected at construction."""
        from training_engine_tensor.data import DataPrefetcher
        from training_engine_tensor.data import _MAX_PREFETCH_DEPTH

        with self.assertRaises(ValueError):
            DataPrefetcher(
                iter([self._make_batch(0)]),
                depth=_MAX_PREFETCH_DEPTH + 1,
            )

    def test_depth2_initial_fill(self) -> None:
        """Two batches enqueued at construction; each carries its own
        event slot."""
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(5)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=2)
        self.assertEqual(len(pf._queue), 2)  # noqa: SLF001
        for event, _ in pf._queue:  # noqa: SLF001
            self.assertIsInstance(event, _FakeEvent)

    def test_depth2_fifo_order(self) -> None:
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(5)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=2)
        out0 = next(pf)
        out1 = next(pf)
        out2 = next(pf)
        # Tokens carry the batch index, so FIFO order means tokens[0]=0,
        # tokens[1]=1, tokens[2]=2.
        self.assertEqual(int(out0["tokens"][0, 0]), 0)
        self.assertEqual(int(out1["tokens"][0, 0]), 1)
        self.assertEqual(int(out2["tokens"][0, 0]), 2)

    def test_depth2_uses_wait_event_not_wait_stream(self) -> None:
        """Per-batch events drive synchronisation in multi-slot mode."""
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(4)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=2)
        next(pf)
        next(pf)
        self.assertEqual(len(self._default.waited_events), 2)
        self.assertEqual(len(self._default.waited_streams), 0)

    def test_depth2_queue_refills_after_next(self) -> None:
        """After ``__next__`` pops one, queue length stays at ``depth``
        until the source is exhausted."""
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(5)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=2)
        next(pf)
        self.assertEqual(len(pf._queue), 2)  # noqa: SLF001
        next(pf)
        self.assertEqual(len(pf._queue), 2)  # noqa: SLF001
        next(pf)
        self.assertEqual(len(pf._queue), 2)  # noqa: SLF001

    def test_depth_drains_then_stop(self) -> None:
        """After source exhaustion the queue drains then raises."""
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(3)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=2)
        # Source has 3 batches; depth=2 prefetched 2 of them. Consumer
        # should observe all 3 in order, then StopIteration.
        outs = [next(pf), next(pf), next(pf)]
        with self.assertRaises(StopIteration):
            next(pf)
        for i, batch in enumerate(outs):
            self.assertEqual(int(batch["tokens"][0, 0]), i)

    def test_depth4_initial_fill(self) -> None:
        """At the cap (depth=_MAX_PREFETCH_DEPTH) ``depth`` batches
        enqueued at construction; no overflow."""
        from training_engine_tensor.data import DataPrefetcher
        from training_engine_tensor.data import _MAX_PREFETCH_DEPTH

        batches = [self._make_batch(i) for i in range(10)]
        pf = DataPrefetcher(
            iter(batches), device="cpu", depth=_MAX_PREFETCH_DEPTH,
        )
        self.assertEqual(len(pf._queue), _MAX_PREFETCH_DEPTH)  # noqa: SLF001

    def test_short_source_initial_fill_drains(self) -> None:
        """Source with fewer batches than ``depth`` partially fills
        the queue then drains gracefully."""
        from training_engine_tensor.data import DataPrefetcher

        batches = [self._make_batch(i) for i in range(2)]
        pf = DataPrefetcher(iter(batches), device="cpu", depth=3)
        self.assertEqual(len(pf._queue), 2)  # noqa: SLF001
        next(pf)
        next(pf)
        with self.assertRaises(StopIteration):
            next(pf)


if __name__ == "__main__":
    unittest.main()
