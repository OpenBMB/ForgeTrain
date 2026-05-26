"""CPU-only unit tests for the per-segment training-step profiler.

These tests pin the **state machine** of :mod:`training_engine_tensor.
profiling` without touching CUDA or torch. Three surfaces are covered:

  * ``StepProfiler`` lifecycle + records bookkeeping — all GPU calls
    (``torch.cuda.Event``, ``elapsed_time``) are stubbed via a
    monkey-patched ``torch`` namespace because the harness CI has no
    CUDA. The hot path that the train loop pays per non-profiled step
    must remain a single ``self._active`` branch.
    set, output unset, or vice versa) is the single most-likely
    operator mistake after a 35-min cctl run, so the failure must
    surface at construction time, not after data was thrown away.
  * ``summarize_segments`` — pure stats over the records list; covers
    the small-n edge case (n<2 ⇒ std=0) and the bucket-by-name shape
    that the JSON consumer (notebook + post-processing tools) depends
    on.

Tests live alongside ``test_suites_helpers.py`` so they share the
``REPO_ROOT`` path-injection idiom and stay runnable via
``harness run unit``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from training_engine_tensor import profiling  # noqa: E402

# ----------------------------------------------------------------------
# CUDA event stub — emulates ``torch.cuda.Event(enable_timing=True)``
# without requiring CUDA. The stub returns deterministic 1ms-per-mark
# spacing so segment ordering is verifiable in pure Python.
# ----------------------------------------------------------------------


class _FakeEvent:
    """Stand-in for ``torch.cuda.Event(enable_timing=True)``.

    Each new instance gets a strictly increasing virtual timestamp
    (``+1.0`` ms per construction). ``elapsed_time(other)`` returns
    the integer-ms difference between the two virtual stamps. This
    matches the contract the profiler relies on: events are
    monotonically ordered on the stream and ``elapsed_time`` is
    expressed in milliseconds.
    """

    _counter = 0.0

    def __init__(self, enable_timing: bool = False) -> None:
        del enable_timing
        type(self)._counter += 1.0
        self._t = type(self)._counter

    def record(self) -> None:  # pragma: no cover — trivial
        pass

    def elapsed_time(self, other: _FakeEvent) -> float:
        return float(other._t - self._t)


def _patched_torch_cuda():
    """Build a stub ``torch.cuda`` namespace with the ``Event`` symbol."""
    fake = mock.MagicMock()
    fake.Event = _FakeEvent
    return fake


# ----------------------------------------------------------------------
# StepProfiler.disabled — the hot-path no-op contract
# ----------------------------------------------------------------------


class TestDisabledProfiler(unittest.TestCase):
    """Disabled profilers must be a *true* no-op on every method.

    This is the contract the ``run_training_loop`` hot path depends on:
    when neither env var is set, ``begin_step`` / ``mark`` / ``flush``
    are called once per segment per step, but they MUST NOT touch
    ``torch.cuda`` (because the framework guard forbids unconditional
    CUDA imports in pure-CPU unit tests). If any of these methods
    ever creates a CUDA event in the disabled path, the test will
    AttributeError on ``torch.cuda.Event`` not being defined.
    """

    def test_disabled_factory_returns_no_op_instance(self) -> None:
        prof = profiling.StepProfiler.disabled()
        self.assertFalse(prof.enabled)
        self.assertEqual(prof.records, [])

    def test_should_profile_step_always_false_when_disabled(self) -> None:
        prof = profiling.StepProfiler.disabled()
        for step in (-1, 0, 50, 9999):
            self.assertFalse(prof.should_profile_step(step))

    def test_disabled_lifecycle_does_not_touch_cuda(self) -> None:
        # No torch.cuda patch — if the disabled path tried to
        # construct a CUDA event the test would raise (the real
        # ``torch.cuda.Event`` is unavailable in CI).
        prof = profiling.StepProfiler.disabled()
        prof.begin_step(0)
        prof.mark("step_start")
        prof.mark("microbatches")
        prof.mark_deep("mb_fwd")
        prof.flush(0)
        self.assertEqual(prof.records, [])

    def test_disabled_write_returns_none(self) -> None:
        prof = profiling.StepProfiler.disabled()
        self.assertIsNone(prof.write())
        self.assertIsNone(prof.write(step_time_ms_by_step={0: 1.0}))


# ----------------------------------------------------------------------
# StepProfiler enabled lifecycle — segment naming + records shape
# ----------------------------------------------------------------------


class TestEnabledProfiler(unittest.TestCase):
    """Enabled lifecycle: segment name = closing mark; records carry
    one entry per profiled step; out-of-range steps no-op without
    polluting ``records``.
    """

    def setUp(self) -> None:
        # Reset the fake event counter so each test starts from 0.0
        # (otherwise inter-test state would make ``ms`` values
        # non-deterministic).
        _FakeEvent._counter = 0.0

    def test_segment_names_use_closing_mark(self) -> None:
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=2,
            output_path="/tmp/_unused.json",
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            prof.begin_step(0)
            prof.mark("step_start")    # opens timeline (event #1)
            prof.mark("microbatches")  # closes "microbatches" segment (event #2)
            prof.mark("optimizer")     # closes "optimizer" segment (event #3)
            prof.flush(0)

        self.assertEqual(len(prof.records), 1)
        rec = prof.records[0]
        self.assertEqual(rec["step"], 0)
        names = [s["name"] for s in rec["segments"]]
        # First mark opens the timeline; only subsequent marks
        # produce named segments.
        self.assertEqual(names, ["microbatches", "optimizer"])
        # 1 ms spacing per fake event: total spans first→last = 2 ms.
        self.assertAlmostEqual(rec["total_ms"], 2.0)
        # Each segment is exactly 1 ms.
        for seg in rec["segments"]:
            self.assertAlmostEqual(seg["ms"], 1.0)

    def test_out_of_range_step_does_not_record(self) -> None:
        prof = profiling.StepProfiler(
            enabled=True, range_start=10, range_end=12,
            output_path="/tmp/_unused.json",
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            for step in (0, 5, 9, 12, 13):
                prof.begin_step(step)
                prof.mark("step_start")
                prof.mark("microbatches")
                prof.flush(step)
        self.assertEqual(prof.records, [])

    def test_in_range_records_only_in_range_steps(self) -> None:
        prof = profiling.StepProfiler(
            enabled=True, range_start=10, range_end=13,
            output_path="/tmp/_unused.json",
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            for step in (8, 9, 10, 11, 12, 13, 14):
                prof.begin_step(step)
                prof.mark("step_start")
                prof.mark("microbatches")
                prof.flush(step)
        recorded_steps = [rec["step"] for rec in prof.records]
        self.assertEqual(recorded_steps, [10, 11, 12])

    def test_begin_step_resets_partial_state(self) -> None:
        """If the train loop early-exits without ``flush`` (e.g.
        tokenized-input absence), the next ``begin_step`` must clear
        the in-flight events list — otherwise the next profiled step
        would inherit stale events and report nonsense ms values.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=2,
            output_path="/tmp/_unused.json",
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            prof.begin_step(0)
            prof.mark("step_start")
            prof.mark("microbatches")
            # Skip flush — simulate early exit.
            prof.begin_step(1)
            prof.mark("step_start")
            prof.mark("microbatches")
            prof.flush(1)
        self.assertEqual(len(prof.records), 1)
        rec = prof.records[0]
        self.assertEqual(rec["step"], 1)
        # Exactly one segment — the stale step-0 events were dropped.
        self.assertEqual(len(rec["segments"]), 1)


# ----------------------------------------------------------------------
# StepProfiler.mark_deep — per-microbatch sub-segment marks
# ----------------------------------------------------------------------


class TestMarkDeep(unittest.TestCase):
    """Deep mode is opt-in: ``mark_deep`` is a no-op when
    ``self.deep is False`` even on a profiled step. When enabled, it
    records identically to ``mark`` so per-MB sub-segments roll up
    by name in ``summarize_segments``.
    """

    def setUp(self) -> None:
        _FakeEvent._counter = 0.0

    def test_mark_deep_is_noop_when_deep_off(self) -> None:
        """Shallow profile gate runs (``PROFILE_RANGE`` set,
        ``PROFILE_DEEP`` unset) MUST NOT collect mb_* events. Otherwise
        the JSON layout drifts between deep and shallow runs and the
        downstream summary key set becomes unstable.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=2,
            output_path="/tmp/_unused.json", deep=False,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            prof.begin_step(0)
            prof.mark("step_start")
            prof.mark_deep("mb_data")  # no-op
            prof.mark_deep("mb_fwd")   # no-op
            prof.mark_deep("mb_accum") # no-op
            prof.mark("microbatches")
            prof.flush(0)
        rec = prof.records[0]
        names = [s["name"] for s in rec["segments"]]
        # Only the closing top-level ``microbatches`` mark is recorded.
        self.assertEqual(names, ["microbatches"])

    def test_mark_deep_records_when_deep_on(self) -> None:
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=2,
            output_path="/tmp/_unused.json", deep=True,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            prof.begin_step(0)
            prof.mark("step_start")
            prof.mark_deep("mb_data")
            prof.mark_deep("mb_fwd")
            prof.mark_deep("mb_ce")
            prof.mark_deep("mb_bwd")
            prof.mark_deep("mb_accum")
            prof.mark("microbatches")
            prof.flush(0)
        rec = prof.records[0]
        names = [s["name"] for s in rec["segments"]]
        self.assertEqual(names, [
            "mb_data", "mb_fwd", "mb_ce", "mb_bwd", "mb_accum", "microbatches",
        ])

    def test_mark_deep_is_noop_when_step_out_of_range(self) -> None:
        """Even with ``deep=True``, a step outside ``[start, end)``
        must not pay event-creation cost — keeps the per-step gate-run
        overhead bounded for warmup steps.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=10, range_end=12,
            output_path="/tmp/_unused.json", deep=True,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            prof.begin_step(0)
            prof.mark("step_start")
            prof.mark_deep("mb_data")
            prof.mark_deep("mb_accum")
            prof.flush(0)
        self.assertEqual(prof.records, [])

    def test_microbatches_per_step_inferred_from_mb_accum_count(self) -> None:
        """``write`` reports ``microbatches_per_step`` = number of
        closing ``mb_accum`` marks per profiled step. This lets the
        notebook display "N MBs × mean(mb_fwd)=…" without re-deriving
        from the JSON.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=2,
            output_path="/tmp/_unused.json", deep=True,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            for step in (0, 1):
                prof.begin_step(step)
                prof.mark("step_start")
                # 3 simulated MBs per step.
                for _mb in range(3):
                    prof.mark_deep("mb_data")
                    prof.mark_deep("mb_fwd")
                    prof.mark_deep("mb_accum")
                prof.mark("microbatches")
                prof.flush(step)
        self.assertEqual(prof._microbatches_per_step, 3)

    def test_mark_deep_must_close_phase_not_open_it(self) -> None:
        """``mark_deep("mb_X")`` must be placed AFTER phase X completes
        — the segment ms stored under name ``mb_X`` is computed as the
        elapsed time between the *previous* mark and ``mb_X``, so a
        before-phase placement would attribute every phase's ms to the
        WRONG name (forward time recorded as ``mb_ce``, backward time
        recorded as ``mb_accum``, etc.). This test pins the contract by
        verifying that the cumulative timing of the closing-mark
        sequence ``data → fwd → ce → bwd → accum`` mirrors the per-phase
        wall-clock when each phase has a known duration: with the fake
        event stub that ticks once per ``mark_deep``, the segment named
        ``mb_X`` MUST report 1 ms per closing mark (i.e., the time
        between the prior mark and this one). Failing this would
        signal that someone re-introduced the before-phase placement
        bug fixed in the M2-P0.5 round.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=1,
            output_path="/tmp/_unused.json", deep=True,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            prof.begin_step(0)
            # step_start opens the timeline; each subsequent mark_deep
            # closes the just-finished phase.
            prof.mark("step_start")
            # simulate "data load done" — the segment from step_start
            # up to here is named mb_data and should measure data load.
            prof.mark_deep("mb_data")
            prof.mark_deep("mb_fwd")    # forward done
            prof.mark_deep("mb_ce")     # CE done
            prof.mark_deep("mb_bwd")    # backward done
            prof.mark_deep("mb_accum")  # accumulation done
            prof.mark("microbatches")   # MB loop fence
            prof.mark("optimizer")      # optimizer done
            prof.flush(0)
        rec = prof.records[0]
        names_in_order = [s["name"] for s in rec["segments"]]
        # Order MUST be the canonical lifecycle, not scrambled — any
        # rename / reorder breaks the analyzer's deep-table column
        # alignment.
        self.assertEqual(names_in_order, [
            "mb_data", "mb_fwd", "mb_ce", "mb_bwd", "mb_accum",
            "microbatches", "optimizer",
        ])
        # Each segment is exactly 1 ms with the fake event tick stub
        # — confirms the closing mark NAMES the just-elapsed phase.
        for seg in rec["segments"]:
            self.assertAlmostEqual(seg["ms"], 1.0, places=6)

    def test_inconsistent_mb_count_signals_minus_one(self) -> None:
        """If different profiled steps observe different MB counts the
        single ``microbatches_per_step`` field is set to -1 — surfaces
        the inconsistency in the JSON without crashing the run.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=2,
            output_path="/tmp/_unused.json", deep=True,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            for step, mbs in ((0, 3), (1, 2)):
                prof.begin_step(step)
                prof.mark("step_start")
                for _mb in range(mbs):
                    prof.mark_deep("mb_data")
                    prof.mark_deep("mb_accum")
                prof.mark("microbatches")
                prof.flush(step)
        self.assertEqual(prof._microbatches_per_step, -1)


# ----------------------------------------------------------------------
# StepProfiler.write — atomic JSON output + summary enrichment
# ----------------------------------------------------------------------


    def test_write_emits_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "profile.json"
            prof = profiling.StepProfiler(
                enabled=True, range_start=0, range_end=2,
                output_path=str(out_path),
            )
            with mock.patch.object(profiling, "torch") as torch_stub:
                torch_stub.cuda = _patched_torch_cuda()
                for step in (0, 1):
                    prof.begin_step(step)
                    prof.mark("step_start")
                    prof.mark("microbatches")
                    prof.mark("optimizer")
                    prof.flush(step)
            written = prof.write(step_time_ms_by_step={0: 4.2, 1: 5.1})
            self.assertEqual(written, str(out_path))
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["range"], [0, 2])
        self.assertFalse(payload["deep_mode"])
        # ``microbatches_per_step`` is omitted in shallow mode.
        self.assertNotIn("microbatches_per_step", payload)
        self.assertEqual(len(payload["records"]), 2)
        # Host-step time merged into each record.
        self.assertAlmostEqual(payload["records"][0]["host_step_ms"], 4.2)
        self.assertAlmostEqual(payload["records"][1]["host_step_ms"], 5.1)
        # Summary keyed by segment name.
        self.assertIn("microbatches", payload["summary"])
        self.assertIn("optimizer", payload["summary"])
        self.assertEqual(payload["summary"]["microbatches"]["n"], 2)

    def test_write_emits_microbatches_per_step_in_deep_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "profile.json"
            prof = profiling.StepProfiler(
                enabled=True, range_start=0, range_end=1,
                output_path=str(out_path), deep=True,
            )
            with mock.patch.object(profiling, "torch") as torch_stub:
                torch_stub.cuda = _patched_torch_cuda()
                prof.begin_step(0)
                prof.mark("step_start")
                for _mb in range(2):
                    prof.mark_deep("mb_data")
                    prof.mark_deep("mb_fwd")
                    prof.mark_deep("mb_accum")
                prof.mark("microbatches")
                prof.flush(0)
            prof.write()
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["deep_mode"])
        self.assertEqual(payload["microbatches_per_step"], 2)
        # All deep phase names show up in the summary.
        for name in ("mb_data", "mb_fwd", "mb_accum"):
            self.assertIn(name, payload["summary"])

    def test_write_returns_none_when_no_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "profile.json"
            prof = profiling.StepProfiler(
                enabled=True, range_start=10, range_end=12,
                output_path=str(out_path),
            )
            # No begin_step calls → nothing recorded.
            self.assertIsNone(prof.write())
            self.assertFalse(out_path.exists())

    def test_write_is_atomic(self) -> None:
        """The tmp-then-rename dance protects against torn JSON when
        the cctl pod gets killed mid-write. Verify the temp file is
        gone after a successful write — its presence would indicate
        the rename failed and the consumer would read a partial file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "profile.json"
            prof = profiling.StepProfiler(
                enabled=True, range_start=0, range_end=1,
                output_path=str(out_path),
            )
            with mock.patch.object(profiling, "torch") as torch_stub:
                torch_stub.cuda = _patched_torch_cuda()
                prof.begin_step(0)
                prof.mark("step_start")
                prof.mark("microbatches")
                prof.flush(0)
            prof.write()
            self.assertTrue(out_path.exists())
            self.assertFalse((out_path.parent / (out_path.name + ".tmp")).exists())


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# summarize_segments — bucket-by-name + small-n edge cases
# ----------------------------------------------------------------------


    def test_empty_records_returns_empty_dict(self) -> None:
        self.assertEqual(profiling.summarize_segments([]), {})

    def test_buckets_by_segment_name(self) -> None:
        records = [
            {"step": 0, "total_ms": 3.0, "segments": [
                {"name": "a", "ms": 1.0},
                {"name": "b", "ms": 2.0},
            ]},
            {"step": 1, "total_ms": 4.0, "segments": [
                {"name": "a", "ms": 1.5},
                {"name": "b", "ms": 2.5},
            ]},
        ]
        out = profiling.summarize_segments(records)
        self.assertEqual(set(out.keys()), {"a", "b"})
        self.assertEqual(out["a"]["n"], 2)
        self.assertAlmostEqual(out["a"]["mean_ms"], 1.25)
        self.assertAlmostEqual(out["b"]["mean_ms"], 2.25)
        self.assertAlmostEqual(out["a"]["total_ms"], 2.5)
        self.assertAlmostEqual(out["b"]["total_ms"], 4.5)

    def test_small_n_does_not_nan_std(self) -> None:
        records = [
            {"step": 0, "total_ms": 1.0, "segments": [{"name": "a", "ms": 1.0}]},
        ]
        out = profiling.summarize_segments(records)
        self.assertEqual(out["a"]["n"], 1)
        self.assertEqual(out["a"]["std_ms"], 0.0)
        self.assertAlmostEqual(out["a"]["p90_ms"], 1.0)


# ----------------------------------------------------------------------
# Public-surface guard for segment-name constants
# ----------------------------------------------------------------------



    def test_top_level_segment_names_present(self) -> None:
        for sym in (
            "SEG_STEP_START", "SEG_MICROBATCHES", "SEG_LOSS_ALLREDUCE",
            "SEG_REDUCE_SCATTER", "SEG_GRAD_NORM", "SEG_ALLGATHER",
            "SEG_OPTIMIZER",
        ):
            self.assertTrue(
                hasattr(profiling, sym),
                msg=f"profiling.{sym} must be defined for the wire-format SSOT",
            )

    def test_deep_phase_names_present_and_ordered(self) -> None:
        for sym in (
            "DEEP_PHASE_DATA", "DEEP_PHASE_FWD", "DEEP_PHASE_CE",
            "DEEP_PHASE_BWD", "DEEP_PHASE_ACCUM", "DEEP_PHASE_ORDER",
        ):
            self.assertTrue(hasattr(profiling, sym))
        # The order tuple must match the canonical lifecycle.
        self.assertEqual(profiling.DEEP_PHASE_ORDER, (
            profiling.DEEP_PHASE_DATA,
            profiling.DEEP_PHASE_FWD,
            profiling.DEEP_PHASE_CE,
            profiling.DEEP_PHASE_BWD,
            profiling.DEEP_PHASE_ACCUM,
        ))

# ----------------------------------------------------------------------
# StepProfiler.mark_host — host-clock timer (HOST_TIMER opt-in)
# ----------------------------------------------------------------------


class _FakeMonotonicClock:
    """Deterministic stand-in for ``time.perf_counter_ns``.

    Each call ticks the clock by a fixed nanosecond amount so segment
    ms values are exact integers — keeps the test assertions free of
    floating-point noise (``perf_counter_ns`` returns ints in the
    1ns-resolution domain on Linux). Tests inject the bound ``__call__``
    via ``mock.patch.object(profiling.time, "perf_counter_ns", ...)``.
    """

    def __init__(self, *, tick_ns: int = 1_000_000) -> None:
        self._t = 0
        self._tick = int(tick_ns)

    def __call__(self) -> int:
        self._t += self._tick
        return self._t


    def test_mark_host_is_noop_when_host_timer_off(self) -> None:
        """Profile gate runs that only set ``PROFILE_RANGE`` MUST NOT
        record host phases — keeps the JSON layout backwards-compatible
        with v2 consumers that predate the host-timer rollout.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=2,
            output_path="/tmp/_unused.json", host_timer=False,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            prof.begin_step(0)
            prof.mark("step_start")
            prof.mark_host("microbatches")  # no-op
            prof.mark_host("optimizer")     # no-op
            prof.mark("optimizer")
            prof.flush(0)
        rec = prof.records[0]
        self.assertNotIn("host_phases", rec)
        self.assertNotIn("host_total_ms", rec)

    def test_mark_host_is_noop_when_step_out_of_range(self) -> None:
        """Even with ``host_timer=True``, a step outside ``[start, end)``
        must skip ``perf_counter_ns`` calls — keeps warmup steps free
        of the (admittedly cheap) syscall.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=10, range_end=12,
            output_path="/tmp/_unused.json", host_timer=True,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            with mock.patch.object(
                profiling.time, "perf_counter_ns", _FakeMonotonicClock(),
            ):
                prof.begin_step(0)  # out of range
                prof.mark("step_start")
                prof.mark_host("microbatches")
                prof.flush(0)
        self.assertEqual(prof.records, [])
        # Host events cleared; no leftover state pollutes the next step.
        self.assertEqual(prof._host_events, [])
        self.assertIsNone(prof._host_step_open_ns)

    def test_host_phase_names_use_closing_mark(self) -> None:
        """``mark_host(name)`` MUST close the host segment opened by
        either ``begin_step`` (first mark) or the previous
        ``mark_host`` — same closing-mark convention as ``segments``.
        Tests the canonical contract: with a 1ms-per-tick fake clock,
        each named host phase reports exactly 1ms (the tick between
        the prior mark and itself).
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=1,
            output_path="/tmp/_unused.json", host_timer=True,
        )
        clock = _FakeMonotonicClock(tick_ns=1_000_000)  # 1ms per tick
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            with mock.patch.object(profiling.time, "perf_counter_ns", clock):
                prof.begin_step(0)        # captures opening anchor (1ms)
                prof.mark("step_start")
                prof.mark_host("mb_data")  # 2ms — phase = 1ms after anchor
                prof.mark_host("mb_fwd")   # 3ms — phase = 1ms
                prof.mark_host("mb_accum") # 4ms — phase = 1ms
                prof.mark("microbatches")
                prof.flush(0)
        rec = prof.records[0]
        host_phases = rec["host_phases"]
        names = [p["name"] for p in host_phases]
        self.assertEqual(names, ["mb_data", "mb_fwd", "mb_accum"])
        for phase in host_phases:
            self.assertAlmostEqual(phase["ms"], 1.0, places=6)
        # host_total_ms = last_mark - opening_anchor = 4ms - 1ms = 3ms.
        self.assertAlmostEqual(rec["host_total_ms"], 3.0, places=6)

    def test_host_timer_independent_of_deep_mode(self) -> None:
        """Host timer is orthogonal to deep mode: a shallow profile run
        with ``host_timer=True`` still produces ``host_phases`` for
        whatever marks the train loop emits, even without deep CUDA
        events. This isolates the two opt-ins so an operator can
        enable just the host timer to size P2 (dataloader headroom)
        without paying deep-mode CUDA event overhead.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=1,
            output_path="/tmp/_unused.json", deep=False, host_timer=True,
        )
        clock = _FakeMonotonicClock(tick_ns=1_000_000)
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            with mock.patch.object(profiling.time, "perf_counter_ns", clock):
                prof.begin_step(0)
                prof.mark("step_start")
                prof.mark_host("microbatches")
                prof.mark_host("optimizer")
                prof.mark("microbatches")
                prof.mark("optimizer")
                prof.flush(0)
        rec = prof.records[0]
        # Device-side segments unchanged by host timer.
        device_names = [s["name"] for s in rec["segments"]]
        self.assertEqual(device_names, ["microbatches", "optimizer"])
        # Host phases recorded in parallel.
        host_names = [p["name"] for p in rec["host_phases"]]
        self.assertEqual(host_names, ["microbatches", "optimizer"])

    def test_host_phases_absent_when_no_marks_called(self) -> None:
        """If the train loop's profiled step never calls ``mark_host``
        (e.g. only top-level ``mark`` is wired), the record MUST NOT
        carry an empty ``host_phases`` list — keeps the per-record JSON
        shape stable for analyzers that branch on ``"host_phases" in
        rec``.
        """
        prof = profiling.StepProfiler(
            enabled=True, range_start=0, range_end=1,
            output_path="/tmp/_unused.json", host_timer=True,
        )
        with mock.patch.object(profiling, "torch") as torch_stub:
            torch_stub.cuda = _patched_torch_cuda()
            with mock.patch.object(
                profiling.time, "perf_counter_ns", _FakeMonotonicClock(),
            ):
                prof.begin_step(0)
                prof.mark("step_start")
                prof.mark("microbatches")
                prof.flush(0)
        rec = prof.records[0]
        self.assertNotIn("host_phases", rec)
        self.assertNotIn("host_total_ms", rec)

    def test_write_emits_host_summary_when_enabled(self) -> None:
        """When ``host_timer`` is on, ``write`` MUST add ``host_timer:
        True`` and ``host_summary`` to the JSON payload — analyzers
        detect host data via the top-level flag without scanning
        records.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "profile.json"
            prof = profiling.StepProfiler(
                enabled=True, range_start=0, range_end=2,
                output_path=str(out_path), host_timer=True,
            )
            clock = _FakeMonotonicClock(tick_ns=1_000_000)
            with mock.patch.object(profiling, "torch") as torch_stub:
                torch_stub.cuda = _patched_torch_cuda()
                with mock.patch.object(profiling.time, "perf_counter_ns", clock):
                    for step in (0, 1):
                        prof.begin_step(step)
                        prof.mark("step_start")
                        prof.mark_host("microbatches")
                        prof.mark_host("optimizer")
                        prof.mark("optimizer")
                        prof.flush(step)
            prof.write()
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertTrue(payload.get("host_timer"))
        self.assertIn("host_summary", payload)
        self.assertIn("microbatches", payload["host_summary"])
        self.assertIn("optimizer", payload["host_summary"])
        self.assertEqual(payload["host_summary"]["microbatches"]["n"], 2)
        # Per-record host_phases are present too.
        for rec in payload["records"]:
            self.assertIn("host_phases", rec)
            self.assertEqual(
                [p["name"] for p in rec["host_phases"]],
                ["microbatches", "optimizer"],
            )

    def test_write_omits_host_summary_when_host_timer_off(self) -> None:
        """Backwards-compat: shallow / non-host runs MUST NOT emit
        ``host_timer`` or ``host_summary`` keys. The shape stays
        identical to v2 dumps from the M2-P0 round.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "profile.json"
            prof = profiling.StepProfiler(
                enabled=True, range_start=0, range_end=1,
                output_path=str(out_path), host_timer=False,
            )
            with mock.patch.object(profiling, "torch") as torch_stub:
                torch_stub.cuda = _patched_torch_cuda()
                prof.begin_step(0)
                prof.mark("step_start")
                prof.mark("microbatches")
                prof.flush(0)
            prof.write()
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertNotIn("host_timer", payload)
        self.assertNotIn("host_summary", payload)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------



