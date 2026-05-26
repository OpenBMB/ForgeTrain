"""Lightweight per-segment training-step profiler.

Per ``recommendation.md`` for the 16-card scenario, the first M2 step
must start with a real cctl-side breakdown of where wall-clock time is
spent inside one training step — otherwise any "MFU optimization" rides
on intuition and gets buried by node noise (~0.5–1pp per single-shot
``loss-gate-200`` run on DP=16 GBS=160 grad_accum=1).

This module ships a CUDA-event-based profiler that:

* is **opt-in** via two env vars (``PROFILE_RANGE`` + ``PROFILE_OUTPUT``)
  — when either is unset, every method is a true no-op and the hot path
  in :mod:`train_loop` is bitwise identical to the pre-profiler code
  (no event allocation, no list append, no conditional branch beyond a
  cheap ``self.enabled`` check);
* uses ``torch.cuda.Event(enable_timing=True)`` markers, NOT host
  ``time.time()`` boundaries — events are recorded asynchronously on
  the current CUDA stream (~1 µs per record), so a 6-segment split adds
  ~6 µs of launch overhead per step (negligible against a ~350ms step
  on 16 cards);
* defers all ``elapsed_time()`` reads to **after** ``cuda.synchronize()``
  at step end (the train loop already does this for its own timing), so
  the breakdown numbers do not introduce any additional sync;
* writes a single JSON artifact at end-of-range listing per-step
  per-segment timings + summary stats (mean / median / p90 / std) so
  the reviewer can act without nsys-ui — useful as a cheap first pass
  before the heavier ``nsys profile -t cuda,nvtx,osrt`` flow.

The two env-var contract is intentional:

  ``PROFILE_RANGE=START,END``    —  inclusive START, exclusive END;
                                    only steps with ``step_idx`` in this
                                    half-open range are profiled.
  ``PROFILE_OUTPUT=<path>``      —  destination JSON. May be relative;
                                    the train loop resolves relative
                                    paths against the current cwd.

When ``PROFILE_RANGE`` is set but ``PROFILE_OUTPUT`` is not (or vice
versa), the profiler raises ``ValueError`` at construction time — both
must be wired together or neither, never half-on (which would silently
collect events in memory but never persist them, defeating the point).

The default constructor is ``StepProfiler.disabled()`` and is what
``train_loop.run_training_loop`` uses on the no-env-vars-set hot path.

Deep mode
---------
The default 6-segment split treats the whole microbatch loop as one
black box (``microbatches`` segment). To investigate *inside* the loop
without resorting to ``nsys profile``, set ``PROFILE_DEEP=1`` together
with the standard ``PROFILE_RANGE`` / ``PROFILE_OUTPUT`` pair. In deep
mode the train loop additionally calls ``profiler.mark_deep(name)``
**after** each microbatch sub-phase completes (closing-mark
convention, identical to the top-level ``mark`` call sites in
``train_loop.py``), producing per-MB sub-segments named ``mb_data`` /
``mb_fwd`` / ``mb_ce`` / ``mb_bwd`` / ``mb_accum``. So the
``mb_data`` segment measures *data load* duration (between the prior
``mark`` and the closing ``mark_deep("mb_data")``), the ``mb_fwd``
segment measures *forward* duration, and so on. This ordering is
mandatory — calling ``mark_deep`` *before* the phase yields a name
that does not match its time interval (each segment ends up labelled
with the *next* phase's name), which silently inverts the priority
ranking on the resulting report and was the root cause of the
M2-P0.5 first-attempt mis-attribution (forward 100ms reported as
``mb_ce``, backward 200ms reported as ``mb_accum``).

``mark_deep`` is a true no-op when deep mode is OFF (a single boolean
``and`` short-circuit in addition to the existing ``self._active``
guard) — gate runs that don't set ``PROFILE_DEEP`` pay zero overhead
even with ``PROFILE_RANGE`` set. With deep mode ON, each profiled step
records ~5 extra ``cudaEventRecord`` calls per microbatch (~5 events
total per step at GBS=160 / DP=16 / mbs=10 → 1 MB), adding negligible
host-launch overhead per step.

Important: in deep mode the top-level ``microbatches`` segment is
**degenerate** — its closing ``mark`` runs immediately after the
final ``mark_deep("mb_accum")``, so its measured ms is just the
trivial Python gap between the two marks (~0–0.05 ms). Consumers
should treat ``microbatches`` as a sentinel in deep mode and use
``sum(mb_*)`` (or ``host_step_ms``) as the denominator when
expressing per-MB phases as a percentage of the step. The JSON
output carries ``deep_mode: bool`` and ``microbatches_per_step:
int | None`` at the top level so post-processing tools can branch
on it without re-parsing env vars.

Host-side timer
---------------
M2-P0.5 surfaced a measurement gap: CUDA events resolve elapsed-time
on the device-side stream, so any host-blocking call (``next(dl)``
which hits the modelbest_sdk Python critical section, ``loss.item()``
which forces a device→host copy, etc.) is invisible to ``mark`` /
``mark_deep`` — the device-side timeline collapses host-blocking ms
into the *next* segment's ms because the recorded event is enqueued
asynchronously and the host has already moved past the target line by
the time the GPU executes the marker. Net effect: a profile that
looks like "mb_data=0.01ms, mb_accum=203ms" really means "host blocked
~17ms inside next(dl), then accum was actually ~6ms but its ms got
fattened by the bwd-tail collapsed into it".

The host timer fixes this by additionally recording ``time.perf_counter_ns``
samples at every ``mark_host(name)`` call. ``mark_host`` is a true
no-op when ``host_timer`` is off (single ``and`` short-circuit on top
of the existing ``self._active`` guard), so gate runs that only set
``PROFILE_RANGE`` keep the bitwise-identical hot path. Enable per-step
host timing by setting ``HOST_TIMER=1`` together with ``PROFILE_RANGE``
+ ``PROFILE_OUTPUT`` — isolated ``HOST_TIMER`` raises ``ValueError``
at construction time (same half-config trap as ``PROFILE_DEEP``).

Each ``mark_host(name)`` records ~50ns (perf_counter_ns is a vDSO
syscall on Linux); 11 marks per step at 16-card scale add ~550ns ≈
0.0002% of a 350ms step — well below measurement noise.

The on-disk JSON gains two optional fields when ``host_timer`` is
enabled:

  * ``host_timer: true`` at the top level (so consumers can detect
    presence of host phase data without scanning records),
  * each record gains a ``host_phases: list[{name, ms}]`` mirroring
    the ``segments`` shape but measured on the host clock,
  * the top-level ``host_summary`` aggregates host phases by name
    (mean / p50 / p90 / std / total) — same shape as ``summary``.

Critically, ``host_phases`` measures the closing-mark interval the
same way ``segments`` does: a ``mark_host("mb_data")`` call placed
**after** ``next(dl)`` returns names the wall-clock interval since
the previous host mark (or ``begin_step``) as ``mb_data``. This is
the inverse measurement of the CUDA event's "GPU was busy for X ms"
— the host clock answers "the host wall-clock spent X ms before
proceeding to the next mark", which is the exact figure needed to
size the upper bound of P2 (dataloader background thread).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover — train loop already requires torch
    torch = None  # type: ignore[assignment]


__all__ = [
    "DEEP_PHASE_ACCUM",
    "DEEP_PHASE_BWD",
    "DEEP_PHASE_CE",
    "DEEP_PHASE_DATA",
    "DEEP_PHASE_FWD",
    "DEEP_PHASE_ORDER",
    "SEG_ALLGATHER",
    "SEG_GRAD_NORM",
    "SEG_LOSS_ALLREDUCE",
    "SEG_MICROBATCHES",
    "SEG_OPTIMIZER",
    "SEG_REDUCE_SCATTER",
    "SEG_STEP_START",
    "StepProfiler",
    "from_config",
    "summarize_segments",
]


# Env-var names. Centralized so callers (suites.py, tests, docs) point at
# a single source of truth and renaming stays painless.
_ENV_PROFILE_RANGE = "PROFILE_RANGE"
_ENV_PROFILE_OUTPUT = "PROFILE_OUTPUT"
_ENV_PROFILE_DEEP = "PROFILE_DEEP"
_ENV_HOST_TIMER = "HOST_TIMER"


# Canonical phase names emitted by ``mark_deep`` inside the microbatch loop.
# Centralized so the train loop, the test fixtures and the post-processing
# notebooks share one spelling — drift would scatter the per-MB samples
# across multiple summary keys (``mb_fwd`` vs ``mb_forward`` etc.) and
# break ``summarize_segments`` aggregation.
DEEP_PHASE_DATA = "mb_data"
DEEP_PHASE_FWD = "mb_fwd"
DEEP_PHASE_CE = "mb_ce"
DEEP_PHASE_BWD = "mb_bwd"
DEEP_PHASE_ACCUM = "mb_accum"
DEEP_PHASE_ORDER: tuple[str, ...] = (
    DEEP_PHASE_DATA,
    DEEP_PHASE_FWD,
    DEEP_PHASE_CE,
    DEEP_PHASE_BWD,
    DEEP_PHASE_ACCUM,
)


# Top-level segment names recorded by the train loop and the gate
# runner (``test_long_train.py``). Centralized to keep both call sites
# emitting the SAME names — drift between them would split the
# summary keys and break the per-segment comparison across gates.
SEG_STEP_START = "step_start"
SEG_MICROBATCHES = "microbatches"
SEG_LOSS_ALLREDUCE = "loss_allreduce"
SEG_REDUCE_SCATTER = "reduce_scatter"
SEG_GRAD_NORM = "grad_norm"
SEG_ALLGATHER = "allgather"
SEG_OPTIMIZER = "optimizer"


class StepProfiler:
    """Per-segment CUDA-event timer for a single training step.

    A *segment* is the interval between two consecutive ``mark(name)``
    calls. The first ``mark()`` opens the timeline; each subsequent
    ``mark()`` closes the previous segment and opens the next. When
    ``flush(step)`` is invoked at end-of-step (after the existing
    ``cuda.synchronize()``), elapsed times are collected and stored.

    Disabled instances (no env vars / step out of range) skip every
    GPU op — ``mark`` / ``flush`` are O(1) Python branches with no
    CUDA event allocation. Construction explicitly distinguishes
    "globally disabled" (``StepProfiler.disabled()``) from "this step
    out of range" (``should_profile_step(step)``) so the call site can
    short-circuit before paying the per-step setup either way.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        range_start: int = 0,
        range_end: int = 0,
        output_path: str | None = None,
        deep: bool = False,
        host_timer: bool = False,
    ) -> None:
        self.enabled = enabled
        self.range_start = int(range_start)
        self.range_end = int(range_end)
        self.output_path = output_path
        # Deep mode is an *instance-level* flag, not per-step: when on,
        # ``mark_deep`` inside the microbatch loop fires CUDA events;
        # when off, ``mark_deep`` is a 1-branch no-op even on profiled
        # steps. Independent of ``enabled`` so a disabled profiler still
        # carries the field (simplifies test fixtures and the
        # ``mark_deep`` no-op contract).
        self.deep = bool(deep)
        # Host timer is an orthogonal opt-in: when on, ``mark_host`` calls
        # capture ``time.perf_counter_ns`` snapshots that resolve in
        # ``flush`` to a parallel ``host_phases`` list per record. Kept
        # independent of ``enabled`` (same rationale as ``deep``) so a
        # disabled profiler still carries the field — simplifies test
        # fixtures and the no-op contract.
        self.host_timer = bool(host_timer)
        # Flat list of (step_idx, [(name, t_ms_from_start), ...])
        # populated in ``flush``. Kept on the StepProfiler so tests
        # can introspect collected data without going through disk.
        self.records: list[dict[str, Any]] = []
        # Per-step transient state (resets on every ``begin_step``).
        self._active: bool = False
        self._events: list[tuple[str, Any]] = []  # name, cuda.Event
        # Per-step host-side state (parallel to ``_events``). Each
        # ``mark_host(name)`` appends a (name, ns) tuple here; ``flush``
        # converts to per-segment ms via closing-mark convention. Stays
        # empty when ``host_timer`` is off.
        self._host_events: list[tuple[str, int]] = []
        # Opening host wall-clock anchor for the current step. Captured
        # once in ``begin_step`` (when host_timer is on) so the FIRST
        # ``mark_host`` measures the interval from begin_step → mark
        # rather than requiring an explicit "open" call site. ``None``
        # outside profiled steps / when host_timer is off.
        self._host_step_open_ns: int | None = None
        # Microbatch count observed during profiled steps (for the JSON
        # payload's top-level field). ``None`` until at least one
        # profiled step records ``DEEP_PHASE_ACCUM`` events; in non-deep
        # mode it stays ``None`` and is omitted from the output.
        self._microbatches_per_step: int | None = None

    # ----- Lifecycle helpers ---------------------------------------

    @classmethod
    def disabled(cls) -> StepProfiler:
        """Return a profiler that no-ops every method.

        Cheaper than passing ``enabled=False`` through kwargs because
        callers can use ``StepProfiler.disabled()`` once at module
        scope and re-use the instance — there is no per-step state in
        a disabled profiler so concurrent reuse is safe (the train loop
        is single-threaded anyway, but the property is documented for
        future async-step orchestrations).
        """
        return cls(enabled=False)

    def should_profile_step(self, step_idx: int) -> bool:
        """Return True iff this step falls inside ``[start, end)``.

        Used by the train loop to short-circuit ``begin_step`` /
        ``mark`` / ``flush`` calls when the step is outside the
        configured profiling range — keeps the per-step overhead at a
        single comparison for non-profiled steps.
        """
        if not self.enabled:
            return False
        return self.range_start <= step_idx < self.range_end

    def begin_step(self, step_idx: int) -> None:
        """Open a new step's timeline. No-op when out-of-range/disabled.

        Guards against accidental re-entry (``begin_step`` called twice
        without a closing ``flush``) — the second call clears the
        in-flight event list so a partial step does not poison the
        next one's measurements. This matters because the train loop's
        ``for _mb in range(num_local_microbatches)`` loop has early
        returns on tokenized-input absence; recovery should be silent
        rather than crash mid-training.

        When ``host_timer`` is enabled, also captures the opening
        ``time.perf_counter_ns`` anchor — the FIRST ``mark_host(name)``
        on this step measures the interval from this anchor to its own
        timestamp, mirroring how the first CUDA ``mark`` opens the
        device timeline.
        """
        if not self.should_profile_step(step_idx):
            self._active = False
            self._host_step_open_ns = None
            return
        self._active = True
        self._events = []
        self._host_events = []
        if self.host_timer:
            self._host_step_open_ns = time.perf_counter_ns()
        else:
            self._host_step_open_ns = None

    def mark(self, name: str) -> None:
        """Place a CUDA event on the current stream tagged with ``name``.

        Called between segments inside the train loop. Cheap when
        ``_active`` is False (the only check on the hot path for non-
        profiled steps). Records on the *current* stream — no
        ``stream`` kwarg because the train loop runs the entire step
        on the default stream; future multi-stream optimizations
        (wgrad / compute overlap) will need to revisit this contract.
        """
        if not self._active:
            return
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._events.append((name, evt))

    def mark_deep(self, name: str) -> None:
        """Like ``mark`` but only fires when deep mode is on.

        Used inside the microbatch loop for per-MB phase boundaries
        (``mb_data`` / ``mb_fwd`` / ``mb_ce`` / ``mb_bwd`` /
        ``mb_accum``). **Must be placed AFTER the phase whose name it
        carries** — see the module docstring's "Deep mode" section
        for why: ``flush()`` names each segment after its closing
        mark, so ``mark_deep("mb_fwd")`` placed BEFORE the forward
        pass would label the *previous* segment ("data load") as
        ``mb_fwd``, silently scrambling the optimization ranking.

        The two-condition guard (``_active`` AND ``deep``) makes
        this a true no-op for:

          * non-profiled steps (``_active`` is False — same fast path
            as the existing ``mark`` short-circuit),
          * shallow gate runs that set ``PROFILE_RANGE`` but not
            ``PROFILE_DEEP`` (``self.deep`` is False — the gate JSON
            stays at the standard 6-segment layout, unchanged from
            the shallow profile rollout).

        When deep mode IS on, this records a CUDA event identically
        to ``mark`` — segments named after the closing ``mark_deep``
        appear in ``flush()``'s segment list and feed
        ``summarize_segments`` aggregation by name (so the same
        ``mb_fwd`` label across all MBs and N profiled steps rolls up
        into a single mean/std/p90 entry).
        """
        if not self._active or not self.deep:
            return
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._events.append((name, evt))

    def mark_host(self, name: str) -> None:
        """Capture a host-clock timestamp tagged with ``name``.

        Closing-mark convention: each ``mark_host(name)`` closes the
        host segment opened by either ``begin_step`` (first mark in the
        step) or the previous ``mark_host`` call. The segment ms is
        wall-clock host time, so it includes Python execution time AND
        any GPU sync the host had to wait on (e.g. ``loss.item()``,
        ``next(dl)`` blocking on the modelbest_sdk Python critical
        section). The CUDA ``mark`` / ``mark_deep`` siblings cannot
        report this number — they measure device-stream elapsed time
        and miss every host-side blocker.

        The two-condition guard (``_active`` AND ``host_timer``) keeps
        this a true no-op for:

          * non-profiled steps (``_active`` is False — same fast-path
            short-circuit as ``mark`` / ``mark_deep``),
          * gate runs that don't set ``HOST_TIMER`` (``self.host_timer``
            is False — the JSON stays at the segment-only layout, matching
            shallow / deep gate runs from earlier rounds).

        Cost when enabled is ~50ns per call (``perf_counter_ns`` is a
        vDSO syscall on Linux); 11 marks per step at GBS=160/DP=16 add
        ~550ns to a 350ms step, ≈ 0.0002% — well below measurement
        noise. Cost when disabled is one boolean ``and`` short-circuit.
        """
        if not self._active or not self.host_timer:
            return
        self._host_events.append((name, time.perf_counter_ns()))

    def flush(self, step_idx: int) -> None:
        """Resolve all event timings for the closing step and store them.

        MUST be called *after* ``torch.cuda.synchronize()`` — the
        train loop already syncs at end-of-step for its own
        ``time.time()`` boundary, so the events are guaranteed
        complete by the time we read them. Non-active steps are a
        no-op.
        """
        if not self._active:
            return
        segments = []
        events = self._events
        for i in range(len(events) - 1):
            name = events[i + 1][0]  # segment name = closing mark name
            ms = events[i][1].elapsed_time(events[i + 1][1])
            segments.append({"name": name, "ms": ms})
        # Total wall-clock between the first and last event — useful
        # for sanity-checking against the host ``time.time()`` step
        # time observed by the train loop (the two should agree to
        # within a few ms on a quiet GPU).
        if len(events) >= 2:
            total_ms = events[0][1].elapsed_time(events[-1][1])
        else:
            total_ms = 0.0
        # Count microbatches by counting closing mb_accum marks in this
        # step's event list. Lets the post-processor display
        # "N MBs × mean(mb_fwd)=…" without re-deriving from the JSON.
        # Stays None outside deep mode (no mb_accum events emitted).
        mb_count = sum(1 for name, _ in events if name == DEEP_PHASE_ACCUM)
        if mb_count > 0:
            if self._microbatches_per_step is None:
                self._microbatches_per_step = mb_count
            elif self._microbatches_per_step != mb_count:
                # Different MB count across profiled steps would break
                # the "N samples per profiled step" mental model.
                # Surface it via the JSON rather than crash so the
                # operator can still inspect the data after the fact.
                self._microbatches_per_step = -1
        record: dict[str, Any] = {
            "step": int(step_idx),
            "total_ms": total_ms,
            "segments": segments,
        }
        # Resolve host-side phases when the host timer was active for
        # this step. Closing-mark convention: each entry is the wall-
        # clock interval since the prior mark (or ``begin_step``). A
        # disabled host timer or a step with no ``mark_host`` calls
        # leaves the field absent — keeps the JSON shape stable for
        # consumers that do not opt into host measurement.
        if self.host_timer and self._host_step_open_ns is not None and self._host_events:
            host_phases: list[dict[str, Any]] = []
            prev_ns = self._host_step_open_ns
            for name, ns in self._host_events:
                ms = (ns - prev_ns) / 1.0e6
                host_phases.append({"name": name, "ms": ms})
                prev_ns = ns
            record["host_phases"] = host_phases
            # Total host wall-clock between begin_step anchor and last
            # ``mark_host``. Useful when comparing to ``host_step_ms``
            # (which the train loop measures separately via
            # ``time.time()`` outside the profiler).
            record["host_total_ms"] = (
                self._host_events[-1][1] - self._host_step_open_ns
            ) / 1.0e6
        self.records.append(record)
        # Release event references so allocator can reuse them.
        self._events = []
        self._host_events = []
        self._host_step_open_ns = None
        self._active = False

    def write(self, *, step_time_ms_by_step: dict[int, float] | None = None) -> str | None:
        """Persist the collected records as JSON.

        Returns the resolved output path (str) on success, or ``None``
        when nothing was collected (no profiled steps actually ran —
        e.g. the configured range was outside the executed step
        indices). Writes are atomic via ``tmp + os.replace``.

        ``step_time_ms_by_step`` is the train loop's host-side
        ``step_time`` dict; merged into each record under
        ``host_step_ms`` so the JSON shows the gap between
        device-event timing and host-side ``time.time()`` timing
        without forcing the consumer to cross-reference a separate
        log.
        """
        if not self.enabled or self.output_path is None:
            return None
        if not self.records:
            return None
        if step_time_ms_by_step:
            for rec in self.records:
                ht = step_time_ms_by_step.get(rec["step"])
                if ht is not None:
                    rec["host_step_ms"] = float(ht)
        payload: dict[str, Any] = {
            "schema_version": 2,
            "range": [self.range_start, self.range_end],
            "deep_mode": bool(self.deep),
            "records": self.records,
            "summary": summarize_segments(self.records),
        }
        if self._microbatches_per_step is not None:
            payload["microbatches_per_step"] = int(self._microbatches_per_step)
        # Host-timer aggregates are emitted only when the timer was
        # enabled — keeps the JSON shape backward-compatible with v2
        # consumers that predate the host-timer rollout. ``host_summary``
        # mirrors the segment ``summary`` shape (bucket-by-name +
        # mean/p50/p90/std/total) so post-processing tooling can re-use
        # the same aggregation pipeline with no special-casing.
        if self.host_timer:
            payload["host_timer"] = True
            payload["host_summary"] = summarize_host_phases(self.records)
        out_path = os.fspath(self.output_path)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
        os.replace(tmp_path, out_path)
        return out_path


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def from_config(cfg: "EngineConfig") -> StepProfiler:  # noqa: F821
    """Construct a ``StepProfiler`` from ``EngineConfig`` fields.

    ``profile_range`` and ``profile_output`` MUST be set together — half-
    configuration raises ``ValueError`` at construction time so a typo /
    missing variable cannot silently drop measurements.

    ``profile_deep=True`` requires ``profile_range`` to also be set;
    setting it alone raises ``ValueError``.

    When neither ``profile_range`` nor ``profile_output`` is set this
    returns a no-op profiler.
    """
    has_range = cfg.profile_range is not None
    has_output = cfg.profile_output is not None

    if not has_range and not has_output:
        if cfg.profile_deep:
            raise ValueError(
                "profile_deep requires both profile_range and profile_output"
            )
        if cfg.host_timer:
            raise ValueError(
                "host_timer requires both profile_range and profile_output"
            )
        return StepProfiler.disabled()

    if has_range ^ has_output:
        missing = "profile_output" if has_range else "profile_range"
        raise ValueError(
            f"profiler fields must be set together: missing {missing}"
        )

    start, end = cfg.profile_range  # type: ignore[misc]
    if end <= start:
        raise ValueError(
            f"profile_range: end ({end}) must be > start ({start})"
        )
    return StepProfiler(
        enabled=True,
        range_start=start,
        range_end=end,
        output_path=cfg.profile_output,  # type: ignore[arg-type]
        deep=cfg.profile_deep,
        host_timer=cfg.host_timer,
    )


def summarize_segments(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-segment statistics across all profiled steps.

    Returns a dict keyed by segment name with mean / median / p90 / std
    over the collected records. Used by ``StepProfiler.write`` to
    enrich the on-disk JSON, and exposed at module scope so consumers
    (notebook / review agent) can re-run summarization on subset
    records (e.g. excluding warmup) without re-implementing the
    aggregation logic.

    Empty input returns ``{}`` rather than raising — calling code
    routinely passes a partial trajectory when no steps fell in the
    profiling range.
    """
    return _bucket_summary(records, "segments")


def summarize_host_phases(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-host-phase statistics across all profiled steps.

    Mirrors :func:`summarize_segments` but reads ``host_phases`` from
    each record (populated by ``StepProfiler.flush`` when ``host_timer``
    is on). Records without ``host_phases`` (e.g. shallow / no-host-
    timer runs) are silently skipped, so calling this on a v2 payload
    with no host data returns ``{}`` instead of raising.

    Same shape as ``summarize_segments`` so the analyzer can re-use the
    same render code by parameterizing on the bucket source.
    """
    return _bucket_summary(records, "host_phases")


def _bucket_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Shared aggregator used by both segment- and host-phase summaries.

    Walks each record's ``key`` list, buckets ms values by ``name``,
    then emits ``mean / median / p90 / std / total / n`` per bucket.
    Centralized so a future change (e.g. emit p99 instead of p90) lands
    in exactly one place rather than diverging between segments and
    host phases.
    """
    if not records:
        return {}
    by_name: dict[str, list[float]] = {}
    for rec in records:
        for entry in rec.get(key, []) or []:
            try:
                ms_val = float(entry["ms"])
            except (KeyError, TypeError, ValueError):
                continue
            by_name.setdefault(entry["name"], []).append(ms_val)
    out: dict[str, Any] = {}
    for name, values in by_name.items():
        n = len(values)
        sorted_vals = sorted(values)
        out[name] = {
            "n": n,
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(sorted_vals),
            # p90 by nearest-rank — adequate for n>=10 typical sample sizes;
            # for very small n falls back to max so the report still parses.
            "p90_ms": (
                sorted_vals[max(0, int(round(0.9 * (n - 1))))]
                if n >= 1 else 0.0
            ),
            "std_ms": (
                statistics.pstdev(values) if n >= 2 else 0.0
            ),
            "total_ms": sum(values),
        }
    return out
