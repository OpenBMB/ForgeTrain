"""Op-agnostic stress runner — bare-metal pytest entry point.

The runner:

  1. Allocates ``fill_gb`` of filler HBM to mimic production occupancy.
  2. Builds the per-op dispatch via :func:`_stress_dispatch.build_dispatch`.
  3. Spawns the requested subset of W1..W7 background workers.
  4. Runs the candidate call in the main thread on a dedicated CUDA stream
     for the configured duration / iter cap, with an in-loop NaN/Inf
     guard every ``oracle_interval`` iters.
  5. After the loop, halts the workers and runs a single post-loop
     numerical oracle comparison vs the reference path.
  6. Asserts a hard watchdog at ``duration_s + grace_s`` via
     ``os._exit`` so a hang on cand_stream.synchronize() never silently
     blocks past the cap.

PASS criteria (all must hold):
    ① no exception from cand_stream.synchronize()  (catches XID 13)
    ② iter time never exceeds ``hang_s`` seconds   (catches hang)
    ③ no NaN/Inf in candidate output on every oracle_interval
    ④ post-loop oracle check shows no numerical regression
    ⑤ no worker died with an unhandled exception

Every knob is overridable via the matching ``STRESS_<KNOB>`` env var so
``pytest tests/`` and ``STRESS_DURATION_S=900 python tests/test_*.py``
share one code path.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch

from tests._stress_dispatch import build_dispatch
from tests._stress_workers import WorkerBase, build_workers


__all__ = ["run_stress", "DEFAULT_PROFILE"]


# Default smoke profile: tuned to finish in ~30 s on a single H100 SXM5
# while still putting the candidate kernel under non-trivial neighbour
# load.  Override individual fields by passing them to ``run_stress``
# or by exporting the matching ``STRESS_<KNOB>=...`` env var.
DEFAULT_PROFILE: dict[str, object] = {
    "duration_s": 30.0,
    "iters": 200_000,            # hard cap; duration is the real throttle
    "fill_gb": 2.0,              # small enough for shared dev GPUs
    "oracle_interval": 50,
    "rtol": 0.05,
    "atol": 0.5,
    "hang_s": 10.0,              # generous to absorb first-iter JIT warmup
    "workers": {"W1", "W2", "W3", "W4", "W5", "W7"},
    "w6_replay_mode": "eager",
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_workers(default: set[str]) -> set[str]:
    raw = os.environ.get("STRESS_WORKERS")
    if raw is None:
        return set(default)
    return {w.strip() for w in raw.split(",") if w.strip()}


def _allocate_filler(target_gb: float, device: str) -> list[torch.Tensor]:
    """Allocate ~``target_gb`` of FP32 filler in 256 MB chunks."""
    fillers: list[torch.Tensor] = []
    allocated_gb = 0.0
    chunk_mb = 256
    while allocated_gb < target_gb:
        try:
            fillers.append(torch.empty(
                chunk_mb * 1024 * 1024 // 4,
                dtype=torch.float32, device=device,
            ))
            allocated_gb += chunk_mb / 1024.0
        except torch.cuda.OutOfMemoryError:
            break
    print(f"[stress] allocated {allocated_gb:.1f} GB filler", flush=True)
    return fillers


def _free_filler_headroom(fillers: list[torch.Tensor],
                          needed_free_gb: float) -> None:
    chunk_gb = 256 / 1024.0
    freed = 0.0
    while fillers and freed < needed_free_gb:
        fillers.pop()
        freed += chunk_gb
    torch.cuda.empty_cache()
    print(
        f"[stress] freed {freed:.1f} GB for workers + kernel headroom",
        flush=True,
    )


def run_stress(
    *,
    op: str,
    kernel: str = "active",
    mode: str = "fwd",
    duration_s: Optional[float] = None,
    iters: Optional[int] = None,
    fill_gb: Optional[float] = None,
    oracle_interval: Optional[int] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    hang_s: Optional[float] = None,
    workers: Optional[set[str]] = None,
    w6_replay_mode: Optional[str] = None,
    device: str = "cuda:0",
    verbose: bool = True,
) -> dict:
    """Run the stress harness once and return the report dict.

    Every parameter falls back to the matching ``STRESS_<KNOB>`` env var,
    then to :data:`DEFAULT_PROFILE`.  The function returns a dict whose
    ``outcome`` field is ``"PASS"`` or ``"FAIL"``; pytest tests should
    assert on that field.
    """
    duration_s = duration_s if duration_s is not None else _env_float(
        "STRESS_DURATION_S", float(DEFAULT_PROFILE["duration_s"]))
    iters = iters if iters is not None else _env_int(
        "STRESS_ITERS", int(DEFAULT_PROFILE["iters"]))
    fill_gb = fill_gb if fill_gb is not None else _env_float(
        "STRESS_FILL_GB", float(DEFAULT_PROFILE["fill_gb"]))
    oracle_interval = oracle_interval if oracle_interval is not None else _env_int(
        "STRESS_ORACLE_INTERVAL", int(DEFAULT_PROFILE["oracle_interval"]))
    rtol = rtol if rtol is not None else _env_float(
        "STRESS_RTOL", float(DEFAULT_PROFILE["rtol"]))
    atol = atol if atol is not None else _env_float(
        "STRESS_ATOL", float(DEFAULT_PROFILE["atol"]))
    hang_s = hang_s if hang_s is not None else _env_float(
        "STRESS_HANG_S", float(DEFAULT_PROFILE["hang_s"]))
    workers = workers if workers is not None else _env_workers(
        DEFAULT_PROFILE["workers"])  # type: ignore[arg-type]
    w6_replay_mode = w6_replay_mode if w6_replay_mode is not None else os.environ.get(
        "STRESS_W6_MODE", str(DEFAULT_PROFILE["w6_replay_mode"]))

    torch.cuda.set_device(device)

    if verbose:
        print(f"[stress] op={op} kernel={kernel} mode={mode}", flush=True)
        print(
            f"[stress] duration={duration_s}s iters_cap={iters} "
            f"fill={fill_gb}GB oracle_every={oracle_interval} "
            f"hang_s={hang_s}",
            flush=True,
        )
        print(
            f"[stress] workers={sorted(workers)} w6_mode={w6_replay_mode}",
            flush=True,
        )

    # Watchdog: never silently block past the duration cap.
    watchdog_deadline = time.perf_counter() + duration_s + 60.0

    def _watchdog() -> None:
        while True:
            time.sleep(5.0)
            if time.perf_counter() > watchdog_deadline:
                print(
                    f"[stress] WATCHDOG FIRE: exceeded duration_s={duration_s}s "
                    "+ 60s grace; os._exit(2)",
                    flush=True,
                )
                os._exit(2)
    threading.Thread(target=_watchdog, daemon=True).start()

    # Filler to mimic production HBM occupancy.
    fillers = _allocate_filler(fill_gb, device)

    # Build the per-op dispatch BEFORE freeing filler so any kernel-internal
    # cache allocations land while filler is at its peak (closer to prod).
    dispatch = build_dispatch(op, kernel, mode, device)
    if verbose:
        print(
            f"[stress] dispatch ready (shape_info={dispatch.shape_info})",
            flush=True,
        )

    # Free some filler so workers + the kernel have room to allocate.
    _free_filler_headroom(fillers, needed_free_gb=6.0)

    # Disable W6 when mode is wgrad (RMW; graph-replay would be unsafe).
    if op in ("gemm_fc1", "gemm_output") and mode == "wgrad" and "W6" in workers:
        print(
            f"[stress] disabling W6 (mode={mode} is not graph-capture-safe)",
            flush=True,
        )
        workers = workers - {"W6"}

    cand_stream = torch.cuda.Stream(device=device, priority=0)
    if verbose:
        print(
            f"[stress] candidate stream cuda_stream=0x{cand_stream.cuda_stream:x}",
            flush=True,
        )

    if verbose:
        print("[stress] warming up candidate...", flush=True)
    with torch.cuda.stream(cand_stream):
        for _ in range(3):
            dispatch.candidate_call()
    cand_stream.synchronize()
    if verbose:
        print("[stress] warmup done.", flush=True)

    bg: list[WorkerBase] = build_workers(
        w6_make_call=dispatch.make_w6_call,
        device=device,
        enabled=workers,
        w6_replay_mode=w6_replay_mode,
    )
    if verbose:
        print(
            f"[stress] starting {len(bg)} workers: {[w.name for w in bg]}",
            flush=True,
        )
    for w_ in bg:
        w_.start()
    time.sleep(2.0)  # Let workers settle.

    iter_times: list[float] = []
    fail_reason: Optional[str] = None
    last_ok_iter = -1
    oracle_checks_done = 0
    t_start = time.perf_counter()

    try:
        for it in range(iters):
            iter_t0 = time.perf_counter()
            with torch.cuda.stream(cand_stream):
                dispatch.candidate_call()
            cand_stream.synchronize()
            dt = time.perf_counter() - iter_t0
            iter_times.append(dt)

            if it < 5 and verbose:
                print(f"  [it={it}] dt={dt*1000:.2f}ms", flush=True)

            if it > 0 and it % oracle_interval == 0:
                with torch.cuda.stream(cand_stream):
                    out_tensor = dispatch.get_candidate_out()
                    bad = (~torch.isfinite(out_tensor)).any()
                cand_stream.synchronize()
                if bool(bad.item()):
                    fail_reason = (
                        f"NaN/Inf in candidate output at iter {it}"
                    )
                    raise RuntimeError(fail_reason)
                oracle_checks_done += 1

            if dt > hang_s:
                fail_reason = (
                    f"hang: iter={it} dt={dt:.2f}s > {hang_s}s"
                )
                raise RuntimeError(fail_reason)

            for w_ in bg:
                if w_.error is not None:
                    fail_reason = f"worker {w_.name} died: {w_.error!r}"
                    raise RuntimeError(fail_reason)

            last_ok_iter = it

            if (it + 1) % 200 == 0 and verbose:
                elapsed = time.perf_counter() - t_start
                recent_p99 = float(np.percentile(iter_times[-200:], 99))
                ws_status = " ".join(
                    f"{w_.name.split('_')[0]}={w_.iter_count}" for w_ in bg)
                print(
                    f"  [{it+1}/{iters}] OK ({elapsed:.1f}s) "
                    f"p99={recent_p99*1000:.1f}ms "
                    f"oracle_ok={oracle_checks_done} workers: {ws_status}",
                    flush=True,
                )

            if (time.perf_counter() - t_start) > duration_s:
                if verbose:
                    print(
                        f"[stress] reached duration cap {duration_s}s; "
                        "stopping",
                        flush=True,
                    )
                break

    except BaseException as exc:
        if fail_reason is None:
            fail_reason = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t_start
        failed_iter = last_ok_iter + 1
        print(
            f"[stress] CRASH at iter={failed_iter} (last OK={last_ok_iter}, "
            f"elapsed={elapsed*1000:.1f}ms): {fail_reason}",
            flush=True,
        )
    finally:
        if verbose:
            print("[stress] stopping workers...", flush=True)
        for w_ in bg:
            w_.stop()

    # ── Post-loop numerical oracle ──
    oracle_max_err: Optional[float] = None
    oracle_fail_pct: Optional[float] = None
    if fail_reason is None and last_ok_iter >= 0:
        if verbose:
            print(
                "[stress] post-loop oracle check (workers stopped)...",
                flush=True,
            )
        try:
            with torch.cuda.stream(cand_stream):
                dispatch.candidate_call()
                cand_final = dispatch.get_candidate_out().detach().clone()
                ref_final = dispatch.oracle_call()
            cand_stream.synchronize()
            cand_cpu = cand_final.float().cpu()
            ref_cpu = ref_final.float().cpu()
            diff = (cand_cpu - ref_cpu).abs()
            tol = atol + rtol * ref_cpu.abs()
            fail_mask = diff > tol
            n_fail = int(fail_mask.sum().item())
            oracle_max_err = float(diff.max().item())
            oracle_fail_pct = n_fail / fail_mask.numel() * 100
            if verbose:
                print(
                    f"[stress] post-loop oracle: max_err={oracle_max_err:.4g} "
                    f"fail_pct={oracle_fail_pct:.2f}% "
                    f"(rtol={rtol} atol={atol})",
                    flush=True,
                )
            if n_fail > 0:
                fail_reason = (
                    f"post-loop numerical: max_err={oracle_max_err:.4g} "
                    f"fail_pct={oracle_fail_pct:.2f}%"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[stress] post-loop oracle failed: {exc}", flush=True)
            if fail_reason is None:
                fail_reason = f"post-loop oracle exception: {exc}"

    elapsed = time.perf_counter() - t_start
    p99_ms = (float(np.percentile(iter_times, 99) * 1000)
              if iter_times else None)
    median_ms = (float(np.median(iter_times) * 1000)
                 if iter_times else None)
    report = {
        "op": op, "kernel": kernel, "mode": mode, "fill_gb": fill_gb,
        "shape_info": dispatch.shape_info,
        "iters_cap": iters, "iters_completed": last_ok_iter + 1,
        "median_iter_ms": median_ms, "p99_iter_ms": p99_ms,
        "oracle_inloop_checks_done": oracle_checks_done,
        "post_loop_oracle_max_err": oracle_max_err,
        "post_loop_oracle_fail_pct": oracle_fail_pct,
        "duration_s": round(elapsed, 1),
        "workers_enabled": sorted([w_.name for w_ in bg]),
        "worker_iters": {w_.name: w_.iter_count for w_ in bg},
        "worker_errors": {w_.name: repr(w_.error)
                          for w_ in bg if w_.error},
        "fail_reason": fail_reason,
        "outcome": "FAIL" if fail_reason else "PASS",
    }
    if verbose:
        print(f"\n=========== {report['outcome']} ===========")
        print(json.dumps(report, indent=2))

    # Drop filler explicitly so a fast pytest session can run multiple
    # tests in the same process without HBM pressure carrying over.
    del fillers
    torch.cuda.empty_cache()

    return report


def main() -> int:  # pragma: no cover -- manual CLI entry
    """Standalone CLI: ``python -m tests._stress_runner <op> [mode]``."""
    if len(sys.argv) < 2:
        print(
            "usage: python -m tests._stress_runner "
            "<gemm_fc1|gemm_output|attention_fwd> [mode]",
            file=sys.stderr,
        )
        return 2
    op = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else (
        "fwd" if op == "attention_fwd" else "all")
    report = run_stress(op=op, kernel="active", mode=mode)
    return 0 if report["outcome"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
