"""Background worker pool for the operator stress tests.

Seven concurrent neighbour workloads (W1–W7) run alongside the candidate
operator on the same GPU.  Each worker owns its own CUDA stream and runs
in its own Python thread; they STOP cleanly when their ``stop_event`` is
set by the runner.

Mapping (worker → HW dimension covered):

    W1 ComputeBlast   SM occupancy, tensor-core throughput, register / smem
                      / warp-scheduler pressure (cuBLAS BF16 GEMM mix)
    W2 MemBlast       HBM bandwidth, L2, copy-engine (D2D memcpy)
    W3 TMAStorm       TMA descriptor density (K-heavy cuBLAS GEMM)
    W4 ClusterCo      Hopper cluster-mode scheduler, mbarrier hw
    W5 StreamChaos    cross-stream event wiring + priority-stream preemption
    W6 GraphMode      CUDA-Graph capture / replay path (opt-in; eager by default)
    W7 AllocChurn     caching-allocator pool fragmentation (16–256 MB / 10 Hz)

This is a bare-metal port of the operator stress harness: no remote
submission, no rsync, no kubectl-style job orchestration.  The runner in
``_stress_runner.py`` instantiates whichever subset of these workers the
caller asks for, runs the candidate operator in the main thread for the
requested duration, and walks the post-loop oracle check.

Notes:
  * W6's CUDA-Graph capture defaults to ``"eager"`` (no capture); pass
    ``w6_replay_mode="graph"`` to opt in.  Graph replay can SIGSEGV the
    whole process under W7's allocator churn (Python cannot catch
    SIGSEGV), so the safe default is eager.
  * W5 deliberately does NOT touch the legacy default stream so the
    candidate's ``cand_stream.synchronize()`` is never serialised
    against the workers' traffic.
"""

from __future__ import annotations

import random
import threading
from typing import Callable, Optional

import torch


__all__ = [
    "WorkerBase",
    "W1_ComputeBlast",
    "W2_MemBlast",
    "W3_TMAStorm",
    "W4_ClusterCo",
    "W5_StreamChaos",
    "W6_GraphMode",
    "W7_AllocChurn",
    "build_workers",
]


# ──────────────────────────────────────────────────────────────────────────
# Base worker
# ──────────────────────────────────────────────────────────────────────────
class WorkerBase:
    """Common lifecycle: ``start()`` → background thread runs ``loop()`` → ``stop()``."""

    def __init__(self, name: str, device: str = "cuda:0",
                 stream_priority: int = 0):
        self.name = name
        self.device = device
        self.stream = torch.cuda.Stream(device=device, priority=stream_priority)
        self.stop_event = threading.Event()
        self.error: Optional[BaseException] = None
        self.iter_count = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._safe_loop, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _safe_loop(self) -> None:
        try:
            torch.cuda.set_device(self.device)
            print(f"[worker {self.name}] thread started", flush=True)
            with torch.cuda.device(self.device):
                self.loop()
        except BaseException as exc:
            self.error = exc
            print(
                f"[worker {self.name}] FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def loop(self) -> None:  # pragma: no cover -- abstract
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────
# Compute / memory / TMA / cluster workers
# ──────────────────────────────────────────────────────────────────────────
class W1_ComputeBlast(WorkerBase):
    """cuBLAS BF16 GEMM mix on a dedicated stream — SM / TC / regs / smem."""

    def __init__(self, device: str = "cuda:0", reseed_every: int = 100):
        super().__init__("W1_ComputeBlast", device, stream_priority=0)
        self.reseed_every = reseed_every
        # Shapes chosen small enough that each kernel < 1 ms so the
        # candidate iters can interleave smoothly; diversity is enough
        # to defeat any cache that would let cuBLAS skip work.
        shapes = [
            (2048, 2048, 2048),
            (1024, 2048, 4096),
            (4096, 1024, 2048),
            (2048, 4096, 1024),
        ]
        self.bufs = []
        for m, n, k in shapes:
            a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
            c = torch.empty(m, n, dtype=torch.bfloat16, device=device)
            self.bufs.append((a, b, c))

    def loop(self) -> None:
        with torch.cuda.stream(self.stream):
            i = 0
            while not self.stop_event.is_set():
                a, b, c = self.bufs[i % len(self.bufs)]
                torch.matmul(a, b, out=c)
                i += 1
                self.iter_count = i
                if i % self.reseed_every == 0:
                    a.normal_()
                    b.normal_()


class W2_MemBlast(WorkerBase):
    """D2D memcpy ≥256 MB at random page-aligned offsets — HBM / L2 / TLB / CE."""

    def __init__(self, device: str = "cuda:0", buf_mb: int = 512,
                 chunk_mb_range: tuple[int, int] = (16, 128),
                 realloc_every: int = 10_000):
        super().__init__("W2_MemBlast", device, stream_priority=0)
        self.buf_mb = buf_mb
        self.chunk_mb_range = chunk_mb_range
        self.realloc_every = realloc_every
        self._alloc()

    def _alloc(self) -> None:
        elems = self.buf_mb * 1024 * 1024 // 2  # bf16 = 2 B
        self.src = torch.empty(elems, dtype=torch.bfloat16, device=self.device)
        self.dst = torch.empty(elems, dtype=torch.bfloat16, device=self.device)
        # bf16 page-aligned step: page = 4096 B, bf16 = 2 B → step = 2048 elems
        self.page_step = 2048

    def loop(self) -> None:
        with torch.cuda.stream(self.stream):
            i = 0
            while not self.stop_event.is_set():
                chunk_mb = random.randint(*self.chunk_mb_range)
                chunk_elems = chunk_mb * 1024 * 1024 // 2
                max_pages = max(
                    1, (self.src.numel() - chunk_elems) // self.page_step)
                off = random.randint(0, max_pages) * self.page_step
                self.dst.narrow(0, off, chunk_elems).copy_(
                    self.src.narrow(0, off, chunk_elems))
                i += 1
                self.iter_count = i
                if i % self.realloc_every == 0:
                    del self.src
                    del self.dst
                    torch.cuda.empty_cache()
                    self._alloc()


class W3_TMAStorm(WorkerBase):
    """K-heavy cuBLAS BF16 GEMM — maximises TMA descriptors / FLOP."""

    def __init__(self, device: str = "cuda:0"):
        super().__init__("W3_TMAStorm", device, stream_priority=0)
        # Small M/N + large K keeps the per-kernel time < 1 ms so it
        # never monopolises the SMs.
        shapes = [
            (512, 512, 16384),
            (256, 512, 32768),
            (512, 256, 32768),
        ]
        self.bufs = []
        for m, n, k in shapes:
            a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
            c = torch.empty(m, n, dtype=torch.bfloat16, device=device)
            self.bufs.append((a, b, c))

    def loop(self) -> None:
        with torch.cuda.stream(self.stream):
            i = 0
            while not self.stop_event.is_set():
                a, b, c = self.bufs[i % len(self.bufs)]
                torch.matmul(a, b, out=c)
                i += 1
                self.iter_count = i


class W4_ClusterCo(WorkerBase):
    """Large cuBLAS BF16 GEMM — drives the Hopper cluster scheduler."""

    def __init__(self, device: str = "cuda:0"):
        super().__init__("W4_ClusterCo", device, stream_priority=0)
        # M, N ≥ 2048 with BF16 → cuBLAS picks cluster_mn on H100.
        shapes = [
            (4096, 4096, 2048),
            (4096, 2048, 4096),
        ]
        self.bufs = []
        for m, n, k in shapes:
            a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
            c = torch.empty(m, n, dtype=torch.bfloat16, device=device)
            self.bufs.append((a, b, c))

    def loop(self) -> None:
        with torch.cuda.stream(self.stream):
            i = 0
            while not self.stop_event.is_set():
                a, b, c = self.bufs[i % len(self.bufs)]
                torch.matmul(a, b, out=c)
                i += 1
                self.iter_count = i


class W5_StreamChaos(WorkerBase):
    """Cross-stream event wiring + periodic high-priority preemption.

    Records events on each neighbour's stream and makes the next stream
    in the ring wait on it, creating an inter-stream dependency chain
    that does not match what the natural scheduler would produce.  Also
    periodically issues a small GEMM on a high-priority stream to take
    SMs away from the neighbours.

    NOTE: we deliberately do NOT touch the legacy default stream — under
    classic-default semantics, any issue on it serialises with every
    other stream (including the candidate's), which makes the candidate
    stream's synchronize() block on worker traffic and hangs the gate.
    """

    def __init__(self, neighbours: list[WorkerBase], device: str = "cuda:0"):
        super().__init__("W5_StreamChaos", device, stream_priority=-1)
        self.neighbours = neighbours
        self.hp_a = torch.randn(2048, 2048, dtype=torch.bfloat16, device=device)
        self.hp_b = torch.randn(2048, 2048, dtype=torch.bfloat16, device=device)
        self.hp_c = torch.empty(2048, 2048, dtype=torch.bfloat16, device=device)

    def loop(self) -> None:
        i = 0
        while not self.stop_event.is_set():
            ring = self.neighbours
            for a, b in zip(ring, ring[1:] + ring[:1]):
                ev = torch.cuda.Event()
                ev.record(a.stream)
                b.stream.wait_event(ev)

            with torch.cuda.stream(self.stream):
                torch.matmul(self.hp_a, self.hp_b, out=self.hp_c)

            i += 1
            self.iter_count = i
            # ~20 Hz; goal is reshuffling, not throughput.
            self.stop_event.wait(0.05)


class W6_GraphMode(WorkerBase):
    """Wrap the candidate as a CUDA Graph and replay it in a parallel stream.

    Operates on its OWN input/output buffers (built by ``make_call_factory``)
    so it never aliases the main candidate driver's tensors.  Default
    ``replay_mode="eager"`` skips graph capture — opt into ``"graph"``
    only when the F1 (graph-launch) dimension is specifically wanted.
    """

    def __init__(self, make_call_factory: Callable[[], Callable[[], None]],
                 device: str = "cuda:0", replay_mode: str = "eager"):
        super().__init__("W6_GraphMode", device, stream_priority=0)
        self.candidate_call = make_call_factory()
        self.replay_mode = replay_mode
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def _try_capture(self) -> None:
        try:
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    self.candidate_call()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.candidate_call()
        except Exception as exc:
            print(
                f"[worker {self.name}] graph capture failed: {exc}; "
                "falling back to eager",
                flush=True,
            )
            self.graph = None

    def loop(self) -> None:
        if self.replay_mode == "graph":
            self._try_capture()
        print(
            f"[worker {self.name}] mode={self.replay_mode} "
            f"effective_graph={self.graph is not None}",
            flush=True,
        )
        i = 0
        with torch.cuda.stream(self.stream):
            while not self.stop_event.is_set():
                try:
                    if self.graph is not None:
                        self.graph.replay()
                    else:
                        self.candidate_call()
                except Exception as exc:
                    print(
                        f"[worker {self.name}] launch failed at iter {i}: "
                        f"{exc}; disabling worker",
                        flush=True,
                    )
                    self.error = exc
                    return
                i += 1
                self.iter_count = i


class W7_AllocChurn(WorkerBase):
    """Random alloc / free of 16–256 MB buffers at ~10 Hz — allocator stress."""

    def __init__(self, device: str = "cuda:0", hz: float = 10.0,
                 size_mb_range: tuple[int, int] = (16, 256)):
        super().__init__("W7_AllocChurn", device, stream_priority=0)
        self.period_s = 1.0 / hz
        self.size_mb_range = size_mb_range

    def loop(self) -> None:
        i = 0
        while not self.stop_event.is_set():
            try:
                size_mb = random.randint(*self.size_mb_range)
                buf = torch.empty(
                    size_mb * 1024 * 1024 // 2,
                    dtype=torch.bfloat16, device=self.device,
                )
                # Touch one element to force the allocation, then drop.
                buf[0] = 0
                del buf
            except torch.cuda.OutOfMemoryError:
                # Acceptable: just skip this iter.
                pass
            i += 1
            self.iter_count = i
            self.stop_event.wait(self.period_s)


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────
def build_workers(
    *,
    w6_make_call: Optional[Callable[[], Callable[[], None]]] = None,
    device: str = "cuda:0",
    enabled: Optional[set[str]] = None,
    w6_replay_mode: str = "eager",
) -> list[WorkerBase]:
    """Build the requested subset of W1..W7.

    ``enabled`` defaults to ``{"W1","W2","W3","W4","W5","W7"}`` — W6 is
    opt-in via the set (and additionally requires ``w6_make_call`` from
    the per-op dispatch in :mod:`_stress_dispatch`).
    """
    enabled = enabled if enabled is not None else {"W1", "W2", "W3", "W4", "W5", "W7"}
    workers: list[WorkerBase] = []
    if "W1" in enabled:
        workers.append(W1_ComputeBlast(device))
    if "W2" in enabled:
        workers.append(W2_MemBlast(device))
    if "W3" in enabled:
        workers.append(W3_TMAStorm(device))
    if "W4" in enabled:
        workers.append(W4_ClusterCo(device))
    if "W5" in enabled:
        neighbours = [w for w in workers if w.name.split("_")[0] in {"W1", "W2", "W3"}]
        workers.append(W5_StreamChaos(neighbours, device))
    if "W6" in enabled:
        if w6_make_call is None:
            raise ValueError(
                "W6 enabled but w6_make_call factory not provided by the "
                "per-op dispatch; drop W6 from `enabled` or pass a factory."
            )
        workers.append(W6_GraphMode(w6_make_call, device, w6_replay_mode))
    if "W7" in enabled:
        workers.append(W7_AllocChurn(device))
    return workers
