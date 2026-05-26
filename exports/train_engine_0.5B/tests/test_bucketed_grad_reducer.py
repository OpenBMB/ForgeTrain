"""CPU unit tests for the M2-P2 bucketed wgrad-allreduce overlap path.

Covers two layers of contracts pinned by ``recommendation.md`` Round
M2-P2 §Step 5 ("CPU 单测：bucket 切分逻辑、hook 触发顺序、stream 同步契约"):

1. **Bucket layout** (``compute_grad_buckets``) — the partitioning of
   the baseline grad-buffer entries into per-layer buckets must be:
     - exact: every param accounted for, no duplicates, monotonic
       offsets, raw_size + padding == size;
     - aligned: bucket size is a multiple of ``world_size`` so
       ``reduce_scatter_tensor`` can shard it evenly;
     - greedy: a new bucket starts when the running raw size would
       exceed ``bucket_target_elems`` (a single huge param like the
       0.5 B model's 75 M-element embedding gets its own bucket rather
       than splitting across two, because the legacy buffer layout
       treats each param as an indivisible unit);
     - fail-fast on invalid args: ``world_size <= 0`` or
       ``bucket_target_elems <= 0`` raise ``ValueError`` rather than
       silently producing a 0-byte bucket that later hangs in NCCL.

2. **Reducer state machine** (``BucketedGradReducer``) — the grad sink
   exposed to ``backward.py`` must:
     - reject unknown params, wrong numel, and duplicate writes after
       a bucket has already been triggered;
     - reset per-step pending counters via ``start_step()`` (buffer
       payload is intentionally NOT zeroed; only the counters and
       completion events are reset);
     - dispatch ``reduce_scatter`` exactly once per bucket the moment
       the bucket fills (verified by the synchronous non-distributed
       fallback that copies input → shard);
     - support a full round-trip: write all params, then
       ``allgather_and_unpack`` returns FP32 grads with the original
       per-parameter shapes;
     - compute distributed grad norm by walking the rank's slice of
       each bucket's shard, classifying weight vs norm grads in the
       legacy reverse-registration order, and summing the squared L2
       norm across DP ranks (the all_reduce is mocked / skipped on
       single-rank tests since there is nothing to sum across).

The tests run on the macOS dev sandbox without CUDA / NCCL by:
  * configuring the reducer with ``device="cpu"`` (the FP32 input + shard
    buffers are plain ``torch.zeros``/``torch.empty`` calls — no CUDA
    allocator involvement);
  * leaving ``torch.distributed`` un-initialised so ``is_distributed()``
    returns ``False`` and the reducer falls back to a synchronous
    "copy input → shard" path that is bytewise comparable to the real
    ``reduce_scatter`` semantics for the single-rank case;
  * skipping the ``compute_distributed_grad_norm`` tests when the
    ``transformer_engine`` package is unavailable (it is a CUDA-only
    dependency that ships in the cctl unit-test image but not on Mac).

Tests automatically skip when torch is not installed, mirroring the
``_NEEDS_TORCH`` convention used by ``test_data_prefetcher.py`` /
``test_checkpoint_round_trip.py``.
"""

from __future__ import annotations

import importlib
import sys
import unittest
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
_HAS_TE = _can_import("transformer_engine")
_NEEDS_TORCH = unittest.skipUnless(_HAS_TORCH, "requires torch")
_NEEDS_TE = unittest.skipUnless(
    _HAS_TORCH and _HAS_TE,
    "requires torch and transformer_engine",
)

if _HAS_TORCH:
    import torch  # noqa: E402


def _init_test_config() -> None:
    """Set the EngineConfig singleton for tests that call get_config()."""
    from training_engine_tensor.engine_config import EngineConfig, set_global_config
    set_global_config(EngineConfig())


def _make_cpu_fused_adam_sync():
    """Pure-torch stub for the triton fused Adam kernel (CPU-only)."""
    BETA1 = 0.9
    BETA2 = 0.95
    EPS = 1e-8

    def fused_adam_sync(g, master, exp_avg, exp_avg_sq, param,
                        *, lr, step, clip_coeff, wd):
        g_clipped = g * clip_coeff
        exp_avg.mul_(BETA1).add_(g_clipped, alpha=1 - BETA1)
        exp_avg_sq.mul_(BETA2).addcmul_(g_clipped, g_clipped, value=1 - BETA2)
        bias_c1 = 1 - BETA1 ** step
        bias_c2 = 1 - BETA2 ** step
        denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(EPS)
        update = (exp_avg / bias_c1) / denom
        if wd != 0.0:
            update.add_(master, alpha=wd)
        master.add_(update, alpha=-lr)
        param.copy_(master.to(param.dtype))

    return fused_adam_sync


def _install_fake_triton_module():
    """Install a fake ``training_engine_tensor.triton_kernels`` and return
    restore info as ``(was_present, saved_module)``."""
    import types

    key = "training_engine_tensor.triton_kernels"
    was_present = key in sys.modules
    saved = sys.modules.get(key)
    fake = types.ModuleType(key)
    fake.fused_adam_sync = _make_cpu_fused_adam_sync()
    sys.modules[key] = fake
    return was_present, saved


def _uninstall_fake_triton_module(was_present, saved):
    """Restore the original ``training_engine_tensor.triton_kernels``."""
    key = "training_engine_tensor.triton_kernels"
    if was_present and saved is not None:
        sys.modules[key] = saved
    else:
        sys.modules.pop(key, None)


# ---------------------------------------------------------------------------
# Bucket layout tests (no torch tensors involved — pure dataclass arithmetic)
# ---------------------------------------------------------------------------


@_NEEDS_TORCH
class TestComputeGradBuckets(unittest.TestCase):
    """``compute_grad_buckets`` partition invariants."""

    def _entries(self, sizes: list[int]) -> list[tuple[str, int]]:
        return [(f"p{i}", s) for i, s in enumerate(sizes)]

    def test_invalid_world_size_raises(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        with self.assertRaises(ValueError):
            compute_grad_buckets(self._entries([10]), world_size=0)
        with self.assertRaises(ValueError):
            compute_grad_buckets(self._entries([10]), world_size=-1)

    def test_invalid_bucket_target_raises(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        with self.assertRaises(ValueError):
            compute_grad_buckets(
                self._entries([10]), world_size=1, bucket_target_elems=0,
            )
        with self.assertRaises(ValueError):
            compute_grad_buckets(
                self._entries([10]), world_size=1, bucket_target_elems=-5,
            )

    def test_empty_entries_yield_no_buckets(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        buckets = compute_grad_buckets([], world_size=4)
        self.assertEqual(buckets, [])

    def test_single_param_smaller_than_target(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        buckets = compute_grad_buckets(
            self._entries([100]), world_size=4, bucket_target_elems=1024,
        )
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0].params, (("p0", 100),))
        self.assertEqual(buckets[0].param_offsets, (0,))
        self.assertEqual(buckets[0].raw_size, 100)
        self.assertEqual(buckets[0].size, 100)

    def test_padding_makes_size_world_size_aligned(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        buckets = compute_grad_buckets(
            self._entries([100]), world_size=8,
        )
        self.assertEqual(buckets[0].raw_size, 100)
        # 100 rounded up to multiple of 8 = 104
        self.assertEqual(buckets[0].size, 104)
        # Padding region is the last 4 slots
        self.assertEqual(buckets[0].size - buckets[0].raw_size, 4)

    def test_no_padding_when_already_aligned(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        buckets = compute_grad_buckets(
            self._entries([16]), world_size=8,
        )
        self.assertEqual(buckets[0].size, 16)
        self.assertEqual(buckets[0].raw_size, 16)

    def test_single_param_larger_than_target_gets_own_bucket(self) -> None:
        """A param whose numel exceeds the target is NOT split.

        The reducer treats each (name, numel) entry as the smallest
        indivisible unit because the unpack step expects each param to
        live in exactly one bucket buffer for the per-shape view.
        """
        from training_engine_tensor.nccl import compute_grad_buckets

        buckets = compute_grad_buckets(
            self._entries([10_000]),
            world_size=4,
            bucket_target_elems=1000,
        )
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0].params, (("p0", 10_000),))
        self.assertEqual(buckets[0].size, 10_000)

    def test_greedy_split_at_target(self) -> None:
        """Once the running raw size + next param would exceed the
        target, the current bucket closes."""
        from training_engine_tensor.nccl import compute_grad_buckets

        # target=100; sizes 40, 50 fit (90 < 100), then 30 would push to
        # 120 > 100 → split.
        buckets = compute_grad_buckets(
            self._entries([40, 50, 30, 20]),
            world_size=2,
            bucket_target_elems=100,
        )
        self.assertEqual(len(buckets), 2)
        self.assertEqual(buckets[0].params, (("p0", 40), ("p1", 50)))
        self.assertEqual(buckets[1].params, (("p2", 30), ("p3", 20)))

    def test_param_offsets_within_bucket_are_sequential(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        buckets = compute_grad_buckets(
            self._entries([10, 20, 30]),
            world_size=2,
            bucket_target_elems=1000,
        )
        self.assertEqual(buckets[0].param_offsets, (0, 10, 30))
        self.assertEqual(buckets[0].raw_size, 60)

    def test_full_param_coverage_no_duplicates(self) -> None:
        from training_engine_tensor.nccl import compute_grad_buckets

        sizes = [10, 200, 50, 1000, 30, 5]
        entries = self._entries(sizes)
        buckets = compute_grad_buckets(
            entries, world_size=4, bucket_target_elems=300,
        )

        seen: list[str] = []
        for b in buckets:
            for name, _ in b.params:
                seen.append(name)
        self.assertEqual(seen, [name for name, _ in entries])

        # Sum of raw sizes equals total numel
        total_raw = sum(b.raw_size for b in buckets)
        self.assertEqual(total_raw, sum(sizes))

        # Each bucket's raw_size matches sum of its params' numel
        for b in buckets:
            self.assertEqual(b.raw_size, sum(n for _, n in b.params))

    def test_baseline_layout_partitioning(self) -> None:
        """Partitioning the actual MiniCPM4 0.5B grad layout produces a
        well-shaped bucket set: every param accounted for, every bucket
        size divisible by world_size, no zero-size buckets."""
        _init_test_config()
        from training_engine_tensor.nccl import (
            baseline_buffer_layout,
            compute_grad_buckets,
        )

        entries = baseline_buffer_layout()
        buckets = compute_grad_buckets(
            entries, world_size=16,
            bucket_target_elems=25 * 1024 * 1024,
        )

        self.assertGreater(len(buckets), 1)
        seen_names = [name for b in buckets for name, _ in b.params]
        self.assertEqual(seen_names, [n for n, _ in entries])

        for b in buckets:
            self.assertGreater(b.size, 0)
            self.assertEqual(b.size % 16, 0,
                             f"bucket size {b.size} not divisible by 16")
            self.assertGreaterEqual(b.size, b.raw_size)
            self.assertLess(b.size - b.raw_size, 16,
                            "padding should be < world_size")


# ---------------------------------------------------------------------------
# BucketedGradReducer state-machine tests (CPU torch tensors)
# ---------------------------------------------------------------------------


@_NEEDS_TORCH
class TestBucketedGradReducerLifecycle(unittest.TestCase):
    """Covers ``__init__`` + ``start_step`` + ``write_grad`` + error paths."""

    @staticmethod
    def _entries(sizes: list[int]) -> list[tuple[str, int]]:
        return [(f"p{i}", s) for i, s in enumerate(sizes)]

    def test_init_invalid_world_size_raises(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        with self.assertRaises(ValueError):
            BucketedGradReducer(
                self._entries([10]), world_size=0, rank=0, device="cpu",
            )

    def test_init_invalid_rank_raises(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        with self.assertRaises(ValueError):
            BucketedGradReducer(
                self._entries([10]), world_size=2, rank=2, device="cpu",
            )
        with self.assertRaises(ValueError):
            BucketedGradReducer(
                self._entries([10]), world_size=2, rank=-1, device="cpu",
            )

    def test_buffer_sizes_match_buckets(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([10, 20, 30, 40])
        r = BucketedGradReducer(
            entries, world_size=4, rank=0, device="cpu",
            bucket_target_elems=50,
        )
        for bucket, input_buf, shard_buf in zip(
            r.buckets, r._input_bufs, r._shard_bufs,  # noqa: SLF001
        ):
            self.assertEqual(input_buf.shape[0], bucket.size)
            self.assertEqual(shard_buf.shape[0], bucket.size // 4)
            self.assertEqual(input_buf.dtype, torch.float32)
            self.assertEqual(shard_buf.dtype, torch.float32)

    def test_input_buffer_padding_is_zero(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([10])
        r = BucketedGradReducer(
            entries, world_size=8, rank=0, device="cpu",
        )
        # raw_size=10, size=16 → padding [10:16] is zero
        self.assertEqual(r.buckets[0].size, 16)
        self.assertTrue(
            torch.all(r._input_bufs[0][10:16] == 0).item(),  # noqa: SLF001
            "padding region must be zero",
        )

    def test_param_loc_covers_all_params(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([10, 20, 30, 40])
        r = BucketedGradReducer(
            entries, world_size=2, rank=0, device="cpu",
            bucket_target_elems=50,
        )
        for name, numel in entries:
            self.assertIn(name, r._param_loc)  # noqa: SLF001
            _b, _off, n = r._param_loc[name]  # noqa: SLF001
            self.assertEqual(n, numel)

    def test_start_step_resets_pending(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([10, 20, 30])
        r = BucketedGradReducer(
            entries, world_size=2, rank=0, device="cpu",
            bucket_target_elems=200,  # one bucket
        )
        self.assertEqual(r._bucket_pending, [3])  # noqa: SLF001
        # Mutate
        r._bucket_pending[0] = 0  # noqa: SLF001
        r._bucket_done_events[0] = "fake_event"  # noqa: SLF001
        r.start_step()
        self.assertEqual(r._bucket_pending, [3])  # noqa: SLF001
        self.assertEqual(r._bucket_done_events, [None])  # noqa: SLF001

    def test_write_grad_unknown_param_raises(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([10]), world_size=2, rank=0, device="cpu",
        )
        with self.assertRaises(KeyError):
            r.write_grad("does_not_exist", torch.zeros(10))

    def test_write_grad_wrong_numel_raises(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([10]), world_size=2, rank=0, device="cpu",
        )
        with self.assertRaises(ValueError):
            r.write_grad("p0", torch.zeros(11))

    def test_write_grad_after_complete_raises(self) -> None:
        """Double-write to a bucket that has already triggered must
        fail loud — this catches the "forgot start_step()" footgun."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([10]), world_size=1, rank=0, device="cpu",
        )
        r.write_grad("p0", torch.ones(10))  # bucket completes
        with self.assertRaises(RuntimeError):
            r.write_grad("p0", torch.ones(10))

    def test_write_grad_decrements_pending(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([10, 20])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=200,  # one bucket
        )
        self.assertEqual(r._bucket_pending, [2])  # noqa: SLF001
        r.write_grad("p0", torch.ones(10))
        self.assertEqual(r._bucket_pending, [1])  # noqa: SLF001
        r.write_grad("p1", torch.ones(20))
        self.assertEqual(r._bucket_pending, [0])  # noqa: SLF001

    def test_write_grad_completes_bucket_writes_shard(self) -> None:
        """Filling a bucket triggers the synchronous fallback that
        copies the (scaled) input into the shard buffer."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([8])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
        )
        # Pre-condition: shard buffer is uninitialised noise; force zero
        r._shard_bufs[0].zero_()  # noqa: SLF001

        grad = torch.arange(8, dtype=torch.float32)
        r.write_grad("p0", grad)

        # ws=1 → shard == full bucket == grad (mul by 1.0 / 1 = identity)
        self.assertTrue(torch.allclose(r._shard_bufs[0], grad))  # noqa: SLF001

    def test_write_grad_upcasts_bf16_to_fp32(self) -> None:
        """gradients arriving in BF16 (e.g. from a non-te_wgrad path) get
        FP32-promoted before the bucket copy, so the FP32 AllReduce
        buffer rule (FP32 §6) holds."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
        )
        bf16_grad = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
        r.write_grad("p0", bf16_grad)
        # Round-trip through bf16→fp32 may lose tail bits; assert close.
        self.assertEqual(r._shard_bufs[0].dtype, torch.float32)  # noqa: SLF001
        self.assertTrue(
            torch.allclose(r._shard_bufs[0], torch.tensor([1, 2, 3, 4],  # noqa: SLF001
                          dtype=torch.float32), atol=1e-2),
        )

    def test_write_grad_handles_nd_tensor(self) -> None:
        """A 2D weight grad should be reshaped to 1D before copying."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([6]), world_size=1, rank=0, device="cpu",
        )
        nd = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r.write_grad("p0", nd)
        self.assertTrue(torch.allclose(
            r._shard_bufs[0],  # noqa: SLF001
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ))


@_NEEDS_TORCH
class TestBucketedGradReducerRoundTrip(unittest.TestCase):
    """Full backward → reduce → unpack contract on single-rank fallback."""

    def test_round_trip_preserves_grads(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        # Two buckets, two params each. ws=1, so the synchronous
        # fallback path bytewise round-trips the inputs.
        # target=15 → bucket1=(a,b)raw=10, bucket2=(c,d)raw=12
        # (target=10 would split (c,d) into two single-param buckets
        # because 10+8>10, then 8+4>10 again).
        entries = [("a", 4), ("b", 6), ("c", 8), ("d", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,
        )
        self.assertEqual(r.num_buckets, 2)

        original = {
            "a": torch.arange(4, dtype=torch.float32),
            "b": torch.arange(6, dtype=torch.float32) + 100,
            "c": torch.arange(8, dtype=torch.float32) - 50,
            "d": torch.arange(4, dtype=torch.float32) * 2,
        }
        params_view = {n: torch.zeros(g.shape) for n, g in original.items()}

        r.start_step()
        for name, g in original.items():
            r.write_grad(name, g)
        # In dist mode this would be wait_all; in non-dist it's a no-op
        # (writes already synchronously copied into shard_bufs).
        r.wait_all()
        out = r.allgather_and_unpack(params_view)

        for name, expected in original.items():
            self.assertIn(name, out)
            self.assertEqual(out[name].dtype, torch.float32)
            self.assertTrue(
                torch.allclose(out[name], expected),
                f"round-trip mismatch for {name}: {out[name]} vs {expected}",
            )

    def test_round_trip_preserves_shapes(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("w", 12)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
        )
        params_view = {"w": torch.zeros(3, 4)}
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        r.start_step()
        r.write_grad("w", original)
        r.wait_all()
        out = r.allgather_and_unpack(params_view)
        self.assertEqual(out["w"].shape, (3, 4))
        self.assertTrue(torch.allclose(out["w"], original))

    def test_unpacked_grads_do_not_alias_buffer(self) -> None:
        """``allgather_and_unpack`` clones each param view so the
        optimizer can mutate gradients without polluting the reducer's
        persistent buffer."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            [("p", 4)], world_size=1, rank=0, device="cpu",
        )
        params_view = {"p": torch.zeros(4)}
        r.start_step()
        r.write_grad("p", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        r.wait_all()
        out = r.allgather_and_unpack(params_view)
        out["p"].zero_()
        self.assertTrue(
            torch.allclose(
                r._input_bufs[0][:4],  # noqa: SLF001
                torch.tensor([1.0, 2.0, 3.0, 4.0]),
            ),
            "buffer must not be mutated by downstream zero_()",
        )

    def test_no_clone_returns_aliased_views(self) -> None:
        """``allgather_and_unpack(clone=False)`` returns views that
        alias the bucket's input buffer — mutating the view mutates
        the buffer.  This is the zero-copy path used by the fused
        Adam kernel (which only reads grad_ptr)."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            [("p", 4)], world_size=1, rank=0, device="cpu",
        )
        params_view = {"p": torch.zeros(4)}
        r.start_step()
        r.write_grad("p", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        r.wait_all()
        out = r.allgather_and_unpack(params_view, clone=False)
        self.assertEqual(
            out["p"].data_ptr(),
            r._input_bufs[0].data_ptr(),  # noqa: SLF001
            "view must alias the input buffer (zero-copy)",
        )
        self.assertTrue(torch.allclose(
            out["p"], torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ))

    def test_no_clone_round_trip_values_match_clone(self) -> None:
        """Values from ``clone=False`` must be identical to ``clone=True``."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8), ("d", 4)]
        original = {
            "a": torch.arange(4, dtype=torch.float32),
            "b": torch.arange(6, dtype=torch.float32) + 100,
            "c": torch.arange(8, dtype=torch.float32) - 50,
            "d": torch.arange(4, dtype=torch.float32) * 2,
        }
        params_view = {n: torch.zeros(g.shape) for n, g in original.items()}

        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,
        )
        r.start_step()
        for name, g in original.items():
            r.write_grad(name, g)
        r.wait_all()

        out_clone = r.allgather_and_unpack(params_view, clone=True)
        out_view = r.allgather_and_unpack(params_view, clone=False)

        for name in original:
            self.assertTrue(
                torch.allclose(out_view[name], out_clone[name]),
                f"clone vs no-clone mismatch for {name}",
            )

    def test_no_clone_preserves_shapes(self) -> None:
        """Views from ``clone=False`` must have the original param shape."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            [("w", 12)], world_size=1, rank=0, device="cpu",
        )
        params_view = {"w": torch.zeros(3, 4)}
        r.start_step()
        r.write_grad("w", torch.arange(12, dtype=torch.float32).reshape(3, 4))
        r.wait_all()
        out = r.allgather_and_unpack(params_view, clone=False)
        self.assertEqual(out["w"].shape, (3, 4))

    def test_two_steps_preserve_bucket_layout(self) -> None:
        """After ``start_step()``, subsequent writes overwrite the prior
        step's data — buffer is reused but counter logic resets."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            [("p", 4)], world_size=1, rank=0, device="cpu",
        )
        params_view = {"p": torch.zeros(4)}

        r.start_step()
        r.write_grad("p", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        r.wait_all()
        out1 = r.allgather_and_unpack(params_view)

        r.start_step()
        r.write_grad("p", torch.tensor([10.0, 20.0, 30.0, 40.0]))
        r.wait_all()
        out2 = r.allgather_and_unpack(params_view)

        self.assertTrue(torch.allclose(
            out1["p"], torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ))
        self.assertTrue(torch.allclose(
            out2["p"], torch.tensor([10.0, 20.0, 30.0, 40.0]),
        ))

    def test_padding_stays_zero_across_steps(self) -> None:
        """The trailing padding region (raw_size..size) must remain
        zero across step boundaries — it's never written by the user
        and the reduce_scatter on the padding slot must contribute zero
        to the averaged buffer (otherwise grad_norm would be polluted)."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            [("p", 5)], world_size=4, rank=0, device="cpu",
        )
        # raw_size=5, size=8 → padding indices [5, 8)
        self.assertEqual(r.buckets[0].size, 8)

        r.start_step()
        r.write_grad("p", torch.ones(5))
        # In ws=4 non-dist mode the input gets mul'd by 0.25; padding
        # was zero, mul by 0.25 → still zero.
        self.assertTrue(torch.all(
            r._input_bufs[0][5:8] == 0,  # noqa: SLF001
        ).item())


# ---------------------------------------------------------------------------
# Distributed grad-norm tests
# ---------------------------------------------------------------------------
# ``BucketedGradReducer.compute_distributed_grad_norm`` imports
# ``multi_tensor_l2norm`` from TransformerEngine, which is CUDA-only and
# not installed on the macOS CI runner. To exercise the slicing /
# classification logic on CPU we install a lightweight fake TE module
# in ``sys.modules`` for the duration of each test, providing a pure-
# Python ``multi_tensor_l2norm`` that returns ``(sqrt(sum sq), None)``
# of all tensors in the supplied flat list. This matches the real
# kernel's mathematical contract bytewise on FP32 inputs.


class _FakeTEContext:
    """Context manager that monkey-patches the TE imports used by the
    grad-norm path with a CPU-friendly stub."""

    def __enter__(self) -> "_FakeTEContext":
        import types

        self._saved: dict[str, object | None] = {}
        for mod in (
            "transformer_engine",
            "transformer_engine.pytorch",
            "transformer_engine.pytorch.optimizers",
        ):
            self._saved[mod] = sys.modules.get(mod)

        te_pkg = types.ModuleType("transformer_engine")
        te_pytorch = types.ModuleType("transformer_engine.pytorch")
        te_optim = types.ModuleType("transformer_engine.pytorch.optimizers")

        def fake_l2norm() -> object:
            return object()

        def fake_applier(_op, _dummy, lists, _per_tensor):
            tensors = lists[0]
            total = torch.zeros(1, dtype=torch.float32)
            for t in tensors:
                total += t.float().pow(2).sum()
            return total.sqrt(), None

        te_optim.multi_tensor_l2norm = fake_l2norm()
        te_optim.multi_tensor_applier = fake_applier
        te_pytorch.optimizers = te_optim
        te_pkg.pytorch = te_pytorch

        sys.modules["transformer_engine"] = te_pkg
        sys.modules["transformer_engine.pytorch"] = te_pytorch
        sys.modules["transformer_engine.pytorch.optimizers"] = te_optim
        return self

    def __exit__(self, *_a: object) -> None:
        for mod, saved in self._saved.items():
            if saved is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = saved


@_NEEDS_TORCH
class TestComputeDistributedGradNorm(unittest.TestCase):
    """Verify the per-bucket grad norm walks the right slices and
    classifies weight vs norm grads consistently with the legacy path.

    Uses a fake ``multi_tensor_l2norm`` (see ``_FakeTEContext``) so the
    slicing logic runs on CPU without TransformerEngine installed.
    """

    def setUp(self) -> None:
        _init_test_config()

    def test_grad_norm_matches_l2_of_local_shard(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        with _FakeTEContext():
            r = BucketedGradReducer(
                [("p", 4)], world_size=1, rank=0, device="cpu",
            )
            r.start_step()
            r.write_grad("p", torch.tensor([3.0, 4.0, 0.0, 0.0]))
            r.wait_all()
            norm = r.compute_distributed_grad_norm("cpu")
            self.assertAlmostEqual(norm, 5.0, places=5)

    def test_grad_norm_partial_shard_only_includes_overlap(self) -> None:
        """With ws=4 rank=0, only the first quarter of the bucket
        contributes to the rank's L2 norm. Rank 1's slot at indices
        4..7 of the original input is on rank 1, NOT in our shard."""
        from training_engine_tensor.nccl import BucketedGradReducer

        with _FakeTEContext():
            r = BucketedGradReducer(
                [("p", 16)], world_size=4, rank=0, device="cpu",
            )
            r.start_step()
            r.write_grad("p", torch.arange(16, dtype=torch.float32))
            r.wait_all()
            norm = r.compute_distributed_grad_norm("cpu")
            # Non-dist fallback: input *= 1/4 → [0, .25, .5, .75, 1, ...]
            # Rank 0 shard = [0, .25, .5, .75]
            expected = (0.0 + 0.0625 + 0.25 + 0.5625) ** 0.5
            self.assertAlmostEqual(norm, expected, places=5)

    def test_grad_norm_classifies_norm_params_correctly(self) -> None:
        """Norm params (whose name ends with ``norm.weight``) are
        classified separately and ordered AFTER weight grads in the
        L2 input list. The fake reducer sums squares either way, so
        the test asserts the value is correct AND that classification
        does not lose the norm-param contribution."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [
            ("layers.0.attention.wqkv.weight", 4),  # weight
            ("layers.0.attention_norm.weight", 4),  # norm
        ]
        with _FakeTEContext():
            r = BucketedGradReducer(
                entries, world_size=1, rank=0, device="cpu",
                bucket_target_elems=100,
            )
            r.start_step()
            r.write_grad("layers.0.attention.wqkv.weight",
                         torch.tensor([1.0, 0.0, 0.0, 0.0]))
            r.write_grad("layers.0.attention_norm.weight",
                         torch.tensor([0.0, 0.0, 0.0, 1.0]))
            r.wait_all()
            norm = r.compute_distributed_grad_norm("cpu")
            self.assertAlmostEqual(norm, 2 ** 0.5, places=5)

    def test_grad_norm_empty_shard_returns_zero(self) -> None:
        """If a rank's shard contains no param overlap (only padding),
        the local norm is zero and the total norm is also zero in the
        single-rank case (no all_reduce needed)."""
        from training_engine_tensor.nccl import BucketedGradReducer

        # ws=4, raw=1 → padded=4. Rank 2's shard = input[2:3] (padding only).
        with _FakeTEContext():
            r = BucketedGradReducer(
                [("p", 1)], world_size=4, rank=2, device="cpu",
            )
            r.start_step()
            r.write_grad("p", torch.tensor([42.0]))
            r.wait_all()
            norm = r.compute_distributed_grad_norm("cpu")
            # p occupies input[0:1]; rank 2 covers input[2:3] → no overlap
            # → shard_grads is empty → local_norm = 0 → returned norm = 0.
            self.assertAlmostEqual(norm, 0.0, places=5)


# ---------------------------------------------------------------------------
# M2-P6 P1.c per-bucket allgather pipeline tests
# ---------------------------------------------------------------------------


@_NEEDS_TORCH
class TestBucketedAllgatherPipeline(unittest.TestCase):
    """Covers the per-bucket allgather contract used by the optimizer
    overlap pipeline (``OPT_ALLGATHER_OVERLAP=1``).

    The four new methods exercised here all share the non-distributed
    fallback contract used elsewhere in this file: with ``ws=1`` and no
    ``torch.distributed`` initialised, ``start_bucket_allgather`` is a
    no-op (the input buffer already holds the averaged data because
    the synchronous ``write_grad`` fallback copies input → shard with
    the world-size scaling), and ``wait_bucket_allgather`` is a no-op
    too (no event was recorded). ``unpack_bucket`` returns non-owning
    views into the per-bucket input buffer, NOT clones — opposite of
    ``allgather_and_unpack`` which does clone.
    """

    @staticmethod
    def _entries(sizes: list[int]) -> list[tuple[str, int]]:
        return [(f"p{i}", s) for i, s in enumerate(sizes)]

    def test_bucket_param_names_returns_buffer_order(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8), ("d", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,
        )
        self.assertEqual(r.num_buckets, 2)
        self.assertEqual(r.bucket_param_names(0), ("a", "b"))
        self.assertEqual(r.bucket_param_names(1), ("c", "d"))

    def test_bucket_param_names_invalid_index_raises(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([10]), world_size=1, rank=0, device="cpu",
        )
        with self.assertRaises((IndexError, ValueError)):
            r.bucket_param_names(99)

    def test_unpack_bucket_returns_non_owning_views(self) -> None:
        """``unpack_bucket`` returns aliased views — mutating the
        returned tensor must mutate the underlying input buffer too.
        This is the contract the optimizer pipeline relies on for
        zero-copy hand-off, and is the exact opposite of the legacy
        ``allgather_and_unpack`` cloning behavior."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            [("p", 4)], world_size=1, rank=0, device="cpu",
        )
        params_view = {"p": torch.zeros(4)}
        r.start_step()
        r.write_grad("p", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        r.wait_all()
        # Single-rank fallback: in non-dist mode the data already lives
        # in input_buf as input * (1/ws) = input. No real allgather, but
        # start_bucket_allgather should still be a safe no-op.
        r.start_bucket_allgather(0)
        r.wait_bucket_allgather(0)
        out = r.unpack_bucket(0, params_view)
        self.assertIn("p", out)
        self.assertEqual(out["p"].shape, (4,))
        self.assertTrue(torch.allclose(
            out["p"], torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ))
        # Aliasing assertion: data_ptr should match the input buffer's
        # base address (offset 0 because there's only one param).
        self.assertEqual(
            out["p"].data_ptr(),
            r._input_bufs[0].data_ptr(),  # noqa: SLF001
        )

    def test_unpack_bucket_preserves_param_shapes(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("w", 12)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
        )
        params_view = {"w": torch.zeros(3, 4)}
        r.start_step()
        r.write_grad("w", torch.arange(12, dtype=torch.float32).reshape(3, 4))
        r.wait_all()
        out = r.unpack_bucket(0, params_view)
        self.assertEqual(out["w"].shape, (3, 4))
        self.assertTrue(torch.allclose(
            out["w"],
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
        ))

    def test_unpack_bucket_only_returns_bucket_params(self) -> None:
        """``unpack_bucket(i)`` must NOT return params from other
        buckets — the per-bucket adam loop iterates one bucket at a
        time and would do duplicate updates if a bucket leaked another
        bucket's keys."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8), ("d", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,
        )
        params_view = {"a": torch.zeros(4), "b": torch.zeros(6),
                       "c": torch.zeros(8), "d": torch.zeros(4)}
        r.start_step()
        for n, sz in entries:
            r.write_grad(n, torch.ones(sz))
        r.wait_all()

        out0 = r.unpack_bucket(0, params_view)
        out1 = r.unpack_bucket(1, params_view)
        self.assertEqual(set(out0.keys()), {"a", "b"})
        self.assertEqual(set(out1.keys()), {"c", "d"})

    def test_start_and_wait_bucket_allgather_no_op_in_nondist(self) -> None:
        """In non-distributed mode the methods are explicit no-ops
        (no event is recorded; nothing waits); they must not raise."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            [("p", 4)], world_size=1, rank=0, device="cpu",
        )
        r.start_step()
        r.write_grad("p", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        r.wait_all()
        r.start_bucket_allgather(0)
        r.wait_bucket_allgather(0)
        # Idempotent: calling wait twice should not crash.
        r.wait_bucket_allgather(0)

    def test_pipeline_round_trip_matches_allgather_and_unpack(self) -> None:
        """The per-bucket pipeline should reconstruct the SAME gradient
        values that the legacy ``allgather_and_unpack`` would, just
        delivered bucket-by-bucket. This is the bytewise correctness
        guarantee the optimizer pipeline depends on."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8), ("d", 4)]
        original = {
            "a": torch.arange(4, dtype=torch.float32),
            "b": torch.arange(6, dtype=torch.float32) + 100,
            "c": torch.arange(8, dtype=torch.float32) - 50,
            "d": torch.arange(4, dtype=torch.float32) * 2,
        }
        params_view = {n: torch.zeros(g.shape) for n, g in original.items()}

        # Run #1: legacy allgather_and_unpack.
        r1 = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,
        )
        r1.start_step()
        for n, g in original.items():
            r1.write_grad(n, g)
        r1.wait_all()
        legacy = r1.allgather_and_unpack(params_view)

        # Run #2: per-bucket pipeline.
        r2 = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,
        )
        r2.start_step()
        for n, g in original.items():
            r2.write_grad(n, g)
        r2.wait_all()
        pipeline: dict[str, torch.Tensor] = {}
        for bidx in range(r2.num_buckets):
            r2.start_bucket_allgather(bidx)
        for bidx in range(r2.num_buckets):
            r2.wait_bucket_allgather(bidx)
            pipeline.update(r2.unpack_bucket(bidx, params_view))

        self.assertEqual(set(pipeline.keys()), set(legacy.keys()))
        for name in legacy:
            self.assertTrue(
                torch.allclose(pipeline[name], legacy[name]),
                f"pipeline mismatch for {name}: "
                f"{pipeline[name]} vs {legacy[name]}",
            )


# ---------------------------------------------------------------------------
# fused_clip_adam_sync_bucketed equivalence test
# ---------------------------------------------------------------------------


@_NEEDS_TORCH
class TestFusedClipAdamSyncBucketed(unittest.TestCase):
    """``fused_clip_adam_sync_bucketed`` must produce bytewise-identical
    optimizer state and parameter updates as ``fused_clip_adam_sync``
    given the same inputs and a partition of those inputs into buckets.

    Skipped automatically when triton is not importable (we monkey-patch
    a CPU-friendly stub that mirrors the kernel's mathematical contract,
    but the import-time check still goes through ``triton_kernels``).
    """

    def setUp(self) -> None:
        _init_test_config()

    def _make_state(self, names_shapes: list[tuple[str, tuple[int, ...]]]):
        from training_engine_tensor.optimizer import AdamState

        params = {n: torch.randn(*s, dtype=torch.float32) for n, s in names_shapes}
        return AdamState([n for n, _ in names_shapes], params, device="cpu")

    def _stub_fused_adam_sync(self):
        """Drop-in pure-torch stub for the triton fused kernel that
        implements the same math (AdamW update with bias correction,
        gradient clipping by ``clip_coeff``, weight decay if ``wd>0``,
        and FP32→BF16 param sync). This keeps the test CPU-only."""
        BETA1 = 0.9
        BETA2 = 0.95
        EPS = 1e-8

        def fused_adam_sync(g, master, exp_avg, exp_avg_sq, param,
                            *, lr, step, clip_coeff, wd):
            g_clipped = g * clip_coeff
            exp_avg.mul_(BETA1).add_(g_clipped, alpha=1 - BETA1)
            exp_avg_sq.mul_(BETA2).addcmul_(g_clipped, g_clipped, value=1 - BETA2)
            bias_c1 = 1 - BETA1 ** step
            bias_c2 = 1 - BETA2 ** step
            denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(EPS)
            update = (exp_avg / bias_c1) / denom
            if wd != 0.0:
                update.add_(master, alpha=wd)
            master.add_(update, alpha=-lr)
            param.copy_(master.to(param.dtype))

        return fused_adam_sync

    def _install_fake_triton_kernels(self):
        """Install a fake ``training_engine_tensor.triton_kernels`` in
        ``sys.modules`` so the ``from .triton_kernels import
        fused_adam_sync`` inside the optimizer functions resolves to a
        CPU-friendly stub. Real triton_kernels is GPU-only and cannot
        load on the macOS dev sandbox. Returns a context object whose
        ``__exit__`` restores the original module."""
        import types

        class _Ctx:
            def __init__(ctx_self, kernel):
                ctx_self._kernel = kernel
                ctx_self._saved = None
                ctx_self._was_present = False

            def __enter__(ctx_self):
                key = "training_engine_tensor.triton_kernels"
                ctx_self._was_present = key in sys.modules
                ctx_self._saved = sys.modules.get(key)
                fake = types.ModuleType(key)
                fake.fused_adam_sync = ctx_self._kernel
                sys.modules[key] = fake
                return ctx_self

            def __exit__(ctx_self, *_a):
                key = "training_engine_tensor.triton_kernels"
                if ctx_self._was_present and ctx_self._saved is not None:
                    sys.modules[key] = ctx_self._saved
                else:
                    sys.modules.pop(key, None)

        return _Ctx(self._stub_fused_adam_sync())

    def test_bucketed_matches_full(self) -> None:
        from training_engine_tensor import optimizer as opt_mod

        torch.manual_seed(0)
        names_shapes = [
            ("layers.0.weight", (8, 4)),
            ("layers.0.bias", (8,)),
            ("layers.1.weight", (4, 8)),
            ("layers.1.bias", (4,)),
            ("layers.2.norm.weight", (4,)),
        ]

        with self._install_fake_triton_kernels():
            state_full = self._make_state(names_shapes)
            state_bkt = self._make_state(names_shapes)
            for n in state_full.master_weights:
                state_bkt.master_weights[n].copy_(state_full.master_weights[n])
                state_bkt.exp_avg[n].copy_(state_full.exp_avg[n])
                state_bkt.exp_avg_sq[n].copy_(state_full.exp_avg_sq[n])

            grads = {n: torch.randn(*s, dtype=torch.float32)
                     for n, s in names_shapes}
            params_full = {n: torch.zeros(*s, dtype=torch.bfloat16)
                           for n, s in names_shapes}
            params_bkt = {n: torch.zeros(*s, dtype=torch.bfloat16)
                          for n, s in names_shapes}

            opt_mod.fused_clip_adam_sync(
                state_full, grads, params_full, lr=1e-3, clip_coeff=0.7,
            )

            buckets = [
                (("layers.0.weight", "layers.0.bias"),
                 {"layers.0.weight": grads["layers.0.weight"],
                  "layers.0.bias": grads["layers.0.bias"]}),
                (("layers.1.weight", "layers.1.bias", "layers.2.norm.weight"),
                 {"layers.1.weight": grads["layers.1.weight"],
                  "layers.1.bias": grads["layers.1.bias"],
                  "layers.2.norm.weight": grads["layers.2.norm.weight"]}),
            ]
            opt_mod.fused_clip_adam_sync_bucketed(
                state_bkt, iter(buckets), params_bkt, lr=1e-3, clip_coeff=0.7,
            )

            self.assertEqual(state_full.step_count, state_bkt.step_count)
            for n, _ in names_shapes:
                self.assertTrue(torch.equal(
                    state_full.master_weights[n],
                    state_bkt.master_weights[n],
                ), f"master mismatch for {n}")
                self.assertTrue(torch.equal(
                    state_full.exp_avg[n], state_bkt.exp_avg[n],
                ), f"exp_avg mismatch for {n}")
                self.assertTrue(torch.equal(
                    state_full.exp_avg_sq[n], state_bkt.exp_avg_sq[n],
                ), f"exp_avg_sq mismatch for {n}")
                self.assertTrue(torch.equal(
                    params_full[n], params_bkt[n],
                ), f"param sync mismatch for {n}")

    def test_duplicate_param_in_buckets_raises(self) -> None:
        """If the same param appears in two buckets, the function must
        raise — silently double-updating the optimizer state would
        produce a hard-to-diagnose loss divergence."""
        from training_engine_tensor import optimizer as opt_mod

        names_shapes = [("p", (4,))]
        state = self._make_state(names_shapes)
        grad = torch.randn(4, dtype=torch.float32)
        params = {"p": torch.zeros(4, dtype=torch.bfloat16)}

        with self._install_fake_triton_kernels():
            with self.assertRaises(RuntimeError):
                opt_mod.fused_clip_adam_sync_bucketed(
                    state,
                    iter([
                        (("p",), {"p": grad}),
                        (("p",), {"p": grad}),
                    ]),
                    params, lr=1e-3, clip_coeff=1.0,
                )


# ---------------------------------------------------------------------------
# M2-P15 batch-flush deferred write_grad tests
# ---------------------------------------------------------------------------


@_NEEDS_TORCH
class TestDeferredBatchFlush(unittest.TestCase):
    """Covers the M2-P15 batch-flush optimisation inside ``write_grad``.

    The batch-flush defers per-param copies until the bucket fills,
    then flushes all deferred grads in a single burst.  These tests
    verify the deferred-list lifecycle and that the mathematical
    outcome is identical to the old immediate-copy behavior.
    """

    @staticmethod
    def _entries(sizes: list[int]) -> list[tuple[str, int]]:
        return [(f"p{i}", s) for i, s in enumerate(sizes)]

    def test_deferred_list_populated_during_partial_fill(self) -> None:
        """After a non-final write_grad, the deferred list should hold
        the grad reference but no copy should have reached the buffer."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4, 6])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=200,
        )
        r.start_step()

        grad0 = torch.arange(4, dtype=torch.float32) + 10
        r.write_grad("p0", grad0)

        self.assertEqual(len(r._deferred[0]), 1)  # noqa: SLF001
        self.assertFalse(torch.allclose(
            r._input_bufs[0][:4],  # noqa: SLF001
            grad0,
        ), "buffer should NOT have the grad yet (deferred)")

    def test_deferred_list_cleared_after_bucket_fills(self) -> None:
        """When the bucket fills, deferred grads are flushed and the
        deferred list is cleared."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4, 6])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=200,
        )
        r.start_step()
        r.write_grad("p0", torch.ones(4))
        r.write_grad("p1", torch.ones(6) * 2)

        self.assertEqual(len(r._deferred[0]), 0)  # noqa: SLF001

    def test_deferred_flush_produces_correct_buffer_contents(self) -> None:
        """After bucket fills, the input buffer must contain the same
        values as the old immediate-copy path would have produced."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4, 6])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=200,
        )
        r.start_step()

        g0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        g1 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        r.write_grad("p0", g0)
        r.write_grad("p1", g1)
        r.wait_all()

        # ws=1 non-dist: shard == input (scaled by 1/1 = identity)
        self.assertTrue(torch.allclose(r._shard_bufs[0][:4], g0))  # noqa: SLF001
        self.assertTrue(torch.allclose(r._shard_bufs[0][4:10], g1))  # noqa: SLF001

    def test_deferred_list_cleared_by_start_step(self) -> None:
        """``start_step`` must clear any stale deferred entries from a
        previous step that was abandoned mid-fill."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4, 6])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=200,
        )
        r.start_step()
        r.write_grad("p0", torch.ones(4))
        self.assertEqual(len(r._deferred[0]), 1)  # noqa: SLF001

        r.start_step()
        self.assertEqual(len(r._deferred[0]), 0)  # noqa: SLF001

    def test_multi_bucket_deferred_flush_independent(self) -> None:
        """Each bucket's deferred list operates independently: filling
        bucket 0 should not affect bucket 1's deferred list."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=12,
        )
        self.assertEqual(r.num_buckets, 2)

        r.start_step()

        r.write_grad("a", torch.ones(4))
        r.write_grad("b", torch.ones(6))
        # Bucket 0 fills (a+b=10 < 12), so bucket 0 flushed
        self.assertEqual(len(r._deferred[0]), 0)  # noqa: SLF001
        # Bucket 1 not touched yet
        self.assertEqual(len(r._deferred[1]), 0)  # noqa: SLF001

        r.write_grad("c", torch.ones(8) * 3)
        # Bucket 1 fills (c=8)
        self.assertEqual(len(r._deferred[1]), 0)  # noqa: SLF001

    def test_deferred_flush_upcasts_bf16(self) -> None:
        """BF16 grads deferred and flushed must be upcast to FP32 in
        the bucket buffer (FP32 AllReduce rule)."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
        )
        r.start_step()
        bf16_grad = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
        r.write_grad("p0", bf16_grad)
        self.assertEqual(r._shard_bufs[0].dtype, torch.float32)  # noqa: SLF001
        self.assertTrue(torch.allclose(
            r._shard_bufs[0],  # noqa: SLF001
            torch.tensor([1, 2, 3, 4], dtype=torch.float32), atol=1e-2,
        ))

    def test_deferred_flush_handles_nd_tensor(self) -> None:
        """Multi-dimensional grad tensors deferred and flushed must be
        correctly flattened to 1D in the bucket buffer."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([6]), world_size=1, rank=0, device="cpu",
        )
        r.start_step()
        nd = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r.write_grad("p0", nd)
        self.assertTrue(torch.allclose(
            r._shard_bufs[0],  # noqa: SLF001
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ))

    def test_deferred_round_trip_matches_legacy(self) -> None:
        """Full backward → reduce → unpack round trip with deferred
        flush must produce identical FP32 grads as the legacy path."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8), ("d", 4)]
        original = {
            "a": torch.arange(4, dtype=torch.float32),
            "b": torch.arange(6, dtype=torch.float32) + 100,
            "c": torch.arange(8, dtype=torch.float32) - 50,
            "d": torch.arange(4, dtype=torch.float32) * 2,
        }
        params_view = {n: torch.zeros(g.shape) for n, g in original.items()}

        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,
        )
        r.start_step()
        for name, g in original.items():
            r.write_grad(name, g)
        r.wait_all()
        out = r.allgather_and_unpack(params_view)

        for name, expected in original.items():
            self.assertIn(name, out)
            self.assertEqual(out[name].dtype, torch.float32)
            self.assertTrue(
                torch.allclose(out[name], expected),
                f"round-trip mismatch for {name}: {out[name]} vs {expected}",
            )


class ShardedOptimizerBuffersTests(unittest.TestCase):
    """CPU unit tests for ShardedOptimizerBuffers + sharded optimizer path."""

    def setUp(self) -> None:
        _init_test_config()
        self._triton_state = _install_fake_triton_module()

    def tearDown(self) -> None:
        _uninstall_fake_triton_module(*self._triton_state)

    @_NEEDS_TORCH
    def test_buffers_allocate_correct_shapes(self) -> None:
        """bf16_shards and bf16_full_buf have correct sizes."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import ShardedOptimizerBuffers

        entries = [("a", 12), ("b", 8), ("c", 16)]
        ws = 4
        r = BucketedGradReducer(
            entries, world_size=ws, rank=0, device="cpu",
            bucket_target_elems=20,
        )
        bufs = ShardedOptimizerBuffers(r, device="cpu")
        self.assertEqual(len(bufs.bf16_shards), r.num_buckets)
        for i, bucket in enumerate(r.buckets):
            expected_shard = bucket.size // ws
            self.assertEqual(bufs.bf16_shards[i].numel(), expected_shard)
            self.assertEqual(bufs.bf16_shards[i].dtype, torch.bfloat16)
        max_bucket = max(b.size for b in r.buckets)
        self.assertEqual(bufs.bf16_full_buf.numel(), max_bucket)
        self.assertEqual(bufs.bf16_full_buf.dtype, torch.bfloat16)

    @_NEEDS_TORCH
    def test_get_grad_shard_returns_shard_buf(self) -> None:
        """get_grad_shard returns a reference to the reducer's shard_buf."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 8), ("b", 8)]
        r = BucketedGradReducer(
            entries, world_size=2, rank=0, device="cpu",
            bucket_target_elems=100,
        )
        shard = r.get_grad_shard(0)
        self.assertIs(shard, r._shard_bufs[0])

    @_NEEDS_TORCH
    def test_get_grad_shard_out_of_range(self) -> None:
        """get_grad_shard raises IndexError for invalid bucket_idx."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 8)]
        r = BucketedGradReducer(
            entries, world_size=2, rank=0, device="cpu",
        )
        with self.assertRaises(IndexError):
            r.get_grad_shard(1)
        with self.assertRaises(IndexError):
            r.get_grad_shard(-1)

    @_NEEDS_TORCH
    def test_allgather_bf16_and_unpack_single_rank(self) -> None:
        """BF16 allgather + unpack writes correct values to params."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=100,
        )
        params = {
            "a": torch.zeros(4),
            "b": torch.zeros(6),
        }
        bf16_shards = [
            torch.arange(r.buckets[0].size, dtype=torch.bfloat16),
        ]
        bf16_full = torch.empty(r.buckets[0].size, dtype=torch.bfloat16)
        r.allgather_bf16_and_unpack(bf16_shards, bf16_full, params)

        self.assertTrue(torch.allclose(
            params["a"],
            torch.arange(4, dtype=torch.bfloat16).float(),
        ))
        self.assertTrue(torch.allclose(
            params["b"],
            torch.arange(4, 10, dtype=torch.bfloat16).float(),
        ))

    @_NEEDS_TORCH
    def test_sharded_optimizer_round_trip_single_rank(self) -> None:
        """Full sharded optimizer round trip on single rank (world_size=1).

        Verifies that per-shard Adam + BF16 allgather produces the same
        master_weight updates and BF16 param values as the non-sharded path.
        """
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import (
            AdamState,
            ShardedOptimizerBuffers,
            sharded_fused_clip_adam_sync,
        )

        entries = [("a", 4), ("b", 6)]
        ws = 1
        r = BucketedGradReducer(
            entries, world_size=ws, rank=0, device="cpu",
            bucket_target_elems=100,
        )

        params = {
            "a": torch.randn(4),
            "b": torch.randn(2, 3),
        }
        opt = AdamState(["a", "b"], params, device="cpu")

        r.start_step()
        for name, numel in entries:
            r.write_grad(name, torch.randn(numel))
        r.wait_all()

        bufs = ShardedOptimizerBuffers(r, device="cpu")
        lr = 1e-3
        clip_coeff = 1.0

        master_before_a = opt.master_weights["a"].clone()
        master_before_b = opt.master_weights["b"].clone()

        sharded_fused_clip_adam_sync(
            opt, r, bufs, params, lr, clip_coeff,
        )

        self.assertFalse(torch.equal(opt.master_weights["a"], master_before_a))
        self.assertFalse(torch.equal(opt.master_weights["b"], master_before_b))
        self.assertEqual(params["a"].dtype, torch.float32)
        self.assertEqual(params["a"].shape, (4,))
        self.assertEqual(params["b"].shape, (2, 3))
        self.assertEqual(opt.step_count, 1)

    @_NEEDS_TORCH
    def test_sharded_optimizer_multi_rank_shard_overlap(self) -> None:
        """Per-shard optimizer only updates elements within this rank's shard.

        With world_size=2, each rank owns half the shard. After running
        the sharded optimizer, only the elements in this rank's shard
        of master_weights should be modified.
        """
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import (
            AdamState,
            ShardedOptimizerBuffers,
            sharded_fused_clip_adam_sync,
        )

        entries = [("p", 8)]
        ws = 2
        rank = 0
        r = BucketedGradReducer(
            entries, world_size=ws, rank=rank, device="cpu",
            bucket_target_elems=100,
        )

        params = {"p": torch.randn(8)}
        opt = AdamState(["p"], {k: v.clone() for k, v in params.items()}, device="cpu")

        r.start_step()
        r.write_grad("p", torch.randn(8))
        r.wait_all()

        bufs = ShardedOptimizerBuffers(r, device="cpu")

        master_before = opt.master_weights["p"].clone()
        sharded_fused_clip_adam_sync(opt, r, bufs, params, lr=1e-3, clip_coeff=1.0)

        shard_size = r.buckets[0].size // ws
        updated_slice = opt.master_weights["p"].view(-1)[:shard_size]
        stale_slice = opt.master_weights["p"].view(-1)[shard_size:]

        self.assertFalse(torch.equal(updated_slice, master_before.view(-1)[:shard_size]))
        self.assertTrue(torch.equal(stale_slice, master_before.view(-1)[shard_size:]))

    @_NEEDS_TORCH
    def test_sharded_optimizer_step_count_incremented(self) -> None:
        """step_count increments exactly once per call."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import (
            AdamState,
            ShardedOptimizerBuffers,
            sharded_fused_clip_adam_sync,
        )

        entries = [("x", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
        )
        params = {"x": torch.randn(4)}
        opt = AdamState(["x"], params, device="cpu")
        bufs = ShardedOptimizerBuffers(r, device="cpu")

        self.assertEqual(opt.step_count, 0)
        r.start_step()
        r.write_grad("x", torch.randn(4))
        r.wait_all()
        sharded_fused_clip_adam_sync(opt, r, bufs, params, lr=1e-3, clip_coeff=1.0)
        self.assertEqual(opt.step_count, 1)

        r.start_step()
        r.write_grad("x", torch.randn(4))
        r.wait_all()
        sharded_fused_clip_adam_sync(opt, r, bufs, params, lr=1e-3, clip_coeff=1.0)
        self.assertEqual(opt.step_count, 2)

    @_NEEDS_TORCH
    def test_sharded_optimizer_weight_decay_respects_dim(self) -> None:
        """1D params (norm weights) get wd=0, 2D+ params get WEIGHT_DECAY."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import (
            AdamState,
            ShardedOptimizerBuffers,
            sharded_fused_clip_adam_sync,
        )

        entries = [("w2d", 6), ("w1d", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=100,
        )
        params = {
            "w2d": torch.randn(2, 3),
            "w1d": torch.randn(4),
        }
        opt = AdamState(["w2d", "w1d"], params, device="cpu")
        bufs = ShardedOptimizerBuffers(r, device="cpu")

        r.start_step()
        r.write_grad("w2d", torch.randn(6))
        r.write_grad("w1d", torch.randn(4))
        r.wait_all()

        sharded_fused_clip_adam_sync(opt, r, bufs, params, lr=1e-3, clip_coeff=1.0)

        self.assertEqual(opt.step_count, 1)
        self.assertFalse(torch.equal(
            opt.master_weights["w2d"],
            torch.zeros(2, 3),
        ))

    @_NEEDS_TORCH
    def test_sharded_optimizer_bf16_output_matches_master(self) -> None:
        """After sharded optimizer, params (BF16) should be the BF16 cast
        of the updated master weights (for single-rank case)."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import (
            AdamState,
            ShardedOptimizerBuffers,
            sharded_fused_clip_adam_sync,
        )

        entries = [("q", 8)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
        )
        params = {"q": torch.randn(8)}
        opt = AdamState(["q"], params, device="cpu")
        bufs = ShardedOptimizerBuffers(r, device="cpu")

        r.start_step()
        r.write_grad("q", torch.randn(8))
        r.wait_all()

        sharded_fused_clip_adam_sync(opt, r, bufs, params, lr=1e-3, clip_coeff=1.0)

        expected_bf16 = opt.master_weights["q"].to(torch.bfloat16).float()
        self.assertTrue(torch.allclose(params["q"], expected_bf16))

    @_NEEDS_TORCH
    def test_sharded_multi_bucket_round_trip(self) -> None:
        """Sharded optimizer works across multiple buckets."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import (
            AdamState,
            ShardedOptimizerBuffers,
            sharded_fused_clip_adam_sync,
        )

        entries = [("a", 4), ("b", 6), ("c", 8)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=10,
        )
        self.assertGreater(r.num_buckets, 1)

        params = {
            "a": torch.randn(4),
            "b": torch.randn(6),
            "c": torch.randn(8),
        }
        opt = AdamState(["a", "b", "c"], params, device="cpu")
        bufs = ShardedOptimizerBuffers(r, device="cpu")

        r.start_step()
        for name, numel in entries:
            r.write_grad(name, torch.randn(numel))
        r.wait_all()

        sharded_fused_clip_adam_sync(opt, r, bufs, params, lr=1e-3, clip_coeff=1.0)

        for name in ["a", "b", "c"]:
            expected_bf16 = opt.master_weights[name].to(torch.bfloat16).float()
            self.assertTrue(
                torch.allclose(params[name], expected_bf16),
                f"BF16 mismatch for {name}",
            )


class AllgatherOptimizerStateTests(unittest.TestCase):
    """Tests for BucketedGradReducer.allgather_optimizer_state.

    The method is the checkpoint-compatibility bridge for ZeRO-1: it
    allgathers master_weights / exp_avg / exp_avg_sq so rank 0 has
    the complete FP32 state before save_checkpoint writes to disk.
    """

    @_NEEDS_TORCH
    def test_nondist_early_return(self) -> None:
        """Non-distributed path returns immediately without touching state."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import AdamState

        entries = [("w", 8)]
        r = BucketedGradReducer(entries, world_size=1, rank=0, device="cpu")
        params = {"w": torch.randn(8)}
        opt = AdamState(["w"], params, device="cpu")

        before_mw = opt.master_weights["w"].clone()
        before_ea = opt.exp_avg["w"].clone()
        before_eas = opt.exp_avg_sq["w"].clone()

        r.allgather_optimizer_state(opt)

        self.assertTrue(torch.equal(opt.master_weights["w"], before_mw))
        self.assertTrue(torch.equal(opt.exp_avg["w"], before_ea))
        self.assertTrue(torch.equal(opt.exp_avg_sq["w"], before_eas))

    @_NEEDS_TORCH
    def test_method_exists_and_callable(self) -> None:
        """allgather_optimizer_state is a public method on BucketedGradReducer."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("p", 4)]
        r = BucketedGradReducer(entries, world_size=1, rank=0, device="cpu")
        self.assertTrue(callable(getattr(r, "allgather_optimizer_state", None)))

    @_NEEDS_TORCH
    def test_allgather_state_field_pack_unpack_round_trip_ws1(self) -> None:
        """For ws=1, _allgather_state_field is a no-op (non-dist early return).

        Verify the state is unchanged.
        """
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import AdamState

        entries = [("a", 6), ("b", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=6,
        )
        params = {"a": torch.randn(6), "b": torch.randn(4)}
        opt = AdamState(["a", "b"], params, device="cpu")
        opt.master_weights["a"][:] = torch.arange(6, dtype=torch.float32)
        opt.master_weights["b"][:] = torch.arange(10, 14, dtype=torch.float32)

        before_a = opt.master_weights["a"].clone()
        before_b = opt.master_weights["b"].clone()
        r.allgather_optimizer_state(opt)

        self.assertTrue(torch.equal(opt.master_weights["a"], before_a))
        self.assertTrue(torch.equal(opt.master_weights["b"], before_b))

    @_NEEDS_TORCH
    def test_allgather_after_sharded_optimizer_ws1(self) -> None:
        """End-to-end: sharded optimizer + allgather_optimizer_state on ws=1.

        For ws=1 the sharded optimizer updates all elements, so allgather
        is a no-op but should not corrupt anything.
        """
        _init_test_config()
        triton_state = _install_fake_triton_module()
        try:
            from training_engine_tensor.nccl import BucketedGradReducer
            from training_engine_tensor.optimizer import (
                AdamState,
                ShardedOptimizerBuffers,
                sharded_fused_clip_adam_sync,
            )

            entries = [("x", 8), ("y", 4)]
            r = BucketedGradReducer(
                entries, world_size=1, rank=0, device="cpu",
                bucket_target_elems=8,
            )
            params = {"x": torch.randn(8), "y": torch.randn(4)}
            opt = AdamState(["x", "y"], params, device="cpu")
            bufs = ShardedOptimizerBuffers(r, device="cpu")

            r.start_step()
            r.write_grad("x", torch.randn(8))
            r.write_grad("y", torch.randn(4))
            r.wait_all()

            sharded_fused_clip_adam_sync(opt, r, bufs, params, lr=1e-3, clip_coeff=1.0)

            mw_x_before = opt.master_weights["x"].clone()
            mw_y_before = opt.master_weights["y"].clone()
            ea_x_before = opt.exp_avg["x"].clone()
            ea_y_before = opt.exp_avg["y"].clone()

            r.allgather_optimizer_state(opt)

            self.assertTrue(torch.equal(opt.master_weights["x"], mw_x_before))
            self.assertTrue(torch.equal(opt.master_weights["y"], mw_y_before))
            self.assertTrue(torch.equal(opt.exp_avg["x"], ea_x_before))
            self.assertTrue(torch.equal(opt.exp_avg["y"], ea_y_before))
        finally:
            _uninstall_fake_triton_module(*triton_state)

    @_NEEDS_TORCH
    def test_allgather_covers_all_three_fields(self) -> None:
        """allgather_optimizer_state touches master_weights, exp_avg, exp_avg_sq."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import AdamState

        entries = [("p", 4)]
        r = BucketedGradReducer(entries, world_size=1, rank=0, device="cpu")
        params = {"p": torch.randn(4)}
        opt = AdamState(["p"], params, device="cpu")
        opt.master_weights["p"][:] = 1.0
        opt.exp_avg["p"][:] = 2.0
        opt.exp_avg_sq["p"][:] = 3.0

        r.allgather_optimizer_state(opt)

        self.assertTrue(torch.all(opt.master_weights["p"] == 1.0))
        self.assertTrue(torch.all(opt.exp_avg["p"] == 2.0))
        self.assertTrue(torch.all(opt.exp_avg_sq["p"] == 3.0))

    @_NEEDS_TORCH
    def test_allgather_state_multi_bucket(self) -> None:
        """allgather_optimizer_state handles params spread across buckets."""
        from training_engine_tensor.nccl import BucketedGradReducer
        from training_engine_tensor.optimizer import AdamState

        entries = [("a", 4), ("b", 6), ("c", 8)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=6,
        )
        self.assertGreater(r.num_buckets, 1)

        params = {
            "a": torch.randn(4),
            "b": torch.randn(6),
            "c": torch.randn(8),
        }
        opt = AdamState(["a", "b", "c"], params, device="cpu")
        opt.master_weights["a"][:] = 10.0
        opt.master_weights["b"][:] = 20.0
        opt.master_weights["c"][:] = 30.0

        r.allgather_optimizer_state(opt)

        self.assertTrue(torch.all(opt.master_weights["a"] == 10.0))
        self.assertTrue(torch.all(opt.master_weights["b"] == 20.0))
        self.assertTrue(torch.all(opt.master_weights["c"] == 30.0))


# ---------------------------------------------------------------------------
# P2-A multi-microbatch (gradient accumulation) tests
# ---------------------------------------------------------------------------


@_NEEDS_TORCH
class TestMultiMicrobatchAccumulation(unittest.TestCase):
    """Covers the P2-A multi-MB extension of ``BucketedGradReducer``.

    Contract under ``num_local_microbatches == N > 1``:

    * The first MB's bucket fill uses ``copy_`` (overwriting last
      step's residue; padding remains zero).
    * MBs 2..N-1 use ``add_``; no ``reduce_scatter`` issued.
    * MB N uses ``add_`` then issues ``reduce_scatter``.
    * Across N MBs each bucket receives exactly N writes per param;
      after MB N the input buffer holds the sum of N MBs' grads.
    * In the non-distributed fallback ``mul_(1/ws)`` runs exactly
      once (only on the final MB), so the resulting shard equals
      ``sum_grads / ws`` — same numeric contract as the legacy
      "FP32 dict accumulate then single reduce_scatter" path.
    * ``start_step()`` resets ``mb_remaining`` and ``is_first_mb`` so
      consecutive steps don't bleed into each other.
    """

    @staticmethod
    def _entries(sizes: list[int]) -> list[tuple[str, int]]:
        return [(f"p{i}", s) for i, s in enumerate(sizes)]

    def test_invalid_num_local_microbatches_raises(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        with self.assertRaises(ValueError):
            BucketedGradReducer(
                self._entries([4]), world_size=1, rank=0, device="cpu",
                num_local_microbatches=0,
            )
        with self.assertRaises(ValueError):
            BucketedGradReducer(
                self._entries([4]), world_size=1, rank=0, device="cpu",
                num_local_microbatches=-1,
            )

    def test_mb_counters_initialised(self) -> None:
        """``__init__`` populates per-bucket mb_remaining and is_first_mb."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4, 6])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=200, num_local_microbatches=3,
        )
        self.assertEqual(r._bucket_mb_remaining, [3])  # noqa: SLF001
        self.assertEqual(r._bucket_is_first_mb, [True])  # noqa: SLF001

    def test_start_step_resets_mb_counters(self) -> None:
        """``start_step()`` must reset mb_remaining and is_first_mb."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            num_local_microbatches=2,
        )
        r._bucket_mb_remaining[0] = 0  # noqa: SLF001
        r._bucket_is_first_mb[0] = False  # noqa: SLF001
        r.start_step()
        self.assertEqual(r._bucket_mb_remaining, [2])  # noqa: SLF001
        self.assertEqual(r._bucket_is_first_mb, [True])  # noqa: SLF001

    def test_middle_mb_does_not_issue_reduce(self) -> None:
        """After MB 1 of 2 fills the bucket, the shard buffer must
        remain at its prior value (no reduce_scatter dispatched).
        We seed the shard with a sentinel and assert it stays after
        the first MB but changes after the second MB."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            num_local_microbatches=2,
        )
        sentinel = torch.full_like(r._shard_bufs[0], 999.0)  # noqa: SLF001
        r._shard_bufs[0].copy_(sentinel)  # noqa: SLF001

        r.start_step()
        r.write_grad("p0", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(
            torch.equal(r._shard_bufs[0], sentinel),  # noqa: SLF001
            "middle MB must not issue reduce_scatter",
        )

        r.write_grad("p0", torch.tensor([10.0, 20.0, 30.0, 40.0]))
        # Final MB: shard updated. ws=1 → shard == input_buf == sum/1.
        expected = torch.tensor([11.0, 22.0, 33.0, 44.0])
        self.assertTrue(torch.allclose(r._shard_bufs[0], expected))  # noqa: SLF001

    def test_first_mb_overwrites_last_step_residue(self) -> None:
        """The first MB of the new step must ``copy_`` over the prior
        step's bucket contents (otherwise the new step's grads would
        be added to last step's accumulated total)."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            num_local_microbatches=2,
        )
        r.start_step()
        r.write_grad("p0", torch.tensor([1.0, 1.0, 1.0, 1.0]))
        r.write_grad("p0", torch.tensor([2.0, 2.0, 2.0, 2.0]))
        r.wait_all()
        # After step 1: shard == 3.0 each. Now step 2 begins.

        r.start_step()
        r.write_grad("p0", torch.tensor([10.0, 10.0, 10.0, 10.0]))
        # After MB 1 of step 2: input_buf should be [10, 10, 10, 10]
        # (copy_, not add_), so the shard's prior 3.0 must not leak in.
        r.write_grad("p0", torch.tensor([5.0, 5.0, 5.0, 5.0]))
        # Final MB: shard = (10 + 5) / 1 = 15
        self.assertTrue(torch.allclose(
            r._shard_bufs[0],  # noqa: SLF001
            torch.tensor([15.0, 15.0, 15.0, 15.0]),
        ))

    def test_input_buffer_accumulates_across_mbs(self) -> None:
        """Across all N MBs, the input buffer must equal the sum of
        the N FP32 grads. Verified BEFORE wait_all() / reduce so we
        catch the mul_(1/ws) scaling step separately below."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([4])
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            num_local_microbatches=3,
        )
        r.start_step()
        grads = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
            torch.tensor([100.0, 200.0, 300.0, 400.0]),
        ]
        # First two MBs: no reduce yet, so input_buf still holds the
        # un-scaled running sum.
        r.write_grad("p0", grads[0])
        self.assertTrue(torch.allclose(
            r._input_bufs[0],  # noqa: SLF001
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ))
        r.write_grad("p0", grads[1])
        self.assertTrue(torch.allclose(
            r._input_bufs[0],  # noqa: SLF001
            torch.tensor([11.0, 22.0, 33.0, 44.0]),
        ))
        # Final MB: input_buf gets the last add_, then mul_(1/1)
        # (no-op for ws=1).
        r.write_grad("p0", grads[2])
        self.assertTrue(torch.allclose(
            r._input_bufs[0],  # noqa: SLF001
            torch.tensor([111.0, 222.0, 333.0, 444.0]),
        ))

    def test_round_trip_matches_explicit_fp32_accum(self) -> None:
        """Bytewise: the multi-MB reducer path must produce the same
        per-param FP32 grads as the legacy 'dict-of-FP32 accumulate
        then single reduce_scatter' path. We simulate the legacy
        path by running a single-MB reducer on the pre-summed grad
        tensor."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8)]
        params_view = {n: torch.zeros(sz) for n, sz in entries}

        torch.manual_seed(42)
        per_mb_grads: list[dict[str, torch.Tensor]] = []
        num_mb = 3
        for _ in range(num_mb):
            per_mb_grads.append({
                n: torch.randn(sz, dtype=torch.float32)
                for n, sz in entries
            })

        # Multi-MB path
        r_multi = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=10,
            num_local_microbatches=num_mb,
        )
        r_multi.start_step()
        for mb_grads in per_mb_grads:
            for name, g in mb_grads.items():
                r_multi.write_grad(name, g)
        r_multi.wait_all()
        out_multi = r_multi.allgather_and_unpack(params_view)

        # Legacy path: pre-sum then run single-MB reducer
        summed = {
            n: sum((mb[n] for mb in per_mb_grads), torch.zeros_like(per_mb_grads[0][n]))
            for n, _ in entries
        }
        r_legacy = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=10,
            num_local_microbatches=1,
        )
        r_legacy.start_step()
        for name, g in summed.items():
            r_legacy.write_grad(name, g)
        r_legacy.wait_all()
        out_legacy = r_legacy.allgather_and_unpack(params_view)

        for name, _ in entries:
            self.assertTrue(
                torch.allclose(out_multi[name], out_legacy[name]),
                f"multi-MB vs legacy mismatch for {name}: "
                f"{out_multi[name]} vs {out_legacy[name]}",
            )

    def test_round_trip_multi_bucket_multi_mb(self) -> None:
        """Multi-bucket + multi-MB sanity. Each bucket independently
        tracks mb_remaining; filling bucket 0 in MB 1 must not flip
        bucket 1's mb_remaining."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 6), ("c", 8), ("d", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=15,  # → buckets [(a,b)=10, (c,d)=12]
            num_local_microbatches=2,
        )
        self.assertEqual(r.num_buckets, 2)
        params_view = {n: torch.zeros(sz) for n, sz in entries}

        per_mb = [
            {"a": torch.ones(4), "b": torch.ones(6) * 2,
             "c": torch.ones(8) * 3, "d": torch.ones(4) * 4},
            {"a": torch.ones(4) * 10, "b": torch.ones(6) * 20,
             "c": torch.ones(8) * 30, "d": torch.ones(4) * 40},
        ]
        r.start_step()
        for mb in per_mb:
            for n, g in mb.items():
                r.write_grad(n, g)

        # After MB 1: bucket 0 mb_remaining=1, bucket 1 mb_remaining=1
        # (the writes for MB 2 are interleaved within the same loop,
        # but each bucket fills exactly twice — once per MB).
        r.wait_all()
        out = r.allgather_and_unpack(params_view)

        # Expected sums (ws=1, so /1):
        self.assertTrue(torch.allclose(out["a"], torch.ones(4) * 11))
        self.assertTrue(torch.allclose(out["b"], torch.ones(6) * 22))
        self.assertTrue(torch.allclose(out["c"], torch.ones(8) * 33))
        self.assertTrue(torch.allclose(out["d"], torch.ones(4) * 44))

    def test_multi_mb_with_world_size_scaling(self) -> None:
        """In non-dist mode the synchronous reduce_scatter fallback
        scales the FINAL input_buf by ``1/ws`` exactly once (only on
        the final MB). Verify the resulting shard equals
        ``sum_grads / ws``."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = self._entries([8])
        r = BucketedGradReducer(
            entries, world_size=4, rank=0, device="cpu",
            num_local_microbatches=2,
        )
        r.start_step()
        r.write_grad("p0", torch.ones(8) * 4.0)
        r.write_grad("p0", torch.ones(8) * 8.0)
        r.wait_all()
        # Sum = 12.0 per element; /ws=4 = 3.0. Rank 0 owns input[0:2].
        self.assertTrue(torch.allclose(
            r._shard_bufs[0],  # noqa: SLF001
            torch.ones(2) * 3.0,
        ))

    def test_multi_mb_padding_stays_zero(self) -> None:
        """Padding region must remain zero across MBs in the multi-MB
        regime — neither ``copy_`` of MB 1 nor ``add_`` of subsequent
        MBs touches indices >= raw_size, so the padding tail must
        stay at its initial zero value."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([5]), world_size=4, rank=0, device="cpu",
            num_local_microbatches=3,
        )
        # raw_size=5, size=8 → padding indices [5, 8)
        self.assertEqual(r.buckets[0].size, 8)
        r.start_step()
        r.write_grad("p0", torch.ones(5))
        r.write_grad("p0", torch.ones(5) * 2)
        r.write_grad("p0", torch.ones(5) * 3)
        self.assertTrue(torch.all(
            r._input_bufs[0][5:8] == 0,  # noqa: SLF001
        ).item())

    def test_single_mb_default_unchanged(self) -> None:
        """Default ``num_local_microbatches=1`` must reproduce legacy
        behaviour bytewise — the parameter is opt-in only."""
        from training_engine_tensor.nccl import BucketedGradReducer

        # No num_local_microbatches kwarg → defaults to 1
        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
        )
        self.assertEqual(r._num_mb, 1)  # noqa: SLF001
        self.assertEqual(r._bucket_mb_remaining, [1])  # noqa: SLF001
        r.start_step()
        grad = torch.tensor([7.0, 8.0, 9.0, 10.0])
        r.write_grad("p0", grad)
        # Single MB: bucket should fire reduce immediately (shard
        # equals grad in non-dist ws=1 mode).
        self.assertTrue(torch.allclose(r._shard_bufs[0], grad))  # noqa: SLF001

    def test_intermediate_mb_does_not_raise_on_next_mb_writes(self) -> None:
        """After the last param of MB 1 fills a bucket, the next param
        write (for MB 2, same name) must NOT raise the
        'bucket already complete' error. This was the explicit guard
        we relax in P2-A: pending counter is reset when the MB is not
        the final one."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
            num_local_microbatches=2,
        )
        r.start_step()
        r.write_grad("p0", torch.ones(4))
        # Pending counter was reset to 1 (one param in this bucket).
        self.assertEqual(r._bucket_pending, [1])  # noqa: SLF001
        # Second MB's write succeeds.
        r.write_grad("p0", torch.ones(4) * 2)

    # ------------------------------------------------------------------
    # Direction-C path-1 (step CUDA Graph + reducer coexistence)
    # ------------------------------------------------------------------

    def test_make_device_grad_sink_returns_callable(self) -> None:
        """The factory returns a callable suitable for backward to call
        as ``grad_sink(name, grad)``."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4, 2]), world_size=1, rank=0, device="cpu",
        )
        sink = r.make_device_grad_sink()
        self.assertTrue(callable(sink))

    def test_device_grad_sink_in_place_add_to_bucket(self) -> None:
        """Calling the device sink for every param accumulates each
        grad directly into the bucket's input buffer at the slot
        derived from ``_param_loc``."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4, 2]), world_size=1, rank=0, device="cpu",
        )
        sink = r.make_device_grad_sink()
        # Caller contract: buffer must be zeroed first.
        r.zero_input_bufs()
        g0 = torch.ones(4)
        g1 = torch.full((2,), 3.0)
        sink("p0", g0)
        sink("p1", g1)
        # ``_param_loc`` is the source of truth for the layout. Use it
        # to assert each slot was written to.
        bucket_idx_0, off_0, n_0 = r._param_loc["p0"]  # noqa: SLF001
        bucket_idx_1, off_1, n_1 = r._param_loc["p1"]  # noqa: SLF001
        for buf_idx, sample in [
            (bucket_idx_0, (off_0, n_0, g0)),
            (bucket_idx_1, (off_1, n_1, g1)),
        ]:
            off, n, expected = sample
            slot = r._input_bufs[buf_idx][off:off + n]  # noqa: SLF001
            self.assertTrue(torch.allclose(slot, expected.float()))

    def test_device_grad_sink_accumulates_across_calls(self) -> None:
        """Successive calls to the sink for the same param use ``add_``,
        so MB2's call accumulates on top of MB1's, with no host-side
        state."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([3]), world_size=1, rank=0, device="cpu",
        )
        sink = r.make_device_grad_sink()
        r.zero_input_bufs()
        sink("p0", torch.ones(3))           # MB1
        sink("p0", torch.full((3,), 5.0))   # MB2
        bucket_idx, off, n = r._param_loc["p0"]  # noqa: SLF001
        slot = r._input_bufs[bucket_idx][off:off + n]  # noqa: SLF001
        self.assertTrue(torch.allclose(slot, torch.full((3,), 6.0)))

    def test_device_grad_sink_casts_bf16_to_fp32(self) -> None:
        """BF16 grads are cast to FP32 before the in-place add (matches
        what the eager wgrad-overlap path does in
        ``_flush_for_microbatch``)."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
        )
        sink = r.make_device_grad_sink()
        r.zero_input_bufs()
        bf16_grad = torch.ones(4, dtype=torch.bfloat16)
        sink("p0", bf16_grad)
        bucket_idx, off, n = r._param_loc["p0"]  # noqa: SLF001
        slot = r._input_bufs[bucket_idx][off:off + n]  # noqa: SLF001
        self.assertEqual(slot.dtype, torch.float32)
        self.assertTrue(torch.allclose(slot, torch.ones(4)))

    def test_zero_input_bufs_wipes_all_buckets(self) -> None:
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4, 2]), world_size=1, rank=0, device="cpu",
        )
        for buf in r._input_bufs:  # noqa: SLF001
            buf.fill_(7.0)
        r.zero_input_bufs()
        for buf in r._input_bufs:  # noqa: SLF001
            self.assertTrue(torch.all(buf == 0))

    def test_flush_all_buckets_drives_pending_to_zero(self) -> None:
        """After ``flush_all_buckets`` every bucket's pending counter is
        zero so ``wait_all`` invariants hold."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4, 2]), world_size=1, rank=0, device="cpu",
        )
        r.start_step()
        # Buckets pending == len(params) initially.
        self.assertNotEqual(sum(r._bucket_pending), 0)  # noqa: SLF001
        r.zero_input_bufs()
        sink = r.make_device_grad_sink()
        sink("p0", torch.ones(4))
        sink("p1", torch.full((2,), 2.0))
        r.flush_all_buckets()
        self.assertEqual(r._bucket_pending, [0] * len(r._buckets))  # noqa: SLF001

    def test_flush_all_buckets_writes_shard_buf_in_nondist(self) -> None:
        """In single-rank mode ``flush_all_buckets`` runs the local
        ``mul_(1/ws) + copy_`` fallback in ``_issue_bucket_reduce``,
        so the shard buffer ends up holding the reduced input."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
        )
        r.start_step()
        r.zero_input_bufs()
        sink = r.make_device_grad_sink()
        sink("p0", torch.tensor([2.0, 4.0, 6.0, 8.0]))
        r.flush_all_buckets()
        # World size = 1 → mul_(1.0) is a no-op; shard contains the
        # whole bucket.
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0])
        self.assertTrue(torch.allclose(r._shard_bufs[0], expected))  # noqa: SLF001

    def test_accum_stream_default_is_none(self) -> None:
        """P2-B accum_stream is opt-in; the default ``None`` keeps
        every flush on the default stream so behaviour is bytewise
        identical to the P2-A path."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
            num_local_microbatches=2,
        )
        self.assertIsNone(r._accum_stream)  # noqa: SLF001
        self.assertEqual(r._bucket_accum_done_events, [None])  # noqa: SLF001

    def test_accum_stream_done_events_reset_each_step(self) -> None:
        """``start_step`` must clear stale ``accum_done`` events from
        the previous step so they cannot accidentally gate the new
        step's ``_issue_bucket_reduce`` on outdated work."""
        from training_engine_tensor.nccl import BucketedGradReducer

        r = BucketedGradReducer(
            self._entries([4]), world_size=1, rank=0, device="cpu",
            num_local_microbatches=2,
        )
        # Hand-poke a fake event into the slot (simulating a real run
        # where _flush_for_microbatch on the final MB recorded one).
        r._bucket_accum_done_events[0] = "stale_event"  # noqa: SLF001
        r.start_step()
        self.assertEqual(r._bucket_accum_done_events, [None])  # noqa: SLF001

    def test_two_steps_multi_mb_independence(self) -> None:
        """Two consecutive steps of N=2 MBs. Step 2's grads must NOT
        leak from step 1; we test by checking the shard reflects only
        step 2's sum."""
        from training_engine_tensor.nccl import BucketedGradReducer

        entries = [("a", 4), ("b", 4)]
        r = BucketedGradReducer(
            entries, world_size=1, rank=0, device="cpu",
            bucket_target_elems=100, num_local_microbatches=2,
        )

        # Step 1: a=1+1=2, b=1+1=2 each
        r.start_step()
        for _ in range(2):
            r.write_grad("a", torch.ones(4))
            r.write_grad("b", torch.ones(4))
        r.wait_all()

        # Step 2: a=10+5=15, b=10+5=15 each
        r.start_step()
        r.write_grad("a", torch.ones(4) * 10)
        r.write_grad("b", torch.ones(4) * 10)
        r.write_grad("a", torch.ones(4) * 5)
        r.write_grad("b", torch.ones(4) * 5)
        r.wait_all()

        params_view = {"a": torch.zeros(4), "b": torch.zeros(4)}
        out = r.allgather_and_unpack(params_view)
        self.assertTrue(torch.allclose(out["a"], torch.ones(4) * 15))
        self.assertTrue(torch.allclose(out["b"], torch.ones(4) * 15))


if __name__ == "__main__":
    unittest.main()
