"""CLI entry point for the MiniCPM4-8B training engine.

Subcommands
-----------

* ``pretrain`` — random-init from-scratch pretraining with a HuggingFace
  dataloader.  Used by ``scripts/entry_hf_pretrain.sh``.

The CLI builds and freezes a global :class:`EngineConfig` before any
framework module is imported (so module-level ``get_config()`` reads see
the resolved values), then:

1. Resolves the distributed environment from the standard torchrun
   variables (``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``).
2. Builds an HF ``DataLoader`` via :func:`hf_dataloader.build_hf_dataloader`.
3. Initialises the model parameter dict from scratch via
   :func:`parameters.self_init_params`.
4. Hands both into :func:`entry.run_training`.

Run ``python -m training_engine_tensor pretrain --help`` for the full
flag surface.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import sys
from pathlib import Path
from typing import Iterator

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # Python < 3.11 fallback
    import tomli as _tomllib  # type: ignore[no-redef]

_SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "model_spec.toml"
with open(_SPEC_PATH, "rb") as _f:
    _SPEC = _tomllib.load(_f)
_TRAINING_SPEC = _SPEC["training"]
_MODEL_SPEC = _SPEC["model"]
_PARALLELISM_SPEC = _SPEC.get("parallelism", {})


def _spec_tp_size() -> int:
    """Return the L1 tensor-parallel size declared in ``model_spec.toml``."""
    return int(_PARALLELISM_SPEC.get("tensor_model_parallel_size", 1))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_engine_config_args(parser: argparse.ArgumentParser) -> None:
    """Auto-generate CLI flags from :class:`EngineConfig` dataclass fields.

    Bool fields become ``--flag`` / ``--no-flag`` pairs (default
    ``None``).  Numeric fields become ``--field-name`` with the
    appropriate Python type (default ``None`` so that
    :func:`engine_config.from_env` defaults are preserved when unset).
    """
    from training_engine_tensor.engine_config import EngineConfig

    group = parser.add_argument_group("EngineConfig overrides")
    for f in dataclasses.fields(EngineConfig):
        cli_name = "--" + f.name.replace("_", "-")
        if f.type == "bool" or f.type is bool:
            group.add_argument(
                cli_name,
                action=argparse.BooleanOptionalAction,
                default=None,
            )
        elif f.type == "int" or f.type is int:
            group.add_argument(cli_name, type=int, default=None)
        elif f.type == "float" or f.type is float:
            group.add_argument(cli_name, type=float, default=None)
        else:
            group.add_argument(cli_name, default=None)


def _apply_engine_config_overrides(
    base: "EngineConfig", args: argparse.Namespace,  # noqa: F821
) -> "EngineConfig":  # noqa: F821
    """Return a new :class:`EngineConfig` with non-``None`` CLI values applied."""
    from training_engine_tensor.engine_config import EngineConfig

    overrides: dict = {}
    for f in dataclasses.fields(EngineConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            overrides[f.name] = v
    if not overrides:
        return base
    return dataclasses.replace(base, **overrides)


def _add_hf_data_args(p: argparse.ArgumentParser) -> None:
    """Register CLI arguments for HuggingFace dataset loading."""
    g = p.add_argument_group("HuggingFace data")
    g.add_argument(
        "--hf-dataset",
        required=True,
        help="HuggingFace dataset path (local dir or Hub name).",
    )
    g.add_argument(
        "--tokenizer-path",
        required=True,
        help="HuggingFace tokenizer path (local dir or Hub name).",
    )
    g.add_argument(
        "--hf-dataset-config",
        default=None,
        help="Dataset config / subset name (e.g. ``main`` for GSM-8K).",
    )
    g.add_argument(
        "--hf-dataset-split",
        default="train",
        help="Dataset split (default: ``train``).",
    )
    g.add_argument(
        "--hf-text-field",
        nargs="+",
        default=None,
        help="Text column name(s) to concatenate from each row.",
    )
    g.add_argument(
        "--hf-text-template",
        default=None,
        help=(
            "Optional Python format string applied to the row dict, e.g. "
            "``Question: {question}\\nAnswer: {answer}``."
        ),
    )
    g.add_argument(
        "--hf-data-format",
        default=None,
        help=(
            "Optional explicit loader name (``parquet`` / ``json`` / ...)."
            "  Requires ``--hf-data-files`` and ignores ``--hf-dataset``."
        ),
    )
    g.add_argument(
        "--hf-data-files",
        nargs="+",
        default=None,
        help=(
            "Glob(s) passed to ``datasets.load_dataset(data_files=...)`` "
            "when ``--hf-data-format`` is set."
        ),
    )


def _add_pretrain_args(p: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the ``pretrain`` subcommand."""
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument(
        "--global-batch-size",
        type=int,
        default=int(_TRAINING_SPEC["global_batch_size"]),
        help=(
            "Effective batch size in samples / step.  Combined with "
            "``--micro-batch-size`` and the world-size to compute the "
            "default ``--grad-accum-steps``."
        ),
    )
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=int(_TRAINING_SPEC["micro_batch_size"]),
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help=(
            "Gradient accumulation steps per optimizer step.  Defaults "
            "to ``model_spec.toml.training.grad_accum_steps``."
        ),
    )
    p.add_argument(
        "--seq-length",
        type=int,
        default=int(_MODEL_SPEC["seq_length"]),
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device",
        default=None,
        help="Override device for the engine (default: ``cuda:LOCAL_RANK``).",
    )
    p.add_argument(
        "--checkpoint-root",
        default=None,
        help=(
            "Optional directory containing a Megatron-format checkpoint to "
            "warm-start from.  When unset, parameters are initialised from "
            "scratch via :func:`parameters.self_init_params`."
        ),
    )
    p.add_argument(
        "--export-report",
        default=None,
        help=(
            "Optional path to write a per-step JSON training report "
            "(loss, grad_norm, step_time)."
        ),
    )
    p.add_argument(
        "--save-checkpoint-dir",
        default=None,
        help=(
            "Optional directory in which to persist a resume-able training "
            "state (one ``rank_<R>.pt`` shard per rank) after the final "
            "step.  The shards are loadable via "
            ":func:`parameters.load_resume_checkpoint`."
        ),
    )

    _add_hf_data_args(p)
    _add_engine_config_args(p)


# ---------------------------------------------------------------------------
# Distributed environment plumbing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _DistEnv:
    rank: int
    local_rank: int
    world_size: int
    device: str
    master_addr: str
    master_port: str


def _resolve_distributed_env(args: argparse.Namespace) -> _DistEnv:
    """Pull rank / device / world size from torchrun env vars."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = args.device or f"cuda:{local_rank}"
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    return _DistEnv(rank, local_rank, world_size, device, master_addr, master_port)


# ---------------------------------------------------------------------------
# pretrain — main flow
# ---------------------------------------------------------------------------


def _build_dataloader(
    args: argparse.Namespace, env: _DistEnv, grad_accum_steps: int,
):
    """Construct an HF ``DataLoader`` sharded over the DP world.

    Computes ``max_packed_rows`` from the training shape so the
    preload tokeniser only walks enough documents to feed ``num_steps``
    optimizer iterations (the iterator wraps with re-shuffle, so a 2×
    safety factor is comfortably enough for short runs while leaving
    long runs uncapped because the rank's full slice is smaller than
    the budget).
    """
    from training_engine_tensor.hf_dataloader import build_hf_dataloader

    tp_size = _spec_tp_size()
    if env.world_size % tp_size != 0:
        raise ValueError(
            f"WORLD_SIZE={env.world_size} is not a multiple of "
            f"tensor_model_parallel_size={tp_size}."
        )
    dp_size = env.world_size // tp_size
    dp_rank = env.rank // tp_size

    # Per-rank rows actually consumed by training:
    #   num_steps * grad_accum_steps * micro_batch_size
    # 2× safety covers shuffle wrap and the partial-chunk drop.
    rows_needed_per_rank = (
        args.num_steps * grad_accum_steps * args.micro_batch_size
    )
    max_packed_rows = max(2 * rows_needed_per_rank, 64)

    return build_hf_dataloader(
        dataset_path=args.hf_dataset,
        tokenizer_path=args.tokenizer_path,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        seed=args.seed,
        dp_rank=dp_rank,
        dp_size=dp_size,
        dataset_config=args.hf_dataset_config,
        dataset_split=args.hf_dataset_split,
        data_format=args.hf_data_format,
        data_files=args.hf_data_files,
        text_fields=args.hf_text_field,
        text_template=args.hf_text_template,
        num_workers=args.num_workers,
        max_packed_rows=max_packed_rows,
    )


def _stream_batches(
    dataloader, device: str,
) -> Iterator[dict]:
    """Yield batches from an HF ``DataLoader`` forever, moving tensors to ``device``."""
    from training_engine_tensor.hf_dataloader import next_hf_batch

    while True:
        dl_iter = iter(dataloader)
        try:
            while True:
                yield next_hf_batch(dl_iter, device)
        except StopIteration:
            continue


def _resolve_grad_accum_steps(
    args: argparse.Namespace, env: _DistEnv,
) -> int:
    """Resolve ``--grad-accum-steps`` from CLI / spec / GBS-MBS arithmetic."""
    if args.grad_accum_steps is not None:
        return max(1, int(args.grad_accum_steps))

    spec_default = int(_TRAINING_SPEC.get("grad_accum_steps", 1))
    if spec_default > 0:
        return spec_default

    tp_size = _spec_tp_size()
    dp_size = max(1, env.world_size // tp_size)
    n = args.global_batch_size // (args.micro_batch_size * dp_size)
    return max(1, n)


def _init_torch_distributed(env: _DistEnv, seed: int) -> None:
    """Bring up the default ``torch.distributed`` process group on NCCL.

    Sets the per-process CUDA device, seeds CPU and CUDA RNGs, and calls
    ``dist.init_process_group`` *without* explicit rank / world_size (those
    are picked up from the standard torchrun env vars).
    """
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(env.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
    )


def _run_pretrain(args: argparse.Namespace) -> int:
    """Random-init pretraining with an HF dataloader."""
    from training_engine_tensor.entry import run_training
    from training_engine_tensor.parameters import self_init_params

    env = _resolve_distributed_env(args)
    _init_torch_distributed(env, args.seed)
    tp_size = _spec_tp_size()
    tp_rank = env.rank % tp_size

    grad_accum_steps = _resolve_grad_accum_steps(args, env)

    if env.rank == 0:
        print(
            f"  [engine] pretrain mode: world={env.world_size} "
            f"TP={tp_size} grad_accum_steps={grad_accum_steps} "
            f"MBS={args.micro_batch_size} GBS={args.global_batch_size} "
            f"seq_len={args.seq_length} num_steps={args.num_steps}",
            flush=True,
        )

    dataloader = _build_dataloader(args, env, grad_accum_steps)
    batches = _stream_batches(dataloader, env.device)

    if args.checkpoint_root is not None:
        params_override = None
        checkpoint_root = args.checkpoint_root
    else:
        params_override = self_init_params(
            device=env.device,
            seed=args.seed,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        checkpoint_root = ""  # unused when params_override is set

    state_out: dict[str, object] = {}
    results = run_training(
        checkpoint_root=checkpoint_root,
        batches=batches,
        num_steps=args.num_steps,
        device=env.device,
        world_size=env.world_size,
        rank=env.rank,
        tp_size=tp_size,
        params_override=params_override,
        state_out=state_out,
        grad_accum_steps=grad_accum_steps,
    )

    if env.rank == 0:
        _emit_training_report(args, env, tp_size, grad_accum_steps, results)

    import torch.distributed as dist

    if args.save_checkpoint_dir and params_override is not None:
        from training_engine_tensor.parameters import save_checkpoint

        dist.barrier()
        if env.rank == 0:
            print(
                f"  [engine] save_checkpoint at step {args.num_steps} -> "
                f"{args.save_checkpoint_dir}",
                flush=True,
            )
        save_checkpoint(
            checkpoint_dir=args.save_checkpoint_dir,
            params=params_override,
            optimizer_state=state_out.get("optimizer_state"),
            step=args.num_steps,
            num_samples=args.num_steps * args.global_batch_size,
            dataloader_state=None,
        )
        dist.barrier()
        if env.rank == 0:
            shard_path = (
                Path(args.save_checkpoint_dir) / f"rank_{env.rank}.pt"
            )
            print(
                f"  [engine] save_checkpoint done; rank0 shard at "
                f"{shard_path}",
                flush=True,
            )
    elif args.save_checkpoint_dir and params_override is None:
        if env.rank == 0:
            print(
                "  [engine] WARN: --save-checkpoint-dir set but params were "
                "loaded from --checkpoint-root, not held in this process; "
                "skipping save (the source checkpoint is already on disk).",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()

    return 0


def _emit_training_report(
    args: argparse.Namespace,
    env: _DistEnv,
    tp_size: int,
    grad_accum_steps: int,
    results: list[dict],
) -> None:
    """Rank-0 post-training output: jsonl + DONE + per-step ``[LOSS]`` lines.

      - ``OUTPUT_DIR/{N}step/ours_results.jsonl`` (one record per step)
      - ``OUTPUT_DIR/{N}step/DONE`` tombstone
      - one ``[LOSS] step=<N> global_loss=<F.10> grad_norm=<F.10>
        time_s=<F.6> mfu_e2e_standard=<F.6>`` line per step on stdout
        (the harness gate parser keys off this exact format)
      - optional ``--export-report`` dump (JSON of the raw results list)
    """
    import json

    from training_engine_tensor.config import (
        H100_PEAK_TFLOPS_BF16,
        compute_training_flops,
    )

    dp_size = max(1, env.world_size // tp_size)
    tokens_per_step = args.micro_batch_size * args.seq_length * dp_size * grad_accum_steps
    flops_per_step = compute_training_flops(tokens_per_step)
    peak_flops = float(H100_PEAK_TFLOPS_BF16) * 1.0e12

    output_dir_env = os.environ.get("OUTPUT_DIR")
    if output_dir_env:
        out_root = Path(output_dir_env).resolve() / f"{args.num_steps}step"
        out_root.mkdir(parents=True, exist_ok=True)
        with (out_root / "ours_results.jsonl").open("w", encoding="utf-8") as fh:
            for step, r in enumerate(results):
                fh.write(json.dumps({
                    "step": step,
                    "loss": r["loss"],
                    "grad_norm": r["grad_norm"],
                    "step_time": r["step_time"],
                }) + "\n")
        (out_root / "DONE").write_text("ok\n", encoding="utf-8")
        print(
            f"  [engine] wrote {out_root / 'ours_results.jsonl'} "
            f"({len(results)} lines)",
            flush=True,
        )

    for step, r in enumerate(results):
        step_time = float(r["step_time"])
        if step_time > 0.0:
            mfu = flops_per_step / step_time / env.world_size / peak_flops * 100.0
        else:
            mfu = float("nan")
        print(
            f"[LOSS] step={step} global_loss={r['loss']:.10f} "
            f"grad_norm={r['grad_norm']:.10f} time_s={step_time:.6f} "
            f"mfu_e2e_standard={mfu:.6f}",
            flush=True,
        )

    if args.export_report:
        out = Path(args.export_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(results, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"  [engine] wrote training report to {out}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(prog="training_engine_tensor")
    sub = parser.add_subparsers(dest="command")

    pt = sub.add_parser(
        "pretrain",
        help="Random-init pretraining with an HF dataloader.",
    )
    _add_pretrain_args(pt)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    # Build and install the global EngineConfig BEFORE any framework
    # module is imported (those may call get_config() at module level).
    from training_engine_tensor.engine_config import (
        from_env,
        set_global_config,
    )

    base_cfg = from_env()
    cfg = _apply_engine_config_overrides(base_cfg, args)
    set_global_config(cfg)

    if args.command == "pretrain":
        return _run_pretrain(args)

    print(f"Command '{args.command}' is not implemented.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
