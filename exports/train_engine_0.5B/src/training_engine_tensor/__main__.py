"""CLI entry point for the MiniCPM4 0.5B training engine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

try:
    import tomllib as _tomllib
except ModuleNotFoundError:
    import tomli as _tomllib  # type: ignore[no-redef]

_SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "model_spec.toml"
with open(_SPEC_PATH, "rb") as _f:
    _SPEC = _tomllib.load(_f)
_TRAINING_SPEC = _SPEC["training"]
_MODEL_SPEC = _SPEC["model"]


def _add_engine_config_args(parser: argparse.ArgumentParser) -> None:
    """Auto-generate CLI flags from ``EngineConfig`` dataclass fields.

    Bool fields become ``--flag`` / ``--no-flag`` pairs (default ``None``).
    Numeric fields become ``--field-name`` with the appropriate type
    (default ``None`` so from_env() defaults are preserved when unset).
    """
    from training_engine_tensor.engine_config import EngineConfig

    for f in dataclasses.fields(EngineConfig):
        cli_name = "--" + f.name.replace("_", "-")
        if f.type == "bool" or f.type is bool:
            parser.add_argument(
                cli_name,
                action=argparse.BooleanOptionalAction,
                default=None,
            )
        elif f.type == "int" or f.type is int:
            parser.add_argument(cli_name, type=int, default=None)
        elif f.type == "float" or f.type is float:
            parser.add_argument(cli_name, type=float, default=None)
        else:
            parser.add_argument(cli_name, default=None)


def _apply_cli_overrides(
    base: "EngineConfig", args: argparse.Namespace,  # noqa: F821
) -> "EngineConfig":  # noqa: F821
    """Return a new ``EngineConfig`` with non-None CLI values applied."""
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
    g.add_argument("--hf-dataset", default=None,
                   help="HuggingFace dataset path (local dir or Hub name).")
    g.add_argument("--tokenizer-path", default=None,
                   help="HuggingFace tokenizer path (required with --hf-dataset)")
    g.add_argument("--hf-dataset-config", default=None,
                   help="Dataset config/subset name (e.g. 'main' for GSM8K)")
    g.add_argument("--hf-dataset-split", default="train",
                   help="Dataset split (default: train)")
    g.add_argument("--hf-text-field", nargs="+", default=None,
                   help="Text column name(s) to extract from dataset")
    g.add_argument("--hf-text-template", default=None,
                   help="Python format string to combine columns, e.g. "
                        "'Question: {question}\\nAnswer: {answer}'")


def _add_distributed_args(p: argparse.ArgumentParser) -> None:
    """Register the minimal distributed/device args shared by ALL subcommands."""
    p.add_argument("--data-parallel-rank", type=int, default=0)
    p.add_argument("--data-parallel-ranks", nargs="+", type=int, default=[])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default=None)
    p.add_argument("--export-checkpoint-root", default=None)


def _add_common_train_args(p: argparse.ArgumentParser) -> None:
    """Register CLI arguments shared by the ``train`` and ``pretrain`` subcommands."""
    _add_distributed_args(p)
    p.add_argument("--num-local-microbatches", type=int, default=None)
    p.add_argument("--global-batch-size", type=int,
                   default=int(_TRAINING_SPEC["global_batch_size"]),
                   help="Global batch size; auto-computes num_local_microbatches if not set")
    p.add_argument("--micro-batch-size", type=int,
                   default=int(_TRAINING_SPEC["micro_batch_size"]))
    p.add_argument("--seq-length", type=int,
                   default=int(_MODEL_SPEC["seq_length"]))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dump-inputs-dir", default=None,
                   help="Dump per-step input tensors for data ordering verification")
    p.add_argument("--save-interval", type=int, default=None,
                   help="Resume checkpoint frequency in global steps "
                        "(unset / <=0 disables periodic saves)")
    p.add_argument("--save-dir", default=None,
                   help="Directory for periodic resume checkpoints "
                        "(written via save_checkpoint(async_io=True))")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint directory previously "
                        "written by --save-dir (loads params + optimizer "
                        "state + dataloader cursor; step_indices auto-"
                        "advance to start at the resumed step)")
    p.add_argument("--mup-emb-scale", type=float, default=0.0,
                   help="muP embedding output scaling factor (0 = disabled)")
    p.add_argument("--mup-depth-scale", type=float, default=0.0,
                   help="muP residual depth scaling factor (0 = disabled)")
    p.add_argument("--max-lr", type=float, default=None,
                   help="Peak learning rate (overrides optimizer.py default)")
    p.add_argument("--min-lr", type=float, default=None,
                   help="Minimum learning rate after decay")
    p.add_argument("--warmup-iters", type=int, default=None,
                   help="Number of warmup iterations")
    p.add_argument("--lr-decay-iters", type=int, default=None,
                   help="Total iterations for LR decay schedule")
    p.add_argument("--wsd-decay-iters", type=int, default=None,
                   help="WSD exponential decay phase iterations")
    p.add_argument("--mtp-num-layers", type=int, default=0,
                   help="Number of Multi-Token Prediction (MTP) layers "
                        "(0 = disabled)")
    p.add_argument("--mtp-loss-scaling-factor", type=float, default=0.1,
                   help="MTP loss weight (per-layer scale = factor / num_layers)")


def main() -> int:
    parser = argparse.ArgumentParser(prog="training_engine_tensor")
    sub = parser.add_subparsers(dest="command")

    ctl = sub.add_parser("checkpoint-train-loop")
    ctl.add_argument("--checkpoint-root", required=True)
    ctl.add_argument("--tokenized-inputs", required=True)
    ctl.add_argument("--step-indices", nargs="+", type=int, required=True)
    _add_distributed_args(ctl)
    ctl.add_argument("--num-local-microbatches", type=int, default=1)

    tr = sub.add_parser("train")
    tr.add_argument("--checkpoint-root", required=True)
    tr.add_argument("--num-steps", type=int, default=None,
                     help="Number of training steps (alternative to --step-indices)")
    tr.add_argument("--step-indices", nargs="+", type=int, default=None)
    _add_hf_data_args(tr)
    _add_common_train_args(tr)

    pt = sub.add_parser("pretrain")
    pt.add_argument("--num-steps", type=int, default=100)
    _add_hf_data_args(pt)
    _add_common_train_args(pt)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    # Build and install the global EngineConfig BEFORE any framework
    # module is imported (those may call get_config() at module level).
    from training_engine_tensor.engine_config import from_env, set_global_config
    base_cfg = from_env()
    overrides: dict = {}

    if args.command in ("train", "pretrain"):
        rc = _validate_resume_save_dir_mutex(args)
        if rc != 0:
            return rc
        _mup_emb = getattr(args, "mup_emb_scale", 0.0) or 0.0
        _mup_depth = getattr(args, "mup_depth_scale", 0.0) or 0.0
        if _mup_emb != base_cfg.mup_emb_scale:
            overrides["mup_emb_scale"] = _mup_emb
        if _mup_depth != base_cfg.mup_depth_scale:
            overrides["mup_depth_scale"] = _mup_depth

        _mtp_num = int(getattr(args, "mtp_num_layers", 0) or 0)
        _mtp_scale = float(getattr(args, "mtp_loss_scaling_factor", 0.1))
        if _mtp_num != base_cfg.mtp_num_layers:
            overrides["mtp_num_layers"] = _mtp_num
        if _mtp_scale != base_cfg.mtp_loss_scaling_factor:
            overrides["mtp_loss_scaling_factor"] = _mtp_scale

        def _cli_override(attr: str, cli_name: str) -> None:
            v = getattr(args, cli_name, None)
            if v is not None:
                overrides[attr] = v
        _cli_override("max_lr", "max_lr")
        _cli_override("min_lr", "min_lr")
        _cli_override("warmup_iters", "warmup_iters")
        _cli_override("lr_decay_iters", "lr_decay_iters")
        _cli_override("wsd_decay_iters", "wsd_decay_iters")

    cfg = dataclasses.replace(base_cfg, **overrides) if overrides else base_cfg
    set_global_config(cfg)

    from training_engine_tensor.op_dispatcher import init as _init_op_dispatcher
    _init_op_dispatcher(env=dict(os.environ))

    if args.command in ("train", "pretrain"):
        from training_engine_tensor import config as _cfg
        if cfg.mup_emb_scale > 0 or cfg.mup_depth_scale > 0:
            print(
                f"[muP] emb_scale={cfg.mup_emb_scale} "
                f"depth_scale={cfg.mup_depth_scale} "
                f"residual_scale={_cfg.mup_residual_scale():.6f} "
                f"width_scale={_cfg.mup_width_scale():.1f}",
                flush=True,
            )
        if cfg.mtp_num_layers > 0:
            print(
                f"[MTP] num_layers={cfg.mtp_num_layers} "
                f"loss_scaling_factor={cfg.mtp_loss_scaling_factor} "
                f"per_layer_scale={_cfg.mtp_per_layer_loss_scale():.6f}",
                flush=True,
            )

    if args.command == "checkpoint-train-loop":
        return _run_checkpoint_train_loop(args)

    if args.command == "train":
        return _run_train(args)

    if args.command == "pretrain":
        return _run_pretrain(args)

    print(f"Command '{args.command}' is not yet implemented.", file=sys.stderr)
    return 1


def _validate_resume_save_dir_mutex(args: argparse.Namespace) -> int:
    """Reject ``--resume DIR --save-dir DIR`` (read-from == write-to).

    The async-save protocol guarantees a single writer, so resuming and
    saving back into the same dir would not corrupt the on-disk file —
    but the semantics are reversed-direction: Phase B would clobber the
    Phase A checkpoint with its own ``training_state.pt``, leaving the
    on-disk record reflecting the *resumed* run rather than the original
    save. Operators almost always intend the two paths to be different
    directories (resume from ``ckpt-A`` while writing into ``ckpt-B``).
    Failing fast here surfaces the typo at startup rather than after a
    successful Phase A.
    """
    resume_dir = getattr(args, "resume", None)
    save_dir = getattr(args, "save_dir", None)
    if resume_dir is None or save_dir is None:
        return 0
    try:
        resume_resolved = Path(resume_dir).resolve()
        save_resolved = Path(save_dir).resolve()
    except OSError as exc:
        print(
            f"ERROR: failed to resolve --resume / --save-dir paths: {exc}",
            file=sys.stderr,
        )
        return 1
    if resume_resolved == save_resolved:
        print(
            f"ERROR: --resume and --save-dir point at the same directory "
            f"({resume_resolved}). Reading the resume checkpoint and writing "
            f"the next periodic save into the same path is almost always a "
            f"typo — pass distinct directories so the original save survives.",
            file=sys.stderr,
        )
        return 2
    return 0


@dataclasses.dataclass(frozen=True, slots=True)
class _DistEnv:
    """Resolved distributed environment — single source for all env-var reads."""
    rank: int
    local_rank: int
    world_size: int
    device: str
    nccl_uid_file: str | None
    master_addr: str
    master_port: str


def _resolve_distributed_env(args: argparse.Namespace) -> _DistEnv:
    """Extract rank/device/world_size from env vars with CLI fallback.

    Environment variables (RANK, LOCAL_RANK, WORLD_SIZE, NCCL_UNIQUE_ID_FILE,
    MASTER_ADDR, MASTER_PORT) are whitelisted process-external contracts
    injected by torchrun / cctl.
    """
    rank = int(os.environ.get("RANK", str(args.data_parallel_rank)))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = args.device or f"cuda:{local_rank}"
    nccl_uid_file = os.environ.get("NCCL_UNIQUE_ID_FILE")
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    return _DistEnv(
        rank, local_rank, world_size, device,
        nccl_uid_file, master_addr, master_port,
    )


def _run_checkpoint_train_loop(args: argparse.Namespace) -> int:
    """Execute the training loop from checkpoint + pre-tokenized inputs."""
    from training_engine_tensor.engine_config import get_config
    from training_engine_tensor.profiling import from_config as profiler_from_config
    from training_engine_tensor.train_loop import run_training_loop

    env = _resolve_distributed_env(args)
    profiler = profiler_from_config(get_config())

    results = run_training_loop(
        checkpoint_root=args.checkpoint_root,
        step_indices=args.step_indices,
        dp_rank=env.rank,
        world_size=env.world_size,
        num_local_microbatches=args.num_local_microbatches,
        seed=args.seed,
        device=env.device,
        tokenized_inputs_path=args.tokenized_inputs,
        profiler=profiler,
    )

    _save_report(results, args.export_checkpoint_root, env.rank)
    return 0


def _resolve_resume_state(
    args: argparse.Namespace,
    rank: int,
    device: str,
    label: str,
) -> dict | None:
    """Load resume checkpoint if ``--resume`` is set, printing a summary."""
    if args.resume is None:
        return None
    from training_engine_tensor.parameters import load_resume_checkpoint
    state = load_resume_checkpoint(args.resume, device=device)
    if rank == 0:
        print(
            f"[{label}] --resume {args.resume} → "
            f"start_step={int(state['step'])} "
            f"num_samples={state['num_samples']}"
        )
    return state


def _resolve_step_indices(
    args: argparse.Namespace,
    resume_state: dict | None,
) -> list[int] | None:
    """Compute step_indices from CLI args + resume state.

    Returns None if an error message has been printed (caller should exit).
    """
    explicit = getattr(args, "step_indices", None)
    num_steps = getattr(args, "num_steps", None)
    start_step = int(resume_state["step"]) if resume_state is not None else 0

    if explicit is not None:
        indices = explicit
        if resume_state is not None:
            indices = [s for s in indices if s >= start_step]
    elif num_steps is not None:
        indices = list(range(start_step, num_steps))
    else:
        print("ERROR: either --step-indices or --num-steps is required", file=sys.stderr)
        return None
    return indices


def _resolve_local_microbatches(args: argparse.Namespace, world_size: int) -> int:
    if args.num_local_microbatches is not None:
        return args.num_local_microbatches
    n = args.global_batch_size // (args.micro_batch_size * world_size)
    return max(n, 1)


def _build_dataloader_and_prefetcher(
    args: argparse.Namespace,
    env: _DistEnv,
    resume_state: dict | None,
):
    """Build dataloader + prefetcher + state provider.

    Backend: ``--hf-dataset`` → HuggingFace ``datasets``.

    Returns ``(data_iterator, dl_provider)`` — both None when no dataset arg is set.
    """
    hf_dataset = getattr(args, "hf_dataset", None)
    data_path = getattr(args, "data_path", None)

    if data_path:
        raise ValueError(
            "--data-path is no longer supported; use --hf-dataset to point "
            "at a HuggingFace dataset (local path or Hub name)."
        )

    if hf_dataset:
        return _build_hf_dataloader_and_prefetcher(args, env)

    return None, None


def _build_hf_dataloader_and_prefetcher(
    args: argparse.Namespace,
    env: _DistEnv,
):
    """Build HuggingFace dataset dataloader + prefetcher.

    Returns ``(data_iterator, None)`` — no state provider (HF dataloader
    restarts from epoch boundary on resume).
    """
    tokenizer_path = getattr(args, "tokenizer_path", None)
    if not tokenizer_path:
        raise ValueError("--tokenizer-path is required when using --hf-dataset")

    from training_engine_tensor.data import DataPrefetcher
    from training_engine_tensor.engine_config import get_config
    from training_engine_tensor.hf_dataloader import build_hf_dataloader

    dataloader = build_hf_dataloader(
        dataset_path=args.hf_dataset,
        tokenizer_path=tokenizer_path,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        seed=args.seed,
        dp_rank=env.rank,
        world_size=env.world_size,
        num_workers=0,
        dataset_config=getattr(args, "hf_dataset_config", None),
        dataset_split=getattr(args, "hf_dataset_split", "train"),
        text_fields=getattr(args, "hf_text_field", None),
        text_template=getattr(args, "hf_text_template", None),
    )

    prefetcher = DataPrefetcher(
        dataloader, device=env.device,
        depth=get_config().data_prefetch_depth,
    )
    return prefetcher, None


def _run_common(args: argparse.Namespace, mode: str) -> int:
    """Shared implementation for 'train' and 'pretrain' subcommands."""
    from training_engine_tensor.engine_config import get_config
    from training_engine_tensor.profiling import from_config as profiler_from_config
    from training_engine_tensor.train_loop import run_training_loop

    is_pretrain = mode == "pretrain"
    tag = mode if mode != "train" else "train-cmd"

    env = _resolve_distributed_env(args)
    profiler = profiler_from_config(get_config())
    resume_state = _resolve_resume_state(args, env.rank, env.device, tag)

    step_indices = _resolve_step_indices(args, resume_state)
    if step_indices is None:
        return 1
    if not step_indices:
        if env.rank == 0:
            print(f"[{tag}] WARNING: empty step_indices after resume", file=sys.stderr)
        return 0

    num_local_microbatches = _resolve_local_microbatches(args, env.world_size)

    data_iterator, dl_provider = _build_dataloader_and_prefetcher(
        args, env, resume_state,
    )
    if data_iterator is not None:
        print(f"[{tag}] MBS={args.micro_batch_size}, GBS={args.global_batch_size}, "
              f"DP={env.world_size}, num_local_microbatches={num_local_microbatches}, "
              f"steps={len(step_indices)}, rank={env.rank}/{env.world_size}")
    elif is_pretrain:
        print(f"[{tag}] using synthetic random data "
              f"(MBS={args.micro_batch_size}, seq_len={args.seq_length})")

    results = run_training_loop(
        checkpoint_root="/tmp" if is_pretrain else args.checkpoint_root,
        step_indices=step_indices,
        dp_rank=env.rank,
        world_size=env.world_size,
        num_local_microbatches=num_local_microbatches,
        seed=args.seed,
        device=env.device,
        data_iterator=data_iterator,
        tokenized_inputs_path=None if is_pretrain else getattr(args, "tokenized_inputs", None),
        dump_inputs_dir=getattr(args, "dump_inputs_dir", None),
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        dataloader_state_provider=dl_provider,
        resume_state=resume_state,
        random_init=(is_pretrain and args.resume is None),
        micro_batch_size=args.micro_batch_size,
        profiler=profiler,
    )

    _save_report(results, args.export_checkpoint_root, env.rank)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """Execute training with a HuggingFace ``datasets`` dataloader."""
    if not getattr(args, "hf_dataset", None):
        print("ERROR: --hf-dataset is required for 'train'", file=sys.stderr)
        return 1
    return _run_common(args, "train")


def _run_pretrain(args: argparse.Namespace) -> int:
    """Pre-training from random initialization."""
    return _run_common(args, "pretrain")


def _save_report(
    results: dict, export_root: str | None, rank: int,
) -> None:
    if export_root and rank == 0:
        out_dir = Path(export_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / "training_report.json"
        report.write_text(
            json.dumps(results, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
