"""Tests for CLI default values — must stay in sync with model_spec.toml."""

from __future__ import annotations

from pathlib import Path


def test_cli_defaults_match_model_spec():
    """CLI default batch sizes must come from model_spec.toml, not hardcoded."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    spec_path = Path(__file__).resolve().parent.parent / "model_spec.toml"
    with open(spec_path, "rb") as f:
        spec = tomllib.load(f)

    from training_engine_tensor.__main__ import _add_common_train_args

    import argparse

    p = argparse.ArgumentParser()
    _add_common_train_args(p)
    defaults = p.parse_args([])

    assert defaults.micro_batch_size == spec["training"]["micro_batch_size"]
    assert defaults.global_batch_size == spec["training"]["global_batch_size"]
    assert defaults.seq_length == spec["model"]["seq_length"]
