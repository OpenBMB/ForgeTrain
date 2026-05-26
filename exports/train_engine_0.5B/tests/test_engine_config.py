"""Tests for training_engine_tensor.engine_config."""

from __future__ import annotations

import warnings

import pytest

from training_engine_tensor.engine_config import (
    ENV_WHITELIST,
    EngineConfig,
    get_config,
    set_global_config,
)


class TestGlobalConfigLifecycle:
    """Verify the set-once / get / reset singleton protocol."""

    def test_get_before_set_raises(self):
        with pytest.raises(RuntimeError, match="set_global_config.*has not been called"):
            get_config()

    def test_set_and_get(self):
        cfg = EngineConfig(fused_ops=True)
        set_global_config(cfg)
        assert get_config() is cfg
        assert get_config().fused_ops is True

    def test_double_set_raises(self):
        set_global_config(EngineConfig())
        with pytest.raises(RuntimeError, match="called twice"):
            set_global_config(EngineConfig())

    def test_default_values(self):
        cfg = EngineConfig()
        assert cfg.fused_ops is False
        assert cfg.wgrad_overlap is True
        assert cfg.max_lr == pytest.approx(3e-4)
        assert cfg.vocab_size == 0

    def test_frozen(self):
        cfg = EngineConfig()
        with pytest.raises(AttributeError):
            cfg.fused_ops = True  # type: ignore[misc]


class TestEnvWhitelistSplit:
    """Verify ENV_WHITELIST only contains process-external contracts."""

    def test_whitelist_excludes_behavioral_flags(self):
        behavioral = {"FUSED_OPS", "STEP_CUDA_GRAPH", "CUSTOM_GEMM", "WGRAD_OVERLAP"}
        for key in behavioral:
            assert key not in ENV_WHITELIST, (
                f"{key} is a behavioral flag and must not be in ENV_WHITELIST"
            )

    def test_whitelist_excludes_profiler_vars(self):
        """PROFILE_* are deprecated migration-bridge vars, not permanent contracts."""
        profiler_vars = {"PROFILE_RANGE", "PROFILE_OUTPUT", "PROFILE_DEEP", "HOST_TIMER"}
        for key in profiler_vars:
            assert key not in ENV_WHITELIST, (
                f"{key} belongs in _DEPRECATED_ENV_BRIDGE, not ENV_WHITELIST"
            )

    def test_whitelist_includes_process_external(self):
        required = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
        for key in required:
            assert key in ENV_WHITELIST, (
                f"{key} is a process-external contract and must be in ENV_WHITELIST"
            )

    def test_deprecated_bridge_exists(self):
        from training_engine_tensor.engine_config import _DEPRECATED_ENV_BRIDGE

        assert isinstance(_DEPRECATED_ENV_BRIDGE, frozenset)
        assert "FUSED_OPS" in _DEPRECATED_ENV_BRIDGE
        assert "STEP_CUDA_GRAPH" in _DEPRECATED_ENV_BRIDGE
        assert "CUSTOM_GEMM" in _DEPRECATED_ENV_BRIDGE

    def test_from_env_warns_on_deprecated(self):
        import os

        from training_engine_tensor.engine_config import from_env

        old_val = os.environ.get("FUSED_OPS")
        try:
            os.environ["FUSED_OPS"] = "1"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from_env()
                dep_warnings = [
                    x for x in w
                    if issubclass(x.category, DeprecationWarning) and "FUSED_OPS" in str(x.message)
                ]
                assert len(dep_warnings) >= 1, "from_env() must warn when deprecated env vars are set"
        finally:
            if old_val is None:
                os.environ.pop("FUSED_OPS", None)
            else:
                os.environ["FUSED_OPS"] = old_val


class TestVocabSizeSSoT:
    """vocab_size should have a single source of truth in model_spec.toml."""

    def test_engine_config_default_zero(self):
        cfg = EngineConfig()
        assert cfg.vocab_size == 0, (
            "EngineConfig default should be 0 (sentinel), actual value comes from model_spec"
        )


class TestCLIArgs:
    """Verify that __main__.py exposes CLI args for all EngineConfig fields."""

    def test_add_engine_config_args_exists(self):
        from training_engine_tensor.__main__ import _add_engine_config_args  # noqa: F401

    def test_cli_bool_flag_override(self):
        import argparse

        from training_engine_tensor.__main__ import _add_engine_config_args

        p = argparse.ArgumentParser()
        _add_engine_config_args(p)
        args = p.parse_args(["--fused-ops"])
        assert args.fused_ops is True

    def test_cli_default_is_none(self):
        """Unset CLI flags should be None so from_env() defaults are preserved."""
        import argparse

        from training_engine_tensor.__main__ import _add_engine_config_args

        p = argparse.ArgumentParser()
        _add_engine_config_args(p)
        args = p.parse_args([])
        assert args.fused_ops is None
