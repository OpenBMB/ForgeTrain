"""Tests for training_engine_tensor.profiling — from_config factory."""

from __future__ import annotations

import os

import pytest

from training_engine_tensor.engine_config import EngineConfig, set_global_config


class TestFromConfig:
    """Verify that StepProfiler can be built from EngineConfig fields."""

    def test_disabled_by_default(self):
        cfg = EngineConfig()
        set_global_config(cfg)
        from training_engine_tensor.profiling import from_config

        profiler = from_config(cfg)
        assert profiler.enabled is False

    def test_enabled_with_range_and_output(self):
        cfg = EngineConfig(
            profile_range=(10, 20),
            profile_output="/tmp/test_profile.json",
        )
        set_global_config(cfg)
        from training_engine_tensor.profiling import from_config

        profiler = from_config(cfg)
        assert profiler.enabled is True
        assert profiler.range_start == 10
        assert profiler.range_end == 20

    def test_half_set_raises(self):
        cfg = EngineConfig(
            profile_range=(10, 20),
            profile_output=None,
        )
        set_global_config(cfg)
        from training_engine_tensor.profiling import from_config

        with pytest.raises(ValueError, match="must be set together"):
            from_config(cfg)

    def test_deep_without_range_raises(self):
        cfg = EngineConfig(profile_deep=True)
        set_global_config(cfg)
        from training_engine_tensor.profiling import from_config

        with pytest.raises(ValueError, match="requires.*profile_range"):
            from_config(cfg)

    def test_deep_with_range(self):
        cfg = EngineConfig(
            profile_range=(5, 15),
            profile_output="/tmp/deep.json",
            profile_deep=True,
        )
        set_global_config(cfg)
        from training_engine_tensor.profiling import from_config

        profiler = from_config(cfg)
        assert profiler.deep is True


class TestFromEnvProfilerBridge:
    """Verify that from_env() populates EngineConfig profiler fields.

    These are the TDD-red tests: from_env() currently does NOT read
    PROFILE_RANGE / PROFILE_OUTPUT / PROFILE_DEEP / HOST_TIMER,
    so the profiler fields remain at their defaults. These tests
    assert the correct behavior AFTER the fix.
    """

    def _with_env(self, overrides: dict[str, str]):
        """Context manager that temporarily sets env vars."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            saved = {}
            for k, v in overrides.items():
                saved[k] = os.environ.get(k)
                os.environ[k] = v
            try:
                yield
            finally:
                for k in overrides:
                    if saved[k] is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = saved[k]
        return _ctx()

    def test_from_env_populates_profile_range_and_output(self):
        import warnings

        from training_engine_tensor.engine_config import from_env

        with self._with_env({"PROFILE_RANGE": "10,20", "PROFILE_OUTPUT": "/tmp/x.json"}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cfg = from_env()

        assert cfg.profile_range == (10, 20)
        assert cfg.profile_output == "/tmp/x.json"

    def test_from_env_populates_profile_deep(self):
        import warnings

        from training_engine_tensor.engine_config import from_env

        with self._with_env({
            "PROFILE_RANGE": "5,15",
            "PROFILE_OUTPUT": "/tmp/deep.json",
            "PROFILE_DEEP": "1",
        }):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cfg = from_env()

        assert cfg.profile_deep is True

    def test_from_env_populates_host_timer(self):
        import warnings

        from training_engine_tensor.engine_config import from_env

        with self._with_env({
            "PROFILE_RANGE": "0,100",
            "PROFILE_OUTPUT": "/tmp/ht.json",
            "HOST_TIMER": "1",
        }):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cfg = from_env()

        assert cfg.host_timer is True

    def test_from_env_no_profile_vars_leaves_defaults(self):
        import warnings

        from training_engine_tensor.engine_config import from_env

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = from_env()

        assert cfg.profile_range is None
        assert cfg.profile_output is None
        assert cfg.profile_deep is False
        assert cfg.host_timer is False


class TestDeadCodeRemoval:
    """Verify that deprecated functions and constants have been removed."""

    def test_from_environment_removed(self):
        """from_environment() was superseded by from_config() and should not exist."""
        from training_engine_tensor import profiling

        assert not hasattr(profiling, "from_environment"), (
            "profiling.from_environment is dead code — use from_config()"
        )

    def test_env_constants_removed(self):
        """ENV_PROFILE_* constants were only used by from_environment()."""
        from training_engine_tensor import profiling

        for name in ("ENV_PROFILE_RANGE", "ENV_PROFILE_OUTPUT",
                     "ENV_PROFILE_DEEP", "ENV_HOST_TIMER"):
            assert not hasattr(profiling, name), (
                f"profiling.{name} is dead code after from_environment removal"
            )
