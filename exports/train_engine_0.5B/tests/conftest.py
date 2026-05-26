"""Shared fixtures for training_engine_tensor tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_global_config():
    """Reset the EngineConfig singleton before and after every test.

    Guarantees test isolation: no test can leak config state to another.
    """
    from training_engine_tensor.engine_config import _reset_global_config

    _reset_global_config()
    yield
    _reset_global_config()
