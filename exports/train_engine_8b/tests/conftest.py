"""Shared fixtures for the training_engine_tensor stress tests.

The fixtures here gate every test in ``tests/`` on a real H100 / SM90a
GPU.  Run with::

    pytest tests/ -v

The runtime knobs (duration, fill-GB, enabled workers, ...) are env-var
overridable so the same test files double as long-run smoke targets for
CI and a 30-second dev loop locally.  See ``tests/_stress_runner.py``
for the full list of ``STRESS_*`` env vars.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Make ``src/`` importable from the tests dir without forcing the user
# to remember ``PYTHONPATH=src`` on the command line.  Both engine code
# (``training_engine_tensor``) and the DSL kernel (``flash_attn_dsl``)
# live under ``src/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _has_h100() -> bool:
    """Return True iff a Hopper SM90 device is visible to this process.

    Set ``STRESS_ALLOW_NON_H100=1`` to bypass the capability check (useful
    when porting the stress harness to a non-Hopper Ampere/Ada dev box —
    the CuTeDSL kernels will still refuse to run, but the worker harness
    + dispatch lookups are useful to validate against PyTorch SDPA / cuBLAS
    on any modern NVIDIA GPU).
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if os.environ.get("STRESS_ALLOW_NON_H100", "0") == "1":
        return True
    cap = torch.cuda.get_device_capability(0)
    return cap[0] == 9


_SKIP_REASON = (
    "stress tests require a real Hopper SM90 GPU (NVIDIA H100 SXM5); "
    "set STRESS_ALLOW_NON_H100=1 to bypass the device-capability check"
)


@pytest.fixture(scope="session", autouse=True)
def _require_h100() -> None:
    """Session-level guard: skip the entire stress suite without a GPU."""
    if not _has_h100():
        pytest.skip(_SKIP_REASON, allow_module_level=True)
