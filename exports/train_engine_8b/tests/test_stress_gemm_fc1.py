"""Bare-metal stress test for ``gemm_fc1`` — SwiGLU column-parallel GEMM.

Exercises the self-developed CuTeDSL kernel
(:mod:`training_engine_tensor.ops.gemm_fc1.kernel`) on the production
8B shape (``S=4096, B=2, I=4096, O=16384``) in the production "all"
cadence (fwd → dgrad → wgrad) while six neighbour workers (W1..W5, W7)
saturate compute / memory / TMA / cluster-mode / event / allocator
paths on the same GPU.

Run as a pytest case (default ~30 s smoke profile)::

    pytest tests/test_stress_gemm_fc1.py -v

Run as a long-duration smoke (e.g. 15 min, 20 GB filler)::

    STRESS_DURATION_S=900 STRESS_FILL_GB=20 \\
        python tests/test_stress_gemm_fc1.py

Set ``STRESS_KERNEL=cublas`` to drive ``torch.matmul`` as a negative
control; the workers + oracle should still pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``tests.*`` and ``training_engine_tensor`` / ``flash_attn_dsl``
# importable both under pytest (where rootdir is on sys.path
# automatically) and under direct ``python tests/test_*.py`` invocation
# (where sys.path[0] is the script's directory).
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tests._stress_runner import run_stress  # noqa: E402


def test_stress_gemm_fc1_all() -> None:
    """Run gemm_fc1 fwd+dgrad+wgrad under W1..W5+W7 neighbour load."""
    kernel = os.environ.get("STRESS_KERNEL", "active")
    report = run_stress(op="gemm_fc1", kernel=kernel, mode="all")
    assert report["outcome"] == "PASS", report.get("fail_reason") or report


if __name__ == "__main__":  # pragma: no cover -- direct CLI entry
    import json
    kernel = os.environ.get("STRESS_KERNEL", "active")
    mode = os.environ.get("STRESS_MODE", "all")
    report = run_stress(op="gemm_fc1", kernel=kernel, mode=mode)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["outcome"] == "PASS" else 1)
