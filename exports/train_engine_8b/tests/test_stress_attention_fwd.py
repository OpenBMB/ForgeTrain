"""Bare-metal stress test for ``attention_fwd`` — SM90a flash-attention DSL forward.

Exercises the self-developed CuTeDSL flash-attention forward kernel
(:func:`flash_attn_dsl.host.run_flash_fwd`) on the production 8B shape:
``B=2, H_q=16, H_kv=1 (GQA), N=4096, D=128`` with causal masking and
BF16 dtype.  Six neighbour workers (W1..W5, W7) saturate compute /
memory / TMA / cluster-mode / event / allocator paths on the same GPU.

The reference oracle is PyTorch's
:func:`torch.nn.functional.scaled_dot_product_attention` (with the
H_kv dim ``.expand`` ed up to H_q — GQA broadcasting is zero-copy).

Run as a pytest case (default ~30 s smoke profile)::

    pytest tests/test_stress_attention_fwd.py -v

Run as a long-duration smoke (e.g. 15 min, 20 GB filler)::

    STRESS_DURATION_S=900 STRESS_FILL_GB=20 \\
        python tests/test_stress_attention_fwd.py

Set ``STRESS_KERNEL=sdpa`` to drive the SDPA reference as a negative
control.
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


def test_stress_attention_fwd() -> None:
    """Run flash-attention forward under W1..W5+W7 neighbour load."""
    kernel = os.environ.get("STRESS_KERNEL", "active")
    report = run_stress(op="attention_fwd", kernel=kernel, mode="fwd")
    assert report["outcome"] == "PASS", report.get("fail_reason") or report


if __name__ == "__main__":  # pragma: no cover -- direct CLI entry
    import json
    kernel = os.environ.get("STRESS_KERNEL", "active")
    report = run_stress(op="attention_fwd", kernel=kernel, mode="fwd")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["outcome"] == "PASS" else 1)
