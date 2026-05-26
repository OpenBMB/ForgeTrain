"""Tests for bundled Cybertron job spec examples."""

from __future__ import annotations

import json
from pathlib import Path


def test_hf_gsm8k_smoke_spec_uses_single_node_8gpu():
    """The public HF GSM8K smoke example should fit an 8-GPU dev node."""
    spec_path = (
        Path(__file__).resolve().parent.parent
        / "job_specs"
        / "smoke"
        / "hf_gsm8k_8gpu_p1.json"
    )
    spec = json.loads(spec_path.read_text())
    task = spec["task"]

    assert task["resources"]["gpuCount"] == 8
    assert "replicas" not in task
