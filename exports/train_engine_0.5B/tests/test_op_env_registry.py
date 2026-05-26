"""Tests for ops/ env var declaration via env_vars.toml.

Ensures every os.environ usage in ops/ code is explicitly declared in
the corresponding env_vars.toml file or in the global ENV_WHITELIST.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from training_engine_tensor.engine_config import ENV_WHITELIST, EngineConfig, set_global_config


def test_op_env_registry_populated():
    """After init(), OP_ENV_REGISTRY should contain keys from env_vars.toml files."""
    set_global_config(EngineConfig())
    from training_engine_tensor.op_dispatcher import OP_ENV_REGISTRY, init

    init(env=dict(os.environ))
    assert len(OP_ENV_REGISTRY) > 0, "OP_ENV_REGISTRY should have entries after init()"
    assert "QKV_DISABLE_CUSTOM_REDUCE" in OP_ENV_REGISTRY
    assert "AOP_NO_STREAM_CACHE" in OP_ENV_REGISTRY
    assert "GEMM_OUT_FWD_TILE_M" in OP_ENV_REGISTRY
    assert "FLASH_ATTENTION_ARCH" in OP_ENV_REGISTRY


def test_op_env_registry_has_op_field():
    """Each entry should track which op it belongs to."""
    set_global_config(EngineConfig())
    from training_engine_tensor.op_dispatcher import OP_ENV_REGISTRY, init

    init(env=dict(os.environ))
    entry = OP_ENV_REGISTRY.get("QKV_DISABLE_CUSTOM_REDUCE")
    assert entry is not None
    assert entry["op"] == "gemm_qkv_proj"


def test_env_vars_toml_covers_known_vars():
    """Spot-check that specific env vars from kernel.py are declared."""
    set_global_config(EngineConfig())
    from training_engine_tensor.op_dispatcher import OP_ENV_REGISTRY, init

    init(env=dict(os.environ))
    expected_keys = [
        "QKV_REDUCE_VARIANT",
        "AOP_FWD_TILE_M",
        "GEMM_OUT_WGRAD_TILE_M",
        "FA_CLC",
    ]
    for key in expected_keys:
        assert key in OP_ENV_REGISTRY, f"{key} should be declared in env_vars.toml"


# ---------------------------------------------------------------------------
# Full-scan test: every os.environ.get("X") in ops/**/*.py must be declared
# ---------------------------------------------------------------------------

_OPS_DIR = Path(__file__).resolve().parent.parent / "src" / "training_engine_tensor" / "ops"

# Vars that are universal system/toolchain contracts, not op-specific
_SYSTEM_VARS = frozenset({
    "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT",
    "CUDA_HOME", "CUTEDSL_CACHE_ROOT",
    "PYTHONPATH", "PATH", "HOME",
    "LD_PRELOAD", "TORCH_CUDA_ARCH_LIST",
    "CUTLASS_INSTALL_DIR",
})

_ENV_GET_RE = re.compile(
    r"""os\.environ\.get\(\s*["']([A-Z_][A-Z0-9_]*)["']"""
)
_ENV_BRACKET_RE = re.compile(
    r"""os\.environ\[["']([A-Z_][A-Z0-9_]*)["']\]"""
)
_ENV_SETDEFAULT_RE = re.compile(
    r"""os\.environ\.setdefault\(\s*["']([A-Z_][A-Z0-9_]*)["']"""
)


def _extract_env_var_names_from_file(path: Path) -> set[str]:
    """Extract all env var names referenced via os.environ in a .py file."""
    content = path.read_text(encoding="utf-8")
    names: set[str] = set()
    names.update(_ENV_GET_RE.findall(content))
    names.update(_ENV_BRACKET_RE.findall(content))
    names.update(_ENV_SETDEFAULT_RE.findall(content))
    return names


def test_all_ops_env_vars_declared():
    """Every env var used in ops/**/*.py must be in env_vars.toml or ENV_WHITELIST.

    This is the architectural lint that prevents undeclared env-var-driven
    behavior from creeping into the framework.
    """
    set_global_config(EngineConfig())
    from training_engine_tensor.op_dispatcher import OP_ENV_REGISTRY, init

    init(env=dict(os.environ))

    declared = set(OP_ENV_REGISTRY.keys()) | ENV_WHITELIST | _SYSTEM_VARS

    undeclared: list[tuple[str, str]] = []  # (var_name, file_path)

    for py_file in sorted(_OPS_DIR.rglob("*.py")):
        var_names = _extract_env_var_names_from_file(py_file)
        for var in sorted(var_names):
            if var not in declared:
                rel = py_file.relative_to(_OPS_DIR)
                undeclared.append((var, str(rel)))

    if undeclared:
        lines = [f"  {var} ({path})" for var, path in undeclared[:30]]
        msg = (
            f"{len(undeclared)} env var(s) used in ops/ but not declared "
            f"in env_vars.toml or ENV_WHITELIST:\n" + "\n".join(lines)
        )
        raise AssertionError(msg)
