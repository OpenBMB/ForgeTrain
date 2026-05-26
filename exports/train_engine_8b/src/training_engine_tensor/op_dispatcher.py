"""Per-operator version dispatcher for the GEMM kernels under ``ops/``.

Each operator directory under :mod:`training_engine_tensor.ops` ships a
``register.toml`` that names the operator's controlling environment
variable, its default version, and the full set of available versions::

    # ops/gemm_fc1/register.toml
    env_var = "OP_GEMM_FC1"
    default = "v1"
    available = ["baseline", "v1"]

This module scans those files at import time and exposes a single
read-only API, :func:`get_op_version`, that returns the active version
string for a given operator.  Resolution order is environment variable
override (e.g. ``OP_GEMM_FC1``) → ``register.toml`` default →
``"baseline"``.

The dispatcher is intentionally tiny and synchronous — it runs once
per process during entry-point bootstrap and the result is then read
by :mod:`training_engine_tensor.custom_gemm` whenever a GEMM call site
needs to choose between the self-developed kernel and the baseline
``torch.matmul`` path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "OpEntry",
    "get_op_version",
    "list_ops",
]


@dataclass(frozen=True)
class OpEntry:
    env_var: str
    default: str
    available: tuple[str, ...]


_REGISTRY: dict[str, OpEntry] = {}
_OPS_DIR = Path(__file__).resolve().parent / "ops"


def _auto_discover() -> None:
    """Populate ``_REGISTRY`` by scanning ``ops/*/register.toml``."""
    if not _OPS_DIR.is_dir():
        return
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover -- Python <3.11 fallback.
        import tomli as tomllib  # type: ignore[no-redef]

    for reg_file in sorted(_OPS_DIR.glob("*/register.toml")):
        op_name = reg_file.parent.name
        if op_name.startswith("_"):
            continue
        try:
            with reg_file.open("rb") as fh:
                data = tomllib.load(fh)
        except Exception:
            # A malformed register.toml MUST NOT crash the framework
            # on import; the operator simply stays unregistered and
            # falls back to baseline at the call site.
            continue
        _REGISTRY[op_name] = OpEntry(
            env_var=data.get("env_var", f"OP_{op_name.upper()}"),
            default=data.get("default", "baseline"),
            available=tuple(data.get("available", ["baseline"])),
        )


_auto_discover()


def get_op_version(op_name: str) -> str:
    """Return the active version for ``op_name``.

    Resolution order:

    1. Environment variable named in the operator's ``register.toml``
       (defaults to ``OP_<NAME>`` with hyphens converted to underscores).
    2. ``register.toml`` default.
    3. ``"baseline"`` for unknown operators.
    """
    entry = _REGISTRY.get(op_name)
    if entry is None:
        env_key = f"OP_{op_name.upper().replace('-', '_')}"
        return os.environ.get(env_key, "baseline")
    return os.environ.get(entry.env_var, entry.default)


def list_ops() -> dict[str, dict[str, str | tuple[str, ...]]]:
    """Snapshot every registered operator and its currently active version."""
    result: dict[str, dict[str, str | tuple[str, ...]]] = {}
    for name, entry in sorted(_REGISTRY.items()):
        result[name] = {
            "env_var": entry.env_var,
            "default": entry.default,
            "active": get_op_version(name),
            "available": entry.available,
        }
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(list_ops(), indent=2, default=str))
