"""Per-operator version dispatcher for Stage 2.

Each operator has an environment variable ``OP_<NAME>`` that selects
which implementation to use.  Values are ``baseline`` (stage1 frozen),
``v1``, ``v2``, etc.

Registration is automatic: each operator directory contains a
``register.toml`` file with its env_var, default version, and available
versions.  Calling ``init(env)`` scans ``workload/ops/*/register.toml``
and freezes versions.  No manual editing of this file is needed.

Usage in forward.py / backward.py::

    from training_engine_tensor.op_dispatcher import get_op_version
    if get_op_version("attention") == "baseline":
        ...  # original TE path
    else:
        from training_engine_tensor.ops.attention.kernel import attention_fwd
        ...
"""

from __future__ import annotations

__all__ = [
    "OP_ENV_REGISTRY",
    "OpEntry",
    "get_op_version",
    "init",
    "list_ops",
]

from dataclasses import dataclass
from pathlib import Path

from .ops._env_config import OpBuildEnv


@dataclass(frozen=True)
class OpEntry:
    env_var: str
    default: str
    available: tuple[str, ...]


_REGISTRY: dict[str, OpEntry] = {}
_FROZEN_VERSIONS: dict[str, str] = {}
_FROZEN_PROCESS_ENV: dict[str, str] | None = None
_BUILD_ENV: OpBuildEnv | None = None
_OP_ENVS: dict[str, dict[str, str]] = {}
OP_ENV_REGISTRY: dict[str, dict] = {}

_OPS_DIR = Path(__file__).resolve().parent / "ops"


def _auto_discover() -> None:
    """Scan workload/ops/*/register.toml and populate _REGISTRY."""
    if not _OPS_DIR.is_dir():
        return
    for reg_file in sorted(_OPS_DIR.glob("*/register.toml")):
        op_name = reg_file.parent.name
        if op_name.startswith("_"):
            continue
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib
            with open(reg_file, "rb") as f:
                data = tomllib.load(f)
        except Exception as exc:
            import warnings
            warnings.warn(f"op_dispatcher: skipping {reg_file}: {exc}", stacklevel=2)
            continue
        _REGISTRY[op_name] = OpEntry(
            env_var=data.get("env_var", f"OP_{op_name.upper()}"),
            default=data.get("default", "baseline"),
            available=tuple(data.get("available", ["baseline"])),
        )


def _freeze_versions(env: dict[str, str]) -> None:
    """Resolve OP_* versions once and cache them."""
    source = env
    _FROZEN_VERSIONS.clear()
    for op_name, entry in _REGISTRY.items():
        _FROZEN_VERSIONS[op_name] = source.get(entry.env_var, entry.default)


def _load_op_envs(env: dict[str, str]) -> None:
    """Parse env_vars.toml for each op, resolve values, and populate OP_ENV_REGISTRY."""
    _OP_ENVS.clear()
    OP_ENV_REGISTRY.clear()
    for op_name in _REGISTRY:
        toml_path = _OPS_DIR / op_name / "env_vars.toml"
        if not toml_path.exists():
            _OP_ENVS[op_name] = {}
            continue
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            _OP_ENVS[op_name] = {}
            continue
        declared = data.get("env_vars", {})
        resolved: dict[str, str] = {}
        for var_name, meta in declared.items():
            if isinstance(meta, dict):
                default = str(meta.get("default", ""))
                description = meta.get("description", "")
            else:
                default = str(meta)
                description = ""
            resolved[var_name] = env.get(var_name, default)
            OP_ENV_REGISTRY[var_name] = {
                "op": op_name,
                "default": default,
                "description": description,
            }
        _OP_ENVS[op_name] = resolved


def init(env: dict[str, str]) -> None:
    """Discover operators and freeze their versions from *env*.

    Must be called once by ``__main__.py`` after the process environment
    is fully set up but before any training code calls
    ``get_op_version()``.
    """
    global _BUILD_ENV, _FROZEN_PROCESS_ENV
    _auto_discover()
    _freeze_versions(env)
    _BUILD_ENV = OpBuildEnv.from_env(env)
    _FROZEN_PROCESS_ENV = dict(env)
    _load_op_envs(env)
    import os as _os
    _os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", _BUILD_ENV.torch_cuda_arch_list,
    )


def get_op_version(op_name: str) -> str:
    """Return the active version string for *op_name*.

    Versions are frozen by ``init(env=...)`` at startup.
    Unknown ops return ``"baseline"``.
    """
    frozen = _FROZEN_VERSIONS.get(op_name)
    if frozen is not None:
        return frozen
    return "baseline"


def get_frozen_env() -> dict[str, str]:
    """Startup environment snapshot passed to ``init()``; same keys as ``os.environ``.

    When ``init()`` has not run, returns ``dict(os.environ)`` so ad-hoc imports
    keep legacy ``os.environ.get`` parity.
    """
    if _FROZEN_PROCESS_ENV is not None:
        return _FROZEN_PROCESS_ENV
    import os as _os

    return dict(_os.environ)


def get_build_env() -> OpBuildEnv:
    """Return the frozen build environment.  Raises if ``init()`` not called."""
    if _BUILD_ENV is None:
        raise RuntimeError(
            "get_build_env(): op_dispatcher.init() has not been called yet."
        )
    return _BUILD_ENV


def get_op_env(op_name: str) -> dict[str, str]:
    """Return the resolved env_vars.toml values for *op_name*."""
    return dict(_OP_ENVS.get(op_name, {}))


def list_ops() -> dict[str, dict[str, str | tuple[str, ...]]]:
    """Return a snapshot of all registered operators and their active versions."""
    result = {}
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


from .engine_config import register_reset_hook as _register_reset_hook  # noqa: E402


def _clear_op_dispatcher() -> None:
    global _BUILD_ENV, _FROZEN_PROCESS_ENV
    _REGISTRY.clear()
    _FROZEN_VERSIONS.clear()
    _BUILD_ENV = None
    _FROZEN_PROCESS_ENV = None
    _OP_ENVS.clear()
    OP_ENV_REGISTRY.clear()


_register_reset_hook(_clear_op_dispatcher)
