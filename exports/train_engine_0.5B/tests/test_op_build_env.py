"""Tests for OpBuildEnv and op_dispatcher env config infrastructure."""

from __future__ import annotations

import ast
from pathlib import Path

from training_engine_tensor.engine_config import EngineConfig, set_global_config


def test_op_build_env_from_dict():
    """OpBuildEnv can be constructed from a plain dict, no os.environ."""
    from training_engine_tensor.ops._env_config import OpBuildEnv

    env = {
        "CUDA_HOME": "/test/cuda",
        "CUTEDSL_CACHE_ROOT": "/test/cache",
        "CUTLASS_INSTALL_DIR": "/test/cutlass",
        "LOCAL_RANK": "3",
    }
    cfg = OpBuildEnv.from_env(env)
    assert cfg.cuda_home == "/test/cuda"
    assert cfg.cutedsl_cache_root == "/test/cache"
    assert cfg.cutlass_install_dir == "/test/cutlass"
    assert cfg.local_rank == 3


def test_op_build_env_defaults():
    """OpBuildEnv uses sensible defaults when env is empty."""
    from training_engine_tensor.ops._env_config import OpBuildEnv

    cfg = OpBuildEnv.from_env({})
    assert cfg.cuda_home == "/usr/local/cuda"
    assert cfg.cutedsl_cache_root == "/tmp"
    assert cfg.cutlass_install_dir == ""
    assert cfg.local_rank == 0
    assert cfg.torch_cuda_arch_list == "9.0a"


def test_op_build_env_is_frozen():
    """OpBuildEnv must be immutable."""
    from training_engine_tensor.ops._env_config import OpBuildEnv

    cfg = OpBuildEnv.from_env({})
    try:
        cfg.cuda_home = "/changed"  # type: ignore[misc]
        raise AssertionError("should have raised FrozenInstanceError")
    except AttributeError:
        pass


def test_op_dispatcher_exposes_build_env():
    """After init(), get_build_env() returns a valid OpBuildEnv."""
    set_global_config(EngineConfig())
    from training_engine_tensor import op_dispatcher
    from training_engine_tensor.ops._env_config import OpBuildEnv

    op_dispatcher.init(env={"CUDA_HOME": "/test/init", "LOCAL_RANK": "7"})
    build_env = op_dispatcher.get_build_env()
    assert isinstance(build_env, OpBuildEnv)
    assert build_env.cuda_home == "/test/init"
    assert build_env.local_rank == 7


def test_op_dispatcher_exposes_op_env():
    """After init(), get_op_env() returns parsed env_vars.toml values."""
    set_global_config(EngineConfig())
    from training_engine_tensor import op_dispatcher

    op_dispatcher.init(env={"FC2_AOT": "0"})
    fc2_env = op_dispatcher.get_op_env("gemm_fc2")
    assert isinstance(fc2_env, dict)
    assert fc2_env["FC2_AOT"] == "0"


def _is_os_environ(node: ast.expr) -> bool:
    """Check if an AST node represents ``os.environ``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def test_no_environ_write_in_ops():
    """ops/ source files must not write to os.environ (os.environ[...] = ...)."""
    ops_dir = Path(__file__).resolve().parent.parent / "src" / "training_engine_tensor" / "ops"
    violations = []
    for py_file in ops_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and _is_os_environ(target.value)
                    ):
                        rel = py_file.relative_to(ops_dir.parent.parent.parent)
                        violations.append(f"{rel}:{node.lineno}")
    assert violations == [], (
        f"os.environ write found in ops/:\n" + "\n".join(violations)
    )


_ALLOWED_ENVIRON_PATTERNS = {
    "dict(os.environ)",
    "_init_build_env(dict(os.environ))",
}


def test_no_environ_get_in_ops():
    """ops/ must not call os.environ.get() — use get_frozen_env()/get_build_env()."""
    ops_dir = Path(__file__).resolve().parent.parent / "src" / "training_engine_tensor" / "ops"
    violations = []
    for py_file in ops_dir.rglob("*.py"):
        if py_file.name == "_env_config.py":
            continue
        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and _is_os_environ(node.func.value)
            ):
                rel = py_file.relative_to(ops_dir.parent.parent.parent)
                violations.append(f"{rel}:{node.lineno}")
            if (
                isinstance(node, ast.Subscript)
                and _is_os_environ(node.value)
            ):
                rel = py_file.relative_to(ops_dir.parent.parent.parent)
                violations.append(f"{rel}:{node.lineno}")
    assert violations == [], (
        f"os.environ.get() / os.environ[...] found in ops/:\n"
        + "\n".join(violations)
    )


def test_no_os_environ_setdefault_in_ops():
    """ops/ must not call os.environ.setdefault()."""
    ops_dir = Path(__file__).resolve().parent.parent / "src" / "training_engine_tensor" / "ops"
    violations = []
    for py_file in ops_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and _is_os_environ(node.func.value)
            ):
                rel = py_file.relative_to(ops_dir.parent.parent.parent)
                violations.append(f"{rel}:{node.lineno}")
    assert violations == [], (
        f"os.environ.setdefault() found in ops/:\n" + "\n".join(violations)
    )
