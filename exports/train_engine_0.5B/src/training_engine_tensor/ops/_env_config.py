"""Shared build-environment configuration for all ops/ kernel modules.

All toolchain paths, local rank, and CUDA architecture target are read
ONCE at ``op_dispatcher.init()`` time, frozen into an ``OpBuildEnv``
dataclass, and then passed to kernel functions as a parameter.
No ops/ module should read ``os.environ`` directly for these values.
"""

from __future__ import annotations

__all__ = ["OpBuildEnv"]

from dataclasses import dataclass


@dataclass(frozen=True)
class OpBuildEnv:
    """Process-external build toolchain paths — read ONCE at init."""

    cuda_home: str = "/usr/local/cuda"
    cutedsl_cache_root: str = "/tmp"
    cutlass_install_dir: str = ""
    torch_cuda_arch_list: str = "9.0a"
    local_rank: int = 0

    @classmethod
    def from_env(cls, env: dict[str, str]) -> OpBuildEnv:
        """Construct from an env-var dict (NOT os.environ directly)."""
        return cls(
            cuda_home=env.get("CUDA_HOME", "/usr/local/cuda"),
            cutedsl_cache_root=env.get("CUTEDSL_CACHE_ROOT", "/tmp"),
            cutlass_install_dir=env.get("CUTLASS_INSTALL_DIR", ""),
            torch_cuda_arch_list=env.get("TORCH_CUDA_ARCH_LIST", "9.0a"),
            local_rank=int(env.get("LOCAL_RANK", env.get("RANK", "0"))),
        )
