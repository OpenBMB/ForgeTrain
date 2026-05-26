"""FA4-style CuTe DSL kernels (local tree).

Lazy-import to avoid pulling in cutlass/quack at module-discovery time.
Use ``from fa4_cute.interface_bwd_spec import ...`` directly.
"""

__all__ = [
    "run_flash_bwd_fa4_spec",
    "fa4_spec_shape_ok",
    "SPEC_B",
    "SPEC_N",
    "SPEC_HQ",
    "SPEC_HKV",
    "SPEC_D",
    "SPEC_SOFTMAX_SCALE",
]

_LAZY_ATTRS = set(__all__)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        from .interface_bwd_spec import (
            run_flash_bwd_fa4_spec,
            fa4_spec_shape_ok,
            SPEC_B,
            SPEC_N,
            SPEC_HQ,
            SPEC_HKV,
            SPEC_D,
            SPEC_SOFTMAX_SCALE,
        )
        _map = {
            "run_flash_bwd_fa4_spec": run_flash_bwd_fa4_spec,
            "fa4_spec_shape_ok": fa4_spec_shape_ok,
            "SPEC_B": SPEC_B,
            "SPEC_N": SPEC_N,
            "SPEC_HQ": SPEC_HQ,
            "SPEC_HKV": SPEC_HKV,
            "SPEC_D": SPEC_D,
            "SPEC_SOFTMAX_SCALE": SPEC_SOFTMAX_SCALE,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    from importlib.metadata import PackageNotFoundError, version
    try:
        __version__ = version("fa4")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except Exception:
    __version__ = "0.0.0"
