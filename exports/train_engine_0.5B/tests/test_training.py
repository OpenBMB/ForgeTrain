"""Unit tests for MiniCPM4 0.5B training engine.

This file will be expanded as the training engine is developed.
Initial state: only config and interface tests.
"""

import importlib
import unittest


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_HAS_TORCH = _can_import("torch")
_HAS_TE = _can_import("transformer_engine")
_NEEDS_TORCH = unittest.skipUnless(_HAS_TORCH, "requires torch")
_NEEDS_GPU = unittest.skipUnless(
    _HAS_TORCH and _HAS_TE,
    "requires torch and transformer_engine (GPU environment)",
)


class TestModulesExist(unittest.TestCase):
    """Verify that all shipped engine modules are importable."""

    @_NEEDS_TORCH
    def test_import_config(self):
        import training_engine_tensor.config  # noqa: F401

    @_NEEDS_GPU
    def test_import_forward(self):
        import training_engine_tensor.forward  # noqa: F401

    @_NEEDS_GPU
    def test_import_backward(self):
        import training_engine_tensor.backward  # noqa: F401

    @_NEEDS_GPU
    def test_import_parameters(self):
        import training_engine_tensor.parameters  # noqa: F401

    @_NEEDS_GPU
    def test_import_kernels(self):
        import training_engine_tensor.kernels  # noqa: F401

    @_NEEDS_GPU
    def test_import_optimizer(self):
        import training_engine_tensor.optimizer  # noqa: F401

    @_NEEDS_GPU
    def test_import_nccl(self):
        import training_engine_tensor.nccl  # noqa: F401


if __name__ == "__main__":
    unittest.main()
