"""Single-host MiniCPM4-8B pretraining engine.

Three self-developed kernels live in this package:

* ``training_engine_tensor.ops.gemm_fc1`` — fused SwiGLU column-parallel GEMM.
* ``training_engine_tensor.ops.gemm_output`` — column-parallel LM-head GEMM.
* ``flash_attn_dsl`` — Hopper SM90a flash-attention forward, written from
  scratch in NVIDIA CuTe Python DSL (sibling top-level package; invoked via
  :mod:`training_engine_tensor.kernels`).

Every other GEMM site (qkv projection, attention output projection, FFN
fc2) runs the baseline ``torch.matmul`` path unconditionally.
"""

__all__: list[str] = []
