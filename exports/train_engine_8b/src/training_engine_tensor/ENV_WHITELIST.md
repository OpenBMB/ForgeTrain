# Environment variable whitelist

The training engine treats `os.environ` as a hostile surface: framework
code reads configuration from `EngineConfig` (see
[engine_config.py](engine_config.py)) and not from environment
variables.  The very small set of env vars listed in `ENV_WHITELIST`
are the only legitimate exceptions, and they exist because the value
is owned by an actor outside the engine.

Categories:

| Category                         | Env var                              | Owner / consumer                                              |
| -------------------------------- | ------------------------------------ | ------------------------------------------------------------- |
| Distributed launcher             | `RANK`, `LOCAL_RANK`, `WORLD_SIZE`   | `torchrun`; consumed in `hf_dataloader.py` / `nccl.py`        |
| Distributed launcher             | `MASTER_ADDR`, `MASTER_PORT`         | `torchrun`; consumed by `torch.distributed.init_process_group` |
| CUDA / NCCL runtime              | `CUDA_DEVICE_MAX_CONNECTIONS`        | CUDA runtime                                                  |
| CUDA / NCCL runtime              | `PYTORCH_CUDA_ALLOC_CONF`            | PyTorch caching allocator                                     |
| CUDA / NCCL runtime              | `CUBLAS_WORKSPACE_CONFIG`            | cuBLAS                                                        |
| CUDA / NCCL runtime              | `NVTE_FUSED_ATTN`                    | TransformerEngine                                             |
| CUDA / NCCL runtime              | `NVTE_ALLOW_NONDETERMINISTIC_ALGO`   | TransformerEngine                                             |
| CUDA / NCCL runtime              | `TORCH_NCCL_USE_COMM_NONBLOCKING`    | NCCL                                                          |
| CuTeDSL bootstrap                | `CUTLASS_DSL_PATH`                   | `kernels.py:_ensure_dsl_imports` — optional `sys.path` hint when cutlass-dsl is not installed as a wheel |
| Module search path               | `PYTHONPATH`                         | Python startup                                                |
| Per-operator dispatch            | `OP_GEMM_FC1`, `OP_GEMM_OUTPUT`, ... | `op_dispatcher.py` — per-operator version selection (`baseline` / `v1` / ...) |

## Migration bridge: `OURS_*` and other `EngineConfig` knobs

The previous architecture read 19 `OURS_*` env vars plus a handful of
training-loop knobs at module-import time.  These have been folded
into typed `EngineConfig` fields; the corresponding env vars are still
honoured by `engine_config.from_env()` for backward compatibility with
existing job specs that pass values via `envVars`.

| Env var                            | EngineConfig field            | Default |
| ---------------------------------- | ----------------------------- | ------- |
| `OURS_FUSED_ROPE`                  | `fused_rope`                  | true    |
| `OURS_FUSED_ROPE_PACK`             | `fused_rope_pack`             | true    |
| `OURS_FUSED_RESIDUAL_RMSNORM`      | `fused_residual_rmsnorm`      | true    |
| `OURS_FUSED_ROPE_QKV`              | `fused_rope_qkv`              | true    |
| `OURS_FUSED_ATTN_RMSNORM_BWD`      | `fused_attn_rmsnorm_bwd`      | true    |
| `OURS_FUSED_MLP_RESIDUAL_RMSNORM`  | `fused_mlp_residual_rmsnorm`  | true    |
| `OURS_FUSED_DW_REDUCE`             | `fused_dw_reduce`             | true    |
| `OURS_FUSED_ROPE_FP32COS`          | `fused_rope_fp32cos`          | true    |
| `OURS_AG_FWD_OVERLAP`              | `ag_fwd_overlap`              | true    |
| `OURS_ADAM_AG_PIPELINE`            | `adam_ag_pipeline`            | true    |
| `OURS_GPU_NORM_CLIP`               | `gpu_norm_clip`               | true    |
| `OURS_FUSED_CE`                    | `fused_ce`                    | true    |
| `OURS_DEFER_WGRAD_SYNC`            | `defer_wgrad_sync`            | true    |
| `OURS_FUSED_SWIGLU`                | `fused_swiglu`                | true    |
| `OURS_DIRECT_ATTN`                 | `direct_attn`                 | true    |
| `OURS_DSL_ATTN_FWD`                | `dsl_attn_fwd`                | true    |
| `OURS_ROPE_BHSD`                   | `rope_bhsd`                   | true    |
| `OURS_RS_BUCKETS`                  | `rs_buckets`                  | 32      |
| `OURS_DW_REDUCE_RPB`               | `dw_reduce_rows_per_block`    | 32      |
| `CUSTOM_GEMM`                      | `custom_gemm`                 | false   |
| `PADDED_VOCAB_SIZE`                | `padded_vocab_size`           | 73448   |

New deployments should pass these values via `__main__` CLI flags
(`--fused-rope` / `--no-fused-rope`, `--rs-buckets 16`, etc.) rather
than environment variables.
