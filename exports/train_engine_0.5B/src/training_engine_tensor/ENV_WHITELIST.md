# Environment Variable Whitelist

The following environment variables are **process-external contracts** —
they are injected by the container runtime (cctl, torchrun, etc.) and
are the ONLY `os.environ` reads permitted inside
`src/training_engine_tensor/`.

All other behavioral configuration MUST flow through `EngineConfig`
(constructed once at CLI entry-point time and passed as a parameter).

## Whitelisted Variables

### Process-External Contracts


| Variable                           | Source            | Purpose                        |
| ---------------------------------- | ----------------- | ------------------------------ |
| `RANK`                             | torchrun / cctl   | Global process rank            |
| `LOCAL_RANK`                       | torchrun / cctl   | Local (per-node) GPU rank      |
| `WORLD_SIZE`                       | torchrun / cctl   | Total number of processes      |
| `MASTER_ADDR`                      | torchrun / cctl   | Rendezvous address             |
| `MASTER_PORT`                      | torchrun / cctl   | Rendezvous port                |
| `NCCL_UNIQUE_ID_FILE`              | cctl              | Shared file for NCCL comm init |
| `CUDA_DEVICE_MAX_CONNECTIONS`      | User / cctl       | CUDA MPS connection limit      |
| `NVTE_FUSED_ATTN`                  | TransformerEngine | TE fused attention backend     |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | TransformerEngine | TE nondeterminism              |
| `PYTORCH_CUDA_ALLOC_CONF`          | PyTorch           | CUDA allocator config          |
| `CUBLAS_WORKSPACE_CONFIG`          | cuBLAS            | Workspace determinism          |
| `TORCH_NCCL_USE_COMM_NONBLOCKING`  | PyTorch           | NCCL async comm init           |
| `MEGATRON_ROOT`                    | Harness           | Path to Megatron-LM checkout   |
| `PYTHONPATH`                       | System            | Python module search path      |


### Observability Contracts


| Variable         | Source         | Purpose                                           |
| ---------------- | -------------- | ------------------------------------------------- |
| `PROFILE_RANGE`  | Harness / cctl | Step range for CUDA-event profiling (`START,END`) |
| `PROFILE_OUTPUT` | Harness / cctl | Output path for profile JSON                      |
| `PROFILE_DEEP`   | Harness / cctl | Enable per-microbatch sub-segment profiling       |
| `HOST_TIMER`     | Harness / cctl | Enable host-side wall-clock timing                |


### Operator Version Contracts


| Variable                | Source        | Purpose                                             |
| ----------------------- | ------------- | --------------------------------------------------- |
| `OP_GEMM_FC1`           | cctl job spec | Operator version for gemm_fc1 (`baseline`/`v1`/...) |
| `OP_GEMM_FC2`           | cctl job spec | Operator version for gemm_fc2                       |
| `OP_GEMM_QKV_PROJ`      | cctl job spec | Operator version for gemm_qkv_proj                  |
| `OP_GEMM_ATTN_OUT_PROJ` | cctl job spec | Operator version for gemm_attn_out_proj             |
| `OP_GEMM_OUTPUT`        | cctl job spec | Operator version for gemm_output                    |
| `OP_ATTENTION`          | cctl job spec | Operator version for attention                      |


### Legacy Migration Bridge (from_env → EngineConfig)

> **Status: DEPRECATED.** These env vars are read ONCE by `from_env()` at
> process startup and converted into `EngineConfig` fields. They exist for
> backward compatibility with cctl job specs that pass configuration via
> `envVars`. New deployments should migrate to CLI arguments. Once all
> entry scripts are migrated, `from_env()` and this section will be removed.


| Variable                           | EngineConfig field                 | Default             |
| ---------------------------------- | ---------------------------------- | ------------------- |
| `FUSED_OPS`                        | `fused_ops`                        | `0`                 |
| `FUSE_CE`                          | `fuse_ce`                          | follows `FUSED_OPS` |
| `FUSE_SWIGLU`                      | `fuse_swiglu`                      | follows `FUSED_OPS` |
| `FUSE_RMSNORM_RESIDUAL`            | `fuse_rmsnorm_residual`            | follows `FUSED_OPS` |
| `FUSE_ROPE`                        | `fuse_rope`                        | follows `FUSED_OPS` |
| `FUSE_ROPE_DOC_AWARE`              | `fuse_rope_doc_aware`              | follows `FUSE_ROPE` |
| `FUSE_ADAM_SYNC`                   | `fuse_adam_sync`                   | follows `FUSED_OPS` |
| `FUSE_GEMM_PAD`                    | `fuse_gemm_pad`                    | follows `FUSED_OPS` |
| `FUSE_DIRECT_ATTN`                 | `fuse_direct_attn`                 | `0`                 |
| `FUSE_INPLACE_RESIDUAL`            | `fuse_inplace_residual`            | follows `FUSED_OPS` |
| `FUSE_INDEX_ADD_EMB_BWD`           | `fuse_index_add_emb_bwd`           | follows `FUSED_OPS` |
| `USE_SPAN_BASED_ATTN`              | `use_span_based_attn`              | `0`                 |
| `FP32_MODE`                        | `fp32_mode`                        | `0`                 |
| `UNFUSED_MODE`                     | `unfused_mode`                     | `0`                 |
| `TE_ROPE`                          | `te_rope`                          | `0`                 |
| `FP32_ALLREDUCE`                   | `fp32_allreduce`                   | `0`                 |
| `FP32_GRAD_ACCUM`                  | `fp32_grad_accum`                  | `0`                 |
| `NO_GRAD_CLIP`                     | `no_grad_clip`                     | `0`                 |
| `GRAD_DIAG`                        | `grad_diag`                        | `0`                 |
| `WGRAD_OVERLAP`                    | `wgrad_overlap`                    | `1`                 |
| `WGRAD_BUCKET_TARGET_ELEMS`        | `wgrad_bucket_target_elems`        | `104857600`         |
| `OPT_ALLGATHER_OVERLAP`            | `opt_allgather_overlap`            | `0`                 |
| `ASYNC_LOSS_AR`                    | `async_loss_ar`                    | `1`                 |
| `SHARDED_OPTIMIZER`                | `sharded_optimizer`                | `1`                 |
| `MULTI_TENSOR_ADAM`                | `multi_tensor_adam`                | `0`                 |
| `CUSTOM_GEMM`                      | `custom_gemm`                      | `0`                 |
| `FORWARD_CUDA_GRAPH`               | `forward_cuda_graph`               | `0`                 |
| `FORWARD_CUDA_GRAPH_WARMUP`        | `forward_cuda_graph_warmup`        | `3`                 |
| `FORWARD_CUDA_GRAPH_TOL`           | `forward_cuda_graph_tol`           | `1e-3`              |
| `FORWARD_CUDA_GRAPH_SANITY`        | `forward_cuda_graph_sanity`        | `0`                 |
| `FORWARD_CUDA_GRAPH_SAMPLE_STRIDE` | `forward_cuda_graph_sample_stride` | `1024`              |
| `STEP_CUDA_GRAPH`                  | `step_cuda_graph`                  | `0`                 |
| `STEP_CUDA_GRAPH_WARMUP`           | `step_cuda_graph_warmup`           | `3`                 |
| `STEP_CUDA_GRAPH_TOL`              | `step_cuda_graph_tol`              | `1e-3`              |
| `STEP_CUDA_GRAPH_SANITY`           | `step_cuda_graph_sanity`           | `0`                 |
| `STEP_CUDA_GRAPH_FULL`             | `step_cuda_graph_full`             | `0`                 |
| `STEP_CUDA_GRAPH_FULL_WARMUP`      | `step_cuda_graph_full_warmup`      | `3`                 |
| `STEP_CUDA_GRAPH_FULL_TOL`         | `step_cuda_graph_full_tol`         | `1e-3`              |
| `STEP_CUDA_GRAPH_FULL_SANITY`      | `step_cuda_graph_full_sanity`      | `0`                 |
| `STEP_CUDA_GRAPH_NCCL_OPT`         | `step_cuda_graph_nccl_opt`         | `0`                 |
| `STEP_CUDA_GRAPH_NCCL_OPT_WARMUP`  | `step_cuda_graph_nccl_opt_warmup`  | `3`                 |
| `STEP_CUDA_GRAPH_OPTIMIZER`        | `step_cuda_graph_optimizer`        | `0`                 |
| `STEP_CUDA_GRAPH_OPTIMIZER_WARMUP` | `step_cuda_graph_optimizer_warmup` | `3`                 |
| `USE_FUSED_NORM_BWD`               | `use_fused_norm_bwd`               | `0`                 |
| `NORM_BWD_NAN_DIAG`                | `norm_bwd_nan_diag`                | `0`                 |
| `EMIT_LOSS_LINES`                  | `emit_loss_lines`                  | `0`                 |
| `GRAD_NORM_DIAG`                   | `grad_norm_diag`                   | `0`                 |
| `MUP_EMB_SCALE`                    | `mup_emb_scale`                    | `0.0`               |
| `MUP_DEPTH_SCALE`                  | `mup_depth_scale`                  | `0.0`               |
| `VOCAB_SIZE`                       | `vocab_size`                       | `73448`             |
| `MTP_NUM_LAYERS`                   | `mtp_num_layers`                   | `0`                 |
| `MTP_LOSS_SCALING_FACTOR`          | `mtp_loss_scaling_factor`          | `0.1`               |
| `MAX_LR`                           | `max_lr`                           | `3e-4`              |
| `MIN_LR`                           | `min_lr`                           | `0.0`               |
| `WARMUP_ITERS`                     | `warmup_iters`                     | `2000`              |
| `LR_DECAY_ITERS`                   | `lr_decay_iters`                   | `240000`            |
| `WSD_DECAY_ITERS`                  | `wsd_decay_iters`                  | `3000`              |
| `ADAM_BETA1`                       | `adam_beta1`                       | `0.9`               |
| `ADAM_BETA2`                       | `adam_beta2`                       | `0.95`              |
| `ADAM_EPS`                         | `adam_eps`                         | `1e-8`              |
| `WEIGHT_DECAY`                     | `weight_decay`                     | `0.1`               |
| `MAX_GRAD_NORM`                    | `max_grad_norm`                    | `1.0`               |
| `RANDOM_INIT`                      | `random_init`                      | `0`                 |


