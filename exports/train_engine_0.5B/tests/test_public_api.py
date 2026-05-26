"""Verify that each module's __all__ contains only externally-consumed symbols.

This is the TDD anchor for the "minimal public interface" principle:
every name in __all__ must be imported by at least one sibling module
or by __main__.py. Internal-only helpers should NOT appear in __all__.

For modules that require GPU-only dependencies (transformer_engine, torch.cuda),
we parse __all__ via AST instead of importing the module.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "training_engine_tensor"


def _parse_all_from_file(filepath: Path) -> set[str]:
    """Extract __all__ from a .py file via AST (no import needed)."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return {
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    return set()


# ── Modules importable without GPU deps (runtime check) ──────────────


def test_nccl_all_minimal():
    from training_engine_tensor import nccl

    expected = {
        "BucketedGradReducer",
        "NcclStaticBuffers",
        "allgather_grads",
        "allgather_grads_persistent",
        "barrier",
        "baseline_buffer_layout",
        "compute_distributed_grad_norm",
        "compute_distributed_grad_norm_tensor",
        "get_comm_stream",
        "get_rank",
        "get_world_size",
        "init_distributed",
        "is_distributed",
        "reduce_scatter_grads",
        "reduce_scatter_grads_persistent",
    }
    assert set(nccl.__all__) == expected


def test_parameters_all_minimal():
    from training_engine_tensor import parameters

    expected = {
        "ParamSpec",
        "all_param_specs",
        "load_megatron_checkpoint",
        "load_megatron_optimizer_state",
        "load_resume_checkpoint",
        "mtp_layer_prefix",
        "save_checkpoint",
        "should_save_after_step",
        "trainable_param_names",
        "wait_for_async_save",
    }
    assert set(parameters.__all__) == expected


def test_profiling_all_minimal():
    from training_engine_tensor import profiling

    expected = {
        "DEEP_PHASE_ACCUM",
        "DEEP_PHASE_BWD",
        "DEEP_PHASE_CE",
        "DEEP_PHASE_DATA",
        "DEEP_PHASE_FWD",
        "DEEP_PHASE_ORDER",
        "SEG_ALLGATHER",
        "SEG_GRAD_NORM",
        "SEG_LOSS_ALLREDUCE",
        "SEG_MICROBATCHES",
        "SEG_OPTIMIZER",
        "SEG_REDUCE_SCATTER",
        "SEG_STEP_START",
        "StepProfiler",
        "from_config",
        "summarize_segments",
    }
    assert set(profiling.__all__) == expected


def test_optimizer_all_minimal():
    from training_engine_tensor import optimizer

    expected = {
        "AdamState",
        "OptimizerScalarBuffers",
        "ShardedOptimizerBuffers",
        "adam_step",
        "clip_gradients_fp32",
        "compute_clip_coeff_device",
        "compute_grad_norm_fp32",
        "compute_lr",
        "fused_clip_adam_sync",
        "fused_clip_adam_sync_bucketed",
        "fused_clip_adam_sync_tensor",
        "sharded_fused_clip_adam_sync",
        "sharded_fused_clip_adam_sync_multi_tensor",
        "sync_params_from_master",
    }
    assert set(optimizer.__all__) == expected


def test_engine_config_all_minimal():
    from training_engine_tensor import engine_config

    expected = {
        "ENV_WHITELIST",
        "EngineConfig",
        "from_env",
        "get_config",
        "register_reset_hook",
        "set_global_config",
    }
    assert set(engine_config.__all__) == expected


def test_op_dispatcher_all_minimal():
    from training_engine_tensor import op_dispatcher

    expected = {"OP_ENV_REGISTRY", "OpEntry", "get_op_version", "init", "list_ops"}
    assert set(op_dispatcher.__all__) == expected


def test_data_all_minimal():
    from training_engine_tensor import data

    expected = {"DataPrefetcher"}
    assert set(data.__all__) == expected


def test_cuda_graph_utils_all_minimal():
    from training_engine_tensor import cuda_graph_utils

    expected = {"restore_state", "snapshot_state"}
    assert set(cuda_graph_utils.__all__) == expected


# ── Modules requiring GPU deps (AST-based check) ─────────────────────


def test_train_loop_all_via_ast():
    expected = {"run_training_loop"}
    assert _parse_all_from_file(_SRC / "train_loop.py") == expected


def test_forward_all_via_ast():
    expected = {"forward_pass_with_save", "mtp_forward_pass_with_save"}
    assert _parse_all_from_file(_SRC / "forward.py") == expected


def test_backward_all_via_ast():
    expected = {"backward_pass", "cross_entropy_loss_backward", "mtp_backward_pass"}
    assert _parse_all_from_file(_SRC / "backward.py") == expected


def test_kernels_all_via_ast():
    expected = {
        "apply_rotary_embeddings_te",
        "attention_backward_te",
        "attention_forward_te",
        "clear_te_attn_cache",
        "combine_qkv_interleaved",
        "linear",
        "linear_backward",
        "precompute_rope_freqs",
        "rmsnorm_te",
        "rmsnorm_te_backward",
        "rmsnorm_te_with_rsigma",
        "rope_backward_te",
        "split_qkv_interleaved",
        "swiglu",
        "swiglu_back",
    }
    assert _parse_all_from_file(_SRC / "kernels.py") == expected


def test_triton_kernels_all_via_ast():
    expected = {
        "fused_adam_sync",
        "fused_adam_sync_tensor",
        "fused_cross_entropy_fwd_bwd",
        "fused_residual_rmsnorm_fwd",
        "fused_rmsnorm_bwd_residual",
        "fused_rope_bwd",
        "fused_rope_bwd_doc_aware",
        "fused_rope_fwd",
        "fused_rope_fwd_doc_aware",
        "fused_swiglu_bwd",
        "fused_swiglu_fwd",
    }
    assert _parse_all_from_file(_SRC / "triton_kernels.py") == expected


def test_custom_gemm_all_via_ast():
    expected = {
        "custom_gemm_attn_out_proj_bwd",
        "custom_gemm_attn_out_proj_fwd",
        "custom_gemm_fc1_bwd",
        "custom_gemm_fc1_fwd",
        "custom_gemm_fc2_bwd",
        "custom_gemm_fc2_fwd",
        "custom_gemm_output_bwd",
        "custom_gemm_output_fwd",
        "custom_gemm_qkv_proj_bwd",
        "custom_gemm_qkv_proj_fwd",
    }
    assert _parse_all_from_file(_SRC / "custom_gemm.py") == expected


def test_step_graph_all_via_ast():
    expected = {"graphed_compute_step", "is_step_cuda_graph_enabled"}
    assert _parse_all_from_file(_SRC / "step_graph.py") == expected


def test_forward_graph_all_via_ast():
    expected = {"graphed_forward_pass_with_save", "is_forward_cuda_graph_enabled"}
    assert _parse_all_from_file(_SRC / "forward_graph.py") == expected


def test_step_graph_full_all_via_ast():
    expected = {"graphed_full_step", "is_step_cuda_graph_full_enabled"}
    assert _parse_all_from_file(_SRC / "step_graph_full.py") == expected


def test_step_graph_nccl_opt_all_via_ast():
    expected = {"graphed_nccl_opt_step", "is_nccl_opt_graph_enabled"}
    assert _parse_all_from_file(_SRC / "step_graph_nccl_opt.py") == expected


def test_step_graph_optimizer_all_via_ast():
    expected = {"graphed_optimizer_step", "is_step_cuda_graph_optimizer_enabled"}
    assert _parse_all_from_file(_SRC / "step_graph_optimizer.py") == expected
