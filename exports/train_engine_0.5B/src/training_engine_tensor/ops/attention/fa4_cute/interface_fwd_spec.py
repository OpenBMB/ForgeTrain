# SM90-only FlashAttention forward: dense, causal, GQA 8:1, D=64
# Uses FlashAttentionForwardSm90 via a flash_attn.cute -> fa4_cute module redirect.
# Compiles on first invocation using real tensors (same pattern as interface_bwd_spec).

from __future__ import annotations

import importlib
import re
import sys
import types
from functools import lru_cache
from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Module redirect: make `flash_attn.cute` resolve to our local `fa4_cute`
# so that flash_fwd_sm90.py / flash_fwd.py absolute imports work.
# ---------------------------------------------------------------------------

def _setup_flash_attn_redirect():
    """Register flash_attn.cute -> fa4_cute in sys.modules (idempotent).

    Only sets up the top-level redirect and pre-registers known submodules
    needed by the SM90 forward kernel. Submodules are imported lazily on
    demand to avoid circular-import issues.
    """
    if "flash_attn.cute" in sys.modules:
        return
    from training_engine_tensor.ops.attention import fa4_cute as _local

    fa_pkg = sys.modules.get("flash_attn")
    if fa_pkg is None:
        fa_pkg = types.ModuleType("flash_attn")
        fa_pkg.__path__ = []
        sys.modules["flash_attn"] = fa_pkg

    fa_pkg.cute = _local
    sys.modules["flash_attn.cute"] = _local

    needed = [
        "utils", "fa_logging", "cute_dsl_utils", "cache_utils", "testing",
        "mask", "softmax", "seqlen_info", "block_info", "block_sparsity",
        "block_sparse_utils", "pipeline", "pack_gqa", "paged_kv",
        "named_barrier", "tile_scheduler", "flash_fwd",
        "ampere_helpers", "copy_utils", "fast_math", "barrier",
    ]
    for sub_name in needed:
        fqn = f"flash_attn.cute.{sub_name}"
        if fqn in sys.modules:
            continue
        try:
            mod = importlib.import_module(f"fa4_cute.{sub_name}")
            sys.modules[fqn] = mod
            setattr(_local, sub_name, mod)
        except ImportError:
            pass

    fqn_fwd90 = "flash_attn.cute.flash_fwd_sm90"
    if fqn_fwd90 not in sys.modules:
        try:
            mod = importlib.import_module("fa4_cute.flash_fwd_sm90")
            sys.modules[fqn_fwd90] = mod
            setattr(_local, "flash_fwd_sm90", mod)
        except ImportError:
            pass


_setup_flash_attn_redirect()

# Now safe to import SM90 forward (its flash_attn.cute.* imports will resolve).
import cutlass
import cutlass.cute as cute
from cutlass import Float32

from .cache_utils import get_jit_cache
from .cute_dsl_utils import to_cute_tensor
from .flash_fwd_sm90 import FlashAttentionForwardSm90
from .testing import is_fake_mode

from training_engine_tensor.op_dispatcher import get_frozen_env as _get_frozen_env

# ---------------------------------------------------------------------------
# Fixed spec
# ---------------------------------------------------------------------------
SPEC_D = 64
SPEC_QHEAD_PER_KV = 8  # 16 Q-heads / 2 KV-heads

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def maybe_contiguous(t):
    return t.contiguous() if t is not None and t.stride(-1) != 1 else t


@lru_cache(maxsize=None)
def _get_device_arch() -> int:
    arch_override = _get_frozen_env().get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        m = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_override)
        if not m:
            raise ValueError(f"Invalid arch: {arch_override}")
        return int(m.group(1)) * 10 + int(m.group(2))
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


# ---------------------------------------------------------------------------
# Public API — compile on first invocation using real tensors
# ---------------------------------------------------------------------------

def _flash_attn_fwd_sm90_spec(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM90 forward for dense causal GQA attention.

    Args:
        q: (B, S, Hq, D)  k: (B, S, Hkv, D)  v: (B, S, Hkv, D)  — BHSD layout
    Returns:
        (out, lse)  out: (B, S, Hq, D)  lse: (B, Hq, S)
    """
    arch = _get_device_arch()
    assert arch // 10 == 9, f"fa4 fwd spec: SM90 only, got arch={arch}"

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]

    batch_size, seqlen_q = q.shape[:2]
    num_head = q.shape[2]
    head_dim = q.shape[3]
    num_head_kv = k.shape[2]
    head_dim_v = v.shape[3]

    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype
    assert num_head % num_head_kv == 0

    if out is None:
        out = torch.empty(
            batch_size, seqlen_q, num_head, head_dim_v,
            dtype=q.dtype, device=q.device,
        )
    lse = torch.empty(
        batch_size, num_head, seqlen_q,
        dtype=torch.float32, device=q.device,
    )

    dtype = torch2cute_dtype_map[q.dtype]
    qhead_per_kvhead = num_head // num_head_kv
    pack_gqa = qhead_per_kvhead > 1
    compile_key = (dtype,)

    if compile_key not in _flash_attn_fwd_sm90_spec.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)

        tile_m, tile_n = 192, 128
        fa_fwd = FlashAttentionForwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=True,
            is_local=False,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=2,
            num_threads=384,
            Q_in_regs=False,
            intra_wg_overlap=True,
            mma_pv_is_rs=True,
            mask_mod=None,
            score_mod=None,
            has_aux_tensors=False,
            q_subtile_factor=None,
            paged_kv_non_tma=False,
        )

        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _flash_attn_fwd_sm90_spec.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
            softmax_scale,
            None, None,     # cu_seqlens_q, cu_seqlens_k
            None, None,     # seqused_q, seqused_k
            None,           # page_table
            None, None,     # window_size_left, window_size_right
            None,           # learnable_sink
            None,           # block_sparse_tensors
            None,           # aux_tensors
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        _flash_attn_fwd_sm90_spec.compile_cache[compile_key](
            q.detach(), k.detach(), v.detach(),
            out.detach(), lse,
            softmax_scale,
            None, None,     # cu_seqlens_q, cu_seqlens_k
            None, None,     # seqused_q, seqused_k
            None,           # page_table
            None, None,     # window_size_left, window_size_right
            None,           # learnable_sink
            None,           # block_sparse_tensors
            None,           # aux_tensors
        )

    return out, lse


_flash_attn_fwd_sm90_spec.compile_cache = get_jit_cache("fwd_sm90_spec")
