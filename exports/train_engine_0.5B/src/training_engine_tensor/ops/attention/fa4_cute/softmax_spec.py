# Copyright (c) 2025, Tri Dao. SM90 bwd spec: fixed log-domain scaling, exp2 only (no softcap).
import math
from cutlass import Float32

# Linear attention scale 1/sqrt(64); in log2 domain: log2(0.125) = -3.0
SPEC_HEAD_DIM: int = 64
LOG2_SOFTMAX_SCALE: float = -3.0
SPEC_SOFTMAX_SCALE_LINEAR: float = 2.0**LOG2_SOFTMAX_SCALE  # 0.125


def linear_scale_from_log2() -> float:
    """1/sqrt(D) for D=64 — matches `SPEC_SOFTMAX_SCALE_LINEAR`."""
    return 1.0 / math.sqrt(SPEC_HEAD_DIM)


def softmax_scale_log2_e_times_linear(softmax_scale_linear: float) -> float:
    """Matches generic bwd: logits use exp2(s * log2(e) * linear_scale) when score_mod is disabled."""
    return softmax_scale_linear * math.log2(math.e)
