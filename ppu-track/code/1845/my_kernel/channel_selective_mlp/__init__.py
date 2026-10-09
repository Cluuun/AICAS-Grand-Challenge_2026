"""Channel-selective MLP operator for Qwen3-VL decoder.

A fused-MLP operator that computes ``down(silu(gate(x)) * up(x))`` on a
per-layer subset of intermediate channels selected by a calibration profile.
Operator-level optimization: reduces decode-time GEMV bandwidth by serving
only the high-importance channels through standard ``nn.Linear``-shaped
weight tensors that downstream kernels (FlashDecodeFFN, cuBLAS GEMV) consume
without modification.
"""

from .runtime import (
    apply_channel_selective_mlp,
    maybe_apply_channel_selective_mlp_from_env,
)
from .calibrate import calibrate_channel_importance

__all__ = [
    "apply_channel_selective_mlp",
    "maybe_apply_channel_selective_mlp_from_env",
    "calibrate_channel_importance",
]
