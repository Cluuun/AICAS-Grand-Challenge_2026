"""INT8 Tensor Core prefill using torch._int_mm (CUTLASS internally).

For prefill (M>1), INT8 Tensor Core on sm_80 has 2x FLOPs vs FP16.
Weights are pre-quantized (per-channel INT8). Activation quantization
is per-tensor for maximum speed (just amax + scale + cast).
"""

import torch
import logging

logger = logging.getLogger(__name__)


def int8_linear_prefill(x, weight_int8, weight_scale, bias=None):
    """INT8 Tensor Core Linear for prefill.

    x: [M, K] or [B, M, K] fp16
    weight_int8: [N, K] int8 (pre-quantized)
    weight_scale: [N] fp32 (per-channel scale)
    bias: [N] fp16 or None

    Returns: [M, K] or [B, M, N] fp16
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight_int8.shape[0]

    # Fast per-tensor activation quantization
    x_scale = x_2d.abs().amax() / 127.0
    x_scale = x_scale.clamp(min=1e-8)
    x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

    # INT8 Tensor Core GEMM via CUTLASS
    result_int32 = torch._int_mm(x_int8, weight_int8.t())

    # Dequantize: result = int32 * x_scale * weight_scale
    result = result_int32.float() * x_scale * weight_scale.unsqueeze(0)

    if bias is not None:
        result = result + bias.float()

    return result.to(x.dtype).reshape(*orig_shape[:-1], N)


def int8_linear_prefill_gate_up(x, weight_int8, weight_scale):
    """INT8 Linear for gate+up fused projection (no bias)."""
    return int8_linear_prefill(x, weight_int8, weight_scale)


def int8_linear_prefill_down(x, weight_int8, weight_scale):
    """INT8 Linear for down projection (no bias)."""
    return int8_linear_prefill(x, weight_int8, weight_scale)
