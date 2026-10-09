"""INT8 Tensor Core GEMM wrapper for prefill (using torch._int_mm = CUTLASS INT8).

Prefill是计算密集型, INT8 Tensor Core在sm_80上有2x FLOPs优势。
加上权重量化还能省带宽。
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# Per-channel INT8 quantize helper (CUDA)
def _quantize_per_channel_cuda(weight):
    """weight: [N, K] fp16 -> (int8, fp32_scale)"""
    scale = weight.abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-8)
    w_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.float()


def _quantize_tensor_per_row(x):
    """Dynamic per-tensor quantization for activation: x[M,K] -> (int8, scale)"""
    scale = x.abs().amax() / 127.0
    scale = scale.clamp(min=1e-8)
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale.float()


def int8_linear_prefill(x, weight_int8, weight_scale, bias=None):
    """INT8 Tensor Core Linear for prefill: y = x @ (W_int8 * scale).T"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight_int8.shape[0]
    if M <= 16:
        raise RuntimeError("int8 prefill requires M > 16")

    # Dynamic activation quantization to INT8
    x_scale = x_2d.abs().amax() / 127.0
    x_scale = x_scale.clamp(min=1e-8)
    x_int8 = (x_2d / x_scale).round().clamp(-128, 127).to(torch.int8)

    # INT8 Tensor Core GEMM via CUTLASS: result[M, N] = x_int8 @ weight_int8.T
    result_int32 = torch._int_mm(x_int8, weight_int8.t())

    # Dequantize
    result = result_int32.float() * x_scale * weight_scale.unsqueeze(0)

    if bias is not None:
        result = result + bias.float()

    return result.to(x.dtype).reshape(*orig_shape[:-1], N)


def int8_linear_prefill_gate_up(x, weight_int8, weight_scale, gate_size, up_size):
    """INT8 Linear for fused gate+up projection.

    Returns gate_up output [M, gate_size+up_size] fp16.
    """
    return int8_linear_prefill(x, weight_int8, weight_scale)


def prefill_quantize_weights(model):
    """Pre-quantize all LM layer weights for INT8 prefill.
    Called after weight fusion (gate_up, qkv).
    """
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3VLTextDecoderLayer":
            # QKV
            if hasattr(module.self_attn, 'qkv_proj'):
                if not hasattr(module.self_attn.qkv_proj, 'weight_int8'):
                    w_int8, scale = _quantize_per_channel_cuda(module.self_attn.qkv_proj.weight.data)
                    module.self_attn.qkv_proj.weight_int8_prefill = w_int8
                    module.self_attn.qkv_proj.weight_scale_prefill = scale
                count += 1

            # O projection
            if hasattr(module.self_attn, 'o_proj'):
                if not hasattr(module.self_attn.o_proj, 'weight_int8'):
                    w_int8, scale = _quantize_per_channel_cuda(module.self_attn.o_proj.weight.data)
                    module.self_attn.o_proj.weight_int8_prefill = w_int8
                    module.self_attn.o_proj.weight_scale_prefill = scale
                count += 1

            # Gate+Up projection
            if hasattr(module.mlp, 'gate_up_proj'):
                if not hasattr(module.mlp.gate_up_proj, 'weight_int8'):
                    w_int8, scale = _quantize_per_channel_cuda(module.mlp.gate_up_proj.weight.data)
                    module.mlp.gate_up_proj.weight_int8_prefill = w_int8
                    module.mlp.gate_up_proj.weight_scale_prefill = scale
                count += 1

            # Down projection
            if hasattr(module.mlp, 'down_proj'):
                if not hasattr(module.mlp.down_proj, 'weight_int8'):
                    w_int8, scale = _quantize_per_channel_cuda(module.mlp.down_proj.weight.data)
                    module.mlp.down_proj.weight_int8_prefill = w_int8
                    module.mlp.down_proj.weight_scale_prefill = scale
                count += 1

    logger.info(f"[prefill_int8] {count} weights quantized for INT8 Tensor Core prefill")
    print(f"[prefill_int8] {count} weights quantized for INT8 Tensor Core prefill")
    return count
