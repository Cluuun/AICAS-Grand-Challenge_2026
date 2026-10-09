"""自定义CUDA算子，注册为torch.library算子以兼容CUDA graph。"""

import torch
import logging

logger = logging.getLogger(__name__)

_ops = None


def _get_ops():
    global _ops
    if _ops is not None:
        return _ops if _ops is not False else None
    try:
        from custom_kernels.cuda import get_cuda_ops
        _ops = get_cuda_ops()
        return _ops
    except Exception as e:
        logger.warning(f"CUDA ops unavailable: {e}")
        _ops = False
        return None


# 用torch.library注册自定义算子，保证graph capture时内存追踪正确
try:
    @torch.library.custom_op("custom_kernels::rms_norm_kernel",
                             mutates_args={"out"}, device_types="cuda")
    def rms_norm_kernel(out: torch.Tensor, input: torch.Tensor,
                        weight: torch.Tensor, epsilon: float) -> None:
        ops = _get_ops()
        if ops is not None:
            ops.rms_norm(out, input, weight, epsilon)
        else:
            # 没有CUDA就算PyTorch的回退方案
            input_dtype = input.dtype
            h = input.float()
            variance = h.pow(2).mean(-1, keepdim=True)
            h = h * torch.rsqrt(variance + epsilon)
            out.copy_((weight * h.to(input_dtype)))

    @torch.library.custom_op("custom_kernels::silu_and_mul_kernel",
                             mutates_args={"out"}, device_types="cuda")
    def silu_and_mul_kernel(out: torch.Tensor, gate_up: torch.Tensor) -> None:
        ops = _get_ops()
        if ops is not None:
            ops.silu_and_mul(out, gate_up)
        else:
            d = gate_up.shape[-1] // 2
            gate, up = gate_up[..., :d], gate_up[..., d:]
            out.copy_(torch.nn.functional.silu(gate) * up)

    CUSTOM_OPS_AVAILABLE = True
    logger.info("Custom ops registered via torch.library")

except Exception as e:
    logger.warning(f"torch.library registration failed: {e}")
    CUSTOM_OPS_AVAILABLE = False


def rms_norm(input, weight, epsilon):
    """用自定义CUDA kernel做RMS归一化。"""
    out = torch.empty_like(input)
    if CUSTOM_OPS_AVAILABLE:
        rms_norm_kernel(out, input, weight, epsilon)
    else:
        input_dtype = input.dtype
        h = input.float()
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + epsilon)
        out.copy_(weight * h.to(input_dtype))
    return out


def silu_and_mul(gate_up):
    """SiLU(x[:d]) * x[d:]，用自定义CUDA kernel。"""
    d = gate_up.shape[-1] // 2
    out = torch.empty(gate_up.shape[:-1] + (d,),
                      dtype=gate_up.dtype, device=gate_up.device)
    if CUSTOM_OPS_AVAILABLE:
        silu_and_mul_kernel(out, gate_up)
    else:
        gate, up = gate_up[..., :d], gate_up[..., d:]
        out.copy_(torch.nn.functional.silu(gate) * up)
    return out
