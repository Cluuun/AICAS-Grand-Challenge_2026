"""CUDA 算子延迟加载与 kernel 包装函数。"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# 延迟加载的cuda算子
_cuda_ops = None
_gemv_ops = None
_fused_gemv_ops = None
_flash_decode_ops = None
_int8_async_ops = None


def _get_gemv_ops():
    # 延迟加载GEMV核心
    global _gemv_ops
    if os.environ.get("AICAS_DISABLE_GEMV", "0") == "1":
        _gemv_ops = False
        return None
    if _gemv_ops is not None:
        return _gemv_ops if _gemv_ops is not False else None
    try:
        from custom_kernels.cuda.gemv_loader import get_gemv_ops
        _gemv_ops = get_gemv_ops()
        if _gemv_ops is not None:
            logger.info("[patch_model] GEMV CUDA ops loaded successfully")
        else:
            _gemv_ops = False
    except Exception as e:
        logger.warning(f"[patch_model] GEMV CUDA ops unavailable: {e}")
        _gemv_ops = False
    return _gemv_ops if _gemv_ops is not False else None


def _get_cuda_ops():
    # 延迟加载cuda算子
    global _cuda_ops
    if _cuda_ops is not None:
        return _cuda_ops if _cuda_ops is not False else None
    try:
        from custom_kernels.cuda import get_cuda_ops
        _cuda_ops = get_cuda_ops()
        if _cuda_ops is not None:
            logger.info("[patch_model] CUDA ops loaded successfully")
        else:
            _cuda_ops = False
    except Exception as e:
        logger.warning(f"[patch_model] CUDA ops unavailable: {e}")
        _cuda_ops = False
    return _cuda_ops if _cuda_ops is not False else None


def _get_fused_gemv_ops():
    global _fused_gemv_ops
    if _fused_gemv_ops is not None:
        return _fused_gemv_ops if _fused_gemv_ops is not False else None
    try:
        from custom_kernels.cuda.fused_gemv_loader import get_fused_gemv_ops
        _fused_gemv_ops = get_fused_gemv_ops()
        if _fused_gemv_ops is not None:
            logger.info("[patch_model] Fused GEMV ops loaded successfully")
        else:
            _fused_gemv_ops = False
    except Exception as e:
        logger.warning(f"[patch_model] Fused GEMV ops unavailable: {e}")
        _fused_gemv_ops = False
    return _fused_gemv_ops if _fused_gemv_ops is not False else None


def _get_flash_decode_ops():
    global _flash_decode_ops
    if _flash_decode_ops is not None:
        return _flash_decode_ops if _flash_decode_ops is not False else None
    try:
        from custom_kernels.cuda.flash_decode_attn_loader import get_flash_decode_ops
        _flash_decode_ops = get_flash_decode_ops()
        if _flash_decode_ops is not None:
            logger.info("[patch_model] Flash-decode ops loaded successfully")
        else:
            _flash_decode_ops = False
    except Exception as e:
        logger.warning(f"[patch_model] Flash-decode ops unavailable: {e}")
        _flash_decode_ops = False
    return _flash_decode_ops if _flash_decode_ops is not False else None


def _get_int8_async_ops():
    global _int8_async_ops
    if _int8_async_ops is not None:
        return _int8_async_ops if _int8_async_ops is not False else None
    if os.environ.get("AICAS_INT8_ASYNC", "0") != "1":
        _int8_async_ops = False
        return None
    try:
        from custom_kernels.cuda.int8_async_loader import get_int8_async_ops
        _int8_async_ops = get_int8_async_ops()
        if _int8_async_ops is not None:
            logger.info("[patch_model] INT8 async ops loaded successfully")
        else:
            _int8_async_ops = False
    except Exception as e:
        logger.warning(f"[patch_model] INT8 async ops unavailable: {e}")
        _int8_async_ops = False
    return _int8_async_ops if _int8_async_ops is not False else None


def _int8_gemv(gemv_ops, op_name, *args, **kwargs):
    """Route INT8 GEMV calls to async ops when AICAS_INT8_ASYNC=1."""
    async_ops = _get_int8_async_ops()
    if async_ops is not None:
        async_name = op_name + "_async"
        if hasattr(async_ops, async_name):
            return getattr(async_ops, async_name)(*args, **kwargs)
    return getattr(gemv_ops, op_name)(*args, **kwargs)


def _rms_norm_cuda(cuda_ops, hidden_states, weight, eps):
    out = torch.empty_like(hidden_states)
    cuda_ops.rms_norm(out, hidden_states, weight, eps)
    return out


def _silu_and_mul_cuda(cuda_ops, gate_up):
    d = gate_up.shape[-1] // 2
    out = torch.empty(
        gate_up.shape[:-1] + (d,),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    cuda_ops.silu_and_mul(out, gate_up)
    return out


def _fused_add_rms_norm_cuda(cuda_ops, hidden_states, residual, weight, eps):
    hidden_states = hidden_states.contiguous()
    residual = residual.contiguous()
    cuda_ops.fused_add_rms_norm(hidden_states, residual, weight, eps)
    return hidden_states, residual
