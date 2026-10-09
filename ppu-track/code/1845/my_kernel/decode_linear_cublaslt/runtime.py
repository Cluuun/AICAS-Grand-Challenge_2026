from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_EXT = None
_EMPTY_BIAS_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    candidates = sorted(_THIS_DIR.glob("decode_linear_cublaslt_ext*.so"))
    if not candidates:
        raise FileNotFoundError(f"Missing decode_linear_cublaslt_ext*.so in {_THIS_DIR}")
    so_path = candidates[0]
    spec = importlib.util.spec_from_file_location("decode_linear_cublaslt_ext", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT = mod
    return mod


def is_available() -> bool:
    try:
        _load_ext()
        return True
    except Exception:
        return False


def _empty_bias(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    bias = _EMPTY_BIAS_CACHE.get(key)
    if bias is None:
        bias = torch.empty((0,), device=device, dtype=dtype)
        _EMPTY_BIAS_CACHE[key] = bias
    return bias


def can_use_decode_linear_cublaslt(
    x: torch.Tensor,
    w_t: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> bool:
    if not isinstance(x, torch.Tensor) or not isinstance(w_t, torch.Tensor):
        return False
    if not x.is_cuda or not w_t.is_cuda:
        return False
    if x.dtype is not torch.bfloat16 or w_t.dtype is not torch.bfloat16:
        return False
    if x.ndim != 2 or w_t.ndim != 2:
        return False
    if int(x.shape[1]) != int(w_t.shape[0]):
        return False
    if not x.is_contiguous() or not w_t.is_contiguous():
        return False
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or not bias.is_cuda:
            return False
        if bias.dtype is not torch.bfloat16 or bias.ndim != 1:
            return False
        if int(bias.shape[0]) != int(w_t.shape[1]) or not bias.is_contiguous():
            return False
    return x.numel() > 0 and w_t.numel() > 0


def can_use_decode_linear_mma(
    x: torch.Tensor,
    w_t: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> bool:
    if os.getenv("AICAS_DECODE_LINEAR_MMA", "0") != "1":
        return False
    if not can_use_decode_linear_cublaslt(x, w_t, bias):
        return False
    if int(x.shape[1]) % 16 != 0 or int(w_t.shape[1]) % 16 != 0:
        return False
    # This kernel is written for decode-ish small M. Let prefill use GEMM.
    return 1 <= int(x.shape[0]) <= int(os.getenv("AICAS_DECODE_LINEAR_MMA_MAX_M", "16"))


def linear_decode_cublaslt(
    x: torch.Tensor,
    w_t: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    ext = _load_ext()
    x2d = x.reshape(-1, x.shape[-1])
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    has_bias = bias is not None
    bias_arg = bias if has_bias else _empty_bias(x2d.device, x2d.dtype)
    expected_shape = (int(x2d.shape[0]), int(w_t.shape[1]))
    if (
        out is None
        or not isinstance(out, torch.Tensor)
        or out.device != x2d.device
        or out.dtype != x2d.dtype
        or tuple(out.shape) != expected_shape
        or not out.is_contiguous()
    ):
        out = torch.empty(expected_shape, device=x2d.device, dtype=x2d.dtype)
    if can_use_decode_linear_mma(x2d, w_t, bias):
        ext.linear_mma_into(x2d, w_t, bias_arg, out, bool(has_bias))
    else:
        ext.linear_into(x2d, w_t, bias_arg, out, bool(has_bias))
    return out
