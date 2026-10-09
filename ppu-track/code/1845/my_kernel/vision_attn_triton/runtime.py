"""Triton non-causal vision MHA (prefill-style) for Qwen3-VL-2B single-image path."""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
import triton

from my_kernel.vision_attn_triton.kernel import vision_mha_fwd_kernel

_EXPECT_H = 16
_EXPECT_D = 64


def _enabled() -> bool:
    # Off by default: small bf16 drift vs cuDNN SDPA can change rare argmax paths.
    return os.getenv("AICAS_VISION_ATTN_TRITON", "0").strip().lower() in ("1", "true", "yes")


def vision_mha_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor | None:
    """q,k,v: [1, H, S, D] bf16. Returns [1,H,S,D] or None to fall back."""
    if not _enabled():
        return None
    if not (q.is_cuda and q.dtype == torch.bfloat16):
        return None
    if q.dim() != 4 or q.shape[0] != 1:
        return None
    if q.shape[1] != _EXPECT_H or q.shape[3] != _EXPECT_D:
        return None
    if q.shape != k.shape or q.shape != v.shape:
        return None
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        return None
    exp = 1.0 / math.sqrt(float(_EXPECT_D))
    if abs(float(scale) - exp) > 1e-4:
        return None

    b, h, s, d = q.shape
    o = torch.empty_like(q)
    TM, TN = 32, 32
    grid = (h, triton.cdiv(s, TM))
    vision_mha_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        float(scale),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        s,
        TM=TM,
        TN=TN,
        D_HEAD=d,
    )
    return o
