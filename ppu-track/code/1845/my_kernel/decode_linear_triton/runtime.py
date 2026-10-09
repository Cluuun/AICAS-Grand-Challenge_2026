from __future__ import annotations

import math

import torch

from .kernel import gemv_row_kernel, gemv_rows_fixed_kernel


def can_use_decode_linear_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> bool:
    if not isinstance(x, torch.Tensor) or not isinstance(w, torch.Tensor):
        return False
    if not x.is_cuda or not w.is_cuda:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16) or w.dtype != x.dtype:
        return False
    if b is not None and (not isinstance(b, torch.Tensor) or b.dtype != x.dtype or not b.is_cuda):
        return False
    if x.numel() != x.shape[-1]:
        return False
    if w.ndim != 2:
        return False
    if int(w.shape[1]) != int(x.shape[-1]):
        return False
    if not x.is_contiguous() or not w.is_contiguous():
        return False
    return True


def can_use_decode_linear_triton_rows(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> bool:
    if not isinstance(x, torch.Tensor) or not isinstance(w, torch.Tensor):
        return False
    if not x.is_cuda or not w.is_cuda:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16) or w.dtype != x.dtype:
        return False
    if b is not None and (not isinstance(b, torch.Tensor) or b.dtype != x.dtype or not b.is_cuda):
        return False
    if x.ndim < 1 or w.ndim != 2:
        return False
    if int(w.shape[1]) != int(x.shape[-1]):
        return False
    if not x.is_contiguous() or not w.is_contiguous():
        return False
    if b is not None and not b.is_contiguous():
        return False
    if x.numel() == 0 or w.numel() == 0:
        return False
    return True


def linear_decode_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    x1d = x.reshape(-1).contiguous()
    n = int(w.shape[0])
    k = int(w.shape[1])

    if out is None or out.device != x.device or out.dtype != x.dtype or out.numel() != n:
        out = torch.empty((n,), device=x.device, dtype=x.dtype)

    grid = (math.ceil(n / 64),)
    gemv_row_kernel[grid](
        x1d,
        w,
        b if b is not None else x1d,
        out,
        n,
        k,
        int(w.stride(0)),
        int(w.stride(1)),
        HAS_BIAS=b is not None,
    )
    return out


def linear_decode_triton_rows_fixed(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1])
    m = int(x2d.shape[0])
    n = int(w.shape[0])
    k = int(w.shape[1])

    expected_shape = (m, n)
    if out is None or out.device != x.device or out.dtype != x.dtype or tuple(out.shape) != expected_shape:
        out = torch.empty(expected_shape, device=x.device, dtype=x.dtype)

    block_n = 64
    block_k = 64
    grid = (math.ceil(n / block_n), m)
    gemv_rows_fixed_kernel[grid](
        x2d,
        w,
        b if b is not None else x2d,
        out,
        m,
        n,
        k,
        int(x2d.stride(0)),
        int(x2d.stride(1)),
        int(w.stride(0)),
        int(w.stride(1)),
        int(out.stride(0)),
        int(out.stride(1)),
        HAS_BIAS=b is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out
