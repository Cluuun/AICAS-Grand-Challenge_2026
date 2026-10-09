from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _w8a16_gemv_dot_kernel(
        x,
        weight_t,
        scales,
        bias,
        out,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m_block = tl.program_id(0)
        n_block = tl.program_id(1)
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for start_k in range(0, k, BLOCK_K):
            k_idx = start_k + offs_k
            x_vals = tl.load(
                x + offs_m[:, None] * k + k_idx[None, :],
                mask=(offs_m[:, None] < m) & (k_idx[None, :] < k),
                other=0.0,
            )
            w_vals = tl.load(
                weight_t + k_idx[:, None] * n + offs_n[None, :],
                mask=(k_idx[:, None] < k) & (offs_n[None, :] < n),
                other=0,
            ).to(tl.float16)
            acc += tl.dot(x_vals, w_vals)

        scale_vals = tl.load(scales + offs_n, mask=offs_n < n, other=0.0).to(tl.float32)
        acc *= scale_vals[None, :]
        if has_bias:
            bias_vals = tl.load(bias + offs_n, mask=offs_n < n, other=0.0).to(tl.float32)
            acc += bias_vals[None, :]
        tl.store(
            out + offs_m[:, None] * n + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )


def triton_w8a16_gemv(
    x2d: torch.Tensor,
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if not _TRITON_AVAILABLE or not x2d.is_cuda:
        return None
    if x2d.dim() != 2 or qweight_t.dim() != 2:
        return None
    m = int(x2d.shape[0])
    k = int(x2d.shape[1])
    if int(qweight_t.shape[0]) != k:
        return None
    n = int(qweight_t.shape[1])
    if n <= 0 or k <= 0:
        return None

    out = torch.empty((m, n), dtype=x2d.dtype, device=x2d.device)
    # tl.dot requires M/N/K tile dimensions >= 16.  Decode usually has M=1,
    # so we compute a padded 16-row tile and only store valid rows.
    block_m = 16
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w8a16_gemv_dot_kernel[grid](
        x2d,
        qweight_t,
        scales,
        bias if bias is not None else scales,
        out,
        m,
        n,
        k,
        bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out
