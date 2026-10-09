"""Triton GEMV kernel，优化batch=1的decode场景。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    X_PTR, W_PTR, OUT_PTR,
    K, N,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMV kernel: out[n] = sum_k(x[k] * W[n, k])，batch=1专用。"""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # 加载输入向量的tile
        x = tl.load(X_PTR + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        # 加载权重tile
        w = tl.load(
            W_PTR + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        # 部分点积
        acc += tl.sum(w * x[None, :], axis=1)

    # 存输出
    tl.store(OUT_PTR + offs_n, acc, mask=mask_n)


def triton_gemv(x, weight):
    """GEMV: x[..., K] @ weight[N, K].T -> out[..., N]，decode优化版。"""
    orig_shape = x.shape
    K = weight.shape[1]
    N = weight.shape[0]

    x_2d = x.reshape(-1, K)
    batch = x_2d.shape[0]
    out = torch.empty(batch, N, dtype=x.dtype, device=x.device)

    BLOCK_N = 64
    BLOCK_K = 256
    grid = (triton.cdiv(N, BLOCK_N),)

    for b in range(batch):
        _gemv_kernel[grid](
            x_2d[b], weight, out[b],
            K=K, N=N,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8,
            num_stages=3,
        )

    return out.reshape(orig_shape[:-1] + (N,))
