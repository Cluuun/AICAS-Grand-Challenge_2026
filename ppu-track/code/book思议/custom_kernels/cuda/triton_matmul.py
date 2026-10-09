"""Triton GEMM kernel，专门优化单token decode场景(batch=1, seq=1)。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_gemv_kernel(
    X_PTR, W_PTR, OUT_PTR,
    N: tl.constexpr,  # 输入维度
    M: tl.constexpr,  # 输出维度
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """单行GeMV kernel: out = x @ W.T"""
    pid = tl.program_id(0)

    # 输出行tile
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # 输入向量指针
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # 加载输入向量tile
    x = tl.load(X_PTR + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # 加载权重tile并算点积
    w = tl.load(W_PTR + offs_m[:, None] * N + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

    # 点积
    out = tl.sum(w * x[None, :], axis=1)

    # 存输出
    tl.store(OUT_PTR + offs_m, out, mask=mask_m)


def triton_decode_gemv(x, weight):
    """单行GEMV: x[1, N] @ weight[M, N].T = out[1, M]，decode专用。"""
    N = weight.shape[1]
    M = weight.shape[0]

    x_flat = x.reshape(-1)  # [N]
    out = torch.empty(M, dtype=x.dtype, device=x.device)

    BLOCK_M = 64
    BLOCK_N = triton.next_power_of_2(N)

    grid = (triton.cdiv(M, BLOCK_M),)
    _decode_gemv_kernel[grid](
        x_flat, weight, out,
        N=N, M=M,
        BLOCK_M=BLOCK_M, BLOCK_N=min(BLOCK_N, 2048),
    )

    return out.reshape(x.shape[:-1] + (M,))
