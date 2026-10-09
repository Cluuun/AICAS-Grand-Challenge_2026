"""自定义GEMV kernel，decode时(seq_len=1)替代cuBLAS。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    X_PTR, W_PTR, OUT_PTR,
    N, K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMV kernel: out[N] = x[K] @ W[N, K].T，每个program算BLOCK_N个输出。"""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_off = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)

        # 加载x的这个block，所有输出元素共用
        x = tl.load(X_PTR + k_off).to(tl.float32)  # [BLOCK_K]

        # 加载W的对应块
        w = tl.load(W_PTR + n_off[:, None] * K + k_off[None, :],
                     mask=n_mask[:, None]).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # 逐元素乘再求和
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(OUT_PTR + n_off, acc.to(OUT_PTR.dtype.element_ty), mask=n_mask)


def triton_gemv(x, weight):
    """GEMV: out[N] = x[K] @ weight[N, K].T"""
    N, K = weight.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)

    BLOCK_N = 32
    BLOCK_K = triton.next_power_of_2(K)

    grid = (triton.cdiv(N, BLOCK_N),)
    _gemv_kernel[grid](
        x, weight, out,
        N=N, K=K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
    )
    return out


@triton.jit
def _fused_rms_norm_gemv_kernel(
    X_PTR, W_PTR, NORM_W_PTR, OUT_PTR,
    N, K: tl.constexpr,
    EPS,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """融合RMS norm + GEMV: 先算norm，再顺便做矩阵乘。"""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_off = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    # 第一阶段：算RMS norm的缩放因子
    sum_sq = tl.zeros([BLOCK_N], dtype=tl.float32)  # 会broadcast
    sq_acc = 0.0
    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        x_val = tl.load(X_PTR + k_off).to(tl.float32)
        sq_acc += tl.sum(x_val * x_val)
    inv_var = tl.math.rsqrt(sq_acc / K + EPS)

    # 第二阶段：边做GEMV边做norm
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)

        # 加载并归一化
        x_val = tl.load(X_PTR + k_off).to(tl.float32)
        nw_val = tl.load(NORM_W_PTR + k_off).to(tl.float32)
        x_norm = x_val * nw_val * inv_var  # [BLOCK_K]

        # 加载权重块
        w = tl.load(W_PTR + n_off[:, None] * K + k_off[None, :],
                     mask=n_mask[:, None]).to(tl.float32)

        acc += tl.sum(w * x_norm[None, :], axis=1)

    tl.store(OUT_PTR + n_off, acc.to(OUT_PTR.dtype.element_ty), mask=n_mask)


def triton_fused_rms_norm_gemv(x, norm_weight, weight, eps):
    """融合RMS norm + GEMV: out = rms_norm(x) @ weight.T，一次搞定。"""
    N, K = weight.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)

    BLOCK_N = 32
    BLOCK_K = triton.next_power_of_2(K)

    grid = (triton.cdiv(N, BLOCK_N),)
    _fused_rms_norm_gemv_kernel[grid](
        x, weight, norm_weight, out,
        N=N, K=K,
        EPS=eps,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
    )
    return out
