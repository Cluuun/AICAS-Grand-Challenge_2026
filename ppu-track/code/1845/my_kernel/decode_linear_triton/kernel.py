from __future__ import annotations

import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def gemv_row_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    K,
    STRIDE_WN: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        x = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        w_ptrs = w_ptr + n_offs[:, None] * STRIDE_WN + k_offs[None, :] * STRIDE_WK
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        b = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        acc += b

    tl.store(y_ptr + n_offs, acc, mask=n_mask)


@triton.jit
def gemv_rows_fixed_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    STRIDE_XM: tl.constexpr,
    STRIDE_XK: tl.constexpr,
    STRIDE_WN: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    STRIDE_YM: tl.constexpr,
    STRIDE_YN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(1)
    pid_n = tl.program_id(0)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        x = tl.load(
            x_ptr + row * STRIDE_XM + k_offs * STRIDE_XK,
            mask=(row < M) & k_mask,
            other=0.0,
        ).to(tl.float32)
        w_ptrs = w_ptr + n_offs[:, None] * STRIDE_WN + k_offs[None, :] * STRIDE_WK
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        b = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        acc += b

    tl.store(
        y_ptr + row * STRIDE_YM + n_offs * STRIDE_YN,
        acc,
        mask=(row < M) & n_mask,
    )
