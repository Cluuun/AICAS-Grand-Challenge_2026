import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_ptrs = x_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
    w_ptrs = w_ptr + col_offsets
    y_ptrs = y_ptr + row_idx * stride_y_row + col_offsets * stride_y_col

    x_in = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x_in.to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.0)

    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)
    y = (x * rstd).to(x_in.dtype) * w
    tl.store(y_ptrs, y, mask=mask)


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    return (weight * normed.to(x.dtype)).to(x.dtype)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("fused_rmsnorm requires CUDA tensor input")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("weight size must match the last dimension of x")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight must have the same dtype")

    hidden_size = x.shape[-1]
    if hidden_size > 8192:
        raise ValueError("fused_rmsnorm currently supports hidden_size <= 8192")

    x2d = x.contiguous().view(-1, hidden_size)
    y2d = torch.empty_like(x2d)
    weight = weight.contiguous()

    block_size = max(16, triton.next_power_of_2(hidden_size))
    num_warps = 4 if block_size <= 1024 else 8

    _fused_rmsnorm_kernel[(x2d.shape[0],)](
        x2d,
        weight,
        y2d,
        x2d.stride(0),
        x2d.stride(1),
        y2d.stride(0),
        y2d.stride(1),
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return y2d.view_as(x)
