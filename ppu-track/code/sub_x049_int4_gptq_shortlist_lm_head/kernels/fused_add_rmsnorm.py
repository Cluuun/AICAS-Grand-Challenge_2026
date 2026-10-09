import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    sum_ptr,
    norm_ptr,
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    stride_sum_row,
    stride_sum_col,
    stride_norm_row,
    stride_norm_col,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_ptrs = x_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
    y_ptrs = y_ptr + row_idx * stride_y_row + col_offsets * stride_y_col
    w_ptrs = w_ptr + col_offsets
    sum_ptrs = sum_ptr + row_idx * stride_sum_row + col_offsets * stride_sum_col
    norm_ptrs = norm_ptr + row_idx * stride_norm_row + col_offsets * stride_norm_col

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptrs, mask=mask, other=0.0)

    # Match PyTorch's two-op fp16 residual semantics without reloading from
    # global memory: round the add result in registers, then reuse it.
    s = (x + y).to(tl.float16, fp_downcast_rounding="rtne")
    s_fp = s.to(tl.float32)
    variance = tl.sum(s_fp * s_fp, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)
    norm = (s_fp * rstd).to(s.dtype) * w

    tl.store(sum_ptrs, s, mask=mask)
    tl.store(norm_ptrs, norm, mask=mask)


def reference_add_rmsnorm(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, eps: float):
    s = x + y
    variance = s.float().pow(2).mean(-1, keepdim=True)
    norm = weight * (s.float() * torch.rsqrt(variance + eps)).to(s.dtype)
    return s, norm


def fused_add_rmsnorm(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, eps: float):
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("fused_add_rmsnorm requires CUDA tensors")
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("weight size must match the last dim of x/y")
    if x.dtype != y.dtype or x.dtype != weight.dtype:
        raise ValueError("x, y and weight must have identical dtype")

    hidden_size = x.shape[-1]
    if hidden_size > 8192:
        raise ValueError("fused_add_rmsnorm currently supports hidden_size <= 8192")

    x2d = x.contiguous().view(-1, hidden_size)
    y2d = y.contiguous().view(-1, hidden_size)
    weight = weight.contiguous()
    sum2d = torch.empty_like(x2d)
    norm2d = torch.empty_like(x2d)

    block_size = max(16, triton.next_power_of_2(hidden_size))
    num_warps = 4 if block_size <= 1024 else 8

    _fused_add_rmsnorm_kernel[(x2d.shape[0],)](
        x2d,
        y2d,
        weight,
        sum2d,
        norm2d,
        x2d.stride(0),
        x2d.stride(1),
        y2d.stride(0),
        y2d.stride(1),
        sum2d.stride(0),
        sum2d.stride(1),
        norm2d.stride(0),
        norm2d.stride(1),
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )

    return sum2d.view_as(x), norm2d.view_as(x)
