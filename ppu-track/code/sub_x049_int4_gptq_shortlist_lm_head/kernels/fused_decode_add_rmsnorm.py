import torch
import triton
import triton.language as tl


@triton.jit
def _decode_add_rmsnorm_kernel(
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
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < n_cols

    x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0.0)
    y = tl.load(y_ptr + row * stride_y_row + offs * stride_y_col, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)

    summed = (x + y).to(tl.float16, fp_downcast_rounding="rtne")
    summed_fp = summed.to(tl.float32)
    variance = tl.sum(summed_fp * summed_fp, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)
    normed = (summed_fp * rstd).to(summed.dtype) * w

    tl.store(sum_ptr + row * stride_sum_row + offs * stride_sum_col, summed, mask=mask)
    tl.store(norm_ptr + row * stride_norm_row + offs * stride_norm_col, normed, mask=mask)


def reference_decode_add_rmsnorm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x + y
    summed_fp32 = summed.float()
    variance = summed_fp32.pow(2).mean(dim=-1, keepdim=True)
    normed = weight * (summed_fp32 * torch.rsqrt(variance + eps)).to(summed.dtype)
    return summed, normed


def _as_rows(x: torch.Tensor, name: str) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim < 2:
        raise ValueError(f"{name} must be at least 2D")
    original_shape = tuple(x.shape)
    rows = x.reshape(-1, int(x.shape[-1]))
    return rows, original_shape


def fused_decode_add_rmsnorm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    block_d: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("fused_decode_add_rmsnorm requires CUDA tensors")
    if tuple(x.shape) != tuple(y.shape):
        raise ValueError("x and y must have identical shapes")
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise ValueError("fused_decode_add_rmsnorm currently supports fp16")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")

    x_rows, original_shape = _as_rows(x, "x")
    y_rows, _ = _as_rows(y, "y")
    hidden_size = int(x_rows.shape[1])

    if weight.ndim != 1 or int(weight.shape[0]) != hidden_size:
        raise ValueError("weight must be 1D and match hidden size")
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    weight = weight.contiguous()

    if out_sum is None:
        sum_rows = torch.empty_like(x_rows)
    else:
        if tuple(out_sum.shape) != original_shape or out_sum.dtype != x.dtype or out_sum.device != x.device:
            raise ValueError("out_sum must match x shape/dtype/device")
        sum_rows, _ = _as_rows(out_sum, "out_sum")

    if out_norm is None:
        norm_rows = torch.empty_like(x_rows)
    else:
        if tuple(out_norm.shape) != original_shape or out_norm.dtype != x.dtype or out_norm.device != x.device:
            raise ValueError("out_norm must match x shape/dtype/device")
        norm_rows, _ = _as_rows(out_norm, "out_norm")

    block_d = int(block_d) if block_d is not None and int(block_d) > 0 else int(triton.next_power_of_2(hidden_size))
    num_warps = int(num_warps) if num_warps is not None and int(num_warps) > 0 else (4 if block_d <= 1024 else 8)
    num_stages = int(num_stages)
    _decode_add_rmsnorm_kernel[(int(x_rows.shape[0]),)](
        x_rows,
        y_rows,
        weight,
        sum_rows,
        norm_rows,
        x_rows.stride(0),
        x_rows.stride(1),
        y_rows.stride(0),
        y_rows.stride(1),
        sum_rows.stride(0),
        sum_rows.stride(1),
        norm_rows.stride(0),
        norm_rows.stride(1),
        hidden_size,
        float(eps),
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return sum_rows.view(original_shape), norm_rows.view(original_shape)
