import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    sum_ptr,
    out_ptr,
    stride_x_r,
    stride_x_c,
    stride_y_r,
    stride_y_c,
    stride_s_r,
    stride_s_c,
    stride_o_r,
    stride_o_c,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    x = tl.load(x_ptr + row * stride_x_r + cols * stride_x_c, mask=mask, other=0.0)
    y = tl.load(y_ptr + row * stride_y_r + cols * stride_y_c, mask=mask, other=0.0)

    # Match PyTorch's two-op fp16 residual semantics without reloading from
    # global memory: round the add result in registers, then reuse it.
    summed = (x + y).to(tl.float16, fp_downcast_rounding="rtne")
    tl.store(sum_ptr + row * stride_s_r + cols * stride_s_c, summed, mask=mask)
    s = summed.to(tl.float32)

    mean = tl.sum(s, axis=0) / hidden_size
    centered = tl.where(mask, s - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / hidden_size
    inv_std = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = centered * inv_std * w + b

    tl.store(out_ptr + row * stride_o_r + cols * stride_o_c, out, mask=mask)


def fused_add_layernorm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out_sum: torch.Tensor = None,
    out_norm: torch.Tensor = None,
):
    if not (x.is_cuda and y.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError("fused_add_layernorm requires CUDA tensors")
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError("x and y must have same shape with ndim >= 2")
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise ValueError("x/y must be fp16")
    hidden_size = x.shape[-1]
    if weight.ndim != 1 or bias.ndim != 1 or weight.shape[0] != hidden_size or bias.shape[0] != hidden_size:
        raise ValueError("weight/bias must be shape [hidden_size]")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise ValueError("weight/bias dtype must match x/y dtype")
    if hidden_size > 8192:
        raise ValueError("fused_add_layernorm currently supports hidden_size <= 8192")

    x2d = x.reshape(-1, hidden_size)
    y2d = y.reshape(-1, hidden_size)
    if out_sum is None:
        out_sum = torch.empty_like(x2d)
    else:
        out_sum = out_sum.reshape(-1, hidden_size)
    if out_norm is None:
        out_norm = torch.empty_like(x2d)
    else:
        out_norm = out_norm.reshape(-1, hidden_size)

    block_size = 1
    while block_size < hidden_size:
        block_size <<= 1
    block_size = min(block_size, 8192)

    _fused_add_layernorm_kernel[(x2d.shape[0],)](
        x2d,
        y2d,
        weight,
        bias,
        out_sum,
        out_norm,
        x2d.stride(0),
        x2d.stride(1),
        y2d.stride(0),
        y2d.stride(1),
        out_sum.stride(0),
        out_sum.stride(1),
        out_norm.stride(0),
        out_norm.stride(1),
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=8 if hidden_size >= 2048 else 4,
        num_stages=2,
    )
    return out_sum.view_as(x), out_norm.view_as(x)
