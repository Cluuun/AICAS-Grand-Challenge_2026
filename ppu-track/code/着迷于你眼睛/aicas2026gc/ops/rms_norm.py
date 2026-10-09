import triton
import triton.language as tl
import torch


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    o_ptr,
    w_ptr,
    stride_x_0: tl.constexpr,
    stride_x_1: tl.constexpr,  # <--- 传入 x 的两个步长
    stride_o_0: tl.constexpr,
    stride_o_1: tl.constexpr,  # <--- 严谨起见，最好把输出 y 的步长也传入
    hidden_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_dim

    x_offsets = row_idx * stride_x_0 + col_offsets * stride_x_1
    o_offsets = row_idx * stride_o_0 + col_offsets * stride_o_1

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    x_sq = x_fp32 * x_fp32
    variance = tl.sum(x_sq, axis=0) / hidden_dim

    rsqrt = tl.math.rsqrt(variance + eps)
    x_normed_fp32 = x_fp32 * rsqrt
    if w_ptr:
        w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x_normed_fp32 * w).to(x.dtype)
    else:
        y = x_normed_fp32.to(x.dtype)

    tl.store(o_ptr + o_offsets, y, mask=mask)


@torch.library.custom_op("qwen3vl::rmsnorm_forward", mutates_args=())  # mutates_args 为空不能有 inplace
def rmsnorm_forward(x: torch.Tensor, hidden_dim: int, weight: torch.Tensor | None = None, inplace: bool = False, eps: float = 1e-6) -> torch.Tensor:
    # 在 BS=1 时，性能好像比 F.rms_norm 好
    x_2d = x.reshape(-1, hidden_dim)
    M, N = x_2d.shape

    y = x if inplace else torch.empty_like(x)
    y_2d = y.view(-1, N)

    BLOCK_SIZE = triton.next_power_of_2(N)

    num_warps = 8 if N > 2048 else 4  # 启发式设置 num_warps

    grid = (M,)

    rmsnorm_kernel[grid](
        x_ptr=x_2d,
        o_ptr=y_2d,
        w_ptr=weight,
        stride_x_0=x_2d.stride(0),
        stride_x_1=x_2d.stride(1),
        stride_o_0=y_2d.stride(0),
        stride_o_1=y_2d.stride(1),
        hidden_dim=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y


@rmsnorm_forward.register_fake
def _fake_rmsnorm_forward(x: torch.Tensor, hidden_dim: int, weight: torch.Tensor | None = None, inplace: bool = False, eps: float = 1e-6) -> torch.Tensor:
    return torch.empty_like(x)


@triton.jit
def fast_1d_rmsnorm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)  # 2048 == D, no mask needed

    # RMSNorm variance 需要 FP32（2048个平方和会溢出 FP16）
    x = tl.load(x_ptr + offsets).to(tl.float32)

    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / D
    rsqrt = tl.math.rsqrt(var + eps)

    x_normed = x * rsqrt

    if w_ptr:
        w = tl.load(w_ptr + offsets).to(tl.float32)
        out = x_normed * w
    else:
        out = x_normed

    tl.store(y_ptr + offsets, out.to(x_ptr.dtype.element_ty))


def fused_1d_rmsnorm(x, weight=None, eps=1e-6, inplace=False, out=None):
    # ONLYE FOR DECODE and SHAPE=[1, 1, 2048]
    D = x.shape[-1]  # 2048
    y = out if out is not None else (x if inplace else torch.empty_like(x))

    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (1,)

    fast_1d_rmsnorm_kernel[grid](x, y, weight, D, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=1)
    return y
