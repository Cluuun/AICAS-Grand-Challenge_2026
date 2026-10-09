"""融合RMSNorm的Triton kernel，一次pass搞定方差、归一化、缩放。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    X,          # 输入指针
    Y,          # 输出指针
    W,          # 权重指针
    stride_x,   # 输入行步长
    stride_y,   # 输出行步长
    N,          # 隐藏层大小
    eps,        # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # 加载并转成float32
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)

    # 算RMS: sqrt(mean(x^2) + eps)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # 加载权重并应用
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _rms_norm_residual_kernel(
    X,          # 输入指针
    Y,          # 输出指针
    W,          # 权重指针
    R,          # residual指针（保存原始输入）
    stride_x,
    stride_y,
    stride_r,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """融合RMSNorm，顺便把原始输入存下来当residual。"""
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y
    R += row * stride_r

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Y + offs, y, mask=mask)
    tl.store(R + offs, x, mask=mask)


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """融合RMSNorm前向传播。"""
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    M, N = x.shape

    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)

    _rms_norm_fwd_kernel[(M,)](
        x, y, weight,
        x.stride(0), y.stride(0),
        N=N, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(shape)


def fused_rms_norm_with_residual(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合RMSNorm，同时返回原始输入作为residual。"""
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    M, N = x.shape

    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)
    r = torch.empty_like(x)

    _rms_norm_residual_kernel[(M,)](
        x, y, weight, r,
        x.stride(0), y.stride(0), r.stride(0),
        N=N, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(shape), r.reshape(shape)


def patch_rms_norm(model: torch.nn.Module):
    """把所有Qwen3VLTextRMSNorm模块patch成Triton版本。"""
    import types
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLTextRMSNorm):
            w = module.weight
            eps = module.variance_epsilon

            def _make_fwd(_w, _eps):
                def fwd(self, hidden_states):
                    return fused_rms_norm(hidden_states, _w, _eps)
                return fwd

            module.forward = types.MethodType(_make_fwd(w, eps), module)
            count += 1

    return count
