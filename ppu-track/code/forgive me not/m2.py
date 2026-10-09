import triton
import triton.language as tl
from torch import nn
import torch
import torch.nn.functional as F

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True).to(tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


class Qwen3VLVisionPatchEmbedNew(nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        config = {
          "deepstack_visual_indexes": [
            5,
            11,
            17
          ],
          "depth": 24,
          "dtype": "float16",
          "hidden_act": "gelu_pytorch_tanh",
          "hidden_size": 1024,
          "in_channels": 3,
          "initializer_range": 0.02,
          "intermediate_size": 4096,
          "model_type": "qwen3_vl",
          "num_heads": 16,
          "num_position_embeddings": 2304,
          "out_hidden_size": 2048,
          "patch_size": 16,
          "spatial_merge_size": 2,
          "temporal_patch_size": 2,
          "transformers_version": "5.2.0"
        }
        self.patch_size = config["patch_size"]
        self.temporal_patch_size = config["temporal_patch_size"]
        self.in_channels = config["in_channels"]
        self.embed_dim = config["hidden_size"]
        
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
        # Keep the channels_last_3d conversion for compatibility; we will not use it directly.
        self.proj = self.proj.to(memory_format=torch.channels_last_3d)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        # Reshape to patches
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # Convert to target dtype (channels_last_3d not needed for our kernel)
        hidden_states = hidden_states.to(dtype=target_dtype)

        M = hidden_states.shape[0]
        K = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        N = self.embed_dim

        # Flatten input to (M, K)
        hidden_states_2d = hidden_states.reshape(M, K).contiguous()

        # Get weight and bias
        weight = self.proj.weight.contiguous()  # shape (N, C, D, H, W)
        weight_2d = weight.view(N, K).contiguous()
        bias = self.proj.bias

        # Output tensor
        output = torch.empty((M, N), dtype=target_dtype, device=hidden_states.device)

        # Strides
        stride_am = hidden_states_2d.stride(0)
        stride_ak = hidden_states_2d.stride(1)
        stride_bk = weight_2d.stride(1)  # stride along K
        stride_bn = weight_2d.stride(0)  # stride along N
        stride_cm = output.stride(0)
        stride_cn = output.stride(1)

        # Kernel launch
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        matmul_kernel[grid](
            hidden_states_2d, weight_2d, output, bias,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_warps=8,
        )

        return output

@triton.jit
def rms_norm_fwd_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_cols,
    eps,
    stride_in_row,
    stride_out_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * stride_in_row

    # first pass: compute sum of squares
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        _sum_sq += x * x
    variance = tl.sum(_sum_sq, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    # second pass: normalize and apply weight
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = x * rstd * w
        tl.store(output_ptr + row_start + cols, out, mask=mask)


class TextRMSNew(nn.Module):
    def __init__(self, hidden_size=2048, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        original_shape = hidden_states.shape
        hidden_states = hidden_states.contiguous()
        hidden_size = original_shape[-1]
        num_rows = hidden_states.numel() // hidden_size

        # flatten to 2D (num_rows, hidden_size)
        hidden_2d = hidden_states.view(num_rows, hidden_size)
        weight_1d = self.weight.contiguous()
        output_2d = torch.empty_like(hidden_2d, dtype=torch.float32)

        # kernel configuration
        BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        grid = (num_rows,)
        rms_norm_fwd_kernel[grid](
            hidden_2d,
            weight_1d,
            output_2d,
            hidden_size,
            self.variance_epsilon,
            hidden_2d.stride(0),
            output_2d.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        output = output_2d.view(original_shape)
        # match the dtype promotion of the original implementation
        output = output.to(torch.promote_types(self.weight.dtype, input_dtype))
        return output