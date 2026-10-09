"""
[SHLEE] Fused RoPE (Rotary Position Embedding) CUDA kernels -- half2 vectorized.

Three specialized kernels for maximum throughput:
  1. Text RoPE:   q[bs, num_heads, seq, head_dim], cos/sin[bs, seq, head_dim]
  2. Vision RoPE: q[seq, num_heads, head_dim], cos/sin[seq, head_dim]
  3. Single-token decode: q[bs, num_heads, 1, head_dim]

Broadcasts cos/sin to head dimension inside the kernel to avoid materializing
expanded tensors in GPU memory.
"""
import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

__global__ void rope_bshd_strided_half2_kernel(
    const half* __restrict__ x,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    half* __restrict__ out,
    int H, int S, int D,
    int64_t x_s0, int64_t x_s1, int64_t x_s2,
    int64_t cos_s0, int64_t cos_s1,
    int64_t sin_s0, int64_t sin_s1,
    int64_t o_s0, int64_t o_s1, int64_t o_s2)
{
    int hs = blockIdx.x;
    int b  = blockIdx.y;

    int h = hs / S;
    int s = hs % S;

    int halfD = D >> 1;

    const half* x_row   = x   + b * x_s0   + h * x_s1   + s * x_s2;
    const half* cos_row = cos + b * cos_s0 + s * cos_s1;
    const half* sin_row = sin + b * sin_s0 + s * sin_s1;
    half* out_row       = out + b * o_s0   + h * o_s1   + s * o_s2;

    for (int d = (threadIdx.x << 1); d < halfD; d += (blockDim.x << 1)) {
        half2 x_lo = *reinterpret_cast<const half2*>(x_row + d);
        half2 x_hi = *reinterpret_cast<const half2*>(x_row + d + halfD);

        half2 c_lo = *reinterpret_cast<const half2*>(cos_row + d);
        half2 c_hi = *reinterpret_cast<const half2*>(cos_row + d + halfD);

        half2 s_lo = *reinterpret_cast<const half2*>(sin_row + d);
        half2 s_hi = *reinterpret_cast<const half2*>(sin_row + d + halfD);

        half2 out_lo = __hsub2(__hmul2(x_lo, c_lo), __hmul2(x_hi, s_lo));
        half2 out_hi = __hadd2(__hmul2(x_hi, c_hi), __hmul2(x_lo, s_hi));

        *reinterpret_cast<half2*>(out_row + d)         = out_lo;
        *reinterpret_cast<half2*>(out_row + d + halfD) = out_hi;
    }
}

__global__ void rope_vision_kernel(
    const half* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    half* __restrict__ out,
    int S,
    int H,
    int64_t x_s0,
    int64_t x_s1,
    int64_t o_s0,
    int64_t o_s1)
{
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;

    const int local_row = lane >> 4;
    const int lane16    = lane & 15;
    const int d         = lane16 * 2;

    const int total_rows = S * H;
    const int rows_per_block = 8;
    const int total_work = (total_rows + rows_per_block - 1) / rows_per_block;

    for (int work = blockIdx.x; work < total_work; work += gridDim.x) {
        const int row_in_block = warp_id * 2 + local_row; // 0..7
        const int row = work * rows_per_block + row_in_block;

        if (row >= total_rows) {
            continue;
        }

        const int s = row / H;
        const int h = row - s * H;

        const half* __restrict__ x_row = x + s * x_s0 + h * x_s1;
        half* __restrict__ out_row     = out + s * o_s0 + h * o_s1;

        const float* __restrict__ cos_row = cos + ((int64_t)s * 64);
        const float* __restrict__ sin_row = sin + ((int64_t)s * 64);

        const half2 x_lo_h2 = *reinterpret_cast<const half2*>(x_row + d);
        const half2 x_hi_h2 = *reinterpret_cast<const half2*>(x_row + d + 32);

        const float2 x_lo = __half22float2(x_lo_h2);
        const float2 x_hi = __half22float2(x_hi_h2);

        const float2 c_lo = *reinterpret_cast<const float2*>(cos_row + d);
        const float2 c_hi = *reinterpret_cast<const float2*>(cos_row + d + 32);
        const float2 s_lo = *reinterpret_cast<const float2*>(sin_row + d);
        const float2 s_hi = *reinterpret_cast<const float2*>(sin_row + d + 32);

        float2 out_lo;
        out_lo.x = x_lo.x * c_lo.x - x_hi.x * s_lo.x;
        out_lo.y = x_lo.y * c_lo.y - x_hi.y * s_lo.y;

        float2 out_hi;
        out_hi.x = x_hi.x * c_hi.x + x_lo.x * s_hi.x;
        out_hi.y = x_hi.y * c_hi.y + x_lo.y * s_hi.y;

        *reinterpret_cast<half2*>(out_row + d)      = __float22half2_rn(out_lo);
        *reinterpret_cast<half2*>(out_row + d + 32) = __float22half2_rn(out_hi);
    }
}

torch::Tensor fused_rope_strided_bshd(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin)
{
    const int B = (int)x.size(0);
    const int H = (int)x.size(1);
    const int S = (int)x.size(2);
    const int D = (int)x.size(3);

    auto out = torch::empty_like(x);

    const int64_t HS = (int64_t)H * (int64_t)S;
    dim3 grid((unsigned int)HS, (unsigned int)B, 1);
    constexpr int BLOCK = 128;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope_bshd_strided_half2_kernel<<<grid, BLOCK, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        H, S, D,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2));
    return out;
}

torch::Tensor fused_rope_vision(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin)
{
    const int S = (int)x.size(0);
    const int H = (int)x.size(1);
    const int D = (int)x.size(2);

    auto out = torch::empty_like(x);

    const int total_rows = S * H;
    const int rows_per_block = 8;
    const int total_work = (total_rows + rows_per_block - 1) / rows_per_block;

    int blocks = total_work;
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;

    dim3 grid((unsigned int)blocks, 1, 1);
    dim3 block(128, 1, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rope_vision_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        S,
        H,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

_CPP_SRC = """
torch::Tensor fused_rope_strided_bshd(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);
torch::Tensor fused_rope_vision(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fused_rope_v6")
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_rope_v6",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_rope_strided_bshd", "fused_rope_vision"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=False,
    )
    return _module

def fused_apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    position_ids=None, unsqueeze_dim=1,
) -> tuple[torch.Tensor, torch.Tensor]:
    _load()
    dtype = q.dtype
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    q_out = _module.fused_rope_strided_bshd(q, cos, sin)
    k_out = _module.fused_rope_strided_bshd(k, cos, sin)
    return q_out, k_out

def fused_apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _load()

    q_out = _module.fused_rope_vision(q, cos, sin)
    k_out = _module.fused_rope_vision(k, cos, sin)
    return q_out, k_out
