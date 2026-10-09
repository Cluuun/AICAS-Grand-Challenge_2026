"""
[SHLEE] Fused LayerNorm CUDA kernel -- vectorized float4 loads, tuned for vision encoder.

Hot path: hidden_size=1024, seq_len in {2304..4096}, 48 norms per forward.

A100 SM 8.0, FP16.
"""
import torch
import torch.nn as nn

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

/*
 * Vectorized LayerNorm.
 * 1024-wide: 128 vec_iters at vec_size=8, BLOCK=128 gives 1 iter/thread.
 * This is the sweet spot: full utilisation, minimal shared-mem, high occupancy.
 */
template <int BLOCK>
__global__ void fused_layernorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    const int hidden_size,
    const float eps)
{
    const int row = blockIdx.x;
    const half* x = input  + row * hidden_size;
    half*       o = output + row * hidden_size;

    float local_sum = 0.0f;
    float local_ss  = 0.0f;

    const int vec_size = 8;
    const int vec_iters = hidden_size / vec_size;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    for (int i = threadIdx.x; i < vec_iters; i += BLOCK) {
        float4 v4 = x_vec[i];
        half2* h2 = reinterpret_cast<half2*>(&v4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f2 = __half22float2(h2[j]);
            local_sum += f2.x + f2.y;
            local_ss  += f2.x * f2.x + f2.y * f2.y;
        }
    }

    /* warp reduce */
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        local_ss  += __shfl_xor_sync(0xffffffff, local_ss,  offset);
    }

    __shared__ float s_mean, s_rstd;
    constexpr int NUM_WARPS = BLOCK / 32;
    __shared__ float smem_sum[NUM_WARPS];
    __shared__ float smem_ss[NUM_WARPS];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    if (lane == 0) {
        smem_sum[wid] = local_sum;
        smem_ss[wid]  = local_ss;
    }
    __syncthreads();

    if (wid == 0) {
        float sv = (lane < NUM_WARPS) ? smem_sum[lane] : 0.0f;
        float ss = (lane < NUM_WARPS) ? smem_ss[lane]  : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sv += __shfl_xor_sync(0xffffffff, sv, offset);
            ss += __shfl_xor_sync(0xffffffff, ss, offset);
        }
        if (lane == 0) {
            float mean = sv / static_cast<float>(hidden_size);
            float var  = ss / static_cast<float>(hidden_size) - mean * mean;
            s_mean = mean;
            s_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    const float mean = s_mean;
    const float rstd = s_rstd;

    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    const float4* b_vec = reinterpret_cast<const float4*>(bias);
    float4*       o_vec = reinterpret_cast<float4*>(o);

    for (int i = threadIdx.x; i < vec_iters; i += BLOCK) {
        float4 xv = x_vec[i];
        float4 wv = w_vec[i];
        float4 bv = b_vec[i];
        half2* xh = reinterpret_cast<half2*>(&xv);
        half2* wh = reinterpret_cast<half2*>(&wv);
        half2* bh = reinterpret_cast<half2*>(&bv);
        float4 ov;
        half2* oh = reinterpret_cast<half2*>(&ov);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            float2 wf = __half22float2(wh[j]);
            float2 bf = __half22float2(bh[j]);
            xf.x = (xf.x - mean) * rstd * wf.x + bf.x;
            xf.y = (xf.y - mean) * rstd * wf.y + bf.y;
            oh[j] = __float22half2_rn(xf);
        }
        o_vec[i] = ov;
    }
}

torch::Tensor fused_layernorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps)
{
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kHalf,
                "input must be CUDA fp16");

    auto flat = input.contiguous().view({-1, input.size(-1)});
    auto output = torch::empty_like(flat);

    const int rows = flat.size(0);
    const int cols = flat.size(1);

    const half* in_ptr  = reinterpret_cast<const half*>(flat.data_ptr<at::Half>());
    const half* w_ptr   = reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
    const half* b_ptr   = reinterpret_cast<const half*>(bias.data_ptr<at::Half>());
    half*       out_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (cols <= 1024) {
        fused_layernorm_kernel<128><<<rows, 128, 0, stream>>>(in_ptr, w_ptr, b_ptr, out_ptr, cols, (float)eps);
    } else {
        fused_layernorm_kernel<256><<<rows, 256, 0, stream>>>(in_ptr, w_ptr, b_ptr, out_ptr, cols, (float)eps);
    }

    return output.view(input.sizes());
}
"""

_CPP_SRC = """
torch::Tensor fused_layernorm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fused_layernorm")
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_layernorm",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_layernorm_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=False,
    )
    return _module

class FusedLayerNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _module.fused_layernorm_forward(x, self.weight, self.bias, self.eps)
