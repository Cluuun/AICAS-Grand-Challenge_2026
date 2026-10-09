"""
[SHLEE] Fused RMSNorm CUDA kernel -- vectorized float4 loads, per-dimension block sizing.

Targets two hot paths:
  - text decoder: hidden_size=2048 (28 layers x 4 norms + 1 final = 113 invocations)
  - q_norm/k_norm: hidden_size=128 (28 layers x 2 = 56 invocations)
"""
import torch
import torch.nn as nn

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

/*
 * Vectorized RMSNorm.  Each block handles one row.
 *
 * BLOCK threads,  each loads VEC_SIZE half values at a time.
 * float4 gives us 4 floats = 8 halfs per 16-byte load on fp16 data.
 *
 * Template params:
 *   BLOCK  - threads per block (must be power-of-2, <=1024)
 *   VEC    - number of fp16 elements per vectorized load (8 for float4)
 */
template <int BLOCK>
__global__ void fused_rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int hidden_size,
    const float eps)
{
    const int row = blockIdx.x;
    const half* x = input  + row * hidden_size;
    half*       o = output + row * hidden_size;

    /* ---- accumulate sum-of-squares in fp32 ---- */
    float local_ss = 0.0f;

    /* Vectorized path: process 8 halfs (= 1 float4 = 16 bytes) at a time */
    const int vec_size = 8;
    const int vec_iters = hidden_size / vec_size;  /* assumes hidden_size % 8 == 0 */

    const float4* x_vec = reinterpret_cast<const float4*>(x);

    for (int i = threadIdx.x; i < vec_iters; i += BLOCK) {
        float4 v4 = x_vec[i];
        half2* h2 = reinterpret_cast<half2*>(&v4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f2 = __half22float2(h2[j]);
            local_ss += f2.x * f2.x + f2.y * f2.y;
        }
    }

    /* ---- warp reduce ---- */
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    /* ---- block reduce via shared memory ---- */
    __shared__ float s_rrms;
    constexpr int NUM_WARPS = BLOCK / 32;
    __shared__ float smem[NUM_WARPS];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    if (lane == 0) smem[wid] = local_ss;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, offset);
        if (lane == 0)
            s_rrms = rsqrtf(val / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();

    const float rrms = s_rrms;

    /* ---- scale output (vectorized) ---- */
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4*       o_vec = reinterpret_cast<float4*>(o);

    for (int i = threadIdx.x; i < vec_iters; i += BLOCK) {
        float4 xv = x_vec[i];
        float4 wv = w_vec[i];
        half2* xh = reinterpret_cast<half2*>(&xv);
        half2* wh = reinterpret_cast<half2*>(&wv);
        float4 ov;
        half2* oh = reinterpret_cast<half2*>(&ov);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            float2 wf = __half22float2(wh[j]);
            xf.x = xf.x * rrms * wf.x;
            xf.y = xf.y * rrms * wf.y;
            oh[j] = __float22half2_rn(xf);
        }
        o_vec[i] = ov;
    }
}

torch::Tensor fused_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps)
{
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kHalf,
                "input must be CUDA fp16");

    auto flat = input.contiguous().view({-1, input.size(-1)});
    auto output = torch::empty_like(flat);

    const int rows = flat.size(0);
    const int cols = flat.size(1);

    /* Pick block size based on hidden dimension:
     * cols=128  -> 32  threads (1 warp,  each does 4 vec iters)
     * cols=2048 -> 128 threads (4 warps, each does 2 vec iters)
     * This maximises occupancy (more blocks per SM).
     */
    const half* in_ptr  = reinterpret_cast<const half*>(flat.data_ptr<at::Half>());
    const half* w_ptr   = reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
    half*       out_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (cols <= 128) {
        fused_rmsnorm_kernel<32><<<rows, 32, 0, stream>>>(in_ptr, w_ptr, out_ptr, cols, (float)eps);
    } else if (cols <= 1024) {
        fused_rmsnorm_kernel<128><<<rows, 128, 0, stream>>>(in_ptr, w_ptr, out_ptr, cols, (float)eps);
    } else {
        fused_rmsnorm_kernel<256><<<rows, 256, 0, stream>>>(in_ptr, w_ptr, out_ptr, cols, (float)eps);
    }

    return output.view(input.sizes());
}
"""

_CPP_SRC = """
torch::Tensor fused_rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fused_rmsnorm")
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_rmsnorm",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_rmsnorm_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=False,
    )
    return _module

class FusedRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _module.fused_rmsnorm_forward(x, self.weight, self.variance_epsilon)
