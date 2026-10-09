#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_sum(float v) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }

  __shared__ float warp_sums[BLOCK_THREADS / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = v;
  }
  __syncthreads();

  v = (threadIdx.x < (BLOCK_THREADS / 32)) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(mask, v, offset);
    }
    if (lane == 0) {
      warp_sums[0] = v;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

template <int BLOCK_THREADS>
__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int64_t stride_m,
    int n,
    float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(row) * stride_m;

  float sumsq = 0.0f;
  for (int col = tid; col < n; col += BLOCK_THREADS) {
    const float v = __bfloat162float(x[base + col]);
    sumsq += v * v;
  }

  const float total = block_sum<BLOCK_THREADS>(sumsq);
  const float inv_rms = rsqrtf(total / static_cast<float>(n) + eps);

  for (int col = tid; col < n; col += BLOCK_THREADS) {
    const float v = __bfloat162float(x[base + col]);
    const float w = __bfloat162float(weight[col]);
    out[base + col] = __float2bfloat16(v * inv_rms * w);
  }
}

}  // namespace

void run_prefill_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps,
    torch::Tensor out) {
  const int64_t m = x.size(0);
  const int n = static_cast<int>(x.size(1));
  const int64_t stride_m = x.stride(0);
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int kThreads = 256;
  rmsnorm_kernel<kThreads><<<static_cast<unsigned int>(m), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      stride_m,
      n,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
