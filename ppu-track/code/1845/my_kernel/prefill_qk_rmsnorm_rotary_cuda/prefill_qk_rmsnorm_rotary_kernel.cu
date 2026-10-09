#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 128;

__device__ __forceinline__ float warp_reduce_sum(float v) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_128(float v) {
  __shared__ float warp_sums[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) {
    warp_sums[warp] = v;
  }
  __syncthreads();
  float total = (threadIdx.x < 4) ? warp_sums[threadIdx.x] : 0.0f;
  if (warp == 0) {
    total = warp_reduce_sum(total);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    warp_sums[0] = total;
  }
  __syncthreads();
  return warp_sums[0];
}

__global__ void prefill_qk_rmsnorm_rotary_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    float eps,
    int seq_len,
    int q_heads,
    int k_heads,
    int64_t q_stride_s,
    int64_t q_stride_h,
    int64_t q_stride_d,
    int64_t k_stride_s,
    int64_t k_stride_h,
    int64_t k_stride_d) {
  const int tid = threadIdx.x;
  const int q_rows = seq_len * q_heads;
  const int row = blockIdx.x;
  const bool is_q = row < q_rows;
  const int local_row = is_q ? row : (row - q_rows);
  const int heads = is_q ? q_heads : k_heads;
  const int token = local_row / heads;
  const int head = local_row - token * heads;

  const __nv_bfloat16* in = is_q
      ? q + token * q_stride_s + head * q_stride_h + tid * q_stride_d
      : k + token * k_stride_s + head * k_stride_h + tid * k_stride_d;
  const __nv_bfloat16* weight = is_q ? q_weight : k_weight;

  const float x = __bfloat162float(*in);
  const float sumsq = block_reduce_128(x * x);
  const float inv_rms = rsqrtf(sumsq * (1.0f / static_cast<float>(kHeadDim)) + eps);

  __shared__ float normed[kHeadDim];
  const float x_norm = __bfloat162float(__float2bfloat16(x * inv_rms));
  normed[tid] = __bfloat162float(__float2bfloat16(x_norm * __bfloat162float(weight[tid])));
  __syncthreads();

  const int half = kHeadDim >> 1;
  const bool first_half = tid < half;
  const int pair = first_half ? (tid + half) : (tid - half);
  const float rotated = first_half ? -normed[pair] : normed[pair];
  const float c = __bfloat162float(cos[token * kHeadDim + tid]);
  const float s = __bfloat162float(sin[token * kHeadDim + tid]);
  const float left = __bfloat162float(__float2bfloat16(normed[tid] * c));
  const float rot = __bfloat162float(__float2bfloat16(rotated));
  const float right = __bfloat162float(__float2bfloat16(rot * s));
  const __nv_bfloat16 outv = __float2bfloat16(left + right);

  if (is_q) {
    q_out[((head * seq_len + token) * kHeadDim) + tid] = outv;
  } else {
    k_out[((head * seq_len + token) * kHeadDim) + tid] = outv;
  }
}

}  // namespace

void run_prefill_qk_rmsnorm_rotary_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    torch::Tensor q_out,
    torch::Tensor k_out) {
  const int seq_len = static_cast<int>(q.size(1));
  const int q_heads = static_cast<int>(q.size(2));
  const int k_heads = static_cast<int>(k.size(2));
  const int total_rows = seq_len * (q_heads + k_heads);
  const dim3 grid(total_rows);
  const dim3 block(kHeadDim);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefill_qk_rmsnorm_rotary_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
      static_cast<float>(eps),
      seq_len,
      q_heads,
      k_heads,
      q.stride(1),
      q.stride(2),
      q.stride(3),
      k.stride(1),
      k.stride(2),
      k.stride(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
