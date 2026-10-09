#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr float kScale = 0.08838834764831845f;  // 1 / sqrt(128)

__inline__ __device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void bf16_verify_attn_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ out,
    int q_len,
    int kv_len) {
  const int qh = blockIdx.x;
  const int qi = qh / kQueryHeads;
  const int hq = qh - qi * kQueryHeads;
  const int hkv = hq >> 1;
  const int d = threadIdx.x;
  const int lane = d & 31;
  const int warp_id = d >> 5;

  __shared__ float warp_reduce[4];
  __shared__ float score_shared;
  const int prefix_len = kv_len - q_len;
  const int end_pos = prefix_len + qi + 1;

  const int q_base = (hq * q_len + qi) * kHeadDim;
  const float qd = __bfloat162float(query[q_base + d]);

  float m = -3.4028234663852886e38f;
  float l = 0.0f;
  float acc = 0.0f;

  for (int s = 0; s < end_pos; ++s) {
    const int kv_base = (hkv * kv_len + s) * kHeadDim;
    float dot = qd * __bfloat162float(key[kv_base + d]);
    dot = warp_sum(dot);
    if (lane == 0) {
      warp_reduce[warp_id] = dot;
    }
    __syncthreads();

    if (warp_id == 0) {
      float block_dot = lane < 4 ? warp_reduce[lane] : 0.0f;
      block_dot = warp_sum(block_dot);
      if (lane == 0) {
        score_shared = block_dot * kScale;
      }
    }
    __syncthreads();

    const float score = score_shared;
    const float m_new = fmaxf(m, score);
    const float p = __expf(score - m_new);
    const float alpha = __expf(m - m_new);
    const float vd = __bfloat162float(value[kv_base + d]);
    acc = acc * alpha + p * vd;
    l = l * alpha + p;
    m = m_new;
    __syncthreads();
  }

  const int out_base = (qi * kQueryHeads + hq) * kHeadDim;
  out[out_base + d] = __float2bfloat16(acc / l);
}

}  // namespace

void run_bf16_verify_attn_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor out) {
  const int q_len = static_cast<int>(query.size(2));
  const int kv_len = static_cast<int>(key.size(2));
  const dim3 grid(q_len * kQueryHeads);
  const dim3 block(kHeadDim);
  auto stream = at::cuda::getCurrentCUDAStream();
  bf16_verify_attn_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      q_len,
      kv_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
