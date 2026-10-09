#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;

template <int Heads>
__device__ inline void row_to_token_head(int row, int* token, int* head) {
  *token = row / Heads;
  *head = row - (*token) * Heads;
}

__global__ void bf16_qk_rmsnorm_rotary_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    float eps,
    int q_len) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int q_rows = q_len * kQueryHeads;
  const bool is_q = row < q_rows;
  const int local_row = is_q ? row : (row - q_rows);

  int token = 0;
  int head = 0;
  if (is_q) {
    row_to_token_head<kQueryHeads>(local_row, &token, &head);
  } else {
    row_to_token_head<kKvHeads>(local_row, &token, &head);
  }

  const __nv_bfloat16* in = is_q
      ? (q + local_row * kHeadDim)
      : (k + local_row * kHeadDim);
  const __nv_bfloat16* weight = is_q ? q_weight : k_weight;
  __nv_bfloat16* out = is_q
      ? (q_out + (head * q_len + token) * kHeadDim)
      : (k_out + (head * q_len + token) * kHeadDim);

  __shared__ float x_sh[kHeadDim];
  __shared__ float w_sh[kHeadDim];
  __shared__ float warp_red[4];

  const float x = __bfloat162float(in[tid]);
  const float w = __bfloat162float(weight[tid]);
  x_sh[tid] = x;
  w_sh[tid] = w;
  float sq = x * x;

  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sq += __shfl_down_sync(mask, sq, offset);
  }
  if ((tid & 31) == 0) {
    warp_red[tid >> 5] = sq;
  }
  __syncthreads();

  float total_sq = 0.0f;
  if (tid < 4) {
    total_sq = warp_red[tid];
  }
  if (tid < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      total_sq += __shfl_down_sync(mask, total_sq, offset);
    }
    if (tid == 0) {
      warp_red[0] = total_sq;
    }
  }
  __syncthreads();

  const float inv_rms = rsqrtf(warp_red[0] / static_cast<float>(kHeadDim) + eps);
  const int half_dim = kHeadDim >> 1;
  const int pair_idx = (tid < half_dim) ? (tid + half_dim) : (tid - half_dim);

  const float v_norm = x * inv_rms * w;
  float pair = x_sh[pair_idx];
  if (tid < half_dim) {
    pair = -pair;
  }
  const float pair_norm = pair * inv_rms * w_sh[pair_idx];

  const int rope_offset = token * kHeadDim + tid;
  const float c = __bfloat162float(cos[rope_offset]);
  const float s = __bfloat162float(sin[rope_offset]);
  out[tid] = __float2bfloat16(v_norm * c + pair_norm * s);
}

}  // namespace

void run_bf16_qk_rmsnorm_rotary_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    torch::Tensor q_out,
    torch::Tensor k_out) {
  const int q_len = static_cast<int>(q.size(0) / kQueryHeads);
  const int total_rows = q_len * (kQueryHeads + kKvHeads);
  const dim3 grid(total_rows);
  const dim3 block(kHeadDim);
  auto stream = at::cuda::getCurrentCUDAStream();

  bf16_qk_rmsnorm_rotary_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
      static_cast<float>(eps),
      q_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
