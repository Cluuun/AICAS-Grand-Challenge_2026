#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

namespace {

template <typename T>
__device__ inline float to_float(T v);
template <>
__device__ inline float to_float<half>(half v) { return __half2float(v); }
template <>
__device__ inline float to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ inline T from_float(float v);
template <>
__device__ inline half from_float<half>(float v) { return __float2half_rn(v); }
template <>
__device__ inline nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename scalar_t>
__global__ void qk_rmsnorm_rotary_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    const float eps,
    const int q_rows,
    const int k_rows,
    const int q_heads,
    const int k_heads,
    const int cos_rows,
    const int dim) {
  const int total_rows = q_rows + k_rows;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= total_rows || tid >= dim) return;

  const bool is_q = row < q_rows;
  const int local_row = is_q ? row : (row - q_rows);
  const int heads = is_q ? q_heads : k_heads;
  const int pos = cos_rows <= 1 ? 0 : (local_row / heads);
  const scalar_t* in = is_q ? (q + local_row * dim) : (k + local_row * dim);
  const scalar_t* weight = is_q ? q_weight : k_weight;
  scalar_t* out = is_q ? (q_out + local_row * dim) : (k_out + local_row * dim);

  __shared__ float x_sh[128];
  __shared__ float w_sh[128];
  __shared__ float warp_red[4];

  const float x = to_float<scalar_t>(in[tid]);
  const float w = to_float<scalar_t>(weight[tid]);
  x_sh[tid] = x;
  w_sh[tid] = w;
  float sq = x * x;

  // Warp reduction for RMS (128 threads => 4 warps).
  unsigned mask = 0xffffffffu;
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
    for (int offset = 16; offset > 0; offset >>= 1) {
      total_sq += __shfl_down_sync(mask, total_sq, offset);
    }
    if (tid == 0) {
      warp_red[0] = total_sq;
    }
  }
  __syncthreads();

  const float inv_rms = rsqrtf(warp_red[0] / static_cast<float>(dim) + eps);
  const int half_dim = dim >> 1;
  const int pair_idx = (tid < half_dim) ? (tid + half_dim) : (tid - half_dim);

  const float v_norm = x * inv_rms * w;
  float pair = x_sh[pair_idx];
  if (tid < half_dim) pair = -pair;
  const float pair_norm = pair * inv_rms * w_sh[pair_idx];

  const int rope_offset = pos * dim + tid;
  const float c = to_float<scalar_t>(cos[rope_offset]);
  const float s = to_float<scalar_t>(sin[rope_offset]);
  out[tid] = from_float<scalar_t>(v_norm * c + pair_norm * s);
}

void check_inputs(const torch::Tensor& q,
                  const torch::Tensor& k,
                  const torch::Tensor& cos,
                  const torch::Tensor& sin,
                  const torch::Tensor& q_weight,
                  const torch::Tensor& k_weight,
                  int64_t q_heads,
                  int64_t k_heads) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos.is_cuda() && sin.is_cuda(), "all tensors must be cuda");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q/k must be contiguous");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
  TORCH_CHECK(q.scalar_type() == cos.scalar_type() && cos.scalar_type() == sin.scalar_type(), "cos/sin dtype mismatch");
  TORCH_CHECK(q.scalar_type() == q_weight.scalar_type() && q.scalar_type() == k_weight.scalar_type(), "weight dtype mismatch");
  TORCH_CHECK(q.dim() == 2 && k.dim() == 2, "q/k must be 2D [rows, dim]");
  TORCH_CHECK(q.size(1) == 128 && k.size(1) == 128, "only dim=128 supported");
  TORCH_CHECK(cos.dim() == sin.dim(), "cos/sin dim mismatch");
  TORCH_CHECK(cos.numel() == sin.numel(), "cos/sin numel mismatch");
  TORCH_CHECK(cos.numel() >= 128 && cos.numel() % 128 == 0, "cos/sin must be [positions,128] or [128]");
  TORCH_CHECK(q_weight.numel() == 128 && k_weight.numel() == 128, "weights must flatten to 128");
  TORCH_CHECK(q_heads > 0 && k_heads > 0, "q_heads/k_heads must be positive");
  TORCH_CHECK(q.size(0) % q_heads == 0, "q rows must be a multiple of q_heads");
  TORCH_CHECK(k.size(0) % k_heads == 0, "k rows must be a multiple of k_heads");
  const int64_t q_positions = q.size(0) / q_heads;
  const int64_t k_positions = k.size(0) / k_heads;
  TORCH_CHECK(q_positions == k_positions, "q/k position count mismatch");
  TORCH_CHECK(cos.numel() == 128 || cos.numel() == q_positions * 128, "cos/sin position count mismatch");
}

template <typename scalar_t>
__global__ void rope_cache_lookup_kernel(
    const scalar_t* __restrict__ cos_table,
    const scalar_t* __restrict__ sin_table,
    const int64_t* __restrict__ position_ids,
    scalar_t* __restrict__ cos_out,
    scalar_t* __restrict__ sin_out,
    const int table_rows,
    const int num_pos,
    const int mrope_h,
    const int mrope_w) {
  const int tid = threadIdx.x;
  if (tid >= 128) return;

  int source = 0;
  if (num_pos == 3) {
    const int half_dim = 64;
    const int d = tid < half_dim ? tid : tid - half_dim;
    const int h_len = mrope_h * 3;
    const int w_len = mrope_w * 3;
    if (d >= 1 && d < h_len && ((d - 1) % 3) == 0) {
      source = 1;
    } else if (d >= 2 && d < w_len && ((d - 2) % 3) == 0) {
      source = 2;
    }
  }

  int64_t pos = position_ids[source];
  if (pos < 0) pos = 0;
  if (pos >= table_rows) pos = table_rows - 1;
  const int64_t offset = pos * 128 + tid;
  cos_out[tid] = cos_table[offset];
  sin_out[tid] = sin_table[offset];
}

} // namespace

std::vector<torch::Tensor> qk_rmsnorm_rotary_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    int64_t q_heads,
    int64_t k_heads) {
  q = q.contiguous();
  k = k.contiguous();
  cos = cos.reshape({-1, 128}).contiguous();
  sin = sin.reshape({-1, 128}).contiguous();
  q_weight = q_weight.reshape({128}).contiguous();
  k_weight = k_weight.reshape({128}).contiguous();
  check_inputs(q, k, cos, sin, q_weight, k_weight, q_heads, k_heads);

  auto q_out = torch::empty_like(q);
  auto k_out = torch::empty_like(k);

  const int threads = 128;
  const int blocks = static_cast<int>(q.size(0) + k.size(0));
  const int cos_rows = static_cast<int>(cos.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == torch::kFloat16) {
    qk_rmsnorm_rotary_kernel<half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(q_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_out.data_ptr<at::Half>()),
        static_cast<float>(eps),
        static_cast<int>(q.size(0)),
        static_cast<int>(k.size(0)),
        static_cast<int>(q_heads),
        static_cast<int>(k_heads),
        cos_rows,
        128);
  } else {
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "only fp16/bf16 supported");
    qk_rmsnorm_rotary_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(q_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
        static_cast<float>(eps),
        static_cast<int>(q.size(0)),
        static_cast<int>(k.size(0)),
        static_cast<int>(q_heads),
        static_cast<int>(k_heads),
        cos_rows,
        128);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out};
}

std::vector<torch::Tensor> rope_cache_lookup_cuda(
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor position_ids,
    int64_t mrope_h,
    int64_t mrope_w) {
  TORCH_CHECK(cos_table.is_cuda() && sin_table.is_cuda() && position_ids.is_cuda(), "all tensors must be cuda");
  TORCH_CHECK(cos_table.scalar_type() == sin_table.scalar_type(), "cos/sin dtype mismatch");
  TORCH_CHECK(cos_table.scalar_type() == torch::kFloat16 || cos_table.scalar_type() == torch::kBFloat16,
              "only fp16/bf16 cos/sin tables supported");
  TORCH_CHECK(position_ids.scalar_type() == torch::kInt64, "position_ids must be int64");
  cos_table = cos_table.reshape({-1, 128}).contiguous();
  sin_table = sin_table.reshape({-1, 128}).contiguous();
  position_ids = position_ids.reshape({-1}).contiguous();
  TORCH_CHECK(cos_table.sizes() == sin_table.sizes(), "cos/sin table shape mismatch");
  TORCH_CHECK(position_ids.numel() == 1 || position_ids.numel() == 3, "position_ids must have 1 or 3 elements");
  TORCH_CHECK(cos_table.size(0) > 0, "empty cos/sin table");

  auto cos_out = torch::empty({1, 128}, cos_table.options());
  auto sin_out = torch::empty({1, 128}, sin_table.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 128;
  const int table_rows = static_cast<int>(cos_table.size(0));
  const int num_pos = static_cast<int>(position_ids.numel());

  if (cos_table.scalar_type() == torch::kFloat16) {
    rope_cache_lookup_kernel<half><<<1, threads, 0, stream>>>(
        reinterpret_cast<const half*>(cos_table.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin_table.data_ptr<at::Half>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<half*>(cos_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(sin_out.data_ptr<at::Half>()),
        table_rows,
        num_pos,
        static_cast<int>(mrope_h),
        static_cast<int>(mrope_w));
  } else {
    rope_cache_lookup_kernel<nv_bfloat16><<<1, threads, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(cos_table.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(sin_table.data_ptr<at::BFloat16>()),
        position_ids.data_ptr<int64_t>(),
        reinterpret_cast<nv_bfloat16*>(cos_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(sin_out.data_ptr<at::BFloat16>()),
        table_rows,
        num_pos,
        static_cast<int>(mrope_h),
        static_cast<int>(mrope_w));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {cos_out, sin_out};
}
