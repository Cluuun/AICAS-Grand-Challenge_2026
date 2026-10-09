#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kColsPerBlock = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ int unpack_signed_int4(uint8_t byte, int n) {
  int v = (n & 1) ? (byte >> 4) : (byte & 0x0f);
  return v >= 8 ? v - 16 : v;
}

__device__ __forceinline__ float load_scale_value(const void* ptr, int index, int dtype_tag) {
  if (dtype_tag == 0) {
    return static_cast<const float*>(ptr)[index];
  }
  if (dtype_tag == 1) {
    return __half2float(static_cast<const half*>(ptr)[index]);
  }
  return __bfloat162float(static_cast<const __nv_bfloat16*>(ptr)[index]);
}

__global__ void custom_int4_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const void* __restrict__ scales,
    const void* __restrict__ zeros,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int k,
    int n,
    int group_size,
    int scale_dtype_tag,
    bool has_zeros,
    bool has_bias) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int col = blockIdx.x * kColsPerBlock + warp_id * kWarpSize + lane;
  if (col >= n) {
    return;
  }

  const int packed_n = n / 2;
  const int groups = k / group_size;
  float acc = 0.0f;

  for (int g = 0; g < groups; ++g) {
    const int scale_idx = g * n + col;
    const float scale = load_scale_value(scales, scale_idx, scale_dtype_tag);
    const float zero = has_zeros ? load_scale_value(zeros, scale_idx, scale_dtype_tag) : 0.0f;
    const int k_base = g * group_size;
    #pragma unroll 4
    for (int kk = 0; kk < group_size; ++kk) {
      const int k_idx = k_base + kk;
      const uint8_t byte = packed_weight[k_idx * packed_n + (col >> 1)];
      const int q = unpack_signed_int4(byte, col);
      const float w = static_cast<float>(q) * scale + zero;
      acc += __bfloat162float(x[k_idx]) * w;
    }
  }

  if (has_bias) {
    acc += __bfloat162float(bias[col]);
  }
  out[col] = __float2bfloat16(acc);
}

__global__ void custom_int4_gemv_splitk_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_weight,
    const void* __restrict__ scales,
    const void* __restrict__ zeros,
    float* __restrict__ partial,
    int k,
    int n,
    int group_size,
    int groups,
    int split_k,
    int scale_dtype_tag,
    bool has_zeros) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int col = blockIdx.x * kColsPerBlock + warp_id * kWarpSize + lane;
  const int split = blockIdx.y;
  if (col >= n) {
    return;
  }

  const int group_begin = (split * groups) / split_k;
  const int group_end = ((split + 1) * groups) / split_k;
  const int packed_n = n / 2;
  float acc = 0.0f;

  for (int g = group_begin; g < group_end; ++g) {
    const int scale_idx = g * n + col;
    const float scale = load_scale_value(scales, scale_idx, scale_dtype_tag);
    const float zero = has_zeros ? load_scale_value(zeros, scale_idx, scale_dtype_tag) : 0.0f;
    const int k_base = g * group_size;
    #pragma unroll 4
    for (int kk = 0; kk < group_size; ++kk) {
      const int k_idx = k_base + kk;
      const uint8_t byte = packed_weight[k_idx * packed_n + (col >> 1)];
      const int q = unpack_signed_int4(byte, col);
      const float w = static_cast<float>(q) * scale + zero;
      acc += __bfloat162float(x[k_idx]) * w;
    }
  }

  partial[split * n + col] = acc;
}

__global__ void custom_int4_gemv_reduce_kernel(
    const float* __restrict__ partial,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int n,
    int split_k,
    bool has_bias) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  float acc = 0.0f;
  for (int s = 0; s < split_k; ++s) {
    acc += partial[s * n + tid];
  }
  if (has_bias) {
    acc += __bfloat162float(bias[tid]);
  }
  out[tid] = __float2bfloat16(acc);
}

int scale_dtype_tag(torch::ScalarType dtype) {
  if (dtype == torch::kFloat32) {
    return 0;
  }
  if (dtype == torch::kFloat16) {
    return 1;
  }
  if (dtype == torch::kBFloat16) {
    return 2;
  }
  TORCH_CHECK(false, "scales/zeros must be fp32/fp16/bf16");
}

}  // namespace

void run_custom_int4_gemv(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor bias,
    torch::Tensor out,
    int64_t group_size,
    bool has_zeros,
    bool has_bias) {
  x = x.contiguous();
  packed_weight = packed_weight.contiguous();
  scales = scales.contiguous();
  if (has_zeros) {
    zeros = zeros.contiguous();
  }
  if (has_bias) {
    bias = bias.contiguous();
  }

  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(
      packed_weight.scalar_type() == torch::kUInt8 || packed_weight.scalar_type() == torch::kInt8,
      "packed_weight must be uint8/int8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must be [1,K]");
  TORCH_CHECK(scales.dim() == 2, "scales must be [K/group_size,N]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1, "out must be [1,N]");
  const int64_t k = x.size(1);
  const int64_t n = scales.size(1);
  TORCH_CHECK(n % 2 == 0, "N must be even");
  TORCH_CHECK(group_size > 0 && k % group_size == 0, "K must be divisible by group_size");
  TORCH_CHECK(scales.size(0) == k / group_size, "scale group count mismatch");
  TORCH_CHECK(out.size(1) == n, "out N mismatch");
  TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must be [K,N/2]");
  TORCH_CHECK(packed_weight.size(0) == k, "packed_weight K mismatch");
  TORCH_CHECK(packed_weight.size(1) == n / 2, "packed_weight packed N mismatch");
  if (has_zeros) {
    TORCH_CHECK(zeros.is_cuda(), "zeros must be CUDA");
    TORCH_CHECK(zeros.scalar_type() == scales.scalar_type(), "zeros dtype must match scales");
    TORCH_CHECK(zeros.dim() == 2 && zeros.size(0) == scales.size(0) && zeros.size(1) == n, "zeros shape mismatch");
  }
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.scalar_type() == torch::kBFloat16, "bias must be bf16");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n, "bias must be [N]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const int blocks = static_cast<int>((n + kColsPerBlock - 1) / kColsPerBlock);
  const int threads = kColsPerBlock;
  custom_int4_gemv_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const uint8_t*>(packed_weight.data_ptr()),
      scales.data_ptr(),
      has_zeros ? zeros.data_ptr() : nullptr,
      has_bias ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()) : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int>(k),
      static_cast<int>(n),
      static_cast<int>(group_size),
      scale_dtype_tag(scales.scalar_type()),
      has_zeros,
      has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_custom_int4_gemv_splitk(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor bias,
    torch::Tensor partial,
    torch::Tensor out,
    int64_t group_size,
    int64_t split_k,
    bool has_zeros,
    bool has_bias) {
  x = x.contiguous();
  packed_weight = packed_weight.contiguous();
  scales = scales.contiguous();
  if (has_zeros) {
    zeros = zeros.contiguous();
  }
  if (has_bias) {
    bias = bias.contiguous();
  }

  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(partial.is_cuda(), "partial must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(partial.scalar_type() == torch::kFloat32, "partial must be fp32");
  TORCH_CHECK(
      packed_weight.scalar_type() == torch::kUInt8 || packed_weight.scalar_type() == torch::kInt8,
      "packed_weight must be uint8/int8");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must be [1,K]");
  TORCH_CHECK(scales.dim() == 2, "scales must be [K/group_size,N]");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == 1, "out must be [1,N]");
  const int64_t k = x.size(1);
  const int64_t n = scales.size(1);
  TORCH_CHECK(n % 2 == 0, "N must be even");
  TORCH_CHECK(group_size > 0 && k % group_size == 0, "K must be divisible by group_size");
  const int64_t groups = k / group_size;
  TORCH_CHECK(split_k > 0 && split_k <= groups, "split_k must be in [1, K/group_size]");
  TORCH_CHECK(scales.size(0) == groups, "scale group count mismatch");
  TORCH_CHECK(out.size(1) == n, "out N mismatch");
  TORCH_CHECK(partial.dim() == 2 && partial.size(0) == split_k && partial.size(1) == n, "partial must be [split_k,N]");
  TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must be [K,N/2]");
  TORCH_CHECK(packed_weight.size(0) == k, "packed_weight K mismatch");
  TORCH_CHECK(packed_weight.size(1) == n / 2, "packed_weight packed N mismatch");
  if (has_zeros) {
    TORCH_CHECK(zeros.is_cuda(), "zeros must be CUDA");
    TORCH_CHECK(zeros.scalar_type() == scales.scalar_type(), "zeros dtype must match scales");
    TORCH_CHECK(zeros.dim() == 2 && zeros.size(0) == scales.size(0) && zeros.size(1) == n, "zeros shape mismatch");
  }
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.scalar_type() == torch::kBFloat16, "bias must be bf16");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == n, "bias must be [N]");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const int n_blocks = static_cast<int>((n + kColsPerBlock - 1) / kColsPerBlock);
  const int threads = kColsPerBlock;
  const dim3 grid(n_blocks, static_cast<unsigned int>(split_k));
  custom_int4_gemv_splitk_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const uint8_t*>(packed_weight.data_ptr()),
      scales.data_ptr(),
      has_zeros ? zeros.data_ptr() : nullptr,
      static_cast<float*>(partial.data_ptr()),
      static_cast<int>(k),
      static_cast<int>(n),
      static_cast<int>(group_size),
      static_cast<int>(groups),
      static_cast<int>(split_k),
      scale_dtype_tag(scales.scalar_type()),
      has_zeros);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int reduce_threads = 256;
  const int reduce_blocks = static_cast<int>((n + reduce_threads - 1) / reduce_threads);
  custom_int4_gemv_reduce_kernel<<<reduce_blocks, reduce_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const float*>(partial.data_ptr()),
      has_bias ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()) : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int>(n),
      static_cast<int>(split_k),
      has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
