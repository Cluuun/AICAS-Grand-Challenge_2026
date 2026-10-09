#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be fp32")
#define CHECK_BYTE(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")
#define CHECK_LONG(x) TORCH_CHECK((x).scalar_type() == at::kLong, #x " must be int64")

constexpr int kBlockN = 32;
constexpr int kBlockK = 128;
constexpr int kHiddenSize = 2048;
constexpr int kScaleBlocks = kHiddenSize / kBlockK;
constexpr int kPackedBytesPerBlock = kBlockK / 2;
constexpr int kPackedWordsPerBlock = kPackedBytesPerBlock / 4;

__forceinline__ __device__ uint32_t load_u32_cg(const uint32_t* ptr) {
  return __ldcg(ptr);
}

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ half2 load_half2_evict_last(const half2* ptr) {
  uint32_t out;
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
      : "=l"(policy));
  asm volatile(
      "ld.global.L1::evict_last.L2::cache_hint.b32 { %0 }, [ %1 ], %2;"
      : "=r"(out)
      : "l"(ptr), "l"(policy));
  return half2_from_u32(out);
}

__forceinline__ __device__ half load_half_evict_first(const half* ptr) {
  uint16_t out;
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
      : "=l"(policy));
  asm volatile(
      "ld.global.L1::evict_first.L2::cache_hint.b16 { %0 }, [ %1 ], %2;"
      : "=h"(out)
      : "l"(ptr), "l"(policy));
  return *reinterpret_cast<half*>(&out);
}

__forceinline__ __device__ float dot_acc_int4_pair(float acc, uint32_t byte_value, half2 x_pair) {
  const float2 xv = __half22float2(x_pair);
  const float q0 = static_cast<float>(static_cast<int>(byte_value & 0x0fu) - 8);
  const float q1 = static_cast<float>(static_cast<int>((byte_value >> 4) & 0x0fu) - 8);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  return acc;
}

__forceinline__ __device__ float dot8_int4xhalf(uint32_t packed, const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed >> 8) & 0xffu, x2[1]);
  acc = dot_acc_int4_pair(acc, (packed >> 16) & 0xffu, x2[2]);
  acc = dot_acc_int4_pair(acc, (packed >> 24) & 0xffu, x2[3]);
  return acc;
}

template <int kLanesPerRow>
__forceinline__ __device__ float row_sum(float value) {
#pragma unroll
  for (int delta = kLanesPerRow / 2; delta > 0; delta >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, delta, 32);
  }
  return value;
}

__forceinline__ __device__ void best_pair_update(float value, int64_t index, float& best_value, int64_t& best_index) {
  if (value > best_value || (value == best_value && index < best_index)) {
    best_value = value;
    best_index = index;
  }
}

__forceinline__ __device__ uint32_t ordered_float_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__forceinline__ __device__ float unordered_float_bits(uint32_t ordered) {
  const uint32_t bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

__forceinline__ __device__ unsigned long long pack_best_pair(float value, int64_t index) {
  const uint32_t ordered = ordered_float_bits(value);
  const uint32_t inv_index = 0xffffffffu - static_cast<uint32_t>(index);
  return (static_cast<unsigned long long>(ordered) << 32) | static_cast<unsigned long long>(inv_index);
}

__global__ __launch_bounds__(1)
void lm_head_atomic_init_kernel(unsigned long long* __restrict__ atomic_state) {
  atomic_state[0] = 0ull;
}

__global__ __launch_bounds__(1)
void lm_head_atomic_finalize_compact_kernel(
    const unsigned long long* __restrict__ atomic_state,
    const int64_t* __restrict__ token_ids,
    half* __restrict__ out_value,
    int64_t* __restrict__ out_token) {
  const unsigned long long packed = atomic_state[0];
  const uint32_t ordered = static_cast<uint32_t>(packed >> 32);
  const uint32_t inv_index = static_cast<uint32_t>(packed);
  const int64_t best_index = static_cast<int64_t>(0xffffffffu - inv_index);
  out_value[0] = __float2half_rn(unordered_float_bits(ordered));
  out_token[0] = token_ids[best_index];
}

__global__
void lm_head_block_top1_int4_dualrow_directx_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const int64_t* __restrict__ token_ids,
    float* __restrict__ partial_values,
    int64_t* __restrict__ partial_indices,
    unsigned long long* __restrict__ atomic_state,
    int n_rows) {
  __shared__ float group_values[16];
  __shared__ int64_t group_indices[16];

  const int tid = threadIdx.x;
  const int lane = tid & 7;
  const int group = (tid >> 3) & 15;
  const int row0 = static_cast<int>(blockIdx.x) * kBlockN + group;
  const int row1 = row0 + 16;
  const int lane_pair = lane * 8;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const int x_pair = block_k * (kBlockK / 2) + lane_pair;
    const int block_base = block_k * n_rows * kPackedWordsPerBlock;
    float block_acc0 = 0.0f;
    if (row0 < n_rows) {
      const uint32_t* row_w32 = weight_w32 + block_base + row0 * kPackedWordsPerBlock + lane * 2;
      block_acc0 += dot8_int4xhalf(load_u32_cg(row_w32), x2 + x_pair);
      block_acc0 += dot8_int4xhalf(load_u32_cg(row_w32 + 1), x2 + x_pair + 4);
    }
    float block_acc1 = 0.0f;
    if (row1 < n_rows) {
      const uint32_t* row_w32 = weight_w32 + block_base + row1 * kPackedWordsPerBlock + lane * 2;
      block_acc1 += dot8_int4xhalf(load_u32_cg(row_w32), x2 + x_pair);
      block_acc1 += dot8_int4xhalf(load_u32_cg(row_w32 + 1), x2 + x_pair + 4);
    }
    uint32_t scale_bits0 = 0;
    uint32_t scale_bits1 = 0;
    if (lane == 0) {
      if (row0 < n_rows) {
        scale_bits0 = static_cast<uint32_t>(
            __half_as_ushort(load_half_evict_first(scales + block_k * n_rows + row0)));
      }
      if (row1 < n_rows) {
        scale_bits1 = static_cast<uint32_t>(
            __half_as_ushort(load_half_evict_first(scales + block_k * n_rows + row1)));
      }
    }
    const int group_base = tid & ~7;
    scale_bits0 = __shfl_sync(0xffffffffu, scale_bits0, group_base, 32);
    scale_bits1 = __shfl_sync(0xffffffffu, scale_bits1, group_base, 32);
    acc0 = fmaf(block_acc0, __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits0))), acc0);
    acc1 = fmaf(block_acc1, __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits1))), acc1);
  }

  acc0 = row_sum<8>(acc0);
  acc1 = row_sum<8>(acc1);

  if (lane == 0) {
    float best_value = -INFINITY;
    int64_t best_index = INT64_MAX;
    if (row0 < n_rows) {
      best_pair_update(acc0, static_cast<int64_t>(row0), best_value, best_index);
    }
    if (row1 < n_rows) {
      best_pair_update(acc1, static_cast<int64_t>(row1), best_value, best_index);
    }
    group_values[group] = best_value;
    group_indices[group] = best_index;
  }
  __syncthreads();

  if (tid < 32) {
    float best_value = tid < 16 ? group_values[tid] : -INFINITY;
    int64_t best_index = tid < 16 ? group_indices[tid] : INT64_MAX;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
      const int64_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
      best_pair_update(other_value, other_index, best_value, best_index);
    }
    if (tid == 0 && best_index != INT64_MAX) {
      atomicMax(atomic_state, pack_best_pair(best_value, best_index));
    }
  }
}

void check_common(
    const torch::Tensor& hidden,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& token_ids,
    const torch::Tensor& partial_values,
    const torch::Tensor& partial_indices,
    const torch::Tensor& atomic_state,
    const torch::Tensor& out_value,
    const torch::Tensor& out_token) {
  CHECK_CUDA(hidden);
  CHECK_CUDA(weight_int4);
  CHECK_CUDA(scales);
  CHECK_CUDA(token_ids);
  CHECK_CUDA(partial_values);
  CHECK_CUDA(partial_indices);
  CHECK_CUDA(atomic_state);
  CHECK_CUDA(out_value);
  CHECK_CUDA(out_token);
  CHECK_CONTIGUOUS(hidden);
  CHECK_CONTIGUOUS(weight_int4);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(token_ids);
  CHECK_CONTIGUOUS(partial_values);
  CHECK_CONTIGUOUS(partial_indices);
  CHECK_CONTIGUOUS(atomic_state);
  CHECK_CONTIGUOUS(out_value);
  CHECK_CONTIGUOUS(out_token);
  CHECK_HALF(hidden);
  CHECK_BYTE(weight_int4);
  CHECK_HALF(scales);
  CHECK_LONG(token_ids);
  CHECK_FLOAT(partial_values);
  CHECK_LONG(partial_indices);
  CHECK_LONG(atomic_state);
  CHECK_HALF(out_value);
  CHECK_LONG(out_token);
  TORCH_CHECK(hidden.numel() == kHiddenSize, "CUDA lm_head INT4 compact expects K=2048");
  TORCH_CHECK(weight_int4.dim() == 3, "weight_int4 must be [16, n_rows, 64]");
  TORCH_CHECK(weight_int4.size(0) == kScaleBlocks, "weight_int4 first dimension must be 16");
  TORCH_CHECK(weight_int4.size(2) == kPackedBytesPerBlock, "weight_int4 last dimension must be 64");
  TORCH_CHECK(scales.dim() == 2, "scales must be [16, n_rows]");
  TORCH_CHECK(scales.size(0) == kScaleBlocks, "scales first dimension must be 16");
  TORCH_CHECK(scales.size(1) == weight_int4.size(1), "scales rows mismatch");
  TORCH_CHECK(token_ids.numel() == weight_int4.size(1), "token_ids and compact weight size mismatch");
  TORCH_CHECK(partial_values.numel() >= (weight_int4.size(1) + kBlockN - 1) / kBlockN,
              "partial_values workspace is too small");
  TORCH_CHECK(partial_indices.numel() >= (weight_int4.size(1) + kBlockN - 1) / kBlockN,
              "partial_indices workspace is too small");
  TORCH_CHECK(atomic_state.numel() >= 1, "atomic_state workspace is too small");
}

void lm_head_top1_int4_compact(
    torch::Tensor hidden,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor token_ids,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor atomic_state,
    torch::Tensor out_value,
    torch::Tensor out_token) {
  check_common(hidden, weight_int4, scales, token_ids, partial_values, partial_indices, atomic_state, out_value, out_token);
  auto stream = at::cuda::getCurrentCUDAStream();
  lm_head_atomic_init_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<unsigned long long*>(atomic_state.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int n_rows = static_cast<int>(weight_int4.size(1));
  const int num_blocks = (n_rows + kBlockN - 1) / kBlockN;
  lm_head_block_top1_int4_dualrow_directx_kernel<<<num_blocks, 128, 0, stream>>>(
      reinterpret_cast<const half*>(hidden.data_ptr<at::Half>()),
      weight_int4.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      token_ids.data_ptr<int64_t>(),
      partial_values.data_ptr<float>(),
      partial_indices.data_ptr<int64_t>(),
      reinterpret_cast<unsigned long long*>(atomic_state.data_ptr<int64_t>()),
      n_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  lm_head_atomic_finalize_compact_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const unsigned long long*>(atomic_state.data_ptr<int64_t>()),
      token_ids.data_ptr<int64_t>(),
      reinterpret_cast<half*>(out_value.data_ptr<at::Half>()),
      out_token.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lm_head_top1_int4_compact", &lm_head_top1_int4_compact, "CUDA symmetric INT4 compact lm_head top1");
}
