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
constexpr int kHiddenItems = 256;

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ uint4 load_v4_evict_last(const uint4* ptr) {
  uint4 out;
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
      : "=l"(policy));
  asm volatile(
      "ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %0, %1, %2, %3 }, [ %4 ], %5;"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
      : "l"(ptr), "l"(policy));
  return out;
}

__forceinline__ __device__ uint2 load_v2_evict_first(const uint2* ptr) {
  uint2 out;
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
      : "=l"(policy));
  asm volatile(
      "ld.global.L1::evict_first.L2::cache_hint.v2.b32 { %0, %1 }, [ %2 ], %3;"
      : "=r"(out.x), "=r"(out.y)
      : "l"(ptr), "l"(policy));
  return out;
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

__forceinline__ __device__ void fp8e4b15x4_to_half2_bits(
    uint32_t packed,
    uint32_t& lo,
    uint32_t& hi) {
  asm volatile(
      "{                                      \n"
      ".reg .b32 a<2>, b<2>;                  \n"
      "prmt.b32 a0, 0, %2, 0x5746;            \n"
      "and.b32 b0, a0, 0x7f007f00;            \n"
      "and.b32 b1, a0, 0x00ff00ff;            \n"
      "and.b32 a1, a0, 0x00800080;            \n"
      "shr.b32  b0, b0, 1;                    \n"
      "add.u32 b1, b1, a1;                    \n"
      "lop3.b32 %0, b0, 0x80008000, a0, 0xf8; \n"
      "shl.b32 %1, b1, 7;                     \n"
      "}                                      \n"
      : "=r"(lo), "=r"(hi)
      : "r"(packed));
}

__forceinline__ __device__ float dot_acc_half2_bits(float acc, uint32_t x_bits, uint32_t w_bits) {
  const float2 xf = __half22float2(half2_from_u32(x_bits));
  const float2 wf = __half22float2(half2_from_u32(w_bits));
  acc = fmaf(xf.x, wf.x, acc);
  acc = fmaf(xf.y, wf.y, acc);
  return acc;
}

__forceinline__ __device__ float dot8_half2_lowreg(uint4 x4, uint2 w2) {
  uint32_t w_lo;
  uint32_t w_hi;
  float acc = 0.0f;
  fp8e4b15x4_to_half2_bits(w2.x, w_lo, w_hi);
  acc = dot_acc_half2_bits(acc, x4.x, w_lo);
  acc = dot_acc_half2_bits(acc, x4.y, w_hi);
  fp8e4b15x4_to_half2_bits(w2.y, w_lo, w_hi);
  acc = dot_acc_half2_bits(acc, x4.z, w_lo);
  acc = dot_acc_half2_bits(acc, x4.w, w_hi);
  return acc;
}

__forceinline__ __device__ float dot16_half2_lowreg(uint4 x0, uint4 x1, uint4 w4) {
  uint32_t w_lo;
  uint32_t w_hi;
  float acc = 0.0f;
  fp8e4b15x4_to_half2_bits(w4.x, w_lo, w_hi);
  acc = dot_acc_half2_bits(acc, x0.x, w_lo);
  acc = dot_acc_half2_bits(acc, x0.y, w_hi);
  fp8e4b15x4_to_half2_bits(w4.y, w_lo, w_hi);
  acc = dot_acc_half2_bits(acc, x0.z, w_lo);
  acc = dot_acc_half2_bits(acc, x0.w, w_hi);
  fp8e4b15x4_to_half2_bits(w4.z, w_lo, w_hi);
  acc = dot_acc_half2_bits(acc, x1.x, w_lo);
  acc = dot_acc_half2_bits(acc, x1.y, w_hi);
  fp8e4b15x4_to_half2_bits(w4.w, w_lo, w_hi);
  acc = dot_acc_half2_bits(acc, x1.z, w_lo);
  acc = dot_acc_half2_bits(acc, x1.w, w_hi);
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

template <bool kCompact>
__global__ __launch_bounds__(1)
void lm_head_atomic_finalize_kernel(
    unsigned long long* __restrict__ atomic_state,
    const int64_t* __restrict__ token_ids,
    half* __restrict__ out_value,
    int64_t* __restrict__ out_token) {
  const unsigned long long packed = atomic_state[0];
  const uint32_t ordered = static_cast<uint32_t>(packed >> 32);
  const uint32_t inv_index = static_cast<uint32_t>(packed);
  const int64_t best_index = static_cast<int64_t>(0xffffffffu - inv_index);
  out_value[0] = __float2half_rn(unordered_float_bits(ordered));
  if constexpr (kCompact) {
    out_token[0] = token_ids[best_index];
  } else {
    out_token[0] = best_index;
  }
  atomic_state[0] = 0ull;
}

template <bool kCompact, int kLanesPerRow, bool kSharedX, bool kAtomic>
__global__
void lm_head_block_top1_fp8_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const int64_t* __restrict__ token_ids,
    const half* __restrict__ bias,
    float* __restrict__ partial_values,
    int64_t* __restrict__ partial_indices,
    unsigned long long* __restrict__ atomic_state,
    int n_rows,
    int k_in,
    int scale_blocks,
    bool has_bias) {
  __shared__ uint4 x_shared[kHiddenItems];
  __shared__ float row_values[kBlockN];
  __shared__ int64_t row_indices[kBlockN];

  if constexpr (kSharedX) {
    for (int item = threadIdx.x; item < kHiddenItems; item += blockDim.x) {
      x_shared[item] = load_v4_evict_last(reinterpret_cast<const uint4*>(x + item * 8));
    }
    __syncthreads();
  }

  const int tid = threadIdx.x;
  const int row_in_cta = tid / kLanesPerRow;
  const int lane = tid - row_in_cta * kLanesPerRow;
  const int row = static_cast<int>(blockIdx.x) * kBlockN + row_in_cta;
  constexpr int kElemsPerLane = kBlockK / kLanesPerRow;
  constexpr int kVecsPerLane = kElemsPerLane / 8;
  const int lane_k = lane * kElemsPerLane;

  float acc = 0.0f;
  float acc_alt = 0.0f;
  if (row < n_rows) {
#pragma unroll
    for (int block_id = 0; block_id < 16; ++block_id) {
      float block_acc = 0.0f;
#pragma unroll
      for (int g = 0; g < kVecsPerLane; ++g) {
        const int local_k = lane_k + g * 8;
        uint4 x4;
        if constexpr (kSharedX) {
          x4 = x_shared[block_id * 16 + local_k / 8];
        } else {
          x4 = load_v4_evict_last(reinterpret_cast<const uint4*>(x + block_id * kBlockK + local_k));
        }
        const uint2 w2 = load_v2_evict_first(
            reinterpret_cast<const uint2*>(
                weight_fp8 + static_cast<int64_t>(row) * k_in + block_id * kBlockK + local_k));
        block_acc += dot8_half2_lowreg(x4, w2);
      }
      uint32_t scale_bits = 0;
      if (lane == 0) {
        scale_bits = static_cast<uint32_t>(
            __half_as_ushort(load_half_evict_first(scales + static_cast<int64_t>(row) * scale_blocks + block_id)));
      }
      scale_bits = __shfl_sync(0xffffffffu, scale_bits, tid & ~(kLanesPerRow - 1), 32);
      const float scaled = block_acc * __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits)));
      if ((block_id & 1) == 0) {
        acc += scaled;
      } else {
        acc_alt += scaled;
      }
    }
  }
  acc = row_sum<kLanesPerRow>(acc + acc_alt);

  if (lane == 0) {
    float value = row < n_rows ? acc : -INFINITY;
    row_values[row_in_cta] = value;
    row_indices[row_in_cta] = row < n_rows ? static_cast<int64_t>(row) : INT64_MAX;
  }
  __syncthreads();

  if (tid < 32) {
    float best_value = tid < kBlockN ? row_values[tid] : -INFINITY;
    int64_t best_index = tid < kBlockN ? row_indices[tid] : INT64_MAX;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_value = __shfl_down_sync(0xffffffffu, best_value, offset);
      const int64_t other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
      best_pair_update(other_value, other_index, best_value, best_index);
    }
    if (tid == 0) {
      if constexpr (kAtomic) {
        if (best_index != INT64_MAX) {
          atomicMax(atomic_state, pack_best_pair(best_value, best_index));
        }
      } else {
        partial_values[blockIdx.x] = best_value;
        partial_indices[blockIdx.x] = best_index;
      }
    }
  }
}

template <bool kCompact, bool kAtomic>
__global__
void lm_head_block_top1_fp8_dualrow_tiled_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const int64_t* __restrict__ token_ids,
    const half* __restrict__ bias,
    float* __restrict__ partial_values,
    int64_t* __restrict__ partial_indices,
    unsigned long long* __restrict__ atomic_state,
    int n_rows,
    int k_in,
    int scale_blocks,
    bool has_bias) {
  __shared__ uint4 x_tile[16];
  __shared__ float group_values[16];
  __shared__ int64_t group_indices[16];

  const int tid = threadIdx.x;
  const int lane = tid & 7;
  const int group = (tid >> 3) & 15;
  const int row0 = static_cast<int>(blockIdx.x) * kBlockN + group;
  const int row1 = row0 + 16;
  const int lane_k = lane * 16;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
#pragma unroll 1
  for (int block_id = 0; block_id < 16; ++block_id) {
    if (tid < 16) {
      x_tile[tid] = load_v4_evict_last(reinterpret_cast<const uint4*>(x + block_id * kBlockK + tid * 8));
    }
    __syncthreads();

    const uint4 x0 = x_tile[lane * 2];
    const uint4 x1 = x_tile[lane * 2 + 1];
    const int local_k = block_id * kBlockK + lane_k;

    float block_acc0 = 0.0f;
    if (row0 < n_rows) {
      const uint4 w0 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(weight_fp8 + static_cast<int64_t>(row0) * k_in + local_k));
      block_acc0 = dot16_half2_lowreg(x0, x1, w0);
    }

    float block_acc1 = 0.0f;
    if (row1 < n_rows) {
      const uint4 w1 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(weight_fp8 + static_cast<int64_t>(row1) * k_in + local_k));
      block_acc1 = dot16_half2_lowreg(x0, x1, w1);
    }

    uint32_t scale_bits0 = 0;
    uint32_t scale_bits1 = 0;
    if (lane == 0) {
      if (row0 < n_rows) {
        scale_bits0 = static_cast<uint32_t>(
            __half_as_ushort(load_half_evict_first(scales + static_cast<int64_t>(row0) * scale_blocks + block_id)));
      }
      if (row1 < n_rows) {
        scale_bits1 = static_cast<uint32_t>(
            __half_as_ushort(load_half_evict_first(scales + static_cast<int64_t>(row1) * scale_blocks + block_id)));
      }
    }
    const int group_base = tid & ~7;
    scale_bits0 = __shfl_sync(0xffffffffu, scale_bits0, group_base, 32);
    scale_bits1 = __shfl_sync(0xffffffffu, scale_bits1, group_base, 32);
    acc0 += block_acc0 * __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits0)));
    acc1 += block_acc1 * __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits1)));
    __syncthreads();
  }

  acc0 = row_sum<8>(acc0);
  acc1 = row_sum<8>(acc1);

  if (lane == 0) {
    float best_value = -INFINITY;
    int64_t best_index = INT64_MAX;
    if (row0 < n_rows) {
      float value0 = acc0;
      best_pair_update(value0, static_cast<int64_t>(row0), best_value, best_index);
    }
    if (row1 < n_rows) {
      float value1 = acc1;
      best_pair_update(value1, static_cast<int64_t>(row1), best_value, best_index);
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
    if (tid == 0) {
      if constexpr (kAtomic) {
        if (best_index != INT64_MAX) {
          atomicMax(atomic_state, pack_best_pair(best_value, best_index));
        }
      } else {
        partial_values[blockIdx.x] = best_value;
        partial_indices[blockIdx.x] = best_index;
      }
    }
  }
}

template <bool kCompact, bool kAtomic>
__global__
void lm_head_block_top1_fp8_dualrow_directx_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const int64_t* __restrict__ token_ids,
    const half* __restrict__ bias,
    float* __restrict__ partial_values,
    int64_t* __restrict__ partial_indices,
    unsigned long long* __restrict__ atomic_state,
    int n_rows,
    int k_in,
    int scale_blocks,
    bool has_bias) {
  __shared__ float group_values[16];
  __shared__ int64_t group_indices[16];

  const int tid = threadIdx.x;
  const int lane = tid & 7;
  const int group = (tid >> 3) & 15;
  const int row0 = static_cast<int>(blockIdx.x) * kBlockN + group;
  const int row1 = row0 + 16;
  const int lane_k = lane * 16;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
#pragma unroll
  for (int block_id = 0; block_id < 16; ++block_id) {
    const int local_k = block_id * kBlockK + lane_k;
    const uint4 x0 = load_v4_evict_last(reinterpret_cast<const uint4*>(x + local_k));
    const uint4 x1 = load_v4_evict_last(reinterpret_cast<const uint4*>(x + local_k + 8));

    float block_acc0 = 0.0f;
    if (row0 < n_rows) {
      const uint4 w0 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(weight_fp8 + static_cast<int64_t>(row0) * k_in + local_k));
      block_acc0 = dot16_half2_lowreg(x0, x1, w0);
    }

    float block_acc1 = 0.0f;
    if (row1 < n_rows) {
      const uint4 w1 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(weight_fp8 + static_cast<int64_t>(row1) * k_in + local_k));
      block_acc1 = dot16_half2_lowreg(x0, x1, w1);
    }

    uint32_t scale_bits0 = 0;
    uint32_t scale_bits1 = 0;
    if (lane == 0) {
      if (row0 < n_rows) {
        scale_bits0 = static_cast<uint32_t>(
            __half_as_ushort(load_half_evict_first(scales + static_cast<int64_t>(row0) * scale_blocks + block_id)));
      }
      if (row1 < n_rows) {
        scale_bits1 = static_cast<uint32_t>(
            __half_as_ushort(load_half_evict_first(scales + static_cast<int64_t>(row1) * scale_blocks + block_id)));
      }
    }
    const int group_base = tid & ~7;
    scale_bits0 = __shfl_sync(0xffffffffu, scale_bits0, group_base, 32);
    scale_bits1 = __shfl_sync(0xffffffffu, scale_bits1, group_base, 32);
    acc0 += block_acc0 * __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits0)));
    acc1 += block_acc1 * __half2float(__ushort_as_half(static_cast<unsigned short>(scale_bits1)));
  }

  acc0 = row_sum<8>(acc0);
  acc1 = row_sum<8>(acc1);

  if (lane == 0) {
    float best_value = -INFINITY;
    int64_t best_index = INT64_MAX;
    if (row0 < n_rows) {
      float value0 = acc0;
      best_pair_update(value0, static_cast<int64_t>(row0), best_value, best_index);
    }
    if (row1 < n_rows) {
      float value1 = acc1;
      best_pair_update(value1, static_cast<int64_t>(row1), best_value, best_index);
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
    if (tid == 0) {
      if constexpr (kAtomic) {
        if (best_index != INT64_MAX) {
          atomicMax(atomic_state, pack_best_pair(best_value, best_index));
        }
      } else {
        partial_values[blockIdx.x] = best_value;
        partial_indices[blockIdx.x] = best_index;
      }
    }
  }
}

template <int kReduceBlock>
__global__ __launch_bounds__(256)
void lm_head_partial_reduce_kernel(
    const float* __restrict__ partial_values,
    const int64_t* __restrict__ partial_indices,
    float* __restrict__ stage_values,
    int64_t* __restrict__ stage_indices,
    int num_blocks) {
  __shared__ float values[256];
  __shared__ int64_t indices[256];

  const int tid = threadIdx.x;
  float best_value = -INFINITY;
  int64_t best_index = INT64_MAX;
#pragma unroll
  for (int i = 0; i < kReduceBlock / 256; ++i) {
    const int idx = static_cast<int>(blockIdx.x) * kReduceBlock + i * 256 + tid;
    if (idx < num_blocks) {
      best_pair_update(partial_values[idx], partial_indices[idx], best_value, best_index);
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      best_pair_update(values[tid + stride], indices[tid + stride], values[tid], indices[tid]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    stage_values[blockIdx.x] = values[0];
    stage_indices[blockIdx.x] = indices[0];
  }
}

__global__ __launch_bounds__(256)
void lm_head_final_reduce_full_kernel(
    const float* __restrict__ stage_values,
    const int64_t* __restrict__ stage_indices,
    half* __restrict__ out_value,
    int64_t* __restrict__ out_token,
    int num_stage_blocks) {
  __shared__ float values[256];
  __shared__ int64_t indices[256];

  const int tid = threadIdx.x;
  float best_value = -INFINITY;
  int64_t best_index = INT64_MAX;
  for (int idx = tid; idx < num_stage_blocks; idx += 256) {
    best_pair_update(stage_values[idx], stage_indices[idx], best_value, best_index);
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      best_pair_update(values[tid + stride], indices[tid + stride], values[tid], indices[tid]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_value[0] = __float2half_rn(values[0]);
    out_token[0] = indices[0];
  }
}

__global__ __launch_bounds__(256)
void lm_head_final_reduce_compact_kernel(
    const float* __restrict__ stage_values,
    const int64_t* __restrict__ stage_indices,
    const int64_t* __restrict__ token_ids,
    half* __restrict__ out_value,
    int64_t* __restrict__ out_token,
    int num_stage_blocks) {
  __shared__ float values[256];
  __shared__ int64_t indices[256];

  const int tid = threadIdx.x;
  float best_value = -INFINITY;
  int64_t best_index = INT64_MAX;
  for (int idx = tid; idx < num_stage_blocks; idx += 256) {
    best_pair_update(stage_values[idx], stage_indices[idx], best_value, best_index);
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      best_pair_update(values[tid + stride], indices[tid + stride], values[tid], indices[tid]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_value[0] = __float2half_rn(values[0]);
    out_token[0] = token_ids[indices[0]];
  }
}

template <bool kCompact>
__global__ __launch_bounds__(256)
void lm_head_direct_reduce_kernel(
    const float* __restrict__ partial_values,
    const int64_t* __restrict__ partial_indices,
    const int64_t* __restrict__ token_ids,
    half* __restrict__ out_value,
    int64_t* __restrict__ out_token,
    int num_blocks) {
  __shared__ float values[256];
  __shared__ int64_t indices[256];

  const int tid = threadIdx.x;
  float best_value = -INFINITY;
  int64_t best_index = INT64_MAX;
  for (int idx = tid; idx < num_blocks; idx += 256) {
    best_pair_update(partial_values[idx], partial_indices[idx], best_value, best_index);
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      best_pair_update(values[tid + stride], indices[tid + stride], values[tid], indices[tid]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_value[0] = __float2half_rn(values[0]);
    const int64_t best_index = indices[0];
    if constexpr (kCompact) {
      out_token[0] = token_ids[best_index];
    } else {
      out_token[0] = best_index;
    }
  }
}

void check_common(
    torch::Tensor hidden,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor stage_values,
    torch::Tensor stage_indices,
    torch::Tensor out_value,
    torch::Tensor out_token,
    int64_t reduce_block) {
  CHECK_CUDA(hidden);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CUDA(bias);
  CHECK_CUDA(partial_values);
  CHECK_CUDA(partial_indices);
  CHECK_CUDA(stage_values);
  CHECK_CUDA(stage_indices);
  CHECK_CUDA(out_value);
  CHECK_CUDA(out_token);
  CHECK_CONTIGUOUS(hidden);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(bias);
  CHECK_CONTIGUOUS(partial_values);
  CHECK_CONTIGUOUS(partial_indices);
  CHECK_CONTIGUOUS(stage_values);
  CHECK_CONTIGUOUS(stage_indices);
  CHECK_CONTIGUOUS(out_value);
  CHECK_CONTIGUOUS(out_token);
  CHECK_HALF(hidden);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  CHECK_HALF(bias);
  CHECK_FLOAT(partial_values);
  CHECK_LONG(partial_indices);
  CHECK_FLOAT(stage_values);
  CHECK_LONG(stage_indices);
  CHECK_HALF(out_value);
  CHECK_LONG(out_token);
  TORCH_CHECK(hidden.numel() == weight_fp8.size(1), "hidden and weight K mismatch");
  TORCH_CHECK(weight_fp8.size(1) == 2048, "CUDA lm_head FP8 top1 currently expects K=2048");
  TORCH_CHECK(scales.size(0) == weight_fp8.size(0), "scale rows mismatch");
  TORCH_CHECK(scales.size(1) == weight_fp8.size(1) / kBlockK, "scale K blocks mismatch");
  TORCH_CHECK(reduce_block == 512 || reduce_block == 1024, "reduce_block must be 512 or 1024");
}

template <bool kCompact>
void launch_block_top1(
    torch::Tensor hidden,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor token_ids,
    torch::Tensor bias,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor atomic_state,
    bool has_bias,
    int64_t variant) {
  const int n_rows = static_cast<int>(weight_fp8.size(0));
  const int k_in = static_cast<int>(weight_fp8.size(1));
  const int scale_blocks = k_in / kBlockK;
  const int num_blocks = (n_rows + kBlockN - 1) / kBlockN;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto x = reinterpret_cast<const half*>(hidden.data_ptr<at::Half>());
  const auto w = weight_fp8.data_ptr<uint8_t>();
  const auto s = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto ids = token_ids.data_ptr<int64_t>();
  const auto b = reinterpret_cast<const half*>(bias.data_ptr<at::Half>());
  auto pv = partial_values.data_ptr<float>();
  auto pi = partial_indices.data_ptr<int64_t>();
  auto atomic = reinterpret_cast<unsigned long long*>(atomic_state.data_ptr<int64_t>());
  switch (static_cast<int>(variant)) {
    case 0:
      lm_head_block_top1_fp8_kernel<kCompact, 4, true, false><<<num_blocks, 128, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 1:
      lm_head_block_top1_fp8_kernel<kCompact, 4, false, false><<<num_blocks, 128, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 2:
      lm_head_block_top1_fp8_kernel<kCompact, 8, false, false><<<num_blocks, 256, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 3:
      lm_head_block_top1_fp8_kernel<kCompact, 16, false, false><<<num_blocks, 512, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 5:
      lm_head_block_top1_fp8_kernel<kCompact, 8, true, false><<<num_blocks, 256, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 6:
      lm_head_block_top1_fp8_kernel<kCompact, 16, true, false><<<num_blocks, 512, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 23:
      lm_head_block_top1_fp8_kernel<kCompact, 16, false, true><<<num_blocks, 512, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 25:
      lm_head_block_top1_fp8_kernel<kCompact, 8, true, true><<<num_blocks, 256, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 53:
      lm_head_block_top1_fp8_kernel<kCompact, 8, true, true><<<num_blocks, 256, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 43:
      lm_head_block_top1_fp8_dualrow_tiled_kernel<kCompact, true><<<num_blocks, 128, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 44:
      lm_head_block_top1_fp8_dualrow_directx_kernel<kCompact, true><<<num_blocks, 128, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    case 52:
      lm_head_block_top1_fp8_dualrow_directx_kernel<kCompact, true><<<num_blocks, 128, 0, stream>>>(
          x, w, s, ids, b, pv, pi, atomic, n_rows, k_in, scale_blocks, has_bias);
      break;
    default:
      TORCH_CHECK(false, "unsupported lm_head FP8 CUDA variant");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce(
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor stage_values,
    torch::Tensor stage_indices,
    int64_t n_rows,
    int64_t reduce_block) {
  const int num_blocks = (static_cast<int>(n_rows) + kBlockN - 1) / kBlockN;
  const int num_stage_blocks = (num_blocks + static_cast<int>(reduce_block) - 1) / static_cast<int>(reduce_block);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (reduce_block == 512) {
    lm_head_partial_reduce_kernel<512><<<num_stage_blocks, 256, 0, stream>>>(
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int64_t>(),
        stage_values.data_ptr<float>(),
        stage_indices.data_ptr<int64_t>(),
        num_blocks);
  } else {
    lm_head_partial_reduce_kernel<1024><<<num_stage_blocks, 256, 0, stream>>>(
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int64_t>(),
        stage_values.data_ptr<float>(),
        stage_indices.data_ptr<int64_t>(),
        num_blocks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void lm_head_top1_fp8_full(
    torch::Tensor hidden,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor stage_values,
    torch::Tensor stage_indices,
    torch::Tensor out_value,
    torch::Tensor out_token,
    bool has_bias,
    int64_t reduce_block,
    int64_t variant) {
  const bool atomic_reduce = variant >= 20;
  const bool direct_reduce = variant >= 10 && variant < 20;
  const int64_t block_variant = atomic_reduce ? variant : (direct_reduce ? variant - 10 : variant);
  check_common(
      hidden,
      weight_fp8,
      scales,
      bias,
      partial_values,
      partial_indices,
      stage_values,
      stage_indices,
      out_value,
      out_token,
      reduce_block);
  TORCH_CHECK(!has_bias, "CUDA lm_head FP8 top1 is specialized for Qwen3-VL lm_head bias=False");
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool skip_atomic_init = variant == 53;
  if (atomic_reduce && !skip_atomic_init) {
    lm_head_atomic_init_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(stage_indices.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  launch_block_top1<false>(
      hidden, weight_fp8, scales, out_token, bias, partial_values, partial_indices, stage_indices, has_bias, block_variant);
  const int num_blocks = (static_cast<int>(weight_fp8.size(0)) + kBlockN - 1) / kBlockN;
  if (atomic_reduce) {
    lm_head_atomic_finalize_kernel<false><<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(stage_indices.data_ptr<int64_t>()),
        out_token.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out_value.data_ptr<at::Half>()),
        out_token.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (direct_reduce) {
    lm_head_direct_reduce_kernel<false><<<1, 256, 0, stream>>>(
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int64_t>(),
        out_token.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out_value.data_ptr<at::Half>()),
        out_token.data_ptr<int64_t>(),
        num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    launch_reduce(partial_values, partial_indices, stage_values, stage_indices, weight_fp8.size(0), reduce_block);
    const int num_stage_blocks = (num_blocks + static_cast<int>(reduce_block) - 1) / static_cast<int>(reduce_block);
    lm_head_final_reduce_full_kernel<<<1, 256, 0, stream>>>(
        stage_values.data_ptr<float>(),
        stage_indices.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out_value.data_ptr<at::Half>()),
        out_token.data_ptr<int64_t>(),
        num_stage_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void lm_head_top1_fp8_compact(
    torch::Tensor hidden,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor token_ids,
    torch::Tensor bias,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor stage_values,
    torch::Tensor stage_indices,
    torch::Tensor out_value,
    torch::Tensor out_token,
    bool has_bias,
    int64_t reduce_block,
    int64_t variant) {
  const bool atomic_reduce = variant >= 20;
  const bool direct_reduce = variant >= 10 && variant < 20;
  const int64_t block_variant = atomic_reduce ? variant : (direct_reduce ? variant - 10 : variant);
  check_common(
      hidden,
      weight_fp8,
      scales,
      bias,
      partial_values,
      partial_indices,
      stage_values,
      stage_indices,
      out_value,
      out_token,
      reduce_block);
  TORCH_CHECK(!has_bias, "CUDA lm_head FP8 top1 is specialized for Qwen3-VL lm_head bias=False");
  CHECK_CUDA(token_ids);
  CHECK_CONTIGUOUS(token_ids);
  CHECK_LONG(token_ids);
  TORCH_CHECK(token_ids.numel() == weight_fp8.size(0), "token_ids and compact weight size mismatch");
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool skip_atomic_init = variant == 52;
  if (atomic_reduce && !skip_atomic_init) {
    lm_head_atomic_init_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(stage_indices.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  launch_block_top1<true>(
      hidden, weight_fp8, scales, token_ids, bias, partial_values, partial_indices, stage_indices, has_bias, block_variant);
  const int num_blocks = (static_cast<int>(weight_fp8.size(0)) + kBlockN - 1) / kBlockN;
  if (atomic_reduce) {
    lm_head_atomic_finalize_kernel<true><<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(stage_indices.data_ptr<int64_t>()),
        token_ids.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out_value.data_ptr<at::Half>()),
        out_token.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (direct_reduce) {
    lm_head_direct_reduce_kernel<true><<<1, 256, 0, stream>>>(
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int64_t>(),
        token_ids.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out_value.data_ptr<at::Half>()),
        out_token.data_ptr<int64_t>(),
        num_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    launch_reduce(partial_values, partial_indices, stage_values, stage_indices, weight_fp8.size(0), reduce_block);
    const int num_stage_blocks = (num_blocks + static_cast<int>(reduce_block) - 1) / static_cast<int>(reduce_block);
    lm_head_final_reduce_compact_kernel<<<1, 256, 0, stream>>>(
        stage_values.data_ptr<float>(),
        stage_indices.data_ptr<int64_t>(),
        token_ids.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out_value.data_ptr<at::Half>()),
        out_token.data_ptr<int64_t>(),
        num_stage_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lm_head_top1_fp8_full", &lm_head_top1_fp8_full, "CUDA FP8 e4b15 lm_head top1 over full vocab");
  m.def(
      "lm_head_top1_fp8_compact",
      &lm_head_top1_fp8_compact,
      "CUDA FP8 e4b15 lm_head top1 over compact shortlist");
}
