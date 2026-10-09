#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

namespace cg = cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be fp32")
#define CHECK_BYTE(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")

constexpr int kHiddenSize = 2048;
constexpr int kBlockN = 16;
constexpr int kBlockK = 128;
constexpr int kThreads = 256;
constexpr int kLanesPerRow = 16;
constexpr int kQueryHeads = 16;
constexpr int kHeadDim = 128;

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ uint32_t u32_from_half2(half2 value) {
  return *reinterpret_cast<uint32_t*>(&value);
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

__forceinline__ __device__ uint4 load_v4_evict_first(const uint4* ptr) {
  uint4 out;
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
      : "=l"(policy));
  asm volatile(
      "ld.global.L1::evict_first.L2::cache_hint.v4.b32 { %0, %1, %2, %3 }, [ %4 ], %5;"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
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

__forceinline__ __device__ half broadcast_halfwarp_scale(const half* ptr, int lane) {
  uint32_t bits = 0;
  if (lane == 0) {
    bits = static_cast<uint32_t>(__half_as_ushort(load_half_evict_first(ptr)));
  }
  const int src_lane = threadIdx.x & 16;
  bits = __shfl_sync(0xffffffffu, bits, src_lane, 32);
  return __ushort_as_half(static_cast<unsigned short>(bits));
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

__forceinline__ __device__ float dot8_half2(
    uint4 x4,
    uint2 w2) {
  uint32_t w01_bits;
  uint32_t w23_bits;
  uint32_t w45_bits;
  uint32_t w67_bits;
  fp8e4b15x4_to_half2_bits(w2.x, w01_bits, w23_bits);
  fp8e4b15x4_to_half2_bits(w2.y, w45_bits, w67_bits);

  float2 x01 = __half22float2(half2_from_u32(x4.x));
  float2 x23 = __half22float2(half2_from_u32(x4.y));
  float2 x45 = __half22float2(half2_from_u32(x4.z));
  float2 x67 = __half22float2(half2_from_u32(x4.w));
  float2 w01 = __half22float2(half2_from_u32(w01_bits));
  float2 w23 = __half22float2(half2_from_u32(w23_bits));
  float2 w45 = __half22float2(half2_from_u32(w45_bits));
  float2 w67 = __half22float2(half2_from_u32(w67_bits));

  float acc = 0.0f;
  acc = fmaf(x01.x, w01.x, acc);
  acc = fmaf(x01.y, w01.y, acc);
  acc = fmaf(x23.x, w23.x, acc);
  acc = fmaf(x23.y, w23.y, acc);
  acc = fmaf(x45.x, w45.x, acc);
  acc = fmaf(x45.y, w45.y, acc);
  acc = fmaf(x67.x, w67.x, acc);
  acc = fmaf(x67.y, w67.y, acc);
  return acc;
}

__forceinline__ __device__ float dot_acc_half2_bits(float acc, uint32_t x_bits, uint32_t w_bits) {
  float2 xf = __half22float2(half2_from_u32(x_bits));
  float2 wf = __half22float2(half2_from_u32(w_bits));
  acc = fmaf(xf.x, wf.x, acc);
  acc = fmaf(xf.y, wf.y, acc);
  return acc;
}

__forceinline__ __device__ float dot8_half2_lowreg(
    uint4 x4,
    uint2 w2) {
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

__forceinline__ __device__ float dot8_half2_haccum(
    uint4 x4,
    uint2 w2) {
  uint32_t w_lo;
  uint32_t w_hi;
  half2 acc = __float2half2_rn(0.0f);
  fp8e4b15x4_to_half2_bits(w2.x, w_lo, w_hi);
  acc = __hfma2(half2_from_u32(x4.x), half2_from_u32(w_lo), acc);
  acc = __hfma2(half2_from_u32(x4.y), half2_from_u32(w_hi), acc);
  fp8e4b15x4_to_half2_bits(w2.y, w_lo, w_hi);
  acc = __hfma2(half2_from_u32(x4.z), half2_from_u32(w_lo), acc);
  acc = __hfma2(half2_from_u32(x4.w), half2_from_u32(w_hi), acc);
  const float2 pair = __half22float2(acc);
  return pair.x + pair.y;
}

__forceinline__ __device__ float dot8_half2_f32_dualacc(
    uint4 x4,
    uint2 w2) {
  uint32_t w01_bits;
  uint32_t w23_bits;
  uint32_t w45_bits;
  uint32_t w67_bits;
  fp8e4b15x4_to_half2_bits(w2.x, w01_bits, w23_bits);
  fp8e4b15x4_to_half2_bits(w2.y, w45_bits, w67_bits);

  const float2 x01 = __half22float2(half2_from_u32(x4.x));
  const float2 x23 = __half22float2(half2_from_u32(x4.y));
  const float2 x45 = __half22float2(half2_from_u32(x4.z));
  const float2 x67 = __half22float2(half2_from_u32(x4.w));
  const float2 w01 = __half22float2(half2_from_u32(w01_bits));
  const float2 w23 = __half22float2(half2_from_u32(w23_bits));
  const float2 w45 = __half22float2(half2_from_u32(w45_bits));
  const float2 w67 = __half22float2(half2_from_u32(w67_bits));

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  acc0 = fmaf(x01.x, w01.x, acc0);
  acc1 = fmaf(x01.y, w01.y, acc1);
  acc0 = fmaf(x23.x, w23.x, acc0);
  acc1 = fmaf(x23.y, w23.y, acc1);
  acc0 = fmaf(x45.x, w45.x, acc0);
  acc1 = fmaf(x45.y, w45.y, acc1);
  acc0 = fmaf(x67.x, w67.x, acc0);
  acc1 = fmaf(x67.y, w67.y, acc1);
  return acc0 + acc1;
}

__forceinline__ __device__ float dot8_int4xhalf_from_u32(uint32_t packed, uint4 x4, int lane_half) {
  uint32_t x_bits;
  if (lane_half == 0) {
    x_bits = x4.x;
  } else if (lane_half == 1) {
    x_bits = x4.y;
  } else if (lane_half == 2) {
    x_bits = x4.z;
  } else {
    x_bits = x4.w;
  }
  const float2 xv = __half22float2(half2_from_u32(x_bits));
  float acc = 0.0f;
  const float w0 = static_cast<float>(static_cast<int>(packed & 0x0fu) - 8);
  const float w1 = static_cast<float>(static_cast<int>((packed >> 4) & 0x0fu) - 8);
  acc = fmaf(xv.x, w0, acc);
  acc = fmaf(xv.y, w1, acc);
  return acc;
}

__forceinline__ __device__ float dot16_int4xhalf(uint4 x4a, uint4 x4b, uint2 w2) {
  float acc = 0.0f;
  acc += dot8_int4xhalf_from_u32(w2.x & 0xffu, x4a, 0);
  acc += dot8_int4xhalf_from_u32((w2.x >> 8) & 0xffu, x4a, 1);
  acc += dot8_int4xhalf_from_u32((w2.x >> 16) & 0xffu, x4a, 2);
  acc += dot8_int4xhalf_from_u32((w2.x >> 24) & 0xffu, x4a, 3);
  acc += dot8_int4xhalf_from_u32(w2.y & 0xffu, x4b, 0);
  acc += dot8_int4xhalf_from_u32((w2.y >> 8) & 0xffu, x4b, 1);
  acc += dot8_int4xhalf_from_u32((w2.y >> 16) & 0xffu, x4b, 2);
  acc += dot8_int4xhalf_from_u32((w2.y >> 24) & 0xffu, x4b, 3);
  return acc;
}

__forceinline__ __device__ float halfwarp_bfly_sum(float value) {
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
  }
  return value;
}

__forceinline__ __device__ float halfwarp_bfly_max(float value) {
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
  }
  return value;
}

__forceinline__ __device__ float first_halfwarp_sum(float value) {
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0x0000ffffu, value, mask, 32);
  }
  return value;
}

__forceinline__ __device__ float first_halfwarp_max(float value) {
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0x0000ffffu, value, mask, 32));
  }
  return value;
}

__forceinline__ __device__ float masked_halfwarp_sum(float value, uint32_t mask) {
#pragma unroll
  for (int delta = 8; delta > 0; delta >>= 1) {
    value += __shfl_xor_sync(mask, value, delta, 32);
  }
  return value;
}

__forceinline__ __device__ float masked_halfwarp_max(float value, uint32_t mask) {
#pragma unroll
  for (int delta = 8; delta > 0; delta >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, delta, 32));
  }
  return value;
}

__forceinline__ __device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__forceinline__ __device__ float warp_bfly_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
  }
  return value;
}

__forceinline__ __device__ float warp8_bfly_sum(float value) {
#pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
  }
  return value;
}

__forceinline__ __device__ uint32_t pack_half2_bits(half lo, half hi) {
  const uint32_t lo_bits = static_cast<uint32_t>(__half_as_ushort(lo));
  const uint32_t hi_bits = static_cast<uint32_t>(__half_as_ushort(hi));
  return lo_bits | (hi_bits << 16);
}

template <bool kKscale, bool kScaleBroadcast>
__forceinline__ __device__ half load_row_scale(
    const half* __restrict__ scales,
    int block_id,
    int row,
    int lane,
    int kScaleBlocks) {
  const half* scale_ptr = kKscale
      ? scales + static_cast<int64_t>(block_id) * kHiddenSize + row
      : scales + static_cast<int64_t>(row) * kScaleBlocks + block_id;
  if constexpr (kScaleBroadcast) {
    return broadcast_halfwarp_scale(scale_ptr, lane);
  } else {
    return load_half_evict_first(scale_ptr);
  }
}

__forceinline__ __device__ half load_row_scale_broadcast8(
    const half* __restrict__ scales,
    int block_id,
    int row,
    int lane,
    int kScaleBlocks) {
  const half* scale_ptr = scales + static_cast<int64_t>(row) * kScaleBlocks + block_id;
  uint32_t bits = 0;
  if (lane == 0) {
    bits = static_cast<uint32_t>(__half_as_ushort(load_half_evict_first(scale_ptr)));
  }
  bits = __shfl_sync(0xffffffffu, bits, threadIdx.x & ~7, 32);
  return __ushort_as_half(static_cast<unsigned short>(bits));
}

__forceinline__ __device__ half load_row_scale_broadcast4(
    const half* __restrict__ scales,
    int block_id,
    int row,
    int lane,
    int kScaleBlocks) {
  const half* scale_ptr = scales + static_cast<int64_t>(row) * kScaleBlocks + block_id;
  uint32_t bits = 0;
  if (lane == 0) {
    bits = static_cast<uint32_t>(__half_as_ushort(load_half_evict_first(scale_ptr)));
  }
  bits = __shfl_sync(0xffffffffu, bits, threadIdx.x & ~3, 32);
  return __ushort_as_half(static_cast<unsigned short>(bits));
}

__forceinline__ __device__ uint4 load_warp_shared_x4(const half* x, int block_id, int lane_k, int lane) {
  uint4 x4{0, 0, 0, 0};
  if ((threadIdx.x & 16) == 0) {
    x4 = load_v4_evict_last(reinterpret_cast<const uint4*>(x + block_id * kBlockK + lane_k));
  }
  const int src_lane = lane;
  x4.x = __shfl_sync(0xffffffffu, x4.x, src_lane, 32);
  x4.y = __shfl_sync(0xffffffffu, x4.y, src_lane, 32);
  x4.z = __shfl_sync(0xffffffffu, x4.z, src_lane, 32);
  x4.w = __shfl_sync(0xffffffffu, x4.w, src_lane, 32);
  return x4;
}

__forceinline__ __device__ uint4 load_cta_shared_x4(
    const half* x,
    int block_id,
    int lane,
    uint4* x_shared) {
  if (threadIdx.x < kLanesPerRow) {
    x_shared[threadIdx.x] = load_v4_evict_last(
        reinterpret_cast<const uint4*>(x + block_id * kBlockK + threadIdx.x * 8));
  }
  __syncthreads();
  uint4 x4 = x_shared[lane];
  __syncthreads();
  return x4;
}

template <
    bool kKscale,
    int kIn,
    int kSplitK,
    bool kUnroll,
    bool kScaleBroadcast,
    bool kDirectStore,
    bool kWarpSharedX,
    bool kCtaSharedX,
    bool kLowReg,
    bool kPreloadSharedX,
    bool kHaccum = false,
    bool kF32DualAcc = false,
    bool kOuterDualAcc = false>
__global__ __launch_bounds__(kThreads)
void gemv_splitk_partial_fp8e4b15_tritonshape_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kIn / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kSharedXItems = kPreloadSharedX ? (kBlocksPerSplit * kLanesPerRow) : kLanesPerRow;
  __shared__ float row_sums[kBlockN];
  __shared__ uint4 x_shared[kSharedXItems];

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int row = blockIdx.x * kBlockN + row_in_block;
  const int split = blockIdx.y;
  const int lane_k = lane * 8;
  const int block_begin = split * kBlocksPerSplit;

  if constexpr (kPreloadSharedX) {
#pragma unroll
    for (int idx = tid; idx < kBlocksPerSplit * kLanesPerRow; idx += kThreads) {
      const int u = idx / kLanesPerRow;
      const int shared_lane = idx - u * kLanesPerRow;
      const int block_id = block_begin + u;
      x_shared[idx] = load_v4_evict_last(
          reinterpret_cast<const uint4*>(x + block_id * kBlockK + shared_lane * 8));
    }
    __syncthreads();
  }

  float acc = 0.0f;
  float acc_alt = 0.0f;

  if constexpr (kUnroll) {
#pragma unroll
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      uint4 x4;
      if constexpr (kPreloadSharedX) {
        x4 = x_shared[u * kLanesPerRow + lane];
      } else if constexpr (kCtaSharedX) {
        x4 = load_cta_shared_x4(x, block_id, lane, x_shared);
      } else if constexpr (kWarpSharedX) {
        x4 = load_warp_shared_x4(x, block_id, lane_k, lane);
      } else {
        x4 = load_v4_evict_last(reinterpret_cast<const uint4*>(x + block_id * kBlockK + lane_k));
      }
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<kKscale, kScaleBroadcast>(
          scales, block_id, row, lane, kScaleBlocks);
      const float dot = kF32DualAcc
          ? dot8_half2_f32_dualacc(x4, w2)
          : kHaccum ? dot8_half2_haccum(x4, w2) : kLowReg ? dot8_half2_lowreg(x4, w2) : dot8_half2(x4, w2);
      if constexpr (kOuterDualAcc) {
        if ((u & 1) == 0) {
          acc = fmaf(dot, __half2float(scale_h), acc);
        } else {
          acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
        }
      } else {
        acc = fmaf(dot, __half2float(scale_h), acc);
      }
    }
  } else {
#pragma unroll 1
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      uint4 x4;
      if constexpr (kPreloadSharedX) {
        x4 = x_shared[u * kLanesPerRow + lane];
      } else if constexpr (kCtaSharedX) {
        x4 = load_cta_shared_x4(x, block_id, lane, x_shared);
      } else if constexpr (kWarpSharedX) {
        x4 = load_warp_shared_x4(x, block_id, lane_k, lane);
      } else {
        x4 = load_v4_evict_last(reinterpret_cast<const uint4*>(x + block_id * kBlockK + lane_k));
      }
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<kKscale, kScaleBroadcast>(
          scales, block_id, row, lane, kScaleBlocks);
      const float dot = kF32DualAcc
          ? dot8_half2_f32_dualacc(x4, w2)
          : kHaccum ? dot8_half2_haccum(x4, w2) : kLowReg ? dot8_half2_lowreg(x4, w2) : dot8_half2(x4, w2);
      if constexpr (kOuterDualAcc) {
        if ((u & 1) == 0) {
          acc = fmaf(dot, __half2float(scale_h), acc);
        } else {
          acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
        }
      } else {
        acc = fmaf(dot, __half2float(scale_h), acc);
      }
    }
  }

  if constexpr (kOuterDualAcc) {
    acc += acc_alt;
  }
  acc = halfwarp_bfly_sum(acc);
  if constexpr (kDirectStore) {
    if (lane == 0) {
      partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
    }
  } else {
    if (lane == 0) {
      row_sums[row_in_block] = acc;
    }
    __syncthreads();
    if (tid < kBlockN) {
      partial[static_cast<int64_t>(split) * kHiddenSize + blockIdx.x * kBlockN + tid] = row_sums[tid];
    }
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ scratch_x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  cg::grid_group grid = cg::this_grid();

  for (int idx = blockIdx.x + threadIdx.x * gridDim.x;
       idx < kHiddenSize;
       idx += blockDim.x * gridDim.x) {
    const int q_head = idx / kHeadDim;
    const int d = idx - q_head * kHeadDim;
    const int state_base = q_head * workspace_splits;
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      m = fmaxf(m, partial_m[state_base + s]);
    }
    float l = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      const float weight = __expf(partial_m[state_base + s] - m);
      l += partial_l[state_base + s] * weight;
      acc += partial_acc[(state_base + s) * kHeadDim + d] * weight;
    }
    scratch_x[idx] = __float2half_rn(acc / l);
  }

  grid.sync();

  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowBlocks = kHiddenSize / kBlockN;
  constexpr int kTasks = kRowBlocks * kSplitK;

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int lane_k = lane * 8;

  for (int task = blockIdx.x; task < kTasks; task += gridDim.x) {
    const int split = task / kRowBlocks;
    const int out_block = task - split * kRowBlocks;
    const int row = out_block * kBlockN + row_in_block;
    const int block_begin = split * kBlocksPerSplit;

    float acc = 0.0f;
    float acc_alt = 0.0f;
#pragma unroll
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      const uint4 x4 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(scratch_x + block_id * kBlockK + lane_k));
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<false, true>(
          scales, block_id, row, lane, kScaleBlocks);
      const float dot = dot8_half2_f32_dualacc(x4, w2);
      if ((u & 1) == 0) {
        acc = fmaf(dot, __half2float(scale_h), acc);
      } else {
        acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
      }
    }

    acc += acc_alt;
    acc = halfwarp_bfly_sum(acc);
    if (lane == 0) {
      o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
    }
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ scratch_x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    unsigned int* __restrict__ barrier_counter,
    int workspace_splits) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < kHiddenSize;
       idx += blockDim.x * gridDim.x) {
    const int q_head = idx / kHeadDim;
    const int d = idx - q_head * kHeadDim;
    const int state_base = q_head * workspace_splits;
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      m = fmaxf(m, partial_m[state_base + s]);
    }
    float l = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      const float weight = __expf(partial_m[state_base + s] - m);
      l += partial_l[state_base + s] * weight;
      acc += partial_acc[(state_base + s) * kHeadDim + d] * weight;
    }
    scratch_x[idx] = __float2half_rn(acc / l);
  }

  __threadfence();
  if (threadIdx.x == 0) {
    atomicAdd(barrier_counter, 1u);
    volatile unsigned int* counter = barrier_counter;
    while (counter[0] < static_cast<unsigned int>(gridDim.x)) {
    }
  }
  __syncthreads();

  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowBlocks = kHiddenSize / kBlockN;
  constexpr int kTasks = kRowBlocks * kSplitK;

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int lane_k = lane * 8;

  for (int task = blockIdx.x; task < kTasks; task += gridDim.x) {
    const int split = task / kRowBlocks;
    const int out_block = task - split * kRowBlocks;
    const int row = out_block * kBlockN + row_in_block;
    const int block_begin = split * kBlocksPerSplit;

    float acc = 0.0f;
#pragma unroll
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      const uint4 x4 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(scratch_x + block_id * kBlockK + lane_k));
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<false, true>(
          scales, block_id, row, lane, kScaleBlocks);
      const float dot = dot8_half2_f32_dualacc(x4, w2);
      acc = fmaf(dot, __half2float(scale_h), acc);
    }

    acc = halfwarp_bfly_sum(acc);
    if (lane == 0) {
      o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
    }
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  __shared__ half x_shared[kBlockK];
  __shared__ float row_sums[kBlockN];

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int row = blockIdx.x * kBlockN + row_in_block;
  const int split = blockIdx.y;
  const int lane_k = lane * 8;
  const int block_begin = split * kBlocksPerSplit;

  float acc = 0.0f;
  float acc_alt = 0.0f;

#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    for (int local_idx = tid; local_idx < kBlockK; local_idx += kThreads) {
      const int idx = block_id * kBlockK + local_idx;
      const int q_head = idx / kHeadDim;
      const int d = idx - q_head * kHeadDim;
      const int state_base = q_head * workspace_splits;
      float m = -INFINITY;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        m = fmaxf(m, partial_m[state_base + s]);
      }
      float l = 0.0f;
      float x_acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        const float weight = __expf(partial_m[state_base + s] - m);
        l += partial_l[state_base + s] * weight;
        x_acc += partial_acc[(state_base + s) * kHeadDim + d] * weight;
      }
      x_shared[local_idx] = __float2half_rn(x_acc / l);
    }
    __syncthreads();

    const uint4 x4 = *reinterpret_cast<const uint4*>(x_shared + lane_k);
    const uint2 w2 = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const half scale_h = load_row_scale<false, true>(
        scales, block_id, row, lane, kScaleBlocks);
    const float dot = dot8_half2_f32_dualacc(x4, w2);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
    __syncthreads();
  }

  acc += acc_alt;
  acc = halfwarp_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <int kActiveSplits, int kSplitK, int kRowBlocksPerCta>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowGroups = (kHiddenSize / kBlockN) / kRowBlocksPerCta;
  __shared__ half x_shared[kBlockK];
  __shared__ float coeff_shared[16];

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int lane_k = lane * 8;
  const int row_group = blockIdx.x;
  const int split = blockIdx.y;
  const int block_begin = split * kBlocksPerSplit;
  if (row_group >= kRowGroups) {
    return;
  }

  float acc[kRowBlocksPerCta];
  float acc_alt[kRowBlocksPerCta];
#pragma unroll
  for (int r = 0; r < kRowBlocksPerCta; ++r) {
    acc[r] = 0.0f;
    acc_alt[r] = 0.0f;
  }

#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const int q_head = block_id;
    const int state_base = q_head * workspace_splits;
    if (tid < 16) {
      const float m_part = tid < kActiveSplits ? partial_m[state_base + tid] : -INFINITY;
      const float m = first_halfwarp_max(m_part);
      const float weight_part = tid < kActiveSplits ? __expf(m_part - m) : 0.0f;
      const float denom_part = tid < kActiveSplits ? partial_l[state_base + tid] * weight_part : 0.0f;
      const float denom = first_halfwarp_sum(denom_part);
      if (tid < kActiveSplits) {
        coeff_shared[tid] = weight_part / denom;
      }
    }
    __syncthreads();
    for (int local_idx = tid; local_idx < kBlockK; local_idx += kThreads) {
      const int idx = block_id * kBlockK + local_idx;
      float x_acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        x_acc += partial_acc[(state_base + s) * kHeadDim + local_idx] * coeff_shared[s];
      }
      x_shared[local_idx] = __float2half_rn(x_acc);
    }
    __syncthreads();

    const uint4 x4 = *reinterpret_cast<const uint4*>(x_shared + lane_k);
#pragma unroll
    for (int r = 0; r < kRowBlocksPerCta; ++r) {
      const int row = (row_group * kRowBlocksPerCta + r) * kBlockN + row_in_block;
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<false, true>(
          scales, block_id, row, lane, kScaleBlocks);
      const float dot = dot8_half2_f32_dualacc(x4, w2);
      if ((u & 1) == 0) {
        acc[r] = fmaf(dot, __half2float(scale_h), acc[r]);
      } else {
        acc_alt[r] = fmaf(dot, __half2float(scale_h), acc_alt[r]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < kRowBlocksPerCta; ++r) {
    const int row = (row_group * kRowBlocksPerCta + r) * kBlockN + row_in_block;
    const float value = halfwarp_bfly_sum(acc[r] + acc_alt[r]);
    if (lane == 0) {
      o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = value;
    }
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerCta = 32;
  __shared__ half x_shared[kBlockK];
  __shared__ float coeff_shared[16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 3;
  const int lane = tid & 7;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;

  float acc = 0.0f;
  float acc_alt = 0.0f;

#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const int q_head = block_id;
    const int state_base = q_head * workspace_splits;
    if (tid < 16) {
      const float m_part = tid < kActiveSplits ? partial_m[state_base + tid] : -INFINITY;
      const float m = first_halfwarp_max(m_part);
      const float weight_part = tid < kActiveSplits ? __expf(m_part - m) : 0.0f;
      const float denom_part = tid < kActiveSplits ? partial_l[state_base + tid] * weight_part : 0.0f;
      const float denom = first_halfwarp_sum(denom_part);
      if (tid < kActiveSplits) {
        coeff_shared[tid] = weight_part / denom;
      }
    }
    __syncthreads();
    if (tid < kBlockK) {
      float x_acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        x_acc += partial_acc[(state_base + s) * kHeadDim + tid] * coeff_shared[s];
      }
      x_shared[tid] = __float2half_rn(x_acc);
    }
    __syncthreads();

    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + lane_k + 8);
    const uint2 w2a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const uint2 w2b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 8));
    const half scale_h = load_row_scale<false, false>(
        scales, block_id, row, 0, kScaleBlocks);
    const float dot = dot8_half2_f32_dualacc(x4a, w2a) + dot8_half2_f32_dualacc(x4b, w2b);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
    __syncthreads();
  }

  acc += acc_alt;
  acc = warp8_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerCta = 32;
  __shared__ half x_shared[kBlocksPerSplit * kBlockK];
  __shared__ float coeff_shared[kBlocksPerSplit * 16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 3;
  const int lane = tid & 7;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;

  if (tid < kBlocksPerSplit * 16) {
    const int u = tid >> 4;
    const int s = tid & 15;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    const uint32_t mask = 0x0000ffffu << ((u & 1) * 16);
    const float m_part = s < kActiveSplits ? partial_m[state_base + s] : -INFINITY;
    const float m = masked_halfwarp_max(m_part, mask);
    const float weight_part = s < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = s < kActiveSplits ? partial_l[state_base + s] * weight_part : 0.0f;
    const float denom = masked_halfwarp_sum(denom_part, mask);
    if (s < kActiveSplits) {
      coeff_shared[u * 16 + s] = weight_part / denom;
    }
  }
  __syncthreads();

  for (int idx = tid; idx < kBlocksPerSplit * kBlockK; idx += kThreads) {
    const int u = idx / kBlockK;
    const int d = idx - u * kBlockK;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    float x_acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      x_acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff_shared[u * 16 + s];
    }
    x_shared[idx] = __float2half_rn(x_acc);
  }
  __syncthreads();

  float acc = 0.0f;
  float acc_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 8);
    const uint2 w2a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const uint2 w2b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 8));
    const half scale_h = load_row_scale<false, false>(scales, block_id, row, 0, kScaleBlocks);
    const float dot = dot8_half2_f32_dualacc(x4a, w2a) + dot8_half2_f32_dualacc(x4b, w2b);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
  }

  acc += acc_alt;
  acc = warp8_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(128)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerCta = 16;
  __shared__ half x_shared[kBlocksPerSplit * kBlockK];
  __shared__ float coeff_shared[kBlocksPerSplit * 16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 3;
  const int lane = tid & 7;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;

  if (tid < kBlocksPerSplit * 16) {
    const int u = tid >> 4;
    const int s = tid & 15;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    const uint32_t mask = 0x0000ffffu << ((u & 1) * 16);
    const float m_part = s < kActiveSplits ? partial_m[state_base + s] : -INFINITY;
    const float m = masked_halfwarp_max(m_part, mask);
    const float weight_part = s < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = s < kActiveSplits ? partial_l[state_base + s] * weight_part : 0.0f;
    const float denom = masked_halfwarp_sum(denom_part, mask);
    if (s < kActiveSplits) {
      coeff_shared[u * 16 + s] = weight_part / denom;
    }
  }
  __syncthreads();

#pragma unroll
  for (int idx = tid; idx < kBlocksPerSplit * kBlockK; idx += 128) {
    const int u = idx / kBlockK;
    const int d = idx - u * kBlockK;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    float x_acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      x_acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff_shared[u * 16 + s];
    }
    x_shared[idx] = __float2half_rn(x_acc);
  }
  __syncthreads();

  float acc = 0.0f;
  float acc_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 8);
    const uint2 w2a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const uint2 w2b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 8));
    const half scale_h = load_row_scale<false, false>(scales, block_id, row, 0, kScaleBlocks);
    const float dot = dot8_half2_f32_dualacc(x4a, w2a) + dot8_half2_f32_dualacc(x4b, w2b);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
  }

  acc += acc_alt;
  acc = warp8_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerPass = 32;
  constexpr int kRowsPerCta = 64;
  __shared__ half x_shared[kBlocksPerSplit * kBlockK];
  __shared__ float coeff_shared[kBlocksPerSplit * 16];

  const int tid = threadIdx.x;
  const int row_in_pass = tid >> 3;
  const int lane = tid & 7;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;
  const int row0 = blockIdx.x * kRowsPerCta + row_in_pass;
  const int row1 = row0 + kRowsPerPass;

  if (tid < kBlocksPerSplit * 16) {
    const int u = tid >> 4;
    const int s = tid & 15;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    const uint32_t mask = 0x0000ffffu << ((u & 1) * 16);
    const float m_part = s < kActiveSplits ? partial_m[state_base + s] : -INFINITY;
    const float m = masked_halfwarp_max(m_part, mask);
    const float weight_part = s < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = s < kActiveSplits ? partial_l[state_base + s] * weight_part : 0.0f;
    const float denom = masked_halfwarp_sum(denom_part, mask);
    if (s < kActiveSplits) {
      coeff_shared[u * 16 + s] = weight_part / denom;
    }
  }
  __syncthreads();

  for (int idx = tid; idx < kBlocksPerSplit * kBlockK; idx += kThreads) {
    const int u = idx / kBlockK;
    const int d = idx - u * kBlockK;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    float x_acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      x_acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff_shared[u * 16 + s];
    }
    x_shared[idx] = __float2half_rn(x_acc);
  }
  __syncthreads();

  float acc0 = 0.0f;
  float acc0_alt = 0.0f;
  float acc1 = 0.0f;
  float acc1_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 8);

    const uint2 w0a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row0) * kBlockK + lane_k));
    const uint2 w0b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row0) * kBlockK + lane_k + 8));
    const half scale0 = load_row_scale<false, false>(scales, block_id, row0, 0, kScaleBlocks);
    const float dot0 = dot8_half2_f32_dualacc(x4a, w0a) + dot8_half2_f32_dualacc(x4b, w0b);

    const uint2 w1a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row1) * kBlockK + lane_k));
    const uint2 w1b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row1) * kBlockK + lane_k + 8));
    const half scale1 = load_row_scale<false, false>(scales, block_id, row1, 0, kScaleBlocks);
    const float dot1 = dot8_half2_f32_dualacc(x4a, w1a) + dot8_half2_f32_dualacc(x4b, w1b);

    if ((u & 1) == 0) {
      acc0 = fmaf(dot0, __half2float(scale0), acc0);
      acc1 = fmaf(dot1, __half2float(scale1), acc1);
    } else {
      acc0_alt = fmaf(dot0, __half2float(scale0), acc0_alt);
      acc1_alt = fmaf(dot1, __half2float(scale1), acc1_alt);
    }
  }

  const float value0 = warp8_bfly_sum(acc0 + acc0_alt);
  const float value1 = warp8_bfly_sum(acc1 + acc1_alt);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row0] = value0;
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row1] = value1;
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(512)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerCta = 64;
  __shared__ half x_shared[kBlocksPerSplit * kBlockK];
  __shared__ float coeff_shared[kBlocksPerSplit * 16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 3;
  const int lane = tid & 7;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;

  if (tid < kBlocksPerSplit * 16) {
    const int u = tid >> 4;
    const int s = tid & 15;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    const uint32_t mask = 0x0000ffffu << ((u & 1) * 16);
    const float m_part = s < kActiveSplits ? partial_m[state_base + s] : -INFINITY;
    const float m = masked_halfwarp_max(m_part, mask);
    const float weight_part = s < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = s < kActiveSplits ? partial_l[state_base + s] * weight_part : 0.0f;
    const float denom = masked_halfwarp_sum(denom_part, mask);
    if (s < kActiveSplits) {
      coeff_shared[u * 16 + s] = weight_part / denom;
    }
  }
  __syncthreads();

#pragma unroll
  for (int idx = tid; idx < kBlocksPerSplit * kBlockK; idx += 512) {
    const int u = idx / kBlockK;
    const int d = idx - u * kBlockK;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    float x_acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      x_acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff_shared[u * 16 + s];
    }
    x_shared[idx] = __float2half_rn(x_acc);
  }
  __syncthreads();

  float acc = 0.0f;
  float acc_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 8);
    const uint2 w2a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const uint2 w2b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 8));
    const half scale_h = load_row_scale<false, false>(scales, block_id, row, 0, kScaleBlocks);
    const float dot = dot8_half2_f32_dualacc(x4a, w2a) + dot8_half2_f32_dualacc(x4b, w2b);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
  }

  acc += acc_alt;
  acc = warp8_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <
    int kActiveSplits,
    int kSplitK,
    int kThreadsPerCta,
    int kRowsPerCta,
    int kDotMode>
__global__ __launch_bounds__(kThreadsPerCta)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  static_assert(kThreadsPerCta == kRowsPerCta * 8, "lane8 generic expects 8 threads per row");
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  __shared__ half x_shared[kBlocksPerSplit * kBlockK];
  __shared__ float coeff_shared[kBlocksPerSplit * 16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 3;
  const int lane = tid & 7;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;

  if (tid < kBlocksPerSplit * 16) {
    const int u = tid >> 4;
    const int s = tid & 15;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    const uint32_t mask = 0x0000ffffu << ((u & 1) * 16);
    const float m_part = s < kActiveSplits ? partial_m[state_base + s] : -INFINITY;
    const float m = masked_halfwarp_max(m_part, mask);
    const float weight_part = s < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = s < kActiveSplits ? partial_l[state_base + s] * weight_part : 0.0f;
    const float denom = masked_halfwarp_sum(denom_part, mask);
    if (s < kActiveSplits) {
      coeff_shared[u * 16 + s] = weight_part / denom;
    }
  }
  __syncthreads();

#pragma unroll
  for (int idx = tid; idx < kBlocksPerSplit * kBlockK; idx += kThreadsPerCta) {
    const int u = idx / kBlockK;
    const int d = idx - u * kBlockK;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    float x_acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      x_acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff_shared[u * 16 + s];
    }
    x_shared[idx] = __float2half_rn(x_acc);
  }
  __syncthreads();

  float acc = 0.0f;
  float acc_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 8);
    float dot = 0.0f;
    half scale_h;
    if constexpr (kDotMode == 3 || kDotMode == 4) {
      const uint4 w4 = load_v4_evict_first(
          reinterpret_cast<const uint4*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const uint2 w2a{w4.x, w4.y};
      const uint2 w2b{w4.z, w4.w};
      scale_h = load_row_scale_broadcast8(scales, block_id, row, lane, kScaleBlocks);
      dot = (kDotMode == 4)
          ? dot8_half2_f32_dualacc(x4a, w2a) + dot8_half2_f32_dualacc(x4b, w2b)
          : dot8_half2_haccum(x4a, w2a) + dot8_half2_haccum(x4b, w2b);
    } else {
      const uint2 w2a = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const uint2 w2b = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k + 8));
      scale_h = load_row_scale<false, false>(scales, block_id, row, 0, kScaleBlocks);
      if constexpr (kDotMode == 1) {
        dot = dot8_half2_lowreg(x4a, w2a) + dot8_half2_lowreg(x4b, w2b);
      } else if constexpr (kDotMode == 2) {
        dot = dot8_half2_haccum(x4a, w2a) + dot8_half2_haccum(x4b, w2b);
      } else {
        dot = dot8_half2_f32_dualacc(x4a, w2a) + dot8_half2_f32_dualacc(x4b, w2b);
      }
    }
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
  }

  acc += acc_alt;
  acc = warp8_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <
    int kActiveSplits,
    int kSplitK,
    int kThreadsPerCta,
    int kRowsPerCta>
__global__ __launch_bounds__(kThreadsPerCta)
void prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  static_assert(kThreadsPerCta == kRowsPerCta * 8, "lane8 int4 expects 8 threads per row");
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kPackedBytesPerBlock = kBlockK / 2;
  __shared__ half x_shared[kBlocksPerSplit * kBlockK];
  __shared__ float coeff_shared[kBlocksPerSplit * 16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 3;
  const int lane = tid & 7;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;

  if (tid < kBlocksPerSplit * 16) {
    const int u = tid >> 4;
    const int s = tid & 15;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    const uint32_t mask = 0x0000ffffu << ((u & 1) * 16);
    const float m_part = s < kActiveSplits ? partial_m[state_base + s] : -INFINITY;
    const float m = masked_halfwarp_max(m_part, mask);
    const float weight_part = s < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = s < kActiveSplits ? partial_l[state_base + s] * weight_part : 0.0f;
    const float denom = masked_halfwarp_sum(denom_part, mask);
    if (s < kActiveSplits) {
      coeff_shared[u * 16 + s] = weight_part / denom;
    }
  }
  __syncthreads();

#pragma unroll
  for (int idx = tid; idx < kBlocksPerSplit * kBlockK; idx += kThreadsPerCta) {
    const int u = idx / kBlockK;
    const int d = idx - u * kBlockK;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    float x_acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      x_acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff_shared[u * 16 + s];
    }
    x_shared[idx] = __float2half_rn(x_acc);
  }
  __syncthreads();

  float acc = 0.0f;
  float acc_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 8);
    const uint2 w2 = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_int4 + static_cast<int64_t>(block_id) * kHiddenSize * kPackedBytesPerBlock +
            static_cast<int64_t>(row) * kPackedBytesPerBlock + lane * 8));
    const half scale_h = load_row_scale<true, false>(scales, block_id, row, 0, kScaleBlocks);
    const float dot = dot16_int4xhalf(x4a, x4b, w2);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
  }

  acc += acc_alt;
  acc = warp8_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <int kActiveSplits, int kSplitK, int kDotMode>
__global__ __launch_bounds__(256)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerCta = 64;
  __shared__ half x_shared[kBlocksPerSplit * kBlockK];
  __shared__ float coeff_shared[kBlocksPerSplit * 16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 2;
  const int lane = tid & 3;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 32;
  const int block_begin = split * kBlocksPerSplit;

  if (tid < kBlocksPerSplit * 16) {
    const int u = tid >> 4;
    const int s = tid & 15;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    const uint32_t mask = 0x0000ffffu << ((u & 1) * 16);
    const float m_part = s < kActiveSplits ? partial_m[state_base + s] : -INFINITY;
    const float m = masked_halfwarp_max(m_part, mask);
    const float weight_part = s < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = s < kActiveSplits ? partial_l[state_base + s] * weight_part : 0.0f;
    const float denom = masked_halfwarp_sum(denom_part, mask);
    if (s < kActiveSplits) {
      coeff_shared[u * 16 + s] = weight_part / denom;
    }
  }
  __syncthreads();

#pragma unroll
  for (int idx = tid; idx < kBlocksPerSplit * kBlockK; idx += 256) {
    const int u = idx / kBlockK;
    const int d = idx - u * kBlockK;
    const int block_id = block_begin + u;
    const int state_base = block_id * workspace_splits;
    float x_acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      x_acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff_shared[u * 16 + s];
    }
    x_shared[idx] = __float2half_rn(x_acc);
  }
  __syncthreads();

  float acc = 0.0f;
  float acc_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 8);
    const uint4 x4c = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 16);
    const uint4 x4d = *reinterpret_cast<const uint4*>(x_shared + u * kBlockK + lane_k + 24);
    const uint2 w2a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const uint2 w2b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 8));
    const uint2 w2c = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 16));
    const uint2 w2d = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 24));
    const half scale_h = load_row_scale_broadcast4(scales, block_id, row, lane, kScaleBlocks);
    float dot = 0.0f;
    if constexpr (kDotMode == 1) {
      dot =
          dot8_half2_haccum(x4a, w2a) +
          dot8_half2_haccum(x4b, w2b) +
          dot8_half2_haccum(x4c, w2c) +
          dot8_half2_haccum(x4d, w2d);
    } else {
      dot =
          dot8_half2_f32_dualacc(x4a, w2a) +
          dot8_half2_f32_dualacc(x4b, w2b) +
          dot8_half2_f32_dualacc(x4c, w2c) +
          dot8_half2_f32_dualacc(x4d, w2d);
    }
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
  }

  acc += acc_alt;
  acc = __shfl_xor_sync(0xffffffffu, acc, 2, 32) + acc;
  acc = __shfl_xor_sync(0xffffffffu, acc, 1, 32) + acc;
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerCta = 64;
  __shared__ half x_shared[kBlockK];
  __shared__ float coeff_shared[16];

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 2;
  const int lane = tid & 3;
  const int row = blockIdx.x * kRowsPerCta + row_in_cta;
  const int split = blockIdx.y;
  const int lane_k = lane * 32;
  const int block_begin = split * kBlocksPerSplit;

  float acc = 0.0f;
  float acc_alt = 0.0f;

#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const int q_head = block_id;
    const int state_base = q_head * workspace_splits;
    const int lane16 = tid & 15;
    const float m_part = lane16 < kActiveSplits ? partial_m[state_base + lane16] : -INFINITY;
    const float m = halfwarp_bfly_max(m_part);
    const float weight_part = lane16 < kActiveSplits ? __expf(m_part - m) : 0.0f;
    const float denom_part = lane16 < kActiveSplits ? partial_l[state_base + lane16] * weight_part : 0.0f;
    const float denom = halfwarp_bfly_sum(denom_part);
    if (tid < 16 && lane16 < kActiveSplits) {
      coeff_shared[lane16] = weight_part / denom;
    }
    __syncthreads();
    if (tid < kBlockK) {
      float x_acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        x_acc += partial_acc[(state_base + s) * kHeadDim + tid] * coeff_shared[s];
      }
      x_shared[tid] = __float2half_rn(x_acc);
    }
    __syncthreads();

    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + lane_k + 8);
    const uint4 x4c = *reinterpret_cast<const uint4*>(x_shared + lane_k + 16);
    const uint4 x4d = *reinterpret_cast<const uint4*>(x_shared + lane_k + 24);
    const uint2 w2a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const uint2 w2b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 8));
    const uint2 w2c = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 16));
    const uint2 w2d = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k + 24));
    const half scale_h = load_row_scale<false, false>(
        scales, block_id, row, 0, kScaleBlocks);
    const float dot =
        dot8_half2_f32_dualacc(x4a, w2a) +
        dot8_half2_f32_dualacc(x4b, w2b) +
        dot8_half2_f32_dualacc(x4c, w2c) +
        dot8_half2_f32_dualacc(x4d, w2d);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
    __syncthreads();
  }

  acc += acc_alt;
  acc = __shfl_xor_sync(0xffffffffu, acc, 2, 32) + acc;
  acc = __shfl_xor_sync(0xffffffffu, acc, 1, 32) + acc;
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowsPerPass = 32;
  constexpr int kRowsPerCta = 64;
  __shared__ half x_shared[kBlockK];
  __shared__ float coeff_shared[16];

  const int tid = threadIdx.x;
  const int row_in_pass = tid >> 3;
  const int lane = tid & 7;
  const int split = blockIdx.y;
  const int lane_k = lane * 16;
  const int block_begin = split * kBlocksPerSplit;

  float acc0 = 0.0f;
  float acc0_alt = 0.0f;
  float acc1 = 0.0f;
  float acc1_alt = 0.0f;

#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const int q_head = block_id;
    const int state_base = q_head * workspace_splits;
    if (tid < 16) {
      const float m_part = tid < kActiveSplits ? partial_m[state_base + tid] : -INFINITY;
      const float m = first_halfwarp_max(m_part);
      const float weight_part = tid < kActiveSplits ? __expf(m_part - m) : 0.0f;
      const float denom_part = tid < kActiveSplits ? partial_l[state_base + tid] * weight_part : 0.0f;
      const float denom = first_halfwarp_sum(denom_part);
      if (tid < kActiveSplits) {
        coeff_shared[tid] = weight_part / denom;
      }
    }
    __syncthreads();
    if (tid < kBlockK) {
      float x_acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        x_acc += partial_acc[(state_base + s) * kHeadDim + tid] * coeff_shared[s];
      }
      x_shared[tid] = __float2half_rn(x_acc);
    }
    __syncthreads();

    const uint4 x4a = *reinterpret_cast<const uint4*>(x_shared + lane_k);
    const uint4 x4b = *reinterpret_cast<const uint4*>(x_shared + lane_k + 8);
    const int row0 = blockIdx.x * kRowsPerCta + row_in_pass;
    const int row1 = row0 + kRowsPerPass;

    const uint2 w0a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row0) * kBlockK + lane_k));
    const uint2 w0b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row0) * kBlockK + lane_k + 8));
    const half scale0 = load_row_scale<false, false>(scales, block_id, row0, 0, kScaleBlocks);
    const float dot0 = dot8_half2_f32_dualacc(x4a, w0a) + dot8_half2_f32_dualacc(x4b, w0b);

    const uint2 w1a = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row1) * kBlockK + lane_k));
    const uint2 w1b = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row1) * kBlockK + lane_k + 8));
    const half scale1 = load_row_scale<false, false>(scales, block_id, row1, 0, kScaleBlocks);
    const float dot1 = dot8_half2_f32_dualacc(x4a, w1a) + dot8_half2_f32_dualacc(x4b, w1b);

    if ((u & 1) == 0) {
      acc0 = fmaf(dot0, __half2float(scale0), acc0);
      acc1 = fmaf(dot1, __half2float(scale1), acc1);
    } else {
      acc0_alt = fmaf(dot0, __half2float(scale0), acc0_alt);
      acc1_alt = fmaf(dot1, __half2float(scale1), acc1_alt);
    }
    __syncthreads();
  }

  const int row0 = blockIdx.x * kRowsPerCta + row_in_pass;
  const int row1 = row0 + kRowsPerPass;
  float value0 = warp8_bfly_sum(acc0 + acc0_alt);
  float value1 = warp8_bfly_sum(acc1 + acc1_alt);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row0] = value0;
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row1] = value1;
  }
}

template <int kActiveSplits, int kSplitK, int kRowBlocksPerCta>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ scratch_x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int workspace_splits) {
  cg::grid_group grid = cg::this_grid();

  for (int idx = blockIdx.x + threadIdx.x * gridDim.x;
       idx < kHiddenSize;
       idx += blockDim.x * gridDim.x) {
    const int q_head = idx / kHeadDim;
    const int d = idx - q_head * kHeadDim;
    const int state_base = q_head * workspace_splits;
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      m = fmaxf(m, partial_m[state_base + s]);
    }
    float l = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kActiveSplits; ++s) {
      const float weight = __expf(partial_m[state_base + s] - m);
      l += partial_l[state_base + s] * weight;
      acc += partial_acc[(state_base + s) * kHeadDim + d] * weight;
    }
    scratch_x[idx] = __float2half_rn(acc / l);
  }

  grid.sync();

  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowBlocks = kHiddenSize / kBlockN;
  constexpr int kRowGroupTasks = kRowBlocks / kRowBlocksPerCta;
  constexpr int kTasks = kRowGroupTasks * kSplitK;

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int lane_k = lane * 8;

  for (int task = blockIdx.x; task < kTasks; task += gridDim.x) {
    const int split = task / kRowGroupTasks;
    const int row_group = task - split * kRowGroupTasks;
    const int row_base = row_group * kRowBlocksPerCta * kBlockN + row_in_block;
    const int block_begin = split * kBlocksPerSplit;

    float acc[kRowBlocksPerCta];
    float acc_alt[kRowBlocksPerCta];
#pragma unroll
    for (int r = 0; r < kRowBlocksPerCta; ++r) {
      acc[r] = 0.0f;
      acc_alt[r] = 0.0f;
    }

#pragma unroll
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      const uint4 x4 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(scratch_x + block_id * kBlockK + lane_k));
#pragma unroll
      for (int r = 0; r < kRowBlocksPerCta; ++r) {
        const int row = row_base + r * kBlockN;
        const uint2 w2 = load_v2_evict_first(
            reinterpret_cast<const uint2*>(
                weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
                static_cast<int64_t>(row) * kBlockK + lane_k));
        const half scale_h = load_row_scale<false, true>(
            scales, block_id, row, lane, kScaleBlocks);
        const float dot = dot8_half2_f32_dualacc(x4, w2);
        if ((u & 1) == 0) {
          acc[r] = fmaf(dot, __half2float(scale_h), acc[r]);
        } else {
          acc_alt[r] = fmaf(dot, __half2float(scale_h), acc_alt[r]);
        }
      }
    }

#pragma unroll
    for (int r = 0; r < kRowBlocksPerCta; ++r) {
      const int row = row_base + r * kBlockN;
      float value = halfwarp_bfly_sum(acc[r] + acc_alt[r]);
      if (lane == 0) {
        o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = value;
      }
    }
  }
}

template <int kSplitK>
__global__ __launch_bounds__(kThreads)
void gemv_splitk_partial_fp8e4b15_kblock_major_taskloop_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowBlocks = kHiddenSize / kBlockN;
  constexpr int kTasks = kRowBlocks * kSplitK;

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int lane_k = lane * 8;

  for (int task = blockIdx.x; task < kTasks; task += gridDim.x) {
    const int split = task / kRowBlocks;
    const int out_block = task - split * kRowBlocks;
    const int row = out_block * kBlockN + row_in_block;
    const int block_begin = split * kBlocksPerSplit;

    float acc = 0.0f;
    float acc_alt = 0.0f;
#pragma unroll
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      const uint4 x4 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(x + block_id * kBlockK + lane_k));
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<false, true>(
          scales, block_id, row, lane, kScaleBlocks);
      const float dot = dot8_half2_f32_dualacc(x4, w2);
      if ((u & 1) == 0) {
        acc = fmaf(dot, __half2float(scale_h), acc);
      } else {
        acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
      }
    }

    acc += acc_alt;
    acc = halfwarp_bfly_sum(acc);
    if (lane == 0) {
      partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
    }
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ scratch_x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int* __restrict__ ready_flags,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowBlocks = kHiddenSize / kBlockN;
  constexpr int kTasks = kRowBlocks * kSplitK;

  const int tid = threadIdx.x;
  if (blockIdx.x < kScaleBlocks) {
    const int block_id = blockIdx.x;
    for (int local_idx = tid; local_idx < kBlockK; local_idx += kThreads) {
      const int idx = block_id * kBlockK + local_idx;
      const int q_head = idx / kHeadDim;
      const int d = idx - q_head * kHeadDim;
      const int state_base = q_head * workspace_splits;
      float m = -INFINITY;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        m = fmaxf(m, partial_m[state_base + s]);
      }
      float l = 0.0f;
      float acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        const float weight = __expf(partial_m[state_base + s] - m);
        l += partial_l[state_base + s] * weight;
        acc += partial_acc[(state_base + s) * kHeadDim + d] * weight;
      }
      scratch_x[idx] = __float2half_rn(acc / l);
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      ready_flags[block_id] = 1;
    }
    return;
  }

  const int consumer_blocks = gridDim.x - kScaleBlocks;
  if (consumer_blocks <= 0) {
    return;
  }
  const int consumer_id = blockIdx.x - kScaleBlocks;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int lane_k = lane * 8;

  for (int task = consumer_id; task < kTasks; task += consumer_blocks) {
    const int split = task / kRowBlocks;
    const int out_block = task - split * kRowBlocks;
    const int row = out_block * kBlockN + row_in_block;
    const int block_begin = split * kBlocksPerSplit;

    float acc = 0.0f;
    float acc_alt = 0.0f;
#pragma unroll
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      if (tid == 0) {
        volatile int* flags = ready_flags;
        while (flags[block_id] == 0) {
        }
      }
      __syncthreads();
      const uint4 x4 = load_v4_evict_last(
          reinterpret_cast<const uint4*>(scratch_x + block_id * kBlockK + lane_k));
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<false, true>(
          scales, block_id, row, lane, kScaleBlocks);
      const float dot = dot8_half2_f32_dualacc(x4, w2);
      if ((u & 1) == 0) {
        acc = fmaf(dot, __half2float(scale_h), acc);
      } else {
        acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
      }
    }

    acc += acc_alt;
    acc = halfwarp_bfly_sum(acc);
    if (lane == 0) {
      o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
    }
  }
}

template <int kActiveSplits, int kSplitK>
__global__ __launch_bounds__(kThreads)
void prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ scratch_x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    float* __restrict__ o_partial,
    int* __restrict__ ready_flags,
    int workspace_splits) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  constexpr int kRowBlocks = kHiddenSize / kBlockN;
  constexpr int kTasks = kRowBlocks * kSplitK;

  const int tid = threadIdx.x;
  if (blockIdx.x < kScaleBlocks) {
    const int q_head = blockIdx.x;
    const int state_base = q_head * workspace_splits;
    for (int d = tid; d < kHeadDim; d += kThreads) {
      float m = -INFINITY;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        m = fmaxf(m, partial_m[state_base + s]);
      }
      float l = 0.0f;
      float acc = 0.0f;
#pragma unroll
      for (int s = 0; s < kActiveSplits; ++s) {
        const float weight = __expf(partial_m[state_base + s] - m);
        l += partial_l[state_base + s] * weight;
        acc += partial_acc[(state_base + s) * kHeadDim + d] * weight;
      }
      scratch_x[q_head * kHeadDim + d] = __float2half_rn(acc / l);
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      ready_flags[q_head] = 1;
    }
    return;
  }

  const int task = blockIdx.x - kScaleBlocks;
  if (task >= kTasks) {
    return;
  }
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int lane_k = lane * 8;
  const int split = task / kRowBlocks;
  const int out_block = task - split * kRowBlocks;
  const int row = out_block * kBlockN + row_in_block;
  const int block_begin = split * kBlocksPerSplit;

  float acc = 0.0f;
  float acc_alt = 0.0f;
#pragma unroll
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    if (tid == 0) {
      volatile int* flags = ready_flags;
      while (flags[block_id] == 0) {
      }
    }
    __syncthreads();
    const uint4 x4 = load_v4_evict_last(
        reinterpret_cast<const uint4*>(scratch_x + block_id * kBlockK + lane_k));
    const uint2 w2 = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const half scale_h = load_row_scale<false, true>(
        scales, block_id, row, lane, kScaleBlocks);
    const float dot = dot8_half2_f32_dualacc(x4, w2);
    if ((u & 1) == 0) {
      acc = fmaf(dot, __half2float(scale_h), acc);
    } else {
      acc_alt = fmaf(dot, __half2float(scale_h), acc_alt);
    }
  }

  acc += acc_alt;
  acc = halfwarp_bfly_sum(acc);
  if (lane == 0) {
    o_partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t split_k) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CUDA(partial);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(partial);
  CHECK_HALF(x);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  CHECK_FLOAT(partial);
  TORCH_CHECK(x.numel() == 2048 || x.numel() == 6144, "x.numel() must be 2048 or 6144");
  TORCH_CHECK(split_k > 1, "split_k must be > 1");
  TORCH_CHECK((x.numel() / kBlockK) % split_k == 0, "K/128 must be divisible by split_k");
  TORCH_CHECK(weight_fp8.numel() == (x.numel() / kBlockK) * kHiddenSize * kBlockK,
              "weight_fp8 must contain [K/128, 2048, 128]");
  TORCH_CHECK(scales.numel() == (x.numel() / kBlockK) * kHiddenSize,
              "scales must contain [2048, K/128] or [K/128, 2048]");
  TORCH_CHECK(partial.numel() == split_k * kHiddenSize, "partial must be [split_k, 2048]");
}

void check_reduce_inputs(
    const torch::Tensor& partial,
    const torch::Tensor& bias,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& sum_out,
    const torch::Tensor& norm_out,
    int64_t split_k) {
  CHECK_CUDA(partial);
  CHECK_CUDA(bias);
  CHECK_CUDA(residual);
  CHECK_CUDA(norm_weight);
  CHECK_CUDA(sum_out);
  CHECK_CUDA(norm_out);
  CHECK_CONTIGUOUS(partial);
  CHECK_CONTIGUOUS(bias);
  CHECK_CONTIGUOUS(residual);
  CHECK_CONTIGUOUS(norm_weight);
  CHECK_CONTIGUOUS(sum_out);
  CHECK_CONTIGUOUS(norm_out);
  CHECK_FLOAT(partial);
  CHECK_HALF(bias);
  CHECK_HALF(residual);
  CHECK_HALF(norm_weight);
  CHECK_HALF(sum_out);
  CHECK_HALF(norm_out);
  TORCH_CHECK(split_k > 1, "split_k must be > 1");
  TORCH_CHECK(partial.numel() == split_k * kHiddenSize, "partial must be [split_k, 2048]");
  TORCH_CHECK(bias.numel() == kHiddenSize, "bias must have 2048 elements");
  TORCH_CHECK(residual.numel() == kHiddenSize, "residual must have 2048 elements");
  TORCH_CHECK(norm_weight.numel() == kHiddenSize, "norm_weight must have 2048 elements");
  TORCH_CHECK(sum_out.numel() == kHiddenSize, "sum_out must have 2048 elements");
  TORCH_CHECK(norm_out.numel() == kHiddenSize, "norm_out must have 2048 elements");
}

void check_prefix_stage2_o_inputs(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    const torch::Tensor* barrier_counter,
    int64_t active_splits,
    int64_t split_k) {
  CHECK_CUDA(partial_m);
  CHECK_CUDA(partial_l);
  CHECK_CUDA(partial_acc);
  CHECK_CUDA(scratch_x);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CUDA(o_partial);
  if (barrier_counter != nullptr) {
    CHECK_CUDA(*barrier_counter);
  }
  CHECK_CONTIGUOUS(partial_m);
  CHECK_CONTIGUOUS(partial_l);
  CHECK_CONTIGUOUS(partial_acc);
  CHECK_CONTIGUOUS(scratch_x);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(o_partial);
  if (barrier_counter != nullptr) {
    CHECK_CONTIGUOUS(*barrier_counter);
  }
  CHECK_FLOAT(partial_m);
  CHECK_FLOAT(partial_l);
  CHECK_FLOAT(partial_acc);
  CHECK_HALF(scratch_x);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  CHECK_FLOAT(o_partial);
  if (barrier_counter != nullptr) {
    TORCH_CHECK(barrier_counter->scalar_type() == at::kInt, "barrier_counter must be int32");
    TORCH_CHECK(barrier_counter->numel() == 1, "barrier_counter must have one int32 element");
  }
  TORCH_CHECK(partial_m.dim() == 2, "partial_m must be [16, workspace_splits]");
  TORCH_CHECK(partial_l.sizes() == partial_m.sizes(), "partial_l shape mismatch");
  TORCH_CHECK(partial_m.size(0) == kQueryHeads, "partial_m must have 16 query heads");
  const int64_t workspace_splits = partial_m.size(1);
  TORCH_CHECK(
      partial_acc.sizes() == at::IntArrayRef({kQueryHeads, workspace_splits, kHeadDim}),
      "partial_acc must be [16, workspace_splits, 128]");
  TORCH_CHECK(scratch_x.numel() == kHiddenSize, "scratch_x must have 2048 fp16 elements");
  TORCH_CHECK(split_k == 4, "stage2/O fused experiment currently supports O split_k=4");
  TORCH_CHECK(active_splits >= 1 && active_splits <= workspace_splits, "active_splits out of range");
  TORCH_CHECK(active_splits <= 16, "stage2/O fused experiment supports active_splits <= 16");
  TORCH_CHECK(weight_fp8.numel() == (kHiddenSize / kBlockK) * kHiddenSize * kBlockK,
              "weight_fp8 must contain [16, 2048, 128]");
  TORCH_CHECK(scales.numel() == (kHiddenSize / kBlockK) * kHiddenSize,
              "scales must contain [2048, 16]");
  TORCH_CHECK(o_partial.sizes() == at::IntArrayRef({split_k, kHiddenSize}),
              "o_partial must be [4, 2048]");
}

void check_prefix_stage2_o_int4_inputs(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits,
    int64_t split_k) {
  CHECK_CUDA(partial_m);
  CHECK_CUDA(partial_l);
  CHECK_CUDA(partial_acc);
  CHECK_CUDA(scratch_x);
  CHECK_CUDA(weight_int4);
  CHECK_CUDA(scales);
  CHECK_CUDA(o_partial);
  CHECK_CONTIGUOUS(partial_m);
  CHECK_CONTIGUOUS(partial_l);
  CHECK_CONTIGUOUS(partial_acc);
  CHECK_CONTIGUOUS(scratch_x);
  CHECK_CONTIGUOUS(weight_int4);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(o_partial);
  CHECK_FLOAT(partial_m);
  CHECK_FLOAT(partial_l);
  CHECK_FLOAT(partial_acc);
  CHECK_HALF(scratch_x);
  CHECK_BYTE(weight_int4);
  CHECK_HALF(scales);
  CHECK_FLOAT(o_partial);
  TORCH_CHECK(partial_m.dim() == 2, "partial_m must be [16, workspace_splits]");
  TORCH_CHECK(partial_l.sizes() == partial_m.sizes(), "partial_l shape mismatch");
  TORCH_CHECK(partial_m.size(0) == kQueryHeads, "partial_m must have 16 query heads");
  const int64_t workspace_splits = partial_m.size(1);
  TORCH_CHECK(
      partial_acc.sizes() == at::IntArrayRef({kQueryHeads, workspace_splits, kHeadDim}),
      "partial_acc must be [16, workspace_splits, 128]");
  TORCH_CHECK(scratch_x.numel() == kHiddenSize, "scratch_x must have 2048 fp16 elements");
  TORCH_CHECK(split_k == 4, "stage2/O INT4 fused experiment currently supports O split_k=4");
  TORCH_CHECK(active_splits >= 1 && active_splits <= workspace_splits, "active_splits out of range");
  TORCH_CHECK(active_splits <= 16, "stage2/O INT4 fused experiment supports active_splits <= 16");
  TORCH_CHECK(weight_int4.numel() == (kHiddenSize / kBlockK) * kHiddenSize * (kBlockK / 2),
              "weight_int4 must contain [16, 2048, 64]");
  TORCH_CHECK(scales.numel() == (kHiddenSize / kBlockK) * kHiddenSize,
              "scales must contain [16, 2048]");
  TORCH_CHECK(o_partial.sizes() == at::IntArrayRef({split_k, kHiddenSize}),
              "o_partial must be [4, 2048]");
}

template <int kSplitK, bool kHasBias, int kReduceThreads>
__global__ __launch_bounds__(kReduceThreads)
void splitk_reduce_add_rmsnorm_2048_kernel(
    const float* __restrict__ partial,
    const half* __restrict__ bias,
    const half* __restrict__ residual,
    const half* __restrict__ norm_weight,
    half* __restrict__ sum_out,
    half* __restrict__ norm_out,
    float eps) {
  constexpr int kItems = kHiddenSize / kReduceThreads;
  __shared__ float warp_sums[kReduceThreads / 32];
  __shared__ float rstd_shared;

  const int tid = threadIdx.x;
  half summed[kItems];
  float thread_sumsq = 0.0f;

#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int idx = tid + item * kReduceThreads;
    float acc = 0.0f;
#pragma unroll
    for (int split = 0; split < kSplitK; ++split) {
      acc += partial[static_cast<int64_t>(split) * kHiddenSize + idx];
    }
    if constexpr (kHasBias) {
      acc += __half2float(bias[idx]);
    }
    acc += __half2float(residual[idx]);
    const half summed_h = __float2half_rn(acc);
    summed[item] = summed_h;
    const float summed_f = __half2float(summed_h);
    thread_sumsq = fmaf(summed_f, summed_f, thread_sumsq);
  }

  float block_sumsq = warp_sum(thread_sumsq);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) {
    warp_sums[warp] = block_sumsq;
  }
  __syncthreads();

  if (warp == 0) {
    float value = lane < (kReduceThreads / 32) ? warp_sums[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) {
      rstd_shared = rsqrtf(value / static_cast<float>(kHiddenSize) + eps);
    }
  }
  __syncthreads();

  const float rstd = rstd_shared;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int idx = tid + item * kReduceThreads;
    const half summed_h = summed[item];
    const half scaled_h = __float2half_rn(__half2float(summed_h) * rstd);
    const half norm_h = __float2half_rn(__half2float(scaled_h) * __half2float(norm_weight[idx]));
    sum_out[idx] = summed_h;
    norm_out[idx] = norm_h;
  }
}

template <int kSplitK, bool kHasBias>
__global__ __launch_bounds__(256)
void splitk_reduce_add_rmsnorm_2048_tritonshape_kernel(
    const float* __restrict__ partial,
    const half* __restrict__ bias,
    const half* __restrict__ residual,
    const half* __restrict__ norm_weight,
    half* __restrict__ sum_out,
    half* __restrict__ norm_out,
    float eps) {
  __shared__ float warp_sums[8];

  const int tid = threadIdx.x;
  const int base = tid * 8;

  const uint4 p0a = *reinterpret_cast<const uint4*>(partial + base);
  const uint4 p0b = *reinterpret_cast<const uint4*>(partial + base + 4);
  const uint4 p1a = *reinterpret_cast<const uint4*>(partial + kHiddenSize + base);
  const uint4 p1b = *reinterpret_cast<const uint4*>(partial + kHiddenSize + base + 4);
  const uint4 p2a = *reinterpret_cast<const uint4*>(partial + 2 * kHiddenSize + base);
  const uint4 p2b = *reinterpret_cast<const uint4*>(partial + 2 * kHiddenSize + base + 4);
  const uint4 res4 = *reinterpret_cast<const uint4*>(residual + base);

  float vals[8];
  vals[0] = __uint_as_float(p0a.x) + __uint_as_float(p1a.x) + __uint_as_float(p2a.x);
  vals[1] = __uint_as_float(p0a.y) + __uint_as_float(p1a.y) + __uint_as_float(p2a.y);
  vals[2] = __uint_as_float(p0a.z) + __uint_as_float(p1a.z) + __uint_as_float(p2a.z);
  vals[3] = __uint_as_float(p0a.w) + __uint_as_float(p1a.w) + __uint_as_float(p2a.w);
  vals[4] = __uint_as_float(p0b.x) + __uint_as_float(p1b.x) + __uint_as_float(p2b.x);
  vals[5] = __uint_as_float(p0b.y) + __uint_as_float(p1b.y) + __uint_as_float(p2b.y);
  vals[6] = __uint_as_float(p0b.z) + __uint_as_float(p1b.z) + __uint_as_float(p2b.z);
  vals[7] = __uint_as_float(p0b.w) + __uint_as_float(p1b.w) + __uint_as_float(p2b.w);

  half2 r01 = half2_from_u32(res4.x);
  half2 r23 = half2_from_u32(res4.y);
  half2 r45 = half2_from_u32(res4.z);
  half2 r67 = half2_from_u32(res4.w);
  float2 rf01 = __half22float2(r01);
  float2 rf23 = __half22float2(r23);
  float2 rf45 = __half22float2(r45);
  float2 rf67 = __half22float2(r67);
  vals[0] += rf01.x;
  vals[1] += rf01.y;
  vals[2] += rf23.x;
  vals[3] += rf23.y;
  vals[4] += rf45.x;
  vals[5] += rf45.y;
  vals[6] += rf67.x;
  vals[7] += rf67.y;

  if constexpr (kHasBias) {
    const uint4 bias4 = *reinterpret_cast<const uint4*>(bias + base);
    half2 b01 = half2_from_u32(bias4.x);
    half2 b23 = half2_from_u32(bias4.y);
    half2 b45 = half2_from_u32(bias4.z);
    half2 b67 = half2_from_u32(bias4.w);
    float2 bf01 = __half22float2(b01);
    float2 bf23 = __half22float2(b23);
    float2 bf45 = __half22float2(b45);
    float2 bf67 = __half22float2(b67);
    vals[0] += bf01.x;
    vals[1] += bf01.y;
    vals[2] += bf23.x;
    vals[3] += bf23.y;
    vals[4] += bf45.x;
    vals[5] += bf45.y;
    vals[6] += bf67.x;
    vals[7] += bf67.y;
  }

  half summed[8];
  float sumsq = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    summed[i] = __float2half_rn(vals[i]);
    const float v = __half2float(summed[i]);
    vals[i] = v;
    sumsq = fmaf(v, v, sumsq);
  }

  float warp_total = warp_bfly_sum(sumsq);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) {
    warp_sums[warp] = warp_total;
  }
  __syncthreads();

  float total_sumsq = tid < 8 ? warp_sums[tid] : 0.0f;
  total_sumsq = warp8_bfly_sum(total_sumsq);
  if (tid == 0) {
    warp_sums[0] = total_sumsq;
  }
  __syncthreads();

  const float rstd = rsqrtf(warp_sums[0] / static_cast<float>(kHiddenSize) + eps);
  const uint4 weight4 = *reinterpret_cast<const uint4*>(norm_weight + base);
  half2 w01 = half2_from_u32(weight4.x);
  half2 w23 = half2_from_u32(weight4.y);
  half2 w45 = half2_from_u32(weight4.z);
  half2 w67 = half2_from_u32(weight4.w);

  uint4 sum4;
  sum4.x = pack_half2_bits(summed[0], summed[1]);
  sum4.y = pack_half2_bits(summed[2], summed[3]);
  sum4.z = pack_half2_bits(summed[4], summed[5]);
  sum4.w = pack_half2_bits(summed[6], summed[7]);
  *reinterpret_cast<uint4*>(sum_out + base) = sum4;

  const half2 scaled01 = __halves2half2(__float2half_rn(vals[0] * rstd), __float2half_rn(vals[1] * rstd));
  const half2 scaled23 = __halves2half2(__float2half_rn(vals[2] * rstd), __float2half_rn(vals[3] * rstd));
  const half2 scaled45 = __halves2half2(__float2half_rn(vals[4] * rstd), __float2half_rn(vals[5] * rstd));
  const half2 scaled67 = __halves2half2(__float2half_rn(vals[6] * rstd), __float2half_rn(vals[7] * rstd));

  uint4 norm4;
  norm4.x = u32_from_half2(__hmul2(scaled01, w01));
  norm4.y = u32_from_half2(__hmul2(scaled23, w23));
  norm4.z = u32_from_half2(__hmul2(scaled45, w45));
  norm4.w = u32_from_half2(__hmul2(scaled67, w67));
  *reinterpret_cast<uint4*>(norm_out + base) = norm4;
}

__global__ __launch_bounds__(kThreads)
void down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    const half* __restrict__ norm_weight,
    half* __restrict__ sum_out,
    half* __restrict__ norm_out,
    float* __restrict__ partial,
    float eps) {
  constexpr int kIn = 6144;
  constexpr int kSplitK = 3;
  constexpr int kScaleBlocks = kIn / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  __shared__ float reduce_warp_sums[8];
  cg::grid_group grid = cg::this_grid();

  const int tid = threadIdx.x;
  const int row_in_block = (tid >> 4) & 15;
  const int lane = tid & 15;
  const int row = blockIdx.x * kBlockN + row_in_block;
  const int split = blockIdx.y;
  const int lane_k = lane * 8;
  const int block_begin = split * kBlocksPerSplit;

  float acc = 0.0f;
#pragma unroll 1
  for (int u = 0; u < kBlocksPerSplit; ++u) {
    const int block_id = block_begin + u;
    const uint4 x4 = load_warp_shared_x4(x, block_id, lane_k, lane);
    const uint2 w2 = load_v2_evict_first(
        reinterpret_cast<const uint2*>(
            weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
            static_cast<int64_t>(row) * kBlockK + lane_k));
    const half scale_h = load_row_scale<true, false>(scales, block_id, row, lane, kScaleBlocks);
    const float dot = dot8_half2_haccum(x4, w2);
    acc = fmaf(dot, __half2float(scale_h), acc);
  }

  acc = halfwarp_bfly_sum(acc);
  if (lane == 0) {
    partial[static_cast<int64_t>(split) * kHiddenSize + row] = acc;
  }

  grid.sync();

  if (blockIdx.x != 0 || blockIdx.y != 0) {
    return;
  }

  const int base = tid * 8;

  const uint4 p0a = *reinterpret_cast<const uint4*>(partial + base);
  const uint4 p0b = *reinterpret_cast<const uint4*>(partial + base + 4);
  const uint4 p1a = *reinterpret_cast<const uint4*>(partial + kHiddenSize + base);
  const uint4 p1b = *reinterpret_cast<const uint4*>(partial + kHiddenSize + base + 4);
  const uint4 p2a = *reinterpret_cast<const uint4*>(partial + 2 * kHiddenSize + base);
  const uint4 p2b = *reinterpret_cast<const uint4*>(partial + 2 * kHiddenSize + base + 4);
  const uint4 res4 = *reinterpret_cast<const uint4*>(residual + base);

  float vals[8];
  vals[0] = __uint_as_float(p0a.x) + __uint_as_float(p1a.x) + __uint_as_float(p2a.x);
  vals[1] = __uint_as_float(p0a.y) + __uint_as_float(p1a.y) + __uint_as_float(p2a.y);
  vals[2] = __uint_as_float(p0a.z) + __uint_as_float(p1a.z) + __uint_as_float(p2a.z);
  vals[3] = __uint_as_float(p0a.w) + __uint_as_float(p1a.w) + __uint_as_float(p2a.w);
  vals[4] = __uint_as_float(p0b.x) + __uint_as_float(p1b.x) + __uint_as_float(p2b.x);
  vals[5] = __uint_as_float(p0b.y) + __uint_as_float(p1b.y) + __uint_as_float(p2b.y);
  vals[6] = __uint_as_float(p0b.z) + __uint_as_float(p1b.z) + __uint_as_float(p2b.z);
  vals[7] = __uint_as_float(p0b.w) + __uint_as_float(p1b.w) + __uint_as_float(p2b.w);

  half2 r01 = half2_from_u32(res4.x);
  half2 r23 = half2_from_u32(res4.y);
  half2 r45 = half2_from_u32(res4.z);
  half2 r67 = half2_from_u32(res4.w);
  const float2 rf01 = __half22float2(r01);
  const float2 rf23 = __half22float2(r23);
  const float2 rf45 = __half22float2(r45);
  const float2 rf67 = __half22float2(r67);
  vals[0] += rf01.x;
  vals[1] += rf01.y;
  vals[2] += rf23.x;
  vals[3] += rf23.y;
  vals[4] += rf45.x;
  vals[5] += rf45.y;
  vals[6] += rf67.x;
  vals[7] += rf67.y;

  half summed[8];
  float sumsq = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    summed[i] = __float2half_rn(vals[i]);
    const float v = __half2float(summed[i]);
    vals[i] = v;
    sumsq = fmaf(v, v, sumsq);
  }

  float warp_total = warp_bfly_sum(sumsq);
  const int reduce_lane = tid & 31;
  const int reduce_warp = tid >> 5;
  if (reduce_lane == 0) {
    reduce_warp_sums[reduce_warp] = warp_total;
  }
  __syncthreads();

  float total_sumsq = tid < 8 ? reduce_warp_sums[tid] : 0.0f;
  total_sumsq = warp8_bfly_sum(total_sumsq);
  if (tid == 0) {
    reduce_warp_sums[0] = total_sumsq;
  }
  __syncthreads();

  const float rstd = rsqrtf(reduce_warp_sums[0] / static_cast<float>(kHiddenSize) + eps);
  const uint4 weight4 = *reinterpret_cast<const uint4*>(norm_weight + base);
  half2 w01 = half2_from_u32(weight4.x);
  half2 w23 = half2_from_u32(weight4.y);
  half2 w45 = half2_from_u32(weight4.z);
  half2 w67 = half2_from_u32(weight4.w);

  uint4 sum4;
  sum4.x = pack_half2_bits(summed[0], summed[1]);
  sum4.y = pack_half2_bits(summed[2], summed[3]);
  sum4.z = pack_half2_bits(summed[4], summed[5]);
  sum4.w = pack_half2_bits(summed[6], summed[7]);
  *reinterpret_cast<uint4*>(sum_out + base) = sum4;

  const half2 scaled01 =
      __halves2half2(__float2half_rn(vals[0] * rstd), __float2half_rn(vals[1] * rstd));
  const half2 scaled23 =
      __halves2half2(__float2half_rn(vals[2] * rstd), __float2half_rn(vals[3] * rstd));
  const half2 scaled45 =
      __halves2half2(__float2half_rn(vals[4] * rstd), __float2half_rn(vals[5] * rstd));
  const half2 scaled67 =
      __halves2half2(__float2half_rn(vals[6] * rstd), __float2half_rn(vals[7] * rstd));

  uint4 norm4;
  norm4.x = u32_from_half2(__hmul2(scaled01, w01));
  norm4.y = u32_from_half2(__hmul2(scaled23, w23));
  norm4.z = u32_from_half2(__hmul2(scaled45, w45));
  norm4.w = u32_from_half2(__hmul2(scaled67, w67));
  *reinterpret_cast<uint4*>(norm_out + base) = norm4;
}

template <int kRowsPerBlock, int kThreadsPerBlock>
__global__ __launch_bounds__(kThreadsPerBlock)
void down_fp8e4b15_kblock_kscale_split3_sum_sumsq_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  constexpr int kIn = 6144;
  constexpr int kSplitK = 3;
  constexpr int kScaleBlocks = kIn / kBlockK;
  constexpr int kBlocksPerSplit = kScaleBlocks / kSplitK;
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int halfwarp = tid >> 4;
  const int lane = tid & 15;
  const int split = halfwarp / kRowsPerBlock;
  const int row_in_tile = halfwarp - split * kRowsPerBlock;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;
  const int lane_k = lane * 8;

  if (split < kSplitK) {
    const int block_begin = split * kBlocksPerSplit;
    float acc = 0.0f;
#pragma unroll 1
    for (int u = 0; u < kBlocksPerSplit; ++u) {
      const int block_id = block_begin + u;
      const uint4 x4 = load_warp_shared_x4(x, block_id, lane_k, lane);
      const uint2 w2 = load_v2_evict_first(
          reinterpret_cast<const uint2*>(
              weight_fp8 + static_cast<int64_t>(block_id) * kHiddenSize * kBlockK +
              static_cast<int64_t>(row) * kBlockK + lane_k));
      const half scale_h = load_row_scale<true, false>(scales, block_id, row, lane, kScaleBlocks);
      const float dot = dot8_half2_haccum(x4, w2);
      acc = fmaf(dot, __half2float(scale_h), acc);
    }
    const float part = halfwarp_bfly_sum(acc);
    if (lane == 0) {
      split_acc[split][row_in_tile] = part;
    }
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total = split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    float local = 0.0f;
#pragma unroll
    for (int i = 0; i < kRowsPerBlock; ++i) {
      local += row_sumsq[i];
    }
    tile_sumsq[blockIdx.x] = local;
  }
}

template <int kTileCount>
__global__ __launch_bounds__(256)
void rmsnorm_from_sum_sumsq_2048_kernel(
    const half* __restrict__ sum_out,
    const float* __restrict__ tile_sumsq,
    const half* __restrict__ norm_weight,
    half* __restrict__ norm_out,
    float eps) {
  __shared__ float warp_sums[8];
  __shared__ float rstd_shared;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float local = 0.0f;
  for (int idx = tid; idx < kTileCount; idx += 256) {
    local += tile_sumsq[idx];
  }

  const float warp_total = warp_sum(local);
  if (lane == 0) {
    warp_sums[warp] = warp_total;
  }
  __syncthreads();

  float total = tid < 8 ? warp_sums[tid] : 0.0f;
  total = warp_sum(total);
  if (tid == 0) {
    rstd_shared = rsqrtf(total / static_cast<float>(kHiddenSize) + eps);
  }
  __syncthreads();

  const float rstd = rstd_shared;
  const int base = tid * 8;
  const uint4 sum4 = *reinterpret_cast<const uint4*>(sum_out + base);
  const uint4 weight4 = *reinterpret_cast<const uint4*>(norm_weight + base);
  const half2 s01 = half2_from_u32(sum4.x);
  const half2 s23 = half2_from_u32(sum4.y);
  const half2 s45 = half2_from_u32(sum4.z);
  const half2 s67 = half2_from_u32(sum4.w);
  const half2 w01 = half2_from_u32(weight4.x);
  const half2 w23 = half2_from_u32(weight4.y);
  const half2 w45 = half2_from_u32(weight4.z);
  const half2 w67 = half2_from_u32(weight4.w);
  const float2 f01 = __half22float2(s01);
  const float2 f23 = __half22float2(s23);
  const float2 f45 = __half22float2(s45);
  const float2 f67 = __half22float2(s67);

  uint4 norm4;
  norm4.x = u32_from_half2(__hmul2(
      __halves2half2(__float2half_rn(f01.x * rstd), __float2half_rn(f01.y * rstd)), w01));
  norm4.y = u32_from_half2(__hmul2(
      __halves2half2(__float2half_rn(f23.x * rstd), __float2half_rn(f23.y * rstd)), w23));
  norm4.z = u32_from_half2(__hmul2(
      __halves2half2(__float2half_rn(f45.x * rstd), __float2half_rn(f45.y * rstd)), w45));
  norm4.w = u32_from_half2(__hmul2(
      __halves2half2(__float2half_rn(f67.x * rstd), __float2half_rn(f67.y * rstd)), w67));
  *reinterpret_cast<uint4*>(norm_out + base) = norm4;
}

template <
    bool kKscale,
    int kIn,
    int kSplitK,
    bool kUnroll,
    bool kScaleBroadcast,
    bool kDirectStore,
    bool kWarpSharedX,
    bool kCtaSharedX,
    bool kLowReg,
    bool kPreloadSharedX,
    bool kHaccum = false,
    bool kF32DualAcc = false,
    bool kOuterDualAcc = false>
void launch_partial(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& partial) {
  dim3 grid(kHiddenSize / kBlockN, kSplitK);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemv_splitk_partial_fp8e4b15_tritonshape_kernel<
      kKscale, kIn, kSplitK, kUnroll, kScaleBroadcast, kDirectStore, kWarpSharedX, kCtaSharedX, kLowReg, kPreloadSharedX, kHaccum, kF32DualAcc, kOuterDualAcc>
      <<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<float*>(partial.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    bool kKscale,
    bool kScaleBroadcast,
    bool kDirectStore,
    bool kWarpSharedX,
    bool kCtaSharedX,
    bool kLowReg,
    bool kPreloadSharedX = false,
    bool kHaccum = false,
    bool kF32DualAcc = false,
    bool kOuterDualAcc = false>
void dispatch_partial_shape(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t split_k) {
  if (x.numel() == 2048 && split_k == 4) {
    launch_partial<kKscale, 2048, 4, true, kScaleBroadcast, kDirectStore, kWarpSharedX, kCtaSharedX, kLowReg, kPreloadSharedX, kHaccum, kF32DualAcc, kOuterDualAcc>(
        x, weight_fp8, scales, partial);
  } else if (x.numel() == 6144 && split_k == 3) {
    launch_partial<kKscale, 6144, 3, false, kScaleBroadcast, kDirectStore, kWarpSharedX, kCtaSharedX, kLowReg, kPreloadSharedX, kHaccum, kF32DualAcc, kOuterDualAcc>(
        x, weight_fp8, scales, partial);
  } else if (x.numel() == 6144 && split_k == 2) {
    launch_partial<kKscale, 6144, 2, false, kScaleBroadcast, kDirectStore, kWarpSharedX, kCtaSharedX, kLowReg, kPreloadSharedX, kHaccum, kF32DualAcc, kOuterDualAcc>(
        x, weight_fp8, scales, partial);
  } else if (x.numel() == 2048 && split_k == 2) {
    launch_partial<kKscale, 2048, 2, true, kScaleBroadcast, kDirectStore, kWarpSharedX, kCtaSharedX, kLowReg, kPreloadSharedX, kHaccum, kF32DualAcc, kOuterDualAcc>(
        x, weight_fp8, scales, partial);
  } else if (x.numel() == 6144 && split_k == 6) {
    launch_partial<kKscale, 6144, 6, false, kScaleBroadcast, kDirectStore, kWarpSharedX, kCtaSharedX, kLowReg, kPreloadSharedX, kHaccum, kF32DualAcc, kOuterDualAcc>(
        x, weight_fp8, scales, partial);
  } else {
    TORCH_CHECK(false, "unsupported CUDA FP8 partial GEMV shape");
  }
}

template <bool kKscale>
void dispatch_partial(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t split_k,
    int64_t variant) {
  switch (variant) {
    case 0:
      dispatch_partial_shape<kKscale, false, false, false, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 1:
      dispatch_partial_shape<kKscale, true, false, false, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 2:
      dispatch_partial_shape<kKscale, false, true, false, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 3:
      dispatch_partial_shape<kKscale, true, true, false, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 4:
      dispatch_partial_shape<kKscale, false, false, true, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 5:
      dispatch_partial_shape<kKscale, true, false, true, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 6:
      dispatch_partial_shape<kKscale, false, true, true, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 7:
      dispatch_partial_shape<kKscale, true, true, true, false, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 8:
      dispatch_partial_shape<kKscale, false, false, false, true, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 9:
      dispatch_partial_shape<kKscale, true, false, false, true, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 10:
      dispatch_partial_shape<kKscale, false, true, false, true, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 11:
      dispatch_partial_shape<kKscale, true, true, false, true, false>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 12:
      dispatch_partial_shape<kKscale, false, false, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 13:
      dispatch_partial_shape<kKscale, true, false, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 14:
      dispatch_partial_shape<kKscale, false, true, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 15:
      dispatch_partial_shape<kKscale, true, true, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 16:
      dispatch_partial_shape<kKscale, false, false, true, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 17:
      dispatch_partial_shape<kKscale, true, false, true, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 18:
      dispatch_partial_shape<kKscale, false, true, true, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 19:
      dispatch_partial_shape<kKscale, true, true, true, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 20:
      dispatch_partial_shape<kKscale, false, false, false, true, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 21:
      dispatch_partial_shape<kKscale, true, false, false, true, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 22:
      dispatch_partial_shape<kKscale, false, true, false, true, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 23:
      dispatch_partial_shape<kKscale, true, true, false, true, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 24:
      dispatch_partial_shape<kKscale, false, false, false, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 25:
      dispatch_partial_shape<kKscale, false, true, false, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 26:
      dispatch_partial_shape<kKscale, false, true, false, false, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 27:
      dispatch_partial_shape<kKscale, false, true, true, false, false, false, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    case 32:
      dispatch_partial_shape<kKscale, true, true, false, false, false, false, true, true>(
          x, weight_fp8, scales, partial, split_k);
      return;
    default:
      TORCH_CHECK(false, "unsupported CUDA FP8 partial GEMV variant");
  }
}

template <int kSplitK, bool kHasBias, int kReduceThreads>
void launch_reduce(
    const torch::Tensor& partial,
    const torch::Tensor& bias,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& sum_out,
    const torch::Tensor& norm_out,
    double eps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  splitk_reduce_add_rmsnorm_2048_kernel<kSplitK, kHasBias, kReduceThreads>
      <<<1, kReduceThreads, 0, stream>>>(
          reinterpret_cast<const float*>(partial.data_ptr<float>()),
          reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
          reinterpret_cast<half*>(sum_out.data_ptr<at::Half>()),
          reinterpret_cast<half*>(norm_out.data_ptr<at::Half>()),
          static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kHasBias>
void launch_reduce_tritonshape(
    const torch::Tensor& partial,
    const torch::Tensor& bias,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& sum_out,
    const torch::Tensor& norm_out,
    double eps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  splitk_reduce_add_rmsnorm_2048_tritonshape_kernel<3, kHasBias>
      <<<1, 256, 0, stream>>>(
          reinterpret_cast<const float*>(partial.data_ptr<float>()),
          reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
          reinterpret_cast<half*>(sum_out.data_ptr<at::Half>()),
          reinterpret_cast<half*>(norm_out.data_ptr<at::Half>()),
          static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& sum_out,
    const torch::Tensor& norm_out,
    const torch::Tensor& partial,
    double eps) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int cooperative = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative != 0, "device does not support cooperative launch");

  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary_kernel,
      kThreads,
      0));
  const int total_blocks = (kHiddenSize / kBlockN) * 3;
  TORCH_CHECK(total_blocks <= sm_count * blocks_per_sm, "down FP8 fused boundary grid does not fit concurrently");

  auto stream = at::cuda::getCurrentCUDAStream();
  const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const uint8_t* weight_ptr = reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>());
  const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const half* residual_ptr = reinterpret_cast<const half*>(residual.data_ptr<at::Half>());
  const half* norm_weight_ptr = reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>());
  half* sum_out_ptr = reinterpret_cast<half*>(sum_out.data_ptr<at::Half>());
  half* norm_out_ptr = reinterpret_cast<half*>(norm_out.data_ptr<at::Half>());
  float* partial_ptr = partial.data_ptr<float>();
  float eps_f = static_cast<float>(eps);
  void* args[] = {
      &x_ptr,
      &weight_ptr,
      &scales_ptr,
      &residual_ptr,
      &norm_weight_ptr,
      &sum_out_ptr,
      &norm_out_ptr,
      &partial_ptr,
      &eps_f,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary_kernel),
      dim3(kHiddenSize / kBlockN, 3, 1),
      dim3(kThreads, 1, 1),
      args,
      0,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_down_fp8e4b15_kblock_kscale_split3_sum_sumsq(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& residual,
    const torch::Tensor& sum_out,
    const torch::Tensor& tile_sumsq) {
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int kHalfwarps = 3 * kRowsPerBlock;
  constexpr int kThreadsPerBlock = ((kHalfwarps * 16 + 31) / 32) * 32;
  down_fp8e4b15_kblock_kscale_split3_sum_sumsq_kernel<kRowsPerBlock, kThreadsPerBlock>
      <<<kHiddenSize / kRowsPerBlock, kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
          reinterpret_cast<half*>(sum_out.data_ptr<at::Half>()),
          tile_sumsq.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kTileCount>
void launch_rmsnorm_from_sum_sumsq_2048(
    const torch::Tensor& sum_out,
    const torch::Tensor& tile_sumsq,
    const torch::Tensor& norm_weight,
    const torch::Tensor& norm_out,
    double eps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_from_sum_sumsq_2048_kernel<kTileCount>
      <<<1, 256, 0, stream>>>(
          reinterpret_cast<const half*>(sum_out.data_ptr<at::Half>()),
          tile_sumsq.data_ptr<float>(),
          reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
          reinterpret_cast<half*>(norm_out.data_ptr<at::Half>()),
          static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t coop_blocks) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int cooperative = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative != 0, "device does not support cooperative launch");

  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      prefix_stage2_o_partial_fp8e4b15_kblock_major_kernel<kActiveSplits, 4>,
      kThreads,
      0));
  const int max_blocks = std::max(1, sm_count * blocks_per_sm);
  int blocks = static_cast<int>(coop_blocks);
  if (blocks <= 0) {
    blocks = std::min(max_blocks, std::max(1, sm_count));
  }
  TORCH_CHECK(blocks <= max_blocks, "requested cooperative grid does not fit concurrently");

  auto stream = at::cuda::getCurrentCUDAStream();
  const float* partial_m_ptr = partial_m.data_ptr<float>();
  const float* partial_l_ptr = partial_l.data_ptr<float>();
  const float* partial_acc_ptr = partial_acc.data_ptr<float>();
  half* scratch_ptr = reinterpret_cast<half*>(scratch_x.data_ptr<at::Half>());
  const uint8_t* weight_ptr = reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>());
  const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  float* o_partial_ptr = o_partial.data_ptr<float>();
  int workspace_splits = static_cast<int>(partial_m.size(1));
  void* args[] = {
      &partial_m_ptr,
      &partial_l_ptr,
      &partial_acc_ptr,
      &scratch_ptr,
      &weight_ptr,
      &scales_ptr,
      &o_partial_ptr,
      &workspace_splits,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(prefix_stage2_o_partial_fp8e4b15_kblock_major_kernel<kActiveSplits, 4>),
      dim3(blocks),
      dim3(kThreads),
      args,
      0,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits,
    int64_t coop_blocks) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<1>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<2>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<3>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<4>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<5>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<6>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<7>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<8>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<9>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<10>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<11>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<12>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<13>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<14>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<15>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_active<16>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    const torch::Tensor& barrier_counter,
    int64_t blocks_hint) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_kernel<kActiveSplits, 4>,
      kThreads,
      0));
  const int max_blocks = std::max(1, sm_count * blocks_per_sm);
  int blocks = static_cast<int>(blocks_hint);
  if (blocks <= 0) {
    blocks = std::min(max_blocks, sm_count * 4);
  }
  TORCH_CHECK(blocks <= max_blocks, "requested barrier grid may deadlock because it cannot fully reside");

  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(barrier_counter.data_ptr<int>(), 0, sizeof(int), stream));
  prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_kernel<kActiveSplits, 4>
      <<<blocks, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<half*>(scratch_x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          reinterpret_cast<unsigned int*>(barrier_counter.data_ptr<int>()),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    const torch::Tensor& barrier_counter,
    int64_t active_splits,
    int64_t blocks_hint) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<1>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<2>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<3>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<4>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<5>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<6>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<7>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<8>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<9>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<10>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<11>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<12>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<13>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<14>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<15>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier_active<16>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, barrier_counter, blocks_hint);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / kBlockN, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_kernel<kActiveSplits, 4>
      <<<grid, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits, int kRowBlocksPerCta>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid((kHiddenSize / kBlockN) / kRowBlocksPerCta, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_kernel<kActiveSplits, 4, kRowBlocksPerCta>
      <<<grid, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / 32, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_kernel<kActiveSplits, 4>
      <<<grid, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / 32, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_kernel<kActiveSplits, 4>
      <<<grid, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / 16, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_kernel<kActiveSplits, 4>
      <<<grid, 128, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / 64, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_kernel<kActiveSplits, 4>
      <<<grid, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / 64, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_kernel<kActiveSplits, 4>
      <<<grid, 512, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <
    int kThreadsPerCta,
    int kRowsPerCta,
    int kDotMode,
    int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  static_assert(kHiddenSize % kRowsPerCta == 0, "generic stage2/O rowgroup expects exact hidden tiling");
  dim3 grid(kHiddenSize / kRowsPerCta, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_kernel<
      kActiveSplits, 4, kThreadsPerCta, kRowsPerCta, kDotMode>
      <<<grid, kThreadsPerCta, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    int kThreadsPerCta,
    int kRowsPerCta,
    int kDotMode>
void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic_active<
          kThreadsPerCta, kRowsPerCta, kDotMode, 16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <
    int kThreadsPerCta,
    int kRowsPerCta,
    int kActiveSplits>
void launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  static_assert(kHiddenSize % kRowsPerCta == 0, "int4 stage2/O rowgroup expects exact hidden tiling");
  dim3 grid(kHiddenSize / kRowsPerCta, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_kernel<
      kActiveSplits, 4, kThreadsPerCta, kRowsPerCta>
      <<<grid, kThreadsPerCta, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kThreadsPerCta, int kRowsPerCta>
void dispatch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 1>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 2>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 3>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 4>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 5>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 6>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 7>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 8>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 9>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 10>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 11>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 12>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 13>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 14>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 15>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4_active<
          kThreadsPerCta, kRowsPerCta, 16>(
          partial_m, partial_l, partial_acc, weight_int4, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int dot_mode) {
  dim3 grid(kHiddenSize / 64, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (dot_mode == 1) {
    prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_kernel<kActiveSplits, 4, 1>
        <<<grid, kThreads, 0, stream>>>(
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_acc.data_ptr<float>(),
            reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            o_partial.data_ptr<float>(),
            static_cast<int>(partial_m.size(1)));
  } else {
    prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_kernel<kActiveSplits, 4, 0>
        <<<grid, kThreads, 0, stream>>>(
            partial_m.data_ptr<float>(),
            partial_l.data_ptr<float>(),
            partial_acc.data_ptr<float>(),
            reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            o_partial.data_ptr<float>(),
            static_cast<int>(partial_m.size(1)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits,
    int dot_mode) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, dot_mode);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / 64, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_kernel<kActiveSplits, 4>
      <<<grid, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial) {
  dim3 grid(kHiddenSize / 64, 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_kernel<kActiveSplits, 4>
      <<<grid, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<1>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<2>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<3>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<4>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<5>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<6>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<7>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<8>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<9>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<10>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<11>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<12>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<13>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<14>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<15>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2_active<16>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kRowBlocksPerCta>
void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active_splits(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<1, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<2, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<3, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<4, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<5, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<6, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<7, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<8, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<9, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<10, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<11, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<12, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<13, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<14, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<15, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active<16, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits,
    int64_t row_blocks_per_cta) {
  if (row_blocks_per_cta == -2) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -3) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup2_lane8_pre4(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -13) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup1_lane8_pre4_t128(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -18) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_serial2(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -19) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_pre4_t512(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -20) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<1024, 128, 0>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -21) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<512, 64, 1>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -22) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<512, 64, 2>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -23) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<512, 64, 3>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -24) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<512, 64, 4>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -25) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<1024, 128, 2>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -26) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<256, 32, 2>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -27) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_lane8_pre4_generic<128, 16, 2>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -28) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits, 1);
    return;
  }
  if (row_blocks_per_cta == -29) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4_pre4(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits, 0);
    return;
  }
  if (row_blocks_per_cta == -4) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane4(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == -8) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup4_lane8_serial2(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
    return;
  }
  if (row_blocks_per_cta == 2) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active_splits<2>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
  } else if (row_blocks_per_cta == 1) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active_splits<1>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
  } else if (row_blocks_per_cta == 4) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active_splits<4>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
  } else if (row_blocks_per_cta == 8) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup_active_splits<8>(
        partial_m, partial_l, partial_acc, weight_fp8, scales, o_partial, active_splits);
  } else {
    TORCH_CHECK(false, "row_blocks_per_cta must be 1, 2, 4, or 8");
  }
}

template <int kActiveSplits, int kRowBlocksPerCta>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t coop_blocks) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int cooperative = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative != 0, "device does not support cooperative launch");

  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_kernel<kActiveSplits, 4, kRowBlocksPerCta>,
      kThreads,
      0));
  const int max_blocks = std::max(1, sm_count * blocks_per_sm);
  int blocks = static_cast<int>(coop_blocks);
  if (blocks <= 0) {
    blocks = std::min(max_blocks, std::max(1, sm_count));
  }
  TORCH_CHECK(blocks <= max_blocks, "requested cooperative grid does not fit concurrently");

  auto stream = at::cuda::getCurrentCUDAStream();
  const float* partial_m_ptr = partial_m.data_ptr<float>();
  const float* partial_l_ptr = partial_l.data_ptr<float>();
  const float* partial_acc_ptr = partial_acc.data_ptr<float>();
  half* scratch_ptr = reinterpret_cast<half*>(scratch_x.data_ptr<at::Half>());
  const uint8_t* weight_ptr = reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>());
  const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  float* o_partial_ptr = o_partial.data_ptr<float>();
  int workspace_splits = static_cast<int>(partial_m.size(1));
  void* args[] = {
      &partial_m_ptr,
      &partial_l_ptr,
      &partial_acc_ptr,
      &scratch_ptr,
      &weight_ptr,
      &scales_ptr,
      &o_partial_ptr,
      &workspace_splits,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_kernel<kActiveSplits, 4, kRowBlocksPerCta>),
      dim3(blocks),
      dim3(kThreads),
      args,
      0,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowBlocksPerCta>
void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active_splits(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits,
    int64_t coop_blocks) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<1, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<2, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<3, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<4, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<5, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<6, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<7, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<8, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<9, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<10, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<11, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<12, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<13, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<14, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<15, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active<16, kRowBlocksPerCta>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, coop_blocks);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    int64_t active_splits,
    int64_t coop_blocks,
    int64_t row_blocks_per_cta) {
  if (row_blocks_per_cta == 2) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active_splits<2>(
        partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, active_splits, coop_blocks);
  } else if (row_blocks_per_cta == 4) {
    dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup_active_splits<4>(
        partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, active_splits, coop_blocks);
  } else {
    TORCH_CHECK(false, "row_blocks_per_cta must be 2 or 4");
  }
}

void launch_gemv_splitk_partial_fp8e4b15_kblock_major_taskloop(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t split_k,
    int64_t blocks) {
  TORCH_CHECK(x.numel() == kHiddenSize, "taskloop diagnostic currently supports K=2048");
  TORCH_CHECK(split_k == 4, "taskloop diagnostic currently supports split_k=4");
  int grid_blocks = static_cast<int>(blocks);
  if (grid_blocks <= 0) {
    grid_blocks = (kHiddenSize / kBlockN) * 4;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  gemv_splitk_partial_fp8e4b15_kblock_major_taskloop_kernel<4>
      <<<grid_blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    const torch::Tensor& ready_flags,
    int64_t coop_blocks) {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int cooperative = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative != 0, "device does not support cooperative launch");

  int sm_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm,
      prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_kernel<kActiveSplits, 4>,
      kThreads,
      0));
  const int max_blocks = std::max(1, sm_count * blocks_per_sm);
  int blocks = static_cast<int>(coop_blocks);
  if (blocks <= 0) {
    blocks = std::min(max_blocks, std::max(1, sm_count));
  }
  TORCH_CHECK(blocks > (kHiddenSize / kBlockK), "producer-consumer grid must have consumer CTAs");
  TORCH_CHECK(blocks <= max_blocks, "requested cooperative grid does not fit concurrently");

  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(ready_flags.data_ptr<int>(), 0, sizeof(int) * (kHiddenSize / kBlockK), stream));
  const float* partial_m_ptr = partial_m.data_ptr<float>();
  const float* partial_l_ptr = partial_l.data_ptr<float>();
  const float* partial_acc_ptr = partial_acc.data_ptr<float>();
  half* scratch_ptr = reinterpret_cast<half*>(scratch_x.data_ptr<at::Half>());
  const uint8_t* weight_ptr = reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>());
  const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  float* o_partial_ptr = o_partial.data_ptr<float>();
  int* ready_ptr = ready_flags.data_ptr<int>();
  int workspace_splits = static_cast<int>(partial_m.size(1));
  void* args[] = {
      &partial_m_ptr,
      &partial_l_ptr,
      &partial_acc_ptr,
      &scratch_ptr,
      &weight_ptr,
      &scales_ptr,
      &o_partial_ptr,
      &ready_ptr,
      &workspace_splits,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_kernel<kActiveSplits, 4>),
      dim3(blocks),
      dim3(kThreads),
      args,
      0,
      stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    const torch::Tensor& ready_flags,
    int64_t active_splits,
    int64_t coop_blocks) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<1>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<2>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<3>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<4>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<5>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<6>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<7>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<8>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<9>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<10>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<11>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<12>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<13>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<14>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<15>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_active<16>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags, coop_blocks);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kActiveSplits>
void launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    const torch::Tensor& ready_flags) {
  constexpr int kScaleBlocks = kHiddenSize / kBlockK;
  constexpr int kTasks = (kHiddenSize / kBlockN) * 4;
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(ready_flags.data_ptr<int>(), 0, sizeof(int) * kScaleBlocks, stream));
  prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_kernel<kActiveSplits, 4>
      <<<kScaleBlocks + kTasks, kThreads, 0, stream>>>(
          partial_m.data_ptr<float>(),
          partial_l.data_ptr<float>(),
          partial_acc.data_ptr<float>(),
          reinterpret_cast<half*>(scratch_x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          o_partial.data_ptr<float>(),
          ready_flags.data_ptr<int>(),
          static_cast<int>(partial_m.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& scratch_x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& o_partial,
    const torch::Tensor& ready_flags,
    int64_t active_splits) {
  switch (active_splits) {
    case 1:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<1>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 2:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<2>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 3:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<3>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 4:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<4>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 5:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<5>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 6:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<6>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 7:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<7>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 8:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<8>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 9:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<9>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 10:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<10>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 11:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<11>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 12:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<12>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 13:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<13>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 14:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<14>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 15:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<15>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    case 16:
      launch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full_active<16>(
          partial_m, partial_l, partial_acc, scratch_x, weight_fp8, scales, o_partial, ready_flags);
      return;
    default:
      TORCH_CHECK(false, "active_splits must be in [1, 16]");
  }
}

template <int kSplitK, bool kHasBias>
void dispatch_reduce_threads(
    const torch::Tensor& partial,
    const torch::Tensor& bias,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& sum_out,
    const torch::Tensor& norm_out,
    double eps,
    int64_t threads) {
  if (threads == 128) {
    launch_reduce<kSplitK, kHasBias, 128>(partial, bias, residual, norm_weight, sum_out, norm_out, eps);
  } else if (threads == 256) {
    launch_reduce<kSplitK, kHasBias, 256>(partial, bias, residual, norm_weight, sum_out, norm_out, eps);
  } else if (threads == 512) {
    launch_reduce<kSplitK, kHasBias, 512>(partial, bias, residual, norm_weight, sum_out, norm_out, eps);
  } else {
    TORCH_CHECK(false, "reduce threads must be 128, 256, or 512");
  }
}

template <bool kHasBias>
void dispatch_reduce_split(
    const torch::Tensor& partial,
    const torch::Tensor& bias,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& sum_out,
    const torch::Tensor& norm_out,
    int64_t split_k,
    double eps,
    int64_t threads) {
  if (split_k == 2) {
    dispatch_reduce_threads<2, kHasBias>(partial, bias, residual, norm_weight, sum_out, norm_out, eps, threads);
  } else if (split_k == 3) {
    dispatch_reduce_threads<3, kHasBias>(partial, bias, residual, norm_weight, sum_out, norm_out, eps, threads);
  } else if (split_k == 4) {
    dispatch_reduce_threads<4, kHasBias>(partial, bias, residual, norm_weight, sum_out, norm_out, eps, threads);
  } else if (split_k == 6) {
    dispatch_reduce_threads<6, kHasBias>(partial, bias, residual, norm_weight, sum_out, norm_out, eps, threads);
  } else {
    TORCH_CHECK(false, "unsupported reduce split_k");
  }
}

}  // namespace

void gemv_splitk_partial_fp8e4b15_kblock_major(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor partial,
    int64_t split_k,
    int64_t variant) {
  check_inputs(x, weight_fp8, scales, partial, split_k);
  dispatch_partial<false>(x, weight_fp8, scales, partial, split_k, variant);
}

void gemv_splitk_partial_fp8e4b15_kblock_major_taskloop(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor partial,
    int64_t split_k,
    int64_t blocks) {
  check_inputs(x, weight_fp8, scales, partial, split_k);
  launch_gemv_splitk_partial_fp8e4b15_kblock_major_taskloop(x, weight_fp8, scales, partial, split_k, blocks);
}

void gemv_splitk_partial_fp8e4b15_kblock_kscale(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor partial,
    int64_t split_k,
    int64_t variant) {
  check_inputs(x, weight_fp8, scales, partial, split_k);
  dispatch_partial<true>(x, weight_fp8, scales, partial, split_k, variant);
}

void splitk_reduce_add_rmsnorm_2048(
    torch::Tensor partial,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor sum_out,
    torch::Tensor norm_out,
    int64_t split_k,
    bool has_bias,
    double eps,
    int64_t threads) {
  check_reduce_inputs(partial, bias, residual, norm_weight, sum_out, norm_out, split_k);
  if (threads == 0) {
    TORCH_CHECK(split_k == 3, "triton-shape CUDA reduce currently supports split_k=3");
    if (has_bias) {
      launch_reduce_tritonshape<true>(partial, bias, residual, norm_weight, sum_out, norm_out, eps);
    } else {
      launch_reduce_tritonshape<false>(partial, bias, residual, norm_weight, sum_out, norm_out, eps);
    }
    return;
  }
  if (has_bias) {
    dispatch_reduce_split<true>(partial, bias, residual, norm_weight, sum_out, norm_out, split_k, eps, threads);
  } else {
    dispatch_reduce_split<false>(partial, bias, residual, norm_weight, sum_out, norm_out, split_k, eps, threads);
  }
}

void down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor sum_out,
    torch::Tensor norm_out,
    torch::Tensor partial,
    double eps) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CUDA(residual);
  CHECK_CUDA(norm_weight);
  CHECK_CUDA(sum_out);
  CHECK_CUDA(norm_out);
  CHECK_CUDA(partial);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(residual);
  CHECK_CONTIGUOUS(norm_weight);
  CHECK_CONTIGUOUS(sum_out);
  CHECK_CONTIGUOUS(norm_out);
  CHECK_CONTIGUOUS(partial);
  CHECK_HALF(x);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  CHECK_HALF(residual);
  CHECK_HALF(norm_weight);
  CHECK_HALF(sum_out);
  CHECK_HALF(norm_out);
  CHECK_FLOAT(partial);
  TORCH_CHECK(x.numel() == 6144, "down FP8 fused boundary expects x with 6144 elements");
  TORCH_CHECK(weight_fp8.numel() == 48 * kHiddenSize * kBlockK,
              "weight_fp8 must contain [48, 2048, 128]");
  TORCH_CHECK(scales.numel() == 48 * kHiddenSize, "scales must contain [48, 2048]");
  TORCH_CHECK(residual.numel() == kHiddenSize, "residual must have 2048 elements");
  TORCH_CHECK(norm_weight.numel() == kHiddenSize, "norm_weight must have 2048 elements");
  TORCH_CHECK(sum_out.numel() == kHiddenSize, "sum_out must have 2048 elements");
  TORCH_CHECK(norm_out.numel() == kHiddenSize, "norm_out must have 2048 elements");
  TORCH_CHECK(partial.sizes() == at::IntArrayRef({3, kHiddenSize}), "partial must be [3, 2048]");
  launch_down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary(
      x,
      weight_fp8,
      scales,
      residual,
      norm_weight,
      sum_out,
      norm_out,
      partial,
      eps);
}

void down_fp8e4b15_kblock_kscale_split3_sum_sumsq(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor residual,
    torch::Tensor sum_out,
    torch::Tensor tile_sumsq,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CUDA(residual);
  CHECK_CUDA(sum_out);
  CHECK_CUDA(tile_sumsq);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(residual);
  CHECK_CONTIGUOUS(sum_out);
  CHECK_CONTIGUOUS(tile_sumsq);
  CHECK_HALF(x);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  CHECK_HALF(residual);
  CHECK_HALF(sum_out);
  CHECK_FLOAT(tile_sumsq);
  TORCH_CHECK(x.numel() == 6144, "down FP8 sum+sumsq expects x with 6144 elements");
  TORCH_CHECK(weight_fp8.numel() == 48 * kHiddenSize * kBlockK,
              "weight_fp8 must contain [48, 2048, 128]");
  TORCH_CHECK(scales.numel() == 48 * kHiddenSize, "scales must contain [48, 2048]");
  TORCH_CHECK(residual.numel() == kHiddenSize, "residual must have 2048 elements");
  TORCH_CHECK(sum_out.numel() == kHiddenSize, "sum_out must have 2048 elements");
  if (rows_per_block == 2) {
    TORCH_CHECK(tile_sumsq.numel() == kHiddenSize / 2, "tile_sumsq must be [1024] for rows_per_block=2");
    launch_down_fp8e4b15_kblock_kscale_split3_sum_sumsq<2>(
        x, weight_fp8, scales, residual, sum_out, tile_sumsq);
  } else if (rows_per_block == 4) {
    TORCH_CHECK(tile_sumsq.numel() == kHiddenSize / 4, "tile_sumsq must be [512] for rows_per_block=4");
    launch_down_fp8e4b15_kblock_kscale_split3_sum_sumsq<4>(
        x, weight_fp8, scales, residual, sum_out, tile_sumsq);
  } else if (rows_per_block == 8) {
    TORCH_CHECK(tile_sumsq.numel() == kHiddenSize / 8, "tile_sumsq must be [256] for rows_per_block=8");
    launch_down_fp8e4b15_kblock_kscale_split3_sum_sumsq<8>(
        x, weight_fp8, scales, residual, sum_out, tile_sumsq);
  } else if (rows_per_block == 16) {
    TORCH_CHECK(tile_sumsq.numel() == kHiddenSize / 16, "tile_sumsq must be [128] for rows_per_block=16");
    launch_down_fp8e4b15_kblock_kscale_split3_sum_sumsq<16>(
        x, weight_fp8, scales, residual, sum_out, tile_sumsq);
  } else {
    TORCH_CHECK(false, "rows_per_block must be 2, 4, 8, or 16");
  }
}

void rmsnorm_from_sum_sumsq_2048(
    torch::Tensor sum_out,
    torch::Tensor tile_sumsq,
    torch::Tensor norm_weight,
    torch::Tensor norm_out,
    double eps) {
  CHECK_CUDA(sum_out);
  CHECK_CUDA(tile_sumsq);
  CHECK_CUDA(norm_weight);
  CHECK_CUDA(norm_out);
  CHECK_CONTIGUOUS(sum_out);
  CHECK_CONTIGUOUS(tile_sumsq);
  CHECK_CONTIGUOUS(norm_weight);
  CHECK_CONTIGUOUS(norm_out);
  CHECK_HALF(sum_out);
  CHECK_FLOAT(tile_sumsq);
  CHECK_HALF(norm_weight);
  CHECK_HALF(norm_out);
  TORCH_CHECK(sum_out.numel() == kHiddenSize, "sum_out must have 2048 elements");
  TORCH_CHECK(norm_weight.numel() == kHiddenSize, "norm_weight must have 2048 elements");
  TORCH_CHECK(norm_out.numel() == kHiddenSize, "norm_out must have 2048 elements");
  const int64_t tile_count = tile_sumsq.numel();
  if (tile_count == 1024) {
    launch_rmsnorm_from_sum_sumsq_2048<1024>(sum_out, tile_sumsq, norm_weight, norm_out, eps);
  } else if (tile_count == 512) {
    launch_rmsnorm_from_sum_sumsq_2048<512>(sum_out, tile_sumsq, norm_weight, norm_out, eps);
  } else if (tile_count == 256) {
    launch_rmsnorm_from_sum_sumsq_2048<256>(sum_out, tile_sumsq, norm_weight, norm_out, eps);
  } else if (tile_count == 128) {
    launch_rmsnorm_from_sum_sumsq_2048<128>(sum_out, tile_sumsq, norm_weight, norm_out, eps);
  } else {
    TORCH_CHECK(false, "tile_sumsq must have 128, 256, 512, or 1024 elements");
  }
}

void prefix_stage2_o_partial_fp8e4b15_kblock_major(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor o_partial,
    int64_t active_splits,
    int64_t split_k,
    int64_t coop_blocks) {
  check_prefix_stage2_o_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      nullptr,
      active_splits,
      split_k);
  dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      active_splits,
      coop_blocks);
}

void prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor o_partial,
    torch::Tensor barrier_counter,
    int64_t active_splits,
    int64_t split_k,
    int64_t blocks_hint) {
  check_prefix_stage2_o_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      &barrier_counter,
      active_splits,
      split_k);
  dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      barrier_counter,
      active_splits,
      blocks_hint);
}

void prefix_stage2_o_partial_fp8e4b15_kblock_major_local(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor o_partial,
    int64_t active_splits,
    int64_t split_k) {
  check_prefix_stage2_o_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      nullptr,
      active_splits,
      split_k);
  dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local(
      partial_m,
      partial_l,
      partial_acc,
      weight_fp8,
      scales,
      o_partial,
      active_splits);
}

void prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor o_partial,
    int64_t active_splits,
    int64_t split_k,
    int64_t row_blocks_per_cta) {
  check_prefix_stage2_o_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      nullptr,
      active_splits,
      split_k);
  dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup(
      partial_m,
      partial_l,
      partial_acc,
      weight_fp8,
      scales,
      o_partial,
      active_splits,
      row_blocks_per_cta);
}

void prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor o_partial,
    int64_t active_splits,
    int64_t split_k,
    int64_t row_blocks_per_cta) {
  check_prefix_stage2_o_int4_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_int4,
      scales,
      o_partial,
      active_splits,
      split_k);
  if (row_blocks_per_cta == -22) {
    dispatch_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup_lane8_pre4<512, 64>(
        partial_m,
        partial_l,
        partial_acc,
        weight_int4,
        scales,
        o_partial,
        active_splits);
    return;
  }
  TORCH_CHECK(false, "INT4 stage2/O fused experiment currently supports row_blocks_per_cta=-22 only");
}

void prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor o_partial,
    int64_t active_splits,
    int64_t split_k,
    int64_t coop_blocks,
    int64_t row_blocks_per_cta) {
  check_prefix_stage2_o_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      nullptr,
      active_splits,
      split_k);
  dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      active_splits,
      coop_blocks,
      row_blocks_per_cta);
}

void prefix_stage2_o_partial_fp8e4b15_kblock_major_pc(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor o_partial,
    torch::Tensor ready_flags,
    int64_t active_splits,
    int64_t split_k,
    int64_t coop_blocks) {
  check_prefix_stage2_o_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      nullptr,
      active_splits,
      split_k);
  CHECK_CUDA(ready_flags);
  CHECK_CONTIGUOUS(ready_flags);
  TORCH_CHECK(ready_flags.scalar_type() == at::kInt, "ready_flags must be int32");
  TORCH_CHECK(ready_flags.numel() == kHiddenSize / kBlockK, "ready_flags must have 16 elements");
  dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      ready_flags,
      active_splits,
      coop_blocks);
}

void prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor scratch_x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor o_partial,
    torch::Tensor ready_flags,
    int64_t active_splits,
    int64_t split_k) {
  check_prefix_stage2_o_inputs(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      nullptr,
      active_splits,
      split_k);
  CHECK_CUDA(ready_flags);
  CHECK_CONTIGUOUS(ready_flags);
  TORCH_CHECK(ready_flags.scalar_type() == at::kInt, "ready_flags must be int32");
  TORCH_CHECK(ready_flags.numel() == kHiddenSize / kBlockK, "ready_flags must have 16 elements");
  dispatch_prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full(
      partial_m,
      partial_l,
      partial_acc,
      scratch_x,
      weight_fp8,
      scales,
      o_partial,
      ready_flags,
      active_splits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gemv_splitk_partial_fp8e4b15_kblock_major",
      &gemv_splitk_partial_fp8e4b15_kblock_major,
      "Triton-shape CUDA FP8 e4b15 kblock-major split-K partial GEMV");
  m.def(
      "gemv_splitk_partial_fp8e4b15_kblock_major_taskloop",
      &gemv_splitk_partial_fp8e4b15_kblock_major_taskloop,
      "Diagnostic task-loop CUDA FP8 e4b15 kblock-major split-K partial GEMV");
  m.def(
      "gemv_splitk_partial_fp8e4b15_kblock_kscale",
      &gemv_splitk_partial_fp8e4b15_kblock_kscale,
      "Triton-shape CUDA FP8 e4b15 kblock/kscale split-K partial GEMV");
  m.def(
      "splitk_reduce_add_rmsnorm_2048",
      &splitk_reduce_add_rmsnorm_2048,
      "CUDA split-K reduce + residual add + RMSNorm specialized for 2048-wide decode vectors");
  m.def(
      "down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary",
      &down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary,
      "Experimental cooperative down FP8 split3 GEMV + residual add + RMSNorm boundary");
  m.def(
      "down_fp8e4b15_kblock_kscale_split3_sum_sumsq",
      &down_fp8e4b15_kblock_kscale_split3_sum_sumsq,
      "Experimental down FP8 split3 GEMV fused with residual add and sumsq");
  m.def(
      "rmsnorm_from_sum_sumsq_2048",
      &rmsnorm_from_sum_sumsq_2048,
      "CUDA RMSNorm from precomputed summed vector and tile sumsq");
  m.def(
      "prefix_stage2_o_partial_fp8e4b15_kblock_major",
      &prefix_stage2_o_partial_fp8e4b15_kblock_major,
      "Experimental cooperative stage2 merge + O FP8 partial GEMV boundary");
  m.def(
      "prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier",
      &prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier,
      "Experimental stage2 merge + O FP8 partial GEMV with in-kernel atomic barrier");
  m.def(
      "prefix_stage2_o_partial_fp8e4b15_kblock_major_local",
      &prefix_stage2_o_partial_fp8e4b15_kblock_major_local,
      "Experimental O-local stage2 merge + O FP8 partial GEMV");
  m.def(
      "prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup",
      &prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup,
      "Experimental grouped-row local stage2 merge + O FP8 partial GEMV");
  m.def(
      "prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup",
      &prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup,
      "Experimental grouped-row local stage2 merge + O symmetric INT4 partial GEMV");
  m.def(
      "prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup",
      &prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup,
      "Experimental cooperative stage2 merge + grouped-row O FP8 partial GEMV");
  m.def(
      "prefix_stage2_o_partial_fp8e4b15_kblock_major_pc",
      &prefix_stage2_o_partial_fp8e4b15_kblock_major_pc,
      "Experimental cooperative producer-consumer stage2 merge + O FP8 partial GEMV");
  m.def(
      "prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full",
      &prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full,
      "Experimental full-grid producer-consumer stage2 merge + O FP8 partial GEMV");
}
