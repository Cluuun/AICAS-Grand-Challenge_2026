#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define CHECK_BYTE(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")

constexpr int kHiddenSize = 2048;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kQSize = kQueryHeads * kHeadDim;
constexpr int kKSize = kKvHeads * kHeadDim;
constexpr int kQkvOutSize = kQSize + kKSize + kKSize;
constexpr int kVOffset = kQSize + kKSize;
constexpr int kQkHeads = kQueryHeads + kKvHeads;
constexpr int kFp8BlockK = 128;
constexpr int kFp8ScaleBlocks = kHiddenSize / kFp8BlockK;
constexpr int kFp8WordsPerBlock = kFp8BlockK / 4;
constexpr int kInt4BlockK = 128;
constexpr int kInt4ScaleBlocks = kHiddenSize / kInt4BlockK;
constexpr int kInt4PackedBytesPerBlock = kInt4BlockK / 2;
constexpr int kInt4PackedWordsPerBlock = kInt4PackedBytesPerBlock / 4;
constexpr int kInt4PackedU16PerBlock = kInt4PackedBytesPerBlock / 2;
constexpr int kHalf2PerVec = 4;
constexpr int kVecCount = (kHiddenSize / 2) / kHalf2PerVec;

__forceinline__ __device__ float warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ uint4 load_uint4_cg(const uint4* ptr) {
  return __ldcg(ptr);
}

__forceinline__ __device__ uint32_t load_u32_cg(const uint32_t* ptr) {
  return __ldcg(ptr);
}

__forceinline__ __device__ uint16_t load_u16_cg(const uint16_t* ptr) {
  return __ldg(ptr);
}

__forceinline__ __device__ float dot_acc_half2(float acc, half2 x, half2 w) {
  float2 xf = __half22float2(x);
  float2 wf = __half22float2(w);
  acc = fmaf(xf.x, wf.x, acc);
  acc = fmaf(xf.y, wf.y, acc);
  return acc;
}

__forceinline__ __device__ float dot4_half2_haccum(
    half2 x01,
    half2 w01,
    half2 x23,
    half2 w23) {
  half2 acc = __float2half2_rn(0.0f);
  acc = __hfma2(x01, w01, acc);
  acc = __hfma2(x23, w23, acc);
  float2 pair = __half22float2(acc);
  return pair.x + pair.y;
}

__forceinline__ __device__ float half_warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, 16);
  }
  return value;
}

__forceinline__ __device__ float quarter_warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, 8);
  }
  return value;
}

__forceinline__ __device__ float dot_acc_int4_pair(float acc, uint32_t byte_value, half2 x_pair) {
  const float2 xv = __half22float2(x_pair);
  const float q0 = static_cast<float>(static_cast<int>(byte_value & 0x0fu) - 8);
  const float q1 = static_cast<float>(static_cast<int>((byte_value >> 4) & 0x0fu) - 8);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  return acc;
}

__forceinline__ __device__ void dot_acc_u4_pair(
    uint32_t byte_value,
    half2 x_pair,
    float& acc,
    float& x_sum) {
  const float2 xv = __half22float2(x_pair);
  const float q0 = static_cast<float>(byte_value & 0x0fu);
  const float q1 = static_cast<float>((byte_value >> 4) & 0x0fu);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  x_sum += xv.x + xv.y;
}

__forceinline__ __device__ float dot4_int4xhalf(uint32_t packed, const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed >> 8) & 0xffu, x2[1]);
  return acc;
}

__forceinline__ __device__ void dot4_u4xhalf(
    uint32_t packed,
    const half2* __restrict__ x2,
    float& acc,
    float& x_sum) {
  dot_acc_u4_pair(packed & 0xffu, x2[0], acc, x_sum);
  dot_acc_u4_pair((packed >> 8) & 0xffu, x2[1], acc, x_sum);
}

__forceinline__ __device__ float dot8_int4xhalf(uint32_t packed, const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed >> 8) & 0xffu, x2[1]);
  acc = dot_acc_int4_pair(acc, (packed >> 16) & 0xffu, x2[2]);
  acc = dot_acc_int4_pair(acc, (packed >> 24) & 0xffu, x2[3]);
  return acc;
}

__forceinline__ __device__ float dot16_int4xhalf(
    uint32_t packed0,
    uint32_t packed1,
    const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed0 & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed0 >> 8) & 0xffu, x2[1]);
  acc = dot_acc_int4_pair(acc, (packed0 >> 16) & 0xffu, x2[2]);
  acc = dot_acc_int4_pair(acc, (packed0 >> 24) & 0xffu, x2[3]);
  acc = dot_acc_int4_pair(acc, packed1 & 0xffu, x2[4]);
  acc = dot_acc_int4_pair(acc, (packed1 >> 8) & 0xffu, x2[5]);
  acc = dot_acc_int4_pair(acc, (packed1 >> 16) & 0xffu, x2[6]);
  acc = dot_acc_int4_pair(acc, (packed1 >> 24) & 0xffu, x2[7]);
  return acc;
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

__global__ void qkv_gemv_half2_vec4_f32acc_cg_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int n_out,
    int rows_per_block,
    bool has_bias) {
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= n_out) {
    return;
  }

  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const uint4* w4 = reinterpret_cast<const uint4*>(
      weight + static_cast<int64_t>(row) * kHiddenSize);

  float acc = 0.0f;

#pragma unroll
  for (int vec = lane; vec < kVecCount; vec += 32) {
    uint4 xv = x4[vec];
    uint4 wv = load_uint4_cg(w4 + vec);

    acc = dot_acc_half2(acc, half2_from_u32(xv.x), half2_from_u32(wv.x));
    acc = dot_acc_half2(acc, half2_from_u32(xv.y), half2_from_u32(wv.y));
    acc = dot_acc_half2(acc, half2_from_u32(xv.z), half2_from_u32(wv.z));
    acc = dot_acc_half2(acc, half2_from_u32(xv.w), half2_from_u32(wv.w));
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if (has_bias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

__global__ void qkv_gemv_fp8e4b15_block128_f32acc_cg_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int n_out,
    int rows_per_block,
    bool has_bias) {
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= n_out) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* w4 = reinterpret_cast<const uint32_t*>(
      weight_fp8 + static_cast<int64_t>(row) * kHiddenSize);
  const half* row_scales = scales + static_cast<int64_t>(row) * kFp8ScaleBlocks;

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    float scale = __half2float(row_scales[block_k]);
    float block_acc = 0.0f;
    int word = block_k * kFp8WordsPerBlock + lane;
    uint32_t packed = load_u32_cg(w4 + word);
    uint32_t w01_bits;
    uint32_t w23_bits;
    fp8e4b15x4_to_half2_bits(packed, w01_bits, w23_bits);

    int x_pair = block_k * (kFp8BlockK / 2) + lane * 2;
    block_acc = dot_acc_half2(block_acc, x2[x_pair], half2_from_u32(w01_bits));
    block_acc = dot_acc_half2(block_acc, x2[x_pair + 1], half2_from_u32(w23_bits));
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if (has_bias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <bool kHasBias>
__global__ void qkv_gemv_fp8e4b15_block128_f32acc_cg_static_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* w4 = reinterpret_cast<const uint32_t*>(
      weight_fp8 + static_cast<int64_t>(row) * kHiddenSize);
  const half* row_scales = scales + static_cast<int64_t>(row) * kFp8ScaleBlocks;

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const float scale = __half2float(row_scales[block_k]);
    float block_acc = 0.0f;
    const int word = block_k * kFp8WordsPerBlock + lane;
    const uint32_t packed = load_u32_cg(w4 + word);
    uint32_t w01_bits;
    uint32_t w23_bits;
    fp8e4b15x4_to_half2_bits(packed, w01_bits, w23_bits);

    const int x_pair = block_k * (kFp8BlockK / 2) + lane * 2;
    block_acc = dot_acc_half2(block_acc, x2[x_pair], half2_from_u32(w01_bits));
    block_acc = dot_acc_half2(block_acc, x2[x_pair + 1], half2_from_u32(w23_bits));
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if constexpr (kHasBias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <bool kHasBias>
__global__ void qkv_gemv_fp8e4b15_block128_hscale_cg_static_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* w4 = reinterpret_cast<const uint32_t*>(
      weight_fp8 + static_cast<int64_t>(row) * kHiddenSize);
  const half* row_scales = scales + static_cast<int64_t>(row) * kFp8ScaleBlocks;

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const float scale = __half2float(row_scales[block_k]);
    const int word = block_k * kFp8WordsPerBlock + lane;
    const uint32_t packed = load_u32_cg(w4 + word);
    uint32_t w01_bits;
    uint32_t w23_bits;
    fp8e4b15x4_to_half2_bits(packed, w01_bits, w23_bits);

    const int x_pair = block_k * (kFp8BlockK / 2) + lane * 2;
    const float block_acc = dot4_half2_haccum(
        x2[x_pair],
        half2_from_u32(w01_bits),
        x2[x_pair + 1],
        half2_from_u32(w23_bits));
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if constexpr (kHasBias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <int kRowsPerBlock>
__global__ void qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int n_out,
    bool has_bias) {
  const int row_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= n_out) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint16_t* weight_u16 = reinterpret_cast<const uint16_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kInt4ScaleBlocks; ++block_k) {
    const uint16_t* row_u16 = weight_u16
        + (static_cast<int64_t>(block_k) * n_out + row) * kInt4PackedU16PerBlock;
    const int x_pair = block_k * (kInt4BlockK / 2) + lane * 2;
    const float block_acc = dot4_int4xhalf(static_cast<uint32_t>(load_u16_cg(row_u16 + lane)), x2 + x_pair);
    const float scale = __half2float(scales[static_cast<int64_t>(block_k) * n_out + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if (has_bias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <int kRowsPerBlock, bool kHasBias>
__global__ void qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_static_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out) {
  const int row_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint16_t* weight_u16 = reinterpret_cast<const uint16_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kInt4ScaleBlocks; ++block_k) {
    const uint16_t* row_u16 = weight_u16
        + (static_cast<int64_t>(block_k) * kQkvOutSize + row) * kInt4PackedU16PerBlock;
    const int x_pair = block_k * (kInt4BlockK / 2) + lane * 2;
    const float block_acc = dot4_int4xhalf(static_cast<uint32_t>(load_u16_cg(row_u16 + lane)), x2 + x_pair);
    const float scale = __half2float(scales[block_k * kQkvOutSize + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if constexpr (kHasBias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <int kRowsPerBlock, bool kHasBias>
__global__ void qkv_gemv_int4_sym_kblock_kscale_fullwarp2_u16_static_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out) {
  const int row_pair_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * kRowsPerBlock + row_pair_in_block * 2;
  const int row1 = row0 + 1;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint16_t* weight_u16 = reinterpret_cast<const uint16_t*>(weight_int4);

  float acc0 = 0.0f;
  float acc1 = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kInt4ScaleBlocks; ++block_k) {
    const int x_pair = block_k * (kInt4BlockK / 2) + lane * 2;
    const half2 xh0 = x2[x_pair];
    const half2 xh1 = x2[x_pair + 1];
    const uint16_t* block_u16 = weight_u16 + static_cast<int64_t>(block_k) * kQkvOutSize * kInt4PackedU16PerBlock;
    const uint16_t* row0_u16 = block_u16 + row0 * kInt4PackedU16PerBlock;
    const uint16_t* row1_u16 = block_u16 + row1 * kInt4PackedU16PerBlock;
    const uint32_t p0 = static_cast<uint32_t>(load_u16_cg(row0_u16 + lane));
    const uint32_t p1 = static_cast<uint32_t>(load_u16_cg(row1_u16 + lane));

    float block_acc0 = 0.0f;
    float block_acc1 = 0.0f;
    block_acc0 = dot_acc_int4_pair(block_acc0, p0 & 0xffu, xh0);
    block_acc0 = dot_acc_int4_pair(block_acc0, p0 >> 8, xh1);
    block_acc1 = dot_acc_int4_pair(block_acc1, p1 & 0xffu, xh0);
    block_acc1 = dot_acc_int4_pair(block_acc1, p1 >> 8, xh1);
    acc0 = fmaf(block_acc0, __half2float(scales[block_k * kQkvOutSize + row0]), acc0);
    acc1 = fmaf(block_acc1, __half2float(scales[block_k * kQkvOutSize + row1]), acc1);
  }

  acc0 = warp_sum(acc0);
  acc1 = warp_sum(acc1);
  if (lane == 0) {
    if constexpr (kHasBias) {
      acc0 += __half2float(bias[row0]);
      acc1 += __half2float(bias[row1]);
    }
    out[row0] = __float2half_rn(acc0);
    out[row1] = __float2half_rn(acc1);
  }
}

template <int kRowsPerBlock, bool kHasBias>
__global__ void qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_static_u4offset_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out) {
  const int row_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint16_t* weight_u16 = reinterpret_cast<const uint16_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kInt4ScaleBlocks; ++block_k) {
    const uint16_t* row_u16 = weight_u16
        + (static_cast<int64_t>(block_k) * kQkvOutSize + row) * kInt4PackedU16PerBlock;
    const int x_pair = block_k * (kInt4BlockK / 2) + lane * 2;
    float block_acc = 0.0f;
    float x_sum = 0.0f;
    dot4_u4xhalf(static_cast<uint32_t>(load_u16_cg(row_u16 + lane)), x2 + x_pair, block_acc, x_sum);
    const float scale = __half2float(scales[block_k * kQkvOutSize + row]);
    acc = fmaf(block_acc - 8.0f * x_sum, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if constexpr (kHasBias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <int kRowsPerBlock, bool kHasBias>
__global__ void qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_static_sharedx_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out) {
  const int row_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  const half2* x2_global = reinterpret_cast<const half2*>(x);
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2 = reinterpret_cast<half2*>(shared_bytes);
  for (int idx = threadIdx.x; idx < kHiddenSize / 2; idx += blockDim.x) {
    x2[idx] = x2_global[idx];
  }
  __syncthreads();

  const uint16_t* weight_u16 = reinterpret_cast<const uint16_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kInt4ScaleBlocks; ++block_k) {
    const uint16_t* row_u16 = weight_u16
        + (static_cast<int64_t>(block_k) * kQkvOutSize + row) * kInt4PackedU16PerBlock;
    const int x_pair = block_k * (kInt4BlockK / 2) + lane * 2;
    const float block_acc = dot4_int4xhalf(static_cast<uint32_t>(load_u16_cg(row_u16 + lane)), x2 + x_pair);
    const float scale = __half2float(scales[block_k * kQkvOutSize + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    if constexpr (kHasBias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <int kRowsPerBlock, bool kBroadcastScale>
__global__ void qkv_gemv_int4_sym_kblock_kscale_halfwarp_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int n_out,
    bool has_bias) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= n_out) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kInt4ScaleBlocks; ++block_k) {
    const uint32_t* row_w32 = weight_w32
        + (static_cast<int64_t>(block_k) * n_out + row) * kInt4PackedWordsPerBlock;
    const int x_pair = block_k * (kInt4BlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(load_u32_cg(row_w32 + lane16), x2 + x_pair);

    float scale;
    if constexpr (kBroadcastScale) {
      scale = 0.0f;
      if (lane16 == 0) {
        scale = __half2float(scales[static_cast<int64_t>(block_k) * n_out + row]);
      }
      scale = __shfl_sync(0xffffffffu, scale, 0, 16);
    } else {
      scale = __half2float(scales[static_cast<int64_t>(block_k) * n_out + row]);
    }
    acc = fmaf(block_acc, scale, acc);
  }

  acc = half_warp_sum(acc);
  if (lane16 == 0) {
    if (has_bias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

template <int kRowsPerBlock, bool kBroadcastScale>
__global__ void qkv_gemv_int4_sym_kblock_kscale_quarterwarp_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int n_out,
    bool has_bias) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group_id = lane >> 3;
  const int lane8 = lane & 7;
  const int row_in_block = warp_id * 4 + group_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= n_out) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kInt4ScaleBlocks; ++block_k) {
    const uint32_t* row_w32 = weight_w32
        + (static_cast<int64_t>(block_k) * n_out + row) * kInt4PackedWordsPerBlock;
    const int word_offset = lane8 * 2;
    const int x_pair = block_k * (kInt4BlockK / 2) + lane8 * 8;
    const float block_acc = dot16_int4xhalf(
        load_u32_cg(row_w32 + word_offset),
        load_u32_cg(row_w32 + word_offset + 1),
        x2 + x_pair);

    float scale;
    if constexpr (kBroadcastScale) {
      scale = 0.0f;
      if (lane8 == 0) {
        scale = __half2float(scales[static_cast<int64_t>(block_k) * n_out + row]);
      }
      scale = __shfl_sync(0xffffffffu, scale, 0, 8);
    } else {
      scale = __half2float(scales[static_cast<int64_t>(block_k) * n_out + row]);
    }
    acc = fmaf(block_acc, scale, acc);
  }

  acc = quarter_warp_sum(acc);
  if (lane8 == 0) {
    if (has_bias) {
      acc += __half2float(bias[row]);
    }
    out[row] = __float2half_rn(acc);
  }
}

__global__ void qkv_qk_rmsnorm_rope_kv_update_gqa2_kernel(
    const half* __restrict__ qkv,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    int64_t stride_cos_b,
    int64_t stride_cos_d,
    int64_t stride_qo_b,
    int64_t stride_qo_h,
    int64_t stride_qo_d,
    int64_t stride_kc_b,
    int64_t stride_kc_h,
    int64_t stride_kc_t,
    int64_t stride_kc_d,
    int64_t stride_vc_b,
    int64_t stride_vc_h,
    int64_t stride_vc_t,
    int64_t stride_vc_d,
    float q_eps,
    float k_eps) {
  int batch = blockIdx.y;
  int kv_head = blockIdx.x;
  int q_head0 = kv_head * 2;
  int q_head1 = q_head0 + 1;
  int tid = threadIdx.x;

  __shared__ float q0_vals[kHeadDim];
  __shared__ float q1_vals[kHeadDim];
  __shared__ float k_vals[kHeadDim];
  __shared__ float red_q0[256];
  __shared__ float red_q1[256];
  __shared__ float red_k[256];
  __shared__ float rstd_q0;
  __shared__ float rstd_q1;
  __shared__ float rstd_k;

  float sum_q0 = 0.0f;
  float sum_q1 = 0.0f;
  float sum_k = 0.0f;

  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    float q0 = __half2float(qkv[q_head0 * kHeadDim + d]);
    float q1 = __half2float(qkv[q_head1 * kHeadDim + d]);
    float k = __half2float(qkv[kQSize + kv_head * kHeadDim + d]);
    q0_vals[d] = q0;
    q1_vals[d] = q1;
    k_vals[d] = k;
    sum_q0 = fmaf(q0, q0, sum_q0);
    sum_q1 = fmaf(q1, q1, sum_q1);
    sum_k = fmaf(k, k, sum_k);
  }

  red_q0[tid] = sum_q0;
  red_q1[tid] = sum_q1;
  red_k[tid] = sum_k;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q0[tid] += red_q0[tid + stride];
      red_q1[tid] += red_q1[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rstd_q0 = rsqrtf(red_q0[0] * (1.0f / static_cast<float>(kHeadDim)) + q_eps);
    rstd_q1 = rsqrtf(red_q1[0] * (1.0f / static_cast<float>(kHeadDim)) + q_eps);
    rstd_k = rsqrtf(red_k[0] * (1.0f / static_cast<float>(kHeadDim)) + k_eps);
  }
  __syncthreads();

  int64_t pos = cache_position[0];
  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    int rot_d = d < (kHeadDim / 2) ? d + (kHeadDim / 2) : d - (kHeadDim / 2);
    float rot_sign = d < (kHeadDim / 2) ? -1.0f : 1.0f;
    float cos_v = __half2float(cos[batch * stride_cos_b + d * stride_cos_d]);
    float sin_v = __half2float(sin[batch * stride_cos_b + d * stride_cos_d]);

    float qw = __half2float(q_weight[d]);
    float qw_rot = __half2float(q_weight[rot_d]);
    float kw = __half2float(k_weight[d]);
    float kw_rot = __half2float(k_weight[rot_d]);

    float q0_out_v = q0_vals[d] * rstd_q0 * qw * cos_v
        + q0_vals[rot_d] * rstd_q0 * qw_rot * rot_sign * sin_v;
    float q1_out_v = q1_vals[d] * rstd_q1 * qw * cos_v
        + q1_vals[rot_d] * rstd_q1 * qw_rot * rot_sign * sin_v;
    float k_out_v = k_vals[d] * rstd_k * kw * cos_v
        + k_vals[rot_d] * rstd_k * kw_rot * rot_sign * sin_v;

    q_out[batch * stride_qo_b + q_head0 * stride_qo_h + d * stride_qo_d] =
        __float2half_rn(q0_out_v);
    q_out[batch * stride_qo_b + q_head1 * stride_qo_h + d * stride_qo_d] =
        __float2half_rn(q1_out_v);
    key_cache[
        batch * stride_kc_b + kv_head * stride_kc_h + pos * stride_kc_t + d * stride_kc_d] =
        __float2half_rn(k_out_v);
    value_cache[
        batch * stride_vc_b + kv_head * stride_vc_h + pos * stride_vc_t + d * stride_vc_d] =
        qkv[kVOffset + kv_head * kHeadDim + d];
  }
}

__global__ void qkv_qk_rmsnorm_rope_kv_update_gqa2_warp_kernel(
    const half* __restrict__ qkv,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    int64_t stride_cos_b,
    int64_t stride_cos_d,
    int64_t stride_qo_b,
    int64_t stride_qo_h,
    int64_t stride_qo_d,
    int64_t stride_kc_b,
    int64_t stride_kc_h,
    int64_t stride_kc_t,
    int64_t stride_kc_d,
    int64_t stride_vc_b,
    int64_t stride_vc_h,
    int64_t stride_vc_t,
    int64_t stride_vc_d,
    float q_eps,
    float k_eps) {
  int batch = blockIdx.y;
  int kv_head = blockIdx.x;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int64_t pos = cache_position[0];

  if (warp == 3) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = lane + i * 32;
      value_cache[
          batch * stride_vc_b + kv_head * stride_vc_h + pos * stride_vc_t + d * stride_vc_d] =
          qkv[kVOffset + kv_head * kHeadDim + d];
    }
    return;
  }
  if (warp > 3) {
    return;
  }

  int q_head = kv_head * 2 + warp;
  bool is_k = warp == 2;
  const half* in = is_k ? qkv + kQSize + kv_head * kHeadDim : qkv + q_head * kHeadDim;
  const half* norm_w = is_k ? k_weight : q_weight;
  float eps = is_k ? k_eps : q_eps;

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int d = lane + i * 32;
    float x = __half2float(in[d]);
    sum = fmaf(x, x, sum);
  }
  float total = warp_sum(sum);
  total = __shfl_sync(0xffffffffu, total, 0);
  float rstd = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + eps);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int d = lane + i * 32;
    int rot_d = d < (kHeadDim / 2) ? d + (kHeadDim / 2) : d - (kHeadDim / 2);
    float rot_sign = d < (kHeadDim / 2) ? -1.0f : 1.0f;
    float x = __half2float(in[d]);
    float x_rot = __half2float(in[rot_d]);
    float w = __half2float(norm_w[d]);
    float w_rot = __half2float(norm_w[rot_d]);
    float cos_v = __half2float(cos[batch * stride_cos_b + d * stride_cos_d]);
    float sin_v = __half2float(sin[batch * stride_cos_b + d * stride_cos_d]);
    float out = x * rstd * w * cos_v + x_rot * rstd * w_rot * rot_sign * sin_v;
    if (is_k) {
      key_cache[
          batch * stride_kc_b + kv_head * stride_kc_h + pos * stride_kc_t + d * stride_kc_d] =
          __float2half_rn(out);
    } else {
      q_out[batch * stride_qo_b + q_head * stride_qo_h + d * stride_qo_d] =
          __float2half_rn(out);
    }
  }
}

template <bool kContigDim>
__global__ void qkv_qk_rmsnorm_rope_kv_update_gqa2_warp_pair_vvec_kernel(
    const half* __restrict__ qkv,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    int64_t stride_cos_b,
    int64_t stride_cos_d,
    int64_t stride_qo_b,
    int64_t stride_qo_h,
    int64_t stride_qo_d,
    int64_t stride_kc_b,
    int64_t stride_kc_h,
    int64_t stride_kc_t,
    int64_t stride_kc_d,
    int64_t stride_vc_b,
    int64_t stride_vc_h,
    int64_t stride_vc_t,
    int64_t stride_vc_d,
    float q_eps,
    float k_eps) {
  int batch = blockIdx.y;
  int kv_head = blockIdx.x;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int64_t pos = cache_position[0];

  if (warp == 3) {
    if (lane < 16) {
      const uint4* src = reinterpret_cast<const uint4*>(
          qkv + kVOffset + kv_head * kHeadDim);
      uint4* dst = reinterpret_cast<uint4*>(
          value_cache + batch * stride_vc_b + kv_head * stride_vc_h + pos * stride_vc_t);
      dst[lane] = src[lane];
    }
    return;
  }
  if (warp > 3) {
    return;
  }

  int q_head = kv_head * 2 + warp;
  bool is_k = warp == 2;
  const half* in = is_k ? qkv + kQSize + kv_head * kHeadDim : qkv + q_head * kHeadDim;
  const half* norm_w = is_k ? k_weight : q_weight;
  float eps = is_k ? k_eps : q_eps;

  int d0 = lane;
  int d1 = lane + 32;
  int d2 = lane + 64;
  int d3 = lane + 96;
  float x0 = __half2float(in[d0]);
  float x1 = __half2float(in[d1]);
  float x2 = __half2float(in[d2]);
  float x3 = __half2float(in[d3]);

  float sum = 0.0f;
  sum = fmaf(x0, x0, sum);
  sum = fmaf(x1, x1, sum);
  sum = fmaf(x2, x2, sum);
  sum = fmaf(x3, x3, sum);
  float total = warp_sum(sum);
  total = __shfl_sync(0xffffffffu, total, 0);
  float rstd = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + eps);

  float w0 = __half2float(norm_w[d0]);
  float w1 = __half2float(norm_w[d1]);
  float w2 = __half2float(norm_w[d2]);
  float w3 = __half2float(norm_w[d3]);
  float n0 = x0 * rstd * w0;
  float n1 = x1 * rstd * w1;
  float n2 = x2 * rstd * w2;
  float n3 = x3 * rstd * w3;

  const int64_t cos_base = batch * stride_cos_b;
  const int64_t cos_d0 = kContigDim ? d0 : d0 * stride_cos_d;
  const int64_t cos_d1 = kContigDim ? d1 : d1 * stride_cos_d;
  const int64_t cos_d2 = kContigDim ? d2 : d2 * stride_cos_d;
  const int64_t cos_d3 = kContigDim ? d3 : d3 * stride_cos_d;
  float cos0 = __half2float(cos[cos_base + cos_d0]);
  float sin0 = __half2float(sin[cos_base + cos_d0]);
  float cos1 = __half2float(cos[cos_base + cos_d1]);
  float sin1 = __half2float(sin[cos_base + cos_d1]);
  float cos2 = __half2float(cos[cos_base + cos_d2]);
  float sin2 = __half2float(sin[cos_base + cos_d2]);
  float cos3 = __half2float(cos[cos_base + cos_d3]);
  float sin3 = __half2float(sin[cos_base + cos_d3]);

  float out0 = n0 * cos0 - n2 * sin0;
  float out1 = n1 * cos1 - n3 * sin1;
  float out2 = n2 * cos2 + n0 * sin2;
  float out3 = n3 * cos3 + n1 * sin3;

  if (is_k) {
    half* out = key_cache + batch * stride_kc_b + kv_head * stride_kc_h + pos * stride_kc_t;
    out[kContigDim ? d0 : d0 * stride_kc_d] = __float2half_rn(out0);
    out[kContigDim ? d1 : d1 * stride_kc_d] = __float2half_rn(out1);
    out[kContigDim ? d2 : d2 * stride_kc_d] = __float2half_rn(out2);
    out[kContigDim ? d3 : d3 * stride_kc_d] = __float2half_rn(out3);
  } else {
    half* out = q_out + batch * stride_qo_b + q_head * stride_qo_h;
    out[kContigDim ? d0 : d0 * stride_qo_d] = __float2half_rn(out0);
    out[kContigDim ? d1 : d1 * stride_qo_d] = __float2half_rn(out1);
    out[kContigDim ? d2 : d2 * stride_qo_d] = __float2half_rn(out2);
    out[kContigDim ? d3 : d3 * stride_qo_d] = __float2half_rn(out3);
  }
}

template <int kHeadsPerBlock>
__global__ void qkv_qk_rmsnorm_rope_kv_update_gqa2_kvgroup_static_kernel(
    const half* __restrict__ qkv,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    int64_t cache_head_stride,
    float q_eps,
    float k_eps) {
  int local_warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int kv_head = blockIdx.x * kHeadsPerBlock + (local_warp >> 2);
  int warp = local_warp & 3;
  int64_t pos = cache_position[0];

  if (warp == 3) {
    if (lane < 16) {
      const uint4* src = reinterpret_cast<const uint4*>(
          qkv + kVOffset + kv_head * kHeadDim);
      uint4* dst = reinterpret_cast<uint4*>(
          value_cache + static_cast<int64_t>(kv_head) * cache_head_stride + pos * kHeadDim);
      dst[lane] = src[lane];
    }
    return;
  }

  int q_head = kv_head * 2 + warp;
  bool is_k = warp == 2;
  const half* in = is_k ? qkv + kQSize + kv_head * kHeadDim : qkv + q_head * kHeadDim;
  const half* norm_w = is_k ? k_weight : q_weight;
  float eps = is_k ? k_eps : q_eps;

  int d0 = lane;
  int d1 = lane + 32;
  int d2 = lane + 64;
  int d3 = lane + 96;
  float x0 = __half2float(in[d0]);
  float x1 = __half2float(in[d1]);
  float x2 = __half2float(in[d2]);
  float x3 = __half2float(in[d3]);

  float sum = 0.0f;
  sum = fmaf(x0, x0, sum);
  sum = fmaf(x1, x1, sum);
  sum = fmaf(x2, x2, sum);
  sum = fmaf(x3, x3, sum);
  float total = warp_sum(sum);
  total = __shfl_sync(0xffffffffu, total, 0);
  float rstd = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + eps);

  float w0 = __half2float(norm_w[d0]);
  float w1 = __half2float(norm_w[d1]);
  float w2 = __half2float(norm_w[d2]);
  float w3 = __half2float(norm_w[d3]);
  float n0 = x0 * rstd * w0;
  float n1 = x1 * rstd * w1;
  float n2 = x2 * rstd * w2;
  float n3 = x3 * rstd * w3;

  float cos0 = __half2float(cos[d0]);
  float sin0 = __half2float(sin[d0]);
  float cos1 = __half2float(cos[d1]);
  float sin1 = __half2float(sin[d1]);
  float cos2 = __half2float(cos[d2]);
  float sin2 = __half2float(sin[d2]);
  float cos3 = __half2float(cos[d3]);
  float sin3 = __half2float(sin[d3]);

  float out0 = n0 * cos0 - n2 * sin0;
  float out1 = n1 * cos1 - n3 * sin1;
  float out2 = n2 * cos2 + n0 * sin2;
  float out3 = n3 * cos3 + n1 * sin3;

  if (is_k) {
    half* out = key_cache + static_cast<int64_t>(kv_head) * cache_head_stride + pos * kHeadDim;
    out[d0] = __float2half_rn(out0);
    out[d1] = __float2half_rn(out1);
    out[d2] = __float2half_rn(out2);
    out[d3] = __float2half_rn(out3);
  } else {
    half* out = q_out + q_head * kHeadDim;
    out[d0] = __float2half_rn(out0);
    out[d1] = __float2half_rn(out1);
    out[d2] = __float2half_rn(out2);
    out[d3] = __float2half_rn(out3);
  }
}

template <int kHeadsPerBlock>
__global__ void qkv_qk_rmsnorm_rope_cache_kv_update_gqa2_kvgroup_static_kernel(
    const half* __restrict__ qkv,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ rope_cache,
    const int64_t* __restrict__ cache_position,
    const int64_t* __restrict__ rope_position,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    int64_t cache_head_stride,
    float q_eps,
    float k_eps) {
  int local_warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int kv_head = blockIdx.x * kHeadsPerBlock + (local_warp >> 2);
  int warp = local_warp & 3;
  int64_t pos = cache_position[0];
  int64_t rope_pos = rope_position[0];

  if (warp == 3) {
    if (lane < 16) {
      const uint4* src = reinterpret_cast<const uint4*>(
          qkv + kVOffset + kv_head * kHeadDim);
      uint4* dst = reinterpret_cast<uint4*>(
          value_cache + static_cast<int64_t>(kv_head) * cache_head_stride + pos * kHeadDim);
      dst[lane] = src[lane];
    }
    return;
  }

  int q_head = kv_head * 2 + warp;
  bool is_k = warp == 2;
  const half* in = is_k ? qkv + kQSize + kv_head * kHeadDim : qkv + q_head * kHeadDim;
  const half* norm_w = is_k ? k_weight : q_weight;
  float eps = is_k ? k_eps : q_eps;

  int d0 = lane;
  int d1 = lane + 32;
  int d2 = lane + 64;
  int d3 = lane + 96;
  float x0 = __half2float(in[d0]);
  float x1 = __half2float(in[d1]);
  float x2 = __half2float(in[d2]);
  float x3 = __half2float(in[d3]);

  float sum = 0.0f;
  sum = fmaf(x0, x0, sum);
  sum = fmaf(x1, x1, sum);
  sum = fmaf(x2, x2, sum);
  sum = fmaf(x3, x3, sum);
  float total = warp_sum(sum);
  total = __shfl_sync(0xffffffffu, total, 0);
  float rstd = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + eps);

  float w0 = __half2float(norm_w[d0]);
  float w1 = __half2float(norm_w[d1]);
  float w2 = __half2float(norm_w[d2]);
  float w3 = __half2float(norm_w[d3]);
  float n0 = x0 * rstd * w0;
  float n1 = x1 * rstd * w1;
  float n2 = x2 * rstd * w2;
  float n3 = x3 * rstd * w3;

  const half* cos = rope_cache + rope_pos * (2 * kHeadDim);
  const half* sin = cos + kHeadDim;
  float cos0 = __half2float(cos[d0]);
  float sin0 = __half2float(sin[d0]);
  float cos1 = __half2float(cos[d1]);
  float sin1 = __half2float(sin[d1]);
  float cos2 = __half2float(cos[d2]);
  float sin2 = __half2float(sin[d2]);
  float cos3 = __half2float(cos[d3]);
  float sin3 = __half2float(sin[d3]);

  float out0 = n0 * cos0 - n2 * sin0;
  float out1 = n1 * cos1 - n3 * sin1;
  float out2 = n2 * cos2 + n0 * sin2;
  float out3 = n3 * cos3 + n1 * sin3;

  if (is_k) {
    half* out = key_cache + static_cast<int64_t>(kv_head) * cache_head_stride + pos * kHeadDim;
    out[d0] = __float2half_rn(out0);
    out[d1] = __float2half_rn(out1);
    out[d2] = __float2half_rn(out2);
    out[d3] = __float2half_rn(out3);
  } else {
    half* out = q_out + q_head * kHeadDim;
    out[d0] = __float2half_rn(out0);
    out[d1] = __float2half_rn(out1);
    out[d2] = __float2half_rn(out2);
    out[d3] = __float2half_rn(out3);
  }
}

__device__ __forceinline__ void finalize_qk_head_from_scratch(
    int head_id,
    int lane,
    const half* __restrict__ qk_scratch,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    int64_t cache_head_stride,
    int64_t pos,
    float q_eps,
    float k_eps) {
  const bool is_k = head_id >= kQueryHeads;
  const int local_head = is_k ? head_id - kQueryHeads : head_id;
  const half* in = qk_scratch + head_id * kHeadDim;
  const half* norm_w = is_k ? k_weight : q_weight;
  const float eps = is_k ? k_eps : q_eps;

  const int d0 = lane;
  const int d1 = lane + 32;
  const int d2 = lane + 64;
  const int d3 = lane + 96;
  const float x0 = __half2float(in[d0]);
  const float x1 = __half2float(in[d1]);
  const float x2 = __half2float(in[d2]);
  const float x3 = __half2float(in[d3]);

  float sum = 0.0f;
  sum = fmaf(x0, x0, sum);
  sum = fmaf(x1, x1, sum);
  sum = fmaf(x2, x2, sum);
  sum = fmaf(x3, x3, sum);
  float total = warp_sum(sum);
  total = __shfl_sync(0xffffffffu, total, 0);
  const float rstd = rsqrtf(total * (1.0f / static_cast<float>(kHeadDim)) + eps);

  const float w0 = __half2float(norm_w[d0]);
  const float w1 = __half2float(norm_w[d1]);
  const float w2 = __half2float(norm_w[d2]);
  const float w3 = __half2float(norm_w[d3]);
  const float n0 = x0 * rstd * w0;
  const float n1 = x1 * rstd * w1;
  const float n2 = x2 * rstd * w2;
  const float n3 = x3 * rstd * w3;

  const float cos0 = __half2float(cos[d0]);
  const float sin0 = __half2float(sin[d0]);
  const float cos1 = __half2float(cos[d1]);
  const float sin1 = __half2float(sin[d1]);
  const float cos2 = __half2float(cos[d2]);
  const float sin2 = __half2float(sin[d2]);
  const float cos3 = __half2float(cos[d3]);
  const float sin3 = __half2float(sin[d3]);

  const float out0 = n0 * cos0 - n2 * sin0;
  const float out1 = n1 * cos1 - n3 * sin1;
  const float out2 = n2 * cos2 + n0 * sin2;
  const float out3 = n3 * cos3 + n1 * sin3;

  if (is_k) {
    half* out = key_cache + static_cast<int64_t>(local_head) * cache_head_stride + pos * kHeadDim;
    out[d0] = __float2half_rn(out0);
    out[d1] = __float2half_rn(out1);
    out[d2] = __float2half_rn(out2);
    out[d3] = __float2half_rn(out3);
  } else {
    half* out = q_out + local_head * kHeadDim;
    out[d0] = __float2half_rn(out0);
    out[d1] = __float2half_rn(out1);
    out[d2] = __float2half_rn(out2);
    out[d3] = __float2half_rn(out3);
  }
}

template <bool kHasBias>
__global__ void qkv_fp8e4b15_hscale_static_qk_update_gqa2_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    half* __restrict__ qk_scratch,
    int32_t* __restrict__ qk_counters,
    int64_t cache_head_stride,
    float q_eps,
    float k_eps) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* w4 = reinterpret_cast<const uint32_t*>(
      weight_fp8 + static_cast<int64_t>(row) * kHiddenSize);
  const half* row_scales = scales + static_cast<int64_t>(row) * kFp8ScaleBlocks;

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const float scale = __half2float(row_scales[block_k]);
    const int word = block_k * kFp8WordsPerBlock + lane;
    const uint32_t packed = load_u32_cg(w4 + word);
    uint32_t w01_bits;
    uint32_t w23_bits;
    fp8e4b15x4_to_half2_bits(packed, w01_bits, w23_bits);

    const int x_pair = block_k * (kFp8BlockK / 2) + lane * 2;
    const float block_acc = dot4_half2_haccum(
        x2[x_pair],
        half2_from_u32(w01_bits),
        x2[x_pair + 1],
        half2_from_u32(w23_bits));
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);

  int trigger = 0;
  if constexpr (kHasBias) {
    acc += __half2float(bias[row]);
  }
  const half out = __float2half_rn(acc);
  const int64_t pos = cache_position[0];

  if (row >= kVOffset) {
    if (lane == 0) {
      const int v_row = row - kVOffset;
      const int kv_head = v_row / kHeadDim;
      const int d = v_row - kv_head * kHeadDim;
      value_cache[static_cast<int64_t>(kv_head) * cache_head_stride + pos * kHeadDim + d] = out;
    }
    return;
  }

  const int head_id = row < kQSize ? row / kHeadDim : kQueryHeads + (row - kQSize) / kHeadDim;
  if (lane == 0) {
    const int d = row & (kHeadDim - 1);
    qk_scratch[head_id * kHeadDim + d] = out;
    __threadfence();
    const int old = atomicAdd(qk_counters + head_id, 1);
    trigger = ((old & (kHeadDim - 1)) == (kHeadDim - 1)) ? 1 : 0;
  }
  trigger = __shfl_sync(0xffffffffu, trigger, 0);
  __syncwarp();
  if (trigger) {
    finalize_qk_head_from_scratch(
        head_id,
        lane,
        qk_scratch,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        key_cache,
        cache_head_stride,
        pos,
        q_eps,
        k_eps);
  }
}

template <bool kHasBias, int kChunkRows>
__global__ void qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    const half* __restrict__ bias,
    const half* __restrict__ q_weight,
    const half* __restrict__ k_weight,
    const half* __restrict__ cos,
    const half* __restrict__ sin,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ q_out,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    half* __restrict__ qk_scratch,
    int32_t* __restrict__ qk_counters,
    int64_t cache_head_stride,
    float q_eps,
    float k_eps) {
  static_assert(kHeadDim % kChunkRows == 0, "q/k chunk rows must divide head_dim");
  constexpr int kChunksPerHead = kHeadDim / kChunkRows;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;
  int row = 0;
  int head_id = 0;
  bool is_qk = block < (kQkHeads * kChunksPerHead);

  if (is_qk) {
    head_id = block / kChunksPerHead;
    const int chunk = block - head_id * kChunksPerHead;
    const int d = chunk * kChunkRows + warp;
    row = head_id < kQueryHeads ? head_id * kHeadDim + d : kQSize + (head_id - kQueryHeads) * kHeadDim + d;
  } else {
    const int v_row = (block - kQkHeads * kChunksPerHead) * kChunkRows + warp;
    row = kVOffset + v_row;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* w4 = reinterpret_cast<const uint32_t*>(
      weight_fp8 + static_cast<int64_t>(row) * kHiddenSize);
  const half* row_scales = scales + static_cast<int64_t>(row) * kFp8ScaleBlocks;

  float acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const float scale = __half2float(row_scales[block_k]);
    const int word = block_k * kFp8WordsPerBlock + lane;
    const uint32_t packed = load_u32_cg(w4 + word);
    uint32_t w01_bits;
    uint32_t w23_bits;
    fp8e4b15x4_to_half2_bits(packed, w01_bits, w23_bits);

    const int x_pair = block_k * (kFp8BlockK / 2) + lane * 2;
    const float block_acc = dot4_half2_haccum(
        x2[x_pair],
        half2_from_u32(w01_bits),
        x2[x_pair + 1],
        half2_from_u32(w23_bits));
    acc = fmaf(block_acc, scale, acc);
  }

  acc = warp_sum(acc);
  if constexpr (kHasBias) {
    acc += __half2float(bias[row]);
  }
  const half out = __float2half_rn(acc);
  const int64_t pos = cache_position[0];

  if (!is_qk) {
    if (lane == 0) {
      const int v_row = row - kVOffset;
      const int kv_head = v_row / kHeadDim;
      const int d = v_row - kv_head * kHeadDim;
      value_cache[static_cast<int64_t>(kv_head) * cache_head_stride + pos * kHeadDim + d] = out;
    }
    return;
  }

  if (lane == 0) {
    const int d = row < kQSize ? row & (kHeadDim - 1) : (row - kQSize) & (kHeadDim - 1);
    qk_scratch[head_id * kHeadDim + d] = out;
    __threadfence();
  }
  __syncthreads();

  int trigger = 0;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(qk_counters + head_id, 1);
    trigger = ((old & (kChunksPerHead - 1)) == (kChunksPerHead - 1)) ? 1 : 0;
  }
  trigger = __shfl_sync(0xffffffffu, trigger, 0);
  if (trigger && warp == 0) {
    finalize_qk_head_from_scratch(
        head_id,
        lane,
        qk_scratch,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        key_cache,
        cache_head_stride,
        pos,
        q_eps,
        k_eps);
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    const torch::Tensor& bias,
    bool has_bias,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(x);
  CHECK_HALF(weight);
  CHECK_HALF(out);
  TORCH_CHECK(x.numel() == kHiddenSize, "x.numel() must be 2048");
  TORCH_CHECK(weight.dim() == 2, "weight must be [N, 2048]");
  TORCH_CHECK(weight.size(1) == kHiddenSize, "weight K dimension must be 2048");
  TORCH_CHECK(out.numel() == weight.size(0), "out must have N elements");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
  if (has_bias) {
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(bias);
    CHECK_HALF(bias);
    TORCH_CHECK(bias.numel() == weight.size(0), "bias must have N elements");
  }
}

void check_fp8_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    const torch::Tensor& bias,
    bool has_bias,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(x);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  CHECK_HALF(out);
  TORCH_CHECK(x.numel() == kHiddenSize, "x.numel() must be 2048");
  TORCH_CHECK(weight_fp8.dim() == 2, "weight_fp8 must be [N, 2048]");
  TORCH_CHECK(weight_fp8.size(1) == kHiddenSize, "weight_fp8 K dimension must be 2048");
  TORCH_CHECK(scales.dim() == 2, "scales must be [N, 16]");
  TORCH_CHECK(scales.size(0) == weight_fp8.size(0), "scales N dimension must match weight_fp8");
  TORCH_CHECK(scales.size(1) == kFp8ScaleBlocks, "scales K-block dimension must be 16");
  TORCH_CHECK(out.numel() == weight_fp8.size(0), "out must have N elements");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
  if (has_bias) {
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(bias);
    CHECK_HALF(bias);
    TORCH_CHECK(bias.numel() == weight_fp8.size(0), "bias must have N elements");
  }
}

void check_int4_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    const torch::Tensor& bias,
    bool has_bias,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_int4);
  CHECK_CUDA(scales);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_int4);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(x);
  CHECK_BYTE(weight_int4);
  CHECK_HALF(scales);
  CHECK_HALF(out);
  TORCH_CHECK(x.numel() == kHiddenSize, "x.numel() must be 2048");
  TORCH_CHECK(weight_int4.dim() == 3, "weight_int4 must be [16, N, 64]");
  TORCH_CHECK(weight_int4.size(0) == kInt4ScaleBlocks, "weight_int4 K-block dimension must be 16");
  TORCH_CHECK(weight_int4.size(2) == kInt4PackedBytesPerBlock, "weight_int4 packed dimension must be 64");
  TORCH_CHECK(scales.dim() == 2, "scales must be [16, N]");
  TORCH_CHECK(scales.size(0) == kInt4ScaleBlocks, "scales K-block dimension must be 16");
  TORCH_CHECK(scales.size(1) == weight_int4.size(1), "scales N dimension must match weight_int4");
  TORCH_CHECK(out.numel() == weight_int4.size(1), "out must have N elements");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
  if (has_bias) {
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(bias);
    CHECK_HALF(bias);
    TORCH_CHECK(bias.numel() == weight_int4.size(1), "bias must have N elements");
  }
}

void check_qkv_qk_update_inputs(
    const torch::Tensor& qkv,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& out_query,
    int64_t num_threads) {
  CHECK_CUDA(qkv);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(cache_position);
  CHECK_CUDA(q_weight);
  CHECK_CUDA(k_weight);
  CHECK_CUDA(cos);
  CHECK_CUDA(sin);
  CHECK_CUDA(out_query);
  CHECK_CONTIGUOUS(qkv);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_CONTIGUOUS(cos);
  CHECK_CONTIGUOUS(sin);
  CHECK_HALF(qkv);
  CHECK_HALF(key_cache);
  CHECK_HALF(value_cache);
  CHECK_HALF(q_weight);
  CHECK_HALF(k_weight);
  CHECK_HALF(cos);
  CHECK_HALF(sin);
  CHECK_HALF(out_query);
  TORCH_CHECK(qkv.numel() == kQSize + kKSize + kKSize, "qkv must contain 4096 fp16 values");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [B, 8, T, 128]");
  TORCH_CHECK(value_cache.dim() == 4, "value_cache must be [B, 8, T, 128]");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key/value cache shapes must match");
  TORCH_CHECK(key_cache.size(0) == 1, "qkv qk update candidate currently supports batch size 1");
  TORCH_CHECK(key_cache.size(1) == kKvHeads, "key_cache KV heads must be 8");
  TORCH_CHECK(key_cache.size(3) == kHeadDim, "key_cache head_dim must be 128");
  TORCH_CHECK(q_weight.numel() == kHeadDim, "q_weight must have 128 elements");
  TORCH_CHECK(k_weight.numel() == kHeadDim, "k_weight must have 128 elements");
  TORCH_CHECK(cos.dim() == 2 && cos.size(1) == kHeadDim, "cos must be [B, 128]");
  TORCH_CHECK(sin.dim() == 2 && sin.size(1) == kHeadDim, "sin must be [B, 128]");
  TORCH_CHECK(cos.size(0) == key_cache.size(0), "cos batch must match cache batch");
  TORCH_CHECK(sin.size(0) == key_cache.size(0), "sin batch must match cache batch");
  TORCH_CHECK(out_query.dim() == 3, "out_query must be [B, 16, 128]");
  TORCH_CHECK(out_query.size(0) == key_cache.size(0), "out_query batch must match cache batch");
  TORCH_CHECK(out_query.size(1) == kQueryHeads, "out_query heads must be 16");
  TORCH_CHECK(out_query.size(2) == kHeadDim, "out_query head_dim must be 128");
  TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");
  TORCH_CHECK(cache_position.scalar_type() == at::kLong, "cache_position must be int64");
  TORCH_CHECK(
      num_threads == 128 || num_threads == 512,
      "num_threads must be 128 or 512 for the qkv qk update candidate");
}

void check_qkv_qk_update_rope_cache_inputs(
    const torch::Tensor& qkv,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& rope_cache,
    const torch::Tensor& rope_position,
    const torch::Tensor& out_query,
    int64_t num_threads) {
  CHECK_CUDA(qkv);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(cache_position);
  CHECK_CUDA(q_weight);
  CHECK_CUDA(k_weight);
  CHECK_CUDA(rope_cache);
  CHECK_CUDA(rope_position);
  CHECK_CUDA(out_query);
  CHECK_CONTIGUOUS(qkv);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_CONTIGUOUS(rope_cache);
  CHECK_HALF(qkv);
  CHECK_HALF(key_cache);
  CHECK_HALF(value_cache);
  CHECK_HALF(q_weight);
  CHECK_HALF(k_weight);
  CHECK_HALF(rope_cache);
  CHECK_HALF(out_query);
  TORCH_CHECK(qkv.numel() == kQSize + kKSize + kKSize, "qkv must contain 4096 fp16 values");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [B, 8, T, 128]");
  TORCH_CHECK(value_cache.dim() == 4, "value_cache must be [B, 8, T, 128]");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key/value cache shapes must match");
  TORCH_CHECK(key_cache.size(0) == 1, "qkv qk update candidate currently supports batch size 1");
  TORCH_CHECK(key_cache.size(1) == kKvHeads, "key_cache KV heads must be 8");
  TORCH_CHECK(key_cache.size(3) == kHeadDim, "key_cache head_dim must be 128");
  TORCH_CHECK(q_weight.numel() == kHeadDim, "q_weight must have 128 elements");
  TORCH_CHECK(k_weight.numel() == kHeadDim, "k_weight must have 128 elements");
  TORCH_CHECK(rope_cache.dim() == 3, "rope_cache must be [T, 2, 128]");
  TORCH_CHECK(rope_cache.size(1) == 2 && rope_cache.size(2) == kHeadDim, "rope_cache must be [T, 2, 128]");
  TORCH_CHECK(out_query.dim() == 3, "out_query must be [B, 16, 128]");
  TORCH_CHECK(out_query.size(0) == key_cache.size(0), "out_query batch must match cache batch");
  TORCH_CHECK(out_query.size(1) == kQueryHeads, "out_query heads must be 16");
  TORCH_CHECK(out_query.size(2) == kHeadDim, "out_query head_dim must be 128");
  TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");
  TORCH_CHECK(cache_position.scalar_type() == at::kLong, "cache_position must be int64");
  TORCH_CHECK(rope_position.numel() == 1, "rope_position must contain one element");
  TORCH_CHECK(rope_position.scalar_type() == at::kLong, "rope_position must be int64");
  TORCH_CHECK(num_threads == 512, "rope-cache qkv qk update supports only num_threads=512");
}

void check_fused_qkv_qk_update_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& bias,
    bool has_bias,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& out_query,
    const torch::Tensor& qk_scratch,
    const torch::Tensor& qk_counters) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(cache_position);
  CHECK_CUDA(q_weight);
  CHECK_CUDA(k_weight);
  CHECK_CUDA(cos);
  CHECK_CUDA(sin);
  CHECK_CUDA(out_query);
  CHECK_CUDA(qk_scratch);
  CHECK_CUDA(qk_counters);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_CONTIGUOUS(cos);
  CHECK_CONTIGUOUS(sin);
  CHECK_CONTIGUOUS(qk_scratch);
  CHECK_CONTIGUOUS(qk_counters);
  CHECK_HALF(x);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  CHECK_HALF(key_cache);
  CHECK_HALF(value_cache);
  CHECK_HALF(q_weight);
  CHECK_HALF(k_weight);
  CHECK_HALF(cos);
  CHECK_HALF(sin);
  CHECK_HALF(out_query);
  CHECK_HALF(qk_scratch);
  CHECK_INT(qk_counters);
  TORCH_CHECK(x.numel() == kHiddenSize, "x.numel() must be 2048");
  TORCH_CHECK(weight_fp8.dim() == 2, "weight_fp8 must be [4096, 2048]");
  TORCH_CHECK(weight_fp8.size(0) == kQkvOutSize, "fused qkv/update expects N=4096");
  TORCH_CHECK(weight_fp8.size(1) == kHiddenSize, "weight_fp8 K dimension must be 2048");
  TORCH_CHECK(scales.dim() == 2, "scales must be [4096, 16]");
  TORCH_CHECK(scales.size(0) == kQkvOutSize, "scales N dimension must be 4096");
  TORCH_CHECK(scales.size(1) == kFp8ScaleBlocks, "scales K-block dimension must be 16");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [1, 8, T, 128]");
  TORCH_CHECK(value_cache.dim() == 4, "value_cache must be [1, 8, T, 128]");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(), "key/value cache shapes must match");
  TORCH_CHECK(key_cache.size(0) == 1, "fused qkv/update supports batch size 1");
  TORCH_CHECK(key_cache.size(1) == kKvHeads, "key_cache KV heads must be 8");
  TORCH_CHECK(key_cache.size(3) == kHeadDim, "key_cache head_dim must be 128");
  TORCH_CHECK(
      key_cache.stride(3) == 1 && key_cache.stride(2) == kHeadDim &&
          value_cache.stride(3) == 1 && value_cache.stride(2) == kHeadDim,
      "fused qkv/update requires contiguous cache head_dim/time strides");
  TORCH_CHECK(q_weight.numel() == kHeadDim, "q_weight must have 128 elements");
  TORCH_CHECK(k_weight.numel() == kHeadDim, "k_weight must have 128 elements");
  TORCH_CHECK(cos.dim() == 2 && cos.size(0) == 1 && cos.size(1) == kHeadDim, "cos must be [1, 128]");
  TORCH_CHECK(sin.dim() == 2 && sin.size(0) == 1 && sin.size(1) == kHeadDim, "sin must be [1, 128]");
  TORCH_CHECK(out_query.dim() == 3, "out_query must be [1, 16, 128]");
  TORCH_CHECK(out_query.size(0) == 1, "out_query batch must be 1");
  TORCH_CHECK(out_query.size(1) == kQueryHeads, "out_query heads must be 16");
  TORCH_CHECK(out_query.size(2) == kHeadDim, "out_query head_dim must be 128");
  TORCH_CHECK(out_query.stride(2) == 1 && out_query.stride(1) == kHeadDim, "out_query must be head-contiguous");
  TORCH_CHECK(qk_scratch.dim() == 2, "qk_scratch must be [24, 128]");
  TORCH_CHECK(qk_scratch.size(0) == kQkHeads && qk_scratch.size(1) == kHeadDim, "qk_scratch must be [24, 128]");
  TORCH_CHECK(qk_counters.numel() == kQkHeads, "qk_counters must contain 24 int32 values");
  TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");
  TORCH_CHECK(cache_position.scalar_type() == at::kLong, "cache_position must be int64");
  if (has_bias) {
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(bias);
    CHECK_HALF(bias);
    TORCH_CHECK(bias.numel() == kQkvOutSize, "bias must have 4096 values");
  }
}

}  // namespace

void qkv_gemv_half2_vec4_f32acc_cg_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias,
    int64_t rows_per_block) {
  check_inputs(x, weight, out, bias, has_bias, rows_per_block);
  int n_out = static_cast<int>(weight.size(0));
  int blocks = static_cast<int>((n_out + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_half2_vec4_f32acc_cg_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      bias_ptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      n_out,
      static_cast<int>(rows_per_block),
      has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qkv_gemv_fp8e4b15_block128_f32acc_cg_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias,
    int64_t rows_per_block) {
  check_fp8_inputs(x, weight_fp8, scales, out, bias, has_bias, rows_per_block);
  int n_out = static_cast<int>(weight_fp8.size(0));
  int blocks = static_cast<int>((n_out + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_fp8e4b15_block128_f32acc_cg_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      bias_ptr,
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      n_out,
      static_cast<int>(rows_per_block),
      has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kHasBias>
void launch_qkv_gemv_fp8e4b15_static_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_fp8e4b15_block128_f32acc_cg_static_k2048_kernel<kHasBias>
      <<<kQkvOutSize, 32, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qkv_gemv_fp8e4b15_block128_f32acc_cg_static_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias) {
  check_fp8_inputs(x, weight_fp8, scales, out, bias, has_bias, 1);
  TORCH_CHECK(weight_fp8.size(0) == kQkvOutSize, "static qkv FP8 kernel requires N=4096");
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;

  if (has_bias) {
    launch_qkv_gemv_fp8e4b15_static_k2048<true>(x, weight_fp8, scales, bias_ptr, out);
  } else {
    launch_qkv_gemv_fp8e4b15_static_k2048<false>(x, weight_fp8, scales, bias_ptr, out);
  }
}

template <bool kHasBias>
void launch_qkv_gemv_fp8e4b15_hscale_static_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_fp8e4b15_block128_hscale_cg_static_k2048_kernel<kHasBias>
      <<<kQkvOutSize, 32, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qkv_gemv_fp8e4b15_block128_hscale_cg_static_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias) {
  check_fp8_inputs(x, weight_fp8, scales, out, bias, has_bias, 1);
  TORCH_CHECK(weight_fp8.size(0) == kQkvOutSize, "static qkv FP8 hscale kernel requires N=4096");
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;

  if (has_bias) {
    launch_qkv_gemv_fp8e4b15_hscale_static_k2048<true>(x, weight_fp8, scales, bias_ptr, out);
  } else {
    launch_qkv_gemv_fp8e4b15_hscale_static_k2048<false>(x, weight_fp8, scales, bias_ptr, out);
  }
}

template <bool kHasBias>
void launch_qkv_fp8e4b15_hscale_static_qk_update_gqa2(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_weight,
    float q_eps,
    const torch::Tensor& k_weight,
    float k_eps,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& out_query,
    const torch::Tensor& qk_scratch,
    const torch::Tensor& qk_counters) {
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_fp8e4b15_hscale_static_qk_update_gqa2_kernel<kHasBias>
      <<<kQkvOutSize, 32, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
          reinterpret_cast<const int64_t*>(cache_position.data_ptr<int64_t>()),
          reinterpret_cast<half*>(out_query.data_ptr<at::Half>()),
          reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
          reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
          reinterpret_cast<half*>(qk_scratch.data_ptr<at::Half>()),
          reinterpret_cast<int32_t*>(qk_counters.data_ptr<int32_t>()),
          key_cache.stride(1),
          q_eps,
          k_eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qkv_fp8e4b15_hscale_static_qk_update_gqa2(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    bool has_bias,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor q_weight,
    double q_eps,
    torch::Tensor k_weight,
    double k_eps,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out_query,
    torch::Tensor qk_scratch,
    torch::Tensor qk_counters) {
  check_fused_qkv_qk_update_inputs(
      x,
      weight_fp8,
      scales,
      bias,
      has_bias,
      key_cache,
      value_cache,
      cache_position,
      q_weight,
      k_weight,
      cos,
      sin,
      out_query,
      qk_scratch,
      qk_counters);
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
  if (has_bias) {
    launch_qkv_fp8e4b15_hscale_static_qk_update_gqa2<true>(
        x,
        weight_fp8,
        scales,
        bias_ptr,
        key_cache,
        value_cache,
        cache_position,
        q_weight,
        static_cast<float>(q_eps),
        k_weight,
        static_cast<float>(k_eps),
        cos,
        sin,
        out_query,
        qk_scratch,
        qk_counters);
  } else {
    launch_qkv_fp8e4b15_hscale_static_qk_update_gqa2<false>(
        x,
        weight_fp8,
        scales,
        bias_ptr,
        key_cache,
        value_cache,
        cache_position,
        q_weight,
        static_cast<float>(q_eps),
        k_weight,
        static_cast<float>(k_eps),
        cos,
        sin,
        out_query,
        qk_scratch,
        qk_counters);
  }
}

template <bool kHasBias, int kChunkRows>
void launch_qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_weight,
    float q_eps,
    const torch::Tensor& k_weight,
    float k_eps,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& out_query,
    const torch::Tensor& qk_scratch,
    const torch::Tensor& qk_counters) {
  static_assert(kHeadDim % kChunkRows == 0, "q/k chunk rows must divide head_dim");
  constexpr int kChunksPerHead = kHeadDim / kChunkRows;
  constexpr int kQkBlocks = kQkHeads * kChunksPerHead;
  constexpr int kVBlocks = kKSize / kChunkRows;
  constexpr int kThreads = kChunkRows * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk_kernel<kHasBias, kChunkRows>
      <<<kQkBlocks + kVBlocks, kThreads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
          reinterpret_cast<const int64_t*>(cache_position.data_ptr<int64_t>()),
          reinterpret_cast<half*>(out_query.data_ptr<at::Half>()),
          reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
          reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
          reinterpret_cast<half*>(qk_scratch.data_ptr<at::Half>()),
          reinterpret_cast<int32_t*>(qk_counters.data_ptr<int32_t>()),
          key_cache.stride(1),
          q_eps,
          k_eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kChunkRows>
void qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    bool has_bias,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor q_weight,
    double q_eps,
    torch::Tensor k_weight,
    double k_eps,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out_query,
    torch::Tensor qk_scratch,
    torch::Tensor qk_counters) {
  check_fused_qkv_qk_update_inputs(
      x,
      weight_fp8,
      scales,
      bias,
      has_bias,
      key_cache,
      value_cache,
      cache_position,
      q_weight,
      k_weight,
      cos,
      sin,
      out_query,
      qk_scratch,
      qk_counters);
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
  if (has_bias) {
    launch_qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk<true, kChunkRows>(
        x,
        weight_fp8,
        scales,
        bias_ptr,
        key_cache,
        value_cache,
        cache_position,
        q_weight,
        static_cast<float>(q_eps),
        k_weight,
        static_cast<float>(k_eps),
        cos,
        sin,
        out_query,
        qk_scratch,
        qk_counters);
  } else {
    launch_qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk<false, kChunkRows>(
        x,
        weight_fp8,
        scales,
        bias_ptr,
        key_cache,
        value_cache,
        cache_position,
        q_weight,
        static_cast<float>(q_eps),
        k_weight,
        static_cast<float>(k_eps),
        cos,
        sin,
        out_query,
        qk_scratch,
        qk_counters);
  }
}

void qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk32(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    bool has_bias,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor q_weight,
    double q_eps,
    torch::Tensor k_weight,
    double k_eps,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out_query,
    torch::Tensor qk_scratch,
    torch::Tensor qk_counters) {
  qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk<32>(
      x,
      weight_fp8,
      scales,
      bias,
      has_bias,
      key_cache,
      value_cache,
      cache_position,
      q_weight,
      q_eps,
      k_weight,
      k_eps,
      cos,
      sin,
      out_query,
      qk_scratch,
      qk_counters);
}

void qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk16(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor bias,
    bool has_bias,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor q_weight,
    double q_eps,
    torch::Tensor k_weight,
    double k_eps,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out_query,
    torch::Tensor qk_scratch,
    torch::Tensor qk_counters) {
  qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk<16>(
      x,
      weight_fp8,
      scales,
      bias,
      has_bias,
      key_cache,
      value_cache,
      cache_position,
      q_weight,
      q_eps,
      k_weight,
      k_eps,
      cos,
      sin,
      out_query,
      qk_scratch,
      qk_counters);
}

template <int kRowsPerBlock>
void launch_qkv_gemv_int4_sym_fullwarp_u16_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int n_out,
    bool has_bias) {
  const int blocks = (n_out + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = kRowsPerBlock * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          n_out,
          has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock, bool kHasBias>
void launch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out) {
  constexpr int kBlocks = kQkvOutSize / kRowsPerBlock;
  const int threads = kRowsPerBlock * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_static_k2048_kernel<kRowsPerBlock, kHasBias>
      <<<kBlocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock, bool kHasBias>
void launch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "fullwarp2 needs even rows_per_block");
  constexpr int kBlocks = kQkvOutSize / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_int4_sym_kblock_kscale_fullwarp2_u16_static_k2048_kernel<kRowsPerBlock, kHasBias>
      <<<kBlocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock, bool kHasBias>
void launch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out) {
  constexpr int kBlocks = kQkvOutSize / kRowsPerBlock;
  const int threads = kRowsPerBlock * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_static_u4offset_k2048_kernel<kRowsPerBlock, kHasBias>
      <<<kBlocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock, bool kHasBias>
void launch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out) {
  constexpr int kBlocks = kQkvOutSize / kRowsPerBlock;
  const int threads = kRowsPerBlock * 32;
  const int shared_bytes = static_cast<int>((kHiddenSize / 2) * sizeof(half2));
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_int4_sym_kblock_kscale_fullwarp_u16_static_sharedx_k2048_kernel<kRowsPerBlock, kHasBias>
      <<<kBlocks, threads, shared_bytes, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kHasBias>
void dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 1:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048<1, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 2:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048<2, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 4:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048<4, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 8:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048<8, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    default:
      TORCH_CHECK(false, "fullwarp-u16-static rows_per_block must be one of 1, 2, 4, or 8");
  }
}

template <bool kHasBias>
void dispatch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048<2, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 4:
      launch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048<4, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 8:
      launch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048<8, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 16:
      launch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048<16, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    default:
      TORCH_CHECK(false, "fullwarp2-u16-static rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <bool kHasBias>
void dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 1:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048<1, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 2:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048<2, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 4:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048<4, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 8:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048<8, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    default:
      TORCH_CHECK(false, "fullwarp-u16-static-u4offset rows_per_block must be one of 1, 2, 4, or 8");
  }
}

template <bool kHasBias>
void dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 1:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048<1, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 2:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048<2, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 4:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048<4, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    case 8:
      launch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048<8, kHasBias>(x, weight_int4, scales, bias_ptr, out);
      break;
    default:
      TORCH_CHECK(false, "fullwarp-u16-static-sharedx rows_per_block must be one of 1, 2, 4, or 8");
  }
}

template <int kRowsPerBlock, bool kBroadcastScale>
void launch_qkv_gemv_int4_sym_halfwarp_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int n_out,
    bool has_bias) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  const int blocks = (n_out + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_int4_sym_kblock_kscale_halfwarp_k2048_kernel<kRowsPerBlock, kBroadcastScale>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          n_out,
          has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock, bool kBroadcastScale>
void launch_qkv_gemv_int4_sym_quarterwarp_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int n_out,
    bool has_bias) {
  static_assert(kRowsPerBlock % 4 == 0, "quarter-warp kernel needs rows_per_block divisible by 4");
  const int blocks = (n_out + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 4) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  qkv_gemv_int4_sym_kblock_kscale_quarterwarp_k2048_kernel<kRowsPerBlock, kBroadcastScale>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          bias_ptr,
          reinterpret_cast<half*>(out.data_ptr<at::Half>()),
          n_out,
          has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kBroadcastScale>
void dispatch_qkv_gemv_int4_sym_halfwarp_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int n_out,
    bool has_bias,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_qkv_gemv_int4_sym_halfwarp_k2048<2, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    case 4:
      launch_qkv_gemv_int4_sym_halfwarp_k2048<4, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    case 8:
      launch_qkv_gemv_int4_sym_halfwarp_k2048<8, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    case 16:
      launch_qkv_gemv_int4_sym_halfwarp_k2048<16, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <bool kBroadcastScale>
void dispatch_qkv_gemv_int4_sym_quarterwarp_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const half* bias_ptr,
    const torch::Tensor& out,
    int n_out,
    bool has_bias,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 4:
      launch_qkv_gemv_int4_sym_quarterwarp_k2048<4, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    case 8:
      launch_qkv_gemv_int4_sym_quarterwarp_k2048<8, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    case 16:
      launch_qkv_gemv_int4_sym_quarterwarp_k2048<16, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    case 32:
      launch_qkv_gemv_int4_sym_quarterwarp_k2048<32, kBroadcastScale>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
      break;
    default:
      TORCH_CHECK(false, "quarter-warp rows_per_block must be one of 4, 8, 16, or 32");
  }
}

void qkv_gemv_int4_sym_kblock_kscale_k2048(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias,
    int64_t rows_per_block,
    std::string mode) {
  check_int4_inputs(x, weight_int4, scales, out, bias, has_bias, rows_per_block);
  int n_out = static_cast<int>(weight_int4.size(1));
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;

  if (mode == "int4-sym-kblock-kscale-fullwarp-u16-k2048") {
    switch (rows_per_block) {
      case 1:
        launch_qkv_gemv_int4_sym_fullwarp_u16_k2048<1>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
        break;
      case 2:
        launch_qkv_gemv_int4_sym_fullwarp_u16_k2048<2>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
        break;
      case 4:
        launch_qkv_gemv_int4_sym_fullwarp_u16_k2048<4>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
        break;
      case 8:
        launch_qkv_gemv_int4_sym_fullwarp_u16_k2048<8>(x, weight_int4, scales, bias_ptr, out, n_out, has_bias);
        break;
      default:
        TORCH_CHECK(false, "fullwarp-u16 rows_per_block must be one of 1, 2, 4, or 8");
    }
    return;
  }
  if (mode == "int4-sym-kblock-kscale-fullwarp-u16-static-k2048") {
    TORCH_CHECK(n_out == kQkvOutSize, "fullwarp-u16-static expects qkv N=4096");
    if (has_bias) {
      dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048<true>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    } else {
      dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_k2048<false>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    }
    return;
  }
  if (mode == "int4-sym-kblock-kscale-fullwarp2-u16-static-k2048") {
    TORCH_CHECK(n_out == kQkvOutSize, "fullwarp2-u16-static expects qkv N=4096");
    if (has_bias) {
      dispatch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048<true>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    } else {
      dispatch_qkv_gemv_int4_sym_fullwarp2_u16_static_k2048<false>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    }
    return;
  }
  if (mode == "int4-sym-kblock-kscale-fullwarp-u16-static-u4offset-k2048") {
    TORCH_CHECK(n_out == kQkvOutSize, "fullwarp-u16-static-u4offset expects qkv N=4096");
    if (has_bias) {
      dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048<true>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    } else {
      dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_u4offset_k2048<false>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    }
    return;
  }
  if (mode == "int4-sym-kblock-kscale-fullwarp-u16-static-sharedx-k2048") {
    TORCH_CHECK(n_out == kQkvOutSize, "fullwarp-u16-static-sharedx expects qkv N=4096");
    if (has_bias) {
      dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048<true>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    } else {
      dispatch_qkv_gemv_int4_sym_fullwarp_u16_static_sharedx_k2048<false>(
          x, weight_int4, scales, bias_ptr, out, rows_per_block);
    }
    return;
  }
  if (mode == "int4-sym-kblock-kscale-halfwarp-k2048") {
    dispatch_qkv_gemv_int4_sym_halfwarp_k2048<false>(
        x, weight_int4, scales, bias_ptr, out, n_out, has_bias, rows_per_block);
    return;
  }
  if (mode == "int4-sym-kblock-kscale-halfwarp-bscale-k2048") {
    dispatch_qkv_gemv_int4_sym_halfwarp_k2048<true>(
        x, weight_int4, scales, bias_ptr, out, n_out, has_bias, rows_per_block);
    return;
  }
  if (mode == "int4-sym-kblock-kscale-quarterwarp-k2048") {
    dispatch_qkv_gemv_int4_sym_quarterwarp_k2048<false>(
        x, weight_int4, scales, bias_ptr, out, n_out, has_bias, rows_per_block);
    return;
  }
  if (mode == "int4-sym-kblock-kscale-quarterwarp-bscale-k2048") {
    dispatch_qkv_gemv_int4_sym_quarterwarp_k2048<true>(
        x, weight_int4, scales, bias_ptr, out, n_out, has_bias, rows_per_block);
    return;
  }
  TORCH_CHECK(false, "unsupported qkv int4 mode: ", mode);
}

void qkv_qk_rmsnorm_rope_kv_update_gqa2(
    torch::Tensor qkv,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor q_weight,
    double q_eps,
    torch::Tensor k_weight,
    double k_eps,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor out_query,
    int64_t num_threads,
    const std::string& mode) {
  check_qkv_qk_update_inputs(
      qkv,
      key_cache,
      value_cache,
      cache_position,
      q_weight,
      k_weight,
      cos,
      sin,
      out_query,
      num_threads);

  auto stream = at::cuda::getCurrentCUDAStream();
  if (mode == "warp") {
    TORCH_CHECK(num_threads == 128, "warp mode requires 128 threads");
    dim3 grid(kKvHeads, static_cast<unsigned>(key_cache.size(0)), 1);
    dim3 block(static_cast<unsigned>(num_threads), 1, 1);
    qkv_qk_rmsnorm_rope_kv_update_gqa2_warp_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<const int64_t*>(cache_position.data_ptr<int64_t>()),
        reinterpret_cast<half*>(out_query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
        cos.stride(0),
        cos.stride(1),
        out_query.stride(0),
        out_query.stride(1),
        out_query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
  } else if (mode == "warp-pair-vvec") {
    TORCH_CHECK(
        value_cache.stride(3) == 1,
        "warp-pair-vvec requires contiguous value-cache head_dim stride");
    TORCH_CHECK(num_threads == 128, "warp-pair-vvec mode requires 128 threads");
    dim3 grid(kKvHeads, static_cast<unsigned>(key_cache.size(0)), 1);
    dim3 block(static_cast<unsigned>(num_threads), 1, 1);
    qkv_qk_rmsnorm_rope_kv_update_gqa2_warp_pair_vvec_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<const int64_t*>(cache_position.data_ptr<int64_t>()),
        reinterpret_cast<half*>(out_query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
        cos.stride(0),
        cos.stride(1),
        out_query.stride(0),
        out_query.stride(1),
        out_query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
  } else if (mode == "warp-pair-vvec-contig") {
    TORCH_CHECK(
        cos.stride(1) == 1 && sin.stride(1) == 1 && out_query.stride(2) == 1 &&
            key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
        "warp-pair-vvec-contig requires contiguous q/k/cos/sin/value head_dim strides");
    TORCH_CHECK(num_threads == 128, "warp-pair-vvec-contig mode requires 128 threads");
    dim3 grid(kKvHeads, static_cast<unsigned>(key_cache.size(0)), 1);
    dim3 block(static_cast<unsigned>(num_threads), 1, 1);
    qkv_qk_rmsnorm_rope_kv_update_gqa2_warp_pair_vvec_kernel<true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<const int64_t*>(cache_position.data_ptr<int64_t>()),
        reinterpret_cast<half*>(out_query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
        cos.stride(0),
        cos.stride(1),
        out_query.stride(0),
        out_query.stride(1),
        out_query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
  } else if (mode == "kvquad-static-contig") {
    TORCH_CHECK(
        cos.stride(1) == 1 && sin.stride(1) == 1 && out_query.stride(2) == 1 &&
            out_query.stride(1) == kHeadDim && key_cache.stride(3) == 1 &&
            key_cache.stride(2) == kHeadDim && value_cache.stride(3) == 1 &&
            value_cache.stride(2) == kHeadDim,
        "kvquad-static-contig requires packed q/k/cos/sin/cache head_dim strides");
    TORCH_CHECK(num_threads == 512, "kvquad-static-contig mode requires 512 threads");
    dim3 grid(kKvHeads / 4, 1, 1);
    dim3 block(static_cast<unsigned>(num_threads), 1, 1);
    qkv_qk_rmsnorm_rope_kv_update_gqa2_kvgroup_static_kernel<4><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<const int64_t*>(cache_position.data_ptr<int64_t>()),
        reinterpret_cast<half*>(out_query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
        key_cache.stride(1),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
  } else {
    TORCH_CHECK(false, "unsupported qkv qk update mode: ", mode);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qkv_qk_rmsnorm_rope_cache_kv_update_gqa2(
    torch::Tensor qkv,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor q_weight,
    double q_eps,
    torch::Tensor k_weight,
    double k_eps,
    torch::Tensor rope_cache,
    torch::Tensor rope_position,
    torch::Tensor out_query,
    int64_t num_threads,
    const std::string& mode) {
  TORCH_CHECK(mode == "kvquad-static-contig", "rope-cache qkv qk update supports only kvquad-static-contig");
  check_qkv_qk_update_rope_cache_inputs(
      qkv,
      key_cache,
      value_cache,
      cache_position,
      q_weight,
      k_weight,
      rope_cache,
      rope_position,
      out_query,
      num_threads);

  TORCH_CHECK(
      out_query.stride(2) == 1 && out_query.stride(1) == kHeadDim &&
          key_cache.stride(3) == 1 && key_cache.stride(2) == kHeadDim &&
          value_cache.stride(3) == 1 && value_cache.stride(2) == kHeadDim,
      "rope-cache kvquad-static-contig requires packed q/k/cache head_dim strides");

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(kKvHeads / 4, 1, 1);
  dim3 block(static_cast<unsigned>(num_threads), 1, 1);
  qkv_qk_rmsnorm_rope_cache_kv_update_gqa2_kvgroup_static_kernel<4><<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(qkv.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(rope_cache.data_ptr<at::Half>()),
      reinterpret_cast<const int64_t*>(cache_position.data_ptr<int64_t>()),
      reinterpret_cast<const int64_t*>(rope_position.data_ptr<int64_t>()),
      reinterpret_cast<half*>(out_query.data_ptr<at::Half>()),
      reinterpret_cast<half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<half*>(value_cache.data_ptr<at::Half>()),
      key_cache.stride(1),
      static_cast<float>(q_eps),
      static_cast<float>(k_eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qkv_gemv_half2_vec4_f32acc_cg_k2048",
      &qkv_gemv_half2_vec4_f32acc_cg_k2048,
      "fp32-accum decode qkv GEMV kernel for Qwen3-VL hidden_size=2048");
  m.def(
      "qkv_gemv_fp8e4b15_block128_f32acc_cg_k2048",
      &qkv_gemv_fp8e4b15_block128_f32acc_cg_k2048,
      "fp8e4b15 block-128 decode qkv GEMV kernel for Qwen3-VL hidden_size=2048");
  m.def(
      "qkv_gemv_fp8e4b15_block128_f32acc_cg_static_k2048",
      &qkv_gemv_fp8e4b15_block128_f32acc_cg_static_k2048,
      "static fp8e4b15 block-128 decode qkv GEMV kernels for Qwen3-VL hidden_size=2048");
  m.def(
      "qkv_gemv_fp8e4b15_block128_hscale_cg_static_k2048",
      &qkv_gemv_fp8e4b15_block128_hscale_cg_static_k2048,
      "static fp8e4b15 block-128 decode qkv GEMV hscale kernel for Qwen3-VL hidden_size=2048");
  m.def(
      "qkv_fp8e4b15_hscale_static_qk_update_gqa2",
      &qkv_fp8e4b15_hscale_static_qk_update_gqa2,
      "fused fp8e4b15 hscale qkv GEMV plus q/k RMSNorm RoPE KV update for Qwen3-VL GQA2");
  m.def(
      "qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk32",
      &qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk32,
      "chunk32 fused fp8e4b15 hscale qkv GEMV plus q/k RMSNorm RoPE KV update for Qwen3-VL GQA2");
  m.def(
      "qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk16",
      &qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk16,
      "chunk16 fused fp8e4b15 hscale qkv GEMV plus q/k RMSNorm RoPE KV update for Qwen3-VL GQA2");
  m.def(
      "qkv_gemv_int4_sym_kblock_kscale_k2048",
      &qkv_gemv_int4_sym_kblock_kscale_k2048,
      "symmetric int4 block-128 decode qkv GEMV kernels for Qwen3-VL hidden_size=2048");
  m.def(
      "qkv_qk_rmsnorm_rope_kv_update_gqa2",
      &qkv_qk_rmsnorm_rope_kv_update_gqa2,
      "decode qkv flat q/k RMSNorm RoPE and KV-cache update for Qwen3-VL GQA2");
  m.def(
      "qkv_qk_rmsnorm_rope_cache_kv_update_gqa2",
      &qkv_qk_rmsnorm_rope_cache_kv_update_gqa2,
      "decode qkv flat q/k RMSNorm RoPE-cache and KV-cache update for Qwen3-VL GQA2");
}
