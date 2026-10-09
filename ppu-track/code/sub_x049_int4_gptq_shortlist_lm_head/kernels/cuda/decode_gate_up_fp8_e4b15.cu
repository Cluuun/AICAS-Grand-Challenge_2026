#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define CHECK_BYTE(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")

constexpr int kHiddenSize = 2048;
constexpr int kIntermediateSize = 6144;
constexpr int kFp8BlockK = 128;
constexpr int kFp8ScaleBlocks = kHiddenSize / kFp8BlockK;
constexpr int kFp8WordsPerBlock = kFp8BlockK / 4;
constexpr int kKblockWords = 2 * kIntermediateSize * kFp8WordsPerBlock;
constexpr int kKblockScaleStride = 2 * kIntermediateSize;

__forceinline__ __device__ float warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__forceinline__ __device__ float half_warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, 16);
  }
  return value;
}

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ uint32_t load_u32_cg(const uint32_t* ptr) {
  return __ldcg(ptr);
}

__forceinline__ __device__ float dot_acc_half2(float acc, half2 x, half2 w) {
  float2 xf = __half22float2(x);
  float2 wf = __half22float2(w);
  acc = fmaf(xf.x, wf.x, acc);
  acc = fmaf(xf.y, wf.y, acc);
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

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w4 = reinterpret_cast<const uint32_t*>(weight_fp8);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const uint32_t* block_w4 = weight_w4 + block_k * kKblockWords;
    const uint32_t* gate_w4 = block_w4 + row * kFp8WordsPerBlock;
    const uint32_t* up_w4 = block_w4 + (kIntermediateSize + row) * kFp8WordsPerBlock;
    uint32_t gate_packed = load_u32_cg(gate_w4 + lane);
    uint32_t up_packed = load_u32_cg(up_w4 + lane);

    uint32_t gate_w01_bits;
    uint32_t gate_w23_bits;
    uint32_t up_w01_bits;
    uint32_t up_w23_bits;
    fp8e4b15x4_to_half2_bits(gate_packed, gate_w01_bits, gate_w23_bits);
    fp8e4b15x4_to_half2_bits(up_packed, up_w01_bits, up_w23_bits);

    int x_pair = block_k * (kFp8BlockK / 2) + lane * 2;
    half2 x01 = x2[x_pair];
    half2 x23 = x2[x_pair + 1];
    float gate_block_acc = 0.0f;
    float up_block_acc = 0.0f;
    gate_block_acc = dot_acc_half2(gate_block_acc, x01, half2_from_u32(gate_w01_bits));
    gate_block_acc = dot_acc_half2(gate_block_acc, x23, half2_from_u32(gate_w23_bits));
    up_block_acc = dot_acc_half2(up_block_acc, x01, half2_from_u32(up_w01_bits));
    up_block_acc = dot_acc_half2(up_block_acc, x23, half2_from_u32(up_w23_bits));

    float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  float gate = warp_sum(gate_acc);
  float up = warp_sum(up_acc);
  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int half_id = lane >> 4;
  int lane16 = lane & 15;
  int row_in_block = warp_id * 2 + half_id;
  int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w4 = reinterpret_cast<const uint32_t*>(weight_fp8);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const uint32_t* block_w4 = weight_w4 + block_k * kKblockWords;
    const uint32_t* gate_w4 = block_w4 + row * kFp8WordsPerBlock;
    const uint32_t* up_w4 = block_w4 + (kIntermediateSize + row) * kFp8WordsPerBlock;

    uint32_t gate_packed0 = load_u32_cg(gate_w4 + lane16);
    uint32_t gate_packed1 = load_u32_cg(gate_w4 + lane16 + 16);
    uint32_t up_packed0 = load_u32_cg(up_w4 + lane16);
    uint32_t up_packed1 = load_u32_cg(up_w4 + lane16 + 16);

    uint32_t gate_w01_bits;
    uint32_t gate_w23_bits;
    uint32_t gate_w45_bits;
    uint32_t gate_w67_bits;
    uint32_t up_w01_bits;
    uint32_t up_w23_bits;
    uint32_t up_w45_bits;
    uint32_t up_w67_bits;
    fp8e4b15x4_to_half2_bits(gate_packed0, gate_w01_bits, gate_w23_bits);
    fp8e4b15x4_to_half2_bits(gate_packed1, gate_w45_bits, gate_w67_bits);
    fp8e4b15x4_to_half2_bits(up_packed0, up_w01_bits, up_w23_bits);
    fp8e4b15x4_to_half2_bits(up_packed1, up_w45_bits, up_w67_bits);

    int x_pair0 = block_k * (kFp8BlockK / 2) + lane16 * 2;
    int x_pair1 = x_pair0 + 32;
    half2 x01 = x2[x_pair0];
    half2 x23 = x2[x_pair0 + 1];
    half2 x45 = x2[x_pair1];
    half2 x67 = x2[x_pair1 + 1];

    float gate_block_acc = 0.0f;
    float up_block_acc = 0.0f;
    gate_block_acc = dot_acc_half2(gate_block_acc, x01, half2_from_u32(gate_w01_bits));
    gate_block_acc = dot_acc_half2(gate_block_acc, x23, half2_from_u32(gate_w23_bits));
    gate_block_acc = dot_acc_half2(gate_block_acc, x45, half2_from_u32(gate_w45_bits));
    gate_block_acc = dot_acc_half2(gate_block_acc, x67, half2_from_u32(gate_w67_bits));
    up_block_acc = dot_acc_half2(up_block_acc, x01, half2_from_u32(up_w01_bits));
    up_block_acc = dot_acc_half2(up_block_acc, x23, half2_from_u32(up_w23_bits));
    up_block_acc = dot_acc_half2(up_block_acc, x45, half2_from_u32(up_w45_bits));
    up_block_acc = dot_acc_half2(up_block_acc, x67, half2_from_u32(up_w67_bits));

    float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  float gate = half_warp_sum(gate_acc);
  float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int half_id = lane >> 4;
  int lane16 = lane & 15;
  int row_in_block = warp_id * 2 + half_id;
  int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w4 = reinterpret_cast<const uint32_t*>(weight_fp8);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const uint32_t* block_w4 = weight_w4 + block_k * kKblockWords;
    const uint32_t* gate_w4 = block_w4 + row * kFp8WordsPerBlock;
    const uint32_t* up_w4 = block_w4 + (kIntermediateSize + row) * kFp8WordsPerBlock;

    int x_pair0 = block_k * (kFp8BlockK / 2) + lane16 * 2;
    int x_pair1 = x_pair0 + 32;

    uint32_t w0;
    uint32_t w1;
    float gate_block_acc = 0.0f;
    w0 = load_u32_cg(gate_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_block_acc = dot_acc_half2(gate_block_acc, x2[x_pair0], half2_from_u32(w0));
    gate_block_acc = dot_acc_half2(gate_block_acc, x2[x_pair0 + 1], half2_from_u32(w1));
    w0 = load_u32_cg(gate_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_block_acc = dot_acc_half2(gate_block_acc, x2[x_pair1], half2_from_u32(w0));
    gate_block_acc = dot_acc_half2(gate_block_acc, x2[x_pair1 + 1], half2_from_u32(w1));

    float up_block_acc = 0.0f;
    w0 = load_u32_cg(up_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_block_acc = dot_acc_half2(up_block_acc, x2[x_pair0], half2_from_u32(w0));
    up_block_acc = dot_acc_half2(up_block_acc, x2[x_pair0 + 1], half2_from_u32(w1));
    w0 = load_u32_cg(up_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_block_acc = dot_acc_half2(up_block_acc, x2[x_pair1], half2_from_u32(w0));
    up_block_acc = dot_acc_half2(up_block_acc, x2[x_pair1 + 1], half2_from_u32(w1));

    float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  float gate = half_warp_sum(gate_acc);
  float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int half_id = lane >> 4;
  int lane16 = lane & 15;
  int row_in_block = warp_id * 2 + half_id;
  int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w4 = reinterpret_cast<const uint32_t*>(weight_fp8);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const uint32_t* block_w4 = weight_w4 + block_k * kKblockWords;
    const uint32_t* gate_w4 = block_w4 + row * kFp8WordsPerBlock;
    const uint32_t* up_w4 = block_w4 + (kIntermediateSize + row) * kFp8WordsPerBlock;

    int x_pair0 = block_k * (kFp8BlockK / 2) + lane16 * 2;
    int x_pair1 = x_pair0 + 32;
    half2 x0 = x2[x_pair0];
    half2 x1 = x2[x_pair0 + 1];
    half2 x2v = x2[x_pair1];
    half2 x3 = x2[x_pair1 + 1];

    uint32_t w0;
    uint32_t w1;
    half2 gate_hacc = __float2half2_rn(0.0f);
    w0 = load_u32_cg(gate_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_hacc = __hfma2(x0, half2_from_u32(w0), gate_hacc);
    gate_hacc = __hfma2(x1, half2_from_u32(w1), gate_hacc);
    w0 = load_u32_cg(gate_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_hacc = __hfma2(x2v, half2_from_u32(w0), gate_hacc);
    gate_hacc = __hfma2(x3, half2_from_u32(w1), gate_hacc);

    half2 up_hacc = __float2half2_rn(0.0f);
    w0 = load_u32_cg(up_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_hacc = __hfma2(x0, half2_from_u32(w0), up_hacc);
    up_hacc = __hfma2(x1, half2_from_u32(w1), up_hacc);
    w0 = load_u32_cg(up_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_hacc = __hfma2(x2v, half2_from_u32(w0), up_hacc);
    up_hacc = __hfma2(x3, half2_from_u32(w1), up_hacc);

    float2 gate_pair = __half22float2(gate_hacc);
    float2 up_pair = __half22float2(up_hacc);
    float gate_block_acc = gate_pair.x + gate_pair.y;
    float up_block_acc = up_pair.x + up_pair.y;
    float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  float gate = half_warp_sum(gate_acc);
  float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int half_id = lane >> 4;
  int lane16 = lane & 15;
  int row_in_block = warp_id * 2 + half_id;
  int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w4 = reinterpret_cast<const uint32_t*>(weight_fp8);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const uint32_t* block_w4 = weight_w4 + block_k * kKblockWords;
    const uint32_t* gate_w4 = block_w4 + row * kFp8WordsPerBlock;
    const uint32_t* up_w4 = block_w4 + (kIntermediateSize + row) * kFp8WordsPerBlock;

    int x_pair0 = block_k * (kFp8BlockK / 2) + lane16 * 2;
    int x_pair1 = x_pair0 + 32;

    uint32_t w0;
    uint32_t w1;
    half2 gate_hacc = __float2half2_rn(0.0f);
    w0 = load_u32_cg(gate_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_hacc = __hfma2(x2[x_pair0], half2_from_u32(w0), gate_hacc);
    gate_hacc = __hfma2(x2[x_pair0 + 1], half2_from_u32(w1), gate_hacc);
    w0 = load_u32_cg(gate_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_hacc = __hfma2(x2[x_pair1], half2_from_u32(w0), gate_hacc);
    gate_hacc = __hfma2(x2[x_pair1 + 1], half2_from_u32(w1), gate_hacc);

    half2 up_hacc = __float2half2_rn(0.0f);
    w0 = load_u32_cg(up_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_hacc = __hfma2(x2[x_pair0], half2_from_u32(w0), up_hacc);
    up_hacc = __hfma2(x2[x_pair0 + 1], half2_from_u32(w1), up_hacc);
    w0 = load_u32_cg(up_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_hacc = __hfma2(x2[x_pair1], half2_from_u32(w0), up_hacc);
    up_hacc = __hfma2(x2[x_pair1 + 1], half2_from_u32(w1), up_hacc);

    float2 gate_pair = __half22float2(gate_hacc);
    float2 up_pair = __half22float2(up_hacc);
    float gate_block_acc = gate_pair.x + gate_pair.y;
    float up_block_acc = up_pair.x + up_pair.y;
    float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  float gate = half_warp_sum(gate_acc);
  float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int half_id = lane >> 4;
  int lane16 = lane & 15;
  int row_in_block = warp_id * 2 + half_id;
  int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w4 = reinterpret_cast<const uint32_t*>(weight_fp8);

  half2 gate_acc_h = __float2half2_rn(0.0f);
  half2 up_acc_h = __float2half2_rn(0.0f);

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    const uint32_t* block_w4 = weight_w4 + block_k * kKblockWords;
    const uint32_t* gate_w4 = block_w4 + row * kFp8WordsPerBlock;
    const uint32_t* up_w4 = block_w4 + (kIntermediateSize + row) * kFp8WordsPerBlock;

    int x_pair0 = block_k * (kFp8BlockK / 2) + lane16 * 2;
    int x_pair1 = x_pair0 + 32;

    uint32_t w0;
    uint32_t w1;
    half2 gate_hacc = __float2half2_rn(0.0f);
    w0 = load_u32_cg(gate_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_hacc = __hfma2(x2[x_pair0], half2_from_u32(w0), gate_hacc);
    gate_hacc = __hfma2(x2[x_pair0 + 1], half2_from_u32(w1), gate_hacc);
    w0 = load_u32_cg(gate_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    gate_hacc = __hfma2(x2[x_pair1], half2_from_u32(w0), gate_hacc);
    gate_hacc = __hfma2(x2[x_pair1 + 1], half2_from_u32(w1), gate_hacc);

    half2 up_hacc = __float2half2_rn(0.0f);
    w0 = load_u32_cg(up_w4 + lane16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_hacc = __hfma2(x2[x_pair0], half2_from_u32(w0), up_hacc);
    up_hacc = __hfma2(x2[x_pair0 + 1], half2_from_u32(w1), up_hacc);
    w0 = load_u32_cg(up_w4 + lane16 + 16);
    fp8e4b15x4_to_half2_bits(w0, w0, w1);
    up_hacc = __hfma2(x2[x_pair1], half2_from_u32(w0), up_hacc);
    up_hacc = __hfma2(x2[x_pair1 + 1], half2_from_u32(w1), up_hacc);

    half2 gate_scale = __half2half2(scales[block_k * kKblockScaleStride + row]);
    half2 up_scale = __half2half2(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc_h = __hfma2(gate_hacc, gate_scale, gate_acc_h);
    up_acc_h = __hfma2(up_hacc, up_scale, up_acc_h);
  }

  float2 gate_pair = __half22float2(gate_acc_h);
  float2 up_pair = __half22float2(up_acc_h);
  float gate = half_warp_sum(gate_pair.x + gate_pair.y);
  float up = half_warp_sum(up_pair.x + up_pair.y);
  if (lane16 == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out,
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
  TORCH_CHECK(weight_fp8.numel() == kFp8ScaleBlocks * 2 * kIntermediateSize * kFp8BlockK,
              "weight_fp8 must contain [16, 2, 6144, 128]");
  TORCH_CHECK(scales.numel() == kFp8ScaleBlocks * 2 * kIntermediateSize,
              "scales must contain [16, 2, 6144]");
  TORCH_CHECK(out.numel() == kIntermediateSize, "out must contain 6144 elements");
  TORCH_CHECK(
      rows_per_block == 1 || rows_per_block == 2 || rows_per_block == 4 || rows_per_block == 8 ||
          rows_per_block == 16,
      "rows_per_block must be one of 1, 2, 4, 8, or 16");
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  int threads = kRowsPerBlock * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_fp8, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 1:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048<1>(x, weight_fp8, scales, out);
      break;
    case 2:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048<2>(x, weight_fp8, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048<4>(x, weight_fp8, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048<8>(x, weight_fp8, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048<16>(x, weight_fp8, scales, out);
      break;
    default:
      TORCH_CHECK(false, "rows_per_block must be one of 1, 2, 4, 8, or 16");
  }
}

void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_fp8, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048<2>(
          x, weight_fp8, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048<4>(
          x, weight_fp8, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048<8>(
          x, weight_fp8, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048<16>(
          x, weight_fp8, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_fp8, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048<2>(
          x, weight_fp8, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048<4>(
          x, weight_fp8, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048<8>(
          x, weight_fp8, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048<16>(
          x, weight_fp8, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_fp8, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048<2>(
          x, weight_fp8, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048<4>(
          x, weight_fp8, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048<8>(
          x, weight_fp8, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048<16>(
          x, weight_fp8, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_fp8, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048<2>(
          x, weight_fp8, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048<4>(
          x, weight_fp8, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048<8>(
          x, weight_fp8, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048<16>(
          x, weight_fp8, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

void gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_fp8, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048<2>(
          x, weight_fp8, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048<4>(
          x, weight_fp8, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048<8>(
          x, weight_fp8, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048<16>(
          x, weight_fp8, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048",
      &gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048,
      "FP8 e4b15 kblock/kscale gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048",
      &gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048,
      "FP8 e4b15 kblock/kscale half-warp gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048",
      &gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048,
      "FP8 e4b15 kblock/kscale low-register half-warp gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048",
      &gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048,
      "FP8 e4b15 kblock/kscale half-accumulating half-warp gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048",
      &gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048,
      "FP8 e4b15 kblock/kscale low-register half-accumulating half-warp gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048",
      &gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048,
      "FP8 e4b15 kblock/kscale half-scaled half-warp gate/up SwiGLU for Qwen3-VL decode");
}
