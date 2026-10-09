#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define CHECK_BYTE(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")

constexpr int kHiddenSize = 2048;
constexpr int kFp8BlockK = 128;
constexpr int kFp8ScaleBlocks = kHiddenSize / kFp8BlockK;
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

__forceinline__ __device__ float dot_acc_half2(float acc, half2 x, half2 w) {
  float2 xf = __half22float2(x);
  float2 wf = __half22float2(w);
  acc = fmaf(xf.x, wf.x, acc);
  acc = fmaf(xf.y, wf.y, acc);
  return acc;
}

__forceinline__ __device__ float dot_acc_half2_scaled(float acc, half2 x, half2 w, float scale) {
  float2 xf = __half22float2(x);
  float2 wf = __half22float2(w);
  acc = fmaf(xf.x, wf.x * scale, acc);
  acc = fmaf(xf.y, wf.y * scale, acc);
  return acc;
}

__forceinline__ __device__ half half_from_fp8_e4m3(uint8_t value) {
  __half_raw raw = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(value), __NV_E4M3);
  return *reinterpret_cast<half*>(&raw);
}

__forceinline__ __device__ half2 half2_from_fp8x2_e4m3(__nv_fp8x2_storage_t value) {
  __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(value, __NV_E4M3);
  return *reinterpret_cast<half2*>(&raw);
}

__global__ void gate_up_swiglu_half2_vec4_f32acc_cg_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const uint4* gate_w4 = reinterpret_cast<const uint4*>(
      weight + static_cast<int64_t>(row) * kHiddenSize);
  const uint4* up_w4 = reinterpret_cast<const uint4*>(
      weight + static_cast<int64_t>(row + intermediate_size) * kHiddenSize);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int vec = lane; vec < kVecCount; vec += 32) {
    uint4 xv = x4[vec];
    uint4 gv = load_uint4_cg(gate_w4 + vec);
    uint4 uv = load_uint4_cg(up_w4 + vec);

    half2 x0 = half2_from_u32(xv.x);
    half2 x1 = half2_from_u32(xv.y);
    half2 x2 = half2_from_u32(xv.z);
    half2 x3 = half2_from_u32(xv.w);

    gate_acc = dot_acc_half2(gate_acc, x0, half2_from_u32(gv.x));
    gate_acc = dot_acc_half2(gate_acc, x1, half2_from_u32(gv.y));
    gate_acc = dot_acc_half2(gate_acc, x2, half2_from_u32(gv.z));
    gate_acc = dot_acc_half2(gate_acc, x3, half2_from_u32(gv.w));
    up_acc = dot_acc_half2(up_acc, x0, half2_from_u32(uv.x));
    up_acc = dot_acc_half2(up_acc, x1, half2_from_u32(uv.y));
    up_acc = dot_acc_half2(up_acc, x2, half2_from_u32(uv.z));
    up_acc = dot_acc_half2(up_acc, x3, half2_from_u32(uv.w));
  }

  float gate = warp_sum(gate_acc);
  float up = warp_sum(up_acc);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

__global__ void quantize_e4m3_block128_k2048_kernel(
    const half* __restrict__ weight,
    uint8_t* __restrict__ weight_fp8,
    half* __restrict__ scales,
    int n_rows) {
  int row = blockIdx.x;
  int block_k = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= n_rows || block_k >= kFp8ScaleBlocks) {
    return;
  }

  int k_begin = block_k * kFp8BlockK;
  const half* row_weight = weight + static_cast<int64_t>(row) * kHiddenSize + k_begin;
  uint8_t* row_fp8 = weight_fp8 + static_cast<int64_t>(row) * kHiddenSize + k_begin;

  float local_absmax = 0.0f;
#pragma unroll
  for (int i = lane; i < kFp8BlockK; i += 32) {
    float v = __half2float(row_weight[i]);
    local_absmax = fmaxf(local_absmax, fabsf(v));
  }

  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_absmax = fmaxf(local_absmax, __shfl_down_sync(mask, local_absmax, offset));
  }
  float absmax = __shfl_sync(mask, local_absmax, 0);
  float scale = fmaxf(absmax / 448.0f, 1.0e-8f);
  float inv_scale = 1.0f / scale;
  if (lane == 0) {
    scales[row * kFp8ScaleBlocks + block_k] = __float2half_rn(scale);
  }

#pragma unroll
  for (int i = lane; i < kFp8BlockK; i += 32) {
    float v = __half2float(row_weight[i]) * inv_scale;
    row_fp8[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void gate_up_swiglu_fp8_e4m3_block128_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_fp8,
    const half* __restrict__ scales,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const auto* gate_w2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(
      weight_fp8 + static_cast<int64_t>(row) * kHiddenSize);
  const auto* up_w2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(
      weight_fp8 + static_cast<int64_t>(row + intermediate_size) * kHiddenSize);
  const half* gate_scales = scales + static_cast<int64_t>(row) * kFp8ScaleBlocks;
  const half* up_scales = scales + static_cast<int64_t>(row + intermediate_size) * kFp8ScaleBlocks;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kFp8ScaleBlocks; ++block_k) {
    float gate_scale = __half2float(gate_scales[block_k]);
    float up_scale = __half2float(up_scales[block_k]);
    int pair_begin = block_k * (kFp8BlockK / 2);
#pragma unroll
    for (int pair = lane; pair < (kFp8BlockK / 2); pair += 32) {
      int k_pair = pair_begin + pair;
      half2 xv = x2[k_pair];
      gate_acc = dot_acc_half2_scaled(gate_acc, xv, half2_from_fp8x2_e4m3(gate_w2[k_pair]), gate_scale);
      up_acc = dot_acc_half2_scaled(up_acc, xv, half2_from_fp8x2_e4m3(up_w2[k_pair]), up_scale);
    }
  }

  float gate = warp_sum(gate_acc);
  float up = warp_sum(up_acc);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
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
  TORCH_CHECK(weight.dim() == 2, "weight must be [2 * intermediate, 2048]");
  TORCH_CHECK(weight.size(1) == kHiddenSize, "weight K dimension must be 2048");
  TORCH_CHECK(weight.size(0) >= 2 * intermediate_size, "weight must contain gate rows followed by up rows");
  TORCH_CHECK(out.numel() == intermediate_size, "out must have intermediate_size elements");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
}

void check_quantize_inputs(
    const torch::Tensor& weight,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales) {
  CHECK_CUDA(weight);
  CHECK_CUDA(weight_fp8);
  CHECK_CUDA(scales);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(weight_fp8);
  CHECK_CONTIGUOUS(scales);
  CHECK_HALF(weight);
  CHECK_BYTE(weight_fp8);
  CHECK_HALF(scales);
  TORCH_CHECK(weight.dim() == 2, "weight must be [N, 2048]");
  TORCH_CHECK(weight.size(1) == kHiddenSize, "weight K dimension must be 2048");
  TORCH_CHECK(weight_fp8.sizes() == weight.sizes(), "weight_fp8 shape must match weight");
  TORCH_CHECK(scales.dim() == 2, "scales must be [N, 16]");
  TORCH_CHECK(scales.size(0) == weight.size(0), "scales N dimension must match weight");
  TORCH_CHECK(scales.size(1) == kFp8ScaleBlocks, "scales K-block dimension must be 16");
}

void check_fp8_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t intermediate_size,
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
  TORCH_CHECK(weight_fp8.dim() == 2, "weight_fp8 must be [2 * intermediate, 2048]");
  TORCH_CHECK(weight_fp8.size(1) == kHiddenSize, "weight_fp8 K dimension must be 2048");
  TORCH_CHECK(weight_fp8.size(0) >= 2 * intermediate_size, "weight_fp8 must contain gate rows followed by up rows");
  TORCH_CHECK(scales.dim() == 2, "scales must be [2 * intermediate, 16]");
  TORCH_CHECK(scales.size(0) == weight_fp8.size(0), "scales N dimension must match weight_fp8");
  TORCH_CHECK(scales.size(1) == kFp8ScaleBlocks, "scales K-block dimension must be 16");
  TORCH_CHECK(out.numel() == intermediate_size, "out must have intermediate_size elements");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
}

}  // namespace

void gate_up_swiglu_half2_vec4_f32acc_cg_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_vec4_f32acc_cg_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_gate_up_e4m3_block128_k2048(
    torch::Tensor weight,
    torch::Tensor weight_fp8,
    torch::Tensor scales) {
  check_quantize_inputs(weight, weight_fp8, scales);
  int n_rows = static_cast<int>(weight.size(0));
  dim3 grid(n_rows, kFp8ScaleBlocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  quantize_e4m3_block128_k2048_kernel<<<grid, 32, 0, stream>>>(
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      n_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_fp8_e4m3_block128_k2048(
    torch::Tensor x,
    torch::Tensor weight_fp8,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  check_fp8_inputs(x, weight_fp8, scales, out, intermediate_size, rows_per_block);
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_fp8_e4m3_block128_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_fp8.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gate_up_swiglu_half2_vec4_f32acc_cg_k2048",
      &gate_up_swiglu_half2_vec4_f32acc_cg_k2048,
      "SOTA fp32-accum decode gate/up SwiGLU kernel for Qwen3-VL hidden_size=2048");
  m.def(
      "quantize_gate_up_e4m3_block128_k2048",
      &quantize_gate_up_e4m3_block128_k2048,
      "Online e4m3 block-128 quantization for Qwen3-VL decode gate/up weight");
  m.def(
      "gate_up_swiglu_fp8_e4m3_block128_k2048",
      &gate_up_swiglu_fp8_e4m3_block128_k2048,
      "FP8 e4m3 block-128 decode gate/up SwiGLU kernel for Qwen3-VL hidden_size=2048");
}
