#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "weightOnlyBatchedGemv/kernelLauncher.h"

namespace wo = acext::kernels::weight_only;

void run_custom_int4_gemv(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor bias,
    torch::Tensor out,
    int64_t group_size,
    bool has_zeros,
    bool has_bias);
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
    bool has_bias);

static int current_arch() {
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  return prop->major * 10 + prop->minor;
}

static void check_common(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const torch::Tensor& out,
    int64_t group_size) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(zeros.is_cuda(), "zeros must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(
      weight.scalar_type() == torch::kUInt8 || weight.scalar_type() == torch::kInt8,
      "weight must be int8/uint8 packed int4");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(zeros.is_contiguous(), "zeros must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be [M,K]");
  TORCH_CHECK(scales.dim() == 2, "scales must be [K/groupsize,N]");
  TORCH_CHECK(zeros.dim() == 2, "zeros must be [K/groupsize,N]");
  TORCH_CHECK(out.dim() == 2, "out must be [M,N]");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(x.size(1) % group_size == 0, "K must be divisible by group_size");
  TORCH_CHECK(scales.size(0) == x.size(1) / group_size, "scales group count mismatch");
  TORCH_CHECK(zeros.size(0) == scales.size(0), "zeros group count mismatch");
  TORCH_CHECK(zeros.size(1) == scales.size(1), "zeros N mismatch");
  TORCH_CHECK(out.size(0) == x.size(0), "out M mismatch");
  TORCH_CHECK(out.size(1) == scales.size(1), "out N mismatch");
}

void linear_groupwise_into(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor out,
    int64_t group_size,
    double alpha,
    bool apply_alpha_in_advance) {
  x = x.contiguous();
  weight = weight.contiguous();
  scales = scales.contiguous();
  zeros = zeros.contiguous();
  check_common(x, weight, scales, zeros, out, group_size);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  wo::Params params(
      x.data_ptr(),
      nullptr,
      weight.data_ptr(),
      scales.data_ptr(),
      zeros.data_ptr(),
      nullptr,
      out.data_ptr(),
      static_cast<float>(alpha),
      static_cast<int>(x.size(0)),
      static_cast<int>(scales.size(1)),
      static_cast<int>(x.size(1)),
      static_cast<int>(group_size),
      wo::KernelType::BF16Int4Groupwise,
      apply_alpha_in_advance);
  wo::kernel_launcher(current_arch(), params, at::cuda::getCurrentCUDAStream());
}

torch::Tensor linear_groupwise(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size,
    double alpha,
    bool apply_alpha_in_advance) {
  x = x.contiguous();
  auto out = torch::empty({x.size(0), scales.size(1)}, x.options());
  linear_groupwise_into(
      x, weight, scales, zeros, out, group_size, alpha, apply_alpha_in_advance);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_groupwise", &linear_groupwise, "BF16 x INT4 groupwise GEMV probe");
  m.def("linear_groupwise_into", &linear_groupwise_into, "BF16 x INT4 groupwise GEMV probe into");
  m.def(
      "custom_linear_into",
      &run_custom_int4_gemv,
      "Custom bs=1 BF16 x signed INT4 groupwise GEMV into");
  m.def(
      "custom_linear_splitk_into",
      &run_custom_int4_gemv_splitk,
      "Custom bs=1 BF16 x signed INT4 groupwise GEMV Split-K into");
  m.def(
      "custom_linear",
      [](torch::Tensor x,
         torch::Tensor packed_weight,
         torch::Tensor scales,
         torch::Tensor zeros,
         torch::Tensor bias,
         int64_t group_size,
         bool has_zeros,
         bool has_bias) {
        x = x.contiguous();
        auto out = torch::empty({x.size(0), scales.size(1)}, x.options());
        run_custom_int4_gemv(
            x,
            packed_weight,
            scales,
            zeros,
            bias,
            out,
            group_size,
            has_zeros,
            has_bias);
        return out;
      },
      "Custom bs=1 BF16 x signed INT4 groupwise GEMV");
}
