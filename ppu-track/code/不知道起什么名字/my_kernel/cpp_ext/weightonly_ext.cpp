#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass_kernels/cutlass_preprocessors.h"
#include "weightOnlyBatchedGemv/kernelLauncher.h"

namespace wo = acext::kernels::weight_only;
namespace ck = acext::kernels::cutlass_kernels;

torch::Tensor preprocess_int8_weight(torch::Tensor weight, int64_t arch) {
  TORCH_CHECK(!weight.is_cuda(), "weight must be a CPU tensor");
  TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [k, n]");
  auto weight_c = weight.contiguous();
  auto out = torch::empty_like(weight_c);
  std::vector<size_t> shape{
      static_cast<size_t>(weight_c.size(0)),
      static_cast<size_t>(weight_c.size(1)),
  };
  ck::preprocess_weights_for_mixed_gemm(
      reinterpret_cast<int8_t*>(out.data_ptr<int8_t>()),
      reinterpret_cast<const int8_t*>(weight_c.data_ptr<int8_t>()),
      shape,
      ck::QuantType::INT8_WEIGHT_ONLY,
      false,
      false,
      static_cast<int>(arch));
  return out;
}

torch::Tensor weightonly_gemv(torch::Tensor input, torch::Tensor weight, torch::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "only fp16 input is supported");
  TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be fp16");
  TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [k, n]");

  auto sizes = input.sizes();
  int64_t m64 = 1;
  for (int i = 0; i < input.dim() - 1; ++i) {
    m64 *= sizes[i];
  }
  const int64_t k64 = sizes[input.dim() - 1];
  TORCH_CHECK(weight.size(0) == k64, "weight first dim must equal input hidden dim");
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(scales.numel() == n64, "scales must have n elements");
  TORCH_CHECK(m64 > 0 && m64 <= 4, "WeightonlyBatchedGemv fast path only supports small m<=4");
  TORCH_CHECK(k64 <= std::numeric_limits<int>::max() && n64 <= std::numeric_limits<int>::max(),
              "shape too large");

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto flat_input = input.contiguous().view({m64, k64});
  auto weight_c = weight.contiguous();
  auto scales_c = scales.contiguous();
  auto output = torch::empty({m64, n64}, input.options());

  wo::Params params(
      flat_input.data_ptr(),
      nullptr,
      weight_c.data_ptr(),
      scales_c.data_ptr(),
      nullptr,
      nullptr,
      output.data_ptr(),
      1.0f,
      static_cast<int>(m64),
      static_cast<int>(n64),
      static_cast<int>(k64),
      0,
      wo::KernelType::FP16Int8PerChannel);
  wo::kernel_launcher(80, params, at::cuda::getCurrentCUDAStream());

  std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
  out_sizes.back() = n64;
  return output.view(out_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_int8_weight", &preprocess_int8_weight, "acext preprocess INT8 weight-only weight");
  m.def("weightonly_gemv", &weightonly_gemv, "acext WeightOnlyBatchedGemv FP16xINT8");
}
