#include <torch/extension.h>

void run_prefill_rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps,
    torch::Tensor out);

static void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bf16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be [M,N]");
  TORCH_CHECK(out.sizes() == x.sizes(), "out shape must match x");
  TORCH_CHECK(weight.numel() == x.size(1), "weight length must match hidden size");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) <= 4096, "hidden size must be in (0,4096]");
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
  x = x.contiguous();
  weight = weight.reshape({x.size(1)}).contiguous();
  auto out = torch::empty_like(x);
  check_inputs(x, weight, out);
  run_prefill_rmsnorm_cuda(x, weight, eps, out);
  return out;
}

void rmsnorm_into(torch::Tensor x, torch::Tensor weight, double eps, torch::Tensor out) {
  x = x.contiguous();
  weight = weight.reshape({x.size(1)}).contiguous();
  check_inputs(x, weight, out);
  run_prefill_rmsnorm_cuda(x, weight, eps, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "Prefill RMSNorm CUDA");
  m.def("rmsnorm_into", &rmsnorm_into, "Prefill RMSNorm CUDA into");
}
