#include <torch/extension.h>

void run_decode_linear_cublaslt(
    torch::Tensor x,
    torch::Tensor w_t,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias);
void run_decode_linear_mma(
    torch::Tensor x,
    torch::Tensor w_t,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias);

static void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& w_t,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w_t.is_cuda(), "w_t must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w_t.scalar_type() == torch::kBFloat16, "w_t must be bf16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w_t.is_contiguous(), "w_t must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be [M,K]");
  TORCH_CHECK(w_t.dim() == 2, "w_t must be [K,N]");
  TORCH_CHECK(out.dim() == 2, "out must be [M,N]");
  TORCH_CHECK(x.size(1) == w_t.size(0), "K mismatch");
  TORCH_CHECK(out.size(0) == x.size(0), "out M mismatch");
  TORCH_CHECK(out.size(1) == w_t.size(1), "out N mismatch");
  if (has_bias) {
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.scalar_type() == torch::kBFloat16, "bias must be bf16");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == w_t.size(1), "bias must be [N]");
  }
}

torch::Tensor linear(torch::Tensor x, torch::Tensor w_t, torch::Tensor bias, bool has_bias) {
  x = x.contiguous();
  w_t = w_t.contiguous();
  if (has_bias) {
    bias = bias.contiguous();
  }
  auto out = torch::empty({x.size(0), w_t.size(1)}, x.options());
  check_inputs(x, w_t, bias, out, has_bias);
  run_decode_linear_cublaslt(x, w_t, bias, out, has_bias);
  return out;
}

void linear_into(
    torch::Tensor x,
    torch::Tensor w_t,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias) {
  x = x.contiguous();
  w_t = w_t.contiguous();
  if (has_bias) {
    bias = bias.contiguous();
  }
  check_inputs(x, w_t, bias, out, has_bias);
  run_decode_linear_cublaslt(x, w_t, bias, out, has_bias);
}

torch::Tensor linear_mma(torch::Tensor x, torch::Tensor w_t, torch::Tensor bias, bool has_bias) {
  x = x.contiguous();
  w_t = w_t.contiguous();
  if (has_bias) {
    bias = bias.contiguous();
  }
  auto out = torch::empty({x.size(0), w_t.size(1)}, x.options());
  check_inputs(x, w_t, bias, out, has_bias);
  run_decode_linear_mma(x, w_t, bias, out, has_bias);
  return out;
}

void linear_mma_into(
    torch::Tensor x,
    torch::Tensor w_t,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias) {
  x = x.contiguous();
  w_t = w_t.contiguous();
  if (has_bias) {
    bias = bias.contiguous();
  }
  check_inputs(x, w_t, bias, out, has_bias);
  run_decode_linear_mma(x, w_t, bias, out, has_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear, "Decode linear cuBLASLt");
  m.def("linear_into", &linear_into, "Decode linear cuBLASLt into");
  m.def("linear_mma", &linear_mma, "Decode linear MMA");
  m.def("linear_mma_into", &linear_mma_into, "Decode linear MMA into");
}
