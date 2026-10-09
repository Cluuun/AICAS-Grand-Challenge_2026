#include <torch/extension.h>
#include <vector>

void run_bf16_qk_rmsnorm_rotary_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    torch::Tensor q_out,
    torch::Tensor k_out);

static void check_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& q_out,
    const torch::Tensor& k_out) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q/k must be CUDA");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos/sin must be CUDA");
  TORCH_CHECK(q_weight.is_cuda() && k_weight.is_cuda(), "weights must be CUDA");
  TORCH_CHECK(q_out.is_cuda() && k_out.is_cuda(), "outputs must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
  TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bf16");
  TORCH_CHECK(cos.scalar_type() == torch::kBFloat16, "cos must be bf16");
  TORCH_CHECK(sin.scalar_type() == torch::kBFloat16, "sin must be bf16");
  TORCH_CHECK(q_weight.scalar_type() == torch::kBFloat16, "q_weight must be bf16");
  TORCH_CHECK(k_weight.scalar_type() == torch::kBFloat16, "k_weight must be bf16");
  TORCH_CHECK(q_out.scalar_type() == torch::kBFloat16, "q_out must be bf16");
  TORCH_CHECK(k_out.scalar_type() == torch::kBFloat16, "k_out must be bf16");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q/k must be contiguous");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos/sin must be contiguous");
  TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(q_out.is_contiguous() && k_out.is_contiguous(), "outputs must be contiguous");

  TORCH_CHECK(q.dim() == 2 && q.size(1) == 128, "q must be [Q*16,128]");
  TORCH_CHECK(k.dim() == 2 && k.size(1) == 128, "k must be [Q*8,128]");
  TORCH_CHECK(q.size(0) % 16 == 0, "q rows must be divisible by 16");
  TORCH_CHECK(k.size(0) % 8 == 0, "k rows must be divisible by 8");
  const auto q_len = q.size(0) / 16;
  TORCH_CHECK(k.size(0) / 8 == q_len, "q/k token counts must match");
  TORCH_CHECK(q_len >= 1 && q_len <= 16, "q_len must be in [1,16]");
  TORCH_CHECK(cos.numel() == q_len * 128, "cos must have Q*128 elements");
  TORCH_CHECK(sin.numel() == q_len * 128, "sin must have Q*128 elements");
  TORCH_CHECK(q_weight.numel() == 128 && k_weight.numel() == 128, "weights must have 128 elements");
  TORCH_CHECK(q_out.dim() == 4, "q_out must be [1,16,Q,128]");
  TORCH_CHECK(k_out.dim() == 4, "k_out must be [1,8,Q,128]");
  TORCH_CHECK(q_out.size(0) == 1 && q_out.size(1) == 16 && q_out.size(2) == q_len && q_out.size(3) == 128,
              "q_out shape mismatch");
  TORCH_CHECK(k_out.size(0) == 1 && k_out.size(1) == 8 && k_out.size(2) == q_len && k_out.size(3) == 128,
              "k_out shape mismatch");
  TORCH_CHECK(q.device() == k.device(), "q/k device mismatch");
  TORCH_CHECK(q.device() == cos.device(), "q/cos device mismatch");
  TORCH_CHECK(q.device() == sin.device(), "q/sin device mismatch");
  TORCH_CHECK(q.device() == q_weight.device(), "q/q_weight device mismatch");
  TORCH_CHECK(q.device() == k_weight.device(), "q/k_weight device mismatch");
  TORCH_CHECK(q.device() == q_out.device(), "q/q_out device mismatch");
  TORCH_CHECK(q.device() == k_out.device(), "q/k_out device mismatch");
}

std::vector<torch::Tensor> run_bf16_qk_rmsnorm_rotary(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps) {
  q = q.contiguous();
  k = k.contiguous();
  const auto q_len = q.size(0) / 16;
  cos = cos.reshape({q_len, 128}).contiguous();
  sin = sin.reshape({q_len, 128}).contiguous();
  q_weight = q_weight.reshape({128}).contiguous();
  k_weight = k_weight.reshape({128}).contiguous();
  auto q_out = torch::empty({1, 16, q_len, 128}, q.options());
  auto k_out = torch::empty({1, 8, q_len, 128}, k.options());
  check_inputs(q, k, cos, sin, q_weight, k_weight, q_out, k_out);
  run_bf16_qk_rmsnorm_rotary_cuda(q, k, cos, sin, q_weight, k_weight, eps, q_out, k_out);
  return {q_out, k_out};
}

void run_bf16_qk_rmsnorm_rotary_into(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    torch::Tensor q_out,
    torch::Tensor k_out) {
  q = q.contiguous();
  k = k.contiguous();
  const auto q_len = q.size(0) / 16;
  cos = cos.reshape({q_len, 128}).contiguous();
  sin = sin.reshape({q_len, 128}).contiguous();
  q_weight = q_weight.reshape({128}).contiguous();
  k_weight = k_weight.reshape({128}).contiguous();
  check_inputs(q, k, cos, sin, q_weight, k_weight, q_out, k_out);
  run_bf16_qk_rmsnorm_rotary_cuda(q, k, cos, sin, q_weight, k_weight, eps, q_out, k_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_bf16_qk_rmsnorm_rotary", &run_bf16_qk_rmsnorm_rotary,
        "bf16 small-Q Q/K RMSNorm + rotary");
  m.def("run_bf16_qk_rmsnorm_rotary_into", &run_bf16_qk_rmsnorm_rotary_into,
        "bf16 small-Q Q/K RMSNorm + rotary into");
}
