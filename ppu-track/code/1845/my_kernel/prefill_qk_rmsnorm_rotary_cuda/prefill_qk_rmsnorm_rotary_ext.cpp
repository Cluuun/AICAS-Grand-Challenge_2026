#include <torch/extension.h>

void run_prefill_qk_rmsnorm_rotary_cuda(
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
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k must be [1,S,H,D]");
  TORCH_CHECK(q.size(0) == 1 && k.size(0) == 1, "batch must be 1");
  TORCH_CHECK(q.size(1) == k.size(1), "q/k sequence length mismatch");
  TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128, "head_dim must be 128");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "cos/sin must be [S,D]");
  TORCH_CHECK(cos.size(0) == q.size(1) && cos.size(1) == 128, "cos shape mismatch");
  TORCH_CHECK(sin.size(0) == q.size(1) && sin.size(1) == 128, "sin shape mismatch");
  TORCH_CHECK(q_weight.numel() == 128 && k_weight.numel() == 128, "weights must have 128 elements");
  TORCH_CHECK(q_out.dim() == 4 && k_out.dim() == 4, "outputs must be [1,H,S,D]");
  TORCH_CHECK(q_out.size(0) == 1 && q_out.size(1) == q.size(2) &&
              q_out.size(2) == q.size(1) && q_out.size(3) == 128,
              "q_out shape mismatch");
  TORCH_CHECK(k_out.size(0) == 1 && k_out.size(1) == k.size(2) &&
              k_out.size(2) == k.size(1) && k_out.size(3) == 128,
              "k_out shape mismatch");
  TORCH_CHECK(q_out.is_contiguous() && k_out.is_contiguous(), "outputs must be contiguous");
}

void run_prefill_qk_rmsnorm_rotary_into(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    torch::Tensor q_out,
    torch::Tensor k_out) {
  if (!cos.is_contiguous()) {
    cos = cos.contiguous();
  }
  if (!sin.is_contiguous()) {
    sin = sin.contiguous();
  }
  q_weight = q_weight.reshape({128}).contiguous();
  k_weight = k_weight.reshape({128}).contiguous();
  check_inputs(q, k, cos, sin, q_weight, k_weight, q_out, k_out);
  run_prefill_qk_rmsnorm_rotary_cuda(q, k, cos, sin, q_weight, k_weight, eps, q_out, k_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_prefill_qk_rmsnorm_rotary_into", &run_prefill_qk_rmsnorm_rotary_into,
        "prefill Q/K RMSNorm + RoPE into preallocated outputs");
}
