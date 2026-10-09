#include <torch/extension.h>

void run_bf16_verify_attn_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor out);

static void check_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& out) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kBFloat16, "query must be bf16");
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16, "key must be bf16");
  TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bf16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  TORCH_CHECK(query.dim() == 4, "query must have shape [1,16,Q,128]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [1,8,S,128]");
  TORCH_CHECK(value.dim() == 4, "value must have shape [1,8,S,128]");
  TORCH_CHECK(out.dim() == 4, "out must have shape [1,Q,16,128]");
  TORCH_CHECK(query.size(0) == 1, "query batch must be 1");
  TORCH_CHECK(query.size(1) == 16, "query heads must be 16");
  TORCH_CHECK(query.size(3) == 128, "query head_dim must be 128");
  TORCH_CHECK(key.size(0) == 1, "key batch must be 1");
  TORCH_CHECK(key.size(1) == 8, "key heads must be 8");
  TORCH_CHECK(key.size(3) == 128, "key head_dim must be 128");
  TORCH_CHECK(value.sizes() == key.sizes(), "value shape must match key shape");
  TORCH_CHECK(out.size(0) == 1, "out batch must be 1");
  TORCH_CHECK(out.size(1) == query.size(2), "out q_len mismatch");
  TORCH_CHECK(out.size(2) == 16, "out heads must be 16");
  TORCH_CHECK(out.size(3) == 128, "out head_dim must be 128");
  TORCH_CHECK(query.size(2) >= 1 && query.size(2) <= 16, "q_len must be in [1,16]");
  TORCH_CHECK(key.size(2) >= query.size(2), "kv_len must be >= q_len");
  TORCH_CHECK(query.device() == key.device(), "query/key device mismatch");
  TORCH_CHECK(query.device() == value.device(), "query/value device mismatch");
  TORCH_CHECK(query.device() == out.device(), "query/out device mismatch");
}

torch::Tensor run_bf16_verify_attn(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
  auto q_len = query.size(2);
  auto out = torch::empty(
      {1, q_len, 16, 128},
      torch::TensorOptions().device(query.device()).dtype(query.dtype()));
  check_inputs(query, key, value, out);
  run_bf16_verify_attn_cuda(query, key, value, out);
  return out;
}

void run_bf16_verify_attn_into(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor out) {
  check_inputs(query, key, value, out);
  run_bf16_verify_attn_cuda(query, key, value, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_bf16_verify_attn", &run_bf16_verify_attn, "bf16 verifier attention");
  m.def("run_bf16_verify_attn_into", &run_bf16_verify_attn_into, "bf16 verifier attention into");
}
