#include <torch/extension.h>

void copy_full_prompt_kv_cuda(
    torch::Tensor key_src,
    torch::Tensor value_src,
    torch::Tensor key_dst,
    torch::Tensor value_dst,
    int64_t seq_len);

static void check_inputs(
    const torch::Tensor& key_src,
    const torch::Tensor& value_src,
    const torch::Tensor& key_dst,
    const torch::Tensor& value_dst,
    int64_t seq_len) {
  TORCH_CHECK(key_src.is_cuda(), "key_src must be CUDA");
  TORCH_CHECK(value_src.is_cuda(), "value_src must be CUDA");
  TORCH_CHECK(key_dst.is_cuda(), "key_dst must be CUDA");
  TORCH_CHECK(value_dst.is_cuda(), "value_dst must be CUDA");
  TORCH_CHECK(key_src.scalar_type() == torch::kBFloat16, "key_src must be bf16");
  TORCH_CHECK(value_src.scalar_type() == torch::kBFloat16, "value_src must be bf16");
  TORCH_CHECK(key_dst.scalar_type() == torch::kBFloat16, "key_dst must be bf16");
  TORCH_CHECK(value_dst.scalar_type() == torch::kBFloat16, "value_dst must be bf16");
  TORCH_CHECK(key_src.is_contiguous(), "key_src must be contiguous [L,B,H,S,D]");
  TORCH_CHECK(value_src.is_contiguous(), "value_src must be contiguous [L,B,H,S,D]");
  TORCH_CHECK(key_dst.is_contiguous(), "key_dst must be contiguous [L,B,H,M,D]");
  TORCH_CHECK(value_dst.is_contiguous(), "value_dst must be contiguous [L,B,H,M,D]");
  TORCH_CHECK(key_src.dim() == 5, "key_src must be [L,B,H,S,D]");
  TORCH_CHECK(value_src.sizes() == key_src.sizes(), "value_src shape must match key_src");
  TORCH_CHECK(key_dst.dim() == 5, "key_dst must be [L,B,H,M,D]");
  TORCH_CHECK(value_dst.sizes() == key_dst.sizes(), "value_dst shape must match key_dst");
  TORCH_CHECK(key_src.size(0) == key_dst.size(0), "layer count mismatch");
  TORCH_CHECK(key_src.size(1) == key_dst.size(1), "batch size mismatch");
  TORCH_CHECK(key_src.size(2) == key_dst.size(2), "head count mismatch");
  TORCH_CHECK(key_src.size(4) == key_dst.size(4), "head dim mismatch");
  TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");
  TORCH_CHECK(seq_len <= key_src.size(3), "seq_len exceeds source length");
  TORCH_CHECK(seq_len <= key_dst.size(3), "seq_len exceeds destination length");
}

void copy_full_prompt_kv(
    torch::Tensor key_src,
    torch::Tensor value_src,
    torch::Tensor key_dst,
    torch::Tensor value_dst,
    int64_t seq_len) {
  check_inputs(key_src, value_src, key_dst, value_dst, seq_len);
  copy_full_prompt_kv_cuda(key_src, value_src, key_dst, value_dst, seq_len);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy_full_prompt_kv", &copy_full_prompt_kv, "Fused full-prompt KV copy");
}
