#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> qk_rmsnorm_rotary_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    int64_t q_heads,
    int64_t k_heads);

std::vector<torch::Tensor> rope_cache_lookup_cuda(
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor position_ids,
    int64_t mrope_h,
    int64_t mrope_w);

std::vector<torch::Tensor> qk_rmsnorm_rotary(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double eps,
    int64_t q_heads = 16,
    int64_t k_heads = 8) {
  return qk_rmsnorm_rotary_cuda(q, k, cos, sin, q_weight, k_weight, eps, q_heads, k_heads);
}

std::vector<torch::Tensor> rope_cache_lookup(
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor position_ids,
    int64_t mrope_h = 20,
    int64_t mrope_w = 20) {
  return rope_cache_lookup_cuda(cos_table, sin_table, position_ids, mrope_h, mrope_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &qk_rmsnorm_rotary, "Decode Q/K RMSNorm + Rotary forward");
  m.def("rope_cache_lookup", &rope_cache_lookup, "Decode RoPE cache lookup");
}
