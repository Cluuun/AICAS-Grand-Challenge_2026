#include <torch/extension.h>

#include <string>

torch::Tensor ffn_cublaslt_text_forward_cuda(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2);
torch::Tensor ffn_cublaslt_vision_forward_cuda(torch::Tensor x,
                                               torch::Tensor w1,
                                               torch::Tensor b1,
                                               torch::Tensor w2,
                                               torch::Tensor b2);
std::string ffn_cublaslt_get_last_report_cuda();

torch::Tensor ffn_pack_text_w13(torch::Tensor w1, torch::Tensor w3) {
  TORCH_CHECK(w1.is_cuda() && w3.is_cuda(), "w1/w3 must be CUDA tensors");
  TORCH_CHECK(w1.dtype() == torch::kBFloat16 && w3.dtype() == torch::kBFloat16,
              "w1/w3 must be bfloat16");
  TORCH_CHECK(w1.dim() == 2 && w3.dim() == 2, "w1/w3 must be rank-2");
  TORCH_CHECK(w1.sizes() == w3.sizes(), "w1 and w3 shape mismatch");
  return torch::cat({w1.contiguous(), w3.contiguous()}, 0).transpose(0, 1).contiguous();
}

torch::Tensor ffn_pack_text_w2(torch::Tensor w2) {
  TORCH_CHECK(w2.is_cuda(), "w2 must be CUDA tensor");
  TORCH_CHECK(w2.dtype() == torch::kBFloat16, "w2 must be bfloat16");
  TORCH_CHECK(w2.dim() == 2, "w2 must be rank-2");
  return w2.contiguous().transpose(0, 1).contiguous();
}

torch::Tensor ffn_pack_vision_w1(torch::Tensor w1) {
  TORCH_CHECK(w1.is_cuda(), "vision w1 must be CUDA tensor");
  TORCH_CHECK(w1.dtype() == torch::kBFloat16, "vision w1 must be bfloat16");
  TORCH_CHECK(w1.dim() == 2, "vision w1 must be rank-2");
  return w1.contiguous().transpose(0, 1).contiguous();
}

torch::Tensor ffn_pack_vision_w2(torch::Tensor w2) {
  TORCH_CHECK(w2.is_cuda(), "vision w2 must be CUDA tensor");
  TORCH_CHECK(w2.dtype() == torch::kBFloat16, "vision w2 must be bfloat16");
  TORCH_CHECK(w2.dim() == 2, "vision w2 must be rank-2");
  return w2.contiguous().transpose(0, 1).contiguous();
}

torch::Tensor ffn_cublaslt_text_forward(torch::Tensor x,
                                        torch::Tensor w13,
                                        torch::Tensor w2) {
  return ffn_cublaslt_text_forward_cuda(x, w13, w2);
}

torch::Tensor ffn_cublaslt_vision_forward(torch::Tensor x,
                                          torch::Tensor w1,
                                          torch::Tensor b1,
                                          torch::Tensor w2,
                                          torch::Tensor b2) {
  return ffn_cublaslt_vision_forward_cuda(x, w1, b1, w2, b2);
}

std::string ffn_cublaslt_get_last_report() {
  return ffn_cublaslt_get_last_report_cuda();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_text_w13", &ffn_pack_text_w13, "Pack text W13 [H,2I]");
  m.def("pack_text_w2", &ffn_pack_text_w2, "Pack text W2 [I,H]");
  m.def("pack_vision_w1", &ffn_pack_vision_w1, "Pack vision W1 [H,I]");
  m.def("pack_vision_w2", &ffn_pack_vision_w2, "Pack vision W2 [I,H]");
  m.def("text_forward", &ffn_cublaslt_text_forward,
        "Text FFN forward (cuBLASLt)");
  m.def("vision_forward", &ffn_cublaslt_vision_forward,
        "Vision FFN forward (cuBLASLt)");
  m.def("get_last_report", &ffn_cublaslt_get_last_report,
        "Get last cuBLASLt heuristic report");
}

