#include <torch/extension.h>

torch::Tensor flashdecodeffn_mlp_cuda(torch::Tensor x,
                                      torch::Tensor w1,
                                      torch::Tensor w3,
                                      torch::Tensor w2,
                                      int64_t split_k);
torch::Tensor flashdecodeffn_mlp_packed_cuda(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2);
void flashdecodeffn_mlp_packed_into_cuda(torch::Tensor x,
                                         torch::Tensor w13,
                                         torch::Tensor w2,
                                         torch::Tensor out13,
                                         torch::Tensor gated,
                                         torch::Tensor out);
torch::Tensor flashdecodeffn_pack_w13_cuda(torch::Tensor w1,
                                           torch::Tensor w3);
torch::Tensor flashdecodeffn_pack_w2_cuda(torch::Tensor w2);
torch::Tensor flashdecodeffn_vision_mlp_packed_cuda(torch::Tensor x,
                                                    torch::Tensor w1,
                                                    torch::Tensor b1,
                                                    torch::Tensor w2,
                                                    torch::Tensor b2);
void flashdecodeffn_vision_mlp_packed_into_cuda(torch::Tensor x,
                                                torch::Tensor w1,
                                                torch::Tensor b1,
                                                torch::Tensor w2,
                                                torch::Tensor b2,
                                                torch::Tensor fc1,
                                                torch::Tensor out);
torch::Tensor flashdecodeffn_pack_vision_w1_cuda(torch::Tensor w1);
torch::Tensor flashdecodeffn_pack_vision_w2_cuda(torch::Tensor w2);
torch::Tensor flashdecodeffn_text_decode_v2_cuda(torch::Tensor x,
                                                 torch::Tensor w13,
                                                 torch::Tensor w2);
torch::Tensor flashdecodeffn_text_prefill_v2_cuda(torch::Tensor x,
                                                  torch::Tensor w13,
                                                  torch::Tensor w2);
torch::Tensor flashdecodeffn_text_forward_v2_cuda(torch::Tensor x,
                                                  torch::Tensor w13,
                                                  torch::Tensor w2);
torch::Tensor flashdecodeffn_vision_forward_v2_cuda(torch::Tensor x,
                                                    torch::Tensor w1,
                                                    torch::Tensor b1,
                                                    torch::Tensor w2,
                                                    torch::Tensor b2);
int64_t flashdecodeffn_get_text_decode_v2_algo_cuda();
int64_t flashdecodeffn_get_text_prefill_v2_algo_cuda();
int64_t flashdecodeffn_get_vision_v2_algo_cuda();

torch::Tensor flashdecodeffn_mlp(torch::Tensor x,
                                 torch::Tensor w1,
                                 torch::Tensor w3,
                                 torch::Tensor w2,
                                 int64_t split_k = 64) {
  return flashdecodeffn_mlp_cuda(x, w1, w3, w2, split_k);
}

torch::Tensor flashdecodeffn_mlp_packed(torch::Tensor x,
                                        torch::Tensor w13,
                                        torch::Tensor w2) {
  return flashdecodeffn_mlp_packed_cuda(x, w13, w2);
}

torch::Tensor flashdecodeffn_mlp_packed_into(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2,
                                             torch::Tensor out13,
                                             torch::Tensor gated,
                                             torch::Tensor out) {
  flashdecodeffn_mlp_packed_into_cuda(x, w13, w2, out13, gated, out);
  return out;
}

torch::Tensor flashdecodeffn_pack_w13(torch::Tensor w1,
                                      torch::Tensor w3) {
  return flashdecodeffn_pack_w13_cuda(w1, w3);
}

torch::Tensor flashdecodeffn_pack_w2(torch::Tensor w2) {
  return flashdecodeffn_pack_w2_cuda(w2);
}

torch::Tensor flashdecodeffn_vision_mlp_packed(torch::Tensor x,
                                               torch::Tensor w1,
                                               torch::Tensor b1,
                                               torch::Tensor w2,
                                               torch::Tensor b2) {
  return flashdecodeffn_vision_mlp_packed_cuda(x, w1, b1, w2, b2);
}

torch::Tensor flashdecodeffn_vision_mlp_packed_into(torch::Tensor x,
                                                    torch::Tensor w1,
                                                    torch::Tensor b1,
                                                    torch::Tensor w2,
                                                    torch::Tensor b2,
                                                    torch::Tensor fc1,
                                                    torch::Tensor out) {
  flashdecodeffn_vision_mlp_packed_into_cuda(x, w1, b1, w2, b2, fc1, out);
  return out;
}

torch::Tensor flashdecodeffn_pack_vision_w1(torch::Tensor w1) {
  return flashdecodeffn_pack_vision_w1_cuda(w1);
}

torch::Tensor flashdecodeffn_pack_vision_w2(torch::Tensor w2) {
  return flashdecodeffn_pack_vision_w2_cuda(w2);
}

torch::Tensor flashdecodeffn_text_decode_v2(torch::Tensor x,
                                            torch::Tensor w13,
                                            torch::Tensor w2) {
  return flashdecodeffn_text_decode_v2_cuda(x, w13, w2);
}

torch::Tensor flashdecodeffn_text_prefill_v2(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2) {
  return flashdecodeffn_text_prefill_v2_cuda(x, w13, w2);
}

torch::Tensor flashdecodeffn_text_forward_v2(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2) {
  return flashdecodeffn_text_forward_v2_cuda(x, w13, w2);
}

torch::Tensor flashdecodeffn_vision_forward_v2(torch::Tensor x,
                                               torch::Tensor w1,
                                               torch::Tensor b1,
                                               torch::Tensor w2,
                                               torch::Tensor b2) {
  return flashdecodeffn_vision_forward_v2_cuda(x, w1, b1, w2, b2);
}

int64_t flashdecodeffn_get_text_decode_v2_algo() {
  return flashdecodeffn_get_text_decode_v2_algo_cuda();
}

int64_t flashdecodeffn_get_text_prefill_v2_algo() {
  return flashdecodeffn_get_text_prefill_v2_algo_cuda();
}

int64_t flashdecodeffn_get_vision_v2_algo() {
  return flashdecodeffn_get_vision_v2_algo_cuda();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flashdecodeffn_mlp,
        "FlashDecodeFFN MLP forward (bf16, sm80 mma PTX)",
        pybind11::arg("x"),
        pybind11::arg("w1"),
        pybind11::arg("w3"),
        pybind11::arg("w2"),
        pybind11::arg("split_k") = 64);
  m.def("forward_packed", &flashdecodeffn_mlp_packed,
        "FlashDecodeFFN MLP forward using packed W13 [H, 2I] and W2 [I, H]",
        pybind11::arg("x"),
        pybind11::arg("w13"),
        pybind11::arg("w2"));
  m.def("forward_packed_into", &flashdecodeffn_mlp_packed_into,
        "FlashDecodeFFN MLP forward into preallocated buffers",
        pybind11::arg("x"),
        pybind11::arg("w13"),
        pybind11::arg("w2"),
        pybind11::arg("out13"),
        pybind11::arg("gated"),
        pybind11::arg("out"));
  m.def("pack_w13", &flashdecodeffn_pack_w13,
        "Pack W1/W3 into W13 [H, 2I]",
        pybind11::arg("w1"),
        pybind11::arg("w3"));
  m.def("pack_w2", &flashdecodeffn_pack_w2,
        "Pack W2 into W2 [I, H]",
        pybind11::arg("w2"));
  m.def("forward_vision_packed", &flashdecodeffn_vision_mlp_packed,
        "FlashDecodeFFN VisionMLP forward using packed W1 [H, I] and W2 [I, H]",
        pybind11::arg("x"),
        pybind11::arg("w1"),
        pybind11::arg("b1"),
        pybind11::arg("w2"),
        pybind11::arg("b2"));
  m.def("forward_vision_packed_into", &flashdecodeffn_vision_mlp_packed_into,
        "FlashDecodeFFN VisionMLP forward into preallocated buffers",
        pybind11::arg("x"),
        pybind11::arg("w1"),
        pybind11::arg("b1"),
        pybind11::arg("w2"),
        pybind11::arg("b2"),
        pybind11::arg("fc1"),
        pybind11::arg("out"));
  m.def("pack_vision_w1", &flashdecodeffn_pack_vision_w1,
        "Pack Vision W1 into [H, I]",
        pybind11::arg("w1"));
  m.def("pack_vision_w2", &flashdecodeffn_pack_vision_w2,
        "Pack Vision W2 into [I, H]",
        pybind11::arg("w2"));
  m.def("forward_text_decode_v2", &flashdecodeffn_text_decode_v2,
        "FlashDecodeFFN text decode v2 (N=1,H=2048,I=6144)",
        pybind11::arg("x"),
        pybind11::arg("w13"),
        pybind11::arg("w2"));
  m.def("forward_text_prefill_v2", &flashdecodeffn_text_prefill_v2,
        "FlashDecodeFFN text prefill v2 (N=690,H=2048,I=6144)",
        pybind11::arg("x"),
        pybind11::arg("w13"),
        pybind11::arg("w2"));
  m.def("forward_text_v2", &flashdecodeffn_text_forward_v2,
        "FlashDecodeFFN text v2 exact-shape dispatch",
        pybind11::arg("x"),
        pybind11::arg("w13"),
        pybind11::arg("w2"));
  m.def("forward_vision_v2", &flashdecodeffn_vision_forward_v2,
        "FlashDecodeFFN vision v2 exact-shape dispatch",
        pybind11::arg("x"),
        pybind11::arg("w1"),
        pybind11::arg("b1"),
        pybind11::arg("w2"),
        pybind11::arg("b2"));
  m.def("pack_text_w13_v2", &flashdecodeffn_pack_w13,
        "Pack text W13 for v2 [H, 2I]",
        pybind11::arg("w1"),
        pybind11::arg("w3"));
  m.def("pack_text_w2_v2", &flashdecodeffn_pack_w2,
        "Pack text W2 for v2 [I, H]",
        pybind11::arg("w2"));
  m.def("pack_vision_w1_v2", &flashdecodeffn_pack_vision_w1,
        "Pack vision W1 for v2 [H, I]",
        pybind11::arg("w1"));
  m.def("pack_vision_w2_v2", &flashdecodeffn_pack_vision_w2,
        "Pack vision W2 for v2 [I, H]",
        pybind11::arg("w2"));
  m.def("get_text_decode_v2_algo", &flashdecodeffn_get_text_decode_v2_algo,
        "Get selected algo for text decode v2");
  m.def("get_text_prefill_v2_algo", &flashdecodeffn_get_text_prefill_v2_algo,
        "Get selected algo for text prefill v2");
  m.def("get_vision_v2_algo", &flashdecodeffn_get_vision_v2_algo,
        "Get selected algo for vision v2");
}
