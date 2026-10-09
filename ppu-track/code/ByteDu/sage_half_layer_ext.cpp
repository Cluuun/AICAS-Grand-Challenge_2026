#include <torch/extension.h>

void sage_half_layer_rstd_cuda(torch::Tensor residual, torch::Tensor rstd, double eps);
void sage_half_layer_gate_up_q8_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor rstd,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rstd", &sage_half_layer_rstd_cuda, "Sage half-layer residual RMS rstd");
    m.def("gate_up_q8", &sage_half_layer_gate_up_q8_cuda, "Sage half-layer pre-scaled Q8 gate_up+SwiGLU");
}
