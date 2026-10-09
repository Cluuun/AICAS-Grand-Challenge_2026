#include <torch/extension.h>

torch::Tensor decode_attention_front_cuda_forward(
    torch::Tensor q_proj,
    torch::Tensor k_proj,
    torch::Tensor v_proj,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cache_position,
    double q_eps,
    double k_eps,
    torch::Tensor q_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &decode_attention_front_cuda_forward,
        "SM80-specialized decode qk rotary + cache update forward");
}
