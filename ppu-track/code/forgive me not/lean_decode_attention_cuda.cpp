#include <torch/extension.h>

torch::Tensor lean_decode_attention_cuda_forward(
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor cache_position,
    torch::Tensor partial_acc,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    double sm_scale,
    int64_t max_splits,
    int64_t tokens_per_split);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &lean_decode_attention_cuda_forward,
        "SM80-specialized lean decode attention forward");
}
