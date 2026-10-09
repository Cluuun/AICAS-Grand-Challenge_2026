#include <torch/extension.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/linear.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" void launch_vision_qkv_rope_transpose_v167a(
    const void* qkv,
    const void* cos,
    const void* sin,
    void* q_out,
    void* k_out,
    void* v_out,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
);

static std::vector<at::Tensor> qkv_rope_transpose_impl(
    const at::Tensor& qkv,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t num_heads
) {
    TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos/sin must be CUDA");
    TORCH_CHECK(qkv.scalar_type() == at::kHalf, "qkv must be float16");
    TORCH_CHECK(cos.scalar_type() == at::kFloat, "cos must be float32");
    TORCH_CHECK(sin.scalar_type() == at::kFloat, "sin must be float32");
    TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos/sin must be contiguous");
    TORCH_CHECK(qkv.dim() == 2, "qkv must be [seq, 3 * hidden]");

    int64_t seq_len = qkv.size(0);
    int64_t qkv_width = qkv.size(1);
    TORCH_CHECK(qkv_width % (3 * num_heads) == 0, "qkv width must be divisible by 3*num_heads");
    int64_t head_dim = qkv_width / (3 * num_heads);
    TORCH_CHECK(cos.size(0) == seq_len && cos.size(1) == head_dim, "cos must be [seq, head_dim]");
    TORCH_CHECK(sin.size(0) == seq_len && sin.size(1) == head_dim, "sin must be [seq, head_dim]");

    auto opts = qkv.options();
    auto q_out = at::empty({1, num_heads, seq_len, head_dim}, opts);
    auto k_out = at::empty({1, num_heads, seq_len, head_dim}, opts);
    auto v_out = at::empty({1, num_heads, seq_len, head_dim}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_vision_qkv_rope_transpose_v167a(
        qkv.data_ptr<at::Half>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        q_out.data_ptr<at::Half>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        static_cast<int>(seq_len),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim),
        stream
    );
    return {q_out, k_out, v_out};
}

std::vector<at::Tensor> qkv_rope_transpose(
    at::Tensor qkv,
    at::Tensor cos,
    at::Tensor sin,
    int64_t num_heads
) {
    return qkv_rope_transpose_impl(qkv, cos, sin, num_heads);
}

std::vector<at::Tensor> vision_attn_prestage(
    const at::Tensor& hidden_states,
    const at::Tensor& ln_weight,
    const at::Tensor& ln_bias,
    double eps,
    const at::Tensor& qkv_weight,
    const c10::optional<at::Tensor>& qkv_bias,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t num_heads
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [seq, hidden]");
    TORCH_CHECK(hidden_states.scalar_type() == at::kHalf, "hidden_states must be float16");
    auto hidden = hidden_states.size(1);
    auto normed = at::layer_norm(hidden_states, {hidden}, ln_weight, ln_bias, eps, true).contiguous();
    auto qkv = at::linear(normed, qkv_weight, qkv_bias).contiguous();
    return qkv_rope_transpose_impl(qkv, cos, sin, num_heads);
}

at::Tensor vision_block_mlp_forward(
    const at::Tensor& hidden_states,
    const at::Tensor& ln_weight,
    const at::Tensor& ln_bias,
    double eps,
    const at::Tensor& fc1_weight,
    const c10::optional<at::Tensor>& fc1_bias,
    const at::Tensor& fc2_weight,
    const c10::optional<at::Tensor>& fc2_bias,
    bool approximate_tanh
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [seq, hidden]");
    auto hidden = hidden_states.size(1);
    auto normed = at::layer_norm(hidden_states, {hidden}, ln_weight, ln_bias, eps, true);
    auto inter = at::linear(normed, fc1_weight, fc1_bias);
    at::gelu_(inter, approximate_tanh ? "tanh" : "none");
    return at::linear(inter, fc2_weight, fc2_bias);
}

at::Tensor vision_patch_merger_forward(
    const at::Tensor& x,
    bool use_postshuffle_norm,
    int64_t hidden_size,
    const at::Tensor& norm_weight,
    const at::Tensor& norm_bias,
    double eps,
    const at::Tensor& fc1_weight,
    const c10::optional<at::Tensor>& fc1_bias,
    const at::Tensor& fc2_weight,
    const c10::optional<at::Tensor>& fc2_bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    at::Tensor flat;
    if (use_postshuffle_norm) {
        flat = x.contiguous().view({-1, hidden_size});
        flat = at::layer_norm(flat, {hidden_size}, norm_weight, norm_bias, eps, true);
    } else {
        auto normed = at::layer_norm(x, {norm_weight.numel()}, norm_weight, norm_bias, eps, true);
        flat = normed.contiguous().view({-1, hidden_size});
    }
    auto inter = at::linear(flat, fc1_weight, fc1_bias);
    at::gelu_(inter, "none");
    return at::linear(inter, fc2_weight, fc2_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_rope_transpose", &qkv_rope_transpose, "v167a fused vision QKV reshape + split-half RoPE + transpose");
    m.def("vision_attn_prestage", &vision_attn_prestage, "v167a vision attention prestage: layernorm + qkv + rope");
    m.def("vision_block_mlp_forward", &vision_block_mlp_forward, "v167a vision block MLP big-op wrapper");
    m.def("vision_patch_merger_forward", &vision_patch_merger_forward, "v167a vision patch merger big-op wrapper");
}
