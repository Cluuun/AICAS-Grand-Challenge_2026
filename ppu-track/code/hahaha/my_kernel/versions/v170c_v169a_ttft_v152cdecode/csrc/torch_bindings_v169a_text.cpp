#include <torch/extension.h>
#include <ATen/ops/linear.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" void launch_text_qkv_norm_rope_transpose_v169a(
    const void* qkv,
    const void* q_norm_weight,
    const void* k_norm_weight,
    const void* cos,
    const void* sin,
    void* q_out,
    void* k_out,
    void* v_out,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float eps,
    int cos_batch,
    int cos_is_half,
    cudaStream_t stream
);

extern "C" void launch_text_swiglu_v169a(
    const void* gateup,
    void* intermediate,
    int batch_size,
    int seq_len,
    int intermediate_size,
    cudaStream_t stream
);

std::vector<at::Tensor> text_attn_prestage(
    const at::Tensor& hidden_states,
    const at::Tensor& qkv_weight,
    const c10::optional<at::Tensor>& qkv_bias,
    const at::Tensor& q_norm_weight,
    const at::Tensor& k_norm_weight,
    double eps,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t num_q_heads,
    int64_t num_kv_heads
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(hidden_states.scalar_type() == at::kHalf, "hidden_states must be float16");
    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be [batch, seq, hidden]");
    TORCH_CHECK(qkv_weight.is_cuda() && qkv_weight.scalar_type() == at::kHalf, "qkv_weight must be CUDA float16");
    TORCH_CHECK(q_norm_weight.is_cuda() && q_norm_weight.scalar_type() == at::kHalf, "q_norm_weight must be CUDA float16");
    TORCH_CHECK(k_norm_weight.is_cuda() && k_norm_weight.scalar_type() == at::kHalf, "k_norm_weight must be CUDA float16");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos/sin must be CUDA");
    TORCH_CHECK((cos.scalar_type() == at::kHalf || cos.scalar_type() == at::kFloat), "cos must be float16/float32");
    TORCH_CHECK((sin.scalar_type() == at::kHalf || sin.scalar_type() == at::kFloat), "sin must be float16/float32");

    const auto batch_size = static_cast<int>(hidden_states.size(0));
    const auto seq_len = static_cast<int>(hidden_states.size(1));
    const auto hidden = static_cast<int>(hidden_states.size(2));
    const auto head_dim = static_cast<int>(q_norm_weight.numel());
    const auto q_width = static_cast<int>(num_q_heads) * head_dim;
    const auto kv_width = static_cast<int>(num_kv_heads) * head_dim;
    TORCH_CHECK(qkv_weight.size(0) == q_width + kv_width + kv_width, "packed qkv weight shape mismatch");
    TORCH_CHECK(qkv_weight.size(1) == hidden, "packed qkv weight hidden size mismatch");
    TORCH_CHECK(k_norm_weight.numel() == head_dim, "q/k norm head_dim mismatch");

    at::Tensor cos_3d;
    at::Tensor sin_3d;
    if (cos.dim() == 2) {
        TORCH_CHECK(cos.size(0) == seq_len && cos.size(1) == head_dim, "cos must be [seq, head_dim]");
        TORCH_CHECK(sin.size(0) == seq_len && sin.size(1) == head_dim, "sin must be [seq, head_dim]");
        cos_3d = cos.unsqueeze(0).contiguous();
        sin_3d = sin.unsqueeze(0).contiguous();
    } else {
        TORCH_CHECK(cos.dim() == 3, "cos must be [seq, head_dim] or [batch, seq, head_dim]");
        TORCH_CHECK(sin.dim() == 3, "sin must be [batch, seq, head_dim]");
        TORCH_CHECK(cos.size(1) == seq_len && cos.size(2) == head_dim, "cos shape mismatch");
        TORCH_CHECK(sin.size(1) == seq_len && sin.size(2) == head_dim, "sin shape mismatch");
        TORCH_CHECK(cos.size(0) == 1 || cos.size(0) == batch_size, "cos batch mismatch");
        TORCH_CHECK(sin.size(0) == 1 || sin.size(0) == batch_size, "sin batch mismatch");
        cos_3d = cos.contiguous();
        sin_3d = sin.contiguous();
    }

    auto qkv = at::linear(hidden_states, qkv_weight, qkv_bias).contiguous();
    auto opts = hidden_states.options();
    auto q_out = at::empty({batch_size, num_q_heads, seq_len, head_dim}, opts);
    auto k_out = at::empty({batch_size, num_kv_heads, seq_len, head_dim}, opts);
    auto v_out = at::empty({batch_size, num_kv_heads, seq_len, head_dim}, opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_text_qkv_norm_rope_transpose_v169a(
        qkv.data_ptr<at::Half>(),
        q_norm_weight.data_ptr<at::Half>(),
        k_norm_weight.data_ptr<at::Half>(),
        cos_3d.data_ptr(),
        sin_3d.data_ptr(),
        q_out.data_ptr<at::Half>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        batch_size,
        seq_len,
        static_cast<int>(num_q_heads),
        static_cast<int>(num_kv_heads),
        head_dim,
        static_cast<float>(eps),
        static_cast<int>(cos_3d.size(0)),
        cos_3d.scalar_type() == at::kHalf ? 1 : 0,
        stream
    );
    return {q_out, k_out, v_out};
}

at::Tensor text_mlp_forward(
    const at::Tensor& hidden_states,
    const at::Tensor& gateup_weight,
    const c10::optional<at::Tensor>& gateup_bias,
    const at::Tensor& down_weight,
    const c10::optional<at::Tensor>& down_bias
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(hidden_states.scalar_type() == at::kHalf, "hidden_states must be float16");
    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be [batch, seq, hidden]");
    TORCH_CHECK(gateup_weight.is_cuda() && gateup_weight.scalar_type() == at::kHalf, "gateup_weight must be CUDA float16");
    TORCH_CHECK(down_weight.is_cuda() && down_weight.scalar_type() == at::kHalf, "down_weight must be CUDA float16");

    const auto batch_size = static_cast<int>(hidden_states.size(0));
    const auto seq_len = static_cast<int>(hidden_states.size(1));
    const auto hidden = static_cast<int>(hidden_states.size(2));
    TORCH_CHECK(gateup_weight.size(1) == hidden, "gateup hidden size mismatch");
    TORCH_CHECK(gateup_weight.size(0) % 2 == 0, "gateup rows must be even");
    const auto intermediate_size = static_cast<int>(gateup_weight.size(0) / 2);
    TORCH_CHECK(down_weight.size(1) == intermediate_size, "down_proj intermediate size mismatch");

    auto gateup = at::linear(hidden_states, gateup_weight, gateup_bias).contiguous();
    auto intermediate = at::empty({batch_size, seq_len, intermediate_size}, hidden_states.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_text_swiglu_v169a(
        gateup.data_ptr<at::Half>(),
        intermediate.data_ptr<at::Half>(),
        batch_size,
        seq_len,
        intermediate_size,
        stream
    );
    return at::linear(intermediate, down_weight, down_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("text_attn_prestage", &text_attn_prestage, "v169a text prefill packed QKV + q/k RMSNorm + RoPE + transpose");
    m.def("text_mlp_forward", &text_mlp_forward, "v169a text prefill packed Gate/Up + fused SiLU*Mul + down_proj");
}
