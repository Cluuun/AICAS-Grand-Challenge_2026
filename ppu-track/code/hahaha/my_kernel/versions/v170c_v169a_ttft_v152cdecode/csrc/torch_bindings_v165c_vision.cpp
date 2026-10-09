#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" void launch_vision_qkv_rope_transpose(
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

std::vector<at::Tensor> qkv_rope_transpose(at::Tensor qkv, at::Tensor cos, at::Tensor sin, int64_t num_heads) {
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
    launch_vision_qkv_rope_transpose(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_rope_transpose", &qkv_rope_transpose, "Fused vision QKV reshape + split-half RoPE + transpose");
}
