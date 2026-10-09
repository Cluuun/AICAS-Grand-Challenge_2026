#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

void launch_causal_gqa_prefill_attention_v159g(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    float scale,
    cudaStream_t stream
);

at::Tensor causal_gqa_prefill(
    at::Tensor query_states,
    at::Tensor key_states,
    at::Tensor value_states,
    double scale
) {
    TORCH_CHECK(query_states.is_cuda(), "query_states must be CUDA");
    TORCH_CHECK(key_states.is_cuda(), "key_states must be CUDA");
    TORCH_CHECK(value_states.is_cuda(), "value_states must be CUDA");
    TORCH_CHECK(query_states.scalar_type() == at::kHalf, "query_states must be float16");
    TORCH_CHECK(key_states.scalar_type() == at::kHalf, "key_states must be float16");
    TORCH_CHECK(value_states.scalar_type() == at::kHalf, "value_states must be float16");
    TORCH_CHECK(query_states.dim() == 4, "query_states must be [B, Hq, N, D]");
    TORCH_CHECK(key_states.dim() == 4, "key_states must be [B, Hkv, N, D]");
    TORCH_CHECK(value_states.dim() == 4, "value_states must be [B, Hkv, N, D]");
    TORCH_CHECK(query_states.size(0) == 1, "v159g only supports batch size 1");
    TORCH_CHECK(key_states.size(0) == 1 && value_states.size(0) == 1, "v159g only supports batch size 1");
    TORCH_CHECK(query_states.size(2) == key_states.size(2), "sequence length mismatch");
    TORCH_CHECK(key_states.sizes() == value_states.sizes(), "key/value shape mismatch");
    TORCH_CHECK(query_states.size(3) == 128, "v159g only supports head_dim=128");
    TORCH_CHECK(key_states.size(3) == 128 && value_states.size(3) == 128, "v159g only supports head_dim=128");
    TORCH_CHECK(query_states.size(1) % key_states.size(1) == 0, "num_q_heads must be divisible by num_kv_heads");

    auto q = query_states.contiguous();
    auto k = key_states.contiguous();
    auto v = value_states.contiguous();
    auto out = at::empty_like(q);

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_causal_gqa_prefill_attention_v159g(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<int>(q.size(1)),
        static_cast<int>(k.size(1)),
        static_cast<int>(q.size(2)),
        static_cast<float>(scale),
        stream
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_gqa_prefill", &causal_gqa_prefill, "Causal GQA prefill attention vectorized rows4 kv-tile32 (v159g)");
}
