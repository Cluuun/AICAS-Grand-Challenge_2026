#include <torch/extension.h>

void sage_decode_k_i8_v_bf16_single_cuda(
    torch::Tensor query,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor output,
    double softmax_scale);

void sage_decode_q_i8_k_i8_v_bf16_single_cuda(
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor output,
    double softmax_scale);

void sage_decode_qkv_k_i8_v_bf16_single_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps);

void sage_decode_k_i8_v_bf16_split_cuda(
    torch::Tensor query,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale);

void sage_decode_q_i8_k_i8_v_bf16_split_cuda(
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale);

void sage_decode_qkv_k_i8_v_bf16_split_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps);

void sage_decode_qkv_prequant_k_i8_v_bf16_split_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps);

void sage_decode_qkv_prequant_k_i8_v_bf16_split_bf16po_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out_bf16,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps);

void sage_decode_qkv_prequant_k_i8_v_i8_split_bf16po_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache_i8,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out_bf16,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps);

void sage_decode_qkv_prequant_k_i8_v_bf16_split128_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "sage_decode_k_i8_v_bf16_single",
        &sage_decode_k_i8_v_bf16_single_cuda,
        "Single-kernel Sage decode attention with BF16 Q, INT8 K cache, BF16 V cache");
    m.def(
        "sage_decode_q_i8_k_i8_v_bf16_single",
        &sage_decode_q_i8_k_i8_v_bf16_single_cuda,
        "Single-kernel Sage decode attention with INT8 Q/K and BF16 V cache");
    m.def(
        "sage_decode_qkv_k_i8_v_bf16_single",
        &sage_decode_qkv_k_i8_v_bf16_single_cuda,
        "Single-kernel Sage fused QKV cache write plus decode attention");
    m.def(
        "sage_decode_k_i8_v_bf16_split",
        &sage_decode_k_i8_v_bf16_split_cuda,
        "Split-K Sage decode attention with BF16 Q, INT8 K cache, BF16 V cache");
    m.def(
        "sage_decode_q_i8_k_i8_v_bf16_split",
        &sage_decode_q_i8_k_i8_v_bf16_split_cuda,
        "Split-K Sage decode attention with INT8 Q/K and BF16 V cache");
    m.def(
        "sage_decode_qkv_k_i8_v_bf16_split",
        &sage_decode_qkv_k_i8_v_bf16_split_cuda,
        "Split-K Sage fused QKV cache write plus decode attention");
    m.def(
        "sage_decode_qkv_prequant_k_i8_v_bf16_split",
        &sage_decode_qkv_prequant_k_i8_v_bf16_split_cuda,
        "Split-K Sage fused QKV preprocess once plus INT8-Q/K decode attention");
    m.def(
        "sage_decode_qkv_prequant_k_i8_v_bf16_split_bf16po",
        &sage_decode_qkv_prequant_k_i8_v_bf16_split_bf16po_cuda,
        "Split-K Sage fused QKV preprocess once plus BF16 partial-out decode attention");
    m.def(
        "sage_decode_qkv_prequant_k_i8_v_i8_split_bf16po",
        &sage_decode_qkv_prequant_k_i8_v_i8_split_bf16po_cuda,
        "Split-K Sage fused QKV preprocess once plus INT8-Q/K/V BF16 partial-out decode attention");
    m.def(
        "sage_decode_qkv_prequant_k_i8_v_bf16_split128",
        &sage_decode_qkv_prequant_k_i8_v_bf16_split128_cuda,
        "Split-K Sage fused QKV preprocess once plus 128-token KV tile decode attention");
}
