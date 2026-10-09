#include <torch/extension.h>

void gate_up_q8_v2_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int rows_per_block);

void rms_norm_store_fp16_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor output,
    double eps);

void swiglu_packed_fp16_cuda(
    torch::Tensor input,
    torch::Tensor output);

void cast_fp16_to_bf16_cuda(
    torch::Tensor input,
    torch::Tensor output);

void gate_up_q8_v3_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor normed_hidden,
    torch::Tensor output,
    double eps,
    int rows_per_block);

void down_add_sumsq_q8_v2_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor partial_sumsq,
    int rows_per_block);

void gate_up_q8_v4_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block);

void down_add_sumsq_q8_v4_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block);

void norm_qkv_sumsq_q8_v4_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor sumsq,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block);

void down_add_sumsq_w4_v5_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block);

void down_norm_qkv_q8_v2_cuda(
    torch::Tensor down_packed_weight,
    torch::Tensor down_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor qkv_packed_weight,
    torch::Tensor qkv_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps,
    int rows_per_block);

void down_norm_qkv_q8_v3_cuda(
    torch::Tensor down_packed_weight,
    torch::Tensor down_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor normed_hidden,
    torch::Tensor qkv_packed_weight,
    torch::Tensor qkv_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps,
    int rows_per_block);

void add_norm_qkv_fp16_q8_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor normed_hidden,
    torch::Tensor qkv_packed_weight,
    torch::Tensor qkv_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps,
    int rows_per_block);

void add_rms_norm_fp16_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor norm_weight,
    torch::Tensor normed_output,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rms_norm_store_fp16",
        &rms_norm_store_fp16_cuda,
        "RMSNorm-store BF16 vector into FP16 output");
    m.def(
        "swiglu_packed_fp16",
        &swiglu_packed_fp16_cuda,
        "SwiGLU on interleaved FP16 [gate0,up0,gate1,up1,...] tensor");
    m.def(
        "cast_fp16_to_bf16",
        &cast_fp16_to_bf16_cuda,
        "Cast FP16 tensor into BF16 output buffer");
    m.def(
        "gate_up_q8_v2",
        &gate_up_q8_v2_cuda,
        "Fused RMSNorm + interleaved Q8 gate/up + SwiGLU v2");
    m.def(
        "gate_up_q8_v3",
        &gate_up_q8_v3_cuda,
        "RMSNorm-store + interleaved Q8 gate/up + SwiGLU v3");
    m.def(
        "down_add_sumsq_q8_v2",
        &down_add_sumsq_q8_v2_cuda,
        "Fused Q8 down + residual add + sumsq v2");
    m.def(
        "gate_up_q8_v4",
        &gate_up_q8_v4_cuda,
        "Vectorized fused RMSNorm + interleaved Q8 gate/up + SwiGLU v4 (128-bit loads, shared input)");
    m.def(
        "down_add_sumsq_q8_v4",
        &down_add_sumsq_q8_v4_cuda,
        "Vectorized fused Q8 down + residual add + sumsq v4 (128-bit loads, shared input, single launch)");
    m.def(
        "down_add_sumsq_w4_v5",
        &down_add_sumsq_w4_v5_cuda,
        "INT4 weight-only fused down + residual add + sumsq v5 (8 nibbles/word, rowwise scale)");
    m.def(
        "norm_qkv_sumsq_q8_v4",
        &norm_qkv_sumsq_q8_v4_cuda,
        "Vectorized fused RMSNorm(+sumsq write) + Q8 QKV GEMV v4 (128-bit loads, shared input)");
    m.def(
        "down_norm_qkv_q8_v2",
        &down_norm_qkv_q8_v2_cuda,
        "Fused Q8 down + residual add + next RMSNorm + next QKV v2");
    m.def(
        "down_norm_qkv_q8_v3",
        &down_norm_qkv_q8_v3_cuda,
        "Fused Q8 down + residual add + norm-store + next QKV v3");
    m.def(
        "add_norm_qkv_fp16_q8",
        &add_norm_qkv_fp16_q8_cuda,
        "FP16 add + residual + next RMSNorm + next QKV");
    m.def(
        "add_rms_norm_fp16",
        &add_rms_norm_fp16_cuda,
        "FP16 add + residual + RMSNorm to FP16 output");
}
