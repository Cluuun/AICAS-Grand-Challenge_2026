#include <torch/extension.h>

torch::Tensor down_add_sumsq_w8a8_dynamic_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor act_i8,
    torch::Tensor act_scale);

torch::Tensor down_add_sumsq_w8a8_prequant_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq);

torch::Tensor down_add_sumsq_w8a8_prequant_rpb2_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq);

torch::Tensor down_add_sumsq_w8a8_prequant_rpb8_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq);

torch::Tensor down_add_sumsq_w8a8_onthefly_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor act_scale);

torch::Tensor down_norm_qkv_w8a8_prequant_cuda(
    torch::Tensor down_weight,
    torch::Tensor down_weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor qkv_weight,
    torch::Tensor qkv_weight_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps);

torch::Tensor quantize_bf16_to_i8_lagged_cuda(
    torch::Tensor input,
    torch::Tensor act_i8,
    torch::Tensor act_scale);

torch::Tensor quantize_bf16_to_i8_fixed_cuda(
    torch::Tensor input,
    torch::Tensor act_i8,
    torch::Tensor act_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "down_add_sumsq_w8a8_dynamic",
        &down_add_sumsq_w8a8_dynamic_cuda,
        "Per-token A8W8 down_proj + residual add + sumsq");
    m.def(
        "down_add_sumsq_w8a8_prequant",
        &down_add_sumsq_w8a8_prequant_cuda,
        "Prequantized A8W8 down_proj + residual add + sumsq");
    m.def(
        "down_add_sumsq_w8a8_prequant_rpb2",
        &down_add_sumsq_w8a8_prequant_rpb2_cuda,
        "Prequantized A8W8 down_proj + residual add + sumsq, 2 rows per block");
    m.def(
        "down_add_sumsq_w8a8_prequant_rpb8",
        &down_add_sumsq_w8a8_prequant_rpb8_cuda,
        "Prequantized A8W8 down_proj + residual add + sumsq, 8 rows per block");
    m.def(
        "down_add_sumsq_w8a8_onthefly",
        &down_add_sumsq_w8a8_onthefly_cuda,
        "Per-token A8W8 down_proj + residual add + sumsq with on-the-fly activation quantization");
    m.def(
        "down_norm_qkv_w8a8_prequant",
        &down_norm_qkv_w8a8_prequant_cuda,
        "Prequantized A8W8 down_proj + residual add + next RMSNorm + next QKV");
    m.def(
        "quantize_bf16_to_i8_lagged",
        &quantize_bf16_to_i8_lagged_cuda,
        "Single-pass BF16/FP16 to int8 quantization using previous scale and updating next scale");
    m.def(
        "quantize_bf16_to_i8_fixed",
        &quantize_bf16_to_i8_fixed_cuda,
        "Single-pass BF16/FP16 to int8 quantization using a fixed caller-provided scale");
}
