#include <torch/extension.h>

void instruction_probe_cuda(torch::Tensor out);

void down_w4a16_group128_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block);

void down_w4a16_group128_half2_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block);

void gate_up_w4a16_group128_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block);

void gate_up_w4a16_group128_awq_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block);

void qkv_w4a16_group128_awq_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block);

void o_proj_add_w4a16_group128_awq_cuda(
    torch::Tensor input,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor residual,
    int warps_per_block);

void down_w4a16_group128_half2_awq_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block);

void qkv_w4a16_group128_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block);

void o_proj_add_w4a16_group128_cuda(
    torch::Tensor input,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor residual,
    int warps_per_block);

void w4_weight_stream_cp_async_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor checksum,
    int blocks);

void w4_weight_stream_any_cp_async_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor checksum,
    int blocks,
    int stages);

void lm_head_argmax_w4a16_cuda(
    torch::Tensor hidden,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor output_token,
    int warps_per_block);

void lm_head_argmax_w4a16_persistent_cuda(
    torch::Tensor hidden,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor output_token,
    int warps_per_block,
    int blocks);

void lm_head_argmax_w4a16_persistent_atomic_cuda(
    torch::Tensor hidden,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor partial_indices,
    torch::Tensor output_token,
    int warps_per_block,
    int blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "instruction_probe",
        &instruction_probe_cuda,
        "Compile/runtime probe for LOP3.b32 and PRMT.b32");
    m.def(
        "down_w4a16_group128_probe",
        &down_w4a16_group128_probe_cuda,
        "W4A16 group-128 down GEMV probe for [2048, 6144]");
    m.def(
        "down_w4a16_group128_half2_probe",
        &down_w4a16_group128_half2_probe_cuda,
        "W4A16 group-128 down GEMV probe with half2 magic dequant");
    m.def(
        "down_add_sumsq_w4a16",
        &down_w4a16_group128_half2_probe_cuda,
        "Runtime W4A16 group-128 down add+sumsq GEMV");
    m.def(
        "gate_up_w4a16",
        &gate_up_w4a16_group128_cuda,
        "Runtime W4A16 group-128 RMSNorm + gate/up + SwiGLU GEMV");
    m.def(
        "gate_up_w4a16_awq",
        &gate_up_w4a16_group128_awq_cuda,
        "Runtime W4A16 group-128 gate/up GEMV with AWQ-interleaved nibble dequant");
    m.def(
        "qkv_w4a16_awq",
        &qkv_w4a16_group128_awq_cuda,
        "Runtime W4A16 group-128 QKV GEMV with AWQ-interleaved nibble dequant");
    m.def(
        "o_proj_add_w4a16_awq",
        &o_proj_add_w4a16_group128_awq_cuda,
        "Runtime W4A16 group-128 o_proj addmv with AWQ-interleaved nibble dequant");
    m.def(
        "down_add_sumsq_w4a16_awq",
        &down_w4a16_group128_half2_awq_cuda,
        "Runtime W4A16 group-128 down add+sumsq GEMV with AWQ-interleaved nibble dequant");
    m.def(
        "qkv_w4a16",
        &qkv_w4a16_group128_cuda,
        "Runtime W4A16 group-128 RMSNorm + QKV GEMV");
    m.def(
        "o_proj_add_w4a16",
        &o_proj_add_w4a16_group128_cuda,
        "Runtime W4A16 group-128 o_proj addmv");
    m.def(
        "w4_weight_stream_cp_async_probe",
        &w4_weight_stream_cp_async_probe_cuda,
        "Pure cp.async streaming load probe for W4 packed down weights");
    m.def(
        "w4_weight_stream_any_cp_async_probe",
        &w4_weight_stream_any_cp_async_probe_cuda,
        "Staged cp.async streaming load probe for arbitrary W4 packed weights");
    m.def(
        "lm_head_argmax_w4a16",
        &lm_head_argmax_w4a16_cuda,
        "Runtime W4A16 group-128 lm_head fused argmax");
    m.def(
        "lm_head_argmax_w4a16_persistent",
        &lm_head_argmax_w4a16_persistent_cuda,
        "Runtime W4A16 group-128 lm_head fused argmax with persistent CTAs");
    m.def(
        "lm_head_argmax_w4a16_persistent_atomic",
        &lm_head_argmax_w4a16_persistent_atomic_cuda,
        "Runtime W4A16 group-128 lm_head fused argmax with persistent CTAs and in-kernel reduce");
}
