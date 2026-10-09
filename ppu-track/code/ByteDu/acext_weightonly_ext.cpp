#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <acext/weightOnlyBatchedGemv/kernelLauncher.h>

namespace wo = acext::kernels::weight_only;

torch::Tensor argmax_fp16_cuda(
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out);
torch::Tensor argmax_fp16_post_step_cuda(
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor cos_out,
    torch::Tensor sin_out,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor base_seqlen,
    torch::Tensor input_ids,
    torch::Tensor generated,
    torch::Tensor step_counter,
    int64_t head_dim);
torch::Tensor cast_bf16_to_fp16_cuda(torch::Tensor input, torch::Tensor output);

torch::Tensor fp16_int8_perchannel(torch::Tensor input, torch::Tensor weight, torch::Tensor scales) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "input/weight must be 2D");
    const int m = input.size(0);
    const int k = input.size(1);
    const int n = weight.size(1);
    auto out = torch::empty({m, n}, input.options());
    wo::Params params(input.data_ptr(), nullptr, weight.data_ptr(), scales.data_ptr(), nullptr, nullptr, out.data_ptr(),
        1.f, m, n, k, 0, wo::KernelType::FP16Int8PerChannel, false);
    wo::kernel_launcher(80, params, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor bf16_int8_perchannel(torch::Tensor input, torch::Tensor weight, torch::Tensor scales) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "input/weight must be 2D");
    const int m = input.size(0);
    const int k = input.size(1);
    const int n = weight.size(1);
    auto out = torch::empty({m, n}, input.options());
    wo::Params params(input.data_ptr(), nullptr, weight.data_ptr(), scales.data_ptr(), nullptr, nullptr, out.data_ptr(),
        1.f, m, n, k, 0, wo::KernelType::BF16Int8PerChannel, false);
    wo::kernel_launcher(80, params, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor fp16_int8_perchannel_out(torch::Tensor input, torch::Tensor weight, torch::Tensor scales, torch::Tensor out) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda() && out.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    const int m = input.size(0);
    const int k = input.size(1);
    const int n = weight.size(1);
    TORCH_CHECK(out.size(0) == m && out.size(1) == n, "out shape mismatch");
    wo::Params params(input.data_ptr(), nullptr, weight.data_ptr(), scales.data_ptr(), nullptr, nullptr, out.data_ptr(),
        1.f, m, n, k, 0, wo::KernelType::FP16Int8PerChannel, false);
    wo::kernel_launcher(80, params, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor fp16_int8_perchannel_argmax(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out) {
    fp16_int8_perchannel_out(input, weight, scales, logits);
    return argmax_fp16_cuda(logits, partial_values, partial_indices, token_out);
}

torch::Tensor bf16_int8_perchannel_argmax_via_fp16(
    torch::Tensor input,
    torch::Tensor input_fp16,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out) {
    cast_bf16_to_fp16_cuda(input, input_fp16);
    fp16_int8_perchannel_out(input_fp16, weight, scales, logits);
    return argmax_fp16_cuda(logits, partial_values, partial_indices, token_out);
}

torch::Tensor bf16_int8_perchannel_argmax_post_step_via_fp16(
    torch::Tensor input,
    torch::Tensor input_fp16,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor cos_out,
    torch::Tensor sin_out,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor base_seqlen,
    torch::Tensor input_ids,
    torch::Tensor generated,
    torch::Tensor step_counter,
    int64_t head_dim) {
    cast_bf16_to_fp16_cuda(input, input_fp16);
    fp16_int8_perchannel_out(input_fp16, weight, scales, logits);
    return argmax_fp16_post_step_cuda(
        logits,
        partial_values,
        partial_indices,
        token_out,
        cos_table,
        sin_table,
        cos_out,
        sin_out,
        cache_seqlens,
        cache_seqlens_next,
        base_seqlen,
        input_ids,
        generated,
        step_counter,
        head_dim);
}

torch::Tensor fp16_int4_perchannel(torch::Tensor input, torch::Tensor weight, torch::Tensor scales) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "packed int4 weight must be int8 storage");
    const int m = input.size(0);
    const int k = input.size(1);
    const int n = scales.numel();
    auto out = torch::empty({m, n}, input.options());
    wo::Params params(input.data_ptr(), nullptr, weight.data_ptr(), scales.data_ptr(), nullptr, nullptr, out.data_ptr(),
        1.f, m, n, k, 0, wo::KernelType::FP16Int4PerChannel, false);
    wo::kernel_launcher(80, params, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor fp16_int4_perchannel_argmax(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out) {
    auto out = fp16_int4_perchannel(input, weight, scales);
    logits.copy_(out);
    return argmax_fp16_cuda(logits, partial_values, partial_indices, token_out);
}

torch::Tensor fp16_int4_groupwise(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scales.is_cuda() && zeros.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "packed int4 weight must be int8 storage");
    const int m = input.size(0);
    const int k = input.size(1);
    const int n = scales.size(1);
    auto out = torch::empty({m, n}, input.options());
    wo::Params params(input.data_ptr(), nullptr, weight.data_ptr(), scales.data_ptr(), zeros.data_ptr(), nullptr, out.data_ptr(),
        1.f, m, n, k, static_cast<int>(group_size), wo::KernelType::FP16Int4Groupwise, false);
    wo::kernel_launcher(80, params, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor fp16_int4_groupwise_argmax(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size,
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out) {
    auto out = fp16_int4_groupwise(input, weight, scales, zeros, group_size);
    logits.copy_(out);
    return argmax_fp16_cuda(logits, partial_values, partial_indices, token_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_int8_perchannel", &fp16_int8_perchannel, "Acext WeightOnly FP16xINT8 per-channel GEMV");
    m.def("bf16_int8_perchannel", &bf16_int8_perchannel, "Acext WeightOnly BF16xINT8 per-channel GEMV");
    m.def("fp16_int8_perchannel_out", &fp16_int8_perchannel_out, "Acext WeightOnly FP16xINT8 per-channel GEMV out");
    m.def("fp16_int8_perchannel_argmax", &fp16_int8_perchannel_argmax, "Acext WeightOnly FP16xINT8 per-channel GEMV argmax");
    m.def("bf16_int8_perchannel_argmax_via_fp16", &bf16_int8_perchannel_argmax_via_fp16,
        "Cast BF16 activation to FP16 scratch, then Acext WeightOnly FP16xINT8 per-channel GEMV argmax");
    m.def("bf16_int8_perchannel_argmax_post_step_via_fp16", &bf16_int8_perchannel_argmax_post_step_via_fp16,
        "Cast BF16 activation to FP16 scratch, Acext WeightOnly FP16xINT8 argmax, and fused decode post-step");
    m.def("fp16_int4_perchannel", &fp16_int4_perchannel, "Acext WeightOnly FP16xINT4 per-channel GEMV");
    m.def("fp16_int4_perchannel_argmax", &fp16_int4_perchannel_argmax, "Acext WeightOnly FP16xINT4 per-channel GEMV argmax");
    m.def("fp16_int4_groupwise", &fp16_int4_groupwise, "Acext WeightOnly FP16xINT4 groupwise GEMV");
    m.def("fp16_int4_groupwise_argmax", &fp16_int4_groupwise_argmax, "Acext WeightOnly FP16xINT4 groupwise GEMV argmax");
}
