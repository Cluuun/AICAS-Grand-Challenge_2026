#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kHeadDim = 128;
constexpr int kQHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kKvGroups = 2;
constexpr int kThreads = 64;
constexpr int kPairsPerHead = kHeadDim / 2;
constexpr int kMaxSplits = 8;
constexpr float kNegInf = -1.0e20f;

struct PairAccum {
    float x;
    float y;
};

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename CacheIndexT>
__global__ void lean_decode_split_kernel(
    const half* __restrict__ q_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    float* __restrict__ partial_acc_ptr,
    float* __restrict__ partial_m_ptr,
    float* __restrict__ partial_l_ptr,
    const CacheIndexT* __restrict__ cache_pos_ptr,
    int64_t q_stride_h,
    int64_t q_stride_d,
    int64_t k_stride_h,
    int64_t k_stride_s,
    int64_t k_stride_d,
    int64_t v_stride_h,
    int64_t v_stride_s,
    int64_t v_stride_d,
    int64_t acc_stride_h,
    int64_t acc_stride_split,
    int64_t acc_stride_d,
    int64_t stats_stride_h,
    int64_t stats_stride_split,
    int kv_capacity,
    float sm_scale,
    int max_splits,
    int tokens_per_split) {
    const int kv_head_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int pair_idx = threadIdx.x;
    const int dim0 = pair_idx * 2;

    const int q_head0 = kv_head_idx * kKvGroups;
    const int q_head1 = q_head0 + 1;

    const int kv_len = max(1, min(static_cast<int>(cache_pos_ptr[0]) + 1, kv_capacity));
    const int num_splits = max(1, min((kv_len + tokens_per_split - 1) / tokens_per_split, max_splits));

    PairAccum acc0{0.0f, 0.0f};
    PairAccum acc1{0.0f, 0.0f};
    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;

    const half2 q0_h2 = *reinterpret_cast<const half2*>(
        q_ptr + q_head0 * q_stride_h + dim0 * q_stride_d);
    const half2 q1_h2 = *reinterpret_cast<const half2*>(
        q_ptr + q_head1 * q_stride_h + dim0 * q_stride_d);
    const float2 q0 = __half22float2(q0_h2);
    const float2 q1 = __half22float2(q1_h2);

    __shared__ float warp_scores0[2];
    __shared__ float warp_scores1[2];
    __shared__ float block_scores[2];

    if (split_idx < num_splits) {
        const int split_start = (split_idx * kv_len) / num_splits;
        const int split_end = ((split_idx + 1) * kv_len) / num_splits;

        for (int token_idx = split_start; token_idx < split_end; ++token_idx) {
            const half2 k_h2 = *reinterpret_cast<const half2*>(
                k_ptr + kv_head_idx * k_stride_h + token_idx * k_stride_s + dim0 * k_stride_d);
            const half2 v_h2 = *reinterpret_cast<const half2*>(
                v_ptr + kv_head_idx * v_stride_h + token_idx * v_stride_s + dim0 * v_stride_d);
            const float2 k = __half22float2(k_h2);
            const float2 v = __half22float2(v_h2);

            float score0 = q0.x * k.x + q0.y * k.y;
            float score1 = q1.x * k.x + q1.y * k.y;
            score0 = warp_sum(score0);
            score1 = warp_sum(score1);

            if ((pair_idx & 31) == 0) {
                const int warp_idx = pair_idx >> 5;
                warp_scores0[warp_idx] = score0;
                warp_scores1[warp_idx] = score1;
            }
            __syncthreads();

            if (pair_idx == 0) {
                block_scores[0] = (warp_scores0[0] + warp_scores0[1]) * sm_scale;
                block_scores[1] = (warp_scores1[0] + warp_scores1[1]) * sm_scale;
            }
            __syncthreads();

            const float s0 = block_scores[0];
            const float s1 = block_scores[1];

            const float m0_new = fmaxf(m0, s0);
            const float m1_new = fmaxf(m1, s1);
            const float alpha0 = expf(m0 - m0_new);
            const float alpha1 = expf(m1 - m1_new);
            const float p0 = expf(s0 - m0_new);
            const float p1 = expf(s1 - m1_new);

            acc0.x = acc0.x * alpha0 + v.x * p0;
            acc0.y = acc0.y * alpha0 + v.y * p0;
            acc1.x = acc1.x * alpha1 + v.x * p1;
            acc1.y = acc1.y * alpha1 + v.y * p1;
            l0 = l0 * alpha0 + p0;
            l1 = l1 * alpha1 + p1;
            m0 = m0_new;
            m1 = m1_new;
        }
    }

    const int64_t acc0_offset =
        static_cast<int64_t>(q_head0) * acc_stride_h + static_cast<int64_t>(split_idx) * acc_stride_split +
        static_cast<int64_t>(dim0) * acc_stride_d;
    const int64_t acc1_offset =
        static_cast<int64_t>(q_head1) * acc_stride_h + static_cast<int64_t>(split_idx) * acc_stride_split +
        static_cast<int64_t>(dim0) * acc_stride_d;

    partial_acc_ptr[acc0_offset] = acc0.x;
    partial_acc_ptr[acc0_offset + acc_stride_d] = acc0.y;
    partial_acc_ptr[acc1_offset] = acc1.x;
    partial_acc_ptr[acc1_offset + acc_stride_d] = acc1.y;

    if (pair_idx == 0) {
        const int64_t stats0_offset =
            static_cast<int64_t>(q_head0) * stats_stride_h + static_cast<int64_t>(split_idx) * stats_stride_split;
        const int64_t stats1_offset =
            static_cast<int64_t>(q_head1) * stats_stride_h + static_cast<int64_t>(split_idx) * stats_stride_split;
        partial_m_ptr[stats0_offset] = split_idx < num_splits ? m0 : kNegInf;
        partial_l_ptr[stats0_offset] = split_idx < num_splits ? l0 : 0.0f;
        partial_m_ptr[stats1_offset] = split_idx < num_splits ? m1 : kNegInf;
        partial_l_ptr[stats1_offset] = split_idx < num_splits ? l1 : 0.0f;
    }
}

template <typename CacheIndexT>
__global__ void lean_decode_reduce_kernel(
    const float* __restrict__ partial_acc_ptr,
    const float* __restrict__ partial_m_ptr,
    const float* __restrict__ partial_l_ptr,
    half* __restrict__ out_ptr,
    const CacheIndexT* __restrict__ cache_pos_ptr,
    int64_t acc_stride_h,
    int64_t acc_stride_split,
    int64_t acc_stride_d,
    int64_t stats_stride_h,
    int64_t stats_stride_split,
    int64_t out_stride_h,
    int64_t out_stride_d,
    int kv_capacity,
    int max_splits,
    int tokens_per_split) {
    const int kv_head_idx = blockIdx.x;
    const int pair_idx = threadIdx.x;
    const int dim0 = pair_idx * 2;

    const int q_head0 = kv_head_idx * kKvGroups;
    const int q_head1 = q_head0 + 1;

    const int kv_len = max(1, min(static_cast<int>(cache_pos_ptr[0]) + 1, kv_capacity));
    const int num_splits = max(1, min((kv_len + tokens_per_split - 1) / tokens_per_split, max_splits));

    PairAccum acc0{0.0f, 0.0f};
    PairAccum acc1{0.0f, 0.0f};
    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;

    __shared__ float split_stats[4];

    for (int split_idx = 0; split_idx < num_splits; ++split_idx) {
        if (split_idx > 0) {
            __syncthreads();
        }

        if (pair_idx == 0) {
            const int64_t stats0_offset =
                static_cast<int64_t>(q_head0) * stats_stride_h + static_cast<int64_t>(split_idx) * stats_stride_split;
            const int64_t stats1_offset =
                static_cast<int64_t>(q_head1) * stats_stride_h + static_cast<int64_t>(split_idx) * stats_stride_split;
            split_stats[0] = partial_m_ptr[stats0_offset];
            split_stats[1] = partial_l_ptr[stats0_offset];
            split_stats[2] = partial_m_ptr[stats1_offset];
            split_stats[3] = partial_l_ptr[stats1_offset];
        }
        __syncthreads();

        const int64_t acc0_offset =
            static_cast<int64_t>(q_head0) * acc_stride_h + static_cast<int64_t>(split_idx) * acc_stride_split +
            static_cast<int64_t>(dim0) * acc_stride_d;
        const int64_t acc1_offset =
            static_cast<int64_t>(q_head1) * acc_stride_h + static_cast<int64_t>(split_idx) * acc_stride_split +
            static_cast<int64_t>(dim0) * acc_stride_d;

        const PairAccum part0{
            partial_acc_ptr[acc0_offset],
            partial_acc_ptr[acc0_offset + acc_stride_d],
        };
        const PairAccum part1{
            partial_acc_ptr[acc1_offset],
            partial_acc_ptr[acc1_offset + acc_stride_d],
        };

        const float part_m0 = split_stats[0];
        const float part_l0 = split_stats[1];
        const float part_m1 = split_stats[2];
        const float part_l1 = split_stats[3];

        const float m0_new = fmaxf(m0, part_m0);
        const float m1_new = fmaxf(m1, part_m1);
        const float alpha0 = expf(m0 - m0_new);
        const float alpha1 = expf(m1 - m1_new);
        const float beta0 = expf(part_m0 - m0_new);
        const float beta1 = expf(part_m1 - m1_new);

        acc0.x = acc0.x * alpha0 + part0.x * beta0;
        acc0.y = acc0.y * alpha0 + part0.y * beta0;
        acc1.x = acc1.x * alpha1 + part1.x * beta1;
        acc1.y = acc1.y * alpha1 + part1.y * beta1;
        l0 = l0 * alpha0 + part_l0 * beta0;
        l1 = l1 * alpha1 + part_l1 * beta1;
        m0 = m0_new;
        m1 = m1_new;
    }

    const float inv_l0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float inv_l1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;

    const int64_t out0_offset = static_cast<int64_t>(q_head0) * out_stride_h + static_cast<int64_t>(dim0) * out_stride_d;
    const int64_t out1_offset = static_cast<int64_t>(q_head1) * out_stride_h + static_cast<int64_t>(dim0) * out_stride_d;

    out_ptr[out0_offset] = __float2half_rn(acc0.x * inv_l0);
    out_ptr[out0_offset + out_stride_d] = __float2half_rn(acc0.y * inv_l0);
    out_ptr[out1_offset] = __float2half_rn(acc1.x * inv_l1);
    out_ptr[out1_offset + out_stride_d] = __float2half_rn(acc1.y * inv_l1);
}

template <typename CacheIndexT>
void launch_lean_decode_attention(
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor cache_position,
    torch::Tensor partial_acc,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor output,
    float sm_scale,
    int max_splits,
    int tokens_per_split) {
    const dim3 split_grid(kKvHeads, max_splits);
    const dim3 reduce_grid(kKvHeads);
    const auto stream = at::cuda::getCurrentCUDAStream(query_states.device().index()).stream();

    lean_decode_split_kernel<CacheIndexT><<<split_grid, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(query_states.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_states.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_states.data_ptr<at::Half>()),
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        cache_position.data_ptr<CacheIndexT>(),
        query_states.stride(1),
        query_states.stride(3),
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_m.stride(0),
        partial_m.stride(1),
        static_cast<int>(key_states.size(2)),
        sm_scale,
        max_splits,
        tokens_per_split);

    lean_decode_reduce_kernel<CacheIndexT><<<reduce_grid, kThreads, 0, stream>>>(
        partial_acc.data_ptr<float>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        cache_position.data_ptr<CacheIndexT>(),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_m.stride(0),
        partial_m.stride(1),
        output.stride(1),
        output.stride(3),
        static_cast<int>(key_states.size(2)),
        max_splits,
        tokens_per_split);
}

void check_inputs(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    int64_t max_splits,
    int64_t tokens_per_split) {
    TORCH_CHECK(query_states.is_cuda(), "query_states must be CUDA");
    TORCH_CHECK(key_states.is_cuda(), "key_states must be CUDA");
    TORCH_CHECK(value_states.is_cuda(), "value_states must be CUDA");
    TORCH_CHECK(cache_position.is_cuda(), "cache_position must be CUDA");
    TORCH_CHECK(partial_acc.is_cuda(), "partial_acc must be CUDA");
    TORCH_CHECK(partial_m.is_cuda(), "partial_m must be CUDA");
    TORCH_CHECK(partial_l.is_cuda(), "partial_l must be CUDA");

    TORCH_CHECK(query_states.scalar_type() == at::kHalf, "query_states must be fp16");
    TORCH_CHECK(key_states.scalar_type() == at::kHalf, "key_states must be fp16");
    TORCH_CHECK(value_states.scalar_type() == at::kHalf, "value_states must be fp16");
    TORCH_CHECK(partial_acc.scalar_type() == at::kFloat, "partial_acc must be fp32");
    TORCH_CHECK(partial_m.scalar_type() == at::kFloat, "partial_m must be fp32");
    TORCH_CHECK(partial_l.scalar_type() == at::kFloat, "partial_l must be fp32");
    TORCH_CHECK(
        cache_position.scalar_type() == at::kLong || cache_position.scalar_type() == at::kInt,
        "cache_position must be int64 or int32");

    TORCH_CHECK(query_states.dim() == 4, "query_states must be rank-4");
    TORCH_CHECK(key_states.dim() == 4, "key_states must be rank-4");
    TORCH_CHECK(value_states.dim() == 4, "value_states must be rank-4");
    TORCH_CHECK(query_states.size(0) == 1 && query_states.size(1) == kQHeads && query_states.size(2) == 1 &&
                    query_states.size(3) == kHeadDim,
        "query_states shape must be [1, 16, 1, 128]");
    TORCH_CHECK(key_states.size(0) == 1 && key_states.size(1) == kKvHeads && key_states.size(3) == kHeadDim,
        "key_states shape must be [1, 8, S, 128]");
    TORCH_CHECK(value_states.sizes() == key_states.sizes(), "value_states must match key_states");
    TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");

    TORCH_CHECK(query_states.stride(3) == 1, "query_states last-dim stride must be 1");
    TORCH_CHECK(key_states.stride(3) == 1, "key_states last-dim stride must be 1");
    TORCH_CHECK(value_states.stride(3) == 1, "value_states last-dim stride must be 1");
    TORCH_CHECK(partial_acc.stride(2) == 1, "partial_acc last-dim stride must be 1");
    TORCH_CHECK(partial_acc.size(0) >= kQHeads && partial_acc.size(2) == kHeadDim, "partial_acc shape mismatch");
    TORCH_CHECK(partial_m.size(0) >= kQHeads, "partial_m shape mismatch");
    TORCH_CHECK(partial_l.size(0) >= kQHeads, "partial_l shape mismatch");
    TORCH_CHECK(partial_acc.size(1) >= max_splits, "partial_acc split dimension too small");
    TORCH_CHECK(partial_m.size(1) >= max_splits, "partial_m split dimension too small");
    TORCH_CHECK(partial_l.size(1) >= max_splits, "partial_l split dimension too small");

    TORCH_CHECK(max_splits >= 1 && max_splits <= kMaxSplits, "max_splits must be in [1, 8]");
    TORCH_CHECK(tokens_per_split >= 1, "tokens_per_split must be positive");
}

}  // namespace

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
    int64_t tokens_per_split) {
    check_inputs(
        query_states,
        key_states,
        value_states,
        cache_position,
        partial_acc,
        partial_m,
        partial_l,
        max_splits,
        tokens_per_split);

    const c10::cuda::CUDAGuard device_guard(query_states.device());
    auto output = torch::empty_like(query_states);

    if (cache_position.scalar_type() == at::kLong) {
        launch_lean_decode_attention<int64_t>(
            query_states,
            key_states,
            value_states,
            cache_position,
            partial_acc,
            partial_m,
            partial_l,
            output,
            static_cast<float>(sm_scale),
            static_cast<int>(max_splits),
            static_cast<int>(tokens_per_split));
    } else {
        launch_lean_decode_attention<int32_t>(
            query_states,
            key_states,
            value_states,
            cache_position,
            partial_acc,
            partial_m,
            partial_l,
            output,
            static_cast<float>(sm_scale),
            static_cast<int>(max_splits),
            static_cast<int>(tokens_per_split));
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
