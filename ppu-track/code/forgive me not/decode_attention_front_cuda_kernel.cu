#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kHalfDim = 64;
constexpr int kQHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kKvGroups = 2;
constexpr int kThreads = 64;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum_64(float value, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp_idx = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        shared[warp_idx] = value;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp_idx == 0) {
        total = lane < 2 ? shared[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            shared[0] = total;
        }
    }
    __syncthreads();
    return shared[0];
}

template <typename CacheIndexT>
__global__ void decode_attention_front_kernel(
    const half* __restrict__ q_proj_ptr,
    const half* __restrict__ k_proj_ptr,
    const half* __restrict__ v_proj_ptr,
    const half* __restrict__ q_weight_ptr,
    const half* __restrict__ k_weight_ptr,
    const half* __restrict__ cos_ptr,
    const half* __restrict__ sin_ptr,
    half* __restrict__ k_cache_ptr,
    half* __restrict__ v_cache_ptr,
    const CacheIndexT* __restrict__ cache_position_ptr,
    half* __restrict__ q_out_ptr,
    int64_t q_out_stride_h,
    int64_t q_out_stride_d,
    int64_t k_cache_stride_h,
    int64_t k_cache_stride_s,
    int64_t k_cache_stride_d,
    int64_t v_cache_stride_h,
    int64_t v_cache_stride_s,
    int64_t v_cache_stride_d,
    float q_eps,
    float k_eps) {
    const int kv_head_idx = blockIdx.x;
    const int i = threadIdx.x;
    const int q_head0 = kv_head_idx * kKvGroups;
    const int q_head1 = q_head0 + 1;
    const int write_pos = static_cast<int>(cache_position_ptr[0]);

    const int dim_first = i;
    const int dim_second = i + kHalfDim;

    const int q0_base = q_head0 * kHeadDim;
    const int q1_base = q_head1 * kHeadDim;
    const int kv_base = kv_head_idx * kHeadDim;

    const float q0_first = __half2float(q_proj_ptr[q0_base + dim_first]);
    const float q0_second = __half2float(q_proj_ptr[q0_base + dim_second]);
    const float q1_first = __half2float(q_proj_ptr[q1_base + dim_first]);
    const float q1_second = __half2float(q_proj_ptr[q1_base + dim_second]);
    const float k_first = __half2float(k_proj_ptr[kv_base + dim_first]);
    const float k_second = __half2float(k_proj_ptr[kv_base + dim_second]);
    const half v_first_h = v_proj_ptr[kv_base + dim_first];
    const half v_second_h = v_proj_ptr[kv_base + dim_second];

    __shared__ float reduce_tmp[2];
    const float q0_var = block_sum_64(q0_first * q0_first + q0_second * q0_second, reduce_tmp) / static_cast<float>(kHeadDim);
    const float q1_var = block_sum_64(q1_first * q1_first + q1_second * q1_second, reduce_tmp) / static_cast<float>(kHeadDim);
    const float k_var = block_sum_64(k_first * k_first + k_second * k_second, reduce_tmp) / static_cast<float>(kHeadDim);

    const float q0_inv = rsqrtf(q0_var + q_eps);
    const float q1_inv = rsqrtf(q1_var + q_eps);
    const float k_inv = rsqrtf(k_var + k_eps);

    const float q_weight_first = __half2float(q_weight_ptr[dim_first]);
    const float q_weight_second = __half2float(q_weight_ptr[dim_second]);
    const float k_weight_first = __half2float(k_weight_ptr[dim_first]);
    const float k_weight_second = __half2float(k_weight_ptr[dim_second]);
    const float cos_first = __half2float(cos_ptr[dim_first]);
    const float cos_second = __half2float(cos_ptr[dim_second]);
    const float sin_first = __half2float(sin_ptr[dim_first]);
    const float sin_second = __half2float(sin_ptr[dim_second]);

    const float q0_norm_first = q0_first * q0_inv * q_weight_first;
    const float q0_norm_second = q0_second * q0_inv * q_weight_second;
    const float q1_norm_first = q1_first * q1_inv * q_weight_first;
    const float q1_norm_second = q1_second * q1_inv * q_weight_second;
    const float k_norm_first = k_first * k_inv * k_weight_first;
    const float k_norm_second = k_second * k_inv * k_weight_second;

    const float q0_out_first = q0_norm_first * cos_first - q0_norm_second * sin_first;
    const float q0_out_second = q0_norm_second * cos_second + q0_norm_first * sin_second;
    const float q1_out_first = q1_norm_first * cos_first - q1_norm_second * sin_first;
    const float q1_out_second = q1_norm_second * cos_second + q1_norm_first * sin_second;
    const float k_out_first = k_norm_first * cos_first - k_norm_second * sin_first;
    const float k_out_second = k_norm_second * cos_second + k_norm_first * sin_second;

    q_out_ptr[q_head0 * q_out_stride_h + dim_first * q_out_stride_d] = __float2half_rn(q0_out_first);
    q_out_ptr[q_head0 * q_out_stride_h + dim_second * q_out_stride_d] = __float2half_rn(q0_out_second);
    q_out_ptr[q_head1 * q_out_stride_h + dim_first * q_out_stride_d] = __float2half_rn(q1_out_first);
    q_out_ptr[q_head1 * q_out_stride_h + dim_second * q_out_stride_d] = __float2half_rn(q1_out_second);

    const int64_t k_cache_offset = static_cast<int64_t>(kv_head_idx) * k_cache_stride_h + static_cast<int64_t>(write_pos) * k_cache_stride_s;
    const int64_t v_cache_offset = static_cast<int64_t>(kv_head_idx) * v_cache_stride_h + static_cast<int64_t>(write_pos) * v_cache_stride_s;
    k_cache_ptr[k_cache_offset + dim_first * k_cache_stride_d] = __float2half_rn(k_out_first);
    k_cache_ptr[k_cache_offset + dim_second * k_cache_stride_d] = __float2half_rn(k_out_second);
    v_cache_ptr[v_cache_offset + dim_first * v_cache_stride_d] = v_first_h;
    v_cache_ptr[v_cache_offset + dim_second * v_cache_stride_d] = v_second_h;
}

void check_inputs(
    const torch::Tensor& q_proj,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_out) {
    for (const auto& tensor : {q_proj, k_proj, v_proj, q_weight, k_weight, cos, sin, k_cache, v_cache, cache_position, q_out}) {
        TORCH_CHECK(tensor.is_cuda(), "all tensors must be CUDA");
    }

    TORCH_CHECK(q_proj.scalar_type() == at::kHalf, "q_proj must be fp16");
    TORCH_CHECK(k_proj.scalar_type() == at::kHalf, "k_proj must be fp16");
    TORCH_CHECK(v_proj.scalar_type() == at::kHalf, "v_proj must be fp16");
    TORCH_CHECK(q_weight.scalar_type() == at::kHalf, "q_weight must be fp16");
    TORCH_CHECK(k_weight.scalar_type() == at::kHalf, "k_weight must be fp16");
    TORCH_CHECK(cos.scalar_type() == at::kHalf, "cos must be fp16");
    TORCH_CHECK(sin.scalar_type() == at::kHalf, "sin must be fp16");
    TORCH_CHECK(k_cache.scalar_type() == at::kHalf, "k_cache must be fp16");
    TORCH_CHECK(v_cache.scalar_type() == at::kHalf, "v_cache must be fp16");
    TORCH_CHECK(q_out.scalar_type() == at::kHalf, "q_out must be fp16");
    TORCH_CHECK(
        cache_position.scalar_type() == at::kLong || cache_position.scalar_type() == at::kInt,
        "cache_position must be int64 or int32");

    TORCH_CHECK(q_proj.sizes() == torch::IntArrayRef({1, 1, kQHeads * kHeadDim}), "q_proj shape mismatch");
    TORCH_CHECK(k_proj.sizes() == torch::IntArrayRef({1, 1, kKvHeads * kHeadDim}), "k_proj shape mismatch");
    TORCH_CHECK(v_proj.sizes() == torch::IntArrayRef({1, 1, kKvHeads * kHeadDim}), "v_proj shape mismatch");
    TORCH_CHECK(q_weight.sizes() == torch::IntArrayRef({kHeadDim}), "q_weight shape mismatch");
    TORCH_CHECK(k_weight.sizes() == torch::IntArrayRef({kHeadDim}), "k_weight shape mismatch");
    TORCH_CHECK(cos.sizes() == torch::IntArrayRef({1, 1, kHeadDim}), "cos shape mismatch");
    TORCH_CHECK(sin.sizes() == torch::IntArrayRef({1, 1, kHeadDim}), "sin shape mismatch");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "cache tensors must be rank-4");
    TORCH_CHECK(k_cache.size(0) == 1 && k_cache.size(1) == kKvHeads && k_cache.size(3) == kHeadDim, "k_cache shape mismatch");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache shape mismatch");
    TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({1, kQHeads, 1, kHeadDim}), "q_out shape mismatch");
    TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");

    TORCH_CHECK(q_proj.stride(2) == 1, "q_proj last-dim stride must be 1");
    TORCH_CHECK(k_proj.stride(2) == 1, "k_proj last-dim stride must be 1");
    TORCH_CHECK(v_proj.stride(2) == 1, "v_proj last-dim stride must be 1");
    TORCH_CHECK(q_weight.stride(0) == 1 && k_weight.stride(0) == 1, "weight tensors must be contiguous");
    TORCH_CHECK(cos.stride(2) == 1 && sin.stride(2) == 1, "cos/sin last-dim stride must be 1");
    TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.stride(3) == 1, "cache last-dim stride must be 1");
    TORCH_CHECK(q_out.stride(3) == 1, "q_out last-dim stride must be 1");
}

template <typename CacheIndexT>
void launch_decode_attention_front(
    torch::Tensor q_proj,
    torch::Tensor k_proj,
    torch::Tensor v_proj,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cache_position,
    float q_eps,
    float k_eps,
    torch::Tensor q_out) {
    const dim3 grid(kKvHeads);
    const dim3 block(kThreads);
    const auto stream = at::cuda::getCurrentCUDAStream(q_proj.device().index()).stream();

    decode_attention_front_kernel<CacheIndexT><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(q_proj.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_proj.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_proj.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<CacheIndexT>(),
        reinterpret_cast<half*>(q_out.data_ptr<at::Half>()),
        q_out.stride(1),
        q_out.stride(3),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        q_eps,
        k_eps);
}

}  // namespace

torch::Tensor decode_attention_front_cuda_forward(
    torch::Tensor q_proj,
    torch::Tensor k_proj,
    torch::Tensor v_proj,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cache_position,
    double q_eps,
    double k_eps,
    torch::Tensor q_out) {
    check_inputs(q_proj, k_proj, v_proj, q_weight, k_weight, cos, sin, k_cache, v_cache, cache_position, q_out);

    const c10::cuda::CUDAGuard device_guard(q_proj.device());
    if (cache_position.scalar_type() == at::kLong) {
        launch_decode_attention_front<int64_t>(
            q_proj,
            k_proj,
            v_proj,
            q_weight,
            k_weight,
            cos,
            sin,
            k_cache,
            v_cache,
            cache_position,
            static_cast<float>(q_eps),
            static_cast<float>(k_eps),
            q_out);
    } else {
        launch_decode_attention_front<int32_t>(
            q_proj,
            k_proj,
            v_proj,
            q_weight,
            k_weight,
            cos,
            sin,
            k_cache,
            v_cache,
            cache_position,
            static_cast<float>(q_eps),
            static_cast<float>(k_eps),
            q_out);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_out;
}
