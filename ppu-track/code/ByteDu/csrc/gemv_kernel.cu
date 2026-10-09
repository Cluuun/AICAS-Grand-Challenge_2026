// Custom GEMV kernel for M=1 decode path (bfloat16)
// Targets: O_proj [2048, 2048], down_proj [2048, 6144], QKV [4096, 2048]
// Key optimization: vectorized loads, efficient warp reduction, L2-cached input broadcast
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Vectorized load type: 8 bf16 values = 16 bytes
using vec_t = uint4;
constexpr int VEC_SIZE = 8;  // bf16 elements per vec_t

// GEMV kernel: y = A @ x
// A: [N, K] row-major, x: [K], y: [N]
template <int BLOCK_N, int BLOCK_DIM>
__global__ void gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const int N,
    const int K
) {
    const int row_base = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_DIM / 32;

    float acc[BLOCK_N] = {};
    const int k_stride = BLOCK_DIM * VEC_SIZE;

    for (int k_base = tid * VEC_SIZE; k_base < K; k_base += k_stride) {
        vec_t x_vec = *reinterpret_cast<const vec_t*>(x + k_base);
        const __nv_bfloat16* x_ptr = reinterpret_cast<const __nv_bfloat16*>(&x_vec);

        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++) {
            const int row = row_base + bn;
            if (row < N) {
                vec_t w_vec = *reinterpret_cast<const vec_t*>(A + (int64_t)row * K + k_base);
                const __nv_bfloat16* w_ptr = reinterpret_cast<const __nv_bfloat16*>(&w_vec);
                const __nv_bfloat162* w_h2 = reinterpret_cast<const __nv_bfloat162*>(w_ptr);
                const __nv_bfloat162* x_h2 = reinterpret_cast<const __nv_bfloat162*>(x_ptr);
                float2 p0 = __bfloat1622float2(__hmul2(w_h2[0], x_h2[0]));
                float2 p1 = __bfloat1622float2(__hmul2(w_h2[1], x_h2[1]));
                float2 p2 = __bfloat1622float2(__hmul2(w_h2[2], x_h2[2]));
                float2 p3 = __bfloat1622float2(__hmul2(w_h2[3], x_h2[3]));
                acc[bn] += (p0.x + p0.y) + (p1.x + p1.y) + (p2.x + p2.y) + (p3.x + p3.y);
            }
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int bn = 0; bn < BLOCK_N; bn++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[bn] += __shfl_xor_sync(0xffffffff, acc[bn], offset);
        }
    }

    // Inter-warp reduction via shared memory
    __shared__ float smem[NUM_WARPS * BLOCK_N];
    if (lane == 0) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++) {
            smem[warp_id * BLOCK_N + bn] = acc[bn];
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane < NUM_WARPS) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++) {
            acc[bn] = smem[lane * BLOCK_N + bn];
        }
        #pragma unroll
        for (int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; bn++) {
                acc[bn] += __shfl_xor_sync(0xffffffff, acc[bn], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; bn++) {
                const int row = row_base + bn;
                if (row < N) {
                    y[row] = __float2bfloat16(acc[bn]);
                }
            }
        }
    }
}

// ADDMV kernel: y += A @ x (in-place add to y)
template <int BLOCK_N, int BLOCK_DIM>
__global__ void addmv_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const int N,
    const int K
) {
    const int row_base = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_DIM / 32;

    float acc[BLOCK_N] = {};
    const int k_stride = BLOCK_DIM * VEC_SIZE;

    for (int k_base = tid * VEC_SIZE; k_base < K; k_base += k_stride) {
        vec_t x_vec = *reinterpret_cast<const vec_t*>(x + k_base);
        const __nv_bfloat16* x_ptr = reinterpret_cast<const __nv_bfloat16*>(&x_vec);

        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++) {
            const int row = row_base + bn;
            if (row < N) {
                vec_t w_vec = *reinterpret_cast<const vec_t*>(A + (int64_t)row * K + k_base);
                const __nv_bfloat16* w_ptr = reinterpret_cast<const __nv_bfloat16*>(&w_vec);
                const __nv_bfloat162* w_h2 = reinterpret_cast<const __nv_bfloat162*>(w_ptr);
                const __nv_bfloat162* x_h2 = reinterpret_cast<const __nv_bfloat162*>(x_ptr);
                float2 p0 = __bfloat1622float2(__hmul2(w_h2[0], x_h2[0]));
                float2 p1 = __bfloat1622float2(__hmul2(w_h2[1], x_h2[1]));
                float2 p2 = __bfloat1622float2(__hmul2(w_h2[2], x_h2[2]));
                float2 p3 = __bfloat1622float2(__hmul2(w_h2[3], x_h2[3]));
                acc[bn] += (p0.x + p0.y) + (p1.x + p1.y) + (p2.x + p2.y) + (p3.x + p3.y);
            }
        }
    }

    #pragma unroll
    for (int bn = 0; bn < BLOCK_N; bn++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[bn] += __shfl_xor_sync(0xffffffff, acc[bn], offset);
        }
    }

    __shared__ float smem[NUM_WARPS * BLOCK_N];
    if (lane == 0) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++) {
            smem[warp_id * BLOCK_N + bn] = acc[bn];
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane < NUM_WARPS) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++) {
            acc[bn] = smem[lane * BLOCK_N + bn];
        }
        #pragma unroll
        for (int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; bn++) {
                acc[bn] += __shfl_xor_sync(0xffffffff, acc[bn], offset);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; bn++) {
                const int row = row_base + bn;
                if (row < N) {
                    y[row] = __float2bfloat16(acc[bn] + __bfloat162float(y[row]));
                }
            }
        }
    }
}


// Dispatch helpers
inline void launch_gemv(const __nv_bfloat16* w, const __nv_bfloat16* x, __nv_bfloat16* y,
                        int N, int K, cudaStream_t stream) {
    constexpr int BD = 256;
    if (N > 2048) {
        constexpr int BN = 2;
        gemv_bf16_kernel<BN, BD><<<(N+BN-1)/BN, BD, 0, stream>>>(w, x, y, N, K);
    } else {
        constexpr int BN = 4;
        gemv_bf16_kernel<BN, BD><<<(N+BN-1)/BN, BD, 0, stream>>>(w, x, y, N, K);
    }
}

inline void launch_addmv(const __nv_bfloat16* w, const __nv_bfloat16* x, __nv_bfloat16* y,
                         int N, int K, cudaStream_t stream) {
    constexpr int BD = 256;
    if (N > 2048) {
        constexpr int BN = 2;
        addmv_bf16_kernel<BN, BD><<<(N+BN-1)/BN, BD, 0, stream>>>(w, x, y, N, K);
    } else {
        constexpr int BN = 4;
        addmv_bf16_kernel<BN, BD><<<(N+BN-1)/BN, BD, 0, stream>>>(w, x, y, N, K);
    }
}

torch::Tensor custom_gemv(torch::Tensor weight, torch::Tensor input) {
    TORCH_CHECK(weight.dim() == 2 && input.dim() == 1);
    TORCH_CHECK(weight.size(1) == input.size(0));
    TORCH_CHECK(weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(weight.is_contiguous() && input.is_contiguous());

    const int N = weight.size(0), K = weight.size(1);
    auto output = torch::empty({N}, weight.options());
    launch_gemv(
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        N, K, c10::cuda::getCurrentCUDAStream());
    return output;
}

void custom_gemv_out(torch::Tensor weight, torch::Tensor input, torch::Tensor output) {
    const int N = weight.size(0), K = weight.size(1);
    launch_gemv(
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        N, K, c10::cuda::getCurrentCUDAStream());
}

void custom_addmv(torch::Tensor weight, torch::Tensor input, torch::Tensor residual) {
    const int N = weight.size(0), K = weight.size(1);
    launch_addmv(
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        N, K, c10::cuda::getCurrentCUDAStream());
}

// Fast fused add + RMS norm kernel (single block, minimal launch overhead)
// Computes: residual_out = x + residual; normed_out = RMSNorm(residual_out, norm_weight, eps)
// Uses exactly 1 thread block of 256 threads for K=2048
__global__ void fused_add_rms_norm_kernel(
    const __nv_bfloat16* __restrict__ x,           // [K]
    const __nv_bfloat16* __restrict__ residual,     // [K]
    const __nv_bfloat16* __restrict__ norm_weight,  // [K]
    __nv_bfloat16* __restrict__ normed_out,         // [K]
    __nv_bfloat16* __restrict__ residual_out,       // [K]
    const int K, const float eps
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    const int k = tid * VEC_SIZE;
    float sum_f[VEC_SIZE];
    float sq_sum = 0.0f;

    vec_t x_vec = *reinterpret_cast<const vec_t*>(x + k);
    vec_t r_vec = *reinterpret_cast<const vec_t*>(residual + k);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x_vec);
    const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&r_vec);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 xf = __bfloat1622float2(x2[i]);
        float2 rf = __bfloat1622float2(r2[i]);
        sum_f[2*i]   = xf.x + rf.x;
        sum_f[2*i+1] = xf.y + rf.y;
        sq_sum += sum_f[2*i] * sum_f[2*i] + sum_f[2*i+1] * sum_f[2*i+1];
    }

    // Write residual_out = x + residual
    {
        __nv_bfloat162 out_pair[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_pair[i] = __floats2bfloat162_rn(sum_f[2*i], sum_f[2*i+1]);
        *reinterpret_cast<vec_t*>(residual_out + k) = *reinterpret_cast<vec_t*>(out_pair);
    }

    // Warp reduce sq_sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);

    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    float rms_scale;
    if (warp_id == 0) {
        sq_sum = (lane < 8) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);
        rms_scale = rsqrtf(sq_sum / float(K) + eps);
        if (lane == 0) warp_sums[0] = rms_scale;
    }
    __syncthreads();
    rms_scale = warp_sums[0];

    // Apply norm weight and write normed output
    vec_t nw_vec = *reinterpret_cast<const vec_t*>(norm_weight + k);
    const __nv_bfloat162* nw2 = reinterpret_cast<const __nv_bfloat162*>(&nw_vec);
    __nv_bfloat162 normed_pair[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 nwf = __bfloat1622float2(nw2[i]);
        normed_pair[i] = __floats2bfloat162_rn(
            sum_f[2*i]   * rms_scale * nwf.x,
            sum_f[2*i+1] * rms_scale * nwf.y
        );
    }
    *reinterpret_cast<vec_t*>(normed_out + k) = *reinterpret_cast<vec_t*>(normed_pair);
}

void custom_fused_add_rms_norm(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor normed_out,
    torch::Tensor residual_out,
    float eps
) {
    const int K = x.size(0);
    fused_add_rms_norm_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(normed_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
        K, eps);
}

// Fused add + RMS norm + GEMV kernel
// Computes: residual_out = x + residual; y = W @ RMSNorm(residual_out, norm_weight, eps)
// For K=2048, each thread handles exactly 8 elements (1 pass).
// All blocks redundantly compute the norm (input ~12KB from L2, negligible).
// Block 0 additionally writes residual_out.
template <int BLOCK_N, int BLOCK_DIM>
__global__ void fused_add_norm_gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,           // [K] - e.g. down_proj output
    const __nv_bfloat16* __restrict__ residual,     // [K] - current residual
    const __nv_bfloat16* __restrict__ norm_weight,  // [K] - RMS norm weight
    const __nv_bfloat16* __restrict__ W,            // [M, K] - GEMV weight (e.g. QKV)
    __nv_bfloat16* __restrict__ y,                  // [M] - GEMV output
    __nv_bfloat16* __restrict__ residual_out,       // [K] - new residual = x + old residual
    const int M, const int K, const float eps
) {
    const int row_base = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_DIM / 32;

    // Phase 1: Load x + residual, compute sum and RMS variance
    float sum_f[VEC_SIZE];
    float normed_f[VEC_SIZE];
    float sq_sum = 0.0f;

    const int k = tid * VEC_SIZE;
    {
        vec_t x_vec = *reinterpret_cast<const vec_t*>(x + k);
        vec_t r_vec = *reinterpret_cast<const vec_t*>(residual + k);
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&x_vec);
        const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&r_vec);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 xf = __bfloat1622float2(x2[i]);
            float2 rf = __bfloat1622float2(r2[i]);
            sum_f[2*i]   = xf.x + rf.x;
            sum_f[2*i+1] = xf.y + rf.y;
            sq_sum += sum_f[2*i] * sum_f[2*i] + sum_f[2*i+1] * sum_f[2*i+1];
        }
    }

    // Warp reduce sq_sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);

    // Inter-warp reduce
    __shared__ float warp_sums[32];
    if (lane == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    float rms_scale;
    if (warp_id == 0) {
        sq_sum = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = (NUM_WARPS >> 1); offset > 0; offset >>= 1)
            sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);
        rms_scale = rsqrtf(sq_sum / float(K) + eps);
        if (lane == 0) warp_sums[0] = rms_scale;
    }
    __syncthreads();
    rms_scale = warp_sums[0];

    // Apply norm weight
    {
        vec_t nw_vec = *reinterpret_cast<const vec_t*>(norm_weight + k);
        const __nv_bfloat162* nw2 = reinterpret_cast<const __nv_bfloat162*>(&nw_vec);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 nwf = __bfloat1622float2(nw2[i]);
            normed_f[2*i]   = sum_f[2*i]   * rms_scale * nwf.x;
            normed_f[2*i+1] = sum_f[2*i+1] * rms_scale * nwf.y;
        }
    }

    // Phase 2: GEMV y = W @ normed (normed in registers, W from HBM)
    float acc[BLOCK_N] = {};
    #pragma unroll
    for (int bn = 0; bn < BLOCK_N; bn++) {
        const int row = row_base + bn;
        if (row < M) {
            vec_t w_vec = *reinterpret_cast<const vec_t*>(W + (int64_t)row * K + k);
            const __nv_bfloat162* w_h2 = reinterpret_cast<const __nv_bfloat162*>(&w_vec);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 wf = __bfloat1622float2(w_h2[i]);
                acc[bn] += wf.x * normed_f[2*i] + wf.y * normed_f[2*i+1];
            }
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int bn = 0; bn < BLOCK_N; bn++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[bn] += __shfl_xor_sync(0xffffffff, acc[bn], offset);
    }

    // Inter-warp reduction via shared memory
    __shared__ float smem[NUM_WARPS * BLOCK_N];
    if (lane == 0) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++)
            smem[warp_id * BLOCK_N + bn] = acc[bn];
    }
    __syncthreads();

    if (warp_id == 0 && lane < NUM_WARPS) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_N; bn++)
            acc[bn] = smem[lane * BLOCK_N + bn];
        #pragma unroll
        for (int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; bn++)
                acc[bn] += __shfl_xor_sync(0xffffffff, acc[bn], offset);
        }
        if (lane == 0) {
            #pragma unroll
            for (int bn = 0; bn < BLOCK_N; bn++) {
                const int row = row_base + bn;
                if (row < M) y[row] = __float2bfloat16(acc[bn]);
            }
        }
    }

    // Phase 3: Block 0 writes residual_out = x + residual (from registers)
    if (blockIdx.x == 0) {
        __nv_bfloat162 out_pair[4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            out_pair[i] = __floats2bfloat162_rn(sum_f[2*i], sum_f[2*i+1]);
        *reinterpret_cast<vec_t*>(residual_out + k) = *reinterpret_cast<vec_t*>(out_pair);
    }
}

inline void launch_fused_add_norm_gemv(
    const __nv_bfloat16* x, const __nv_bfloat16* residual,
    const __nv_bfloat16* norm_weight, const __nv_bfloat16* W,
    __nv_bfloat16* y, __nv_bfloat16* residual_out,
    int M, int K, float eps, cudaStream_t stream
) {
    constexpr int BD = 256;
    if (M > 4096) {
        constexpr int BN = 2;
        fused_add_norm_gemv_bf16_kernel<BN, BD><<<(M+BN-1)/BN, BD, 0, stream>>>(
            x, residual, norm_weight, W, y, residual_out, M, K, eps);
    } else {
        // M=4096: BN=4 gives grid=1024, 1.19 waves (vs BN=2 grid=2048, 2.37 waves)
        // Fewer waves = less Phase 1 redundancy, better wavefront efficiency
        constexpr int BN = 4;
        fused_add_norm_gemv_bf16_kernel<BN, BD><<<(M+BN-1)/BN, BD, 0, stream>>>(
            x, residual, norm_weight, W, y, residual_out, M, K, eps);
    }
}

void custom_fused_add_norm_gemv(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor residual_out,
    float eps
) {
    const int M = weight.size(0), K = weight.size(1);
    launch_fused_add_norm_gemv(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
        M, K, eps, c10::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &custom_gemv, "Custom GEMV y = A @ x (bf16)");
    m.def("gemv_out", &custom_gemv_out, "Custom GEMV with pre-allocated output (bf16)");
    m.def("addmv", &custom_addmv, "Custom ADDMV y += A @ x in-place (bf16)");
    m.def("fused_add_norm_gemv", &custom_fused_add_norm_gemv,
          "Fused add + RMS norm + GEMV: residual_out = x + residual; y = W @ RMSNorm(residual_out, norm_weight, eps)");
    m.def("fused_add_rms_norm", &custom_fused_add_rms_norm,
          "Fast fused add + RMS norm: residual_out = x + residual; normed_out = RMSNorm(residual_out, norm_weight, eps)");
}
