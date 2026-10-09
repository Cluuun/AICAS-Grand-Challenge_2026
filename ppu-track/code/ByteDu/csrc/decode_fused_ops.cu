// CUDA kernels for decode cell optimization on A800 (SM80)
// Targets: bs=1, Qwen3-VL-2B (hidden_size=2048, intermediate=6144)
// Key idea: fuse norm + GEMV + activation boundaries to reduce kernel launches
// and improve HBM bandwidth utilization by keeping intermediate values in registers.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>

using vec_t = uint4;  // 16 bytes = 8 × bf16
constexpr int VEC_SIZE = 8;
constexpr int WARP_SIZE = 32;

// ================================================================
// Kernel 1: Fused RMSNorm + Gate/Up GEMV + SwiGLU
// ================================================================
// Replaces Triton fused_norm_gate_up_swiglu (33.4% of decode, 58% BW)
// Each block computes one or more output elements of the MLP intermediate.
// K=2048 with BD=256 → each thread handles exactly 8 bf16 elements (single pass).
// Norm is redundantly computed per block from L2-cached input (~4KB).
//
// Template parameter INTERLEAVED:
//   true:  weight is [2*N, K] with gate[i]=row[2i], up[i]=row[2i+1] (adjacent, better TLB)
//   false: gate_or_interleaved=[N,K] is gate, up_weight=[N,K] is up (24MB apart)

template <bool INTERLEAVED, int BLOCK_ROWS = 1, int BLOCK_DIM = 256>
__global__ void fused_norm_gate_up_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x,                    // [K] residual
    const __nv_bfloat16* __restrict__ norm_weight,           // [K]
    const __nv_bfloat16* __restrict__ gate_or_interleaved,   // [N,K] or [2N,K]
    const __nv_bfloat16* __restrict__ up_weight,             // [N,K] (unused if INTERLEAVED)
    __nv_bfloat16* __restrict__ out,                         // [N]
    const int N, const int K, const float eps
) {
    const int row_base = blockIdx.x * BLOCK_ROWS;
    if (row_base >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_DIM / WARP_SIZE;

    const int k = tid * VEC_SIZE;  // K=2048, BD=256, VEC=8 → exactly 1 pass

    // ── Phase 1: Load x, compute squared-sum for RMSNorm ──
    float x_f[VEC_SIZE];
    float sq_sum = 0.0f;
    {
        vec_t xv = *reinterpret_cast<const vec_t*>(x + k);
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE / 2; i++) {
            float2 xf = __bfloat1622float2(x2[i]);
            x_f[2 * i]     = xf.x;
            x_f[2 * i + 1] = xf.y;
            sq_sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    // Warp reduce sq_sum
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, off);

    __shared__ float smem[2 * BLOCK_ROWS * NUM_WARPS];
    if (lane == 0) smem[warp_id] = sq_sum;
    __syncthreads();

    float rms_scale;
    if (warp_id == 0) {
        sq_sum = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = (NUM_WARPS >> 1); off > 0; off >>= 1)
            sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, off);
        rms_scale = rsqrtf(sq_sum / float(K) + eps);
        if (lane == 0) smem[0] = rms_scale;
    }
    __syncthreads();
    rms_scale = smem[0];

    // ── Phase 2: Norm → dot(gate, normed) + dot(up, normed) ──
    float normed_f[VEC_SIZE];
    {
        vec_t nwv = *reinterpret_cast<const vec_t*>(norm_weight + k);
        const __nv_bfloat162* nw2 = reinterpret_cast<const __nv_bfloat162*>(&nwv);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE / 2; i++) {
            float2 nwf = __bfloat1622float2(nw2[i]);
            normed_f[2 * i] = x_f[2 * i] * rms_scale * nwf.x;
            normed_f[2 * i + 1] = x_f[2 * i + 1] * rms_scale * nwf.y;
        }
    }

    float gate_acc[BLOCK_ROWS] = {};
    float up_acc[BLOCK_ROWS] = {};

    #pragma unroll
    for (int bn = 0; bn < BLOCK_ROWS; ++bn) {
        const int row_idx = row_base + bn;
        if (row_idx >= N) {
            continue;
        }

        // Load gate/up weight vectors
        vec_t gw, uw;
        if constexpr (INTERLEAVED) {
            const int64_t g_off = (int64_t)(2 * row_idx) * K + k;
            const int64_t u_off = (int64_t)(2 * row_idx + 1) * K + k;
            gw = *reinterpret_cast<const vec_t*>(gate_or_interleaved + g_off);
            uw = *reinterpret_cast<const vec_t*>(gate_or_interleaved + u_off);
        } else {
            const int64_t row_off = (int64_t)row_idx * K + k;
            gw = *reinterpret_cast<const vec_t*>(gate_or_interleaved + row_off);
            uw = *reinterpret_cast<const vec_t*>(up_weight + row_off);
        }
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gw);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&uw);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE / 2; i++) {
            float2 gf = __bfloat1622float2(g2[i]);
            float2 uf = __bfloat1622float2(u2[i]);
            gate_acc[bn] += gf.x * normed_f[2 * i] + gf.y * normed_f[2 * i + 1];
            up_acc[bn]   += uf.x * normed_f[2 * i] + uf.y * normed_f[2 * i + 1];
        }
    }

    // Warp reduce gate_acc and up_acc
    #pragma unroll
    for (int bn = 0; bn < BLOCK_ROWS; ++bn) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gate_acc[bn] += __shfl_xor_sync(0xffffffff, gate_acc[bn], off);
            up_acc[bn]   += __shfl_xor_sync(0xffffffff, up_acc[bn], off);
        }
    }

    // Inter-warp reduce
    if (lane == 0) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_ROWS; ++bn) {
            smem[warp_id * BLOCK_ROWS + bn] = gate_acc[bn];
            smem[NUM_WARPS * BLOCK_ROWS + warp_id * BLOCK_ROWS + bn] = up_acc[bn];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int bn = 0; bn < BLOCK_ROWS; ++bn) {
            float gate_val = (lane < NUM_WARPS) ? smem[lane * BLOCK_ROWS + bn] : 0.0f;
            float up_val = (lane < NUM_WARPS) ? smem[NUM_WARPS * BLOCK_ROWS + lane * BLOCK_ROWS + bn] : 0.0f;
            #pragma unroll
            for (int off = (NUM_WARPS >> 1); off > 0; off >>= 1) {
                gate_val += __shfl_xor_sync(0xffffffff, gate_val, off);
                up_val += __shfl_xor_sync(0xffffffff, up_val, off);
            }
            if (lane == 0 && row_base + bn < N) {
                float silu_gate = gate_val / (1.0f + __expf(-gate_val));
                out[row_base + bn] = __float2bfloat16(silu_gate * up_val);
            }
        }
    }
}


// ================================================================
// Kernel 2: ADDMV with BN=1 for better o_proj bandwidth
// ================================================================
// y[i] += dot(A[i,:], x) for each row i independently
// BN=1 gives 1 block per row → 2048 blocks for [2048, 2048]
// Achieves higher occupancy (100%) vs BN=4 (62%) for N=2048

template <int BLOCK_DIM = 256>
__global__ void addmv_bn1_kernel(
    const __nv_bfloat16* __restrict__ A,   // [N, K]
    const __nv_bfloat16* __restrict__ x,   // [K]
    __nv_bfloat16* __restrict__ y,         // [N]
    const int N, const int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BLOCK_DIM / WARP_SIZE;

    float acc = 0.0f;
    // Single pass for K=2048 (BD=256 * VEC=8 = 2048)
    const int k = tid * VEC_SIZE;
    {
        vec_t xv = *reinterpret_cast<const vec_t*>(x + k);
        vec_t wv = *reinterpret_cast<const vec_t*>(A + (int64_t)row * K + k);
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE / 2; i++) {
            float2 xf = __bfloat1622float2(x2[i]);
            float2 wf = __bfloat1622float2(w2[i]);
            acc += xf.x * wf.x + xf.y * wf.y;
        }
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    __shared__ float smem[8];
    if (lane == 0) smem[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        acc = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = (NUM_WARPS >> 1); off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) {
            y[row] = __float2bfloat16(acc + __bfloat162float(y[row]));
        }
    }
}


// ================================================================
// Kernel 3: LM Head GEMV + Argmax (two-stage)
// ================================================================
// Stage 1: Each block processes BLOCK_ROWS rows, computes dot products
//          and finds local maximum (value + index)
// Stage 2: Single block finds global maximum across all stage-1 results

constexpr int LM_HEAD_BLOCK_ROWS = 256;

__global__ void lm_head_gemv_local_max_kernel(
    const __nv_bfloat16* __restrict__ x,     // [K]
    const __nv_bfloat16* __restrict__ W,     // [N, K]
    float* __restrict__ partial_vals,         // [num_blocks]
    int* __restrict__ partial_idxs,           // [num_blocks]
    const int N, const int K
) {
    constexpr int BD = 256;
    const int block_start = blockIdx.x * LM_HEAD_BLOCK_ROWS;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int NUM_WARPS = BD / WARP_SIZE;

    // Load x into registers (single pass, K=2048)
    float x_f[VEC_SIZE];
    {
        vec_t xv = *reinterpret_cast<const vec_t*>(x + tid * VEC_SIZE);
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE / 2; i++) {
            float2 xf = __bfloat1622float2(x2[i]);
            x_f[2 * i] = xf.x;
            x_f[2 * i + 1] = xf.y;
        }
    }

    // Process BLOCK_ROWS rows, compute dot products and track max
    float local_max_val = -1e30f;
    int local_max_idx = -1;

    for (int r = 0; r < LM_HEAD_BLOCK_ROWS; r++) {
        const int row = block_start + r;
        if (row >= N) break;

        float acc = 0.0f;
        const int k = tid * VEC_SIZE;
        {
            vec_t wv = *reinterpret_cast<const vec_t*>(W + (int64_t)row * K + k);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE / 2; i++) {
                float2 wf = __bfloat1622float2(w2[i]);
                acc += wf.x * x_f[2 * i] + wf.y * x_f[2 * i + 1];
            }
        }

        // Warp reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);

        // Inter-warp reduce via shared memory
        __shared__ float warp_accs[8];
        if (lane == 0) warp_accs[warp_id] = acc;
        __syncthreads();

        if (warp_id == 0) {
            acc = (lane < NUM_WARPS) ? warp_accs[lane] : 0.0f;
            #pragma unroll
            for (int off = (NUM_WARPS >> 1); off > 0; off >>= 1)
                acc += __shfl_xor_sync(0xffffffff, acc, off);

            // Thread 0 of warp 0 tracks the max
            if (lane == 0 && acc > local_max_val) {
                local_max_val = acc;
                local_max_idx = row;
            }
        }
        __syncthreads();  // ensure warp_accs not overwritten before all warps done
    }

    // Write partial results
    if (tid == 0) {
        partial_vals[blockIdx.x] = local_max_val;
        partial_idxs[blockIdx.x] = local_max_idx;
    }
}

// Stage 2: global argmax over partial results
__global__ void lm_head_global_argmax_kernel(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    int64_t* __restrict__ output_token,       // scalar output
    const int num_blocks
) {
    // Single block, tid < num_blocks
    const int tid = threadIdx.x;
    float val = (tid < num_blocks) ? partial_vals[tid] : -1e30f;
    int idx = (tid < num_blocks) ? partial_idxs[tid] : -1;

    // Warp reduce to find max
    const int lane = tid & 31;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, val, off);
        int other_idx = __shfl_xor_sync(0xffffffff, idx, off);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }

    // Inter-warp reduce
    __shared__ float sval[32];
    __shared__ int sidx[32];
    const int warp_id = tid >> 5;
    if (lane == 0) {
        sval[warp_id] = val;
        sidx[warp_id] = idx;
    }
    __syncthreads();

    constexpr int NUM_WARPS = 32;  // max warps in a block of 1024 threads
    if (warp_id == 0 && lane < ((num_blocks + 31) / 32)) {
        val = sval[lane];
        idx = sidx[lane];
    } else if (warp_id == 0) {
        val = -1e30f;
        idx = -1;
    }

    if (warp_id == 0) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, val, off);
            int other_idx = __shfl_xor_sync(0xffffffff, idx, off);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (lane == 0) {
            output_token[0] = (int64_t)idx;
        }
    }
}


// ================================================================
// Python bindings
// ================================================================

void cuda_fused_norm_gate_up_swiglu_interleaved(
    torch::Tensor x,             // [K]
    torch::Tensor norm_weight,   // [K]
    torch::Tensor weight,        // [2*N, K] interleaved
    torch::Tensor out,           // [N]
    float eps
) {
    const int N = out.size(0);
    const int K = x.size(0);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const char* env = std::getenv("JUNKRAT_DECODE_FUSED_MLP_BLOCK_ROWS");
    const int block_rows = env ? std::atoi(env) : 1;
    if (block_rows == 4) {
        fused_norm_gate_up_swiglu_kernel<true, 4, 256><<<(N + 3) / 4, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, K, eps);
    } else if (block_rows == 2) {
        fused_norm_gate_up_swiglu_kernel<true, 2, 256><<<(N + 1) / 2, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, K, eps);
    } else {
        fused_norm_gate_up_swiglu_kernel<true, 1, 256><<<N, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            nullptr,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, K, eps);
    }
}

void cuda_fused_norm_gate_up_swiglu_separate(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor gate_weight,   // [N, K]
    torch::Tensor up_weight,     // [N, K]
    torch::Tensor out,
    float eps
) {
    const int N = out.size(0);
    const int K = x.size(0);
    auto stream = c10::cuda::getCurrentCUDAStream();
    const char* env = std::getenv("JUNKRAT_DECODE_FUSED_MLP_BLOCK_ROWS");
    const int block_rows = env ? std::atoi(env) : 1;
    if (block_rows == 4) {
        fused_norm_gate_up_swiglu_kernel<false, 4, 256><<<(N + 3) / 4, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gate_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, K, eps);
    } else if (block_rows == 2) {
        fused_norm_gate_up_swiglu_kernel<false, 2, 256><<<(N + 1) / 2, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gate_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, K, eps);
    } else {
        fused_norm_gate_up_swiglu_kernel<false, 1, 256><<<N, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gate_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up_weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, K, eps);
    }
}

void cuda_addmv_bn1(torch::Tensor weight, torch::Tensor input, torch::Tensor residual) {
    const int N = weight.size(0);
    const int K = weight.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();
    addmv_bn1_kernel<256><<<N, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        N, K);
}

void cuda_lm_head_argmax(
    torch::Tensor x,            // [K]
    torch::Tensor weight,       // [N, K]
    torch::Tensor partial_vals, // [num_blocks]
    torch::Tensor partial_idxs, // [num_blocks]
    torch::Tensor output_token  // [1] int64
) {
    const int N = weight.size(0);
    const int K = weight.size(1);
    const int num_blocks = (N + LM_HEAD_BLOCK_ROWS - 1) / LM_HEAD_BLOCK_ROWS;
    auto stream = c10::cuda::getCurrentCUDAStream();

    lm_head_gemv_local_max_kernel<<<num_blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        partial_vals.data_ptr<float>(),
        partial_idxs.data_ptr<int>(),
        N, K);

    // Stage 2: num_blocks threads in 1 block
    int stage2_threads = ((num_blocks + 31) / 32) * 32;
    if (stage2_threads < 32) stage2_threads = 32;
    if (stage2_threads > 1024) stage2_threads = 1024;
    lm_head_global_argmax_kernel<<<1, stage2_threads, 0, stream>>>(
        partial_vals.data_ptr<float>(),
        partial_idxs.data_ptr<int>(),
        output_token.data_ptr<int64_t>(),
        num_blocks);
}

int cuda_lm_head_argmax_num_blocks(int vocab_size) {
    return (vocab_size + LM_HEAD_BLOCK_ROWS - 1) / LM_HEAD_BLOCK_ROWS;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_norm_gate_up_swiglu_interleaved",
          &cuda_fused_norm_gate_up_swiglu_interleaved,
          "Fused RMSNorm + Gate/Up GEMV + SwiGLU (interleaved weight layout)");
    m.def("fused_norm_gate_up_swiglu_separate",
          &cuda_fused_norm_gate_up_swiglu_separate,
          "Fused RMSNorm + Gate/Up GEMV + SwiGLU (separate gate/up weights)");
    m.def("addmv_bn1", &cuda_addmv_bn1,
          "ADDMV with BN=1 (better occupancy for N<=2048)");
    m.def("lm_head_argmax", &cuda_lm_head_argmax,
          "Fused LM head GEMV + argmax (two-stage)");
    m.def("lm_head_argmax_num_blocks", &cuda_lm_head_argmax_num_blocks,
          "Get number of blocks for LM head argmax stage 1");
}
