/**
 * Fused Decode with __ldg() cached reads  (ported from MegaQwen)
 *
 * Full transformer decode pipeline for Qwen3-VL-2B in a single persistent
 * kernel with AtomicGridSync + flag-based partial barriers:
 * embed -> 28x(RMSNorm+QKV -> QKNorm+RoPE+Cache -> Attention ->
 * OProj+PostNorm+MLP) -> FinalNorm -> LM Head.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "sm_profiler.h"

// sm-profiler event IDs
#define SM_PROF_EMBEDDING        0
#define SM_PROF_QKV_PROJ         1
#define SM_PROF_QK_NORM_ROPE     2
#define SM_PROF_ATTN_COMPUTE     3
#define SM_PROF_ATTN_PREFETCH    4
#define SM_PROF_O_PROJ_MLP       5
#define SM_PROF_FINAL_NORM       6
#define SM_PROF_GRID_SYNC        7
#define SM_PROF_NUM_EVENTS       8

// =============================================================================
// Configuration & model constants
// =============================================================================

constexpr int WARP_SIZE = 32;

constexpr int HIDDEN_SIZE      = 2048;
constexpr int INTERMEDIATE_SIZE = 6144;
constexpr int NUM_Q_HEADS      = 16;
constexpr int NUM_KV_HEADS     = 8;
constexpr int HEAD_DIM         = 128;
constexpr int Q_SIZE  = NUM_Q_HEADS  * HEAD_DIM;   // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;   // 1024

constexpr int LDG_BLOCK_SIZE = 512;
constexpr int LDG_NUM_WARPS  = LDG_BLOCK_SIZE / WARP_SIZE;  // 16
constexpr float LDG_RMS_EPS  = 1e-6f;

// Shared memory bank conflict padding: 1 pad per 8 elements (stride 9)
// Reduces 8-way bank conflicts to ≤2-way in float4 matvec reads.
// Safe for float4: each lane's 8 consecutive elements never cross a pad slot.
#define SMEM_PAD_IDX(i) ((i) + (i) / 8)
#define SMEM_PAD_SIZE(n) ((n) + (n) / 8)
constexpr size_t LDG_STAGE_SMEM_BYTES =
    static_cast<size_t>(INTERMEDIATE_SIZE) * sizeof(float);

// LM head
constexpr int LDG_LM_NUM_BLOCKS  = 1024;
constexpr int LDG_LM_BLOCK_SIZE  = 256;
constexpr int LDG_VOCAB_SIZE     = 151936;

struct LDGLayerWeights {
    const __half* input_layernorm_weight;
    const __half* q_proj_weight;
    const __half* k_proj_weight;
    const __half* v_proj_weight;
    const __half* q_norm_weight;
    const __half* k_norm_weight;
    const __half* o_proj_weight;
    const __half* post_attn_layernorm_weight;
    const __half* gate_proj_weight;
    const __half* up_proj_weight;
    const __half* down_proj_weight;
    const int8_t* q_proj_q8_weight;
    const float* q_proj_q8_scale;
    const int8_t* gate_proj_q8_weight;
    const float* gate_proj_q8_scale;
    const int8_t* up_proj_q8_weight;
    const float* up_proj_q8_scale;
    const int8_t* down_proj_q8_weight;
    const float* down_proj_q8_scale;
};

// =============================================================================
// Atomic barrier for persistent kernel (replaces cooperative grid.sync())
// =============================================================================

struct AtomicGridSync {
  unsigned int *counter;
  unsigned int *generation;
  unsigned int nblocks;
  unsigned int local_gen;

  __device__ void sync() {
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int my_gen = local_gen;
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      unsigned int arrived = atomicAdd(counter, 1);
      if (arrived == nblocks - 1) {
        *counter = 0;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(generation, 1);
      } else {
        volatile unsigned int *vgen = (volatile unsigned int *)generation;
        while (*vgen <= my_gen) {
        }
      }
      local_gen = my_gen + 1;
    }
    __syncthreads();
  }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float ldg_silu(float x) {
    return x / (1.0f + expf(-x));
}

union LdgHalf2Alias {
    uint32_t u;
    __half2 h2;
};

union LdgInt8x8Alias {
    uint2 u;
    int8_t i8[8];
};

__device__ __forceinline__ float2 ldg_unpack_half2(uint32_t packed) {
    LdgHalf2Alias alias;
    alias.u = packed;
    return __half22float2(alias.h2);
}

__device__ __forceinline__ float ldg_dot8_int8_float(
    const uint2& packed,
    float scale,
    float s0,
    float s1,
    float s2,
    float s3,
    float s4,
    float s5,
    float s6,
    float s7
) {
    LdgInt8x8Alias alias;
    alias.u = packed;
    float sum = 0.0f;
    sum = fmaf(float(alias.i8[0]) * scale, s0, sum);
    sum = fmaf(float(alias.i8[1]) * scale, s1, sum);
    sum = fmaf(float(alias.i8[2]) * scale, s2, sum);
    sum = fmaf(float(alias.i8[3]) * scale, s3, sum);
    sum = fmaf(float(alias.i8[4]) * scale, s4, sum);
    sum = fmaf(float(alias.i8[5]) * scale, s5, sum);
    sum = fmaf(float(alias.i8[6]) * scale, s6, sum);
    sum = fmaf(float(alias.i8[7]) * scale, s7, sum);
    return sum;
}

__device__ __forceinline__ float ldg_dot8_half_float(
    const uint4& packed,
    float s0,
    float s1,
    float s2,
    float s3,
    float s4,
    float s5,
    float s6,
    float s7
) {
    float2 w01 = ldg_unpack_half2(packed.x);
    float2 w23 = ldg_unpack_half2(packed.y);
    float2 w45 = ldg_unpack_half2(packed.z);
    float2 w67 = ldg_unpack_half2(packed.w);
    float sum = 0.0f;
    sum = fmaf(w01.x, s0, sum);
    sum = fmaf(w01.y, s1, sum);
    sum = fmaf(w23.x, s2, sum);
    sum = fmaf(w23.y, s3, sum);
    sum = fmaf(w45.x, s4, sum);
    sum = fmaf(w45.y, s5, sum);
    sum = fmaf(w67.x, s6, sum);
    sum = fmaf(w67.y, s7, sum);
    return sum;
}

__device__ __forceinline__ void ldg_dual_dot8_half_float(
    const uint4& g_packed,
    const uint4& u_packed,
    float s0,
    float s1,
    float s2,
    float s3,
    float s4,
    float s5,
    float s6,
    float s7,
    float& gate_sum,
    float& up_sum
) {
    float2 g01 = ldg_unpack_half2(g_packed.x);
    float2 g23 = ldg_unpack_half2(g_packed.y);
    float2 g45 = ldg_unpack_half2(g_packed.z);
    float2 g67 = ldg_unpack_half2(g_packed.w);
    float2 u01 = ldg_unpack_half2(u_packed.x);
    float2 u23 = ldg_unpack_half2(u_packed.y);
    float2 u45 = ldg_unpack_half2(u_packed.z);
    float2 u67 = ldg_unpack_half2(u_packed.w);

    gate_sum = fmaf(g01.x, s0, gate_sum);
    gate_sum = fmaf(g01.y, s1, gate_sum);
    gate_sum = fmaf(g23.x, s2, gate_sum);
    gate_sum = fmaf(g23.y, s3, gate_sum);
    gate_sum = fmaf(g45.x, s4, gate_sum);
    gate_sum = fmaf(g45.y, s5, gate_sum);
    gate_sum = fmaf(g67.x, s6, gate_sum);
    gate_sum = fmaf(g67.y, s7, gate_sum);

    up_sum = fmaf(u01.x, s0, up_sum);
    up_sum = fmaf(u01.y, s1, up_sum);
    up_sum = fmaf(u23.x, s2, up_sum);
    up_sum = fmaf(u23.y, s3, up_sum);
    up_sum = fmaf(u45.x, s4, up_sum);
    up_sum = fmaf(u45.y, s5, up_sum);
    up_sum = fmaf(u67.x, s6, up_sum);
    up_sum = fmaf(u67.y, s7, up_sum);
}

__device__ __forceinline__ void ldg_dual_dot8_int8_float(
    const uint2& g_packed,
    const uint2& u_packed,
    float gate_scale,
    float up_scale,
    float s0,
    float s1,
    float s2,
    float s3,
    float s4,
    float s5,
    float s6,
    float s7,
    float& gate_sum,
    float& up_sum
) {
    LdgInt8x8Alias g_alias;
    LdgInt8x8Alias u_alias;
    g_alias.u = g_packed;
    u_alias.u = u_packed;

    gate_sum = fmaf(float(g_alias.i8[0]) * gate_scale, s0, gate_sum);
    gate_sum = fmaf(float(g_alias.i8[1]) * gate_scale, s1, gate_sum);
    gate_sum = fmaf(float(g_alias.i8[2]) * gate_scale, s2, gate_sum);
    gate_sum = fmaf(float(g_alias.i8[3]) * gate_scale, s3, gate_sum);
    gate_sum = fmaf(float(g_alias.i8[4]) * gate_scale, s4, gate_sum);
    gate_sum = fmaf(float(g_alias.i8[5]) * gate_scale, s5, gate_sum);
    gate_sum = fmaf(float(g_alias.i8[6]) * gate_scale, s6, gate_sum);
    gate_sum = fmaf(float(g_alias.i8[7]) * gate_scale, s7, gate_sum);

    up_sum = fmaf(float(u_alias.i8[0]) * up_scale, s0, up_sum);
    up_sum = fmaf(float(u_alias.i8[1]) * up_scale, s1, up_sum);
    up_sum = fmaf(float(u_alias.i8[2]) * up_scale, s2, up_sum);
    up_sum = fmaf(float(u_alias.i8[3]) * up_scale, s3, up_sum);
    up_sum = fmaf(float(u_alias.i8[4]) * up_scale, s4, up_sum);
    up_sum = fmaf(float(u_alias.i8[5]) * up_scale, s5, up_sum);
    up_sum = fmaf(float(u_alias.i8[6]) * up_scale, s6, up_sum);
    up_sum = fmaf(float(u_alias.i8[7]) * up_scale, s7, up_sum);
}

// Forward declaration for prefetch (defined in Phase 3)
__device__ void ldg_prefetch_weights_l2(
    const __half* __restrict__ weights, int num_elements
);

// =============================================================================
// Phase 1: RMSNorm + QKV Projection
// =============================================================================

__device__ __noinline__ void ldg_matvec_qkv(
    AtomicGridSync& grid,
    const __half* __restrict__ input,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ q_weight,
    const __half* __restrict__ k_weight,
    const __half* __restrict__ v_weight,
    const int8_t* __restrict__ q_q8_weight,
    const float* __restrict__ q_q8_scale,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    const bool use_q_q8 = (q_q8_weight != nullptr && q_q8_scale != nullptr);

    // Redundant RMSNorm: every block computes its own copy in shared memory
    // to avoid a grid.sync() between norm and QKV projection
    __shared__ float s_normalized[SMEM_PAD_SIZE(HIDDEN_SIZE)];
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __half2float(__ldg(input + i));
            s_normalized[SMEM_PAD_IDX(i)] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();

        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __half2float(__ldg(norm_weight + i));
            s_normalized[SMEM_PAD_IDX(i)] = s_normalized[SMEM_PAD_IDX(i)] * rstd * w;
        }
        __syncthreads();
    }

    // Distributed residual write: each block writes its share
    {
        int res_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int res_start = block_id * res_per_block;
        int res_end = min(res_start + res_per_block, HIDDEN_SIZE);
        for (int i = res_start + threadIdx.x; i < res_end; i += LDG_BLOCK_SIZE)
            g_residual[i] = __half2float(__ldg(input + i));
    }

    // QKV projection with vec4 __ldg (reads from shared memory, no grid.sync needed)
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __half* weight_row;
            const int8_t* q8_weight_row = nullptr;
            float q8_scale = 1.0f;
            float* output_ptr;

            if (m < Q_SIZE) {
                weight_row = q_weight + m * HIDDEN_SIZE;
                if (use_q_q8) {
                    q8_weight_row = q_q8_weight + m * HIDDEN_SIZE;
                    q8_scale = __ldg(q_q8_scale + m);
                }
                output_ptr = q_out + m;
            } else if (m < Q_SIZE + KV_SIZE) {
                weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
                output_ptr = k_out + (m - Q_SIZE);
            } else {
                weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
                output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
            }

            float sum = 0.0f;
            if (q8_weight_row != nullptr) {
                #pragma unroll 4
                for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                    int pk = SMEM_PAD_IDX(k);
                    uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(q8_weight_row + k));
                    sum += ldg_dot8_int8_float(
                        w_u2,
                        q8_scale,
                        s_normalized[pk + 0],
                        s_normalized[pk + 1],
                        s_normalized[pk + 2],
                        s_normalized[pk + 3],
                        s_normalized[pk + 4],
                        s_normalized[pk + 5],
                        s_normalized[pk + 6],
                        s_normalized[pk + 7]
                    );
                }
            } else {
                #pragma unroll 4
                for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                    uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));
                    int pk = SMEM_PAD_IDX(k);
                    sum += ldg_dot8_half_float(
                        w_u4,
                        s_normalized[pk + 0],
                        s_normalized[pk + 1],
                        s_normalized[pk + 2],
                        s_normalized[pk + 3],
                        s_normalized[pk + 4],
                        s_normalized[pk + 5],
                        s_normalized[pk + 6],
                        s_normalized[pk + 7]
                    );
                }
            }
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) *output_ptr = sum;
        }
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 2: QK Norm + RoPE + KV Cache Write
// =============================================================================

__device__ __noinline__ void ldg_qk_norm_rope_cache(
    AtomicGridSync& grid,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __half* __restrict__ q_norm_weight,
    const __half* __restrict__ k_norm_weight,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    int position,
    int max_seq_len,
    unsigned int* __restrict__ kv_flag,
    unsigned int* __restrict__ attn_flag,
    int layer_idx,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __half* cos_pos = cos_table + position * HEAD_DIM;
    const __half* sin_pos = sin_table + position * HEAD_DIM;

    // Q heads
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += q_head[i] * q_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            q_local[j] = q_head[i] * scale * __half2float(__ldg(q_norm_weight + i));

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __half2float(__ldg(cos_pos + i));
            float sin_v = __half2float(__ldg(sin_pos + i));
            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);
            if (i < HEAD_DIM/2)
                q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
            else
                q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
        }
    }

    // K heads + cache write
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __half* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __half* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += k_head[i] * k_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            k_local[j] = k_head[i] * scale * __half2float(__ldg(k_norm_weight + i));

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __half2float(__ldg(cos_pos + i));
            float sin_v = __half2float(__ldg(sin_pos + i));
            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

            float k_final;
            if (i < HEAD_DIM/2)
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            else
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            k_head[i] = k_final;
            k_cache_head[i] = __float2half(k_final);
            v_cache_head[i] = __float2half(v_head[i]);
        }
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 3: Flash-decoding Attention (+ L2 weight prefetch by idle blocks)
// =============================================================================

__device__ void ldg_prefetch_weights_l2(
    const __half* __restrict__ weights, int num_elements
) {
    // Use PTX prefetch.global.L2::evict_last to hint L2 to keep data persistent.
    // Each prefetch touches one cache line (128 bytes = 64 bf16 elements).
    // 256 threads × 128 bytes = 32 KB per iteration = 256 cache lines.
    // Stride: LDG_BLOCK_SIZE cache lines = 256 * 128 bytes = 32768 bytes per step.
    const char* base = reinterpret_cast<const char*>(weights);
    int total_bytes = num_elements * 2;  // bf16 = 2 bytes
    for (int offset = threadIdx.x * 128; offset < total_bytes; offset += LDG_BLOCK_SIZE * 128) {
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(base + offset));
    }
}

__device__ __noinline__ void ldg_attention(
    AtomicGridSync& grid,
    const float* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    float* __restrict__ attn_out,
    float* __restrict__ attn_partial_max,
    float* __restrict__ attn_partial_sum,
    float* __restrict__ attn_partial_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    unsigned int* __restrict__ attn_flag,
    int layer_idx,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    sm_profiler_event_start(profiler_buffer, SM_PROF_ATTN_COMPUTE, prof_on);

    __shared__ float s_max_score[LDG_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_NUM_WARPS];
    __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];
    int base_blocks_per_head = num_blocks / NUM_Q_HEADS;
    int extra_blocks = num_blocks % NUM_Q_HEADS;
    int qh = 0;
    int head_block_start = 0;
    int head_block_count = 0;
    #pragma unroll
    for (int h = 0; h < NUM_Q_HEADS; ++h) {
        int cnt = base_blocks_per_head + (h < extra_blocks ? 1 : 0);
        int next_start = head_block_start + cnt;
        if (block_id < next_start) {
            qh = h;
            head_block_count = cnt;
            break;
        }
        head_block_start = next_start;
    }
    int local_block_idx = block_id - head_block_start;
    int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
    const float* q_head = q + qh * HEAD_DIM;

    int chunk_start = (cache_len * local_block_idx) / head_block_count;
    int chunk_end = (cache_len * (local_block_idx + 1)) / head_block_count;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int q_idx = lane_id * 4;
    float q_local[4];
    q_local[0] = q_head[q_idx + 0];
    q_local[1] = q_head[q_idx + 1];
    q_local[2] = q_head[q_idx + 2];
    q_local[3] = q_head[q_idx + 3];

    for (int pos = chunk_start + warp_id; pos < chunk_end; pos += LDG_NUM_WARPS) {
        const __half* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
        const __half* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

        float score = 0.0f;
        uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_pos + q_idx));
        __half* k_ptr = reinterpret_cast<__half*>(&k_u2);
        score += q_local[0] * __half2float(k_ptr[0]) +
                 q_local[1] * __half2float(k_ptr[1]) +
                 q_local[2] * __half2float(k_ptr[2]) +
                 q_local[3] * __half2float(k_ptr[3]);
        score = ldg_warp_reduce_sum(score) * attn_scale;
        score = __shfl_sync(0xffffffff, score, 0);

        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + expf(score - max_score);
        float weight = expf(score - max_score);

        uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(v_pos + q_idx));
        __half* v_ptr = reinterpret_cast<__half*>(&v_u2);
        out_acc[0] = out_acc[0] * exp_diff + weight * __half2float(v_ptr[0]);
        out_acc[1] = out_acc[1] * exp_diff + weight * __half2float(v_ptr[1]);
        out_acc[2] = out_acc[2] * exp_diff + weight * __half2float(v_ptr[2]);
        out_acc[3] = out_acc[3] * exp_diff + weight * __half2float(v_ptr[3]);
    }

    if (lane_id == 0) {
        s_max_score[warp_id] = max_score;
        s_sum_exp[warp_id] = sum_exp;
    }
    int out_base = lane_id * 4;
    s_out_acc[warp_id][out_base + 0] = out_acc[0];
    s_out_acc[warp_id][out_base + 1] = out_acc[1];
    s_out_acc[warp_id][out_base + 2] = out_acc[2];
    s_out_acc[warp_id][out_base + 3] = out_acc[3];
    __syncthreads();

    if (warp_id == 0) {
        float block_max = -INFINITY;
        for (int w = 0; w < LDG_NUM_WARPS; ++w) {
            if (s_max_score[w] > -INFINITY) {
                block_max = fmaxf(block_max, s_max_score[w]);
            }
        }

        float block_sum = 0.0f;
        float block_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < LDG_NUM_WARPS; ++w) {
            if (s_max_score[w] > -INFINITY) {
                float sc = expf(s_max_score[w] - block_max);
                block_sum += s_sum_exp[w] * sc;
                int base = lane_id * 4;
                block_out[0] += s_out_acc[w][base + 0] * sc;
                block_out[1] += s_out_acc[w][base + 1] * sc;
                block_out[2] += s_out_acc[w][base + 2] * sc;
                block_out[3] += s_out_acc[w][base + 3] * sc;
            }
        }

        if (lane_id == 0) {
            attn_partial_max[block_id] = block_max;
            attn_partial_sum[block_id] = block_sum;
        }
        attn_partial_out[block_id * HEAD_DIM + out_base + 0] = block_out[0];
        attn_partial_out[block_id * HEAD_DIM + out_base + 1] = block_out[1];
        attn_partial_out[block_id * HEAD_DIM + out_base + 2] = block_out[2];
        attn_partial_out[block_id * HEAD_DIM + out_base + 3] = block_out[3];
    }
    __syncthreads();
    sm_profiler_event_end(profiler_buffer, SM_PROF_ATTN_COMPUTE, prof_on);
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    if (local_block_idx == 0) {
        float global_max = -INFINITY;
        for (int b = 0; b < head_block_count; ++b) {
            float part_max = attn_partial_max[head_block_start + b];
            if (part_max > -INFINITY) {
                global_max = fmaxf(global_max, part_max);
            }
        }

        if (warp_id == 0) {
            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int b = 0; b < head_block_count; ++b) {
                int src_block = head_block_start + b;
                float part_max = attn_partial_max[src_block];
                if (part_max > -INFINITY) {
                    float sc = expf(part_max - global_max);
                    total_sum_exp += attn_partial_sum[src_block] * sc;
                    int base = lane_id * 4;
                    const float* src = attn_partial_out + src_block * HEAD_DIM + base;
                    final_out[0] += src[0] * sc;
                    final_out[1] += src[1] * sc;
                    final_out[2] += src[2] * sc;
                    final_out[3] += src[3] * sc;
                }
            }
            int base = lane_id * 4;
            float inv_sum = 1.0f / total_sum_exp;
            float* out_head = attn_out + qh * HEAD_DIM;
            out_head[base + 0] = final_out[0] * inv_sum;
            out_head[base + 1] = final_out[1] * inv_sum;
            out_head[base + 2] = final_out[2] * inv_sum;
            out_head[base + 3] = final_out[3] * inv_sum;
        }
    }
    __syncthreads();
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 4: O Projection + Post-Norm + MLP
// =============================================================================

__device__ __noinline__ void ldg_o_proj_postnorm_mlp(
    AtomicGridSync& grid,
    const __half* __restrict__ o_weight,
    const __half* __restrict__ post_norm_weight,
    const int8_t* __restrict__ gate_q8_weight,
    const float* __restrict__ gate_q8_scale,
    const int8_t* __restrict__ up_q8_weight,
    const float* __restrict__ up_q8_scale,
    const int8_t* __restrict__ down_q8_weight,
    const float* __restrict__ down_q8_scale,
    const float* __restrict__ attn_out,
    float* __restrict__ g_residual,
    float* __restrict__ g_activations,
    float* __restrict__ g_mlp_intermediate,
    __half* __restrict__ hidden_out,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Cache attn_out in shared memory to avoid repeated global reads
    __shared__ float s_attn[SMEM_PAD_SIZE(Q_SIZE)];
    for (int i = threadIdx.x; i < Q_SIZE; i += LDG_BLOCK_SIZE)
        s_attn[SMEM_PAD_IDX(i)] = attn_out[i];
    __syncthreads();

    // O projection + residual
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < hid_end) {
            const __half* o_row = o_weight + m * Q_SIZE;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(o_row + k));
                int pk = SMEM_PAD_IDX(k);
                sum += ldg_dot8_half_float(
                    w_u4,
                    s_attn[pk + 0],
                    s_attn[pk + 1],
                    s_attn[pk + 2],
                    s_attn[pk + 3],
                    s_attn[pk + 4],
                    s_attn[pk + 5],
                    s_attn[pk + 6],
                    s_attn[pk + 7]
                );
            }
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) g_activations[m] = sum + g_residual[m];
        }
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    // Post-attention RMSNorm: redundant across all blocks into shared memory
    // Eliminates grid.sync between norm and gate/up projection
    __shared__ float s_post_normalized[SMEM_PAD_SIZE(HIDDEN_SIZE)];
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = g_activations[i];
            s_post_normalized[SMEM_PAD_IDX(i)] = v;
            local_sum_sq += v * v;
        }
        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __half2float(__ldg(post_norm_weight + i));
            s_post_normalized[SMEM_PAD_IDX(i)] = s_post_normalized[SMEM_PAD_IDX(i)] * rstd * w;
        }
        __syncthreads();
    }

    // Distributed residual update: each block writes its share of g_residual
    {
        int res_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int res_start = block_id * res_per_block;
        int res_end = min(res_start + res_per_block, HIDDEN_SIZE);
        for (int i = res_start + threadIdx.x; i < res_end; i += LDG_BLOCK_SIZE)
            g_residual[i] = g_activations[i];
    }

    // Gate + Up + SiLU: all blocks participate
    {
        int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
        int int_start = block_id * int_per_block;
        int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

        for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < int_end) {
                const int8_t* gate_q8_row = gate_q8_weight + m * HIDDEN_SIZE;
                const int8_t* up_q8_row = up_q8_weight + m * HIDDEN_SIZE;
                float gate_scale = __ldg(gate_q8_scale + m);
                float up_scale = __ldg(up_q8_scale + m);

                float gate_sum = 0.0f, up_sum = 0.0f;
                #pragma unroll 4
                for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                    int pk = SMEM_PAD_IDX(k);
                    float s0 = s_post_normalized[pk + 0], s1 = s_post_normalized[pk + 1],
                          s2 = s_post_normalized[pk + 2], s3 = s_post_normalized[pk + 3],
                          s4 = s_post_normalized[pk + 4], s5 = s_post_normalized[pk + 5],
                          s6 = s_post_normalized[pk + 6], s7 = s_post_normalized[pk + 7];
                    uint2 g_u2 = __ldg(reinterpret_cast<const uint2*>(gate_q8_row + k));
                    uint2 u_u2 = __ldg(reinterpret_cast<const uint2*>(up_q8_row + k));
                    ldg_dual_dot8_int8_float(
                        g_u2,
                        u_u2,
                        gate_scale,
                        up_scale,
                        s0,
                        s1,
                        s2,
                        s3,
                        s4,
                        s5,
                        s6,
                        s7,
                        gate_sum,
                        up_sum
                    );
                }
                gate_sum = ldg_warp_reduce_sum(gate_sum);
                up_sum   = ldg_warp_reduce_sum(up_sum);
                if (lane_id == 0)
                    g_mlp_intermediate[m] = ldg_silu(gate_sum) * up_sum;
            }
        }
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    // Restore mega-qwen's down-proj consumer staging using a dynamic shared-memory
    // page. On Qwen3-VL-2B the 6144-wide padded intermediate no longer fits in
    // the kernel's static shared allocation, but it does fit comfortably as a
    // runtime-configured shared page on sm_80.
    extern __shared__ float s_mlp[];
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += LDG_BLOCK_SIZE)
        s_mlp[i] = g_mlp_intermediate[i];
    __syncthreads();

    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < hid_end) {
            const int8_t* down_q8_row = down_q8_weight + m * INTERMEDIATE_SIZE;
            float down_scale = __ldg(down_q8_scale + m);
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
                int pk = k;
                uint2 d_u2 = __ldg(reinterpret_cast<const uint2*>(down_q8_row + k));
                sum += ldg_dot8_int8_float(
                    d_u2,
                    down_scale,
                    s_mlp[pk + 0],
                    s_mlp[pk + 1],
                    s_mlp[pk + 2],
                    s_mlp[pk + 3],
                    s_mlp[pk + 4],
                    s_mlp[pk + 5],
                    s_mlp[pk + 6],
                    s_mlp[pk + 7]
                );
            }
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2half(sum + g_residual[m]);
        }
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

constexpr int LDG_EOS_HOST_POLL_INTERVAL = 8;

// =============================================================================
// Main persistent decode kernel (all layers fused)
// =============================================================================

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
ldg_decode_kernel(
    int input_token_id,
    const __half* __restrict__ embed_weight,
    const LDGLayerWeights* __restrict__ layer_weights,
    const __half* __restrict__ final_norm_weight,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    __half* __restrict__ hidden_buffer,
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ attn_partial_max,
    float* __restrict__ attn_partial_sum,
    float* __restrict__ attn_partial_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    int num_layers,
    int position,
    int cache_len,
    const int* __restrict__ d_input_token_id,
    const int* __restrict__ d_position,
    const int* __restrict__ d_done_flag,
    int max_seq_len,
    float attn_scale,
    uint64_t* __restrict__ profiler_buffer,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_sense,
    unsigned int* __restrict__ kv_flag,
    unsigned int* __restrict__ attn_flag
) {
    if (d_done_flag != nullptr && *d_done_flag != 0) {
        return;
    }
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    if (d_position != nullptr) {
        position = *d_position;
        cache_len = position + 1;
        input_token_id = *d_input_token_id;
    }

    bool prof_on = (profiler_buffer != nullptr);

    AtomicGridSync grid{barrier_counter, barrier_sense, (unsigned int)gridDim.x, 0};

    // Embedding lookup
    sm_profiler_event_start(profiler_buffer, SM_PROF_EMBEDDING, prof_on);
    const __half* embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * LDG_BLOCK_SIZE)
        hidden_buffer[i] = __ldg(embed_row + i);
    sm_profiler_event_end(profiler_buffer, SM_PROF_EMBEDDING, prof_on);
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const LDGLayerWeights& w = layer_weights[layer];
        __half* layer_k = k_cache + layer * kv_cache_layer_stride;
        __half* layer_v = v_cache + layer * kv_cache_layer_stride;

        sm_profiler_event_start(profiler_buffer, SM_PROF_QKV_PROJ, prof_on);
        ldg_matvec_qkv(grid, hidden_buffer, w.input_layernorm_weight,
                        w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
                        w.q_proj_q8_weight, w.q_proj_q8_scale,
                        g_activations, g_residual, g_q, g_k, g_v,
                        profiler_buffer, prof_on);
        sm_profiler_event_end(profiler_buffer, SM_PROF_QKV_PROJ, prof_on);

        sm_profiler_event_start(profiler_buffer, SM_PROF_QK_NORM_ROPE, prof_on);
        ldg_qk_norm_rope_cache(grid, g_q, g_k, g_v,
                                w.q_norm_weight, w.k_norm_weight,
                                cos_table, sin_table,
                                layer_k, layer_v,
                                position, max_seq_len,
                                kv_flag, attn_flag, layer,
                                profiler_buffer, prof_on);
        sm_profiler_event_end(profiler_buffer, SM_PROF_QK_NORM_ROPE, prof_on);

        ldg_attention(grid, g_q, layer_k, layer_v, g_attn_out,
                       attn_partial_max, attn_partial_sum, attn_partial_out,
                       cache_len, max_seq_len, attn_scale,
                       attn_flag, layer,
                       profiler_buffer, prof_on);

        sm_profiler_event_start(profiler_buffer, SM_PROF_O_PROJ_MLP, prof_on);
        ldg_o_proj_postnorm_mlp(grid,
                                 w.o_proj_weight, w.post_attn_layernorm_weight,
                                 w.gate_proj_q8_weight, w.gate_proj_q8_scale,
                                 w.up_proj_q8_weight, w.up_proj_q8_scale,
                                 w.down_proj_q8_weight, w.down_proj_q8_scale,
                                 g_attn_out, g_residual, g_activations, g_mlp_intermediate,
                                 hidden_buffer,
                                 profiler_buffer, prof_on);
        sm_profiler_event_end(profiler_buffer, SM_PROF_O_PROJ_MLP, prof_on);
    }

    // Final RMSNorm (block 0 only)
    sm_profiler_event_start(profiler_buffer, SM_PROF_FINAL_NORM, prof_on);
    if (block_id == 0) {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __half2float(hidden_buffer[i]);
            g_activations[i] = v;
            local_sum_sq += v * v;
        }
        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float wt = __half2float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
    sm_profiler_event_end(profiler_buffer, SM_PROF_FINAL_NORM, prof_on);
}

// =============================================================================
// LM Head – full logits (for correctness testing)
// =============================================================================

__global__ void ldg_lm_head_logits(
    const float* __restrict__ hidden,
    const __half* __restrict__ weight,
    float* __restrict__ logits
) {
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE)
        s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE) {
        const __half* w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __half* w_ptr = reinterpret_cast<__half*>(&w_u2);
            sum += __half2float(w_ptr[0]) * s_hidden[k]   +
                   __half2float(w_ptr[1]) * s_hidden[k+1] +
                   __half2float(w_ptr[2]) * s_hidden[k+2] +
                   __half2float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0) logits[m] = sum;
    }
}

// =============================================================================
// LM Head – two-phase argmax (for token output)
// =============================================================================

__global__ void ldg_lm_head_phase1(
    const float* __restrict__ hidden,
    const __half* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs,
    const int* __restrict__ done_flag
) {
    if (done_flag != nullptr && *done_flag != 0) {
        return;
    }
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE)
        s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE) {
        const __half* w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __half* w_ptr = reinterpret_cast<__half*>(&w_u2);
            sum += __half2float(w_ptr[0]) * s_hidden[k]   +
                   __half2float(w_ptr[1]) * s_hidden[k+1] +
                   __half2float(w_ptr[2]) * s_hidden[k+2] +
                   __half2float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int   warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) { warp_max[warp_id] = local_max; warp_idx[warp_id] = local_max_idx; }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int   max_idx = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;
        for (int off = WARP_SIZE / 2; off > 0; off /= 2) {
            float ov = __shfl_down_sync(0xffffffff, max_val, off);
            int   oi = __shfl_down_sync(0xffffffff, max_idx, off);
            if (ov > max_val) { max_val = ov; max_idx = oi; }
        }
        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = max_val;
            block_max_idxs[blockIdx.x] = max_idx;
        }
    }
}

__global__ void ldg_lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks,
    const int* __restrict__ done_flag
) {
    if (done_flag != nullptr && *done_flag != 0) {
        return;
    }
    __shared__ float s_max_vals[1024];
    __shared__ int   s_max_idxs[1024];
    int tid = threadIdx.x;
    float local_max = -INFINITY;
    int   local_idx = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > local_max) { local_max = v; local_idx = block_max_idxs[i]; }
    }
    s_max_vals[tid] = local_max;
    s_max_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_max_vals[tid + s] > s_max_vals[tid]) {
            s_max_vals[tid] = s_max_vals[tid + s];
            s_max_idxs[tid] = s_max_idxs[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) *output_token = s_max_idxs[0];
}

// =============================================================================
// Launch function – decode + argmax only (no full logits)
// =============================================================================

__global__ void ldg_update_step(const int* __restrict__ lm_output,
                                int* __restrict__ d_token_id,
                                int* __restrict__ d_position,
                                int* __restrict__ output_log,
                                int* __restrict__ d_step_counter,
                                int* __restrict__ d_done_flag,
                                int eos_token_id) {
    if (*d_done_flag != 0) {
        return;
    }
    int tok = *lm_output;
    int step = *d_step_counter;
    *d_token_id = tok;
    *d_position = *d_position + 1;
    output_log[step] = tok;
    *d_step_counter = step + 1;
    if (tok == eos_token_id) {
        *d_done_flag = 1;
    }
}

__global__ void ldg_init_generate_state_from_host(
    int first_token_id,
    int start_position,
    int* __restrict__ d_token_id,
    int* __restrict__ d_position,
    int* __restrict__ d_step_counter,
    int* __restrict__ d_done_flag,
    int eos_token_id
) {
    *d_token_id = first_token_id;
    *d_position = start_position;
    *d_step_counter = 0;
    *d_done_flag = (first_token_id == eos_token_id) ? 1 : 0;
}

__global__ void ldg_init_generate_state_from_device(
    const int* __restrict__ first_token_device,
    int start_position,
    int* __restrict__ d_token_id,
    int* __restrict__ d_position,
    int* __restrict__ d_step_counter,
    int* __restrict__ d_done_flag,
    int eos_token_id
) {
    int first_token_id = *first_token_device;
    *d_token_id = first_token_id;
    *d_position = start_position;
    *d_step_counter = 0;
    *d_done_flag = (first_token_id == eos_token_id) ? 1 : 0;
}

static inline void ldg_configure_decode_kernel() {
    static bool configured = false;
    if (configured) {
        return;
    }
    cudaFuncSetAttribute(
        ldg_decode_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(LDG_STAGE_SMEM_BYTES)
    );
    cudaFuncSetAttribute(
        ldg_decode_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );
    configured = true;
}

extern "C" void launch_ldg_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
) {
    ldg_configure_decode_kernel();
    // Static device memory for atomic barriers
    static unsigned int* d_barrier_counter = nullptr;
    static unsigned int* d_barrier_sense = nullptr;
    static unsigned int* d_kv_flag = nullptr;
    static unsigned int* d_attn_flag = nullptr;
    static bool barrier_init = false;
    if (!barrier_init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        barrier_init = true;
    }
    // Reset barrier state before every kernel launch
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

    ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), LDG_STAGE_SMEM_BYTES, stream>>>(
        input_token_id,
        (const __half*)embed_weight,
        (const LDGLayerWeights*)layer_weights,
        (const __half*)final_norm_weight,
        (const __half*)cos_table,
        (const __half*)sin_table,
        (__half*)k_cache,
        (__half*)v_cache,
        (__half*)hidden_buffer,
        (float*)g_activations,
        (float*)g_residual,
        (float*)g_q,
        (float*)g_k,
        (float*)g_v,
        (float*)g_attn_out,
        (float*)attn_partial_max,
        (float*)attn_partial_sum,
        (float*)attn_partial_out,
        (float*)g_mlp_intermediate,
        (float*)g_normalized,
        num_layers,
        position,
        cache_len,
        nullptr,
        nullptr,
        nullptr,
        max_seq_len,
        attn_scale,
        profiler_buffer,
        d_barrier_counter,
        d_barrier_sense,
        d_kv_flag,
        d_attn_flag
    );

    // Argmax phase 1 + 2 (no full logits)
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __half*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs,
        nullptr
    );

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS,
        nullptr
    );
}

extern "C" void launch_ldg_generate_nosync(
    int first_token_id,
    int num_steps,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int* output_log,
    int* valid_steps,
    int eos_token_id,
    int num_blocks,
    int num_layers,
    int start_position,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
) {
    ldg_configure_decode_kernel();
    static unsigned int* d_barrier_counter = nullptr;
    static unsigned int* d_barrier_sense = nullptr;
    static unsigned int* d_kv_flag = nullptr;
    static unsigned int* d_attn_flag = nullptr;
    static int* d_mutable_position = nullptr;
    static int* d_mutable_token_id = nullptr;
    static int* d_output_token = nullptr;
    static int* d_step_counter = nullptr;
    static int* d_done_flag = nullptr;
    static bool init = false;
    if (!init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        cudaMalloc(&d_mutable_position, sizeof(int));
        cudaMalloc(&d_mutable_token_id, sizeof(int));
        cudaMalloc(&d_output_token, sizeof(int));
        cudaMalloc(&d_step_counter, sizeof(int));
        cudaMalloc(&d_done_flag, sizeof(int));
        init = true;
    }

    ldg_init_generate_state_from_host<<<1, 1, 0, stream>>>(
        first_token_id,
        start_position,
        d_mutable_token_id,
        d_mutable_position,
        d_step_counter,
        d_done_flag,
        eos_token_id
    );

    for (int step = 0; step < num_steps; ++step) {
        cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
        cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

        ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), LDG_STAGE_SMEM_BYTES, stream>>>(
            first_token_id,
            (const __half*)embed_weight,
            (const LDGLayerWeights*)layer_weights,
            (const __half*)final_norm_weight,
            (const __half*)cos_table,
            (const __half*)sin_table,
            (__half*)k_cache,
            (__half*)v_cache,
            (__half*)hidden_buffer,
            (float*)g_activations,
            (float*)g_residual,
            (float*)g_q,
            (float*)g_k,
            (float*)g_v,
            (float*)g_attn_out,
            (float*)attn_partial_max,
            (float*)attn_partial_sum,
            (float*)attn_partial_out,
            (float*)g_mlp_intermediate,
            (float*)g_normalized,
            num_layers,
            start_position,
            start_position + 1,
            d_mutable_token_id,
            d_mutable_position,
            d_done_flag,
            max_seq_len,
            attn_scale,
            profiler_buffer,
            d_barrier_counter,
            d_barrier_sense,
            d_kv_flag,
            d_attn_flag
        );

        ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
            (const float*)g_normalized,
            (const __half*)lm_head_weight,
            (float*)block_max_vals,
            (int*)block_max_idxs,
            d_done_flag
        );

        ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
            (const float*)block_max_vals,
            (const int*)block_max_idxs,
            d_output_token,
            LDG_LM_NUM_BLOCKS,
            d_done_flag
        );

        ldg_update_step<<<1, 1, 0, stream>>>(
            d_output_token, d_mutable_token_id, d_mutable_position,
            output_log, d_step_counter, d_done_flag, eos_token_id);

        if (((step + 1) % LDG_EOS_HOST_POLL_INTERVAL) == 0 || step == num_steps - 1) {
            int host_done = 0;
            cudaMemcpyAsync(&host_done, d_done_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (host_done != 0) {
                break;
            }
        }
    }

    if (valid_steps != nullptr) {
        cudaMemcpyAsync(valid_steps, d_step_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
}

extern "C" void launch_ldg_generate_from_device_nosync(
    const int* first_token_device,
    int num_steps,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int* output_log,
    int* valid_steps,
    int eos_token_id,
    int num_blocks,
    int num_layers,
    int start_position,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
) {
    ldg_configure_decode_kernel();
    static unsigned int* d_barrier_counter = nullptr;
    static unsigned int* d_barrier_sense = nullptr;
    static unsigned int* d_kv_flag = nullptr;
    static unsigned int* d_attn_flag = nullptr;
    static int* d_mutable_position = nullptr;
    static int* d_mutable_token_id = nullptr;
    static int* d_output_token = nullptr;
    static int* d_step_counter = nullptr;
    static int* d_done_flag = nullptr;
    static bool init = false;
    if (!init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        cudaMalloc(&d_mutable_position, sizeof(int));
        cudaMalloc(&d_mutable_token_id, sizeof(int));
        cudaMalloc(&d_output_token, sizeof(int));
        cudaMalloc(&d_step_counter, sizeof(int));
        cudaMalloc(&d_done_flag, sizeof(int));
        init = true;
    }

    ldg_init_generate_state_from_device<<<1, 1, 0, stream>>>(
        first_token_device,
        start_position,
        d_mutable_token_id,
        d_mutable_position,
        d_step_counter,
        d_done_flag,
        eos_token_id
    );

    for (int step = 0; step < num_steps; ++step) {
        cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
        cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

        ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), LDG_STAGE_SMEM_BYTES, stream>>>(
            0,
            (const __half*)embed_weight,
            (const LDGLayerWeights*)layer_weights,
            (const __half*)final_norm_weight,
            (const __half*)cos_table,
            (const __half*)sin_table,
            (__half*)k_cache,
            (__half*)v_cache,
            (__half*)hidden_buffer,
            (float*)g_activations,
            (float*)g_residual,
            (float*)g_q,
            (float*)g_k,
            (float*)g_v,
            (float*)g_attn_out,
            (float*)attn_partial_max,
            (float*)attn_partial_sum,
            (float*)attn_partial_out,
            (float*)g_mlp_intermediate,
            (float*)g_normalized,
            num_layers,
            start_position,
            start_position + 1,
            d_mutable_token_id,
            d_mutable_position,
            d_done_flag,
            max_seq_len,
            attn_scale,
            profiler_buffer,
            d_barrier_counter,
            d_barrier_sense,
            d_kv_flag,
            d_attn_flag
        );

        ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
            (const float*)g_normalized,
            (const __half*)lm_head_weight,
            (float*)block_max_vals,
            (int*)block_max_idxs,
            d_done_flag
        );

        ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
            (const float*)block_max_vals,
            (const int*)block_max_idxs,
            d_output_token,
            LDG_LM_NUM_BLOCKS,
            d_done_flag
        );

        ldg_update_step<<<1, 1, 0, stream>>>(
            d_output_token, d_mutable_token_id, d_mutable_position,
            output_log, d_step_counter, d_done_flag, eos_token_id);

        if (((step + 1) % LDG_EOS_HOST_POLL_INTERVAL) == 0 || step == num_steps - 1) {
            int host_done = 0;
            cudaMemcpyAsync(&host_done, d_done_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (host_done != 0) {
                break;
            }
        }
    }

    if (valid_steps != nullptr) {
        cudaMemcpyAsync(valid_steps, d_step_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
}

// =============================================================================
// Launch function – decode + full logits + argmax
// =============================================================================

extern "C" void launch_ldg_decode_with_logits(
    int input_token_id,
    int* output_token_id,
    float* logits_output,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
) {
    ldg_configure_decode_kernel();
    // Static device memory for atomic barriers
    static unsigned int* d_barrier_counter = nullptr;
    static unsigned int* d_barrier_sense = nullptr;
    static unsigned int* d_kv_flag = nullptr;
    static unsigned int* d_attn_flag = nullptr;
    static bool barrier_init = false;
    if (!barrier_init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        barrier_init = true;
    }
    // Reset barrier state before every kernel launch
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

    ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), LDG_STAGE_SMEM_BYTES, stream>>>(
        input_token_id,
        (const __half*)embed_weight,
        (const LDGLayerWeights*)layer_weights,
        (const __half*)final_norm_weight,
        (const __half*)cos_table,
        (const __half*)sin_table,
        (__half*)k_cache,
        (__half*)v_cache,
        (__half*)hidden_buffer,
        (float*)g_activations,
        (float*)g_residual,
        (float*)g_q,
        (float*)g_k,
        (float*)g_v,
        (float*)g_attn_out,
        (float*)attn_partial_max,
        (float*)attn_partial_sum,
        (float*)attn_partial_out,
        (float*)g_mlp_intermediate,
        (float*)g_normalized,
        num_layers,
        position,
        cache_len,
        nullptr,
        nullptr,
        nullptr,
        max_seq_len,
        attn_scale,
        profiler_buffer,
        d_barrier_counter,
        d_barrier_sense,
        d_kv_flag,
        d_attn_flag
    );

    // Full logits
    ldg_lm_head_logits<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __half*)lm_head_weight,
        logits_output
    );

    // Argmax phase 1 + 2
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __half*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs,
        nullptr
    );

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS,
        nullptr
    );
}
