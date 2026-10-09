#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 4;
constexpr int kHalf2PerLane = 2;
constexpr int kScalarsPerLane = 4;
constexpr int kKeyTile = 32;
constexpr int kVecElems = 8;  // 8 half = 16 bytes

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void causal_gqa_prefill_rows4_vec_kvtile32_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    float scale
) {
    const int q_head = blockIdx.y;
    const int row_base = blockIdx.x * kRowsPerBlock;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);

    if (q_head >= num_q_heads || tid >= kThreads) {
        return;
    }

    const int query_idx = row_base + warp_id;
    const bool row_active = warp_id < kRowsPerBlock && query_idx < seq_len;

    const int kv_group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / kv_group_size;
    const int max_query_idx = min(seq_len - 1, row_base + kRowsPerBlock - 1);

    __shared__ __align__(16) half k_tile[kKeyTile * kHeadDim];
    __shared__ __align__(16) half v_tile[kKeyTile * kHeadDim];
    __shared__ float alpha_s[kRowsPerBlock];
    __shared__ float beta_s[kRowsPerBlock];
    __shared__ float l_s[kRowsPerBlock];

    half2 q_reg[kHalf2PerLane];
    float2 out_acc[kHalf2PerLane];
    q_reg[0] = __float2half2_rn(0.0f);
    q_reg[1] = __float2half2_rn(0.0f);
    out_acc[0] = make_float2(0.0f, 0.0f);
    out_acc[1] = make_float2(0.0f, 0.0f);

    float m = -1.0e20f;
    float l = 0.0f;

    if (row_active) {
        const half* q_ptr = q + ((q_head * seq_len + query_idx) * kHeadDim) + lane * kScalarsPerLane;
        const half2* q_ptr_h2 = reinterpret_cast<const half2*>(q_ptr);
        q_reg[0] = q_ptr_h2[0];
        q_reg[1] = q_ptr_h2[1];
    }

    for (int tile_start = 0; tile_start <= max_query_idx; tile_start += kKeyTile) {
        const int tile_len = min(kKeyTile, max_query_idx - tile_start + 1);
        const int tile_elems = tile_len * kHeadDim;
        const int tile_vecs = tile_elems / kVecElems;

        const uint4* k_src_vec = reinterpret_cast<const uint4*>(
            k + ((kv_head * seq_len + tile_start) * kHeadDim)
        );
        const uint4* v_src_vec = reinterpret_cast<const uint4*>(
            v + ((kv_head * seq_len + tile_start) * kHeadDim)
        );
        uint4* k_dst_vec = reinterpret_cast<uint4*>(k_tile);
        uint4* v_dst_vec = reinterpret_cast<uint4*>(v_tile);

        for (int vec_idx = tid; vec_idx < tile_vecs; vec_idx += kThreads) {
            k_dst_vec[vec_idx] = k_src_vec[vec_idx];
            v_dst_vec[vec_idx] = v_src_vec[vec_idx];
        }
        __syncthreads();

        if (row_active && query_idx >= tile_start) {
            const int valid_keys = min(tile_len, query_idx - tile_start + 1);
            for (int local_key = 0; local_key < valid_keys; ++local_key) {
                const int base = local_key * kHeadDim + lane * kScalarsPerLane;
                const half2* k_h2 = reinterpret_cast<const half2*>(k_tile + base);

                const float2 prod0 = __half22float2(__hmul2(q_reg[0], k_h2[0]));
                const float2 prod1 = __half22float2(__hmul2(q_reg[1], k_h2[1]));
                float dot_local = prod0.x + prod0.y + prod1.x + prod1.y;

                const float dot = warp_reduce_sum(dot_local);
                if (lane == 0) {
                    const float score = dot * scale;
                    const float m_new = fmaxf(m, score);
                    const float alpha = (m > -1.0e19f) ? __expf(m - m_new) : 0.0f;
                    const float beta = __expf(score - m_new);
                    m = m_new;
                    l = l * alpha + beta;
                    alpha_s[warp_id] = alpha;
                    beta_s[warp_id] = beta;
                    l_s[warp_id] = l;
                }
                __syncwarp();

                const float alpha = alpha_s[warp_id];
                const float beta = beta_s[warp_id];
                const half2* v_h2 = reinterpret_cast<const half2*>(v_tile + base);
                const float2 v0 = __half22float2(v_h2[0]);
                const float2 v1 = __half22float2(v_h2[1]);

                out_acc[0].x = out_acc[0].x * alpha + beta * v0.x;
                out_acc[0].y = out_acc[0].y * alpha + beta * v0.y;
                out_acc[1].x = out_acc[1].x * alpha + beta * v1.x;
                out_acc[1].y = out_acc[1].y * alpha + beta * v1.y;
            }
        }

        __syncthreads();
    }

    if (row_active) {
        half* out_ptr = out + ((q_head * seq_len + query_idx) * kHeadDim) + lane * kScalarsPerLane;
        half2* out_ptr_h2 = reinterpret_cast<half2*>(out_ptr);
        const float denom = l_s[warp_id];
        const float inv_l = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        out_ptr_h2[0] = __floats2half2_rn(out_acc[0].x * inv_l, out_acc[0].y * inv_l);
        out_ptr_h2[1] = __floats2half2_rn(out_acc[1].x * inv_l, out_acc[1].y * inv_l);
    }
}

}  // namespace

void launch_causal_gqa_prefill_attention_v159g(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    float scale,
    cudaStream_t stream
) {
    const int num_query_tiles = (seq_len + kRowsPerBlock - 1) / kRowsPerBlock;
    dim3 grid(num_query_tiles, num_q_heads, 1);
    dim3 block(kThreads, 1, 1);
    causal_gqa_prefill_rows4_vec_kvtile32_kernel<<<grid, block, 0, stream>>>(
        q,
        k,
        v,
        out,
        num_q_heads,
        num_kv_heads,
        seq_len,
        scale
    );
}
