#include "flashdecode_api.h"

#define FLASHDECODE_ENABLE_MMA 1
#define FLASHDECODE_ENABLE_TCGEN05 0
#define FLASHDECODE_MMA_AICAS_SPLIT_POLICY 1
#define FLASHDECODE_NAMESPACE flashdecode_mma_impl
#include "flashdecode-test.cu"

namespace FLASHDECODE_NAMESPACE {
template <typename T, int STAGES, int WARPS_PB, int NPACK, int Kdim>
struct FlashDecodeSharedStorage
{
    static constexpr float inv_sqrt_128 = 0.0883883476482f;
    static constexpr float neg_inf = -FLT_MAX;

    T Qs[NPACK][Kdim];
    T KVs[STAGES][WARPS_PB][16][Kdim];
    T As[NPACK][WARPS_PB][16];
    float Ls[NPACK][WARPS_PB][16];

    float m_w[NPACK];
    float l_w[NPACK];
    float alpha_w[NPACK];

    static __device__ __forceinline__ float warp_reduce_max(float val)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_down_sync(0xffffffffu, val, offset);
            val = other > val ? other : val;
        }
        return val;
    }

    static __device__ __forceinline__ float warp_reduce_sum(float val)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return val;
    }

    static __device__ __forceinline__ float warp_reduce_max_broadcast(float val)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_down_sync(0xffffffffu, val, offset);
            val = other > val ? other : val;
        }
        return __shfl_sync(0xffffffffu, val, 0);
    }

    static __device__ __forceinline__ float warp_reduce_sum_broadcast(float val)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        return __shfl_sync(0xffffffffu, val, 0);
    }
};



template <typename T, int STAGES, int WARPS_PB, int NPACK, int Kdim, int S_tile>
static __global__ void flash_decode_h128_splitS(
    T const *__restrict__ Q,
    T const *__restrict__ K_base,
    T const *__restrict__ V_base,
    T const *const *__restrict__ K_ptrs,
    T const *const *__restrict__ V_ptrs,
    int const *__restrict__ seq_lengths,
    float *__restrict__ red_av,
    float *__restrict__ red_m,
    float *__restrict__ red_l,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S_total,
    int use_kv_ptrs,
    int S_split,
    int virtual_q)
{
    const int g_pack = blockIdx.x; // Group维度，这里是G=2，即两个Q头对应一个KV头
    const int b = blockIdx.y; // Batchsize维度，这里是1
    const int split_id = blockIdx.z / H_kv; // 表示对sequence维度进行切分的第几个split，sequence维度被切分成了多个split，每个split包含S_split个token
    const int kv_id = blockIdx.z % H_kv; // 表示第几个KV头

    const int lane = threadIdx.x; // warp内，每个warp32个线程
    const int warp_id = threadIdx.y; // warp id，每个warp
    const int tid = lane + 32 * warp_id;

    const int G_pack_base = g_pack * NPACK;

    const int cols_this = i_max(0, i_min(NPACK, G - G_pack_base));
    if (cols_this <= 0)
        return;

    const int real_b = virtual_q > 0 ? (b / virtual_q) : b;
    const int qi = virtual_q > 0 ? (b - real_b * virtual_q) : 0;
    int S_cur = seq_lengths ? (seq_lengths[b] + 1) : S_total;
    if (virtual_q > 0)
        S_cur = S_total - virtual_q + qi + 1;

    const int s_begin = split_id * S_split;
    const int s_end = i_min(s_begin + S_split, S_cur);
    if (s_begin >= s_end)
        return;

    const int tiles = (s_end - s_begin + S_tile - 1) / S_tile;
    const int total_stages = tiles * 2;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    using Smem = FlashDecodeSharedStorage<T, STAGES, WARPS_PB, NPACK, Kdim>;
    Smem &S = *(Smem *)(smem_raw);

    if (tid < NPACK)
    {
        S.m_w[tid] = Smem::neg_inf;
        S.l_w[tid] = 0.f;
    }
    __syncthreads();

    auto prefetch_KV_tile = [&](int stage, int s_base)
    {
        int r = lane & 15;
        int seg = lane >> 4;
#pragma unroll
        for (int blk = 0; blk < 8; ++blk)
        {
            int c0 = blk * 16 + seg * 8;
            int srow = s_base + warp_id * 16 + r;
            T *kvdst = &S.KVs[stage % STAGES][warp_id][r][c0];
            if (srow < s_end)
            {
                const T *KVbase = nullptr;
                if (use_kv_ptrs)
                {
                    int h0 = kv_id * G;
                    KVbase = (stage & 1 ? V_ptrs[b * H + h0] : K_ptrs[b * H + h0]) + srow * Kdim + c0;
                }
                else
                {
                    size_t base_off = ((size_t)real_b * H_kv + kv_id) * (size_t)S_total * Kdim + (size_t)srow * Kdim + c0;
                    KVbase = (stage & 1 ? V_base : K_base) + base_off;
                }
                cp_async<16>(kvdst, KVbase);
            }
            else
            {
                clear_16B(kvdst);
            }
        }
    };

    auto prefetch_far = [&](int cur_stage) -> bool
    {
        int far_stage = cur_stage + (STAGES - 1);
        if (far_stage < total_stages)
        {
            int far_s_tile = far_stage / 2;
            int s_base = s_begin + far_s_tile * S_tile;
            prefetch_KV_tile(far_stage, s_base);
            cp_async_commit();
            return true;
        }
        return false;
    };

    auto load_K_Aregs_16x16 = [&](T *k_tile, uint32_t (&areg)[4])
    {
        uint2 offset = ldmatrix_x4_offset(lane);
        T *base = k_tile + offset.x * Kdim + offset.y;
        ldmatrix_sync_x4(base, areg[0], areg[1], areg[2], areg[3]);
    };

    auto load_V_Aregs_16x16 = [&](T *v_tile, uint32_t (&areg)[4])
    {
        uint2 offset = ldmatrix_x4_offset<true>(lane);
        T *base = v_tile + offset.x * Kdim + offset.y;
        ldmatrix_sync_x4<true>(base, areg[0], areg[1], areg[2], areg[3]);
    };

    auto load_Q_Bregs_16x8 = [&](int kk, uint32_t (&breg)[2])
    {
        T *base = &S.Qs[lane % 8][kk * 16] + ((lane / 8) % 2) * 8;
        ldmatrix_sync_x2(base, breg[0], breg[1]);
    };

    auto load_A_Bregs_16x8 = [&](int kk, uint32_t (&breg)[2])
    {
        T *base = &S.As[lane % 8][kk][0] + ((lane / 8) % 2) * 8;
        ldmatrix_sync_x2(base, breg[0], breg[1]);
    };

    {
        int c = tid / 16;
        int off8 = (tid % 16) * 8;
        T *dst = &S.Qs[c][off8];
        if (c < cols_this)
        {
            int g = G_pack_base + c;
            const T *qsrc = Q + (((b * H_kv + kv_id) * Q_stride + g) * Kdim);
            cp_async<16>(dst, qsrc + off8);
        }
        else
        {
            clear_16B(dst);
        }
    }

    int warm = i_min(STAGES - 1, total_stages);
    for (int warm_stage = 0; warm_stage < warm; ++warm_stage)
    {
        int s_tile = warm_stage / 2;
        int s = s_begin + s_tile * S_tile;
        prefetch_KV_tile(warm_stage, s);
        cp_async_commit();
    }

    if constexpr (STAGES >= 2)
        cp_async_wait<1>();
    else
        cp_async_wait<0>();
    __syncthreads();

    float AVs[2][4]{0};
    int stage = 0;

    for (int s_tile = 0; s_tile < tiles; s_tile++)
    {
        int s_tile_len = i_min(s_end - (s_begin + s_tile * S_tile), S_tile);
        bool prefetched = prefetch_far(stage);

        float d_qk[4]{0.f, 0.f, 0.f, 0.f};
        if constexpr (dtype_of<T>::dtype == DataType::f16)
        {
            uint32_t d_qk_acc[2]{0u, 0u};
#pragma unroll
            for (int kk = 0; kk < 8; ++kk)
            {
                uint32_t A_reg[4], B_reg[2];
                load_K_Aregs_16x16(&S.KVs[stage % STAGES][warp_id][0][kk * 16], A_reg);
                load_Q_Bregs_16x8(kk, B_reg);
                mma_sync_m16n8k16_row_col_f16_f16_accu(
                    d_qk_acc[0], d_qk_acc[1],
                    A_reg[0], A_reg[1], A_reg[2], A_reg[3],
                    B_reg[0], B_reg[1]);
            }
            __half2 acc0 = *(__half2 *)(&d_qk_acc[0]);
            __half2 acc1 = *(__half2 *)(&d_qk_acc[1]);
            d_qk[0] = float(acc0.x);
            d_qk[1] = float(acc0.y);
            d_qk[2] = float(acc1.x);
            d_qk[3] = float(acc1.y);
        }
        else
        {
#pragma unroll
            for (int kk = 0; kk < 8; ++kk)
            {
                uint32_t A_reg[4], B_reg[2];
                load_K_Aregs_16x16(&S.KVs[stage % STAGES][warp_id][0][kk * 16], A_reg);
                load_Q_Bregs_16x8(kk, B_reg);
                mma_sync_m16n8k16_row_col_f16bf16_f32_accu<dtype_of<T>::dtype>(
                    d_qk[0], d_qk[1], d_qk[2], d_qk[3],
                    A_reg[0], A_reg[1], A_reg[2], A_reg[3],
                    B_reg[0], B_reg[1]);
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int c = (lane % 4) * 2 + (i & 1);
            int r = (lane / 4) + (i / 2) * 8;
            if (c < cols_this)
            {
                if (warp_id * 16 + r < s_tile_len)
                    S.Ls[c][warp_id][r] = d_qk[i] * Smem::inv_sqrt_128;
                else
                    S.Ls[c][warp_id][r] = Smem::neg_inf;
            }
        }
        __syncthreads();

        for (int i = warp_id; i < cols_this; i += WARPS_PB)
        {
            int c = lane / 16;
            int r = lane % 16;
            bool row_valid_0 = lane < s_tile_len;
            bool row_valid_1 = lane + 32 < s_tile_len;
            float v0 = S.Ls[i][c][r];
            float v1 = S.Ls[i][c + 2][r];
            float tmax = Smem::neg_inf;
            if (row_valid_0)
                tmax = v0;
            if (row_valid_1)
                tmax = fmaxf(tmax, v1);
            tmax = S.warp_reduce_max_broadcast(tmax);
            float m_prev = S.m_w[i];
            float m_new = fmaxf(tmax, m_prev);
            float e0 = row_valid_0 ? __expf(v0 - m_new) : 0.f;
            float e1 = row_valid_1 ? __expf(v1 - m_new) : 0.f;
            S.As[i][c][r] = T(e0);
            S.As[i][c + 2][r] = T(e1);
            float sum = e0 + e1;
            sum = S.warp_reduce_sum(sum);
            if (lane == 0)
            {
                float alpha = (m_prev == Smem::neg_inf) ? 0.f : __expf(m_prev - m_new);
                S.alpha_w[i] = alpha;
                S.m_w[i] = m_new;
                S.l_w[i] = S.l_w[i] * alpha + sum;
            }
        }

        stage += 1;
        if (prefetched)
        {
            if constexpr (STAGES >= 2)
                cp_async_wait<1>();
            else
                cp_async_wait<0>();
            prefetch_far(stage);
        }
        else
            cp_async_wait<0>();
        __syncthreads();

        for (int i = 0; i < 4; i++)
        {
            int c = (lane % 4) * 2 + (i & 1);
            if (c < cols_this)
            {
                AVs[0][i] *= S.alpha_w[c];
                AVs[1][i] *= S.alpha_w[c];
            }
        }

        if constexpr (dtype_of<T>::dtype == DataType::f16)
        {
            uint32_t AV_acc[2][2];
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                __half h0 = __float2half(AVs[rr][0]);
                __half h1 = __float2half(AVs[rr][1]);
                __half h2 = __float2half(AVs[rr][2]);
                __half h3 = __float2half(AVs[rr][3]);
                __half2 packed0 = __halves2half2(h0, h1);
                __half2 packed1 = __halves2half2(h2, h3);
                AV_acc[rr][0] = *reinterpret_cast<uint32_t *>(&packed0);
                AV_acc[rr][1] = *reinterpret_cast<uint32_t *>(&packed1);
            }
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
            {
                if (kk * 16 >= s_tile_len)
                    break;
                uint32_t B_reg[2];
                load_A_Bregs_16x8(kk, B_reg);
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
                    uint32_t A_reg[4];
                    load_V_Aregs_16x16(&S.KVs[stage % STAGES][kk][0][warp_id * 32 + rr * 16], A_reg);
                    mma_sync_m16n8k16_row_col_f16_f16_accu(
                        AV_acc[rr][0], AV_acc[rr][1],
                        A_reg[0], A_reg[1], A_reg[2], A_reg[3],
                        B_reg[0], B_reg[1]);
                }
            }
#pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                __half2 packed0 = *reinterpret_cast<__half2 *>(&AV_acc[rr][0]);
                __half2 packed1 = *reinterpret_cast<__half2 *>(&AV_acc[rr][1]);
                AVs[rr][0] = __half2float(__low2half(packed0));
                AVs[rr][1] = __half2float(__high2half(packed0));
                AVs[rr][2] = __half2float(__low2half(packed1));
                AVs[rr][3] = __half2float(__high2half(packed1));
            }
        }
        else
        {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
            {
                if (kk * 16 >= s_tile_len)
                    break;
                uint32_t B_reg[2];
                load_A_Bregs_16x8(kk, B_reg);
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
                    uint32_t A_reg[4];
                    load_V_Aregs_16x16(&S.KVs[stage % STAGES][kk][0][warp_id * 32 + rr * 16], A_reg);
                    mma_sync_m16n8k16_row_col_f16bf16_f32_accu<dtype_of<T>::dtype>(
                        AVs[rr][0], AVs[rr][1], AVs[rr][2], AVs[rr][3],
                        A_reg[0], A_reg[1], A_reg[2], A_reg[3],
                        B_reg[0], B_reg[1]);
                }
            }
        }

        stage += 1;
        if (prefetched)
        {
            if constexpr (STAGES >= 2)
                cp_async_wait<1>();
            else
                cp_async_wait<0>();
            __syncthreads();
        }
    }

    uint32_t write_base = ((split_id * B + b) * H_kv + kv_id) * ((G + NPACK - 1) / NPACK) + g_pack;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int c = (lane % 4) * 2 + (i & 1);
        int r = (lane / 4) + (i / 2) * 8;
        if (c < cols_this)
        {
            red_av[(write_base * NPACK + c) * Kdim + warp_id * 32 + r] = AVs[0][i];
            red_av[(write_base * NPACK + c) * Kdim + warp_id * 32 + r + 16] = AVs[1][i];
        }
    }
    if (tid < cols_this)
    {
        red_m[write_base * NPACK + tid] = S.m_w[tid];
        red_l[write_base * NPACK + tid] = S.l_w[tid];
    }
}

} // namespace FLASHDECODE_NAMESPACE

extern "C" void flashdecode_mma_init()
{
    using namespace flashdecode_mma_impl;
    FlashDecode<fp16>::init_kernel();
}

extern "C" void flashdecode_mma_init_bf16()
{
    using namespace flashdecode_mma_impl;
    FlashDecode<bf16>::init_kernel();
}

extern "C" size_t flashdecode_mma_workspace_size(int B, int S, int H_kv, int G)
{
    using namespace flashdecode_mma_impl;
    return FlashDecode<fp16>::workspace_size(B, S, H_kv, G);
}

extern "C" void flashdecode_mma_run(
    const __half *q,
    const __half *const *k_ptrs,
    const __half *const *v_ptrs,
    const int *seq_lengths,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int H,
    int S,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<fp16>::run<FlashDecodeKernelKind::mma_sm80>(
        q, k_ptrs, v_ptrs, seq_lengths, out, temp, B, H_kv, G, H, S, stream);
}

extern "C" void flashdecode_mma_run_qstride(
    const __half *q,
    const __half *const *k_ptrs,
    const __half *const *v_ptrs,
    const int *seq_lengths,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<fp16>::run_qstride<FlashDecodeKernelKind::mma_sm80>(
        q, k_ptrs, v_ptrs, seq_lengths, out, temp, B, H_kv, G, Q_stride, H, S, stream);
}

extern "C" void flashdecode_mma_run_qstride_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *const *k_ptrs,
    const __nv_bfloat16 *const *v_ptrs,
    const int *seq_lengths,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<bf16>::run_qstride<FlashDecodeKernelKind::mma_sm80>(
        q, k_ptrs, v_ptrs, seq_lengths, out, temp, B, H_kv, G, Q_stride, H, S, stream);
}

extern "C" void flashdecode_mma_run_qstride_contig(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    const int *seq_lengths,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<fp16>::run_qstride_contig<FlashDecodeKernelKind::mma_sm80>(
        q, k_base, v_base, seq_lengths, out, temp, B, H_kv, G, Q_stride, H, S, stream);
}

extern "C" void flashdecode_mma_run_qstride_contig_noseqlens(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<fp16>::run_qstride_contig_noseqlens<FlashDecodeKernelKind::mma_sm80>(
        q, k_base, v_base, out, temp, B, H_kv, G, Q_stride, H, S, stream);
}

extern "C" void flashdecode_mma_run_qstride_contig_noseqlens_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k_base,
    const __nv_bfloat16 *v_base,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<bf16>::run_qstride_contig_noseqlens<FlashDecodeKernelKind::mma_sm80>(
        q, k_base, v_base, out, temp, B, H_kv, G, Q_stride, H, S, stream);
}

extern "C" void flashdecode_mma_run_qstride_contig_causal(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    int virtual_q,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<fp16>::run_qstride_contig_causal<FlashDecodeKernelKind::mma_sm80>(
        q, k_base, v_base, out, temp, B, H_kv, G, Q_stride, H, S, virtual_q, stream);
}

extern "C" void flashdecode_mma_run_qstride_contig_causal_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k_base,
    const __nv_bfloat16 *v_base,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    int virtual_q,
    cudaStream_t stream)
{
    using namespace flashdecode_mma_impl;
    FlashDecode<bf16>::run_qstride_contig_causal<FlashDecodeKernelKind::mma_sm80>(
        q, k_base, v_base, out, temp, B, H_kv, G, Q_stride, H, S, virtual_q, stream);
}
