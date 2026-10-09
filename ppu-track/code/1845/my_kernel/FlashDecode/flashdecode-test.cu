#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <cfloat>

#ifndef FLASHDECODE_ENABLE_MMA
#define FLASHDECODE_ENABLE_MMA 1
#endif

#ifndef FLASHDECODE_ENABLE_TCGEN05
#define FLASHDECODE_ENABLE_TCGEN05 1
#endif

namespace FLASHDECODE_NAMESPACE {

using fp16 = __half;
using bf16 = __nv_bfloat16;

enum class DataType : uint32_t
{
    f16 = 1,
    bf16 = 2
};

template <typename T>
struct dtype_of;

template <>
struct dtype_of<fp16>
{
    static constexpr DataType dtype = DataType::f16;
};

template <>
struct dtype_of<bf16>
{
    static constexpr DataType dtype = DataType::bf16;
};

#include "sm_80.h"

__host__ __device__ __forceinline__ int i_max(int a, int b)
{
    return a > b ? a : b;
}

__host__ __device__ __forceinline__ int i_min(int a, int b)
{
    return a < b ? a : b;
}


enum class UmmaMajor : uint8_t
{
    K = 0,
    MN = 1
};

enum class UmmaLayoutType : uint8_t
{
    SWIZZLE_NONE = 0,
    SWIZZLE_32B = 6
};

template <int M, int N, int K>
struct SharedStorage
{
    static_assert((K % 8) == 0, "K must be a multiple of 8 for u128 staging");
    static constexpr int kK_u128 = K / 8;
    alignas(128) uint4 A_u128[kK_u128 * M];
    alignas(128) uint4 B_u128[kK_u128 * N];
    alignas(16) uint64_t mma_barrier;
    alignas(16) uint32_t tmem_base_ptr;
};

union SmemDescriptor
{
    uint64_t desc = 0;
    struct
    {
        uint16_t start_address : 14, : 2;
        uint16_t leading_byte_offset : 14, : 2;
        uint16_t stride_byte_offset : 14, version : 2;
        uint8_t : 1, base_offset : 3, lbo_mode : 1, : 3;
        uint8_t : 5, layout_type : 3;
    };
    __host__ __device__ constexpr operator uint64_t() const noexcept { return desc; }
};

union InstrDescriptor
{
    uint32_t desc = 0;
    struct
    {
        uint16_t sparse_id2 : 2;
        uint16_t sparse_flag : 1;
        uint16_t saturate : 1;
        uint16_t c_format : 2;
        uint16_t : 1;
        uint16_t a_format : 3;
        uint16_t b_format : 3;
        uint16_t a_negate : 1;
        uint16_t b_negate : 1;
        uint16_t a_major : 1;
        uint16_t b_major : 1;
        uint16_t n_dim : 6;
        uint16_t : 1;
        uint16_t m_dim : 5;
        uint16_t : 1;
        uint16_t max_shift : 2;
    };
    __host__ __device__ constexpr explicit operator uint32_t() const noexcept { return desc; }
};


__device__ __forceinline__ SmemDescriptor make_smem_desc_kmajor(
    void const *smem_ptr_u128,
    uint16_t leading_u128,
    uint16_t stride_u128,
    UmmaLayoutType layout_type = UmmaLayoutType::SWIZZLE_NONE)
{
    SmemDescriptor d{};
    d.version = 1;
    d.layout_type = static_cast<uint8_t>(layout_type);
    uint32_t start = cvt_shared_ptr(const_cast<void *>(smem_ptr_u128));
    d.start_address = static_cast<uint16_t>(start >> 4);
    d.leading_byte_offset = leading_u128;
    d.stride_byte_offset = stride_u128;
    d.base_offset = 0;
    d.lbo_mode = 0;
    return d;
}

template <int M, int N, typename T1, typename T2>
__device__ __forceinline__ uint64_t make_mma_instr_desc_m64n16_kmajor()
{
    InstrDescriptor id{};
    constexpr int fmt_out = std::is_same_v<T2, float> ? 1 : 0;
    id.c_format = fmt_out;

    constexpr int fmt_in = 0;

    id.a_format = fmt_in;
    id.b_format = fmt_in;
    id.a_major = static_cast<uint8_t>(UmmaMajor::K);
    id.b_major = static_cast<uint8_t>(UmmaMajor::K);
    id.m_dim = (M >> 4);
    id.n_dim = (N >> 3);
    return (static_cast<uint64_t>(static_cast<uint32_t>(id)) << 32);
}

template <typename T1, typename T2>
__device__ __forceinline__ uint64_t make_mma_instr_desc_m64n8_kmajor()
{
    return make_mma_instr_desc_m64n16_kmajor<64, 8, T1, T2>();
}

enum class FlashDecodeKernelKind : uint8_t
{
    mma_sm80 = 0,
};

template <typename T, int STAGES, int WARPS_PB, int NPACK, int Kdim>
struct FlashDecodeSharedStorage;

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
    int virtual_q);


struct FlashDecodeReduceOps
{
    static constexpr float neg_inf = -FLT_MAX;

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


template <typename T, int STAGES, int WARPS_PB, int NPACK, int Kdim>
static __global__ void flash_decode_reduce_h128_splitS(
    const float *__restrict__ red_av,
    const float *__restrict__ red_m,
    const float *__restrict__ red_l,
    T *__restrict__ out,
    int const *__restrict__ seq_lengths,
    int B, int H_kv, int G, int S_total, int S_split, int virtual_q)
{
    const int g_pack = blockIdx.x;
    const int b = blockIdx.y;
    const int kv_id = blockIdx.z;

    const int lane = threadIdx.x;
    const int c = threadIdx.y;

    const int g_packs = (G + NPACK - 1) / NPACK;
    const int G_pack0 = g_pack * NPACK;
    const int cols_this = i_max(0, i_min(NPACK, G - G_pack0));
    if (c >= cols_this)
        return;

    using ReduceOps = FlashDecodeReduceOps;

    int S_cur = seq_lengths ? (seq_lengths[b] + 1) : S_total;
    if (virtual_q > 0)
    {
        const int qi = b % virtual_q;
        S_cur = S_total - virtual_q + qi + 1;
    }
    int valid_splits = (S_cur + S_split - 1) / S_split;

    auto base_idx = [&](int split, int col) -> size_t
    { return (((split * B + b) * H_kv + kv_id) * g_packs + g_pack) * NPACK + col; };

    // Fast path: no cross-split reduction needed.
    if (valid_splits == 1)
    {
        size_t bidx = base_idx(0, c);
        float l0 = red_l[bidx];
        float inv_l = (l0 > 0.f) ? (1.f / l0) : 0.f;
        const int g = G_pack0 + c;
        for (int r = lane; r < Kdim; r += 32)
        {
            float out_val = red_av[bidx * Kdim + r] * inv_l;
            if (g < G)
            {
                size_t out_idx = (((size_t)b * H_kv + kv_id) * G + g) * Kdim + r;
                out[out_idx] = T(out_val);
            }
        }
        return;
    }

    float local_max = ReduceOps::neg_inf;
    float local_sum = 0.f;
    for (int s = lane; s < valid_splits; s += 32)
    {
        size_t idx = base_idx(s, c);
        local_max = fmaxf(local_max, red_m[idx]);
    }
    float Mc = ReduceOps::warp_reduce_max_broadcast(local_max);
    const bool has_valid = (valid_splits > 0) && (Mc > ReduceOps::neg_inf);
    if (has_valid)
    {
        for (int s = lane; s < valid_splits; s += 32)
        {
            size_t idx = base_idx(s, c);
            local_sum += __expf(red_m[idx] - Mc) * red_l[idx];
        }
    }
    float Lc = ReduceOps::warp_reduce_sum_broadcast(local_sum);

    for (int r = lane; r < Kdim; r += 32)
    {
        float acc = 0.f;
        if (has_valid)
        {
            for (int s = 0; s < valid_splits; ++s)
            {
                size_t bidx = base_idx(s, c);
                float av = red_av[bidx * Kdim + r];
                acc += __expf(red_m[bidx] - Mc) * av;
            }
        }
        float out_val = (Lc > 0.f) ? (acc / Lc) : 0.f;

        const int g = G_pack0 + c;
        if (g < G)
        {
            size_t out_idx = (((size_t)b * H_kv + kv_id) * G + g) * Kdim + r;
            out[out_idx] = T(out_val);
        }
    }
}

template <typename T>
struct FlashDecode
{
    static constexpr int STAGES_MMA = 3;
    static constexpr int STAGES_TCGEN = 4;
    static constexpr int BANKS_TCGEN = 3;
    static constexpr int WARPS_PB_MMA = 4;
    static constexpr int WARPS_PB_TCGEN = 4;
    static constexpr int NPACK = 4;
    static constexpr int Kdim = 128;
    static constexpr int S_iter_tile_mma = WARPS_PB_MMA * 16;
    static constexpr int S_iter_tile_tcgen = WARPS_PB_TCGEN * 16;
    static constexpr int S_tile = 128;
    static constexpr int S_tile_tcgen = 256;
    static constexpr int S_tile_min = (S_tile < S_tile_tcgen) ? S_tile : S_tile_tcgen;
    static constexpr int TCGEN_SPLIT_WAVES = 3;

    static int get_sm_count_cached()
    {
        static int sm_count = 0;
        if (sm_count <= 0)
        {
            int device = 0;
            cudaGetDevice(&device);
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        }
        return sm_count;
    }

    static uint2 get_split_config(int S, int B, int H_kv, int G, int sm_count = -1, int s_tile = S_tile_tcgen)
    {
        if (sm_count <= 0)
            sm_count = get_sm_count_cached();
        const int safe_S = i_max(S, 1);
        const int g_packs = i_max((G + NPACK - 1) / NPACK, 1);
        const int parallelism = i_max(1, i_max(B, 1) * i_max(H_kv, 1) * g_packs);
        int max_splits = sm_count > 0 ? (sm_count + parallelism - 1) / parallelism : 0;
        max_splits = i_max(max_splits, 1);
        int tile = i_max(s_tile, 1);
        int splits = (safe_S + tile - 1) / tile;

        if (splits > max_splits)
        {
            tile = (safe_S + max_splits - 1) / max_splits;
            tile = ((tile + s_tile - 1) / s_tile) * s_tile;
            if (tile > safe_S)
                tile = safe_S;
            splits = (safe_S + tile - 1) / tile;
            if (splits > max_splits)
            {
                splits = max_splits;
                tile = (safe_S + splits - 1) / splits;
            }
        }

        return make_uint2(splits, tile);
    }

    static int select_tcgen_split_tile(int S, int B, int H_kv, int G)
    {
        const int safe_S = i_max(S, 1);
        if (const char *tile_env = std::getenv("FD_TCGEN_S_TILE"))
        {
            int forced_tile = std::atoi(tile_env);
            if (forced_tile > 0)
                return i_min(i_max(forced_tile, 64), safe_S);
        }
        const int g_packs = i_max((G + NPACK - 1) / NPACK, 1);
        const int parallelism = i_max(1, i_max(B, 1) * i_max(H_kv, 1) * g_packs);

        // Keep baseline behavior at B=1,H_kv=8,G=4,S=1024 (tile=256),
        // but reduce split overhead when parallelism is already high.
        int tile = S_tile_tcgen;
        if (parallelism >= 16)
            tile = 1024;
        if (parallelism >= 32)
            tile = 2048;

        // For short sequence + low parallelism, increase blocks to improve occupancy.
        if (safe_S <= 768 && parallelism <= 8)
            tile = 128;

        tile = i_min(i_max(tile, 64), safe_S);
        return tile;
    }

    static int select_mma_split_tile(int S, int B, int H_kv, int G)
    {
        const int safe_S = i_max(S, 1);
        if (const char *tile_env = std::getenv("FD_MMA_S_TILE"))
        {
            int forced_tile = std::atoi(tile_env);
            if (forced_tile > 0)
                return i_min(i_max(forced_tile, 32), safe_S);
        }
        int tile = 32;

#if defined(FLASHDECODE_MMA_AICAS_SPLIT_POLICY)
        // AICAS sequence distribution tuning for MMA path:
        // target shape B=1, H_kv=8, G=2, K=128 with seq in [595, 2067].
        if (B == 1 && H_kv == 8 && G == 2)
        {
            if (safe_S <= 1024)
                tile = 64;
            else
                tile = 128;
        }
#else
        // Legacy single-shape tuning.
        if (B == 1 && H_kv == 8 && G == 2 && safe_S == 1024)
            tile = 64;
#endif

        tile = i_min(i_max(tile, 32), safe_S);
        return tile;
    }

    static size_t workspace_size(int B, int S, int H_kv, int G)
    {
        int sm_for_split = get_sm_count_cached();
        int s_tile_for_split = S_tile_min;
#if FLASHDECODE_ENABLE_MMA && !FLASHDECODE_ENABLE_TCGEN05
        sm_for_split *= 16;
        s_tile_for_split = select_mma_split_tile(S, B, H_kv, G);
#elif FLASHDECODE_ENABLE_TCGEN05 && !FLASHDECODE_ENABLE_MMA
        s_tile_for_split = select_tcgen_split_tile(S, B, H_kv, G);
#endif
        const uint2 split_cfg = get_split_config(S, B, H_kv, G, sm_for_split, s_tile_for_split);
        const int Splits = static_cast<int>(split_cfg.x);
        const int g_packs = (G + NPACK - 1) / NPACK;
        size_t red_items = (size_t)Splits * B * H_kv * g_packs * NPACK;
        size_t red_av_elems = red_items * Kdim;
        return sizeof(float) * (red_av_elems + red_items * 2);
    }

    static void init_kernel()
    {
#if FLASHDECODE_ENABLE_MMA
        size_t smem_bytes = sizeof(FlashDecodeSharedStorage<T, STAGES_MMA, WARPS_PB_MMA, NPACK, Kdim>);
        cudaFuncSetAttribute(
            flash_decode_h128_splitS<T, STAGES_MMA, WARPS_PB_MMA, NPACK, Kdim, S_iter_tile_mma>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
#endif
#if FLASHDECODE_ENABLE_TCGEN05
        size_t smem_tcgen_bytes = sizeof(FlashDecodeSharedStorageTcgen05<T, BANKS_TCGEN, WARPS_PB_TCGEN, NPACK, Kdim>);
        cudaFuncSetAttribute(
            flash_decode_h128_splitS_tcgen05<T, STAGES_TCGEN, BANKS_TCGEN, WARPS_PB_TCGEN, NPACK, Kdim, S_iter_tile_tcgen>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_tcgen_bytes);
#endif
    }

    template <FlashDecodeKernelKind Kind>
    static void run_backend(
        T const *q,
        T const *k_base,
        T const *v_base,
        T const *const *k_ptrs,
        T const *const *v_ptrs,
        int const *seq_lengths,
        T *out,
        void *temp,
        int B, int H_kv, int G, int Q_stride, int H, int S, bool use_kv_ptrs,
        int virtual_q,
        cudaStream_t stream = 0)
    {
        const int g_packs = (G + NPACK - 1) / NPACK;
        int Splits = 1;
        int S_split = i_max(S, 1);
        int s_tile = S_tile;
        int sm_for_split = get_sm_count_cached();
        {
            s_tile = select_mma_split_tile(S, B, H_kv, G);
            sm_for_split *= 16;
        }
        const uint2 split_cfg = get_split_config(S, B, H_kv, G, sm_for_split, s_tile);
        Splits = split_cfg.x;
        S_split = split_cfg.y;

        size_t red_items = (size_t)Splits * B * H_kv * g_packs * NPACK;
        size_t red_av_elems = red_items * Kdim;

        float *red_av = (float *)temp;
        float *red_m = red_av + red_av_elems;
        float *red_l = red_m + red_items;

        dim3 grid(g_packs, B, Splits * H_kv);
        dim3 block;

        {
#if FLASHDECODE_ENABLE_MMA
            block = dim3(32, WARPS_PB_MMA, 1);
            size_t smem_bytes = sizeof(FlashDecodeSharedStorage<T, STAGES_MMA, WARPS_PB_MMA, NPACK, Kdim>);
            flash_decode_h128_splitS<T, STAGES_MMA, WARPS_PB_MMA, NPACK, Kdim, S_iter_tile_mma><<<grid, block, smem_bytes, stream>>>(
                q, k_base, v_base, k_ptrs, v_ptrs, seq_lengths,
                red_av, red_m, red_l,
                B, H_kv, G, Q_stride, H, S, use_kv_ptrs ? 1 : 0, S_split, virtual_q);
#else
            static_assert(Kind != FlashDecodeKernelKind::mma_sm80, "mma backend is disabled in this translation unit");
#endif
        }

        grid = dim3(g_packs, B, H_kv);
        dim3 block2(32, NPACK, 1);
        {
            flash_decode_reduce_h128_splitS<T, STAGES_MMA, WARPS_PB_MMA, NPACK, Kdim><<<grid, block2, 0, stream>>>(
                red_av, red_m, red_l, out, seq_lengths,
                B, H_kv, G, S, S_split, virtual_q);
        }
    }

    template <FlashDecodeKernelKind Kind>
    static void run(
        T const *q,
        T const *const *k_ptrs,
        T const *const *v_ptrs,
        int const *seq_lengths,
        T *out,
        void *temp,
        int B, int H_kv, int G, int H, int S,
        cudaStream_t stream = 0)
    {
        run_backend<FlashDecodeKernelKind::mma_sm80>(
            q, nullptr, nullptr, k_ptrs, v_ptrs, seq_lengths, out, temp, B, H_kv, G, G + 2, H, S, true, 0, stream);
    }

    template <FlashDecodeKernelKind Kind>
    static void run_qstride(
        T const *q,
        T const *const *k_ptrs,
        T const *const *v_ptrs,
        int const *seq_lengths,
        T *out,
        void *temp,
        int B, int H_kv, int G, int Q_stride, int H, int S,
        cudaStream_t stream = 0)
    {
        run_backend<FlashDecodeKernelKind::mma_sm80>(
            q, nullptr, nullptr, k_ptrs, v_ptrs, seq_lengths, out, temp, B, H_kv, G, Q_stride, H, S, true, 0, stream);
    }

    template <FlashDecodeKernelKind Kind>
    static void run_qstride_contig(
        T const *q,
        T const *k_base,
        T const *v_base,
        int const *seq_lengths,
        T *out,
        void *temp,
        int B, int H_kv, int G, int Q_stride, int H, int S,
        cudaStream_t stream = 0)
    {
        run_backend<FlashDecodeKernelKind::mma_sm80>(
            q, k_base, v_base, nullptr, nullptr, seq_lengths, out, temp, B, H_kv, G, Q_stride, H, S, false, 0, stream);
    }

    template <FlashDecodeKernelKind Kind>
    static void run_qstride_contig_noseqlens(
        T const *q,
        T const *k_base,
        T const *v_base,
        T *out,
        void *temp,
        int B, int H_kv, int G, int Q_stride, int H, int S,
        cudaStream_t stream = 0)
    {
        run_backend<FlashDecodeKernelKind::mma_sm80>(
            q, k_base, v_base, nullptr, nullptr, nullptr, out, temp, B, H_kv, G, Q_stride, H, S, false, 0, stream);
    }

    template <FlashDecodeKernelKind Kind>
    static void run_qstride_contig_causal(
        T const *q,
        T const *k_base,
        T const *v_base,
        T *out,
        void *temp,
        int B, int H_kv, int G, int Q_stride, int H, int S, int virtual_q,
        cudaStream_t stream = 0)
    {
        run_backend<FlashDecodeKernelKind::mma_sm80>(
            q, k_base, v_base, nullptr, nullptr, nullptr, out, temp, B, H_kv, G, Q_stride, H, S, false, virtual_q, stream);
    }

};

} // namespace FLASHDECODE_NAMESPACE
