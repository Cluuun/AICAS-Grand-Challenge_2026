#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t cvt_shared_ptr(void *ptr)
{
    return __cvta_generic_to_shared(ptr);
}

__device__ __forceinline__ void mbarrier_init(uint32_t bar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}

__device__ __forceinline__ void mbarrier_init(uint64_t *barrier_shared, int count)
{
    uint32_t bar = cvt_shared_ptr(barrier_shared);
    mbarrier_init(bar, count);
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

template <int N = 0>
__device__ __forceinline__ void cp_async_wait()
{
    static_assert(N >= 0 && N <= 4, "cp.async.wait_group N must be in [0,4]");
    if constexpr (N == 0)
        asm volatile("cp.async.wait_all;\n");
    else
        asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

template <uint32_t N, bool USE_CA = false>
__device__ __forceinline__ void cp_async(void *dst_shared, void const *src_global)
{
    static_assert(N == 4 || N == 8 || N == 16, "cp.async size must be 4, 8, or 16");
    uint32_t dst = cvt_shared_ptr(dst_shared);
    uint64_t src = (uint64_t)src_global;
    if constexpr (N != 16)
    {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" : : "r"(dst), "l"(src), "n"(N));
    }
    else
    {
        if constexpr (USE_CA)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(dst), "l"(src));
        else
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(dst), "l"(src));
    }
}

__device__ __forceinline__ void clear_16B(void *dst)
{
    *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
}

template <typename T>
__device__ __forceinline__ uint16_t to_u16_bits(T v);

template <>
__device__ __forceinline__ uint16_t to_u16_bits<fp16>(fp16 v)
{
    return reinterpret_cast<const __half_raw &>(v).x;
}

template <>
__device__ __forceinline__ uint16_t to_u16_bits<bf16>(bf16 v)
{
    return reinterpret_cast<const __nv_bfloat16_raw &>(v).x;
}

template <typename T>
__device__ __forceinline__ uint32_t pack_2x16(T a, T b)
{
    uint32_t lo = static_cast<uint32_t>(to_u16_bits(a));
    uint32_t hi = static_cast<uint32_t>(to_u16_bits(b));
    return lo | (hi << 16);
}

template <typename T>
__device__ __forceinline__ uint4 pack_u128_8x16(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
{
    return make_uint4(
        pack_2x16(v0, v1),
        pack_2x16(v2, v3),
        pack_2x16(v4, v5),
        pack_2x16(v6, v7));
}

__device__ __forceinline__ uint2 ldmatrix_x2_offset(uint32_t lane)
{
    return make_uint2(lane & 7, (lane & 8) << 3);
}

template <bool TRANS = false>
__device__ __forceinline__ uint2 ldmatrix_x4_offset(uint32_t lane)
{
    if constexpr (TRANS)
        return make_uint2((lane & 7) + ((lane >> 4) << 3), lane & 8);
    else
        return make_uint2(lane & 15, (lane >> 4) << 3);
}

__device__ __forceinline__ void ldmatrix_sync_x2(void *src_shared, uint32_t &a0, uint32_t &a1)
{
    uint32_t src = cvt_shared_ptr(src_shared);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(a0), "=r"(a1)
                 : "r"(src));
}

template <bool TRANS = false>
__device__ __forceinline__ void ldmatrix_sync_x4(void *src_shared, uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3)
{
    uint32_t src = cvt_shared_ptr(src_shared);
    if constexpr (TRANS)
    {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "r"(src));
    }
    else
    {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "r"(src));
    }
}

template <DataType dtype>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16bf16_f32_accu(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    uint32_t &b0, uint32_t &b1)
{
    if constexpr (dtype == DataType::f16)
    {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, "
                     "{%4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%0, %1, %2, %3};\n"
                     : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                       "r"(b0), "r"(b1));
    }
    else if constexpr (dtype == DataType::bf16)
    {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, "
                     "{%4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%0, %1, %2, %3};\n"
                     : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                       "r"(b0), "r"(b1));
    }
}

__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16_f16_accu(
    uint32_t &d0, uint32_t &d1,
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    uint32_t &b0, uint32_t &b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%0, %1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                   "r"(b0), "r"(b1));
}


__device__ __forceinline__ void mbarrier_wait(uint64_t *smem_barrier, int phase_bit)
{
    uint32_t smem_addr = cvt_shared_ptr(smem_barrier);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "L_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra.uni L_WAIT;\n"
        "}\n" ::"r"(smem_addr),
        "r"(phase_bit) : "memory");
}

__device__ __forceinline__ void umma_commit_arrive(uint64_t *smem_barrier)
{
    uint32_t smem_addr = cvt_shared_ptr(smem_barrier);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(smem_addr) : "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t *smem_dst_ptr, int num_columns)
{
    uint32_t smem_addr = cvt_shared_ptr(smem_dst_ptr);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_addr), "r"(num_columns) : "memory");
}

__device__ __forceinline__ void tmem_free(uint32_t tmem_base, int num_columns)
{
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem_base), "r"(num_columns) : "memory");
}

__device__ __forceinline__ void tmem_relinquish_alloc_permit()
{
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync()
{
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_mma_bf16f16(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t scaleC,
    uint64_t idescE)
{
    uint32_t mask0 = 0, mask1 = 0, mask2 = 0, mask3 = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
        "}\n"
        :
        : "r"(tmem_c),
          "l"(desc_a),
          "l"(desc_b),
          "r"(static_cast<uint32_t>(idescE >> 32)),
          "r"(scaleC),
          "r"(mask0),
          "r"(mask1),
          "r"(mask2),
          "r"(mask3)
        : "memory");
}

__device__ __forceinline__ void tcgen05_mma_i8_s32(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t enable_input_d,
    uint32_t idesc)
{
    uint32_t mask0 = 0, mask1 = 0, mask2 = 0, mask3 = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %8, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, {%4, %5, %6, %7}, p;\n\t"
        "}\n"
        :
        : "r"(tmem_c),
          "l"(desc_a),
          "l"(desc_b),
          "r"(idesc),
          "r"(mask0),
          "r"(mask1),
          "r"(mask2),
          "r"(mask3),
          "r"(enable_input_d)
        : "memory");
}

__device__ __forceinline__ void tcgen05_tmem_ld_32dp32b8x(uint32_t src_addr, uint32_t (&dst)[8])
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "[%8];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
          "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(src_addr)
        : "memory");
}
