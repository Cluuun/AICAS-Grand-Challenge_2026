#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHidden = 2048;
constexpr int kIntermediate = 6144;
constexpr int kGroupSize = 128;
constexpr int kGroups = kIntermediate / kGroupSize;
constexpr int kPackedK4 = kIntermediate / 8;
constexpr int kW4WeightBytes = kHidden * kPackedK4 * static_cast<int>(sizeof(int));
constexpr int kGateGroups = kHidden / kGroupSize;
constexpr int kGatePackedK4 = kHidden / 8;
constexpr int kQkvRows = 4096;
constexpr int kOProjRows = 2048;
constexpr int kLmHeadK = 2048;
constexpr int kLmHeadGroups = kLmHeadK / kGroupSize;
constexpr int kLmHeadPackedK4 = kLmHeadK / 8;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ unsigned ordered_float_bits(float v) {
    const unsigned bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ unsigned lop3_and_u32(unsigned a, unsigned b) {
    unsigned out;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xc0;"
                 : "=r"(out)
                 : "r"(a), "r"(b), "r"(0u));
    return out;
}

__device__ __forceinline__ unsigned lop3_or_u32(unsigned a, unsigned b) {
    unsigned out;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xfc;"
                 : "=r"(out)
                 : "r"(a), "r"(b), "r"(0u));
    return out;
}

__device__ __forceinline__ unsigned prmt_identity_u32(unsigned word) {
    unsigned out;
    asm volatile("prmt.b32 %0, %1, %2, %3;"
                 : "=r"(out)
                 : "r"(word), "r"(word), "r"(0x3210u));
    return out;
}

__device__ __forceinline__ int signed_nibble_lop3(unsigned word, int shift) {
    const unsigned nib = lop3_and_u32(word >> shift, 0xFu);
    return static_cast<int>(nib) - 8;
}

__device__ __forceinline__ __half2 half2_from_bits(unsigned bits) {
    return *reinterpret_cast<__half2*>(&bits);
}

__device__ __forceinline__ __half2 dequant_pair_magic(unsigned pair_bits, __half2 bias) {
    constexpr unsigned kFp16Magic = 0x64006400u;
    const unsigned fp16_bits = lop3_or_u32(pair_bits, kFp16Magic);
    return __hsub2(half2_from_bits(fp16_bits), bias);
}

__device__ __forceinline__ __half2 dequant_nibbles_01(unsigned word, __half2 bias) {
    return dequant_pair_magic((word & 0x0000000fu) | ((word & 0x000000f0u) << 12), bias);
}

__device__ __forceinline__ __half2 dequant_nibbles_23(unsigned word, __half2 bias) {
    return dequant_pair_magic(((word & 0x00000f00u) >> 8) | ((word & 0x0000f000u) << 4), bias);
}

__device__ __forceinline__ __half2 dequant_nibbles_45(unsigned word, __half2 bias) {
    return dequant_pair_magic(((word & 0x000f0000u) >> 16) | ((word & 0x00f00000u) >> 4), bias);
}

__device__ __forceinline__ __half2 dequant_nibbles_67(unsigned word, __half2 bias) {
    return dequant_pair_magic(((word & 0x0f000000u) >> 24) | ((word & 0xf0000000u) >> 12), bias);
}

// lop3 with immLut 0xEA computes (a AND b) OR c in a single instruction.
__device__ __forceinline__ unsigned lop3_and_or_u32(unsigned a, unsigned b, unsigned c) {
    unsigned out;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;"
                 : "=r"(out)
                 : "r"(a), "r"(b), "r"(c));
    return out;
}

// AWQ-interleaved dequant: the two target nibbles already sit at bit
// positions [0:4] and [16:20] of `shifted_word`, so we extract+inject the
// fp16 magic with a single lop3 and one half2 subtract.
__device__ __forceinline__ __half2 dequant_awq_pair(unsigned shifted_word, __half2 bias) {
    constexpr unsigned kMask = 0x000f000fu;
    constexpr unsigned kFp16Magic = 0x64006400u;
    const unsigned bits = lop3_and_or_u32(shifted_word, kMask, kFp16Magic);
    return __hsub2(half2_from_bits(bits), bias);
}

__device__ __forceinline__ unsigned smem_u32_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* global_ptr) {
    const unsigned smem = smem_u32_addr(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :
                 : "r"(smem), "l"(global_ptr));
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* global_ptr) {
    const unsigned smem = smem_u32_addr(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(smem), "l"(global_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;" ::);
}

bool env_flag_enabled(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    return !(raw[0] == '0' || raw[0] == 'f' || raw[0] == 'F' || raw[0] == 'n' || raw[0] == 'N');
}

int env_int_value(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    const int value = std::atoi(raw);
    return value > 0 ? value : default_value;
}

__global__ void instruction_probe_kernel(unsigned* __restrict__ out) {
    const unsigned a = 0x12345678u;
    const unsigned b = 0x87654321u;
    const unsigned c = 0x0f0f0f0fu;
    unsigned lop3_out;
    unsigned prmt_out;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
                 : "=r"(lop3_out)
                 : "r"(a), "r"(b), "r"(c));
    asm volatile("prmt.b32 %0, %1, %2, %3;"
                 : "=r"(prmt_out)
                 : "r"(a), "r"(b), "r"(0x5410u));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = lop3_out;
        out[1] = prmt_out;
    }
}

__global__ void w4_weight_stream_cp_async_probe_kernel(
    const int4* __restrict__ packed_weight,
    unsigned* __restrict__ checksum,
    int total_vecs) {
    constexpr int kThreads = 256;
    constexpr int kTileVecs = kThreads;
    __shared__ int4 smem[kTileVecs];

    const int tid = threadIdx.x;
    const int tiles = (total_vecs + kTileVecs - 1) / kTileVecs;
    unsigned local = 0u;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int vec_idx = tile * kTileVecs + tid;
        if (vec_idx < total_vecs) {
            cp_async_ca_16(&smem[tid], &packed_weight[vec_idx]);
        } else {
            smem[tid] = make_int4(0, 0, 0, 0);
        }
        cp_async_commit();
        cp_async_wait_all();
        const int4 v = smem[tid];
        local ^= static_cast<unsigned>(v.x) + 0x9e3779b9u + (local << 6) + (local >> 2);
        local ^= static_cast<unsigned>(v.y) + 0x85ebca6bu + (local << 6) + (local >> 2);
        local ^= static_cast<unsigned>(v.z) + 0xc2b2ae35u + (local << 6) + (local >> 2);
        local ^= static_cast<unsigned>(v.w) + 0x27d4eb2fu + (local << 6) + (local >> 2);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local ^= __shfl_xor_sync(0xffffffff, local, offset);
    }
    if ((tid & 31) == 0) {
        atomicXor(&checksum[blockIdx.x], local);
    }
}

template <int Stages>
__global__ void w4_weight_stream_any_cp_async_probe_kernel(
    const int4* __restrict__ packed_weight,
    unsigned* __restrict__ checksum,
    int total_vecs) {
    constexpr int kThreads = 256;
    __shared__ int4 smem[Stages][kThreads];

    const int tid = threadIdx.x;
    const int total_tiles = (total_vecs + kThreads - 1) / kThreads;
    unsigned local = 0u;
    for (int tile0 = blockIdx.x * Stages; tile0 < total_tiles; tile0 += gridDim.x * Stages) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            const int vec_idx = (tile0 + stage) * kThreads + tid;
            if (vec_idx < total_vecs) {
                cp_async_ca_16(&smem[stage][tid], &packed_weight[vec_idx]);
            } else {
                smem[stage][tid] = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            const int4 v = smem[stage][tid];
            local ^= static_cast<unsigned>(v.x) + 0x9e3779b9u + (local << 6) + (local >> 2);
            local ^= static_cast<unsigned>(v.y) + 0x85ebca6bu + (local << 6) + (local >> 2);
            local ^= static_cast<unsigned>(v.z) + 0xc2b2ae35u + (local << 6) + (local >> 2);
            local ^= static_cast<unsigned>(v.w) + 0x27d4eb2fu + (local << 6) + (local >> 2);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local ^= __shfl_xor_sync(0xffffffff, local, offset);
    }
    if ((tid & 31) == 0) {
        atomicXor(&checksum[blockIdx.x], local);
    }
}

template <int WarpsPerBlock>
__global__ void gate_up_w4a16_group128_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += nthreads) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (nthreads >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += nthreads) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= kIntermediate) {
        return;
    }
    const int gate_row = row * 2;
    const int up_row = gate_row + 1;
    const int4* __restrict__ gate_w =
        reinterpret_cast<const int4*>(packed_weight + gate_row * kGatePackedK4);
    const int4* __restrict__ up_w =
        reinterpret_cast<const int4*>(packed_weight + up_row * kGatePackedK4);
    const __nv_bfloat16* __restrict__ gate_s = scale + gate_row * kGateGroups;
    const __nv_bfloat16* __restrict__ up_s = scale + up_row * kGateGroups;

    __half2 gate_acc0 = __float2half2_rn(0.0f);
    __half2 gate_acc1 = __float2half2_rn(0.0f);
    __half2 gate_acc2 = __float2half2_rn(0.0f);
    __half2 gate_acc3 = __float2half2_rn(0.0f);
    __half2 up_acc0 = __float2half2_rn(0.0f);
    __half2 up_acc1 = __float2half2_rn(0.0f);
    __half2 up_acc2 = __float2half2_rn(0.0f);
    __half2 up_acc3 = __float2half2_rn(0.0f);
    const __half2 bias = __float2half2_rn(1032.0f);
    constexpr int kInt4VecsPerRow = kGatePackedK4 / 4;

    for (int g = lane; g < kInt4VecsPerRow; g += 32) {
        const int4 gate4 = gate_w[g];
        const int4 up4 = up_w[g];
        const unsigned gw[4] = {
            static_cast<unsigned>(gate4.x),
            static_cast<unsigned>(gate4.y),
            static_cast<unsigned>(gate4.z),
            static_cast<unsigned>(gate4.w),
        };
        const unsigned uw[4] = {
            static_cast<unsigned>(up4.x),
            static_cast<unsigned>(up4.y),
            static_cast<unsigned>(up4.z),
            static_cast<unsigned>(up4.w),
        };
        const int ibase = g * 32;
        const __half2 gate_scale2 = __float2half2_rn(bf16_to_float(gate_s[ibase / kGroupSize]));
        const __half2 up_scale2 = __float2half2_rn(bf16_to_float(up_s[ibase / kGroupSize]));

        const int kbase0 = ibase;
        const __half2 x001 = s_input2[(kbase0 + 0) >> 1];
        const __half2 x023 = s_input2[(kbase0 + 2) >> 1];
        const __half2 x045 = s_input2[(kbase0 + 4) >> 1];
        const __half2 x067 = s_input2[(kbase0 + 6) >> 1];
        gate_acc0 = __hfma2(__hmul2(dequant_nibbles_01(gw[0], bias), gate_scale2), x001, gate_acc0);
        gate_acc0 = __hfma2(__hmul2(dequant_nibbles_23(gw[0], bias), gate_scale2), x023, gate_acc0);
        gate_acc0 = __hfma2(__hmul2(dequant_nibbles_45(gw[0], bias), gate_scale2), x045, gate_acc0);
        gate_acc0 = __hfma2(__hmul2(dequant_nibbles_67(gw[0], bias), gate_scale2), x067, gate_acc0);
        up_acc0 = __hfma2(__hmul2(dequant_nibbles_01(uw[0], bias), up_scale2), x001, up_acc0);
        up_acc0 = __hfma2(__hmul2(dequant_nibbles_23(uw[0], bias), up_scale2), x023, up_acc0);
        up_acc0 = __hfma2(__hmul2(dequant_nibbles_45(uw[0], bias), up_scale2), x045, up_acc0);
        up_acc0 = __hfma2(__hmul2(dequant_nibbles_67(uw[0], bias), up_scale2), x067, up_acc0);

        const int kbase1 = ibase + 8;
        const __half2 x101 = s_input2[(kbase1 + 0) >> 1];
        const __half2 x123 = s_input2[(kbase1 + 2) >> 1];
        const __half2 x145 = s_input2[(kbase1 + 4) >> 1];
        const __half2 x167 = s_input2[(kbase1 + 6) >> 1];
        gate_acc1 = __hfma2(__hmul2(dequant_nibbles_01(gw[1], bias), gate_scale2), x101, gate_acc1);
        gate_acc1 = __hfma2(__hmul2(dequant_nibbles_23(gw[1], bias), gate_scale2), x123, gate_acc1);
        gate_acc1 = __hfma2(__hmul2(dequant_nibbles_45(gw[1], bias), gate_scale2), x145, gate_acc1);
        gate_acc1 = __hfma2(__hmul2(dequant_nibbles_67(gw[1], bias), gate_scale2), x167, gate_acc1);
        up_acc1 = __hfma2(__hmul2(dequant_nibbles_01(uw[1], bias), up_scale2), x101, up_acc1);
        up_acc1 = __hfma2(__hmul2(dequant_nibbles_23(uw[1], bias), up_scale2), x123, up_acc1);
        up_acc1 = __hfma2(__hmul2(dequant_nibbles_45(uw[1], bias), up_scale2), x145, up_acc1);
        up_acc1 = __hfma2(__hmul2(dequant_nibbles_67(uw[1], bias), up_scale2), x167, up_acc1);

        const int kbase2 = ibase + 16;
        const __half2 x201 = s_input2[(kbase2 + 0) >> 1];
        const __half2 x223 = s_input2[(kbase2 + 2) >> 1];
        const __half2 x245 = s_input2[(kbase2 + 4) >> 1];
        const __half2 x267 = s_input2[(kbase2 + 6) >> 1];
        gate_acc2 = __hfma2(__hmul2(dequant_nibbles_01(gw[2], bias), gate_scale2), x201, gate_acc2);
        gate_acc2 = __hfma2(__hmul2(dequant_nibbles_23(gw[2], bias), gate_scale2), x223, gate_acc2);
        gate_acc2 = __hfma2(__hmul2(dequant_nibbles_45(gw[2], bias), gate_scale2), x245, gate_acc2);
        gate_acc2 = __hfma2(__hmul2(dequant_nibbles_67(gw[2], bias), gate_scale2), x267, gate_acc2);
        up_acc2 = __hfma2(__hmul2(dequant_nibbles_01(uw[2], bias), up_scale2), x201, up_acc2);
        up_acc2 = __hfma2(__hmul2(dequant_nibbles_23(uw[2], bias), up_scale2), x223, up_acc2);
        up_acc2 = __hfma2(__hmul2(dequant_nibbles_45(uw[2], bias), up_scale2), x245, up_acc2);
        up_acc2 = __hfma2(__hmul2(dequant_nibbles_67(uw[2], bias), up_scale2), x267, up_acc2);

        const int kbase3 = ibase + 24;
        const __half2 x301 = s_input2[(kbase3 + 0) >> 1];
        const __half2 x323 = s_input2[(kbase3 + 2) >> 1];
        const __half2 x345 = s_input2[(kbase3 + 4) >> 1];
        const __half2 x367 = s_input2[(kbase3 + 6) >> 1];
        gate_acc3 = __hfma2(__hmul2(dequant_nibbles_01(gw[3], bias), gate_scale2), x301, gate_acc3);
        gate_acc3 = __hfma2(__hmul2(dequant_nibbles_23(gw[3], bias), gate_scale2), x323, gate_acc3);
        gate_acc3 = __hfma2(__hmul2(dequant_nibbles_45(gw[3], bias), gate_scale2), x345, gate_acc3);
        gate_acc3 = __hfma2(__hmul2(dequant_nibbles_67(gw[3], bias), gate_scale2), x367, gate_acc3);
        up_acc3 = __hfma2(__hmul2(dequant_nibbles_01(uw[3], bias), up_scale2), x301, up_acc3);
        up_acc3 = __hfma2(__hmul2(dequant_nibbles_23(uw[3], bias), up_scale2), x323, up_acc3);
        up_acc3 = __hfma2(__hmul2(dequant_nibbles_45(uw[3], bias), up_scale2), x345, up_acc3);
        up_acc3 = __hfma2(__hmul2(dequant_nibbles_67(uw[3], bias), up_scale2), x367, up_acc3);
    }

    __half2 gate_sum2 = __hadd2(__hadd2(gate_acc0, gate_acc1), __hadd2(gate_acc2, gate_acc3));
    __half2 up_sum2 = __hadd2(__hadd2(up_acc0, up_acc1), __hadd2(up_acc2, up_acc3));
    float gate_acc = __low2float(gate_sum2) + __high2float(gate_sum2);
    float up_acc = __low2float(up_sum2) + __high2float(up_sum2);
    gate_acc = warp_sum(gate_acc);
    up_acc = warp_sum(up_acc);
    if (lane == 0) {
        const float silu = gate_acc / (1.0f + expf(-gate_acc));
        output[row] = __float2bfloat16(silu * up_acc);
    }
}

__global__ void gate_up_w4a16_group128_split2_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_gate_partial[32];
    __shared__ float s_up_partial[32];
    const int tid = threadIdx.x;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row = blockIdx.x * rows_per_block + local_row;
    if (row >= kIntermediate) {
        return;
    }
    const int gate_row = row * 2;
    const int up_row = gate_row + 1;
    const int4* __restrict__ gate_w =
        reinterpret_cast<const int4*>(packed_weight + gate_row * kGatePackedK4);
    const int4* __restrict__ up_w =
        reinterpret_cast<const int4*>(packed_weight + up_row * kGatePackedK4);
    const __nv_bfloat16* __restrict__ gate_s = scale + gate_row * kGateGroups;
    const __nv_bfloat16* __restrict__ up_s = scale + up_row * kGateGroups;

    __half2 gate_acc0 = __float2half2_rn(0.0f);
    __half2 gate_acc1 = __float2half2_rn(0.0f);
    __half2 gate_acc2 = __float2half2_rn(0.0f);
    __half2 gate_acc3 = __float2half2_rn(0.0f);
    __half2 up_acc0 = __float2half2_rn(0.0f);
    __half2 up_acc1 = __float2half2_rn(0.0f);
    __half2 up_acc2 = __float2half2_rn(0.0f);
    __half2 up_acc3 = __float2half2_rn(0.0f);
    const __half2 bias = __float2half2_rn(1032.0f);
    const int g = lane + part * 32;
    const int4 gate4 = gate_w[g];
    const int4 up4 = up_w[g];
    const unsigned gw[4] = {
        static_cast<unsigned>(gate4.x),
        static_cast<unsigned>(gate4.y),
        static_cast<unsigned>(gate4.z),
        static_cast<unsigned>(gate4.w),
    };
    const unsigned uw[4] = {
        static_cast<unsigned>(up4.x),
        static_cast<unsigned>(up4.y),
        static_cast<unsigned>(up4.z),
        static_cast<unsigned>(up4.w),
    };
    const int ibase = g * 32;
    const __half2 gate_scale2 = __float2half2_rn(bf16_to_float(gate_s[ibase / kGroupSize]));
    const __half2 up_scale2 = __float2half2_rn(bf16_to_float(up_s[ibase / kGroupSize]));

    const int kbase0 = ibase;
    const __half2 x001 = s_input2[(kbase0 + 0) >> 1];
    const __half2 x023 = s_input2[(kbase0 + 2) >> 1];
    const __half2 x045 = s_input2[(kbase0 + 4) >> 1];
    const __half2 x067 = s_input2[(kbase0 + 6) >> 1];
    gate_acc0 = __hfma2(__hmul2(dequant_nibbles_01(gw[0], bias), gate_scale2), x001, gate_acc0);
    gate_acc0 = __hfma2(__hmul2(dequant_nibbles_23(gw[0], bias), gate_scale2), x023, gate_acc0);
    gate_acc0 = __hfma2(__hmul2(dequant_nibbles_45(gw[0], bias), gate_scale2), x045, gate_acc0);
    gate_acc0 = __hfma2(__hmul2(dequant_nibbles_67(gw[0], bias), gate_scale2), x067, gate_acc0);
    up_acc0 = __hfma2(__hmul2(dequant_nibbles_01(uw[0], bias), up_scale2), x001, up_acc0);
    up_acc0 = __hfma2(__hmul2(dequant_nibbles_23(uw[0], bias), up_scale2), x023, up_acc0);
    up_acc0 = __hfma2(__hmul2(dequant_nibbles_45(uw[0], bias), up_scale2), x045, up_acc0);
    up_acc0 = __hfma2(__hmul2(dequant_nibbles_67(uw[0], bias), up_scale2), x067, up_acc0);

    const int kbase1 = ibase + 8;
    const __half2 x101 = s_input2[(kbase1 + 0) >> 1];
    const __half2 x123 = s_input2[(kbase1 + 2) >> 1];
    const __half2 x145 = s_input2[(kbase1 + 4) >> 1];
    const __half2 x167 = s_input2[(kbase1 + 6) >> 1];
    gate_acc1 = __hfma2(__hmul2(dequant_nibbles_01(gw[1], bias), gate_scale2), x101, gate_acc1);
    gate_acc1 = __hfma2(__hmul2(dequant_nibbles_23(gw[1], bias), gate_scale2), x123, gate_acc1);
    gate_acc1 = __hfma2(__hmul2(dequant_nibbles_45(gw[1], bias), gate_scale2), x145, gate_acc1);
    gate_acc1 = __hfma2(__hmul2(dequant_nibbles_67(gw[1], bias), gate_scale2), x167, gate_acc1);
    up_acc1 = __hfma2(__hmul2(dequant_nibbles_01(uw[1], bias), up_scale2), x101, up_acc1);
    up_acc1 = __hfma2(__hmul2(dequant_nibbles_23(uw[1], bias), up_scale2), x123, up_acc1);
    up_acc1 = __hfma2(__hmul2(dequant_nibbles_45(uw[1], bias), up_scale2), x145, up_acc1);
    up_acc1 = __hfma2(__hmul2(dequant_nibbles_67(uw[1], bias), up_scale2), x167, up_acc1);

    const int kbase2 = ibase + 16;
    const __half2 x201 = s_input2[(kbase2 + 0) >> 1];
    const __half2 x223 = s_input2[(kbase2 + 2) >> 1];
    const __half2 x245 = s_input2[(kbase2 + 4) >> 1];
    const __half2 x267 = s_input2[(kbase2 + 6) >> 1];
    gate_acc2 = __hfma2(__hmul2(dequant_nibbles_01(gw[2], bias), gate_scale2), x201, gate_acc2);
    gate_acc2 = __hfma2(__hmul2(dequant_nibbles_23(gw[2], bias), gate_scale2), x223, gate_acc2);
    gate_acc2 = __hfma2(__hmul2(dequant_nibbles_45(gw[2], bias), gate_scale2), x245, gate_acc2);
    gate_acc2 = __hfma2(__hmul2(dequant_nibbles_67(gw[2], bias), gate_scale2), x267, gate_acc2);
    up_acc2 = __hfma2(__hmul2(dequant_nibbles_01(uw[2], bias), up_scale2), x201, up_acc2);
    up_acc2 = __hfma2(__hmul2(dequant_nibbles_23(uw[2], bias), up_scale2), x223, up_acc2);
    up_acc2 = __hfma2(__hmul2(dequant_nibbles_45(uw[2], bias), up_scale2), x245, up_acc2);
    up_acc2 = __hfma2(__hmul2(dequant_nibbles_67(uw[2], bias), up_scale2), x267, up_acc2);

    const int kbase3 = ibase + 24;
    const __half2 x301 = s_input2[(kbase3 + 0) >> 1];
    const __half2 x323 = s_input2[(kbase3 + 2) >> 1];
    const __half2 x345 = s_input2[(kbase3 + 4) >> 1];
    const __half2 x367 = s_input2[(kbase3 + 6) >> 1];
    gate_acc3 = __hfma2(__hmul2(dequant_nibbles_01(gw[3], bias), gate_scale2), x301, gate_acc3);
    gate_acc3 = __hfma2(__hmul2(dequant_nibbles_23(gw[3], bias), gate_scale2), x323, gate_acc3);
    gate_acc3 = __hfma2(__hmul2(dequant_nibbles_45(gw[3], bias), gate_scale2), x345, gate_acc3);
    gate_acc3 = __hfma2(__hmul2(dequant_nibbles_67(gw[3], bias), gate_scale2), x367, gate_acc3);
    up_acc3 = __hfma2(__hmul2(dequant_nibbles_01(uw[3], bias), up_scale2), x301, up_acc3);
    up_acc3 = __hfma2(__hmul2(dequant_nibbles_23(uw[3], bias), up_scale2), x323, up_acc3);
    up_acc3 = __hfma2(__hmul2(dequant_nibbles_45(uw[3], bias), up_scale2), x345, up_acc3);
    up_acc3 = __hfma2(__hmul2(dequant_nibbles_67(uw[3], bias), up_scale2), x367, up_acc3);

    __half2 gate_sum2 = __hadd2(__hadd2(gate_acc0, gate_acc1), __hadd2(gate_acc2, gate_acc3));
    __half2 up_sum2 = __hadd2(__hadd2(up_acc0, up_acc1), __hadd2(up_acc2, up_acc3));
    float gate_acc = __low2float(gate_sum2) + __high2float(gate_sum2);
    float up_acc = __low2float(up_sum2) + __high2float(up_sum2);
    gate_acc = warp_sum(gate_acc);
    up_acc = warp_sum(up_acc);
    if (lane == 0) {
        s_gate_partial[warp] = gate_acc;
        s_up_partial[warp] = up_acc;
    }
    __syncthreads();
    if (part == 0 && lane == 0) {
        const float gate = s_gate_partial[warp] + s_gate_partial[warp + 1];
        const float up = s_up_partial[warp] + s_up_partial[warp + 1];
        const float silu = gate / (1.0f + expf(-gate));
        output[row] = __float2bfloat16(silu * up);
    }
}

// Same structure as gate_up_w4a16_group128_split2_kernel, but expects the
// packed weight to be in AWQ-interleaved nibble order ([0,2,4,6,1,3,5,7]
// within each int32) so dequant uses a single lop3 per half2 instead of the
// per-nibble mask/shift/or sequence. Numerically identical to the sequential
// kernel given a matching offline repack.
__global__ void gate_up_w4a16_group128_split2_awq_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_gate_partial[32];
    __shared__ float s_up_partial[32];
    const int tid = threadIdx.x;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row = blockIdx.x * rows_per_block + local_row;
    if (row >= kIntermediate) {
        return;
    }
    const int gate_row = row * 2;
    const int up_row = gate_row + 1;
    const int4* __restrict__ gate_w =
        reinterpret_cast<const int4*>(packed_weight + gate_row * kGatePackedK4);
    const int4* __restrict__ up_w =
        reinterpret_cast<const int4*>(packed_weight + up_row * kGatePackedK4);
    const __nv_bfloat16* __restrict__ gate_s = scale + gate_row * kGateGroups;
    const __nv_bfloat16* __restrict__ up_s = scale + up_row * kGateGroups;

    const __half2 bias = __float2half2_rn(1032.0f);
    const int g = lane + part * 32;
    const int4 gate4 = gate_w[g];
    const int4 up4 = up_w[g];
    const unsigned gw[4] = {
        static_cast<unsigned>(gate4.x), static_cast<unsigned>(gate4.y),
        static_cast<unsigned>(gate4.z), static_cast<unsigned>(gate4.w),
    };
    const unsigned uw[4] = {
        static_cast<unsigned>(up4.x), static_cast<unsigned>(up4.y),
        static_cast<unsigned>(up4.z), static_cast<unsigned>(up4.w),
    };
    const int ibase = g * 32;
    const __half2 gate_scale2 = __float2half2_rn(bf16_to_float(gate_s[ibase / kGroupSize]));
    const __half2 up_scale2 = __float2half2_rn(bf16_to_float(up_s[ibase / kGroupSize]));

    __half2 gate_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                           __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    __half2 up_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                         __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int kb = ibase + i * 8;
        const __half2 x01 = s_input2[(kb + 0) >> 1];
        const __half2 x23 = s_input2[(kb + 2) >> 1];
        const __half2 x45 = s_input2[(kb + 4) >> 1];
        const __half2 x67 = s_input2[(kb + 6) >> 1];
        const unsigned gwi = gw[i];
        gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi, bias), gate_scale2), x01, gate_acc[i]);
        gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 4, bias), gate_scale2), x23, gate_acc[i]);
        gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 8, bias), gate_scale2), x45, gate_acc[i]);
        gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 12, bias), gate_scale2), x67, gate_acc[i]);
        const unsigned uwi = uw[i];
        up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi, bias), up_scale2), x01, up_acc[i]);
        up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 4, bias), up_scale2), x23, up_acc[i]);
        up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 8, bias), up_scale2), x45, up_acc[i]);
        up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 12, bias), up_scale2), x67, up_acc[i]);
    }

    __half2 gate_sum2 = __hadd2(__hadd2(gate_acc[0], gate_acc[1]), __hadd2(gate_acc[2], gate_acc[3]));
    __half2 up_sum2 = __hadd2(__hadd2(up_acc[0], up_acc[1]), __hadd2(up_acc[2], up_acc[3]));
    float gate_partial = __low2float(gate_sum2) + __high2float(gate_sum2);
    float up_partial = __low2float(up_sum2) + __high2float(up_sum2);
    gate_partial = warp_sum(gate_partial);
    up_partial = warp_sum(up_partial);
    if (lane == 0) {
        s_gate_partial[warp] = gate_partial;
        s_up_partial[warp] = up_partial;
    }
    __syncthreads();
    if (part == 0 && lane == 0) {
        const float gate = s_gate_partial[warp] + s_gate_partial[warp + 1];
        const float up = s_up_partial[warp] + s_up_partial[warp + 1];
        const float silu = gate / (1.0f + expf(-gate));
        output[row] = __float2bfloat16(silu * up);
    }
}

__device__ __forceinline__ void prefetch_gate_up_awq_stage(
    int4* __restrict__ s_gate_weight,
    int4* __restrict__ s_up_weight,
    int stage_offset,
    int tid,
    bool active,
    const int* __restrict__ packed_weight,
    int row,
    int lane,
    int part) {
    if (active) {
        const int g = lane + part * 32;
        const int gate_row = row * 2;
        const int up_row = gate_row + 1;
        const int4* __restrict__ gate_w =
            reinterpret_cast<const int4*>(packed_weight + gate_row * kGatePackedK4);
        const int4* __restrict__ up_w =
            reinterpret_cast<const int4*>(packed_weight + up_row * kGatePackedK4);
        cp_async_cg_16(&s_gate_weight[stage_offset + tid], &gate_w[g]);
        cp_async_cg_16(&s_up_weight[stage_offset + tid], &up_w[g]);
    } else {
        s_gate_weight[stage_offset + tid] = make_int4(0, 0, 0, 0);
        s_up_weight[stage_offset + tid] = make_int4(0, 0, 0, 0);
    }
}

template <int Stages>
__global__ void gate_up_w4a16_group128_split2_awq_cp_async_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_gate_partial[32];
    __shared__ float s_up_partial[32];
    extern __shared__ int4 s_weight_stage[];
    int4* __restrict__ s_gate_weight = s_weight_stage;
    int4* __restrict__ s_up_weight = s_gate_weight + Stages * blockDim.x;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row_stride = gridDim.x * rows_per_block;
    const int base_row = blockIdx.x * rows_per_block + local_row;
    const int total_iters = (kIntermediate + row_stride - 1) / row_stride;

    int stage = 0;
    int stage_offset = 0;
    int row = base_row;
    prefetch_gate_up_awq_stage(
        s_gate_weight, s_up_weight, stage_offset, tid, row < kIntermediate,
        packed_weight, row, lane, part);
    cp_async_commit();

    const __half2 bias = __float2half2_rn(1032.0f);
    for (int iter = 0; iter < total_iters; ++iter) {
        row = base_row + iter * row_stride;
        const bool active = row < kIntermediate;
        cp_async_wait_all();
        __syncthreads();

        const int next_iter = iter + 1;
        const int next_stage = (stage + 1) % Stages;
        if (next_iter < total_iters) {
            const int next_row = base_row + next_iter * row_stride;
            const int next_stage_offset = next_stage * blockDim.x;
            prefetch_gate_up_awq_stage(
                s_gate_weight, s_up_weight, next_stage_offset, tid, next_row < kIntermediate,
                packed_weight, next_row, lane, part);
            cp_async_commit();
        }

        if (active) {
            const int g = lane + part * 32;
            const int ibase = g * 32;
            const int gate_row = row * 2;
            const int up_row = gate_row + 1;
            const __nv_bfloat16* __restrict__ gate_s = scale + gate_row * kGateGroups;
            const __nv_bfloat16* __restrict__ up_s = scale + up_row * kGateGroups;
            const __half2 gate_scale2 = __float2half2_rn(bf16_to_float(gate_s[ibase / kGroupSize]));
            const __half2 up_scale2 = __float2half2_rn(bf16_to_float(up_s[ibase / kGroupSize]));
            const int4 gate4 = s_gate_weight[stage_offset + tid];
            const int4 up4 = s_up_weight[stage_offset + tid];
            const unsigned gw[4] = {
                static_cast<unsigned>(gate4.x), static_cast<unsigned>(gate4.y),
                static_cast<unsigned>(gate4.z), static_cast<unsigned>(gate4.w),
            };
            const unsigned uw[4] = {
                static_cast<unsigned>(up4.x), static_cast<unsigned>(up4.y),
                static_cast<unsigned>(up4.z), static_cast<unsigned>(up4.w),
            };

            __half2 gate_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                                   __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
            __half2 up_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                                 __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kb = ibase + i * 8;
                const __half2 x01 = s_input2[(kb + 0) >> 1];
                const __half2 x23 = s_input2[(kb + 2) >> 1];
                const __half2 x45 = s_input2[(kb + 4) >> 1];
                const __half2 x67 = s_input2[(kb + 6) >> 1];
                const unsigned gwi = gw[i];
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi, bias), gate_scale2), x01, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 4, bias), gate_scale2), x23, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 8, bias), gate_scale2), x45, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 12, bias), gate_scale2), x67, gate_acc[i]);
                const unsigned uwi = uw[i];
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi, bias), up_scale2), x01, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 4, bias), up_scale2), x23, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 8, bias), up_scale2), x45, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 12, bias), up_scale2), x67, up_acc[i]);
            }

            __half2 gate_sum2 = __hadd2(__hadd2(gate_acc[0], gate_acc[1]), __hadd2(gate_acc[2], gate_acc[3]));
            __half2 up_sum2 = __hadd2(__hadd2(up_acc[0], up_acc[1]), __hadd2(up_acc[2], up_acc[3]));
            float gate_partial = __low2float(gate_sum2) + __high2float(gate_sum2);
            float up_partial = __low2float(up_sum2) + __high2float(up_sum2);
            gate_partial = warp_sum(gate_partial);
            up_partial = warp_sum(up_partial);
            if (lane == 0) {
                s_gate_partial[warp] = gate_partial;
                s_up_partial[warp] = up_partial;
            }
        }
        __syncthreads();
        if (active && part == 0 && lane == 0) {
            const float gate = s_gate_partial[warp] + s_gate_partial[warp + 1];
            const float up = s_up_partial[warp] + s_up_partial[warp + 1];
            const float silu = gate / (1.0f + expf(-gate));
            output[row] = __float2bfloat16(silu * up);
        }
        __syncthreads();
        stage = next_stage;
        stage_offset = stage * blockDim.x;
    }
}

template <bool PreNorm>
__global__ void gate_up_w4a16_group128_split2_awq_persistent_tpl_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_gate_partial[32];
    __shared__ float s_up_partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (PreNorm) {
        // Activations are pre-normalized upstream; just stage them as half2.
        for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
            const int k = k2 * 2;
            s_input2[k2] = __halves2half2(
                __float2half(bf16_to_float(residual[k + 0])),
                __float2half(bf16_to_float(residual[k + 1])));
        }
        __syncthreads();
    } else {
    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();
    }

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row_stride = gridDim.x * rows_per_block;
    const int base_row = blockIdx.x * rows_per_block + local_row;
    const int total_iters = (kIntermediate + row_stride - 1) / row_stride;
    const __half2 bias = __float2half2_rn(1032.0f);

    for (int iter = 0; iter < total_iters; ++iter) {
        const int row = base_row + iter * row_stride;
        const bool active = row < kIntermediate;
        if (active) {
            const int gate_row = row * 2;
            const int up_row = gate_row + 1;
            const int4* __restrict__ gate_w =
                reinterpret_cast<const int4*>(packed_weight + gate_row * kGatePackedK4);
            const int4* __restrict__ up_w =
                reinterpret_cast<const int4*>(packed_weight + up_row * kGatePackedK4);
            const __nv_bfloat16* __restrict__ gate_s = scale + gate_row * kGateGroups;
            const __nv_bfloat16* __restrict__ up_s = scale + up_row * kGateGroups;
            const int g = lane + part * 32;
            const int4 gate4 = gate_w[g];
            const int4 up4 = up_w[g];
            const unsigned gw[4] = {
                static_cast<unsigned>(gate4.x), static_cast<unsigned>(gate4.y),
                static_cast<unsigned>(gate4.z), static_cast<unsigned>(gate4.w),
            };
            const unsigned uw[4] = {
                static_cast<unsigned>(up4.x), static_cast<unsigned>(up4.y),
                static_cast<unsigned>(up4.z), static_cast<unsigned>(up4.w),
            };
            const int ibase = g * 32;
            const __half2 gate_scale2 = __float2half2_rn(bf16_to_float(gate_s[ibase / kGroupSize]));
            const __half2 up_scale2 = __float2half2_rn(bf16_to_float(up_s[ibase / kGroupSize]));

            __half2 gate_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                                   __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
            __half2 up_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                                 __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kb = ibase + i * 8;
                const __half2 x01 = s_input2[(kb + 0) >> 1];
                const __half2 x23 = s_input2[(kb + 2) >> 1];
                const __half2 x45 = s_input2[(kb + 4) >> 1];
                const __half2 x67 = s_input2[(kb + 6) >> 1];
                const unsigned gwi = gw[i];
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi, bias), gate_scale2), x01, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 4, bias), gate_scale2), x23, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 8, bias), gate_scale2), x45, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 12, bias), gate_scale2), x67, gate_acc[i]);
                const unsigned uwi = uw[i];
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi, bias), up_scale2), x01, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 4, bias), up_scale2), x23, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 8, bias), up_scale2), x45, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 12, bias), up_scale2), x67, up_acc[i]);
            }

            __half2 gate_sum2 = __hadd2(__hadd2(gate_acc[0], gate_acc[1]), __hadd2(gate_acc[2], gate_acc[3]));
            __half2 up_sum2 = __hadd2(__hadd2(up_acc[0], up_acc[1]), __hadd2(up_acc[2], up_acc[3]));
            float gate_partial = __low2float(gate_sum2) + __high2float(gate_sum2);
            float up_partial = __low2float(up_sum2) + __high2float(up_sum2);
            gate_partial = warp_sum(gate_partial);
            up_partial = warp_sum(up_partial);
            if (lane == 0) {
                s_gate_partial[warp] = gate_partial;
                s_up_partial[warp] = up_partial;
            }
        }
        __syncthreads();
        if (active && part == 0 && lane == 0) {
            const float gate = s_gate_partial[warp] + s_gate_partial[warp + 1];
            const float up = s_up_partial[warp] + s_up_partial[warp + 1];
            const float silu = gate / (1.0f + expf(-gate));
            output[row] = __float2bfloat16(silu * up);
        }
        __syncthreads();
    }
}

// Double-buffered split-2 persistent gate_up: issues the NEXT row's gate+up int4
// loads before dequanting the current row, so the DRAM load latency overlaps the
// dequant+reduce instead of being exposed. Targets the gap between the GEMV's
// ~1170 GB/s and the streaming ceiling (~1455) — the kernel is latency/MLP-bound,
// not DRAM-saturated, so more loads in flight raises achieved bandwidth.
__global__ void gate_up_w4a16_group128_split2_awq_persistent_pf_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_gate_partial[32];
    __shared__ float s_up_partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) s_warp_sums[warp] = sq_sum;
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) block_sq = warp_sum(block_sq);
    if (tid == 0) rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    __syncthreads();
    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row_stride = gridDim.x * rows_per_block;
    const int base_row = blockIdx.x * rows_per_block + local_row;
    const int total_iters = (kIntermediate + row_stride - 1) / row_stride;
    const __half2 bias = __float2half2_rn(1032.0f);
    const int g = lane + part * 32;
    const int ibase = g * 32;
    const int sgi = ibase / kGroupSize;

    auto load_row = [&](int row, int4& g4, int4& u4) {
        const int4* __restrict__ gw = reinterpret_cast<const int4*>(packed_weight + (row * 2) * kGatePackedK4);
        const int4* __restrict__ uw = reinterpret_cast<const int4*>(packed_weight + (row * 2 + 1) * kGatePackedK4);
        g4 = gw[g];
        u4 = uw[g];
    };

    int4 g4_cur, u4_cur;
    if (base_row < kIntermediate) load_row(base_row, g4_cur, u4_cur);
    for (int iter = 0; iter < total_iters; ++iter) {
        const int row = base_row + iter * row_stride;
        const bool active = row < kIntermediate;
        const int next_row = row + row_stride;
        int4 g4_next, u4_next;
        if (next_row < kIntermediate) load_row(next_row, g4_next, u4_next);  // prefetch
        if (active) {
            const __nv_bfloat16* __restrict__ gate_s = scale + (row * 2) * kGateGroups;
            const __nv_bfloat16* __restrict__ up_s = scale + (row * 2 + 1) * kGateGroups;
            const __half2 gate_scale2 = __float2half2_rn(bf16_to_float(gate_s[sgi]));
            const __half2 up_scale2 = __float2half2_rn(bf16_to_float(up_s[sgi]));
            const unsigned gw[4] = {(unsigned)g4_cur.x, (unsigned)g4_cur.y, (unsigned)g4_cur.z, (unsigned)g4_cur.w};
            const unsigned uw[4] = {(unsigned)u4_cur.x, (unsigned)u4_cur.y, (unsigned)u4_cur.z, (unsigned)u4_cur.w};
            __half2 gate_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
            __half2 up_acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kb = ibase + i * 8;
                const __half2 x01 = s_input2[(kb + 0) >> 1];
                const __half2 x23 = s_input2[(kb + 2) >> 1];
                const __half2 x45 = s_input2[(kb + 4) >> 1];
                const __half2 x67 = s_input2[(kb + 6) >> 1];
                const unsigned gwi = gw[i];
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi, bias), gate_scale2), x01, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 4, bias), gate_scale2), x23, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 8, bias), gate_scale2), x45, gate_acc[i]);
                gate_acc[i] = __hfma2(__hmul2(dequant_awq_pair(gwi >> 12, bias), gate_scale2), x67, gate_acc[i]);
                const unsigned uwi = uw[i];
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi, bias), up_scale2), x01, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 4, bias), up_scale2), x23, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 8, bias), up_scale2), x45, up_acc[i]);
                up_acc[i] = __hfma2(__hmul2(dequant_awq_pair(uwi >> 12, bias), up_scale2), x67, up_acc[i]);
            }
            __half2 gate_sum2 = __hadd2(__hadd2(gate_acc[0], gate_acc[1]), __hadd2(gate_acc[2], gate_acc[3]));
            __half2 up_sum2 = __hadd2(__hadd2(up_acc[0], up_acc[1]), __hadd2(up_acc[2], up_acc[3]));
            float gate_partial = warp_sum(__low2float(gate_sum2) + __high2float(gate_sum2));
            float up_partial = warp_sum(__low2float(up_sum2) + __high2float(up_sum2));
            if (lane == 0) { s_gate_partial[warp] = gate_partial; s_up_partial[warp] = up_partial; }
        }
        __syncthreads();
        if (active && part == 0 && lane == 0) {
            const float gate = s_gate_partial[warp] + s_gate_partial[warp + 1];
            const float up = s_up_partial[warp] + s_up_partial[warp + 1];
            const float silu = gate / (1.0f + expf(-gate));
            output[row] = __float2bfloat16(silu * up);
        }
        g4_cur = g4_next;
        u4_cur = u4_next;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Unsplit (1 warp per output row) AWQ GEMV kernels with register prefetch.
//
// Rationale (measured): the split-2 persistent kernels issue only 1-2
// outstanding weight loads per thread and pay two __syncthreads() per row for
// the cross-warp reduction, leaving them ~20-32% below the DRAM streaming
// ceiling for their read size. Mapping one output row to a single warp removes
// all in-loop block synchronization (warp-shuffle reduction only) and lets each
// lane keep 4 weight int4 loads in flight; an explicit next-row prefetch raises
// memory-level parallelism further so the GEMV approaches pure-streaming BW.
// ---------------------------------------------------------------------------

// Accumulate one packed int4 (4x int32 = 32 weights, AWQ-interleaved layout)
// into a 4-wide half2 accumulator. ibase is the K offset of this int4's first
// weight; s_input2 holds the (normalized) activations as half2 pairs.
__device__ __forceinline__ void accumulate_int4_awq(
    const int4 w4,
    const __half2 scale2,
    const int ibase,
    const __half2* __restrict__ s_input2,
    const __half2 bias,
    __half2 (&acc)[4]) {
    const unsigned ww[4] = {
        static_cast<unsigned>(w4.x), static_cast<unsigned>(w4.y),
        static_cast<unsigned>(w4.z), static_cast<unsigned>(w4.w),
    };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int kb = ibase + j * 8;
        const __half2 x01 = s_input2[(kb + 0) >> 1];
        const __half2 x23 = s_input2[(kb + 2) >> 1];
        const __half2 x45 = s_input2[(kb + 4) >> 1];
        const __half2 x67 = s_input2[(kb + 6) >> 1];
        const unsigned w = ww[j];
        acc[j] = __hfma2(__hmul2(dequant_awq_pair(w, bias), scale2), x01, acc[j]);
        acc[j] = __hfma2(__hmul2(dequant_awq_pair(w >> 4, bias), scale2), x23, acc[j]);
        acc[j] = __hfma2(__hmul2(dequant_awq_pair(w >> 8, bias), scale2), x45, acc[j]);
        acc[j] = __hfma2(__hmul2(dequant_awq_pair(w >> 12, bias), scale2), x67, acc[j]);
    }
}

__device__ __forceinline__ float reduce_acc4_warp(const __half2 (&acc)[4]) {
    const __half2 s2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
    float v = __low2float(s2) + __high2float(s2);
    return warp_sum(v);
}

template <int WarpsPerBlock, bool Prefetch>
__global__ void gate_up_w4a16_group128_unsplit_awq_persistent_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // RMSNorm over the residual (block-cooperative), then fold norm_weight into
    // the activations and stash as half2 in shared. One sync pair, hoisted out
    // of the row loop entirely.
    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const __half2 bias = __float2half2_rn(1032.0f);
    const int warps_total = gridDim.x * WarpsPerBlock;
    const int first_row = blockIdx.x * WarpsPerBlock + warp;
    const int g0 = lane;
    const int g1 = lane + 32;
    const int ibase0 = g0 * 32;
    const int ibase1 = g1 * 32;
    // scale group index for these int4s (group_size 128 => g*32/128 == g/4)
    const int sg0 = g0 >> 2;
    const int sg1 = g1 >> 2;

    auto load_row = [&](int row, int4& g4a, int4& g4b, int4& u4a, int4& u4b) {
        const int gate_row = row * 2;
        const int up_row = gate_row + 1;
        const int4* __restrict__ gw = reinterpret_cast<const int4*>(packed_weight + gate_row * kGatePackedK4);
        const int4* __restrict__ uw = reinterpret_cast<const int4*>(packed_weight + up_row * kGatePackedK4);
        g4a = gw[g0];
        g4b = gw[g1];
        u4a = uw[g0];
        u4b = uw[g1];
    };

    int row = first_row;
    int4 g4a, g4b, u4a, u4b;
    if (Prefetch && row < kIntermediate) {
        load_row(row, g4a, g4b, u4a, u4b);
    }
    while (row < kIntermediate) {
        const int next_row = row + warps_total;
        int4 ng4a, ng4b, nu4a, nu4b;
        if (Prefetch) {
            if (next_row < kIntermediate) {
                // Issue the next row's four weight loads up front so they stream
                // in behind the current row's dequant + FMA work.
                load_row(next_row, ng4a, ng4b, nu4a, nu4b);
            }
        } else {
            load_row(row, g4a, g4b, u4a, u4b);
        }
        const int gate_row = row * 2;
        const int up_row = gate_row + 1;
        const __nv_bfloat16* __restrict__ gate_s = scale + gate_row * kGateGroups;
        const __nv_bfloat16* __restrict__ up_s = scale + up_row * kGateGroups;
        const __half2 gsa = __float2half2_rn(bf16_to_float(gate_s[sg0]));
        const __half2 gsb = __float2half2_rn(bf16_to_float(gate_s[sg1]));
        const __half2 usa = __float2half2_rn(bf16_to_float(up_s[sg0]));
        const __half2 usb = __float2half2_rn(bf16_to_float(up_s[sg1]));

        __half2 gacc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
        __half2 uacc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
        accumulate_int4_awq(g4a, gsa, ibase0, s_input2, bias, gacc);
        accumulate_int4_awq(g4b, gsb, ibase1, s_input2, bias, gacc);
        accumulate_int4_awq(u4a, usa, ibase0, s_input2, bias, uacc);
        accumulate_int4_awq(u4b, usb, ibase1, s_input2, bias, uacc);

        const float gate = reduce_acc4_warp(gacc);
        const float up = reduce_acc4_warp(uacc);
        if (lane == 0) {
            const float silu = gate / (1.0f + expf(-gate));
            output[row] = __float2bfloat16(silu * up);
        }
        if (Prefetch) {
            g4a = ng4a; g4b = ng4b; u4a = nu4a; u4b = nu4b;
        }
        row = next_row;
    }
}

template <int WarpsPerBlock, bool Prefetch>
__global__ void qkv_w4a16_group128_unsplit_awq_persistent_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const __half2 bias = __float2half2_rn(1032.0f);
    const int warps_total = gridDim.x * WarpsPerBlock;
    const int g0 = lane;
    const int g1 = lane + 32;
    const int ibase0 = g0 * 32;
    const int ibase1 = g1 * 32;
    const int sg0 = g0 >> 2;
    const int sg1 = g1 >> 2;

    auto load_row = [&](int row, int4& w4a, int4& w4b) {
        const int4* __restrict__ rw = reinterpret_cast<const int4*>(packed_weight + row * kGatePackedK4);
        w4a = rw[g0];
        w4b = rw[g1];
    };

    int row = blockIdx.x * WarpsPerBlock + warp;
    int4 w4a, w4b;
    if (Prefetch && row < kQkvRows) {
        load_row(row, w4a, w4b);
    }
    while (row < kQkvRows) {
        const int next_row = row + warps_total;
        int4 nw4a, nw4b;
        if (Prefetch) {
            if (next_row < kQkvRows) {
                load_row(next_row, nw4a, nw4b);
            }
        } else {
            load_row(row, w4a, w4b);
        }
        const __nv_bfloat16* __restrict__ row_s = scale + row * kGateGroups;
        const __half2 sa = __float2half2_rn(bf16_to_float(row_s[sg0]));
        const __half2 sb = __float2half2_rn(bf16_to_float(row_s[sg1]));
        __half2 acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
        accumulate_int4_awq(w4a, sa, ibase0, s_input2, bias, acc);
        accumulate_int4_awq(w4b, sb, ibase1, s_input2, bias, acc);
        const float v = reduce_acc4_warp(acc);
        if (lane == 0) {
            output[row] = __float2bfloat16(v);
        }
        if (Prefetch) {
            w4a = nw4a; w4b = nw4b;
        }
        row = next_row;
    }
}

// Unsplit o_proj: 1 warp per output row, input loaded to shared (no RMSNorm),
// result accumulated into residual. Same MLP-maximizing structure as the qkv
// unsplit kernel (4 weight loads/lane, warp-shuffle reduction, no in-loop sync).
template <int WarpsPerBlock>
__global__ void o_proj_add_w4a16_group128_unsplit_awq_persistent_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ residual) {
    __shared__ __half2 s_input2[kHidden / 2];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        s_input2[k2] = __halves2half2(
            __float2half(bf16_to_float(input[k + 0])),
            __float2half(bf16_to_float(input[k + 1])));
    }
    __syncthreads();

    const __half2 bias = __float2half2_rn(1032.0f);
    const int warps_total = gridDim.x * WarpsPerBlock;
    const int g0 = lane;
    const int g1 = lane + 32;
    const int ibase0 = g0 * 32;
    const int ibase1 = g1 * 32;
    const int sg0 = g0 >> 2;
    const int sg1 = g1 >> 2;
    for (int row = blockIdx.x * WarpsPerBlock + warp; row < kOProjRows; row += warps_total) {
        const int4* __restrict__ rw = reinterpret_cast<const int4*>(packed_weight + row * kGatePackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + row * kGateGroups;
        const int4 w4a = rw[g0];
        const int4 w4b = rw[g1];
        const __half2 sa = __float2half2_rn(bf16_to_float(row_s[sg0]));
        const __half2 sb = __float2half2_rn(bf16_to_float(row_s[sg1]));
        __half2 acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
        accumulate_int4_awq(w4a, sa, ibase0, s_input2, bias, acc);
        accumulate_int4_awq(w4b, sb, ibase1, s_input2, bias, acc);
        const float v = reduce_acc4_warp(acc);
        if (lane == 0) {
            residual[row] = __float2bfloat16(bf16_to_float(residual[row]) + v);
        }
    }
}

__global__ void qkv_w4a16_group128_split2_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row = blockIdx.x * rows_per_block + local_row;
    if (row >= kQkvRows) {
        return;
    }
    const int4* __restrict__ row_w =
        reinterpret_cast<const int4*>(packed_weight + row * kGatePackedK4);
    const __nv_bfloat16* __restrict__ row_s = scale + row * kGateGroups;

    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 acc3 = __float2half2_rn(0.0f);
    const __half2 bias = __float2half2_rn(1032.0f);
    const int g = lane + part * 32;
    const int4 w4 = row_w[g];
    const unsigned ww[4] = {
        static_cast<unsigned>(w4.x),
        static_cast<unsigned>(w4.y),
        static_cast<unsigned>(w4.z),
        static_cast<unsigned>(w4.w),
    };
    const int ibase = g * 32;
    const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));

    const int kbase0 = ibase;
    acc0 = __hfma2(__hmul2(dequant_nibbles_01(ww[0], bias), scale2), s_input2[(kbase0 + 0) >> 1], acc0);
    acc0 = __hfma2(__hmul2(dequant_nibbles_23(ww[0], bias), scale2), s_input2[(kbase0 + 2) >> 1], acc0);
    acc0 = __hfma2(__hmul2(dequant_nibbles_45(ww[0], bias), scale2), s_input2[(kbase0 + 4) >> 1], acc0);
    acc0 = __hfma2(__hmul2(dequant_nibbles_67(ww[0], bias), scale2), s_input2[(kbase0 + 6) >> 1], acc0);
    const int kbase1 = ibase + 8;
    acc1 = __hfma2(__hmul2(dequant_nibbles_01(ww[1], bias), scale2), s_input2[(kbase1 + 0) >> 1], acc1);
    acc1 = __hfma2(__hmul2(dequant_nibbles_23(ww[1], bias), scale2), s_input2[(kbase1 + 2) >> 1], acc1);
    acc1 = __hfma2(__hmul2(dequant_nibbles_45(ww[1], bias), scale2), s_input2[(kbase1 + 4) >> 1], acc1);
    acc1 = __hfma2(__hmul2(dequant_nibbles_67(ww[1], bias), scale2), s_input2[(kbase1 + 6) >> 1], acc1);
    const int kbase2 = ibase + 16;
    acc2 = __hfma2(__hmul2(dequant_nibbles_01(ww[2], bias), scale2), s_input2[(kbase2 + 0) >> 1], acc2);
    acc2 = __hfma2(__hmul2(dequant_nibbles_23(ww[2], bias), scale2), s_input2[(kbase2 + 2) >> 1], acc2);
    acc2 = __hfma2(__hmul2(dequant_nibbles_45(ww[2], bias), scale2), s_input2[(kbase2 + 4) >> 1], acc2);
    acc2 = __hfma2(__hmul2(dequant_nibbles_67(ww[2], bias), scale2), s_input2[(kbase2 + 6) >> 1], acc2);
    const int kbase3 = ibase + 24;
    acc3 = __hfma2(__hmul2(dequant_nibbles_01(ww[3], bias), scale2), s_input2[(kbase3 + 0) >> 1], acc3);
    acc3 = __hfma2(__hmul2(dequant_nibbles_23(ww[3], bias), scale2), s_input2[(kbase3 + 2) >> 1], acc3);
    acc3 = __hfma2(__hmul2(dequant_nibbles_45(ww[3], bias), scale2), s_input2[(kbase3 + 4) >> 1], acc3);
    acc3 = __hfma2(__hmul2(dequant_nibbles_67(ww[3], bias), scale2), s_input2[(kbase3 + 6) >> 1], acc3);

    const __half2 sum2 = __hadd2(__hadd2(acc0, acc1), __hadd2(acc2, acc3));
    float acc = __low2float(sum2) + __high2float(sum2);
    acc = warp_sum(acc);
    if (lane == 0) {
        s_partial[warp] = acc;
    }
    __syncthreads();
    if (part == 0 && lane == 0) {
        output[row] = __float2bfloat16(s_partial[warp] + s_partial[warp + 1]);
    }
}

// AWQ-interleaved nibble variant of qkv_w4a16_group128_split2_kernel.
__global__ void qkv_w4a16_group128_split2_awq_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row = blockIdx.x * rows_per_block + local_row;
    if (row >= kQkvRows) {
        return;
    }
    const int4* __restrict__ row_w =
        reinterpret_cast<const int4*>(packed_weight + row * kGatePackedK4);
    const __nv_bfloat16* __restrict__ row_s = scale + row * kGateGroups;

    __half2 acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                      __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    const __half2 bias = __float2half2_rn(1032.0f);
    const int g = lane + part * 32;
    const int4 w4 = row_w[g];
    const unsigned ww[4] = {
        static_cast<unsigned>(w4.x), static_cast<unsigned>(w4.y),
        static_cast<unsigned>(w4.z), static_cast<unsigned>(w4.w),
    };
    const int ibase = g * 32;
    const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int kb = ibase + i * 8;
        const unsigned w = ww[i];
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w, bias), scale2), s_input2[(kb + 0) >> 1], acc[i]);
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 4, bias), scale2), s_input2[(kb + 2) >> 1], acc[i]);
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 8, bias), scale2), s_input2[(kb + 4) >> 1], acc[i]);
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 12, bias), scale2), s_input2[(kb + 6) >> 1], acc[i]);
    }

    const __half2 sum2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
    float v = __low2float(sum2) + __high2float(sum2);
    v = warp_sum(v);
    if (lane == 0) {
        s_partial[warp] = v;
    }
    __syncthreads();
    if (part == 0 && lane == 0) {
        output[row] = __float2bfloat16(s_partial[warp] + s_partial[warp + 1]);
    }
}

__global__ void qkv_w4a16_group128_split2_awq_persistent_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = warp_sum(sq);
    __shared__ float s_warp_sums[32];
    if (lane == 0) {
        s_warp_sums[warp] = sq_sum;
    }
    __syncthreads();
    float block_sq = (tid < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
    if (warp == 0) {
        block_sq = warp_sum(block_sq);
    }
    if (tid == 0) {
        rstd_shared = rsqrtf(block_sq / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();

    const float rstd = rstd_shared;
    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        const float x0 = bf16_to_float(residual[k + 0]) * bf16_to_float(norm_weight[k + 0]) * rstd;
        const float x1 = bf16_to_float(residual[k + 1]) * bf16_to_float(norm_weight[k + 1]) * rstd;
        s_input2[k2] = __halves2half2(__float2half(x0), __float2half(x1));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row_stride = gridDim.x * rows_per_block;
    const int base_row = blockIdx.x * rows_per_block + local_row;
    const int total_iters = (kQkvRows + row_stride - 1) / row_stride;
    const __half2 bias = __float2half2_rn(1032.0f);

    for (int iter = 0; iter < total_iters; ++iter) {
        const int row = base_row + iter * row_stride;
        const bool active = row < kQkvRows;
        if (active) {
            const int4* __restrict__ row_w =
                reinterpret_cast<const int4*>(packed_weight + row * kGatePackedK4);
            const __nv_bfloat16* __restrict__ row_s = scale + row * kGateGroups;
            __half2 acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                              __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
            const int g = lane + part * 32;
            const int4 w4 = row_w[g];
            const unsigned ww[4] = {
                static_cast<unsigned>(w4.x), static_cast<unsigned>(w4.y),
                static_cast<unsigned>(w4.z), static_cast<unsigned>(w4.w),
            };
            const int ibase = g * 32;
            const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kb = ibase + i * 8;
                const unsigned w = ww[i];
                acc[i] = __hfma2(__hmul2(dequant_awq_pair(w, bias), scale2), s_input2[(kb + 0) >> 1], acc[i]);
                acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 4, bias), scale2), s_input2[(kb + 2) >> 1], acc[i]);
                acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 8, bias), scale2), s_input2[(kb + 4) >> 1], acc[i]);
                acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 12, bias), scale2), s_input2[(kb + 6) >> 1], acc[i]);
            }

            const __half2 sum2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
            float v = __low2float(sum2) + __high2float(sum2);
            v = warp_sum(v);
            if (lane == 0) {
                s_partial[warp] = v;
            }
        }
        __syncthreads();
        if (active && part == 0 && lane == 0) {
            output[row] = __float2bfloat16(s_partial[warp] + s_partial[warp + 1]);
        }
        __syncthreads();
    }
}

__global__ void o_proj_add_w4a16_group128_split2_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ residual) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        s_input2[k2] = __halves2half2(
            __float2half(bf16_to_float(input[k + 0])),
            __float2half(bf16_to_float(input[k + 1])));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row = blockIdx.x * rows_per_block + local_row;
    if (row >= kOProjRows) {
        return;
    }
    const int4* __restrict__ row_w =
        reinterpret_cast<const int4*>(packed_weight + row * kGatePackedK4);
    const __nv_bfloat16* __restrict__ row_s = scale + row * kGateGroups;

    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 acc3 = __float2half2_rn(0.0f);
    const __half2 bias = __float2half2_rn(1032.0f);
    const int g = lane + part * 32;
    const int4 w4 = row_w[g];
    const unsigned ww[4] = {
        static_cast<unsigned>(w4.x),
        static_cast<unsigned>(w4.y),
        static_cast<unsigned>(w4.z),
        static_cast<unsigned>(w4.w),
    };
    const int ibase = g * 32;
    const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));

    const int kbase0 = ibase;
    acc0 = __hfma2(__hmul2(dequant_nibbles_01(ww[0], bias), scale2), s_input2[(kbase0 + 0) >> 1], acc0);
    acc0 = __hfma2(__hmul2(dequant_nibbles_23(ww[0], bias), scale2), s_input2[(kbase0 + 2) >> 1], acc0);
    acc0 = __hfma2(__hmul2(dequant_nibbles_45(ww[0], bias), scale2), s_input2[(kbase0 + 4) >> 1], acc0);
    acc0 = __hfma2(__hmul2(dequant_nibbles_67(ww[0], bias), scale2), s_input2[(kbase0 + 6) >> 1], acc0);
    const int kbase1 = ibase + 8;
    acc1 = __hfma2(__hmul2(dequant_nibbles_01(ww[1], bias), scale2), s_input2[(kbase1 + 0) >> 1], acc1);
    acc1 = __hfma2(__hmul2(dequant_nibbles_23(ww[1], bias), scale2), s_input2[(kbase1 + 2) >> 1], acc1);
    acc1 = __hfma2(__hmul2(dequant_nibbles_45(ww[1], bias), scale2), s_input2[(kbase1 + 4) >> 1], acc1);
    acc1 = __hfma2(__hmul2(dequant_nibbles_67(ww[1], bias), scale2), s_input2[(kbase1 + 6) >> 1], acc1);
    const int kbase2 = ibase + 16;
    acc2 = __hfma2(__hmul2(dequant_nibbles_01(ww[2], bias), scale2), s_input2[(kbase2 + 0) >> 1], acc2);
    acc2 = __hfma2(__hmul2(dequant_nibbles_23(ww[2], bias), scale2), s_input2[(kbase2 + 2) >> 1], acc2);
    acc2 = __hfma2(__hmul2(dequant_nibbles_45(ww[2], bias), scale2), s_input2[(kbase2 + 4) >> 1], acc2);
    acc2 = __hfma2(__hmul2(dequant_nibbles_67(ww[2], bias), scale2), s_input2[(kbase2 + 6) >> 1], acc2);
    const int kbase3 = ibase + 24;
    acc3 = __hfma2(__hmul2(dequant_nibbles_01(ww[3], bias), scale2), s_input2[(kbase3 + 0) >> 1], acc3);
    acc3 = __hfma2(__hmul2(dequant_nibbles_23(ww[3], bias), scale2), s_input2[(kbase3 + 2) >> 1], acc3);
    acc3 = __hfma2(__hmul2(dequant_nibbles_45(ww[3], bias), scale2), s_input2[(kbase3 + 4) >> 1], acc3);
    acc3 = __hfma2(__hmul2(dequant_nibbles_67(ww[3], bias), scale2), s_input2[(kbase3 + 6) >> 1], acc3);

    const __half2 sum2 = __hadd2(__hadd2(acc0, acc1), __hadd2(acc2, acc3));
    float acc = __low2float(sum2) + __high2float(sum2);
    acc = warp_sum(acc);
    if (lane == 0) {
        s_partial[warp] = acc;
    }
    __syncthreads();
    if (part == 0 && lane == 0) {
        const float added = bf16_to_float(residual[row]) + s_partial[warp] + s_partial[warp + 1];
        residual[row] = __float2bfloat16(added);
    }
}

// AWQ-interleaved nibble variant of o_proj_add_w4a16_group128_split2_kernel.
__global__ void o_proj_add_w4a16_group128_split2_awq_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ residual) {
    __shared__ __half2 s_input2[kHidden / 2];
    __shared__ float s_partial[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int k2 = tid; k2 < kHidden / 2; k2 += blockDim.x) {
        const int k = k2 * 2;
        s_input2[k2] = __halves2half2(
            __float2half(bf16_to_float(input[k + 0])),
            __float2half(bf16_to_float(input[k + 1])));
    }
    __syncthreads();

    const int local_row = warp >> 1;
    const int part = warp & 1;
    const int rows_per_block = blockDim.x >> 6;
    const int row = blockIdx.x * rows_per_block + local_row;
    if (row >= kOProjRows) {
        return;
    }
    const int4* __restrict__ row_w =
        reinterpret_cast<const int4*>(packed_weight + row * kGatePackedK4);
    const __nv_bfloat16* __restrict__ row_s = scale + row * kGateGroups;

    __half2 acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                      __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    const __half2 bias = __float2half2_rn(1032.0f);
    const int g = lane + part * 32;
    const int4 w4 = row_w[g];
    const unsigned ww[4] = {
        static_cast<unsigned>(w4.x), static_cast<unsigned>(w4.y),
        static_cast<unsigned>(w4.z), static_cast<unsigned>(w4.w),
    };
    const int ibase = g * 32;
    const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int kb = ibase + i * 8;
        const unsigned w = ww[i];
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w, bias), scale2), s_input2[(kb + 0) >> 1], acc[i]);
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 4, bias), scale2), s_input2[(kb + 2) >> 1], acc[i]);
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 8, bias), scale2), s_input2[(kb + 4) >> 1], acc[i]);
        acc[i] = __hfma2(__hmul2(dequant_awq_pair(w >> 12, bias), scale2), s_input2[(kb + 6) >> 1], acc[i]);
    }

    const __half2 sum2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
    float v = __low2float(sum2) + __high2float(sum2);
    v = warp_sum(v);
    if (lane == 0) {
        s_partial[warp] = v;
    }
    __syncthreads();
    if (part == 0 && lane == 0) {
        const float added = bf16_to_float(residual[row]) + s_partial[warp] + s_partial[warp + 1];
        residual[row] = __float2bfloat16(added);
    }
}

template <int WarpsPerBlock>
__global__ void down_w4a16_group128_probe_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    __shared__ float s_input[kIntermediate];
    __shared__ float s_partial[WarpsPerBlock];

    const int tid = threadIdx.x;
    for (int k = tid; k < kIntermediate; k += blockDim.x) {
        s_input[k] = bf16_to_float(input[k]);
    }

    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;

    float acc = 0.0f;
    if (row < kHidden) {
        const int4* __restrict__ row_w =
            reinterpret_cast<const int4*>(packed_weight + row * kPackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + row * kGroups;
        constexpr int kInt4VecsPerRow = kPackedK4 / 4;
        for (int g = lane; g < kInt4VecsPerRow; g += 32) {
            const int4 packed4 = row_w[g];
            const int* packed_words = reinterpret_cast<const int*>(&packed4);
            const int ibase = g * 32;
            const float group_scale = bf16_to_float(row_s[ibase / kGroupSize]);
            float local = 0.0f;
#pragma unroll
            for (int word_idx = 0; word_idx < 4; ++word_idx) {
                const unsigned word = prmt_identity_u32(static_cast<unsigned>(packed_words[word_idx]));
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const int w = signed_nibble_lop3(word, n * 4);
                    local += static_cast<float>(w) * s_input[ibase + word_idx * 8 + n];
                }
            }
            acc += local * group_scale;
        }
        acc = warp_sum(acc);
    }

    float out = 0.0f;
    if (row < kHidden && lane == 0) {
        out = acc + bf16_to_float(residual[row]);
        residual_out[row] = __float2bfloat16(out);
    }
    if (lane == 0) {
        s_partial[warp] = out * out;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sq = 0.0f;
#pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            block_sq += s_partial[w];
        }
        atomicAdd(sumsq, block_sq);
    }
}

template <int WarpsPerBlock>
__global__ void down_w4a16_group128_half2_probe_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    __shared__ __half s_input[kIntermediate];
    __shared__ float s_partial[WarpsPerBlock];

    const int tid = threadIdx.x;
    for (int k = tid; k < kIntermediate; k += blockDim.x) {
        s_input[k] = __float2half(bf16_to_float(input[k]));
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    __half2 acc2 = __float2half2_rn(0.0f);
    const __half2 bias = __float2half2_rn(1032.0f);

    if (row < kHidden) {
        const int4* __restrict__ row_w =
            reinterpret_cast<const int4*>(packed_weight + row * kPackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + row * kGroups;
        constexpr int kInt4VecsPerRow = kPackedK4 / 4;
        for (int g = lane; g < kInt4VecsPerRow; g += 32) {
            const int4 packed4 = row_w[g];
            const unsigned words[4] = {
                static_cast<unsigned>(packed4.x),
                static_cast<unsigned>(packed4.y),
                static_cast<unsigned>(packed4.z),
                static_cast<unsigned>(packed4.w),
            };
            const int ibase = g * 32;
            const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
#pragma unroll
            for (int word_idx = 0; word_idx < 4; ++word_idx) {
                const unsigned word = words[word_idx];
                const int kbase = ibase + word_idx * 8;
                const __half2 x01 = __halves2half2(s_input[kbase + 0], s_input[kbase + 1]);
                const __half2 x23 = __halves2half2(s_input[kbase + 2], s_input[kbase + 3]);
                const __half2 x45 = __halves2half2(s_input[kbase + 4], s_input[kbase + 5]);
                const __half2 x67 = __halves2half2(s_input[kbase + 6], s_input[kbase + 7]);
                acc2 = __hfma2(__hmul2(dequant_nibbles_01(word, bias), scale2), x01, acc2);
                acc2 = __hfma2(__hmul2(dequant_nibbles_23(word, bias), scale2), x23, acc2);
                acc2 = __hfma2(__hmul2(dequant_nibbles_45(word, bias), scale2), x45, acc2);
                acc2 = __hfma2(__hmul2(dequant_nibbles_67(word, bias), scale2), x67, acc2);
            }
        }
    }

    float acc = __low2float(acc2) + __high2float(acc2);
    acc = warp_sum(acc);

    float out = 0.0f;
    if (row < kHidden && lane == 0) {
        out = acc + bf16_to_float(residual[row]);
        residual_out[row] = __float2bfloat16(out);
    }
    if (lane == 0) {
        s_partial[warp] = out * out;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sq = 0.0f;
#pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            block_sq += s_partial[w];
        }
        atomicAdd(sumsq, block_sq);
    }
}

// AWQ-interleaved nibble variant of down_w4a16_group128_half2_probe_kernel.
template <int WarpsPerBlock>
__global__ void down_w4a16_group128_half2_awq_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    __shared__ __half s_input[kIntermediate];
    __shared__ float s_partial[WarpsPerBlock];

    const int tid = threadIdx.x;
    for (int k = tid; k < kIntermediate; k += blockDim.x) {
        s_input[k] = __float2half(bf16_to_float(input[k]));
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    __half2 acc2 = __float2half2_rn(0.0f);
    const __half2 bias = __float2half2_rn(1032.0f);

    if (row < kHidden) {
        const int4* __restrict__ row_w =
            reinterpret_cast<const int4*>(packed_weight + row * kPackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + row * kGroups;
        constexpr int kInt4VecsPerRow = kPackedK4 / 4;
        for (int g = lane; g < kInt4VecsPerRow; g += 32) {
            const int4 packed4 = row_w[g];
            const unsigned words[4] = {
                static_cast<unsigned>(packed4.x), static_cast<unsigned>(packed4.y),
                static_cast<unsigned>(packed4.z), static_cast<unsigned>(packed4.w),
            };
            const int ibase = g * 32;
            const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
#pragma unroll
            for (int word_idx = 0; word_idx < 4; ++word_idx) {
                const unsigned word = words[word_idx];
                const int kbase = ibase + word_idx * 8;
                const __half2 x01 = __halves2half2(s_input[kbase + 0], s_input[kbase + 1]);
                const __half2 x23 = __halves2half2(s_input[kbase + 2], s_input[kbase + 3]);
                const __half2 x45 = __halves2half2(s_input[kbase + 4], s_input[kbase + 5]);
                const __half2 x67 = __halves2half2(s_input[kbase + 6], s_input[kbase + 7]);
                acc2 = __hfma2(__hmul2(dequant_awq_pair(word, bias), scale2), x01, acc2);
                acc2 = __hfma2(__hmul2(dequant_awq_pair(word >> 4, bias), scale2), x23, acc2);
                acc2 = __hfma2(__hmul2(dequant_awq_pair(word >> 8, bias), scale2), x45, acc2);
                acc2 = __hfma2(__hmul2(dequant_awq_pair(word >> 12, bias), scale2), x67, acc2);
            }
        }
    }

    float acc = __low2float(acc2) + __high2float(acc2);
    acc = warp_sum(acc);

    float out = 0.0f;
    if (row < kHidden && lane == 0) {
        out = acc + bf16_to_float(residual[row]);
        residual_out[row] = __float2bfloat16(out);
    }
    if (lane == 0) {
        s_partial[warp] = out * out;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sq = 0.0f;
#pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            block_sq += s_partial[w];
        }
        atomicAdd(sumsq, block_sq);
    }
}

template <int WarpsPerBlock>
__global__ void lm_head_w4a16_stage1_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    float* __restrict__ partial_values,
    int* __restrict__ partial_indices,
    int vocab_size) {
    __shared__ __half2 s_input2[kLmHeadK / 2];
    __shared__ float s_values[WarpsPerBlock];
    __shared__ int s_indices[WarpsPerBlock];

    const int tid = threadIdx.x;
    for (int k2 = tid; k2 < kLmHeadK / 2; k2 += blockDim.x) {
        s_input2[k2] = __halves2half2(
            __float2half(bf16_to_float(hidden[k2 * 2 + 0])),
            __float2half(bf16_to_float(hidden[k2 * 2 + 1])));
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 acc3 = __float2half2_rn(0.0f);
    const __half2 bias = __float2half2_rn(1032.0f);

    if (row < vocab_size) {
        const int4* __restrict__ row_w =
            reinterpret_cast<const int4*>(packed_weight + static_cast<int64_t>(row) * kLmHeadPackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + static_cast<int64_t>(row) * kLmHeadGroups;
        constexpr int kInt4VecsPerRow = kLmHeadPackedK4 / 4;
        for (int g = lane; g < kInt4VecsPerRow; g += 32) {
            const int4 packed4 = row_w[g];
            const unsigned words[4] = {
                static_cast<unsigned>(packed4.x),
                static_cast<unsigned>(packed4.y),
                static_cast<unsigned>(packed4.z),
                static_cast<unsigned>(packed4.w),
            };
            const int ibase = g * 32;
            const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
            const int kbase0 = ibase;
            const __half2 x001 = s_input2[(kbase0 + 0) >> 1];
            const __half2 x023 = s_input2[(kbase0 + 2) >> 1];
            const __half2 x045 = s_input2[(kbase0 + 4) >> 1];
            const __half2 x067 = s_input2[(kbase0 + 6) >> 1];
            acc0 = __hfma2(__hmul2(dequant_nibbles_01(words[0], bias), scale2), x001, acc0);
            acc0 = __hfma2(__hmul2(dequant_nibbles_23(words[0], bias), scale2), x023, acc0);
            acc0 = __hfma2(__hmul2(dequant_nibbles_45(words[0], bias), scale2), x045, acc0);
            acc0 = __hfma2(__hmul2(dequant_nibbles_67(words[0], bias), scale2), x067, acc0);

            const int kbase1 = ibase + 8;
            const __half2 x101 = s_input2[(kbase1 + 0) >> 1];
            const __half2 x123 = s_input2[(kbase1 + 2) >> 1];
            const __half2 x145 = s_input2[(kbase1 + 4) >> 1];
            const __half2 x167 = s_input2[(kbase1 + 6) >> 1];
            acc1 = __hfma2(__hmul2(dequant_nibbles_01(words[1], bias), scale2), x101, acc1);
            acc1 = __hfma2(__hmul2(dequant_nibbles_23(words[1], bias), scale2), x123, acc1);
            acc1 = __hfma2(__hmul2(dequant_nibbles_45(words[1], bias), scale2), x145, acc1);
            acc1 = __hfma2(__hmul2(dequant_nibbles_67(words[1], bias), scale2), x167, acc1);

            const int kbase2 = ibase + 16;
            const __half2 x201 = s_input2[(kbase2 + 0) >> 1];
            const __half2 x223 = s_input2[(kbase2 + 2) >> 1];
            const __half2 x245 = s_input2[(kbase2 + 4) >> 1];
            const __half2 x267 = s_input2[(kbase2 + 6) >> 1];
            acc2 = __hfma2(__hmul2(dequant_nibbles_01(words[2], bias), scale2), x201, acc2);
            acc2 = __hfma2(__hmul2(dequant_nibbles_23(words[2], bias), scale2), x223, acc2);
            acc2 = __hfma2(__hmul2(dequant_nibbles_45(words[2], bias), scale2), x245, acc2);
            acc2 = __hfma2(__hmul2(dequant_nibbles_67(words[2], bias), scale2), x267, acc2);

            const int kbase3 = ibase + 24;
            const __half2 x301 = s_input2[(kbase3 + 0) >> 1];
            const __half2 x323 = s_input2[(kbase3 + 2) >> 1];
            const __half2 x345 = s_input2[(kbase3 + 4) >> 1];
            const __half2 x367 = s_input2[(kbase3 + 6) >> 1];
            acc3 = __hfma2(__hmul2(dequant_nibbles_01(words[3], bias), scale2), x301, acc3);
            acc3 = __hfma2(__hmul2(dequant_nibbles_23(words[3], bias), scale2), x323, acc3);
            acc3 = __hfma2(__hmul2(dequant_nibbles_45(words[3], bias), scale2), x345, acc3);
            acc3 = __hfma2(__hmul2(dequant_nibbles_67(words[3], bias), scale2), x367, acc3);
        }
    }

    acc0 = __hadd2(__hadd2(acc0, acc1), __hadd2(acc2, acc3));
    float acc = (row < vocab_size) ? (__low2float(acc0) + __high2float(acc0)) : -3.402823466e38F;
    acc = warp_sum(acc);
    if (lane == 0) {
        s_values[warp] = acc;
        s_indices[warp] = row;
    }
    __syncthreads();
    if (tid == 0) {
        float best = -3.402823466e38F;
        int best_idx = -1;
#pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            const float v = s_values[w];
            const int idx = s_indices[w];
            if (v > best) {
                best = v;
                best_idx = idx;
            }
        }
        partial_values[blockIdx.x] = best;
        partial_indices[blockIdx.x] = best_idx;
    }
}

template <int WarpsPerBlock>
__global__ void lm_head_w4a16_stage1_persistent_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    float* __restrict__ partial_values,
    int* __restrict__ partial_indices,
    int vocab_size) {
    __shared__ __half2 s_input2[kLmHeadK / 2];
    __shared__ float s_values[WarpsPerBlock];
    __shared__ int s_indices[WarpsPerBlock];

    const int tid = threadIdx.x;
    for (int k2 = tid; k2 < kLmHeadK / 2; k2 += blockDim.x) {
        s_input2[k2] = __halves2half2(
            __float2half(bf16_to_float(hidden[k2 * 2 + 0])),
            __float2half(bf16_to_float(hidden[k2 * 2 + 1])));
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __half2 bias = __float2half2_rn(1032.0f);
    constexpr int kInt4VecsPerRow = kLmHeadPackedK4 / 4;
    float best = -3.402823466e38F;
    int best_idx = -1;

    for (int row = blockIdx.x * WarpsPerBlock + warp;
         row < vocab_size;
         row += gridDim.x * WarpsPerBlock) {
        __half2 acc0 = __float2half2_rn(0.0f);
        const int4* __restrict__ row_w =
            reinterpret_cast<const int4*>(packed_weight + static_cast<int64_t>(row) * kLmHeadPackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + static_cast<int64_t>(row) * kLmHeadGroups;
        for (int g = lane; g < kInt4VecsPerRow; g += 32) {
            const int4 packed4 = row_w[g];
            const unsigned words[4] = {
                static_cast<unsigned>(packed4.x),
                static_cast<unsigned>(packed4.y),
                static_cast<unsigned>(packed4.z),
                static_cast<unsigned>(packed4.w),
            };
            const int ibase = g * 32;
            const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));

            __half2 gacc0 = __float2half2_rn(0.0f);
            __half2 gacc1 = __float2half2_rn(0.0f);
            __half2 gacc2 = __float2half2_rn(0.0f);
            __half2 gacc3 = __float2half2_rn(0.0f);

            const int kbase0 = ibase;
            gacc0 = __hfma2(dequant_nibbles_01(words[0], bias), s_input2[(kbase0 + 0) >> 1], gacc0);
            gacc0 = __hfma2(dequant_nibbles_23(words[0], bias), s_input2[(kbase0 + 2) >> 1], gacc0);
            gacc0 = __hfma2(dequant_nibbles_45(words[0], bias), s_input2[(kbase0 + 4) >> 1], gacc0);
            gacc0 = __hfma2(dequant_nibbles_67(words[0], bias), s_input2[(kbase0 + 6) >> 1], gacc0);

            const int kbase1 = ibase + 8;
            gacc1 = __hfma2(dequant_nibbles_01(words[1], bias), s_input2[(kbase1 + 0) >> 1], gacc1);
            gacc1 = __hfma2(dequant_nibbles_23(words[1], bias), s_input2[(kbase1 + 2) >> 1], gacc1);
            gacc1 = __hfma2(dequant_nibbles_45(words[1], bias), s_input2[(kbase1 + 4) >> 1], gacc1);
            gacc1 = __hfma2(dequant_nibbles_67(words[1], bias), s_input2[(kbase1 + 6) >> 1], gacc1);

            const int kbase2 = ibase + 16;
            gacc2 = __hfma2(dequant_nibbles_01(words[2], bias), s_input2[(kbase2 + 0) >> 1], gacc2);
            gacc2 = __hfma2(dequant_nibbles_23(words[2], bias), s_input2[(kbase2 + 2) >> 1], gacc2);
            gacc2 = __hfma2(dequant_nibbles_45(words[2], bias), s_input2[(kbase2 + 4) >> 1], gacc2);
            gacc2 = __hfma2(dequant_nibbles_67(words[2], bias), s_input2[(kbase2 + 6) >> 1], gacc2);

            const int kbase3 = ibase + 24;
            gacc3 = __hfma2(dequant_nibbles_01(words[3], bias), s_input2[(kbase3 + 0) >> 1], gacc3);
            gacc3 = __hfma2(dequant_nibbles_23(words[3], bias), s_input2[(kbase3 + 2) >> 1], gacc3);
            gacc3 = __hfma2(dequant_nibbles_45(words[3], bias), s_input2[(kbase3 + 4) >> 1], gacc3);
            gacc3 = __hfma2(dequant_nibbles_67(words[3], bias), s_input2[(kbase3 + 6) >> 1], gacc3);

            acc0 = __hfma2(__hadd2(__hadd2(gacc0, gacc1), __hadd2(gacc2, gacc3)), scale2, acc0);
        }
        float acc = __low2float(acc0) + __high2float(acc0);
        acc = warp_sum(acc);
        if (lane == 0 && acc > best) {
            best = acc;
            best_idx = row;
        }
    }
    if (lane == 0) {
        s_values[warp] = best;
        s_indices[warp] = best_idx;
    }
    __syncthreads();
    if (tid == 0) {
        float block_best = -3.402823466e38F;
        int block_best_idx = -1;
#pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            const float v = s_values[w];
            const int idx = s_indices[w];
            if (v > block_best) {
                block_best = v;
                block_best_idx = idx;
            }
        }
        partial_values[blockIdx.x] = block_best;
        partial_indices[blockIdx.x] = block_best_idx;
    }
}

__global__ void lm_head_argmax_stage2_kernel(
    const float* __restrict__ partial_values,
    const int* __restrict__ partial_indices,
    void* __restrict__ output_token,
    int num_blocks,
    bool output_i64) {
    float best = -3.402823466e38F;
    int best_idx = -1;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        const float v = partial_values[i];
        const int idx = partial_indices[i];
        if (v > best) {
            best = v;
            best_idx = idx;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_v = __shfl_down_sync(0xffffffff, best, offset);
        const int other_i = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_v > best) {
            best = other_v;
            best_idx = other_i;
        }
    }
    __shared__ float s_values[32];
    __shared__ int s_indices[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        s_values[warp] = best;
        s_indices[warp] = best_idx;
    }
    __syncthreads();
    if (warp == 0) {
        best = (lane < (blockDim.x + 31) / 32) ? s_values[lane] : -3.402823466e38F;
        best_idx = (lane < (blockDim.x + 31) / 32) ? s_indices[lane] : -1;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_v = __shfl_down_sync(0xffffffff, best, offset);
            const int other_i = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (other_v > best) {
                best = other_v;
                best_idx = other_i;
            }
        }
        if (lane == 0) {
            if (output_i64) {
                *reinterpret_cast<int64_t*>(output_token) = static_cast<int64_t>(best_idx);
            } else {
                *reinterpret_cast<int*>(output_token) = best_idx;
            }
        }
    }
}

template <int WarpsPerBlock>
__global__ void lm_head_w4a16_persistent_atomic_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    unsigned long long* __restrict__ best_pair,
    unsigned* __restrict__ done_counter,
    void* __restrict__ output_token,
    int vocab_size,
    bool output_i64) {
    __shared__ __half2 s_input2[kLmHeadK / 2];
    __shared__ float s_values[WarpsPerBlock];
    __shared__ int s_indices[WarpsPerBlock];

    const int tid = threadIdx.x;
    for (int k2 = tid; k2 < kLmHeadK / 2; k2 += blockDim.x) {
        s_input2[k2] = __halves2half2(
            __float2half(bf16_to_float(hidden[k2 * 2 + 0])),
            __float2half(bf16_to_float(hidden[k2 * 2 + 1])));
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const __half2 bias = __float2half2_rn(1032.0f);
    constexpr int kInt4VecsPerRow = kLmHeadPackedK4 / 4;
    float best = -3.402823466e38F;
    int best_idx = -1;

    for (int row = blockIdx.x * WarpsPerBlock + warp;
         row < vocab_size;
         row += gridDim.x * WarpsPerBlock) {
        __half2 acc0 = __float2half2_rn(0.0f);
        const int4* __restrict__ row_w =
            reinterpret_cast<const int4*>(packed_weight + static_cast<int64_t>(row) * kLmHeadPackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + static_cast<int64_t>(row) * kLmHeadGroups;
        for (int g = lane; g < kInt4VecsPerRow; g += 32) {
            const int4 packed4 = row_w[g];
            const unsigned words[4] = {
                static_cast<unsigned>(packed4.x),
                static_cast<unsigned>(packed4.y),
                static_cast<unsigned>(packed4.z),
                static_cast<unsigned>(packed4.w),
            };
            const int ibase = g * 32;
            const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
            __half2 gacc0 = __float2half2_rn(0.0f);
            __half2 gacc1 = __float2half2_rn(0.0f);
            __half2 gacc2 = __float2half2_rn(0.0f);
            __half2 gacc3 = __float2half2_rn(0.0f);
            const int kbase0 = ibase;
            gacc0 = __hfma2(dequant_nibbles_01(words[0], bias), s_input2[(kbase0 + 0) >> 1], gacc0);
            gacc0 = __hfma2(dequant_nibbles_23(words[0], bias), s_input2[(kbase0 + 2) >> 1], gacc0);
            gacc0 = __hfma2(dequant_nibbles_45(words[0], bias), s_input2[(kbase0 + 4) >> 1], gacc0);
            gacc0 = __hfma2(dequant_nibbles_67(words[0], bias), s_input2[(kbase0 + 6) >> 1], gacc0);
            const int kbase1 = ibase + 8;
            gacc1 = __hfma2(dequant_nibbles_01(words[1], bias), s_input2[(kbase1 + 0) >> 1], gacc1);
            gacc1 = __hfma2(dequant_nibbles_23(words[1], bias), s_input2[(kbase1 + 2) >> 1], gacc1);
            gacc1 = __hfma2(dequant_nibbles_45(words[1], bias), s_input2[(kbase1 + 4) >> 1], gacc1);
            gacc1 = __hfma2(dequant_nibbles_67(words[1], bias), s_input2[(kbase1 + 6) >> 1], gacc1);
            const int kbase2 = ibase + 16;
            gacc2 = __hfma2(dequant_nibbles_01(words[2], bias), s_input2[(kbase2 + 0) >> 1], gacc2);
            gacc2 = __hfma2(dequant_nibbles_23(words[2], bias), s_input2[(kbase2 + 2) >> 1], gacc2);
            gacc2 = __hfma2(dequant_nibbles_45(words[2], bias), s_input2[(kbase2 + 4) >> 1], gacc2);
            gacc2 = __hfma2(dequant_nibbles_67(words[2], bias), s_input2[(kbase2 + 6) >> 1], gacc2);
            const int kbase3 = ibase + 24;
            gacc3 = __hfma2(dequant_nibbles_01(words[3], bias), s_input2[(kbase3 + 0) >> 1], gacc3);
            gacc3 = __hfma2(dequant_nibbles_23(words[3], bias), s_input2[(kbase3 + 2) >> 1], gacc3);
            gacc3 = __hfma2(dequant_nibbles_45(words[3], bias), s_input2[(kbase3 + 4) >> 1], gacc3);
            gacc3 = __hfma2(dequant_nibbles_67(words[3], bias), s_input2[(kbase3 + 6) >> 1], gacc3);
            acc0 = __hfma2(__hadd2(__hadd2(gacc0, gacc1), __hadd2(gacc2, gacc3)), scale2, acc0);
        }
        float acc = __low2float(acc0) + __high2float(acc0);
        acc = warp_sum(acc);
        if (lane == 0 && acc > best) {
            best = acc;
            best_idx = row;
        }
    }
    if (lane == 0) {
        s_values[warp] = best;
        s_indices[warp] = best_idx;
    }
    __syncthreads();
    if (tid == 0) {
        float block_best = -3.402823466e38F;
        int block_best_idx = -1;
#pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            if (s_values[w] > block_best) {
                block_best = s_values[w];
                block_best_idx = s_indices[w];
            }
        }
        const unsigned long long packed =
            (static_cast<unsigned long long>(ordered_float_bits(block_best)) << 32)
            | static_cast<unsigned>(block_best_idx);
        atomicMax(best_pair, packed);
        __threadfence();
        const unsigned ticket = atomicInc(done_counter, 0x7fffffffu);
        if (ticket == static_cast<unsigned>(gridDim.x - 1)) {
            const int best_token = static_cast<int>(*best_pair & 0xffffffffu);
            if (output_i64) {
                *reinterpret_cast<int64_t*>(output_token) = static_cast<int64_t>(best_token);
            } else {
                *reinterpret_cast<int*>(output_token) = best_token;
            }
            *done_counter = 0u;
        }
    }
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void instruction_probe_cuda(torch::Tensor out) {
    check_cuda_tensor(out, "out");
    TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32");
    TORCH_CHECK(out.numel() >= 2, "out must have at least 2 elements");
    auto stream = at::cuda::getCurrentCUDAStream();
    instruction_probe_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned*>(out.data_ptr<int>()));
}

void w4_weight_stream_cp_async_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor checksum,
    int blocks) {
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(checksum, "checksum");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(checksum.scalar_type() == at::kInt, "checksum must be int32");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kHidden && packed_weight.size(1) == kPackedK4,
                "packed_weight must be [2048, 768]");
    const int launch_blocks = blocks > 0 ? blocks : 128;
    TORCH_CHECK(checksum.numel() >= launch_blocks, "checksum must have at least blocks elements");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(checksum.data_ptr<int>(), 0, launch_blocks * sizeof(int), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));
    const int total_vecs = kW4WeightBytes / static_cast<int>(sizeof(int4));
    w4_weight_stream_cp_async_probe_kernel<<<launch_blocks, 256, 0, stream>>>(
        reinterpret_cast<const int4*>(packed_weight.data_ptr<int>()),
        reinterpret_cast<unsigned*>(checksum.data_ptr<int>()),
        total_vecs);
}

void w4_weight_stream_any_cp_async_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor checksum,
    int blocks,
    int stages) {
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(checksum, "checksum");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(checksum.scalar_type() == at::kInt, "checksum must be int32");
    TORCH_CHECK(packed_weight.numel() % 4 == 0, "packed_weight numel must be divisible by 4 int32 words");
    const int launch_blocks = blocks > 0 ? blocks : 256;
    TORCH_CHECK(checksum.numel() >= launch_blocks, "checksum must have at least blocks elements");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(checksum.data_ptr<int>(), 0, launch_blocks * sizeof(int), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));
    const int total_vecs = static_cast<int>(packed_weight.numel() / 4);
    if (stages >= 8) {
        w4_weight_stream_any_cp_async_probe_kernel<8><<<launch_blocks, 256, 0, stream>>>(
            reinterpret_cast<const int4*>(packed_weight.data_ptr<int>()),
            reinterpret_cast<unsigned*>(checksum.data_ptr<int>()),
            total_vecs);
    } else if (stages >= 4) {
        w4_weight_stream_any_cp_async_probe_kernel<4><<<launch_blocks, 256, 0, stream>>>(
            reinterpret_cast<const int4*>(packed_weight.data_ptr<int>()),
            reinterpret_cast<unsigned*>(checksum.data_ptr<int>()),
            total_vecs);
    } else if (stages >= 2) {
        w4_weight_stream_any_cp_async_probe_kernel<2><<<launch_blocks, 256, 0, stream>>>(
            reinterpret_cast<const int4*>(packed_weight.data_ptr<int>()),
            reinterpret_cast<unsigned*>(checksum.data_ptr<int>()),
            total_vecs);
    } else {
        w4_weight_stream_any_cp_async_probe_kernel<1><<<launch_blocks, 256, 0, stream>>>(
            reinterpret_cast<const int4*>(packed_weight.data_ptr<int>()),
            reinterpret_cast<unsigned*>(checksum.data_ptr<int>()),
            total_vecs);
    }
}

void down_w4a16_group128_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block) {
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(input, "input");
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(residual_out, "residual_out");
    check_cuda_tensor(sumsq, "sumsq");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kHidden && packed_weight.size(1) == kPackedK4,
                "packed_weight must be [2048, 768]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kHidden && scale.size(1) == kGroups,
                "scale must be [2048, 48]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have 6144 elements");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have 2048 elements");
    TORCH_CHECK(residual_out.numel() == kHidden, "residual_out must have 2048 elements");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq must have at least one element");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));

    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 8;
    const int grid = (kHidden + warps - 1) / warps;
    if (warps == 4) {
        down_w4a16_group128_probe_kernel<4><<<grid, 4 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    } else if (warps == 16) {
        down_w4a16_group128_probe_kernel<16><<<grid, 16 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    } else if (warps == 32) {
        down_w4a16_group128_probe_kernel<32><<<grid, 32 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    } else {
        down_w4a16_group128_probe_kernel<8><<<grid, 8 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    }
}

void down_w4a16_group128_half2_probe_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block) {
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(input, "input");
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(residual_out, "residual_out");
    check_cuda_tensor(sumsq, "sumsq");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kHidden && packed_weight.size(1) == kPackedK4,
                "packed_weight must be [2048, 768]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kHidden && scale.size(1) == kGroups,
                "scale must be [2048, 48]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have 6144 elements");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have 2048 elements");
    TORCH_CHECK(residual_out.numel() == kHidden, "residual_out must have 2048 elements");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq must have at least one element");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));

    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 8;
    const int grid = (kHidden + warps - 1) / warps;
    if (warps == 4) {
        down_w4a16_group128_half2_probe_kernel<4><<<grid, 4 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    } else if (warps == 16) {
        down_w4a16_group128_half2_probe_kernel<16><<<grid, 16 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    } else if (warps == 32) {
        down_w4a16_group128_half2_probe_kernel<32><<<grid, 32 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    } else {
        down_w4a16_group128_half2_probe_kernel<8><<<grid, 8 * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    }
}

// ILP variant of down_w4a16_group128_half2_awq_kernel: identical structure
// (1 warp per output row, K=6144 reduced in-warp) but accumulates each of the
// four nibble-pair lanes into its own half2 so the FMA chain is 4 independent
// ~6-deep chains instead of one ~24-deep chain. Helps when the single-acc
// version is FMA-latency-bound below the memory ceiling. Numerics differ only
// by within-warp reassociation (same magnitude as the other AWQ kernels).
template <int WarpsPerBlock>
__global__ void down_w4a16_group128_half2_awq_ilp_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    __shared__ __half s_input[kIntermediate];
    __shared__ float s_partial[WarpsPerBlock];

    const int tid = threadIdx.x;
    for (int k = tid; k < kIntermediate; k += blockDim.x) {
        s_input[k] = __float2half(bf16_to_float(input[k]));
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    __half2 acc[4] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f),
                      __float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    const __half2 bias = __float2half2_rn(1032.0f);

    if (row < kHidden) {
        const int4* __restrict__ row_w =
            reinterpret_cast<const int4*>(packed_weight + row * kPackedK4);
        const __nv_bfloat16* __restrict__ row_s = scale + row * kGroups;
        constexpr int kInt4VecsPerRow = kPackedK4 / 4;
        for (int g = lane; g < kInt4VecsPerRow; g += 32) {
            const int4 packed4 = row_w[g];
            const unsigned words[4] = {
                static_cast<unsigned>(packed4.x), static_cast<unsigned>(packed4.y),
                static_cast<unsigned>(packed4.z), static_cast<unsigned>(packed4.w),
            };
            const int ibase = g * 32;
            const __half2 scale2 = __float2half2_rn(bf16_to_float(row_s[ibase / kGroupSize]));
#pragma unroll
            for (int word_idx = 0; word_idx < 4; ++word_idx) {
                const unsigned word = words[word_idx];
                const int kbase = ibase + word_idx * 8;
                const __half2 x01 = __halves2half2(s_input[kbase + 0], s_input[kbase + 1]);
                const __half2 x23 = __halves2half2(s_input[kbase + 2], s_input[kbase + 3]);
                const __half2 x45 = __halves2half2(s_input[kbase + 4], s_input[kbase + 5]);
                const __half2 x67 = __halves2half2(s_input[kbase + 6], s_input[kbase + 7]);
                acc[word_idx] = __hfma2(__hmul2(dequant_awq_pair(word, bias), scale2), x01, acc[word_idx]);
                acc[word_idx] = __hfma2(__hmul2(dequant_awq_pair(word >> 4, bias), scale2), x23, acc[word_idx]);
                acc[word_idx] = __hfma2(__hmul2(dequant_awq_pair(word >> 8, bias), scale2), x45, acc[word_idx]);
                acc[word_idx] = __hfma2(__hmul2(dequant_awq_pair(word >> 12, bias), scale2), x67, acc[word_idx]);
            }
        }
    }

    const __half2 sum2 = __hadd2(__hadd2(acc[0], acc[1]), __hadd2(acc[2], acc[3]));
    float accf = __low2float(sum2) + __high2float(sum2);
    accf = warp_sum(accf);

    float out = 0.0f;
    if (row < kHidden && lane == 0) {
        out = accf + bf16_to_float(residual[row]);
        residual_out[row] = __float2bfloat16(out);
    }
    if (lane == 0) {
        s_partial[warp] = out * out;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sq = 0.0f;
#pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            block_sq += s_partial[w];
        }
        atomicAdd(sumsq, block_sq);
    }
}

void down_w4a16_group128_half2_awq_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block) {
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(input, "input");
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(residual_out, "residual_out");
    check_cuda_tensor(sumsq, "sumsq");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kHidden && packed_weight.size(1) == kPackedK4,
                "packed_weight must be [2048, 768]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kHidden && scale.size(1) == kGroups,
                "scale must be [2048, 48]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have 6144 elements");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have 2048 elements");
    TORCH_CHECK(residual_out.numel() == kHidden, "residual_out must have 2048 elements");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq must have at least one element");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));

    const int warps = (warps_per_block == 4 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 8;
    const int grid = (kHidden + warps - 1) / warps;
    const bool ilp = env_flag_enabled("JUNKRAT_W4A16_DOWN_ILP", false);
#define JK_DOWN_AWQ_LAUNCH(W)                                                          \
    do { if (ilp) {                                                                    \
        down_w4a16_group128_half2_awq_ilp_kernel<W><<<grid, (W) * 32, 0, stream>>>(    \
            packed_weight.data_ptr<int>(),                                            \
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),   \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),   \
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),\
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),  \
            sumsq.data_ptr<float>());                                                 \
    } else {                                                                          \
        down_w4a16_group128_half2_awq_kernel<W><<<grid, (W) * 32, 0, stream>>>(        \
            packed_weight.data_ptr<int>(),                                            \
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),   \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),   \
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),\
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),  \
            sumsq.data_ptr<float>());                                                 \
    } } while (0)
    if (warps == 4) {
        JK_DOWN_AWQ_LAUNCH(4);
    } else if (warps == 16) {
        JK_DOWN_AWQ_LAUNCH(16);
    } else if (warps == 32) {
        JK_DOWN_AWQ_LAUNCH(32);
    } else {
        JK_DOWN_AWQ_LAUNCH(8);
    }
#undef JK_DOWN_AWQ_LAUNCH
}

void gate_up_w4a16_group128_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block) {
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(norm_weight, "norm_weight");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have 2048 elements");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kIntermediate * 2 && packed_weight.size(1) == kGatePackedK4,
                "packed_weight must be [12288, 256]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kIntermediate * 2 && scale.size(1) == kGateGroups,
                "scale must be [12288, 16]");
    TORCH_CHECK(output.numel() == kIntermediate, "output must have 6144 elements");

    auto stream = at::cuda::getCurrentCUDAStream();
    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 32;
    const int grid = (kIntermediate + warps - 1) / warps;
    if (warps == 4) {
        gate_up_w4a16_group128_kernel<4><<<grid, 4 * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else {
        const int rows_per_block = warps / 2;
        const int grid_split2 = (kIntermediate + rows_per_block - 1) / rows_per_block;
        gate_up_w4a16_group128_split2_kernel<<<grid_split2, warps * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    }
}

void gate_up_w4a16_group128_awq_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block) {
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(norm_weight, "norm_weight");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have 2048 elements");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kIntermediate * 2 && packed_weight.size(1) == kGatePackedK4,
                "packed_weight must be [12288, 256]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kIntermediate * 2 && scale.size(1) == kGateGroups,
                "scale must be [12288, 16]");
    TORCH_CHECK(output.numel() == kIntermediate, "output must have 6144 elements");

    auto stream = at::cuda::getCurrentCUDAStream();
    const int warps = (warps_per_block >= 8 && warps_per_block <= 32 && (warps_per_block & 1) == 0)
        ? warps_per_block
        : 32;
    const int rows_per_block = warps / 2;
    const int row_blocks = (kIntermediate + rows_per_block - 1) / rows_per_block;

    if (env_flag_enabled("JUNKRAT_W4A16_GATE_UP_UNSPLIT", false)) {
        const int unsplit_rows_per_block = warps;  // 1 warp per row
        const int unsplit_row_blocks = (kIntermediate + unsplit_rows_per_block - 1) / unsplit_rows_per_block;
        int launch_blocks = env_int_value("JUNKRAT_W4A16_GATE_UP_UNSPLIT_BLOCKS", 216);
        if (launch_blocks > unsplit_row_blocks) launch_blocks = unsplit_row_blocks;
        if (launch_blocks < 1) launch_blocks = 1;
        const int threads = warps * 32;
        auto res_p = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>());
        auto nw_p = reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>());
        auto pk_p = packed_weight.data_ptr<int>();
        auto sc_p = reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>());
        auto out_p = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
        const float epsf = static_cast<float>(eps);
        const bool pf = env_flag_enabled("JUNKRAT_W4A16_GATE_UP_UNSPLIT_PREFETCH", false);
#define GU_LAUNCH(W, PF) gate_up_w4a16_group128_unsplit_awq_persistent_kernel<W, PF><<<launch_blocks, threads, 0, stream>>>(res_p, nw_p, pk_p, sc_p, out_p, epsf)
#define GU_WARP(W) do { if (pf) GU_LAUNCH(W, true); else GU_LAUNCH(W, false); } while (0)
        switch (warps) {
            case 8: GU_WARP(8); break;
            case 16: GU_WARP(16); break;
            case 24: GU_WARP(24); break;
            default: GU_WARP(32); break;
        }
#undef GU_WARP
#undef GU_LAUNCH
        return;
    }

    if (env_flag_enabled("JUNKRAT_W4A16_GATE_UP_CP_ASYNC", false)) {
        int launch_blocks = env_int_value("JUNKRAT_W4A16_GATE_UP_CP_ASYNC_BLOCKS", 128);
        if (launch_blocks > row_blocks) {
            launch_blocks = row_blocks;
        }
        if (launch_blocks < 1) {
            launch_blocks = 1;
        }
        int stages = env_int_value("JUNKRAT_W4A16_GATE_UP_CP_ASYNC_STAGES", 2);
        if (stages < 2) {
            stages = 2;
        }
        if (stages > 3) {
            stages = 3;
        }
        const int threads = warps * 32;
        const int smem_bytes = 2 * stages * threads * static_cast<int>(sizeof(int4));
        if (stages == 3) {
            const cudaError_t attr_status = cudaFuncSetAttribute(
                gate_up_w4a16_group128_split2_awq_cp_async_kernel<3>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            if (attr_status == cudaSuccess) {
                gate_up_w4a16_group128_split2_awq_cp_async_kernel<3><<<launch_blocks, threads, smem_bytes, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
                    packed_weight.data_ptr<int>(),
                    reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                    static_cast<float>(eps));
                return;
            }
        } else {
            const cudaError_t attr_status = cudaFuncSetAttribute(
                gate_up_w4a16_group128_split2_awq_cp_async_kernel<2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            if (attr_status == cudaSuccess) {
                gate_up_w4a16_group128_split2_awq_cp_async_kernel<2><<<launch_blocks, threads, smem_bytes, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
                    packed_weight.data_ptr<int>(),
                    reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                    static_cast<float>(eps));
                return;
            }
        }
    }

    if (env_flag_enabled("JUNKRAT_W4A16_GATE_UP_PF", false)) {
        int launch_blocks = env_int_value("JUNKRAT_W4A16_GATE_UP_PF_BLOCKS", 128);
        if (launch_blocks > row_blocks) launch_blocks = row_blocks;
        if (launch_blocks < 1) launch_blocks = 1;
        gate_up_w4a16_group128_split2_awq_persistent_pf_kernel<<<launch_blocks, warps * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
        return;
    }

    if (env_flag_enabled("JUNKRAT_W4A16_GATE_UP_PERSISTENT", true)) {
        int launch_blocks = env_int_value("JUNKRAT_W4A16_GATE_UP_PERSISTENT_BLOCKS", row_blocks / 3);
        if (launch_blocks > row_blocks) {
            launch_blocks = row_blocks;
        }
        if (launch_blocks < 1) {
            launch_blocks = 1;
        }
        gate_up_w4a16_group128_split2_awq_persistent_tpl_kernel<false><<<launch_blocks, warps * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
        return;
    }

    gate_up_w4a16_group128_split2_awq_kernel<<<row_blocks, warps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        packed_weight.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<float>(eps));
}

void gate_up_w4a16_group128_unsplit_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block) {
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(norm_weight, "norm_weight");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 32;
    const int grid = (kIntermediate + warps - 1) / warps;
    if (warps == 4) {
        gate_up_w4a16_group128_kernel<4><<<grid, 4 * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else if (warps == 8) {
        gate_up_w4a16_group128_kernel<8><<<grid, 8 * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else if (warps == 16) {
        gate_up_w4a16_group128_kernel<16><<<grid, 16 * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else {
        gate_up_w4a16_group128_kernel<32><<<grid, 32 * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    }
}

void qkv_w4a16_group128_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block) {
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(norm_weight, "norm_weight");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have 2048 elements");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kQkvRows && packed_weight.size(1) == kGatePackedK4,
                "packed_weight must be [4096, 256]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kQkvRows && scale.size(1) == kGateGroups,
                "scale must be [4096, 16]");
    TORCH_CHECK(output.numel() >= kQkvRows, "output must have at least 4096 elements");

    const int warps = (warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 32;
    const int rows_per_block = warps / 2;
    const int grid = (kQkvRows + rows_per_block - 1) / rows_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    qkv_w4a16_group128_split2_kernel<<<grid, warps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        packed_weight.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<float>(eps));
}

void qkv_w4a16_group128_awq_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block) {
    check_cuda_tensor(residual, "residual");
    check_cuda_tensor(norm_weight, "norm_weight");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have 2048 elements");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kQkvRows && packed_weight.size(1) == kGatePackedK4,
                "packed_weight must be [4096, 256]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kQkvRows && scale.size(1) == kGateGroups,
                "scale must be [4096, 16]");
    TORCH_CHECK(output.numel() >= kQkvRows, "output must have at least 4096 elements");

    const int warps = (warps_per_block >= 8 && warps_per_block <= 32 && (warps_per_block & 1) == 0)
        ? warps_per_block
        : 32;
    const int rows_per_block = warps / 2;
    const int row_blocks = (kQkvRows + rows_per_block - 1) / rows_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (env_flag_enabled("JUNKRAT_W4A16_QKV_UNSPLIT", true)) {
        const int unsplit_row_blocks = (kQkvRows + warps - 1) / warps;
        int launch_blocks = env_int_value("JUNKRAT_W4A16_QKV_UNSPLIT_BLOCKS", 162);
        if (launch_blocks > unsplit_row_blocks) launch_blocks = unsplit_row_blocks;
        if (launch_blocks < 1) launch_blocks = 1;
        const int threads = warps * 32;
        auto res_p = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>());
        auto nw_p = reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>());
        auto pk_p = packed_weight.data_ptr<int>();
        auto sc_p = reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>());
        auto out_p = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
        const float epsf = static_cast<float>(eps);
        const bool pf = env_flag_enabled("JUNKRAT_W4A16_QKV_UNSPLIT_PREFETCH", false);
#define QKV_LAUNCH(W, PF) qkv_w4a16_group128_unsplit_awq_persistent_kernel<W, PF><<<launch_blocks, threads, 0, stream>>>(res_p, nw_p, pk_p, sc_p, out_p, epsf)
#define QKV_WARP(W) do { if (pf) QKV_LAUNCH(W, true); else QKV_LAUNCH(W, false); } while (0)
        switch (warps) {
            case 8: QKV_WARP(8); break;
            case 16: QKV_WARP(16); break;
            case 24: QKV_WARP(24); break;
            default: QKV_WARP(32); break;
        }
#undef QKV_WARP
#undef QKV_LAUNCH
        return;
    }
    if (env_flag_enabled("JUNKRAT_W4A16_QKV_PERSISTENT", false)) {
        int launch_blocks = env_int_value("JUNKRAT_W4A16_QKV_PERSISTENT_BLOCKS", row_blocks / 3);
        if (launch_blocks > row_blocks) {
            launch_blocks = row_blocks;
        }
        if (launch_blocks < 1) {
            launch_blocks = 1;
        }
        qkv_w4a16_group128_split2_awq_persistent_kernel<<<launch_blocks, warps * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
        return;
    }
    qkv_w4a16_group128_split2_awq_kernel<<<row_blocks, warps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        packed_weight.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<float>(eps));
}

void o_proj_add_w4a16_group128_cuda(
    torch::Tensor input,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor residual,
    int warps_per_block) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(residual, "residual");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(input.numel() == kHidden, "input must have 2048 elements");
    TORCH_CHECK(residual.numel() >= kOProjRows, "residual must have at least 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kOProjRows && packed_weight.size(1) == kGatePackedK4,
                "packed_weight must be [2048, 256]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kOProjRows && scale.size(1) == kGateGroups,
                "scale must be [2048, 16]");

    const int warps = (warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 32;
    const int rows_per_block = warps / 2;
    const int grid = (kOProjRows + rows_per_block - 1) / rows_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    o_proj_add_w4a16_group128_split2_kernel<<<grid, warps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        packed_weight.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()));
}

void o_proj_add_w4a16_group128_awq_cuda(
    torch::Tensor input,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor residual,
    int warps_per_block) {
    check_cuda_tensor(input, "input");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(residual, "residual");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(input.numel() == kHidden, "input must have 2048 elements");
    TORCH_CHECK(residual.numel() >= kOProjRows, "residual must have at least 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kOProjRows && packed_weight.size(1) == kGatePackedK4,
                "packed_weight must be [2048, 256]");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == kOProjRows && scale.size(1) == kGateGroups,
                "scale must be [2048, 16]");

    const int warps = (warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 24 || warps_per_block == 32)
        ? warps_per_block
        : 32;
    const int rows_per_block = warps / 2;
    const int grid = (kOProjRows + rows_per_block - 1) / rows_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (env_flag_enabled("JUNKRAT_W4A16_O_PROJ_UNSPLIT", true)) {
        const int unsplit_row_blocks = (kOProjRows + warps - 1) / warps;
        int launch_blocks = env_int_value("JUNKRAT_W4A16_O_PROJ_UNSPLIT_BLOCKS", 108);
        if (launch_blocks > unsplit_row_blocks) launch_blocks = unsplit_row_blocks;
        if (launch_blocks < 1) launch_blocks = 1;
        const int threads = warps * 32;
        auto in_p = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
        auto pk_p = packed_weight.data_ptr<int>();
        auto sc_p = reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>());
        auto rs_p = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>());
        switch (warps) {
            case 8: o_proj_add_w4a16_group128_unsplit_awq_persistent_kernel<8><<<launch_blocks, threads, 0, stream>>>(in_p, pk_p, sc_p, rs_p); break;
            case 16: o_proj_add_w4a16_group128_unsplit_awq_persistent_kernel<16><<<launch_blocks, threads, 0, stream>>>(in_p, pk_p, sc_p, rs_p); break;
            case 24: o_proj_add_w4a16_group128_unsplit_awq_persistent_kernel<24><<<launch_blocks, threads, 0, stream>>>(in_p, pk_p, sc_p, rs_p); break;
            default: o_proj_add_w4a16_group128_unsplit_awq_persistent_kernel<32><<<launch_blocks, threads, 0, stream>>>(in_p, pk_p, sc_p, rs_p); break;
        }
        return;
    }
    o_proj_add_w4a16_group128_split2_awq_kernel<<<grid, warps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        packed_weight.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()));
}

void lm_head_argmax_w4a16_cuda(
    torch::Tensor hidden,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor output_token,
    int warps_per_block) {
    check_cuda_tensor(hidden, "hidden");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(partial_values, "partial_values");
    check_cuda_tensor(partial_indices, "partial_indices");
    check_cuda_tensor(output_token, "output_token");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(partial_values.scalar_type() == at::kFloat, "partial_values must be FP32");
    TORCH_CHECK(partial_indices.scalar_type() == at::kInt, "partial_indices must be int32");
    TORCH_CHECK(output_token.scalar_type() == at::kLong || output_token.scalar_type() == at::kInt,
                "output_token must be int64 or int32");
    TORCH_CHECK(hidden.numel() >= kLmHeadK, "hidden must have at least 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(1) == kLmHeadPackedK4,
                "packed_weight must be [vocab, 256]");
    const int vocab_size = static_cast<int>(packed_weight.size(0));
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == vocab_size && scale.size(1) == kLmHeadGroups,
                "scale must be [vocab, 16]");
    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 16;
    const int grid = (vocab_size + warps - 1) / warps;
    TORCH_CHECK(partial_values.numel() >= grid, "partial_values too small");
    TORCH_CHECK(partial_indices.numel() >= grid, "partial_indices too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    const __nv_bfloat16* hidden_ptr =
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()) + (hidden.numel() - kLmHeadK);
    if (warps == 4) {
        lm_head_w4a16_stage1_kernel<4><<<grid, 4 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), vocab_size);
    } else if (warps == 8) {
        lm_head_w4a16_stage1_kernel<8><<<grid, 8 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), vocab_size);
    } else if (warps == 16) {
        lm_head_w4a16_stage1_kernel<16><<<grid, 16 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), vocab_size);
    } else {
        lm_head_w4a16_stage1_kernel<32><<<grid, 32 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), vocab_size);
    }
    lm_head_argmax_stage2_kernel<<<1, 256, 0, stream>>>(
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int>(),
        output_token.data_ptr(),
        grid,
        output_token.scalar_type() == at::kLong);
}

void lm_head_argmax_w4a16_persistent_cuda(
    torch::Tensor hidden,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor output_token,
    int warps_per_block,
    int blocks) {
    check_cuda_tensor(hidden, "hidden");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(partial_values, "partial_values");
    check_cuda_tensor(partial_indices, "partial_indices");
    check_cuda_tensor(output_token, "output_token");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(partial_values.scalar_type() == at::kFloat, "partial_values must be FP32");
    TORCH_CHECK(partial_indices.scalar_type() == at::kInt, "partial_indices must be int32");
    TORCH_CHECK(output_token.scalar_type() == at::kLong || output_token.scalar_type() == at::kInt,
                "output_token must be int64 or int32");
    TORCH_CHECK(hidden.numel() >= kLmHeadK, "hidden must have at least 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(1) == kLmHeadPackedK4,
                "packed_weight must be [vocab, 256]");
    const int vocab_size = static_cast<int>(packed_weight.size(0));
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == vocab_size && scale.size(1) == kLmHeadGroups,
                "scale must be [vocab, 16]");
    const int warps = (warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 16;
    const int launch_blocks = blocks > 0 ? blocks : 512;
    TORCH_CHECK(partial_values.numel() >= launch_blocks, "partial_values too small");
    TORCH_CHECK(partial_indices.numel() >= launch_blocks, "partial_indices too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    const __nv_bfloat16* hidden_ptr =
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()) + (hidden.numel() - kLmHeadK);
    if (warps == 8) {
        lm_head_w4a16_stage1_persistent_kernel<8><<<launch_blocks, 8 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), vocab_size);
    } else if (warps == 32) {
        lm_head_w4a16_stage1_persistent_kernel<32><<<launch_blocks, 32 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), vocab_size);
    } else {
        lm_head_w4a16_stage1_persistent_kernel<16><<<launch_blocks, 16 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), vocab_size);
    }
    lm_head_argmax_stage2_kernel<<<1, 256, 0, stream>>>(
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int>(),
        output_token.data_ptr(),
        launch_blocks,
        output_token.scalar_type() == at::kLong);
}

void lm_head_argmax_w4a16_persistent_atomic_cuda(
    torch::Tensor hidden,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor partial_indices,
    torch::Tensor output_token,
    int warps_per_block,
    int blocks) {
    check_cuda_tensor(hidden, "hidden");
    check_cuda_tensor(packed_weight, "packed_weight");
    check_cuda_tensor(scale, "scale");
    check_cuda_tensor(partial_indices, "partial_indices");
    check_cuda_tensor(output_token, "output_token");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(partial_indices.scalar_type() == at::kInt, "partial_indices must be int32");
    TORCH_CHECK(output_token.scalar_type() == at::kLong || output_token.scalar_type() == at::kInt,
                "output_token must be int64 or int32");
    TORCH_CHECK(partial_indices.numel() >= 4, "partial_indices needs at least 4 int32 slots for atomic workspace");
    TORCH_CHECK(hidden.numel() >= kLmHeadK, "hidden must have at least 2048 elements");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(1) == kLmHeadPackedK4,
                "packed_weight must be [vocab, 256]");
    const int vocab_size = static_cast<int>(packed_weight.size(0));
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == vocab_size && scale.size(1) == kLmHeadGroups,
                "scale must be [vocab, 16]");
    const int warps = (warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32)
        ? warps_per_block
        : 32;
    const int launch_blocks = blocks > 0 ? blocks : 128;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(partial_indices.data_ptr<int>(), 0, 4 * sizeof(int), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));
    const __nv_bfloat16* hidden_ptr =
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()) + (hidden.numel() - kLmHeadK);
    auto* best_pair = reinterpret_cast<unsigned long long*>(partial_indices.data_ptr<int>());
    auto* done_counter = reinterpret_cast<unsigned*>(partial_indices.data_ptr<int>() + 2);
    if (warps == 8) {
        lm_head_w4a16_persistent_atomic_kernel<8><<<launch_blocks, 8 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            best_pair, done_counter, output_token.data_ptr(), vocab_size,
            output_token.scalar_type() == at::kLong);
    } else if (warps == 16) {
        lm_head_w4a16_persistent_atomic_kernel<16><<<launch_blocks, 16 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            best_pair, done_counter, output_token.data_ptr(), vocab_size,
            output_token.scalar_type() == at::kLong);
    } else {
        lm_head_w4a16_persistent_atomic_kernel<32><<<launch_blocks, 32 * 32, 0, stream>>>(
            hidden_ptr, packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            best_pair, done_counter, output_token.data_ptr(), vocab_size,
            output_token.scalar_type() == at::kLong);
    }
}
