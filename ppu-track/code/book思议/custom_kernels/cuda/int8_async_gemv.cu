/*
 * cp.async pipelined INT8 GEMV kernels (SM80)
 *
 * Approach: cp.async weight prefetch into shared memory, double-buffered.
 * x stays in registers (loaded via __ldg) — no smem x cache.
 * This avoids __syncthreads per tile and reduces smem pressure.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// PTX cp.async: async copy 16 bytes from global to shared
static __device__ __forceinline__ void cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"((unsigned)__cvta_generic_to_shared(smem_ptr)),
           "l"(gmem_ptr)
    );
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// ---------------------------------------------------------------------------
// INT8 GEMV: double-buffered weight tiles via cp.async, x from __ldg
// Each warp handles one output row. No shared x cache.
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK, int TILE_K>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_int8_async_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    int N, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    // Shared memory: double-buffered weight tiles for this warp only
    __shared__ int8_t w_smem[2][WARPS_PER_BLOCK][TILE_K];

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;
    const int n_tiles = K / TILE_K;

    // Prefetch first weight tile
    for (int i = lane * 16; i < TILE_K; i += 32 * 16)
        cp_async_16b(&w_smem[0][warp_id][i], w_row + i);
    cp_async_commit();

    for (int t = 0; t < n_tiles; t++) {
        // Prefetch next tile while computing current
        if (t + 1 < n_tiles) {
            for (int i = lane * 16; i < TILE_K; i += 32 * 16)
                cp_async_16b(&w_smem[1 - (t & 1)][warp_id][i], w_row + (t + 1) * TILE_K + i);
            cp_async_commit();
        }

        cp_async_wait_all();

        // Load x tile from HBM (no smem, direct __ldg)
        const __half* x_tile = x + t * TILE_K;
        constexpr int N_ITER = TILE_K / 8;
        int cur_buf = t & 1;

        #pragma unroll 4
        for (int i = lane; i < N_ITER; i += 32) {
            const int8_t* wi = &w_smem[cur_buf][warp_id][i * 8];
            int2 wv_raw = *reinterpret_cast<const int2*>(wi);
            const int8_t* w8 = reinterpret_cast<const int8_t*>(&wv_raw);
            float4 xv = __ldg(reinterpret_cast<const float4*>(x_tile) + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            sum += (float)w8[0]*xf0.x + (float)w8[1]*xf0.y
                 + (float)w8[2]*xf1.x + (float)w8[3]*xf1.y
                 + (float)w8[4]*xf2.x + (float)w8[5]*xf2.y
                 + (float)w8[6]*xf3.x + (float)w8[7]*xf3.y;
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = __float2half(sum * row_scale);
}

// ---------------------------------------------------------------------------
// INT8 GEMV + SiLU + Mul: gate and up in two passes
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK, int TILE_K>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_silu_mul_int8_async_kernel(
    const int8_t* __restrict__ W_gate_up,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    int out_dim, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    __shared__ int8_t w_smem[2][WARPS_PER_BLOCK][TILE_K];

    const int8_t* w_gate = W_gate_up + row * K;
    const int8_t* w_up   = W_gate_up + (row + out_dim) * K;
    float gate_scale = scale[row];
    float up_scale   = scale[row + out_dim];

    const int n_tiles = K / TILE_K;
    float gate_sum = 0.0f, up_sum = 0.0f;

    // ---- Gate pass ----
    for (int i = lane * 16; i < TILE_K; i += 32 * 16)
        cp_async_16b(&w_smem[0][warp_id][i], w_gate + i);
    cp_async_commit();

    for (int t = 0; t < n_tiles; t++) {
        if (t + 1 < n_tiles) {
            for (int i = lane * 16; i < TILE_K; i += 32 * 16)
                cp_async_16b(&w_smem[1 - (t & 1)][warp_id][i], w_gate + (t + 1) * TILE_K + i);
            cp_async_commit();
        }
        cp_async_wait_all();

        const __half* x_tile = x + t * TILE_K;
        constexpr int N_ITER = TILE_K / 8;
        int cur_buf = t & 1;

        #pragma unroll 4
        for (int i = lane; i < N_ITER; i += 32) {
            const int8_t* wi = &w_smem[cur_buf][warp_id][i * 8];
            int2 wv = *reinterpret_cast<const int2*>(wi);
            const int8_t* w8 = reinterpret_cast<const int8_t*>(&wv);
            float4 xv = __ldg(reinterpret_cast<const float4*>(x_tile) + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            gate_sum += (float)w8[0]*xf0.x + (float)w8[1]*xf0.y
                      + (float)w8[2]*xf1.x + (float)w8[3]*xf1.y
                      + (float)w8[4]*xf2.x + (float)w8[5]*xf2.y
                      + (float)w8[6]*xf3.x + (float)w8[7]*xf3.y;
        }
    }

    // ---- Up pass ----
    for (int i = lane * 16; i < TILE_K; i += 32 * 16)
        cp_async_16b(&w_smem[0][warp_id][i], w_up + i);
    cp_async_commit();

    for (int t = 0; t < n_tiles; t++) {
        if (t + 1 < n_tiles) {
            for (int i = lane * 16; i < TILE_K; i += 32 * 16)
                cp_async_16b(&w_smem[1 - (t & 1)][warp_id][i], w_up + (t + 1) * TILE_K + i);
            cp_async_commit();
        }
        cp_async_wait_all();

        const __half* x_tile = x + t * TILE_K;
        constexpr int N_ITER = TILE_K / 8;
        int cur_buf = t & 1;

        #pragma unroll 4
        for (int i = lane; i < N_ITER; i += 32) {
            const int8_t* wi = &w_smem[cur_buf][warp_id][i * 8];
            int2 wv = *reinterpret_cast<const int2*>(wi);
            const int8_t* w8 = reinterpret_cast<const int8_t*>(&wv);
            float4 xv = __ldg(reinterpret_cast<const float4*>(x_tile) + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            up_sum += (float)w8[0]*xf0.x + (float)w8[1]*xf0.y
                    + (float)w8[2]*xf1.x + (float)w8[3]*xf1.y
                    + (float)w8[4]*xf2.x + (float)w8[5]*xf2.y
                    + (float)w8[6]*xf3.x + (float)w8[7]*xf3.y;
        }
    }

    gate_sum = warp_reduce_sum(gate_sum);
    up_sum = warp_reduce_sum(up_sum);
    if (lane == 0) {
        float silu_val = gate_sum * gate_scale / (1.0f + expf(-(gate_sum * gate_scale)));
        y[row] = __float2half(silu_val * up_sum * up_scale);
    }
}

// ---------------------------------------------------------------------------
// INT8 GEMV + residual add + extra addend
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK, int TILE_K>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_add_both_int8_async_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    const __half* __restrict__ addend,
    int N, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    __shared__ int8_t w_smem[2][WARPS_PER_BLOCK][TILE_K];

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;
    const int n_tiles = K / TILE_K;

    for (int i = lane * 16; i < TILE_K; i += 32 * 16)
        cp_async_16b(&w_smem[0][warp_id][i], w_row + i);
    cp_async_commit();

    for (int t = 0; t < n_tiles; t++) {
        if (t + 1 < n_tiles) {
            for (int i = lane * 16; i < TILE_K; i += 32 * 16)
                cp_async_16b(&w_smem[1 - (t & 1)][warp_id][i], w_row + (t + 1) * TILE_K + i);
            cp_async_commit();
        }
        cp_async_wait_all();

        const __half* x_tile = x + t * TILE_K;
        constexpr int N_ITER = TILE_K / 8;
        int cur_buf = t & 1;

        #pragma unroll 4
        for (int i = lane; i < N_ITER; i += 32) {
            const int8_t* wi = &w_smem[cur_buf][warp_id][i * 8];
            int2 wv_raw = *reinterpret_cast<const int2*>(wi);
            const int8_t* w8 = reinterpret_cast<const int8_t*>(&wv_raw);
            float4 xv = __ldg(reinterpret_cast<const float4*>(x_tile) + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            sum += (float)w8[0]*xf0.x + (float)w8[1]*xf0.y
                 + (float)w8[2]*xf1.x + (float)w8[3]*xf1.y
                 + (float)w8[4]*xf2.x + (float)w8[5]*xf2.y
                 + (float)w8[6]*xf3.x + (float)w8[7]*xf3.y;
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        float old = __half2float(y[row]);
        float extra = __half2float(addend[row]);
        y[row] = __float2half(old + extra + sum * row_scale);
    }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

torch::Tensor gemv_int8_async_cuda(
    torch::Tensor W_int8, torch::Tensor x, torch::Tensor scale)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);
    auto y = torch::empty({N}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    constexpr int WPB = 8;
    constexpr int TILE_K = 256;
    int grid = (N + WPB - 1) / WPB;

    gemv_int8_async_kernel<WPB, TILE_K><<<grid, WPB * 32, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K);
    return y;
}

torch::Tensor gemv_silu_mul_int8_async_cuda(
    torch::Tensor W_gate_up_int8, torch::Tensor x, torch::Tensor scale)
{
    int total_rows = W_gate_up_int8.size(0);
    int out_dim = total_rows / 2;
    int K = W_gate_up_int8.size(1);
    auto y = torch::empty({out_dim}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    constexpr int WPB = 8;
    constexpr int TILE_K = 256;
    int grid = (out_dim + WPB - 1) / WPB;

    gemv_silu_mul_int8_async_kernel<WPB, TILE_K><<<grid, WPB * 32, 0, stream>>>(
        W_gate_up_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, K);
    return y;
}

void gemv_add_both_int8_async_cuda(
    torch::Tensor W_int8, torch::Tensor x, torch::Tensor scale,
    torch::Tensor y, torch::Tensor addend)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    constexpr int WPB = 8;
    constexpr int TILE_K = 256;
    int grid = (N + WPB - 1) / WPB;

    gemv_add_both_int8_async_kernel<WPB, TILE_K><<<grid, WPB * 32, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
        N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_int8_async", &gemv_int8_async_cuda,
          "INT8 GEMV with cp.async weight pipeline (SM80)");
    m.def("gemv_silu_mul_int8_async", &gemv_silu_mul_int8_async_cuda,
          "INT8 GEMV+SiLU+Mul with cp.async weight pipeline (SM80)");
    m.def("gemv_add_both_int8_async", &gemv_add_both_int8_async_cuda,
          "INT8 GEMV+residual+addend with cp.async weight pipeline (SM80)");
}
