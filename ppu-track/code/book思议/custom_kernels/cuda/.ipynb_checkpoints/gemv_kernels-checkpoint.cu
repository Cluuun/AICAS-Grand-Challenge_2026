/*
 * 合并访存的GEMV核心 给A800(sm_80) decode用的
 *
 * 为啥要自己写GEMV: 朴素实现1个线程算1行 warp里线程读不同行 访存完全对不齐
 *
 * 解决办法: 一个warp算一行
 *   - warp里32个线程读同一行的连续元素 访存完美合并
 *   - 用warp shuffle做规约 不用shared memory
 *   - 每个block 8个warp 更多block 更好填满SM
 *
 * Qwen3-VL-2B decode的矩阵尺寸:
 *   qkv:     [2048]->[3072]   gate_up: [2048]->[6144]
 *   o_proj:  [2048]->[2048]   down:    [3072]->[2048]
 *
 * SM80 专用优化 (cp.async 软件流水线):
 *   - cp.async: Ampere 新增的异步 global→shared 拷贝指令
 *   - 在计算当前 tile 的同时，异步预取下一个 tile 到 shared memory
 *   - 隐藏 HBM 延迟，提升带宽利用率
 *   - 适用于 in_dim 较大的 GEMV（qkv/gate_up/down）
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// warp shuffle规约 sum
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// 1. 合并访存GEMV: y = W @ x
//    一个warp算一行 线程读连续元素 → 访存完美合并
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_coalesced_kernel(
    const __half* __restrict__ W,    // 权重 [out_dim, in_dim]
    const __half* __restrict__ x,    // 输入向量 [in_dim]
    __half* __restrict__ y,          // 输出 [out_dim]
    int out_dim,
    int in_dim)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const float4* w_row4 = reinterpret_cast<const float4*>(W + row * in_dim);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_f4 = in_dim / 8;  // 2048->256 (每个float4 = 8个FP16)

    float sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_f4; i += 32) {
        float4 wv = __ldg(w_row4 + i);
        float4 xv = __ldg(x4 + i);
        __half2* wp = reinterpret_cast<__half2*>(&wv);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 wf0 = __half22float2(wp[0]); float2 xf0 = __half22float2(xp[0]);
        float2 wf1 = __half22float2(wp[1]); float2 xf1 = __half22float2(xp[1]);
        float2 wf2 = __half22float2(wp[2]); float2 xf2 = __half22float2(xp[2]);
        float2 wf3 = __half22float2(wp[3]); float2 xf3 = __half22float2(xp[3]);
        sum += wf0.x * xf0.x + wf0.y * xf0.y + wf1.x * xf1.x + wf1.y * xf1.y
             + wf2.x * xf2.x + wf2.y * xf2.y + wf3.x * xf3.x + wf3.y * xf3.y;
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// ---------------------------------------------------------------------------
// 1b. cp.async 软件流水线 GEMV (SM80 专用)
//
// 原理：
//   当前 kernel 用 __ldg 同步加载权重，每次 load 都 stall 等待数据
//   cp.async 可以在计算当前 tile 时异步预取下一个 tile 到 shared memory
//   实现 compute-load overlap，隐藏 HBM 延迟
//
// 布局：
//   每个 block 处理 WARPS_PER_BLOCK 行输出
//   shared memory 双缓冲：buf[2][WARPS_PER_BLOCK][TILE_K]
//   TILE_K = 256 (每个 warp 处理 256/32=8 个 float4 = 64 个 FP16)
//   每个 tile 大小 = WARPS_PER_BLOCK × TILE_K × 2B = 8 × 256 × 2 = 4KB
//   双缓冲 = 8KB smem，远小于 A800 的 96KB/SM
//
// 流水线：
//   iter 0: prefetch tile[0] → buf[0]
//   iter 1: compute tile[0] from buf[0], prefetch tile[1] → buf[1]
//   iter 2: compute tile[1] from buf[1], prefetch tile[2] → buf[0]
//   ...
// ---------------------------------------------------------------------------

#include <cuda/pipeline>

template <int WARPS_PER_BLOCK, int TILE_K>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_async_kernel(
    const __half* __restrict__ W,    // [out_dim, in_dim]
    const __half* __restrict__ x,    // [in_dim]
    __half* __restrict__ y,          // [out_dim]
    int out_dim,
    int in_dim)
{
    // 双缓冲 shared memory: [2 buffers][WARPS rows][TILE_K elements]
    __shared__ __half smem[2][WARPS_PER_BLOCK][TILE_K];

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= out_dim) return;

    const int n_tiles = in_dim / TILE_K;  // 2048/256=8, 3072/256=12
    const __half* w_row = W + row * in_dim;

    // cp.async pipeline (Ampere 专用)
    auto pipe = cuda::make_pipeline();

    float sum = 0.0f;
    int cur_buf = 0;

    // 预取第一个 tile
    pipe.producer_acquire();
    // 每个 warp 负责自己那行的 TILE_K 个元素
    // 每个线程负责 TILE_K/32 个 __half2 (= TILE_K/32 * 2 个 FP16)
    // 用 cp.async 异步拷贝 16B (= 8 FP16) per thread
    constexpr int ELEMS_PER_THREAD = TILE_K / 32;  // 8
    constexpr int BYTES_PER_THREAD = ELEMS_PER_THREAD * sizeof(__half);  // 16B

    // 预取 tile 0
    cuda::memcpy_async(
        smem[cur_buf][warp_id] + lane * ELEMS_PER_THREAD,
        w_row + lane * ELEMS_PER_THREAD,
        cuda::aligned_size_t<16>(BYTES_PER_THREAD),
        pipe
    );
    pipe.producer_commit();

    // 主循环
    for (int t = 0; t < n_tiles; t++) {
        int next_buf = 1 - cur_buf;

        // 预取下一个 tile（如果还有）
        if (t + 1 < n_tiles) {
            pipe.producer_acquire();
            cuda::memcpy_async(
                smem[next_buf][warp_id] + lane * ELEMS_PER_THREAD,
                w_row + (t + 1) * TILE_K + lane * ELEMS_PER_THREAD,
                cuda::aligned_size_t<16>(BYTES_PER_THREAD),
                pipe
            );
            pipe.producer_commit();
        }

        // 等待当前 tile 加载完成
        pipe.consumer_wait();

        // 计算当前 tile
        const float4* w4 = reinterpret_cast<const float4*>(smem[cur_buf][warp_id]);
        const float4* x4 = reinterpret_cast<const float4*>(x + t * TILE_K);
        constexpr int N_F4 = TILE_K / 8;  // 32

        #pragma unroll
        for (int i = lane; i < N_F4; i += 32) {
            float4 wv = w4[i];
            float4 xv = __ldg(x4 + i);
            __half2* wp = reinterpret_cast<__half2*>(&wv);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 wf0 = __half22float2(wp[0]); float2 xf0 = __half22float2(xp[0]);
            float2 wf1 = __half22float2(wp[1]); float2 xf1 = __half22float2(xp[1]);
            float2 wf2 = __half22float2(wp[2]); float2 xf2 = __half22float2(xp[2]);
            float2 wf3 = __half22float2(wp[3]); float2 xf3 = __half22float2(xp[3]);
            sum += wf0.x*xf0.x + wf0.y*xf0.y + wf1.x*xf1.x + wf1.y*xf1.y
                 + wf2.x*xf2.x + wf2.y*xf2.y + wf3.x*xf3.x + wf3.y*xf3.y;
        }

        pipe.consumer_release();
        cur_buf = next_buf;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = __float2half(sum);
}

torch::Tensor gemv_async_cuda(
    torch::Tensor W,
    torch::Tensor x)
{
    int out_dim = W.size(0);
    int in_dim  = W.size(1);
    auto y = torch::empty({out_dim}, W.options());

    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int TILE_K = 256;  // 每个 tile 256 个 FP16 = 512B per warp

    TORCH_CHECK(in_dim % TILE_K == 0,
        "gemv_async: in_dim must be divisible by TILE_K=256, got ", in_dim);

    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int block = WARPS_PER_BLOCK * 32;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_async_kernel<WARPS_PER_BLOCK, TILE_K><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, in_dim);

    return y;
}

// ---------------------------------------------------------------------------
// 1c. lm_head 专用 GEMV: 更大的 WARPS_PER_BLOCK 提升 HBM 带宽利用率
//     lm_head [151936, 2048] = 594MB，需要最大化 HBM 带宽
//     用 __ldcg (L2 bypass) 避免污染 L2 cache，让其他 kernel 的数据留在 L2
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_kernel(
    const __half* __restrict__ W,    // 权重 [out_dim, in_dim]
    const __half* __restrict__ x,    // 输入向量 [in_dim]
    __half* __restrict__ y,          // 输出 [out_dim]
    int out_dim,
    int in_dim)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const float4* w_row4 = reinterpret_cast<const float4*>(W + row * in_dim);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_f4 = in_dim / 8;

    float sum = 0.0f;

    // 用 __ldcg 绕过 L2 cache，直接从 HBM 读权重
    // 避免 lm_head 的大权重污染 L2，让其他 kernel 的数据保留在 L2
    #pragma unroll 4
    for (int i = lane; i < n_f4; i += 32) {
        float4 wv, xv;
        // __ldcg: cache at global level (L2 bypass for weight, keeps L1 clean)
        asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(reinterpret_cast<unsigned*>(&wv)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&wv)[1]),
                       "=r"(reinterpret_cast<unsigned*>(&wv)[2]),
                       "=r"(reinterpret_cast<unsigned*>(&wv)[3])
                     : "l"(w_row4 + i));
        xv = __ldg(x4 + i);  // x 很小(4KB)，用 L1 cache
        __half2* wp = reinterpret_cast<__half2*>(&wv);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 wf0 = __half22float2(wp[0]); float2 xf0 = __half22float2(xp[0]);
        float2 wf1 = __half22float2(wp[1]); float2 xf1 = __half22float2(xp[1]);
        float2 wf2 = __half22float2(wp[2]); float2 xf2 = __half22float2(xp[2]);
        float2 wf3 = __half22float2(wp[3]); float2 xf3 = __half22float2(xp[3]);
        sum += wf0.x * xf0.x + wf0.y * xf0.y + wf1.x * xf1.x + wf1.y * xf1.y
             + wf2.x * xf2.x + wf2.y * xf2.y + wf3.x * xf3.x + wf3.y * xf3.y;
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// ---------------------------------------------------------------------------
// 2. 融合 GEMV + SiLU + Mul
//    y[i] = silu(W_gate[i] @ x) * (W_up[i] @ x)
//    W布局: [gate | up] = [2*out_dim, in_dim] 前半gate后半up
//    每个warp同时算自己那行的gate和up 两边都能合并访存
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_silu_mul_v2_kernel(
    const __half* __restrict__ W_gate_up,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int out_dim,
    int in_dim)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    int n_f4 = in_dim / 8;  // 2048->256 (每个float4 = 8个FP16)

    const float4* w_gate4 = reinterpret_cast<const float4*>(W_gate_up + row * in_dim);
    const float4* w_up4 = reinterpret_cast<const float4*>(W_gate_up + (row + out_dim) * in_dim);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_f4; i += 32) {
        float4 xv = __ldg(x4 + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        float4 gv = __ldg(w_gate4 + i);
        __half2* gp = reinterpret_cast<__half2*>(&gv);
        float2 gf0 = __half22float2(gp[0]); float2 gf1 = __half22float2(gp[1]);
        float2 gf2 = __half22float2(gp[2]); float2 gf3 = __half22float2(gp[3]);
        gate_sum += gf0.x * xf0.x + gf0.y * xf0.y + gf1.x * xf1.x + gf1.y * xf1.y
                  + gf2.x * xf2.x + gf2.y * xf2.y + gf3.x * xf3.x + gf3.y * xf3.y;

        float4 uv = __ldg(w_up4 + i);
        __half2* up = reinterpret_cast<__half2*>(&uv);
        float2 uf0 = __half22float2(up[0]); float2 uf1 = __half22float2(up[1]);
        float2 uf2 = __half22float2(up[2]); float2 uf3 = __half22float2(up[3]);
        up_sum += uf0.x * xf0.x + uf0.y * xf0.y + uf1.x * xf1.x + uf1.y * xf1.y
                + uf2.x * xf2.x + uf2.y * xf2.y + uf3.x * xf3.x + uf3.y * xf3.y;
    }

    gate_sum = warp_reduce_sum(gate_sum);
    up_sum = warp_reduce_sum(up_sum);

    if (lane == 0) {
        float silu_val = gate_sum / (1.0f + expf(-gate_sum));
        y[row] = __float2half(silu_val * up_sum);
    }
}

// ---------------------------------------------------------------------------
// 3. GEMV + Add: residual += W @ x  融合残差加 省一次kernel launch
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_add_kernel(
    const __half* __restrict__ W,    // 权重 [out_dim, in_dim]
    const __half* __restrict__ x,    // 输入向量 [in_dim]
    __half* __restrict__ y,          // 残差buffer [out_dim] (in-place add)
    int out_dim,
    int in_dim)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const float4* w_row4 = reinterpret_cast<const float4*>(W + row * in_dim);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_f4 = in_dim / 8;  // 2048->256 (每个float4 = 8个FP16)

    float sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_f4; i += 32) {
        float4 wv = __ldg(w_row4 + i);
        float4 xv = __ldg(x4 + i);
        __half2* wp = reinterpret_cast<__half2*>(&wv);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 wf0 = __half22float2(wp[0]); float2 xf0 = __half22float2(xp[0]);
        float2 wf1 = __half22float2(wp[1]); float2 xf1 = __half22float2(xp[1]);
        float2 wf2 = __half22float2(wp[2]); float2 xf2 = __half22float2(xp[2]);
        float2 wf3 = __half22float2(wp[3]); float2 xf3 = __half22float2(xp[3]);
        sum += wf0.x * xf0.x + wf0.y * xf0.y + wf1.x * xf1.x + wf1.y * xf1.y
             + wf2.x * xf2.x + wf2.y * xf2.y + wf3.x * xf3.x + wf3.y * xf3.y;
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        float old = __half2float(y[row]);
        y[row] = __float2half(old + sum);
    }
}

// ---------------------------------------------------------------------------
// 4. 融合QKV拆分 + Q/K RMS norm + RoPE + V copy (decode专用)
//    替代Triton kernel 去掉Python dispatch overhead
//    一个warp处理一个head: 32线程 * 4元素/线程 = 128 = HEAD_DIM
//    RoPE用shuffle交换first/second half的paired元素
// ---------------------------------------------------------------------------

template <int HEAD_DIM, int HALF_DIM, int N_Q_HEADS, int N_KV_HEADS, int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
fused_qkv_norm_rope_cuda_kernel(
    const __half* __restrict__ qkv,      // [q_dim + 2*kv_dim] GEMV输出
    const __half* __restrict__ q_norm_w,  // [head_dim]
    const __half* __restrict__ k_norm_w,  // [head_dim]
    const __half* __restrict__ cos_data,  // [head_dim] fp16 (直接接受, kernel内部转fp32)
    const __half* __restrict__ sin_data,  // [head_dim] fp16
    __half* __restrict__ Q_out,           // [n_q_heads, head_dim]
    __half* __restrict__ K_out,           // [n_kv_heads, head_dim]
    __half* __restrict__ V_out,           // [n_kv_heads, head_dim]
    int q_dim,
    int kv_dim,
    float q_eps,
    float k_eps)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int head_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // 每个线程处理HEAD_DIM/32=4个元素
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;

    // ===== 处理Q头 =====
    if (head_idx < N_Q_HEADS) {
        const __half* q_head = qkv + head_idx * HEAD_DIM;
        __half* q_out = Q_out + head_idx * HEAD_DIM;

        float my_vals[ELEMS_PER_THREAD];
        float my_norm_w[ELEMS_PER_THREAD];
        float sum_sq = 0.0f;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = lane * ELEMS_PER_THREAD + i;
            float v = __half2float(__ldg(q_head + idx));
            float w = __half2float(__ldg(q_norm_w + idx));
            my_vals[i] = v;
            my_norm_w[i] = w;
            sum_sq += v * v;
        }

        // Warp reduce + broadcast (所有lane都需要sum_sq)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 16);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 8);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 4);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 2);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 1);
        sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);  // broadcast from lane 0

        float inv_rsqrt = rsqrtf(sum_sq / HEAD_DIM + q_eps);

        // Normalize
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            my_vals[i] = my_vals[i] * inv_rsqrt * my_norm_w[i];
        }

        // RoPE: 先交换paired值存到临时数组 然后再算结果
        // 这样避免in-place修改导致second half读到已修改的值
        // lane i (first half, i<16) 的 paired 在 lane i+16
        // lane i (second half, i>=16) 的 paired 在 lane i-16
        float my_paired[ELEMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int peer_lane = (lane < 16) ? (lane + 16) : (lane - 16);
            my_paired[i] = __shfl_sync(0xffffffff, my_vals[i], peer_lane);
        }

        // 现在用原始my_vals和paired值算RoPE
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = lane * ELEMS_PER_THREAD + i;
            float val = my_vals[i];

            if (idx < HALF_DIM) {
                float c = __half2float(__ldg(cos_data + idx));
                float s = __half2float(__ldg(sin_data + idx));
                my_vals[i] = val * c + (-my_paired[i]) * s;
            } else {
                float c = __half2float(__ldg(cos_data + idx));
                float s = __half2float(__ldg(sin_data + idx));
                my_vals[i] = val * c + my_paired[i] * s;
            }
        }

        // Write output
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = lane * ELEMS_PER_THREAD + i;
            q_out[idx] = __float2half(my_vals[i]);
        }
    }

    // ===== 处理K头 + 拷贝V =====
    if (head_idx < N_KV_HEADS) {
        const __half* k_head = qkv + q_dim + head_idx * HEAD_DIM;
        __half* k_out = K_out + head_idx * HEAD_DIM;

        float my_vals[ELEMS_PER_THREAD];
        float my_norm_w[ELEMS_PER_THREAD];
        float sum_sq = 0.0f;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = lane * ELEMS_PER_THREAD + i;
            float v = __half2float(__ldg(k_head + idx));
            float w = __half2float(__ldg(k_norm_w + idx));
            my_vals[i] = v;
            my_norm_w[i] = w;
            sum_sq += v * v;
        }

        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 16);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 8);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 4);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 2);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, 1);
        sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

        float inv_rsqrt = rsqrtf(sum_sq / HEAD_DIM + k_eps);

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            my_vals[i] = my_vals[i] * inv_rsqrt * my_norm_w[i];
        }

        // RoPE for K: 先交换 再计算
        float k_paired[ELEMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int peer_lane = (lane < 16) ? (lane + 16) : (lane - 16);
            k_paired[i] = __shfl_sync(0xffffffff, my_vals[i], peer_lane);
        }

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = lane * ELEMS_PER_THREAD + i;
            float val = my_vals[i];

            if (idx < HALF_DIM) {
                float c = __half2float(cos_data[idx]);
                float s = __half2float(sin_data[idx]);
                my_vals[i] = val * c + (-k_paired[i]) * s;
            } else {
                float c = __half2float(cos_data[idx]);
                float s = __half2float(sin_data[idx]);
                my_vals[i] = val * c + k_paired[i] * s;
            }
        }

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = lane * ELEMS_PER_THREAD + i;
            k_out[idx] = __float2half(my_vals[i]);
        }

        // Copy V (no norm, no RoPE)
        const __half* v_head = qkv + q_dim + kv_dim + head_idx * HEAD_DIM;
        __half* v_out = V_out + head_idx * HEAD_DIM;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = lane * ELEMS_PER_THREAD + i;
            v_out[idx] = __ldg(v_head + idx);
        }
    }
}


// ===========================================================================
// Batch INT4 GEMV: 一次读取权重，处理 BATCH 个输入向量
// 投机解码 batch verify 用 — 权重读取在 batch 元素间共享
// 内存流量 ≈ 单 token decode (权重只读一次)
// ===========================================================================

template <int WARPS_PER_BLOCK, int BATCH>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_int4_batch_kernel(
    const uint8_t* __restrict__ W_packed,  // [N, K/2] uint8
    const __half* __restrict__ x_batch,     // [BATCH, K] fp16
    const float* __restrict__ scale,        // [N, num_groups] fp32
    __half* __restrict__ y_batch,           // [BATCH, N] fp16 (row-major: y[b*N+row])
    int N, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);

    float accum[BATCH];
    #pragma unroll
    for (int b = 0; b < BATCH; b++) accum[b] = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        // 权重只读一次，所有 batch 元素共享
        uint32_t packed = __ldg(w_row + i);
        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);

        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const float4* x4 = reinterpret_cast<const float4*>(x_batch + b * K);
            float4 xv = __ldg(x4 + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

            float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;
            float raw = (float)((packed      ) & 0xF) * xf0.x
                      + (float)((packed >>  4) & 0xF) * xf0.y
                      + (float)((packed >>  8) & 0xF) * xf1.x
                      + (float)((packed >> 12) & 0xF) * xf1.y
                      + (float)((packed >> 16) & 0xF) * xf2.x
                      + (float)((packed >> 20) & 0xF) * xf2.y
                      + (float)((packed >> 24) & 0xF) * xf3.x
                      + (float)( packed >> 28        ) * xf3.y;

            accum[b] += (raw - 8.0f * is) * gs;
        }
    }

    #pragma unroll
    for (int b = 0; b < BATCH; b++) {
        accum[b] = warp_reduce_sum(accum[b]);
        if (lane == 0) y_batch[b * N + row] = __float2half(accum[b]);
    }
}

// Batch INT4 SiLU+Mul: gate_up 融合，权重只读一次
template <int WARPS_PER_BLOCK, int BATCH>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_silu_mul_int4_batch_kernel(
    const uint8_t* __restrict__ W_packed,  // [2*out_dim, K/2]
    const __half* __restrict__ x_batch,     // [BATCH, K]
    const float* __restrict__ scale,        // [2*out_dim, num_groups]
    __half* __restrict__ y_batch,           // [BATCH, out_dim]
    int out_dim, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const uint32_t* w_gate = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    const uint32_t* w_up   = reinterpret_cast<const uint32_t*>(W_packed + (row + out_dim) * (K / 2));
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);

    float gate_acc[BATCH], up_acc[BATCH];
    #pragma unroll
    for (int b = 0; b < BATCH; b++) { gate_acc[b] = 0.0f; up_acc[b] = 0.0f; }

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        // 两个权重行各读一次
        uint32_t pg = __ldg(w_gate + i);
        uint32_t pu = __ldg(w_up + i);
        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);
        float us = __ldg(scale + (row + out_dim) * num_groups + g);

        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const float4* x4 = reinterpret_cast<const float4*>(x_batch + b * K);
            float4 xv = __ldg(x4 + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

            float gate_raw = (float)((pg      ) & 0xF) * xf0.x
                           + (float)((pg >>  4) & 0xF) * xf0.y
                           + (float)((pg >>  8) & 0xF) * xf1.x
                           + (float)((pg >> 12) & 0xF) * xf1.y
                           + (float)((pg >> 16) & 0xF) * xf2.x
                           + (float)((pg >> 20) & 0xF) * xf2.y
                           + (float)((pg >> 24) & 0xF) * xf3.x
                           + (float)( pg >> 28        ) * xf3.y;

            float up_raw = (float)((pu      ) & 0xF) * xf0.x
                         + (float)((pu >>  4) & 0xF) * xf0.y
                         + (float)((pu >>  8) & 0xF) * xf1.x
                         + (float)((pu >> 12) & 0xF) * xf1.y
                         + (float)((pu >> 16) & 0xF) * xf2.x
                         + (float)((pu >> 20) & 0xF) * xf2.y
                         + (float)((pu >> 24) & 0xF) * xf3.x
                         + (float)( pu >> 28        ) * xf3.y;

            gate_acc[b] += (gate_raw - 8.0f * is) * gs;
            up_acc[b] += (up_raw - 8.0f * is) * us;
        }
    }

    #pragma unroll
    for (int b = 0; b < BATCH; b++) {
        gate_acc[b] = warp_reduce_sum(gate_acc[b]);
        up_acc[b] = warp_reduce_sum(up_acc[b]);
        if (lane == 0) {
            float g_val = gate_acc[b];
            float u_val = up_acc[b];
            y_batch[b * out_dim + row] = __float2half(
                (g_val / (1.0f + expf(-g_val))) * u_val);
        }
    }
}

// Batch INT4 lm_head: L2 bypass + 大 block
template <int WARPS_PER_BLOCK, int BATCH>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_int4_batch_kernel(
    const uint8_t* __restrict__ W_packed,
    const __half* __restrict__ x_batch,
    const float* __restrict__ scale,
    __half* __restrict__ y_batch,
    int N, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);

    float accum[BATCH];
    #pragma unroll
    for (int b = 0; b < BATCH; b++) accum[b] = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        uint32_t packed;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(w_row + i));
        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);

        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const float4* x4 = reinterpret_cast<const float4*>(x_batch + b * K);
            float4 xv = __ldg(x4 + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;
            float raw = (float)((packed      ) & 0xF) * xf0.x
                      + (float)((packed >>  4) & 0xF) * xf0.y
                      + (float)((packed >>  8) & 0xF) * xf1.x
                      + (float)((packed >> 12) & 0xF) * xf1.y
                      + (float)((packed >> 16) & 0xF) * xf2.x
                      + (float)((packed >> 20) & 0xF) * xf2.y
                      + (float)((packed >> 24) & 0xF) * xf3.x
                      + (float)( packed >> 28        ) * xf3.y;
            accum[b] += (raw - 8.0f * is) * gs;
        }
    }

    #pragma unroll
    for (int b = 0; b < BATCH; b++) {
        accum[b] = warp_reduce_sum(accum[b]);
        if (lane == 0) y_batch[b * N + row] = __float2half(accum[b]);
    }
}

// Python 接口: batch INT4 GEMV
torch::Tensor gemv_int4_batch_cuda(
    torch::Tensor W_packed,
    torch::Tensor x_batch,
    torch::Tensor scale,
    int num_groups,
    int batch_size)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({batch_size, N}, x_batch.options());

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto wp = W_packed.data_ptr<uint8_t>();
    auto xp = reinterpret_cast<const __half*>(x_batch.data_ptr<at::Half>());
    auto sp = scale.data_ptr<float>();
    auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

    if (batch_size == 3) {
        gemv_int4_batch_kernel<WARPS_PER_BLOCK, 3><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else if (batch_size == 5) {
        gemv_int4_batch_kernel<WARPS_PER_BLOCK, 5><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else if (batch_size == 4) {
        gemv_int4_batch_kernel<WARPS_PER_BLOCK, 4><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else if (batch_size == 7) {
        gemv_int4_batch_kernel<WARPS_PER_BLOCK, 7><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else {
        TORCH_CHECK(false, "gemv_int4_batch: unsupported batch_size ", batch_size,
                    " (supported: 3,4,5,7)");
    }
    return y;
}

// Python 接口: batch INT4 SiLU+Mul
torch::Tensor gemv_silu_mul_int4_batch_cuda(
    torch::Tensor W_packed,
    torch::Tensor x_batch,
    torch::Tensor scale,
    int num_groups,
    int batch_size)
{
    int out_dim = W_packed.size(0) / 2;
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({batch_size, out_dim}, x_batch.options());

    constexpr int WARPS_PER_BLOCK = 8;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto wp = W_packed.data_ptr<uint8_t>();
    auto xp = reinterpret_cast<const __half*>(x_batch.data_ptr<at::Half>());
    auto sp = scale.data_ptr<float>();
    auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

    if (batch_size == 3) {
        gemv_silu_mul_int4_batch_kernel<WARPS_PER_BLOCK, 3><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, out_dim, K, num_groups);
    } else if (batch_size == 5) {
        gemv_silu_mul_int4_batch_kernel<WARPS_PER_BLOCK, 5><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, out_dim, K, num_groups);
    } else if (batch_size == 4) {
        gemv_silu_mul_int4_batch_kernel<WARPS_PER_BLOCK, 4><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, out_dim, K, num_groups);
    } else if (batch_size == 7) {
        gemv_silu_mul_int4_batch_kernel<WARPS_PER_BLOCK, 7><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, out_dim, K, num_groups);
    } else {
        TORCH_CHECK(false, "gemv_silu_mul_int4_batch: unsupported batch_size ", batch_size);
    }
    return y;
}

// Python 接口: batch INT4 lm_head
torch::Tensor gemv_lmhead_int4_batch_cuda(
    torch::Tensor W_packed,
    torch::Tensor x_batch,
    torch::Tensor scale,
    int num_groups,
    int batch_size)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({batch_size, N}, x_batch.options());

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto wp = W_packed.data_ptr<uint8_t>();
    auto xp = reinterpret_cast<const __half*>(x_batch.data_ptr<at::Half>());
    auto sp = scale.data_ptr<float>();
    auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

    if (batch_size == 3) {
        gemv_lmhead_int4_batch_kernel<WARPS_PER_BLOCK, 3><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else if (batch_size == 5) {
        gemv_lmhead_int4_batch_kernel<WARPS_PER_BLOCK, 5><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else if (batch_size == 4) {
        gemv_lmhead_int4_batch_kernel<WARPS_PER_BLOCK, 4><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else if (batch_size == 7) {
        gemv_lmhead_int4_batch_kernel<WARPS_PER_BLOCK, 7><<<grid, block_size, 0, stream>>>(
            wp, xp, sp, yp, N, K, num_groups);
    } else {
        TORCH_CHECK(false, "gemv_lmhead_int4_batch: unsupported batch_size ", batch_size);
    }
    return y;
}


// ===========================================================================
// Python接口
// ===========================================================================

torch::Tensor gemv_coalesced_cuda(
    torch::Tensor W,     // 权重 [out_dim, in_dim]
    torch::Tensor x)     // 输入 [in_dim]
{
    int out_dim = W.size(0);
    int in_dim = W.size(1);
    auto y = torch::empty({out_dim}, W.options());

    constexpr int WARPS_PER_BLOCK = 8;
    int block_size = WARPS_PER_BLOCK * 32;  // 256个线程
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_coalesced_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, in_dim);

    return y;
}

// lm_head 专用 GEMV：更大的 WARPS_PER_BLOCK + L2 bypass
torch::Tensor gemv_lmhead_cuda(
    torch::Tensor W,     // 权重 [out_dim, in_dim]
    torch::Tensor x)     // 输入 [in_dim]
{
    int out_dim = W.size(0);
    int in_dim = W.size(1);
    auto y = torch::empty({out_dim}, W.options());

    // 用更大的 WARPS_PER_BLOCK=16 (512 threads/block)
    // 减少 block 数量，提高每个 SM 的 warp 数量，更好地 hide memory latency
    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;  // 512个线程
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_lmhead_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, in_dim);

    return y;
}

torch::Tensor gemv_silu_mul_coalesced_cuda(
    torch::Tensor W_gate_up,  // gate+up纵向拼接 [2*out_dim, in_dim]
    torch::Tensor x)          // 输入 [in_dim]
{
    int total_rows = W_gate_up.size(0);
    int out_dim = total_rows / 2;
    int in_dim = W_gate_up.size(1);
    auto y = torch::empty({out_dim}, W_gate_up.options());

    constexpr int WARPS_PER_BLOCK = 8;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_silu_mul_v2_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W_gate_up.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, in_dim);

    return y;
}


void gemv_add_cuda(
    torch::Tensor W,     // 权重 [out_dim, in_dim]
    torch::Tensor x,     // 输入 [in_dim]
    torch::Tensor y)     // 残差 [out_dim] (in-place add)
{
    int out_dim = W.size(0);
    int in_dim = W.size(1);

    constexpr int WARPS_PER_BLOCK = 8;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_add_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, in_dim);
}


void fused_qkv_norm_rope_cuda(
    torch::Tensor qkv,          // [q_dim + 2*kv_dim] GEMV结果
    torch::Tensor q_norm_w,     // [head_dim]
    torch::Tensor k_norm_w,     // [head_dim]
    torch::Tensor cos_data,     // [head_dim] fp16
    torch::Tensor sin_data,     // [head_dim] fp16
    torch::Tensor Q_out,        // [n_q_heads, head_dim]
    torch::Tensor K_out,        // [n_kv_heads, head_dim]
    torch::Tensor V_out,        // [n_kv_heads, head_dim]
    int q_dim, int kv_dim,
    float q_eps, float k_eps)
{
    constexpr int N_Q_HEADS = 16;
    constexpr int N_KV_HEADS = 8;
    constexpr int HEAD_DIM = 128;
    constexpr int HALF_DIM = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    int block_size = WARPS_PER_BLOCK * 32;

    int grid = (max(N_Q_HEADS, N_KV_HEADS) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    fused_qkv_norm_rope_cuda_kernel<HEAD_DIM, HALF_DIM, N_Q_HEADS, N_KV_HEADS, WARPS_PER_BLOCK>
    <<<grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q_norm_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_norm_w.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(cos_data.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(sin_data.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Q_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(K_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(V_out.data_ptr<at::Half>()),
        q_dim, kv_dim, q_eps, k_eps);
}


// ---------------------------------------------------------------------------
// 5. 融合K+V cache写入: 一次调用写一对K/V到flash cache
//    比index_copy_快5-10x 因为:
//    - 一次kernel写K+V两份数据 减少一半launch次数
//    - 用half2一次写2个元素
//    - 1024元素用1个warp就能搞定 不需要多个block
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(32)
write_kv_cache_kernel(
    __half* __restrict__ k_cache,  // [max_seq, flat_dim] flat view
    __half* __restrict__ v_cache,  // [max_seq, flat_dim] flat view
    const __half* __restrict__ k_new,  // [flat_dim]
    const __half* __restrict__ v_new,  // [flat_dim]
    const int32_t* __restrict__ pos_ptr,  // device pointer, graph-safe
    int half_flat_dim)              // flat_dim/2 (for half2)
{
    int pos = *pos_ptr;  // 从device tensor读pos CUDA graph兼容
    int idx = threadIdx.x;  // 1 warp = 32 threads
    const __half2* k2 = reinterpret_cast<const __half2*>(k_new);
    const __half2* v2 = reinterpret_cast<const __half2*>(v_new);
    __half2* kc2 = reinterpret_cast<__half2*>(k_cache) + pos * half_flat_dim;
    __half2* vc2 = reinterpret_cast<__half2*>(v_cache) + pos * half_flat_dim;

    #pragma unroll 4
    for (int i = idx; i < half_flat_dim; i += 32) {
        kc2[i] = __ldg(k2 + i);
        vc2[i] = __ldg(v2 + i);
    }
}


void write_kv_cache_cuda(
    torch::Tensor k_cache,  // [max_seq, flat_dim]
    torch::Tensor v_cache,  // [max_seq, flat_dim]
    torch::Tensor k_new,    // [flat_dim]
    torch::Tensor v_new,    // [flat_dim]
    torch::Tensor pos)      // [1] int32 device tensor (CUDA graph兼容)
{
    int flat_dim = k_new.size(0);
    int half_flat_dim = flat_dim / 2;
    TORCH_CHECK(flat_dim % 2 == 0, "flat_dim must be even");
    TORCH_CHECK(pos.scalar_type() == torch::kInt32, "pos must be int32");
    TORCH_CHECK(pos.is_cuda(), "pos must be on CUDA");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    write_kv_cache_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<__half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_new.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_new.data_ptr<at::Half>()),
        pos.data_ptr<int32_t>(),
        half_flat_dim);
}


// ---------------------------------------------------------------------------
// 6. Custom decode attention for batch=1 GQA (16Q / 8KV heads, head_dim=128)
//    Replaces flash_attn_with_kvcache for single-query decode.
//    Each block handles 1 KV head group (2 Q heads sharing 1 KV head).
//    Fused: KV cache write + attention compute in one kernel.
// ---------------------------------------------------------------------------

template <int HEAD_DIM, int Q_PER_KV>
__global__ void __launch_bounds__(Q_PER_KV * 32)
decode_attn_kernel(
    const __half* __restrict__ q,          // [n_qo_heads, head_dim]
    const __half* __restrict__ k_new,      // [n_kv_heads, head_dim]
    const __half* __restrict__ v_new,      // [n_kv_heads, head_dim]
    __half* __restrict__ k_cache,          // [max_seq, n_kv, head_dim]
    __half* __restrict__ v_cache,          // [max_seq, n_kv, head_dim]
    __half* __restrict__ output,           // [n_qo_heads, head_dim]
    const int32_t* __restrict__ seq_len_ptr, // device pointer [1]
    int max_seq)
{
    int kv_head = blockIdx.x;  // 1 block per KV head
    int tid = threadIdx.x;
    int lane = tid & 31;
    int q_idx = tid / 32;  // which Q head within this KV group (0 or 1)
    int seq_len = *seq_len_ptr;

    // Step 1: Write new K/V to cache at position seq_len
    // All threads in block cooperate to write K and V
    int write_pos = seq_len;  // position to write new K/V
    __half* k_cache_pos = k_cache + write_pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
    __half* v_cache_pos = v_cache + write_pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
    const __half* k_new_head = k_new + kv_head * HEAD_DIM;
    const __half* v_new_head = v_new + kv_head * HEAD_DIM;

    for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
        k_cache_pos[i] = __ldg(k_new_head + i);
        v_cache_pos[i] = __ldg(v_new_head + i);
    }
    __syncthreads();

    // Step 2: Compute attention for each Q head
    int total_seq = seq_len + 1;  // including the new token
    float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Each warp handles 1 Q head
    int global_q_head = kv_head * Q_PER_KV + q_idx;
    if (q_idx >= Q_PER_KV) return;

    const __half* q_head = q + global_q_head * HEAD_DIM;

    // Load Q head into registers (4 fp16 per thread)
    float q_vals[HEAD_DIM / 32];  // 4 values per thread
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        int idx = lane * (HEAD_DIM / 32) + i;
        q_vals[i] = __half2float(__ldg(q_head + idx));
    }

    // Compute attention scores: q @ K^T
    // Process K cache in tiles to avoid running out of registers
    float max_score = -1e30f;
    float score_buf[4];  // small buffer for scores per tile

    // First pass: compute scores and find max
    // We'll store scores in shared memory for the second pass
    extern __shared__ float s_scores[];  // [Q_PER_KV * max_seq_total]
    float* my_scores = s_scores + q_idx * max_seq;

    for (int pos = 0; pos < total_seq; pos++) {
        const __half* k_pos = k_cache + pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int idx = lane * (HEAD_DIM / 32) + i;
            float k_val = __half2float(__ldg(k_pos + idx));
            dot += q_vals[i] * k_val;
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float score = dot * sm_scale;
            my_scores[pos] = score;
            if (score > max_score) max_score = score;
        }
    }

    // Broadcast max_score within warp
    __shared__ float s_max[Q_PER_KV];
    if (lane == 0) s_max[q_idx] = max_score;
    __syncthreads();
    max_score = s_max[q_idx];

    // Second pass: compute exp(score - max) and sum
    float exp_sum = 0.0f;
    if (lane == 0) {
        for (int pos = 0; pos < total_seq; pos++) {
            my_scores[pos] = expf(my_scores[pos] - max_score);
            exp_sum += my_scores[pos];
        }
        // Store inverse sum
        s_max[q_idx] = 1.0f / (exp_sum + 1e-8f);
    }
    __syncthreads();
    float inv_sum = s_max[q_idx];

    // Third pass: weighted sum of V
    float out_vals[HEAD_DIM / 32];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 32; i++) out_vals[i] = 0.0f;

    for (int pos = 0; pos < total_seq; pos++) {
        float weight;
        if (lane == 0) weight = my_scores[pos] * inv_sum;
        weight = __shfl_sync(0xffffffff, weight, 0);

        const __half* v_pos = v_cache + pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int idx = lane * (HEAD_DIM / 32) + i;
            float v_val = __half2float(__ldg(v_pos + idx));
            out_vals[i] += weight * v_val;
        }
    }

    // Write output
    __half* out_head = output + global_q_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        int idx = lane * (HEAD_DIM / 32) + i;
        out_head[idx] = __float2half(out_vals[i]);
    }
}


template <int HEAD_DIM, int Q_PER_KV>
__global__ void __launch_bounds__(Q_PER_KV * 32)
decode_attn_stream_kernel(
    const __half* __restrict__ q,          // [n_qo_heads, head_dim]
    const __half* __restrict__ k_new,      // [n_kv_heads, head_dim]
    const __half* __restrict__ v_new,      // [n_kv_heads, head_dim]
    __half* __restrict__ k_cache,          // [max_seq, n_kv, head_dim]
    __half* __restrict__ v_cache,          // [max_seq, n_kv, head_dim]
    __half* __restrict__ output,           // [n_qo_heads, head_dim]
    const int32_t* __restrict__ seq_len_ptr,
    int max_seq)
{
    int kv_head = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int q_idx = tid / 32;
    int seq_len = *seq_len_ptr;

    int write_pos = seq_len;
    __half* k_cache_pos = k_cache + write_pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
    __half* v_cache_pos = v_cache + write_pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
    const __half* k_new_head = k_new + kv_head * HEAD_DIM;
    const __half* v_new_head = v_new + kv_head * HEAD_DIM;

    for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
        k_cache_pos[i] = __ldg(k_new_head + i);
        v_cache_pos[i] = __ldg(v_new_head + i);
    }
    __syncthreads();

    if (q_idx >= Q_PER_KV) return;

    int total_seq = seq_len + 1;
    float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);
    int global_q_head = kv_head * Q_PER_KV + q_idx;
    const __half* q_head = q + global_q_head * HEAD_DIM;

    float q_vals[HEAD_DIM / 32];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        int idx = lane * (HEAD_DIM / 32) + i;
        q_vals[i] = __half2float(__ldg(q_head + idx));
    }

    float max_score = -1e30f;
    float denom = 0.0f;

    // Streaming softmax stats. This avoids the old score-buffer pass and
    // removes the large dynamic shared-memory allocation.
    for (int pos = 0; pos < total_seq; pos++) {
        const __half* k_pos = k_cache + pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int idx = lane * (HEAD_DIM / 32) + i;
            dot += q_vals[i] * __half2float(__ldg(k_pos + idx));
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float score = dot * sm_scale;
            float new_max = fmaxf(max_score, score);
            denom = denom * expf(max_score - new_max) + expf(score - new_max);
            max_score = new_max;
        }
    }

    max_score = __shfl_sync(0xffffffff, max_score, 0);
    denom = __shfl_sync(0xffffffff, denom, 0);

    float out_vals[HEAD_DIM / 32];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 32; i++) out_vals[i] = 0.0f;

    for (int pos = 0; pos < total_seq; pos++) {
        const __half* k_pos = k_cache + pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int idx = lane * (HEAD_DIM / 32) + i;
            dot += q_vals[i] * __half2float(__ldg(k_pos + idx));
        }
        dot = warp_reduce_sum(dot);
        float weight = 0.0f;
        if (lane == 0) {
            weight = expf(dot * sm_scale - max_score) / (denom + 1e-8f);
        }
        weight = __shfl_sync(0xffffffff, weight, 0);

        const __half* v_pos = v_cache + pos * gridDim.x * HEAD_DIM + kv_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / 32; i++) {
            int idx = lane * (HEAD_DIM / 32) + i;
            out_vals[i] += weight * __half2float(__ldg(v_pos + idx));
        }
    }

    __half* out_head = output + global_q_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        int idx = lane * (HEAD_DIM / 32) + i;
        out_head[idx] = __float2half(out_vals[i]);
    }
}


void decode_attn_cuda(
    torch::Tensor q,          // [n_qo, head_dim]
    torch::Tensor k_new,      // [n_kv, head_dim]
    torch::Tensor v_new,      // [n_kv, head_dim]
    torch::Tensor k_cache,    // [max_seq, n_kv, head_dim]
    torch::Tensor v_cache,    // [max_seq, n_kv, head_dim]
    torch::Tensor output,     // [n_qo, head_dim]
    torch::Tensor seq_len)    // [1] int32
{
    int n_kv = k_new.size(0);
    int max_seq = k_cache.size(0);
    int head_dim = k_new.size(1);
    int q_per_kv = q.size(0) / n_kv;  // 2 for 16Q/8KV

    int block_size = q_per_kv * 32;  // 64 threads (2 warps)
    int grid = n_kv;  // 8 blocks (1 per KV head)
    int shared_mem = q_per_kv * max_seq * sizeof(float);  // score buffer

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const char* stream_softmax = std::getenv("AICAS_CUSTOM_DECODE_ATTN_STREAM");
    if (stream_softmax != nullptr && stream_softmax[0] == '1') {
        decode_attn_stream_kernel<128, 2><<<grid, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_new.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_new.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(k_cache.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(v_cache.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            seq_len.data_ptr<int32_t>(),
            max_seq);
    } else {
        decode_attn_kernel<128, 2><<<grid, block_size, shared_mem, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_new.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_new.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(k_cache.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(v_cache.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            seq_len.data_ptr<int32_t>(),
            max_seq);
    }
}


// (flash_decode_attn moved to flash_decode_attn.cu)


// ---------------------------------------------------------------------------
// 6b. 融合 RMS norm + GEMV: y = W @ rms_norm(x, weight, eps)
//    Phase 1: 所有线程算RMS norm 结果存shared memory
//    Phase 2: 每个warp用shared memory中的normed值做GEMV
//    省掉一次独立的RMS norm kernel launch (28次/step)
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK, int IN_DIM>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_rmsnorm_kernel(
    const __half* __restrict__ W,           // [out_dim, in_dim]
    const __half* __restrict__ x,           // [in_dim] 未归一化的输入
    const __half* __restrict__ norm_weight, // [in_dim]
    __half* __restrict__ y,                 // [out_dim]
    float eps,
    int out_dim)
{
    __shared__ __half s_normed[IN_DIM];  // 归一化后的输入 (4KB for IN_DIM=2048)
    __shared__ float s_inv_var;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Phase 1: RMS norm → shared memory
    float sum_sq = 0.0f;
    for (int i = tid; i < IN_DIM; i += block_size) {
        float val = __half2float(__ldg(x + i));
        sum_sq += val * val;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    // Block-level reduce (for >1 warp)
    __shared__ float s_partial[32];  // max warps
    int warp_id = tid / 32;
    int lane = tid & 31;
    if (lane == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < (block_size / 32)) ? s_partial[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_inv_var = rsqrtf(v / IN_DIM + eps);
    }
    __syncthreads();

    // Normalize and store in shared memory
    for (int i = tid; i < IN_DIM; i += block_size) {
        float val = __half2float(__ldg(x + i));
        float w = __half2float(__ldg(norm_weight + i));
        s_normed[i] = __float2half(val * s_inv_var * w);
    }
    __syncthreads();

    // Phase 2: GEMV using shared memory normed values
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const float4* w_row4 = reinterpret_cast<const float4*>(W + row * IN_DIM);
    const float4* x4 = reinterpret_cast<const float4*>(s_normed);
    int n_f4 = IN_DIM / 8;  // 2048->256 (每个float4 = 8个FP16)

    float sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_f4; i += 32) {
        float4 wv = __ldg(w_row4 + i);
        float4 xv = x4[i];  // 从shared memory读 不需要__ldg
        __half2* wp = reinterpret_cast<__half2*>(&wv);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 wf0 = __half22float2(wp[0]); float2 xf0 = __half22float2(xp[0]);
        float2 wf1 = __half22float2(wp[1]); float2 xf1 = __half22float2(xp[1]);
        float2 wf2 = __half22float2(wp[2]); float2 xf2 = __half22float2(xp[2]);
        float2 wf3 = __half22float2(wp[3]); float2 xf3 = __half22float2(xp[3]);
        sum += wf0.x * xf0.x + wf0.y * xf0.y + wf1.x * xf1.x + wf1.y * xf1.y
             + wf2.x * xf2.x + wf2.y * xf2.y + wf3.x * xf3.x + wf3.y * xf3.y;
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}


torch::Tensor gemv_rmsnorm_cuda(
    torch::Tensor W,           // [out_dim, in_dim]
    torch::Tensor x,           // [in_dim]
    torch::Tensor norm_weight, // [in_dim]
    float eps)
{
    int out_dim = W.size(0);
    int in_dim = W.size(1);
    auto y = torch::empty({out_dim}, W.options());

    constexpr int WARPS_PER_BLOCK = 8;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 实例化IN_DIM=2048的版本 (编译时常量 优化shared memory)
    gemv_rmsnorm_kernel<WARPS_PER_BLOCK, 2048><<<grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        eps, out_dim);

    return y;
}


// ===========================================================================
// INT8 Weight-Only GEMV kernels
// INT8权重 + per-channel scale，decode时省50%权重带宽
// ===========================================================================

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_int8_kernel(
    const int8_t* __restrict__ W,     // [N, K] int8
    const __half* __restrict__ x,     // [K] fp16
    const float* __restrict__ scale,  // [N] fp32 per-channel scale
    __half* __restrict__ y,           // [N] fp16
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;

    // 8 int8 per iteration, matched with 8 fp16 input
    int n_iter = K / 8;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        // Load 8 int8 weights (8 bytes via int2)
        int2 wv_raw = __ldg(reinterpret_cast<const int2*>(w_row) + i);
        const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);

        // Load 8 fp16 inputs (16 bytes via float4)
        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        sum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
             + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
             + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
             + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = __float2half(sum * row_scale);
}

// INT8 GEMV with L2 cache bypass (ld.global.cg) for weight reads.
// Keeps per-layer weight traffic from evicting KV cache data from L2.
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_int8_cg_kernel(
    const int8_t* __restrict__ W,     // [N, K] int8
    const __half* __restrict__ x,     // [K] fp16
    const float* __restrict__ scale,  // [N] fp32 per-channel scale
    __half* __restrict__ y,           // [N] fp16
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;
    int n_iter = K / 8;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        // __ldcg for weights: bypass L2, keep KV cache data in L2
        int2 wv_raw;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                     : "l"(reinterpret_cast<const int2*>(w_row) + i));
        const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);

        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        sum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
             + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
             + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
             + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = __float2half(sum * row_scale);
}

// Dual-row INT8 GEMV: each warp processes 2 rows for better latency hiding.
// Doubles the effective loop iterations (8 → 16 for K=2048), improving BW util.
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_int8_dual_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * 2;
    if (row0 >= N) return;

    const int8_t* w_row0 = W + row0 * K;
    float rs0 = scale[row0];
    float sum0 = 0.0f;

    const int8_t* w_row1 = nullptr;
    float rs1 = 0.0f;
    float sum1 = 0.0f;
    bool has_row1 = (row0 + 1 < N);
    if (has_row1) {
        w_row1 = W + (row0 + 1) * K;
        rs1 = scale[row0 + 1];
    }

    int n_iter = K / 8;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        int2 wv0 = __ldg(reinterpret_cast<const int2*>(w_row0) + i);
        const int8_t* wi0 = reinterpret_cast<const int8_t*>(&wv0);
        sum0 += (float)wi0[0]*xf0.x + (float)wi0[1]*xf0.y
              + (float)wi0[2]*xf1.x + (float)wi0[3]*xf1.y
              + (float)wi0[4]*xf2.x + (float)wi0[5]*xf2.y
              + (float)wi0[6]*xf3.x + (float)wi0[7]*xf3.y;

        if (has_row1) {
            int2 wv1 = __ldg(reinterpret_cast<const int2*>(w_row1) + i);
            const int8_t* wi1 = reinterpret_cast<const int8_t*>(&wv1);
            sum1 += (float)wi1[0]*xf0.x + (float)wi1[1]*xf0.y
                  + (float)wi1[2]*xf1.x + (float)wi1[3]*xf1.y
                  + (float)wi1[4]*xf2.x + (float)wi1[5]*xf2.y
                  + (float)wi1[6]*xf3.x + (float)wi1[7]*xf3.y;
        }
    }

    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    if (lane == 0) {
        y[row0] = __float2half(sum0 * rs0);
        if (has_row1) y[row0 + 1] = __float2half(sum1 * rs1);
    }
}

// INT8 lm_head: L2 bypass + 大block
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_int8_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;

    int n_iter = K / 8;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        // __ldcg: cache global, bypass L2 for large weight
        int2 wv_raw;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                     : "l"(reinterpret_cast<const int2*>(w_row) + i));
        const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);

        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        sum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
             + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
             + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
             + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = __float2half(sum * row_scale);
}

// INT8 fused GEMV + SiLU + Mul for MLP
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_silu_mul_int8_kernel(
    const int8_t* __restrict__ W_gate_up,  // [2*out_dim, K] int8
    const __half* __restrict__ x,           // [K] fp16
    const float* __restrict__ scale,        // [2*out_dim] fp32
    __half* __restrict__ y,                 // [out_dim] fp16
    int out_dim, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const int8_t* w_gate = W_gate_up + row * K;
    const int8_t* w_up = W_gate_up + (row + out_dim) * K;
    float gate_scale = scale[row];
    float up_scale = scale[row + out_dim];

    int n_iter = K / 8;
    float gate_sum = 0.0f, up_sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        int2 gv = __ldg(reinterpret_cast<const int2*>(w_gate) + i);
        const int8_t* gi = reinterpret_cast<const int8_t*>(&gv);
        gate_sum += (float)gi[0]*xf0.x + (float)gi[1]*xf0.y
                  + (float)gi[2]*xf1.x + (float)gi[3]*xf1.y
                  + (float)gi[4]*xf2.x + (float)gi[5]*xf2.y
                  + (float)gi[6]*xf3.x + (float)gi[7]*xf3.y;

        int2 uv = __ldg(reinterpret_cast<const int2*>(w_up) + i);
        const int8_t* ui = reinterpret_cast<const int8_t*>(&uv);
        up_sum += (float)ui[0]*xf0.x + (float)ui[1]*xf0.y
                + (float)ui[2]*xf1.x + (float)ui[3]*xf1.y
                + (float)ui[4]*xf2.x + (float)ui[5]*xf2.y
                + (float)ui[6]*xf3.x + (float)ui[7]*xf3.y;
    }

    gate_sum = warp_reduce_sum(gate_sum);
    up_sum = warp_reduce_sum(up_sum);

    if (lane == 0) {
        float silu_val = gate_sum * gate_scale / (1.0f + expf(-(gate_sum * gate_scale)));
        y[row] = __float2half(silu_val * up_sum * up_scale);
    }
}

// INT8 fused GEMV + residual add
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_add_int8_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,  // residual [N] in-place add
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;
    int n_iter = K / 8;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        int2 wv_raw = __ldg(reinterpret_cast<const int2*>(w_row) + i);
        const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);

        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        sum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
             + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
             + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
             + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        float old = __half2float(y[row]);
        y[row] = __float2half(old + sum * row_scale);
    }
}

// INT8 fused GEMV + residual add + extra addend:
// y[row] = y[row] + addend[row] + (W @ x)[row] * scale[row]
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_add_both_int8_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    const __half* __restrict__ addend,
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;
    int n_iter = K / 8;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        int2 wv_raw = __ldg(reinterpret_cast<const int2*>(w_row) + i);
        const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);

        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        sum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
             + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
             + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
             + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        float old = __half2float(y[row]);
        float extra = __half2float(addend[row]);
        y[row] = __float2half(old + extra + sum * row_scale);
    }
}

// ---------------------------------------------------------------------------
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_add_both_int8_cg_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    const __half* __restrict__ addend,
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int8_t* w_row = W + row * K;
    float row_scale = scale[row];
    float sum = 0.0f;
    int n_iter = K / 8;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        int2 wv_raw;
        asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                     : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                       "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                     : "l"(reinterpret_cast<const int2*>(w_row) + i));
        const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);

        float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        sum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
             + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
             + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
             + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        float old = __half2float(y[row]);
        float extra = __half2float(addend[row]);
        y[row] = __float2half(old + extra + sum * row_scale);
    }
}

// Python接口 for INT8 GEMV
torch::Tensor gemv_int8_cuda(
    torch::Tensor W_int8,    // [N, K] int8
    torch::Tensor x,         // [K] fp16
    torch::Tensor scale)     // [N] fp32
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);
    auto y = torch::empty({N}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Dual-row mode: each warp processes 2 rows, better latency hiding for small K
    const char* dual = std::getenv("AICAS_INT8_GEMV_DUAL");
    const char* w24 = std::getenv("AICAS_INT8_GEMV_WARPS24");
    const char* w16 = std::getenv("AICAS_INT8_GEMV_WARPS16");
    const char* cg = std::getenv("AICAS_INT8_GEMV_CG");

    if (dual != nullptr && dual[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 8;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = ((N + 1) / 2 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_int8_dual_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    } else if (cg != nullptr && cg[0] == '1' && w16 != nullptr && w16[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 16;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_int8_cg_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    } else if (cg != nullptr && cg[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 8;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_int8_cg_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    } else if (w24 != nullptr && w24[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 24;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    } else if (w16 != nullptr && w16[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 16;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    } else {
        constexpr int WARPS_PER_BLOCK = 8;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    }

    return y;
}

torch::Tensor gemv_lmhead_int8_cuda(
    torch::Tensor W_int8,
    torch::Tensor x,
    torch::Tensor scale)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);
    auto y = torch::empty({N}, x.options());

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_lmhead_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K);

    return y;
}

// INT8 lm_head + fused argmax: same GEMV but finds argmax in-place,
// avoids writing 151K logits to global memory
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_int8_argmax_stage1_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ block_max_val,
    int* __restrict__ block_max_idx,
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid = (row < N);

    float accum = 0.0f;
    if (valid) {
        const int8_t* w_row = W + row * K;
        float row_scale = scale[row];
        int n_iter = K / 8;

        #pragma unroll 4
        for (int i = lane; i < n_iter; i += 32) {
            int2 wv_raw;
            asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                         : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                           "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                         : "l"(reinterpret_cast<const int2*>(w_row) + i));
            const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);
            float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            accum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
                   + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
                   + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
                   + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
        }
        accum *= row_scale;
    }
    accum = warp_reduce_sum(accum);

    // Warp-level argmax
    float best_val = (valid && lane == 0) ? accum : -1e30f;
    int best_row = (valid && lane == 0) ? row : N;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, offset);
        int oi = __shfl_down_sync(0xffffffff, best_row, offset);
        if (ov > best_val || (ov == best_val && oi < best_row)) {
            best_val = ov; best_row = oi;
        }
    }

    // Block-level argmax
    __shared__ float smem_val[WARPS_PER_BLOCK];
    __shared__ int smem_idx[WARPS_PER_BLOCK];
    if (lane == 0) { smem_val[warp_id] = best_val; smem_idx[warp_id] = best_row; }
    __syncthreads();

    if (warp_id == 0) {
        float bv = (lane < WARPS_PER_BLOCK) ? smem_val[lane] : -1e30f;
        int bi = (lane < WARPS_PER_BLOCK) ? smem_idx[lane] : N;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, offset);
            int oi = __shfl_down_sync(0xffffffff, bi, offset);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            block_max_val[blockIdx.x] = bv;
            block_max_idx[blockIdx.x] = bi;
        }
    }
}

// Stage 2 for INT8 argmax (reuse the generic argmax_reduce_kernel below)
__global__ void int8_argmax_reduce_kernel(
    const float* vals, const int* idxs, int64_t* out, int nblocks)
{
    float best_val = -1e30f;
    int best_idx = 0x7fffffff;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
        float v = vals[i]; int idx = idxs[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v; best_idx = idx;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, offset);
        int oi = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (ov > best_val || (ov == best_val && oi < best_idx)) {
            best_val = ov; best_idx = oi;
        }
    }

    __shared__ float warp_vals[8];
    __shared__ int warp_idxs[8];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_vals[warp_id] = best_val;
        warp_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        best_val = (lane < 8) ? warp_vals[lane] : -1e30f;
        best_idx = (lane < 8) ? warp_idxs[lane] : 0x7fffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, best_val, offset);
            int oi = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (ov > best_val || (ov == best_val && oi < best_idx)) {
                best_val = ov; best_idx = oi;
            }
        }
        if (lane == 0) out[0] = best_idx;
    }
}

__global__ void int8_argmax_reduce_stage2_kernel(
    const float* vals, const int* idxs,
    float* partial_vals, int* partial_idxs,
    int nblocks)
{
    int tile = blockIdx.x;
    int start = tile * blockDim.x;
    int end = min(start + blockDim.x, nblocks);
    float best_val = -1e30f;
    int best_idx = 0x7fffffff;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = vals[i];
        int idx = idxs[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, offset);
        int oi = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (ov > best_val || (ov == best_val && oi < best_idx)) {
            best_val = ov;
            best_idx = oi;
        }
    }

    __shared__ float warp_vals[8];
    __shared__ int warp_idxs[8];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_vals[warp_id] = best_val;
        warp_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        best_val = (lane < 8) ? warp_vals[lane] : -1e30f;
        best_idx = (lane < 8) ? warp_idxs[lane] : 0x7fffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, best_val, offset);
            int oi = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (ov > best_val || (ov == best_val && oi < best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            partial_vals[tile] = best_val;
            partial_idxs[tile] = best_idx;
        }
    }
}

void gemv_lmhead_int8_argmax_cuda(
    torch::Tensor W_int8,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor block_vals,
    torch::Tensor block_idxs,
    torch::Tensor result)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_lmhead_int8_argmax_stage1_kernel<WARPS_PER_BLOCK>
    <<<grid, block_size, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        N, K);

    int8_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        result.data_ptr<int64_t>(),
        grid);
}

// FP16 lm_head + fused argmax. This preserves lm_head precision while avoiding
// a 151K-element logits tensor write plus a separate torch.argmax reduction.
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_fp16_argmax_stage1_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    float* __restrict__ block_max_val,
    int* __restrict__ block_max_idx,
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid = row < N;

    float sum = 0.0f;
    if (valid) {
        const float4* w_row4 = reinterpret_cast<const float4*>(W + row * K);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        int n_iter = K / 8;

        #pragma unroll 4
        for (int i = lane; i < n_iter; i += 32) {
            float4 wv;
            asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(reinterpret_cast<unsigned*>(&wv)[0]),
                           "=r"(reinterpret_cast<unsigned*>(&wv)[1]),
                           "=r"(reinterpret_cast<unsigned*>(&wv)[2]),
                           "=r"(reinterpret_cast<unsigned*>(&wv)[3])
                         : "l"(w_row4 + i));
            float4 xv = __ldg(x4 + i);
            __half2* wp = reinterpret_cast<__half2*>(&wv);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 wf0 = __half22float2(wp[0]); float2 xf0 = __half22float2(xp[0]);
            float2 wf1 = __half22float2(wp[1]); float2 xf1 = __half22float2(xp[1]);
            float2 wf2 = __half22float2(wp[2]); float2 xf2 = __half22float2(xp[2]);
            float2 wf3 = __half22float2(wp[3]); float2 xf3 = __half22float2(xp[3]);
            sum += wf0.x * xf0.x + wf0.y * xf0.y
                 + wf1.x * xf1.x + wf1.y * xf1.y
                 + wf2.x * xf2.x + wf2.y * xf2.y
                 + wf3.x * xf3.x + wf3.y * xf3.y;
        }
    }
    sum = warp_reduce_sum(sum);

    // Only lane 0 has the complete warp-reduced dot product. Other lanes hold
    // partial sums and must not participate in the row argmax.
    float best_val = (valid && lane == 0) ? sum : -1e30f;
    int best_row = valid ? row : N;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, offset);
        int oi = __shfl_down_sync(0xffffffff, best_row, offset);
        if (ov > best_val || (ov == best_val && oi < best_row)) {
            best_val = ov;
            best_row = oi;
        }
    }

    __shared__ float smem_val[WARPS_PER_BLOCK];
    __shared__ int smem_idx[WARPS_PER_BLOCK];
    if (lane == 0) {
        smem_val[warp_id] = best_val;
        smem_idx[warp_id] = best_row;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bv = (lane < WARPS_PER_BLOCK) ? smem_val[lane] : -1e30f;
        int bi = (lane < WARPS_PER_BLOCK) ? smem_idx[lane] : N;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, offset);
            int oi = __shfl_down_sync(0xffffffff, bi, offset);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            block_max_val[blockIdx.x] = bv;
            block_max_idx[blockIdx.x] = bi;
        }
    }
}

void gemv_lmhead_fp16_argmax_cuda(
    torch::Tensor W,
    torch::Tensor x,
    torch::Tensor block_vals,
    torch::Tensor block_idxs,
    torch::Tensor result)
{
    int N = W.size(0);
    int K = W.size(1);

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_lmhead_fp16_argmax_stage1_kernel<WARPS_PER_BLOCK>
    <<<grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        N, K);

    int8_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        result.data_ptr<int64_t>(),
        grid);
}

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
lmhead_fp16_refine_candidates_kernel(
    const __half* __restrict__ W_fp16,
    const __half* __restrict__ x,
    float* __restrict__ vals,
    int* __restrict__ idxs,
    int n_candidates,
    int N,
    int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int cand_i = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (cand_i >= n_candidates) return;

    int row = idxs[cand_i];
    if (row < 0 || row >= N) {
        if (lane == 0) vals[cand_i] = -1e30f;
        return;
    }
    const float4* w4 = reinterpret_cast<const float4*>(W_fp16 + row * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_iter = K / 8;
    float sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        float4 wv = __ldg(w4 + i);
        float4 xv = __ldg(x4 + i);
        __half2* wp = reinterpret_cast<__half2*>(&wv);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 wf0 = __half22float2(wp[0]); float2 xf0 = __half22float2(xp[0]);
        float2 wf1 = __half22float2(wp[1]); float2 xf1 = __half22float2(xp[1]);
        float2 wf2 = __half22float2(wp[2]); float2 xf2 = __half22float2(xp[2]);
        float2 wf3 = __half22float2(wp[3]); float2 xf3 = __half22float2(xp[3]);
        sum += wf0.x * xf0.x + wf0.y * xf0.y
             + wf1.x * xf1.x + wf1.y * xf1.y
             + wf2.x * xf2.x + wf2.y * xf2.y
             + wf3.x * xf3.x + wf3.y * xf3.y;
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) vals[cand_i] = sum;
}

void gemv_lmhead_int8_refine_cuda(
    torch::Tensor W_int8,
    torch::Tensor W_fp16,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor block_vals,
    torch::Tensor block_idxs,
    torch::Tensor result)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_lmhead_int8_argmax_stage1_kernel<WARPS_PER_BLOCK>
    <<<grid, block_size, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        N, K);

    // Coarse INT8 reduce: filter 9497 block winners → top-K candidates
    // before expensive FP16 refinement.  Saves ~35MB FP16 weight reads.
    int n_candidates = grid;
    const char* coarse_env = std::getenv("AICAS_LMHEAD_REFINE_COARSE");
    if (coarse_env != nullptr) {
        int coarse_group = atoi(coarse_env);
        if (coarse_group > 1) {
            int n_tiles = (grid + coarse_group - 1) / coarse_group;
            // In-place partial reduce: each block finds best in its chunk
            int8_argmax_reduce_stage2_kernel<<<n_tiles, 256, 0, stream>>>(
                block_vals.data_ptr<float>(),
                block_idxs.data_ptr<int>(),
                block_vals.data_ptr<float>(),
                block_idxs.data_ptr<int>(),
                grid);
            n_candidates = n_tiles;
        }
    }

    int refine_grid = (n_candidates + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    lmhead_fp16_refine_candidates_kernel<WARPS_PER_BLOCK>
    <<<refine_grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        n_candidates, N, K);

    int8_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        result.data_ptr<int64_t>(),
        n_candidates);
}

template <int WARPS_PER_BLOCK, int TOPK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_int8_topk_stage1_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ cand_vals,
    int* __restrict__ cand_idxs,
    int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int block_start = blockIdx.x * WARPS_PER_BLOCK;
    int row = block_start + warp_id;
    bool valid = (row < N);

    float accum = 0.0f;
    if (valid) {
        const int8_t* w_row = W + row * K;
        float row_scale = scale[row];
        int n_iter = K / 8;

        #pragma unroll 4
        for (int i = lane; i < n_iter; i += 32) {
            int2 wv_raw;
            asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];"
                         : "=r"(reinterpret_cast<unsigned*>(&wv_raw)[0]),
                           "=r"(reinterpret_cast<unsigned*>(&wv_raw)[1])
                         : "l"(reinterpret_cast<const int2*>(w_row) + i));
            const int8_t* wi = reinterpret_cast<const int8_t*>(&wv_raw);
            float4 xv = __ldg(reinterpret_cast<const float4*>(x) + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
            accum += (float)wi[0]*xf0.x + (float)wi[1]*xf0.y
                   + (float)wi[2]*xf1.x + (float)wi[3]*xf1.y
                   + (float)wi[4]*xf2.x + (float)wi[5]*xf2.y
                   + (float)wi[6]*xf3.x + (float)wi[7]*xf3.y;
        }
        accum *= row_scale;
    }
    accum = warp_reduce_sum(accum);

    __shared__ float smem_val[WARPS_PER_BLOCK];
    __shared__ int smem_idx[WARPS_PER_BLOCK];
    if (lane == 0) {
        smem_val[warp_id] = valid ? accum : -1e30f;
        smem_idx[warp_id] = valid ? row : N;
    }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        bool used[WARPS_PER_BLOCK];
        #pragma unroll
        for (int i = 0; i < WARPS_PER_BLOCK; ++i) used[i] = false;

        #pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float best_val = -1e30f;
            int best_idx = N;
            int best_slot = -1;
            #pragma unroll
            for (int i = 0; i < WARPS_PER_BLOCK; ++i) {
                if (used[i]) continue;
                float v = smem_val[i];
                int idx = smem_idx[i];
                if (v > best_val || (v == best_val && idx < best_idx)) {
                    best_val = v;
                    best_idx = idx;
                    best_slot = i;
                }
            }
            if (best_slot >= 0) used[best_slot] = true;
            cand_vals[blockIdx.x * TOPK + k] = best_val;
            cand_idxs[blockIdx.x * TOPK + k] = best_idx;
        }
    }
}

void gemv_lmhead_int8_refine_topk_cuda(
    torch::Tensor W_int8,
    torch::Tensor W_fp16,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor cand_vals,
    torch::Tensor cand_idxs,
    torch::Tensor result)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);

    constexpr int WARPS_PER_BLOCK = 16;
    constexpr int TOPK = 4;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_lmhead_int8_topk_stage1_kernel<WARPS_PER_BLOCK, TOPK>
    <<<grid, block_size, 0, stream>>>(
        W_int8.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        cand_vals.data_ptr<float>(),
        cand_idxs.data_ptr<int>(),
        N, K);

    int n_candidates = grid * TOPK;
    int refine_grid = (n_candidates + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    lmhead_fp16_refine_candidates_kernel<WARPS_PER_BLOCK>
    <<<refine_grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(W_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        cand_vals.data_ptr<float>(),
        cand_idxs.data_ptr<int>(),
        n_candidates, N, K);

    int8_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        cand_vals.data_ptr<float>(),
        cand_idxs.data_ptr<int>(),
        result.data_ptr<int64_t>(),
        n_candidates);
}

torch::Tensor gemv_silu_mul_int8_cuda(
    torch::Tensor W_gate_up_int8,
    torch::Tensor x,
    torch::Tensor scale)
{
    int total_rows = W_gate_up_int8.size(0);
    int out_dim = total_rows / 2;
    int K = W_gate_up_int8.size(1);
    auto y = torch::empty({out_dim}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const char* w16 = std::getenv("AICAS_INT8_GEMV_WARPS16");
    if (w16 != nullptr && w16[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 16;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_silu_mul_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_gate_up_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, K);
    } else {
        constexpr int WARPS_PER_BLOCK = 8;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_silu_mul_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_gate_up_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, K);
    }

    return y;
}

void gemv_add_int8_cuda(
    torch::Tensor W_int8,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor y)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const char* w24 = std::getenv("AICAS_INT8_ADD_WARPS24");
    const char* w16 = std::getenv("AICAS_INT8_GEMV_WARPS16");
    if (w24 != nullptr && w24[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 24;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    } else if (w16 != nullptr && w16[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 16;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    } else {
        constexpr int WARPS_PER_BLOCK = 8;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, K);
    }
}

void gemv_add_both_int8_cuda(
    torch::Tensor W_int8,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor y,
    torch::Tensor addend)
{
    int N = W_int8.size(0);
    int K = W_int8.size(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const char* cg = std::getenv("AICAS_INT8_ADD_BOTH_CG");
    const char* w24 = std::getenv("AICAS_INT8_ADD_BOTH_WARPS24");
    const char* w16 = std::getenv("AICAS_INT8_GEMV_WARPS16");
    if (cg != nullptr && cg[0] == '1' && w24 != nullptr && w24[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 24;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_both_int8_cg_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
            N, K);
    } else if (cg != nullptr && cg[0] == '1' && w16 != nullptr && w16[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 16;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_both_int8_cg_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
            N, K);
    } else if (cg != nullptr && cg[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 8;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_both_int8_cg_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
            N, K);
    } else if (w24 != nullptr && w24[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 24;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_both_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
            N, K);
    } else if (w16 != nullptr && w16[0] == '1') {
        constexpr int WARPS_PER_BLOCK = 16;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_both_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
            N, K);
    } else {
        constexpr int WARPS_PER_BLOCK = 8;
        int block_size = WARPS_PER_BLOCK * 32;
        int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        gemv_add_both_int8_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
            W_int8.data_ptr<int8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
            N, K);
    }
}


// ===========================================================================
// INT4 Weight-Only GEMV kernels (per-channel)
// Packing: 2 int4 per byte (unsigned with bias 8), per-channel float32 scale
// Optimization: bias correction trick — unsigned dot product minus 8*input_sum
//   avoids per-element sign extension, minimal instruction overhead
// ===========================================================================

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_int4_kernel(
    const uint8_t* __restrict__ W_packed,  // [N, K/2] uint8, 2 int4 per byte
    const __half* __restrict__ x,           // [K] fp16
    const float* __restrict__ scale,        // [N, num_groups] or [N] fp32 per-group scale
    __half* __restrict__ y,                 // [N] fp16
    int N, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);  // packed uint32 per group
    float accum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        uint32_t packed;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(w_row + i));
        float4 xv = __ldg(x4 + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

        float raw = (float)((packed      ) & 0xF) * xf0.x
                  + (float)((packed >>  4) & 0xF) * xf0.y
                  + (float)((packed >>  8) & 0xF) * xf1.x
                  + (float)((packed >> 12) & 0xF) * xf1.y
                  + (float)((packed >> 16) & 0xF) * xf2.x
                  + (float)((packed >> 20) & 0xF) * xf2.y
                  + (float)((packed >> 24) & 0xF) * xf3.x
                  + (float)( packed >> 28        ) * xf3.y;

        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);
        accum += (raw - 8.0f * is) * gs;
    }

    accum = warp_reduce_sum(accum);
    if (lane == 0) y[row] = __float2half(accum);
}

// INT4 lm_head: L2 bypass + large block, per-group scale
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_int4_kernel(
    const uint8_t* __restrict__ W_packed,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    int N, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);
    float accum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        uint32_t packed;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(w_row + i));

        float4 xv = __ldg(x4 + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

        float raw = (float)((packed      ) & 0xF) * xf0.x
                  + (float)((packed >>  4) & 0xF) * xf0.y
                  + (float)((packed >>  8) & 0xF) * xf1.x
                  + (float)((packed >> 12) & 0xF) * xf1.y
                  + (float)((packed >> 16) & 0xF) * xf2.x
                  + (float)((packed >> 20) & 0xF) * xf2.y
                  + (float)((packed >> 24) & 0xF) * xf3.x
                  + (float)( packed >> 28        ) * xf3.y;

        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);
        accum += (raw - 8.0f * is) * gs;
    }

    accum = warp_reduce_sum(accum);
    if (lane == 0) y[row] = __float2half(accum);
}

// INT4 fused GEMV + SiLU + Mul for MLP gate_up
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_silu_mul_int4_kernel(
    const uint8_t* __restrict__ W_packed,  // [2*out_dim, K/2]
    const __half* __restrict__ x,
    const float* __restrict__ scale,        // [2*out_dim, num_groups]
    __half* __restrict__ y,
    int out_dim, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const uint32_t* w_gate = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    const uint32_t* w_up   = reinterpret_cast<const uint32_t*>(W_packed + (row + out_dim) * (K / 2));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);
    float gate_accum = 0.0f, up_accum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        float4 xv = __ldg(x4 + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
        float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

        uint32_t pg;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(pg) : "l"(w_gate + i));
        float gate_raw = (float)((pg      ) & 0xF) * xf0.x
                       + (float)((pg >>  4) & 0xF) * xf0.y
                       + (float)((pg >>  8) & 0xF) * xf1.x
                       + (float)((pg >> 12) & 0xF) * xf1.y
                       + (float)((pg >> 16) & 0xF) * xf2.x
                       + (float)((pg >> 20) & 0xF) * xf2.y
                       + (float)((pg >> 24) & 0xF) * xf3.x
                       + (float)( pg >> 28        ) * xf3.y;

        uint32_t pu;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(pu) : "l"(w_up + i));
        float up_raw = (float)((pu      ) & 0xF) * xf0.x
                     + (float)((pu >>  4) & 0xF) * xf0.y
                     + (float)((pu >>  8) & 0xF) * xf1.x
                     + (float)((pu >> 12) & 0xF) * xf1.y
                     + (float)((pu >> 16) & 0xF) * xf2.x
                     + (float)((pu >> 20) & 0xF) * xf2.y
                     + (float)((pu >> 24) & 0xF) * xf3.x
                     + (float)( pu >> 28        ) * xf3.y;

        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);
        float us = __ldg(scale + (row + out_dim) * num_groups + g);
        gate_accum += (gate_raw - 8.0f * is) * gs;
        up_accum += (up_raw - 8.0f * is) * us;
    }

    gate_accum = warp_reduce_sum(gate_accum);
    up_accum = warp_reduce_sum(up_accum);

    if (lane == 0) {
        float g = gate_accum;
        float u = up_accum;
        float silu_val = g / (1.0f + expf(-g));
        y[row] = __float2half(silu_val * u);
    }
}

// INT4 fused GEMV + residual add, per-group scale
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_add_int4_kernel(
    const uint8_t* __restrict__ W_packed,
    const __half* __restrict__ x,
    const float* __restrict__ scale,        // [N, num_groups]
    __half* __restrict__ y,  // residual [N] in-place add
    int N, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);
    float accum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        uint32_t packed;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(w_row + i));
        float4 xv = __ldg(x4 + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

        float raw = (float)((packed      ) & 0xF) * xf0.x
                  + (float)((packed >>  4) & 0xF) * xf0.y
                  + (float)((packed >>  8) & 0xF) * xf1.x
                  + (float)((packed >> 12) & 0xF) * xf1.y
                  + (float)((packed >> 16) & 0xF) * xf2.x
                  + (float)((packed >> 20) & 0xF) * xf2.y
                  + (float)((packed >> 24) & 0xF) * xf3.x
                  + (float)( packed >> 28        ) * xf3.y;

        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);
        accum += (raw - 8.0f * is) * gs;
    }

    accum = warp_reduce_sum(accum);
    if (lane == 0) {
        float old = __half2float(y[row]);
        y[row] = __float2half(old + accum);
    }
}


// Python接口 for INT4 GEMV (per-channel)
torch::Tensor gemv_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor scale,
    int num_groups)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({N}, x.options());

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_int4_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K, num_groups);

    return y;
}

// INT4 GEMV with 16 warps/block for medium-sized matrices
torch::Tensor gemv_int4_w16_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor scale,
    int num_groups)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({N}, x.options());

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_int4_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K, num_groups);

    return y;
}

torch::Tensor gemv_lmhead_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor scale,
    int num_groups)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({N}, x.options());

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_lmhead_int4_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K, num_groups);

    return y;
}

// Fused lm_head INT4 GEMV + argmax: computes W @ x and finds argmax in one kernel
// Saves writing 151936 logits to global memory + separate argmax kernel
// Two-stage: stage1 writes per-block (val, idx) pairs, stage2 reduces them
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_lmhead_argmax_int4_stage1_kernel(
    const uint8_t* __restrict__ W_packed,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ block_max_val,  // [num_blocks]
    int* __restrict__ block_max_idx,     // [num_blocks]
    int N, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid = (row < N);

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);
    float accum = 0.0f;

    if (valid) {
        #pragma unroll 4
        for (int i = lane; i < n_iter; i += 32) {
            uint32_t packed;
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(w_row + i));

            float4 xv = __ldg(x4 + i);
            __half2* xp = reinterpret_cast<__half2*>(&xv);
            float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
            float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

            float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

            float raw = (float)((packed      ) & 0xF) * xf0.x
                      + (float)((packed >>  4) & 0xF) * xf0.y
                      + (float)((packed >>  8) & 0xF) * xf1.x
                      + (float)((packed >> 12) & 0xF) * xf1.y
                      + (float)((packed >> 16) & 0xF) * xf2.x
                      + (float)((packed >> 20) & 0xF) * xf2.y
                      + (float)((packed >> 24) & 0xF) * xf3.x
                      + (float)( packed >> 28        ) * xf3.y;

            int g = i / group_elems;
            float gs = __ldg(scale + row * num_groups + g);
            accum += (raw - 8.0f * is) * gs;
        }
    }

    // Warp reduce sum (OOB lanes contribute 0, which is neutral)
    accum = warp_reduce_sum(accum);

    // For argmax: valid lanes use their computed value, invalid use -inf
    float best_val = (valid && lane == 0) ? accum : -1e30f;
    int best_row = (valid && lane == 0) ? row : N;
    // Tree-style reduce: lane 0 gets the warp's maximum
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int other_row = __shfl_down_sync(0xffffffff, best_row, offset);
        if (other_val > best_val || (other_val == best_val && other_row < best_row)) {
            best_val = other_val;
            best_row = other_row;
        }
    }

    // Block-level reduce: lane 0 of each warp shares via shared memory
    __shared__ float smem_val[WARPS_PER_BLOCK];
    __shared__ int smem_idx[WARPS_PER_BLOCK];
    if (lane == 0) {
        smem_val[warp_id] = best_val;
        smem_idx[warp_id] = best_row;
    }
    __syncthreads();

    // Warp 0 does final block reduce
    if (warp_id == 0) {
        float bv = (lane < WARPS_PER_BLOCK) ? smem_val[lane] : -1e30f;
        int bi = (lane < WARPS_PER_BLOCK) ? smem_idx[lane] : -1;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, offset);
            int oi = __shfl_down_sync(0xffffffff, bi, offset);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            block_max_val[blockIdx.x] = bv;
            block_max_idx[blockIdx.x] = bi;
        }
    }
}

// Stage 2: find global max from per-block results
__global__ void argmax_reduce_kernel(
    const float* __restrict__ vals,
    const int* __restrict__ idxs,
    int64_t* __restrict__ out,
    int N)
{
    float best_val = -1e30f;
    int best_idx = -1;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = vals[i];
        int idx = idxs[i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }
    // Warp reduce using down (tree reduce, lane 0 gets result)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, offset);
        int oi = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (ov > best_val || (ov == best_val && oi < best_idx)) {
            best_val = ov;
            best_idx = oi;
        }
    }
    if (threadIdx.x == 0) {
        out[0] = best_idx;
    }
}

torch::Tensor gemv_lmhead_argmax_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor scale,
    int num_groups)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Allocate per-block buffers
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto opts_l = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
    auto block_vals = torch::empty({grid}, opts_f);
    auto block_idxs = torch::empty({grid}, opts_i);
    auto result = torch::empty({1}, opts_l);

    gemv_lmhead_argmax_int4_stage1_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        N, K, num_groups);

    argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int>(),
        result.data_ptr<int64_t>(),
        grid);

    return result;
}

torch::Tensor gemv_silu_mul_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor scale,
    int num_groups)
{
    int total_rows = W_packed.size(0);
    int out_dim = total_rows / 2;
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({out_dim}, x.options());

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_silu_mul_int4_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        out_dim, K, num_groups);

    return y;
}

void gemv_add_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor y,
    int num_groups)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_add_int4_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, K, num_groups);
}

// INT4 GEMV + residual add + extra addend: y[row] += addend[row] + (W @ x)[row]
// Fuses residual.add_(o_flat) + down_proj into one kernel
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_add_both_int4_kernel(
    const uint8_t* __restrict__ W_packed,
    const __half* __restrict__ x,
    const float* __restrict__ scale,
    __half* __restrict__ y,
    const __half* __restrict__ addend,
    int N, int K, int num_groups)
{
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (K / 2));
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n_iter = K / 8;
    int group_elems = K / (num_groups * 8);
    float accum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        uint32_t packed;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(w_row + i));
        float4 xv = __ldg(x4 + i);
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

        float raw = (float)((packed      ) & 0xF) * xf0.x
                  + (float)((packed >>  4) & 0xF) * xf0.y
                  + (float)((packed >>  8) & 0xF) * xf1.x
                  + (float)((packed >> 12) & 0xF) * xf1.y
                  + (float)((packed >> 16) & 0xF) * xf2.x
                  + (float)((packed >> 20) & 0xF) * xf2.y
                  + (float)((packed >> 24) & 0xF) * xf3.x
                  + (float)( packed >> 28        ) * xf3.y;

        int g = i / group_elems;
        float gs = __ldg(scale + row * num_groups + g);
        accum += (raw - 8.0f * is) * gs;
    }

    accum = warp_reduce_sum(accum);
    if (lane == 0) {
        float old = __half2float(y[row]);
        float extra = __half2float(addend[row]);
        y[row] = __float2half(old + extra + accum);
    }
}

void gemv_add_both_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor y,
    torch::Tensor addend,
    int num_groups)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    gemv_add_both_int4_kernel<WARPS_PER_BLOCK><<<grid, block_size, 0, stream>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(addend.data_ptr<at::Half>()),
        N, K, num_groups);
}


// ---------------------------------------------------------------------------
// 融合 RMS norm + INT4 GEMV: 省掉一次独立的RMS norm kernel launch
// 输入未归一化的x和norm_weight，先做RMS norm到shared memory，再做INT4 GEMV
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK, int IN_DIM>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_rmsnorm_int4_kernel(
    const uint8_t* __restrict__ W_packed,  // [N, K/2] INT4 packed
    const __half* __restrict__ x,           // [K] 未归一化的输入
    const __half* __restrict__ norm_weight, // [K] RMS norm权重
    const float* __restrict__ scale,        // [N] INT4 per-channel scale
    __half* __restrict__ y,                 // [N] 输出
    int N, float eps)
{
    __shared__ __align__(16) __half s_normed[IN_DIM];
    __shared__ float s_inv_var;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int block_sz = blockDim.x;
    int nwarps = block_sz / 32;

    // Phase 1: RMS norm → shared memory
    float sum_sq = 0.0f;
    for (int i = tid; i < IN_DIM; i += block_sz) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float s_partial[32];
    if (lane == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float total = 0.0f;
        for (int i = 0; i < nwarps; i++) total += s_partial[i];
        if (lane == 0) s_inv_var = rsqrtf(total / IN_DIM + eps);
    }
    __syncthreads();

    for (int i = tid; i < IN_DIM; i += block_sz) {
        float val = __half2float(x[i]);
        float w = __half2float(norm_weight[i]);
        s_normed[i] = __float2half(val * s_inv_var * w);
    }
    __syncthreads();

    // Phase 2: INT4 GEMV using shared memory normed values
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const uint32_t* w_row = reinterpret_cast<const uint32_t*>(W_packed + row * (IN_DIM / 2));
    float row_scale = __ldg(scale + row);
    const float4* x4 = reinterpret_cast<const float4*>(s_normed);
    int n_iter = IN_DIM / 8;
    float raw_sum = 0.0f;
    float input_sum = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        uint32_t packed;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(packed) : "l"(w_row + i));
        float4 xv = x4[i];
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);

        input_sum += xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;

        raw_sum += (float)((packed      ) & 0xF) * xf0.x
                 + (float)((packed >>  4) & 0xF) * xf0.y
                 + (float)((packed >>  8) & 0xF) * xf1.x
                 + (float)((packed >> 12) & 0xF) * xf1.y
                 + (float)((packed >> 16) & 0xF) * xf2.x
                 + (float)((packed >> 20) & 0xF) * xf2.y
                 + (float)((packed >> 24) & 0xF) * xf3.x
                 + (float)( packed >> 28        ) * xf3.y;
    }

    float corrected = raw_sum - 8.0f * input_sum;
    corrected = warp_reduce_sum(corrected);
    if (lane == 0) y[row] = __float2half(corrected * row_scale);
}

torch::Tensor gemv_rmsnorm_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor scale,
    float eps)
{
    int N = W_packed.size(0);
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({N}, W_packed.options().dtype(torch::kFloat16));

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (K == 2048) {
        gemv_rmsnorm_int4_kernel<WARPS_PER_BLOCK, 2048><<<grid, block_size, 0, stream>>>(
            W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, eps);
    } else if (K == 3072) {
        gemv_rmsnorm_int4_kernel<WARPS_PER_BLOCK, 3072><<<grid, block_size, 0, stream>>>(
            W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, eps);
    } else {
        // Fallback for other dimensions (uses dynamic shared memory)
        gemv_rmsnorm_int4_kernel<WARPS_PER_BLOCK, 2048><<<grid, block_size, 0, stream>>>(
            W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            N, eps);
    }
    return y;
}


// ---------------------------------------------------------------------------
// 融合 Add + RMS norm + INT4 GEMV SiLU+Mul: 省掉一次fused_add_rms_norm kernel
// 输入hidden(残差), attn_output, norm_weight → add → RMS norm → INT4 GEMV SiLU+Mul
// ---------------------------------------------------------------------------

template <int WARPS_PER_BLOCK, int IN_DIM>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32)
gemv_addrmsnorm_silu_mul_int4_kernel(
    const uint8_t* __restrict__ W_packed,  // [2*out_dim, K/2] INT4 packed
    const __half* __restrict__ hidden,      // [K] 残差(hidden_states)
    const __half* __restrict__ attn_out,    // [K] attention输出
    const __half* __restrict__ norm_weight, // [K] RMS norm权重
    const float* __restrict__ scale,        // [2*out_dim]
    __half* __restrict__ hidden_out,        // [K] 更新后的hidden (hidden += attn_out)
    __half* __restrict__ y,                 // [out_dim] MLP中间结果
    int out_dim, float eps)
{
    __shared__ __align__(16) __half s_normed[IN_DIM];
    __shared__ float s_inv_var;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int block_sz = blockDim.x;
    int nwarps = block_sz / 32;

    // Phase 1: add + RMS norm → shared memory
    float sum_sq = 0.0f;
    for (int i = tid; i < IN_DIM; i += block_sz) {
        float h = __half2float(hidden[i]);
        float a = __half2float(attn_out[i]);
        float val = h + a;
        sum_sq += val * val;
        if (hidden_out != nullptr) {
            hidden_out[i] = __float2half(val);
        }
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float s_partial[32];
    if (lane == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float total = 0.0f;
        for (int i = 0; i < nwarps; i++) total += s_partial[i];
        if (lane == 0) s_inv_var = rsqrtf(total / IN_DIM + eps);
    }
    __syncthreads();

    for (int i = tid; i < IN_DIM; i += block_sz) {
        float h = __half2float(hidden[i]);
        float a = __half2float(attn_out[i]);
        float val = h + a;
        float w = __half2float(norm_weight[i]);
        s_normed[i] = __float2half(val * s_inv_var * w);
    }
    __syncthreads();

    // Phase 2: INT4 GEMV SiLU+Mul using shared memory normed values
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const uint32_t* w_gate = reinterpret_cast<const uint32_t*>(W_packed + row * (IN_DIM / 2));
    const uint32_t* w_up   = reinterpret_cast<const uint32_t*>(W_packed + (row + out_dim) * (IN_DIM / 2));
    float gate_scale = __ldg(scale + row);
    float up_scale   = __ldg(scale + row + out_dim);
    const float4* x4 = reinterpret_cast<const float4*>(s_normed);
    int n_iter = IN_DIM / 8;
    float gate_raw = 0.0f, up_raw = 0.0f;
    float gate_in = 0.0f, up_in = 0.0f;

    #pragma unroll 4
    for (int i = lane; i < n_iter; i += 32) {
        float4 xv = x4[i];
        __half2* xp = reinterpret_cast<__half2*>(&xv);
        float2 xf0 = __half22float2(xp[0]); float2 xf1 = __half22float2(xp[1]);
        float2 xf2 = __half22float2(xp[2]); float2 xf3 = __half22float2(xp[3]);
        float is = xf0.x + xf0.y + xf1.x + xf1.y + xf2.x + xf2.y + xf3.x + xf3.y;
        gate_in += is;
        up_in += is;

        uint32_t pg;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(pg) : "l"(w_gate + i));
        gate_raw += (float)((pg      ) & 0xF) * xf0.x
                  + (float)((pg >>  4) & 0xF) * xf0.y
                  + (float)((pg >>  8) & 0xF) * xf1.x
                  + (float)((pg >> 12) & 0xF) * xf1.y
                  + (float)((pg >> 16) & 0xF) * xf2.x
                  + (float)((pg >> 20) & 0xF) * xf2.y
                  + (float)((pg >> 24) & 0xF) * xf3.x
                  + (float)( pg >> 28        ) * xf3.y;

        uint32_t pu;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(pu) : "l"(w_up + i));
        up_raw   += (float)((pu      ) & 0xF) * xf0.x
                  + (float)((pu >>  4) & 0xF) * xf0.y
                  + (float)((pu >>  8) & 0xF) * xf1.x
                  + (float)((pu >> 12) & 0xF) * xf1.y
                  + (float)((pu >> 16) & 0xF) * xf2.x
                  + (float)((pu >> 20) & 0xF) * xf2.y
                  + (float)((pu >> 24) & 0xF) * xf3.x
                  + (float)( pu >> 28        ) * xf3.y;
    }

    float gate_corrected = gate_raw - 8.0f * gate_in;
    float up_corrected = up_raw - 8.0f * up_in;
    gate_corrected = warp_reduce_sum(gate_corrected);
    up_corrected = warp_reduce_sum(up_corrected);

    if (lane == 0) {
        float g = gate_corrected * gate_scale;
        float u = up_corrected * up_scale;
        float silu_val = g / (1.0f + expf(-g));
        y[row] = __float2half(silu_val * u);
    }
}

torch::Tensor gemv_addrmsnorm_silu_mul_int4_cuda(
    torch::Tensor W_packed,
    torch::Tensor hidden,       // [K] residual (updated in-place)
    torch::Tensor attn_out,     // [K] attention output
    torch::Tensor norm_weight,  // [K]
    torch::Tensor scale,        // [2*out_dim]
    float eps)
{
    int out_dim = W_packed.size(0) / 2;
    int K = W_packed.size(1) * 2;
    auto y = torch::empty({out_dim}, W_packed.options().dtype(torch::kFloat16));

    constexpr int WARPS_PER_BLOCK = 16;
    int block_size = WARPS_PER_BLOCK * 32;
    int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (K == 2048) {
        gemv_addrmsnorm_silu_mul_int4_kernel<WARPS_PER_BLOCK, 2048>
        <<<grid, block_size, 0, stream>>>(
            W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    } else {
        gemv_addrmsnorm_silu_mul_int4_kernel<WARPS_PER_BLOCK, 2048>
        <<<grid, block_size, 0, stream>>>(
            W_packed.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__half*>(hidden.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
            out_dim, eps);
    }
    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &gemv_coalesced_cuda, "合并访存GEMV (SM80)");
    m.def("gemv_async", &gemv_async_cuda, "cp.async软件流水线GEMV (SM80专用)");
    m.def("gemv_lmhead", &gemv_lmhead_cuda, "lm_head专用GEMV: L2 bypass + 大block (SM80)");
    m.def("gemv_rmsnorm", &gemv_rmsnorm_cuda,
          "融合RMS norm+GEMV 省一次kernel launch (SM80)");
    m.def("gemv_silu_mul", &gemv_silu_mul_coalesced_cuda,
          "融合GEMV+SiLU+Mul 合并访存 (SM80)");
    m.def("gemv_add", &gemv_add_cuda,
          "融合GEMV+残差加 合并访存 (SM80)");
    m.def("fused_qkv_norm_rope", &fused_qkv_norm_rope_cuda,
          "融合QKV拆分+Q/K RMS norm+RoPE CUDA C++ (decode用)");
    m.def("gemv_rmsnorm_int4", &gemv_rmsnorm_int4_cuda,
          "融合RMS norm+INT4 GEMV 省一次kernel launch");
    m.def("gemv_addrmsnorm_silu_mul_int4", &gemv_addrmsnorm_silu_mul_int4_cuda,
          "融合Add+RMS norm+INT4 GEMV SiLU+Mul 省一次kernel launch");
    m.def("write_kv_cache", &write_kv_cache_cuda,
          "融合K+V cache写入 (1 warp, decode用)");
    m.def("decode_attn", &decode_attn_cuda,
          "Custom decode attention for batch=1 GQA (SM80)");
    // INT8 weight-only GEMV
    m.def("gemv_int8", &gemv_int8_cuda, "INT8 weight-only GEMV (SM80)");
    m.def("gemv_lmhead_int8", &gemv_lmhead_int8_cuda, "INT8 lm_head GEMV (SM80)");
    m.def("gemv_lmhead_int8_argmax", &gemv_lmhead_int8_argmax_cuda, "INT8 lm_head GEMV+argmax fused (SM80)");
    m.def("gemv_lmhead_fp16_argmax", &gemv_lmhead_fp16_argmax_cuda, "FP16 lm_head GEMV+argmax fused (SM80)");
    m.def("gemv_lmhead_int8_refine", &gemv_lmhead_int8_refine_cuda, "INT8 lm_head shortlist + FP16 refine (SM80)");
    m.def("gemv_lmhead_int8_refine_topk", &gemv_lmhead_int8_refine_topk_cuda, "INT8 lm_head topK shortlist + FP16 refine (SM80)");
    m.def("gemv_silu_mul_int8", &gemv_silu_mul_int8_cuda, "INT8 GEMV+SiLU+Mul (SM80)");
    m.def("gemv_add_int8", &gemv_add_int8_cuda, "INT8 GEMV+残差加 (SM80)");
    m.def("gemv_add_both_int8", &gemv_add_both_int8_cuda, "INT8 GEMV+残差加+附加加 (SM80)");
    // INT4 weight-only GEMV (per-group, group_size=128)
    m.def("gemv_int4", &gemv_int4_cuda, "INT4 weight-only GEMV (SM80)");
    m.def("gemv_int4_w16", &gemv_int4_w16_cuda, "INT4 GEMV 16warp/block (SM80)");
    m.def("gemv_lmhead_int4", &gemv_lmhead_int4_cuda, "INT4 lm_head GEMV (SM80)");
    m.def("gemv_lmhead_argmax_int4", &gemv_lmhead_argmax_int4_cuda, "INT4 lm_head GEMV+argmax fused (SM80)");
    m.def("gemv_silu_mul_int4", &gemv_silu_mul_int4_cuda, "INT4 GEMV+SiLU+Mul (SM80)");
    m.def("gemv_add_int4", &gemv_add_int4_cuda, "INT4 GEMV+残差加 (SM80)");
    m.def("gemv_add_both_int4", &gemv_add_both_int4_cuda, "INT4 GEMV+残差加+附加加 (SM80)");
    // Batch INT4 GEMV (投机解码 batch verify)
    m.def("gemv_int4_batch", &gemv_int4_batch_cuda, "Batch INT4 GEMV: shared weight read (SM80)");
    m.def("gemv_silu_mul_int4_batch", &gemv_silu_mul_int4_batch_cuda, "Batch INT4 GEMV+SiLU+Mul: shared weight read (SM80)");
    m.def("gemv_lmhead_int4_batch", &gemv_lmhead_int4_batch_cuda, "Batch INT4 lm_head GEMV: shared weight read (SM80)");
}
