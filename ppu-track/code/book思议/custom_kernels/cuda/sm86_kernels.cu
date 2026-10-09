/*
 * SM86优化核心 给RTX 3090 / A800用的
 *
 * 包含: rms_norm, silu+mul, fused add+norm, fused QKV+norm+RoPE(prefill)
 * 编译: -O3 --use_fast_math -gencode=arch=compute_86,code=sm_86
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>
#include <cassert>

// ---------------------------------------------------------------------------
// 辅助工具函数
// ---------------------------------------------------------------------------

// half2乘法 把两个fp16打包成__half2然后逐元素乘
static __device__ __forceinline__ __half2 h2mul(__half2 a, __half2 b) {
#if __CUDA_ARCH__ >= 530
    return __hmul2(a, b);
#else
    float2 af = __half22float2(a);
    float2 bf = __half22float2(b);
    af.x *= bf.x;
    af.y *= bf.y;
    return __float22half2_rn(af);
#endif
}

// half2加法
static __device__ __forceinline__ __half2 h2add(__half2 a, __half2 b) {
#if __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 af = __half22float2(a);
    float2 bf = __half22float2(b);
    af.x += bf.x;
    af.y += bf.y;
    return __float22half2_rn(af);
#endif
}

// warp级别的求和规约 用shuffle搞 不需要shared memory
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 全warp allreduce 每个线程都能拿到总和
static __device__ __forceinline__ float warp_allreduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ---------------------------------------------------------------------------
// 1. RMS Norm — half2向量化读 warp shuffle做规约
// ---------------------------------------------------------------------------
// H必须是偶数(2048, 1024, 128都是偶数 没问题)
// 每个线程处理 H/(2*blockDim.x) 个half2对
// 一个block处理一行(行太小的话就多行 但我们这里一行一个block)

template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
rms_norm_kernel(
    const __half* __restrict__ X,   // [n_rows, H]
    const __half* __restrict__ W,   // [H]
    __half* __restrict__ OUT,       // [n_rows, H]
    int n_rows,
    int H,
    float eps)
{
    const int half_H = H / 2;  // 每行有多少个half2
    const int row = blockIdx.x;

    if (row >= n_rows) return;

    const __half2* x2 = reinterpret_cast<const __half2*>(X + row * H);
    const __half2* w2 = reinterpret_cast<const __half2*>(W);
    __half2* out2 = reinterpret_cast<__half2*>(OUT + row * H);

    // 每个线程处理的half2数量
    const int stride = half_H / BLOCK_SIZE;
    const int tid = threadIdx.x;

    // 局部平方和
    float sum_sq = 0.0f;

    for (int i = 0; i < stride; i++) {
        int idx = tid + i * BLOCK_SIZE;
        __half2 x_val = __ldg(x2 + idx);  // 走只读缓存 带宽友好
        float2 xf = __half22float2(x_val);
        sum_sq += xf.x * xf.x + xf.y * xf.y;
    }

    // warp内规约求和
    sum_sq = warp_reduce_sum(sum_sq);

    // 跨warp规约用shared memory 然后广播inv_var给所有线程
    __shared__ float smem[33];  // [0..31]存各warp的和 [32]存inv_var
    int lane = tid & 31;
    int wid = tid >> 5;

    if (BLOCK_SIZE > 32) {
        if (lane == 0) smem[wid] = sum_sq;
        __syncthreads();
        if (wid == 0) {
            sum_sq = (tid < (BLOCK_SIZE >> 5)) ? smem[tid] : 0.0f;
            sum_sq = warp_reduce_sum(sum_sq);
        }
    }

    // 线程0算好inv_var然后广播给所有人
    if (tid == 0) smem[32] = __frsqrt_rn(sum_sq / (float)H + eps);
    __syncthreads();
    float inv_var = smem[32];

    // 写输出: out = x * inv_var * w
    for (int i = 0; i < stride; i++) {
        int idx = tid + i * BLOCK_SIZE;
        __half2 x_val = __ldg(x2 + idx);
        float2 xf = __half22float2(x_val);
        __half2 w_val = __ldg(w2 + idx);
        float2 wf = __half22float2(w_val);
        xf.x = xf.x * inv_var * wf.x;
        xf.y = xf.y * inv_var * wf.y;
        out2[idx] = __float22half2_rn(xf);
    }
}

// ---------------------------------------------------------------------------
// 2. SiLU + Mul — half2向量化 俩op融合成一个
// ---------------------------------------------------------------------------
// gate_up: [n_rows, 2*D] (前半是gate 后半是up)
// output:  [n_rows, D]

template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
silu_and_mul_kernel(
    const __half* __restrict__ GATE_UP,  // [n_rows, 2*D]
    __half* __restrict__ OUT,             // [n_rows, D]
    int n_rows,
    int half_D)  // D (half2对的数量 = D/2)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const int D = half_D;
    const int half2_D = D / 2;

    const __half2* gu2 = reinterpret_cast<const __half2*>(GATE_UP + row * 2 * D);
    const __half2* up2 = reinterpret_cast<const __half2*>(GATE_UP + row * 2 * D + D);
    __half2* out2 = reinterpret_cast<__half2*>(OUT + row * D);

    const int stride = half2_D / BLOCK_SIZE;
    const int tid = threadIdx.x;

    for (int i = 0; i < stride; i++) {
        int idx = tid + i * BLOCK_SIZE;
        __half2 g = __ldg(gu2 + idx);
        __half2 u = __ldg(up2 + idx);

        float2 gf = __half22float2(g);
        float2 uf = __half22float2(u);

        // silu(g) = g * sigmoid(g) = g / (1 + exp(-g)) 再乘上up
        gf.x = gf.x / (1.0f + expf(-gf.x)) * uf.x;
        gf.y = gf.y / (1.0f + expf(-gf.y)) * uf.y;

        out2[idx] = __float22half2_rn(gf);
    }
}

// ---------------------------------------------------------------------------
// 3. Fused Add + RMS Norm — 残差加和归一化一步搞定
// ---------------------------------------------------------------------------

template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_add_rms_norm_kernel(
    const __half* __restrict__ INPUT,     // [n_rows, H]
    const __half* __restrict__ RESIDUAL,  // [n_rows, H]
    const __half* __restrict__ W,         // [H]
    __half* __restrict__ OUT,             // [n_rows, H]  (归一化后的输出)
    __half* __restrict__ RES_OUT,         // [n_rows, H]  (更新后的残差)
    int n_rows,
    int H,
    float eps)
{
    const int half_H = H / 2;
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const __half2* in2 = reinterpret_cast<const __half2*>(INPUT + row * H);
    const __half2* res2 = reinterpret_cast<const __half2*>(RESIDUAL + row * H);
    const __half2* w2 = reinterpret_cast<const __half2*>(W);
    __half2* out2 = reinterpret_cast<__half2*>(OUT + row * H);
    __half2* resout2 = reinterpret_cast<__half2*>(RES_OUT + row * H);

    const int stride = half_H / BLOCK_SIZE;
    const int tid = threadIdx.x;

    float sum_sq = 0.0f;

    // 第一遍: 加残差 + 算方差
    // 加完的结果存到res_out
    for (int i = 0; i < stride; i++) {
        int idx = tid + i * BLOCK_SIZE;
        __half2 i_val = __ldg(in2 + idx);
        __half2 r_val = __ldg(res2 + idx);

        float2 iv = __half22float2(i_val);
        float2 rv = __half22float2(r_val);
        rv.x += iv.x;
        rv.y += iv.y;

        sum_sq += rv.x * rv.x + rv.y * rv.y;

        // 存更新后的残差
        resout2[idx] = __float22half2_rn(rv);
    }

    // 规约 + 广播inv_var 跟rms_norm_kernel一样的套路
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float smem[33];  // [0..31]存各warp的和 [32]存inv_var
    int lane = tid & 31;
    int wid = tid >> 5;

    if (BLOCK_SIZE > 32) {
        if (lane == 0) smem[wid] = sum_sq;
        __syncthreads();
        if (wid == 0) {
            sum_sq = (tid < (BLOCK_SIZE >> 5)) ? smem[tid] : 0.0f;
            sum_sq = warp_reduce_sum(sum_sq);
        }
    }

    // 线程0算好inv_var然后广播给所有人
    if (tid == 0) smem[32] = __frsqrt_rn(sum_sq / (float)H + eps);
    __syncthreads();
    float inv_var = smem[32];

    // 第二遍: 归一化 + 乘权重
    for (int i = 0; i < stride; i++) {
        int idx = tid + i * BLOCK_SIZE;
        __half2 w_val = __ldg(w2 + idx);
        float2 wf = __half22float2(w_val);

        // 重新读刚才写的残差
        float2 rv = __half22float2(resout2[idx]);

        rv.x = rv.x * inv_var * wf.x;
        rv.y = rv.y * inv_var * wf.y;
        out2[idx] = __float22half2_rn(rv);
    }
}

// ---------------------------------------------------------------------------
// 4. Fused QKV拆分 + Q/K的RMS Norm + RoPE (prefill阶段)
// ---------------------------------------------------------------------------
// 输入: qkv [seq_len, q_dim + 2*kv_dim]
// 输出: q [seq_len, n_q_heads, head_dim], k, v (同样布局)
//
// 这是最复杂的kernel 每个(头, 位置)对应一个线程
// half2用来做半维RoPE旋转

template <int HEAD_DIM>
__global__ void __launch_bounds__(256)
fused_qkv_norm_rope_prefill_kernel(
    const __half* __restrict__ QKV,       // [seq_len, total_dim]
    const __half* __restrict__ Q_W,       // [head_dim]
    const __half* __restrict__ K_W,       // [head_dim]
    const __half* __restrict__ COS,       // [seq_len, head_dim]
    const __half* __restrict__ SIN,       // [seq_len, head_dim]
    __half* __restrict__ Q_OUT,           // [n_q_heads, seq_len, head_dim]
    __half* __restrict__ K_OUT,           // [n_kv_heads, seq_len, head_dim]
    __half* __restrict__ V_OUT,           // [n_kv_heads, seq_len, head_dim]
    int seq_len,
    int q_dim,
    int kv_dim,
    int n_q_heads,
    int n_kv_heads,
    int total_dim,
    float q_eps,
    float k_eps)
{
    // Grid: (max(n_q_heads, n_kv_heads) * seq_len,)
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = max(n_q_heads, n_kv_heads) * seq_len;
    if (pid >= total_threads) return;

    int head = pid / seq_len;
    int pos = pid % seq_len;
    constexpr int HALF_DIM = HEAD_DIM / 2;

    // 用half2加载这个位置的cos/sin
    const __half2* cos2 = reinterpret_cast<const __half2*>(COS + pos * HEAD_DIM);
    const __half2* sin2 = reinterpret_cast<const __half2*>(SIN + pos * HEAD_DIM);

    // 处理Q头
    if (head < n_q_heads) {
        const __half* q_in = QKV + pos * total_dim + head * HEAD_DIM;
        __half* q_out = Q_OUT + head * seq_len * HEAD_DIM + pos * HEAD_DIM;

        // 加载Q的权重
        const __half2* qw2 = reinterpret_cast<const __half2*>(Q_W);

        // 算方差 + 加载数据
        float sum_sq = 0.0f;
        float q_data[HEAD_DIM];  // 寄存器缓存

        #pragma unroll
        for (int j = 0; j < HALF_DIM; j++) {
            __half2 val = __ldg(reinterpret_cast<const __half2*>(q_in) + j);
            float2 vf = __half22float2(val);
            q_data[j * 2] = vf.x;
            q_data[j * 2 + 1] = vf.y;
            sum_sq += vf.x * vf.x + vf.y * vf.y;
        }

        float inv_var = __frsqrt_rn(sum_sq / (float)HEAD_DIM + q_eps);

        // 归一化 + 上RoPE
        #pragma unroll
        for (int j = 0; j < HALF_DIM; j++) {
            __half2 w = __ldg(qw2 + j);
            float2 wf = __half22float2(w);

            float q0 = q_data[j * 2] * inv_var * wf.x;
            float q1 = q_data[j * 2 + 1] * inv_var * wf.y;

            // RoPE旋转: 前半 = q0*cos - q1*sin, 后半 = q1*cos + q0*sin
            __half2 c = __ldg(cos2 + j);
            __half2 s = __ldg(sin2 + j);
            float2 cf = __half22float2(c);
            float2 sf = __half22float2(s);

            float out0 = q0 * cf.x - q1 * sf.x;
            float out1 = q1 * cf.y + q0 * sf.y;

            reinterpret_cast<__half2*>(q_out)[j] = __float22half2_rn(
                make_float2(out0, out1));
        }
    }

    // 处理K头 + 拷贝V
    if (head < n_kv_heads) {
        const __half* k_in = QKV + pos * total_dim + q_dim + head * HEAD_DIM;
        __half* k_out = K_OUT + head * seq_len * HEAD_DIM + pos * HEAD_DIM;

        const __half2* kw2 = reinterpret_cast<const __half2*>(K_W);

        float sum_sq = 0.0f;
        float k_data[HEAD_DIM];

        #pragma unroll
        for (int j = 0; j < HALF_DIM; j++) {
            __half2 val = __ldg(reinterpret_cast<const __half2*>(k_in) + j);
            float2 vf = __half22float2(val);
            k_data[j * 2] = vf.x;
            k_data[j * 2 + 1] = vf.y;
            sum_sq += vf.x * vf.x + vf.y * vf.y;
        }

        float inv_var = __frsqrt_rn(sum_sq / (float)HEAD_DIM + k_eps);

        #pragma unroll
        for (int j = 0; j < HALF_DIM; j++) {
            __half2 w = __ldg(kw2 + j);
            float2 wf = __half22float2(w);

            float k0 = k_data[j * 2] * inv_var * wf.x;
            float k1 = k_data[j * 2 + 1] * inv_var * wf.y;

            __half2 c = __ldg(cos2 + j);
            __half2 s = __ldg(sin2 + j);
            float2 cf = __half22float2(c);
            float2 sf = __half22float2(s);

            float out0 = k0 * cf.x - k1 * sf.x;
            float out1 = k1 * cf.y + k0 * sf.y;

            reinterpret_cast<__half2*>(k_out)[j] = __float22half2_rn(
                make_float2(out0, out1));
        }

        // 拷贝V 不做归一化 也不上RoPE 直接搬过来就行
        const __half* v_in = QKV + pos * total_dim + q_dim + kv_dim + head * HEAD_DIM;
        __half* v_out = V_OUT + head * seq_len * HEAD_DIM + pos * HEAD_DIM;

        #pragma unroll
        for (int j = 0; j < HALF_DIM; j++) {
            __half2 val = __ldg(reinterpret_cast<const __half2*>(v_in) + j);
            reinterpret_cast<__half2*>(v_out)[j] = val;
        }
    }
}


// ===========================================================================
// C++接口函数 Python通过pybind11调用这些
// ===========================================================================

torch::Tensor rms_norm_cuda(
    torch::Tensor input,    // [..., H]
    torch::Tensor weight,   // [H]
    double eps)
{
    auto orig_shape = input.sizes();
    int H = input.size(-1);
    int n_rows = input.numel() / H;

    auto x = input.reshape({n_rows, H});
    auto out = torch::empty_like(x);

    int half_H = H / 2;

    // 根据每行的half2数量选block size
    // 让每个线程至少处理2个half2 利用率才够
    int block_size;
    if (half_H <= 64) block_size = 32;       // 1个warp 每线程2个元素
    else if (half_H <= 256) block_size = 128; // 4个warp
    else if (half_H <= 512) block_size = 256; // 8个warp
    else block_size = 512;                    // 16个warp

    int grid = n_rows;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    #define LAUNCH_RMS(BS) \
        rms_norm_kernel<BS><<<grid, BS, 0, stream>>>( \
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), \
            n_rows, H, (float)eps)

    switch (block_size) {
        case 32:  LAUNCH_RMS(32); break;
        case 128: LAUNCH_RMS(128); break;
        case 256: LAUNCH_RMS(256); break;
        case 512: LAUNCH_RMS(512); break;
    }
    #undef LAUNCH_RMS

    return out.reshape(orig_shape);
}


torch::Tensor silu_and_mul_cuda(
    torch::Tensor gate_up)   // [..., 2*D]
{
    auto orig_shape = gate_up.sizes();
    int D = gate_up.size(-1) / 2;
    int n_rows = gate_up.numel() / (2 * D);

    auto gu = gate_up.reshape({n_rows, 2 * D});
    auto out = torch::empty({n_rows, D}, gu.options());

    int half2_D = D / 2;

    int block_size;
    if (half2_D <= 64) block_size = 32;
    else if (half2_D <= 256) block_size = 128;
    else if (half2_D <= 512) block_size = 256;
    else block_size = 512;

    int grid = n_rows;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    #define LAUNCH_SILU(BS) \
        silu_and_mul_kernel<BS><<<grid, BS, 0, stream>>>( \
            reinterpret_cast<const __half*>(gu.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), \
            n_rows, D)

    switch (block_size) {
        case 32:  LAUNCH_SILU(32); break;
        case 128: LAUNCH_SILU(128); break;
        case 256: LAUNCH_SILU(256); break;
        case 512: LAUNCH_SILU(512); break;
    }
    #undef LAUNCH_SILU

    auto new_shape = orig_shape.vec();
    new_shape.back() = D;
    return out.reshape(new_shape);
}


std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps)
{
    auto orig_shape = input.sizes();
    int H = input.size(-1);
    int n_rows = input.numel() / H;

    auto inp = input.reshape({n_rows, H});
    auto res = residual.reshape({n_rows, H});

    auto out = torch::empty_like(inp);
    auto res_out = torch::empty_like(res);

    int half_H = H / 2;
    int block_size;
    if (half_H <= 64) block_size = 32;
    else if (half_H <= 256) block_size = 128;
    else if (half_H <= 512) block_size = 256;
    else block_size = 512;

    int grid = n_rows;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    #define LAUNCH_FARN(BS) \
        fused_add_rms_norm_kernel<BS><<<grid, BS, 0, stream>>>( \
            reinterpret_cast<const __half*>(inp.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(res.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(res_out.data_ptr<at::Half>()), \
            n_rows, H, (float)eps)

    switch (block_size) {
        case 32:  LAUNCH_FARN(32); break;
        case 128: LAUNCH_FARN(128); break;
        case 256: LAUNCH_FARN(256); break;
        case 512: LAUNCH_FARN(512); break;
    }
    #undef LAUNCH_FARN

    return std::make_tuple(out.reshape(orig_shape), res_out.reshape(orig_shape));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_norm_rope_prefill_cuda(
    torch::Tensor qkv,        // [batch, seq_len, q_dim + 2*kv_dim]
    torch::Tensor q_weight,   // [head_dim]
    torch::Tensor k_weight,   // [head_dim]
    torch::Tensor cos,        // [seq_len, head_dim]
    torch::Tensor sin,        // [seq_len, head_dim]
    int64_t q_dim,
    int64_t kv_dim,
    int64_t head_dim,
    double q_eps,
    double k_eps)
{
    int n_q_heads = q_dim / head_dim;
    int n_kv_heads = kv_dim / head_dim;
    int batch = qkv.size(0);
    int seq_len = qkv.size(1);
    int total_dim = q_dim + 2 * kv_dim;

    // 把batch压进seq里 我们这边batch都是1 简单处理
    auto qkv_flat = qkv.reshape({seq_len, total_dim});

    auto q_out = torch::empty({n_q_heads, seq_len, head_dim}, qkv.options());
    auto k_out = torch::empty({n_kv_heads, seq_len, head_dim}, qkv.options());
    auto v_out = torch::empty({n_kv_heads, seq_len, head_dim}, qkv.options());

    int total_threads = max(n_q_heads, n_kv_heads) * seq_len;
    int block_size = 256;
    int grid = (total_threads + block_size - 1) / block_size;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    #define LAUNCH_QKV(HD) \
        fused_qkv_norm_rope_prefill_kernel<HD><<<grid, block_size, 0, stream>>>( \
            reinterpret_cast<const __half*>(qkv_flat.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(q_weight.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(k_weight.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(cos.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(sin.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(q_out.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(k_out.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(v_out.data_ptr<at::Half>()), \
            seq_len, q_dim, kv_dim, n_q_heads, n_kv_heads, total_dim, \
            (float)q_eps, (float)k_eps)

    switch (head_dim) {
        case 64:  LAUNCH_QKV(64); break;
        case 128: LAUNCH_QKV(128); break;
        default:
            // 其他head_dim没优化 直接报错
            AT_ERROR("fused_qkv_norm_rope_prefill_cuda: unsupported head_dim ", head_dim);
    }
    #undef LAUNCH_QKV

    // 重排成 [batch, seq, heads, head_dim] 返回
    return std::make_tuple(
        q_out.permute({1, 0, 2}).reshape({batch, seq_len, n_q_heads, head_dim}),
        k_out.permute({1, 0, 2}).reshape({batch, seq_len, n_kv_heads, head_dim}),
        v_out.permute({1, 0, 2}).reshape({batch, seq_len, n_kv_heads, head_dim})
    );
}


// pybind11模块注册
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm_cuda, "RMS Norm (SM86优化版)");
    m.def("silu_and_mul", &silu_and_mul_cuda, "SiLU + Mul (SM86优化版)");
    m.def("fused_add_rms_norm", &fused_add_rms_norm_cuda, "Fused Add + RMS Norm (SM86优化版)");
    m.def("fused_qkv_norm_rope_prefill", &fused_qkv_norm_rope_prefill_cuda,
          "Fused QKV + Norm + RoPE Prefill (SM86优化版)");
}
