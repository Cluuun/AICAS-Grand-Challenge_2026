// Custom split-K flash-decode attention for Qwen3-VL-2B greedy decode, a drop-in
// replacement for the prebuilt sage_decode_qkv_k_i8_v_bf16. Three kernels we own:
//   1. prologue: q/k RMSNorm + RoPE + per-head int8 quant; writes the current
//      token into the KV cache and emits q_i8/q_scale. Done ONCE (24 blocks).
//   2. partial: split-K online-softmax attention over key tiles via 4x dp4a. A
//      small tile oversubscribes the 108 SMs and hides the strided V latency.
//   3. reduce: online-combines the per-split partials into the head output.
// Preamble/RoPE/quant ported verbatim from the native sage reference so output
// matches the prebuilt path bit-closely.
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kNumQHeads = 16;
constexpr int kNumKVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = kHeadDim / 2;
constexpr int kGqa = kNumQHeads / kNumKVHeads;   // 2
constexpr int kThreads = 256;                    // 8 warps (partial)
constexpr int kPrologueThreads = 128;            // one head row per block
constexpr int kMaxTile = 64;
constexpr float kNegInf = -1e30f;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ signed char round_i8(float x) {
    x = fminf(127.0f, fmaxf(-127.0f, x));
    float r = x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
    return static_cast<signed char>(r);
}
__device__ __forceinline__ int pack_i8x4(signed char x0, signed char x1, signed char x2, signed char x3) {
    return (static_cast<unsigned char>(x0)) | (static_cast<unsigned char>(x1) << 8)
        | (static_cast<unsigned char>(x2) << 16) | (static_cast<unsigned char>(x3) << 24);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = threadIdx.x < warps ? sh[lane] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float block_max(float v) {
    __shared__ float sh[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = threadIdx.x < warps ? sh[lane] : kNegInf;
    if (warp == 0) for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float rope_value(
    const __nv_bfloat16* __restrict__ row, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos, const __nv_bfloat16* __restrict__ sin, int d, float rstd) {
    const int half = d & (kHalfHeadDim - 1);
    const bool right = d >= kHalfHeadDim;
    const float left = bf16_to_float(row[half]) * bf16_to_float(weight[half]) * rstd;
    const float right_v = bf16_to_float(row[half + kHalfHeadDim]) * bf16_to_float(weight[half + kHalfHeadDim]) * rstd;
    const float c = bf16_to_float(cos[half]);
    const float s = bf16_to_float(sin[half]);
    return right ? (right_v * c + left * s) : (left * c - right_v * s);
}

__device__ __forceinline__ float warp_sum_full(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max_full(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// 1. Prologue: norm + rope + int8 quant per q/k head; write current token to
//    cache; emit q_i8/q_scale. ONE WARP PER HEAD (32 lanes x 4 dims), warp-only
//    reductions -> no __syncthreads, few blocks, much cheaper than the per-block
//    variant. grid = ceil((Q+KV)/8) blocks of 8 warps.
__global__ void prologue_kernel(
    const __nv_bfloat16* __restrict__ flat_qkv,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    signed char* __restrict__ k_i8,
    __nv_bfloat16* __restrict__ v,
    float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    signed char* __restrict__ q_i8,
    float* __restrict__ q_scale,
    float q_eps,
    float k_eps) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= kNumQHeads + kNumKVHeads) return;
    const int lane = threadIdx.x & 31;
    const int q_size = kNumQHeads * kHeadDim;
    const int kv_size = kNumKVHeads * kHeadDim;
    const bool is_q = row < kNumQHeads;
    const int head = is_q ? row : (row - kNumQHeads);
    const __nv_bfloat16* row_ptr = is_q ? (flat_qkv + head * kHeadDim) : (flat_qkv + q_size + head * kHeadDim);
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;
    const float eps = is_q ? q_eps : k_eps;
    // Each lane owns dims lane, lane+32, lane+64, lane+96.
    float vals[4], sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { vals[i] = bf16_to_float(row_ptr[lane + 32 * i]); sq += vals[i] * vals[i]; }
    sq = warp_sum_full(sq);
    const float rstd = rsqrtf(sq / static_cast<float>(kHeadDim) + eps);
    float roped[4], amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { roped[i] = rope_value(row_ptr, weight, cos, sin, lane + 32 * i, rstd); amax = fmaxf(amax, fabsf(roped[i])); }
    amax = warp_max_full(amax);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    if (is_q) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) q_i8[head * kHeadDim + lane + 32 * i] = round_i8(roped[i] / scale);
        if (lane == 0) q_scale[head] = scale;
    } else {
        const int cache_pos = cache_seqlens[0];
        const int cur_base = (cache_pos * kNumKVHeads + head) * kHeadDim;
        const __nv_bfloat16* v_src = flat_qkv + q_size + kv_size + head * kHeadDim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + 32 * i;
            k_i8[cur_base + d] = round_i8(roped[i] / scale);
            v[cur_base + d] = v_src[d];
        }
        if (lane == 0) k_scale[cache_pos * kNumKVHeads + head] = scale;
    }
}

// 2. Partial: one block per (kv_head, split). Online softmax over its key tile
//    for both GQA query heads via 4x dp4a (current token already in cache).
__global__ void partial_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens_next,
    int max_splits, int tile, float softmax_scale,
    float* __restrict__ partial_max, float* __restrict__ partial_sum, float* __restrict__ partial_out) {
    __shared__ float scores0[kMaxTile];
    __shared__ float scores1[kMaxTile];
    __shared__ float sh[4];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_len = cache_seqlens_next[0];
    const int start = split * tile;
    if (start >= kv_len) return;
    const int end = min(start + tile, kv_len);
    const int tile_len = end - start;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const float qs0 = q_scale[qh0];
    const float qs1 = q_scale[qh1];
    const int* __restrict__ q0p = reinterpret_cast<const int*>(q_i8 + qh0 * kHeadDim);
    const int* __restrict__ q1p = reinterpret_cast<const int*>(q_i8 + qh1 * kHeadDim);

    for (int ti = warp; ti < tile_len; ti += 8) {
        const int t = start + ti;
        const int* __restrict__ kp = reinterpret_cast<const int*>(k_i8 + (t * kNumKVHeads + kvh) * kHeadDim);
        const int kv = kp[lane];
        const float d0 = warp_sum(static_cast<float>(__dp4a(q0p[lane], kv, 0)));
        const float d1 = warp_sum(static_cast<float>(__dp4a(q1p[lane], kv, 0)));
        if (lane == 0) {
            const float ks = k_scale[t * kNumKVHeads + kvh] * softmax_scale;
            scores0[ti] = d0 * qs0 * ks;
            scores1[ti] = d1 * qs1 * ks;
        }
    }
    __syncthreads();

    // tile softmax (warp0 -> head0, warp1 -> head1)
    if (warp < 2) {
        float* sc = warp == 0 ? scores0 : scores1;
        float m = kNegInf;
        for (int i = lane; i < tile_len; i += 32) m = fmaxf(m, sc[i]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
        m = __shfl_sync(0xffffffff, m, 0);
        float l = 0.0f;
        for (int i = lane; i < tile_len; i += 32) { const float e = __expf(sc[i] - m); sc[i] = e; l += e; }
        l = warp_sum(l);
        if (lane == 0) { sh[warp * 2] = m; sh[warp * 2 + 1] = l; }
    }
    __syncthreads();
    if (tid == 0) {
        partial_max[qh0 * max_splits + split] = sh[0];
        partial_sum[qh0 * max_splits + split] = sh[1];
        partial_max[qh1 * max_splits + split] = sh[2];
        partial_sum[qh1 * max_splits + split] = sh[3];
    }

    if (tid < kHeadDim) {
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int t = start + ti;
            const float value = bf16_to_float(v[(t * kNumKVHeads + kvh) * kHeadDim + tid]);
            acc0 += scores0[ti] * value;
            acc1 += scores1[ti] * value;
        }
        partial_out[(qh0 * max_splits + split) * kHeadDim + tid] = acc0;
        partial_out[(qh1 * max_splits + split) * kHeadDim + tid] = acc1;
    }
}

// 3. Reduce: online-combine splits -> output. grid (kNumQHeads, kHeadDim/32),
//    one warp per 32-dim group -> 4x the blocks of the per-head version, hiding
//    the strided partial_out latency. Max/denom recomputed per warp (cheap,
//    n_splits is small) but warp-only -> no __syncthreads.
__global__ void reduce_kernel(
    const int* __restrict__ cache_seqlens_next, int max_splits, int tile,
    const float* __restrict__ partial_max, const float* __restrict__ partial_sum,
    const float* __restrict__ partial_out, __nv_bfloat16* __restrict__ out) {
    const int qh = blockIdx.x;
    const int dgroup = blockIdx.y;          // 0..(kHeadDim/32 - 1)
    const int lane = threadIdx.x;           // 0..31
    const int d = dgroup * 32 + lane;
    const int kv_len = cache_seqlens_next[0];
    const int n_splits = (kv_len + tile - 1) / tile;
    const float* pm = partial_max + qh * max_splits;
    const float* ps = partial_sum + qh * max_splits;
    // global max + denom over splits (warp-parallel).
    float m = kNegInf;
    for (int s = lane; s < n_splits; s += 32) m = fmaxf(m, pm[s]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    float denom = 0.0f;
    for (int s = lane; s < n_splits; s += 32) denom += __expf(pm[s] - m) * ps[s];
    for (int o = 16; o > 0; o >>= 1) denom += __shfl_xor_sync(0xffffffff, denom, o);
    // weighted-sum for this lane's dim d across all splits.
    float acc = 0.0f;
    const float* po = partial_out + (qh * max_splits) * kHeadDim + d;
    for (int s = 0; s < n_splits; ++s) acc += __expf(pm[s] - m) * po[s * kHeadDim];
    out[qh * kHeadDim + d] = __float2bfloat16(acc / fmaxf(denom, 1e-20f));
}

}  // namespace

void fused_decode_attn_qkv(
    torch::Tensor flat_qkv, torch::Tensor q_weight, torch::Tensor k_weight,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor k_i8, torch::Tensor v, torch::Tensor k_scale,
    torch::Tensor cache_seqlens, torch::Tensor cache_seqlens_next,
    torch::Tensor q_i8, torch::Tensor q_scale,
    torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor partial_out,
    torch::Tensor out, double softmax_scale, double q_eps, double k_eps, int max_splits, int tile) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int prologue_blocks = (kNumQHeads + kNumKVHeads + 7) / 8;  // 8 warps/block
    prologue_kernel<<<prologue_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        reinterpret_cast<signed char*>(k_i8.data_ptr<int8_t>()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        k_scale.data_ptr<float>(), cache_seqlens.data_ptr<int>(),
        reinterpret_cast<signed char*>(q_i8.data_ptr<int8_t>()), q_scale.data_ptr<float>(),
        static_cast<float>(q_eps), static_cast<float>(k_eps));
    dim3 grid(kNumKVHeads, max_splits);
    partial_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const signed char*>(q_i8.data_ptr<int8_t>()), q_scale.data_ptr<float>(),
        reinterpret_cast<const signed char*>(k_i8.data_ptr<int8_t>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()), k_scale.data_ptr<float>(),
        cache_seqlens_next.data_ptr<int>(), max_splits, tile, static_cast<float>(softmax_scale),
        partial_max.data_ptr<float>(), partial_sum.data_ptr<float>(), partial_out.data_ptr<float>());
    dim3 rgrid(kNumQHeads, kHeadDim / 32);
    reduce_kernel<<<rgrid, 32, 0, stream>>>(
        cache_seqlens_next.data_ptr<int>(), max_splits, tile,
        partial_max.data_ptr<float>(), partial_sum.data_ptr<float>(), partial_out.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_decode_attn_qkv", &fused_decode_attn_qkv, "Custom split-K flash-decode attention (prologue+partial+reduce)");
}
