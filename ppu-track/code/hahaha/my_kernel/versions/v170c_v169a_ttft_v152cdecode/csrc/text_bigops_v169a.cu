#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float load_scalar(const void* ptr, bool is_half, int idx) {
    if (is_half) {
        return __half2float(reinterpret_cast<const half*>(ptr)[idx]);
    }
    return reinterpret_cast<const float*>(ptr)[idx];
}

__global__ void text_q_norm_rope_kernel_v169a(
    const half* __restrict__ qkv,
    const half* __restrict__ q_norm_weight,
    const void* __restrict__ cos,
    const void* __restrict__ sin,
    half* __restrict__ q_out,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int head_dim,
    int qkv_width,
    float eps,
    int cos_batch,
    bool cos_is_half
) {
    int d = threadIdx.x;
    int block = blockIdx.x;
    int h = block % num_q_heads;
    int tmp = block / num_q_heads;
    int s = tmp % seq_len;
    int b = tmp / seq_len;
    if (b >= batch_size || d >= head_dim) {
        return;
    }

    int token_base = (b * seq_len + s) * qkv_width;
    int q_base = token_base + h * head_dim;
    int idx = q_base + d;
    float q = __half2float(qkv[idx]);

    __shared__ float sumsq[128];
    sumsq[d] = q * q;
    __syncthreads();

    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            sumsq[d] += sumsq[d + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sumsq[0] / static_cast<float>(head_dim) + eps);
    int half_dim = head_dim / 2;
    int partner_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
    float q_partner = __half2float(qkv[q_base + partner_d]);
    float qn = q * inv_rms * __half2float(q_norm_weight[d]);
    float qpn = q_partner * inv_rms * __half2float(q_norm_weight[partner_d]);
    float q_rot = (d < half_dim) ? -qpn : qpn;

    int cos_b = (cos_batch == 1) ? 0 : b;
    int rope_idx = (cos_b * seq_len + s) * head_dim + d;
    float c = load_scalar(cos, cos_is_half, rope_idx);
    float sn = load_scalar(sin, cos_is_half, rope_idx);

    int out_idx = ((b * num_q_heads + h) * seq_len + s) * head_dim + d;
    q_out[out_idx] = __float2half_rn(qn * c + q_rot * sn);
}

__global__ void text_kv_norm_rope_kernel_v169a(
    const half* __restrict__ qkv,
    const half* __restrict__ k_norm_weight,
    const void* __restrict__ cos,
    const void* __restrict__ sin,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int qkv_width,
    float eps,
    int cos_batch,
    bool cos_is_half
) {
    int d = threadIdx.x;
    int block = blockIdx.x;
    int h = block % num_kv_heads;
    int tmp = block / num_kv_heads;
    int s = tmp % seq_len;
    int b = tmp / seq_len;
    if (b >= batch_size || d >= head_dim) {
        return;
    }

    int q_width = num_q_heads * head_dim;
    int kv_width = num_kv_heads * head_dim;
    int token_base = (b * seq_len + s) * qkv_width;
    int k_base = token_base + q_width + h * head_dim;
    int v_base = token_base + q_width + kv_width + h * head_dim;
    float k = __half2float(qkv[k_base + d]);

    __shared__ float sumsq[128];
    sumsq[d] = k * k;
    __syncthreads();

    for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            sumsq[d] += sumsq[d + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sumsq[0] / static_cast<float>(head_dim) + eps);
    int half_dim = head_dim / 2;
    int partner_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
    float k_partner = __half2float(qkv[k_base + partner_d]);
    float kn = k * inv_rms * __half2float(k_norm_weight[d]);
    float kpn = k_partner * inv_rms * __half2float(k_norm_weight[partner_d]);
    float k_rot = (d < half_dim) ? -kpn : kpn;

    int cos_b = (cos_batch == 1) ? 0 : b;
    int rope_idx = (cos_b * seq_len + s) * head_dim + d;
    float c = load_scalar(cos, cos_is_half, rope_idx);
    float sn = load_scalar(sin, cos_is_half, rope_idx);

    int out_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim + d;
    k_out[out_idx] = __float2half_rn(kn * c + k_rot * sn);
    v_out[out_idx] = qkv[v_base + d];
}

__global__ void text_swiglu_kernel_v169a(
    const half* __restrict__ gateup,
    half* __restrict__ intermediate,
    int total_tokens,
    int intermediate_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * intermediate_size;
    if (idx >= total) {
        return;
    }

    int token = idx / intermediate_size;
    int col = idx - token * intermediate_size;
    int base = token * (2 * intermediate_size);
    float gate = __half2float(gateup[base + col]);
    float up = __half2float(gateup[base + intermediate_size + col]);
    float sigmoid;
    if (gate >= 0.0f) {
        float z = expf(-gate);
        sigmoid = 1.0f / (1.0f + z);
    } else {
        float z = expf(gate);
        sigmoid = z / (1.0f + z);
    }
    float silu = gate * sigmoid;
    intermediate[idx] = __float2half_rn(silu * up);
}

extern "C" void launch_text_qkv_norm_rope_transpose_v169a(
    const void* qkv,
    const void* q_norm_weight,
    const void* k_norm_weight,
    const void* cos,
    const void* sin,
    void* q_out,
    void* k_out,
    void* v_out,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float eps,
    int cos_batch,
    int cos_is_half_int,
    cudaStream_t stream
) {
    int q_blocks = batch_size * seq_len * num_q_heads;
    int kv_blocks = batch_size * seq_len * num_kv_heads;
    bool cos_is_half = cos_is_half_int != 0;
    int qkv_width = (num_q_heads + 2 * num_kv_heads) * head_dim;

    text_q_norm_rope_kernel_v169a<<<q_blocks, head_dim, 0, stream>>>(
        static_cast<const half*>(qkv),
        static_cast<const half*>(q_norm_weight),
        cos,
        sin,
        static_cast<half*>(q_out),
        batch_size,
        seq_len,
        num_q_heads,
        head_dim,
        qkv_width,
        eps,
        cos_batch,
        cos_is_half
    );

    text_kv_norm_rope_kernel_v169a<<<kv_blocks, head_dim, 0, stream>>>(
        static_cast<const half*>(qkv),
        static_cast<const half*>(k_norm_weight),
        cos,
        sin,
        static_cast<half*>(k_out),
        static_cast<half*>(v_out),
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        qkv_width,
        eps,
        cos_batch,
        cos_is_half
    );
}

extern "C" void launch_text_swiglu_v169a(
    const void* gateup,
    void* intermediate,
    int batch_size,
    int seq_len,
    int intermediate_size,
    cudaStream_t stream
) {
    int total_tokens = batch_size * seq_len;
    int total = total_tokens * intermediate_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    text_swiglu_kernel_v169a<<<grid, block, 0, stream>>>(
        static_cast<const half*>(gateup),
        static_cast<half*>(intermediate),
        total_tokens,
        intermediate_size
    );
}
