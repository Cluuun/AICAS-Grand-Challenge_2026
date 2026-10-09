#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void vision_qkv_rope_transpose_kernel_v167a(
    const half* __restrict__ qkv,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total) {
        return;
    }

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int h = tmp % num_heads;
    int s = tmp / num_heads;

    int hidden = num_heads * head_dim;
    int base = s * (3 * hidden) + h * head_dim + d;
    int half_dim = head_dim / 2;
    int partner_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
    int partner_base = s * (3 * hidden) + h * head_dim + partner_d;

    float c = cos[s * head_dim + d];
    float sn = sin[s * head_dim + d];

    float q = __half2float(qkv[base]);
    float q_partner = __half2float(qkv[partner_base]);
    float q_rot = (d < half_dim) ? -q_partner : q_partner;

    int k_base = base + hidden;
    int k_partner_base = partner_base + hidden;
    float k = __half2float(qkv[k_base]);
    float k_partner = __half2float(qkv[k_partner_base]);
    float k_rot = (d < half_dim) ? -k_partner : k_partner;

    int out_idx = (h * seq_len + s) * head_dim + d;
    q_out[out_idx] = __float2half_rn(q * c + q_rot * sn);
    k_out[out_idx] = __float2half_rn(k * c + k_rot * sn);
    v_out[out_idx] = qkv[base + 2 * hidden];
}

extern "C" void launch_vision_qkv_rope_transpose_v167a(
    const void* qkv,
    const void* cos,
    const void* sin,
    void* q_out,
    void* k_out,
    void* v_out,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    int total = seq_len * num_heads * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    vision_qkv_rope_transpose_kernel_v167a<<<grid, block, 0, stream>>>(
        static_cast<const half*>(qkv),
        static_cast<const float*>(cos),
        static_cast<const float*>(sin),
        static_cast<half*>(q_out),
        static_cast<half*>(k_out),
        static_cast<half*>(v_out),
        seq_len,
        num_heads,
        head_dim
    );
}
