/*
 * 独立CUDA核心 拿来给Qwen3-VL-2B做优化
 *
 * 包含: rms_norm, fused_add_rms_norm, silu_and_mul
 * 只依赖cuda toolkit和pytorch 不需要vllm
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// 1. RMS Norm 核心
// ---------------------------------------------------------------------------
// 算法: out = weight * input * rsqrt(mean(input^2) + eps)
// 用FP32累加 保证数值稳定
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int hidden_size)
{
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float s_inv_var;

    const int tid = threadIdx.x;
    float variance = 0.0f;

    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
        float x = static_cast<float>(input[blockIdx.x * hidden_size + idx]);
        variance += x * x;
    }

    variance = BlockReduce(temp_storage).Sum(variance);

    if (tid == 0) {
        s_inv_var = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
        float x = static_cast<float>(input[blockIdx.x * hidden_size + idx]);
        float w = static_cast<float>(weight[idx]);
        out[blockIdx.x * hidden_size + idx] = static_cast<scalar_t>(x * s_inv_var * w);
    }
}

// ---------------------------------------------------------------------------
// 2. 融合 Add + RMS Norm 核心
// ---------------------------------------------------------------------------
// 计算:
//   residual[i] += input[i]
//   input[i] = weight[i] * residual[i] * rsqrt(mean(residual^2) + eps)
// input会被归一化结果覆盖 residual会被累加和覆盖
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,       // [..., hidden_size] -> 归一化输出
    const int64_t input_stride,
    scalar_t* __restrict__ residual,    // [..., hidden_size] -> 累加和
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int hidden_size)
{
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float s_inv_var;

    const int tid = threadIdx.x;
    float variance = 0.0f;

    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
        float val = static_cast<float>(input[blockIdx.x * input_stride + idx]);
        float res = static_cast<float>(residual[blockIdx.x * hidden_size + idx]);
        val += res;
        variance += val * val;
        residual[blockIdx.x * hidden_size + idx] = static_cast<scalar_t>(val);
    }

    variance = BlockReduce(temp_storage).Sum(variance);

    if (tid == 0) {
        s_inv_var = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    for (int idx = tid; idx < hidden_size; idx += blockDim.x) {
        float val = static_cast<float>(residual[blockIdx.x * hidden_size + idx]);
        float w = static_cast<float>(weight[idx]);
        input[blockIdx.x * input_stride + idx] = static_cast<scalar_t>(val * s_inv_var * w);
    }
}

// ---------------------------------------------------------------------------
// 3. SiLU 和 Mul 融合核心
// ---------------------------------------------------------------------------
// 计算: out[i] = silu(x[i]) * y[i]  其中 x = input[:, :d], y = input[:, d:]
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// 用FP32算 保证精度
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int d)
{
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
        float x = static_cast<float>(input[blockIdx.x * 2 * d + idx]);
        float y = static_cast<float>(input[blockIdx.x * 2 * d + d + idx]);
        float silu_x = x / (1.0f + expf(-x));
        out[blockIdx.x * d + idx] = static_cast<scalar_t>(silu_x * y);
    }
}

// ---------------------------------------------------------------------------
// C++接口函数
// ---------------------------------------------------------------------------

void rms_norm_cuda(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double epsilon)
{
    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
                "input and weight must have the same dtype");

    dim3 grid(num_tokens);
    dim3 block(1024);

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm", ([&] {
        rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            static_cast<float>(epsilon),
            hidden_size);
    }));
}

void fused_add_rms_norm_cuda(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double epsilon)
{
    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;
    int64_t input_stride = input.stride(-2);

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
                "input and residual must have the same dtype");

    dim3 grid(num_tokens);
    dim3 block(1024);

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_add_rms_norm", ([&] {
        fused_add_rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            input_stride,
            residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            static_cast<float>(epsilon),
            hidden_size);
    }));
}

void silu_and_mul_cuda(
    torch::Tensor& out,
    const torch::Tensor& input)
{
    int d = input.size(-1) / 2;
    int num_tokens = input.numel() / (2 * d);

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    dim3 grid(num_tokens);
    dim3 block(min(d, 1024));

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "silu_and_mul", ([&] {
        silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            d);
    }));
}

// ---------------------------------------------------------------------------
// 6. Sign-Shift Perturbation (anti-degenerate for MLP skip)
// ---------------------------------------------------------------------------
// out = input + eps * sign(input)
// Fused into single kernel: avoids Python overhead + separate sign/add ops
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void perturb_sign_shift_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const float eps,
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = static_cast<float>(input[idx]);
    float s = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    out[idx] = static_cast<scalar_t>(x + eps * s);
}

void perturb_sign_shift_cuda(
    torch::Tensor out,
    const torch::Tensor input,
    double eps)
{
    int n = input.numel();
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");

    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "perturb_sign_shift", ([&] {
        perturb_sign_shift_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            static_cast<float>(eps),
            n);
    }));
}

// ---------------------------------------------------------------------------
// 7. Sign+Norm Perturbation (anti-degenerate, scale-aware)
// ---------------------------------------------------------------------------
// out = input + eps * rms_norm(input) * sign(input)
// Fuses L2-norm reduction + sign + perturbation into one kernel
// One block per token (row), reduces over hidden_size dimension
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void perturb_sign_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const float eps,
    const int hidden_size)
{
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float s_scale;

    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    float sum_sq = 0.0f;

    // Pass 1: compute L2 norm
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(input[row * hidden_size + i]);
        sum_sq += x * x;
    }
    sum_sq = BlockReduce(temp_storage).Sum(sum_sq);

    if (tid == 0) {
        s_scale = eps * sqrtf(sum_sq / hidden_size);
    }
    __syncthreads();

    // Pass 2: apply perturbation
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(input[row * hidden_size + i]);
        float s = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        out[row * hidden_size + i] = static_cast<scalar_t>(x + s_scale * s);
    }
}

void perturb_sign_norm_cuda(
    torch::Tensor out,
    const torch::Tensor input,
    double eps)
{
    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");

    dim3 grid(num_tokens);
    dim3 block(1024);

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "perturb_sign_norm", ([&] {
        perturb_sign_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            static_cast<float>(eps),
            hidden_size);
    }));
}

// ---------------------------------------------------------------------------
// 8. GPU Repeat Detector — detect repeated n-gram suffixes on GPU
// ---------------------------------------------------------------------------
// Checks the last N tokens for:
//   (a) a run of >= 8 identical trailing tokens
//   (b) a repeated n-gram (width 3..48) repeated >= 3 times at the tail
//   (c) a degenerate tail (<= 6 unique tokens in last 32 positions)
// Writes trim_length (1..n) to output. If no repeat, output = n.
// Single-thread kernel: fast enough for n <= 192 tokens, avoids CPU sync.
// ---------------------------------------------------------------------------

__global__ void detect_repeat_kernel(
    const int64_t* __restrict__ tokens,
    int n,
    int* __restrict__ out_trim)
{
    // (a) Trailing run of identical tokens
    {
        int run = 1;
        int64_t last = tokens[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            if (tokens[i] != last) break;
            ++run;
        }
        if (run >= 8) {
            *out_trim = n - run + 2;
            if (*out_trim < 1) *out_trim = 1;
            return;
        }
    }

    // (b) Repeated n-gram at tail (width 3..48, need >= 3 repeats)
    {
        int max_w = min(48, n / 3);
        for (int w = 3; w <= max_w; ++w) {
            int reps = 1;
            int pos = n - w;
            while (pos - w >= 0) {
                bool match = true;
                for (int j = 0; j < w; ++j) {
                    if (tokens[pos - w + j] != tokens[n - w + j]) {
                        match = false;
                        break;
                    }
                }
                if (!match) break;
                ++reps;
                pos -= w;
                if (reps >= 3) {
                    int trim = n - w * (reps - 1);
                    *out_trim = trim > 0 ? trim : 1;
                    return;
                }
            }
        }
    }

    // (c) Degenerate tail: <= 6 unique tokens in last 32 positions
    if (n >= 24) {
        int dw = min(n, 32);
        int unique = 0;
        for (int i = n - dw; i < n; ++i) {
            bool first = true;
            for (int j = n - dw; j < i; ++j) {
                if (tokens[j] == tokens[i]) { first = false; break; }
            }
            if (first) ++unique;
        }
        if (unique <= 6) {
            int cv = 0;
            int64_t cvset[16];
            for (int i = n - 16; i < n; ++i) {
                bool found = false;
                for (int k = 0; k < cv; ++k) {
                    if (cvset[k] == tokens[i]) { found = true; break; }
                }
                if (!found && cv < 16) cvset[cv++] = tokens[i];
            }
            int lo = n - dw - 1;
            int hi = (n - 128 > 0) ? n - 128 : 0;
            for (int i = lo; i >= hi; --i) {
                bool in_cv = false;
                for (int k = 0; k < cv; ++k) {
                    if (cvset[k] == tokens[i]) { in_cv = true; break; }
                }
                if (!in_cv) {
                    *out_trim = i + 1;
                    if (*out_trim < 1) *out_trim = 1;
                    return;
                }
            }
            *out_trim = n - dw;
            if (*out_trim < 1) *out_trim = 1;
            return;
        }
    }

    *out_trim = n;
}

void detect_repeat_cuda(
    torch::Tensor tokens,
    torch::Tensor out_trim)
{
    int n = tokens.numel();
    TORCH_CHECK(tokens.is_cuda(), "tokens must be a CUDA tensor");
    TORCH_CHECK(out_trim.is_cuda(), "out_trim must be a CUDA tensor");
    TORCH_CHECK(tokens.scalar_type() == torch::kInt64, "tokens must be int64");
    TORCH_CHECK(out_trim.scalar_type() == torch::kInt, "out_trim must be int32");

    auto stream = at::cuda::getCurrentCUDAStream();
    detect_repeat_kernel<<<1, 1, 0, stream>>>(
        tokens.data_ptr<int64_t>(),
        n,
        out_trim.data_ptr<int>());
}

// ---------------------------------------------------------------------------
// 9. GPU EOS check — check if token is in EOS set, result stays on GPU
// ---------------------------------------------------------------------------
// Compares token[0,0] against eos_set[0..n_eos-1], writes 1 to flag if match
// ---------------------------------------------------------------------------

__global__ void eos_check_kernel(
    const int64_t* __restrict__ token,
    const int64_t* __restrict__ eos_set,
    int n_eos,
    int* __restrict__ flag)
{
    int64_t t = token[0];
    for (int i = threadIdx.x; i < n_eos; i += blockDim.x) {
        if (eos_set[i] == t) {
            *flag = 1;
            return;
        }
    }
}

void eos_check_cuda(
    torch::Tensor token,
    torch::Tensor eos_set,
    torch::Tensor flag)
{
    int n_eos = eos_set.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    eos_check_kernel<<<1, min(n_eos, 1024), 0, stream>>>(
        token.data_ptr<int64_t>(),
        eos_set.data_ptr<int64_t>(),
        n_eos,
        flag.data_ptr<int>());
}

// ---------------------------------------------------------------------------
// Python绑定
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm_cuda, "RMS归一化 (CUDA)");
    m.def("fused_add_rms_norm", &fused_add_rms_norm_cuda,
          "融合残差加法+RMS归一化 (CUDA)");
    m.def("silu_and_mul", &silu_and_mul_cuda, "SiLU激活乘法 (CUDA)");
    m.def("perturb_sign_shift", &perturb_sign_shift_cuda,
          "Anti-degenerate sign-shift perturbation (CUDA)");
    m.def("perturb_sign_norm", &perturb_sign_norm_cuda,
          "Anti-degenerate sign+norm perturbation (CUDA)");
    m.def("detect_repeat", &detect_repeat_cuda,
          "GPU repeat pattern detector (CUDA)");
    m.def("eos_check", &eos_check_cuda,
          "GPU EOS token check (CUDA)");
}
