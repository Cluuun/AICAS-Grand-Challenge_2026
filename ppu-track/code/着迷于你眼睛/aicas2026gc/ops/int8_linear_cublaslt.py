import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WORKSPACE_MB = 32
DEFAULT_MAX_ALGOS = 8


cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#define CHECK_CUBLAS(status) TORCH_CHECK((status) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error: ", int(status))

namespace {

constexpr int kLayoutColMajorTrick = 0;
constexpr int kLayoutRowMajor = 1;
constexpr int kThreads = 256;

void check_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_int8(const torch::Tensor& t, const char* name) {
    check_cuda(t, name);
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Char, name, " must be torch.int8");
}

void check_float_scale(const torch::Tensor& t, const char* name) {
    check_cuda(t, name);
    TORCH_CHECK(t.scalar_type() == at::ScalarType::Float, name, " must be torch.float32");
}

template <typename scalar_t>
__global__ void rowwise_quantize_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scale,
    int rows,
    int cols
) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float local_max = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = static_cast<float>(x[row * cols + col]);
        local_max = fmaxf(local_max, fabsf(v));
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float s = smem[0] / 127.0f;
    if (!(s > 0.0f) || !isfinite(s)) {
        s = 1.0e-8f;
    }
    if (tid == 0) {
        scale[row] = s;
    }
    __syncthreads();

    const float inv_s = 1.0f / s;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = static_cast<float>(x[row * cols + col]) * inv_s;
        int qv = __float2int_rn(v);
        if (qv > 127) {
            qv = 127;
        } else if (qv < -127) {
            qv = -127;
        }
        q[row * cols + col] = static_cast<int8_t>(qv);
    }
}

template <typename scalar_t>
__global__ void rmsnorm_rowwise_quant_kernel(
    const scalar_t* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scale,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * cols;

    float local_sumsq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = static_cast<float>(x[base + col]);
        local_sumsq += v * v;
    }

    smem[tid] = local_sumsq;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(smem[0] / static_cast<float>(cols) + eps);
    __syncthreads();

    float local_max = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = static_cast<float>(x[base + col]) * inv_rms;
        local_max = fmaxf(local_max, fabsf(v));
    }

    smem[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float s = smem[0] / 127.0f;
    if (!(s > 0.0f) || !isfinite(s)) {
        s = 1.0e-8f;
    }
    if (tid == 0) {
        scale[row] = s;
    }
    __syncthreads();

    const float inv_s = 1.0f / s;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = static_cast<float>(x[base + col]) * inv_rms * inv_s;
        int qv = __float2int_rn(v);
        if (qv > 127) {
            qv = 127;
        } else if (qv < -127) {
            qv = -127;
        }
        q[base + col] = static_cast<int8_t>(qv);
    }
}

template <typename scalar_t>
__global__ void swiglu_rowwise_quant_kernel(
    const scalar_t* __restrict__ gate_up,
    int8_t* __restrict__ q,
    float* __restrict__ scale,
    int rows,
    int cols
) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int in_base = row * cols * 2;
    const int out_base = row * cols;

    float local_max = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float gate = static_cast<float>(gate_up[in_base + col]);
        const float up = static_cast<float>(gate_up[in_base + cols + col]);
        const float sig = 1.0f / (1.0f + expf(-gate));
        const float v = gate * sig * up;
        local_max = fmaxf(local_max, fabsf(v));
    }

    smem[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float s = smem[0] / 127.0f;
    if (!(s > 0.0f) || !isfinite(s)) {
        s = 1.0e-8f;
    }
    if (tid == 0) {
        scale[row] = s;
    }
    __syncthreads();

    const float inv_s = 1.0f / s;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float gate = static_cast<float>(gate_up[in_base + col]);
        const float up = static_cast<float>(gate_up[in_base + cols + col]);
        const float sig = 1.0f / (1.0f + expf(-gate));
        const float v = gate * sig * up * inv_s;
        int qv = __float2int_rn(v);
        if (qv > 127) {
            qv = 127;
        } else if (qv < -127) {
            qv = -127;
        }
        q[out_base + col] = static_cast<int8_t>(qv);
    }
}

template <typename scalar_t>
__global__ void dequant_i32_to_fp_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int total,
    int n_cols,
    bool has_bias,
    bool has_residual
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int m = idx / n_cols;
    const int n = idx - m * n_cols;
    float v = static_cast<float>(acc[idx]) * x_scale[m] * w_scale[n];
    if (has_bias) {
        v += static_cast<float>(bias[n]);
    }
    if (has_residual) {
        v += static_cast<float>(residual[idx]);
    }
    out[idx] = static_cast<scalar_t>(v);
}

void create_i8_matmul_descs(
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t layout_mode,
    cublasLtMatmulDesc_t* matmul_desc,
    cublasLtMatrixLayout_t* a_desc,
    cublasLtMatrixLayout_t* b_desc,
    cublasLtMatrixLayout_t* c_desc,
    cublasLtMatrixLayout_t* d_desc
) {
    CHECK_CUBLAS(cublasLtMatmulDescCreate(matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    if (layout_mode == kLayoutColMajorTrick) {
        cublasOperation_t trans_a = CUBLAS_OP_T;
        cublasOperation_t trans_b = CUBLAS_OP_N;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(a_desc, CUDA_R_8I, k, n, k));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(b_desc, CUDA_R_8I, k, m, k));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(c_desc, CUDA_R_32I, n, m, n));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(d_desc, CUDA_R_32I, n, m, n));
        return;
    }

    TORCH_CHECK(layout_mode == kLayoutRowMajor, "layout_mode must be 0(col-major trick) or 1(row-major)");

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(*matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(a_desc, CUDA_R_8I, m, k, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(b_desc, CUDA_R_8I, n, k, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(c_desc, CUDA_R_32I, m, n, n));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(d_desc, CUDA_R_32I, m, n, n));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(*d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
}

void destroy_descs(
    cublasLtMatmulDesc_t matmul_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc,
    cublasLtMatrixLayout_t d_desc
) {
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmul_desc));
}

int get_heuristics(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t matmul_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc,
    cublasLtMatrixLayout_t d_desc,
    size_t workspace_size,
    std::vector<cublasLtMatmulHeuristicResult_t>& heuristics
) {
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)
    ));

    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmul_desc,
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        preference,
        static_cast<int>(heuristics.size()),
        heuristics.data(),
        &returned
    ));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    return returned;
}

void run_cublaslt_i8i32(
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& workspace,
    torch::Tensor& acc,
    int64_t heuristic_index,
    int64_t layout_mode
) {
    const int64_t m = x_q.size(0);
    const int64_t k = x_q.size(1);
    const int64_t n = w_q.size(0);

    cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t c_desc;
    cublasLtMatrixLayout_t d_desc;
    create_i8_matmul_descs(m, k, n, layout_mode, &matmul_desc, &a_desc, &b_desc, &c_desc, &d_desc);

    constexpr int kMaxAlgos = 32;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(kMaxAlgos);
    const size_t workspace_size = workspace.numel() > 0 ? static_cast<size_t>(workspace.nbytes()) : 0;
    int returned = get_heuristics(handle, matmul_desc, a_desc, b_desc, c_desc, d_desc, workspace_size, heuristics);
    TORCH_CHECK(returned > 0, "cublasLtMatmulAlgoGetHeuristic returned no INT8 algorithms");

    int algo_index = static_cast<int>(heuristic_index);
    if (algo_index < 0) {
        algo_index = 0;
    }
    if (algo_index >= returned) {
        algo_index = returned - 1;
    }

    const int32_t alpha = 1;
    const int32_t beta = 0;
    void* workspace_ptr = workspace_size > 0 ? workspace.data_ptr() : nullptr;

    const void* a_ptr = nullptr;
    const void* b_ptr = nullptr;
    if (layout_mode == kLayoutColMajorTrick) {
        a_ptr = w_q.data_ptr();
        b_ptr = x_q.data_ptr();
    } else {
        a_ptr = x_q.data_ptr();
        b_ptr = w_q.data_ptr();
    }

    CHECK_CUBLAS(cublasLtMatmul(
        handle,
        matmul_desc,
        &alpha,
        a_ptr,
        a_desc,
        b_ptr,
        b_desc,
        &beta,
        acc.data_ptr(),
        c_desc,
        acc.data_ptr(),
        d_desc,
        &heuristics[algo_index].algo,
        workspace_ptr,
        workspace_size,
        stream
    ));

    destroy_descs(matmul_desc, a_desc, b_desc, c_desc, d_desc);
}

}  // namespace

void rowwise_quantize_out(torch::Tensor x, torch::Tensor q, torch::Tensor scale) {
    check_cuda(x, "x");
    check_int8(q, "q");
    check_float_scale(scale, "scale");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(q.sizes() == x.sizes(), "q shape mismatch");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == x.size(0), "scale shape mismatch");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Float,
                "rowwise_quantize_out supports float16 and float32");

    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rowwise_quantize_out", [&] {
        rowwise_quantize_kernel<scalar_t><<<rows, kThreads, kThreads * sizeof(float), stream>>>(
            x.data_ptr<scalar_t>(),
            reinterpret_cast<int8_t*>(q.data_ptr<int8_t>()),
            scale.data_ptr<float>(),
            rows,
            cols
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rmsnorm_rowwise_quant_out(torch::Tensor x, torch::Tensor q, torch::Tensor scale, double eps) {
    check_cuda(x, "x");
    check_int8(q, "q");
    check_float_scale(scale, "scale");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(q.sizes() == x.sizes(), "q shape mismatch");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == x.size(0), "scale shape mismatch");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Float,
                "rmsnorm_rowwise_quant_out supports float16 and float32");

    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rmsnorm_rowwise_quant_out", [&] {
        rmsnorm_rowwise_quant_kernel<scalar_t><<<rows, kThreads, kThreads * sizeof(float), stream>>>(
            x.data_ptr<scalar_t>(),
            reinterpret_cast<int8_t*>(q.data_ptr<int8_t>()),
            scale.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(eps)
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void swiglu_rowwise_quant_out(torch::Tensor gate_up, torch::Tensor q, torch::Tensor scale) {
    check_cuda(gate_up, "gate_up");
    check_int8(q, "q");
    check_float_scale(scale, "scale");
    TORCH_CHECK(gate_up.dim() == 2, "gate_up must be 2D");
    TORCH_CHECK(gate_up.size(1) % 2 == 0, "gate_up hidden dimension must be even");
    const int rows = static_cast<int>(gate_up.size(0));
    const int cols = static_cast<int>(gate_up.size(1) / 2);
    TORCH_CHECK(q.dim() == 2 && q.size(0) == rows && q.size(1) == cols, "q shape mismatch");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == rows, "scale shape mismatch");
    TORCH_CHECK(gate_up.scalar_type() == at::ScalarType::Half || gate_up.scalar_type() == at::ScalarType::Float,
                "swiglu_rowwise_quant_out supports float16 and float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(gate_up.scalar_type(), "swiglu_rowwise_quant_out", [&] {
        swiglu_rowwise_quant_kernel<scalar_t><<<rows, kThreads, kThreads * sizeof(float), stream>>>(
            gate_up.data_ptr<scalar_t>(),
            reinterpret_cast<int8_t*>(q.data_ptr<int8_t>()),
            scale.data_ptr<float>(),
            rows,
            cols
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor int8_linear_prequant_out(
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor workspace,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
) {
    check_int8(x_q, "x_q");
    check_int8(w_q, "w_q");
    check_float_scale(x_scale, "x_scale");
    check_float_scale(w_scale, "w_scale");
    check_cuda(workspace, "workspace");
    check_cuda(acc, "acc");
    check_cuda(out, "out");
    TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Byte, "workspace must be torch.uint8");
    TORCH_CHECK(acc.scalar_type() == at::ScalarType::Int, "acc must be torch.int32");
    TORCH_CHECK(x_q.dim() == 2 && w_q.dim() == 2, "x_q and w_q must be 2D");
    TORCH_CHECK(x_q.size(1) == w_q.size(1), "K mismatch");

    const int64_t m = x_q.size(0);
    const int64_t n = w_q.size(0);
    TORCH_CHECK(x_scale.dim() == 1 && x_scale.size(0) == m, "x_scale shape mismatch");
    TORCH_CHECK(w_scale.dim() == 1 && w_scale.size(0) == n, "w_scale shape mismatch");
    TORCH_CHECK(acc.dim() == 2 && acc.size(0) == m && acc.size(1) == n, "acc shape mismatch");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == m && out.size(1) == n, "out shape mismatch");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Half || out.scalar_type() == at::ScalarType::Float,
                "out must be float16 or float32");
    TORCH_CHECK(bias.numel() == 0 || (bias.is_cuda() && bias.is_contiguous()), "bias must be empty or CUDA contiguous");
    TORCH_CHECK(bias.numel() == 0 || (bias.dim() == 1 && bias.size(0) == n), "bias shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == out.scalar_type(), "bias dtype must match out dtype");

    run_cublaslt_i8i32(x_q, w_q, workspace, acc, heuristic_index, layout_mode);

    const int total = static_cast<int>(m * n);
    const int blocks = (total + kThreads - 1) / kThreads;
    const bool has_bias = bias.numel() > 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(out.scalar_type(), "dequant_i32_to_fp", [&] {
        const scalar_t* bias_ptr = has_bias ? bias.data_ptr<scalar_t>() : nullptr;
        dequant_i32_to_fp_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            acc.data_ptr<int32_t>(),
            x_scale.data_ptr<float>(),
            w_scale.data_ptr<float>(),
            bias_ptr,
            nullptr,
            out.data_ptr<scalar_t>(),
            total,
            static_cast<int>(n),
            has_bias,
            false
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int8_linear_prequant_residual_out(
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor workspace,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
) {
    check_int8(x_q, "x_q");
    check_int8(w_q, "w_q");
    check_float_scale(x_scale, "x_scale");
    check_float_scale(w_scale, "w_scale");
    check_cuda(workspace, "workspace");
    check_cuda(acc, "acc");
    check_cuda(out, "out");
    check_cuda(residual, "residual");
    TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Byte, "workspace must be torch.uint8");
    TORCH_CHECK(acc.scalar_type() == at::ScalarType::Int, "acc must be torch.int32");
    TORCH_CHECK(x_q.dim() == 2 && w_q.dim() == 2, "x_q and w_q must be 2D");
    TORCH_CHECK(x_q.size(1) == w_q.size(1), "K mismatch");

    const int64_t m = x_q.size(0);
    const int64_t n = w_q.size(0);
    TORCH_CHECK(x_scale.dim() == 1 && x_scale.size(0) == m, "x_scale shape mismatch");
    TORCH_CHECK(w_scale.dim() == 1 && w_scale.size(0) == n, "w_scale shape mismatch");
    TORCH_CHECK(acc.dim() == 2 && acc.size(0) == m && acc.size(1) == n, "acc shape mismatch");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == m && out.size(1) == n, "out shape mismatch");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == m && residual.size(1) == n, "residual shape mismatch");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Half || out.scalar_type() == at::ScalarType::Float,
                "out must be float16 or float32");
    TORCH_CHECK(residual.scalar_type() == out.scalar_type(), "residual dtype must match out dtype");
    TORCH_CHECK(bias.numel() == 0 || (bias.is_cuda() && bias.is_contiguous()), "bias must be empty or CUDA contiguous");
    TORCH_CHECK(bias.numel() == 0 || (bias.dim() == 1 && bias.size(0) == n), "bias shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.scalar_type() == out.scalar_type(), "bias dtype must match out dtype");

    run_cublaslt_i8i32(x_q, w_q, workspace, acc, heuristic_index, layout_mode);

    const int total = static_cast<int>(m * n);
    const int blocks = (total + kThreads - 1) / kThreads;
    const bool has_bias = bias.numel() > 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(out.scalar_type(), "dequant_i32_to_fp_residual", [&] {
        const scalar_t* bias_ptr = has_bias ? bias.data_ptr<scalar_t>() : nullptr;
        dequant_i32_to_fp_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            acc.data_ptr<int32_t>(),
            x_scale.data_ptr<float>(),
            w_scale.data_ptr<float>(),
            bias_ptr,
            residual.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            total,
            static_cast<int>(n),
            has_bias,
            true
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor int8_linear_dynamic_out(
    torch::Tensor x,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor workspace,
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
) {
    rowwise_quantize_out(x, x_q, x_scale);
    return int8_linear_prequant_out(
        x_q,
        x_scale,
        w_q,
        w_scale,
        bias,
        workspace,
        acc,
        out,
        heuristic_index,
        layout_mode
    );
}

torch::Tensor int8_linear_dynamic_residual_out(
    torch::Tensor x,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor workspace,
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
) {
    rowwise_quantize_out(x, x_q, x_scale);
    return int8_linear_prequant_residual_out(
        x_q,
        x_scale,
        w_q,
        w_scale,
        bias,
        residual,
        workspace,
        acc,
        out,
        heuristic_index,
        layout_mode
    );
}

int64_t int8_heuristic_count(
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t workspace_bytes,
    int64_t layout_mode
) {
    TORCH_CHECK(m > 0 && k > 0 && n > 0, "m, k, n must be positive");
    TORCH_CHECK(workspace_bytes >= 0, "workspace_bytes must be non-negative");

    cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t c_desc;
    cublasLtMatrixLayout_t d_desc;
    create_i8_matmul_descs(m, k, n, layout_mode, &matmul_desc, &a_desc, &b_desc, &c_desc, &d_desc);

    constexpr int kMaxAlgos = 32;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(kMaxAlgos);
    const int returned = get_heuristics(
        handle,
        matmul_desc,
        a_desc,
        b_desc,
        c_desc,
        d_desc,
        static_cast<size_t>(workspace_bytes),
        heuristics
    );
    destroy_descs(matmul_desc, a_desc, b_desc, c_desc, d_desc);
    return static_cast<int64_t>(returned);
}
"""


cpp_source = r"""
#include <torch/extension.h>

void rowwise_quantize_out(torch::Tensor x, torch::Tensor q, torch::Tensor scale);
void rmsnorm_rowwise_quant_out(torch::Tensor x, torch::Tensor q, torch::Tensor scale, double eps);
void swiglu_rowwise_quant_out(torch::Tensor gate_up, torch::Tensor q, torch::Tensor scale);

torch::Tensor int8_linear_prequant_out(
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor workspace,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
);

torch::Tensor int8_linear_dynamic_out(
    torch::Tensor x,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor workspace,
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
);

torch::Tensor int8_linear_prequant_residual_out(
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor workspace,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
);

torch::Tensor int8_linear_dynamic_residual_out(
    torch::Tensor x,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor workspace,
    torch::Tensor x_q,
    torch::Tensor x_scale,
    torch::Tensor acc,
    torch::Tensor out,
    int64_t heuristic_index,
    int64_t layout_mode
);

int64_t int8_heuristic_count(
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t workspace_bytes,
    int64_t layout_mode
);
"""


_module = load_inline(
    name="int8_linear_cublaslt_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    with_cuda=True,
    functions=[
        "rowwise_quantize_out",
        "rmsnorm_rowwise_quant_out",
        "swiglu_rowwise_quant_out",
        "int8_linear_prequant_out",
        "int8_linear_dynamic_out",
        "int8_linear_prequant_residual_out",
        "int8_linear_dynamic_residual_out",
        "int8_heuristic_count",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lcublasLt"],
    verbose=False,
)


def _load_autotuner():
    try:
        from aicas2026gc.autotuner import RuntimeAutotuner

        return RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[1] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("int8_linear_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RuntimeAutotuner


def rowwise_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    _module.rowwise_quantize_out(x.contiguous(), q, scale)
    return q, scale


def rmsnorm_rowwise_quant_out(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, eps: float) -> None:
    _module.rmsnorm_rowwise_quant_out(x.contiguous(), q, scale, float(eps))


def swiglu_rowwise_quant_out(gate_up: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> None:
    _module.swiglu_rowwise_quant_out(gate_up.contiguous(), q, scale)


def prepack_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return rowwise_quantize(weight.contiguous())


def _empty_bias(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((0,), device=device, dtype=dtype)


def _make_cases() -> dict[str, dict[str, object]]:
    return {
        # Language prefill attention.
        "lang_attn_qkv_768": {"m": 768, "k": 2048, "n": 4096, "bias": False, "residual": False, "post_op": "none"},
        "lang_attn_qkv_1024": {"m": 1024, "k": 2048, "n": 4096, "bias": False, "residual": False, "post_op": "none"},
        "lang_attn_o_768": {"m": 768, "k": 2048, "n": 2048, "bias": False, "residual": True, "post_op": "none"},
        "lang_attn_o_1024": {"m": 1024, "k": 2048, "n": 2048, "bias": False, "residual": True, "post_op": "none"},
        # Language prefill MLP.
        "lang_gate_up_768": {"m": 768, "k": 2048, "n": 12288, "bias": False, "residual": False, "post_op": "none"},
        "lang_gate_up_1024": {"m": 1024, "k": 2048, "n": 12288, "bias": False, "residual": False, "post_op": "none"},
        "lang_down_768": {"m": 768, "k": 6144, "n": 2048, "bias": False, "residual": True, "post_op": "none"},
        "lang_down_1024": {"m": 1024, "k": 6144, "n": 2048, "bias": False, "residual": True, "post_op": "none"},
        # Vision prefill patch/attention/MLP/merger.
        "vision_patch_1024": {"m": 1024, "k": 1536, "n": 1024, "bias": True, "residual": True, "post_op": "none"},
        "vision_patch_4096": {"m": 4096, "k": 1536, "n": 1024, "bias": True, "residual": True, "post_op": "none"},
        "vision_attn_qkv_1024": {"m": 1024, "k": 1024, "n": 3072, "bias": True, "residual": False, "post_op": "none"},
        "vision_attn_qkv_4096": {"m": 4096, "k": 1024, "n": 3072, "bias": True, "residual": False, "post_op": "none"},
        "vision_attn_proj_4096": {"m": 4096, "k": 1024, "n": 1024, "bias": True, "residual": True, "post_op": "none"},
        "vision_mlp_fc1_4096": {"m": 4096, "k": 1024, "n": 4096, "bias": True, "residual": False, "post_op": "gelu"},
        "vision_mlp_fc2_4096": {"m": 4096, "k": 4096, "n": 1024, "bias": True, "residual": True, "post_op": "none"},
        "vision_merger_fc1_1024": {"m": 1024, "k": 4096, "n": 4096, "bias": True, "residual": False, "post_op": "gelu"},
        "vision_merger_fc2_1024": {"m": 1024, "k": 4096, "n": 2048, "bias": True, "residual": False, "post_op": "none"},
    }


class CublasLtInt8Linear:
    """Runtime INT8 linear runner for fixed-shape prefill buckets.

    The runner keeps quantized weights and reusable scratch buffers alive so the
    selected cuBLASLt path can be captured by CUDA Graph. It intentionally leaves
    activation quantization dynamic, because prefill activations change per
    sample.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        name: str,
        workspace_mb: int = DEFAULT_WORKSPACE_MB,
        max_algos: int = DEFAULT_MAX_ALGOS,
        max_m: int = 1050,
        layout_mode: int = 0,
        check_correctness: bool = False,
        autotune: bool = False,
    ) -> None:
        RuntimeAutotuner = _load_autotuner()
        if not weight.is_cuda:
            raise ValueError("CublasLtInt8Linear expects CUDA weight")
        if weight.dim() != 2:
            raise ValueError("weight must be [out_features, in_features]")

        self.weight = weight.detach().contiguous()
        self.weight_t = self.weight.t().contiguous()
        self.bias = bias.detach().contiguous() if bias is not None and bias.numel() > 0 else _empty_bias(weight.device, weight.dtype)
        self.w_q, self.w_scale = prepack_weight(self.weight)

        self.name = name
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])
        self.workspace_mb = int(workspace_mb)
        self.workspace_bytes = self.workspace_mb * 1024 * 1024
        self.max_algos = int(max_algos)
        self.max_m = int(max_m)
        self.layout_mode = int(layout_mode)
        self.autotune = bool(autotune)
        self.log_enabled = os.environ.get("INT8_LINEAR_LOG", "1").lower() not in ("0", "false", "f", "no")
        self._logged_first_call = False
        self._logged_buckets: set[str] = set()
        self.tuner = RuntimeAutotuner(
            name=f"int8_linear_cublaslt_runtime_{name}_{self.in_features}x{self.out_features}",
            fallback="torch_mm_out",
            check_correctness=check_correctness,
            strict=False,
            atol=1.0,
            rtol=1.0,
        )
        self.prequant_tuner = RuntimeAutotuner(
            name=f"int8_linear_cublaslt_runtime_prequant_{name}_{self.in_features}x{self.out_features}",
            fallback="i8_pre_coltrick_h0",
            check_correctness=False,
            strict=False,
            atol=1.0,
            rtol=1.0,
        )
        self.heuristic_counts: dict[int, int] = {}
        self.workspace: torch.Tensor | None = None
        self.acc: torch.Tensor | None = None
        self.out: torch.Tensor | None = None
        self.x_q: torch.Tensor | None = None
        self.x_scale: torch.Tensor | None = None
        self.torch_out: torch.Tensor | None = None
        if self.log_enabled:
            print(
                f"[INT8Linear][init] name={self.name} K={self.in_features} N={self.out_features} "
                f"bias={int(self.bias.numel() > 0)} max_m={self.max_m} max_algos={self.max_algos} "
                f"workspace_mb={self.workspace_mb} autotune={int(self.autotune)} layout=coltrick",
                flush=True,
            )

    def _fallback(self, x_2d: torch.Tensor, residual_2d: torch.Tensor | None = None) -> torch.Tensor:
        if residual_2d is not None:
            return torch.addmm(residual_2d, x_2d, self.weight_t)
        if self.bias.numel() > 0:
            return F.linear(x_2d, self.weight, self.bias)
        return torch.mm(x_2d, self.weight_t)

    def _ensure_buffers(self, m: int, device: torch.device, dtype: torch.dtype) -> None:
        alloc_m = self.max_m
        if self.workspace is not None and self.acc is not None and self.acc.shape[0] >= alloc_m:
            return
        self.workspace = torch.empty((self.workspace_bytes,), device=device, dtype=torch.uint8)
        self.acc = torch.empty((alloc_m, self.out_features), device=device, dtype=torch.int32)
        self.out = torch.empty((alloc_m, self.out_features), device=device, dtype=dtype)
        self.x_q = torch.empty((alloc_m, self.in_features), device=device, dtype=torch.int8)
        self.x_scale = torch.empty((alloc_m,), device=device, dtype=torch.float32)
        self.torch_out = torch.empty((alloc_m, self.out_features), device=device, dtype=dtype)

    def can_prequant(self, m: int, dtype: torch.dtype) -> bool:
        return 0 < int(m) <= self.max_m and dtype in (torch.float16, torch.float32)

    def prequant_buffers(
        self,
        m: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_buffers(int(m), device, dtype)
        assert self.x_q is not None
        assert self.x_scale is not None
        return self.x_q[: int(m), :], self.x_scale[: int(m)]

    def _heuristic_count(self, m: int) -> int:
        m = int(m)
        if m not in self.heuristic_counts:
            try:
                self.heuristic_counts[m] = int(
                    _module.int8_heuristic_count(m, self.in_features, self.out_features, self.workspace_bytes, self.layout_mode)
                )
            except RuntimeError:
                self.heuristic_counts[m] = 0
        return self.heuristic_counts[m]

    def _direct_int8(
        self,
        x_2d: torch.Tensor,
        residual_2d: torch.Tensor | None,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        acc: torch.Tensor,
        out_buf: torch.Tensor,
    ) -> torch.Tensor:
        if residual_2d is None:
            return _module.int8_linear_dynamic_out(
                x_2d,
                self.w_q,
                self.w_scale,
                self.bias,
                self.workspace,
                x_q,
                x_scale,
                acc,
                out_buf,
                0,
                self.layout_mode,
            )
        return _module.int8_linear_dynamic_residual_out(
            x_2d,
            self.w_q,
            self.w_scale,
            self.bias,
            residual_2d,
            self.workspace,
            x_q,
            x_scale,
            acc,
            out_buf,
            0,
            self.layout_mode,
        )

    def _prequant_h0(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        residual_2d: torch.Tensor | None,
        acc: torch.Tensor,
        out_buf: torch.Tensor,
    ) -> torch.Tensor:
        if residual_2d is None:
            return _module.int8_linear_prequant_out(
                x_q,
                x_scale,
                self.w_q,
                self.w_scale,
                self.bias,
                self.workspace,
                acc,
                out_buf,
                0,
                self.layout_mode,
            )
        return _module.int8_linear_prequant_residual_out(
            x_q,
            x_scale,
            self.w_q,
            self.w_scale,
            self.bias,
            residual_2d,
            self.workspace,
            acc,
            out_buf,
            0,
            self.layout_mode,
        )

    def forward_prequant(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        *,
        input_shape: tuple[int, ...],
        bucket: int | None = None,
        is_warmup: bool = False,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = int(x_q.shape[0])
        dtype = residual.dtype if residual is not None else self.weight.dtype
        if not self.can_prequant(m, dtype):
            raise RuntimeError(f"{self.name}: prequant path requires 0 < M <= {self.max_m} and fp16/fp32 output dtype")

        self._ensure_buffers(m, x_q.device, dtype)
        assert self.workspace is not None
        assert self.acc is not None
        assert self.out is not None
        assert self.torch_out is not None

        acc = self.acc[:m, :]
        out_buf = self.out[:m, :]
        torch_out = self.torch_out[:m, :]
        residual_2d = None if residual is None else residual.view(-1, self.out_features)

        if self.log_enabled and not self._logged_first_call:
            print(
                f"[INT8Linear][call] name={self.name} M={m} K={self.in_features} N={self.out_features} "
                f"residual={int(residual_2d is not None)} bias={int(self.bias.numel() > 0)} "
                f"autotune={int(self.autotune)} path={'prequant_h0' if not self.autotune else 'prequant_autotune'}",
                flush=True,
            )
            self._logged_first_call = True

        if not self.autotune:
            selected = self._prequant_h0(x_q, x_scale, residual_2d, acc, out_buf)
            return selected.view(*input_shape[:-1], self.out_features)

        candidates = {}
        count = min(self._heuristic_count(m), self.max_algos)
        if count <= 0:
            selected = self._prequant_h0(x_q, x_scale, residual_2d, acc, out_buf)
            return selected.view(*input_shape[:-1], self.out_features)
        for h in range(count):
            if residual_2d is None:
                candidates[f"i8_pre_coltrick_h{h}"] = (
                    lambda _unused, h=h: _module.int8_linear_prequant_out(
                        x_q,
                        x_scale,
                        self.w_q,
                        self.w_scale,
                        self.bias,
                        self.workspace,
                        acc,
                        out_buf,
                        h,
                        self.layout_mode,
                    )
                )
            else:
                candidates[f"i8_pre_coltrick_h{h}"] = (
                    lambda _unused, h=h: _module.int8_linear_prequant_residual_out(
                        x_q,
                        x_scale,
                        self.w_q,
                        self.w_scale,
                        self.bias,
                        residual_2d,
                        self.workspace,
                        acc,
                        out_buf,
                        h,
                        self.layout_mode,
                    )
                )

        dummy = torch_out
        fallback_name = "i8_pre_coltrick_h0"
        if bucket is None:
            once_name = f"pre_M{m}_K{self.in_features}_N{self.out_features}_res{int(residual_2d is not None)}_ws{self.workspace_mb}"
            selected = self.prequant_tuner.dispatch_once(candidates, name=once_name, fallback=fallback_name, args=(dummy,))
            log_key = f"once:{once_name}"
        else:
            selected = self.prequant_tuner.dispatch(int(bucket), candidates, is_warmup=is_warmup, fallback=fallback_name, args=(dummy,))
            log_key = f"bucket:{int(bucket)}"
        if self.log_enabled and log_key not in self._logged_buckets:
            table_key = f"once:{once_name}" if bucket is None else int(bucket)
            best = self.prequant_tuner.dispatch_table.get(table_key, "unknown")
            print(
                f"[INT8Linear][select] name={self.name} {log_key} best={best} "
                f"prequant=1 residual={int(residual_2d is not None)} bias={int(self.bias.numel() > 0)}",
                flush=True,
            )
            self._logged_buckets.add(log_key)
        return selected.view(*input_shape[:-1], self.out_features)

    def __call__(
        self,
        x: torch.Tensor,
        *,
        bucket: int | None = None,
        is_warmup: bool = False,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = tuple(x.shape)
        x_2d = x.view(-1, self.in_features)
        m = int(x_2d.shape[0])
        if m <= 0 or m > self.max_m or x_2d.dtype not in (torch.float16, torch.float32):
            out = self._fallback(x_2d, None if residual is None else residual.view(-1, self.out_features))
            return out.view(*orig_shape[:-1], self.out_features)

        self._ensure_buffers(m, x_2d.device, x_2d.dtype)
        assert self.workspace is not None
        assert self.acc is not None
        assert self.out is not None
        assert self.x_q is not None
        assert self.x_scale is not None
        assert self.torch_out is not None

        acc = self.acc[:m, :]
        out_buf = self.out[:m, :]
        x_q = self.x_q[:m, :]
        x_scale = self.x_scale[:m]
        torch_out = self.torch_out[:m, :]
        residual_2d = None if residual is None else residual.view(-1, self.out_features)
        if self.log_enabled and not self._logged_first_call:
            print(
                f"[INT8Linear][call] name={self.name} M={m} K={self.in_features} N={self.out_features} "
                f"residual={int(residual_2d is not None)} bias={int(self.bias.numel() > 0)} "
                f"autotune={int(self.autotune)} path={'direct_h0' if not self.autotune else 'autotune'}",
                flush=True,
            )
            self._logged_first_call = True

        if not self.autotune:
            selected = self._direct_int8(x_2d, residual_2d, x_q, x_scale, acc, out_buf)
            return selected.view(*orig_shape[:-1], self.out_features)

        def torch_mm_out(x_arg: torch.Tensor) -> torch.Tensor:
            if residual_2d is not None:
                return torch.addmm(residual_2d, x_arg, self.weight_t, out=torch_out)
            if self.bias.numel() > 0:
                return torch.addmm(self.bias, x_arg, self.weight_t, out=torch_out)
            return torch.mm(x_arg, self.weight_t, out=torch_out)

        candidates = {"torch_mm_out": torch_mm_out}
        count = min(self._heuristic_count(m), self.max_algos)
        for h in range(count):
            if residual_2d is None:
                candidates[f"i8_dyn_coltrick_h{h}"] = (
                    lambda x_arg, h=h: _module.int8_linear_dynamic_out(
                        x_arg,
                        self.w_q,
                        self.w_scale,
                        self.bias,
                        self.workspace,
                        x_q,
                        x_scale,
                        acc,
                        out_buf,
                        h,
                        self.layout_mode,
                    )
                )
            else:
                candidates[f"i8_dyn_coltrick_h{h}"] = (
                    lambda x_arg, h=h: _module.int8_linear_dynamic_residual_out(
                        x_arg,
                        self.w_q,
                        self.w_scale,
                        self.bias,
                        residual_2d,
                        self.workspace,
                        x_q,
                        x_scale,
                        acc,
                        out_buf,
                        h,
                        self.layout_mode,
                    )
                )

        if bucket is None:
            once_name = f"M{m}_K{self.in_features}_N{self.out_features}_res{int(residual_2d is not None)}_ws{self.workspace_mb}"
            selected = self.tuner.dispatch_once(candidates, name=once_name, args=(x_2d,))
            log_key = f"once:{once_name}"
        else:
            selected = self.tuner.dispatch(int(bucket), candidates, is_warmup=is_warmup, args=(x_2d,))
            log_key = f"bucket:{int(bucket)}"
        if self.log_enabled and log_key not in self._logged_buckets:
            table_key = f"once:{once_name}" if bucket is None else int(bucket)
            best = self.tuner.dispatch_table.get(table_key, "unknown")
            print(
                f"[INT8Linear][select] name={self.name} {log_key} best={best} "
                f"residual={int(residual_2d is not None)} bias={int(self.bias.numel() > 0)}",
                flush=True,
            )
            self._logged_buckets.add(log_key)
        return selected.view(*orig_shape[:-1], self.out_features)


def _error_stats(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    diff = (out.float() - ref.float()).abs()
    ref_f = ref.float()
    denom = ref_f.abs().mean().clamp_min(1.0e-6)
    cos = F.cosine_similarity(out.float().flatten(), ref_f.flatten(), dim=0)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_mean": (diff.mean() / denom).item(),
        "cos": cos.item(),
    }


def _print_stats(name: str, ref: torch.Tensor, out: torch.Tensor) -> None:
    stats = _error_stats(ref, out)
    print(
        f"[error][{name}] "
        f"max_abs={stats['max_abs']:.6f}, "
        f"mean_abs={stats['mean_abs']:.6f}, "
        f"rel_mean={stats['rel_mean']:.6f}, "
        f"cos={stats['cos']:.8f}"
    )


def run_case(
    case_name: str,
    m: int,
    k: int,
    n: int,
    has_bias: bool,
    has_residual: bool,
    post_op: str,
    workspace_mb: int,
    max_algos: int,
    seed: int,
    probe_rowmajor: bool,
) -> None:
    RuntimeAutotuner = _load_autotuner()

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.float16
    workspace_bytes = int(workspace_mb) * 1024 * 1024

    x = torch.randn((m, k), device=device, dtype=dtype)
    weight = torch.randn((n, k), device=device, dtype=dtype)
    bias = torch.randn((n,), device=device, dtype=dtype) if has_bias else _empty_bias(device, dtype)
    residual = torch.randn((m, n), device=device, dtype=dtype) if has_residual else None
    weight_t = weight.t().contiguous()
    zero_bias = torch.zeros((n,), device=device, dtype=dtype)

    w_q, w_scale = prepack_weight(weight)
    x_q_static, x_scale_static = rowwise_quantize(x)

    workspace = torch.empty((workspace_bytes,), device=device, dtype=torch.uint8)
    acc = torch.empty((m, n), device=device, dtype=torch.int32)
    out_pre = torch.empty((m, n), device=device, dtype=dtype)
    out_dyn = torch.empty((m, n), device=device, dtype=dtype)
    x_q_dyn = torch.empty((m, k), device=device, dtype=torch.int8)
    x_scale_dyn = torch.empty((m,), device=device, dtype=torch.float32)
    torch_out = torch.empty((m, n), device=device, dtype=dtype)

    if post_op not in ("none", "gelu"):
        raise ValueError(f"unsupported post_op={post_op}")

    print(
        f"[INT8Linear][bench] case={case_name} M={m} K={k} N={n} "
        f"bias={int(has_bias)} residual={int(has_residual)} post_op={post_op} "
        f"workspace_mb={workspace_mb} max_algos={max_algos}",
        flush=True,
    )

    def apply_post_op(y: torch.Tensor) -> torch.Tensor:
        if post_op == "gelu":
            return F.gelu(y, approximate="tanh")
        return y

    def torch_ref(x_arg: torch.Tensor) -> torch.Tensor:
        y = F.linear(x_arg, weight, None if not has_bias else bias)
        if residual is not None:
            y = y + residual
        return apply_post_op(y)

    def torch_mm_out(x_arg: torch.Tensor) -> torch.Tensor:
        if residual is not None:
            if has_bias:
                y = torch.addmm(residual + bias, x_arg, weight_t, out=torch_out)
            else:
                y = torch.addmm(residual, x_arg, weight_t, out=torch_out)
        elif has_bias:
            y = torch.addmm(bias, x_arg, weight_t, out=torch_out)
        else:
            y = torch.mm(x_arg, weight_t, out=torch_out)
        return apply_post_op(y)

    candidates = {
        "torch_ref": torch_ref,
        "torch_mm_out": torch_mm_out,
    }

    if residual is not None and post_op == "none":
        try:
            from aicas2026gc.ops.fused_linear_add import fused_linear_add

            fused_bias = bias if has_bias else zero_bias
            candidates["fp16_fused_add"] = lambda x_arg: fused_linear_add(residual, x_arg, weight, fused_bias)
        except Exception as exc:
            print(f"[fp16_baseline][{case_name}] fused_linear_add unavailable: {exc}", flush=True)

    if residual is None and post_op == "gelu" and has_bias:
        try:
            from aicas2026gc.ops.fused_linear_gelu import fused_linear_gelu

            candidates["fp16_fused_gelu"] = lambda x_arg: fused_linear_gelu(x_arg, weight, bias)
        except Exception as exc:
            print(f"[fp16_baseline][{case_name}] fused_linear_gelu unavailable: {exc}", flush=True)

    layout_names = {0: "coltrick"}
    if probe_rowmajor:
        layout_names[1] = "rowmajor"
    heuristic_counts: dict[int, int] = {}
    for layout_mode, layout_name in layout_names.items():
        try:
            count = int(_module.int8_heuristic_count(m, k, n, workspace_bytes, layout_mode))
        except RuntimeError as exc:
            print(f"[heuristic][{case_name}][{layout_name}] failed: {exc}")
            count = 0
        heuristic_counts[layout_mode] = count
        print(f"[heuristic][{case_name}][{layout_name}] count={count}")

        for h in range(min(count, max_algos)):
            if residual is None:
                candidates[f"i8_pre_{layout_name}_h{h}"] = (
                    lambda x_arg, h=h, layout_mode=layout_mode: apply_post_op(
                        _module.int8_linear_prequant_out(
                            x_q_static,
                            x_scale_static,
                            w_q,
                            w_scale,
                            bias,
                            workspace,
                            acc,
                            out_pre,
                            h,
                            layout_mode,
                        )
                    )
                )
                candidates[f"i8_dyn_{layout_name}_h{h}"] = (
                    lambda x_arg, h=h, layout_mode=layout_mode: apply_post_op(
                        _module.int8_linear_dynamic_out(
                            x_arg,
                            w_q,
                            w_scale,
                            bias,
                            workspace,
                            x_q_dyn,
                            x_scale_dyn,
                            acc,
                            out_dyn,
                            h,
                            layout_mode,
                        )
                    )
                )
            else:
                candidates[f"i8_pre_{layout_name}_h{h}"] = (
                    lambda x_arg, h=h, layout_mode=layout_mode: apply_post_op(
                        _module.int8_linear_prequant_residual_out(
                            x_q_static,
                            x_scale_static,
                            w_q,
                            w_scale,
                            bias,
                            residual,
                            workspace,
                            acc,
                            out_pre,
                            h,
                            layout_mode,
                        )
                    )
                )
                candidates[f"i8_dyn_{layout_name}_h{h}"] = (
                    lambda x_arg, h=h, layout_mode=layout_mode: apply_post_op(
                        _module.int8_linear_dynamic_residual_out(
                            x_arg,
                            w_q,
                            w_scale,
                            bias,
                            residual,
                            workspace,
                            x_q_dyn,
                            x_scale_dyn,
                            acc,
                            out_dyn,
                            h,
                            layout_mode,
                        )
                    )
                )

    tuner = RuntimeAutotuner(
        name=f"int8_linear_cublaslt_{case_name}",
        fallback="torch_ref",
        check_correctness=False,
        strict=False,
        atol=1.0,
        rtol=1.0,
    )

    bucket_name = f"{case_name}_M{m}_N{n}_K{k}_bias{int(has_bias)}_res{int(has_residual)}_{post_op}_ws{workspace_mb}"
    print(f"\n=== {bucket_name} ===")
    out = tuner.dispatch_once(candidates, name=bucket_name, args=(x,))
    best = tuner.dispatch_table[f"once:{bucket_name}"]

    ref = torch_ref(x)
    _print_stats(best, ref, out)

    for layout_mode, layout_name in layout_names.items():
        if heuristic_counts.get(layout_mode, 0) <= 0:
            continue
        pre_name = f"i8_pre_{layout_name}_h0"
        dyn_name = f"i8_dyn_{layout_name}_h0"
        _print_stats(pre_name, ref, candidates[pre_name](x))
        _print_stats(dyn_name, ref, candidates[dyn_name](x))

    print(f"[done][{case_name}] best={best}")


def parse_args() -> argparse.Namespace:
    cases = _make_cases()
    parser = argparse.ArgumentParser(description="cuBLASLt INT8 linear benchmark.")
    parser.add_argument("--case", default="lang_gate_up_768", choices=["all", "custom", *cases.keys()])
    parser.add_argument("--m", type=int, default=768)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=12288)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--post-op", choices=["none", "gelu"], default="none")
    parser.add_argument("--workspace-mb", type=int, default=DEFAULT_WORKSPACE_MB)
    parser.add_argument("--max-algos", type=int, default=DEFAULT_MAX_ALGOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-rowmajor", action="store_true")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this benchmark must run on a CUDA GPU.")

    args = parse_args()
    cases = _make_cases()

    if args.case == "all":
        for name, params in cases.items():
            try:
                run_case(
                    name,
                    int(params["m"]),
                    int(params["k"]),
                    int(params["n"]),
                    bool(params["bias"]),
                    bool(params["residual"]),
                    str(params["post_op"]),
                    args.workspace_mb,
                    args.max_algos,
                    args.seed,
                    args.probe_rowmajor,
                )
            except Exception as exc:
                print(f"[case_failed][{name}] {type(exc).__name__}: {exc}", flush=True)
        return

    if args.case == "custom":
        run_case(
            "custom",
            args.m,
            args.k,
            args.n,
            args.bias,
            args.residual,
            args.post_op,
            args.workspace_mb,
            args.max_algos,
            args.seed,
            args.probe_rowmajor,
        )
        return

    params = cases[args.case]
    run_case(
        args.case,
        int(params["m"]),
        int(params["k"]),
        int(params["n"]),
        bool(params["bias"]),
        bool(params["residual"]),
        str(params["post_op"]),
        args.workspace_mb,
        args.max_algos,
        args.seed,
        args.probe_rowmajor,
    )


if __name__ == "__main__":
    main()
