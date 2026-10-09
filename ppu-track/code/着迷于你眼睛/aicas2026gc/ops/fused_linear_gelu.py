import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


DEFAULT_WORKSPACE_MB = 32
_workspace_cache = {}


cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>

#define CHECK_CUBLAS(status) TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt Error: ", status)

namespace {

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& workspace
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(weight.size(1) == x.size(1), "in_features mismatch");
    TORCH_CHECK(weight.size(0) == bias.size(0), "out_features mismatch");
    TORCH_CHECK(
        x.scalar_type() == weight.scalar_type() && weight.scalar_type() == bias.scalar_type(),
        "x, weight, and bias must share the same dtype"
    );
}

void get_cublas_types(
    at::ScalarType scalar_type,
    cudaDataType_t* dtype,
    cublasComputeType_t* compute_type
) {
    if (scalar_type == at::ScalarType::Half) {
        *dtype = CUDA_R_16F;
        *compute_type = CUBLAS_COMPUTE_32F;
        return;
    }
    if (scalar_type == at::ScalarType::BFloat16) {
        *dtype = CUDA_R_16BF;
        *compute_type = CUBLAS_COMPUTE_32F;
        return;
    }
    if (scalar_type == at::ScalarType::Float) {
        *dtype = CUDA_R_32F;
        *compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        return;
    }
    TORCH_CHECK(false, "only float16, bfloat16, and float32 are supported");
}

}  // namespace

torch::Tensor fused_linear_gelu_cublaslt(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& workspace
) {
    check_inputs(x, weight, bias, workspace);

    auto X = x.contiguous();
    auto W = weight.contiguous();
    auto B = bias.contiguous();

    const int64_t M = X.size(0);
    const int64_t K = X.size(1);
    const int64_t N = W.size(0);

    auto O = torch::empty({M, N}, X.options());

    cudaDataType_t dtype;
    cublasComputeType_t compute_type;
    get_cublas_types(X.scalar_type(), &dtype, &compute_type);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatmulDesc_t operationDesc;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, dtype, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, dtype, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, dtype, N, M, N));

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, compute_type, CUDA_R_32F));

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    const void* bias_ptr = B.data_ptr();
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    CHECK_CUBLAS(cublasLtMatmul(
        handle,
        operationDesc,
        &alpha,
        W.data_ptr(), layoutA,
        X.data_ptr(), layoutB,
        &beta,
        O.data_ptr(), layoutC,
        O.data_ptr(), layoutC,
        nullptr,
        workspace.data_ptr(),
        workspace.nbytes(),
        stream
    ));

    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));

    return O;
}
"""


cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_linear_gelu_cublaslt(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& workspace
);
"""


_cublaslt_module = load_inline(
    name="cublaslt_gelu_ultimate_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    with_cuda=True,
    functions=["fused_linear_gelu_cublaslt"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublasLt"],
    verbose=False,
)


def _get_workspace(device, workspace_mb):
    workspace_bytes = int(workspace_mb) * 1024 * 1024
    key = (str(device), workspace_bytes)
    workspace = _workspace_cache.get(key)
    if workspace is None or workspace.device != device or workspace.numel() != workspace_bytes:
        workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device=device)
        _workspace_cache[key] = workspace
    return workspace


def fused_linear_gelu(x, weight, bias, workspace_mb=DEFAULT_WORKSPACE_MB):
    workspace = _get_workspace(x.device, workspace_mb)
    return _cublaslt_module.fused_linear_gelu_cublaslt(x, weight, bias, workspace)


def torch_linear_gelu(x, weight, bias):
    return F.gelu(F.linear(x, weight, bias), approximate="tanh")


__all__ = [
    "DEFAULT_WORKSPACE_MB",
    "fused_linear_gelu",
    "torch_linear_gelu",
]


if __name__ == "__main__":
    try:
        from aicas2026gc.autotuner import RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[1] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("fused_linear_gelu_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        RuntimeAutotuner = module.RuntimeAutotuner

    hidden_size = 4096
    batch_size = 4096
    device = "cuda"
    dtype = torch.float16

    x = torch.randn((batch_size, hidden_size), dtype=dtype, device=device)
    weight = torch.randn((hidden_size, hidden_size), dtype=dtype, device=device)
    bias = torch.randn((hidden_size,), dtype=dtype, device=device)

    tuner = RuntimeAutotuner(
        name="fused_linear_gelu",
        fallback="torch_baseline",
        strict=False,
        atol=5e-2,
        rtol=5e-2,
    )

    candidates = {
        "torch_baseline": lambda x: torch_linear_gelu(x, weight, bias),
        "cublaslt": lambda x: fused_linear_gelu(x, weight, bias),
    }

    bucket_name = f"M{batch_size}_N{hidden_size}_K{hidden_size}_{str(dtype).split('.')[-1]}"
    out = tuner.dispatch_once(candidates, name=bucket_name, args=(x,))
    ref = torch_linear_gelu(x, weight, bias)
    max_diff = (ref - out).abs().max().item()
    best = tuner.dispatch_table[f"once:{bucket_name}"]
    print(f"best={best}, max_diff={max_diff:.6f}")
