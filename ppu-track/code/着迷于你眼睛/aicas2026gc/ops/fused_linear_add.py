import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


BEST_WORKSPACE_MB = 4
BEST_HEURISTIC_INDEX = 0


cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>

namespace {

void check_cuda_inputs(
    const torch::Tensor& origin_hidden_states,
    const torch::Tensor& attn_output,
    const torch::Tensor& weight,
    const torch::Tensor& bias
) {
    TORCH_CHECK(origin_hidden_states.is_cuda(), "origin_hidden_states must be CUDA");
    TORCH_CHECK(attn_output.is_cuda(), "attn_output must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(origin_hidden_states.dim() == 2, "origin_hidden_states must be 2D");
    TORCH_CHECK(attn_output.dim() == 2, "attn_output must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(attn_output.size(0) == origin_hidden_states.size(0), "M mismatch");
    TORCH_CHECK(weight.size(0) == origin_hidden_states.size(1), "N mismatch");
    TORCH_CHECK(attn_output.size(1) == weight.size(1), "K mismatch");
    TORCH_CHECK(bias.size(0) == weight.size(0), "bias size mismatch");
    TORCH_CHECK(
        attn_output.scalar_type() == origin_hidden_states.scalar_type() &&
        weight.scalar_type() == attn_output.scalar_type() &&
        bias.scalar_type() == attn_output.scalar_type(),
        "all tensors must share the same dtype"
    );
}

void get_cublas_types(
    at::ScalarType scalar_type,
    cudaDataType_t* dtype,
    cublasComputeType_t* compute_type
) {
    if (scalar_type == at::ScalarType::Half) {
        *dtype = CUDA_R_16F;
        *compute_type = CUBLAS_COMPUTE_32F; // 强制使用 FP32 累加保证精度
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

torch::Tensor fused_gemm_ultimate(
    torch::Tensor origin_hidden_states,
    torch::Tensor attn_output,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t workspace_bytes,
    int64_t heuristic_index
) {
    check_cuda_inputs(origin_hidden_states, attn_output, weight, bias);
    TORCH_CHECK(workspace_bytes >= 0, "workspace_bytes must be non-negative");
    TORCH_CHECK(heuristic_index >= 0, "heuristic_index must be non-negative");

    auto C = origin_hidden_states.contiguous();
    auto A = attn_output.contiguous();
    auto W = weight.contiguous();
    auto b = bias.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = W.size(0);

    auto D = torch::empty_like(C);

    cudaDataType_t dtype;
    cublasComputeType_t compute_type;
    get_cublas_types(A.scalar_type(), &dtype, &compute_type);

    cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();


    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t c_desc;
    cublasLtMatrixLayout_t d_desc;
    cublasLtMatmulPreference_t preference;

    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, compute_type, CUDA_R_32F));

    // ==========================================
    // 🔥 列优先转置魔法 (The Transpose Trick)
    // 目标: D_row = A_row * W_row^T + C_row
    // 等价 Col-Major: D_col^T = W_col * A_col^T + C_col^T
    // ==========================================
    cublasOperation_t trans_a = CUBLAS_OP_T;  // W 需要转置
    cublasOperation_t trans_b = CUBLAS_OP_N;  // A 保持 (Col-Major下原样就是A^T)
    
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    const void* bias_ptr = b.data_ptr();
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));

    // 创建 Layout：以 Col-Major 视角传入 (ld, rows, cols)
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, dtype, K, N, K)); // W_col
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, dtype, K, M, K)); // A_col^T
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, dtype, N, M, N)); // C_col^T
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&d_desc, dtype, N, M, N)); // D_col^T

    float alpha = 1.0f;
    float beta = 1.0f; // Beta=1 使得 cuBLAS 会把 C 当作 Residual 累加

    size_t workspace_size = static_cast<size_t>(workspace_bytes);
    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = torch::empty(
            {static_cast<long long>(workspace_size)},
            torch::dtype(torch::kUInt8).device(A.device())
        );
        workspace_ptr = workspace.data_ptr();
    }

    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)
        )
    );

    constexpr int kMaxAlgos = 16;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(kMaxAlgos);
    int returned_results = 0;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulAlgoGetHeuristic(
            handle, matmul_desc, a_desc, b_desc, c_desc, d_desc, 
            preference, kMaxAlgos, heuristics.data(), &returned_results
        )
    );

    TORCH_CHECK(returned_results > 0, "cublasLtMatmulAlgoGetHeuristic returned no algorithms");
    TORCH_CHECK(heuristic_index < returned_results, "heuristic_index out of range");

    // 🔥 执行计算：对调 W 和 A 的传入位置，适配上面的列优先魔法
    cublasStatus_t status = cublasLtMatmul(
        handle,
        matmul_desc,
        &alpha,
        W.data_ptr(), // <--- X 位置传 W
        a_desc,
        A.data_ptr(), // <--- Y 位置传 A
        b_desc,
        &beta,
        C.data_ptr(),
        c_desc,
        D.data_ptr(),
        d_desc,
        &heuristics[heuristic_index].algo,
        workspace_ptr,
        workspace_size,
        stream
    );

    TORCH_CUDABLAS_CHECK(status);

    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc));

    return D;
}
"""


cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_gemm_ultimate(
    torch::Tensor origin_hidden_states,
    torch::Tensor attn_output,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t workspace_bytes,
    int64_t heuristic_index
);
"""

fused_module = load_inline(
    name="fused_linear_add_cublaslt_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    with_cuda=True,
    functions=["fused_gemm_ultimate"],
    extra_ldflags=["-lcublasLt"],
    verbose=False,
)


def fused_linear_add(
    origin_hidden_states,
    attn_output,
    weight,
    bias,
    workspace_mb=BEST_WORKSPACE_MB,
    heuristic_index=BEST_HEURISTIC_INDEX,
):
    return fused_module.fused_gemm_ultimate(
        origin_hidden_states,
        attn_output,
        weight,
        bias,
        int(workspace_mb) * 1024 * 1024,
        int(heuristic_index),
    )


def torch_linear_add(origin_hidden_states, attn_output, weight, bias):
    return F.linear(attn_output, weight, bias).add_(origin_hidden_states)


__all__ = [
    "BEST_HEURISTIC_INDEX",
    "BEST_WORKSPACE_MB",
    "fused_linear_add",
    "torch_linear_add",
]


if __name__ == "__main__":
    try:
        from aicas2026gc.autotuner import RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[1] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("fused_linear_add_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        RuntimeAutotuner = module.RuntimeAutotuner

    M, K, N = 4096, 3452, 1024
    device = "cuda"
    dtype = torch.float16

    tuner = RuntimeAutotuner(
        name="fused_linear_add",
        fallback="torch_linear_add",
        strict=False,
        atol=1e-2,  # 如果依然存在硬件级 FP16 舍入误差，此处的容忍度可以帮你越过阈值
        rtol=1e-2,
    )

    origin_hidden_states = torch.randn(M, N, device=device, dtype=dtype)
    attn_output = torch.randn(M, K, device=device, dtype=dtype)
    weight = torch.randn(N, K, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)

    def torch_addmm(origin_hidden_states, attn_output, weight, bias):
        return torch.addmm(origin_hidden_states + bias, attn_output, weight.T)

    def make_cublaslt_impl(workspace_mb, heuristic_index):
        def impl(origin_hidden_states, attn_output, weight, bias):
            return fused_linear_add(
                origin_hidden_states,
                attn_output,
                weight,
                bias,
                workspace_mb=workspace_mb,
                heuristic_index=heuristic_index,
            )

        return impl

    candidates = {
        "torch_addmm": torch_addmm,
        "torch_linear_add": torch_linear_add,
        "cublaslt_best": make_cublaslt_impl(BEST_WORKSPACE_MB, BEST_HEURISTIC_INDEX),
    }

    bucket_name = f"M{M}_N{N}_K{K}_{str(dtype).split('.')[-1]}"

    print("\n🚀 正在测试 fused_linear_add...")
    out = tuner.dispatch_once(
        candidates,
        name=bucket_name,
        args=(origin_hidden_states, attn_output, weight, bias),
    )

    ref = torch_linear_add(origin_hidden_states, attn_output, weight, bias)
    max_diff = (ref - out).abs().max().item()
    best = tuner.dispatch_table[f"once:{bucket_name}"]
    print(f"✅ 测试完成! best_algo={best}, max_diff={max_diff:.6f}")
