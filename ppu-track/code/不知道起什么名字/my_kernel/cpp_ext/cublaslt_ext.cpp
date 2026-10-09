#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

#define CHECK_CUDA_STATUS(expr)                                                   \
  do {                                                                            \
    cudaError_t _status = (expr);                                                 \
    TORCH_CHECK(_status == cudaSuccess, #expr " failed: ",                       \
                cudaGetErrorString(_status));                                     \
  } while (0)

#define CHECK_CUBLAS_STATUS(expr)                                                 \
  do {                                                                            \
    cublasStatus_t _status = (expr);                                              \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS, #expr " failed with status ",  \
                static_cast<int>(_status));                                       \
  } while (0)

struct LtHandle {
  cublasLtHandle_t handle = nullptr;

  LtHandle() { CHECK_CUBLAS_STATUS(cublasLtCreate(&handle)); }

  ~LtHandle() {
    if (handle != nullptr) {
      cublasLtDestroy(handle);
    }
  }
};

struct CachedAlgo {
  cublasLtMatmulAlgo_t algo;
  size_t workspace_size = 0;
};

LtHandle& lt_handle() {
  static LtHandle state;
  return state;
}

std::mutex& algo_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, CachedAlgo>& algo_cache() {
  static std::unordered_map<std::string, CachedAlgo> cache;
  return cache;
}

cudaDataType_t scalar_to_cuda_type(const torch::Tensor& t) {
  if (t.scalar_type() == torch::kFloat16) {
    return CUDA_R_16F;
  }
  if (t.scalar_type() == torch::kBFloat16) {
    return CUDA_R_16BF;
  }
  TORCH_CHECK(false, "cublasLt epilogue only supports fp16/bf16 tensors");
}

std::string make_key(
    int device,
    int major,
    int minor,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaDataType_t dtype,
    bool has_bias,
    size_t workspace_size) {
  std::ostringstream oss;
  // The selected Lt algorithm is generally reusable across the row count for
  // the same output/features shape. Omitting n avoids a new heuristic query for
  // every image resolution while keeping a fallback path at the Python level.
  (void)n;
  oss << device << ":" << major << "." << minor << ":" << m << "x* x"
      << k << ":" << static_cast<int>(dtype) << ":" << has_bias << ":"
      << workspace_size;
  return oss.str();
}

void set_row_major(cublasLtMatrixLayout_t layout) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

CachedAlgo select_algo(
    cublasLtMatmulDesc_t op_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc,
    cublasLtMatrixLayout_t d_desc,
    const std::string& key,
    size_t workspace_size) {
  {
    std::lock_guard<std::mutex> lock(algo_mutex());
    auto it = algo_cache().find(key);
    if (it != algo_cache().end()) {
      return it->second;
    }
  }

  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_CUBLAS_STATUS(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLAS_STATUS(cublasLtMatmulPreferenceSetAttribute(
      pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size,
      sizeof(workspace_size)));

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned_results = 0;
  CHECK_CUBLAS_STATUS(cublasLtMatmulAlgoGetHeuristic(
      lt_handle().handle,
      op_desc,
      a_desc,
      b_desc,
      c_desc,
      d_desc,
      pref,
      1,
      &heuristic,
      &returned_results));
  CHECK_CUBLAS_STATUS(cublasLtMatmulPreferenceDestroy(pref));

  TORCH_CHECK(returned_results > 0, "cublasLt found no matmul algorithm");
  CachedAlgo cached{heuristic.algo, heuristic.workspaceSize};
  {
    std::lock_guard<std::mutex> lock(algo_mutex());
    algo_cache()[key] = cached;
  }
  return cached;
}

}  // namespace

torch::Tensor matmul_bias(
    torch::Tensor a,
    torch::Tensor b,
    c10::optional<torch::Tensor> bias,
    torch::Tensor out,
    torch::Tensor workspace) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "a, b and out must be CUDA tensors");
  TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2, "a, b and out must be 2D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "a, b and out must be contiguous");
  TORCH_CHECK(workspace.is_contiguous() && workspace.scalar_type() == torch::kUInt8,
              "workspace must be a contiguous uint8 tensor");
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == out.scalar_type(),
              "a, b and out must have the same dtype");
  TORCH_CHECK(a.size(1) == b.size(0), "a.shape[1] must equal b.shape[0]");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1), "out has wrong shape");

  const bool has_bias = bias.has_value() && bias.value().defined();
  if (has_bias) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.value().scalar_type() == out.scalar_type(), "bias dtype must match out");
    TORCH_CHECK(bias.value().numel() == b.size(1), "bias length must match output features");
  }

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(a));
  TORCH_CHECK(b.get_device() == a.get_device() && out.get_device() == a.get_device(),
              "a, b and out must be on the same device");
  TORCH_CHECK(workspace.get_device() == a.get_device(), "workspace must be on the same device");
  if (has_bias) {
    TORCH_CHECK(bias.value().get_device() == a.get_device(), "bias must be on the same device");
  }

  const int64_t row_m = a.size(0);
  const int64_t row_k = a.size(1);
  const int64_t row_n = b.size(1);
  TORCH_CHECK(row_m > 0 && row_n > 0 && row_k > 0, "empty matmul is not supported");

  cudaDataType_t dtype = scalar_to_cuda_type(a);
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  cudaDataType_t scale_type = CUDA_R_32F;

  // Row-major C = A[M,K] @ B[K,N] is represented as column-major
  // D_col[N,M] = B_col[N,K] @ A_col[K,M]. This makes cuBLASLt bias length N,
  // matching nn.Linear output features.
  const int64_t m = row_n;
  const int64_t n = row_m;
  const int64_t k = row_k;

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  CHECK_CUBLAS_STATUS(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));

  cublasOperation_t trans = CUBLAS_OP_N;
  CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

  cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
  CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  if (has_bias) {
    const void* bias_ptr = bias.value().data_ptr();
    CHECK_CUBLAS_STATUS(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }

  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&a_desc, dtype, m, k, m));
  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&b_desc, dtype, k, n, k));
  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&c_desc, dtype, m, n, m));
  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutCreate(&d_desc, dtype, m, n, m));

  const size_t workspace_size = static_cast<size_t>(workspace.numel());

  int device = a.get_device();
  cudaDeviceProp prop{};
  CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, device));
  std::string key = make_key(device, prop.major, prop.minor, m, n, k, dtype, has_bias, workspace_size);
  CachedAlgo algo = select_algo(op_desc, a_desc, b_desc, c_desc, d_desc, key, workspace_size);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CHECK_CUBLAS_STATUS(cublasLtMatmul(
      lt_handle().handle,
      op_desc,
      &alpha,
      b.data_ptr(),
      a_desc,
      a.data_ptr(),
      b_desc,
      &beta,
      out.data_ptr(),
      c_desc,
      out.data_ptr(),
      d_desc,
      &algo.algo,
      workspace.data_ptr(),
      algo.workspace_size,
      at::cuda::getCurrentCUDAStream()));

  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutDestroy(d_desc));
  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS_STATUS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS_STATUS(cublasLtMatmulDescDestroy(op_desc));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_bias", &matmul_bias, "cuBLASLt matmul with optional bias epilogue");
}
