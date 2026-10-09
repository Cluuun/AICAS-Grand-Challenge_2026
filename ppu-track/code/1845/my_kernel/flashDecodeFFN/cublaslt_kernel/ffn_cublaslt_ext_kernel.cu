#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kAlpha = 0.7978845608028654f;
  constexpr float kBeta = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

__global__ void silu_gate_to_bf16_kernel(
    const __nv_bfloat16* __restrict__ out13,
    __nv_bfloat16* __restrict__ gated,
    int total,
    int i_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int row = idx / i_size;
  int col = idx - row * i_size;
  int base = row * (2 * i_size);
  float gate = __bfloat162float(out13[base + col]);
  float up = __bfloat162float(out13[base + i_size + col]);
  gated[idx] = __float2bfloat16(silu(gate) * up);
}

__global__ void gelu_tanh_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  x[idx] = __float2bfloat16(gelu_tanh(__bfloat162float(x[idx])));
}

struct LtKey {
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
  int epilogue;
};

bool operator==(const LtKey& a, const LtKey& b) {
  return a.m == b.m && a.n == b.n && a.k == b.k &&
         a.lda == b.lda && a.ldb == b.ldb && a.ldc == b.ldc &&
         a.epilogue == b.epilogue;
}

struct LtKeyHash {
  size_t operator()(const LtKey& key) const {
    size_t h = static_cast<size_t>(key.m);
    h = h * 1315423911u + static_cast<size_t>(key.n);
    h = h * 1315423911u + static_cast<size_t>(key.k);
    h = h * 1315423911u + static_cast<size_t>(key.lda);
    h = h * 1315423911u + static_cast<size_t>(key.ldb);
    h = h * 1315423911u + static_cast<size_t>(key.ldc);
    h = h * 1315423911u + static_cast<size_t>(key.epilogue);
    return h;
  }
};

struct LtAlgoEntry {
  bool valid = false;
  cublasLtMatmulAlgo_t algo{};
  size_t workspace_bytes = 0;
  int algo_id = -1;
  int tile_id = -1;
  int splitk = -1;
  int swizzle = -1;
};

std::mutex g_algo_mu;
std::unordered_map<LtKey, LtAlgoEntry, LtKeyHash> g_algo_cache;
std::string g_last_report = "none";

thread_local at::Tensor t_workspace;
thread_local size_t t_workspace_bytes = 0;
thread_local int t_workspace_device = -1;

inline void check_cublas(cublasStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, msg);
}

void* ensure_workspace(int64_t device_idx, size_t bytes) {
  if (bytes == 0) return nullptr;
  if (!t_workspace.defined() ||
      t_workspace_bytes < bytes ||
      t_workspace_device != static_cast<int>(device_idx)) {
    auto opts = torch::TensorOptions()
        .device(torch::kCUDA, device_idx)
        .dtype(torch::kUInt8);
    t_workspace = torch::empty({static_cast<int64_t>(bytes)}, opts);
    t_workspace_bytes = bytes;
    t_workspace_device = static_cast<int>(device_idx);
  }
  return t_workspace.data_ptr();
}

LtAlgoEntry pick_algo(cublasLtHandle_t lt_handle,
                      const LtKey& key,
                      cublasLtMatmulDesc_t op_desc,
                      cublasLtMatrixLayout_t a_desc,
                      cublasLtMatrixLayout_t b_desc,
                      cublasLtMatrixLayout_t c_desc) {
  {
    std::lock_guard<std::mutex> guard(g_algo_mu);
    auto it = g_algo_cache.find(key);
    if (it != g_algo_cache.end()) return it->second;
  }

  constexpr size_t kMaxWorkspace = 64ull * 1024ull * 1024ull;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate failed");
  check_cublas(cublasLtMatmulPreferenceSetAttribute(
      pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kMaxWorkspace,
      sizeof(kMaxWorkspace)), "cublasLtMatmulPreferenceSetAttribute failed");

  constexpr int kRequest = 32;
  cublasLtMatmulHeuristicResult_t heur[kRequest];
  int found = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      lt_handle,
      op_desc,
      a_desc,
      b_desc,
      c_desc,
      c_desc,
      pref,
      kRequest,
      heur,
      &found);

  LtAlgoEntry entry;
  if (status == CUBLAS_STATUS_SUCCESS && found > 0) {
    entry.valid = true;
    entry.algo = heur[0].algo;
    entry.workspace_bytes = heur[0].workspaceSize;
    size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(
        &entry.algo, CUBLASLT_ALGO_CONFIG_ID,
        &entry.algo_id, sizeof(entry.algo_id), &written);
    cublasLtMatmulAlgoConfigGetAttribute(
        &entry.algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
        &entry.tile_id, sizeof(entry.tile_id), &written);
    cublasLtMatmulAlgoConfigGetAttribute(
        &entry.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        &entry.splitk, sizeof(entry.splitk), &written);
    cublasLtMatmulAlgoConfigGetAttribute(
        &entry.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        &entry.swizzle, sizeof(entry.swizzle), &written);
  } else {
    entry.valid = false;
  }
  check_cublas(cublasLtMatmulPreferenceDestroy(pref), "cublasLtMatmulPreferenceDestroy failed");

  {
    std::lock_guard<std::mutex> guard(g_algo_mu);
    g_algo_cache[key] = entry;
  }
  return entry;
}

void update_report(const char* tag, const LtKey& key, const LtAlgoEntry& entry) {
  std::ostringstream oss;
  if (entry.valid) {
    oss << tag << " m=" << key.m << " n=" << key.n << " k=" << key.k
        << " algo=" << entry.algo_id
        << " tile=" << entry.tile_id
        << " splitk=" << entry.splitk
        << " swizzle=" << entry.swizzle
        << " ws=" << entry.workspace_bytes;
  } else {
    oss << tag << " m=" << key.m << " n=" << key.n << " k=" << key.k
        << " fallback=cublasGemmEx";
  }
  std::lock_guard<std::mutex> guard(g_algo_mu);
  g_last_report = oss.str();
}

void gemm_rowmajor_bf16_lt(const torch::Tensor& a,
                           const torch::Tensor& b,
                           torch::Tensor& c,
                           const char* tag,
                           cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT,
                           const torch::Tensor* bias = nullptr) {
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "gemm_rowmajor_bf16_lt requires contiguous tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
              "gemm expects rank-2 tensors");
  TORCH_CHECK(a.dtype() == torch::kBFloat16 &&
                  b.dtype() == torch::kBFloat16 &&
                  c.dtype() == torch::kBFloat16,
              "gemm expects bf16 tensors");

  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  TORCH_CHECK(b.size(0) == k, "gemm K mismatch");
  TORCH_CHECK(c.size(0) == m && c.size(1) == n, "gemm output shape mismatch");

  auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>());
  auto* b_ptr = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>());
  auto* c_ptr = reinterpret_cast<__nv_bfloat16*>(c.data_ptr<at::BFloat16>());

  cublasLtHandle_t lt_handle = at::cuda::getCurrentCUDABlasLtHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
  check_cublas(cublasLtMatmulDescCreate(
      &op_desc, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F),
      "cublasLtMatmulDescCreate failed");

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  check_cublas(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)),
      "cublasLtMatmulDescSetAttribute TRANSA failed");
  check_cublas(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)),
      "cublasLtMatmulDescSetAttribute TRANSB failed");
  if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
    check_cublas(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
        "cublasLtMatmulDescSetAttribute EPILOGUE failed");
    TORCH_CHECK(bias != nullptr && bias->defined(), "epilogue requires bias");
    TORCH_CHECK(bias->is_cuda() && bias->dtype() == torch::kBFloat16 && bias->is_contiguous(),
                "bias must be contiguous CUDA bf16");
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n, "bias must be [N]");
    const void* bias_ptr = bias->data_ptr<at::BFloat16>();
    cudaDataType_t bias_dtype = CUDA_R_16BF;
    check_cublas(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)),
        "cublasLtMatmulDescSetAttribute BIAS_POINTER failed");
    check_cublas(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype, sizeof(bias_dtype)),
        "cublasLtMatmulDescSetAttribute BIAS_DATA_TYPE failed");
  }

  // The PPU cuBLASLt build does not support the all-row-major BF16 order
  // combination.  Keep the public row-major contract C[m,n] = A[m,k] @ B[k,n],
  // but present the same memory as column-major C^T[n,m] = B^T[n,k] @ A^T[k,m].
  check_cublas(cublasLtMatrixLayoutCreate(
      &a_desc, CUDA_R_16BF, n, k, n), "cublasLtMatrixLayoutCreate A failed");
  check_cublas(cublasLtMatrixLayoutCreate(
      &b_desc, CUDA_R_16BF, k, m, k), "cublasLtMatrixLayoutCreate B failed");
  check_cublas(cublasLtMatrixLayoutCreate(
      &c_desc, CUDA_R_16BF, n, m, n), "cublasLtMatrixLayoutCreate C failed");

  LtKey key{
      static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
      static_cast<int>(n), static_cast<int>(k), static_cast<int>(n),
      static_cast<int>(epilogue)};
  LtAlgoEntry entry = pick_algo(lt_handle, key, op_desc, a_desc, b_desc, c_desc);
  update_report(tag, key, entry);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  if (entry.valid) {
    void* ws_ptr = ensure_workspace(a.get_device(), entry.workspace_bytes);
    status = cublasLtMatmul(
        lt_handle,
        op_desc,
        &alpha,
        b_ptr,
        a_desc,
        a_ptr,
        b_desc,
        &beta,
        c_ptr,
        c_desc,
        c_ptr,
        c_desc,
        &entry.algo,
        ws_ptr,
        entry.workspace_bytes,
        stream);
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    if (entry.valid) {
      entry.valid = false;
      {
        std::lock_guard<std::mutex> guard(g_algo_mu);
        g_algo_cache[key] = entry;
      }
      update_report(tag, key, entry);
    }
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream failed");
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        b_ptr, CUDA_R_16BF, static_cast<int>(n),
        a_ptr, CUDA_R_16BF, static_cast<int>(k),
        &beta,
        c_ptr, CUDA_R_16BF, static_cast<int>(n),
        CUBLAS_COMPUTE_32F_FAST_16BF,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLt/cublas gemm failed");
  }

  check_cublas(cublasLtMatrixLayoutDestroy(a_desc), "cublasLtMatrixLayoutDestroy A failed");
  check_cublas(cublasLtMatrixLayoutDestroy(b_desc), "cublasLtMatrixLayoutDestroy B failed");
  check_cublas(cublasLtMatrixLayoutDestroy(c_desc), "cublasLtMatrixLayoutDestroy C failed");
  check_cublas(cublasLtMatmulDescDestroy(op_desc), "cublasLtMatmulDescDestroy failed");
}

void resolve_text_weight_layout(const torch::Tensor& w13_c,
                                const torch::Tensor& w2_c,
                                int64_t h,
                                torch::Tensor& w13_for_mm,
                                torch::Tensor& w2_for_mm,
                                int64_t& i) {
  i = 0;
  if (w13_c.size(0) == h && (w13_c.size(1) % 2 == 0)) {
    i = w13_c.size(1) / 2;
    w13_for_mm = w13_c;
  } else if (w13_c.size(1) == h && (w13_c.size(0) % 2 == 0)) {
    i = w13_c.size(0) / 2;
    w13_for_mm = w13_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false, "w13 must be [H,2I] or [2I,H]");
  }

  if (w2_c.size(0) == i && w2_c.size(1) == h) {
    w2_for_mm = w2_c;
  } else if (w2_c.size(0) == h && w2_c.size(1) == i) {
    w2_for_mm = w2_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false, "w2 must be [I,H] or [H,I]");
  }
}

void resolve_vision_weight_layout(const torch::Tensor& w1_c,
                                  const torch::Tensor& w2_c,
                                  int64_t h,
                                  torch::Tensor& w1_for_mm,
                                  torch::Tensor& w2_for_mm,
                                  int64_t& i) {
  i = 0;
  if (w1_c.size(0) == h) {
    i = w1_c.size(1);
    w1_for_mm = w1_c;
  } else if (w1_c.size(1) == h) {
    i = w1_c.size(0);
    w1_for_mm = w1_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false, "vision w1 must be [H,I] or [I,H]");
  }

  if (w2_c.size(0) == i && w2_c.size(1) == h) {
    w2_for_mm = w2_c;
  } else if (w2_c.size(0) == h && w2_c.size(1) == i) {
    w2_for_mm = w2_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false, "vision w2 must be [I,H] or [H,I]");
  }
}

}  // namespace

torch::Tensor ffn_cublaslt_text_forward_cuda(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2) {
  TORCH_CHECK(x.is_cuda() && w13.is_cuda() && w2.is_cuda(),
              "text tensors must be CUDA");
  TORCH_CHECK(x.dtype() == torch::kBFloat16 &&
                  w13.dtype() == torch::kBFloat16 &&
                  w2.dtype() == torch::kBFloat16,
              "text tensors must be bf16");
  TORCH_CHECK(x.dim() == 2, "x must be [N,H]");

  auto x_c = x.contiguous();
  auto w13_c = w13.contiguous();
  auto w2_c = w2.contiguous();
  int64_t n = x_c.size(0);
  int64_t h = x_c.size(1);

  torch::Tensor w13_mm;
  torch::Tensor w2_mm;
  int64_t i = 0;
  resolve_text_weight_layout(w13_c, w2_c, h, w13_mm, w2_mm, i);

  auto opts = x_c.options().dtype(torch::kBFloat16);
  auto out13 = torch::empty({n, 2 * i}, opts);
  auto gated = torch::empty({n, i}, opts);
  auto out = torch::empty({n, h}, opts);

  gemm_rowmajor_bf16_lt(x_c, w13_mm, out13, "text.gemm1");

  auto out13_ptr = reinterpret_cast<const __nv_bfloat16*>(out13.data_ptr<at::BFloat16>());
  auto gated_ptr = reinterpret_cast<__nv_bfloat16*>(gated.data_ptr<at::BFloat16>());
  int total = static_cast<int>(n * i);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  silu_gate_to_bf16_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      out13_ptr, gated_ptr, total, static_cast<int>(i));
  C10_CUDA_CHECK(cudaGetLastError());

  gemm_rowmajor_bf16_lt(gated, w2_mm, out, "text.gemm2");
  return out;
}

torch::Tensor ffn_cublaslt_vision_forward_cuda(torch::Tensor x,
                                               torch::Tensor w1,
                                               torch::Tensor b1,
                                               torch::Tensor w2,
                                               torch::Tensor b2) {
  TORCH_CHECK(x.is_cuda() && w1.is_cuda() && b1.is_cuda() && w2.is_cuda() && b2.is_cuda(),
              "vision tensors must be CUDA");
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && w1.dtype() == torch::kBFloat16 &&
                  b1.dtype() == torch::kBFloat16 && w2.dtype() == torch::kBFloat16 &&
                  b2.dtype() == torch::kBFloat16,
              "vision tensors must be bf16");
  TORCH_CHECK(x.dim() == 2, "vision x must be [N,H]");
  TORCH_CHECK(b1.dim() == 1 && b2.dim() == 1, "vision b1/b2 must be [I]/[H]");

  auto x_c = x.contiguous();
  auto w1_c = w1.contiguous();
  auto b1_c = b1.contiguous();
  auto w2_c = w2.contiguous();
  auto b2_c = b2.contiguous();

  int64_t n = x_c.size(0);
  int64_t h = x_c.size(1);
  torch::Tensor w1_mm;
  torch::Tensor w2_mm;
  int64_t i = 0;
  resolve_vision_weight_layout(w1_c, w2_c, h, w1_mm, w2_mm, i);

  TORCH_CHECK(b1_c.size(0) == i, "vision b1 shape mismatch");
  TORCH_CHECK(b2_c.size(0) == h, "vision b2 shape mismatch");

  auto opts = x_c.options().dtype(torch::kBFloat16);
  auto fc1 = torch::empty({n, i}, opts);
  auto out = torch::empty({n, h}, opts);

  gemm_rowmajor_bf16_lt(x_c, w1_mm, fc1, "vision.gemm1");
  fc1.add_(b1_c);
  int total = static_cast<int>(n * i);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kThreads = 256;
  auto fc1_ptr = reinterpret_cast<__nv_bfloat16*>(fc1.data_ptr<at::BFloat16>());
  gelu_tanh_inplace_bf16_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      fc1_ptr, total);
  C10_CUDA_CHECK(cudaGetLastError());

  gemm_rowmajor_bf16_lt(fc1, w2_mm, out, "vision.gemm2");
  out.add_(b2_c);
  return out;
}

std::string ffn_cublaslt_get_last_report_cuda() {
  std::lock_guard<std::mutex> guard(g_algo_mu);
  return g_last_report;
}
