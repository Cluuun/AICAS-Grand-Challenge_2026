#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <mutex>
#include <cstdint>
#include <unordered_map>

namespace {

struct LtKey {
  int m;
  int n;
  int k;
  int has_bias;
};

bool operator==(const LtKey& a, const LtKey& b) {
  return a.m == b.m && a.n == b.n && a.k == b.k && a.has_bias == b.has_bias;
}

struct LtKeyHash {
  size_t operator()(const LtKey& key) const {
    size_t h = static_cast<size_t>(key.m);
    h = h * 1315423911u + static_cast<size_t>(key.n);
    h = h * 1315423911u + static_cast<size_t>(key.k);
    h = h * 1315423911u + static_cast<size_t>(key.has_bias);
    return h;
  }
};

struct LtAlgoEntry {
  bool valid = false;
  cublasLtMatmulAlgo_t algo{};
  size_t workspace_bytes = 0;
};

std::mutex g_algo_mu;
std::unordered_map<LtKey, LtAlgoEntry, LtKeyHash> g_algo_cache;

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

  constexpr size_t kMaxWorkspace = 32ull * 1024ull * 1024ull;
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
  }
  check_cublas(cublasLtMatmulPreferenceDestroy(pref), "cublasLtMatmulPreferenceDestroy failed");

  {
    std::lock_guard<std::mutex> guard(g_algo_mu);
    g_algo_cache[key] = entry;
  }
  return entry;
}

__device__ __forceinline__ uint32_t cvt_shared_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}

__device__ __forceinline__ uint2 ldmatrix_x2_offset(uint32_t lane) {
  return make_uint2(lane & 7, (lane & 8) << 3);
}

__device__ __forceinline__ uint2 ldmatrix_x4_offset(uint32_t lane) {
  return make_uint2(lane & 15, (lane >> 4) << 3);
}

__device__ __forceinline__ void ldmatrix_sync_x1(const __nv_bfloat16* src_shared, uint32_t& a0) {
  uint32_t src = cvt_shared_u32(src_shared);
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(a0)
               : "r"(src));
}

__device__ __forceinline__ void ldmatrix_sync_x2(const __nv_bfloat16* src_shared, uint32_t& a0, uint32_t& a1) {
  uint32_t src = cvt_shared_u32(src_shared);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(a0), "=r"(a1)
               : "r"(src));
}

__device__ __forceinline__ void ldmatrix_sync_x4(const __nv_bfloat16* src_shared,
                                                 uint32_t& a0,
                                                 uint32_t& a1,
                                                 uint32_t& a2,
                                                 uint32_t& a3) {
  uint32_t src = cvt_shared_u32(src_shared);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(src));
}

__device__ __forceinline__ void mma_sync_m16n8k16_bf16(float& d0, float& d1, float& d2, float& d3,
                                                       uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                       uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_rowcol_m16n8k16(const __nv_bfloat16* a_tile,
                                                    const __nv_bfloat16* b_colmajor,
                                                    int lane,
                                                    float (&acc)[4]) {
  constexpr int KStep = 16;
  uint2 a_off = ldmatrix_x4_offset(static_cast<uint32_t>(lane));
  const __nv_bfloat16* a_base = a_tile + a_off.x * KStep + a_off.y;
  uint32_t a0, a1, a2, a3;
  ldmatrix_sync_x4(a_base, a0, a1, a2, a3);

  const __nv_bfloat16* b_base = b_colmajor + (lane & 7) * KStep + ((lane >> 3) & 1) * 8;
  uint32_t b0, b1;
  ldmatrix_sync_x2(b_base, b0, b1);
  mma_sync_m16n8k16_bf16(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
}

template <int WarpsPerRow>
__global__ void decode_linear_mma_tiled_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_t,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int M,
    int K,
    int N,
    bool has_bias) {
  constexpr int KStep = 16;
  constexpr int kWarps = WarpsPerRow;
  constexpr int kThreads = 32 * kWarps;
  constexpr int kPerWarpElems = 24 * KStep;  // a[16,KStep] + b[8,KStep]
  if (threadIdx.x >= kThreads) return;

  const int tile_group = blockIdx.x;
  const int row = blockIdx.y;
  if (row >= M) return;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile = tile_group * WarpsPerRow + warp;
  const int n_base = tile * 16;
  if (n_base >= N) return;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* x_sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* scratch = x_sh + K;
  __nv_bfloat16* warp_base = scratch + warp * kPerWarpElems;
  __nv_bfloat16* a_tile = warp_base;
  __nv_bfloat16* b_tile = a_tile + 16 * KStep;

  for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
    x_sh[idx] = x[row * K + idx];
  }
  __syncthreads();

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k0 = 0; k0 < K; k0 += KStep) {
    for (int t = lane; t < 16 * KStep; t += 32) {
      const int r = t / KStep;
      const int kk = t - r * KStep;
      a_tile[t] = w_t[(k0 + kk) * N + n_base + r];
    }
    for (int t = lane; t < 8 * KStep; t += 32) {
      const int col = t / KStep;
      const int kk = t - col * KStep;
      b_tile[t] = x_sh[k0 + kk];
    }
    __syncwarp();
    mma_rowcol_m16n8k16(a_tile, b_tile, lane, acc);
    __syncwarp();
  }

  if ((lane & 3) == 0) {
    const int r0 = lane >> 2;
    const int r1 = r0 + 8;
    float y0 = acc[0];
    float y1 = acc[2];
    if (has_bias) {
      y0 += __bfloat162float(bias[n_base + r0]);
      y1 += __bfloat162float(bias[n_base + r1]);
    }
    out[row * N + n_base + r0] = __float2bfloat16(y0);
    out[row * N + n_base + r1] = __float2bfloat16(y1);
  }
}

template <int WarpsPerRow>
__global__ void decode_linear_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_t,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int M,
    int K,
    int N,
    bool has_bias) {
  constexpr int KStep = 16;
  constexpr int kWarps = WarpsPerRow;
  constexpr int kThreads = 32 * kWarps;
  constexpr int kPerWarpElems = 24 * KStep;  // a[16,KStep] + b[8,KStep]
  if (threadIdx.x >= kThreads) return;

  const int row = blockIdx.x;
  if (row >= M) return;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* x_sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* scratch = x_sh + K;
  __nv_bfloat16* warp_base = scratch + warp * kPerWarpElems;
  __nv_bfloat16* a_tile = warp_base;
  __nv_bfloat16* b_tile = a_tile + 16 * KStep;

  for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
    x_sh[idx] = x[row * K + idx];
  }
  __syncthreads();

  const int n_tiles = N / 16;
  for (int tile = warp; tile < n_tiles; tile += WarpsPerRow) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int n_base = tile * 16;
    for (int k0 = 0; k0 < K; k0 += KStep) {
      for (int t = lane; t < 16 * KStep; t += 32) {
        const int r = t / KStep;
        const int kk = t - r * KStep;
        a_tile[t] = w_t[(k0 + kk) * N + n_base + r];
      }
      for (int t = lane; t < 8 * KStep; t += 32) {
        const int col = t / KStep;
        const int kk = t - col * KStep;
        b_tile[t] = x_sh[k0 + kk];
      }
      __syncwarp();
      mma_rowcol_m16n8k16(a_tile, b_tile, lane, acc);
      __syncwarp();
    }

    if ((lane & 3) == 0) {
      const int r0 = lane >> 2;
      const int r1 = r0 + 8;
      float y0 = acc[0];
      float y1 = acc[2];
      if (has_bias) {
        y0 += __bfloat162float(bias[n_base + r0]);
        y1 += __bfloat162float(bias[n_base + r1]);
      }
      out[row * N + n_base + r0] = __float2bfloat16(y0);
      out[row * N + n_base + r1] = __float2bfloat16(y1);
    }
  }
}

void launch_decode_linear_mma(
    const torch::Tensor& x,
    const torch::Tensor& w_t,
    const torch::Tensor& bias,
    torch::Tensor& out,
    bool has_bias,
    cudaStream_t stream) {
  constexpr int kWarpsPerRow = 8;
  constexpr int kThreads = 32 * kWarpsPerRow;
  const int m = static_cast<int>(x.size(0));
  const int k = static_cast<int>(x.size(1));
  const int n = static_cast<int>(w_t.size(1));
  TORCH_CHECK(k % 16 == 0 && n % 16 == 0, "MMA decode linear requires K/N multiples of 16");
  auto x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto w_ptr = reinterpret_cast<const __nv_bfloat16*>(w_t.data_ptr<at::BFloat16>());
  auto bias_ptr = has_bias
      ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>())
      : nullptr;
  auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const size_t smem = static_cast<size_t>(k + kWarpsPerRow * 24 * 16) * sizeof(__nv_bfloat16);
  static bool smem_attr_set = false;
  if (!smem_attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        decode_linear_mma_tiled_kernel<kWarpsPerRow>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));
    smem_attr_set = true;
  }
  dim3 grid((n / 16 + kWarpsPerRow - 1) / kWarpsPerRow, m, 1);
  decode_linear_mma_tiled_kernel<kWarpsPerRow><<<grid, kThreads, smem, stream>>>(
      x_ptr, w_ptr, bias_ptr, out_ptr, m, k, n, has_bias);
  C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void run_decode_linear_cublaslt(
    torch::Tensor x,
    torch::Tensor w_t,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias) {
  const int64_t m64 = x.size(0);
  const int64_t k64 = x.size(1);
  const int64_t n64 = w_t.size(1);
  TORCH_CHECK(m64 > 0 && n64 > 0 && k64 > 0, "invalid matmul shape");
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "shape too large");

  const int m = static_cast<int>(m64);
  const int n = static_cast<int>(n64);
  const int k = static_cast<int>(k64);

  auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto* w_ptr = reinterpret_cast<const __nv_bfloat16*>(w_t.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

  cublasLtHandle_t lt_handle = at::cuda::getCurrentCUDABlasLtHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;

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

  if (has_bias) {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    check_cublas(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
        "cublasLtMatmulDescSetAttribute EPILOGUE failed");
    const void* bias_ptr = bias.data_ptr<at::BFloat16>();
    cudaDataType_t bias_dtype = CUDA_R_16BF;
    check_cublas(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)),
        "cublasLtMatmulDescSetAttribute BIAS_POINTER failed");
    check_cublas(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype, sizeof(bias_dtype)),
        "cublasLtMatmulDescSetAttribute BIAS_DATA_TYPE failed");
  }

  check_cublas(cublasLtMatrixLayoutCreate(
      &a_desc, CUDA_R_16BF, m, k, k), "cublasLtMatrixLayoutCreate A failed");
  check_cublas(cublasLtMatrixLayoutCreate(
      &b_desc, CUDA_R_16BF, k, n, n), "cublasLtMatrixLayoutCreate B failed");
  check_cublas(cublasLtMatrixLayoutCreate(
      &c_desc, CUDA_R_16BF, m, n, n), "cublasLtMatrixLayoutCreate C failed");

  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  check_cublas(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
      "cublasLtMatrixLayoutSetAttribute A order failed");
  check_cublas(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
      "cublasLtMatrixLayoutSetAttribute B order failed");
  check_cublas(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
      "cublasLtMatrixLayoutSetAttribute C order failed");

  LtKey key{m, n, k, has_bias ? 1 : 0};
  LtAlgoEntry entry = pick_algo(lt_handle, key, op_desc, a_desc, b_desc, c_desc);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
  if (entry.valid) {
    void* ws_ptr = ensure_workspace(x.get_device(), entry.workspace_bytes);
    status = cublasLtMatmul(
        lt_handle,
        op_desc,
        &alpha,
        x_ptr,
        a_desc,
        w_ptr,
        b_desc,
        &beta,
        out_ptr,
        c_desc,
        out_ptr,
        c_desc,
        &entry.algo,
        ws_ptr,
        entry.workspace_bytes,
        stream);
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream failed");
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        w_ptr, CUDA_R_16BF, n,
        x_ptr, CUDA_R_16BF, k,
        &beta,
        out_ptr, CUDA_R_16BF, n,
        CUBLAS_COMPUTE_32F_FAST_16BF,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "decode linear cublasLt/cublas failed");
    if (has_bias) {
      out.add_(bias);
    }
  }

  check_cublas(cublasLtMatrixLayoutDestroy(a_desc), "cublasLtMatrixLayoutDestroy A failed");
  check_cublas(cublasLtMatrixLayoutDestroy(b_desc), "cublasLtMatrixLayoutDestroy B failed");
  check_cublas(cublasLtMatrixLayoutDestroy(c_desc), "cublasLtMatrixLayoutDestroy C failed");
  check_cublas(cublasLtMatmulDescDestroy(op_desc), "cublasLtMatmulDescDestroy failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_decode_linear_mma(
    torch::Tensor x,
    torch::Tensor w_t,
    torch::Tensor bias,
    torch::Tensor out,
    bool has_bias) {
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = w_t.size(1);
  TORCH_CHECK(m > 0 && k == w_t.size(0) && n > 0, "invalid decode linear MMA shape");
  TORCH_CHECK(k % 16 == 0 && n % 16 == 0, "decode linear MMA requires K/N multiples of 16");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_decode_linear_mma(x, w_t, bias, out, has_bias, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
