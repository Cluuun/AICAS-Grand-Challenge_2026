#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>

torch::Tensor flashdecodeffn_pack_w13_cuda(torch::Tensor w1,
                                           torch::Tensor w3);
torch::Tensor flashdecodeffn_pack_w2_cuda(torch::Tensor w2);
torch::Tensor flashdecodeffn_mlp_packed_cuda(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2);
void flashdecodeffn_mlp_packed_into_cuda(torch::Tensor x,
                                         torch::Tensor w13,
                                         torch::Tensor w2,
                                         torch::Tensor out13,
                                         torch::Tensor gated,
                                         torch::Tensor out);

torch::Tensor flashdecodeffn_pack_vision_w1_cuda(torch::Tensor w1);
torch::Tensor flashdecodeffn_pack_vision_w2_cuda(torch::Tensor w2);
torch::Tensor flashdecodeffn_vision_mlp_packed_cuda(torch::Tensor x,
                                                    torch::Tensor w1,
                                                    torch::Tensor b1,
                                                    torch::Tensor w2,
                                                    torch::Tensor b2);
void flashdecodeffn_vision_mlp_packed_into_cuda(torch::Tensor x,
                                                torch::Tensor w1,
                                                torch::Tensor b1,
                                                torch::Tensor w2,
                                                torch::Tensor b2,
                                                torch::Tensor fc1,
                                                torch::Tensor out);
torch::Tensor flashdecodeffn_text_decode_v2_cuda(torch::Tensor x,
                                                 torch::Tensor w13,
                                                 torch::Tensor w2);
torch::Tensor flashdecodeffn_text_prefill_v2_cuda(torch::Tensor x,
                                                  torch::Tensor w13,
                                                  torch::Tensor w2);
torch::Tensor flashdecodeffn_text_forward_v2_cuda(torch::Tensor x,
                                                  torch::Tensor w13,
                                                  torch::Tensor w2);
torch::Tensor flashdecodeffn_vision_forward_v2_cuda(torch::Tensor x,
                                                    torch::Tensor w1,
                                                    torch::Tensor b1,
                                                    torch::Tensor w2,
                                                    torch::Tensor b2);

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2 / pi)
  constexpr float kBeta = 0.044715f;
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

__global__ void silu_gate_to_bf16_kernel(
    const __nv_bfloat16* __restrict__ out13,
    __nv_bfloat16* __restrict__ gated,
    int N, int I) {
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * I;
  int total_pairs = total >> 1;
  if (pair_idx < total_pairs) {
    int flat = pair_idx << 1;
    int n = flat / I;
    int i = flat - n * I;
    int row_base = n * (2 * I);

    const __nv_bfloat162 gate2 =
        *reinterpret_cast<const __nv_bfloat162*>(&out13[row_base + i]);
    const __nv_bfloat162 up2 =
        *reinterpret_cast<const __nv_bfloat162*>(&out13[row_base + I + i]);

    float2 gf = __bfloat1622float2(gate2);
    float2 uf = __bfloat1622float2(up2);
    float y0 = silu(gf.x) * uf.x;
    float y1 = silu(gf.y) * uf.y;
    *reinterpret_cast<__nv_bfloat162*>(&gated[n * I + i]) =
        __floats2bfloat162_rn(y0, y1);
  }

  if ((total & 1) && pair_idx == 0) {
    int flat = total - 1;
    int n = flat / I;
    int i = flat - n * I;
    int row_base = n * (2 * I);
    float gate = __bfloat162float(out13[row_base + i]);
    float up = __bfloat162float(out13[row_base + I + i]);
    gated[n * I + i] = __float2bfloat16(silu(gate) * up);
  }
}

__global__ void gelu_tanh_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float v = __bfloat162float(x[idx]);
  x[idx] = __float2bfloat16(gelu_tanh(v));
}

int text_gate_threads(int64_t n, int64_t i) {
  if (n == 1 && i == 6144) return 512;
  if (n == 690 && i == 6144) return 256;
  return 256;
}

void gemm_rowmajor_bf16(const torch::Tensor& a,
                        const torch::Tensor& b,
                        torch::Tensor& c) {
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
              "gemm_rowmajor_bf16 requires contiguous tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "gemm expects rank-2 tensors");
  TORCH_CHECK(a.dtype() == torch::kBFloat16 && b.dtype() == torch::kBFloat16 &&
                  c.dtype() == torch::kBFloat16,
              "gemm expects bfloat16 tensors");

  const int64_t n = a.size(0);
  const int64_t k = a.size(1);
  const int64_t k_b = b.size(0);
  const int64_t m = b.size(1);
  TORCH_CHECK(k == k_b, "gemm K mismatch");
  TORCH_CHECK(c.size(0) == n && c.size(1) == m, "gemm output shape mismatch");

  auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>());
  auto* b_ptr = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>());
  auto* c_ptr = reinterpret_cast<__nv_bfloat16*>(c.data_ptr<at::BFloat16>());

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasStatus_t status = cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      static_cast<int>(m),   // rows of C_colmajor
      static_cast<int>(n),   // cols of C_colmajor
      static_cast<int>(k),
      &alpha,
      b_ptr, CUDA_R_16BF, static_cast<int>(m),   // B_colmajor (from row-major B)
      a_ptr, CUDA_R_16BF, static_cast<int>(k),   // A_colmajor (from row-major A)
      &beta,
      c_ptr, CUDA_R_16BF, static_cast<int>(m),   // C_colmajor (row-major C)
      CUBLAS_COMPUTE_32F_FAST_16BF,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed");
}

void resolve_text_weight_layout(const torch::Tensor& w13_c,
                                const torch::Tensor& w2_c,
                                int64_t h,
                                torch::Tensor& w13_for_mm,
                                torch::Tensor& w2_for_mm,
                                int64_t& i) {
  i = 0;
  if (w13_c.size(0) == h && (w13_c.size(1) % 2 == 0)) {
    // Preferred packed W13: [H, 2I]
    i = w13_c.size(1) / 2;
    w13_for_mm = w13_c;
  } else if (w13_c.size(1) == h && (w13_c.size(0) % 2 == 0)) {
    // Legacy packed W13: [2I, H]
    i = w13_c.size(0) / 2;
    w13_for_mm = w13_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false,
                "w13 must be [H, 2I] packed or [2I, H] legacy layout");
  }

  if (w2_c.size(0) == i && w2_c.size(1) == h) {
    // Preferred packed W2: [I, H]
    w2_for_mm = w2_c;
  } else if (w2_c.size(0) == h && w2_c.size(1) == i) {
    // Legacy/original W2: [H, I]
    w2_for_mm = w2_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false,
                "w2 must be [I, H] packed or [H, I] legacy/original layout");
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
    // Preferred packed W1: [H, I]
    i = w1_c.size(1);
    w1_for_mm = w1_c;
  } else if (w1_c.size(1) == h) {
    // Legacy/original W1: [I, H]
    i = w1_c.size(0);
    w1_for_mm = w1_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false,
                "vision w1 must be [H, I] packed or [I, H] legacy/original");
  }

  if (w2_c.size(0) == i && w2_c.size(1) == h) {
    // Preferred packed W2: [I, H]
    w2_for_mm = w2_c;
  } else if (w2_c.size(0) == h && w2_c.size(1) == i) {
    // Legacy/original W2: [H, I]
    w2_for_mm = w2_c.transpose(0, 1);
  } else {
    TORCH_CHECK(false,
                "vision w2 must be [I, H] packed or [H, I] legacy/original");
  }
}

enum class V2Algo : int {
  kUnknown = 0,
  kLegacy = 1,
  kFusedK8 = 2,
  kFusedK16 = 3,
};

static int g_text_decode_v2_algo = static_cast<int>(V2Algo::kUnknown);
static int g_text_prefill_v2_algo = static_cast<int>(V2Algo::kUnknown);
static int g_vision_v2_algo = static_cast<int>(V2Algo::kUnknown);

template <int KStep>
struct MmaVariantTraits {};

template <>
struct MmaVariantTraits<8> {
  static constexpr int kK = 8;
};

template <>
struct MmaVariantTraits<16> {
  static constexpr int kK = 16;
};

__device__ __forceinline__ uint32_t cvt_shared_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}

__device__ __forceinline__ uint2 ldmatrix_x2_offset(uint32_t lane) {
  return make_uint2(lane & 7, (lane & 8) << 3);
}

__device__ __forceinline__ uint2 ldmatrix_x4_offset(uint32_t lane) {
  return make_uint2(lane & 15, (lane >> 4) << 3);
}

__device__ __forceinline__ void ldmatrix_sync_x1(const void* src_shared, uint32_t& a0) {
  uint32_t src = cvt_shared_u32(src_shared);
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(a0)
               : "r"(src));
}

__device__ __forceinline__ void ldmatrix_sync_x2(const void* src_shared, uint32_t& a0, uint32_t& a1) {
  uint32_t src = cvt_shared_u32(src_shared);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(a0), "=r"(a1)
               : "r"(src));
}

__device__ __forceinline__ void ldmatrix_sync_x4(const void* src_shared,
                                                 uint32_t& a0, uint32_t& a1,
                                                 uint32_t& a2, uint32_t& a3) {
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

__device__ __forceinline__ void mma_sync_m16n8k8_bf16(float& d0, float& d1, float& d2, float& d3,
                                                      uint32_t a0, uint32_t a1,
                                                      uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5}, "
               "{%6}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(b0));
}

template <int KStep>
__device__ __forceinline__ void mma_rowcol_m16n8(const __nv_bfloat16* a_tile,
                                                 const __nv_bfloat16* b_colmajor,
                                                 int lane,
                                                 float (&acc)[4]) {
  if constexpr (KStep == 16) {
    uint2 a_off = ldmatrix_x4_offset(static_cast<uint32_t>(lane));
    const __nv_bfloat16* a_base = a_tile + a_off.x * KStep + a_off.y;
    uint32_t a0, a1, a2, a3;
    ldmatrix_sync_x4(a_base, a0, a1, a2, a3);

    const __nv_bfloat16* b_base = b_colmajor + (lane & 7) * KStep + ((lane >> 3) & 1) * 8;
    uint32_t b0, b1;
    ldmatrix_sync_x2(b_base, b0, b1);
    mma_sync_m16n8k16_bf16(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
  } else {
    uint2 a_off = ldmatrix_x2_offset(static_cast<uint32_t>(lane));
    const __nv_bfloat16* a_base = a_tile + a_off.x * KStep + a_off.y;
    uint32_t a0, a1;
    ldmatrix_sync_x2(a_base, a0, a1);

    const __nv_bfloat16* b_base = b_colmajor + (lane & 7) * KStep;
    uint32_t b0;
    ldmatrix_sync_x1(b_base, b0);
    mma_sync_m16n8k8_bf16(acc[0], acc[1], acc[2], acc[3], a0, a1, b0);
  }
}

template <int KStep, int MTile, int WarpsPerRow>
__global__ void text_fused_tc_block_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w13,
    const __nv_bfloat16* __restrict__ w2,
    __nv_bfloat16* __restrict__ out,
    int N, int H, int I) {
  constexpr int kWarps = MTile * WarpsPerRow;
  constexpr int kThreads = 32 * kWarps;
  constexpr int kPerWarpElems = 64 * KStep;
  if (threadIdx.x >= kThreads) return;

  int row_base = blockIdx.x * MTile;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row_local = warp / WarpsPerRow;
  int warp_local = warp % WarpsPerRow;
  int row = row_base + row_local;
  int rows_this = N - row_base;
  if (rows_this > MTile) rows_this = MTile;
  if (rows_this < 0) rows_this = 0;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* x_sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [MTile, H]
  __nv_bfloat16* gated_sh = x_sh + MTile * H;                          // [MTile, I]
  __nv_bfloat16* scratch = gated_sh + MTile * I;                       // [warps, 64*KStep]

  __nv_bfloat16* warp_base = scratch + warp * kPerWarpElems;
  __nv_bfloat16* a_gate = warp_base;
  __nv_bfloat16* a_up = a_gate + 16 * KStep;
  __nv_bfloat16* b_x = a_up + 16 * KStep;
  __nv_bfloat16* a_out = b_x + 8 * KStep;
  __nv_bfloat16* b_g = a_out + 16 * KStep;

  for (int idx = threadIdx.x; idx < rows_this * H; idx += blockDim.x) {
    int lr = idx / H;
    int hc = idx - lr * H;
    x_sh[lr * H + hc] = x[(row_base + lr) * H + hc];
  }
  __syncthreads();

  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  if (row < N) {
    const __nv_bfloat16* x_row = x_sh + row_local * H;
    __nv_bfloat16* g_row = gated_sh + row_local * I;
    int i_tiles = I / 16;
    for (int tile = warp_local; tile < i_tiles; tile += WarpsPerRow) {
      float gate_acc[4] = {0.f, 0.f, 0.f, 0.f};
      float up_acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < H; k0 += KStep) {
        for (int t = lane; t < 16 * KStep; t += 32) {
          int r = t / KStep;
          int kk = t - r * KStep;
          int i_col = tile * 16 + r;
          a_gate[t] = w13[(k0 + kk) * (2 * I) + i_col];
          a_up[t] = w13[(k0 + kk) * (2 * I) + I + i_col];
        }
        for (int t = lane; t < 8 * KStep; t += 32) {
          int col = t / KStep;
          int kk = t - col * KStep;
          b_x[t] = (col == 0) ? x_row[k0 + kk] : zero;
        }
        __syncwarp();
        mma_rowcol_m16n8<KStep>(a_gate, b_x, lane, gate_acc);
        mma_rowcol_m16n8<KStep>(a_up, b_x, lane, up_acc);
        __syncwarp();
      }

      if ((lane & 3) == 0) {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        g_row[tile * 16 + r0] = __float2bfloat16(silu(gate_acc[0]) * up_acc[0]);
        g_row[tile * 16 + r1] = __float2bfloat16(silu(gate_acc[2]) * up_acc[2]);
      }
    }
  }
  __syncthreads();

  if (row < N) {
    const __nv_bfloat16* g_row = gated_sh + row_local * I;
    int h_tiles = H / 16;
    for (int h_tile = warp_local; h_tile < h_tiles; h_tile += WarpsPerRow) {
      int h_base = h_tile * 16;
      float out_acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < I; k0 += KStep) {
        for (int t = lane; t < 16 * KStep; t += 32) {
          int r = t / KStep;
          int kk = t - r * KStep;
          a_out[t] = w2[(k0 + kk) * H + h_base + r];
        }
        for (int t = lane; t < 8 * KStep; t += 32) {
          int col = t / KStep;
          int kk = t - col * KStep;
          b_g[t] = (col == 0) ? g_row[k0 + kk] : zero;
        }
        __syncwarp();
        mma_rowcol_m16n8<KStep>(a_out, b_g, lane, out_acc);
        __syncwarp();
      }

      if ((lane & 3) == 0) {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        out[row * H + h_base + r0] = __float2bfloat16(out_acc[0]);
        out[row * H + h_base + r1] = __float2bfloat16(out_acc[2]);
      }
    }
  }
}

template <int KStep, int MTile, int WarpsPerRow>
__global__ void vision_fused_tc_block_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w1,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ w2,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ out,
    int N, int H, int I) {
  constexpr int kWarps = MTile * WarpsPerRow;
  constexpr int kThreads = 32 * kWarps;
  constexpr int kPerWarpElems = 48 * KStep;
  if (threadIdx.x >= kThreads) return;

  int row_base = blockIdx.x * MTile;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row_local = warp / WarpsPerRow;
  int warp_local = warp % WarpsPerRow;
  int row = row_base + row_local;
  int rows_this = N - row_base;
  if (rows_this > MTile) rows_this = MTile;
  if (rows_this < 0) rows_this = 0;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* x_sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [MTile, H]
  __nv_bfloat16* act_sh = x_sh + MTile * H;                            // [MTile, I]
  __nv_bfloat16* scratch = act_sh + MTile * I;                         // [warps, 48*KStep]

  __nv_bfloat16* warp_base = scratch + warp * kPerWarpElems;
  __nv_bfloat16* a_fc1 = warp_base;
  __nv_bfloat16* b_x = a_fc1 + 16 * KStep;
  __nv_bfloat16* a_out = b_x + 8 * KStep;
  __nv_bfloat16* b_act = a_out + 16 * KStep;

  for (int idx = threadIdx.x; idx < rows_this * H; idx += blockDim.x) {
    int lr = idx / H;
    int hc = idx - lr * H;
    x_sh[lr * H + hc] = x[(row_base + lr) * H + hc];
  }
  __syncthreads();

  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  if (row < N) {
    const __nv_bfloat16* x_row = x_sh + row_local * H;
    __nv_bfloat16* a_row = act_sh + row_local * I;
    int i_tiles = I / 16;
    for (int tile = warp_local; tile < i_tiles; tile += WarpsPerRow) {
      float fc1_acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < H; k0 += KStep) {
        for (int t = lane; t < 16 * KStep; t += 32) {
          int r = t / KStep;
          int kk = t - r * KStep;
          int i_col = tile * 16 + r;
          a_fc1[t] = w1[(k0 + kk) * I + i_col];
        }
        for (int t = lane; t < 8 * KStep; t += 32) {
          int col = t / KStep;
          int kk = t - col * KStep;
          b_x[t] = (col == 0) ? x_row[k0 + kk] : zero;
        }
        __syncwarp();
        mma_rowcol_m16n8<KStep>(a_fc1, b_x, lane, fc1_acc);
        __syncwarp();
      }

      if ((lane & 3) == 0) {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        int i0 = tile * 16 + r0;
        int i1 = tile * 16 + r1;
        float v0 = fc1_acc[0] + __bfloat162float(b1[i0]);
        float v1 = fc1_acc[2] + __bfloat162float(b1[i1]);
        a_row[i0] = __float2bfloat16(gelu_tanh(v0));
        a_row[i1] = __float2bfloat16(gelu_tanh(v1));
      }
    }
  }
  __syncthreads();

  if (row < N) {
    const __nv_bfloat16* a_row = act_sh + row_local * I;
    int h_tiles = H / 16;
    for (int h_tile = warp_local; h_tile < h_tiles; h_tile += WarpsPerRow) {
      int h_base = h_tile * 16;
      float out_acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < I; k0 += KStep) {
        for (int t = lane; t < 16 * KStep; t += 32) {
          int r = t / KStep;
          int kk = t - r * KStep;
          a_out[t] = w2[(k0 + kk) * H + h_base + r];
        }
        for (int t = lane; t < 8 * KStep; t += 32) {
          int col = t / KStep;
          int kk = t - col * KStep;
          b_act[t] = (col == 0) ? a_row[k0 + kk] : zero;
        }
        __syncwarp();
        mma_rowcol_m16n8<KStep>(a_out, b_act, lane, out_acc);
        __syncwarp();
      }

      if ((lane & 3) == 0) {
        int r0 = lane >> 2;
        int r1 = r0 + 8;
        int h0 = h_base + r0;
        int h1 = h_base + r1;
        out[row * H + h0] = __float2bfloat16(out_acc[0] + __bfloat162float(b2[h0]));
        out[row * H + h1] = __float2bfloat16(out_acc[2] + __bfloat162float(b2[h1]));
      }
    }
  }
}

template <int KStep>
void launch_text_fused_decode_kernel(const torch::Tensor& x,
                                     const torch::Tensor& w13,
                                     const torch::Tensor& w2,
                                     torch::Tensor& out,
                                     cudaStream_t stream) {
  constexpr int kMTile = 1;
  constexpr int kWarpsPerRow = 4;
  constexpr int kWarps = kMTile * kWarpsPerRow;
  constexpr int kThreads = 32 * kWarps;
  const int n = static_cast<int>(x.size(0));
  const int h = static_cast<int>(x.size(1));
  const int i = static_cast<int>(w2.size(0));
  auto x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto w13_ptr = reinterpret_cast<const __nv_bfloat16*>(w13.data_ptr<at::BFloat16>());
  auto w2_ptr = reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr<at::BFloat16>());
  auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  size_t smem = static_cast<size_t>(kMTile * h + kMTile * i + kWarps * 64 * MmaVariantTraits<KStep>::kK) *
                sizeof(__nv_bfloat16);
  int grid = (n + kMTile - 1) / kMTile;
  static bool smem_attr_set = false;
  if (!smem_attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        text_fused_tc_block_kernel<KStep, kMTile, kWarpsPerRow>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));
    smem_attr_set = true;
  }
  text_fused_tc_block_kernel<KStep, kMTile, kWarpsPerRow><<<grid, kThreads, smem, stream>>>(
      x_ptr, w13_ptr, w2_ptr, out_ptr, n, h, i);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int KStep>
void launch_text_fused_prefill_kernel(const torch::Tensor& x,
                                      const torch::Tensor& w13,
                                      const torch::Tensor& w2,
                                      torch::Tensor& out,
                                      cudaStream_t stream) {
  constexpr int kMTile = 4;
  constexpr int kWarpsPerRow = 2;
  constexpr int kWarps = kMTile * kWarpsPerRow;
  constexpr int kThreads = 32 * kWarps;
  const int n = static_cast<int>(x.size(0));
  const int h = static_cast<int>(x.size(1));
  const int i = static_cast<int>(w2.size(0));
  auto x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto w13_ptr = reinterpret_cast<const __nv_bfloat16*>(w13.data_ptr<at::BFloat16>());
  auto w2_ptr = reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr<at::BFloat16>());
  auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  size_t smem = static_cast<size_t>(kMTile * h + kMTile * i + kWarps * 64 * MmaVariantTraits<KStep>::kK) *
                sizeof(__nv_bfloat16);
  int grid = (n + kMTile - 1) / kMTile;
  static bool smem_attr_set = false;
  if (!smem_attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        text_fused_tc_block_kernel<KStep, kMTile, kWarpsPerRow>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));
    smem_attr_set = true;
  }
  text_fused_tc_block_kernel<KStep, kMTile, kWarpsPerRow><<<grid, kThreads, smem, stream>>>(
      x_ptr, w13_ptr, w2_ptr, out_ptr, n, h, i);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <int KStep>
void launch_vision_fused_kernel(const torch::Tensor& x,
                                const torch::Tensor& w1,
                                const torch::Tensor& b1,
                                const torch::Tensor& w2,
                                const torch::Tensor& b2,
                                torch::Tensor& out,
                                cudaStream_t stream) {
  constexpr int kMTile = 4;
  constexpr int kWarpsPerRow = 2;
  constexpr int kWarps = kMTile * kWarpsPerRow;
  constexpr int kThreads = 32 * kWarps;
  const int n = static_cast<int>(x.size(0));
  const int h = static_cast<int>(x.size(1));
  const int i = static_cast<int>(w1.size(1));
  auto x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  auto w1_ptr = reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr<at::BFloat16>());
  auto b1_ptr = reinterpret_cast<const __nv_bfloat16*>(b1.data_ptr<at::BFloat16>());
  auto w2_ptr = reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr<at::BFloat16>());
  auto b2_ptr = reinterpret_cast<const __nv_bfloat16*>(b2.data_ptr<at::BFloat16>());
  auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  size_t smem = static_cast<size_t>(kMTile * h + kMTile * i + kWarps * 48 * MmaVariantTraits<KStep>::kK) *
                sizeof(__nv_bfloat16);
  int grid = (n + kMTile - 1) / kMTile;
  static bool smem_attr_set = false;
  if (!smem_attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        vision_fused_tc_block_kernel<KStep, kMTile, kWarpsPerRow>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));
    smem_attr_set = true;
  }
  vision_fused_tc_block_kernel<KStep, kMTile, kWarpsPerRow><<<grid, kThreads, smem, stream>>>(
      x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr, n, h, i);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <typename RunFn>
float bench_ms(cudaStream_t stream, RunFn&& fn, int warmup = 1, int iters = 1) {
  for (int i = 0; i < warmup; ++i) fn();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, end;
  C10_CUDA_CHECK(cudaEventCreate(&start));
  C10_CUDA_CHECK(cudaEventCreate(&end));
  C10_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) fn();
  C10_CUDA_CHECK(cudaEventRecord(end, stream));
  C10_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.0f;
  C10_CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  C10_CUDA_CHECK(cudaEventDestroy(start));
  C10_CUDA_CHECK(cudaEventDestroy(end));
  return ms / static_cast<float>(iters);
}

bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  return status != cudaStreamCaptureStatusNone;
}

int read_env_int(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return default_value;
  char* end = nullptr;
  long val = std::strtol(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0')) return default_value;
  if (val < 1) return default_value;
  if (val > 1024) return 1024;
  return static_cast<int>(val);
}

V2Algo forced_v2_algo_from_env() {
  const char* raw = std::getenv("FLASHDECODE_V2_FORCE_ALGO");
  if (raw == nullptr || *raw == '\0') return V2Algo::kUnknown;
  if (std::strcmp(raw, "fallback") == 0) return V2Algo::kLegacy;
  if (std::strcmp(raw, "k8") == 0) return V2Algo::kFusedK8;
  if (std::strcmp(raw, "k16") == 0) return V2Algo::kFusedK16;
  return V2Algo::kUnknown;
}

}  // namespace

torch::Tensor flashdecodeffn_mlp_cuda(torch::Tensor x,
                                      torch::Tensor w1,
                                      torch::Tensor w3,
                                      torch::Tensor w2,
                                      int64_t split_k) {
  (void)split_k;
  auto w13 = flashdecodeffn_pack_w13_cuda(w1, w3);
  auto w2_packed = flashdecodeffn_pack_w2_cuda(w2);
  return flashdecodeffn_mlp_packed_cuda(x, w13, w2_packed);
}

torch::Tensor flashdecodeffn_pack_w13_cuda(torch::Tensor w1,
                                           torch::Tensor w3) {
  TORCH_CHECK(w1.is_cuda() && w3.is_cuda(), "w1 and w3 must be CUDA tensors");
  TORCH_CHECK(w1.dtype() == torch::kBFloat16 && w3.dtype() == torch::kBFloat16,
              "w1 and w3 must be torch.bfloat16");
  TORCH_CHECK(w1.dim() == 2 && w3.dim() == 2, "w1 and w3 must be rank-2");
  TORCH_CHECK(w1.sizes() == w3.sizes(), "w1 and w3 shapes must match");
  // Pack as [H, 2I] for direct GEMM use.
  auto w13 = torch::cat({w1.contiguous(), w3.contiguous()}, 0).transpose(0, 1);
  return w13.contiguous();
}

torch::Tensor flashdecodeffn_pack_w2_cuda(torch::Tensor w2) {
  TORCH_CHECK(w2.is_cuda(), "w2 must be CUDA tensor");
  TORCH_CHECK(w2.dtype() == torch::kBFloat16, "w2 must be torch.bfloat16");
  TORCH_CHECK(w2.dim() == 2, "w2 must be rank-2");
  // Pack as [I, H] for direct GEMM use.
  return w2.contiguous().transpose(0, 1).contiguous();
}

torch::Tensor flashdecodeffn_mlp_packed_cuda(torch::Tensor x,
                                             torch::Tensor w13,
                                             torch::Tensor w2) {
  auto x_c = x.contiguous();
  auto w13_c = w13.contiguous();
  auto w2_c = w2.contiguous();

  const int64_t n = x_c.size(0);
  const int64_t h = x_c.size(1);
  const int64_t i = (w13_c.size(0) == h) ? (w13_c.size(1) / 2) : (w13_c.size(0) / 2);

  auto opts_bf16 = x_c.options().dtype(torch::kBFloat16);
  auto out13 = torch::empty({n, 2 * i}, opts_bf16);
  auto gated = torch::empty({n, i}, opts_bf16);
  auto out = torch::empty({n, h}, opts_bf16);

  flashdecodeffn_mlp_packed_into_cuda(x_c, w13_c, w2_c, out13, gated, out);
  return out;
}

void flashdecodeffn_mlp_packed_into_cuda(torch::Tensor x,
                                         torch::Tensor w13,
                                         torch::Tensor w2,
                                         torch::Tensor out13,
                                         torch::Tensor gated,
                                         torch::Tensor out) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
  TORCH_CHECK(w13.is_cuda() && w2.is_cuda(), "weights must be CUDA tensors");
  TORCH_CHECK(out13.is_cuda() && gated.is_cuda() && out.is_cuda(),
              "out13/gated/out must be CUDA tensors");
  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be torch.bfloat16");
  TORCH_CHECK(w13.dtype() == torch::kBFloat16 && w2.dtype() == torch::kBFloat16,
              "weights must be torch.bfloat16");
  TORCH_CHECK(out13.dtype() == torch::kBFloat16 &&
                  gated.dtype() == torch::kBFloat16 &&
                  out.dtype() == torch::kBFloat16,
              "out13/gated/out must be torch.bfloat16");
  TORCH_CHECK(x.dim() == 2, "x must be [N, H]");
  TORCH_CHECK(w13.dim() == 2 && w2.dim() == 2, "weights must be rank-2");
  TORCH_CHECK(out13.dim() == 2 && gated.dim() == 2 && out.dim() == 2,
              "out13/gated/out must be rank-2");

  auto x_c = x.contiguous();
  auto w13_c = w13.contiguous();
  auto w2_c = w2.contiguous();

  TORCH_CHECK(out13.is_contiguous() && gated.is_contiguous() && out.is_contiguous(),
              "out13/gated/out must be contiguous");

  const int64_t n = x_c.size(0);
  const int64_t h = x_c.size(1);

  torch::Tensor w13_for_mm;
  torch::Tensor w2_for_mm;
  int64_t i = 0;
  resolve_text_weight_layout(w13_c, w2_c, h, w13_for_mm, w2_for_mm, i);

  TORCH_CHECK(out13.size(0) == n && out13.size(1) == 2 * i,
              "out13 must be [N, 2I]");
  TORCH_CHECK(gated.size(0) == n && gated.size(1) == i,
              "gated must be [N, I]");
  TORCH_CHECK(out.size(0) == n && out.size(1) == h,
              "out must be [N, H]");

  gemm_rowmajor_bf16(x_c, w13_for_mm, out13);

  auto out13_ptr =
      reinterpret_cast<const __nv_bfloat16*>(out13.data_ptr<at::BFloat16>());
  auto gated_ptr =
      reinterpret_cast<__nv_bfloat16*>(gated.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int total = static_cast<int>(n * i);
  const int total_pairs = (total + 1) >> 1;
  const int threads = text_gate_threads(n, i);
  silu_gate_to_bf16_kernel<<<(total_pairs + threads - 1) / threads, threads, 0, stream>>>(
      out13_ptr, gated_ptr, static_cast<int>(n), static_cast<int>(i));
  C10_CUDA_CHECK(cudaGetLastError());

  gemm_rowmajor_bf16(gated, w2_for_mm, out);
}

torch::Tensor flashdecodeffn_pack_vision_w1_cuda(torch::Tensor w1) {
  TORCH_CHECK(w1.is_cuda(), "vision w1 must be CUDA tensor");
  TORCH_CHECK(w1.dtype() == torch::kBFloat16, "vision w1 must be bfloat16");
  TORCH_CHECK(w1.dim() == 2, "vision w1 must be rank-2");
  // Pack as [H, I] from original [I, H].
  return w1.contiguous().transpose(0, 1).contiguous();
}

torch::Tensor flashdecodeffn_pack_vision_w2_cuda(torch::Tensor w2) {
  TORCH_CHECK(w2.is_cuda(), "vision w2 must be CUDA tensor");
  TORCH_CHECK(w2.dtype() == torch::kBFloat16, "vision w2 must be bfloat16");
  TORCH_CHECK(w2.dim() == 2, "vision w2 must be rank-2");
  // Pack as [I, H] from original [H, I].
  return w2.contiguous().transpose(0, 1).contiguous();
}

torch::Tensor flashdecodeffn_vision_mlp_packed_cuda(torch::Tensor x,
                                                    torch::Tensor w1,
                                                    torch::Tensor b1,
                                                    torch::Tensor w2,
                                                    torch::Tensor b2) {
  auto x_c = x.contiguous();
  auto w1_c = w1.contiguous();
  auto w2_c = w2.contiguous();
  const int64_t n = x_c.size(0);
  const int64_t h = x_c.size(1);
  const int64_t i = (w1_c.size(0) == h) ? w1_c.size(1) : w1_c.size(0);
  auto opts_bf16 = x_c.options().dtype(torch::kBFloat16);
  auto fc1 = torch::empty({n, i}, opts_bf16);
  auto out = torch::empty({n, h}, opts_bf16);
  flashdecodeffn_vision_mlp_packed_into_cuda(x_c, w1_c, b1, w2_c, b2, fc1, out);
  return out;
}

void flashdecodeffn_vision_mlp_packed_into_cuda(torch::Tensor x,
                                                torch::Tensor w1,
                                                torch::Tensor b1,
                                                torch::Tensor w2,
                                                torch::Tensor b2,
                                                torch::Tensor fc1,
                                                torch::Tensor out) {
  TORCH_CHECK(x.is_cuda() && w1.is_cuda() && b1.is_cuda() && w2.is_cuda() && b2.is_cuda(),
              "vision tensors must be CUDA");
  TORCH_CHECK(fc1.is_cuda() && out.is_cuda(), "vision workspace/output must be CUDA");
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && w1.dtype() == torch::kBFloat16 &&
                  b1.dtype() == torch::kBFloat16 && w2.dtype() == torch::kBFloat16 &&
                  b2.dtype() == torch::kBFloat16 && fc1.dtype() == torch::kBFloat16 &&
                  out.dtype() == torch::kBFloat16,
              "vision tensors must be bfloat16");
  TORCH_CHECK(x.dim() == 2, "vision x must be [N, H]");
  TORCH_CHECK(w1.dim() == 2 && w2.dim() == 2, "vision weights must be rank-2");
  TORCH_CHECK(b1.dim() == 1 && b2.dim() == 1, "vision bias must be rank-1");
  TORCH_CHECK(fc1.dim() == 2 && out.dim() == 2, "vision fc1/out must be rank-2");

  auto x_c = x.contiguous();
  auto w1_c = w1.contiguous();
  auto b1_c = b1.contiguous();
  auto w2_c = w2.contiguous();
  auto b2_c = b2.contiguous();

  TORCH_CHECK(fc1.is_contiguous() && out.is_contiguous(),
              "vision fc1/out must be contiguous");

  const int64_t n = x_c.size(0);
  const int64_t h = x_c.size(1);
  torch::Tensor w1_for_mm;
  torch::Tensor w2_for_mm;
  int64_t i = 0;
  resolve_vision_weight_layout(w1_c, w2_c, h, w1_for_mm, w2_for_mm, i);

  TORCH_CHECK(b1_c.size(0) == i, "vision b1 must be [I]");
  TORCH_CHECK(b2_c.size(0) == h, "vision b2 must be [H]");
  TORCH_CHECK(fc1.size(0) == n && fc1.size(1) == i, "vision fc1 must be [N, I]");
  TORCH_CHECK(out.size(0) == n && out.size(1) == h, "vision out must be [N, H]");

  at::addmm_out(fc1, b1_c, x_c, w1_for_mm);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto fc1_ptr = reinterpret_cast<__nv_bfloat16*>(fc1.data_ptr<at::BFloat16>());
  int total = static_cast<int>(n * i);
  constexpr int kThreads = 256;
  gelu_tanh_inplace_bf16_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      fc1_ptr, total);
  C10_CUDA_CHECK(cudaGetLastError());

  at::addmm_out(out, b2_c, fc1, w2_for_mm);
}

torch::Tensor flashdecodeffn_text_decode_v2_cuda(torch::Tensor x,
                                                 torch::Tensor w13,
                                                 torch::Tensor w2) {
  TORCH_CHECK(x.is_cuda() && w13.is_cuda() && w2.is_cuda(),
              "x/w13/w2 must be CUDA tensors");
  TORCH_CHECK(x.dtype() == torch::kBFloat16 &&
                  w13.dtype() == torch::kBFloat16 &&
                  w2.dtype() == torch::kBFloat16,
              "x/w13/w2 must be torch.bfloat16");
  TORCH_CHECK(x.dim() == 2, "x must be [N, H]");

  auto x_c = x.contiguous();
  auto w13_c = w13.contiguous();
  auto w2_c = w2.contiguous();
  const int64_t n = x_c.size(0);
  const int64_t h = x_c.size(1);
  TORCH_CHECK(n == 1 && h == 2048, "text decode v2 requires x shape [1, 2048]");

  torch::Tensor w13_for_mm;
  torch::Tensor w2_for_mm;
  int64_t i = 0;
  resolve_text_weight_layout(w13_c, w2_c, h, w13_for_mm, w2_for_mm, i);
  TORCH_CHECK(i == 6144, "text decode v2 requires I=6144");
  auto w13_dense = w13_for_mm.contiguous();
  auto w2_dense = w2_for_mm.contiguous();

  auto opts_bf16 = x_c.options().dtype(torch::kBFloat16);
  auto out = torch::empty({n, h}, opts_bf16);
  auto out13 = torch::empty({n, 2 * i}, opts_bf16);
  auto gated = torch::empty({n, i}, opts_bf16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto run_legacy = [&]() {
    flashdecodeffn_mlp_packed_into_cuda(x_c, w13_dense, w2_dense, out13, gated, out);
  };
  auto run_k8 = [&]() {
    launch_text_fused_decode_kernel<8>(x_c, w13_dense, w2_dense, out, stream);
  };
  auto run_k16 = [&]() {
    launch_text_fused_decode_kernel<16>(x_c, w13_dense, w2_dense, out, stream);
  };

  V2Algo algo = static_cast<V2Algo>(g_text_decode_v2_algo);
  V2Algo forced_algo = forced_v2_algo_from_env();
  if (forced_algo != V2Algo::kUnknown) {
    algo = forced_algo;
    g_text_decode_v2_algo = static_cast<int>(algo);
  } else if (algo == V2Algo::kUnknown) {
    if (stream_is_capturing(stream)) {
      algo = V2Algo::kLegacy;
    } else {
      int tune_iters = read_env_int("FLASHDECODE_V2_AUTOTUNE_ITERS", 8);
      int warmup = (tune_iters >= 4) ? 2 : 1;
      float t_legacy = bench_ms(stream, run_legacy, warmup, tune_iters);
      float t_k8 = bench_ms(stream, run_k8, warmup, tune_iters);
      float t_k16 = bench_ms(stream, run_k16, warmup, tune_iters);

      float best = t_legacy;
      algo = V2Algo::kLegacy;
      if (t_k8 < best) {
        best = t_k8;
        algo = V2Algo::kFusedK8;
      }
      if (t_k16 < best) {
        algo = V2Algo::kFusedK16;
      }
    }
    g_text_decode_v2_algo = static_cast<int>(algo);
  }

  switch (algo) {
    case V2Algo::kFusedK8:
      run_k8();
      break;
    case V2Algo::kFusedK16:
      run_k16();
      break;
    case V2Algo::kLegacy:
    default:
      run_legacy();
      break;
  }
  return out;
}

torch::Tensor flashdecodeffn_text_prefill_v2_cuda(torch::Tensor x,
                                                  torch::Tensor w13,
                                                  torch::Tensor w2) {
  TORCH_CHECK(x.is_cuda() && w13.is_cuda() && w2.is_cuda(),
              "x/w13/w2 must be CUDA tensors");
  TORCH_CHECK(x.dtype() == torch::kBFloat16 &&
                  w13.dtype() == torch::kBFloat16 &&
                  w2.dtype() == torch::kBFloat16,
              "x/w13/w2 must be torch.bfloat16");
  TORCH_CHECK(x.dim() == 2, "x must be [N, H]");

  auto x_c = x.contiguous();
  auto w13_c = w13.contiguous();
  auto w2_c = w2.contiguous();
  const int64_t n = x_c.size(0);
  const int64_t h = x_c.size(1);
  TORCH_CHECK(n == 690 && h == 2048,
              "text prefill v2 requires x shape [690, 2048]");

  torch::Tensor w13_for_mm;
  torch::Tensor w2_for_mm;
  int64_t i = 0;
  resolve_text_weight_layout(w13_c, w2_c, h, w13_for_mm, w2_for_mm, i);
  TORCH_CHECK(i == 6144, "text prefill v2 requires I=6144");
  auto w13_dense = w13_for_mm.contiguous();
  auto w2_dense = w2_for_mm.contiguous();

  auto opts_bf16 = x_c.options().dtype(torch::kBFloat16);
  auto out = torch::empty({n, h}, opts_bf16);
  auto out13 = torch::empty({n, 2 * i}, opts_bf16);
  auto gated = torch::empty({n, i}, opts_bf16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto run_legacy = [&]() {
    flashdecodeffn_mlp_packed_into_cuda(x_c, w13_dense, w2_dense, out13, gated, out);
  };
  auto run_k8 = [&]() {
    launch_text_fused_prefill_kernel<8>(x_c, w13_dense, w2_dense, out, stream);
  };
  auto run_k16 = [&]() {
    launch_text_fused_prefill_kernel<16>(x_c, w13_dense, w2_dense, out, stream);
  };

  V2Algo algo = static_cast<V2Algo>(g_text_prefill_v2_algo);
  V2Algo forced_algo = forced_v2_algo_from_env();
  if (forced_algo != V2Algo::kUnknown) {
    algo = forced_algo;
    g_text_prefill_v2_algo = static_cast<int>(algo);
  } else if (algo == V2Algo::kUnknown) {
    if (stream_is_capturing(stream)) {
      algo = V2Algo::kLegacy;
    } else {
      int tune_iters = read_env_int("FLASHDECODE_V2_AUTOTUNE_ITERS", 6);
      int warmup = (tune_iters >= 4) ? 2 : 1;
      float t_legacy = bench_ms(stream, run_legacy, warmup, tune_iters);
      float t_k8 = bench_ms(stream, run_k8, warmup, tune_iters);
      float t_k16 = bench_ms(stream, run_k16, warmup, tune_iters);

      float best = t_legacy;
      algo = V2Algo::kLegacy;
      if (t_k8 < best) {
        best = t_k8;
        algo = V2Algo::kFusedK8;
      }
      if (t_k16 < best) {
        algo = V2Algo::kFusedK16;
      }
    }
    g_text_prefill_v2_algo = static_cast<int>(algo);
  }

  switch (algo) {
    case V2Algo::kFusedK8:
      run_k8();
      break;
    case V2Algo::kFusedK16:
      run_k16();
      break;
    case V2Algo::kLegacy:
    default:
      run_legacy();
      break;
  }
  return out;
}

torch::Tensor flashdecodeffn_text_forward_v2_cuda(torch::Tensor x,
                                                  torch::Tensor w13,
                                                  torch::Tensor w2) {
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  const int64_t n = x.size(0);
  const int64_t h = x.size(1);
  auto w13_c = w13.contiguous();
  int64_t i = 0;
  if (w13_c.size(0) == h) i = w13_c.size(1) / 2;
  if (w13_c.size(1) == h) i = w13_c.size(0) / 2;

  if (n == 1 && h == 2048 && i == 6144) {
    return flashdecodeffn_text_decode_v2_cuda(x, w13, w2);
  }
  if (n == 690 && h == 2048 && i == 6144) {
    return flashdecodeffn_text_prefill_v2_cuda(x, w13, w2);
  }
  return flashdecodeffn_mlp_packed_cuda(x, w13, w2);
}

torch::Tensor flashdecodeffn_vision_forward_v2_cuda(torch::Tensor x,
                                                    torch::Tensor w1,
                                                    torch::Tensor b1,
                                                    torch::Tensor w2,
                                                    torch::Tensor b2) {
  TORCH_CHECK(x.dim() == 2, "x must be rank-2");
  const int64_t n = x.size(0);
  const int64_t h = x.size(1);
  auto x_c = x.contiguous();
  auto w1_c = w1.contiguous();
  auto b1_c = b1.contiguous();
  auto w2_c = w2.contiguous();
  auto b2_c = b2.contiguous();

  torch::Tensor w1_for_mm;
  torch::Tensor w2_for_mm;
  int64_t resolved_i = 0;
  resolve_vision_weight_layout(w1_c, w2_c, h, w1_for_mm, w2_for_mm, resolved_i);

  if (!(n == 2688 && h == 1024 && resolved_i == 4096)) {
    return flashdecodeffn_vision_mlp_packed_cuda(x_c, w1_c, b1_c, w2_c, b2_c);
  }

  auto w1_dense = w1_for_mm.contiguous();
  auto w2_dense = w2_for_mm.contiguous();
  auto opts_bf16 = x_c.options().dtype(torch::kBFloat16);
  auto out = torch::empty({n, h}, opts_bf16);
  auto fc1 = torch::empty({n, resolved_i}, opts_bf16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto run_legacy = [&]() {
    flashdecodeffn_vision_mlp_packed_into_cuda(
        x_c, w1_dense, b1_c, w2_dense, b2_c, fc1, out);
  };
  auto run_k8 = [&]() {
    launch_vision_fused_kernel<8>(x_c, w1_dense, b1_c, w2_dense, b2_c, out, stream);
  };
  auto run_k16 = [&]() {
    launch_vision_fused_kernel<16>(x_c, w1_dense, b1_c, w2_dense, b2_c, out, stream);
  };

  V2Algo algo = static_cast<V2Algo>(g_vision_v2_algo);
  V2Algo forced_algo = forced_v2_algo_from_env();
  if (forced_algo != V2Algo::kUnknown) {
    algo = forced_algo;
    g_vision_v2_algo = static_cast<int>(algo);
  } else if (algo == V2Algo::kUnknown) {
    if (stream_is_capturing(stream)) {
      algo = V2Algo::kLegacy;
    } else {
      int tune_iters = read_env_int("FLASHDECODE_V2_AUTOTUNE_ITERS", 6);
      int warmup = (tune_iters >= 4) ? 2 : 1;
      float t_legacy = bench_ms(stream, run_legacy, warmup, tune_iters);
      float t_k8 = bench_ms(stream, run_k8, warmup, tune_iters);
      float t_k16 = bench_ms(stream, run_k16, warmup, tune_iters);

      float best = t_legacy;
      algo = V2Algo::kLegacy;
      if (t_k8 < best) {
        best = t_k8;
        algo = V2Algo::kFusedK8;
      }
      if (t_k16 < best) {
        algo = V2Algo::kFusedK16;
      }
    }
    g_vision_v2_algo = static_cast<int>(algo);
  }

  switch (algo) {
    case V2Algo::kFusedK8:
      run_k8();
      break;
    case V2Algo::kFusedK16:
      run_k16();
      break;
    case V2Algo::kLegacy:
    default:
      run_legacy();
      break;
  }
  return out;
}

int64_t flashdecodeffn_get_text_decode_v2_algo_cuda() {
  return static_cast<int64_t>(g_text_decode_v2_algo);
}

int64_t flashdecodeffn_get_text_prefill_v2_algo_cuda() {
  return static_cast<int64_t>(g_text_prefill_v2_algo);
}

int64_t flashdecodeffn_get_vision_v2_algo_cuda() {
  return static_cast<int64_t>(g_vision_v2_algo);
}
