import ctypes
import importlib.util
import os

# os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
# os.environ["PATH"] = os.environ["CUDA_PATH"] + "/bin:" + os.environ.get("PATH", "")
# ctypes.CDLL("/usr/local/PPU_SDK/targets/x86_64-linux/lib/libhggcrt1.so", mode=ctypes.RTLD_GLOBAL)

import tilelang
import tilelang.language as T
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
from torch.utils.cpp_extension import load_inline
import os

# 强制设置算力，避免警告并优化编译
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"


import torch
from torch.utils.cpp_extension import load_inline
import os

# 强制设置算力，启用 WMMA 需要至少 7.0 以上，A100 是 8.0
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_ln_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor w_reduced);
"""

cuda_source = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

// Tensor Core 基础指令块尺寸
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 线程块 (Block) 的宏观 Tile 尺寸 
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32

// 每个 Block 使用 4 个 Warp (128 线程)
#define THREADS_PER_BLOCK 128

__global__ void fused_ln_wmma_kernel(
    const half* __restrict__ X,          // [M, K]
    const half* __restrict__ W,          // [N_out, K]  (注意：在内存中实际是 N_out 行 K 列)
    const half* __restrict__ bias,       // [N_out]
    const float* __restrict__ W_red,     // [N_out]
    half* __restrict__ Out,              // [M, N_out]
    int M, int K_dim, int N_out
) {
    // 1. 申请共享内存 (加上 +8 Padding 彻底打碎 Bank Conflict)
    __shared__ half s_X[BLOCK_M][BLOCK_K + 8]; 
    __shared__ half s_W[BLOCK_N][BLOCK_K + 8]; 
    __shared__ float s_Acc[BLOCK_M][BLOCK_N]; 
    __shared__ float s_Mean[BLOCK_M];
    __shared__ float s_Rstd[BLOCK_M];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tid = threadIdx.x;
    
    int row_start = bx * BLOCK_M;
    int col_start = by * BLOCK_N;

    // 2. 声明 Tensor Core 寄存器碎片
    // X 作为 A 矩阵 (Row-Major), W 作为 B 矩阵 (由于物理存储是 [N,K]，相对于 GEMM 刚好是 Col-Major)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b0, b1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_00, acc_01, acc_10, acc_11;
    
    wmma::fill_fragment(acc_00, 0.0f);
    wmma::fill_fragment(acc_01, 0.0f);
    wmma::fill_fragment(acc_10, 0.0f);
    wmma::fill_fragment(acc_11, 0.0f);

    // 3. 统计量局部累加器 (每个线程追踪自己负责的那部分行的和)
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;

    // 计算当前线程负责搬运的行和列
    // X 矩阵：128 线程搬 64x32。每行分配 2 个线程，每个线程搬 16 个 half (即 2 个 float4)
    int x_r = tid / 2;             // 0~63
    int x_c_base = (tid % 2) * 8;  // 0 或 8 (由于 float4 一次读8个half，所以是0~7 和 8~15，以及 16~23 和 24~31)

    // W 矩阵：同理
    int w_r = tid / 2;
    int w_c_base = (tid % 2) * 8;

    // 4. K 维大循环
    for (int k_step = 0; k_step < K_dim; k_step += BLOCK_K) {
        
        // ==========================================
        // 动作 A：向量化加载 (float4) + 计算隐藏
        // ==========================================
        // 加载 X 的前半部分 (8 个 half)
        if (row_start + x_r < M && k_step + x_c_base < K_dim) {
            float4 val = *((float4*)(&X[(row_start + x_r) * K_dim + k_step + x_c_base]));
            *((float4*)(&s_X[x_r][x_c_base])) = val;
            
            // 数据刚落入寄存器，顺手把 LN 统计量算了 (掩盖访存延迟)
            half* h = (half*)&val;
            #pragma unroll
            for(int j=0; j<8; ++j) {
                float v = __half2float(h[j]);
                thread_sum += v;
                thread_sq_sum += v*v;
            }
        }
        // 加载 X 的后半部分
        if (row_start + x_r < M && k_step + x_c_base + 16 < K_dim) {
            float4 val = *((float4*)(&X[(row_start + x_r) * K_dim + k_step + x_c_base + 16]));
            *((float4*)(&s_X[x_r][x_c_base + 16])) = val;
            
            half* h = (half*)&val;
            #pragma unroll
            for(int j=0; j<8; ++j) {
                float v = __half2float(h[j]);
                thread_sum += v;
                thread_sq_sum += v*v;
            }
        }

        // 向量化加载 W
        if (col_start + w_r < N_out && k_step + w_c_base < K_dim) {
            *((float4*)(&s_W[w_r][w_c_base])) = *((float4*)(&W[(col_start + w_r) * K_dim + k_step + w_c_base]));
        }
        if (col_start + w_r < N_out && k_step + w_c_base + 16 < K_dim) {
            *((float4*)(&s_W[w_r][w_c_base + 16])) = *((float4*)(&W[(col_start + w_r) * K_dim + k_step + w_c_base + 16]));
        }

        __syncthreads(); // 等待数据落入 Shared Memory

        // ==========================================
        // 动作 B：召唤 A100 Tensor Cores (WMMA)
        // ==========================================
        int warp_id = tid / 32;
        int warp_r = (warp_id / 2) * 32; // Warp 负责的行起始
        int warp_c = (warp_id % 2) * 32; // Warp 负责的列起始

        #pragma unroll
        for (int k_i = 0; k_i < BLOCK_K; k_i += WMMA_K) {
            // 从 Shared Memory 加载到 Tensor Core
            wmma::load_matrix_sync(a0, &s_X[warp_r][k_i], BLOCK_K + 8);
            wmma::load_matrix_sync(a1, &s_X[warp_r + 16][k_i], BLOCK_K + 8);
            
            wmma::load_matrix_sync(b0, &s_W[warp_c][k_i], BLOCK_K + 8);
            wmma::load_matrix_sync(b1, &s_W[warp_c + 16][k_i], BLOCK_K + 8);
            
            // 执行矩阵乘加 (1 周期轰出成百上千次 FMA)
            wmma::mma_sync(acc_00, a0, b0, acc_00);
            wmma::mma_sync(acc_01, a0, b1, acc_01);
            wmma::mma_sync(acc_10, a1, b0, acc_10);
            wmma::mma_sync(acc_11, a1, b1, acc_11);
        }
        __syncthreads(); // 准备下一个 K_step
    }

    // ==========================================
    // 5. 跨线程极速规约 LN 统计量
    // ==========================================
    // 对于每一行(共64行)，有 2 个线程(tid和tid^1)分别持有一半的 sum
    // 一次 Shuffle 就可以让这两个线程共享完整的 sum！
    float row_sum = thread_sum + __shfl_xor_sync(0xFFFFFFFF, thread_sum, 1);
    float row_sq = thread_sq_sum + __shfl_xor_sync(0xFFFFFFFF, thread_sq_sum, 1);

    // 让偶数号线程把结果写入 Shared Memory 供后续校正使用
    if (tid % 2 == 0 && row_start + x_r < M) {
        float mean = row_sum / (float)K_dim;
        float var = (row_sq / (float)K_dim) - (mean * mean);
        s_Mean[x_r] = mean;
        s_Rstd[x_r] = rsqrtf(fmaxf(0.0f, var) + 1e-6f);
    }

    // ==========================================
    // 6. Epilogue：写回 Shared -> 校正 -> 连续写出 Global
    // ==========================================
    int warp_id = tid / 32;
    int warp_r = (warp_id / 2) * 32;
    int warp_c = (warp_id % 2) * 32;

    wmma::store_matrix_sync(&s_Acc[warp_r][warp_c], acc_00, BLOCK_N, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_Acc[warp_r][warp_c + 16], acc_01, BLOCK_N, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_Acc[warp_r + 16][warp_c], acc_10, BLOCK_N, wmma::mem_row_major);
    wmma::store_matrix_sync(&s_Acc[warp_r + 16][warp_c + 16], acc_11, BLOCK_N, wmma::mem_row_major);
    
    __syncthreads();

    // 128 个线程分摊 64x64=4096 个元素的写回。每个线程负责 32 个。
    // 使用连续映射保证全局显存的合并写出 (Coalesced Write)
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int linear_idx = i * 128 + tid;
        int r = linear_idx / BLOCK_N;
        int c = linear_idx % BLOCK_N;
        
        if (row_start + r < M && col_start + c < N_out) {
            float val = s_Acc[r][c];
            float mean = s_Mean[r];
            float rstd = s_Rstd[r];
            
            // 终极纠偏
            float final_val = (val - mean * W_red[col_start + c]) * rstd + __half2float(bias[col_start + c]);
            Out[(row_start + r) * N_out + col_start + c] = __float2half(final_val);
        }
    }
}

torch::Tensor fused_ln_linear_cuda(
    torch::Tensor x, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    torch::Tensor w_reduced
) {
    int M = x.size(0);
    int K = x.size(1);
    int N_out = weight.size(0);

    auto out = torch::empty({M, N_out}, x.options());

    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((M + BLOCK_M - 1) / BLOCK_M, (N_out + BLOCK_N - 1) / BLOCK_N);

    // 获取当前 PyTorch 运行所在的 CUDA 流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    fused_ln_wmma_kernel<<<blocks, threads, 0, stream>>>(
        (half*)x.data_ptr(),
        (half*)weight.data_ptr(),
        (half*)bias.data_ptr(),
        w_reduced.data_ptr<float>(),
        (half*)out.data_ptr(),
        M, K, N_out
    );

    return out;
}
"""

# 编译
adaptive_ext = load_inline(
    name="adaptive_rerank_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_ln_linear_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,  # 强力建议开着，观察编译时间
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RuntimeAutotuner = load_module("vision_prefill_autotuner", "/root/Test/aicas2026gc/autotuner.py").RuntimeAutotuner
load_module("vision_prefill_rope", "/root/Test/aicas2026gc/ops/rope.py")

num_heads = 16
hidden_size = 1024
head_dim = hidden_size // num_heads
qkv_out = hidden_size * 3
block_m = 32
block_n = 128
block_k = 64
qkv = nn.Linear(hidden_size, qkv_out).cuda().half()
qkv_weight_reduced = qkv.weight.float().sum(dim=1).contiguous()
seq_lengths = [4096, 4096 - 64]


@tilelang.jit(out_idx=[-1])
def build_tilelang_kernel(block_m=32, block_n=128, block_k=64, num_stages=3, threads=128):
    num_tokens = T.dynamic("num_tokens")
    in_dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), in_dtype),
        w: T.Tensor((qkv_out, hidden_size), in_dtype),
        b: T.Tensor((qkv_out,), in_dtype),
        w_reduced: T.Tensor((qkv_out,), accum_dtype),
        out: T.Tensor((num_tokens, qkv_out), in_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), qkv_out // block_n, threads=threads) as (bx, by):
            # 分块矩阵累加
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)

            # sum_part = T.alloc_fragment((block_m, 4), accum_dtype) # 局部累加器。它把 block_K 里的 64 个元素先缩减成 4 个局部和
            # sqsum_part = T.alloc_fragment((block_m, 4), accum_dtype)

            # TODO: 按照 Marlin 一样，可以也做 INT4 量化，手动排布一下？但是 tilelang 能做这件事吗
            # 每行的和，没行的平方和，均值，倒数标准差
            sum_row = T.alloc_fragment((block_m,), accum_dtype)
            sqsum_row = T.alloc_fragment((block_m,), accum_dtype)
            mean_row = T.alloc_fragment((block_m,), accum_dtype)
            rstd_row = T.alloc_fragment((block_m,), accum_dtype)

            # 共享内存分配
            bias_shared = T.alloc_shared((block_n,), in_dtype)
            reduced_shared = T.alloc_shared((block_n,), accum_dtype)

            T.clear(acc)
            # T.clear(sum_part)
            # T.clear(sqsum_part)
            T.clear(sum_row)
            T.clear(sqsum_row)
            T.copy(b[by * block_n], bias_shared)  # block_n, global -> shared
            T.copy(w_reduced[by * block_n], reduced_shared)  # block_n, global -> shared

            # 主循环：计算 GEMM 和 LayerNorm 的中间统计量
            for bk in T.Pipelined(hidden_size // block_k, num_stages=num_stages):
                x_shared = T.alloc_shared((block_m, block_k), in_dtype)
                w_shared = T.alloc_shared((block_n, block_k), in_dtype)
                x_frag_16 = T.alloc_fragment((block_m, block_k), in_dtype)  # 后面 shared -> fragment
                x_frag = T.alloc_fragment((block_m, block_k), accum_dtype)
                T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared), w_shared: tilelang.layout.make_swizzled_layout(w_shared)})

                T.copy(x[bx * block_m, bk * block_k], x_shared)  # global to shared
                T.copy(w[by * block_n, bk * block_k], w_shared)

                # 矩阵乘法
                T.copy(x_shared, x_frag_16)
                T.copy(x_frag_16, x_frag)

                # ---- old reduction path ----
                # 统计量计算（基于 Shared Memory，规避布局冲突）
                # for kk in T.serial(block_k // 4):
                #     for i, j in T.Parallel(block_m, 4):
                #         x_val = T.Cast(accum_dtype, x_shared[i, kk * 4 + j])
                #         sum_part[i, j] += x_val
                #         sqsum_part[i, j] += x_val * x_val
                #
                # T.reduce_sum(sum_part, sum_row, dim=1) # block_m
                # T.reduce_sum(sqsum_part, sqsum_row, dim=1) # block_m
                # ---- old reduction path ----

                # ++++ new reduction path begin ++++
                T.reduce_sum(x_frag, sum_row, dim=1, clear=False)
                for i, j in T.Parallel(block_m, block_k):
                    x_frag[i, j] = x_frag[i, j] * x_frag[i, j]
                T.reduce_sum(x_frag, sqsum_row, dim=1, clear=False)
                # ++++ new reduction path end ++++
                T.gemm(x_frag_16, w_shared, acc, transpose_B=True, clear_accum=False)  # Tensor Core?

            # 跨线程规约得到每一行的均值和方差
            # T.reduce_sum(sum_part, sum_row, dim=1) # block_m
            # T.reduce_sum(sqsum_part, sqsum_row, dim=1) # block_m

            for i in T.Parallel(block_m):
                mean_row[i] = sum_row[i] / T.float32(hidden_size)
                variance = (sqsum_row[i] / T.float32(hidden_size)) - (mean_row[i] * mean_row[i])
                rstd_row[i] = T.rsqrt(variance + T.float32(1e-6))  # TODO: eps 移出去

            # Register-only epilogue: acc -> out_frag -> global
            for i, j in T.Parallel(block_m, block_n):
                acc[i, j] = (acc[i, j] - mean_row[i] * reduced_shared[j]) * rstd_row[i] + T.Cast(accum_dtype, bias_shared[j])

            T.copy(acc, out[bx * block_m, by * block_n])

    return main


tilelang_kernel = build_tilelang_kernel()


@tilelang.jit(out_idx=[-2, -1])
def build_tilelang_stats_kernel(block_m=64, block_k=64, num_stages=3, threads=128):
    num_tokens = T.dynamic("num_tokens")
    in_dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), in_dtype),
        mean: T.Tensor((num_tokens,), accum_dtype),
        rstd: T.Tensor((num_tokens,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), threads=threads) as bx:
            sum_row = T.alloc_fragment((block_m,), accum_dtype)
            sqsum_row = T.alloc_fragment((block_m,), accum_dtype)

            T.clear(sum_row)
            T.clear(sqsum_row)

            for bk in T.Pipelined(hidden_size // block_k, num_stages=num_stages):
                x_shared = T.alloc_shared((block_m, block_k), in_dtype)
                x_frag_16 = T.alloc_fragment((block_m, block_k), in_dtype)
                x_frag = T.alloc_fragment((block_m, block_k), accum_dtype)
                T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared)})

                T.copy(x[bx * block_m, bk * block_k], x_shared)
                T.copy(x_shared, x_frag_16)
                T.copy(x_frag_16, x_frag)
                T.reduce_sum(x_frag, sum_row, dim=1, clear=False)

                for i, j in T.Parallel(block_m, block_k):
                    x_frag[i, j] = x_frag[i, j] * x_frag[i, j]
                T.reduce_sum(x_frag, sqsum_row, dim=1, clear=False)

            for i in T.Parallel(block_m):
                mean_row = sum_row[i] / T.float32(hidden_size)
                variance = (sqsum_row[i] / T.float32(hidden_size)) - (mean_row * mean_row)
                mean[bx * block_m + i] = mean_row
                rstd[bx * block_m + i] = T.rsqrt(T.max(variance, T.float32(0.0)) + T.float32(1e-6))

    return main


@tilelang.jit(out_idx=[-1])
def build_tilelang_gemm_kernel(block_m=64, block_n=128, block_k=64, num_stages=3, threads=128):
    num_tokens = T.dynamic("num_tokens")
    in_dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), in_dtype),
        w: T.Tensor((qkv_out, hidden_size), in_dtype),
        b: T.Tensor((qkv_out,), in_dtype),
        w_reduced: T.Tensor((qkv_out,), accum_dtype),
        mean: T.Tensor((num_tokens,), accum_dtype),
        rstd: T.Tensor((num_tokens,), accum_dtype),
        out: T.Tensor((num_tokens, qkv_out), in_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), qkv_out // block_n, threads=threads) as (bx, by):
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)
            bias_frag = T.alloc_fragment((block_n,), in_dtype)
            reduced_frag = T.alloc_fragment((block_n,), accum_dtype)
            mean_row = T.alloc_fragment((block_m,), accum_dtype)
            rstd_row = T.alloc_fragment((block_m,), accum_dtype)
            bias_shared = T.alloc_shared((block_n,), in_dtype)
            reduced_shared = T.alloc_shared((block_n,), accum_dtype)
            mean_shared = T.alloc_shared((block_m,), accum_dtype)
            rstd_shared = T.alloc_shared((block_m,), accum_dtype)

            T.clear(acc)
            T.copy(b[by * block_n], bias_shared)
            T.copy(w_reduced[by * block_n], reduced_shared)
            T.copy(mean[bx * block_m], mean_shared)
            T.copy(rstd[bx * block_m], rstd_shared)
            T.copy(bias_shared, bias_frag)
            T.copy(reduced_shared, reduced_frag)
            T.copy(mean_shared, mean_row)
            T.copy(rstd_shared, rstd_row)

            for bk in T.Pipelined(hidden_size // block_k, num_stages=num_stages):
                x_shared = T.alloc_shared((block_m, block_k), in_dtype)
                w_shared = T.alloc_shared((block_n, block_k), in_dtype)
                T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared), w_shared: tilelang.layout.make_swizzled_layout(w_shared)})

                T.copy(x[bx * block_m, bk * block_k], x_shared)
                T.copy(w[by * block_n, bk * block_k], w_shared)
                T.gemm(x_shared, w_shared, acc, transpose_B=True, clear_accum=False)

            for i, j in T.Parallel(block_m, block_n):
                acc[i, j] = (acc[i, j] - mean_row[i] * reduced_frag[j]) * rstd_row[i] + T.Cast(accum_dtype, bias_frag[j])

            T.copy(acc, out[bx * block_m, by * block_n])

    return main


tilelang_stats_kernel = build_tilelang_stats_kernel()
tilelang_gemm_kernel = build_tilelang_gemm_kernel()


@tilelang.jit(out_idx=[-1])
def build_tilelang_norm_kernel(block_m=64, block_k=64, num_stages=3, threads=128):
    num_tokens = T.dynamic("num_tokens")
    in_dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), in_dtype),
        mean: T.Tensor((num_tokens,), accum_dtype),
        rstd: T.Tensor((num_tokens,), accum_dtype),
        out: T.Tensor((num_tokens, hidden_size), in_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), hidden_size // block_k, threads=threads) as (bx, bk):
            x_shared = T.alloc_shared((block_m, block_k), in_dtype)
            mean_shared = T.alloc_shared((block_m,), accum_dtype)
            rstd_shared = T.alloc_shared((block_m,), accum_dtype)
            mean_row = T.alloc_fragment((block_m,), accum_dtype)
            rstd_row = T.alloc_fragment((block_m,), accum_dtype)
            out_frag = T.alloc_fragment((block_m, block_k), in_dtype)
            x_frag = T.alloc_fragment((block_m, block_k), accum_dtype)
            T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared)})

            T.copy(x[bx * block_m, bk * block_k], x_shared)
            T.copy(mean[bx * block_m], mean_shared)
            T.copy(rstd[bx * block_m], rstd_shared)
            T.copy(mean_shared, mean_row)
            T.copy(rstd_shared, rstd_row)
            T.copy(x_shared, x_frag)

            for i, j in T.Parallel(block_m, block_k):
                out_frag[i, j] = T.Cast(in_dtype, (x_frag[i, j] - mean_row[i]) * rstd_row[i])

            T.copy(out_frag, out[bx * block_m, bk * block_k])

    return main


@tilelang.jit(out_idx=[-1])
def build_tilelang_linear_kernel(block_m=64, block_n=128, block_k=64, num_stages=3, threads=128):
    num_tokens = T.dynamic("num_tokens")
    in_dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), in_dtype),
        w: T.Tensor((qkv_out, hidden_size), in_dtype),
        b: T.Tensor((qkv_out,), in_dtype),
        out: T.Tensor((num_tokens, qkv_out), in_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), qkv_out // block_n, threads=threads) as (bx, by):
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)
            bias_frag = T.alloc_fragment((block_n,), in_dtype)
            bias_shared = T.alloc_shared((block_n,), in_dtype)

            T.clear(acc)
            T.copy(b[by * block_n], bias_shared)
            T.copy(bias_shared, bias_frag)

            for bk in T.Pipelined(hidden_size // block_k, num_stages=num_stages):
                x_shared = T.alloc_shared((block_m, block_k), in_dtype)
                w_shared = T.alloc_shared((block_n, block_k), in_dtype)
                T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared), w_shared: tilelang.layout.make_swizzled_layout(w_shared)})

                T.copy(x[bx * block_m, bk * block_k], x_shared)
                T.copy(w[by * block_n, bk * block_k], w_shared)
                T.gemm(x_shared, w_shared, acc, transpose_B=True, clear_accum=False)

            for i, j in T.Parallel(block_m, block_n):
                acc[i, j] += T.Cast(accum_dtype, bias_frag[j])

            T.copy(acc, out[bx * block_m, by * block_n])

    return main


tilelang_norm_kernel = build_tilelang_norm_kernel()
tilelang_linear_kernel = build_tilelang_linear_kernel()


def vision_prefill_module(hidden_states, qkv, cos, sin):
    seq_length = hidden_states.shape[0]
    hidden_states = F.layer_norm(hidden_states, [hidden_states.shape[-1]], eps=1e-6)
    query_states, key_states, value_states = qkv(hidden_states).reshape(seq_length, 3, num_heads, -1).unbind(1)
    query_states, key_states = torch.ops.qwen3vl.apply_rotary_pos_emb_vision_triton(query_states, key_states, cos, sin, inplace=False)
    return query_states, key_states, value_states


def vision_prefill_tilelang(hidden_states, qkv, cos, sin):
    seq_length = hidden_states.shape[0]
    qkv_states = tilelang_kernel(hidden_states, qkv.weight, qkv.bias, qkv_weight_reduced)
    query_states, key_states, value_states = qkv_states.reshape(seq_length, 3, num_heads, -1).unbind(1)
    query_states, key_states = torch.ops.qwen3vl.apply_rotary_pos_emb_vision_triton(query_states, key_states, cos, sin, inplace=False)
    return query_states, key_states, value_states


def vision_prefill_tilelang_split(hidden_states, qkv, cos, sin):
    seq_length = hidden_states.shape[0]
    mean, rstd = tilelang_stats_kernel(hidden_states)
    qkv_states = tilelang_gemm_kernel(hidden_states, qkv.weight, qkv.bias, qkv_weight_reduced, mean, rstd)
    query_states, key_states, value_states = qkv_states.reshape(seq_length, 3, num_heads, -1).unbind(1)
    query_states, key_states = torch.ops.qwen3vl.apply_rotary_pos_emb_vision_triton(query_states, key_states, cos, sin, inplace=False)
    return query_states, key_states, value_states


def vision_prefill_tilelang_norm_linear(hidden_states, qkv, cos, sin):
    seq_length = hidden_states.shape[0]
    mean, rstd = tilelang_stats_kernel(hidden_states)
    normed = tilelang_norm_kernel(hidden_states, mean, rstd)
    qkv_states = tilelang_linear_kernel(normed, qkv.weight, qkv.bias)
    query_states, key_states, value_states = qkv_states.reshape(seq_length, 3, num_heads, -1).unbind(1)
    query_states, key_states = torch.ops.qwen3vl.apply_rotary_pos_emb_vision_triton(query_states, key_states, cos, sin, inplace=False)
    return query_states, key_states, value_states


def vision_prefill_cuda(hidden_states, qkv, cos, sin):
    # 调用我们刚编好的纯融合算子
    # 注意：传进去的 qkv.weight 形状必须是 [N_out, K]
    qkv_states = adaptive_ext.fused_ln_linear_cuda(hidden_states, qkv.weight, qkv.bias, qkv_weight_reduced)

    # 手动在 Python 里做剩下的 RoPE 部分，用来对 Baseline
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = qkv_states.reshape(seq_length, 3, num_heads, -1).unbind(1)
    query_states, key_states = torch.ops.qwen3vl.apply_rotary_pos_emb_vision_triton(query_states, key_states, cos, sin, inplace=False)
    return query_states, key_states, value_states


tuner = RuntimeAutotuner(name="vision_prefill_fused_layernorm_qkv_rope", fallback="module", strict=False)

for seq_length in seq_lengths:
    hidden_states = torch.randn(seq_length, hidden_size, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq_length, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq_length, head_dim, device="cuda", dtype=torch.float16)

    # print("???? Warming up...")
    # print(vision_prefill_cuda(hidden_states, qkv, cos, sin))
    # print("???? Warmup done, start autotuning...")

    candidates = {
        "module": vision_prefill_module,
        "tilelang": vision_prefill_tilelang,
        "tilelang_split": vision_prefill_tilelang_split,
        "tilelang_norm_linear": vision_prefill_tilelang_norm_linear,
        # "cuda": vision_prefill_cuda
    }

    outputs = tuner.dispatch_once(candidates, name=f"seq{seq_length}", args=(hidden_states, qkv, cos, sin))
    print(seq_length, tuner.dispatch_table[f"once:seq{seq_length}"], [x.shape for x in outputs])
