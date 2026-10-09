import torch
import torch.nn as nn
import time
import math
from torch.utils.cpp_extension import load_inline

# =====================================================================
# 1. 纯 C++ & CUDA 算子源码
# =====================================================================

cpp_source = """
#include <torch/extension.h>

// 声明 CUDA 函数
torch::Tensor adaptive_rerank_cuda(
    torch::Tensor s_approx,
    torch::Tensor threshold_tensor,
    torch::Tensor W,
    torch::Tensor x);
"""

cuda_source = """
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Warp 级别的规约求和
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void adaptive_rerank_kernel(
    const float* __restrict__ s_approx,
    const float* __restrict__ threshold_tensor,
    const half* __restrict__ W,
    const half* __restrict__ x,
    float* __restrict__ out_scores,
    int V, int D) 
{
    // 动态分配 Shared Memory 来缓存输入向量 x
    extern __shared__ half smem_x[];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    
    // 全局词汇索引：每个 Warp 负责一个词
    int v_idx = blockIdx.x * warps_per_block + warp_id;

    // 1. 协作将 x 载入 Shared Memory (只需 4KB，极速)
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        smem_x[d] = x[d];
    }
    __syncthreads(); // 确保 SMEM 载入完毕

    // 边界检查
    if (v_idx >= V) return;

    // 2. 核心逻辑：Warp 级别的 Early Exit
    // 只要近似分数低于阈值，这 32 个线程直接跳过，0 次全局访存！
    if (s_approx[v_idx] < threshold_tensor[0]) {
        return; 
    }

    // 3. 幸存者进行精确内积计算
    float sum = 0.0f;
    int offset = v_idx * D; // 矩阵 W 的行起始位置
    
    // Warp 内 32 个线程循环读取 D 维度 (完美连续的 Coalesced Memory Access)
    for (int d = lane_id; d < D; d += 32) {
        // 将 half 转为 float 进行高精度累加
        sum += __half2float(W[offset + d]) * __half2float(smem_x[d]);
    }

    // 4. Warp 内规约
    sum = warpReduceSum(sum);

    // 5. 0号线程将最终得分写出
    if (lane_id == 0) {
        out_scores[v_idx] = sum;
    }
}

torch::Tensor adaptive_rerank_cuda(
    torch::Tensor s_approx,
    torch::Tensor threshold_tensor,
    torch::Tensor W,
    torch::Tensor x) 
{
    int V = W.size(0);
    int D = W.size(1);

    // 创建输出 Tensor，初始化为非常小的负数 (-1e9)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(W.device());
    torch::Tensor out_scores = torch::full({V}, -1e9, options);

    // 线程块配置
    int threads_per_block = 256; 
    int warps_per_block = threads_per_block / 32; // 8 个 Warp
    int blocks = (V + warps_per_block - 1) / warps_per_block;
    
    // Shared memory 大小 = D * sizeof(half)
    int shared_mem_size = D * sizeof(half);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    adaptive_rerank_kernel<<<blocks, threads_per_block, shared_mem_size, stream>>>(
        s_approx.data_ptr<float>(),
        threshold_tensor.data_ptr<float>(),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        out_scores.data_ptr<float>(),
        V, D
    );

    return out_scores;
}
"""

print("[*] 正在 JIT 编译纯 CUDA 算子，请稍候 (约需 10-30 秒)...")
adaptive_ext = load_inline(
    name="adaptive_rerank_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["adaptive_rerank_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],  # 去掉了 -arch，让它自动匹配你当前的 GPU
    verbose=False,
)
print("[*] CUDA 算子编译成功！")


# =====================================================================
# 2. 挂载到 Python 逻辑
# =====================================================================


class FastAdaptiveSafeRecallLMHead(nn.Module):
    def __init__(self):
        super().__init__()

    def _post_init(self, linear_layer: nn.Linear, rank: int = 256):

        self.device = linear_layer.weight.device
        self.D, self.V = linear_layer.in_features, linear_layer.out_features
        self.W = linear_layer.weight.data.contiguous()  # 确保 W 内存连续
        self.rank = rank

        # --- 离线 SVD ---
        W_fp32 = self.W.to(torch.float32)
        U, S, Vh = torch.linalg.svd(W_fp32.T, full_matrices=False)
        self.L_proj = U[:, : self.rank].contiguous().half()  # 转为 half 提升投影速度
        self.R_proj = (torch.diag(S[: self.rank]) @ Vh[: self.rank, :]).contiguous().half()

        self.sigma_r1 = S[self.rank]
        Vh_discarded = Vh[self.rank :, :]
        self.L_max = (Vh_discarded**2).sum(dim=0).max()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = x.squeeze(0).contiguous()

        # 1. 纯 cuBLAS 极速近似投影
        a_proj = x @ self.L_proj  # [1, r]
        s_approx = a_proj @ self.R_proj  # [1, V], FP16
        s_approx_fp32 = s_approx.float()  # 转 FP32 以防溢出

        # 2. 计算动态阈值
        norm_x_sq = (x.float() ** 2).sum()
        norm_proj_sq = (a_proj.float() ** 2).sum()
        E_perp = torch.clamp(norm_x_sq - norm_proj_sq, min=0.0)

        epsilon_max = self.sigma_r1 * torch.sqrt(E_perp) * torch.sqrt(self.L_max)
        s_max_approx = s_approx_fp32.max()
        threshold = s_max_approx - 2 * epsilon_max

        # 3. 调用手撸的 CUDA 算子做精准重排
        # 这个算子会直接返回 [V] 长度的数组，没被触发的词全是 -1e9
        exact_scores = adaptive_ext.adaptive_rerank_cuda(s_approx_fp32, threshold, self.W, x)

        # 4. PyTorch 极速 Argmax
        return exact_scores.argmax()


# =====================================================================
# 3. 压测对比代码 (合成真实分布)
# =====================================================================
def run_benchmark():
    torch.manual_seed(42)
    device = torch.device("cuda")
    V, D = 151936, 2048

    print("\n>>> 初始化模拟权重...")
    U_rand, _ = torch.linalg.qr(torch.randn(D, D, device=device))
    V_rand = torch.randn(D, V, device=device)
    V_norm = V_rand / V_rand.norm(dim=1, keepdim=True)
    S = torch.exp(-torch.linspace(0, 8, D, device=device)) * 100.0
    W = (U_rand @ torch.diag(S) @ V_norm).T.contiguous().half()

    lm_head = nn.Linear(D, V, bias=False, device=device, dtype=torch.float16)
    lm_head.weight.data.copy_(W)

    print(">>> 构建 CUDA 加速的自适应召回器...")
    fast_head = FastAdaptiveSafeRecallLMHead(lm_head, rank=256)

    # 模拟真实激活输入 (99%能量在主成分，带少量底噪)
    aligned_coeffs = torch.randn(256, 1, dtype=torch.float16, device=device) * 5.0
    x_aligned = (fast_head.L_proj @ aligned_coeffs).T
    x_noise = torch.randn(1, D, dtype=torch.float16, device=device) * 0.05
    x = (x_aligned + x_noise).contiguous()

    # 预热
    for _ in range(10):
        ans_base = lm_head(x).argmax().item()
        ans_fast = fast_head.forward(x)
    torch.cuda.synchronize()

    # 测速 Baseline
    t0 = time.time()
    for _ in range(500):
        ans_base = lm_head(x).argmax()
    torch.cuda.synchronize()
    time_base = (time.time() - t0) / 500 * 1000

    # 测速 Fast CUDA
    t1 = time.time()
    for _ in range(500):
        ans_fast = fast_head.forward(x)
    torch.cuda.synchronize()
    time_fast = (time.time() - t1) / 500 * 1000

    print(f"\n【精度验证】")
    print(f"原生 Argmax : {ans_base}")
    print(f"CUDA Argmax : {ans_fast}")
    assert ans_base == ans_fast, "❌ 精度翻车！"
    print("✅ 精度 100% 绝对一致！")

    print(f"\n【性能暴打现场】")
    print(f"全量 GEMV (PyTorch) 耗时: {time_base:.4f} ms")
    print(f"CUDA 自适应召回算子 耗时: {time_fast:.4f} ms")
    print(f"🚀 真实加速比: {time_base / time_fast:.2f}x")


if __name__ == "__main__":
    run_benchmark()
