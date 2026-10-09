import importlib.util
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    str(Path(__file__).resolve().parents[2] / ".torch_extensions"),
)


cpp_source = r"""
#include <torch/extension.h>

void addmm_residual_warp_out_cuda(
    torch::Tensor residual,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out
);

void addmm_residual_wmma_out_cuda(
    torch::Tensor residual,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out
);
"""


cuda_source = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

__device__ __forceinline__ float warp_sum_float(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void addmm_residual_warp_kernel(
    const half* __restrict__ residual,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int N,
    int K
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= N) {
        return;
    }

    const half2* __restrict__ x2 = reinterpret_cast<const half2*>(x);
    const half2* __restrict__ w2 = reinterpret_cast<const half2*>(weight + row * K);
    float acc = 0.0f;

    const int K2 = K >> 1;
    for (int i = lane; i < K2; i += 32) {
        const float2 xv = __half22float2(x2[i]);
        const float2 wv = __half22float2(w2[i]);
        acc = fmaf(xv.x, wv.x, acc);
        acc = fmaf(xv.y, wv.y, acc);
    }

    acc = warp_sum_float(acc);
    if (lane == 0) {
        acc += __half2float(residual[row]);
        out[row] = __float2half_rn(acc);
    }
}

__global__ void addmm_residual_wmma_kernel(
    const half* __restrict__ residual,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int N,
    int K
) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;

    extern __shared__ unsigned char smem_raw[];
    half* smem_a = reinterpret_cast<half*>(smem_raw);
    float* smem_c = reinterpret_cast<float*>(smem_a + kWarpsPerBlock * kWmmaM * kWmmaK);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int tile_n = blockIdx.x * kWarpsPerBlock + warp_id;
    const int n_start = tile_n * kWmmaN;
    if (n_start >= N) {
        return;
    }

    half* a_tile = smem_a + warp_id * kWmmaM * kWmmaK;
    float* c_tile = smem_c + warp_id * kWmmaM * kWmmaN;

    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kWmmaK) {
        for (int i = lane; i < kWmmaM * kWmmaK; i += 32) {
            const int col = i & (kWmmaK - 1);
            a_tile[i] = x[k0 + col];
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, a_tile, kWmmaK);
        wmma::load_matrix_sync(b_frag, weight + n_start * K + k0, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(c_tile, c_frag, kWmmaN, wmma::mem_row_major);
    __syncwarp();

    if (lane < kWmmaN && n_start + lane < N) {
        const float v = c_tile[lane] + __half2float(residual[n_start + lane]);
        out[n_start + lane] = __float2half_rn(v);
    }
#endif
}

void check_common(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out
) {
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(residual.scalar_type() == at::ScalarType::Half, "residual must be float16");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, "x must be float16");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half, "weight must be float16");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Half, "out must be float16");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K]");
    TORCH_CHECK(residual.numel() == weight.size(0), "residual numel must equal N");
    TORCH_CHECK(out.numel() == weight.size(0), "out numel must equal N");
    TORCH_CHECK(x.numel() == weight.size(1), "x numel must equal K");
    TORCH_CHECK(weight.size(1) % 32 == 0, "K must be divisible by 32");
}

}  // namespace

void addmm_residual_warp_out_cuda(
    torch::Tensor residual,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out
) {
    check_common(residual, x, weight, out);
    const int N = static_cast<int>(weight.size(0));
    const int K = static_cast<int>(weight.size(1));
    const int blocks = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    addmm_residual_warp_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        N,
        K
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void addmm_residual_wmma_out_cuda(
    torch::Tensor residual,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out
) {
    check_common(residual, x, weight, out);
    TORCH_CHECK(weight.size(0) % kWmmaN == 0, "N must be divisible by 16 for WMMA");
    TORCH_CHECK(weight.size(1) % kWmmaK == 0, "K must be divisible by 16 for WMMA");
    const int N = static_cast<int>(weight.size(0));
    const int K = static_cast<int>(weight.size(1));
    const int tiles_n = (N + kWmmaN - 1) / kWmmaN;
    const int blocks = (tiles_n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const size_t smem_bytes =
        kWarpsPerBlock * kWmmaM * kWmmaK * sizeof(half) +
        kWarpsPerBlock * kWmmaM * kWmmaN * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    addmm_residual_wmma_kernel<<<blocks, kThreadsPerBlock, smem_bytes, stream>>>(
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        N,
        K
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_module = load_inline(
    name="fused_addmm_residual_gemv_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    with_cuda=True,
    functions=["addmm_residual_warp_out_cuda", "addmm_residual_wmma_out_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def _flat(t):
    return t.reshape(-1).contiguous()


def addmm_residual_warp_out(residual, x, weight, out):
    _module.addmm_residual_warp_out_cuda(_flat(residual), _flat(x), weight.contiguous(), _flat(out))
    return out


def addmm_residual_wmma_out(residual, x, weight, out):
    _module.addmm_residual_wmma_out_cuda(_flat(residual), _flat(x), weight.contiguous(), _flat(out))
    return out


def addmm_residual_warp(residual, x, weight):
    out = torch.empty_like(residual)
    return addmm_residual_warp_out(residual, x, weight, out)


def addmm_residual_wmma(residual, x, weight):
    out = torch.empty_like(residual)
    return addmm_residual_wmma_out(residual, x, weight, out)


def torch_addmm_residual(residual, x, weight):
    n, k = weight.shape
    return torch.addmm(residual.reshape(1, n), x.reshape(1, k), weight.t()).view_as(residual)


__all__ = [
    "addmm_residual_warp",
    "addmm_residual_warp_out",
    "addmm_residual_wmma",
    "addmm_residual_wmma_out",
    "torch_addmm_residual",
]


if __name__ == "__main__":
    import sys

    try:
        from aicas2026gc.autotuner import RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[1] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("fused_addmm_residual_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        RuntimeAutotuner = module.RuntimeAutotuner

    sys.path.insert(0, "/root/marlin")
    try:
        import marlin

        HAS_MARLIN = True
    except Exception as exc:  # pragma: no cover - benchmark-only optional path
        print(f"marlin import failed: {exc}")
        HAS_MARLIN = False

    torch.manual_seed(2026)
    device = "cuda"
    dtype = torch.float16
    shapes = [
        ("o_proj", 2048, 2048),
        ("down_proj", 2048, 6144),
    ]

    for shape_name, n, k in shapes:
        residual = torch.randn(1, 1, n, device=device, dtype=dtype).contiguous()
        x = torch.randn(1, 1, k, device=device, dtype=dtype).contiguous()
        weight = torch.randn(n, k, device=device, dtype=dtype).contiguous()
        warp_out = torch.empty_like(residual)
        wmma_out = torch.empty_like(residual)
        marlin_variants = {}

        if HAS_MARLIN:
            groupsize = 128
            maxq = 2**4 - 1
            linear = torch.nn.Linear(k, n, bias=False, device=device, dtype=dtype)
            linear.weight.data.copy_(weight)
            w_kn = weight.t().contiguous()
            scales = w_kn.reshape(k // groupsize, groupsize, n).abs().amax(dim=1)
            scales = (scales * (2.0 / maxq)).clamp_min(1e-6).contiguous()
            marlin_layer = marlin.Layer(k, n, groupsize=groupsize).to(device)
            marlin_layer.pack(linear, scales.t().contiguous())

            for tk, tn in ((64, 256), (128, 128)):
                c_buf = torch.empty(1, n, device=device, dtype=dtype)

                def make_marlin_impl(thread_k, thread_n, c_out):
                    def impl(residual_arg, x_arg, weight_arg):
                        del weight_arg
                        marlin.mul(
                            x_arg.reshape(1, k),
                            marlin_layer.B,
                            c_out,
                            marlin_layer.s,
                            marlin_layer.workspace,
                            thread_k,
                            thread_n,
                            64,
                        )
                        c_out.add_(residual_arg.reshape(1, n))
                        return c_out.view_as(residual_arg)

                    return impl

                marlin_variants[f"marlin_int4_g128_tk{tk}_tn{tn}"] = make_marlin_impl(tk, tn, c_buf)

        def warp_impl(residual_arg, x_arg, weight_arg):
            return addmm_residual_warp_out(residual_arg, x_arg, weight_arg, warp_out)

        def wmma_impl(residual_arg, x_arg, weight_arg):
            return addmm_residual_wmma_out(residual_arg, x_arg, weight_arg, wmma_out)

        candidates = {
            "torch_addmm": torch_addmm_residual,
            "warp_half2": warp_impl,
            "wmma_m16": wmma_impl,
        }
        candidates.update(marlin_variants)

        tuner = RuntimeAutotuner(
            name=f"fused_addmm_residual_{shape_name}",
            fallback="torch_addmm",
            check_correctness=False,
            strict=False,
            atol=6e-2,
            rtol=6e-2,
        )

        bucket_name = f"N{n}_K{k}_{str(dtype).split('.')[-1]}"
        print(f"\n>>> {shape_name}: residual + [1,{k}] @ [{n},{k}].T")
        out = tuner.dispatch_once(candidates, name=bucket_name, args=(residual, x, weight))
        torch.cuda.synchronize()
        ref = torch_addmm_residual(residual, x, weight)
        max_diff = (ref - out).abs().max().item()
        best = tuner.dispatch_table[f"once:{bucket_name}"]
        print(f"best={best}, max_diff={max_diff:.6f}")
        if marlin_variants:
            for name, fn in marlin_variants.items():
                cand = fn(residual, x, weight)
                torch.cuda.synchronize()
                rel = (cand - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
                print(f"{name}: mean_rel_diff={float(rel.item()):.6f}")
