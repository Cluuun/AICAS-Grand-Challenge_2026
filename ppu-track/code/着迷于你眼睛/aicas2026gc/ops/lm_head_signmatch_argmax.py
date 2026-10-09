import importlib.util
import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


LM_HEAD_HIDDEN = 2048
SIGN_PACK_WORDS = LM_HEAD_HIDDEN // 32
MAX_LOCAL_K = 16
DEFAULT_ROWS_PER_BLOCK = 512
DEFAULT_LOCAL_K = 1


os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    str(Path(__file__).resolve().parents[2] / ".torch_extensions"),
)


cpp_source = r"""
#include <torch/extension.h>

torch::Tensor pack_lm_head_signs_cuda(torch::Tensor weight);

void signmatch_argmax_out_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor packed_signs,
    torch::Tensor block_vals,
    torch::Tensor block_idxs,
    torch::Tensor out,
    int64_t rows_per_block,
    int64_t local_k
);
"""


cuda_source = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <climits>

namespace {

constexpr int kHidden = 2048;
constexpr int kPackWords = 64;
constexpr int kBitsPerWord = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kMaxLocalK = 16;
constexpr float kNegInf = -3.4028234663852886e38f;

__device__ __forceinline__ int warp_sum_int(int v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_sum_float(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ bool better_score(int score, int idx, int best_score, int best_idx) {
    return score > best_score || (score == best_score && idx < best_idx);
}

__device__ __forceinline__ bool better_logit(float val, int idx, float best_val, int best_idx) {
    return val > best_val || (val == best_val && idx < best_idx);
}

__global__ void pack_signs_kernel(
    const half* __restrict__ weight,
    int32_t* __restrict__ packed,
    int vocab_size
) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = vocab_size * kPackWords;
    if (linear >= total) {
        return;
    }

    const int row = linear / kPackWords;
    const int word = linear - row * kPackWords;
    const int base = row * kHidden + word * kBitsPerWord;
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(weight);

    uint32_t bits = 0;
    #pragma unroll
    for (int bit = 0; bit < kBitsPerWord; ++bit) {
        const uint32_t is_non_negative = ((raw[base + bit] & 0x8000u) == 0u);
        bits |= (is_non_negative << bit);
    }
    packed[linear] = static_cast<int32_t>(bits);
}

__global__ void signmatch_candidates_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const uint32_t* __restrict__ packed_signs,
    float* __restrict__ block_vals,
    int32_t* __restrict__ block_idxs,
    int vocab_size,
    int rows_per_block,
    int local_k
) {
    __shared__ uint32_t x_sign[kPackWords];
    __shared__ float warp_vals[kWarpsPerBlock];
    __shared__ int warp_idxs[kWarpsPerBlock];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const uint16_t* x_raw = reinterpret_cast<const uint16_t*>(x);

    for (int word = tid; word < kPackWords; word += blockDim.x) {
        uint32_t bits = 0;
        const int base = word * kBitsPerWord;
        #pragma unroll
        for (int bit = 0; bit < kBitsPerWord; ++bit) {
            const uint32_t is_non_negative = ((x_raw[base + bit] & 0x8000u) == 0u);
            bits |= (is_non_negative << bit);
        }
        x_sign[word] = bits;
    }
    __syncthreads();

    int top_scores[kMaxLocalK];
    int top_idxs[kMaxLocalK];
    #pragma unroll
    for (int i = 0; i < kMaxLocalK; ++i) {
        top_scores[i] = -1;
        top_idxs[i] = INT_MAX;
    }

    const int block_row_start = blockIdx.x * rows_per_block;
    for (int row_offset = warp_id; row_offset < rows_per_block; row_offset += kWarpsPerBlock) {
        const int row = block_row_start + row_offset;
        int lane_score = 0;
        if (row < vocab_size) {
            const uint32_t* row_signs = packed_signs + row * kPackWords;
            const uint32_t mismatch0 = row_signs[lane] ^ x_sign[lane];
            const uint32_t mismatch1 = row_signs[lane + 32] ^ x_sign[lane + 32];
            lane_score = 64 - __popc(mismatch0) - __popc(mismatch1);
        }
        const int score = warp_sum_int(lane_score);

        if (lane == 0 && row < vocab_size) {
            for (int pos = 0; pos < local_k; ++pos) {
                if (better_score(score, row, top_scores[pos], top_idxs[pos])) {
                    for (int shift = local_k - 1; shift > pos; --shift) {
                        top_scores[shift] = top_scores[shift - 1];
                        top_idxs[shift] = top_idxs[shift - 1];
                    }
                    top_scores[pos] = score;
                    top_idxs[pos] = row;
                    break;
                }
            }
        }
    }

    float warp_best_val = kNegInf;
    int warp_best_idx = INT_MAX;
    const half2* x2 = reinterpret_cast<const half2*>(x);

    for (int rank = 0; rank < local_k; ++rank) {
        int candidate = top_idxs[rank];
        candidate = __shfl_sync(0xffffffff, candidate, 0);
        if (candidate == INT_MAX) {
            continue;
        }

        const half2* w2 = reinterpret_cast<const half2*>(weight + candidate * kHidden);
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = lane; i < kHidden / 2; i += 32) {
            const float2 xv = __half22float2(x2[i]);
            const float2 wv = __half22float2(w2[i]);
            sum = fmaf(xv.x, wv.x, sum);
            sum = fmaf(xv.y, wv.y, sum);
        }
        sum = warp_sum_float(sum);

        if (lane == 0 && better_logit(sum, candidate, warp_best_val, warp_best_idx)) {
            warp_best_val = sum;
            warp_best_idx = candidate;
        }
    }

    if (lane == 0) {
        warp_vals[warp_id] = warp_best_val;
        warp_idxs[warp_id] = warp_best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float best_val = (lane < kWarpsPerBlock) ? warp_vals[lane] : kNegInf;
        int best_idx = (lane < kWarpsPerBlock) ? warp_idxs[lane] : INT_MAX;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
            const int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (better_logit(other_val, other_idx, best_val, best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane == 0) {
            block_vals[blockIdx.x] = best_val;
            block_idxs[blockIdx.x] = best_idx;
        }
    }
}

__global__ void reduce_blocks_kernel(
    const float* __restrict__ block_vals,
    const int32_t* __restrict__ block_idxs,
    int64_t* __restrict__ out,
    int num_blocks
) {
    __shared__ float vals[256];
    __shared__ int idxs[256];

    const int tid = threadIdx.x;
    float best_val = kNegInf;
    int best_idx = INT_MAX;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        const float val = block_vals[i];
        const int idx = static_cast<int>(block_idxs[i]);
        if (better_logit(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_val = vals[tid + stride];
            const int other_idx = idxs[tid + stride];
            if (better_logit(other_val, other_idx, vals[tid], idxs[tid])) {
                vals[tid] = other_val;
                idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = static_cast<int64_t>(idxs[0]);
    }
}

}  // namespace

torch::Tensor pack_lm_head_signs_cuda(torch::Tensor weight) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half, "weight must be float16");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(weight.size(1) == kHidden, "hidden size must be 2048");

    const int vocab_size = static_cast<int>(weight.size(0));
    auto packed = torch::empty(
        {vocab_size, kPackWords},
        torch::TensorOptions().device(weight.device()).dtype(torch::kInt32)
    );

    const int total = vocab_size * kPackWords;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_signs_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        packed.data_ptr<int32_t>(),
        vocab_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

void signmatch_argmax_out_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor packed_signs,
    torch::Tensor block_vals,
    torch::Tensor block_idxs,
    torch::Tensor out,
    int64_t rows_per_block,
    int64_t local_k
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(packed_signs.is_cuda(), "packed_signs must be CUDA");
    TORCH_CHECK(block_vals.is_cuda(), "block_vals must be CUDA");
    TORCH_CHECK(block_idxs.is_cuda(), "block_idxs must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(packed_signs.is_contiguous(), "packed_signs must be contiguous");
    TORCH_CHECK(block_vals.is_contiguous(), "block_vals must be contiguous");
    TORCH_CHECK(block_idxs.is_contiguous(), "block_idxs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, "x must be float16");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half, "weight must be float16");
    TORCH_CHECK(packed_signs.scalar_type() == at::ScalarType::Int, "packed_signs must be int32");
    TORCH_CHECK(block_vals.scalar_type() == at::ScalarType::Float, "block_vals must be float32");
    TORCH_CHECK(block_idxs.scalar_type() == at::ScalarType::Int, "block_idxs must be int32");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Long, "out must be int64");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(weight.size(1) == kHidden, "hidden size must be 2048");
    TORCH_CHECK(x.numel() == kHidden, "x must have 2048 elements");
    TORCH_CHECK(packed_signs.size(0) == weight.size(0), "packed vocab mismatch");
    TORCH_CHECK(packed_signs.size(1) == kPackWords, "packed signs must have 64 words per row");
    TORCH_CHECK(rows_per_block > 0, "rows_per_block must be positive");
    TORCH_CHECK(local_k > 0 && local_k <= kMaxLocalK, "local_k must be in [1, 16]");
    TORCH_CHECK(out.numel() >= 1, "out must have at least one element");

    const int vocab_size = static_cast<int>(weight.size(0));
    const int rows = static_cast<int>(rows_per_block);
    const int lk = static_cast<int>(local_k);
    const int num_blocks = (vocab_size + rows - 1) / rows;
    TORCH_CHECK(block_vals.numel() >= num_blocks, "block_vals scratch too small");
    TORCH_CHECK(block_idxs.numel() >= num_blocks, "block_idxs scratch too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    signmatch_candidates_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(packed_signs.data_ptr<int32_t>()),
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int32_t>(),
        vocab_size,
        rows,
        lk
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    reduce_blocks_kernel<<<1, 256, 0, stream>>>(
        block_vals.data_ptr<float>(),
        block_idxs.data_ptr<int32_t>(),
        out.data_ptr<int64_t>(),
        num_blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


signmatch_module = load_inline(
    name="lm_head_signmatch_argmax_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    with_cuda=True,
    functions=["pack_lm_head_signs_cuda", "signmatch_argmax_out_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class SignmatchArgmaxWorkspace:
    def __init__(self, vocab_size, device, rows_per_block=DEFAULT_ROWS_PER_BLOCK):
        self.rows_per_block = int(rows_per_block)
        self.num_blocks = math.ceil(int(vocab_size) / self.rows_per_block)
        self.block_vals = torch.empty(self.num_blocks, device=device, dtype=torch.float32)
        self.block_idxs = torch.empty(self.num_blocks, device=device, dtype=torch.int32)
        self.out = torch.empty(1, device=device, dtype=torch.long)


def pack_lm_head_signs(weight):
    if weight.dtype != torch.float16:
        raise TypeError("pack_lm_head_signs currently supports float16 weights only")
    if weight.shape[-1] != LM_HEAD_HIDDEN:
        raise ValueError("lm_head hidden size must be 2048")
    return signmatch_module.pack_lm_head_signs_cuda(weight.contiguous())


def signmatch_argmax_out(
    x,
    weight,
    packed_signs,
    workspace,
    rows_per_block=DEFAULT_ROWS_PER_BLOCK,
    local_k=DEFAULT_LOCAL_K,
    token_id_map=None,
):
    x_flat = x.reshape(-1)
    signmatch_module.signmatch_argmax_out_cuda(
        x_flat.contiguous(),
        weight.contiguous(),
        packed_signs,
        workspace.block_vals,
        workspace.block_idxs,
        workspace.out,
        int(rows_per_block),
        int(local_k),
    )
    if token_id_map is None or token_id_map.numel() == 0:
        return workspace.out
    return token_id_map[workspace.out]


def signmatch_argmax(
    x,
    weight,
    packed_signs=None,
    workspace=None,
    rows_per_block=DEFAULT_ROWS_PER_BLOCK,
    local_k=DEFAULT_LOCAL_K,
    token_id_map=None,
):
    packed = packed_signs if packed_signs is not None else pack_lm_head_signs(weight)
    ws = workspace
    if ws is None or ws.rows_per_block != int(rows_per_block):
        ws = SignmatchArgmaxWorkspace(weight.shape[0], weight.device, rows_per_block)
    return signmatch_argmax_out(x, weight, packed, ws, rows_per_block, local_k, token_id_map=token_id_map)


def torch_lm_head_argmax(x, weight):
    logits = torch.matmul(x.reshape(1, -1), weight.t())
    return torch.argmax(logits, dim=-1).to(torch.long)


__all__ = [
    "DEFAULT_LOCAL_K",
    "DEFAULT_ROWS_PER_BLOCK",
    "LM_HEAD_HIDDEN",
    "MAX_LOCAL_K",
    "SIGN_PACK_WORDS",
    "SignmatchArgmaxWorkspace",
    "pack_lm_head_signs",
    "signmatch_argmax",
    "signmatch_argmax_out",
    "torch_lm_head_argmax",
]


if __name__ == "__main__":
    try:
        from aicas2026gc.autotuner import RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[1] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("lm_head_signmatch_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        RuntimeAutotuner = module.RuntimeAutotuner

    torch.manual_seed(2026)
    device = "cuda"
    dtype = torch.float16
    vocab_size = int(os.environ.get("LM_HEAD_BENCH_VOCAB", "151936"))
    hidden_size = LM_HEAD_HIDDEN

    x = torch.randn(1, 1, hidden_size, device=device, dtype=dtype).contiguous()
    weight = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype).contiguous()

    print("\n>>> packing lm_head signs...")
    packed = pack_lm_head_signs(weight)
    torch.cuda.synchronize()
    print(f"packed_signs={tuple(packed.shape)} int32, bytes={packed.numel() * packed.element_size() / 1024 / 1024:.1f} MiB")

    configs = {
        "sign_r256_lk1": (256, 1),
        "sign_r256_lk2": (256, 2),
        "sign_r256_lk4": (256, 4),
        "sign_r512_lk1": (512, 1),
        "sign_r512_lk2": (512, 2),
        "sign_r512_lk4": (512, 4),
        "sign_r512_lk8": (512, 8),
        "sign_r1024_lk1": (1024, 1),
        "sign_r1024_lk2": (1024, 2),
        "sign_r1024_lk4": (1024, 4),
        "sign_r1024_lk8": (1024, 8),
        "sign_r2048_lk8": (2048, 8),
        "sign_r2048_lk16": (2048, 16),
    }
    workspaces = {
        name: SignmatchArgmaxWorkspace(vocab_size, device, rows_per_block=rows_per_block)
        for name, (rows_per_block, _local_k) in configs.items()
    }

    def make_signmatch_impl(name, rows_per_block, local_k):
        workspace = workspaces[name]

        def impl(x_arg, weight_arg):
            return signmatch_argmax_out(
                x_arg,
                weight_arg,
                packed,
                workspace,
                rows_per_block=rows_per_block,
                local_k=local_k,
            )

        return impl

    candidates = {
        "torch_exact": torch_lm_head_argmax,
    }
    for cfg_name, (rows_per_block, local_k) in configs.items():
        candidates[cfg_name] = make_signmatch_impl(cfg_name, rows_per_block, local_k)

    tuner = RuntimeAutotuner(
        name="lm_head_signmatch_argmax",
        fallback="torch_exact",
        check_correctness=False,
        strict=False,
    )

    bucket_name = f"V{vocab_size}_D{hidden_size}_{str(dtype).split('.')[-1]}"
    print("\n>>> dispatch_once speed test (correctness check disabled for approximate candidates)...")
    out = tuner.dispatch_once(candidates, name=bucket_name, args=(x, weight))
    torch.cuda.synchronize()

    ref = torch_lm_head_argmax(x, weight)
    best = tuner.dispatch_table[f"once:{bucket_name}"]
    print(f"best_algo={best}")
    print(f"torch_exact_token={int(ref.item())}, selected_token={int(out.item())}, sample_match={bool((ref == out).item())}")
