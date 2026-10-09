import importlib.util

import tilelang
import tilelang.language as T
import torch
import torch.nn as nn


# ROPE 参考代码
#
# def rotate_half(x):
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)
#
# def apply_rotary_pos_emb_python(q, k, cos, sin):
#     # q, k: (seq_length, num_heads, head_dim), (4096, 16, 64)
#     # cos, sin: (seq_length, rope_half), (4096, 64)
#     cos = cos.unsqueeze(1)
#     sin = sin.unsqueeze(1)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RuntimeAutotuner = load_module("vision_prefill_qkv_rope_autotuner", "/root/Test/aicas2026gc/autotuner.py").RuntimeAutotuner
load_module("vision_prefill_qkv_rope_ref", "/root/Test/aicas2026gc/ops/rope.py")

num_heads = 16
hidden_size = 1024
head_dim = hidden_size // num_heads  # 64
rope_half = head_dim // 2  # 32
qkv_out = hidden_size * 3
q_tiles = hidden_size // 128  # 1024 // 128 = 8
qk_tiles = q_tiles * 2  # 16
seq_lengths = [4096, 4096 - 64]
qkv = nn.Linear(hidden_size, qkv_out).cuda().half()


@tilelang.jit(out_idx=[-1])
def build_tilelang_qkv_rope_kernel(block_m=64, block_n=128, block_k=64, num_stages=3, threads=128):
    num_tokens = T.dynamic("num_tokens")
    in_dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden_size), in_dtype),
        w: T.Tensor((qkv_out, hidden_size), in_dtype),
        b: T.Tensor((qkv_out,), in_dtype),
        cos: T.Tensor((num_tokens, rope_half), in_dtype),
        sin: T.Tensor((num_tokens, rope_half), in_dtype),
        out: T.Tensor((num_tokens, qkv_out), in_dtype),
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m), qkv_out // block_n, threads=threads) as (bx, by):
            acc = T.alloc_fragment((block_m, block_n), accum_dtype)
            acc_shared = T.alloc_shared((block_m, block_n), in_dtype)
            bias_shared = T.alloc_shared((block_n,), in_dtype)
            cos_shared = T.alloc_shared((block_m, rope_half), in_dtype)
            sin_shared = T.alloc_shared((block_m, rope_half), in_dtype)
            T.annotate_layout({acc_shared: tilelang.layout.make_swizzled_layout(acc_shared)})

            T.clear(acc)
            T.copy(b[by * block_n], bias_shared)

            if by < qk_tiles:
                T.copy(cos[bx * block_m, 0], cos_shared)
                T.copy(sin[bx * block_m, 0], sin_shared)

            for bk in T.Pipelined(hidden_size // block_k, num_stages=num_stages):
                x_shared = T.alloc_shared((block_m, block_k), in_dtype)
                w_shared = T.alloc_shared((block_n, block_k), in_dtype)
                T.annotate_layout({x_shared: tilelang.layout.make_swizzled_layout(x_shared), w_shared: tilelang.layout.make_swizzled_layout(w_shared)})
                T.copy(x[bx * block_m, bk * block_k], x_shared)
                T.copy(w[by * block_n, bk * block_k], w_shared)
                T.gemm(x_shared, w_shared, acc, transpose_B=True, clear_accum=False)

            if by < qk_tiles:
                for i, j in T.Parallel(block_m, block_n):
                    acc_shared[i, j] = T.Cast(in_dtype, acc[i, j] + T.Cast(accum_dtype, bias_shared[j]))
                for h in T.serial(block_n // head_dim):
                    base = h * head_dim
                    for i, d in T.Parallel(block_m, rope_half):
                        q1 = T.Cast(accum_dtype, acc_shared[i, base + d])
                        q2 = T.Cast(accum_dtype, acc_shared[i, base + rope_half + d])
                        c = T.Cast(accum_dtype, cos_shared[i, d])
                        s = T.Cast(accum_dtype, sin_shared[i, d])
                        out[bx * block_m + i, by * block_n + base + d] = T.Cast(in_dtype, q1 * c - q2 * s)
                        out[bx * block_m + i, by * block_n + base + rope_half + d] = T.Cast(in_dtype, q2 * c + q1 * s)
            else:
                for i, j in T.Parallel(block_m, block_n):
                    out[bx * block_m + i, by * block_n + j] = T.Cast(in_dtype, acc[i, j] + T.Cast(accum_dtype, bias_shared[j]))

    return main


tilelang_kernel = build_tilelang_qkv_rope_kernel()


def vision_prefill_module(hidden_states, qkv, cos, sin):
    seq_length = hidden_states.shape[0]
    qkv_states = qkv(hidden_states)
    query_states, key_states, value_states = qkv_states.reshape(seq_length, 3, num_heads, head_dim).unbind(1)
    query_states, key_states = torch.ops.qwen3vl.apply_rotary_pos_emb_vision_triton(query_states, key_states, cos, sin, inplace=False)
    return query_states, key_states, value_states


def vision_prefill_tilelang(hidden_states, qkv, cos, sin):
    seq_length = hidden_states.shape[0]
    qkv_states = tilelang_kernel(hidden_states, qkv.weight, qkv.bias, cos, sin)
    query_states, key_states, value_states = qkv_states.reshape(seq_length, 3, num_heads, head_dim).unbind(1)
    return query_states, key_states, value_states


tuner = RuntimeAutotuner(name="vision_prefill_fused_qkv_rope", fallback="module", strict=False)

for seq_length in seq_lengths:
    hidden_states = torch.randn(seq_length, hidden_size, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq_length, rope_half, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq_length, rope_half, device="cuda", dtype=torch.float16)

    candidates = {
        "module": vision_prefill_module,
        "tilelang": vision_prefill_tilelang,
    }

    outputs = tuner.dispatch_once(candidates, name=f"seq{seq_length}", args=(hidden_states, qkv, cos, sin))
    print(seq_length, tuner.dispatch_table[f"once:seq{seq_length}"], [x.shape for x in outputs])
