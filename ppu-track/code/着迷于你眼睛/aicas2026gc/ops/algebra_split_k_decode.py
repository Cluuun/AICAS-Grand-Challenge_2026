import importlib.util
from pathlib import Path

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_with_kvcache


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_D": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_D": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 32}, num_stages=3, num_warps=4),
    ],
    key=["cg_bucket_size"],
)
@triton.jit
def split_k_attn_local_kernel_opt(
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Acc_partial_ptr,
    M_partial_ptr,
    L_partial_ptr,
    seq_len_ptr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kc_s: tl.constexpr,
    stride_kc_h: tl.constexpr,
    stride_kc_d: tl.constexpr,
    stride_vc_s: tl.constexpr,
    stride_vc_h: tl.constexpr,
    stride_vc_d: tl.constexpr,
    stride_ap_h: tl.constexpr,
    stride_ap_c: tl.constexpr,
    stride_ap_d: tl.constexpr,
    stride_mp_h: tl.constexpr,
    stride_mp_c: tl.constexpr,
    stride_lp_h: tl.constexpr,
    stride_lp_c: tl.constexpr,
    cg_bucket_size: int,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GQA_GROUP: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    seq_len = tl.load(seq_len_ptr)
    active_chunks = tl.cdiv(seq_len, BLOCK_SEQ)

    q0_head = kv_head_idx * GQA_GROUP + 0
    q1_head = kv_head_idx * GQA_GROUP + 1

    if chunk_idx >= active_chunks:
        tl.store(M_partial_ptr + q0_head * stride_mp_h + chunk_idx * stride_mp_c, -float("inf"))
        tl.store(M_partial_ptr + q1_head * stride_mp_h + chunk_idx * stride_mp_c, -float("inf"))
        tl.store(L_partial_ptr + q0_head * stride_lp_h + chunk_idx * stride_lp_c, 0.0)
        tl.store(L_partial_ptr + q1_head * stride_lp_h + chunk_idx * stride_lp_c, 0.0)
        return

    start_seq = chunk_idx * BLOCK_SEQ
    offs_seq = start_seq + tl.arange(0, BLOCK_SEQ)
    seq_mask = offs_seq < seq_len

    qk0 = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
    qk1 = tl.zeros([BLOCK_SEQ], dtype=tl.float32)

    for d in range(0, HEAD_DIM, BLOCK_D):
        offs_d = d + tl.arange(0, BLOCK_D)

        q0_ptrs = Q_ptr + q0_head * stride_qh + offs_d * stride_qd
        q1_ptrs = Q_ptr + q1_head * stride_qh + offs_d * stride_qd

        if BLOCK_D == HEAD_DIM:
            q0 = tl.load(q0_ptrs)
            q1 = tl.load(q1_ptrs)

            k_ptrs = K_cache_ptr + offs_seq[:, None] * stride_kc_s + kv_head_idx * stride_kc_h + offs_d[None, :] * stride_kc_d
            k_blk = tl.load(k_ptrs, mask=seq_mask[:, None], other=0.0)
        else:
            d_mask = offs_d < HEAD_DIM
            q0 = tl.load(q0_ptrs, mask=d_mask, other=0.0)
            q1 = tl.load(q1_ptrs, mask=d_mask, other=0.0)

            kd_mask = seq_mask[:, None] & d_mask[None, :]
            k_ptrs = K_cache_ptr + offs_seq[:, None] * stride_kc_s + kv_head_idx * stride_kc_h + offs_d[None, :] * stride_kc_d
            k_blk = tl.load(k_ptrs, mask=kd_mask, other=0.0)

        qk0 += tl.sum(k_blk * q0[None, :], axis=1)
        qk1 += tl.sum(k_blk * q1[None, :], axis=1)

    qk0 = tl.where(seq_mask, qk0, -float("inf"))
    qk1 = tl.where(seq_mask, qk1, -float("inf"))

    m0 = tl.max(qk0, axis=0)
    m1 = tl.max(qk1, axis=0)

    p0 = tl.exp(qk0 - m0)
    p1 = tl.exp(qk1 - m1)

    l0 = tl.sum(p0, axis=0)
    l1 = tl.sum(p1, axis=0)

    for d in range(0, HEAD_DIM, BLOCK_D):
        offs_d = d + tl.arange(0, BLOCK_D)

        v_ptrs = V_cache_ptr + offs_seq[:, None] * stride_vc_s + kv_head_idx * stride_vc_h + offs_d[None, :] * stride_vc_d

        if BLOCK_D == HEAD_DIM:
            v_blk = tl.load(v_ptrs, mask=seq_mask[:, None], other=0.0)

            acc0_blk = tl.sum(p0[:, None] * v_blk, axis=0)
            acc1_blk = tl.sum(p1[:, None] * v_blk, axis=0)

            acc0_ptrs = Acc_partial_ptr + q0_head * stride_ap_h + chunk_idx * stride_ap_c + offs_d * stride_ap_d
            acc1_ptrs = Acc_partial_ptr + q1_head * stride_ap_h + chunk_idx * stride_ap_c + offs_d * stride_ap_d

            tl.store(acc0_ptrs, acc0_blk.to(Acc_partial_ptr.dtype.element_ty))
            tl.store(acc1_ptrs, acc1_blk.to(Acc_partial_ptr.dtype.element_ty))
        else:
            d_mask = offs_d < HEAD_DIM
            vd_mask = seq_mask[:, None] & d_mask[None, :]
            v_blk = tl.load(v_ptrs, mask=vd_mask, other=0.0)

            acc0_blk = tl.sum(p0[:, None] * v_blk, axis=0)
            acc1_blk = tl.sum(p1[:, None] * v_blk, axis=0)

            acc0_ptrs = Acc_partial_ptr + q0_head * stride_ap_h + chunk_idx * stride_ap_c + offs_d * stride_ap_d
            acc1_ptrs = Acc_partial_ptr + q1_head * stride_ap_h + chunk_idx * stride_ap_c + offs_d * stride_ap_d

            tl.store(acc0_ptrs, acc0_blk.to(Acc_partial_ptr.dtype.element_ty), mask=d_mask)
            tl.store(acc1_ptrs, acc1_blk.to(Acc_partial_ptr.dtype.element_ty), mask=d_mask)

    tl.store(M_partial_ptr + q0_head * stride_mp_h + chunk_idx * stride_mp_c, m0)
    tl.store(M_partial_ptr + q1_head * stride_mp_h + chunk_idx * stride_mp_c, m1)

    tl.store(L_partial_ptr + q0_head * stride_lp_h + chunk_idx * stride_lp_c, l0)
    tl.store(L_partial_ptr + q1_head * stride_lp_h + chunk_idx * stride_lp_c, l1)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
    ],
    key=["MAX_CHUNKS", "HEAD_DIM"],
)
@triton.jit
def split_k_attn_merge_kernel_opt(
    Acc_partial_ptr,
    M_partial_ptr,
    L_partial_ptr,
    Out_ptr,
    seq_len_ptr,
    stride_ap_h: tl.constexpr,
    stride_ap_c: tl.constexpr,
    stride_ap_d: tl.constexpr,
    stride_mp_h: tl.constexpr,
    stride_mp_c: tl.constexpr,
    stride_lp_h: tl.constexpr,
    stride_lp_c: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    q_head_idx = tl.program_id(0)

    seq_len = tl.load(seq_len_ptr)
    active_chunks = tl.cdiv(seq_len, BLOCK_SEQ)

    offs_d = tl.arange(0, HEAD_DIM)

    gm = -float("inf")
    gl = 0.0
    ga = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for chunk_idx in range(MAX_CHUNKS):
        if chunk_idx < active_chunks:
            m_i = tl.load(M_partial_ptr + q_head_idx * stride_mp_h + chunk_idx * stride_mp_c)
            l_i = tl.load(L_partial_ptr + q_head_idx * stride_lp_h + chunk_idx * stride_lp_c)

            if l_i > 0:
                acc_i = tl.load(Acc_partial_ptr + q_head_idx * stride_ap_h + chunk_idx * stride_ap_c + offs_d * stride_ap_d).to(tl.float32)

                mn = tl.maximum(gm, m_i)
                ag = tl.exp(gm - mn)
                ai = tl.exp(m_i - mn)

                ga = ga * ag + acc_i * ai
                gl = gl * ag + l_i * ai
                gm = mn

    out_ptrs = Out_ptr + q_head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, (ga / gl).to(Out_ptr.dtype.element_ty))


def split_k_decode_attention(
    q_states,
    k_cache,
    v_cache,
    cache_seqlens,
    acc_partial_buffer,
    m_partial_buffer,
    l_partial_buffer,
    cg_bucket_size,
):
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    BLOCK_SEQ = 128

    if q_states.dim() == 3:
        q_states = q_states.squeeze(0)

    k_cache = k_cache.squeeze(0)
    v_cache = v_cache.squeeze(0)

    MAX_CHUNKS = (cg_bucket_size + BLOCK_SEQ - 1) // BLOCK_SEQ
    grid_local = (NUM_KV_HEADS, MAX_CHUNKS)

    split_k_attn_local_kernel_opt[grid_local](
        q_states,
        k_cache,
        v_cache,
        acc_partial_buffer,
        m_partial_buffer,
        l_partial_buffer,
        cache_seqlens,
        q_states.stride(0),
        q_states.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        acc_partial_buffer.stride(0),
        acc_partial_buffer.stride(1),
        acc_partial_buffer.stride(2),
        m_partial_buffer.stride(0),
        m_partial_buffer.stride(1),
        l_partial_buffer.stride(0),
        l_partial_buffer.stride(1),
        cg_bucket_size=cg_bucket_size,
        BLOCK_SEQ=BLOCK_SEQ,
        HEAD_DIM=HEAD_DIM,
        GQA_GROUP=NUM_Q_HEADS // NUM_KV_HEADS,
    )

    grid_merge = (NUM_Q_HEADS,)
    split_k_attn_merge_kernel_opt[grid_merge](
        acc_partial_buffer,
        m_partial_buffer,
        l_partial_buffer,
        q_states,
        cache_seqlens,
        acc_partial_buffer.stride(0),
        acc_partial_buffer.stride(1),
        acc_partial_buffer.stride(2),
        m_partial_buffer.stride(0),
        m_partial_buffer.stride(1),
        l_partial_buffer.stride(0),
        l_partial_buffer.stride(1),
        q_states.stride(0),
        q_states.stride(1),
        BLOCK_SEQ=BLOCK_SEQ,
        MAX_CHUNKS=MAX_CHUNKS,
        HEAD_DIM=HEAD_DIM,
    )

    return q_states[None, None]


algebra_split_k_decode_cudagraph = split_k_decode_attention


__all__ = ["split_k_decode_attention", "algebra_split_k_decode_cudagraph"]


if __name__ == "__main__":
    try:
        from aicas2026gc.autotuner import RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[2] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("algebra_split_k_decode_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        RuntimeAutotuner = module.RuntimeAutotuner

    torch.manual_seed(0)

    dtype = torch.float16
    device = "cuda"
    seq_len = 512
    cg_bucket_size = 512
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    max_chunks = (4096 + 127) // 128

    q_states = torch.randn(num_q_heads, head_dim, device=device, dtype=dtype) * (head_dim**-0.5)
    q_states_ref = q_states.clone()
    q_states_work = torch.empty_like(q_states)
    k_cache = torch.randn(1, 4096, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(1, 4096, num_kv_heads, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    acc_partial_buffer = torch.empty(num_q_heads, max_chunks, head_dim, device=device, dtype=dtype)
    m_partial_buffer = torch.empty(num_q_heads, max_chunks, device=device, dtype=torch.float32)
    l_partial_buffer = torch.empty(num_q_heads, max_chunks, device=device, dtype=torch.float32)

    tuner = RuntimeAutotuner(
        name="split_k_decode_attention",
        fallback="flash_attn",
        strict=False,
        atol=1e-2,
        rtol=1e-2,
    )

    def flash_attn_ref(q, k, v, seqlens):
        return flash_attn_with_kvcache(
            q=q[None, None],
            k_cache=k,
            v_cache=v,
            cache_seqlens=seqlens,
            softmax_scale=1.0,
        )

    def algebra_impl(q, k, v, seqlens):
        q_states_work.copy_(q)
        return split_k_decode_attention(
            q_states_work,
            k,
            v,
            seqlens,
            acc_partial_buffer,
            m_partial_buffer,
            l_partial_buffer,
            cg_bucket_size,
        )

    candidates = {
        "flash_attn": flash_attn_ref,
        "algebra": algebra_impl,
    }

    bucket_name = f"seq{seq_len}_bucket{cg_bucket_size}_{str(dtype).split('.')[-1]}"
    out = tuner.dispatch_once(candidates, name=bucket_name, args=(q_states, k_cache, v_cache, cache_seqlens))
    ref = flash_attn_ref(q_states_ref, k_cache, v_cache, cache_seqlens)
    max_diff = (ref - out).abs().max().item()
    best = tuner.dispatch_table[f"once:{bucket_name}"]
    print(f"best={best}, max_diff={max_diff:.6f}")
