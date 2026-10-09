"""自定义decode attention kernel，替代flash_attn_with_kvcache。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attn_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr, K_new_ptr, V_new_ptr, Out_ptr,
    cache_seqlens_ptr,
    seq_stride,  # = num_kv_heads * head_dim
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    kv_group_size: tl.constexpr,  # = num_q_heads // num_kv_heads
    head_dim: tl.constexpr,
    scale,
    BLOCK_SEQ: tl.constexpr,
):
    """decode attention，用online softmax。每个program处理一个Q head。"""
    q_head = tl.program_id(0)
    kv_head = q_head // kv_group_size

    dim = tl.arange(0, head_dim)

    # 读这个head的Q向量
    q = tl.load(Q_ptr + q_head * head_dim + dim).to(tl.float32)

    # 读当前cache长度
    seq_len = tl.load(cache_seqlens_ptr).to(tl.int32)

    # ---- 对cache做online softmax (0到seq_len-1) ----
    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for block_start in range(0, seq_len, BLOCK_SEQ):
        block_off = block_start + tl.arange(0, BLOCK_SEQ)
        mask = block_off < seq_len

        # 从cache加载K block
        k_ptrs = (K_cache_ptr
                  + block_off[:, None] * seq_stride
                  + kv_head * head_dim
                  + dim[None, :])
        k_block = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # QK^T打分
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(mask, scores, float('-inf'))

        # online softmax更新
        m_new = tl.maximum(m_i, tl.max(scores))
        alpha = tl.exp(m_i - m_new)
        beta = tl.where(mask, tl.exp(scores - m_new), 0.0)

        l_i = l_i * alpha + tl.sum(beta)
        acc = acc * alpha

        # 加载V block
        v_ptrs = (V_cache_ptr
                  + block_off[:, None] * seq_stride
                  + kv_head * head_dim
                  + dim[None, :])
        v_block = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # 累加加权的V
        acc += tl.sum(beta[:, None] * v_block, axis=0)
        m_i = m_new

    # ---- 处理新的K/V (在seq_len位置) ----
    k_new = tl.load(K_new_ptr + kv_head * head_dim + dim).to(tl.float32)
    score_new = tl.sum(q * k_new) * scale

    m_new = tl.maximum(m_i, score_new)
    alpha = tl.exp(m_i - m_new)
    beta_new = tl.exp(score_new - m_new)

    l_i = l_i * alpha + beta_new
    acc = acc * alpha

    v_new = tl.load(V_new_ptr + kv_head * head_dim + dim).to(tl.float32)
    acc += beta_new * v_new

    # 最终输出
    acc = acc / l_i

    # 存attention输出
    tl.store(Out_ptr + q_head * head_dim + dim, acc.to(Out_ptr.dtype.element_ty))

    # 更新cache：把新K/V写到seq_len位置，每个KV group只写一次
    if q_head % kv_group_size == 0:
        tl.store(K_cache_ptr + seq_len * seq_stride + kv_head * head_dim + dim,
                 k_new.to(K_cache_ptr.dtype.element_ty))
        tl.store(V_cache_ptr + seq_len * seq_stride + kv_head * head_dim + dim,
                 v_new.to(V_cache_ptr.dtype.element_ty))


def triton_decode_attn(q, k_cache, v_cache, k_new, v_new, cache_seqlens,
                       num_q_heads=16, num_kv_heads=8, head_dim=128):
    """自定义decode attention，替代flash_attn_with_kvcache。"""
    kv_group_size = num_q_heads // num_kv_heads
    seq_stride = num_kv_heads * head_dim
    scale = head_dim ** -0.5

    # 展平成2D给kernel用
    q_flat = q.reshape(num_q_heads, head_dim)
    k_new_flat = k_new.reshape(num_kv_heads, head_dim)
    v_new_flat = v_new.reshape(num_kv_heads, head_dim)

    out = torch.empty(num_q_heads, head_dim, dtype=q.dtype, device=q.device)

    BLOCK_SEQ = 128
    grid = (num_q_heads,)

    _decode_attn_kernel[grid](
        q_flat, k_cache, v_cache, k_new_flat, v_new_flat, out,
        cache_seqlens,
        seq_stride,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        kv_group_size=kv_group_size,
        head_dim=head_dim,
        scale=scale,
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=4,
    )

    return out.reshape(1, 1, num_q_heads, head_dim)
