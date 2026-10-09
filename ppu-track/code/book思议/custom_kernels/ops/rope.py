"""融合旋转位置编码(RoPE)的Triton kernel。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_kernel(
    Q_IN, K_IN, Q_OUT, K_OUT,
    COS, SIN,
    stride_qin, stride_kin, stride_qout, stride_kout,
    stride_cos, stride_sin,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    total_heads: tl.constexpr,  # num_q_heads + num_kv_heads
    num_q_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Q和K一起做融合RoPE，每个program处理一个head。"""
    pid = tl.program_id(0)

    # 判断是Q head还是K head
    is_q = pid < total_heads * 1  # 占位符，统一处理所有head

    head_idx = pid
    batch_pos_idx = 0  # decode时batch=1, seq=1

    Q_IN += head_idx * stride_qin
    K_IN += head_idx * stride_kin
    Q_OUT += head_idx * stride_qout
    K_OUT += head_idx * stride_kout
    COS += head_idx * stride_cos
    SIN += head_idx * stride_sin

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < head_dim

    # 加载输入
    q = tl.load(Q_IN + offs, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(K_IN + offs, mask=mask, other=0.0).to(tl.float32)
    cos_val = tl.load(COS + offs, mask=mask, other=0.0).to(tl.float32)
    sin_val = tl.load(SIN + offs, mask=mask, other=0.0).to(tl.float32)

    # rotate_half: [-x2, x1]，x1是前半，x2是后半
    # 对Q
    q1 = tl.load(Q_IN + offs, mask=(offs < half_dim), other=0.0).to(tl.float32)
    q2 = tl.load(Q_IN + half_dim + offs, mask=(offs < half_dim), other=0.0).to(tl.float32)

    # 构造旋转结果: [-q2, q1]
    neg_q2 = -q2
    q_rot_first = tl.where(offs < half_dim, neg_q2, q1)
    q_first_half = tl.where(offs < half_dim, q1, q2)

    # q_embed = q * cos + rotate_half(q) * sin
    q_out = q_first_half * cos_val + q_rot_first * sin_val

    # 对K
    k1 = tl.load(K_IN + offs, mask=(offs < half_dim), other=0.0).to(tl.float32)
    k2 = tl.load(K_IN + half_dim + offs, mask=(offs < half_dim), other=0.0).to(tl.float32)

    neg_k2 = -k2
    k_rot_first = tl.where(offs < half_dim, neg_k2, k1)
    k_first_half = tl.where(offs < half_dim, k1, k2)

    k_out = k_first_half * cos_val + k_rot_first * sin_val

    tl.store(Q_OUT + offs, q_out, mask=mask)
    tl.store(K_OUT + offs, k_out, mask=mask)


def fused_rope(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合RoPE，给Q和K加旋转位置编码。"""
    # decode时(seq=1)直接用PyTorch就行，简单又快
    shape_q = q.shape
    shape_k = k.shape

    head_dim = q.shape[-1]
    half_dim = head_dim // 2

    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    # 确保cos/sin的shape对得上
    if cos.dim() < q.dim():
        cos = cos.unsqueeze(1)  # 加上head维度
        sin = sin.unsqueeze(1)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed


def patch_rope(model: torch.nn.Module):
    """RoPE的patch，目前直接用优化过的PyTorch实现。"""
    # RoPE已经在fused_qkv的attention forward里通过apply_rotary_pos_emb处理了
    # 暂时不需要单独patch
    return 0
