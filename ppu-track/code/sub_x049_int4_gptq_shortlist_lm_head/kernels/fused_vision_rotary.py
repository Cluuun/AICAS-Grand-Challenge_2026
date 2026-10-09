import torch
import triton
import triton.language as tl


@triton.jit
def _fused_apply_rotary_pos_emb_vision_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    stride_qo_s,
    stride_qo_h,
    stride_qo_d,
    stride_ko_s,
    stride_ko_h,
    stride_ko_d,
    num_heads,
    head_dim,
    half_dim,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    seq_idx = row // num_heads
    head_idx = row % num_heads

    offs = blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < head_dim

    q_base = q_ptr + seq_idx * stride_q_s + head_idx * stride_q_h
    k_base = k_ptr + seq_idx * stride_k_s + head_idx * stride_k_h
    qo_base = q_out_ptr + seq_idx * stride_qo_s + head_idx * stride_qo_h
    ko_base = k_out_ptr + seq_idx * stride_ko_s + head_idx * stride_ko_h

    q = tl.load(q_base + offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_base + offs * stride_k_d, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + seq_idx * stride_cos_s + offs * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + seq_idx * stride_sin_s + offs * stride_sin_d, mask=mask, other=0.0).to(tl.float32)

    rot_idx = tl.where(offs < half_dim, offs + half_dim, offs - half_dim)
    rot_sign = tl.where(offs < half_dim, -1.0, 1.0)
    rot_mask = rot_idx < head_dim
    q_rot = tl.load(q_base + rot_idx * stride_q_d, mask=rot_mask, other=0.0).to(tl.float32)
    k_rot = tl.load(k_base + rot_idx * stride_k_d, mask=rot_mask, other=0.0).to(tl.float32)
    q_rot = q_rot * rot_sign
    k_rot = k_rot * rot_sign

    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin

    tl.store(qo_base + offs * stride_qo_d, q_out, mask=mask)
    tl.store(ko_base + offs * stride_ko_d, k_out, mask=mask)


def fused_apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out_q: torch.Tensor = None,
    out_k: torch.Tensor = None,
):
    if not (q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda):
        raise ValueError("fused_apply_rotary_pos_emb_vision requires CUDA tensors")
    if q.dtype != torch.float16 or k.dtype != torch.float16:
        raise ValueError("q/k must be fp16")
    if q.shape != k.shape or q.ndim != 3:
        raise ValueError("q and k must share shape [seq_len, num_heads, head_dim]")
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("cos and sin must share shape [seq_len, head_dim]")
    seq_len, num_heads, head_dim = q.shape
    if cos.shape[0] != seq_len or cos.shape[1] != head_dim:
        raise ValueError("cos/sin shape mismatch with q/k")
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    if out_q is None:
        out_q = torch.empty_like(q)
    if out_k is None:
        out_k = torch.empty_like(k)
    if out_q.shape != q.shape or out_k.shape != k.shape:
        raise ValueError("out_q/out_k shape mismatch")

    block_d = 128 if head_dim > 64 else 64
    grid = (seq_len * num_heads, triton.cdiv(head_dim, block_d))
    _fused_apply_rotary_pos_emb_vision_kernel[grid](
        q,
        k,
        cos,
        sin,
        out_q,
        out_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        num_heads,
        head_dim,
        head_dim // 2,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out_q, out_k
