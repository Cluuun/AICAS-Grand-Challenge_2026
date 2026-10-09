import torch
import triton
import triton.language as tl


@triton.jit
def _fused_prefill_rope_embeddings_kernel(
    pos_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    stride_pos_a,
    stride_pos_b,
    stride_pos_s,
    stride_cos_b,
    stride_cos_s,
    stride_cos_d,
    stride_sin_b,
    stride_sin_s,
    stride_sin_d,
    seq_len,
    half_dim,
    len_h,
    len_w,
    scale,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)

    b = row // seq_len
    s = row - b * seq_len
    offs = blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < half_dim

    pos_t = tl.load(pos_ptr + b * stride_pos_b + s * stride_pos_s + 0 * stride_pos_a).to(tl.float32)
    pos_h = tl.load(pos_ptr + b * stride_pos_b + s * stride_pos_s + 1 * stride_pos_a).to(tl.float32)
    pos_w = tl.load(pos_ptr + b * stride_pos_b + s * stride_pos_s + 2 * stride_pos_a).to(tl.float32)

    inv = tl.load(inv_freq_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mod3 = offs - (offs // 3) * 3

    use_h = (offs < len_h) & (mod3 == 1)
    use_w = (offs < len_w) & (mod3 == 2)
    pos = tl.where(use_h, pos_h, pos_t)
    pos = tl.where(use_w, pos_w, pos)

    angle = pos * inv
    cos_half = tl.cos(angle) * scale
    sin_half = tl.sin(angle) * scale

    cos_base = cos_ptr + b * stride_cos_b + s * stride_cos_s
    sin_base = sin_ptr + b * stride_sin_b + s * stride_sin_s
    tl.store(cos_base + offs * stride_cos_d, cos_half, mask=mask)
    tl.store(sin_base + offs * stride_sin_d, sin_half, mask=mask)

    offs_2 = offs + half_dim
    tl.store(cos_base + offs_2 * stride_cos_d, cos_half, mask=mask)
    tl.store(sin_base + offs_2 * stride_sin_d, sin_half, mask=mask)


def fused_build_qwen3vl_prefill_rope_embeddings(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    mrope_section,
    out_cos: torch.Tensor | None = None,
    out_sin: torch.Tensor | None = None,
):
    if not (position_ids.is_cuda and inv_freq.is_cuda):
        raise ValueError("fused_build_qwen3vl_prefill_rope_embeddings requires CUDA tensors")
    if position_ids.ndim != 3 or int(position_ids.shape[0]) != 3:
        raise ValueError("position_ids must have shape [3, batch, seq_len]")
    if inv_freq.ndim != 1:
        raise ValueError("inv_freq must be a 1D tensor")
    if position_ids.dtype != torch.long:
        position_ids = position_ids.to(dtype=torch.long)
    if inv_freq.dtype != torch.float32:
        inv_freq = inv_freq.to(dtype=torch.float32)
    if not inv_freq.is_contiguous():
        inv_freq = inv_freq.contiguous()

    bsz = int(position_ids.shape[1])
    seq_len = int(position_ids.shape[2])
    half_dim = int(inv_freq.shape[0])
    head_dim = half_dim * 2

    if out_cos is None:
        out_cos = torch.empty((bsz, seq_len, head_dim), device=position_ids.device, dtype=torch.float16)
    if out_sin is None:
        out_sin = torch.empty((bsz, seq_len, head_dim), device=position_ids.device, dtype=torch.float16)
    if out_cos.shape != (bsz, seq_len, head_dim) or out_sin.shape != (bsz, seq_len, head_dim):
        raise ValueError("out_cos/out_sin shape mismatch")
    if out_cos.dtype != torch.float16 or out_sin.dtype != torch.float16:
        raise ValueError("out_cos/out_sin must be fp16")

    len_h = 0
    len_w = 0
    if isinstance(mrope_section, (list, tuple)) and len(mrope_section) >= 3:
        len_h = int(mrope_section[1]) * 3
        len_w = int(mrope_section[2]) * 3
    len_h = max(0, min(len_h, half_dim))
    len_w = max(0, min(len_w, half_dim))

    block_d = 128 if half_dim > 64 else 64
    grid = (bsz * seq_len, triton.cdiv(half_dim, block_d))
    _fused_prefill_rope_embeddings_kernel[grid](
        position_ids,
        inv_freq,
        out_cos,
        out_sin,
        position_ids.stride(0),
        position_ids.stride(1),
        position_ids.stride(2),
        out_cos.stride(0),
        out_cos.stride(1),
        out_cos.stride(2),
        out_sin.stride(0),
        out_sin.stride(1),
        out_sin.stride(2),
        seq_len,
        half_dim,
        len_h,
        len_w,
        float(attention_scaling),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out_cos, out_sin
