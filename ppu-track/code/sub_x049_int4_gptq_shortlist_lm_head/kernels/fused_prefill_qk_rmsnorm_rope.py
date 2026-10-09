import torch
import triton
import triton.language as tl


@triton.jit
def _fused_prefill_qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_t,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_t,
    stride_k_d,
    stride_cos_b,
    stride_cos_s,
    stride_cos_d,
    stride_sin_b,
    stride_sin_s,
    stride_sin_d,
    stride_qo_b,
    stride_qo_h,
    stride_qo_t,
    stride_qo_d,
    stride_ko_b,
    stride_ko_h,
    stride_ko_t,
    stride_ko_d,
    q_num_heads,
    kv_num_heads,
    seq_len,
    head_dim,
    cos_batch_size,
    q_eps,
    k_eps,
    q_rows,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    half_dim = head_dim // 2
    rot_offs = tl.where(offs < half_dim, offs + half_dim, offs - half_dim)
    rot_sign = tl.where(offs < half_dim, -1.0, 1.0)

    q_path = row < q_rows
    if q_path:
        q_hs = q_num_heads * seq_len
        batch = row // q_hs
        rem = row - batch * q_hs
        head = rem // seq_len
        seq = rem - head * seq_len
        cos_batch = tl.where(cos_batch_size == 1, 0, batch)

        base = q_ptr + batch * stride_q_b + head * stride_q_h + seq * stride_q_t
        out_base = q_out_ptr + batch * stride_qo_b + head * stride_qo_h + seq * stride_qo_t

        x = tl.load(base + offs * stride_q_d, mask=mask, other=0.0)
        x_fp = x.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x_fp * x_fp, axis=0) / head_dim + q_eps)

        w = tl.load(q_w_ptr + offs, mask=mask, other=0.0)
        norm = ((x_fp * rstd).to(x.dtype) * w).to(x.dtype)

        x_rot = tl.load(base + rot_offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
        w_rot = tl.load(q_w_ptr + rot_offs, mask=mask, other=0.0)
        norm_rot = ((x_rot * rstd).to(x.dtype) * w_rot).to(x.dtype) * rot_sign

        cos = tl.load(
            cos_ptr + cos_batch * stride_cos_b + seq * stride_cos_s + offs * stride_cos_d,
            mask=mask,
            other=0.0,
        )
        sin = tl.load(
            sin_ptr + cos_batch * stride_sin_b + seq * stride_sin_s + offs * stride_sin_d,
            mask=mask,
            other=0.0,
        )
        prod_a = (norm * cos).to(x.dtype)
        prod_b = (norm_rot * sin).to(x.dtype)
        out = (prod_a + prod_b).to(x.dtype)
        tl.store(out_base + offs * stride_qo_d, out, mask=mask)
        return

    k_row = row - q_rows
    kv_hs = kv_num_heads * seq_len
    batch = k_row // kv_hs
    rem = k_row - batch * kv_hs
    head = rem // seq_len
    seq = rem - head * seq_len
    cos_batch = tl.where(cos_batch_size == 1, 0, batch)

    base = k_ptr + batch * stride_k_b + head * stride_k_h + seq * stride_k_t
    out_base = k_out_ptr + batch * stride_ko_b + head * stride_ko_h + seq * stride_ko_t

    x = tl.load(base + offs * stride_k_d, mask=mask, other=0.0)
    x_fp = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x_fp * x_fp, axis=0) / head_dim + k_eps)

    w = tl.load(k_w_ptr + offs, mask=mask, other=0.0)
    norm = ((x_fp * rstd).to(x.dtype) * w).to(x.dtype)

    x_rot = tl.load(base + rot_offs * stride_k_d, mask=mask, other=0.0).to(tl.float32)
    w_rot = tl.load(k_w_ptr + rot_offs, mask=mask, other=0.0)
    norm_rot = ((x_rot * rstd).to(x.dtype) * w_rot).to(x.dtype) * rot_sign

    cos = tl.load(
        cos_ptr + cos_batch * stride_cos_b + seq * stride_cos_s + offs * stride_cos_d,
        mask=mask,
        other=0.0,
    )
    sin = tl.load(
        sin_ptr + cos_batch * stride_sin_b + seq * stride_sin_s + offs * stride_sin_d,
        mask=mask,
        other=0.0,
    )
    prod_a = (norm * cos).to(x.dtype)
    prod_b = (norm_rot * sin).to(x.dtype)
    out = (prod_a + prod_b).to(x.dtype)
    tl.store(out_base + offs * stride_ko_d, out, mask=mask)


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x.float() * torch.rsqrt(variance + eps)
    return weight * x_norm.to(x.dtype)


def reference_prefill_qk_rmsnorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k_weight: torch.Tensor,
    k_eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = _rmsnorm_reference(query, q_weight, q_eps)
    k_norm = _rmsnorm_reference(key, k_weight, k_eps)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_half = query.shape[-1] // 2
    q_rot = torch.cat((-q_norm[..., q_half:], q_norm[..., :q_half]), dim=-1)
    k_rot = torch.cat((-k_norm[..., q_half:], k_norm[..., :q_half]), dim=-1)
    return (q_norm * cos) + (q_rot * sin), (k_norm * cos) + (k_rot * sin)


def fused_prefill_qk_rmsnorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k_weight: torch.Tensor,
    k_eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    out_query: torch.Tensor | None = None,
    out_key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (query.is_cuda and key.is_cuda and cos.is_cuda and sin.is_cuda):
        raise ValueError("fused_prefill_qk_rmsnorm_rope requires CUDA tensors")
    if query.dtype != torch.float16 or key.dtype != torch.float16:
        raise ValueError("fused_prefill_qk_rmsnorm_rope currently supports fp16 q/k")
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query/key must have shape [batch, heads, seq_len, head_dim]")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2] or query.shape[3] != key.shape[3]:
        raise ValueError("query/key batch, seq_len and head_dim must match")
    if cos.ndim != 3 or sin.ndim != 3 or cos.shape != sin.shape:
        raise ValueError("cos/sin must have shape [batch or 1, seq_len, head_dim]")

    batch, q_heads, seq_len, head_dim = query.shape
    _, kv_heads, _, _ = key.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    if cos.shape[1] != seq_len or cos.shape[2] != head_dim:
        raise ValueError("cos/sin shape must match query/key seq_len and head_dim")
    if cos.shape[0] not in (1, batch):
        raise ValueError("cos/sin batch must be 1 or match query/key batch")
    if q_weight.ndim != 1 or int(q_weight.shape[0]) != int(head_dim):
        raise ValueError("q_weight must be 1D and match head_dim")
    if k_weight.ndim != 1 or int(k_weight.shape[0]) != int(head_dim):
        raise ValueError("k_weight must be 1D and match head_dim")

    if q_weight.device != query.device:
        q_weight = q_weight.to(device=query.device)
    if k_weight.device != key.device:
        k_weight = k_weight.to(device=key.device)
    if q_weight.dtype != query.dtype:
        q_weight = q_weight.to(dtype=query.dtype)
    if k_weight.dtype != key.dtype:
        k_weight = k_weight.to(dtype=key.dtype)
    if cos.dtype != query.dtype:
        cos = cos.to(dtype=query.dtype)
    if sin.dtype != query.dtype:
        sin = sin.to(dtype=query.dtype)
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()

    if out_query is None:
        out_query = torch.empty_like(query, memory_format=torch.contiguous_format)
    if out_key is None:
        out_key = torch.empty_like(key, memory_format=torch.contiguous_format)
    if out_query.shape != query.shape or out_query.dtype != query.dtype or out_query.device != query.device:
        raise ValueError("out_query must match query shape/dtype/device")
    if out_key.shape != key.shape or out_key.dtype != key.dtype or out_key.device != key.device:
        raise ValueError("out_key must match key shape/dtype/device")

    block_d = max(16, triton.next_power_of_2(int(head_dim)))
    q_rows = int(batch) * int(q_heads) * int(seq_len)
    total_rows = q_rows + int(batch) * int(kv_heads) * int(seq_len)
    _fused_prefill_qk_rmsnorm_rope_kernel[(total_rows,)](
        query,
        key,
        q_weight,
        k_weight,
        cos,
        sin,
        out_query,
        out_key,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        out_query.stride(0),
        out_query.stride(1),
        out_query.stride(2),
        out_query.stride(3),
        out_key.stride(0),
        out_key.stride(1),
        out_key.stride(2),
        out_key.stride(3),
        int(q_heads),
        int(kv_heads),
        int(seq_len),
        int(head_dim),
        int(cos.shape[0]),
        float(q_eps),
        float(k_eps),
        int(q_rows),
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 128 else 8,
        num_stages=2,
    )
    return out_query, out_key
