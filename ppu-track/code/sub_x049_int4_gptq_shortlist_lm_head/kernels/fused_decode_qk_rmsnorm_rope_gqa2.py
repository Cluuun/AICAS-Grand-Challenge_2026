import torch
import triton
import triton.language as tl


@triton.jit
def _decode_qk_rmsnorm_rope_gqa2_kernel(
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
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_d,
    stride_cos_b,
    stride_cos_d,
    stride_qo_b,
    stride_qo_h,
    stride_qo_d,
    stride_ko_b,
    stride_ko_h,
    stride_ko_d,
    kv_heads,
    head_dim,
    q_eps,
    k_eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // kv_heads
    kv_head = row - batch * kv_heads
    q_head0 = kv_head * 2
    q_head1 = q_head0 + 1
    half = head_dim // 2

    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    rot_offs = tl.where(offs < half, offs + half, offs - half)
    rot_sign = tl.where(offs < half, -1.0, 1.0)

    cos_vals = tl.load(cos_ptr + batch * stride_cos_b + offs * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin_vals = tl.load(sin_ptr + batch * stride_cos_b + offs * stride_cos_d, mask=mask, other=0.0).to(tl.float32)

    q_w = tl.load(q_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q_w_rot = tl.load(q_w_ptr + rot_offs, mask=mask, other=0.0).to(tl.float32)
    k_w = tl.load(k_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    k_w_rot = tl.load(k_w_ptr + rot_offs, mask=mask, other=0.0).to(tl.float32)

    q0_ptr = q_ptr + batch * stride_q_b + q_head0 * stride_q_h
    q1_ptr = q_ptr + batch * stride_q_b + q_head1 * stride_q_h
    k_ptr_row = k_ptr + batch * stride_k_b + kv_head * stride_k_h

    q0 = tl.load(q0_ptr + offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
    q1 = tl.load(q1_ptr + offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr_row + offs * stride_k_d, mask=mask, other=0.0).to(tl.float32)

    q0_rstd = tl.rsqrt(tl.sum(q0 * q0, axis=0) / head_dim + q_eps)
    q1_rstd = tl.rsqrt(tl.sum(q1 * q1, axis=0) / head_dim + q_eps)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=0) / head_dim + k_eps)

    q0_norm = q0 * q0_rstd
    q1_norm = q1 * q1_rstd
    k_norm = k * k_rstd

    q0_rot = tl.load(q0_ptr + rot_offs * stride_q_d, mask=mask, other=0.0).to(tl.float32) * q0_rstd
    q1_rot = tl.load(q1_ptr + rot_offs * stride_q_d, mask=mask, other=0.0).to(tl.float32) * q1_rstd
    k_rot = tl.load(k_ptr_row + rot_offs * stride_k_d, mask=mask, other=0.0).to(tl.float32) * k_rstd

    q0_out = (q0_norm * q_w * cos_vals) + (q0_rot * q_w_rot * rot_sign * sin_vals)
    q1_out = (q1_norm * q_w * cos_vals) + (q1_rot * q_w_rot * rot_sign * sin_vals)
    k_out = (k_norm * k_w * cos_vals) + (k_rot * k_w_rot * rot_sign * sin_vals)

    tl.store(
        q_out_ptr + batch * stride_qo_b + q_head0 * stride_qo_h + offs * stride_qo_d,
        q0_out,
        mask=mask,
    )
    tl.store(
        q_out_ptr + batch * stride_qo_b + q_head1 * stride_qo_h + offs * stride_qo_d,
        q1_out,
        mask=mask,
    )
    tl.store(
        k_out_ptr + batch * stride_ko_b + kv_head * stride_ko_h + offs * stride_ko_d,
        k_out,
        mask=mask,
    )


@triton.jit
def _decode_qk_rmsnorm_rope_kv_update_gqa2_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_w_ptr,
    k_w_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_pos_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_d,
    stride_cos_b,
    stride_cos_d,
    stride_qo_b,
    stride_qo_h,
    stride_qo_d,
    stride_kc_b,
    stride_kc_h,
    stride_kc_t,
    stride_kc_d,
    stride_vc_b,
    stride_vc_h,
    stride_vc_t,
    stride_vc_d,
    kv_heads,
    head_dim,
    q_eps,
    k_eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // kv_heads
    kv_head = row - batch * kv_heads
    q_head0 = kv_head * 2
    q_head1 = q_head0 + 1
    half = head_dim // 2
    pos = tl.load(cache_pos_ptr)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    rot_offs = tl.where(offs < half, offs + half, offs - half)
    rot_sign = tl.where(offs < half, -1.0, 1.0)

    cos_vals = tl.load(cos_ptr + batch * stride_cos_b + offs * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
    sin_vals = tl.load(sin_ptr + batch * stride_cos_b + offs * stride_cos_d, mask=mask, other=0.0).to(tl.float32)

    q_w = tl.load(q_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q_w_rot = tl.load(q_w_ptr + rot_offs, mask=mask, other=0.0).to(tl.float32)
    k_w = tl.load(k_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    k_w_rot = tl.load(k_w_ptr + rot_offs, mask=mask, other=0.0).to(tl.float32)

    q0_ptr = q_ptr + batch * stride_q_b + q_head0 * stride_q_h
    q1_ptr = q_ptr + batch * stride_q_b + q_head1 * stride_q_h
    k_ptr_row = k_ptr + batch * stride_k_b + kv_head * stride_k_h
    v_ptr_row = v_ptr + batch * stride_v_b + kv_head * stride_v_h

    q0 = tl.load(q0_ptr + offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
    q1 = tl.load(q1_ptr + offs * stride_q_d, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr_row + offs * stride_k_d, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr_row + offs * stride_v_d, mask=mask, other=0.0)

    q0_rstd = tl.rsqrt(tl.sum(q0 * q0, axis=0) / head_dim + q_eps)
    q1_rstd = tl.rsqrt(tl.sum(q1 * q1, axis=0) / head_dim + q_eps)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=0) / head_dim + k_eps)

    q0_norm = q0 * q0_rstd
    q1_norm = q1 * q1_rstd
    k_norm = k * k_rstd

    q0_rot = tl.load(q0_ptr + rot_offs * stride_q_d, mask=mask, other=0.0).to(tl.float32) * q0_rstd
    q1_rot = tl.load(q1_ptr + rot_offs * stride_q_d, mask=mask, other=0.0).to(tl.float32) * q1_rstd
    k_rot = tl.load(k_ptr_row + rot_offs * stride_k_d, mask=mask, other=0.0).to(tl.float32) * k_rstd

    q0_out = (q0_norm * q_w * cos_vals) + (q0_rot * q_w_rot * rot_sign * sin_vals)
    q1_out = (q1_norm * q_w * cos_vals) + (q1_rot * q_w_rot * rot_sign * sin_vals)
    k_out = (k_norm * k_w * cos_vals) + (k_rot * k_w_rot * rot_sign * sin_vals)

    tl.store(
        q_out_ptr + batch * stride_qo_b + q_head0 * stride_qo_h + offs * stride_qo_d,
        q0_out,
        mask=mask,
    )
    tl.store(
        q_out_ptr + batch * stride_qo_b + q_head1 * stride_qo_h + offs * stride_qo_d,
        q1_out,
        mask=mask,
    )
    tl.store(
        k_cache_ptr + batch * stride_kc_b + kv_head * stride_kc_h + pos * stride_kc_t + offs * stride_kc_d,
        k_out,
        mask=mask,
    )
    tl.store(
        v_cache_ptr + batch * stride_vc_b + kv_head * stride_vc_h + pos * stride_vc_t + offs * stride_vc_d,
        v,
        mask=mask,
    )


def _normalize_decode_tensor(x: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if x.ndim == 4:
        if int(x.shape[2]) != 1:
            raise ValueError(f"{name} expects decode shape [B, H, 1, D] for 4D input")
        return x.squeeze(2), True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"{name} expects x shape [B, H, D] or [B, H, 1, D]")


def _normalize_decode_pos(pos: torch.Tensor, batch_size: int, head_dim: int, name: str) -> torch.Tensor:
    if pos.ndim == 4:
        if int(pos.shape[1]) != 1:
            raise ValueError(f"{name} 4D tensor must have singleton head dimension")
        pos = pos[:, 0, -1, :]
    elif pos.ndim == 3:
        pos = pos[:, -1, :]
    elif pos.ndim != 2:
        raise ValueError(f"{name} must have 2D/3D/4D decode-compatible shape")

    if int(pos.shape[-1]) != head_dim:
        raise ValueError(f"{name} last dim {int(pos.shape[-1])} must match head_dim {head_dim}")
    if int(pos.shape[0]) == 1 and batch_size > 1:
        pos = pos.expand(batch_size, head_dim)
    elif int(pos.shape[0]) != batch_size:
        raise ValueError(f"{name} batch dim {int(pos.shape[0])} must match x batch {batch_size}")
    return pos.contiguous()


def fused_decode_qk_rmsnorm_rope_gqa2(
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
    block_d: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (query.is_cuda and key.is_cuda):
        raise ValueError("fused_decode_qk_rmsnorm_rope_gqa2 requires CUDA tensor inputs")
    if query.dtype != torch.float16 or key.dtype != torch.float16:
        raise ValueError("fused_decode_qk_rmsnorm_rope_gqa2 currently supports fp16")
    if query.device != key.device:
        raise ValueError("query and key must be on the same device")

    q3d, q_expand_back = _normalize_decode_tensor(query, "query")
    k3d, k_expand_back = _normalize_decode_tensor(key, "key")
    batch, query_heads, head_dim = q3d.shape
    key_batch, kv_heads, key_head_dim = k3d.shape
    if key_batch != batch or key_head_dim != head_dim:
        raise ValueError("query/key batch and head_dim must match")
    if query_heads != 2 * kv_heads:
        raise ValueError("fused_decode_qk_rmsnorm_rope_gqa2 requires exactly 2 query heads per KV head")
    if head_dim % 2 != 0:
        raise ValueError("fused_decode_qk_rmsnorm_rope_gqa2 requires even head_dim")

    if q_weight.ndim != 1 or int(q_weight.shape[0]) != head_dim:
        raise ValueError("q_weight must be 1D and match head_dim")
    if k_weight.ndim != 1 or int(k_weight.shape[0]) != head_dim:
        raise ValueError("k_weight must be 1D and match head_dim")
    if q_weight.dtype != q3d.dtype:
        q_weight = q_weight.to(dtype=q3d.dtype)
    if k_weight.dtype != k3d.dtype:
        k_weight = k_weight.to(dtype=k3d.dtype)
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()

    cos2d = _normalize_decode_pos(cos, batch, head_dim, "cos")
    sin2d = _normalize_decode_pos(sin, batch, head_dim, "sin")
    if cos2d.dtype != q3d.dtype:
        cos2d = cos2d.to(dtype=q3d.dtype)
    if sin2d.dtype != q3d.dtype:
        sin2d = sin2d.to(dtype=q3d.dtype)

    if out_query is None:
        q_out3d = torch.empty_like(q3d)
    else:
        q_out3d, _ = _normalize_decode_tensor(out_query, "out_query")
        if tuple(q_out3d.shape) != tuple(q3d.shape):
            raise ValueError("out_query shape must match query shape")
        if q_out3d.dtype != q3d.dtype or q_out3d.device != q3d.device:
            raise ValueError("out_query must match query dtype/device")

    if out_key is None:
        k_out3d = torch.empty_like(k3d)
    else:
        k_out3d, _ = _normalize_decode_tensor(out_key, "out_key")
        if tuple(k_out3d.shape) != tuple(k3d.shape):
            raise ValueError("out_key shape must match key shape")
        if k_out3d.dtype != k3d.dtype or k_out3d.device != k3d.device:
            raise ValueError("out_key must match key dtype/device")

    block_d = int(block_d) if block_d is not None and int(block_d) > 0 else int(triton.next_power_of_2(head_dim))
    num_warps = int(num_warps)
    num_stages = int(num_stages)
    _decode_qk_rmsnorm_rope_gqa2_kernel[(batch * kv_heads,)](
        q3d,
        k3d,
        q_weight,
        k_weight,
        cos2d,
        sin2d,
        q_out3d,
        k_out3d,
        q3d.stride(0),
        q3d.stride(1),
        q3d.stride(2),
        k3d.stride(0),
        k3d.stride(1),
        k3d.stride(2),
        cos2d.stride(0),
        cos2d.stride(1),
        q_out3d.stride(0),
        q_out3d.stride(1),
        q_out3d.stride(2),
        k_out3d.stride(0),
        k_out3d.stride(1),
        k_out3d.stride(2),
        kv_heads,
        head_dim,
        float(q_eps),
        float(k_eps),
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    q_out = q_out3d.unsqueeze(2) if q_expand_back else q_out3d
    k_out = k_out3d.unsqueeze(2) if k_expand_back else k_out3d
    return q_out, k_out


def fused_decode_qk_rmsnorm_rope_kv_update_gqa2(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache_position: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k_weight: torch.Tensor,
    k_eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    out_query: torch.Tensor | None = None,
    block_d: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    if not (query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda):
        raise ValueError("fused_decode_qk_rmsnorm_rope_kv_update_gqa2 requires CUDA tensor inputs")
    if query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16:
        raise ValueError("fused_decode_qk_rmsnorm_rope_kv_update_gqa2 currently supports fp16")
    if key_cache.dtype != key.dtype or value_cache.dtype != value.dtype:
        raise ValueError("cache dtype must match state dtype")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query/key/value must be on the same device")
    if key_cache.device != query.device or value_cache.device != query.device:
        raise ValueError("cache and state tensors must be on the same device")
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("key_cache and value_cache must have the same shape")
    if not torch.is_tensor(cache_position) or cache_position.numel() != 1:
        raise ValueError("cache_position must contain exactly one element")
    if cache_position.device != query.device:
        raise ValueError("cache_position must be on the same device as query")
    if cache_position.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache_position must be int32/int64")

    q3d, q_expand_back = _normalize_decode_tensor(query, "query")
    k3d, _ = _normalize_decode_tensor(key, "key")
    v3d, _ = _normalize_decode_tensor(value, "value")
    batch, query_heads, head_dim = q3d.shape
    key_batch, kv_heads, key_head_dim = k3d.shape
    if tuple(v3d.shape) != tuple(k3d.shape):
        raise ValueError("value shape must match key shape")
    if key_batch != batch or key_head_dim != head_dim:
        raise ValueError("query/key batch and head_dim must match")
    if query_heads != 2 * kv_heads:
        raise ValueError("fused_decode_qk_rmsnorm_rope_kv_update_gqa2 requires exactly 2 query heads per KV head")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("cache tensors must be 4D")
    if int(key_cache.shape[0]) != batch or int(key_cache.shape[1]) != kv_heads or int(key_cache.shape[3]) != head_dim:
        raise ValueError("cache shape must match key states")
    if head_dim % 2 != 0:
        raise ValueError("fused_decode_qk_rmsnorm_rope_kv_update_gqa2 requires even head_dim")

    if q_weight.ndim != 1 or int(q_weight.shape[0]) != head_dim:
        raise ValueError("q_weight must be 1D and match head_dim")
    if k_weight.ndim != 1 or int(k_weight.shape[0]) != head_dim:
        raise ValueError("k_weight must be 1D and match head_dim")
    if q_weight.dtype != q3d.dtype:
        q_weight = q_weight.to(dtype=q3d.dtype)
    if k_weight.dtype != k3d.dtype:
        k_weight = k_weight.to(dtype=k3d.dtype)
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()

    cos2d = _normalize_decode_pos(cos, batch, head_dim, "cos")
    sin2d = _normalize_decode_pos(sin, batch, head_dim, "sin")
    if cos2d.dtype != q3d.dtype:
        cos2d = cos2d.to(dtype=q3d.dtype)
    if sin2d.dtype != q3d.dtype:
        sin2d = sin2d.to(dtype=q3d.dtype)

    if out_query is None:
        q_out3d = torch.empty_like(q3d)
    else:
        q_out3d, _ = _normalize_decode_tensor(out_query, "out_query")
        if tuple(q_out3d.shape) != tuple(q3d.shape):
            raise ValueError("out_query shape must match query shape")
        if q_out3d.dtype != q3d.dtype or q_out3d.device != q3d.device:
            raise ValueError("out_query must match query dtype/device")

    block_d = int(block_d) if block_d is not None and int(block_d) > 0 else int(triton.next_power_of_2(head_dim))
    num_warps = int(num_warps)
    num_stages = int(num_stages)
    _decode_qk_rmsnorm_rope_kv_update_gqa2_kernel[(batch * kv_heads,)](
        q3d,
        k3d,
        v3d,
        q_weight,
        k_weight,
        cos2d,
        sin2d,
        q_out3d,
        key_cache,
        value_cache,
        cache_position,
        q3d.stride(0),
        q3d.stride(1),
        q3d.stride(2),
        k3d.stride(0),
        k3d.stride(1),
        k3d.stride(2),
        v3d.stride(0),
        v3d.stride(1),
        v3d.stride(2),
        cos2d.stride(0),
        cos2d.stride(1),
        q_out3d.stride(0),
        q_out3d.stride(1),
        q_out3d.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        kv_heads,
        head_dim,
        float(q_eps),
        float(k_eps),
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q_out3d.unsqueeze(2) if q_expand_back else q_out3d
