from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_prefill_kv_cache_write_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    cache_position_ptr,
    stride_k_b,
    stride_k_h,
    stride_k_t,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_t,
    stride_v_d,
    stride_kc_b,
    stride_kc_h,
    stride_kc_t,
    stride_kc_d,
    stride_vc_b,
    stride_vc_h,
    stride_vc_t,
    stride_vc_d,
    kv_heads,
    seq_len,
    head_dim,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // kv_heads
    head = pid_bh - batch * kv_heads

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    mask_t = offs_t < seq_len
    mask_d = offs_d < head_dim
    mask = mask_t[:, None] & mask_d[None, :]

    cache_pos = tl.load(cache_position_ptr + offs_t, mask=mask_t, other=0)

    key_base = key_ptr + batch * stride_k_b + head * stride_k_h
    value_base = value_ptr + batch * stride_v_b + head * stride_v_h
    key_cache_base = key_cache_ptr + batch * stride_kc_b + head * stride_kc_h
    value_cache_base = value_cache_ptr + batch * stride_vc_b + head * stride_vc_h

    src_key = key_base + offs_t[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
    src_value = value_base + offs_t[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
    dst_key = key_cache_base + cache_pos[:, None] * stride_kc_t + offs_d[None, :] * stride_kc_d
    dst_value = value_cache_base + cache_pos[:, None] * stride_vc_t + offs_d[None, :] * stride_vc_d

    key = tl.load(src_key, mask=mask, other=0.0)
    value = tl.load(src_value, mask=mask, other=0.0)
    tl.store(dst_key, key, mask=mask)
    tl.store(dst_value, value, mask=mask)


def fused_prefill_kv_cache_write(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    *,
    block_t: int = 1,
    block_d: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    if not (key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda):
        raise ValueError("fused_prefill_kv_cache_write requires CUDA tensors")
    if key.dtype != torch.float16 or value.dtype != torch.float16:
        raise ValueError("key/value must be fp16")
    if key_cache.dtype != key.dtype or value_cache.dtype != value.dtype:
        raise ValueError("cache dtype must match key/value dtype")
    if key.ndim != 4 or value.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key/value/cache tensors must be rank-4 [batch, heads, seq, dim]")
    if key.shape != value.shape:
        raise ValueError("key/value shape mismatch")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache/value_cache shape mismatch")
    if key.shape[0] != key_cache.shape[0] or key.shape[1] != key_cache.shape[1] or key.shape[3] != key_cache.shape[3]:
        raise ValueError("cache must match key/value batch, heads and head_dim")
    if cache_position.ndim != 1 or int(cache_position.numel()) != int(key.shape[2]):
        raise ValueError("cache_position must be 1D and match seq_len")
    if cache_position.dtype != torch.long:
        raise ValueError("cache_position must be int64")

    batch, kv_heads, seq_len, head_dim = key.shape
    block_t = int(block_t)
    if block_t <= 0:
        raise ValueError("block_t must be positive")
    if block_d is None:
        block_d = triton.next_power_of_2(int(head_dim))
    block_d = int(block_d)
    if block_d < int(head_dim):
        raise ValueError("block_d must cover head_dim")

    grid = (triton.cdiv(int(seq_len), int(block_t)), int(batch) * int(kv_heads))
    _fused_prefill_kv_cache_write_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        cache_position,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        int(kv_heads),
        int(seq_len),
        int(head_dim),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
