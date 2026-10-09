"""flash_attn registered as torch.library.custom_op so torch.compile traces
the FA2 calls without graph breaks."""
from __future__ import annotations

import torch
from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
from flash_attn import flash_attn_func as _flash_attn_func


@torch.library.custom_op(
    'my::flash_attn_kvcache',
    mutates_args=('k_cache', 'v_cache', 'cache_seqlens'))
def flash_attn_kvcache_op(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_new: torch.Tensor, v_new: torch.Tensor, cache_seqlens: torch.Tensor,
    causal: bool, softmax_scale: float,
) -> torch.Tensor:
    return _flash_attn_with_kvcache(
        q, k_cache, v_cache, k_new, v_new,
        cache_seqlens=cache_seqlens, causal=causal, softmax_scale=softmax_scale)


@flash_attn_kvcache_op.register_fake
def _flash_attn_kvcache_fake(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_new: torch.Tensor, v_new: torch.Tensor, cache_seqlens: torch.Tensor,
    causal: bool, softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op('my::flash_attn_vit', mutates_args=())
def flash_attn_vit_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool, softmax_scale: float,
) -> torch.Tensor:
    return _flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)


@flash_attn_vit_op.register_fake
def _flash_attn_vit_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool, softmax_scale: float,
) -> torch.Tensor:
    batch, seq, heads, head_dim = q.shape
    return torch.empty(
        (batch, seq, heads, head_dim),
        device=q.device, dtype=q.dtype, requires_grad=q.requires_grad)
