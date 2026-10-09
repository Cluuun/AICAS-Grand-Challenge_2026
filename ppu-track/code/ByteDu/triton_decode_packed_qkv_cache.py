from __future__ import annotations

import os

import torch
import torch.nn.functional as F

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/junkrat_triton_cache")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        try:
            from triton.language.extra.cuda.libdevice import rsqrt
        except ModuleNotFoundError:
            from triton.language.math import rsqrt
else:
    rsqrt = None


def rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def decode_packed_qkv_cache_reference(
    packed_qkv_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_index: int | torch.Tensor,
    query_output: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_eps: float,
    k_eps: float,
) -> torch.Tensor:
    rows = packed_qkv_states.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_key_value_heads * head_dim

    query_states = packed_qkv_states[:, :q_size].view(rows, num_heads, head_dim)
    key_states = packed_qkv_states[:, q_size : q_size + kv_size].view(rows, num_key_value_heads, head_dim)
    value_states = packed_qkv_states[:, q_size + kv_size :].view(rows, num_key_value_heads, head_dim)

    query_states = F.rms_norm(query_states, (head_dim,), q_weight, q_eps)
    key_states = F.rms_norm(key_states, (head_dim,), k_weight, k_eps)
    cos_row = cos.unsqueeze(1)
    sin_row = sin.unsqueeze(1)
    query_states = (query_states * cos_row) + (rotate_half_reference(query_states) * sin_row)
    key_states = (key_states * cos_row) + (rotate_half_reference(key_states) * sin_row)

    token_value = int(token_index.reshape(1)[0].item()) if isinstance(token_index, torch.Tensor) else int(token_index)
    query_output.copy_(query_states)
    k_cache[token_value : token_value + rows].copy_(key_states)
    v_cache[token_value : token_value + rows].copy_(value_states)
    return query_output


def can_use_triton_decode_packed_qkv_cache(
    packed_qkv_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_index: torch.Tensor | int,
    query_output: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if isinstance(token_index, int):
        return False
    if not (
        packed_qkv_states.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and k_cache.is_cuda
        and v_cache.is_cuda
        and query_output.is_cuda
        and token_index.is_cuda
    ):
        return False
    if not (
        packed_qkv_states.device
        == q_weight.device
        == k_weight.device
        == cos.device
        == sin.device
        == k_cache.device
        == v_cache.device
        == query_output.device
        == token_index.device
    ):
        return False
    if packed_qkv_states.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not (
        packed_qkv_states.dtype
        == query_output.dtype
        == k_cache.dtype
        == v_cache.dtype
        == cos.dtype
        == sin.dtype
    ):
        return False
    if packed_qkv_states.ndim != 2 or packed_qkv_states.shape[0] != 1:
        return False
    if not (
        q_weight.ndim == 1
        and k_weight.ndim == 1
        and q_weight.shape[0] == head_dim
        and k_weight.shape[0] == head_dim
    ):
        return False
    if not (
        cos.shape == sin.shape == (1, head_dim)
        and query_output.shape == (1, num_heads, head_dim)
        and k_cache.ndim == 3
        and v_cache.shape == k_cache.shape
        and k_cache.shape[1] == num_key_value_heads
        and k_cache.shape[2] == head_dim
    ):
        return False
    total_dim = (num_heads + 2 * num_key_value_heads) * head_dim
    if packed_qkv_states.shape[1] != total_dim or head_dim % 2 != 0:
        return False
    if token_index.numel() != 1 or token_index.dtype not in (torch.int32, torch.int64):
        return False
    return (
        packed_qkv_states.is_contiguous()
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and query_output.is_contiguous()
        and token_index.is_contiguous()
    )


if TRITON_AVAILABLE:

    @triton.jit
    def _triton_decode_packed_qkv_cache_inplace(
        packed_qkv_ptr,
        packed_qkv_row_stride,
        q_out_ptr,
        q_out_row_stride,
        k_cache_ptr,
        k_cache_token_stride,
        v_cache_ptr,
        v_cache_token_stride,
        token_index_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        cos_row_stride,
        sin_ptr,
        sin_row_stride,
        n_rows,
        q_eps,
        k_eps,
        q_size: tl.constexpr,
        kv_size: tl.constexpr,
        n_q_head: tl.constexpr,
        n_k_head: tl.constexpr,
        head_dim: tl.constexpr,
        pad_n_q_head: tl.constexpr,
        pad_n_k_head: tl.constexpr,
        pad_half_head_dim: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        half_offsets = tl.arange(0, pad_half_head_dim)
        half_mask = half_offsets < (head_dim // 2)

        packed_base = packed_qkv_ptr + row_idx * packed_qkv_row_stride
        q_out_base = q_out_ptr + row_idx * q_out_row_stride
        token_index = tl.load(token_index_ptr).to(tl.int64) + row_idx
        k_cache_base = k_cache_ptr + token_index * k_cache_token_stride
        v_cache_base = v_cache_ptr + token_index * v_cache_token_stride
        cos_base = cos_ptr + row_idx * cos_row_stride
        sin_base = sin_ptr + row_idx * sin_row_stride

        cos_half = tl.load(cos_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        sin_half = tl.load(sin_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_left = tl.load(q_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_right = tl.load(q_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_left = tl.load(k_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_right = tl.load(k_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)

        q_head_idx = tl.arange(0, pad_n_q_head)
        kv_head_idx = tl.arange(0, pad_n_k_head)
        q_head_offsets = q_head_idx[:, None] * head_dim + half_offsets[None, :]
        kv_head_offsets = kv_head_idx[:, None] * head_dim + half_offsets[None, :]
        q_mask = (q_head_idx[:, None] < n_q_head) & half_mask[None, :]
        kv_mask = (kv_head_idx[:, None] < n_k_head) & half_mask[None, :]

        q_left = tl.load(packed_base + q_head_offsets, mask=q_mask, other=0)
        q_right = tl.load(packed_base + q_head_offsets + (head_dim // 2), mask=q_mask, other=0)
        k_left = tl.load(packed_base + q_size + kv_head_offsets, mask=kv_mask, other=0)
        k_right = tl.load(packed_base + q_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)
        v_left = tl.load(packed_base + q_size + kv_size + kv_head_offsets, mask=kv_mask, other=0)
        v_right = tl.load(packed_base + q_size + kv_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)

        q_dtype = q_left.dtype
        k_dtype = k_left.dtype
        v_dtype = v_left.dtype

        q_left_fp = q_left.to(tl.float32)
        q_right_fp = q_right.to(tl.float32)
        k_left_fp = k_left.to(tl.float32)
        k_right_fp = k_right.to(tl.float32)

        q_mean_square = tl.sum(q_left_fp * q_left_fp + q_right_fp * q_right_fp, axis=1) / head_dim
        k_mean_square = tl.sum(k_left_fp * k_left_fp + k_right_fp * k_right_fp, axis=1) / head_dim
        q_rstd = rsqrt(q_mean_square + q_eps)
        k_rstd = rsqrt(k_mean_square + k_eps)

        q_left_fp = q_left_fp * q_rstd[:, None] * q_weight_left[None, :]
        q_right_fp = q_right_fp * q_rstd[:, None] * q_weight_right[None, :]
        k_left_fp = k_left_fp * k_rstd[:, None] * k_weight_left[None, :]
        k_right_fp = k_right_fp * k_rstd[:, None] * k_weight_right[None, :]

        q_out_left = q_left_fp * cos_half[None, :] - q_right_fp * sin_half[None, :]
        q_out_right = q_right_fp * cos_half[None, :] + q_left_fp * sin_half[None, :]
        k_out_left = k_left_fp * cos_half[None, :] - k_right_fp * sin_half[None, :]
        k_out_right = k_right_fp * cos_half[None, :] + k_left_fp * sin_half[None, :]

        tl.store(q_out_base + q_head_offsets, q_out_left.to(q_dtype), mask=q_mask)
        tl.store(q_out_base + q_head_offsets + (head_dim // 2), q_out_right.to(q_dtype), mask=q_mask)
        tl.store(k_cache_base + kv_head_offsets, k_out_left.to(k_dtype), mask=kv_mask)
        tl.store(k_cache_base + kv_head_offsets + (head_dim // 2), k_out_right.to(k_dtype), mask=kv_mask)
        tl.store(v_cache_base + kv_head_offsets, v_left.to(v_dtype), mask=kv_mask)
        tl.store(v_cache_base + kv_head_offsets + (head_dim // 2), v_right.to(v_dtype), mask=kv_mask)

    @triton.jit
    def _triton_decode_packed_qkv_cache_i8_inplace(
        packed_qkv_ptr,
        packed_qkv_row_stride,
        q_out_ptr,
        q_out_row_stride,
        k_cache_i8_ptr,
        k_cache_token_stride,
        v_cache_i8_ptr,
        v_cache_token_stride,
        k_scale_ptr,
        k_scale_token_stride,
        v_scale_ptr,
        v_scale_token_stride,
        token_index_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        cos_row_stride,
        sin_ptr,
        sin_row_stride,
        n_rows,
        q_eps,
        k_eps,
        q_size: tl.constexpr,
        kv_size: tl.constexpr,
        n_q_head: tl.constexpr,
        n_k_head: tl.constexpr,
        head_dim: tl.constexpr,
        pad_n_q_head: tl.constexpr,
        pad_n_k_head: tl.constexpr,
        pad_half_head_dim: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        half_offsets = tl.arange(0, pad_half_head_dim)
        half_mask = half_offsets < (head_dim // 2)

        packed_base = packed_qkv_ptr + row_idx * packed_qkv_row_stride
        q_out_base = q_out_ptr + row_idx * q_out_row_stride
        token_index = tl.load(token_index_ptr).to(tl.int64) + row_idx
        k_cache_base = k_cache_i8_ptr + token_index * k_cache_token_stride
        v_cache_base = v_cache_i8_ptr + token_index * v_cache_token_stride
        k_scale_base = k_scale_ptr + token_index * k_scale_token_stride
        v_scale_base = v_scale_ptr + token_index * v_scale_token_stride
        cos_base = cos_ptr + row_idx * cos_row_stride
        sin_base = sin_ptr + row_idx * sin_row_stride

        cos_half = tl.load(cos_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        sin_half = tl.load(sin_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_left = tl.load(q_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_right = tl.load(q_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_left = tl.load(k_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_right = tl.load(k_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)

        q_head_idx = tl.arange(0, pad_n_q_head)
        kv_head_idx = tl.arange(0, pad_n_k_head)
        q_head_offsets = q_head_idx[:, None] * head_dim + half_offsets[None, :]
        kv_head_offsets = kv_head_idx[:, None] * head_dim + half_offsets[None, :]
        q_mask = (q_head_idx[:, None] < n_q_head) & half_mask[None, :]
        kv_mask = (kv_head_idx[:, None] < n_k_head) & half_mask[None, :]
        kv_head_mask = kv_head_idx < n_k_head

        q_left = tl.load(packed_base + q_head_offsets, mask=q_mask, other=0)
        q_right = tl.load(packed_base + q_head_offsets + (head_dim // 2), mask=q_mask, other=0)
        k_left = tl.load(packed_base + q_size + kv_head_offsets, mask=kv_mask, other=0)
        k_right = tl.load(packed_base + q_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)
        v_left = tl.load(packed_base + q_size + kv_size + kv_head_offsets, mask=kv_mask, other=0)
        v_right = tl.load(packed_base + q_size + kv_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)

        q_dtype = q_left.dtype
        q_left_fp = q_left.to(tl.float32)
        q_right_fp = q_right.to(tl.float32)
        k_left_fp = k_left.to(tl.float32)
        k_right_fp = k_right.to(tl.float32)
        v_left_fp = v_left.to(tl.float32)
        v_right_fp = v_right.to(tl.float32)

        q_mean_square = tl.sum(q_left_fp * q_left_fp + q_right_fp * q_right_fp, axis=1) / head_dim
        k_mean_square = tl.sum(k_left_fp * k_left_fp + k_right_fp * k_right_fp, axis=1) / head_dim
        q_rstd = rsqrt(q_mean_square + q_eps)
        k_rstd = rsqrt(k_mean_square + k_eps)

        q_left_fp = q_left_fp * q_rstd[:, None] * q_weight_left[None, :]
        q_right_fp = q_right_fp * q_rstd[:, None] * q_weight_right[None, :]
        k_left_fp = k_left_fp * k_rstd[:, None] * k_weight_left[None, :]
        k_right_fp = k_right_fp * k_rstd[:, None] * k_weight_right[None, :]

        q_out_left = q_left_fp * cos_half[None, :] - q_right_fp * sin_half[None, :]
        q_out_right = q_right_fp * cos_half[None, :] + q_left_fp * sin_half[None, :]
        k_out_left = k_left_fp * cos_half[None, :] - k_right_fp * sin_half[None, :]
        k_out_right = k_right_fp * cos_half[None, :] + k_left_fp * sin_half[None, :]

        k_abs = tl.maximum(tl.abs(k_out_left), tl.abs(k_out_right))
        v_abs = tl.maximum(tl.abs(v_left_fp), tl.abs(v_right_fp))
        k_max = tl.max(k_abs, axis=1)
        v_max = tl.max(v_abs, axis=1)
        k_scale = tl.where(k_max > 0.0, k_max / 127.0, 1.0)
        v_scale = tl.where(v_max > 0.0, v_max / 127.0, 1.0)

        k_l_qf = tl.maximum(-127.0, tl.minimum(127.0, k_out_left / k_scale[:, None]))
        k_r_qf = tl.maximum(-127.0, tl.minimum(127.0, k_out_right / k_scale[:, None]))
        v_l_qf = tl.maximum(-127.0, tl.minimum(127.0, v_left_fp / v_scale[:, None]))
        v_r_qf = tl.maximum(-127.0, tl.minimum(127.0, v_right_fp / v_scale[:, None]))
        k_l_q = tl.where(k_l_qf >= 0.0, k_l_qf + 0.5, k_l_qf - 0.5).to(tl.int32)
        k_r_q = tl.where(k_r_qf >= 0.0, k_r_qf + 0.5, k_r_qf - 0.5).to(tl.int32)
        v_l_q = tl.where(v_l_qf >= 0.0, v_l_qf + 0.5, v_l_qf - 0.5).to(tl.int32)
        v_r_q = tl.where(v_r_qf >= 0.0, v_r_qf + 0.5, v_r_qf - 0.5).to(tl.int32)

        tl.store(q_out_base + q_head_offsets, q_out_left.to(q_dtype), mask=q_mask)
        tl.store(q_out_base + q_head_offsets + (head_dim // 2), q_out_right.to(q_dtype), mask=q_mask)
        tl.store(k_cache_base + kv_head_offsets, k_l_q.to(tl.int8), mask=kv_mask)
        tl.store(k_cache_base + kv_head_offsets + (head_dim // 2), k_r_q.to(tl.int8), mask=kv_mask)
        tl.store(v_cache_base + kv_head_offsets, v_l_q.to(tl.int8), mask=kv_mask)
        tl.store(v_cache_base + kv_head_offsets + (head_dim // 2), v_r_q.to(tl.int8), mask=kv_mask)
        tl.store(k_scale_base + kv_head_idx, k_scale, mask=kv_head_mask)
        tl.store(v_scale_base + kv_head_idx, v_scale, mask=kv_head_mask)

    @triton.jit
    def _triton_decode_packed_qkv_cache_k_i8_v_bf16_inplace(
        packed_qkv_ptr,
        packed_qkv_row_stride,
        q_out_ptr,
        q_out_row_stride,
        k_cache_i8_ptr,
        k_cache_token_stride,
        v_cache_ptr,
        v_cache_token_stride,
        k_scale_ptr,
        k_scale_token_stride,
        token_index_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        cos_row_stride,
        sin_ptr,
        sin_row_stride,
        n_rows,
        q_eps,
        k_eps,
        q_size: tl.constexpr,
        kv_size: tl.constexpr,
        n_q_head: tl.constexpr,
        n_k_head: tl.constexpr,
        head_dim: tl.constexpr,
        pad_n_q_head: tl.constexpr,
        pad_n_k_head: tl.constexpr,
        pad_half_head_dim: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        half_offsets = tl.arange(0, pad_half_head_dim)
        half_mask = half_offsets < (head_dim // 2)

        packed_base = packed_qkv_ptr + row_idx * packed_qkv_row_stride
        q_out_base = q_out_ptr + row_idx * q_out_row_stride
        token_index = tl.load(token_index_ptr).to(tl.int64) + row_idx
        k_cache_base = k_cache_i8_ptr + token_index * k_cache_token_stride
        v_cache_base = v_cache_ptr + token_index * v_cache_token_stride
        k_scale_base = k_scale_ptr + token_index * k_scale_token_stride
        cos_base = cos_ptr + row_idx * cos_row_stride
        sin_base = sin_ptr + row_idx * sin_row_stride

        cos_half = tl.load(cos_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        sin_half = tl.load(sin_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_left = tl.load(q_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_right = tl.load(q_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_left = tl.load(k_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_right = tl.load(k_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)

        q_head_idx = tl.arange(0, pad_n_q_head)
        kv_head_idx = tl.arange(0, pad_n_k_head)
        q_head_offsets = q_head_idx[:, None] * head_dim + half_offsets[None, :]
        kv_head_offsets = kv_head_idx[:, None] * head_dim + half_offsets[None, :]
        q_mask = (q_head_idx[:, None] < n_q_head) & half_mask[None, :]
        kv_mask = (kv_head_idx[:, None] < n_k_head) & half_mask[None, :]
        kv_head_mask = kv_head_idx < n_k_head

        q_left = tl.load(packed_base + q_head_offsets, mask=q_mask, other=0)
        q_right = tl.load(packed_base + q_head_offsets + (head_dim // 2), mask=q_mask, other=0)
        k_left = tl.load(packed_base + q_size + kv_head_offsets, mask=kv_mask, other=0)
        k_right = tl.load(packed_base + q_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)
        v_left = tl.load(packed_base + q_size + kv_size + kv_head_offsets, mask=kv_mask, other=0)
        v_right = tl.load(packed_base + q_size + kv_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)

        q_dtype = q_left.dtype
        v_dtype = v_left.dtype
        q_left_fp = q_left.to(tl.float32)
        q_right_fp = q_right.to(tl.float32)
        k_left_fp = k_left.to(tl.float32)
        k_right_fp = k_right.to(tl.float32)

        q_mean_square = tl.sum(q_left_fp * q_left_fp + q_right_fp * q_right_fp, axis=1) / head_dim
        k_mean_square = tl.sum(k_left_fp * k_left_fp + k_right_fp * k_right_fp, axis=1) / head_dim
        q_rstd = rsqrt(q_mean_square + q_eps)
        k_rstd = rsqrt(k_mean_square + k_eps)

        q_left_fp = q_left_fp * q_rstd[:, None] * q_weight_left[None, :]
        q_right_fp = q_right_fp * q_rstd[:, None] * q_weight_right[None, :]
        k_left_fp = k_left_fp * k_rstd[:, None] * k_weight_left[None, :]
        k_right_fp = k_right_fp * k_rstd[:, None] * k_weight_right[None, :]

        q_out_left = q_left_fp * cos_half[None, :] - q_right_fp * sin_half[None, :]
        q_out_right = q_right_fp * cos_half[None, :] + q_left_fp * sin_half[None, :]
        k_out_left = k_left_fp * cos_half[None, :] - k_right_fp * sin_half[None, :]
        k_out_right = k_right_fp * cos_half[None, :] + k_left_fp * sin_half[None, :]

        k_abs = tl.maximum(tl.abs(k_out_left), tl.abs(k_out_right))
        k_max = tl.max(k_abs, axis=1)
        k_scale = tl.where(k_max > 0.0, k_max / 127.0, 1.0)
        k_l_qf = tl.maximum(-127.0, tl.minimum(127.0, k_out_left / k_scale[:, None]))
        k_r_qf = tl.maximum(-127.0, tl.minimum(127.0, k_out_right / k_scale[:, None]))
        k_l_q = tl.where(k_l_qf >= 0.0, k_l_qf + 0.5, k_l_qf - 0.5).to(tl.int32)
        k_r_q = tl.where(k_r_qf >= 0.0, k_r_qf + 0.5, k_r_qf - 0.5).to(tl.int32)

        tl.store(q_out_base + q_head_offsets, q_out_left.to(q_dtype), mask=q_mask)
        tl.store(q_out_base + q_head_offsets + (head_dim // 2), q_out_right.to(q_dtype), mask=q_mask)
        tl.store(k_cache_base + kv_head_offsets, k_l_q.to(tl.int8), mask=kv_mask)
        tl.store(k_cache_base + kv_head_offsets + (head_dim // 2), k_r_q.to(tl.int8), mask=kv_mask)
        tl.store(k_scale_base + kv_head_idx, k_scale, mask=kv_head_mask)
        tl.store(v_cache_base + kv_head_offsets, v_left.to(v_dtype), mask=kv_mask)
        tl.store(v_cache_base + kv_head_offsets + (head_dim // 2), v_right.to(v_dtype), mask=kv_mask)

    @triton.jit
    def _triton_prefill_packed_qkv_cache_k_bf16_i8_v_bf16_inplace(
        packed_qkv_ptr,
        packed_qkv_row_stride,
        q_out_ptr,
        q_out_row_stride,
        k_cache_bf16_ptr,
        k_cache_bf16_token_stride,
        k_cache_i8_ptr,
        k_cache_i8_token_stride,
        v_cache_ptr,
        v_cache_token_stride,
        k_scale_ptr,
        k_scale_token_stride,
        token_index_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        cos_row_stride,
        sin_ptr,
        sin_row_stride,
        n_rows,
        q_eps,
        k_eps,
        q_size: tl.constexpr,
        kv_size: tl.constexpr,
        n_q_head: tl.constexpr,
        n_k_head: tl.constexpr,
        head_dim: tl.constexpr,
        pad_n_q_head: tl.constexpr,
        pad_n_k_head: tl.constexpr,
        pad_half_head_dim: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        half_offsets = tl.arange(0, pad_half_head_dim)
        half_mask = half_offsets < (head_dim // 2)

        packed_base = packed_qkv_ptr + row_idx * packed_qkv_row_stride
        q_out_base = q_out_ptr + row_idx * q_out_row_stride
        token_index = tl.load(token_index_ptr).to(tl.int64) + row_idx
        k_cache_bf16_base = k_cache_bf16_ptr + token_index * k_cache_bf16_token_stride
        k_cache_i8_base = k_cache_i8_ptr + token_index * k_cache_i8_token_stride
        v_cache_base = v_cache_ptr + token_index * v_cache_token_stride
        k_scale_base = k_scale_ptr + token_index * k_scale_token_stride
        cos_base = cos_ptr + row_idx * cos_row_stride
        sin_base = sin_ptr + row_idx * sin_row_stride

        cos_half = tl.load(cos_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        sin_half = tl.load(sin_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_left = tl.load(q_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_right = tl.load(q_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_left = tl.load(k_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_right = tl.load(k_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)

        q_head_idx = tl.arange(0, pad_n_q_head)
        kv_head_idx = tl.arange(0, pad_n_k_head)
        q_head_offsets = q_head_idx[:, None] * head_dim + half_offsets[None, :]
        kv_head_offsets = kv_head_idx[:, None] * head_dim + half_offsets[None, :]
        q_mask = (q_head_idx[:, None] < n_q_head) & half_mask[None, :]
        kv_mask = (kv_head_idx[:, None] < n_k_head) & half_mask[None, :]
        kv_head_mask = kv_head_idx < n_k_head

        q_left = tl.load(packed_base + q_head_offsets, mask=q_mask, other=0)
        q_right = tl.load(packed_base + q_head_offsets + (head_dim // 2), mask=q_mask, other=0)
        k_left = tl.load(packed_base + q_size + kv_head_offsets, mask=kv_mask, other=0)
        k_right = tl.load(packed_base + q_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)
        v_left = tl.load(packed_base + q_size + kv_size + kv_head_offsets, mask=kv_mask, other=0)
        v_right = tl.load(packed_base + q_size + kv_size + kv_head_offsets + (head_dim // 2), mask=kv_mask, other=0)

        q_dtype = q_left.dtype
        kv_dtype = v_left.dtype
        q_left_fp = q_left.to(tl.float32)
        q_right_fp = q_right.to(tl.float32)
        k_left_fp = k_left.to(tl.float32)
        k_right_fp = k_right.to(tl.float32)

        q_mean_square = tl.sum(q_left_fp * q_left_fp + q_right_fp * q_right_fp, axis=1) / head_dim
        k_mean_square = tl.sum(k_left_fp * k_left_fp + k_right_fp * k_right_fp, axis=1) / head_dim
        q_rstd = rsqrt(q_mean_square + q_eps)
        k_rstd = rsqrt(k_mean_square + k_eps)

        q_left_fp = q_left_fp * q_rstd[:, None] * q_weight_left[None, :]
        q_right_fp = q_right_fp * q_rstd[:, None] * q_weight_right[None, :]
        k_left_fp = k_left_fp * k_rstd[:, None] * k_weight_left[None, :]
        k_right_fp = k_right_fp * k_rstd[:, None] * k_weight_right[None, :]

        q_out_left = q_left_fp * cos_half[None, :] - q_right_fp * sin_half[None, :]
        q_out_right = q_right_fp * cos_half[None, :] + q_left_fp * sin_half[None, :]
        k_out_left = k_left_fp * cos_half[None, :] - k_right_fp * sin_half[None, :]
        k_out_right = k_right_fp * cos_half[None, :] + k_left_fp * sin_half[None, :]

        k_abs = tl.maximum(tl.abs(k_out_left), tl.abs(k_out_right))
        k_max = tl.max(k_abs, axis=1)
        k_scale = tl.where(k_max > 0.0, k_max / 127.0, 1.0)
        k_l_qf = tl.maximum(-127.0, tl.minimum(127.0, k_out_left / k_scale[:, None]))
        k_r_qf = tl.maximum(-127.0, tl.minimum(127.0, k_out_right / k_scale[:, None]))
        k_l_q = tl.where(k_l_qf >= 0.0, k_l_qf + 0.5, k_l_qf - 0.5).to(tl.int32)
        k_r_q = tl.where(k_r_qf >= 0.0, k_r_qf + 0.5, k_r_qf - 0.5).to(tl.int32)

        tl.store(q_out_base + q_head_offsets, q_out_left.to(q_dtype), mask=q_mask)
        tl.store(q_out_base + q_head_offsets + (head_dim // 2), q_out_right.to(q_dtype), mask=q_mask)
        tl.store(k_cache_bf16_base + kv_head_offsets, k_out_left.to(kv_dtype), mask=kv_mask)
        tl.store(k_cache_bf16_base + kv_head_offsets + (head_dim // 2), k_out_right.to(kv_dtype), mask=kv_mask)
        tl.store(k_cache_i8_base + kv_head_offsets, k_l_q.to(tl.int8), mask=kv_mask)
        tl.store(k_cache_i8_base + kv_head_offsets + (head_dim // 2), k_r_q.to(tl.int8), mask=kv_mask)
        tl.store(k_scale_base + kv_head_idx, k_scale, mask=kv_head_mask)
        tl.store(v_cache_base + kv_head_offsets, v_left.to(kv_dtype), mask=kv_mask)
        tl.store(v_cache_base + kv_head_offsets + (head_dim // 2), v_right.to(kv_dtype), mask=kv_mask)


def decode_packed_qkv_cache_inplace(
    packed_qkv_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_index: int | torch.Tensor,
    query_output: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_eps: float = 1e-6,
    k_eps: float = 1e-6,
) -> torch.Tensor:
    if not can_use_triton_decode_packed_qkv_cache(
        packed_qkv_states,
        q_weight,
        k_weight,
        cos,
        sin,
        k_cache,
        v_cache,
        token_index,
        query_output,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    ):
        return decode_packed_qkv_cache_reference(
            packed_qkv_states,
            q_weight,
            k_weight,
            cos,
            sin,
            k_cache,
            v_cache,
            token_index,
            query_output,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            q_eps=q_eps,
            k_eps=k_eps,
        )

    n_rows = packed_qkv_states.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    pad_n_q_head = triton.next_power_of_2(num_heads)
    pad_n_k_head = triton.next_power_of_2(num_key_value_heads)
    pad_half_head_dim = triton.next_power_of_2(head_dim // 2)
    total_heads = num_heads + 2 * num_key_value_heads
    num_warps = 4 if total_heads <= 32 else 8

    _triton_decode_packed_qkv_cache_inplace[(n_rows,)](
        packed_qkv_states,
        packed_qkv_states.stride(0),
        query_output,
        query_output.stride(0),
        k_cache,
        k_cache.stride(0),
        v_cache,
        v_cache.stride(0),
        token_index,
        q_weight,
        k_weight,
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        n_rows,
        q_eps,
        k_eps,
        q_size,
        kv_size,
        num_heads,
        num_key_value_heads,
        head_dim,
        pad_n_q_head,
        pad_n_k_head,
        pad_half_head_dim,
        num_warps=num_warps,
    )
    return query_output


def decode_packed_qkv_cache_k_i8_v_bf16_inplace(
    packed_qkv_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache_i8: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    token_index: int | torch.Tensor,
    query_output: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_eps: float = 1e-6,
    k_eps: float = 1e-6,
) -> torch.Tensor:
    if not (
        TRITON_AVAILABLE
        and not isinstance(token_index, int)
        and packed_qkv_states.is_cuda
        and k_cache_i8.is_cuda
        and v_cache.is_cuda
        and k_scale.is_cuda
        and packed_qkv_states.ndim == 2
        and packed_qkv_states.shape[0] == 1
        and packed_qkv_states.is_contiguous()
        and k_cache_i8.is_contiguous()
        and v_cache.is_contiguous()
        and k_scale.is_contiguous()
        and k_cache_i8.dtype == torch.int8
        and v_cache.dtype == packed_qkv_states.dtype
        and k_scale.dtype == torch.float32
    ):
        raise RuntimeError("K-INT8/V-BF16 KV cache decode requires contiguous CUDA tensors and tensor token_index.")

    n_rows = packed_qkv_states.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    pad_n_q_head = triton.next_power_of_2(num_heads)
    pad_n_k_head = triton.next_power_of_2(num_key_value_heads)
    pad_half_head_dim = triton.next_power_of_2(head_dim // 2)
    total_heads = num_heads + 2 * num_key_value_heads
    num_warps = 4 if total_heads <= 32 else 8

    _triton_decode_packed_qkv_cache_k_i8_v_bf16_inplace[(n_rows,)](
        packed_qkv_states,
        packed_qkv_states.stride(0),
        query_output,
        query_output.stride(0),
        k_cache_i8,
        k_cache_i8.stride(0),
        v_cache,
        v_cache.stride(0),
        k_scale,
        k_scale.stride(0),
        token_index,
        q_weight,
        k_weight,
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        n_rows,
        q_eps,
        k_eps,
        q_size,
        kv_size,
        num_heads,
        num_key_value_heads,
        head_dim,
        pad_n_q_head,
        pad_n_k_head,
        pad_half_head_dim,
        num_warps=num_warps,
    )
    return query_output


def decode_packed_qkv_cache_i8_inplace(
    packed_qkv_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache_i8: torch.Tensor,
    v_cache_i8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    token_index: int | torch.Tensor,
    query_output: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_eps: float = 1e-6,
    k_eps: float = 1e-6,
) -> torch.Tensor:
    if not (
        TRITON_AVAILABLE
        and not isinstance(token_index, int)
        and packed_qkv_states.is_cuda
        and k_cache_i8.is_cuda
        and v_cache_i8.is_cuda
        and k_scale.is_cuda
        and v_scale.is_cuda
        and packed_qkv_states.ndim == 2
        and packed_qkv_states.shape[0] == 1
        and packed_qkv_states.is_contiguous()
        and k_cache_i8.is_contiguous()
        and v_cache_i8.is_contiguous()
        and k_scale.is_contiguous()
        and v_scale.is_contiguous()
        and k_cache_i8.dtype == torch.int8
        and v_cache_i8.dtype == torch.int8
        and k_scale.dtype == torch.float32
        and v_scale.dtype == torch.float32
    ):
        raise RuntimeError("INT8 KV cache decode requires contiguous CUDA tensors and tensor token_index.")

    n_rows = packed_qkv_states.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    pad_n_q_head = triton.next_power_of_2(num_heads)
    pad_n_k_head = triton.next_power_of_2(num_key_value_heads)
    pad_half_head_dim = triton.next_power_of_2(head_dim // 2)
    total_heads = num_heads + 2 * num_key_value_heads
    num_warps = 4 if total_heads <= 32 else 8

    _triton_decode_packed_qkv_cache_i8_inplace[(n_rows,)](
        packed_qkv_states,
        packed_qkv_states.stride(0),
        query_output,
        query_output.stride(0),
        k_cache_i8,
        k_cache_i8.stride(0),
        v_cache_i8,
        v_cache_i8.stride(0),
        k_scale,
        k_scale.stride(0),
        v_scale,
        v_scale.stride(0),
        token_index,
        q_weight,
        k_weight,
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        n_rows,
        q_eps,
        k_eps,
        q_size,
        kv_size,
        num_heads,
        num_key_value_heads,
        head_dim,
        pad_n_q_head,
        pad_n_k_head,
        pad_half_head_dim,
        num_warps=num_warps,
    )
    return query_output


def prefill_packed_qkv_cache_k_bf16_i8_v_bf16_inplace(
    packed_qkv_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache_bf16: torch.Tensor,
    k_cache_i8: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    token_index: int | torch.Tensor,
    query_output: torch.Tensor,
    *,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    q_eps: float = 1e-6,
    k_eps: float = 1e-6,
) -> torch.Tensor:
    if not (
        TRITON_AVAILABLE
        and not isinstance(token_index, int)
        and packed_qkv_states.is_cuda
        and k_cache_bf16.is_cuda
        and k_cache_i8.is_cuda
        and v_cache.is_cuda
        and k_scale.is_cuda
        and packed_qkv_states.ndim == 2
        and packed_qkv_states.is_contiguous()
        and k_cache_bf16.is_contiguous()
        and k_cache_i8.is_contiguous()
        and v_cache.is_contiguous()
        and k_scale.is_contiguous()
        and k_cache_i8.dtype == torch.int8
        and k_cache_bf16.dtype == packed_qkv_states.dtype
        and v_cache.dtype == packed_qkv_states.dtype
        and k_scale.dtype == torch.float32
    ):
        raise RuntimeError("Sage prefill K-BF16/K-INT8/V-BF16 cache write requires contiguous CUDA tensors and tensor token_index.")

    n_rows = packed_qkv_states.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    pad_n_q_head = triton.next_power_of_2(num_heads)
    pad_n_k_head = triton.next_power_of_2(num_key_value_heads)
    pad_half_head_dim = triton.next_power_of_2(head_dim // 2)
    total_heads = num_heads + 2 * num_key_value_heads
    num_warps = 4 if total_heads <= 32 else 8

    _triton_prefill_packed_qkv_cache_k_bf16_i8_v_bf16_inplace[(n_rows,)](
        packed_qkv_states,
        packed_qkv_states.stride(0),
        query_output,
        query_output.stride(0),
        k_cache_bf16,
        k_cache_bf16.stride(0),
        k_cache_i8,
        k_cache_i8.stride(0),
        v_cache,
        v_cache.stride(0),
        k_scale,
        k_scale.stride(0),
        token_index,
        q_weight,
        k_weight,
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        n_rows,
        q_eps,
        k_eps,
        q_size,
        kv_size,
        num_heads,
        num_key_value_heads,
        head_dim,
        pad_n_q_head,
        pad_n_k_head,
        pad_half_head_dim,
        num_warps=num_warps,
    )
    return query_output


def prewarm_triton_decode_packed_qkv_cache(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    if not TRITON_AVAILABLE:
        return

    total_dim = (num_heads + 2 * num_key_value_heads) * head_dim
    with torch.inference_mode():
        packed_qkv = torch.zeros((1, total_dim), device=device, dtype=dtype)
        query_output = torch.zeros((1, num_heads, head_dim), device=device, dtype=dtype)
        weight = torch.ones((head_dim,), device=device, dtype=dtype)
        cos = torch.zeros((1, head_dim), device=device, dtype=dtype)
        sin = torch.zeros_like(cos)
        k_cache = torch.zeros((2, num_key_value_heads, head_dim), device=device, dtype=dtype)
        v_cache = torch.zeros_like(k_cache)
        token_index = torch.zeros((1,), device=device, dtype=torch.int32)
        decode_packed_qkv_cache_inplace(
            packed_qkv,
            weight,
            weight,
            cos,
            sin,
            k_cache,
            v_cache,
            token_index,
            query_output,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        k_cache_i8 = torch.zeros((2, num_key_value_heads, head_dim), device=device, dtype=torch.int8)
        v_cache_i8 = torch.zeros_like(k_cache_i8)
        k_scale = torch.ones((2, num_key_value_heads), device=device, dtype=torch.float32)
        v_scale = torch.ones_like(k_scale)
        decode_packed_qkv_cache_i8_inplace(
            packed_qkv,
            weight,
            weight,
            cos,
            sin,
            k_cache_i8,
            v_cache_i8,
            k_scale,
            v_scale,
            token_index,
            query_output,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        v_cache_bf16 = torch.zeros((2, num_key_value_heads, head_dim), device=device, dtype=dtype)
        decode_packed_qkv_cache_k_i8_v_bf16_inplace(
            packed_qkv,
            weight,
            weight,
            cos,
            sin,
            k_cache_i8,
            v_cache_bf16,
            k_scale,
            token_index,
            query_output,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        torch.cuda.synchronize(device=device)
