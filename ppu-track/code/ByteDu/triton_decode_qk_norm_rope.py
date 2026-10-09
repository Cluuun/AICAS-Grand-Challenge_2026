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


DEFAULT_MIN_TRITON_DECODE_QK_NORM_ROPE_ROWS = 1


def _get_min_rows() -> int:
    raw_value = os.getenv(
        "JUNKRAT_TRITON_DECODE_QK_NORM_ROPE_MIN_ROWS",
        str(DEFAULT_MIN_TRITON_DECODE_QK_NORM_ROPE_ROWS),
    )
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return DEFAULT_MIN_TRITON_DECODE_QK_NORM_ROPE_ROWS


def rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def decode_qk_norm_rope_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = F.rms_norm(query_states, (query_states.shape[-1],), q_weight, q_eps)
    k_out = F.rms_norm(key_states, (key_states.shape[-1],), k_weight, k_eps)
    cos_row = cos.unsqueeze(1)
    sin_row = sin.unsqueeze(1)
    q_out = (q_out * cos_row) + (rotate_half_reference(q_out) * sin_row)
    k_out = (k_out * cos_row) + (rotate_half_reference(k_out) * sin_row)
    return q_out, k_out


def can_use_triton_decode_qk_norm_rope(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    min_rows: int | None = None,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if min_rows is None:
        min_rows = _get_min_rows()
    if not (query_states.is_cuda and key_states.is_cuda and q_weight.is_cuda and k_weight.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    if not (
        query_states.device == key_states.device == q_weight.device == k_weight.device == cos.device == sin.device
    ):
        return False
    if query_states.dtype != key_states.dtype or query_states.dtype not in (torch.float16, torch.bfloat16):
        return False
    if query_states.ndim != 3 or key_states.ndim != 3 or cos.ndim != 2 or sin.shape != cos.shape:
        return False
    if query_states.shape[0] != key_states.shape[0] or query_states.shape[0] != cos.shape[0]:
        return False
    if query_states.shape[-1] != key_states.shape[-1] or query_states.shape[-1] != cos.shape[-1]:
        return False
    if query_states.shape[-1] % 2 != 0:
        return False
    if query_states.numel() == 0 or key_states.numel() == 0:
        return False
    if not (query_states.is_contiguous() and key_states.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()):
        return False
    if not (
        q_weight.ndim == 1
        and k_weight.ndim == 1
        and q_weight.shape[0] == query_states.shape[-1]
        and k_weight.shape[0] == key_states.shape[-1]
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
    ):
        return False
    return query_states.shape[0] >= min_rows


if TRITON_AVAILABLE:

    @triton.jit
    def _triton_decode_qk_norm_rope_inplace(
        q_ptr,
        q_row_stride,
        k_ptr,
        k_row_stride,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        cos_row_stride,
        sin_ptr,
        sin_row_stride,
        n_rows,
        q_eps,
        k_eps,
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

        q_base = q_ptr + row_idx * q_row_stride
        k_base = k_ptr + row_idx * k_row_stride
        cos_base = cos_ptr + row_idx * cos_row_stride
        sin_base = sin_ptr + row_idx * sin_row_stride

        cos_half = tl.load(cos_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        sin_half = tl.load(sin_base + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_left = tl.load(q_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        q_weight_right = tl.load(q_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_left = tl.load(k_weight_ptr + half_offsets, mask=half_mask, other=0).to(tl.float32)
        k_weight_right = tl.load(k_weight_ptr + (head_dim // 2) + half_offsets, mask=half_mask, other=0).to(tl.float32)

        q_head_offsets = tl.arange(0, pad_n_q_head)[:, None] * head_dim + half_offsets[None, :]
        k_head_offsets = tl.arange(0, pad_n_k_head)[:, None] * head_dim + half_offsets[None, :]
        q_mask = (tl.arange(0, pad_n_q_head)[:, None] < n_q_head) & half_mask[None, :]
        k_mask = (tl.arange(0, pad_n_k_head)[:, None] < n_k_head) & half_mask[None, :]

        q_left = tl.load(q_base + q_head_offsets, mask=q_mask, other=0)
        q_right = tl.load(q_base + q_head_offsets + (head_dim // 2), mask=q_mask, other=0)
        k_left = tl.load(k_base + k_head_offsets, mask=k_mask, other=0)
        k_right = tl.load(k_base + k_head_offsets + (head_dim // 2), mask=k_mask, other=0)

        q_dtype = q_left.dtype
        k_dtype = k_left.dtype

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

        tl.store(q_base + q_head_offsets, q_out_left.to(q_dtype), mask=q_mask)
        tl.store(q_base + q_head_offsets + (head_dim // 2), q_out_right.to(q_dtype), mask=q_mask)
        tl.store(k_base + k_head_offsets, k_out_left.to(k_dtype), mask=k_mask)
        tl.store(k_base + k_head_offsets + (head_dim // 2), k_out_right.to(k_dtype), mask=k_mask)


def decode_qk_norm_rope_inplace(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_eps: float = 1e-6,
    k_eps: float = 1e-6,
    min_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if min_rows is None:
        min_rows = _get_min_rows()
    if not can_use_triton_decode_qk_norm_rope(
        query_states,
        key_states,
        q_weight,
        k_weight,
        cos,
        sin,
        min_rows=min_rows,
    ):
        q_out, k_out = decode_qk_norm_rope_reference(
            query_states,
            key_states,
            q_weight,
            k_weight,
            cos,
            sin,
            q_eps=q_eps,
            k_eps=k_eps,
        )
        query_states.copy_(q_out)
        key_states.copy_(k_out)
        return query_states, key_states

    n_rows, n_q_head, head_dim = query_states.shape
    n_k_head = key_states.shape[1]
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_k_head = triton.next_power_of_2(n_k_head)
    pad_half_head_dim = triton.next_power_of_2(head_dim // 2)
    total_heads = n_q_head + n_k_head
    num_warps = 4 if total_heads <= 16 else 8

    _triton_decode_qk_norm_rope_inplace[(n_rows,)](
        query_states,
        query_states.stride(0),
        key_states,
        key_states.stride(0),
        q_weight,
        k_weight,
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        n_rows,
        q_eps,
        k_eps,
        n_q_head,
        n_k_head,
        head_dim,
        pad_n_q_head,
        pad_n_k_head,
        pad_half_head_dim,
        num_warps=num_warps,
    )
    return query_states, key_states


def prewarm_triton_decode_qk_norm_rope(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    if not TRITON_AVAILABLE:
        return

    with torch.inference_mode():
        q = torch.zeros((1, num_heads, head_dim), device=device, dtype=dtype)
        k = torch.zeros((1, num_key_value_heads, head_dim), device=device, dtype=dtype)
        weight = torch.ones((head_dim,), device=device, dtype=dtype)
        cos = torch.zeros((1, head_dim), device=device, dtype=dtype)
        sin = torch.zeros_like(cos)
        decode_qk_norm_rope_inplace(q, k, weight, weight, cos, sin, min_rows=1)
        torch.cuda.synchronize(device=device)