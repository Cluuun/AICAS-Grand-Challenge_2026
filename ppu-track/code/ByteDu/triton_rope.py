from __future__ import annotations

import os

import torch

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


DEFAULT_MIN_TRITON_TEXT_ROPE_ROWS = 1


def _get_min_rows() -> int:
    raw_value = os.getenv("JUNKRAT_TRITON_TEXT_ROPE_MIN_ROWS", str(DEFAULT_MIN_TRITON_TEXT_ROPE_ROWS))
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return DEFAULT_MIN_TRITON_TEXT_ROPE_ROWS


def rotate_half_reference(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def text_rope_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (
        (query_states * cos) + (rotate_half_reference(query_states) * sin),
        (key_states * cos) + (rotate_half_reference(key_states) * sin),
    )


def can_use_triton_text_rope(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    min_rows: int | None = None,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if min_rows is None:
        min_rows = _get_min_rows()
    if not (query_states.is_cuda and key_states.is_cuda and cos.is_cuda and sin.is_cuda):
        return False
    if not (query_states.device == key_states.device == cos.device == sin.device):
        return False
    if query_states.dtype != key_states.dtype or query_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if query_states.ndim != 4 or key_states.ndim != 4 or cos.ndim not in (2, 3) or sin.shape != cos.shape:
        return False
    if query_states.shape[0] != key_states.shape[0] or query_states.shape[1] != key_states.shape[1]:
        return False
    if query_states.shape[-1] != key_states.shape[-1]:
        return False
    if query_states.shape[-1] % 2 != 0:
        return False
    if cos.ndim == 2:
        if cos.shape[0] != query_states.shape[1]:
            return False
    else:
        if cos.shape[0] not in (1, query_states.shape[0]):
            return False
        if cos.shape[1] != query_states.shape[1]:
            return False
    if cos.shape[-1] != query_states.shape[-1]:
        return False
    if query_states.numel() == 0 or key_states.numel() == 0:
        return False
    # The kernel assumes BSHD contiguous layout so head blocks are packed at stride `head_dim`.
    if not (query_states.is_contiguous() and key_states.is_contiguous()):
        return False
    return query_states.shape[0] * query_states.shape[1] >= min_rows


if TRITON_AVAILABLE:

    @triton.jit
    def _triton_text_rope_bshd(
        q_ptr,
        q_row_stride,
        k_ptr,
        k_row_stride,
        cos_ptr,
        cos_batch_stride,
        cos_seq_stride,
        sin_ptr,
        sin_batch_stride,
        sin_seq_stride,
        seq_len,
        batch_size: tl.constexpr,
        cos_batch_size: tl.constexpr,
        n_q_head: tl.constexpr,
        n_k_head: tl.constexpr,
        head_dim: tl.constexpr,
        pad_n_q_head: tl.constexpr,
        pad_n_k_head: tl.constexpr,
        pad_head_dim: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)

        batch_idx = pid // seq_len
        seq_idx = pid % seq_len

        q_ptr = q_ptr + pid * q_row_stride
        k_ptr = k_ptr + pid * k_row_stride

        cos_ptr = cos_ptr + tl.where(
            cos_batch_size == 1,
            seq_idx * cos_seq_stride,
            batch_idx * cos_batch_stride + seq_idx * cos_seq_stride,
        )
        sin_ptr = sin_ptr + tl.where(
            cos_batch_size == 1,
            seq_idx * sin_seq_stride,
            batch_idx * sin_batch_stride + seq_idx * sin_seq_stride,
        )

        half_offsets = tl.arange(0, pad_head_dim // 2)
        half_mask = half_offsets < head_dim // 2
        cos_row = tl.load(cos_ptr + half_offsets, mask=half_mask, other=0)
        sin_row = tl.load(sin_ptr + half_offsets, mask=half_mask, other=0)

        q_offsets = tl.arange(0, pad_n_q_head)[:, None] * head_dim + tl.arange(0, pad_head_dim // 2)[None, :]
        k_offsets = tl.arange(0, pad_n_k_head)[:, None] * head_dim + tl.arange(0, pad_head_dim // 2)[None, :]
        q_mask = (tl.arange(0, pad_n_q_head)[:, None] < n_q_head) & (
            tl.arange(0, pad_head_dim // 2)[None, :] < head_dim // 2
        )
        k_mask = (tl.arange(0, pad_n_k_head)[:, None] < n_k_head) & (
            tl.arange(0, pad_head_dim // 2)[None, :] < head_dim // 2
        )

        q_left = tl.load(q_ptr + q_offsets, mask=q_mask, other=0).to(cos_row.dtype)
        q_right = tl.load(q_ptr + q_offsets + (head_dim // 2), mask=q_mask, other=0).to(cos_row.dtype)
        k_left = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(cos_row.dtype)
        k_right = tl.load(k_ptr + k_offsets + (head_dim // 2), mask=k_mask, other=0).to(cos_row.dtype)

        tl.store(q_ptr + q_offsets, q_left * cos_row - q_right * sin_row, mask=q_mask)
        tl.store(q_ptr + q_offsets + (head_dim // 2), q_right * cos_row + q_left * sin_row, mask=q_mask)
        tl.store(k_ptr + k_offsets, k_left * cos_row - k_right * sin_row, mask=k_mask)
        tl.store(k_ptr + k_offsets + (head_dim // 2), k_right * cos_row + k_left * sin_row, mask=k_mask)


def text_rope_inference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    min_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if min_rows is None:
        min_rows = _get_min_rows()
    if not can_use_triton_text_rope(query_states, key_states, cos, sin, min_rows=min_rows):
        return text_rope_reference(query_states, key_states, cos, sin)

    cos_phys = cos if cos.ndim == 3 else cos.unsqueeze(0)
    sin_phys = sin if sin.ndim == 3 else sin.unsqueeze(0)

    batch_size, seq_len, n_q_head, head_dim = query_states.shape
    n_k_head = key_states.shape[2]
    n_rows = batch_size * seq_len
    pad_head_dim = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_k_head = triton.next_power_of_2(n_k_head)

    _triton_text_rope_bshd[(n_rows,)](
        query_states,
        query_states.stride(1),
        key_states,
        key_states.stride(1),
        cos_phys,
        cos_phys.stride(0),
        cos_phys.stride(1),
        sin_phys,
        sin_phys.stride(0),
        sin_phys.stride(1),
        seq_len,
        batch_size,
        cos_phys.shape[0],
        n_q_head,
        n_k_head,
        head_dim,
        pad_n_q_head,
        pad_n_k_head,
        pad_head_dim,
    )
    return query_states, key_states


def prewarm_triton_text_rope(
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
        q = torch.zeros((1, 1, num_heads, head_dim), device=device, dtype=dtype)
        k = torch.zeros((1, 1, num_key_value_heads, head_dim), device=device, dtype=dtype)
        cos = torch.zeros((1, 1, head_dim), device=device, dtype=dtype)
        sin = torch.zeros_like(cos)
        text_rope_inference(q, k, cos, sin, min_rows=1)
        torch.cuda.synchronize(device=device)
