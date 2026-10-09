from __future__ import annotations

import math
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


DEFAULT_TRITON_LINEAR_ARGMAX_BLOCK_ROWS = 16
DEFAULT_TRITON_LINEAR_ARGMAX_BLOCK_K = 512


def _get_block_rows() -> int:
    raw_value = os.getenv(
        "JUNKRAT_TRITON_LINEAR_ARGMAX_BLOCK_ROWS",
        str(DEFAULT_TRITON_LINEAR_ARGMAX_BLOCK_ROWS),
    )
    try:
        return max(int(raw_value), 16)
    except ValueError:
        return DEFAULT_TRITON_LINEAR_ARGMAX_BLOCK_ROWS


def _get_block_k() -> int:
    raw_value = os.getenv(
        "JUNKRAT_TRITON_LINEAR_ARGMAX_BLOCK_K",
        str(DEFAULT_TRITON_LINEAR_ARGMAX_BLOCK_K),
    )
    try:
        return max(int(raw_value), 16)
    except ValueError:
        return DEFAULT_TRITON_LINEAR_ARGMAX_BLOCK_K


def get_triton_linear_argmax_num_blocks(num_rows: int, block_rows: int | None = None) -> int:
    if block_rows is None:
        block_rows = _get_block_rows()
    return max(math.ceil(num_rows / block_rows), 1)


def can_use_triton_linear_argmax(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
    *,
    block_rows: int | None = None,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if block_rows is None:
        block_rows = _get_block_rows()
    num_blocks = get_triton_linear_argmax_num_blocks(weight.shape[0], block_rows)
    return (
        hidden_states.is_cuda
        and weight.is_cuda
        and partial_values.is_cuda
        and partial_indices.is_cuda
        and output_token.is_cuda
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] == 1
        and hidden_states.shape[-1] == weight.shape[-1]
        and hidden_states.dtype == weight.dtype
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and hidden_states.is_contiguous()
        and weight.ndim == 2
        and weight.is_contiguous()
        and partial_values.ndim == 1
        and partial_values.shape[0] >= num_blocks
        and partial_values.dtype == torch.float32
        and partial_indices.ndim == 1
        and partial_indices.shape[0] >= num_blocks
        and partial_indices.dtype == torch.int32
        and output_token.numel() == 1
        and output_token.dtype in (torch.int32, torch.int64)
    )


def can_use_triton_linear_argmax_q8(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
    *,
    block_rows: int | None = None,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if block_rows is None:
        block_rows = _get_block_rows()
    num_blocks = get_triton_linear_argmax_num_blocks(packed_weight.shape[0], block_rows)
    flattened_scale = scale.view(-1)
    hidden_size = hidden_states.shape[-1]
    packed_hidden_size = math.ceil(hidden_size / 4)
    return (
        hidden_states.is_cuda
        and packed_weight.is_cuda
        and flattened_scale.is_cuda
        and partial_values.is_cuda
        and partial_indices.is_cuda
        and output_token.is_cuda
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] == 1
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and hidden_states.is_contiguous()
        and packed_weight.ndim == 2
        and packed_weight.shape[1] == packed_hidden_size
        and packed_weight.dtype == torch.int32
        and packed_weight.is_contiguous()
        and flattened_scale.ndim == 1
        and flattened_scale.shape[0] == packed_weight.shape[0]
        and flattened_scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and flattened_scale.is_contiguous()
        and partial_values.ndim == 1
        and partial_values.shape[0] >= num_blocks
        and partial_values.dtype == torch.float32
        and partial_indices.ndim == 1
        and partial_indices.shape[0] >= num_blocks
        and partial_indices.dtype == torch.int32
        and output_token.numel() == 1
        and output_token.dtype in (torch.int32, torch.int64)
    )


def can_use_triton_linear_argmax_w8a8(
    hidden_states: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    act_i8: torch.Tensor,
    act_scale: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
    *,
    block_rows: int | None = None,
) -> bool:
    if not TRITON_AVAILABLE:
        return False
    if block_rows is None:
        block_rows = _get_block_rows()
    num_blocks = get_triton_linear_argmax_num_blocks(weight_int8.shape[0], block_rows)
    flattened_scale = weight_scale.view(-1)
    hidden_size = hidden_states.shape[-1]
    return (
        hidden_states.is_cuda
        and weight_int8.is_cuda
        and flattened_scale.is_cuda
        and act_i8.is_cuda
        and act_scale.is_cuda
        and partial_values.is_cuda
        and partial_indices.is_cuda
        and output_token.is_cuda
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] == 1
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and hidden_states.is_contiguous()
        and weight_int8.ndim == 2
        and weight_int8.shape[1] == hidden_size
        and weight_int8.dtype == torch.int8
        and weight_int8.is_contiguous()
        and flattened_scale.ndim == 1
        and flattened_scale.shape[0] == weight_int8.shape[0]
        and flattened_scale.is_contiguous()
        and act_i8.ndim == 1
        and act_i8.numel() >= hidden_size
        and act_i8.dtype == torch.int8
        and act_scale.numel() >= 1
        and partial_values.ndim == 1
        and partial_values.shape[0] >= num_blocks
        and partial_values.dtype == torch.float32
        and partial_indices.ndim == 1
        and partial_indices.shape[0] >= num_blocks
        and partial_indices.dtype == torch.int32
        and output_token.numel() == 1
        and output_token.dtype in (torch.int32, torch.int64)
    )


def _refine_mode() -> str:
    return os.getenv("JUNKRAT_Q8_LM_HEAD_REFINE_MODE", "none").strip().lower()


def linear_argmax_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output_token: torch.Tensor,
) -> torch.Tensor:
    token = torch.argmax(F.linear(hidden_states[:, -1, :], weight), dim=-1, keepdim=True)
    output_token.copy_(token.to(dtype=output_token.dtype))
    return output_token.view(1, 1)


if TRITON_AVAILABLE:

    @triton.jit
    def _triton_quantize_hidden_i8(
        hidden_ptr,
        act_i8_ptr,
        act_scale_ptr,
        hidden_size,
        block_k: tl.constexpr,
    ):
        offsets = tl.arange(0, block_k)
        mask = offsets < hidden_size
        x = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        max_abs = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(max_abs / 127.0, 1.0e-8)
        q = x / scale
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        q = tl.where(q >= 0.0, q + 0.5, q - 0.5).to(tl.int8)
        tl.store(act_i8_ptr + offsets, q, mask=mask)
        tl.store(act_scale_ptr, scale)


    @triton.jit
    def _triton_linear_argmax_w8a8_stage1(
        act_i8_ptr,
        act_scale_ptr,
        weight_int8_ptr,
        weight_scale_ptr,
        partial_values_ptr,
        partial_indices_ptr,
        hidden_size,
        vocab_size,
        weight_row_stride,
        block_rows: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row_offsets = pid * block_rows + tl.arange(0, block_rows)
        row_mask = row_offsets < vocab_size
        accum = tl.zeros((block_rows,), dtype=tl.int32)

        for k_start in tl.range(0, hidden_size, block_k):
            k_offsets = k_start + tl.arange(0, block_k)
            k_mask = k_offsets < hidden_size
            act = tl.load(act_i8_ptr + k_offsets, mask=k_mask, other=0).to(tl.int32)
            weights = tl.load(
                weight_int8_ptr + row_offsets[:, None] * weight_row_stride + k_offsets[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.int32)
            accum += tl.sum(weights * act[None, :], axis=1)

        weight_scale = tl.load(weight_scale_ptr + row_offsets, mask=row_mask, other=0).to(tl.float32)
        act_scale = tl.load(act_scale_ptr).to(tl.float32)
        accum = accum.to(tl.float32) * weight_scale * act_scale
        neg_inf = tl.full((block_rows,), -float("inf"), dtype=tl.float32)
        accum = tl.where(row_mask, accum, neg_inf)
        best_value = tl.max(accum, axis=0)
        best_offset = tl.argmax(accum, axis=0)
        best_index = pid * block_rows + best_offset

        tl.store(partial_values_ptr + pid, best_value)
        tl.store(partial_indices_ptr + pid, best_index.to(tl.int32))


    @triton.jit
    def _triton_linear_argmax_stage1(
        hidden_ptr,
        weight_ptr,
        partial_values_ptr,
        partial_indices_ptr,
        hidden_size,
        vocab_size,
        weight_row_stride,
        block_rows: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row_offsets = pid * block_rows + tl.arange(0, block_rows)
        row_mask = row_offsets < vocab_size
        accum = tl.zeros((block_rows,), dtype=tl.float32)

        for k_start in tl.range(0, hidden_size, block_k):
            k_offsets = k_start + tl.arange(0, block_k)
            k_mask = k_offsets < hidden_size
            hidden = tl.load(hidden_ptr + k_offsets, mask=k_mask, other=0).to(tl.float32)
            weights = tl.load(
                weight_ptr + row_offsets[:, None] * weight_row_stride + k_offsets[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            accum += tl.sum(weights * hidden[None, :], axis=1)

        neg_inf = tl.full((block_rows,), -float("inf"), dtype=tl.float32)
        accum = tl.where(row_mask, accum, neg_inf)
        best_value = tl.max(accum, axis=0)
        best_offset = tl.argmax(accum, axis=0)
        best_index = pid * block_rows + best_offset

        tl.store(partial_values_ptr + pid, best_value)
        tl.store(partial_indices_ptr + pid, best_index.to(tl.int32))


    @triton.jit
    def _triton_linear_argmax_stage2(
        partial_values_ptr,
        partial_indices_ptr,
        output_token_ptr,
        n_blocks,
        block_size: tl.constexpr,
    ):
        block_offsets = tl.arange(0, block_size)
        mask = block_offsets < n_blocks
        values = tl.load(partial_values_ptr + block_offsets, mask=mask, other=-float("inf"))
        best_offset = tl.argmax(values, axis=0)
        best_index = tl.load(partial_indices_ptr + best_offset)
        tl.store(output_token_ptr, best_index.to(output_token_ptr.dtype.element_ty))


    @triton.jit
    def _triton_linear_argmax_q8_stage1(
        hidden_ptr,
        packed_weight_ptr,
        scale_ptr,
        partial_values_ptr,
        partial_indices_ptr,
        hidden_size,
        packed_hidden_size,
        vocab_size,
        packed_weight_row_stride,
        block_rows: tl.constexpr,
        block_packed_k: tl.constexpr,
        candidates_per_block: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row_offsets = pid * block_rows + tl.arange(0, block_rows)
        row_mask = row_offsets < vocab_size
        accum = tl.zeros((block_rows,), dtype=tl.float32)
        for packed_k_start in tl.range(0, packed_hidden_size, block_packed_k):
            packed_k_offsets = packed_k_start + tl.arange(0, block_packed_k)
            packed_k_mask = packed_k_offsets < packed_hidden_size
            hidden_base_offsets = packed_k_offsets * 4
            hidden0 = tl.load(hidden_ptr + hidden_base_offsets, mask=hidden_base_offsets < hidden_size, other=0).to(tl.float32)
            hidden1 = tl.load(hidden_ptr + hidden_base_offsets + 1, mask=hidden_base_offsets + 1 < hidden_size, other=0).to(tl.float32)
            hidden2 = tl.load(hidden_ptr + hidden_base_offsets + 2, mask=hidden_base_offsets + 2 < hidden_size, other=0).to(tl.float32)
            hidden3 = tl.load(hidden_ptr + hidden_base_offsets + 3, mask=hidden_base_offsets + 3 < hidden_size, other=0).to(tl.float32)
            packed = tl.load(
                packed_weight_ptr + row_offsets[:, None] * packed_weight_row_stride + packed_k_offsets[None, :],
                mask=row_mask[:, None] & packed_k_mask[None, :],
                other=0,
            ).to(tl.int32)
            w0 = (packed & 0xFF).to(tl.float32) - 128.0
            w1 = ((packed >> 8) & 0xFF).to(tl.float32) - 128.0
            w2 = ((packed >> 16) & 0xFF).to(tl.float32) - 128.0
            w3 = ((packed >> 24) & 0xFF).to(tl.float32) - 128.0
            accum += tl.sum(
                w0 * hidden0[None, :]
                + w1 * hidden1[None, :]
                + w2 * hidden2[None, :]
                + w3 * hidden3[None, :],
                axis=1,
            )

        scales = tl.load(scale_ptr + row_offsets, mask=row_mask, other=0).to(tl.float32)
        accum *= scales
        neg_inf = tl.full((block_rows,), -float("inf"), dtype=tl.float32)
        accum = tl.where(row_mask, accum, neg_inf)
        for candidate_idx in tl.static_range(0, candidates_per_block):
            best_value = tl.max(accum, axis=0)
            best_offset = tl.argmax(accum, axis=0)
            best_index = pid * block_rows + best_offset
            out_offset = pid * candidates_per_block + candidate_idx
            tl.store(partial_values_ptr + out_offset, best_value)
            tl.store(partial_indices_ptr + out_offset, best_index.to(tl.int32))
            accum = tl.where(row_offsets == best_index, neg_inf, accum)


    @triton.jit
    def _triton_linear_argmax_refine_candidates(
        hidden_ptr,
        weight_ptr,
        partial_values_ptr,
        partial_indices_ptr,
        hidden_size,
        weight_row_stride,
        n_candidates,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row = tl.load(partial_indices_ptr + pid).to(tl.int64)
        accum = tl.zeros((), dtype=tl.float32)

        for k_start in tl.range(0, hidden_size, block_k):
            k_offsets = k_start + tl.arange(0, block_k)
            k_mask = k_offsets < hidden_size
            hidden = tl.load(hidden_ptr + k_offsets, mask=k_mask, other=0).to(tl.float32)
            weights = tl.load(
                weight_ptr + row * weight_row_stride + k_offsets,
                mask=k_mask,
                other=0,
            ).to(tl.float32)
            accum += tl.sum(weights * hidden, axis=0)

        tl.store(partial_values_ptr + pid, accum)


    @triton.jit
    def _triton_linear_argmax_q8_refine_candidates(
        hidden_ptr,
        packed_weight_ptr,
        scale_ptr,
        partial_values_ptr,
        partial_indices_ptr,
        hidden_size,
        packed_hidden_size,
        packed_weight_row_stride,
        n_candidates,
        block_packed_k: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row = tl.load(partial_indices_ptr + pid).to(tl.int64)
        accum = tl.zeros((), dtype=tl.float32)

        for packed_k_start in tl.range(0, packed_hidden_size, block_packed_k):
            packed_k_offsets = packed_k_start + tl.arange(0, block_packed_k)
            packed_k_mask = packed_k_offsets < packed_hidden_size
            hidden_base_offsets = packed_k_offsets * 4
            hidden0 = tl.load(hidden_ptr + hidden_base_offsets, mask=hidden_base_offsets < hidden_size, other=0).to(tl.float32)
            hidden1 = tl.load(hidden_ptr + hidden_base_offsets + 1, mask=hidden_base_offsets + 1 < hidden_size, other=0).to(tl.float32)
            hidden2 = tl.load(hidden_ptr + hidden_base_offsets + 2, mask=hidden_base_offsets + 2 < hidden_size, other=0).to(tl.float32)
            hidden3 = tl.load(hidden_ptr + hidden_base_offsets + 3, mask=hidden_base_offsets + 3 < hidden_size, other=0).to(tl.float32)
            packed = tl.load(
                packed_weight_ptr + row * packed_weight_row_stride + packed_k_offsets,
                mask=packed_k_mask,
                other=0,
            ).to(tl.int32)
            w0 = (packed & 0xFF).to(tl.float32) - 128.0
            w1 = ((packed >> 8) & 0xFF).to(tl.float32) - 128.0
            w2 = ((packed >> 16) & 0xFF).to(tl.float32) - 128.0
            w3 = ((packed >> 24) & 0xFF).to(tl.float32) - 128.0
            accum += tl.sum(
                w0 * hidden0
                + w1 * hidden1
                + w2 * hidden2
                + w3 * hidden3,
                axis=0,
            )

        scales = tl.load(scale_ptr + row).to(tl.float32)
        tl.store(partial_values_ptr + pid, accum * scales)


def linear_argmax_inference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
    *,
    block_rows: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    if block_rows is None:
        block_rows = _get_block_rows()
    if block_k is None:
        block_k = _get_block_k()
    if not can_use_triton_linear_argmax(
        hidden_states,
        weight,
        partial_values,
        partial_indices,
        output_token,
        block_rows=block_rows,
    ):
        return linear_argmax_reference(hidden_states, weight, output_token)

    hidden_vector = hidden_states.view(-1)
    num_blocks = get_triton_linear_argmax_num_blocks(weight.shape[0], block_rows)
    stage1_num_warps = 4 if block_rows <= 128 else 8
    _triton_linear_argmax_stage1[(num_blocks,)](
        hidden_vector,
        weight,
        partial_values,
        partial_indices,
        hidden_vector.shape[0],
        weight.shape[0],
        weight.stride(0),
        block_rows=block_rows,
        block_k=block_k,
        num_warps=stage1_num_warps,
    )

    stage2_block_size = triton.next_power_of_2(num_blocks)
    stage2_num_warps = 4 if stage2_block_size <= 256 else 8
    _triton_linear_argmax_stage2[(1,)](
        partial_values,
        partial_indices,
        output_token.view(-1),
        num_blocks,
        block_size=stage2_block_size,
        num_warps=stage2_num_warps,
    )
    return output_token.view(1, 1)


def linear_argmax_q8_inference(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
    *,
    exact_weight: torch.Tensor | None = None,
    refine_mode: str | None = None,
    candidates_per_block: int = 1,
    refine_block_k: int | None = None,
    block_rows: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    if block_rows is None:
        block_rows = _get_block_rows()
    if block_k is None:
        block_k = _get_block_k()
    flattened_scale = scale.view(-1)
    if not can_use_triton_linear_argmax_q8(
        hidden_states,
        packed_weight,
        flattened_scale,
        partial_values,
        partial_indices,
        output_token,
        block_rows=block_rows,
    ):
        raise RuntimeError("Packed q8 Triton linear argmax called with unsupported tensors.")

    hidden_vector = hidden_states.view(-1)
    num_blocks = get_triton_linear_argmax_num_blocks(packed_weight.shape[0], block_rows)
    candidates_per_block = max(int(candidates_per_block), 1)
    total_candidates = num_blocks * candidates_per_block
    if partial_values.shape[0] < total_candidates or partial_indices.shape[0] < total_candidates:
        raise RuntimeError("Packed q8 Triton linear argmax candidate buffers are too small.")
    block_packed_k = max(math.ceil(block_k / 4), 1)
    stage1_num_warps = 4 if block_rows <= 128 else 8
    _triton_linear_argmax_q8_stage1[(num_blocks,)](
        hidden_vector,
        packed_weight,
        flattened_scale,
        partial_values,
        partial_indices,
        hidden_vector.shape[0],
        packed_weight.shape[1],
        packed_weight.shape[0],
        packed_weight.stride(0),
        block_rows=block_rows,
        block_packed_k=block_packed_k,
        candidates_per_block=candidates_per_block,
        num_warps=stage1_num_warps,
    )

    if refine_mode is None:
        refine_mode = _refine_mode()
    refine_mode = refine_mode.strip().lower()
    if refine_block_k is None:
        refine_block_k = 256
    if refine_mode == "bf16":
        if exact_weight is None:
            raise RuntimeError("Q8 lm_head BF16 refinement requires exact_weight.")
        if (
            not exact_weight.is_cuda
            or exact_weight.ndim != 2
            or not exact_weight.is_contiguous()
            or exact_weight.shape[0] != packed_weight.shape[0]
            or exact_weight.shape[1] != hidden_vector.shape[0]
            or exact_weight.dtype != hidden_states.dtype
        ):
            raise RuntimeError("Q8 exact-refine lm_head weight has unsupported layout.")
        _triton_linear_argmax_refine_candidates[(total_candidates,)](
            hidden_vector,
            exact_weight,
            partial_values,
            partial_indices,
            hidden_vector.shape[0],
            exact_weight.stride(0),
            total_candidates,
            block_k=refine_block_k,
            num_warps=4,
        )
    elif refine_mode == "q8":
        refine_packed_k = max(math.ceil(refine_block_k / 4), 1)
        _triton_linear_argmax_q8_refine_candidates[(total_candidates,)](
            hidden_vector,
            packed_weight,
            flattened_scale,
            partial_values,
            partial_indices,
            hidden_vector.shape[0],
            packed_weight.shape[1],
            packed_weight.stride(0),
            total_candidates,
            block_packed_k=refine_packed_k,
            num_warps=4,
        )
    elif refine_mode not in {"none", ""}:
        raise RuntimeError(f"Unsupported Q8 lm_head refinement mode: {refine_mode!r}")

    stage2_block_size = triton.next_power_of_2(total_candidates)
    stage2_num_warps = 4 if stage2_block_size <= 256 else 8
    _triton_linear_argmax_stage2[(1,)](
        partial_values,
        partial_indices,
        output_token.view(-1),
        total_candidates,
        block_size=stage2_block_size,
        num_warps=stage2_num_warps,
    )
    return output_token.view(1, 1)


def linear_argmax_w8a8_inference(
    hidden_states: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    act_i8: torch.Tensor,
    act_scale: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
    *,
    block_rows: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    if block_rows is None:
        block_rows = _get_block_rows()
    if block_k is None:
        block_k = _get_block_k()
    flattened_scale = weight_scale.view(-1)
    if not can_use_triton_linear_argmax_w8a8(
        hidden_states,
        weight_int8,
        flattened_scale,
        act_i8,
        act_scale,
        partial_values,
        partial_indices,
        output_token,
        block_rows=block_rows,
    ):
        raise RuntimeError("W8A8 Triton linear argmax called with unsupported tensors.")

    hidden_vector = hidden_states.view(-1)
    quant_block = triton.next_power_of_2(hidden_vector.shape[0])
    _triton_quantize_hidden_i8[(1,)](
        hidden_vector,
        act_i8,
        act_scale,
        hidden_vector.shape[0],
        block_k=quant_block,
        num_warps=8,
    )
    num_blocks = get_triton_linear_argmax_num_blocks(weight_int8.shape[0], block_rows)
    stage1_num_warps = 4 if block_rows <= 128 else 8
    _triton_linear_argmax_w8a8_stage1[(num_blocks,)](
        act_i8,
        act_scale,
        weight_int8,
        flattened_scale,
        partial_values,
        partial_indices,
        hidden_vector.shape[0],
        weight_int8.shape[0],
        weight_int8.stride(0),
        block_rows=block_rows,
        block_k=block_k,
        num_warps=stage1_num_warps,
    )

    stage2_block_size = triton.next_power_of_2(num_blocks)
    stage2_num_warps = 4 if stage2_block_size <= 256 else 8
    _triton_linear_argmax_stage2[(1,)](
        partial_values,
        partial_indices,
        output_token.view(-1),
        num_blocks,
        block_size=stage2_block_size,
        num_warps=stage2_num_warps,
    )
    return output_token.view(1, 1)


def prewarm_triton_linear_argmax(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
) -> None:
    if not TRITON_AVAILABLE:
        return

    with torch.inference_mode():
        linear_argmax_inference(
            hidden_states,
            weight,
            partial_values,
            partial_indices,
            output_token,
        )
        torch.cuda.synchronize(device=hidden_states.device)


def prewarm_triton_linear_argmax_q8(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
    exact_weight: torch.Tensor | None = None,
    refine_mode: str | None = None,
    candidates_per_block: int = 1,
    refine_block_k: int | None = None,
) -> None:
    if not TRITON_AVAILABLE:
        return

    with torch.inference_mode():
        linear_argmax_q8_inference(
            hidden_states,
            packed_weight,
            scale,
            partial_values,
            partial_indices,
            output_token,
            exact_weight=exact_weight,
            refine_mode=refine_mode,
            candidates_per_block=candidates_per_block,
            refine_block_k=refine_block_k,
        )
        torch.cuda.synchronize(device=hidden_states.device)


def prewarm_triton_linear_argmax_w8a8(
    hidden_states: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    act_i8: torch.Tensor,
    act_scale: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    output_token: torch.Tensor,
) -> None:
    if not TRITON_AVAILABLE:
        return

    with torch.inference_mode():
        linear_argmax_w8a8_inference(
            hidden_states,
            weight_int8,
            weight_scale,
            act_i8,
            act_scale,
            partial_values,
            partial_indices,
            output_token,
        )
        torch.cuda.synchronize(device=hidden_states.device)
