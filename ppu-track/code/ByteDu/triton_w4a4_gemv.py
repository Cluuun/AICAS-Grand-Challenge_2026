from __future__ import annotations

import math
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


DEFAULT_BLOCK_ROWS = 8
DEFAULT_BLOCK_PACKED_K = 256
DEFAULT_GROUP_SIZE = 128


def _group_size() -> int:
    raw = os.getenv("JUNKRAT_W4A4_GROUP_SIZE", str(DEFAULT_GROUP_SIZE))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_GROUP_SIZE
    if value <= 0:
        return DEFAULT_GROUP_SIZE
    return value


def _block_rows() -> int:
    raw = os.getenv("JUNKRAT_W4A4_BLOCK_ROWS", str(DEFAULT_BLOCK_ROWS))
    try:
        return max(int(raw), 1)
    except ValueError:
        return DEFAULT_BLOCK_ROWS


def _block_packed_k() -> int:
    raw = os.getenv("JUNKRAT_W4A4_BLOCK_PACKED_K", str(DEFAULT_BLOCK_PACKED_K))
    try:
        return max(int(raw), 16)
    except ValueError:
        return DEFAULT_BLOCK_PACKED_K


def pack_weight_q4_rowwise(weight: torch.Tensor, group_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric group-wise W4 packing.

    Values are clamped to signed int4 range and stored in int8 containers.
    The runtime arithmetic uses only the W4/A4 integer values. Scales are
    per output row and input group, avoiding the large error from one scale
    spanning 2048/6144 heterogeneous channels.
    """

    if group_size is None:
        group_size = _group_size()
    weight_fp = weight.detach().to(torch.float32)
    in_features = weight_fp.shape[1]
    groups = math.ceil(in_features / group_size)
    padded = groups * group_size
    if padded != in_features:
        weight_fp = torch.nn.functional.pad(weight_fp, (0, padded - in_features))
    grouped = weight_fp.view(weight_fp.shape[0], groups, group_size)
    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    q = torch.round(grouped / scale).clamp_(-8, 7).to(torch.int8).view(weight_fp.shape[0], padded)
    packed = pack_q4_values(q)
    packed = packed.contiguous()
    return packed, scale.to(device=weight.device, dtype=weight.dtype).contiguous()


def pack_q4_values(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError(f"q4 packing expects a 2D tensor, got shape={tuple(q.shape)}.")
    if q.dtype != torch.int8:
        q = q.to(torch.int8)
    if q.shape[1] % 2 != 0:
        q = torch.nn.functional.pad(q, (0, 1))
    q16 = q.to(torch.int16)
    lo = torch.bitwise_and(q16[:, 0::2], 0xF)
    hi = torch.bitwise_left_shift(torch.bitwise_and(q16[:, 1::2], 0xF), 4)
    return torch.bitwise_or(lo, hi).to(torch.uint8).contiguous()


def unpack_q4_values(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    raw = packed.to(torch.int16)
    lo = torch.bitwise_and(raw, 0xF)
    hi = torch.bitwise_and(torch.bitwise_right_shift(raw, 4), 0xF)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    q = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.int16)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    return q[:, :in_features].to(torch.float32)


def dequantize_weight_q4(packed: torch.Tensor, scale: torch.Tensor, in_features: int) -> torch.Tensor:
    if packed.dtype == torch.uint8 or packed.shape[1] == math.ceil(in_features / 2):
        q = unpack_q4_values(packed, in_features)
    else:
        q = packed[:, :in_features].to(torch.float32)
    scale_fp = scale.to(torch.float32)
    if scale_fp.ndim >= 3 and scale_fp.shape[-1] == 1:
        scale_fp = scale_fp.squeeze(-1)
    if scale_fp.ndim == 2 and scale_fp.shape[1] == 1:
        return (q * scale_fp).to(dtype=scale.dtype)
    if scale_fp.ndim == 1:
        return (q * scale_fp.view(-1, 1)).to(dtype=scale.dtype)
    if scale_fp.ndim == 2:
        group_size = math.ceil(in_features / scale_fp.shape[1])
        group_idx = torch.arange(in_features, device=packed.device) // group_size
        return (q * scale_fp[:, group_idx]).to(dtype=scale.dtype)
    if scale.ndim == 2 or scale.shape[-1] == 1:
        return (q * scale.view(-1, 1).to(torch.float32)).to(dtype=scale.dtype)
    group_size = math.ceil(in_features / scale.shape[1])
    group_idx = torch.arange(in_features, device=packed.device) // group_size
    return (q * scale[:, group_idx].to(torch.float32)).to(dtype=scale.dtype)


def can_use_w4a4_gemv(input_vec: torch.Tensor, packed_weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and input_vec.is_cuda
        and packed_weight.is_cuda
        and weight_scale.is_cuda
        and input_vec.ndim == 1
        and packed_weight.ndim == 2
        and packed_weight.dtype in (torch.uint8, torch.int8)
        and weight_scale.numel() >= packed_weight.shape[0]
        and input_vec.dtype in (torch.float16, torch.bfloat16)
        and input_vec.is_contiguous()
        and packed_weight.is_contiguous()
        and weight_scale.is_contiguous()
        and packed_weight.shape[1] in (input_vec.shape[0], math.ceil(input_vec.shape[0] / 2))
    )


if TRITON_AVAILABLE:

    @triton.jit
    def _quantize_activation_q4_group_kernel(
        input_ptr,
        packed_act_ptr,
        act_scale_ptr,
        k_size,
        group_size,
        clip_ratio,
        packed_block: tl.constexpr,
    ):
        group_id = tl.program_id(0).to(tl.int64)
        group_start = group_id * group_size
        offsets = tl.arange(0, packed_block)
        cols = group_start + offsets
        mask = cols < k_size
        x = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        max_abs = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum((max_abs * clip_ratio) / 7.0, 1.0e-8)
        q = tl.floor(tl.minimum(tl.maximum(x / scale, -8.0), 7.0) + 0.5).to(tl.int8)
        tl.store(packed_act_ptr + cols, q, mask=mask)
        tl.store(act_scale_ptr + group_id, scale)

    @triton.jit
    def _w4a4_gemv_kernel(
        packed_act_ptr,
        act_scale_ptr,
        packed_weight_ptr,
        weight_scale_ptr,
        output_ptr,
        n_rows,
        packed_k,
        weight_row_stride,
        num_groups,
        group_size,
        block_rows: tl.constexpr,
        block_packed_k: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row_offsets = pid * block_rows + tl.arange(0, block_rows)
        row_mask = row_offsets < n_rows
        packed_offsets = tl.arange(0, block_packed_k)
        accum = tl.zeros((block_rows,), dtype=tl.float32)

        for packed_start in tl.range(0, packed_k, block_packed_k):
            cols = packed_start + packed_offsets
            col_mask = cols < packed_k
            group_offsets = cols // group_size
            act_raw = tl.load(packed_act_ptr + cols, mask=col_mask, other=0).to(tl.int32)
            act_raw = tl.where(act_raw > 7, act_raw - 256, act_raw)
            act_q = act_raw.to(tl.float32)
            act_scale = tl.load(act_scale_ptr + group_offsets, mask=col_mask, other=0.0).to(tl.float32)
            w_raw = tl.load(
                packed_weight_ptr + row_offsets[:, None] * weight_row_stride + cols[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0,
            ).to(tl.int32)
            w_raw = tl.where(w_raw > 7, w_raw - 256, w_raw)
            w_q = w_raw.to(tl.float32)
            weight_scale = tl.load(
                weight_scale_ptr + row_offsets[:, None] * num_groups + group_offsets[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accum += tl.sum(w_q * act_q[None, :] * weight_scale * act_scale[None, :], axis=1)

        tl.store(output_ptr + row_offsets, accum, mask=row_mask)

    @triton.jit
    def _w4a4_gemv_group_kernel(
        packed_act_ptr,
        act_scale_ptr,
        packed_weight_ptr,
        weight_scale_ptr,
        output_ptr,
        n_rows,
        packed_k,
        weight_row_stride,
        num_groups,
        group_size,
        block_rows: tl.constexpr,
        block_group_bytes: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row_offsets = pid * block_rows + tl.arange(0, block_rows)
        row_mask = row_offsets < n_rows
        group_byte_offsets = tl.arange(0, block_group_bytes)
        accum = tl.zeros((block_rows,), dtype=tl.float32)

        for group_id in tl.range(0, num_groups):
            cols0 = group_id * group_size + group_byte_offsets * 2
            cols1 = cols0 + 1
            mask0 = (group_byte_offsets * 2 < group_size) & (cols0 < packed_k)
            mask1 = (group_byte_offsets * 2 + 1 < group_size) & (cols1 < packed_k)
            act0 = tl.load(packed_act_ptr + cols0, mask=mask0, other=0).to(tl.int32)
            act1 = tl.load(packed_act_ptr + cols1, mask=mask1, other=0).to(tl.int32)
            act0 = tl.where(act0 > 7, act0 - 256, act0).to(tl.float32)
            act1 = tl.where(act1 > 7, act1 - 256, act1).to(tl.float32)
            w_raw = tl.load(
                packed_weight_ptr + row_offsets[:, None] * weight_row_stride + (cols0[None, :] // 2),
                mask=row_mask[:, None] & mask0[None, :],
                other=0,
            ).to(tl.int32)
            w0 = w_raw & 0xF
            w1 = (w_raw >> 4) & 0xF
            w0 = tl.where(w0 >= 8, w0 - 16, w0).to(tl.float32)
            w1 = tl.where(w1 >= 8, w1 - 16, w1).to(tl.float32)
            partial = tl.sum(w0 * act0[None, :] + w1 * act1[None, :], axis=1)
            weight_scale = tl.load(
                weight_scale_ptr + row_offsets * num_groups + group_id,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            act_scale = tl.load(act_scale_ptr + group_id).to(tl.float32)
            accum += partial * weight_scale * act_scale

        tl.store(output_ptr + row_offsets, accum, mask=row_mask)


def w4a4_gemv_inference(
    input_vec: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    packed_activation: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    activation_clip_ratio: float = 1.0,
    group_size: int | None = None,
    block_rows: int | None = None,
    block_packed_k: int | None = None,
) -> torch.Tensor:
    if not can_use_w4a4_gemv(input_vec, packed_weight, weight_scale):
        raise RuntimeError("W4A4 GEMV requires contiguous CUDA fp16/bf16 input and int8 W4 values.")
    if block_rows is None:
        block_rows = _block_rows()
    if block_packed_k is None:
        block_packed_k = _block_packed_k()
    if group_size is None:
        group_size = math.ceil(input_vec.shape[0] / weight_scale.shape[1]) if weight_scale.ndim >= 2 else input_vec.shape[0]

    packed_k = input_vec.shape[0]
    packed_weight_cols = packed_weight.shape[1]
    num_groups = math.ceil(packed_k / group_size)
    if packed_activation is None or packed_activation.numel() < packed_k or packed_activation.device != input_vec.device:
        packed_activation = torch.empty((packed_k,), device=input_vec.device, dtype=torch.int8)
    if activation_scale is None or activation_scale.numel() < num_groups or activation_scale.device != input_vec.device:
        activation_scale = torch.empty((num_groups,), device=input_vec.device, dtype=torch.float32)
    if output is None:
        output = torch.empty((packed_weight.shape[0],), device=input_vec.device, dtype=input_vec.dtype)

    packed_block = triton.next_power_of_2(group_size)
    packed_group_bytes = triton.next_power_of_2(math.ceil(group_size / 2))
    _quantize_activation_q4_group_kernel[(num_groups,)](
        input_vec,
        packed_activation,
        activation_scale,
        input_vec.shape[0],
        group_size,
        float(activation_clip_ratio),
        packed_block=packed_block,
        num_warps=4,
    )
    grid = (triton.cdiv(packed_weight.shape[0], block_rows),)
    _w4a4_gemv_group_kernel[grid](
        packed_activation,
        activation_scale,
        packed_weight,
        weight_scale.view(-1),
        output,
        packed_weight.shape[0],
        packed_k,
        packed_weight.stride(0),
        num_groups,
        group_size,
        block_rows=block_rows,
        block_group_bytes=packed_group_bytes,
        num_warps=4,
    )
    return output


def w4a4_gemv_reference(input_vec: torch.Tensor, packed_weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    scale_fp = weight_scale
    if scale_fp.ndim >= 3 and scale_fp.shape[-1] == 1:
        scale_fp = scale_fp.squeeze(-1)
    if scale_fp.ndim == 2:
        group_size = math.ceil(input_vec.shape[0] / scale_fp.shape[1])
        groups = scale_fp.shape[1]
        padded = groups * group_size
        x = input_vec.float()
        if padded != x.numel():
            x = torch.nn.functional.pad(x, (0, padded - x.numel()))
        grouped = x.view(groups, group_size)
        act_scale = torch.clamp(grouped.abs().amax(dim=1, keepdim=True) / 7.0, min=1.0e-8)
        act_q = torch.floor(torch.clamp(grouped / act_scale, -8, 7) + 0.5).view(-1)[: input_vec.shape[0]]
        act_deq = act_q * act_scale.view(-1).repeat_interleave(group_size)[: input_vec.shape[0]]
    else:
        act_scale = torch.clamp(input_vec.float().abs().amax() / 7.0, min=1.0e-8)
        act_q = torch.floor(torch.clamp(input_vec.float() / act_scale, -8, 7) + 0.5)
        act_deq = act_q * act_scale
    weight = dequantize_weight_q4(packed_weight, weight_scale, input_vec.shape[0]).float()
    return (weight @ act_deq).to(dtype=input_vec.dtype)
