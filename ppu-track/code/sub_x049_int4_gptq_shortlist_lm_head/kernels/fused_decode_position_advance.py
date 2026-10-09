import torch
import triton
import triton.language as tl


@triton.jit
def _fused_decode_position_advance_kernel(
    cache_pos_ptr,
    position_ids_ptr,
    stride_p0,
    stride_p1,
    stride_p2,
    batch_size,
    BLOCK_B: tl.constexpr,
):
    cur = tl.load(cache_pos_ptr)
    tl.store(cache_pos_ptr, cur + 1)

    offs = tl.arange(0, BLOCK_B)
    mask = offs < batch_size

    p0 = position_ids_ptr + 0 * stride_p0 + offs * stride_p1 + 0 * stride_p2
    p1 = position_ids_ptr + 1 * stride_p0 + offs * stride_p1 + 0 * stride_p2
    p2 = position_ids_ptr + 2 * stride_p0 + offs * stride_p1 + 0 * stride_p2

    v0 = tl.load(p0, mask=mask)
    v1 = tl.load(p1, mask=mask)
    v2 = tl.load(p2, mask=mask)

    tl.store(p0, v0 + 1, mask=mask)
    tl.store(p1, v1 + 1, mask=mask)
    tl.store(p2, v2 + 1, mask=mask)


@triton.jit
def _fused_decode_set_start_kernel(
    cache_pos_ptr,
    position_ids_ptr,
    cache_value,
    position_value,
    stride_p0,
    stride_p1,
    stride_p2,
    batch_size,
    BLOCK_B: tl.constexpr,
):
    tl.store(cache_pos_ptr, cache_value)

    offs = tl.arange(0, BLOCK_B)
    mask = offs < batch_size

    p0 = position_ids_ptr + 0 * stride_p0 + offs * stride_p1 + 0 * stride_p2
    p1 = position_ids_ptr + 1 * stride_p0 + offs * stride_p1 + 0 * stride_p2
    p2 = position_ids_ptr + 2 * stride_p0 + offs * stride_p1 + 0 * stride_p2

    tl.store(p0, position_value, mask=mask)
    tl.store(p1, position_value, mask=mask)
    tl.store(p2, position_value, mask=mask)


def _compute_block_b(batch_size: int) -> int:
    block_b = 1
    while block_b < batch_size:
        block_b <<= 1
    return min(block_b, 128)


def fused_decode_position_advance(cache_position: torch.Tensor, position_ids: torch.Tensor):
    batch_size = int(position_ids.shape[1])
    block_b = _compute_block_b(batch_size)
    _fused_decode_position_advance_kernel[(1,)](
        cache_position,
        position_ids,
        position_ids.stride(0),
        position_ids.stride(1),
        position_ids.stride(2),
        batch_size,
        BLOCK_B=block_b,
        num_warps=1,
        num_stages=1,
    )


def fused_decode_set_start(
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
    cache_value: int,
    position_value: int,
):
    batch_size = int(position_ids.shape[1])
    block_b = _compute_block_b(batch_size)
    _fused_decode_set_start_kernel[(1,)](
        cache_position,
        position_ids,
        int(cache_value),
        int(position_value),
        position_ids.stride(0),
        position_ids.stride(1),
        position_ids.stride(2),
        batch_size,
        BLOCK_B=block_b,
        num_warps=1,
        num_stages=1,
    )
