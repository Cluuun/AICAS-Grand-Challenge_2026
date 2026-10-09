from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _eagle3_cache_select_kernel(
    keys_ptr,
    values_ptr,
    node_ptr,
    K_LEN,
    H: tl.constexpr,
    D: tl.constexpr,
    START_POS: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPEND_COUNT: tl.constexpr,
):
    pid_h = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    for pid_n in tl.static_range(0, APPEND_COUNT):
        src_node = tl.load(node_ptr + pid_n).to(tl.int64)
        src_pos = src_node + START_POS
        dst_pos = pid_n + START_POS

        key_vals = tl.load(
            keys_ptr
            + pid_h * stride_kh
            + src_pos * stride_ks
            + offs_d * stride_kd,
            mask=mask_d & (src_pos >= 0) & (src_pos < K_LEN),
            other=0.0,
        )
        val_vals = tl.load(
            values_ptr
            + pid_h * stride_vh
            + src_pos * stride_vs
            + offs_d * stride_vd,
            mask=mask_d & (src_pos >= 0) & (src_pos < K_LEN),
            other=0.0,
        )
        tl.store(
            keys_ptr
            + pid_h * stride_kh
            + dst_pos * stride_ks
            + offs_d * stride_kd,
            key_vals,
            mask=mask_d & (dst_pos >= 0) & (dst_pos < K_LEN),
        )
        tl.store(
            values_ptr
            + pid_h * stride_vh
            + dst_pos * stride_vs
            + offs_d * stride_vd,
            val_vals,
            mask=mask_d & (dst_pos >= 0) & (dst_pos < K_LEN),
        )


def _can_use(keys: torch.Tensor, values: torch.Tensor, node_indices: torch.Tensor, start_pos: int) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (keys, values, node_indices)):
        return False
    if not (keys.is_cuda and values.is_cuda and node_indices.is_cuda):
        return False
    if keys.ndim != 4 or values.ndim != 4:
        return False
    if keys.shape != values.shape:
        return False
    if int(keys.shape[0]) != 1:
        return False
    if keys.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if values.dtype != keys.dtype:
        return False
    if node_indices.dtype != torch.long or node_indices.ndim != 1:
        return False
    append_count = int(node_indices.numel())
    if append_count < 1 or append_count > 8:
        return False
    if int(start_pos) < 0 or int(start_pos) + append_count > int(keys.shape[2]):
        return False
    if int(keys.shape[-1]) < 1 or int(keys.shape[-1]) > 256:
        return False
    return True


def run_eagle3_cache_select(
    keys: torch.Tensor,
    values: torch.Tensor,
    node_indices: torch.Tensor,
    *,
    start_pos: int,
) -> bool:
    if not _can_use(keys, values, node_indices, start_pos):
        return False
    append_count = int(node_indices.numel())
    head_count = int(keys.shape[1])
    head_dim = int(keys.shape[-1])
    block_d = triton.next_power_of_2(head_dim)
    _eagle3_cache_select_kernel[(head_count,)](
        keys,
        values,
        node_indices,
        K_LEN=int(keys.shape[2]),
        H=head_count,
        D=head_dim,
        START_POS=int(start_pos),
        stride_kb=int(keys.stride(0)),
        stride_kh=int(keys.stride(1)),
        stride_ks=int(keys.stride(2)),
        stride_kd=int(keys.stride(3)),
        stride_vb=int(values.stride(0)),
        stride_vh=int(values.stride(1)),
        stride_vs=int(values.stride(2)),
        stride_vd=int(values.stride(3)),
        BLOCK_D=block_d,
        APPEND_COUNT=append_count,
        num_warps=4,
        num_stages=2,
    )
    return True


@triton.jit
def _eagle3_cache_select_compact_kernel(
    keys_ptr,
    values_ptr,
    node_ptr,
    K_LEN,
    D: tl.constexpr,
    START_POS,
    NEW_LEN,
    PREVIOUS_LEN,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vh,
    stride_vs,
    stride_vd,
    BLOCK_N: tl.constexpr,
    BLOCK_CLEAR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPEND_COUNT,
    CLEAR_COUNT,
):
    pid_h = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < APPEND_COUNT
    mask_d = offs_d < D

    src_node = tl.load(node_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)
    src_pos = src_node + START_POS
    dst_pos = offs_n + START_POS
    key_vals = tl.load(
        keys_ptr
        + pid_h * stride_kh
        + src_pos[:, None] * stride_ks
        + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None] & mask_d[None, :] & (src_pos[:, None] >= 0) & (src_pos[:, None] < K_LEN),
        other=0.0,
    )
    val_vals = tl.load(
        values_ptr
        + pid_h * stride_vh
        + src_pos[:, None] * stride_vs
        + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None] & mask_d[None, :] & (src_pos[:, None] >= 0) & (src_pos[:, None] < K_LEN),
        other=0.0,
    )
    tl.store(
        keys_ptr
        + pid_h * stride_kh
        + dst_pos[:, None] * stride_ks
        + offs_d[None, :] * stride_kd,
        key_vals,
        mask=mask_n[:, None] & mask_d[None, :] & (dst_pos[:, None] >= 0) & (dst_pos[:, None] < K_LEN),
    )
    tl.store(
        values_ptr
        + pid_h * stride_vh
        + dst_pos[:, None] * stride_vs
        + offs_d[None, :] * stride_vd,
        val_vals,
        mask=mask_n[:, None] & mask_d[None, :] & (dst_pos[:, None] >= 0) & (dst_pos[:, None] < K_LEN),
    )

    offs_c = tl.arange(0, BLOCK_CLEAR)
    clear_pos = NEW_LEN + offs_c
    clear_mask = (offs_c < CLEAR_COUNT) & (clear_pos < PREVIOUS_LEN) & (clear_pos < K_LEN)
    zeros = tl.zeros((BLOCK_CLEAR, BLOCK_D), dtype=tl.float32)
    tl.store(
        keys_ptr
        + pid_h * stride_kh
        + clear_pos[:, None] * stride_ks
        + offs_d[None, :] * stride_kd,
        zeros,
        mask=clear_mask[:, None] & mask_d[None, :],
    )
    tl.store(
        values_ptr
        + pid_h * stride_vh
        + clear_pos[:, None] * stride_vs
        + offs_d[None, :] * stride_vd,
        zeros,
        mask=clear_mask[:, None] & mask_d[None, :],
    )


def run_eagle3_cache_select_compact(
    keys: torch.Tensor,
    values: torch.Tensor,
    node_indices: torch.Tensor,
    *,
    start_pos: int,
    previous_len: int,
) -> bool:
    append_count = int(node_indices.numel())
    new_len = int(start_pos) + append_count
    previous_len = max(new_len, min(int(previous_len), int(keys.shape[2])))
    clear_count = previous_len - new_len
    if not can_use_eagle3_cache_select_compact(
        keys,
        values,
        node_indices,
        start_pos=start_pos,
        previous_len=previous_len,
    ):
        return False
    head_count = int(keys.shape[1])
    head_dim = int(keys.shape[-1])
    block_d = triton.next_power_of_2(head_dim)
    _eagle3_cache_select_compact_kernel[(head_count,)](
        keys,
        values,
        node_indices,
        K_LEN=int(keys.shape[2]),
        D=head_dim,
        START_POS=int(start_pos),
        NEW_LEN=new_len,
        PREVIOUS_LEN=previous_len,
        stride_kh=int(keys.stride(1)),
        stride_ks=int(keys.stride(2)),
        stride_kd=int(keys.stride(3)),
        stride_vh=int(values.stride(1)),
        stride_vs=int(values.stride(2)),
        stride_vd=int(values.stride(3)),
        BLOCK_N=8,
        BLOCK_CLEAR=16,
        BLOCK_D=block_d,
        APPEND_COUNT=append_count,
        CLEAR_COUNT=clear_count,
        num_warps=4,
        num_stages=2,
    )
    return True


def can_use_eagle3_cache_select_compact(
    keys: torch.Tensor,
    values: torch.Tensor,
    node_indices: torch.Tensor,
    *,
    start_pos: int,
    previous_len: int,
) -> bool:
    if not _can_use(keys, values, node_indices, start_pos):
        return False
    append_count = int(node_indices.numel())
    new_len = int(start_pos) + append_count
    previous_len = max(new_len, min(int(previous_len), int(keys.shape[2])))
    clear_count = previous_len - new_len
    if append_count > 16 or clear_count > 16:
        return False
    return True
