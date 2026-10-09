from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=3),
    ],
    key=["D"],
)
@triton.jit
def _kv_update_single_token_kernel(
    key_cache_ptr,
    value_cache_ptr,
    key_state_ptr,
    value_state_ptr,
    cache_pos_ptr,
    B,
    H,
    S,
    D,
    kc_sb,
    kc_sh,
    kc_ss,
    kc_sd,
    vc_sb,
    vc_sh,
    vc_ss,
    vc_sd,
    ks_sb,
    ks_sh,
    ks_ss,
    ks_sd,
    vs_sb,
    vs_sh,
    vs_ss,
    vs_sd,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    h = pid_bh % H
    b = pid_bh // H

    pos = tl.load(cache_pos_ptr)
    valid_pos = (pos >= 0) & (pos < S)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    mask = d_mask & valid_pos

    key_dst = key_cache_ptr + b * kc_sb + h * kc_sh + pos * kc_ss + d_offs * kc_sd
    value_dst = value_cache_ptr + b * vc_sb + h * vc_sh + pos * vc_ss + d_offs * vc_sd

    key_src = key_state_ptr + b * ks_sb + h * ks_sh + d_offs * ks_sd
    value_src = value_state_ptr + b * vs_sb + h * vs_sh + d_offs * vs_sd

    key_val = tl.load(key_src, mask=d_mask, other=0.0)
    value_val = tl.load(value_src, mask=d_mask, other=0.0)

    tl.store(key_dst, key_val, mask=mask)
    tl.store(value_dst, value_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=3),
    ],
    key=["D"],
)
@triton.jit
def _kv_update_multi_token_kernel(
    key_cache_ptr,
    value_cache_ptr,
    key_state_ptr,
    value_state_ptr,
    cache_pos_ptr,
    B,
    H,
    Q,
    S,
    D,
    kc_sb,
    kc_sh,
    kc_ss,
    kc_sd,
    vc_sb,
    vc_sh,
    vc_ss,
    vc_sd,
    ks_sb,
    ks_sh,
    ks_ss,
    ks_sd,
    vs_sb,
    vs_sh,
    vs_ss,
    vs_sd,
    BLOCK_D: tl.constexpr,
):
    pid_bhq = tl.program_id(0)
    pid_d = tl.program_id(1)

    q_idx = pid_bhq % Q
    bh = pid_bhq // Q
    h = bh % H
    b = bh // H

    pos = tl.load(cache_pos_ptr + q_idx)
    valid_pos = (pos >= 0) & (pos < S)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    mask = d_mask & valid_pos

    key_dst = key_cache_ptr + b * kc_sb + h * kc_sh + pos * kc_ss + d_offs * kc_sd
    value_dst = value_cache_ptr + b * vc_sb + h * vc_sh + pos * vc_ss + d_offs * vc_sd

    key_src = key_state_ptr + b * ks_sb + h * ks_sh + q_idx * ks_ss + d_offs * ks_sd
    value_src = value_state_ptr + b * vs_sb + h * vs_sh + q_idx * vs_ss + d_offs * vs_sd

    key_val = tl.load(key_src, mask=d_mask, other=0.0)
    value_val = tl.load(value_src, mask=d_mask, other=0.0)

    tl.store(key_dst, key_val, mask=mask)
    tl.store(value_dst, value_val, mask=mask)


@triton.jit
def _kv_update_multi_token_fixed_kernel(
    key_cache_ptr,
    value_cache_ptr,
    key_state_ptr,
    value_state_ptr,
    cache_pos_ptr,
    B,
    H,
    Q,
    S,
    D,
    kc_sb,
    kc_sh,
    kc_ss,
    kc_sd,
    vc_sb,
    vc_sh,
    vc_ss,
    vc_sd,
    ks_sb,
    ks_sh,
    ks_ss,
    ks_sd,
    vs_sb,
    vs_sh,
    vs_ss,
    vs_sd,
    BLOCK_D: tl.constexpr,
):
    pid_bhq = tl.program_id(0)
    pid_d = tl.program_id(1)

    q_idx = pid_bhq % Q
    bh = pid_bhq // Q
    h = bh % H
    b = bh // H

    pos = tl.load(cache_pos_ptr + q_idx)
    valid_pos = (pos >= 0) & (pos < S)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    mask = d_mask & valid_pos

    key_dst = key_cache_ptr + b * kc_sb + h * kc_sh + pos * kc_ss + d_offs * kc_sd
    value_dst = value_cache_ptr + b * vc_sb + h * vc_sh + pos * vc_ss + d_offs * vc_sd

    key_src = key_state_ptr + b * ks_sb + h * ks_sh + q_idx * ks_ss + d_offs * ks_sd
    value_src = value_state_ptr + b * vs_sb + h * vs_sh + q_idx * vs_ss + d_offs * vs_sd

    key_val = tl.load(key_src, mask=d_mask, other=0.0)
    value_val = tl.load(value_src, mask=d_mask, other=0.0)

    tl.store(key_dst, key_val, mask=mask)
    tl.store(value_dst, value_val, mask=mask)


@triton.jit
def _cache_update_multi_token_one_kernel(
    cache_ptr,
    state_ptr,
    cache_pos_ptr,
    B,
    H,
    Q,
    S,
    D,
    c_sb,
    c_sh,
    c_ss,
    c_sd,
    s_sb,
    s_sh,
    s_ss,
    s_sd,
    BLOCK_D: tl.constexpr,
):
    pid_bhq = tl.program_id(0)
    pid_d = tl.program_id(1)

    q_idx = pid_bhq % Q
    bh = pid_bhq // Q
    h = bh % H
    b = bh // H

    pos = tl.load(cache_pos_ptr + q_idx)
    valid_pos = (pos >= 0) & (pos < S)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    mask = d_mask & valid_pos

    dst = cache_ptr + b * c_sb + h * c_sh + pos * c_ss + d_offs * c_sd
    src = state_ptr + b * s_sb + h * s_sh + q_idx * s_ss + d_offs * s_sd
    val = tl.load(src, mask=d_mask, other=0.0)
    tl.store(dst, val, mask=mask)


def can_use_static_cache_single_token_update(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (key_cache, value_cache, key_states, value_states, cache_position)):
        return False
    if not (key_cache.is_cuda and value_cache.is_cuda and key_states.is_cuda and value_states.is_cuda and cache_position.is_cuda):
        return False
    if key_cache.ndim != 4 or value_cache.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        return False
    if key_states.shape[2] != 1 or value_states.shape[2] != 1:
        return False
    if cache_position.numel() != 1:
        return False
    if key_cache.dtype not in (torch.float16, torch.bfloat16):
        return False
    if value_cache.dtype != key_cache.dtype or key_states.dtype != key_cache.dtype or value_states.dtype != key_cache.dtype:
        return False
    if key_cache.shape[0] != key_states.shape[0] or key_cache.shape[1] != key_states.shape[1] or key_cache.shape[3] != key_states.shape[3]:
        return False
    if value_cache.shape[0] != value_states.shape[0] or value_cache.shape[1] != value_states.shape[1] or value_cache.shape[3] != value_states.shape[3]:
        return False
    return True


def can_use_static_cache_multi_token_update(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (key_cache, value_cache, key_states, value_states, cache_position)):
        return False
    if not (key_cache.is_cuda and value_cache.is_cuda and key_states.is_cuda and value_states.is_cuda and cache_position.is_cuda):
        return False
    if key_cache.ndim != 4 or value_cache.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        return False
    if key_states.shape[2] <= 1 or value_states.shape[2] != key_states.shape[2]:
        return False
    if cache_position.numel() != key_states.shape[2]:
        return False
    if cache_position.dtype not in (torch.int64, torch.int32):
        return False
    if key_cache.dtype not in (torch.float16, torch.bfloat16):
        return False
    if value_cache.dtype != key_cache.dtype or key_states.dtype != key_cache.dtype or value_states.dtype != key_cache.dtype:
        return False
    if key_cache.shape[0] != key_states.shape[0] or key_cache.shape[1] != key_states.shape[1] or key_cache.shape[3] != key_states.shape[3]:
        return False
    if value_cache.shape[0] != value_states.shape[0] or value_cache.shape[1] != value_states.shape[1] or value_cache.shape[3] != value_states.shape[3]:
        return False
    return True


def update_static_cache_single_token_triton(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> None:
    B = int(key_states.shape[0])
    H = int(key_states.shape[1])
    D = int(key_states.shape[3])
    S = int(key_cache.shape[2])
    if B <= 0 or H <= 0 or D <= 0 or S <= 0:
        return

    grid = lambda meta: (B * H, triton.cdiv(D, meta["BLOCK_D"]))
    _kv_update_single_token_kernel[grid](
        key_cache,
        value_cache,
        key_states,
        value_states,
        cache_position.reshape(-1),
        B,
        H,
        S,
        D,
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(value_cache.stride(0)),
        int(value_cache.stride(1)),
        int(value_cache.stride(2)),
        int(value_cache.stride(3)),
        int(key_states.stride(0)),
        int(key_states.stride(1)),
        int(key_states.stride(2)),
        int(key_states.stride(3)),
        int(value_states.stride(0)),
        int(value_states.stride(1)),
        int(value_states.stride(2)),
        int(value_states.stride(3)),
    )


def update_static_cache_multi_token_triton(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> None:
    B = int(key_states.shape[0])
    H = int(key_states.shape[1])
    Q = int(key_states.shape[2])
    D = int(key_states.shape[3])
    S = int(key_cache.shape[2])
    if B <= 0 or H <= 0 or Q <= 1 or D <= 0 or S <= 0:
        return

    if D <= 64:
        block_d = 64
    elif D <= 128:
        block_d = 128
    else:
        block_d = 256
    grid = lambda meta: (B * H * Q, triton.cdiv(D, meta["BLOCK_D"]))
    _cache_update_multi_token_one_kernel[grid](
        key_cache,
        key_states,
        cache_position.reshape(-1),
        B,
        H,
        Q,
        S,
        D,
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(key_states.stride(0)),
        int(key_states.stride(1)),
        int(key_states.stride(2)),
        int(key_states.stride(3)),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    _cache_update_multi_token_one_kernel[grid](
        value_cache,
        value_states,
        cache_position.reshape(-1),
        B,
        H,
        Q,
        S,
        D,
        int(value_cache.stride(0)),
        int(value_cache.stride(1)),
        int(value_cache.stride(2)),
        int(value_cache.stride(3)),
        int(value_states.stride(0)),
        int(value_states.stride(1)),
        int(value_states.stride(2)),
        int(value_states.stride(3)),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
