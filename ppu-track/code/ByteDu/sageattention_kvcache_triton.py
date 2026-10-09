"""SageAttentionWithKVCache Triton kernels for Qwen3-VL text decode.

This module is intentionally local to the project: it does not import the
external sageattention package.  The supported fast path is batch=1, q_len=1,
GQA 16 query heads / 8 KV heads, head_dim=128, matching the benchmark decode
hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_KV = 64


@dataclass
class SageDecodeWorkspace:
    partial_max: torch.Tensor
    partial_sum: torch.Tensor
    partial_out: torch.Tensor
    output: torch.Tensor
    output_4d: torch.Tensor
    max_splits: int


def _next_power_of_2(n: int) -> int:
    n = max(n, 1)
    p = 1
    while p < n:
        p *= 2
    return p


def required_max_splits(max_cache_len: int) -> int:
    return _next_power_of_2(max(1, math.ceil(max_cache_len / BLOCK_KV)))


def allocate_workspace(
    *,
    max_cache_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SageDecodeWorkspace:
    if num_heads != NUM_Q_HEADS or head_dim != HEAD_DIM:
        raise ValueError(f"Sage decode expects Hq={NUM_Q_HEADS}, D={HEAD_DIM}; got Hq={num_heads}, D={head_dim}.")
    ms = required_max_splits(max_cache_len)
    output = torch.empty((num_heads, head_dim), device=device, dtype=dtype)
    return SageDecodeWorkspace(
        partial_max=torch.empty((num_heads, ms), device=device, dtype=torch.float32),
        partial_sum=torch.empty((num_heads, ms), device=device, dtype=torch.float32),
        partial_out=torch.empty((num_heads, ms, head_dim), device=device, dtype=torch.float32),
        output=output,
        output_4d=output.view(1, 1, num_heads, head_dim),
        max_splits=ms,
    )


def ensure_loaded(*, required: bool = True) -> bool:
    return True


@triton.jit
def _quantize_kv_cache_i8_kernel(
    K, V, K_I8, V_I8, KS, VS, token_index_ptr, count: tl.constexpr,
    stride_kv_s: tl.constexpr, stride_kv_h: tl.constexpr,
    stride_scale_s: tl.constexpr, stride_scale_h: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    token = tl.load(token_index_ptr).to(tl.int64) + row
    base = token * stride_kv_s + kv_head * stride_kv_h
    scale_base = token * stride_scale_s + kv_head * stride_scale_h

    k = tl.load(K + base + d).to(tl.float32)
    v = tl.load(V + base + d).to(tl.float32)
    k_abs = tl.max(tl.abs(k))
    v_abs = tl.max(tl.abs(v))
    k_scale = tl.where(k_abs > 0.0, k_abs / 127.0, 1.0)
    v_scale = tl.where(v_abs > 0.0, v_abs / 127.0, 1.0)

    k_qf = tl.maximum(-127.0, tl.minimum(127.0, k / k_scale))
    v_qf = tl.maximum(-127.0, tl.minimum(127.0, v / v_scale))
    k_q = tl.where(k_qf >= 0.0, k_qf + 0.5, k_qf - 0.5).to(tl.int32)
    v_q = tl.where(v_qf >= 0.0, v_qf + 0.5, v_qf - 0.5).to(tl.int32)

    tl.store(K_I8 + base + d, k_q.to(tl.int8))
    tl.store(V_I8 + base + d, v_q.to(tl.int8))
    tl.store(KS + scale_base, k_scale)
    tl.store(VS + scale_base, v_scale)


@triton.jit
def _quantize_k_cache_i8_kernel(
    K, K_I8, KS, token_index_ptr, count: tl.constexpr,
    stride_kv_s: tl.constexpr, stride_kv_h: tl.constexpr,
    stride_scale_s: tl.constexpr, stride_scale_h: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)
    token = tl.load(token_index_ptr).to(tl.int64) + row
    base = token * stride_kv_s + kv_head * stride_kv_h
    scale_base = token * stride_scale_s + kv_head * stride_scale_h

    k = tl.load(K + base + d).to(tl.float32)
    k_abs = tl.max(tl.abs(k))
    k_scale = tl.where(k_abs > 0.0, k_abs / 127.0, 1.0)
    k_qf = tl.maximum(-127.0, tl.minimum(127.0, k / k_scale))
    k_q = tl.where(k_qf >= 0.0, k_qf + 0.5, k_qf - 0.5).to(tl.int32)

    tl.store(K_I8 + base + d, k_q.to(tl.int8))
    tl.store(KS + scale_base, k_scale)


@triton.jit
def _sage_decode_i8_compute(
    Q, K_I8, V_I8, KS, VS, seqlens,
    pm, ps, po, out,
    scale,
    stride_kv_s,
    stride_kv_h,
    stride_scale_s,
    stride_scale_h,
    max_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    kv_head = tl.program_id(0)
    split = tl.program_id(1)

    kv_len = tl.load(seqlens).to(tl.int32)
    n_splits = tl.cdiv(kv_len, BLOCK_KV)
    if kv_len <= 0 or split >= n_splits:
        return

    GQA: tl.constexpr = 2
    qh0 = kv_head * GQA
    qh1 = qh0 + 1
    d = tl.arange(0, HEAD_DIM)

    q0 = tl.load(Q + qh0 * HEAD_DIM + d).to(tl.float32)
    q1 = tl.load(Q + qh1 * HEAD_DIM + d).to(tl.float32)

    base = split * BLOCK_KV
    t = base + tl.arange(0, BLOCK_KV)
    alive = t < kv_len
    mask2 = alive[:, None]

    kbase = K_I8 + kv_head * stride_kv_h
    vbase = V_I8 + kv_head * stride_kv_h
    ks_base = KS + kv_head * stride_scale_h
    vs_base = VS + kv_head * stride_scale_h
    k_scale = tl.load(ks_base + t * stride_scale_s, mask=alive, other=1.0).to(tl.float32)
    v_scale = tl.load(vs_base + t * stride_scale_s, mask=alive, other=1.0).to(tl.float32)
    k = tl.load(kbase + t[:, None] * stride_kv_s + d[None, :], mask=mask2, other=0).to(tl.float32) * k_scale[:, None]

    s0 = tl.sum(k * q0[None, :], axis=1) * scale
    s1 = tl.sum(k * q1[None, :], axis=1) * scale
    s0 = tl.where(alive, s0, float("-inf"))
    s1 = tl.where(alive, s1, float("-inf"))

    m0 = tl.max(s0)
    m1 = tl.max(s1)
    e0 = tl.where(alive, tl.exp(s0 - m0), 0.0)
    e1 = tl.where(alive, tl.exp(s1 - m1), 0.0)
    l0 = tl.sum(e0)
    l1 = tl.sum(e1)

    v = tl.load(vbase + t[:, None] * stride_kv_s + d[None, :], mask=mask2, other=0).to(tl.float32) * v_scale[:, None]
    w0 = tl.sum(e0[:, None] * v, axis=0)
    w1 = tl.sum(e1[:, None] * v, axis=0)

    if n_splits == 1:
        inv0 = 1.0 / tl.maximum(l0, 1e-20)
        inv1 = 1.0 / tl.maximum(l1, 1e-20)
        tl.store(out + qh0 * HEAD_DIM + d, (w0 * inv0).to(Q.dtype.element_ty))
        tl.store(out + qh1 * HEAD_DIM + d, (w1 * inv1).to(Q.dtype.element_ty))
    else:
        tl.store(pm + qh0 * max_splits + split, m0)
        tl.store(ps + qh0 * max_splits + split, l0)
        tl.store(po + (qh0 * max_splits + split) * HEAD_DIM + d, w0)
        tl.store(pm + qh1 * max_splits + split, m1)
        tl.store(ps + qh1 * max_splits + split, l1)
        tl.store(po + (qh1 * max_splits + split) * HEAD_DIM + d, w1)


@triton.jit
def _sage_decode_i8dot_compute(
    Q, K_I8, V_I8, KS, VS, seqlens,
    pm, ps, po, out,
    scale,
    stride_kv_s,
    stride_kv_h,
    stride_scale_s,
    stride_scale_h,
    max_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    kv_head = tl.program_id(0)
    split = tl.program_id(1)

    kv_len = tl.load(seqlens).to(tl.int32)
    n_splits = tl.cdiv(kv_len, BLOCK_KV)
    if kv_len <= 0 or split >= n_splits:
        return

    GQA: tl.constexpr = 2
    qh0 = kv_head * GQA
    qh1 = qh0 + 1
    d = tl.arange(0, HEAD_DIM)

    q0_f = tl.load(Q + qh0 * HEAD_DIM + d).to(tl.float32)
    q1_f = tl.load(Q + qh1 * HEAD_DIM + d).to(tl.float32)
    q0_abs = tl.max(tl.abs(q0_f))
    q1_abs = tl.max(tl.abs(q1_f))
    q0_scale = tl.where(q0_abs > 0.0, q0_abs / 127.0, 1.0)
    q1_scale = tl.where(q1_abs > 0.0, q1_abs / 127.0, 1.0)
    q0_qf = tl.maximum(-127.0, tl.minimum(127.0, q0_f / q0_scale))
    q1_qf = tl.maximum(-127.0, tl.minimum(127.0, q1_f / q1_scale))
    q0_i8 = tl.where(q0_qf >= 0.0, q0_qf + 0.5, q0_qf - 0.5).to(tl.int8)
    q1_i8 = tl.where(q1_qf >= 0.0, q1_qf + 0.5, q1_qf - 0.5).to(tl.int8)

    base = split * BLOCK_KV
    t = base + tl.arange(0, BLOCK_KV)
    alive = t < kv_len
    mask2 = alive[:, None]

    kbase = K_I8 + kv_head * stride_kv_h
    ks_base = KS + kv_head * stride_scale_h
    k_scale = tl.load(ks_base + t * stride_scale_s, mask=alive, other=1.0).to(tl.float32)
    k_i8 = tl.load(kbase + t[:, None] * stride_kv_s + d[None, :], mask=mask2, other=0).to(tl.int8)

    dot_cols = tl.arange(0, 16)
    q_i8 = tl.where(
        dot_cols[None, :] == 0,
        q0_i8[:, None],
        tl.where(dot_cols[None, :] == 1, q1_i8[:, None], 0),
    ).to(tl.int8)
    qk_dot = tl.dot(k_i8, q_i8, out_dtype=tl.int32).to(tl.float32)
    q0_dot = tl.sum(tl.where(dot_cols[None, :] == 0, qk_dot, 0.0), axis=1)
    q1_dot = tl.sum(tl.where(dot_cols[None, :] == 1, qk_dot, 0.0), axis=1)
    s0 = q0_dot * (q0_scale * k_scale * scale)
    s1 = q1_dot * (q1_scale * k_scale * scale)
    s0 = tl.where(alive, s0, float("-inf"))
    s1 = tl.where(alive, s1, float("-inf"))

    m0 = tl.max(s0)
    m1 = tl.max(s1)
    e0 = tl.where(alive, tl.exp(s0 - m0), 0.0)
    e1 = tl.where(alive, tl.exp(s1 - m1), 0.0)
    l0 = tl.sum(e0)
    l1 = tl.sum(e1)

    vbase = V_I8 + kv_head * stride_kv_h
    vs_base = VS + kv_head * stride_scale_h
    v_scale = tl.load(vs_base + t * stride_scale_s, mask=alive, other=1.0).to(tl.float32)
    v = tl.load(vbase + t[:, None] * stride_kv_s + d[None, :], mask=mask2, other=0).to(tl.float32) * v_scale[:, None]
    w0 = tl.sum(e0[:, None] * v, axis=0)
    w1 = tl.sum(e1[:, None] * v, axis=0)

    if n_splits == 1:
        inv0 = 1.0 / tl.maximum(l0, 1e-20)
        inv1 = 1.0 / tl.maximum(l1, 1e-20)
        tl.store(out + qh0 * HEAD_DIM + d, (w0 * inv0).to(Q.dtype.element_ty))
        tl.store(out + qh1 * HEAD_DIM + d, (w1 * inv1).to(Q.dtype.element_ty))
    else:
        tl.store(pm + qh0 * max_splits + split, m0)
        tl.store(ps + qh0 * max_splits + split, l0)
        tl.store(po + (qh0 * max_splits + split) * HEAD_DIM + d, w0)
        tl.store(pm + qh1 * max_splits + split, m1)
        tl.store(ps + qh1 * max_splits + split, l1)
        tl.store(po + (qh1 * max_splits + split) * HEAD_DIM + d, w1)


@triton.jit
def _sage_decode_k_i8_v_bf16_compute(
    Q, K_I8, V, KS, seqlens,
    pm, ps, po, out,
    scale,
    stride_kv_s,
    stride_kv_h,
    stride_v_s,
    stride_v_h,
    stride_scale_s,
    stride_scale_h,
    max_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    kv_head = tl.program_id(0)
    split = tl.program_id(1)

    kv_len = tl.load(seqlens).to(tl.int32)
    n_splits = tl.cdiv(kv_len, BLOCK_KV)
    if kv_len <= 0 or split >= n_splits:
        return

    GQA: tl.constexpr = 2
    qh0 = kv_head * GQA
    qh1 = qh0 + 1
    d = tl.arange(0, HEAD_DIM)

    q0 = tl.load(Q + qh0 * HEAD_DIM + d).to(tl.float32)
    q1 = tl.load(Q + qh1 * HEAD_DIM + d).to(tl.float32)
    base = split * BLOCK_KV
    t = base + tl.arange(0, BLOCK_KV)
    alive = t < kv_len
    mask2 = alive[:, None]

    kbase = K_I8 + kv_head * stride_kv_h
    ks_base = KS + kv_head * stride_scale_h
    k_scale = tl.load(ks_base + t * stride_scale_s, mask=alive, other=1.0).to(tl.float32)
    k = tl.load(kbase + t[:, None] * stride_kv_s + d[None, :], mask=mask2, other=0).to(tl.float32) * k_scale[:, None]
    s0 = tl.sum(k * q0[None, :], axis=1) * scale
    s1 = tl.sum(k * q1[None, :], axis=1) * scale
    s0 = tl.where(alive, s0, float("-inf"))
    s1 = tl.where(alive, s1, float("-inf"))

    m0 = tl.max(s0)
    m1 = tl.max(s1)
    e0 = tl.where(alive, tl.exp(s0 - m0), 0.0)
    e1 = tl.where(alive, tl.exp(s1 - m1), 0.0)
    l0 = tl.sum(e0)
    l1 = tl.sum(e1)

    vbase = V + kv_head * stride_v_h
    v = tl.load(vbase + t[:, None] * stride_v_s + d[None, :], mask=mask2, other=0.0).to(tl.float32)
    w0 = tl.sum(e0[:, None] * v, axis=0)
    w1 = tl.sum(e1[:, None] * v, axis=0)

    if n_splits == 1:
        inv0 = 1.0 / tl.maximum(l0, 1e-20)
        inv1 = 1.0 / tl.maximum(l1, 1e-20)
        tl.store(out + qh0 * HEAD_DIM + d, (w0 * inv0).to(Q.dtype.element_ty))
        tl.store(out + qh1 * HEAD_DIM + d, (w1 * inv1).to(Q.dtype.element_ty))
    else:
        tl.store(pm + qh0 * max_splits + split, m0)
        tl.store(ps + qh0 * max_splits + split, l0)
        tl.store(po + (qh0 * max_splits + split) * HEAD_DIM + d, w0)
        tl.store(pm + qh1 * max_splits + split, m1)
        tl.store(ps + qh1 * max_splits + split, l1)
        tl.store(po + (qh1 * max_splits + split) * HEAD_DIM + d, w1)


@triton.jit
def _sage_decode_reduce(
    seqlens,
    pm, ps, po, out,
    max_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    qh = tl.program_id(0)
    kv_len = tl.load(seqlens).to(tl.int32)
    n_splits = tl.cdiv(kv_len, BLOCK_KV)
    if n_splits <= 1:
        return

    si = tl.arange(0, MAX_SPLITS)
    si_mask = si < n_splits
    pm_all = tl.load(pm + qh * max_splits + si, mask=si_mask, other=float("-inf"))
    ps_all = tl.load(ps + qh * max_splits + si, mask=si_mask, other=0.0)
    gm = tl.max(pm_all)
    rescale = tl.where(si_mask, tl.exp(pm_all - gm), 0.0)
    denom = tl.sum(rescale * ps_all)

    d = tl.arange(0, HEAD_DIM)
    po_tile = tl.load(
        po + (qh * max_splits + si[:, None]) * HEAD_DIM + d[None, :],
        mask=si_mask[:, None], other=0.0,
    )
    acc = tl.sum(rescale[:, None] * po_tile, axis=0) / tl.maximum(denom, 1e-20)
    tl.store(out + qh * HEAD_DIM + d, acc.to(out.dtype.element_ty))


def quantize_kv_cache_i8(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    token_index: torch.Tensor,
    *,
    count: int,
) -> None:
    if count <= 0:
        return
    _quantize_kv_cache_i8_kernel[(count, NUM_KV_HEADS)](
        key_cache,
        value_cache,
        key_cache_i8,
        value_cache_i8,
        key_scale,
        value_scale,
        token_index,
        count,
        key_cache.stride(0),
        key_cache.stride(1),
        key_scale.stride(0),
        key_scale.stride(1),
        HEAD_DIM=HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )


def quantize_k_cache_i8(
    key_cache: torch.Tensor,
    key_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    token_index: torch.Tensor,
    *,
    count: int,
) -> None:
    if count <= 0:
        return
    _quantize_k_cache_i8_kernel[(count, NUM_KV_HEADS)](
        key_cache,
        key_cache_i8,
        key_scale,
        token_index,
        count,
        key_cache.stride(0),
        key_cache.stride(1),
        key_scale.stride(0),
        key_scale.stride(1),
        HEAD_DIM=HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )


def decode_attention_i8(
    query: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    workspace: SageDecodeWorkspace,
    softmax_scale: float,
) -> torch.Tensor:
    _sage_decode_i8_compute[(NUM_KV_HEADS, workspace.max_splits)](
        query,
        key_cache_i8,
        value_cache_i8,
        key_scale,
        value_scale,
        cache_seqlens,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_out,
        workspace.output,
        softmax_scale,
        key_cache_i8.stride(0),
        key_cache_i8.stride(1),
        key_scale.stride(0),
        key_scale.stride(1),
        workspace.max_splits,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
        num_warps=2,
        num_stages=1,
    )
    _sage_decode_reduce[(NUM_Q_HEADS,)](
        cache_seqlens,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_out,
        workspace.output,
        workspace.max_splits,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
        MAX_SPLITS=workspace.max_splits,
        num_warps=2,
        num_stages=1,
    )
    return workspace.output


def decode_attention_i8dot(
    query: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    workspace: SageDecodeWorkspace,
    softmax_scale: float,
) -> torch.Tensor:
    _sage_decode_i8dot_compute[(NUM_KV_HEADS, workspace.max_splits)](
        query,
        key_cache_i8,
        value_cache_i8,
        key_scale,
        value_scale,
        cache_seqlens,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_out,
        workspace.output,
        softmax_scale,
        key_cache_i8.stride(0),
        key_cache_i8.stride(1),
        key_scale.stride(0),
        key_scale.stride(1),
        workspace.max_splits,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
        num_warps=4,
        num_stages=3,
    )
    _sage_decode_reduce[(NUM_Q_HEADS,)](
        cache_seqlens,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_out,
        workspace.output,
        workspace.max_splits,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
        MAX_SPLITS=workspace.max_splits,
        num_warps=2,
        num_stages=1,
    )
    return workspace.output


def decode_attention_k_i8_v_bf16(
    query: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    workspace: SageDecodeWorkspace,
    softmax_scale: float,
) -> torch.Tensor:
    _sage_decode_k_i8_v_bf16_compute[(NUM_KV_HEADS, workspace.max_splits)](
        query,
        key_cache_i8,
        value_cache,
        key_scale,
        cache_seqlens,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_out,
        workspace.output,
        softmax_scale,
        key_cache_i8.stride(0),
        key_cache_i8.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        key_scale.stride(0),
        key_scale.stride(1),
        workspace.max_splits,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
        num_warps=2,
        num_stages=1,
    )
    _sage_decode_reduce[(NUM_Q_HEADS,)](
        cache_seqlens,
        workspace.partial_max,
        workspace.partial_sum,
        workspace.partial_out,
        workspace.output,
        workspace.max_splits,
        HEAD_DIM=HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
        MAX_SPLITS=workspace.max_splits,
        num_warps=2,
        num_stages=1,
    )
    return workspace.output


def decode_attention_i8_4d(
    query: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    workspace: SageDecodeWorkspace,
    softmax_scale: float,
) -> torch.Tensor:
    decode_attention_i8(
        query,
        key_cache_i8,
        value_cache_i8,
        key_scale,
        value_scale,
        cache_seqlens,
        workspace,
        softmax_scale,
    )
    return workspace.output_4d


def decode_attention_k_i8_v_bf16_4d(
    query: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    workspace: SageDecodeWorkspace,
    softmax_scale: float,
) -> torch.Tensor:
    decode_attention_k_i8_v_bf16(
        query,
        key_cache_i8,
        value_cache,
        key_scale,
        cache_seqlens,
        workspace,
        softmax_scale,
    )
    return workspace.output_4d


def decode_attention_i8dot_4d(
    query: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    workspace: SageDecodeWorkspace,
    softmax_scale: float,
) -> torch.Tensor:
    decode_attention_i8dot(
        query,
        key_cache_i8,
        value_cache_i8,
        key_scale,
        value_scale,
        cache_seqlens,
        workspace,
        softmax_scale,
    )
    return workspace.output_4d
