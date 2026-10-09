from __future__ import annotations

import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_configs(), key=["TOTAL"])
@triton.jit
def qk_rotary_prefill_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_qos,
    stride_qoh,
    stride_qod,
    stride_kos,
    stride_koh,
    stride_kod,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    QH: tl.constexpr,
    KH: tl.constexpr,
    D: tl.constexpr,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL

    q_tokens_total: tl.constexpr = QH * D
    per_token: tl.constexpr = (QH + KH) * D
    s = offs // per_token
    rem = offs - s * per_token
    q_path = rem < q_tokens_total
    h = tl.where(q_path, rem // D, (rem - q_tokens_total) // D)
    d = rem - h * D
    d = tl.where(q_path, d, rem - q_tokens_total - h * D)

    half = D // 2
    first_half = d < half
    pair_d = tl.where(first_half, d + half, d - half)
    sign = tl.where(first_half, -1.0, 1.0)

    q_ptrs = q_ptr + s * stride_qs + h * stride_qh + d * stride_qd
    q_pair_ptrs = q_ptr + s * stride_qs + h * stride_qh + pair_d * stride_qd
    k_ptrs = k_ptr + s * stride_ks + h * stride_kh + d * stride_kd
    k_pair_ptrs = k_ptr + s * stride_ks + h * stride_kh + pair_d * stride_kd
    cos_ptrs = cos_ptr + s * stride_cos_s + d * stride_cos_d
    sin_ptrs = sin_ptr + s * stride_sin_s + d * stride_sin_d

    cosv = tl.load(cos_ptrs, mask=mask, other=0.0).to(tl.float32)
    sinv = tl.load(sin_ptrs, mask=mask, other=0.0).to(tl.float32)
    qv = tl.load(q_ptrs, mask=mask & q_path, other=0.0).to(tl.float32)
    q_pair = tl.load(q_pair_ptrs, mask=mask & q_path, other=0.0).to(tl.float32)
    kv = tl.load(k_ptrs, mask=mask & (~q_path), other=0.0).to(tl.float32)
    k_pair = tl.load(k_pair_ptrs, mask=mask & (~q_path), other=0.0).to(tl.float32)

    q_left = (qv * cosv).to(tl.bfloat16).to(tl.float32)
    q_rotated = (sign * q_pair).to(tl.bfloat16).to(tl.float32)
    q_right = (q_rotated * sinv).to(tl.bfloat16).to(tl.float32)
    q_rot = (q_left + q_right).to(tl.bfloat16)

    k_left = (kv * cosv).to(tl.bfloat16).to(tl.float32)
    k_rotated = (sign * k_pair).to(tl.bfloat16).to(tl.float32)
    k_right = (k_rotated * sinv).to(tl.bfloat16).to(tl.float32)
    k_rot = (k_left + k_right).to(tl.bfloat16)

    q_out_ptrs = q_out_ptr + s * stride_qos + h * stride_qoh + d * stride_qod
    k_out_ptrs = k_out_ptr + s * stride_kos + h * stride_koh + d * stride_kod
    tl.store(q_out_ptrs, q_rot, mask=mask & q_path)
    tl.store(k_out_ptrs, k_rot, mask=mask & (~q_path))
