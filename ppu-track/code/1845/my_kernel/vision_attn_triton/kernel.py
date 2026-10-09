# Non-causal multi-head attention for Qwen3-VL-2B vision (B=1, H=16, D=64, bf16).
# Online softmax; one program per (head, query_tile).
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def vision_mha_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_oh,
    stride_os,
    stride_od,
    S,
    TM: tl.constexpr,
    TN: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    head = tl.program_id(0)
    m_block = tl.program_id(1)
    m0 = m_block * TM
    offs_m = m0 + tl.arange(0, TM)
    mask_m = offs_m < S
    offs_d = tl.arange(0, D_HEAD)

    q_offs = head * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptr + q_offs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    m_i = tl.full((TM,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((TM,), dtype=tl.float32)
    acc = tl.zeros((TM, D_HEAD), dtype=tl.float32)

    for n0 in range(0, S, TN):
        offs_n = n0 + tl.arange(0, TN)
        mask_n = offs_n < S
        k_offs = head * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptr + k_offs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        v_offs = head * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptr + v_offs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(p.to(tl.float32), v)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_offs = head * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptr + o_offs, acc.to(tl.bfloat16), mask=mask_m[:, None])
