from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_AICAS_CAUSAL_VALID_KV_CACHE_KEY = None
_AICAS_CAUSAL_VALID_KV_CACHE_VALUE = None


class _AICASNoClearList(list):
    """Reuse white-box runner argument storage across token steps."""

    def clear(self):
        return None


def _aicas_extract_causal_valid_kv_len(
    attention_mask: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
) -> Optional[int]:
    """
    Detect a standard causal or decode-prefix mask with optional right-padding columns.

    When detected, KV can be compacted to the valid prefix and the explicit
    attention mask can be dropped, allowing lower-overhead causal attention
    backends to engage during prefill and decode.
    """
    global _AICAS_CAUSAL_VALID_KV_CACHE_KEY
    global _AICAS_CAUSAL_VALID_KV_CACHE_VALUE
    if attention_mask is None or (not torch.is_tensor(attention_mask)):
        return None
    if attention_mask.ndim != 4:
        return None
    if int(attention_mask.shape[0]) != 1 or int(attention_mask.shape[1]) != 1:
        return None
    if int(attention_mask.shape[-2]) != int(q_len) or int(attention_mask.shape[-1]) != int(kv_len):
        return None
    if q_len <= 1 or kv_len <= 0:
        return None
    cache_key = (
        int(id(attention_mask)),
        int(q_len),
        int(kv_len),
        tuple(int(dim) for dim in attention_mask.shape),
    )
    if _AICAS_CAUSAL_VALID_KV_CACHE_KEY == cache_key:
        return _AICAS_CAUSAL_VALID_KV_CACHE_VALUE
    try:
        mask_2d = attention_mask[0, 0]
        if mask_2d.dtype == torch.bool:
            keep = mask_2d
        elif mask_2d.dtype.is_floating_point:
            keep = mask_2d > -0.5
        else:
            keep = mask_2d != 0

        row_sums = keep.sum(dim=-1, dtype=torch.int32)
        valid_kv_len = int(row_sums[-1].item())
        if valid_kv_len <= 0 or valid_kv_len > kv_len:
            return None
        if q_len == 1:
            if not bool(keep[0, :valid_kv_len].all().item()):
                return None
        else:
            expected = torch.arange(1, q_len + 1, device=row_sums.device, dtype=row_sums.dtype)
            if not bool((row_sums == expected).all().item()):
                return None
        if valid_kv_len < kv_len and bool(keep[:, valid_kv_len:].any().item()):
            return None
        _AICAS_CAUSAL_VALID_KV_CACHE_KEY = cache_key
        _AICAS_CAUSAL_VALID_KV_CACHE_VALUE = valid_kv_len
        return valid_kv_len
    except Exception:
        return None


def _aicas_vision_patch_merger_fastpath(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    *,
    eps: float = 1e-6,
    use_postshuffle_norm: bool = False,
) -> Optional[torch.Tensor]:
    if (
        (not isinstance(x, torch.Tensor))
        or (not isinstance(norm_weight, torch.Tensor))
        or (not isinstance(fc1_weight, torch.Tensor))
        or (not isinstance(fc2_weight, torch.Tensor))
        or x.ndim != 2
        or fc1_weight.ndim != 2
        or fc2_weight.ndim != 2
        or int(fc1_weight.shape[0]) != int(fc1_weight.shape[1])
        or int(fc2_weight.shape[1]) != int(fc1_weight.shape[0])
    ):
        return None
    hidden_size = int(fc1_weight.shape[1])
    if hidden_size <= 0 or int(x.numel()) % hidden_size != 0:
        return None
    if use_postshuffle_norm:
        if int(norm_weight.numel()) != hidden_size:
            return None
        normed = F.layer_norm(x.view(-1, hidden_size), (hidden_size,), norm_weight, norm_bias, eps)
        merged = normed.view(-1, hidden_size)
    else:
        base_hidden = int(norm_weight.numel())
        if base_hidden <= 0 or int(x.shape[1]) != base_hidden:
            return None
        if hidden_size % base_hidden != 0:
            return None
        normed = F.layer_norm(x, (base_hidden,), norm_weight, norm_bias, eps)
        merged = normed.view(-1, hidden_size)
    hidden = F.linear(merged, fc1_weight, fc1_bias)
    hidden = F.gelu(hidden, approximate="none")
    return F.linear(hidden, fc2_weight, fc2_bias)


def _aicas_triton_vision_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(q, torch.Tensor))
        or (not isinstance(k, torch.Tensor))
        or (not isinstance(cos, torch.Tensor))
        or (not isinstance(sin, torch.Tensor))
        or (not q.is_cuda)
        or (not k.is_cuda)
        or (not cos.is_cuda)
        or (not sin.is_cuda)
        or q.ndim != 3
        or k.ndim != 3
        or cos.ndim != 2
        or sin.ndim != 2
        or tuple(int(v) for v in q.shape) != tuple(int(v) for v in k.shape)
        or int(q.shape[-1]) % 2 != 0
        or int(cos.shape[0]) != int(q.shape[0])
        or int(sin.shape[0]) != int(q.shape[0])
        or int(cos.shape[1]) != int(q.shape[-1])
        or int(sin.shape[1]) != int(q.shape[-1])
        or q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or cos.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or sin.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return None
    s = int(q.shape[0])
    h = int(q.shape[1])
    d = int(q.shape[2])
    total = s * h * d
    if total <= 0:
        return None
    q_out = torch.empty((s, h, d), device=q.device, dtype=q.dtype)
    k_out = torch.empty((s, h, d), device=k.device, dtype=k.dtype)
    block = 256
    _aicas_vision_rope_qk_kernel[(triton.cdiv(total, block),)](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        total,
        s,
        h,
        d,
        int(q.stride(0)),
        int(q.stride(1)),
        int(q.stride(2)),
        int(k.stride(0)),
        int(k.stride(1)),
        int(k.stride(2)),
        int(cos.stride(0)),
        int(cos.stride(1)),
        int(sin.stride(0)),
        int(sin.stride(1)),
        BLOCK=block,
    )
    return q_out, k_out


if _TRITON_AVAILABLE:
    @triton.jit
    def _aicas_vision_rope_qk_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        TOTAL: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        Q_STRIDE_S: tl.constexpr,
        Q_STRIDE_H: tl.constexpr,
        Q_STRIDE_D: tl.constexpr,
        K_STRIDE_S: tl.constexpr,
        K_STRIDE_H: tl.constexpr,
        K_STRIDE_D: tl.constexpr,
        COS_STRIDE_S: tl.constexpr,
        COS_STRIDE_D: tl.constexpr,
        SIN_STRIDE_S: tl.constexpr,
        SIN_STRIDE_D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < TOTAL
        d = offs % D
        h = (offs // D) % H
        s = offs // (H * D)
        half = D // 2
        pair_d = tl.where(d < half, d + half, d - half)
        sign = tl.where(d < half, -1.0, 1.0)

        q_base = q_ptr + s * Q_STRIDE_S + h * Q_STRIDE_H
        k_base = k_ptr + s * K_STRIDE_S + h * K_STRIDE_H
        c = tl.load(cos_ptr + s * COS_STRIDE_S + d * COS_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
        sn = tl.load(sin_ptr + s * SIN_STRIDE_S + d * SIN_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_base + d * Q_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
        qp = tl.load(q_base + pair_d * Q_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_base + d * K_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
        kp = tl.load(k_base + pair_d * K_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
        tl.store(q_out_ptr + offs, q * c + sign * qp * sn, mask=mask)
        tl.store(k_out_ptr + offs, k * c + sign * kp * sn, mask=mask)

    @triton.jit
    def _aicas_row_w8a16_gateup_kernel(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_k_base
            x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
            q = tl.load(
                qweight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += (q * scales[:, None]) * x[None, :]
        val = tl.sum(acc, axis=1)
        tl.store(out_ptr + offs_n, val, mask=offs_n < N)

    @triton.jit
    def _aicas_row_w8a16_gateup_silu_mul_kernel(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        act_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        HALF_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc_gate = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < K
            x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            q_gate = tl.load(
                qweight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < HALF_N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            q_up = tl.load(
                qweight_ptr + (HALF_N + offs_n[:, None]) * K + offs_k[None, :],
                mask=(offs_n[:, None] < HALF_N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            scales_gate = tl.load(scales_ptr + offs_n, mask=offs_n < HALF_N, other=0.0).to(tl.float32)
            scales_up = tl.load(scales_ptr + HALF_N + offs_n, mask=offs_n < HALF_N, other=0.0).to(tl.float32)
            acc_gate += (q_gate * scales_gate[:, None]) * x[None, :]
            acc_up += (q_up * scales_up[:, None]) * x[None, :]
        gate = tl.sum(acc_gate, axis=1)
        up = tl.sum(acc_up, axis=1)
        act = gate * tl.sigmoid(gate) * up
        tl.store(act_ptr + offs_n, act, mask=offs_n < HALF_N)

    @triton.jit
    def _aicas_row_w8a16_gateup_norm_kernel(
        x_ptr,
        sumsq_ptr,
        norm_weight_ptr,
        qweight_ptr,
        scales_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        rsigma = tl.rsqrt(tl.load(sumsq_ptr + 0).to(tl.float32) / K + EPS)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < K
            x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            nw = tl.load(norm_weight_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            x = (x * rsigma * nw).to(tl.float16).to(tl.float32)
            q = tl.load(
                qweight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += (q * scales[:, None]) * x[None, :]
        val = tl.sum(acc, axis=1)
        tl.store(out_ptr + offs_n, val, mask=offs_n < N)

    @triton.jit
    def _aicas_row_w8a16_down_silu_kernel(
        gate_up_ptr,
        qweight_ptr,
        scales_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < K
            gate = tl.load(gate_up_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            up = tl.load(gate_up_ptr + K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            x = (gate * tl.sigmoid(gate) * up).to(tl.float32)
            q = tl.load(
                qweight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += (q * scales[:, None]) * x[None, :]
        val = tl.sum(acc, axis=1)
        tl.store(out_ptr + offs_n, val, mask=offs_n < N)

    @triton.jit
    def _aicas_fp16_down_silu_kernel(
        gate_up_ptr,
        weight_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < K
            gate = tl.load(gate_up_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            up = tl.load(gate_up_ptr + K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            x = gate * tl.sigmoid(gate) * up
            w = tl.load(
                weight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            acc += w * x[None, :]
        val = tl.sum(acc, axis=1)
        tl.store(out_ptr + offs_n, val, mask=offs_n < N)

    @triton.jit
    def _aicas_down_silu_preact_kernel(
        gate_up_ptr,
        act_ptr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        gate = tl.load(gate_up_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        up = tl.load(gate_up_ptr + K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        act = gate * tl.sigmoid(gate) * up
        tl.store(act_ptr + offs_k, act, mask=mask_k)

    @triton.jit
    def _aicas_fp16_down_from_act_kernel(
        act_ptr,
        weight_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < K
            x = tl.load(act_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            w = tl.load(
                weight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            acc += w * x[None, :]
        val = tl.sum(acc, axis=1)
        tl.store(out_ptr + offs_n, val, mask=offs_n < N)

    @triton.jit
    def _aicas_row_w8a16_down_from_act_kernel(
        act_ptr,
        qweight_ptr,
        scales_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < K
            x = tl.load(act_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            q = tl.load(
                qweight_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float32)
            scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += (q * scales[:, None]) * x[None, :]
        val = tl.sum(acc, axis=1)
        tl.store(out_ptr + offs_n, val, mask=offs_n < N)

    @triton.jit
    def _aicas_row_w8a16_linear2048_kernel(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, 2048, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < 2048
            x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            q = tl.load(
                qweight_ptr + offs_n[:, None] * 2048 + offs_k[None, :],
                mask=(offs_n[:, None] < 2048) & mask_k[None, :],
                other=0,
            ).to(tl.float32)
            scales = tl.load(scales_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
            acc += (q * scales[:, None]) * x[None, :]
        val = tl.sum(acc, axis=1)
        tl.store(out_ptr + offs_n, val, mask=offs_n < 2048)

    @triton.jit
    def _aicas_row_w8a16_linear2048_add_kernel(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        out_ptr,
        HAS_ADD0: tl.constexpr,
        HAS_ADD1: tl.constexpr,
        HAS_ADD2: tl.constexpr,
        STORE_FP16: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, 2048, BLOCK_K):
            offs_k = k0 + offs_k_base
            mask_k = offs_k < 2048
            x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            q = tl.load(
                qweight_ptr + offs_n[:, None] * 2048 + offs_k[None, :],
                mask=(offs_n[:, None] < 2048) & mask_k[None, :],
                other=0,
            ).to(tl.float32)
            scales = tl.load(scales_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
            acc += (q * scales[:, None]) * x[None, :]
        val = tl.sum(acc, axis=1)
        if HAS_ADD0:
            val += tl.load(add0_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
        if HAS_ADD1:
            val += tl.load(add1_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
        if HAS_ADD2:
            val += tl.load(add2_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
        if STORE_FP16:
            tl.store(out_ptr + offs_n, val.to(tl.float16), mask=offs_n < 2048)
        else:
            tl.store(out_ptr + offs_n, val, mask=offs_n < 2048)

    @triton.jit
    def _aicas_fp16_linear2048_dot_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        STORE_FP16: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k0 in range(0, 2048, BLOCK_K):
            offs_k = k0 + offs_k_base
            x = tl.load(x_ptr + offs_k, mask=offs_k < 2048, other=0.0).to(tl.float16)
            x = tl.broadcast_to(tl.reshape(x, (BLOCK_K, 1)), (BLOCK_K, BLOCK_M))
            w = tl.load(
                weight_ptr + offs_n[:, None] * 2048 + offs_k[None, :],
                mask=(offs_n[:, None] < 2048) & (offs_k[None, :] < 2048),
                other=0.0,
            ).to(tl.float16)
            acc += tl.dot(w, x, out_dtype=tl.float32)
        val = tl.sum(tl.where(offs_m[None, :] == 0, acc, 0.0), axis=1)
        if STORE_FP16:
            tl.store(out_ptr + offs_n, val.to(tl.float16), mask=offs_n < 2048)
        else:
            tl.store(out_ptr + offs_n, val, mask=offs_n < 2048)

    @triton.jit
    def _aicas_fp16_linear2048_dot_add_kernel(
        x_ptr,
        weight_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        out_ptr,
        HAS_ADD0: tl.constexpr,
        HAS_ADD1: tl.constexpr,
        HAS_ADD2: tl.constexpr,
        STORE_FP16: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k_base = tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k0 in range(0, 2048, BLOCK_K):
            offs_k = k0 + offs_k_base
            x = tl.load(x_ptr + offs_k, mask=offs_k < 2048, other=0.0).to(tl.float16)
            x = tl.broadcast_to(tl.reshape(x, (BLOCK_K, 1)), (BLOCK_K, BLOCK_M))
            w = tl.load(
                weight_ptr + offs_n[:, None] * 2048 + offs_k[None, :],
                mask=(offs_n[:, None] < 2048) & (offs_k[None, :] < 2048),
                other=0.0,
            ).to(tl.float16)
            acc += tl.dot(w, x, out_dtype=tl.float32)
        val = tl.sum(tl.where(offs_m[None, :] == 0, acc, 0.0), axis=1)
        if HAS_ADD0:
            val += tl.load(add0_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
        if HAS_ADD1:
            val += tl.load(add1_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
        if HAS_ADD2:
            val += tl.load(add2_ptr + offs_n, mask=offs_n < 2048, other=0.0).to(tl.float32)
        if STORE_FP16:
            tl.store(out_ptr + offs_n, val.to(tl.float16), mask=offs_n < 2048)
        else:
            tl.store(out_ptr + offs_n, val, mask=offs_n < 2048)

    @triton.jit
    def _aicas_rmsnorm_from_sumsq_fp16_m1_kernel(
        x_ptr,
        sumsq_ptr,
        norm_weight_ptr,
        out_ptr,
        K: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_k = tl.arange(0, BLOCK_K)
        mask = offs_k < K
        rsigma = tl.rsqrt(tl.load(sumsq_ptr + 0).to(tl.float32) / K + EPS)
        x = tl.load(x_ptr + offs_k, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(norm_weight_ptr + offs_k, mask=mask, other=0.0).to(tl.float32)
        out = (x * rsigma * w).to(tl.float16)
        tl.store(out_ptr + offs_k, out, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "K"],
    )
    @triton.jit
    def _aicas_rmsnorm_linear_m1_kernel(
        x_ptr,
        rms_w_ptr,
        linear_w_ptr,
        bias_ptr,
        out_ptr,
        K,
        N,
        stride_wn,
        stride_wk,
        eps,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        sumsq = tl.zeros((BLOCK_K,), dtype=tl.float32)
        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
            sumsq += x * x
            k += BLOCK_K
        denom = tl.sum(sumsq, axis=0) / K + eps
        rsigma = tl.rsqrt(denom)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
            rw = tl.load(rms_w_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
            x_norm = (x * rsigma) * rw
            w_ptrs = linear_w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * x_norm[None, :], axis=1)
            k += BLOCK_K

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += bias
        tl.store(out_ptr + offs_n, acc.to(tl.float16), mask=offs_n < N)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        ],
        key=["N", "K"],
    )
    @triton.jit
    def _aicas_fp16_decode_m1_block_top1_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        block_max_ptr,
        block_idx_ptr,
        N,
        K,
        stride_wn,
        stride_wk,
        HAS_BIAS: tl.constexpr,
        FP16_COMPARE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0).to(tl.float16)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float16)
            acc += tl.sum(w * x[None, :], axis=1)
            k += BLOCK_K

        logits = acc
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            logits = logits + bias
        if FP16_COMPARE:
            # Match eager greedy decode: lm_head writes fp16 logits before argmax.
            logits = logits.to(tl.float16).to(tl.float32)
        logits = tl.where(offs_n < N, logits, -1.0e30)

        local_max = tl.max(logits, axis=0)
        local_idx = tl.argmax(logits, axis=0)
        global_idx = pid_n * BLOCK_N + local_idx
        valid = global_idx < N
        safe_idx = tl.where(valid, global_idx, 0)
        safe_max = tl.where(valid, local_max, -1.0e30)
        tl.store(block_max_ptr + pid_n, safe_max)
        tl.store(block_idx_ptr + pid_n, safe_idx)

    @triton.jit
    def _aicas_fp16_decode_m1_block_top1_fixed_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        block_max_ptr,
        block_idx_ptr,
        N,
        K,
        stride_wn,
        stride_wk,
        HAS_BIAS: tl.constexpr,
        FP16_COMPARE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0).to(tl.float16)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float16)
            acc += tl.sum(w * x[None, :], axis=1)
            k += BLOCK_K

        logits = acc
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            logits = logits + bias
        if FP16_COMPARE:
            logits = logits.to(tl.float16).to(tl.float32)
        logits = tl.where(offs_n < N, logits, -1.0e30)

        local_max = tl.max(logits, axis=0)
        local_idx = tl.argmax(logits, axis=0)
        global_idx = pid_n * BLOCK_N + local_idx
        valid = global_idx < N
        safe_idx = tl.where(valid, global_idx, 0)
        safe_max = tl.where(valid, local_max, -1.0e30)
        tl.store(block_max_ptr + pid_n, safe_max)
        tl.store(block_idx_ptr + pid_n, safe_idx)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        ],
        key=["N", "K"],
    )
    @triton.jit
    def _aicas_fp16_decode_m1_block_top1_dot_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        block_max_ptr,
        block_idx_ptr,
        N,
        K,
        stride_wn,
        stride_wk,
        HAS_BIAS: tl.constexpr,
        FP16_COMPARE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N, 1), dtype=tl.float32)

        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + offs_k, mask=offs_k < K, other=0).to(tl.float16)
            x = tl.reshape(x, (BLOCK_K, 1))
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float16)
            acc += tl.dot(w, x, out_dtype=tl.float32)
            k += BLOCK_K

        logits = tl.reshape(acc, (BLOCK_N,))
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            logits = logits + bias
        if FP16_COMPARE:
            logits = logits.to(tl.float16).to(tl.float32)
        logits = tl.where(offs_n < N, logits, -1.0e30)

        local_max = tl.max(logits, axis=0)
        local_idx = tl.argmax(logits, axis=0)
        global_idx = pid_n * BLOCK_N + local_idx
        valid = global_idx < N
        safe_idx = tl.where(valid, global_idx, 0)
        safe_max = tl.where(valid, local_max, -1.0e30)
        tl.store(block_max_ptr + pid_n, safe_max)
        tl.store(block_idx_ptr + pid_n, safe_idx)

    @triton.jit
    def _aicas_top1_reduce_1block_i64_kernel(
        block_max_ptr,
        block_idx_ptr,
        out_idx64_ptr,
        num_blocks,
        N,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        vals = tl.load(block_max_ptr + offs, mask=offs < num_blocks, other=-1.0e30).to(tl.float32)
        max_pos = tl.argmax(vals, axis=0)
        max_pos = tl.where(max_pos < num_blocks, max_pos, 0)
        token = tl.load(block_idx_ptr + max_pos).to(tl.int32)
        token = tl.where(token < 0, 0, token)
        token = tl.where(token >= N, N - 1, token)
        tl.store(out_idx64_ptr, token.to(tl.int64))

    @triton.jit
    def _aicas_fp16_decode_small_m_block_top1_fixed_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        block_max_ptr,
        block_idx_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        max_blocks,
        HAS_BIAS: tl.constexpr,
        FP16_COMPARE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=(pid_m < M) & (offs_k < K),
                other=0,
            ).to(tl.float16)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float16)
            acc += tl.sum(w * x[None, :], axis=1)
            k += BLOCK_K

        logits = acc
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            logits = logits + bias
        if FP16_COMPARE:
            logits = logits.to(tl.float16).to(tl.float32)
        logits = tl.where((pid_m < M) & (offs_n < N), logits, -1.0e30)

        local_max = tl.max(logits, axis=0)
        local_idx = tl.argmax(logits, axis=0)
        global_idx = pid_n * BLOCK_N + local_idx
        valid = global_idx < N
        safe_idx = tl.where(valid, global_idx, 0)
        safe_max = tl.where(valid, local_max, -1.0e30)
        out_offset = pid_m * max_blocks + pid_n
        tl.store(block_max_ptr + out_offset, safe_max)
        tl.store(block_idx_ptr + out_offset, safe_idx)

    @triton.jit
    def _aicas_top1_reduce_small_m_i64_kernel(
        block_max_ptr,
        block_idx_ptr,
        out_idx64_ptr,
        M,
        N,
        max_blocks,
        BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        offs = tl.arange(0, BLOCK)
        vals = tl.load(
            block_max_ptr + pid_m * max_blocks + offs,
            mask=(pid_m < M) & (offs < max_blocks),
            other=-1.0e30,
        ).to(tl.float32)
        max_pos = tl.argmax(vals, axis=0)
        max_pos = tl.where(max_pos < max_blocks, max_pos, 0)
        token = tl.load(block_idx_ptr + pid_m * max_blocks + max_pos).to(tl.int32)
        token = tl.where(token < 0, 0, token)
        token = tl.where(token >= N, N - 1, token)
        tl.store(out_idx64_ptr + pid_m, token.to(tl.int64), mask=pid_m < M)

    @triton.jit
    def _aicas_rmsnorm_fp16_decode_m1_block_top1_fixed_kernel(
        rms_w_ptr,
        residual_ptr,
        mlp_out_ptr,
        sumsq_ptr,
        w_ptr,
        bias_ptr,
        block_max_ptr,
        block_idx_ptr,
        N,
        K,
        stride_wn,
        stride_wk,
        eps,
        HAS_BIAS: tl.constexpr,
        FP16_COMPARE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        rsigma = tl.rsqrt(tl.load(sumsq_ptr + 0).to(tl.float32) / K + eps)

        k = 0
        while k < K:
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            residual = tl.load(residual_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            mlp_out = tl.load(mlp_out_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            rms_w = tl.load(rms_w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            x = ((residual + mlp_out) * rsigma).to(tl.float16).to(tl.float32) * rms_w
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w = tl.load(
                w_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.float16)
            acc += tl.sum(w * x[None, :], axis=1)
            k += BLOCK_K

        logits = acc
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            logits = logits + bias
        if FP16_COMPARE:
            logits = logits.to(tl.float16).to(tl.float32)
        logits = tl.where(offs_n < N, logits, -1.0e30)

        local_max = tl.max(logits, axis=0)
        local_idx = tl.argmax(logits, axis=0)
        global_idx = pid_n * BLOCK_N + local_idx
        valid = global_idx < N
        safe_idx = tl.where(valid, global_idx, 0)
        safe_max = tl.where(valid, local_max, -1.0e30)
        tl.store(block_max_ptr + pid_n, safe_max)
        tl.store(block_idx_ptr + pid_n, safe_idx)

    @triton.jit
    def _aicas_decode_softmax_value_gqa_m1_kernel(
        scores_ptr,
        value_cache_ptr,
        cache_position_ptr,
        out_ptr,
        STEP_OFFSET: tl.constexpr,
        KV_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)

        cache_pos = tl.load(cache_position_ptr + 0).to(tl.int32) + STEP_OFFSET
        valid_k = (offs_k < KV_LEN) & (offs_k <= cache_pos)
        score = tl.load(scores_ptr + pid_h * KV_LEN + offs_k, mask=offs_k < KV_LEN, other=-float("inf")).to(tl.float32)
        score = tl.where(valid_k, score, -float("inf"))
        max_score = tl.max(score, axis=0)
        exp_score = tl.exp(score - max_score)
        exp_score = tl.where(valid_k, exp_score, 0.0)
        denom = tl.sum(exp_score, axis=0)
        inv_denom = 1.0 / denom

        kv_h = pid_h // 2
        offs_v_base = tl.arange(0, BLOCK_V)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k0 in range(0, KV_LEN, BLOCK_V):
            offs_v = k0 + offs_v_base
            valid_v = (offs_v < KV_LEN) & (offs_v <= cache_pos)
            score_v = tl.load(scores_ptr + pid_h * KV_LEN + offs_v, mask=offs_v < KV_LEN, other=-float("inf")).to(tl.float32)
            prob_v = tl.exp(score_v - max_score) * inv_denom
            prob_v = tl.where(valid_v, prob_v, 0.0)
            value_offsets = kv_h * KV_LEN * HEAD_DIM + offs_v[:, None] * HEAD_DIM + offs_d[None, :]
            value = tl.load(
                value_cache_ptr + value_offsets,
                mask=valid_v[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(value * prob_v[:, None], axis=0)
        tl.store(out_ptr + pid_h * HEAD_DIM + offs_d, acc, mask=offs_d < HEAD_DIM)

    @triton.jit
    def _aicas_decode_qkv_softmax_value_gqa_m1_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        cache_position_ptr,
        out_ptr,
        STEP_OFFSET: tl.constexpr,
        KV_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        ATTENTION_CAP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        kv_h = pid_h // 2
        offs_d = tl.arange(0, BLOCK_D)
        offs_k_base = tl.arange(0, BLOCK_K)
        d_mask = offs_d < HEAD_DIM
        q = tl.load(query_ptr + pid_h * HEAD_DIM + offs_d, mask=d_mask, other=0.0).to(tl.float16)
        q = tl.reshape(q, (1, BLOCK_D))
        cache_pos = tl.load(cache_position_ptr + 0).to(tl.int32) + STEP_OFFSET
        if ATTENTION_CAP > 0:
            cache_pos = tl.minimum(cache_pos, ATTENTION_CAP - 1)

        max_score = tl.full((1,), -float("inf"), dtype=tl.float32)
        denom = tl.full((1,), 0.0, dtype=tl.float32)
        acc = tl.zeros((1, BLOCK_D), dtype=tl.float32)
        k0 = 0
        while (k0 < KV_LEN) and (k0 <= cache_pos):
            offs_k = k0 + offs_k_base
            valid_k = (offs_k < KV_LEN) & (offs_k <= cache_pos)
            key = tl.load(
                key_cache_ptr + kv_h * KV_LEN * HEAD_DIM + offs_k[None, :] * HEAD_DIM + offs_d[:, None],
                mask=valid_k[None, :] & d_mask[:, None],
                other=0.0,
            ).to(tl.float16)
            score = tl.dot(q, key, out_dtype=tl.float32) * SCALE
            score = tl.where(valid_k[None, :], score, -float("inf"))
            next_max = tl.maximum(max_score, tl.max(score, axis=1))
            scale_old = tl.exp(max_score - next_max)
            prob = tl.exp(score - next_max)
            prob = tl.where(valid_k[None, :], prob, 0.0)
            denom = denom * scale_old + tl.sum(prob, axis=1)
            value = tl.load(
                value_cache_ptr + kv_h * KV_LEN * HEAD_DIM + offs_k[:, None] * HEAD_DIM + offs_d[None, :],
                mask=valid_k[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float16)
            acc = acc * scale_old[:, None] + tl.dot(prob.to(tl.float16), value, out_dtype=tl.float32)
            max_score = next_max
            k0 += BLOCK_K

        acc = acc / denom
        acc_1d = tl.reshape(acc, (BLOCK_D,))
        tl.store(out_ptr + pid_h * HEAD_DIM + offs_d, acc_1d, mask=d_mask)

    @triton.jit
    def _aicas_decode_qkv_softmax_value_gqa_m1_fp32_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        cache_position_ptr,
        out_ptr,
        STEP_OFFSET: tl.constexpr,
        KV_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        ATTENTION_CAP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        kv_h = pid_h // 2
        offs_d = tl.arange(0, BLOCK_D)
        offs_k_base = tl.arange(0, BLOCK_K)
        d_mask = offs_d < HEAD_DIM
        q = tl.load(query_ptr + pid_h * HEAD_DIM + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        cache_pos = tl.load(cache_position_ptr + 0).to(tl.int32) + STEP_OFFSET
        if ATTENTION_CAP > 0:
            cache_pos = tl.minimum(cache_pos, ATTENTION_CAP - 1)

        max_score = tl.full((), -float("inf"), dtype=tl.float32)
        denom = tl.full((), 0.0, dtype=tl.float32)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        k0 = 0
        while (k0 < KV_LEN) and (k0 <= cache_pos):
            offs_k = k0 + offs_k_base
            valid_k = (offs_k < KV_LEN) & (offs_k <= cache_pos)
            key = tl.load(
                key_cache_ptr + kv_h * KV_LEN * HEAD_DIM + offs_k[None, :] * HEAD_DIM + offs_d[:, None],
                mask=valid_k[None, :] & d_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q[:, None] * key, axis=0) * SCALE
            score = tl.where(valid_k, score, -float("inf"))
            next_max = tl.maximum(max_score, tl.max(score, axis=0))
            scale_old = tl.exp(max_score - next_max)
            prob = tl.exp(score - next_max)
            prob = tl.where(valid_k, prob, 0.0)
            denom = denom * scale_old + tl.sum(prob, axis=0)
            value = tl.load(
                value_cache_ptr + kv_h * KV_LEN * HEAD_DIM + offs_k[:, None] * HEAD_DIM + offs_d[None, :],
                mask=valid_k[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = acc * scale_old + tl.sum(value * prob[:, None], axis=0)
            max_score = next_max
            k0 += BLOCK_K

        acc = acc / denom
        tl.store(out_ptr + pid_h * HEAD_DIM + offs_d, acc, mask=d_mask)

    @triton.jit
    def _aicas_decode_qkv_softmax_value_gqa_pair_m1_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        cache_position_ptr,
        out_ptr,
        STEP_OFFSET: tl.constexpr,
        KV_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        ATTENTION_CAP: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_kv = tl.program_id(0)
        pid_h0 = pid_kv * 2
        pid_h1 = pid_h0 + 1
        offs_d = tl.arange(0, BLOCK_D)
        offs_k_base = tl.arange(0, BLOCK_K)
        d_mask = offs_d < HEAD_DIM
        q0 = tl.load(query_ptr + pid_h0 * HEAD_DIM + offs_d, mask=d_mask, other=0.0).to(tl.float16)
        q1 = tl.load(query_ptr + pid_h1 * HEAD_DIM + offs_d, mask=d_mask, other=0.0).to(tl.float16)
        q0 = tl.reshape(q0, (1, BLOCK_D))
        q1 = tl.reshape(q1, (1, BLOCK_D))
        cache_pos = tl.load(cache_position_ptr + 0).to(tl.int32) + STEP_OFFSET
        if ATTENTION_CAP > 0:
            cache_pos = tl.minimum(cache_pos, ATTENTION_CAP - 1)

        max0 = tl.full((1,), -float("inf"), dtype=tl.float32)
        max1 = tl.full((1,), -float("inf"), dtype=tl.float32)
        denom0 = tl.full((1,), 0.0, dtype=tl.float32)
        denom1 = tl.full((1,), 0.0, dtype=tl.float32)
        acc0 = tl.zeros((1, BLOCK_D), dtype=tl.float32)
        acc1 = tl.zeros((1, BLOCK_D), dtype=tl.float32)
        k0 = 0
        while (k0 < KV_LEN) and (k0 <= cache_pos):
            offs_k = k0 + offs_k_base
            valid_k = (offs_k < KV_LEN) & (offs_k <= cache_pos)
            key = tl.load(
                key_cache_ptr + pid_kv * KV_LEN * HEAD_DIM + offs_k[None, :] * HEAD_DIM + offs_d[:, None],
                mask=valid_k[None, :] & d_mask[:, None],
                other=0.0,
            ).to(tl.float16)

            score0 = tl.dot(q0, key, out_dtype=tl.float32) * SCALE
            score1 = tl.dot(q1, key, out_dtype=tl.float32) * SCALE
            score0 = tl.where(valid_k[None, :], score0, -float("inf"))
            score1 = tl.where(valid_k[None, :], score1, -float("inf"))

            next_max0 = tl.maximum(max0, tl.max(score0, axis=1))
            next_max1 = tl.maximum(max1, tl.max(score1, axis=1))
            scale_old0 = tl.exp(max0 - next_max0)
            scale_old1 = tl.exp(max1 - next_max1)
            prob0 = tl.exp(score0 - next_max0)
            prob1 = tl.exp(score1 - next_max1)
            prob0 = tl.where(valid_k[None, :], prob0, 0.0)
            prob1 = tl.where(valid_k[None, :], prob1, 0.0)
            denom0 = denom0 * scale_old0 + tl.sum(prob0, axis=1)
            denom1 = denom1 * scale_old1 + tl.sum(prob1, axis=1)

            value = tl.load(
                value_cache_ptr + pid_kv * KV_LEN * HEAD_DIM + offs_k[:, None] * HEAD_DIM + offs_d[None, :],
                mask=valid_k[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float16)
            acc0 = acc0 * scale_old0[:, None] + tl.dot(prob0.to(tl.float16), value, out_dtype=tl.float32)
            acc1 = acc1 * scale_old1[:, None] + tl.dot(prob1.to(tl.float16), value, out_dtype=tl.float32)
            max0 = next_max0
            max1 = next_max1
            k0 += BLOCK_K

        acc0 = acc0 / denom0
        acc1 = acc1 / denom1
        acc0_1d = tl.reshape(acc0, (BLOCK_D,))
        acc1_1d = tl.reshape(acc1, (BLOCK_D,))
        tl.store(out_ptr + pid_h0 * HEAD_DIM + offs_d, acc0_1d, mask=d_mask)
        tl.store(out_ptr + pid_h1 * HEAD_DIM + offs_d, acc1_1d, mask=d_mask)

    @triton.jit
    def _aicas_decode_value_merge_gqa_m1_kernel(
        probs_ptr,
        value_cache_ptr,
        out_ptr,
        BUCKET_LEN: tl.constexpr,
        SEGMENT_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_v_base = tl.arange(0, BLOCK_V)
        kv_h = pid_h // 2
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for seg in tl.static_range(0, 3):
            for k0 in range(0, SEGMENT_LEN, BLOCK_V):
                offs_v = k0 + offs_v_base
                bucket_pos = offs_v + seg * SEGMENT_LEN
                valid_v = (offs_v < SEGMENT_LEN) & (bucket_pos < BUCKET_LEN)
                prob = tl.load(
                    probs_ptr + pid_h * BUCKET_LEN + bucket_pos,
                    mask=valid_v,
                    other=0.0,
                ).to(tl.float32)
                value_offsets = (
                    kv_h * BUCKET_LEN * HEAD_DIM
                    + seg * SEGMENT_LEN * HEAD_DIM
                    + offs_v[:, None] * HEAD_DIM
                    + offs_d[None, :]
                )
                value = tl.load(
                    value_cache_ptr + value_offsets,
                    mask=valid_v[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(value * prob[:, None], axis=0)
        tl.store(out_ptr + pid_h * HEAD_DIM + offs_d, acc, mask=offs_d < HEAD_DIM)

    @triton.jit
    def _aicas_decode_value_merge_gqa_pair_m1_kernel(
        probs_ptr,
        value_cache_ptr,
        out_ptr,
        BUCKET_LEN: tl.constexpr,
        SEGMENT_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_kv = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_v_base = tl.arange(0, BLOCK_V)
        pid_h0 = pid_kv * 2
        pid_h1 = pid_h0 + 1
        acc0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for seg in tl.static_range(0, 3):
            for k0 in range(0, SEGMENT_LEN, BLOCK_V):
                offs_v = k0 + offs_v_base
                bucket_pos = offs_v + seg * SEGMENT_LEN
                valid_v = (offs_v < SEGMENT_LEN) & (bucket_pos < BUCKET_LEN)
                prob0 = tl.load(
                    probs_ptr + pid_h0 * BUCKET_LEN + bucket_pos,
                    mask=valid_v,
                    other=0.0,
                ).to(tl.float32)
                prob1 = tl.load(
                    probs_ptr + pid_h1 * BUCKET_LEN + bucket_pos,
                    mask=valid_v,
                    other=0.0,
                ).to(tl.float32)
                value_offsets = (
                    pid_kv * BUCKET_LEN * HEAD_DIM
                    + seg * SEGMENT_LEN * HEAD_DIM
                    + offs_v[:, None] * HEAD_DIM
                    + offs_d[None, :]
                )
                value = tl.load(
                    value_cache_ptr + value_offsets,
                    mask=valid_v[:, None] & (offs_d[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                acc0 += tl.sum(value * prob0[:, None], axis=0)
                acc1 += tl.sum(value * prob1[:, None], axis=0)
        tl.store(out_ptr + pid_h0 * HEAD_DIM + offs_d, acc0, mask=offs_d < HEAD_DIM)
        tl.store(out_ptr + pid_h1 * HEAD_DIM + offs_d, acc1, mask=offs_d < HEAD_DIM)

    @triton.jit
    def _aicas_decode_value_merge_atomic_gqa_m1_kernel(
        probs_ptr,
        value_cache_ptr,
        out_ptr,
        KV_LEN: tl.constexpr,
        EFFECTIVE_KV_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CHUNK_V: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_chunk = tl.program_id(1)
        pid_d = tl.program_id(2)
        chunk_start = pid_chunk * CHUNK_V
        offs_v = chunk_start + tl.arange(0, BLOCK_V)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_v = (offs_v < EFFECTIVE_KV_LEN) & (offs_v < (chunk_start + CHUNK_V))
        kv_h = pid_h // 2
        prob = tl.load(
            probs_ptr + pid_h * KV_LEN + offs_v,
            mask=valid_v,
            other=0.0,
        ).to(tl.float32)
        value_offsets = kv_h * KV_LEN * HEAD_DIM + offs_v[:, None] * HEAD_DIM + offs_d[None, :]
        value = tl.load(
            value_cache_ptr + value_offsets,
            mask=valid_v[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(value * prob[:, None], axis=0)
        tl.atomic_add(out_ptr + pid_h * HEAD_DIM + offs_d, acc, sem="relaxed", mask=offs_d < HEAD_DIM)

def _aicas_triton_row_w8a16_gateup(
    hidden: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 64,
) -> bool:
    """Launch a row-major int8 W8A16 gate/up GEMV for the generated token block."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(hidden, torch.Tensor))
        or (not isinstance(qweight, torch.Tensor))
        or (not isinstance(scales, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not hidden.is_cuda)
        or (not qweight.is_cuda)
        or (not scales.is_cuda)
        or (not out.is_cuda)
        or int(hidden.numel()) != 2048
        or qweight.dtype != torch.int8
        or scales.dtype != torch.float16
        or out.dtype not in (torch.float16, torch.float32)
        or qweight.ndim != 2
        or tuple(int(v) for v in qweight.shape) != (12288, 2048)
        or int(scales.numel()) != 12288
        or int(out.numel()) < 12288
        or (not hidden.is_contiguous())
        or (not qweight.is_contiguous())
        or (not scales.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    grid = (triton.cdiv(12288, block_n),)
    _aicas_row_w8a16_gateup_kernel[grid](
        hidden.view(-1),
        qweight,
        scales.view(-1),
        out.view(-1),
        12288,
        2048,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_row_w8a16_gateup_silu_mul(
    hidden: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    act: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 64,
) -> bool:
    """Launch row-major int8 W8A16 gate/up GEMV and materialize SiLU(gate) * up."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(hidden, torch.Tensor))
        or (not isinstance(qweight, torch.Tensor))
        or (not isinstance(scales, torch.Tensor))
        or (not isinstance(act, torch.Tensor))
        or (not hidden.is_cuda)
        or (not qweight.is_cuda)
        or (not scales.is_cuda)
        or (not act.is_cuda)
        or int(hidden.numel()) != 2048
        or qweight.dtype != torch.int8
        or scales.dtype != torch.float16
        or act.dtype != torch.float32
        or qweight.ndim != 2
        or tuple(int(v) for v in qweight.shape) != (12288, 2048)
        or int(scales.numel()) != 12288
        or int(act.numel()) < 6144
        or (not hidden.is_contiguous())
        or (not qweight.is_contiguous())
        or (not scales.is_contiguous())
        or (not act.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    _aicas_row_w8a16_gateup_silu_mul_kernel[(triton.cdiv(6144, block_n),)](
        hidden.view(-1),
        qweight,
        scales.view(-1),
        act.view(-1),
        12288,
        2048,
        6144,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_row_w8a16_gateup_norm(
    hidden: torch.Tensor,
    sum_squares: torch.Tensor,
    norm_weight: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 64,
    eps: float = 1.0e-6,
) -> bool:
    """Launch fused RMSNorm + row-major int8 W8A16 gate/up GEMV."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(hidden, torch.Tensor))
        or (not isinstance(sum_squares, torch.Tensor))
        or (not isinstance(norm_weight, torch.Tensor))
        or (not isinstance(qweight, torch.Tensor))
        or (not isinstance(scales, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not hidden.is_cuda)
        or (not sum_squares.is_cuda)
        or (not norm_weight.is_cuda)
        or (not qweight.is_cuda)
        or (not scales.is_cuda)
        or (not out.is_cuda)
        or int(hidden.numel()) != 2048
        or int(sum_squares.numel()) < 1
        or int(norm_weight.numel()) != 2048
        or qweight.dtype != torch.int8
        or scales.dtype != torch.float16
        or out.dtype != torch.float32
        or qweight.ndim != 2
        or tuple(int(v) for v in qweight.shape) != (12288, 2048)
        or int(scales.numel()) != 12288
        or int(out.numel()) < 12288
        or (not hidden.is_contiguous())
        or (not sum_squares.is_contiguous())
        or (not norm_weight.is_contiguous())
        or (not qweight.is_contiguous())
        or (not scales.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    grid = (triton.cdiv(12288, block_n),)
    _aicas_row_w8a16_gateup_norm_kernel[grid](
        hidden.view(-1),
        sum_squares.view(-1),
        norm_weight.view(-1),
        qweight,
        scales.view(-1),
        out.view(-1),
        12288,
        2048,
        EPS=float(eps),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_row_w8a16_down_silu(
    gate_up: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 128,
) -> bool:
    """Launch row-major int8 W8A16 down_proj GEMV with fused SiLU(gate) * up."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(gate_up, torch.Tensor))
        or (not isinstance(qweight, torch.Tensor))
        or (not isinstance(scales, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not gate_up.is_cuda)
        or (not qweight.is_cuda)
        or (not scales.is_cuda)
        or (not out.is_cuda)
        or gate_up.dtype not in (torch.float16, torch.float32)
        or qweight.dtype != torch.int8
        or scales.dtype != torch.float16
        or out.dtype != torch.float32
        or qweight.ndim != 2
        or tuple(int(v) for v in qweight.shape) != (2048, 6144)
        or int(scales.numel()) != 2048
        or int(gate_up.numel()) < 12288
        or int(out.numel()) < 2048
        or (not gate_up.is_contiguous())
        or (not qweight.is_contiguous())
        or (not scales.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    grid = (triton.cdiv(2048, block_n),)
    _aicas_row_w8a16_down_silu_kernel[grid](
        gate_up.view(-1),
        qweight,
        scales.view(-1),
        out.view(-1),
        2048,
        6144,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_row_w8a16_linear2048(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 128,
) -> bool:
    """Launch row-major int8 W8A16 GEMV for a 2048x2048 projection."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(x, torch.Tensor))
        or (not isinstance(qweight, torch.Tensor))
        or (not isinstance(scales, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not x.is_cuda)
        or (not qweight.is_cuda)
        or (not scales.is_cuda)
        or (not out.is_cuda)
        or x.dtype not in (torch.float16, torch.float32)
        or qweight.dtype != torch.int8
        or scales.dtype != torch.float16
        or out.dtype not in (torch.float16, torch.float32)
        or qweight.ndim != 2
        or tuple(int(v) for v in qweight.shape) != (2048, 2048)
        or int(scales.numel()) != 2048
        or int(x.numel()) < 2048
        or int(out.numel()) < 2048
        or (not x.is_contiguous())
        or (not qweight.is_contiguous())
        or (not scales.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    _aicas_row_w8a16_linear2048_kernel[(triton.cdiv(2048, block_n),)](
        x.view(-1),
        qweight,
        scales.view(-1),
        out.view(-1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_row_w8a16_linear2048_add(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    add0: Optional[torch.Tensor] = None,
    add1: Optional[torch.Tensor] = None,
    add2: Optional[torch.Tensor] = None,
    store_fp16: bool = False,
    block_n: int = 16,
    block_k: int = 128,
) -> bool:
    """Launch row-major int8 W8A16 GEMV plus up to three 2048-wide addends."""
    addends = (add0, add1, add2)
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(x, torch.Tensor))
        or (not isinstance(qweight, torch.Tensor))
        or (not isinstance(scales, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not x.is_cuda)
        or (not qweight.is_cuda)
        or (not scales.is_cuda)
        or (not out.is_cuda)
        or x.dtype not in (torch.float16, torch.float32)
        or qweight.dtype != torch.int8
        or scales.dtype != torch.float16
        or out.dtype not in (torch.float16, torch.float32)
        or qweight.ndim != 2
        or tuple(int(v) for v in qweight.shape) != (2048, 2048)
        or int(scales.numel()) != 2048
        or int(x.numel()) < 2048
        or int(out.numel()) < 2048
        or (not x.is_contiguous())
        or (not qweight.is_contiguous())
        or (not scales.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False
    for add in addends:
        if add is not None and (
            (not isinstance(add, torch.Tensor))
            or (not add.is_cuda)
            or add.dtype not in (torch.float16, torch.float32)
            or int(add.numel()) < 2048
            or (not add.is_contiguous())
        ):
            return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    dummy = out
    _aicas_row_w8a16_linear2048_add_kernel[(triton.cdiv(2048, block_n),)](
        x.view(-1),
        qweight,
        scales.view(-1),
        (add0 if add0 is not None else dummy).view(-1),
        (add1 if add1 is not None else dummy).view(-1),
        (add2 if add2 is not None else dummy).view(-1),
        out.view(-1),
        HAS_ADD0=add0 is not None,
        HAS_ADD1=add1 is not None,
        HAS_ADD2=add2 is not None,
        STORE_FP16=bool(store_fp16),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_fp16_linear2048_dot(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 64,
    block_k: int = 64,
) -> bool:
    """Launch fp16 tensor-core GEMV for a 2048x2048 projection."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(x, torch.Tensor))
        or (not isinstance(weight, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not x.is_cuda)
        or (not weight.is_cuda)
        or (not out.is_cuda)
        or x.dtype not in (torch.float16, torch.float32)
        or weight.dtype != torch.float16
        or out.dtype not in (torch.float16, torch.float32)
        or weight.ndim != 2
        or tuple(int(v) for v in weight.shape) != (2048, 2048)
        or int(x.numel()) < 2048
        or int(out.numel()) < 2048
        or (not x.is_contiguous())
        or (not weight.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False
    block_n = max(16, int(block_n))
    block_k = max(32, int(block_k))
    _aicas_fp16_linear2048_dot_kernel[(triton.cdiv(2048, block_n),)](
        x.view(-1),
        weight,
        out.view(-1),
        STORE_FP16=out.dtype == torch.float16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=16,
    )
    return True


def _aicas_triton_fp16_linear2048_dot_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    add0: Optional[torch.Tensor] = None,
    add1: Optional[torch.Tensor] = None,
    add2: Optional[torch.Tensor] = None,
    block_n: int = 64,
    block_k: int = 64,
) -> bool:
    """Launch fp16 tensor-core GEMV plus up to three 2048-wide addends."""
    addends = (add0, add1, add2)
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(x, torch.Tensor))
        or (not isinstance(weight, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not x.is_cuda)
        or (not weight.is_cuda)
        or (not out.is_cuda)
        or x.dtype not in (torch.float16, torch.float32)
        or weight.dtype != torch.float16
        or out.dtype not in (torch.float16, torch.float32)
        or weight.ndim != 2
        or tuple(int(v) for v in weight.shape) != (2048, 2048)
        or int(x.numel()) < 2048
        or int(out.numel()) < 2048
        or (not x.is_contiguous())
        or (not weight.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False
    for add in addends:
        if add is not None and (
            (not isinstance(add, torch.Tensor))
            or (not add.is_cuda)
            or add.dtype not in (torch.float16, torch.float32)
            or int(add.numel()) < 2048
            or (not add.is_contiguous())
        ):
            return False
    block_n = max(16, int(block_n))
    block_k = max(32, int(block_k))
    dummy = out
    _aicas_fp16_linear2048_dot_add_kernel[(triton.cdiv(2048, block_n),)](
        x.view(-1),
        weight,
        (add0 if add0 is not None else dummy).view(-1),
        (add1 if add1 is not None else dummy).view(-1),
        (add2 if add2 is not None else dummy).view(-1),
        out.view(-1),
        HAS_ADD0=add0 is not None,
        HAS_ADD1=add1 is not None,
        HAS_ADD2=add2 is not None,
        STORE_FP16=out.dtype == torch.float16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=16,
    )
    return True


def _aicas_triton_row_w8a16_down_silu_preact(
    gate_up: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    act: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 128,
    act_block_k: int = 1024,
) -> bool:
    """Launch row-major int8 W8A16 down_proj GEMV after computing SiLU(gate) * up once."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(gate_up, torch.Tensor))
        or (not isinstance(qweight, torch.Tensor))
        or (not isinstance(scales, torch.Tensor))
        or (not isinstance(act, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not gate_up.is_cuda)
        or (not qweight.is_cuda)
        or (not scales.is_cuda)
        or (not act.is_cuda)
        or (not out.is_cuda)
        or gate_up.dtype not in (torch.float16, torch.float32)
        or qweight.dtype != torch.int8
        or scales.dtype != torch.float16
        or act.dtype not in (torch.float16, torch.float32)
        or out.dtype != torch.float32
        or qweight.ndim != 2
        or tuple(int(v) for v in qweight.shape) != (2048, 6144)
        or int(scales.numel()) != 2048
        or int(gate_up.numel()) < 12288
        or int(act.numel()) < 6144
        or int(out.numel()) < 2048
        or (not gate_up.is_contiguous())
        or (not qweight.is_contiguous())
        or (not scales.is_contiguous())
        or (not act.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    act_block_k = max(16, int(act_block_k))
    _aicas_down_silu_preact_kernel[(triton.cdiv(6144, act_block_k),)](
        gate_up.view(-1),
        act.view(-1),
        6144,
        BLOCK_K=act_block_k,
    )
    _aicas_row_w8a16_down_from_act_kernel[(triton.cdiv(2048, block_n),)](
        act.view(-1),
        qweight,
        scales.view(-1),
        out.view(-1),
        2048,
        6144,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_fp16_down_silu(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 8,
    block_k: int = 128,
) -> bool:
    """Launch exact fp16 down_proj GEMV with fused SiLU(gate) * up."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(gate_up, torch.Tensor))
        or (not isinstance(weight, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not gate_up.is_cuda)
        or (not weight.is_cuda)
        or (not out.is_cuda)
        or gate_up.dtype != torch.float32
        or weight.dtype != torch.float16
        or out.dtype != torch.float32
        or weight.ndim != 2
        or tuple(int(v) for v in weight.shape) != (2048, 6144)
        or int(gate_up.numel()) < 12288
        or int(out.numel()) < 2048
        or (not gate_up.is_contiguous())
        or (not weight.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    _aicas_fp16_down_silu_kernel[(triton.cdiv(2048, block_n),)](
        gate_up.view(-1),
        weight,
        out.view(-1),
        2048,
        6144,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_silu_preact(
    gate_up: torch.Tensor,
    act: torch.Tensor,
    *,
    block_k: int = 1024,
) -> bool:
    """Materialize SiLU(gate) * up for the generated token block."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(gate_up, torch.Tensor))
        or (not isinstance(act, torch.Tensor))
        or (not gate_up.is_cuda)
        or (not act.is_cuda)
        or gate_up.dtype not in (torch.float16, torch.float32)
        or act.dtype not in (torch.float16, torch.float32)
        or int(gate_up.numel()) < 12288
        or int(act.numel()) < 6144
        or (not gate_up.is_contiguous())
        or (not act.is_contiguous())
    ):
        return False

    block_k = max(16, int(block_k))
    _aicas_down_silu_preact_kernel[(triton.cdiv(6144, block_k),)](
        gate_up.view(-1),
        act.view(-1),
        6144,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_fp16_down_silu_preact(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    act: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 8,
    block_k: int = 128,
    act_block_k: int = 1024,
) -> bool:
    """Launch exact fp16 down_proj GEMV after precomputing SiLU(gate) * up once."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(gate_up, torch.Tensor))
        or (not isinstance(weight, torch.Tensor))
        or (not isinstance(act, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not gate_up.is_cuda)
        or (not weight.is_cuda)
        or (not act.is_cuda)
        or (not out.is_cuda)
        or gate_up.dtype != torch.float32
        or weight.dtype != torch.float16
        or act.dtype != torch.float32
        or out.dtype != torch.float32
        or weight.ndim != 2
        or tuple(int(v) for v in weight.shape) != (2048, 6144)
        or int(gate_up.numel()) < 12288
        or int(act.numel()) < 6144
        or int(out.numel()) < 2048
        or (not gate_up.is_contiguous())
        or (not weight.is_contiguous())
        or (not act.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    act_block_k = max(16, int(act_block_k))
    _aicas_down_silu_preact_kernel[(triton.cdiv(6144, act_block_k),)](
        gate_up.view(-1),
        act.view(-1),
        6144,
        BLOCK_K=act_block_k,
    )
    _aicas_fp16_down_from_act_kernel[(triton.cdiv(2048, block_n),)](
        act.view(-1),
        weight,
        out.view(-1),
        2048,
        6144,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_fp16_down_from_act(
    act: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int = 8,
    block_k: int = 128,
) -> bool:
    """Launch exact fp16 down_proj GEMV from a precomputed SiLU(gate) * up vector."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(act, torch.Tensor))
        or (not isinstance(weight, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not act.is_cuda)
        or (not weight.is_cuda)
        or (not out.is_cuda)
        or act.dtype != torch.float32
        or weight.dtype != torch.float16
        or out.dtype != torch.float32
        or weight.ndim != 2
        or tuple(int(v) for v in weight.shape) != (2048, 6144)
        or int(act.numel()) < 6144
        or int(out.numel()) < 2048
        or (not act.is_contiguous())
        or (not weight.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False

    block_n = max(1, int(block_n))
    block_k = max(16, int(block_k))
    _aicas_fp16_down_from_act_kernel[(triton.cdiv(2048, block_n),)](
        act.view(-1),
        weight,
        out.view(-1),
        2048,
        6144,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return True


def _aicas_triton_rmsnorm_from_sumsq_fp16_m1(
    hidden: torch.Tensor,
    sum_squares: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> bool:
    """Normalize one decode vector into a reusable fp16 buffer."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(hidden, torch.Tensor))
        or (not isinstance(sum_squares, torch.Tensor))
        or (not isinstance(norm_weight, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not hidden.is_cuda)
        or (not sum_squares.is_cuda)
        or (not norm_weight.is_cuda)
        or (not out.is_cuda)
        or int(hidden.numel()) != 2048
        or int(sum_squares.numel()) < 1
        or int(norm_weight.numel()) != 2048
        or int(out.numel()) < 2048
        or hidden.dtype != torch.float16
        or norm_weight.dtype != torch.float16
        or out.dtype != torch.float16
        or (not hidden.is_contiguous())
        or (not sum_squares.is_contiguous())
        or (not norm_weight.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False
    try:
        _aicas_rmsnorm_from_sumsq_fp16_m1_kernel[(1,)](
            hidden.view(-1),
            sum_squares.view(-1),
            norm_weight.view(-1),
            out.view(-1),
            K=2048,
            EPS=float(eps),
            BLOCK_K=2048,
        )
        return True
    except Exception:
        return False


def _aicas_triton_decode_softmax_value_gqa_m1(
    scores: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    out: torch.Tensor,
    *,
    step_offset: int = 0,
) -> bool:
    """Fuse decode softmax, V matmul, and GQA head merge for one token block callsite."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(scores, torch.Tensor))
        or (not isinstance(value_cache, torch.Tensor))
        or (not isinstance(cache_position, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not scores.is_cuda)
        or (not value_cache.is_cuda)
        or (not cache_position.is_cuda)
        or (not out.is_cuda)
        or scores.dtype != torch.float32
        or value_cache.dtype != torch.float16
        or out.dtype != torch.float32
        or scores.ndim != 3
        or value_cache.ndim != 4
        or int(scores.shape[0]) != 16
        or int(scores.shape[1]) != 1
        or int(value_cache.shape[0]) != 1
        or int(value_cache.shape[1]) != 8
        or int(value_cache.shape[3]) != 128
        or int(out.numel()) < 2048
        or int(cache_position.numel()) < 1
    ):
        return False
    kv_len = int(value_cache.shape[2])
    if int(scores.shape[2]) != kv_len:
        return False
    try:
        _aicas_decode_softmax_value_gqa_m1_kernel[(16,)](
            scores,
            value_cache,
            cache_position,
            out.view(-1),
            STEP_OFFSET=int(step_offset),
            KV_LEN=kv_len,
            HEAD_DIM=128,
            BLOCK_K=triton.next_power_of_2(kv_len),
            BLOCK_V=64,
            BLOCK_D=128,
        )
        return True
    except Exception:
        return False


def _aicas_triton_decode_qkv_softmax_value_gqa_m1(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    out: torch.Tensor,
    *,
    step_offset: int = 0,
) -> bool:
    """Fuse decode QK scores, softmax, value matmul, and GQA head merge."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(query, torch.Tensor))
        or (not isinstance(key_cache, torch.Tensor))
        or (not isinstance(value_cache, torch.Tensor))
        or (not isinstance(cache_position, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not query.is_cuda)
        or (not key_cache.is_cuda)
        or (not value_cache.is_cuda)
        or (not cache_position.is_cuda)
        or (not out.is_cuda)
        or query.dtype not in (torch.float16, torch.float32)
        or key_cache.dtype != torch.float16
        or value_cache.dtype != torch.float16
        or out.dtype != torch.float32
        or key_cache.ndim != 4
        or value_cache.ndim != 4
        or int(key_cache.shape[0]) != 1
        or int(value_cache.shape[0]) != 1
        or int(key_cache.shape[1]) != 8
        or int(value_cache.shape[1]) != 8
        or int(key_cache.shape[3]) != 128
        or int(value_cache.shape[3]) != 128
        or int(key_cache.shape[2]) != int(value_cache.shape[2])
        or int(query.numel()) < 16 * 128
        or int(out.numel()) < 2048
        or int(cache_position.numel()) < 1
        or (not key_cache.is_contiguous())
        or (not value_cache.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False
    kv_len = int(key_cache.shape[2])
    if kv_len <= 0:
        return False
    try:
        use_pair = False
        use_fp32 = True
        block_k = 64
        attention_cap = 0
        scale = 0.29730177875068026
        if use_pair:
            _aicas_decode_qkv_softmax_value_gqa_pair_m1_kernel[(8,)](
                query.view(-1),
                key_cache,
                value_cache,
                cache_position,
                out.view(-1),
                STEP_OFFSET=int(step_offset),
                KV_LEN=kv_len,
                HEAD_DIM=128,
                SCALE=scale,
                ATTENTION_CAP=attention_cap,
                BLOCK_K=block_k,
                BLOCK_D=128,
            )
            return True
        if use_fp32:
            _aicas_decode_qkv_softmax_value_gqa_m1_fp32_kernel[(16,)](
                query.view(-1),
                key_cache,
                value_cache,
                cache_position,
                out.view(-1),
                STEP_OFFSET=int(step_offset),
                KV_LEN=kv_len,
                HEAD_DIM=128,
                SCALE=scale,
                ATTENTION_CAP=attention_cap,
                BLOCK_K=block_k,
                BLOCK_D=128,
            )
            return True
        _aicas_decode_qkv_softmax_value_gqa_m1_kernel[(16,)](
            query.view(-1),
            key_cache,
            value_cache,
            cache_position,
            out.view(-1),
            STEP_OFFSET=int(step_offset),
            KV_LEN=kv_len,
            HEAD_DIM=128,
            SCALE=scale,
            ATTENTION_CAP=attention_cap,
            BLOCK_K=block_k,
            BLOCK_D=128,
        )
        return True
    except Exception:
        return False


def _aicas_triton_decode_value_merge_gqa_m1(
    probs: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    effective_kv_len: Optional[int] = None,
    segment_len: Optional[int] = None,
    bucket_len: Optional[int] = None,
) -> bool:
    """Fuse Inductor's segmented decode value matmul with the 3-way GQA merge."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(probs, torch.Tensor))
        or (not isinstance(value_cache, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not probs.is_cuda)
        or (not value_cache.is_cuda)
        or (not out.is_cuda)
        or probs.dtype != torch.float32
        or value_cache.dtype != torch.float16
        or out.dtype != torch.float32
        or value_cache.ndim != 4
        or int(value_cache.shape[0]) != 1
        or int(value_cache.shape[1]) != 8
        or int(value_cache.shape[3]) != 128
        or int(out.numel()) < 2048
        or (not probs.is_contiguous())
        or (not value_cache.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False
    kv_len = int(value_cache.shape[2])
    try:
        bucket = int(bucket_len) if bucket_len is not None else kv_len
    except Exception:
        bucket = kv_len
    bucket = max(0, min(int(kv_len), int(bucket)))
    try:
        seg_len = int(segment_len) if segment_len is not None else int(effective_kv_len)
    except Exception:
        seg_len = 0
    if seg_len <= 0:
        seg_len = (bucket + 2) // 3
    if bucket <= 0 or seg_len <= 0 or bucket > seg_len * 3 or int(probs.numel()) < 16 * bucket:
        return False
    try:
        if os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FUSED_VALUE_MERGE_PAIR", "0") != "0":
            _aicas_decode_value_merge_gqa_pair_m1_kernel[(8, 4)](
                probs.view(-1),
                value_cache,
                out.view(-1),
                BUCKET_LEN=bucket,
                SEGMENT_LEN=seg_len,
                HEAD_DIM=128,
                BLOCK_V=64,
                BLOCK_D=32,
            )
        else:
            _aicas_decode_value_merge_gqa_m1_kernel[(16, 4)](
                probs.view(-1),
                value_cache,
                out.view(-1),
                BUCKET_LEN=bucket,
                SEGMENT_LEN=seg_len,
                HEAD_DIM=128,
                BLOCK_V=64,
                BLOCK_D=32,
            )
        return True
    except Exception:
        return False


def _aicas_triton_decode_value_merge_atomic_gqa_m1(
    probs: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    effective_kv_len: Optional[int] = None,
) -> bool:
    """Fuse value matmul and GQA merge while preserving the 3-chunk value parallelism."""
    if (
        (not _TRITON_AVAILABLE)
        or (not isinstance(probs, torch.Tensor))
        or (not isinstance(value_cache, torch.Tensor))
        or (not isinstance(out, torch.Tensor))
        or (not probs.is_cuda)
        or (not value_cache.is_cuda)
        or (not out.is_cuda)
        or probs.dtype != torch.float32
        or value_cache.dtype != torch.float16
        or out.dtype != torch.float32
        or value_cache.ndim != 4
        or int(value_cache.shape[0]) != 1
        or int(value_cache.shape[1]) != 8
        or int(value_cache.shape[3]) != 128
        or int(out.numel()) < 2048
        or (not probs.is_contiguous())
        or (not value_cache.is_contiguous())
        or (not out.is_contiguous())
    ):
        return False
    kv_len = int(value_cache.shape[2])
    try:
        eff_len = int(effective_kv_len) if effective_kv_len is not None else kv_len
    except Exception:
        eff_len = kv_len
    eff_len = max(0, min(int(kv_len), int(eff_len)))
    if eff_len <= 0 or int(probs.numel()) < 16 * kv_len:
        return False
    try:
        out.zero_()
        _aicas_decode_value_merge_atomic_gqa_m1_kernel[(16, 3, 4)](
            probs.view(-1),
            value_cache,
            out.view(-1),
            KV_LEN=kv_len,
            EFFECTIVE_KV_LEN=eff_len,
            HEAD_DIM=128,
            CHUNK_V=triton.cdiv(eff_len, 3),
            BLOCK_V=128,
            BLOCK_D=32,
        )
        return True
    except Exception:
        return False


def _aicas_triton_rmsnorm_linear_decode_m1(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    bias: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if (
        (not _TRITON_AVAILABLE)
        or (not x.is_cuda)
        or (not rms_weight.is_cuda)
        or (not linear_weight.is_cuda)
        or x.ndim != 2
        or int(x.shape[0]) != 1
        or x.dtype != torch.float16
        or rms_weight.dtype != torch.float16
        or linear_weight.dtype != torch.float16
        or int(x.shape[1]) != int(rms_weight.numel())
        or int(x.shape[1]) != int(linear_weight.shape[1])
    ):
        x_norm = x.to(torch.float32)
        x_norm = x_norm * torch.rsqrt(x_norm.pow(2).mean(-1, keepdim=True) + float(eps))
        x_norm = (x_norm.to(x.dtype) * rms_weight).to(x.dtype)
        return F.linear(x_norm, linear_weight, bias)

    x_1d = x.contiguous().view(-1)
    K = int(x_1d.shape[0])
    N = int(linear_weight.shape[0])
    out = torch.empty((N,), device=x.device, dtype=torch.float16)
    has_bias = (
        isinstance(bias, torch.Tensor)
        and bias.is_cuda
        and int(bias.numel()) == N
        and bias.device == x.device
        and bias.dtype == torch.float16
    )
    bias_ptr = bias if has_bias else linear_weight
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    _aicas_rmsnorm_linear_m1_kernel[grid](
        x_1d,
        rms_weight,
        linear_weight,
        bias_ptr,
        out,
        K,
        N,
        linear_weight.stride(0),
        linear_weight.stride(1),
        float(eps),
        HAS_BIAS=has_bias,
    )
    return out.view(1, N)


def _aicas_triton_fp16_top1_decode_m1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    block_max_buffer: torch.Tensor = None,
    block_idx_buffer: torch.Tensor = None,
    out_idx64_buffer: torch.Tensor = None,
    fp16_compare: bool = True,
    dot_kernel: bool = False,
    fixed_block_n: int = 0,
    fixed_block_k: int = 0,
) -> torch.Tensor:
    """
    Fused decode top-1 for a dense fp16 lm_head without materializing full logits.

    x: [1, hidden]
    weight: [vocab, hidden]
    Returns next token id tensor with shape [1] and dtype torch.long.
    """
    if (
        (not _TRITON_AVAILABLE)
        or (not x.is_cuda)
        or (not weight.is_cuda)
        or x.ndim != 2
        or int(x.shape[0]) != 1
        or weight.ndim != 2
        or int(x.shape[1]) != int(weight.shape[1])
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
    ):
        logits = torch.nn.functional.linear(x, weight, bias=bias)
        return torch.argmax(logits, dim=-1).to(torch.long)

    x_fp16 = x.contiguous().view(-1)
    weight_fp16 = weight.contiguous()
    K = int(x_fp16.shape[0])
    N = int(weight_fp16.shape[0])

    use_fixed_kernel = bool(fixed_block_n > 0 and fixed_block_k > 0)
    # Prefer caller-provided fixed launch blocks when available.
    block_n = int(fixed_block_n) if use_fixed_kernel else 512
    block_k = int(fixed_block_k) if use_fixed_kernel else 128
    max_blocks = triton.cdiv(N, block_n)

    # Reuse buffers if provided
    if isinstance(block_max_buffer, torch.Tensor) and isinstance(block_idx_buffer, torch.Tensor):
        if block_max_buffer.numel() >= max_blocks and block_idx_buffer.numel() >= max_blocks:
            block_max = block_max_buffer[:max_blocks]
            block_idx = block_idx_buffer[:max_blocks]
        else:
            block_max = torch.empty((max_blocks,), device=x.device, dtype=torch.float32)
            block_idx = torch.empty((max_blocks,), device=x.device, dtype=torch.int32)
    else:
        block_max = torch.empty((max_blocks,), device=x.device, dtype=torch.float32)
        block_idx = torch.empty((max_blocks,), device=x.device, dtype=torch.int32)

    block_max.fill_(-1.0e30)
    block_idx.zero_()

    has_bias = (
        isinstance(bias, torch.Tensor)
        and bias.is_cuda
        and int(bias.numel()) == N
        and bias.device == x.device
    )
    bias_ptr = bias if has_bias else weight_fp16

    if use_fixed_kernel:
        grid = (triton.cdiv(N, block_n),)
        _aicas_fp16_decode_m1_block_top1_fixed_kernel[grid](
            x_fp16,
            weight_fp16,
            bias_ptr,
            block_max,
            block_idx,
            N,
            K,
            weight_fp16.stride(0),
            weight_fp16.stride(1),
            HAS_BIAS=has_bias,
            FP16_COMPARE=bool(fp16_compare),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
    else:
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
        block_kernel = (
            _aicas_fp16_decode_m1_block_top1_dot_kernel
            if bool(dot_kernel)
            else _aicas_fp16_decode_m1_block_top1_kernel
        )
        block_kernel[grid](
            x_fp16,
            weight_fp16,
            bias_ptr,
            block_max,
            block_idx,
            N,
            K,
            weight_fp16.stride(0),
            weight_fp16.stride(1),
            HAS_BIAS=has_bias,
            FP16_COMPARE=bool(fp16_compare),
        )

    # Fast reduction path for small number of blocks
    if max_blocks <= 2048:
        if (
            isinstance(out_idx64_buffer, torch.Tensor)
            and out_idx64_buffer.is_cuda
            and out_idx64_buffer.device == x.device
            and out_idx64_buffer.dtype == torch.int64
            and int(out_idx64_buffer.numel()) >= 1
        ):
            out_idx64 = out_idx64_buffer.view(-1)[:1]
        else:
            out_idx64 = torch.empty((1,), device=x.device, dtype=torch.int64)
        _aicas_top1_reduce_1block_i64_kernel[(1,)](
            block_max,
            block_idx,
            out_idx64,
            max_blocks,
            N,
            BLOCK=2048,
        )
        return out_idx64.view(1)

    best_block = torch.argmax(block_max, dim=0)
    return block_idx[best_block].to(torch.long).clamp_(0, max(N - 1, 0)).view(1)


def _aicas_triton_fp16_top1_decode_small_m(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    fp16_compare: bool = True,
    fixed_block_n: int = 512,
    fixed_block_k: int = 128,
) -> torch.Tensor:
    if (
        (not _TRITON_AVAILABLE)
        or (not x.is_cuda)
        or (not weight.is_cuda)
        or x.ndim != 2
        or weight.ndim != 2
        or int(x.shape[1]) != int(weight.shape[1])
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
    ):
        logits = torch.nn.functional.linear(x, weight, bias=bias)
        return torch.argmax(logits, dim=-1).to(torch.long)

    x_fp16 = x.contiguous()
    weight_fp16 = weight.contiguous()
    M = int(x_fp16.shape[0])
    K = int(x_fp16.shape[1])
    N = int(weight_fp16.shape[0])
    block_n = int(fixed_block_n)
    block_k = int(fixed_block_k)
    max_blocks = triton.cdiv(N, block_n)
    block_max = torch.empty((M, max_blocks), device=x.device, dtype=torch.float32)
    block_idx = torch.empty((M, max_blocks), device=x.device, dtype=torch.int32)
    out_idx64 = torch.empty((M,), device=x.device, dtype=torch.int64)
    has_bias = (
        isinstance(bias, torch.Tensor)
        and bias.is_cuda
        and int(bias.numel()) == N
        and bias.device == x.device
    )
    bias_ptr = bias if has_bias else weight_fp16
    _aicas_fp16_decode_small_m_block_top1_fixed_kernel[(M, max_blocks)](
        x_fp16,
        weight_fp16,
        bias_ptr,
        block_max,
        block_idx,
        M,
        N,
        K,
        x_fp16.stride(0),
        x_fp16.stride(1),
        weight_fp16.stride(0),
        weight_fp16.stride(1),
        max_blocks,
        HAS_BIAS=has_bias,
        FP16_COMPARE=bool(fp16_compare),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    _aicas_top1_reduce_small_m_i64_kernel[(M,)](
        block_max,
        block_idx,
        out_idx64,
        M,
        N,
        max_blocks,
        BLOCK=2048,
    )
    return out_idx64


def _aicas_triton_rmsnorm_fp16_top1_decode_m1(
    rms_weight: torch.Tensor,
    residual: torch.Tensor,
    mlp_out: torch.Tensor,
    sum_squares: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    block_max_buffer: torch.Tensor = None,
    block_idx_buffer: torch.Tensor = None,
    out_idx64_buffer: torch.Tensor = None,
    eps: float = 1e-6,
    fp16_compare: bool = False,
    fixed_block_n: int = 512,
    fixed_block_k: int = 128,
) -> torch.Tensor:
    """
    Fused final RMSNorm + dense fp16 lm_head + top-1 for generated token blocks.

    This mirrors the final Inductor artifact pattern:
    logits = lm_head(rms_norm(residual + mlp_out)), argmax(logits)
    but avoids materializing the full [1, vocab] logits tensor.
    """
    if (
        (not _TRITON_AVAILABLE)
        or (not rms_weight.is_cuda)
        or (not residual.is_cuda)
        or (not mlp_out.is_cuda)
        or (not sum_squares.is_cuda)
        or (not weight.is_cuda)
        or rms_weight.ndim != 1
        or residual.ndim != 1
        or mlp_out.ndim != 1
        or weight.ndim != 2
        or int(residual.numel()) != int(mlp_out.numel())
        or int(residual.numel()) != int(rms_weight.numel())
        or int(weight.shape[1]) != int(residual.numel())
        or residual.dtype != torch.float16
        or mlp_out.dtype not in (torch.float16, torch.float32)
        or rms_weight.dtype != torch.float16
        or weight.dtype != torch.float16
    ):
        hidden = (
            (residual + mlp_out)
            * torch.rsqrt(sum_squares.reshape(()) / float(max(1, residual.numel())) + float(eps))
            * rms_weight.float()
        ).to(dtype=torch.float16).view(1, -1)
        logits = torch.nn.functional.linear(hidden, weight, bias=bias)
        return torch.argmax(logits, dim=-1).to(torch.long)

    K = int(residual.numel())
    N = int(weight.shape[0])
    block_n = int(fixed_block_n) if int(fixed_block_n) > 0 else 512
    block_k = int(fixed_block_k) if int(fixed_block_k) > 0 else 128
    max_blocks = triton.cdiv(N, block_n)

    if isinstance(block_max_buffer, torch.Tensor) and isinstance(block_idx_buffer, torch.Tensor):
        if block_max_buffer.numel() >= max_blocks and block_idx_buffer.numel() >= max_blocks:
            block_max = block_max_buffer[:max_blocks]
            block_idx = block_idx_buffer[:max_blocks]
        else:
            block_max = torch.empty((max_blocks,), device=residual.device, dtype=torch.float32)
            block_idx = torch.empty((max_blocks,), device=residual.device, dtype=torch.int32)
    else:
        block_max = torch.empty((max_blocks,), device=residual.device, dtype=torch.float32)
        block_idx = torch.empty((max_blocks,), device=residual.device, dtype=torch.int32)

    block_max.fill_(-1.0e30)
    block_idx.zero_()

    has_bias = (
        isinstance(bias, torch.Tensor)
        and bias.is_cuda
        and int(bias.numel()) == N
        and bias.device == residual.device
    )
    bias_ptr = bias if has_bias else weight
    grid = (max_blocks,)
    _aicas_rmsnorm_fp16_decode_m1_block_top1_fixed_kernel[grid](
        rms_weight.contiguous(),
        residual.contiguous(),
        mlp_out.contiguous(),
        sum_squares.contiguous(),
        weight.contiguous(),
        bias_ptr,
        block_max,
        block_idx,
        N,
        K,
        weight.stride(0),
        weight.stride(1),
        float(eps),
        HAS_BIAS=has_bias,
        FP16_COMPARE=bool(fp16_compare),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    if (
        isinstance(out_idx64_buffer, torch.Tensor)
        and out_idx64_buffer.is_cuda
        and out_idx64_buffer.device == residual.device
        and out_idx64_buffer.dtype == torch.int64
        and int(out_idx64_buffer.numel()) >= 1
    ):
        out_idx64 = out_idx64_buffer.view(-1)[:1]
    else:
        out_idx64 = torch.empty((1,), device=residual.device, dtype=torch.int64)
    _aicas_top1_reduce_1block_i64_kernel[(1,)](
        block_max,
        block_idx,
        out_idx64,
        max_blocks,
        N,
        BLOCK=2048,
    )
    return out_idx64.view(1)
