# Triton: fused GELU-tanh + down-proj for Qwen3-VL-2B ViT MLP (I=4096, H=1024, bf16).
# One program per token row (grid M) to avoid M×(H/BLOCK_H) launch storms.
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gelu_tanh_bf16(x):
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    xf = x.to(tl.float32)
    t = sqrt_2_over_pi * (xf + coeff * xf * xf * xf)
    return (0.5 * xf * (1.0 + libdevice.tanh(t))).to(x.dtype)


@triton.jit
def gelu_fc2_qwen3vl_2b_row(
    up_ptr,
    w2_ptr,
    b2_ptr,
    out_ptr,
    M,
    stride_um,
    stride_om,
    stride_w2_0,
    stride_w2_1,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    up_row = up_ptr + row * stride_um
    out_row = out_ptr + row * stride_om

    for h0 in range(0, H, BLOCK_H):
        hh = h0 + tl.arange(0, BLOCK_H)
        mask_h = hh < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i0 in range(0, I, BLOCK_I):
            ii = i0 + tl.arange(0, BLOCK_I)
            mask_i = ii < I
            u = tl.load(up_row + ii, mask=mask_i, other=0.0).to(tl.bfloat16)
            u = _gelu_tanh_bf16(u)
            w_blk = w2_ptr + hh[:, None] * stride_w2_0 + ii[None, :] * stride_w2_1
            m2 = mask_h[:, None] & mask_i[None, :]
            wv = tl.load(w_blk, mask=m2, other=0.0).to(tl.float32)
            acc += tl.sum(wv * u.to(tl.float32)[None, :], axis=1)
        b2v = tl.load(b2_ptr + hh, mask=mask_h, other=0.0).to(tl.float32)
        acc = acc + b2v
        tl.store(out_row + hh, acc.to(tl.bfloat16), mask=mask_h)
