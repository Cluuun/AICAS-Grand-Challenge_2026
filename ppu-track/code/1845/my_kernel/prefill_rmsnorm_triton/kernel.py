from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def rmsnorm_prefill_rowwise_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_m,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_row = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x_row * x_row, axis=0) / N
    inv = tl.rsqrt(var + eps)
    normed = (x_row * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = normed * w
    tl.store(y_ptr + row * stride_m + offs, y, mask=mask)
