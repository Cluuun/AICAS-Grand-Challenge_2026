from __future__ import annotations

import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_configs(), key=["N"])
@triton.jit
def rmsnorm_rowwise_kernel(
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
    x_row_ptr = x_ptr + row * stride_m + offs
    x = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(var + eps)

    normed = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = normed * w
    tl.store(y_ptr + row * stride_m + offs, y, mask=mask)
