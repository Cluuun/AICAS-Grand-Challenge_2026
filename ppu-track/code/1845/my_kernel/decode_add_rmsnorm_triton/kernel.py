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
def add_rmsnorm_rowwise_kernel(
    a_ptr,
    b_ptr,
    w_ptr,
    added_ptr,
    norm_ptr,
    stride_m,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(a_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    s = (a + b).to(tl.bfloat16).to(tl.float32)
    tl.store(added_ptr + row * stride_m + offs, s, mask=mask)

    var = tl.sum(s * s, axis=0) / N
    inv = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = (s * inv).to(tl.bfloat16).to(tl.float32)
    y = normed * w
    tl.store(norm_ptr + row * stride_m + offs, y, mask=mask)


@triton.jit
def add_rmsnorm_rowwise_sum_kernel(
    a_ptr,
    b_ptr,
    added_ptr,
    partials_ptr,
    stride_m,
    N,
    stride_partial,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(a_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    s = (a + b).to(tl.bfloat16).to(tl.float32)
    tl.store(added_ptr + row * stride_m + offs, s, mask=mask)
    partial = tl.sum(s * s, axis=0)
    tl.store(partials_ptr + row * stride_partial + tile, partial)


@triton.jit
def add_rmsnorm_rowwise_finalize_kernel(
    added_ptr,
    w_ptr,
    norm_ptr,
    sumsq_ptr,
    stride_m,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    s = tl.load(added_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(sumsq_ptr + row).to(tl.float32) / N
    inv = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = (s * inv).to(tl.bfloat16).to(tl.float32)
    y = normed * w
    tl.store(norm_ptr + row * stride_m + offs, y, mask=mask)


@triton.jit
def add_rmsnorm_rowwise_reduce_kernel(
    partials_ptr,
    sumsq_ptr,
    num_tiles,
):
    row = tl.program_id(0)
    tile = tl.arange(0, 64)
    mask = tile < num_tiles
    partials = tl.load(partials_ptr + row * num_tiles + tile, mask=mask, other=0.0).to(tl.float32)
    tl.store(sumsq_ptr + row, tl.sum(partials, axis=0))
