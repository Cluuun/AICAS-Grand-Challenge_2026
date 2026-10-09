from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _embedding_gather_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    batch,
    q_len,
    vocab_size,
    hidden_size,
    stride_ids_b,
    stride_ids_q,
    stride_wv,
    stride_wh,
    stride_ob,
    stride_oq,
    stride_oh,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    q = pid % q_len
    b = pid // q_len

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden_size

    token_id = tl.load(ids_ptr + b * stride_ids_b + q * stride_ids_q).to(tl.int64)
    valid = (token_id >= 0) & (token_id < vocab_size)

    w_ptr = weight_ptr + token_id * stride_wv + h_offs * stride_wh
    out = tl.load(w_ptr, mask=h_mask & valid, other=0.0)
    tl.store(out_ptr + b * stride_ob + q * stride_oq + h_offs * stride_oh, out, mask=h_mask)


def can_use_embedding_gather_triton(input_ids: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None) -> bool:
    if not isinstance(input_ids, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= 0:
        return False
    if weight.ndim != 2:
        return False
    if not input_ids.is_cuda or not weight.is_cuda:
        return False
    if input_ids.dtype not in (torch.int32, torch.int64):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if out is not None:
        if not isinstance(out, torch.Tensor):
            return False
        if out.shape != (int(input_ids.shape[0]), int(input_ids.shape[1]), int(weight.shape[1])):
            return False
        if out.device != input_ids.device or out.dtype != weight.dtype:
            return False
    return True


def embedding_gather_triton(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_h: int = 256,
):
    if not can_use_embedding_gather_triton(input_ids, weight, out):
        raise RuntimeError("embedding_gather_triton requires CUDA 2D ids and 2D weight")

    batch = int(input_ids.shape[0])
    q_len = int(input_ids.shape[1])
    hidden = int(weight.shape[1])
    vocab = int(weight.shape[0])

    if out is None:
        out = torch.empty((batch, q_len, hidden), device=input_ids.device, dtype=weight.dtype)

    grid = (batch * q_len, triton.cdiv(hidden, block_h))
    _embedding_gather_kernel[grid](
        input_ids,
        weight,
        out,
        batch,
        q_len,
        vocab,
        hidden,
        int(input_ids.stride(0)),
        int(input_ids.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=2,
    )
    return out
