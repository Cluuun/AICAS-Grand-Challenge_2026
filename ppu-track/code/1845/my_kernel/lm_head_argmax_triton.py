from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lm_head_block_max_bf16_kernel(
    x_ptr,
    w_ptr,
    out_vals_ptr,
    out_idxs_ptr,
    V,
    K,
    stride_wv,
    stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < V

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k_start = 0
    while k_start < K:
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + row_offs[:, None] * stride_wv + k_offs[None, :] * stride_wk,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        k_start += BLOCK_K

    acc = acc.to(tl.bfloat16).to(tl.float32)

    neg_inf = float("-inf")
    acc = tl.where(row_mask, acc, neg_inf)
    max_val = tl.max(acc, axis=0)
    max_pos = tl.argmax(acc, axis=0)
    max_idx = pid * BLOCK_M + max_pos

    tl.store(out_vals_ptr + pid, max_val)
    tl.store(out_idxs_ptr + pid, max_idx)


def can_use_lm_head_argmax_triton(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
        return False
    if weight.ndim != 2:
        return False
    if hidden_states.shape[-1] != weight.shape[1]:
        return False
    if not hidden_states.is_cuda or not weight.is_cuda:
        return False
    if hidden_states.dtype != torch.bfloat16:
        return False
    if hidden_states.dtype != weight.dtype:
        return False
    if bias is not None:
        return False
    return True


def can_use_lm_head_argmax_triton_multi(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if hidden_states.ndim == 3:
        if hidden_states.shape[0] != 1 or hidden_states.shape[1] <= 0:
            return False
        hidden_dim = hidden_states.shape[-1]
    elif hidden_states.ndim == 2:
        if hidden_states.shape[0] <= 0:
            return False
        hidden_dim = hidden_states.shape[-1]
    else:
        return False
    if weight.ndim != 2 or hidden_dim != weight.shape[1]:
        return False
    if not hidden_states.is_cuda or not weight.is_cuda:
        return False
    if hidden_states.dtype != torch.bfloat16:
        return False
    if hidden_states.dtype != weight.dtype:
        return False
    if bias is not None:
        return False
    return True


def can_use_lm_head_topk_triton_multi(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    top_k: int = 8,
) -> bool:
    if not can_use_lm_head_argmax_triton_multi(hidden_states, weight, bias):
        return False
    return 1 <= int(top_k) <= 16


@triton.jit
def _lm_head_multi_block_max_bf16_kernel(
    x_ptr,
    w_ptr,
    out_vals_ptr,
    out_idxs_ptr,
    V,
    K,
    NB,
    stride_xs,
    stride_xk,
    stride_wv,
    stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    row_offs = block_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < V

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k_start = 0
    while k_start < K:
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(
            x_ptr + seq_pid * stride_xs + k_offs * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + row_offs[:, None] * stride_wv + k_offs[None, :] * stride_wk,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        k_start += BLOCK_K

    acc = acc.to(tl.bfloat16).to(tl.float32)

    neg_inf = float("-inf")
    acc = tl.where(row_mask, acc, neg_inf)
    max_val = tl.max(acc, axis=0)
    max_pos = tl.argmax(acc, axis=0)
    max_idx = block_pid * BLOCK_M + max_pos
    out_offset = seq_pid * NB + block_pid

    tl.store(out_vals_ptr + out_offset, max_val)
    tl.store(out_idxs_ptr + out_offset, max_idx)


@triton.jit
def _lm_head_multi_block_stats_bf16_kernel(
    x_ptr,
    w_ptr,
    out_max_ptr,
    out_sum_ptr,
    V,
    K,
    NB,
    stride_xs,
    stride_xk,
    stride_wv,
    stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    row_offs = block_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < V

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k_start = 0
    while k_start < K:
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(
            x_ptr + seq_pid * stride_xs + k_offs * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + row_offs[:, None] * stride_wv + k_offs[None, :] * stride_wk,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        k_start += BLOCK_K

    acc = acc.to(tl.bfloat16).to(tl.float32)

    neg_inf = float("-inf")
    acc = tl.where(row_mask, acc, neg_inf)
    block_max = tl.max(acc, axis=0)
    exp_sum = tl.sum(tl.exp(acc - block_max), axis=0)
    out_offset = seq_pid * NB + block_pid
    tl.store(out_max_ptr + out_offset, block_max)
    tl.store(out_sum_ptr + out_offset, exp_sum)


@triton.jit
def _lm_head_multi_block_topk_bf16_kernel(
    x_ptr,
    w_ptr,
    out_vals_ptr,
    out_idxs_ptr,
    V,
    K,
    NB,
    TOP_K: tl.constexpr,
    stride_xs,
    stride_xk,
    stride_wv,
    stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    row_offs = block_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < V

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k_start = 0
    while k_start < K:
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(
            x_ptr + seq_pid * stride_xs + k_offs * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + row_offs[:, None] * stride_wv + k_offs[None, :] * stride_wk,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        k_start += BLOCK_K

    acc = acc.to(tl.bfloat16).to(tl.float32)

    neg_inf = float("-inf")
    acc = tl.where(row_mask, acc, neg_inf)
    base = (seq_pid * NB + block_pid) * TOP_K
    for rank in tl.static_range(0, TOP_K):
        max_val = tl.max(acc, axis=0)
        max_pos = tl.argmax(acc, axis=0)
        max_idx = block_pid * BLOCK_M + max_pos
        tl.store(out_vals_ptr + base + rank, max_val)
        tl.store(out_idxs_ptr + base + rank, max_idx)
        acc = tl.where(row_offs == max_idx, neg_inf, acc)


@triton.jit
def _lm_head_multi_final_topk_kernel(
    block_max_ptr,
    block_sum_ptr,
    cand_vals_ptr,
    cand_idxs_ptr,
    out_vals_ptr,
    out_idxs_ptr,
    NB: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    block_mask = offs < NB
    bmax = tl.load(block_max_ptr + seq_pid * NB + offs, mask=block_mask, other=float("-inf"))
    bsum = tl.load(block_sum_ptr + seq_pid * NB + offs, mask=block_mask, other=0.0)
    global_max = tl.max(bmax, axis=0)
    denom = tl.sum(tl.where(block_mask, bsum * tl.exp(bmax - global_max), 0.0), axis=0)
    lse = global_max + tl.log(denom)

    cand_count: tl.constexpr = NB * TOP_K
    cand_mask = offs < cand_count
    vals = tl.load(cand_vals_ptr + seq_pid * cand_count + offs, mask=cand_mask, other=float("-inf")) - lse
    idxs = tl.load(cand_idxs_ptr + seq_pid * cand_count + offs, mask=cand_mask, other=0)
    vals = tl.where(cand_mask, vals, float("-inf"))

    base = seq_pid * TOP_K
    for rank in tl.static_range(0, TOP_K):
        max_val = tl.max(vals, axis=0)
        max_pos = tl.argmax(vals, axis=0)
        max_idx = tl.max(tl.where(offs == max_pos, idxs, 0), axis=0)
        tl.store(out_vals_ptr + base + rank, max_val)
        tl.store(out_idxs_ptr + base + rank, max_idx)
        vals = tl.where(offs == max_pos, float("-inf"), vals)


def lm_head_argmax_triton_multi(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    workspace: dict | None = None,
    block_m: int = 128,
    block_k: int = 128,
):
    if hidden_states.ndim == 3:
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
    else:
        x = hidden_states
    if not x.is_contiguous():
        x = x.contiguous()

    seq_len = int(x.shape[0])
    vocab_size = int(weight.shape[0])
    num_blocks = triton.cdiv(vocab_size, block_m)

    if (
        workspace is None
        or not isinstance(workspace, dict)
        or workspace.get("device") != x.device
        or workspace.get("seq_len") != seq_len
        or workspace.get("num_blocks") != num_blocks
    ):
        workspace = {
            "device": x.device,
            "seq_len": seq_len,
            "num_blocks": num_blocks,
            "vals": torch.empty((seq_len, num_blocks), device=x.device, dtype=torch.float32),
            "idxs": torch.empty((seq_len, num_blocks), device=x.device, dtype=torch.int32),
        }

    out_vals = workspace["vals"]
    out_idxs = workspace["idxs"]
    if bias is not None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError("lm_head_argmax_triton_multi requires BF16 no-bias inputs")

    _lm_head_multi_block_max_bf16_kernel[(seq_len, num_blocks)](
        x,
        weight,
        out_vals,
        out_idxs,
        vocab_size,
        int(weight.shape[1]),
        num_blocks,
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    block_vals = out_vals[:, :num_blocks]
    block_idxs = out_idxs[:, :num_blocks]
    best_blocks = torch.argmax(block_vals, dim=1)
    tokens = block_idxs.gather(1, best_blocks.view(seq_len, 1)).to(dtype=torch.long).view(seq_len)
    return tokens, workspace


def lm_head_topk_triton_multi(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    top_k: int = 8,
    workspace: dict | None = None,
    block_m: int = 128,
    block_k: int = 128,
):
    if hidden_states.ndim == 3:
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
    else:
        x = hidden_states
    if not x.is_contiguous():
        x = x.contiguous()

    top_k = int(top_k)
    seq_len = int(x.shape[0])
    vocab_size = int(weight.shape[0])
    num_blocks = triton.cdiv(vocab_size, block_m)

    if (
        workspace is None
        or not isinstance(workspace, dict)
        or workspace.get("device") != x.device
        or workspace.get("seq_len") != seq_len
        or workspace.get("num_blocks") != num_blocks
        or workspace.get("top_k") != top_k
    ):
        workspace = {
            "device": x.device,
            "seq_len": seq_len,
            "num_blocks": num_blocks,
            "top_k": top_k,
            "block_max": torch.empty((seq_len, num_blocks), device=x.device, dtype=torch.float32),
            "block_sum": torch.empty((seq_len, num_blocks), device=x.device, dtype=torch.float32),
            "top_vals": torch.empty((seq_len, num_blocks, top_k), device=x.device, dtype=torch.float32),
            "top_idxs": torch.empty((seq_len, num_blocks, top_k), device=x.device, dtype=torch.int32),
            "out_vals": torch.empty((seq_len, top_k), device=x.device, dtype=torch.float32),
            "out_idxs": torch.empty((seq_len, top_k), device=x.device, dtype=torch.int32),
        }

    if bias is not None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError("lm_head_topk_triton_multi requires BF16 no-bias inputs")

    _lm_head_multi_block_stats_bf16_kernel[(seq_len, num_blocks)](
        x,
        weight,
        workspace["block_max"],
        workspace["block_sum"],
        vocab_size,
        int(weight.shape[1]),
        num_blocks,
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    _lm_head_multi_block_topk_bf16_kernel[(seq_len, num_blocks)](
        x,
        weight,
        workspace["top_vals"],
        workspace["top_idxs"],
        vocab_size,
        int(weight.shape[1]),
        num_blocks,
        top_k,
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    block_n = triton.next_power_of_2(num_blocks * top_k)
    _lm_head_multi_final_topk_kernel[(seq_len,)](
        workspace["block_max"],
        workspace["block_sum"],
        workspace["top_vals"],
        workspace["top_idxs"],
        workspace["out_vals"],
        workspace["out_idxs"],
        NB=num_blocks,
        TOP_K=top_k,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )
    return workspace["out_vals"], workspace["out_idxs"].to(dtype=torch.long), workspace


def lm_head_argmax_triton(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    workspace: dict | None = None,
    block_m: int = 128,
    block_k: int = 128,
):
    x = hidden_states.reshape(-1)
    vocab_size = int(weight.shape[0])
    num_blocks = triton.cdiv(vocab_size, block_m)

    if (
        workspace is None
        or not isinstance(workspace, dict)
        or workspace.get("device") != x.device
        or workspace.get("num_blocks") != num_blocks
    ):
        workspace = {
            "device": x.device,
            "num_blocks": num_blocks,
            "vals": torch.empty((num_blocks,), device=x.device, dtype=torch.float32),
            "idxs": torch.empty((num_blocks,), device=x.device, dtype=torch.int32),
        }

    out_vals = workspace["vals"]
    out_idxs = workspace["idxs"]
    if bias is not None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError("lm_head_argmax_triton requires BF16 no-bias inputs")

    _lm_head_block_max_bf16_kernel[(num_blocks,)](
        x,
        weight,
        out_vals,
        out_idxs,
        vocab_size,
        int(weight.shape[1]),
        int(weight.stride(0)),
        int(weight.stride(1)),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    block_vals = out_vals[:num_blocks]
    block_idxs = out_idxs[:num_blocks]
    best_block = torch.argmax(block_vals, dim=0)
    token = block_idxs.index_select(0, best_block.view(1)).to(dtype=torch.long)
    return token, workspace
