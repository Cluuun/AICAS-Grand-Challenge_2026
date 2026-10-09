from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _eagle3_expand_topk4_kernel(
    topk_p_ptr,
    topk_index_ptr,
    prev_scores_ptr,
    out_hidden_ptr,
    tree_mask_ptr,
    cu_scores_ptr,
    out_scores_ptr,
    topk_cs_index_ptr,
    out_ids_ptr,
    input_ids_ptr,
    input_hidden_ptr,
    next_tree_mask_ptr,
    H: tl.constexpr,
    MASK_WIDTH: tl.constexpr,
    stride_p0: tl.constexpr,
    stride_p1: tl.constexpr,
    stride_idx0: tl.constexpr,
    stride_idx1: tl.constexpr,
    stride_oh1: tl.constexpr,
    stride_oh2: tl.constexpr,
    stride_tm2: tl.constexpr,
    stride_tm3: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_MASK: tl.constexpr,
):
    cand = tl.arange(0, 16)
    parent = cand // 4
    child = cand - parent * 4

    vals = (
        tl.load(topk_p_ptr + parent * stride_p0 + child * stride_p1).to(tl.float32)
        + tl.load(prev_scores_ptr + parent).to(tl.float32)
    )
    tl.store(cu_scores_ptr + cand, vals)

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    mask_offs = tl.arange(0, BLOCK_MASK)
    old_mask = mask_offs < MASK_WIDTH
    new_mask = mask_offs < MASK_WIDTH + 4

    for rank in tl.static_range(0, 4):
        best_val = tl.max(vals, axis=0)
        best_pos = tl.argmax(vals, axis=0)
        best_parent = best_pos // 4
        best_child = best_pos - best_parent * 4
        token = tl.load(topk_index_ptr + best_parent * stride_idx0 + best_child * stride_idx1)

        tl.store(out_scores_ptr + rank, best_val)
        tl.store(topk_cs_index_ptr + rank, best_pos)
        tl.store(out_ids_ptr + rank, best_parent)
        tl.store(input_ids_ptr + rank, token)

        hidden = tl.load(
            out_hidden_ptr + best_parent * stride_oh1 + h_offs * stride_oh2,
            mask=h_mask,
            other=0.0,
        )
        tl.store(input_hidden_ptr + rank * H + h_offs, hidden, mask=h_mask)

        inherited = tl.load(
            tree_mask_ptr + best_parent * stride_tm2 + mask_offs * stride_tm3,
            mask=old_mask,
            other=0.0,
        )
        appended = tl.where(mask_offs - MASK_WIDTH == rank, 1.0, 0.0)
        mask_vals = tl.where(old_mask, inherited, appended)
        tl.store(next_tree_mask_ptr + rank * (MASK_WIDTH + 4) + mask_offs, mask_vals, mask=new_mask)
        vals = tl.where(cand == best_pos, float("-inf"), vals)


def _can_use(
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    scores: torch.Tensor,
    out_hidden: torch.Tensor,
    tree_mask: torch.Tensor,
    top_k: int,
) -> bool:
    if int(top_k) != 4:
        return False
    if not all(isinstance(t, torch.Tensor) for t in (topk_index, topk_p, scores, out_hidden, tree_mask)):
        return False
    if not all(t.is_cuda for t in (topk_index, topk_p, scores, out_hidden, tree_mask)):
        return False
    if topk_index.shape != (4, 4) or topk_p.shape != (4, 4) or scores.shape != (4,):
        return False
    if out_hidden.ndim != 3 or int(out_hidden.shape[0]) != 1 or int(out_hidden.shape[1]) != 4:
        return False
    if tree_mask.ndim != 4 or tuple(tree_mask.shape[:3]) != (1, 1, 4):
        return False
    if topk_index.dtype != torch.long:
        return False
    if topk_p.dtype != torch.float32 or scores.dtype != torch.float32:
        return False
    if out_hidden.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if tree_mask.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    hidden_dim = int(out_hidden.shape[-1])
    mask_width = int(tree_mask.shape[-1])
    return hidden_dim > 0 and hidden_dim <= 8192 and mask_width > 0 and mask_width <= 128


def _workspace(
    workspace: dict | None,
    *,
    device: torch.device,
    hidden_dim: int,
    hidden_dtype: torch.dtype,
    score_dtype: torch.dtype,
    tree_mask_dtype: torch.dtype,
    score_slots: int,
) -> dict:
    score_slots = max(1, int(score_slots))
    if (
        workspace is None
        or not isinstance(workspace, dict)
        or workspace.get("device") != device
        or workspace.get("hidden_dim") != hidden_dim
        or workspace.get("hidden_dtype") != hidden_dtype
        or workspace.get("score_dtype") != score_dtype
        or workspace.get("tree_mask_dtype") != tree_mask_dtype
        or int(workspace.get("score_slots", 0)) < score_slots
    ):
        workspace = {
            "device": device,
            "hidden_dim": hidden_dim,
            "hidden_dtype": hidden_dtype,
            "score_dtype": score_dtype,
            "tree_mask_dtype": tree_mask_dtype,
            "score_slots": score_slots,
            "tree_masks": {},
            "cu_scores": torch.empty((score_slots, 4, 4), device=device, dtype=score_dtype),
            "scores": torch.empty((score_slots, 4), device=device, dtype=score_dtype),
            "topk_cs_index_slots": torch.empty((score_slots, 4), device=device, dtype=torch.long),
            "out_ids": torch.empty((4,), device=device, dtype=torch.long),
            "input_ids": torch.empty((1, 4), device=device, dtype=torch.long),
            "input_hidden": torch.empty((1, 4, hidden_dim), device=device, dtype=hidden_dtype),
        }
    return workspace


def _tree_mask_workspace(workspace: dict, mask_width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = int(mask_width)
    tree_masks = workspace.setdefault("tree_masks", {})
    out = tree_masks.get(key)
    shape = (1, 1, 4, key + 4)
    if out is None or out.device != device or out.dtype != dtype or tuple(out.shape) != shape:
        out = torch.empty(shape, device=device, dtype=dtype)
        tree_masks[key] = out
    return out


def run_eagle3_expand_topk4(
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    scores: torch.Tensor,
    out_hidden: torch.Tensor,
    tree_mask: torch.Tensor,
    *,
    workspace: dict | None = None,
    score_slot: int = 0,
    score_slots: int = 1,
):
    if not _can_use(topk_index, topk_p, scores, out_hidden, tree_mask, 4):
        return None, workspace

    hidden_dim = int(out_hidden.shape[-1])
    mask_width = int(tree_mask.shape[-1])
    workspace = _workspace(
        workspace,
        device=out_hidden.device,
        hidden_dim=hidden_dim,
        hidden_dtype=out_hidden.dtype,
        score_dtype=topk_p.dtype,
        tree_mask_dtype=tree_mask.dtype,
        score_slots=score_slots,
    )
    score_slot = int(score_slot) % int(workspace["score_slots"])
    cu_scores = workspace["cu_scores"][score_slot]
    out_scores = workspace["scores"][score_slot]
    topk_cs_index = workspace["topk_cs_index_slots"][score_slot]
    next_tree_mask = _tree_mask_workspace(workspace, mask_width, out_hidden.device, tree_mask.dtype)
    block_h = triton.next_power_of_2(hidden_dim)
    block_mask = triton.next_power_of_2(mask_width + 4)

    _eagle3_expand_topk4_kernel[(1,)](
        topk_p,
        topk_index,
        scores,
        out_hidden,
        tree_mask,
        cu_scores,
        out_scores,
        topk_cs_index,
        workspace["out_ids"],
        workspace["input_ids"],
        workspace["input_hidden"],
        next_tree_mask,
        H=hidden_dim,
        MASK_WIDTH=mask_width,
        stride_p0=int(topk_p.stride(0)),
        stride_p1=int(topk_p.stride(1)),
        stride_idx0=int(topk_index.stride(0)),
        stride_idx1=int(topk_index.stride(1)),
        stride_oh1=int(out_hidden.stride(1)),
        stride_oh2=int(out_hidden.stride(2)),
        stride_tm2=int(tree_mask.stride(2)),
        stride_tm3=int(tree_mask.stride(3)),
        BLOCK_H=block_h,
        BLOCK_MASK=block_mask,
        num_warps=8,
        num_stages=2,
    )
    return (
        cu_scores,
        topk_cs_index,
        out_scores,
        workspace["out_ids"],
        workspace["input_hidden"],
        workspace["input_ids"],
        next_tree_mask,
    ), workspace
