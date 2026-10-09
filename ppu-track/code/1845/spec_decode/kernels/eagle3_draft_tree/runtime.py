from __future__ import annotations

import torch
import triton
import triton.language as tl

from spec_decode.kernels.eagle3_tree_metadata.runtime import run_eagle3_tree_metadata


@triton.jit
def _eagle3_tree_select_kernel(
    sample_token_ptr,
    root_tokens_ptr,
    root_scores_ptr,
    score_slots_ptr,
    topk_ids_slots_ptr,
    topk_cs_index_slots_ptr,
    sorted_index_ptr,
    mask_index_ptr,
    draft_tokens_ptr,
    DEPTH: tl.constexpr,
    TOTAL: tl.constexpr,
    CAND: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_C)
    valid = offs < CAND
    is_root = offs < 4
    rel = offs - 4
    slot = rel // 16
    pos = rel - slot * 16
    safe_slot = tl.maximum(slot, 0)
    safe_pos = tl.maximum(pos, 0)

    root_scores = tl.load(root_scores_ptr + tl.minimum(offs, 3), mask=is_root, other=float("-inf"))
    step_scores = tl.load(score_slots_ptr + safe_slot * 16 + safe_pos, mask=valid & ~is_root, other=float("-inf"))
    vals = tl.where(is_root, root_scores, step_scores)

    for rank in tl.static_range(0, TOTAL):
        best_val = tl.max(vals, axis=0)
        best_pos = tl.argmax(vals, axis=0)
        tl.store(sorted_index_ptr + rank, best_pos)
        vals = tl.where(offs == best_pos, float("-inf"), vals)

    t_offs = tl.arange(0, BLOCK_T)
    sel = tl.load(sorted_index_ptr + t_offs, mask=t_offs < TOTAL, other=CAND + 1)
    for rank in tl.static_range(0, TOTAL):
        min_idx = tl.min(sel, axis=0)
        tl.store(sorted_index_ptr + rank, min_idx)
        sel = tl.where(sel == min_idx, CAND + 1, sel)

    root_token = tl.load(sample_token_ptr)
    tl.store(draft_tokens_ptr, root_token)

    for rank in tl.static_range(0, TOTAL):
        idx = tl.load(sorted_index_ptr + rank)
        is_root_idx = idx < 4
        idx_rel = idx - 4
        idx_slot = idx_rel // 16
        idx_pos = idx_rel - idx_slot * 16
        safe_slot = tl.maximum(idx_slot, 0)
        safe_pos = tl.maximum(idx_pos, 0)
        root_tok = tl.load(root_tokens_ptr + tl.minimum(idx, 3), mask=is_root_idx, other=0)
        step_tok = tl.load(
            topk_ids_slots_ptr + safe_slot * 16 + safe_pos,
            mask=~is_root_idx,
            other=0,
        )
        token = tl.where(
            is_root_idx,
            root_tok,
            step_tok,
        )
        tl.store(draft_tokens_ptr + rank + 1, token)

        parent_pos = idx // 4
        parent_slot = (parent_pos - 1) // 4
        parent_rank = parent_pos - 1 - parent_slot * 4
        safe_parent_slot = tl.maximum(parent_slot, 0)
        safe_parent_rank = tl.maximum(parent_rank, 0)
        bias = 1 + 16 * tl.maximum(safe_parent_slot - 1, 0) + tl.where(safe_parent_slot > 0, 4, 0)
        parent_from_slot = tl.load(
            topk_cs_index_slots_ptr + safe_parent_slot * 4 + safe_parent_rank,
            mask=parent_pos > 0,
            other=0,
        )
        parent = tl.where(
            parent_pos == 0,
            0,
            parent_from_slot + bias,
        )
        parent_score_index = parent - 1

        search = tl.full((), 0, dtype=tl.int32)
        for j in tl.static_range(0, TOTAL):
            cur = tl.load(sorted_index_ptr + j)
            search += tl.where(cur < parent_score_index, 1, 0)
        mask = tl.where(parent == 0, -1, search) + 1
        tl.store(mask_index_ptr + rank, mask)


def _can_use(
    sample_token: torch.Tensor,
    root_tokens: torch.Tensor,
    root_scores: torch.Tensor,
    score_slots: torch.Tensor,
    topk_ids_slots: torch.Tensor,
    topk_cs_index_slots: torch.Tensor,
    *,
    total_tokens: int,
    depth: int,
) -> bool:
    tensors = (sample_token, root_tokens, root_scores, score_slots, topk_ids_slots, topk_cs_index_slots)
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
        return False
    if int(depth) < 1 or int(depth) > 8 or int(total_tokens) < 1 or int(total_tokens) > 63:
        return False
    if int(root_tokens.numel()) != 4 or int(root_scores.numel()) != 4:
        return False
    if tuple(score_slots.shape[:3]) != (int(depth), 4, 4):
        return False
    if tuple(topk_ids_slots.shape[:3]) != (int(depth), 4, 4):
        return False
    if tuple(topk_cs_index_slots.shape[:2]) != (int(depth), 4):
        return False
    return (
        sample_token.dtype == torch.long
        and root_tokens.dtype == torch.long
        and topk_ids_slots.dtype == torch.long
        and topk_cs_index_slots.dtype == torch.long
        and root_scores.dtype == torch.float32
        and score_slots.dtype == torch.float32
    )


def _workspace(
    workspace: dict | None,
    *,
    device: torch.device,
    total_tokens: int,
    depth: int,
    max_depth: int,
) -> dict:
    n_nodes = int(total_tokens) + 1
    if (
        workspace is None
        or not isinstance(workspace, dict)
        or workspace.get("device") != device
        or int(workspace.get("total_tokens", -1)) != int(total_tokens)
        or int(workspace.get("depth", -1)) != int(depth)
        or int(workspace.get("max_depth", -1)) != int(max_depth)
    ):
        workspace = {
            "device": device,
            "total_tokens": int(total_tokens),
            "depth": int(depth),
            "max_depth": int(max_depth),
            "sorted_index": torch.empty((total_tokens,), device=device, dtype=torch.long),
            "mask_index": torch.empty((total_tokens,), device=device, dtype=torch.long),
            "draft_tokens": torch.empty((1, n_nodes), device=device, dtype=torch.long),
            "metadata": None,
        }
    return workspace


def run_eagle3_draft_tree_select_metadata(
    sample_token: torch.Tensor,
    root_tokens: torch.Tensor,
    root_scores: torch.Tensor,
    score_slots: torch.Tensor,
    topk_ids_slots: torch.Tensor,
    topk_cs_index_slots: torch.Tensor,
    *,
    total_tokens: int,
    depth: int,
    max_depth: int,
    workspace: dict | None = None,
):
    if not _can_use(
        sample_token,
        root_tokens,
        root_scores,
        score_slots,
        topk_ids_slots,
        topk_cs_index_slots,
        total_tokens=total_tokens,
        depth=depth,
    ):
        return None, workspace

    total_tokens = int(total_tokens)
    depth = int(depth)
    max_depth = int(max_depth)
    cand = 4 + depth * 16
    workspace = _workspace(
        workspace,
        device=root_tokens.device,
        total_tokens=total_tokens,
        depth=depth,
        max_depth=max_depth,
    )
    _eagle3_tree_select_kernel[(1,)](
        sample_token.reshape(-1),
        root_tokens.reshape(-1),
        root_scores.reshape(-1),
        score_slots,
        topk_ids_slots,
        topk_cs_index_slots,
        workspace["sorted_index"],
        workspace["mask_index"],
        workspace["draft_tokens"],
        DEPTH=depth,
        TOTAL=total_tokens,
        CAND=cand,
        BLOCK_C=triton.next_power_of_2(cand),
        BLOCK_T=triton.next_power_of_2(total_tokens),
        num_warps=1,
        num_stages=2,
    )
    metadata, metadata_workspace = run_eagle3_tree_metadata(
        workspace["mask_index"],
        total_tokens=total_tokens,
        max_depth=max_depth,
        workspace=workspace.get("metadata"),
    )
    workspace["metadata"] = metadata_workspace
    if metadata is None:
        return None, workspace
    tree_mask, retrieve_indices, tree_position_ids, _leaf_count, _actual_depth = metadata
    return (
        workspace["draft_tokens"],
        retrieve_indices,
        tree_mask,
        tree_position_ids,
    ), workspace
