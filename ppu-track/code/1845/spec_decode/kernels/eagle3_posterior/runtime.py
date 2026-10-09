from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _eagle3_posterior_kernel(
    target_tokens_ptr,
    draft_tokens_ptr,
    retrieve_ptr,
    path_targets_ptr,
    candidates_ptr,
    best_candidate_ptr,
    accept_length_ptr,
    next_token_ptr,
    CAND_COUNT: tl.constexpr,
    DEPTH: tl.constexpr,
    stride_r0: tl.constexpr,
    stride_r1: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_C)
    cols = tl.arange(0, BLOCK_D)
    row_in_bounds = rows < CAND_COUNT
    col_valid = cols < DEPTH

    idxs = tl.load(
        retrieve_ptr + rows[:, None] * stride_r0 + cols[None, :] * stride_r1,
        mask=row_in_bounds[:, None] & col_valid[None, :],
        other=-1,
    )
    valid = idxs >= 0
    row_has_root = tl.load(retrieve_ptr + rows * stride_r0, mask=row_in_bounds, other=-1) >= 0
    row_valid = row_in_bounds & row_has_root
    path_vals = tl.load(target_tokens_ptr + idxs, mask=row_valid[:, None] & col_valid[None, :] & valid, other=-1)
    cand_vals = tl.load(draft_tokens_ptr + idxs, mask=row_valid[:, None] & col_valid[None, :] & valid, other=-1)
    tl.store(path_targets_ptr + rows[:, None] * DEPTH + cols[None, :], path_vals, mask=row_in_bounds[:, None] & col_valid[None, :])
    tl.store(candidates_ptr + rows[:, None] * DEPTH + cols[None, :], cand_vals, mask=row_in_bounds[:, None] & col_valid[None, :])

    alive = row_valid
    accept = tl.full((BLOCK_C,), 0, dtype=tl.int32)
    for col in tl.static_range(0, DEPTH - 1):
        cur_cand = tl.load(candidates_ptr + rows * DEPTH + (col + 1), mask=row_valid, other=-2)
        cur_target = tl.load(path_targets_ptr + rows * DEPTH + col, mask=row_valid, other=-3)
        alive = alive & (cur_cand == cur_target)
        accept += tl.where(alive, 1, 0)

    tie_score = accept * 64 + (63 - rows)
    tie_score = tl.where(row_valid, tie_score, -1)
    best_score = tl.max(tie_score, axis=0)
    best_pos = tl.argmax(tie_score, axis=0)
    best_accept = best_score // 64

    next_idx = tl.load(
        retrieve_ptr + best_pos * stride_r0 + best_accept * stride_r1,
        mask=best_accept < DEPTH,
        other=-1,
    )
    next_token = tl.load(target_tokens_ptr + next_idx, mask=next_idx >= 0, other=-1)

    tl.store(best_candidate_ptr, best_pos)
    tl.store(accept_length_ptr, best_accept)
    tl.store(next_token_ptr, next_token)


def _can_use(target_tokens: torch.Tensor, draft_tokens: torch.Tensor, retrieve_indices: torch.Tensor) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (target_tokens, draft_tokens, retrieve_indices)):
        return False
    if not all(t.is_cuda for t in (target_tokens, draft_tokens, retrieve_indices)):
        return False
    if target_tokens.dtype != torch.long or draft_tokens.dtype != torch.long or retrieve_indices.dtype != torch.long:
        return False
    if retrieve_indices.ndim != 2:
        return False
    cand_count = int(retrieve_indices.shape[0])
    depth = int(retrieve_indices.shape[1])
    if cand_count < 1 or cand_count > 32 or depth < 2 or depth > 8:
        return False
    if target_tokens.numel() < 1 or draft_tokens.numel() < 1:
        return False
    return True


def _workspace(
    workspace: dict | None,
    *,
    device: torch.device,
    cand_count: int,
    depth: int,
) -> dict:
    if (
        workspace is None
        or not isinstance(workspace, dict)
        or workspace.get("device") != device
        or workspace.get("cand_count") != cand_count
        or workspace.get("depth") != depth
    ):
        workspace = {
            "device": device,
            "cand_count": cand_count,
            "depth": depth,
            "path_targets": torch.empty((cand_count, depth), device=device, dtype=torch.long),
            "candidates": torch.empty((cand_count, depth), device=device, dtype=torch.long),
            "best_candidate": torch.empty((), device=device, dtype=torch.long),
            "accept_length": torch.empty((), device=device, dtype=torch.long),
            "next_token": torch.empty((), device=device, dtype=torch.long),
        }
    return workspace


def run_eagle3_posterior(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    *,
    workspace: dict | None = None,
):
    if not _can_use(target_tokens, draft_tokens, retrieve_indices):
        return None, workspace

    target_flat = target_tokens.reshape(-1)
    draft_flat = draft_tokens.reshape(-1)
    cand_count = int(retrieve_indices.shape[0])
    depth = int(retrieve_indices.shape[1])
    workspace = _workspace(
        workspace,
        device=retrieve_indices.device,
        cand_count=cand_count,
        depth=depth,
    )
    block_c = triton.next_power_of_2(cand_count)
    block_d = triton.next_power_of_2(depth)
    _eagle3_posterior_kernel[(1,)](
        target_flat,
        draft_flat,
        retrieve_indices,
        workspace["path_targets"],
        workspace["candidates"],
        workspace["best_candidate"],
        workspace["accept_length"],
        workspace["next_token"],
        CAND_COUNT=cand_count,
        DEPTH=depth,
        stride_r0=int(retrieve_indices.stride(0)),
        stride_r1=int(retrieve_indices.stride(1)),
        BLOCK_C=block_c,
        BLOCK_D=block_d,
        num_warps=1,
        num_stages=2,
    )
    return (
        workspace["path_targets"],
        workspace["candidates"],
        workspace["best_candidate"],
        workspace["accept_length"],
        workspace["next_token"],
    ), workspace
