from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _eagle3_tree_metadata_kernel(
    mask_index_ptr,
    tree_mask_ptr,
    tree_position_ids_ptr,
    retrieve_indices_ptr,
    leaf_count_ptr,
    actual_depth_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_RD: tl.constexpr,
):
    node_cols = tl.arange(0, BLOCK_N)
    parent_offs = tl.arange(0, BLOCK_N)
    parents = tl.load(mask_index_ptr + parent_offs, mask=parent_offs < N - 1, other=-1)
    parents = tl.where((parents >= 0) & (parents < N), parents, -1)

    rd_offs = tl.arange(0, BLOCK_RD)
    tl.store(retrieve_indices_ptr + rd_offs, -1, mask=rd_offs < N * D)

    leaf_rank = tl.full((), 0, dtype=tl.int32)
    max_depth_seen = tl.full((), 0, dtype=tl.int32)

    for node in tl.static_range(0, N):
        visible = node_cols == 0
        cur = tl.full((), node, dtype=tl.int32)
        depth = tl.full((), 0, dtype=tl.int32)

        for _step in tl.static_range(0, N):
            active = (cur > 0) & (cur < N)
            visible = visible | ((node_cols == cur) & active)
            depth += tl.where(active, 1, 0)
            parent = tl.load(mask_index_ptr + cur - 1, mask=active, other=0)
            parent = tl.where((parent >= 0) & (parent < N), parent, 0)
            cur = tl.where(active, parent, 0)

        tl.store(tree_mask_ptr + node * N + node_cols, visible.to(tl.float32), mask=node_cols < N)
        tl.store(tree_position_ids_ptr + node, depth)
        max_depth_seen = tl.maximum(max_depth_seen, depth)

        has_child = tl.max(tl.where(parents == node, 1, 0), axis=0) > 0
        is_leaf = ~has_child

        cid = tl.full((), node, dtype=tl.int32)
        for rev in tl.static_range(0, D):
            col = depth - rev
            in_path = is_leaf & (rev <= depth) & (col >= 0) & (col < D) & (leaf_rank < N)
            tl.store(retrieve_indices_ptr + leaf_rank * D + col, cid, mask=in_path)
            active = (cid > 0) & (cid < N)
            parent = tl.load(mask_index_ptr + cid - 1, mask=active, other=0)
            parent = tl.where((parent >= 0) & (parent < N), parent, 0)
            cid = tl.where(active, parent, 0)

        leaf_rank += tl.where(is_leaf, 1, 0)

    tl.store(leaf_count_ptr, leaf_rank)
    tl.store(actual_depth_ptr, max_depth_seen + 1)



def _can_use(mask_index: torch.Tensor, total_tokens: int, max_depth: int) -> bool:
    if not isinstance(mask_index, torch.Tensor):
        return False
    if not mask_index.is_cuda or mask_index.dtype != torch.long or mask_index.ndim != 1:
        return False
    total_tokens = int(total_tokens)
    max_depth = int(max_depth)
    if total_tokens < 1 or total_tokens > 63:
        return False
    if int(mask_index.numel()) != total_tokens:
        return False
    return 2 <= max_depth <= 16


def _workspace(
    workspace: dict | None,
    *,
    device: torch.device,
    total_tokens: int,
    max_depth: int,
) -> dict:
    n_nodes = int(total_tokens) + 1
    max_depth = int(max_depth)
    if (
        workspace is None
        or not isinstance(workspace, dict)
        or workspace.get("device") != device
        or workspace.get("n_nodes") != n_nodes
        or workspace.get("max_depth") != max_depth
    ):
        workspace = {
            "device": device,
            "n_nodes": n_nodes,
            "max_depth": max_depth,
            "tree_mask": torch.empty((1, 1, n_nodes, n_nodes), device=device, dtype=torch.float32),
            "tree_position_ids": torch.empty((n_nodes,), device=device, dtype=torch.long),
            "retrieve_indices": torch.empty((n_nodes, max_depth), device=device, dtype=torch.long),
            "leaf_count": torch.empty((), device=device, dtype=torch.long),
            "actual_depth": torch.empty((), device=device, dtype=torch.long),
        }
    return workspace


def run_eagle3_tree_metadata(
    mask_index: torch.Tensor,
    *,
    total_tokens: int,
    max_depth: int,
    workspace: dict | None = None,
):
    if not _can_use(mask_index, total_tokens, max_depth):
        return None, workspace

    total_tokens = int(total_tokens)
    max_depth = int(max_depth)
    n_nodes = total_tokens + 1
    workspace = _workspace(
        workspace,
        device=mask_index.device,
        total_tokens=total_tokens,
        max_depth=max_depth,
    )
    block_n = triton.next_power_of_2(n_nodes)
    block_rd = triton.next_power_of_2(n_nodes * max_depth)
    _eagle3_tree_metadata_kernel[(1,)](
        mask_index,
        workspace["tree_mask"],
        workspace["tree_position_ids"],
        workspace["retrieve_indices"],
        workspace["leaf_count"],
        workspace["actual_depth"],
        N=n_nodes,
        D=max_depth,
        BLOCK_N=block_n,
        BLOCK_RD=block_rd,
        num_warps=1,
        num_stages=2,
    )
    return (
        workspace["tree_mask"],
        workspace["retrieve_indices"],
        workspace["tree_position_ids"],
        workspace["leaf_count"],
        workspace["actual_depth"],
    ), workspace
