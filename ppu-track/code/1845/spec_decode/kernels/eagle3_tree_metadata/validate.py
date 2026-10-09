from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spec_decode.kernels.eagle3_tree_metadata.runtime import run_eagle3_tree_metadata


def _reference(mask_index: torch.Tensor, total_tokens: int):
    mask_index_list = [int(x) for x in mask_index.detach().cpu().tolist()]
    tree_mask_cpu = torch.eye(total_tokens + 1).bool()
    tree_mask_cpu[:, 0] = True
    for i in range(total_tokens):
        tree_mask_cpu[i + 1].add_(tree_mask_cpu[mask_index_list[i]])
    tree_position_ids = torch.sum(tree_mask_cpu, dim=1) - 1
    tree_mask = tree_mask_cpu.float()[None, None].to(mask_index.device)

    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num
    retrieve_indices = torch.zeros(leaf_num, int(max_depth.item()), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()
    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            node_depth = position_ids_list[i]
            for j in reversed(range(node_depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=mask_index.device)
    tree_position_ids = tree_position_ids.to(mask_index.device)
    return tree_mask, retrieve_indices, tree_position_ids


def _run_case(mask_values: list[int], max_depth: int) -> None:
    device = "cuda"
    mask_index = torch.tensor(mask_values, device=device, dtype=torch.long)
    total_tokens = len(mask_values)
    result, _workspace = run_eagle3_tree_metadata(
        mask_index,
        total_tokens=total_tokens,
        max_depth=max_depth,
    )
    if result is None:
        raise RuntimeError(f"tree_metadata rejected total_tokens={total_tokens}, max_depth={max_depth}")
    torch.cuda.synchronize()
    tree_mask, retrieve_full, tree_position_ids, leaf_count, actual_depth = result
    ref_mask, ref_retrieve, ref_pos = _reference(mask_index, total_tokens)
    if int(leaf_count.item()) != int(ref_retrieve.shape[0]):
        raise RuntimeError(f"leaf_count mismatch: got={int(leaf_count.item())} ref={int(ref_retrieve.shape[0])}")
    if int(actual_depth.item()) != int(ref_retrieve.shape[1]):
        raise RuntimeError(f"actual_depth mismatch: got={int(actual_depth.item())} ref={int(ref_retrieve.shape[1])}")
    retrieve = retrieve_full[: ref_retrieve.shape[0], : ref_retrieve.shape[1]]
    invalid = retrieve_full[ref_retrieve.shape[0] :]
    if invalid.numel() and not torch.equal(invalid, torch.full_like(invalid, -1)):
        raise RuntimeError(f"invalid retrieve rows are not padded: {invalid.detach().cpu()}")
    if not torch.equal(tree_mask, ref_mask):
        raise RuntimeError(f"tree_mask mismatch: got={tree_mask.detach().cpu()} ref={ref_mask.detach().cpu()}")
    if not torch.equal(tree_position_ids, ref_pos):
        raise RuntimeError(f"tree_position_ids mismatch: got={tree_position_ids.detach().cpu()} ref={ref_pos.detach().cpu()}")
    if not torch.equal(retrieve, ref_retrieve):
        raise RuntimeError(f"retrieve_indices mismatch: got={retrieve.detach().cpu()} ref={ref_retrieve.detach().cpu()}")
    print(f"tree_metadata total_tokens={total_tokens} max_depth={max_depth} passed")


def main() -> None:
    try:
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda")
    except Exception as exc:
        raise RuntimeError("failed to allocate a CUDA/PPU tensor") from exc

    _run_case([0, 0, 0, 0], 3)
    _run_case([0, 1, 2, 3, 4], 7)
    _run_case([0, 0, 1, 1, 2, 2, 3, 3], 5)
    _run_case([0, 0, 0, 1, 1, 5, 5, 2, 2, 9, 9, 3, 3, 13, 13], 6)
    print("eagle3_tree_metadata validation passed")


if __name__ == "__main__":
    main()
