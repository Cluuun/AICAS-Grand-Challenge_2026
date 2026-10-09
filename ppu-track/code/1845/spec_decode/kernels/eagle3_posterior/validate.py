from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spec_decode.kernels.eagle3_posterior.runtime import run_eagle3_posterior


def _reference(target_tokens: torch.Tensor, draft_tokens: torch.Tensor, retrieve_indices: torch.Tensor):
    target_ext = torch.cat(
        (
            target_tokens.reshape(-1).to(device=retrieve_indices.device, dtype=torch.long),
            torch.full((1,), -1, device=retrieve_indices.device, dtype=torch.long),
        ),
        dim=0,
    )
    path_target_tokens = target_ext[retrieve_indices]
    draft_tokens_pad = torch.cat(
        (
            draft_tokens.to(device=retrieve_indices.device, dtype=torch.long),
            torch.full((1, 1), -1, device=retrieve_indices.device, dtype=torch.long),
        ),
        dim=1,
    )
    candidates = draft_tokens_pad[0, retrieve_indices]
    valid_rows = candidates[:, 0].ge(0) & path_target_tokens[:, 0].ge(0)
    posterior_mask = candidates[:, 1:].eq(path_target_tokens[:, :-1]).int()
    accept_lengths = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
    accept_lengths = torch.where(valid_rows, accept_lengths, torch.full_like(accept_lengths, -1))
    accept_length = accept_lengths.max()
    accept_length_i = max(0, int(accept_length.item()))
    if accept_length_i == 0:
        valid_indices = torch.nonzero(valid_rows, as_tuple=False).reshape(-1)
        if int(valid_indices.numel()) > 0:
            best_candidate = valid_indices[0].to(torch.long)
        else:
            best_candidate = torch.tensor(0, dtype=torch.long, device=retrieve_indices.device)
    else:
        best_candidate = torch.argmax(accept_lengths).to(torch.long)
    next_token = path_target_tokens[best_candidate, accept_length_i].reshape(())
    return path_target_tokens, candidates, best_candidate, accept_length.to(torch.long), next_token


def _make_retrieve(cand_count: int, depth: int, token_count: int, device: str) -> torch.Tensor:
    rows = torch.full((cand_count, depth), -1, device=device, dtype=torch.long)
    for row in range(cand_count):
        length = 1 + (row % depth)
        start = (row * 3) % token_count
        vals = (torch.arange(length, device=device, dtype=torch.long) + start) % token_count
        rows[row, :length] = vals
    return rows


def _run_case(cand_count: int, depth: int) -> None:
    device = "cuda"
    token_count = 15
    torch.manual_seed(2000 + cand_count * 10 + depth)
    target_tokens = torch.randint(0, 5000, (token_count,), device=device, dtype=torch.long)
    draft_tokens = torch.randint(0, 5000, (1, token_count), device=device, dtype=torch.long)
    retrieve_indices = _make_retrieve(cand_count, depth, token_count, device)

    if cand_count > 2 and depth > 2:
        draft_tokens[0, retrieve_indices[1, 1]] = target_tokens[retrieve_indices[1, 0]]
        draft_tokens[0, retrieve_indices[2, 1]] = target_tokens[retrieve_indices[2, 0]]
        draft_tokens[0, retrieve_indices[2, 2]] = target_tokens[retrieve_indices[2, 1]]
    long_row = min(cand_count - 1, depth - 1)
    if cand_count > long_row and depth > 3:
        draft_tokens[0, retrieve_indices[long_row, 1]] = target_tokens[retrieve_indices[long_row, 0]]
        draft_tokens[0, retrieve_indices[long_row, 2]] = target_tokens[retrieve_indices[long_row, 1]]
        draft_tokens[0, retrieve_indices[long_row, 3]] = target_tokens[retrieve_indices[long_row, 2]]

    result, _workspace = run_eagle3_posterior(target_tokens, draft_tokens, retrieve_indices)
    if result is None:
        raise RuntimeError(f"posterior rejected cand_count={cand_count}, depth={depth}")
    torch.cuda.synchronize()
    ref = _reference(target_tokens, draft_tokens, retrieve_indices)
    names = ("path_target_tokens", "candidates", "best_candidate", "accept_length", "next_token")
    for name, got, expected in zip(names, result, ref, strict=True):
        if not torch.equal(got, expected):
            raise RuntimeError(f"{name} mismatch: got={got.detach().cpu()} ref={expected.detach().cpu()}")
    print(f"posterior cand_count={cand_count} depth={depth} passed")


def main() -> None:
    try:
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda")
    except Exception as exc:
        raise RuntimeError("failed to allocate a CUDA/PPU tensor") from exc
    for cand_count, depth in ((1, 2), (3, 3), (8, 4), (15, 5), (31, 8)):
        _run_case(cand_count, depth)
    retrieve = _make_retrieve(8, 4, 15, "cuda")
    retrieve[5:] = -1
    target = torch.arange(15, device="cuda", dtype=torch.long)
    draft = torch.arange(15, device="cuda", dtype=torch.long).reshape(1, 15)
    result, _workspace = run_eagle3_posterior(target, draft, retrieve)
    ref = _reference(target, draft, retrieve)
    for name, got, expected in zip(("path_target_tokens", "candidates", "best_candidate", "accept_length", "next_token"), result, ref, strict=True):
        if not torch.equal(got, expected):
            raise RuntimeError(f"fixed-shape {name} mismatch: got={got.detach().cpu()} ref={expected.detach().cpu()}")
    print("eagle3_posterior validation passed")


if __name__ == "__main__":
    main()
