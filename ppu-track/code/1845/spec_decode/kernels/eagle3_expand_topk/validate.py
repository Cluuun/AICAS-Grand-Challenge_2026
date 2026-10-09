from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spec_decode.kernels.eagle3_expand_topk.runtime import run_eagle3_expand_topk4


def _reference(
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    scores: torch.Tensor,
    out_hidden: torch.Tensor,
    tree_mask: torch.Tensor,
):
    top_k = 4
    cu_scores = topk_p + scores[:, None]
    topk_cs = torch.topk(cu_scores.reshape(-1), top_k, dim=-1)
    topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
    out_ids = topk_cs_index // top_k
    input_hidden = out_hidden[:, out_ids]
    input_ids = topk_index.reshape(-1)[topk_cs_index][None]
    tree_mask_init = torch.eye(top_k, device=tree_mask.device, dtype=tree_mask.dtype)[None, None]
    next_tree_mask = torch.cat((tree_mask[:, :, out_ids], tree_mask_init), dim=3)
    return cu_scores, topk_cs_index, topk_cs_p, out_ids, input_hidden, input_ids, next_tree_mask


def _assert_equal(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    if got.is_floating_point():
        ok = torch.equal(got, ref)
        max_abs = float((got.float() - ref.float()).abs().max().item()) if got.numel() else 0.0
    else:
        ok = torch.equal(got, ref)
        max_abs = 0.0
    if not ok:
        raise RuntimeError(f"{name} mismatch: got={got.detach().cpu()} ref={ref.detach().cpu()} max_abs={max_abs}")


def _run_case(hidden_dim: int, hidden_dtype: torch.dtype) -> None:
    device = "cuda"
    torch.manual_seed(1000 + hidden_dim)
    workspace = None
    scores = torch.randn(4, device=device, dtype=torch.float32)
    tree_mask = torch.eye(4, device=device, dtype=torch.float32)[None, None]

    for step, mask_width in enumerate((4, 8, 12)):
        if int(tree_mask.shape[-1]) != mask_width:
            raise RuntimeError(f"unexpected tree_mask width before step {step}: {tuple(tree_mask.shape)}")
        topk_index = torch.randint(0, 32000, (4, 4), device=device, dtype=torch.long)
        topk_p = torch.randn(4, 4, device=device, dtype=torch.float32)
        topk_p = topk_p + torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4) * 1.0e-4
        out_hidden = torch.randn(1, 4, hidden_dim, device=device, dtype=hidden_dtype)

        result, workspace = run_eagle3_expand_topk4(
            topk_index,
            topk_p,
            scores,
            out_hidden,
            tree_mask,
            workspace=workspace,
            score_slot=step,
            score_slots=3,
        )
        if result is None:
            raise RuntimeError(f"expand_topk4 rejected hidden_dim={hidden_dim} dtype={hidden_dtype}")
        torch.cuda.synchronize()
        ref = _reference(topk_index, topk_p, scores, out_hidden, tree_mask)
        names = (
            "cu_scores",
            "topk_cs_index",
            "topk_cs_p",
            "out_ids",
            "input_hidden",
            "input_ids",
            "next_tree_mask",
        )
        for name, got, expected in zip(names, result, ref, strict=True):
            _assert_equal(name, got, expected)
        scores = result[2]
        tree_mask = result[6]

    print(f"expand_topk4 hidden_dim={hidden_dim} dtype={hidden_dtype} passed")


def main() -> None:
    try:
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda")
    except Exception as exc:
        raise RuntimeError("failed to allocate a CUDA/PPU tensor") from exc
    for hidden_dim in (1, 7, 128, 2048):
        for hidden_dtype in (torch.bfloat16, torch.float16, torch.float32):
            _run_case(hidden_dim, hidden_dtype)
    print("eagle3_expand_topk validation passed")


if __name__ == "__main__":
    main()
