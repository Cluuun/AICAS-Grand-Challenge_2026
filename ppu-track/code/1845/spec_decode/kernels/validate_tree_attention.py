from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spec_decode.multitoken_attention import _triton_tree_spec_attn


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x[:, :, None, :, :].expand(-1, -1, groups, -1, -1).reshape(
        int(x.shape[0]),
        int(x.shape[1]) * groups,
        int(x.shape[2]),
        int(x.shape[3]),
    )


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    groups = int(q.shape[1]) // int(k.shape[1])
    kk = _repeat_kv(k, groups)
    vv = _repeat_kv(v, groups)
    scores = torch.matmul(q, kk.transpose(2, 3)) * float(1.0 / math.sqrt(128.0))
    scores = scores + mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, vv).transpose(1, 2).contiguous()


def _tree_mask(q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    prefix = kv_len - q_len
    mask = torch.empty((1, 1, q_len, kv_len), device=device, dtype=dtype)
    mask.fill_(torch.finfo(dtype).min)
    if prefix > 0:
        mask[:, :, :, :prefix].zero_()
    for q_idx in range(q_len):
        mask[:, :, q_idx, prefix].zero_()
        mask[:, :, q_idx, prefix + q_idx].zero_()
        if q_idx > 0:
            parent = 1 + (q_idx - 1) // 4
            mask[:, :, q_idx, prefix + parent].zero_()
    return mask


def _run_case(q_len: int, kv_len: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(1234 + q_len * 1000 + kv_len)
    q = torch.randn((1, 16, q_len, 128), device=device, dtype=dtype)
    k = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype)
    v = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype)
    mask = _tree_mask(q_len, kv_len, device, dtype)
    got = _triton_tree_spec_attn(q, k, v, mask)
    ref = _reference(q, k[:, :, : int(mask.shape[-1]), :], v[:, :, : int(mask.shape[-1]), :], mask)
    torch.cuda.synchronize()
    diff = (got.float() - ref.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    print(f"q={q_len} kv={kv_len} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 0.125:
        raise RuntimeError(f"tree attention diff too high: q={q_len} kv={kv_len} max={max_diff}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    for q_len, kv_len in ((1, 64), (4, 128), (15, 807), (15, 1280)):
        _run_case(q_len, kv_len)
    print("tree attention validation passed")


if __name__ == "__main__":
    main()
