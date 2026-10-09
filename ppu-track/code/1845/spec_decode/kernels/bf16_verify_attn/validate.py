from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spec_decode.kernels.bf16_verify_attn.runtime import run_bf16_verify_attention


def _reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    q_len = int(query.shape[2])
    kv_len = int(key.shape[2])
    key_rep = key.repeat_interleave(2, dim=1)
    value_rep = value.repeat_interleave(2, dim=1)
    mask = torch.zeros(q_len, kv_len, device=query.device, dtype=torch.bool)
    prefix_len = kv_len - q_len
    for i in range(q_len):
        mask[i, : prefix_len + i + 1] = True
    ref = F.scaled_dot_product_attention(
        query,
        key_rep,
        value_rep,
        attn_mask=mask,
        dropout_p=0.0,
        scale=1.0 / math.sqrt(128.0),
    )
    return ref.transpose(1, 2).contiguous()


def main() -> None:
    torch.manual_seed(0)
    try:
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda")
    except Exception as exc:
        raise RuntimeError("failed to allocate a CUDA/PPU tensor") from exc
    cases = []
    for q_len in (1, 2, 3, 4, 8, 16):
        for kv_len in (q_len, 33, 257, 1024):
            cases.append((q_len, kv_len))

    for q_len, kv_len in cases:
        query = torch.randn(1, 16, q_len, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 8, kv_len, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(1, 8, kv_len, 128, device="cuda", dtype=torch.bfloat16)
        out = run_bf16_verify_attention(query, key, value)
        if out is None:
            raise RuntimeError(f"bf16 verifier backend rejected q_len={q_len}, kv_len={kv_len}")
        ref = _reference(query, key, value)
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        print(f"q={q_len} kv={kv_len} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")
        if max_abs > 0.03:
            raise RuntimeError(f"diff too high for q_len={q_len}, kv_len={kv_len}: {max_abs}")

    print("bf16_verify_attn validation passed")


if __name__ == "__main__":
    main()
