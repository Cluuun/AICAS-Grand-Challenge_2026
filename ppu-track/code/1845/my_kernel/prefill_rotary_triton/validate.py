from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_kernel.prefill_rotary_triton.runtime import rotary_qk_prefill_triton  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb  # noqa: E402


def _check_case(
    *,
    batch: int,
    q_heads: int,
    k_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    cos_3d: bool,
) -> None:
    device = "cuda"
    q = torch.randn(batch, q_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, k_heads, seq_len, head_dim, device=device, dtype=dtype)
    cos_ref = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype)
    sin_ref = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype)
    cos_kernel = cos_ref if cos_3d else cos_ref[0]
    sin_kernel = sin_ref if cos_3d else sin_ref[0]

    got = rotary_qk_prefill_triton(q, k, cos_kernel, sin_kernel)
    if got is None:
        raise RuntimeError(
            f"prefill rotary triton did not trigger: batch={batch} q_heads={q_heads} "
            f"k_heads={k_heads} seq={seq_len} dim={head_dim} cos_3d={cos_3d}"
        )
    ref = apply_rotary_pos_emb(q, k, cos_ref, sin_ref)
    torch.cuda.synchronize()

    for name, actual, expected in (("q", got[0], ref[0]), ("k", got[1], ref[1])):
        max_abs = (actual.float() - expected.float()).abs().max().item()
        max_ref = expected.float().abs().max().item()
        print(
            f"case batch={batch} qh={q_heads} kh={k_heads} seq={seq_len} dim={head_dim} "
            f"cos_3d={int(cos_3d)} {name}: max_abs={max_abs:.6g} max_ref={max_ref:.6g} "
            f"shape={tuple(actual.shape)} stride={actual.stride()}",
            flush=True,
        )
        if not torch.allclose(actual.float(), expected.float(), rtol=2e-2, atol=1.25e-1):
            raise RuntimeError(f"{name} mismatch: max_abs={max_abs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prefill rotary Triton kernel.")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ.setdefault("AICAS_PREFILL_ROTARY_TRITON", "1")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    cases = [
        (1, 16, 8, 17, 128, False),
        (1, 16, 8, 128, 128, False),
        (1, 16, 8, 513, 128, False),
        (1, 16, 8, 128, 128, True),
        (1, 16, 16, 128, 128, False),
    ]
    for batch, q_heads, k_heads, seq_len, head_dim, cos_3d in cases:
        _check_case(
            batch=batch,
            q_heads=q_heads,
            k_heads=k_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            cos_3d=cos_3d,
        )
    print("prefill_rotary_triton validation passed", flush=True)


if __name__ == "__main__":
    main()
