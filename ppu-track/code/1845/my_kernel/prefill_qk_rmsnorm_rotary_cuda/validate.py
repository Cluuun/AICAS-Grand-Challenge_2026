from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from my_kernel.prefill_qk_rmsnorm_rotary_cuda.runtime import run_prefill_qk_rmsnorm_rotary


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y.to(x.dtype) * weight).to(x.dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=787)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--k-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tol", type=float, default=1e-2)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn((1, args.seq_len, args.q_heads, args.head_dim), device=device, dtype=dtype)
    k = torch.randn((1, args.seq_len, args.k_heads, args.head_dim), device=device, dtype=dtype)
    q_w = torch.randn((args.head_dim,), device=device, dtype=dtype)
    k_w = torch.randn((args.head_dim,), device=device, dtype=dtype)
    cos = torch.randn((args.seq_len, args.head_dim), device=device, dtype=dtype)
    sin = torch.randn((args.seq_len, args.head_dim), device=device, dtype=dtype)
    eps = 1e-6

    got = run_prefill_qk_rmsnorm_rotary(q, k, cos, sin, q_w, k_w, eps)
    if got is None:
        raise RuntimeError("fused kernel unavailable")
    q_got, k_got = got

    qn = rmsnorm(q, q_w, eps).transpose(1, 2)
    kn = rmsnorm(k, k_w, eps).transpose(1, 2)
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    q_ref = (qn * c + rotate_half(qn) * s).to(dtype)
    k_ref = (kn * c + rotate_half(kn) * s).to(dtype)

    q_err = (q_got.float() - q_ref.float()).abs().max().item()
    k_err = (k_got.float() - k_ref.float()).abs().max().item()
    print(f"q_err={q_err:.6g} k_err={k_err:.6g}")
    if q_err > args.tol or k_err > args.tol:
        raise SystemExit(1)
    print("prefill qk rmsnorm rotary validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
