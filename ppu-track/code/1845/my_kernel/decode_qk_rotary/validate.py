from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent


def _load_ext():
    cands = sorted(_THIS_DIR.glob("decode_qk_rotary_ext*.so"))
    if not cands:
        raise FileNotFoundError(f"Missing decode_qk_rotary extension in {_THIS_DIR}")
    name = "decode_qk_rotary_ext"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(cands[-1]))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load extension spec from {cands[-1]}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _bench(label: str, fn, warmup: int, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    us = float(start.elapsed_time(end)) * 1000.0 / max(1, iters)
    print(f"{label}: {us:.3f} us")
    return us


def _assert_close(label: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    torch.cuda.synchronize()
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    print(f"{label}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    if max_abs > tol:
        raise RuntimeError(f"{label} mismatch max_abs={max_abs:.6g} tol={tol:.6g}")


def run_case(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    ext = _load_ext()

    q_len = int(args.q_len)
    q_heads = int(args.q_heads)
    k_heads = int(args.k_heads)
    dim = 128
    torch.manual_seed(int(args.seed) + q_len * 1009 + q_heads * 17 + k_heads)
    q = torch.randn((q_len * q_heads, dim), device=device, dtype=dtype)
    k = torch.randn((q_len * k_heads, dim), device=device, dtype=dtype)
    cos = torch.randn((q_len, dim), device=device, dtype=dtype)
    sin = torch.randn((q_len, dim), device=device, dtype=dtype)
    q_weight = torch.randn((dim,), device=device, dtype=dtype)
    k_weight = torch.randn((dim,), device=device, dtype=dtype)
    eps = float(args.eps)

    ref_q, ref_k = ext.forward(q, k, cos, sin, q_weight, k_weight, eps, q_heads, k_heads)
    _assert_close(f"decode_qk_rotary_baseline_self q_len={q_len} dtype={args.dtype} q", ref_q, ref_q, 0.0)
    _assert_close(f"decode_qk_rotary_baseline_self q_len={q_len} dtype={args.dtype} k", ref_k, ref_k, 0.0)

    if int(args.iters) > 0:
        _bench(
            f"decode_qk_rotary_baseline q_len={q_len} dtype={args.dtype}",
            lambda: ext.forward(q, k, cos, sin, q_weight, k_weight, eps, q_heads, k_heads),
            int(args.warmup),
            int(args.iters),
        )

    if _env_flag("AICAS_EXPERIMENT_WARP_ROPE"):
        if not hasattr(ext, "forward_warp_rope"):
            raise RuntimeError("extension has no forward_warp_rope entry")
        got_q, got_k = ext.forward_warp_rope(q, k, cos, sin, q_weight, k_weight, eps, q_heads, k_heads)
        _assert_close(f"decode_qk_rotary_warp_rope q_len={q_len} dtype={args.dtype} q", got_q, ref_q, float(args.tol))
        _assert_close(f"decode_qk_rotary_warp_rope q_len={q_len} dtype={args.dtype} k", got_k, ref_k, float(args.tol))
        if int(args.iters) > 0:
            _bench(
                f"decode_qk_rotary_warp_rope q_len={q_len} dtype={args.dtype}",
                lambda: ext.forward_warp_rope(q, k, cos, sin, q_weight, k_weight, eps, q_heads, k_heads),
                int(args.warmup),
                int(args.iters),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--k-heads", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()
    run_case(args)
    print("decode_qk_rotary validation passed")


if __name__ == "__main__":
    main()
