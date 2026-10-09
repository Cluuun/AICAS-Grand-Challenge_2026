#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from my_kernel.flashAttention_tilelang import flashattn_cuda


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def _run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    entries = {}
    reports = {}

    for q_len in args.lengths:
        q = torch.randn(1, 16, q_len, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, 8, q_len, 128, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, 8, q_len, 128, dtype=torch.bfloat16, device=device)
        k_gqa = k.repeat_interleave(2, dim=1)
        v_gqa = v.repeat_interleave(2, dim=1)

        torch_ms = _bench(
            lambda: flashattn_cuda._ORIG_SDPA(q, k_gqa, v_gqa, is_causal=True),
            warmup=args.warmup,
            iters=args.iters,
        )
        ref = flashattn_cuda._ORIG_SDPA(q, k_gqa, v_gqa, is_causal=True)

        best_name = "torch"
        best_ms = torch_ms
        candidates = []

        for spec in flashattn_cuda._TEXT_KERNEL_SPECS:
            if q_len > spec.cfg.seq_len:
                continue
            loaded = flashattn_cuda._load_kernel(spec)
            if loaded is None:
                continue

            out = flashattn_cuda._try_dispatch_to_kernel(loaded, q, k, v, True, None)
            if out is None:
                continue

            max_abs = float((out - ref).abs().max().item())
            tile_ms = _bench(
                lambda: flashattn_cuda._try_dispatch_to_kernel(loaded, q, k, v, True, None),
                warmup=args.warmup,
                iters=args.iters,
            )
            speedup = torch_ms / max(tile_ms, 1e-9)
            decision_ok = (speedup >= args.min_speedup) and (max_abs <= args.max_abs)
            candidates.append(
                {
                    "kernel": spec.name,
                    "ms": tile_ms,
                    "speedup": speedup,
                    "max_abs": max_abs,
                    "eligible": bool(decision_ok),
                }
            )
            if decision_ok and tile_ms < best_ms:
                best_ms = tile_ms
                best_name = f"tile:{spec.name}"

        key = f"1|1|16|8|128|{q_len}"
        entries[key] = best_name
        reports[key] = {
            "q_len": q_len,
            "torch_ms": torch_ms,
            "decision": best_name,
            "candidates": candidates,
        }

    output = {
        "version": 1,
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "profile_lengths": args.lengths,
            "shape": {
                "is_causal": True,
                "batch": 1,
                "heads_q": 16,
                "heads_kv": 8,
                "head_dim": 128,
            },
            "bench": {
                "warmup": args.warmup,
                "iters": args.iters,
                "min_speedup": args.min_speedup,
                "max_abs": args.max_abs,
                "seed": args.seed,
            },
        },
        "entries": entries,
        "benchmarks": reports,
    }

    out_path = Path(args.output).resolve()
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "dispatch_policy.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    parser.add_argument("--max-abs", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[594, 597, 640, 687, 689, 690, 691, 692, 695, 704, 751, 753, 768, 784, 788, 832, 1024, 1041, 1044, 1088],
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _run(_parse_args())
