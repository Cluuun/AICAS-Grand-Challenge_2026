#!/usr/bin/env python3
"""Microbenchmark FlashDecodeFFN v3 text prefill v2 kernels.

This bypasses the full benchmark stack and directly compares:
  FLASHDECODE_V2_PREFILL_FORCE_ALGO=fallback
  FLASHDECODE_V2_PREFILL_FORCE_ALGO=k8
  FLASHDECODE_V2_PREFILL_FORCE_ALGO=k16

Each algo runs in a separate subprocess so the extension's static algo cache
cannot leak between variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ALGOS = ("fallback", "k8", "k16")


def _load_ext():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import flashdecodeffn_v3_ext  # type: ignore

    return flashdecodeffn_v3_ext


def _cuda_timer(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def _run_one(args: argparse.Namespace) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    ext = _load_ext()

    n = int(args.n)
    h = int(args.h)
    i = int(args.i)
    x = torch.randn((n, h), device=device, dtype=dtype)
    w13 = torch.randn((h, 2 * i), device=device, dtype=dtype)
    w2 = torch.randn((i, h), device=device, dtype=dtype)

    def run():
        return ext.forward_text_v2(x, w13, w2)

    y = run()
    torch.cuda.synchronize()
    ref = ext.forward_packed(x, w13, w2)
    torch.cuda.synchronize()
    diff = (y.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1.0)
    rel = diff / denom

    ms = _cuda_timer(run, args.warmup, args.iters)
    return {
        "algo": os.environ.get("FLASHDECODE_V2_PREFILL_FORCE_ALGO", ""),
        "n": n,
        "h": h,
        "i": i,
        "ms": ms,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
        "out_norm": float(y.float().norm().item()),
    }


def _subprocess_one(args: argparse.Namespace, algo: str, n: int) -> dict:
    env = os.environ.copy()
    env["FLASHDECODE_V2_FORCE_ALGO"] = "fallback"
    env["FLASHDECODE_V2_PREFILL_FORCE_ALGO"] = algo
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--algo",
        algo,
        "--n",
        str(n),
        "--h",
        str(args.h),
        "--i",
        str(args.i),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--timeout",
        str(args.timeout),
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=float(args.timeout),
        )
    except subprocess.TimeoutExpired:
        return {
            "algo": algo,
            "n": n,
            "h": int(args.h),
            "i": int(args.i),
            "ms": 0.0,
            "error": f"timeout>{args.timeout}s",
        }
    if proc.returncode != 0:
        return {
            "algo": algo,
            "n": n,
            "h": int(args.h),
            "i": int(args.i),
            "ms": 0.0,
            "error": (proc.stderr or proc.stdout).strip()[-300:],
        }
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {
            "algo": algo,
            "n": n,
            "h": int(args.h),
            "i": int(args.i),
            "ms": 0.0,
            "error": proc.stdout.strip()[-300:],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=787)
    parser.add_argument(
        "--n-list",
        default="528,592,690,787,818,1045",
        help="Comma-separated prefill lengths to test in parent mode.",
    )
    parser.add_argument("--h", type=int, default=2048)
    parser.add_argument("--i", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--algo", choices=ALGOS, default="fallback")
    args = parser.parse_args()

    if args.child:
        print(json.dumps(_run_one(args), sort_keys=True))
        return 0

    n_values = []
    for part in str(args.n_list).split(","):
        part = part.strip()
        if part:
            n_values.append(int(part))
    if not n_values:
        n_values = [int(args.n)]

    rows = []
    for n in n_values:
        for algo in ALGOS:
            rows.append(_subprocess_one(args, algo, n))

    print("algo\tn\th\ti\tms\tspeedup_vs_fallback\tmax_abs\tmean_abs\tmax_rel\tmean_rel")
    for r in rows:
        base = next(
            b for b in rows
            if b["algo"] == "fallback" and b["n"] == r["n"] and b["h"] == r["h"] and b["i"] == r["i"]
        )
        if "error" in r:
            print(
                f"{r['algo']}\t{r['n']}\t{r['h']}\t{r['i']}\t"
                f"ERR\t0.000\t{r['error']}"
            )
            continue
        speedup = float(base["ms"]) / float(r["ms"]) if float(r["ms"]) > 0 else 0.0
        print(
            f"{r['algo']}\t{r['n']}\t{r['h']}\t{r['i']}\t"
            f"{r['ms']:.4f}\t{speedup:.3f}\t"
            f"{r['max_abs']:.6g}\t{r['mean_abs']:.6g}\t"
            f"{r['max_rel']:.6g}\t{r['mean_rel']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
