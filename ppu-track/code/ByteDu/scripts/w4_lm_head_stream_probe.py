#!/usr/bin/env python3
"""Measure pure W4 lm_head-sized streaming bandwidth for staged cp.async variants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch


def load_ext(path: str):
    spec = importlib.util.spec_from_file_location("w4a16_probe_ext", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ext", default="prebuilt/w4a16_probe_ext.so")
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", default="reports/w4_lm_head_stream_probe.json")
    parser.add_argument("--blocks", default="128,256,512,768,1024,1536")
    parser.add_argument("--stages", default="1,2,4,8")
    args = parser.parse_args()

    if args.hidden % 8 != 0:
        raise ValueError("--hidden must be divisible by 8 for W4 int32 packing")
    torch.cuda.init()
    ext = load_ext(args.ext)
    device = torch.device("cuda")
    packed = torch.randint(
        low=-(2**31),
        high=2**31 - 1,
        size=(args.vocab, args.hidden // 8),
        device=device,
        dtype=torch.int32,
    )
    bytes_read = args.vocab * args.hidden // 2
    results = {
        "shape": {"vocab": args.vocab, "hidden": args.hidden, "packed_shape": list(packed.shape)},
        "bytes_read": bytes_read,
        "rows": [],
    }
    # Keep allocation outside timed region. One checksum slot per launched CTA.
    max_blocks = max(int(x) for x in args.blocks.split(",") if x)
    checksum = torch.empty((max_blocks,), device=device, dtype=torch.int32)
    blocks_list = [int(x) for x in args.blocks.split(",") if x]
    stages_list = [int(x) for x in args.stages.split(",") if x]
    for blocks in blocks_list:
        for stages in stages_list:
            fn = lambda b=blocks, s=stages: ext.w4_weight_stream_any_cp_async_probe(packed, checksum, b, s)
            us = bench(fn, args.warmup, args.iters)
            gbps = bytes_read / us / 1e3
            row = {"blocks": blocks, "stages": stages, "us": us, "effective_gbps": gbps}
            print(json.dumps(row), flush=True)
            results["rows"].append(row)
    results["best"] = max(results["rows"], key=lambda r: r["effective_gbps"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print("best:", json.dumps(results["best"]))


if __name__ == "__main__":
    main()
