#!/usr/bin/env python3
"""Blocking W4A16 feasibility probe for PPU-ZW810E/A100-compatible sm_80.

This intentionally benchmarks only down_proj shape n=2048,k=6144 before any
runtime routing is enabled. Passing requires the W4A16 probe to beat the Q8
control by the configured gate; otherwise the INT4 rollout must stop.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import operator
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
HIDDEN = 2048
INTERMEDIATE = 6144
GROUP = 128
GROUPS = INTERMEDIATE // GROUP
PACKED_K4 = INTERMEDIATE // 8
PACKED_K8 = INTERMEDIATE // 4


def load_ext(name: str, so_path: Path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension spec: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, *, warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end) * 1000.0 / iters)


def gbps(num_bytes: int, us: float) -> float:
    return (num_bytes / 1e9) / (us / 1e6)


def pack_int4(w_int4: torch.Tensor) -> torch.Tensor:
    reshaped = (w_int4 + 8).view(HIDDEN, PACKED_K4, 8)
    nibbles = (reshaped & 0xF).to(torch.int64)
    shifts = torch.arange(8, device=w_int4.device, dtype=torch.int64) * 4
    return (nibbles << shifts).sum(dim=2).to(torch.int32)


def pack_int8(w_int8: torch.Tensor) -> torch.Tensor:
    reshaped = (w_int8.to(torch.int32) + 128).view(HIDDEN, PACKED_K8, 4)
    return (
        reshaped[:, :, 0]
        | (reshaped[:, :, 1] << 8)
        | (reshaped[:, :, 2] << 16)
        | (reshaped[:, :, 3] << 24)
    ).to(torch.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warps", type=int, default=16)
    parser.add_argument("--q8-warps", type=int, default=None)
    parser.add_argument("--load-blocks", type=int, default=128)
    parser.add_argument("--gate", type=float, default=1.6)
    parser.add_argument("--json", type=Path, default=ROOT / "reports" / "w4a16_hardware_probe.json")
    args = parser.parse_args()
    q8_warps = args.warps if args.q8_warps is None else args.q8_warps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for W4A16 hardware probe")

    probe = load_ext("w4a16_probe_ext", ROOT / "prebuilt" / "w4a16_probe_ext.so")
    q8 = load_ext("native_q8_mlp_v2_ext", ROOT / "prebuilt" / "native_q8_mlp_v2_ext.so")

    dev = torch.device("cuda")
    torch.manual_seed(0)

    instruction_out = torch.empty(2, device=dev, dtype=torch.int32)
    probe.instruction_probe(instruction_out)
    torch.cuda.synchronize()

    w4_int = torch.randint(-8, 8, (HIDDEN, INTERMEDIATE), device=dev, dtype=torch.int32)
    w4_scale = (torch.rand(HIDDEN, GROUPS, device=dev, dtype=torch.float32) * 0.002 + 0.001).to(torch.bfloat16)
    w4_packed = pack_int4(w4_int)
    inp = (torch.randn(INTERMEDIATE, device=dev, dtype=torch.bfloat16) * 0.3).contiguous()
    residual = (torch.randn(HIDDEN, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()
    out_w4 = torch.empty(HIDDEN, device=dev, dtype=torch.bfloat16)
    sumsq_w4 = torch.empty(1, device=dev, dtype=torch.float32)
    out_w4_half2 = torch.empty(HIDDEN, device=dev, dtype=torch.bfloat16)
    sumsq_w4_half2 = torch.empty(1, device=dev, dtype=torch.float32)
    load_checksum = torch.empty(args.load_blocks, device=dev, dtype=torch.int32)

    # Q8 control uses rowwise scale, matching the current decode down control.
    w8_int = torch.randint(-128, 128, (HIDDEN, INTERMEDIATE), device=dev, dtype=torch.int32)
    w8_scale = (torch.rand(HIDDEN, device=dev, dtype=torch.float32) * 0.002 + 0.001).to(torch.bfloat16)
    w8_packed = pack_int8(w8_int)
    out_q8 = torch.empty(HIDDEN, device=dev, dtype=torch.bfloat16)
    sumsq_q8 = torch.empty(1, device=dev, dtype=torch.float32)

    def w4_fn() -> None:
        probe.down_w4a16_group128_probe(w4_packed, w4_scale, inp, residual, out_w4, sumsq_w4, args.warps)

    def w4_half2_fn() -> None:
        probe.down_w4a16_group128_half2_probe(w4_packed, w4_scale, inp, residual, out_w4_half2, sumsq_w4_half2, args.warps)

    def q8_fn() -> None:
        q8.down_add_sumsq_q8_v4(w8_packed, w8_scale, inp, residual, out_q8, sumsq_q8, q8_warps)

    def load_fn() -> None:
        probe.w4_weight_stream_cp_async_probe(w4_packed, load_checksum, args.load_blocks)

    w4_fn()
    w4_half2_fn()
    q8_fn()
    load_fn()
    torch.cuda.synchronize()

    scale_f = w4_scale.float().repeat_interleave(GROUP, dim=1)
    ref = ((w4_int.float() * scale_f) @ inp.float() + residual.float()).to(torch.bfloat16)
    diff = (ref.float() - out_w4.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(ref.float(), out_w4.float(), dim=0).item()
    diff_half2 = (ref.float() - out_w4_half2.float()).abs()
    cosine_half2 = torch.nn.functional.cosine_similarity(ref.float(), out_w4_half2.float(), dim=0).item()

    w4_us = bench(w4_fn, warmup=args.warmup, iters=args.iters)
    w4_half2_us = bench(w4_half2_fn, warmup=args.warmup, iters=args.iters)
    q8_us = bench(q8_fn, warmup=args.warmup, iters=args.iters)
    load_us = bench(load_fn, warmup=args.warmup, iters=args.iters)
    w4_bytes = HIDDEN * INTERMEDIATE // 2 + HIDDEN * GROUPS * 2 + INTERMEDIATE * 2 + 2 * HIDDEN * 2 + 4
    load_bytes = HIDDEN * INTERMEDIATE // 2
    q8_bytes = HIDDEN * INTERMEDIATE + HIDDEN * 2 + INTERMEDIATE * 2 + 2 * HIDDEN * 2 + 4
    w4_gbps = gbps(w4_bytes, w4_us)
    load_gbps = gbps(load_bytes, load_us)
    q8_gbps = gbps(q8_bytes, q8_us)
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "instruction_probe_int32": [int(x) for x in instruction_out.cpu().tolist()],
        "shape": {"n": HIDDEN, "k": INTERMEDIATE, "group": GROUP},
        "warps": args.warps,
        "q8_control_warps": q8_warps,
        "w4_us": w4_us,
        "w4_half2_magic_us": w4_half2_us,
        "w4_weight_load_cp_async_us": load_us,
        "q8_control_us": q8_us,
        "w4_bytes": w4_bytes,
        "w4_weight_load_bytes": load_bytes,
        "q8_control_bytes": q8_bytes,
        "w4_effective_gbps": w4_gbps,
        "w4_half2_magic_effective_gbps": w4_gbps if w4_half2_us <= 0 else gbps(w4_bytes, w4_half2_us),
        "w4_weight_load_cp_async_gbps": load_gbps,
        "q8_control_effective_gbps": q8_gbps,
        "latency_speedup": q8_us / w4_us,
        "half2_latency_speedup": q8_us / w4_half2_us,
        "bandwidth_ratio": w4_gbps / q8_gbps,
        "half2_bandwidth_ratio": gbps(w4_bytes, w4_half2_us) / q8_gbps,
        "pass_gate": (q8_us / w4_us) >= args.gate,
        "half2_pass_gate": (q8_us / w4_half2_us) >= args.gate,
        "correctness": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "cosine": float(cosine),
            "sumsq_probe": float(sumsq_w4.item()),
            "sumsq_ref": float((ref.float() * ref.float()).sum().item()),
            "load_checksum_xor": int(functools.reduce(operator.xor, (int(x) for x in load_checksum.cpu().tolist()), 0)),
        },
        "half2_correctness": {
            "max_abs": float(diff_half2.max().item()),
            "mean_abs": float(diff_half2.mean().item()),
            "cosine": float(cosine_half2),
            "sumsq_probe": float(sumsq_w4_half2.item()),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    print(f"device={result['device']} capability={tuple(result['capability'])}")
    print(f"LOP3/PRMT probe words={result['instruction_probe_int32']}")
    print(
        f"W4A16 group128 down: {w4_us:.3f} us, {w4_gbps:.1f} GB/s, "
        f"correctness cos={cosine:.6f}, max_abs={diff.max().item():.5g}"
    )
    print(
        f"W4 half2 magic down: {w4_half2_us:.3f} us, {gbps(w4_bytes, w4_half2_us):.1f} GB/s, "
        f"correctness cos={cosine_half2:.6f}, max_abs={diff_half2.max().item():.5g}"
    )
    print(f"W4 cp.async load:    {load_us:.3f} us, {load_gbps:.1f} GB/s")
    print(f"Q8 control down:      {q8_us:.3f} us, {q8_gbps:.1f} GB/s")
    print(
        f"latency_speedup={q8_us / w4_us:.3f}x bandwidth_ratio={w4_gbps / q8_gbps:.3f}x "
        f"gate={args.gate:.2f} pass={result['pass_gate']}"
    )
    print(
        f"half2_latency_speedup={q8_us / w4_half2_us:.3f}x "
        f"half2_bandwidth_ratio={gbps(w4_bytes, w4_half2_us) / q8_gbps:.3f}x "
        f"gate={args.gate:.2f} pass={result['half2_pass_gate']}"
    )
    print(args.json)


if __name__ == "__main__":
    main()
