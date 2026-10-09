#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP, Qwen3VLVisionMLP

import ffn_cublaslt_ext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cuBLASLt FFN kernels")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def time_cuda(fn, warmup: int, iters: int) -> float:
    with torch.no_grad():
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
    return start.elapsed_time(end) / iters


def bench_text(tokens: int, warmup: int, iters: int) -> None:
    cfg = Qwen3VLTextConfig(hidden_size=2048, intermediate_size=6144, hidden_act="silu")
    mlp_torch = Qwen3VLTextMLP(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)

    w13 = ffn_cublaslt_ext.pack_text_w13(
        mlp_torch.gate_proj.weight.detach(), mlp_torch.up_proj.weight.detach()
    )
    w2 = ffn_cublaslt_ext.pack_text_w2(mlp_torch.down_proj.weight.detach())

    with torch.no_grad():
        y_torch = mlp_torch(x)
        y_ext = ffn_cublaslt_ext.text_forward(x, w13, w2)
    diff = (y_torch.float() - y_ext.float()).abs()

    torch_ms = time_cuda(lambda: mlp_torch(x), warmup=warmup, iters=iters)
    ext_ms = time_cuda(lambda: ffn_cublaslt_ext.text_forward(x, w13, w2), warmup=warmup, iters=iters)
    report = ffn_cublaslt_ext.get_last_report()
    print(f"[text] N={tokens},H=2048,I=6144 mean_err={diff.mean().item():.6e} max_err={diff.max().item():.6e}")
    print(f"[text] torch={torch_ms:.4f} ms cublaslt={ext_ms:.4f} ms speedup={(torch_ms / ext_ms):.2f}x")
    print(f"[text] {report}")


def bench_vision(warmup: int, iters: int) -> None:
    cfg = Qwen3VLVisionConfig(hidden_size=1024, intermediate_size=4096, hidden_act="gelu_pytorch_tanh")
    mlp_torch = Qwen3VLVisionMLP(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(2688, 1024, device="cuda", dtype=torch.bfloat16)

    w1 = ffn_cublaslt_ext.pack_vision_w1(mlp_torch.linear_fc1.weight.detach())
    w2 = ffn_cublaslt_ext.pack_vision_w2(mlp_torch.linear_fc2.weight.detach())
    b1 = mlp_torch.linear_fc1.bias.detach().contiguous()
    b2 = mlp_torch.linear_fc2.bias.detach().contiguous()

    with torch.no_grad():
        y_torch = mlp_torch(x)
        y_ext = ffn_cublaslt_ext.vision_forward(x, w1, b1, w2, b2)
    diff = (y_torch.float() - y_ext.float()).abs()

    torch_ms = time_cuda(lambda: mlp_torch(x), warmup=warmup, iters=iters)
    ext_ms = time_cuda(
        lambda: ffn_cublaslt_ext.vision_forward(x, w1, b1, w2, b2),
        warmup=warmup,
        iters=iters,
    )
    report = ffn_cublaslt_ext.get_last_report()
    print(f"[vision] N=2688,H=1024,I=4096 mean_err={diff.mean().item():.6e} max_err={diff.max().item():.6e}")
    print(f"[vision] torch={torch_ms:.4f} ms cublaslt={ext_ms:.4f} ms speedup={(torch_ms / ext_ms):.2f}x")
    print(f"[vision] {report}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)
    bench_text(tokens=1, warmup=args.warmup, iters=args.iters)
    bench_text(tokens=690, warmup=args.warmup, iters=args.iters)
    bench_vision(warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()

