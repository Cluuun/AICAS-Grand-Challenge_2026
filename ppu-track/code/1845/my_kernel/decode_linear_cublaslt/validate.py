from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from my_kernel.decode_linear_cublaslt.runtime import (  # noqa: E402
    can_use_decode_linear_cublaslt,
    is_available,
    linear_decode_cublaslt,
)


def _parse_shapes(value: str) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        parts = item.split("x")
        if len(parts) != 3:
            raise ValueError(f"shape must be MxKxN, got {item!r}")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not out:
        raise ValueError("empty shape list")
    return out


def _assert_close(label: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    torch.cuda.synchronize()
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    print(f"{label}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    if max_abs > float(tol):
        idx = int(diff.reshape(-1).argmax().item()) if diff.numel() else -1
        raise RuntimeError(
            f"{label} mismatch max_abs={max_abs:.6g} tol={float(tol):.6g} "
            f"idx={idx} got={got.reshape(-1)[idx].item() if idx >= 0 else None} "
            f"ref={ref.reshape(-1)[idx].item() if idx >= 0 else None}"
        )


def _bench(label: str, fn, *, warmup: int, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(int(iters)):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = float(start.elapsed_time(end)) / max(1, int(iters))
    print(f"{label}: {ms * 1000.0:.3f} us")
    return ms


def _run_case(m: int, k: int, n: int, args: argparse.Namespace) -> None:
    device = torch.device("cuda", int(args.device))
    torch.manual_seed(int(args.seed) + m * 1000003 + k * 9176 + n)
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    w_t = w.transpose(0, 1).contiguous()
    bias = torch.randn((n,), device=device, dtype=torch.bfloat16) if args.bias else None
    if not can_use_decode_linear_cublaslt(x, w_t, bias):
        raise RuntimeError(f"can_use rejected shape M={m} K={k} N={n} bias={args.bias}")

    got = linear_decode_cublaslt(x, w_t, bias)
    ref = torch.nn.functional.linear(x, w, bias)
    _assert_close(f"shape={m}x{k}x{n} bias={args.bias}", got, ref, float(args.tol))

    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    got_out = linear_decode_cublaslt(x, w_t, bias, out=out)
    if got_out.data_ptr() != out.data_ptr():
        raise RuntimeError("linear_decode_cublaslt did not use provided output buffer")
    _assert_close(f"shape={m}x{k}x{n} out bias={args.bias}", got_out, ref, float(args.tol))

    if int(args.iters) > 0:
        _bench(
            f"decode_ext shape={m}x{k}x{n} bias={args.bias}",
            lambda: linear_decode_cublaslt(x, w_t, bias, out=out),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        mm_out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        _bench(
            f"torch.mm_out shape={m}x{k}x{n}",
            lambda: torch.mm(x, w_t, out=mm_out),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        if bias is None:
            zero = torch.zeros((m, n), device=device, dtype=torch.bfloat16)
            _bench(
                f"torch.addmm_zero_out shape={m}x{k}x{n}",
                lambda: torch.addmm(zero, x, w_t, beta=0.0, out=mm_out),
                warmup=int(args.warmup),
                iters=int(args.iters),
            )
        else:
            _bench(
                f"torch.addmm_bias_out shape={m}x{k}x{n}",
                lambda: torch.addmm(bias, x, w_t, out=mm_out),
                warmup=int(args.warmup),
                iters=int(args.iters),
            )
        _bench(
            f"torch.linear shape={m}x{k}x{n} bias={args.bias}",
            lambda: torch.nn.functional.linear(x, w, bias),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="1x2048x4096,1x2048x2048,4x2048x4096,11x2048x4096",
        help="Comma separated MxKxN shapes.",
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not is_available():
        raise RuntimeError("decode_linear_cublaslt extension is not available")
    torch.cuda.set_device(int(args.device))
    for m, k, n in _parse_shapes(args.shapes):
        _run_case(m, k, n, args)
    print("decode_linear_cublaslt validation passed")


if __name__ == "__main__":
    main()
