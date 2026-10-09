from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from my_kernel.prefill_rmsnorm_cuda.runtime import _rmsnorm_prefill_cuda, is_available  # noqa: E402


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"shape must be MxN, got {item!r}")
        m, n = item.lower().split("x", 1)
        out.append((int(m), int(n)))
    if not out:
        raise ValueError("empty shape list")
    return out


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + float(eps))
    return (xf * inv * weight.float()).to(torch.bfloat16)


def _guarded_view(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    guard_elems: int,
    fill_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    arena = torch.empty(int(guard_elems) + numel + int(guard_elems), device=device, dtype=dtype)
    arena[:guard_elems].fill_(fill_value)
    arena[guard_elems + numel :].fill_(fill_value)
    view = arena[guard_elems : guard_elems + numel].view(shape)
    return arena, view, arena.detach().clone()


def _check_guard(label: str, arena: torch.Tensor, snap: torch.Tensor, guard_elems: int) -> None:
    if int(guard_elems) <= 0:
        return
    torch.cuda.synchronize()
    if not torch.equal(arena[:guard_elems], snap[:guard_elems]):
        raise RuntimeError(f"{label} pre-guard corrupted")
    if not torch.equal(arena[-guard_elems:], snap[-guard_elems:]):
        raise RuntimeError(f"{label} post-guard corrupted")


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


def _run_case(m: int, n: int, args: argparse.Namespace) -> None:
    device = torch.device("cuda", int(args.device))
    eps = float(args.eps)
    guard = int(args.guard_elems)
    tol = float(args.tol)

    torch.manual_seed(int(args.seed) + int(m) * 1009 + int(n))
    x_arena, x, x_snap = _guarded_view(
        (m, n), device=device, dtype=torch.bfloat16, guard_elems=guard, fill_value=-123.0)
    out_arena, out, out_snap = _guarded_view(
        (m, n), device=device, dtype=torch.bfloat16, guard_elems=guard, fill_value=77.0)
    weight_arena, weight, weight_snap = _guarded_view(
        (n,), device=device, dtype=torch.bfloat16, guard_elems=guard, fill_value=19.0)

    x.normal_(mean=0.0, std=1.0)
    weight.normal_(mean=1.0, std=0.2)
    ref = _reference(x, weight, eps)

    got = _rmsnorm_prefill_cuda(x.view(1, m, n), weight, eps, out_buffer=out).view(m, n)
    _assert_close(f"shape={m}x{n} workspace", got, ref, tol)
    if got.data_ptr() != out.data_ptr():
        raise RuntimeError("workspace path did not return provided output buffer")

    got_alloc = _rmsnorm_prefill_cuda(x.view(1, m, n), weight, eps).view(m, n)
    _assert_close(f"shape={m}x{n} alloc", got_alloc, ref, tol)

    _check_guard("x", x_arena, x_snap, guard)
    _check_guard("out", out_arena, out_snap, guard)
    _check_guard("weight", weight_arena, weight_snap, guard)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="1x2048,2x2048,16x2048,128x2048,498x2048,700x2048,1045x2048,498x128,498x4096",
        help="Comma separated MxN shapes.",
    )
    parser.add_argument("--tol", type=float, default=0.03125)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--guard-elems", type=int, default=4096)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not is_available():
        raise RuntimeError("prefill_rmsnorm_cuda extension is not available")
    torch.cuda.set_device(int(args.device))

    for m, n in _parse_shapes(args.shapes):
        _run_case(m, n, args)
    print("prefill_rmsnorm_cuda validation passed")


if __name__ == "__main__":
    main()
