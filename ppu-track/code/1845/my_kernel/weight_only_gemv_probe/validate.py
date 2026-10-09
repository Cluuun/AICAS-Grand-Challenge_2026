from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import acext
except Exception as exc:  # pragma: no cover
    raise RuntimeError("acext Python package is required for SDK packing") from exc

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import weight_only_gemv_probe_ext  # noqa: E402


def _parse_shapes(value: str) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for item in value.split(","):
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


def _quantize_groupwise_symmetric(
    w_nk: torch.Tensor,
    group_size: int,
    scale_dtype: torch.dtype,
    pack_mode: str,
    zero_mode: str,
    orientation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = w_nk.shape
    if k % group_size != 0:
        raise ValueError("K must be divisible by group_size")
    groups = k // group_size
    w_gsn = w_nk.t().contiguous().view(groups, group_size, n)
    scales = (w_gsn.float().abs().amax(dim=1) / 7.0).clamp_min(1.0e-6)
    q = torch.round(w_gsn.float() / scales[:, None, :]).clamp(-8, 7).to(torch.int8)
    q_kn = q.view(k, n).contiguous()
    if orientation == "kn":
        q_for_pack = q_kn
    elif orientation == "nk":
        q_for_pack = q_kn.t().contiguous()
    else:
        raise ValueError(f"unknown orientation={orientation!r}")
    q_packed = acext.pack_int4s(q_for_pack.cpu()).contiguous()
    preprocess_modes = {
        "preprocess": (False, True),
        "preprocess01": (False, True),
        "preprocess00": (False, False),
        "preprocess10": (True, False),
        "preprocess11": (True, True),
    }
    if pack_mode in preprocess_modes:
        flag0, flag1 = preprocess_modes[pack_mode]
        qweight = acext.preprocess_weights_for_mixed_gemm(
            q_packed, torch.quint4x2, flag0, flag1
        ).contiguous().cuda()
    elif pack_mode == "raw":
        qweight = q_packed.contiguous().cuda()
    else:
        raise ValueError(f"unknown pack_mode={pack_mode!r}")
    scales_d = scales.to(device=w_nk.device, dtype=scale_dtype).contiguous()
    if zero_mode == "zero":
        zeros_d = torch.zeros_like(scales_d).contiguous()
    elif zero_mode == "minus8_scale":
        zeros_d = (-8.0 * scales).to(device=w_nk.device, dtype=scale_dtype).contiguous()
    elif zero_mode == "plus8_scale":
        zeros_d = (8.0 * scales).to(device=w_nk.device, dtype=scale_dtype).contiguous()
    else:
        raise ValueError(f"unknown zero_mode={zero_mode!r}")
    w_ref = (q.float() * scales[:, None, :]).view(k, n).t().to(w_nk.dtype).contiguous()
    return qweight, scales_d, zeros_d, w_ref


def _pack_custom_int4_kmajor(q_kn: torch.Tensor) -> torch.Tensor:
    if q_kn.dtype != torch.int8:
        raise TypeError("q_kn must be int8")
    if q_kn.ndim != 2 or q_kn.shape[1] % 2 != 0:
        raise ValueError("q_kn must be [K,N] with even N")
    q = q_kn.to(torch.int16)
    if int(q.min()) < -8 or int(q.max()) > 7:
        raise ValueError("q_kn values must be signed int4 [-8,7]")
    u = torch.where(q < 0, q + 16, q).to(torch.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return packed


def _quantize_custom_symmetric(
    w_nk: torch.Tensor,
    group_size: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = w_nk.shape
    if k % group_size != 0:
        raise ValueError("K must be divisible by group_size")
    groups = k // group_size
    w_gsn = w_nk.t().contiguous().view(groups, group_size, n)
    scales = (w_gsn.float().abs().amax(dim=1) / 7.0).clamp_min(1.0e-6)
    q = torch.round(w_gsn.float() / scales[:, None, :]).clamp(-8, 7).to(torch.int8)
    q_kn = q.view(k, n).contiguous()
    packed = _pack_custom_int4_kmajor(q_kn.cpu()).cuda()
    scales_d = scales.to(device=w_nk.device, dtype=scale_dtype).contiguous()
    zeros_d = torch.zeros_like(scales_d).contiguous()
    w_ref = (q.float() * scales[:, None, :]).view(k, n).t().to(w_nk.dtype).contiguous()
    return packed, scales_d, zeros_d, w_ref


def _assert_close(label: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    torch.cuda.synchronize()
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    print(f"{label}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    if max_abs > tol:
        idx = int(diff.reshape(-1).argmax().item()) if diff.numel() else -1
        raise RuntimeError(
            f"{label} mismatch max_abs={max_abs:.6g} tol={tol:.6g} "
            f"idx={idx} got={got.reshape(-1)[idx].item() if idx >= 0 else None} "
            f"ref={ref.reshape(-1)[idx].item() if idx >= 0 else None}"
        )


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


def _run_constant_diagnosis(args: argparse.Namespace) -> None:
    device = torch.device("cuda", int(args.device))
    m, k, n = _parse_shapes(args.shapes)[0]
    scale_dtype = torch.bfloat16 if args.scale_dtype == "bf16" else torch.float16
    x = torch.ones((m, k), device=device, dtype=torch.bfloat16)
    scales = torch.ones((k // int(args.group_size), n), device=device, dtype=scale_dtype)
    zero_values = {
        "zero": 0.0,
        "minus8_scale": -8.0,
        "plus8_scale": 8.0,
    }
    for pack_mode in args.pack_modes.split(","):
        pack_mode = pack_mode.strip()
        if not pack_mode:
            continue
        for q_value in (-8, -1, 0, 1, 7):
            q_kn = torch.full((k, n), q_value, dtype=torch.int8)
            q_packed = acext.pack_int4s(q_kn).contiguous()
            preprocess_modes = {
                "preprocess": (False, True),
                "preprocess01": (False, True),
                "preprocess00": (False, False),
                "preprocess10": (True, False),
                "preprocess11": (True, True),
            }
            if pack_mode in preprocess_modes:
                flag0, flag1 = preprocess_modes[pack_mode]
                qweight = acext.preprocess_weights_for_mixed_gemm(
                    q_packed, torch.quint4x2, flag0, flag1
                ).contiguous().cuda()
            elif pack_mode == "raw":
                qweight = q_packed.contiguous().cuda()
            else:
                continue
            for zero_mode in args.zero_modes.split(","):
                zero_mode = zero_mode.strip()
                if zero_mode not in zero_values:
                    continue
                zeros = torch.full_like(scales, zero_values[zero_mode])
                out = weight_only_gemv_probe_ext.linear_groupwise(
                    x, qweight, scales, zeros, int(args.group_size), 1.0, False
                )
                torch.cuda.synchronize()
                print(
                    f"diag shape={m}x{k}x{n} scale={args.scale_dtype} "
                    f"pack={pack_mode} q={q_value} zero={zero_mode} "
                    f"mean={float(out.float().mean().item()):.6g} "
                    f"min={float(out.float().min().item()):.6g} "
                    f"max={float(out.float().max().item()):.6g}"
                )


def _run_onehot_diagnosis(args: argparse.Namespace) -> None:
    device = torch.device("cuda", int(args.device))
    m, k, n = _parse_shapes(args.shapes)[0]
    if m != 1:
        raise ValueError("onehot diagnosis expects M=1")
    scale_dtype = torch.bfloat16 if args.scale_dtype == "bf16" else torch.float16
    q_kn = torch.empty((k, n), dtype=torch.int8)
    if args.diagnose_pattern == "k":
        for kk in range(k):
            q_kn[kk, :] = (kk % 16) - 8
    elif args.diagnose_pattern == "n":
        for nn in range(n):
            q_kn[:, nn] = (nn % 16) - 8
    elif args.diagnose_pattern == "n_hi":
        for nn in range(n):
            q_kn[:, nn] = ((nn // 16) % 16) - 8
    else:
        raise ValueError(f"unknown diagnose pattern={args.diagnose_pattern!r}")
    q_packed = acext.pack_int4s(q_kn).contiguous()
    pack_mode = args.pack_modes.split(",")[0].strip()
    preprocess_modes = {
        "preprocess": (False, True),
        "preprocess01": (False, True),
        "preprocess00": (False, False),
        "preprocess10": (True, False),
        "preprocess11": (True, True),
    }
    if pack_mode in preprocess_modes:
        flag0, flag1 = preprocess_modes[pack_mode]
        qweight = acext.preprocess_weights_for_mixed_gemm(
            q_packed, torch.quint4x2, flag0, flag1
        ).contiguous().cuda()
    elif pack_mode == "raw":
        qweight = q_packed.contiguous().cuda()
    else:
        raise ValueError(f"unknown pack_mode={pack_mode!r}")
    scales = torch.ones((k // int(args.group_size), n), device=device, dtype=scale_dtype)
    zeros = torch.zeros_like(scales)
    limit = min(k, int(args.diagnose_limit))
    values = []
    x = torch.zeros((1, k), device=device, dtype=torch.bfloat16)
    for kk in range(limit if args.diagnose_pattern == "k" else 1):
        x.zero_()
        x[0, kk] = 1
        out = weight_only_gemv_probe_ext.linear_groupwise(
            x, qweight, scales, zeros, int(args.group_size), 1.0, False
        )
        torch.cuda.synchronize()
        if args.diagnose_pattern == "k":
            values.append(int(round(float(out.float().mean().item()))))
        else:
            values = [int(round(float(v))) for v in out.float().flatten()[:limit].tolist()]
    print(
        f"onehot shape={m}x{k}x{n} scale={args.scale_dtype} pack={pack_mode} "
        f"pattern={args.diagnose_pattern} values[:{limit}]={values}"
    )


def _run_case(m: int, k: int, n: int, args: argparse.Namespace) -> None:
    device = torch.device("cuda", int(args.device))
    torch.manual_seed(int(args.seed) + m * 1000003 + k * 9176 + n)
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.15
    scale_dtype = torch.bfloat16 if args.scale_dtype == "bf16" else torch.float16
    if args.backend == "custom":
        qweight, scales, zeros, w_ref = _quantize_custom_symmetric(
            w, int(args.group_size), scale_dtype
        )
        ref = torch.nn.functional.linear(x, w_ref, None)
        out = weight_only_gemv_probe_ext.custom_linear(
            x, qweight, scales, zeros, torch.empty((0,), device=device, dtype=torch.bfloat16),
            int(args.group_size), False, False
        )
        _assert_close(f"custom_gemv shape={m}x{k}x{n} scale={args.scale_dtype}", out, ref, float(args.tol))
        if int(args.split_k) > 1:
            split_out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            partial_check = torch.empty((int(args.split_k), n), device=device, dtype=torch.float32)
            empty_bias = torch.empty((0,), device=device, dtype=torch.bfloat16)
            weight_only_gemv_probe_ext.custom_linear_splitk_into(
                x, qweight, scales, zeros, empty_bias, partial_check, split_out,
                int(args.group_size), int(args.split_k), False, False
            )
            _assert_close(
                f"custom_gemv_splitk{int(args.split_k)} shape={m}x{k}x{n} scale={args.scale_dtype}",
                split_out,
                ref,
                float(args.tol),
            )
        if int(args.iters) > 0:
            out_buf = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            empty_bias = torch.empty((0,), device=device, dtype=torch.bfloat16)
            _bench(
                f"custom_gemv shape={m}x{k}x{n}",
                lambda: weight_only_gemv_probe_ext.custom_linear_into(
                    x, qweight, scales, zeros, empty_bias, out_buf, int(args.group_size), False, False
                ),
                int(args.warmup),
                int(args.iters),
            )
            if int(args.split_k) > 1:
                split_out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
                partial = torch.empty((int(args.split_k), n), device=device, dtype=torch.float32)
                _bench(
                    f"custom_gemv_splitk{int(args.split_k)} shape={m}x{k}x{n}",
                    lambda: weight_only_gemv_probe_ext.custom_linear_splitk_into(
                        x, qweight, scales, zeros, empty_bias, partial, split_out,
                        int(args.group_size), int(args.split_k), False, False
                    ),
                    int(args.warmup),
                    int(args.iters),
                )
            torch_out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            w_t = w_ref.t().contiguous()
            _bench(
                f"torch.mm_bf16 shape={m}x{k}x{n}",
                lambda: torch.mm(x, w_t, out=torch_out),
                int(args.warmup),
                int(args.iters),
            )
        return

    modes = []
    for pack_mode in args.pack_modes.split(","):
        pack_mode = pack_mode.strip()
        if not pack_mode:
            continue
        for zero_mode in args.zero_modes.split(","):
            zero_mode = zero_mode.strip()
            if zero_mode:
                modes.append((pack_mode, zero_mode))
    if not modes:
        raise ValueError("empty mode list")

    ref = None
    best = None
    selected = None
    for pack_mode, zero_mode in modes:
        for orientation in args.orientations.split(","):
            orientation = orientation.strip()
            if not orientation:
                continue
            qweight, scales, zeros, w_ref = _quantize_groupwise_symmetric(
                w, int(args.group_size), scale_dtype, pack_mode, zero_mode, orientation
            )
            if ref is None:
                ref = torch.nn.functional.linear(x, w_ref, None)
            for apply_alpha in (False, True) if args.try_alpha_modes else (False,):
                out = weight_only_gemv_probe_ext.linear_groupwise(
                    x, qweight, scales, zeros, int(args.group_size), 1.0, apply_alpha
                )
                torch.cuda.synchronize()
                diff = (out.float() - ref.float()).abs()
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
                mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
                label = (
                    f"public_gemv shape={m}x{k}x{n} scale={args.scale_dtype} "
                    f"pack={pack_mode} zero={zero_mode} orient={orientation} "
                    f"apply_alpha={apply_alpha}"
                )
                print(f"{label}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
                if best is None or mean_abs < best[0]:
                    best = (mean_abs, max_abs, label)
                if max_abs <= float(args.tol):
                    selected = (qweight, scales, zeros, label)
                    break
            if selected is not None:
                break
        if selected is not None:
            break
    if selected is None:
        assert best is not None
        raise RuntimeError(
            f"no public_gemv mode passed tol={float(args.tol):.6g}; "
            f"best mean_abs={best[0]:.6g} max_abs={best[1]:.6g} mode={best[2]}"
        )
    qweight, scales, zeros, selected_label = selected
    print(f"selected {selected_label}")

    if int(args.iters) > 0:
        out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        weight_only_gemv_probe_ext.linear_groupwise_into(
            x, qweight, scales, zeros, out, int(args.group_size), 1.0, False
        )
        _bench(
            f"public_gemv shape={m}x{k}x{n}",
            lambda: weight_only_gemv_probe_ext.linear_groupwise_into(
                x, qweight, scales, zeros, out, int(args.group_size), 1.0, False
            ),
            int(args.warmup),
            int(args.iters),
        )
        torch_out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        w_t = w_ref.t().contiguous()
        _bench(
            f"torch.mm_bf16 shape={m}x{k}x{n}",
            lambda: torch.mm(x, w_t, out=torch_out),
            int(args.warmup),
            int(args.iters),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="1x2048x4096,1x2048x10368,1x5184x2048,1x6144x2048",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--backend", choices=("public", "custom"), default="public")
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--pack-modes", default="preprocess,raw")
    parser.add_argument("--zero-modes", default="zero,minus8_scale,plus8_scale")
    parser.add_argument("--orientations", default="kn,nk")
    parser.add_argument("--try-alpha-modes", action="store_true")
    parser.add_argument("--diagnose-constant", action="store_true")
    parser.add_argument("--diagnose-onehot", action="store_true")
    parser.add_argument("--diagnose-limit", type=int, default=128)
    parser.add_argument("--diagnose-pattern", choices=("k", "n", "n_hi"), default="k")
    parser.add_argument("--tol", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.device))
    if args.diagnose_constant:
        _run_constant_diagnosis(args)
        return
    if args.diagnose_onehot:
        _run_onehot_diagnosis(args)
        return
    for m, k, n in _parse_shapes(args.shapes):
        _run_case(m, k, n, args)
    print("weight_only_gemv_probe validation passed")


if __name__ == "__main__":
    main()
