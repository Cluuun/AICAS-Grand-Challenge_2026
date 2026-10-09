import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

from aicas2026gc.ops.fused_addmm_residual_gemv import (  # noqa: E402
    addmm_residual_warp_out,
    addmm_residual_wmma_out,
)
from aicas2026gc.ops.int8_linear_cublaslt import (  # noqa: E402
    _module as int8_module,
    prepack_weight,
    rowwise_quantize,
)


DEFAULT_WORKSPACE_MB = 32
DEFAULT_MAX_ALGOS = 8
LAYOUT_COLTRICK = 0


def _load_autotuner():
    try:
        from aicas2026gc.autotuner import RuntimeAutotuner

        return RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[1] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("int8_decode_linear_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RuntimeAutotuner


def _empty_bias(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((0,), device=device, dtype=dtype)


def _make_cases() -> dict[str, dict[str, object]]:
    return {
        "attn_qkv": {"k": 2048, "n": 4096, "bias": False, "residual": False},
        "attn_o": {"k": 2048, "n": 2048, "bias": False, "residual": True},
        "mlp_gate_up": {"k": 2048, "n": 12288, "bias": False, "residual": False},
        "mlp_down": {"k": 6144, "n": 2048, "bias": False, "residual": True},
    }


def _error_stats(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    diff = (out.float() - ref.float()).abs()
    ref_f = ref.float()
    denom = ref_f.abs().mean().clamp_min(1.0e-6)
    cos = F.cosine_similarity(out.float().flatten(), ref_f.flatten(), dim=0)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_mean": (diff.mean() / denom).item(),
        "cos": cos.item(),
    }


def _print_stats(name: str, ref: torch.Tensor, out: torch.Tensor) -> None:
    stats = _error_stats(ref, out)
    print(
        f"[error][{name}] "
        f"max_abs={stats['max_abs']:.6f}, "
        f"mean_abs={stats['mean_abs']:.6f}, "
        f"rel_mean={stats['rel_mean']:.6f}, "
        f"cos={stats['cos']:.8f}"
    )


def _heuristic_count(m: int, k: int, n: int, workspace_bytes: int) -> int:
    try:
        return int(int8_module.int8_heuristic_count(m, k, n, workspace_bytes, LAYOUT_COLTRICK))
    except RuntimeError as exc:
        print(f"[heuristic][M{m}_N{n}_K{k}] failed: {exc}", flush=True)
        return 0


def run_case(
    case_name: str,
    k: int,
    n: int,
    has_bias: bool,
    has_residual: bool,
    workspace_mb: int,
    max_algos: int,
    seed: int,
    pad_m: int,
) -> None:
    RuntimeAutotuner = _load_autotuner()

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.float16
    workspace_bytes = int(workspace_mb) * 1024 * 1024

    x = torch.randn((1, k), device=device, dtype=dtype).contiguous()
    weight = torch.randn((n, k), device=device, dtype=dtype).contiguous()
    weight_t = weight.t().contiguous()
    bias = torch.randn((n,), device=device, dtype=dtype) if has_bias else _empty_bias(device, dtype)
    residual = torch.randn((1, n), device=device, dtype=dtype).contiguous() if has_residual else None
    zero_residual = torch.zeros((1, n), device=device, dtype=dtype)

    w_q, w_scale = prepack_weight(weight)
    x_q_pre, x_scale_pre = rowwise_quantize(x)

    workspace = torch.empty((workspace_bytes,), device=device, dtype=torch.uint8)
    acc_m1 = torch.empty((1, n), device=device, dtype=torch.int32)
    out_m1 = torch.empty((1, n), device=device, dtype=dtype)
    x_q_m1 = torch.empty((1, k), device=device, dtype=torch.int8)
    x_scale_m1 = torch.empty((1,), device=device, dtype=torch.float32)
    torch_out = torch.empty((1, n), device=device, dtype=dtype)

    x_pad = torch.zeros((pad_m, k), device=device, dtype=dtype)
    x_pad[0].copy_(x[0])
    residual_pad = torch.zeros((pad_m, n), device=device, dtype=dtype)
    if residual is not None:
        residual_pad[0].copy_(residual[0])
    x_q_pad = torch.empty((pad_m, k), device=device, dtype=torch.int8)
    x_scale_pad = torch.empty((pad_m,), device=device, dtype=torch.float32)
    acc_pad = torch.empty((pad_m, n), device=device, dtype=torch.int32)
    out_pad = torch.empty((pad_m, n), device=device, dtype=dtype)

    fp16_warp_out = torch.empty((1, n), device=device, dtype=dtype)
    fp16_wmma_out = torch.empty((1, n), device=device, dtype=dtype)

    print(
        f"\n[INT8Decode][bench] case={case_name} M=1 K={k} N={n} "
        f"bias={int(has_bias)} residual={int(has_residual)} pad_m={pad_m} "
        f"workspace_mb={workspace_mb} max_algos={max_algos}",
        flush=True,
    )

    def torch_ref(x_arg: torch.Tensor) -> torch.Tensor:
        y = F.linear(x_arg, weight, None if not has_bias else bias)
        if residual is not None:
            y = y + residual
        return y

    def torch_mm_out(x_arg: torch.Tensor) -> torch.Tensor:
        if residual is not None:
            if has_bias:
                return torch.addmm(residual + bias, x_arg, weight_t, out=torch_out)
            return torch.addmm(residual, x_arg, weight_t, out=torch_out)
        if has_bias:
            return torch.addmm(bias, x_arg, weight_t, out=torch_out)
        return torch.mm(x_arg, weight_t, out=torch_out)

    gemv_residual = residual if residual is not None else zero_residual

    def fp16_warp(x_arg: torch.Tensor) -> torch.Tensor:
        return addmm_residual_warp_out(gemv_residual, x_arg, weight, fp16_warp_out)

    def fp16_wmma(x_arg: torch.Tensor) -> torch.Tensor:
        return addmm_residual_wmma_out(gemv_residual, x_arg, weight, fp16_wmma_out)

    candidates = {
        "torch_ref": torch_ref,
        "torch_mm_out": torch_mm_out,
        "fp16_warp_half2": fp16_warp,
        "fp16_wmma_m16": fp16_wmma,
    }

    count_m1 = min(_heuristic_count(1, k, n, workspace_bytes), max_algos)
    count_pad = min(_heuristic_count(pad_m, k, n, workspace_bytes), max_algos)
    print(f"[heuristic][{case_name}] m1={count_m1} pad{pad_m}={count_pad}", flush=True)

    for h in range(count_m1):
        if residual is None:
            candidates[f"i8_m1_pre_h{h}"] = (
                lambda x_arg, h=h: int8_module.int8_linear_prequant_out(
                    x_q_pre,
                    x_scale_pre,
                    w_q,
                    w_scale,
                    bias,
                    workspace,
                    acc_m1,
                    out_m1,
                    h,
                    LAYOUT_COLTRICK,
                )
            )
            candidates[f"i8_m1_dyn_h{h}"] = (
                lambda x_arg, h=h: int8_module.int8_linear_dynamic_out(
                    x_arg,
                    w_q,
                    w_scale,
                    bias,
                    workspace,
                    x_q_m1,
                    x_scale_m1,
                    acc_m1,
                    out_m1,
                    h,
                    LAYOUT_COLTRICK,
                )
            )
        else:
            candidates[f"i8_m1_pre_h{h}"] = (
                lambda x_arg, h=h: int8_module.int8_linear_prequant_residual_out(
                    x_q_pre,
                    x_scale_pre,
                    w_q,
                    w_scale,
                    bias,
                    residual,
                    workspace,
                    acc_m1,
                    out_m1,
                    h,
                    LAYOUT_COLTRICK,
                )
            )
            candidates[f"i8_m1_dyn_h{h}"] = (
                lambda x_arg, h=h: int8_module.int8_linear_dynamic_residual_out(
                    x_arg,
                    w_q,
                    w_scale,
                    bias,
                    residual,
                    workspace,
                    x_q_m1,
                    x_scale_m1,
                    acc_m1,
                    out_m1,
                    h,
                    LAYOUT_COLTRICK,
                )
            )

    for h in range(count_pad):
        if residual is None:
            candidates[f"i8_pad{pad_m}_static_h{h}"] = (
                lambda x_arg, h=h: int8_module.int8_linear_dynamic_out(
                    x_pad,
                    w_q,
                    w_scale,
                    bias,
                    workspace,
                    x_q_pad,
                    x_scale_pad,
                    acc_pad,
                    out_pad,
                    h,
                    LAYOUT_COLTRICK,
                )[:1]
            )

            def make_pad_copy_nores(algo_h: int):
                def impl(x_arg: torch.Tensor) -> torch.Tensor:
                    x_pad[0].copy_(x_arg[0])
                    return int8_module.int8_linear_dynamic_out(
                        x_pad,
                        w_q,
                        w_scale,
                        bias,
                        workspace,
                        x_q_pad,
                        x_scale_pad,
                        acc_pad,
                        out_pad,
                        algo_h,
                        LAYOUT_COLTRICK,
                    )[:1]

                return impl

            candidates[f"i8_pad{pad_m}_copy_h{h}"] = make_pad_copy_nores(h)
        else:
            candidates[f"i8_pad{pad_m}_static_h{h}"] = (
                lambda x_arg, h=h: int8_module.int8_linear_dynamic_residual_out(
                    x_pad,
                    w_q,
                    w_scale,
                    bias,
                    residual_pad,
                    workspace,
                    x_q_pad,
                    x_scale_pad,
                    acc_pad,
                    out_pad,
                    h,
                    LAYOUT_COLTRICK,
                )[:1]
            )

            def make_pad_copy_res(algo_h: int):
                def impl(x_arg: torch.Tensor) -> torch.Tensor:
                    x_pad[0].copy_(x_arg[0])
                    residual_pad[0].copy_(residual[0])
                    return int8_module.int8_linear_dynamic_residual_out(
                        x_pad,
                        w_q,
                        w_scale,
                        bias,
                        residual_pad,
                        workspace,
                        x_q_pad,
                        x_scale_pad,
                        acc_pad,
                        out_pad,
                        algo_h,
                        LAYOUT_COLTRICK,
                    )[:1]

                return impl

            candidates[f"i8_pad{pad_m}_copy_h{h}"] = make_pad_copy_res(h)

    tuner = RuntimeAutotuner(
        name=f"int8_decode_linear_{case_name}",
        fallback="torch_ref",
        check_correctness=False,
        strict=False,
        atol=1.0,
        rtol=1.0,
    )

    bucket_name = f"{case_name}_M1_N{n}_K{k}_bias{int(has_bias)}_res{int(has_residual)}_pad{pad_m}"
    out = tuner.dispatch_once(candidates, name=bucket_name, args=(x,))
    best = tuner.dispatch_table[f"once:{bucket_name}"]
    ref = torch_ref(x)
    _print_stats(best, ref, out)

    for name in ("fp16_warp_half2", "fp16_wmma_m16", "i8_m1_dyn_h0", f"i8_pad{pad_m}_copy_h0"):
        if name in candidates:
            _print_stats(name, ref, candidates[name](x))
    print(f"[done][{case_name}] best={best}", flush=True)


def parse_args() -> argparse.Namespace:
    cases = _make_cases()
    parser = argparse.ArgumentParser(description="Decode M=1 INT8 linear/addmm benchmark.")
    parser.add_argument("--case", choices=["all", "custom", *cases.keys()], default="all")
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--workspace-mb", type=int, default=DEFAULT_WORKSPACE_MB)
    parser.add_argument("--max-algos", type=int, default=DEFAULT_MAX_ALGOS)
    parser.add_argument("--pad-m", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this benchmark must run on a CUDA GPU.")

    args = parse_args()
    cases = _make_cases()
    if args.case == "all":
        for name, params in cases.items():
            try:
                run_case(
                    name,
                    int(params["k"]),
                    int(params["n"]),
                    bool(params["bias"]),
                    bool(params["residual"]),
                    args.workspace_mb,
                    args.max_algos,
                    args.seed,
                    args.pad_m,
                )
            except Exception as exc:
                print(f"[case_failed][{name}] {type(exc).__name__}: {exc}", flush=True)
        return

    if args.case == "custom":
        run_case("custom", args.k, args.n, args.bias, args.residual, args.workspace_mb, args.max_algos, args.seed, args.pad_m)
        return

    params = cases[args.case]
    run_case(
        args.case,
        int(params["k"]),
        int(params["n"]),
        bool(params["bias"]),
        bool(params["residual"]),
        args.workspace_mb,
        args.max_algos,
        args.seed,
        args.pad_m,
    )


if __name__ == "__main__":
    main()
