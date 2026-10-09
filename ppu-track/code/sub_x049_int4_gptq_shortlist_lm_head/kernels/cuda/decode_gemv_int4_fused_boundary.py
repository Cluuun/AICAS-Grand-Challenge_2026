from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "decode_gemv_int4_fused_boundary.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_decode_gemv_int4_fused_boundary"


def _set_default_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def _ensure_ninja_on_path() -> None:
    try:
        import ninja  # type: ignore
    except Exception:
        return
    bin_dir = getattr(ninja, "BIN_DIR", None)
    if not bin_dir:
        return
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if str(bin_dir) not in paths:
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def load_cuda_decode_gemv_int4_fused_boundary_module(verbose: bool = False):
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _set_default_arch_list()
    _ensure_ninja_on_path()
    return load(
        name=_MODULE_NAME,
        sources=[str(_SOURCE)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(_BUILD_DIR),
        verbose=verbose,
    )


def cuda_down_int4_split4_add_rmsnorm_boundary(
    x: torch.Tensor,
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    tile_sumsq: torch.Tensor | None = None,
    rows_per_block: int = 4,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (x.is_cuda and weight_int4.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("CUDA down INT4 fused boundary requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("CUDA down INT4 fused boundary expects fp16 x/scales")
    if residual.dtype != torch.float16 or norm_weight.dtype != torch.float16:
        raise ValueError("CUDA down INT4 fused boundary expects fp16 residual/norm_weight")
    if weight_int4.dtype != torch.uint8:
        raise ValueError("weight_int4 must be uint8 packed int4 storage")

    x_flat = x.reshape(-1).contiguous()
    residual_flat = residual.reshape(-1).contiguous()
    norm_weight_flat = norm_weight.reshape(-1).contiguous()
    if int(x_flat.numel()) != 6144:
        raise ValueError("CUDA down INT4 fused boundary is specialized for K=6144")
    if int(residual_flat.numel()) != 2048 or int(norm_weight_flat.numel()) != 2048:
        raise ValueError("residual/norm_weight must have 2048 elements")
    if int(weight_int4.numel()) != 48 * 2048 * 64:
        raise ValueError("weight_int4 must contain [48, 2048, 64]")
    if int(scales.numel()) != 48 * 2048:
        raise ValueError("scales must contain [48, 2048]")
    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4):
        raise ValueError("rows_per_block must be 2 or 4")

    if out_sum is None:
        out_sum = torch.empty_like(residual_flat)
    else:
        out_sum = out_sum.reshape(-1)
    if out_norm is None:
        out_norm = torch.empty_like(residual_flat)
    else:
        out_norm = out_norm.reshape(-1)
    if tile_sumsq is None:
        tile_sumsq = torch.empty((2048 // rows_per_block,), device=x.device, dtype=torch.float32)
    if out_sum.dtype != torch.float16 or out_norm.dtype != torch.float16:
        raise ValueError("out_sum/out_norm must be fp16")
    if int(out_sum.numel()) != 2048 or int(out_norm.numel()) != 2048:
        raise ValueError("out_sum/out_norm must have 2048 elements")
    if tile_sumsq.dtype != torch.float32 or int(tile_sumsq.numel()) != 2048 // rows_per_block:
        raise ValueError(f"tile_sumsq must be fp32 [{2048 // rows_per_block}]")

    module = load_cuda_decode_gemv_int4_fused_boundary_module(verbose=verbose)
    module.down_int4_split4_add_rmsnorm_boundary(
        x_flat,
        weight_int4.contiguous(),
        scales.contiguous(),
        residual_flat,
        norm_weight_flat,
        out_sum.contiguous(),
        out_norm.contiguous(),
        tile_sumsq.contiguous(),
        float(eps),
        rows_per_block,
    )
    return out_sum.view_as(residual), out_norm.view_as(residual)
