from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "decode_gate_up_fp8_e4b15.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_decode_gate_up_fp8_e4b15"
_KBLOCK_KSCALE_MODE = "fp8e4b15-kblock-kscale-m6144-k2048"
_KBLOCK_KSCALE_HALFWARP_MODE = "fp8e4b15-kblock-kscale-halfwarp-m6144-k2048"
_KBLOCK_KSCALE_HALFWARP_LOWREG_MODE = "fp8e4b15-kblock-kscale-halfwarp-lowreg-m6144-k2048"
_KBLOCK_KSCALE_HALFWARP_HACCUM_MODE = "fp8e4b15-kblock-kscale-halfwarp-haccum-m6144-k2048"
_KBLOCK_KSCALE_HALFWARP_HACCUM_LOWREG_MODE = "fp8e4b15-kblock-kscale-halfwarp-haccum-lowreg-m6144-k2048"
_KBLOCK_KSCALE_HALFWARP_HSCALE_MODE = "fp8e4b15-kblock-kscale-halfwarp-hscale-m6144-k2048"


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
def load_cuda_decode_gate_up_fp8_e4b15_module(verbose: bool = False):
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


def cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int = 6144,
    rows_per_block: int = 1,
    mode: str = _KBLOCK_KSCALE_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA FP8 e4b15 gate/up SwiGLU using the Triton kblock/kscale pack."""
    if str(mode) != _KBLOCK_KSCALE_MODE:
        raise ValueError(f"unsupported gate/up FP8 mode: {mode!r}")
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be torch.uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1).contiguous()
    intermediate_size = int(intermediate_size)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"gate/up FP8 e4b15 expects K=2048, got {x_flat.numel()}")
    if intermediate_size != 6144:
        raise ValueError("gate/up FP8 e4b15 is specialized for intermediate_size=6144")
    if int(gate_up_weight_fp8.numel()) != 16 * 2 * intermediate_size * 128:
        raise ValueError("gate_up_weight_fp8 must contain [16, 2, 6144, 128]")
    if int(gate_up_scales.numel()) != 16 * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [16, 2, 6144]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    elif not out.is_cuda or out.dtype != torch.float16:
        raise ValueError("out must be a CUDA fp16 tensor")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block not in (1, 2, 4, 8, 16):
        raise ValueError("rows_per_block must be one of 1, 2, 4, 8, or 16")

    module = load_cuda_decode_gate_up_fp8_e4b15_module(verbose=verbose)
    module.gate_up_swiglu_fp8e4b15_kblock_kscale_m6144_k2048(
        x_flat,
        gate_up_weight_fp8.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        rows_per_block,
    )
    return out


def cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int = 6144,
    rows_per_block: int = 2,
    mode: str = _KBLOCK_KSCALE_HALFWARP_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA FP8 e4b15 gate/up where each half-warp computes one row."""
    if str(mode) != _KBLOCK_KSCALE_HALFWARP_MODE:
        raise ValueError(f"unsupported gate/up FP8 half-warp mode: {mode!r}")
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be torch.uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1).contiguous()
    intermediate_size = int(intermediate_size)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"gate/up FP8 e4b15 expects K=2048, got {x_flat.numel()}")
    if intermediate_size != 6144:
        raise ValueError("gate/up FP8 e4b15 is specialized for intermediate_size=6144")
    if int(gate_up_weight_fp8.numel()) != 16 * 2 * intermediate_size * 128:
        raise ValueError("gate_up_weight_fp8 must contain [16, 2, 6144, 128]")
    if int(gate_up_scales.numel()) != 16 * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [16, 2, 6144]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    elif not out.is_cuda or out.dtype != torch.float16:
        raise ValueError("out must be a CUDA fp16 tensor")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4, 8, 16):
        raise ValueError("half-warp rows_per_block must be one of 2, 4, 8, or 16")

    module = load_cuda_decode_gate_up_fp8_e4b15_module(verbose=verbose)
    module.gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_m6144_k2048(
        x_flat,
        gate_up_weight_fp8.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        rows_per_block,
    )
    return out


def cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_lowreg(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int = 6144,
    rows_per_block: int = 2,
    mode: str = _KBLOCK_KSCALE_HALFWARP_LOWREG_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA FP8 e4b15 gate/up half-warp variant with shorter live ranges."""
    if str(mode) != _KBLOCK_KSCALE_HALFWARP_LOWREG_MODE:
        raise ValueError(f"unsupported gate/up FP8 low-register half-warp mode: {mode!r}")
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_lowreg requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_lowreg expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be torch.uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1).contiguous()
    intermediate_size = int(intermediate_size)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"gate/up FP8 e4b15 expects K=2048, got {x_flat.numel()}")
    if intermediate_size != 6144:
        raise ValueError("gate/up FP8 e4b15 is specialized for intermediate_size=6144")
    if int(gate_up_weight_fp8.numel()) != 16 * 2 * intermediate_size * 128:
        raise ValueError("gate_up_weight_fp8 must contain [16, 2, 6144, 128]")
    if int(gate_up_scales.numel()) != 16 * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [16, 2, 6144]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    elif not out.is_cuda or out.dtype != torch.float16:
        raise ValueError("out must be a CUDA fp16 tensor")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4, 8, 16):
        raise ValueError("low-register half-warp rows_per_block must be one of 2, 4, 8, or 16")

    module = load_cuda_decode_gate_up_fp8_e4b15_module(verbose=verbose)
    module.gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_lowreg_m6144_k2048(
        x_flat,
        gate_up_weight_fp8.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        rows_per_block,
    )
    return out


def cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_haccum(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int = 6144,
    rows_per_block: int = 2,
    mode: str = _KBLOCK_KSCALE_HALFWARP_HACCUM_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA FP8 e4b15 gate/up half-warp variant with local half2 accumulation."""
    if str(mode) != _KBLOCK_KSCALE_HALFWARP_HACCUM_MODE:
        raise ValueError(f"unsupported gate/up FP8 half-accum mode: {mode!r}")
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_haccum requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_haccum expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be torch.uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1).contiguous()
    intermediate_size = int(intermediate_size)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"gate/up FP8 e4b15 expects K=2048, got {x_flat.numel()}")
    if intermediate_size != 6144:
        raise ValueError("gate/up FP8 e4b15 is specialized for intermediate_size=6144")
    if int(gate_up_weight_fp8.numel()) != 16 * 2 * intermediate_size * 128:
        raise ValueError("gate_up_weight_fp8 must contain [16, 2, 6144, 128]")
    if int(gate_up_scales.numel()) != 16 * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [16, 2, 6144]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    elif not out.is_cuda or out.dtype != torch.float16:
        raise ValueError("out must be a CUDA fp16 tensor")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4, 8, 16):
        raise ValueError("half-accum half-warp rows_per_block must be one of 2, 4, 8, or 16")

    module = load_cuda_decode_gate_up_fp8_e4b15_module(verbose=verbose)
    module.gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_m6144_k2048(
        x_flat,
        gate_up_weight_fp8.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        rows_per_block,
    )
    return out


def cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_haccum_lowreg(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int = 6144,
    rows_per_block: int = 2,
    mode: str = _KBLOCK_KSCALE_HALFWARP_HACCUM_LOWREG_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA FP8 e4b15 gate/up half-warp variant with half2 accumulation and short live ranges."""
    if str(mode) != _KBLOCK_KSCALE_HALFWARP_HACCUM_LOWREG_MODE:
        raise ValueError(f"unsupported gate/up FP8 half-accum low-register mode: {mode!r}")
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError(
            "cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_haccum_lowreg requires CUDA tensors"
        )
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError(
            "cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_haccum_lowreg expects fp16 x/scales"
        )
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be torch.uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1).contiguous()
    intermediate_size = int(intermediate_size)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"gate/up FP8 e4b15 expects K=2048, got {x_flat.numel()}")
    if intermediate_size != 6144:
        raise ValueError("gate/up FP8 e4b15 is specialized for intermediate_size=6144")
    if int(gate_up_weight_fp8.numel()) != 16 * 2 * intermediate_size * 128:
        raise ValueError("gate_up_weight_fp8 must contain [16, 2, 6144, 128]")
    if int(gate_up_scales.numel()) != 16 * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [16, 2, 6144]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    elif not out.is_cuda or out.dtype != torch.float16:
        raise ValueError("out must be a CUDA fp16 tensor")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4, 8, 16):
        raise ValueError("half-accum low-register half-warp rows_per_block must be one of 2, 4, 8, or 16")

    module = load_cuda_decode_gate_up_fp8_e4b15_module(verbose=verbose)
    module.gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_haccum_lowreg_m6144_k2048(
        x_flat,
        gate_up_weight_fp8.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        rows_per_block,
    )
    return out


def cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_hscale(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int = 6144,
    rows_per_block: int = 2,
    mode: str = _KBLOCK_KSCALE_HALFWARP_HSCALE_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA FP8 e4b15 gate/up half-warp variant accumulating scaled k-blocks in half2."""
    if str(mode) != _KBLOCK_KSCALE_HALFWARP_HSCALE_MODE:
        raise ValueError(f"unsupported gate/up FP8 half-scale mode: {mode!r}")
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_hscale requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_hscale expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be torch.uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1).contiguous()
    intermediate_size = int(intermediate_size)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"gate/up FP8 e4b15 expects K=2048, got {x_flat.numel()}")
    if intermediate_size != 6144:
        raise ValueError("gate/up FP8 e4b15 is specialized for intermediate_size=6144")
    if int(gate_up_weight_fp8.numel()) != 16 * 2 * intermediate_size * 128:
        raise ValueError("gate_up_weight_fp8 must contain [16, 2, 6144, 128]")
    if int(gate_up_scales.numel()) != 16 * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [16, 2, 6144]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    elif not out.is_cuda or out.dtype != torch.float16:
        raise ValueError("out must be a CUDA fp16 tensor")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4, 8, 16):
        raise ValueError("half-scale half-warp rows_per_block must be one of 2, 4, 8, or 16")

    module = load_cuda_decode_gate_up_fp8_e4b15_module(verbose=verbose)
    module.gate_up_swiglu_fp8e4b15_kblock_kscale_halfwarp_hscale_m6144_k2048(
        x_flat,
        gate_up_weight_fp8.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        rows_per_block,
    )
    return out
