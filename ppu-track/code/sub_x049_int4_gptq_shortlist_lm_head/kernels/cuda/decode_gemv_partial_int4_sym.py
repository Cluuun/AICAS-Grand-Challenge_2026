from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "decode_gemv_partial_int4_sym.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_decode_gemv_partial_int4_sym"


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
def load_cuda_decode_gemv_partial_int4_sym_module(verbose: bool = False):
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


def cuda_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp(
    x: torch.Tensor,
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    *,
    split_k: int,
    rows_per_block: int = 16,
    partial: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA symmetric-int4 split-K partial GEMV over [K/128, 2048, 64] packed weights."""
    if not (x.is_cuda and weight_int4.is_cuda and scales.is_cuda):
        raise ValueError("CUDA int4 partial GEMV requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("CUDA int4 partial GEMV expects fp16 x/scales")
    if weight_int4.dtype != torch.uint8:
        raise ValueError("weight_int4 must be uint8 packed int4 storage")

    x_flat = x.reshape(-1).contiguous()
    n_out = 2048
    k_in = int(x_flat.numel())
    if k_in not in (2048, 6144):
        raise ValueError(f"CUDA int4 partial GEMV is specialized for K=2048 or 6144, got {k_in}")
    split_k = int(split_k)
    if split_k <= 1:
        raise ValueError("split_k must be > 1")
    if (k_in // 128) % split_k != 0:
        raise ValueError("K/128 must be divisible by split_k")
    if int(weight_int4.numel()) != (k_in // 128) * n_out * 64:
        raise ValueError("weight_int4 must contain [K/128, 2048, 64]")
    if int(scales.numel()) != (k_in // 128) * n_out:
        raise ValueError("scales must contain [K/128, 2048]")

    if partial is None:
        partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    if not partial.is_cuda or partial.dtype != torch.float32 or not partial.is_contiguous():
        raise ValueError("partial must be a contiguous CUDA fp32 tensor")
    if tuple(partial.shape) != (split_k, n_out):
        raise ValueError(f"partial must be [{split_k}, {n_out}]")

    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4, 8, 16, 32):
        raise ValueError("rows_per_block must be one of 2, 4, 8, 16, or 32")

    module = load_cuda_decode_gemv_partial_int4_sym_module(verbose=verbose)
    module.gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp(
        x_flat,
        weight_int4.contiguous(),
        scales.contiguous(),
        partial,
        split_k,
        rows_per_block,
    )
    return partial
