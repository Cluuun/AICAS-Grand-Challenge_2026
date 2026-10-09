from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "decode_gate_up_int4_sym.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_decode_gate_up_int4_sym"
_KBLOCK_KSCALE_HALFWARP_XREUSE_U4OFFSET_NOBCAST_MODE = (
    "int4-sym-kblock-kscale-halfwarp-xreuse-u4offset-nobcast-m6144-k2048"
)


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
def load_cuda_decode_gate_up_int4_sym_module(verbose: bool = False):
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


def cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp(
    x: torch.Tensor,
    gate_up_weight_int4: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int = 6144,
    rows_per_block: int = 4,
    mode: str = _KBLOCK_KSCALE_HALFWARP_XREUSE_U4OFFSET_NOBCAST_MODE,
    candidate_idx: int = 0,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA symmetric-int4 gate/up SwiGLU using the Triton int4 kblock/kscale pack."""
    mode = str(mode)
    if mode != _KBLOCK_KSCALE_HALFWARP_XREUSE_U4OFFSET_NOBCAST_MODE:
        raise ValueError(
            "CUDA gate/up INT4 only supports "
            f"mode={_KBLOCK_KSCALE_HALFWARP_XREUSE_U4OFFSET_NOBCAST_MODE!r}"
        )
    if not x.is_cuda or not gate_up_weight_int4.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp expects fp16 x/scales")
    if gate_up_weight_int4.dtype != torch.uint8:
        raise ValueError("gate_up_weight_int4 must be torch.uint8 packed int4 storage")

    x_flat = x.reshape(-1).contiguous()
    intermediate_size = int(intermediate_size)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"gate/up int4 expects K=2048, got {x_flat.numel()}")
    if intermediate_size != 6144:
        raise ValueError("gate/up int4 is specialized for intermediate_size=6144")
    if int(gate_up_weight_int4.numel()) != 16 * 2 * intermediate_size * 64:
        raise ValueError("gate_up_weight_int4 must contain [16, 2, 6144, 64]")
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
        raise ValueError("gate/up INT4 rows_per_block must be one of 2, 4, 8, or 16")

    module = load_cuda_decode_gate_up_int4_sym_module(verbose=verbose)
    module.gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048(
        x_flat,
        gate_up_weight_int4.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        rows_per_block,
        False,
        False,
        False,
        int(candidate_idx),
    )
    return out
