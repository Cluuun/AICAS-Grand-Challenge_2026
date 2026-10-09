from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "decode_gate_up_swiglu.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_decode_gate_up_swiglu"
_SOTA_MODE = "half2-vec4-f32acc-cg-k2048"
_FP8_MODE = "fp8-e4m3-block128-k2048"
_LEGACY_MODE_ALIASES = {
    "half2-vec4-cg-k2048": _SOTA_MODE,
}


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
def load_cuda_decode_gate_up_swiglu_module(verbose: bool = False):
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


def cuda_decode_gate_up_swiglu(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int | None = None,
    rows_per_block: int = 1,
    block_k: int = 128,
    use_shared_x: bool = False,
    mode: str = _SOTA_MODE,
    rows_per_warp: int = 1,
    warps_per_block: int = 1,
    unroll: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    """Decode-only fp16 CUDA fused gate/up projection and SwiGLU activation.

    The active CU file intentionally contains only the current SOTA kernel.
    Older experimental kernels are preserved in
    ``decode_gate_up_swiglu_experiments.cu`` and are not exported here.
    """
    del block_k, use_shared_x, rows_per_warp, warps_per_block, unroll

    mode = _LEGACY_MODE_ALIASES.get(str(mode), str(mode))
    if mode != _SOTA_MODE:
        raise ValueError(f"active CUDA gate/up SwiGLU only supports mode={_SOTA_MODE!r}")
    if not x.is_cuda or not gate_up_weight.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_weight.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu currently expects fp16 tensors")
    if gate_up_weight.ndim != 2:
        raise ValueError("gate_up_weight must be [2 * intermediate, 2048]")
    if gate_up_bias is not None:
        raise ValueError("cuda_decode_gate_up_swiglu is specialized for Qwen3-VL decode MLP without bias")

    x_flat = x.reshape(-1).contiguous()
    n_fused, k_in = int(gate_up_weight.shape[0]), int(gate_up_weight.shape[1])
    if int(x_flat.numel()) != 2048 or k_in != 2048:
        raise ValueError(f"SOTA CUDA gate/up SwiGLU expects K=2048, got x={x_flat.numel()} weight_K={k_in}")
    if intermediate_size is None:
        if n_fused % 2 != 0:
            raise ValueError("gate_up_weight rows must be even when intermediate_size is omitted")
        intermediate_size = n_fused // 2
    intermediate_size = int(intermediate_size)
    if n_fused < 2 * intermediate_size:
        raise ValueError("gate_up_weight must contain gate rows followed by up rows")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1).contiguous()
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block < 1 or rows_per_block > 32:
        raise ValueError("rows_per_block must be in [1, 32]")

    module = load_cuda_decode_gate_up_swiglu_module(verbose=verbose)
    module.gate_up_swiglu_half2_vec4_f32acc_cg_k2048(
        x_flat,
        gate_up_weight.contiguous(),
        out_flat,
        intermediate_size,
        rows_per_block,
    )
    return out


def quantize_gate_up_e4m3_block128(
    gate_up_weight: torch.Tensor,
    *,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize packed fp16 gate/up weights to e4m3 block-128 storage.

    Returns ``(weight_fp8_uint8, scales_fp16)`` where scales has shape
    ``[2 * intermediate, 16]`` for the fixed Qwen3-VL decode K=2048 path.
    """
    if not gate_up_weight.is_cuda or gate_up_weight.dtype != torch.float16:
        raise ValueError("quantize_gate_up_e4m3_block128 expects fp16 CUDA weight")
    if gate_up_weight.ndim != 2 or int(gate_up_weight.shape[1]) != 2048:
        raise ValueError("gate_up_weight must be [N, 2048]")
    weight = gate_up_weight.contiguous()
    weight_fp8 = torch.empty_like(weight, dtype=torch.uint8)
    scales = torch.empty((int(weight.shape[0]), 16), device=weight.device, dtype=torch.float16)
    module = load_cuda_decode_gate_up_swiglu_module(verbose=verbose)
    module.quantize_gate_up_e4m3_block128_k2048(weight, weight_fp8, scales)
    return weight_fp8, scales


def cuda_decode_gate_up_swiglu_fp8(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int | None = None,
    rows_per_block: int = 1,
    mode: str = _FP8_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """Decode-only e4m3 block-128 gate/up projection plus SwiGLU activation."""
    if str(mode) != _FP8_MODE:
        raise ValueError(f"FP8 CUDA gate/up SwiGLU only supports mode={_FP8_MODE!r}")
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8 requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_gate_up_swiglu_fp8 expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be torch.uint8")
    if gate_up_weight_fp8.ndim != 2 or int(gate_up_weight_fp8.shape[1]) != 2048:
        raise ValueError("gate_up_weight_fp8 must be [2 * intermediate, 2048]")
    if gate_up_scales.ndim != 2 or int(gate_up_scales.shape[1]) != 16:
        raise ValueError("gate_up_scales must be [2 * intermediate, 16]")
    if int(gate_up_scales.shape[0]) != int(gate_up_weight_fp8.shape[0]):
        raise ValueError("gate_up_scales N dimension must match gate_up_weight_fp8")

    x_flat = x.reshape(-1).contiguous()
    n_fused = int(gate_up_weight_fp8.shape[0])
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"FP8 CUDA gate/up SwiGLU expects K=2048, got x={x_flat.numel()}")
    if intermediate_size is None:
        if n_fused % 2 != 0:
            raise ValueError("gate_up_weight_fp8 rows must be even when intermediate_size is omitted")
        intermediate_size = n_fused // 2
    intermediate_size = int(intermediate_size)
    if n_fused < 2 * intermediate_size:
        raise ValueError("gate_up_weight_fp8 must contain gate rows followed by up rows")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1).contiguous()
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    rows_per_block = int(rows_per_block)
    if rows_per_block < 1 or rows_per_block > 32:
        raise ValueError("rows_per_block must be in [1, 32]")

    module = load_cuda_decode_gate_up_swiglu_module(verbose=verbose)
    module.gate_up_swiglu_fp8_e4m3_block128_k2048(
        x_flat,
        gate_up_weight_fp8.contiguous(),
        gate_up_scales.contiguous(),
        out_flat,
        intermediate_size,
        rows_per_block,
    )
    return out
