from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "decode_gemv_partial_fp8_e4b15.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_decode_gemv_partial_fp8_e4b15"


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
def load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose: bool = False):
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


def _check_common(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    split_k: int,
    partial: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda):
        raise ValueError("CUDA FP8 partial GEMV requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("CUDA FP8 partial GEMV expects fp16 x/scales")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1).contiguous()
    n_out = 2048
    k_in = int(x_flat.numel())
    if k_in not in (2048, 6144):
        raise ValueError(f"CUDA FP8 partial GEMV is specialized for K=2048 or 6144, got {k_in}")
    split_k = int(split_k)
    if split_k <= 1:
        raise ValueError("split_k must be > 1")
    if (k_in // 128) % split_k != 0:
        raise ValueError("K/128 must be divisible by split_k")
    if int(weight_fp8.numel()) != (k_in // 128) * n_out * 128:
        raise ValueError("weight_fp8 must contain [K/128, 2048, 128]")
    if int(scales.numel()) != (k_in // 128) * n_out:
        raise ValueError("scales must contain [2048, K/128] or [K/128, 2048]")

    if partial is None:
        partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    if not partial.is_cuda or partial.dtype != torch.float32 or not partial.is_contiguous():
        raise ValueError("partial must be a contiguous CUDA fp32 tensor")
    if tuple(partial.shape) != (split_k, n_out):
        raise ValueError(f"partial must be [{split_k}, {n_out}]")
    return x_flat, partial


def cuda_splitk_reduce_add_rmsnorm_2048(
    partial: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    threads: int = 256,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (partial.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("CUDA reduce/add/RMSNorm requires CUDA tensors")
    if partial.dtype != torch.float32:
        raise ValueError("partial must be fp32")
    if partial.ndim != 2 or int(partial.shape[1]) != 2048:
        raise ValueError("partial must be [split_k, 2048]")
    split_k = int(partial.shape[0])
    if split_k <= 1:
        raise ValueError("split_k must be > 1")
    residual_flat = residual.reshape(-1).contiguous()
    if residual_flat.dtype != torch.float16 or int(residual_flat.numel()) != 2048:
        raise ValueError("residual must be fp16 with 2048 elements")
    if norm_weight.dtype != torch.float16:
        norm_weight = norm_weight.to(dtype=torch.float16)
    norm_weight_flat = norm_weight.reshape(-1).contiguous()
    if int(norm_weight_flat.numel()) != 2048:
        raise ValueError("norm_weight must have 2048 elements")
    if bias is None:
        bias_flat = residual_flat
        has_bias = False
    else:
        bias_flat = bias.reshape(-1).contiguous()
        if bias_flat.dtype != torch.float16 or int(bias_flat.numel()) != 2048:
            raise ValueError("bias must be fp16 with 2048 elements")
        has_bias = True
    if out_sum is None:
        out_sum = torch.empty((2048,), device=partial.device, dtype=torch.float16)
    if out_norm is None:
        out_norm = torch.empty((2048,), device=partial.device, dtype=torch.float16)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if (
        out_sum_flat.dtype != torch.float16
        or out_norm_flat.dtype != torch.float16
        or int(out_sum_flat.numel()) != 2048
        or int(out_norm_flat.numel()) != 2048
    ):
        raise ValueError("out tensors must be fp16 with 2048 elements")
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.splitk_reduce_add_rmsnorm_2048(
        partial.contiguous(),
        bias_flat,
        residual_flat,
        norm_weight_flat,
        out_sum_flat,
        out_norm_flat,
        split_k,
        has_bias,
        float(eps),
        int(threads),
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def cuda_gemv_splitk_partial_fp8_e4b15_kblock_major(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    split_k: int,
    variant: int = 0,
    partial: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA partial GEMV matching Triton's kblock-major 16x128/8-warp tile shape."""
    x_flat, partial = _check_common(x, weight_fp8, scales, int(split_k), partial)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.gemv_splitk_partial_fp8e4b15_kblock_major(
        x_flat,
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial,
        int(split_k),
        int(variant),
    )
    return partial


def cuda_gemv_splitk_partial_fp8_e4b15_kblock_major_taskloop(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    split_k: int,
    blocks: int = 0,
    partial: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Diagnostic task-loop version matching the O section inside fused cooperative kernels."""
    x_flat, partial = _check_common(x, weight_fp8, scales, int(split_k), partial)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.gemv_splitk_partial_fp8e4b15_kblock_major_taskloop(
        x_flat,
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial,
        int(split_k),
        int(blocks),
    )
    return partial


def cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    coop_blocks: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental cooperative stage2 merge plus O FP8 partial GEMV."""
    if not (partial_m.is_cuda and partial_l.is_cuda and partial_acc.is_cuda):
        raise ValueError("stage2/O fused path requires CUDA attention partials")
    if partial_m.dtype != torch.float32 or partial_l.dtype != torch.float32 or partial_acc.dtype != torch.float32:
        raise ValueError("attention partial tensors must be fp32")
    if partial_m.ndim != 2 or int(partial_m.shape[0]) != 16:
        raise ValueError("partial_m must be [16, workspace_splits]")
    if tuple(partial_l.shape) != tuple(partial_m.shape):
        raise ValueError("partial_l shape mismatch")
    workspace_splits = int(partial_m.shape[1])
    if tuple(partial_acc.shape) != (16, workspace_splits, 128):
        raise ValueError("partial_acc must be [16, workspace_splits, 128]")
    if int(active_splits) < 1 or int(active_splits) > min(workspace_splits, 16):
        raise ValueError("active_splits must be in [1, min(workspace_splits,16)]")
    if int(split_k) != 4:
        raise ValueError("stage2/O fused path currently supports split_k=4")
    if weight_fp8.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise ValueError("stage2/O fused path expects uint8 fp8 weights and fp16 scales")
    if int(weight_fp8.numel()) != 16 * 2048 * 128:
        raise ValueError("weight_fp8 must contain [16,2048,128] kblock-major O weights")
    if int(scales.numel()) != 16 * 2048:
        raise ValueError("scales must contain [2048,16] O scales")
    device = partial_m.device
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=device, dtype=torch.float32)
    if scratch_x.dtype != torch.float16 or int(scratch_x.numel()) != 2048 or not scratch_x.is_cuda:
        raise ValueError("scratch_x must be CUDA fp16 with 2048 elements")
    if partial.dtype != torch.float32 or tuple(partial.shape) != (4, 2048) or not partial.is_cuda:
        raise ValueError("partial must be CUDA fp32 with shape [4,2048]")
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_fp8e4b15_kblock_major(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        int(active_splits),
        int(split_k),
        int(coop_blocks),
    )
    return scratch_x, partial


def cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_barrier(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    barrier_counter: torch.Tensor | None = None,
    blocks: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental in-kernel-barrier stage2 merge plus O FP8 partial GEMV."""
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=partial_m.device, dtype=torch.float32)
    if barrier_counter is None:
        barrier_counter = torch.empty((1,), device=partial_m.device, dtype=torch.int32)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_fp8e4b15_kblock_major_barrier(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        barrier_counter.contiguous(),
        int(active_splits),
        int(split_k),
        int(blocks),
    )
    return scratch_x, partial


def cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_local(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental O-kblock-local stage2 merge plus O FP8 partial GEMV."""
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=partial_m.device, dtype=torch.float32)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_fp8e4b15_kblock_major_local(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        int(active_splits),
        int(split_k),
    )
    return scratch_x, partial


def cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_rowgroup(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    coop_blocks: int = 0,
    row_blocks_per_cta: int = 2,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental cooperative stage2 merge plus grouped-row O FP8 partial GEMV."""
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=partial_m.device, dtype=torch.float32)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_fp8e4b15_kblock_major_rowgroup(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        int(active_splits),
        int(split_k),
        int(coop_blocks),
        int(row_blocks_per_cta),
    )
    return scratch_x, partial


def cuda_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    row_blocks_per_cta: int = -22,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental local stage2 merge plus O symmetric-INT4 partial GEMV."""
    if int(split_k) != 4:
        raise ValueError("stage2/O INT4 fused path currently supports split_k=4")
    if weight_int4.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise ValueError("stage2/O INT4 fused path expects uint8 int4 weights and fp16 scales")
    if int(weight_int4.numel()) != 16 * 2048 * 64:
        raise ValueError("weight_int4 must contain [16,2048,64] kblock-major O weights")
    if int(scales.numel()) != 16 * 2048:
        raise ValueError("scales must contain [16,2048] O scales")
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=partial_m.device, dtype=torch.float32)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_int4.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        int(active_splits),
        int(split_k),
        int(row_blocks_per_cta),
    )
    return scratch_x, partial


def cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_pc(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    ready_flags: torch.Tensor | None = None,
    coop_blocks: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental producer-consumer stage2 merge plus O FP8 partial GEMV."""
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=partial_m.device, dtype=torch.float32)
    if ready_flags is None:
        ready_flags = torch.empty((16,), device=partial_m.device, dtype=torch.int32)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_fp8e4b15_kblock_major_pc(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        ready_flags.contiguous(),
        int(active_splits),
        int(split_k),
        int(coop_blocks),
    )
    return scratch_x, partial


def cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_local_rowgroup(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    row_blocks_per_cta: int = 2,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental non-cooperative grouped-row local stage2/O fusion."""
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=partial_m.device, dtype=torch.float32)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_fp8e4b15_kblock_major_local_rowgroup(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        int(active_splits),
        int(split_k),
        int(row_blocks_per_cta),
    )
    return scratch_x, partial


def cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_pc_full(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    active_splits: int,
    split_k: int = 4,
    scratch_x: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    ready_flags: torch.Tensor | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental full-grid producer-consumer stage2 merge plus O FP8 partial GEMV."""
    if scratch_x is None:
        scratch_x = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    if partial is None:
        partial = torch.empty((4, 2048), device=partial_m.device, dtype=torch.float32)
    if ready_flags is None:
        ready_flags = torch.empty((16,), device=partial_m.device, dtype=torch.int32)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.prefix_stage2_o_partial_fp8e4b15_kblock_major_pc_full(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        scratch_x.contiguous(),
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial.contiguous(),
        ready_flags.contiguous(),
        int(active_splits),
        int(split_k),
    )
    return scratch_x, partial


def cuda_gemv_splitk_partial_fp8_e4b15_kblock_kscale(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    split_k: int,
    variant: int = 0,
    partial: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA partial GEMV matching Triton's kblock/kscale 16x128/8-warp tile shape."""
    x_flat, partial = _check_common(x, weight_fp8, scales, int(split_k), partial)
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.gemv_splitk_partial_fp8e4b15_kblock_kscale(
        x_flat,
        weight_fp8.contiguous(),
        scales.contiguous(),
        partial,
        int(split_k),
        int(variant),
    )
    return partial


def cuda_down_fp8_e4b15_kblock_kscale_split3_add_rmsnorm_boundary(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    partial: torch.Tensor | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental cooperative down FP8 GEMV + residual add + RMSNorm boundary."""
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("CUDA down FP8 fused boundary requires CUDA tensors")
    if x.dtype != torch.float16 or residual.dtype != torch.float16 or norm_weight.dtype != torch.float16:
        raise ValueError("CUDA down FP8 fused boundary expects fp16 x/residual/norm_weight")
    if scales.dtype != torch.float16:
        raise ValueError("CUDA down FP8 fused boundary expects fp16 scales")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")
    x_flat = x.reshape(-1).contiguous()
    residual_flat = residual.reshape(-1).contiguous()
    norm_weight_flat = norm_weight.reshape(-1).contiguous()
    if int(x_flat.numel()) != 6144:
        raise ValueError("CUDA down FP8 fused boundary is specialized for K=6144")
    if int(residual_flat.numel()) != 2048 or int(norm_weight_flat.numel()) != 2048:
        raise ValueError("residual/norm_weight must have 2048 elements")
    if int(weight_fp8.numel()) != 48 * 2048 * 128:
        raise ValueError("weight_fp8 must contain [48, 2048, 128]")
    if int(scales.numel()) != 48 * 2048:
        raise ValueError("scales must contain [48, 2048]")
    if out_sum is None:
        out_sum = torch.empty_like(residual_flat)
    else:
        out_sum = out_sum.reshape(-1)
    if out_norm is None:
        out_norm = torch.empty_like(residual_flat)
    else:
        out_norm = out_norm.reshape(-1)
    if partial is None:
        partial = torch.empty((3, 2048), device=x.device, dtype=torch.float32)
    if out_sum.dtype != torch.float16 or out_norm.dtype != torch.float16:
        raise ValueError("out_sum/out_norm must be fp16")
    if int(out_sum.numel()) != 2048 or int(out_norm.numel()) != 2048:
        raise ValueError("out_sum/out_norm must have 2048 elements")
    if partial.dtype != torch.float32 or tuple(partial.shape) != (3, 2048):
        raise ValueError("partial must be fp32 [3, 2048]")
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.down_fp8e4b15_kblock_kscale_split3_add_rmsnorm_boundary(
        x_flat,
        weight_fp8.contiguous(),
        scales.contiguous(),
        residual_flat,
        norm_weight_flat,
        out_sum.contiguous(),
        out_norm.contiguous(),
        partial.contiguous(),
        float(eps),
    )
    return out_sum.view_as(residual), out_norm.view_as(residual)


def cuda_down_fp8_e4b15_kblock_kscale_split3_sum_sumsq_boundary(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    rows_per_block: int = 4,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    tile_sumsq: torch.Tensor | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-kernel down FP8 boundary that writes sum+sumsq instead of split-K partials."""
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("CUDA down FP8 sum+sumsq boundary requires CUDA tensors")
    if x.dtype != torch.float16 or residual.dtype != torch.float16 or norm_weight.dtype != torch.float16:
        raise ValueError("CUDA down FP8 sum+sumsq boundary expects fp16 x/residual/norm_weight")
    if scales.dtype != torch.float16:
        raise ValueError("CUDA down FP8 sum+sumsq boundary expects fp16 scales")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")
    rows_per_block = int(rows_per_block)
    if rows_per_block not in (2, 4, 8, 16):
        raise ValueError("rows_per_block must be 2, 4, 8, or 16")
    x_flat = x.reshape(-1).contiguous()
    residual_flat = residual.reshape(-1).contiguous()
    norm_weight_flat = norm_weight.reshape(-1).contiguous()
    if int(x_flat.numel()) != 6144:
        raise ValueError("CUDA down FP8 sum+sumsq boundary is specialized for K=6144")
    if int(residual_flat.numel()) != 2048 or int(norm_weight_flat.numel()) != 2048:
        raise ValueError("residual/norm_weight must have 2048 elements")
    if int(weight_fp8.numel()) != 48 * 2048 * 128:
        raise ValueError("weight_fp8 must contain [48, 2048, 128]")
    if int(scales.numel()) != 48 * 2048:
        raise ValueError("scales must contain [48, 2048]")
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
        raise ValueError("tile_sumsq shape mismatch")
    module = load_cuda_decode_gemv_partial_fp8_e4b15_module(verbose=verbose)
    module.down_fp8e4b15_kblock_kscale_split3_sum_sumsq(
        x_flat,
        weight_fp8.contiguous(),
        scales.contiguous(),
        residual_flat,
        out_sum.contiguous(),
        tile_sumsq.contiguous(),
        rows_per_block,
    )
    module.rmsnorm_from_sum_sumsq_2048(
        out_sum.contiguous(),
        tile_sumsq.contiguous(),
        norm_weight_flat,
        out_norm.contiguous(),
        float(eps),
    )
    return out_sum.view_as(residual), out_norm.view_as(residual)
