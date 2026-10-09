from __future__ import annotations

import os
import sys
import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "prefix_attention.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_prefix_attention"
_FORCE_REBUILD_ENV = "AICAS_CUDA_PREFIX_ATTENTION_FORCE_REBUILD"
_LONG_SPLIT_M_ENV = "AICAS_CUDA_DECODE_PREFIX_ATTENTION_LONG_SPLIT_M"
_LONG_KIND_ENV = "AICAS_CUDA_DECODE_PREFIX_ATTENTION_LONG_KIND"


def _force_rebuild_enabled() -> bool:
    raw = os.environ.get(_FORCE_REBUILD_ENV, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{_FORCE_REBUILD_ENV} must be '0' or '1', got {raw!r}")
    return raw == "1"


def _prebuilt_so_path() -> Path:
    return _BUILD_DIR / f"{_MODULE_NAME}.so"


def _load_prebuilt_module(verbose: bool):
    so_path = _prebuilt_so_path()
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        return existing
    if verbose:
        print(f"[cuda.prefix_attention] loading prebuilt module: {so_path}")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


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


def _extra_cuda_cflags() -> list[str]:
    flags = ["-O3", "--use_fast_math"]
    host_cxx = os.environ.get("CUDAHOSTCXX")
    if host_cxx:
        flags.append(f"-ccbin={host_cxx}")
    return flags


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


@lru_cache(maxsize=1)
def load_cuda_prefix_attention_module(verbose: bool = False):
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not _force_rebuild_enabled() and _prebuilt_so_path().exists():
        return _load_prebuilt_module(verbose=verbose)
    _set_default_arch_list()
    _ensure_ninja_on_path()
    return load(
        name=_MODULE_NAME,
        sources=[str(_SOURCE)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=_extra_cuda_cflags(),
        build_directory=str(_BUILD_DIR),
        verbose=verbose,
    )


def make_prefix_attention_split_workspace(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    *,
    split_m: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not query.is_cuda or not key_cache.is_cuda:
        raise ValueError("prefix attention workspace requires CUDA tensors")
    batch = int(query.shape[0])
    query_heads = int(query.shape[1])
    max_k = int(key_cache.shape[2])
    head_dim = int(query.shape[3])
    split_m = int(split_m)
    if split_m <= 0:
        raise ValueError("split_m must be positive")
    num_splits = (max_k + split_m - 1) // split_m
    if batch != 1:
        raise ValueError("prefix attention CUDA experiment is specialized for batch=1")
    partial_m = torch.empty((query_heads, num_splits), device=query.device, dtype=torch.float32)
    partial_l = torch.empty((query_heads, num_splits), device=query.device, dtype=torch.float32)
    partial_acc = torch.zeros(
        (query_heads, num_splits, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    return partial_m, partial_l, partial_acc


def make_prefix_attention_atomic_workspace(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    *,
    split_m: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    partial_m, partial_l, partial_acc = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    counters = torch.zeros((int(query.shape[1]),), device=query.device, dtype=torch.int32)
    return partial_m, partial_l, partial_acc, counters


def cuda_decode_prefix_attention_bucket_config(prefix_len: int) -> tuple[int, str]:
    prefix_len = int(prefix_len)
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    if prefix_len <= 512:
        return 32, "split"
    if prefix_len <= 768:
        return 48, "split"
    if prefix_len <= 1024:
        return 64, "split"
    default_split_m = 96
    split_m = _env_int(_LONG_SPLIT_M_ENV, default_split_m)
    if split_m not in (32, 48, 64, 96, 128, 256):
        raise ValueError(f"{_LONG_SPLIT_M_ENV} must be one of 32,48,64,96,128,256")
    kind = os.environ.get(_LONG_KIND_ENV, "split").strip().lower()
    if kind not in ("split", "atomic"):
        raise ValueError(f"{_LONG_KIND_ENV} must be 'split' or 'atomic'")
    return split_m, kind


def make_prefix_attention_bucketed_workspace(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    *,
    prefix_len: int | None = None,
) -> tuple[torch.Tensor, ...]:
    bucket_len = int(key_cache.shape[2]) if prefix_len is None else int(prefix_len)
    split_m, kind = cuda_decode_prefix_attention_bucket_config(bucket_len)
    if kind == "atomic":
        return make_prefix_attention_atomic_workspace(query, key_cache, split_m=split_m)
    return make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)


def _validate_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    out: torch.Tensor | None,
    output_layout: str,
    split_m: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not (query.is_cuda and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda):
        raise ValueError("cuda prefix attention requires CUDA tensors")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("cuda prefix attention currently supports fp16 query/key/value")
    if query.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected query/key/value tensors with shape [B, H, T, D]")
    if int(query.shape[2]) != 1:
        raise ValueError("cuda prefix attention only supports q_len=1")
    if int(query.shape[3]) != 128:
        raise ValueError("cuda prefix attention is specialized for head_dim=128")
    if int(key_cache.shape[3]) != 128 or int(value_cache.shape[3]) != 128:
        raise ValueError("cuda prefix attention is specialized for head_dim=128")
    if int(query.shape[1]) != 2 * int(key_cache.shape[1]):
        raise ValueError("cuda prefix attention requires exactly 2 query heads per KV head")
    if int(query.shape[0]) != 1:
        raise ValueError("cuda prefix attention experiment is specialized for batch=1")
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("key_cache and value_cache shapes must match")
    if int(query.shape[0]) != int(key_cache.shape[0]):
        raise ValueError("query/key/value batch sizes must match")
    if cache_position.numel() != 1:
        raise ValueError("cache_position must contain a single decode index")
    if not (query.is_contiguous() and key_cache.is_contiguous() and value_cache.is_contiguous()):
        raise ValueError("cuda prefix attention expects contiguous query/key/value tensors")
    if split_m not in (32, 48, 64, 96, 128, 256):
        raise ValueError("cuda prefix attention split_m must be one of 32, 48, 64, 96, 128, 256")

    query_heads = int(query.shape[1])
    head_dim = int(query.shape[3])
    flat_shape = (1, 1, query_heads * head_dim)
    qhd_shape = tuple(query.shape)
    if output_layout == "flat":
        if out is None:
            out = torch.empty(flat_shape, device=query.device, dtype=query.dtype)
        if tuple(out.shape) != flat_shape:
            raise ValueError("flat out must have shape [B, 1, query_heads * head_dim]")
    elif output_layout == "qhd":
        if out is None:
            out = torch.empty_like(query)
        if tuple(out.shape) != qhd_shape:
            raise ValueError("qhd out must match query shape")
    else:
        raise ValueError("output_layout must be 'qhd' or 'flat'")
    if out.dtype != query.dtype or out.device != query.device or not out.is_contiguous():
        raise ValueError("out must be contiguous with dtype/device matching query")

    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    max_k = int(key_cache.shape[2])
    num_splits = (max_k + int(split_m) - 1) // int(split_m)
    expected_m_shape = (query_heads, num_splits)
    expected_acc_shape = (query_heads, num_splits, head_dim)
    if tuple(partial_m.shape) != expected_m_shape or tuple(partial_l.shape) != expected_m_shape:
        raise ValueError(f"partial_m/partial_l workspace must have shape {expected_m_shape}")
    if tuple(partial_acc.shape) != expected_acc_shape:
        raise ValueError(f"partial_acc workspace must have shape {expected_acc_shape}")
    if partial_m.dtype != torch.float32 or partial_l.dtype != torch.float32 or partial_acc.dtype != torch.float32:
        raise ValueError("prefix attention workspace tensors must be float32")
    if not (partial_m.is_cuda and partial_l.is_cuda and partial_acc.is_cuda):
        raise ValueError("prefix attention workspace tensors must be CUDA tensors")
    if not (partial_m.is_contiguous() and partial_l.is_contiguous() and partial_acc.is_contiguous()):
        raise ValueError("prefix attention workspace tensors must be contiguous")
    return out, workspace


def _validate_atomic_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    out: torch.Tensor | None,
    output_layout: str,
    split_m: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if workspace is None:
        workspace = make_prefix_attention_atomic_workspace(query, key_cache, split_m=split_m)
    partial_workspace = (workspace[0], workspace[1], workspace[2])
    out, _ = _validate_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        split_m,
        partial_workspace,
    )
    counters = workspace[3]
    expected_shape = (int(query.shape[1]),)
    if tuple(counters.shape) != expected_shape:
        raise ValueError(f"prefix attention atomic counters must have shape {expected_shape}")
    if counters.dtype != torch.int32:
        raise ValueError("prefix attention atomic counters must be int32")
    if not counters.is_cuda or not counters.is_contiguous():
        raise ValueError("prefix attention atomic counters must be contiguous CUDA tensors")
    return out, workspace


def _validate_single_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    out: torch.Tensor | None,
    output_layout: str,
    num_threads: int,
) -> torch.Tensor:
    if not (query.is_cuda and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda):
        raise ValueError("cuda prefix attention requires CUDA tensors")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("cuda prefix attention currently supports fp16 query/key/value")
    if query.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected query/key/value tensors with shape [B, H, T, D]")
    if int(query.shape[0]) != 1 or int(key_cache.shape[0]) != 1:
        raise ValueError("cuda prefix attention experiment is specialized for batch=1")
    if int(query.shape[1]) != 16 or int(key_cache.shape[1]) != 8:
        raise ValueError("cuda prefix attention is specialized for 16 query heads and 8 KV heads")
    if int(query.shape[2]) != 1:
        raise ValueError("cuda prefix attention only supports q_len=1")
    if int(query.shape[3]) != 128 or int(key_cache.shape[3]) != 128 or int(value_cache.shape[3]) != 128:
        raise ValueError("cuda prefix attention is specialized for head_dim=128")
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("key_cache and value_cache shapes must match")
    if cache_position.numel() != 1:
        raise ValueError("cache_position must contain a single decode index")
    if not (query.is_contiguous() and key_cache.is_contiguous() and value_cache.is_contiguous()):
        raise ValueError("cuda prefix attention expects contiguous query/key/value tensors")
    if int(num_threads) not in (128, 256):
        raise ValueError("cuda single prefix attention num_threads must be 128 or 256")

    flat_shape = (1, 1, 16 * 128)
    if output_layout == "flat":
        if out is None:
            out = torch.empty(flat_shape, device=query.device, dtype=query.dtype)
        if tuple(out.shape) != flat_shape:
            raise ValueError("flat out must have shape [1, 1, 2048]")
    elif output_layout == "qhd":
        if out is None:
            out = torch.empty_like(query)
        if tuple(out.shape) != tuple(query.shape):
            raise ValueError("qhd out must match query shape")
    else:
        raise ValueError("output_layout must be 'qhd' or 'flat'")
    if out.dtype != query.dtype or out.device != query.device or not out.is_contiguous():
        raise ValueError("out must be contiguous with dtype/device matching query")
    return out


def _validate_tiled_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    out: torch.Tensor | None,
    output_layout: str,
    block_m: int,
    num_threads: int,
) -> torch.Tensor:
    if not (query.is_cuda and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda):
        raise ValueError("cuda prefix attention requires CUDA tensors")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("cuda prefix attention currently supports fp16 query/key/value")
    if query.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected query/key/value tensors with shape [B, H, T, D]")
    if int(query.shape[0]) != 1 or int(key_cache.shape[0]) != 1:
        raise ValueError("cuda prefix attention experiment is specialized for batch=1")
    if int(query.shape[1]) != 16 or int(key_cache.shape[1]) != 8:
        raise ValueError("cuda prefix attention is specialized for 16 query heads and 8 KV heads")
    if int(query.shape[2]) != 1:
        raise ValueError("cuda prefix attention only supports q_len=1")
    if int(query.shape[3]) != 128 or int(key_cache.shape[3]) != 128 or int(value_cache.shape[3]) != 128:
        raise ValueError("cuda prefix attention is specialized for head_dim=128")
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("key_cache and value_cache shapes must match")
    if cache_position.numel() != 1:
        raise ValueError("cache_position must contain a single decode index")
    if not (query.is_contiguous() and key_cache.is_contiguous() and value_cache.is_contiguous()):
        raise ValueError("cuda prefix attention expects contiguous query/key/value tensors")
    if int(block_m) not in (64, 128, 256):
        raise ValueError("cuda tiled prefix attention block_m must be 64, 128, or 256")
    if int(num_threads) not in (128, 256, 512, 1024):
        raise ValueError("cuda tiled prefix attention num_threads must be 128, 256, 512, or 1024")

    flat_shape = (1, 1, 16 * 128)
    if output_layout == "flat":
        if out is None:
            out = torch.empty(flat_shape, device=query.device, dtype=query.dtype)
        if tuple(out.shape) != flat_shape:
            raise ValueError("flat out must have shape [1, 1, 2048]")
    elif output_layout == "qhd":
        if out is None:
            out = torch.empty_like(query)
        if tuple(out.shape) != tuple(query.shape):
            raise ValueError("qhd out must match query shape")
    else:
        raise ValueError("output_layout must be 'qhd' or 'flat'")
    if out.dtype != query.dtype or out.device != query.device or not out.is_contiguous():
        raise ValueError("out must be contiguous with dtype/device matching query")
    return out


def cuda_decode_prefix_attention_tiled_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    block_m: int = 128,
    num_threads: int = 256,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA qhead kernel following the Triton tiled online-softmax structure."""
    block_m = int(block_m)
    num_threads = int(num_threads)
    out = _validate_tiled_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        block_m,
        num_threads,
    )
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_tiled_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        float(scale),
        block_m,
        num_threads,
    )
    return out


def cuda_decode_prefix_attention_tiled_warpred_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    block_m: int = 128,
    num_threads: int = 256,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA qhead kernel with tile reductions confined to one warp."""
    block_m = int(block_m)
    num_threads = int(num_threads)
    out = _validate_tiled_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        block_m,
        num_threads,
    )
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_tiled_warpred_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        float(scale),
        block_m,
        num_threads,
    )
    return out


def cuda_decode_prefix_attention_tiled_vgroup_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    block_m: int = 128,
    num_threads: int = 512,
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA qhead kernel with grouped lanes reducing each output dimension."""
    block_m = int(block_m)
    num_threads = int(num_threads)
    out = _validate_tiled_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        block_m,
        num_threads,
    )
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_tiled_vgroup_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        float(scale),
        block_m,
        num_threads,
    )
    return out


def cuda_decode_prefix_attention_split_vgroup_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    split_m: int = 64,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> torch.Tensor:
    """Split-prefix CUDA kernel whose stage1 uses grouped lanes per output dim."""
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda split-vgroup num_threads must be 128, 256, 512, or 1024")
    out, workspace = _validate_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        split_m,
        workspace,
    )
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_vgroup_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return out


def cuda_decode_prefix_attention_stage1_vgroup_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    split_m: int,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda stage1-vgroup num_threads must be 128, 256, 512, or 1024")
    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage1_vgroup_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return partial_m, partial_l, partial_acc


def cuda_decode_prefix_attention_stage1_vgroup_2d_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    split_m: int,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda stage1-vgroup-2d num_threads must be 128, 256, 512, or 1024")
    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage1_vgroup_2d_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return partial_m, partial_l, partial_acc


def cuda_decode_prefix_attention_stage1_vgroup_h2dot_2d_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    split_m: int,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda stage1-vgroup-h2dot-2d num_threads must be 128, 256, 512, or 1024")
    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage1_vgroup_h2dot_2d_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return partial_m, partial_l, partial_acc


def cuda_decode_prefix_attention_stage1_h2dot_direct_2d_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    split_m: int,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda stage1-h2dot-direct-2d num_threads must be 128, 256, 512, or 1024")
    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage1_h2dot_direct_2d_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return partial_m, partial_l, partial_acc


def cuda_decode_prefix_attention_stage1_h2dot_direct_qcache_2d_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    split_m: int,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda stage1-h2dot-direct-qcache-2d num_threads must be 128, 256, 512, or 1024")
    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage1_h2dot_direct_qcache_2d_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return partial_m, partial_l, partial_acc


def cuda_decode_prefix_attention_stage1_direct_2d_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    split_m: int,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda stage1-direct-2d num_threads must be 128, 256, 512, or 1024")
    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage1_direct_2d_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return partial_m, partial_l, partial_acc


def cuda_decode_prefix_attention_stage1_qpair_direct_2d_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    split_m: int,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda stage1-qpair-direct-2d num_threads must be 128, 256, 512, or 1024")
    if workspace is None:
        workspace = make_prefix_attention_split_workspace(query, key_cache, split_m=split_m)
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage1_qpair_direct_2d_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return partial_m, partial_l, partial_acc


def cuda_decode_prefix_attention_stage2_gqa2(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    *,
    active_splits: int,
    out: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    if not (partial_m.is_cuda and partial_l.is_cuda and partial_acc.is_cuda):
        raise ValueError("cuda stage2 prefix attention requires CUDA tensors")
    if partial_m.dtype != torch.float32 or partial_l.dtype != torch.float32 or partial_acc.dtype != torch.float32:
        raise ValueError("stage2 partial tensors must be fp32")
    if partial_m.ndim != 2 or int(partial_m.shape[0]) != 16:
        raise ValueError("partial_m must be [16, workspace_splits]")
    if tuple(partial_l.shape) != tuple(partial_m.shape):
        raise ValueError("partial_l shape mismatch")
    workspace_splits = int(partial_m.shape[1])
    if tuple(partial_acc.shape) != (16, workspace_splits, 128):
        raise ValueError("partial_acc must be [16, workspace_splits, 128]")
    if int(active_splits) < 1 or int(active_splits) > min(workspace_splits, 16):
        raise ValueError("active_splits must be in [1, min(workspace_splits,16)]")
    if out is None:
        out = torch.empty((2048,), device=partial_m.device, dtype=torch.float16)
    out_flat = out.reshape(-1)
    if out_flat.dtype != torch.float16 or not out_flat.is_cuda or int(out_flat.numel()) != 2048:
        raise ValueError("out must be CUDA fp16 with 2048 elements")
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_stage2_gqa2(
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        out_flat.contiguous(),
        int(active_splits),
    )
    return out_flat.view_as(out)


def cuda_decode_prefix_attention_split_vgroup_atomic_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    split_m: int = 64,
    num_threads: int = 512,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    active_splits: int = 0,
    verbose: bool = False,
) -> torch.Tensor:
    """Split-prefix CUDA kernel whose last completed split merges in-kernel."""
    split_m = int(split_m)
    num_threads = int(num_threads)
    if num_threads not in (128, 256, 512, 1024):
        raise ValueError("cuda split-vgroup-atomic num_threads must be 128, 256, 512, or 1024")
    out, workspace = _validate_atomic_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        split_m,
        workspace,
    )
    partial_m, partial_l, partial_acc, counters = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_vgroup_atomic_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        float(scale),
        split_m,
        num_threads,
        int(active_splits),
    )
    return out


def cuda_decode_prefix_attention_single_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    num_threads: int = 128,
    verbose: bool = False,
) -> torch.Tensor:
    """Experimental single-kernel CUDA exact attention for bsz=1 q_len=1 GQA2."""
    num_threads = int(num_threads)
    out = _validate_single_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        num_threads,
    )
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_single_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        float(scale),
        num_threads,
    )
    return out


def cuda_decode_prefix_attention_bucketed_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    num_threads: int = 1024,
    workspace: tuple[torch.Tensor, ...] | None = None,
    prefix_len: int | None = None,
    active_splits: int | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Bucketed CUDA prefix attention used by decode graph replay.

    ``prefix_len=None`` selects the bucket from the static cache capacity and
    launches all workspace splits, which is graph-safe as cache_position grows
    during replay. Benchmarks can pass an exact prefix_len to measure only the
    currently active prefix splits.
    """
    bucket_len = int(key_cache.shape[2]) if prefix_len is None else int(prefix_len)
    split_m, kind = cuda_decode_prefix_attention_bucket_config(bucket_len)
    if workspace is None:
        workspace = make_prefix_attention_bucketed_workspace(query, key_cache, prefix_len=bucket_len)

    if active_splits is None:
        active_splits_value = 0 if prefix_len is None else (bucket_len + split_m - 1) // split_m
    else:
        active_splits_value = int(active_splits)

    if kind == "atomic":
        return cuda_decode_prefix_attention_split_vgroup_atomic_gqa2(
            query,
            key_cache,
            value_cache,
            cache_position,
            scale,
            out=out,
            output_layout=output_layout,
            split_m=split_m,
            num_threads=num_threads,
            workspace=workspace,  # type: ignore[arg-type]
            active_splits=active_splits_value,
            verbose=verbose,
        )
    return cuda_decode_prefix_attention_split_vgroup_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        scale,
        out=out,
        output_layout=output_layout,
        split_m=split_m,
        num_threads=num_threads,
        workspace=workspace,  # type: ignore[arg-type]
        active_splits=active_splits_value,
        verbose=verbose,
    )


def cuda_decode_prefix_attention_split_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    split_m: int = 128,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Experimental split-prefix CUDA exact attention for q_len=1 GQA2.

    Stage 1 computes per-prefix-split online-softmax state for each query head.
    Stage 2 merges the split states and writes the final attention output.
    """
    split_m = int(split_m)
    out, workspace = _validate_inputs(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        output_layout,
        split_m,
        workspace,
    )
    partial_m, partial_l, partial_acc = workspace
    module = load_cuda_prefix_attention_module(verbose=verbose)
    module.prefix_attention_split_gqa2(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        float(scale),
        split_m,
    )
    return out
