from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "decode_qkv_gemv.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_decode_qkv_gemv"
_SOTA_MODE = "half2-vec4-f32acc-cg-k2048"
_FP8_E4B15_MODE = "fp8e4b15-block128-f32acc-cg-k2048"
_FP8_E4B15_STATIC_MODE = "fp8e4b15-block128-f32acc-cg-static-k2048"
_FP8_E4B15_HSCALE_STATIC_MODE = "fp8e4b15-block128-hscale-cg-static-k2048"
_INT4_SYM_KBLOCK_KSCALE_MODE = "int4-sym-kblock-kscale-fullwarp-u16-static-k2048"


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
def load_cuda_decode_qkv_gemv_module(verbose: bool = False):
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


def cuda_decode_qkv_gemv(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    rows_per_block: int = 2,
    mode: str = _SOTA_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """Decode-only fp16 qkv GEMV for the packed Qwen3-VL q/k/v projection."""
    if str(mode) != _SOTA_MODE:
        raise ValueError(f"CUDA qkv GEMV only supports mode={_SOTA_MODE!r}")
    if not x.is_cuda or not qkv_weight.is_cuda:
        raise ValueError("cuda_decode_qkv_gemv requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_weight.dtype != torch.float16:
        raise ValueError("cuda_decode_qkv_gemv currently expects fp16 tensors")
    if qkv_weight.ndim != 2:
        raise ValueError("qkv_weight must be [N, 2048]")

    x_flat = x.reshape(-1).contiguous()
    n_out, k_in = int(qkv_weight.shape[0]), int(qkv_weight.shape[1])
    if int(x_flat.numel()) != 2048 or k_in != 2048:
        raise ValueError(f"CUDA qkv GEMV expects K=2048, got x={x_flat.numel()} weight_K={k_in}")

    bias_flat = None
    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != torch.float16 or int(qkv_bias.numel()) != n_out:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = qkv_bias.reshape(-1).contiguous()
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1).contiguous()
    if int(out_flat.numel()) != n_out:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {n_out}")

    rows_per_block = int(rows_per_block)
    if rows_per_block < 1 or rows_per_block > 32:
        raise ValueError("rows_per_block must be in [1, 32]")

    module = load_cuda_decode_qkv_gemv_module(verbose=verbose)
    module.qkv_gemv_half2_vec4_f32acc_cg_k2048(
        x_flat,
        qkv_weight.contiguous(),
        bias_flat,
        out_flat,
        qkv_bias is not None,
        rows_per_block,
    )
    return out_flat.view(*x.shape[:-1], n_out)


def cuda_decode_qkv_gemv_fp8(
    x: torch.Tensor,
    qkv_weight_fp8: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    rows_per_block: int = 2,
    mode: str = _FP8_E4B15_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """Decode-only fp8e4b15 qkv GEMV for the packed Qwen3-VL q/k/v projection."""
    mode = str(mode)
    if mode not in {
        _FP8_E4B15_MODE,
        _FP8_E4B15_STATIC_MODE,
        _FP8_E4B15_HSCALE_STATIC_MODE,
    }:
        raise ValueError(
            "CUDA qkv FP8 GEMV only supports "
            f"{_FP8_E4B15_MODE!r}, {_FP8_E4B15_STATIC_MODE!r}, "
            f"or {_FP8_E4B15_HSCALE_STATIC_MODE!r}"
        )
    if not x.is_cuda or not qkv_weight_fp8.is_cuda or not qkv_scales.is_cuda:
        raise ValueError("cuda_decode_qkv_gemv_fp8 requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_qkv_gemv_fp8 expects fp16 x/scales")
    if qkv_weight_fp8.dtype != torch.uint8:
        raise ValueError("qkv_weight_fp8 must be uint8 fp8e4b15 storage")
    if qkv_weight_fp8.ndim != 2 or qkv_scales.ndim != 2:
        raise ValueError("qkv_weight_fp8 must be [N, 2048] and qkv_scales [N, 16]")

    x_flat = x.reshape(-1).contiguous()
    n_out, k_in = int(qkv_weight_fp8.shape[0]), int(qkv_weight_fp8.shape[1])
    if int(x_flat.numel()) != 2048 or k_in != 2048:
        raise ValueError(f"CUDA qkv FP8 GEMV expects K=2048, got x={x_flat.numel()} weight_K={k_in}")
    if int(qkv_scales.shape[0]) != n_out or int(qkv_scales.shape[1]) != 16:
        raise ValueError("qkv_scales shape does not match qkv_weight_fp8")

    bias_flat = None
    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != torch.float16 or int(qkv_bias.numel()) != n_out:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = qkv_bias.reshape(-1).contiguous()
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1).contiguous()
    if int(out_flat.numel()) != n_out:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {n_out}")

    rows_per_block = int(rows_per_block)
    if rows_per_block < 1 or rows_per_block > 32:
        raise ValueError("rows_per_block must be in [1, 32]")

    module = load_cuda_decode_qkv_gemv_module(verbose=verbose)
    if mode == _FP8_E4B15_MODE:
        module.qkv_gemv_fp8e4b15_block128_f32acc_cg_k2048(
            x_flat,
            qkv_weight_fp8.contiguous(),
            qkv_scales.contiguous(),
            bias_flat,
            out_flat,
            qkv_bias is not None,
            rows_per_block,
        )
    elif mode == _FP8_E4B15_STATIC_MODE:
        if n_out != 4096:
            raise ValueError("static CUDA qkv FP8 GEMV expects N=4096")
        module.qkv_gemv_fp8e4b15_block128_f32acc_cg_static_k2048(
            x_flat,
            qkv_weight_fp8.contiguous(),
            qkv_scales.contiguous(),
            bias_flat,
            out_flat,
            qkv_bias is not None,
        )
    else:
        if n_out != 4096:
            raise ValueError("static CUDA qkv FP8 hscale GEMV expects N=4096")
        module.qkv_gemv_fp8e4b15_block128_hscale_cg_static_k2048(
            x_flat,
            qkv_weight_fp8.contiguous(),
            qkv_scales.contiguous(),
            bias_flat,
            out_flat,
            qkv_bias is not None,
        )
    return out_flat.view(*x.shape[:-1], n_out)


def cuda_decode_qkv_gemv_int4_sym_kblock_kscale(
    x: torch.Tensor,
    qkv_weight_int4: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    rows_per_block: int = 1,
    mode: str = _INT4_SYM_KBLOCK_KSCALE_MODE,
    verbose: bool = False,
) -> torch.Tensor:
    """Decode-only symmetric INT4 qkv GEMV for packed Qwen3-VL q/k/v projection."""
    if str(mode) != _INT4_SYM_KBLOCK_KSCALE_MODE:
        raise ValueError(f"CUDA qkv INT4 only supports mode={_INT4_SYM_KBLOCK_KSCALE_MODE!r}")
    if not x.is_cuda or not qkv_weight_int4.is_cuda or not qkv_scales.is_cuda:
        raise ValueError("cuda_decode_qkv_gemv_int4_sym_kblock_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_scales.dtype != torch.float16:
        raise ValueError("cuda_decode_qkv_gemv_int4_sym_kblock_kscale expects fp16 x/scales")
    if qkv_weight_int4.dtype != torch.uint8:
        raise ValueError("qkv_weight_int4 must be uint8 packed symmetric int4 storage")
    if qkv_weight_int4.ndim != 3 or qkv_scales.ndim != 2:
        raise ValueError("qkv_weight_int4 must be [16, N, 64] and qkv_scales [16, N]")

    x_flat = x.reshape(-1).contiguous()
    scale_blocks, n_out, packed_k = (int(v) for v in qkv_weight_int4.shape)
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"CUDA qkv INT4 expects K=2048, got x={x_flat.numel()}")
    if scale_blocks != 16 or packed_k != 64:
        raise ValueError("CUDA qkv INT4 expects qkv_weight_int4 shape [16, N, 64]")
    if int(qkv_scales.shape[0]) != 16 or int(qkv_scales.shape[1]) != n_out:
        raise ValueError("qkv_scales shape does not match qkv_weight_int4")

    bias_flat = None
    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != torch.float16 or int(qkv_bias.numel()) != n_out:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = qkv_bias.reshape(-1).contiguous()
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1).contiguous()
    if int(out_flat.numel()) != n_out:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {n_out}")

    rows_per_block = int(rows_per_block)
    if rows_per_block != 1:
        raise ValueError("CUDA qkv INT4 only supports rows_per_block=1")

    module = load_cuda_decode_qkv_gemv_module(verbose=verbose)
    module.qkv_gemv_int4_sym_kblock_kscale_k2048(
        x_flat,
        qkv_weight_int4.contiguous(),
        qkv_scales.contiguous(),
        bias_flat,
        out_flat,
        qkv_bias is not None,
        rows_per_block,
        str(mode),
    )
    return out_flat.view(*x.shape[:-1], n_out)


def cuda_decode_qkv_qk_rmsnorm_rope_kv_update_gqa2(
    qkv_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k_weight: torch.Tensor,
    k_eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    out_query: torch.Tensor | None = None,
    num_threads: int = 128,
    mode: str = "warp",
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA post-qkv decode q/k RMSNorm+RoPE and KV-cache update for Qwen3-VL GQA2."""
    if not (
        qkv_states.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and cache_position.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and cos.is_cuda
        and sin.is_cuda
    ):
        raise ValueError("cuda_decode_qkv_qk_rmsnorm_rope_kv_update_gqa2 requires CUDA tensors")
    if qkv_states.dtype != torch.float16:
        raise ValueError("qkv_states must be fp16")
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("key/value caches must be fp16")
    if q_weight.dtype != torch.float16 or k_weight.dtype != torch.float16:
        raise ValueError("q/k norm weights must be fp16")
    if cos.dtype != torch.float16 or sin.dtype != torch.float16:
        raise ValueError("cos/sin must be fp16")
    if cache_position.dtype != torch.long:
        raise ValueError("cache_position must be torch.long")

    qkv_flat = qkv_states.reshape(-1).contiguous()
    if int(qkv_flat.numel()) != 4096:
        raise ValueError(f"qkv_states must contain 4096 values, got {qkv_flat.numel()}")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key/value caches must be [B, 8, T, 128]")
    batch = int(key_cache.shape[0])
    if batch != 1:
        raise ValueError("cuda qkv qk update candidate currently supports batch size 1")
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("key/value cache shapes must match")
    if int(key_cache.shape[1]) != 8 or int(key_cache.shape[3]) != 128:
        raise ValueError("key/value cache shape must be [B, 8, T, 128]")
    if int(q_weight.numel()) != 128 or int(k_weight.numel()) != 128:
        raise ValueError("q/k norm weights must have 128 elements")

    if cos.ndim == 4:
        cos2d = cos[:, 0, -1, :]
    elif cos.ndim == 3:
        cos2d = cos[:, -1, :]
    elif cos.ndim == 2:
        cos2d = cos
    else:
        raise ValueError("cos must be decode-compatible 2D/3D/4D")
    if sin.ndim == 4:
        sin2d = sin[:, 0, -1, :]
    elif sin.ndim == 3:
        sin2d = sin[:, -1, :]
    elif sin.ndim == 2:
        sin2d = sin
    else:
        raise ValueError("sin must be decode-compatible 2D/3D/4D")
    if int(cos2d.shape[-1]) != 128 or int(sin2d.shape[-1]) != 128:
        raise ValueError("cos/sin last dim must be 128")
    if int(cos2d.shape[0]) == 1 and batch > 1:
        cos2d = cos2d.expand(batch, 128)
    if int(sin2d.shape[0]) == 1 and batch > 1:
        sin2d = sin2d.expand(batch, 128)
    if int(cos2d.shape[0]) != batch or int(sin2d.shape[0]) != batch:
        raise ValueError("cos/sin batch must match cache batch")
    cos2d = cos2d.contiguous()
    sin2d = sin2d.contiguous()

    if out_query is None:
        out_query = torch.empty((batch, 16, 128), device=qkv_states.device, dtype=torch.float16)
        return_query = out_query
    else:
        return_query = out_query
        if out_query.ndim == 4:
            if int(out_query.shape[2]) != 1:
                raise ValueError("out_query 4D decode tensor must have singleton q_len")
            out_query = out_query.squeeze(2)
        if tuple(out_query.shape) != (batch, 16, 128):
            raise ValueError("out_query must be [B, 16, 128] or [B, 16, 1, 128]")
        if out_query.dtype != torch.float16 or out_query.device != qkv_states.device:
            raise ValueError("out_query must match qkv dtype/device")

    num_threads = int(num_threads)
    mode = str(mode)
    if mode not in {
        "warp",
        "warp-pair-vvec",
        "warp-pair-vvec-contig",
        "kvquad-static-contig",
    }:
        raise ValueError(
            "mode must be 'warp', 'warp-pair-vvec', 'warp-pair-vvec-contig', "
            "or 'kvquad-static-contig'"
        )
    if mode == "kvquad-static-contig":
        if num_threads != 512:
            raise ValueError(f"{mode} mode requires num_threads=512")
    elif num_threads != 128:
        raise ValueError(f"{mode} mode requires num_threads=128")

    module = load_cuda_decode_qkv_gemv_module(verbose=verbose)
    module.qkv_qk_rmsnorm_rope_kv_update_gqa2(
        qkv_flat,
        key_cache,
        value_cache,
        cache_position.reshape(-1).contiguous(),
        q_weight.reshape(-1).contiguous(),
        float(q_eps),
        k_weight.reshape(-1).contiguous(),
        float(k_eps),
        cos2d,
        sin2d,
        out_query,
        num_threads,
        mode,
    )
    return return_query


def cuda_decode_qkv_fp8_hscale_qk_rmsnorm_rope_kv_update_gqa2(
    x: torch.Tensor,
    qkv_weight_fp8: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_bias: torch.Tensor | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k_weight: torch.Tensor,
    k_eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    out_query: torch.Tensor | None = None,
    qk_scratch: torch.Tensor | None = None,
    qk_counters: torch.Tensor | None = None,
    variant: str = "row",
    verbose: bool = False,
) -> torch.Tensor:
    """Fused decode FP8 qkv GEMV plus q/k RMSNorm+RoPE and KV update.

    This is a fixed-shape Qwen3-VL candidate for the qkv+qk-update fusion goal.
    It intentionally supports only the current SOTA hscale-static FP8 layout.
    """
    if not (
        x.is_cuda
        and qkv_weight_fp8.is_cuda
        and qkv_scales.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and cache_position.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and cos.is_cuda
        and sin.is_cuda
    ):
        raise ValueError("fused CUDA qkv/qk update requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_scales.dtype != torch.float16:
        raise ValueError("fused CUDA qkv/qk update expects fp16 x/scales")
    if qkv_weight_fp8.dtype != torch.uint8:
        raise ValueError("qkv_weight_fp8 must be uint8 fp8e4b15 storage")
    if q_weight.dtype != torch.float16 or k_weight.dtype != torch.float16:
        raise ValueError("q/k norm weights must be fp16")
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("key/value cache tensors must be fp16")
    if cache_position.dtype != torch.long:
        raise ValueError("cache_position must be torch.long")

    x_flat = x.reshape(-1).contiguous()
    if int(x_flat.numel()) != 2048:
        raise ValueError(f"fused CUDA qkv/qk update expects K=2048, got x={x_flat.numel()}")
    if tuple(qkv_weight_fp8.shape) != (4096, 2048):
        raise ValueError("qkv_weight_fp8 must be [4096, 2048]")
    if tuple(qkv_scales.shape) != (4096, 16):
        raise ValueError("qkv_scales must be [4096, 16]")
    if key_cache.ndim != 4 or value_cache.ndim != 4 or tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("key/value cache tensors must be matching [1, 8, T, 128]")
    if int(key_cache.shape[0]) != 1 or int(key_cache.shape[1]) != 8 or int(key_cache.shape[3]) != 128:
        raise ValueError("key/value cache tensors must be [1, 8, T, 128]")
    if int(q_weight.numel()) != 128 or int(k_weight.numel()) != 128:
        raise ValueError("q/k norm weights must have 128 elements")

    if cos.ndim == 4:
        cos2d = cos[:, 0, -1, :]
    elif cos.ndim == 3:
        cos2d = cos[:, -1, :]
    elif cos.ndim == 2:
        cos2d = cos
    else:
        raise ValueError("cos must be decode-compatible 2D/3D/4D")
    if sin.ndim == 4:
        sin2d = sin[:, 0, -1, :]
    elif sin.ndim == 3:
        sin2d = sin[:, -1, :]
    elif sin.ndim == 2:
        sin2d = sin
    else:
        raise ValueError("sin must be decode-compatible 2D/3D/4D")
    if tuple(cos2d.shape) != (1, 128) or tuple(sin2d.shape) != (1, 128):
        raise ValueError("cos/sin must resolve to [1, 128]")
    cos2d = cos2d.contiguous()
    sin2d = sin2d.contiguous()

    bias_flat = None
    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != torch.float16 or int(qkv_bias.numel()) != 4096:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with 4096 values")
        bias_flat = qkv_bias.reshape(-1).contiguous()
    else:
        bias_flat = x_flat

    if out_query is None:
        out_query = torch.empty((1, 16, 128), device=x.device, dtype=torch.float16)
        return_query = out_query
    else:
        return_query = out_query
        if out_query.ndim == 4:
            if tuple(out_query.shape) != (1, 16, 1, 128):
                raise ValueError("out_query 4D tensor must be [1, 16, 1, 128]")
            out_query = out_query.squeeze(2)
        if tuple(out_query.shape) != (1, 16, 128):
            raise ValueError("out_query must be [1, 16, 128] or [1, 16, 1, 128]")
        if out_query.dtype != torch.float16 or out_query.device != x.device:
            raise ValueError("out_query must match x dtype/device")

    if qk_scratch is None:
        qk_scratch = torch.empty((24, 128), device=x.device, dtype=torch.float16)
    if qk_counters is None:
        qk_counters = torch.empty((24,), device=x.device, dtype=torch.int32)
        qk_counters.zero_()
    if tuple(qk_scratch.shape) != (24, 128) or qk_scratch.dtype != torch.float16 or qk_scratch.device != x.device:
        raise ValueError("qk_scratch must be fp16 CUDA tensor [24, 128]")
    if tuple(qk_counters.shape) != (24,) or qk_counters.dtype != torch.int32 or qk_counters.device != x.device:
        raise ValueError("qk_counters must be int32 CUDA tensor [24]")

    module = load_cuda_decode_qkv_gemv_module(verbose=verbose)
    variant = str(variant)
    if variant == "row":
        fn = module.qkv_fp8e4b15_hscale_static_qk_update_gqa2
    elif variant == "chunk16":
        fn = module.qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk16
    elif variant == "chunk32":
        fn = module.qkv_fp8e4b15_hscale_static_qk_update_gqa2_chunk32
    else:
        raise ValueError("variant must be 'row', 'chunk16', or 'chunk32'")
    fn(
        x_flat,
        qkv_weight_fp8.contiguous(),
        qkv_scales.contiguous(),
        bias_flat,
        qkv_bias is not None,
        key_cache,
        value_cache,
        cache_position.reshape(-1).contiguous(),
        q_weight.reshape(-1).contiguous(),
        float(q_eps),
        k_weight.reshape(-1).contiguous(),
        float(k_eps),
        cos2d,
        sin2d,
        out_query,
        qk_scratch,
        qk_counters,
    )
    return return_query


def cuda_decode_qkv_qk_rmsnorm_rope_cache_kv_update_gqa2(
    qkv_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k_weight: torch.Tensor,
    k_eps: float,
    rope_cache: torch.Tensor,
    rope_position: torch.Tensor | None = None,
    *,
    out_query: torch.Tensor | None = None,
    num_threads: int = 512,
    mode: str = "kvquad-static-contig",
    verbose: bool = False,
) -> torch.Tensor:
    """CUDA post-qkv decode q/k RMSNorm+RoPE using a precomputed RoPE table."""
    if rope_position is None:
        rope_position = cache_position
    if not (
        qkv_states.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and cache_position.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and rope_cache.is_cuda
        and rope_position.is_cuda
    ):
        raise ValueError("cuda_decode_qkv_qk_rmsnorm_rope_cache_kv_update_gqa2 requires CUDA tensors")
    if qkv_states.dtype != torch.float16:
        raise ValueError("qkv_states must be fp16")
    if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("key/value caches must be fp16")
    if q_weight.dtype != torch.float16 or k_weight.dtype != torch.float16:
        raise ValueError("q/k norm weights must be fp16")
    if rope_cache.dtype != torch.float16:
        raise ValueError("rope_cache must be fp16")
    if cache_position.dtype != torch.long:
        raise ValueError("cache_position must be torch.long")
    if rope_position.dtype != torch.long:
        raise ValueError("rope_position must be torch.long")

    qkv_flat = qkv_states.reshape(-1).contiguous()
    if int(qkv_flat.numel()) != 4096:
        raise ValueError(f"qkv_states must contain 4096 values, got {qkv_flat.numel()}")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key/value caches must be [B, 8, T, 128]")
    batch = int(key_cache.shape[0])
    if batch != 1:
        raise ValueError("cuda qkv qk update candidate currently supports batch size 1")
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("key/value cache shapes must match")
    if int(key_cache.shape[1]) != 8 or int(key_cache.shape[3]) != 128:
        raise ValueError("key/value cache shape must be [B, 8, T, 128]")
    if int(q_weight.numel()) != 128 or int(k_weight.numel()) != 128:
        raise ValueError("q/k norm weights must have 128 elements")
    if rope_cache.ndim != 3 or tuple(rope_cache.shape[1:]) != (2, 128):
        raise ValueError("rope_cache must be [T, 2, 128]")

    if out_query is None:
        out_query = torch.empty((batch, 16, 128), device=qkv_states.device, dtype=torch.float16)
        return_query = out_query
    else:
        return_query = out_query
        if out_query.ndim == 4:
            if int(out_query.shape[2]) != 1:
                raise ValueError("out_query 4D decode tensor must have singleton q_len")
            out_query = out_query.squeeze(2)
        if tuple(out_query.shape) != (batch, 16, 128):
            raise ValueError("out_query must be [B, 16, 128] or [B, 16, 1, 128]")
        if out_query.dtype != torch.float16 or out_query.device != qkv_states.device:
            raise ValueError("out_query must match qkv dtype/device")

    num_threads = int(num_threads)
    mode = str(mode)
    if mode != "kvquad-static-contig":
        raise ValueError("rope-cache qkv qk update supports only mode='kvquad-static-contig'")
    if num_threads != 512:
        raise ValueError("rope-cache qkv qk update requires num_threads=512")

    module = load_cuda_decode_qkv_gemv_module(verbose=verbose)
    module.qkv_qk_rmsnorm_rope_cache_kv_update_gqa2(
        qkv_flat,
        key_cache,
        value_cache,
        cache_position.reshape(-1).contiguous(),
        q_weight.reshape(-1).contiguous(),
        float(q_eps),
        k_weight.reshape(-1).contiguous(),
        float(k_eps),
        rope_cache.contiguous(),
        rope_position.reshape(-1).contiguous(),
        out_query,
        num_threads,
        mode,
    )
    return return_query
