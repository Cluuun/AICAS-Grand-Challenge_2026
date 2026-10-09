from __future__ import annotations

"""Qwen3-VL vision encoder 算子优化：patch embed GEMM、位置/rope 缓存、可选 MLP fast path."""

from collections import OrderedDict
from types import MethodType
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conf import conf_bool, conf_int, conf_str

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_BACKEND_STATS: Dict[str, int] = {
    "patch_embed_cublaslt": 0,
    "patch_embed_cublaslt_fallback": 0,
    "patch_embed_triton": 0,
    "patch_embed_triton_fallback": 0,
    "patch_embed_linear": 0,
    "patch_embed_mm_out": 0,
    "patch_embed_matmul": 0,
    "patch_embed_fallback": 0,
    "pos_add_ln0_triton": 0,
    "pos_add_ln0_fallback": 0,
    "pos_interp_indexed": 0,
    "pos_interp_indexed_fallback": 0,
    "pos_interp_triton": 0,
    "pos_interp_triton_fallback": 0,
    "vision_linear_cublaslt": 0,
    "vision_linear_cublaslt_fallback": 0,
    "vision_linear_mm_out": 0,
    "vision_linear_fallback": 0,
    "cu_seqlens_host": 0,
    "cu_seqlens_torch": 0,
    "pos_cache_hit": 0,
    "pos_cache_miss": 0,
    "rope_cache_hit": 0,
    "rope_cache_miss": 0,
    "vision_rope_triton": 0,
    "vision_rope_triton_fallback": 0,
    "vision_forward_fast": 0,
    "vision_forward_fallback": 0,
    "vision_mlp_fast": 0,
    "vision_gelu_functional": 0,
    "fixed_vision_shape_key_hit": 0,
    "fixed_vision_shape_key_fallback": 0,
    "vision_forward_prewarm": 0,
}

_CUBLASLT_MATMUL_BIAS = None
_CUBLASLT_IMPORT_TRIED = False


def _stat_inc(name: str) -> None:
    _BACKEND_STATS[name] = int(_BACKEND_STATS.get(name, 0)) + 1


def _env_on(name: str, default: str = "1") -> bool:
    return conf_bool(name, default)


def _env_str(name: str, default: str) -> str:
    return conf_str(name, default, lower=True)


def _parse_fixed_resolution() -> Tuple[int, int]:
    text = conf_str("FIXED_IMAGE_RESOLUTION", conf_str("VISION_FIXED_RESOLUTION", "336x336"), lower=True)
    text = text.replace("*", "x")
    if "x" not in text:
        return 336, 336
    left, right = text.split("x", 1)
    try:
        return max(1, int(left)), max(1, int(right))
    except Exception:
        return 336, 336


def _fixed_vision_shape_enabled() -> bool:
    if not _env_on("ENABLE_FIXED_VISION_SHAPE_CACHE", "1"):
        return False
    if not _env_on("ENABLE_FIXED_336_FASTPATH", "1"):
        return False
    return _parse_fixed_resolution() == (336, 336)


def _confirm_fixed_grid_key(visual, grid_thw: torch.Tensor) -> Tuple:
    """预热阶段一次性确认 grid_thw，推理路径只读常量键。"""
    cached = getattr(visual, "_fixed_vision_grid_key", None)
    if cached is not None:
        return cached
    key = tuple(int(x) for x in grid_thw.reshape(-1).detach().cpu().tolist())
    visual._fixed_vision_grid_key = key
    print(f"[vision_kernel_fusions] fixed_336 grid key confirmed: {key}")
    return key


def _resolve_grid_thw_key(visual, grid_thw: torch.Tensor) -> Tuple:
    if _fixed_vision_shape_enabled() and visual is not None:
        cached = getattr(visual, "_fixed_vision_grid_key", None)
        if cached is not None:
            _stat_inc("fixed_vision_shape_key_hit")
            return cached
        _stat_inc("fixed_vision_shape_key_fallback")
        # 首次在 eager 路径确认并缓存固定 key；后续（含 CUDA Graph capture/replay）只读常量，避免 cpu()/tolist() 同步。
        key = tuple(int(x) for x in grid_thw.reshape(-1).detach().cpu().tolist())
        visual._fixed_vision_grid_key = key
        return key
    return tuple(int(x) for x in grid_thw.reshape(-1).detach().cpu().tolist())


def _build_patch_linear_weight(conv: nn.Conv3d) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Conv3d(kernel=stride) -> [out_features, in_features] for F.linear."""
    w = conv.weight
    out_features = w.shape[0]
    in_features = w.numel() // out_features
    weight_2d = w.reshape(out_features, in_features).contiguous()
    bias = conv.bias
    if bias is not None:
        bias = bias.contiguous()
    return weight_2d, bias


def _get_patch_mm_workspace(
    patch_embed,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (rows, cols, str(dtype), str(device))
    ws = getattr(patch_embed, "_vision_patch_mm_workspace", None)
    if getattr(patch_embed, "_vision_patch_mm_workspace_key", None) == key and ws is not None:
        return ws
    ws = torch.empty((rows, cols), dtype=dtype, device=device)
    patch_embed._vision_patch_mm_workspace_key = key
    patch_embed._vision_patch_mm_workspace = ws
    return ws


def _get_linear_weight_t(linear: nn.Linear) -> torch.Tensor:
    key = linear.weight.data_ptr()
    w_t = getattr(linear, "_vision_linear_weight_t", None)
    if getattr(linear, "_vision_linear_weight_t_key", None) == key and w_t is not None:
        return w_t
    w_t = linear.weight.transpose(0, 1).contiguous()
    linear._vision_linear_weight_t_key = key
    linear._vision_linear_weight_t = w_t
    return w_t


def _get_linear_workspace(
    linear: nn.Linear,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (rows, cols, str(dtype), str(device))
    ws = getattr(linear, "_vision_linear_workspace", None)
    if getattr(linear, "_vision_linear_workspace_key", None) == key and ws is not None:
        return ws
    ws = torch.empty((rows, cols), dtype=dtype, device=device)
    linear._vision_linear_workspace_key = key
    linear._vision_linear_workspace = ws
    return ws


def _cublaslt_matmul_bias_out(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
) -> Optional[torch.Tensor]:
    global _CUBLASLT_IMPORT_TRIED, _CUBLASLT_MATMUL_BIAS
    if not _env_on("ENABLE_VISION_CUBLASLT_EPILOGUE", "1"):
        return None
    try:
        if not _CUBLASLT_IMPORT_TRIED:
            _CUBLASLT_IMPORT_TRIED = True
            from .cublaslt_epilogue import cublaslt_matmul_bias

            _CUBLASLT_MATMUL_BIAS = cublaslt_matmul_bias
        if _CUBLASLT_MATMUL_BIAS is None:
            return None
        return _CUBLASLT_MATMUL_BIAS(a, b, bias, out)
    except Exception:
        return None


def _linear_fast_out(linear: nn.Linear, x2d: torch.Tensor) -> torch.Tensor:
    rows = int(x2d.shape[0])
    out_features = int(linear.out_features)
    out = _get_linear_workspace(linear, rows, out_features, x2d.dtype, x2d.device)
    if _env_on("ENABLE_VISION_LINEAR_CUBLASLT", "1"):
        cublaslt_out = _cublaslt_matmul_bias_out(x2d, _get_linear_weight_t(linear), linear.bias, out)
        if cublaslt_out is not None:
            _stat_inc("vision_linear_cublaslt")
            return cublaslt_out
        _stat_inc("vision_linear_cublaslt_fallback")

    torch.mm(x2d, _get_linear_weight_t(linear), out=out)
    if linear.bias is not None:
        out.add_(linear.bias)
    _stat_inc("vision_linear_mm_out")
    return out


def _vision_linear_mm_out_forward(
    linear: nn.Linear,
    x: torch.Tensor,
    *,
    _original_forward: Callable,
) -> torch.Tensor:
    if (
        not _env_on("ENABLE_VISION_LINEAR_MM_OUT", "0")
        or torch.is_grad_enabled()
        or not x.is_cuda
        or x.dim() != 2
    ):
        return _original_forward(x)
    try:
        x2d = x.contiguous() if not x.is_contiguous() else x
        return _linear_fast_out(linear, x2d)
    except Exception:
        _stat_inc("vision_linear_fallback")
        return _original_forward(x)


if _TRITON_AVAILABLE:

    @triton.jit
    def _patch_embed_matmul_bias_kernel(
        a,
        b,
        bias,
        out,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            k_idxs = k0 + offs_k
            a_vals = tl.load(
                a + offs_m[:, None] * k + k_idxs[None, :],
                mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
                other=0.0,
            )
            b_vals = tl.load(
                b + k_idxs[:, None] * n + offs_n[None, :],
                mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(a_vals, b_vals, out_dtype=tl.float32)

        if has_bias:
            bias_vals = tl.load(bias + offs_n, mask=offs_n < n, other=0.0).to(tl.float32)
            acc += bias_vals[None, :]

        tl.store(
            out + offs_m[:, None] * n + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )

    @triton.jit
    def _pos_add_layernorm_kernel(
        x,
        pos,
        weight,
        bias,
        out_x,
        out_norm,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        base = row * n_cols + offs

        vals = (
            tl.load(x + base, mask=mask, other=0.0).to(tl.float32)
            + tl.load(pos + base, mask=mask, other=0.0).to(tl.float32)
        )
        mean = tl.sum(vals, axis=0) / n_cols
        centered = vals - mean
        var = tl.sum(centered * centered, axis=0) / n_cols
        inv = tl.rsqrt(var + eps)
        w = tl.load(weight + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias + offs, mask=mask, other=0.0).to(tl.float32)
        normed = centered * inv * w + b

        tl.store(out_x + base, vals, mask=mask)
        tl.store(out_norm + base, normed, mask=mask)

    @triton.jit
    def _pos_interp_single_image_kernel(
        pos_weight,
        out,
        total_tokens: tl.constexpr,
        hidden: tl.constexpr,
        t: tl.constexpr,
        h: tl.constexpr,
        w: tl.constexpr,
        grid_side: tl.constexpr,
        merge_size: tl.constexpr,
        h_scale: tl.constexpr,
        w_scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

        spatial = offs_m % (h * w)
        intra_w = spatial % merge_size
        tmp = spatial // merge_size
        intra_h = tmp % merge_size
        tmp = tmp // merge_size
        block_w = tmp % (w // merge_size)
        block_h = tmp // (w // merge_size)
        row = block_h * merge_size + intra_h
        col = block_w * merge_size + intra_w

        h_pos = row.to(tl.float32) * h_scale
        w_pos = col.to(tl.float32) * w_scale
        h_floor = tl.floor(h_pos).to(tl.int32)
        w_floor = tl.floor(w_pos).to(tl.int32)
        h_ceil = tl.minimum(h_floor + 1, grid_side - 1)
        w_ceil = tl.minimum(w_floor + 1, grid_side - 1)
        dh = h_pos - h_floor.to(tl.float32)
        dw = w_pos - w_floor.to(tl.float32)

        idx00 = h_floor * grid_side + w_floor
        idx01 = h_floor * grid_side + w_ceil
        idx10 = h_ceil * grid_side + w_floor
        idx11 = h_ceil * grid_side + w_ceil
        w00 = (1.0 - dh) * (1.0 - dw)
        w01 = (1.0 - dh) * dw
        w10 = dh * (1.0 - dw)
        w11 = dh * dw

        mask = (offs_m[:, None] < total_tokens) & (offs_c[None, :] < hidden)
        vals = (
            tl.load(pos_weight + idx00[:, None] * hidden + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            * w00[:, None]
            + tl.load(pos_weight + idx01[:, None] * hidden + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            * w01[:, None]
            + tl.load(pos_weight + idx10[:, None] * hidden + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            * w10[:, None]
            + tl.load(pos_weight + idx11[:, None] * hidden + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            * w11[:, None]
        )
        tl.store(out + offs_m[:, None] * hidden + offs_c[None, :], vals, mask=mask)

    @triton.jit
    def _vision_rope_qk_kernel(
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        seq_len,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        q_stride_s: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_s: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = seq_len * num_heads * head_dim
        mask = offs < total
        d = offs % head_dim
        tmp = offs // head_dim
        h = tmp % num_heads
        s = tmp // num_heads
        half = head_dim // 2
        pair_d = tl.where(d < half, d + half, d - half)
        sign = tl.where(d < half, -1.0, 1.0)

        q_base = s * q_stride_s + h * q_stride_h
        k_base = s * k_stride_s + h * k_stride_h
        q_self = tl.load(q + q_base + d * q_stride_d, mask=mask, other=0.0).to(tl.float32)
        q_pair = tl.load(q + q_base + pair_d * q_stride_d, mask=mask, other=0.0).to(tl.float32)
        k_self = tl.load(k + k_base + d * k_stride_d, mask=mask, other=0.0).to(tl.float32)
        k_pair = tl.load(k + k_base + pair_d * k_stride_d, mask=mask, other=0.0).to(tl.float32)
        cos_v = tl.load(cos + s * head_dim + d, mask=mask, other=0.0).to(tl.float32)
        sin_v = tl.load(sin + s * head_dim + d, mask=mask, other=0.0).to(tl.float32)

        q_rot = q_self * cos_v + sign * q_pair * sin_v
        k_rot = k_self * cos_v + sign * k_pair * sin_v
        tl.store(q_out + offs, q_rot, mask=mask)
        tl.store(k_out + offs, k_rot, mask=mask)


def _patch_embed_triton_matmul(
    patch_embed,
    flat_in: torch.Tensor,
    weight_t: torch.Tensor,
    bias: Optional[torch.Tensor],
    embed_dim: int,
) -> Optional[torch.Tensor]:
    if (
        not _TRITON_AVAILABLE
        or not flat_in.is_cuda
        or not flat_in.is_contiguous()
        or not weight_t.is_cuda
        or not weight_t.is_contiguous()
        or torch.is_grad_enabled()
    ):
        return None

    rows = int(flat_in.shape[0])
    in_features = int(flat_in.shape[1])
    if rows <= 0 or in_features <= 0 or embed_dim <= 0:
        return None

    out = _get_patch_mm_workspace(patch_embed, rows, embed_dim, flat_in.dtype, flat_in.device)
    has_bias = bias is not None and bias.is_cuda

    # PatchEmbed 的典型形状约 M<=几千、K=1176、N=1152；32x64 tile 在 PPU/GPU 上启动成本和占用较均衡。
    block_m = 32
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(rows, block_m), triton.cdiv(embed_dim, block_n))
    _patch_embed_matmul_bias_kernel[grid](
        flat_in,
        weight_t,
        bias if has_bias else flat_in,
        out,
        rows,
        embed_dim,
        in_features,
        has_bias,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def _triton_pos_add_layernorm0(
    hidden_states: torch.Tensor,
    pos_embeds: torch.Tensor,
    norm_module: nn.LayerNorm,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if (
        not _TRITON_AVAILABLE
        or not _env_on("ENABLE_VISION_FUSED_POS_LN0", "0")
        or torch.is_grad_enabled()
        or not hidden_states.is_cuda
        or not pos_embeds.is_cuda
        or not hidden_states.is_contiguous()
        or not pos_embeds.is_contiguous()
        or hidden_states.dim() != 2
        or pos_embeds.shape != hidden_states.shape
        or not hasattr(norm_module, "weight")
        or not hasattr(norm_module, "bias")
    ):
        return None

    n_cols = int(hidden_states.shape[-1])
    if n_cols <= 0 or n_cols > 4096:
        return None

    out_norm = torch.empty_like(hidden_states)
    block_n = triton.next_power_of_2(n_cols)
    _pos_add_layernorm_kernel[(hidden_states.shape[0],)](
        hidden_states,
        pos_embeds,
        norm_module.weight,
        norm_module.bias,
        hidden_states,
        out_norm,
        n_cols,
        float(getattr(norm_module, "eps", 1e-6)),
        BLOCK_N=block_n,
        num_warps=8,
    )
    _stat_inc("pos_add_ln0_triton")
    return hidden_states, out_norm


def _patch_embed_linear_forward(
    patch_embed,
    hidden_states: torch.Tensor,
    *,
    _original_forward: Callable,
) -> torch.Tensor:
    conv = patch_embed.proj
    tps = patch_embed.temporal_patch_size
    ps = patch_embed.patch_size
    in_ch = patch_embed.in_channels
    embed_dim = patch_embed.embed_dim
    target_dtype = conv.weight.dtype

    raw = hidden_states
    hidden_states = hidden_states.view(-1, in_ch, tps, ps, ps)
    num_patches = hidden_states.shape[0]
    flat_in = hidden_states.reshape(num_patches, in_ch * tps * ps * ps).to(dtype=target_dtype)

    weight_2d = getattr(patch_embed, "_vision_patch_linear_weight", None)
    bias = getattr(patch_embed, "_vision_patch_linear_bias", None)
    weight_t = getattr(patch_embed, "_vision_patch_linear_weight_t", None)
    if weight_2d is None:
        weight_2d, bias = _build_patch_linear_weight(conv)
        patch_embed._vision_patch_linear_weight = weight_2d
        patch_embed._vision_patch_linear_bias = bias
    if weight_t is None:
        weight_t = weight_2d.transpose(0, 1).contiguous()
        patch_embed._vision_patch_linear_weight_t = weight_t

    try:
        backend = _env_str("VISION_PATCH_EMBED_BACKEND", "mm_out")
        if backend == "cublaslt" and not torch.is_grad_enabled() and flat_in.is_cuda:
            out = _get_patch_mm_workspace(patch_embed, num_patches, embed_dim, flat_in.dtype, flat_in.device)
            cublaslt_out = _cublaslt_matmul_bias_out(flat_in, weight_t, bias, out)
            if cublaslt_out is not None:
                _stat_inc("patch_embed_cublaslt")
                return cublaslt_out.view(-1, embed_dim)
            _stat_inc("patch_embed_cublaslt_fallback")
            backend = "mm_out"

        if backend == "triton":
            out = _patch_embed_triton_matmul(patch_embed, flat_in, weight_t, bias, embed_dim)
            if out is not None:
                _stat_inc("patch_embed_triton")
                return out.view(-1, embed_dim)
            else:
                _stat_inc("patch_embed_triton_fallback")
                backend = "mm_out"

        if backend == "mm_out" and not torch.is_grad_enabled() and flat_in.is_cuda:
            out = _get_patch_mm_workspace(patch_embed, num_patches, embed_dim, flat_in.dtype, flat_in.device)
            torch.mm(flat_in, weight_t, out=out)
            if bias is not None:
                out.add_(bias)
            _stat_inc("patch_embed_mm_out")
        elif backend == "matmul":
            out = flat_in @ weight_t
            if bias is not None:
                out = out + bias
            _stat_inc("patch_embed_matmul")
        else:
            out = F.linear(flat_in, weight_2d, bias)
            _stat_inc("patch_embed_linear")
        return out.view(-1, embed_dim)
    except Exception:
        _stat_inc("patch_embed_fallback")
        return _original_forward(raw)


def _apply_vision_rope_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if (
        not _TRITON_AVAILABLE
        or not _env_on("ENABLE_VISION_ROPE_TRITON", "1")
        or torch.is_grad_enabled()
        or query_states.dim() != 3
        or key_states.dim() != 3
        or query_states.shape != key_states.shape
        or not query_states.is_cuda
        or not key_states.is_cuda
    ):
        return None

    try:
        cos, sin = position_embeddings
        seq_len = int(query_states.shape[0])
        num_heads = int(query_states.shape[1])
        head_dim = int(query_states.shape[2])
        if (
            head_dim <= 0
            or head_dim % 2 != 0
            or cos.shape != (seq_len, head_dim)
            or sin.shape != (seq_len, head_dim)
            or not cos.is_cuda
            or not sin.is_cuda
        ):
            return None
        cos = cos.contiguous() if not cos.is_contiguous() else cos
        sin = sin.contiguous() if not sin.is_contiguous() else sin
        q_out = torch.empty((seq_len, num_heads, head_dim), dtype=query_states.dtype, device=query_states.device)
        k_out = torch.empty((seq_len, num_heads, head_dim), dtype=key_states.dtype, device=key_states.device)
        block = 256
        total = seq_len * num_heads * head_dim
        _vision_rope_qk_kernel[(triton.cdiv(total, block),)](
            query_states,
            key_states,
            cos,
            sin,
            q_out,
            k_out,
            seq_len,
            num_heads,
            head_dim,
            int(query_states.stride(0)),
            int(query_states.stride(1)),
            int(query_states.stride(2)),
            int(key_states.stride(0)),
            int(key_states.stride(1)),
            int(key_states.stride(2)),
            BLOCK=block,
            num_warps=4,
        )
        _stat_inc("vision_rope_triton")
        return q_out, k_out
    except Exception:
        _stat_inc("vision_rope_triton_fallback")
        return None


def _vision_attention_fast_forward(
    attn,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *,
    _original_forward: Callable,
    **kwargs,
) -> torch.Tensor:
    if position_embeddings is None or not _env_on("ENABLE_VISION_ROPE_TRITON", "1"):
        return _original_forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    try:
        qkv = attn.qkv(hidden_states)
        seq_length = int(hidden_states.shape[0])
        query_states, key_states, value_states = (
            qkv.reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        rotated = _apply_vision_rope_triton(query_states, key_states, position_embeddings)
        if rotated is None:
            return _original_forward(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        query_states, key_states = rotated

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

        attention_interface: Callable = qwen3_vl.eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attention_interface = qwen3_vl.ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

        if attn.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                attn,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=attn.scaling,
                dropout=0.0 if not attn.training else attn.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    attn,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=attn.scaling,
                    dropout=0.0 if not attn.training else attn.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = attn.proj(attn_output)
        return attn_output
    except Exception:
        _stat_inc("vision_rope_triton_fallback")
        return _original_forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )


def _first_block_with_pre_norm(
    blk,
    hidden_states: torch.Tensor,
    norm1_hidden: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    hidden_states = hidden_states + blk.attn(
        norm1_hidden,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))
    return hidden_states


def _get_pos_cache(visual) -> "OrderedDict":
    cache = getattr(visual, "_vision_pos_embed_cache", None)
    if cache is None:
        cache = OrderedDict()
        visual._vision_pos_embed_cache = cache
    return cache


def _get_pos_index_cache(visual) -> "OrderedDict":
    cache = getattr(visual, "_vision_pos_index_cache", None)
    if cache is None:
        cache = OrderedDict()
        visual._vision_pos_index_cache = cache
    return cache


def _get_rope_cache(visual) -> "OrderedDict":
    cache = getattr(visual, "_vision_rope_cache", None)
    if cache is None:
        cache = OrderedDict()
        visual._vision_rope_cache = cache
    return cache


def _pos_cache_key(visual, grid_thw: torch.Tensor) -> Tuple:
    return (
        _resolve_grid_thw_key(visual, grid_thw),
        str(visual.pos_embed.weight.dtype),
        str(visual.pos_embed.weight.device),
        int(visual.pos_embed.weight.data_ptr()),
    )


def _rope_cache_key(visual, grid_thw: torch.Tensor) -> Tuple:
    return (
        _resolve_grid_thw_key(visual, grid_thw),
        str(grid_thw.dtype),
        str(grid_thw.device),
    )


def _lru_get(cache: OrderedDict, key: Tuple, max_entries: int):
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def _lru_put(cache: OrderedDict, key: Tuple, value: Any, max_entries: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _fast_pos_embed_interpolate_triton(visual, grid_thw: torch.Tensor) -> Optional[torch.Tensor]:
    if (
        not _TRITON_AVAILABLE
        or not _env_on("ENABLE_VISION_POS_TRITON", "1")
        or torch.is_grad_enabled()
        or grid_thw.dim() != 2
        or int(grid_thw.shape[0]) != 1
    ):
        return None

    pos_weight = visual.pos_embed.weight
    if not pos_weight.is_cuda or pos_weight.dim() != 2 or not pos_weight.is_contiguous():
        return None

    try:
        grid_key = _resolve_grid_thw_key(visual, grid_thw)
        t, h, w = int(grid_key[0]), int(grid_key[1]), int(grid_key[2])
        merge_size = int(getattr(visual.config, "spatial_merge_size", visual.spatial_merge_size))
        grid_side = int(visual.num_grid_per_side)
        hidden = int(pos_weight.shape[-1])
        if (
            t <= 0
            or h <= 0
            or w <= 0
            or hidden <= 0
            or merge_size <= 0
            or h % merge_size != 0
            or w % merge_size != 0
        ):
            return None
        total_tokens = t * h * w
        out = torch.empty((total_tokens, hidden), dtype=pos_weight.dtype, device=pos_weight.device)
        block_m = 16
        block_c = 64
        grid = (triton.cdiv(total_tokens, block_m), triton.cdiv(hidden, block_c))
        _pos_interp_single_image_kernel[grid](
            pos_weight,
            out,
            total_tokens,
            hidden,
            t,
            h,
            w,
            grid_side,
            merge_size,
            float(grid_side - 1) / float(max(h - 1, 1)),
            float(grid_side - 1) / float(max(w - 1, 1)),
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            num_warps=4,
        )
        _stat_inc("pos_interp_triton")
        return out
    except Exception:
        _stat_inc("pos_interp_triton_fallback")
        return None


def _build_pos_interp_indices_for_key(
    visual,
    grid_key: Tuple,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_side = int(visual.num_grid_per_side)
    merge_size = int(getattr(visual.config, "spatial_merge_size", visual.spatial_merge_size))
    idx_rows = [[], [], [], []]
    weight_rows = [[], [], [], []]

    for i in range(0, len(grid_key), 3):
        t, h, w = int(grid_key[i]), int(grid_key[i + 1]), int(grid_key[i + 2])
        if t <= 0 or h <= 0 or w <= 0 or h % merge_size != 0 or w % merge_size != 0:
            raise ValueError("invalid grid_thw for position interpolation")
        h_scale = float(grid_side - 1) / float(max(h - 1, 1))
        w_scale = float(grid_side - 1) / float(max(w - 1, 1))
        for _frame in range(t):
            for block_h in range(h // merge_size):
                for block_w in range(w // merge_size):
                    for intra_h in range(merge_size):
                        row = block_h * merge_size + intra_h
                        h_pos = row * h_scale
                        h_floor = int(h_pos)
                        h_ceil = min(h_floor + 1, grid_side - 1)
                        dh = h_pos - h_floor
                        base_h = h_floor * grid_side
                        base_h_ceil = h_ceil * grid_side
                        for intra_w in range(merge_size):
                            col = block_w * merge_size + intra_w
                            w_pos = col * w_scale
                            w_floor = int(w_pos)
                            w_ceil = min(w_floor + 1, grid_side - 1)
                            dw = w_pos - w_floor
                            idx_rows[0].append(base_h + w_floor)
                            idx_rows[1].append(base_h + w_ceil)
                            idx_rows[2].append(base_h_ceil + w_floor)
                            idx_rows[3].append(base_h_ceil + w_ceil)
                            weight_rows[0].append((1.0 - dh) * (1.0 - dw))
                            weight_rows[1].append((1.0 - dh) * dw)
                            weight_rows[2].append(dh * (1.0 - dw))
                            weight_rows[3].append(dh * dw)

    idx = torch.tensor(idx_rows, dtype=torch.long, device=device)
    weights = torch.tensor(weight_rows, dtype=dtype, device=device)
    return idx, weights


def _fast_pos_embed_interpolate_indexed(visual, grid_thw: torch.Tensor) -> Optional[torch.Tensor]:
    if (
        not _env_on("ENABLE_VISION_POS_INDEX_CACHE", "1")
        or torch.is_grad_enabled()
        or grid_thw.dim() != 2
    ):
        return None

    pos_weight = visual.pos_embed.weight
    if not pos_weight.is_cuda or pos_weight.dim() != 2:
        return None

    try:
        grid_key = _resolve_grid_thw_key(visual, grid_thw)
        key = (
            grid_key,
            int(visual.num_grid_per_side),
            int(getattr(visual.config, "spatial_merge_size", visual.spatial_merge_size)),
            str(pos_weight.dtype),
            str(pos_weight.device),
        )
        cache = _get_pos_index_cache(visual)
        max_entries = conf_int("VISION_POS_CACHE_SIZE", "8", minimum=1)
        cached = _lru_get(cache, key, max_entries)
        if cached is None:
            cached = _build_pos_interp_indices_for_key(
                visual,
                grid_key,
                dtype=pos_weight.dtype,
                device=pos_weight.device,
            )
            _lru_put(cache, key, cached, max_entries)
        idx, weights = cached
        gathered = pos_weight[idx]
        out = (gathered * weights[:, :, None]).sum(dim=0)
        _stat_inc("pos_interp_indexed")
        return out
    except Exception:
        _stat_inc("pos_interp_indexed_fallback")
        return None


def _cached_fast_pos_embed_interpolate(visual, grid_thw: torch.Tensor, *, _original) -> torch.Tensor:
    if not _env_on("ENABLE_VISION_POS_CACHE", "1"):
        return _original(grid_thw)

    key = _pos_cache_key(visual, grid_thw)
    cache = _get_pos_cache(visual)
    max_entries = conf_int("VISION_POS_CACHE_SIZE", "8", minimum=1)
    cached = _lru_get(cache, key, max_entries)
    if cached is not None:
        _stat_inc("pos_cache_hit")
        return cached

    _stat_inc("pos_cache_miss")
    out = _fast_pos_embed_interpolate_indexed(visual, grid_thw)
    if out is None:
        out = _fast_pos_embed_interpolate_triton(visual, grid_thw)
    if out is None:
        out = _original(grid_thw)
    if not torch.is_grad_enabled():
        _lru_put(cache, key, out, max_entries)
    return out


def _build_cu_seqlens_host(grid_key: Tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """用 tiny host 前缀和替代 repeat_interleave，避免为几项长度启动重排 kernel。"""
    prefix = [0]
    for i in range(0, len(grid_key), 3):
        t, h, w = int(grid_key[i]), int(grid_key[i + 1]), int(grid_key[i + 2])
        step = h * w
        for _ in range(t):
            prefix.append(prefix[-1] + step)
    _stat_inc("cu_seqlens_host")
    return torch.tensor(prefix, dtype=dtype, device=device)


def _build_cu_seqlens(grid_thw: torch.Tensor, grid_key: Tuple) -> torch.Tensor:
    cu_dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32
    if _env_on("ENABLE_VISION_CU_SEQLENS_HOST", "1") and not torch.is_grad_enabled():
        return _build_cu_seqlens_host(grid_key, cu_dtype, grid_thw.device)

    _stat_inc("cu_seqlens_torch")
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=cu_dtype)
    return F.pad(cu_seqlens, (1, 0), value=0)


def _prepare_position_and_cu_seqlens(
    visual,
    grid_thw: torch.Tensor,
    seq_len: int,
    *,
    _original_rot_pos_emb: Callable,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """返回 (rotary_pos_emb_flat, (cos, sin), cu_seqlens)。"""
    use_rope_cache = _env_on("ENABLE_VISION_ROPE_CACHE", "1") and not torch.is_grad_enabled()
    grid_key = _resolve_grid_thw_key(visual, grid_thw)

    if use_rope_cache:
        key = _rope_cache_key(visual, grid_thw)
        cache = _get_rope_cache(visual)
        max_entries = conf_int("VISION_POS_CACHE_SIZE", "8", minimum=1)
        cached = _lru_get(cache, key, max_entries)
        if cached is not None:
            _stat_inc("rope_cache_hit")
            rotary_pos_emb, position_embeddings, cu_seqlens = cached
            return rotary_pos_emb, position_embeddings, cu_seqlens
        _stat_inc("rope_cache_miss")

    rotary_pos_emb = _original_rot_pos_emb(grid_thw)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = _build_cu_seqlens(grid_thw, grid_key)

    if use_rope_cache:
        _lru_put(
            cache,
            key,
            (rotary_pos_emb, position_embeddings, cu_seqlens),
            max_entries,
        )

    return rotary_pos_emb, position_embeddings, cu_seqlens


def _vision_mlp_fast_forward(mlp, hidden_state: torch.Tensor, *, _original_forward) -> torch.Tensor:
    if not _env_on("ENABLE_VISION_MLP_FAST", "1"):
        return _original_forward(hidden_state)

    if not hidden_state.is_cuda or not hidden_state.is_contiguous():
        return _original_forward(hidden_state)

    try:
        h = hidden_state if hidden_state.dim() == 2 else hidden_state.reshape(-1, hidden_state.shape[-1])
        if (
            _env_on("ENABLE_VISION_LINEAR_MM_OUT", "1")
            and not torch.is_grad_enabled()
            and h.dim() == 2
            and h.is_cuda
        ):
            h = h.contiguous() if not h.is_contiguous() else h
            mid = _linear_fast_out(mlp.linear_fc1, h)
        else:
            mid = F.linear(h, mlp.linear_fc1.weight, mlp.linear_fc1.bias)
        mid = _vision_activation_fast(mlp.act_fn, mid)
        if (
            _env_on("ENABLE_VISION_LINEAR_MM_OUT", "1")
            and not torch.is_grad_enabled()
            and mid.dim() == 2
            and mid.is_cuda
        ):
            mid = mid.contiguous() if not mid.is_contiguous() else mid
            out = _linear_fast_out(mlp.linear_fc2, mid)
        else:
            out = F.linear(mid, mlp.linear_fc2.weight, mlp.linear_fc2.bias)
        _stat_inc("vision_mlp_fast")
        if hidden_state.dim() != 2:
            return out.view_as(hidden_state)
        return out
    except Exception:
        return _original_forward(hidden_state)


def _vision_activation_fast(act_fn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    name = act_fn.__class__.__name__.lower()
    if name == "gelutanh":
        _stat_inc("vision_gelu_functional")
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        _stat_inc("vision_gelu_functional")
        return F.gelu(x)
    return act_fn(x)


def _vision_merger_fast_forward(merger, x: torch.Tensor, *, _original_forward) -> torch.Tensor:
    if not _env_on("ENABLE_VISION_MLP_FAST", "1"):
        return _original_forward(x)

    if not x.is_cuda:
        return _original_forward(x)

    try:
        hidden_size = merger.hidden_size
        use_post = merger.use_postshuffle_norm
        if use_post:
            h = merger.norm(x.view(-1, hidden_size)).view(-1, hidden_size)
        else:
            h = merger.norm(x).view(-1, hidden_size)
        if (
            _env_on("ENABLE_VISION_LINEAR_MM_OUT", "1")
            and not torch.is_grad_enabled()
            and h.dim() == 2
            and h.is_cuda
        ):
            h = h.contiguous() if not h.is_contiguous() else h
            mid = _linear_fast_out(merger.linear_fc1, h)
        else:
            mid = F.linear(h, merger.linear_fc1.weight, merger.linear_fc1.bias)
        mid = _vision_activation_fast(merger.act_fn, mid)
        if (
            _env_on("ENABLE_VISION_LINEAR_MM_OUT", "1")
            and not torch.is_grad_enabled()
            and mid.dim() == 2
            and mid.is_cuda
        ):
            mid = mid.contiguous() if not mid.is_contiguous() else mid
            out = _linear_fast_out(merger.linear_fc2, mid)
        else:
            out = F.linear(mid, merger.linear_fc2.weight, merger.linear_fc2.bias)
        _stat_inc("vision_mlp_fast")
        return out
    except Exception:
        return _original_forward(x)


def _patch_visual_forward(visual) -> bool:
    if getattr(visual, "_vision_forward_patched", False):
        return False

    original_forward = visual.forward
    visual._vision_forward_original = original_forward

    def vision_forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ):
        if not _env_on("ENABLE_VISION_KERNEL_FUSIONS", "1"):
            _stat_inc("vision_forward_fallback")
            return original_forward(hidden_states, grid_thw, **kwargs)

        try:
            hidden_states = self.patch_embed(hidden_states)

            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

            first_norm = None
            blocks = getattr(self, "blocks", None)
            if blocks:
                fused = _triton_pos_add_layernorm0(hidden_states, pos_embeds, blocks[0].norm1)
                if fused is not None:
                    hidden_states, first_norm = fused
                else:
                    _stat_inc("pos_add_ln0_fallback")
                    hidden_states = hidden_states + pos_embeds
            else:
                hidden_states = hidden_states + pos_embeds

            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len, -1)

            rot_fn = getattr(self, "_vision_rot_pos_emb_original", self.rot_pos_emb)
            _, position_embeddings, cu_seqlens = _prepare_position_and_cu_seqlens(
                self,
                grid_thw,
                seq_len,
                _original_rot_pos_emb=rot_fn,
            )

            deepstack_feature_lists = []
            deepstack_merger_map = getattr(self, "_vision_deepstack_merger_map", None)
            if deepstack_merger_map is None:
                deepstack_merger_map = {
                    int(layer_idx): merger
                    for layer_idx, merger in zip(self.deepstack_visual_indexes, self.deepstack_merger_list)
                }
                self._vision_deepstack_merger_map = deepstack_merger_map
            for layer_num, blk in enumerate(self.blocks):
                if layer_num == 0 and first_norm is not None:
                    hidden_states = _first_block_with_pre_norm(
                        blk,
                        hidden_states,
                        first_norm,
                        cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                else:
                    hidden_states = blk(
                        hidden_states,
                        cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                deepstack_merger = deepstack_merger_map.get(layer_num)
                if deepstack_merger is not None:
                    deepstack_feature = deepstack_merger(hidden_states)
                    deepstack_feature_lists.append(deepstack_feature)

            hidden_states = self.merger(hidden_states)
            _stat_inc("vision_forward_fast")
            return hidden_states, deepstack_feature_lists
        except Exception:
            _stat_inc("vision_forward_fallback")
            return original_forward(hidden_states, grid_thw, **kwargs)

    visual.forward = MethodType(vision_forward, visual)
    visual._vision_forward_patched = True
    return True


def _patch_visual_linears(visual) -> bool:
    if not _env_on("ENABLE_VISION_LINEAR_MM_OUT", "1"):
        return False
    patched_any = False
    for module in visual.modules():
        if not isinstance(module, nn.Linear) or getattr(module, "_vision_linear_mm_out_patched", False):
            continue
        original_forward = module.forward

        def linear_forward(self, x, _orig=original_forward):
            return _vision_linear_mm_out_forward(self, x, _original_forward=_orig)

        module.forward = MethodType(linear_forward, module)
        module._vision_linear_mm_out_patched = True
        patched_any = True
    return patched_any


def _prewarm_patch_embed_cublaslt(patch_embed) -> None:
    if (
        _env_str("VISION_PATCH_EMBED_BACKEND", "mm_out") != "cublaslt"
        or not _env_on("ENABLE_VISION_CUBLASLT_PREWARM", "1")
    ):
        return
    try:
        weight_t = getattr(patch_embed, "_vision_patch_linear_weight_t", None)
        bias = getattr(patch_embed, "_vision_patch_linear_bias", None)
        if weight_t is None or not weight_t.is_cuda:
            return
        rows = conf_int("VISION_CUBLASLT_PREWARM_ROWS", "256", minimum=1)
        cols = int(patch_embed.embed_dim)
        k = int(weight_t.shape[0])
        dummy = torch.empty((rows, k), dtype=weight_t.dtype, device=weight_t.device)
        out = _get_patch_mm_workspace(patch_embed, rows, cols, weight_t.dtype, weight_t.device)
        with torch.no_grad():
            _cublaslt_matmul_bias_out(dummy, weight_t, bias, out)
    except Exception:
        _stat_inc("patch_embed_cublaslt_fallback")


def _patch_visual_attentions(visual) -> bool:
    if not _env_on("ENABLE_VISION_ROPE_TRITON", "1"):
        return False
    patched_any = False
    for module in visual.modules():
        if (
            module.__class__.__name__ != "Qwen3VLVisionAttention"
            or getattr(module, "_vision_rope_triton_patched", False)
        ):
            continue
        original_forward = module.forward

        def attention_forward(
            self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=None,
            position_embeddings=None,
            _orig=original_forward,
            **kwargs,
        ):
            return _vision_attention_fast_forward(
                self,
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                _original_forward=_orig,
                **kwargs,
            )

        module.forward = MethodType(attention_forward, module)
        module._vision_rope_triton_patched = True
        patched_any = True
    return patched_any


def prewarm_vision_encoder(model, processor=None) -> bool:
    """模型初始化后用 dummy 336x336 图跑 vision forward，预热 patch GEMM / RoPE / allocator。"""
    if not _env_on("ENABLE_VISION_FORWARD_PREWARM", "1"):
        return False
    if getattr(model, "_vision_forward_prewarmed", False):
        return True

    vlm = getattr(model, "model", None)
    if vlm is None or not hasattr(vlm, "get_image_features"):
        return False
    visual = getattr(vlm, "visual", None)
    if visual is None:
        return False

    image_processor = getattr(processor, "image_processor", None) if processor is not None else None
    if image_processor is None:
        return False

    try:
        from PIL import Image
    except Exception:
        return False

    width, height = _parse_fixed_resolution()
    device = next(model.parameters()).device
    dummy = Image.new("RGB", (width, height), color=(0, 0, 0))
    batch = image_processor(images=dummy, return_tensors="pt")
    pixel_values = batch["pixel_values"].to(device=device)
    image_grid_thw = batch["image_grid_thw"].to(device=device)

    if _fixed_vision_shape_enabled():
        _confirm_fixed_grid_key(visual, image_grid_thw)

    with torch.inference_mode():
        _ = vlm.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            _ttft_skip_cache=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    model._vision_forward_prewarmed = True
    _stat_inc("vision_forward_prewarm")
    print(f"[vision_kernel_fusions] vision forward prewarm done ({width}x{height})")
    return True


def _prewarm_vision_rope_triton(visual) -> None:
    if not _env_on("ENABLE_VISION_ROPE_TRITON_PREWARM", "1"):
        return
    try:
        first_attn = None
        for module in visual.modules():
            if module.__class__.__name__ == "Qwen3VLVisionAttention":
                first_attn = module
                break
        if first_attn is None:
            return
        weight = first_attn.qkv.weight
        if not weight.is_cuda:
            return
        seq_len = 1
        num_heads = int(first_attn.num_heads)
        head_dim = int(first_attn.head_dim)
        qkv = torch.empty(
            (seq_len, 3, num_heads, head_dim),
            dtype=weight.dtype,
            device=weight.device,
        )
        q, k, _ = qkv.permute(1, 0, 2, 3).unbind(0)
        cos = torch.empty((seq_len, head_dim), dtype=weight.dtype, device=weight.device)
        sin = torch.empty_like(cos)
        with torch.no_grad():
            _apply_vision_rope_triton(q, k, (cos, sin))
    except Exception:
        _stat_inc("vision_rope_triton_fallback")


def apply_vision_kernel_fusions(model) -> bool:
    """对 Qwen3-VL visual encoder 应用 patch embed GEMM、位置/rope 缓存等优化。"""
    if not _env_on("ENABLE_VISION_KERNEL_FUSIONS", "1"):
        return False

    vlm = getattr(model, "model", None)
    visual = getattr(vlm, "visual", None)
    if visual is None:
        return False
    if getattr(model, "_vision_kernel_fusions_patched", False):
        return True

    patched_any = False

    # 1) Patch embed: Conv3d -> F.linear (cuBLAS/aublas GEMM)
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is not None and _env_on("ENABLE_VISION_PATCH_EMBED_LINEAR", "1"):
        if not getattr(patch_embed, "_vision_patch_embed_patched", False):
            original_pe_forward = patch_embed.forward

            def pe_forward(self, hidden_states, _orig=original_pe_forward):
                return _patch_embed_linear_forward(
                    self, hidden_states, _original_forward=_orig
                )

            patch_embed.forward = MethodType(pe_forward, patch_embed)
            patch_embed._vision_patch_embed_patched = True
            w2d, bias = _build_patch_linear_weight(patch_embed.proj)
            patch_embed._vision_patch_linear_weight = w2d
            patch_embed._vision_patch_linear_bias = bias
            patch_embed._vision_patch_linear_weight_t = w2d.transpose(0, 1).contiguous()
            _prewarm_patch_embed_cublaslt(patch_embed)
            patched_any = True

    # 2) fast_pos_embed_interpolate cache
    if hasattr(visual, "fast_pos_embed_interpolate") and not getattr(
        visual, "_vision_fast_pos_patched", False
    ):
        original_fast_pos = visual.fast_pos_embed_interpolate

        def fast_pos_cached(self, grid_thw, _orig=original_fast_pos):
            return _cached_fast_pos_embed_interpolate(self, grid_thw, _original=_orig)

        visual.fast_pos_embed_interpolate = MethodType(fast_pos_cached, visual)
        visual._vision_fast_pos_patched = True
        patched_any = True

    # 保留 rot_pos_emb 原函数供缓存 miss 使用
    if not hasattr(visual, "_vision_rot_pos_emb_original"):
        visual._vision_rot_pos_emb_original = visual.rot_pos_emb

    # 3) 替换 visual.forward，复用 rope/cu_seqlens 缓存
    if _patch_visual_forward(visual):
        patched_any = True

    # 4) Vision attention RoPE: fused q/k rotate_half without aten::neg/cat
    if _patch_visual_attentions(visual):
        _prewarm_vision_rope_triton(visual)
        patched_any = True

    # 5) Vision blocks / mergers Linear: addmm -> cuBLASLt/mm(out=workspace)+bias
    if _patch_visual_linears(visual):
        patched_any = True

    # 6) 可选 MLP / merger fast path
    if _env_on("ENABLE_VISION_MLP_FAST", "0"):
        for blk in getattr(visual, "blocks", []):
            mlp = getattr(blk, "mlp", None)
            if mlp is not None and not getattr(mlp, "_vision_mlp_patched", False):
                orig = mlp.forward

                def mlp_fwd(self, hidden_state, _o=orig):
                    return _vision_mlp_fast_forward(self, hidden_state, _original_forward=_o)

                mlp.forward = MethodType(mlp_fwd, mlp)
                mlp._vision_mlp_patched = True
                patched_any = True

        merger = getattr(visual, "merger", None)
        if merger is not None and not getattr(merger, "_vision_merger_patched", False):
            orig_m = merger.forward

            def merger_fwd(self, x, _o=orig_m):
                return _vision_merger_fast_forward(self, x, _original_forward=_o)

            merger.forward = MethodType(merger_fwd, merger)
            merger._vision_merger_patched = True
            patched_any = True

        for merger in getattr(visual, "deepstack_merger_list", []):
            if getattr(merger, "_vision_merger_patched", False):
                continue
            orig_ds = merger.forward

            def ds_fwd(self, x, _o=orig_ds):
                return _vision_merger_fast_forward(self, x, _original_forward=_o)

            merger.forward = MethodType(ds_fwd, merger)
            merger._vision_merger_patched = True
            patched_any = True

    if patched_any:
        model._vision_kernel_fusions_patched = True
        model._vision_backend_stats = _BACKEND_STATS
        print(
            "[vision_kernel_fusions] patched visual encoder "
            f"(patch_linear={_env_on('ENABLE_VISION_PATCH_EMBED_LINEAR', '1')}, "
            f"pos_cache={_env_on('ENABLE_VISION_POS_CACHE', '1')}, "
            f"rope_cache={_env_on('ENABLE_VISION_ROPE_CACHE', '1')})"
        )
    return patched_any


def get_vision_backend_stats(model) -> Dict[str, int]:
    return dict(getattr(model, "_vision_backend_stats", _BACKEND_STATS))
