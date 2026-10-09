"""Unified Sage decode-attention adapter for Qwen3-VL text generation.

This module is the only high-level SageAttention entry point used by
``evaluation_wrapper.py``.  It intentionally targets the benchmark decode hot
path (batch=1, q_len=1, Hq=16, Hkv=8, D=128) and does not expose a prefill
attention replacement.  The external SageAttention repository does not provide
a KV-cache decode API, so this adapter keeps the Sage design assumptions that
matter for decode: INT8 K/Q metadata, BF16 or INT8 V cache choices, static
workspace, GQA-aware kernels, and explicit hit/miss accounting.
"""

from __future__ import annotations

import atexit
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class SageDecodePolicy:
    kv_dtype: str = "int8"
    granularity: str = "per_token_head"
    qk_path: str = "i8"
    value_path: str = "bf16"
    block_kv: int = 64


@dataclass
class SageDecodeStats:
    decode_calls: int = 0
    triton_calls: int = 0
    cuda_calls: int = 0
    torch_ref_calls: int = 0
    cache_update_calls: int = 0
    graph_decode_replay_calls: int = 0
    miss_count: int = 0
    miss_reasons: dict[str, int] = field(default_factory=dict)


SageAttentionQuantPolicy = SageDecodePolicy
_STATS = SageDecodeStats()
_REGISTERED_ATEXIT = False
_CUDA_EXT: Any | None = None
_NATIVE_CUDA_EXT: Any | None = None
_NATIVE_CUDA_BACKEND = "off"


def set_cuda_extension(module: Any | None) -> None:
    global _CUDA_EXT
    _CUDA_EXT = module


def set_native_cuda_extension(module: Any | None, backend: str = "off") -> None:
    global _NATIVE_CUDA_EXT, _NATIVE_CUDA_BACKEND
    _NATIVE_CUDA_EXT = module
    _NATIVE_CUDA_BACKEND = backend


def reset_stats() -> None:
    global _STATS
    _STATS = SageDecodeStats()


def get_stats() -> dict[str, Any]:
    return asdict(_STATS)


def record_miss(reason: str) -> None:
    _STATS.miss_count += 1
    _STATS.miss_reasons[reason] = _STATS.miss_reasons.get(reason, 0) + 1


def record_cache_update(count: int = 1) -> None:
    _STATS.cache_update_calls += int(count)


def record_graph_decode_hits(*, num_layers: int, steps: int) -> None:
    if steps <= 0:
        return
    hits = int(num_layers) * int(steps)
    _STATS.decode_calls += hits
    _STATS.cuda_calls += hits
    _STATS.graph_decode_replay_calls += hits


def record_decode_hit(*, backend: str = "cuda", count: int = 1) -> None:
    hits = max(int(count), 0)
    if hits == 0:
        return
    _STATS.decode_calls += hits
    if backend == "triton":
        _STATS.triton_calls += hits
    elif backend == "torch_ref":
        _STATS.torch_ref_calls += hits
    else:
        _STATS.cuda_calls += hits


def write_stats(path: str = "sage_decode_attention_stats.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(get_stats(), f, indent=2, sort_keys=True)


def register_atexit(path: str = "sage_decode_attention_stats.json") -> None:
    global _REGISTERED_ATEXIT
    if _REGISTERED_ATEXIT:
        return

    def _write() -> None:
        try:
            write_stats(path)
        except Exception:
            pass

    atexit.register(_write)
    _REGISTERED_ATEXIT = True


def _policy_from_any(policy: SageDecodePolicy | dict[str, Any] | None) -> SageDecodePolicy:
    if policy is None:
        return SageDecodePolicy()
    if isinstance(policy, SageDecodePolicy):
        return policy
    values = {k: v for k, v in policy.items() if k in SageDecodePolicy.__dataclass_fields__}
    return SageDecodePolicy(**values)


def _layer_cache(cache: torch.Tensor, layer_idx: int | None) -> torch.Tensor:
    if cache.dim() == 5:
        if layer_idx is None:
            raise ValueError("layer_idx is required for a 5D [L,B,S,H,D] KV cache.")
        return cache[layer_idx, 0]
    if cache.dim() == 4:
        return cache[0]
    if cache.dim() == 3:
        return cache
    raise ValueError(f"Unsupported KV cache shape: {tuple(cache.shape)}")


def _layer_scale(scale: torch.Tensor | None, layer_idx: int | None) -> torch.Tensor | None:
    if scale is None:
        return None
    if scale.dim() == 3:
        if layer_idx is None:
            raise ValueError("layer_idx is required for a 3D [L,S,H] scale cache.")
        return scale[layer_idx]
    if scale.dim() == 2:
        return scale
    raise ValueError(f"Unsupported scale cache shape: {tuple(scale.shape)}")


def _cache_len(cache_seqlens: torch.Tensor | int | None, fallback: int) -> int:
    if cache_seqlens is None:
        return fallback
    if isinstance(cache_seqlens, int):
        return int(cache_seqlens)
    return int(cache_seqlens.detach().view(-1)[0].item())


def _quantize_token_headwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    scale = torch.amax(torch.abs(xf), dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round((xf / scale.unsqueeze(-1)).clamp(-127.0, 127.0)).to(torch.int8)
    return q, scale.float()


def torch_ref_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    cache_seqlens: torch.Tensor | int | None,
    softmax_scale: float,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    q2 = q.view(1, 1, q.shape[-2], q.shape[-1])
    kv_len = _cache_len(cache_seqlens, k_cache.shape[0])
    k = k_cache[:kv_len]
    v = v_cache[:kv_len]
    if k.dtype == torch.int8:
        if k_scale is None:
            raise ValueError("INT8 torch_ref decode requires k_scale.")
        k = k.float() * k_scale[:kv_len].unsqueeze(-1)
        if v.dtype == torch.int8:
            if v_scale is None:
                raise ValueError("INT8 V torch_ref decode requires v_scale.")
            v = v.float() * v_scale[:kv_len].unsqueeze(-1)
    k = k.view(1, kv_len, k.shape[-2], k.shape[-1])
    v = v.view(1, kv_len, v.shape[-2], v.shape[-1])
    groups = q2.shape[2] // k.shape[2]
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q2.float(), k.float()) * float(softmax_scale)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.float())
    return out.to(dtype=q.dtype)


def quantize_kv_cache_i8(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_cache_i8: torch.Tensor,
    value_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    token_index: torch.Tensor,
    *,
    count: int,
    impl: str = "triton",
) -> None:
    if count <= 0:
        return
    if impl in {"triton", "cuda"}:
        import sageattention_kvcache_triton as triton_impl

        triton_impl.quantize_kv_cache_i8(
            key_cache,
            value_cache,
            key_cache_i8,
            value_cache_i8,
            key_scale,
            value_scale,
            token_index,
            count=count,
        )
    elif impl == "torch_ref":
        start = int(token_index.detach().view(-1)[0].item())
        k_q, k_s = _quantize_token_headwise(key_cache[start : start + count])
        v_q, v_s = _quantize_token_headwise(value_cache[start : start + count])
        key_cache_i8[start : start + count].copy_(k_q)
        value_cache_i8[start : start + count].copy_(v_q)
        key_scale[start : start + count].copy_(k_s)
        value_scale[start : start + count].copy_(v_s)
    else:
        raise ValueError(f"Unsupported Sage decode cache update impl: {impl!r}")
    record_cache_update(count)


def quantize_k_cache_i8(
    key_cache: torch.Tensor,
    key_cache_i8: torch.Tensor,
    key_scale: torch.Tensor,
    token_index: torch.Tensor,
    *,
    count: int,
    impl: str = "triton",
) -> None:
    if count <= 0:
        return
    if impl in {"triton", "cuda"}:
        import sageattention_kvcache_triton as triton_impl

        triton_impl.quantize_k_cache_i8(
            key_cache,
            key_cache_i8,
            key_scale,
            token_index,
            count=count,
        )
    elif impl == "torch_ref":
        start = int(token_index.detach().view(-1)[0].item())
        k_q, k_s = _quantize_token_headwise(key_cache[start : start + count])
        key_cache_i8[start : start + count].copy_(k_q)
        key_scale[start : start + count].copy_(k_s)
    else:
        raise ValueError(f"Unsupported Sage decode K-cache update impl: {impl!r}")
    record_cache_update(count)


def allocate_workspace(*, max_cache_len: int, num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    import sageattention_kvcache_triton as triton_impl

    return triton_impl.allocate_workspace(
        max_cache_len=max_cache_len,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def sage_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    cache_seqlens: torch.Tensor | int | None,
    layer_idx: int | None = None,
    softmax_scale: float | None = None,
    quant_policy: SageDecodePolicy | dict[str, Any] | None = None,
    impl: str = "cuda",
    workspace: Any | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    assert_hit: bool = False,
) -> torch.Tensor:
    policy = _policy_from_any(quant_policy)
    if policy.kv_dtype != "int8":
        raise ValueError(f"Sage decode requires an INT8 K cache policy; got {policy}.")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    k_layer = _layer_cache(k_cache, layer_idx)
    v_layer = _layer_cache(v_cache, layer_idx)
    ks_layer = _layer_scale(k_scale, layer_idx)
    vs_layer = _layer_scale(v_scale, layer_idx)

    _STATS.decode_calls += 1
    if impl == "torch_ref":
        _STATS.torch_ref_calls += 1
        return torch_ref_decode(
            q,
            k_layer,
            v_layer,
            cache_seqlens=cache_seqlens,
            softmax_scale=float(softmax_scale),
            k_scale=ks_layer,
            v_scale=vs_layer,
        )

    if workspace is None:
        record_miss("missing_workspace")
        if assert_hit:
            raise RuntimeError("Sage decode requires a static workspace.")
        raise ValueError("Sage decode requires a static workspace.")
    if ks_layer is None:
        record_miss("missing_k_scale")
        if assert_hit:
            raise RuntimeError("Sage decode requires K scale cache.")
        raise ValueError("Sage decode requires K scale cache.")
    if k_layer.dtype != torch.int8:
        record_miss("non_int8_k_cache")
        if assert_hit:
            raise RuntimeError("Sage decode requires an INT8 K cache.")
        raise ValueError("Sage decode requires an INT8 K cache.")

    if impl == "cuda":
        if (
            policy.value_path == "bf16"
            and _NATIVE_CUDA_EXT is not None
            and _NATIVE_CUDA_BACKEND == "split"
            and hasattr(_NATIVE_CUDA_EXT, "sage_decode_k_i8_v_bf16_split")
        ):
            _STATS.cuda_calls += 1
            _NATIVE_CUDA_EXT.sage_decode_k_i8_v_bf16_split(
                q.view(q.shape[-2], q.shape[-1]),
                k_layer,
                v_layer,
                ks_layer,
                cache_seqlens,
                workspace.partial_max,
                workspace.partial_sum,
                workspace.partial_out,
                workspace.output,
                float(softmax_scale),
            )
            return workspace.output_4d
        if (
            policy.value_path == "bf16"
            and _NATIVE_CUDA_EXT is not None
            and _NATIVE_CUDA_BACKEND == "single"
            and hasattr(_NATIVE_CUDA_EXT, "sage_decode_k_i8_v_bf16_single")
        ):
            _STATS.cuda_calls += 1
            _NATIVE_CUDA_EXT.sage_decode_k_i8_v_bf16_single(
                q.view(q.shape[-2], q.shape[-1]),
                k_layer,
                v_layer,
                ks_layer,
                cache_seqlens,
                workspace.output,
                float(softmax_scale),
            )
            return workspace.output_4d
        if _CUDA_EXT is None:
            record_miss("missing_cuda_extension")
            if assert_hit:
                raise RuntimeError("Sage decode CUDA extension is unavailable.")
            raise ValueError("Sage decode CUDA extension is unavailable.")
        if policy.value_path == "bf16":
            if not hasattr(_CUDA_EXT, "sage_decode_k_i8_v_bf16"):
                record_miss("missing_cuda_k_i8_v_bf16")
                if assert_hit:
                    raise RuntimeError("Sage decode CUDA K-I8/V-BF16 kernel is unavailable.")
                raise ValueError("Sage decode CUDA K-I8/V-BF16 kernel is unavailable.")
            _STATS.cuda_calls += 1
            _CUDA_EXT.sage_decode_k_i8_v_bf16(
                q.view(q.shape[-2], q.shape[-1]),
                k_layer,
                v_layer,
                ks_layer,
                cache_seqlens,
                workspace.partial_max,
                workspace.partial_sum,
                workspace.partial_out,
                workspace.output,
                float(softmax_scale),
            )
            return workspace.output_4d
        if v_layer.dtype != torch.int8 or vs_layer is None:
            record_miss("missing_int8_v_cache")
            if assert_hit:
                raise RuntimeError("Sage INT8-V decode requires V cache and scales.")
            raise ValueError("Sage INT8-V decode requires V cache and scales.")
        if policy.qk_path == "i8" and hasattr(_CUDA_EXT, "sage_decode_q_i8_k_i8_v_i8"):
            record_miss("q_i8_cuda_requires_prequant_q")
            if assert_hit:
                raise RuntimeError("Direct Q-I8 CUDA path must be called from the fused wrapper.")
        impl = "triton"

    if impl != "triton":
        record_miss(f"unsupported_impl:{impl}")
        if assert_hit:
            raise RuntimeError(f"Unsupported Sage decode impl: {impl}")
        raise ValueError(f"Unsupported Sage decode impl: {impl!r}")

    import sageattention_kvcache_triton as triton_impl

    _STATS.triton_calls += 1
    if policy.value_path == "bf16":
        return triton_impl.decode_attention_k_i8_v_bf16_4d(
            q.view(q.shape[-2], q.shape[-1]),
            k_layer,
            v_layer,
            ks_layer,
            cache_seqlens,
            workspace,
            float(softmax_scale),
        )
    if v_layer.dtype != torch.int8 or vs_layer is None:
        record_miss("missing_int8_v_cache")
        if assert_hit:
            raise RuntimeError("Sage INT8-V decode requires V cache and scales.")
        raise ValueError("Sage INT8-V decode requires V cache and scales.")
    if policy.qk_path == "int8_dot":
        return triton_impl.decode_attention_i8dot_4d(
            q.view(q.shape[-2], q.shape[-1]),
            k_layer,
            v_layer,
            ks_layer,
            vs_layer,
            cache_seqlens,
            workspace,
            float(softmax_scale),
        )
    if policy.qk_path not in {"i8", "dequant_i8"}:
        record_miss(f"unsupported_qk_path:{policy.qk_path}")
        if assert_hit:
            raise RuntimeError(f"Unsupported Sage decode qk_path: {policy.qk_path}")
        raise ValueError(f"Unsupported Sage decode qk_path: {policy.qk_path!r}")
    return triton_impl.decode_attention_i8_4d(
        q.view(q.shape[-2], q.shape[-1]),
        k_layer,
        v_layer,
        ks_layer,
        vs_layer,
        cache_seqlens,
        workspace,
        float(softmax_scale),
    )


if os.getenv("JUNKRAT_SAGE_DEBUG", "0") == "1":
    register_atexit()
