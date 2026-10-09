from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from my_kernel.decode_qk_rotary.runtime import (  # noqa: E402
    apply_decode_qk_rotary_ext_fastpath,
    decode_qk_rotary_graph_linear_mode,
)
from transformers.cache_utils import StaticCache  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    Qwen3VLTextAttention,
    Qwen3VLTextConfig,
    Qwen3VLTextRotaryEmbedding,
)


class _Layer(torch.nn.Module):
    def __init__(self, attn: torch.nn.Module):
        super().__init__()
        self.self_attn = attn


class _LanguageModel(torch.nn.Module):
    def __init__(self, *layers: torch.nn.Module):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class _PatchWrapper(torch.nn.Module):
    def __init__(self, attns: list[torch.nn.Module]):
        super().__init__()
        self.model = SimpleNamespace(language_model=_LanguageModel(*[_Layer(attn) for attn in attns]))

    def generate(self, *args, **kwargs):
        raise RuntimeError("validate_decode_qk_rotary_graph_cache does not call generate")


def _make_config(args: argparse.Namespace) -> Qwen3VLTextConfig:
    cfg = Qwen3VLTextConfig(
        hidden_size=int(args.hidden_size),
        intermediate_size=max(int(args.hidden_size) * 4, 1),
        num_attention_heads=int(args.num_heads),
        num_key_value_heads=int(args.num_kv_heads),
        head_dim=int(args.head_dim),
        num_hidden_layers=max(int(args.num_layers), 1),
        attention_bias=False,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_attention(cfg: Qwen3VLTextConfig, device: torch.device, dtype: torch.dtype, layer_idx: int) -> Qwen3VLTextAttention:
    attn = Qwen3VLTextAttention(cfg, layer_idx=int(layer_idx)).to(device=device, dtype=dtype)
    attn.eval()
    for param in attn.parameters():
        param.requires_grad_(False)
    return attn


def _patch_attentions(attns: list[torch.nn.Module]) -> None:
    count = apply_decode_qk_rotary_ext_fastpath(_PatchWrapper(attns))
    if count != len(attns):
        raise RuntimeError(f"decode qk rotary fastpath patch count={count}, expected {len(attns)}")


def _make_position_embeddings(
    cfg: Qwen3VLTextConfig,
    hidden_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Qwen3VLTextRotaryEmbedding(cfg).to(device=hidden_states.device)
    position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1).contiguous()
    return rotary(hidden_states, position_ids)


def _make_attention_mask(
    dtype: torch.dtype,
    device: torch.device,
    q_len: int,
    start_pos: int,
    max_cache_len: int,
) -> torch.Tensor:
    mask = torch.full(
        (1, 1, int(q_len), int(max_cache_len)),
        torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    for idx in range(int(q_len)):
        mask[:, :, idx, : int(start_pos) + idx + 1] = 0
    return mask


def _make_static_cache(
    cfg: Qwen3VLTextConfig,
    device: torch.device,
    dtype: torch.dtype,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    start_pos: int,
    max_cache_len: int,
) -> StaticCache:
    cache = StaticCache(config=cfg, max_cache_len=int(max_cache_len))
    seed_k = torch.zeros(1, int(cfg.num_key_value_heads), 1, int(cfg.head_dim), device=device, dtype=dtype)
    seed_v = torch.zeros_like(seed_k)
    for layer_idx, layer in enumerate(cache.layers):
        layer.lazy_initialization(seed_k)
        layer.keys.zero_()
        layer.values.zero_()
        layer.keys[:, :, : int(start_pos), :].copy_(prefix_k[layer_idx])
        layer.values[:, :, : int(start_pos), :].copy_(prefix_v[layer_idx])
        try:
            layer.cumulative_length = int(start_pos)
        except Exception:
            pass
    cache._aicas_eagle3_seq_len = int(start_pos)
    return cache


def _region_stats(graph_cache: StaticCache, ref_cache: StaticCache, start_pos: int, total_len: int, layer_idx: int) -> dict[str, float]:
    g_layer = graph_cache.layers[int(layer_idx)]
    r_layer = ref_cache.layers[int(layer_idx)]
    gk = g_layer.keys
    gv = g_layer.values
    rk = r_layer.keys
    rv = r_layer.values
    start_pos = int(start_pos)
    total_len = int(total_len)
    stats = {
        "prefix_k": 0.0,
        "prefix_v": 0.0,
        "append_k": float((gk[:, :, start_pos:total_len, :].float() - rk[:, :, start_pos:total_len, :].float()).abs().max().item()),
        "append_v": float((gv[:, :, start_pos:total_len, :].float() - rv[:, :, start_pos:total_len, :].float()).abs().max().item()),
        "graph_append_k_abs": float(gk[:, :, start_pos:total_len, :].float().abs().max().item()),
        "graph_append_v_abs": float(gv[:, :, start_pos:total_len, :].float().abs().max().item()),
        "ref_append_k_abs": float(rk[:, :, start_pos:total_len, :].float().abs().max().item()),
        "ref_append_v_abs": float(rv[:, :, start_pos:total_len, :].float().abs().max().item()),
    }
    if start_pos > 0:
        stats["prefix_k"] = float((gk[:, :, :start_pos, :].float() - rk[:, :, :start_pos, :].float()).abs().max().item())
        stats["prefix_v"] = float((gv[:, :, :start_pos, :].float() - rv[:, :, :start_pos, :].float()).abs().max().item())
    return stats


def _run_forward(
    attns: list[torch.nn.Module],
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    cache: StaticCache,
    cache_position: torch.Tensor,
) -> None:
    out = hidden_states
    with torch.no_grad(), decode_qk_rotary_graph_linear_mode(True):
        for attn in attns:
            out, _ = attn(
                out,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
            )


def run_case(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(int(args.seed))

    q_len = int(args.q_len)
    start_pos = int(args.start_pos)
    max_cache_len = int(args.max_cache_len)
    total_len = start_pos + q_len
    if total_len > max_cache_len:
        raise RuntimeError(f"start_pos + q_len must fit max_cache_len: {total_len} > {max_cache_len}")

    os.environ.setdefault("AICAS_DECODE_QK_ROTARY_EXT_MAX_Q", str(max(16, q_len)))
    layers_to_run = max(1, int(args.layers_to_run))
    args.num_layers = max(int(args.num_layers), layers_to_run)
    cfg = _make_config(args)
    graph_attns = [_make_attention(cfg, device, dtype, idx) for idx in range(layers_to_run)]
    ref_attns = [_make_attention(cfg, device, dtype, idx) for idx in range(layers_to_run)]
    for ref_attn, graph_attn in zip(ref_attns, graph_attns):
        ref_attn.load_state_dict(graph_attn.state_dict())
    _patch_attentions(graph_attns)
    _patch_attentions(ref_attns)

    cache_position = torch.arange(start_pos, total_len, device=device, dtype=torch.long)
    hidden_buf = torch.empty(1, q_len, int(args.hidden_size), device=device, dtype=dtype)
    position_embeddings = _make_position_embeddings(cfg, hidden_buf, cache_position)
    attention_mask = None
    if not args.no_mask:
        attention_mask = _make_attention_mask(dtype, device, q_len, start_pos, max_cache_len)

    prefix_k = torch.randn(layers_to_run, 1, int(args.num_kv_heads), start_pos, int(args.head_dim), device=device, dtype=dtype)
    prefix_v = torch.randn_like(prefix_k)

    warm_hidden = torch.randn_like(hidden_buf)
    warm_cache = _make_static_cache(cfg, device, dtype, prefix_k, prefix_v, start_pos, max_cache_len)
    _run_forward(graph_attns, warm_hidden, position_embeddings, attention_mask, warm_cache, cache_position)
    torch.cuda.synchronize()

    graph_cache = _make_static_cache(cfg, device, dtype, prefix_k, prefix_v, start_pos, max_cache_len)
    hidden_buf.copy_(torch.randn_like(hidden_buf))
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), decode_qk_rotary_graph_linear_mode(True):
        with torch.cuda.graph(graph):
            out = hidden_buf
            for attn in graph_attns:
                out, _ = attn(
                    out,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=graph_cache,
                    cache_position=cache_position,
                )

    replay_hidden = torch.randn_like(hidden_buf)
    for layer_idx in range(layers_to_run):
        graph_cache.layers[layer_idx].keys[:, :, :start_pos, :].copy_(prefix_k[layer_idx])
        graph_cache.layers[layer_idx].values[:, :, :start_pos, :].copy_(prefix_v[layer_idx])
        graph_cache.layers[layer_idx].keys[:, :, start_pos:, :].zero_()
        graph_cache.layers[layer_idx].values[:, :, start_pos:, :].zero_()
    hidden_buf.copy_(replay_hidden)
    graph.replay()
    torch.cuda.synchronize()

    ref_cache = _make_static_cache(cfg, device, dtype, prefix_k, prefix_v, start_pos, max_cache_len)
    _run_forward(ref_attns, replay_hidden, position_embeddings, attention_mask, ref_cache, cache_position)
    torch.cuda.synchronize()

    tol = float(args.tol)
    failed = False
    for layer_idx in range(layers_to_run):
        stats = _region_stats(graph_cache, ref_cache, start_pos, total_len, layer_idx)
        print(
            "decode_qk_rotary graph cache "
            f"layer={layer_idx} layers={layers_to_run} "
            f"q={q_len} start={start_pos} max_cache={max_cache_len} dtype={dtype} "
            f"graph_linear={os.getenv('AICAS_DECODE_GRAPH_LINEAR_BACKEND', 'cublaslt')} "
            f"mask={int(attention_mask is not None)} "
            f"prefix_k={stats['prefix_k']:.6g} prefix_v={stats['prefix_v']:.6g} "
            f"append_k={stats['append_k']:.6g} append_v={stats['append_v']:.6g} "
            f"graph_append_abs=({stats['graph_append_k_abs']:.6g},{stats['graph_append_v_abs']:.6g}) "
            f"ref_append_abs=({stats['ref_append_k_abs']:.6g},{stats['ref_append_v_abs']:.6g})"
        )
        failed = failed or stats["prefix_k"] > tol or stats["prefix_v"] > tol or stats["append_k"] > tol or stats["append_v"] > tol
    if failed:
        raise RuntimeError("decode qk rotary graph cache mismatch")
    print("decode qk rotary graph cache validation passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--q-len", type=int, default=15)
    parser.add_argument("--start-pos", type=int, default=817)
    parser.add_argument("--max-cache-len", type=int, default=832)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--layers-to-run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--no-mask", action="store_true")
    run_case(parser.parse_args())


if __name__ == "__main__":
    main()
