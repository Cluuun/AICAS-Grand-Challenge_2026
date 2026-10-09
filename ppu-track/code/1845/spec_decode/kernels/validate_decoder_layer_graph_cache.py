from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from my_kernel.decode_add_rmsnorm_triton.runtime import apply_decode_add_rmsnorm_triton  # noqa: E402
from my_kernel.decode_qk_rotary.runtime import (  # noqa: E402
    apply_decode_qk_rotary_ext_fastpath,
    decode_qk_rotary_graph_linear_mode,
)
from my_kernel.decode_rmsnorm_triton.runtime import apply_decode_rmsnorm_triton  # noqa: E402
from spec_decode.multitoken_attention import spec_multitoken_attention  # noqa: E402
from spec_decode.qk_rotary import spec_qk_rotary  # noqa: E402
from transformers.cache_utils import StaticCache  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    Qwen3VLTextConfig,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRotaryEmbedding,
)


class _LanguageModel(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.config = None


class _ModelContainer(torch.nn.Module):
    def __init__(self, language_model: _LanguageModel, cfg: Qwen3VLTextConfig):
        super().__init__()
        language_model.config = cfg
        self.language_model = language_model
        self.config = cfg


class _PatchWrapper(torch.nn.Module):
    def __init__(self, layers: list[torch.nn.Module], cfg: Qwen3VLTextConfig):
        super().__init__()
        self.config = cfg
        self.model = _ModelContainer(_LanguageModel(layers), cfg)

    def generate(self, *args, **kwargs):
        raise RuntimeError("validate_decoder_layer_graph_cache does not call generate")


def _make_config(args: argparse.Namespace) -> Qwen3VLTextConfig:
    cfg = Qwen3VLTextConfig(
        hidden_size=int(args.hidden_size),
        intermediate_size=int(args.intermediate_size),
        num_attention_heads=int(args.num_heads),
        num_key_value_heads=int(args.num_kv_heads),
        head_dim=int(args.head_dim),
        num_hidden_layers=max(int(args.num_layers), int(args.layers_to_run), 1),
        attention_bias=False,
        hidden_act="silu",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _make_layers(
    cfg: Qwen3VLTextConfig,
    device: torch.device,
    dtype: torch.dtype,
    count: int,
    qk_norm_scale: float = 1.0,
) -> list[Qwen3VLTextDecoderLayer]:
    layers = [Qwen3VLTextDecoderLayer(cfg, layer_idx=idx).to(device=device, dtype=dtype) for idx in range(count)]
    for layer in layers:
        layer.eval()
        if float(qk_norm_scale) != 1.0:
            with torch.no_grad():
                layer.self_attn.q_norm.weight.mul_(float(qk_norm_scale))
                layer.self_attn.k_norm.weight.mul_(float(qk_norm_scale))
        for param in layer.parameters():
            param.requires_grad_(False)
    return layers


def _patch_layers(wrapper: _PatchWrapper, args: argparse.Namespace) -> None:
    os.environ.setdefault("AICAS_DECODE_QK_ROTARY_EXT_MAX_Q", str(max(16, int(args.q_len))))
    if args.flashdecode_attn_patch:
        from my_kernel.flashDecode.flashdecode_runtime import patch_qwen3vl_flashdecode

        patch_qwen3vl_flashdecode(
            wrapper,
            min_seq=int(args.flashdecode_min_seq),
            max_seq=int(args.flashdecode_max_seq),
            max_S=int(args.flashdecode_max_s),
        )
    if not args.no_qk_ext:
        count = apply_decode_qk_rotary_ext_fastpath(wrapper)
        if count != int(args.layers_to_run):
            raise RuntimeError(f"decode qk rotary patch count={count}, expected {args.layers_to_run}")
    if not args.no_add_rmsnorm:
        apply_decode_add_rmsnorm_triton(wrapper)
    if not args.no_rmsnorm:
        apply_decode_rmsnorm_triton(wrapper)


def _make_position_embeddings(
    cfg: Qwen3VLTextConfig,
    hidden_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Qwen3VLTextRotaryEmbedding(cfg).to(device=hidden_states.device)
    position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1).contiguous()
    return rotary(hidden_states, position_ids)


def _make_tree_position_ids(cache_position: torch.Tensor) -> torch.Tensor:
    text_position_ids = cache_position.view(1, -1).contiguous()
    rotary_position_ids = text_position_ids.view(1, 1, -1).expand(3, 1, -1)
    return torch.cat((text_position_ids.unsqueeze(0), rotary_position_ids), dim=0).contiguous()


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


def _reset_static_cache(cache: StaticCache, prefix_k: torch.Tensor, prefix_v: torch.Tensor, start_pos: int) -> None:
    for layer_idx, layer in enumerate(cache.layers):
        layer.keys[:, :, : int(start_pos), :].copy_(prefix_k[layer_idx])
        layer.values[:, :, : int(start_pos), :].copy_(prefix_v[layer_idx])
        layer.keys[:, :, int(start_pos) :, :].zero_()
        layer.values[:, :, int(start_pos) :, :].zero_()
        try:
            layer.cumulative_length = int(start_pos)
        except Exception:
            pass
    cache._aicas_eagle3_seq_len = int(start_pos)


def _run_layers(
    wrapper: _PatchWrapper,
    layers: list[torch.nn.Module],
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    rotary: Qwen3VLTextRotaryEmbedding | None,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    cache: StaticCache,
    cache_position: torch.Tensor,
) -> torch.Tensor:
    out = hidden_states
    with torch.no_grad(), decode_qk_rotary_graph_linear_mode(True), spec_qk_rotary(wrapper), spec_multitoken_attention(wrapper):
        if position_embeddings is None:
            if rotary is None or position_ids is None:
                raise RuntimeError("position_embeddings or rotary+position_ids are required")
            rotary_position_ids = position_ids[1:] if int(position_ids.shape[0]) == 4 else position_ids
            position_embeddings = rotary(out, rotary_position_ids)
        for layer in layers:
            out = layer(
                out,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
    return out


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

    cfg = _make_config(args)
    if args.spec_qk_rotary:
        os.environ["AICAS_SPEC_QK_ROTARY"] = "1"
        os.environ.setdefault("AICAS_SPEC_KERNEL_MAX_Q", str(max(16, q_len)))
    graph_layers = _make_layers(cfg, device, dtype, int(args.layers_to_run), float(args.qk_norm_scale))
    ref_layers = _make_layers(cfg, device, dtype, int(args.layers_to_run), float(args.qk_norm_scale))
    for ref_layer, graph_layer in zip(ref_layers, graph_layers):
        ref_layer.load_state_dict(graph_layer.state_dict())
    graph_wrapper = _PatchWrapper(graph_layers, cfg)
    ref_wrapper = _PatchWrapper(ref_layers, cfg)
    _patch_layers(graph_wrapper, args)
    _patch_layers(ref_wrapper, args)

    cache_position = torch.arange(start_pos, total_len, device=device, dtype=torch.long)
    hidden_buf = torch.empty(1, q_len, int(args.hidden_size), device=device, dtype=dtype)
    rotary = Qwen3VLTextRotaryEmbedding(cfg).to(device=hidden_buf.device)
    position_ids = _make_tree_position_ids(cache_position)
    position_embeddings = None if args.rotary_in_graph else _make_position_embeddings(cfg, hidden_buf, cache_position)
    attention_mask = None if args.no_mask else _make_attention_mask(dtype, device, q_len, start_pos, max_cache_len)
    prefix_k = torch.randn(int(cfg.num_hidden_layers), 1, int(args.num_kv_heads), start_pos, int(args.head_dim), device=device, dtype=dtype)
    prefix_v = torch.randn_like(prefix_k)
    if float(args.prefix_scale) != 1.0:
        prefix_k.mul_(float(args.prefix_scale))
        prefix_v.mul_(float(args.prefix_scale))

    warm_cache = _make_static_cache(cfg, device, dtype, prefix_k, prefix_v, start_pos, max_cache_len)
    warm_hidden = torch.randn_like(hidden_buf).mul_(float(args.hidden_scale))
    _run_layers(graph_wrapper, graph_layers, warm_hidden, position_embeddings, rotary, position_ids, attention_mask, warm_cache, cache_position)
    torch.cuda.synchronize()

    graph_cache = _make_static_cache(cfg, device, dtype, prefix_k, prefix_v, start_pos, max_cache_len)
    hidden_buf.copy_(torch.randn_like(hidden_buf).mul_(float(args.hidden_scale)))
    out_buf = torch.empty_like(hidden_buf)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), decode_qk_rotary_graph_linear_mode(True), spec_qk_rotary(graph_wrapper), spec_multitoken_attention(graph_wrapper):
        with torch.cuda.graph(graph):
            out = hidden_buf
            graph_position_embeddings = position_embeddings
            if graph_position_embeddings is None:
                graph_position_embeddings = rotary(out, position_ids[1:])
            for layer in graph_layers:
                out = layer(
                    out,
                    position_embeddings=graph_position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=graph_cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
            out_buf.copy_(out)

    replay_hidden = torch.randn_like(hidden_buf).mul_(float(args.hidden_scale))
    _reset_static_cache(graph_cache, prefix_k, prefix_v, start_pos)
    hidden_buf.copy_(replay_hidden)
    graph.replay()
    torch.cuda.synchronize()

    ref_cache = _make_static_cache(cfg, device, dtype, prefix_k, prefix_v, start_pos, max_cache_len)
    ref_out = _run_layers(ref_wrapper, ref_layers, replay_hidden, position_embeddings, rotary, position_ids, attention_mask, ref_cache, cache_position)
    torch.cuda.synchronize()

    hidden_diff = float((out_buf.float() - ref_out.float()).abs().max().item())
    print(
        "decoder_layer graph "
        f"layers={int(args.layers_to_run)} q={q_len} start={start_pos} max_cache={max_cache_len} "
        f"dtype={dtype} graph_linear={os.getenv('AICAS_DECODE_GRAPH_LINEAR_BACKEND', 'cublaslt')} "
        f"mask={int(attention_mask is not None)} rotary_in_graph={int(args.rotary_in_graph)} "
        f"spec_qk_rotary={int(os.getenv('AICAS_SPEC_QK_ROTARY', '0') == '1')} "
        f"hidden_scale={float(args.hidden_scale):.6g} prefix_scale={float(args.prefix_scale):.6g} "
        f"qk_norm_scale={float(args.qk_norm_scale):.6g} "
        f"hidden_maxdiff={hidden_diff:.6g}"
    )

    tol = float(args.tol)
    failed = hidden_diff > tol
    for layer_idx in range(int(args.layers_to_run)):
        stats = _region_stats(graph_cache, ref_cache, start_pos, total_len, layer_idx)
        print(
            "decoder_layer graph cache "
            f"layer={layer_idx} "
            f"prefix_k={stats['prefix_k']:.6g} prefix_v={stats['prefix_v']:.6g} "
            f"append_k={stats['append_k']:.6g} append_v={stats['append_v']:.6g} "
            f"graph_append_abs=({stats['graph_append_k_abs']:.6g},{stats['graph_append_v_abs']:.6g}) "
            f"ref_append_abs=({stats['ref_append_k_abs']:.6g},{stats['ref_append_v_abs']:.6g})"
        )
        failed = failed or stats["prefix_k"] > tol or stats["prefix_v"] > tol or stats["append_k"] > tol or stats["append_v"] > tol
    if failed:
        raise RuntimeError("decoder layer graph cache mismatch")
    print("decoder layer graph cache validation passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--q-len", type=int, default=15)
    parser.add_argument("--start-pos", type=int, default=817)
    parser.add_argument("--max-cache-len", type=int, default=832)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--layers-to-run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-qk-ext", action="store_true")
    parser.add_argument("--no-add-rmsnorm", action="store_true")
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--flashdecode-attn-patch", action="store_true")
    parser.add_argument("--flashdecode-min-seq", type=int, default=480)
    parser.add_argument("--flashdecode-max-seq", type=int, default=2067)
    parser.add_argument("--flashdecode-max-s", type=int, default=4096)
    parser.add_argument("--rotary-in-graph", action="store_true")
    parser.add_argument("--spec-qk-rotary", action="store_true")
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--prefix-scale", type=float, default=1.0)
    parser.add_argument("--qk-norm-scale", type=float, default=1.0)
    run_case(parser.parse_args())


if __name__ == "__main__":
    main()
