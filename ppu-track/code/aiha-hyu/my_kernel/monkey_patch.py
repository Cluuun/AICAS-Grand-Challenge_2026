"""
[SHLEE] Monkey-patching entry point.

Walks the model tree and replaces modules / forward methods with
optimized versions from my_kernel.  Called once from evaluation_wrapper.py.
"""

import torch
import torch.nn as nn

from my_kernel.linear_patch_embed import LinearVisionPatchEmbed
from my_kernel.fused_rmsnorm import FusedRMSNorm, _load as _load_rmsnorm
from my_kernel.fused_layernorm import FusedLayerNorm, _load as _load_layernorm
from my_kernel.fused_rope_ver6 import (
    fused_apply_rotary_pos_emb,
    fused_apply_rotary_pos_emb_vision,
    _load as _load_rope,
)
from my_kernel.fused_silu_mul_ver2 import _load as _load_silu_mul
from my_kernel.fused_gelu import fused_gelu, _load as _load_gelu
from my_kernel.fused_qkv import FusedQKVProjection
from my_kernel.fused_gate_up import FusedGateUpProjection
from my_kernel.cuda_graph_runner import OptimizedRunner
from my_kernel.fused_deepstack_process import fused_deepstack_process, _load as _load_deepstack_process

def _precompile_kernels():
    """JIT-compile all CUDA extensions up-front (once)."""
    import os as _os
    print("[my_kernel] Compiling CUDA kernels...")
    loaders = [
        ("rmsnorm", _load_rmsnorm),
        ("layernorm", _load_layernorm),
        ("rope", _load_rope),
        ("silu_mul", _load_silu_mul),
        ("gelu", _load_gelu),
        ("deepstack", _load_deepstack_process),
    ]
    _skip_rope = _os.environ.get("MY_KERNEL_SKIP_FUSED_ROPE", "0") == "1"
    ok = []
    for name, fn in loaders:
        if name == "rope" and _skip_rope:
            print(f"[my_kernel]   - {name} skipped (MY_KERNEL_SKIP_FUSED_ROPE=1)")
            continue
        try:
            fn()
            ok.append(name)
        except Exception as e:
            print(f"[my_kernel]   WARNING: {name} compilation failed: {e}")
    print(f"[my_kernel] Compiled kernels: {ok}")

def _patch_vision_encoder(model: nn.Module):
    """Replace vision encoder ops with fused versions."""
    visual = model.model.visual

    pe = visual.patch_embed
    visual.patch_embed = LinearVisionPatchEmbed(
        pe.proj,
        in_channels=pe.in_channels,
        temporal_patch_size=pe.temporal_patch_size,
        patch_size=pe.patch_size,
        embed_dim=pe.embed_dim,
    )
    print("[my_kernel]   + LinearVisionPatchEmbed")

    n_blocks = len(visual.blocks)
    for blk in visual.blocks:
        blk.norm1 = FusedLayerNorm(blk.norm1.weight.data, blk.norm1.bias.data,
                                   blk.norm1.eps)
        blk.norm2 = FusedLayerNorm(blk.norm2.weight.data, blk.norm2.bias.data,
                                   blk.norm2.eps)
    print(f"[my_kernel]   + FusedLayerNorm in {n_blocks} vision blocks")

    merger = visual.merger
    if hasattr(merger, "norm"):
        merger.norm = FusedLayerNorm(merger.norm.weight.data,
                                    merger.norm.bias.data,
                                    merger.norm.eps)
        print("[my_kernel]   + FusedLayerNorm in merger(s)")

    import os as _os, transformers.models.qwen3_vl.modeling_qwen3_vl as _mod
    if _os.environ.get("MY_KERNEL_SKIP_FUSED_ROPE", "0") != "1":
        _mod.apply_rotary_pos_emb_vision = fused_apply_rotary_pos_emb_vision
        print("[my_kernel]   + fused_apply_rotary_pos_emb_vision")
    else:
        print("[my_kernel]   - fused_apply_rotary_pos_emb_vision (SKIPPED)")

    for blk in visual.blocks:
        _patch_vision_mlp_forward(blk.mlp)
    print(f"[my_kernel]   + fused_gelu in {n_blocks} vision MLP layers")

    _patch_merger_mlp_forward(merger)
    print("[my_kernel]   + fused_gelu in merger MLP")

    if hasattr(visual, "deepstack_merger_list"):
        for ds_merger in visual.deepstack_merger_list:
            ds_merger.norm = FusedLayerNorm(ds_merger.norm.weight.data,
                                            ds_merger.norm.bias.data,
                                            ds_merger.norm.eps)
            _patch_merger_mlp_forward(ds_merger)
        print(f"[my_kernel]   + FusedLayerNorm + fused_gelu in {len(visual.deepstack_merger_list)} deepstack mergers")

def _patch_text_decoder(model: nn.Module):
    """Replace text decoder ops with fused versions."""
    lang = model.model.language_model

    n_layers = len(lang.layers)

    for layer in lang.layers:
        layer.input_layernorm = FusedRMSNorm(
            layer.input_layernorm.weight.data,
            layer.input_layernorm.variance_epsilon)
        layer.post_attention_layernorm = FusedRMSNorm(
            layer.post_attention_layernorm.weight.data,
            layer.post_attention_layernorm.variance_epsilon)
        layer.self_attn.q_norm = FusedRMSNorm(
            layer.self_attn.q_norm.weight.data,
            layer.self_attn.q_norm.variance_epsilon)
        layer.self_attn.k_norm = FusedRMSNorm(
            layer.self_attn.k_norm.weight.data,
            layer.self_attn.k_norm.variance_epsilon)

    lang.norm = FusedRMSNorm(lang.norm.weight.data, lang.norm.variance_epsilon)
    print(f"[my_kernel]   + FusedRMSNorm in {n_layers} decoder layers + final norm")

    for layer in lang.layers:
        _patch_fused_qkv(layer.self_attn)
    print(f"[my_kernel]   + FusedQKV in {n_layers} attention layers")

    import os as _os, transformers.models.qwen3_vl.modeling_qwen3_vl as _mod
    if _os.environ.get("MY_KERNEL_SKIP_FUSED_ROPE", "0") != "1":
        _mod.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
        print("[my_kernel]   + fused_apply_rotary_pos_emb (text)")
    else:
        print("[my_kernel]   - fused_apply_rotary_pos_emb (text) (SKIPPED)")

    for layer in lang.layers:
        layer.mlp = FusedGateUpProjection(layer.mlp)
    print(f"[my_kernel]   + FusedGateUp in {n_layers} MLP layers")

    _patch_deepstack_process(model)

def _patch_fused_qkv(attn: nn.Module):
    """Replace q/k/v_proj + q/k_norm with a single fused QKV projection,
    and patch the attention forward to use it."""
    import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen_mod
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    fused = FusedQKVProjection(attn)
    attn.fused_qkv = fused

    del attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm

    def fused_attn_forward(
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]

        query_states, key_states, value_states = attn.fused_qkv(hidden_states)

        cos, sin = position_embeddings
        query_states, key_states = _qwen_mod.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin, "cos": cos, "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs,
            )

        _impl_name = attn.config._attn_implementation
        if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                _impl_name, _qwen_mod.eager_attention_forward,
            )
        else:
            try:
                attention_interface = ALL_ATTENTION_FUNCTIONS[_impl_name]
            except (KeyError, TypeError):
                attention_interface = _qwen_mod.eager_attention_forward
        attn_output, _ = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, None

    attn.forward = fused_attn_forward

def _patch_vision_mlp_forward(mlp: nn.Module):
    """Replace vision MLP forward to use fused GELU (tanh approx)."""

    def fused_forward(hidden_state):
        return mlp.linear_fc2(fused_gelu(mlp.linear_fc1(hidden_state)))

    mlp.forward = fused_forward

def _patch_merger_mlp_forward(merger: nn.Module):
    """Replace merger forward to use fused GELU."""

    def fused_forward(x):
        x = merger.norm(
            x.view(-1, merger.hidden_size) if merger.use_postshuffle_norm else x
        ).view(-1, merger.hidden_size)
        x = merger.linear_fc2(fused_gelu(merger.linear_fc1(x)))
        return x

    merger.forward = fused_forward

def _patch_deepstack_process(model: nn.Module):

    try:
        lm = model.model.language_model
        cls = type(lm)
        if getattr(cls, "_deepstack_fused_installed", False):
            return
        orig = cls._deepstack_process

        def fused_forward(self, hidden_states, visual_pos_masks, visual_embeds):
            v = visual_embeds.shape[0]
            if v == 0:
                return hidden_states

            fused_deepstack_process(hidden_states, visual_embeds, 4)
            return hidden_states

        cls._deepstack_process = fused_forward
        cls._deepstack_fused_installed = True
        print(f"[my_kernel]   + fused_deepstack")
    except Exception as e:
        print(f"[my_kernel]   FAIL: fused_deepstack: {e}")

def _exact_layer_skip(model: nn.Module):
    import os
    from my_kernel.layer_skip import ExactQwen3VLText

    raw = os.environ.get("MY_KERNEL_EXACT_LAYERS", "")
    if not raw.strip():
        print("[my_kernel]   + ExactLayerSkip: MY_KERNEL_EXACT_LAYERS not set, skipping")
        return

    layer_indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    exact = ExactQwen3VLText(model, layer_indices=layer_indices)
    n_before = len(exact.layers)
    pruned = exact.remove_layers()
    print(
        f"[my_kernel]   + ExactLayerSkip: {len(pruned)}/{n_before} removed "
        f"(indices: {pruned}), {len(exact.layers)} remaining"
    )

def _exact_vit_layer_skip(model: nn.Module):
    import os
    from my_kernel.layer_skip import ExactQwen3VLVIT

    raw = os.environ.get("MY_KERNEL_EXACT_VIT_LAYERS", "4")
    if not raw.strip():
        print("[my_kernel]   + ExactLayerSkip-VIT: MY_KERNEL_EXACT_VIT_LAYERS not set, skipping")
        return

    block_indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    exact = ExactQwen3VLVIT(model, block_indices=block_indices)
    n_before = len(exact.blocks)
    pruned = exact.remove_blocks()
    print(
        f"[my_kernel]   + ExactLayerSkip-VIT: {len(pruned)}/{n_before} removed "
        f"(indices: {pruned}), {len(exact.blocks)} remaining"
    )

def apply_optimizations(model: nn.Module, processor, device: str = "cuda:0"):
    print("[my_kernel] Applying optimizations...")

    _precompile_kernels()

    import os as _os
    _vit_skip_mode = "none"
    if _vit_skip_mode == "exact":
        _exact_vit_layer_skip(model)

    _patch_vision_encoder(model)

    _skip_mode = "exact"
    if _skip_mode == "exact":
        _exact_layer_skip(model)

    import os
    _llm_drop = int(os.environ.get("MY_KERNEL_LLM_DROP_TAIL", "0"))
    if _llm_drop > 0:
        lang = model.model.language_model
        keep = max(1, len(lang.layers) - _llm_drop)
        while len(lang.layers) > keep:
            del lang.layers[-1]

        try:
            model.config.get_text_config().num_hidden_layers = keep
        except Exception:
            pass
        print(f"[my_kernel]   + LLM tail-drop: {keep} layers")

    _patch_text_decoder(model)

    try:
        from my_kernel.quant_swap import apply_quant, is_quant_enabled
        if is_quant_enabled():
            apply_quant(model, device=device)
            print("[my_kernel]   + quant_swap applied")
    except Exception as e:
        print(f"[my_kernel]   FAIL: quant_swap: {e}")

    try:
        from my_kernel.lean_prefill import install_lean_prefill
        install_lean_prefill(model)
        print("[my_kernel]   + Lean prefill (bypass nn.Module dispatch)")
    except Exception as e:
        print(f"[my_kernel]   FAIL: Lean prefill: {e}")

    try:
        from my_kernel.lean_vision import install_lean_vision
        install_lean_vision(model)
        print("[my_kernel]   + Lean vision encoder (bypass nn.Module dispatch)")
    except Exception as e:
        print(f"[my_kernel]   FAIL: Lean vision: {e}")

    try:
        tiny_k = int(os.environ.get("MY_KERNEL_TINY_LM_HEAD_K", "0"))
        if tiny_k > 0:
            from my_kernel.quant_swap import make_tiny_lm_head
            low_cap = int(os.environ.get("MY_KERNEL_TINY_LM_HEAD_LOW_CAP", "4000"))
            make_tiny_lm_head(model, tiny_k, low_cap, device=device)
            print(f"[my_kernel]   + tiny lm_head (K={tiny_k}, LOW_CAP={low_cap})")
    except Exception as e:
        print(f"[my_kernel]   FAIL: tiny lm_head: {e}")

    print("[my_kernel] Building OptimizedRunner...")
    runner = OptimizedRunner(model, processor, device=device)
    print("[my_kernel] All optimizations applied.")

    return runner
