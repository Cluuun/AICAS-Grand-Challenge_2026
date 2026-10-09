"""vAxel33: flag_gems fused_add_rms_norm based decode forward.

Strategy:
  - Replace the standard transformers Qwen3VLTextModel decode loop with a
    hand-rolled loop that threads (hidden_normed, residual) across layers.
  - Each layer does:
      attn_out = self_attn(hidden_normed)
      # in-place: attn_out := rmsnorm(attn_out + residual) * post_norm.weight,
      #           residual := attn_out + residual
      fused_add_rms_norm(attn_out, residual, post_norm.weight, eps)
      mlp_out = mlp(attn_out)
      # in-place: mlp_out := rmsnorm(mlp_out + residual) * NEXT_norm.weight,
      #           residual := mlp_out + residual
      fused_add_rms_norm(mlp_out, residual, NEXT_norm.weight, eps)
  - Last layer's "NEXT_norm" is the final llm_norm — saves the separate final
    rms_norm call.

Op-bench (M=1, hidden=2048, FP16, eager):
  fused_add_rms_norm 39.5us  vs  torch (add + rms separated) 60.9us → 1.54x

Fusion sites per layer = 2; layers = 28 → 56 fusions per token.
Savings ≈ 21us × 56 = 1.18ms/token (eager bench upper bound). In CUDA-graph
mode the savings are smaller but still compounding across all replays.

Wraps the in-place op as a torch.library.custom_op so dynamo + CUDA-graph
treat it as a black-box kernel call (matching the flash_attn pattern in
aicas_runtime/custom_ops.py).

Modular hooks:
  - `enable_flag_gems()` does nothing structural; flag_gems ops are accessed
    explicitly via the custom_op wrapper.
  - `make_fast_decode_fns(...)` returns (decode_step, decode_n_steps) drop-in
    replacements that match make_decode_fns from aicas_runtime/decode.py.

Toggle: env AICAS_FAST_DECODE=1 (set in evaluation_wrapper_vAxel33).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_flag_gems_loaded():
    """Idempotent import. flag_gems registers Triton kernels on import."""
    import flag_gems  # noqa: F401


# ── custom_op wrapper for fused_add_rms_norm ──
# in-place mutation of x and residual; output is in `x` after the call.

@torch.library.custom_op(
    'my::fused_add_rms_norm', mutates_args=('x', 'residual'))
def fused_add_rms_norm_op(
    x: torch.Tensor, residual: torch.Tensor,
    weight: torch.Tensor, eps: float,
) -> None:
    import flag_gems
    flag_gems.fused_add_rms_norm(
        x, residual, [weight.shape[0]], weight, eps)


@fused_add_rms_norm_op.register_fake
def _fused_add_rms_norm_fake(
    x: torch.Tensor, residual: torch.Tensor,
    weight: torch.Tensor, eps: float,
) -> None:
    return None


# ── flag_gems plain rms_norm wrapper (for the very first norm in the loop) ──

@torch.library.custom_op('my::gems_rms_norm', mutates_args=())
def gems_rms_norm_op(
    x: torch.Tensor, weight: torch.Tensor, eps: float,
) -> torch.Tensor:
    import flag_gems
    return flag_gems.rms_norm(x, [weight.shape[0]], weight, eps)


@gems_rms_norm_op.register_fake
def _gems_rms_norm_fake(
    x: torch.Tensor, weight: torch.Tensor, eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


def make_fast_lm_forward(language_model):
    """Build a hand-rolled forward for Qwen3VLTextModel that uses
    fused_add_rms_norm threading.

    Returns a callable that accepts (input_ids, position_ids, fa_seqlens) and
    returns last_hidden_state[:, -1, :] (already normed by the final fused
    op — the caller does lm_head_apply on the returned tensor).
    """
    _ensure_flag_gems_loaded()

    embed_tokens = language_model.embed_tokens
    layers = language_model.layers
    final_norm = language_model.norm
    rotary_emb = language_model.rotary_emb

    n_layers = len(layers)

    # Pre-extract per-layer norm weights and modules for tight loop access.
    layer_input_norm_weights = [l.input_layernorm.weight for l in layers]
    layer_post_norm_weights = [l.post_attention_layernorm.weight for l in layers]
    layer_input_norm_eps = [l.input_layernorm.variance_epsilon for l in layers]
    layer_post_norm_eps = [l.post_attention_layernorm.variance_epsilon for l in layers]
    final_norm_weight = final_norm.weight
    final_norm_eps = final_norm.variance_epsilon

    self_attns = [l.self_attn for l in layers]
    mlps = [l.mlp for l in layers]

    def _forward(input_ids, position_ids, fa_seqlens):
        inputs_embeds = embed_tokens(input_ids)
        cos, sin = rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        # ── First norm: separate (no residual yet) ──
        residual = inputs_embeds
        hidden_states = gems_rms_norm_op(
            inputs_embeds, layer_input_norm_weights[0], layer_input_norm_eps[0])

        for i in range(n_layers):
            # ── Self-Attention ──
            hidden_states, _ = self_attns[i](
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
            )

            # ── Fuse 1: (hidden + residual) → residual; rmsnorm → hidden ──
            #   weight = post_attn_layernorm
            fused_add_rms_norm_op(
                hidden_states, residual,
                layer_post_norm_weights[i], layer_post_norm_eps[i])

            # ── MLP ──
            hidden_states = mlps[i](hidden_states)

            # ── Fuse 2: (hidden + residual) → residual; rmsnorm → hidden ──
            # Use NEXT layer's input_layernorm weight; for the LAST layer,
            # use final_norm so we save the separate final rmsnorm.
            if i + 1 < n_layers:
                next_w = layer_input_norm_weights[i + 1]
                next_eps = layer_input_norm_eps[i + 1]
            else:
                next_w = final_norm_weight
                next_eps = final_norm_eps
            fused_add_rms_norm_op(
                hidden_states, residual, next_w, next_eps)

        # `hidden_states` is now final-normed (rmsnorm of last residual stream).
        return hidden_states

    return _forward


def make_fast_lm_forward_skip(language_model, active_layers):
    """Variant of make_fast_lm_forward that runs only `active_layers` (sorted
    list of indices into language_model.layers). Skipped layers are treated as
    identity — residual passes through unchanged, KV cache for those layers is
    not extended at decode positions.

    Maintains the same residual-thread + fused_add_rms_norm pattern. The first
    active layer's input_layernorm is used for the initial gems_rms_norm, the
    last active layer's "next norm" is final_norm.
    """
    _ensure_flag_gems_loaded()

    embed_tokens = language_model.embed_tokens
    layers = language_model.layers
    final_norm = language_model.norm
    rotary_emb = language_model.rotary_emb

    active = sorted(set(int(i) for i in active_layers))
    assert len(active) > 0, "active_layers must be non-empty"
    assert all(0 <= i < len(layers) for i in active), "out-of-range layer index"

    layer_input_norm_weights = [layers[i].input_layernorm.weight for i in active]
    layer_post_norm_weights = [layers[i].post_attention_layernorm.weight for i in active]
    layer_input_norm_eps = [layers[i].input_layernorm.variance_epsilon for i in active]
    layer_post_norm_eps = [layers[i].post_attention_layernorm.variance_epsilon for i in active]
    final_norm_weight = final_norm.weight
    final_norm_eps = final_norm.variance_epsilon

    self_attns = [layers[i].self_attn for i in active]
    mlps = [layers[i].mlp for i in active]

    n_active = len(active)

    def _forward(input_ids, position_ids, fa_seqlens):
        inputs_embeds = embed_tokens(input_ids)
        cos, sin = rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        residual = inputs_embeds
        hidden_states = gems_rms_norm_op(
            inputs_embeds, layer_input_norm_weights[0], layer_input_norm_eps[0])

        for i in range(n_active):
            hidden_states, _ = self_attns[i](
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
            )

            fused_add_rms_norm_op(
                hidden_states, residual,
                layer_post_norm_weights[i], layer_post_norm_eps[i])

            hidden_states = mlps[i](hidden_states)

            if i + 1 < n_active:
                next_w = layer_input_norm_weights[i + 1]
                next_eps = layer_input_norm_eps[i + 1]
            else:
                next_w = final_norm_weight
                next_eps = final_norm_eps
            fused_add_rms_norm_op(
                hidden_states, residual, next_w, next_eps)

        return hidden_states

    return _forward


def make_fast_lm_forward_mlp_skip(language_model, mlp_skip_layers):
    """Variant that runs all 28 attention sublayers but skips MLP on
    `mlp_skip_layers` (set of indices). Each skipped MLP layer collapses the
    two fused_add_rms_norm calls into one (fuse the post-attn-add directly
    with the next layer's input-norm, since MLP is identity).

    Rationale: MLP is ~2× MAC of attention per layer; skipping MLP only
    preserves all attention/KV info while trimming the heaviest path.
    """
    _ensure_flag_gems_loaded()

    embed_tokens = language_model.embed_tokens
    layers = language_model.layers
    final_norm = language_model.norm
    rotary_emb = language_model.rotary_emb

    n_layers = len(layers)
    mlp_skip = set(int(i) for i in mlp_skip_layers)

    layer_input_norm_weights = [l.input_layernorm.weight for l in layers]
    layer_post_norm_weights = [l.post_attention_layernorm.weight for l in layers]
    layer_input_norm_eps = [l.input_layernorm.variance_epsilon for l in layers]
    layer_post_norm_eps = [l.post_attention_layernorm.variance_epsilon for l in layers]
    final_norm_weight = final_norm.weight
    final_norm_eps = final_norm.variance_epsilon

    self_attns = [l.self_attn for l in layers]
    mlps = [l.mlp for l in layers]

    def _forward(input_ids, position_ids, fa_seqlens):
        inputs_embeds = embed_tokens(input_ids)
        cos, sin = rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        residual = inputs_embeds
        hidden_states = gems_rms_norm_op(
            inputs_embeds, layer_input_norm_weights[0], layer_input_norm_eps[0])

        for i in range(n_layers):
            hidden_states, _ = self_attns[i](
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
            )

            if i in mlp_skip:
                # MLP is identity → only the post-attn add+norm pair, then
                # immediately go to next layer's input_norm (or final_norm).
                if i + 1 < n_layers:
                    next_w = layer_input_norm_weights[i + 1]
                    next_eps = layer_input_norm_eps[i + 1]
                else:
                    next_w = final_norm_weight
                    next_eps = final_norm_eps
                fused_add_rms_norm_op(
                    hidden_states, residual, next_w, next_eps)
            else:
                fused_add_rms_norm_op(
                    hidden_states, residual,
                    layer_post_norm_weights[i], layer_post_norm_eps[i])
                hidden_states = mlps[i](hidden_states)
                if i + 1 < n_layers:
                    next_w = layer_input_norm_weights[i + 1]
                    next_eps = layer_input_norm_eps[i + 1]
                else:
                    next_w = final_norm_weight
                    next_eps = final_norm_eps
                fused_add_rms_norm_op(
                    hidden_states, residual, next_w, next_eps)

        return hidden_states

    return _forward


def make_fast_decode_fns(
    *, language_model, lm_head_apply, inv_perm, fa_seqlens, n_steps,
    active_layers=None, mlp_skip_layers=None,
):
    """Drop-in replacement for aicas_runtime.decode.make_decode_fns that
    routes the LM forward through the fused-norm path.

    Three variants (mutually exclusive):
      - default: full 28-layer fast forward
      - active_layers set: full layer skip (vAxel34)
      - mlp_skip_layers set: MLP-only skip (vAxel34b — preserves all attn)
    """
    if active_layers is not None and mlp_skip_layers is not None:
        raise ValueError("active_layers and mlp_skip_layers are mutually exclusive")
    if active_layers is not None:
        fast_lm = make_fast_lm_forward_skip(language_model, active_layers)
    elif mlp_skip_layers is not None:
        fast_lm = make_fast_lm_forward_mlp_skip(language_model, mlp_skip_layers)
    else:
        fast_lm = make_fast_lm_forward(language_model)

    if inv_perm is not None:
        @torch.compile(mode="default", dynamic=False)
        def decode_step(input_ids, position_ids, fa_seqlens):
            hidden = fast_lm(input_ids, position_ids, fa_seqlens)
            h = hidden[:, -1, :]
            small = lm_head_apply(h).argmax(dim=-1, keepdim=True)
            token = inv_perm[small.flatten()].view_as(small)
            fa_seqlens.add_(1)
            return token

        @torch.compile(mode="default", dynamic=False)
        def decode_n_steps(start_token, pos_n, fa_seqlens):
            cur = start_token
            tokens = []
            for i in range(n_steps):
                hidden = fast_lm(cur, pos_n[i], fa_seqlens)
                h = hidden[:, -1, :]
                small = lm_head_apply(h).argmax(dim=-1, keepdim=True)
                cur = inv_perm[small.flatten()].view_as(small)
                tokens.append(cur)
                fa_seqlens.add_(1)
            return torch.cat(tokens, dim=1)
    else:
        @torch.compile(mode="default", dynamic=False)
        def decode_step(input_ids, position_ids, fa_seqlens):
            hidden = fast_lm(input_ids, position_ids, fa_seqlens)
            h = hidden[:, -1, :]
            token = lm_head_apply(h).argmax(dim=-1, keepdim=True)
            fa_seqlens.add_(1)
            return token

        @torch.compile(mode="default", dynamic=False)
        def decode_n_steps(start_token, pos_n, fa_seqlens):
            cur = start_token
            tokens = []
            for i in range(n_steps):
                hidden = fast_lm(cur, pos_n[i], fa_seqlens)
                h = hidden[:, -1, :]
                cur = lm_head_apply(h).argmax(dim=-1, keepdim=True)
                tokens.append(cur)
                fa_seqlens.add_(1)
            return torch.cat(tokens, dim=1)

    return decode_step, decode_n_steps
