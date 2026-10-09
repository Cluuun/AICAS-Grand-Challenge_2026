"""Language Model 前向传播优化：跳过不必要操作 + RoPE 预计算。"""

import math
import types
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _precompute_rope_table(rotary_emb, max_pos, device, dtype):
    """Precompute interleaved mrope cos/sin for all text positions.
    For decode, all 3 mrope dimensions use the same position.
    Returns cos_table, sin_table of shape [max_pos, head_dim].
    """
    inv_freq = rotary_emb.inv_freq.float()  # [head_dim//2]
    mrope_section = rotary_emb.mrope_section

    # Compute freqs for positions 0..max_pos-1
    positions = torch.arange(max_pos, dtype=torch.float32, device=device)
    # freqs: [max_pos, head_dim//2]
    freqs = torch.outer(positions, inv_freq)

    # Apply interleaved mrope: all 3 dims use same position for text
    # freqs_t starts as dim T, then patches in H and W at interleaved positions
    freqs_t = freqs.clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        # For text-only decode, all dims have same freqs, so this is identity
        freqs_t[:, idx] = freqs[:, idx]

    emb = torch.cat([freqs_t, freqs_t], dim=-1)  # [max_pos, head_dim]
    cos_table = emb.cos().to(dtype=dtype)
    sin_table = emb.sin().to(dtype=dtype)
    return cos_table, sin_table


def patch_language_model_forward(model):
    """Patch Qwen3VLTextModel.forward to skip unnecessary ops during decode.

    During decode (seq_len=1 with cache):
    - create_causal_mask is unnecessary (flash_attn handles masking internally)
    - DeepStack visual feature injection is unnecessary (only during prefill)
    - mrope cos/sin precomputed and looked up by position (saves ~80us/step)
    """
    lm = model.model.language_model
    _orig_forward = lm.__class__.forward

    # Precompute RoPE table
    text_config = getattr(model.config, 'text_config', model.config)
    head_dim = getattr(text_config, 'head_dim',
                        text_config.hidden_size // text_config.num_attention_heads)
    max_pos = 4096  # max sequence length for precomputation
    try:
        cos_table, sin_table = _precompute_rope_table(
            lm.rotary_emb, max_pos, device=next(lm.parameters()).device,
            dtype=torch.float16)
        lm._rope_cos_table = cos_table  # [max_pos, head_dim]
        lm._rope_sin_table = sin_table  # [max_pos, head_dim]
    except Exception:
        lm._rope_cos_table = None
        lm._rope_sin_table = None

    # Position-dependent perturbation table for layer-skip degeneration fix.
    # When entire layers are skipped, the hidden state can collapse into
    # degenerate attractors causing repetitive output (e.g., "0000...", "2222...").
    # Adding a tiny position-indexed perturbation to the embedding breaks this
    # positive feedback loop.  Inspired by:
    #   - Reservoir computing / echo-state theory (perturbation prevents convergence)
    #   - SLED (Google): intermediate-layer signal preserves output diversity
    #   - FlexiDepth: per-token adaptive depth prevents quality collapse
    _skip_noise_scale = float(os.environ.get("AICAS_SKIP_NOISE_SCALE", "0.0"))
    if _skip_noise_scale > 0:
        hidden_size = text_config.hidden_size
        device = next(lm.parameters()).device
        # Pre-compute a fixed random perturbation per position (deterministic, CUDA-graph safe).
        # Use orthogonal-ish vectors so consecutive positions get diverse perturbations.
        with torch.no_grad():
            raw = torch.randn(max_pos, hidden_size, device='cpu', dtype=torch.float32)
            # Normalize each row to unit norm, then scale
            raw = raw / (raw.norm(dim=1, keepdim=True) + 1e-8)
            lm._skip_noise_table = raw.to(device=device, dtype=torch.float16)  # [max_pos, hidden]
        lm._skip_noise_scale = _skip_noise_scale
        logger.info(f"[skip_noise] Initialized perturbation table: scale={_skip_noise_scale}")
    else:
        lm._skip_noise_table = None
        lm._skip_noise_scale = 0.0

    def patched_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=False,
                        output_hidden_states=None, **kwargs):
        # True single-token decode (CUDA graph path)
        is_decode = (input_ids is not None and input_ids.shape[-1] == 1
                     and past_key_values is not None)

        # Batch verify with layer skipping: EAGLE-3 fast verify processes T>1
        # tokens through a reduced set of layers. Controlled by
        # AICAS_BATCH_VERIFY_SKIP_LAYERS env var (set temporarily by
        # batch_verify_fast). When set, layer skipping is enabled even
        # for multi-token inputs, but RoPE still uses the standard path
        # (not the precomputed single-position table).
        _batch_skip_spec = os.environ.get("AICAS_BATCH_VERIFY_SKIP_LAYERS", "").strip()
        _apply_skip = is_decode or bool(_batch_skip_spec)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # EAGLE-3 needs intermediate hidden states for 3-layer concat draft input.
        # Collect at every layer boundary when requested.
        _collect_hiddens = output_hidden_states is True or output_hidden_states
        # .clone() critical: decoder layers modify hidden_states in-place (residual.add_),
        # so the initial inputs_embeds tensor gets mutated. Without cloning, all_hidden_states[0]
        # would end up with the same values as all_hidden_states[1] (layer 0 output).
        all_hidden_states = () if not _collect_hiddens else (inputs_embeds.clone(),)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        attention_mask = None

        hidden_states = inputs_embeds

        # --- Layer skip configuration ---
        decode_skip_last = int(os.environ.get("AICAS_DECODE_SKIP_LAST_LAYERS", "0")) if is_decode else 0
        decode_skip_start = len(self.layers) - decode_skip_last
        decode_skip_layers = set()
        decode_skip_mlp_layers = set()
        if _apply_skip:
            # For batch verify skip, use the batch-specific spec;
            # for real decode, use the standard decode skip env vars.
            if _batch_skip_spec:
                skip_spec = _batch_skip_spec
            else:
                skip_spec = os.environ.get("AICAS_DECODE_SKIP_LAYERS", "").strip()
            if skip_spec:
                decode_skip_layers = {int(x) for x in skip_spec.replace("+", ",").split(",") if x.strip()}
            if is_decode:
                skip_mlp_spec = os.environ.get("AICAS_DECODE_SKIP_MLP_LAYERS", "").strip()
                if skip_mlp_spec:
                    decode_skip_mlp_layers = {int(x) for x in skip_mlp_spec.replace("+", ",").split(",") if x.strip()}

        # --- RoPE: only use precomputed table for true single-token decode ---
        if is_decode and self._rope_cos_table is not None:
            pos = position_ids[0, 0, :1]
            cos_slice = self._rope_cos_table[pos]
            sin_slice = self._rope_sin_table[pos]
            position_embeddings = (cos_slice, sin_slice)
        else:
            # Standard RoPE for prefill and batch verify (multi-position)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        _skip_noise_applied = False
        for layer_idx, decoder_layer in enumerate(self.layers):
            if _apply_skip and (
                (decode_skip_last > 0 and layer_idx >= decode_skip_start) or
                layer_idx in decode_skip_layers
            ):
                # Anti-degeneration noise (decode only, not batch verify)
                if (not _skip_noise_applied and is_decode
                        and len(decode_skip_layers) >= 10
                        and self._skip_noise_scale > 0
                        and self._skip_noise_table is not None
                        and text_position_ids is not None):
                    pos_idx = text_position_ids[0, 0] % self._skip_noise_table.shape[0]
                    hidden_states = hidden_states + (
                        self._skip_noise_table[pos_idx].unsqueeze(0).unsqueeze(0)
                        * self._skip_noise_scale
                    )
                    _skip_noise_applied = True
                if _collect_hiddens:
                    all_hidden_states = all_hidden_states + (hidden_states.clone(),)
                continue
            _skip_mlp = is_decode and layer_idx in decode_skip_mlp_layers
            if _skip_mlp:
                if not hasattr(decoder_layer, '_skip_mlp_perturb_eps'):
                    _eps = float(os.environ.get("AICAS_SKIP_MLP_PERTURB_EPS", "0"))
                    decoder_layer._skip_mlp_perturb_eps = _eps
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                skip_mlp=_skip_mlp,
                **kwargs,
            )
            hidden_states = layer_outputs

            if _collect_hiddens:
                all_hidden_states = all_hidden_states + (hidden_states.clone(),)

            # DeepStack visual injection: only during prefill (not decode/batch-verify)
            if not _apply_skip and hasattr(self, '_deepstack_process'):
                deepstack_visual_embeds = kwargs.get('deepstack_visual_embeds', None)
                visual_pos_masks = kwargs.get('visual_pos_masks', None)
                if deepstack_visual_embeds is not None and visual_pos_masks is not None:
                    if layer_idx < len(deepstack_visual_embeds):
                        hidden_states = self._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_idx],
                        )

        if not (is_decode and getattr(self, '_skip_final_norm', False)):
            hidden_states = self.norm(hidden_states)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if _collect_hiddens else None,
        )

    lm.forward = types.MethodType(patched_forward, lm)
    return 1
