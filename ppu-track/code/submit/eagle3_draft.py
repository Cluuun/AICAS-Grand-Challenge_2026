"""
Eagle3 Draft Model for speculative decoding with Qwen3-VL-2B.

Architecture: 1 transformer layer (LlamaDecoderLayer) + fc projection + norm + lm_head.
Takes auxiliary hidden states from target model layers (2, 14, 25) and predicts
future tokens autoregressively using a reduced vocabulary (32000 tokens).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from evaluation_wrapper import MODEL_DTYPE
from triton_fused_kernels import triton_rmsnorm, fused_swiglu, batch_rmsnorm, batch_fused_swiglu, fused_residual_rmsnorm, batch_fused_residual_rmsnorm, fused_dual_rmsnorm_cat, fused_qk_rope, fused_greedy_output, fused_embed_dual_rmsnorm_cat

flash_attn_with_kvcache = None
try:
    from fa3_fwd_interface import flash_attn_with_kvcache
except (ImportError, ModuleNotFoundError):
    pass
if flash_attn_with_kvcache is None:
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except (ImportError, ModuleNotFoundError):
        pass
if flash_attn_with_kvcache is None:
    try:
        from flash_attn import flash_attn_with_kvcache
    except (ImportError, ModuleNotFoundError):
        pass
if flash_attn_with_kvcache is None:
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    except (ImportError, ModuleNotFoundError):
        pass


AUX_HIDDEN_STATE_LAYERS = (2, 14, 25)


class Eagle3Model:
    """Eagle3 draft model for speculative decoding.

    Single transformer layer that takes target model hidden states + token embeddings
    to predict future tokens from a 32000-token draft vocabulary.
    """

    def __init__(self, eagle3_path: str, target_model, device: str = "cuda:0"):
        self.device = device
        self.hidden_size = 2048
        self.num_heads = 16
        self.num_kv_heads = 8
        self.head_dim = 128
        self.intermediate_size = 6144
        self.eps = 1e-6
        self.draft_vocab_size = 32000
        self.scaling = 1.0 / (self.head_dim ** 0.5)

        self._load_weights(eagle3_path)
        self._setup_embeddings(target_model)
        self._setup_kv_cache()
        self._setup_rope(target_model)
        self._combine_buf = torch.zeros(1, self.hidden_size * 3, dtype=MODEL_DTYPE, device=device)
        self.num_extra_layers = len(self._extra_layer_weights) if hasattr(self, '_extra_layer_weights') else 0

    def _load_weights(self, path: str):
        with safe_open(f"{path}/model.safetensors", framework="pt") as f:
            # FC projection: 3*hidden -> hidden
            self.fc_weight = f.get_tensor("fc.weight").to(dtype=MODEL_DTYPE, device=self.device)

            # Midlayer norms
            self.input_layernorm_weight = f.get_tensor("midlayer.input_layernorm.weight").to(dtype=MODEL_DTYPE, device=self.device)
            self.hidden_norm_weight = f.get_tensor("midlayer.hidden_norm.weight").to(dtype=MODEL_DTYPE, device=self.device)
            self.post_attn_norm_weight = f.get_tensor("midlayer.post_attention_layernorm.weight").to(dtype=MODEL_DTYPE, device=self.device)

            # Attention weights (QKV input_size=4096 for first layer: embeds+hidden concatenated)
            self.q_proj_weight = f.get_tensor("midlayer.self_attn.q_proj.weight").to(dtype=MODEL_DTYPE, device=self.device)
            self.k_proj_weight = f.get_tensor("midlayer.self_attn.k_proj.weight").to(dtype=MODEL_DTYPE, device=self.device)
            self.v_proj_weight = f.get_tensor("midlayer.self_attn.v_proj.weight").to(dtype=MODEL_DTYPE, device=self.device)
            self.o_proj_weight = f.get_tensor("midlayer.self_attn.o_proj.weight").to(dtype=MODEL_DTYPE, device=self.device)

            # MLP weights
            self.gate_proj_weight = f.get_tensor("midlayer.mlp.gate_proj.weight").to(dtype=MODEL_DTYPE, device=self.device)
            self.up_proj_weight = f.get_tensor("midlayer.mlp.up_proj.weight").to(dtype=MODEL_DTYPE, device=self.device)
            self.down_proj_weight = f.get_tensor("midlayer.mlp.down_proj.weight").to(dtype=MODEL_DTYPE, device=self.device)

            # Final norm
            self.final_norm_weight = f.get_tensor("norm.weight").to(dtype=MODEL_DTYPE, device=self.device)

            # LM head
            full_lm_head = f.get_tensor("lm_head.weight").to(dtype=MODEL_DTYPE, device=self.device)

            # P-Eagle mask_hidden for parallel draft
            if "mask_hidden" in f.keys():
                self.mask_hidden = f.get_tensor("mask_hidden").to(dtype=MODEL_DTYPE, device=self.device)  # [1, 1, 6144]
                self.mask_token_id = 151669
            else:
                self.mask_hidden = None
                self.mask_token_id = None

            # Extra layers (layer 1+) for multi-layer P-Eagle
            self._extra_layer_weights = []
            layer_idx = 0
            while f"extra_layers.{layer_idx}.self_attn.q_proj.weight" in f.keys():
                prefix = f"extra_layers.{layer_idx}"
                lw = {
                    'input_layernorm_weight': f.get_tensor(f"{prefix}.input_layernorm.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'q_proj_weight': f.get_tensor(f"{prefix}.self_attn.q_proj.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'k_proj_weight': f.get_tensor(f"{prefix}.self_attn.k_proj.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'v_proj_weight': f.get_tensor(f"{prefix}.self_attn.v_proj.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'o_proj_weight': f.get_tensor(f"{prefix}.self_attn.o_proj.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'post_attn_norm_weight': f.get_tensor(f"{prefix}.post_attention_layernorm.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'gate_proj_weight': f.get_tensor(f"{prefix}.mlp.gate_proj.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'up_proj_weight': f.get_tensor(f"{prefix}.mlp.up_proj.weight").to(dtype=MODEL_DTYPE, device=self.device),
                    'down_proj_weight': f.get_tensor(f"{prefix}.mlp.down_proj.weight").to(dtype=MODEL_DTYPE, device=self.device),
                }
                lw['gate_up_weight'] = torch.cat([lw['gate_proj_weight'], lw['up_proj_weight']], dim=0)
                self._extra_layer_weights.append(lw)
                layer_idx += 1
            if self._extra_layer_weights:
                print(f"[Eagle3] Loaded {len(self._extra_layer_weights)} extra layers")

            # Vocab mapping: draft_id + d2t[draft_id] = target_id
            d2t = f.get_tensor("d2t").to(device=self.device)
            full_draft_to_target = (torch.arange(self.draft_vocab_size, device=self.device) + d2t).long()

            # Reduce draft vocab if index file exists
            reduced_path = f"{path}/reduced_vocab_draft.pt"
            print("-"*100)
            print(reduced_path)
            import os as _os
            if _os.path.exists(reduced_path):
                used_draft_ids = torch.load(reduced_path, map_location=self.device, weights_only=True)
                num_used = len(used_draft_ids)
                aligned_size = ((num_used + 255) // 256) * 256

                reduced_weight = torch.zeros(aligned_size, full_lm_head.shape[1], dtype=MODEL_DTYPE, device=self.device)
                reduced_weight[:num_used] = full_lm_head[used_draft_ids]
                self.lm_head_weight = reduced_weight
                self.draft_vocab_size = aligned_size

                # reduced_draft_id → target_token_id
                self.draft_to_target = torch.zeros(aligned_size, dtype=torch.long, device=self.device)
                self.draft_to_target[:num_used] = full_draft_to_target[used_draft_ids]

                # target_id → reduced_draft_id (-1 if not in reduced vocab)
                self.target_to_draft = torch.full((151936,), -1, dtype=torch.long, device=self.device)
                self.target_to_draft[self.draft_to_target[:num_used]] = torch.arange(num_used, device=self.device)

                del full_lm_head
                print(f"[Eagle3] Draft vocab reduced: 32000 → {aligned_size} ({aligned_size*2048*2/1024/1024:.1f} MB)")
            else:
                self.lm_head_weight = full_lm_head
                self.draft_to_target = full_draft_to_target
                self.target_to_draft = torch.full((151936,), -1, dtype=torch.long, device=self.device)
                self.target_to_draft[self.draft_to_target] = torch.arange(self.draft_vocab_size, device=self.device)

        # Pre-concatenate gate+up for fused SwiGLU
        self.gate_up_weight = torch.cat([self.gate_proj_weight, self.up_proj_weight], dim=0)

        # Pre-concatenate QKV for single GEMV instead of 3
        self.qkv_weight = torch.cat([self.q_proj_weight, self.k_proj_weight, self.v_proj_weight], dim=0)
        self._q_dim = self.num_heads * self.head_dim
        self._kv_dim = self.num_kv_heads * self.head_dim

        # Pre-concatenate QKV for extra layers
        for lw in self._extra_layer_weights:
            lw['qkv_weight'] = torch.cat([lw['q_proj_weight'], lw['k_proj_weight'], lw['v_proj_weight']], dim=0)

    def _setup_embeddings(self, target_model):
        """Share embedding layer with target model."""
        self.embed_tokens = target_model.model.get_input_embeddings()

        # Pre-compute P-Eagle parallel draft constants
        if self.mask_hidden is not None:
            with torch.no_grad():
                self.mask_hidden_projected = F.linear(
                    self.mask_hidden.view(1, -1), self.fc_weight
                ).squeeze(0)  # [2048]
                self.mask_embed = self.embed_tokens(
                    torch.tensor([self.mask_token_id], device=self.device)
                ).squeeze(0).squeeze(0)  # [2048]

    def _setup_kv_cache(self, max_len=1500):
        """Create KV caches for all layers.

        Must be large enough for the full sequence (prefill + decode).
        """
        # BSND format: [batch=1, max_len, num_kv_heads, head_dim]
        num_extra = len(self._extra_layer_weights) if hasattr(self, '_extra_layer_weights') else 0
        self.k_cache = torch.zeros(
            1, max_len, self.num_kv_heads, self.head_dim,
            dtype=MODEL_DTYPE, device=self.device
        )
        self.v_cache = torch.zeros(
            1, max_len, self.num_kv_heads, self.head_dim,
            dtype=MODEL_DTYPE, device=self.device
        )
        self.extra_k_caches = []
        self.extra_v_caches = []
        for _ in range(num_extra):
            self.extra_k_caches.append(torch.zeros(
                1, max_len, self.num_kv_heads, self.head_dim,
                dtype=MODEL_DTYPE, device=self.device
            ))
            self.extra_v_caches.append(torch.zeros(
                1, max_len, self.num_kv_heads, self.head_dim,
                dtype=MODEL_DTYPE, device=self.device
            ))
        self.cache_seqlens = torch.zeros(1, dtype=torch.int32, device=self.device)

    def _setup_rope(self, target_model):
        """Precompute RoPE table using target model's MRoPE (theta=5M)."""
        max_pos = 1500
        lang_model = target_model.model.language_model
        rotary_emb = lang_model.rotary_emb
        self._rotary_emb = rotary_emb
        positions = torch.arange(max_pos, device=self.device)
        position_ids = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()
        dummy_x = torch.zeros(1, max_pos, 1, dtype=MODEL_DTYPE, device=self.device)
        with torch.no_grad():
            cos, sin = rotary_emb(dummy_x, position_ids)
        self.rope_cos = cos.squeeze(0).contiguous()
        self.rope_sin = sin.squeeze(0).contiguous()

    def reset_cache(self):
        """Reset KV cache for new sequence."""
        self.cache_seqlens.zero_()

    def combine_hidden_states(self, aux_h2, aux_h14, aux_h25):
        """Combine 3 auxiliary hidden states into one via FC projection.

        aux_h* are each [hidden_size] or [1, hidden_size].
        Returns [hidden_size] combined hidden state.
        """
        # Use pre-allocated buffer to avoid torch.cat allocation
        buf = self._combine_buf
        hs = self.hidden_size
        buf[0, :hs] = aux_h2.view(-1)
        buf[0, hs:2*hs] = aux_h14.view(-1)
        buf[0, 2*hs:] = aux_h25.view(-1)
        return F.linear(buf, self.fc_weight).squeeze(0)  # [2048]

    def combine_hidden_states_batch(self, aux_h2, aux_h14, aux_h25):
        """Batch version: aux_h* are [seq_len, hidden_size]. Returns [seq_len, hidden_size]."""
        combined = torch.cat([aux_h2, aux_h14, aux_h25], dim=-1)  # [seq_len, 6144]
        return F.linear(combined, self.fc_weight)  # [seq_len, 2048]

    @torch.no_grad()
    def prefill(self, input_ids, combined_hidden_all, seq_len, mrope_position_ids=None):
        """Populate eagle3's KV cache with prefill context.

        Args:
            input_ids: [seq_len] token IDs (shifted: eagle3_ids[i] = target_ids[i+1])
            combined_hidden_all: [seq_len, 2048] per-position FC outputs
            seq_len: number of positions to process
            mrope_position_ids: [3, seq_len] MRoPE positions (temporal, height, width).
                                If None, uses sequential [0, 1, 2, ...] for all dims.
        """
        from flash_attn import flash_attn_func
        from triton_fused_kernels import batch_rmsnorm, batch_fused_swiglu

        self.reset_cache()

        input_embeds = self.embed_tokens(input_ids.unsqueeze(0))  # [1, seq_len, 2048]
        hidden_state = combined_hidden_all.unsqueeze(0)  # [1, seq_len, 2048]

        normed_embeds = batch_rmsnorm(input_embeds, self.input_layernorm_weight, self.eps)
        normed_hidden = batch_rmsnorm(hidden_state, self.hidden_norm_weight, self.eps)
        x = torch.cat([normed_embeds, normed_hidden], dim=-1)  # [1, seq_len, 4096]

        qkv = F.linear(x, self.qkv_weight)
        q = qkv[..., :self._q_dim].view(1, seq_len, self.num_heads, self.head_dim)
        k = qkv[..., self._q_dim:self._q_dim + self._kv_dim].view(1, seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[..., self._q_dim + self._kv_dim:].view(1, seq_len, self.num_kv_heads, self.head_dim)

        if mrope_position_ids is not None:
            # Compute cos/sin from actual MRoPE positions using the rotary_emb
            pos_3d = mrope_position_ids.unsqueeze(1).to(self.device)  # [3, 1, seq_len]
            dummy_x = torch.zeros(1, seq_len, 1, dtype=MODEL_DTYPE, device=self.device)
            cos, sin = self._rotary_emb(dummy_x, pos_3d)
            cos = cos.unsqueeze(2)  # [1, seq_len, 1, head_dim]
            sin = sin.unsqueeze(2)
            # Store the decode-start MRoPE position (last position + 1)
            self._decode_mrope_start = mrope_position_ids[:, -1].max().item() + 1
        else:
            positions = torch.arange(seq_len, device=self.device)
            cos = self.rope_cos[positions].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, 128]
            sin = self.rope_sin[positions].unsqueeze(0).unsqueeze(2)
            self._decode_mrope_start = seq_len

        q = self._apply_rope_batch(q, cos, sin)
        k = self._apply_rope_batch(k, cos, sin)

        attn_out = flash_attn_func(q, k, v, causal=True, softmax_scale=self.scaling)
        attn_out = attn_out.view(1, seq_len, -1)
        o_out = F.linear(attn_out, self.o_proj_weight)

        hs = hidden_state + o_out
        normed_mlp = batch_rmsnorm(hs, self.post_attn_norm_weight, self.eps)
        gate_up_out = F.linear(normed_mlp, self.gate_up_weight)
        mlp_out = F.linear(batch_fused_swiglu(gate_up_out, self.intermediate_size), self.down_proj_weight)
        hs = hs + mlp_out

        # Store K/V into cache (layer 0)
        self.k_cache[0, :seq_len] = k[0]
        self.v_cache[0, :seq_len] = v[0]

        # Extra layers
        q_dim_el = self.num_heads * self.head_dim
        kv_dim_el = self.num_kv_heads * self.head_dim
        for li, lw in enumerate(self._extra_layer_weights):
            normed_el = batch_rmsnorm(hs, lw['input_layernorm_weight'], self.eps)
            qkv_el = F.linear(normed_el, lw['qkv_weight'])
            q_el = qkv_el[..., :q_dim_el].view(1, seq_len, self.num_heads, self.head_dim)
            k_el = qkv_el[..., q_dim_el:q_dim_el + kv_dim_el].view(1, seq_len, self.num_kv_heads, self.head_dim)
            v_el = qkv_el[..., q_dim_el + kv_dim_el:].view(1, seq_len, self.num_kv_heads, self.head_dim)
            q_el = self._apply_rope_batch(q_el, cos, sin)
            k_el = self._apply_rope_batch(k_el, cos, sin)
            attn_el = flash_attn_func(q_el, k_el, v_el, causal=True, softmax_scale=self.scaling)
            attn_el = F.linear(attn_el.view(1, seq_len, -1), lw['o_proj_weight'])
            hs = hs + attn_el
            normed_mlp_el = batch_rmsnorm(hs, lw['post_attn_norm_weight'], self.eps)
            gu_el = F.linear(normed_mlp_el, lw['gate_up_weight'])
            mlp_el = F.linear(batch_fused_swiglu(gu_el, self.intermediate_size), lw['down_proj_weight'])
            hs = hs + mlp_el
            self.extra_k_caches[li][0, :seq_len] = k_el[0]
            self.extra_v_caches[li][0, :seq_len] = v_el[0]

        self.cache_seqlens[0] = seq_len
        self._prefill_len = seq_len

    @torch.no_grad()
    def cache_update(self, token_ids, combined_hiddens, start_pos):
        """Update eagle3 KV cache with real token data (fixing stale mask entries).

        Processes tokens at positions start_pos..start_pos+n-1, writing correct KV.
        Uses flash_attn_with_kvcache to attend to existing prefix + causal among new tokens.

        Args:
            token_ids: [n] token IDs
            combined_hiddens: [n, 2048] FC-projected hidden states
            start_pos: starting cache position
        """
        n = token_ids.shape[0]
        if n == 0:
            return

        input_embeds = self.embed_tokens(token_ids.unsqueeze(0))  # [1, n, 2048]
        hidden_state = combined_hiddens.unsqueeze(0)  # [1, n, 2048]

        normed_embeds = batch_rmsnorm(input_embeds, self.input_layernorm_weight, self.eps)
        normed_hidden = batch_rmsnorm(hidden_state, self.hidden_norm_weight, self.eps)
        x = torch.cat([normed_embeds, normed_hidden], dim=-1)  # [1, n, 4096]

        qkv = F.linear(x, self.qkv_weight)
        q = qkv[..., :self._q_dim].view(1, n, self.num_heads, self.head_dim)
        k = qkv[..., self._q_dim:self._q_dim + self._kv_dim].view(1, n, self.num_kv_heads, self.head_dim)
        v = qkv[..., self._q_dim + self._kv_dim:].view(1, n, self.num_kv_heads, self.head_dim)

        rope_start = self._get_rope_position(start_pos)
        positions = torch.arange(rope_start, rope_start + n, device=self.device)
        cos = self.rope_cos[positions].unsqueeze(0).unsqueeze(2)
        sin = self.rope_sin[positions].unsqueeze(0).unsqueeze(2)

        q = self._apply_rope_batch(q, cos, sin)
        k = self._apply_rope_batch(k, cos, sin)

        # Set cache_seqlens to start_pos so flash_attn uses prefix [0..start_pos-1]
        self.cache_seqlens[0] = start_pos
        attn_out = flash_attn_with_kvcache(
            q, self.k_cache, self.v_cache,
            k=k, v=v,
            cache_seqlens=self.cache_seqlens,
            causal=True, softmax_scale=self.scaling,
        )
        attn_out = F.linear(attn_out.reshape(1, n, -1), self.o_proj_weight)

        hs = hidden_state + attn_out
        normed_mlp = batch_rmsnorm(hs, self.post_attn_norm_weight, self.eps)
        gate_up_out = F.linear(normed_mlp, self.gate_up_weight)
        mlp_out = F.linear(batch_fused_swiglu(gate_up_out, self.intermediate_size), self.down_proj_weight)
        hs = hs + mlp_out

        q_dim_el = self.num_heads * self.head_dim
        kv_dim_el = self.num_kv_heads * self.head_dim
        for li, lw in enumerate(self._extra_layer_weights):
            normed_el = batch_rmsnorm(hs, lw['input_layernorm_weight'], self.eps)
            qkv_el = F.linear(normed_el, lw['qkv_weight'])
            q_el = qkv_el[..., :q_dim_el].view(1, n, self.num_heads, self.head_dim)
            k_el = qkv_el[..., q_dim_el:q_dim_el + kv_dim_el].view(1, n, self.num_kv_heads, self.head_dim)
            v_el = qkv_el[..., q_dim_el + kv_dim_el:].view(1, n, self.num_kv_heads, self.head_dim)
            q_el = self._apply_rope_batch(q_el, cos, sin)
            k_el = self._apply_rope_batch(k_el, cos, sin)
            attn_el = flash_attn_with_kvcache(
                q_el, self.extra_k_caches[li], self.extra_v_caches[li],
                k=k_el, v=v_el,
                cache_seqlens=self.cache_seqlens,
                causal=True, softmax_scale=self.scaling,
            )
            attn_el = F.linear(attn_el.reshape(1, n, -1), lw['o_proj_weight'])
            hs = hs + attn_el
            normed_mlp_el = batch_rmsnorm(hs, lw['post_attn_norm_weight'], self.eps)
            gu_el = F.linear(normed_mlp_el, lw['gate_up_weight'])
            mlp_el = F.linear(batch_fused_swiglu(gu_el, self.intermediate_size), lw['down_proj_weight'])
            hs = hs + mlp_el

        self.cache_seqlens[0] = start_pos + n

    def _apply_rope_batch(self, x, cos, sin):
        """Apply rotary embeddings to multi-token input. x: [B, S, H, D]."""
        half_dim = self.head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        cos_half = cos[..., :half_dim]
        sin_half = sin[..., :half_dim]
        out1 = x1 * cos_half - x2 * sin_half
        out2 = x2 * cos_half + x1 * sin_half
        return torch.cat([out1, out2], dim=-1).to(x.dtype)

    def forward_layer(self, input_embeds, hidden_state, position):
        """Forward through the single eagle3 transformer layer.

        Args:
            input_embeds: [1, 1, 2048] token embeddings (already embedded)
            hidden_state: [1, 1, 2048] combined hidden state from target
            position: scalar position for RoPE

        Returns:
            hidden_states: [1, 1, 2048] output hidden states
        """
        # Normalize embeds and hidden state
        normed_embeds = triton_rmsnorm(input_embeds, self.input_layernorm_weight, self.eps)
        normed_hidden = triton_rmsnorm(hidden_state, self.hidden_norm_weight, self.eps)

        # Concatenate: [1, 1, 4096]
        x = torch.cat([normed_embeds, normed_hidden], dim=-1)

        # QKV projection (single GEMV)
        qkv = F.linear(x, self.qkv_weight)
        q = qkv[..., :self._q_dim].view(1, 1, self.num_heads, self.head_dim)
        k = qkv[..., self._q_dim:self._q_dim + self._kv_dim].view(1, 1, self.num_kv_heads, self.head_dim)
        v = qkv[..., self._q_dim + self._kv_dim:].view(1, 1, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        cos = self.rope_cos[position].unsqueeze(0).unsqueeze(0)  # [1, 1, head_dim]
        sin = self.rope_sin[position].unsqueeze(0).unsqueeze(0)

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Attention with KV cache
        attn_out = flash_attn_with_kvcache(
            q, self.k_cache, self.v_cache,
            k=k, v=v,
            cache_seqlens=self.cache_seqlens,
            causal=True,
            softmax_scale=self.scaling,
        )

        # Output projection
        attn_out = attn_out.view(1, 1, -1)  # [1, 1, 2048]
        attn_out = F.linear(attn_out, self.o_proj_weight)

        # Fused residual + post-attention norm + MLP
        normed_for_mlp, hidden_states = fused_residual_rmsnorm(hidden_state, attn_out, self.post_attn_norm_weight, self.eps)
        gate_up_out = F.linear(normed_for_mlp, self.gate_up_weight)
        mlp_out = F.linear(fused_swiglu(gate_up_out, self.intermediate_size), self.down_proj_weight)

        # Extra layers (use same cache_seqlens — not yet incremented)
        for li, lw in enumerate(self._extra_layer_weights):
            normed_el, hidden_states = fused_residual_rmsnorm(hidden_states, mlp_out, lw['input_layernorm_weight'], self.eps)
            qkv_el = F.linear(normed_el, lw['qkv_weight'])
            q_el = qkv_el[..., :self._q_dim].view(1, 1, self.num_heads, self.head_dim)
            k_el = qkv_el[..., self._q_dim:self._q_dim + self._kv_dim].view(1, 1, self.num_kv_heads, self.head_dim)
            v_el = qkv_el[..., self._q_dim + self._kv_dim:].view(1, 1, self.num_kv_heads, self.head_dim)
            q_el = self._apply_rope(q_el, cos, sin)
            k_el = self._apply_rope(k_el, cos, sin)
            attn_el = flash_attn_with_kvcache(
                q_el, self.extra_k_caches[li], self.extra_v_caches[li],
                k=k_el, v=v_el,
                cache_seqlens=self.cache_seqlens,
                causal=True, softmax_scale=self.scaling,
            )
            attn_el = F.linear(attn_el.view(1, 1, -1), lw['o_proj_weight'])
            normed_mlp_el, hidden_states = fused_residual_rmsnorm(hidden_states, attn_el, lw['post_attn_norm_weight'], self.eps)
            gu_el = F.linear(normed_mlp_el, lw['gate_up_weight'])
            mlp_out = F.linear(fused_swiglu(gu_el, self.intermediate_size), lw['down_proj_weight'])

        hidden_states = hidden_states + mlp_out

        # Update cache length AFTER all layers have processed
        self.cache_seqlens += 1

        return hidden_states

    def _apply_rope(self, x, cos, sin):
        """Apply rotary embeddings. x: [B, S, H, D]."""
        half_dim = self.head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        cos_half = cos[..., :half_dim]
        sin_half = sin[..., :half_dim]
        # Standard rotate_half RoPE
        out1 = x1 * cos_half - x2 * sin_half
        out2 = x2 * cos_half + x1 * sin_half
        return torch.cat([out1, out2], dim=-1).to(x.dtype)

    def setup_draft_buffers(self, draft_k=3, capture_standalone_graph=False):
        """Initialize buffers and warmup for k-step draft generation.

        Args:
            capture_standalone_graph: If True, also capture a standalone CUDA graph
                for _draft_fn (used by eagle3.draft() in the non-fused path).
                If False (default), only init buffers + warmup — _draft_fn will be
                captured as part of the fused draft+verify graph instead.
        """
        self._embed_weight = self.embed_tokens.weight
        self._draft_k = draft_k

        # Static buffers for graph inputs
        self._draft_static_token = torch.zeros(1, dtype=torch.long, device=self.device)
        self._draft_static_hidden = torch.zeros(1, 1, self.hidden_size, dtype=MODEL_DTYPE, device=self.device)
        self._draft_static_cos = torch.zeros(draft_k, 1, 1, self.head_dim, dtype=MODEL_DTYPE, device=self.device)
        self._draft_static_sin = torch.zeros(draft_k, 1, 1, self.head_dim, dtype=MODEL_DTYPE, device=self.device)
        # Static outputs
        self._draft_static_outs = torch.zeros(draft_k, dtype=torch.long, device=self.device)
        # Confidence output (max logit per step) for adaptive gamma predictor
        self._draft_static_confidence = torch.zeros(draft_k, dtype=torch.float32, device=self.device)

        # Warmup (3 iterations to compile all kernels)
        saved_seqlens = self.cache_seqlens.clone()
        for pos in range(250, 253):
            self._draft_static_cos[:, 0, 0].copy_(self.rope_cos[pos:pos + draft_k])
            self._draft_static_sin[:, 0, 0].copy_(self.rope_sin[pos:pos + draft_k])
            self.cache_seqlens.copy_(saved_seqlens)
            self._draft_fn()
        self.cache_seqlens.copy_(saved_seqlens)
        torch.cuda.synchronize()

        if capture_standalone_graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            self._draft_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._draft_graph, stream=stream):
                self._draft_fn()
            torch.cuda.current_stream().wait_stream(stream)
            self.cache_seqlens.copy_(saved_seqlens)
            torch.cuda.synchronize()
            print(f"[Eagle3] Draft CUDA graph captured (k={draft_k})")
        else:
            print(f"[Eagle3] Draft buffers initialized (k={draft_k})")

        self._draft_graph_ready = True

    def setup_parallel_draft_buffers(self, draft_k=4):
        """Initialize buffers for parallel P-Eagle draft (one forward pass for k tokens)."""
        self._embed_weight = self.embed_tokens.weight
        self._par_draft_k = draft_k

        # Static buffers
        self._par_static_token = torch.zeros(1, dtype=torch.long, device=self.device)
        self._par_static_hidden = torch.zeros(1, self.hidden_size, dtype=MODEL_DTYPE, device=self.device)
        self._par_static_cos = torch.zeros(1, draft_k, 1, self.head_dim, dtype=MODEL_DTYPE, device=self.device)
        self._par_static_sin = torch.zeros(1, draft_k, 1, self.head_dim, dtype=MODEL_DTYPE, device=self.device)
        self._par_static_outs = torch.zeros(draft_k, dtype=torch.long, device=self.device)

        # Pre-allocated embed/hidden buffers with mask constants pre-filled
        self._par_embeds_buf = torch.zeros(1, draft_k, self.hidden_size, dtype=MODEL_DTYPE, device=self.device)
        self._par_hidden_buf = torch.zeros(1, draft_k, self.hidden_size, dtype=MODEL_DTYPE, device=self.device)
        # Fill mask positions (1..k-1) with precomputed constants
        for i in range(1, draft_k):
            self._par_embeds_buf[0, i].copy_(self.mask_embed)
            self._par_hidden_buf[0, i].copy_(self.mask_hidden_projected)

        # Warmup only (no standalone graph capture — _parallel_draft_fn will be
        # captured as part of the fused draft+verify graph instead, avoiding
        # CUDA graph memory pool conflicts from double-capturing the same function)
        saved_seqlens = self.cache_seqlens.clone()
        print(f"[Eagle3] Warmup cache_seqlens={saved_seqlens.item()}, mask_hidden_projected norm={self.mask_hidden_projected.norm():.2f}")
        for pos in range(250, 253):
            self._par_static_cos[0, :, 0].copy_(self.rope_cos[pos:pos + draft_k])
            self._par_static_sin[0, :, 0].copy_(self.rope_sin[pos:pos + draft_k])
            self.cache_seqlens.copy_(saved_seqlens)
            self._parallel_draft_fn()
        self.cache_seqlens.copy_(saved_seqlens)
        torch.cuda.synchronize()
        print(f"[Eagle3] Parallel draft buffers initialized (k={draft_k})")

    def _parallel_draft_fn(self):
        """One batched forward pass for k draft tokens (P-Eagle parallel).

        Single causal attention matching COD training mask:
        pos_i sees prefix + pos_0..pos_{i-1} (causal among new tokens).
        flash_attn_with_kvcache(causal=True) with all k tokens gives exactly this.
        """
        k = self._par_draft_k

        # Position 0: real token + real hidden (dynamic per call)
        self._par_embeds_buf[0, 0] = F.embedding(self._par_static_token, self._embed_weight).squeeze()
        self._par_hidden_buf[0, 0].copy_(self._par_static_hidden[0])

        # Batch norms + concat → [1, k, 4096]
        normed_e = batch_rmsnorm(self._par_embeds_buf, self.input_layernorm_weight, self.eps)
        normed_h = batch_rmsnorm(self._par_hidden_buf, self.hidden_norm_weight, self.eps)
        x = torch.cat([normed_e, normed_h], dim=-1)

        # QKV for all k positions (fused single GEMM)
        qkv = F.linear(x, self.qkv_weight)
        q = qkv[..., :self._q_dim].view(1, k, self.num_heads, self.head_dim)
        k_new = qkv[..., self._q_dim:self._q_dim + self._kv_dim].view(1, k, self.num_kv_heads, self.head_dim)
        v_new = qkv[..., self._q_dim + self._kv_dim:].view(1, k, self.num_kv_heads, self.head_dim)

        # RoPE
        q = self._apply_rope_batch(q, self._par_static_cos, self._par_static_sin)
        k_new = self._apply_rope_batch(k_new, self._par_static_cos, self._par_static_sin)

        # Single causal attention: pos_i sees prefix + pos_0..pos_{i-1}
        attn_out = flash_attn_with_kvcache(
            q, self.k_cache, self.v_cache,
            k=k_new, v=v_new,
            cache_seqlens=self.cache_seqlens,
            causal=True, softmax_scale=self.scaling,
        )

        # Residual + MLP (fused residual+norm)
        attn_out = F.linear(attn_out.reshape(1, k, -1), self.o_proj_weight)
        normed_mlp, hs = batch_fused_residual_rmsnorm(self._par_hidden_buf, attn_out, self.post_attn_norm_weight, self.eps)
        gate_up_out = F.linear(normed_mlp, self.gate_up_weight)
        mlp_out = F.linear(batch_fused_swiglu(gate_up_out, self.intermediate_size), self.down_proj_weight)

        # Extra layers (same single causal call per layer)
        for li, lw in enumerate(self._extra_layer_weights):
            normed_el, hs = batch_fused_residual_rmsnorm(hs, mlp_out, lw['input_layernorm_weight'], self.eps)
            qkv_el = F.linear(normed_el, lw['qkv_weight'])
            q_el = qkv_el[..., :self._q_dim].view(1, k, self.num_heads, self.head_dim)
            k_el = qkv_el[..., self._q_dim:self._q_dim + self._kv_dim].view(1, k, self.num_kv_heads, self.head_dim)
            v_el = qkv_el[..., self._q_dim + self._kv_dim:].view(1, k, self.num_kv_heads, self.head_dim)
            q_el = self._apply_rope_batch(q_el, self._par_static_cos, self._par_static_sin)
            k_el = self._apply_rope_batch(k_el, self._par_static_cos, self._par_static_sin)
            attn_el = flash_attn_with_kvcache(
                q_el, self.extra_k_caches[li], self.extra_v_caches[li],
                k=k_el, v=v_el,
                cache_seqlens=self.cache_seqlens,
                causal=True, softmax_scale=self.scaling,
            )
            attn_el = F.linear(attn_el.reshape(1, k, -1), lw['o_proj_weight'])
            normed_mlp_el, hs = batch_fused_residual_rmsnorm(hs, attn_el, lw['post_attn_norm_weight'], self.eps)
            gu_el = F.linear(normed_mlp_el, lw['gate_up_weight'])
            mlp_out = F.linear(batch_fused_swiglu(gu_el, self.intermediate_size), lw['down_proj_weight'])

        # Advance cache: flash_attn_with_kvcache does NOT update cache_seqlens
        self.cache_seqlens += k

        # Final norm + lm_head → argmax
        normed, hs = batch_fused_residual_rmsnorm(hs, mlp_out, self.final_norm_weight, self.eps)
        logits = F.linear(normed.squeeze(0), self.lm_head_weight)  # [k, vocab]
        draft_ids = logits.argmax(dim=-1)  # [k]
        self._par_static_outs.copy_(self.draft_to_target[draft_ids])

    def get_parallel_graph_context(self):
        """Return state needed to integrate parallel draft into fused CUDA graph."""
        return {
            'parallel_draft_fn': self._parallel_draft_fn,
            'static_token': self._par_static_token,
            'static_hidden': self._par_static_hidden,
            'static_cos': self._par_static_cos,
            'static_sin': self._par_static_sin,
            'static_outs': self._par_static_outs,
            'cache_seqlens': self.cache_seqlens,
            'rope_cos': self.rope_cos,
            'rope_sin': self.rope_sin,
            'draft_k': self._par_draft_k,
            'fc_weight': self.fc_weight,
            'combine_fn': self.combine_hidden_states,
        }

    def _draft_one_step(self, token_id, hidden_state, cos, sin, out_token, out_confidence):
        """One draft step: embed token, run 4 layers, greedy decode.

        Writes target_id to out_token and max_logit to out_confidence in-place.
        Returns hidden_out for the next step.
        """
        x = fused_embed_dual_rmsnorm_cat(token_id, self._embed_weight, hidden_state,
                                         self.input_layernorm_weight, self.hidden_norm_weight, self.eps)
        qkv = F.linear(x, self.qkv_weight)
        q = qkv[..., :self._q_dim].view(1, 1, self.num_heads, self.head_dim)
        k = qkv[..., self._q_dim:self._q_dim + self._kv_dim].view(1, 1, self.num_kv_heads, self.head_dim)
        v = qkv[..., self._q_dim + self._kv_dim:].view(1, 1, self.num_kv_heads, self.head_dim)
        fused_qk_rope(q, k, cos, sin, self.num_heads, self.num_kv_heads)
        attn_out = flash_attn_with_kvcache(
            q, self.k_cache, self.v_cache, k=k, v=v,
            cache_seqlens=self.cache_seqlens, causal=True, softmax_scale=self.scaling,
        )
        attn_out = F.linear(attn_out.view(1, 1, -1), self.o_proj_weight)
        normed_mlp, hidden_out = fused_residual_rmsnorm(hidden_state, attn_out, self.post_attn_norm_weight, self.eps)
        gate_up_out = F.linear(normed_mlp, self.gate_up_weight)
        mlp_out = F.linear(fused_swiglu(gate_up_out, self.intermediate_size), self.down_proj_weight)

        for li, lw in enumerate(self._extra_layer_weights):
            normed_el, hidden_out = fused_residual_rmsnorm(hidden_out, mlp_out, lw['input_layernorm_weight'], self.eps)
            qkv_el = F.linear(normed_el, lw['qkv_weight'])
            q_el = qkv_el[..., :self._q_dim].view(1, 1, self.num_heads, self.head_dim)
            k_el = qkv_el[..., self._q_dim:self._q_dim + self._kv_dim].view(1, 1, self.num_kv_heads, self.head_dim)
            v_el = qkv_el[..., self._q_dim + self._kv_dim:].view(1, 1, self.num_kv_heads, self.head_dim)
            fused_qk_rope(q_el, k_el, cos, sin, self.num_heads, self.num_kv_heads)
            attn_el = flash_attn_with_kvcache(
                q_el, self.extra_k_caches[li], self.extra_v_caches[li],
                k=k_el, v=v_el,
                cache_seqlens=self.cache_seqlens, causal=True, softmax_scale=self.scaling,
            )
            attn_el = F.linear(attn_el.view(1, 1, -1), lw['o_proj_weight'])
            normed_mlp_el, hidden_out = fused_residual_rmsnorm(hidden_out, attn_el, lw['post_attn_norm_weight'], self.eps)
            gu_el = F.linear(normed_mlp_el, lw['gate_up_weight'])
            mlp_out = F.linear(fused_swiglu(gu_el, self.intermediate_size), lw['down_proj_weight'])

        self.cache_seqlens += 1

        normed, hidden_out = fused_residual_rmsnorm(hidden_out, mlp_out, self.final_norm_weight, self.eps)
        logits = F.linear(normed.view(1, -1), self.lm_head_weight)
        fused_greedy_output(logits, self.draft_to_target, out_token, out_confidence)
        return hidden_out

    def _draft_fn(self):
        """Draft k tokens using static buffers."""
        token_id = self._draft_static_token
        hidden_state = self._draft_static_hidden
        for step in range(self._draft_k):
            hidden_state = self._draft_one_step(
                token_id, hidden_state,
                self._draft_static_cos[step], self._draft_static_sin[step],
                self._draft_static_outs[step], self._draft_static_confidence[step],
            )
            token_id = self._draft_static_outs[step]

    def get_fused_graph_context(self):
        """Return all state needed to capture the draft in a fused CUDA graph."""
        return {
            'draft_fn': self._draft_fn,
            'static_token': self._draft_static_token,
            'static_hidden': self._draft_static_hidden,
            'static_cos': self._draft_static_cos,
            'static_sin': self._draft_static_sin,
            'static_outs': self._draft_static_outs,
            'cache_seqlens': self.cache_seqlens,
            'rope_cos': self.rope_cos,
            'rope_sin': self.rope_sin,
            'draft_k': self._draft_k,
            'fc_weight': self.fc_weight,
            'combine_fn': self.combine_hidden_states,
        }

    def _get_rope_position(self, eagle3_cache_pos):
        """Map eagle3 cache position to correct MRoPE table index for decode."""
        if hasattr(self, '_decode_mrope_start') and hasattr(self, '_prefill_len'):
            return self._decode_mrope_start + (eagle3_cache_pos - self._prefill_len)
        return eagle3_cache_pos

    @torch.no_grad()
    def parallel_draft(self, target_token_id, combined_hidden, k=4, start_position=0):
        """P-Eagle parallel draft: generate k tokens in one forward pass."""
        if not hasattr(self, '_par_draft_k') or self._par_draft_k != k:
            self.setup_parallel_draft_buffers(draft_k=k)
        self._par_static_token[0] = target_token_id
        self._par_static_hidden[0].copy_(combined_hidden.view(-1))
        rope_pos = self._get_rope_position(start_position)
        self._par_static_cos[0, :, 0].copy_(self.rope_cos[rope_pos:rope_pos + k])
        self._par_static_sin[0, :, 0].copy_(self.rope_sin[rope_pos:rope_pos + k])
        self._parallel_draft_fn()
        return [self._par_static_outs[i] for i in range(k)]

    @torch.no_grad()
    def draft(self, target_token_id, combined_hidden, k=3, start_position=0):
        """Generate k draft tokens using CUDA graph if available."""
        rope_pos = self._get_rope_position(start_position)
        if getattr(self, '_draft_graph', None) is not None and k == self._draft_k:
            self._draft_static_token[0] = target_token_id
            self._draft_static_hidden[0, 0].copy_(combined_hidden.view(-1))
            self._draft_static_cos[:, 0, 0].copy_(self.rope_cos[rope_pos:rope_pos + k])
            self._draft_static_sin[:, 0, 0].copy_(self.rope_sin[rope_pos:rope_pos + k])
            self._draft_graph.replay()
            return [self._draft_static_outs[i] for i in range(k)]

        # Eager fallback
        draft_tokens = []
        input_embeds = self.embed_tokens(target_token_id.view(1, 1))
        hidden_state = combined_hidden.view(1, 1, -1)

        hidden_out = self.forward_layer(input_embeds, hidden_state, position=rope_pos)

        normed = triton_rmsnorm(hidden_out, self.final_norm_weight, self.eps)
        logits = F.linear(normed.view(1, -1), self.lm_head_weight)
        draft_id = logits.argmax(dim=-1).squeeze()
        target_id = self.draft_to_target[draft_id]
        draft_tokens.append(target_id)

        for step in range(1, k):
            input_embeds = self.embed_tokens(target_id.view(1, 1))
            hidden_state = hidden_out

            hidden_out = self.forward_layer(input_embeds, hidden_state, position=rope_pos + step)

            normed = triton_rmsnorm(hidden_out, self.final_norm_weight, self.eps)
            logits = F.linear(normed.view(1, -1), self.lm_head_weight)
            draft_id = logits.argmax(dim=-1).squeeze()
            target_id = self.draft_to_target[draft_id]
            draft_tokens.append(target_id)

        return draft_tokens

