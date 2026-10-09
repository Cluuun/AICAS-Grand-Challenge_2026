"""HuggingFace EAGLE3 draft model adapter for Qwen3-VL-2B-Instruct-Eagle3.

Loads the HuggingFace model from ``taobao-mnn/Qwen3-VL-2B-Instruct-Eagle3``
and wraps it with the same interface as the custom-trained ``Eagle3DraftModel``
from ``eagle3.py``, so it can be plugged into the existing spec-decode pipeline.

The HF model uses ``LlamaForCausalLMEagle3`` (Llama-based) architecture with
standard 1D RoPE, while the current custom model uses Qwen3VL-based architecture
with mRoPE.  This adapter provides a Llama-style decoder layer and implements
the tree-search algorithm (``topK_genrate``) in pure PyTorch, avoiding the
CUDA-graph / custom-kernel optimisations that are tightly coupled to the
original ``Eagle3DraftModel``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO = "taobao-mnn/Qwen3-VL-2B-Instruct-Eagle3"
EAGLE3_DEBUG_VAR = "AICAS_EAGLE3_DEBUG"


def _debug(msg: str) -> None:
    if os.environ.get(EAGLE3_DEBUG_VAR, "0") == "1":
        print(f"[hf_eagle3] {msg}")


# ---------------------------------------------------------------------------
# RoPE helpers  (standard 1D rotary, no mRoPE)
# ---------------------------------------------------------------------------
def _precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(end, device=device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, dim/2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply standard 1D rotary embeddings.  x: (batch, heads, seq, head_dim)."""
    head_dim = x.shape[-1]
    if position_ids is not None:
        # cos/sin: (1, 1, max_seq, dim/2) → index to (1, 1, seq, dim/2)
        if position_ids.dim() == 2:
            position_ids = position_ids[0]  # (batch, seq) -> (seq,)
        cos = cos[:, :, position_ids, :]   # (1, 1, seq, dim/2)
        sin = sin[:, :, position_ids, :]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    # Expand cos/sin to match batch and heads
    batch, heads, seq, half_dim = x1.shape
    cos = cos.expand(batch, heads, seq, half_dim)
    sin = sin.expand(batch, heads, seq, half_dim)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Llama-style EAGLE3 attention  (QKV from hidden_size*2, standard RoPE)
# ---------------------------------------------------------------------------
class HfEagle3Attention(nn.Module):
    """Single-layer attention matching the HF LlamaForCausalLMEagle3 weights.

    Key difference from the custom ``Qwen3VLEagle3Attention``:
    - No Q/K RMS norms
    - Standard 1D RoPE (not mRoPE)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 131072)

        in_dim = config.hidden_size * 2  # because [input_emb | hidden_states] are concatenated
        self.q_proj = nn.Linear(in_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Pre-compute cos/sin caches (will be extended on demand)
        self.register_buffer("_cos_cached", None, persistent=False)
        self.register_buffer("_sin_cached", None, persistent=False)

    def _extend_rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        current = getattr(self, "_cos_cached", None)
        if current is not None and current.shape[-2] >= seq_len:
            return
        cos, sin = _precompute_freqs_cis(
            self.head_dim,
            max(seq_len, self.max_position_embeddings + 20),
            theta=self.rope_theta,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("_cos_cached", cos.to(dtype=dtype), persistent=False)
        self.register_buffer("_sin_cached", sin.to(dtype=dtype), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,        # (batch, seq, hidden_size*2)
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,   # (batch, seq)  or None
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
        self._extend_rope_cache(int(position_ids.max()) + 1, q.device, q.dtype)
        cos = self._cos_cached.to(device=q.device, dtype=q.dtype)
        sin = self._sin_cached.to(device=q.device, dtype=q.dtype)

        q = _apply_rotary_emb(q, cos, sin, position_ids)
        k = _apply_rotary_emb(k, cos, sin, position_ids)

        # KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v)

        # GQA: expand key/value heads
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled dot-product attention (handles causal mask internally)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,  # we handle mask via attention_mask
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present


# ---------------------------------------------------------------------------
# Llama-style EAGLE3 decoder layer
# ---------------------------------------------------------------------------
class HfEagle3MLP(nn.Module):
    """Standard Llama SiLU-gated MLP, matches HF weight naming midlayer.mlp.*."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HfEagle3DecoderLayer(nn.Module):
    """Single decoder layer that matches the HF LlamaForCausalLMEagle3 midlayer."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HfEagle3Attention(config)
        self.mlp = HfEagle3MLP(config)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,          # (batch, seq, hidden_size)
        hidden_states: torch.Tensor,      # (batch, seq, hidden_size) — projected feature
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states

        norm_hidden = self.hidden_norm(hidden_states)
        norm_input = self.input_layernorm(input_emb)
        # Expand hidden to match input sequence length (hidden has seq_len=1 during decode)
        if norm_hidden.shape[1] != norm_input.shape[1]:
            norm_hidden = norm_hidden.expand(-1, norm_input.shape[1], -1)
        attn_input = torch.cat([norm_input, norm_hidden], dim=-1)  # (batch, seq, 2*H)

        attn_out, present_kv = self.self_attn(
            attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Main adapter model
# ---------------------------------------------------------------------------
class HfEagle3DraftModel(nn.Module):
    """EAGLE3 draft model loaded from HuggingFace, compatible with existing pipeline.

    Implements the same external interface as ``Eagle3DraftModel``:
    - ``init_tree(total_tokens, depth, top_k)``
    - ``topK_genrate(feature, input_ids)``
    - ``reset_kv()``
    - ``project_feature(feature)``
    - ``draft_vocab_ids``  (vocab mapping buffer)
    - ``_stable_kv``       (KV cache)
    """

    def __init__(
        self,
        text_config,
        hf_config: LlamaConfig,
        *,
        lm_head_weight: torch.Tensor | None = None,
        embed_weight: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
    ):
        super().__init__()
        self.text_config = text_config
        self.hf_config = hf_config
        self.hidden_size = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.draft_vocab_size = int(hf_config.draft_vocab_size)

        # Feature projection: concat of 3 hidden states → hidden_size
        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)

        # Token embedding (copied from target model)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=text_config.pad_token_id)

        # Single decoder layer
        self.midlayer = HfEagle3DecoderLayer(hf_config)

        # Output norm + lm_head
        self.norm = Qwen3VLTextRMSNorm(self.hidden_size, eps=text_config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)

        # Vocab mapping: draft output index → target vocab ID.
        if d2t is not None:
            self.register_buffer("draft_vocab_ids", d2t.clone().detach().cpu().to(dtype=torch.long), persistent=True)
        else:
            self.register_buffer("draft_vocab_ids", torch.empty(0, dtype=torch.long), persistent=False)

        # Load external weights
        if embed_weight is not None:
            self.embed_tokens.weight = nn.Parameter(embed_weight.detach().clone().to(dtype=torch.bfloat16))
        if lm_head_weight is not None:
            self.lm_head.weight = nn.Parameter(lm_head_weight.detach().clone().to(dtype=torch.bfloat16))

        # Freeze embedding
        for p in self.embed_tokens.parameters():
            p.requires_grad_(False)

    # draft_vocab_ids is a persistent buffer registered in __init__

        # Tree search state
        self.total_tokens = 14
        self.depth = 3
        self.top_k = 4
        self.tree_mask = None
        self.tree_mask_init = None
        self._topk_cs_index = None
        self._zero_parent = None
        self.position_ids = None

        # KV cache
        self._stable_kv = None

    # -- Feature projection -------------------------------------------------

    def project_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Project 3-layer hidden-state concat to model hidden size."""
        if feature.dim() == 1:
            feature = feature.reshape(1, 1, -1)
        elif feature.dim() == 2:
            feature = feature.unsqueeze(1)
        if int(feature.shape[-1]) == self.hidden_size:
            return feature.to(dtype=self.fc.weight.dtype)
        feature = feature.to(dtype=self.fc.weight.dtype)
        feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        return self.fc(feature)

    # -- Tree initialisation -------------------------------------------------

    def init_tree(self, total_tokens: int = 15, depth: int = 3, top_k: int = 4) -> None:
        self.total_tokens = max(1, int(total_tokens) - 1)
        self.depth = max(1, int(depth))
        self.top_k = min(max(1, int(top_k)), int(self.draft_vocab_size))
        device = self.embed_tokens.weight.device
        self.tree_mask_init = torch.eye(self.top_k, device=device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=device, dtype=torch.long)
        self._topk_cs_index = torch.arange(self.top_k, device=device, dtype=torch.long)
        self._zero_parent = torch.zeros(1, dtype=torch.long, device=device)

    def reset_kv(self) -> None:
        self._stable_kv = None
        self.tree_mask = None

    # -- Forward pass --------------------------------------------------------

    def _embed_input_ids(self, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.embed_tokens(input_ids).to(dtype=dtype)

    def forward_block(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Single forward pass: embed tokens, add feature, run decoder layer."""
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        input_ids = input_ids.to(hidden_states.device)
        inputs_embeds = self._embed_input_ids(input_ids, hidden_states.dtype)

        batch, seq_len, _ = hidden_states.shape

        if position_ids is None:
            past_len = self._kv_seq_len(past_key_value)
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=hidden_states.device, dtype=torch.long,
            ).unsqueeze(0)

        # Build attention mask
        attn_mask = self._build_attention_mask(hidden_states, past_key_value, position_ids)

        hidden_states, present_kv = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        return hidden_states, present_kv

    def _kv_seq_len(self, past_key_value) -> int:
        if past_key_value is None:
            return 0
        try:
            # past_key_value may be ((k,v),) or just (k,v)
            pair = past_key_value
            if isinstance(pair, (tuple, list)) and len(pair) == 1 and isinstance(pair[0], (tuple, list)):
                pair = pair[0]
            if isinstance(pair, (tuple, list)) and len(pair) >= 2:
                return int(pair[0].shape[2])
            return int(past_key_value[0].shape[2])
        except Exception:
            return 0

    def _build_attention_mask(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Causal + tree mask for the draft model.

        Tree-attention logic (based on ``_eagle3_tree_attention_mask_`` in
        ``eagle3.py``):
          1. Everything starts as *hidden* (-inf).
          2. Prefix KV positions (before the first tree token) are made
             visible to **all** query tokens.
          3. Tree KV positions are visible **only** where the tree-mask
             allows it — the tree-mask encodes the ancestor relationships
             between draft tokens across all expansion steps.
          4. When there is **no** tree-mask (prefill / single-token decode)
             a standard causal mask is used.
        """
        batch, q_len, _ = hidden_states.shape
        past_len = self._kv_seq_len(past_key_value)
        kv_len = past_len + q_len

        if q_len <= 1:
            return None  # single-token decode needs no mask

        neg = torch.finfo(hidden_states.dtype).min
        mask = torch.full((1, 1, q_len, kv_len), neg, device=hidden_states.device, dtype=hidden_states.dtype)

        # ---- prefix KV (always visible to every query) ----
        if past_len > 0:
            mask[:, :, :, :past_len] = 0.0

        tree_mask = getattr(self, "tree_mask", None)
        if tree_mask is not None:
            # ---- tree-attention mode ----
            tree_mask = tree_mask.to(device=hidden_states.device)
            _, _, tree_q, tree_k = map(int, tree_mask.shape)
            if tree_q == q_len and tree_k <= kv_len:
                tree_start = kv_len - tree_k  # = prefix_len
                if tree_start >= 0:
                    mask[:, :, :, tree_start:tree_start + tree_k].masked_fill_(
                        tree_mask.to(dtype=torch.bool), 0.0
                    )
        else:
            # ---- standard causal mode (prefill) ----
            if q_len > 0 and past_len < kv_len:
                q_pos = torch.arange(past_len, past_len + q_len, device=hidden_states.device, dtype=torch.long)
                kv_pos = torch.arange(past_len, kv_len, device=hidden_states.device, dtype=torch.long)
                causal = kv_pos.view(1, 1, 1, q_len) <= q_pos.view(1, 1, q_len, 1)
                mask[:, :, :, past_len:].masked_fill_(causal, 0.0)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool | None = None,
    ):
        """Standard forward pass used by the tree search."""
        hidden_states = self.project_feature(hidden_states)
        input_ids = input_ids.to(hidden_states.device)
        past_key_value = past_key_values[0] if past_key_values is not None else None

        batch, seq_len = input_ids.shape
        past_len = self._kv_seq_len(past_key_value)

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=hidden_states.device, dtype=torch.long,
            ).unsqueeze(0)

        inputs_embeds = self._embed_input_ids(input_ids, hidden_states.dtype)
        attn_mask = self._build_attention_mask(hidden_states, past_key_value, position_ids)

        hidden_states, present_kv = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )

        if use_cache:
            return hidden_states, (present_kv,)
        return hidden_states

    # -- LM head top-k -------------------------------------------------------

    @torch.no_grad()
    def _lm_head_logp_topk(
        self,
        hidden: torch.Tensor,
        top_k: int,
        logits_processor: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k log probabilities from the last hidden state.

        Returns token ids in the **target** vocabulary space, already mapped
        via ``draft_vocab_ids`` (d2t).
        """
        if logits_processor is not None:
            raise RuntimeError("HfEagle3 does not support logits_processor")
        hidden = hidden.to(dtype=self.lm_head.weight.dtype)
        logits = self.lm_head(hidden)  # (batch, seq, draft_vocab_size)
        logits = logits.float()
        topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
        # Map from draft lm_head index space → target vocabulary space
        if self.draft_vocab_ids.numel() > 0:
            topk_idx = self.draft_vocab_ids.to(device=topk_idx.device).index_select(
                0, topk_idx.reshape(-1)
            ).reshape_as(topk_idx)
        return topk_idx, topk_vals

    # -- Tree generation -----------------------------------------------------

    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        logits_processor: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate draft tokens using EAGLE3 tree search.

        Returns:
            draft_tokens:      (1, total_tokens+1)
            retrieve_indices:  (leaf_count, max_depth)
            tree_mask:         (1, 1, total_tokens+1, total_tokens+1)
            tree_position_ids: (total_tokens+1,)
        """
        if self.tree_mask_init is None:
            self.init_tree()

        # Always reset KV cache at the start of each topK_genrate call.
        # The non-full-graph pipeline calls topK_genrate each round and does
        # not manage _stable_kv between rounds (the tree expansion injects
        # many extra KV positions that do not correspond to the input prefix).
        # Resetting is correct (full prefill each round) and avoids complex
        # inter-round cache management.
        self.reset_kv()

        input_ids = input_ids.to(hidden_states.device, dtype=torch.long).contiguous()
        total_tokens = int(self.total_tokens)
        depth = int(self.depth)
        top_k = int(self.top_k)

        sample_token = input_ids[:, -1]
        shifted_input = input_ids[:, 1:].contiguous()
        len_posi = int(shifted_input.shape[1])
        self.tree_mask = None

        # ---- Full prefill (no KV cache reuse) ----
        out_hidden, past_key_values = self(
            hidden_states,
            input_ids=shifted_input,
            use_cache=True,
        )
        # Top-k from last token
        last_hidden_for_topk = out_hidden[:, -1]
        topk_index, topk_p = self._lm_head_logp_topk(
            self.norm(last_hidden_for_topk),
            top_k,
            logits_processor,
        )

        last_hidden = out_hidden[:, -1]
        scores = topk_p[0]
        scores_list = [scores[None]]
        parents_list = [self._zero_parent]
        sampled_tokens = [topk_index]
        input_ids_step = topk_index

        # Feature for current step
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init

        # ---- Tree expansion ----
        for i in range(depth):
            self.tree_mask = tree_mask
            step_position_ids = len_posi + self.position_ids

            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids_step,
                past_key_values=past_key_values,
                position_ids=step_position_ids,
                use_cache=True,
            )

            # Top-k
            step_topk, step_topk_p = self._lm_head_logp_topk(
                self.norm(out_hidden),
                top_k,
                logits_processor,
            )
            # Squeeze batch dim for expand steps
            step_topk = step_topk.squeeze(0)      # (k, k)
            step_topk_p = step_topk_p.squeeze(0)  # (k, k)

            # Expand top-k: merge scores with cumulative
            cu_scores = step_topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids, :]  # (1, k, H)
            input_ids_step = step_topk.view(-1)[topk_cs_index][None]  # (1, k)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            len_posi += 1
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1

            scores = topk_cs_p
            parents = topk_cs_index + bias
            parents_list.append(parents)
            sampled_tokens.append(step_topk)
            scores_list.append(cu_scores)

        self._stable_kv = past_key_values

        # ---- Build tree metadata ----
        scores_all = torch.cat(scores_list, dim=0).view(-1)
        sampled_all = torch.cat(sampled_tokens, dim=0).view(-1)
        total_tokens = min(total_tokens, int(scores_all.numel()))
        top_scores_idx = torch.topk(scores_all, total_tokens, dim=-1).indices
        top_scores_idx = torch.sort(top_scores_idx).values

        draft_tokens = sampled_all[top_scores_idx]
        draft_tokens = torch.cat((sample_token.reshape(-1), draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_idx // top_k].long()
        mask_index = torch.searchsorted(top_scores_idx, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1

        retrieve_indices, tree_mask_out, tree_position_ids = self._tree_metadata(
            mask_index, total_tokens, logits_processor=logits_processor,
        )
        draft_tokens = draft_tokens[None]
        return draft_tokens, retrieve_indices, tree_mask_out, tree_position_ids

    # -- Tree metadata helpers -----------------------------------------------

    def _tree_metadata(
        self,
        mask_index: torch.Tensor,
        total_tokens: int,
        *,
        logits_processor: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_index_list = mask_index.tolist()

        tree_mask_cpu = torch.eye(total_tokens + 1).bool()
        tree_mask_cpu[:, 0] = True
        for i in range(total_tokens):
            tree_mask_cpu[i + 1].add_(tree_mask_cpu[mask_index_list[i]])
        tree_position_ids = torch.sum(tree_mask_cpu, dim=1) - 1

        tree_mask = tree_mask_cpu.float()[None, None].to(mask_index.device)

        max_depth = int(torch.max(tree_position_ids)) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        leaf_num = total_tokens - (len(noleaf_index) - 1)

        retrieve_indices = torch.zeros(leaf_num, max_depth, dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        pos_list = tree_position_ids.tolist()
        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                node_depth = pos_list[i]
                for j in reversed(range(node_depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5
            retrieve_indices = sorted(
                retrieve_indices,
                key=lambda row: [x if x >= 0 else maxitem for x in row],
            )

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=mask_index.device)
        tree_position_ids = tree_position_ids.to(mask_index.device)
        return retrieve_indices, tree_mask, tree_position_ids

    # -- KV cache compatibility (for eagle3_worker) --------------------------

    def _draft_step_graph_past_pair(self, past_key_values):
        """Extract (k, v) pair from KV cache, matching Eagle3DraftModel interface."""
        if past_key_values is None:
            return None
        pair = past_key_values
        if isinstance(pair, (tuple, list)) and len(pair) == 1 and isinstance(pair[0], (tuple, list)):
            pair = pair[0]
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            return None
        k, v = pair[0], pair[1]
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            return None
        return k, v


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_hf_eagle3_draft(
    model_name_or_path: str,
    target_model,
    draft_len: int,
    device: str | torch.device,
) -> HfEagle3DraftModel:
    """Load the HuggingFace EAGLE3 draft model and wrap it.

    Args:
        model_name_or_path: HF repo ID (default ``taobao-mnn/...``) or local path.
        target_model: The Qwen3VL target model (provides embed / lm_head weights).
        draft_len: Number of draft tokens (unused by HF, kept for compat).
        device: Target device.

    Returns:
        HfEagle3DraftModel instance, ready for ``init_tree()``.
    """
    text_config = target_model.config.text_config
    text_model = getattr(getattr(target_model, "model", None), "language_model", None)
    embed_weight = getattr(getattr(text_model, "embed_tokens", None), "weight", None)
    # Note: HF model has its own lm_head (32000 draft vocab).
    # We do NOT pass target_model.lm_head.weight here, because the HF
    # checkpoint contains its own trained lm_head.weight (32000, 2048).
    # The d2t buffer maps draft indices -> target vocab ids.

    # Download or use cached config
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=model_name_or_path)
    _debug(f"Model cached at: {local_dir}")

    import json
    config_path = os.path.join(local_dir, "config.json")
    with open(config_path) as f:
        hf_config_dict = json.load(f)
    hf_config = LlamaConfig.from_dict(hf_config_dict)

    # Download model weights
    safetensors_path = os.path.join(local_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        safetensors_path = hf_hub_download(model_name_or_path, "model.safetensors")

    _debug("Loading safetensors weights ...")
    state_dict = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            if k == "d2t":
                state_dict[k] = f.get_tensor(k).to(device="cpu", dtype=torch.long)
            elif k == "t2d":
                state_dict[k] = f.get_tensor(k).to(device="cpu", dtype=torch.bool)
            else:
                state_dict[k] = f.get_tensor(k).to(device=device, dtype=torch.bfloat16)

    d2t = state_dict.pop("d2t", None)
    t2d = state_dict.pop("t2d", None)  # not needed for inference

    # The d2t buffer in SpecForge is stored as an *offset*:
    #   d2t[i] = used_tokens[i] - i
    # where used_tokens[i] is the i-th most common target-vocab token.
    # To get the absolute target token id we must add the draft index back:
    #   target_id = d2t[i] + i
    # We convert to the absolute format expected by the pipeline
    # (draft_vocab_ids[i] = used_tokens[i]) at load time.
    if d2t is not None:
        indices = torch.arange(d2t.size(0), dtype=d2t.dtype, device=d2t.device)
        d2t = d2t + indices  # offset → absolute

    # Create model
    model = HfEagle3DraftModel(
        text_config,
        hf_config,
        lm_head_weight=None,  # HF checkpoint has its own trained lm_head
        embed_weight=embed_weight,
        d2t=d2t,  # now in absolute format
    )
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Load state dict.
    # embed_tokens.weight is not in the HF checkpoint (comes from target model),
    # and draft_vocab_ids was already set from d2t in __init__.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _debug(f"Loaded state dict: missing={missing} unexpected={unexpected}")
    if unexpected:
        _debug(f"Unexpected keys: {unexpected}")

    # Set layer indices matching SpecForge's training convention.
    # SpecForge's QwenVLOnlineEagle3Model._prepare_data uses:
    #   low_aux_layer  = 1 + offset   → 2
    #   mid_aux_layer  = n_layers//2 - 1 + offset → n_layers//2
    #   last_aux_layer = n_layers - 4 + offset    → n_layers - 3
    # where n_layers = num_hidden_layers and offset = 1 (embedding).
    n_layers = int(text_config.num_hidden_layers)
    model.layer_indices = (2, n_layers // 2, n_layers - 3)
    _debug(f"Set layer_indices={model.layer_indices}")

    return model
