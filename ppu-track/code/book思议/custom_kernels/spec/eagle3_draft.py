"""EAGLE-3 draft model for speculative decoding integration.

Loads the trained EAGLE-3 checkpoint and provides a lightweight interface
for the decode loop.  The heavy architecture (EagleDraftModel) is only
needed during training — at inference we only need:
  1. fc(6144→2048)          — project 3-layer hidden concat
  2. midlayer (Transformer)  — single decoder layer
  3. norm + lm_head           — project to reduced vocab

This module avoids importing the full training codebase.
"""

import os
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_cache = {}  # device -> Eagle3Draft


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal architecture (inference only)
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Eagle3Draft(nn.Module):
    """Lightweight EAGLE-3 draft model for inference.

    Architecture mirrors train/train_eagle.py EagleDraftModel but without
    the training-only TTT loop.
    """
    def __init__(self, hidden_size=2048, num_heads=16, num_kv_heads=8,
                 head_dim=128, intermediate_size=6144, draft_vocab=32000,
                 full_vocab=151936, rope_theta=1_000_000.0, norm_eps=1e-6,
                 ttt_length=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.ttt_length = ttt_length
        self.draft_vocab = draft_vocab

        # FC: 3-layer concat → hidden_size
        self.fc = nn.Linear(hidden_size * 3, hidden_size, bias=False)

        # Single decoder layer components
        self.hidden_norm = RMSNorm(hidden_size, norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, norm_eps)

        # Attention (GQA) — input is cat(embed_normed, hidden_normed) = 2*hidden_size
        self.q_proj = nn.Linear(hidden_size * 2, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size * 2, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size * 2, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # MLP
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Output
        self.norm = RMSNorm(hidden_size, norm_eps)
        self.lm_head = nn.Linear(hidden_size, draft_vocab, bias=False)

        # Embedding (frozen, loaded from base model)
        self.embed_tokens = nn.Embedding(full_vocab, hidden_size)

        # RoPE
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Vocab mapping (loaded from checkpoint)
        self.register_buffer('d2t', torch.zeros(draft_vocab, dtype=torch.long))
        self.register_buffer('t2d', torch.zeros(full_vocab, dtype=torch.bool))

    def _attention(self, hidden, input_emb, position_offset=0, past_kv=None):
        """Single attention layer with KV cache for TTT multi-step prediction.

        CRITICAL: Training uses full-sequence causal attention (each position
        sees all previous).  Without KV cache, inference only sees itself
        (attn weight = 1.0 for seq_len=1), producing garbage predictions.
        The KV cache accumulates context across TTT steps, matching training.

        hidden:    [B, hs]
        input_emb: [B, hs]  (from embed_tokens)
        past_kv:   tuple(k_cache, v_cache) or None — kv_heads cached entries
        Returns:   (hidden [B, hs], new_kv)
        """
        # Normalize dimensions: both should be [B, seq, hs]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)       # [B, 1, hs]
        if input_emb.dim() == 2:
            input_emb = input_emb.unsqueeze(1)  # [B, 1, hs]

        B = hidden.shape[0]

        # Concatenate normalized inputs
        h_normed = self.hidden_norm(hidden)
        e_normed = self.input_layernorm(input_emb)
        combined = torch.cat([e_normed, h_normed], dim=-1)  # [B, 1, 2*hs]

        # QKV projections from combined
        q = self.q_proj(combined).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(combined).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(combined).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to new Q and K BEFORE cache concatenation
        t = torch.arange(position_offset, position_offset + 1, device=q.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [1, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        q = (q * cos) + (rotate(q) * sin)
        k = (k * cos) + (rotate(k) * sin)

        # Concatenate with past KV cache (RoPE already applied to cached entries)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        # GQA expand on full K,V
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim)

        # Attention: compute in fp32 to avoid overflow in Q@K^T.
        # Q/K values can be O(300) → 128-dim dot product ≈ 11M ≫ fp16 max (65k).
        q32 = q.float()
        k32 = k.float()
        v32 = v.float()
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q32, k32.transpose(-2, -1)) * scale
        # No causal mask needed — single query attends to all past + self
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v32).to(hidden.dtype)  # [B, heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)
        out = self.o_proj(out)

        # Residual
        hidden = hidden + out

        # MLP (fp16 is fine — SiLU + gate/up/down are elementwise/small matmul)
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
        hidden = residual + hidden

        hidden = hidden.squeeze(1)  # [B, hs]
        return hidden, new_kv

    @torch.no_grad()
    def draft_tokens(self, hidden_concat, cur_token_id, gamma, position_offset=0):
        """Predict gamma future tokens using TTT loop with KV cache.

        Args:
            hidden_concat: [1, hidden_size*3] — 3-layer hidden concat from target model
            cur_token_id: int — current token
            gamma: int — number of draft tokens to predict
            position_offset: int — NOT USED (kept for API compat).
                                 TTT loop uses positions 0..gamma-1 to match training.

        Returns:
            draft_tokens: list[int] of length gamma (full vocab IDs)
            draft_logits: [gamma, draft_vocab] fp16 — for acceptance check
        """
        hidden = self.fc(hidden_concat)  # [1, hs]
        input_emb = self.embed_tokens(torch.tensor([[cur_token_id]], device=hidden_concat.device))
        # input_emb: [1, hs]

        draft_tokens_list = []
        draft_logits = []
        past_kv = None  # Fresh KV cache for each draft_tokens call

        for step in range(min(gamma, self.ttt_length)):
            # Use position 0..gamma-1 to match training (positions start from 0)
            hidden, past_kv = self._attention(
                hidden, input_emb,
                position_offset=step,
                past_kv=past_kv,
            )
            logits = self.lm_head(self.norm(hidden))  # [1, draft_vocab]
            token = logits[0].argmax(dim=-1).item()
            draft_tokens_list.append(token)
            draft_logits.append(logits[0])

            if step < gamma - 1:
                # Shift: next step uses draft's own output
                input_emb = self.embed_tokens(torch.tensor([[token]], device=hidden_concat.device))

        # Map draft tokens to full vocab
        full_vocab_tokens = [self.d2t[t].item() for t in draft_tokens_list]

        return full_vocab_tokens, torch.stack(draft_logits)


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_eagle3_draft(device="cuda"):
    """Load the trained EAGLE-3 draft model.  Cached per device."""
    if device in _cache:
        return _cache[device]

    ckpt_path = os.environ.get(
        "AICAS_EAGLE3_CKPT",
        "eagle_pipeline/data/checkpoints/eagle_draft_best.pt"
    )
    logger.info(f"[EAGLE-3] Loading draft model from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = Eagle3Draft(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg.get("num_heads", 16),
        num_kv_heads=cfg.get("num_kv_heads", 8),
        head_dim=cfg.get("head_dim", 128),
        intermediate_size=cfg.get("intermediate_size", 6144),
        draft_vocab=cfg["draft_vocab_size"],
        full_vocab=cfg.get("vocab_size", 151936),
        ttt_length=cfg.get("ttt_length", 7),
        rope_theta=cfg.get("rope_theta", 10000.0),  # match training EagleConfig
    )

    # Remap checkpoint keys: training uses nested modules:
    #   midlayer.hidden_norm.weight
    #   midlayer.self_attn.q_proj.weight  →  q_proj.weight
    #   midlayer.mlp.gate_proj.weight     →  gate_proj.weight
    # Inference model has flat keys.
    raw_sd = ckpt["state_dict"]
    remapped = {}
    _prefix_map = {
        "midlayer.self_attn.": "",
        "midlayer.mlp.": "",
        "midlayer.": "",
    }
    for k, v in raw_sd.items():
        if k in ("d2t", "t2d"):
            continue  # loaded separately as buffers
        new_key = k
        for old_pfx, new_pfx in _prefix_map.items():
            if new_key.startswith(old_pfx):
                new_key = new_pfx + new_key[len(old_pfx):]
                break
        remapped[new_key] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.debug(f"[EAGLE-3] Missing keys (expected for frozen params): {missing[:5]}...")
    if unexpected:
        logger.warning(f"[EAGLE-3] Unexpected keys in checkpoint: {unexpected[:5]}...")

    # Load vocab mapping: may be at checkpoint top-level OR in state_dict
    d2t = ckpt["d2t"] if "d2t" in ckpt else raw_sd.get("d2t")
    t2d = ckpt["t2d"] if "t2d" in ckpt else raw_sd.get("t2d")
    if d2t is not None:
        model.d2t.copy_(d2t)
    if t2d is not None:
        model.t2d.copy_(t2d)
    model.eval().to(device, dtype=torch.float16)
    # Buffers (d2t, t2d, inv_freq) keep their original dtypes
    model.d2t = model.d2t.long()
    model.t2d = model.t2d.bool()
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[EAGLE-3] Draft model loaded: {n_params:,} params, "
                f"ttt_length={model.ttt_length}, draft_vocab={model.draft_vocab}")

    _cache[device] = model
    return model
