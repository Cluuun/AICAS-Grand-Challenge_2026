"""EAGLE-3 draft model v2 — single-hidden input, no 3-layer concat.

Key change from v1:
  - Input is a SINGLE hidden state (not 3-layer concat)
  - fc layer: hidden_size → hidden_size (not 3*hidden_size)
  - This eliminates the need for output_hidden_states=True during verify,
    which was the #1 throughput bottleneck.

Checkpoint loading: auto-detects v1 (3-layer fc) vs v2 (1-layer fc) from
the fc.weight shape in the checkpoint.  v2 checkpoints use 'fc.weight'
of shape [2048, 2048].
"""

import os
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_cache = {}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Eagle3DraftV2(nn.Module):
    """Lightweight EAGLE-3 draft model v2 — single hidden input."""

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

        # FC: single hidden → hidden_size (NOT 3x concat)
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

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

        # Vocab mapping
        self.register_buffer('d2t', torch.zeros(draft_vocab, dtype=torch.long))
        self.register_buffer('t2d', torch.zeros(full_vocab, dtype=torch.bool))

        # Pre-allocated KV cache buffers for draft loop (avoid malloc per step)
        self._kv_k = None
        self._kv_v = None

    def _attention(self, hidden, input_emb, position_offset=0, past_kv=None):
        """Single attention layer with KV cache."""
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        if input_emb.dim() == 2:
            input_emb = input_emb.unsqueeze(1)

        B = hidden.shape[0]

        h_normed = self.hidden_norm(hidden)
        e_normed = self.input_layernorm(input_emb)
        combined = torch.cat([e_normed, h_normed], dim=-1)

        q = self.q_proj(combined).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(combined).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(combined).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        t = torch.arange(position_offset, position_offset + 1, device=q.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        def rotate(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        q = (q * cos) + (rotate(q) * sin)
        k = (k * cos) + (rotate(k) * sin)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                B, self.num_heads, -1, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)
        out = self.o_proj(out)

        hidden = hidden + out

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
        hidden = residual + hidden

        hidden = hidden.squeeze(1)
        return hidden, new_kv

    @torch.no_grad()
    def draft_tokens(self, hidden_single, cur_token_id, gamma, position_offset=0):
        """Predict gamma future tokens.  Input is a SINGLE hidden state [1, hs].

        Returns:
            draft_tokens: list[int] of length gamma (full vocab IDs)
            draft_logits: [gamma, draft_vocab] fp16
        """
        hidden = self.fc(hidden_single)  # [1, hs]
        input_emb = self.embed_tokens(torch.tensor([[cur_token_id]], device=hidden_single.device))

        draft_tokens_list = []
        draft_logits = []
        past_kv = None

        for step in range(min(gamma, self.ttt_length)):
            hidden, past_kv = self._attention(
                hidden, input_emb,
                position_offset=step,
                past_kv=past_kv,
            )
            logits = self.lm_head(self.norm(hidden))
            token = logits[0].argmax(dim=-1).item()
            draft_tokens_list.append(token)
            draft_logits.append(logits[0])

            if step < gamma - 1:
                input_emb = self.embed_tokens(torch.tensor([[token]], device=hidden_single.device))

        full_vocab_tokens = [self.d2t[t].item() for t in draft_tokens_list]
        return full_vocab_tokens, torch.stack(draft_logits)


def load_eagle3_draft_v2(device="cuda"):
    """Load EAGLE-3 draft model v2 (single-hidden input)."""
    if device in _cache:
        return _cache[device]

    ckpt_path = os.environ.get(
        "AICAS_EAGLE3_CKPT",
        "eagle_pipeline/data/checkpoints_v3/eagle_draft_best.pt"
    )
    logger.info(f"[EAGLE-3 v2] Loading draft model from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = Eagle3DraftV2(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg.get("num_heads", 16),
        num_kv_heads=cfg.get("num_kv_heads", 8),
        head_dim=cfg.get("head_dim", 128),
        intermediate_size=cfg.get("intermediate_size", 6144),
        draft_vocab=cfg["draft_vocab_size"],
        full_vocab=cfg.get("vocab_size", 151936),
        ttt_length=cfg.get("ttt_length", 7),
        rope_theta=cfg.get("rope_theta", 10000.0),
    )

    # Load state dict with key remapping
    raw_sd = ckpt["state_dict"]
    remapped = {}
    _prefix_map = {
        "midlayer.self_attn.": "",
        "midlayer.mlp.": "",
        "midlayer.": "",
    }
    for k, v in raw_sd.items():
        if k in ("d2t", "t2d"):
            continue
        new_key = k
        for old_pfx, new_pfx in _prefix_map.items():
            if new_key.startswith(old_pfx):
                new_key = new_pfx + new_key[len(old_pfx):]
                break

        # Handle fc.weight: v1 has [2048, 6144], v2 expects [2048, 2048]
        # If loading v1 checkpoint, average the 3 segments of fc.weight
        if new_key == "fc.weight" and v.shape[1] == cfg["hidden_size"] * 3:
            hs = cfg["hidden_size"]
            w1 = v[:, :hs]
            w2 = v[:, hs:2*hs]
            w3 = v[:, 2*hs:]
            v = (w1 + w2 + w3) / 3.0
            logger.info(f"[EAGLE-3 v2] Converted fc.weight from 3-layer concat to single "
                        f"({v.shape})")

        remapped[new_key] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.debug(f"[EAGLE-3 v2] Missing keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"[EAGLE-3 v2] Unexpected keys: {unexpected[:5]}...")

    # Load vocab mapping
    d2t = ckpt["d2t"] if "d2t" in ckpt else raw_sd.get("d2t")
    t2d = ckpt["t2d"] if "t2d" in ckpt else raw_sd.get("t2d")
    if d2t is not None:
        model.d2t.copy_(d2t)
    if t2d is not None:
        model.t2d.copy_(t2d)
    model.eval().to(device, dtype=torch.float16)
    model.d2d = model.d2t.long()
    model.t2d = model.t2d.bool()
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[EAGLE-3 v2] Draft model loaded: {n_params:,} params, "
                f"ttt_length={model.ttt_length}, draft_vocab={model.draft_vocab}")

    _cache[device] = model
    return model
