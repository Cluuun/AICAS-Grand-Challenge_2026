"""Lightweight Medusa-style parallel draft heads.

Supports two modes:
  1. Full vocab: heads predict directly in 151936-way space
  2. Reduced vocab: heads predict in reduced space, then map back to full tokens

Training: KL divergence against teacher logits + CE loss.
"""

import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_cache = {}  # device -> MedusaHeads


class MedusaHeads(nn.Module):
    """N parallel prediction heads on top of the target model's hidden state."""

    def __init__(self, hidden_size: int = 2048, num_heads: int = 4,
                 full_vocab: int = 151936, reduced_vocab: int = 0,
                 bottleneck: int = 0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.full_vocab = full_vocab
        self.reduced_vocab = reduced_vocab or full_vocab
        self.bottleneck = bottleneck

        effective_vocab = self.reduced_vocab

        if bottleneck > 0:
            self.shared = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, bottleneck),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            in_dim = bottleneck
        else:
            self.shared = nn.LayerNorm(hidden_size)
            in_dim = hidden_size

        self.heads = nn.ModuleList([
            nn.Linear(in_dim, effective_vocab, bias=False)
            for _ in range(num_heads)
        ])

        # Mapping for reduced vocab mode
        self.register_buffer("reduced_to_token",
                             torch.arange(full_vocab, dtype=torch.long))
        self._using_reduced = self.reduced_vocab < self.full_vocab

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict N future tokens in parallel.

        Args:
            hidden: [hidden_size] fp16 hidden state from the decode step.

        Returns:
            draft_tokens: [num_heads] long tensor of predicted token IDs (full vocab).
        """
        h = self.shared(hidden.float().unsqueeze(0))  # [1, in_dim] float32
        tokens = []
        for head in self.heads:
            reduced_id = head(h)[0].argmax(dim=-1)  # scalar
            if self._using_reduced:
                full_id = self.reduced_to_token[reduced_id]
            else:
                full_id = reduced_id
            tokens.append(full_id)
        return torch.stack(tokens, dim=0)  # [num_heads]


def load_medusa_heads(device: str = "cuda") -> MedusaHeads:
    """Load trained Medusa draft heads from checkpoint."""
    if device in _cache:
        return _cache[device]

    ckpt_path = os.environ.get("AICAS_MEDUSA_CKPT", "model/medusa_heads.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Medusa checkpoint not found: {ckpt_path}. "
            f"Run: python3 training/train.py"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden_size = int(ckpt["hidden_size"])
    num_heads = int(ckpt["num_heads"])
    full_vocab = int(ckpt.get("full_vocab", 151936))
    reduced_vocab = int(ckpt.get("reduced_vocab", full_vocab))
    bottleneck = int(ckpt.get("bottleneck", 0))

    model = MedusaHeads(hidden_size, num_heads, full_vocab,
                        reduced_vocab, bottleneck)

    # Load weights (strict=False for buffers that may differ)
    sd = ckpt["state_dict"]
    model.load_state_dict(sd, strict=False)

    # Restore reduced_to_token mapping from checkpoint
    # Checkpoint has two sources:
    #   ckpt["reduced_to_token"] = [8192] tensor mapping reduced_id → full_token_id
    #   sd["reduced_to_token"] = buffer in state_dict (if saved)
    if "reduced_to_token" in ckpt and ckpt["reduced_to_token"].numel() == model.reduced_vocab:
        # This is the [reduced_vocab] mapping: reduced_id → full_token_id
        r2t = ckpt["reduced_to_token"].to(device)
        model.reduced_to_token[:model.reduced_vocab].copy_(r2t)
    elif "reduced_to_token" in sd:
        model.reduced_to_token.copy_(sd["reduced_to_token"])

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    mode = f"reduced={reduced_vocab}" if reduced_vocab < full_vocab else "full"
    logger.info(f"[MedusaHeads] Loaded from {ckpt_path}: "
                f"{num_heads} heads, bottleneck={bottleneck}, "
                f"vocab={mode}, {n_params:,} params")
    if "val_acc" in ckpt:
        logger.info(f"[MedusaHeads] Val acc: {ckpt['val_acc']}")

    _cache[device] = model
    return model
