"""Lightweight draft model for speculative decoding.

Predicts future hidden states from the current hidden state using a tiny MLP.
Architecture: LayerNorm(2048) -> Linear(2048,256) -> SiLU -> Linear(256,256)
              -> SiLU -> Linear(256, horizon*2048) + residual connection.
"""

import os
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_cache = {}  # device -> DraftHead


class DraftHead(nn.Module):
    """Predict future hidden states from current hidden state.

    Output[i] = input + delta[i] * learnable_scale[i]  for i in 0..horizon-1
    Position 0 is a reconstruction of the current hidden state (training artefact).
    Positions 1..horizon-1 are the actual future predictions.
    """

    def __init__(self, hidden_size: int = 2048, horizon: int = 4, width: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, horizon * hidden_size),
        )
        self.residual_scale = nn.Parameter(torch.ones(horizon) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, hidden_size] (float32) -> [B, horizon, hidden_size] (float32)"""
        x_float = x.float()
        delta = self.net(x_float).view(x.shape[0], self.horizon, self.hidden_size)
        scale = self.residual_scale.view(1, self.horizon, 1).float()
        return x_float.unsqueeze(1) + delta * scale


def load_draft_head(device: str = "cuda") -> DraftHead:
    """Load the trained draft model.  Results are cached per device."""
    if device in _cache:
        return _cache[device]

    ckpt_path = os.environ.get(
        "AICAS_EAGLE_DRAFT_CKPT", "trash/draft_hidden_12x48_ce.pt"
    )
    logger.info(f"[DraftHead] Loading from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden_size = int(ckpt["hidden_size"])
    horizon = int(ckpt["horizon"])
    width = int(ckpt["width"])

    model = DraftHead(hidden_size, horizon, width)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[DraftHead] Loaded (hidden={hidden_size}, horizon={horizon}, "
                f"width={width}, params={n_params:,})")

    _cache[device] = model
    return model
