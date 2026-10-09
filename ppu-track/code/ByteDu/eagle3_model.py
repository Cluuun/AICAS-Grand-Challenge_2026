from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Eagle3DraftConfig:
    layer_ids: tuple[int, ...]
    hidden_size: int
    vocab_size: int
    draft_len: int = 4
    mlp_hidden_size: int | None = None

    @classmethod
    def from_checkpoint(cls, payload: dict, *, default_hidden_size: int, default_vocab_size: int) -> "Eagle3DraftConfig":
        metadata = payload.get("metadata", {})
        layer_ids = tuple(int(layer_id) for layer_id in payload.get("layer_ids", metadata.get("layer_ids", (3, 13, 27))))
        hidden_size = int(payload.get("hidden_size", metadata.get("hidden_size", default_hidden_size)))
        vocab_size = int(payload.get("vocab_size", metadata.get("vocab_size", default_vocab_size)))
        draft_len = int(payload.get("draft_len", metadata.get("draft_len", 4)))
        mlp_hidden_size = payload.get("mlp_hidden_size", metadata.get("mlp_hidden_size"))
        if mlp_hidden_size is not None:
            mlp_hidden_size = int(mlp_hidden_size)
        return cls(
            layer_ids=layer_ids,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            draft_len=draft_len,
            mlp_hidden_size=mlp_hidden_size,
        )


class Eagle3DraftModel(nn.Module):
    """EAGLE-3 style draft head using fused target hidden states.

    The module predicts token logits directly from selected low/mid/high target
    hidden states plus the previous token embedding. It intentionally does not
    regress hidden features from the target model.
    """

    def __init__(self, config: Eagle3DraftConfig) -> None:
        super().__init__()
        self.config = config
        fused_hidden = config.hidden_size if config.mlp_hidden_size is None else config.mlp_hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(len(config.layer_ids) * config.hidden_size, fused_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(fused_hidden, config.hidden_size, bias=False),
        )
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.step = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def fuse_hidden(self, selected_hidden: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(selected_hidden) != len(self.config.layer_ids):
            raise ValueError(
                f"Expected {len(self.config.layer_ids)} selected hidden tensors, got {len(selected_hidden)}."
            )
        last_token_states = []
        for hidden in selected_hidden:
            if hidden.ndim == 3:
                last_token_states.append(hidden[:, -1, :])
            elif hidden.ndim == 2:
                last_token_states.append(hidden)
            else:
                raise ValueError(f"Expected selected hidden tensor rank 2 or 3, got shape {tuple(hidden.shape)}.")
        return self.fusion(torch.cat(last_token_states, dim=-1))

    def forward(self, selected_hidden: Sequence[torch.Tensor], previous_token: torch.Tensor) -> torch.Tensor:
        state = self.fuse_hidden(selected_hidden)
        token_emb = self.token_embed(previous_token.view(-1).to(dtype=torch.long))
        state = self.step(torch.cat([state, token_emb], dim=-1))
        return self.lm_head(state)

    @torch.inference_mode()
    def draft(self, selected_hidden: Sequence[torch.Tensor], previous_token: torch.Tensor, steps: int) -> torch.Tensor:
        if steps <= 0:
            return torch.empty((1, 0), device=previous_token.device, dtype=torch.long)
        state = self.fuse_hidden(selected_hidden)
        prev = previous_token.view(-1).to(dtype=torch.long)
        out = torch.empty((1, steps), device=previous_token.device, dtype=torch.long)
        for step_idx in range(steps):
            token_emb = self.token_embed(prev)
            state = self.step(torch.cat([state, token_emb], dim=-1))
            logits = self.lm_head(state)
            prev = torch.argmax(logits, dim=-1)
            out[:, step_idx] = prev
        return out


def load_eagle3_draft_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    vocab_size: int,
) -> Eagle3DraftModel:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"EAGLE-3 checkpoint must be a dict payload, got {type(payload)!r}.")
    config = Eagle3DraftConfig.from_checkpoint(
        payload,
        default_hidden_size=hidden_size,
        default_vocab_size=vocab_size,
    )
    if config.hidden_size != hidden_size:
        raise ValueError(f"EAGLE-3 hidden_size mismatch: checkpoint={config.hidden_size}, target={hidden_size}.")
    if config.vocab_size != vocab_size:
        raise ValueError(f"EAGLE-3 vocab_size mismatch: checkpoint={config.vocab_size}, target={vocab_size}.")
    state_dict = payload.get("state_dict", payload.get("model_state_dict"))
    if state_dict is None:
        raise KeyError("EAGLE-3 checkpoint is missing `state_dict`.")
    model = Eagle3DraftModel(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"EAGLE-3 checkpoint load mismatch: missing={missing}, unexpected={unexpected}.")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def eagle3_multistep_loss(
    model: Eagle3DraftModel,
    selected_hidden: Sequence[torch.Tensor],
    previous_tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    step_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape [batch, steps], got {tuple(labels.shape)}.")
    steps = labels.shape[1]
    if previous_tokens.ndim == 2:
        prev = previous_tokens[:, 0]
    else:
        prev = previous_tokens.view(-1)
    state = model.fuse_hidden(selected_hidden)
    losses = []
    weights = step_weights or tuple(1.0 / (idx + 1) for idx in range(steps))
    if len(weights) < steps:
        raise ValueError(f"Need at least {steps} step weights, got {len(weights)}.")
    for step_idx in range(steps):
        token_emb = model.token_embed(prev)
        state = model.step(torch.cat([state, token_emb], dim=-1))
        logits = model.lm_head(state)
        losses.append(F.cross_entropy(logits.float(), labels[:, step_idx].long()) * float(weights[step_idx]))
        prev = labels[:, step_idx].long()
    return torch.stack(losses).sum() / max(float(sum(weights[:steps])), 1e-6)
