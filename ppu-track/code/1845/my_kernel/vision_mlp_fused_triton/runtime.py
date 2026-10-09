"""ViT MLP: PyTorch norm+up-proj+act, Triton-fused GELU-tanh + down-proj (Qwen3-VL-2B)."""
from __future__ import annotations

import os
from types import MethodType
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from my_kernel.vision_mlp_fused_triton.kernel import gelu_fc2_qwen3vl_2b_row

if TYPE_CHECKING:
    pass

_EXPECT_H = 1024
_EXPECT_I = 4096


def _enabled() -> bool:
    return os.getenv("AICAS_VISION_MLP_FUSED_TRITON", "1").strip().lower() in ("1", "true", "yes")


def _can_run_mid(m: torch.Tensor, mlp: nn.Module) -> bool:
    if not _enabled():
        return False
    if not (m.is_cuda and m.dtype == torch.bfloat16):
        return False
    if m.dim() != 2 or m.shape[1] != _EXPECT_I:
        return False
    if m.stride(1) != 1 or not m.is_contiguous():
        return False
    fc2 = getattr(mlp, "linear_fc2", None)
    if fc2 is None or fc2.weight.shape != (_EXPECT_H, _EXPECT_I):
        return False
    if fc2.bias is None:
        return False
    return True


def fused_gelu_down_forward(mlp: nn.Module, mid: torch.Tensor) -> torch.Tensor:
    """mid: [M, I] bf16 (post act_fn). Returns linear_fc2(act) equivalent via Triton."""
    if not _can_run_mid(mid, mlp):
        return mlp.linear_fc2(mid)

    M = mid.shape[0]
    w2, b2 = mlp.linear_fc2.weight, mlp.linear_fc2.bias
    out2 = torch.empty((M, _EXPECT_H), device=mid.device, dtype=torch.bfloat16)
    stride_w2_0, stride_w2_1 = w2.stride()
    BLOCK_H = 32
    gelu_fc2_qwen3vl_2b_row[(M,)](
        mid,
        w2,
        b2,
        out2,
        M,
        mid.stride(0),
        out2.stride(0),
        stride_w2_0,
        stride_w2_1,
        H=_EXPECT_H,
        I=_EXPECT_I,
        BLOCK_H=BLOCK_H,
        BLOCK_I=128,
    )
    return out2


def fused_norm2_mlp_forward(norm2: nn.LayerNorm, mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Same as mlp(norm2(x)) for Qwen3-VL-2B ViT; GELU-tanh + down uses Triton when eligible."""
    mid = mlp.linear_fc1(norm2(x))
    return fused_gelu_down_forward(mlp, mid)


def patch_vision_blocks_fused_mlp(visual) -> int:
    if not _enabled():
        return 0
    patched = 0
    for blk in getattr(visual, "blocks", []) or []:
        if getattr(blk, "_aicas_fused_mlp_fwd", False):
            continue
        norm2 = getattr(blk, "norm2", None)
        mlp = getattr(blk, "mlp", None)
        if norm2 is None or mlp is None:
            continue

        def _fwd(
            self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=None,
            position_embeddings=None,
            **kwargs,
        ):
            attn_out = self.attn(
                self.norm1(hidden_states),
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            h = hidden_states + attn_out
            mlp_out = fused_norm2_mlp_forward(self.norm2, self.mlp, h)
            return h + mlp_out

        blk.forward = MethodType(_fwd, blk)
        blk._aicas_fused_mlp_fwd = True
        patched += 1
    return patched
