"""
Decode-optimized fused FFN Triton kernel for single-token generation (n=1).

Replaces the broken FlashDecodeFFN v3 fused kernel which launches only
1 block × 4 warps (128 threads) for the entire FFN, leaving 99% of the
GPU idle.

This implementation splits the work across I (intermediate) dimension:
  - grid=(I/128) blocks, each computing gate+up+silu for an I-chunk
    AND contributing to a partial down-projection output
  - A tiny reduction kernel sums across blocks

For Qwen3-VL-2B with I=5184, H=2048: ~41 blocks in step 1, ~16 in reduce.

NOTE: This kernel reads bf16 weights directly.  The cuBLAS legacy path uses
int4-quantized weights (4× smaller), so this kernel will be ~3-4× slower per
layer in raw GPU time.  The benefit comes from (a) enabling the fused path
for the future int4 version, and (b) reducing kernel launch count from 5 to 2
per layer, which matters outside of CUDA graphs.

Weight layout (v3 packed format):
  w13: [H, 2*I] bf16  — gate at [:, :I], up at [:, I:]
  w2:  [I, H] bf16
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ===========================================================================
# Step 1: fused gate+up+silu + partial down-project contribution
# ===========================================================================

@triton.jit
def _fused_step1_kernel(
    x_ptr,            # [1, H] bf16 contiguous
    w13_ptr,          # [H, 2*I] bf16 contiguous
    w2_ptr,           # [I, H] bf16 contiguous
    partial_out_ptr,  # [N_BLOCKS, H] float32
    H: int,
    I: int,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Each block handles one I-chunk: gate+up+silu → partial down contrib."""
    pid = tl.program_id(0)
    i_start = pid * BLOCK_I
    i_offs = i_start + tl.arange(0, BLOCK_I)
    i_mask = i_offs < I

    # ── Phase 1: gate = x @ w_gate, up = x @ w_up ──
    gate = tl.zeros([BLOCK_I], dtype=tl.float32)
    up = tl.zeros([BLOCK_I], dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        x_tile = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        # w13 layout: [H, 2*I].  Gate at [:, :I], up at [:, I:].
        wg_base = k_offs[:, None] * (2 * I) + i_offs[None, :]
        wu_base = k_offs[:, None] * (2 * I) + I + i_offs[None, :]
        m2d = k_mask[:, None] & i_mask[None, :]

        wg_tile = tl.load(w13_ptr + wg_base, mask=m2d, other=0.0)
        wu_tile = tl.load(w13_ptr + wu_base, mask=m2d, other=0.0)

        gate += tl.sum(x_tile[:, None] * wg_tile.to(tl.float32), axis=0)
        up += tl.sum(x_tile[:, None] * wu_tile.to(tl.float32), axis=0)

    # ── Activation: silu(gate) * up ──
    gated = gate * tl.sigmoid(gate) * up  # [BLOCK_I] float32

    # ── Phase 2: partial down projection for ALL H outputs ──
    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        # w2 layout: [I, H]
        w2_base = i_offs[:, None] * H + h_offs[None, :]
        m2d = i_mask[:, None] & h_mask[None, :]
        w2_tile = tl.load(w2_ptr + w2_base, mask=m2d, other=0.0)

        partial = tl.sum(gated[:, None] * w2_tile.to(tl.float32), axis=0)  # [BLOCK_H]

        # partial_out[pid, h_start:h_start+BLOCK_H] = partial
        tl.store(partial_out_ptr + pid * H + h_offs, partial, mask=h_mask)


# ===========================================================================
# Step 2: reduce partial outputs across blocks
# ===========================================================================

@triton.jit
def _fused_reduce_kernel(
    partial_out_ptr,  # [N_BLOCKS, H] float32
    out_ptr,          # [1, H] bf16
    H: int,
    N_BLOCKS: int,
    BLOCK_H: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Sum partial_out across the block dimension → final output."""
    pid = tl.program_id(0)
    h_start = pid * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for b_start in range(0, N_BLOCKS, BLOCK_B):
        b_offs = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offs < N_BLOCKS

        # Load partial[b_offs, h_offs]
        addr = b_offs[:, None] * H + h_offs[None, :]
        m2d = b_mask[:, None] & h_mask[None, :]
        partial = tl.load(partial_out_ptr + addr, mask=m2d, other=0.0)

        acc += tl.sum(partial.to(tl.float32), axis=0)

    tl.store(out_ptr + h_offs, acc.to(tl.bfloat16), mask=h_mask)


# ===========================================================================
# Workspace management
# ===========================================================================

class _DecodeFFNTritonState:
    def __init__(self):
        self.partial_out: torch.Tensor | None = None
        self._key: tuple | None = None

    def ensure(self, H: int, I: int, device: torch.device):
        n_blocks = triton.cdiv(I, 128)  # BLOCK_I=128
        key = (H, I, device, n_blocks)
        if key == self._key and self.partial_out is not None:
            return
        self.partial_out = torch.empty(n_blocks, H, device=device, dtype=torch.float32)
        self._key = key


_state = _DecodeFFNTritonState()


# ===========================================================================
# Public API
# ===========================================================================

# Fixed tile sizes — tuned for Qwen3-VL-2B (H=2048, I=5184) on PPU
_FIXED_BI = 128
_FIXED_BK = 64
_FIXED_BH = 128
_FIXED_BB = 16  # reduce block tile


def fused_ffn_decode_triton(
    x: torch.Tensor,           # [1, H] bf16
    w13_packed: torch.Tensor,  # [H, 2*I] bf16
    w2_packed: torch.Tensor,   # [I, H] bf16
) -> torch.Tensor:
    """Fused FFN for single-token decode.  Returns [1, H] bf16."""
    H = int(x.shape[1])
    I = int(w2_packed.shape[0])

    xc = x.contiguous().reshape(1, H)
    w13c = w13_packed.contiguous()
    w2c = w2_packed.contiguous()

    st = _state
    st.ensure(H, I, xc.device)
    n_blocks = st.partial_out.shape[0]

    out = torch.empty(1, H, dtype=torch.bfloat16, device=xc.device)

    # Step 1: gate+up+silu + partial down projection
    grid1 = (n_blocks,)
    _fused_step1_kernel[grid1](
        xc, w13c, w2c, st.partial_out,
        H, I,
        BLOCK_I=_FIXED_BI, BLOCK_K=_FIXED_BK, BLOCK_H=_FIXED_BH,
    )

    # Step 2: reduce partial outputs
    grid2 = (triton.cdiv(H, _FIXED_BH),)
    _fused_reduce_kernel[grid2](
        st.partial_out, out,
        H, n_blocks,
        BLOCK_H=_FIXED_BH, BLOCK_B=_FIXED_BB,
    )

    return out


def can_use_decode_ffn_triton(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.ndim == 2
        and x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
    )
