from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _eagle3_full_verify_prep_kernel(
    tree_mask_ptr,
    tree_pos_ptr,
    start_pos_ptr,
    rope_delta_ptr,
    attn_ptr,
    cache_pos_ptr,
    pos_ids_ptr,
    Q: tl.constexpr,
    KV: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    q = tl.arange(0, BLOCK_Q)
    k = tl.arange(0, BLOCK_K)
    q_mask = q < Q
    k_mask = k < KV
    start = tl.load(start_pos_ptr).to(tl.int64)

    q2 = q[:, None]
    k2 = k[None, :]
    before = k2 < start
    local = k2 - start
    local_valid = (local >= 0) & (local < Q)
    visible = tl.load(tree_mask_ptr + q2 * Q + local, mask=q_mask[:, None] & local_valid, other=0.0) != 0.0
    keep = before | visible
    neg = float("-3.3895313892515355e38")
    vals = tl.where(keep, 0.0, neg)
    tl.store(attn_ptr + q2 * KV + k2, vals, mask=q_mask[:, None] & k_mask[None, :])

    pos = tl.load(tree_pos_ptr + q, mask=q_mask, other=0).to(tl.int64) + start
    tl.store(cache_pos_ptr + q, start + q, mask=q_mask)
    tl.store(pos_ids_ptr + q, pos, mask=q_mask)
    delta = tl.load(rope_delta_ptr, mask=HAS_DELTA, other=0).to(tl.int64)
    mpos = pos + delta
    tl.store(pos_ids_ptr + Q + q, mpos, mask=q_mask)
    tl.store(pos_ids_ptr + 2 * Q + q, mpos, mask=q_mask)
    tl.store(pos_ids_ptr + 3 * Q + q, mpos, mask=q_mask)


def run_eagle3_full_verify_prep(
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
    start_pos: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    rope_delta: torch.Tensor | None = None,
) -> bool:
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in (
        tree_mask,
        tree_position_ids,
        start_pos,
        attention_mask,
        cache_position,
        position_ids,
    )):
        return False
    if tree_mask.ndim != 4 or attention_mask.ndim != 4 or position_ids.ndim != 3:
        return False
    q = int(tree_mask.shape[-1])
    if int(tree_mask.shape[-2]) != q:
        return False
    if tuple(attention_mask.shape[:3]) != (1, 1, q):
        return False
    if tuple(position_ids.shape) != (4, 1, q):
        return False
    if int(cache_position.numel()) != q or int(tree_position_ids.numel()) != q:
        return False
    if start_pos.numel() != 1:
        return False
    kv = int(attention_mask.shape[-1])
    if kv < q:
        return False
    if rope_delta is None or not isinstance(rope_delta, torch.Tensor) or not rope_delta.is_cuda:
        rope_delta = torch.empty((1,), device=tree_mask.device, dtype=torch.long)
        has_delta = False
    else:
        rope_delta = rope_delta.reshape(-1)
        has_delta = int(rope_delta.numel()) > 0
    _eagle3_full_verify_prep_kernel[(1,)](
        tree_mask,
        tree_position_ids.reshape(-1),
        start_pos.reshape(-1),
        rope_delta,
        attention_mask,
        cache_position,
        position_ids,
        Q=q,
        KV=kv,
        HAS_DELTA=has_delta,
        BLOCK_Q=triton.next_power_of_2(q),
        BLOCK_K=triton.next_power_of_2(kv),
        num_warps=8,
        num_stages=2,
    )
    return True
