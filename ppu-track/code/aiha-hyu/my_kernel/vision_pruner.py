"""
[SHLEE] Vision token pruning — VisionZip-inspired, training-free, upstream.

Hook point: after vision encoder produces `image_embeds` (the merger output,
shape [N_visual, H]) and before they are scattered into inputs_embeds for the
LLM.  We select K = round(N * keep_ratio) tokens to keep, drop the rest, and
compact every tensor that depends on the original sequence length.

Scoring: "dominant-token" rank inspired by VisionZip.  We compute the cosine
similarity matrix S = cos(v_i, v_j) and define score_i = mean_j S_{j,i} —
i.e. how strongly token i is "looked at" by other visual tokens.  High score
means the token's content is broadly representative.  No model attention
weights are needed; just two matmuls.

Constraints respected:
  * Kept indices are returned in original order so the contiguity assumption
    of fused_deepstack_process (start_offset=4, contiguous visual block) is
    preserved.
  * No training, no calibration.
"""

import os
import torch

def _read_keep_ratio() -> float:

    v = os.environ.get("MY_KERNEL_VISION_KEEP_RATIO", "1.0")
    try:
        r = float(v)
    except ValueError:
        return 1.0
    return max(0.05, min(1.0, r))

def _read_prune_thresh() -> int:
    """Selective pruning: only prune when N_visual > threshold."""
    v = os.environ.get("MY_KERNEL_VISION_PRUNE_THRESH", "0")
    try:
        return int(v)
    except ValueError:
        return 0

def is_pruning_enabled_for(num_visual_tokens: int) -> bool:
    """Cheap check for callers to short-circuit before scoring kernels."""
    if _read_keep_ratio() >= 0.999:
        return False
    t = _read_prune_thresh()
    return num_visual_tokens > t if t > 0 else True

def is_pruning_enabled() -> bool:
    return _read_keep_ratio() < 0.999

def select_dominant_indices(image_embeds: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    Args:
        image_embeds: [N, H] visual tokens after merger.
        keep_ratio: in (0, 1].
    Returns:
        kept_idx: [K] LongTensor, sorted ascending (preserves original order).
    """
    N = image_embeds.shape[0]
    K = max(1, int(round(N * keep_ratio)))
    if K >= N:
        return torch.arange(N, device=image_embeds.device)

    x = image_embeds.float()
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    sim = x @ x.transpose(0, 1)
    sim.fill_diagonal_(0.0)
    score = sim.mean(dim=0)

    topk = torch.topk(score, K, largest=True).indices
    topk, _ = torch.sort(topk)
    return topk

def compact_sequence(
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    visual_pos_in_seq: torch.Tensor,
    kept_visual_local_idx: torch.Tensor,
    deepstack_visual_embeds,
):
    """Remove the dropped visual positions from the sequence.

    Returns: (inputs_embeds', position_ids', visual_pos_masks', deepstack')
    """
    device = inputs_embeds.device
    S = inputs_embeds.shape[1]

    keep_seq = torch.ones(S, dtype=torch.bool, device=device)
    N = visual_pos_in_seq.numel()
    if kept_visual_local_idx.numel() < N:
        drop_mask = torch.ones(N, dtype=torch.bool, device=device)
        drop_mask[kept_visual_local_idx] = False
        drop_seq_pos = visual_pos_in_seq[drop_mask]
        keep_seq[drop_seq_pos] = False

    inputs_embeds_c = inputs_embeds[:, keep_seq, :].contiguous()

    if position_ids is None:
        position_ids_c = None
    else:
        position_ids_c = position_ids[..., keep_seq].contiguous()

    visual_pos_masks_c = torch.zeros(
        inputs_embeds_c.shape[0], inputs_embeds_c.shape[1],
        dtype=torch.bool, device=device,
    )

    new_index = keep_seq.long().cumsum(0) - 1
    kept_seq_pos = visual_pos_in_seq[kept_visual_local_idx]
    visual_pos_masks_c[0, new_index[kept_seq_pos]] = True

    deepstack_c = None
    if deepstack_visual_embeds is not None:
        deepstack_c = [d.index_select(0, kept_visual_local_idx) for d in deepstack_visual_embeds]

    return inputs_embeds_c, position_ids_c, visual_pos_masks_c, deepstack_c

def maybe_prune(
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    image_embeds: torch.Tensor,
    deepstack_visual_embeds,
):
    """
    Returns either (None,) signalling "no-op" or
    (inputs_embeds, position_ids, visual_pos_masks, deepstack, kept_K).
    """
    keep_ratio = _read_keep_ratio()
    if keep_ratio >= 0.999:
        return None

    visual_pos_in_seq = (input_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    if visual_pos_in_seq.numel() == 0:
        return None

    thresh = _read_prune_thresh()
    if thresh > 0 and visual_pos_in_seq.numel() <= thresh:
        return None

    kept_idx = select_dominant_indices(image_embeds, keep_ratio)

    inputs_embeds_c, position_ids_c, vpm_c, ds_c = compact_sequence(
        inputs_embeds, position_ids, visual_pos_in_seq, kept_idx,
        deepstack_visual_embeds,
    )
    return inputs_embeds_c, position_ids_c, vpm_c, ds_c, kept_idx.numel()
