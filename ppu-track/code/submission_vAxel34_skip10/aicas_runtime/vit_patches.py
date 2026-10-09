"""Tensor-only replacements for ViT positional / RoPE / get_rope_index helpers
that the upstream HF Qwen3-VL implementation expresses in Python loops."""
from __future__ import annotations

import torch


def make_fast_pos_embed_interpolate(vit_obj):
    pos_embed = vit_obj.pos_embed
    num_grid = vit_obj.num_grid_per_side
    ng_minus_1 = float(num_grid - 1)
    merge_size = vit_obj.spatial_merge_size
    embed_dim = pos_embed.embedding_dim

    def fast_pos_embed_interpolate(grid_thw):
        device = pos_embed.weight.device
        dtype = pos_embed.weight.dtype
        t = grid_thw[0, 0]
        h = grid_thw[0, 1]
        w = grid_thw[0, 2]

        h_scale = ng_minus_1 / (h.float() - 1.0).clamp(min=1.0)
        w_scale = ng_minus_1 / (w.float() - 1.0).clamp(min=1.0)

        h_idxs = torch.arange(h, device=device, dtype=torch.float32) * h_scale
        w_idxs = torch.arange(w, device=device, dtype=torch.float32) * w_scale

        h_floor = h_idxs.long()
        h_ceil = (h_floor + 1).clamp(max=num_grid - 1)
        w_floor = w_idxs.long()
        w_ceil = (w_floor + 1).clamp(max=num_grid - 1)

        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()

        idx_00 = (h_floor[:, None] * num_grid + w_floor[None, :]).reshape(-1)
        idx_01 = (h_floor[:, None] * num_grid + w_ceil[None, :]).reshape(-1)
        idx_10 = (h_ceil[:, None] * num_grid + w_floor[None, :]).reshape(-1)
        idx_11 = (h_ceil[:, None] * num_grid + w_ceil[None, :]).reshape(-1)

        w_00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1, 1).to(dtype)
        w_01 = ((1 - dh)[:, None] * dw[None, :]).reshape(-1, 1).to(dtype)
        w_10 = (dh[:, None] * (1 - dw)[None, :]).reshape(-1, 1).to(dtype)
        w_11 = (dh[:, None] * dw[None, :]).reshape(-1, 1).to(dtype)

        patch_pos = (w_00 * pos_embed(idx_00) + w_01 * pos_embed(idx_01)
                     + w_10 * pos_embed(idx_10) + w_11 * pos_embed(idx_11))

        patch_pos = patch_pos.unsqueeze(0).expand(t, -1, -1).reshape(t * h * w, embed_dim)
        patch_pos = (patch_pos
                     .view(t, h // merge_size, merge_size, w // merge_size, merge_size, embed_dim)
                     .permute(0, 1, 3, 2, 4, 5)
                     .flatten(0, 4))
        return patch_pos

    return fast_pos_embed_interpolate


def make_fast_rot_pos_emb(vit_obj):
    MAX_POS = 512
    with torch.no_grad():
        full_freq = vit_obj.rotary_pos_emb(MAX_POS)
    freq_table = full_freq.detach()
    merge_size = vit_obj.spatial_merge_size

    def fast_rot_pos_emb(grid_thw):
        h = grid_thw[0, 1]
        w = grid_thw[0, 2]
        merged_h = h // merge_size
        merged_w = w // merge_size

        block_r = torch.arange(merged_h, device=grid_thw.device)
        block_c = torch.arange(merged_w, device=grid_thw.device)
        intra_r = torch.arange(merge_size, device=grid_thw.device)
        intra_c = torch.arange(merge_size, device=grid_thw.device)

        row_idx = (
            block_r[:, None, None, None] * merge_size + intra_r[None, None, :, None]
        ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        col_idx = (
            block_c[None, :, None, None] * merge_size + intra_c[None, None, None, :]
        ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        row_freq = freq_table[row_idx]
        col_freq = freq_table[col_idx]

        return torch.stack([row_freq, col_freq], dim=1).reshape(-1, freq_table.shape[1] * 2)

    return fast_rot_pos_emb


def make_fast_get_rope_index(model_obj):
    spatial_merge_size = model_obj.config.vision_config.spatial_merge_size
    orig_get_rope_index = model_obj.get_rope_index
    image_token_id = model_obj.config.image_token_id

    def fast_get_rope_index(
        input_ids=None, image_grid_thw=None,
        video_grid_thw=None, attention_mask=None
    ):
        if input_ids is None or image_grid_thw is None:
            return orig_get_rope_index(
                input_ids, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, attention_mask=attention_mask)

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        mm_token_type_ids = (input_ids == image_token_id).long()

        position_ids = torch.zeros(3, batch_size, seq_len,
                                   dtype=input_ids.dtype, device=device)
        mrope_position_deltas = []
        image_idx = 0

        for batch_idx in range(batch_size):
            tok_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                mask = attention_mask[batch_idx].bool()
                tok_type = tok_type[mask]
                cur_len = int(mask.sum().item())
            else:
                cur_len = seq_len

            is_vis = (tok_type == 1)
            n_vis = int(is_vis.sum().item())

            if n_vis == 0:
                text_pos = torch.arange(cur_len, device=device)
                llm_positions = text_pos.view(1, -1).expand(3, -1)
                mrope_position_deltas.append(torch.tensor(0, device=device))
            else:
                n_prefix = int((~is_vis).long().cumsum(0)[is_vis].min().item())
                grid_thw = image_grid_thw[image_idx]
                llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
                llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
                image_idx += 1
                current_pos = n_prefix

                if n_prefix > 0:
                    text_before = torch.arange(n_prefix, device=device).view(1, -1).expand(3, -1)
                else:
                    text_before = torch.zeros(3, 0, dtype=torch.long, device=device)

                img_seq_len = llm_grid_h * llm_grid_w
                pos_w = torch.arange(current_pos, current_pos + llm_grid_w, device=device).repeat(llm_grid_h)
                pos_h = torch.arange(current_pos, current_pos + llm_grid_h, device=device).repeat_interleave(llm_grid_w)
                pos_t = torch.full((img_seq_len,), current_pos, device=device, dtype=torch.long)
                vision_ids = torch.stack([pos_t, pos_h, pos_w], dim=0)

                vision_advance = max(llm_grid_h, llm_grid_w)
                current_pos_after = current_pos + vision_advance
                n_suffix = cur_len - n_prefix - n_vis

                if n_suffix > 0:
                    text_after = torch.arange(current_pos_after, current_pos_after + n_suffix, device=device).view(1, -1).expand(3, -1)
                else:
                    text_after = torch.zeros(3, 0, dtype=torch.long, device=device)

                llm_positions = torch.cat([text_before, vision_ids, text_after], dim=1)
                delta = int(llm_positions.max().item()) + 1 - cur_len
                mrope_position_deltas.append(
                    torch.tensor(delta, device=device, dtype=torch.long))

            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions

        mrope_position_deltas = torch.stack(mrope_position_deltas).unsqueeze(1)
        return position_ids, mrope_position_deltas

    return fast_get_rope_index
