"""Manual ViT forward (FA2 + fused QKV/MLP + optional in-block prune).

Returns (merged, (ds0, ds1, ds2), keep_blocks_or_None).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from prune_optim import (
    PruneConfig, compute_n_keep, select_keep_blocks, gather_blocks_4row,
)
from quant_optim import w8a8_linear

from .custom_ops import flash_attn_vit_op


def make_manual_vit_forward_fused(vit, prune_config: PruneConfig = None,
                                  vit_w8a8_fc1=None):
    if prune_config is None:
        prune_config = PruneConfig(enabled=False)
    rows_per_block = prune_config.rows_per_block
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    patch_embed = vit.patch_embed
    merger = vit.merger
    deepstack_mergers = vit.deepstack_merger_list
    ds_merger_0 = deepstack_mergers[0]
    ds_merger_1 = deepstack_mergers[1]
    ds_merger_2 = deepstack_mergers[2]
    fast_pos_embed_fn = vit.fast_pos_embed_interpolate
    fast_rot_fn = vit.rot_pos_emb
    spatial_merge_size = vit.spatial_merge_size
    assert spatial_merge_size * spatial_merge_size == rows_per_block, (
        f"PruneConfig.rows_per_block ({rows_per_block}) must equal "
        f"spatial_merge_size**2 ({spatial_merge_size**2})"
    )

    vit_block_params = []
    for blk in vit.blocks:
        attn = blk.attn
        mlp = blk.mlp
        vit_block_params.append({
            'qkv_weight': attn.qkv.weight,
            'qkv_bias': attn.qkv.bias,
            'proj_weight': attn.proj.weight,
            'proj_bias': attn.proj.bias,
            'num_heads': attn.num_heads,
            'scaling': attn.scaling,
            'fc1_weight': mlp.linear_fc1.weight,
            'fc1_bias': mlp.linear_fc1.bias,
            'fc2_weight': mlp.linear_fc2.weight,
            'fc2_bias': mlp.linear_fc2.bias,
            'norm1': blk.norm1,
            'norm2': blk.norm2,
            'fc1_w8a8': (vit_w8a8_fc1.fc1[len(vit_block_params)]
                         if vit_w8a8_fc1 is not None else None),
        })

    def manual_vit_forward(pixel_values, grid_thw):
        hidden_states = patch_embed(pixel_values)
        pos_embeds = fast_pos_embed_fn(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = fast_rot_fn(grid_thw)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        ds_feat_0 = None
        ds_feat_1 = None
        ds_feat_2 = None
        kept_blocks_out = None

        cos_pe, sin_pe = position_embeddings

        for layer_num, bp in enumerate(vit_block_params):
            residual = hidden_states
            hidden_states = bp['norm1'](hidden_states)

            qkv = F.linear(hidden_states, bp['qkv_weight'], bp['qkv_bias'])
            seq_l = hidden_states.shape[0]
            num_heads = bp['num_heads']
            head_dim = qkv.shape[-1] // (3 * num_heads)

            q, k, v = qkv.reshape(seq_l, 3, num_heads, head_dim).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, cos_pe, sin_pe)

            q_fa = q.transpose(0, 1).unsqueeze(0).contiguous()
            k_fa = k.transpose(0, 1).unsqueeze(0).contiguous()
            v_fa = v.transpose(0, 1).unsqueeze(0).contiguous()

            attn_out = flash_attn_vit_op(q_fa, k_fa, v_fa, False, bp['scaling'])
            attn_out = (attn_out.squeeze(0).transpose(0, 1)
                        .reshape(seq_l, -1).contiguous())

            attn_out = F.linear(attn_out, bp['proj_weight'], bp['proj_bias'])
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = bp['norm2'](hidden_states)

            if bp['fc1_w8a8'] is not None:
                hidden_states = w8a8_linear(hidden_states, bp['fc1_w8a8'])
            else:
                hidden_states = F.linear(hidden_states, bp['fc1_weight'], bp['fc1_bias'])
            hidden_states = F.gelu(hidden_states, approximate='tanh')
            hidden_states = F.linear(hidden_states, bp['fc2_weight'], bp['fc2_bias'])
            hidden_states = residual + hidden_states

            if layer_num == 5:
                ds_feat_0 = ds_merger_0(hidden_states)
                if (prune_config is not None and prune_config.enabled
                        and prune_config.prune_ratio > 0.0):
                    n_merged = ds_feat_0.shape[0]
                    n_keep = compute_n_keep(n_merged, prune_config)
                    if n_keep < n_merged:
                        importance = ds_feat_0.norm(dim=-1)
                        keep_blocks = select_keep_blocks(importance, n_keep)
                        hidden_states = gather_blocks_4row(hidden_states, keep_blocks, rows_per_block)
                        cos_pe = gather_blocks_4row(cos_pe, keep_blocks, rows_per_block)
                        sin_pe = gather_blocks_4row(sin_pe, keep_blocks, rows_per_block)
                        ds_feat_0 = ds_feat_0.index_select(0, keep_blocks)
                        kept_blocks_out = keep_blocks
            elif layer_num == 11:
                ds_feat_1 = ds_merger_1(hidden_states)
            elif layer_num == 17:
                ds_feat_2 = ds_merger_2(hidden_states)

        merged_hidden_states = merger(hidden_states)
        return merged_hidden_states, (ds_feat_0, ds_feat_1, ds_feat_2), kept_blocks_out

    return manual_vit_forward
