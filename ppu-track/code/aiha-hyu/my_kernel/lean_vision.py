"""
[SHLEE] Lean Vision Encoder: bypasses nn.Module.__call__ dispatch overhead in 24 ViT blocks.

Optimizations:
1. Bypass nn.Module.__call__ dispatch (no hooks, no grad mode checks per block)
2. Cache pos_embed and rotary embeddings per grid_thw
3. Direct F.linear + flash_attn_varlen_func calls
"""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

from my_kernel.fused_gelu import fused_gelu
from my_kernel.fused_layernorm import FusedLayerNorm
import my_kernel.fused_layernorm as _ln_module_holder

import transformers.models.qwen3_vl.modeling_qwen3_vl as _qmod

def _fused_layer_norm(x, weight, bias, eps):
    """Direct call to our tuned vision layernorm kernel — bypasses both
    nn.Module dispatch (vs blk.norm1(x)) AND torch's generic F.layer_norm
    dispatch.  Kernel is vectorised float4 with BLOCK=128, tuned for the
    hidden=1024 vision encoder shape.
    """
    mod = _ln_module_holder._module
    if mod is None:

        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return mod.fused_layernorm_forward(x, weight, bias, eps)

class LeanVisionState:
    """Pre-extracted weights for lean vision forward."""

    def __init__(self, visual):
        import os
        total_blocks = len(visual.blocks)
        cfg = visual.config
        ds_idx = list(getattr(cfg, 'deepstack_visual_indexes', []) or [])

        drop = int(os.environ.get("MY_KERNEL_VIT_DROP_TAIL", "0"))
        min_blocks = (max(ds_idx) + 1) if ds_idx else 1
        keep = max(min_blocks, total_blocks - max(0, drop))
        if keep < total_blocks:
            print(f"[lean_vision] ViT tail-drop: keeping {keep}/{total_blocks} "
                  f"blocks (deepstack indices preserved: {ds_idx})")

        self.num_blocks = keep
        self.total_blocks = total_blocks
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.intermediate_size = cfg.intermediate_size
        self.spatial_merge_size = cfg.spatial_merge_size
        self.deepstack_visual_indexes = ds_idx

        self.n1_w = []
        self.n1_b = []
        self.n1_eps = []
        self.qkv_w = []
        self.qkv_b = []
        self.proj_w = []
        self.proj_b = []
        self.n2_w = []
        self.n2_b = []
        self.n2_eps = []
        self.fc1_w = []
        self.fc1_b = []
        self.fc2_w = []
        self.fc2_b = []

        for blk in visual.blocks:
            n1 = blk.norm1
            if isinstance(n1, FusedLayerNorm):
                self.n1_w.append(n1.weight)
                self.n1_b.append(n1.bias)
                self.n1_eps.append(n1.eps)
            else:
                self.n1_w.append(n1.weight.data)
                self.n1_b.append(n1.bias.data)
                self.n1_eps.append(n1.eps)

            self.qkv_w.append(blk.attn.qkv.weight.data)
            self.qkv_b.append(blk.attn.qkv.bias.data)
            self.proj_w.append(blk.attn.proj.weight.data)
            self.proj_b.append(blk.attn.proj.bias.data)

            n2 = blk.norm2
            if isinstance(n2, FusedLayerNorm):
                self.n2_w.append(n2.weight)
                self.n2_b.append(n2.bias)
                self.n2_eps.append(n2.eps)
            else:
                self.n2_w.append(n2.weight.data)
                self.n2_b.append(n2.bias.data)
                self.n2_eps.append(n2.eps)

            self.fc1_w.append(blk.mlp.linear_fc1.weight.data)
            self.fc1_b.append(blk.mlp.linear_fc1.bias.data)
            self.fc2_w.append(blk.mlp.linear_fc2.weight.data)
            self.fc2_b.append(blk.mlp.linear_fc2.bias.data)

        self.visual = visual
        self.patch_embed = visual.patch_embed
        self.merger = visual.merger
        self.deepstack_merger_list = (
            visual.deepstack_merger_list if hasattr(visual, 'deepstack_merger_list') else []
        )

        self._pos_cache = {}
        self._rope_cache = {}

def lean_vision_forward(st, hidden_states, grid_thw):
    """
    Lean replacement for Qwen3VLVisionModel.forward.
    hidden_states: raw pixel patches (before patch_embed)
    grid_thw: [num_images, 3] tensor of (t, h, w)
    Returns: (merged_hidden_states, deepstack_features)

    HiPrune (env MY_KERNEL_VIT_HIPRUNE_KEEP, default 1.0 = off):
      After the last deepstack feature is collected, score each token
      by hidden-state L2 norm, keep top-`keep_ratio` (by image group),
      drop the rest.  Remaining ViT blocks + merger run on the pruned
      set, and the LLM sees fewer visual tokens.
    """
    import os
    hiprune_keep = float(os.environ.get("MY_KERNEL_VIT_HIPRUNE_KEEP", "1.0"))
    hiprune_keep = max(0.05, min(1.0, hiprune_keep))

    from my_kernel.profiler import profiler
    profiler.mark("vit.start")

    hidden_states = st.patch_embed(hidden_states)
    profiler.mark("vit.patch_embed")
    S = hidden_states.shape[0]
    H = st.hidden_size
    nH = st.num_heads
    hd = st.head_dim

    grid_list = grid_thw.tolist()
    cache_key = tuple(grid_list[0]) if grid_thw.shape[0] == 1\
                else tuple(map(tuple, grid_list))
    if cache_key in st._pos_cache:
        pos_embeds = st._pos_cache[cache_key]
    else:
        pos_embeds = st.visual.fast_pos_embed_interpolate(grid_thw)
        st._pos_cache[cache_key] = pos_embeds

    hidden_states = hidden_states + pos_embeds

    if cache_key in st._rope_cache:
        cos, sin = st._rope_cache[cache_key]
    else:
        rotary = st.visual.rot_pos_emb(grid_thw)
        rotary = rotary.reshape(S, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        st._rope_cache[cache_key] = (cos, sin)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    max_seqlen = max(int(t * h * w) for t, h, w in grid_list)

    deepstack_features = []

    profiler.mark("vit.pre_blocks")
    _profile_quarter = max(1, st.num_blocks // 4)
    for i in range(st.num_blocks):
        residual = hidden_states

        normed = _fused_layer_norm(hidden_states, st.n1_w[i], st.n1_b[i], st.n1_eps[i])

        qkv = F.linear(normed, st.qkv_w[i], st.qkv_b[i])
        qkv = qkv.reshape(S, 3, nH, hd)
        q, k, v = qkv.unbind(1)

        q, k = _qmod.apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )
        attn_out = attn_out.reshape(S, H)

        hidden_states = residual + F.linear(attn_out, st.proj_w[i], st.proj_b[i])

        residual = hidden_states
        normed2 = _fused_layer_norm(hidden_states, st.n2_w[i], st.n2_b[i], st.n2_eps[i])
        fc1_out = fused_gelu(F.linear(normed2, st.fc1_w[i], st.fc1_b[i]))
        hidden_states = residual + F.linear(fc1_out, st.fc2_w[i], st.fc2_b[i])
        if i in st.deepstack_visual_indexes:
            ds_idx = st.deepstack_visual_indexes.index(i)
            deepstack_features.append(st.deepstack_merger_list[ds_idx](hidden_states))

        if (hiprune_keep < 1.0
                and st.deepstack_visual_indexes
                and i == st.deepstack_visual_indexes[-1]):
            merge_unit = st.spatial_merge_size ** 2
            device = hidden_states.device
            num_images = grid_thw.shape[0]

            token_keep_list = []
            group_keep_list = []
            cumulative_keep_tokens = [0]
            group_offset = 0

            unit_range = torch.arange(merge_unit, device=device, dtype=torch.long)

            for img_idx in range(num_images):
                start = int(cu_seqlens[img_idx].item())
                end   = int(cu_seqlens[img_idx + 1].item())
                length = end - start
                if length % merge_unit != 0:

                    token_keep_list.append(torch.arange(
                        start, end, device=device, dtype=torch.long))
                    cumulative_keep_tokens.append(
                        cumulative_keep_tokens[-1] + length)
                    g = length // merge_unit
                    group_keep_list.append(torch.arange(
                        group_offset, group_offset + g,
                        device=device, dtype=torch.long))
                    group_offset += g
                    continue

                G = length // merge_unit
                k_target = int(round(G * hiprune_keep))
                K_groups = max(1, k_target)
                if K_groups >= G:
                    token_keep_list.append(torch.arange(
                        start, end, device=device, dtype=torch.long))
                    cumulative_keep_tokens.append(
                        cumulative_keep_tokens[-1] + length)
                    group_keep_list.append(torch.arange(
                        group_offset, group_offset + G,
                        device=device, dtype=torch.long))
                    group_offset += G
                    continue

                grp_view = hidden_states[start:end].view(G, merge_unit, -1)
                grp_scores = grp_view.float().norm(dim=-1).mean(dim=-1)
                top_groups = torch.topk(grp_scores, K_groups, largest=True).indices
                top_groups, _ = torch.sort(top_groups)

                tok_idx = (top_groups.unsqueeze(-1) * merge_unit
                           + unit_range.unsqueeze(0)).flatten() + start
                token_keep_list.append(tok_idx)
                cumulative_keep_tokens.append(
                    cumulative_keep_tokens[-1] + K_groups * merge_unit)

                group_keep_list.append(top_groups + group_offset)
                group_offset += G

            keep_idx = torch.cat(token_keep_list)
            group_idx = torch.cat(group_keep_list)

            hidden_states = hidden_states.index_select(0, keep_idx).contiguous()
            cos = cos.index_select(0, keep_idx).contiguous()
            sin = sin.index_select(0, keep_idx).contiguous()

            deepstack_features = [
                d.index_select(0, group_idx).contiguous()
                for d in deepstack_features
            ]
            S = hidden_states.shape[0]
            cu_seqlens = torch.tensor(
                cumulative_keep_tokens, device=device, dtype=torch.int32,
            )
            max_seqlen = max(
                cumulative_keep_tokens[i + 1] - cumulative_keep_tokens[i]
                for i in range(len(cumulative_keep_tokens) - 1)
            ) if len(cumulative_keep_tokens) > 1 else S

        if (i + 1) % _profile_quarter == 0:
            profiler.mark(f"vit.blocks_{i + 1}")

    profiler.mark("vit.blocks_done")
    merged = st.merger(hidden_states)
    profiler.mark("vit.merger")
    profiler.flush()

    return merged, deepstack_features

def install_lean_vision(model):
    """Monkey-patch visual.forward to use lean vision."""
    visual = model.model.visual

    if not hasattr(visual, 'fast_pos_embed_interpolate'):
        raise RuntimeError("visual.fast_pos_embed_interpolate not found, skipping lean vision")

    st = LeanVisionState(visual)

    class _VisualOutput:
        """Return value of visual.forward — supports:
          - tuple unpacking      (HF: `image_embeds, ds = self.visual(...)`)
          - attribute getattr    (`.pooler_output`, `.deepstack_features`, ...)
          - attribute SETATTR    (HF 5.8: `vision_output.pooler_output = ...`)
          - indexing             (`out[0]`, `out[1]`)
        """
        __slots__ = ("pooler_output", "deepstack_features",
                     "image_embeds", "last_hidden_state")

        def __init__(self, image_embeds, deepstack_features):
            self.pooler_output     = image_embeds
            self.image_embeds      = image_embeds
            self.last_hidden_state = image_embeds
            self.deepstack_features = deepstack_features

        def __iter__(self):
            yield self.image_embeds
            yield self.deepstack_features

        def __getitem__(self, i):
            if i == 0: return self.image_embeds
            if i == 1: return self.deepstack_features
            raise IndexError(i)

        def __len__(self):
            return 2

    def _lean_visual_fwd(hidden_states, grid_thw, **kwargs):
        merged, ds_features = lean_vision_forward(st, hidden_states, grid_thw)
        return _VisualOutput(merged, ds_features if ds_features else None)

    visual.forward = _lean_visual_fwd
    print("[lean_vision] Lean vision encoder installed")
    return st
