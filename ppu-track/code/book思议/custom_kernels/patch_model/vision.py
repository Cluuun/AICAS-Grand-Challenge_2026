"""Vision 组件优化：RoPE 替换 + 位置编码缓存 + DeepStack。"""

import types
import logging

import torch

logger = logging.getLogger(__name__)

# 模块级缓存
_pos_embed_cache = {}
_rot_emb_cache = {}


def patch_vision_rope():
    # 把vision的apply_rotary_pos_emb_vision换成triton核心
    from custom_kernels.cuda.triton_kernels import triton_vision_rope

    import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_module
    qwen3_vl_module.apply_rotary_pos_emb_vision = triton_vision_rope
    return 1


def patch_vision_caches(model):
    # 缓存vision的位置编码和旋转编码 大部分图片尺寸都重复
    visual = model.model.visual

    _orig_pos_embed = visual.fast_pos_embed_interpolate
    _orig_rot_emb = visual.rot_pos_emb

    def _cached_pos_embed(grid_thw):
        key = str(grid_thw.tolist())
        if key not in _pos_embed_cache:
            _pos_embed_cache[key] = _orig_pos_embed(grid_thw)
        return _pos_embed_cache[key]

    def _cached_rot_emb(grid_thw):
        key = str(grid_thw.tolist())
        if key not in _rot_emb_cache:
            _rot_emb_cache[key] = _orig_rot_emb(grid_thw)
        return _rot_emb_cache[key]

    visual.fast_pos_embed_interpolate = _cached_pos_embed
    visual.rot_pos_emb = _cached_rot_emb
    return 1


def _deepstack_process_graph_safe(self, hidden_states, visual_pos_masks, visual_embeds):
    # 图安全的deepstack 用cumsum做索引映射+scatter
    # 全部操作都图安全 没有.item()、nonzero、masked_select
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

    num_visual = visual_embeds.shape[0]
    if num_visual == 0:
        return hidden_states

    # cumsum算出每个True位置的1-based索引
    # 必须用float32 cumsum 超过2048后float16精度不够 索引会越界
    mask_float = visual_pos_masks.float()
    cum = mask_float.cumsum(dim=-1)

    # 用索引取出visual_embeds
    idx = (cum - 1).clamp(min=0).long().squeeze(0)
    full_visual = visual_embeds[idx]

    # 非visual位置清零
    mask_fp16 = visual_pos_masks.to(dtype=hidden_states.dtype)
    full_visual = full_visual * mask_fp16.transpose(0, 1)

    # torch.where加回去
    mask = visual_pos_masks.unsqueeze(-1)
    result = torch.where(mask, hidden_states + full_visual.unsqueeze(0), hidden_states)
    return result


def patch_deepstack(model):
    # 换成图安全版本的deepstack
    model.model.language_model._deepstack_process = types.MethodType(
        _deepstack_process_graph_safe, model.model.language_model
    )
    return 1
