"""融合QKV投影，把三个矩阵乘合成一个。"""

import torch
import torch.nn.functional as F


def _make_fused_attn_forward(fused_weight, split_sizes, head_dim, num_heads, num_kv_heads,
                              layer_idx, config, o_proj, q_norm, k_norm, scaling, attn_impl):
    """生成融合后的attention forward函数。"""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def forward(self_ref, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        # 融合QKV投影，一次矩阵乘搞定
        qkv = F.linear(hidden_states, fused_weight)
        q, k, v = qkv.split(split_sizes, dim=-1)

        # 每个head单独归一化（q_norm/k_norm已经patch成Triton版本了）
        query_states = q_norm(q.view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k.view(hidden_shape)).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        # 旋转位置编码
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 更新KV cache
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, layer_idx
            )

        # 走SDPA / Flash Attention / Eager
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_impl, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self_ref,
            query_states, key_states, value_states,
            attention_mask,
            dropout=0.0,
            scaling=scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def patch_qkv(model: torch.nn.Module):
    """把所有Qwen3VLTextAttention模块patch成融合QKV版本。"""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLTextAttention):
            # 把q/k/v三个权重拼成一个: [2048+1024+1024, 2048] = [4096, 2048]
            fused_weight = torch.cat(
                [module.q_proj.weight, module.k_proj.weight, module.v_proj.weight],
                dim=0
            ).contiguous()

            q_size = module.q_proj.weight.shape[0]
            k_size = module.k_proj.weight.shape[0]
            v_size = module.v_proj.weight.shape[0]
            split_sizes = [q_size, k_size, v_size]

            fwd = _make_fused_attn_forward(
                fused_weight=fused_weight,
                split_sizes=split_sizes,
                head_dim=module.head_dim,
                num_heads=module.config.num_attention_heads,
                num_kv_heads=module.config.num_key_value_heads,
                layer_idx=module.layer_idx,
                config=module.config,
                o_proj=module.o_proj,
                q_norm=module.q_norm,
                k_norm=module.k_norm,
                scaling=module.scaling,
                attn_impl=module.config._attn_implementation,
            )

            # 绑定为方法
            import types
            module.forward = types.MethodType(fwd, module)
            count += 1

    return count
