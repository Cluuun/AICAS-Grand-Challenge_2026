"""Decoder 层优化：融合残差加 + RMS norm。"""

import types
import logging
import os

import torch

from .cuda_ops import (
    _get_cuda_ops, _get_gemv_ops, _get_fused_gemv_ops,
    _fused_add_rms_norm_cuda, _int8_gemv,
)

logger = logging.getLogger(__name__)


def _make_decoder_layer_forward():
    # 融合残差加和post-attention rms norm + kernel fusion
    cuda_ops = _get_cuda_ops()
    gemv_ops = _get_gemv_ops()
    fused_ops = _get_fused_gemv_ops()

    if cuda_ops is not None:
        def patched_forward(self, hidden_states, position_embeddings, attention_mask=None,
                            position_ids=None, past_key_values=None, use_cache=False,
                            skip_mlp=False, **kwargs):
            residual = hidden_states
            is_decode = hidden_states.shape[1] == 1

            # Decode + 融合RMS norm可用时跳过input_layernorm（由QKV GEMV内部做）
            skip_norm = (is_decode and
                         hasattr(self.self_attn, '_skip_input_norm') and
                         self.self_attn._skip_input_norm)
            if not skip_norm:
                hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Skip MLP: keep attention context mixing, only skip the heavy MLP weights
            if is_decode and skip_mlp:
                _adapter = getattr(self.mlp, '_skip_adapter', None)
                _scale = getattr(self, '_mlp_skip_scale', 1.0)
                _is_default_scale = isinstance(_scale, (int, float)) and _scale == 1.0
                if not _is_default_scale:
                    scaled_attn = hidden_states * _scale
                else:
                    scaled_attn = hidden_states
                if _adapter is not None:
                    # SwiGLU adapter path: use fused RMS norm instead of manual PyTorch ops
                    normed, combined = _fused_add_rms_norm_cuda(
                        cuda_ops, scaled_attn, residual,
                        self.post_attention_layernorm.weight,
                        self.post_attention_layernorm.variance_epsilon,
                    )
                    adapter_out = _adapter(normed)
                    out = combined + adapter_out
                else:
                    out = residual + scaled_attn
                    # Per-layer bias compensation: E[MLP_out]
                    _bias = getattr(self, '_mlp_skip_bias', None)
                    if _bias is not None:
                        out = out + _bias
                    # Low-rank linear compensation: U*(V*LN(combined)) approximates MLP
                    _lr = getattr(self, '_lr_comp', None)
                    if _lr is not None:
                        _lr_U, _lr_V, _lr_bias = _lr
                        # Use fused RMS norm for the compensation path too
                        normed, _ = _fused_add_rms_norm_cuda(
                            cuda_ops, hidden_states, residual,
                            self.post_attention_layernorm.weight,
                            self.post_attention_layernorm.variance_epsilon,
                        )
                        correction = _lr_U @ (_lr_V @ normed.reshape(-1)) + _lr_bias
                        out = out + correction.reshape_as(out)
                _perturb_eps = getattr(self, '_skip_mlp_perturb_eps', 0.0)
                if _perturb_eps > 0:
                    _s = out.sign()
                    _nrm = out.norm() / (out.shape[-1] ** 0.5)
                    out = out + _perturb_eps * _nrm * _s
                return out

            # Decode: Low-rank MLP replacement (2 kernels, 4MB weight vs 18MB)
            if is_decode and hasattr(self.mlp, '_lr_W_int8'):
                hidden_states, residual = _fused_add_rms_norm_cuda(
                    cuda_ops, hidden_states, residual,
                    self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.variance_epsilon)
                _int8_gemv(gemv_ops, 'gemv_add_both_int8',
                    self.mlp._lr_W_int8, hidden_states.reshape(-1), self.mlp._lr_W_scale,
                    residual.reshape(-1), hidden_states.reshape(-1))
                return residual
            # Decode: INT4 fused path. If only gate_up is INT4, keep down in INT8.
            elif is_decode and fused_ops is not None and gemv_ops is not None and hasattr(self.mlp.gate_up_proj, 'weight_int4'):
                o_flat = hidden_states.reshape(-1)
                mid = fused_ops.gemv_addrmsnorm_silu_mul_int4(
                    self.mlp.gate_up_proj.weight_int4,
                    residual.reshape(-1), o_flat,
                    self.post_attention_layernorm.weight,
                    self.mlp.gate_up_proj.weight_scale_int4,
                    self.post_attention_layernorm.variance_epsilon,
                )
                if hasattr(self.mlp.down_proj, 'weight_int4'):
                    gemv_ops.gemv_add_both_int4(
                        self.mlp.down_proj.weight_int4, mid,
                        self.mlp.down_proj.weight_scale_int4,
                        residual.reshape(-1),
                        o_flat,
                        self.mlp.down_proj.weight_scale_int4.shape[1])
                elif hasattr(self.mlp.down_proj, 'weight_int8'):
                    _int8_gemv(gemv_ops, 'gemv_add_both_int8',
                        self.mlp.down_proj.weight_int8, mid,
                        self.mlp.down_proj.weight_scale,
                        residual.reshape(-1),
                        o_flat)
                else:
                    residual.add_(hidden_states)
                    gemv_ops.gemv_add(
                        self.mlp.down_proj.weight, mid,
                        residual.reshape(-1))
            # Decode: INT8/FP16 path
            elif is_decode and gemv_ops is not None:
                if (os.environ.get("AICAS_DISABLE_FUSED_INT8_MLP_NORM", "0") != "1" and
                        fused_ops is not None and
                        hasattr(fused_ops, 'gemv_addrmsnorm_silu_mul_int8') and
                        hasattr(gemv_ops, 'gemv_add_both_int8') and
                        hasattr(self.mlp.gate_up_proj, 'weight_int8') and
                        hasattr(self.mlp.down_proj, 'weight_int8')):
                    o_flat = hidden_states.reshape(-1)
                    use_w8a8_gate = (
                        os.environ.get("AICAS_W8A8_GATE_UP", "0") == "1" and
                        hasattr(fused_ops, 'gemv_addrmsnorm_silu_mul_w8a8')
                    )
                    gate_up_fn = (fused_ops.gemv_addrmsnorm_silu_mul_w8a8
                                  if use_w8a8_gate
                                  else fused_ops.gemv_addrmsnorm_silu_mul_int8)
                    mid = gate_up_fn(
                        self.mlp.gate_up_proj.weight_int8,
                        residual.reshape(-1), o_flat,
                        self.post_attention_layernorm.weight,
                        self.mlp.gate_up_proj.weight_scale,
                        self.post_attention_layernorm.variance_epsilon,
                    )
                    _int8_gemv(gemv_ops, 'gemv_add_both_int8',
                        self.mlp.down_proj.weight_int8, mid,
                        self.mlp.down_proj.weight_scale,
                        residual.reshape(-1),
                        o_flat)
                else:
                    hidden_states, residual = _fused_add_rms_norm_cuda(
                        cuda_ops,
                        hidden_states, residual,
                        self.post_attention_layernorm.weight,
                        self.post_attention_layernorm.variance_epsilon
                    )
                    x_flat = hidden_states.reshape(-1)
                    if hasattr(self.mlp.gate_up_proj, 'weight_int4'):
                        mid = gemv_ops.gemv_silu_mul_int4(
                            self.mlp.gate_up_proj.weight_int4, x_flat,
                            self.mlp.gate_up_proj.weight_scale_int4,
                            self.mlp.gate_up_proj.weight_scale_int4.shape[1])
                    elif hasattr(self.mlp.gate_up_proj, 'weight_int8'):
                        mid = _int8_gemv(gemv_ops, 'gemv_silu_mul_int8',
                            self.mlp.gate_up_proj.weight_int8, x_flat, self.mlp.gate_up_proj.weight_scale)
                    else:
                        mid = gemv_ops.gemv_silu_mul(self.mlp.gate_up_proj.weight, x_flat)

                    if hasattr(self.mlp.down_proj, 'weight_int4'):
                        gemv_ops.gemv_add_int4(
                            self.mlp.down_proj.weight_int4, mid,
                            self.mlp.down_proj.weight_scale_int4,
                            residual.reshape(-1),
                            self.mlp.down_proj.weight_scale_int4.shape[1])
                    elif hasattr(self.mlp.down_proj, 'weight_int8'):
                        _int8_gemv(gemv_ops, 'gemv_add_int8',
                            self.mlp.down_proj.weight_int8, mid, self.mlp.down_proj.weight_scale, residual.reshape(-1))
                    else:
                        gemv_ops.gemv_add(self.mlp.down_proj.weight, mid, residual.reshape(-1))
            else:
                hidden_states, residual = _fused_add_rms_norm_cuda(
                    cuda_ops,
                    hidden_states, residual,
                    self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.variance_epsilon
                )
                hidden_states = self.mlp(hidden_states)
                residual.add_(hidden_states)

            return residual
    else:
        def patched_forward(self, hidden_states, position_embeddings, attention_mask=None,
                            position_ids=None, past_key_values=None, use_cache=False, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # triton后备方案
            from custom_kernels.cuda.triton_kernels import triton_fused_add_rms_norm
            hidden_states, residual = triton_fused_add_rms_norm(
                hidden_states, residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon
            )

            # MLP + 原地残差加
            hidden_states = self.mlp(hidden_states)
            residual.add_(hidden_states)

            return residual

    return patched_forward


def patch_decoder_layers(model):
    # 给decoder层打补丁 用融合残差加+norm
    patched = 0
    forward_fn = _make_decoder_layer_forward()

    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3VLTextDecoderLayer":
            # 给attention模块设input_layernorm信息，用于融合RMS norm + QKV GEMV
            if (os.environ.get("AICAS_ENABLE_FUSED_QKV_INPUT_NORM", "0") == "1" and
                    hasattr(module, 'self_attn') and hasattr(module.self_attn, '_skip_input_norm')):
                module.self_attn._input_norm_weight = module.input_layernorm.weight
                module.self_attn._input_norm_eps = module.input_layernorm.variance_epsilon
                module.self_attn._skip_input_norm = True
            module.forward = types.MethodType(forward_fn, module)
            patched += 1

    return patched
