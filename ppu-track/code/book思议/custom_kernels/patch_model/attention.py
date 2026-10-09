"""Attention 优化：QKV 融合 + Flash 解码。"""

import types
import logging
import os

import torch
import torch.nn as nn

from .cuda_ops import _get_gemv_ops, _get_fused_gemv_ops, _get_flash_decode_ops, _int8_gemv

logger = logging.getLogger(__name__)


# 这里用torch.compiler.disable包一下 不然28个layer_idx会导致重编译28次
@torch.compiler.disable
def _cache_update(cache, key_states, value_states, layer_idx):
    return cache.layers[layer_idx].update(key_states, value_states)


def _make_attn_forward(q_dim, kv_dim):
    # 融合QKV + flash解码的attention forward
    # decode时用INT8 GEMV省50%权重带宽
    gemv_ops = _get_gemv_ops()
    fused_ops = _get_fused_gemv_ops()

    def patched_forward(self, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]

        is_decode = hidden_states.shape[1] == 1

        # QKV投影: decode用INT4/INT8 GEMV, prefill用标准FP16
        if is_decode and gemv_ops is not None:
            x_flat = hidden_states.reshape(-1)
            # 融合RMS norm + QKV INT4 GEMV (decode时input_layernorm被跳过)
            if (hasattr(self.qkv_proj, 'weight_int4') and
                    fused_ops is not None and
                    hasattr(self, '_skip_input_norm') and self._skip_input_norm and
                    self._input_norm_weight is not None):
                qkv_flat = fused_ops.gemv_rmsnorm_int4(
                    self.qkv_proj.weight_int4, x_flat,
                    self._input_norm_weight,
                    self.qkv_proj.weight_scale_int4,
                    self._input_norm_eps)
            elif (hasattr(self.qkv_proj, 'weight_int8') and
                  fused_ops is not None and
                  hasattr(fused_ops, 'gemv_rmsnorm_int8') and
                  hasattr(self, '_skip_input_norm') and self._skip_input_norm and
                  self._input_norm_weight is not None):
                qkv_flat = fused_ops.gemv_rmsnorm_int8(
                    self.qkv_proj.weight_int8, x_flat,
                    self._input_norm_weight,
                    self.qkv_proj.weight_scale,
                    self._input_norm_eps)
            elif hasattr(self.qkv_proj, 'weight_int4'):
                qkv_flat = gemv_ops.gemv_int4(
                    self.qkv_proj.weight_int4, x_flat,
                    self.qkv_proj.weight_scale_int4,
                    self.qkv_proj.weight_scale_int4.shape[1])
            elif hasattr(self.qkv_proj, 'weight_int8'):
                qkv_flat = _int8_gemv(gemv_ops, 'gemv_int8',
                    self.qkv_proj.weight_int8, x_flat, self.qkv_proj.weight_scale)
            else:
                qkv_flat = gemv_ops.gemv(self.qkv_proj.weight, x_flat)
            qkv = qkv_flat.reshape(1, 1, -1)
        else:
            if hasattr(self.qkv_proj, 'weight_int8_prefill') and hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0] > 16:
                from custom_kernels.quant.int8_prefill import int8_linear_prefill
                qkv = int8_linear_prefill(
                    hidden_states, self.qkv_proj.weight_int8_prefill,
                    self.qkv_proj.weight_scale_prefill,
                    self.qkv_proj.bias,
                )
            else:
                qkv = self.qkv_proj(hidden_states)

        cos, sin = position_embeddings

        has_flash_cache = (past_key_values is not None and
                           hasattr(past_key_values, 'flash_k_caches'))
        has_flashinfer = (past_key_values is not None and
                          hasattr(past_key_values, 'flashinfer_wrapper'))

        if is_decode:
            # decode: 融合 split + Q/K norm + RoPE
            gemv_ops_local = _get_gemv_ops()
            if gemv_ops_local is not None and hasattr(gemv_ops_local, 'fused_qkv_norm_rope'):
                qkv_flat = qkv.reshape(-1)
                cos_flat = cos.reshape(self.head_dim)
                sin_flat = sin.reshape(self.head_dim)
                gemv_ops_local.fused_qkv_norm_rope(
                    qkv_flat,
                    self.q_norm.weight, self.k_norm.weight,
                    cos_flat, sin_flat,
                    self._q_buf, self._k_buf, self._v_buf,
                    q_dim, kv_dim,
                    self.q_norm.variance_epsilon, self.k_norm.variance_epsilon,
                )
                n_q = q_dim // self.head_dim
                n_kv = kv_dim // self.head_dim
                query_states = self._q_buf.reshape(1, 1, n_q, self.head_dim)
                key_states = self._k_buf.reshape(1, 1, n_kv, self.head_dim)
                value_states = self._v_buf.reshape(1, 1, n_kv, self.head_dim)
            else:
                # Fallback to Triton
                from custom_kernels.cuda.triton_kernels import triton_fused_qkv_norm_rope
                q_bs, k_bs, v_bs = triton_fused_qkv_norm_rope(
                    qkv, self.q_norm.weight, self.k_norm.weight, cos, sin,
                    q_dim, kv_dim, self.head_dim,
                    self.q_norm.variance_epsilon, self.k_norm.variance_epsilon,
                )
                query_states = q_bs
                key_states = k_bs
                value_states = v_bs
        else:
            # prefill: 融合 split + Q/K norm + RoPE 全塞进一个核心
            from custom_kernels.cuda.triton_kernels import triton_fused_qkv_norm_rope_prefill
            q_bs, k_bs, v_bs = triton_fused_qkv_norm_rope_prefill(
                qkv, self.q_norm.weight, self.k_norm.weight, cos, sin,
                q_dim, kv_dim, self.head_dim,
                self.q_norm.variance_epsilon, self.k_norm.variance_epsilon,
            )
            query_states = q_bs
            key_states = k_bs
            value_states = v_bs

        if is_decode and has_flash_cache:
            # decode: flash_attn_with_kvcache 自己管cache更新
            layer_idx = self.layer_idx
            k_cache = past_key_values.flash_k_caches[layer_idx]
            v_cache = past_key_values.flash_v_caches[layer_idx]
            cache_seqlens = past_key_values.cache_seqlens

            gemv_ops_local = _get_gemv_ops()
            if has_flashinfer and gemv_ops_local is not None and hasattr(gemv_ops_local, "write_kv_cache"):
                gemv_ops_local.write_kv_cache(
                    k_cache[0].reshape(k_cache.shape[1], -1),
                    v_cache[0].reshape(v_cache.shape[1], -1),
                    key_states.reshape(-1),
                    value_states.reshape(-1),
                    past_key_values.flashinfer_cache_pos,
                )
                attn_output = past_key_values.flashinfer_wrapper.run(
                    query_states[:, 0],
                    (k_cache, v_cache),
                ).reshape(*input_shape, -1)
            elif (os.environ.get("AICAS_ENABLE_CUSTOM_DECODE_ATTN", "0") == "1" and
                    gemv_ops_local is not None and hasattr(gemv_ops_local, "decode_attn")):
                out = self._attn_out_buf.reshape(-1, self.head_dim)
                gemv_ops_local.decode_attn(
                    query_states.reshape(-1, self.head_dim),
                    key_states.reshape(-1, self.head_dim),
                    value_states.reshape(-1, self.head_dim),
                    k_cache[0],
                    v_cache[0],
                    out,
                    cache_seqlens,
                )
                attn_output = out.reshape(*input_shape, -1)
            elif (os.environ.get("AICAS_FLASH_DECODE_ATTN", "0") == "1" and
                    _get_flash_decode_ops() is not None):
                fd_ops = _get_flash_decode_ops()
                num_splits = int(os.environ.get("AICAS_FLASH_DECODE_NUM_SPLITS", "16"))
                self._fd_fused_oproj_done = False
                # Try zero-copy combine+O_proj fusion
                if (os.environ.get("AICAS_FLASH_DECODE_FUSED_OPROJ", "1") == "1" and
                        hasattr(fd_ops, 'flash_decode_attn_fused') and
                        hasattr(self.o_proj, 'weight_int8')):
                    out_oproj = self._attn_out_buf  # reuse as O_proj output [2048]
                    fd_ops.flash_decode_attn_fused(
                        query_states.reshape(-1, self.head_dim),
                        key_states.reshape(-1, self.head_dim),
                        value_states.reshape(-1, self.head_dim),
                        k_cache[0],
                        v_cache[0],
                        out_oproj,
                        cache_seqlens,
                        self._fd_ws_o,
                        self._fd_ws_l,
                        self._fd_ws_m,
                        self.o_proj.weight_int8,
                        self.o_proj.weight_scale,
                        self._fd_oproj_ws,
                        num_splits,
                    )
                    attn_output = out_oproj.reshape(*input_shape, -1)
                    self._fd_fused_oproj_done = True
                else:
                    out = self._attn_out_buf.reshape(-1, self.head_dim)
                    fd_ops.flash_decode_attn(
                        query_states.reshape(-1, self.head_dim),
                        key_states.reshape(-1, self.head_dim),
                        value_states.reshape(-1, self.head_dim),
                        k_cache[0],
                        v_cache[0],
                        out,
                        cache_seqlens,
                        self._fd_ws_o,
                        self._fd_ws_l,
                        self._fd_ws_m,
                        num_splits,
                    )
                    attn_output = out.reshape(*input_shape, -1)
            else:
                from flash_attn import flash_attn_with_kvcache

                attn_output = flash_attn_with_kvcache(
                    q=query_states,
                    k_cache=k_cache, v_cache=v_cache,
                    k=key_states,
                    v=value_states,
                    cache_seqlens=cache_seqlens,
                    num_splits=int(os.environ.get("AICAS_FLASH_ATTN_NUM_SPLITS", "0")),
                )
                attn_output = attn_output.reshape(*input_shape, -1)

        elif has_flash_cache:
            # T>1 batch decode with flash KV cache (speculative decoding verify).
            # flash_attn_with_kvcache writes new K/V directly to flash_k/v_caches,
            # keeping the cache consistent with the graph engine's flash_decode_attn.
            layer_idx = self.layer_idx
            k_cache = past_key_values.flash_k_caches[layer_idx]
            v_cache = past_key_values.flash_v_caches[layer_idx]
            cache_seqlens = past_key_values.cache_seqlens

            from flash_attn import flash_attn_with_kvcache
            # causal=True: query at local index i (position cache_seqlens+i) attends
            # to cache[0:cache_seqlens] + new tokens[0:i] only. This makes the batched
            # T>1 verify forward bit-equivalent to running T sequential decode steps,
            # which is required for speculative-decoding acceptance correctness.
            attn_output = flash_attn_with_kvcache(
                q=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                k=key_states,
                v=value_states,
                cache_seqlens=cache_seqlens,
                softmax_scale=self.scaling,
                causal=True,
            )
            attn_output = attn_output.reshape(*input_shape, -1)
        else:
            # prefill: 用flash_attn替换SDPA（更优的tiling，IO效率更高）
            if past_key_values is not None:
                # DynamicCache/GraphStaticKVCache store [batch, heads, seq, dim],
                # while flash_attn_func consumes [batch, seq, heads, dim].
                key_cache = key_states.transpose(1, 2).contiguous()
                value_cache = value_states.transpose(1, 2).contiguous()
                key_cache, value_cache = _cache_update(
                    past_key_values, key_cache, value_cache, self.layer_idx
                )
                key_states = key_cache.transpose(1, 2)
                value_states = value_cache.transpose(1, 2)

            from flash_attn import flash_attn_func
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True,
                softmax_scale=self.scaling,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # o_proj: decode用INT4/INT8 GEMV, prefill用标准FP16
        if getattr(self, '_fd_fused_oproj_done', False):
            # O_proj already computed inside flash_decode_attn_fused
            self._fd_fused_oproj_done = False
        elif is_decode and gemv_ops is not None:
            if hasattr(self.o_proj, 'weight_int4'):
                o_flat = gemv_ops.gemv_int4(
                    self.o_proj.weight_int4, attn_output.reshape(-1),
                    self.o_proj.weight_scale_int4,
                    self.o_proj.weight_scale_int4.shape[1])
            elif hasattr(self.o_proj, 'weight_int8'):
                o_flat = _int8_gemv(gemv_ops, 'gemv_int8',
                    self.o_proj.weight_int8, attn_output.reshape(-1), self.o_proj.weight_scale)
            else:
                o_flat = gemv_ops.gemv(self.o_proj.weight, attn_output.reshape(-1))
            attn_output = o_flat.reshape(*input_shape, -1)
        else:
            if hasattr(self.o_proj, 'weight_int8_prefill') and attn_output.reshape(-1, attn_output.shape[-1]).shape[0] > 16:
                from custom_kernels.quant.int8_prefill import int8_linear_prefill
                attn_output = int8_linear_prefill(
                    attn_output, self.o_proj.weight_int8_prefill,
                    self.o_proj.weight_scale_prefill,
                    self.o_proj.bias,
                )
            else:
                attn_output = self.o_proj(attn_output)
        return attn_output, None

    return patched_forward


def patch_qkv(model):
    # 把Q/K/V三个投影拼成一个
    patched = 0

    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3VLTextAttention":
            q_w = module.q_proj.weight
            k_w = module.k_proj.weight
            v_w = module.v_proj.weight

            q_dim = q_w.shape[0]
            kv_dim = k_w.shape[0]

            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)

            has_bias = module.q_proj.bias is not None
            qkv_proj = nn.Linear(
                q_w.shape[1], qkv_w.shape[0],
                bias=has_bias,
            )
            qkv_proj.weight = nn.Parameter(qkv_w)
            if has_bias:
                qkv_proj.bias = nn.Parameter(
                    torch.cat([module.q_proj.bias, module.k_proj.bias, module.v_proj.bias], dim=0)
                )
            qkv_proj = qkv_proj.to(device=q_w.device, dtype=q_w.dtype)

            module.qkv_proj = qkv_proj
            del module.q_proj
            del module.k_proj
            del module.v_proj

            # 预分配QKV输出buffer 避免forward中torch.empty
            head_dim = module.head_dim
            n_q_heads = q_dim // head_dim
            n_kv_heads = kv_dim // head_dim
            module._q_buf = torch.empty(n_q_heads, head_dim, dtype=torch.float16, device=q_w.device)
            module._k_buf = torch.empty(n_kv_heads, head_dim, dtype=torch.float16, device=q_w.device)
            module._v_buf = torch.empty(n_kv_heads, head_dim, dtype=torch.float16, device=q_w.device)
            module._attn_out_buf = torch.empty(n_q_heads * head_dim, dtype=torch.float16, device=q_w.device)
            # Flash-decode split-K workspace (pre-allocated for CUDA graph compatibility)
            max_splits = int(os.environ.get("AICAS_FLASH_DECODE_NUM_SPLITS", "16"))
            module._fd_ws_o = torch.empty(n_kv_heads, max_splits, n_q_heads // n_kv_heads, head_dim,
                                          dtype=torch.float32, device=q_w.device)
            module._fd_ws_l = torch.empty(n_kv_heads, max_splits, n_q_heads // n_kv_heads,
                                          dtype=torch.float32, device=q_w.device)
            module._fd_ws_m = torch.empty(n_kv_heads, max_splits, n_q_heads // n_kv_heads,
                                          dtype=torch.float32, device=q_w.device)
            # Zero-copy combine+O_proj fusion workspace
            hidden_dim = n_q_heads * head_dim  # 2048
            module._fd_oproj_ws = torch.empty(n_q_heads, hidden_dim,
                                              dtype=torch.float32, device=q_w.device)
            module._fd_fused_oproj_done = False
            # 用于融合RMS norm + QKV GEMV
            module._skip_input_norm = False
            module._input_norm_weight = None
            module._input_norm_eps = 0.0

            forward_fn = _make_attn_forward(q_dim, kv_dim)
            module.forward = types.MethodType(forward_fn, module)
            patched += 1

    return patched
