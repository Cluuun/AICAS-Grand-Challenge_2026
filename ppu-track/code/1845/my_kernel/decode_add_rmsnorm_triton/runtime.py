from __future__ import annotations

import os
from types import MethodType

import torch
import triton
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask

from .kernel import (
    add_rmsnorm_rowwise_finalize_kernel,
    add_rmsnorm_rowwise_kernel,
    add_rmsnorm_rowwise_reduce_kernel,
    add_rmsnorm_rowwise_sum_kernel,
)

_DECODE_ADD_RMSNORM_TRITON_ENABLED = True
_TRACE_ONCE: set[str] = set()

try:
    import torch._dynamo as _torch_dynamo
except Exception:
    _torch_dynamo = None


def _is_dynamo_compiling() -> bool:
    if _torch_dynamo is None:
        return False
    try:
        return bool(_torch_dynamo.is_compiling())
    except Exception:
        return False


def set_decode_add_rmsnorm_triton_enabled(enabled: bool) -> None:
    global _DECODE_ADD_RMSNORM_TRITON_ENABLED
    _DECODE_ADD_RMSNORM_TRITON_ENABLED = bool(enabled)


def _trace_once(key: str, message: str) -> None:
    if os.getenv("AICAS_TIMING_PROFILE", "0") != "1" or key in _TRACE_ONCE:
        return
    _TRACE_ONCE.add(key)
    print(message)


def _seq_len_allowed(seq_len: int) -> bool:
    max_seq = 4
    if os.getenv("AICAS_SPEC_DECODE_ADD_RMSNORM_TRITON", "1") == "1":
        max_seq = max(max_seq, int(os.getenv("AICAS_SPEC_KERNEL_MAX_Q", "16")))
    return 1 <= seq_len <= max_seq


def _can_use(hidden_states: torch.Tensor, attn_out: torch.Tensor) -> bool:
    seq_len = int(hidden_states.shape[1]) if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3 else 0
    return (
        _DECODE_ADD_RMSNORM_TRITON_ENABLED
        and isinstance(hidden_states, torch.Tensor)
        and isinstance(attn_out, torch.Tensor)
        and hidden_states.shape == attn_out.shape
        and hidden_states.is_cuda
        and attn_out.is_cuda
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and _seq_len_allowed(seq_len)
        and hidden_states.is_contiguous()
        and attn_out.is_contiguous()
        and hidden_states.dtype == attn_out.dtype
        and hidden_states.dtype is torch.bfloat16
    )


def _fused_add_rmsnorm(
    residual: torch.Tensor,
    attn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    added_out: torch.Tensor | None,
    norm_out: torch.Tensor | None,
    partials_out: torch.Tensor | None,
    sumsq_out: torch.Tensor | None,
):
    a2d = residual.view(-1, residual.shape[-1])
    b2d = attn_out.view(-1, attn_out.shape[-1])
    m, n = a2d.shape

    if (
        added_out is None
        or added_out.device != a2d.device
        or added_out.dtype != a2d.dtype
        or added_out.shape != a2d.shape
    ):
        added_out = torch.empty_like(a2d)

    if (
        norm_out is None
        or norm_out.device != a2d.device
        or norm_out.dtype != a2d.dtype
        or norm_out.shape != a2d.shape
    ):
        norm_out = torch.empty_like(a2d)

    if n <= 4096:
        add_rmsnorm_rowwise_kernel[(m,)](
            a2d,
            b2d,
            norm_weight,
            added_out,
            norm_out,
            a2d.stride(0),
            n,
            eps,
        )
        return added_out.view_as(residual), norm_out.view_as(residual), added_out, norm_out, partials_out, sumsq_out

    block_size = 4096
    num_tiles = triton.cdiv(n, block_size)
    if (
        partials_out is None
        or partials_out.device != a2d.device
        or partials_out.dtype != torch.float32
        or partials_out.shape != (m, num_tiles)
    ):
        partials_out = torch.empty((m, num_tiles), device=a2d.device, dtype=torch.float32)
    if (
        sumsq_out is None
        or sumsq_out.device != a2d.device
        or sumsq_out.dtype != torch.float32
        or sumsq_out.shape != (m,)
    ):
        sumsq_out = torch.empty((m,), device=a2d.device, dtype=torch.float32)
    add_rmsnorm_rowwise_sum_kernel[(m, num_tiles)](
        a2d,
        b2d,
        added_out,
        partials_out,
        a2d.stride(0),
        n,
        partials_out.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=3,
    )
    add_rmsnorm_rowwise_reduce_kernel[(m,)](
        partials_out,
        sumsq_out,
        num_tiles,
        num_warps=1,
        num_stages=1,
    )
    add_rmsnorm_rowwise_finalize_kernel[(m, num_tiles)](
        added_out,
        norm_weight,
        norm_out,
        sumsq_out,
        a2d.stride(0),
        n,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=3,
    )
    return added_out.view_as(residual), norm_out.view_as(residual), added_out, norm_out, partials_out, sumsq_out


def _workspace_tuple(owner, key):
    workspaces = getattr(owner, "_aicas_decode_add_rmsnorm_workspaces", None)
    if not isinstance(workspaces, dict):
        workspaces = {}
        owner._aicas_decode_add_rmsnorm_workspaces = workspaces
    return workspaces, workspaces.get(key, (None, None, None, None))


def _try_fused_add_rmsnorm_for_module(
    owner,
    residual: torch.Tensor,
    addend: torch.Tensor,
    norm_module,
):
    if not _can_use(residual, addend):
        return None
    if norm_module is None or not hasattr(norm_module, "weight") or not hasattr(norm_module, "variance_epsilon"):
        return None

    key = (
        residual.device,
        residual.dtype,
        tuple(residual.shape),
        id(norm_module),
    )
    workspaces, (ws_add, ws_norm, ws_partial, ws_sum) = _workspace_tuple(owner, key)
    fused_residual, normed, ws_add, ws_norm, ws_partial, ws_sum = _fused_add_rmsnorm(
        residual,
        addend,
        norm_module.weight,
        float(norm_module.variance_epsilon),
        ws_add,
        ws_norm,
        ws_partial,
        ws_sum,
    )
    workspaces[key] = (ws_add, ws_norm, ws_partial, ws_sum)
    return fused_residual, normed


def _decode_model_fastpath_allowed(hidden_states: torch.Tensor, deepstack_visual_embeds) -> bool:
    seq_len = int(hidden_states.shape[1]) if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3 else 0
    return (
        _DECODE_ADD_RMSNORM_TRITON_ENABLED
        and not _is_dynamo_compiling()
        and deepstack_visual_embeds is None
        and isinstance(hidden_states, torch.Tensor)
        and hidden_states.is_cuda
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and _seq_len_allowed(seq_len)
        and hidden_states.dtype is torch.bfloat16
    )


def _needs_full_model_outputs(kwargs: dict) -> bool:
    return bool(
        kwargs.get("output_hidden_states", False)
        or kwargs.get("output_attentions", False)
        or kwargs.get("target_verify_layer_indices", None) is not None
    )


def _patch_text_model_forward(model) -> bool:
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None or not hasattr(lm, "layers") or getattr(lm, "_aicas_decode_add_rmsnorm_model_fused", False):
        return False

    original_forward = lm.forward

    def _forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if _needs_full_model_outputs(kwargs) or not _decode_model_fastpath_allowed(inputs_embeds, deepstack_visual_embeds):
            return original_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        pre_norm = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if pre_norm is None:
                normed = decoder_layer.input_layernorm(hidden_states)
            else:
                normed = pre_norm

            residual = hidden_states
            attn_out, _ = decoder_layer.self_attn(
                hidden_states=normed,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            fused = _try_fused_add_rmsnorm_for_module(
                decoder_layer,
                residual,
                attn_out,
                decoder_layer.post_attention_layernorm,
            )
            if fused is None:
                hidden_states = residual.add_(attn_out)
                post_norm = decoder_layer.post_attention_layernorm(hidden_states)
            else:
                hidden_states, post_norm = fused

            mlp_out = decoder_layer.mlp(post_norm)
            is_last = layer_idx == len(self.layers) - 1
            next_norm_module = self.norm if is_last else self.layers[layer_idx + 1].input_layernorm
            fused = _try_fused_add_rmsnorm_for_module(
                decoder_layer,
                hidden_states,
                mlp_out,
                next_norm_module,
            )
            if fused is None:
                hidden_states = hidden_states.add_(mlp_out)
                pre_norm = None
            else:
                hidden_states, pre_norm = fused

        if pre_norm is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = pre_norm

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    lm.forward = MethodType(_forward, lm)
    lm._aicas_decode_add_rmsnorm_model_fused = True
    lm._aicas_decode_add_rmsnorm_original_forward = original_forward
    return True


def apply_decode_add_rmsnorm_triton(model) -> int:
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None or not hasattr(lm, "layers"):
        return 0

    patched = 0
    for layer in lm.layers:
        if getattr(layer, "_aicas_decode_add_rmsnorm_triton", False):
            continue
        original_forward = layer.forward

        def _make_forward(orig):
            def _forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=None,
                cache_position=None,
                **kwargs,
            ):
                if _is_dynamo_compiling():
                    return orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )
                seq_len = int(hidden_states.shape[1]) if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3 else 0
                decode_only = (
                    _DECODE_ADD_RMSNORM_TRITON_ENABLED
                    and isinstance(hidden_states, torch.Tensor)
                    and hidden_states.is_cuda
                    and hidden_states.ndim == 3
                    and hidden_states.shape[0] == 1
                    and _seq_len_allowed(seq_len)
                    and hidden_states.dtype is torch.bfloat16
                )
                if not decode_only:
                    return orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )

                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                attn_out, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                if _can_use(residual, attn_out):
                    try:
                        workspace_key = (residual.device, residual.dtype, tuple(residual.shape))
                        workspaces = getattr(self, "_aicas_decode_add_rmsnorm_workspaces", None)
                        if not isinstance(workspaces, dict):
                            workspaces = {}
                            self._aicas_decode_add_rmsnorm_workspaces = workspaces
                        ws_add, ws_norm, ws_partial, ws_sum = workspaces.get(workspace_key, (None, None, None, None))
                        fused_residual, normed, ws_add, ws_norm, ws_partial, ws_sum = _fused_add_rmsnorm(
                            residual,
                            attn_out,
                            self.post_attention_layernorm.weight,
                            float(self.post_attention_layernorm.variance_epsilon),
                            ws_add,
                            ws_norm,
                            ws_partial,
                            ws_sum,
                        )
                        workspaces[workspace_key] = (ws_add, ws_norm, ws_partial, ws_sum)
                        residual = fused_residual
                        hidden_states = normed
                    except Exception:
                        hidden_states = residual.add_(attn_out)
                        residual = hidden_states
                        hidden_states = self.post_attention_layernorm(hidden_states)
                else:
                    hidden_states = residual.add_(attn_out)
                    residual = hidden_states
                    hidden_states = self.post_attention_layernorm(hidden_states)

                hidden_states = self.mlp(hidden_states)
                hidden_states = residual.add_(hidden_states)
                return hidden_states

            return _forward

        layer.forward = MethodType(_make_forward(original_forward), layer)
        layer._aicas_decode_add_rmsnorm_triton = True
        patched += 1

    original_generate = model.generate

    def _generate_with_policy(*args, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", None)
        if max_new_tokens is None:
            enable = True
        else:
            try:
                token_count = int(max_new_tokens)
            except Exception:
                token_count = 0
            min_tokens = int(os.getenv("AICAS_DECODE_FASTPATH_MIN_NEW_TOKENS", "2"))
            max_tokens = int(os.getenv("AICAS_DECODE_FASTPATH_MAX_NEW_TOKENS", "256"))
            enable = min_tokens <= token_count <= max_tokens
        set_decode_add_rmsnorm_triton_enabled(enable)
        try:
            return original_generate(*args, **kwargs)
        finally:
            set_decode_add_rmsnorm_triton_enabled(True)

    model.generate = _generate_with_policy
    patched += int(_patch_text_model_forward(model))
    return patched
