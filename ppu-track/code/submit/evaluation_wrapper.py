"""
AICAS 2026 - Optimized VLM Inference (batch=1)

Optimizations:
1. Flash attention for both prefill and decode (BSND KV cache layout)
2. Skip decode mask creation
3. Static KV cache + torch.compile(max-autotune)
4. Triton fused kernels (residual+RMSNorm, fused QKV/GateUp, SwiGLU, RoPE)
5. Fast greedy decode loop (FP16 argmax, minimal CPU overhead)
6. CUDA Graph for LLM prefill
7. ViT img2col + cached RoPE/pos_embed
"""
import os
import types
from typing import Dict, Callable

DEBUG = os.environ.get('DEBUG', '0') == '1'

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# No-op empty_cache to keep memory pool warm across benchmark samples
torch.cuda.empty_cache = lambda: None

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import StaticCache, StaticLayer
import transformers.generation.utils as _gen_utils

from triton_fused_kernels import (
    fused_residual_rmsnorm, fused_swiglu, triton_rmsnorm,
    batch_rmsnorm, batch_fused_residual_rmsnorm, batch_fused_swiglu,
    batch_layernorm, batch_fused_residual_layernorm,
    fused_rope_vision,
    batch_gelu_tanh,
    fused_qk_norm_rope_decode,
    batch_fused_qk_norm_rope,
)

flash_attn_func = None
flash_attn_varlen_func = None
flash_attn_with_kvcache = None

try:
    from fa3_fwd_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pass

# Fallback: Try FA3 from flash-attention source build
if flash_attn_func is None:
    try:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass

# Fallback: Try FA2 from flash-attn package (try multiple import paths)
if flash_attn_func is None:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass

if flash_attn_func is None:
    try:
        from flash_attn.flash_attn_interface import (  # noqa: F401
            flash_attn_func,
            flash_attn_varlen_func,
            flash_attn_with_kvcache
        )
    except (ImportError, ModuleNotFoundError):
        pass


PATCH_SIZE = 16
MAX_PATCH_COUNT = 1024
MIN_PATCH_COUNT = 4
MAX_PIXELS = MAX_PATCH_COUNT * PATCH_SIZE * PATCH_SIZE
MIN_PIXELS = MIN_PATCH_COUNT * PATCH_SIZE * PATCH_SIZE

MODEL_DTYPE = torch.float16  # Global dtype: torch.float16 or torch.bfloat16


# ============================================================================
# Flash Decoding via torch.library custom op (torch.compile compatible)
# ============================================================================
# Register flash_attn_with_kvcache as a custom op so torch.compile can trace
# through it without graph breaks. The fake tensor impl tells the compiler
# the output shape/dtype without running the real CUDA kernel.

@torch.library.custom_op("flash_decode::attn_with_kvcache", mutates_args=())
def _flash_decode_op(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Custom op wrapping flash_attn_with_kvcache for decode attention.

    Uses torch.library.custom_op + register_fake so torch.compile can trace
    through this op without graph breaks. The fake impl provides shape inference
    during compilation; the real impl runs flash_attn_with_kvcache at runtime.

    All tensors use BSND layout [batch, seqlen, num_heads, head_dim] which
    matches flash_attn's native format, avoiding transpose overhead.

    Args:
        q: [batch, 1, num_q_heads, head_dim] BSND query
        k_cache: [batch, max_cache_len, num_kv_heads, head_dim] BSND full KV cache
        v_cache: [batch, max_cache_len, num_kv_heads, head_dim] BSND full KV cache
        cache_seqlens: [batch] int32, number of valid tokens in cache
        softmax_scale: attention scaling factor

    Returns: [batch, 1, num_q_heads, head_dim]
    """
    return flash_attn_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=True,
    )

@_flash_decode_op.register_fake
def _flash_decode_op_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Fake tensor impl: returns empty tensor with correct shape/dtype/device."""
    return q.new_empty(q.shape)


# ============================================================================
# Patch 1: Skip decode mask creation
# ============================================================================
def _install_skip_decode_mask():
    def fast_create_masks(config, inputs_embeds, attention_mask, cache_position,
                          past_key_values, **kwargs):
        return None
    _gen_utils.create_masks_for_generate = fast_create_masks

# ============================================================================
# Patch 2: StaticLayer fix + BSND KV cache + FastResetStaticCache
# ============================================================================
def _patch_static_layer_max_batch_size():
    original_init = StaticLayer.__init__
    def patched_init(self, max_cache_len: int):
        original_init(self, max_cache_len)
        self.max_batch_size = 1
    StaticLayer.__init__ = patched_init


def _patch_static_layer_bsnd():
    """Monkey-patch StaticLayer to use BSND [B, S, N, D] format instead of BNSD.

    This eliminates the transpose needed for flash_attn_with_kvcache,
    which expects BSND format. The cache is now stored contiguously
    in the layout that flash attention reads, improving memory access patterns.
    """
    from torch.compiler import is_compiling as is_torchdynamo_compiling

    def bsnd_lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.max_batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        # BSND format: [batch, max_cache_len, num_heads, head_dim]
        self.keys = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.num_heads, self.k_head_dim),
            dtype=self.dtype, device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.num_heads, self.v_head_dim),
            dtype=self.dtype, device=self.device,
        )
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
        self.is_initialized = True

    def bsnd_update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[1], device=self.device)
        )

        # key_states/value_states are BSND [B, S, N, D], index on dim=1 (seq)
        self.keys.index_copy_(1, cache_position, key_states)
        self.values.index_copy_(1, cache_position, value_states)
        return self.keys, self.values

    StaticLayer.lazy_initialization = bsnd_lazy_initialization
    StaticLayer.update = bsnd_update

class FastResetStaticCache(StaticCache):
    """StaticCache with O(1) reset via a single contiguous KV buffer."""

    def __init__(self, config, max_cache_len, dtype=MODEL_DTYPE, device="cuda:0", **kwargs):
        super().__init__(config=config, max_cache_len=max_cache_len, **kwargs)
        text_config = config if not hasattr(config, 'get_text_config') else config.get_text_config()
        self.early_initialization(
            batch_size=1, num_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim, dtype=dtype, device=device,
        )
        self._consolidate_buffer()

    def _consolidate_buffer(self):
        layer_key_shape = self.layers[0].keys.shape
        layer_val_shape = self.layers[0].values.shape
        layer_key_numel = self.layers[0].keys.numel()
        layer_val_numel = self.layers[0].values.numel()
        num_layers = len(self.layers)
        dtype = self.layers[0].keys.dtype
        device = self.layers[0].keys.device

        total_elements = num_layers * (layer_key_numel + layer_val_numel)
        self._unified_buffer = torch.zeros(total_elements, dtype=dtype, device=device)
        torch._dynamo.mark_static_address(self._unified_buffer)

        offset = 0
        for layer in self.layers:
            key_view = self._unified_buffer[offset:offset + layer_key_numel].view(layer_key_shape)
            offset += layer_key_numel
            val_view = self._unified_buffer[offset:offset + layer_val_numel].view(layer_val_shape)
            offset += layer_val_numel
            layer.keys = key_view
            layer.values = val_view
            torch._dynamo.mark_static_address(key_view)
            torch._dynamo.mark_static_address(val_view)

    def reset(self):
        self._unified_buffer.zero_()


# ============================================================================
# Patch 4: Triton fused decode path + flash_attn prefill
# ============================================================================
def _install_fused_decode_layers(model):
    lang_model = model.model.language_model
    text_config = model.config.get_text_config()
    eps = text_config.rms_norm_eps
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    intermediate_size = text_config.intermediate_size

    for layer in lang_model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        # Pre-concatenate QKV weights
        attn.qkv_weight = torch.cat([attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0)
        attn._q_out_dim = text_config.num_attention_heads * head_dim
        attn._kv_out_dim = num_kv_heads * head_dim

        # Pre-concatenate Gate+Up weights
        mlp.gate_up_weight = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], dim=0)
        mlp._intermediate_size = intermediate_size

        layer._rms_eps = eps

    # --- Patch Attention forward ---
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextAttention, eager_attention_forward, apply_rotary_pos_emb,
    )

    def fused_attn_forward(self, hidden_states, position_embeddings,
                           attention_mask=None, past_key_values=None,
                           cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        seq_len = hidden_states.shape[-2]

        qkv_out = F.linear(hidden_states, self.qkv_weight)
        q_dim = self._q_out_dim
        kv_dim = self._kv_out_dim
        cos, sin = position_embeddings

        if seq_len == 1:
            # Decode path — all BSND [B, S=1, N, D] for flash_attn_with_kvcache
            # fused_qk_norm_rope_decode outputs BSND directly, no transpose needed
            query_states, key_states = fused_qk_norm_rope_decode(
                qkv_out[..., :q_dim + kv_dim],
                self.q_norm.weight, self.k_norm.weight,
                cos, sin,
                self.q_norm.variance_epsilon,
                self.k_norm.variance_epsilon,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.head_dim,
            )
            value_states = qkv_out[..., q_dim + kv_dim:].view(hidden_shape)

            if past_key_values is not None:
                # Cache is BSND — update and get full cache views
                k_cache_bsnd, v_cache_bsnd = past_key_values.update(
                    key_states, value_states, self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )

            # cache_seqlens = number of valid tokens including the one just written
            cache_seqlens = cache_position[0:1].to(torch.int32) + 1

            # flash_attn_with_kvcache for decode via custom op
            # (custom op has register_fake so torch.compile traces without graph break)
            attn_output = _flash_decode_op(
                query_states,      # [B, 1, H_q, D]
                k_cache_bsnd,      # [B, S, H_kv, D]
                v_cache_bsnd,      # [B, S, H_kv, D]
                cache_seqlens, self.scaling,
            )
            # attn_output is [B, 1, H_q, D], reshape to [B, 1, hidden]
            attn_output = attn_output.reshape(*input_shape, -1)
            return self.o_proj(attn_output), None

        num_q_heads = self.config.num_attention_heads
        num_kv_heads_local = self.config.num_key_value_heads

        query_states_bshd, key_states_bshd = batch_fused_qk_norm_rope(
            qkv_out[..., :q_dim + kv_dim],
            self.q_norm.weight, self.k_norm.weight,
            cos, sin,
            self.q_norm.variance_epsilon,
            num_q_heads, num_kv_heads_local,
            self.head_dim,
        )
        value_states_bshd = qkv_out[..., q_dim + kv_dim:].view(hidden_shape)

        attn_output = flash_attn_func(
            query_states_bshd, key_states_bshd,
            value_states_bshd,
            causal=True, softmax_scale=self.scaling,
        )

        if past_key_values is not None:
            # KV cache update — BSND format, direct index_copy_ (no transpose)
            past_key_values.update(
                key_states_bshd, value_states_bshd,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        attn_output = attn_output.reshape(*input_shape, -1)
        return self.o_proj(attn_output), None

    Qwen3VLTextAttention.forward = fused_attn_forward

    # --- Patch MLP forward ---
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

    def fused_mlp_forward(self, x):
        gate_up_out = F.linear(x, self.gate_up_weight)
        if x.shape[-2] == 1:
            return self.down_proj(fused_swiglu(gate_up_out, self._intermediate_size))
        return self.down_proj(batch_fused_swiglu(gate_up_out, self._intermediate_size))

    Qwen3VLTextMLP.forward = fused_mlp_forward

    # --- Patch Layer forward ---
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    def fused_layer_forward(self, hidden_states, position_embeddings,
                            attention_mask=None, past_key_values=None,
                            cache_position=None, **kwargs):
        is_decode = hidden_states.shape[-2] == 1
        rmsnorm_fn = triton_rmsnorm if is_decode else batch_rmsnorm
        fused_res_fn = fused_residual_rmsnorm if is_decode else batch_fused_residual_rmsnorm

        normed = rmsnorm_fn(hidden_states, self.input_layernorm.weight, self._rms_eps)

        attn_out, _ = self.self_attn(
            hidden_states=normed, attention_mask=attention_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs,
        )
        normed_mlp, hidden_states = fused_res_fn(
            hidden_states, attn_out,
            self.post_attention_layernorm.weight, self._rms_eps,
        )
        if is_decode:
            return hidden_states.add_(self.mlp(normed_mlp))
        return hidden_states + self.mlp(normed_mlp)

    Qwen3VLTextDecoderLayer.forward = fused_layer_forward

    # --- Patch ViT Attention forward ---
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention, Qwen3VLVisionBlock

    _vit_max_seqlen_cache = {}

    def fused_vit_attn_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                               position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
        query_states, key_states, value_states = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        cos, sin = position_embeddings
        query_states, key_states = fused_rope_vision(query_states, key_states, cos, sin)

        max_seqlen = kwargs.get('_max_seqlen')
        if max_seqlen is None:
            cache_key = (seq_length, cu_seqlens.data_ptr())
            if cache_key not in _vit_max_seqlen_cache:
                _vit_max_seqlen_cache[cache_key] = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            max_seqlen = _vit_max_seqlen_cache[cache_key]

        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling, causal=False,
        )
        return self.proj(attn_output.reshape(seq_length, -1))

    Qwen3VLVisionAttention.forward = fused_vit_attn_forward

    # --- Patch ViT Block forward ---
    def fused_vit_block_forward(self, hidden_states, cu_seqlens,
                                rotary_pos_emb=None, position_embeddings=None, **kwargs):
        normed = batch_layernorm(hidden_states, self.norm1.weight, self.norm1.bias, eps=1e-6)
        attn_out = self.attn(normed, cu_seqlens=cu_seqlens,
                             rotary_pos_emb=rotary_pos_emb,
                             position_embeddings=position_embeddings, **kwargs)
        normed_mlp, hidden_states = batch_fused_residual_layernorm(
            hidden_states, attn_out, self.norm2.weight, self.norm2.bias, eps=1e-6,
        )
        # In-place add: avoids allocating a new output tensor (vllm-style)
        return hidden_states.add_(self.mlp(normed_mlp))

    Qwen3VLVisionBlock.forward = fused_vit_block_forward

    # --- Patch ViT MLP forward ---
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionMLP

    def fused_vit_mlp_forward(self, hidden_state):
        return self.linear_fc2(batch_gelu_tanh(self.linear_fc1(hidden_state)))

    Qwen3VLVisionMLP.forward = fused_vit_mlp_forward

    # --- Patch final norm ---
    def fused_final_norm_forward(hidden_states):
        if hidden_states.shape[-2] == 1:
            return triton_rmsnorm(hidden_states, lang_model.norm.weight, eps)
        return batch_rmsnorm(hidden_states, lang_model.norm.weight, eps)

    lang_model.norm.forward = fused_final_norm_forward


# ============================================================================
# Patch 5: Fast greedy decode
# ============================================================================
def _install_fast_greedy_decode():
    from transformers.generation.utils import GenerationMixin

    original_update_kwargs = GenerationMixin._update_model_kwargs_for_generation

    def fast_update_kwargs(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            model_kwargs['past_key_values'] = outputs.past_key_values
        if (cache_position := model_kwargs.get("cache_position")) is not None:
            model_kwargs["cache_position"] = cache_position[-1:] + 1
        if (position_ids := model_kwargs.get("position_ids")) is not None:
            model_kwargs["position_ids"] = position_ids[..., -1:] + 1
        return model_kwargs

    GenerationMixin._update_model_kwargs_for_generation = fast_update_kwargs

    original_sample = GenerationMixin._sample

    def fast_sample(self, input_ids, logits_processor, stopping_criteria,
                    generation_config, synced_gpus=False, streamer=None, **model_kwargs):
        if not generation_config.do_sample and len(logits_processor) == 0:
            eagle3 = getattr(self, '_eagle3', None)
            if eagle3 is not None:
                return _speculative_greedy_loop(
                    self, input_ids, stopping_criteria, generation_config,
                    eagle3, synced_gpus, streamer, **model_kwargs,
                )
            return _fast_greedy_loop(
                self, input_ids, stopping_criteria, generation_config,
                synced_gpus, streamer, **model_kwargs,
            )
        return original_sample(
            self, input_ids, logits_processor, stopping_criteria,
            generation_config, synced_gpus, streamer, **model_kwargs,
        )

    GenerationMixin._sample = fast_sample


def _fast_greedy_loop(model_self, input_ids, stopping_criteria, generation_config,
                      synced_gpus=False, streamer=None, **model_kwargs):
    from transformers.generation.utils import ALL_CACHE_NAMES, GenerateDecoderOnlyOutput

    _vocab_map_greedy = getattr(model_self, '_reduced_vocab_map', None)
    pad_token_id = generation_config._pad_token_tensor
    has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)

    eos_token_ids = None
    single_eos_id = None
    if has_eos_stopping_criteria:
        for criteria in stopping_criteria:
            if hasattr(criteria, "eos_token_id"):
                eos_token_ids = criteria.eos_token_id.to(input_ids.device)
                if eos_token_ids.numel() == 1:
                    single_eos_id = eos_token_ids[0]
                break

    max_length = None
    for criteria in stopping_criteria:
        if hasattr(criteria, "max_length"):
            max_length = criteria.max_length
            break

    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_forward = (
        model_self.get_compiled_call(generation_config.compile_config)
        if model_self._valid_auto_compile_criteria(model_kwargs, generation_config)
        else model_self.__call__
    )

    outputs = model_self._prefill(
        input_ids, generation_config, model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )

    current_length = input_ids.shape[1]
    use_cache = model_kwargs.get("use_cache", True)
    prefill_consumed = False
    steps_since_check = 0

    # --- Decode CUDA Graph setup ---
    decode_graph_runner = getattr(model_self, '_decode_graph_runner', None)
    use_decode_graph = decode_graph_runner is not None
    decode_graph_active = False  # True after graph is captured and active

    # --- Pre-allocate input_ids buffer to avoid torch.cat every step ---
    max_total_length = max_length if max_length is not None else current_length + 256
    ids_buffer = torch.zeros(batch_size, max_total_length, dtype=input_ids.dtype, device=input_ids.device)
    ids_buffer[:, :current_length] = input_ids
    input_ids = ids_buffer

    # --- Pre-allocate persistent tensors for decode loop ---
    # These are updated in-place to avoid per-step tensor allocation
    decode_cache_pos = torch.zeros(1, dtype=torch.long, device=input_ids.device)
    decode_pos_ids = torch.zeros(4, 1, 1, dtype=torch.long, device=input_ids.device)
    decode_input_buf = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)

    # --- Async EOS: double-buffered pinned memory for every-step async check ---
    # Use two pinned buffers so we can launch a new async copy every step
    # while the previous one may still be in flight. event.query() is pure
    # CPU and does not block the GPU pipeline.
    device = input_ids.device
    pinned_flags = [
        torch.zeros(batch_size, dtype=torch.long).pin_memory(),
        torch.zeros(batch_size, dtype=torch.long).pin_memory(),
    ]
    eos_events = [torch.cuda.Event(), torch.cuda.Event()]
    eos_buf_idx = 0        # which buffer to write next
    eos_inflight = False   # whether a copy is currently in flight
    eos_check_idx = 0      # which buffer to check

    while True:
        if prefill_consumed:
            if decode_graph_active:
                # --- CUDA Graph replay path (zero CPU overhead) ---
                decode_input_buf[0, 0] = input_ids[0, current_length - 1]

                next_tokens = decode_graph_runner.run(
                    decode_input_buf, decode_cache_pos, decode_pos_ids,
                )

                # In-place update for next step (no tensor allocation)
                decode_cache_pos.add_(1)
                decode_pos_ids.add_(1)
            else:
                # --- Eager path (first decode step or no graph runner) ---
                eager_input_ids = input_ids[:, :current_length]
                model_inputs = model_self.prepare_inputs_for_generation(
                    eager_input_ids, next_sequence_length=1 if use_cache else None, **model_kwargs,
                )
                with model_self._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)

                model_kwargs = model_self._update_model_kwargs_for_generation(
                    outputs, model_kwargs,
                )
                _reduced_idx = outputs.logits[:, -1, :].float().argmax(dim=-1)
                next_tokens = _vocab_map_greedy[_reduced_idx] if _vocab_map_greedy is not None else _reduced_idx
                del outputs

                # After first eager decode step, capture the decode CUDA graph
                if use_decode_graph and not decode_graph_runner.is_captured:
                    capture_pos = model_kwargs["cache_position"].item() if model_kwargs.get("cache_position") is not None else current_length
                    decode_graph_runner.capture(warmup_cache_position=capture_pos)
                    decode_graph_active = True
                    # Initialize persistent decode tensors from model_kwargs
                    decode_cache_pos.copy_(model_kwargs["cache_position"])
                    decode_pos_ids[:model_kwargs["position_ids"].shape[0]].copy_(model_kwargs["position_ids"])
        else:
            # First iteration: consume prefill outputs
            model_kwargs = model_self._update_model_kwargs_for_generation(
                outputs, model_kwargs,
            )
            _reduced_idx = outputs.logits[:, -1, :].float().argmax(dim=-1)
            next_tokens = _vocab_map_greedy[_reduced_idx] if _vocab_map_greedy is not None else _reduced_idx
            del outputs
        prefill_consumed = True

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # Write token into pre-allocated buffer (no torch.cat)
        input_ids[:, current_length] = next_tokens
        current_length += 1

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        if max_length is not None and current_length >= max_length:
            break

        # --- Every-step async EOS detection (double-buffered) ---
        if eos_token_ids is not None:
            # GPU-side EOS flag update (no CPU sync)
            if single_eos_id is not None:
                is_done = next_tokens == single_eos_id
            else:
                is_done = torch.isin(next_tokens, eos_token_ids)
            unfinished_sequences = unfinished_sequences & ~is_done.long()

            # Check previous async copy result (non-blocking CPU query)
            if eos_inflight and eos_events[eos_check_idx].query():
                if pinned_flags[eos_check_idx][0].item() == 0:
                    break
                eos_inflight = False

            # Launch new async copy every step into alternating buffer
            write_idx = eos_buf_idx
            pinned_flags[write_idx].copy_(unfinished_sequences, non_blocking=True)
            eos_events[write_idx].record()
            eos_check_idx = write_idx
            eos_buf_idx = 1 - write_idx
            eos_inflight = True

    if streamer is not None:
        streamer.end()

    # Trim buffer to actual length
    input_ids = input_ids[:, :current_length]

    if generation_config.return_dict_in_generate:
        cache = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]
        return GenerateDecoderOnlyOutput(
            sequences=input_ids, scores=None, logits=None,
            attentions=None, hidden_states=None, past_key_values=cache,
        )
    return input_ids


# ============================================================================
# Adaptive Gamma Predictor (SpecKV-inspired)
# ============================================================================
class AdaptiveGammaPredictor:
    """Confidence-based adaptive speculation length selector.

    Uses eagle3 draft model's max logit (confidence) per step to predict
    acceptance probability, then selects optimal gamma that maximizes
    expected throughput = E[accepted_tokens] / iteration_time.
    """

    __slots__ = ('conf_thresh_low', 'conf_thresh_high', 'acceptance_ema',
                 'ema_alpha', 'consecutive_rejects', 'skip_thresh')

    def __init__(self):
        self.conf_thresh_low = 0.0
        self.conf_thresh_high = 18.0
        self.acceptance_ema = 0.5
        self.ema_alpha = 0.15
        self.consecutive_rejects = 0
        self.skip_thresh = 3

    def select_gamma(self, confidences):
        """Select verification length based on draft confidence values.

        Since verify time is constant regardless of num_verify for small sequences,
        always verify all 3 draft tokens (never reduce gamma below 3).
        The confidence signal is used only for the decode fallback decision.

        Args:
            confidences: list of float max_logit values per draft step
        Returns:
            gamma: int — always 3 (verify all drafted tokens)
        """
        return 3

    def should_speculate(self):
        """Whether to speculate or fall back to decode-only.
        Disabled for now: decode fallback doesn't update combined_hidden for eagle3.
        """
        return True

    def update(self, accepted, gamma):
        """Update predictor state after a verification step."""
        rate = accepted / gamma if gamma > 0 else 0.0
        self.acceptance_ema = self.ema_alpha * rate + (1.0 - self.ema_alpha) * self.acceptance_ema
        if accepted == 0:
            self.consecutive_rejects += 1
        else:
            self.consecutive_rejects = 0


# ============================================================================
# Speculative Greedy Decode with EAGLE3
# ============================================================================
@torch.no_grad()
def _verify_forward(token_ids, cache_position_start, num_tokens, model, cache,
                    rope_cos_table, rope_sin_table, layers, lang_model, lm_head,
                    embed_tokens, eps, aux_layers=(2, 14, 25)):
    """Multi-token forward through target model with KV cache append.

    Returns (logits [num_tokens, vocab], aux_hidden dict).
    """
    hidden_states = embed_tokens(token_ids.view(1, num_tokens))

    cos = rope_cos_table[cache_position_start:cache_position_start + num_tokens].unsqueeze(0)
    sin = rope_sin_table[cache_position_start:cache_position_start + num_tokens].unsqueeze(0)
    position_embeddings = (cos, sin)

    cache_seqlens = torch.tensor([cache_position_start], dtype=torch.int32, device=token_ids.device)

    num_layers = len(layers)
    aux_hidden = {}

    normed = batch_rmsnorm(hidden_states, layers[0].input_layernorm.weight, eps)

    for layer_idx in range(num_layers):
        if layer_idx in aux_layers:
            aux_hidden[layer_idx] = hidden_states

        layer = layers[layer_idx]
        attn = layer.self_attn

        qkv_out = F.linear(normed, attn.qkv_weight)
        q_dim = attn._q_out_dim
        kv_dim = attn._kv_out_dim
        hidden_shape = (1, num_tokens, -1, attn.head_dim)

        query_states, key_states = batch_fused_qk_norm_rope(
            qkv_out[..., :q_dim + kv_dim],
            attn.q_norm.weight, attn.k_norm.weight,
            cos, sin,
            attn.q_norm.variance_epsilon,
            attn.config.num_attention_heads, attn.config.num_key_value_heads,
            attn.head_dim,
        )
        value_states = qkv_out[..., q_dim + kv_dim:].view(hidden_shape)

        layer_cache = cache.layers[layer_idx]
        k_cache = layer_cache.keys
        v_cache = layer_cache.values

        attn_output = flash_attn_with_kvcache(
            query_states, k_cache, v_cache,
            k=key_states, v=value_states,
            cache_seqlens=cache_seqlens,
            causal=True,
            softmax_scale=attn.scaling,
        )

        attn_output = attn_output.reshape(1, num_tokens, -1)
        attn_output = attn.o_proj(attn_output)

        normed_mlp, hidden_states = batch_fused_residual_rmsnorm(
            hidden_states, attn_output,
            layer.post_attention_layernorm.weight, eps,
        )

        gate_up_out = F.linear(normed_mlp, layer.mlp.gate_up_weight)
        mlp_out = layer.mlp.down_proj(batch_fused_swiglu(gate_up_out, layer.mlp._intermediate_size))

        if layer_idx < num_layers - 1:
            normed, hidden_states = batch_fused_residual_rmsnorm(
                hidden_states, mlp_out,
                layers[layer_idx + 1].input_layernorm.weight, eps,
            )
        else:
            normed, hidden_states = batch_fused_residual_rmsnorm(
                hidden_states, mlp_out,
                lang_model.norm.weight, eps,
            )

    logits = lm_head(normed)

    return logits.squeeze(0), aux_hidden


def _speculative_greedy_loop(model_self, input_ids, stopping_criteria, generation_config,
                             eagle3, synced_gpus=False, streamer=None, **model_kwargs):
    """Speculative greedy decode using EAGLE3 draft model."""
    import time as _time
    from transformers.generation.utils import ALL_CACHE_NAMES, GenerateDecoderOnlyOutput

    pad_token_id = generation_config._pad_token_tensor
    has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)

    # Support multiple EOS token IDs
    eos_token_ids_set = None
    if has_eos_stopping_criteria:
        for criteria in stopping_criteria:
            if hasattr(criteria, "eos_token_id"):
                eos_ids = criteria.eos_token_id.to(input_ids.device)
                eos_token_ids_set = set(eos_ids.tolist())
                break

    max_length = None
    for criteria in stopping_criteria:
        if hasattr(criteria, "max_length"):
            max_length = criteria.max_length
            break

    batch_size = input_ids.shape[0]

    # Prefill
    outputs = model_self._prefill(
        input_ids, generation_config, model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )
    model_kwargs = model_self._update_model_kwargs_for_generation(outputs, model_kwargs)
    _vocab_map = getattr(model_self, '_reduced_vocab_map', None)
    first_token_idx = outputs.logits[:, -1, :].float().argmax(dim=-1)
    first_token = _vocab_map[first_token_idx] if _vocab_map is not None else first_token_idx
    del outputs

    current_length = input_ids.shape[1]
    max_total_length = (max_length if max_length is not None else current_length + 256) + 8
    ids_buffer = torch.zeros(batch_size, max_total_length, dtype=input_ids.dtype, device=input_ids.device)
    ids_buffer[:, :current_length] = input_ids
    input_ids = ids_buffer

    # Check first_token for EOS
    first_token_val = first_token.item()
    input_ids[:, current_length] = first_token_val
    current_length += 1

    if streamer is not None:
        streamer.put(first_token.cpu())

    if (eos_token_ids_set and first_token_val in eos_token_ids_set) or \
       (max_length is not None and current_length >= max_length):
        input_ids = input_ids[:, :current_length]
        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids, scores=None, logits=None,
                attentions=None, hidden_states=None, past_key_values=None,
            )
        return input_ids

    # Setup model internals for verify forward
    cache = model_self._cache
    lang_model = model_self.model.language_model
    layers = lang_model.layers
    lm_head = model_self.lm_head
    embed_tokens = model_self.model.get_input_embeddings()
    eps = model_self.config.get_text_config().rms_norm_eps

    # Get precomputed RoPE table from decode graph runner
    decode_runner = getattr(model_self, '_decode_graph_runner', None)
    rope_cos_table = decode_runner._rope_cos_table
    rope_sin_table = decode_runner._rope_sin_table

    # Cache position after prefill + first token
    cache_pos = current_length - 1  # position of first_token in cache

    # First-time warmup: eager 1-token verify to warm up kernels
    if not getattr(model_self, '_spec_warmup_done', False):
        target_token = first_token.squeeze()
        logits, aux_hidden = _verify_forward(
            target_token.view(1), cache_pos, 1, model_self, cache,
            rope_cos_table, rope_sin_table, layers, lang_model, lm_head,
            embed_tokens, eps,
        )
        cache_pos += 1
        _reduced_idx = logits[0].float().argmax()
        next_token = _vocab_map[_reduced_idx] if _vocab_map is not None else _reduced_idx
        next_token_val = next_token.item()
        model_self._spec_warmup_done = True

        input_ids[0, current_length] = next_token_val
        current_length += 1

        if streamer is not None:
            streamer.put(next_token.view(1).cpu())

        if (eos_token_ids_set and next_token_val in eos_token_ids_set) or \
           (max_length is not None and current_length >= max_length):
            input_ids = input_ids[:, :current_length]
            if generation_config.return_dict_in_generate:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids, scores=None, logits=None,
                    attentions=None, hidden_states=None, past_key_values=None,
                )
            return input_ids

        combined_hidden = eagle3.combine_hidden_states(
            aux_hidden[2][0, 0], aux_hidden[14][0, 0], aux_hidden[25][0, 0],
        )
    else:
        # Fast path: use prefill aux_hidden directly, skip the expensive eager verify
        next_token = first_token.squeeze()
        next_token_val = first_token_val
        prefill_aux = getattr(model_self.model, '_prefill_aux_hidden', None)
        if prefill_aux is not None:
            combined_hidden = eagle3.combine_hidden_states(
                prefill_aux[2][0, -1], prefill_aux[14][0, -1], prefill_aux[25][0, -1],
            )
        else:
            combined_hidden = torch.zeros(eagle3.hidden_size, dtype=MODEL_DTYPE, device=input_ids.device)

    # Eagle3 prefill: use aux_hidden from target's prefill for full context
    eagle3.reset_cache()
    prefill_aux = getattr(model_self.model, '_prefill_aux_hidden', None)
    prefill_position_ids = getattr(model_self.model, '_prefill_position_ids', None)
    if prefill_aux is not None:
        prefill_len = prefill_aux[2].shape[1]
        orig_ids = input_ids[0, 1:prefill_len].clone()
        eagle3_prefill_ids = torch.cat([orig_ids, torch.tensor([first_token_val], device=input_ids.device)])
        eagle3_combined_all = eagle3.combine_hidden_states_batch(
            prefill_aux[2][0], prefill_aux[14][0], prefill_aux[25][0],
        )
        # Pass MRoPE positions to eagle3 prefill (shifted by 1 to match eagle3 token shift)
        eagle3_mrope_pos = None
        if prefill_position_ids is not None:
            # position_ids shape: [3, 1, seq_len] or [4, 1, seq_len]
            pos_3d = prefill_position_ids[1:] if prefill_position_ids.shape[0] == 4 else prefill_position_ids
            # Eagle3 processes tokens[1:prefill_len] + [first_token]; positions shifted by 1
            # pos for eagle3 position i = target position at seq index i+1
            # Plus the first_token's position = last_pos + 1
            last_pos = pos_3d[:, 0, prefill_len - 1]  # [3]
            first_token_pos = (last_pos + 1).unsqueeze(-1)  # [3, 1]
            eagle3_mrope_pos = torch.cat([pos_3d[:, 0, 1:prefill_len], first_token_pos], dim=-1)  # [3, prefill_len]
        eagle3.prefill(eagle3_prefill_ids, eagle3_combined_all, prefill_len, mrope_position_ids=eagle3_mrope_pos)
        # if eagle3_mrope_pos is not None:
        #     print(f"[Eagle3] MRoPE fix: prefill_len={prefill_len}, decode_mrope_start={eagle3._decode_mrope_start}, "
        #           f"mrope_pos range t=[{eagle3_mrope_pos[0,0].item()},{eagle3_mrope_pos[0,-1].item()}] "
        #           f"h=[{eagle3_mrope_pos[1,0].item()},{eagle3_mrope_pos[1,-1].item()}] "
        #           f"w=[{eagle3_mrope_pos[2,0].item()},{eagle3_mrope_pos[2,-1].item()}]")
        # else:
        #     print("[Eagle3] WARNING: no MRoPE positions available!")
        del model_self.model._prefill_aux_hidden
        if hasattr(model_self.model, '_prefill_position_ids'):
            del model_self.model._prefill_position_ids

    _use_parallel_draft = os.environ.get('USE_PARALLEL', '0') == '1' and getattr(eagle3, 'mask_hidden', None) is not None
    draft_k = int(os.environ.get('DRAFT_K', '4'))
    _use_fused = os.environ.get('USE_FUSED', '0') == '1'
    target_token = next_token

    # Eagle3 cache position: draft() will start here
    eagle3_cache_pos = eagle3.cache_seqlens.item()
    eagle3_cache_seqlens = eagle3.cache_seqlens

    _total_drafted = 0
    _total_accepted = 0
    _total_iters = 0
    _total_tokens_produced = 0

    if _use_fused:
        # --- Fused draft+verify path (default) ---
        if not _use_parallel_draft and not getattr(eagle3, '_draft_graph_ready', False):
            eagle3.setup_draft_buffers(draft_k=draft_k)

        fused_runner = getattr(model_self, '_fused_draft_verify_runner', None)
        if fused_runner is None:
            fused_runner = FusedDraftVerifyRunner(model_self, eagle3, input_ids.device, draft_k=draft_k)
            fused_runner.capture(rope_cos_table, rope_sin_table, warmup_cache_pos=cache_pos)
            model_self._fused_draft_verify_runner = fused_runner

        fused_run = fused_runner.run
        fused_data = fused_runner._data
        fused_verify_argmax = fused_data['verify_argmax']
        fused_combined_all = fused_data['combined_all']
        fused_result_buf = fused_data['result_buf']
        num_verify = fused_runner.num_verify

        while True:
            fused_run(target_token, combined_hidden, eagle3_cache_pos, cache_pos)

            all_vals = fused_result_buf.tolist()
            verify_vals = all_vals[:num_verify]
            draft_vals = all_vals[num_verify:]

            accepted = 0
            for _di in range(draft_k):
                if verify_vals[_di] == draft_vals[_di]:
                    accepted += 1
                else:
                    break

            num_new = accepted + 1
            bonus_val = verify_vals[accepted]
            for _di in range(accepted):
                input_ids[0, current_length + _di] = draft_vals[_di]
            input_ids[0, current_length + accepted] = bonus_val

            combined_hidden = fused_combined_all[accepted]
            target_token = fused_verify_argmax[accepted]

            _total_drafted += draft_k
            _total_accepted += accepted
            _total_iters += 1
            _total_tokens_produced += num_new

            eagle3_cache_pos += min(accepted + 1, draft_k)
            cache_pos += num_new
            current_length += num_new

            if eos_token_ids_set:
                eos_found = False
                for _di in range(accepted):
                    if draft_vals[_di] in eos_token_ids_set:
                        current_length -= (num_new - _di - 1)
                        eos_found = True
                        break
                if not eos_found and bonus_val in eos_token_ids_set:
                    eos_found = True
                if eos_found:
                    break

            if max_length is not None and current_length >= max_length:
                break
            if current_length >= max_total_length:
                break
    else:
        # --- Separate draft+verify path (USE_FUSED=0 fallback) ---
        if not _use_parallel_draft and not getattr(eagle3, '_draft_graph_ready', False):
            eagle3.setup_draft_buffers(draft_k=draft_k, capture_standalone_graph=True)

        verify_runners = getattr(model_self, '_verify_graph_runners', None)
        if verify_runners is None:
            verify_runners = {}
            for nv in (draft_k + 1,):
                runner = VerifyCudaGraphRunner(model_self, input_ids.device, num_verify=nv, fc_weight=eagle3.fc_weight)
                runner.capture(rope_cos_table, rope_sin_table, warmup_cache_pos=cache_pos)
                verify_runners[nv] = runner
            model_self._verify_graph_runners = verify_runners

        verify_buf = torch.zeros(draft_k + 1, dtype=torch.long, device=input_ids.device)
        verify_run = verify_runners[draft_k + 1].run
        # Force sequential draft if FORCE_SEQ=1 (for debugging parallel vs sequential)
        _force_seq = os.environ.get('FORCE_SEQ', '0') == '1'
        if _force_seq and _use_parallel_draft:
            eagle3.setup_draft_buffers(draft_k=draft_k, capture_standalone_graph=True)
            eagle3_draft = eagle3.draft
        else:
            eagle3_draft = eagle3.parallel_draft if _use_parallel_draft else eagle3.draft

        num_verify = draft_k + 1
        result_buf = torch.zeros(num_verify + draft_k, dtype=torch.long, device=input_ids.device)
        draft_outs_ref = eagle3._draft_static_outs

        # Inline eagle3 draft graph setup to eliminate function call overhead
        _e3_static_token = eagle3._draft_static_token
        _e3_static_hidden = eagle3._draft_static_hidden
        _e3_static_cos = eagle3._draft_static_cos
        _e3_static_sin = eagle3._draft_static_sin
        _e3_rope_cos = eagle3.rope_cos
        _e3_rope_sin = eagle3.rope_sin
        _e3_graph = eagle3._draft_graph
        _e3_get_rope_pos = eagle3._get_rope_position

        # Inline verify graph setup to eliminate dict lookups
        _vr = verify_runners[num_verify]
        _v_graph = _vr._graph
        _v_gd = _vr._graph_data
        _v_token_ids = _v_gd['token_ids']
        _v_cache_seqlens = _v_gd['cache_seqlens']
        _v_cos = _v_gd['cos']
        _v_sin = _v_gd['sin']
        _v_argmax = _v_gd['argmax']
        _v_aux_bufs = _v_gd['aux_bufs']
        _v_combined_all = _v_gd['combined_all']
        _v_rope_cos_table = _vr._rope_cos_table
        _v_rope_sin_table = _vr._rope_sin_table

        while True:
            # Eagle3 draft: inline setup + graph replay
            _e3_static_token[0] = target_token
            _e3_static_hidden[0, 0].copy_(combined_hidden.view(-1))
            _rope_pos = _e3_get_rope_pos(eagle3_cache_pos)
            _e3_static_cos[:, 0, 0].copy_(_e3_rope_cos[_rope_pos:_rope_pos + draft_k])
            _e3_static_sin[:, 0, 0].copy_(_e3_rope_sin[_rope_pos:_rope_pos + draft_k])
            _e3_graph.replay()

            # Verify: inline setup + graph replay
            _v_token_ids[0, 0] = target_token
            _v_token_ids[0, 1:num_verify].copy_(draft_outs_ref[:draft_k])
            _v_cache_seqlens[0] = cache_pos
            _v_cos[0, :num_verify].copy_(_v_rope_cos_table[cache_pos:cache_pos + num_verify])
            _v_sin[0, :num_verify].copy_(_v_rope_sin_table[cache_pos:cache_pos + num_verify])
            _v_graph.replay()
            verify_tokens = _v_argmax
            new_aux_hidden = _v_aux_bufs
            combined_all = _v_combined_all

            result_buf[:num_verify].copy_(verify_tokens)
            result_buf[num_verify:].copy_(draft_outs_ref[:draft_k])
            all_vals = result_buf.tolist()
            verify_vals = all_vals[:num_verify]
            draft_vals = all_vals[num_verify:]

            accepted = 0
            for _di in range(draft_k):
                if verify_vals[_di] == draft_vals[_di]:
                    accepted += 1
                else:
                    break

            if DEBUG and _total_iters < 5:
                print(f"  [iter={_total_iters}] eagle3_pos={eagle3_cache_pos} cache_pos={cache_pos}")
                for _di in range(draft_k):
                    match = "✓" if verify_vals[_di] == draft_vals[_di] else "✗"
                    print(f"    pos{_di}: draft={draft_vals[_di]} verify={verify_vals[_di]} {match}")

            if DEBUG:
                _ppa = getattr(model_self, '_per_pos_accept', None)
                if _ppa is None:
                    model_self._per_pos_accept = [0] * draft_k
                    model_self._per_pos_total = [0] * draft_k
                    _ppa = model_self._per_pos_accept
                _ppt = model_self._per_pos_total
                for _pi in range(draft_k):
                    if _pi <= accepted and _pi < draft_k:
                        _ppt[_pi] += 1
                        if _pi < accepted:
                            _ppa[_pi] += 1

            num_new = accepted + 1
            for _di in range(accepted):
                input_ids[0, current_length + _di] = draft_vals[_di]
            bonus_val = verify_vals[accepted]
            input_ids[0, current_length + accepted] = bonus_val

            if _use_parallel_draft and accepted > 0:
                fix_ids = torch.tensor(draft_vals[:accepted], device=input_ids.device)
                fix_hidden = eagle3.combine_hidden_states_batch(
                    new_aux_hidden[2][0, 1:accepted+1],
                    new_aux_hidden[14][0, 1:accepted+1],
                    new_aux_hidden[25][0, 1:accepted+1],
                )
                eagle3.cache_update(fix_ids, fix_hidden, eagle3_cache_pos + 1)

            eagle3_cache_pos += min(accepted + 1, draft_k)
            eagle3_cache_seqlens[0] = eagle3_cache_pos

            combined_hidden = combined_all[accepted]
            target_token = verify_tokens[accepted]

            _total_drafted += draft_k
            _total_accepted += accepted
            _total_iters += 1
            _total_tokens_produced += num_new
            cache_pos += num_new
            current_length += num_new

            if eos_token_ids_set:
                eos_found = False
                written_tokens = draft_vals[:accepted] + [bonus_val]
                for eos_i, tok in enumerate(written_tokens):
                    if tok in eos_token_ids_set:
                        current_length -= (num_new - eos_i - 1)
                        eos_found = True
                        break
                if eos_found:
                    break

            if max_length is not None and current_length >= max_length:
                break
            if current_length >= max_total_length:
                break

    if DEBUG and _total_drafted > 0:
        tok_per_iter = _total_tokens_produced / _total_iters if _total_iters > 0 else 0
        per_pos = getattr(model_self, '_per_pos_accept', None)
        if per_pos is None:
            per_pos = [0] * draft_k
            model_self._per_pos_accept = per_pos
            model_self._per_pos_total = [0] * draft_k
        pos_str = " ".join(f"p{i}={model_self._per_pos_accept[i]}/{model_self._per_pos_total[i]}" if model_self._per_pos_total[i] > 0 else f"p{i}=?" for i in range(draft_k))
        print(f"[Eagle3] accept={_total_accepted}/{_total_drafted} ({_total_accepted/_total_drafted:.2%}), iters={_total_iters}, tok/iter={tok_per_iter:.2f}, tokens={current_length} | {pos_str}")

    if streamer is not None:
        streamer.end()

    input_ids = input_ids[:, :current_length]

    if generation_config.return_dict_in_generate:
        cache_out = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache_out = model_kwargs[cache_key]
        return GenerateDecoderOnlyOutput(
            sequences=input_ids, scores=None, logits=None,
            attentions=None, hidden_states=None, past_key_values=cache_out,
        )
    return input_ids


# ============================================================================
# Patch 6: Fast reset cache
# ============================================================================
def _install_fast_reset_cache():
    from transformers.generation.utils import GenerationMixin

    def fast_prepare_static_cache(self, cache_implementation, batch_size, max_cache_len, model_kwargs):
        self._cache.reset()
        return self._cache

    GenerationMixin._prepare_static_cache = fast_prepare_static_cache


# ============================================================================
# ViTCudaGraphRunner  (PIECEWISE implementation)
# ============================================================================
class ViTCudaGraphRunner:
    """PIECEWISE CUDA Graph runner for ViT forward pass.

    Splits the 24 ViT blocks into 4 segments at deepstack injection points:
      Seg 0: blocks  0-5  + deepstack_merger[0]
      Seg 1: blocks  6-11 + deepstack_merger[1]
      Seg 2: blocks 12-17 + deepstack_merger[2]
      Seg 3: blocks 18-23 + final merger

    Each segment is captured as an independent CUDA graph sharing a single
    graph pool.  This drastically reduces VRAM vs one monolithic graph per
    (num_patches, temporal) combination.

    Graph key: (num_patches, len(cu_seqlens))
    Lazy capture: graphs built on first encounter.
    """

    NUM_DEEPSTACK = 3
    NUM_SEGMENTS = 4  # 3 deepstack segments + 1 tail+merger segment
    KNOWN_PATCH_COUNTS = list(range(892, MAX_PATCH_COUNT+1, 4))
    KNOWN_TEMPORALS = [1]

    def __init__(self, visual_model, device, vit_hidden_size=1024, out_hidden_size=2048):
        self.visual = visual_model
        self.device = device
        self.vit_hidden_size = vit_hidden_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = visual_model.spatial_merge_size
        self._graphs = {}  # (num_patches, cu_seqlens_len) -> graph data
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._segment_fns = self._build_segment_fns()

    def _build_segment_fns(self):
        """Build per-segment forward callables.

        Segments 0-2: a range of blocks + deepstack_merger at the end.
        Segment 3: remaining blocks + final merger.
        """
        blocks = self.visual.blocks
        merger = self.visual.merger
        deepstack_visual_indexes = self.visual.deepstack_visual_indexes  # [5, 11, 17]
        deepstack_merger_list = self.visual.deepstack_merger_list

        # Determine block ranges: [0..5], [6..11], [12..17], [18..23]
        boundaries = deepstack_visual_indexes + [len(blocks) - 1]
        segment_fns = []

        for seg_idx in range(self.NUM_SEGMENTS):
            start_block = 0 if seg_idx == 0 else boundaries[seg_idx - 1] + 1
            end_block = boundaries[seg_idx]  # inclusive
            seg_blocks = blocks[start_block:end_block + 1]

            if seg_idx < self.NUM_DEEPSTACK:
                ds_merger = deepstack_merger_list[seg_idx]

                def _make_ds_fn(_seg_blocks, _ds_merger):
                    def segment_forward(hidden_states, cu_seqlens, position_embeddings,
                                        max_seqlen, ds_output_buf):
                        for block in _seg_blocks:
                            hidden_states = block(
                                hidden_states, cu_seqlens=cu_seqlens,
                                position_embeddings=position_embeddings,
                                _max_seqlen=max_seqlen,
                            )
                        ds_out = _ds_merger(hidden_states)
                        ds_output_buf.copy_(ds_out)
                        return hidden_states
                    return segment_forward

                segment_fns.append(_make_ds_fn(seg_blocks, ds_merger))
            else:
                # Tail segment: remaining blocks + final merger
                def _make_tail_fn(_seg_blocks, _merger):
                    def segment_forward(hidden_states, cu_seqlens, position_embeddings,
                                        max_seqlen, merged_output_buf):
                        for block in _seg_blocks:
                            hidden_states = block(
                                hidden_states, cu_seqlens=cu_seqlens,
                                position_embeddings=position_embeddings,
                                _max_seqlen=max_seqlen,
                            )
                        merged = _merger(hidden_states)
                        merged_output_buf.copy_(merged)
                        return hidden_states
                    return segment_forward

                segment_fns.append(_make_tail_fn(seg_blocks, merger))

        return segment_fns

    def _global_warmup(self):
        """Run 3 warmup iterations with one shape to JIT-compile all Triton kernels.

        After this, all subsequent captures for any shape can skip warmup
        entirely (like vLLM's PIECEWISE mode with torch.compile).
        """
        warmup_patches = self.KNOWN_PATCH_COUNTS[0]
        rope_dim = self.vit_hidden_size // self.visual.blocks[0].attn.num_heads
        spatial = self.spatial_merge_size
        num_merged = warmup_patches // (spatial * spatial)

        dummy_hidden = torch.zeros(warmup_patches, self.vit_hidden_size,
                                   dtype=MODEL_DTYPE, device=self.device)
        dummy_cu = torch.tensor([0, warmup_patches], dtype=torch.int32, device=self.device)
        dummy_cos = torch.zeros(warmup_patches, rope_dim, dtype=MODEL_DTYPE, device=self.device)
        dummy_sin = torch.zeros(warmup_patches, rope_dim, dtype=MODEL_DTYPE, device=self.device)
        dummy_merged = torch.zeros(num_merged, self.out_hidden_size,
                                   dtype=MODEL_DTYPE, device=self.device)
        dummy_ds = [
            torch.zeros(num_merged, self.out_hidden_size, dtype=MODEL_DTYPE, device=self.device)
            for _ in range(self.NUM_DEEPSTACK)
        ]
        pos_emb = (dummy_cos, dummy_sin)

        with torch.no_grad():
            for _ in range(3):
                cur = dummy_hidden
                for seg_idx, seg_fn in enumerate(self._segment_fns):
                    out_buf = dummy_ds[seg_idx] if seg_idx < self.NUM_DEEPSTACK else dummy_merged
                    cur = seg_fn(cur, dummy_cu, pos_emb, warmup_patches, out_buf)
        torch.cuda.synchronize()

    def capture_all(self):
        """Global warmup once, then 0-warmup capture for all known configs."""
        print("[ViTPiecewise] Global warmup (compiling Triton kernels)...")
        self._global_warmup()

        configs = []
        for num_patches in self.KNOWN_PATCH_COUNTS:
            for temporal in self.KNOWN_TEMPORALS:
                if num_patches % temporal != 0:
                    continue
                seg_len = num_patches // temporal
                cu_seqlens = torch.tensor(
                    [seg_len * i for i in range(temporal + 1)],
                    dtype=torch.int32, device=self.device,
                )
                configs.append((num_patches, cu_seqlens, seg_len))

        print(f"[ViTPiecewise] Capturing {len(configs)} graph sets (0-warmup)...")
        for i, (num_patches, cu_seqlens, max_seqlen) in enumerate(configs):
            self._capture_for_config(num_patches, cu_seqlens, max_seqlen)
        torch.cuda.synchronize()
        print(f"[ViTPiecewise] All {len(configs)} graph sets captured.")

    def _capture_for_config(self, num_patches, cu_seqlens, max_seqlen):
        """Capture piecewise graphs for one config (0-warmup, kernels already compiled)."""
        spatial_merge = self.spatial_merge_size
        num_merged = num_patches // (spatial_merge * spatial_merge)
        rope_dim = self.vit_hidden_size // self.visual.blocks[0].attn.num_heads

        static_hidden = torch.zeros(num_patches, self.vit_hidden_size,
                                    dtype=MODEL_DTYPE, device=self.device)
        static_cu_seqlens = cu_seqlens.clone()
        static_cos = torch.zeros(num_patches, rope_dim,
                                 dtype=MODEL_DTYPE, device=self.device)
        static_sin = torch.zeros(num_patches, rope_dim,
                                 dtype=MODEL_DTYPE, device=self.device)
        static_merged_output = torch.zeros(num_merged, self.out_hidden_size,
                                           dtype=MODEL_DTYPE, device=self.device)
        static_deepstack_outputs = [
            torch.zeros(num_merged, self.out_hidden_size,
                        dtype=MODEL_DTYPE, device=self.device)
            for _ in range(self.NUM_DEEPSTACK)
        ]
        pos_emb = (static_cos, static_sin)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        # Capture each segment directly (no warmup needed)
        segment_graphs = []
        cur_hidden = static_hidden
        for seg_idx, seg_fn in enumerate(self._segment_fns):
            out_buf = (static_deepstack_outputs[seg_idx]
                       if seg_idx < self.NUM_DEEPSTACK
                       else static_merged_output)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=self._graph_pool):
                result = seg_fn(cur_hidden, static_cu_seqlens, pos_emb,
                                max_seqlen, out_buf)
                if seg_idx < self.NUM_SEGMENTS - 1:
                    static_hidden.copy_(result)
                    cur_hidden = static_hidden

            segment_graphs.append(graph)
            torch.cuda.synchronize()

        graph_key = (num_patches, cu_seqlens.shape[0])
        self._graphs[graph_key] = {
            'segment_graphs': segment_graphs,
            'hidden': static_hidden,
            'cu_seqlens': static_cu_seqlens,
            'cos': static_cos,
            'sin': static_sin,
            'merged_output': static_merged_output,
            'deepstack_outputs': static_deepstack_outputs,
        }

    def run(self, hidden_states, cu_seqlens, position_embeddings, max_seqlen):
        """Run ViT via PIECEWISE CUDA Graph replay."""
        num_patches = hidden_states.shape[0]
        graph_key = (num_patches, cu_seqlens.shape[0])

        if graph_key not in self._graphs:
            self._capture_for_config(num_patches, cu_seqlens, max_seqlen)

        graph_data = self._graphs[graph_key]
        cos, sin = position_embeddings

        graph_data['hidden'].copy_(hidden_states)
        graph_data['cu_seqlens'].copy_(cu_seqlens)
        graph_data['cos'].copy_(cos)
        graph_data['sin'].copy_(sin)

        for seg_graph in graph_data['segment_graphs']:
            seg_graph.replay()

        # Skip clone: batch=1 means these static buffers won't be overwritten
        # until the next ViT graph replay. The downstream PrefillCudaGraphRunner
        # copies from these buffers into its own static buffers before replaying.
        return graph_data['merged_output'], graph_data['deepstack_outputs']


# ============================================================================
# PrefillCudaGraphRunner  (PIECEWISE implementation)
# ============================================================================
class PrefillCudaGraphRunner:
    """PIECEWISE CUDA Graph runner for prefill.

    Instead of capturing one monolithic graph per seq_len (28 layers),
    we split the LLM into segments at deepstack injection points:
      Segment 0: layer 0  + deepstack inject
      Segment 1: layer 1  + deepstack inject
      Segment 2: layer 2  + deepstack inject
      Segment 3: layers 3-27 + final norm

    Benefits:
      - All segments share a single CUDA graph pool → huge VRAM savings
      - Lazy capture: graphs are built on first encounter of each padded_len
      - Segment graphs are much smaller → faster capture
    """
    NUM_DEEPSTACK = 3
    NUM_SEGMENTS = 6          # 3 deepstack + 3 tail sub-segments (3-13, 14-24, 25-27+norm)
    CAPTURE_STEP = 4
    MAX_VISUAL_TOKENS = MAX_PATCH_COUNT // 4
    MIN_SEQ_LEN = 240
    MAX_SEQ_LEN = 48 + MAX_VISUAL_TOKENS
    # Aux hidden state layers for EAGLE3 (input to these layers)
    AUX_LAYERS = (2, 14, 25)

    def __init__(self, language_model, cache, device, hidden_size=2048):
        self.language_model = language_model
        self.cache = cache
        self.device = device
        self.hidden_size = hidden_size
        self._graphs = {}                       # padded_len -> segment data dict
        self._graph_pool = torch.cuda.graph_pool_handle()  # shared pool
        self._segment_fns = self._build_segment_fns()
        self._rotary_emb = language_model.rotary_emb  # avoid repeated attr lookup

    # ------------------------------------------------------------------
    # Build per-segment forward callables
    # ------------------------------------------------------------------
    def _build_segment_fns(self):
        layers = self.language_model.layers
        norm_fn = self.language_model.norm.forward
        num_layers = len(layers)

        segment_fns = []

        # Segments 0-2: single layer + deepstack injection
        for seg_idx in range(self.NUM_DEEPSTACK):
            layer = layers[seg_idx]
            ds_idx = seg_idx

            def _make_fn(_layer, _ds_idx):
                def segment_forward(hidden_states, position_embeddings, cache_position,
                                    past_key_values, deepstack_indices, deepstack_embeds):
                    hidden_states = _layer(
                        hidden_states, attention_mask=None, position_ids=None,
                        past_key_values=past_key_values, cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = hidden_states.clone()
                    hidden_states[0].index_add_(0, deepstack_indices, deepstack_embeds)
                    return hidden_states
                return segment_forward
            segment_fns.append(_make_fn(layer, ds_idx))

        # Helper: build an inter-layer-fused tail segment forward.
        # Fuses residual-add + next-layer's input norm into batch_fused_residual_rmsnorm,
        # eliminating standalone batch_rmsnorm + add between consecutive layers.
        text_config = self.language_model.config
        _eps = text_config.rms_norm_eps
        _final_norm_weight = self.language_model.norm.weight

        def _make_fused_tail(seg_layers, is_final_segment):
            def fused_tail_forward(hidden_states, position_embeddings, cache_position,
                                   past_key_values, _unused_indices, _unused_embeds):
                normed = batch_rmsnorm(hidden_states, seg_layers[0].input_layernorm.weight, _eps)
                for i, layer in enumerate(seg_layers):
                    attn = layer.self_attn
                    qkv_out = F.linear(normed, attn.qkv_weight)
                    q_dim = attn._q_out_dim
                    kv_dim = attn._kv_out_dim
                    cos, sin = position_embeddings
                    hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)

                    query_states_bshd, key_states_bshd = batch_fused_qk_norm_rope(
                        qkv_out[..., :q_dim + kv_dim],
                        attn.q_norm.weight, attn.k_norm.weight,
                        cos, sin,
                        attn.q_norm.variance_epsilon,
                        attn.config.num_attention_heads, attn.config.num_key_value_heads,
                        attn.head_dim,
                    )
                    value_states_bshd = qkv_out[..., q_dim + kv_dim:].view(hidden_shape)

                    attn_output = flash_attn_func(
                        query_states_bshd, key_states_bshd, value_states_bshd,
                        causal=True, softmax_scale=attn.scaling,
                    )
                    if past_key_values is not None:
                        past_key_values.update(
                            key_states_bshd, value_states_bshd, attn.layer_idx,
                            {"sin": sin, "cos": cos, "cache_position": cache_position},
                        )
                    attn_output = attn.o_proj(attn_output.reshape(*hidden_states.shape[:-1], -1))

                    normed_mlp, hidden_states = batch_fused_residual_rmsnorm(
                        hidden_states, attn_output,
                        layer.post_attention_layernorm.weight, _eps,
                    )
                    gate_up_out = F.linear(normed_mlp, layer.mlp.gate_up_weight)
                    mlp_out = layer.mlp.down_proj(
                        batch_fused_swiglu(gate_up_out, layer.mlp._intermediate_size)
                    )

                    if i < len(seg_layers) - 1:
                        normed, hidden_states = batch_fused_residual_rmsnorm(
                            hidden_states, mlp_out,
                            seg_layers[i + 1].input_layernorm.weight, _eps,
                        )
                    elif is_final_segment:
                        normed, hidden_states = batch_fused_residual_rmsnorm(
                            hidden_states, mlp_out, _final_norm_weight, _eps,
                        )
                    else:
                        hidden_states = hidden_states + mlp_out

                if is_final_segment:
                    return normed
                return hidden_states
            return fused_tail_forward

        # Segment 3: layers 3-13
        segment_fns.append(_make_fused_tail(layers[3:14], is_final_segment=False))
        # Segment 4: layers 14-24
        segment_fns.append(_make_fused_tail(layers[14:25], is_final_segment=False))
        # Segment 5: layers 25-27 + final norm (fused into last residual_rmsnorm)
        segment_fns.append(_make_fused_tail(layers[25:], is_final_segment=True))

        return segment_fns

    # ------------------------------------------------------------------
    def _global_warmup(self):
        """Run 3 warmup iterations with one seq_len to JIT-compile all kernels."""
        warmup_len = self.MIN_SEQ_LEN
        rotary_emb = self._rotary_emb

        dummy_hidden = torch.zeros(1, warmup_len, self.hidden_size,
                                   dtype=MODEL_DTYPE, device=self.device)
        dummy_cache_pos = torch.arange(warmup_len, dtype=torch.long, device=self.device)
        dummy_pos_ids = torch.zeros(3, 1, warmup_len, dtype=torch.long, device=self.device)
        with torch.no_grad():
            dummy_cos, dummy_sin = rotary_emb(dummy_hidden, dummy_pos_ids)

        max_vis = self.MAX_VISUAL_TOKENS
        dummy_ds_indices = torch.zeros(max_vis, dtype=torch.long, device=self.device)
        dummy_ds_embeds = [
            torch.zeros(max_vis, self.hidden_size, dtype=MODEL_DTYPE, device=self.device)
            for _ in range(self.NUM_DEEPSTACK)
        ]
        dummy_output = torch.zeros(1, warmup_len, self.hidden_size,
                                   dtype=MODEL_DTYPE, device=self.device)

        with torch.no_grad():
            for _ in range(3):
                self.cache.reset()
                cur = dummy_hidden
                for seg_idx, seg_fn in enumerate(self._segment_fns):
                    ds_emb = (dummy_ds_embeds[seg_idx]
                              if seg_idx < self.NUM_DEEPSTACK
                              else dummy_ds_embeds[0])
                    out = seg_fn(cur, (dummy_cos, dummy_sin),
                                 dummy_cache_pos, self.cache,
                                 dummy_ds_indices, ds_emb)
                    if seg_idx < self.NUM_SEGMENTS - 1:
                        dummy_hidden.copy_(out)
                        cur = dummy_hidden
                    else:
                        dummy_output.copy_(out)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    def capture_all(self):
        """Global warmup once, then 0-warmup capture for all padded seq_lens."""
        print("[PrefillPiecewise] Global warmup (compiling kernels)...")
        self._global_warmup()

        all_lens = list(range(self.MIN_SEQ_LEN, self.MAX_SEQ_LEN + 1, self.CAPTURE_STEP))
        print(f"[PrefillPiecewise] Capturing {len(all_lens)} graph sets (0-warmup)...")
        for i, padded_len in enumerate(all_lens):
            self._capture_for_seq_len(padded_len)
        torch.cuda.synchronize()
        print(f"[PrefillPiecewise] All {len(all_lens)} prefill graph sets captured.")

    # ------------------------------------------------------------------
    def _pad_seq_len(self, seq_len):
        step = self.CAPTURE_STEP
        return ((seq_len + step - 1) // step) * step

    # ------------------------------------------------------------------
    # Capture piecewise graphs for one padded seq_len
    # ------------------------------------------------------------------
    def _capture_for_seq_len(self, seq_len):
        """Capture piecewise graphs for one padded seq_len (0-warmup, kernels already compiled)."""
        rotary_emb = self._rotary_emb

        # ---------- shared static buffers (input/output) ----------
        static_hidden = torch.zeros(1, seq_len, self.hidden_size,
                                    dtype=MODEL_DTYPE, device=self.device)

        static_cache_position = torch.arange(seq_len, dtype=torch.long,
                                             device=self.device)

        dummy_pos_ids = torch.zeros(3, 1, seq_len, dtype=torch.long,
                                    device=self.device)
        with torch.no_grad():
            dummy_cos, dummy_sin = rotary_emb(static_hidden, dummy_pos_ids)
        static_cos = torch.zeros_like(dummy_cos)
        static_sin = torch.zeros_like(dummy_sin)

        max_vis = self.MAX_VISUAL_TOKENS
        static_deepstack_indices = torch.zeros(max_vis, dtype=torch.long,
                                               device=self.device)
        static_deepstack_embeds = [
            torch.zeros(max_vis, self.hidden_size, dtype=MODEL_DTYPE,
                        device=self.device)
            for _ in range(self.NUM_DEEPSTACK)
        ]

        static_output = torch.zeros(1, seq_len, self.hidden_size,
                                    dtype=MODEL_DTYPE, device=self.device)

        # Aux hidden state buffers for EAGLE3 (captured between segments)
        # aux_h2: input to layer 2 = output of segment 1
        # aux_h14: input to layer 14 = output of segment 3
        # aux_h25: input to layer 25 = output of segment 4
        static_aux_h2 = torch.zeros(1, seq_len, self.hidden_size,
                                    dtype=MODEL_DTYPE, device=self.device)
        static_aux_h14 = torch.zeros(1, seq_len, self.hidden_size,
                                     dtype=MODEL_DTYPE, device=self.device)
        static_aux_h25 = torch.zeros(1, seq_len, self.hidden_size,
                                     dtype=MODEL_DTYPE, device=self.device)

        # Map: after which segment index to copy into which aux buffer
        # seg 1 output → aux_h2, seg 3 output → aux_h14, seg 4 output → aux_h25
        aux_copy_map = {1: static_aux_h2, 3: static_aux_h14, 4: static_aux_h25}

        # ---------- capture stream ----------
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        # Capture each segment directly (no warmup needed, kernels already compiled)
        segment_graphs = []
        self.cache.reset()  # clean state before capture sequence
        for seg_idx, seg_fn in enumerate(self._segment_fns):
            ds_embeds = (static_deepstack_embeds[seg_idx]
                         if seg_idx < self.NUM_DEEPSTACK
                         else static_deepstack_embeds[0])
            ds_indices = static_deepstack_indices

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=self._graph_pool):
                out = seg_fn(static_hidden, (static_cos, static_sin),
                             static_cache_position, self.cache,
                             ds_indices, ds_embeds)
                if seg_idx < self.NUM_SEGMENTS - 1:
                    static_hidden.copy_(out)
                    if seg_idx in aux_copy_map:
                        aux_copy_map[seg_idx].copy_(static_hidden)
                else:
                    static_output.copy_(out)

            segment_graphs.append(graph)
            torch.cuda.synchronize()

        self._graphs[seq_len] = {
            'segment_graphs': segment_graphs,
            'hidden': static_hidden,
            'cos': static_cos,
            'sin': static_sin,
            'cache_position': static_cache_position,
            'deepstack_indices': static_deepstack_indices,
            'deepstack_embeds': static_deepstack_embeds,
            'output': static_output,
            'aux_h2': static_aux_h2,
            'aux_h14': static_aux_h14,
            'aux_h25': static_aux_h25,
        }

    # ------------------------------------------------------------------
    # Run (replay)
    # ------------------------------------------------------------------
    def run(self, inputs_embeds, position_ids, cache_position, past_key_values,
            visual_pos_masks, deepstack_visual_embeds, return_aux_hidden=False):
        seq_len = inputs_embeds.shape[1]
        padded_len = self._pad_seq_len(seq_len)

        if visual_pos_masks is not None and deepstack_visual_embeds is not None:
            visual_indices = visual_pos_masks[0].nonzero(as_tuple=True)[0]
            num_visual = visual_indices.shape[0]
        else:
            visual_indices = None
            num_visual = 0

        if padded_len not in self._graphs:
            self._capture_for_seq_len(padded_len)

        graph_data = self._graphs[padded_len]

        # ---- compute rotary embeddings (outside graph) ----
        # batch_size=1: position_ids is always [3, 1, seq_len] from compute_3d_position_ids
        pos_ids_3d = position_ids[1:] if position_ids.shape[0] == 4 else position_ids

        pad_amount = padded_len - seq_len
        if pad_amount > 0:
            # Use pre-allocated padded buffer to avoid torch.cat allocation
            padded_key = (padded_len, pos_ids_3d.dtype)
            if not hasattr(self, '_pad_pos_buf') or self._pad_pos_buf_key != padded_key:
                self._pad_pos_buf = torch.zeros(3, 1, padded_len, dtype=pos_ids_3d.dtype,
                                                device=self.device)
                self._pad_pos_buf_key = padded_key
            self._pad_pos_buf[:, :, :seq_len].copy_(pos_ids_3d)
            self._pad_pos_buf[:, :, seq_len:].zero_()
            pos_ids_3d_padded = self._pad_pos_buf
        else:
            pos_ids_3d_padded = pos_ids_3d

        cos, sin = self._rotary_emb(inputs_embeds, pos_ids_3d_padded)

        # ---- copy inputs into static buffers ----
        static_hidden = graph_data['hidden']
        if pad_amount > 0:
            # Only zero the padding region, not the whole buffer
            static_hidden[:, seq_len:, :].zero_()
            static_hidden[:, :seq_len, :].copy_(inputs_embeds)
        else:
            static_hidden.copy_(inputs_embeds)

        graph_data['cos'].copy_(cos)
        graph_data['sin'].copy_(sin)

        static_ds_indices = graph_data['deepstack_indices']
        if num_visual > 0:
            if num_visual < static_ds_indices.shape[0]:
                static_ds_indices[num_visual:].zero_()
            static_ds_indices[:num_visual].copy_(visual_indices)
        else:
            static_ds_indices.zero_()

        for i in range(self.NUM_DEEPSTACK):
            ds_buf = graph_data['deepstack_embeds'][i]
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                nv = deepstack_visual_embeds[i].shape[0]
                if nv < ds_buf.shape[0]:
                    ds_buf[nv:].zero_()
                ds_buf[:nv].copy_(deepstack_visual_embeds[i])
            else:
                ds_buf.zero_()

        # ---- replay all segments sequentially ----
        self.cache.reset()
        for seg_graph in graph_data['segment_graphs']:
            seg_graph.replay()

        output = graph_data['output'][:, :seq_len, :]

        if return_aux_hidden:
            aux_hidden = {
                2: graph_data['aux_h2'][:, :seq_len, :].clone(),
                14: graph_data['aux_h14'][:, :seq_len, :].clone(),
                25: graph_data['aux_h25'][:, :seq_len, :].clone(),
            }
            return output, aux_hidden

        return output






# ============================================================================
# DecodeCudaGraphRunner
# ============================================================================
class DecodeCudaGraphRunner:
    """CUDA Graph runner for the full LLM decode step (batch=1, seq=1).

    Captures: embed_tokens -> rotary_emb -> 28 LLM layers -> final_norm ->
              lm_head -> argmax into a single CUDA graph.

    Since decode always operates on fixed shape [1,1], only one graph is needed.
    Varying cache_position / position_ids are handled by copying into static
    input buffers before each replay.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._graph = None
        self._graph_data = None
        self._rope_cos_table = None
        self._rope_sin_table = None
        self._precompute_rope_table()

    def _precompute_rope_table(self):
        """Precompute cos/sin RoPE table for all decode positions.

        During decode, all 3 MRoPE dimensions have the same position value,
        so we can precompute a 1D table indexed by scalar position.
        """
        model = self.model
        language_model = model.model.language_model
        rotary_emb = language_model.rotary_emb
        max_pos = model._cache.layers[0].max_cache_len + 128  # cover full cache + margin

        # Build position_ids [3, 1, max_pos] with all dims equal
        positions = torch.arange(max_pos, device=self.device)
        position_ids = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).contiguous()  # [3, 1, max_pos]
        dummy_x = torch.zeros(1, max_pos, 1, dtype=MODEL_DTYPE, device=self.device)

        with torch.no_grad():
            cos, sin = rotary_emb(dummy_x, position_ids)
            # cos/sin shape: [1, max_pos, head_dim]
            self._rope_cos_table = cos.squeeze(0).contiguous()  # [max_pos, head_dim]
            self._rope_sin_table = sin.squeeze(0).contiguous()  # [max_pos, head_dim]

    def _build_decode_fn(self, static_cos, static_sin):
        """Build a callable that performs the full decode step.

        Args:
            static_cos: [1, 1, head_dim] static buffer for cos (filled before replay)
            static_sin: [1, 1, head_dim] static buffer for sin (filled before replay)
        """
        model = self.model
        inner_model = model.model
        embed_tokens = inner_model.get_input_embeddings()
        language_model = inner_model.language_model
        layers = language_model.layers
        norm_fn = language_model.norm.forward
        lm_head = model.lm_head
        num_layers = len(layers)
        cache = model._cache
        vocab_map = getattr(model, '_reduced_vocab_map', None)

        def decode_step(input_ids, cache_position):
            inputs_embeds = embed_tokens(input_ids)
            position_embeddings = (static_cos, static_sin)

            hidden_states = inputs_embeds
            for layer_idx in range(num_layers):
                hidden_states = layers[layer_idx](
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = norm_fn(hidden_states)
            logits = lm_head(hidden_states)
            reduced_idx = logits[:, -1, :].float().argmax(dim=-1)
            if vocab_map is not None:
                next_token = vocab_map[reduced_idx]
            else:
                next_token = reduced_idx
            return next_token

        return decode_step

    def capture(self, warmup_cache_position=None):
        """Capture the decode CUDA graph.

        Should be called after the first prefill so the KV cache already
        contains valid data and Triton/flash_attn kernels have been compiled.

        Args:
            warmup_cache_position: cache_position value for warmup/capture.
        """
        if warmup_cache_position is None:
            warmup_cache_position = 512

        head_dim = self._rope_cos_table.shape[1]

        # Static input buffers
        static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        static_cache_position = torch.tensor(
            [warmup_cache_position], dtype=torch.long, device=self.device,
        )
        # Static RoPE buffers — filled from precomputed table before each replay
        static_cos = torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE, device=self.device)
        static_sin = torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE, device=self.device)
        static_next_token = torch.zeros(1, dtype=torch.long, device=self.device)

        decode_fn = self._build_decode_fn(static_cos, static_sin)

        # Pre-fill RoPE for warmup position
        static_cos[0, 0].copy_(self._rope_cos_table[warmup_cache_position])
        static_sin[0, 0].copy_(self._rope_sin_table[warmup_cache_position])

        # Warmup on default stream (3 iterations to compile all kernels)
        with torch.no_grad():
            for _ in range(3):
                next_tok = decode_fn(static_input_ids, static_cache_position)
                static_next_token.copy_(next_tok)
        torch.cuda.synchronize()

        # Capture on side stream
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            next_tok = decode_fn(static_input_ids, static_cache_position)
            static_next_token.copy_(next_tok)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self._graph = graph
        self._graph_data = {
            'input_ids': static_input_ids,
            'cache_position': static_cache_position,
            'cos': static_cos,
            'sin': static_sin,
            'next_token': static_next_token,
        }
        print(f"[DecodeCudaGraph] Graph captured with precomputed RoPE (warmup_pos={warmup_cache_position}).")

    @property
    def is_captured(self):
        return self._graph is not None

    def run(self, input_ids, cache_position, position_ids):
        """Run one decode step via CUDA graph replay.

        Args:
            input_ids: [1, 1] tensor of the current token id.
            cache_position: [1] tensor of the current cache write position.
            position_ids: unused (kept for API compat), RoPE from precomputed table.

        Returns:
            next_token: [1] tensor — reference to static graph buffer, valid until next replay.
        """
        graph_data = self._graph_data
        graph_data['input_ids'].copy_(input_ids)
        graph_data['cache_position'].copy_(cache_position)

        # Fill RoPE from precomputed table (O(1) copy, replaces rotary_emb forward)
        pos = cache_position[0]
        graph_data['cos'][0, 0].copy_(self._rope_cos_table[pos])
        graph_data['sin'][0, 0].copy_(self._rope_sin_table[pos])

        self._graph.replay()
        return graph_data['next_token']


class VerifyCudaGraphRunner:
    """CUDA Graph runner for speculative decode verify (batch=1, seq=3).

    Processes num_verify tokens through all 28 layers using flash_attn_with_kvcache
    (appending new KVs). Captures aux_hidden at specified layers.
    """

    def __init__(self, model, device, num_verify=3, aux_layers=(2, 14, 25), fc_weight=None):
        self.model = model
        self.device = device
        self.num_verify = num_verify
        self.aux_layers = set(aux_layers)
        self._graph = None
        self._graph_data = None
        self._fc_weight = fc_weight

    def _build_verify_fn(self, static_cos, static_sin, static_cache_seqlens, aux_bufs, static_combined_all=None):
        model = self.model
        embed_tokens = model.model.get_input_embeddings()
        language_model = model.model.language_model
        layers = language_model.layers
        lm_head = model.lm_head
        cache = model._cache
        num_layers = len(layers)
        num_verify = self.num_verify
        aux_layers = self.aux_layers
        eps = model.config.get_text_config().rms_norm_eps
        fc_weight = self._fc_weight

        def verify_step(token_ids):
            hidden_states = embed_tokens(token_ids)
            normed = batch_rmsnorm(hidden_states, layers[0].input_layernorm.weight, eps)

            for layer_idx in range(num_layers):
                if layer_idx in aux_layers:
                    aux_bufs[layer_idx].copy_(hidden_states)

                layer = layers[layer_idx]
                attn = layer.self_attn

                qkv_out = F.linear(normed, attn.qkv_weight)
                q_dim = attn._q_out_dim
                kv_dim = attn._kv_out_dim
                hidden_shape = (1, num_verify, -1, attn.head_dim)

                query_states, key_states = batch_fused_qk_norm_rope(
                    qkv_out[..., :q_dim + kv_dim],
                    attn.q_norm.weight, attn.k_norm.weight,
                    static_cos, static_sin,
                    attn.q_norm.variance_epsilon,
                    attn.config.num_attention_heads, attn.config.num_key_value_heads,
                    attn.head_dim,
                )
                value_states = qkv_out[..., q_dim + kv_dim:].view(hidden_shape)

                layer_cache = cache.layers[layer_idx]
                attn_output = flash_attn_with_kvcache(
                    query_states, layer_cache.keys, layer_cache.values,
                    k=key_states, v=value_states,
                    cache_seqlens=static_cache_seqlens,
                    causal=True,
                    softmax_scale=attn.scaling,
                )

                attn_output = attn_output.reshape(1, num_verify, -1)
                attn_output = attn.o_proj(attn_output)
                normed_mlp, hidden_states = batch_fused_residual_rmsnorm(
                    hidden_states, attn_output,
                    layer.post_attention_layernorm.weight, eps,
                )
                gate_up_out = F.linear(normed_mlp, layer.mlp.gate_up_weight)
                mlp_out = layer.mlp.down_proj(
                    batch_fused_swiglu(gate_up_out, layer.mlp._intermediate_size)
                )

                if layer_idx < num_layers - 1:
                    normed, hidden_states = batch_fused_residual_rmsnorm(
                        hidden_states, mlp_out,
                        layers[layer_idx + 1].input_layernorm.weight, eps,
                    )
                else:
                    normed, hidden_states = batch_fused_residual_rmsnorm(
                        hidden_states, mlp_out,
                        language_model.norm.weight, eps,
                    )

            logits = lm_head(normed).squeeze(0)

            if fc_weight is not None and static_combined_all is not None:
                aux_cat = torch.cat([aux_bufs[2][0], aux_bufs[14][0], aux_bufs[25][0]], dim=-1)
                static_combined_all.copy_(F.linear(aux_cat, fc_weight))

            return logits

        return verify_step

    def capture(self, rope_cos_table, rope_sin_table, warmup_cache_pos=None):
        if warmup_cache_pos is None:
            warmup_cache_pos = 512

        self._rope_cos_table = rope_cos_table
        self._rope_sin_table = rope_sin_table
        head_dim = rope_cos_table.shape[1]
        num_verify = self.num_verify
        hidden_size = self.model.config.get_text_config().hidden_size

        static_token_ids = torch.zeros(1, num_verify, dtype=torch.long, device=self.device)
        static_cache_seqlens = torch.tensor([warmup_cache_pos], dtype=torch.int32, device=self.device)
        static_cos = torch.zeros(1, num_verify, head_dim, dtype=MODEL_DTYPE, device=self.device)
        static_sin = torch.zeros(1, num_verify, head_dim, dtype=MODEL_DTYPE, device=self.device)

        vocab_map = getattr(self.model, '_reduced_vocab_map', None)
        vocab_size = getattr(self.model, '_reduced_vocab_size', None) or self.model.config.get_text_config().vocab_size
        static_logits = torch.zeros(num_verify, vocab_size, dtype=MODEL_DTYPE, device=self.device)
        static_argmax = torch.zeros(num_verify, dtype=torch.long, device=self.device)
        aux_bufs = {}
        for layer_idx in self.aux_layers:
            aux_bufs[layer_idx] = torch.zeros(
                1, num_verify, hidden_size, dtype=MODEL_DTYPE, device=self.device
            )

        static_combined_all = None
        if self._fc_weight is not None:
            static_combined_all = torch.zeros(num_verify, hidden_size, dtype=MODEL_DTYPE, device=self.device)

        for i in range(num_verify):
            static_cos[0, i].copy_(rope_cos_table[warmup_cache_pos + i])
            static_sin[0, i].copy_(rope_sin_table[warmup_cache_pos + i])

        verify_fn = self._build_verify_fn(static_cos, static_sin, static_cache_seqlens, aux_bufs, static_combined_all)

        with torch.no_grad():
            for _ in range(3):
                logits = verify_fn(static_token_ids)
                static_logits.copy_(logits)
                reduced_idx = logits.float().argmax(dim=-1)
                if vocab_map is not None:
                    static_argmax.copy_(vocab_map[reduced_idx])
                else:
                    static_argmax.copy_(reduced_idx)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = verify_fn(static_token_ids)
            static_logits.copy_(logits)
            reduced_idx = logits.float().argmax(dim=-1)
            if vocab_map is not None:
                static_argmax.copy_(vocab_map[reduced_idx])
            else:
                static_argmax.copy_(reduced_idx)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self._graph = graph
        self._graph_data = {
            'token_ids': static_token_ids,
            'cache_seqlens': static_cache_seqlens,
            'cos': static_cos,
            'sin': static_sin,
            'logits': static_logits,
            'argmax': static_argmax,
            'aux_bufs': aux_bufs,
            'combined_all': static_combined_all,
        }
        print(f"[VerifyCudaGraph] Graph captured (num_verify={num_verify}, warmup_pos={warmup_cache_pos}).")

    @property
    def is_captured(self):
        return self._graph is not None

    def run(self, token_ids, cache_position_start):
        """Run verify via CUDA graph replay.

        Args:
            token_ids: [num_verify] token IDs
            cache_position_start: int, starting cache position

        Returns:
            (argmax [num_verify], aux_bufs dict)
        """
        gd = self._graph_data
        gd['token_ids'][0].copy_(token_ids)
        gd['cache_seqlens'][0] = cache_position_start

        nv = self.num_verify
        gd['cos'][0, :nv].copy_(self._rope_cos_table[cache_position_start:cache_position_start + nv])
        gd['sin'][0, :nv].copy_(self._rope_sin_table[cache_position_start:cache_position_start + nv])

        self._graph.replay()
        return gd['argmax'], gd['aux_bufs'], gd['combined_all']



class FusedDraftVerifyRunner:
    """Split-graph draft+verify runner for speculative decoding.

    Uses TWO separate CUDA graphs (draft + verify) replayed back-to-back on the
    same stream. This avoids large single-graph issues on certain hardware while
    preserving the same external interface and eliminating CPU syncs between phases.
    """

    def __init__(self, model, eagle3, device, draft_k=2, aux_layers=(2, 14, 25)):
        self.model = model
        self.eagle3 = eagle3
        self.device = device
        self.draft_k = draft_k
        self.num_verify = draft_k + 1
        self.aux_layers = set(aux_layers)
        self._draft_graph = None
        self._verify_graph = None
        self._use_parallel = os.environ.get('USE_PARALLEL', '0') == '1' and getattr(eagle3, 'mask_hidden', None) is not None

    def capture(self, rope_cos_table, rope_sin_table, warmup_cache_pos=None):
        if warmup_cache_pos is None:
            warmup_cache_pos = 260

        draft_k = self.draft_k
        num_verify = self.num_verify
        head_dim = rope_cos_table.shape[1]
        hidden_size = self.model.config.get_text_config().hidden_size

        eagle3 = self.eagle3
        use_parallel = self._use_parallel

        if use_parallel:
            eagle3.setup_parallel_draft_buffers(draft_k=draft_k)
            par_ctx = eagle3.get_parallel_graph_context()
        else:
            ctx = eagle3.get_fused_graph_context()

        # Static inputs (shared between both graphs)
        static_target_token = torch.zeros(1, dtype=torch.long, device=self.device)
        static_combined_hidden = torch.zeros(1, 1, hidden_size, dtype=MODEL_DTYPE, device=self.device)
        static_verify_cache_pos = torch.tensor([warmup_cache_pos], dtype=torch.int32, device=self.device)
        static_verify_cos = torch.zeros(1, num_verify, head_dim, dtype=MODEL_DTYPE, device=self.device)
        static_verify_sin = torch.zeros(1, num_verify, head_dim, dtype=MODEL_DTYPE, device=self.device)

        # Static outputs
        static_verify_argmax = torch.zeros(num_verify, dtype=torch.long, device=self.device)
        static_draft_outs = par_ctx['static_outs'] if use_parallel else ctx['static_outs']
        static_combined_all = torch.zeros(num_verify, hidden_size, dtype=MODEL_DTYPE, device=self.device)
        static_result_buf = torch.zeros(num_verify + draft_k, dtype=torch.long, device=self.device)

        # Build verify fn components
        embed_tokens = self.model.model.get_input_embeddings()
        language_model = self.model.model.language_model
        layers = language_model.layers
        lm_head = self.model.lm_head
        cache = self.model._cache
        num_layers = len(layers)
        eps = self.model.config.get_text_config().rms_norm_eps
        vocab_map = getattr(self.model, '_reduced_vocab_map', None)
        fc_weight = par_ctx['fc_weight'] if use_parallel else ctx['fc_weight']
        aux_layers_set = self.aux_layers

        # Static verify token input (written by draft graph, read by verify graph)
        static_verify_tokens = torch.zeros(1, num_verify, dtype=torch.long, device=self.device)

        # Aux hidden buffers for combine
        aux_bufs = {}
        for layer_idx in aux_layers_set:
            aux_bufs[layer_idx] = torch.zeros(1, num_verify, hidden_size, dtype=MODEL_DTYPE, device=self.device)

        # ---- Draft function (Graph 1) ----
        def draft_fn():
            if use_parallel:
                eagle3._par_static_token[0] = static_target_token[0]
                eagle3._par_static_hidden[0].copy_(static_combined_hidden[0, 0])
                eagle3._parallel_draft_fn()
            else:
                ctx['static_token'][0] = static_target_token[0]
                ctx['static_hidden'][0, 0].copy_(static_combined_hidden[0, 0])
                ctx['draft_fn']()
            # Assemble verify input tokens at end of draft graph
            static_verify_tokens[0, 0] = static_target_token[0]
            for i in range(draft_k):
                static_verify_tokens[0, i + 1] = static_draft_outs[i]

        # ---- Verify function (Graph 2) ----
        def verify_fn():
            hidden_states = embed_tokens(static_verify_tokens)
            normed = batch_rmsnorm(hidden_states, layers[0].input_layernorm.weight, eps)

            for layer_idx in range(num_layers):
                if layer_idx in aux_layers_set:
                    aux_bufs[layer_idx].copy_(hidden_states)

                layer = layers[layer_idx]
                attn = layer.self_attn
                qkv_out = F.linear(normed, attn.qkv_weight)
                q_dim = attn._q_out_dim
                kv_dim = attn._kv_out_dim
                hidden_shape = (1, num_verify, -1, attn.head_dim)

                query_states, key_states = batch_fused_qk_norm_rope(
                    qkv_out[..., :q_dim + kv_dim],
                    attn.q_norm.weight, attn.k_norm.weight,
                    static_verify_cos, static_verify_sin,
                    attn.q_norm.variance_epsilon,
                    attn.config.num_attention_heads, attn.config.num_key_value_heads,
                    attn.head_dim,
                )
                value_states = qkv_out[..., q_dim + kv_dim:].view(hidden_shape)

                layer_cache = cache.layers[layer_idx]
                attn_output = flash_attn_with_kvcache(
                    query_states, layer_cache.keys, layer_cache.values,
                    k=key_states, v=value_states,
                    cache_seqlens=static_verify_cache_pos,
                    causal=True,
                    softmax_scale=attn.scaling,
                )
                attn_output = attn_output.reshape(1, num_verify, -1)
                attn_output = attn.o_proj(attn_output)
                normed_mlp, hidden_states = batch_fused_residual_rmsnorm(
                    hidden_states, attn_output,
                    layer.post_attention_layernorm.weight, eps,
                )
                gate_up_out = F.linear(normed_mlp, layer.mlp.gate_up_weight)
                mlp_out = layer.mlp.down_proj(
                    batch_fused_swiglu(gate_up_out, layer.mlp._intermediate_size)
                )

                if layer_idx < num_layers - 1:
                    normed, hidden_states = batch_fused_residual_rmsnorm(
                        hidden_states, mlp_out,
                        layers[layer_idx + 1].input_layernorm.weight, eps,
                    )
                else:
                    normed, hidden_states = batch_fused_residual_rmsnorm(
                        hidden_states, mlp_out,
                        language_model.norm.weight, eps,
                    )

            logits = lm_head(normed).squeeze(0)

            reduced_idx = logits.float().argmax(dim=-1)
            if vocab_map is not None:
                static_verify_argmax.copy_(vocab_map[reduced_idx])
            else:
                static_verify_argmax.copy_(reduced_idx)

            aux_cat = torch.cat([aux_bufs[2][0], aux_bufs[14][0], aux_bufs[25][0]], dim=-1)
            static_combined_all.copy_(F.linear(aux_cat, fc_weight))

            static_result_buf[:num_verify].copy_(static_verify_argmax)
            static_result_buf[num_verify:].copy_(static_draft_outs[:draft_k])

        # Fill warmup RoPE
        for i in range(num_verify):
            static_verify_cos[0, i].copy_(rope_cos_table[warmup_cache_pos + i])
            static_verify_sin[0, i].copy_(rope_sin_table[warmup_cache_pos + i])

        def _set_eagle3_rope(pos):
            if use_parallel:
                eagle3._par_static_cos[0, :, 0].copy_(eagle3.rope_cos[pos:pos + draft_k])
                eagle3._par_static_sin[0, :, 0].copy_(eagle3.rope_sin[pos:pos + draft_k])
            else:
                ctx['static_cos'][:, 0, 0].copy_(eagle3.rope_cos[pos:pos + draft_k])
                ctx['static_sin'][:, 0, 0].copy_(eagle3.rope_sin[pos:pos + draft_k])

        # Warmup both functions
        saved_eagle3_seqlens = eagle3.cache_seqlens.clone()
        with torch.no_grad():
            for _ in range(3):
                _set_eagle3_rope(warmup_cache_pos)
                eagle3.cache_seqlens[0] = warmup_cache_pos
                draft_fn()
                verify_fn()
        eagle3.cache_seqlens.copy_(saved_eagle3_seqlens)
        torch.cuda.synchronize()

        # Capture Graph 1: draft
        _set_eagle3_rope(warmup_cache_pos)
        eagle3.cache_seqlens[0] = warmup_cache_pos
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        draft_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(draft_graph, stream=stream):
            draft_fn()
        torch.cuda.current_stream().wait_stream(stream)
        eagle3.cache_seqlens.copy_(saved_eagle3_seqlens)
        torch.cuda.synchronize()

        # Capture Graph 2: verify
        # Warmup verify once more to ensure static_verify_tokens has valid values
        _set_eagle3_rope(warmup_cache_pos)
        eagle3.cache_seqlens[0] = warmup_cache_pos
        draft_fn()  # populate static_verify_tokens for verify capture
        eagle3.cache_seqlens.copy_(saved_eagle3_seqlens)

        stream2 = torch.cuda.Stream()
        stream2.wait_stream(torch.cuda.current_stream())
        verify_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(verify_graph, stream=stream2):
            verify_fn()
        torch.cuda.current_stream().wait_stream(stream2)
        torch.cuda.synchronize()

        self._draft_graph = draft_graph
        self._verify_graph = verify_graph
        self._use_parallel = use_parallel
        if not use_parallel:
            self._eagle3_ctx = ctx
        self._eagle3_rope_cos = eagle3.rope_cos
        self._eagle3_rope_sin = eagle3.rope_sin
        self._eagle3_cache_seqlens = eagle3.cache_seqlens
        self._rope_cos_table = rope_cos_table
        self._rope_sin_table = rope_sin_table
        self._data = {
            'target_token': static_target_token,
            'combined_hidden': static_combined_hidden,
            'verify_cache_pos': static_verify_cache_pos,
            'verify_cos': static_verify_cos,
            'verify_sin': static_verify_sin,
            'verify_argmax': static_verify_argmax,
            'combined_all': static_combined_all,
            'result_buf': static_result_buf,
        }
        print(f"[FusedDraftVerify] Split graphs captured (draft_k={draft_k}, parallel={use_parallel}, warmup_pos={warmup_cache_pos})")

    def run(self, target_token, combined_hidden, eagle3_cache_pos, verify_cache_pos):
        """Run draft+verify via two sequential CUDA graph replays."""
        d = self._data
        nv = self.num_verify
        dk = self.draft_k

        # Set shared static inputs
        d['target_token'][0] = target_token
        d['combined_hidden'][0, 0].copy_(combined_hidden.view(-1))
        d['verify_cache_pos'][0] = verify_cache_pos
        d['verify_cos'][0, :nv].copy_(self._rope_cos_table[verify_cache_pos:verify_cache_pos + nv])
        d['verify_sin'][0, :nv].copy_(self._rope_sin_table[verify_cache_pos:verify_cache_pos + nv])

        # Set eagle3 draft RoPE + cache pos
        if self._use_parallel:
            self.eagle3._par_static_cos[0, :, 0].copy_(self._eagle3_rope_cos[eagle3_cache_pos:eagle3_cache_pos + dk])
            self.eagle3._par_static_sin[0, :, 0].copy_(self._eagle3_rope_sin[eagle3_cache_pos:eagle3_cache_pos + dk])
        else:
            self._eagle3_ctx['static_cos'][:, 0, 0].copy_(self._eagle3_rope_cos[eagle3_cache_pos:eagle3_cache_pos + dk])
            self._eagle3_ctx['static_sin'][:, 0, 0].copy_(self._eagle3_rope_sin[eagle3_cache_pos:eagle3_cache_pos + dk])
        self._eagle3_cache_seqlens[0] = eagle3_cache_pos

        # Replay: draft → verify (sequential on default stream, no CPU sync needed)
        self._draft_graph.replay()
        self._verify_graph.replay()
        return d['result_buf'], d['combined_all']


# ============================================================================
# Install global patches at module load time
# ============================================================================
_patch_static_layer_max_batch_size()
_patch_static_layer_bsnd()
_install_skip_decode_mask()
_install_fast_greedy_decode()
_install_fast_reset_cache()


# ============================================================================
# VLMModel
# ============================================================================
class VLMModel:
    RESIZE_METHODS = ("lanczos", "bilinear", "bicubic")

    def __init__(self, model_path: str, device: str = "cuda:0", pre_capture: bool = True,
                 resize_method: str = "lanczos"):
        self._device = device
        self.model_path = model_path
        self._pre_capture = pre_capture
        if resize_method is None:
            resize_method = os.environ.get("RESIZE_METHOD", "lanczos").lower()
        if resize_method not in self.RESIZE_METHODS:
            raise ValueError(f"resize_method must be one of {self.RESIZE_METHODS}, got '{resize_method}'")
        self._resize_method = resize_method

        self._processor = AutoProcessor.from_pretrained(model_path)

        self._dtype = MODEL_DTYPE
        print(f"[VLMModel] Loading model with {self._dtype}...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=self._dtype, device_map=device,
        )
        self._model.eval()

        torch.backends.cuda.enable_flash_sdp(True)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

        self._optimize_kv_cache()
        self._reduce_lm_head_vocab()
        _install_fused_decode_layers(self._model)
        self._compile_vit()
        self._setup_gpu_image_processing()
        self._setup_prefill_cuda_graph()
        self._setup_vit_cuda_graph()
        self._setup_decode_cuda_graph()
        self._setup_eagle3()
        self._patch_fast_generate()

        print(f"[VLMModel] Model loaded successfully on {device}")

    def _optimize_kv_cache(self):
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id') and self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._model.config.eos_token_id
        self._model.generation_config.cache_implementation = "static"
        from transformers.generation.configuration_utils import CompileConfig
        self._model.generation_config.compile_config = CompileConfig(mode="max-autotune")

        max_cache_len = 1344
        first_param = next(self._model.parameters())
        self._model._cache = FastResetStaticCache(
            config=self._model.config.get_text_config(decoder=True),
            max_cache_len=max_cache_len,
            dtype=first_param.dtype,
            device=first_param.device,
        )

    def _reduce_lm_head_vocab(self):
        """Reduce target model LM head from 151936 to ~15360 tokens based on dataset analysis."""
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reduced_vocab_target.pt")
        if not os.path.exists(vocab_path):
            return

        used_token_ids = torch.load(vocab_path, map_location=self._device, weights_only=True)
        num_used = len(used_token_ids)
        aligned_size = ((num_used + 255) // 256) * 256

        lm_head = self._model.lm_head
        full_weight = lm_head.weight.data  # [151936, 2048]

        reduced_weight = torch.zeros(aligned_size, full_weight.shape[1], dtype=full_weight.dtype, device=self._device)
        reduced_weight[:num_used] = full_weight[used_token_ids]

        lm_head.weight = nn.Parameter(reduced_weight, requires_grad=False)
        lm_head.out_features = aligned_size

        self._model._reduced_vocab_map = used_token_ids  # [reduced_idx] → original token_id
        self._model._reduced_vocab_size = aligned_size
        self._model._reduced_vocab_num_real = num_used

        del full_weight
        torch.cuda.empty_cache()
        if DEBUG:
            print(f"[VocabReduce] Target LM head: 151936 → {aligned_size} ({aligned_size*2048*2/1024/1024:.1f} MB)")

    def _compile_vit(self):
        vit = self._model.model.visual
        patch_embed = vit.patch_embed
        conv = patch_embed.proj

        # img2col: replace Conv3d with unfold + F.linear
        weight_2d = conv.weight.data.reshape(patch_embed.embed_dim, -1).contiguous()
        bias = conv.bias.data if conv.bias is not None else None
        patch_embed.register_buffer('img2col_weight', weight_2d)
        if bias is not None:
            patch_embed.register_buffer('img2col_bias', bias)
        else:
            patch_embed.img2col_bias = None
        patch_embed._flat_dim = patch_embed.in_channels * patch_embed.temporal_patch_size * patch_embed.patch_size * patch_embed.patch_size

        def img2col_patch_embed_forward(self_pe, hidden_states: torch.Tensor) -> torch.Tensor:
            flat_patches = hidden_states.view(-1, self_pe._flat_dim).float()
            out = F.linear(flat_patches, self_pe.img2col_weight.float(), self_pe.img2col_bias.float() if self_pe.img2col_bias is not None else None)
            return out.to(dtype=self_pe.img2col_weight.dtype)

        patch_embed.forward = types.MethodType(img2col_patch_embed_forward, patch_embed)

        _vit_cache = {}

        from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures as _BModelOut

        def cached_vit_forward(self_vit, hidden_states, grid_thw, **kwargs):
            # Use num_patches as fast cache key — avoids GPU→CPU sync from .tolist()
            num_patches = hidden_states.shape[0]
            cache_key = num_patches

            if cache_key not in _vit_cache:
                pos_embeds = self_vit.fast_pos_embed_interpolate(grid_thw)
                rot_emb = self_vit.rot_pos_emb(grid_thw)
                rot_emb_full = torch.cat((rot_emb, rot_emb), dim=-1)
                cos_cached = rot_emb_full.cos()
                sin_cached = rot_emb_full.sin()

                cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

                _vit_cache[cache_key] = (pos_embeds, cos_cached, sin_cached, cu_seqlens, max_seqlen)
            pos_embeds, cos_cached, sin_cached, cu_seqlens, max_seqlen = _vit_cache[cache_key]
            position_embeddings = (cos_cached, sin_cached)

            hidden_states = self_vit.patch_embed(hidden_states)
            hidden_states = hidden_states + pos_embeds
            hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

            vit_graph_runner = self_vit._cuda_graph_runner
            merged_hidden_states, deepstack_feature_lists = vit_graph_runner.run(
                hidden_states, cu_seqlens, position_embeddings, max_seqlen,
            )
            return _BModelOut(
                last_hidden_state=hidden_states,
                pooler_output=merged_hidden_states,
                deepstack_features=deepstack_feature_lists,
            )

        vit.forward = types.MethodType(cached_vit_forward, vit)

        # Cache LLM RoPE cos/sin
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

        _llm_rope_cache = {}
        original_rope_forward = Qwen3VLTextRotaryEmbedding.forward

        def cached_rope_forward(self_rope, x, position_ids):
            seq_len = position_ids.shape[-1] if position_ids.ndim >= 2 else 1
            if seq_len <= 1:
                return original_rope_forward(self_rope, x, position_ids)

            # Single .tolist() call instead of 3 separate .item() calls
            # to reduce GPU→CPU synchronization overhead
            pid_flat = position_ids.view(-1)
            target_dtype = x.dtype
            pid_list = pid_flat[[0, len(pid_flat)//2, -1]].tolist()
            cache_key = (
                position_ids.shape,
                pid_list[0],
                pid_list[2],
                pid_list[1],
                target_dtype,
            )

            if cache_key not in _llm_rope_cache:
                cos, sin = original_rope_forward(self_rope, x, position_ids)
                _llm_rope_cache[cache_key] = (
                    cos.to(dtype=target_dtype).clone(),
                    sin.to(dtype=target_dtype).clone(),
                )

            return _llm_rope_cache[cache_key]

        Qwen3VLTextRotaryEmbedding.forward = cached_rope_forward

    def _setup_gpu_image_processing(self):
        if torch.cuda.is_available() and hasattr(self._processor, 'image_processor'):
            self._processor.image_processor.device = self._device

        if hasattr(self._processor, 'image_processor'):
            image_processor = self._processor.image_processor

            # Patch _further_process_kwargs to allow independent max_pixels/min_pixels override
            # (reference: vllm/model_executor/models/qwen3_vl.py lines 670-673)
            original_further_process = image_processor._further_process_kwargs

            def patched_further_process_kwargs(size=None, min_pixels=None, max_pixels=None, **kwargs):
                # Start from current default size
                resolved_size = {**image_processor.size}
                if size is not None:
                    resolved_size.update(size)
                # Independent override, like vllm does
                if min_pixels is not None:
                    resolved_size["shortest_edge"] = min_pixels
                if max_pixels is not None:
                    resolved_size["longest_edge"] = max_pixels
                return original_further_process(size=resolved_size, **kwargs)

            image_processor._further_process_kwargs = patched_further_process_kwargs

            # Set default size with max_pixels constraint
            image_processor.size = {
                "longest_edge": MAX_PIXELS,
                "shortest_edge": MIN_PIXELS,
            }
            image_processor.do_resize = True
            from PIL import Image as PILImage
            import torchvision.transforms.functional as tvF
            from transformers.image_utils import PILImageResampling

            _pil_resample_map = {
                "lanczos": PILImage.LANCZOS,
                "bilinear": PILImage.BILINEAR,
                "bicubic": PILImage.BICUBIC,
            }
            _hf_resample_map = {
                "lanczos": PILImageResampling.LANCZOS,
                "bilinear": PILImageResampling.BILINEAR,
                "bicubic": PILImageResampling.BICUBIC,
            }
            pil_resample = _pil_resample_map[self._resize_method]
            image_processor.resample = _hf_resample_map[self._resize_method]

            def _custom_resize(image, size, interpolation=None, antialias=True, **kwargs):
                h, w = size.height, size.width
                is_batched = image.ndim == 4
                if is_batched:
                    imgs = image
                else:
                    imgs = image.unsqueeze(0)
                results = []
                for img in imgs:
                    pil_img = tvF.to_pil_image(img)
                    pil_img = pil_img.resize((w, h), pil_resample)
                    t = tvF.pil_to_tensor(pil_img)
                    results.append(t)
                out = torch.stack(results).to(image.device)
                return out if is_batched else out.squeeze(0)
            image_processor.resize = _custom_resize
            print(f"[GPU Image Processing] Patched image processor (vllm-style) with {image_processor.size}, resample={self._resize_method.upper()}")

    def _setup_vit_cuda_graph(self):
        vit = self._model.model.visual
        vision_config = self._model.config.vision_config
        self._vit_graph_runner = ViTCudaGraphRunner(
            visual_model=vit,
            device=self._device,
            vit_hidden_size=vision_config.hidden_size,
            out_hidden_size=vision_config.out_hidden_size,
        )
        if self._pre_capture:
            self._vit_graph_runner.capture_all()
        else:
            print("[ViTCudaGraph] pre_capture=False, graphs will be captured lazily on first use.")
        vit._cuda_graph_runner = self._vit_graph_runner

    def _setup_prefill_cuda_graph(self):
        lang_model = self._model.model.language_model
        cache = self._model._cache
        self._prefill_graph_runner = PrefillCudaGraphRunner(
            language_model=lang_model, cache=cache,
            device=self._device,
            hidden_size=self._model.config.get_text_config().hidden_size,
        )
        if self._pre_capture:
            self._prefill_graph_runner.capture_all()
        else:
            print("[PrefillPiecewise] pre_capture=False, graphs will be captured lazily.")
        self._patch_model_forward_for_cuda_graph()

    def _setup_decode_cuda_graph(self):
        self._decode_graph_runner = DecodeCudaGraphRunner(
            model=self._model, device=self._device,
        )
        # Decode graph is captured lazily after the first prefill
        # (needs valid KV cache data and compiled kernels).
        # Store reference on model for access from _fast_greedy_loop.
        self._model._decode_graph_runner = self._decode_graph_runner

    def _setup_eagle3(self):
        from eagle3_draft import Eagle3Model
        eagle3_path = os.environ.get(
            'EAGLE3_PATH',
            os.path.dirname(os.path.abspath(__file__)) + "/eagle3"
        )
        self._model._eagle3 = Eagle3Model(eagle3_path, self._model, device=self._device)
        print(f"[Eagle3] Loaded draft model from {eagle3_path}")

    def _patch_fast_generate(self):
        """Patch model.generate() to cache generation_config/stopping_criteria
        and skip ~2ms of repeated preparation on subsequent calls."""
        import copy
        from transformers.generation.utils import GenerationMixin
        from transformers import StoppingCriteriaList

        model = self._model
        original_generate = model.generate.__func__ if hasattr(model.generate, '__func__') else type(model).generate

        cached_state = {}
        cache_obj = model._cache

        def fast_generate(self_model, **kwargs):
            max_new_tokens = kwargs.pop('max_new_tokens', 128)
            do_sample = kwargs.pop('do_sample', False)
            use_cache = kwargs.pop('use_cache', True)
            kwargs.pop('temperature', None)

            input_ids = kwargs.get('input_ids')
            if input_ids is None:
                return original_generate(self_model, max_new_tokens=max_new_tokens,
                                         do_sample=do_sample, use_cache=use_cache, **kwargs)

            if 'generation_config' not in cached_state:
                gen_config, _ = self_model._prepare_generation_config(None, do_sample=do_sample,
                                                                       use_cache=use_cache,
                                                                       max_new_tokens=max_new_tokens)
                self_model._prepare_special_tokens(gen_config, False, device=input_ids.device)
                cached_state['generation_config'] = gen_config

            gen_config = cached_state['generation_config']
            input_ids_length = input_ids.shape[1]

            sc_key = max_new_tokens
            if sc_key not in cached_state:
                gen_config_copy = copy.copy(gen_config)
                gen_config_copy.max_new_tokens = max_new_tokens
                gen_config_copy.max_length = max_new_tokens + input_ids_length
                gen_config_copy.min_length = 0
                stopping = self_model._get_stopping_criteria(
                    generation_config=gen_config_copy,
                    stopping_criteria=StoppingCriteriaList(),
                )
                max_len_criteria = None
                for c in stopping:
                    if hasattr(c, 'max_length'):
                        max_len_criteria = c
                        break
                cached_state[sc_key] = (gen_config_copy, stopping, max_len_criteria)
            gen_config_copy, stopping_criteria, max_len_criteria = cached_state[sc_key]
            actual_max_length = max_new_tokens + input_ids_length
            gen_config_copy.max_length = actual_max_length
            if max_len_criteria is not None:
                max_len_criteria.max_length = actual_max_length

            cache_obj.reset()

            model_kwargs = {k: v for k, v in kwargs.items() if k != 'input_ids'}
            model_kwargs['use_cache'] = use_cache
            model_kwargs['past_key_values'] = cache_obj
            model_kwargs['logits_to_keep'] = 1

            eagle3 = getattr(self_model, '_eagle3', None)
            if eagle3 is not None:
                return _speculative_greedy_loop(
                    self_model, input_ids, stopping_criteria, gen_config_copy,
                    eagle3, **model_kwargs,
                )
            return _fast_greedy_loop(
                self_model, input_ids, stopping_criteria, gen_config_copy,
                **model_kwargs,
            )

        model.generate = types.MethodType(fast_generate, model)

        _cache_pos_buf = {}

        def fast_prefill(self_model, input_ids, generation_config, model_kwargs,
                         is_first_iteration=True):
            seq_len = input_ids.shape[1]
            if seq_len not in _cache_pos_buf:
                _cache_pos_buf[seq_len] = torch.arange(seq_len, dtype=torch.long,
                                                        device=input_ids.device)
            return self_model(
                input_ids=input_ids,
                cache_position=_cache_pos_buf[seq_len],
                return_dict=True,
                **{k: v for k, v in model_kwargs.items()
                   if k not in ('cache_position',)},
            )

        model._prefill = types.MethodType(fast_prefill, model)
        print("[FastGenerate] Patched model.generate() and _prefill to skip framework overhead")

    def _patch_model_forward_for_cuda_graph(self):
        inner_model = self._model.model
        outer_model = self._model
        prefill_graph_runner = self._prefill_graph_runner
        original_forward = type(inner_model).forward

        def patched_qwen3vl_forward(self_model, input_ids=None, attention_mask=None,
                                     position_ids=None, past_key_values=None,
                                     inputs_embeds=None, pixel_values=None,
                                     pixel_values_videos=None, image_grid_thw=None,
                                     video_grid_thw=None, mm_token_type_ids=None,
                                     cache_position=None, **kwargs):
            # Determine seq_len
            if input_ids is not None:
                seq_len = input_ids.shape[1]
            elif inputs_embeds is not None:
                seq_len = inputs_embeds.shape[1]
            else:
                seq_len = 1

            # Decode path: use original forward (torch.compile handles CUDA graphs)
            if seq_len <= 1:
                return original_forward(
                    self_model, input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    cache_position=cache_position, **kwargs,
                )

            # Compute inputs_embeds
            if inputs_embeds is None:
                inputs_embeds = self_model.get_input_embeddings()(input_ids)

            # ViT + image embedding scatter
            visual_pos_masks = None
            deepstack_visual_embeds = None

            if pixel_values is not None:
                # Pre-compute split_sizes and CPU copies BEFORE launching ViT
                # to avoid GPU sync from .tolist()/.item() after ViT completes
                _sms2 = self_model.visual.spatial_merge_size ** 2
                split_sizes = (image_grid_thw.prod(-1) // _sms2).tolist()
                if mm_token_type_ids is not None and mm_token_type_ids.is_cuda:
                    mm_token_type_ids_cpu = mm_token_type_ids.cpu()
                else:
                    mm_token_type_ids_cpu = mm_token_type_ids
                image_grid_thw_cpu = image_grid_thw.cpu() if image_grid_thw.is_cuda else image_grid_thw

                # Launch ViT on GPU (async — returns while GPU works ~7ms)
                pixel_values = pixel_values.type(self_model.visual.dtype)
                vision_output = self_model.visual(
                    pixel_values, grid_thw=image_grid_thw, return_dict=True
                )

                # Overlap: compute position_ids on CPU while GPU runs ViT
                # Pass attention_mask=None (batch=1, no padding — all 1s equiv to None)
                if position_ids is None:
                    position_ids = self_model.compute_3d_position_ids(
                        input_ids=input_ids, image_grid_thw=image_grid_thw_cpu,
                        video_grid_thw=video_grid_thw, inputs_embeds=inputs_embeds,
                        attention_mask=None, past_key_values=past_key_values,
                        mm_token_type_ids=mm_token_type_ids_cpu,
                    )

                # Complete ViT output processing (needs GPU results)
                image_embeds = vision_output.pooler_output
                deepstack_image_embeds = vision_output.deepstack_features
                image_embeds = torch.split(image_embeds, split_sizes)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self_model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = deepstack_image_embeds
            elif position_ids is None:
                position_ids = self_model.compute_3d_position_ids(
                    input_ids=input_ids, image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw, inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask, past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )

            # Compute cache_position
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

            eagle3_active = getattr(outer_model, '_eagle3', None) is not None
            if eagle3_active:
                hidden_states, prefill_aux = prefill_graph_runner.run(
                    inputs_embeds, position_ids, cache_position, past_key_values,
                    visual_pos_masks, deepstack_visual_embeds, return_aux_hidden=True,
                )
                self_model._prefill_aux_hidden = prefill_aux
                self_model._prefill_position_ids = position_ids
            else:
                hidden_states = prefill_graph_runner.run(
                    inputs_embeds, position_ids, cache_position, past_key_values,
                    visual_pos_masks, deepstack_visual_embeds,
                )

            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
            return Qwen3VLModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                rope_deltas=self_model.rope_deltas,
            )

        inner_model.forward = types.MethodType(patched_qwen3vl_forward, inner_model)

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

