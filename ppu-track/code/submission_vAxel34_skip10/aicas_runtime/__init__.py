"""aicas_runtime — modular runtime for vAxel30 (Qwen3-VL-2B on PPU).

Decomposes vAxel29's 1488-line monolith into focused modules:
  constants, custom_ops, chat_template, vit_patches, llm_fusion,
  vit_forward, prefill, decode, vocab_whitelist.

Bug fixes applied vs vAxel29:
  B1: torch.compile mode 'max-autotune-no-cudagraphs' → 'default' (cold-cache
      autotune sweep was the dominant first-sample latency on the harness).
  B2: dropped dead `_compiled_prefill_layer` — never invoked.
  B4: torch._dynamo.config.recompile_limit 64 → 8.
"""
from . import constants
from .constants import (
    AICAS_QUANT_ENABLE, AICAS_QUANT_LM_HEAD, AICAS_QUANT_DECODE_LLM,
    AICAS_QUANT_VIT_FC1, AICAS_QUANT_VIT_FC1_LO, AICAS_QUANT_VIT_FC1_HI,
    AICAS_QUANT_VIT_CALIBRATE, AICAS_QUANT_N_CALIB, AICAS_QUANT_VIT_ALPHA,
    AICAS_QUANT_TEXTVQA_DIR,
    AICAS_VOCAB_WHITELIST, AICAS_VOCAB_K, AICAS_VOCAB_TEXTVQA_DIR,
    AICAS_VOCAB_CORPUS,
    AICAS_MAX_PIXELS, PRUNE_CONFIG,
    MAX_CACHE_LEN, MAX_CACHE_LEN_ACC, N_DECODE_STEPS,
    NUM_LLM_LAYERS, NUM_KV_HEADS, HEAD_DIM, NUM_Q_HEADS,
    IM_START_ID, IM_END_ID, VISION_START_ID, VISION_END_ID,
    IMAGE_PAD_ID, NEWLINE_ID, USER_ID, ASSISTANT_ID,
    CHAT_PREFIX, CHAT_SUFFIX_BEFORE_Q, CHAT_SUFFIX_AFTER_Q,
)
from .custom_ops import flash_attn_kvcache_op, flash_attn_vit_op
from .chat_template import make_fast_apply_chat_template
from .vit_patches import (
    make_fast_pos_embed_interpolate,
    make_fast_rot_pos_emb,
    make_fast_get_rope_index,
)
from .llm_fusion import install_llm_fusions
from .vit_forward import make_manual_vit_forward_fused
from .prefill import prefill_one_layer, make_compiled_full_prefill, make_compiled_full_prefill_hidden, make_eager_full_prefill_hidden
from .decode import make_decode_fns, capture_decode_graphs
from .fast_decode import make_fast_decode_fns
from .vocab_whitelist import build_vocab_whitelist
