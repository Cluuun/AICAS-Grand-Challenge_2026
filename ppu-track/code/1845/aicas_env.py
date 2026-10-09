"""
AICAS 2026 — 环境变量配置 & 默认值管理

通过环境变量控制所有优化模块的开关与参数，支持：
- BASE_DEFAULTS: 默认配置（TTFT/profile 模式）
- TTFT_LOCK_OVERRIDES: TTFT 锁定模式下强制覆盖的开关
- SAFE_PROFILE_OVERRIDES: AICAS_PROFILE=safe 时禁用重优化的安全配置

每个环境变量的注释说明其控制的优化模块及影响范围。
标记 # unused 的为暂未接入代码的遗留开关，已注释掉。
"""

from __future__ import annotations

import os
from typing import Mapping

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n"}


def _norm(value: object) -> str:
    return str(value).strip().lower()


# ---------------------------------------------------------------------------
# 公共读取函数
# ---------------------------------------------------------------------------

def get_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值，仅在本文件内部使用."""
    value = os.getenv(name)
    if value is None:
        return default
    lowered = _norm(value)
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    return default


def get_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """从环境变量读取整数，evaluation_wrapper 和 flashdecodeffn 使用."""
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def get_choice(name: str, default: str, choices: set[str]) -> str:
    """从环境变量读取受限制的字符串选项，仅在本文件内部使用."""
    value = _norm(os.getenv(name, default))
    return value if value in choices else default


# ---------------------------------------------------------------------------
# 内部辅助：默认值、别名、FlashDecode 模式快捷开关
# ---------------------------------------------------------------------------

def _set_default(name: str, value: object) -> None:
    os.environ.setdefault(name, str(value))


def _set_defaults(defaults: Mapping[str, object]) -> None:
    for name, value in defaults.items():
        _set_default(name, value)

# ===========================================================================
# BASE_DEFAULTS — 所有优化模块的默认配置
# ===========================================================================
# 使用 setdefault 语义：仅在环境变量未设置时生效，用户可通过 export 覆盖。
# 每个变量注释格式：<模块名> — <作用说明> — <取值/默认值>

BASE_DEFAULTS = {
    "AICAS_VISION_COMPILE_MODE" : "default",


    # ---- PyTorch 系统级 ----
    # PyTorch CUDA 内存分配器配置, expandable_segments 减少显存碎片
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",

    # ---- Prefill 阶段 ----
    # Prefill Rotary: 用 Triton kernel 替换 Qwen3-VL 原生 apply_rotary_pos_emb
    "AICAS_PREFILL_ROTARY_TRITON": "1",
    # Prefill RMSNorm: Triton fused RMSNorm 替换 Prefill 阶段的 LayerNorm (28 layers)
    "AICAS_PREFILL_RMSNORM_TRITON": "1",
    # Prefill RMSNorm CUDA extension: experimental A/B path, default off until validated.
    "AICAS_PREFILL_RMSNORM_CUDA": "0",
    # Prefill FlashAttention: 强制使用内置 SDPA (FlashAttention backend)
    "AICAS_FORCE_BUILTIN_SDPA": "1",
    # Decode RoPE cache: experimental, replace single-token decode cos/sin with table lookup.
    "AICAS_DECODE_ROPE_CACHE": "1",
    "AICAS_DECODE_ROPE_CACHE_MAX_LEN": "4096",
    # Cap validation images to the 320K-pixel bucket profile.  This keeps
    # random-sample TTFT bounded while preserving enough visual detail for the
    # current accuracy-acceptable configuration.
    "AICAS_IMAGE_MAX_PIXELS": "327680",

    # ---- KV Cache ----
    # Merged KV cache: Radix tree prefix reuse + Image KV cache
    # 控制 text prefix cache (radix tree, 跨请求共享相同 prompt 前缀的 KV)
    "AICAS_ENABLE_PREFIX_KVCACHE": "1",
    # 控制 image KV cache (哈希匹配, 跨请求共享相同图像的 KV)
    "AICAS_ENABLE_IMAGE_KVCACHE": "1",
    # Force full prefill (disable cache for debugging)
    "AICAS_KV_PREFIX_CACHE_FORCE_FULL": "0",
    # Exact full-prompt KV shortcut: experimental, skips partial prefill replay on full hit.
    "AICAS_KVCACHE_FULL_HIT_SHORTCUT": "1",
    # Debug logging for merged KV cache
    "AICAS_KV_CACHE_DEBUG": "0",
    # Keep CUDA graph pools and KV cache from competing for all free memory.
    "AICAS_KV_CACHE_MEMORY_FRACTION": "0.12",
    "AICAS_KVCACHE_FULL_HIT_MEMORY_FRACTION": "0.1",
    # TTFT only needs the full-prompt shortcut for the immediately following
    # throughput call; avoid extra image/radix KV clones on TTFT misses.
    "AICAS_KVCACHE_TTFT_STORE_FULL_ONLY": "1",
    "AICAS_KVCACHE_TTFT_FULL_ONLY_REUSE": "1",
    # Max entries in Layer 1 (image KV cache)
    "AICAS_KV_CACHE_MAX_IMAGE_ENTRIES": "160",
    # Keep random-sample performance independent of fixed-order dataset prewarm.
    "AICAS_IMAGE_KV_PREWARM": "0",
    "AICAS_IMAGE_KV_PREWARM_SAMPLES": "150",
    # Max blocks in Layer 2 radix tree (block_size=16 tokens per block)
    "AICAS_KV_CACHE_MAX_BLOCKS": "256",
    # ---- TTFT Prefill CUDA Graph (TTFT 加速) ----
    "AICAS_TTFT_PRECAPTURE_SHAPE_LOG" : "shape_log.json",
    # 统一 CUDA Graph 开关：0=禁用全部(prefill+decode), 1=启用
    "AICAS_CUDA_GRAPH": "1",
    # 是否启用 Prefill TTFT CUDA Graph (legacy, 由 AICAS_CUDA_GRAPH 统一控制)
    "AICAS_ENABLE_PREFILL_TTFT_GRAPH": "1",
    # on-demand 捕获：首次遇到不会命中 graph 的 prompt 长度时自动捕获新 graph
    "AICAS_PREFILL_TTFT_CAPTURE_ON_DEMAND": "0",
    # on-demand 捕获时是否专门为 TTFT（1 token）做捕获
    "AICAS_PREFILL_TTFT_CAPTURE_ON_DEMAND_TTFT": "0",
    # TTFT bucket 策略: "ceil" 向上取整到最近的 bucket, "exact" 精确匹配
    "AICAS_PREFILL_TTFT_BUCKET_MODE": "ceil",
    # "ceil" 模式下最大允许的 padding 长度
    "AICAS_PREFILL_TTFT_MAX_PAD": "64",
    # 是否也为非 TTFT 的 max_new_tokens 做捕获
    "AICAS_PREFILL_TTFT_CAPTURE_NON_TTFT": "0",
    # warmup 阶段捕获时使用的 max_new_tokens
    "AICAS_PREFILL_TTFT_CAPTURE_WARMUP_MAX_NEW_TOKENS": "10",
    # TTFT graph cache 的最大条目数
    "AICAS_PREFILL_TTFT_GRAPH_MAX_ENTRIES": "320",
    # 调试用：捕获 prefill graph 后额外跑 eager next-token 校验；正式 benchmark 默认关闭。
    "AICAS_PREFILL_GRAPH_VALIDATE": "0",
    # TTFT graph 是否 bypass vision CUDA graph（独立 prefill）
    "AICAS_PREFILL_TTFT_BYPASS_VISION_GRAPH": "0",
    # 是否从 dataset 预捕获 prefill graph（init 时扫描前 N 个样本）
    "AICAS_PREFILL_TTFT_PRECAPTURE_DATASET": "1",
    # dataset 预捕获时扫描的 unique shape 数；0=捕获 shape_log 里的全部
    # unique (image_grid_thw, prompt_len)，避免随机长尾样本首遇时走 eager。
    "AICAS_PREFILL_TTFT_PRECAPTURE_SAMPLES": "0",
    # Warm remaining TTFT shapes with eager dummy forwards so random samples do
    # not pay first-seen compile/autotune cost when they miss the graph cache.
    "AICAS_PREFILL_TTFT_EAGER_WARMUP_MISSES": "0",
    # 0=warm all profile shapes not already captured as CUDA graphs.
    "AICAS_PREFILL_TTFT_EAGER_WARMUP_SAMPLES": "0",
    # Add shapes from a deterministic random sample to avoid optimizing only
    # the most frequent or first-N dataset order.
    "AICAS_PREFILL_TTFT_PRECAPTURE_RANDOM_SAMPLES": "0",
    "AICAS_PREFILL_TTFT_PRECAPTURE_RANDOM_SEED": "20260608",
    # KV-cache-hit partial prefill graph shapes generated by
    # scripts/generate_partial_prefill_shape_log.py.
    "AICAS_PARTIAL_PREFILL_SHAPE_LOG": "partial_prefill_shape_log.json",
    # 是否启用带 past_key_values/cache_position 的 partial prefill CUDA graph.
    "AICAS_PARTIAL_PREFILL_GRAPH": "0",
    # partial prefill graph 预捕获 shape 数；0=捕获 profile 内全部 unique shape.
    "AICAS_PARTIAL_PREFILL_PRECAPTURE_SAMPLES": "0",
    # 对哪些 max_new_tokens 预捕获 partial graph；TTFT=1, Throughput=128.
    "AICAS_PARTIAL_PREFILL_PRECAPTURE_MAX_NEW": "1,128",
    # Benchmark dataset path (used for pre-capture)
    "AICAS_DATASET_PATH": "/root/data",
    # TTFT bucket 列表（逗号分隔的 prompt 长度）
    "AICAS_PREFILL_TTFT_BUCKETS": (
        "448,480,512,544,576,608,640,672,704,736,768,800,832,864,"
        "896,928,960,992,1024,1044,1056,1088,1120,1152,1184,1216"
    ),

    # ---- MLP 通道剪枝 ----
    # 是否启用 channel selective MLP (基于 mlp_channel_indices.pt)
    "AICAS_ENABLE_MLP_PRUNE": "0",
    "AICAS_MLP_PRUNE_MAX_KEEP": "0.15",
    "AICAS_MLP_PRUNE_ACCURACY_FULL": "0",
    "AICAS_ACCURACY_MIN_NEW_TOKENS": "512",
    "AICAS_ACCURACY_BASELINE_MODEL": "0",
    # Exact text projection fusion: merge same-input GEMVs without changing math.
    "AICAS_TEXT_QKV_FUSION": "0",
    "AICAS_TEXT_MLP_GATE_UP_FUSION": "1",

    # ---- Prefill 阶段 FlashDecode 禁用 ----
    # 捕获 CUDA Graph 期间是否禁用 FlashDecode（避免图捕获 break）
    "AICAS_PREFILL_TTFT_DISABLE_FLASHDECODE_DURING_CAPTURE": "0",

    # ---- Vision Encoder ----
    # 是否启用 connector merger patch（视觉特征融合优化）
    "AICAS_ENABLE_MERGER_PATCH": "0",
    # Vision Transformer block 是否使用 inplace residual (add_)
    "AICAS_VISION_INPLACE_RESIDUAL": "0",
    # Vision patch_embed Conv3d 是否替换为等价 flat Linear（固定 patch shape 专用）
    "AICAS_VISION_PATCH_EMBED_LINEAR": "1",
    # Vision CUDA Graph 是否 clone 输出（安全性 vs 性能）
    "AICAS_VISION_GRAPH_CLONE_OUTPUT": "0",
    # Vision torch.compile 是否使用 dynamic shape
    "AICAS_VISION_COMPILE_DYNAMIC": "1",
    "AICAS_VISION_LAYER_DROP": "1",
    "AICAS_VISION_LAYER_DROP_LAYERS": "0-4,6-10,12-16,18-23",
    "AICAS_VISION_LAYER_DROP_CONDITIONAL": "1",
    # ---- TTFT Lock Profile ----
    # 是否启用 TTFT 锁定模式（强制覆盖部分开关以优化 TTFT）
    "AICAS_TTFT_LOCK_PROFILE": "1",

    # ---- Decode Graph (CUDA Graph for Decode Loop) ----
    # TTFT 场景是否使用原始 decode（不经过 CUDA Graph）
    "AICAS_DECODE_GRAPH_TTFT_USE_ORIG": "0",
    # TTFT 场景是否禁用 decode graph cache 复用
    "AICAS_DECODE_GRAPH_TTFT_DISABLE_CACHE": "0",
    # Prefill→Decode 过渡时是否复用 Prefill graph cache
    "AICAS_DECODE_GRAPH_PREFILL_CACHE_REUSE": "1",
    # Prefill cache 池的最大条目数
    "AICAS_DECODE_GRAPH_PREFILL_CACHE_POOL_MAX_ENTRIES": "4",
    # Decode graph chunk steps（分步执行 chunk 数）
    "AICAS_DECODE_GRAPH_CHUNK_STEPS": "1",
    # Shape-log derived decode cache buckets for benchmark prompt_len + 128.
    "AICAS_DECODE_GRAPH_MCL_BUCKETS": "448,512,576,640,704,768,832,896,960,1024,1088,1152,1216",
    # EAGLE3 tree cache select compact copy is experimental; keep off by default.
    "AICAS_EAGLE3_COMPACT_CACHE_SELECT": "1",
    # Fuse compact cache_select path KV gather + short tail clear into one kernel/layer.
    "AICAS_EAGLE3_COMPACT_CACHE_SELECT_FUSED": "1",
    # Refresh benchmark precapture entries last so they survive the draft graph LRU.
    "AICAS_EAGLE3_SPEC_PRECAPTURE_REFRESH_BENCHMARK_LAST": "1",
    # Let wrapper-owned pre-benchmark warmups fill missing EAGLE3 CUDA graphs.
    "AICAS_EAGLE3_INTERNAL_WARMUP_ALLOW_GRAPH_CAPTURE": "1",
    # Add representative cache-bucket probes after benchmark/profile shapes.
    "AICAS_EAGLE3_SPEC_PRECAPTURE_INCLUDE_BUCKET_PROBES": "1",
    # Allow short pre-benchmark spec calls to continue capturing missing graphs.
    "AICAS_EAGLE3_PRECAPTURE_ON_DEMAND_MAX_NEW_TOKENS": "16",
    # 是否使用 null attention mask（兼容 CUDA Graph 捕获）
    "AICAS_DECODE_GRAPH_NULL_ATTN_MASK": "1",
    # sanity 校验 max_new_tokens（开发调试用, 0 关闭）
    "AICAS_DECODE_GRAPH_SANITY_VALIDATE_MAX_NEW_TOKENS": "0",
    # warmup 安全 max_new_tokens 上限
    "AICAS_DECODE_GRAPH_WARMUP_SAFE_MAX_NEW_TOKENS": "16",

    # ---- Decode 通用 ----
    # 是否强制 warmup min_new_tokens（保证首次 inference 正确性）
    "AICAS_FORCE_WARMUP_MIN_NEW_TOKENS": "1",
    # Decode fastpath 最小 max_new_tokens（低于此值走 fastpath）
    "AICAS_DECODE_FASTPATH_MIN_NEW_TOKENS": "2",
    # Decode fastpath 最大 max_new_tokens（高于此值不走 fastpath）
    "AICAS_DECODE_FASTPATH_MAX_NEW_TOKENS": "128",
    # Decode CUDA Graph 最小 max_new_tokens
    "AICAS_DECODE_CUDAGRAPH_MIN_NEW_TOKENS": "2",
    # Decode CUDA Graph 最大 max_new_tokens
    "AICAS_DECODE_CUDAGRAPH_MAX_NEW_TOKENS": "128",
    # Pre-capture benchmark decode buckets so measured throughput avoids lazy graph capture.
    "AICAS_DECODE_CUDAGRAPH_PRECAPTURE": "1",
    "AICAS_DECODE_ADD_RMSNORM_TRITON": "1",

    # ---- unused: 未接入代码的遗留开关 ----
    # "AICAS_FLASHDECODE_MAXS_BUCKETS": "576,640,704,768,832,896,960,1024,1088,1152,1280,1408,1536,1792,2112",

    # ---- Decode Linear（Attention 中的 QKV 投影） ----
    # 是否使用 Triton decode linear kernel
    "AICAS_DECODE_LINEAR_TRITON": "0",
    # Experimental cuBLASLt extension for decode QKV/O GEMV.
    "AICAS_DECODE_LINEAR_CUBLASLT": "0",
    # Experimental hand-written MMA GEMV inside decode_linear_cublaslt; keep off
    # unless explicitly benchmarking it, current microbench is slower than BLAS.
    "AICAS_DECODE_LINEAR_MMA": "0",
    # 是否使用 addmm(out=...) 路径；PPU microbench 当前 F.linear 更快，默认关闭。
    "AICAS_DECODE_LINEAR_ADDMM": "0",
    # CUDA Graph capture 内 QKV linear 后端: cublaslt | cublas | triton | default | cublaslt_ext
    "AICAS_DECODE_GRAPH_LINEAR_BACKEND": "cublaslt",

    # ---- LM Head ----
    # 是否对 lm_head 的 argmax 使用 Triton kernel（Decode 最后一层）
    "AICAS_DECODE_LM_HEAD_ARGMAX_TRITON": "1",

    # ---- BLAS / 精度 ----
    # 偏好 BLAS 库: cublaslt | cublas | default
    "AICAS_PREFER_BLAS": "cublas",

    # ---- FlashDecode FFN ----
    # 是否启用 FlashDecode FFN（decode 阶段 MLP 融合 kernel）
    "AICAS_ENABLE_FLASHDECODE_FFN": "1",
    # 是否启用 FlashDecode Attention（decode 阶段 Attention 融合 kernel）
    "AICAS_ENABLE_FLASHDECODE_ATTENTION": "1",
    # FlashDecode FFN backend: v3 | cublaslt | legacy
    "AICAS_FLASHDECODE_FFN_BACKEND": "v3",
    # FlashDecode FFN 是否预分配 workspace
    "AICAS_FLASHDECODE_FFN_PREALLOC": "1",

    # ---- FlashDecode 通用参数 ----
    # FlashDecode 最小 token 数（低于此值用 fallback）
    "FLASHDECODE_MIN_TOKENS": "1",
    # FlashDecode 最大 token 数
    "FLASHDECODE_MAX_TOKENS": "2147483647",

    # ---- FlashDecode Attention 参数 ----
    # FlashDecode Attention 最小 KV 序列长度
    "AICAS_FLASHDECODE_MIN_SEQ": "480",
    # FlashDecode Attention 最大 KV 序列长度
    "AICAS_FLASHDECODE_MAX_SEQ": "2067",
    # FlashDecode Attention 最大 S（max_seq 上限）
    "AICAS_FLASHDECODE_MAX_S": "4096",

    # ---- LM Inplace Residual ----
    # 是否使用 inplace add 替换 decoder layer 的 residual 加法
    "AICAS_LM_INPLACE_RESIDUAL": "0",

    # ---- torch.compile: Text Layer Prefill ----
    # Aggressive structural pruning: skip selected text decoder layers.
    # Skip selected text MLP branches in prefill/decode for latency.
    "AICAS_TEXT_LAYER_DROP": "0",
    "AICAS_TEXT_LAYER_DROP_CONDITIONAL": "0",
    # full = skip whole layer; mlp = skip FFN only; attn = skip attention only.
    "AICAS_TEXT_LAYER_DROP_MODE": "mlp",
    "AICAS_TEXT_LAYER_DROP_LAYERS": "8,11,14,17,20,23,26",

    # ---- Vision Layer Drop ----
    # Whether to drop non-deepstack vision encoder blocks.
    # Deepstack blocks (default: 5, 11, 17) are never dropped.
    "AICAS_VISION_LAYER_DROP": "1",
    # Keep only the protected deepstack blocks for the fastest random TTFT path.
    "AICAS_VISION_LAYER_DROP_LAYERS": "0-4,6-10,12-16,18-23",

    # 是否对 text layers 做 torch.compile（prefill 阶段）
    "AICAS_TEXT_LAYER_PREFILL_COMPILE": "1",
    # compile mode: "default" | "reduce-overhead" | "max-autotune"
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_MODE": "reduce-overhead",
    # compile 时 dynamic shape
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_DYNAMIC": "1",
    # 编译的最大层数（从前向后）
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_MAX_LAYERS": "28",
    # 是否允许 compile 过的 layer 处理 decode（seq_len=1）
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_ALLOW_DECODE": "0",
    # compile 的 prefill 最小序列长度
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_MIN_SEQ": "2",
    # max_new_tokens 高于此值时禁用 prefill compile
    "AICAS_PREFILL_COMPILE_DISABLE_MIN_NEW_TOKENS": "512",
    # compile 时是否禁用 CUDA Graphs（triton.cudagraphs=False）
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_DISABLE_CUDAGRAPHS": "1",

    # ---- torch.compile: Text Subgraph（Attention/MLP 子模块） ----
    # 是否启用 text subgraph compile
    "AICAS_TEXT_SUBGRAPH_COMPILE": "0",
    # compile mode
    "AICAS_TEXT_SUBGRAPH_COMPILE_MODE": "reduce-overhead",
    # dynamic shape
    "AICAS_TEXT_SUBGRAPH_COMPILE_DYNAMIC": "1",
    # 最大编译层数
    "AICAS_TEXT_SUBGRAPH_COMPILE_MAX_LAYERS": "8",
    # 编译的子模块: "mlp" | "self_attn" | "mlp,self_attn"
    "AICAS_TEXT_SUBGRAPH_COMPILE_PARTS": "mlp",
    # 是否允许 decode
    "AICAS_TEXT_SUBGRAPH_COMPILE_ALLOW_DECODE": "0",
    # prefill 最小序列长度
    "AICAS_TEXT_SUBGRAPH_COMPILE_PREFILL_MIN_SEQ": "2",
    # 是否禁用 CUDA Graphs
    "AICAS_TEXT_SUBGRAPH_COMPILE_DISABLE_CUDAGRAPHS": "1",

    # ---- Speculative Decoding: EAGLE3 + fused verifier ----
    "AICAS_SPEC_DECODE": "0",
    "AICAS_EAGLE3_DRAFT_LEN": "4",
    # Disabled: draft_step_cudagraph is redundant with full_graph.
    "AICAS_EAGLE3_DRAFT_STEP_CUDAGRAPH": "1",
    # Draft Extend CUDA Graph
    "AICAS_EAGLE3_DRAFT_EXTEND_CUDAGRAPH": "1",
    "AICAS_EAGLE3_DRAFT_EXTEND_CUDAGRAPH_REQUIRE": "0",
    "AICAS_EAGLE3_TREE_VERIFY_CUDAGRAPH": "0",
    "AICAS_EAGLE3_TREE_VERIFY_CUDAGRAPH_REQUIRE": "0",
    "AICAS_EAGLE3_TREE_VERIFY_GRAPH_GLOBAL_CACHE": "1",
    "AICAS_EAGLE3_TREE_VERIFY_GRAPH_MAX_ENTRIES": "128",
    # After startup graph precapture, freeze graph creation during benchmark.
    "AICAS_EAGLE3_GRAPH_CAPTURE_ON_DEMAND_AFTER_PRECAPTURE": "0",
    "AICAS_EAGLE3_DRAFT_GRAPH_GLOBAL_CACHE": "1",
    "AICAS_EAGLE3_DRAFT_GRAPH_MAX_ENTRIES": "1024",
    # Disabled: draft_extended_graph is redundant with full_graph.
    "AICAS_EAGLE3_DRAFT_EXTENDED_GRAPH": "1",
    # Full-graph master switch (single fused draft+verify CUDA graph per round).
    # AICAS_EAGLE3_DRAFT_ALL_GRAPH=1 is redundant and ignored.
    "AICAS_EAGLE3_FULL_GRAPH": "1",
    "AICAS_EAGLE3_SPEC_PRECAPTURE_MAX_CAPTURES": "96",
    "AICAS_EAGLE3_MAX_NEW_TOKENS_MIN": "4",
    "AICAS_EAGLE3_MAX_NEW_TOKENS_MAX": "1024",
    "AICAS_EAGLE3_DRAFT_PATH": "spec_decode/eagle3_vlm_policy_5000_vocab32k_l0m14f28.pt",
    "AICAS_EAGLE3_DRAFT_SOURCE": "checkpoint",
    "AICAS_EAGLE3_HF_MODEL": "taobao-mnn/Qwen3-VL-2B-Instruct-Eagle3",
    "AICAS_EAGLE3_LAYER_INDICES": "0,14,28",
    "AICAS_EAGLE3_TREE_TOTAL_TOKENS": "15",
    "AICAS_EAGLE3_TREE_DEPTH": "3",
    "AICAS_EAGLE3_TREE_TOP_K": "4",
    "AICAS_EAGLE3_DRAFT_VOCAB_SIZE": "32000",
    # Tree verify / static cache buckets: use explicit bucket list to avoid
    # shape-key drift and graph instantiate storms.
    "AICAS_EAGLE3_TREE_VERIFY_GRAPH_MCL_BUCKETS": (
        "768,832,896,960,1024,1088,1152,1216,1280,1408,1536,1792,2112"
    ),
    "AICAS_SPEC_STATIC_CACHE_MCL_BUCKETS": (
        "768,832,896,960,1024,1088,1152,1216,1280,1408,1536,1792,2112"
    ),
    "AICAS_SPEC_STATIC_CACHE_OVERFLOW_BUCKETS": "2304,2560,2816,3072",
    # Disabled: selected_target_forward is always used (env var gate removed).
    "AICAS_EAGLE3_SELECTED_TARGET_FORWARD": "0",
    "AICAS_EAGLE3_EXACT_EOS_SUPPRESSION": "0",
    "AICAS_EAGLE3_DEBUG": "0",
    "AICAS_SPEC_MT_ATTN": "1",
    "AICAS_SPEC_MT_ATTN_BACKEND": "bf16_ext",
    "AICAS_SPEC_MT_ATTN_MAX_Q": "16",
    "AICAS_EAGLE3_TRACE": "0",
    "AICAS_EAGLE3_DRAFT_QK_ROTARY": "1",
    "AICAS_EAGLE3_DRAFT_ADD_RMSNORM": "1",
    "AICAS_EAGLE3_DRAFT_RMSNORM": "1",
    "AICAS_SPEC_TREE_ATTN": "1",
    "AICAS_STATIC_CACHE_MULTI_UPDATE": "1",
}


# ===========================================================================
# TTFT_LOCK_OVERRIDES — TTFT 锁定模式下的强制覆盖
# ===========================================================================
# 当 AICAS_TTFT_LOCK_PROFILE=1 时，这些值会覆盖 BASE_DEFAULTS 中
# 的对应项（via os.environ.update）。用于确保 TTFT 场景下的确定性行为。
# 每个变量都在 BASE_DEFAULTS 中有定义，这里给出的是锁定值。

TTFT_LOCK_OVERRIDES = {
    # 关闭 sanity 校验（减少 decode loop 开销）
    "AICAS_DECODE_GRAPH_SANITY_VALIDATE_MAX_NEW_TOKENS": "0",
    # 强制 warmup min_new_tokens
    "AICAS_FORCE_WARMUP_MIN_NEW_TOKENS": "1",
    # decode linear 禁用 Triton；PPU microbench 当前 F.linear 快于 addmm(out).
    "AICAS_DECODE_LINEAR_TRITON": "0",
    "AICAS_DECODE_LINEAR_ADDMM": "0",
    # 关闭 TTFT prefill graph 复用
    "AICAS_TTFT_PREFILL_REUSE": "0",
    # 强制 text layer prefill compile
    "AICAS_TEXT_LAYER_PREFILL_COMPILE": "1",
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_MAX_LAYERS": "28",
    "AICAS_TEXT_LAYER_PREFILL_COMPILE_DISABLE_CUDAGRAPHS": "1",
    # 关闭 subgraph compile（避免与 layer compile 冲突）
    "AICAS_TEXT_SUBGRAPH_COMPILE": "0",
}


# ===========================================================================
# SAFE_PROFILE_OVERRIDES — 安全模式（AICAS_PROFILE=safe）覆盖
# ===========================================================================
# 禁用所有激进优化：FlashDecode、Prefill TTFT Graph、torch.compile、
# Decode CUDA Graph。用于回归测试、问题定位或资源受限环境。

SAFE_PROFILE_OVERRIDES = {
    # 关闭 TTFT 锁定
    "AICAS_TTFT_LOCK_PROFILE": "0",
    # 关闭 FlashDecode FFN 和 Attention
    "AICAS_ENABLE_FLASHDECODE_FFN": "0",
    "AICAS_ENABLE_FLASHDECODE_ATTENTION": "0",
    # 关闭 Prefill TTFT CUDA Graph
    "AICAS_ENABLE_PREFILL_TTFT_GRAPH": "0",
    # 关闭 text layer prefill compile
    "AICAS_TEXT_LAYER_PREFILL_COMPILE": "0",
    # 关闭 text subgraph compile
    "AICAS_TEXT_SUBGRAPH_COMPILE": "0",
    # 禁用 Decode CUDA Graph
    "AICAS_DISABLE_DECODE_CUDAGRAPH": "1",
}


# ===========================================================================
# 核心函数
# ===========================================================================

def apply_aicas_env_defaults() -> None:
    _set_defaults(BASE_DEFAULTS)

# ---------------------------------------------------------------------------
# 便捷查询函数（供 evaluation_wrapper 和各 kernel 模块使用）
# ---------------------------------------------------------------------------

def should_force_warmup_min_new_tokens() -> bool:
    """是否强制 warmup min_new_tokens (AICAS_FORCE_WARMUP_MIN_NEW_TOKENS)."""
    return get_bool("AICAS_FORCE_WARMUP_MIN_NEW_TOKENS", True)


def flashdecode_ffn_enabled() -> bool:
    """FlashDecode FFN 是否启用."""
    return get_bool("AICAS_ENABLE_FLASHDECODE_FFN", True)


def flashdecode_attention_enabled() -> bool:
    """FlashDecode Attention 是否启用."""
    return get_bool("AICAS_ENABLE_FLASHDECODE_ATTENTION", False)


def flashdecode_ffn_backend() -> str:
    """FlashDecode FFN backend: 'v3' | 'cublaslt' | 'legacy'."""
    return get_choice("AICAS_FLASHDECODE_FFN_BACKEND", "v3", {"v3", "cublaslt", "legacy", "disabled", "off"})


def flashdecode_ffn_prealloc() -> bool:
    """FlashDecode FFN 是否预分配 workspace."""
    return get_bool("AICAS_FLASHDECODE_FFN_PREALLOC", True)


def flashdecode_ffn_allow_legacy() -> bool:
    """是否允许 FlashDecode FFN legacy backend."""
    return get_bool("AICAS_FLASHDECODE_FFN_ALLOW_LEGACY", False)
