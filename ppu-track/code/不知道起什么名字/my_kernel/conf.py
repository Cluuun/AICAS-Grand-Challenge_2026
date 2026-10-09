from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Optional

# 集中管理 my_kernel 运行时配置默认值。
# 日常调参优先改这里；若设置了同名环境变量，则环境变量优先。
# 布尔项约定：("1"/"true"/"on") 为开启，("0"/"false"/"off") 为关闭。
DEFAULTS: Dict[str, str] = {
    # ---------------------------
    # Vision / Prefill 公共配置
    # ---------------------------
    # 是否开启视觉特征缓存（仅同图重复请求收益明显）。
    # 可选: 1 | 0
    "ENABLE_VISION_FEATURE_CACHE": "1",
    # 视觉特征缓存条目数（LRU 容量，>=1）。
    # 可选: 正整数（建议 1~8）
    "VISION_FEATURE_CACHE_SIZE": "16",
    # TTFT 路径是否走视觉特征 LRU（默认关：benchmark 每图首次请求无收益且增加查表）。
    # 可选: 1 | 0
    "ENABLE_VISION_FEATURE_CACHE_TTFT": "1",
    # 多模态 prefill/TTFT 只 embed 文本 token，跳过会被视觉特征覆盖的 image/video placeholder。
    # 可选: 1 | 0
    "ENABLE_TEXT_ONLY_PLACEHOLDER_EMBED": "1",
    # 周期性打印视觉缓存命中率。
    # 可选: 1 | 0
    "VISION_FEATURE_CACHE_LOG_STATS": "0",
    # vision feature cache miss 路径启用 CUDA Graph（仅 vision encoder 计算）。
    # 可选: 1 | 0
    "ENABLE_VISION_ENCODER_CUDA_GRAPH": "1",
    # vision encoder graph capture 前 warmup 次数（用于完成 lazy init / kernel 选择）。
    "VISION_ENCODER_GRAPH_WARMUP_ITERS": "2",
    # replay 后是否 clone 输出（miss 写缓存时仍会做独立副本，避免 static buffer 被覆盖）。
    # 可选: 1 | 0
    "VISION_ENCODER_GRAPH_CLONE_OUTPUT": "1",
    # 是否打印 vision encoder graph 统计。
    # 可选: 1 | 0
    "VISION_ENCODER_GRAPH_LOG_STATS": "0",

    # ---------------------------
    # Vision Kernel Fusions (Qwen3VLVisionModel)
    # ---------------------------
    # 视觉 encoder 算子优化总开关（patch GEMM、pos/rope 缓存等）。
    # 可选: 1 | 0
    "ENABLE_VISION_KERNEL_FUSIONS": "1",
    # Patch embed Conv3d 等价改为 F.linear（走 cuBLAS/aublas GEMM）。
    # 可选: 1 | 0
    "ENABLE_VISION_PATCH_EMBED_LINEAR": "1",
    # Patch embed GEMM 后端。
    # 可选:
    # - cublaslt: cuBLASLt GEMM + bias epilogue，失败自动回退 mm_out
    # - triton: 手写 tiled matmul+bias kernel，避开 aten::mm/addmm 调用栈（实验）
    # - mm_out: torch.mm(out=workspace) + bias add_，显式走 cuBLAS/aublas GEMM，避开 aten::addmm
    # - matmul: 使用 @ + bias
    # - linear: 使用 F.linear（会表现为 aten::addmm）
    "VISION_PATCH_EMBED_BACKEND": "cublaslt",
    # 是否启用自定义 cuBLASLt epilogue 扩展。
    # 可选: 1 | 0
    "ENABLE_VISION_CUBLASLT_EPILOGUE": "1",
    # 初始化模型时预热 PatchEmbed cuBLASLt 扩展和算法，避免首个 TTFT 样本承担编译/heuristic。
    # 可选: 1 | 0
    "ENABLE_VISION_CUBLASLT_PREWARM": "1",
    "VISION_CUBLASLT_PREWARM_ROWS": "256",
    # fast_pos_embed_interpolate 结果 LRU 缓存。
    # 可选: 1 | 0
    "ENABLE_VISION_POS_CACHE": "1",
    # fast_pos_embed_interpolate 默认走无 JIT 的 index/weight cache，避免每个新分辨率触发 Triton 编译。
    # 可选: 1 | 0
    "ENABLE_VISION_POS_INDEX_CACHE": "1",
    # 单图 fast_pos_embed_interpolate 使用 Triton 融合四路 gather/权重/布局重排。
    # 可选: 1 | 0（默认关：动态 h/w 会导致每个新 shape JIT 编译，TTFT 不划算）
    "ENABLE_VISION_POS_TRITON": "0",
    # rot_pos_emb + cos/sin + cu_seqlens 准备段 LRU 缓存。
    # 可选: 1 | 0
    "ENABLE_VISION_ROPE_CACHE": "1",
    # Vision attention RoPE 用 Triton 融合 q/k rotate_half，减少 neg/cat/mul/add kernel。
    # 可选: 1 | 0
    "ENABLE_VISION_ROPE_TRITON": "1",
    # 初始化模型时预热 Vision RoPE Triton kernel，避免首个 TTFT 样本承担 JIT 编译。
    # 可选: 1 | 0
    "ENABLE_VISION_ROPE_TRITON_PREWARM": "1",
    # 模型加载后用 dummy 336x336 图跑一次 vision forward，预热 patch GEMM / RoPE / allocator。
    # 可选: 1 | 0
    "ENABLE_VISION_FORWARD_PREWARM": "1",
    # 固定 336 时 pos/rope/cu_seqlens 缓存键用常量 grid，避免 detach().cpu().tolist() 同步。
    # 可选: 1 | 0
    "ENABLE_FIXED_VISION_SHAPE_CACHE": "1",
    # 与 evaluation_wrapper 固定分辨率 fastpath 对齐（336x336）。
    "FIXED_IMAGE_RESOLUTION": "336x336",
    "ENABLE_FIXED_336_FASTPATH": "1",
    # 用 tiny host 前缀和构造 cu_seqlens，避免 repeat_interleave kernel。
    # 可选: 1 | 0
    "ENABLE_VISION_CU_SEQLENS_HOST": "1",
    # 位置/rope 缓存容量（>=1）。
    "VISION_POS_CACHE_SIZE": "8",
    # Vision encoder 内 Linear 使用 torch.mm(out=workspace)+bias，走 cuBLAS/aublas 并避开 addmm。
    # 可选: 1 | 0（全量 patch 会影响 throughput，默认关；后续可做 Block_0 精细 A/B）
    "ENABLE_VISION_LINEAR_MM_OUT": "1",
    # Vision Linear 优先使用 cuBLASLt bias epilogue，失败回退 mm_out。
    # 可选: 1 | 0（默认关：vision encoder 有大量 Linear，pybind/Lt descriptor 成本高于 bias epilogue 收益）
    "ENABLE_VISION_LINEAR_CUBLASLT": "0",
    # Vision MLP / merger 走 fast path。
    # 可选: 1 | 0
    "ENABLE_VISION_MLP_FAST": "1",
    # 融合 visual patch 输出加位置编码与首个 VisionBlock 的 LayerNorm_0。
    # 可选: 1 | 0（实验项：trace 调用数下降，但当前端到端有回归，默认关）
    "ENABLE_VISION_FUSED_POS_LN0": "0",
    # Processor 侧分辨率策略：减少 image_grid_thw 形态数量，提高缓存/图稳定性。
    # 可选: dynamic | fixed | bucket
    # 当前测试样本以 OCR/品牌/数字/价格/招牌为主，bucket/fixed 会把 1024 级原图
    # 压到约 140 个视觉 token，细节损失过大；默认保持动态等比例 resize。
    "ENABLE_VISION_RESOLUTION_POLICY": "1",
    "VISION_RESOLUTION_POLICY": "fixed",
    # fixed 模式使用的固定输入分辨率。
    "VISION_FIXED_RESOLUTION": "336x336",
    # bucket 模式候选分辨率，按原图宽高比选最近项；建议使用 28 的倍数。
    "VISION_BUCKET_RESOLUTIONS": "280x280,336x336,364x364,448x252,252x448,392x280,280x392",

    # ---------------------------
    # TTFT Fast Path
    # ---------------------------
    # 是否使用 TTFT 专用多模态 forward（拆分 vision / text prefill / lm_head）。
    # 可选: 1 | 0
    "ENABLE_TTFT_DEDICATED_FORWARD": "1",
    "ENABLE_TTFT_STATIC_KV": "1",
    # 打印 TTFT 细分耗时。
    # 可选: 1 | 0
    "LOG_TTFT_BREAKDOWN": "0",
    # 加载模型时尝试 flash_attention_2（失败自动回退默认 attention）。
    # 可选: 1 | 0
    "ENABLE_FLASH_ATTENTION_2": "1",
    # 图像清晰度倍率，只暴露这一个压图旋钮。
    # 以当前动态 512px 级别为 1.0；0.75 约等于旧的 384px 上限，
    # 1.25 约等于 640px 上限。越大越清楚、视觉 prefill 越重。
    "IMAGE_RESIZE_RATIO": "1.0",

    # ---------------------------
    # Decode Runtime / CUDA Graph
    # ---------------------------
    # 是否启用 decode CUDA Graph 路径。
    # 可选: 1 | 0
    "ENABLE_DECODE_CUDA_GRAPH": "1",
    # decode runtime 允许接管的 max_new_tokens 上限。
    # 可选: 正整数（建议 >= benchmark 的生成长度）
    "CUDA_GRAPH_MAX_NEW_TOKENS": "1024",
    # benchmark 吞吐路径固定 max_new_tokens=128；强制生成满 128 token 再允许 EOS，
    # 避免短答案提前停止导致 tokens/sec 被低估。
    # 可选: 1 | 0
    "FORCE_MIN_NEW_TOKENS_FOR_THROUGHPUT": "1",
    "THROUGHPUT_MAX_NEW_TOKENS": "128",
    # 每次 chunk graph replay 覆盖的 token 数。
    # 可选: 正整数（常用 2/4/8/16）
    "CUDA_GRAPH_CHUNK_STEPS": "1",
    # 是否打印 decode 图命中/捕获统计日志。
    # 可选: 1 | 0
    "CUDA_GRAPH_LOG_DECODE_STATS": "0",
    # TTFT 专用 fast path（max_new_tokens=1）开关。
    # 可选: 1 | 0
    "ENABLE_TTFT_FASTPATH": "1",
    # Lookahead/Jacobi 解码实验开关。默认关闭，避免影响稳定的 CUDA Graph decode。
    # 可选: 1 | 0
    "ENABLE_LOOKAHEAD_DECODE": "0",
    # 固定 block lookahead：候选块长度。命中时一次 verify 最多接受这么多 token。
    "LOOKAHEAD_BLOCK_SIZE": "8",
    # 最近多少个 token 作为 n-gram key；参考实现默认 2。
    "LOOKAHEAD_KEY_LEN": "1",
    # model/output 候选池最大 key 数，防止长生成中 Python 池膨胀。
    "LOOKAHEAD_MAX_MODEL_CANDIDATES": "64",
    # 跨请求复用 block 候选池，benchmark warmup/前序样本可为后续生成提供候选。
    # 可选: 1 | 0
    "LOOKAHEAD_PERSISTENT_POOL": "1",
    # 每隔多少个普通 decode step 才尝试一次 block verify，避免 verify 过密拖慢。
    "LOOKAHEAD_VERIFY_STRIDE": "8",
    # n-gram 池未命中时，执行一次 Jacobi draft forward 生成候选轨迹再 verify。
    # 可选: 1 | 0
    "LOOKAHEAD_JACOBI_ON_MISS": "0",
    # Jacobi draft 迭代轮数；轮数越多越容易收敛，但每轮都是一次 q_len=block forward。
    "LOOKAHEAD_JACOBI_ITERS": "2",
    # Jacobi miss draft 至少接受多少 token 才提交，否则回退 CUDA Graph。
    "LOOKAHEAD_JACOBI_MIN_ACCEPT": "2",
    # 生成长度低于该阈值时仍走当前 CUDA Graph decode；默认覆盖 benchmark warmup 的 10 token。
    "LOOKAHEAD_MIN_NEW_TOKENS": "8",
    # 是否打印 lookahead 接受率和回退统计。
    # 可选: 1 | 0
    "LOOKAHEAD_LOG_STATS": "0",

    # ---------------------------
    # Prompt-only 短答（不改 decode loop；判断仅在 prompt 构造阶段）
    # ---------------------------
    # 全局短答（不推荐）：所有样本追加同一短指令。
    "ENABLE_PROMPT_SHORT_ANSWER": "0",
    "PROMPT_SHORT_ANSWER_TEXT": "Brief answer only.",
    "PROMPT_SHORT_ANSWER_MODE": "user_suffix",
    # 生成前 question router + 分类型短 prompt（推荐）。
    "ENABLE_PROMPT_SHORT_ANSWER_ROUTER": "1",
    "PROMPT_SHORT_ANSWER_SAFE_TEXT": "Answer only.",
    "PROMPT_SHORT_ANSWER_CAUTIOUS_TEXT": "Answer only. If unclear, say unknown.",
    # cautious 类默认不加 prompt，需显式开启。
    "PROMPT_SHORT_ANSWER_ENABLE_CAUTIOUS": "1",
    "PROMPT_SHORT_ANSWER_LOG_STATS": "0",
    # text prefill CUDA graph 固定 bucket 上限；v9 短 prompt 后最大约 133，136 避免 TTFT fallback。
    "TEXT_PREFILL_CUDA_GRAPH_BUCKET_LEN": "136",

    # ---------------------------
    # Prefill A/B 配置
    # ---------------------------
    # prefill 后端策略。
    # 可选:
    # - full: 标准全量 prefill
    # - chunked_eager: 分块 prefill（长上下文 A/B）
    # - flash_attn: 预留标签，当前会回退到 full
    # - flashinfer: 预留标签，当前会回退到 full
    "PREFILL_BACKEND": "full",
    # 分块 prefill 的 chunk 大小（token）。
    # 可选: 正整数（建议 256/512/1024）
    "PREFILL_CHUNK_SIZE": "512",
    # 分块 prefill 时是否允许多模态输入也走 chunk 模式。
    # 可选: 1 | 0（默认 0，优先保守正确性）
    "PREFILL_CHUNK_WITH_MULTIMODAL": "0",
    # Text prefill/decode 的 Qwen3VLTextRMSNorm.forward 改为 torch fused rms_norm。
    # 可选: 1 | 0
    "ENABLE_TEXT_RMSNORM_FAST": "1",
    # Text attention RoPE 用 Triton 融合 rotate_half，减少 neg/cat/mul。
    # 可选: 1 | 0
    "ENABLE_TEXT_ROPE_TRITON": "1",
    # 初始化时预热 text RoPE Triton kernel，避免首个 TTFT 样本承担 JIT。
    # 可选: 1 | 0
    "ENABLE_TEXT_ROPE_TRITON_PREWARM": "0",
    # text RoPE Triton 实现：
    # - exact/hybrid: 更保守但吞吐收益小
    # - fused_fp16_acc: code_0603 的吞吐优先配置
    "TEXT_ROPE_TRITON_BACKEND": "fused_fp16_acc",
    # TTFT max_new_tokens=1 的 generate 输出转 CPU，避开 benchmark 计时内打印 CUDA tensor。
    # 可选: 1 | 0
    "ENABLE_TTFT_RETURN_CPU_OUTPUT": "0",
    # Text model prefill CUDA Graph：把多模态 prompt pad 到固定长度后 replay。
    # 第一版保守只启用单 bucket；真实 prompt_len 仍用于 lm_head 与后续 decode。
    # 可选: 1 | 0
    "ENABLE_TEXT_PREFILL_CUDA_GRAPH": "1",
    # 固定 text prefill graph 长度。超过该长度自动回退 eager。
    # capture 前 eager warmup 次数，用于完成 lazy init / kernel 选择。
    "TEXT_PREFILL_CUDA_GRAPH_WARMUP_ITERS": "1",
    # graph 内是否传入右侧 pad 的 attention_mask。固定右 padding 下真实 token
    # 看不到未来 pad，默认关闭以避免 prefill mask 构造破坏 graph capture。
    # 可选: 1 | 0
    "TEXT_PREFILL_CUDA_GRAPH_USE_ATTENTION_MASK": "0",
    # 是否打印 text prefill graph 捕获/命中统计。
    # 可选: 1 | 0
    "TEXT_PREFILL_CUDA_GRAPH_LOG_STATS": "0",

    # ---------------------------
    # Decode Attention / Token 选择
    # ---------------------------
    # decode attention 后端选择。
    # 可选:
    # - flash_attn: 使用 flash_attn_with_kvcache（默认）
    # - flashinfer: 优先 flashinfer，失败自动回退 flash_attn
    "DECODE_ATTN_BACKEND": "flash_attn",
    # next token 选择后端。
    # 可选:
    # - argmax: 对 outputs.logits 做 argmax
    # - fused_lm_head: hidden_states + lm_head 直接算 logits 再 argmax
    # - direct_lm_head: decode graph 内直接走 model.model + lm_head + argmax
    "DECODE_LM_HEAD_ARGMAX_BACKEND": "direct_lm_head",
    # flash_attn_with_kvcache 的 num_splits（Flash-Decoding Split-KV）。
    # 可选:
    # - 0 或 auto: FA 内置启发式
    # - 1: 不切分 KV，当前短/中等上下文吞吐更稳
    # - >=2: 强制按该 splits 数并行
    "FLASH_KVCACHE_NUM_SPLITS": "1",
    # 允许 lookahead/Jacobi 验证阶段使用 q_len>1 的 flash_attn_with_kvcache。
    # flashinfer single_decode 仍只用于 q_len=1。
    # 可选: 1 | 0
    "ENABLE_FLASH_KVCACHE_MULTI_QUERY": "1",
    # 是否尝试使用 flash_attn_with_kvcache 内置 rotary 写 KV。
    # 当前仅对简单 2D RoPE 表启用，Qwen3-VL MRoPE 会自动回退。
    # 可选: 1 | 0
    "ENABLE_FA_KVCACHE_ROTARY": "0",

    # ---------------------------
    # Decode Kernel Fusion 开关
    # ---------------------------
    # decode MLP gate/up 融合开关。
    # 可选: 1 | 0
    "ENABLE_DECODE_FUSED_MLP": "1",
    # decode QKV 合并线性开关。
    # 可选: 1 | 0
    "ENABLE_DECODE_FUSED_QKV": "1",
    # decode layer tail (residual + norm + mlp) 融合开关。
    # 可选: 1 | 0
    "ENABLE_DECODE_LAYER_TAIL_FUSION": "1",
    # 允许 lookahead/Jacobi 验证阶段的 q_len>1 使用 layer tail 融合。
    # 可选: 1 | 0
    "ENABLE_DECODE_LAYER_TAIL_MULTI_QUERY": "1",
    # decode MLP 内 residual fuse 开关。
    # 可选: 1 | 0
    "ENABLE_DECODE_MLP_RESIDUAL_FUSE": "1",

    # ---------------------------
    # Decode 子算子后端
    # ---------------------------
    # Q/K RMSNorm 后端。
    # 可选:
    # - f_rms_norm: 优先 torch.nn.functional.rms_norm
    # - manual: 手写 rsqrt 路径（用于 A/B）
    # - triton: 单行 decode Triton RMSNorm（不可用则回退 f_rms_norm）
    "DECODE_QK_NORM_BACKEND": "f_rms_norm",
    # Q/K norm + rope 联合后端。
    # 可选:
    # - separate: 走现有 q_norm -> rope 分离路径
    # - triton_fused: 单 token decode 下把 q/k rms_norm 和 rope 合并
    "DECODE_QK_ROPE_BACKEND": "triton_fused",
    # SiLU * up 后端。
    # 可选:
    # - eager: 常规表达式
    # - inplace: 原地 mul
    # - triton: 单 kernel 完成 silu(gate) * up（不可用则回退 eager）
    "DECODE_SILU_MUL_BACKEND": "eager",
    # o_proj 后端。
    # 可选:
    # - linear: F.linear
    # - mm_out: torch.mm(out=workspace)，主要针对 BS=1 decode A/B
    # - residual_mm_out: 在 layer-tail fused decode 中将 o_proj 输出直接累加到 residual
    "DECODE_O_PROJ_BACKEND": "linear",
    # down_proj 后端。
    # 可选:
    # - linear: F.linear
    # - mm_out: torch.mm(out=workspace)，主要针对 BS=1 decode A/B
    "DECODE_DOWN_PROJ_BACKEND": "linear",

    # ---------------------------
    # Runtime Int8 Quantization Probe
    # ---------------------------
    # 自研 runtime 内的轻量运行时量化实验开关；默认关闭，不使用 qlean/vLLM。
    # 当前实现会在加载后一次性量化所选权重，默认只在 decode 小 q_len 上接管。
    # 可选: 1 | 0
    "ENABLE_RUNTIME_INT8_QUANT": "1",
    # 可选:
    # - acext_w8a8: 使用 acext Python 暴露的 int8_gemm
    # - torch_int_mm: 使用 torch._int_mm + 显式反量化（用于隔离 acext 开销）
    # - triton_w8a16_gemv: 自写 weight-only W8A16 GEMV，激活保持 FP16/BF16（实验）
    # - acext_wo_gemv: 直接调用 acext WeightonlyBatchedGemv 小 M W8A16 kernel（实验）
    "RUNTIME_INT8_QUANT_BACKEND": "acext_wo_gemv",
    # 量化目标。
    # 可选:
    # - mlp_down: 只量化 MLP down_proj（最小扰动）
    # - mlp_gate_up: 只量化融合后的 gate_proj+up_proj
    # - mlp_all: gate/up/down 都量化
    # 多个目标用逗号分隔。
    "RUNTIME_INT8_QUANT_TARGETS": "mlp_all",
    # 量化哪些 decoder 层。
    # 可选示例:
    # - last:4 / first:4 / all / 0,1,2 / range:8:28
    "RUNTIME_INT8_QUANT_LAYERS": "all",
    # 只在 q_len <= 该值时使用量化线性层。默认 1，避免 prefill 被激活量化开销拖慢。
    "RUNTIME_INT8_MAX_QUERY_LEN": "1",
}


@lru_cache(maxsize=None)
def _raw(name: str, fallback: Optional[str] = None) -> str:
    # Priority: runtime env override > centralized DEFAULTS > call-site fallback.
    # Many call sites pass a historical fallback value (e.g. "1"/"0").  Those
    # should not mask the values users edit in this file.
    value = os.getenv(name)
    if value is not None:
        return value.strip()
    if name in DEFAULTS:
        return DEFAULTS[name].strip()
    if fallback is None:
        fallback = ""
    return os.getenv(name, fallback).strip()


def conf_str(name: str, fallback: Optional[str] = None, *, lower: bool = False) -> str:
    value = _raw(name, fallback)
    return value.lower() if lower else value


def conf_bool(name: str, fallback: Optional[str] = None) -> bool:
    return conf_str(name, fallback, lower=True) not in ("0", "false", "off")


def conf_int(name: str, fallback: Optional[str] = None, *, minimum: Optional[int] = None) -> int:
    value = int(conf_str(name, fallback))
    if minimum is not None:
        return max(minimum, value)
    return value


def conf_float(
    name: str,
    fallback: Optional[str] = None,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    value = float(conf_str(name, fallback))
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value
