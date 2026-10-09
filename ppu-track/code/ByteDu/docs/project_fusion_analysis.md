# Project fusion analysis

## 比赛评测机制

- 官方/本地 benchmark 通过 `evaluation_wrapper.VLMModel` 加载模型，要求暴露 `processor`、`model`、`device`。
- 性能路径直接调用 `model.model.generate(**inputs, max_new_tokens=1/128, do_sample=False, use_cache=True)`；TTFT 是 1 token 生成耗时，Throughput 是 128 token 端到端 token/s。
- `benchmark.py` 本地会先 warmup 10 条，再对指定样本数生成 `result_fusion.json`，包含 `performance.avg_ttft_ms`、`performance.avg_throughput_tokens_per_sec` 和 `answers`。

## `JunkratDev` 性能优化路径

- 单图 batch=1 快路径：绕过 HF 通用 generate，使用 `FastMinimalQwen3VLModel.generate`。
- prefill/decode 分离：prefill 负责视觉编码、文本全序列、KV 写入和首 token；decode 使用单 token workspace。
- 静态 KV cache 与 decode scratch 预分配，避免每步分配。
- `flash_attn_with_kvcache`/FlashAttention、Triton RoPE/RMSNorm/SwiGLU/packed QKV cache write、Triton lm_head argmax。
- CUDA Graph decode replay、prefill bucket graph、radix cache、视觉分辨率路由/缩放，用于降低 TTFT。

## `Junkrat` 量化路径

- 在 `FastMinimalQwen3VLModel` 内维护 packed text layer weights。
- decode GEMV 使用 prebuilt CUDA kernels：Q8 packed weight、W8A8 weight/activation、fused add-norm-GEMV、fused norm-gate-up-SwiGLU。
- lm_head 支持 Q8 packed linear+argmax。
- SageAttention/KV cache 路径现在通过 `sage_decode_attention.py` 单一 decode adapter 进入，支持 BF16、K INT8、Q/K/V INT8 变体；默认使用 fused mixed-KV（Q/K INT8 + V BF16），V 全 INT8 保留为 ablation。
- W4A4 draft speculative runtime 存在，但依赖接受率和 checkpoint，默认关闭。
- 源项目自带 `llmcompressor` 离线量化工具链，可跑 `smoothquant-gptq` 并导出仅含 `*.weight_scale` 的 `quant_codebook.safetensors`。

## 可复用代码

- 从 `Junkrat` 复用并清理：`evaluation_wrapper.py`、`minimal_qwen3_vl.py`、`eagle3_*`、`w4a4_draft.py`、`sage_decode_attention.py`、低层 `sageattention_kvcache_triton.py`、`triton_*.py`、`prebuilt/*.so`。
- 从 `JunkratDev` 复用：`vision_route_policy.py` 和分辨率路由策略；本融合版实现为 `fusion_resolution.py`，只包裹 processor，不复制整套项目。
- 复用本地 `benchmark.py` 作为自测入口。

## 冲突点

- `JunkratDev` 的完整 processor/profile/radix/vision graph 逻辑与 `Junkrat` 的量化 wrapper 都修改 `evaluation_wrapper.py`，直接覆盖会丢失一侧能力。
- 低精度 KV cache/Sage INT8 对不同样本收益不稳定，且更容易引入精度漂移。
- W4A4 speculative draft 缺少稳定 checkpoint/接受率保障，默认开启可能吞吐波动。
- `Junkrat` 离线 PTQ 大文件未作为提交必要文件复制；当前默认使用运行时打包的真实 Q8/W8A8 decode kernels。

## 融合方案

- 以 `Junkrat` 量化版 wrapper 为主干，保留真实 quantized decode compute、Sage decode、静态 KV、CUDA Graph、Triton/CUDA kernels。
- `FusionResolutionProcessor` 现默认关闭所有图像重采样；processor 侧只在模型网格对齐需要时做无插值补边。若显式开启 `AICAS_VISION_RESIZE_MODE`，仍可恢复题型路由缩放实验路径。
- 默认离线量化码本路径：`bundled_quantization/rowwise_q8/quant_codebook.safetensors`，覆盖 text Q/K/V/O、gate/up/down 的 rowwise-Q8 scale；旧 `smoothquant-gptq` W8A8 codebook 因长答案重复漂移保留为 ablation。
- 默认路径：no-resize / pad-only processor + fast minimal model + static KV + CUDA/Triton fused decode + bundled rowwise-Q8 codebook + runtime Q8 text decode + Q8 lm_head (`BLOCK_K=256`) + fused down->QKV + Sage K/Q INT8 + V BF16 decode from token 1。
- 收益不稳定路径默认关闭：W4A4 draft speculative、EAGLE3 draft、INT8 KV/Sage qk_i8/v_i8。
