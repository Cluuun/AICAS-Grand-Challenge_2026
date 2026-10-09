# Fusion implementation report

## 新增/复制文件

- 新增：`fusion_resolution.py`、`run_fusion_eval.sh`、`docs/project_fusion_analysis.md`、`docs/fusion_implementation_report.md`。
- 复制并裁剪复用：`evaluation_wrapper.py`、`minimal_qwen3_vl.py`、`benchmark.py`、`eagle3_model.py`、`eagle3_runtime.py`、`w4a4_draft.py`、`sage_decode_attention.py`、低层 `sageattention_kvcache_triton.py`、`triton_*.py`、`prebuilt/*.so`、`requirements.txt`。
- 验证产物：`result_fusion.json`、`result_fusion_10.json`、`result_fusion_30.json`、`result_fusion_150.json`。

## 模型/data 软链

- `Qwen3-VL-2B-Instruct -> /root/JunkratDev/Junkrat/Qwen3-VL-2B-Instruct`
- `data -> /root/JunkratDev/Junkrat/data`

## 融合的性能优化

- `FastMinimalQwen3VLModel.generate` 命中官方 benchmark 的直接 generate 路径。
- prefill/decode 分离、静态 KV cache、decode workspace 预分配。
- FlashAttention / `flash_attn_with_kvcache`。
- Triton/CUDA RMSNorm、RoPE、SwiGLU、packed QKV cache write、decode attention、lm_head argmax。
- CUDA Graph decode replay。
- 视觉侧默认采用 processor 级 128-768 visual-token budget，减少 prompt KV 长度并保留接近 PyTorch-50 的语义代理；更激进的 128-512/128-256 作为显式 throughput ablation。

## 融合的量化策略

- text decoder decode 默认启用 bundled rowwise-Q8 codebook + runtime Q8 text decode；rowwise codebook 覆盖 text Q/K/V/O、gate/up/down scale，旧 SmoothQuant+GPTQ codebook 与 W8A8 GEMV kernel 保留为显式 opt-in。
- lm_head 默认使用 Q8 packed argmax，Triton `BLOCK_K=256`；和 runtime Q8 text decode、Sage min_new_tokens=1、fused down->QKV 链组合后 30-sample 吞吐稳定在 533+ tok/s 且重复率保持低位。
- Q8 lm_head 的 512 tile 可把 10-sample 吞吐推到 531-534 tok/s，但会改变接近分数 token 的排序并在自回归中放大成语义漂移；已保留 opt-in 的 `JUNKRAT_Q8_LM_HEAD_REFINE_MODE`/候选复核实验开关，但默认仍使用稳定的 128x256 tile。
- 新 rowwise-Q8 codebook + fused mixed-KV 默认路径在 30-sample 上达到 TTFT 24.21 ms / throughput 537.48 tok/s / repetition 0.138，替代旧默认 runtime-only Q8（529.28 tok/s）和未融合 mixed-KV 路径（533.69 tok/s）。
- 按 PPU SDK v2.0 文档接入 Acext 量化库：`JUNKRAT_ENABLE_ACEXT_PREFILL_A8W8=1` 默认启用，`JUNKRAT_ACEXT_PREFILL_TARGETS=auto` 会在启动时对 prefill `qkv/gate_up/down_proj/o_proj` 做 A8W8 micro-profile，并同时检查 Acext 输出相对 BF16 GEMM 的 mean/max 相对误差；权重 scale 以 FP32 传入 `acext.int8_gemm`。当前 ZW810E 上 `gate_up` 在 1024-token profile 规模稳定胜出且通过误差门控，其他路径保留显式 ablation。
- True W8A8 已接入为显式实验路径：`JUNKRAT_ENABLE_W8A8_DECODE=1` 时 `JUNKRAT_W8A8_TARGETS=auto` 会在启动时分别 micro-profile `qkv`、`gate_up`、`down_proj`、`o_proj`，只保留明显快于 packed-Q8 的 A8 子路径并释放负收益 int8 buffers；也可显式设置 `qkv,gate_up,down_proj,o_proj`/`all` 做 ablation。`down_proj` 额外提供 `prebuilt/a8w8_decode_ext.so` 中的 PerToken A8 + PerChannel W8 DP4A fused `down_proj+residual+sumsq` 变种，并保留 fixed-scale / lagged-scale / on-the-fly quant 实验入口。`JUNKRAT_ENABLE_W8A8_LM_HEAD=1` 会运行时生成 lm_head int8 权重并预热 W8A8 argmax。当前 PPU 上 A8 decode/lm_head 仍未超过 packed-Q8，因此默认主路径继续使用稳定 Q8。
- Decode 侧新增 Acext WeightOnlyBatchedGemv lm_head：默认尝试 `JUNKRAT_ENABLE_ACEXT_DECODE_LM_HEAD=1` + `JUNKRAT_ACEXT_DECODE_LM_HEAD_MODE=int8_pc`，用 A16W8 per-channel GEMV、预分配 FP16 activation scratch 和专用多 block argmax 替代 Q8 lm_head argmax；Acext/扩展不可用时自动回退 Q8。argmax stage2 可融合 graph post-step bookkeeping，但实测收益很小；multi-token graph unroll 保留为 `JUNKRAT_DECODE_GRAPH_BLOCK_STEPS` opt-in，默认 1。`JUNKRAT_ENABLE_ACEXT_A8W8_DECODE_LM_HEAD=1` 与 `int4_pc|int4_g64|int4_g128` 仍仅作为显式 ablation。
- SageAttention decode 默认开启 fused mixed-KV：Q/K INT8 + V BF16，并融合 QKV cache write 与 attention。Sage-active layer 的 attention output 直接走 BF16 `addmv_bn1` o_proj handoff，避免旧 Q8 o_proj handoff 吞掉 attention/cache 收益。旧 prefill Sage wrapper 已移除；当前只在 text decode 热路径使用 `sage_decode_attention.py` 单一 adapter，prefill 继续使用 FlashAttention/静态 cache 写入，避免多入口 dispatch 混乱。
- W4A4 draft speculative runtime 已接入但默认关闭。
- 新增 `smoothquant-gptq` 离线量化码本：基于 64 calibration samples 生成量化模型，再导出 `bundled_quantization/w8a8/quant_codebook.safetensors`。

## 默认路径

AutoProcessor 128-768 visual-token budget -> `FastMinimalQwen3VLModel` -> bundled rowwise-Q8 codebook -> Acext speed+error auto-gated prefill A8W8 gate/up -> prefill 写静态 KV -> runtime Q8 text decode -> fused Sage Q/K INT8 + V BF16 mixed-KV -> Acext WeightOnly A16W8 int8_pc lm_head（不可用时回退 Q8 lm_head argmax `BLOCK_K=256`）+ fused down->norm->QKV chain。量化默认开启；W8A8 true-fusion 和旧 W8A8 codebook 仅用于显式 ablation。

## env flags

- 视觉：默认 `JUNKRAT_PROCESSOR_MIN_VISUAL_TOKENS=128`、`JUNKRAT_PROCESSOR_MAX_VISUAL_TOKENS=768`。可显式设 `JUNKRAT_PROCESSOR_MAX_VISUAL_TOKENS=512` 或 `256` 追求更高 throughput/更低 TTFT，但 PyTorch-50 token_F1 代理会下降；`AICAS_VISION_*` 路由缩放仍保留为实验路径。
- 量化：默认开启 `JUNKRAT_ENABLE_BUNDLED_QUANT_CODEBOOK=1`（rowwise-Q8）、`JUNKRAT_ENABLE_RUNTIME_Q8_TEXT_DECODE=1`、Acext WeightOnly int8_pc lm_head 和 Acext prefill A8W8 auto gating；Acext prefill auto 还受 `JUNKRAT_ACEXT_PREFILL_AUTO_MAX_MEAN_RELERR=0.015`、`JUNKRAT_ACEXT_PREFILL_AUTO_MAX_MAX_RELERR=0.05` 约束。`JUNKRAT_ENABLE_W8A8_DECODE`、`JUNKRAT_ENABLE_W8A8_LM_HEAD`、有长答案重复漂移的旧 W8A8 codebook、W4A4 speculative 仍按吞吐和语义实测保留为 opt-in。
- W8A8 decode：默认 auto margin 为 `JUNKRAT_W8A8_AUTO_MARGIN=0.90`，避免 qkv 这类 microbench 小赢但端到端变慢的路径被自动启用。`down_proj` 可设置 `JUNKRAT_W8A8_DOWN_FIXED_ACT_SCALE>0` 做 fixed-scale 单 pass quant ablation；该路径吞吐高于动态 down A8W8，但仍低于 packed-Q8，默认关闭。
- Acext decode：默认 `JUNKRAT_ENABLE_ACEXT_DECODE_LM_HEAD=1`、`JUNKRAT_ACEXT_DECODE_LM_HEAD_MODE=int8_pc`；可设 `JUNKRAT_ENABLE_ACEXT_DECODE_LM_HEAD=0` 禁用并回到 Q8 lm_head。A8W8 ablation 用 `JUNKRAT_ENABLE_ACEXT_A8W8_DECODE_LM_HEAD=1`；INT4 ablation 用 `JUNKRAT_ACEXT_DECODE_LM_HEAD_MODE=int4_pc|int4_g64|int4_g128`。
- Sage：`JUNKRAT_ENABLE_SAGE_DECODE_ATTN=1`、`JUNKRAT_SAGE_IMPL=cuda`、`JUNKRAT_SAGE_QK_PATH=i8`、`JUNKRAT_SAGE_VALUE_PATH=bf16`、`JUNKRAT_ENABLE_SAGE_FUSED_QKV_DECODE=1`、`JUNKRAT_SAGE_LAYERS=all`。`JUNKRAT_DISABLE_SAGE_BF16_O_PROJ=1` 可回退旧 Q8 o_proj handoff 做 ablation。
- 默认关闭：`ENABLE_W4A4_DRAFT_SPEC=0`。

## 依赖环境

- 本地验证环境：Python 3.12.3，Linux 5.10，Torch 2.8.0 + CUDA 12.9，GPU `PPU-ZW810E`，compute capability 8.0。
- Python 包实测版本：`transformers==4.57.0`、`flash_attn==2.7.4.post1`、`triton==3.4.0`、`safetensors==0.7.0`、`datasets==4.8.5`、`Pillow==12.1.0`、`numpy==2.2.6`、`tqdm==4.67.1`、`psutil==7.2.2`。
- 提交包内包含 `requirements.txt`，显式列出 `torch`、`transformers`、`accelerate`、`safetensors`、`numpy`、`datasets`、`Pillow`、`tqdm`、`psutil`、`triton`、`flash-attn`。
- 自定义算子不依赖评测端 JIT 编译，提交包提供 `prebuilt/gemv_kernel.so`、`prebuilt/decode_fused_ops.so`、`prebuilt/custom_decode_attention_sm80.so`、`prebuilt/cutlass_dual_gemm_swiglu.so`、`prebuilt/a8w8_decode_ext.so`。
- Acext WeightOnly decode 算子预编译为 `prebuilt/acext_weightonly_ext.so`，源码保留为 `acext_weightonly_ext.cpp` 和 `acext_weightonly_argmax_kernel.cu`；默认 int8_pc 路径会尝试加载，加载失败时自动回退 Q8 lm_head。
- 预编译 `.so` 面向当前 PPU/Ampere SM80 + Torch 2.8/CUDA 12.9 环境；若评测端 ABI 变化，应重编这些扩展而不是退回 Python fallback。

## 实测结果

| samples | TTFT ms | Throughput tok/s | 失败数 |
|---:|---:|---:|---:|
| 10 after A8W8 follow-up | 43.69 | 527.75 | 0 |
| 10 W8A8 auto follow-up (`enabled=[]`) | 42.42 | 528.63 | 0 |
| selected 50 (128-768 visual cap) | 43.69 | 528.02 | 0 |
| 10 | 23.87 | 534.30 | 0 |
| 30 | 24.21 | 537.48 | 0 |
| 150 | 27.81 | 504.85 | 0 |

## Visual-token throughput ablation

当前代码重新验证 10-sample benchmark + PyTorch-50 token_F1 代理：

| variant | TTFT ms | Throughput tok/s | avg visual tokens | token_F1 vs PyTorch50 | 结论 |
|---|---:|---:|---:|---:|---|
| default/no cap | 55.29 | 524.48 | ~770 | ~92.29% (历史) | prompt KV 较长 |
| 128-768 | 42.80 / selected50 43.69 | 528.14 / selected50 528.02 | 568.72 | 91.90% | 默认启用，语义/吞吐平衡 |
| 128-512 | 32.03 | 532.58 | 374.00 | 88.65% | opt-in 高吞吐 |
| 128-256 | 19.47 | 535.48 | 189.66 | 86.18% | opt-in 激进吞吐 |

## Acext decode aligned-50 ablation

使用 `analysis_outputs/qwen3_vl_pytorch_50_answers.json` 中的 50 个 question_id 对齐重测；正式 accuracy 仍不可得，记录相对默认输出的 exact/similarity 与退化输出率作为语义代理。

| variant | TTFT ms | Throughput tok/s | degenerate | exact vs default | sim vs default | 结论 |
|---|---:|---:|---:|---:|---:|---|
| default Q8 lm_head | 61.84 | 520.69 | 0/50 | 1.00 | 1.000 | Q8 fallback |
| native W8A8 decode auto | 61.26 | 518.52 | 0/50 | 0.28 | 0.753 | auto 启用 qkv，但端到端吞吐下降 |
| Triton W8A8 lm_head | 61.60 | 468.85 | 0/50 | 0.28 | 0.676 | scalar argmax 慢于 Q8 fused argmax |
| Acext A8W8 lm_head | 61.41 | 484.60 | 0/50 | 0.10 | 0.648 | full logits + activation quantization 成本高 |
| Acext A16W8 int8_pc lm_head | 61.37 | 521.60 | 0/50 | 0.42 | 0.832 | 默认启用；TTFT/吞吐均优于 Q8 fallback |
| Acext A16W4 int4_pc lm_head | 62.08 | 488.19 | 0/50 | 0.04 | 0.546 | INT4 per-channel 漂移过大 |
| Acext A16W4 int4_g64 lm_head | 61.51 | 497.70 | 0/50 | 0.08 | 0.582 | groupwise 精度/吞吐均未达默认 |
| Acext A16W4 int4_g128 lm_head | 61.45 | 508.56 | 1/50 | 0.04 | 0.568 | INT4 中吞吐最好但语义漂移仍大 |

## accuracy sanity check

- 10/30/150 samples 均生成非空答案：empty prediction = 0。
- 本地数据集不包含标准答案列，无法在 `benchmark.py` 内计算正式 accuracy；抽样输出格式正常，未出现异常/空答/格式损坏。
- 对 `smoothquant-gptq` 量化模型做了 100-sample drift 评估：`last_token_top1_agreement_rate=1.0`、`avg_last_token_logit_cosine=0.9975`、`avg_last_token_hidden_cosine=0.9961`、`avg_token_f1=0.7924`、`candidate_empty_output_count=0`、`candidate_abnormally_short_output_count=1`。
- 当前默认提交自动启用 rowwise-Q8 **weight_scale codebook** 与 runtime Q8 text decode；旧 SmoothQuant+GPTQ codebook 保留在包内但默认不使用。

## 放弃的策略及原因

- INT8 KV / Sage qk_i8/v_i8：源项目历史 10-sample 吞吐低于当前默认 BF16 Sage 路径，默认关闭，保留开关。
- W4A4 speculative draft：依赖 draft checkpoint 和接受率，收益不稳定，默认关闭。
- vision encoder 量化：当前没有稳定实测收益和精度边界，未默认启用。
- 整套 `JunkratDev` wrapper 覆盖：会与 `Junkrat` 量化 dispatch 冲突，改为只融合高收益 processor 分辨率策略。
- 默认直接把离线码本 attach 成 `weight_int8`：即使不启用 W8A8，也会因额外 buffers 拖慢 packed-Q8 默认路径；现在仅在 `JUNKRAT_ENABLE_W8A8_DECODE=1` 或 `JUNKRAT_ENABLE_W8A8_LM_HEAD=1` 时 attach int8 buffers。

## 下一步 3 个优化点

1. 将 `JunkratDev` 的 prefill graph bucket 和当前量化 wrapper 更深合并，减少首轮 prefill overhead。
2. 对 Sage INT8 KV 做分样本 A/B，找到只对长 decode 启用的安全阈值。
3. 若继续推进 W8A8/WeightOnly，需要做真正 fused GEMV+argmax 或复用 Acext 输出归约接口；当前 WeightOnly int8_pc 已默认启用，A8W8/INT4 仍需新 fused kernel 才可能超过默认。
