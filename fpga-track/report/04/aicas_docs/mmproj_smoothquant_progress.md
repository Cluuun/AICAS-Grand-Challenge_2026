# mmproj SmoothQuant PTQ 结果说明

## 目标

本次工作的目标是在当前仓库的 `mmproj` 路径上实现一种适配现有约束的 SmoothQuant 风格 PTQ 方案，并满足以下条件：

- 只处理 `mulmat / GEMM` 类线性算子
- 只覆盖当前 `mmproj` 路径，不扩展到文本主干
- 激活使用静态非对称 `per-tensor`
- 权重支持两种静态对称量化：
  - `per-channel`
  - `per-tensor`
- 当前阶段只验证量化精度与推理正确性
- 当前阶段不接 NPU 执行，只跑通 CPU reference path

评测入口保持与现有流程一致：

- [`AICAS/acc_eval.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/acc_eval.py)
- [`AICAS/scripts/run_local_acc_eval.sh`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/scripts/run_local_acc_eval.sh)

## 方法

### 1. SmoothQuant 等价变换

对每个候选线性层：

```text
y = xW + b
```

按输入通道构造 smoothing factor：

```text
s_j = (a_j ^ alpha) / (w_j ^ (1 - alpha) + eps)
```

其中：

- `a_j`：该层输入激活第 `j` 个输入通道的代表性幅值
- `w_j`：该层权重第 `j` 个输入通道的代表性幅值
- `alpha`：搜索超参数
- `eps`：数值稳定项

随后做等价变换：

```text
x' = x / s
W' = W * s
```

这次实现中：

- `a_j` 采用校准集上的逐输入通道 `max_abs`
- `w_j` 采用权重矩阵按输入通道的 `max_abs(W[:, j])`
- `alpha` 搜索集合为 `[0.3, 0.4, 0.5, 0.6, 0.7]`
- `eps = 1e-6`

### 2. 量化格式

激活：

- 静态非对称 `uint8`
- `per-tensor`
- 推理时统一使用每层固定的 `act_scale + act_zero_point`

权重：

- 静态对称 `int8`
- 支持两种模式：
  - `per-channel`
  - `per-tensor`

累加与反量化：

- `int8 x int8 -> int32`
- 之后按 `act_scale * weight_scale` 反量化到 `fp32`

Bias：

- 保持高精度
- 不做量化

### 3. 推理时的实际执行方式

这次实现不是只做离线导出，而是真正修改了 `mmproj` 的运行时路径。

实际执行时：

- 权重使用已经融合 `W * s` 后再量化得到的 `int8` 权重
- 激活在量化前先执行逐输入通道的 `x / s`
- 然后进入现有 `W8A8` 路径

也就是说，当前 CPU reference path 真正执行的是 SmoothQuant 版 `W8A8`。

## 候选层范围

本次只覆盖当前 `mmproj` 路径中识别出的 `73` 个 `mulmat` 线性层。

具体包括：

- `mm.model.fc.weight` x1
- `v.blk.{0..11}.attn_q.weight` x12
- `v.blk.{0..11}.attn_k.weight` x12
- `v.blk.{0..11}.attn_v.weight` x12
- `v.blk.{0..11}.attn_out.weight` x12
- `v.blk.{0..11}.ffn_up.weight` x12
- `v.blk.{0..11}.ffn_down.weight` x12

形状分布：

- `projector`: `12288 -> 960`
- `attn_q / attn_k / attn_v / attn_out`: `768 -> 768`
- `ffn_up`: `3072 -> 768`
- `ffn_down`: `768 -> 3072`

这些候选层全部属于当前 `mmproj` 图中的 `prefill mulmat`，没有 `decode GEMV`，因此这一轮没有 `C` 类跳过项。

## 校准数据

校准集沿用现有数据流程，从 [`AICAS/calib/calib_manifest.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/calib/calib_manifest.json) 读取。

特点：

- 校准样本数：`256`
- 数据根目录：`AICAS/data`
- 显式排除评测集 [`AICAS/sampled.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/sampled.json)
- 按数据集轮转抽样

## 实现文件

### 修改的现有文件

- [`tools/mtmd/clip.cpp`](/home/gugugu/work/llama.cpp-kv260-20260407/tools/mtmd/clip.cpp)
- [`AICAS/tools/mmproj_collect_act_stats.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_collect_act_stats.py)
- [`AICAS/tools/mmproj_pack_gguf.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_pack_gguf.py)

### 新增脚本

- [`AICAS/tools/mmproj_smoothquant_prepare.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_smoothquant_prepare.py)
- [`AICAS/tools/mmproj_smoothquant_make_policy.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_smoothquant_make_policy.py)
- [`AICAS/tools/mmproj_smoothquant_experiment.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_smoothquant_experiment.py)

## 关键产物

### 校准与候选配置

- 原始激活统计：
  - [`AICAS/artifacts/smoothquant_raw_act_stats.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/smoothquant_raw_act_stats.json)
- SmoothQuant 候选：
  - [`AICAS/artifacts/smoothquant_candidates.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/smoothquant_candidates.json)

### per-channel 结果

输出目录：

- [`AICAS/output/smoothquant`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant)

关键文件：

- 最终汇总：
  - [`AICAS/output/smoothquant/final_summary.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant/final_summary.json)
- 最佳 policy：
  - [`AICAS/output/smoothquant/full-alpha-0_5-minmax/policy.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant/full-alpha-0_5-minmax/policy.json)
- 最佳 pack 摘要：
  - [`AICAS/output/smoothquant/full-alpha-0_5-minmax/pack_summary.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant/full-alpha-0_5-minmax/pack_summary.json)
- 最佳量化 mmproj：
  - [`AICAS/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf)
- 最佳评测结果：
  - [`AICAS/output/smoothquant/full-alpha-0_5-minmax/result.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant/full-alpha-0_5-minmax/result.json)

### per-tensor 结果

输出目录：

- [`AICAS/output/smoothquant-per-tensor`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant-per-tensor)

关键文件：

- 最终汇总：
  - [`AICAS/output/smoothquant-per-tensor/final_summary.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant-per-tensor/final_summary.json)
- 最佳 policy：
  - [`AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/policy.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/policy.json)
- 最佳 pack 摘要：
  - [`AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/pack_summary.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/pack_summary.json)
- 最佳量化 mmproj：
  - [`AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/mmproj.gguf`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/mmproj.gguf)
- 最佳评测结果：
  - [`AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/result.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant-per-tensor/full-alpha-0_5-minmax/result.json)

## 实验结果

### 基线

本轮使用你确认的 fp16 baseline：

- `53 / 100`

并据此设定门槛：

- `threshold = 48 / 100`

### SmoothQuant + weight per-channel

pilot 24 样本：

- `alpha=0.3` -> `13 / 24`
- `alpha=0.4` -> `13 / 24`
- `alpha=0.5` -> `14 / 24`
- `alpha=0.6` -> `13 / 24`
- `alpha=0.7` -> `13 / 24`

full 100 样本：

- `alpha=0.5 + minmax` -> `56 / 100`
- `alpha=0.3 + minmax` -> `49 / 100`

最终最佳：

- `alpha=0.5`
- 激活范围策略：`minmax`
- 权重量化：`per-channel`
- 启用层数：`73 / 73`
- 跳过层数：`0`
- 最终精度：`56 / 100`

相对 fp16 baseline：

- `+3`

### SmoothQuant + weight per-tensor

pilot 24 样本：

- `alpha=0.3` -> `11 / 24`
- `alpha=0.4` -> `13 / 24`
- `alpha=0.5` -> `14 / 24`
- `alpha=0.6` -> `12 / 24`
- `alpha=0.7` -> `14 / 24`

full 100 样本：

- `alpha=0.5 + minmax` -> `54 / 100`
- `alpha=0.7 + minmax` -> `53 / 100`

最终最佳：

- `alpha=0.5`
- 激活范围策略：`minmax`
- 权重量化：`per-tensor`
- 启用层数：`73 / 73`
- 跳过层数：`0`
- 最终精度：`54 / 100`

相对 fp16 baseline：

- `+1`

相对 per-channel：

- `-2`

## 结论

1. 当前 `mmproj` 路径上的 SmoothQuant PTQ 已经真实跑通。
2. 当前最佳方案是：
   - SmoothQuant
   - 激活静态非对称 `per-tensor`
   - 权重静态对称 `per-channel`
   - `alpha=0.5`
   - `minmax`
3. `per-tensor` 权重版本也可用，且仍然达到 `54 / 100`，说明它是一个可接受的过渡版本。
4. 因此如果后续 NPU 当前只稳定支持 `weight per-tensor`，可以先落这个过渡版本：
   - SmoothQuant + `W8A8`
   - 权重先走 `per-tensor`
   - 等 `per-channel` 权重量化路径完善后，再切换到正式最佳配置

## 如何复现

### 复现 per-channel 最佳实验

```bash
python3 AICAS/tools/mmproj_smoothquant_experiment.py \
  --server-bin build-host/bin/llama-server \
  --python python3 \
  --skip-venv \
  --baseline-score 53 \
  --port-base 18120 \
  --request-timeout 120 \
  --request-retries 0 \
  --retry-delay 0.5
```

### 复现 per-tensor 实验

```bash
python3 AICAS/tools/mmproj_smoothquant_experiment.py \
  --server-bin build-host/bin/llama-server \
  --python python3 \
  --skip-venv \
  --baseline-score 53 \
  --port-base 18220 \
  --request-timeout 120 \
  --request-retries 0 \
  --retry-delay 0.5 \
  --weight-granularity per_tensor \
  --output-root AICAS/output/smoothquant-per-tensor
```

## 对接 NPU 的建议

从当前实现和实验结果看，后续上 NPU 时最实用的路线是：

- 保留离线 `W' = W * s`
- 在 host 侧 activation pack/quant 阶段执行 `x / s`
- NPU 继续做普通 `W8A8 GEMM`

这样：

- 不需要先改 GEMM 主阵列
- 可以先支持 `SmoothQuant + weight per-tensor`
- 后续再演进到 `SmoothQuant + weight per-channel`

