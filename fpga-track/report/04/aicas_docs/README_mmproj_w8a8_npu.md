# mmproj W8A8 + ggml-npu 使用说明

本文档说明如何使用 `AICAS` 下现有脚本与产物，并让 **被量化的 mmproj 层优先走 NPU** 计算。

## 1. 适用范围

- 模型：`SmolVLM2-500M-Video-Instruct`
- 量化范围：`mmproj` 内 `mul_mat` 线性层
- 量化元数据：`aicas.w8a8.*`
- 运行目标：W8A8 层使用 NPU；非 W8A8 层保持原路径（F16 fallback）

## 2. 关键文件

- 进度总览：`AICAS/mmproj_w8a8_progress.md`
- 基础工作流：`AICAS/README_mmproj_w8a8.md`
- 量化脚本：
  - `AICAS/tools/mmproj_manifest.py`
  - `AICAS/tools/mmproj_make_calib_manifest.py`
  - `AICAS/tools/mmproj_collect_act_stats.py`
  - `AICAS/tools/mmproj_calibrate.py`
  - `AICAS/tools/mmproj_make_layer_policy.py`
  - `AICAS/tools/mmproj_pack_gguf.py`
- 运行时实现：`tools/mtmd/clip.cpp`
- NPU 后端实现：`ggml/src/ggml-npu/*`
- KV260 交叉编译与部署：`AICAS/scripts/deploy_and_run_kv260_npu_eval.sh`

## 3. 快速开始（直接用现有量化产物）

如果你已经有量化后的 mmproj（例如 `mixed_v1`），最短路径如下：

```bash
# 1) 构建（需启用 NPU）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NPU=ON -DGGML_CCACHE=OFF
cmake --build build --config Release -j8 --target llama-server

# 2) 用 NPU 跑本地精度评测（被量化层会走 NPU）
MTMD_BACKEND_DEVICE=NPU \
bash AICAS/scripts/run_local_acc_eval.sh \
  --server-bin build/bin/llama-server \
  --model f16 \
  --mmproj AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
  --output-folder /tmp/local-acc-eval-w8a8-mixed-v1-npu \
  --save-name mixed_v1_npu
```

### 3.1 KV260 交叉编译

如果当前板端正在跑别的任务，或者你只想先拿到独立的 NPU 交叉编译产物，可直接运行：

```bash
bash AICAS/scripts/deploy_and_run_kv260_npu_eval.sh --build-only
```

默认产物与参数：

- 构建目录：`build-kv260-npu`
- 默认模型：`AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf`
- 默认 mmproj：`AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`
- 关键 CMake 开关：`-DGGML_NPU=ON`

### 3.2 KV260 部署脚本行为

`AICAS/scripts/deploy_and_run_kv260_npu_eval.sh` 在任何远端写入之前，会先做只读 readiness probe：

- `xmutil listapps` 中是否存在 `mynpu`
- `/dev/npu_kv260` 是否存在
- `npu_kv260` 内核模块是否已加载
- `/sys/bus/platform/devices` 中是否出现 NPU 平台设备（匹配 `npu` 或 `a0000000`）

只要任一条件不满足，脚本就会在 `scp`、远端 `mkdir`、远端 `llama-server` 启动之前直接退出。

NPU 运行默认值：

- `MTMD_BACKEND_DEVICE=NPU`
- `GGML_NPU_SPM_BYTES=524288`
- `GGML_NPU_ACC_BYTES=524288`
- `GGML_NPU_GUARD_BYTES=4096`
- `GGML_NPU_STAGE2_K_BYTES=4096`
- `NPU_CMA_SIZE=256M`

## 4. 从头生成量化产物（完整流程）

### 4.1 构建 layer 清单

```bash
python3 AICAS/tools/mmproj_manifest.py \
  --mmproj-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --profile-json /tmp/mmproj_profile.json \
  --output-manifest AICAS/artifacts/layer_manifest.json \
  --output-inventory AICAS/artifacts/mmproj_mulmat_inventory.json
```

### 4.2 构建校准图片清单

```bash
python3 AICAS/tools/mmproj_make_calib_manifest.py \
  --data-root AICAS/data \
  --eval-json AICAS/sampled.json \
  --output AICAS/calib/calib_manifest.json \
  --num-images 256
```

### 4.3 收集激活统计

建议用 `llama-mtmd-profiler` 收集（该步骤主要用于标定，不要求 NPU）：

```bash
python3 AICAS/tools/mmproj_collect_act_stats.py \
  --profiler-bin build-host/bin/llama-mtmd-profiler \
  --model-gguf AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
  --mmproj-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --calib-manifest AICAS/calib/calib_manifest.json \
  --output AICAS/artifacts/mmproj_act_stats.json \
  --limit 64 \
  --samples-per-tensor 4096
```

### 4.4 生成激活量化参数

```bash
python3 AICAS/tools/mmproj_calibrate.py \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --act-stats AICAS/artifacts/mmproj_act_stats.json \
  --calib-manifest AICAS/calib/calib_manifest.json \
  --output AICAS/artifacts/quant_params.json \
  --percentile-low 0.1 \
  --percentile-high 99.9
```

### 4.5 （可选）生成 mixed policy

```bash
python3 AICAS/tools/mmproj_make_layer_policy.py \
  --input-quant-params AICAS/artifacts/quant_params.json \
  --output AICAS/artifacts/layer_policy.v1.json \
  --fallback-kind ffn_up \
  --fallback-kind attn_out \
  --fallback-tensor mm.model.fc.weight
```

### 4.6 打包量化 mmproj GGUF

全量 W8A8：

```bash
python3 AICAS/tools/mmproj_pack_gguf.py \
  --input-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --quant-params AICAS/artifacts/quant_params.json \
  --output-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf \
  --output-quant-params AICAS/artifacts/pack_summary.full.json \
  --mode quantize
```

mixed_v1：

```bash
python3 AICAS/tools/mmproj_pack_gguf.py \
  --input-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --quant-params AICAS/artifacts/layer_policy.v1.json \
  --output-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
  --output-quant-params AICAS/artifacts/pack_summary.v1.json \
  --mode quantize
```

注意：
- `--quant-params` 与 `--output-quant-params` 不能是同一路径。

## 5. NPU 运行行为（现在的逻辑）

- 运行时会读取 `aicas.w8a8.*` 元数据。
- 若后端是 NPU（`MTMD_BACKEND_DEVICE=NPU`）：
  - W8A8 层参数会注册到 `ggml-npu`；
  - 被量化层走 `ggml_mul_mat` 并由 NPU 执行。
- 当前执行路径仍保持“激活先在 CPU 侧打包为有符号 int8，补偿项在 CPU 侧恢复”的实现；
  `ggml-npu` 目前不会把这一路切换成 `asymmetric_activations=true` 的硬件输入解释。
- 若不是 NPU 后端：
  - 保持原行为：检测到 AICAS W8A8 元数据后 mmproj 强制走 CPU 路径。

## 6. 调试与验证

### 6.1 推荐环境变量

```bash
export MTMD_BACKEND_DEVICE=NPU
export AICAS_MMPROJ_W8A8_DEBUG=1
```

可选（采样激活）：

```bash
export AICAS_MMPROJ_ACT_STATS_FILE=/tmp/mmproj_act_stats.json
export AICAS_MMPROJ_ACT_SAMPLES=4096
```

### 6.2 关键日志

运行时可关注：

- `registered <N> AICAS W8A8 tensors for NPU`  
  表示量化层参数已注册到 NPU 后端。
- `AICAS W8A8 metadata detected, forcing mmproj to CPU backend`  
  表示当前并未使用 NPU 后端（或 NPU 不可用），因此回落 CPU。

## 7. 常见问题

1. 设了 `MTMD_BACKEND_DEVICE=NPU` 但仍回落 CPU  
- 检查是否启用 NPU 编译：`-DGGML_NPU=ON`
- 确认命令没有禁用 mmproj offload（例如 `--no-mmproj-offload`）
- 查看日志里是否有 `CLIP using NPU backend`

2. KV260 NPU 部署脚本在 readiness probe 阶段退出
- 这是预期保护：脚本不会去修改板端状态
- 先确认 `xmutil listapps` 能看到 `mynpu`
- 再确认 `/dev/npu_kv260`、`npu_kv260` 模块、对应平台设备都已经准备好

3. `llama-mtmd-profiler` 链接报 NPU runtime 符号缺失  
- 这是环境链接问题（`npu_runtime` 动态库/路径），与量化脚本本身无关
- 可先使用 host 版本 profiler 做标定，再用 NPU 版本 `llama-server` 做推理/评测

4. 某些层没有走 W8A8  
- 检查打包摘要：`AICAS/artifacts/pack_summary*.json`
- `policy=F16_FALLBACK` 的层会保留原始路径
