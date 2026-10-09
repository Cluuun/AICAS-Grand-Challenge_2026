# 2026-05-20 MNN SmolVLM 全量准确率服务器评测计划

## 目标

在 `/3.2T/work/fpga-fa` 服务器环境中评估 MNN SmolVLM 的准确率，并为后续 KV260/ZU15EG 板端一致性与 PL shadow/replacement 做基线。

需要区分三件事：

1. MNN 模型大概率是 INT8/量化执行路径，但不是 llama.cpp GGUF `Q8_0` 原生格式。
2. 服务器全量结果用于工程筛选；比赛最终仍需 KV260/ZU15EG 板端同脚本、同 sample、同解码参数复现。
3. PL 接入前先做 MNN CPU baseline；PL 接入后先跑 shadow mode 比较 tensor，再替换输出。

## 推荐目录

```text
/3.2T/work/fpga-fa/
  aicas_semi/
    AICAS-code-复赛*.md
    code/
    SERVER_FULL_EVAL_PLAN_20260520.md
    scripts/
    docs/
  third_party/MNN/
  models/
    SmolVLM2-500M-Video-Instruct-MNN/
    gguf/
  datasets/
    OCRBench/
      FullTest.json
      data/
  eval_runs/
    mnn_smolvlm_20260520/
      manifests/
      server_cpu/
      server_cuda_probe/
      board_compare/
      logs/
```

缓存不要进 `/home/luci`：

```bash
export HF_HOME=/3.2T/work/fpga-fa/.cache/huggingface
export MODELSCOPE_CACHE=/3.2T/work/fpga-fa/.cache/modelscope
export XDG_CACHE_HOME=/3.2T/work/fpga-fa/.cache
```

## P0：模型与运行环境确认

先确认 MNN 模型目录是否已经存在：

```bash
find /3.2T/work/fpga-fa -maxdepth 5 \
  \( -name llm.mnn -o -name llm.mnn.weight -o -name visual.mnn -o -name visual.mnn.weight -o -name llm_config.json \) \
  -print
```

MNN 模型目录至少应包含：

```text
config.json
llm_config.json
llm.mnn
llm.mnn.weight
visual.mnn
visual.mnn.weight
tokenizer.txt / tokenizer.mtok / tokenizer.model 等 tokenizer 文件
```

若远程没有模型，优先从本机或板端已有目录同步，不优先重新下载。

当前已确认远程存在：

```text
/3.2T/work/fpga-fa/models/SmolVLM2-500M-Video-Instruct-MNN/
  config.json
  llm_config.json
  llm.mnn
  llm.mnn.weight
  visual.mnn
  visual.mnn.weight
  tokenizer.txt
  embeddings_bf16.bin
```

`config.json` 当前默认：

```json
{
  "backend_type": "cpu",
  "thread_num": 4,
  "precision": "low",
  "memory": "low"
}
```

因此第一轮 accuracy baseline 应按 MNN CPU/low precision 口径跑。

## P0.5：数据集上传完成后的验收

数据集目标目录固定为：

```text
/3.2T/work/fpga-fa/datasets/OCRBench/
  FullTest.json
  data.zip
  data/
```

本机来源校验值：

```text
FullTest.json sha256 = DF11B622647DB81F4D4D7DBDED41D78569EC721BBB4436BF8237BB8B3B5935C6
data.zip      sha256 = 4F5EF67788ADFE37F7DAD5D8B25570A81E62D5E180C1D0C47D9B489401EEB6BE
```

上传完成后先在服务器执行：

```bash
cd /3.2T/work/fpga-fa/datasets/OCRBench

sha256sum FullTest.json data.zip

if [ ! -d data ]; then
  unzip -q data.zip
fi

find data -type f | wc -l
du -sh FullTest.json data.zip data
```

期望大致为：

```text
FullTest.json ~= 15 MB
data.zip      ~= 2.8 GB
data/         ~= 2.9 GB
image files   ~= 45509
```

若服务器已有 `/3.2T/work/fpga-fa/inference/problem/data_extracted` 或其他 OCRBench 目录，也可以复用，但必须先确认 `FullTest.json` 内的 image path 能在 `mnn_acc_eval.py --image_folder` 指向的目录下解析成功。

## P1：固定官方 30 条 sample

复赛官方 accuracy 是 30 条，分层随机采样，分布 `3+3+3+3+3+3+12`。

```bash
mkdir -p /3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/manifests
cd /3.2T/work/fpga-fa/aicas_semi/code

python sample.py \
  -i /3.2T/work/fpga-fa/datasets/OCRBench/FullTest.json \
  -o /3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/manifests/sample_30.json
```

这个 `sample_30.json` 后续必须固定保存，用于：

- 服务器 MNN CPU
- 服务器 MNN CUDA probe
- KV260/ZU15EG MNN CPU
- KV260/ZU15EG MNN+PL shadow/replacement
- llama.cpp Q8_0 对比

## P2：MNN CPU baseline

先使用 MNN CPU 后端跑：

先启动 MNN server。具体二进制位置以服务器已有构建为准；推荐运行时统一把模型目录和端口写进日志：

```bash
mkdir -p /3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/logs

export AICAS_MNN_MODEL=/3.2T/work/fpga-fa/models/SmolVLM2-500M-Video-Instruct-MNN/config.json
export AICAS_MNN_PORT=8091

# 示例，实际 server 命令按当前 aicas_mnn_server/llm_demo 包装脚本替换：
# /path/to/aicas_mnn_server --model "$AICAS_MNN_MODEL" --port "$AICAS_MNN_PORT" \
#   > /3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/logs/server_cpu_8091.log 2>&1
```

```bash
python mnn_acc_eval.py \
  --server http://127.0.0.1:8091 \
  --image_folder /3.2T/work/fpga-fa/datasets/OCRBench/data \
  --OCRBench_file /3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/manifests/sample_30.json \
  --output_folder /3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/server_cpu \
  --save_name SmolVLM2_MNN_CPU_30
```

随后扩展：

```text
sample_300.json
sample_1000.json
sample_full.json
```

建议顺序是 `30 -> 300 -> 1000 -> full`，不要一开始直接全量。

每轮结束后保存：

```bash
sha256sum \
  /3.2T/work/fpga-fa/models/SmolVLM2-500M-Video-Instruct-MNN/config.json \
  /3.2T/work/fpga-fa/models/SmolVLM2-500M-Video-Instruct-MNN/llm.mnn \
  /3.2T/work/fpga-fa/models/SmolVLM2-500M-Video-Instruct-MNN/llm.mnn.weight \
  /3.2T/work/fpga-fa/models/SmolVLM2-500M-Video-Instruct-MNN/visual.mnn \
  /3.2T/work/fpga-fa/models/SmolVLM2-500M-Video-Instruct-MNN/visual.mnn.weight \
  > /3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/model_sha256.txt
```

## P3：CUDA 后端可行性探测

服务器有 8 张 P40，可以探索 MNN CUDA 后端，但需要先小样本确认：

1. MNN 是否已编译 CUDA 后端。
2. MNN-LLM server 是否允许选择 CUDA backend。
3. CUDA backend 是否覆盖当前 SmolVLM 的量化 `Convolution/DenseConvInt8` 路径。
4. 输出是否与 CPU baseline 一致。

P40 是 Pascal，支持 DP4A，但没有后续 Tensor Core。CUDA 可能提升 vision/大 conv，也可能因为 MNN LLM INT8 kernel 覆盖不足而没有收益。先以 `sample_30` 验证准确率和速度，再决定是否全量。

当前远程已确认 GPU：

```text
Tesla P40 x8
driver 570.86.15
```

但当前只确认有 A53 构建：

```text
/3.2T/work/fpga-fa/third_party/MNN/build-a53-llm
```

尚未确认已有 x86 CUDA 版 MNN-LLM。因此 CUDA 探测应作为单独子任务：

```text
build-linux-cuda-llm/
  CMAKE_CUDA_ARCHITECTURES=61
  MNN_CUDA=ON
  MNN_BUILD_LLM=ON
```

验收顺序：

1. CUDA 版 server 能加载同一 MNN 模型。
2. `sample_5` 或 `sample_30` 输出不崩溃。
3. raw answer 与 CPU baseline 做逐条 diff。
4. 速度有实际收益后，再考虑 `sample_300/full`。

如果 CUDA backend 对量化 `DenseConvInt8TiledExecutor` 覆盖不足，可能会回退 CPU 或产生额外 layout/copy 开销；这种情况下不要把 CUDA 作为全量主口径。

建议记录：

```text
backend
GPU id
CUDA/MNN commit
model sha256
sample json sha256
raw answer
score
latency
tokens/s
```

## P4：全量评测

全量评测仅在以下条件满足后启动：

- 30 条 CPU baseline 通过。
- 300/1000 条没有明显系统性错误。
- CUDA probe 如果启用，30 条输出与 CPU 基本一致。
- 输出目录和日志路径已经固定。

全量结果目录建议：

```text
/3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/server_cpu/full/
/3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/server_cuda_probe/full/
```

全量任务建议交给远程 agent 后按下面的 checklist 执行：

```text
1. 验证数据集 sha256 和图片数量。
2. 生成并冻结 sample_30/sample_300/sample_1000/sample_full。
3. 启动 MNN CPU server，记录 server 命令和日志。
4. 跑 sample_30，保存 raw result 和 score。
5. 跑 sample_300/sample_1000，检查系统性错误。
6. 只有前面正常才跑 sample_full。
7. 若尝试 CUDA，先跑 sample_30，不要直接跑 full。
8. 每个结果目录保存：sample json、output json、server log、model sha256、git/script sha。
```

## P5：板端一致性

服务器结果不能直接替代板端提交结果。板端一致性最少要做：

1. 同一个 `sample_30.json`。
2. 同一个 MNN 模型目录和 sha256。
3. 同一个 prompt 模板与解码参数。
4. 保存 raw answer 并和服务器逐条 diff。
5. 若接 PL，先 shadow mode：CPU 输出用于最终回答，PL 输出只做 tensor/error/profiling 记录。

PL replacement 只有在 shadow mode tensor 误差稳定后再打开。

板端一致性建议与服务器共享同一份：

```text
/3.2T/work/fpga-fa/eval_runs/mnn_smolvlm_20260520/manifests/sample_30.json
```

不要在板端重新 sample，否则结果不可直接对比。

## 建议结论

短期最值得跑的是：

1. 服务器 MNN CPU `sample_30`。
2. 服务器 MNN CPU `sample_300`。
3. CUDA backend `sample_30` 探测。
4. 板端同 `sample_30` 一致性复测。
5. 全量只在上述通过后执行。
