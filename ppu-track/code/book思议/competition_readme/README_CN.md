# AICAS 2026 - 面向AI芯片的VLM高效推理与优化赛道

##  目录
- [概述](#概述)
- [代码结构](#代码结构)
- [核心文件](#核心文件)
- [快速开始](#快速开始)
- [评测指标](#评测指标)
- [比赛规则](#比赛规则)
- [重要提示](#重要提示)
- [提交指南](#提交指南)


## 概述

本次竞赛专注于优化视觉语言模型（VLM）的推理性能。参赛者需要修改 `evaluation_wrapper.py` 中的 `VLMModel` 类，在保持准确率的同时提升首 Token 时间（TTFT）和吞吐量（Throughput）。

## 代码结构

```
AICASGC/
├── benchmark.py              # 基准测试脚本
├── evaluation_wrapper.py     # 模型包装器（选手在此实现优化）
├── requirements.txt          # Python 依赖包
├── data/                     # 验证数据集
│   ├── data-*.arrow          # 数据集文件
│   ├── dataset_info.json     # 数据集元信息
│   └── state.json            # 数据集状态
├── Qwen3-VL-2B-Instruct/    # 模型权重目录（需要选手自行下载）
└── README.md / README_CN.md   # 说明文档
```


## 核心文件

- **`benchmark.py`** - 自测基准脚本（⚠️ **不建议修改**）
- **`evaluation_wrapper.py`** - 模型包装器，参赛者在此实现优化
- **`Qwen3-VL-2B-Instruct/`** - 竞赛模型权重（需要选手自行下载，见"快速开始"部分）
- **`data/`** - 验证数据集
- **`requirements.txt`** - Python 依赖包

## 快速开始

### 0. 下载模型（首次使用）

模型文件较大，需要单独下载。请先创建模型目录，然后下载模型：

```bash
# 创建模型目录
mkdir -p Qwen3-VL-2B-Instruct

# 安装 huggingface_hub（如果未安装）
pip install -U huggingface_hub

# 设置镜像源（国内用户推荐，加速下载）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型到指定目录
huggingface-cli download \
  --resume-download \
  Qwen/Qwen3-VL-2B-Instruct \
  --local-dir ./Qwen3-VL-2B-Instruct \
  --local-dir-use-symlinks False
```

**注意：**
- 模型大小约 4-5GB，下载可能需要一些时间
- 如果下载中断，可以重新运行命令，会自动续传（`--resume-download`）
- 下载完成后，`Qwen3-VL-2B-Instruct/` 文件夹会包含所有模型文件
- 确保有足够的磁盘空间（至少 5GB）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 100
```

### 3. 实现你的优化

编辑 `evaluation_wrapper.py` 中的 `VLMModel` 类。优化采用**模块化设计**，每个优化方向对应一个独立方法。

#### 3.1 探索模型结构（可选）

在开始优化前，可以先探索模型结构，了解优化目标：

```python
class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        # ... 加载模型 ...
        
        # 可选：探索模型结构
        self._explore_model_structure()  # 会打印模型结构信息
```

#### 3.2 启用优化方法

在 `__init__` 方法中，通过注释/取消注释来启用/禁用不同的优化：

```python
class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        # ... 加载模型 ...
        
        # ================================================================
        # 选手优化区域 - 启用/禁用优化方法
        # ================================================================
        
        # 1. Vision Encoder 加速（优化大分辨率图像处理）
        # self._optimize_vision_encoder()
        
        # 2. KV Cache 管理（优化生成过程中的内存碎片）
        # self._optimize_kv_cache()
        
        # 3. 跨模态融合层优化（优化 Cross-modal Connector）
        # self._optimize_cross_modal_connector()
        
        # 4. Flash Attention 优化
        # self._enable_flash_attention()
        
        # 5. 量化优化
        # self._apply_quantization()
```

#### 3.3 实现优化代码

在各个优化方法中实现你的优化逻辑。例如，优化 Vision Encoder：

```python
def _optimize_vision_encoder(self):
    """在 evaluation_wrapper.py 中找到这个方法，实现你的优化"""
    
    # 示例：替换注意力算子
    # from your_optimization import optimized_attention
    # if hasattr(self._model, 'vision_model'):
    #     for layer in self._model.vision_model.encoder.layers:
    #         layer.self_attn.forward = optimized_attention
    
    # TODO: 实现你的 Vision Encoder 优化
    pass
```




### 4. 测试你的优化模型

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result_optimized.json \
    --num-samples 100
```

### 5. 生成完整结果用于提交

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 5000
```

## 评测指标

最终得分计算公式：

```
最终得分 = 0.4 × 准确率 + 0.3 × TTFT提升率 + 0.3 × 吞吐量提升率
```

### 指标说明

- **TTFT (Time To First Token)**: 从输入准备到生成第一个 Token 的时间（毫秒）
  - 包含：图像编码、文本编码、跨模态交互、Prefill 阶段、第一个 Token 生成
  - Baseline: ~80ms
  - 提升率 = (Baseline - 你的TTFT) / Baseline

- **Throughput (吞吐量)**: 端到端 Token 生成速率（tokens/秒）
  - Baseline: ~55 tokens/sec
  - 提升率 = (你的吞吐量 - Baseline) / Baseline

- **Accuracy (准确率)**: 验证集上的 VQA 准确率（5000 个样本）
  - 支持多个标准答案的软匹配

## 比赛规则

###  重要规则


1. **不要修改 `benchmark.py`**
   - 此基准脚本仅用于自测
   - 最终评测将使用独立的官方基准系统
   - 修改此文件可能导致本地结果与最终评测结果不一致

2. **仅修改 `evaluation_wrapper.py`**


3. **保持必需的属性**
   - `VLMModel` 类必须暴露 `processor`、`model` 和 `device` 属性
   - Benchmark 使用这些属性来访问模型和处理器
   - `generate()` 方法是可选的，主要用于调试

4. **禁止行为**
   - 禁止硬编码答案
   - 禁止修改数据集
   - 禁止使用外部 API 或服务
   - 所有优化必须是本地且自包含的




### 优化方向
- 鼓励实现算子替换与内核优化：使用Triton、CUDA C++等重写或替换标准算子实现（如Attention、LayerNorm、Conv2d等）

- 鼓励实现内存与缓存优化：优化KV Cache内存布局、减少内存碎片、优化显存访问模式


- 鼓励实现编译与图优化：使用torch.compile进行计算图优化、自定义内核调度


- 鼓励实现注意力机制优化：实现Flash Attention、内存高效注意力、稀疏注意力

- 鼓励实现生成过程优化：优化解码策略、缓存管理、生成配置参数


**不允许：**
- 使用外部服务：禁止调用外部API、云服务或任何需要网络连接的功能

- 数据与答案作弊：禁止使用测试数据进行训练、预计算答案、硬编码输出

- 模型替换与篡改：希望选手着重做算子优化，不要用额外的数据集去训练模型、改变模型架构、直接修改权重数值等。


- 过拟合优化：禁止针对特定评测样本进行条件分支或特殊处理

- 黑盒工具套用：仅修改配置文件而无实质性代码贡献的行为不被认可

- 环境操纵：禁止通过修改系统环境、GPU频率锁定等方式干扰公平评测



## 重要提示

### 样本选择

- 提供的 `benchmark.py` 使用**固定顺序**（从索引 0 开始的前 N 个样本）
- 运行 `--num-samples 100` 时，会评测样本 0-99
- 这确保了本地自测的可复现性
- **注意**：竞赛委员会使用的官方评测系统可能采用不同的采样策略（包括随机采样）进行最终验证

### 硬件信息

基准测试会自动记录详细的硬件信息：
- Python 版本、PyTorch 版本、CUDA 版本
- GPU 名称、显存、计算能力
- CPU 型号、核心数、频率
- 系统信息（操作系统、内核、架构）
- PPU 信息（如果可用）

这些信息保存在 `result.json` 的 `system_info` 字段中，用于统计分析。

### 性能测量

- **预热**：在实际测量前使用 10 个样本进行 GPU 预热
- **TTFT 测量**：测量从输入准备到第一个 Token 的时间（包含所有预处理）
- **吞吐量测量**：测量生成 128 个 Token 的端到端时间
- **状态隔离**：在测量之间清理 GPU 缓存，确保公平性

### 随机种子

- `--random-seed` 参数仅影响 PyTorch 的随机数生成器
- 它**不会**影响样本选择顺序（始终是固定的）
- 用于模型推理随机性的可复现性

### 输出格式

`result.json` 文件包含：
```json
{
  "system_info": {
    "timestamp": "...",
    "python_version": "...",
    "torch_version": "...",
    "cuda_version": "...",
    "gpu_name": "...",
    ...
  },
  "performance": {
    "avg_ttft_ms": 90.55,
    "avg_throughput_tokens_per_sec": 57.77
  },
  "answers": [
    {
      "question_id": 34602,
      "prediction": "你的答案文本"
    },
    ...
  ]
}
```

## 提交指南

### 初赛提交必需文件

1. **`result.json`** - 通过运行 `benchmark.py` 生成
   - 包含所有样本的预测 
   - 必须包含有效的 `performance` 指标
   - **重要**：上传到天池平台的 `result.json` 仅用于参考。最终成绩将由竞赛委员会使用标准化硬件和官方评测系统进行评测。

2. **你的优化代码** - 包含你优化的 `VLMModel` 类的 `evaluation_wrapper.py`

3. **Docker 镜像**- 包含你优化环境的容器



### 评测流程

1. **自测**：使用提供的 `benchmark.py` 在本地测试你的优化
2. **提交**：将你的 `result.json` 上传到天池平台（仅用于参考）
3. **官方评测**：竞赛委员会将使用以下方式评测你的代码：
   - 提交Docker镜像
   - 标准化硬件环境
   - 官方评测代码
   - 完整验证集，随机采样进行验证
4. **最终排名**：基于官方评测系统计算的最终得分



## 祝你好运！

希望你会专注于算子级优化、内核替换和高效的内存管理。记住：准确率和速度同样重要！祝你好运！




