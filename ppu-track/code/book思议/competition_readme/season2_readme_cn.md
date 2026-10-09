# 2026 AICASGC- PPU Track - 第二赛季 - 选手提交与说明


本文说明在天池提交后，由评测 Worker 拉取、解压并执行时的**ZIP 结构**、**依赖策略**以及**当前评测机软件环境**（供本地对齐参考）。**实际以评测方部署的 Worker 为准**，如有变更以组委会通知为准。

---

## 提交 ZIP 里应包含什么（推荐结构）

评测 Worker 会解压 ZIP，用官方下发的 **`benchmark.py`** 调用你包里的 **`evaluation_wrapper.py`**。下面是一种**最小可行**布局；文件也可放在**子目录**中（`evaluation_wrapper.py`  会在整个解压树内被搜索）。

```text
your_submission.zip
├── evaluation_wrapper.py      # 必选：实现 VLMModel
├── requirements.txt           # 可选：省略或与官方完全一致
├── my_kernel/                 # 可选：自定义算子、辅助模块（.py / 预编译 .so 等）
│   ├── __init__.py
│   └── ...
└── README.md                  # 可选：说明依赖与编译要求（评测脚本不读取，仅给人看）
```

| 路径 / 文件 | 是否必需 | 说明 |
|-------------|----------|------|
| `evaluation_wrapper.py` | **必需** | 实现 `VLMModel`。 |
| `requirements.txt` | 可选 | 见下文「依赖策略」。 |
| 其他 `.py`、资源、**预编译**扩展（如 `.so`） | 可选 | 会合并进任务工作目录；由你在 `evaluation_wrapper` 中 `load_library` / `import` / `ctypes` 加载。 |
| `benchmark.py` | **不必上传** | ZIP 内的 `benchmark.py` **会被忽略**，始终以官方基准为准。 |

**不要**在 ZIP 里夹带过大无关文件或与评测无关的二进制，以免拖慢下载与解压。


---

## 一、如何提交

### 1. 打包格式

- 提交物必须是 **ZIP 压缩包**（`.zip`），不要提交 rar/7z 等格式。
- 路径需合法（防止 Zip Slip，不要使用 `..` 等跳出目录的路径）。

### 2. 必需文件：`evaluation_wrapper.py`

- 评测会在解压目录（含子目录）中查找 **`evaluation_wrapper.py`** 。
- 其中应实现 **`VLMModel`**（与官方模板一致），供 `benchmark.py` 通过 `from evaluation_wrapper import VLMModel` 加载。


### 3. 不要依赖上传 `benchmark.py`

- ZIP 里若带有 `benchmark.py`，**会被忽略**。性能与格式以官方 `benchmark.py` 为准。

### 4. 可选文件：`requirements.txt`


### 5. 其他文件

- ZIP 中其余文件（在排除上述规则后）会合并到任务工作目录；请控制体积与必要性。

---

## 二、评测端软件环境



| 项目 | 示例值 |
|------|--------|
| 操作系统 | Linux（x86_64） |
| GPU | NVIDIA A800 80GB PCIe |
| NVIDIA 驱动 | 590.48.01 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128（CUDA 12.8 构建的 wheel） |
| PyTorch 随包 CUDA 运行时 | 12.8（与 `torch` wheel 绑定，用于推理） |
| CUDA Toolkit（`nvcc`） | 12.8（`nvcc` 报告 12.8.61）|

说明：

  - 若你开发时用了 **完整 CUDA Toolkit**，**建议在文档里写明 Toolkit 大版本与 PyTorch 的 CUDA 大版本一致**（示例均为 **12.8**），避免 ABI 不匹配。
  - **评测默认流程不会自动编译** ZIP 里的 `.cu` 源码；选手应在本地（与评测机器相同 CUDA 大版本的环境下） **编译好扩展**，在 ZIP 中提交 **预编译的 `.so`（及对应 Python 封装）**，或仅使用 **Triton / PyTorch 官方扩展的 pip 包**（有 wheel 时无需 `nvcc`）。

---

## 三、常见问题

### Q1：CUDA 大版本需要和评测机器一致吗？

需要保持一致。

- 选手编译自定义 `.so` 时，使用与评测机一致的 CUDA 大版本，降低 ABI / 运行时不兼容风险。
- 本评测机示例版本已写在上面的环境表中：  
  - PyTorch 侧 CUDA：`12.8`（`torch 2.8.0+cu128`）  
  - CUDA Toolkit（`nvcc`）：`12.8`（`V12.8.61`）


### Q2：新的评测逻辑是什么？评测一次大约多久？

- 当前评测逻辑固定从原数据集中选取150个子集。
- 评测一次大约耗时15-30分钟。

### Q3：排行榜评测的指标是什么？最终初赛成绩怎么计算？

- 排行榜的总分= 0.4 x $Ratio_{accuracy}$ + 0.3 x $TTFT_{improvements}$  + 0.3 x $Throughput_{improvement}$
- 最终初赛成绩计算与排行榜有差异。初赛每个团队最终成绩 = 0.4 × (准确率 / 最高准确率) + 0.3 × (TTFT提升率 /
最高TTFT提升率) + 0.3 × (吞吐量提升率 / 最高吞吐量提升率)
- 初赛前16名的团队晋级复赛。

## 四、选手提交前30秒自查表
- ZIP 内有 evaluation_wrapper.py
- VLMModel 能在本地同环境跑通
- 若用 .so：Linux x86_64 编译，CUDA 大版本与评测机一致
---

如有环境与提交规则更新，请以组委会 / 评测方最新文档为准。
