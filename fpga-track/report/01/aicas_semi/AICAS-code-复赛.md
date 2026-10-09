# AICAS 2026 Grand Challenge Hardware - 复赛

## 概述

复赛延续初赛的目标——在KV260平台的硬件条件下部署和运行SmolVLM2-500M-Video-Instruct，利用片上的CPU以及FPGA资源，开展多模态视觉语言模型（VLM）的端侧推理芯片架构设计优化。

**复赛重点考核PL端硬件加速能力**：本轮起，参赛队伍必须利用KV260的可编程逻辑（PL）资源对模型推理过程进行硬件加速；仅通过llama.cpp软件参数调优、CPU多线程绑定或编译开关获得的吞吐量提升将不再被认可。

## 复赛核心调整

相较于初赛，复赛在以下方面进行了调整：

1. **强制PL端硬件参与**：所有参赛队伍必须提交可在KV260上加载的bitstream/Overlay；组委会在统一板卡上加载，加载失败的方案不进入评分。
2. **新增能效比指标 (Tokens/Joule)**：抑制单纯依靠堆叠CPU线程数、放宽功耗预算来刷高吞吐量的优化路径。
3. **新增首Token时延指标 (TTFT)**：避免为追求吞吐量而牺牲用户首响体验的batch化策略。
4. **统一基线 + 组委会复测**：原始基线 $T_{\text{ori}}$、$E_{\text{ori}}$、$\bar{a}_{\text{ori}}$、$\bar{b}_{\text{ori}}$ 由组委会在标准KV260板卡上统一测得并公布，最终成绩以组委会侧的复测结果为准，选手自报JSON仅作初评参考。
5. **精度测试样本数缩减**：考虑到 KV260 端侧推理速度，复赛精度测试样本数由初赛的 100 缩减为 30（分层随机抽样，3+3+3+3+3+3+12）。


## 软件环境

复赛阶段每支参赛队伍获分发一块独立的 KV260 开发板。组委会提供一份压缩包作为基础工具，包含：

- 复赛各项评测脚本（`sample.py`、`acc_eval.py`、`throughput_eval.py`、`energy_eval.py`、`ttft_eval_multiprompt.py`、`merge_results.py`）

参赛队伍可基于此工具包自行编译 KV260 Linux 镜像、配置运行时环境、替换或重新编译 llama.cpp（包括嵌入 XRT 调用以连接自研 PL 加速逻辑）、烧入自研 bitstream 等。组委会不限制参赛队伍的具体软件栈选择。

模型权重 SmolVLM2-500M-Video-Instruct GGUF 与 OCRBench 数据集请参赛队伍从公开渠道自行下载。

提交时本队伍交付的镜像 + bitstream + 源代码 + 启动脚本需保证组委会侧标准 KV260 板卡上可完整复现测试流程。


## 代码介绍

### sample.py

​	提供了采样代码，对**57817**个原始测试数据进行**分层随机**采样，得到 30 个测试数据。

​	考虑到 KV260 平台的端侧推理速度，复赛精度测试样本数由初赛的 100 缩减为 30，以控制单次评测耗时。复赛阶段的子任务抽样配比为：

| 测试名称                      | 采样个数 |
| ----------------------------- | -------- |
| Regular Text Recognition      | 3        |
| Irregular Text Recognition    | 3        |
| Artistic Text Recognition     | 3        |
| Handwriting Recognition       | 3        |
| Digit String Recognition      | 3        |
| Non-Semantic Text Recognition | 3        |
| Scene Text-centric VQA        | 12       |

​	请注意，复赛抽样总数与子类配比与初赛不同，n=30 用于所有参赛队伍。

### acc_eval.py

​	对采样得到的数据集进行推理，并得到准确率和原始回答结果。

### throughput_eval.py

​	使用较长的prompt和较大的图片进行推理，测试得到模型prefill与decoding两个阶段的吞吐量。

### energy_eval.py（复赛新增）

​	通过KV260板载PMBus / sysfs hwmon接口在推理期间高频采样板卡瞬时功耗，结合输出token计数计算端到端能效比（Tokens/Joule），输出JSON包含采样轨迹、平均功耗、总能耗与生成token数。

### ttft_eval_multiprompt.py（复赛新增）

​	通过llama-server的流式输出接口测量从请求送达到接收第一个token的端到端时延（Time-To-First-Token）。脚本读取 `ttft_config.json` 配置的多张图片（尺寸不同）与多种 prompt 长度（短/长/超长），对每个 (image, prompt) 组合各跑一次推理；每次请求在 prompt 首部注入随机 nonce 强制 LCP cache miss。输出 JSON 包含全部 case 的原始 TTFT 与按 image 分组对 prompt 长度做最小二乘线性拟合后的斜率与截距。

### merge_results.py（复赛新增）

​	将 `acc_eval.py`、`throughput_eval.py`、`energy_eval.py`、`ttft_eval_multiprompt.py` 四个评测脚本的输出文件合并为一份提交用的总 JSON，便于打包上传与组委会复测。


## 测试流程

本测试流程在初赛流程基础上扩展，**复赛提交方案必须使用KV260平台的FPGA资源**，仅基于CPU的方案不参与排名。

### 环境准备

参赛队伍可自行准备 python 运行环境。


### Bitstream 加载

按参赛队伍提交镜像中预设的方式加载 bitstream / overlay（PYNQ Overlay、`fpgautil`、自研 driver 均可），加载成功后方可继续后续测试。组委会复测时按本队伍提交的启动脚本执行，加载失败将直接判定为未通过硬件门槛。

### 数据集采样

```bash
python sample.py -i <FULL_TEST_JSON_PATH> -o <SAMPLED_JSON_PATH> 
```

### 准确率测试

在 shell 中启动 llama-server（或参赛队伍自研的等价推理服务）：

```bash
./llama-server -m <GGUF_PATH>/SmolVLM2-500M-Video-Instruct-f16.gguf --mmproj <GGUF_PATH>/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf
```

打开另一个 shell，运行准确率测试：

```bash
python acc_eval.py -i <IMAGE_FOLDER_PATH> -d <SAMPLED_JSON_PATH> -o <ACC_OUTPUT_JSON>
```

其中 `<IMAGE_FOLDER_PATH>` 为参赛队伍本地存放 OCRBench 数据集的目录，`<SAMPLED_JSON_PATH>` 为 `sample.py` 输出的采样数据集 JSON 路径，`<ACC_OUTPUT_JSON>` 为本次准确率测试结果输出 JSON 路径（例如 `acc_eval_results.json`）。

### 吞吐量测试

吞吐量测试所需输入的图片包含在组委会提供的压缩包中。

不要关闭 llama-server，直接运行吞吐量测试：

```bash
python throughput_eval.py -i <THROUGHPUT_IMAGE_PATH> -o <OUTPUT_JSON_PATH> 
```

其中 `<THROUGHPUT_IMAGE_PATH>` 为压缩包中提供的 `image.png` 文件。

### 能效比测试（复赛新增）

不要关闭llama-server，运行能效采样脚本：

```bash
python energy_eval.py -i <THROUGHPUT_IMAGE_PATH> -o <ENERGY_OUTPUT_JSON> --sample_hz 100
```

脚本将以100Hz采样PMBus读数并在采样期间触发推理，输出JSON包含累计能耗与生成token计数，能效比 $E$（Tokens/Joule）由脚本自动计算并写入。

### 首Token时延测试（复赛新增）

不要关闭 llama-server，按 `ttft_config.json` 中配置的多 image × 多 prompt 组合运行 TTFT 测试：

```bash
python ttft_eval_multiprompt.py -c ttft_config.json -o <TTFT_OUTPUT_JSON>
```

脚本对每个 case 在 prompt 首部注入 `[Nonce: timestamp]` 强制 cache miss，每个 case 跑一次。完成后按 image 分组对 prompt 长度做最小二乘线性拟合，输出 JSON 包含每个 case 的原始 TTFT、每张 image 的拟合斜率 (`slope_ms_per_char`) 与截距 (`intercept_ms`)，以及跨 image 平均后的聚合斜率与截距。

### 结果合并

所有评测项跑完后，合并为一份提交 JSON：

```bash
python merge_results.py \
    --acc        <ACC_OUTPUT_JSON> \
    --throughput <THROUGHPUT_OUTPUT_JSON> \
    --energy     <ENERGY_OUTPUT_JSON> \
    --ttft       <TTFT_OUTPUT_JSON> \
    -o           aicas_submission.json
```

之后请统一命名输出文件为天池平台规定的格式并上传。

