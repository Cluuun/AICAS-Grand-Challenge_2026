# KV260 复赛提交材料

队伍：芯溯猫猫队

本目录按组委会复赛提交要求整理，只保留提交和复现需要的材料。完整校验清单、打包过程文件和临时编译日志未放入本提交目录。

## 目录说明

```text
01_Bitstream与Overlay文件包/
02_完整Linux镜像/
03_优化方法源代码/
04_复现视频/
05_吞吐量_能效_TTFT测试JSON文件/
06_测试报告与技术文档/
07_模型文件/
```

## 1. Bitstream / Overlay 文件包

目录：`01_Bitstream与Overlay文件包/`

包含 KV260 运行所需的 bitstream、HWH、FPGA manager 加载工具、运行脚本、llama-server 运行时二进制和相关动态库。

主要文件：

- `tpu_soc_wrapper.bit`
- `tpu_soc_wrapper.bin`
- `tpu_soc_wrapper.bit.bin`
- `tpu_soc_wrapper.hwh`
- `FPGA加载工具/load_bitstream_fpga_manager`
- `板端运行脚本/`
- `运行时二进制/`

## 2. 完整 Linux 镜像

目录：`02_完整Linux镜像/`

包含可直接烧录到 SD 卡运行的 KV260 Linux 镜像压缩包：

- `KV260_完整Linux镜像_32GB_initramfs_20260612_1008.img.xz`

建议使用 32 GB 或更大的 SD 卡。烧录示例：

```bash
xzcat KV260_完整Linux镜像_32GB_initramfs_20260612_1008.img.xz | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

其中 `/dev/sdX` 需要替换为实际 SD 卡设备，执行前请确认设备名。

## 3. 优化方法的源代码

目录：`03_优化方法源代码/`

包含 HLS/RTL 工程、host 侧组件、构建脚本和相关说明。提交层级目录已按用途命名，源码树内部结构保持原样，方便复现构建和检查实现。

构建入口：

```bash
cd 03_优化方法源代码
JOBS=4 ./01_构建脚本/build_host_software_kv260.sh
```

## 4. 复现视频

目录：`04_复现视频/`

包含复现视频：

- `KV260_复现视频_镜像烧录启动测评_x265.mp4`

视频展示镜像烧录、板卡启动、进入运行目录、系统检查、bitstream 加载、server 启动、官方测评脚本运行和结果输出。

## 5. 吞吐量、能效、TTFT 测试 JSON 文件

目录：`05_吞吐量_能效_TTFT测试JSON文件/`

包含 KV260 板端测评输出和合并提交结果：

- `throughput_metrics.json`
- `energy_metrics.json`
- `ttft_eval_results.json`
- `ttft_metrics.json`
- `acc_eval_results.json`
- `aicas_submission.json`
- `score_summary.json`
- `README_JSON测试结果.md`

## 6. 测试报告与技术文档

目录：`06_测试报告与技术文档/`

包含测试报告和技术文档 PDF，按用途拆分为：

- `01_技术报告/KV260_技术报告.pdf`
- `01_技术报告/长上下文KVCache压缩方案.pdf`
- `01_技术报告/LoRA模型适配与长输出优化报告.pdf`
- `02_测试报告/KV260_复赛测试说明.pdf`

## 7. 模型文件

目录：`07_模型文件/`

包含可直接核对的模型文件：

- `SmolVLM2-256M-Video-Instruct-aicas-lora-strong-r64-2ep-f16.gguf`
- `mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf`

完整 Linux 镜像中已包含板端运行所需模型，本目录用于在提交包内直接查看模型文件。

## 运行入口

镜像启动后登录信息：

- 用户名：`xilinx`
- 密码：`xilinx`

交付包目录：

```bash
cd /home/xilinx/kv260_llama_server_minimal
```

建议先检查系统，再启动服务：

```bash
./scripts/check_system.sh
./scripts/start_server.sh
```

服务地址：

```text
http://127.0.0.1:8080/v1
```

数据集挂载路径请按实际测试环境配置。文档中的 NFS 地址仅作为参考示例。
