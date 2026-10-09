# AICAS 2026 FPGA 赛道 KV260 最终提交包

本目录是 KV260 SmolVLM2 优化方案的复赛提交包，包含可加载 PL overlay、MNN 推理运行时、模型文件、修改源码、复现脚本、评测 JSON、低功耗 Linux 镜像压缩包、复现演示视频和技术文档。

## 提交口径

最终推荐口径：

```text
MNN prefix-delta cache + SDF5 PL real replace
PL 替换形状：1024x64x1024
不使用 XRT / ZOCL / xclbin
```

对应硬件加载方式：

```text
FPGA Manager + configfs device-tree overlay + generic-uio + CMA helper
```

计分主 runtime：

```text
/home/ubuntu/aicas/mnn_cacheexp_20260601/bin/aicas_mnn_server
```

注意：`source/aicas_semi/code/kv260_stable_pl8_visual_demo.sh` 是复现视频展示脚本，使用同一个稳定 runtime 和同一个 SDF5 PL8 replace 口径。

## 目录结构

```text
aicas2026_kv260_submission_20260616/
  README.md
  MANIFEST.json
  SHA256SUMS.txt
  video/
    复现演示视频.mp4
  hardware/sdf5_kv260/
    system.bit
    system.bit.bin
    system.hwh
    system.xsa
    pl_sdf5_kvfit_kv260_uio_only.dtbo
    reports/
  image/
    BOOT.BIN
    image.ub
    system.dtb
    system-zynqmp-sck-kv-g-revB.dtb
    petalinux-sdimage_20260526_205333.wic.gz
    petalinux-sdimage_20260526_205333.wic.gz.sha256
  runtime/
    mnn_server/
    model/
  source/
    fpga/
    mnn_modified/
    aicas_semi/
    scripts/
  results/
    aicas_submission_prefix_delta_pl8_retest_20260607.json
    final_retest_20260616/
  docs/
    技术报告.md
    测试报告.md
    technical_report.md
    test_report.md
    submission_checklist_20260616.md
    20260616_kv260复赛最终视频演示流程.md
    20260616_kv260板端空间整理与流式演示录制命令.md
```

## Linux 镜像

压缩镜像：

```text
image/petalinux-sdimage_20260526_205333.wic.gz
```

该文件解压后是可直接烧录到 SD 卡的 KV260 PetaLinux 镜像。烧录后默认用户：

```text
ubuntu
```

实测根文件系统：

```text
/dev/mmcblk1p2
```

额外数据分区：

```text
/home/ubuntu/aicas_data
```

## 快速复现流程

启动板卡后，登录：

```bash
ssh ubuntu@192.168.137.102
```

加载提交版 SDF5 overlay：

```bash
sudo bash /home/ubuntu/aicas/mnn_cacheexp_20260601/scripts/kv260_load_sdf5_configfs_safe.sh \
  /home/ubuntu/aicas/stage_hw/sdf5/system.bit.bin \
  /home/ubuntu/aicas/stage_hw/sdf5/pl_sdf5_kvfit_kv260_uio_only.dtbo
```

运行 PL smoke：

```bash
sudo /home/ubuntu/aicas/mnn_cacheexp_20260601/bin/sdf5_kvfit_batch_test \
  --quick --mnn-only --case mnn_block_1024x64x1024
```

复跑最终 PL8 提交口径时推荐使用 systemd 托管版，避免 SSH 断开后长任务被登录会话清理：

```bash
chmod +x /home/ubuntu/aicas_data/final_submit_metrics_20260616/kv260_start_final_pl8_retest_systemd.sh
AICAS_FINAL_RETEST_ID=final_pl8_retest_20260616_systemd \
AICAS_FINAL_RETEST_UNIT=aicas-final-pl8-retest \
bash /home/ubuntu/aicas_data/final_submit_metrics_20260616/kv260_start_final_pl8_retest_systemd.sh
```

如需在当前终端前台运行，可直接执行：

```bash
bash /home/ubuntu/aicas_data/final_submit_metrics_20260616/kv260_run_final_pl8_retest_only.sh
```

结果目录：

```text
/home/ubuntu/aicas_data/final_submit_metrics_20260616/prefix_delta_pl8_only_final_pl8_retest_20260616_systemd
```

2026-06-16 实际复核中，181-token 完整重跑在 throughput 后段触发 server SIGABRT；tok16 短复核完成 throughput/energy 并记录真实 PL replace，TTFT stream=True 子项未返回首个 chunk。完整计分 JSON 仍以 2026-06-07 全量结果为准；6/16 结果作为复核与诊断归档。

录制视频推荐演示命令：

```bash
AICAS_VISUAL_MAX_TOKENS=64 \
AICAS_VISUAL_TYPEWRITER_DELAY=0.015 \
bash /home/ubuntu/aicas_data/stream_demo_20260616/scripts/kv260_stable_pl8_visual_demo.sh
```

该命令会自动加载 SDF5 overlay、执行提交 shape smoke、启动稳定 MNN server、输出 VLM 文本，并校验：

```text
replace_shapes=['1024x64x1024']
```

## 推荐 JSON

历史稳定推荐：

```text
results/aicas_submission_prefix_delta_pl8_retest_20260607.json
```

2026-06-16 复跑与诊断归档：

```text
results/final_retest_20260616/
```

其中：

```text
tok16_partial_success/      throughput + energy + PL trace
tok181_abort_diagnostics/   181-token 完整重跑 abort 诊断日志
```

## 注意事项

- 本方案不提交 `.xclbin`，因为 PL 侧采用 direct-runtime，而不是 XRT/ZOCL。
- 复现视频已放入 `video/复现演示视频.mp4`，脚本、日志和文档也一并归档。
- `kv260_stable_pl8_visual_demo.sh` 使用最终稳定 runtime；不要使用实验性 `aicas_mnn_server.stream_demo` 作为计分主线。
- accuracy 未在 2026-06-16 复跑；6/16 复核完成 overlay、PL smoke、tok16 throughput/energy 和 PL trace，完整 181-token throughput/energy/TTFT 采用 2026-06-07 已归档 JSON。
