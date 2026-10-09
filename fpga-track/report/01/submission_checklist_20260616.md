# AICAS 2026 KV260 复赛提交清单

## 必交项映射

| 复赛要求 | 本包位置 | 状态 |
| --- | --- | --- |
| Bitstream / Overlay 文件包 | `hardware/sdf5_kv260/` | 已包含 `.bit`、`.bit.bin`、`.hwh`、`.xsa`、`.dtbo` |
| Python/C++ driver / host runtime | `runtime/mnn_server/`、`source/mnn_modified/`、`source/scripts/` | 已包含稳定 MNN server、PL runtime、加载脚本 |
| 完整 Linux 镜像 | `image/petalinux-sdimage_20260526_205333.wic.gz` | 已压缩准备 |
| 优化方法源代码 | `source/fpga/`、`source/mnn_modified/`、`source/aicas_semi/` | 已包含 |
| 复现视频 | `video/复现演示视频.mp4` | 已放入待提交包 |
| 吞吐量、能效、TTFT JSON | `results/` | 6/7 全量归档 + 6/16 PL8 复核/诊断归档 |
| 测试报告与技术文档 | `docs/技术报告.md`、`docs/测试报告.md`、本清单 | 已包含中文正文与中文文件名 |

## 最终硬件与 runtime

```text
overlay:
  hardware/sdf5_kv260/system.bit.bin
  hardware/sdf5_kv260/pl_sdf5_kvfit_kv260_uio_only.dtbo

runtime:
  runtime/mnn_server/bin/aicas_mnn_server

PL replace:
  shape = 1024x64x1024
  limit = 8
```

## 2026-06-16 板端复现验证

已验证命令：

```bash
sudo bash /home/ubuntu/aicas/mnn_cacheexp_20260601/scripts/kv260_load_sdf5_configfs_safe.sh \
  /home/ubuntu/aicas/stage_hw/sdf5/system.bit.bin \
  /home/ubuntu/aicas/stage_hw/sdf5/pl_sdf5_kvfit_kv260_uio_only.dtbo

sudo /home/ubuntu/aicas/mnn_cacheexp_20260601/bin/sdf5_kvfit_batch_test \
  --quick --mnn-only --case mnn_block_1024x64x1024
```

PL smoke 输出：

```text
SDF5_KVFIT_BATCH_TEST PASS
```

视频演示脚本输出会包含：

```text
matmul_replace_ok_count=8
pl_linear_packed_ok_count=8
replace_shapes=['1024x64x1024']
```

## 2026-06-16 非 accuracy 指标复核

推荐用 systemd 托管，防止 SSH 断开导致长任务被终止：

```bash
chmod +x /home/ubuntu/aicas_data/final_submit_metrics_20260616/kv260_start_final_pl8_retest_systemd.sh
AICAS_FINAL_RETEST_ID=final_pl8_retest_20260616_systemd \
AICAS_FINAL_RETEST_UNIT=aicas-final-pl8-retest \
bash /home/ubuntu/aicas_data/final_submit_metrics_20260616/kv260_start_final_pl8_retest_systemd.sh
```

该脚本只复跑最终提交主 case：

```text
prefix_delta_pl8
shape = 1024x64x1024
limit = 8
metrics = throughput, energy, ttft
```

输出目录：

```text
/home/ubuntu/aicas_data/final_submit_metrics_20260616/prefix_delta_pl8_only_final_pl8_retest_20260616_systemd
```

实际复核结论：

```text
tok16_partial_success:
  throughput_metrics.json
  energy_metrics.json
  pl_matmul.jsonl, replace_ok = 8

tok181_abort_diagnostics:
  完整 181-token 重跑在 throughput 后段触发 server SIGABRT
  PL trace 已记录 replace_ok = 8
```

正式提交主 JSON 仍使用 2026-06-07 全量稳定结果：

```text
results/aicas_submission_prefix_delta_pl8_retest_20260607.json
```

