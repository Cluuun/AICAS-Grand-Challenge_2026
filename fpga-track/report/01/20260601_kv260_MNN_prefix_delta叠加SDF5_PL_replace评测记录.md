# 2026-06-01 KV260 MNN prefix_delta 叠加 SDF5 PL replace 评测记录

## 目标

在已跑通的 MNN `prefix_delta` warm-cache 能耗优化基础上，重新引入真实 PL replace，而不是 shadow，生成可用于 aicas_semi 口径的吞吐、能效、TTFT JSON。

本轮不跑 accuracy，只跑：

- throughput
- energy
- TTFT
- merged `aicas_submission.json`
- local `score.json`

## 板端路径

KV260：

```text
ubuntu@192.168.137.102
```

MNN runtime：

```text
/home/ubuntu/aicas/mnn_cacheexp_20260601/
```

本轮结果目录：

```text
/home/ubuntu/aicas/power_diag/mnn_cacheexp_20260601/prefix_delta_pl8_submission_20221112_120053
```

本地归档：

```text
output/kv260_mnn_prefix_delta_pl8_submission_20260601/
```

可直接查看的最终 JSON：

```text
output/kv260_mnn_prefix_delta_pl8_submission_20260601/aicas_submission_prefix_delta_pl8.json
output/kv260_mnn_prefix_delta_pl8_submission_20260601/score_prefix_delta_pl8.json
output/kv260_mnn_prefix_delta_pl8_submission_20260601/final_comparison.json
```

## SDF5 overlay 修复

当前低功耗 SD/rootfs 中，base DT 已经预置：

```text
smolvlm-llama-fpga-cma
```

原始 SDF5 dtbo 同时声明 SDF5 UIO 节点和 CMA helper，导致 configfs overlay 可能出现：

```text
create_overlay: Failed to create overlay (err=-22)
```

本轮新增 UIO-only overlay：

```text
fpga/sdf5_mnn_prefill_accel/board/kv260_dt/pl_sdf5_kvfit_kv260_uio_only.dtsi
output/sdf5_kv260_board_pkg_20260524/pl_sdf5_kvfit_kv260_uio_only.dtbo
```

加载方式：

```text
FPGA Manager + configfs overlay
```

加载结果：

```text
Overlay status: applied
/sys/class/uio/uio4 name=sdf4-4-linear addr=0x0000000096000000 size=0x1000
```

注意：SDF5 KV260 alias ABI 的可用 magic 在 lane0：

```text
0x96000000 = 0x53444635  # "SDF5"
```

legacy debug window：

```text
0x96000154 = 0
```

因此不能只用 `0x154` 判断 SDF5 是否可用。

## 独立 C 程序验证

运行：

```text
sudo /home/ubuntu/aicas/mnn_cacheexp_20260601/bin/sdf5_kvfit_batch_test --quick --mnn-only
```

关键 PASS：

| Case | Result | GMAC/s |
|---|---:|---:|
| 80x960x320 | PASS | 5.57 |
| 1024x64x1024 | PASS | 2.38 |
| 1024x1024x64 | PASS | 6.42 |
| 80x960x960 | PASS | 5.55 |
| 80x960x2560 | PASS | 5.55 |
| 80x2560x960 | PASS | 5.89 |
| 490x960x320 | PASS | 6.03 |
| 490x960x960 | PASS | 6.03 |
| 490x960x2560 | PASS | 6.03 |
| 490x2560x960 | PASS | 6.43 |
| 1024x768x768 | PASS | 6.30 |
| 1024x3072x768 | PASS | 6.68 |
| 1024x768x3072 | PASS | 6.30 |

失败：

```text
64x12288x960 FAIL
```

原因：该形状超出当前 SDF5 replace allowlist/硬件可用范围。本轮 MNN real replace 只允许：

```text
AICAS_PL_MATMUL_SHAPES=1024x64x1024
AICAS_PL_MATMUL_REPLACE_LIMIT=8
```

## MNN 评测配置

三组：

```text
baseline
prefix_delta
prefix_delta_pl8
```

`prefix_delta_pl8` 关键环境：

```text
AICAS_MNN_STABLE_IMAGE_CACHE=1
AICAS_MNN_PREFIX_CACHE_NAME=aicas_official_sequence_prefix
AICAS_MNN_PREFIX_CACHE_REARM=1
AICAS_MNN_PREFIX_DELTA_DECODE=1
AICAS_MNN_PREFIX_CACHE_HASHED=1
AICAS_PL_REG_BASE=0x96000000
AICAS_PL_CMA_DEV=/dev/smolvlm_llama_fpga_cma
AICAS_PL_MATMUL_REPLACE=1
AICAS_PL_MATMUL_REPLACE_LIMIT=8
AICAS_PL_MATMUL_PHASE_FILTER=all
AICAS_PL_MATMUL_SHAPES=1024x64x1024
AICAS_PL_MATMUL_VERIFY=none
```

## 结果

| Case | Prefill t/s | Decode t/s | Avg Power W | Energy J | Tokens/J | TTFT slope | TTFT intercept | Online score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 30.0877 | 8.5134 | 4.4489 | 235.4053 | 0.7689 | 7.1216 | 15383.3 | 31.8115 |
| prefix_delta | 29.8672 | 5.9509 | 4.6086 | 116.8992 | 1.5483 | 6.9403 | 16352.9 | 42.3994 |
| prefix_delta_pl8 | 29.9539 | 6.1963 | 4.6851 | 112.2255 | 1.6128 | 6.8688 | 16348.9 | 44.3249 |

相对 `prefix_delta`：

```text
Decode:     +4.12%
Energy:     116.90 J -> 112.23 J, -4.00%
Tokens/J:   1.5483 -> 1.6128, +4.17%
Score:      42.3994 -> 44.3249, +1.9255
```

相对 `baseline`：

```text
Energy:     235.41 J -> 112.23 J, -52.33%
Tokens/J:   0.7689 -> 1.6128, +109.76%
Score:      31.8115 -> 44.3249, +12.5134
```

## PL trace

`prefix_delta_pl8` trace：

```text
matrix/prefix_delta_pl8/pl_matmul.jsonl
```

统计：

```text
matmul_replace_ok_count = 8
pl_linear_packed_ok_count = 8
replace_shapes = ["1024x64x1024"]
packed_duration_us_sum = 1,171,997 us
packed_wait_us_sum = 233,171 us
```

结论：本轮不是 PL shadow。MNN 确实将 8 个 `1024x64x1024` MatMul 交给 SDF5 PL 计算，并使用 PL 输出。

## 判断

这次 PL replace 有小但明确的正收益：

- PL 静态/动态功耗使平均功耗从 `4.61W` 升到 `4.69W`。
- 但推理时间从 `25.37s` 降到 `23.95s`。
- 因此总能量和 tokens/J 都改善。

当前 PL 吞吐仍低，单个 `1024x64x1024` wait 约 `28-30ms`，约 `2.26-2.35 GMAC/s`。这不是最终理想加速器，但已经满足“真实 PL replace 接入 MNN 并改善 score”的闭环证据。

下一步若继续优化，优先方向仍是：

1. 扩大可替换形状，但避开 `K=12288` 等当前硬件不支持形状。
2. 降低 pack/dequant/submit 包装开销。
3. 提升 PL raw throughput 和 AXI 并发。
4. 保持 `prefix_delta` 的 warm-cache 能效收益，同时只在能缩短 wall time 的 op 上启用 PL replace。
