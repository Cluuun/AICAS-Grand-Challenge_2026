# AICAS 2026 KV260 测试报告

## 推荐提交候选

推荐提交结果文件：

```text
results/aicas_submission_prefix_delta_pl8_retest_20260607.json
```

运行模式：

```text
MNN prefix-delta cache + SDF5 PL real replace
PL 替换形状：1024x64x1024
PL 替换次数上限：8
```

## 指标结果

| 指标 | 实测值 |
|---|---:|
| Prefill 吞吐量 | 29.9060 token/s |
| Decode 吞吐量 | 7.0307 token/s |
| 能效比 | 1.5959 token/J |
| Energy 测试平均功耗 | 4.6455 W |
| Energy 测试推理时长 | 24.4142 s |
| 生成 token 数 | 181 |
| TTFT 斜率 | 6.5651 ms/char |
| TTFT 截距 | 17337.93 ms |
| 线上分数估计 | 47.3893 |

相对官方 baseline 的提升率：

| 指标 | 提升率 |
|---|---:|
| Prefill | 0.899351 |
| Decode | 0.210607 |
| Energy | 0.392190 |
| TTFT | 0.565916 |

## PL 参与情况

该结果使用真实 PL replace，不是 shadow-only profiling。JSON 中记录：

```text
replace_ok = 8
replace_shapes = ["1024x64x1024"]
```

这表示 MNN 推理过程中选定的 linear 任务被提交到 SDF5 PL sidecar，并且 runtime 使用了 PL 返回的输出。

## 功耗测量方法

功耗采样来源为 KV260 板载 INA260 sysfs 传感器：

```text
/sys/class/hwmon/hwmon0/name = ina260_u14
/sys/class/hwmon/hwmon0/power1_input
```

Energy 指标按照比赛定义，在推理窗口内积分板级功耗，并用生成 token 数除以总能量，得到 `tokens/J`。

## 启动与运行条件

测试镜像采用 headless KV260 PetaLinux 配置，禁用了未使用的 display、DPDMA、USB controller 和 CAN controller。开始推理前，板卡先进入稳定空闲状态。服务裁剪、网络速率和 CPU hotplug 实验表明，系统级静态功耗优化主要带来毫瓦级收益；最终分数主要由推理耗时、prefix-delta cache 命中、PL replace 开销和 MNN runtime 配置决定。

## 支撑日志

详细日志保存在：

```text
results/kv260_mnn_opt_full181_pl8_20260606/
results/kv260_mnn_decode_opt_candidates_20260607/
docs/20260601_kv260_MNN_prefix_delta叠加SDF5_PL_replace评测记录.md
docs/20260527_kv260低功耗镜像静态功耗复测.md
```

