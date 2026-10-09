# 提交说明

本提交没有对 Qwen3-VL 模型权重做微调。

`layer_stats/` 目录下的文件是离线校准产物。具体来说，decode 阶段的
layer-skip bridge 使用校准样本采集到的 hidden-state pair，把第 6 层输出
映射到第 14 层输出附近，然后再进入第 15 层。

这里的 low-rank payload 不是 LoRA adapter，也不是通过梯度下降训练得到的。
它是根据校准 hidden-state pair 解析计算出来的：

- `decode_skip_6_14_ls_affine.pt`：从 source hidden state 到 target hidden
  state 的对角 affine 校准参数。
- `decode_skip_6_14_ls_residual_lowrank_k128.pt`：rank-128 residual bridge，
  用校准 hidden-state residual 拟合得到。

运行时，这些文件只作为跳过 decode layers 7-14 后的确定性校准桥使用。
原始 checkpoint 权重不变。

`aicas2026gc/runtime_autotune.json` 只是 kernel autotune 缓存，记录历史 shape
下哪个算子实现最快；它不包含训练出来的模型参数。

# Submission Notes

This submission does not fine-tune the Qwen3-VL model weights.

The files under `layer_stats/` are offline calibration artifacts. In particular,
the decode layer-skip bridge uses hidden-state pairs collected from calibration
samples to map the layer 6 output close to the layer 14 output before entering
layer 15.

The low-rank payload is not a LoRA adapter and is not trained with gradient
descent. It is computed analytically from calibration hidden-state pairs:

- `decode_skip_6_14_ls_affine.pt`: diagonal affine calibration from source
  hidden-state statistics to target hidden-state statistics.
- `decode_skip_6_14_ls_residual_lowrank_k128.pt`: a rank-128 residual bridge
  fitted from calibration hidden-state residuals.

At runtime, these files are used only as a deterministic calibration bridge
after skipping decode layers 7-14. The original checkpoint weights are unchanged.

`aicas2026gc/runtime_autotune.json` is only a kernel autotune cache. It records
which implementation was fastest for previously seen shapes and does not contain
trained model parameters.
