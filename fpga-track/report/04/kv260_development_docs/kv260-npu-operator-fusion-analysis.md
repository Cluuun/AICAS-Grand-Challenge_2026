# KV260 NPU 算子融合方案分析

本文记录 2026-05-19 semi throughput profiling 后得到的 NPU 算子融合方向。重点不是单个算子
是否能上 NPU，而是哪些融合能避免大块 F32 中间结果在 CPU DRAM/CMA 之间来回搬运。

## 1. 结论和优先级

当前最高价值原则：

```text
不要把大中间张量完整 mvout 到 CPU 内存，再由 CPU 做 elementwise/softmax/pack 后重新送回 NPU。
```

推荐优先级：

| 优先级 | 方向 | 结论 |
| --- | --- | --- |
| P0 | 保持当前 FP32 mvout + NEON activation pack | 已经把 NPU profile 总耗时从约 20.2 s 降到 17.6 s，应保留 |
| P1 | mmproj fused vision attention | mmproj 最大未硬件化热点，CPU QK/PV 约 11.84 s，必须融合 softmax，不能只拆成孤立 GEMM |
| P2 | text attention fused QK-softmax-PV | 理论计算收益大，但只有避免 QK 分数矩阵回 CPU DRAM 才划算 |
| P3 | text MLP tile fusion: gate/up + SwiGLU + down | 有必要做，但按当前安排暂缓；需要记录为后续硬件方向 |
| P4 | mmproj MLP fusion: ffn_up + GELU + ffn_down | 小于 attention，但能减少 ffn_up F32 写回和 ffn_down 重新 pack |
| P5 | output residual fusion: mvout/dequant + residual add | 实现相对简单，收益主要是减少一遍 CPU elementwise 和内存 pass |
| P6 | standalone activation quant/pack hardware | 不建议单独做；如果先把 F32 搬进 CMA，流量会变成 int8 path 的约 4 倍 |

## 2. 实测依据

主要 profile run：

```text
20260519T-profile-mmproj-text-decode1-after-fp32neon
```

attention 拆 QK/PV 实验：

```text
20260519T-profile-text-attn-qk-pv-f16kv
```

完整 no-profile semi throughput sanity run：

```text
20260519T-semi-throughput-fp32neon-noprofile
```

profile-only run 使用 `max_tokens=1`，输出 token 是 `The`。

优化后 no-profile semi throughput：

```text
prompt tokens:     501
completion tokens: 408
prefill:           12.10 tok/s
decode:             5.13 tok/s
energy:             0.9175 tokens/J
TTFT slope:         8.9109 ms/char
TTFT intercept: 33817.31 ms
```

优化后 decode-token=1 profile 总览：

```text
prefill total:      54170.58 ms
mmproj encode:      20288.17 ms
merged text:        33868.54 ms

NPU total node:     17643.29 ms
activation pack:     1590.44 ms
DMA in pair:         3873.43 ms
DMA in bias:         1053.37 ms
GEMM:                3309.15 ms
DMA out:             2144.07 ms
postprocess:         4941.36 ms
```

## 3. 当前主要瓶颈

### 3.1 postprocess 不是量化后处理，主要是 CMA 到 ggml tensor copy

当前使用 FP32 mvout，NPU 已经做了 scale，postprocess 高主要是把 NPU 输出从 CMA mmap
内存复制到普通 ggml tensor 内存。

```text
output bytes total: 881.87 MB
postprocess:        4941.36 ms
effective copy:     about 178 MB/s

mmproj output bytes: 339.98 MB, postprocess 1916.35 ms
text output bytes:   541.88 MB, postprocess 3025.01 ms
```

所以单纯优化 CPU 后处理循环空间有限。更有效的是减少中间层 F32 mvout，或者让下一个 NPU
算子直接消费上一个 NPU 算子的输出。

### 3.2 activation pack 已优化，但不适合单独搬到硬件

NEON pack 后：

```text
activation pack total: 1590.44 ms
packed activation bytes: 216.86 MB
```

之前软件 pack 约 `4143.76 ms`，现在已经明显下降。若单独做硬件 pack，数据流可能变成：

```text
当前 host path:
  CPU read F32/F16 source
  CPU write int8 packed activation to CMA
  NPU DMA read int8 activation

naive hardware pack path:
  CPU/DMA copy F32/F16 source to CMA
  NPU read F32/F16 from CMA
  NPU write int8 activation to CMA/SPM
  NPU GEMM read int8 activation
```

F32 source 本身就是 int8 payload 的约 4 倍，再加一次 CMA 写读，可能比软件 pack 更差。
因此 activation quant/pack 硬件化应该和 producer 或 consumer 融合：

```text
RMS_NORM -> quant/pack -> Q/K/V or FFN input
SwiGLU/GELU -> quant/pack -> down projection
attention output -> quant/pack -> attn_out
```

如果实现，pack 输出应直接进入 SPM 或下一个 GEMM input buffer，不应再落一个完整 DRAM tensor。

## 4. Text 侧融合方案

### 4.1 相邻 NPU 算子模式

text 当前基本是 NPU matmul 和 CPU elementwise/layout/attention 交替：

```text
RMS_NORM(CPU)
  -> Q/K/V MUL_MAT(NPU)
  -> ROPE(CPU) + SET_ROWS(CPU)
  -> FLASH_ATTN_EXT(CPU)
  -> attn_out MUL_MAT(NPU)
  -> residual ADD(CPU)

RMS_NORM(CPU)
  -> ffn_gate MUL_MAT(NPU)
  -> ffn_up MUL_MAT(NPU)
  -> SWIGLU(CPU)
  -> ffn_down MUL_MAT(NPU)
  -> residual ADD(CPU)
```

这说明最适合和 NPU `MUL_MAT` 融合的 elementwise 不是孤立小算子，而是能阻断大中间张量
回 CPU 的算子：`SwiGLU/GELU`、`softmax`、`residual add`、`RMS_NORM+quant/pack`。

### 4.2 Text MLP tile fusion: gate/up + SwiGLU + down

这个方向有必要做，但暂缓实现，先作为后续硬件方案记录。

融合目标：

```text
ffn_gate MUL_MAT
ffn_up   MUL_MAT
SwiGLU
ffn_down MUL_MAT
```

主要收益不只是去掉 CPU `SWIGLU` 本身。当前 `SWIGLU` 约 `220 ms`，但更大的问题是
`ffn_gate` 和 `ffn_up` 都会写出大 F32 中间张量，之后 `ffn_down` 又要把 SwiGLU 结果
重新 quant/pack。tile fusion 可以避免：

```text
gate/up full F32 mvout
CPU SwiGLU over large tensor
SwiGLU tensor DRAM write/read
ffn_down activation quant/pack
```

建议硬件 contract：

```text
gate_acc_tile, up_acc_tile
  -> dequant or approximate dequant
  -> silu(gate) * up
  -> quantize to int8 activation tile
  -> feed ffn_down GEMM
  -> optional residual add on final mvout
```

### 4.3 Text attention: 不建议只把 flashattention 拆成孤立 QK/PV GEMM

拆 attention 实验配置：

```text
--flash-attn off
--cache-type-k f16
--cache-type-v f16
```

测得 text attention：

```text
q8 cache + FLASH_ATTN_EXT:
  FLASH_ATTN_EXT: 7505.00 ms

f16 cache + split attention:
  QK MUL_MAT: 3571.94 ms
  SOFT_MAX:    920.03 ms
  PV MUL_MAT: 2282.32 ms
  total:      6774.29 ms
```

拆开后更快不是因为 split attention 本质更优，而是同时改变了 KV cache type：

```text
q8 KV cache 在当前 llama.cpp 配置里会走 flash attention；
q8 flash path 有 dequant 和 fused kernel 开销；
f16 split path 使用普通 CPU matmul kernel，对当前 shape 更有效；
f16 KV 让 SET_ROWS 从约 13.5 ms 增到约 209 ms，但 attention compute 降得更多。
```

text split attention shape：

```text
QK:
  cache_k shape [64, 512, 5, 1]
  Q shape       [64, 501, 15, 1]
  output kq     [512, 501, 15, 1]

PV:
  cache_v shape [512, 64, 5, 1]
  softmax shape [512, 501, 15, 1]
  output kqv    [64, 501, 15, 1]
```

按当前 text projection 的 NPU 吞吐估计：

```text
projection MAC total: 153.91 GMAC
GEMM-only time:       2405.66 ms -> 63.98 GMAC/s
end-to-end time:     12168.81 ms -> 12.65 GMAC/s

QK: about 7.88 GMAC across 32 layers
PV: about 7.88 GMAC across 32 layers
```

理想 GEMM-only 估计：

```text
QK on NPU:    about 123 ms
PV on NPU:    about 123 ms
QK + PV:      about 246 ms
```

用当前 projection end-to-end 效率保守估计：

```text
QK:           about 623 ms
PV:           about 623 ms
QK + PV:      about 1246 ms
plus softmax: about 920 ms if left on CPU
total:        about 2166 ms before extra QK/softmax transfers
```

但是 QK 输出很大：

```text
512 * 501 * 15 * 32 layers * fp32 ~= 492.5 MB
```

如果 QK 写到 DRAM/CMA，再给 CPU softmax，再 pack/搬回 NPU 做 PV，搬运会吃掉很多理论收益。
所以 attention 硬件化需要一个 fused attention block。

最低硬件需求：

```text
QK int8 GEMM
attention scale: 1 / sqrt(head_dim)
mask support
row-wise softmax: max/sub/exp/sum/reciprocal
PV GEMM consuming softmax tiles directly
head/layout merge
optional output quant/pack to feed attn_out
```

如果目标是 attention 期间不回 CPU DRAM，除 softmax 外还需要支持 Q/K layout、mask、
PV、head merge、可选 ROPE/KV cache layout 更新。ROPE 当前约 `91 ms`，不是第一瓶颈，但要做
完整 no-CPU-roundtrip attention 时需要纳入数据路径。

## 5. mmproj 融合方案

### 5.1 mmproj 当前结构和热点

mmproj 是 12 层 vision transformer 加最终 projector，不只是线性层。

```text
mmproj total encode: 20288.17 ms
CPU scheduled nodes: 201
NPU scheduled nodes: 217
```

mmproj CPU hotspots：

```text
MUL_MAT CPU: 11843.85 ms
SOFT_MAX:      920.06 ms
GELU:          340.63 ms
CONT/layout:   188.70 ms
ADD:            77.26 ms
NORM:           75.87 ms
IM2COL:          7.88 ms
```

NPU-offloaded linears 已经大多是 `MUL_MAT + ADD`：

```text
attn_q / attn_k / attn_v: 12 each
attn_out:                 12, recorded as unclassified MUL_MAT+ADD
ffn_down:                 12, M=3072, N=1024, K=768
ffn_up:                   12, M=768,  N=1024, K=3072
final mm.model.fc:         1, M=960,  N=64,   K=12288
```

mmproj NPU group timing：

```text
ffn_down:      1941.07 ms
ffn_up:        1312.21 ms
attn_out/etc:   600.87 ms
attn_q:         540.68 ms
attn_v:         540.08 ms
attn_k:         539.57 ms
```

### 5.2 P1: fused vision attention

最佳 mmproj 融合目标：

```text
Q linear / K linear / V linear
  -> reshape / permute / head split
  -> QK
  -> scale + mask + softmax
  -> PV
  -> head merge / layout
  -> optional attn_out linear
  -> residual add
```

CPU attention matmul signature：

```text
QK:
  src0 shape [64, 1024, 12, 1]
  src1 shape [64, 1024, 12, 1]
  dst  shape [1024, 1024, 12, 1]

PV:
  src0 shape [1024, 64, 12, 1]
  src1 shape [1024, 1024, 12, 1]
  dst  shape [64, 1024, 12, 1]
```

QK score tensor：

```text
1024 * 1024 * 12 * fp32 ~= 50.3 MB per layer
12 layers ~= 604 MB
```

因此 mmproj attention 不能只做 isolated QK/PV offload。真正有价值的是 streaming attention：
score tile 留在片上完成 softmax，并直接进入 PV。

需要硬件能力：

```text
QK GEMM over [heads, query, key]
attention scale
optional mask
row-wise softmax: max/sub/exp/sum/reciprocal
PV GEMM consuming softmax tiles without CPU round-trip
layout/head merge for attn_out
optional direct feed into attn_out linear
```

预期收益：

```text
消除或大幅降低 mmproj CPU attention matmul: ~11.84 s
消除或降低 CPU softmax: ~0.92 s
减少 Q/K/V projection output 和 score tensor 的 DRAM traffic
可能降低 Q/K/V NPU postprocess，因为 projection 可直接喂 attention
```

### 5.3 P4: mmproj MLP fusion

mmproj FFN pattern：

```text
NORM(CPU)
  -> ffn_up MUL_MAT+ADD(NPU)
  -> GELU(CPU)
  -> ffn_down MUL_MAT+ADD(NPU)
  -> residual ADD(CPU)
```

当前相关耗时：

```text
ffn_up NPU total:   1312.21 ms
ffn_down NPU total: 1941.07 ms
GELU CPU:            340.63 ms
```

融合目标：

```text
ffn_up accumulator / F32 tile
  -> GELU
  -> quantize / pack as int8 activation
  -> ffn_down GEMM
  -> residual add on mvout
```

mmproj 使用 GELU，不是 text 的 SwiGLU，所以硬件比 text MLP fusion 简单。收益仍然不是
单独省 `~341 ms` GELU，而是避免 `ffn_up` 完整 F32 writeback 和 `ffn_down` 重新 pack。

### 5.4 mmproj 低优先级项

projection + residual output fusion：

```text
NPU mvout/dequant output + residual input -> final output
```

这能减少 CPU `ADD` 和一次内存 pass，但比 attention/MLP fusion 小。

norm + quant/pack：

```text
NORM -> quant/pack -> Q/K/V or FFN input
```

`NORM` 只有约 `75.87 ms`，不建议单独硬化。只有和 quant/pack 融合才有意义。

patch embedding：

```text
node_3 MUL_MAT: about 149 ms
shape: [768, 1024] x [768, 768]
```

这只是一个小节点，优先级低于 attention 和 FFN。

## 6. 硬件能力清单

按融合价值排序，建议 NPU 后续补充：

```text
1. streaming attention block:
   QK + scale/mask + softmax + PV + head merge

2. accumulator-side elementwise:
   SwiGLU for text MLP
   GELU for mmproj MLP

3. fused quant/pack:
   elementwise output directly becomes next GEMM int8 input
   output goes to SPM or next input buffer, not full DRAM tensor

4. output fusion:
   mvout/dequant + residual add

5. layout support:
   head split/merge, attention expected strides, optional KV cache layout
```

不建议优先做：

```text
standalone F32-to-CMA hardware activation pack
isolated QK/PV GEMM offload with CPU softmax between them
standalone norm hardware
patch embedding-only offload
```

## 7. 待办记录

1. `text MLP tile fusion: gate/up + SwiGLU + down` 有必要做，但暂缓。
2. mmproj 第一优先是 fused vision attention，不是单独补 QK/PV GEMM。
3. activation quant/pack 硬件化只有在 fused producer/consumer 模式下划算。
4. postprocess 高主要是 CMA 到 ggml tensor copy，应通过融合减少中间输出，而不是只优化 copy 循环。
5. 如果做 attention 硬件化，除 softmax 外还要处理 scale、mask、PV、head merge、layout，以及可选 ROPE/KV cache 数据路径。
