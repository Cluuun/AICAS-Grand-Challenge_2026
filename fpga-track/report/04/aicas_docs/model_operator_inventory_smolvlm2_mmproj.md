# SmolVLM2 文本模型与 mmproj 算子清单

本文面向这两个实际文件：

- 文本模型：`/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- 视觉编码器 + projector：`/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf`

重点关注：

- 这两个文件里实际有哪些算子
- 每类算子的参数是什么
- 典型输入输出形状是什么
- 特别是激活函数到底有哪些、在什么位置、公式是什么

## 1. 总览

### 1.1 文本模型

- `general.architecture = llama`
- `llama.block_count = 32`
- `llama.embedding_length = 960`
- `llama.feed_forward_length = 2560`
- `llama.attention.head_count = 15`
- `llama.attention.head_count_kv = 5`
- `llama.attention.key_length = 64`
- `llama.attention.value_length = 64`
- `llama.rope.dimension_count = 64`
- `llama.rope.freq_base = 100000.0`
- `llama.attention.layer_norm_rms_epsilon = 1e-5`
- `llama.context_length = 8192`
- `llama.vocab_size = 49280`

从张量命名和运行时图看，它是标准 `Llama` 解码器主干：

- Token Embedding
- 32 个 decoder block
- 每层都是 `RMSNorm -> Self-Attention -> Residual -> RMSNorm -> SwiGLU FFN -> Residual`
- 最后 `RMSNorm -> LM Head`

### 1.2 mmproj 模型

- `general.architecture = clip`
- `general.type = clip-vision`
- `clip.projector_type = idefics3`
- `clip.vision.image_size = 512`
- `clip.vision.patch_size = 16`
- `clip.vision.embedding_length = 768`
- `clip.vision.feed_forward_length = 3072`
- `clip.vision.block_count = 12`
- `clip.vision.attention.head_count = 12`
- `clip.vision.projection_dim = 960`
- `clip.vision.projector.scale_factor = 4`
- `clip.vision.attention.layer_norm_epsilon = 1e-5`
- `clip.use_gelu = True`
- `clip.vision.image_mean = [0.5, 0.5, 0.5]`
- `clip.vision.image_std = [0.5, 0.5, 0.5]`

从运行时图看，它由两部分组成：

- Vision Transformer 编码器
- IDEFICS3 projector

即：

`图像 slice -> patch embedding -> 12 层 vision block -> post layernorm -> pixel shuffle / patch merge -> 线性投影到 960 维`

## 2. 文本模型算子清单

下面用 `T` 表示当前 token 数，通常 prefill 时 `T` 是提示词长度。

### 2.1 输入与输出

#### 2.1.1 Token Embedding

- 权重：`token_embd.weight`
- 形状：`[960, 49280]`
- 类型：`Q8_0`
- 算子：`get_rows / embedding lookup`
- 输入：`[T]` 个 token id
- 输出：`[960, T]`

#### 2.1.2 输出头

- 末尾归一化权重：`output_norm.weight`
- 形状：`[960]`
- 类型：`F32`
- 输出投影权重：`output.weight`
- 形状：`[960, 49280]`
- 类型：`Q8_0`
- 输出 logits 形状：`[49280, T]`

### 2.2 每个 decoder block 的算子

每层都重复以下结构。

#### 2.2.1 Attention RMSNorm

- 权重：`blk.{i}.attn_norm.weight`
- 形状：`[960]`
- 算子：`RMSNorm`
- `eps = 1e-5`
- 输入输出形状：`[960, T] -> [960, T]`

公式：

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

这里没有 bias，只有 scale weight。

#### 2.2.2 Q / K / V 线性投影

- `blk.{i}.attn_q.weight`：`[960, 960]`
- `blk.{i}.attn_k.weight`：`[960, 320]`
- `blk.{i}.attn_v.weight`：`[960, 320]`
- 类型：`Q8_0`
- 算子：`MulMat`

运行时语义：

- `Q = Wq * x`，输出 `[960, T]`
- `K = Wk * x`，输出 `[320, T]`
- `V = Wv * x`，输出 `[320, T]`

再 reshape 为：

- `Q -> [64, 15, T]`
- `K -> [64, 5, T]`
- `V -> [64, 5, T]`

这说明：

- 每个 head 维度 `d_head = 64`
- Query 头数 `15`
- KV 头数 `5`
- 这是 `GQA`，不是 full MHA

#### 2.2.3 RoPE

- 算子：`ggml_rope_ext`
- 参数：
  - `n_rot = 64`
  - `rope.dimension_count = 64`
  - `freq_base = 100000.0`

输入输出形状：

- `Q: [64, 15, T] -> [64, 15, T]`
- `K: [64, 5, T] -> [64, 5, T]`

含义：

- 每个 head 的 64 维都参与 rotary position embedding

#### 2.2.4 Attention Score / Softmax / Value 聚合

主要算子：

- `MulMat` 计算 `K^T Q`
- `SoftMaxExt`
- `MulMat` 计算 `V * softmax(score)`
- `Permute / Contiguous / Reshape`

关键参数：

- `kq_scale = 1 / sqrt(64) = 0.125`

softmax 的输入最后会变成按注意力 head 展开的分数张量，逻辑上可理解为：

- 每个 query head 对所有历史位置做 softmax

虽然 `ggml` 内部 layout 比较底层，但语义上等价于：

```text
scores = (Q K^T) * 0.125
probs  = softmax(scores)
ctx    = probs V
```

#### 2.2.5 Attention 输出投影

- 权重：`blk.{i}.attn_output.weight`
- 形状：`[960, 960]`
- 类型：`Q8_0`
- 算子：`MulMat`
- 输入输出形状：`[960, T] -> [960, T]`

#### 2.2.6 残差加法

- 算子：`Add`
- 输入输出形状：`[960, T] + [960, T] -> [960, T]`

#### 2.2.7 FFN RMSNorm

- 权重：`blk.{i}.ffn_norm.weight`
- 形状：`[960]`
- 算子：`RMSNorm`
- `eps = 1e-5`
- 输入输出形状：`[960, T] -> [960, T]`

#### 2.2.8 FFN 线性层

- `blk.{i}.ffn_gate.weight`：`[960, 2560]`
- `blk.{i}.ffn_up.weight`：`[960, 2560]`
- `blk.{i}.ffn_down.weight`：`[2560, 960]`
- 类型：`Q8_0`

运行时语义：

- `gate = W_gate * x`，形状 `[2560, T]`
- `up   = W_up   * x`，形状 `[2560, T]`
- `ffn  = SwiGLU(gate, up)`，形状 `[2560, T]`
- `out  = W_down * ffn`，形状 `[960, T]`

#### 2.2.9 激活函数：SiLU / SwiGLU

文本模型最关键的非线性就在这里。

`SiLU` 公式：

```text
SiLU(x) = x / (1 + exp(-x))
```

`SwiGLU` 公式：

```text
SwiGLU(gate, up) = SiLU(gate) * up
```

对本模型的形状来说：

- `gate`：`[2560, T]`
- `up`：`[2560, T]`
- `SiLU(gate)`：`[2560, T]`
- `SwiGLU` 输出：`[2560, T]`

结论：

- 文本主干里没有 `GELU`
- 文本主干的 FFN 激活是 `SwiGLU`
- `SwiGLU` 内部实际用到的标量激活是 `SiLU`

## 3. mmproj 算子清单

mmproj 的运行时图来自 `tools/mtmd/clip.cpp` 的 `PROJECTOR_TYPE_IDEFICS3` 路径。

下面用单个 `512 x 512` 图像 slice 说明。对 IDEFICS3 来说，大图会先切成多个 slice，每个 slice 独立过这条图。

### 3.1 预处理

预处理发生在 GGUF 图之外，但它决定了后续形状：

- 原图先按长边约束做等比例 resize
- 再切成一个 overview + 若干 refined slice
- 每个 slice 会被规范到 `512 x 512` 分辨率路径
- 像素归一化：
  - `mean = [0.5, 0.5, 0.5]`
  - `std  = [0.5, 0.5, 0.5]`

因此单个 slice 进入编码器前的张量可视为：

- 输入图像：`[512, 512, 3]`

### 3.2 Patch Embedding

- 权重：`v.patch_embd.weight`
- 形状：`[16, 16, 3, 768]`
- bias：`v.patch_embd.bias`
- 形状：`[768]`
- 算子：`Conv2D`
- 参数：
  - kernel size = `16 x 16`
  - stride = `16 x 16`
  - padding = `0`

单个 `512 x 512` slice 的输出：

- patch 网格：`32 x 32`
- patch 数：`1024`
- 输出形状：`[768, 1024]`

### 3.3 位置编码

- 权重：`v.position_embd.weight`
- 形状：`[768, 1024]`
- 算子：`Add`

输入输出形状：

- `[768, 1024] + [768, 1024] -> [768, 1024]`

这里用的是 learned 2D patch position embedding，不是 RoPE。

### 3.4 12 个 vision block

每层的隐藏维度固定是 `768`，head 数固定是 `12`，所以每个 head 维度：

```text
d_head = 768 / 12 = 64
```

每个 block 结构如下。

#### 3.4.1 LayerNorm 1

- 权重：`v.blk.{i}.ln1.weight`
- bias：`v.blk.{i}.ln1.bias`
- 形状：`[768]`
- 算子：`LayerNorm`
- `eps = 1e-5`
- 输入输出：`[768, 1024] -> [768, 1024]`

公式：

```text
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

这里和文本模型不同：

- 文本模型是 `RMSNorm`
- vision tower 是标准 `LayerNorm`

#### 3.4.2 Q / K / V 投影

每层都有 bias。

- `v.blk.{i}.attn_q.weight`：`[768, 768]`
- `v.blk.{i}.attn_k.weight`：`[768, 768]`
- `v.blk.{i}.attn_v.weight`：`[768, 768]`
- `v.blk.{i}.attn_q.bias`：`[768]`
- `v.blk.{i}.attn_k.bias`：`[768]`
- `v.blk.{i}.attn_v.bias`：`[768]`

输入输出形状：

- 输入：`[768, 1024]`
- 线性后：`[768, 1024]`
- reshape 后：
  - `Q -> [64, 12, 1024]`
  - `K -> [64, 12, 1024]`
  - `V -> [64, 12, 1024]`

#### 3.4.3 Attention Softmax

关键参数：

- `kq_scale = 1 / sqrt(64) = 0.125`

主要算子：

- `Permute`
- `MulMat`
- `SoftMaxExt`
- `MulMat`
- `Permute`
- `Contiguous`

语义等价于：

```text
scores = (Q K^T) * 0.125
probs  = softmax(scores)
ctx    = probs V
```

softmax 输出与 attention score 同形，逻辑上是：

- 每个 head
- 每个 query patch
- 对全部 `1024` 个 key patch 做归一化

#### 3.4.4 Attention 输出投影

- `v.blk.{i}.attn_out.weight`：`[768, 768]`
- `v.blk.{i}.attn_out.bias`：`[768]`
- 输入输出：`[768, 1024] -> [768, 1024]`

#### 3.4.5 残差加法

- `Add`
- `[768, 1024] + [768, 1024] -> [768, 1024]`

#### 3.4.6 LayerNorm 2

- `v.blk.{i}.ln2.weight`：`[768]`
- `v.blk.{i}.ln2.bias`：`[768]`
- 算子：`LayerNorm`
- `eps = 1e-5`
- 输入输出：`[768, 1024] -> [768, 1024]`

#### 3.4.7 FFN

这里需要特别说明一个容易误读的点。

GGUF 里这组张量名是：

- `v.blk.{i}.ffn_down.weight`：`[768, 3072]`
- `v.blk.{i}.ffn_down.bias`：`[3072]`
- `v.blk.{i}.ffn_up.weight`：`[3072, 768]`
- `v.blk.{i}.ffn_up.bias`：`[768]`

但 `tools/mtmd/clip.cpp` 在加载时会检测这是 IDEFICS3 的“旧命名”，然后把 `ffn_up` 和 `ffn_down` 在运行时交换回来。也就是说：

- **GGUF 存储名** 有历史反转
- **运行时真正语义** 仍然是标准 MLP：`768 -> 3072 -> GELU -> 768`

运行时真实计算语义：

- 上投影：`[768, 1024] -> [3072, 1024]`
- `GELU`
- 下投影：`[3072, 1024] -> [768, 1024]`

#### 3.4.8 激活函数：GELU

mmproj 的 vision tower 使用 `GELU`，不是 `SwiGLU`。

公式：

```text
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * x * (1 + 0.044715 * x^2)))
```

参数常数：

- `sqrt(2 / pi) = 0.7978845608...`
- `coef = 0.044715`

在本模型里的形状：

- 输入：`[3072, 1024]`
- 输出：`[3072, 1024]`

结论：

- mmproj 视觉主干的 FFN 激活是 `GELU`
- 不是 `GEGLU`
- 也不是 `SiLU / SwiGLU`

### 3.5 Post LayerNorm

- `v.post_ln.weight`：`[768]`
- `v.post_ln.bias`：`[768]`
- 算子：`LayerNorm`
- `eps = 1e-5`
- 输入输出：`[768, 1024] -> [768, 1024]`

### 3.6 IDEFICS3 Pixel Shuffle / Patch Merge

这是 mmproj 路径最特殊的非线性前形状变换，虽然它不是激活函数，但很关键。

- 算子：
  - `Reshape`
  - `Pad`（如果宽高不能整除 `scale_factor`）
  - `Permute`
  - `Contiguous`
- 参数：
  - `scale_factor = 4`

对单个 `512 x 512` slice：

- patch grid 初始是 `32 x 32`
- `768` 通道

变换前：

- `cur = [768, 1024]`

重排成 2D patch 网格后：

- `[768, 32, 32]`

经过 `scale_factor = 4` 的 patch merge 后：

- 空间尺寸：`32 x 32 -> 8 x 8`
- 通道数：`768 -> 768 * 4 * 4 = 12288`

最终输出：

- `[12288, 64]`

这一步不是数值非线性，而是把相邻 patch 信息折叠进通道维。

### 3.7 Projector 线性层

- 权重：`mm.model.fc.weight`
- 形状：`[12288, 960]`
- 当前文件里类型：`I8`
- 无 bias
- 算子：`MulMat` 或 AICAS 自定义 `W8A8` 路径

输入输出形状：

- 输入：`[12288, 64]`
- 输出：`[960, 64]`

也就是说：

- 每个 `512 x 512` slice 最终变成 `64` 个视觉 token
- 每个 token 的维度是 `960`
- 这和文本主干 hidden size 对齐

## 4. 激活函数专项总结

这是你特别关注的部分，这里单独汇总。

### 4.1 文本模型里有哪些激活函数

#### 4.1.1 SiLU

位置：

- `SwiGLU` 的 gate 分支内部

公式：

```text
SiLU(x) = x / (1 + exp(-x))
```

参数：

- 无可学习参数
- 无超参数

典型形状：

- 输入：`[2560, T]`
- 输出：`[2560, T]`

#### 4.1.2 SwiGLU

位置：

- 每个 decoder block 的 FFN

公式：

```text
SwiGLU(gate, up) = SiLU(gate) * up
```

参数：

- 无额外可学习参数
- gate 来自 `ffn_gate.weight`
- up 来自 `ffn_up.weight`

典型形状：

- `gate = [2560, T]`
- `up   = [2560, T]`
- 输出 `= [2560, T]`

#### 4.1.3 Softmax

位置：

- self-attention

参数：

- scale = `1 / sqrt(64) = 0.125`

典型逻辑形状：

- 对每个 head、每个 query 位置
- 在 key 序列维上归一化

#### 4.1.4 RMSNorm

虽然通常不叫“激活函数”，但它确实是关键逐元素非线性归一化。

位置：

- 每层 attention 前
- 每层 FFN 前
- 最终输出层前

参数：

- learnable weight：`[960]`
- `eps = 1e-5`

### 4.2 mmproj 里有哪些激活函数

#### 4.2.1 GELU

位置：

- 每个 vision block 的 FFN 中间

公式：

```text
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * x * (1 + 0.044715 * x^2)))
```

参数：

- 无可学习参数
- 常数 `0.044715`

典型形状：

- 输入：`[3072, 1024]`
- 输出：`[3072, 1024]`

#### 4.2.2 Softmax

位置：

- vision self-attention

参数：

- scale = `1 / sqrt(64) = 0.125`

典型逻辑形状：

- 每个 head
- 每个 query patch
- 对 `1024` 个 key patch 做归一化

#### 4.2.3 LayerNorm

位置：

- 每个 vision block 的 `ln1`
- 每个 vision block 的 `ln2`
- `post_ln`

参数：

- learnable weight：`[768]`
- learnable bias：`[768]`
- `eps = 1e-5`

结论：

- mmproj 不使用 `SiLU`
- mmproj 不使用 `SwiGLU`
- mmproj FFN 的主激活是 `GELU`

## 5. mmproj 的 W8A8 量化参数

当前 `mmproj.gguf` 不是普通 F16 视觉模型，而是 AICAS 的 SmoothQuant/W8A8 版本。

元数据里：

- `aicas.w8a8.schema = smolvlm2_idefics3_static_w8a8_v1`
- `aicas.w8a8.tensor_count = 73`

覆盖的线性层一共 73 个：

- `mm.model.fc.weight` x1
- `v.blk.{0..11}.attn_q.weight` x12
- `v.blk.{0..11}.attn_k.weight` x12
- `v.blk.{0..11}.attn_v.weight` x12
- `v.blk.{0..11}.attn_out.weight` x12
- `v.blk.{0..11}.ffn_up.weight` x12
- `v.blk.{0..11}.ffn_down.weight` x12

每个量化层会带这些参数：

- `act_scale`
- `act_scale_q8_24`
- `act_zero_point`
- `act_quant_mode`
- `weight_scale_mode`
- `weight_scale`
- `dequant_scale_q8_24`
- `smooth_enabled`
- `smooth_alpha`
- `smooth_eps`
- `smooth_scale`
- `sum_w`

### 5.1 这些参数分别表示什么

#### 5.1.1 激活量化参数

- `act_quant_mode = asymmetric_u8`
- `act_scale`：激活量化 scale，标量
- `act_zero_point`：激活 zero point，标量

量化公式：

```text
q = round(x / act_scale) + act_zero_point
q clamp 到 [0, 255]
```

#### 5.1.2 权重量化参数

- 权重类型：`I8`
- `weight_scale_mode = per_channel`
- `weight_scale`：按输出通道给 scale

因此：

- `attn_q/k/v/out` 的 `weight_scale` 长度是 `768`
- `ffn_up` 运行时上投影输出维是 `3072`，对应 `weight_scale` 长度 `3072`
- `ffn_down` 运行时下投影输出维是 `768`，对应 `weight_scale` 长度 `768`
- `projector` 输出维是 `960`，对应 `weight_scale` 长度 `960`

#### 5.1.3 SmoothQuant 参数

- `smooth_enabled = true`
- `smooth_alpha = 0.5`
- `smooth_eps = 1e-6`
- `smooth_scale`：按输入通道的 smoothing 因子

长度规律：

- attention 线性层输入通道 `768`，所以 `smooth_scale` 长度 `768`
- vision FFN 上投影输入通道 `768`，所以长度 `768`
- vision FFN 下投影输入通道 `3072`，所以长度 `3072`
- projector 输入通道 `12288`，所以长度 `12288`

## 6. 最后给一个最短结论

如果只看“激活函数到底是什么”：

- 文本模型主干：`SwiGLU`
  - 内部标量激活是 `SiLU`
  - 形状核心是 `2560`
- mmproj 视觉主干：`GELU`
  - FFN 形状核心是 `3072`
- 两边 attention 都有 `Softmax`
  - scale 都是 `1 / sqrt(64) = 0.125`
- 文本归一化是 `RMSNorm`
- mmproj 归一化是 `LayerNorm`

如果你后面要继续做 NPU 算子支持，优先级通常会是：

1. `MulMat / GEMM`
2. `LayerNorm` 和 `RMSNorm`
3. `Softmax`
4. `SiLU / SwiGLU`
5. `GELU`
6. `Reshape / Permute / Contiguous / Pad`

