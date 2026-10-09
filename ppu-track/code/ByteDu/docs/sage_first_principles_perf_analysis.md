# SageAttention first-principles performance analysis

## Executive model

SageAttention's advertised speed is a kernel-level result: it replaces full-precision `QK^T`/`PV` attention with quantized `QK^T`, optional low-precision `PV`, better tiling, and lower HBM traffic. The AICAS decode workload is not a long-sequence attention benchmark. It is batch-1, `q_len=1`, 28-layer autoregressive generation where each generated token also pays QKV projection, o_proj, MLP gate/up/down, lm_head argmax, graph replay, cache maintenance, and Python/control overhead.

Current project measurements show Sage is already hitting the hot path and improving attention/cache time, but that saving is too small relative to the full layer executor:

| Stage over 127 decode steps x 28 layers | Current Sage CUDA time | System implication |
| --- | ---: | --- |
| QKV projection | ~30.6 ms | Already comparable to attention; weight path matters. |
| Sage attention/cache | ~45.9 ms | Better than non-Sage attention/cache, but not dominant. |
| o_proj handoff | ~21.5 ms | Still a separate GEMV/residual kernel. |
| MLP gate/up + SwiGLU | ~68.6 ms | Largest steady-state bottleneck. |
| down_proj / fused down | ~45.9 ms | Comparable to the whole Sage attention path. |
| lm_head + argmax | ~20.2 ms | Must avoid materialized logits. |
| Total profiled CUDA | ~234.6 ms | 545 tok/s means ~1.83 ms/token end-to-end. |

To reach 660 tok/s for 128-token generation, steady decode must fall to about `127 / 660 = 192 ms` plus small benchmark overhead. That requires roughly 40 ms less CUDA time than the current Sage path. A perfect attention kernel cannot supply that alone: the remaining non-attention rows are already over 185 ms. Sage therefore has to become the center of a decode executor contract, not just an attention call.

## 1. Where SageAttention's theoretical speedup comes from

Upstream `/root/JunkratDev/SageAttention` exposes `sageattn`, `sageattn_qk_int8_pv_fp16_triton`, `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda`, and Hopper/Blackwell variants. Its assumptions are visible in `sageattention/core.py`, `sageattention/quant.py`, and the CUDA kernels under `csrc/qattn/`.

| Mechanism | Upstream assumption | AICAS decode relevance |
| --- | --- | --- |
| INT8 Q/K | Q and K are quantized per block/per warp, scales are FP32, then `QK^T` uses lower precision. | Useful only if Q quant and K cache quant are fused into decode; allocating/quantizing full tensors per token would erase gains. |
| FP16/BF16 or FP8 V | `PV` can use FP16 or FP8 V with specialized kernels. README notes `8+16` and `8+8` variants. | Current default uses K INT8 + V BF16. V BF16 keeps quality and avoids slower V-int8 path seen in this repo, but leaves V bandwidth as a remaining limit. |
| Reduced HBM traffic | K/Q bytes shrink and some intermediate writes are avoided inside the attention kernel. | Benefit scales with KV length and attention share. For `q_len=1`, the layer still pays large GEMV/MLP traffic every token. |
| Tiled attention/softmax | Kernel fuses score, softmax, and value accumulation with efficient reductions. | Helpful, but softmax over short/medium KV is not the only bottleneck. |
| Better occupancy/register tradeoff | CUDA implementations target specific SMs and head dimensions. | The project uses a custom decode kernel rather than upstream full-sequence API because upstream does not provide a KV-cache decode API. |
| Long-context amortization | Kernel benchmark TOPS exclude quantization/smoothing and are strongest when attention dominates. | AICAS fixed `max_new_tokens=128` and prompt buckets make average KV length modest; quant/cache overhead is amplified relative to long-context attention. |

The levels must be separated:

1. **Kernel microbenchmark speedup**: upstream attention-only TOPS, explicitly excluding quantization/smoothing in the README. This is not an end-to-end claim.
2. **Attention-stage speedup**: in this repo, attention/cache improved from about 61.6 ms to 45.9 ms over a 1-sample profile when using fused Sage K-INT8/V-BF16.
3. **Decode-layer speedup**: a layer also includes QKV, o_proj, MLP, residual/norm, and cache writes. Current Sage changes only part of that layer.
4. **Generate throughput speedup**: benchmark throughput also includes prefill, lm_head, graph replay, EOS behavior, token copying, and answer collection. Current 1-sample baseline is 545.35 tok/s, far below 660.

## 2. Does the competition decode scene fit Sage's best conditions?

Fixed scene:

```text
batch = 1
decode q_len = 1
max_new_tokens = 128
num_layers = 28
hidden_size = 2048
num_q_heads = 16
num_kv_heads = 8
head_dim = 128
GQA = 16Q / 8KV
greedy decode
CUDA Graph
static workspace
Qwen3-VL-2B
```

This scene only partially matches Sage's optimal conditions.

| Question | Answer |
| --- | --- |
| Is attention the largest bottleneck? | No. Current profile rows put MLP gate/up at ~68.6 ms, down_proj at ~45.9 ms, QKV at ~30.6 ms, lm_head at ~20.2 ms, and Sage attention/cache at ~45.9 ms. |
| How large is KV length? | It starts at prompt length and grows by 127 tokens. Image-token routing controls prompt length; KV is not an extreme long-context benchmark. Sage gains are therefore bounded. |
| Is `q_len=1` favorable? | It is favorable for KV-cache decode if the kernel is native to decode, but unfavorable for upstream full-sequence Sage APIs because quant/scale overhead has no batch/sequence amortization. |
| Is online quant amplified? | Yes. Per-token, per-layer K quant and scale write are unavoidable for a K-INT8 cache, but full-history K/V requant, dynamic scale allocation, or runtime weight pack would be fatal. |
| Are scales expensive? | Per-token K scale is only `[8]` floats per layer, but random/separate scale loads can still hurt if not coalesced. Current layout is `[layer, token, kv_head]`, contiguous for the active token. |
| Does GQA duplicate KV reads? | The Python path does not expand KV; the native Sage decode maps 16 Q heads to 8 KV heads internally. However, the kernel can still reload the same KV head for two Q heads unless it explicitly reuses GQA groups in shared/register memory. |
| Is V BF16 a new bottleneck? | Yes, partly. K is 1 byte + scale, V remains 2 bytes. For decode `PV`, V bandwidth can dominate after K compression. V INT8/FP8 is the natural next direction, but prior V-int8 path was slower/quality-risky and must be gated by end-to-end data. |
| Does CUDA Graph solve executor overhead? | It removes most Python/launch overhead from the loop. Graph-off throughput collapses in prior measurements, so every new path must be graph-capturable. |

## 3. Quantization audit: too little vs too much

### Current effective quantization status

| Component | Current status | Risk |
| --- | --- | --- |
| Q in Sage attention | Quantized in native Sage fused decode scratch (`_decode_query_i8_scratch`) for CUDA path. | Good; upstream's best branch expects Q/K INT8. |
| K cache | INT8 side cache with FP32 scale, written during prefill/decode. | Good; must avoid full-prefix requant. |
| V cache | BF16 default for Sage. INT8 path exists but is opt-in/fallback. | V bandwidth remains, but previous V-int8 data did not justify default. |
| QKV projection weights | Packed rowwise Q8 by default when codebook/runtime Q8 is enabled. | Good; must ensure no decode-loop pack/transpose. |
| o_proj weights | Packed Q8 exists, but current Sage 128-token branch uses BF16 `addmv_bn1` because it measured faster than Q8 handoff. | This is a case where "more quant" was worse. |
| MLP gate/up | Interleaved packed Q8 and native fused norm+gate_up+SwiGLU path. | Largest bottleneck remains; W8A8 can be tested but activation quant cost may outweigh. |
| down_proj | Packed Q8 and fused down+residual+sumsq path. | Still ~46 ms; A8W8/W8A8 down variants need end-to-end gating. |
| lm_head | Q8 direct argmax is enabled by default, avoiding full logits in decode. | Still ~20 ms; can tune block layout or use native weight-only argmax if faster. |

### "Too little quantization"

Only K INT8 + V BF16 does not fully realize Sage's ideal `8+8` path. In this workload, after K is compressed, `PV` still streams BF16 V and returns BF16 attention output. Also, o_proj/MLP/down/lm_head dominate the layer, so attention-only quantization cannot deliver 660 tok/s. Full executor quantization is required, but only where it is net positive.

### "Too much quantization"

The repo already found examples where quantization hurts:

- Q8 o_proj after Sage was slower than BF16 `addmv_bn1` for the 128-token branch.
- Half-layer fusion that recomputed/exported RMS stats reduced gate row time but added a new launch/HBM scan and reduced total throughput.
- W8A8 paths require activation quant every token. For `q_len=1`, activation quant can cost more than the GEMV bytes it saves unless fused into an existing norm/add kernel.
- V INT8 writes V scale and dequantizes during attention; it can lose if the V quant/dequant path is not deeply fused or if quality falls.

Rules:

| Quantization type | Keep/delete/offline policy |
| --- | --- |
| Weight quantization | Must be offline at model init or package build; decode loop only reads packed weights/scales. |
| Activation quantization | Keep only when fused into add/norm/GEMV or when benchmarked faster per target. |
| KV cache quantization | K INT8 should stay for Sage. V INT8/FP8 should be long-KV or measured-default only. |
| Scale tensors | Allocate once; scales must be contiguous, graph-stable, and FP32 where native extensions require FP32. |
| Runtime pack/transpose | Forbidden in decode loop. Existing transposes/packing are in `_build_packed_text_layers`, acceptable at init. |

## 4. Offline quantization and packing opportunities

Current `_build_packed_text_layers()` already constructs:

- concatenated QKV weights and transpose;
- concatenated/interleaved gate_up weights and transpose;
- packed rowwise Q8 for QKV, gate_up, interleaved gate_up, down_proj, and o_proj;
- optional int8 buffers for W8A8;
- BF16 transposed o_proj and down_proj weights.

Remaining offline/package-build priorities:

| Item | Desired Sage-native form | Decode-loop requirement |
| --- | --- | --- |
| QKV | Q8 packed rows and/or int8 rows already attached from bundled codebook. | No `torch.cat`, `transpose`, `contiguous`, or weight quant in token loop. |
| o_proj | Keep both Q8 packed and BF16 transposed; dispatch by measured branch. | Sage 128-token default can use BF16 `addmv_bn1`; long-answer can use conservative path. |
| MLP gate/up | Interleaved gate/up Q8 and optional int8 W8A8 buffers. | Gate/up activation quant only if fused with norm. |
| down_proj | Q8 packed plus optional int8. | Down activation quant only if fused with down+residual+sumsq. |
| lm_head | Q8 packed direct argmax, optional int8/Acext weight-only. | Never materialize full logits in steady decode. |
| KV layout metadata | K scale `[L,S,Hkv]` contiguous per token/head. | No dynamic scale creation. |
| GQA layout | K/V `[S,Hkv,D]` per layer. | Native kernel maps two Q heads per KV head without Python repeat. |

## 5. KV cache and scale layout

Current active layout:

```text
K int8 cache:  [num_layers, 1, max_cache_len, 8, 128]
V bf16 cache:  [num_layers, 1, max_cache_len, 8, 128]
K scale cache: [num_layers, max_cache_len, 8]
Q scratch:     [16, 128] plus Q int8/scale scratch
Output:        [1, 1, 16, 128]
```

For a fixed layer this exposes contiguous `[token, kv_head, head_dim]` slices. K head-dim loads are contiguous; scale loads are contiguous across 8 KV heads per token. This is reasonable for the current native kernel, but not necessarily Sage-native enough.

AB layout matrix to keep:

| KV layout | Expected effect | Current conclusion |
| --- | --- | --- |
| K0 current `[S,H,D]`, scale separate | Simple and working. | Default. |
| K1 token-major current | Same as K0. | Good for appending one token and scanning by token. |
| K2 head-major `[H,S,D]` | Better per-head sequential scan, worse append/coalescing depending kernel. | Needs native kernel change. |
| K3 interleaved K+scale | Reduces separate scale pointer/load, may improve locality. | Requires new native cache writer/attention ABI. |
| K4 separated K/scale | Current. | Baseline. |
| K5 GQA-grouped | Store KV head next to its two Q-head outputs or schedule two Q heads per KV tile. | Best theoretical reuse, kernel-level work. |
| K6 layer-local packed | Current layer indexing already passes layer-local views. | Fine. |

AB value matrix:

| Value path | Attention time | Total tok/s | Accuracy risk | Conclusion |
| --- | ---: | ---: | --- | --- |
| K INT8 + V BF16 | ~45.9 ms attention/cache | ~545 tok/s current | Low | Default because measured stable. |
| K INT8 + V INT8 | Pending rerun | Pending | Medium | Keep opt-in until end-to-end wins and text is stable. |
| K INT8 + V FP8 | Not implemented for KV-cache decode | Pending | Medium | Promising only if native decode kernel supports FP8 V without extra transforms. |
| K BF16 + V BF16 | Slower attention/cache historically | ~494 tok/s no-Sage graph-on | Low | Baseline/accuracy path, not throughput. |
| K scale interleaved | Not implemented | Pending | Low | Next native ABI target. |

## 6. Online vs offline quantization strategy

Decode-loop bans:

- no weight quantization;
- no weight transpose;
- no weight pack;
- no dynamic scale tensor allocation;
- no layout transforms independent of the current token;
- no per-step Python dispatch change inside captured graph.

Allowed online work:

- K cache quant for the newly written token, fused with Q/K norm and RoPE;
- Q quant for the current token, fused with Q/K norm and cache write;
- activation quant for W8A8 only if fused into an existing norm/add/down/lm_head stage;
- per-token lm_head hidden quant only if it beats Q8/BF16 argmax and quality is stable.

## 7. System A/B experiment matrix

Every row must record 10-sample and 30-sample throughput, Sage hit, graph hit, fallback count, token_count, attention/cache time, QKV time, o_proj time, MLP time, lm_head time, total CUDA, and quality.

### Attention quant matrix

| ID | Config | Current status |
| --- | --- | --- |
| A0 | No Sage, graph on | Historical ~494 tok/s. |
| A1 | Sage K INT8 + V BF16 | Current default, ~545 tok/s 1-sample baseline. |
| A2 | Sage Q/K INT8 + V BF16 | Current native fused path effectively does this. |
| A3 | Sage K INT8 + V INT8 | Implemented opt-in, not default. |
| A4 | Sage Q/K/V INT8 | Native entry exists, must rerun quality/perf. |
| A5 | Sage only on long KV lengths | Layer-selective/mixed policy was slower; length-selective still needs a cheap static rule. |
| A6 | Sage layer selective | Historical partial layer policies slower. |

### Weight quant / pack matrix

| ID | Config | Current status |
| --- | --- | --- |
| W0 | BF16 QKV/o_proj/MLP/lm_head | Accuracy baseline, too slow. |
| W1 | offline Q8 QKV only | Available. |
| W2 | offline Q8 o_proj only | Available; slower than BF16 handoff in Sage 128-token branch. |
| W3 | offline Q8 MLP gate/up/down | Available and default. |
| W4 | offline Q8 lm_head | Available and default. |
| W5 | Q8 QKV + Q8 o_proj | Available; not current best with Sage. |
| W6 | Q8 MLP + Q8 lm_head | Available and default. |
| W7 | all feasible offline Q8 | Current default except Sage o_proj branch uses BF16 because measured faster. |

### KV layout matrix

See section 5. K0/K4 is current; K3/K5 are the native-kernel redesign targets.

### Runtime fusion matrix

| ID | Config | Current status |
| --- | --- | --- |
| F0 | current fused QKV prep + Sage + BF16 o_proj | Default. |
| F1 | QKV prep fused with quant/cache write | Implemented in native `sage_decode_qkv_k_i8_v_bf16`. |
| F2 | Sage + o_proj handoff optimized | BF16 `addmv_bn1` implemented. |
| F3 | o_proj + residual/norm | Half-layer attempt showed separate rstd launch loses; needs native epilogue fusion. |
| F4 | MLP gate_up/SwiGLU optimized | Interleaved Q8 fused norm/gate_up exists, still largest bottleneck. |
| F5 | lm_head argmax optimized | Q8 direct argmax exists; Acext weight-only/native variants need gating. |

## 8. Sage-native decode executor contract

Target dataflow:

```text
token_id
 -> embed
 -> layer 0..27:
      input_norm
      offline-packed QKV matvec
      fused online Q/K norm + RoPE + Q quant + K cache quant/write + V cache write
      Sage decode attention with Sage-native KV/scale layout
      o_proj + residual epilogue
      post-attn norm + MLP gate/up/SwiGLU
      down_proj + residual + next-layer norm/QKV
 -> final_norm
 -> offline-packed lm_head argmax
 -> next_token
```

The current executor already does pieces of this, but the missing 660 path is:

1. Reduce MLP gate/up and down rows, not just attention.
2. Make o_proj epilogue compute residual sumsq for the next RMSNorm without an extra launch.
3. Keep lm_head argmax native/packed and avoid logits.
4. Preserve CUDA Graph coverage and self-incrementing decode loop.
5. Add native Sage KV layout variants only when they remove real HBM traffic.

## 9. Root-cause conclusion

SageAttention has not reached 660+ because the system is no longer attention-bound. Its theory is valid, but the current scene consumes the theoretical benefit through:

- `q_len=1` quant/scale overhead with modest KV lengths;
- V BF16 bandwidth after K compression;
- MLP gate/up and down projection dominating total CUDA time;
- separate o_proj/residual/RMSNorm handoff;
- lm_head still costing about 20 ms even with direct argmax;
- graph/executor constraints that make extra launches expensive;
- upstream Sage being a full-attention API, not a native KV-cache decode executor.

The next optimization must therefore treat Sage as a contract for the whole layer: offline-packed weights, fused token quant/cache write, GQA-aware KV/scale layout, o_proj epilogue fusion, MLP/down acceleration, and graph-stable lm_head argmax. If a change only improves an attention microbenchmark and does not reduce the 10/30-sample benchmark, it is not a valid success.
