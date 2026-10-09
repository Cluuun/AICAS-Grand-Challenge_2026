# Decode throughput next report

This round continued from the fused Sage decode path instead of combining unrelated processor or vision-token switches. The current default improves the 128-token decode benchmark above the 533 tok/s historical floor, but it does **not** satisfy the user's 560 tok/s success target.

## Current default command

```bash
AICAS_VISION_TOKEN_OPT=0 \
AICAS_VISION_RESIZE_MODE=off \
JUNKRAT_PROCESSOR_MAX_VISUAL_TOKENS=768 \
JUNKRAT_ENABLE_SAGE_DECODE_ATTN=1 \
JUNKRAT_SAGE_IMPL=cuda \
JUNKRAT_SAGE_QK_PATH=i8 \
JUNKRAT_SAGE_VALUE_PATH=bf16 \
JUNKRAT_ENABLE_SAGE_FUSED_QKV_DECODE=1 \
JUNKRAT_SAGE_LAYERS=all \
JUNKRAT_DISABLE_DECODE_GRAPH=0 \
JUNKRAT_ENABLE_ACEXT_DECODE_LM_HEAD=0 \
JUNKRAT_ENABLE_TRITON_LINEAR_ARGMAX=1 \
JUNKRAT_TRITON_LINEAR_ARGMAX_BLOCK_ROWS=16 \
JUNKRAT_TRITON_LINEAR_ARGMAX_BLOCK_K=512 \
python benchmark.py --model-path ./Qwen3-VL-2B-Instruct --dataset-path ./data --output result_decode_next_30.json --num-samples 30
```

## End-to-end results

| Run | Throughput | TTFT | Artifact |
| --- | ---: | ---: | --- |
| Historical best floor | 533 tok/s | n/a | user-provided reference |
| Sage + BF16 o_proj, before lm_head retune, 10 samples | 535.61 tok/s | 42.40 ms | `result_sage_bf16_oproj_10.json` |
| Sage + BF16 o_proj, before lm_head retune, 30 samples | 535.63 tok/s | 42.69 ms | `result_sage_bf16_oproj_30.json` |
| Current default, 10 samples | 544.84 tok/s | 42.75 ms | `result_decode_next_10.json` |
| Current default, 30 samples | 543.73 tok/s | 43.43 ms | `result_decode_next_30.json` |
| Current default, 150 samples | 544.56 tok/s | 43.49 ms | `result_decode_next_150.json` |

The net gain versus the 533 tok/s floor is about +2.0%, but the 30-sample result is still 16.27 tok/s below the 560 tok/s target.

## Why the earlier CUDA delta looked inconsistent

The previous statement that `attention/cache` saved 15.70 ms and `o_proj` saved 3.61 ms while total CUDA only moved by 3.37 ms mixed two different comparisons:

```text
现象：attention/cache 和 o_proj 的 row-level deltas looked larger than the total CUDA delta.
证据：no-Sage -> old Sage explained the attention/cache drop; old Sage -> BF16 o_proj explained the o_proj drop.
根因：the reported rows were not a single before/after pair. One row compared no-Sage vs old Sage, another compared old Sage vs new Sage.
修复：re-profiled with one consistent current Sage path and treated the old row as an accounting artifact.
修复后数据：current Sage profiler self CUDA total is 230.032 ms; major rows sum to the same total order, with Sage partial+reduce at 44.197 ms and BF16 o_proj at 21.546 ms.
```

## Decode module breakdown

The exact 533 tok/s implementation is a historical throughput reference, not an isolated profile artifact in this working tree. The table therefore uses the captured no-Sage/old-Sage profile data where available and the current default profile for the final column.

| Module | Historical / prior path | Current Sage path | Difference | Conclusion |
| --- | ---: | ---: | ---: | --- |
| embed / token select | ~0.49 ms indexSelect | 0.487 ms | flat | Not a target. |
| RMSNorm / fused add RMSNorm | ~1.36 ms boundary + QKV norm folded | 1.362 ms boundary, 30.470 ms folded into QKV | flat | The hot norm is already inside Q8 QKV kernels. |
| QKV projection | Q8 packed executor | 30.470 ms (`norm_gemv_sumsq_q8_bf16`, 3429 calls) + 1.362 ms boundary | not improved | BF16 QKV tested faster in isolation but worse end-to-end. |
| Q/K norm + RoPE + quant + cache write + attention | prior attention/cache row ~61.59 ms | 34.978 ms Sage partial + 9.219 ms reduce | faster | Sage is the main accepted decode improvement. |
| SageAttention kernel | absent | 44.197 ms total | new cost replacing Flash/cache path | Hit rate is 100% in graph path. |
| o_proj | old Sage Q8 handoff ~25.06 ms | 21.546 ms BF16 `addmv_bn1` | -3.5 ms | Sage output now feeds o_proj without Q8 repack handoff. |
| MLP gate_up + SwiGLU | fused Q8 | 67.830 ms | largest row | This is now the largest remaining bottleneck. |
| down_proj | fused Q8 down/add/sumsq | 42.551 ms | major row | Second largest remaining per-layer cost. |
| lm_head + argmax | Acext decode lm_head was slower in this path | 18.063 ms Q8 Triton stage1 + 0.668 ms stage2 | improved after retune | Still one of the largest non-layer costs. |
| Python / host overhead | graph-dependent | graph on 544.68 vs graph off 495.57 tok/s | graph saves ~49 tok/s | Graph still covers the critical path. |
| memory allocation / fill / copy | low CUDA but visible CPU allocation in profiler | Memset 0.486 ms, DtoD 0.009 ms, `empty_strided` CPU 22.94 ms in profiler setup | mostly outside steady graph | Not the main CUDA bottleneck. |

Current default profiler artifact: `analysis_outputs/decode_next_ab/profile/default_next_profiler.txt`.

## Current CUDA hotspot table

| Kernel / operation | Calls | Self CUDA | Avg per call | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `fused_norm_gate_up_swiglu_q8_interleaved_kernel` | 3556 | 67.830 ms | 19.075 us | Largest bottleneck; gate/up and SwiGLU are already fused. |
| `down_add_sumsq_q8_bf16_kernel` | 3429 | 42.551 ms | 12.409 us | Down projection plus residual/sumsq remains large. |
| `sage_decode_qkv_k_i8_v_bf16_partial_kernel` | 3556 | 34.978 ms | 9.836 us | Fused Sage Q/K norm, RoPE, K quant/scale, V write, partial attention. |
| `norm_gemv_sumsq_q8_bf16_kernel` | 3429 | 30.470 ms | 8.886 us | QKV projection path remains expensive. |
| `addmv_bn1_kernel` | 3556 | 21.546 ms | 6.059 us | BF16 Sage o_proj handoff; faster than old Q8 handoff. |
| `_triton_linear_argmax_q8_stage1` | 127 | 18.063 ms | 142.227 us | Tuned Q8 lm_head; still a large 128-token cost. |
| `sage_decode_reduce_kernel` | 3556 | 9.219 ms | 2.593 us | Sage reduction stage. |
| `_triton_linear_argmax_stage2` | 127 | 0.668 ms | 5.261 us | Not the main lm_head bottleneck. |

## Launch count

For 127 decode steps and 28 text layers, the profile records 3556 layer invocations.

| Scope | Approximate launches |
| --- | ---: |
| Per layer, hot path | 6 main launches: QKV, Sage partial, Sage reduce, o_proj, gate_up+SwiGLU, down/add/sumsq |
| Per decode token | about 168 layer launches plus lm_head stage1/stage2 and boundary kernels |
| 128-token throughput profile | 3556 Sage partial, 3556 Sage reduce, 3556 o_proj, 3556 gate_up, 3429 down, 3429 QKV, 127 lm_head stage1/stage2 |

This launch count explains why attention-only savings are diluted: each token still executes many non-attention GEMV/pointwise kernels across 28 layers.

## MLP A/B

| Path | gate_up | SwiGLU | down_proj | 10-sample throughput | Conclusion |
| --- | --- | --- | --- | ---: | --- |
| Current default | packed Q8 fused | fused in gate_up kernel | Q8 down/add/sumsq | 544.84 tok/s final default | Best accepted default. |
| Disable fused norm MLP experiment | altered existing MLP switch | altered existing MLP switch | unchanged | 545.10 tok/s | Small noisy gain; not adopted as a Sage-specific code improvement. |
| Disable Triton fused MLP experiment | altered existing MLP switch | altered existing MLP switch | unchanged | 545.19 tok/s | Small noisy gain; not adopted. |
| W8A8 gate | W8A8 gate/up | fused | Q8 down | 536.16 / 539.96 tok/s | Negative versus current default. |
| W8A8 down | Q8 | fused | W8A8 down | 504.68 tok/s | Strongly negative. |
| W8A8 qkv | Q8 MLP | fused | Q8 down | 532.17 tok/s | Negative. |
| W8A8 o_proj / auto | existing opt-in W8A8 switch | fused | Q8 down | 545.76 tok/s | Slightly faster in 10-sample A/B, but not adopted because the user disallowed combining existing optimization switches. |

The accepted MLP conclusion is that gate_up+SwiGLU and down_proj now dominate, but the available switches do not provide a clean Sage-specific path to 560. A new native fusion is needed.

## lm_head A/B

| Path | 10-sample throughput | Conclusion |
| --- | ---: | --- |
| Acext decode lm_head enabled | slower than Q8 path in current Sage run | Disabled by default. |
| Runtime Q8 lm_head disabled | 414.74 tok/s | Required for throughput. |
| Q8 Triton argmax, effective BLOCK_ROWS=16, BLOCK_K=512 | 545.36 tok/s in sweep, 544.84 final default | Best accepted setting. |
| Q8 Triton argmax, effective BLOCK_ROWS=16, BLOCK_K=768 | 544.75 tok/s | Slightly slower. |
| Q8 Triton argmax, effective BLOCK_ROWS=16, BLOCK_K=1024 | 543.90 tok/s | Slower. |
| Q8 Triton argmax, effective BLOCK_ROWS=16, BLOCK_K=256 / 384 | 500.11 tok/s | Bad on this GPU/runtime. |

`triton_linear_argmax.py` clamps block rows to at least 16, so older `rows1` sweep labels were effectively `rows16`. The defaults now use `BLOCK_ROWS=16` explicitly to avoid misleading reports.

## CUDA Graph A/B

| Path | Throughput | TTFT | Sage stats |
| --- | ---: | ---: | --- |
| Sage + graph on | 544.68 tok/s | 42.61 ms | `decode_calls=28644`, `cuda_calls=28644`, `graph_decode_replay_calls=28644`, `miss_count=0` |
| Sage + graph off | 495.57 tok/s | 42.64 ms | eager stats not accumulated in the graph replay counter |
| no-Sage + graph on | 499.54 tok/s | 42.52 ms | comparison path |
| no-Sage + graph off | 198.07 tok/s | 42.72 ms | comparison path |

Sage does not break CUDA Graph capture in the accepted path. Graph replay covers the fused Sage kernels, BF16 o_proj, MLP kernels, and Q8 lm_head.

## Sage layer policy

| Policy | 10-sample throughput | Conclusion |
| --- | ---: | --- |
| all | 544.87 tok/s | Best. |
| 0-13 | 513.11 tok/s | Mixed fallback/cache path is too expensive. |
| 14-27 | 516.13 tok/s | Mixed fallback/cache path is too expensive. |
| even layers | 499.30 tok/s | Bad. |
| odd layers | 497.76 tok/s | Bad. |

All 28 text layers should use Sage in the decode path.

## KV and scale layout

Current Sage decode cache layout:

```text
K INT8 cache: [num_layers, 1, max_cache_len, 8, 128]
V BF16 cache: [num_layers, 1, max_cache_len, 8, 128]
K scale:      [num_layers, max_cache_len, 8]
```

The fused Sage partial kernel writes K, V, and K scale once per new token/layer and reads history without expanding GQA in Python. GQA mapping stays inside the CUDA kernel, avoiding explicit 16-query-head by 8-KV-head expansion. Remaining cost is the two-stage partial/reduce structure and the separate o_proj kernel, not Python-side layout copies.

## Current blockers to 560+

```text
现象：30-sample throughput is 543.73 tok/s, below the 560 target.
证据：current profile is dominated by MLP gate_up+SwiGLU (67.830 ms), down_proj (42.551 ms), QKV (30.470 ms), and lm_head stage1 (18.063 ms), while Sage attention itself is already stable and graph-captured.
根因：Sage removed a large attention/cache cost, but the decode cell still launches about six core kernels per layer. MLP and QKV GEMV bandwidth now dominate, and lm_head remains a large per-token tail.
修复：accepted fixes were Sage BF16 o_proj handoff, disabling slower Acext decode lm_head by default, and retuning the Q8 Triton lm_head to effective BLOCK_ROWS=16/BLOCK_K=512.
修复后数据：10/30/150 samples are 544.84 / 543.73 / 544.56 tok/s. This beats 533 but does not reach 560.
```

The next required implementation work is new native fusion rather than more env sweeps:

1. Fuse `down_add_sumsq_q8_bf16_kernel` with the next layer `norm_gemv_sumsq_q8_bf16_kernel` to collapse down_proj/residual/norm/QKV into one native kernel.
2. Fuse Sage attention output, o_proj, residual, and the next RMSNorm where possible, so Sage output does not materialize through a separate o_proj launch.
3. Replace the Q8 lm_head Triton stage1 with a native argmax GEMV kernel specialized for hidden=2048 and vocab layout.
4. Consider a half-layer executor for `RMSNorm -> QKV -> Sage -> o_proj -> residual+RMSNorm`, but this needs new CUDA extension source; the current prebuilt `decode_fused_ops.so` cannot be safely modified in Python.

## Status

This round is a measured improvement over the historical 533 tok/s floor, but it is not a successful 560+ decode-pipeline result. The implementation is cleanly Sage-focused, graph-safe, and has zero Sage fallback in the graph path; the remaining gap requires new CUDA fusion work for MLP/QKV/lm_head rather than more Sage kernel dispatch tuning.
