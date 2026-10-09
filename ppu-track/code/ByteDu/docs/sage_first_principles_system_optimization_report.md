# Sage-first decode system optimization report

## Result status

The current Sage-native decode path is stable but **does not meet the 660 tok/s target**.

| Benchmark | Throughput | TTFT | Artifact |
| --- | ---: | ---: | --- |
| 1 sample baseline | 545.35 tok/s | 42.64 ms | `result_sage_system_baseline_1.json` |
| 10 samples | 544.91 tok/s | 43.08 ms | `result_sage_system_10.json` |
| 30 samples | 545.09 tok/s | 42.90 ms | `result_sage_system_30.json` |
| 150 samples | 544.88 tok/s | 43.31 ms | `result_sage_system_150.json` |

Output smoke check across 10/30/150 found no empty outputs and no obvious repetitive tails. The 10-sample answer heads remain semantically normal for text/number/brand questions.

## Default env used

The measured default is the existing Sage system path:

```bash
JUNKRAT_VLM_BACKEND=fast
JUNKRAT_ENABLE_SAGE_DECODE_ATTN=1
JUNKRAT_SAGE_IMPL=cuda
JUNKRAT_SAGE_QK_PATH=i8
JUNKRAT_SAGE_VALUE_PATH=bf16
JUNKRAT_ENABLE_SAGE_FUSED_QKV_DECODE=1
JUNKRAT_SAGE_LAYERS=all
JUNKRAT_ENABLE_RUNTIME_Q8_TEXT_DECODE=1
JUNKRAT_ENABLE_RUNTIME_Q8_LM_HEAD=1
JUNKRAT_ENABLE_TRITON_LINEAR_ARGMAX=1
JUNKRAT_TRITON_LINEAR_ARGMAX_BLOCK_ROWS=16
JUNKRAT_TRITON_LINEAR_ARGMAX_BLOCK_K=512
JUNKRAT_DISABLE_FUSED_DOWN_QKV=0
JUNKRAT_ENABLE_W8A8_DECODE=0
JUNKRAT_ENABLE_W8A8_LM_HEAD=0
ENABLE_W4A4_DRAFT_SPEC=0
```

## A/B matrix executed this round

All rows are 1-sample throughput probes with normal text output unless noted. They were used to identify net-positive directions before longer runs.

| ID | Config | Throughput | TTFT | Conclusion |
| --- | --- | ---: | ---: | --- |
| A1/W7/F0 | Default Sage K INT8 + V BF16, offline Q8 text/lm_head | 544.90 tok/s | 43.03 ms | Best stable default. |
| A3/A4 | `JUNKRAT_SAGE_VALUE_PATH=int8` | 538.91 tok/s | 42.57 ms | V INT8 is slower; do not default. |
| W8A8-auto | `JUNKRAT_ENABLE_W8A8_DECODE=1`, auto targets | 546.52 tok/s | 42.12 ms | Auto selected no targets; gain is noise, not a real path. |
| W8A8-all | W8A8 qkv/gate/down/o_proj | 493.98 tok/s | 42.72 ms | Activation quant overhead dominates. |
| W8A8-qkv | QKV only | 535.18 tok/s | 42.66 ms | Slower. |
| W8A8-gate | gate/up only | 539.96 tok/s | 42.97 ms | Slower. |
| W8A8-down | down only | 506.82 tok/s | 42.26 ms | Much slower. |
| W8A8-o | o_proj only | 546.70 tok/s | 41.96 ms | Sage BF16 o_proj bypasses this; apparent gain is noise. |
| LM-Acext-int8 | Acext int8 lm_head | 535.09 tok/s | 42.75 ms | Slower than Q8 Triton argmax. |
| LM-Acext-int4pc | Acext int4 lm_head | 508.49 tok/s | 42.44 ms | Slower. |
| Graph-block4 | `JUNKRAT_DECODE_GRAPH_BLOCK_STEPS=4` | 544.40 tok/s | 42.55 ms | No benefit; replay overhead is not the current limiter. |
| BF16-QKV | `JUNKRAT_ENABLE_SAGE_BF16_QKV=1` | 514.95 tok/s | 43.05 ms | Q8 QKV remains required. |
| W4A4 spec | `ENABLE_W4A4_DRAFT_SPEC=1` | 544.43 tok/s | 42.34 ms | Did not improve; speculative path not useful here. |
| EAGLE3 | `AICAS_ENABLE_EAGLE3=1` | 544.32 tok/s | 43.08 ms | Draft checkpoint missing, path falls back. |
| Vision route | `AICAS_VISION_RESIZE_MODE=route`, token opt on | 545.97 tok/s | 42.77 ms | No material throughput change on sample 0. |

## Profiler evidence

Current relevant profiler artifacts:

```text
analysis_outputs/sage_half_layer_ab/current_profile.txt
analysis_outputs/sage_bf16_oproj_validation/sage_new_profiler.txt
analysis_outputs/decode_next_ab/profile/q8lm_sage_profiler.txt
```

The stable profile decomposition remains:

| Stage | Current Sage profile | Interpretation |
| --- | ---: | --- |
| QKV projection | ~30.6 ms | Offline Q8 helps, but this row is still large. |
| Sage attention/cache | ~45.9 ms | Sage is hit and useful; not enough alone. |
| o_proj | ~21.5 ms | BF16 `addmv_bn1` is the best current Sage handoff. |
| MLP gate/up + SwiGLU | ~68.6 ms | Largest row; must be attacked natively. |
| down_proj / fused down | ~45.9 ms | Comparable to attention; W8A8 down was slower. |
| lm_head + argmax | ~20.2 ms | Q8 Triton argmax is faster than native BF16/Acext tested paths. |
| Total CUDA | ~234.6 ms | 660 tok/s requires about <=192 ms for 127-token steady decode. |

Additional lm_head microbenchmark on random 151936x2048 weights:

| lm_head path | Time/call | Conclusion |
| --- | ---: | --- |
| Triton packed Q8 argmax, rows=16, K=512 | ~0.149 ms | Current best. |
| Native BF16 `decode_fused_ops.lm_head_argmax` | ~0.530 ms | Too slow. |

## Root-cause after A/B

The failure is not caused by Sage misses, graph misses, hidden Python fallback, or lack of Q/K INT8. It is a system balance problem:

1. The attention/cache row already improved, but the sum of QKV + o_proj + MLP + down + lm_head is still too high.
2. V INT8, W8A8, Acext lm_head, BF16 QKV, and graph-block replay all failed end-to-end tests, so defaulting them would be worse or only noise.
3. The missing 40+ ms cannot be recovered by env flags. It requires new native ABI work that is not present in the current prebuilt extensions.

## Required next native work

To make the 660 target realistic, the next implementation must change the native executor contract:

1. **o_proj epilogue fusion**: `addmv_bn1(o_proj)` must accumulate `residual += o_proj(attn)` and produce RMS sumsq/rstd for the following MLP without an extra residual scan or launch.
2. **MLP gate/up redesign**: current interleaved Q8 fused norm/gate_up/SwiGLU remains ~68 ms. A new native kernel should reuse the o_proj epilogue rstd, avoid redundant RMS scans per output tile, and preserve packed Q8 scale layout.
3. **down_proj redesign**: current Q8 down+residual+sumsq is still ~46 ms. Existing W8A8 down variants are slower; a better path needs fused activation quant or a more efficient packed-weight layout.
4. **Sage KV ABI v2**: K/scale interleave or GQA-grouped scheduling should be implemented only after MLP/down work, because current attention/cache is not the largest row.
5. **Package-time packing**: move any remaining runtime rowwise Q8 packing into the packaged codebook when possible, but this only affects initialization unless a packed buffer is missing.

## Final decision

No new experimental runtime path was made default because every tested non-default path failed the end-to-end criterion or was measurement noise. The deliverable documents the first-principles model and the negative A/B results; the code package remains on the stable Sage K-INT8/V-BF16 + offline Q8 executor because it is the fastest verified default in this workspace.
