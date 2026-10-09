# SageAttention decode benchmark report

## Scope

This report treats the historical project best, about **533 tok/s**, as the minimum baseline. The new change is Sage-only: it does not combine vision routing, processor token caps, merge-size changes, W8A8 toggles, or other existing optimization switches.

The optimized path is:

```text
benchmark.py
  -> VLMModel.model.generate(..., max_new_tokens=128, use_cache=True)
  -> FastMinimalQwen3VLModel._generate_from_prefilled_state
  -> CUDA graph replay
  -> _decode_one_token_hidden_flash
  -> sage_decode_qkv_k_i8_v_bf16
  -> addmv_bn1 Sage-only o_proj handoff
```

## Sage-only code change

The previous Sage path optimized attention/cache but still fed the attention output into Q8 `addmv_q8` for `o_proj`. That left a hidden handoff bottleneck immediately after SageAttention:

```text
SageAttention output [16,128] -> view(-1) -> Q8 o_proj addmv_q8
```

The current patch keeps the Sage attention/cache kernel unchanged and changes only the Sage-active o_proj handoff:

```text
SageAttention output [16,128] -> view(-1) -> BF16 addmv_bn1
```

This is gated to Sage decode layers only:

```bash
JUNKRAT_DISABLE_SAGE_BF16_O_PROJ=1  # opt out, restores old Q8 o_proj handoff
```

The micro timing that motivated the patch on the real layer-0 o_proj was:

| o_proj handoff | avg kernel time |
|---|---:|
| Q8 `addmv_q8` | 5.81 us |
| BF16 `addmv_bn1` | 4.17 us |

The end-to-end benchmark below is the acceptance signal; the micro timing was only used to locate the bottleneck.

## End-to-end results vs 533 tok/s baseline

Default command:

```bash
./run_fusion_eval.sh 10
./run_fusion_eval.sh 30
./run_fusion_eval.sh 150
```

Default Sage env:

```bash
JUNKRAT_ENABLE_SAGE_DECODE_ATTN=1
JUNKRAT_SAGE_IMPL=cuda
JUNKRAT_SAGE_QK_PATH=i8
JUNKRAT_SAGE_VALUE_PATH=bf16
JUNKRAT_ENABLE_SAGE_FUSED_QKV_DECODE=1
JUNKRAT_SAGE_LAYERS=all
```

| samples | config | TTFT ms | throughput tok/s | vs 533 tok/s | answer failures |
|---:|---|---:|---:|---:|---:|
| 10 | historical best floor | - | 533.00 | baseline | - |
| 10 | previous Sage, Q8 o_proj | 42.63 | 528.44 | -0.86% | 0 |
| 10 | current Sage, BF16 o_proj | 43.06 | 535.61 | +0.49% | 0 |
| 30 | historical best floor | - | 533.00 | baseline | - |
| 30 | current Sage, BF16 o_proj | 43.40 | 535.63 | +0.49% | 0 |
| 150 | historical best floor | - | 533.00 | baseline | - |
| 150 | current Sage, BF16 o_proj | 43.31 | 535.00 | +0.38% | 0 |

Artifacts:

- `result_sage_bf16_oproj_10.json`
- `result_sage_bf16_oproj_30.json`
- `result_sage_bf16_oproj_150.json`
- `analysis_outputs/sage_bf16_oproj_validation/`

## Decode time breakdown

One-sample `torch.profiler` runs were captured with `max_new_tokens=128`. Times are self CUDA over the 127 steady-state decode replays.

| 模块 | 原最优 533 路径 | 当前 Sage 路径 | 差异 | 结论 |
|---|---:|---:|---:|---|
| embed | not visible in steady decode profile | not visible | ~0 | decode graph uses token id scratch; not a bottleneck |
| RMSNorm / fused_add_rms_norm | 1.60 ms | 1.61 ms | +0.01 ms | unchanged |
| QKV projection | 30.79 ms | 30.56 ms | -0.23 ms | unchanged packed Q8 GEMV path |
| Q/K norm + RoPE + quant + cache write | 9.74 ms cache write in non-Sage path | fused inside Sage partial kernel | included below | Sage removes the separate cache-write kernel |
| SageAttention / attention compute | 35.94 ms compute + 15.92 ms reduce | 35.98 ms partial + 9.90 ms reduce | -5.98 ms attention reduce | Sage reduce is faster |
| attention/cache total | 61.59 ms | 45.89 ms | -15.70 ms | Q/K quant + scale write + V cache write is net faster when fused |
| o_proj | 25.06 ms with old Sage Q8 handoff | 21.45 ms BF16 `addmv_bn1` | -3.62 ms | fixed hidden post-Sage bottleneck |
| MLP gate_up + SwiGLU | 68.50 ms | 68.65 ms | +0.15 ms | remaining largest bottleneck, not Sage-specific |
| down_proj / fused down | 46.04 ms | 45.88 ms | -0.16 ms | unchanged |
| lm_head + argmax | 20.16 ms | 20.18 ms | +0.02 ms | unchanged |
| Python loop / host overhead | graph replay loop only | graph replay loop only | no new per-layer Python | Sage kernels are inside captured graph |
| CUDA Graph replay | enabled | enabled | required | Sage is graph-safe and replayed |
| memory allocation / memset / copy | 0.70 ms | 0.70 ms | ~0 | no new large copy/contiguous path |
| total profiled CUDA | 238.01 ms previous Sage | 234.64 ms current Sage | -3.36 ms | o_proj handoff gain converts to end-to-end throughput |

Profiler artifacts:

- `analysis_outputs/sage_bf16_oproj_validation/nosage_profiler.txt`
- `analysis_outputs/sage_bf16_oproj_validation/sage_old_oproj_profiler.txt`
- `analysis_outputs/sage_bf16_oproj_validation/sage_new_profiler.txt`

## CUDA Graph on/off

All runs keep the same non-vision benchmark setup and change only Sage/graph flags.

| config | TTFT ms | throughput tok/s | conclusion |
|---|---:|---:|---|
| Sage + graph on | 42.41 | 535.78 | best path; Sage kernels captured and replayed |
| Sage + graph off | 42.29 | 496.35 | Python/layer launch overhead dominates without graph |
| Flash/current non-Sage + graph on | 42.47 | 494.33 | graph works, but attention/cache path is slower |
| Flash/current non-Sage + graph off | 42.33 | 196.19 | non-graph executor is not competitive |

Sage hit stats for graph-on validation:

```json
{
  "decode_calls": 28644,
  "cuda_calls": 28644,
  "graph_decode_replay_calls": 28644,
  "miss_count": 0,
  "miss_reasons": {},
  "triton_calls": 0,
  "torch_ref_calls": 0
}
```

The one-generation profiler reports the exact 128-token hot path:

```json
{
  "decode_calls": 3556,
  "cuda_calls": 3556,
  "graph_decode_replay_calls": 3556,
  "miss_count": 0,
  "miss_reasons": {}
}
```

`3556 = 127 decode steps * 28 layers`.

## Layer policy sweep

Partial Sage is intentionally treated as a designed mixed policy, not a silent fallback. It is slower because disabled layers must use the non-Sage INT8 fallback cache path, while enabled layers still pay mixed-cache bookkeeping.

| `JUNKRAT_SAGE_LAYERS` | TTFT ms | throughput tok/s | conclusion |
|---|---:|---:|---|
| `all` | 42.65 | 535.70 | best |
| `0-13` | 42.34 | 508.99 | worse |
| `14-27` | 42.50 | 508.70 | worse |
| even layers | 42.38 | 494.70 | worse |
| odd layers | 42.18 | 494.28 | worse |

Final policy: enable Sage on all 28 text decoder layers.

## KV / scale layout audit

The active Sage cache layout is:

```text
K int8 cache:   [num_layers, 1, max_cache_len, 8, 128]
V bf16 cache:   [num_layers, 1, max_cache_len, 8, 128]
K scale cache:  [num_layers, max_cache_len, 8]
query/output:   [16,128] / [1,1,16,128]
```

For a fixed layer, the kernel receives contiguous `[token, kv_head, head_dim]` slices. K head_dim is contiguous, scale is contiguous per token across the 8 KV heads, and V remains BF16 to avoid the previously slower V-int8 path. GQA is handled inside the CUDA Sage kernel without expanding KV to 16 heads in Python.

The profiler shows no extra transpose/contiguous penalty: `aten::copy_` is 0.176 ms over the profiled generation, and DtoD memcpy is 0.009 ms.

## Answers to the blocking questions

1. The first Sage version did not exceed 533 because attention/cache saved 15.7 ms, but the immediately following Q8 o_proj handoff still cost about 25 ms and swallowed enough of the gain.
2. The largest remaining decode bottleneck is MLP gate_up/SwiGLU at about 68.65 ms, followed by down_proj/fused down at about 45.88 ms and QKV projection at about 30.56 ms.
3. Sage did not add hidden copy/allocation overhead; copy/memcpy stayed under 1 ms. The hidden overhead was the unchanged post-Sage Q8 o_proj kernel.
4. Q/K quant, K scale write, and BF16 V cache write are net better only when fused into `sage_decode_qkv_k_i8_v_bf16`; non-fused and V-int8 variants are slower.
5. GQA does not expand KV in Python; the fused CUDA kernel maps 16 Q heads to 8 KV heads internally.
6. `attn_out.view(-1)` is a view of the Sage workspace output. The fixed issue was not a contiguous copy; it was the slower Q8 o_proj handoff.
7. CUDA Graph remains active. Sage graph-on reaches 535.78 tok/s; graph-off falls to 496.35 tok/s.
8. There are still many per-layer kernels inside the graph, but no per-layer Python dispatch during replay.
9. Yes, Sage needed executor-level fusion past attention; the BF16 o_proj handoff is the first post-attention executor fusion.
10. No non-Sage throughput switch is required for the current result.
11. Sage enters the captured graph and the packed decode path; hit stats show zero fallback.
12. Sage keeps K int8 / V bf16 side caches and does not slow other modules measurably.

## Required profiler conclusion

现象：
The initial Sage path improved attention/cache but stayed below the 533 tok/s historical target. After fixing the Sage-to-o_proj handoff, 30-sample throughput is 535.63 tok/s and 150-sample throughput is 535.00 tok/s.

证据：
Previous Sage with Q8 o_proj reached 528.44 tok/s on the 10-sample validation rerun. Current Sage with BF16 `addmv_bn1` reaches 535.61 tok/s on 10 samples, 535.63 tok/s on 30 samples, and 535.00 tok/s on 150 samples. Profiler stats show 3556/3556 steady-state calls hit CUDA Sage with `miss_count=0`.

根因：
The first implementation stopped at attention/cache. It removed the separate QKV cache-write kernel and reduced the attention reduce stage, but then read Sage's BF16 output through the Q8 o_proj path. For this 2048x2048 batch-1 GEMV, the existing BF16 `addmv_bn1` kernel is faster than the Q8 o_proj kernel and avoids quantized handoff overhead.

修复：
When a layer is Sage-active and `packed.o_proj_weight_t` is available, route `attn_out.view(-1)` directly to `decode_fused_ops.addmv_bn1(...)`. Keep the old Q8 path as `JUNKRAT_DISABLE_SAGE_BF16_O_PROJ=1` for ablation.

修复后数据：
Attention/cache remains about 45.89 ms, o_proj drops from 25.06 ms to 21.45 ms, total profiled CUDA drops from 238.01 ms to 234.64 ms, and the official benchmark path exceeds the 533 tok/s floor on 10/30/150 samples.
