# Native Sage split-K decode attention report

## Current path audit

The current default 128-token decode path is already native Sage through the prebuilt `decode_fused_ops.so`:

```text
benchmark.py
  -> evaluation_wrapper.VLMModel
  -> FastMinimalQwen3VLModel._generate_from_prefilled_state
  -> CUDA graph replay
  -> _decode_one_token_hidden_flash
  -> _sage_decode_fused_qkv_attention_output
  -> decode_fused_ops.sage_decode_qkv_k_i8_v_bf16
  -> decode_fused_ops.addmv_bn1
```

Fallback paths are still present:

```text
non-Sage INT8 KV fallback: custom_decode_attention_module.decode_attention_i8_4d
Triton Sage fallback:      sageattention_kvcache_triton.decode_attention_k_i8_v_bf16_4d
QKV/cache writer fallback: triton_decode_packed_qkv_cache.decode_packed_qkv_cache_k_i8_v_bf16_inplace
```

The active KV cache layout is unchanged because it already supports coalesced per-head tile reads:

```text
K INT8 cache: [num_layers, 1, max_cache_len, 8, 128]
V BF16 cache: [num_layers, 1, max_cache_len, 8, 128]
K scale:      [num_layers, max_cache_len, 8]
Q/output:     [16,128] / [1,1,16,128]
```

For a fixed layer, native kernels receive contiguous `[token, kv_head, head_dim]` slices. `head_dim` is contiguous, scale is one FP32 value per token/head, and GQA maps two query heads to one KV head.

## Why the single-block native path failed

The first source implementation used one CUDA block per KV head and serially scanned the entire cache. That removed sequence parallelism, so it only helped at very short cache lengths and collapsed on real prompts.

Evidence from the rejected forced run:

| config | TTFT ms | throughput tok/s | conclusion |
|---|---:|---:|---|
| forced native single-block before guard | 43.09 | 48.26 | rejected; serial cache scan is not viable |

The single-block kernels remain exported only as debug/reference functions:

```text
sage_decode_k_i8_v_bf16_single
sage_decode_q_i8_k_i8_v_bf16_single
sage_decode_qkv_k_i8_v_bf16_single
```

They are not used by default.

## New split-K native backend

Added source-buildable split-K kernels in:

```text
native_sage_decode_ext.cpp
native_sage_decode_kernels.cu
build_native_sage_decode_ext.py
scripts/native_sage_decode_microbench.py
```

New split-K exports:

```text
sage_decode_k_i8_v_bf16_split
sage_decode_q_i8_k_i8_v_bf16_split
sage_decode_qkv_k_i8_v_bf16_split
sage_decode_qkv_prequant_k_i8_v_bf16_split
sage_decode_qkv_prequant_k_i8_v_bf16_split_bf16po
sage_decode_qkv_prequant_k_i8_v_bf16_split128
```

### Block/grid design

Stage 1 uses a split-sequence grid:

```text
grid.x = kv_head        # 0..7
grid.y = split_id       # ceil(cache_len / 64)
blockDim.x = 128        # one lane per head_dim element
```

Each block handles one `(kv_head, split_id)` tile:

```text
tile = cache tokens [split_id * 64, min((split_id + 1) * 64, cache_len))
q_head0 = kv_head * 2
q_head1 = q_head0 + 1
```

It loads one K tile and one V tile for the KV head, computes both GQA query heads against the same loaded K/V tile, and writes partial state:

```text
partial_max: [16, max_splits]
partial_sum: [16, max_splits]
partial_out: [16, max_splits, 128]
```

Stage 2 merges partial softmax states per query head:

```text
global_m = max(partial_max)
denom = sum(exp(partial_max - global_m) * partial_sum)
out[d] = sum(exp(partial_max - global_m) * partial_out[d]) / denom
```

The original fused QKV split-K variant also performed current-token Q/K RMSNorm, RoPE, K quantization, K scale write, and BF16 V cache write inside each `(kv_head, split)` block. That was correct but repeated Q/K preprocessing once per split. The current opt-in native path uses the newer `sage_decode_qkv_prequant_k_i8_v_bf16_split` pipeline:

```text
qkv_prequant_cache_k_i8_v_bf16_kernel
  -> compute Q/K RMSNorm + RoPE once
  -> write Q INT8 + Q scale scratch
  -> write current K INT8, K scale, and V BF16 cache once
decode_q_i8_k_i8_v_bf16_split_partial_opt_kernel
  -> split-K INT8 Q/K attention, online softmax partial, V accumulate
decode_split_reduce_kernel
  -> merge partial softmax state and write [1,1,16,128] output
```

This keeps the existing K/V cache layout unchanged and avoids the repeated preprocess work that made the first fused source split-K path much slower in CUDA Graph replay.

The latest source partial kernel also parallelizes the per-tile softmax. The previous implementation computed tile max, exponentials, and sums for both GQA query heads on thread 0 after the `dp4a` dot pass. The current kernel performs max/exp/sum across the block, then reuses the shared score weights for the V accumulation. This keeps the same split-K partial state but removes a serial exp loop from every `(kv_head, split_id)` block.

### GQA reuse

GQA reuse is explicit: each `(kv_head, split_id)` block computes both query heads sharing that KV head. The K/V tile is loaded once per `kv_head/split`, then two dot products are reduced together with a pair reduction. This avoids expanding KV to 16 heads in memory and halves K/V tile loads relative to one block per query head.

## Microbench results

Commands:

```bash
python build_native_sage_decode_ext.py
for s in 64 128 256 512 1024 2048 4096; do
  python scripts/native_sage_decode_microbench.py --seq-len "$s" --warmup 5 --iters 20 --include-qkv
done
```

The latest source split-K kernels use tile-score partials and `dp4a` for the QI8/KI8 dot path. The standalone rows include cache read + INT8 K scale/dequant + online softmax + V accumulate + split reduce. They do not include QKV projection, Q/K norm, RoPE, or cache write.

Standalone attention latency:

| cache_len | Triton Sage ms | current default CUDA ms | native split-K BF16-Q ms | native split-K QI8 ms |
|---:|---:|---:|---:|---:|
| 64 | 0.0696 | 0.0080 | 0.0246 | 0.0175 |
| 128 | 0.0701 | 0.0078 | 0.0255 | 0.0177 |
| 256 | 0.0683 | 0.0081 | 0.0227 | 0.0177 |
| 512 | 0.0688 | 0.0081 | 0.0243 | 0.0180 |
| 1024 | 0.0687 | 0.0111 | 0.0288 | 0.0205 |
| 2048 | 0.0721 | 0.0153 | 0.0251 | 0.0202 |
| 4096 | 0.0716 | 0.0254 | 0.0271 | 0.0234 |

28-layer standalone attention loop:

| cache_len | Triton Sage ms | current default CUDA ms | native split-K BF16-Q ms | native split-K QI8 ms |
|---:|---:|---:|---:|---:|
| 64 | 1.6708 | 0.2969 | 0.5182 | 0.4894 |
| 128 | 1.6391 | 0.2930 | 0.5214 | 0.4921 |
| 256 | 1.6500 | 0.2181 | 0.5267 | 0.4947 |
| 512 | 1.6605 | 0.2953 | 0.5262 | 0.4959 |
| 1024 | 1.6707 | 0.3135 | 0.6080 | 0.5668 |
| 2048 | 1.6782 | 0.4289 | 0.6370 | 0.5678 |
| 4096 | 1.6847 | 0.5406 | 0.7584 | 0.6557 |

The fused QKV/cache-write rows include current-token Q/K RMSNorm, RoPE, K INT8 quantization, K scale write, BF16 V cache write, cache read, INT8 QK, online softmax, V accumulate, and split reduce. They start from already-projected packed QKV; QKV projection itself is not included in any compared kernel.

| cache_len | Triton QKV + Triton attention ms | current default QKV CUDA ms | native fused QKV split-K ms | debug single ms |
|---:|---:|---:|---:|---:|
| 64 | 0.4829 | 0.0200 | 0.0312 | 0.0396 |
| 128 | 0.1177 | 0.0101 | 0.0318 | 0.0774 |
| 256 | 0.1202 | 0.0103 | 0.0308 | 0.1509 |
| 512 | 0.1196 | 0.0103 | 0.0320 | 0.3018 |
| 1024 | 0.1225 | 0.0138 | 0.0378 | 0.5956 |
| 2048 | 0.1226 | 0.0187 | 0.0358 | 1.1982 |
| 4096 | 0.1961 | 0.0231 | 0.0366 | 2.3687 |

Correctness highlights:

| check | result |
|---|---:|
| standalone native split-K BF16-Q max abs diff vs Triton | 0 to 0.000122 |
| standalone native split-K QI8 max abs diff vs Triton | 0.000976 to 0.005859 |
| fused QKV native split-K max abs diff vs Triton | 0.000977 to 0.011719 |
| current-token K INT8 cache match | true |
| current-token V max abs diff | 0.0 |
| current-token K scale max abs diff | <= 1.49e-8 |

### 2026-05-28 fused-prequant update

The microbench gap did not explain the first 545 -> 286 tok/s end-to-end regression because that microbench used `max_splits = ceil(seq_len / 64)`, while CUDA Graph replay captures the decode workspace bucket. For a 1024-token bucket, the old source fused split-K partial launched 16 splits and recomputed Q/K RMSNorm, RoPE, Q quantization, K quantization, and V cache write in every split block. The preprocess-once path fixes that root cause.

Updated fused QKV/cache-write + attention latency:

| cache_len | default prebuilt ms | old native fused split-K ms | native preprocess-once split-K ms |
|---:|---:|---:|---:|
| 64 | 0.0101 | 0.0314 | 0.0211 |
| 128 | 0.0100 | 0.0319 | 0.0212 |
| 256 | 0.0102 | 0.0309 | 0.0214 |
| 512 | 0.0103 | 0.0311 | 0.0215 |
| 1024 | 0.0138 | 0.0390 | 0.0240 |
| 2048 | 0.0187 | 0.0348 | 0.0240 |
| 4096 | 0.0232 | 0.0358 | 0.0270 |

The one-Q-head-per-block variant improved some short-cache microbench rows but regressed 4096 and end-to-end, so the active path remains the GQA two-query-head block mapping.

### 2026-05-28 parallel tile-softmax update

Command:

```bash
python build_native_sage_decode_ext.py
cp native_sage_decode_ext.so prebuilt/native_sage_decode_ext.so
for s in 64 128 256 512 1024 2048 4096; do
  python scripts/native_sage_decode_microbench.py --seq-len "$s" --warmup 30 --iters 120 --include-qkv
done
```

Fused QKV/cache-write + attention latency after parallelizing tile softmax:

| cache_len | default prebuilt ms | default decomposed ms | native prequant split ms | native BF16 partial_out ms |
|---:|---:|---:|---:|---:|
| 64 | 0.0091 | 0.0093 | 0.0127 | 0.0128 |
| 128 | 0.0094 | 0.0093 | 0.0126 | 0.0126 |
| 256 | 0.0190 | 0.0246 | 0.0211 | 0.0183 |
| 512 | 0.0092 | 0.0096 | 0.0130 | 0.0130 |
| 1024 | 0.0126 | 0.0128 | 0.0135 | 0.0134 |
| 2048 | 0.0177 | 0.0167 | 0.0153 | 0.0153 |
| 4096 | 0.0221 | 0.0208 | 0.0207 | 0.0205 |

This is a real source-kernel fix: the prequant split path improved from about 0.0215 ms to 0.0130 ms at cache_len 512 and from about 0.0271 ms to 0.0207 ms at cache_len 4096. It is still slower than default in the benchmark's real cache range around 588-716 tokens, but it is no longer the 2x regression seen before.

Two follow-up experiments were kept opt-in/debug only:

| experiment | control | result | decision |
|---|---|---|---|
| BF16 `partial_out` | `JUNKRAT_NATIVE_SAGE_BF16_PARTIAL_OUT=1` | small microbench gain, 30-sample throughput 512.00 tok/s | not default; partial_out bandwidth is not the dominant gap |
| 128-token KV tile | `JUNKRAT_NATIVE_SAGE_TILE128=1` | faster only at 4096, slower at 512/1024 | not default; real benchmark cache range prefers 64-token tiles |
| shared reduce weights | reverted | reduce time increased and 30-sample throughput dropped to 508.80 tok/s | removed from final source |

Profiling tools from the Aliyun PPU SDK documentation:

| tool | use in this task |
|---|---|
| Asight Systems (`asys`) | CUDA Graph/API timeline, graph replay count, kernel instance counts |
| Asight Compute | kernel-level metric tool; available in SDK docs, not required for the final decision because end-to-end still regressed |
| PPU SMI | device/driver visibility check |
| PyTorch profiler | quick kernel/API counts inside repeatable scripts |

Graph/profile summary for a 128-token generate on sample 0:

| config | CUDA Graph | graph launches | decode attention kernels/layer/token | alloc/sync profile | notable kernels |
|---|---|---:|---:|---|---|
| default | enabled | 127 | 2 | same `cudaMalloc`, `cudaFree`, `cudaStreamSynchronize` profile as native | `sage_decode_qkv_k_i8_v_bf16_partial_kernel`, `sage_decode_reduce_kernel` |
| native preprocess-once | enabled | 127 | 3 | no extra per-token allocation or device sync vs default; same captured graph path | `qkv_prequant_cache_k_i8_v_bf16_kernel`, `decode_q_i8_k_i8_v_bf16_split_partial_opt_kernel`, `decode_split_reduce_kernel` |

Asight/PyTorch profiling showed the native preprocess-once path preserves CUDA Graph replay but adds one captured kernel per layer/token. After parallel tile softmax, native partial time was about 38.1 ms total across the profiled run, reduce about 11.0 ms, and preprocess about 10.3 ms. Default prebuilt fused partial was about 34.9 ms and reduce about 9.2 ms. This confirms the remaining gap is source partial/reduce efficiency plus the extra preprocess boundary, not graph disablement, CPU synchronization, or allocator churn.

## End-to-end results

Commands:

```bash
./run_fusion_eval.sh 1
./run_fusion_eval.sh 10
./run_fusion_eval.sh 30
JUNKRAT_ENABLE_NATIVE_SAGE_SPLITK=1 ./run_fusion_eval.sh 1
JUNKRAT_ENABLE_NATIVE_SAGE_SPLITK=1 ./run_fusion_eval.sh 10
JUNKRAT_ENABLE_NATIVE_SAGE_SPLITK=1 ./run_fusion_eval.sh 30
```

| config | TTFT ms | throughput tok/s | 1024-answer path | conclusion |
|---|---:|---:|---|---|
| default, 1 sample | 42.61 | 543.99 | non-empty answer | unchanged stable path |
| default, 10 samples | 42.71 | 544.81 | 10/10 non-empty answers | unchanged stable path |
| default, 30 samples | 43.18 | 544.13 | 30/30 non-empty answers | unchanged stable path |
| opt-in native split-K old fused, 1 sample | 43.36 | 286.72 | non-empty answer | rejected old path |
| opt-in native preprocess-once split-K, 1 sample | 42.38 | 452.40 | non-empty answer | improved but still slower |
| opt-in native preprocess-once split-K, 30 samples | 43.01 | 453.09 | 30/30 non-empty answers | smoke passes but throughput fails |
| opt-in native parallel-softmax split-K, 1 sample | 42.89 | 510.09 | non-empty answer | latest source improvement |
| opt-in native parallel-softmax split-K, 30 samples | 43.51 | 510.63 | 30/30 non-empty answers | still below default |
| opt-in native parallel-softmax + BF16 partial_out, 30 samples | 43.07 | 512.00 | 30/30 non-empty answers | best native source run, still below default |

## Integration decision

The split-K backend is controlled by:

```bash
JUNKRAT_ENABLE_NATIVE_SAGE_SPLITK=1
```

The single-block debug backend is controlled separately:

```bash
JUNKRAT_ENABLE_NATIVE_SAGE_SINGLE=1
JUNKRAT_NATIVE_SAGE_DECODE_MAX_CACHE_LEN=64
```

Additional experimental knobs:

```bash
JUNKRAT_NATIVE_SAGE_BF16_PARTIAL_OUT=1
JUNKRAT_NATIVE_SAGE_TILE128=1
```

Both are disabled by default. The default path remains `decode_fused_ops.sage_decode_qkv_k_i8_v_bf16` because the new source split-K implementation beats Triton but not the current prebuilt native default.

## Conclusion

The new implementation fixes the single-block design error: it preserves split-K sequence parallelism, writes partial online-softmax state, merges partials correctly, keeps the existing coalesced `[token, kv_head, head_dim]` KV layout, and explicitly reuses K/V tiles for the two GQA query heads.

It does **not** meet the default-enablement bar. The preprocess-once update fixed the main implementation bug in the source fused path and improved forced throughput from about 286 tok/s to about 453 tok/s. The parallel tile-softmax update further improved it to about 510-512 tok/s, with TTFT unchanged and non-empty 1024-token answers. However, the current prebuilt default still reaches about 544 tok/s. Profiling shows CUDA Graph replay remains enabled and allocator/sync behavior is not the blocker; the remaining gap is source partial/reduce efficiency versus the prebuilt fused kernel and the extra preprocess boundary. Until the native path beats default end-to-end, it remains opt-in only.
