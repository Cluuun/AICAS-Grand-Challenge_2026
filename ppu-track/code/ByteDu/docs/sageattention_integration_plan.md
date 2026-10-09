# SageAttention decode integration plan

## Source audit findings

The upstream `/root/JunkratDev/SageAttention` public APIs are in `sageattention/core.py`: `sageattn`, `sageattn_qk_int8_pv_fp16_triton`, `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp8_cuda_sm90`, and `sageattn_varlen`. They accept FP16/BF16 `q`, `k`, `v` in `HND=[B,H,N,D]` or `NHD=[B,N,H,D]`, require contiguous head dim, support `head_dim <= 128`, and require `num_q_heads % num_kv_heads == 0` for GQA.

SageAttention's core path quantizes Q/K to INT8, stores FP32 Q/K scales, optionally subtracts K mean (`smooth_k`), and computes PV with FP16 or FP8 V depending on backend. Its causal kernels are full-sequence kernels: causal mode assumes `q_len == kv_len`; there is no native `flash_attn_with_kvcache`-style API for token-by-token decode with a growing KV cache.

## Current decode data flow

`benchmark.py` calls `evaluation_wrapper.VLMModel(...).model.generate(..., max_new_tokens=128)`. The fast path is:

1. `FastMinimalQwen3VLModel.generate`
2. `_prefill_logits_with_cache` writes prefix KV into static caches.
3. `_generate_from_prefilled_state` runs steady-state decode, normally through CUDA graph replay.
4. `_decode_one_token_hidden_flash` iterates 28 text layers.
5. Per layer: fused add/norm/QKV projection -> Q/K RMSNorm + RoPE + KV cache write -> `_decode_attention_output` -> o_proj -> fused MLP/down->next QKV.

Decode cache layout is fixed:

- BF16 K/V: `[num_layers, 1, max_cache_len, 8, 128]`
- INT8 K/V: same shape
- scales: `[num_layers, max_cache_len, 8]`
- query scratch: `[16, 128]` and `[1, 1, 16, 128]`

## New integration boundary

The single high-level Sage decode entry is now `sage_decode_attention.py`.

- `sage_decode_attention(...)` handles q_len=1 decode against a static KV cache.
- `SageDecodePolicy(qk_path, value_path)` describes the quantized path.
- `allocate_workspace(...)` returns static split-K workspace compatible with CUDA graph capture.
- `quantize_k_cache_i8(...)` and `quantize_kv_cache_i8(...)` maintain decode cache metadata.
- Stats are explicit via `get_stats()` and `sage_decode_attention_stats.json` when `JUNKRAT_SAGE_DEBUG=1`.

Old wrapper/prefill experiments were removed from root runtime: `sageattention_kvcache.py` and `sageattention_prefill_triton.py`. The remaining `sageattention_kvcache_triton.py` is treated as a low-level kernel module, not a competing dispatch surface.

## Decode insertion points

The decode insertion is in `evaluation_wrapper.py`:

- `_project_prefill_qkv_cache`: for Sage K-I8/BF16-V decode, prefill writes BF16 K plus INT8 K and FP32 K scale so decode does not re-quantize the prefix.
- `_decode_one_token_hidden_flash`: hot per-layer decode can call `_sage_decode_fused_qkv_attention_output` when `JUNKRAT_ENABLE_SAGE_FUSED_QKV_DECODE=1`.
- `_decode_attention_output`: non-fused Sage decode calls go through `sage_decode.sage_decode_attention`.
- `_quantize_kv_cache_layer`: layer policy decides whether disabled layers need full INT8 K/V for the non-Sage fallback.

The preferred path is fused native CUDA:

```text
flat_qkv [1,4096]
  -> sage_decode_qkv_k_i8_v_bf16
       Q/K RMSNorm + RoPE
       K INT8 quant + K scale write
       BF16 V cache write
       GQA decode attention
  -> attn_out [1,1,16,128]
  -> o_proj
```

This avoids per-token transposes, full-history requantization, and repeated KV expansion.

## Env flags

- `JUNKRAT_ENABLE_SAGE_DECODE_ATTN=1|0`: enable decode Sage path.
- `JUNKRAT_SAGE_IMPL=cuda|triton|torch_ref`: backend for adapter fallback.
- `JUNKRAT_SAGE_QK_PATH=i8|bf16|dequant_i8|int8_dot`: QK path.
- `JUNKRAT_SAGE_VALUE_PATH=bf16|int8`: V cache path.
- `JUNKRAT_SAGE_LAYERS=all|0-27|0,1,...`: layer policy for sweeps.
- `JUNKRAT_SAGE_DEBUG=1`: write stats and print first hit per layer.
- `JUNKRAT_SAGE_ASSERT_HIT=1`: fail if requested Sage decode records misses.
- `JUNKRAT_ENABLE_SAGE_FUSED_QKV_DECODE=1|0`: fused QKV/cache/attention path.
- `JUNKRAT_DISABLE_SAGE_NATIVE_CACHE_WRITER=1`: force Triton cache writer fallback.
- `JUNKRAT_DISABLE_SAGE_BF16_O_PROJ=1`: restore the older Q8 o_proj handoff after Sage attention; default uses BF16 `addmv_bn1` because it is faster in the Sage decode path.

Legacy `AICAS_*SAGE*` flags are no longer read by the root runtime.

## Performance assumptions and risks

SageAttention's published kernel gains exclude quantization and smoothing overhead, and upstream does not include KV-cache decode. For Qwen3-VL decode, the useful assumption is not "call upstream SageAttention directly"; it is "keep K quantized and cache-friendly, fuse Q/K norm + RoPE + cache write, avoid KV expansion, and keep graph/static workspace stable."

Expected bottlenecks if throughput does not improve:

- attention is a small fraction of per-token time versus QKV/o_proj/MLP/lm_head;
- graph capture disabled by dynamic allocation or host sync;
- K scale/cache layout causes non-coalesced loads;
- per-layer native extension path not hit because a guard fails;
- partial layer policy leaves non-Sage fallback reading uninitialized INT8 V;
- CUDA native fused path is slower than the custom BF16 split-K kernel for short sequence buckets.

## Optimization plan

1. Validate correctness against baseline token/text output on 1 sample.
2. Compare end-to-end 10-sample throughput with Sage disabled versus enabled.
3. Profile current fused CUDA Sage path.
4. If no gain, run layer sweeps using `JUNKRAT_SAGE_LAYERS`.
5. If layer sweeps show no useful region, inspect kernel timings for attention versus QKV/o_proj/MLP/lm_head and decide whether to fuse o_proj or change KV scale layout next.
