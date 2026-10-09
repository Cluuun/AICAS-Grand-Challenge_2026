# mmproj W8A8 Progress Summary

## Scope

This work is strictly scoped to:
- mmproj only
- `mul_mat`-backed linear layers only
- static W8A8 reference path in `llama.cpp`
- SmolVLM2 IDEFICS3 mmproj only

Out of scope:
- text model
- non-`mul_mat` operators
- dynamic quantization
- NPU backend integration

## Target Model

- Text model: `AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf`
- Base mmproj: `AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf`
- Runtime integration point: [`tools/mtmd/clip.cpp`](/home/gugugu/work/llama.cpp-kv260-20260407/tools/mtmd/clip.cpp)

## Current Result

As of 2026-04-10:
- Full 73-layer W8A8 mmproj: `26 / 100`
- Mixed policy `mixed_v1`: `54 / 100`
- Requirement `>= 48 / 100` is met by `mixed_v1`

Current recommended mmproj artifact:
- [`AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf)

## Design Summary

### Quantization scheme

Activation:
- `uint8`
- static asymmetric quantization
- per-layer `act_scale`
- per-layer `act_zero_point`

Weight:
- `int8`
- static symmetric quantization
- per-output-channel `weight_scale[j]`
- no weight zero-point

Integer MAC model:
- runtime uses `int8 x int8 -> int32`
- activation is converted from `uint8` to signed domain by `A_i8 = A_u8 - 128`
- correction term: `(128 - Z_a) * SumW[j]`

### Runtime policy

- only whitelisted mmproj linear layers use the custom W8A8 path
- non-target layers stay on the original path
- if AICAS W8A8 metadata is detected, mmproj is forced to CPU backend

## Layer Inventory

Target layer count: 73

Breakdown:
- `v.blk.{0..11}.attn_q.weight` x12
- `v.blk.{0..11}.attn_k.weight` x12
- `v.blk.{0..11}.attn_v.weight` x12
- `v.blk.{0..11}.attn_out.weight` x12
- `v.blk.{0..11}.ffn_up.weight` x12
- `v.blk.{0..11}.ffn_down.weight` x12
- `mm.model.fc.weight` x1

Manifest:
- [`AICAS/artifacts/layer_manifest.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/layer_manifest.json)
- [`AICAS/artifacts/mmproj_mulmat_inventory.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/mmproj_mulmat_inventory.json)

## Implemented Components

### Runtime

Modified file:
- [`tools/mtmd/clip.cpp`](/home/gugugu/work/llama.cpp-kv260-20260407/tools/mtmd/clip.cpp)

Implemented:
- parsing `aicas.w8a8.*` metadata
- `build_mmproj_linear()` dispatch point for target linear layers
- CPU reference W8A8 kernel via `ggml_map_custom3`
- activation sampling hook via `ggml_map_custom1`
- CPU-only enforcement for mmproj W8A8 path

### Scripts

Implemented scripts:
- [`AICAS/tools/mmproj_manifest.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_manifest.py)
- [`AICAS/tools/mmproj_make_calib_manifest.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_make_calib_manifest.py)
- [`AICAS/tools/mmproj_collect_act_stats.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_collect_act_stats.py)
- [`AICAS/tools/mmproj_calibrate.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_calibrate.py)
- [`AICAS/tools/mmproj_pack_gguf.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_pack_gguf.py)
- [`AICAS/tools/mmproj_make_layer_policy.py`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/tools/mmproj_make_layer_policy.py)

## Artifact Summary

### Calibration and policy

- Calibration image manifest:
  - [`AICAS/calib/calib_manifest.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/calib/calib_manifest.json)
- Activation stats:
  - [`AICAS/artifacts/mmproj_act_stats.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/mmproj_act_stats.json)
- Calibration params:
  - [`AICAS/artifacts/quant_params.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/quant_params.json)
- Mixed policy v1:
  - [`AICAS/artifacts/layer_policy.v1.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/layer_policy.v1.json)
- Pack summary v1:
  - [`AICAS/artifacts/pack_summary.v1.json`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/artifacts/pack_summary.v1.json)

### GGUF outputs

- Full W8A8:
  - [`AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf)
- Mixed v1:
  - [`AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf)

### Evaluation outputs

- Latest full-W8A8 local eval output:
  - [`AICAS/output/local-acc-eval`](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS/output/local-acc-eval)
- Latest mixed-v1 local eval output:
  - `/tmp/local-acc-eval-w8a8-mixed-v1`

## Reproducible Workflow

### 1. Build layer manifest

```bash
python3 AICAS/tools/mmproj_manifest.py \
  --mmproj-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --profile-json /tmp/mmproj_profile.json \
  --output-manifest AICAS/artifacts/layer_manifest.json \
  --output-inventory AICAS/artifacts/mmproj_mulmat_inventory.json
```

### 2. Build calibration image manifest

```bash
python3 AICAS/tools/mmproj_make_calib_manifest.py \
  --data-root AICAS/data \
  --eval-json AICAS/sampled.json \
  --output AICAS/calib/calib_manifest.json \
  --num-images 256
```

### 3. Collect activation stats

Requires rebuilt `llama-mtmd-profiler` with the activation sampling hook.

```bash
python3 AICAS/tools/mmproj_collect_act_stats.py \
  --profiler-bin build-host/bin/llama-mtmd-profiler \
  --model-gguf AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
  --mmproj-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --calib-manifest AICAS/calib/calib_manifest.json \
  --output AICAS/artifacts/mmproj_act_stats.json \
  --limit 64 \
  --samples-per-tensor 4096
```

### 4. Generate calibration params

```bash
python3 AICAS/tools/mmproj_calibrate.py \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --act-stats AICAS/artifacts/mmproj_act_stats.json \
  --calib-manifest AICAS/calib/calib_manifest.json \
  --output AICAS/artifacts/quant_params.json \
  --percentile-low 0.1 \
  --percentile-high 99.9
```

### 5. Pack full W8A8 mmproj

Note:
- `--quant-params` and `--output-quant-params` must not be the same path
- otherwise the calibration file will be overwritten by the pack summary

```bash
python3 AICAS/tools/mmproj_pack_gguf.py \
  --input-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --quant-params AICAS/artifacts/quant_params.json \
  --output-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf \
  --output-quant-params AICAS/artifacts/pack_summary.full.json \
  --mode quantize
```

### 6. Build mixed policy v1

Policy v1 fallback:
- all `ffn_up`
- all `attn_out`
- `mm.model.fc.weight`

```bash
python3 AICAS/tools/mmproj_make_layer_policy.py \
  --input-quant-params AICAS/artifacts/quant_params.json \
  --output AICAS/artifacts/layer_policy.v1.json \
  --fallback-kind ffn_up \
  --fallback-kind attn_out \
  --fallback-tensor mm.model.fc.weight
```

### 7. Pack mixed v1 mmproj

```bash
python3 AICAS/tools/mmproj_pack_gguf.py \
  --input-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --quant-params AICAS/artifacts/layer_policy.v1.json \
  --output-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
  --output-quant-params AICAS/artifacts/pack_summary.v1.json \
  --mode quantize
```

### 8. Run local accuracy evaluation

```bash
bash AICAS/scripts/run_local_acc_eval.sh \
  --model f16 \
  --mmproj AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
  --output-folder /tmp/local-acc-eval-w8a8-mixed-v1 \
  --save-name mixed_v1
```

## Accuracy Results

### Full 73-layer W8A8

Model:
- `AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf`

Result:
- Final Score: `26 / 100`

Interpretation:
- full enablement is not acceptable
- sensitive layers must be statically reverted to F16

### Mixed v1

Model:
- `AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`

Result:
- Text Recognition: `37 / 60`
- Scene Text-centric VQA: `17 / 40`
- Final Score: `54 / 100`

Interpretation:
- meets the current `>= 48 / 100` target
- `mixed_v1` is the current recommended baseline

## Mixed v1 Layer Split

### Quantized in W8A8

48 tensors:
- `v.blk.{0..11}.attn_q.weight`
- `v.blk.{0..11}.attn_k.weight`
- `v.blk.{0..11}.attn_v.weight`
- `v.blk.{0..11}.ffn_down.weight`

### Kept in F16 fallback

25 tensors:
- `mm.model.fc.weight`
- `v.blk.{0..11}.attn_out.weight`
- `v.blk.{0..11}.ffn_up.weight`

## Calibration Observations

Observed from `AICAS/artifacts/quant_params.json`:
- all 73 target layers were calibrated successfully
- `ffn_up` layers had the highest clipping ratios
- `attn_q / attn_k / attn_v / attn_out / ffn_down` were much more stable

Representative examples:
- `mm.model.fc.weight`: `scale=0.054924663842893115`, `zp=163`, `clip=0.002685546875`
- `v.blk.0.ffn_up.weight`: `scale=0.00793443614361333`, `zp=21`, `clip=0.04150390625`
- `v.blk.4.ffn_up.weight`: `clip=0.05810546875`
- `v.blk.6.ffn_up.weight`: `clip=0.052734375`

Interpretation:
- projector and especially `ffn_up` are more sensitive
- the current mixed policy aligns with the observed clipping risk

## Important Notes

- Weight quantization is symmetric `int8`, not asymmetric.
- `act_zero_point` belongs to the activation feeding a layer, not to the weight tensor named in that entry.
- The current reference path is correctness-oriented; it is not optimized for throughput.
- `build-host/bin/llama-mtmd-profiler` must be rebuilt after `clip.cpp` changes before running activation-stat collection.

## Recommended Next Step

Current baseline to keep:
- `mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`

If accuracy needs to go higher than `54 / 100`, next trial should be:
1. keep `mixed_v1`
2. additionally fallback all `attn_v`
3. re-evaluate accuracy and cost
