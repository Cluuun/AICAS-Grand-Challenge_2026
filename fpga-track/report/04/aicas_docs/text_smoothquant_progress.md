# Text SmoothQuant Progress

## Scope

- Text-side SmoothQuant PTQ for `mulmat / GEMM` only
- Decode-side `GEMV` is not quantized
- FlashAttention kernels are not modified
- Activation quantization: static asymmetric per-tensor `u8`
- Weight quantization: tested both symmetric `int8` per-channel and per-tensor
- Smoothing factor `s` is applied per input channel

## Candidate Layers

- Manifest: `AICAS/artifacts/text_sq_manifest.json`
- Manifest summary: `AICAS/artifacts/text_sq_manifest_summary.json`
- Total 2D text linear candidates: `225`
- Enabled prefill GEMM candidates: `224`
- Default skip: `output.weight`

Observed runtime activation stats still miss 3 tensors in the last block:

- `blk.31.ffn_down.weight`
- `blk.31.ffn_gate.weight`
- `blk.31.ffn_up.weight`

For this round, candidate preparation uses `same_kind_prev_layer` backfill so that policy generation reaches `224/224`. The backfilled tensors are marked in the candidate file via `stat_source`.

## Main Files

- Text act-stat collector:
  `AICAS/tools/text_smoothquant_collect_act_stats.py`
- Text candidate preparation:
  `AICAS/tools/text_smoothquant_prepare.py`
- Text policy generation:
  `AICAS/tools/text_smoothquant_make_policy.py`
- Text GGUF packer:
  `AICAS/tools/text_smoothquant_pack_gguf.py`
- Text experiment driver:
  `AICAS/tools/text_smoothquant_experiment.py`

## Results

### Calib 32 Pilot24

- Summary:
  `AICAS/output/text-smoothquant-pilot/f16-calib32/summary.json`
- Candidate file:
  `AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/candidates.json`
- Ready layers: `224`
- Backfilled layers: `3`
- Missing layers after backfill: `0`

Pilot24 results:

- `alpha=0.4, per-channel`: `16/24`
- `alpha=0.4, per-tensor`: `16/24`
- `alpha=0.5, per-channel`: `16/24`
- `alpha=0.5, per-tensor`: `17/24`
- `alpha=0.6, per-channel`: `17/24`
- `alpha=0.6, per-tensor`: `14/24`

### Calib 64 Pilot24

- Raw act stats:
  `AICAS/output/text-smoothquant-pilot/f16-calib64/raw_act_stats.json`
- Candidate file:
  `AICAS/output/text-smoothquant-pilot/f16-calib64/candidates.json`
- Ready layers: `224`
- Backfilled layers: `3`
- Missing layers after backfill: `0`

Checked targeted configs:

- `alpha=0.5, per-tensor`: `16/24`
- `alpha=0.6, per-channel`: `17/24`

### Full 100 Eval

Selected full-eval config:

- text base: `F16`
- calibration size: `32`
- `alpha=0.5`
- weight quantization: `per-tensor`
- activation quantization: static asymmetric per-tensor
- clip mode: `minmax`

Artifacts:

- Model:
  `AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/a0_5-minmax-per_tensor/model.gguf`
- Full eval result:
  `AICAS/output/text-smoothquant-full/f16-calib32-a05-pt/text_sq_f16_calib32_a05_pt.json`
- Full eval server log:
  `AICAS/output/text-smoothquant-full/f16-calib32-a05-pt/server.log`

Result:

- `52/100`
- User-provided FP16 baseline: `53/100`
- Accuracy delta: `-1`

## Q8_0 Refresh

User requirement for final deployment baseline:

- text base must use official `Q8_0`
- if new SmoothQuant variants on top of `Q8_0` are not good enough, keep the best previously verified `Q8_0` result instead of switching to `FP16`

### Previously Verified Q8_0 Text SQ

- Model:
  `AICAS/output/text-smoothquant/q8base-a05/model.gguf`
- Pack summary:
  `AICAS/output/text-smoothquant/q8base-a05/pack_summary.json`
- Full eval:
  `AICAS/output/text-smoothquant/q8base-a05/result.json`

Config:

- text base: official `Q8_0`
- source weights for SQ packing: `F16`
- `alpha=0.5`
- weight granularity: `per-channel`
- activation quantization: static asymmetric per-tensor
- enabled SQ text layers in this older run: `221`

Result:

- `51/100`

### Newly Re-run Q8_0 Text SQ Variants

New calibration stats were recollected directly on the `Q8_0` base:

- raw act stats:
  `AICAS/output/text-smoothquant-q80-refresh/calib32_q80_raw_act_stats.json`
- candidate file:
  `AICAS/output/text-smoothquant-q80-refresh/calib32_q80_candidates.json`

Observed runtime stats still cover `221` layers, and the same 3 block-31 FFN tensors require backfill to reach policy-side `224/224`.

Checked full-eval variants:

- `alpha=0.5, per-tensor`
  model: `AICAS/output/text-smoothquant-q80-refresh/q80_a05_pt_model.gguf`
  result: `AICAS/output/text-smoothquant-q80-refresh/q80_a05_pt_eval/q80_a05_pt.json`
  score: `48/100`

- `alpha=0.6, per-channel`
  model: `AICAS/output/text-smoothquant-q80-refresh/q80_a06_pc_model.gguf`
  result: `AICAS/output/text-smoothquant-q80-refresh/q80_a06_pc_eval/q80_a06_pc.json`
  score: `49/100`

Conclusion for `Q8_0` text base:

- Newly re-run variants do **not** beat the previously verified `51/100`
- Current best text-side `Q8_0 + SmoothQuant` model remains:
  `AICAS/output/text-smoothquant/q8base-a05/model.gguf`

## Final Recommendation

### Best Text Model Under Q8_0 Base Constraint

- Model:
  `AICAS/output/text-smoothquant/q8base-a05/model.gguf`
- Result:
  `51/100`
- Weight granularity:
  `per-channel`

### Best mmproj SmoothQuant Model

- Model:
  `AICAS/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf`
- Result:
  `56/100`
- Quantized mmproj layers:
  `73/73`
- Weight granularity:
  `per-channel`

### Best Verified Combined Pair

- Text model:
  `AICAS/output/text-smoothquant/q8base-a05/model.gguf`
- mmproj:
  `AICAS/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf`
- Combined eval:
  `AICAS/output/text-mmproj-sq-combined/text_q8base_sq_mmproj_sq.json`
- Combined score:
  `51/100`

## Q8_0 mmproj SmoothQuant Trial

To align with the "all bases should be `Q8_0`" requirement, mmproj packing was extended to support:

- runtime base GGUF: official `Q8_0 mmproj`
- floating-point source weights for SQ repack: `F16 mmproj`

Implementation detail:

- `AICAS/tools/mmproj_pack_gguf.py` now supports `--source-weights-gguf`

Trial artifact:

- Q8_0-base mmproj SQ model:
  `AICAS/output/mmproj-q80-sq-test/mmproj.gguf`
- Pack summary:
  `AICAS/output/mmproj-q80-sq-test/pack_summary.json`

This trial replaced `73` tensors with new SQ `I8` weights.

Combined eval with the current best `Q8_0` text SQ model:

- Text:
  `AICAS/output/text-smoothquant/q8base-a05/model.gguf`
- mmproj:
  `AICAS/output/mmproj-q80-sq-test/mmproj.gguf`
- Result:
  `AICAS/output/text-mmproj-q80sq-combined/text_q80sq_mmproj_q80sq.json`
- Score:
  `51/100`

Conclusion:

- `Q8_0 mmproj -> SmoothQuant` is workable
- Combined score ties the previous best verified combined score
- So if the requirement is "all runtime bases should be `Q8_0`", this mmproj file is acceptable as the current final mmproj choice under that constraint

## Reproduction

### 1. Build candidate manifest

```bash
python AICAS/tools/text_smoothquant_manifest.py \
  --model-gguf AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf
```

### 2. Collect text activation stats

```bash
python AICAS/tools/text_smoothquant_collect_act_stats.py \
  --profiler-bin build-host/bin/llama-mtmd-profiler \
  --model-gguf AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
  --mmproj-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --eval-json AICAS/sampled.json \
  --image-root AICAS/data \
  --limit 32 \
  --output AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/raw_act_stats.json
```

### 3. Prepare candidates with backfill

```bash
python AICAS/tools/text_smoothquant_prepare.py \
  --manifest AICAS/artifacts/text_sq_manifest.json \
  --act-stats AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/raw_act_stats.json \
  --model-gguf AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
  --alpha-grid 0.4,0.5,0.6 \
  --missing-stat-strategy same_kind_prev_layer \
  --output AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/candidates.json
```

### 4. Generate policy and pack GGUF

```bash
python AICAS/tools/text_smoothquant_make_policy.py \
  --candidates AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/candidates.json \
  --alpha 0.5 \
  --clip-mode minmax \
  --output AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/a0_5-minmax-per_tensor/policy.json

python AICAS/tools/text_smoothquant_pack_gguf.py \
  --input-gguf AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
  --policy AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/a0_5-minmax-per_tensor/policy.json \
  --output-gguf AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/a0_5-minmax-per_tensor/model.gguf \
  --output-summary AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/a0_5-minmax-per_tensor/pack_summary.json \
  --weight-granularity per_tensor
```

### 5. Run full accuracy eval

```bash
AICAS/scripts/run_local_acc_eval.sh \
  --server-bin build-host/bin/llama-server \
  --model AICAS/output/text-smoothquant-pilot/f16-calib32/calib-32/a0_5-minmax-per_tensor/model.gguf \
  --mmproj AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --image-folder AICAS/data \
  --ocrbench-file AICAS/sampled.json \
  --output-folder AICAS/output/text-smoothquant-full/f16-calib32-a05-pt \
  --save-name text_sq_f16_calib32_a05_pt \
  --port 18083 \
  --threads 16 \
  --skip-venv
```
