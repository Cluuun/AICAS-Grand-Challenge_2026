# mmproj W8A8 Workflow (SmolVLM2 / IDEFICS3)

This workflow is scoped to:
- `AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf`
- `mul_mat` layers inside mmproj only
- static W8A8 metadata under `aicas.w8a8.*`

## 1) Build layer manifest

```bash
python3 AICAS/tools/mmproj_manifest.py \
  --mmproj-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --profile-json /tmp/mmproj_profile.json \
  --output-manifest AICAS/artifacts/layer_manifest.json \
  --output-inventory AICAS/artifacts/mmproj_mulmat_inventory.json
```

## 2) Build calibration image manifest

```bash
python3 AICAS/tools/mmproj_make_calib_manifest.py \
  --data-root AICAS/data \
  --eval-json AICAS/sampled.json \
  --output AICAS/calib/calib_manifest.json \
  --num-images 256
```

## 3) Collect real mmproj activation stats

This requires the new `clip.cpp` instrumentation to be compiled into `llama-mtmd-profiler`.

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

## 4) Generate static activation params from collected stats

```bash
python3 AICAS/tools/mmproj_calibrate.py \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --act-stats AICAS/artifacts/mmproj_act_stats.json \
  --calib-manifest AICAS/calib/calib_manifest.json \
  --output AICAS/artifacts/quant_params.json \
  --percentile-low 0.1 \
  --percentile-high 99.9
```

## 5) Pack W8A8 mmproj GGUF

```bash
python3 AICAS/tools/mmproj_pack_gguf.py \
  --input-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf \
  --layer-manifest AICAS/artifacts/layer_manifest.json \
  --quant-params AICAS/artifacts/quant_params.json \
  --output-gguf AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf \
  --output-quant-params AICAS/artifacts/quant_params.json \
  --mode quantize
```

## 6) Run local accuracy eval

```bash
bash AICAS/scripts/run_local_acc_eval.sh \
  --model f16 \
  --mmproj AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf
```

## Runtime debug

- `AICAS_MMPROJ_W8A8_DEBUG=1` prints W8A8/fallback layer decisions.
- `AICAS_MMPROJ_ACT_STATS_FILE=/tmp/mmproj_act_stats.json` enables per-layer activation sampling in mmproj.
- `AICAS_MMPROJ_ACT_SAMPLES=4096` controls the per-layer reservoir size used by the runtime sampler.
- `MTMD_DEBUG_GRAPH=1` enables named debug graph nodes (heavy; debug-only).
