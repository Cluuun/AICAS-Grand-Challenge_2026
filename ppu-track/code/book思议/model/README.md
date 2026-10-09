# Dynamic Resolution Router

This folder contains an offline training pipeline for a tiny runtime router that
chooses the lowest image resolution likely to preserve answer quality.

The runtime router is not an answer cache and does not use hand-written keyword
guards. It only reads question text and image metadata, then chooses one fixed
image pixel budget or the original high-resolution processor setting.

Pseudo-label visual-detail router:

```bash
python model/train_resolution_router.py \
  --router-out model/router_model.bin \
  --resolutions 262144,original \
  --router-min-pixels 262144 \
  --pseudo-label-visual-detail \
  --skip-generation \
  --skip-judge \
  --resume
```

Typical training command:

```bash
pip install modelscope

python model/train_resolution_router.py \
  --dataset data \
  --vlm-model-path Qwen3-VL-2B-Instruct \
  --sample-size 1000 \
  --validation-size 200 \
  --extra-sample-spec eval_subset_150.json \
  --resolutions 49152,65536,98304,131072,196608,original \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --work-dir model/runs/res_router_001 \
  --router-out model/router_model.bin \
  --router-min-pixels 49152 \
  --router-threshold 0.30 \
  --resolution-workers 1 \
  --fresh
```

Useful options:

- `--resume`: skip generations and judgments already written under `--work-dir`.
  Use it after an interrupted run; do not combine it with `--fresh`.
- `--fresh`: delete `--work-dir` first, forcing a complete rerun.
- `--extra-sample-spec eval_subset_150.json`: append a fixed 150-question JSON
  to the random sample pool. This makes the training set roughly random 1000
  plus the known 150 eval-style samples, deduplicated by `question_id`.
- `--incremental-labels path.jsonl`: include labels from an earlier run when
  training the final router.
- `--resolution-workers N`: run resolution-generation workers in parallel. On a
  single PPU this can OOM; `1` is safest.
- `--judge-device auto|cuda:0|cpu`: device for the ModelScope text judge.
- `--router-min-pixels 49152`: clamp very low predictions at runtime. This is a
  global safety floor, not a keyword rule. Use `65536` for a more conservative
  run.
- `--router-threshold 0.30`: lower values make the router more willing to move
  up to a higher-resolution bin. Higher values do the opposite and keep more
  samples in the lowest bin.
- `--router-conservative-steps 1`: optionally bump every prediction one
  resolution level higher to trade throughput for safety.

The final runtime model is `model/router_model.bin`. `evaluation_wrapper.py`
loads it automatically if present. If loading fails, runtime falls back to the
original image size instead of applying a keyword rule.
