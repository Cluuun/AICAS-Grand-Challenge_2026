# Submission config

- profile: `clamp160`
- summary: 128/160 visual-token clamp (smaller than clamp192 for lower TTFT). 54 enumerated grids << 192 buckets -> all pre-captured (clamp/bucket compatible).
- baked defaults:
  - `JUNKRAT_PROCESSOR_MIN_VISUAL_TOKENS=128`
  - `JUNKRAT_PROCESSOR_MAX_VISUAL_TOKENS=160`
  - `JUNKRAT_PROCESSOR_MERGE_SIZE=0`
  - `JUNKRAT_LLM_VISUAL_POOLING=mean`
- package note: runtime quantization, Sage mixed-KV decode, fused down->norm->QKV, default W4A16 gate_up/down/qkv/o_proj, default W4A16 lm_head, and rowwise-Q8 bundled codebook remain enabled by default.
