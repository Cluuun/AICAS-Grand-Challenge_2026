# aicas_ppu_v195_20260604

Submit package built from `versions/v195_dead_arg_trim`.

Key changes over v192:

- Keep v188 no-bootstrap grid-sync initialization.
- Keep v192 no-padding dynamic MLP staging before down_proj.
- Remove stale prefetch/fallback arguments from decode-kernel call boundaries.
- Specialize Phase-4 gate/up/down and down_proj decode math to the active W8
  path.

Local PPU validation on 150 samples:

- TTFT: 19.49 ms
- Throughput: 370.73 tokens/sec

Model weights, quantization data, activation precision, and visual-token policy
are unchanged from the v192 lineage.
