# Run `2026-06-05_161344_ship_submission`

## vs ver0.0 baseline

| 指标 | 当前 | baseline (ver0.0) | diff |
|---|---|---|---|
| prefill (t/s)  | 16.44 | 16.24 | +1.25% |
| decode  (t/s)  | 7.05  | 7.02  | +0.33% |
| energy (t/J)   | 1.323 | 1.292 | +2.34% |
| TTFT a (ms/ch) | 6.92 | 7.12 | +2.77% |
| TTFT b (ms)    | 17705 | 17567 | -0.79% |
| **Tianchi 总分** | **41.67** | 40.89 | **+0.78** |

## 复赛 4 项指标

| 维度 | 实测 | baseline (官方) | Ratio | 加权 |
|---|---|---|---|---|
| Prefill (t/s) | 16.44 | 3.01  | 0.817 | 20·R = 16.34 |
| Decode (t/s)  | 7.05  | 5.55  | 0.213 | 30·R = 6.38 |
| Energy (t/J)  | 1.323 | 0.97 | 0.267 | 30·R = 8.00 |
| TTFT (avg)    | a=6.92, b=17705 | a=12, b=54000 | 0.548 | 20·R = 10.95 |

## 配置

- Binary version: `ship_submission` (llama.cpp git `420515e`)
- Overlay: `fpga_gemm_v2.1_phase15a`
- Model: SmolVLM2-500M-Video-Instruct-Q8_0 + mmproj Q8_0
- Env: `GGML_XRT_CAST=1`, `LD_LIBRARY_PATH=lib/`, KV cache f16, no `-fa`
- Server: port 8080, `-c 4096 -t 4 --no-warmup`
- 评测口径: throughput/energy 用 `test2.jpg` + LONG_PROMPT (max 4096 tok),TTFT 6 case

## 原始数据

- `raw/throughput.json` / `raw/energy.json` / `raw/ttft.json` (+ `raw/acc.json` if --with-acc)
- `raw/server.log` (clean run) / `raw/chrono.stderr` (chrono run)
- `aicas_submission.json` (merge_results.py 输出, 天池提交格式)
