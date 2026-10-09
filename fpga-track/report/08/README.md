# Tryo KV260 Board Evaluation Package

Copy this package to the board, for example:

```bash
/home/xilinx/kv260_llama_pynq/tryo_board
```

All provided evaluation commands run the PL clock at 300 MHz.

## 1. Program The Bitstream

```bash
cd /home/xilinx/kv260_llama_pynq/tryo_board
bash scripts/program_bitstream.sh
```

## 2. Start The Tryo Server

Open a terminal and keep it running:

```bash
cd /home/xilinx/kv260_llama_pynq/tryo_board
bash scripts/start_server_300mhz.sh
```

The server listens on:

```text
http://127.0.0.1:8080/v1
```

## 3. Run Evaluation

Open another terminal.

Throughput:

```bash
cd /home/xilinx/kv260_llama_pynq/tryo_board
bash scripts/run_throughput.sh
```

Energy:

```bash
cd /home/xilinx/kv260_llama_pynq/tryo_board
bash scripts/run_energy.sh
```

TTFT:

```bash
cd /home/xilinx/kv260_llama_pynq/tryo_board
bash scripts/run_ttft.sh
```

Optional image demo:

```bash
cd /home/xilinx/kv260_llama_pynq/tryo_board
bash scripts/run_demo.sh
```

## Package Layout

```text
overlay/      bitstream and hardware handoff files
weights/      packed decoder weights, vocab projection weights, token embeddings
runtime/      board runtime and AXI-Lite register access
scripts/      bitstream programming, server startup, and run wrappers
eval/         throughput, energy, and TTFT evaluation scripts/config
models/       GGUF model files used by the vision-token exporter
tokenizer/    tokenizer/config files used by text decoding
tools/        llama-smolvlm-vit-encode
```
