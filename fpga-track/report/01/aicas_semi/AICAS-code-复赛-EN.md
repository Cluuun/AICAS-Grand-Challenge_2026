# AICAS 2026 Grand Challenge Hardware - Second Round

## Overview

The second round continues the goal of the preliminary round — deploying and running SmolVLM2-500M-Video-Instruct on the KV260 platform, leveraging the on-chip CPU and FPGA resources to optimize the chip architecture for edge-side inference of multimodal Vision-Language Models (VLMs).

**The second round focuses on PL-side hardware acceleration**: from this round on, participating teams MUST use the KV260's Programmable Logic (PL) resources to accelerate model inference. Throughput improvements obtained solely through llama.cpp software parameter tuning, CPU multi-threading affinity, or compile-time switches will no longer be recognized.

## Key Changes for the Second Round

The second round introduces the following changes compared with the preliminary round:

1. **Mandatory PL-side hardware participation**: All participating teams must submit a bitstream/overlay loadable on KV260. The organizing committee will load it on a standard board; submissions that fail to load will not enter scoring.
2. **New Energy Efficiency metric (Tokens/Joule)**: Suppresses the optimization path of stacking CPU threads or relaxing the power budget purely to inflate throughput.
3. **New Time-To-First-Token metric (TTFT)**: Discourages batching strategies that sacrifice first-response user experience for the sake of throughput.
4. **Unified baselines + committee re-test**: The original baselines $T_{\text{ori}}$, $E_{\text{ori}}$, $\bar{a}_{\text{ori}}$, $\bar{b}_{\text{ori}}$ are uniformly measured and published by the organizing committee on the standard KV260 board. The final score is determined by the committee's re-test; self-reported JSON serves as initial-scoring reference only.
5. **Reduced accuracy sample count**: Considering the on-device inference speed of KV260, the accuracy test sample count is reduced from the preliminary round's 100 to 30 in the second round (stratified random sampling, 3+3+3+3+3+3+12).


## Software Environment

In the second round, each qualifying team is shipped a dedicated KV260 development board. The organizing committee provides a base toolkit archive containing:

- Second-round evaluation scripts (`sample.py`, `acc_eval.py`, `throughput_eval.py`, `energy_eval.py`, `ttft_eval_multiprompt.py`, `merge_results.py`)

Based on this toolkit, participating teams are free to compile their own KV260 Linux image, configure the runtime environment, replace or recompile llama.cpp (including embedding XRT calls to invoke custom PL acceleration logic), flash custom bitstreams, and so on. The committee imposes no restrictions on the specific software stack.

The model weights of SmolVLM2-500M-Video-Instruct GGUF and the OCRBench dataset should be downloaded by participating teams from public sources.

At submission time, the team-delivered image + bitstream + source code + startup script must be sufficient for the committee's standard KV260 board to fully reproduce the testing flow.


## Code Introduction

### sample.py

​	Provides the sampling code that performs **stratified random** sampling over the **57817** original test items to obtain 30 test samples.

​	Considering the on-device inference speed of the KV260 platform, the second round's accuracy sample count is reduced from the preliminary round's 100 to 30 in order to keep a single evaluation run tractable. The sub-task sampling distribution for the second round is:

| Test Category                 | Sample Count |
| ----------------------------- | ------------ |
| Regular Text Recognition      | 3            |
| Irregular Text Recognition    | 3            |
| Artistic Text Recognition     | 3            |
| Handwriting Recognition       | 3            |
| Digit String Recognition      | 3            |
| Non-Semantic Text Recognition | 3            |
| Scene Text-centric VQA        | 12           |

​	Note that both the total sample count and the sub-category distribution differ from the preliminary round; n=30 applies to all participating teams.

### acc_eval.py

​	Performs inference on the sampled dataset and obtains the accuracy and original response results.

### throughput_eval.py

​	Performs inference using longer prompts and larger images, measuring the throughput of the model's prefill and decoding stages.

### energy_eval.py (New in Second Round)

​	Samples the KV260 board's instantaneous power at high frequency through the on-board PMBus / sysfs hwmon interface during inference. Combined with the output token count, it computes the end-to-end energy efficiency (Tokens/Joule). The output JSON contains the sampling trace, average power, total energy, and the number of generated tokens.

### ttft_eval_multiprompt.py (New in Second Round)

​	Measures the end-to-end latency from request submission to receipt of the first token (Time-To-First-Token) via the streaming interface of llama-server. The script reads `ttft_config.json` to drive multiple images (different sizes) and multiple prompt lengths (short / long / xlong); each (image, prompt) combination is run once. A random nonce is injected at the head of each prompt to force an LCP cache miss. The output JSON contains the raw TTFT for every case, plus the per-image least-squares linear fit (slope and intercept) over the prompt-length axis.

### merge_results.py (New in Second Round)

​	Merges the output files of the four evaluation scripts (`acc_eval.py`, `throughput_eval.py`, `energy_eval.py`, `ttft_eval_multiprompt.py`) into a single submission JSON for packaging, upload, and committee re-test.


## Testing Process

This testing process extends that of the preliminary round. **The submitted second-round solution must utilize the FPGA resources of the KV260 platform**; CPU-only solutions will not be ranked.

### Environment Preparation

Participating teams may prepare their own Python runtime.


### Bitstream Loading

Load the bitstream / overlay according to the team's submitted image (PYNQ Overlay, `fpgautil`, or a custom driver are all acceptable). Subsequent tests can only proceed after a successful load. During the committee's re-test, the team's submitted startup script will be executed; any load failure is treated as a failed hardware gate.

### Dataset Sampling

```bash
python sample.py -i <FULL_TEST_JSON_PATH> -o <SAMPLED_JSON_PATH>
```

### Accuracy Testing

In a shell, start llama-server (or the team's custom equivalent inference service):

```bash
./llama-server -m <GGUF_PATH>/SmolVLM2-500M-Video-Instruct-f16.gguf --mmproj <GGUF_PATH>/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf
```

Open another shell and run the accuracy test:

```bash
python acc_eval.py -i <IMAGE_FOLDER_PATH> -d <SAMPLED_JSON_PATH> -o <ACC_OUTPUT_JSON>
```

Where `<IMAGE_FOLDER_PATH>` is the local directory holding the OCRBench dataset, `<SAMPLED_JSON_PATH>` is the sampled-dataset JSON produced by `sample.py`, and `<ACC_OUTPUT_JSON>` is the output JSON path for this accuracy run (e.g. `acc_eval_results.json`).

### Throughput Testing

The input image required for the throughput test is included in the organizing committee's archive.

Without closing llama-server, run the throughput test directly:

```bash
python throughput_eval.py -i <THROUGHPUT_IMAGE_PATH> -o <OUTPUT_JSON_PATH>
```

Where `<THROUGHPUT_IMAGE_PATH>` is the `image.png` file provided in the archive.

### Energy Efficiency Testing (New in Second Round)

Without closing llama-server, run the energy sampling script:

```bash
python energy_eval.py -i <THROUGHPUT_IMAGE_PATH> -o <ENERGY_OUTPUT_JSON> --sample_hz 100
```

The script samples PMBus readings at 100 Hz while triggering inference. The output JSON contains the cumulative energy and the count of generated tokens; the energy efficiency $E$ (Tokens/Joule) is automatically computed and written.

### Time-To-First-Token Testing (New in Second Round)

Without closing llama-server, run the TTFT test under the multi-image × multi-prompt combinations defined by `ttft_config.json`:

```bash
python ttft_eval_multiprompt.py -c ttft_config.json -o <TTFT_OUTPUT_JSON>
```

The script injects `[Nonce: timestamp]` at the head of each prompt to force a cache miss; each case is run once. After completion, it groups by image and performs a least-squares linear fit against prompt length. The output JSON contains the raw TTFT for every case, the per-image fit slope (`slope_ms_per_char`) and intercept (`intercept_ms`), and the cross-image aggregated slope and intercept.

### Result Merging

After all evaluations are complete, merge them into a single submission JSON:

```bash
python merge_results.py \
    --acc        <ACC_OUTPUT_JSON> \
    --throughput <THROUGHPUT_OUTPUT_JSON> \
    --energy     <ENERGY_OUTPUT_JSON> \
    --ttft       <TTFT_OUTPUT_JSON> \
    -o           aicas_submission.json
```

Then rename the output file according to the format specified by the Tianchi platform and upload it.