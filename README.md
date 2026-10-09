# IEEE AICAS 2026 Grand Challenge

This repository organizes and open-sources the code and technical reports submitted by the finalist teams of the IEEE AICAS 2026 Grand Challenge.

The contents of this repository are organized based on what the participating teams actually provided, and only include materials that have been publicly authorized or confirmed to be publishable. Each track includes at most the 16 teams that advanced to the finals; whether the code, reports, and reproducible experiment materials are complete is subject to the actual files in the corresponding team directory.

## About the Competition

The IEEE AICAS 2026 Grand Challenge focuses on the efficient deployment of multimodal large models on specialized hardware and edge devices, with an emphasis on model inference optimization, hardware acceleration, and hardware-software co-design. The competition is organized by IEEE AICAS 2026, and registration and team formation are conducted through the Tianchi platform.

- [IEEE AICAS 2026 Grand Challenge Official Website](https://2026.ieee-aicas.org/grand-challenge/)
- [IEEE AICAS 2026 Official Website](https://2026.ieee-aicas.org/)

## Tracks

### Track 1: VLM Efficient Inference and Optimization for AI Chips

This track centers on the Tongyi Qianwen Qwen3-VL-2B-Instruct multimodal model, carrying out efficient inference optimization on designated AI chips and server platforms. Participating teams can improve model inference efficiency through operator optimization and fusion, computation scheduling, memory management, quantization, and deployment strategies.

The preliminary round mainly evaluates model capability and inference efficiency, including accuracy, Time to First Token (TTFT), and throughput. Teams advancing to the finals will complete the final software and hardware co-optimization in a unified Alibaba Cloud APG server environment.

- [Track 1 Tianchi Competition Page](https://tianchi.aliyun.com/competition/entrance/532450)
- Directory: [ppu-track](./ppu-track)

### Track 2: FPGA Hardware-Software System Design for On-Device VLM Inference

This track targets vision-language model on-device inference on the KV260 platform, carrying out FPGA hardware acceleration and hardware-software co-design around the SmolVLM2 model. Participating teams can optimize model deployment through computation architecture, data reuse, pipelining, memory access scheduling, and ARM-FPGA collaboration.

The preliminary round mainly evaluates model capability, Prefill-stage throughput, and Decode-stage throughput. Teams advancing to the finals will continue to complete the hardware system design and inference performance optimization on the KV260 development board.

- [Track 2 Tianchi Competition Page](https://tianchi.aliyun.com/competition/entrance/532451)
- Directory: [fpga-track](./fpga-track)

## Repository Structure

```text
.
├── fpga-track/
│   ├── code/        # Source code and project files submitted by participating teams
│   └── report/      # Technical reports submitted by participating teams
└── ppu-track/
    ├── code/        # Source code and project files submitted by participating teams
    └── report/      # Technical reports submitted by participating teams
```

Team numbers or team names are used to associate code and reports. Because the materials submitted by different teams vary in form and completeness, please refer to the documentation provided by each team for specific compilation methods, dependency environments, hardware requirements, and reproducible experiment steps.

## Scope of Publication and Licensing

This repository is intended for competition results display, academic exchange, and technical reproduction. The copyright and licensing of the code and reports remain with the original authors or the respective rights holders; unless explicitly stated in the files, downloaders should obtain permission from the original authors before using, modifying, or redistributing them, and comply with the licensing terms of third-party dependencies, models, and datasets.

The repository does not contain private test data, access credentials, API keys, or unauthorized model weights from the competition platform.

## Acknowledgments

We thank the IEEE AICAS 2026 Grand Challenge Organizing Committee, the Tianchi competition platform, and all participating teams for their support in organizing these results.