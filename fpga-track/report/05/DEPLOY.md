# DEPLOY.md — KV260 板侧部署与复现

全程目录结构假设板上工作根为 `~/AICAS2026FINAL/`。

---

## ① 板侧环境前提

- 板卡：AMD Kria KV260（`xck26-sfvc784-2LV-c`，Zynq UltraScale+，PS = Cortex-A53 + PL）
- 操作系统：Ubuntu 22.04（KV260 官方/PYNQ rootfs 均可；官方镜像见 `07_image/`）
- 运行时：**XRT 2.13** + `xmutil`（`xlnx-config` / `kria-utils`），`xbutil` 可用

系统依赖安装（干净镜像必做，**缺 libopenblas 则 llama-server 直接起不来**）：
```bash
sudo apt update
sudo apt install -y libopenblas0-pthread xrt   # Kria apt 源自带 xrt 2.13.479（zocl 在官方内核里）
xbutil --version        # 应为 2.13.x
which xmutil xbutil
```

Python venv（**路径必须精确为 `~/AICAS2026FINAL/eval/venv`**，`run_full_eval.sh` 硬依赖）：
```bash
mkdir -p ~/AICAS2026FINAL/eval
python3 -m venv ~/AICAS2026FINAL/eval/venv
~/AICAS2026FINAL/eval/venv/bin/pip install openai pillow tqdm
```

---

## ② 部署 PL overlay（4 个文件）

**一键（推荐，初赛官方测评同款流程）**——安装 + 激活 + exec 位兜底：

```bash
sudo bash 04_scripts/install_overlay.sh
```

等价手动版：

```bash
sudo mkdir -p /lib/firmware/xilinx/fpga_gemm_v2.1_phase15a
sudo cp 01_overlay/fpga_gemm_v2.1_phase15a.bit.bin \
        01_overlay/fpga_gemm_v2.1_phase15a.xclbin \
        01_overlay/fpga_gemm_v2.1_phase15a.dtbo \
        01_overlay/shell.json \
        /lib/firmware/xilinx/fpga_gemm_v2.1_phase15a/
```

> `shell.json` 内 `num_slots` 必须是字符串 `"1"`（已是正确值）；否则 `xmutil loadapp` 静默返回 -1。

---

## ③ 部署运行时二进制 + 库 + GGUF

把本包 `00_runtime/bin` `00_runtime/lib` cp 到板上 `~/AICAS2026FINAL/{bin,lib}`：

```bash
mkdir -p ~/AICAS2026FINAL/bin ~/AICAS2026FINAL/lib
cp 00_runtime/bin/llama-server   ~/AICAS2026FINAL/bin/
cp 00_runtime/lib/*.so           ~/AICAS2026FINAL/lib/
chmod +x ~/AICAS2026FINAL/bin/llama-server
```

**GGUF 模型**（不在本包内，体积大）：从 HuggingFace 官方仓库
[HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
下载 Q8_0 两个文件，放到 `~/AICAS2026FINAL/gguf/`，**文件名必须精确一致**：

```bash
mkdir -p ~/AICAS2026FINAL/gguf
# 放入以下两个文件：
#   ~/AICAS2026FINAL/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf
#   ~/AICAS2026FINAL/gguf/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
md5sum ~/AICAS2026FINAL/gguf/*Q8_0.gguf
```

md5 应分别为（板上实测值，保证与本队伍复现口径一致）：

| 文件 | md5 |
|---|---|
| `SmolVLM2-500M-Video-Instruct-Q8_0.gguf` | `471b0799baaaa669fa90ea886322f16d` |
| `mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf` | `5b16ac2e1e7109b482ba58c9f9e79e36` |

把本包 `04_scripts/`（含 `helpers/` 打分三件套）与 `00_runtime/eval/` 也放到对应位置（脚本默认从 `~/AICAS2026FINAL` 找）：
```bash
mkdir -p ~/AICAS2026FINAL/scripts ~/AICAS2026FINAL/eval/semifinal
cp 04_scripts/*.sh            ~/AICAS2026FINAL/scripts/
cp -r 04_scripts/helpers      ~/AICAS2026FINAL/scripts/helpers
cp 00_runtime/eval/*          ~/AICAS2026FINAL/eval/semifinal/
chmod +x ~/AICAS2026FINAL/scripts/*.sh
```

---

## ④ 加载 overlay

（②已用 `install_overlay.sh` 则已激活，跳过）

```bash
sudo xmutil unloadapp || true
sudo xmutil loadapp fpga_gemm_v2.1_phase15a
xbutil examine                 # 应列出 kernel fpga_gemm_kernel
```

> ⚠ 每次 reboot 后默认 overlay 是 k26-starter-kits，必须重新 loadapp，否则全部回退 CPU 无报错。

---

## ⑤ 启动 llama-server

```bash
cd ~/AICAS2026FINAL
./scripts/run_server.sh
```

---

## ⑥ 验证 PL 已接上（关键标志）

`run_server.sh` 末尾会自动 grep；手动核对：

```bash
grep -E "ggml-xrt: loaded|kernel fpga_gemm" ~/AICAS2026FINAL/server.log
```

看到类似
```
ggml-xrt: loaded /lib/firmware/xilinx/fpga_gemm_v2.1_phase15a/...xclbin, kernel fpga_gemm_kernel
```
---

## ⑦ 跑评测

**推荐：一键全维（throughput + energy + TTFT + 打分报告）**——会自杀旧 server、按 ship 环境重启、产出 `metrics.json` 与 Tianchi 分：

```bash
~/AICAS2026FINAL/scripts/run_full_eval.sh        # 默认 Q8/Q8，约 15 分钟
# 输出: ~/AICAS2026FINAL/results/<时间戳>_phase15a_ship/{report.md, metrics.json}
```

或手动单维（服务起来后，默认 `http://127.0.0.1:8080`）：

```bash
source ~/AICAS2026FINAL/eval/venv/bin/activate
cd ~/AICAS2026FINAL/eval/semifinal
python throughput_eval.py         # 默认 image=test2.jpg, max_tokens=4096, temp=0.0, 非流式
python energy_eval.py  --image test2.jpg          # Tokens/J，PMBus 100Hz
python ttft_eval_multiprompt.py -c ttft_config.json
```

- 读数源：llama-server `timings.prompt_per_second`（prefill）/ `predicted_per_second`（decode）
- **Acc 维度**（`acc_eval.py`，OCRBench n=30 子串包含）需 OCRBench 图片库（不在本包内，体积大）：
  使用组委会下发的 OCRBench 图集目录，
  `python acc_eval.py -i <OCRBench_Images 路径>`；题目清单 `sample_30.json` 已在包内。
  一键脚本加 `--with-acc` 同理需先备好图集。

---

## ⑧ 板况自检（成绩异常时先查这三项）

1. **DP 中断风暴**：`run_server.sh` 已自动 unbind 显示控制器兜底。核查：两次
   `grep fd4a0000.display /proc/interrupts` 间隔 1s 不应增长数千；
2. **polkitd 卡死吃一整核**：脚本已自动 `systemctl stop polkit`；
3. **overlay 未加载**：reboot 后默认 overlay 是 k26-starter-kits，prefill 会静默掉到 CPU；
   重新 `sudo xmutil loadapp fpga_gemm_v2.1_phase15a` 并看 ⑥ 的正向证据。

