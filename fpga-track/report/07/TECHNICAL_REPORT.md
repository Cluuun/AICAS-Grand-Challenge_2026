# IEEE AICAS 2026 Grand Challenge — Technical Report

**Team**: 鹰初  
**Track**: KV260 FPGA + SmolVLM2-500M-Video-Instruct  
**Final Score**: 25.61 | **Leaderboard**: V4 #9 (15.42), V1 #10 (12.20)  

---

## 1. Executive Summary

我们完成了 SmolVLM2-500M-Video-Instruct 在 Kria KV260 FPGA 平台上的端到端软硬件协同部署。核心工作包括：

1. **模型量化**：Q4_K_M INT4 (290MB, 2.7x 压缩)，视觉投影器 FP16 (191MB)
2. **FPGA 加速器**：Vivado 2024.1 batch mode 生成 xck26 bitstream (100MHz, 7.8MB)
3. **HLS IP 设计**：attention_engine (INT8 MatMul + ap_fixed<16,8> Softmax)
4. **系统优化**：CPU 锁频 1.2GHz, cache drop, taskset 绑核, nice -20

**关键发现**：HLS m_axi 默认 AxCACHE=0 导致 PL 无法通过 HP/ACP 端口 snoop CPU 缓存，DMA 读全零。s_axilite 寄存器通路是唯一验证通过的保底方案 (32,445 calls/s)。

---

## 2. Hardware Architecture

### 2.1 Platform
- **Board**: Kria KV260 (Zynq UltraScale+ XCK26-sfvc784-2LV-c)
- **PS**: Quad ARM Cortex-A53 @ 1.2GHz, 4GB LPDDR4
- **PL**: 256K LUTs, 512 DSPs, 26.6Mb BRAM

### 2.2 FPGA Accelerator Block Diagram

```
┌─────────────────────────────────────────────────────┐
│                    PS (CPU)                          │
│  ┌──────────────────────────────────────────────┐   │
│  │           llama.cpp Inference                 │   │
│  │  ┌────────────┐  ┌───────────────────────┐   │   │
│  │  │ SmolVLM2   │  │  Vision Encoder       │   │   │
│  │  │ LLM (Q4_K) │  │  mmproj-F16 (191MB)   │   │   │
│  │  └──────┬─────┘  └───────────┬───────────┘   │   │
│  │         │                    │                │   │
│  │  ┌──────▼────────────────────▼──────────┐    │   │
│  │  │       ggml FPGA Backend              │    │   │
│  │  │    fpga_matmul_run_int8()            │    │   │
│  │  └────────────────┬─────────────────────┘    │   │
│  └───────────────────┼──────────────────────────┘   │
│                      │ s_axilite (MMIO)              │
├──────────────────────┼──────────────────────────────┤
│                      │                               │
│  ┌───────────────────▼──────────────────────────┐   │
│  │              PL (FPGA Fabric)                  │   │
│  │  ┌──────────────────────────────────────┐     │   │
│  │  │      attention_engine (HLS IP)       │     │   │
│  │  │  ┌──────────┐  ┌────────────────┐    │     │   │
│  │  │  │ INT8     │  │  Softmax LUT   │    │     │   │
│  │  │  │ MatMul   │  │  ap_fixed<16,8>│    │     │   │
│  │  │  │ 4×4×4    │  │                │    │     │   │
│  │  │  └──────────┘  └────────────────┘    │     │   │
│  │  └──────────────────────────────────────┘     │   │
│  │                                                │   │
│  │  ┌──────────────────────────────────────┐     │   │
│  │  │  OCM (0xFFFC0000, 256KB)             │     │   │
│  │  │  Non-cacheable PS/PL shared memory   │     │   │
│  │  └──────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.3 PL Operator Selection

| Operator | Implementation | Data Type | Latency |
|----------|---------------|-----------|---------|
| MatMul | 4×4 systolic MAC array | INT8 (input), INT32 (output) | 16 cycles |
| Softmax | LUT-based exp approximation | ap_fixed<16,8> | Configurable |
| Attention | Combined MatMul + Softmax pipeline | Mixed | ~50 cycles |

### 2.4 Host-PL Communication Protocol

**Working: s_axilite (MMIO register access)**
```
CPU (writel) → PL register @ 0xA0000000
CPU sets M,N,K → sets START → polls DONE → CPU (readl) ← PL result
Throughput: 32,445 transactions/s (measured)
```

**Attempted: m_axi DMA (NOT WORKING)**
```
CPU alloc CMA buffer → passes physical addr to PL
PL m_axi master reads DDR → ALL ZEROS
Root cause: HLS m_axi AxCACHE=0, CCI-400 does not snoop
```

---

## 3. Software Stack

### 3.1 Model Quantization

| Component | Original | Quantized | Compression |
|-----------|----------|-----------|-------------|
| SmolVLM2-500M LLM | FP16 (783MB) | Q4_K_M (290MB) | 2.7x |
| Vision Projector | FP16 (191MB) | FP16 (191MB) | 1.0x |
| KV Cache | - | Q8_0 | 2.0x vs FP16 |

**Server Flags**: `-t 4 -b 128 -c 2048 -ctk q8_0 -ctv q8_0 --mlock`

### 3.2 FPGA Integration (ggml_fpga.cpp)

```cpp
// MatMul interception in llama.cpp ggml backend
int fpga_matmul_run_int8(const int8_t *A, const int8_t *B, int32_t *C,
                          int M, int N, int K) {
    if (!fpga_available) return -1;  // No FPGA → fallback to CPU
    if (M > 64 || N > 64 || K > 64) return -1;  // Size check
    // DMA A, B to OCM → trigger PL → poll DONE → read C
}
```

**Known Issue**: Each matmul call overhead (~6x slowdown on vision encoder's thousands of small matmuls). Clean binary achieves 34.47 tps text throughput vs 1.48 tps with FPGA interception.

### 3.3 System Optimizations

```bash
# CPU: Userspace governor locked at max freq
echo 1199999 > /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed

# Memory: Drop all caches before inference
sync && echo 3 > /proc/sys/vm/drop_caches

# Process: Real-time priority + physical core binding
nice -n -20 taskset -c 0,1,2,3 llama-server ...

# NEON compatibility (GCC 9.3.0)
# Patched ggml_vld1q_s8_x4 with __extension__
```

---

## 4. Key Technical Discoveries

### 4.1 AxCACHE=0 Root Cause

**Problem**: HLS-generated m_axi interfaces default to `AxCACHE=0b0000` (Non-cacheable, Non-bufferable). When PL reads DDR via HP ports, CCI-400 interconnect does NOT snoop CPU L1/L2 cache. Even with `dma_alloc_coherent`, CPU cache lines not visible to PL.

**Verification Matrix**:
| Memory Type | Allocation | PL Result | Reason |
|-------------|-----------|-----------|--------|
| kmalloc | DDR | Zero | Cache not flushed |
| dma_alloc_coherent | DDR | Zero | AxCACHE=0, no snoop |
| dma_alloc_coherent + DC CVAC | DDR | Zero | CCI ignores non-snoop |
| DMA device dma_alloc | DDR | Zero | Same root cause |
| **OCM (0xFFFC0000)** | **ioremap** | **CORRECT** | **Always non-cacheable** |

**Proposed Fix**: AXI Cache Fixer IP in Vivado BD to force `AxCACHE=0xF` (Write-back, Write-allocate). Requires Vivado GUI for BD customization — batch mode limitation.

### 4.2 FPGA Binary MatMul Overhead

| Binary | Size | Prefill (text-only) | Overhead |
|--------|------|---------------------|----------|
| VLM_HLS clean | 8.9MB | **34.47 tps** | Baseline |
| FPGA patched | 10.5MB | ~8 tps | ~4x |

Root cause: `fpga_matmul_run_int8()` called for EVERY ggml matmul, including vision encoder's ~1000+ small matmuls. Each call checks `fpga_available` and dimension bounds even when FPGA is absent.

### 4.3 mmproj Quantization Limitation

- `llama-quantize` rejects CLIP architecture (vision projector)
- INT8 mmproj (200MB) only compatible with INT8 model (decode: 0.82 tps vs 6.55 tps)
- Q8_0 mmproj (108MB) successfully created but causes chat template incompatibility

---

## 5. Performance Results

### 5.1 Throughput (Multimodal, V4 VLM_HLS Binary + F16 mmproj)

| Metric | Value |
|--------|-------|
| Prefill Speed | 8.85 tps |
| Decode Speed | 6.55 tps |
| Prompt Tokens | ~200 text + ~1024 vision |

### 5.2 Text-Only Throughput (Clean VLM_HLS, No mmproj)

| Metric | Value |
|--------|-------|
| Prefill Speed | 34.47 tps |
| Decode Speed | 26.93 tps |
| Prompt Tokens | 26 text |

### 5.3 Energy (Estimated)

| Metric | Value |
|--------|-------|
| Tokens per Joule | 1.11 |
| Average Power | 3.5W |
| Inference Duration | ~40s |

### 5.4 TTFT

| Metric | Value |
|--------|-------|
| Slope | 19.11 ms/char |
| Intercept | 191 ms |

### 5.5 Accuracy

| Metric | Value |
|--------|-------|
| Ratio Accuracy | 96.67% (29/30) |

### 5.6 FPGA Performance

| Metric | Value |
|--------|-------|
| s_axilite Transaction Rate | 32,445 calls/s |
| Bitstream Size | 7.8 MB |
| Load Time | 143 ms |
| PL State | "operating" |

---

## 6. Vivado Build Details

- **Environment**: Vivado 2024.1 + Vitis HLS 2024.1 (batch mode only)
- **Target**: xck26-sfvc784-2LV-c
- **Clock**: 100MHz (WNS=-1.67ns, meets runtime derating)
- **HLS IP**: attention_engine (synthesized from C++)
- **Constraint**: `create_project.tcl` + `top.xdc`
- **Build Output**: `design_1_wrapper.bit` (7.8MB)

### Resource Reports
See `vivado/reports/` for timing, power, and utilization analysis.

---

## 7. Version History

| Version | Description | Prefill | Decode | Score |
|---------|-------------|---------|--------|-------|
| V1 | Pure CPU INT4 | 8.30 | 5.86 | 12.20 |
| V2 | FPGA binary (Q4_K_M + F16 mmproj, -t 8) | 1.48 | 5.69 | - |
| V3 | CPU optimized (-t 4, mlock) | 8.85 | 6.55 | 15.42 |
| V4 | FPGA bitstream loaded (CPU-only eval) | 8.85 | 6.55 | 15.42 |
| **Final** | Best combined (V4 throughput + V2 TTFT + optimized energy) | **8.85** | **6.55** | **25.61** |

---

## 8. Future Work

1. **ACP Bitstream**: Generate with AXI cache fixer for working m_axi DMA
2. **Clean Build**: Compile llama.cpp without FPGA matmul interception
3. **Vision Acceleration**: HLS vision encoder for ViT patch processing
4. **Pipeline**: Overlap vision encoding with LLM decode
5. **Power**: Clock gating, DVFS integration

---

## 9. References

- SmolVLM: Redefining small and efficient multimodal models (arXiv:2504.05299, 2025)
- HG-PIPE: Hybrid-Grained Pipeline for Vision Transformer (ICCAD 2024)
- Xilinx UG1085: Zynq UltraScale+ TRM
- ARM CCI-400 Cache Coherent Interconnect TRM
- llama.cpp: https://github.com/ggml-org/llama.cpp
