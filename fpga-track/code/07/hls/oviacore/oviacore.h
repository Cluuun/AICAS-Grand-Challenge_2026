/**
 * OViA: OCM-anchored Vision Accelerator
 * ===========================================
 * IEEE AICAS 2026 Grand Challenge - KV260 FPGA
 *
 * Architecture:
 *   - 16×16 systolic array for INT8 matrix multiply
 *   - OCM (0xFFFC0000) as PS-PL shared memory via AXI master
 *   - s_axilite control interface (start/done/M/N/K/result)
 *   - 3-stage pipeline: load → compute → store
 *
 * Target: xck26-sfvc784-2LV-c @ 100MHz
 * Resource estimate: ~24K LUT, ~16K FF, ~256 DSP, ~64 BRAM
 */

#pragma once

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

// ============================================================
// Architecture Configuration
// ============================================================

// Systolic array dimensions
const int SA_ROWS = 16;        // 16×16 PE array
const int SA_COLS = 16;

// OCM memory map (relative to base 0xFFFC0000)
const int OCM_OFFSET_A   = 0x0000;   // Matrix A:  M×K bytes (INT8)
const int OCM_OFFSET_B   = 0x4000;   // Matrix B:  K×N bytes (INT8)
const int OCM_OFFSET_C   = 0x8000;   // Matrix C:  M×N × 4 bytes (INT32)

// OCM size limits
const int OCM_SIZE       = 0x10000;  // 64KB per section (256KB total OCM)
const int MAX_M          = 1024;     // Max matrix dimension
const int MAX_N          = 1024;
const int MAX_K          = 1024;

// Data types
typedef ap_int<8>   dma_word_t;      // INT8 matrix element
typedef ap_int<32>  acc_word_t;      // INT32 accumulator
typedef ap_uint<64> addr_t;          // 64-bit address for OCM

// ============================================================
// PE (Processing Element) definition
// ============================================================
struct pe_data_t {
    dma_word_t a;    // Input from West (A matrix row)
    dma_word_t b;    // Input from North (B matrix column)
    acc_word_t c;    // Partial sum (accumulated)
};

// ============================================================
// Top-level function declaration
// s_axilite: M, N, K + standard HLS IP control (start/done via return)
// m_axi: ocm (AXI master to OCM at 0xFFFC0000)
// ============================================================
void ovia_core(
    volatile dma_word_t* ocm,   // AXI Master to OCM
    int M,                       // A rows / C rows
    int N,                       // B cols / C cols
    int K                        // A cols / B rows
);
