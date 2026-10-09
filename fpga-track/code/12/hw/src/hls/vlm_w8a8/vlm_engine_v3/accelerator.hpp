// accelerator.hpp — Top-level declarations for vlm_engine_v3.
#pragma once

#include <ap_int.h>
#include <stdint.h>

typedef ap_uint<128> vlm_w8a8_axi_t;

extern "C" void vlm_engine_v3(
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        const vlm_w8a8_axi_t *act_arena,
        vlm_w8a8_axi_t *out_arena,
        ap_uint<64> task_offset_bytes,
        ap_uint<64> profile_offset_bytes,
        uint32_t task_count,
        uint32_t &status);
