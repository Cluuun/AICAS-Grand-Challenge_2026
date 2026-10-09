#pragma once

#include "../common/vlm_w8a8_common.hpp"

using accelerator_task_t = linear_task_t;

extern "C" void vlm_engine_v2(
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        const vlm_w8a8_axi_t * act_arena,
        vlm_w8a8_axi_t * out_arena,
        ap_uint<64> task_offset_bytes,
        ap_uint<64> profile_offset_bytes,
        uint32_t task_count,
        uint32_t & status);
