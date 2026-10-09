// axi_io.hpp --- task descriptor load + scalar-write helpers for v4.
// Profile path is removed (vs v3) to free LUT.
#pragma once

#include "types.hpp"

namespace v4 {

// Estimated cycles: ~24 (one beat per 16 bytes of task struct, II=1, +setup)
static void load_task(const axi_word_t *act_arena,
                      uint64_t byte_offset,
                      task_t *out_task) {
    #pragma HLS INLINE off
    const uint32_t word_offset = byte_offset / kAxiBytes;
    const uint32_t task_words  = sizeof(task_t) / kAxiBytes;

    ap_uint<8> buf[sizeof(task_t)];
    #pragma HLS ARRAY_PARTITION variable=buf complete

    for (uint32_t w = 0; w < task_words; ++w) {
        #pragma HLS PIPELINE II=1
        axi_word_t word = act_arena[word_offset + w];
        for (uint32_t b = 0; b < kAxiBytes; ++b) {
            #pragma HLS UNROLL
            buf[w * kAxiBytes + b] = word(b * 8 + 7, b * 8);
        }
    }

    uint8_t *dst = reinterpret_cast<uint8_t *>(out_task);
    for (uint32_t i = 0; i < sizeof(task_t); ++i) {
        #pragma HLS UNROLL
        dst[i] = buf[i];
    }
}

} // namespace v4
