// accelerator.hpp --- V5 top-level entry declaration.
#pragma once

#include "src/streams.hpp"

extern "C" void vlm_engine_v5(
    v5::axi_word_t *gmem_w0,
    v5::axi_word_t *gmem_w1,
    v5::axi_word_t *gmem_act,
    v5::axi_word_t *gmem_out,
    uint64_t        task_byte_offset,

    // AXIS interfaces toward the PE array IP (BD-stitched)
    v5::a_stream_t   &m_axis_a_stream,
    v5::w_stream_t   &m_axis_w_stream,
    v5::ctrl_stream_t &m_axis_ctrl_stream,
    v5::psum_stream_t &s_axis_psum_stream
);
