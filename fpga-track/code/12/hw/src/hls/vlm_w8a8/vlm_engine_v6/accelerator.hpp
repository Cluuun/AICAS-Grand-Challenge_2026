// accelerator.hpp --- V6 top-level entry declaration.
#pragma once

#include "src/streams.hpp"

extern "C" void vlm_engine_v6(
    v6::axi_word_t *gmem_w0,
    v6::axi_word_t *gmem_w1,
    v6::axi_word_t *gmem_act,
    v6::axi_word_t *gmem_out,
    uint64_t        task_byte_offset,

    // AXIS interfaces toward the PE array IP (BD-stitched)
    v6::a_stream_t   &m_axis_a_stream,
    v6::w_stream_t   &m_axis_w0_stream,
    v6::w_stream_t   &m_axis_w1_stream,
    v6::ctrl_stream_t &m_axis_ctrl_stream,
    v6::psum_stream_t &s_axis_psum_stream
);
