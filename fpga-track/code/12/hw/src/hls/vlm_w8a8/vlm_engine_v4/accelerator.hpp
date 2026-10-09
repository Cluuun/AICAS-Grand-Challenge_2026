// accelerator.hpp --- v4 top-level entry declaration.
#pragma once

#include "src/streams.hpp"

extern "C" void vlm_engine_v4(
    v4::axi_word_t *gmem_w0,
    v4::axi_word_t *gmem_w1,
    v4::axi_word_t *gmem_act,
    v4::axi_word_t *gmem_out,
    uint64_t        task_byte_offset,

    // AXIS interfaces toward the PE array IP (BD-stitched)
    v4::a_stream_t   &m_axis_a_stream,
    v4::w_stream_t   &m_axis_w_stream,
    v4::ctrl_stream_t &m_axis_ctrl_stream,
    v4::psum_stream_t &s_axis_psum_stream
);
