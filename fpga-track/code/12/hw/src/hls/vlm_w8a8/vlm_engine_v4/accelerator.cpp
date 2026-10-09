// accelerator.cpp --- v4 top-level. Dispatches one task per invocation.
//
// Interfaces:
//   m_axi gmem_w0/w1 (HP0/HP1) : weight + weight scale arenas (read-only)
//   m_axi gmem_act   (HP2)     : activation arena (read-only for compute)
//   m_axi gmem_out   (HP3)     : output arena (write)
//   m_axis a/w/ctrl            : feeders to pe_array_v4 IP (BD-connected)
//   s_axis psum                : drain from pe_array_v4 IP
//   s_axilite control          : ap_ctrl_hs + scalar arg port
//
// vs v3: profile arena removed (LUT savings ~5K).

#include "accelerator.hpp"
#include "src/axi_io.hpp"
#include "src/dense_engine.hpp"
#include "src/ffn_engine.hpp"

extern "C" void vlm_engine_v4(
    v4::axi_word_t *gmem_w0,
    v4::axi_word_t *gmem_w1,
    v4::axi_word_t *gmem_act,
    v4::axi_word_t *gmem_out,
    uint64_t        task_byte_offset,
    v4::a_stream_t &m_axis_a_stream,
    v4::w_stream_t &m_axis_w_stream,
    v4::ctrl_stream_t &m_axis_ctrl_stream,
    v4::psum_stream_t &s_axis_psum_stream
) {
    using namespace v4;
    #pragma HLS INTERFACE m_axi  port=gmem_w0  bundle=gmem_w0  offset=slave \
        depth=kArenaDepthWords max_read_burst_length=128 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi  port=gmem_w1  bundle=gmem_w1  offset=slave \
        depth=kArenaDepthWords max_read_burst_length=128 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi  port=gmem_act bundle=gmem_act offset=slave \
        depth=kArenaDepthWords max_read_burst_length=128 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi  port=gmem_out bundle=gmem_out offset=slave \
        depth=kArenaDepthWords max_write_burst_length=32 num_write_outstanding=2
    #pragma HLS INTERFACE s_axilite port=gmem_w0          bundle=control
    #pragma HLS INTERFACE s_axilite port=gmem_w1          bundle=control
    #pragma HLS INTERFACE s_axilite port=gmem_act         bundle=control
    #pragma HLS INTERFACE s_axilite port=gmem_out         bundle=control
    #pragma HLS INTERFACE s_axilite port=task_byte_offset bundle=control
    #pragma HLS INTERFACE s_axilite port=return           bundle=control
    #pragma HLS INTERFACE axis port=m_axis_a_stream
    #pragma HLS INTERFACE axis port=m_axis_w_stream
    #pragma HLS INTERFACE axis port=m_axis_ctrl_stream
    #pragma HLS INTERFACE axis port=s_axis_psum_stream

    task_t task;
    load_task(gmem_act, task_byte_offset, &task);

    if (task.magic != VLM_W8A8_TASK_MAGIC || task.version != VLM_W8A8_ABI_VERSION) {
        return;
    }

    static ActCache act_cache;
    static ActCache mid_cache;
    #pragma HLS BIND_STORAGE variable=act_cache.buf type=ram_s2p impl=uram
    #pragma HLS BIND_STORAGE variable=mid_cache.buf type=ram_s2p impl=uram
    // UltraRAM: 4K (depth) x 72bits (width) per block.
    // KV260 total URAM: 64 blocks, total BRAM_18K: 288 blocks.
    // The dense and FFN schedulers share run_ntile_accumulate() below, so the
    // large load/compute/drain datapath stays a single reusable HLS module.
    #pragma HLS ALLOCATION function instances=run_ntile_accumulate limit=1

    if (task_is_dense_or_qkv(task.task_type)) {
        run_prefill_dense(gmem_w0, gmem_w1, gmem_act, gmem_out, task,
            act_cache,
            m_axis_a_stream, m_axis_w_stream,
            m_axis_ctrl_stream, s_axis_psum_stream);
    } else if (task_is_ffn(task.task_type)) {
        run_prefill_ffn(gmem_w0, gmem_w1, gmem_act, gmem_out, task,
            act_cache, mid_cache,
            m_axis_a_stream, m_axis_w_stream,
            m_axis_ctrl_stream, s_axis_psum_stream);
    }
    // Decode (M=1) and other task types: handled by separate v3 light engine
    // outside this top entry; integrate later if needed.
}
