// accelerator.cpp --- V6 top-level. Dispatches one task per invocation.
//
// Interfaces:
//   m_axi gmem_w0/w1 (HP0/HP1) : weight + weight scale arenas (read-only)
//   m_axi gmem_act   (HP2)     : activation arena (read-only for compute)
//   m_axi gmem_out   (HP3)     : output arena (write)
//   m_axis a/w0/w1/ctrl        : feeders to pe_array_v6 IP (BD-connected)
//   s_axis psum                : drain from pe_array_v6 IP
//   s_axilite control          : ap_ctrl_hs + scalar arg port
//
// vs v3: profile arena removed (LUT savings ~5K).

#include "accelerator.hpp"
#include "src/axi_io.hpp"
#include "src/dense_engine.hpp"
#include "src/ffn_engine.hpp"

extern "C" void vlm_engine_v6(
    v6::axi_word_t *gmem_w0,
    v6::axi_word_t *gmem_w1,
    v6::axi_word_t *gmem_act,
    v6::axi_word_t *gmem_out,
    uint64_t        task_byte_offset,
    v6::a_stream_t &m_axis_a_stream,
    v6::w_stream_t &m_axis_w0_stream,
    v6::w_stream_t &m_axis_w1_stream,
    v6::ctrl_stream_t &m_axis_ctrl_stream,
    v6::psum_stream_t &s_axis_psum_stream
) {
    using namespace v6;
    // Keep the HP ports burst-friendly, but do not let the Vitis AXI adapters
    // consume excessive BRAM for deep outstanding queues.  The v6 true
    // ping-pong scheduler hides most weight latency with local banks; four
    // outstanding 64-beat bursts per weight port are enough to keep the HP
    // channels busy while saving routing-critical BRAM near the PE/DSP columns.
    #pragma HLS INTERFACE m_axi  port=gmem_w0  bundle=gmem_w0  offset=slave \
        depth=kArenaDepthWords max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi  port=gmem_w1  bundle=gmem_w1  offset=slave \
        depth=kArenaDepthWords max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi  port=gmem_act bundle=gmem_act offset=slave \
        depth=kArenaDepthWords max_read_burst_length=64 num_read_outstanding=4
    #pragma HLS INTERFACE m_axi  port=gmem_out bundle=gmem_out offset=slave \
        depth=kArenaDepthWords max_write_burst_length=32 num_write_outstanding=2
    #pragma HLS INTERFACE s_axilite port=gmem_w0          bundle=control
    #pragma HLS INTERFACE s_axilite port=gmem_w1          bundle=control
    #pragma HLS INTERFACE s_axilite port=gmem_act         bundle=control
    #pragma HLS INTERFACE s_axilite port=gmem_out         bundle=control
    #pragma HLS INTERFACE s_axilite port=task_byte_offset bundle=control
    #pragma HLS INTERFACE s_axilite port=return           bundle=control
    #pragma HLS INTERFACE axis port=m_axis_a_stream
    #pragma HLS INTERFACE axis port=m_axis_w0_stream
    #pragma HLS INTERFACE axis port=m_axis_w1_stream
    #pragma HLS INTERFACE axis port=m_axis_ctrl_stream
    #pragma HLS INTERFACE axis port=s_axis_psum_stream

    task_t task;
    load_task(gmem_act, task_byte_offset, &task);

    if (task.magic != VLM_W8A8_TASK_MAGIC || task.version != VLM_W8A8_ABI_VERSION_V6) {
        return;
    }

    static ActCache act_cache;
    static DownCtxPool down_ctx;
    static OutAccBatch work0;
    static OutAccBatch work1;
    static out_acc_t bias_cache[kBiasCacheCols];
    #pragma HLS BIND_STORAGE variable=act_cache.buf type=ram_s2p impl=uram
    #pragma HLS BIND_STORAGE variable=down_ctx.v type=ram_s2p impl=uram
    // work0/work1 are the reusable 40-bit output accumulators used by dense,
    // FFN gate/up, and FFN down.  They were previously inferred as many small
    // BRAM banks (~48 BRAM_18K total after cyclic lane partitioning), which
    // pushes the full KV260 design close to the physical BRAM-tile limit.
    // URAM has enough headroom in v6 and keeps the same accumulator lifetime,
    // so move these scratch buffers off BRAM without changing the datapath.
    #pragma HLS BIND_STORAGE variable=work0.slab type=ram_s2p impl=uram
    #pragma HLS BIND_STORAGE variable=work1.slab type=ram_s2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=down_ctx.v cyclic factor=kScaleDrainColLanes dim=2
    // The shared dense/down bias cache is cold compared with PE replay and
    // absorb.  Keeping it in URAM frees physical BRAM tiles for the PE-side
    // FIFOs and weight buffers without changing the bias load/tile semantics.
    #pragma HLS BIND_STORAGE variable=bias_cache type=ram_2p impl=uram
    // UltraRAM: 4K (depth) x 72bits (width) per block.
    // KV260 total URAM: 64 blocks, total BRAM_18K: 288 blocks.
    // The dense and FFN schedulers share run_ntile_accumulate_batch(), so the
    // large load/compute/drain datapath stays a single reusable HLS module.
    #pragma HLS ALLOCATION function instances=run_ntile_accumulate_batch limit=1

    if (task_is_dense_or_qkv(task.task_type)) {
        run_prefill_dense(gmem_w0, gmem_w1, gmem_act, gmem_out, task,
            act_cache, work0, bias_cache,
            m_axis_a_stream, m_axis_w0_stream, m_axis_w1_stream,
            m_axis_ctrl_stream, s_axis_psum_stream);
    } else if (task_is_ffn(task.task_type)) {
        run_prefill_ffn(gmem_w0, gmem_w1, gmem_act, gmem_out, task,
            act_cache, down_ctx, work0, work1, bias_cache,
            m_axis_a_stream, m_axis_w0_stream, m_axis_w1_stream,
            m_axis_ctrl_stream, s_axis_psum_stream);
    }
    // Decode linear/FFN uses the same task types with rows=1. Attention stays
    // on CPU in this v6 design.
}
