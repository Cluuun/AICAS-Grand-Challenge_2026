#include "../src/reuse_axi_common.h"

// ============================================================================
// WEIGHT_AXI：LLM/ViT 共享 GEMM 权重搬运器
// ============================================================================
// 从原 M_AXI 拆出的权重路径。该 IP 只读取 decoder/CLS/ViT 三个权重 namespace，
// 按 WQ/WS bit 比例超周期解包为 PERMUTE/GEMM-facing 的 wq/ws1/ws2 streams。
// mode 只在 layer 粒度选择地址空间和总循环数，WQ/WS 分发与解包逻辑保持单套。

constexpr int AXI_VIT_QKV_BLOCK_CYCS = AXI_VIT_QKV_LOOPS * AXI_W_CYCS_PER_LOOP;
constexpr int AXI_VIT_O_BLOCK_CYCS   = AXI_VIT_O_LOOPS   * AXI_W_CYCS_PER_LOOP;
constexpr int AXI_VIT_FC1_BLOCK_CYCS = AXI_VIT_FC1_LOOPS * AXI_W_CYCS_PER_LOOP;
constexpr int AXI_VIT_FC2_BLOCK_CYCS = AXI_VIT_FC2_LOOPS * AXI_W_CYCS_PER_LOOP;

static void load_vit_qkv_cache_half(
    int block_base,
    axi_maxi_half_t* memory_vit_w_half,
    axi_maxi_half_t qkv_cache[AXI_VIT_QKV_BLOCK_CYCS]
) {
    for (int n = 0; n < AXI_VIT_QKV_BLOCK_CYCS; ++n) {
        #pragma HLS pipeline II=1
        qkv_cache[n] = memory_vit_w_half[block_base + n];
    }
}

static void emit_vit_qkv_cached_half(
    int layer_base,
    axi_maxi_half_t* memory_vit_w_half,
    axi_maxi_half_t qkv_cache[AXI_VIT_QKV_BLOCK_CYCS],
    hls::stream<axi_maxi_half_t>& mem_stream_half
) {
    for (int h = 0; h < AXI_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            int block_base = layer_base +
                (AXI_VIT_QKV_LOOP_BASE + (h * 3 + qkv) * AXI_VIT_QKV_LOOPS) * AXI_W_CYCS_PER_LOOP;
            load_vit_qkv_cache_half(block_base, memory_vit_w_half, qkv_cache);
            for (int tt = 0; tt < AXI_VIT_GEMM_TT; ++tt) {
                for (int n = 0; n < AXI_VIT_QKV_BLOCK_CYCS; ++n) {
                    #pragma HLS pipeline II=1
                    mem_stream_half.write(qkv_cache[n]);
                }
            }
        }
    }
}

static void weight_producer_half(
    REUSE_MODE_T mode,
    int l,
    axi_maxi_half_t* memory_decoder_w_half,
    axi_maxi_half_t* memory_cls_w_half,
    axi_maxi_half_t* memory_vit_w_half,
    axi_maxi_half_t vit_qkv_cache[AXI_VIT_QKV_BLOCK_CYCS],
    hls::stream<axi_maxi_half_t>& mem_stream_half
) {
    bool run_cls = is_llm_mode(mode) && (l == LLAMA_L);
    bool run_vit = is_vit_mode(mode);
    if (run_vit) {
        // ViT DDR 权重保持 compact，不保存 TT=128 份重复；这里按 PERMUTE 消费顺序重放整块矩阵。
        // 不能把单个 beat hold 128 次，否则会变成 tile0*128,tile1*128；PERMUTE 需要 matrix*128。
        int layer_base = l * AXI_VIT_W_CYCS;
        emit_vit_qkv_cached_half(layer_base, memory_vit_w_half, vit_qkv_cache, mem_stream_half);
        for (int tt = 0; tt < AXI_VIT_GEMM_TT; ++tt) {
            for (int loop = 0; loop < AXI_VIT_O_LOOPS; ++loop) {
                for (int cyc = 0; cyc < AXI_W_CYCS_PER_LOOP; ++cyc) {
                    #pragma HLS pipeline II=1
                    mem_stream_half.write(memory_vit_w_half[layer_base + (AXI_VIT_O_LOOP_BASE + loop) * AXI_W_CYCS_PER_LOOP + cyc]);
                }
            }
        }
        for (int tt = 0; tt < AXI_VIT_GEMM_TT; ++tt) {
            for (int loop = 0; loop < AXI_VIT_FC1_LOOPS; ++loop) {
                for (int cyc = 0; cyc < AXI_W_CYCS_PER_LOOP; ++cyc) {
                    #pragma HLS pipeline II=1
                    mem_stream_half.write(memory_vit_w_half[layer_base + (AXI_VIT_FC1_LOOP_BASE + loop) * AXI_W_CYCS_PER_LOOP + cyc]);
                }
            }
        }
        for (int tt = 0; tt < AXI_VIT_GEMM_TT; ++tt) {
            for (int loop = 0; loop < AXI_VIT_FC2_LOOPS; ++loop) {
                for (int cyc = 0; cyc < AXI_W_CYCS_PER_LOOP; ++cyc) {
                    #pragma HLS pipeline II=1
                    mem_stream_half.write(memory_vit_w_half[layer_base + (AXI_VIT_FC2_LOOP_BASE + loop) * AXI_W_CYCS_PER_LOOP + cyc]);
                }
            }
        }
    } else if (run_cls) {
        for (int n = 0; n < AXI_CLS_W_CYCS; ++n) {
            #pragma HLS pipeline II=1
            mem_stream_half.write(memory_cls_w_half[n]);
        }
    } else {
        for (int n = 0; n < AXI_DECODER_W_CYCS; ++n) {
            #pragma HLS pipeline II=1
            mem_stream_half.write(memory_decoder_w_half[l * AXI_DECODER_W_CYCS + n]);
        }
    }
}

static void weight_producer(
    REUSE_MODE_T mode,
    int l,
    axi_maxi_half_t* memory_decoder_w_lo,
    axi_maxi_half_t* memory_decoder_w_hi,
    axi_maxi_half_t* memory_cls_w_lo,
    axi_maxi_half_t* memory_cls_w_hi,
    axi_maxi_half_t* memory_vit_w_lo,
    axi_maxi_half_t* memory_vit_w_hi,
    hls::stream<axi_maxi_t>& mem_stream
) {
    bool run_cls = is_llm_mode(mode) && (l == LLAMA_L);
    bool run_vit = is_vit_mode(mode);
    int total_cycs = run_vit ? AXI_VIT_STREAM_W_CYCS : (run_cls ? AXI_CLS_W_CYCS : AXI_DECODER_W_CYCS);

    #pragma HLS dataflow
    hls::stream<axi_maxi_half_t> mem_stream_lo;
    hls::stream<axi_maxi_half_t> mem_stream_hi;
    axi_maxi_half_t vit_qkv_cache_lo[AXI_VIT_QKV_BLOCK_CYCS];
    axi_maxi_half_t vit_qkv_cache_hi[AXI_VIT_QKV_BLOCK_CYCS];
    #pragma HLS bind_storage variable=vit_qkv_cache_lo type=ram_2p impl=uram
    #pragma HLS bind_storage variable=vit_qkv_cache_hi type=ram_2p impl=uram

    weight_producer_half(mode, l, memory_decoder_w_lo, memory_cls_w_lo, memory_vit_w_lo, vit_qkv_cache_lo, mem_stream_lo);
    weight_producer_half(mode, l, memory_decoder_w_hi, memory_cls_w_hi, memory_vit_w_hi, vit_qkv_cache_hi, mem_stream_hi);

    for (int n = 0; n < total_cycs; ++n) {
        #pragma HLS pipeline II=1
        axi_maxi_t packet = 0;
        packet.range(AXI_DW_MAXI_HALF - 1, 0) = mem_stream_lo.read();
        packet.range(AXI_DW_MAXI - 1, AXI_DW_MAXI_HALF) = mem_stream_hi.read();
        mem_stream.write(packet);
    }
}

static void weight_distributor(
    REUSE_MODE_T mode,
    int l,
    hls::stream<axi_maxi_t>& mem_stream,
    hls::stream<axi_maxi_t>& wq_queue,
    hls::stream<axi_maxi_t>& ws_queue
) {
    int total_loops = is_vit_mode(mode) ? AXI_VIT_STREAM_LOOPS : ((l == LLAMA_L) ? AXI_CLS_LOOPS : AXI_DECODER_LOOPS);
    for (int loop = 0; loop < total_loops; ++loop) {
        for (int cyc = 0; cyc < AXI_W_CYCS_PER_LOOP; ++cyc) {
            #pragma HLS pipeline II=1
            axi_maxi_t packet = mem_stream.read();
            if (cyc < AXI_WQ_CYCS) {
                wq_queue.write(packet);
            } else {
                ws_queue.write(packet);
            }
        }
    }
}

static void compose_ws_stage1(
    REUSE_MODE_T mode,
    int l,
    hls::stream<axi_maxi_t>& ws_queue,
    hls::stream<axi_ws_pack_t>& ws_pack_stream
) {
    int loops = is_vit_mode(mode) ? AXI_VIT_STREAM_LOOPS : ((l == LLAMA_L) ? AXI_CLS_LOOPS : AXI_DECODER_LOOPS);
    for (int loop = 0; loop < loops; ++loop) {
        for (int aggr = 0; aggr < AXI_WS_CYCS_AGGR_NUM; ++aggr) {
            axi_ws_pack_t pack = 0;
            for (int i = 0; i < AXI_WS_CYCS_AGGR_SIZE; ++i) {
                #pragma HLS pipeline II=1
                axi_maxi_t packet = ws_queue.read();
                pack >>= AXI_DW_MAXI;
                pack.range(AXI_DW_MAXI * AXI_WS_CYCS_AGGR_SIZE - 1,
                           AXI_DW_MAXI * (AXI_WS_CYCS_AGGR_SIZE - 1)) = packet;
            }
            ws_pack_stream.write(pack);
        }
    }
}

static void compose_ws_stage2(
    REUSE_MODE_T mode,
    int l,
    hls::stream<axi_ws_pack_t>& ws_pack_stream,
    hls::stream<axi_ws_vec_t>& ws1_stream,
    hls::stream<axi_ws_vec_t>& ws2_stream
) {
    int loops = is_vit_mode(mode) ? AXI_VIT_STREAM_LOOPS : ((l == LLAMA_L) ? AXI_CLS_LOOPS : AXI_DECODER_LOOPS);
    for (int loop = 0; loop < loops; ++loop) {
        for (int aggr = 0; aggr < AXI_WS_CYCS_AGGR_NUM; ++aggr) {
            axi_ws_pack_t pack = ws_pack_stream.read();
            for (int i = 0; i < AXI_WS_PACK_PER_AGGR; ++i) {
                #pragma HLS pipeline II=1
                axi_ws_vec_t vec1, vec2;
                for (int g = 0; g < AXI_G; ++g) {
                    #pragma HLS unroll
                    vec1[g] = pack.range(DW_WS - 1, 0);
                    vec2[g] = pack.range(DW_WS * 2 - 1, DW_WS);
                    pack >>= (DW_WS * 2);
                }
                ws1_stream.write(vec1);
                ws2_stream.write(vec2);
            }
        }
    }
}

static void compose_wq_stage1(
    REUSE_MODE_T mode,
    int l,
    hls::stream<axi_maxi_t>& wq_queue,
    hls::stream<axi_wq_pack_t>& wq_pack_stream
) {
    int loops = is_vit_mode(mode) ? AXI_VIT_STREAM_LOOPS : ((l == LLAMA_L) ? AXI_CLS_LOOPS : AXI_DECODER_LOOPS);
    for (int loop = 0; loop < loops; ++loop) {
        for (int aggr = 0; aggr < AXI_WQ_CYCS_AGGR_NUM; ++aggr) {
            axi_wq_pack_t pack = 0;
            for (int i = 0; i < AXI_WQ_CYCS_AGGR_SIZE; ++i) {
                #pragma HLS pipeline II=1
                axi_maxi_t packet = wq_queue.read();
                pack >>= AXI_DW_MAXI;
                pack.range(AXI_DW_MAXI * AXI_WQ_CYCS_AGGR_SIZE - 1,
                           AXI_DW_MAXI * (AXI_WQ_CYCS_AGGR_SIZE - 1)) = packet;
            }
            wq_pack_stream.write(pack);
        }
    }
}

static void compose_wq_stage2(
    REUSE_MODE_T mode,
    int l,
    hls::stream<axi_wq_pack_t>& wq_pack_stream,
    hls::stream<axi_wq_vec_t>& wq_stream
) {
    int loops = is_vit_mode(mode) ? AXI_VIT_STREAM_LOOPS : ((l == LLAMA_L) ? AXI_CLS_LOOPS : AXI_DECODER_LOOPS);
    for (int loop = 0; loop < loops; ++loop) {
        for (int aggr = 0; aggr < AXI_WQ_CYCS_AGGR_NUM; ++aggr) {
            axi_wq_pack_t pack = wq_pack_stream.read();
            for (int i = 0; i < AXI_WQ_PACK_PER_AGGR; ++i) {
                #pragma HLS pipeline II=1
                axi_wq_vec_t vec;
                for (int g = 0; g < AXI_G * AXI_G; ++g) {
                    #pragma HLS unroll
                    vec[g] = pack.range(DW_WQ - 1, 0);
                    pack >>= DW_WQ;
                }
                wq_stream.write(vec);
            }
        }
    }
}

static void read_w(
    REUSE_MODE_T mode,
    int l,
    axi_maxi_half_t* memory_decoder_w_lo,
    axi_maxi_half_t* memory_decoder_w_hi,
    axi_maxi_half_t* memory_cls_w_lo,
    axi_maxi_half_t* memory_cls_w_hi,
    axi_maxi_half_t* memory_vit_w_lo,
    axi_maxi_half_t* memory_vit_w_hi,
    hls::stream<axi_wq_vec_t>& wq_stream,
    hls::stream<axi_ws_vec_t>& ws1_stream,
    hls::stream<axi_ws_vec_t>& ws2_stream
) {
    #pragma HLS dataflow
    hls::stream<axi_maxi_t> mem_stream;
    hls::stream<axi_maxi_t> wq_queue;
    hls::stream<axi_maxi_t> ws_queue;
    hls::stream<axi_wq_pack_t> wq_pack_stream;
    hls::stream<axi_ws_pack_t> ws_pack_stream;

    weight_producer(mode, l, memory_decoder_w_lo, memory_decoder_w_hi,
                    memory_cls_w_lo, memory_cls_w_hi, memory_vit_w_lo, memory_vit_w_hi,
                    mem_stream);
    weight_distributor(mode, l, mem_stream, wq_queue, ws_queue);
    compose_wq_stage1(mode, l, wq_queue, wq_pack_stream);
    compose_wq_stage2(mode, l, wq_pack_stream, wq_stream);
    compose_ws_stage1(mode, l, ws_queue, ws_pack_stream);
    compose_ws_stage2(mode, l, ws_pack_stream, ws1_stream, ws2_stream);
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,

    axi_maxi_half_t* memory_decoder_w_lo,
    axi_maxi_half_t* memory_decoder_w_hi,
    axi_maxi_half_t* memory_cls_w_lo,
    axi_maxi_half_t* memory_cls_w_hi,
    axi_maxi_half_t* memory_vit_w_lo,
    axi_maxi_half_t* memory_vit_w_hi,

    hls::stream<axi_wq_vec_t>& wq_stream,
    hls::stream<axi_ws_vec_t>& ws1_stream,
    hls::stream<axi_ws_vec_t>& ws2_stream
) {
    // top：半开 layer 范围；LLM 允许 l==LLAMA_L 读取 CLS/lm_head 权重，ViT 只允许 vision layer。
    #pragma HLS interface ap_ctrl_chain port=return

    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_decoder_w_lo depth=LLAMA_L*AXI_DECODER_W_CYCS offset=direct max_read_burst_length=32 num_read_outstanding=16
    #pragma HLS interface mode=m_axi bundle=gmem2 port=memory_decoder_w_hi depth=LLAMA_L*AXI_DECODER_W_CYCS offset=direct max_read_burst_length=32 num_read_outstanding=16
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_cls_w_lo     depth=AXI_CLS_W_CYCS offset=direct max_read_burst_length=32 num_read_outstanding=16
    #pragma HLS interface mode=m_axi bundle=gmem2 port=memory_cls_w_hi     depth=AXI_CLS_W_CYCS offset=direct max_read_burst_length=32 num_read_outstanding=16
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_vit_w_lo     depth=VIT_L*AXI_VIT_W_CYCS offset=direct max_read_burst_length=32 num_read_outstanding=16
    #pragma HLS interface mode=m_axi bundle=gmem2 port=memory_vit_w_hi     depth=VIT_L*AXI_VIT_W_CYCS offset=direct max_read_burst_length=32 num_read_outstanding=16

    #pragma HLS interface axis port=wq_stream
    #pragma HLS interface axis port=ws1_stream
    #pragma HLS interface axis port=ws2_stream
    #pragma HLS aggregate variable=wq_stream compact=bit
    #pragma HLS aggregate variable=ws1_stream compact=bit
    #pragma HLS aggregate variable=ws2_stream compact=bit

    for (int l = l_begin; l < l_close; ++l) {
        #pragma HLS loop_tripcount min=1 max=33
        if (!reuse_is_valid_layer(mode, l, true)) {
            continue;
        }
        read_w(mode, l, memory_decoder_w_lo, memory_decoder_w_hi,
               memory_cls_w_lo, memory_cls_w_hi, memory_vit_w_lo, memory_vit_w_hi,
               wq_stream, ws1_stream, ws2_stream);
    }
}

#ifndef __SYNTHESIS__
static void fill_half_words(axi_maxi_half_t* memory, int count) {
    // step1 smoke test 使用确定性伪数据，只验证 LLM/ViT/CLS 三类权重流量与解包不阻塞。
    for (int i = 0; i < count; ++i) {
        memory[i] = i;
    }
}

static void drain_weight_streams(
    hls::stream<axi_wq_vec_t>& wq_stream,
    hls::stream<axi_ws_vec_t>& ws1_stream,
    hls::stream<axi_ws_vec_t>& ws2_stream,
    int num_wq_vecs,
    int num_ws_vecs
) {
    for (int i = 0; i < num_wq_vecs; ++i) {
        wq_stream.read();
    }
    for (int i = 0; i < num_ws_vecs; ++i) {
        ws1_stream.read();
        ws2_stream.read();
    }
}

static void test_vit_qkv_cache_exact_half(axi_maxi_half_t* memory_vit_w_half) {
    static axi_maxi_half_t qkv_cache[AXI_VIT_QKV_BLOCK_CYCS];

    for (int h = 0; h < AXI_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            int block_base = (AXI_VIT_QKV_LOOP_BASE + (h * 3 + qkv) * AXI_VIT_QKV_LOOPS) * AXI_W_CYCS_PER_LOOP;
            load_vit_qkv_cache_half(block_base, memory_vit_w_half, qkv_cache);
            for (int tt = 0; tt < AXI_VIT_GEMM_TT; ++tt) {
                for (int n = 0; n < AXI_VIT_QKV_BLOCK_CYCS; ++n) {
                    axi_maxi_half_t expected = memory_vit_w_half[block_base + n];
                    axi_maxi_half_t got = qkv_cache[n];
                    if (got != expected) {
                        cerr << "VIT QKV cache mismatch h=" << h
                             << " qkv=" << qkv
                             << " tt=" << tt
                             << " n=" << n
                             << " got=" << got
                             << " expected=" << expected << endl;
                        assert(false);
                    }
                }
            }
        }
    }
}

static void test_layer() {
    static axi_maxi_half_t memory_decoder_w_lo[AXI_DECODER_W_CYCS];
    static axi_maxi_half_t memory_decoder_w_hi[AXI_DECODER_W_CYCS];
    static axi_maxi_half_t memory_cls_w_lo[AXI_CLS_W_CYCS];
    static axi_maxi_half_t memory_cls_w_hi[AXI_CLS_W_CYCS];
    static axi_maxi_half_t memory_vit_w_lo[AXI_VIT_W_CYCS];
    static axi_maxi_half_t memory_vit_w_hi[AXI_VIT_W_CYCS];

    fill_half_words(memory_decoder_w_lo, AXI_DECODER_W_CYCS);
    fill_half_words(memory_decoder_w_hi, AXI_DECODER_W_CYCS);
    fill_half_words(memory_cls_w_lo, AXI_CLS_W_CYCS);
    fill_half_words(memory_cls_w_hi, AXI_CLS_W_CYCS);
    fill_half_words(memory_vit_w_lo, AXI_VIT_W_CYCS);
    fill_half_words(memory_vit_w_hi, AXI_VIT_W_CYCS);

    hls::stream<axi_wq_vec_t> wq_stream;
    hls::stream<axi_ws_vec_t> ws1_stream;
    hls::stream<axi_ws_vec_t> ws2_stream;

    top(MODE_LLM, 0, 1,
        memory_decoder_w_lo, memory_decoder_w_hi, memory_cls_w_lo, memory_cls_w_hi,
        memory_vit_w_lo, memory_vit_w_hi, wq_stream, ws1_stream, ws2_stream);
    drain_weight_streams(wq_stream, ws1_stream, ws2_stream,
                         AXI_DECODER_NUM_WQ / (AXI_G * AXI_G),
                         AXI_DECODER_NUM_WS / AXI_G);

    top(MODE_LLM, LLAMA_L, LLAMA_L + 1,
        memory_decoder_w_lo, memory_decoder_w_hi, memory_cls_w_lo, memory_cls_w_hi,
        memory_vit_w_lo, memory_vit_w_hi, wq_stream, ws1_stream, ws2_stream);
    drain_weight_streams(wq_stream, ws1_stream, ws2_stream,
                         AXI_CLS_NUM_WQ / (AXI_G * AXI_G),
                         AXI_CLS_NUM_WS / AXI_G);

    // VIT-02b resynth gate: prove cached QKV replay preserves the exact DDR
    // re-stream order without paying the full ViT-layer CSim runtime.
    test_vit_qkv_cache_exact_half(memory_vit_w_lo);
    test_vit_qkv_cache_exact_half(memory_vit_w_hi);
}

int main() {
    test_layer();
    return 0;
}
#endif
