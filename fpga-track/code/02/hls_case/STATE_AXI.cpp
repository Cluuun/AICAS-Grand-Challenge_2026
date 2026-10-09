#include "../src/reuse_axi_common.h"

// ============================================================================
// STATE_AXI：memory-backed RESIDUAL state replay/writeback + CLS index writeback
// ============================================================================
// RESIDUAL 已不再缓存完整 hidden state，本 IP 按 op 显式调度外部 state memory：
//   - AXI_STATE_REPLAY_TOKEN：token-major 重放，供 RESIDUAL 输出 Pre-LN state。
//   - AXI_STATE_REPLAY_DELTA：按 delta 流顺序重放同一 state，供 RESIDUAL 做 x + delta。
//   - AXI_STATE_WRITEBACK：按 RESIDUAL y_stream 顺序写回 token-major state memory。
//   - AXI_STATE_WRITE_CLS：写回 DEMUX argmax 后的 8 个 CLS token id。
// 这样 PYNQ/Spinal 可以用小调度窗口依次启动 replay/writeback，避免单个 mover
// 在硬件里一边长时间写 x_stream 一边等待 y_stream 而形成 FIFO 反压死锁。

static void replay_token_major(
    axi_maxi_half_t* memory_state,
    int num_tokens,
    int num_ct,
    hls::stream<axi_x_vec_t>& x_stream
) {
    // Pre-LN replay 顺序为 token -> CT，和 RESIDUAL 的 res_o_stream 文件语义一致。
    // 该 mover 连接 m_axi 读适配器；关闭自动 loop pipeline 可避免 token replay
    // FSM 到 AXI 读缓冲使能形成高扇出关键路径，代价只是 state replay 多几个周期。
    for (int t = 0; t < num_tokens; ++t) {
        for (int ct = 0; ct < num_ct; ++ct) {
            #pragma HLS pipeline off
            x_stream.write(axi_read_state_vec(memory_state, t * num_ct + ct));
        }
    }
}

static void replay_llm_delta_order(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream
) {
    // LLM delta/writeback 顺序为 CT -> token，匹配 DEMUX_OD。
    for (int ct = 0; ct < AXI_LLM_CT; ++ct) {
        for (int t = 0; t < AXI_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            x_stream.write(axi_read_state_vec(memory_state, t * AXI_LLM_CT + ct));
        }
    }
}

static void replay_vit_delta_order(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream
) {
    // ViT delta/writeback 顺序为 TT_D -> CT -> TP，匹配 DEMUX_OFC2。
    for (int tt_d = 0; tt_d < AXI_VIT_TT_D; ++tt_d) {
        for (int ct = 0; ct < AXI_VIT_CT; ++ct) {
            for (int tp = 0; tp < AXI_VIT_DEMUX_TP; ++tp) {
                #pragma HLS pipeline II=1
                int token = tt_d * AXI_VIT_DEMUX_TP + tp;
                x_stream.write(axi_read_state_vec(memory_state, token * AXI_VIT_CT + ct));
            }
        }
    }
}

static void replay_llm_delta_order_indexed(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream,
    hls::stream<int>& tile_idx_stream
) {
    // AXI_STATE_LAYER 专用 producer：只负责按 LLM delta 顺序发 x 和对应
    // token-major memory index，不读取 y，避免与 RESIDUAL 闭环互等。
    for (int ct = 0; ct < AXI_LLM_CT; ++ct) {
        for (int t = 0; t < AXI_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            int tile_idx = t * AXI_LLM_CT + ct;
            tile_idx_stream.write(tile_idx);
            x_stream.write(axi_read_state_vec(memory_state, tile_idx));
        }
    }
}

static void replay_vit_delta_order_indexed(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream,
    hls::stream<int>& tile_idx_stream
) {
    // AXI_STATE_LAYER 专用 producer：ViT delta 顺序为 TT_D -> CT -> TP，
    // 同时把 token-major memory index 传给 writeback 进程。
    for (int tt_d = 0; tt_d < AXI_VIT_TT_D; ++tt_d) {
        for (int ct = 0; ct < AXI_VIT_CT; ++ct) {
            for (int tp = 0; tp < AXI_VIT_DEMUX_TP; ++tp) {
                #pragma HLS pipeline II=1
                int token = tt_d * AXI_VIT_DEMUX_TP + tp;
                int tile_idx = token * AXI_VIT_CT + ct;
                tile_idx_stream.write(tile_idx);
                x_stream.write(axi_read_state_vec(memory_state, tile_idx));
            }
        }
    }
}

static void writeback_llm_delta_order(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& y_stream
) {
    // y_stream 来自 RESIDUAL，顺序仍是 CT -> token；写回时恢复 token-major memory 地址。
    for (int ct = 0; ct < AXI_LLM_CT; ++ct) {
        for (int t = 0; t < AXI_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            axi_x_vec_t vec = y_stream.read();
            axi_write_state_vec(memory_state, t * AXI_LLM_CT + ct, vec);
        }
    }
}

static void writeback_vit_delta_order(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& y_stream
) {
    // ViT y_stream 顺序为 TT_D -> CT -> TP；写回后 memory 继续保持 token-major。
    for (int tt_d = 0; tt_d < AXI_VIT_TT_D; ++tt_d) {
        for (int ct = 0; ct < AXI_VIT_CT; ++ct) {
            for (int tp = 0; tp < AXI_VIT_DEMUX_TP; ++tp) {
                #pragma HLS pipeline II=1
                int token = tt_d * AXI_VIT_DEMUX_TP + tp;
                axi_x_vec_t vec = y_stream.read();
                axi_write_state_vec(memory_state, token * AXI_VIT_CT + ct, vec);
            }
        }
    }
}

static void writeback_indexed_order(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& y_stream,
    hls::stream<int>& tile_idx_stream,
    int num_tiles
) {
    // AXI_STATE_LAYER 专用 consumer：按 producer 给出的 memory index 消费
    // RESIDUAL y_stream 并写回。与 replay producer 通过 tile_idx_stream 解耦，
    // 让 x_stream 可以先进入 RESIDUAL，再等待 y_stream 回来。
    for (int i = 0; i < num_tiles; ++i) {
        #pragma HLS pipeline II=1
        int tile_idx = tile_idx_stream.read();
        axi_x_vec_t vec = y_stream.read();
        axi_write_state_vec(memory_state, tile_idx, vec);
    }
}

static void replay_writeback_llm_delta_order(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream,
    hls::stream<axi_x_vec_t>& y_stream
) {
    // 顶层闭合调度使用：每发出一个 delta-order state tile，就等待 RESIDUAL 对应
    // y tile 并立刻写回 token-major memory。这样 pass1 可读取 pass0 更新后的 state，
    // 同时不需要在 Spinal 顶层放整层 y_stream FIFO。
    // replay 与 writeback 必须拆成 dataflow 两个进程；单进程相邻
    // x_stream.write/y_stream.read 在 RTL 中可能被调度成先等 y 再发 x。
    #pragma HLS dataflow
    hls::stream<int> tile_idx_stream;
    #pragma HLS stream variable=tile_idx_stream depth=64
    replay_llm_delta_order_indexed(memory_state, x_stream, tile_idx_stream);
    writeback_indexed_order(memory_state, y_stream, tile_idx_stream, AXI_LLM_NUM_X_TILE);
}

static void replay_writeback_vit_delta_order(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream,
    hls::stream<axi_x_vec_t>& y_stream
) {
    // ViT delta/writeback 仍保持 DEMUX_OFC2 的 TT_D -> CT -> TP 顺序；
    // memory 地址恢复为 token-major，避免 RESIDUAL 内部缓存完整 ViT hidden state。
    // 与 LLM 路径一样拆成 dataflow producer/consumer，避免运行时闭环互等。
    #pragma HLS dataflow
    hls::stream<int> tile_idx_stream;
    #pragma HLS stream variable=tile_idx_stream depth=64
    replay_vit_delta_order_indexed(memory_state, x_stream, tile_idx_stream);
    writeback_indexed_order(memory_state, y_stream, tile_idx_stream, AXI_VIT_NUM_X_TILE);
}

static void run_llm_layer_state(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream,
    hls::stream<axi_x_vec_t>& y_stream
) {
    // LLM decoder 每层两次 residual pass：pass0 MHA、pass1 MLP。pass1 的 token replay
    // 必须读取 pass0 writeback 后的 memory state。
    for (int pass = 0; pass < 2; ++pass) {
        replay_token_major(memory_state, AXI_LLM_T, AXI_LLM_CT, x_stream);
        replay_writeback_llm_delta_order(memory_state, x_stream, y_stream);
    }
}

static void run_vit_layer_state(
    axi_maxi_half_t* memory_state,
    hls::stream<axi_x_vec_t>& x_stream,
    hls::stream<axi_x_vec_t>& y_stream
) {
    // ViT encoder 同样两次 residual pass；delta/writeback 顺序由 RESIDUAL/DEMUX 固定。
    for (int pass = 0; pass < 2; ++pass) {
        replay_token_major(memory_state, AXI_VIT_T, AXI_VIT_CT, x_stream);
        replay_writeback_vit_delta_order(memory_state, x_stream, y_stream);
    }
}

static void write_cls_index(
    axi_cls_index_t* memory_cls_y,
    hls::stream<axi_cls_vec_t>& cls_stream
) {
    // DEMUX 已完成 lm_head argmax；这里只写 T=8 个 int32 token id，不搬完整 logits。
    axi_cls_vec_t vec = cls_stream.read();
    for (int t = 0; t < AXI_LLM_T; ++t) {
        #pragma HLS pipeline II=1
        memory_cls_y[t] = vec[t];
    }
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    AXI_STATE_OP_T op,

    axi_maxi_half_t* memory_decoder_state,
    axi_maxi_half_t* memory_vit_state,
    axi_cls_index_t* memory_cls_y,

    hls::stream<axi_x_vec_t>& x_stream,
    hls::stream<axi_x_vec_t>& y_stream,
    hls::stream<axi_cls_vec_t>& cls_stream
) {
    // top：op 每次只做一种搬运动作。STATE buffer 由 PYNQ 按 layer/pass 切换，
    // 因此本 IP 只处理 l_begin 指定的当前层；l_close 保留为冻结接口兼容校验。
    #pragma HLS interface ap_ctrl_chain port=return

    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_decoder_state depth=AXI_LLM_STATE_PACKS offset=direct
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_vit_state     depth=AXI_VIT_STATE_PACKS offset=direct
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_cls_y         depth=AXI_NUM_CLS_INDEX offset=direct

    #pragma HLS interface axis port=x_stream
    #pragma HLS interface axis port=y_stream
    #pragma HLS interface axis port=cls_stream
    #pragma HLS aggregate variable=x_stream compact=bit
    #pragma HLS aggregate variable=y_stream compact=bit
    #pragma HLS aggregate variable=cls_stream compact=bit

    if (op == AXI_STATE_WRITE_CLS) {
        if (is_llm_mode(mode)) {
            write_cls_index(memory_cls_y, cls_stream);
        }
        return;
    }

    if (l_begin >= l_close || !reuse_is_valid_layer(mode, l_begin, true)) {
        return;
    }

    if (is_vit_mode(mode)) {
        if (op == AXI_STATE_LAYER) {
            run_vit_layer_state(memory_vit_state, x_stream, y_stream);
        } else if (op == AXI_STATE_REPLAY_TOKEN) {
            replay_token_major(memory_vit_state, AXI_VIT_T, AXI_VIT_CT, x_stream);
        } else if (op == AXI_STATE_REPLAY_DELTA) {
            replay_vit_delta_order(memory_vit_state, x_stream);
        } else {
            writeback_vit_delta_order(memory_vit_state, y_stream);
        }
    } else {
        if (op == AXI_STATE_LAYER && l_begin < LLAMA_L) {
            run_llm_layer_state(memory_decoder_state, x_stream, y_stream);
        } else if (op == AXI_STATE_REPLAY_TOKEN) {
            replay_token_major(memory_decoder_state, AXI_LLM_T, AXI_LLM_CT, x_stream);
        } else if (op == AXI_STATE_REPLAY_DELTA && l_begin < LLAMA_L) {
            replay_llm_delta_order(memory_decoder_state, x_stream);
        } else if (op == AXI_STATE_WRITEBACK && l_begin < LLAMA_L) {
            writeback_llm_delta_order(memory_decoder_state, y_stream);
        }
    }
}

#ifndef __SYNTHESIS__
static void fill_state_memory(axi_maxi_half_t* memory_state, int count) {
    // step1 smoke test 用递增 pack 填充外部 state，验证 replay/writeback 的流量和顺序不阻塞。
    for (int i = 0; i < count; ++i) {
        memory_state[i] = i;
    }
}

static void drain_x_stream(hls::stream<axi_x_vec_t>& x_stream, int count) {
    for (int i = 0; i < count; ++i) {
        x_stream.read();
    }
}

static void feed_y_stream(hls::stream<axi_x_vec_t>& y_stream, int count) {
    axi_x_vec_t vec;
    for (int lane = 0; lane < AXI_TP * AXI_CP; ++lane) {
        vec[lane] = lane;
    }
    for (int i = 0; i < count; ++i) {
        y_stream.write(vec);
    }
}

static void test_layer() {
    static axi_maxi_half_t memory_decoder_state[AXI_LLM_STATE_PACKS];
    static axi_maxi_half_t memory_vit_state[AXI_VIT_STATE_PACKS];
    static axi_cls_index_t memory_cls_y[AXI_NUM_CLS_INDEX];

    fill_state_memory(memory_decoder_state, AXI_LLM_STATE_PACKS);
    fill_state_memory(memory_vit_state, AXI_VIT_STATE_PACKS);

    hls::stream<axi_x_vec_t> x_stream;
    hls::stream<axi_x_vec_t> y_stream;
    hls::stream<axi_cls_vec_t> cls_stream;

    top(MODE_LLM, 0, 1, AXI_STATE_REPLAY_TOKEN,
        memory_decoder_state, memory_vit_state, memory_cls_y, x_stream, y_stream, cls_stream);
    drain_x_stream(x_stream, AXI_LLM_NUM_X_TILE);

    top(MODE_VIT, 0, 1, AXI_STATE_REPLAY_DELTA,
        memory_decoder_state, memory_vit_state, memory_cls_y, x_stream, y_stream, cls_stream);
    drain_x_stream(x_stream, AXI_VIT_NUM_X_TILE);

    feed_y_stream(y_stream, AXI_LLM_NUM_X_TILE);
    top(MODE_LLM, 0, 1, AXI_STATE_WRITEBACK,
        memory_decoder_state, memory_vit_state, memory_cls_y, x_stream, y_stream, cls_stream);

    feed_y_stream(y_stream, 2 * AXI_LLM_NUM_X_TILE);
    top(MODE_LLM, 0, 1, AXI_STATE_LAYER,
        memory_decoder_state, memory_vit_state, memory_cls_y, x_stream, y_stream, cls_stream);
    drain_x_stream(x_stream, 4 * AXI_LLM_NUM_X_TILE);

    axi_cls_vec_t cls_vec;
    for (int t = 0; t < AXI_LLM_T; ++t) {
        cls_vec[t] = t;
    }
    cls_stream.write(cls_vec);
    top(MODE_LLM, LLAMA_L, LLAMA_L + 1, AXI_STATE_WRITE_CLS,
        memory_decoder_state, memory_vit_state, memory_cls_y, x_stream, y_stream, cls_stream);
}

int main() {
    test_layer();
    return 0;
}
#endif
