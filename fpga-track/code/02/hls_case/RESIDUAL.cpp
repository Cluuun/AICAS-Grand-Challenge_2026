#include "../src/reuse_common.h"
#include "../src/utils.h"
// ============================================================================
// LLM/ViT 复用 RESIDUAL：memory-backed 版本
// ============================================================================
// 功能：
//   1. 本模块不再缓存完整 hidden state，避免 ViT 1024x768 residual 常驻 60+ URAM。
//   2. 外部 M_AXI/调度器负责把当前 state 从 memory 按需要重放到 x_stream。
//   3. 每个 pass 先读一份 x 写到 res_o_stream，供 RMSNorm/LayerNorm 使用；
//      子层 delta 回来后，再读一份同一 state，与 res_i_stream 相加后写到 y_stream。
//   4. y_stream 是每个 pass 的 writeback 流，外部写回 memory，作为下一 pass/层的 state。
//
// 流顺序：
//   - res_o_stream：token -> CT，保持旧 RESIDUAL_O 文件顺序，供 RMSNorm/LayerNorm 链式仿真。
//   - LLM add/writeback：CT -> token，匹配 LLM DEMUX_OD/GEMM 输出顺序。
//   - ViT add/writeback：TT_D(8-token tile) -> CT -> TP，匹配 ViT DEMUX_OFC2 输出顺序。
//
// 资源意图：
//   - RESIDUAL 自身只保留 8-lane 加法和少量控制逻辑；完整 state 放外部 memory。
//   - mode 只选择 token/channel 边界与流顺序，不在 lane 加法内复制大 datapath。

constexpr int MB_CP             = REUSE_CP;
constexpr int MB_TP             = 1;

constexpr int MB_LLM_T          = REUSE_LLM_TILE_T;
constexpr int MB_LLM_C          = LLAMA_C;
constexpr int MB_LLM_CT         = MB_LLM_C / MB_CP;
constexpr int MB_LLM_NUM_X      = MB_LLM_T * MB_LLM_C;
constexpr int MB_LLM_POS        = 96;

constexpr int MB_VIT_T          = VIT_S;
constexpr int MB_VIT_C          = VIT_C;
constexpr int MB_VIT_CT         = MB_VIT_C / MB_CP;
constexpr int MB_VIT_NUM_X      = MB_VIT_T * MB_VIT_C;
// ViT 新导出按 batch*patch 保存，当前 vision_0/new_txt 与其他层同批为 13*1024。
constexpr int MB_VIT_T_LOAD     = 13312;
constexpr int MB_VIT_POS        = 0;
constexpr int MB_VIT_DEMUX_TP   = 8;
constexpr int MB_VIT_TT_D       = MB_VIT_T / MB_VIT_DEMUX_TP;

static_assert(MB_TP == 1, "RESIDUAL keeps one token per stream beat");
static_assert(MB_LLM_C % MB_CP == 0, "LLM C must be divisible by CP");
static_assert(MB_VIT_C % MB_CP == 0, "ViT C must be divisible by CP");

typedef hls::vector<REUSE_X_T, MB_CP> mb_residual_vec_t;

static VIT_X_T mb_vit_x_from_reuse(REUSE_X_T val) {
    #pragma HLS inline
    return reuse_truncate<VIT_X_T, REUSE_X_T>(val);
}

static REUSE_X_T mb_reuse_x_from_vit(VIT_X_T val) {
    #pragma HLS inline
    return reuse_extend<REUSE_X_T, VIT_X_T>(val);
}

static mb_residual_vec_t mb_add_llm_vec(mb_residual_vec_t x_vec, mb_residual_vec_t d_vec) {
    #pragma HLS inline
    mb_residual_vec_t y_vec;
    for (int cp = 0; cp < MB_CP; ++cp) {
        #pragma HLS unroll
        X_T x_val = (X_T)x_vec[cp];
        X_T d_val = (X_T)d_vec[cp];
        y_vec[cp] = (REUSE_X_T)(x_val + d_val);
    }
    return y_vec;
}

static mb_residual_vec_t mb_add_vit_vec(mb_residual_vec_t x_vec, mb_residual_vec_t d_vec) {
    #pragma HLS inline
    mb_residual_vec_t y_vec;
    for (int cp = 0; cp < MB_CP; ++cp) {
        #pragma HLS unroll
        VIT_X_T x_val = mb_vit_x_from_reuse(x_vec[cp]);
        VIT_X_T d_val = mb_vit_x_from_reuse(d_vec[cp]);
        y_vec[cp] = mb_reuse_x_from_vit((VIT_X_T)(x_val + d_val));
    }
    return y_vec;
}

void mb_emit_token_major(
    int num_tokens,
    int num_ct,
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_o_stream
) {
    // Pre-LN 输出：外部按 token -> CT 从 memory 重放当前 state，这里原样转发。
    for (int t = 0; t < num_tokens; ++t) {
        for (int ct = 0; ct < num_ct; ++ct) {
            #pragma HLS pipeline II=1
            mb_residual_vec_t x_vec = x_stream.read();
            res_o_stream.write(x_vec);
        }
    }
}

void mb_apply_llm_delta(
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_i_stream,
    hls::stream<mb_residual_vec_t>& y_stream
) {
    // LLM writeback 顺序匹配 DEMUX_OD：CT 外层，token 内层；M_AXI 按该顺序回写 state memory。
    for (int ct = 0; ct < MB_LLM_CT; ++ct) {
        for (int t = 0; t < MB_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            mb_residual_vec_t x_vec = x_stream.read();
            mb_residual_vec_t d_vec = res_i_stream.read();
            y_stream.write(mb_add_llm_vec(x_vec, d_vec));
        }
    }
}

void mb_apply_vit_delta(
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_i_stream,
    hls::stream<mb_residual_vec_t>& y_stream
) {
    // ViT writeback 顺序保持 DEMUX_OFC2：TT_D -> CT -> 8 token，避免在 RESIDUAL 内重排大矩阵。
    for (int tt_d = 0; tt_d < MB_VIT_TT_D; ++tt_d) {
        for (int ct = 0; ct < MB_VIT_CT; ++ct) {
            for (int tp = 0; tp < MB_VIT_DEMUX_TP; ++tp) {
                #pragma HLS pipeline II=1
                mb_residual_vec_t x_vec = x_stream.read();
                mb_residual_vec_t d_vec = res_i_stream.read();
                y_stream.write(mb_add_vit_vec(x_vec, d_vec));
            }
        }
    }
}

void mb_run_llm_residual(
    int l_begin,
    int l_close,
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_i_stream,
    hls::stream<mb_residual_vec_t>& res_o_stream,
    hls::stream<mb_residual_vec_t>& y_stream
) {
    // LLM CLS/lm_head 路径没有 residual add，只需要把 memory state 重放给下游 RMSNorm/CLS。
    if (l_begin == LLAMA_L || l_close == LLAMA_L + 1) {
        mb_emit_token_major(MB_LLM_T, MB_LLM_CT, x_stream, res_o_stream);
        return;
    }

    for (int l = l_begin; l < l_close && l < LLAMA_L; ++l) {
        for (int pass = 0; pass < 2; ++pass) {
            mb_emit_token_major(MB_LLM_T, MB_LLM_CT, x_stream, res_o_stream);
            mb_apply_llm_delta(x_stream, res_i_stream, y_stream);
        }
    }
}

void mb_run_vit_residual(
    int l_begin,
    int l_close,
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_i_stream,
    hls::stream<mb_residual_vec_t>& res_o_stream,
    hls::stream<mb_residual_vec_t>& y_stream
) {
    for (int l = l_begin; l < l_close && l < VIT_L; ++l) {
        for (int pass = 0; pass < 2; ++pass) {
            mb_emit_token_major(MB_VIT_T, MB_VIT_CT, x_stream, res_o_stream);
            mb_apply_vit_delta(x_stream, res_i_stream, y_stream);
        }
    }
}

// top：端口形态保持 mode-aware RESIDUAL 接口；x_stream/y_stream 语义改为 memory-backed replay/writeback。
void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_i_stream,
    hls::stream<mb_residual_vec_t>& res_o_stream,
    hls::stream<mb_residual_vec_t>& y_stream
) {
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=x_stream
    #pragma HLS interface axis port=res_i_stream
    #pragma HLS interface axis port=res_o_stream
    #pragma HLS interface axis port=y_stream
    #pragma HLS aggregate variable=x_stream compact=bit
    #pragma HLS aggregate variable=res_i_stream compact=bit
    #pragma HLS aggregate variable=res_o_stream compact=bit
    #pragma HLS aggregate variable=y_stream compact=bit

    if (is_vit_mode(mode)) {
        mb_run_vit_residual(l_begin, l_close, x_stream, res_i_stream, res_o_stream, y_stream);
    } else {
        mb_run_llm_residual(l_begin, l_close, x_stream, res_i_stream, res_o_stream, y_stream);
    }
}

// ============================================================================
// Testbench：模拟 M_AXI 的 replay/writeback 调度，同时覆盖 LLM 和 ViT mode。
// ============================================================================

int64_t MB_LLM_REF_X             [MB_LLM_NUM_X];
int64_t MB_LLM_REF_Y             [MB_LLM_NUM_X];
int64_t MB_LLM_REF_MHA_X         [MB_LLM_NUM_X];
int64_t MB_LLM_REF_MHA_O         [MB_LLM_NUM_X];
int64_t MB_LLM_REF_MLP_X         [MB_LLM_NUM_X];
int64_t MB_LLM_REF_MLP_D         [MB_LLM_NUM_X];
int64_t MB_LLM_REF_MHA_O_RES     [MB_LLM_NUM_X];
int64_t MB_LLM_REF_MLP_XD_RES    [MB_LLM_NUM_X];
int64_t MB_LLM_REF_CONDENSED_OD  [2 * MB_LLM_NUM_X];
int64_t MB_LLM_REF_RESIDUAL_O    [2 * MB_LLM_NUM_X];
int64_t MB_LLM_REF_WRITEBACK     [2 * MB_LLM_NUM_X];
int64_t MB_LLM_DUT_RESIDUAL_O    [2 * MB_LLM_NUM_X];
int64_t MB_LLM_DUT_WRITEBACK     [2 * MB_LLM_NUM_X];
int64_t MB_LLM_DUT_Y             [MB_LLM_NUM_X];

int64_t MB_VIT_REF_X             [MB_VIT_NUM_X];
int64_t MB_VIT_REF_MHA_X         [MB_VIT_NUM_X];
int64_t MB_VIT_REF_MHA_O         [MB_VIT_NUM_X];
int64_t MB_VIT_REF_MLP_X         [MB_VIT_NUM_X];
int64_t MB_VIT_REF_MLP_FC2       [MB_VIT_NUM_X];
int64_t MB_VIT_REF_MHA_O_RES     [MB_VIT_NUM_X];
int64_t MB_VIT_REF_MLP_FC2_RES   [MB_VIT_NUM_X];
int64_t MB_VIT_REF_CONDENSED_OFC2[2 * MB_VIT_NUM_X];
int64_t MB_VIT_REF_RESIDUAL_O    [2 * MB_VIT_NUM_X];
int64_t MB_VIT_REF_WRITEBACK     [2 * MB_VIT_NUM_X];
int64_t MB_VIT_DUT_RESIDUAL_O    [2 * MB_VIT_NUM_X];
int64_t MB_VIT_DUT_WRITEBACK     [2 * MB_VIT_NUM_X];
int64_t MB_VIT_DUT_Y             [MB_VIT_NUM_X];

void mb_write_token_major_x(
    int64_t data[],
    int num_tokens,
    int num_ct,
    int channels,
    hls::stream<mb_residual_vec_t>& x_stream
) {
    // M_AXI replay：按 token -> CT 顺序供 Pre-LN 输出。
    for (int t = 0; t < num_tokens; ++t) {
        for (int ct = 0; ct < num_ct; ++ct) {
            mb_residual_vec_t vec;
            for (int cp = 0; cp < MB_CP; ++cp) {
                vec[cp] = (REUSE_X_T)data[t * channels + ct * MB_CP + cp];
            }
            x_stream.write(vec);
        }
    }
}

void mb_write_llm_delta_order_x(int64_t data[], hls::stream<mb_residual_vec_t>& x_stream) {
    // M_AXI replay：按 LLM delta 顺序 CT -> token 重放同一份 state，用于 x + delta。
    for (int ct = 0; ct < MB_LLM_CT; ++ct) {
        for (int t = 0; t < MB_LLM_T; ++t) {
            mb_residual_vec_t vec;
            for (int cp = 0; cp < MB_CP; ++cp) {
                vec[cp] = (REUSE_X_T)data[t * MB_LLM_C + ct * MB_CP + cp];
            }
            x_stream.write(vec);
        }
    }
}

void mb_write_vit_delta_order_x(int64_t data[], hls::stream<mb_residual_vec_t>& x_stream) {
    // M_AXI replay：按 ViT DEMUX_OFC2 顺序 TT_D -> CT -> TP 重放 state，用于 x + delta。
    for (int tt_d = 0; tt_d < MB_VIT_TT_D; ++tt_d) {
        for (int ct = 0; ct < MB_VIT_CT; ++ct) {
            for (int tp = 0; tp < MB_VIT_DEMUX_TP; ++tp) {
                int token = tt_d * MB_VIT_DEMUX_TP + tp;
                mb_residual_vec_t vec;
                for (int cp = 0; cp < MB_CP; ++cp) {
                    vec[cp] = (REUSE_X_T)data[token * MB_VIT_C + ct * MB_CP + cp];
                }
                x_stream.write(vec);
            }
        }
    }
}

void mb_read_llm_writeback(hls::stream<mb_residual_vec_t>& y_stream, int64_t out[2 * MB_LLM_NUM_X]) {
    // y_stream 是 CT -> token 顺序，读回后恢复成 pass-major token-major 数组便于和 golden 比较。
    for (int pass = 0; pass < 2; ++pass) {
        for (int ct = 0; ct < MB_LLM_CT; ++ct) {
            for (int t = 0; t < MB_LLM_T; ++t) {
                mb_residual_vec_t vec = y_stream.read();
                for (int cp = 0; cp < MB_CP; ++cp) {
                    out[pass * MB_LLM_NUM_X + t * MB_LLM_C + ct * MB_CP + cp] = vec[cp];
                }
            }
        }
    }
}

void mb_read_vit_writeback(hls::stream<mb_residual_vec_t>& y_stream, int64_t out[2 * MB_VIT_NUM_X]) {
    // y_stream 是 TT_D -> CT -> TP 顺序，读回后恢复成 pass-major token-major 数组。
    for (int pass = 0; pass < 2; ++pass) {
        for (int tt_d = 0; tt_d < MB_VIT_TT_D; ++tt_d) {
            for (int ct = 0; ct < MB_VIT_CT; ++ct) {
                for (int tp = 0; tp < MB_VIT_DEMUX_TP; ++tp) {
                    int token = tt_d * MB_VIT_DEMUX_TP + tp;
                    mb_residual_vec_t vec = y_stream.read();
                    for (int cp = 0; cp < MB_CP; ++cp) {
                        out[pass * MB_VIT_NUM_X + token * MB_VIT_C + ct * MB_CP + cp] = vec[cp];
                    }
                }
            }
        }
    }
}

void mb_prepare_llm_data(
    int l,
    int l_begin,
    const string& binaries_path,
    const string& condense_path,
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_i_stream
) {
    auto condensed_od = read_tensor<int64_t>(condense_path + "/CONDENSED_DEMUX_OD.bin");
    tensor2array<int64_t>(condensed_od, MB_LLM_REF_CONDENSED_OD, 1, 1, 1, 1, 2 * MB_LLM_NUM_X, 2 * MB_LLM_NUM_X);

    auto mha_x      = read_tensor<int64_t>(binaries_path + "/MHA_X.bin");
    auto mha_o      = read_tensor<int64_t>(binaries_path + "/MHA_O.bin");
    auto mha_o_res  = read_tensor<int64_t>(binaries_path + "/MHA_O_RES.bin");
    auto mlp_x      = read_tensor<int64_t>(binaries_path + "/MLP_X.bin");
    auto mlp_xd     = read_tensor<int64_t>(binaries_path + "/MLP_XD.bin");
    auto mlp_xd_res = read_tensor<int64_t>(binaries_path + "/MLP_XD_RES.bin");

    // LLM residual 参考按当前 binaries 文件长度推导 T_LOAD，避免旧 612 常量造成窗口错位。
    tensor2array_dynamic_t<int64_t>(mha_x,      MB_LLM_REF_MHA_X,      1, 1, MB_LLM_POS, MB_LLM_T, MB_LLM_C, MB_LLM_C, "MHA_X");
    tensor2array_dynamic_t<int64_t>(mha_o,      MB_LLM_REF_MHA_O,      1, 1, MB_LLM_POS, MB_LLM_T, MB_LLM_C, MB_LLM_C, "MHA_O");
    tensor2array_dynamic_t<int64_t>(mlp_x,      MB_LLM_REF_MLP_X,      1, 1, MB_LLM_POS, MB_LLM_T, MB_LLM_C, MB_LLM_C, "MLP_X");
    tensor2array_dynamic_t<int64_t>(mlp_xd,     MB_LLM_REF_MLP_D,      1, 1, MB_LLM_POS, MB_LLM_T, MB_LLM_C, MB_LLM_C, "MLP_XD");
    tensor2array_dynamic_t<int64_t>(mha_o_res,  MB_LLM_REF_MHA_O_RES,  1, 1, MB_LLM_POS, MB_LLM_T, MB_LLM_C, MB_LLM_C, "MHA_O_RES");
    tensor2array_dynamic_t<int64_t>(mlp_xd_res, MB_LLM_REF_MLP_XD_RES, 1, 1, MB_LLM_POS, MB_LLM_T, MB_LLM_C, MB_LLM_C, "MLP_XD_RES");

    for (int t = 0; t < MB_LLM_T; ++t) {
        for (int c = 0; c < MB_LLM_C; ++c) {
            assert(MB_LLM_REF_MHA_X[t * MB_LLM_C + c] + MB_LLM_REF_MHA_O[t * MB_LLM_C + c] == MB_LLM_REF_MHA_O_RES[t * MB_LLM_C + c]);
            assert(MB_LLM_REF_MLP_X[t * MB_LLM_C + c] + MB_LLM_REF_MLP_D[t * MB_LLM_C + c] == MB_LLM_REF_MLP_XD_RES[t * MB_LLM_C + c]);
            MB_LLM_REF_RESIDUAL_O[0 * MB_LLM_NUM_X + t * MB_LLM_C + c] = MB_LLM_REF_MHA_X[t * MB_LLM_C + c];
            MB_LLM_REF_RESIDUAL_O[1 * MB_LLM_NUM_X + t * MB_LLM_C + c] = MB_LLM_REF_MLP_X[t * MB_LLM_C + c];
            MB_LLM_REF_WRITEBACK [0 * MB_LLM_NUM_X + t * MB_LLM_C + c] = MB_LLM_REF_MHA_O_RES[t * MB_LLM_C + c];
            MB_LLM_REF_WRITEBACK [1 * MB_LLM_NUM_X + t * MB_LLM_C + c] = MB_LLM_REF_MLP_XD_RES[t * MB_LLM_C + c];
            MB_LLM_REF_Y[t * MB_LLM_C + c] = MB_LLM_REF_MLP_XD_RES[t * MB_LLM_C + c];
        }
    }

    array2stream<int64_t, REUSE_X_T, 1, 1, 1, 1, 2 * MB_LLM_NUM_X, MB_CP>(MB_LLM_REF_CONDENSED_OD, res_i_stream, "LLM CONDENSED OD", true);
    save_condensed_tensor<int64_t, REUSE_X_T, 2 * MB_LLM_NUM_X, MB_CP>(condense_path + "/CONDENSED_RESIDUAL_I.bin", res_i_stream);

    if (l == l_begin) {
        mb_write_token_major_x(MB_LLM_REF_MHA_X, MB_LLM_T, MB_LLM_CT, MB_LLM_C, x_stream);
        mb_write_llm_delta_order_x(MB_LLM_REF_MHA_X, x_stream);
        mb_write_token_major_x(MB_LLM_REF_MLP_X, MB_LLM_T, MB_LLM_CT, MB_LLM_C, x_stream);
        mb_write_llm_delta_order_x(MB_LLM_REF_MLP_X, x_stream);
        save_tensor<int64_t>(condense_path + "/CONDENSED_RESIDUAL_X.bin", MB_LLM_REF_MHA_X, MB_LLM_NUM_X);
    }
}

void mb_compare_llm_data(
    int l,
    int l_close,
    const string& condense_path,
    hls::stream<mb_residual_vec_t>& res_o_stream,
    hls::stream<mb_residual_vec_t>& y_stream
) {
    hls::stream<mb_residual_vec_t> sim_res_o_stream("llm_sim_res_o_stream");
    stream2stream<REUSE_X_T, 2 * MB_LLM_NUM_X, MB_CP>(res_o_stream, sim_res_o_stream);
    save_condensed_tensor<int64_t, REUSE_X_T, 2 * MB_LLM_NUM_X, MB_CP>(condense_path + "/CONDENSED_RESIDUAL_O.bin", sim_res_o_stream);
    stream2array<int64_t, REUSE_X_T, 2, MB_LLM_T, MB_TP, MB_LLM_C, MB_CP>(sim_res_o_stream, MB_LLM_DUT_RESIDUAL_O, "LLM RESIDUAL O", true);
    assert(sim_res_o_stream.size() == 0);
    compare<int64_t>(MB_LLM_REF_RESIDUAL_O, MB_LLM_DUT_RESIDUAL_O, 2 * MB_LLM_NUM_X, "LLM RESIDUAL O");

    if (l == l_close - 1) {
        hls::stream<mb_residual_vec_t> sim_y_stream("llm_sim_y_stream");
        stream2stream<REUSE_X_T, 2 * MB_LLM_NUM_X, MB_CP>(y_stream, sim_y_stream);
        save_condensed_tensor<int64_t, REUSE_X_T, 2 * MB_LLM_NUM_X, MB_CP>(condense_path + "/CONDENSED_RESIDUAL_WRITEBACK.bin", sim_y_stream);
        mb_read_llm_writeback(sim_y_stream, MB_LLM_DUT_WRITEBACK);
        compare<int64_t>(MB_LLM_REF_WRITEBACK, MB_LLM_DUT_WRITEBACK, 2 * MB_LLM_NUM_X, "LLM RESIDUAL WRITEBACK");
        for (int i = 0; i < MB_LLM_NUM_X; ++i) {
            MB_LLM_DUT_Y[i] = MB_LLM_DUT_WRITEBACK[MB_LLM_NUM_X + i];
        }
        save_tensor<int64_t>(condense_path + "/CONDENSED_RESIDUAL_Y.bin", MB_LLM_DUT_Y, MB_LLM_NUM_X);
        compare<int64_t>(MB_LLM_REF_Y, MB_LLM_DUT_Y, MB_LLM_NUM_X, "LLM Y");
    }
}

void mb_test_llm_layer(int l_begin, int l_close) {
    hls::stream<mb_residual_vec_t> x_stream("LLM X REPLAY");
    hls::stream<mb_residual_vec_t> res_i_stream("LLM RESIDUAL I");
    hls::stream<mb_residual_vec_t> res_o_stream("LLM RESIDUAL O");
    hls::stream<mb_residual_vec_t> y_stream("LLM WRITEBACK");

    for (int l = l_begin; l < l_close && l < LLAMA_L; ++l) {
        mb_prepare_llm_data(l, l_begin, BINARIES_PATH + to_string(l), CONDENSE_PATH + to_string(l), x_stream, res_i_stream);
    }

    top(MODE_LLM, l_begin, l_close, x_stream, res_i_stream, res_o_stream, y_stream);

    for (int l = l_begin; l < l_close && l < LLAMA_L; ++l) {
        mb_compare_llm_data(l, l_close, CONDENSE_PATH + to_string(l), res_o_stream, y_stream);
    }
}

void mb_prepare_vit_data(
    int l,
    int l_begin,
    const string& binaries_path,
    const string& condense_path,
    hls::stream<mb_residual_vec_t>& x_stream,
    hls::stream<mb_residual_vec_t>& res_i_stream
) {
    auto condensed_ofc2 = read_tensor<int64_t>(condense_path + "/CONDENSED_DEMUX_OFC2.bin");
    tensor2array<int64_t>(condensed_ofc2, MB_VIT_REF_CONDENSED_OFC2, 1, 1, 1, 1, 2 * MB_VIT_NUM_X, 2 * MB_VIT_NUM_X);

    auto mha_x        = read_tensor<int64_t>(binaries_path + "/MHA_LN_X.bin");
    auto mha_o        = read_tensor<int64_t>(binaries_path + "/MHA_O.bin");
    auto mha_o_res    = read_tensor<int64_t>(binaries_path + "/MHA_O_RES.bin");
    auto mlp_x        = read_tensor<int64_t>(binaries_path + "/MLP_LN_X.bin");
    auto mlp_xfc2     = read_tensor<int64_t>(binaries_path + "/MLP_XFC2.bin");
    auto mlp_xfc2_res = read_tensor<int64_t>(binaries_path + "/MLP_XFC2_RES.bin");

    tensor2array<int64_t>(mha_x,        MB_VIT_REF_MHA_X,        1, 1, MB_VIT_T_LOAD, MB_VIT_POS, MB_VIT_T, MB_VIT_C, MB_VIT_C);
    tensor2array<int64_t>(mha_o,        MB_VIT_REF_MHA_O,        1, 1, MB_VIT_T_LOAD, MB_VIT_POS, MB_VIT_T, MB_VIT_C, MB_VIT_C);
    tensor2array<int64_t>(mlp_x,        MB_VIT_REF_MLP_X,        1, 1, MB_VIT_T_LOAD, MB_VIT_POS, MB_VIT_T, MB_VIT_C, MB_VIT_C);
    tensor2array<int64_t>(mlp_xfc2,     MB_VIT_REF_MLP_FC2,      1, 1, MB_VIT_T_LOAD, MB_VIT_POS, MB_VIT_T, MB_VIT_C, MB_VIT_C);
    tensor2array<int64_t>(mha_o_res,    MB_VIT_REF_MHA_O_RES,    1, 1, MB_VIT_T_LOAD, MB_VIT_POS, MB_VIT_T, MB_VIT_C, MB_VIT_C);
    tensor2array<int64_t>(mlp_xfc2_res, MB_VIT_REF_MLP_FC2_RES,  1, 1, MB_VIT_T_LOAD, MB_VIT_POS, MB_VIT_T, MB_VIT_C, MB_VIT_C);

    for (int t = 0; t < MB_VIT_T; ++t) {
        for (int c = 0; c < MB_VIT_C; ++c) {
            assert(MB_VIT_REF_MHA_X[t * MB_VIT_C + c] + MB_VIT_REF_MHA_O[t * MB_VIT_C + c] == MB_VIT_REF_MHA_O_RES[t * MB_VIT_C + c]);
            assert(MB_VIT_REF_MHA_O_RES[t * MB_VIT_C + c] == MB_VIT_REF_MLP_X[t * MB_VIT_C + c]);
            assert(MB_VIT_REF_MLP_X[t * MB_VIT_C + c] + MB_VIT_REF_MLP_FC2[t * MB_VIT_C + c] == MB_VIT_REF_MLP_FC2_RES[t * MB_VIT_C + c]);
            MB_VIT_REF_RESIDUAL_O[0 * MB_VIT_NUM_X + t * MB_VIT_C + c] = MB_VIT_REF_MHA_X[t * MB_VIT_C + c];
            MB_VIT_REF_RESIDUAL_O[1 * MB_VIT_NUM_X + t * MB_VIT_C + c] = MB_VIT_REF_MLP_X[t * MB_VIT_C + c];
            MB_VIT_REF_WRITEBACK [0 * MB_VIT_NUM_X + t * MB_VIT_C + c] = MB_VIT_REF_MHA_O_RES[t * MB_VIT_C + c];
            MB_VIT_REF_WRITEBACK [1 * MB_VIT_NUM_X + t * MB_VIT_C + c] = MB_VIT_REF_MLP_FC2_RES[t * MB_VIT_C + c];
            MB_VIT_REF_X[t * MB_VIT_C + c] = MB_VIT_REF_MLP_FC2_RES[t * MB_VIT_C + c];
        }
    }

    array2stream<int64_t, REUSE_X_T, 1, 1, 1, 1, 2 * MB_VIT_NUM_X, MB_CP>(MB_VIT_REF_CONDENSED_OFC2, res_i_stream, "ViT CONDENSED OFC2", true);
    save_condensed_tensor<int64_t, REUSE_X_T, 2 * MB_VIT_NUM_X, MB_CP>(condense_path + "/CONDENSED_RESIDUAL_I.bin", res_i_stream);

    if (l == l_begin) {
        mb_write_token_major_x(MB_VIT_REF_MHA_X, MB_VIT_T, MB_VIT_CT, MB_VIT_C, x_stream);
        mb_write_vit_delta_order_x(MB_VIT_REF_MHA_X, x_stream);
        mb_write_token_major_x(MB_VIT_REF_MLP_X, MB_VIT_T, MB_VIT_CT, MB_VIT_C, x_stream);
        mb_write_vit_delta_order_x(MB_VIT_REF_MLP_X, x_stream);
        save_tensor<int64_t>(condense_path + "/CONDENSED_RESIDUAL_X.bin", MB_VIT_REF_MHA_X, MB_VIT_NUM_X);
    }
}

void mb_compare_vit_data(
    int l,
    int l_close,
    const string& condense_path,
    hls::stream<mb_residual_vec_t>& res_o_stream,
    hls::stream<mb_residual_vec_t>& y_stream
) {
    hls::stream<mb_residual_vec_t> sim_res_o_stream("vit_sim_res_o_stream");
    stream2stream<REUSE_X_T, 2 * MB_VIT_NUM_X, MB_CP>(res_o_stream, sim_res_o_stream);
    save_condensed_tensor<int64_t, REUSE_X_T, 2 * MB_VIT_NUM_X, MB_CP>(condense_path + "/CONDENSED_RESIDUAL_O.bin", sim_res_o_stream);
    stream2array<int64_t, REUSE_X_T, 2, MB_VIT_T, MB_TP, MB_VIT_C, MB_CP>(sim_res_o_stream, MB_VIT_DUT_RESIDUAL_O, "ViT RESIDUAL O", true);
    assert(sim_res_o_stream.size() == 0);
    compare<int64_t>(MB_VIT_REF_RESIDUAL_O, MB_VIT_DUT_RESIDUAL_O, 2 * MB_VIT_NUM_X, "ViT RESIDUAL O");

    if (l == l_close - 1) {
        hls::stream<mb_residual_vec_t> sim_y_stream("vit_sim_y_stream");
        stream2stream<REUSE_X_T, 2 * MB_VIT_NUM_X, MB_CP>(y_stream, sim_y_stream);
        save_condensed_tensor<int64_t, REUSE_X_T, 2 * MB_VIT_NUM_X, MB_CP>(condense_path + "/CONDENSED_RESIDUAL_WRITEBACK.bin", sim_y_stream);
        mb_read_vit_writeback(sim_y_stream, MB_VIT_DUT_WRITEBACK);
        compare<int64_t>(MB_VIT_REF_WRITEBACK, MB_VIT_DUT_WRITEBACK, 2 * MB_VIT_NUM_X, "ViT RESIDUAL WRITEBACK");
        for (int i = 0; i < MB_VIT_NUM_X; ++i) {
            MB_VIT_DUT_Y[i] = MB_VIT_DUT_WRITEBACK[MB_VIT_NUM_X + i];
        }
        save_tensor<int64_t>(condense_path + "/CONDENSED_RESIDUAL_Y.bin", MB_VIT_DUT_Y, MB_VIT_NUM_X);
        compare<int64_t>(MB_VIT_REF_X, MB_VIT_DUT_Y, MB_VIT_NUM_X, "ViT Y");
    }
}

void mb_test_vit_layer(int l_begin, int l_close) {
    hls::stream<mb_residual_vec_t> x_stream("ViT X REPLAY");
    hls::stream<mb_residual_vec_t> res_i_stream("ViT RESIDUAL I");
    hls::stream<mb_residual_vec_t> res_o_stream("ViT RESIDUAL O");
    hls::stream<mb_residual_vec_t> y_stream("ViT WRITEBACK");

    for (int l = l_begin; l < l_close && l < VIT_L; ++l) {
        mb_prepare_vit_data(l, l_begin, VIT_BINARIES_PATH + to_string(l), VIT_CONDENSE_PATH + to_string(l), x_stream, res_i_stream);
    }

    top(MODE_VIT, l_begin, l_close, x_stream, res_i_stream, res_o_stream, y_stream);

    for (int l = l_begin; l < l_close && l < VIT_L; ++l) {
        mb_compare_vit_data(l, l_close, VIT_CONDENSE_PATH + to_string(l), res_o_stream, y_stream);
    }
}

int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        mb_test_llm_layer(0, 1);
        printf("LLM mode layer 0 passed\n");
    }

    mb_test_vit_layer(vit_layer, vit_layer + 1);
    printf("ViT mode layer %d passed\n", vit_layer);

    return 0;
}
