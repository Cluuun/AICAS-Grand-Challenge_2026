#include "../src/reuse_common.h"
#include "../src/silu.h"
#include "../src/gelu.h"
#include "../src/utils.h"
#include <vector>

// ============================================================================
// SILU_GELU：LLM SwiGLU 与 ViT GELU 的 mode-aware 条件复用 IP
// ============================================================================
// LLM mode: DEMUX_UG -> split(U/G) -> SiLU(G) * U -> shared quantizer
// ViT mode: DEMUX_FC1 -> GELU -> shared quantizer
// 两种 mode 的算法前半段不同，但都会汇入 27-bit common activation stream，
// 再进入同一个 8-lane 动态量化函数，避免为 LLM/ViT 分别例化 Quantizer。
// ============================================================================

// simulation hyperparameters
constexpr int SG_LLM_POS    = 96;
// ViT 新导出按 batch*patch 保存，当前 vision_0/new_txt 与其他层同批为 13*1024。
constexpr int SG_VIT_T_LOAD = 13312;
constexpr int SG_VIT_POS    = 0;

// shared interface hyperparameters
constexpr int SG_CP      = REUSE_CP;
constexpr int SG_Q_BITS  = DW_AQ;
constexpr int SG_Q_MAX   = +(1 << (SG_Q_BITS - 1)) - 1;
constexpr int SG_Q_MIN   = -(1 << (SG_Q_BITS - 1));
constexpr int SG_A_SMAX  = 15;

// LLM MLP hyperparameters
constexpr int SG_LLM_T          = REUSE_LLM_TILE_T;
constexpr int SG_LLM_C          = LLAMA_CM;
constexpr int SG_LLM_CT         = SG_LLM_C / SG_CP;
constexpr int SG_LLM_NUM_UG     = 2 * SG_LLM_T * SG_LLM_C;
constexpr int SG_LLM_ACT_VECS   = SG_LLM_T * SG_LLM_C / SG_CP;

// ViT MLP hyperparameters
constexpr int SG_VIT_T          = VIT_S;
constexpr int SG_VIT_C          = VIT_MLP_DIM;
constexpr int SG_VIT_CG         = SG_VIT_C / VIT_G;
constexpr int SG_VIT_DEMUX_TP   = 8;
constexpr int SG_VIT_NUM_FC1    = SG_VIT_T * SG_VIT_C;
constexpr int SG_VIT_ACT_VECS   = SG_VIT_T * SG_VIT_C / SG_CP;

// 量化前统一中间类型：覆盖 LLM gate*up 截断值和 ViT GELU 输出值。
constexpr int SG_DW_ACT = const_max(DW_XM_TRUNC, VIT_DW_FC1_GELU);
typedef ap_int<SG_DW_ACT> SG_ACT_T;

// LLM 侧保留原 SILU_CP=1 的表访问并行度，避免把 SiLU LUT 扩成 8 份。
SILU<XUG_TRUNC_T, SILU_T, SG_LLM_T, 1, SG_LLM_C, 1, SG_CP> sg_silu_inst;

// LLM 输入来自 DEMUX_UG，顺序为 CT -> UG(U/G) -> token -> CP。
// 这里只做流拆分和位宽回落，保持 U 与 G 的相对顺序供后续 element-wise multiply 对齐。
void llm_split_ug(
    hls::stream<hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> > &ug_stream,
    hls::stream<hls::vector<XUG_TRUNC_T,        SG_CP> > &u_stream,
    hls::stream<hls::vector<XUG_TRUNC_T,        SG_CP> > &g_stream
) {
    for (int ct = 0; ct < SG_LLM_CT; ++ct) {
        for (int ug = 0; ug < 2; ++ug) {
            for (int t = 0; t < SG_LLM_T; ++t) {
                #pragma HLS pipeline II=1
                hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> in_vec = ug_stream.read();
                hls::vector<XUG_TRUNC_T, SG_CP> out_vec;
                for (int cp = 0; cp < SG_CP; ++cp) {
                    #pragma HLS unroll
                    out_vec[cp] = reuse_truncate<XUG_TRUNC_T>(in_vec[cp]);
                }
                if (ug == 0) {
                    u_stream.write(out_vec);
                } else {
                    g_stream.write(out_vec);
                }
            }
        }
    }
}

// LLM SwiGLU merge：SiLU(G) 与 U 同序相乘并右移到 XM 位宽，再扩展到 common activation。
void llm_merge_silu_u(
    hls::stream<hls::vector<XUG_TRUNC_T, SG_CP> > &u_stream,
    hls::stream<hls::vector<SILU_T,      SG_CP> > &silu_stream,
    hls::stream<hls::vector<SG_ACT_T,    SG_CP> > &act_stream
) {
    for (int ct = 0; ct < SG_LLM_CT; ++ct) {
        for (int t = 0; t < SG_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            hls::vector<XUG_TRUNC_T, SG_CP> u_vec    = u_stream.read();
            hls::vector<SILU_T,      SG_CP> silu_vec = silu_stream.read();
            hls::vector<SG_ACT_T,    SG_CP> act_vec;
            for (int cp = 0; cp < SG_CP; ++cp) {
                #pragma HLS unroll
                // LLM SwiGLU 的 21x21 乘法是本 IP 的 OOC 最差路径；显式加深 DSP
                // 流水，保持 II=1，同时避免单个 DSP 级联组合路径卡住 2.5ns。
                ap_int<XUG_TRUNC_T::width + SILU_T::width> xm_product = u_vec[cp] * silu_vec[cp];
                #pragma HLS bind_op variable=xm_product op=mul impl=dsp latency=3
                XM_T xm_val = xm_product >> MLP_TRUNC_MUL;
                act_vec[cp] = (SG_ACT_T)xm_val;
            }
            act_stream.write(act_vec);
        }
    }
}

// LLM 激活分支：保持原来的 split -> SiLU -> multiply 数据流，输出 common activation。
void llm_silu_em_to_common(
    hls::stream<hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> > &ug_stream,
    hls::stream<hls::vector<SG_ACT_T,           SG_CP> > &act_stream
) {
    #pragma HLS dataflow

    hls::stream<hls::vector<XUG_TRUNC_T, SG_CP> > u_stream("u_stream");
    hls::stream<hls::vector<XUG_TRUNC_T, SG_CP> > g_stream("g_stream");
    hls::stream<hls::vector<SILU_T,      SG_CP> > silu_stream("silu_stream");
    #pragma HLS stream variable=u_stream    depth=16
    #pragma HLS stream variable=g_stream    depth=16
    #pragma HLS stream variable=silu_stream depth=16

    llm_split_ug(ug_stream, u_stream, g_stream);
    sg_silu_inst.do_silu(g_stream, silu_stream);
    llm_merge_silu_u(u_stream, silu_stream, act_stream);
}

// 单元素 GELU 查表，使用 ViT 已冻结的主表 + shift 表，输出扩展到 common activation。
SG_ACT_T vit_gelu_one(VIT_FC1_TRUNC_T x) {
    #pragma HLS inline
    int lut_idx   = (x - GELU_ALPHA) >> GELU_LOG2DENOM;
    int lut_s_idx = (x - GELU_ALPHA) >> GELU_LOG2DENOM_S;
    lut_idx       = clamp(lut_idx,   0, GELU_ENTRIES   - 1);
    lut_s_idx     = clamp(lut_s_idx, 0, GELU_ENTRIES_S - 1);

    VIT_FC1_GELU_T gelu_val = GELU_TABLE[lut_idx] << GELU_TABLE_S[lut_s_idx];
    return (SG_ACT_T)gelu_val;
}

// ViT 激活分支：输入是 DEMUX_FC1 的线性展开顺序（TT_D -> CT -> TP -> CP）。
// GELU 是逐元素操作，量化也按每拍 8 个 channel 成组，因此保持输入拍顺序原样输出。
void vit_gelu_to_common(
    hls::stream<hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> > &fc1_stream,
    hls::stream<hls::vector<SG_ACT_T,           SG_CP> > &act_stream
) {
    for (int vec = 0; vec < SG_VIT_ACT_VECS; ++vec) {
        #pragma HLS pipeline II=1
        hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> fc1_vec = fc1_stream.read();
        hls::vector<SG_ACT_T, SG_CP> act_vec;
        for (int cp = 0; cp < SG_CP; ++cp) {
            #pragma HLS unroll
            VIT_FC1_TRUNC_T x = reuse_truncate<VIT_FC1_TRUNC_T>(fc1_vec[cp]);
            act_vec[cp] = vit_gelu_one(x);
        }
        act_stream.write(act_vec);
    }
}

// mode 只在激活分支粒度选择算法；输出统一为 common activation，后级量化单实例复用。
void shared_activation_to_common(
    REUSE_MODE_T mode,
    hls::stream<hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> > &mlp_i_stream,
    hls::stream<hls::vector<SG_ACT_T,           SG_CP> > &act_stream
) {
    if (is_vit_mode(mode)) {
        vit_gelu_to_common(mlp_i_stream, act_stream);
    } else {
        llm_silu_em_to_common(mlp_i_stream, act_stream);
    }
}

// 共享动态量化器：固定 CP=G=8，每拍产生 8 个 q 和 1 个 scale。
// total_vec 在 layer 粒度由 mode 选择，量化 datapath 本身不按 mode 分裂。
void shared_mlp_quantize(
    int total_vec,
    hls::stream<hls::vector<SG_ACT_T,     SG_CP> > &act_stream,
    hls::stream<hls::vector<REUSE_AQ_T,   SG_CP> > &q_stream,
    hls::stream<hls::vector<REUSE_AS_T,   1    > > &s_stream
) {
    for (int vec = 0; vec < total_vec; ++vec) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=2560 max=393216
        hls::vector<SG_ACT_T, SG_CP> act_vec = act_stream.read();
        hls::vector<REUSE_AQ_T, SG_CP> q_vec;
        hls::vector<REUSE_AS_T, 1> s_vec;

        SG_ACT_T abs_max = 0;
        for (int cp = 0; cp < SG_CP; ++cp) {
            #pragma HLS unroll
            abs_max = max(abs_max, (SG_ACT_T)abs(act_vec[cp]));
        }

        int8_t s_val_i = log2ceil(abs_max) - (SG_Q_BITS - 1);
        REUSE_AS_T s_val = clamp(s_val_i, 0, SG_A_SMAX);
        s_vec[0] = s_val;

        for (int cp = 0; cp < SG_CP; ++cp) {
            #pragma HLS unroll
            SG_ACT_T q_val = act_vec[cp];
            if (s_val != 0) {
                q_val = q_val >> (s_val - 1);
                q_val = q_val + 1;
                q_val = q_val >> 1;
            }
            // shared activation quantizer 与软件一致做 signed int8 双向饱和。
            q_vec[cp] = clamp(q_val, (SG_ACT_T)SG_Q_MIN, (SG_ACT_T)SG_Q_MAX);
        }

        q_stream.write(q_vec);
        s_stream.write(s_vec);
    }
}

// top function
void top(
    // scalar inputs
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    // streams
    hls::stream<hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> > &mlp_i_stream,
    hls::stream<hls::vector<REUSE_AQ_T,         SG_CP> > &q_stream,
    hls::stream<hls::vector<REUSE_AS_T,         1    > > &s_stream
) {
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface ap_none port=mode
    #pragma HLS interface ap_none port=l_begin
    #pragma HLS interface ap_none port=l_close
    #pragma HLS interface axis port=mlp_i_stream
    #pragma HLS interface axis port=q_stream
    #pragma HLS interface axis port=s_stream

    #pragma HLS aggregate variable=mlp_i_stream compact=bit
    #pragma HLS aggregate variable=q_stream     compact=bit
    #pragma HLS aggregate variable=s_stream     compact=bit

    bool vit = is_vit_mode(mode);
    int layer_limit = reuse_layer_limit(mode);
    int total_vec = vit ? SG_VIT_ACT_VECS : SG_LLM_ACT_VECS;

    for (int l = l_begin; l < l_close && l < layer_limit; ++l) {
        hls::stream<hls::vector<SG_ACT_T, SG_CP> > act_stream("act_stream");
        #pragma HLS stream variable=act_stream depth=32
        #pragma HLS dataflow

        shared_activation_to_common(mode, mlp_i_stream, act_stream);
        shared_mlp_quantize(total_vec, act_stream, q_stream, s_stream);
    }
}

void test_llm_layer(int l) {
    string file_path = BINARIES_PATH + to_string(l);
    string save_path = CONDENSE_PATH + to_string(l);

    std::vector<int8_t> ref_xm_q(SG_LLM_T * SG_LLM_C);
    std::vector<int8_t> ref_xm_s(SG_LLM_T * SG_LLM_CT);
    std::vector<int8_t> dut_xm_q(SG_LLM_T * SG_LLM_C);
    std::vector<int8_t> dut_xm_s(SG_LLM_T * SG_LLM_CT);

    // LLM 参考来自软件量化输出，按 POS 截取当前 8-token decode tile。
    auto ref_q_tensor = read_tensor<int8_t>(file_path + "/MLP_XM_Q.bin");
    auto ref_s_tensor = read_tensor<int8_t>(file_path + "/MLP_XM_S.bin");
    tensor2array_dynamic_t<int8_t>(ref_q_tensor, ref_xm_q.data(), 1, 1, SG_LLM_POS, SG_LLM_T, SG_LLM_C,  SG_LLM_C,  "MLP_XM_Q");
    tensor2array_dynamic_t<int8_t>(ref_s_tensor, ref_xm_s.data(), 1, 1, SG_LLM_POS, SG_LLM_T, SG_LLM_CT, SG_LLM_CT, "MLP_XM_S");

    hls::stream<hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> > mlp_i_stream("llm_ug_stream");
    hls::stream<hls::vector<REUSE_AQ_T,         SG_CP> > q_stream("llm_q_stream");
    hls::stream<hls::vector<REUSE_AS_T,         1    > > s_stream("llm_s_stream");

    // 输入使用 DEMUX 已生成的 UG condense 文件，保持 CT -> UG -> token -> CP 的线性顺序。
    auto condensed_ug = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_UG.bin");
    assert(condensed_ug.size() >= (size_t)SG_LLM_NUM_UG);
    array2stream<int64_t, REUSE_MLP1_TRUNC_T, 1, 1, 1, 1, SG_LLM_NUM_UG, SG_CP>(
        condensed_ug.data(), mlp_i_stream, "LLM CONDENSED UG", true
    );

    top(MODE_LLM, l, l + 1, mlp_i_stream, q_stream, s_stream);

    // 输出文件名保持 LLM 历史命名，供 MUX/下游模块直接读取。
    save_condensed_tensor<int8_t, REUSE_AQ_T, SG_LLM_T * SG_LLM_C,  SG_CP>(save_path + "/CONDENSED_SILU_EM_QUANT_XM_Q.bin", q_stream);
    save_condensed_tensor<int8_t, REUSE_AS_T, SG_LLM_T * SG_LLM_CT, 1    >(save_path + "/CONDENSED_SILU_EM_QUANT_XM_S.bin", s_stream);

    stream2array_unpack<int8_t, REUSE_AQ_T, 1, SG_LLM_T, SG_LLM_T, SG_LLM_C,  SG_CP>(q_stream, dut_xm_q.data(), "LLM XM_Q", true);
    stream2array_unpack<int8_t, REUSE_AS_T, 1, SG_LLM_T, SG_LLM_T, SG_LLM_CT, 1    >(s_stream, dut_xm_s.data(), "LLM XM_S", true);

    assert(mlp_i_stream.empty());
    assert(q_stream.empty());
    assert(s_stream.empty());

    compare<int8_t>(ref_xm_q.data(), dut_xm_q.data(), SG_LLM_T * SG_LLM_C,  "LLM XM_Q");
    compare<int8_t>(ref_xm_s.data(), dut_xm_s.data(), SG_LLM_T * SG_LLM_CT, "LLM XM_S");
}

void test_vit_layer(int l) {
    string file_path = VIT_BINARIES_PATH + to_string(l);
    string save_path = VIT_CONDENSE_PATH + to_string(l);

    std::vector<int8_t> ref_xm_q(SG_VIT_T * SG_VIT_C);
    std::vector<int8_t> ref_xm_s(SG_VIT_T * SG_VIT_CG);
    std::vector<int8_t> dut_xm_q(SG_VIT_T * SG_VIT_C);
    std::vector<int8_t> dut_xm_s(SG_VIT_T * SG_VIT_CG);

    // ViT 参考来自软件 GELU+Quant 输出，取第一张图的 1024 patches。
    auto ref_q_tensor = read_tensor<int8_t>(file_path + "/MLP_XM_Q.bin");
    auto ref_s_tensor = read_tensor<int8_t>(file_path + "/MLP_XM_S.bin");
    assert(ref_q_tensor.size() == (size_t)(SG_VIT_T_LOAD * SG_VIT_C));
    assert(ref_s_tensor.size() == (size_t)(SG_VIT_T_LOAD * SG_VIT_CG));
    tensor2array<int8_t>(ref_q_tensor, ref_xm_q.data(), 1, 1, SG_VIT_T_LOAD, SG_VIT_POS, SG_VIT_T, SG_VIT_C,  SG_VIT_C);
    tensor2array<int8_t>(ref_s_tensor, ref_xm_s.data(), 1, 1, SG_VIT_T_LOAD, SG_VIT_POS, SG_VIT_T, SG_VIT_CG, SG_VIT_CG);

    hls::stream<hls::vector<REUSE_MLP1_TRUNC_T, SG_CP> > mlp_i_stream("vit_fc1_stream");
    hls::stream<hls::vector<REUSE_AQ_T,         SG_CP> > q_stream("vit_q_stream");
    hls::stream<hls::vector<REUSE_AS_T,         1    > > s_stream("vit_s_stream");

    // 输入使用 DEMUX_FC1 condense 文件，保持 TT_D -> CT -> TP -> CP 的线性顺序。
    auto condensed_fc1 = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_FC1.bin");
    assert(condensed_fc1.size() >= (size_t)SG_VIT_NUM_FC1);
    array2stream<int64_t, REUSE_MLP1_TRUNC_T, 1, 1, 1, 1, SG_VIT_NUM_FC1, SG_CP>(
        condensed_fc1.data(), mlp_i_stream, "ViT CONDENSED FC1", true
    );

    top(MODE_VIT, l, l + 1, mlp_i_stream, q_stream, s_stream);

    // 输出文件名保持 ViT 历史命名，供 MUX/下游模块直接读取。
    save_condensed_tensor<int8_t, REUSE_AQ_T, SG_VIT_T * SG_VIT_C,  SG_CP>(save_path + "/CONDENSED_GELU_XM_Q.bin", q_stream);
    save_condensed_tensor<int8_t, REUSE_AS_T, SG_VIT_T * SG_VIT_CG, 1    >(save_path + "/CONDENSED_GELU_XM_S.bin", s_stream);

    stream2array_unpack<int8_t, REUSE_AQ_T, 1, SG_VIT_T, SG_VIT_DEMUX_TP, SG_VIT_C,  SG_CP>(q_stream, dut_xm_q.data(), "ViT XM_Q", true);
    stream2array_unpack<int8_t, REUSE_AS_T, 1, SG_VIT_T, SG_VIT_DEMUX_TP, SG_VIT_CG, 1    >(s_stream, dut_xm_s.data(), "ViT XM_S", true);

    assert(mlp_i_stream.empty());
    assert(q_stream.empty());
    assert(s_stream.empty());

    compare<int8_t>(ref_xm_q.data(), dut_xm_q.data(), SG_VIT_T * SG_VIT_C,  "ViT XM_Q");
    compare<int8_t>(ref_xm_s.data(), dut_xm_s.data(), SG_VIT_T * SG_VIT_CG, "ViT XM_S");
}

int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        test_llm_layer(0);
        printf("LLM layer 0 passed\n");
    }

    test_vit_layer(vit_layer);
    printf("ViT layer %d passed\n", vit_layer);

    return 0;
}
