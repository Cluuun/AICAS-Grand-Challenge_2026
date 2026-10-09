#ifndef __INT_VIT_RV_GEMM_H__
#define __INT_VIT_RV_GEMM_H__

// ViT RV_GEMM — 双向注意力，无外部 V Cache
// 设计考量：
// 1. T = T_BATCH（e.g. 8）为 R 处理的 mini-batch 大小，避免 R 方阵 buffer（1 MB/头）
// 2. S = S_VIT（全序列长度 1024），既是 R 的 key 维度，也是 V 的 token 数
// 3. Phase 1（每头）：将 S 个 V token 打包、转置、量化，存入内部 vq_cache_buf[HC][S]
// 4. Phase 2（每头 × ST_VIT mini-batch）：逐批缓冲 T 行 R，重复 V，BMM 产生 A
// 5. BMM H 维度 = H_VIT * ST_VIT，将 "头×mini-batch" 展平给通用 BMM 处理

#include "src/common.h"
#include "src/bmm.h"
#include "src/utils.h"
#include "src/quantizer.h"

template<
    class aq_t,   // 激活量化类型（R & V 量化后）
    class as_t,   // 激活 scale 类型
    class w_t,    // V 原始数据类型（量化前）
    class acc_t,  // 累加器类型
    class of_t,   // 输出截断类型

    int TRUNC,

    int H,    // 注意力头数（ViT = 12）
    int T,    // R mini-batch 查询 token 数（= T_BATCH，如 8）
    int TP,   // token 并行度
    int S,    // 全序列长度（= S_VIT = 1024）
    int SP,   // 序列并行度（= V 转置后的分组宽度）
    int HC,   // head channel 维度（= VIT_HC = 64）
    int CP    // head channel 并行度
>
class VIT_RV_GEMM{
public:
    VIT_RV_GEMM(){}

    static constexpr int TT      = T  / TP;          // mini-batch Q tiles
    static constexpr int ST      = S  / SP;           // key tiles
    static constexpr int CT      = HC / CP;           // output channel tiles (COT)
    static constexpr int ST_VIT  = S  / T;            // mini-batch 数量 (= 128)
    static constexpr int TT_V    = TT * ST_VIT;       // V 需要重复的次数 = S/TP

    static_assert(T  % TP == 0, "T  % TP  != 0");
    static_assert(S  % SP == 0, "S  % SP  != 0");
    static_assert(HC % CP == 0, "HC % CP  != 0");
    static_assert(S  % T  == 0, "S  % T   != 0 (S must be multiple of T_BATCH)");

    // BMM：将 H × ST_VIT 个 mini-batch 展平为 H*ST_VIT 个"头"
    // a=R[T][S], w=V^T[HC][S], o=A[T][HC]
    BMM<aq_t, as_t, aq_t, as_t, acc_t, of_t, TRUNC,
        H * ST_VIT,  // 展平后的"头"数
        T,   TP,     // R 查询 token
        S,   SP,     // R/V^T 的收缩维度（key 序列）
        HC,  CP      // V^T 的输出维度（head channel）
    > bmm_inst;

    // V 量化器：转置后 V 为 [HC][S]，按 HC 行 × S 列，SP 分组量化
    QUANTIZER<w_t, aq_t, as_t, 1, HC, 1, S, SP, SP> quantizer_inst;

    void do_vit_rv_gemm(
        hls::stream<hls::vector<aq_t, TP*SP > >& rq_stream,   // R quant（Softmax 输出）
        hls::stream<hls::vector<as_t, TP    > >& rs_stream,   // R scale
        hls::stream<hls::vector<w_t,     CP > >& v_stream,    // V 原始（未量化）
        hls::stream<hls::vector<of_t, TP*CP > >& o_stream     // 输出 A
    ){
        #pragma HLS dataflow

        // 内部跨头共享的 dataflow 流
        hls::stream<hls::vector<aq_t, SP*CP> > vq_stream_r("vq_stream_r");
        hls::stream<hls::vector<as_t, SP   > > vs_stream_r("vs_stream_r");
        hls::stream<hls::vector<aq_t, TP*SP> > rq_stream_r("rq_stream_r");
        hls::stream<hls::vector<as_t, TP   > > rs_stream_r("rs_stream_r");

        // ── Phase 1+2（V）：逐头填满转置量化，然后重复 TT_V 次 ───────────
        for(int h=0; h<H; ++h){
            w_t  v_buf_tmp   [S ][HC ];   // 原始 V [S_VIT][HC]
            aq_t vq_cache_buf[HC][S  ];   // 转置量化后 V^T [HC][S_VIT]
            as_t vs_cache_buf[HC][ST ];   // scale [HC][S_VIT/SP]

            #pragma HLS stream        variable=v_buf_tmp    type=pipo depth=2
            #pragma HLS stream        variable=vq_cache_buf type=pipo depth=2
            #pragma HLS stream        variable=vs_cache_buf type=pipo depth=2

            #pragma HLS array_reshape variable=v_buf_tmp    cyclic factor=CP  dim=2
            #pragma HLS array_reshape variable=vq_cache_buf cyclic factor=SP  dim=2
            #pragma HLS array_reshape variable=vs_cache_buf cyclic factor=CP  dim=1
            #pragma HLS array_reshape variable=vs_cache_buf cyclic factor=1   dim=2

            #pragma HLS bind_storage  variable=v_buf_tmp    type=ram_2p impl=bram
            #pragma HLS bind_storage  variable=vq_cache_buf type=ram_2p impl=uram
            #pragma HLS bind_storage  variable=vs_cache_buf type=ram_2p impl=bram

            // per-head 内部流
            hls::stream<hls::vector<w_t,  SP*1> > vt_stream ("vt_stream" );
            hls::stream<hls::vector<aq_t, SP*1> > vq_stream ("vq_stream" );
            hls::stream<hls::vector<as_t, 1   > > vs_q_stream("vs_q_stream");

            #pragma HLS dataflow
            // Phase 1a: 打包 V 原始数据 → v_buf_tmp[S][HC]（DEMUX 格式 [TT_D][CT][TP_D]×CP）
            pack_tokens_demux<w_t,    S,      HC,   CP, CP>(v_stream,  v_buf_tmp    );
            // Phase 1b: 转置 [S][HC] → vt_stream（HC 行 × S 列，SP 并行）
            transpose_tokens<w_t,     S,  SP, HC,    1>(v_buf_tmp,     vt_stream    );
            // Phase 1c: 量化 → vq_stream（int8）+ vs_q_stream（scale）
            quantizer_inst.do_quant                    (vt_stream,     vq_stream,  vs_q_stream);
            // Phase 1d: 填入 vq_cache_buf[HC][S] 和 vs_cache_buf[HC][ST]
            buffer_tokens  <aq_t,     HC,   1, S,  SP>(vq_stream,     vq_cache_buf );
            buffer_tokens  <as_t,     HC,   1, ST,  1>(vs_q_stream,   vs_cache_buf );
            // Phase 2: PIPO 完成后，重复 TT_V 次（= S_VIT/TP），产生 BMM 所需 V 流
            // vs_stream_r 需要 COP=CP 个 scale/clock，对应 BMM ws_stream<as_t, COP>
            // repeat_w_tokens<as_t, TT_V, T=HC, TP=CP, C=ST, CP=1>：
            //   每 clock 输出 TP×1=CP 个 scale：vs_cache_buf[cot×CP+0..CP-1][cit]
            repeat_w_tokens<aq_t, TT_V, HC, CP, S,  SP>(vq_cache_buf, vq_stream_r  );
            repeat_w_tokens<as_t, TT_V, HC, CP, ST,  1>(vs_cache_buf, vs_stream_r  );
        }

        // ── Phase 1+2（R）：逐头 × 逐 mini-batch，缓冲 T 行 R 并重复 CT 次 ──
        for(int h=0; h<H; ++h){
            for(int n=0; n<ST_VIT; ++n){
                aq_t rq_buf[T][S ];   // T_BATCH × S_VIT = 8 × 1024 = 8KB
                as_t rs_buf[T][ST];   // T_BATCH × S_VIT/SP

                #pragma HLS stream        variable=rq_buf type=pipo depth=2
                #pragma HLS stream        variable=rs_buf type=pipo depth=2

                #pragma HLS array_reshape variable=rq_buf cyclic factor=TP  dim=1
                #pragma HLS array_reshape variable=rq_buf cyclic factor=SP  dim=2
                #pragma HLS array_reshape variable=rs_buf cyclic factor=TP  dim=1
                #pragma HLS array_reshape variable=rs_buf cyclic factor=1   dim=2

                #pragma HLS bind_storage  variable=rq_buf type=ram_2p impl=lutram
                #pragma HLS bind_storage  variable=rs_buf type=ram_2p impl=lutram

                #pragma HLS dataflow
                // 填满 T 行 R（来自 Softmax 流）
                buffer_tokens  <aq_t,      T,  TP,  S,  SP>(rq_stream,  rq_buf      );
                buffer_tokens  <as_t,      T,  TP,  ST,  1>(rs_stream,  rs_buf      );
                // 为每个 V 输出通道组（CT = HC/CP 次）重复 R
                repeat_x_tokens<aq_t, CT, T, TP,   S,  SP>(rq_buf,     rq_stream_r );
                repeat_x_tokens<as_t, CT, T, TP,  ST,   1>(rs_buf,     rs_stream_r );
            }
        }

        bmm_inst.do_bmm(rq_stream_r, rs_stream_r, vq_stream_r, vs_stream_r, o_stream);
    }
};

#endif // __INT_VIT_RV_GEMM_H__
