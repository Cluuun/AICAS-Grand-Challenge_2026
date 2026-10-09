#ifndef __INT_VIT_QK_GEMM_H__
#define __INT_VIT_QK_GEMM_H__

// ViT QK_GEMM — 双向注意力，无外部 KV Cache，无 RoPE，无因果掩码
// 设计考量：
// 1. T == S == S_VIT（所有 1024 个 token 作为 query/key）
// 2. Phase 1：收满 Q[S][HC] 和 K[S][HC]（PIPO 保证先填后算）
// 3. Phase 2：repeat_x_tokens / repeat_w_tokens 生成 BMM 所需数据流
// 4. 输入为交织 QK 单流（[h][qk=Q/K][t][hct]），模块内拆分为 Q/K 两路

#include "src/common.h"
#include "src/bmm.h"
#include "src/utils.h"

template<
    class aq_t,   // 激活量化类型
    class as_t,   // 激活 scale 类型
    class acc_t,  // 累加器类型
    class of_t,   // 输出截断类型

    int TRUNC,

    int H,    // 注意力头数
    int T,    // token 总数（= S，双向注意力）
    int TP,   // token 并行度
    int S,    // key 序列长度（= T）
    int SP,   // key 序列并行度
    int HC,   // head channel 维度
    int CP    // head channel 并行度
>
class VIT_QK_GEMM{
public:
    VIT_QK_GEMM(){}

    static constexpr int TT  = T  / TP;
    static constexpr int ST  = S  / SP;
    static constexpr int HCT = HC / CP;

    static_assert(T  % TP == 0, "T  % TP  != 0");
    static_assert(S  % SP == 0, "S  % SP  != 0");
    static_assert(HC % CP == 0, "HC % CP  != 0");
    static_assert(S  % T == 0,  "ViT QK_GEMM requires S % T == 0 (S must be multiple of T_TILE)");

    // BMM: Q[H,T,HC] @ K^T[H,S,HC] → R[H,T,S]
    BMM<aq_t, as_t, aq_t, as_t, acc_t, of_t, TRUNC, H, T, TP, HC, CP, S, SP> bmm_inst;

    void do_vit_qk_gemm(
        hls::stream<hls::vector<aq_t, CP > >& qk_q_stream, // 交织 QK quant
        hls::stream<hls::vector<as_t,  1 > >& qk_s_stream, // 交织 QK scale
        hls::stream<hls::vector<of_t, TP*SP> >& o_stream  // 注意力得分输出
    ){
        #pragma HLS dataflow

        hls::stream<hls::vector<aq_t, CP   > > qq_stream("qq_stream");
        hls::stream<hls::vector<as_t, 1    > > qs_stream("qs_stream");
        hls::stream<hls::vector<aq_t, CP   > > kq_stream("kq_stream");
        hls::stream<hls::vector<as_t, 1    > > ks_stream("ks_stream");
        hls::stream<hls::vector<aq_t, TP*CP> > qq_stream_r("qq_stream_r");
        hls::stream<hls::vector<as_t, TP   > > qs_stream_r("qs_stream_r");
        hls::stream<hls::vector<aq_t, SP*CP> > kq_stream_r("kq_stream_r");
        hls::stream<hls::vector<as_t, SP   > > ks_stream_r("ks_stream_r");

        split_interleaved_streams<aq_t, H, T, 1, HC,  CP>(qk_q_stream, qq_stream, kq_stream);
        split_interleaved_streams<as_t, H, T, 1, HCT, 1 >(qk_s_stream, qs_stream, ks_stream);

        // ── Phase 1+2（Q）：逐头填满后 repeat ──────────────────────────
        for(int h=0; h<H; ++h){
            aq_t qq_buf[T][HC];
            as_t qs_buf[T][HCT];
            #pragma HLS stream        variable=qq_buf type=pipo depth=2
            #pragma HLS stream        variable=qs_buf type=pipo depth=2
            #pragma HLS array_reshape variable=qq_buf cyclic factor=TP  dim=1
            #pragma HLS array_reshape variable=qq_buf cyclic factor=CP  dim=2
            #pragma HLS array_reshape variable=qs_buf cyclic factor=TP  dim=1
            #pragma HLS array_reshape variable=qs_buf cyclic factor=1   dim=2
            #pragma HLS bind_storage  variable=qq_buf type=ram_2p impl=bram
            #pragma HLS bind_storage  variable=qs_buf type=ram_2p impl=bram

            #pragma HLS dataflow
            // Phase 1: 填满 Q buffer（读入该头的所有 S 个 token）
            // 使用 DEMUX 格式读取：流顺序为 [tt_d=T/CP][hct=HC/CP][t_i=CP]，TP_D=CP=8
            pack_tokens_demux<aq_t,      T,     HC,   CP, CP>(qq_stream,         qq_buf       );
            pack_tokens_demux<as_t,      T,    HCT,    1, CP>(qs_stream,         qs_buf       );
            // Phase 2: 为每个 K 列块（ST 次）重复 Q，产生 BMM 所需 Q 流
            repeat_x_tokens<aq_t, ST, T, TP, HC,   CP>(qq_buf,            qq_stream_r  );
            repeat_x_tokens<as_t, ST, T, TP, HCT,   1>(qs_buf,            qs_stream_r  );
        }

        // ── Phase 1+2（K）：逐头填满后 repeat ──────────────────────────
        for(int h=0; h<H; ++h){
            aq_t kq_buf[S][HC];
            as_t ks_buf[S][HCT];
            #pragma HLS stream        variable=kq_buf type=pipo depth=2
            #pragma HLS stream        variable=ks_buf type=pipo depth=2
            #pragma HLS array_reshape variable=kq_buf cyclic factor=CP  dim=2
            #pragma HLS array_reshape variable=ks_buf cyclic factor=1   dim=2
            #pragma HLS bind_storage  variable=kq_buf type=ram_2p impl=uram
            #pragma HLS bind_storage  variable=ks_buf type=ram_2p impl=bram

            #pragma HLS dataflow
            // Phase 1: 填满 K buffer（读入该头的所有 S 个 token）
            // 使用 DEMUX 格式读取：流顺序为 [tt_d=S/CP][hct=HC/CP][t_i=CP]，TP_D=CP=8
            pack_tokens_demux<aq_t,      S,     HC,   CP, CP>(kq_stream,         kq_buf       );
            pack_tokens_demux<as_t,      S,    HCT,    1, CP>(ks_stream,         ks_buf       );
            // Phase 2: 为每组 Q token tile（TT 次）重复 K，产生 BMM 所需 K 流
            repeat_w_tokens<aq_t, TT, S, SP, HC,   CP>(kq_buf,            kq_stream_r  );
            repeat_w_tokens<as_t, TT, S, SP, HCT,   1>(ks_buf,            ks_stream_r  );
        }

        bmm_inst.do_bmm(qq_stream_r, qs_stream_r, kq_stream_r, ks_stream_r, o_stream);
    }
};

#endif // __INT_VIT_QK_GEMM_H__
