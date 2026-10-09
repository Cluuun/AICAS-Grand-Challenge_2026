#ifndef __INT_VIT_SOFTMAX_H__
#define __INT_VIT_SOFTMAX_H__

// ViT Softmax — 双向注意力，无因果掩码
// 设计考量：
// 1. 去掉 LLM 版本的因果掩码条件 (chunk*T+t >= s)
// 2. 全行缓冲（R_BUF[TP][S]）：整行数据收满后计算 exp/sum/recip
// 3. LUT 参数从 ref/ViT/ 加载，SOFTMAX_TRUNC_MUL 等与 LLM 版本不同

#include "src/common.h"
#include "src/utils.h"
#include "src/adapter.h"

// ViT Softmax LUT 超参数（与 LLM 版本路径不同）
constexpr int VIT_SOFTMAX_HYPERPARAMS[] = {
    #include "ref/ViT/MHA_SOFTMAX_HYPERPARAMS.txt"
};
constexpr int VIT_EXP_LOG2DENOM    = VIT_SOFTMAX_HYPERPARAMS[0];
constexpr int VIT_EXP_ENTRIES      = VIT_SOFTMAX_HYPERPARAMS[1];
constexpr int VIT_RECIP_NUM_TABLES = VIT_SOFTMAX_HYPERPARAMS[2];
constexpr int VIT_RECIP_ENTRIES    = VIT_SOFTMAX_HYPERPARAMS[3];
constexpr int VIT_SOFTMAX_TRUNC_MUL= VIT_SOFTMAX_HYPERPARAMS[4];

constexpr int VIT_EXP_TABLE_DATA[] = {
    #include "ref/ViT/MHA_SOFTMAX_EXP_TABLE.txt"
};
constexpr int VIT_RECIP_TABLES_DATA[] = {
    #include "ref/ViT/MHA_SOFTMAX_RECIP_TABLES.txt"
};
constexpr int VIT_RECIP_ALPHAS_DATA[] = {
    #include "ref/ViT/MHA_SOFTMAX_RECIP_ALPHAS.txt"
};
constexpr int VIT_RECIP_LOG2DENOMS_DATA[] = {
    #include "ref/ViT/MHA_SOFTMAX_RECIP_LOG2DENOMS.txt"
};
constexpr int VIT_RECIP_OFFSETS_DIFF_DATA[] = {
    #include "ref/ViT/MHA_SOFTMAX_RECIP_OFFSETS_DIFF.txt"
};


// 树形归约：在 N 个元素之间做 pairwise max，深度 ceil(log2 N)，
// 完全展开后形成平衡比较树而非线性链
// 用结构体偏特化代替 `if constexpr`，避免依赖 -std=c++17
template<int N, class T>
struct VitTreeMax{
    static inline T eval(const T* arr){
        #pragma HLS inline
        T l = VitTreeMax<N/2,     T>::eval(arr      );
        T r = VitTreeMax<N - N/2, T>::eval(arr + N/2);
        return (l > r) ? l : r;
    }
};
template<class T>
struct VitTreeMax<1, T>{
    static inline T eval(const T* arr){
        #pragma HLS inline
        return arr[0];
    }
};
template<int N, class T>
inline T vit_tree_max(const T* arr){
    #pragma HLS inline
    return VitTreeMax<N, T>::eval(arr);
}


template<
    class r_t,          // 输入注意力得分类型（来自 QK_GEMM 截断后）
    class r_masked_t,   // ViT 中无掩码，但类型保持一致
    class exp_t,
    class exp_sum_t,
    class recip_t,
    class softmax_t,

    int H,
    int T,    // token 总数（= S_VIT）
    int TP,
    int S,    // key 序列长度（= T = S_VIT）
    int SAP,  // 输入流 adapter packing
    int SP    // softmax 内部并行度
>
class VIT_SOFTMAX{
public:
    static_assert(T % TP == 0, "T must be divisible by TP");
    static_assert(S % SP == 0, "S must be divisible by SP");

    static constexpr int TT = T / TP;
    static constexpr int ST = S / SP;

    // 内部缓冲（每次处理 TP 行，每行 S 个 key）
    r_masked_t  R_BUF    [TP][S];
    exp_t       R_EXP    [TP][S];
    r_masked_t  R_MAXVAL [TP];
    exp_sum_t   R_EXP_SUM[TP];
    recip_t     R_RECIP  [TP];

    VIT_SOFTMAX(){}

    void do_vit_softmax_func(
        hls::stream<hls::vector<r_t,       TP*SP> >& i_stream,
        hls::stream<hls::vector<softmax_t, TP*SP> >& o_stream
    ){
        #pragma HLS array_reshape variable=R_BUF     complete             dim=1
        #pragma HLS array_reshape variable=R_BUF     cyclic   factor=SP   dim=2
        #pragma HLS array_reshape variable=R_EXP     complete             dim=1
        #pragma HLS array_reshape variable=R_EXP     cyclic   factor=SP   dim=2
        #pragma HLS array_reshape variable=R_MAXVAL  complete             dim=1
        #pragma HLS array_reshape variable=R_EXP_SUM complete             dim=1
        #pragma HLS array_reshape variable=R_RECIP   complete             dim=1

        LOOP_H: for(int h=0; h<H; ++h){
            LOOP_TT: for(int tt=0; tt<TT; ++tt){

                // Pass 1：读入整行 S 个得分，同时维护 MAXVAL
                // ── 与 LLM 版本的唯一区别：去掉因果掩码条件 ──────────
                for(int st=0; st<ST; ++st){
                    #pragma HLS pipeline II=1

                    hls::vector<r_t, TP*SP> i_vec = i_stream.read();

                    if(st == 0){
                        for(int tp=0; tp<TP; ++tp){
                            #pragma HLS unroll
                            R_MAXVAL [tp] = r_masked_t(MASK_NEG_INF);
                            R_EXP_SUM[tp] = 0;
                        }
                    }

                    // 双向注意力：直接写入，无因果掩码
                    // 关键时序优化：每行的 SP 个新值先做 tree-reduce 得到 local_max，
                    // 再与 R_MAXVAL[tp] 单级合并；loop-carried 路径仅 1 级 max，
                    // 把原 9 级（1+8）串行链压成 ceil(log2 SP)+1 级（=4）非串行树
                    for(int tp=0; tp<TP; ++tp){
                        #pragma HLS unroll
                        r_masked_t lane[SP];
                        #pragma HLS array_partition variable=lane complete
                        for(int sp=0; sp<SP; ++sp){
                            #pragma HLS unroll
                            lane[sp] = r_masked_t(i_vec[tp*SP + sp]);
                            R_BUF[tp][st*SP + sp] = lane[sp];
                        }
                        r_masked_t local_max = vit_tree_max<SP>(lane);
                        R_MAXVAL[tp] = max(R_MAXVAL[tp], local_max);
                    }
                }

                // Pass 2：计算 exp 及 exp_sum
                for(int st=0; st<ST; ++st){
                    #pragma HLS pipeline II=1
                    for(int tp=0; tp<TP; ++tp){
                        for(int sp=0; sp<SP; ++sp){
                            #pragma HLS unroll
                            int   INDEX   = clamp((R_MAXVAL[tp] - R_BUF[tp][st*SP+sp]) >> VIT_EXP_LOG2DENOM,
                                                  0, VIT_EXP_ENTRIES - 1);
                            exp_t EXP_VAL = VIT_EXP_TABLE_DATA[INDEX];
                            R_EXP    [tp][st*SP+sp] = EXP_VAL;
                            R_EXP_SUM[tp]           = R_EXP_SUM[tp] + EXP_VAL;
                        }
                    }
                }

                // Pass 2 后处理：计算 recip（1 / exp_sum）
                for(int tp=0; tp<TP; ++tp){
                    #pragma HLS pipeline off

                    int LUT_IDX = 0;
                    for(int i=0; i<VIT_RECIP_NUM_TABLES; ++i){
                        #pragma HLS unroll
                        if(R_EXP_SUM[tp] >= VIT_RECIP_ALPHAS_DATA[i]){
                            LUT_IDX = i;
                        }
                    }
                    auto ALPHA       = VIT_RECIP_ALPHAS_DATA    [LUT_IDX];
                    auto LOG2DENOM   = VIT_RECIP_LOG2DENOMS_DATA[LUT_IDX];
                    auto OFFSET_DIFF = VIT_RECIP_OFFSETS_DIFF_DATA[LUT_IDX];
                    auto INDEX       = clamp((R_EXP_SUM[tp] - ALPHA) >> LOG2DENOM,
                                            0, VIT_RECIP_ENTRIES - 1);
                    R_RECIP[tp] = VIT_RECIP_TABLES_DATA[LUT_IDX * VIT_RECIP_ENTRIES + INDEX] << OFFSET_DIFF;
                }

                // Pass 3：输出 softmax 值
                for(int st=0; st<ST; ++st){
                    #pragma HLS pipeline II=1
                    hls::vector<softmax_t, TP*SP> o_vec;
                    for(int tp=0; tp<TP; ++tp){
                        for(int sp=0; sp<SP; ++sp){
                            #pragma HLS unroll
                            o_vec[tp*SP + sp] = (R_EXP[tp][st*SP+sp] * R_RECIP[tp]) >> VIT_SOFTMAX_TRUNC_MUL;
                        }
                    }
                    o_stream.write(o_vec);
                }

            } // end TT
        } // end H
    }

    void do_vit_softmax(
        hls::stream<hls::vector<r_t,       TP*SAP> >& i_stream,
        hls::stream<hls::vector<softmax_t, TP*SAP> >& o_stream
    ){
        Adapter<r_t,       H*T, TP, S, SAP, SP > adapter_i;
        Adapter<softmax_t, H*T, TP, S, SP,  SAP> adapter_o;

        hls::stream<hls::vector<r_t,       TP*SP> > i_adapted;
        hls::stream<hls::vector<softmax_t, TP*SP> > softmax_out;

        #pragma HLS dataflow
        adapter_i.do_adapt     (i_stream,    i_adapted   );
        do_vit_softmax_func    (i_adapted,   softmax_out );
        adapter_o.do_adapt     (softmax_out, o_stream    );
    }
};

#endif // __INT_VIT_SOFTMAX_H__
