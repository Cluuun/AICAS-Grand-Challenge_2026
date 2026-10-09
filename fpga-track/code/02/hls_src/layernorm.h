#ifndef __INT_LAYERNORM_H__
#define __INT_LAYERNORM_H__

#include "src/common.h"
#include "src/utils.h"

// ViT IntLayerNorm 标量（与 int_vit_encoder.IntLayerNorm 及 ref/ViT/VIT_LAYERNORM_HYPERPARAMS.txt 一致）
constexpr int VIT_LAYERNORM_HYPERPARAMS[] = {
    #include "ref/ViT/VIT_LAYERNORM_HYPERPARAMS.txt"
};
constexpr int VIT_LAYERNORM_C           = VIT_LAYERNORM_HYPERPARAMS[0];   // C（通常 768）
constexpr int VIT_LAYERNORM_O_IN        = VIT_LAYERNORM_HYPERPARAMS[1];   // 输入定点指数 O_in
constexpr int VIT_LAYERNORM_O_LNW       = VIT_LAYERNORM_HYPERPARAMS[2];   // gamma 定点指数 O_lnw
constexpr int VIT_LAYERNORM_TRUNC_MUL1  = VIT_LAYERNORM_HYPERPARAMS[3];   // x*rsqrt 后右移
constexpr int VIT_LAYERNORM_TRUNC_MUL2  = VIT_LAYERNORM_HYPERPARAMS[4];   // *lnw 后右移
constexpr int VIT_LAYERNORM_O_OUT       = VIT_LAYERNORM_HYPERPARAMS[5];   // 输出定点指数
constexpr int VIT_LAYERNORM_NUM_TABLES  = VIT_LAYERNORM_HYPERPARAMS[6];   // rsqrt 分段 LUT 个数
constexpr int VIT_LAYERNORM_ENTRIES     = VIT_LAYERNORM_HYPERPARAMS[7];   // 每段表项数 2^addrwidth
constexpr int VIT_LAYERNORM_RSQRT_LEN   = VIT_LAYERNORM_NUM_TABLES * VIT_LAYERNORM_ENTRIES;
static_assert(VIT_LAYERNORM_C == VIT_C, "ViT LayerNorm C must match VIT_C");

// ============================================================================
// ViT IntLayerNorm — 与 int_vit_encoder.py IntLayerNorm 对齐
// ============================================================================
// 与 RMSNorm 的区别:
//   1. 多了减均值: x_mean = x_sum / C
//   2. 方差用中心化数据: x_var = sum((x - mean)^2)
//   3. 有 bias (lnb): output = (x_c * rsqrt >> T1) * lnw >> T2 + lnb
//
// 软件流程 (IntLayerNorm.forward):
//   x_int          — 输入 int64
//   x_sum          — sum(x_int, dim=-1)
//   x_mean         — x_sum // C           (python 整除)
//   x_centered     — x_int - x_mean
//   x_var          — sum(x_centered^2, dim=-1)
//   x_lut_idx      — 分段选表索引
//   rsqrt_flat     — LUT[lut_idx][addr]
//   x_rsqrt        — rsqrt_flat << offsets_diff[lut_idx]
//   x_out_mul1     — (x_centered * x_rsqrt) >> Trunc_mul1
//   x_out_mul2     — (x_out_mul1 * lnw) >> Trunc_mul2
//   x_out_with_bias — x_out_mul2 + lnb
// ============================================================================

// rsqrt 分段 LUT 参数（从 ref/ViT/ 加载）
constexpr int64_t VIT_LAYERNORM_ALPHAS[] = {
    #include "ref/ViT/VIT_LAYERNORM_ALPHAS.txt"
};
constexpr int VIT_LAYERNORM_LOG2DENOMS[] = {
    #include "ref/ViT/VIT_LAYERNORM_LOG2DENOMS.txt"
};
constexpr int VIT_LAYERNORM_OFFSETS_DIFF[] = {
    #include "ref/ViT/VIT_LAYERNORM_OFFSETS_DIFF.txt"
};
constexpr int VIT_LAYERNORM_TABLES[] = {
    #include "ref/ViT/VIT_LAYERNORM_TABLES.txt"
};

// LayerNorm 权重 (gamma) — MHA0, MLP0, MHA1, MLP1, ... 交错排列，与 RMSNorm 一致
constexpr int VIT_LAYERNORM_LNW[] = {
    #include "ref/ViT/VIT_LAYERNORM_LNW.txt"
};

// LayerNorm 偏置 (beta) — 同样交错排列
constexpr int64_t VIT_LAYERNORM_LNB[] = {
    #include "ref/ViT/VIT_LAYERNORM_LNB.txt"
};

// 计算分段 beta 上界: beta_i = alpha_i + 2^log2denom_i * (entries - 1)
inline int64_t vit_layernorm_segment_beta(int i) {
    return VIT_LAYERNORM_ALPHAS[i] +
           (int64_t(1) << VIT_LAYERNORM_LOG2DENOMS[i]) * int64_t(VIT_LAYERNORM_ENTRIES - 1);
}


template<
    int L,          // ViT Encoder 层数
    int T,          // token 并行粒度
    int TP,         // token 并行度
    int C,          // hidden_size
    int CP          // channel 并行度
>
class LAYERNORM{
public:

    static_assert(T % TP == 0, "T % TP != 0");
    static_assert(C % CP == 0, "C % CP != 0");

    static constexpr int TT = T / TP;
    static constexpr int CT = C / CP;

    // L 是 Encoder 层数，每层只保留 2 个 pre-LayerNorm (MHA 前 + MLP 前)。
    // ViT post_layernorm/final LayerNorm 不在 PL 中实现，由 PS/软件侧处理或直接旁路。
    VIT_LNW_T   LNW     [2*L][C];
    ap_int<64>  LNB     [2*L][C];

    VIT_X_T     X_BUF       [TP][C];
    ap_int<64>  X_SUM       [TP];       // 用于存 sum，之后复用存 mean
    ap_uint<64> X_VAR       [TP];
    VIT_X_RSQRT_T X_RSQRT  [TP];

    template<typename lnw_init_t, typename lnb_init_t>
    LAYERNORM(const lnw_init_t lnw_init[2*L*C], const lnb_init_t lnb_init[2*L*C]){
        // init encoder LNWs: 交错排列 MHA0, MLP0, MHA1, MLP1, ...
        for(int l=0; l<L; ++l){
            for(int mha_or_mlp=0; mha_or_mlp<2; ++mha_or_mlp){
                for(int c=0; c<C; ++c){
                    LNW[l*2 + mha_or_mlp][c] = lnw_init[l*2*C + mha_or_mlp*C + c];
                    LNB[l*2 + mha_or_mlp][c] = lnb_init[l*2*C + mha_or_mlp*C + c];
                }
            }
        }
    }


    void do_layernorm(int l, int mha_or_mlp,
                      hls::stream<hls::vector<VIT_X_T, TP * CP> > &i_stream,
                      hls::stream<hls::vector<VIT_XLN_T, TP * CP> > &o_stream)
    {
        // shaping
        #pragma HLS array_reshape variable=X_BUF        complete              dim=1
        #pragma HLS array_reshape variable=X_BUF        cyclic      factor=CP dim=2
        #pragma HLS array_reshape variable=X_SUM         complete
        #pragma HLS array_reshape variable=X_VAR         complete
        #pragma HLS array_reshape variable=X_RSQRT       complete
        #pragma HLS array_reshape variable=LNW           cyclic      factor=CP dim=2
        #pragma HLS array_reshape variable=LNB           cyclic      factor=CP dim=2
        // bind storage
        #pragma HLS bind_storage variable=X_BUF          type=ram_2p impl=URAM
        #pragma HLS bind_storage variable=X_SUM          type=ram_2p impl=LUTRAM
        #pragma HLS bind_storage variable=X_VAR          type=ram_2p impl=LUTRAM
        #pragma HLS bind_storage variable=X_RSQRT        type=ram_2p impl=LUTRAM
        #pragma HLS bind_storage variable=LNW            type=ram_2p impl=BRAM
        #pragma HLS bind_storage variable=LNB            type=ram_2p impl=BRAM

        LOOP_TT: for(int tt=0; tt<TT; ++tt){

            // ========== pass1: 读入数据 + 累加 X_SUM ==========
            LOOP_CT1: for(int ct=0; ct<CT; ++ct){
                #pragma HLS pipeline II=1

                hls::vector<VIT_X_T, TP * CP> i_vec = i_stream.read();

                if(ct == 0){
                    for(int tp=0; tp<TP; ++tp){
                        #pragma HLS unroll
                        X_SUM[tp] = 0;
                    }
                }

                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        X_BUF[tp][ct*CP + cp] = i_vec[tp*CP + cp];
                        X_SUM[tp] = X_SUM[tp] + (ap_int<64>)i_vec[tp*CP + cp];
                    }
                }
            }

            // ========== pass1 post: 计算 mean = sum // C ==========
            // 与软件 x_mean = x_sum // C 对齐：Python / PyTorch 的 // 是 floor（向 -∞），
            // C++ 有符号 / 是向0 截断，负的 x_sum 时二者不同，会导致方差/rsqrt/量化全链偏差。
            LOOP_TP_MEAN: for(int tp=0; tp<TP; ++tp){
                #pragma HLS pipeline off
                ap_int<64> s  = X_SUM[tp];
                ap_int<64> c64 = (ap_int<64>)C;
                ap_int<64> mu;
                if(s >= 0){
                    mu = (ap_int<64>)(s / c64);
                } else {
                    mu = (ap_int<64>)(-((-s + c64 - 1) / c64));
                }
                X_SUM[tp] = mu;     // 复用 X_SUM 存 mean
            }

            // ========== pass2: 中心化 + 累加方差 ==========
            LOOP_CT2: for(int ct=0; ct<CT; ++ct){
                #pragma HLS pipeline II=1

                if(ct == 0){
                    for(int tp=0; tp<TP; ++tp){
                        #pragma HLS unroll
                        X_VAR[tp] = 0;
                    }
                }

                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        ap_int<64> diff = (ap_int<64>)X_BUF[tp][ct*CP + cp] - X_SUM[tp];
                        ap_uint<64> dp2 = (ap_uint<64>)(diff * diff);
                        X_VAR[tp] = X_VAR[tp] + dp2;
                    }
                }
            }

            // ========== pass2 post: rsqrt LUT 查表 ==========
            // 与软件 IntSegmentedLUT 完全对齐:
            //   lut_idx: x_var >= alphas[i] 时选 i
            //   addr:    (x_var - alpha) >> log2denom, clamp [0, entries-1]
            //   rsqrt:   tables[lut_idx * entries + addr] << offsets_diff[lut_idx]
            LOOP_TP_RSQRT: for(int tp=0; tp<TP; ++tp){
                #pragma HLS pipeline off

                int LUT_IDX = 0;
                for(int i=0; i<VIT_LAYERNORM_NUM_TABLES; ++i){
                    #pragma HLS unroll
                    if(X_VAR[tp] >= (ap_uint<64>)VIT_LAYERNORM_ALPHAS[i]){
                        LUT_IDX = i;
                    }
                }
                auto ALPHA          = VIT_LAYERNORM_ALPHAS      [LUT_IDX];
                auto LOG2DENOM      = VIT_LAYERNORM_LOG2DENOMS  [LUT_IDX];
                auto OFFSET_DIFF    = VIT_LAYERNORM_OFFSETS_DIFF[LUT_IDX];
                auto INDEX          = clamp((ap_int<64>)(X_VAR[tp] - (ap_uint<64>)ALPHA) >> LOG2DENOM, (ap_int<64>)0, (ap_int<64>)(VIT_LAYERNORM_ENTRIES - 1));

                X_RSQRT[tp] = ((X_RSQRT_T)VIT_LAYERNORM_TABLES[LUT_IDX * VIT_LAYERNORM_ENTRIES + (int)INDEX]) << OFFSET_DIFF;
            }


            // ========== pass3: 乘 rsqrt * lnw + lnb 输出 ==========
            LOOP_CT3: for(int ct=0; ct<CT; ++ct){
                #pragma HLS pipeline II=1
                hls::vector<VIT_XLN_T, TP * CP> o_vec;
                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        // x_centered * rsqrt >> Trunc_mul1
                        ap_int<64> diff = (ap_int<64>)X_BUF[tp][ct*CP + cp] - X_SUM[tp];
                        auto X_MUL_RSQRT = (diff * (ap_int<64>)X_RSQRT[tp]) >> VIT_LAYERNORM_TRUNC_MUL1;
                        // * lnw >> Trunc_mul2
                        int sub_l = l*2 + mha_or_mlp;
                        auto X_MUL_LNW = (X_MUL_RSQRT * (ap_int<64>)LNW[sub_l][ct*CP + cp]) >> VIT_LAYERNORM_TRUNC_MUL2;
                        // + lnb
                        o_vec[tp*CP + cp] = (VIT_XLN_T)(X_MUL_LNW + LNB[sub_l][ct*CP + cp]);
                    }
                }
                o_stream.write(o_vec);
            } // end of ct loop
        } // end of tt loop

    }

};

#endif // __INT_LAYERNORM_H__
