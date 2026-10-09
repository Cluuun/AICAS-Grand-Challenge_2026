#ifndef __INT_CORDIC_H__
#define __INT_CORDIC_H__

#include "common.h"
#include "utils.h"

// LLM constants
constexpr int CORDIC_HYPERPARAMS   [] = {
    #include "ref/LLM/MHA_CORDIC_HYPERPARAMS.txt"
};

// CORDIC hyperparameters
constexpr int CORDIC_N_ITER         = CORDIC_HYPERPARAMS[0];
constexpr int CORDIC_K              = CORDIC_HYPERPARAMS[1];
constexpr int CORDIC_O_ANGLE        = CORDIC_HYPERPARAMS[2];
constexpr int CORDIC_O_K            = CORDIC_HYPERPARAMS[3];
constexpr int CORDIC_FIXED_4_PI     = CORDIC_HYPERPARAMS[4];
constexpr int CORDIC_FIXED_PI_2     = CORDIC_HYPERPARAMS[5];
constexpr int CORDIC_FIXED_PI       = CORDIC_HYPERPARAMS[6];
constexpr int CORDIC_FIXED_3PI_2    = CORDIC_HYPERPARAMS[7];
constexpr int CORDIC_FIXED_2PI      = CORDIC_HYPERPARAMS[8];

// ROPE hyperparameters
constexpr int ROPE_HYPERPARAMS      [] = {
    #include "ref/LLM/MHA_ROPE_HYPERPARAMS.txt"
};
constexpr int ROPE_TRUNC_MUL        = ROPE_HYPERPARAMS[0];

// constant arrays
constexpr int CORDIC_THETAS      [] = { 
    #include "ref/LLM/MHA_CORDIC_THETAS.txt"
};
constexpr int ROPE_THETAS        [] = { 
    #include "ref/LLM/MHA_ROPE_THETAS.txt"
};

typedef ap_int<10> N_QUARTER_T;
typedef ap_uint<DW_POS> ROPE_POS_T;
typedef ap_uint<DW_FREQS + 1> ROPE_THETA_T;
typedef ap_uint<DW_POS + DW_FREQS + 1> ROPE_MUL_T;

inline FREQS_T rope_mul_shift_wide(int pos_id, int theta) {
    #pragma HLS inline
    // RoPE 软件参考使用宽整数语义；pos=128、theta=2^24 时已经超过 int32 正数范围。
    // 显式用足够位宽完成乘法再右移，避免 C++ int 乘法溢出被 HLS RTL 固化为错误角度。
    ROPE_MUL_T prod_full = (ROPE_POS_T)pos_id * (ROPE_THETA_T)theta;
    return (FREQS_T)(prod_full >> ROPE_TRUNC_MUL);
}

template<
    int H,
    int T,
    int TP,
    int C,
    int CP
>
class CORDIC{
public:

    static constexpr int HC     = C     / H;
    static constexpr int TT     = T     / TP;
    static constexpr int HC2    = HC    / 2;
    static constexpr int HC2T   = HC2   / CP;

    CORDIC() = default;

    // split to 3 parts
    // stage1: preprocess, calculate GAMMA, N_QUARTER, and pass to next stage
    // stage2: apply cordic, and pass to next stage
    // stage3: post-process, and write to output stream

    // stage 1 input: pos_id,           output: GAMMA, N_QUARTER
    // stage 2 input: GAMMA             output: X, Y, N_QUARTER
    // stage 3 input: X, Y, N_QUATER    output: COS, SIN

    // stage 1
    void stage1_preprocess(
        int chunk, 
        hls::stream<hls::vector<FREQS_T,        TP*CP> > &gamma_stream, 
        hls::stream<hls::vector<N_QUARTER_T,    TP*CP> > &n_quarter_stream
    ){
        for(int tt=0; tt<TT; ++tt){
            for(int hc2t=0; hc2t<HC2T; ++hc2t){
                #pragma HLS pipeline II=1

                // generate all pos_ids
                int pos_ids[TP];
                for(int tp=0; tp<TP; ++tp){
                    #pragma HLS unroll
                    pos_ids[tp] = chunk*T + tt*TP + tp;
                }

                // generate input angles
                FREQS_T freqs[TP * CP];
                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        freqs[tp*CP + cp] = rope_mul_shift_wide(pos_ids[tp], ROPE_THETAS[hc2t*CP + cp]);
                    }
                }

                // get GAMMA and N and CIRCLES
                hls::vector<FREQS_T,        TP*CP>    GAMMA;
                hls::vector<N_QUARTER_T,    TP*CP>    N_QUARTER;
                hls::vector<N_QUARTER_T,    TP*CP>    CIRCLES;

                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll

                        int idx = tp*CP + cp;

                        GAMMA       [idx] = freqs[idx];
                        N_QUARTER   [idx] = (GAMMA[idx] * CORDIC_FIXED_4_PI) >> (2 * CORDIC_O_ANGLE);
                        CIRCLES     [idx] = N_QUARTER[idx] >> 3;
                        N_QUARTER   [idx] = N_QUARTER[idx] - (CIRCLES[idx] << 3);
                        GAMMA       [idx] = GAMMA[idx] - CIRCLES[idx] * CORDIC_FIXED_2PI;

                        // preprocessing to bring GAMMA into range [0, 4PI)
                        if(      N_QUARTER[idx] == 1){
                            GAMMA[idx] = CORDIC_FIXED_PI_2 - GAMMA[idx];
                        } else if(N_QUARTER[idx] == 2){
                            GAMMA[idx] = GAMMA[idx] - CORDIC_FIXED_PI_2;
                        } else if(N_QUARTER[idx] == 3){
                            GAMMA[idx] = CORDIC_FIXED_PI - GAMMA[idx];
                        } else if(N_QUARTER[idx] == 4){
                            GAMMA[idx] = GAMMA[idx] - CORDIC_FIXED_PI;
                        } else if(N_QUARTER[idx] == 5){
                            GAMMA[idx] = CORDIC_FIXED_3PI_2 - GAMMA[idx];
                        } else if(N_QUARTER[idx] == 6){
                            GAMMA[idx] = GAMMA[idx] - CORDIC_FIXED_3PI_2;
                        } else if(N_QUARTER[idx] == 7){
                            GAMMA[idx] = CORDIC_FIXED_2PI - GAMMA[idx];
                        } else {
                            // do nothing
                        }

                    }
                }

                // write to stream
                gamma_stream.write(GAMMA);
                n_quarter_stream.write(N_QUARTER);

            }
        }
    }

    // stage 2
    void stage2_cordic(
        hls::stream<hls::vector<FREQS_T,   TP * CP> > &gamma_stream, 
        hls::stream<hls::vector<COS_SIN_T, TP * CP> > &x_stream, 
        hls::stream<hls::vector<COS_SIN_T, TP * CP> > &y_stream
    ){
        hls::vector<COS_SIN_T,  TP * CP> X;
        hls::vector<COS_SIN_T,  TP * CP> Y;
        hls::vector<FREQS_T,    TP * CP> GAMMA;

        for(int tt=0; tt<TT; ++tt){
            for(int hc2t=0; hc2t<HC2T; ++hc2t){
                for(int iter=1; iter<=CORDIC_N_ITER; ++iter){
                    #pragma HLS pipeline II=1

                    if(iter == 1){
                        // initialize X and Y;
                        for(int tp=0; tp<TP; ++tp){
                            for(int cp=0; cp<CP; ++cp){
                                #pragma HLS unroll
                                int idx = tp*CP + cp;
                                X[idx] = 1 << CORDIC_O_ANGLE;
                                Y[idx] = 0;
                            }
                        }
                        // get GAMMA
                        GAMMA = gamma_stream.read();
                    }

                    for(int tp=0; tp<TP; ++tp){
                        for(int cp=0; cp<CP; ++cp){
                            #pragma HLS unroll
                            int idx = tp*CP + cp;
                            COS_SIN_T NEW_GAMMA = GAMMA[idx] > 0 ? GAMMA[idx] - CORDIC_THETAS[iter] : GAMMA[idx] + CORDIC_THETAS[iter];
                            COS_SIN_T NEW_X     = GAMMA[idx] > 0 ? X[idx] - (Y[idx] >> iter) : X[idx] + (Y[idx] >> iter);
                            COS_SIN_T NEW_Y     = GAMMA[idx] > 0 ? Y[idx] + (X[idx] >> iter) : Y[idx] - (X[idx] >> iter);
                            GAMMA[idx] = NEW_GAMMA;
                            X[idx]     = NEW_X;
                            Y[idx]     = NEW_Y;
                        }
                    }

                    if(iter == CORDIC_N_ITER){
                        // write to stream
                        x_stream.write(X);
                        y_stream.write(Y);
                    }
                }
            }
        }
    }

    void stage3_postprocess(
        hls::stream<hls::vector<COS_SIN_T,        TP*CP> > &x_stream, 
        hls::stream<hls::vector<COS_SIN_T,        TP*CP> > &y_stream, 
        hls::stream<hls::vector<N_QUARTER_T,      TP*CP> > &n_quarter_stream, 
        hls::stream<hls::vector<COS_SIN_T,      2*TP*CP> > &cos_sin_stream
    ){
        for(int tt=0; tt<TT; ++tt){
            for(int hc2t=0; hc2t<HC2T; ++hc2t){
                #pragma HLS pipeline II=1

                // post-processing
                hls::vector<COS_SIN_T, TP * CP> COS;
                hls::vector<COS_SIN_T, TP * CP> SIN;

                // read X and Y and N_QUATER
                hls::vector<COS_SIN_T, TP * CP> X = x_stream.read();
                hls::vector<COS_SIN_T, TP * CP> Y = y_stream.read();

                hls::vector<N_QUARTER_T, TP * CP> N_QUARTER = n_quarter_stream.read();

                // apply K scale
                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        int idx = tp*CP + cp;
                        X[idx] = (X[idx] * CORDIC_K) >> CORDIC_O_K;
                        Y[idx] = (Y[idx] * CORDIC_K) >> CORDIC_O_K;
                    }
                }

                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        int idx = tp*CP + cp;
                        if(N_QUARTER[idx] == 0){
                            COS[idx] = +X[idx];
                            SIN[idx] = +Y[idx];
                        } else if(N_QUARTER[idx] == 1){
                            COS[idx] = +Y[idx];
                            SIN[idx] = +X[idx];
                        } else if(N_QUARTER[idx] == 2){
                            COS[idx] = -Y[idx];
                            SIN[idx] = +X[idx];
                        } else if(N_QUARTER[idx] == 3){
                            COS[idx] = -X[idx];
                            SIN[idx] = +Y[idx];
                        } else if(N_QUARTER[idx] == 4){
                            COS[idx] = -X[idx];
                            SIN[idx] = -Y[idx];
                        } else if(N_QUARTER[idx] == 5){
                            COS[idx] = -Y[idx];
                            SIN[idx] = -X[idx];
                        } else if(N_QUARTER[idx] == 6){
                            COS[idx] = +Y[idx];
                            SIN[idx] = -X[idx];
                        } else {                        // full case
                            COS[idx] = +X[idx];
                            SIN[idx] = -Y[idx];
                        }
                            
                    }
                }

                hls::vector<COS_SIN_T, 2*TP*CP> COS_SIN;
                for(int tp=0; tp<TP; ++tp){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        COS_SIN[0*TP*CP + tp*CP + cp] = COS[tp*CP + cp];
                        COS_SIN[1*TP*CP + tp*CP + cp] = SIN[tp*CP + cp];
                    }
                }
                cos_sin_stream.write(COS_SIN);
            }
        }
    }


    void do_cordic(int chunk, hls::stream<hls::vector<COS_SIN_T, 2*TP*CP> > &cos_sin_stream){
        #pragma HLS interface ap_ctrl_chain port=return
        #pragma HLS interface axis port=cos_sin_stream

        #pragma HLS dataflow

        hls::stream<hls::vector<FREQS_T,        TP * CP> > gamma_stream;
        hls::stream<hls::vector<N_QUARTER_T,    TP * CP> > n_quarter_stream;
        hls::stream<hls::vector<COS_SIN_T,      TP * CP> > x_stream;
        hls::stream<hls::vector<COS_SIN_T,      TP * CP> > y_stream;

        stage1_preprocess   (chunk,     gamma_stream,   n_quarter_stream                                            );
        stage2_cordic       (           gamma_stream,   x_stream,   y_stream                                        );
        stage3_postprocess  (                           x_stream,   y_stream,   n_quarter_stream, cos_sin_stream    );
    }

};

#endif
