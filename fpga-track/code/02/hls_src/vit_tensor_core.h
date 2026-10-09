#ifndef __INT_MAIN_VIT_TENSOR_CORE_H__
#define __INT_MAIN_VIT_TENSOR_CORE_H__

#include "src/common.h"
#include "src/utils.h"

// ViT encoder GEMM：仅单一 trip 计数 T_TOTAL，无 LLM CLS / Connector 第二套周期。

template<
    class if_t,
    class w_t,
    class of_t,

    int TP,
    int CIP,
    int COP,
    int T_TOTAL
>
class VIT_TENSOR_CORE{
public:

    VIT_TENSOR_CORE() {}

    void do_tensor_core(
        hls::stream<hls::vector<if_t, TP *CIP> >& i_stream,
        hls::stream<hls::vector<w_t,  COP*CIP> >& w_stream,
        hls::stream<hls::vector<of_t, TP *COP> >& o_stream
    ) {
        #ifndef __SYNTHESIS__
        ProgressBar pb("VIT_TENSOR_CORE", T_TOTAL / 1000 + 1);
        #endif

        for(int n=0; n<T_TOTAL; ++n) {
            #pragma HLS pipeline II=1

            #ifndef __SYNTHESIS__
            if(n % 1000 == 0) pb.update(1);
            #endif

            hls::vector<if_t, TP *CIP> i_vec = i_stream.read();
            hls::vector<w_t,  COP*CIP> w_vec = w_stream.read();
            hls::vector<of_t, TP *COP> o_vec;

            for(int tp=0; tp<TP; ++tp) {
                for(int cop=0; cop<COP; ++cop) {
                    #pragma HLS unroll
                    o_vec[tp*COP + cop] = 0;
                }
            }

            for(int tp=0; tp<TP; ++tp) {
                for(int cop=0; cop<COP; ++cop) {
                    for(int cip=0; cip<CIP; ++cip) {
                        #pragma HLS unroll
                        auto mul_res = i_vec[tp*CIP + cip] * w_vec[cop*CIP + cip];
                        #pragma HLS bind_op variable=mul_res op=mul impl=dsp
                        o_vec[tp*COP + cop] += mul_res;
                    }
                }
            }
            o_stream.write(o_vec);
        }
    }

};

#endif
