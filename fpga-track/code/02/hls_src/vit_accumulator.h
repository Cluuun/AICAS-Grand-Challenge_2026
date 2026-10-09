#ifndef __INT_MAIN_VIT_ACCUMULATOR_H__
#define __INT_MAIN_VIT_ACCUMULATOR_H__

#include "src/common.h"
#include "src/utils.h"

// ViT encoder 累加器：仅两档 ACC_CYCS1/ACC_CYCS2 与 ACC_REPEAT1/ACC_REPEAT2，
// 不包含 LLM/Connector 的第三档 ACC_CYCS3/ACC_REPEAT3，以缩短综合/仿真需覆盖的总周期。

template<
    class if_t,
    class as_t,
    class ws_t,
    class acc_t,
    class of_t,

    int TRUNC_BASE,

    int TP,
    int CP,

    int ACC_CYCS1,
    int ACC_CYCS2,

    int ACC_REPEAT1,
    int ACC_REPEAT2
>
class VIT_ACCUMULATOR{
public:

    static constexpr int DECODER_TOTAL_CYCS = ACC_REPEAT1*ACC_CYCS1 + ACC_REPEAT2*ACC_CYCS2;
    static constexpr int DECODER_BOUND_CYCS = ACC_REPEAT1*ACC_CYCS1;

    VIT_ACCUMULATOR() {}

    void stage1_accumulation(
        hls::stream<hls::vector<if_t, TP*CP     > >& i_stream,
        hls::stream<hls::vector<as_t, TP        > >& s_stream,
        hls::stream<hls::vector<ws_t, CP        > >& s1_stream,
        hls::stream<hls::vector<ws_t, CP        > >& s2_stream,
        hls::stream<hls::vector<of_t, TP*CP/2   > >& o1_stream,
        hls::stream<hls::vector<of_t, TP*CP/2   > >& o2_stream
    ) {
        acc_t psum[TP*CP];
        #pragma HLS array_reshape variable=psum complete dim=1

        for(int cyc=0, sub_cyc=0; cyc<DECODER_TOTAL_CYCS; ++cyc){
            #pragma HLS pipeline II=1
            #pragma HLS EXPRESSION_BALANCE OFF
            if(sub_cyc == 0){
                for(int tp = 0; tp < TP; tp++) {
                    for(int cp = 0; cp < CP; cp++) {
                        #pragma HLS unroll
                        psum[tp*CP + cp] = 0;
                    }
                }
            }

            hls::vector<ws_t, CP   >    s1_vec  =   s1_stream.read();
            hls::vector<ws_t, CP   >    s2_vec  =   s2_stream.read();

            constexpr int s1_width = if_t::width + (1 <<  as_t::width     ) - 1;
            constexpr int s2_width = if_t::width + (1 << (as_t::width + 1)) - 1;
            typedef ap_int<s1_width> if_s1_t;
            typedef ap_int<s2_width> if_s2_t;

            hls::vector<if_t, TP*CP>    i_vec   =   i_stream .read();
            hls::vector<as_t, TP   >    s_vec   =   s_stream .read();

            for(int tp = 0; tp < TP; tp++) {
                for(int cp = 0; cp < CP; cp++) {
                    #pragma HLS unroll
                    auto x_val     = i_vec [tp*CP + cp];
                    auto s_val     = s_vec [tp];
                    auto s1_val    = s1_vec[cp];
                    auto s2_val    = s2_vec[cp];

                    auto x_shift   = if_s1_t(x_val) << s_val;
                    psum[tp*CP + cp] += (if_s2_t(x_shift) << s1_val) + (if_s2_t(x_shift) << s2_val);
                }
            }

            int SUB_CYCS = (cyc < DECODER_BOUND_CYCS) ? ACC_CYCS1 : ACC_CYCS2;

            if(sub_cyc+1 == SUB_CYCS){
                sub_cyc = 0;
                hls::vector<of_t, TP*CP/2> o1_vec;
                hls::vector<of_t, TP*CP/2> o2_vec;
                for(int index=0; index<TP*CP/2; ++index){
                    #pragma HLS unroll
                    o1_vec[index] = psum[index          ] >> TRUNC_BASE;
                    o2_vec[index] = psum[index + TP*CP/2] >> TRUNC_BASE;
                }
                o1_stream.write(o1_vec);
                o2_stream.write(o2_vec);
            } else {
                ++sub_cyc;
            }
        }
    }

    void stage2_unpack(
        hls::stream<hls::vector<of_t, TP*CP/2> >& o1_stream,
        hls::stream<hls::vector<of_t, TP*CP/2> >& o2_stream,
        hls::stream<hls::vector<of_t, CP     > >& o_stream
    ) {
        const int NUM_OUTPUT = ACC_REPEAT1 + ACC_REPEAT2;

        for(int num_output=0; num_output<NUM_OUTPUT; ++num_output){
            hls::vector<of_t, TP*CP/2> o1_vec = o1_stream.read();
            hls::vector<of_t, TP*CP/2> o2_vec = o2_stream.read();
            of_t obuf[TP*CP];
            #pragma HLS array_reshape variable=obuf complete dim=1
            for(int index=0; index<TP*CP/2; ++index){
                #pragma HLS unroll
                obuf[index          ] = o1_vec[index];
                obuf[index + TP*CP/2] = o2_vec[index];
            }

            for(int tp=0; tp<TP; ++tp){
                #pragma HLS pipeline II=1
                hls::vector<of_t, CP> o_vec;

                for(int cp=0; cp<CP; ++cp){
                    #pragma HLS unroll
                    o_vec[cp] = obuf[cp];
                }
                for(int tpi=0; tpi<TP-1; ++tpi){
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        obuf[tpi*CP + cp] = obuf[(tpi+1)*CP + cp];
                    }
                }
                o_stream.write(o_vec);
            }
        }
    }

    void do_accumulator(
        hls::stream<hls::vector<if_t, TP*CP > >& i_stream,
        hls::stream<hls::vector<as_t, TP    > >& s_stream,
        hls::stream<hls::vector<ws_t, CP    > >& s1_stream,
        hls::stream<hls::vector<ws_t, CP    > >& s2_stream,
        hls::stream<hls::vector<of_t, CP    > >& o_stream
    ) {
        #pragma HLS interface ap_ctrl_chain port=return
        #pragma HLS interface axis port=i_stream
        #pragma HLS interface axis port=s_stream
        #pragma HLS interface axis port=s1_stream
        #pragma HLS interface axis port=s2_stream
        #pragma HLS interface axis port=o_stream

        #pragma HLS dataflow

        hls::stream<hls::vector<of_t, TP*CP/2> > o1_stream;
        hls::stream<hls::vector<of_t, TP*CP/2> > o2_stream;

        stage1_accumulation (i_stream,  s_stream,   s1_stream, s2_stream, o1_stream, o2_stream           );
        stage2_unpack       (                                              o1_stream, o2_stream,  o_stream);
    }

};

#endif
