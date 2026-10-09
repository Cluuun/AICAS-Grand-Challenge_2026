#ifndef VEC_ENGINE_V3_HPP
#define VEC_ENGINE_V3_HPP

#include <ap_int.h>

void vec_engine_v3(
    const ap_uint<128>* A_in,
    const ap_uint<128>* B_in1,
    const ap_uint<128>* B_in2,
    const ap_uint<128>* B_in3,
    const ap_uint<128>* B_in4,
    const ap_uint<128>* scales1,
    const ap_uint<128>* scales2,
    const ap_uint<128>* scales3,
    const ap_uint<128>* scales4,
    int k_block_num,
    int n_block_num,
    int op,
    ap_uint<128>* C_out);

#endif
