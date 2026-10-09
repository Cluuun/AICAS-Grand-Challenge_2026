#ifndef GEMM_KERNEL_UTILS_H
#define GEMM_KERNEL_UTILS_H

#include "gemm_kernel.h"

#include <ap_int.h>

static const int SCALE_INDIV_FIXED_BITS = 16;
static const int SCALE_FIXED_BITS = 24;
static const int SCALE_COMBINE_SHIFT =
    2 * SCALE_INDIV_FIXED_BITS - SCALE_FIXED_BITS;

typedef ap_int<8> ACT_HW_DTYPE;
typedef ap_int<8> WGT_HW_DTYPE;
typedef ap_uint<16> SCALE_BITS_HW_DTYPE;
typedef ap_int<18> PACK_PROD_HW_DTYPE;
typedef ap_int<32> SCALE_INDIV_FIXED_HW_DTYPE;
typedef ap_int<32> SCALE_FIXED_HW_DTYPE;
typedef ap_int<56> OUT_ACC_HW_DTYPE;

// Convert one FP16 scale bit pattern into an internal Q16 fixed-point value.
// This keeps the external ABI unchanged while removing floating-point work
// from the tile-preparation hotspot.
static inline SCALE_INDIV_FIXED_HW_DTYPE fp16_bits_to_fixed_q16(
    SCALE_BITS_HW_DTYPE bits) {
#pragma HLS INLINE

    const bool sign = bits[15];
    const ap_uint<5> exp = bits.range(14, 10);
    const ap_uint<10> mantissa = bits.range(9, 0);
    SCALE_INDIV_FIXED_HW_DTYPE magnitude = 0;

    if (exp == 0) {
        if (mantissa != 0) {
            const SCALE_INDIV_FIXED_HW_DTYPE rounding = 1 << 7;
            magnitude =
                (SCALE_INDIV_FIXED_HW_DTYPE(mantissa) + rounding) >> 8;
        }
    } else if (exp >= 30) {
        magnitude = 0x7fffffff;
    } else {
        SCALE_INDIV_FIXED_HW_DTYPE significand =
            SCALE_INDIV_FIXED_HW_DTYPE(ap_uint<11>(0x400) | mantissa);
        const int shift = int(exp) - 9;

        if (shift >= 0) {
            magnitude = significand << shift;
        } else {
            const int rshift = -shift;
            const SCALE_INDIV_FIXED_HW_DTYPE rounding = 1 << (rshift - 1);
            magnitude = (significand + rounding) >> rshift;
        }
    }

    if (sign) {
        return SCALE_INDIV_FIXED_HW_DTYPE(-magnitude);
    }

    return magnitude;
}



// Multiply two internal Q16 scales and return the combined Q24 scale.
static inline SCALE_FIXED_HW_DTYPE fixed_combined_scale_q(
    SCALE_INDIV_FIXED_HW_DTYPE lhs_q16,
    SCALE_INDIV_FIXED_HW_DTYPE rhs_q16) {
#pragma HLS INLINE

    const ap_int<64> product =
        ap_int<64>(lhs_q16) * ap_int<64>(rhs_q16);
    const ap_int<64> rounding = ap_int<64>(1) << (SCALE_COMBINE_SHIFT - 1);
    const ap_int<64> adjusted =
        (product >= 0) ? (product + rounding) : (product - rounding);

    return static_cast<SCALE_FIXED_HW_DTYPE>(adjusted >> SCALE_COMBINE_SHIFT);
}



// DSP Packing
static inline void pack_mul_2int8(ACT_HW_DTYPE common,
                                  WGT_HW_DTYPE rhs0,
                                  WGT_HW_DTYPE rhs1,
                                  PACK_PROD_HW_DTYPE &out0,
                                  PACK_PROD_HW_DTYPE &out1) {
#pragma HLS INLINE

    const ap_int<27> packed = (ap_int<27>(rhs1) << 18) + ap_int<27>(ap_int<18>(rhs0));
    const ap_int<45> product = packed * ap_int<18>(common);

    PACK_PROD_HW_DTYPE low = product.range(17, 0);
    PACK_PROD_HW_DTYPE high = product >> 18;

    if (low < 0) {
        high += 1;
    }

    out0 = low;
    out1 = high;
}

#endif
