#include "gemm_kernel.h"
#include "gemm_kernel_utils.h"
#include <ap_int.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

static uint16_t f32_to_fp16_bits(float f) {
    // Minimal IEEE fp16 converter for scales (positive values only here).
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 0x1;
    int32_t  exp  = ((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3ff;
    if (exp <= 0) return 0;
    if (exp >= 31) return (uint16_t)((sign << 15) | (31 << 10));
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

static float fp16_bits_to_f32(uint16_t h) {
    // Approximate fp16 -> float recovery for positive, normal-range scales.
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) return 0.0f;
    uint32_t sig = (1u << 10) | mant;
    int shift = (int)(25 - exp);
    if (shift >= 0 && shift < 31) return (float)sig / (float)(1u << shift);
    return (float)sig * std::ldexp(1.0f, -shift);
}

int main() {
    // Tile-aligned csim shape — matches v2.1 Path C TILE_M=64, TILE_N=64, TILE_K=32.
    // Kernel drops boundary checks, so all three dims must be multiples of the
    // corresponding tile. K=128 gives 4 k-groups; total mem = ~160 KB, csim-fast.
    const int M = 256, K = 128, N = 128;
    const int K_G = K / 32;

    std::vector<int8_t>   act(M*K), weight(N*K);
    std::vector<uint16_t> act_scales(M*K_G), weight_scales(N*K_G);
    std::vector<float>    out(M*N), ref(M*N);

    srand(42);
    for (auto &a : act)    a = (int8_t)((rand() & 0xff) - 128);
    for (auto &w : weight) w = (int8_t)((rand() & 0xff) - 128);
    for (auto &s : act_scales)    s = f32_to_fp16_bits(0.01f + 0.001f*(rand()%10));
    for (auto &s : weight_scales) s = f32_to_fp16_bits(0.02f + 0.001f*(rand()%10));

    // Reference: bit-match the kernel's Q16→Q24 fixed-point scale pipeline
    // exactly (fp16_bits_to_fixed_q16 + fixed_combined_scale_q from utils.h).
    // Using host fp32 scales here would drift from the kernel by ~1-2%
    // after a few k-groups and produce false "mismatches" at the 1e-2 gate.
    // v2.1 ABI: weight is [N][K], weight_scales is [N][K/32].
    const float fixed_to_float = 1.0f / (float)(1u << SCALE_FIXED_BITS);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int kg = 0; kg < K_G; ++kg) {
                SCALE_INDIV_FIXED_HW_DTYPE as_q = fp16_bits_to_fixed_q16(
                    SCALE_BITS_HW_DTYPE(act_scales[m*K_G + kg]));
                SCALE_INDIV_FIXED_HW_DTYPE ws_q = fp16_bits_to_fixed_q16(
                    SCALE_BITS_HW_DTYPE(weight_scales[n*K_G + kg]));
                SCALE_FIXED_HW_DTYPE sc_q24 =
                    fixed_combined_scale_q(as_q, ws_q);
                int64_t dot = 0;
                for (int kl = 0; kl < 32; ++kl) {
                    dot += (int64_t)act[m*K + kg*32 + kl] *
                           (int64_t)weight[n*K + kg*32 + kl];
                }
                int64_t scaled = dot * (int64_t)(int32_t)sc_q24;
                acc += (float)scaled * fixed_to_float;
            }
            ref[m*N + n] = acc;
        }
    }

    // v2.1+wide ABI: act/weight are ap_uint<256>* (32-byte beats). Pack the
    // raw int8 test vectors into beats so each beat carries 32 contiguous
    // bytes of one row. Byte b of beat (m,kg) maps to act[m*K + kg*32 + b].
    std::vector<ap_uint<256>> act_packed(M*K_G), weight_packed(N*K_G);
    for (int m = 0; m < M; ++m) {
        for (int kg = 0; kg < K_G; ++kg) {
            ap_uint<256> beat = 0;
            for (int b = 0; b < 32; ++b) {
                uint8_t byte_val = (uint8_t)act[m*K + kg*32 + b];
                beat.range(b*8+7, b*8) = byte_val;
            }
            act_packed[m*K_G + kg] = beat;
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int kg = 0; kg < K_G; ++kg) {
            ap_uint<256> beat = 0;
            for (int b = 0; b < 32; ++b) {
                uint8_t byte_val = (uint8_t)weight[n*K + kg*32 + b];
                beat.range(b*8+7, b*8) = byte_val;
            }
            weight_packed[n*K_G + kg] = beat;
        }
    }

    fpga_gemm_kernel(act_packed.data(),
                     weight_packed.data(),
                     act_scales.data(), weight_scales.data(),
                     out.data(), M, K, N);

    double max_rel_err = 0.0;
    int mismatches = 0;
    for (int i = 0; i < M*N; ++i) {
        double err = std::abs((double)(out[i] - ref[i]));
        double den = std::max(1e-6, (double)std::abs(ref[i]));
        double rel = err / den;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 1e-2) {
            if (mismatches < 8) {
                printf("mismatch i=%d out=%f ref=%f rel=%f\n",
                       i, out[i], ref[i], rel);
            }
            mismatches++;
        }
    }
    printf("max_rel_err = %.6f, mismatches = %d / %d\n",
           max_rel_err, mismatches, M*N);
    return (max_rel_err > 1e-2) ? 1 : 0;
}
