#include "attention.hpp"

#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#  include "hls_math.h"
#else
#  include <cmath>
#endif

// Per-head fused scoring + softmax + value aggregation.
//
// Q/K/V layout is the post-reshape ggml view:
//   Q[m, h, d] = Q[m*C + h*DH + d]   where C = N_HEAD*DH = 768.
//
// Per head:
//   S[p1, p2] = (1/sqrt(DH)) * sum_d Q[p1, h, d] * K[p2, h, d]
//   S[p1, :]  = softmax(S[p1, :])
//   O[p1, d]  = sum_p2 S[p1, p2] * V[p2, h, d]
//
// The score row is consumed immediately by the value multiply — only one
// row of S (size M=1024 floats = 4 KB) is alive at a time. The whole
// score tensor never materializes in DDR.

extern "C" void k_attention(
    const float * Q,
    const float * K,
    const float * V,
    float       * O_out)
{
    constexpr int M  = vh::N_POS;
    constexpr int C  = vh::C_EMB;
    constexpr int H  = vh::N_HEAD;
    constexpr int DH = vh::D_HEAD;

#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
    const float scale = 1.0f / hls::sqrtf((float) DH);
#else
    const float scale = 1.0f / std::sqrt((float) DH);
#endif

    // One score row per p1 (size M); fits comfortably in BRAM in synth.
    static float Srow[M];

    for (int h = 0; h < H; ++h) {
        for (int p1 = 0; p1 < M; ++p1) {
            const float * q = Q + (size_t) p1 * C + h * DH;

            // 1) compute scores row
            float mx = -1e30f;
            for (int p2 = 0; p2 < M; ++p2) {
                const float * k = K + (size_t) p2 * C + h * DH;
                float s = 0.0f;
                for (int d = 0; d < DH; ++d) s += q[d] * k[d];
                s *= scale;
                Srow[p2] = s;
                if (s > mx) mx = s;
            }

            // 2) softmax in place
            float ssum = 0.0f;
            for (int p2 = 0; p2 < M; ++p2) {
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
                Srow[p2] = hls::expf(Srow[p2] - mx);
#else
                Srow[p2] = std::exp(Srow[p2] - mx);
#endif
                ssum += Srow[p2];
            }
            const float inv = 1.0f / ssum;
            for (int p2 = 0; p2 < M; ++p2) Srow[p2] *= inv;

            // 3) value aggregation for this (h, p1)
            float * o = O_out + (size_t) p1 * C + h * DH;
            for (int d = 0; d < DH; ++d) {
                float acc = 0.0f;
                for (int p2 = 0; p2 < M; ++p2) {
                    acc += Srow[p2] * V[(size_t) p2 * C + h * DH + d];
                }
                o[d] = acc;
            }
        }
    }
}
