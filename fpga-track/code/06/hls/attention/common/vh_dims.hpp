#pragma once

// Model shape constants only — safe for Vitis HLS csynth (no ap_fixed/libstdc++).

namespace vh {

constexpr int IMG_C    = 3;
constexpr int IMG_HW   = 512;
constexpr int PATCH    = 16;
constexpr int N_GRID   = IMG_HW / PATCH;
// Cosim-only small-slice override (A1-P4b-G4 RTL-vs-Cmodel repro). Production
// builds pass no -D and get the canonical 1024/768/12 shape unchanged.
#ifdef VH_N_POS_OVERRIDE
constexpr int N_POS    = VH_N_POS_OVERRIDE;
#else
constexpr int N_POS    = N_GRID * N_GRID;
#endif
#ifdef VH_N_HEAD_OVERRIDE
constexpr int N_HEAD   = VH_N_HEAD_OVERRIDE;
constexpr int C_EMB    = VH_N_HEAD_OVERRIDE * 64;
#else
constexpr int C_EMB    = 768;
constexpr int N_HEAD   = 12;
#endif
constexpr int N_LAYER  = 12;
constexpr int D_HEAD   = C_EMB / N_HEAD;
constexpr int F_FFN    = 3072;
constexpr int SHUF_SF  = 4;
constexpr int N_OUT    = (N_GRID / SHUF_SF) * (N_GRID / SHUF_SF);
constexpr int C_SHUF   = C_EMB * SHUF_SF * SHUF_SF;
constexpr int D_OUT    = 960;

}  // namespace vh
