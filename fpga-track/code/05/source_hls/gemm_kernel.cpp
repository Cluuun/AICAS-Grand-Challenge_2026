#include "gemm_kernel.h"
#include "gemm_kernel_utils.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

// ============================================================================
// ggml-xrt v2.1+wide FPGA GEMM Kernel.
//
// Design brief:
//   Prioritize csynth-pass + full-pipeline walk-through over peak t/s.
//   Tile 64x64x32, unroll 2x2x32 = 128 MAC/cycle. ABI uses ap_uint<256>*
//   for act/weight payload ports — one 256-bit beat per m_axi read.
//
// Architectural layout:
//   - Full activation resident in URAM (M * GEMM_SHIP_K bytes, compiled capacity)
//   - Weight panel resident in BRAM per n0 iteration (TILE_N * GEMM_SHIP_K bytes)
//   - Output accumulator in BRAM per (m0, n0) tile (TILE_M * TILE_N floats)
//   - Residency buffers internally use hls::vector<int8,32> + AGGREGATE compact=bit
//     (preserves A v2.5 URAM breakthrough)
//
// Iteration history (see docs/v2.1_hls_kernel_iteration_notes.md):
//   A v1-v2.5 — dual-residency 4x8x32 unroll + ACT_VEC ABI.
//                A v2.5 reached URAM 32/64 + compute II=2 via AGGREGATE.
//                Residual: load II 36/40, Fmax -0.81 ns, 5 reviews No-ship on
//                the feasibility of 4x8x32 + 256x128 residency for KV260.
//   A v3      — internal reinterpret_cast to ACT_VEC. No effect on port width.
//   A v4      — ACT_VEC* / WGT_VEC* top-level params. csynth failed at clang-LTO.
//                Also hit alignment UB: hls::vector<int8,32> has alignas(32),
//                std::vector<int8_t> does not (adversarial-review #6 high+medium).
//   Path C    — Byte-pointer ABI (2026-04-15) reverted both A v4 blockers.
//                Tile dropped to 64x64x32, unroll 2x2x32. AGGREGATE +
//                Q16->Q24 scale pipeline kept. csim/csynth passed but
//                `[HLS 200-885] limited memory ports` forced load II to 32
//                because the 32-wide UNROLL'd byte reads all compete on one
//                m_axi port — end-to-end prefill gain vs CPU was only ~3%.
//   v2.1+wide — (2026-04-17) act/weight switched to `ap_uint<256>*`. Step 0A
//                POC proved ap_uint<256> has neither the A v4 clang-LTO nor
//                the alignas(32) clash; csynth clean with II=1 and 256-bit
//                RDATA. m_axi depth literals on act/weight renormalized to
//                beat count (÷32).
// ============================================================================

extern "C" void fpga_gemm_kernel(
    ap_uint<256> *act,
    ap_uint<256> *weight,
    SCALE_DTYPE  *act_scales,
    SCALE_DTYPE  *weight_scales,
    OUT_DTYPE    *out,
    int M, int K, int N)
{
// m_axi layout (ABI v2.1+wide, 2026-04-17):
//   - act/weight are native 256-bit beat ports. Each beat = 32 contiguous
//     int8 bytes (one Q8 group). Step 0A POC proved csynth on xck26 emits a
//     256-bit m_axi RDATA bus with II=1 on a pipelined beat-load — root fix
//     for the byte-ABI `[HLS 200-885] limited memory ports` II=32 regression.
//   - Scales on their own bundle gmem2 so the load loop touches two disjoint
//     AR channels per iter (payload + scale) instead of multiplexing.
//   - Bundles map to KV260 HP masters: gmem0 (act in + out out) -> HPC0;
//     gmem1 (weight) -> HPC1; gmem2 (scales) -> HPC1.
//   - depth is in units of the PORT ELEMENT TYPE. For act/weight the element
//     type is `ap_uint<256>` (32 bytes/beat), so ABI-max depth = byte-max / 32.
//     act  : 1024 M * 3072 K / 32 = 98304 beats (was 3145728 bytes)
//     weight: 3072 N * 3072 K / 32 = 294912 beats (was 9437184 bytes)
//     Scales/out ports are unchanged (uint16_t / float elements).
//     `max_widen_bitwidth=256` is dropped on act/weight — csynth confirms it
//     was a no-op on an already-256-bit port ("could not widen since type
//     i256"). Kept on scales/out where narrow element types still benefit.
#pragma HLS INTERFACE m_axi port=act           offset=slave bundle=gmem0 depth=98304   max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=weight        offset=slave bundle=gmem1 depth=294912  max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=act_scales    offset=slave bundle=gmem2 depth=98304   max_read_burst_length=256  max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=weight_scales offset=slave bundle=gmem2 depth=294912  max_read_burst_length=256  max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=out           offset=slave bundle=gmem0 depth=3145728 max_write_burst_length=256 max_widen_bitwidth=256

// KV260 s_axilite byte-strobe workaround: force every arg into its own
// 16B-aligned slot. Non-16B-aligned writes lose WSTRB on this platform
// (both FPD-128b and LPD-32b paths; root cause still in vpl/PFM), so
// scalar args MUST sit on 16B boundaries. See docs/KV260_axilite_128bit_bug_fix.md.
#pragma HLS INTERFACE s_axilite port=act           bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=weight        bundle=control offset=0x20
#pragma HLS INTERFACE s_axilite port=act_scales    bundle=control offset=0x30
#pragma HLS INTERFACE s_axilite port=weight_scales bundle=control offset=0x40
#pragma HLS INTERFACE s_axilite port=out           bundle=control offset=0x50
#pragma HLS INTERFACE s_axilite port=M             bundle=control offset=0x60
#pragma HLS INTERFACE s_axilite port=K             bundle=control offset=0x70
#pragma HLS INTERFACE s_axilite port=N             bundle=control offset=0x80
#pragma HLS INTERFACE s_axilite port=return        bundle=control

    // ------------------------------------------------------------------
    // Vitis HLS 2022.1 workaround: force READ_WRITE on every m_axi port.
    //
    // HLS auto-optimizes ports to READ_ONLY / WRITE_ONLY when it only sees
    // one direction. The collapsed AW/AR sub-channels then break v++ link's
    // dr.bd.tcl. Volatile dummy read+write self-stores inside an
    // unreachable M == -1 branch keep HLS from specializing the ports.
    // (Preserved from v1.1 / v2.0 — regression risk is real.)
    // ------------------------------------------------------------------
    if (M == -1) {
        // Wide ABI: act/weight are ap_uint<256>*, still trivially self-storable.
        volatile ap_uint<256> *act_rw           = act;
        volatile ap_uint<256> *weight_rw        = weight;
        volatile SCALE_DTYPE  *act_scales_rw    = act_scales;
        volatile SCALE_DTYPE  *weight_scales_rw = weight_scales;
        volatile OUT_DTYPE    *out_rw           = out;
        act_rw[0]           = act_rw[0];
        weight_rw[0]        = weight_rw[0];
        act_scales_rw[0]    = act_scales_rw[0];
        weight_scales_rw[0] = weight_scales_rw[0];
        out_rw[0]           = out_rw[0];
        return;
    }

    if (!gemm_dims_valid(M, K, N)) {
        // Zero-out fallback — avoid silent-stale (adversarial-review #5/#6).
        // Bounded by ABI max so garbage M/N never trigger a runaway write.
        if (M > 0 && N > 0 && M <= GEMM_MAX_M && N <= GEMM_MAX_N) {
            zero_invalid: for (int i = 0; i < M * N; ++i) {
#pragma HLS PIPELINE II=1
                out[i] = 0.0f;
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // On-chip buffers (K packed as 32-int8 vector):
    //   act buffers      → URAM  (cyclic factor=2 dim=1, matches TM_UNROLL=2)
    //   weight panel     → BRAM  (cyclic factor=2 dim=1, matches TN_UNROLL=2)
    //   output tile      → BRAM  (TM=64, TN=64 fp32, factor 2x2 banks)
    //
    // Internal residency keeps `hls::vector<int8,32>` (ACT_VEC/WGT_VEC) +
    // AGGREGATE compact=bit — the A v2.5 URAM breakthrough. The public ABI
    // is now `ap_uint<256>*` (one beat = one Q8 group); the load loops split
    // each beat into an ACT_VEC/WGT_VEC before storing into residency.
    // ------------------------------------------------------------------
    static ACT_VEC     act_buf   [GEMM_MAX_M][GEMM_SHIP_K / GEMM_Q8_GROUP_SIZE];
    static SCALE_DTYPE act_sc_buf[GEMM_MAX_M][GEMM_SHIP_K / GEMM_Q8_GROUP_SIZE];
    static WGT_VEC     wgt_buf   [TILE_N][GEMM_SHIP_K / GEMM_Q8_GROUP_SIZE];
    static SCALE_DTYPE wgt_sc_buf[TILE_N][GEMM_SHIP_K / GEMM_Q8_GROUP_SIZE];
    static OUT_DTYPE   out_acc   [TILE_M][TILE_N];

// AGGREGATE compact=bit: force HLS to store each `hls::vector<int8,32>`
// cell as a single 256-bit wide memory word. Without this, csynth A v2
// schedule showed `RAM_T2P_BRAM Ports=2 Width=8 Depth=12288` — i.e. HLS
// unpacked the vector into 32 scalar byte cells, so every "vector read"
// was 32 sequential byte reads → compute II floored at 16 and URAM
// auto-replicated 6× to compensate. See docs/v2.1_hls_kernel_iteration_notes.md #5.
#pragma HLS BIND_STORAGE variable=act_buf    type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=act_sc_buf type=ram_t2p impl=bram
#pragma HLS AGGREGATE variable=act_buf compact=bit
#pragma HLS ARRAY_PARTITION variable=act_buf    cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=act_sc_buf cyclic factor=2 dim=1
#pragma HLS BIND_STORAGE variable=wgt_buf    type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=wgt_sc_buf type=ram_t2p impl=bram
#pragma HLS AGGREGATE variable=wgt_buf compact=bit
#pragma HLS ARRAY_PARTITION variable=wgt_buf    cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=wgt_sc_buf cyclic factor=2 dim=1
#pragma HLS BIND_STORAGE variable=out_acc type=ram_t2p impl=bram
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=2 dim=2

    const int K_G = K / GEMM_Q8_GROUP_SIZE;

    // ======== Load activation to URAM once (whole op) ========
    // `act` is a 256-bit beat pointer; each inner iter reads ONE beat and
    // unpacks it into an ACT_VEC. Beat stride = 32 bytes = 1 Q8 group, so
    // the beat index is `m*K_G + kg` (NOT `m*K + kg*32` which was the byte
    // address in the prior byte-ABI). HLS Step 0A POC confirmed this emits
    // II=1 with one external m_axi read per iter.
    load_act_m: for (int m = 0; m < M; ++m) {
        load_act_kg: for (int kg = 0; kg < K_G; ++kg) {
#pragma HLS PIPELINE II=1
            ap_uint<256> beat = act[m*K_G + kg];
            ACT_VEC tmp;
            load_act_b: for (int b = 0; b < GEMM_Q8_GROUP_SIZE; ++b) {
#pragma HLS UNROLL
                tmp[b] = (ACT_DTYPE)(ap_int<8>)beat.range(b*8+7, b*8);
            }
            act_buf[m][kg]    = tmp;
            act_sc_buf[m][kg] = act_scales[m*K_G + kg];
        }
    }

    // ======== Iterate over N in panels of TN=128 ========
    n_panel: for (int n0 = 0; n0 < N; n0 += TILE_N) {
        // Load weight panel [n0:n0+TN][:K] to BRAM.
        // weight is row-major 256-bit beat pointer [N][K/32]. Beat index =
        // `(n0+tn)*K_G + kg`. One external m_axi read per iter, II=1.
        load_wgt_n: for (int tn = 0; tn < TILE_N; ++tn) {
            load_wgt_kg: for (int kg = 0; kg < K_G; ++kg) {
#pragma HLS PIPELINE II=1
                ap_uint<256> beat = weight[(n0+tn)*K_G + kg];
                WGT_VEC tmp;
                load_wgt_b: for (int b = 0; b < GEMM_Q8_GROUP_SIZE; ++b) {
#pragma HLS UNROLL
                    tmp[b] = (WGT_DTYPE)(ap_int<8>)beat.range(b*8+7, b*8);
                }
                wgt_buf[tn][kg]    = tmp;
                wgt_sc_buf[tn][kg] = weight_scales[(n0+tn)*K_G + kg];
            }
        }

        // ======== Iterate over M in tiles of TILE_M=64 (Path C) ========
        m_tile: for (int m0 = 0; m0 < M; m0 += TILE_M) {
            // Clear output accumulator.
            clr_m: for (int tm = 0; tm < TILE_M; ++tm) {
                clr_n: for (int tn = 0; tn < TILE_N; ++tn) {
#pragma HLS UNROLL factor=2
                    out_acc[tm][tn] = 0.0f;
                }
            }

            // ======== K reduction: microtile-pipelined outer-product =======
            // Path C loop nest (2x2x32 = 128 MACs / cycle microtile):
            //   kg      — rolled, sequential reduction over k-groups
            //   tm_i    — rolled, TILE_M / TM_UNROLL = 32 iters
            //   tn_i    — rolled, PIPELINE II=1 — flattens with tm_i so
            //             each pipelined iter computes ONE 2x2x32 microtile
            //
            // Per-iteration body has two stages:
            //   Stage 1: unpack one ACT_VEC (32B) per dm and one WGT_VEC per
            //            dn into flat scalar FF arrays `a_reg[2][32]` /
            //            `w_reg[2][32]` with `complete dim=0` partition.
            //            This keeps A v2.5's scalar-FF staging (one FF per
            //            scalar cell) so Stage 2 broadcasts cost zero BRAM
            //            port pressure. Scales fp16->Q16 likewise staged.
            //   Stage 2: split 2a (int8 MACs -> acc_int[2][2] ap_int<32> FF)
            //            and 2b (Q16*Q16=Q24 scale apply -> out_acc float).
            //            Split preserved from A v4 to shorten combinational
            //            chains — may or may not be needed at this unroll,
            //            but costs nothing and helps Fmax headroom.
            k_group_loop: for (int kg = 0; kg < K_G; ++kg) {
                tm_outer: for (int tm_i = 0; tm_i < TILE_M; tm_i += TM_UNROLL) {
                    tn_outer: for (int tn_i = 0; tn_i < TILE_N; tn_i += TN_UNROLL) {
#pragma HLS PIPELINE II=1
                        // ---- Stage 1: stage microtile into scalar FF ----
                        ACT_DTYPE                  a_reg [TM_UNROLL][GEMM_Q8_GROUP_SIZE];
                        WGT_DTYPE                  w_reg [TN_UNROLL][GEMM_Q8_GROUP_SIZE];
                        SCALE_INDIV_FIXED_HW_DTYPE as_reg[TM_UNROLL];
                        SCALE_INDIV_FIXED_HW_DTYPE ws_reg[TN_UNROLL];
#pragma HLS ARRAY_PARTITION variable=a_reg  complete dim=0
#pragma HLS ARRAY_PARTITION variable=w_reg  complete dim=0
#pragma HLS ARRAY_PARTITION variable=as_reg complete
#pragma HLS ARRAY_PARTITION variable=ws_reg complete

                        for (int dm = 0; dm < TM_UNROLL; ++dm) {
#pragma HLS UNROLL
                            ACT_VEC av = act_buf[m0 + tm_i + dm][kg];
                            for (int kl = 0; kl < GEMM_Q8_GROUP_SIZE; ++kl) {
#pragma HLS UNROLL
                                a_reg[dm][kl] = av[kl];
                            }
                            as_reg[dm] = fp16_bits_to_fixed_q16(
                                SCALE_BITS_HW_DTYPE(act_sc_buf[m0 + tm_i + dm][kg]));
                        }
                        for (int dn = 0; dn < TN_UNROLL; ++dn) {
#pragma HLS UNROLL
                            WGT_VEC wv = wgt_buf[tn_i + dn][kg];
                            for (int kl = 0; kl < GEMM_Q8_GROUP_SIZE; ++kl) {
#pragma HLS UNROLL
                                w_reg[dn][kl] = wv[kl];
                            }
                            ws_reg[dn] = fp16_bits_to_fixed_q16(
                                SCALE_BITS_HW_DTYPE(wgt_sc_buf[tn_i + dn][kg]));
                        }

                        // ---- Stage 2a: TM_UNROLL × TN_UNROLL int8 MACs -> acc_int FF ----
                        // Break the combinational chain here so Stage 2b's fp
                        // conversion doesn't share a clock window with the
                        // multiply-accumulate tree.
                        ap_int<32> acc_int[TM_UNROLL][TN_UNROLL];
#pragma HLS ARRAY_PARTITION variable=acc_int complete dim=0

                        for (int dm = 0; dm < TM_UNROLL; ++dm) {
#pragma HLS UNROLL
                            for (int dn = 0; dn < TN_UNROLL; ++dn) {
#pragma HLS UNROLL
                                ap_int<32> acc = 0;
                                for (int kl = 0; kl < GEMM_Q8_GROUP_SIZE; ++kl) {
#pragma HLS UNROLL
                                    ap_int<8> a = a_reg[dm][kl];
                                    ap_int<8> w = w_reg[dn][kl];
                                    acc += (ap_int<16>)a * (ap_int<16>)w;
                                }
                                acc_int[dm][dn] = acc;
                            }
                        }

                        // ---- Stage 2b: fp16→Q16→Q24 scale apply → out_acc ----
                        // acc_int max ~2^19, sc_q24 max ~2^31 → product fits
                        // in ap_int<64>. Combining scales first avoids three-way
                        // Q16*Q16*acc overflow.
                        for (int dm = 0; dm < TM_UNROLL; ++dm) {
#pragma HLS UNROLL
                            for (int dn = 0; dn < TN_UNROLL; ++dn) {
#pragma HLS UNROLL
                                SCALE_FIXED_HW_DTYPE sc_q24 =
                                    fixed_combined_scale_q(as_reg[dm], ws_reg[dn]);
                                ap_int<64> scaled_q24 =
                                    (ap_int<64>)acc_int[dm][dn] * (ap_int<64>)sc_q24;
                                float delta = (float)scaled_q24 /
                                              (float)(1u << SCALE_FIXED_BITS);
                                out_acc[tm_i + dm][tn_i + dn] += delta;
                            }
                        }
                    }
                }
            }

            // ======== Write output tile to DDR ========
            // Per-element II=1 write — NOT memcpy. v1.1 learned the hard way
            // that memcpy from a locally-partitioned fp32 array to an m_axi
            // port silently drops the AW channel and every element reads back
            // as 0. See gemm_kernel.cpp pre-v2.1 history (now superseded).
            store_m: for (int tm = 0; tm < TILE_M; ++tm) {
                store_n: for (int tn = 0; tn < TILE_N; ++tn) {
#pragma HLS PIPELINE II=1
                    out[(m0+tm)*N + (n0+tn)] = out_acc[tm][tn];
                }
            }
        }
    }
}
