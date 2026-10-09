#include "gemm_kernel.h"

#define GEMM_DO_PRAGMA(x) _Pragma(#x)
#if defined(__SYNTHESIS__)
#define GEMM_HLS_PRAGMA(x) GEMM_DO_PRAGMA(x)
#else
#define GEMM_HLS_PRAGMA(x)
#endif

namespace {

/* ===================================================================
 * Pack / unpack helpers
 * =================================================================== */

inline int8_t unpack_i8(const ap_uint<128>& w, int byte) {
    GEMM_HLS_PRAGMA(HLS INLINE)
    ap_uint<8> b = w.range(byte * 8 + 7, byte * 8);
    return (int8_t)(signed char) b.to_uint();
}

inline int8_t unpack_i8_256(const ap_uint<256>& w, int byte) {
    GEMM_HLS_PRAGMA(HLS INLINE)
    ap_uint<8> b = w.range(byte * 8 + 7, byte * 8);
    return (int8_t)(signed char) b.to_uint();
}

inline float unpack_f32(const ap_uint<128>& w, int lane) {
    GEMM_HLS_PRAGMA(HLS INLINE)
    ap_uint<32> bits = w.range(lane * 32 + 31, lane * 32);
    union { uint32_t u; float f; } cvt;
    cvt.u = bits.to_uint();
    return cvt.f;
}

inline ap_uint<128> pack_f32x4(float a, float b, float c, float d) {
    GEMM_HLS_PRAGMA(HLS INLINE)
    ap_uint<128> w = 0;
    union { uint32_t u; float f; } x;
    x.f = a; w.range(31, 0)   = x.u;
    x.f = b; w.range(63, 32)  = x.u;
    x.f = c; w.range(95, 64)  = x.u;
    x.f = d; w.range(127, 96) = x.u;
    return w;
}

/* ===================================================================
 * load_a_panel  (FP32 in -> on-chip Q8_0 quantize)
 *
 *  * Reads one FP32 word (4 floats) at a time from a_f32, accumulates
 *    a QK8_0=32-float block (= 8 FP32 words = 2 packed INT8 words).
 *  * Per block: amax tree reduce -> d = amax/127  (d=0 if amax==0).
 *  * Writes 2 packed INT8 words into a_panel and one FP32 d into
 *    a_scale_panel. Layout matches the previous INT8-A path exactly,
 *    so compute_tile / store_c_panel_slice are unchanged.
 *  * Out-of-range rows / k-blocks: zero-fill (unconditional bursts).
 * =================================================================== */
static void load_a_panel(
    const gemm_f32_pack_t* __restrict a_f32,
    int m_macro,
    int m,
    int k,
    int k_blocks,
    ap_uint<128> a_panel      [GEMM_M_PANEL][GEMM_A_PANEL_WORDS],
    float        a_scale_panel[GEMM_M_PANEL][GEMM_MAX_K_BLOCKS]) {
    GEMM_HLS_PRAGMA(HLS INLINE off)
    GEMM_HLS_PRAGMA(HLS ALLOCATION operation instances=fdiv limit=4)

    const int rows_rem   = m - m_macro;
    const int rows_valid = (rows_rem > GEMM_M_PANEL) ? GEMM_M_PANEL
                           : ((rows_rem < 0) ? 0 : rows_rem);

    constexpr int F32_WORDS_PER_BLOCK = QK8_0 / GEMM_PACK_F;  /* 8 */

    /* ---- Valid rows: read FP32 row, quantize per-block, write URAM ---- */
    for (int mm = 0; mm < rows_valid; ++mm) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=128)
        const int row    = m_macro + mm;
        const int base_w = (row * k) / GEMM_PACK_F;   /* FP32 word index */

        for (int kb = 0; kb < k_blocks; ++kb) {
            GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=96)
            GEMM_HLS_PRAGMA(HLS PIPELINE II=8)

            float blk[QK8_0];
            GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=blk complete)

            for (int w = 0; w < F32_WORDS_PER_BLOCK; ++w) {
                GEMM_HLS_PRAGMA(HLS UNROLL)
                ap_uint<128> word = a_f32[base_w + kb * F32_WORDS_PER_BLOCK + w];
                for (int lane = 0; lane < GEMM_PACK_F; ++lane) {
                    GEMM_HLS_PRAGMA(HLS UNROLL)
                    blk[w * GEMM_PACK_F + lane] = unpack_f32(word, lane);
                }
            }

            /* amax tree reduce */
            float amax = 0.0f;
            for (int i = 0; i < QK8_0; ++i) {
                GEMM_HLS_PRAGMA(HLS UNROLL)
                const float v = blk[i];
                const float av = (v < 0.0f) ? -v : v;
                amax = (av > amax) ? av : amax;
            }
            const bool  d_nz   = (amax > 0.0f);
            const float d      = d_nz ? (amax / 127.0f) : 0.0f;
            /* Use x / d to match the host reference exactly (bit-equal INT8). */
            const float quant_d = d_nz ? d : 1.0f;
            a_scale_panel[mm][kb] = d;

            ap_uint<128> w_lo = 0;
            ap_uint<128> w_hi = 0;
            for (int kk = 0; kk < GEMM_PACK_B; ++kk) {
                GEMM_HLS_PRAGMA(HLS UNROLL)
                const float fv_lo = d_nz ? (blk[kk]               / quant_d) : 0.0f;
                const float fv_hi = d_nz ? (blk[kk + GEMM_PACK_B] / quant_d) : 0.0f;
                const int   ri_lo = (int)(fv_lo + (fv_lo >= 0.0f ? 0.5f : -0.5f));
                const int   ri_hi = (int)(fv_hi + (fv_hi >= 0.0f ? 0.5f : -0.5f));
                const int   qi_lo = (ri_lo >  127) ?  127 : ((ri_lo < -128) ? -128 : ri_lo);
                const int   qi_hi = (ri_hi >  127) ?  127 : ((ri_hi < -128) ? -128 : ri_hi);
                w_lo.range(kk * 8 + 7, kk * 8) = ap_uint<8>((int8_t)qi_lo);
                w_hi.range(kk * 8 + 7, kk * 8) = ap_uint<8>((int8_t)qi_hi);
            }
            a_panel[mm][kb * GEMM_WORDS_PER_BLOCK + 0] = w_lo;
            a_panel[mm][kb * GEMM_WORDS_PER_BLOCK + 1] = w_hi;
        }

        /* Zero the tail of this row beyond the live k_blocks. */
        for (int kb = k_blocks; kb < GEMM_MAX_K_BLOCKS; ++kb) {
            GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=0 max=96)
            GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
            a_scale_panel[mm][kb] = 0.0f;
            a_panel[mm][kb * GEMM_WORDS_PER_BLOCK + 0] = 0;
            a_panel[mm][kb * GEMM_WORDS_PER_BLOCK + 1] = 0;
        }
    }

    /* ---- Padding rows beyond rows_valid: full zero ---- */
    for (int mm = rows_valid; mm < GEMM_M_PANEL; ++mm) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=0 max=128)
        for (int w = 0; w < GEMM_A_PANEL_WORDS; ++w) {
            GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
            a_panel[mm][w] = 0;
        }
        for (int kb = 0; kb < GEMM_MAX_K_BLOCKS; ++kb) {
            GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
            a_scale_panel[mm][kb] = 0.0f;
        }
    }
}

/* ===================================================================
 * load_b_tile
 *
 *   * Bytes: unconditional bursts on (valid cols x valid k-words);
 *            tails zero-filled separately.
 *   * Scales: alignment-aware extraction from up-to-two packed words
 *            (kb0 is always a multiple of GEMM_PACK_F because TILE_K is
 *            a multiple of QK8_0*GEMM_PACK_F).
 * =================================================================== */
static void load_b_tile(
    const gemm_b_axi_t*    __restrict b_t_0,
    const gemm_b_axi_t*    __restrict b_t_1,
    const gemm_f32_pack_t* __restrict b_t_scales,
    int col0,
    int k0,
    int n,
    int k,
    int k_blocks,
    ap_uint<256> b_tile_p[GEMM_TILE_N][GEMM_TILE_KB],
    float        b_scale_tile[GEMM_TILE_N][GEMM_TILE_KB]) {
    GEMM_HLS_PRAGMA(HLS INLINE off)

    const int kb0          = k0 / QK8_0;
    const int row_words    = GEMM_TILE_K / GEMM_PACK_B;      /* 8 */
    const int rem_k        = k - k0;
    const int kw_raw       = (rem_k + GEMM_PACK_B - 1) / GEMM_PACK_B;
    const int kw_valid     = (kw_raw > row_words) ? row_words
                              : ((kw_raw < 0) ? 0 : kw_raw);

    const int cols_rem     = n - col0;
    const int cols_valid   = (cols_rem > GEMM_TILE_N) ? GEMM_TILE_N
                              : ((cols_rem < 0) ? 0 : cols_rem);

    /* ---- B bytes: valid cols x kb blocks, packed 256-bit words ---- */
    for (int nn_pair = 0; nn_pair < cols_valid; nn_pair += 2) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=16)

        int col0_idx = col0 + nn_pair;
        int col1_idx = col0 + nn_pair + 1;

        const int base0 = ((col0_idx / 2) * k + k0) / GEMM_PACK_B;
        const int base1 = ((col1_idx / 2) * k + k0) / GEMM_PACK_B;

        for (int kb = 0; kb < GEMM_TILE_KB; ++kb) {
            GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=4)
            GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
            const int w0 = 2 * kb;
            const int w1 = 2 * kb + 1;

            ap_uint<256> packed0 = 0;
            ap_uint<256> packed1 = 0;
            if (w0 < kw_valid) {
                ap_uint<128> chunk0 = b_t_0[base0 + w0];
                if (w1 < kw_valid) {
                    ap_uint<128> chunk1 = b_t_0[base0 + w1];
                    packed0 = (ap_uint<256>(chunk1) << 128) | ap_uint<256>(chunk0);
                } else {
                    packed0 = ap_uint<256>(chunk0);
                }
            }
            if (w0 < kw_valid) {
                ap_uint<128> chunk0 = b_t_1[base1 + w0];
                if (w1 < kw_valid) {
                    ap_uint<128> chunk1 = b_t_1[base1 + w1];
                    packed1 = (ap_uint<256>(chunk1) << 128) | ap_uint<256>(chunk0);
                } else {
                    packed1 = ap_uint<256>(chunk0);
                }
            }
            b_tile_p[nn_pair][kb]     = packed0;
            b_tile_p[nn_pair + 1][kb] = packed1;
        }
    }
    /* ---- B bytes: padding cols ---- */
    for (int nn_pair = cols_valid; nn_pair < GEMM_TILE_N; nn_pair += 2) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=0 max=16)
        for (int kb = 0; kb < GEMM_TILE_KB; ++kb) {
            GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
            b_tile_p[nn_pair][kb]     = 0;
            b_tile_p[nn_pair + 1][kb] = 0;
        }
    }

    /* ---- B scales: alignment-aware extraction ---- */
    for (int nn = 0; nn < cols_valid; ++nn) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=32)
        GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
        const int col       = col0 + nn;
        const int flat_base = col * k_blocks + kb0;
        const int word_base = flat_base / GEMM_PACK_F;
        const int lane_base = flat_base - word_base * GEMM_PACK_F;
        ap_uint<128> w0 = b_t_scales[word_base];
        ap_uint<128> w1 = (lane_base != 0) ? b_t_scales[word_base + 1] : ap_uint<128>(0);
        for (int kb = 0; kb < GEMM_TILE_KB; ++kb) {
            GEMM_HLS_PRAGMA(HLS UNROLL)
            const int  src_lane = lane_base + kb;
            const bool from_w1  = (src_lane >= GEMM_PACK_F);
            const int  real_lane = from_w1 ? (src_lane - GEMM_PACK_F) : src_lane;
            const bool kb_in_range = (kb0 + kb) < k_blocks;
            b_scale_tile[nn][kb] = kb_in_range
                ? unpack_f32(from_w1 ? w1 : w0, real_lane)
                : 0.0f;
        }
    }
    for (int nn = cols_valid; nn < GEMM_TILE_N; ++nn) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=0 max=32)
        GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
        for (int kb = 0; kb < GEMM_TILE_KB; ++kb) {
            GEMM_HLS_PRAGMA(HLS UNROLL)
            b_scale_tile[nn][kb] = 0.0f;
        }
    }
}

/* ===================================================================
 * compute_tile (URAM-aware A access)
 *
 * Per (2-row mm pair, nn0):
 *   * Stage 0 (prologue): build row-specific ab_scale = a_d * b_d
 *                         (one fmul lifted out of the per-cycle chain).
 *   * Stage 1 (II=1):     fetch 32 A bytes for two rows + 32 B bytes,
 *                         pack the two A rows into one DSP multiply per lane,
 *                         and produce two independent row accumulators.
 *   * Stage 2:            sitofp.
 *   * Stage 3:            * ab_scale -> kb_partial.
 *   * Stage 4 (reduce):   sum kb_partial over kb into c_panel.
 * =================================================================== */
static void compute_tile(
    const ap_uint<128> a_panel      [GEMM_M_PANEL][GEMM_A_PANEL_WORDS],
    const float        a_scale_panel[GEMM_M_PANEL][GEMM_MAX_K_BLOCKS],
    const ap_uint<256> b_tile_p[GEMM_TILE_N][GEMM_TILE_KB],
    const float        b_scale_tile[GEMM_TILE_N][GEMM_TILE_KB],
    int row_in_panel,
    int k0,
    int k_blocks,
    float c_panel[GEMM_M_PANEL][GEMM_TILE_N]) {
    GEMM_HLS_PRAGMA(HLS INLINE off)

    const int kb0         = k0 / QK8_0;
    const int word_base_k = k0 / GEMM_PACK_B;
    const int TOTAL_STEPS = (GEMM_TILE_M / GEMM_M_PAR) * (GEMM_TILE_N / GEMM_N_PAR) * GEMM_TILE_KB;

    /* Per-kb partial results for this tile.
     *
     * The previous design accumulated each kblock directly into c_panel with a
     * hand-asserted "DEPENDENCE ... distance=16 RAW" so the float read-modify-
     * write could pipeline at II=1. That schedule is bit-exact in csim but
     * reads a STALE c_panel value on real hardware (and in C/RTL co-sim)
     * whenever a tile spans more than one kblock, silently dropping earlier
     * kblock contributions.
     *
     * Instead, write every step's contribution into a slot keyed by kb that is
     * written exactly once (no loop-carried dependence -> safe II=1), then
     * reduce the kb slots into c_panel once after the step loop. dim1 (kb) is
     * fully split so the reduction reads all kblocks in parallel; dim2/dim3
     * mirror c_panel so the 2-row x 8-lane writes land in independent banks. */
    float acc_kb[GEMM_TILE_KB][GEMM_TILE_M][GEMM_TILE_N];
    GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=acc_kb complete dim=1)
    GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=acc_kb cyclic factor=GEMM_M_PAR dim=2)
    GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=acc_kb cyclic factor=GEMM_N_PAR dim=3)

    for (int step = 0; step < TOTAL_STEPS; ++step) {
        GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=64 max=64)

        int nn0_idx = step % (GEMM_TILE_N / GEMM_N_PAR);
        int tmp     = step / (GEMM_TILE_N / GEMM_N_PAR);
        int mm_pair = tmp % (GEMM_TILE_M / GEMM_M_PAR);
        int kb      = tmp / (GEMM_TILE_M / GEMM_M_PAR);

        int nn0 = nn0_idx * GEMM_N_PAR;
        int row0_idx = row_in_panel + mm_pair * GEMM_M_PAR;
        int row1_idx = row0_idx + 1;
        int lrow0 = mm_pair * GEMM_M_PAR;     /* tile-local row of the pair */
        int lrow1 = lrow0 + 1;

        const int  kb_total    = kb0 + kb;
        const bool kb_in_range = (kb_total < k_blocks);
        const float a_d0 = kb_in_range ? a_scale_panel[row0_idx][kb_total] : 0.0f;
        const float a_d1 = kb_in_range ? a_scale_panel[row1_idx][kb_total] : 0.0f;

        const int w0 = word_base_k + kb * GEMM_WORDS_PER_BLOCK;
        const int w1 = w0 + 1;
        ap_uint<128> a0w0 = a_panel[row0_idx][w0];
        ap_uint<128> a0w1 = a_panel[row0_idx][w1];
        ap_uint<128> a1w0 = a_panel[row1_idx][w0];
        ap_uint<128> a1w1 = a_panel[row1_idx][w1];

        int8_t a0_vals[QK8_0];
        int8_t a1_vals[QK8_0];
        GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=a0_vals complete)
        GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=a1_vals complete)
        for (int b = 0; b < GEMM_PACK_B; ++b) {
            GEMM_HLS_PRAGMA(HLS UNROLL)
            a0_vals[b]               = unpack_i8(a0w0, b);
            a0_vals[b + GEMM_PACK_B] = unpack_i8(a0w1, b);
            a1_vals[b]               = unpack_i8(a1w0, b);
            a1_vals[b + GEMM_PACK_B] = unpack_i8(a1w1, b);
        }

        ap_int<48> partial_lo_packed[GEMM_N_PAR];
        ap_int<48> partial_hi_packed[GEMM_N_PAR];
        GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=partial_lo_packed complete)
        GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=partial_hi_packed complete)
        for (int lane = 0; lane < GEMM_N_PAR; ++lane) {
            GEMM_HLS_PRAGMA(HLS UNROLL)
            partial_lo_packed[lane] = 0;
            partial_hi_packed[lane] = 0;
        }

        ap_uint<256> bw[GEMM_N_PAR];
        GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=bw complete)
        for (int lane = 0; lane < GEMM_N_PAR; ++lane) {
            GEMM_HLS_PRAGMA(HLS UNROLL)
            bw[lane] = b_tile_p[nn0 + lane][kb];
        }

        for (int kk = 0; kk < QK8_0 / 2; ++kk) {
            GEMM_HLS_PRAGMA(HLS UNROLL)
            const int8_t a0_lo = a0_vals[kk];
            const int8_t a1_lo = a1_vals[kk];
            const int8_t a0_hi = a0_vals[kk + QK8_0 / 2];
            const int8_t a1_hi = a1_vals[kk + QK8_0 / 2];
            const ap_int<27> a_lo_packed =
                ((ap_int<27>)a1_lo << 19) + (ap_int<27>)a0_lo;
            const ap_int<27> a_hi_packed =
                ((ap_int<27>)a1_hi << 19) + (ap_int<27>)a0_hi;
            for (int lane = 0; lane < GEMM_N_PAR; ++lane) {
                GEMM_HLS_PRAGMA(HLS UNROLL)
                const int8_t b_lo = unpack_i8_256(bw[lane], kk);
                const int8_t b_hi = unpack_i8_256(bw[lane], kk + QK8_0 / 2);
                partial_lo_packed[lane] += a_lo_packed * (ap_int<18>)b_lo;
                partial_hi_packed[lane] += a_hi_packed * (ap_int<18>)b_hi;
            }
        }

        for (int lane = 0; lane < GEMM_N_PAR; ++lane) {
            GEMM_HLS_PRAGMA(HLS UNROLL)
            ap_int<19> low_lo = partial_lo_packed[lane].range(18, 0);
            ap_int<19> low_hi = partial_hi_packed[lane].range(18, 0);
            ap_int<29> high_lo = partial_lo_packed[lane].range(47, 19) + partial_lo_packed[lane][18];
            ap_int<29> high_hi = partial_hi_packed[lane].range(47, 19) + partial_hi_packed[lane][18];
            int32_t int_sum0 = (int32_t)low_lo + (int32_t)low_hi;
            int32_t int_sum1 = (int32_t)high_lo + (int32_t)high_hi;

            float b_d = b_scale_tile[nn0 + lane][kb];
            float ab_scale0 = a_d0 * b_d;
            float ab_scale1 = a_d1 * b_d;

            float p0 = (float)int_sum0 * ab_scale0;
            float p1 = (float)int_sum1 * ab_scale1;

            /* Single write per (kb, row, col) slot: no loop-carried dependence. */
            acc_kb[kb][lrow0][nn0 + lane] = p0;
            acc_kb[kb][lrow1][nn0 + lane] = p1;
        }
    }

    /* Reduce the per-kb partials into c_panel: overwrite on first k-tile,
     * accumulate on subsequent k-tiles of this column tile. */
    for (int lr = 0; lr < GEMM_TILE_M; ++lr) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=8 max=8)
        for (int cc = 0; cc < GEMM_TILE_N; ++cc) {
            GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
            float s = 0.0f;
            for (int kb = 0; kb < GEMM_TILE_KB; ++kb) {
                GEMM_HLS_PRAGMA(HLS UNROLL)
                s += acc_kb[kb][lr][cc];
            }
            if (kb0 == 0) {
                c_panel[row_in_panel + lr][cc] = s;
            } else {
                c_panel[row_in_panel + lr][cc] += s;
            }
        }
    }
}

/* ===================================================================
 * compute_all_rows
 *
 * Thin wrapper around the row_tile loop so the inner load_b_tile and
 * compute_all_rows can be put inside an HLS DATAFLOW region as two
 * independent tasks (one producer, one consumer).
 * =================================================================== */
static void compute_all_rows(
    const ap_uint<128> a_panel      [GEMM_M_PANEL][GEMM_A_PANEL_WORDS],
    const float        a_scale_panel[GEMM_M_PANEL][GEMM_MAX_K_BLOCKS],
    const ap_uint<256> b_tile_p[GEMM_TILE_N][GEMM_TILE_KB],
    const float        b_scale_tile[GEMM_TILE_N][GEMM_TILE_KB],
    int row_tiles_this_macro,
    int k0,
    int k_blocks,
    float c_panel[GEMM_M_PANEL][GEMM_TILE_N]) {
    GEMM_HLS_PRAGMA(HLS INLINE off)
    for (int row_tile = 0; row_tile < row_tiles_this_macro; ++row_tile) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=16)
        const int row_in_panel = row_tile * GEMM_TILE_M;
        compute_tile(a_panel, a_scale_panel,
                     b_tile_p, b_scale_tile,
                     row_in_panel, k0, k_blocks, c_panel);
    }
}

/* ===================================================================
 * compute_col_tile  (Option 4 / 1B-alpha: DATAFLOW compute task)
 *
 * Stream all k_tiles (load_b + compute); compute_tile initializes c_panel
 * on the first k-tile (kb0==0) and accumulates thereafter.  Kept as a
 * separate INLINE-off function so the col_tile loop can overlap this with
 * store_c_panel_slice on a ping-pong buffer.
 * =================================================================== */
static void compute_col_tile(
    const gemm_b_axi_t*    __restrict b_t_0,
    const gemm_b_axi_t*    __restrict b_t_1,
    const gemm_f32_pack_t* __restrict b_t_scales,
    const ap_uint<128>     a_panel      [GEMM_M_PANEL][GEMM_A_PANEL_WORDS],
    const float            a_scale_panel[GEMM_M_PANEL][GEMM_MAX_K_BLOCKS],
    int col0,
    int n,
    int k,
    int k_blocks,
    int k_tiles,
    int row_tiles_this_macro,
    float c_panel[GEMM_M_PANEL][GEMM_TILE_N]) {
    GEMM_HLS_PRAGMA(HLS INLINE off)
    ap_uint<256> b_tile_p    [GEMM_TILE_N][GEMM_TILE_KB];
    float        b_scale_tile[GEMM_TILE_N][GEMM_TILE_KB];
    GEMM_HLS_PRAGMA(HLS BIND_STORAGE    variable=b_tile_p     type=ram_2p impl=lutram latency=2)
    GEMM_HLS_PRAGMA(HLS BIND_STORAGE    variable=b_scale_tile type=ram_2p impl=lutram)
    GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=b_tile_p     cyclic factor=GEMM_N_PAR dim=1)
    GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=b_tile_p     complete                 dim=2)
    GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=b_scale_tile complete                 dim=0)

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=24)
        const int k0 = k_tile * GEMM_TILE_K;
        load_b_tile(b_t_0, b_t_1, b_t_scales, col0, k0, n, k, k_blocks,
                    b_tile_p, b_scale_tile);
        compute_all_rows(a_panel, a_scale_panel,
                         b_tile_p, b_scale_tile,
                         row_tiles_this_macro, k0, k_blocks,
                         c_panel);
    }
}

/* ===================================================================
 * store_c_panel_slice
 *
 * Per-row 128-bit packed write burst across the valid columns of this
 * column tile. Requires (n % GEMM_PACK_F == 0) which is enforced in
 * gemm_kernel(). No conditional inside the burst loop, so HLS infers a
 * fixed-length write burst per row.
 * =================================================================== */
static void store_c_panel_slice(
    gemm_f32_pack_t* __restrict c,
    int m_macro,
    int m,
    int col0,
    int n,
    const float c_panel[GEMM_M_PANEL][GEMM_TILE_N]) {
    GEMM_HLS_PRAGMA(HLS INLINE off)

    const int rows_rem    = m - m_macro;
    const int rows_valid  = (rows_rem > GEMM_M_PANEL) ? GEMM_M_PANEL
                            : ((rows_rem < 0) ? 0 : rows_rem);
    const int cols_rem    = n - col0;
    const int cols_valid  = (cols_rem > GEMM_TILE_N) ? GEMM_TILE_N
                            : ((cols_rem < 0) ? 0 : cols_rem);
    const int n_words     = cols_valid / GEMM_PACK_F;   /* exact: n%PACK_F == 0 */

    for (int mm = 0; mm < rows_valid; ++mm) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=128)
        const int row       = m_macro + mm;
        const int word_base = (row * n + col0) / GEMM_PACK_F;
        for (int w = 0; w < n_words; ++w) {
            GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=8)
            GEMM_HLS_PRAGMA(HLS PIPELINE II=1)
            const int idx = w * GEMM_PACK_F;
            c[word_base + w] = pack_f32x4(
                c_panel[mm][idx + 0],
                c_panel[mm][idx + 1],
                c_panel[mm][idx + 2],
                c_panel[mm][idx + 3]);
        }
    }
}

} // namespace

/* ===================================================================
 * Top
 * =================================================================== */
void gemm_kernel(
    const gemm_f32_pack_t* __restrict a_f32,
    const gemm_b_axi_t*    __restrict b_t_0,
    const gemm_b_axi_t*    __restrict b_t_1,
    const gemm_f32_pack_t* __restrict b_t_scales,
    gemm_f32_pack_t*       __restrict c,
    int m,
    int n,
    int k) {
    GEMM_HLS_PRAGMA(HLS INTERFACE m_axi port=a_f32        offset=slave bundle=gmem0 \
        depth=786432  max_read_burst_length=128  num_read_outstanding=4 max_widen_bitwidth=128)
    GEMM_HLS_PRAGMA(HLS INTERFACE m_axi port=b_t_0        offset=slave bundle=gmem1 \
        depth=147456  max_read_burst_length=256  num_read_outstanding=16 max_widen_bitwidth=128)
    GEMM_HLS_PRAGMA(HLS INTERFACE m_axi port=b_t_1        offset=slave bundle=gmem4 \
        depth=147456  max_read_burst_length=256  num_read_outstanding=16 max_widen_bitwidth=128)
    GEMM_HLS_PRAGMA(HLS INTERFACE m_axi port=b_t_scales   offset=slave bundle=gmem2 \
        depth=73728   max_read_burst_length=128  num_read_outstanding=4 max_widen_bitwidth=128)
    GEMM_HLS_PRAGMA(HLS INTERFACE m_axi port=c            offset=slave bundle=gmem3 \
        depth=786432  max_write_burst_length=128 num_write_outstanding=4 max_widen_bitwidth=128)

    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=a_f32      bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=b_t_0      bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=b_t_1      bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=b_t_scales bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=c          bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=m          bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=n          bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=k          bundle=control)
    GEMM_HLS_PRAGMA(HLS INTERFACE s_axilite port=return     bundle=control)

    if (m <= 0 || m > GEMM_MAX_M ||
        n <= 0 || n > GEMM_MAX_N ||
        k <= 0 || k > GEMM_MAX_K ||
        (k % QK8_0) != 0 ||
        (n % GEMM_PACK_F) != 0) {
        return;
    }

    const int k_blocks  = k / QK8_0;
    const int col_tiles = (n + GEMM_TILE_N - 1) / GEMM_TILE_N;
    const int k_tiles   = (k + GEMM_TILE_K - 1) / GEMM_TILE_K;

    /* ---- On-chip storage ----
     *
     * A panel: 128 rows x 192 packed 128-bit words = 384 KB.
     *   cyclic factor 2 on dim 2 -> 2 banks of 12288 entries x 128 bit.
     *
     * A scale panel: 128 rows x 96 floats = 48 KB (unpacked floats).
     *   One float per kb so compute_tile uses a single load (no lane shift).
     *
     * C panel: BRAM, cyclic factor N_PAR on dim 2 and M_PAR on dim 1 so
     * the two-row packed compute can update both rows in the same cycle.  */
    ap_uint<128> a_panel      [GEMM_M_PANEL][GEMM_A_PANEL_WORDS];
    float        a_scale_panel[GEMM_M_PANEL][GEMM_MAX_K_BLOCKS];

    /* H1 (Stage 1 timing fix): URAM latency=2 enables the internal output
     * register inside the URAM288 macro. The Vivado WNS=-0.319 ns endpoint
     * at 300 MHz was dominated by URAM clock-to-Q (~2.5 ns); the output
     * register collapses that to ~0.5 ns at the cost of 1 extra cycle of
     * read latency. compute_tile is PIPELINE II=1 so steady-state throughput
     * is unchanged.
     *
     * H3 (Stage 1 timing fix): widen cyclic partition 2 -> 4 so each row
     * exposes 4 URAM banks; halves net fanout to the MAC/DSP cluster and
     * reduces the 0.810 ns route delay on the URAM->DSP path.
     */
    GEMM_HLS_PRAGMA(HLS BIND_STORAGE    variable=a_panel       type=ram_2p impl=uram latency=2)
    GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=a_panel       cyclic factor=4 dim=2)
    GEMM_HLS_PRAGMA(HLS BIND_STORAGE    variable=a_scale_panel type=ram_2p impl=bram)

    for (int m_macro = 0; m_macro < m; m_macro += GEMM_M_PANEL) {
        GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=8)

        const int rem = m - m_macro;
        const int row_tiles_this_macro =
            (rem >= GEMM_M_PANEL) ? GEMM_ROW_TILES_PER_PANEL
                                  : ((rem + GEMM_TILE_M - 1) / GEMM_TILE_M);

        load_a_panel(a_f32, m_macro, m, k, k_blocks,
                     a_panel, a_scale_panel);

        /* Canonical DATAFLOW: a fresh c_panel is declared in the loop body
         * so HLS auto-allocates a ping-pong channel between compute_col_tile
         * (producer) and store_c_panel_slice (consumer). The region holds
         * ONLY one variable declaration + two function calls -> no if/else,
         * so it is a canonical dataflow region (no HLS 214-114). This lets
         * store(col N) overlap with compute(col N+1), hiding the AXI write
         * latency that previously serialized behind compute.                */
        for (int col0 = 0; col0 < n; col0 += GEMM_TILE_N) {
            GEMM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=96)
            GEMM_HLS_PRAGMA(HLS DATAFLOW)

            float c_panel[GEMM_M_PANEL][GEMM_TILE_N];
            GEMM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=c_panel cyclic factor=4 dim=2)

            compute_col_tile(b_t_0, b_t_1, b_t_scales, a_panel, a_scale_panel,
                             col0, n, k, k_blocks, k_tiles,
                             row_tiles_this_macro, c_panel);
            store_c_panel_slice(c, m_macro, m, col0, n, c_panel);
        }
    }
}
