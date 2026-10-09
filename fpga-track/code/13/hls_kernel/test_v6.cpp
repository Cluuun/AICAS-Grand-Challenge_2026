#include <ap_int.h>
#include "hls_half.h"
#include <stdint.h>


const int N = 16;
const int M = 16;
const int TOT_M = 512;
const int TOT_N = 512;
const int TOT_K = 256;
static const int K = 32;
static const int TILES = TOT_K/K;
// K-tile: t%K in [0,N) systolic; [N,2N) fills dsbk columns (per-tile scale product, float32).
// One dual-port LUTRAM bank: writes use port A while capture reads other columns on port B (no ping-pong).

#ifndef __SYNTHESIS__
static const int RTMAX = (TILES * K) + (2 * N - 2) + N + 16;
ap_int<24> g_res_diag_trace[RTMAX][N + 1][2 * N + 1];
int g_res_diag_trace_len = 0;
#endif

struct PE {
    ap_int<24> acc = 0;

    void step(
        ap_int<8> a_in, ap_int<8> b_in, ap_int<8> &a_out, ap_int<8> &b_out,
        bool ctrl_in, bool &ctrl_h_out, bool &ctrl_v_out,
        ap_int<24> res_in, ap_int<24> &res_out
    ) {
        #pragma HLS INLINE

        res_out = ctrl_in ? acc : res_in;

        // Multiplication and Accumulation
        ap_int<24> product = a_in * b_in;
        #pragma HLS BIND_OP variable=product op=mul impl=dsp

        if (ctrl_in) acc = product;
        else acc += product;

        a_out = a_in;
        b_out = b_in;
        ctrl_h_out = ctrl_in;
        ctrl_v_out = ctrl_in;
    }
};

void sa_wf(
    ap_int<8> a_in_vec[N],
    ap_int<8> b_in_vec[N],
    bool ctrl_start, // 仅从 [0,0] 输入的一个启动脉冲
    ap_int<24> out[2*N-1]
) {
    static ap_int<8> an[N][N + 1];
    static ap_int<8> bn[N + 1][N];

    // 控制信号网格：h 用于水平传递，v 用于垂直传递
    static bool ctrl_h[N][N + 1];
    static bool ctrl_v[N + 1][N];

    static ap_int<24> res_diag[N + 1][2 * N + 1];

    static PE pg[N][N];

    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=pg complete
    #pragma HLS ARRAY_PARTITION variable=an complete
    #pragma HLS ARRAY_PARTITION variable=bn complete
    #pragma HLS ARRAY_PARTITION variable=ctrl_h complete
    #pragma HLS ARRAY_PARTITION variable=ctrl_v complete
    #pragma HLS ARRAY_PARTITION variable=res_diag complete

    // 1. 边界条件
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL
        an[i][0] = a_in_vec[i];
        bn[0][i] = b_in_vec[i];
        ctrl_h[i][0] = (i == 0) ? ctrl_start : false;
        ctrl_v[0][i] = (i == 0) ? ctrl_start : false;
        res_diag[i][0] = 0;
        res_diag[0][i] = 0;
        res_diag[i][N + i] = 0;
    }
    // Clear top-row high diagonal band; it is the source of the shift chain.
    for (int j = 0; j < (N - 1); ++j) {
        #pragma HLS UNROLL
        res_diag[0][N + j] = 0;
    }
    ROW1: for (int i = N - 1; i >= 0; i--) { // 改为倒序
        #pragma HLS UNROLL
        COL2: for (int j = N - 2; j >= 0; j--) {
            #pragma HLS UNROLL
            res_diag[i + 1][N + 1 + j] = res_diag[i][N + j];
        }
    }

    // 2. 阵列逻辑
    ROW: for (int i = N - 1; i >= 0; --i) {
        #pragma HLS UNROLL
        COL: for (int j = N - 1; j >= 0; --j) {
            #pragma HLS UNROLL

            bool current_ctrl = ctrl_h[i][j] | ctrl_v[i][j];

            pg[i][j].step(
                an[i][j], bn[i][j],
                an[i][j+1], bn[i+1][j],
                current_ctrl, ctrl_h[i][j+1], ctrl_v[i+1][j],
                res_diag[i][j], res_diag[i+1][j+1]
            );
        }
    }

    OUT_ROW: for (int i = 0; i < (2 * N - 1); i++) {
        #pragma HLS UNROLL
        out[i] = res_diag[N][i + 1];
    }
#ifndef __SYNTHESIS__
    if (g_res_diag_trace_len < RTMAX) {
        for (int i = 0; i < (N + 1); ++i) {
            for (int j = 0; j < (2 * N + 1); ++j) {
                g_res_diag_trace[g_res_diag_trace_len][i][j] = res_diag[i][j];
            }
        }
        ++g_res_diag_trace_len;
    }
#endif
}

void fa_store_ag(
    const ap_uint<128> w,
    ap_uint<128> ag[TOT_M / M][TOT_K],
    bool vva[TOT_M / M],
    int k_idx,
    int m_idx
) {
    #pragma HLS INLINE
    if (k_idx < TOT_K && !vva[m_idx]) {
        ag[m_idx][k_idx] = w;
    }
    if (k_idx == (TOT_K - 1)) {
        vva[m_idx] = true;
    }
}

void fa_feed(
    ap_int<8> a_in_vec[M],
    ap_uint<128> ag[TOT_M / M][TOT_K],
    ap_int<8> al[M][TOT_K],
    bool vva[TOT_M / M],
    int k_idx,
    int m_idx
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    if (k_idx < TOT_K && vva[m_idx]) {
        ap_uint<128> word = ag[m_idx][k_idx];
        for (int i = 0; i < M; i++) {
            #pragma HLS UNROLL
            al[i][k_idx] = (ap_int<8>)word.range((i << 3) + 7, i << 3);
        }
    }
    for (int i = 0; i < M; i++) {
        #pragma HLS UNROLL
        int idx = k_idx - i;
        if (idx >= 0 && idx < TOT_K) {
            a_in_vec[i] = al[i][idx];
        } else {
            a_in_vec[i] = (ap_int<8>)0;
        }
    }
}

void fb_store_bg(
    const ap_uint<128> w,
    ap_uint<128> bg[TOT_N / N][TOT_K],
    bool vvb[TOT_N / N],
    int k_idx,
    int n_idx
) {
    #pragma HLS INLINE
    if (k_idx < TOT_K && !vvb[n_idx]) {
        bg[n_idx][k_idx] = w;
    }
    if (k_idx == (TOT_K - 1)) {
        vvb[n_idx] = true;
    }
}

void fb_feed(
    ap_int<8> b_in_vec[N],
    ap_uint<128> bg[TOT_N / N][TOT_K],
    ap_int<8> bl[TOT_K][N],
    bool vvb[TOT_N / N],
    int k_idx,
    int n_idx
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    if (k_idx < TOT_K && vvb[n_idx]) {
        ap_uint<128> word = bg[n_idx][k_idx];
        for (int j = 0; j < N; ++j) {
            #pragma HLS UNROLL
            bl[k_idx][j] = (ap_int<8>)word.range((j << 3) + 7, j << 3);
        }
    }
    for (int j = 0; j < N; ++j) {
        #pragma HLS UNROLL
        int idx = k_idx - j;
        if (idx >= 0 && idx < TOT_K) {
            b_in_vec[j] = bl[idx][j];
        } else {
            b_in_vec[j] = (ap_int<8>)0;
        }
    }
}

// One (mt, nt) systolic + dequant capture into pp; then acc folds pp -> c_g.
static void compute_rsa_phase(
    int mt,
    int nt,
    int total_cycles,
    ap_uint<512> pp[TILES][N],
    half dsa[TOT_M][TILES],
    half dsb[TILES][TOT_N],
    float dsbk[M][N],
    ap_uint<128> ag[TOT_M / M][TOT_K],
    ap_int<8> al[M][TOT_K],
    ap_uint<128> bg[TOT_N / N][TOT_K],
    ap_int<8> bl[TOT_K][N],
    bool vva[TOT_M / M],
    bool vvb[TOT_N / N]
) {
    float deq_int[N];
    #pragma HLS ARRAY_PARTITION variable=deq_int complete dim=1

    rsa: for (int t = 0; t < total_cycles; ++t) {
        #pragma HLS PIPELINE II=1

        #pragma HLS DEPENDENCE variable=pp inter false
        #pragma HLS DEPENDENCE variable=vva inter false
        #pragma HLS DEPENDENCE variable=vvb inter false
        #pragma HLS DEPENDENCE variable=ag inter false
        #pragma HLS DEPENDENCE variable=al inter false
        #pragma HLS DEPENDENCE variable=bg inter false
        #pragma HLS DEPENDENCE variable=bl inter false
        ap_int<8> a_in_vec[N];
        ap_int<8> b_in_vec[N];
        ap_int<24> out_diag[2 * N - 1];
        #pragma HLS ARRAY_PARTITION variable=a_in_vec complete
        #pragma HLS ARRAY_PARTITION variable=b_in_vec complete
        #pragma HLS ARRAY_PARTITION variable=out_diag complete

        fa_feed(a_in_vec, ag, al, vva, t, mt);
        fb_feed(b_in_vec, bg, bl, vvb, t, nt);

        const int phase_k = t % K;
        const bool e_dsw =
            (t < (TILES * K))
            && (phase_k >= N)
            && (phase_k < (2 * N));
        if (e_dsw) {
            const int k_tile = t / K;
            const int col_wr = phase_k - N;
            const int m_base = mt * M;
            const int n_scale = nt * N + col_wr;
            dsf: for (int i = 0; i < M; i++) {
                #pragma HLS UNROLL
                // Widen halves before multiply: fp16 hmul flushes subnormals when both
                // operands are < ~1e-3 (typical dequant scales), zeroing the tile path.
                float tmp =
                    (float)dsa[m_base + i][k_tile] * (float)dsb[k_tile][n_scale];
                #pragma HLS BIND_OP variable=tmp op=fmul impl=fulldsp latency=7
                dsbk[i][col_wr] = tmp;
            }
        }

        const bool ctrl_start = ((t % K) == 0) && (t <= (TILES * K));
        sa_wf(a_in_vec, b_in_vec, ctrl_start, out_diag);

        const int x = t - (K + (N - 2));
        const bool x_nonneg = (x >= 0);
        const int tile = x_nonneg ? (x / K) : 0;
        const int m = x_nonneg ? (x % K) : 0;
        const bool enter_capture_accum =
            x_nonneg && (tile < TILES) && (m >= 1) && (m <= N);
        if (enter_capture_accum) {
            const int col = m - 1;
            for (int out_idx = 0; out_idx < N; ++out_idx) {
                #pragma HLS UNROLL
                int d = (N - 1 - out_idx) + col;
                float deq = (float)out_diag[d] * dsbk[out_idx][col];
                #pragma HLS BIND_OP variable=deq op=fmul impl=maxdsp latency=7
                deq_int[out_idx] = deq;
            }
            ap_uint<512> packed;
            for (int i = 0; i < N; ++i) {
                #pragma HLS UNROLL
                union FloatBits {
                    uint32_t u32;
                    float f;
                    FloatBits() : u32(0) {}
                } bits;
                bits.f = deq_int[i];
                packed.range((i + 1) * 32 - 1, i * 32) = bits.u32;
            }
            pp[tile][col] = packed;
        }
    }
}

static void compute_acc_phase(
    int mt,
    int nt,
    ap_uint<512> pp[TILES][N],
    ap_uint<512> c_g[TOT_M / M][TOT_N / N][N]
) {
    float acc_partial[N][N];
    #pragma HLS ARRAY_PARTITION variable=acc_partial complete dim=1
    #pragma HLS ARRAY_PARTITION variable=acc_partial complete dim=2

    acc_init: for (int col = 0; col < N; ++col) {
        #pragma HLS PIPELINE II=1
        ap_uint<512> prev_c = c_g[mt][nt][col];
        acc_init_lane: for (int i = 0; i < N; ++i) {
            #pragma HLS UNROLL
            ap_uint<32> raw = prev_c.range((i + 1) * 32 - 1, i * 32);
            union FloatBits {
                uint32_t u32;
                float f;
                FloatBits() : u32(0) {}
            } bits;
            bits.u32 = (uint32_t)raw.to_uint();
            acc_partial[col][i] = bits.f;
        }
    }

    acc_kt: for (int kt = 0; kt < TILES; ++kt) {
        #pragma HLS PIPELINE II=1
        acc: for (int col = 0; col < N; ++col) {
            #pragma HLS UNROLL factor=4
            ap_uint<512> pp_row = pp[kt][col];
            acc_lane: for (int i = 0; i < N; ++i) {
                #pragma HLS UNROLL
                ap_uint<32> rw =
                    pp_row.range((i + 1) * 32 - 1, i * 32);
                union FloatBits {
                    uint32_t u32;
                    float f;
                    FloatBits() : u32(0) {}
                } bits;
                bits.u32 = (uint32_t)rw.to_uint();
                float sum_next = acc_partial[col][i] + bits.f;
                #pragma HLS BIND_OP variable=sum_next op=fadd impl=fulldsp latency=12
                acc_partial[col][i] = sum_next;
            }
        }
    }

    acc_wr: for (int col = 0; col < N; ++col) {
        #pragma HLS PIPELINE II=1
        ap_uint<512> outw = 0;
        acc_wr_lane: for (int i = 0; i < N; ++i) {
            #pragma HLS UNROLL
            union FloatBits {
                uint32_t u32;
                float f;
                FloatBits() : u32(0) {}
            } outb;
            outb.f = acc_partial[col][i];
            outw.range((i + 1) * 32 - 1, i * 32) = outb.u32;
        }
        c_g[mt][nt][col] = outw;
    }
}

static const int A_WORDS = (TOT_M * TOT_K) / 16;
static const int B_WORDS = (TOT_K * TOT_N) / 16;
static const int C_WORDS = (TOT_M * TOT_N) / 16;
// One 128b word holds 8 half scales along M (or N); one word per (M-bundle, tile) or (N-bundle, tile).
static const int AS_WORDS = (TOT_M / 8) * TILES;
static const int BS_WORDS = (TOT_N / 8) * TILES;

// s_axilite opcode: one phase per ap_start (state in static memories below).
static const int OP_LOAD_AS = 0;
static const int OP_LOAD_BS = 1;
static const int OP_PRELOAD_A = 2;   // iva + pam into fa caches
static const int OP_PRELOAD_B = 3;   // ivb + pbn into fb caches
static const int OP_INIT_C_GLOBAL = 4;
static const int OP_COMPUTE = 5;     // mtx / ntx / rsa
static const int OP_STORE_C = 6;

union HalfBits {
    uint16_t u16;
    half h;
    HalfBits() : u16(0) {}
};

static void task_load_as(
    int use_m,
    const ap_uint<512> AS[AS_WORDS/4],
    half dsa[TOT_M][TILES]
) {
    #pragma HLS INLINE off
    las: for (int i = 0; i < use_m / 8; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        for (int j = 0; j < TILES / 4; ++j) {
            #pragma HLS PIPELINE II=4
            ap_uint<512> word = AS[i * (TILES / 4) + j];
            for (int t = 0; t < 4; ++t) {
                #pragma HLS UNROLL
                ap_uint<128> w128 = word.range(127 + t * 128, t * 128);
                const int tile = j * 4 + t;
                for (int k = 0; k < 8; ++k) {
                    #pragma HLS UNROLL
                    HalfBits bits;
                    bits.u16 = (uint16_t)w128.range((k << 4) + 15, k << 4).to_uint();
                    dsa[i * 8 + k][tile] = bits.h;
                }
            }
        }
    }
}

static void task_load_bs(
    int use_n,
    const ap_uint<512> BS[BS_WORDS/4],
    half dsb[TILES][TOT_N]
) {
    #pragma HLS INLINE off
    lbs: for (int i = 0; i < use_n / 8; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        for (int j = 0; j < TILES / 4; ++j) {
            #pragma HLS PIPELINE II=4
            ap_uint<512> word = BS[i * (TILES / 4) + j];
            for (int t = 0; t < 4; ++t) {
                #pragma HLS UNROLL
                ap_uint<128> w128 = word.range(127 + t * 128, t * 128);
                const int tile = j * 4 + t;
                for (int k = 0; k < 8; ++k) {
                    #pragma HLS UNROLL
                    HalfBits bits;
                    bits.u16 = (uint16_t)w128.range((k << 4) + 15, k << 4).to_uint();
                    dsb[tile][i * 8 + k] = bits.h;
                }
            }
        }
    }
}

static void task_preload_a(
    int n_mt,
    const ap_uint<512> A[A_WORDS/4],
    ap_uint<128> ag[TOT_M / M][TOT_K],
    bool vva[TOT_M / M]
) {
    #pragma HLS INLINE off
    iva: for (int i = 0; i < n_mt; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        vva[i] = false;
    }
    pam: for (int mt = 0; mt < n_mt; ++mt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        pak: for (int k = 0; k < TOT_K / 4; ++k) {
            #pragma HLS PIPELINE II=4
            #pragma HLS LOOP_TRIPCOUNT min=16 max=64
            ap_uint<512> word = A[mt * (TOT_K / 4) + k];
            for (int j = 0; j < 4; ++j) {
                #pragma HLS UNROLL
                ap_uint<128> word_128 = word.range(127 + j * 128, j * 128);
                fa_store_ag(word_128, ag, vva, k * 4 + j, mt);
            }
        }
    }
}

static void task_preload_b(
    int n_nt,
    const ap_uint<512> B[B_WORDS/4],
    ap_uint<128> bg[TOT_N / N][TOT_K],
    bool vvb[TOT_N / N]
) {
    #pragma HLS INLINE off
    ivb: for (int i = 0; i < n_nt; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        vvb[i] = false;
    }
    pbn: for (int nt = 0; nt < n_nt; ++nt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        pbk: for (int k = 0; k < TOT_K / 4; ++k) {
            #pragma HLS PIPELINE II=4
            #pragma HLS LOOP_TRIPCOUNT min=16 max=64
            ap_uint<512> word = B[nt * (TOT_K / 4) + k];
            for (int j = 0; j < 4; ++j) {
                #pragma HLS UNROLL
                ap_uint<128> word_128 = word.range(127 + j * 128, j * 128);
                fb_store_bg(word_128, bg, vvb, k * 4 + j, nt);
            }
        }
    }
}

void test_v6(
    int op,
    int use_m,
    int use_n,
    int c_offset,
    int c_stride,
    const ap_uint<512> A[A_WORDS/4],
    const ap_uint<512> B[B_WORDS/4],
    const ap_uint<512> AS[AS_WORDS/4],
    const ap_uint<512> BS[BS_WORDS/4],
    ap_uint<512> C[C_WORDS]
) {
    #pragma HLS INTERFACE s_axilite port=op bundle=control
    #pragma HLS INTERFACE s_axilite port=use_m bundle=control
    #pragma HLS INTERFACE s_axilite port=use_n bundle=control
    #pragma HLS INTERFACE s_axilite port=c_offset bundle=control
    #pragma HLS INTERFACE s_axilite port=c_stride bundle=control
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 depth=A_WORDS/4 max_read_burst_length=128 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=B_WORDS/4 max_read_burst_length=128 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=AS offset=slave bundle=gmem1 depth=AS_WORDS/4 max_read_burst_length=128 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=BS offset=slave bundle=gmem1 depth=BS_WORDS/4 max_read_burst_length=128 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem1 depth=C_WORDS max_write_burst_length=128 num_write_outstanding=16
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=AS bundle=control
    #pragma HLS INTERFACE s_axilite port=BS bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Persistent across op phases (multi ap_start).
    static ap_uint<512> c_g[TOT_M / M][TOT_N / N][N];
    // Per K-tile dequant column vectors (512b = M FP32 lanes); filled in rsa, summed in acc.
    static ap_uint<512> pp[TILES][N];
    #pragma HLS BIND_STORAGE variable=pp type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=pp type=cyclic factor=4 dim=2
    #pragma HLS BIND_STORAGE variable=c_g type=ram_2p impl=uram

    static half dsa[TOT_M][TILES];
    static half dsb[TILES][TOT_N];
    #pragma HLS BIND_STORAGE variable=dsa type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=dsa type=cyclic factor=M dim=1
    #pragma HLS BIND_STORAGE variable=dsb type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=dsb type=cyclic factor=N dim=2

    static float dsbk[M][N];
    #pragma HLS BIND_STORAGE variable=dsbk type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=dsbk complete dim=1

    static bool vva[TOT_M / M];
    static bool vvb[TOT_N / N];

    static ap_uint<128> ag[TOT_M / M][TOT_K];
    static ap_int<8> al[M][TOT_K];
    static ap_uint<128> bg[TOT_N / N][TOT_K];
    static ap_int<8> bl[TOT_K][N];
    #pragma HLS BIND_STORAGE variable=ag type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=bg type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=al type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=al complete dim=1
    #pragma HLS BIND_STORAGE variable=bl type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=bl complete dim=2

    const int op_v = op;
    const int n_mt = use_m / M;
    const int n_nt = use_n / N;

    if (op_v == OP_LOAD_AS) {
        task_load_as(use_m, AS, dsa);
    } else if (op_v == OP_LOAD_BS) {
        task_load_bs(use_n, BS, dsb);
    } else if (op_v == OP_PRELOAD_A) {
        task_preload_a(n_mt, A, ag, vva);
    } else if (op_v == OP_PRELOAD_B) {
        task_preload_b(n_nt, B, bg, vvb);
    } else if (op_v == OP_INIT_C_GLOBAL) {
    icm: for (int mt = 0; mt < (TOT_M / M); ++mt) {
        icn: for (int nt = 0; nt < (TOT_N / N); ++nt) {
            ici: for (int m = 0; m < N; ++m) {
                #pragma HLS PIPELINE II=1
                c_g[mt][nt][m] = 0;
            }
        }
    }
    } else if (op_v == OP_COMPUTE) {
#ifndef __SYNTHESIS__
    g_res_diag_trace_len = 0;
#endif
    const int TOTAL_MAIN = (TILES * K) + (2 * N - 2) + 1;
    const int TOTAL_CYCLES = TOTAL_MAIN;
    mtx: for (int mt = 0; mt < n_mt; ++mt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        ntx: for (int nt = 0; nt < n_nt; ++nt) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=32
            #pragma HLS PIPELINE off
            compute_rsa_phase(
                mt, nt, TOTAL_CYCLES, pp, dsa, dsb, dsbk,
                ag, al, bg, bl, vva, vvb);
            compute_acc_phase(mt, nt, pp, c_g);
        }
    }
    } else if (op_v == OP_STORE_C) {
    // c_g[mt][nt][j_local]: 512b = 16 FP32; one DDR word per (C^T row j, M-tile mt).
    // base = c_offset + j*c_stride + mt  (c_stride = words per C^T row, usually use_m/16).
    st_nt: for (int nt = 0; nt < n_nt; ++nt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        st_j: for (int j_local = 0; j_local < N; ++j_local) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            const int j = nt * N + j_local;
            st_mt: for (int mt = 0; mt < n_mt; ++mt) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                ap_uint<512> colword = c_g[mt][nt][j_local];
                const int base = c_offset + j * c_stride + mt;
                C[base] = colword;
            }
        }
    }
    }
}