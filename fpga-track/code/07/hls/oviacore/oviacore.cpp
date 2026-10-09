/**
 * OViA: OCM-anchored Vision Accelerator — Core Implementation
 * ============================================================
 *
 * Key Innovation:
 *   Uses OCM (On-Chip Memory, 0xFFFC0000, 256KB) as PS-PL shared memory
 *   bridge to circumvent HLS m_axi AxCACHE=0 cache coherence issue.
 *
 * Architecture:
 *   - 16×16 systolic array for INT8 matrix multiply
 *   - AXI Master to OCM (read A/B input, write C output)
 *   - s_axilite for M/N/K dimension registers
 *   - Standard HLS IP control protocol (start/done via return)
 */

#include "oviacore.h"

// ============================================================
// Processing Element: C += A × B
// ============================================================
static acc_word_t pe_mac(
    dma_word_t a_in, dma_word_t b_in,
    dma_word_t &a_out, dma_word_t &b_out,
    acc_word_t c_in
) {
    #pragma HLS INLINE
    a_out = a_in;
    b_out = b_in;
    return c_in + (acc_word_t)a_in * (acc_word_t)b_in;
}

// ============================================================
// 16×16 Systolic Array
// ============================================================
static void systolic_16x16(
    dma_word_t a_tile[SA_ROWS][SA_ROWS],
    dma_word_t b_tile[SA_COLS][SA_ROWS],
    acc_word_t c_tile[SA_ROWS][SA_COLS]
) {
    #pragma HLS ARRAY_PARTITION variable=a_tile dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=b_tile dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=c_tile dim=0 complete

    // PE pipeline registers
    dma_word_t a_pipe[SA_ROWS][SA_COLS+1];
    dma_word_t b_pipe[SA_ROWS+1][SA_COLS];
    acc_word_t c_reg[SA_ROWS][SA_COLS];
    #pragma HLS ARRAY_PARTITION variable=a_pipe dim=0 complete
    #pragma HLS ARRAY_PARTITION variable=b_pipe dim=0 complete
    #pragma HLS ARRAY_PARTITION variable=c_reg dim=0 complete

    // Clear accumulators
    for (int i = 0; i < SA_ROWS; i++)
        for (int j = 0; j < SA_COLS; j++)
            #pragma HLS UNROLL
            c_reg[i][j] = 0;

    // 3-phase systolic: inject → compute → drain
    for (int t = 0; t < SA_ROWS + SA_COLS + SA_ROWS; t++) {
        #pragma HLS PIPELINE II=1

        // Inject A row [t] from left, B col [t] from top
        if (t < SA_ROWS) {
            for (int j = 0; j < SA_COLS; j++)
                #pragma HLS UNROLL
                a_pipe[t][j] = (j < SA_ROWS) ? a_tile[t][j] : (dma_word_t)0;
        }
        if (t < SA_COLS) {
            for (int i = 0; i < SA_ROWS; i++)
                #pragma HLS UNROLL
                b_pipe[i][t] = (i < SA_ROWS) ? b_tile[t][i] : (dma_word_t)0;
        }

        // Shift data and compute
        for (int i = SA_ROWS-1; i >= 0; i--) {
            for (int j = SA_COLS-1; j >= 0; j--) {
                #pragma HLS UNROLL
                int phase = t - i - j;
                if (phase >= 0 && phase < SA_ROWS) {
                    c_reg[i][j] = pe_mac(
                        a_pipe[i][j], b_pipe[i][j],
                        a_pipe[i][j], b_pipe[i][j],
                        c_reg[i][j]
                    );
                }
                if (i > 0) a_pipe[i-1][j] = a_pipe[i][j];
                if (j > 0) b_pipe[i][j-1] = b_pipe[i][j];
            }
        }
    }

    // Writeback
    for (int i = 0; i < SA_ROWS; i++)
        for (int j = 0; j < SA_COLS; j++)
            #pragma HLS UNROLL
            c_tile[i][j] = c_reg[i][j];
}

// ============================================================
// Tiled Matrix Multiply C[M][N] = A[M][K] × B[K][N]
// OCM Layout: A@0x0000, B@0x4000, C@0x8000
// ============================================================
static void tiled_matmul(
    volatile dma_word_t* ocm,
    int M, int N, int K
) {
    dma_word_t a_buf[SA_ROWS][SA_ROWS];
    dma_word_t b_buf[SA_COLS][SA_ROWS];
    acc_word_t c_buf[SA_ROWS][SA_COLS];
    #pragma HLS ARRAY_PARTITION variable=a_buf dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=b_buf dim=1 complete

    for (int it = 0; it < M; it += SA_ROWS) {
        for (int jt = 0; jt < N; jt += SA_COLS) {

            // Clear accumulator tile
            for (int ti = 0; ti < SA_ROWS; ti++)
                for (int tj = 0; tj < SA_COLS; tj++)
                    #pragma HLS UNROLL
                    c_buf[ti][tj] = 0;

            // Reduce over K tiles
            for (int kt = 0; kt < K; kt += SA_ROWS) {
                // Load A tile from OCM
                for (int ti = 0; ti < SA_ROWS && (it+ti) < M; ti++)
                    for (int tk = 0; tk < SA_ROWS && (kt+tk) < K; tk++)
                        #pragma HLS PIPELINE II=1
                        a_buf[ti][tk] = ocm[(it+ti)*K + (kt+tk)];

                // Load B tile from OCM (transposed for systolic)
                for (int tk = 0; tk < SA_ROWS && (kt+tk) < K; tk++)
                    for (int tj = 0; tj < SA_COLS && (jt+tj) < N; tj++)
                        #pragma HLS PIPELINE II=1
                        b_buf[tj][tk] = ocm[((kt+tk)*N + (jt+tj)) + (OCM_OFFSET_B/sizeof(dma_word_t))];

                // Compute
                systolic_16x16(a_buf, b_buf, c_buf);
            }

            // Store C tile to OCM (4 bytes per INT32 element)
            for (int ti = 0; ti < SA_ROWS && (it+ti) < M; ti++) {
                for (int tj = 0; tj < SA_COLS && (jt+tj) < N; tj++) {
                    #pragma HLS PIPELINE II=1
                    int addr_c = (OCM_OFFSET_C/sizeof(dma_word_t)) + (it+ti)*N + (jt+tj);
                    acc_word_t val = c_buf[ti][tj];
                    ocm[addr_c]     = (dma_word_t)(val & 0xFF);
                    ocm[addr_c+1]   = (dma_word_t)((val >> 8) & 0xFF);
                    ocm[addr_c+2]   = (dma_word_t)((val >> 16) & 0xFF);
                    ocm[addr_c+3]   = (dma_word_t)((val >> 24) & 0xFF);
                }
            }
        }
    }
}

// ============================================================
// Top-Level OViA Core
// Called once via HLS IP control protocol
// ============================================================
void ovia_core(
    volatile dma_word_t* ocm,
    int M, int N, int K
) {
    #pragma HLS INTERFACE m_axi port=ocm depth=65536 bundle=OCM
    #pragma HLS INTERFACE s_axilite port=ocm bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=M bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=N bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=K bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=return bundle=CONTROL

    // Validate dimensions
    if (M > MAX_M || N > MAX_N || K > MAX_K ||
        M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    // Execute tiled matrix multiply
    tiled_matmul(ocm, M, N, K);
}
