#include <ap_int.h>
#include <hls_math.h>
#include <stdint.h>

// ---------- hardware limits ----------
#define MAX_N   256
#define MAX_K   960
#define MAX_M   4096

// ---------- tiling parameters ----------
#define BLOCK_M  256
#define TILE_SIZE 32

typedef ap_uint<128> uint128_t;

static float get_f32_lane(uint128_t word, int lane) {
#pragma HLS INLINE
    union {
        uint32_t u;
        float    f;
    } cvt;
    cvt.u = (uint32_t) word(lane * 32 + 31, lane * 32);
    return cvt.f;
}

static uint128_t set_f32_lane(uint128_t word, int lane, float value) {
#pragma HLS INLINE
    union {
        uint32_t u;
        float    f;
    } cvt;
    cvt.f = value;
    word(lane * 32 + 31, lane * 32) = cvt.u;
    return word;
}

static int8_t quantize_to_i8(float val, float scale) {
#pragma HLS INLINE
    int32_t q = (int32_t) hls::roundf(val * scale);
    if (q > 127) {
        q = 127;
    } else if (q < -127) {
        q = -127;
    }
    return (int8_t) q;
}

static void load_f32x4(const uint128_t * data, int idx, float vals[4]) {
#pragma HLS INLINE
    int lane0 = idx & 3;
    uint128_t w0 = data[idx >> 2];
    uint128_t w1 = 0;
    if (lane0 != 0) {
        w1 = data[(idx >> 2) + 1];
    }
    for (int lane = 0; lane < 4; lane++) {
#pragma HLS UNROLL
        int src_lane = lane0 + lane;
        vals[lane] = (src_lane < 4) ? get_f32_lane(w0, src_lane)
                                    : get_f32_lane(w1, src_lane - 4);
    }
}

extern "C"
void mmult_accel(const uint128_t * A, const uint128_t * B_vec, uint128_t * C,
                 int N, int K, int M, int update_A, float scale_B)
{
#pragma HLS INTERFACE m_axi port=A     offset=slave bundle=gmemA depth=(2*MAX_N + MAX_N*MAX_K + 3)/4 max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B_vec offset=slave bundle=gmemB depth=(MAX_K*MAX_M + 3)/4         max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C     offset=slave bundle=gmemC depth=(MAX_N*MAX_M + 3)/4         max_write_burst_length=64 num_write_outstanding=16

#pragma HLS INTERFACE s_axilite port=A        bundle=control
#pragma HLS INTERFACE s_axilite port=B_vec    bundle=control
#pragma HLS INTERFACE s_axilite port=C        bundle=control
#pragma HLS INTERFACE s_axilite port=N        bundle=control
#pragma HLS INTERFACE s_axilite port=K        bundle=control
#pragma HLS INTERFACE s_axilite port=M        bundle=control
#pragma HLS INTERFACE s_axilite port=update_A bundle=control
#pragma HLS INTERFACE s_axilite port=scale_B  bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    // 【优化 1 & 2】：提取大容量数组为 static，并映射到 URAM
    static int8_t A_bram[MAX_N][MAX_K];
    static float  scale_A_bram[MAX_N];
    static float  inv_scale[MAX_N];
    static int8_t B_bram[MAX_K][BLOCK_M]; 

#pragma HLS BIND_STORAGE variable=A_bram type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=A_bram dim=1 factor=32 cyclic

#pragma HLS BIND_STORAGE variable=scale_A_bram type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=scale_A_bram factor=32 cyclic

#pragma HLS BIND_STORAGE variable=inv_scale type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=inv_scale factor=32 cyclic

#pragma HLS BIND_STORAGE variable=B_bram type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=B_bram dim=2 factor=32 cyclic

    const int A_DATA_FLOAT_OFFSET = 2 * MAX_N;

    if (update_A) {
    load_scale_A:
        for (int i = 0; i < MAX_N; i += 4) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N avg=64
#pragma HLS PIPELINE II=1
            if (i < N) {
                uint128_t scale_word = A[i >> 2];
                for (int lane = 0; lane < 4; lane++) {
#pragma HLS UNROLL
                    if (i + lane < N) {
                        scale_A_bram[i + lane] = get_f32_lane(scale_word, lane);
                    }
                }
            }
        }

    quantize_A_rows:
        for (int i = 0; i < MAX_N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N avg=64
            if (i < N) {
                float scale_A = scale_A_bram[i];

            quant_A:
                for (int k = 0; k < MAX_K; k += 4) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_K avg=768
#pragma HLS PIPELINE II=1
                    if (k < K) {
                        float vals[4];
#pragma HLS ARRAY_PARTITION variable=vals complete
                        load_f32x4(A, A_DATA_FLOAT_OFFSET + i * K + k, vals);
                        for (int lane = 0; lane < 4; lane++) {
#pragma HLS UNROLL
                            if (k + lane < K) {
                                A_bram[i][k + lane] = quantize_to_i8(vals[lane], scale_A);
                            }
                        }
                    }
                }
            }
        }
    }

compute_inv_scale:
    for (int i = 0; i < MAX_N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N avg=64
#pragma HLS PIPELINE II=1
        if (i < N) {
            inv_scale[i] = 1.0f / (scale_A_bram[i] * scale_B);
        }
    }

outer_j_block:
    for (int j_block = 0; j_block < M; j_block += BLOCK_M) {
        int current_block_M = ((j_block + BLOCK_M) <= M) ? BLOCK_M : (M - j_block);

    quantize_B_block:
        for (int k = 0; k < MAX_K; k++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_K avg=768
            if (k < K) {
                for (int j = 0; j < BLOCK_M; j += 4) {
#pragma HLS PIPELINE II=1
                    if (j < current_block_M) {
                        float vals[4];
#pragma HLS ARRAY_PARTITION variable=vals complete
                        load_f32x4(B_vec, k * M + j_block + j, vals);
                        for (int lane = 0; lane < 4; lane++) {
#pragma HLS UNROLL
                            if (j + lane < current_block_M) {
                                B_bram[k][j + lane] = quantize_to_i8(vals[lane], scale_B);
                            }
                        }
                    }
                }
            }
        }

    tile_i:
        for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
        tile_j:
            for (int j0 = 0; j0 < current_block_M; j0 += TILE_SIZE) {
                int32_t localC[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=localC dim=0 complete

            init_c:
                for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS UNROLL
                    for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS UNROLL
                        localC[ii][jj] = 0;
                    }
                }

                int8_t a_vec[TILE_SIZE];
                int8_t b_vec[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=a_vec complete
#pragma HLS ARRAY_PARTITION variable=b_vec complete

            k_loop:
                for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                compute_direct:
                    for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
                        int gk = k0 + kk;
                        for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS UNROLL
                            int gi = i0 + ii;
                            a_vec[ii] = (gi < N && gk < K) ? A_bram[gi][gk] : (int8_t) 0;
                        }
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS UNROLL
                            int gj = j0 + jj;
                            b_vec[jj] = (gk < K && gj < current_block_M) ? B_bram[gk][gj] : (int8_t) 0;
                        }
                        
                        // 【优化 3】：严格使用 ap_int<8> 控制 DSP 乘法器位宽推断
                        for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS UNROLL
                            ap_int<8> a_val = a_vec[ii];
                            for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS UNROLL
                                ap_int<8> b_val = b_vec[jj];
                                int32_t mult_res = a_val * b_val;
                                // 8-bit 相乘后，再累加到 32-bit 的 localC 中
                                #pragma HLS BIND_OP variable=mult_res op=mul impl=dsp
                                localC[ii][jj] += mult_res;
                            }
                        }
                    }
                }

            writeC:
                for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS PIPELINE II=1
                    for (int jj = 0; jj < TILE_SIZE; jj += 4) {
#pragma HLS UNROLL
                        int gi = i0 + ii;
                        int gj = j0 + jj;
                        if (gi < N && gj < current_block_M) {
                            int out_idx = gi * M + (j_block + gj);
                            int base_lane = out_idx & 3;
                            if (base_lane == 0 && gj + 3 < current_block_M) {
                                uint128_t word = 0;
                                for (int lane = 0; lane < 4; lane++) {
#pragma HLS UNROLL
                                    float out = (float) localC[ii][jj + lane] * inv_scale[gi];
                                    word = set_f32_lane(word, lane, out);
                                }
                                C[out_idx >> 2] = word;
                            } else {
                                for (int lane = 0; lane < 4; lane++) {
#pragma HLS UNROLL
                                    if (gj + lane < current_block_M) {
                                        int dst_idx = out_idx + lane;
                                        uint128_t word = C[dst_idx >> 2];
                                        float out = (float) localC[ii][jj + lane] * inv_scale[gi];
                                        word = set_f32_lane(word, dst_idx & 3, out);
                                        C[dst_idx >> 2] = word;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}