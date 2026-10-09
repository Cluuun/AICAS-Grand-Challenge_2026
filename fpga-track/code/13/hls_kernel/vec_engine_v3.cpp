#include <ap_int.h>
#include "hls_stream.h"
#include <stdint.h>
#include <hls_half.h>

ap_uint<128> pack_floats(float f0, float f1, float f2, float f3) {
    // 1. 将 float 的 32 位数据完美重新解释为 ap_uint<32>，不改变底层二进制位
    ap_uint<32> b0 = *reinterpret_cast<ap_uint<32>*>(&f0);
    ap_uint<32> b1 = *reinterpret_cast<ap_uint<32>*>(&f1);
    ap_uint<32> b2 = *reinterpret_cast<ap_uint<32>*>(&f2);
    ap_uint<32> b3 = *reinterpret_cast<ap_uint<32>*>(&f3);
    
    // 2. 使用 HLS 的拼接运算符 (,) 组合成 128 位
    // 注意：最左侧为高位 [127:96]，最右侧为低位 [31:0]
    ap_uint<128> result = (b3, b2, b1, b0);
    
    return result;
}

float raw_to_float(ap_uint<16> in_raw) {
    // 1. 将 ap_uint<16> 的数据地址强转为 half 指型，重新解释位数据
    // 注意：在较新版本的 Vitis HLS 中，为了让流水线和仿真完全一致，这是最常用的位操作手段
    half h_val = *reinterpret_cast<half*>(&in_raw);
    
    // 2. 将 half 转换为 float（HLS 内部会自动生成 Half-to-Float 的 IP 核心，耗费极少逻辑资源）
    float f_val = (float)h_val; 
    
    return f_val;
}

struct mac_chunk {
    ap_int<22> C[64];
};

struct dq_chunk {
    float C[64];
};

struct b_burst_chunk {
    ap_uint<512> B;
};

// New design plan
// We're now using 4 m_axi ports to load on the inner_n dimension. 
// Each inner_n contains 16x256=4096 bytes, perfect for DDR controller. 
// n-tile-1 --> accumulation pipeline 1
// n-tile-2 --> accumulation pipeline 2
// n-tile-3 --> accumulation pipeline 3
// n-tile-4 --> accumulation pipeline 4
// these happens in parallel. 
// To simplify the convention, n_block_num is now the outer_n iteration space. 
// For example, tot_n=512 -> n_block_num=512/(16*4)=8
//              tot_n=448 -> n_block_num=448/(16*4)=7
//              tot_n=320 -> n_block_num=320/(16*4)=5 /* three options on the target model */
// k_block_num is now the outer_k interation space. 
// Each tile has 256 on K dimension, so
//              tot_k=2560 -> k_block_num=2560/256=10
//              tot_k=1024 -> k_block_num=1024/256=4 /* two options on the target model */
// The new iteration space is now: 
// For outer_n in range(n_block_num)
//     For outer_k in range(k_block_num)
//         For inner_k in range(256)
//             For lane in range(64)
//                 int8_mac
//             if inner_k%32==0
//                 dequant --> accumulation pipeline 1

// data layout of B data. 
// 512x256 tile, in total 10 tiles, continuous in memory. 
// Within each tile, 32 16x256 sub-tile. Each subtile is column major. 
// B_in1 point to the starting address of the first tile, first subtile. 
// B_in2 point to the starting address of the first tile, second subtile. 
// B_in3 point to the starting address of the first tile, third subtile. 
// B_in4 point to the starting address of the first tile, fourth subtile. 

static void ve2_read_b_burst(
    const ap_uint<128>* B_in1,
    const ap_uint<128>* B_in2,
    const ap_uint<128>* B_in3,
    const ap_uint<128>* B_in4,
    int k_block_num,
    int n_block_num,
    hls::stream<b_burst_chunk>& s_b
) {

    for (int idn = 0; idn < n_block_num; idn++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 8
        for (int idk = 0; idk < k_block_num; idk++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 10
            int base = idk * (n_block_num * 64 / 16) * 256;
            int offset = idn * (64 / 16) * 256;
            int start_addr = base + offset;

            for (int inner_k = 0; inner_k < 256; inner_k++) {
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 256
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor = 1
                
                b_burst_chunk bd; // 在循环内部定义，避免数据残留
                
                // 4个 Master 独立地址映射，HLS 会自动识别并为其生成 4 个独立的 AXI Burst 通道
                bd.B.range(127, 0)   = B_in1[start_addr + inner_k];
                bd.B.range(255, 128) = B_in2[start_addr + inner_k];
                bd.B.range(383, 256) = B_in3[start_addr + inner_k];
                bd.B.range(511, 384) = B_in4[start_addr + inner_k];
                
                s_b.write(bd);
            }
        }
    }
}

static void ve3_compute_int8_mac(
    ap_int<8> A[2560],
    hls::stream<b_burst_chunk>& s_b,
    int k_block_num,
    int n_block_num,
    hls::stream<mac_chunk>& s_ab_mac
) {
    // scalar on each lane
    for (int idn = 0; idn < n_block_num; idn++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 8
        for (int idk = 0; idk < k_block_num; idk++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 10
            for (int inner_k = 0; inner_k < 8; inner_k++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
                ap_int<22> sum[64];
                for (int lane = 0; lane < 64; lane++) {
#pragma HLS UNROLL
                    sum[lane] = 0;
                }
                for (int most_inner_k = 0; most_inner_k < 32; most_inner_k++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 32
#pragma HLS PIPELINE II = 1
                    ap_int<8> a_i = A[idk * 256 + inner_k * 32 + most_inner_k];
                    b_burst_chunk bd = s_b.read();
                    for (int lane = 0; lane < 64; lane++) {
                        #pragma HLS UNROLL
                        ap_int<8> b_i = bd.B.range(lane * 8 + 7, lane * 8);
                        ap_int<16> prod = a_i * b_i;
                        #pragma HLS BIND_OP variable = prod op = mul impl = dsp
                        sum[lane] += prod;
                    }
                }
                mac_chunk c_chunk;
#pragma HLS ARRAY_PARTITION variable = c_chunk.C complete dim = 1
                for (int lane = 0; lane < 64; lane++) {
#pragma HLS UNROLL
                    c_chunk.C[lane] = sum[lane];
                }
                s_ab_mac.write(c_chunk);
            }
        }
    }
}

// j_group: N pack index 0..63 (8 outputs per 128b word); inner_k: 32-K block inside K-tile.
static ap_uint<128> ve3_load_b_scale_word(
    const ap_uint<128> scale_b_up[32][8][10],
    const ap_uint<128> scale_b_down[32][8][10],
    int j_group,
    int inner_k,
    int idk) {
#pragma HLS INLINE
    if (j_group < 32) {
        return scale_b_up[j_group][inner_k][idk];
    }
    return scale_b_down[j_group - 32][inner_k][idk];
}

static void ve3_dequant(
    hls::stream<mac_chunk>& s_ab_mac,
    const ap_uint<128> a_scale[10],
    const ap_uint<128> scale_b_up[32][8][10],
    const ap_uint<128> scale_b_down[32][8][10],
    int k_block_num,
    int n_block_num,
    hls::stream<dq_chunk>& s_d0
) {

    for (int idn = 0; idn < n_block_num; idn++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 8
        for (int idk = 0; idk < k_block_num; idk++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 10
            ap_uint<128> a_scale_word = a_scale[idk];
            for (int inner_k = 0; inner_k < 8; inner_k++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
                ap_uint<16> a_scale_i =
                    a_scale_word.range(inner_k * 16 + 15, inner_k * 16);
                const float a_scale_f = raw_to_float(a_scale_i);
                ap_uint<128> b_scale_word_tmp[8];
#pragma HLS ARRAY_PARTITION variable = b_scale_word_tmp complete dim = 1
            ld_b_scale:
                for (int outer_lane = 0; outer_lane < 8; outer_lane++) {
#pragma HLS PIPELINE II = 1
                    const int j_group = idn * 8 + outer_lane;
                    b_scale_word_tmp[outer_lane] = ve3_load_b_scale_word(
                        scale_b_up, scale_b_down, j_group, inner_k, idk);
                }
                mac_chunk c_chunk = s_ab_mac.read();
                dq_chunk dq_out;
#pragma HLS ARRAY_PARTITION variable = dq_out.C complete dim = 1
                for (int outer_lane = 0; outer_lane < 8; outer_lane++) {
#pragma HLS UNROLL
                    const ap_uint<128> b_scale_word = b_scale_word_tmp[outer_lane];
                    for (int inner_lane = 0; inner_lane < 8; inner_lane++) {
#pragma HLS UNROLL
                        const int lane = outer_lane * 8 + inner_lane;
                        ap_uint<16> b_scale_i =
                            b_scale_word.range(inner_lane * 16 + 15, inner_lane * 16);
                        const float b_scale_f = raw_to_float(b_scale_i);
                        const float scale = a_scale_f * b_scale_f;
                        float prod = (float)c_chunk.C[lane] * scale;
#pragma HLS BIND_OP variable = prod op = fmul impl = maxdsp latency = 7
                        dq_out.C[lane] = prod;
                    }
                }
                s_d0.write(dq_out);
            }
        }
    }
}

static void ve3_acc(
    hls::stream<dq_chunk>& s_d0,
    int k_block_num,
    int n_block_num,
    float C[512]
) {

    for (int idn = 0; idn < n_block_num; idn++) {
#pragma HLS LOOP_TRIPCOUNT min = 5 max = 8
        float acc_buf[64];
#pragma HLS ARRAY_PARTITION variable = acc_buf complete dim = 1
        for (int lane = 0; lane < 64; lane++) {
            #pragma HLS UNROLL
            acc_buf[lane] = 0;
        }
        for (int idk = 0; idk < k_block_num; idk++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 10
            for (int inner_k = 0; inner_k < 8; inner_k++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 8
                dq_chunk dq_chunk = s_d0.read();
                for (int lane = 0; lane < 64; lane++) {
                    #pragma HLS UNROLL
                    float sum = acc_buf[lane] + dq_chunk.C[lane];
                    #pragma HLS BIND_OP variable = sum op = fadd impl = fulldsp latency = 12
                    acc_buf[lane] = sum;
                }
            }
        }
        for (int lane = 0; lane < 64; lane++) {
            #pragma HLS UNROLL
            C[idn * 64 + lane] = acc_buf[lane];
        }
    }
}


// end new design plan

static void ve3_dfs(
    ap_int<8> A[2560],
    const ap_uint<128>* B_in1,
    const ap_uint<128>* B_in2,
    const ap_uint<128>* B_in3,
    const ap_uint<128>* B_in4,
    const ap_uint<128> scale_a_local[10],
    const ap_uint<128> scale_b_local_up[32][8][10],
    const ap_uint<128> scale_b_local_down[32][8][10],
    int k_block_num,
    int n_block_num,
    float C_buf[512]) {
    hls::stream<b_burst_chunk> s_b("s_b");
    hls::stream<mac_chunk> s_ab_mac("s_ab_mac");
    hls::stream<dq_chunk> s_d0("s_d0");
#pragma HLS STREAM variable = s_b depth = 64
#pragma HLS STREAM variable = s_ab_mac depth = 16
#pragma HLS STREAM variable = s_d0 depth = 16
#pragma HLS BIND_STORAGE variable = s_b type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_ab_mac type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = s_d0 type = fifo impl = srl
#pragma HLS DATAFLOW
    ve2_read_b_burst(B_in1, B_in2, B_in3, B_in4, k_block_num, n_block_num, s_b);
    ve3_compute_int8_mac(A, s_b, k_block_num, n_block_num, s_ab_mac);
    ve3_dequant(s_ab_mac, scale_a_local, scale_b_local_up, scale_b_local_down,
                k_block_num, n_block_num, s_d0);
    ve3_acc(s_d0, k_block_num, n_block_num, C_buf);
}

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
    ap_uint<128>* C_out) {
#pragma HLS INTERFACE m_axi port = A_in offset = slave bundle = gmem0 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = B_in1 offset = slave bundle = gmem0 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = B_in2 offset = slave bundle = gmem1 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = B_in3 offset = slave bundle = gmem2 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = B_in4 offset = slave bundle = gmem3 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = C_out offset = slave bundle = gmem0 max_write_burst_length = 128 num_write_outstanding=64
#pragma HLS INTERFACE m_axi port = scales1 offset = slave bundle = gmem0 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = scales2 offset = slave bundle = gmem1 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = scales3 offset = slave bundle = gmem2 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE m_axi port = scales4 offset = slave bundle = gmem3 max_read_burst_length = 128 num_read_outstanding=64
#pragma HLS INTERFACE s_axilite port = A_in bundle = control
#pragma HLS INTERFACE s_axilite port = B_in1 bundle = control
#pragma HLS INTERFACE s_axilite port = B_in2 bundle = control
#pragma HLS INTERFACE s_axilite port = B_in3 bundle = control
#pragma HLS INTERFACE s_axilite port = B_in4 bundle = control
#pragma HLS INTERFACE s_axilite port = scales1 bundle = control
#pragma HLS INTERFACE s_axilite port = scales2 bundle = control
#pragma HLS INTERFACE s_axilite port = scales3 bundle = control
#pragma HLS INTERFACE s_axilite port = scales4 bundle = control
#pragma HLS INTERFACE s_axilite port = C_out bundle = control
#pragma HLS INTERFACE s_axilite port = k_block_num bundle = control
#pragma HLS INTERFACE s_axilite port = n_block_num bundle = control
#pragma HLS INTERFACE s_axilite port = op bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // op: 0=LOAD_A, 1=LOAD_SCALE_B, 2=LOAD_SCALE_A, 3=COMPUTE, 4=READ_C_BUF
    if (op != 0 && op != 1 && op != 2 && op != 3 && op != 4) {
        return;
    }

    if (n_block_num > 8 || (n_block_num < 5)) {
        return;
    }

    if (k_block_num > 10 || (k_block_num % 2 != 0) || (k_block_num < 4)) {
        return;
    }

    static ap_int<8> A[2560];
    static ap_uint<128> scale_b_local_up[32][8][10];
    static ap_uint<128> scale_b_local_down[32][8][10];
    static ap_uint<128> scale_a_local[10];
    static float C_buf[512];

#pragma HLS ARRAY_PARTITION variable = A cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = scale_b_local_up type=cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = scale_b_local_up type=cyclic factor = 2 dim = 3
#pragma HLS ARRAY_PARTITION variable = scale_b_local_down type=cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = scale_b_local_down type=cyclic factor = 2 dim = 3
// #pragma HLS ARRAY_PARTITION variable = scale_a_local complete dim = 1
#pragma HLS ARRAY_PARTITION variable = C_buf cyclic factor = 64 dim = 1

#pragma HLS BIND_STORAGE variable = A type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = scale_b_local_up type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = scale_b_local_down type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = scale_a_local type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = C_buf type = ram_2p impl = lutram

    if (op == 0) {
    ld_a:
        for (int a_idx = 0; a_idx < k_block_num * 16; a_idx++) {
#pragma HLS LOOP_TRIPCOUNT min = 64 max = 160
#pragma HLS PIPELINE II = 1
            const ap_uint<128> a_word = A_in[a_idx];
            for (int m = 0; m < 16; m++) {
#pragma HLS UNROLL
                const int bit_lo = m * 8;
                A[a_idx * 16 + m] = a_word.range(bit_lo + 7, bit_lo);
            }
        }
        return;
    }

// B-scale DDR layout (per K-tile pair idk_outer):
//   512 words = 64 j_groups x 8 inner_k; each 128b word packs 8 fp16 on N (lane%8).
//   scales1/scales3: j_group 0..31 (N 0..255), scales2/scales4: j_group 32..63 (N 256..511).
//   BRAM: scale_b_{up,down}[j_group%32][inner_k][idk]; load idn_ddr=j*8+inner_k. 
    if (op == 1) {
    ld_sc_b:
        for (int idk_outer = 0; idk_outer < k_block_num / 2; idk_outer++) {
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 5
            const int offset = idk_outer * (4096 * 4 * 8) / 128;
            for (int idn = 0; idn < 256; idn++) {
                int idx = idn / 8; 
                int idk = idn % 8;
#pragma HLS PIPELINE II = 1
                scale_b_local_up[idx][idk][idk_outer * 2] =
                    scales1[offset + idn];
                scale_b_local_down[idx][idk][idk_outer * 2] =
                    scales2[offset + idn];
                scale_b_local_up[idx][idk][idk_outer * 2 + 1] =
                    scales3[offset + idn];
                scale_b_local_down[idx][idk][idk_outer * 2 + 1] =
                    scales4[offset + idn];
            }
        }
        return;
    }

    if (op == 2) {
    ld_sc_a:
        for (int idk = 0; idk < k_block_num; idk++) {
#pragma HLS LOOP_TRIPCOUNT min = 4 max = 10
#pragma HLS PIPELINE II = 1
            scale_a_local[idk] = scales1[idk];
        }
        return;
    }

    if (op == 3) {
        ve3_dfs(A, B_in1, B_in2, B_in3, B_in4, scale_a_local, scale_b_local_up,
                scale_b_local_down, k_block_num, n_block_num, C_buf);
        return;
    }

    if (op == 4) {
    read_c_buf:
        for (int idx = 0; idx < n_block_num * 16; idx++) {
#pragma HLS LOOP_TRIPCOUNT min = 80 max = 128
#pragma HLS PIPELINE II = 1
            const ap_uint<128> packed = pack_floats(
                C_buf[idx * 4 + 0], C_buf[idx * 4 + 1], C_buf[idx * 4 + 2],
                C_buf[idx * 4 + 3]);
            C_out[idx] = packed;
        }
        return;
    }
}
