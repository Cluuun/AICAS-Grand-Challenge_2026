#ifndef __INT_REUSE_AXI_COMMON_H__
#define __INT_REUSE_AXI_COMMON_H__

// WEIGHT_AXI 的 WQ 聚合包宽度为 1280 bit，step1 CSim 需要在包含 ap_int 前抬高上限。
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif

#include "reuse_common.h"
#include "utils.h"

// ============================================================================
// Mode-aware AXI mover shared constants
// ============================================================================
// 这些定义服务 M_AXI/WEIGHT_AXI/STATE_AXI 三个 mover。拆分后仍保持同一套
// 256-bit 权重打包比例、128-bit state 打包方式和共享 stream lane 类型，避免
// 三个 case 各自复制一份容易漂移的地址/pack 公式。

constexpr int AXI_G                 = 8;
constexpr int AXI_LLM_T             = REUSE_LLM_TILE_T;
constexpr int AXI_LLM_C             = LLAMA_C;
constexpr int AXI_LLM_CM            = LLAMA_CM;
constexpr int AXI_LLM_VOCAB         = LLAMA_VOCAB;
constexpr int AXI_LLM_H             = LLAMA_H;
constexpr int AXI_LLM_KVH           = LLAMA_KVH;
constexpr int AXI_LLM_HC            = LLAMA_HC;
constexpr int AXI_VIT_T             = VIT_S;
constexpr int AXI_VIT_C             = VIT_C;
constexpr int AXI_VIT_CM            = VIT_MLP_DIM;
constexpr int AXI_VIT_H             = VIT_H;
constexpr int AXI_VIT_GEMM_TT       = AXI_VIT_T / AXI_G;

constexpr int AXI_DW_X_MEM          = 32;
constexpr int AXI_DW_MAXI           = 256;
constexpr int AXI_DW_MAXI_HALF      = AXI_DW_MAXI / 2;
constexpr int AXI_BYTE_PER_MAXI     = AXI_DW_MAXI / 8;
constexpr int AXI_BYTE_PER_HALF     = AXI_DW_MAXI_HALF / 8;
constexpr int AXI_TP                = 1;
constexpr int AXI_CP                = AXI_G;

// WQ/WS 超周期：先按真实 bit 比例求最简整数比，再放大到可对齐 256-bit M_AXI。
constexpr int AXI_WQ_UNIT_BITS       = DW_WQ * AXI_G * AXI_G;
constexpr int AXI_WS_UNIT_BITS       = DW_WS * 2 * AXI_G;
constexpr int AXI_WQ_WS_GCD          = gcd(AXI_WQ_UNIT_BITS, AXI_WS_UNIT_BITS);
constexpr int AXI_WQ_RATIO           = AXI_WQ_UNIT_BITS / AXI_WQ_WS_GCD;
constexpr int AXI_WS_RATIO           = AXI_WS_UNIT_BITS / AXI_WQ_WS_GCD;
constexpr int AXI_WQ_CYCS_AGGR_SIZE  = lcm(AXI_DW_MAXI, AXI_WQ_UNIT_BITS) / AXI_DW_MAXI;
constexpr int AXI_WS_CYCS_AGGR_SIZE  = lcm(AXI_DW_MAXI, AXI_WS_UNIT_BITS) / AXI_DW_MAXI;
constexpr int AXI_K_WQ               = AXI_WQ_CYCS_AGGR_SIZE / gcd(AXI_WQ_RATIO, AXI_WQ_CYCS_AGGR_SIZE);
constexpr int AXI_K_WS               = AXI_WS_CYCS_AGGR_SIZE / gcd(AXI_WS_RATIO, AXI_WS_CYCS_AGGR_SIZE);
constexpr int AXI_K_SUPER            = lcm(AXI_K_WQ, AXI_K_WS);
constexpr int AXI_WQ_CYCS            = AXI_WQ_RATIO * AXI_K_SUPER;
constexpr int AXI_WS_CYCS            = AXI_WS_RATIO * AXI_K_SUPER;
static_assert(AXI_WQ_CYCS % AXI_WQ_CYCS_AGGR_SIZE == 0, "WQ_CYCS must align to WQ_CYCS_AGGR_SIZE");
static_assert(AXI_WS_CYCS % AXI_WS_CYCS_AGGR_SIZE == 0, "WS_CYCS must align to WS_CYCS_AGGR_SIZE");
constexpr int AXI_WQ_CYCS_AGGR_NUM   = AXI_WQ_CYCS / AXI_WQ_CYCS_AGGR_SIZE;
constexpr int AXI_WS_CYCS_AGGR_NUM   = AXI_WS_CYCS / AXI_WS_CYCS_AGGR_SIZE;
static_assert(AXI_DW_MAXI * AXI_WQ_CYCS_AGGR_SIZE % AXI_WQ_UNIT_BITS == 0, "WQ aggregation must align");
static_assert(AXI_DW_MAXI * AXI_WS_CYCS_AGGR_SIZE % AXI_WS_UNIT_BITS == 0, "WS aggregation must align");
constexpr int AXI_WQ_PACK_PER_AGGR   = AXI_DW_MAXI * AXI_WQ_CYCS_AGGR_SIZE / AXI_WQ_UNIT_BITS;
constexpr int AXI_WS_PACK_PER_AGGR   = AXI_DW_MAXI * AXI_WS_CYCS_AGGR_SIZE / AXI_WS_UNIT_BITS;

constexpr int AXI_DECODER_NUM_WQ     = (AXI_LLM_H + 2 * AXI_LLM_KVH) * AXI_LLM_HC * AXI_LLM_C
                                      + AXI_LLM_C * AXI_LLM_C
                                      + 2 * AXI_LLM_C * AXI_LLM_CM
                                      + AXI_LLM_C * AXI_LLM_CM;
constexpr int AXI_DECODER_NUM_WS     = AXI_DECODER_NUM_WQ / AXI_G;
constexpr int AXI_DECODER_W_CYCS     = (AXI_DECODER_NUM_WQ * DW_WQ + AXI_DECODER_NUM_WS * DW_WS * 2) / AXI_DW_MAXI;
constexpr int AXI_DECODER_LOOPS      = AXI_DECODER_W_CYCS / (AXI_WQ_CYCS + AXI_WS_CYCS);
constexpr int AXI_CLS_NUM_WQ         = AXI_LLM_VOCAB * AXI_LLM_C;
constexpr int AXI_CLS_NUM_WS         = AXI_CLS_NUM_WQ / AXI_G;
constexpr int AXI_CLS_W_CYCS         = (AXI_CLS_NUM_WQ * DW_WQ + AXI_CLS_NUM_WS * DW_WS * 2) / AXI_DW_MAXI;
constexpr int AXI_CLS_LOOPS          = AXI_CLS_W_CYCS / (AXI_WQ_CYCS + AXI_WS_CYCS);
constexpr int AXI_VIT_NUM_WQ         = 4 * AXI_VIT_C * AXI_VIT_C + 2 * AXI_VIT_C * AXI_VIT_CM;
constexpr int AXI_VIT_NUM_WS         = AXI_VIT_NUM_WQ / AXI_G;
constexpr int AXI_VIT_W_CYCS         = (AXI_VIT_NUM_WQ * DW_WQ + AXI_VIT_NUM_WS * DW_WS * 2) / AXI_DW_MAXI;
constexpr int AXI_VIT_LOOPS          = AXI_VIT_W_CYCS / (AXI_WQ_CYCS + AXI_WS_CYCS);
constexpr int AXI_W_CYCS_PER_LOOP    = AXI_WQ_CYCS + AXI_WS_CYCS;
constexpr int AXI_WQ_ELEMS_PER_LOOP  = AXI_WQ_PACK_PER_AGGR * AXI_WQ_CYCS_AGGR_NUM * AXI_G * AXI_G;
constexpr int AXI_WS_ELEMS_PER_LOOP  = AXI_WS_PACK_PER_AGGR * AXI_WS_CYCS_AGGR_NUM * AXI_G;
constexpr int AXI_VIT_QKV_LOOPS      = (AXI_VIT_C / AXI_VIT_H) * AXI_VIT_C / AXI_WQ_ELEMS_PER_LOOP;
constexpr int AXI_VIT_O_LOOPS        = AXI_VIT_C * AXI_VIT_C / AXI_WQ_ELEMS_PER_LOOP;
constexpr int AXI_VIT_FC1_LOOPS      = AXI_VIT_CM * AXI_VIT_C / AXI_WQ_ELEMS_PER_LOOP;
constexpr int AXI_VIT_FC2_LOOPS      = AXI_VIT_C * AXI_VIT_CM / AXI_WQ_ELEMS_PER_LOOP;
constexpr int AXI_VIT_QKV_GROUPS     = AXI_VIT_H * 3;
constexpr int AXI_VIT_QKV_LOOP_BASE  = 0;
constexpr int AXI_VIT_O_LOOP_BASE    = AXI_VIT_QKV_LOOP_BASE + AXI_VIT_QKV_GROUPS * AXI_VIT_QKV_LOOPS;
constexpr int AXI_VIT_FC1_LOOP_BASE  = AXI_VIT_O_LOOP_BASE + AXI_VIT_O_LOOPS;
constexpr int AXI_VIT_FC2_LOOP_BASE  = AXI_VIT_FC1_LOOP_BASE + AXI_VIT_FC1_LOOPS;
constexpr int AXI_VIT_STREAM_LOOPS   = AXI_VIT_GEMM_TT * AXI_VIT_LOOPS;
constexpr int AXI_VIT_STREAM_W_CYCS  = AXI_VIT_STREAM_LOOPS * AXI_W_CYCS_PER_LOOP;
static_assert(AXI_DECODER_W_CYCS % (AXI_WQ_CYCS + AXI_WS_CYCS) == 0, "LLM W cycles must align");
static_assert(AXI_CLS_W_CYCS     % (AXI_WQ_CYCS + AXI_WS_CYCS) == 0, "CLS W cycles must align");
static_assert(AXI_VIT_W_CYCS     % (AXI_WQ_CYCS + AXI_WS_CYCS) == 0, "ViT W cycles must align");
static_assert(AXI_WQ_ELEMS_PER_LOOP == AXI_WS_ELEMS_PER_LOOP * AXI_G, "WQ/WS loop element ratio must match GEMM tile");
static_assert((AXI_VIT_C / AXI_VIT_H) * AXI_VIT_C % AXI_WQ_ELEMS_PER_LOOP == 0, "ViT QKV loops must align");
static_assert(AXI_VIT_C * AXI_VIT_C % AXI_WQ_ELEMS_PER_LOOP == 0, "ViT O loops must align");
static_assert(AXI_VIT_CM * AXI_VIT_C % AXI_WQ_ELEMS_PER_LOOP == 0, "ViT FC1 loops must align");
static_assert(AXI_VIT_C * AXI_VIT_CM % AXI_WQ_ELEMS_PER_LOOP == 0, "ViT FC2 loops must align");
static_assert(AXI_VIT_FC2_LOOP_BASE + AXI_VIT_FC2_LOOPS == AXI_VIT_LOOPS, "ViT compact loop map must cover one layer");

// State memory 使用 128-bit 半宽 M_AXI，每个 stream beat 是 8 个 int32 lane，分两拍读写。
constexpr int AXI_X_PACK_PER_TILE    = AXI_TP * AXI_CP * AXI_DW_X_MEM / AXI_DW_MAXI_HALF;
constexpr int AXI_X_PER_PACK         = AXI_DW_MAXI_HALF / AXI_DW_X_MEM;
constexpr int AXI_I32_PER_MAXI       = AXI_DW_MAXI / 32;
constexpr int AXI_I64_PER_MAXI       = AXI_DW_MAXI / 64;
constexpr int AXI_LLM_CT             = AXI_LLM_C / AXI_CP;
constexpr int AXI_VIT_CT             = AXI_VIT_C / AXI_CP;
constexpr int AXI_LLM_NUM_X          = AXI_LLM_T * AXI_LLM_C;
constexpr int AXI_VIT_NUM_X          = AXI_VIT_T * AXI_VIT_C;
constexpr int AXI_LLM_NUM_X_TILE     = AXI_LLM_NUM_X / (AXI_TP * AXI_CP);
constexpr int AXI_VIT_NUM_X_TILE     = AXI_VIT_NUM_X / (AXI_TP * AXI_CP);
constexpr int AXI_LLM_STATE_PACKS    = AXI_LLM_NUM_X_TILE * AXI_X_PACK_PER_TILE;
constexpr int AXI_VIT_STATE_PACKS    = AXI_VIT_NUM_X_TILE * AXI_X_PACK_PER_TILE;
constexpr int AXI_VIT_DEMUX_TP       = REUSE_GEMM_TP;
constexpr int AXI_VIT_TT_D           = AXI_VIT_T / AXI_VIT_DEMUX_TP;
constexpr int AXI_NUM_CLS_INDEX      = AXI_LLM_T;

// 参数 memory pack：bias/gamma 为 int32 pack；ViT beta 为 int64 pack。
constexpr int AXI_VIT_LAYER_BIAS_COUNT = 3 * AXI_VIT_C + AXI_VIT_C + AXI_VIT_CM + AXI_VIT_C;
constexpr int AXI_VIT_LAYER_BIAS_PACKS = AXI_VIT_LAYER_BIAS_COUNT / AXI_I32_PER_MAXI;
constexpr int AXI_VIT_LAYER_BIAS_VECS  = AXI_VIT_LAYER_BIAS_COUNT / AXI_CP;
constexpr int AXI_VIT_LN_PARAM_COUNT   = 2 * AXI_VIT_C;
constexpr int AXI_LLM_LNW_PACKS        = LLAMA_L * 2 * AXI_LLM_C / AXI_I32_PER_MAXI;
constexpr int AXI_CLS_LNW_PACKS        = AXI_LLM_C / AXI_I32_PER_MAXI;
constexpr int AXI_VIT_LNW_PACKS        = VIT_L * AXI_VIT_LN_PARAM_COUNT / AXI_I32_PER_MAXI;
constexpr int AXI_VIT_LNB_PACKS        = VIT_L * AXI_VIT_LN_PARAM_COUNT / AXI_I64_PER_MAXI;
static_assert(AXI_VIT_LAYER_BIAS_COUNT % AXI_I32_PER_MAXI == 0, "ViT bias memory must be 256-bit aligned");
static_assert(AXI_LLM_C % AXI_I32_PER_MAXI == 0, "LLM norm memory must be 256-bit aligned");
static_assert(AXI_VIT_LN_PARAM_COUNT % AXI_I32_PER_MAXI == 0, "ViT norm weight memory must be 256-bit aligned");
static_assert(AXI_VIT_LN_PARAM_COUNT % AXI_I64_PER_MAXI == 0, "ViT norm bias memory must be 256-bit aligned");

typedef ap_int <AXI_DW_MAXI>       axi_maxi_t;
typedef ap_int <AXI_DW_MAXI_HALF>  axi_maxi_half_t;
typedef ap_int <AXI_DW_X_MEM>      axi_x_mem_t;
typedef REUSE_X_T                  axi_x_t;
typedef REUSE_CLS_INDEX_T          axi_cls_index_t;
typedef REUSE_WQ_T                 axi_wq_t;
typedef REUSE_WS_T                 axi_ws_t;
typedef REUSE_DEMUX_BIAS_T         axi_bias_t;
typedef ap_int <const_max(DW_LNW, VIT_DW_LNW)> axi_lnw_t;
typedef ap_int <64>                axi_lnb_t;
typedef ap_uint<AXI_DW_MAXI * AXI_WS_CYCS_AGGR_SIZE> axi_ws_pack_t;
typedef ap_uint<AXI_DW_MAXI * AXI_WQ_CYCS_AGGR_SIZE> axi_wq_pack_t;
typedef ap_uint<AXI_CP * REUSE_DW_DEMUX_BIAS>         axi_bias_pack_t;

typedef hls::vector<axi_x_t,         AXI_TP * AXI_CP> axi_x_vec_t;
typedef hls::vector<axi_wq_t,        AXI_G * AXI_G>   axi_wq_vec_t;
typedef hls::vector<axi_ws_t,        AXI_G>           axi_ws_vec_t;
typedef hls::vector<axi_cls_index_t, AXI_LLM_T>       axi_cls_vec_t;
typedef hls::vector<axi_bias_t,      AXI_CP>          axi_bias_vec_t;
typedef hls::vector<axi_lnw_t,       AXI_CP>          axi_lnw_vec_t;
typedef hls::vector<axi_lnb_t,       AXI_CP>          axi_lnb_vec_t;

typedef ap_uint<1> AXI_PARAM_OP_T;
constexpr int AXI_PARAM_BIAS = 0;
constexpr int AXI_PARAM_NORM = 1;

typedef ap_uint<3> AXI_STATE_OP_T;
constexpr int AXI_STATE_REPLAY_TOKEN = 0;
constexpr int AXI_STATE_REPLAY_DELTA = 1;
constexpr int AXI_STATE_WRITEBACK    = 2;
constexpr int AXI_STATE_WRITE_CLS    = 3;
// Spinal 顶层闭合 RESIDUAL 数据流使用：同一次 layer 调用内按 pass 交替执行
// token-major replay 与 delta-order replay/writeback，避免 y_stream 需要整层大 FIFO。
constexpr int AXI_STATE_LAYER        = 4;

template<typename lane_t>
void axi_load_i32_vec(
    axi_maxi_t* memory,
    int elem_base,
    hls::vector<lane_t, AXI_CP>& vec
) {
    // int32 参数在 DDR 中按 8 lane 打成 256-bit；这里在 CP 粒度统一解包。
    int pack_base = elem_base / AXI_I32_PER_MAXI;
    axi_maxi_t packet = memory[pack_base];
    for (int i = 0; i < AXI_CP; ++i) {
        #pragma HLS unroll
        axi_x_mem_t lane = packet.range(31, 0);
        vec[i] = (lane_t)lane;
        packet >>= 32;
    }
}

static void axi_load_i64_vec(
    axi_maxi_t* memory,
    int elem_base,
    axi_lnb_vec_t& vec
) {
    // ViT LayerNorm beta 为 int64，256-bit memory 一拍只有 4 lane，因此两拍组成 CP=8。
    int pack_base = elem_base / AXI_I64_PER_MAXI;
    for (int p = 0; p < AXI_CP / AXI_I64_PER_MAXI; ++p) {
        #pragma HLS pipeline II=1
        axi_maxi_t packet = memory[pack_base + p];
        for (int i = 0; i < AXI_I64_PER_MAXI; ++i) {
            #pragma HLS unroll
            axi_lnb_t lane = packet.range(63, 0);
            vec[p * AXI_I64_PER_MAXI + i] = lane;
            packet >>= 64;
        }
    }
}

static axi_bias_pack_t axi_pack_bias_vec(axi_bias_vec_t vec) {
    #pragma HLS inline
    axi_bias_pack_t pack = 0;
    for (int i = 0; i < AXI_CP; ++i) {
        #pragma HLS unroll
        ap_int<REUSE_DW_DEMUX_BIAS> lane = vec[i];
        pack.range((i + 1) * REUSE_DW_DEMUX_BIAS - 1, i * REUSE_DW_DEMUX_BIAS) = lane;
    }
    return pack;
}

static axi_bias_vec_t axi_unpack_bias_vec(axi_bias_pack_t pack) {
    #pragma HLS inline
    axi_bias_vec_t vec;
    for (int i = 0; i < AXI_CP; ++i) {
        #pragma HLS unroll
        ap_uint<REUSE_DW_DEMUX_BIAS> bits = pack.range((i + 1) * REUSE_DW_DEMUX_BIAS - 1,
                                                       i * REUSE_DW_DEMUX_BIAS);
        ap_int<REUSE_DW_DEMUX_BIAS> lane;
        lane.range(REUSE_DW_DEMUX_BIAS - 1, 0) = bits;
        vec[i] = (axi_bias_t)lane;
    }
    return vec;
}

static axi_x_vec_t axi_read_state_vec(axi_maxi_half_t* memory_state, int tile_idx) {
    // state memory 采用 token-major tile 排列；每个 tile 是 8 lane int32，分两拍 128-bit 读取。
    axi_x_vec_t vec;
    for (int n_pack = 0; n_pack < AXI_X_PACK_PER_TILE; ++n_pack) {
        #pragma HLS pipeline II=1
        axi_maxi_half_t packet = memory_state[tile_idx * AXI_X_PACK_PER_TILE + n_pack];
        for (int lane_idx = 0; lane_idx < AXI_X_PER_PACK; ++lane_idx) {
            #pragma HLS unroll
            axi_x_mem_t lane = packet.range(AXI_DW_X_MEM - 1, 0);
            vec[n_pack * AXI_X_PER_PACK + lane_idx] = (axi_x_t)lane;
            packet >>= AXI_DW_X_MEM;
        }
    }
    return vec;
}

static void axi_write_state_vec(axi_maxi_half_t* memory_state, int tile_idx, axi_x_vec_t vec) {
    // writeback 仍写回 token-major tile 地址，让下一 pass/layer 可直接从同一 state buffer 重放。
    for (int n_pack = 0; n_pack < AXI_X_PACK_PER_TILE; ++n_pack) {
        #pragma HLS pipeline II=1
        axi_maxi_half_t packet = 0;
        for (int lane_idx = AXI_X_PER_PACK - 1; lane_idx >= 0; --lane_idx) {
            #pragma HLS unroll
            packet <<= AXI_DW_X_MEM;
            packet.range(AXI_DW_X_MEM - 1, 0) = (axi_x_mem_t)vec[n_pack * AXI_X_PER_PACK + lane_idx];
        }
        memory_state[tile_idx * AXI_X_PACK_PER_TILE + n_pack] = packet;
    }
}

#endif // __INT_REUSE_AXI_COMMON_H__
