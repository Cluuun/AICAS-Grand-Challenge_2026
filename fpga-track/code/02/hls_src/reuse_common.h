#ifndef __INT_REUSE_COMMON_H__
#define __INT_REUSE_COMMON_H__

#include "common.h"
#include "quantizer.h"

#ifndef __SYNTHESIS__
#include <cstdlib>
#endif

// ============================================================================
// Mode-aware HLS reuse primitives
// ============================================================================
// These helpers freeze the external mode contract shared by reused LLM/ViT IP.
// Keep runtime mode selection at module/layer/tile boundaries; do not put these
// branches inside TP/CP unrolled inner loops.

typedef ap_uint<1> REUSE_MODE_T;

constexpr int MODE_LLM = 0;
constexpr int MODE_VIT = 1;

constexpr int REUSE_LLM_TILE_T = 8;
constexpr int REUSE_GEMM_TP    = 8;
constexpr int REUSE_CP         = 8;
constexpr int REUSE_GEMM_LANES = REUSE_GEMM_TP * REUSE_CP;

static_assert(LLAMA_G == REUSE_CP, "LLM group/CP must stay 8 for reuse IP");
static_assert(VIT_G    == REUSE_CP, "ViT group/CP must stay 8 for reuse IP");
static_assert(LLAMA_HC == VIT_HC,   "LLM/ViT head dim must match for attention reuse");
static_assert(DW_AQ    == VIT_DW_AQ, "LLM/ViT activation q width must match");
static_assert(DW_AS    == VIT_DW_AS, "LLM/ViT activation scale width must match");
static_assert(DW_WQ    == VIT_DW_WQ, "LLM/ViT weight q width must match");
static_assert(DW_WS    == VIT_DW_WS, "LLM/ViT weight scale width must match");

inline bool is_vit_mode(REUSE_MODE_T mode) {
    #pragma HLS inline
    return mode == MODE_VIT;
}

inline bool is_llm_mode(REUSE_MODE_T mode) {
    #pragma HLS inline
    return mode == MODE_LLM;
}

inline int reuse_layer_limit(REUSE_MODE_T mode) {
    #pragma HLS inline
    return is_vit_mode(mode) ? VIT_L : LLAMA_L;
}

inline int reuse_layer_limit_with_cls(REUSE_MODE_T mode) {
    #pragma HLS inline
    return is_vit_mode(mode) ? VIT_L : (LLAMA_L + 1);
}

inline bool reuse_is_valid_layer(REUSE_MODE_T mode, int layer, bool allow_llm_cls = false) {
    #pragma HLS inline
    int limit = allow_llm_cls ? reuse_layer_limit_with_cls(mode) : reuse_layer_limit(mode);
    return layer >= 0 && layer < limit;
}

inline bool reuse_is_llm_cls(REUSE_MODE_T mode, int layer) {
    #pragma HLS inline
    return is_llm_mode(mode) && layer == LLAMA_L;
}

struct reuse_pos_t {
    int chunk;
    int chunk_pos;
};

inline reuse_pos_t reuse_decode_pos(REUSE_MODE_T mode, int pos) {
    #pragma HLS inline
    reuse_pos_t decoded;
    decoded.chunk = 0;
    decoded.chunk_pos = 0;
    if (is_llm_mode(mode)) {
        decoded.chunk = pos / REUSE_LLM_TILE_T;
        decoded.chunk_pos = pos & (REUSE_LLM_TILE_T - 1);
    }
    return decoded;
}

inline int reuse_llm_valid_s(int pos) {
    #pragma HLS inline
    int valid_s = pos + 1;
    if (valid_s < 1) {
        valid_s = 1;
    }
    return (valid_s > LLAMA_S) ? LLAMA_S : valid_s;
}

inline int reuse_attention_valid_s(REUSE_MODE_T mode, int pos) {
    #pragma HLS inline
    // LLM causal decode 只需要读到当前绝对 pos；ViT 必须保留完整 1024 token。
    return is_vit_mode(mode) ? LLAMA_S : reuse_llm_valid_s(pos);
}

inline int reuse_attention_valid_st(REUSE_MODE_T mode, int pos) {
    #pragma HLS inline
    int valid_s = reuse_attention_valid_s(mode, pos);
    return (valid_s + REUSE_CP - 1) / REUSE_CP;
}

#ifndef __SYNTHESIS__
inline int reuse_tb_vit_layer() {
    // CSim 专用：允许临时只重生成某个 ViT layer 的 condense，不影响 HLS top 接口和综合路径。
    const char *env = std::getenv("VLM_VIT_LAYER");
    return (env && env[0] != '\0') ? std::atoi(env) : 0;
}

inline bool reuse_tb_only_vit() {
    // CSim 专用：layer1 数据准备时跳过 LLM/CLS，避免无关路径拖慢或覆盖调试输出。
    const char *env = std::getenv("VLM_ONLY_VIT");
    return env && env[0] != '\0' && env[0] != '0';
}

inline bool reuse_tb_only_llm() {
    // CSim 专用：定位 LLM-only 问题时跳过 ViT 大窗口，避免无关路径拖慢。
    const char *env = std::getenv("VLM_ONLY_LLM");
    return env && env[0] != '\0' && env[0] != '0';
}

inline int reuse_tb_llm_layer() {
    // CSim 专用：允许 PYNQ golden 定位时复测指定 LLM layer，不影响综合接口。
    const char *env = std::getenv("VLM_LLM_LAYER");
    return (env && env[0] != '\0') ? std::atoi(env) : 0;
}

inline int reuse_tb_llm_pos(int default_pos) {
    // CSim 专用：允许复测低 POS 或 CLS POS 窗口，用来对齐板上 golden smoke。
    const char *env = std::getenv("VLM_LLM_POS");
    return (env && env[0] != '\0') ? std::atoi(env) : default_pos;
}

inline int reuse_tb_llm_top_pos(int default_top_pos) {
    // CSim 专用：允许覆盖当前 tile 的真实最后 token，用来回归 prompt partial tile。
    const char *env = std::getenv("VLM_LLM_TOP_POS");
    return (env && env[0] != '\0') ? std::atoi(env) : default_top_pos;
}
#else
inline int reuse_tb_vit_layer() {
    #pragma HLS inline
    return 0;
}

inline bool reuse_tb_only_vit() {
    #pragma HLS inline
    return false;
}

inline bool reuse_tb_only_llm() {
    #pragma HLS inline
    return false;
}

inline int reuse_tb_llm_layer() {
    #pragma HLS inline
    return 0;
}

inline int reuse_tb_llm_pos(int default_pos) {
    #pragma HLS inline
    return default_pos;
}

inline int reuse_tb_llm_top_pos(int default_top_pos) {
    #pragma HLS inline
    return default_top_pos;
}
#endif

// ============================================================================
// Shared max-width data types for mode-aware stream boundaries
// ============================================================================

constexpr int REUSE_DW_X            = const_max(DW_X,            VIT_DW_X);
constexpr int REUSE_DW_XLN          = const_max(DW_XLN,          VIT_DW_XLN);
constexpr int REUSE_DW_QKV_TRUNC    = const_max(DW_QKV_TRUNC,    VIT_DW_QKV_TRUNC);
constexpr int REUSE_DW_GEMM         = const_max(DW_GEMM,         VIT_DW_GEMM);
constexpr int REUSE_DW_GEMM_ACC     = const_max(DW_GEMM_ACC,     VIT_DW_GEMM_ACC);
constexpr int REUSE_DW_GEMM_TRUNC   = const_max(DW_GEMM_TRUNC,   VIT_DW_GEMM_TRUNC);
constexpr int REUSE_DW_R_TRUNC      = const_max(DW_R_TRUNC,      VIT_DW_R_TRUNC);
constexpr int REUSE_DW_R_MASKED     = const_max(DW_R_MASKED,     VIT_DW_R_MASKED);
constexpr int REUSE_DW_SOFTMAX      = const_max(DW_SOFTMAX,      VIT_DW_SOFTMAX);
constexpr int REUSE_DW_A_TRUNC      = const_max(DW_A_TRUNC,      VIT_DW_A_TRUNC);
constexpr int REUSE_DW_V_TRUNC      = const_max(DW_QKV_TRUNC,    VIT_DW_QKV_TRUNC);
constexpr int REUSE_DW_MLP1_TRUNC   = const_max(DW_XUG_TRUNC,    VIT_DW_FC1_TRUNC);
constexpr int REUSE_DW_MLP2_TRUNC   = const_max(DW_XD_TRUNC,     VIT_DW_FC2_TRUNC);
constexpr int REUSE_DW_MLP_OUT_Q    = DW_AQ;
constexpr int REUSE_DW_CLS_TRUNC    = DW_CLS_TRUNC;
constexpr int REUSE_DW_CLS_INDEX    = DW_CLS_INDEX;
// DEMUX 的 ViT bias 由外部调度按拍输入，宽度覆盖 QKV/O/FC1/FC2 的本地 bias 位宽。
constexpr int REUSE_DW_DEMUX_BIAS   = const_max(VIT_DW_QKV_TRUNC, VIT_DW_FC1_TRUNC, VIT_DW_FC2_TRUNC);

typedef ap_int <REUSE_DW_X          > REUSE_X_T;
typedef ap_int <REUSE_DW_XLN        > REUSE_XLN_T;
typedef ap_int <REUSE_DW_QKV_TRUNC  > REUSE_QKV_TRUNC_T;
typedef ap_int <REUSE_DW_GEMM       > REUSE_GEMM_T;
typedef ap_int <REUSE_DW_GEMM_ACC   > REUSE_GEMM_ACC_T;
typedef ap_int <REUSE_DW_GEMM_TRUNC > REUSE_GEMM_TRUNC_T;
typedef ap_int <REUSE_DW_R_TRUNC    > REUSE_R_TRUNC_T;
typedef ap_int <REUSE_DW_R_MASKED   > REUSE_R_MASKED_T;
typedef ap_int <REUSE_DW_SOFTMAX    > REUSE_SOFTMAX_T;
typedef ap_int <REUSE_DW_A_TRUNC    > REUSE_A_TRUNC_T;
typedef ap_int <REUSE_DW_V_TRUNC    > REUSE_V_TRUNC_T;
typedef ap_int <REUSE_DW_MLP1_TRUNC > REUSE_MLP1_TRUNC_T;
typedef ap_int <REUSE_DW_MLP2_TRUNC > REUSE_MLP2_TRUNC_T;
typedef ap_int <REUSE_DW_CLS_TRUNC  > REUSE_CLS_TRUNC_T;
typedef ap_int <REUSE_DW_CLS_INDEX  > REUSE_CLS_INDEX_T;
typedef ap_int <REUSE_DW_DEMUX_BIAS > REUSE_DEMUX_BIAS_T;

typedef AQ_T REUSE_AQ_T;
typedef AS_T REUSE_AS_T;
typedef WQ_T REUSE_WQ_T;
typedef WS_T REUSE_WS_T;

typedef hls::vector<REUSE_AQ_T, REUSE_GEMM_LANES> REUSE_AQ_TILE_T;
typedef hls::vector<REUSE_AS_T, REUSE_GEMM_TP   > REUSE_AS_TILE_T;
typedef hls::vector<REUSE_X_T,  REUSE_GEMM_LANES> REUSE_X_TILE_T;

template<class data_t, int LANES>
using reuse_vec_t = hls::vector<data_t, LANES>;

template<class data_t, int LANES>
using reuse_stream_t = hls::stream<hls::vector<data_t, LANES> >;

// ============================================================================
// Explicit bit-width conversion helpers
// ============================================================================

template<class dst_t, class src_t>
dst_t reuse_extend(src_t value) {
    #pragma HLS inline
    static_assert(dst_t::width >= src_t::width, "reuse_extend requires dst width >= src width");
    return (dst_t)value;
}

template<class dst_t, class src_t>
dst_t reuse_truncate(src_t value) {
    #pragma HLS inline
    static_assert(dst_t::width <= src_t::width, "reuse_truncate requires dst width <= src width");
    dst_t out;
    out.range(dst_t::width - 1, 0) = value.range(dst_t::width - 1, 0);
    return out;
}

template<class dst_t, class src_t>
dst_t reuse_cast_width(src_t value) {
    #pragma HLS inline
    return (dst_t)value;
}

template<class dst_t, class src_t, int LANES>
hls::vector<dst_t, LANES> reuse_cast_vec(hls::vector<src_t, LANES> in_vec) {
    #pragma HLS inline
    hls::vector<dst_t, LANES> out_vec;
    for (int lane = 0; lane < LANES; ++lane) {
        #pragma HLS unroll
        out_vec[lane] = reuse_cast_width<dst_t, src_t>(in_vec[lane]);
    }
    return out_vec;
}

template<class dst_t, class src_t, int LANES>
hls::vector<dst_t, LANES> reuse_extend_vec(hls::vector<src_t, LANES> in_vec) {
    #pragma HLS inline
    hls::vector<dst_t, LANES> out_vec;
    for (int lane = 0; lane < LANES; ++lane) {
        #pragma HLS unroll
        out_vec[lane] = reuse_extend<dst_t, src_t>(in_vec[lane]);
    }
    return out_vec;
}

template<class dst_t, class src_t, int LANES>
hls::vector<dst_t, LANES> reuse_truncate_vec(hls::vector<src_t, LANES> in_vec) {
    #pragma HLS inline
    hls::vector<dst_t, LANES> out_vec;
    for (int lane = 0; lane < LANES; ++lane) {
        #pragma HLS unroll
        out_vec[lane] = reuse_truncate<dst_t, src_t>(in_vec[lane]);
    }
    return out_vec;
}

// ============================================================================
// Quantizer wrapper with shared q/scale output types
// ============================================================================

template<
    class if_t,
    int L,
    int T,
    int TP,
    int C,
    int CP,
    int G
> class REUSE_QUANTIZER {
public:
    QUANTIZER<if_t, REUSE_AQ_T, REUSE_AS_T, L, T, TP, C, CP, G> quantizer_inst;

    void do_quant(
        hls::stream<hls::vector<if_t,       TP*CP     > >& i_stream,
        hls::stream<hls::vector<REUSE_AQ_T, TP*CP     > >& q_stream,
        hls::stream<hls::vector<REUSE_AS_T, TP*(CP/G) > >& s_stream
    ) {
        #pragma HLS inline off
        quantizer_inst.do_quant(i_stream, q_stream, s_stream);
    }
};

#endif // __INT_REUSE_COMMON_H__
