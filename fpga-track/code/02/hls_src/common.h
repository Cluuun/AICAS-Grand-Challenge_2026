#ifndef __INT_COMMON_H__
#define __INT_COMMON_H__

// ============================================================================
// SmolVLM2-500M-Video-Instruct HLS Design Common Header
// ============================================================================
// 基于 的设计，针对 SmolVLM2 的 Text Decoder (LLaMA3) 进行适配
//
// 主要修改：
//   1. hidden_size: 896 -> 960
//   2. num_heads: 14 -> 15
//   3. num_kv_heads: 2 -> 5 (GQA ratio: 3)
//   4. intermediate_size: 4864 -> 2560
//   5. num_layers: 24 -> 32
//   6. vocab_size: 151936 -> 49280
// ============================================================================

// standard library
#include <cassert>
#include <ctime>
#include <iostream>
#include <cstdint>
#include <random>
#include <fstream>
#include <numeric>

// hls library
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_vector.h>

// user defined library
#include "adapter.h"
#include "utils.h"

using namespace std;

constexpr int log2ce(int n){
    // 计算 n 的以 2 为底的对数（向上取整），用于位宽/组数计算
    return (n <= 1) ? 0 : 1 + log2ce(n / 2);
}

// RAM 存储风格选择（综合时映射到不同硬件资源）
constexpr int BRAM_STYLE = 0;   // Block RAM 风格
constexpr int URAM_STYLE = 1;   // Ultra RAM 风格
constexpr int LRAM_STYLE = 2;   // 分布式/逻辑 RAM 风格


// ============================================================================
// Connector（Pixel Shuffle + Linear）— 与 modeling_smolvlm.SmolVLMConnector / configuration_smolvlm 一致
// ref/ViT/VIT_CONNECTOR_HYPERPARAMS.txt；权重与链式仿真数据根目录：data_connector/
// ============================================================================
constexpr int CONNECTOR_PARAMS[] = {
    #include "ref/ViT/VIT_CONNECTOR_HYPERPARAMS.txt"
};
constexpr int CONNECTOR_IN_TOKENS     = CONNECTOR_PARAMS[0];   // 1024
constexpr int CONNECTOR_IN_DIM        = CONNECTOR_PARAMS[1];   // 768
constexpr int CONNECTOR_OUT_TOKENS    = CONNECTOR_PARAMS[2];   // 64
constexpr int CONNECTOR_OUT_DIM       = CONNECTOR_PARAMS[3];   // 960
constexpr int CONNECTOR_SCALE         = CONNECTOR_PARAMS[4];   // 4 → 每组合并 4×4 patch
constexpr int CONNECTOR_DW_AQ         = CONNECTOR_PARAMS[5];   // 激活量化位宽
constexpr int CONNECTOR_DW_WQ         = CONNECTOR_PARAMS[6];   // 权重量化位宽

constexpr int CONNECTOR_SHUFFLE_DIM   = CONNECTOR_IN_DIM * CONNECTOR_SCALE * CONNECTOR_SCALE;  // 768 * 16 = 12288
constexpr int CONNECTOR_TRUNC     = 9;

constexpr int DW_CONNECTOR         =   64;
constexpr int DW_CONNECTOR_TRUNC   =  64;

typedef ap_int<DW_CONNECTOR       > CONECTOR_T;
typedef ap_int<DW_CONNECTOR_TRUNC > CONNECTOR_TRUNC_T;

// ============================================================================
// SmolVLM2 Text Decoder 超参数（从外部文件加载）
// ============================================================================
constexpr int DECODER_HYPERPARAMS[] = {
    #include "ref/LLM/DECODER_HYPERPARAMS.txt"
};

// 模型维度参数
constexpr int LLAMA_C       = DECODER_HYPERPARAMS[0];   // hidden_size = 960
constexpr int LLAMA_GA      = DECODER_HYPERPARAMS[1];   // 注意力分组数 G
constexpr int LLAMA_GW      = DECODER_HYPERPARAMS[2];
constexpr int LLAMA_GR      = DECODER_HYPERPARAMS[3];
constexpr int LLAMA_H       = DECODER_HYPERPARAMS[4];   // num_attention_heads = 15
constexpr int LLAMA_KVH     = DECODER_HYPERPARAMS[5];   // num_kv_heads = 5 (GQA)
constexpr int LLAMA_HC      = DECODER_HYPERPARAMS[6];   // head_dim = 64
constexpr int LLAMA_CM      = DECODER_HYPERPARAMS[7];   // intermediate_size = 2560
constexpr int LLAMA_VOCAB   = DECODER_HYPERPARAMS[8];   // vocab_size = 49280
constexpr int DW_AQ         = DECODER_HYPERPARAMS[9];   // 激活量化位宽 = 8
constexpr int DW_WQ         = DECODER_HYPERPARAMS[10];  // 权重量化位宽 = 5

// 层数和序列长度
constexpr int LLAMA_L       = 32;   // SmolVLM2-500M 有 32 层 Decoder
constexpr int LLAMA_S       = 1024;  // 最大支持序列长度（增加以支持多模态上下文）

// static checks：编译期断言
static_assert(LLAMA_GA == LLAMA_GW && LLAMA_GW == LLAMA_GR, "GA, GW, GR must be equal");
constexpr int LLAMA_G = LLAMA_GA;   // 统一的注意力分组数

static_assert(LLAMA_HC * LLAMA_H == LLAMA_C, "HC * H must be equal to C");

// ============================================================================
// MHA（Multi-Head Attention）各阶段数据截断位宽
// ============================================================================
constexpr int MHA_TRUNCS[] = {
    #include "ref/LLM/MHA_TRUNCS.txt"
};
constexpr int MHA_TRUNC_QK  = MHA_TRUNCS[0];   // QK  GEN后的截断位宽
constexpr int MHA_TRUNC_V   = MHA_TRUNCS[1];   // V   GEN后的截断位宽
constexpr int MHA_TRUNC_ROT = MHA_TRUNCS[2];   // RoPE 旋转后的截断位宽
constexpr int MHA_TRUNC_R   = MHA_TRUNCS[3];   // attention scores 的截断位宽
constexpr int MHA_TRUNC_A   = MHA_TRUNCS[4];   // softmax 后注意力权重的截断位宽
constexpr int MHA_TRUNC_O   = MHA_TRUNCS[5];   // 注意力输出 O 的截断位宽

// ============================================================================
// MLP（Feed-Forward Network）各阶段数据截断位宽
// ============================================================================
constexpr int MLP_TRUNCS[] = {
    #include "ref/LLM/MLP_TRUNCS.txt"
};
constexpr int MLP_TRUNC_UG  = MLP_TRUNCS[0];   // gate/up 线性输出截断位宽
constexpr int MLP_TRUNC_MUL = MLP_TRUNCS[1];   // gate * up 相乘后的截断位宽
constexpr int MLP_TRUNC_D   = MLP_TRUNCS[2];   // down 投影输出截断位宽

// 分类头输出截断位宽
constexpr int CLS_TRUNC     = 9;

constexpr int LLAMA_TRUNC_BASE = const_min(
    MHA_TRUNC_QK,
    MHA_TRUNC_V,
    MHA_TRUNC_O,
    MLP_TRUNC_UG,
    MLP_TRUNC_D,
    CLS_TRUNC
);



// 用于 attention mask 的"负无穷"数值 (匹配软件 int_llm_decoder.py)
constexpr int MASK_NEG_INF = -(1 << 30);

// ============================================================================
// 数据类型位宽定义
// ============================================================================
constexpr int DW_X              = 25;   // 主隐状态/激活 X 的位宽25
constexpr int DW_X_POW2SUM      = 50;   // LayerNorm 中平方和累加位宽48  +2
constexpr int DW_X_RSQRT        = 18;   // LayerNorm 中 1/sqrt(x) 的位宽；样本统计最大到 86528，17-bit signed 会回绕
constexpr int DW_LNW            = 20;   // LayerNorm 权重的位宽17  不够
constexpr int DW_XLN            = 22;   // LayerNorm 输出的位宽22

constexpr int DW_QKV            = 35;   // Q/K/V 线性层输出的位宽
constexpr int DW_QKV_TRUNC      = 23;   // Q/K/V 截断后的位宽


constexpr int DW_POS            = 12;   // 位置编码索引位宽
constexpr int DW_FREQS          = 24;   // RoPE 频率 theta 的位宽
constexpr int DW_COS_SIN        = 14;   // cos/sin 查表结果的位宽
constexpr int DW_ROT            = DW_QKV_TRUNC;   // 旋转后 Q/K 的位宽

constexpr int DW_R              = 43;   // Q·K^T 得分 R 的位宽43
constexpr int DW_R_TRUNC        = 31;   // R 截断后的位宽31
constexpr int DW_R_MASKED       = 32;   // 加 mask 后的 R 位宽 大一位，加了mask

constexpr int DW_EXP            = 9;   // exp(R) 近似结果的位宽 9
constexpr int DW_EXP_SUM        = 19;   // softmax 分母的位宽  19
constexpr int DW_RECIP          = 18;   // 倒数 1/sum 的位宽   18
constexpr int DW_SOFTMAX        = 21;   // softmax 输出的位宽  21

constexpr int DW_A              = 40;   // 注意力权重 A 的位宽
constexpr int DW_A_TRUNC        = 23;   // A 截断后的位宽
constexpr int DW_O              = 39;   // 注意力输出 O 的位宽
constexpr int DW_O_TRUNC        = 21;   // O 截断后的位宽
constexpr int DW_XUG            = 33;   // MLP gate/up 的位宽
constexpr int DW_XUG_TRUNC      = 21;   // gate/up 截断后的位宽
constexpr int DW_SILU           = 21;   // SiLU 输出的位宽
constexpr int DW_XM_TRUNC       = 27;   // gate*up 截断后的位宽
constexpr int DW_XD             = 36;   // MLP down 输出的位宽
constexpr int DW_XD_TRUNC       = 25;   // down 截断后的位宽
constexpr int DW_CLS            = 32;   // 分类头 logits 的位宽
constexpr int DW_CLS_TRUNC      = 23;   // 分类头截断后的位宽
constexpr int DW_CLS_INDEX      = 32;   // DEMUX 内 argmax 后 token index，按 int32 写回 PS

constexpr int DW_WS             = 4;    // 权重量化 scale 的位宽
constexpr int DW_AS             = 4;    // 激活量化 scale 的位宽

// 派生数据类型位宽
constexpr int DW_GEMM           = DW_AQ + DW_WQ + log2ce(LLAMA_G);
constexpr int DW_GEMM_ACC       = const_max(DW_QKV, DW_O, DW_XUG, DW_XD);
constexpr int DW_GEMM_TRUNC     = DW_GEMM_ACC - LLAMA_TRUNC_BASE;


// ============================================================================
// HLS 定点类型定义
// ============================================================================
typedef ap_int<DW_X             > X_T;
typedef ap_int<DW_X_POW2SUM     > X_POW2SUM_T;
typedef ap_int<DW_X_RSQRT       > X_RSQRT_T;
typedef ap_int<DW_LNW           > LNW_T;
typedef ap_int<DW_XLN           > XLN_T;
typedef ap_int<DW_QKV           > QKV_T;
typedef ap_int<DW_QKV_TRUNC     > QKV_TRUNC_T;

typedef ap_int<DW_FREQS         > FREQS_T;
typedef ap_int<DW_COS_SIN       > COS_SIN_T;
typedef ap_int<DW_ROT           > ROT_T;
typedef ap_int<DW_R             > R_T;
typedef ap_int<DW_R_TRUNC       > R_TRUNC_T;
typedef ap_int<DW_R_MASKED      > R_MASKED_T;
typedef ap_int<DW_EXP           > EXP_T;
typedef ap_int<DW_EXP_SUM       > EXP_SUM_T;
typedef ap_int<DW_RECIP         > RECIP_T;
typedef ap_int<DW_SOFTMAX       > SOFTMAX_T;
typedef ap_int<DW_A             > A_T;
typedef ap_int<DW_A_TRUNC       > A_TRUNC_T;
typedef ap_int<DW_O             > O_T;
typedef ap_int<DW_O_TRUNC       > O_TRUNC_T;
typedef ap_int<DW_XUG           > XUG_T;
typedef ap_int<DW_XUG_TRUNC     > XUG_TRUNC_T;
typedef ap_int<DW_SILU          > SILU_T;
typedef ap_int<DW_XM_TRUNC      > XM_T;
typedef ap_int<DW_XD            > XD_T;
typedef ap_int<DW_XD_TRUNC      > XD_TRUNC_T;
typedef ap_int<DW_CLS           > CLS_T;
typedef ap_int<DW_CLS_TRUNC     > CLS_TRUNC_T;
typedef ap_int<DW_GEMM          > GEMM_T;
typedef ap_int<DW_GEMM_TRUNC    > GEMM_TRUNC_T;
typedef ap_int<DW_CLS_INDEX     > CLS_INDEX_T;


typedef ap_int <DW_AQ           > AQ_T;
typedef ap_uint<DW_AS           > AS_T;
typedef ap_int <DW_WQ           > WQ_T;
typedef ap_uint<DW_WS           > WS_T;

// ============================================================================
// SmolVLM2 Vision Encoder（ViT / SigLIP）— 与 LLM 侧 DECODER_HYPERPARAMS / MHA_TRUNCS / MLP_TRUNCS 同一组织方式
// ViT IntLayerNorm 位宽/标量/LUT：layernorm.h
// ============================================================================
constexpr int VIT_HYPERPARAMS[] = {
    #include "ref/ViT/VIT_VISION_HYPERPARAMS.txt"
};
constexpr int VIT_C       = VIT_HYPERPARAMS[0];   // hidden_size
constexpr int VIT_G       = VIT_HYPERPARAMS[1];   // 注意力 GEMM 分组数 G
constexpr int VIT_H       = VIT_HYPERPARAMS[2];   // num_attention_heads
constexpr int VIT_HC      = VIT_HYPERPARAMS[3];   // head_dim
constexpr int VIT_MLP_DIM = VIT_HYPERPARAMS[4];   // intermediate_size（ViT MLP 隐层）
static_assert(VIT_HC * VIT_H == VIT_C, "VIT_HC * VIT_H must equal VIT_C");

constexpr int VIT_L       = 12;
constexpr int VIT_S       = 1024;

// Vision MHA 各阶段截断位宽（与 ref/MHA_TRUNCS.txt 语义对应；ViT 无单独 RoPE 截断项，共 5 项）
constexpr int VIT_MHA_TRUNCS[] = {
    #include "ref/ViT/VIT_VISION_MHA_TRUNCS.txt"
};
constexpr int VIT_MHA_TRUNC_QK = VIT_MHA_TRUNCS[0];   // Q·K^T 后截断
constexpr int VIT_MHA_TRUNC_V  = VIT_MHA_TRUNCS[1];   // V 路径截断
constexpr int VIT_MHA_TRUNC_R  = VIT_MHA_TRUNCS[2];   // attention scores 截断
constexpr int VIT_MHA_TRUNC_A  = VIT_MHA_TRUNCS[3];   // softmax 后权重截断
constexpr int VIT_MHA_TRUNC_O  = VIT_MHA_TRUNCS[4];   // 注意力输出 O 截断

// Vision MLP：仅 fc1 / fc2 矩阵乘后有 Trunc；GELU（IntGELU）无单独 Trunc（LUT + 左移），与 int_vit_encoder 一致
constexpr int VIT_MLP_TRUNCS[] = {
    #include "ref/ViT/VIT_VISION_MLP_TRUNCS.txt"
};
constexpr int VIT_MLP_TRUNC_FC1  = VIT_MLP_TRUNCS[0];
constexpr int VIT_MLP_TRUNC_FC2  = VIT_MLP_TRUNCS[1];

constexpr int VIT_TRUNC_BASE = const_min(
    VIT_MHA_TRUNC_QK,
    VIT_MHA_TRUNC_V,
    VIT_MHA_TRUNC_R,
    VIT_MHA_TRUNC_A,
    VIT_MHA_TRUNC_O,
    VIT_MLP_TRUNC_FC1,
    VIT_MLP_TRUNC_FC2
    //CONNECTOR_TRUNC
);


//DW_AS、DW_WS、AQ、WQ都共用
constexpr int VIT_DW_WS             = DW_WS;
constexpr int VIT_DW_AS             = DW_AS;
constexpr int VIT_DW_AQ             = DW_AQ;
constexpr int VIT_DW_WQ             = DW_WQ;

constexpr int VIT_DW_X              = 23;   // 残差流 (layer_in / residual / out)
constexpr int VIT_DW_X_POW2SUM      = 40;   // LayerNorm 方差
constexpr int VIT_DW_X_RSQRT        = 24;   // LayerNorm rsqrt LUT 查表后再 <<offsets_diff；段0 最大约 156<<9，超出原 16 位
constexpr int VIT_DW_LNW            = 16;   // LayerNorm weight (gamma)
constexpr int VIT_DW_XLN            = 24;   // LayerNorm 输出
constexpr int VIT_DW_QKV            = 34;   // QKV matmul
constexpr int VIT_DW_QKV_TRUNC      = 20;   // QKV trunc 后
constexpr int VIT_DW_BIAS           = 20;   // QKV + bias
constexpr int VIT_DW_R              = 40;   // QK^T matmul
constexpr int VIT_DW_R_TRUNC        = 31;   // QK^T trunc 后
constexpr int VIT_DW_R_MASKED       = 31;   // QK^T + mask (双向, 同 R)
constexpr int VIT_DW_EXP            = 9;    // exp LUT 输出
constexpr int VIT_DW_EXP_SUM        = 19;   // exp 求和
constexpr int VIT_DW_RECIP          = 18;   // recip LUT 输出
constexpr int VIT_DW_SOFTMAX        = 23;   // softmax 最终输出
constexpr int VIT_DW_A              = 40;   // softmax @ V
constexpr int VIT_DW_A_TRUNC        = 23;   // RV trunc 后
constexpr int VIT_DW_O              = 36;   // output projection + bias
constexpr int VIT_DW_O_TRUNC        = 17;   // output trunc 后
constexpr int VIT_DW_FC1            = 38;   // fc1 matmul
constexpr int VIT_DW_FC1_TRUNC      = 28;   // fc1 trunc 后 
constexpr int VIT_DW_FC1_GELU       = 22;   // GELU 输出，就是VIT_DW_FC2，VIT里面没有XM计算
constexpr int VIT_DW_FC2_TRUNC      = 23;   // fc2 trunc 后

constexpr int VIT_DW_GEMM           = DW_GEMM;
constexpr int VIT_DW_GEMM_ACC = const_max(VIT_DW_QKV, VIT_DW_O, VIT_DW_FC1, VIT_DW_FC1_GELU);
constexpr int VIT_DW_GEMM_TRUNC = VIT_DW_GEMM_ACC - VIT_TRUNC_BASE;

// ============================================================================
// ViT HLS 定点类型定义
// ============================================================================
typedef ap_int<VIT_DW_X             > VIT_X_T;
typedef ap_int<VIT_DW_X_POW2SUM     > VIT_X_POW2SUM_T;
typedef ap_int<VIT_DW_X_RSQRT       > VIT_X_RSQRT_T;
typedef ap_int<VIT_DW_LNW           > VIT_LNW_T;
typedef ap_int<VIT_DW_XLN           > VIT_XLN_T;
typedef ap_int<VIT_DW_QKV           > VIT_QKV_T;
typedef ap_int<VIT_DW_QKV_TRUNC     > VIT_QKV_TRUNC_T;
typedef ap_int<VIT_DW_BIAS          > VIT_BIAS_T;
typedef ap_int<VIT_DW_R             > VIT_R_T;
typedef ap_int<VIT_DW_R_TRUNC       > VIT_R_TRUNC_T;
typedef ap_int<VIT_DW_R_MASKED      > VIT_R_MASKED_T;
typedef ap_int<VIT_DW_EXP           > VIT_EXP_T;
typedef ap_int<VIT_DW_EXP_SUM       > VIT_EXP_SUM_T;
typedef ap_int<VIT_DW_RECIP         > VIT_RECIP_T;
typedef ap_int<VIT_DW_SOFTMAX       > VIT_SOFTMAX_T;
typedef ap_int<VIT_DW_A             > VIT_A_T;
typedef ap_int<VIT_DW_A_TRUNC       > VIT_A_TRUNC_T;
typedef ap_int<VIT_DW_O             > VIT_O_T;
typedef ap_int<VIT_DW_O_TRUNC       > VIT_O_TRUNC_T;
typedef ap_int<VIT_DW_FC1           > VIT_FC1_T;
typedef ap_int<VIT_DW_FC1_TRUNC     > VIT_FC1_TRUNC_T;
typedef ap_int<VIT_DW_FC1_GELU      > VIT_FC1_GELU_T;
typedef ap_int<VIT_DW_FC2_TRUNC     > VIT_FC2_TRUNC_T;
typedef ap_int<VIT_DW_GEMM          > VIT_GEMM_T;
typedef ap_int<VIT_DW_GEMM_TRUNC    > VIT_GEMM_TRUNC_T;

typedef ap_uint<VIT_DW_AS           > VIT_AS_T;
typedef ap_uint<VIT_DW_WS           > VIT_WS_T;
typedef ap_int <VIT_DW_AQ           > VIT_AQ_T;
typedef ap_int <VIT_DW_WQ           > VIT_WQ_T;


// 仿真时权重与 condense 的路径（相对 csim 运行目录 instances_*/proj_*/work/solution/csim/build）
const string BINARIES_PATH="../../../../../../Weights/LLM/binaries/decoder_";
const string CONDENSE_PATH="../../../../../../Weights/LLM/condense/decoder_";

// 仿真时 Vision 路径（与 LLM 侧 BINARIES_PATH / CONDENSE_PATH 同级组织：binaries 放权重与主向量；condense 保留给链式仿真产生的中间/压缩张量）
const string VIT_BINARIES_PATH  = "../../../../../../Weights/ViT/binaries/vision_";
const string VIT_CONDENSE_PATH  = "../../../../../../Weights/ViT/condense/vision_";

// Connector 仿真数据路径（与 data_vit 同级）
const string CONNECTOR_BINARIES_PATH  = "../../../../../../Weights/Connector/binaries";
const string CONNECTOR_CONDENSE_PATH  = "../../../../../../Weights/Connector/condense";

#endif // __INT_COMMON_H__
