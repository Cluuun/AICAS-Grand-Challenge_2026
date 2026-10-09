#pragma once
/**
 * AICAS 2026 - 融合注意力引擎头文件
 */

#include <ap_fixed.h>
#include <hls_stream.h>

// 配置参数
const int MAX_SEQ_LEN = 2048;
const int HEAD_DIM = 64;
const int NUM_HEADS = 16;
const int NUM_KV_HEADS = 4;
const int HEADS_PER_GROUP = NUM_HEADS / NUM_KV_HEADS;

// 数据类型（避免与标准库冲突，使用自定义名）
typedef ap_fixed<16,6,AP_RND,AP_SAT> fp16_t;
typedef ap_fixed<32,12,AP_RND,AP_SAT> fp32_t;
typedef ap_int<8>  hls_int8_t;
typedef ap_int<32> hls_int32_t;

// 顶层函数声明
void attention_engine(
    fp16_t* Q,
    fp16_t* K,
    fp16_t* V,
    fp16_t* output,
    int seq_len,
    int kv_len
);
