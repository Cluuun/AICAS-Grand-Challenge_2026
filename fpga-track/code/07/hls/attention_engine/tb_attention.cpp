/**
 * AICAS 2026 - 注意力引擎测试平台
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "attention_engine.h"

// 测试数据生成
void generate_test_data(
    fp16_t* Q, fp16_t* K, fp16_t* V,
    int seq_len, int kv_len
) {
    // 随机初始化
    for (int i = 0; i < seq_len * NUM_HEADS * HEAD_DIM; i++) {
        Q[i] = (fp16_t)(rand() / (float)RAND_MAX - 0.5);
    }
    for (int i = 0; i < kv_len * NUM_KV_HEADS * HEAD_DIM; i++) {
        K[i] = (fp16_t)(rand() / (float)RAND_MAX - 0.5);
        V[i] = (fp16_t)(rand() / (float)RAND_MAX - 0.5);
    }
}

// 参考实现 (软模型)
void reference_attention(
    fp16_t* Q, fp16_t* K, fp16_t* V,
    fp16_t* ref_output,
    int seq_len, int kv_len
) {
    for (int h = 0; h < NUM_HEADS; h++) {
        int kv_group = h / HEADS_PER_GROUP;

        for (int q_pos = 0; q_pos < seq_len; q_pos++) {
            // QK^T
            fp16_t scores[MAX_SEQ_LEN];
            fp16_t max_score = -1e6;

            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                fp16_t acc = 0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    acc += Q[q_pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d]
                         * K[kv_pos * NUM_KV_HEADS * HEAD_DIM + kv_group * HEAD_DIM + d];
                }
                scores[kv_pos] = acc / sqrt(HEAD_DIM);
                if (scores[kv_pos] > max_score) {
                    max_score = scores[kv_pos];
                }
            }

            // Softmax
            fp16_t sum_exp = 0;
            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                scores[kv_pos] = exp(scores[kv_pos] - max_score);
                sum_exp += scores[kv_pos];
            }
            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                scores[kv_pos] /= sum_exp;
            }

            // Score @ V
            for (int d = 0; d < HEAD_DIM; d++) {
                fp16_t acc = 0;
                for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                    acc += scores[kv_pos]
                         * V[kv_pos * NUM_KV_HEADS * HEAD_DIM + kv_group * HEAD_DIM + d];
                }
                ref_output[q_pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d] = acc;
            }
        }
    }
}

// 比较结果
bool compare_results(
    fp16_t* hw_output, fp16_t* ref_output,
    int size, float tolerance
) {
    float max_error = 0;
    int error_count = 0;

    for (int i = 0; i < size; i++) {
        float error = fabs((float)hw_output[i] - (float)ref_output[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > tolerance) {
            error_count++;
        }
    }

    printf("最大误差: %f\n", max_error);
    printf("超差样本数: %d / %d (%.2f%%)\n",
           error_count, size, 100.0 * error_count / size);

    return error_count == 0;
}

// 主测试函数
int main() {
    printf("=== 注意力引擎测试 ===\n");

    // 测试配置
    int test_seq_len = 128;
    int test_kv_len = 128;
    float tolerance = 0.01;  // 1%容忍度

    printf("测试参数: seq_len=%d, kv_len=%d\n", test_seq_len, test_kv_len);

    // 分配内存
    fp16_t* Q = (fp16_t*)malloc(test_seq_len * NUM_HEADS * HEAD_DIM * sizeof(fp16_t));
    fp16_t* K = (fp16_t*)malloc(test_kv_len * NUM_KV_HEADS * HEAD_DIM * sizeof(fp16_t));
    fp16_t* V = (fp16_t*)malloc(test_kv_len * NUM_KV_HEADS * HEAD_DIM * sizeof(fp16_t));
    fp16_t* hw_output = (fp16_t*)malloc(test_seq_len * NUM_HEADS * HEAD_DIM * sizeof(fp16_t));
    fp16_t* ref_output = (fp16_t*)malloc(test_seq_len * NUM_HEADS * HEAD_DIM * sizeof(fp16_t));

    // 生成测试数据
    printf("生成测试数据...\n");
    generate_test_data(Q, K, V, test_seq_len, test_kv_len);

    // 软模型参考
    printf("运行软模型参考...\n");
    reference_attention(Q, K, V, ref_output, test_seq_len, test_kv_len);

    // HLS仿真
    printf("运行HLS仿真...\n");
    attention_engine(Q, K, V, hw_output, test_seq_len, test_kv_len);

    // 比较结果
    printf("比较结果...\n");
    bool pass = compare_results(hw_output, ref_output,
                               test_seq_len * NUM_HEADS * HEAD_DIM,
                               tolerance);

    if (pass) {
        printf("\n✓ 测试通过!\n");
    } else {
        printf("\n✗ 测试失败!\n");
    }

    // 释放内存
    free(Q);
    free(K);
    free(V);
    free(hw_output);
    free(ref_output);

    return pass ? 0 : 1;
}
