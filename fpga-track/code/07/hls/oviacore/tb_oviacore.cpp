/**
 * OViA Core Testbench
 *
 * Tests:
 *   1. 4×4×4 INT8 matmul (basic correctness)
 *   2. 16×16×16 INT8 matmul (single tile)
 *   3. 32×32×32 INT8 matmul (multi-tile)
 *   4. Edge cases: zero matrix, dimension bounds
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "oviacore.h"

// ============================================================
// Software reference matmul
// ============================================================
void ref_matmul_8(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    memset(C, 0, M * N * sizeof(int32_t));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================
// Compare results
// ============================================================
bool compare_results(
    const int32_t* c_ref,
    const int32_t* c_hw,
    int M, int N,
    const char* test_name
) {
    int errors = 0;
    int max_errors_to_show = 10;
    int32_t max_diff = 0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t diff = abs(c_hw[i * N + j] - c_ref[i * N + j]);
            if (diff > max_diff) max_diff = diff;
            if (diff != 0 && errors < max_errors_to_show) {
                printf("  MISMATCH [%d][%d]: ref=%d hw=%d diff=%d\n",
                       i, j, (int)c_ref[i * N + j], (int)c_hw[i * N + j], (int)diff);
                errors++;
            }
        }
    }

    bool pass = (errors == 0);
    printf("%s: %s (max_diff=%d, errors=%d)\n",
           test_name, pass ? "PASS" : "FAIL", (int)max_diff, errors);
    return pass;
}

// ============================================================
// Run test with given dimensions
// ============================================================
bool run_test(int M, int N, int K, const char* name) {
    const int OCM_SIZE_WORDS = 65536;  // 256KB / 4 bytes
    static dma_word_t ocm_sim[OCM_SIZE_WORDS];

    // Clear OCM simulation memory
    memset(ocm_sim, 0, OCM_SIZE_WORDS * sizeof(dma_word_t));

    // Allocate reference arrays
    int8_t* A = (int8_t*)malloc(M * K * sizeof(int8_t));
    int8_t* B = (int8_t*)malloc(K * N * sizeof(int8_t));
    int32_t* C_ref = (int32_t*)malloc(M * N * sizeof(int32_t));

    if (!A || !B || !C_ref) {
        printf("  Memory allocation failed\n");
        free(A); free(B); free(C_ref);
        return false;
    }

    // Generate random data
    srand(42);
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)(rand() % 256 - 128);

    // Write A and B to OCM simulation memory
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            int idx = (OCM_OFFSET_A + i * K + k) / sizeof(dma_word_t);
            ocm_sim[idx] = A[i * K + k];
        }
    }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            int idx = (OCM_OFFSET_B + k * N + j) / sizeof(dma_word_t);
            ocm_sim[idx] = B[k * N + j];
        }
    }

    // Compute reference
    ref_matmul_8(A, B, C_ref, M, N, K);

    // == HLS DUT call (standard IP protocol: start/done handled by return) ==
    ovia_core(ocm_sim, M, N, K);

    // Read C from OCM simulation memory
    int32_t* C_hw = (int32_t*)malloc(M * N * sizeof(int32_t));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int base = (OCM_OFFSET_C + i * N + j) / sizeof(dma_word_t);
            C_hw[i * N + j] =
                (int32_t)(uint32_t)(
                    (uint8_t)ocm_sim[base] |
                    ((uint8_t)ocm_sim[base+1] << 8) |
                    ((uint8_t)ocm_sim[base+2] << 16) |
                    ((uint8_t)ocm_sim[base+3] << 24)
                );
        }
    }

    // Compare
    bool pass = compare_results(C_ref, C_hw, M, N, name);

    free(A); free(B); free(C_ref); free(C_hw);
    return pass;
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("=== OViA Accelerator Test Suite ===\n\n");

    int passed = 0;
    int total  = 6;

    // Test 1: 4×4×4 (smallest)
    if (run_test(4, 4, 4, "Test 1: 4x4x4 matmul")) passed++;

    // Test 2: 16×16×16 (single tile)
    if (run_test(16, 16, 16, "Test 2: 16x16x16 matmul (single tile)")) passed++;

    // Test 3: 32×16×16 (non-square)
    if (run_test(32, 16, 16, "Test 3: 32x16x16 matmul")) passed++;

    // Test 4: 16×32×16 (non-square)
    if (run_test(16, 32, 16, "Test 4: 16x32x16 matmul")) passed++;

    // Test 5: 32×32×32 (multi-tile)
    if (run_test(32, 32, 32, "Test 5: 32x32x32 matmul (multi-tile)")) passed++;

    // Test 6: Identity test (1×1×1)
    if (run_test(1, 1, 1, "Test 6: 1x1x1 matmul (identity)")) passed++;

    printf("\n=== Results: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
