// testbench.cpp - C-simulation coverage for vlm_engine_v3.
#include "accelerator.hpp"
#include "../common/vlm_w8a8_abi_v10.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr uint32_t ARENA_WORDS = 4 * 1024 * 1024;
static constexpr uint32_t AXI_BYTES = 16;
static constexpr uint32_t SA_N = VLM_W8A8_V3_SA_N;
static constexpr uint32_t TILE_K = VLM_W8A8_V3_TILE_K;
static constexpr uint32_t GROUP_SIZE = VLM_W8A8_QK;
static constexpr uint32_t GROUPS_PER_KTILE = TILE_K / GROUP_SIZE;
static constexpr uint32_t WEIGHT_COLS_PER_PORT = SA_N / VLM_W8A8_WEIGHT_PORTS;
static constexpr uint32_t WEIGHT_WORDS_PER_PORT_KTILE = GROUPS_PER_KTILE * GROUP_SIZE;
static constexpr uint32_t SCALE_WORDS_PER_PORT_KTILE = (GROUPS_PER_KTILE * WEIGHT_COLS_PER_PORT + AXI_BYTES - 1) / AXI_BYTES;

static vlm_w8a8_axi_t weight0_arena[ARENA_WORDS];
static vlm_w8a8_axi_t weight1_arena[ARENA_WORDS];
static vlm_w8a8_axi_t act_arena[ARENA_WORDS];
static vlm_w8a8_axi_t out_arena[ARENA_WORDS];

static void write_byte(vlm_w8a8_axi_t *arena, uint64_t byte_addr, uint8_t val) {
    const uint64_t word_idx = byte_addr / AXI_BYTES;
    const uint32_t byte_off = byte_addr % AXI_BYTES;
    vlm_w8a8_axi_t w = arena[word_idx];
    w(byte_off * 8 + 7, byte_off * 8) = val;
    arena[word_idx] = w;
}

static uint8_t read_byte(const vlm_w8a8_axi_t *arena, uint64_t byte_addr) {
    const uint64_t word_idx = byte_addr / AXI_BYTES;
    const uint32_t byte_off = byte_addr % AXI_BYTES;
    return static_cast<uint8_t>(arena[word_idx](byte_off * 8 + 7, byte_off * 8));
}

static int32_t read_i32(const vlm_w8a8_axi_t *arena, uint64_t byte_addr) {
    uint32_t val = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        val |= static_cast<uint32_t>(read_byte(arena, byte_addr + i)) << (8 * i);
    }
    return static_cast<int32_t>(val);
}

static void reset_arenas() {
    std::memset(weight0_arena, 0, sizeof(weight0_arena));
    std::memset(weight1_arena, 0, sizeof(weight1_arena));
    std::memset(act_arena, 0, sizeof(act_arena));
    std::memset(out_arena, 0, sizeof(out_arena));
}

static void write_task(uint64_t task_offset, const linear_task_t &task) {
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&task);
    for (uint32_t i = 0; i < sizeof(task); ++i) {
        write_byte(act_arena, task_offset + i, bytes[i]);
    }
}

static void fill_activation(uint64_t act_offset, uint64_t scale_offset, uint32_t rows, uint32_t k_dim, uint32_t row_stride, int8_t *a) {
    const uint32_t scale_groups = (k_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t k = 0; k < k_dim; ++k) {
            const int8_t v = static_cast<int8_t>((static_cast<int>(m * 5 + k * 3) % 13) - 6);
            a[m * k_dim + k] = v;
            write_byte(act_arena, act_offset + static_cast<uint64_t>(m) * row_stride + k, static_cast<uint8_t>(v));
        }
        for (uint32_t g = 0; g < scale_groups; ++g) {
            write_byte(act_arena, scale_offset + static_cast<uint64_t>(m) * scale_groups + g, 0);
        }
    }
}

static void fill_weight(uint32_t output_seed, uint32_t k_dim, uint32_t n_dim, int8_t *w) {
    for (uint32_t k = 0; k < k_dim; ++k) {
        for (uint32_t n = 0; n < n_dim; ++n) {
            w[k * n_dim + n] = static_cast<int8_t>((static_cast<int>(output_seed + k * 7 + n * 5) % 11) - 5);
        }
    }
}

static void pack_weight_ports(uint64_t q0_offset, uint64_t q1_offset, uint64_t s0_offset, uint64_t s1_offset,
                              const int8_t *w, uint32_t k_dim, uint32_t n_dim) {
    const uint32_t k_tiles = (k_dim + TILE_K - 1) / TILE_K;
    const uint32_t n_tiles = (n_dim + SA_N - 1) / SA_N;
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            const uint64_t q0_kt = q0_offset + static_cast<uint64_t>(nt * k_tiles + kt) * WEIGHT_WORDS_PER_PORT_KTILE * AXI_BYTES;
            const uint64_t q1_kt = q1_offset + static_cast<uint64_t>(nt * k_tiles + kt) * WEIGHT_WORDS_PER_PORT_KTILE * AXI_BYTES;
            const uint64_t s0_kt = s0_offset + static_cast<uint64_t>(nt * k_tiles + kt) * SCALE_WORDS_PER_PORT_KTILE * AXI_BYTES;
            const uint64_t s1_kt = s1_offset + static_cast<uint64_t>(nt * k_tiles + kt) * SCALE_WORDS_PER_PORT_KTILE * AXI_BYTES;
            for (uint32_t g = 0; g < GROUPS_PER_KTILE; ++g) {
                for (uint32_t kk = 0; kk < GROUP_SIZE; ++kk) {
                    const uint32_t k_abs = kt * TILE_K + g * GROUP_SIZE + kk;
                    const uint64_t word_off = (g * GROUP_SIZE + kk) * AXI_BYTES;
                    for (uint32_t c = 0; c < WEIGHT_COLS_PER_PORT; ++c) {
                        const uint32_t n0 = nt * SA_N + c;
                        const uint32_t n1 = nt * SA_N + WEIGHT_COLS_PER_PORT + c;
                        const int8_t v0 = (k_abs < k_dim && n0 < n_dim) ? w[k_abs * n_dim + n0] : 0;
                        const int8_t v1 = (k_abs < k_dim && n1 < n_dim) ? w[k_abs * n_dim + n1] : 0;
                        write_byte(weight0_arena, q0_kt + word_off + c, static_cast<uint8_t>(v0));
                        write_byte(weight1_arena, q1_kt + word_off + c, static_cast<uint8_t>(v1));
                    }
                }
                for (uint32_t c = 0; c < WEIGHT_COLS_PER_PORT; ++c) {
                    write_byte(weight0_arena, s0_kt + g * WEIGHT_COLS_PER_PORT + c, 0);
                    write_byte(weight1_arena, s1_kt + g * WEIGHT_COLS_PER_PORT + c, 0);
                }
            }
        }
    }
}

static void golden_gemm(const int8_t *a, const int8_t *w, int32_t *c, uint32_t m_dim, uint32_t k_dim, uint32_t n_dim) {
    for (uint32_t m = 0; m < m_dim; ++m) {
        for (uint32_t n = 0; n < n_dim; ++n) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < k_dim; ++k) {
                acc += static_cast<int32_t>(a[m * k_dim + k]) * static_cast<int32_t>(w[k * n_dim + n]);
            }
            c[m * n_dim + n] = acc;
        }
    }
}

static int32_t golden_hard_silu(int32_t x) {
    int32_t x_plus_3 = x + 3;
    int32_t clamped = (x_plus_3 < 0) ? 0 : ((x_plus_3 > 6) ? 6 : x_plus_3);
    int64_t prod = static_cast<int64_t>(x) * static_cast<int64_t>(clamped);
    return static_cast<int32_t>((prod * 2796203LL) >> 24);
}

static int32_t golden_hard_gelu(int32_t x) {
    int32_t x_plus_1p5 = x + (3 << 7);
    int32_t three_q8 = 3 << 8;
    int32_t clamped = (x_plus_1p5 < 0) ? 0 : ((x_plus_1p5 > three_q8) ? three_q8 : x_plus_1p5);
    int64_t prod = static_cast<int64_t>(x) * static_cast<int64_t>(clamped);
    return static_cast<int32_t>((prod * 5592405LL) >> 24);
}

static void golden_quantize_32(const int32_t *values, int8_t *out_q, int8_t &scale_exp) {
    int32_t max_abs = 0;
    for (uint32_t i = 0; i < SA_N; ++i) {
        int32_t abs_val = (values[i] < 0) ? -values[i] : values[i];
        if (abs_val > max_abs) max_abs = abs_val;
    }
    uint8_t shift = 0;
    if (max_abs > 127) {
        uint32_t tmp = static_cast<uint32_t>(max_abs);
        while (tmp > 127) {
            tmp >>= 1;
            shift++;
        }
    }
    scale_exp = static_cast<int8_t>(shift);
    for (uint32_t i = 0; i < SA_N; ++i) {
        int32_t rounded = (shift > 0) ? ((values[i] + (1 << (shift - 1))) >> shift) : values[i];
        if (rounded > 127) rounded = 127;
        if (rounded < -128) rounded = -128;
        out_q[i] = static_cast<int8_t>(rounded);
    }
}

static int32_t golden_shift(int32_t x, int8_t exp) {
    if (exp >= 31) return 0;
    if (exp >= 0) return x << exp;
    int rshift = -static_cast<int>(exp);
    if (rshift >= 31) return 0;
    return x >> rshift;
}

static void golden_ffn(const int8_t *a, const int8_t *gate_w, const int8_t *up_w, const int8_t *down_w,
                       int32_t *out, uint32_t rows, uint32_t k_dim, uint32_t mid_dim,
                       uint32_t out_dim, uint32_t activation_type) {
    int8_t mid_q[2][128];
    int8_t mid_scale[2][4];
    std::memset(mid_q, 0, sizeof(mid_q));
    std::memset(mid_scale, 0, sizeof(mid_scale));
    std::memset(out, 0, sizeof(int32_t) * rows * out_dim);

    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t mt = 0; mt < (mid_dim + SA_N - 1) / SA_N; ++mt) {
            int32_t fused[SA_N];
            for (uint32_t c = 0; c < SA_N; ++c) {
                uint32_t mid_col = mt * SA_N + c;
                int32_t gate_acc = 0;
                int32_t up_acc = 0;
                if (mid_col < mid_dim) {
                    for (uint32_t k = 0; k < k_dim; ++k) {
                        gate_acc += static_cast<int32_t>(a[m * k_dim + k]) * static_cast<int32_t>(gate_w[k * mid_dim + mid_col]);
                        up_acc += static_cast<int32_t>(a[m * k_dim + k]) * static_cast<int32_t>(up_w[k * mid_dim + mid_col]);
                    }
                }
                int32_t activated = (activation_type == LINEAR_FFN_ACT_SILU) ? golden_hard_silu(gate_acc) : golden_hard_gelu(gate_acc);
                fused[c] = static_cast<int32_t>((static_cast<int64_t>(activated) * static_cast<int64_t>(up_acc)) >> 16);
            }
            golden_quantize_32(fused, &mid_q[m][mt * SA_N], mid_scale[m][mt]);
        }
    }

    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t n = 0; n < out_dim; ++n) {
            int32_t total = 0;
            for (uint32_t mt = 0; mt < (mid_dim + SA_N - 1) / SA_N; ++mt) {
                int32_t part = 0;
                for (uint32_t c = 0; c < SA_N; ++c) {
                    uint32_t mid_col = mt * SA_N + c;
                    if (mid_col < mid_dim) {
                        part += static_cast<int32_t>(mid_q[m][mid_col]) * static_cast<int32_t>(down_w[mid_col * out_dim + n]);
                    }
                }
                total += golden_shift(part, mid_scale[m][mt]);
            }
            out[m * out_dim + n] = total;
        }
    }
}

static int compare_output(const char *name, uint64_t out_offset, const int32_t *gold, uint32_t rows, uint32_t cols) {
    const uint32_t row_stride = ((cols * 4 + AXI_BYTES - 1) / AXI_BYTES) * AXI_BYTES;
    int errors = 0;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t n = 0; n < cols; ++n) {
            const int32_t hw = read_i32(out_arena, out_offset + static_cast<uint64_t>(m) * row_stride + n * 4);
            const int32_t ref = gold[m * cols + n];
            if (hw != ref) {
                if (errors < 16) {
                    std::printf("%s mismatch [%u,%u]: hw=%d ref=%d\n", name, m, n, hw, ref);
                }
                ++errors;
            }
        }
    }
    if (errors == 0) {
        std::printf("PASS: %s\n", name);
    } else {
        std::printf("FAIL: %s errors=%d\n", name, errors);
    }
    return errors;
}

static linear_task_t make_task(uint32_t task_type, uint32_t engine, uint32_t rows, uint32_t k_dim,
                               uint64_t act_offset, uint64_t act_scale_offset) {
    linear_task_t task;
    std::memset(&task, 0, sizeof(task));
    task.magic = VLM_W8A8_TASK_MAGIC;
    task.version = VLM_W8A8_ABI_VERSION;
    task.task_type = task_type;
    task.engine = engine;
    task.rows = rows;
    task.input_cols = k_dim;
    task.act_row_stride = ((k_dim + AXI_BYTES - 1) / AXI_BYTES) * AXI_BYTES;
    task.act_q_offset_bytes = act_offset;
    task.act_scale_offset_bytes = act_scale_offset;
    return task;
}

static int run_dense_like_case(const char *name, uint32_t task_type, uint32_t engine, uint32_t rows, uint32_t outputs) {
    reset_arenas();
    const uint32_t k_dim = 128;
    const uint32_t n_dim = 32;
    const uint64_t task_offset = 0;
    const uint64_t act_offset = 4096;
    const uint64_t act_scale_offset = 65536;
    const uint64_t q0_base[3] = { 131072, 196608, 262144 };
    const uint64_t q1_base[3] = { 327680, 393216, 458752 };
    const uint64_t s0_base[3] = { 524288, 540672, 557056 };
    const uint64_t s1_base[3] = { 573440, 589824, 606208 };
    const uint64_t out_base[3] = { 1048576, 1114112, 1179648 };

    int8_t a[2 * 128];
    int8_t w[3][128 * 32];
    int32_t gold[3][2 * 32];
    fill_activation(act_offset, act_scale_offset, rows, k_dim, ((k_dim + AXI_BYTES - 1) / AXI_BYTES) * AXI_BYTES, a);

    linear_task_t task = make_task(task_type, engine, rows, k_dim, act_offset, act_scale_offset);
    task.output_count = outputs;
    for (uint32_t o = 0; o < outputs; ++o) {
        task.out_cols[o] = n_dim;
        task.weight_q_offset_bytes[o] = q0_base[o];
        task.weight_q_offset_bytes_port1[o] = q1_base[o];
        task.weight_scale_offset_bytes[o] = s0_base[o];
        task.weight_scale_offset_bytes_port1[o] = s1_base[o];
        task.dst_offset_bytes[o] = out_base[o];
        fill_weight(o * 17 + task_type, k_dim, n_dim, w[o]);
        pack_weight_ports(q0_base[o], q1_base[o], s0_base[o], s1_base[o], w[o], k_dim, n_dim);
        golden_gemm(a, w[o], gold[o], rows, k_dim, n_dim);
    }
    write_task(task_offset, task);

    uint32_t status = 0;
    vlm_engine_v3(weight0_arena, weight1_arena, act_arena, out_arena,
                  task_offset, 2 * 1024 * 1024, 1, status);
    if (status != VLM_W8A8_STATUS_DONE) {
        std::printf("FAIL: %s status=%u\n", name, status);
        return 1;
    }

    int errors = 0;
    for (uint32_t o = 0; o < outputs; ++o) {
        char label[96];
        std::snprintf(label, sizeof(label), "%s output%u", name, o);
        errors += compare_output(label, out_base[o], gold[o], rows, n_dim);
    }
    return errors;
}

static int run_ffn_case(const char *name, uint32_t task_type, uint32_t engine, uint32_t rows, uint32_t activation_type) {
    reset_arenas();
    const uint32_t k_dim = 128;
    const uint32_t mid_dim = 128;
    const uint32_t out_dim = 32;
    const uint64_t task_offset = 0;
    const uint64_t act_offset = 4096;
    const uint64_t act_scale_offset = 65536;
    const uint64_t gate_q0 = 131072;
    const uint64_t gate_q1 = 196608;
    const uint64_t gate_s0 = 262144;
    const uint64_t gate_s1 = 278528;
    const uint64_t up_q0 = 327680;
    const uint64_t up_q1 = 393216;
    const uint64_t up_s0 = 458752;
    const uint64_t up_s1 = 475136;
    const uint64_t down_q0 = 524288;
    const uint64_t down_q1 = 589824;
    const uint64_t down_s0 = 655360;
    const uint64_t down_s1 = 671744;
    const uint64_t out_offset = 1048576;

    int8_t a[2 * 128];
    int8_t gate_w[128 * 128];
    int8_t up_w[128 * 128];
    int8_t down_w[128 * 32];
    int32_t gold[2 * 32];

    fill_activation(act_offset, act_scale_offset, rows, k_dim,
                    ((k_dim + AXI_BYTES - 1) / AXI_BYTES) * AXI_BYTES, a);
    fill_weight(31 + task_type, k_dim, mid_dim, gate_w);
    fill_weight(47 + task_type, k_dim, mid_dim, up_w);
    fill_weight(73 + task_type, mid_dim, out_dim, down_w);

    pack_weight_ports(gate_q0, gate_q1, gate_s0, gate_s1, gate_w, k_dim, mid_dim);
    pack_weight_ports(up_q0, up_q1, up_s0, up_s1, up_w, k_dim, mid_dim);
    pack_weight_ports(down_q0, down_q1, down_s0, down_s1, down_w, mid_dim, out_dim);
    golden_ffn(a, gate_w, up_w, down_w, gold, rows, k_dim, mid_dim, out_dim, activation_type);

    linear_task_t task = make_task(task_type, engine, rows, k_dim, act_offset, act_scale_offset);
    task.ffn_activation = activation_type;
    task.ffn_input_cols = k_dim;
    task.ffn_intermediate_cols = mid_dim;
    task.ffn_output_cols = out_dim;
    task.gate_q_offset_bytes = gate_q0;
    task.gate_q_offset_bytes_port1 = gate_q1;
    task.gate_scale_offset_bytes = gate_s0;
    task.gate_scale_offset_bytes_port1 = gate_s1;
    task.up_q_offset_bytes = up_q0;
    task.up_q_offset_bytes_port1 = up_q1;
    task.up_scale_offset_bytes = up_s0;
    task.up_scale_offset_bytes_port1 = up_s1;
    task.down_q_offset_bytes = down_q0;
    task.down_q_offset_bytes_port1 = down_q1;
    task.down_scale_offset_bytes = down_s0;
    task.down_scale_offset_bytes_port1 = down_s1;
    task.ffn_dst_offset_bytes = out_offset;
    write_task(task_offset, task);

    uint32_t status = 0;
    vlm_engine_v3(weight0_arena, weight1_arena, act_arena, out_arena,
                  task_offset, 2 * 1024 * 1024, 1, status);
    if (status != VLM_W8A8_STATUS_DONE) {
        std::printf("FAIL: %s status=%u\n", name, status);
        return 1;
    }
    return compare_output(name, out_offset, gold, rows, out_dim);
}

int main() {
    std::printf("vlm_engine_v3 testbench\n");
    int errors = 0;
    errors += run_dense_like_case("GEMM_DENSE_O M2", GEMM_DENSE_O, LINEAR_ENGINE_GEMM, 2, 1);
    errors += run_dense_like_case("GEMM_FUSED_QKV M2", GEMM_FUSED_QKV, LINEAR_ENGINE_GEMM, 2, 3);
    errors += run_dense_like_case("GEMV_DENSE_O M1", GEMV_DENSE_O, LINEAR_ENGINE_GEMV, 1, 1);
    errors += run_dense_like_case("GEMV_FUSED_DECODE_QKV M1", GEMV_FUSED_DECODE_QKV, LINEAR_ENGINE_GEMV, 1, 3);
    errors += run_ffn_case("GEMM_FUSED_TEXT_FFN M2", GEMM_FUSED_TEXT_FFN, LINEAR_ENGINE_GEMM, 2, LINEAR_FFN_ACT_SILU);
    errors += run_ffn_case("GEMM_FUSED_VISION_FFN M2", GEMM_FUSED_VISION_FFN, LINEAR_ENGINE_GEMM, 2, LINEAR_FFN_ACT_GELU);
    errors += run_ffn_case("GEMV_FUSED_DECODE_FFN M1", GEMV_FUSED_DECODE_FFN, LINEAR_ENGINE_GEMV, 1, LINEAR_FFN_ACT_SILU);

    if (errors == 0) {
        std::printf("vlm_engine_v3_tb: pass\n");
    } else {
        std::printf("vlm_engine_v3_tb: fail errors=%d\n", errors);
    }
    return errors == 0 ? 0 : 1;
}
