#include "accelerator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHidden = VLM_W8A8_TEXT_HIDDEN;
constexpr uint32_t kKv = VLM_W8A8_TEXT_KV;
constexpr uint32_t kFfn = VLM_W8A8_TEXT_FFN;
constexpr uint32_t kVisionHidden = VLM_W8A8_VISION_HIDDEN;
constexpr uint32_t kVisionFfn = VLM_W8A8_VISION_FFN;
constexpr uint32_t kGenericMaxDim = VLM_W8A8_LINEAR_GENERIC_MAX_DIM;
constexpr uint32_t kTileN = VLM_W8A8_LINEAR_TILE_N;
constexpr uint32_t kTileK = VLM_W8A8_LINEAR_K_BLOCK;
constexpr uint32_t kScalePerTile = kTileK / VLM_W8A8_QK;
constexpr uint32_t kRowsQ = 2;
constexpr uint32_t kRowsF = 2;
constexpr uint32_t kRowsGeneric = 2;
constexpr uint32_t kHiddenRowStride = ((kHidden + kTileK - 1u) / kTileK) * kTileK;
constexpr uint32_t kArenaDepthWords = 1048576;
constexpr size_t kArenaAlign = 128;
constexpr int8_t kTbScaleExpDisabled = -128;

struct ArenaBuilder {
    std::vector<uint8_t> bytes;

    uint64_t append_zero(size_t len, size_t align = kArenaAlign) {
        return append(nullptr, len, align);
    }

    uint64_t append(const void * src, size_t len, size_t align = kArenaAlign) {
        const size_t mask = align - 1u;
        const size_t offset = (bytes.size() + mask) & ~mask;
        if (offset > bytes.size()) {
            bytes.resize(offset, 0);
        }
        const uint64_t out = static_cast<uint64_t>(offset);
        bytes.resize(offset + len, 0);
        if (src != nullptr && len != 0u) {
            std::memcpy(bytes.data() + offset, src, len);
        }
        return out;
    }
};

static uint32_t align_up_u32(uint32_t value, uint32_t align) {
    return ((value + align - 1u) / align) * align;
}

static uint64_t packed_q_index(uint32_t k_padded, uint32_t out_cols_padded, uint32_t col, uint32_t k_idx) {
    const uint32_t k_tile_count = k_padded / kTileK;
    const uint32_t n_tile = col / kTileN;
    const uint32_t lane = col % kTileN;
    const uint32_t k_tile = k_idx / kTileK;
    const uint32_t k_in_tile = k_idx % kTileK;
    (void) out_cols_padded;
    return (((n_tile * k_tile_count + k_tile) * kTileN + lane) * kTileK) + k_in_tile;
}

static uint64_t packed_scale_index(uint32_t k_padded, uint32_t out_cols_padded, uint32_t col, uint32_t kb) {
    const uint32_t k_tile_count = k_padded / kTileK;
    const uint32_t n_tile = col / kTileN;
    const uint32_t lane = col % kTileN;
    (void) out_cols_padded;
    return (((n_tile * k_tile_count + (kb / kScalePerTile)) * kTileN + lane) * kScalePerTile) + (kb % kScalePerTile);
}

static uint64_t packed_q_port_index(uint32_t k_padded, uint32_t out_cols_padded, uint32_t col, uint32_t k_idx) {
    const uint32_t port_tile_n = kTileN / VLM_W8A8_WEIGHT_PORTS;
    const uint32_t k_tile_count = k_padded / kTileK;
    const uint32_t n_tile = col / kTileN;
    const uint32_t lane = col % port_tile_n;
    const uint32_t k_tile = k_idx / kTileK;
    const uint32_t k_in_tile = k_idx % kTileK;
    (void) out_cols_padded;
    return (((n_tile * k_tile_count + k_tile) * port_tile_n + lane) * kTileK) + k_in_tile;
}

static uint64_t packed_scale_port_index(uint32_t k_padded, uint32_t out_cols_padded, uint32_t col, uint32_t kb) {
    const uint32_t port_tile_n = kTileN / VLM_W8A8_WEIGHT_PORTS;
    const uint32_t k_tile_count = k_padded / kTileK;
    const uint32_t n_tile = col / kTileN;
    const uint32_t lane = col % port_tile_n;
    (void) out_cols_padded;
    return (((n_tile * k_tile_count + (kb / kScalePerTile)) * port_tile_n + lane) * kScalePerTile) + (kb % kScalePerTile);
}

static float exp2_scale(int8_t exp) {
    if (exp == kTbScaleExpDisabled) {
        return 0.0f;
    }
    return std::ldexp(1.0f, static_cast<int>(exp));
}

static float tb_gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118f));
}

static float tb_hard_ffn_activation(float x, uint32_t activation) {
    if (activation == LINEAR_FFN_ACT_GELU) {
        const float gate = std::max(0.0f, std::min(3.0f, x + 1.5f));
        return x * gate * (1.0f / 3.0f);
    }
    const float gate = std::max(0.0f, std::min(6.0f, x + 3.0f));
    return x * gate * (1.0f / 6.0f);
}

static int8_t clamp_exp(int exp) {
    exp = std::max(-30, std::min(30, exp));
    return static_cast<int8_t>(exp);
}

static int8_t encode_pot_ceil(float scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        return 0;
    }
    return clamp_exp(static_cast<int>(std::ceil(std::log2(scale))));
}

static void encode_apot(float scale, int8_t * e0, int8_t * e1) {
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        *e0 = kTbScaleExpDisabled;
        *e1 = kTbScaleExpDisabled;
        return;
    }
    *e0 = clamp_exp(static_cast<int>(std::floor(std::log2(scale))));
    const float residual = std::max(0.0f, scale - exp2_scale(*e0));
    *e1 = residual > 0.0f ? clamp_exp(static_cast<int>(std::round(std::log2(residual)))) :
            kTbScaleExpDisabled;
}

static float apot_scale(const std::vector<int8_t> & scales, uint64_t idx) {
    const int8_t e0 = scales[static_cast<size_t>(idx * 2u)];
    const int8_t e1 = scales[static_cast<size_t>(idx * 2u + 1u)];
    return exp2_scale(e0) + exp2_scale(e1);
}

static void fill_activation(uint32_t rows, uint32_t k_size, uint32_t row_stride, std::vector<int8_t> * act_q, std::vector<int8_t> * act_scale) {
    const uint32_t scale_groups = row_stride / VLM_W8A8_QK;
    act_q->assign(static_cast<size_t>(rows * row_stride), 0);
    act_scale->assign(static_cast<size_t>(rows * scale_groups), 0);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t group = 0; group < scale_groups; ++group) {
            const float scale = 0.01171875f + 0.001953125f * static_cast<float>((row + group) % 9u);
            (*act_scale)[static_cast<size_t>(row) * scale_groups + group] = encode_pot_ceil(scale);
            const uint32_t base = group * VLM_W8A8_QK;
            for (uint32_t off = 0; off < VLM_W8A8_QK; ++off) {
                const uint32_t col = base + off;
                if (col >= k_size) {
                    break;
                }
                const int value = static_cast<int>((row * 17u + col * 5u + group * 3u) % 19u) - 9;
                (*act_q)[static_cast<size_t>(row) * row_stride + col] = static_cast<int8_t>(value);
            }
        }
    }
}

static void fill_weight(uint32_t out_cols, uint32_t k_size, std::vector<int8_t> * qbytes, std::vector<int8_t> * scales) {
    const uint32_t k_padded = align_up_u32(k_size, kTileK);
    const uint32_t out_padded = align_up_u32(out_cols, kTileN);
    const uint32_t k_blocks = k_padded / VLM_W8A8_QK;
    qbytes->assign(static_cast<size_t>(out_padded * k_padded), 0);
    scales->assign(static_cast<size_t>(out_padded * k_blocks * 2u), kTbScaleExpDisabled);
    for (uint32_t col = 0; col < out_cols; ++col) {
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            const float scale = 0.0078125f + 0.0009765625f * static_cast<float>((col + kb) % 7u);
            int8_t e0 = 0;
            int8_t e1 = kTbScaleExpDisabled;
            encode_apot(scale, &e0, &e1);
            const uint64_t scale_idx = packed_scale_index(k_padded, out_padded, col, kb);
            (*scales)[static_cast<size_t>(scale_idx * 2u)] = e0;
            (*scales)[static_cast<size_t>(scale_idx * 2u + 1u)] = e1;
            const uint32_t k_base = kb * VLM_W8A8_QK;
            for (uint32_t kk = 0; kk < VLM_W8A8_QK; ++kk) {
                const uint32_t k_idx = k_base + kk;
                if (k_idx >= k_size) {
                    continue;
                }
                const int value = static_cast<int>((col * 3u + k_idx * 11u + kb) % 23u) - 11;
                (*qbytes)[packed_q_index(k_padded, out_padded, col, k_idx)] = static_cast<int8_t>(value);
            }
        }
    }
}

struct WeightPortPayload {
    std::vector<int8_t> q0;
    std::vector<int8_t> q1;
    std::vector<int8_t> scale0;
    std::vector<int8_t> scale1;
};

static WeightPortPayload split_weight_ports(
        const std::vector<int8_t> & qbytes,
        const std::vector<int8_t> & scales,
        uint32_t out_cols,
        uint32_t k_size) {
    const uint32_t k_padded = align_up_u32(k_size, kTileK);
    const uint32_t out_padded = align_up_u32(out_cols, kTileN);
    const uint32_t k_blocks = k_padded / VLM_W8A8_QK;
    const uint32_t port_tile_n = kTileN / VLM_W8A8_WEIGHT_PORTS;
    WeightPortPayload payload;
    payload.q0.assign(static_cast<size_t>((out_padded / VLM_W8A8_WEIGHT_PORTS) * k_padded), 0);
    payload.q1.assign(static_cast<size_t>((out_padded / VLM_W8A8_WEIGHT_PORTS) * k_padded), 0);
    payload.scale0.assign(static_cast<size_t>((out_padded / VLM_W8A8_WEIGHT_PORTS) * k_blocks * 2u), kTbScaleExpDisabled);
    payload.scale1.assign(static_cast<size_t>((out_padded / VLM_W8A8_WEIGHT_PORTS) * k_blocks * 2u), kTbScaleExpDisabled);
    for (uint32_t col = 0; col < out_padded; ++col) {
        const bool use_port1 = (col % kTileN) >= port_tile_n;
        for (uint32_t k_idx = 0; k_idx < k_padded; ++k_idx) {
            const uint64_t src = packed_q_index(k_padded, out_padded, col, k_idx);
            const uint64_t dst = packed_q_port_index(k_padded, out_padded, col, k_idx);
            (use_port1 ? payload.q1 : payload.q0)[static_cast<size_t>(dst)] = qbytes[static_cast<size_t>(src)];
        }
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            const uint64_t src = packed_scale_index(k_padded, out_padded, col, kb);
            const uint64_t dst = packed_scale_port_index(k_padded, out_padded, col, kb);
            std::vector<int8_t> & port_scale = use_port1 ? payload.scale1 : payload.scale0;
            port_scale[static_cast<size_t>(dst * 2u)] = scales[static_cast<size_t>(src * 2u)];
            port_scale[static_cast<size_t>(dst * 2u + 1u)] = scales[static_cast<size_t>(src * 2u + 1u)];
        }
    }
    return payload;
}

static void append_weight_ports(
        ArenaBuilder * weight0_arena,
        ArenaBuilder * weight1_arena,
        const std::vector<int8_t> & qbytes,
        const std::vector<int8_t> & scales,
        uint32_t out_cols,
        uint32_t k_size,
        uint64_t * q0_offset,
        uint64_t * scale0_offset,
        uint64_t * q1_offset,
        uint64_t * scale1_offset) {
    const WeightPortPayload payload = split_weight_ports(qbytes, scales, out_cols, k_size);
    *q0_offset = weight0_arena->append(payload.q0.data(), payload.q0.size() * sizeof(int8_t));
    *scale0_offset = weight0_arena->append(payload.scale0.data(), payload.scale0.size() * sizeof(int8_t));
    *q1_offset = weight1_arena->append(payload.q1.data(), payload.q1.size() * sizeof(int8_t));
    *scale1_offset = weight1_arena->append(payload.scale1.data(), payload.scale1.size() * sizeof(int8_t));
}

static std::vector<float> reference_linear(
        const std::vector<int8_t> & act_q,
        const std::vector<int8_t> & act_scale_bits,
        uint32_t rows,
        uint32_t row_stride,
        const std::vector<int8_t> & weight_q,
        const std::vector<int8_t> & weight_scale_bits,
        uint32_t out_cols,
        uint32_t k_size) {
    const uint32_t k_padded = align_up_u32(k_size, kTileK);
    const uint32_t out_padded = align_up_u32(out_cols, kTileN);
    const uint32_t k_blocks = k_padded / VLM_W8A8_QK;
    const uint32_t act_scale_groups = row_stride / VLM_W8A8_QK;
    std::vector<float> out(static_cast<size_t>(rows * out_cols), 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < out_cols; ++col) {
            float acc = 0.0f;
            for (uint32_t kb = 0; kb < k_blocks; ++kb) {
                int32_t dot = 0;
                const float a_scale = exp2_scale(act_scale_bits[static_cast<size_t>(row) * act_scale_groups + kb]);
                const float w_scale = apot_scale(weight_scale_bits, packed_scale_index(k_padded, out_padded, col, kb));
                const uint32_t k_base = kb * VLM_W8A8_QK;
                for (uint32_t kk = 0; kk < VLM_W8A8_QK; ++kk) {
                    const uint32_t k_idx = k_base + kk;
                    if (k_idx >= k_size) {
                        continue;
                    }
                    dot += static_cast<int32_t>(act_q[static_cast<size_t>(row) * row_stride + k_idx]) *
                           static_cast<int32_t>(weight_q[packed_q_index(k_padded, out_padded, col, k_idx)]);
                }
                acc += static_cast<float>(dot) * (a_scale * w_scale);
            }
            out[static_cast<size_t>(row) * out_cols + col] = acc;
        }
    }
    return out;
}

static std::vector<vlm_w8a8_axi_t> pack_words(const std::vector<uint8_t> & bytes) {
    const size_t arena_bytes = static_cast<size_t>(kArenaDepthWords) * sizeof(vlm_w8a8_axi_t);
    if (bytes.size() > arena_bytes) {
        throw std::runtime_error("arena payload exceeds declared m_axi depth");
    }
    std::vector<vlm_w8a8_axi_t> words(kArenaDepthWords, 0);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

static std::vector<uint8_t> unpack_words(const std::vector<vlm_w8a8_axi_t> & words) {
    std::vector<uint8_t> bytes(words.size() * sizeof(vlm_w8a8_axi_t), 0);
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}

static float load_f32(const std::vector<uint8_t> & bytes, uint64_t offset) {
    uint32_t bits = 0;
    std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
    return vlm_w8a8_bits_to_f32(bits);
}

static vlm_w8a8_profile_t load_profile(const std::vector<uint8_t> & bytes, uint64_t offset) {
    vlm_w8a8_profile_t profile = {};
    std::memcpy(&profile, bytes.data() + offset, sizeof(profile));
    return profile;
}

static bool task_type_is_ffn(uint32_t task_type) {
    return task_type == GEMM_FUSED_TEXT_FFN ||
            task_type == GEMM_FUSED_VISION_FFN ||
            task_type == GEMV_FUSED_DECODE_FFN;
}

static void expect_profile_nonzero(const vlm_w8a8_profile_t & profile, uint32_t task_type, uint32_t rows) {
    if (profile.cycles_total == 0u) {
        throw std::runtime_error("profile cycles_total should be non-zero");
    }
    if (profile.cycles_read_task == 0u || profile.cycles_load_act == 0u || profile.cycles_load_weight == 0u) {
        throw std::runtime_error("profile stage counters should be non-zero");
    }
    if (profile.cycles_compute_a == 0u || profile.cycles_compute_b == 0u || profile.cycles_store == 0u) {
        throw std::runtime_error("profile compute/store counters should be non-zero");
    }
    if (profile.total_rows != rows) {
        throw std::runtime_error("profile total_rows mismatch");
    }
    if (!task_type_is_ffn(task_type)) {
        if (profile.task_count_dense != 1u || profile.cycles_dense == 0u) {
            throw std::runtime_error("dense profile counters missing");
        }
    } else {
        if (profile.task_count_ffn != 1u || profile.cycles_ffn == 0u) {
            throw std::runtime_error("ffn profile counters missing");
        }
        if (profile.cycles_ffn_gate == 0u || profile.cycles_ffn_up == 0u ||
            profile.cycles_ffn_quantize == 0u || profile.cycles_ffn_down == 0u) {
            throw std::runtime_error("ffn breakdown counters missing");
        }
    }
}

static std::vector<float> reference_ffn_down_tiled(
        const std::vector<float> & mid,
        uint32_t rows,
        uint32_t intermediate_cols,
        uint32_t output_cols,
        const std::vector<int8_t> & down_q,
        const std::vector<int8_t> & down_scale_bits) {
    const uint32_t k_padded = align_up_u32(intermediate_cols, kTileK);
    const uint32_t out_padded = align_up_u32(output_cols, kTileN);
    std::vector<float> out(static_cast<size_t>(rows * output_cols), 0.0f);

    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t mid_base = 0; mid_base < intermediate_cols; mid_base += kTileN) {
            int8_t mid_q_tile[kTileN];
            float act_scale_tile[kTileN / VLM_W8A8_QK];
            for (uint32_t group = 0; group < kTileN / VLM_W8A8_QK; ++group) {
                float max_abs = 0.0f;
                for (uint32_t off = 0; off < VLM_W8A8_QK; ++off) {
                    const uint32_t lane = group * VLM_W8A8_QK + off;
                    max_abs = std::max(max_abs, std::fabs(mid[static_cast<size_t>(row) * intermediate_cols + mid_base + lane]));
                }
                const int8_t act_exp = encode_pot_ceil(max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f);
                act_scale_tile[group] = exp2_scale(act_exp);
                for (uint32_t off = 0; off < VLM_W8A8_QK; ++off) {
                    const uint32_t lane = group * VLM_W8A8_QK + off;
                    float qf = mid[static_cast<size_t>(row) * intermediate_cols + mid_base + lane] / act_scale_tile[group];
                    if (qf > 127.0f) {
                        qf = 127.0f;
                    } else if (qf < -127.0f) {
                        qf = -127.0f;
                    }
                    mid_q_tile[lane] = static_cast<int8_t>(qf >= 0.0f ? qf + 0.5f : qf - 0.5f);
                }
            }

            for (uint32_t col = 0; col < output_cols; ++col) {
                for (uint32_t group = 0; group < kTileN / VLM_W8A8_QK; ++group) {
                    int32_t dot = 0;
                    const uint32_t kb = (mid_base / VLM_W8A8_QK) + group;
                    const float w_scale = apot_scale(down_scale_bits, packed_scale_index(k_padded, out_padded, col, kb));
                    for (uint32_t off = 0; off < VLM_W8A8_QK; ++off) {
                        const uint32_t lane = group * VLM_W8A8_QK + off;
                        dot += static_cast<int32_t>(mid_q_tile[lane]) *
                               static_cast<int32_t>(down_q[packed_q_index(k_padded, out_padded, col, mid_base + lane)]);
                    }
                    out[static_cast<size_t>(row) * output_cols + col] += static_cast<float>(dot) * (act_scale_tile[group] * w_scale);
                }
            }
        }
    }

    return out;
}

static void check_fixed_activation_approximation() {
    float max_silu_error = 0.0f;
    float max_gelu_error = 0.0f;
    for (int i = -64; i <= 64; ++i) {
        const float x = static_cast<float>(i) * 0.0625f;
        const float silu = x / (1.0f + std::exp(-x));
        max_silu_error = std::max(max_silu_error, std::fabs(tb_hard_ffn_activation(x, LINEAR_FFN_ACT_SILU) - silu));
        max_gelu_error = std::max(max_gelu_error, std::fabs(tb_hard_ffn_activation(x, LINEAR_FFN_ACT_GELU) - tb_gelu(x)));
    }
    if (max_silu_error > 0.2f || max_gelu_error > 0.2f) {
        throw std::runtime_error("fixed FFN activation approximation drifted");
    }
}

static void run_qkv_case(uint32_t rows, uint32_t task_type, uint32_t engine, const char * label) {
    std::vector<int8_t> act_q;
    std::vector<int8_t> act_scale;
    std::vector<int8_t> q_weight;
    std::vector<int8_t> k_weight;
    std::vector<int8_t> v_weight;
    std::vector<int8_t> q_scale;
    std::vector<int8_t> k_scale;
    std::vector<int8_t> v_scale;
    fill_activation(rows, kHidden, kHiddenRowStride, &act_q, &act_scale);
    fill_weight(kHidden, kHidden, &q_weight, &q_scale);
    fill_weight(kKv, kHidden, &k_weight, &k_scale);
    fill_weight(kKv, kHidden, &v_weight, &v_scale);

    ArenaBuilder data_arena;
    ArenaBuilder weight0_arena;
    ArenaBuilder weight1_arena;
    const uint64_t task_offset = data_arena.append_zero(sizeof(accelerator_task_t), alignof(accelerator_task_t));
    const uint64_t act_q_offset = data_arena.append(act_q.data(), act_q.size() * sizeof(int8_t));
    const uint64_t act_scale_offset = data_arena.append(act_scale.data(), act_scale.size() * sizeof(int8_t));
    uint64_t q_weight_offset = 0, q_scale_offset = 0, q_weight_offset_p1 = 0, q_scale_offset_p1 = 0;
    uint64_t k_weight_offset = 0, k_scale_offset = 0, k_weight_offset_p1 = 0, k_scale_offset_p1 = 0;
    uint64_t v_weight_offset = 0, v_scale_offset = 0, v_weight_offset_p1 = 0, v_scale_offset_p1 = 0;
    append_weight_ports(&weight0_arena, &weight1_arena, q_weight, q_scale, kHidden, kHidden,
            &q_weight_offset, &q_scale_offset, &q_weight_offset_p1, &q_scale_offset_p1);
    append_weight_ports(&weight0_arena, &weight1_arena, k_weight, k_scale, kKv, kHidden,
            &k_weight_offset, &k_scale_offset, &k_weight_offset_p1, &k_scale_offset_p1);
    append_weight_ports(&weight0_arena, &weight1_arena, v_weight, v_scale, kKv, kHidden,
            &v_weight_offset, &v_scale_offset, &v_weight_offset_p1, &v_scale_offset_p1);
    const uint64_t q_dst_offset = data_arena.append_zero(static_cast<size_t>(rows * kHidden) * sizeof(float));
    const uint64_t k_dst_offset = data_arena.append_zero(static_cast<size_t>(rows * kKv) * sizeof(float));
    const uint64_t v_dst_offset = data_arena.append_zero(static_cast<size_t>(rows * kKv) * sizeof(float));
    const uint64_t profile_offset = data_arena.append_zero(sizeof(vlm_w8a8_profile_t), alignof(vlm_w8a8_profile_t));

    accelerator_task_t task = {};
    task.magic = VLM_W8A8_TASK_MAGIC;
    task.version = VLM_W8A8_ABI_VERSION;
    task.task_type = task_type;
    task.rows = rows;
    task.act_q_offset_bytes = act_q_offset;
    task.act_scale_offset_bytes = act_scale_offset;
    task.act_row_stride = kHiddenRowStride;
    task.output_count = 3;
    task.input_cols = kHidden;
    task.engine = engine;
    task.out_cols[0] = kHidden;
    task.out_cols[1] = kKv;
    task.out_cols[2] = kKv;
    task.weight_q_offset_bytes[0] = q_weight_offset;
    task.weight_scale_offset_bytes[0] = q_scale_offset;
    task.weight_q_offset_bytes_port1[0] = q_weight_offset_p1;
    task.weight_scale_offset_bytes_port1[0] = q_scale_offset_p1;
    task.dst_offset_bytes[0] = q_dst_offset;
    task.weight_q_offset_bytes[1] = k_weight_offset;
    task.weight_scale_offset_bytes[1] = k_scale_offset;
    task.weight_q_offset_bytes_port1[1] = k_weight_offset_p1;
    task.weight_scale_offset_bytes_port1[1] = k_scale_offset_p1;
    task.dst_offset_bytes[1] = k_dst_offset;
    task.weight_q_offset_bytes[2] = v_weight_offset;
    task.weight_scale_offset_bytes[2] = v_scale_offset;
    task.weight_q_offset_bytes_port1[2] = v_weight_offset_p1;
    task.weight_scale_offset_bytes_port1[2] = v_scale_offset_p1;
    task.dst_offset_bytes[2] = v_dst_offset;
    task.profile_offset_bytes = profile_offset;
    std::memcpy(data_arena.bytes.data() + task_offset, &task, sizeof(task));

    std::vector<vlm_w8a8_axi_t> data_words = pack_words(data_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight0_words = pack_words(weight0_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight1_words = pack_words(weight1_arena.bytes);
    uint32_t status = 0;
    vlm_engine_v2(weight0_words.data(), weight1_words.data(), data_words.data(), data_words.data(), task_offset, profile_offset, 1, status);
    if (status != VLM_W8A8_STATUS_DONE) {
        throw std::runtime_error(std::string(label) + " status failed");
    }

    const std::vector<uint8_t> out_bytes = unpack_words(data_words);
    const vlm_w8a8_profile_t profile = load_profile(out_bytes, profile_offset);
    const std::vector<float> q_ref = reference_linear(act_q, act_scale, rows, kHiddenRowStride, q_weight, q_scale, kHidden, kHidden);
    const std::vector<float> k_ref = reference_linear(act_q, act_scale, rows, kHiddenRowStride, k_weight, k_scale, kKv, kHidden);
    const std::vector<float> v_ref = reference_linear(act_q, act_scale, rows, kHiddenRowStride, v_weight, v_scale, kKv, kHidden);
    float max_abs = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < kHidden; ++col) {
            const float got = load_f32(out_bytes, q_dst_offset + (static_cast<uint64_t>(row) * kHidden + col) * sizeof(float));
            max_abs = std::max(max_abs, std::fabs(got - q_ref[static_cast<size_t>(row) * kHidden + col]));
        }
        for (uint32_t col = 0; col < kKv; ++col) {
            const float got_k = load_f32(out_bytes, k_dst_offset + (static_cast<uint64_t>(row) * kKv + col) * sizeof(float));
            const float got_v = load_f32(out_bytes, v_dst_offset + (static_cast<uint64_t>(row) * kKv + col) * sizeof(float));
            max_abs = std::max(max_abs, std::fabs(got_k - k_ref[static_cast<size_t>(row) * kKv + col]));
            max_abs = std::max(max_abs, std::fabs(got_v - v_ref[static_cast<size_t>(row) * kKv + col]));
        }
    }
    if (max_abs > 1e-3f) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
    expect_profile_nonzero(profile, task_type, rows);
}

static void run_decode_gemv_case() {
    constexpr uint32_t kRowsDecode = 1;
    std::vector<int8_t> act_q;
    std::vector<int8_t> act_scale;
    std::vector<int8_t> weight_q;
    std::vector<int8_t> weight_scale;
    fill_activation(kRowsDecode, kHidden, kHiddenRowStride, &act_q, &act_scale);
    fill_weight(kHidden, kHidden, &weight_q, &weight_scale);

    ArenaBuilder data_arena;
    ArenaBuilder weight0_arena;
    ArenaBuilder weight1_arena;
    const uint64_t task_offset = data_arena.append_zero(sizeof(accelerator_task_t), alignof(accelerator_task_t));
    const uint64_t act_q_offset = data_arena.append(act_q.data(), act_q.size() * sizeof(int8_t));
    const uint64_t act_scale_offset = data_arena.append(act_scale.data(), act_scale.size() * sizeof(int8_t));
    uint64_t weight_offset = 0, scale_offset = 0, weight_offset_p1 = 0, scale_offset_p1 = 0;
    append_weight_ports(&weight0_arena, &weight1_arena, weight_q, weight_scale, kHidden, kHidden,
            &weight_offset, &scale_offset, &weight_offset_p1, &scale_offset_p1);
    const uint64_t dst_offset = data_arena.append_zero(static_cast<size_t>(kRowsDecode * kHidden) * sizeof(float));
    const uint64_t profile_offset = data_arena.append_zero(sizeof(vlm_w8a8_profile_t), alignof(vlm_w8a8_profile_t));

    accelerator_task_t task = {};
    task.magic = VLM_W8A8_TASK_MAGIC;
    task.version = VLM_W8A8_ABI_VERSION;
    task.task_type = GEMV_DENSE_O;
    task.rows = kRowsDecode;
    task.act_q_offset_bytes = act_q_offset;
    task.act_scale_offset_bytes = act_scale_offset;
    task.act_row_stride = kHiddenRowStride;
    task.output_count = 1;
    task.input_cols = kHidden;
    task.engine = LINEAR_ENGINE_GEMV;
    task.out_cols[0] = kHidden;
    task.weight_q_offset_bytes[0] = weight_offset;
    task.weight_scale_offset_bytes[0] = scale_offset;
    task.weight_q_offset_bytes_port1[0] = weight_offset_p1;
    task.weight_scale_offset_bytes_port1[0] = scale_offset_p1;
    task.dst_offset_bytes[0] = dst_offset;
    task.profile_offset_bytes = profile_offset;
    std::memcpy(data_arena.bytes.data() + task_offset, &task, sizeof(task));

    std::vector<vlm_w8a8_axi_t> data_words = pack_words(data_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight0_words = pack_words(weight0_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight1_words = pack_words(weight1_arena.bytes);
    uint32_t status = 0;
    vlm_engine_v2(weight0_words.data(), weight1_words.data(), data_words.data(), data_words.data(), task_offset, profile_offset, 1, status);
    if (status != VLM_W8A8_STATUS_DONE) {
        throw std::runtime_error("decode gemv status failed");
    }

    const std::vector<uint8_t> out_bytes = unpack_words(data_words);
    const vlm_w8a8_profile_t profile = load_profile(out_bytes, profile_offset);
    const std::vector<float> ref = reference_linear(act_q, act_scale, kRowsDecode, kHiddenRowStride, weight_q, weight_scale, kHidden, kHidden);
    float max_abs = 0.0f;
    for (uint32_t col = 0; col < kHidden; ++col) {
        const float got = load_f32(out_bytes, dst_offset + static_cast<uint64_t>(col) * sizeof(float));
        max_abs = std::max(max_abs, std::fabs(got - ref[col]));
    }
    if (max_abs > 1e-3f) {
        throw std::runtime_error("decode gemv output mismatch");
    }
    expect_profile_nonzero(profile, GEMV_DENSE_O, kRowsDecode);
}

static void run_ffn_case(
        uint32_t rows,
        uint32_t input_cols,
        uint32_t intermediate_cols,
        uint32_t output_cols,
        uint32_t activation,
        uint32_t task_type,
        uint32_t engine,
        const char * label) {
    const uint32_t row_stride = align_up_u32(input_cols, kTileK);
    std::vector<int8_t> act_q;
    std::vector<int8_t> act_scale;
    std::vector<int8_t> gate_q;
    std::vector<int8_t> up_q;
    std::vector<int8_t> down_q;
    std::vector<int8_t> gate_scale;
    std::vector<int8_t> up_scale;
    std::vector<int8_t> down_scale;
    fill_activation(rows, input_cols, row_stride, &act_q, &act_scale);
    fill_weight(intermediate_cols, input_cols, &gate_q, &gate_scale);
    fill_weight(intermediate_cols, input_cols, &up_q, &up_scale);
    fill_weight(output_cols, intermediate_cols, &down_q, &down_scale);

    ArenaBuilder data_arena;
    ArenaBuilder weight0_arena;
    ArenaBuilder weight1_arena;
    const uint64_t task_offset = data_arena.append_zero(sizeof(accelerator_task_t), alignof(accelerator_task_t));
    const uint64_t act_q_offset = data_arena.append(act_q.data(), act_q.size() * sizeof(int8_t));
    const uint64_t act_scale_offset = data_arena.append(act_scale.data(), act_scale.size() * sizeof(int8_t));
    uint64_t gate_q_offset = 0, gate_scale_offset = 0, gate_q_offset_p1 = 0, gate_scale_offset_p1 = 0;
    uint64_t up_q_offset = 0, up_scale_offset = 0, up_q_offset_p1 = 0, up_scale_offset_p1 = 0;
    uint64_t down_q_offset = 0, down_scale_offset = 0, down_q_offset_p1 = 0, down_scale_offset_p1 = 0;
    append_weight_ports(&weight0_arena, &weight1_arena, gate_q, gate_scale, intermediate_cols, input_cols,
            &gate_q_offset, &gate_scale_offset, &gate_q_offset_p1, &gate_scale_offset_p1);
    append_weight_ports(&weight0_arena, &weight1_arena, up_q, up_scale, intermediate_cols, input_cols,
            &up_q_offset, &up_scale_offset, &up_q_offset_p1, &up_scale_offset_p1);
    append_weight_ports(&weight0_arena, &weight1_arena, down_q, down_scale, output_cols, intermediate_cols,
            &down_q_offset, &down_scale_offset, &down_q_offset_p1, &down_scale_offset_p1);
    const uint64_t dst_offset = data_arena.append_zero(static_cast<size_t>(rows * output_cols) * sizeof(float));
    const uint64_t profile_offset = data_arena.append_zero(sizeof(vlm_w8a8_profile_t), alignof(vlm_w8a8_profile_t));

    accelerator_task_t task = {};
    task.magic = VLM_W8A8_TASK_MAGIC;
    task.version = VLM_W8A8_ABI_VERSION;
    task.task_type = task_type;
    task.rows = rows;
    task.act_q_offset_bytes = act_q_offset;
    task.act_scale_offset_bytes = act_scale_offset;
    task.act_row_stride = row_stride;
    task.engine = engine;
    task.ffn_activation = activation;
    task.gate_q_offset_bytes = gate_q_offset;
    task.gate_scale_offset_bytes = gate_scale_offset;
    task.gate_q_offset_bytes_port1 = gate_q_offset_p1;
    task.gate_scale_offset_bytes_port1 = gate_scale_offset_p1;
    task.up_q_offset_bytes = up_q_offset;
    task.up_scale_offset_bytes = up_scale_offset;
    task.up_q_offset_bytes_port1 = up_q_offset_p1;
    task.up_scale_offset_bytes_port1 = up_scale_offset_p1;
    task.down_q_offset_bytes = down_q_offset;
    task.down_scale_offset_bytes = down_scale_offset;
    task.down_q_offset_bytes_port1 = down_q_offset_p1;
    task.down_scale_offset_bytes_port1 = down_scale_offset_p1;
    task.ffn_dst_offset_bytes = dst_offset;
    task.profile_offset_bytes = profile_offset;
    task.ffn_input_cols = input_cols;
    task.ffn_intermediate_cols = intermediate_cols;
    task.ffn_output_cols = output_cols;
    std::memcpy(data_arena.bytes.data() + task_offset, &task, sizeof(task));

    std::vector<vlm_w8a8_axi_t> data_words = pack_words(data_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight0_words = pack_words(weight0_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight1_words = pack_words(weight1_arena.bytes);
    uint32_t status = 0;
    vlm_engine_v2(weight0_words.data(), weight1_words.data(), data_words.data(), data_words.data(), task_offset, profile_offset, 1, status);
    if (status != VLM_W8A8_STATUS_DONE) {
        throw std::runtime_error(std::string(label) + " status failed");
    }

    const std::vector<float> gate_ref = reference_linear(act_q, act_scale, rows, row_stride, gate_q, gate_scale, intermediate_cols, input_cols);
    const std::vector<float> up_ref = reference_linear(act_q, act_scale, rows, row_stride, up_q, up_scale, intermediate_cols, input_cols);
    std::vector<float> mid(static_cast<size_t>(rows * intermediate_cols), 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < intermediate_cols; ++col) {
            const size_t idx = static_cast<size_t>(row) * intermediate_cols + col;
            mid[idx] = tb_hard_ffn_activation(gate_ref[idx], activation) * up_ref[idx];
        }
    }

    const std::vector<float> down_ref = reference_ffn_down_tiled(mid, rows, intermediate_cols, output_cols, down_q, down_scale);

    const std::vector<uint8_t> out_bytes = unpack_words(data_words);
    const vlm_w8a8_profile_t profile = load_profile(out_bytes, profile_offset);
    float max_abs = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < output_cols; ++col) {
            const uint64_t idx = static_cast<uint64_t>(row) * output_cols + col;
            const float got = load_f32(out_bytes, dst_offset + idx * sizeof(float));
            max_abs = std::max(max_abs, std::fabs(got - down_ref[static_cast<size_t>(idx)]));
        }
    }
    if (max_abs > 2e-2f) {
        throw std::runtime_error(std::string(label) + " output mismatch max_abs=" + std::to_string(max_abs));
    }
    expect_profile_nonzero(profile, task_type, rows);
}

static void run_bad_ffn_shape_guard() {
    ArenaBuilder data_arena;
    ArenaBuilder weight0_arena;
    ArenaBuilder weight1_arena;
    const uint64_t task_offset = data_arena.append_zero(sizeof(accelerator_task_t), alignof(accelerator_task_t));
    const uint64_t profile_offset = data_arena.append_zero(sizeof(vlm_w8a8_profile_t), alignof(vlm_w8a8_profile_t));

    accelerator_task_t task = {};
    task.magic = VLM_W8A8_TASK_MAGIC;
    task.version = VLM_W8A8_ABI_VERSION;
    task.task_type = GEMM_FUSED_VISION_FFN;
    task.rows = kRowsF;
    task.act_row_stride = kHiddenRowStride;
    task.engine = LINEAR_ENGINE_GEMM;
    task.ffn_activation = LINEAR_FFN_ACT_SILU;
    task.ffn_input_cols = kHidden;
    task.ffn_intermediate_cols = kFfn;
    task.ffn_output_cols = kHidden;
    task.profile_offset_bytes = profile_offset;
    std::memcpy(data_arena.bytes.data() + task_offset, &task, sizeof(task));

    std::vector<vlm_w8a8_axi_t> data_words = pack_words(data_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight0_words = pack_words(weight0_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight1_words = pack_words(weight1_arena.bytes);
    uint32_t status = VLM_W8A8_STATUS_DONE;
    vlm_engine_v2(weight0_words.data(), weight1_words.data(), data_words.data(), data_words.data(), task_offset, profile_offset, 1, status);
    if (status != VLM_W8A8_STATUS_BAD_SHAPE) {
        throw std::runtime_error("bad FFN shape guard did not reject mismatched vision task");
    }
}

static void run_generic_dense_case(uint32_t rows, uint32_t k_size, uint32_t out_cols) {
    const uint32_t row_stride = align_up_u32(k_size, kTileK);
    std::vector<int8_t> act_q;
    std::vector<int8_t> act_scale;
    std::vector<int8_t> weight_q;
    std::vector<int8_t> weight_scale;
    fill_activation(rows, k_size, row_stride, &act_q, &act_scale);
    fill_weight(out_cols, k_size, &weight_q, &weight_scale);

    ArenaBuilder data_arena;
    ArenaBuilder weight0_arena;
    ArenaBuilder weight1_arena;
    const uint64_t task_offset = data_arena.append_zero(sizeof(accelerator_task_t), alignof(accelerator_task_t));
    const uint64_t act_q_offset = data_arena.append(act_q.data(), act_q.size() * sizeof(int8_t));
    const uint64_t act_scale_offset = data_arena.append(act_scale.data(), act_scale.size() * sizeof(int8_t));
    uint64_t weight_q_offset = 0, weight_scale_offset = 0, weight_q_offset_p1 = 0, weight_scale_offset_p1 = 0;
    append_weight_ports(&weight0_arena, &weight1_arena, weight_q, weight_scale, out_cols, k_size,
            &weight_q_offset, &weight_scale_offset, &weight_q_offset_p1, &weight_scale_offset_p1);
    const uint64_t dst_offset = data_arena.append_zero(static_cast<size_t>(rows * out_cols) * sizeof(float));
    const uint64_t profile_offset = data_arena.append_zero(sizeof(vlm_w8a8_profile_t), alignof(vlm_w8a8_profile_t));

    accelerator_task_t task = {};
    task.magic = VLM_W8A8_TASK_MAGIC;
    task.version = VLM_W8A8_ABI_VERSION;
    task.task_type = rows == 1u ? GEMV_DENSE_O : GEMM_DENSE_O;
    task.rows = rows;
    task.act_q_offset_bytes = act_q_offset;
    task.act_scale_offset_bytes = act_scale_offset;
    task.act_row_stride = row_stride;
    task.output_count = 1;
    task.input_cols = k_size;
    task.engine = rows == 1u ? LINEAR_ENGINE_GEMV : LINEAR_ENGINE_GEMM;
    task.out_cols[0] = out_cols;
    task.weight_q_offset_bytes[0] = weight_q_offset;
    task.weight_scale_offset_bytes[0] = weight_scale_offset;
    task.weight_q_offset_bytes_port1[0] = weight_q_offset_p1;
    task.weight_scale_offset_bytes_port1[0] = weight_scale_offset_p1;
    task.dst_offset_bytes[0] = dst_offset;
    task.profile_offset_bytes = profile_offset;
    std::memcpy(data_arena.bytes.data() + task_offset, &task, sizeof(task));

    std::vector<vlm_w8a8_axi_t> data_words = pack_words(data_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight0_words = pack_words(weight0_arena.bytes);
    std::vector<vlm_w8a8_axi_t> weight1_words = pack_words(weight1_arena.bytes);
    uint32_t status = 0;
    vlm_engine_v2(weight0_words.data(), weight1_words.data(), data_words.data(), data_words.data(), task_offset, profile_offset, 1, status);
    if (status != VLM_W8A8_STATUS_DONE) {
        throw std::runtime_error("accelerator generic dense status failed");
    }

    const std::vector<uint8_t> out_bytes = unpack_words(data_words);
    const vlm_w8a8_profile_t profile = load_profile(out_bytes, profile_offset);
    const std::vector<float> ref = reference_linear(act_q, act_scale, rows, row_stride, weight_q, weight_scale, out_cols, k_size);
    float max_abs = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < out_cols; ++col) {
            const float got = load_f32(out_bytes, dst_offset + (static_cast<uint64_t>(row) * out_cols + col) * sizeof(float));
            max_abs = std::max(max_abs, std::fabs(got - ref[static_cast<size_t>(row) * out_cols + col]));
        }
    }
    if (max_abs > 2e-2f) {
        throw std::runtime_error("accelerator generic dense output mismatch");
    }
    expect_profile_nonzero(profile, task.task_type, rows);
}

} // namespace

int main() {
    try {
        check_fixed_activation_approximation();
        run_qkv_case(kRowsQ, GEMM_FUSED_QKV, LINEAR_ENGINE_GEMM, "GEMM fused QKV");
        run_qkv_case(1u, GEMV_FUSED_DECODE_QKV, LINEAR_ENGINE_GEMV, "GEMV fused decode QKV");
        run_decode_gemv_case();
        run_ffn_case(kRowsF, kHidden, kFfn, kHidden, LINEAR_FFN_ACT_SILU, GEMM_FUSED_TEXT_FFN, LINEAR_ENGINE_GEMM, "GEMM fused text FFN");
        run_ffn_case(kRowsF, kVisionHidden, kVisionFfn, kVisionHidden, LINEAR_FFN_ACT_GELU, GEMM_FUSED_VISION_FFN, LINEAR_ENGINE_GEMM, "GEMM fused vision FFN");
        run_ffn_case(1u, kHidden, kFfn, kHidden, LINEAR_FFN_ACT_SILU, GEMV_FUSED_DECODE_FFN, LINEAR_ENGINE_GEMV, "GEMV fused decode FFN");
        run_bad_ffn_shape_guard();
        run_generic_dense_case(kRowsGeneric, 768u, 768u);
        run_generic_dense_case(kRowsGeneric, 768u, kGenericMaxDim);
        run_generic_dense_case(1u, kGenericMaxDim, 768u);
        std::cout << "vlm_gemm_gemv_engine_tb: pass\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "vlm_gemm_gemv_engine_tb: " << ex.what() << '\n';
        return 1;
    }
}
