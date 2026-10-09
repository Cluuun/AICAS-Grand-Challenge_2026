#ifndef FPGA_DRIVER_H
#define FPGA_DRIVER_H

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cerrno>
#include <stdexcept>

#include <filesystem>

#include <omp.h>

#include <arm_neon.h>

#include "xrt/xrt.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"

enum class Op : uint32_t {
    OP_LOAD_AS = 0,
    OP_LOAD_BS = 1,
    OP_PRELOAD_A = 2,   // iva + pam into fa caches
    OP_PRELOAD_B = 3,   // ivb + pbn into fb caches
    OP_INIT_C_GLOBAL = 4,
    OP_COMPUTE = 5,     // mtx / ntx / rsa
    OP_STORE_C = 6
};

enum class GemvOp : uint32_t {
    LOAD_A = 0,
    LOAD_B_SCALE = 1,
    LOAD_A_SCALE = 2,
    COMPUTE = 3,
    STORE_C = 4
};

// Geometry must match synthesized kernel (see test_v6.cpp).
struct GEMMShape {
    static constexpr int TOT_M = 512;
    static constexpr int TOT_N = 512;
    static constexpr int TOT_K = 256;
    static constexpr int K_BLK = 32;
    static constexpr int TILES = TOT_K / K_BLK; // 8
    static constexpr int M_TILE = 16;
    static constexpr int N_TILE = 16;
};

static constexpr int QK8_0 = 32;

static constexpr size_t WEIGHT_BO_BYTES =
    static_cast<size_t>(512) * 1024u * 1024u;
static constexpr size_t WEIGHT_SCALE_BO_BYTES =
    static_cast<size_t>(512 / 32) * 1024u * 1024u * sizeof(uint16_t);

inline uint16_t float_to_half(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));
    const uint32_t sign = (f_bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>(((f_bits >> 23) & 0xFFu)) - 127;
    uint32_t fraction = f_bits & 0x007FFFFFu;
    if (exponent == 128)
        return static_cast<uint16_t>(sign | 0x7C00u | (fraction ? (fraction >> 13) | 1u : 0u));
    if (exponent > 15)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (exponent < -14)
        return static_cast<uint16_t>(sign);
    uint32_t h_exp = static_cast<uint32_t>(exponent + 15);
    uint32_t h_frac = fraction >> 13;
    const uint32_t round_bit = 1u << 12;
    if ((fraction & round_bit) && ((fraction & (round_bit - 1)) || (h_frac & 1u)))
        h_frac++;
    if (h_frac >= 0x0400u) {
        h_frac = 0;
        h_exp++;
    }
    return static_cast<uint16_t>(sign | (h_exp << 10) | h_frac);
}

inline float half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t fraction = h & 0x03FFu;
    uint32_t f_bits = 0;
    if (exponent == 31) {
        f_bits = sign | 0x7F800000u | (fraction << 13);
    } else if (exponent == 0) {
        f_bits = sign;
    } else {
        exponent = exponent - 15 + 127;
        f_bits = sign | (static_cast<uint32_t>(exponent) << 23) | (fraction << 13);
    }
    float f;
    std::memcpy(&f, &f_bits, sizeof(f));
    return f;
}

struct FpgaGemmInnerProf {
    using Clock = std::chrono::steady_clock;

    static double ms(Clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    }

    static void record_gemm_set(double ms) {
        sum_set_ms_ += ms;
        hits_set_++;
    }
    static void record_gemm_run(double ms) {
        sum_run_ms_ += ms; hits_run_++;
    }
    static void record_gemm_load(double ms) {
        sum_load_ms_ += ms; hits_load_++;
    }

    static void take_sums_and_reset(
            double & set_ms,
            double & run_ms,
            double & load_ms,
            uint64_t & hits_set,
            uint64_t & hits_run,
            uint64_t & hits_load) {
        set_ms = sum_set_ms_;
        run_ms = sum_run_ms_;
        load_ms = sum_load_ms_;
        hits_set = hits_set_;
        hits_run = hits_run_;
        hits_load = hits_load_;
        sum_set_ms_ = 0.0;
        sum_run_ms_ = 0.0;
        sum_load_ms_ = 0.0;
        hits_set_ = 0;
        hits_run_ = 0;
        hits_load_ = 0;
    }

private:
    static inline double   sum_set_ms_  = 0.0;
    static inline double   sum_run_ms_  = 0.0;
    static inline double   sum_load_ms_ = 0.0;
    static inline uint64_t hits_set_    = 0;
    static inline uint64_t hits_run_    = 0;
    static inline uint64_t hits_load_   = 0;
};

class FpgaDriver {
public:
    static constexpr const char* kDefaultWeightBinDir =
        "/root/jupyter_notebooks/hw/driver/model/bin";
    /// On-disk blobs: `{tensor_name}.data.bin` / `{tensor_name}.scale.bin`
    /// where tensor_name includes `.weight` (e.g. `v.blk.1.attn_out.weight`).
    static constexpr const char* kSuffixWeightData = ".data.bin";
    static constexpr const char* kSuffixWeightScale = ".scale.bin";

    // FPGA-tiled on-disk LM weight sizes (model/bin/*.data.bin), not dense ne0*ne1.
    static constexpr size_t kTiledLmQoDataBytes =
        static_cast<size_t>(512) * 256 * 4 + static_cast<size_t>(448) * 256 * 4;
    static constexpr size_t kTiledLmKvDataBytes = static_cast<size_t>(320) * 256 * 4;
    static constexpr size_t kTiledLmFfnUpGateDataBytes =
        static_cast<size_t>(512) * 256 * 4 * 5;
    static constexpr size_t kTiledLmFfnDownDataBytes =
        static_cast<size_t>(512) * 256 * 10 + static_cast<size_t>(448) * 256 * 10;

    struct WeightSlot {
        uint64_t data_offset = 0;
        uint64_t scale_offset = 0;
        uint64_t data_bytes = 0;
        uint64_t scale_bytes = 0;
    };

    struct WeightEntry {
        uint64_t data_start_addr = 0;
        uint64_t scale_start_addr = 0;
        int n_count = 0;
    };

    struct ActSlot {
        uint64_t data_start_addr = 0;
        uint64_t scale_start_addr = 0;
        int m_count = 0;
    };

    // ---------- AXI-Lite control map (Vitis-generated, matches hls/gemm_driver.h) ----------
    static constexpr uint32_t ADDR_AP_CTRL       = 0x00;
    static constexpr uint32_t ADDR_GIE           = 0x04;
    static constexpr uint32_t ADDR_IER           = 0x08;
    static constexpr uint32_t ADDR_ISR           = 0x0c;
    static constexpr uint32_t ADDR_OP_DATA       = 0x10;
    static constexpr uint32_t ADDR_USE_M_DATA    = 0x18;
    static constexpr uint32_t ADDR_USE_N_DATA    = 0x20;
    static constexpr uint32_t ADDR_C_OFFSET_DATA = 0x28;
    static constexpr uint32_t ADDR_C_STRIDE_DATA = 0x30;
    static constexpr uint32_t ADDR_A_DATA        = 0x38;
    static constexpr uint32_t ADDR_B_DATA        = 0x44;
    static constexpr uint32_t ADDR_AS_DATA       = 0x50;
    static constexpr uint32_t ADDR_BS_DATA       = 0x5c;
    static constexpr uint32_t ADDR_C_DATA        = 0x68;
    /// Minimum /dev/mem mmap span covering all control registers (C @ 0x68 + 64b).
    static constexpr size_t CONTROL_MAP_BYTES = 0x70;

    // ---------- AXI-Lite control map (Vitis-generated, matches hls/gemv_driver.h) ----------
    static constexpr uint32_t GEMV_ADDR_A_IN_DATA        = 0x10;
    static constexpr uint32_t GEMV_ADDR_B_IN1_DATA       = 0x1c;
    static constexpr uint32_t GEMV_ADDR_B_IN2_DATA       = 0x28;
    static constexpr uint32_t GEMV_ADDR_B_IN3_DATA       = 0x34;
    static constexpr uint32_t GEMV_ADDR_B_IN4_DATA       = 0x40;
    static constexpr uint32_t GEMV_ADDR_SCALES1_DATA     = 0x4c;
    static constexpr uint32_t GEMV_ADDR_SCALES2_DATA     = 0x58;
    static constexpr uint32_t GEMV_ADDR_SCALES3_DATA     = 0x64;
    static constexpr uint32_t GEMV_ADDR_SCALES4_DATA     = 0x70;
    static constexpr uint32_t GEMV_ADDR_K_BLOCK_NUM_DATA = 0x7c;
    static constexpr uint32_t GEMV_ADDR_N_BLOCK_NUM_DATA = 0x84;
    static constexpr uint32_t GEMV_ADDR_OP_DATA          = 0x8c;
    static constexpr uint32_t GEMV_ADDR_C_OUT_DATA       = 0x94;

    static constexpr uint32_t BIT_AP_START     = 0x01;
    static constexpr uint32_t BIT_AP_DONE      = 0x02;
    static constexpr uint32_t BIT_AP_IDLE      = 0x04;
    static constexpr uint32_t BIT_AP_READY     = 0x08;
    static constexpr uint32_t BIT_AUTO_RESTART = 0x80;

    /// map_control=false skips /dev/mem (weight-only bring-up / memcpy test).
    explicit FpgaDriver(bool map_control = true)
        : device(0),
          weight_bo(device, WEIGHT_BO_BYTES, xrt::bo::flags::cacheable, 0),
          weight_scale_bo(device, WEIGHT_SCALE_BO_BYTES, xrt::bo::flags::cacheable, 0),
          act_bo(device, 20 * 1024 * 1024 * sizeof(char), xrt::bo::flags::cacheable, 0),
          act_scale_bo(device, 20 * 1024 * (1024 / 32) * sizeof(uint16_t), xrt::bo::flags::cacheable, 0),
          act_result_bo(device, 20 * 1024 * 1024 * sizeof(float), xrt::bo::flags::cacheable, 0),
          act_decode_bo(device, 20 * 1024 * 1024 * sizeof(char), xrt::bo::flags::normal, 0),
          act_scale_decode_bo(device, 20 * 1024 * (1024 / 32) * sizeof(uint16_t), xrt::bo::flags::normal, 0),
          act_result_decode_bo(device, 20 * 1024 * 1024 * sizeof(float), xrt::bo::flags::normal, 0) {
        if (map_control) {
            constexpr uint64_t kCtrl = 0xA0000000ull;
            if (!init(kCtrl, 0x1000))
                std::fprintf(stderr, "[FpgaDriver] control mmap failed (weights still load)\n");
            if (!load_weights_from_bin(kDefaultWeightBinDir))
                std::fprintf(stderr, "[FpgaDriver] load_weights_from_bin failed\n");
        } else {
            constexpr uint64_t kCtrl = 0xA0000000ull;
            if (!init(kCtrl, 0x1000))
                std::fprintf(stderr, "[FpgaDriver] control mmap failed (weights still load)\n");
        }
    }

    // for no xrt case
    explicit FpgaDriver(int device_index) {
        (void) device_index;
        init(0xA0000000ull, 0x1000);
    }

    ~FpgaDriver() { shutdown_control(); }

    bool init(uint64_t physAddr, size_t mapSize = 0x1000) {
        shutdown_control();
        m_fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (m_fd < 0) {
            std::fprintf(stderr, "open /dev/mem failed: %s\n", strerror(errno));
            return false;
        }
        m_base = reinterpret_cast<volatile uint32_t*>(
            mmap(nullptr, mapSize, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, physAddr));
        if (m_base == MAP_FAILED) {
            std::fprintf(stderr, "mmap failed: %s\n", strerror(errno));
            close(m_fd);
            m_fd = -1;
            m_base = nullptr;
            return false;
        }
        m_mapSize = mapSize;
        return true;
    }

    /// Pack pre-baked test_v6 weight bins into weight_bo / weight_scale_bo.
    bool load_weights_from_bin(const std::string& dir) {
        namespace fs = std::filesystem;
        if (!fs::is_directory(dir)) {
            std::fprintf(stderr, "weight bin dir not found: %s\n", dir.c_str());
            return false;
        }

        auto list_suffix = [&](const char* suffix) {
            std::vector<std::string> names;
            for (const auto& ent : fs::directory_iterator(dir)) {
                if (!ent.is_regular_file())
                    continue;
                const std::string fname = ent.path().filename().string();
                const std::size_t slen = std::strlen(suffix);
                if (fname.size() >= slen &&
                    fname.compare(fname.size() - slen, slen, suffix) == 0)
                    names.push_back(fname);
            }
            std::sort(names.begin(), names.end());
            return names;
        };

        auto key_from = [](const std::string& fname, const char* suffix) -> std::string {
            const std::size_t slen = std::strlen(suffix);
            if (fname.size() <= slen || fname.compare(fname.size() - slen, slen, suffix) != 0)
                throw std::runtime_error("bad weight filename: " + fname);
            return fname.substr(0, fname.size() - slen);
        };

        const std::vector<std::string> data_files = list_suffix(kSuffixWeightData);
        const std::vector<std::string> scale_files = list_suffix(kSuffixWeightScale);
        if (data_files.empty()) {
            std::fprintf(stderr, "no *%s under %s\n", kSuffixWeightData, dir.c_str());
            return false;
        }

        m_weights.clear();
        m_weight_data_used = 0;
        m_weight_scale_used = 0;

        auto* data_host = weight_bo.map<uint8_t*>();
        auto* scale_host = weight_scale_bo.map<uint16_t*>();
        std::memset(data_host, 0, WEIGHT_BO_BYTES);
        std::memset(scale_host, 0, WEIGHT_SCALE_BO_BYTES);

        auto append_weight = [&](const std::string& key) -> bool {
            if (m_weights.find(key) != m_weights.end())
                return true;

            const std::string data_path = dir + "/" + key + kSuffixWeightData;
            const std::string scale_path = dir + "/" + key + kSuffixWeightScale;

            std::ifstream df(data_path, std::ios::binary);
            if (!df) {
                std::fprintf(stderr, "open failed: %s\n", data_path.c_str());
                return false;
            }
            df.seekg(0, std::ios::end);
            const std::streamoff data_fsize = df.tellg();
            if (data_fsize < 0) {
                std::fprintf(stderr, "tellg failed: %s\n", data_path.c_str());
                return false;
            }
            df.seekg(0, std::ios::beg);
            const std::size_t data_nbytes = static_cast<std::size_t>(data_fsize);
            if (m_weight_data_used + data_nbytes > WEIGHT_BO_BYTES) {
                std::fprintf(stderr, "weight_bo overflow at %s\n", key.c_str());
                return false;
            }
            df.read(reinterpret_cast<char*>(data_host + m_weight_data_used),
                    static_cast<std::streamsize>(data_nbytes));
            if (!df) {
                std::fprintf(stderr, "read failed: %s\n", data_path.c_str());
                return false;
            }

            std::ifstream sf(scale_path, std::ios::binary);
            if (!sf) {
                std::fprintf(stderr, "open failed: %s\n", scale_path.c_str());
                return false;
            }
            sf.seekg(0, std::ios::end);
            const std::streamoff scale_fsize = sf.tellg();
            if (scale_fsize < 0) {
                std::fprintf(stderr, "tellg failed: %s\n", scale_path.c_str());
                return false;
            }
            sf.seekg(0, std::ios::beg);
            const std::size_t scale_nbytes = static_cast<std::size_t>(scale_fsize);
            if (m_weight_scale_used + scale_nbytes > WEIGHT_SCALE_BO_BYTES) {
                std::fprintf(stderr, "weight_scale_bo overflow at %s\n", key.c_str());
                return false;
            }
            sf.read(reinterpret_cast<char*>(scale_host) +
                        static_cast<std::ptrdiff_t>(m_weight_scale_used),
                    static_cast<std::streamsize>(scale_nbytes));
            if (!sf) {
                std::fprintf(stderr, "read failed: %s\n", scale_path.c_str());
                return false;
            }

            WeightSlot slot;
            slot.data_offset = m_weight_data_used;
            slot.data_bytes = data_nbytes;
            slot.scale_offset = m_weight_scale_used;
            slot.scale_bytes = scale_nbytes;
            m_weights[key] = slot;
            m_weight_data_used += data_nbytes;
            m_weight_scale_used += scale_nbytes;
            return true;
        };

        static constexpr const char* kOutputWeightKey = "output.weight";
        if (!append_weight(kOutputWeightKey))
            return false;

        for (const std::string& fname : data_files) {
            const std::string key = key_from(fname, kSuffixWeightData);
            if (m_weights.find(key) != m_weights.end())
                continue;
            const std::string path = dir + "/" + fname;
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "open failed: %s\n", path.c_str());
                return false;
            }
            f.seekg(0, std::ios::end);
            const std::streamoff fsize = f.tellg();
            if (fsize < 0) {
                std::fprintf(stderr, "tellg failed: %s\n", path.c_str());
                return false;
            }
            f.seekg(0, std::ios::beg);
            const std::size_t nbytes = static_cast<std::size_t>(fsize);
            if (m_weight_data_used + nbytes > WEIGHT_BO_BYTES) {
                std::fprintf(stderr, "weight_bo overflow at %s\n", key.c_str());
                return false;
            }
            f.read(reinterpret_cast<char*>(data_host + m_weight_data_used),
                   static_cast<std::streamsize>(nbytes));
            if (!f) {
                std::fprintf(stderr, "read failed: %s\n", path.c_str());
                return false;
            }
            WeightSlot slot;
            slot.data_offset = m_weight_data_used;
            slot.data_bytes = nbytes;
            m_weights[key] = slot;
            m_weight_data_used += nbytes;
        }

        for (const std::string& fname : scale_files) {
            const std::string key = key_from(fname, kSuffixWeightScale);
            auto it = m_weights.find(key);
            if (it == m_weights.end()) {
                std::fprintf(stderr, "scale without data: %s\n", key.c_str());
                return false;
            }
            if (it->second.scale_bytes != 0)
                continue;
            const std::string path = dir + "/" + fname;
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "open failed: %s\n", path.c_str());
                return false;
            }
            f.seekg(0, std::ios::end);
            const std::streamoff fsize = f.tellg();
            if (fsize < 0) {
                std::fprintf(stderr, "tellg failed: %s\n", path.c_str());
                return false;
            }
            f.seekg(0, std::ios::beg);
            const std::size_t nbytes = static_cast<std::size_t>(fsize);
            if (m_weight_scale_used + nbytes > WEIGHT_SCALE_BO_BYTES) {
                std::fprintf(stderr, "weight_scale_bo overflow at %s\n", key.c_str());
                return false;
            }
            f.read(reinterpret_cast<char*>(scale_host) +
                       static_cast<std::ptrdiff_t>(m_weight_scale_used),
                   static_cast<std::streamsize>(nbytes));
            if (!f) {
                std::fprintf(stderr, "read failed: %s\n", path.c_str());
                return false;
            }
            it->second.scale_offset = m_weight_scale_used;
            it->second.scale_bytes = nbytes;
            m_weight_scale_used += nbytes;
        }

        for (const auto& kv : m_weights) {
            if (kv.second.scale_bytes == 0) {
                std::fprintf(stderr, "data without scale: %s\n", kv.first.c_str());
                return false;
            }
        }

        weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weight_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        m_weights_loaded = true;

        std::fprintf(stderr,
                     "[FpgaDriver] loaded %zu tensors: data=%zu B scale=%zu B\n",
                     m_weights.size(),
                     m_weight_data_used,
                     m_weight_scale_used);
        return true;
    }

    const std::unordered_map<std::string, WeightSlot>& weights() const { return m_weights; }
    bool weights_loaded() const { return m_weights_loaded; }
    size_t weight_data_used_bytes() const { return m_weight_data_used; }
    size_t weight_scale_used_bytes() const { return m_weight_scale_used; }

    uint64_t weight_data_phys(const std::string& name) const {
        const WeightSlot& s = weight_slot_at(name);
        return weight_bo.address() + s.data_offset;
    }

    uint64_t weight_scale_phys(const std::string& name) const {
        const WeightSlot& s = weight_slot_at(name);
        return weight_scale_bo.address() + s.scale_offset;
    }

    const WeightSlot& weight_slot_at(const std::string& name) const {
        const auto it = m_weights.find(name);
        if (it == m_weights.end())
            throw std::out_of_range("unknown weight tensor: " + name);
        return it->second;
    }

    /// Compare BO host maps against on-disk bins (byte-for-byte).
    bool verify_weights_against_bin(const std::string& dir) {
        if (!m_weights_loaded)
            return false;
        auto* data_host = weight_bo.map<uint8_t*>();
        auto* scale_host = weight_scale_bo.map<uint16_t*>();
        std::size_t mismatches = 0;

        for (const auto& kv : m_weights) {
            const std::string& key = kv.first;
            const WeightSlot& slot = kv.second;
            const std::string data_path = dir + "/" + key + kSuffixWeightData;
            const std::string scale_path = dir + "/" + key + kSuffixWeightScale;

            std::ifstream df(data_path, std::ios::binary);
            std::ifstream sf(scale_path, std::ios::binary);
            if (!df || !sf) {
                std::fprintf(stderr, "verify: open failed for %s\n", key.c_str());
                return false;
            }
            std::vector<uint8_t> ref_data(slot.data_bytes);
            std::vector<uint8_t> ref_scale(slot.scale_bytes);
            df.read(reinterpret_cast<char*>(ref_data.data()),
                    static_cast<std::streamsize>(ref_data.size()));
            sf.read(reinterpret_cast<char*>(ref_scale.data()),
                    static_cast<std::streamsize>(ref_scale.size()));
            if (!df || !sf) {
                std::fprintf(stderr, "verify: read failed for %s\n", key.c_str());
                return false;
            }

            const uint8_t* bo_data = data_host + slot.data_offset;
            const uint8_t* bo_scale =
                reinterpret_cast<const uint8_t*>(scale_host) + slot.scale_offset;
            if (std::memcmp(bo_data, ref_data.data(), ref_data.size()) != 0) {
                std::fprintf(stderr, "verify: data mismatch %s\n", key.c_str());
                ++mismatches;
            }
            if (std::memcmp(bo_scale, ref_scale.data(), ref_scale.size()) != 0) {
                std::fprintf(stderr, "verify: scale mismatch %s\n", key.c_str());
                ++mismatches;
            }
        }
        std::fprintf(stderr,
                     "[FpgaDriver] verify %zu tensors: %s\n",
                     m_weights.size(),
                     mismatches == 0 ? "OK" : "FAIL");
        return mismatches == 0;
    }

    /// Byte-compare one loaded tensor against `{dir}/{key}.data.bin` and `.scale.bin`.
    bool verify_weight_against_bin(const std::string& dir, const std::string& key) {
        if (!m_weights_loaded)
            return false;
        const auto it = m_weights.find(key);
        if (it == m_weights.end())
            return false;
        const WeightSlot& slot = it->second;
        const std::string data_path = dir + "/" + key + kSuffixWeightData;
        const std::string scale_path = dir + "/" + key + kSuffixWeightScale;

        std::ifstream df(data_path, std::ios::binary);
        std::ifstream sf(scale_path, std::ios::binary);
        if (!df || !sf)
            return false;

        std::vector<uint8_t> ref_data(static_cast<std::size_t>(slot.data_bytes));
        std::vector<uint8_t> ref_scale(static_cast<std::size_t>(slot.scale_bytes));
        df.read(reinterpret_cast<char*>(ref_data.data()),
                static_cast<std::streamsize>(ref_data.size()));
        sf.read(reinterpret_cast<char*>(ref_scale.data()),
                static_cast<std::streamsize>(ref_scale.size()));
        if (!df || !sf)
            return false;

        auto* data_host = weight_bo.map<uint8_t*>();
        auto* scale_host = weight_scale_bo.map<uint16_t*>();
        const uint8_t* bo_data = data_host + slot.data_offset;
        const uint8_t* bo_scale =
            reinterpret_cast<const uint8_t*>(scale_host) + slot.scale_offset;
        return std::memcmp(bo_data, ref_data.data(), ref_data.size()) == 0 &&
               std::memcmp(bo_scale, ref_scale.data(), ref_scale.size()) == 0;
    }

    void start() {
        uint32_t v = rd(ADDR_AP_CTRL) & BIT_AUTO_RESTART;
        wr(ADDR_AP_CTRL, v | BIT_AP_START);
    }

    bool isDone()  const { return (rd(ADDR_AP_CTRL) >> 1) & 1; }
    bool isIdle()  const { return (rd(ADDR_AP_CTRL) >> 2) & 1; }
    bool isReady() const { return !(rd(ADDR_AP_CTRL) & BIT_AP_START); }

    void enableAutoRestart()  { wr(ADDR_AP_CTRL, BIT_AUTO_RESTART); }
    void disableAutoRestart() { wr(ADDR_AP_CTRL, 0); }

    void interruptGlobalEnable()  { wr(ADDR_GIE, 1); }
    void interruptGlobalDisable() { wr(ADDR_GIE, 0); }
    void interruptEnable(uint32_t mask)  { wr(ADDR_IER, rd(ADDR_IER) | mask); }
    void interruptDisable(uint32_t mask) { wr(ADDR_IER, rd(ADDR_IER) & ~mask); }
    void interruptClear(uint32_t mask)   { wr(ADDR_ISR, mask); }
    uint32_t interruptGetEnabled() const { return rd(ADDR_IER); }
    uint32_t interruptGetStatus()  const { return rd(ADDR_ISR); }

    void    set_a(uint64_t v)  { wr64(ADDR_A_DATA, v); }
    uint64_t get_a() const     { return rd64(ADDR_A_DATA); }
    void    set_b(uint64_t v)  { wr64(ADDR_B_DATA, v); }
    uint64_t get_b() const     { return rd64(ADDR_B_DATA); }
    void    set_as(uint64_t v) { wr64(ADDR_AS_DATA, v); }
    uint64_t get_as() const    { return rd64(ADDR_AS_DATA); }
    void    set_bs(uint64_t v) { wr64(ADDR_BS_DATA, v); }
    uint64_t get_bs() const    { return rd64(ADDR_BS_DATA); }
    void    set_c(uint64_t v)  { wr64(ADDR_C_DATA, v); }
    uint64_t get_c() const     { return rd64(ADDR_C_DATA); }

    void     set_op(uint32_t v) { wr(ADDR_OP_DATA, v); }
    uint32_t get_op() const     { return rd(ADDR_OP_DATA); }
    void     set_mode(uint32_t v) { set_op(v); }
    uint32_t get_mode() const     { return get_op(); }

    void set_use_m_reg(int use_m) { wr(ADDR_USE_M_DATA, static_cast<uint32_t>(use_m)); }
    void set_use_n_reg(int use_n) { wr(ADDR_USE_N_DATA, static_cast<uint32_t>(use_n)); }
    uint32_t get_use_m_reg() const { return rd(ADDR_USE_M_DATA); }
    uint32_t get_use_n_reg() const { return rd(ADDR_USE_N_DATA); }

    void     gemv_set_a_in(uint64_t v)   { wr64(GEMV_ADDR_A_IN_DATA, v); }
    uint64_t gemv_get_a_in() const       { return rd64(GEMV_ADDR_A_IN_DATA); }
    void     gemv_set_b_in1(uint64_t v)  { wr64(GEMV_ADDR_B_IN1_DATA, v); }
    uint64_t gemv_get_b_in1() const      { return rd64(GEMV_ADDR_B_IN1_DATA); }
    void     gemv_set_b_in2(uint64_t v)  { wr64(GEMV_ADDR_B_IN2_DATA, v); }
    uint64_t gemv_get_b_in2() const      { return rd64(GEMV_ADDR_B_IN2_DATA); }
    void     gemv_set_b_in3(uint64_t v)  { wr64(GEMV_ADDR_B_IN3_DATA, v); }
    uint64_t gemv_get_b_in3() const      { return rd64(GEMV_ADDR_B_IN3_DATA); }
    void     gemv_set_b_in4(uint64_t v)  { wr64(GEMV_ADDR_B_IN4_DATA, v); }
    uint64_t gemv_get_b_in4() const      { return rd64(GEMV_ADDR_B_IN4_DATA); }
    void     gemv_set_scales1(uint64_t v) { wr64(GEMV_ADDR_SCALES1_DATA, v); }
    uint64_t gemv_get_scales1() const    { return rd64(GEMV_ADDR_SCALES1_DATA); }
    void     gemv_set_scales2(uint64_t v) { wr64(GEMV_ADDR_SCALES2_DATA, v); }
    uint64_t gemv_get_scales2() const    { return rd64(GEMV_ADDR_SCALES2_DATA); }
    void     gemv_set_scales3(uint64_t v) { wr64(GEMV_ADDR_SCALES3_DATA, v); }
    uint64_t gemv_get_scales3() const    { return rd64(GEMV_ADDR_SCALES3_DATA); }
    void     gemv_set_scales4(uint64_t v) { wr64(GEMV_ADDR_SCALES4_DATA, v); }
    uint64_t gemv_get_scales4() const    { return rd64(GEMV_ADDR_SCALES4_DATA); }
    void     gemv_set_c_out(uint64_t v)  { wr64(GEMV_ADDR_C_OUT_DATA, v); }
    uint64_t gemv_get_c_out() const      { return rd64(GEMV_ADDR_C_OUT_DATA); }

    void     gemv_set_op(uint32_t v) { wr(GEMV_ADDR_OP_DATA, v); }
    uint32_t gemv_get_op() const     { return rd(GEMV_ADDR_OP_DATA); }
    void     gemv_set_mode(uint32_t v) { gemv_set_op(v); }
    uint32_t gemv_get_mode() const     { return gemv_get_op(); }

    void     gemv_set_k_block_num(uint32_t v) { wr(GEMV_ADDR_K_BLOCK_NUM_DATA, v); }
    uint32_t gemv_get_k_block_num() const     { return rd(GEMV_ADDR_K_BLOCK_NUM_DATA); }
    void     gemv_set_n_block_num(uint32_t v) { wr(GEMV_ADDR_N_BLOCK_NUM_DATA, v); }
    uint32_t gemv_get_n_block_num() const     { return rd(GEMV_ADDR_N_BLOCK_NUM_DATA); }

    // Legacy single-bank names (alias b_in1 / scales1).
    void     gemv_set_b_in(uint64_t v)  { gemv_set_b_in1(v); }
    uint64_t gemv_get_b_in() const      { return gemv_get_b_in1(); }
    void     gemv_set_scales(uint64_t v) { gemv_set_scales1(v); }
    uint64_t gemv_get_scales() const    { return gemv_get_scales1(); }

    void set_c_store_region(int c_offset, int c_stride) {
        wr(ADDR_C_OFFSET_DATA, static_cast<uint32_t>(c_offset));
        wr(ADDR_C_STRIDE_DATA, static_cast<uint32_t>(c_stride));
    }
    uint32_t get_c_offset_reg() const { return rd(ADDR_C_OFFSET_DATA); }
    uint32_t get_c_stride_reg() const { return rd(ADDR_C_STRIDE_DATA); }

    void gemv_set(
        uint64_t a_addr, 
        uint64_t b_addr, 
        uint64_t scales_addr, 
        uint64_t a_scales_addr, 
        uint64_t c_out_addr,
        int k_block_num,
        int n_block_num
    ) {
        // must set k_block_num and n_block_num first
        gemv_set_k_block_num(k_block_num);
        gemv_set_n_block_num(n_block_num);
        // rest comes afterwards
        gemv_set_a_in(a_addr);
        gemv_set_mode(static_cast<uint32_t>(GemvOp::LOAD_A));
        start();
        while (!isDone()) {
        }
        gemv_set_scales1(scales_addr);
        uint64_t scales2_addr = scales_addr + 4096 * sizeof(char);
        gemv_set_scales2(scales2_addr);
        uint64_t scales3_addr = scales_addr + 8192 * sizeof(char);
        gemv_set_scales3(scales3_addr);
        uint64_t scales4_addr = scales_addr + 12288 * sizeof(char);
        gemv_set_scales4(scales4_addr);
        gemv_set_mode(static_cast<uint32_t>(GemvOp::LOAD_B_SCALE));
        start();
        while (!isDone()) {
        }
        gemv_set_scales1(a_scales_addr);
        gemv_set_mode(static_cast<uint32_t>(GemvOp::LOAD_A_SCALE));
        start();
        while (!isDone()) {
        }
        // related addresses
        gemv_set_b_in1(b_addr);
        uint64_t b_in2_addr = b_addr + 4096 * sizeof(char);
        gemv_set_b_in2(b_in2_addr);
        uint64_t b_in3_addr = b_addr + 8192 * sizeof(char);
        gemv_set_b_in3(b_in3_addr);
        uint64_t b_in4_addr = b_addr + 12288 * sizeof(char);
        gemv_set_b_in4(b_in4_addr);
        gemv_set_c_out(c_out_addr);
    }

    void gemv_run() {
        gemv_set_mode(static_cast<uint32_t>(GemvOp::COMPUTE));
        start();
        while (!isDone()) {
        }
        gemv_set_mode(static_cast<uint32_t>(GemvOp::STORE_C));
        start();
        while (!isDone()) {
        }
    }

    void gemm_set(uint64_t a_addr,
                  uint64_t as_addr,
                  uint64_t b_addr,
                  uint64_t bs_addr,
                  int use_m,
                  int use_n) {
        set_use_m_reg(use_m);
        set_use_n_reg(use_n);

        set_a(a_addr);
        set_mode(static_cast<uint32_t>(Op::OP_PRELOAD_A));
        start();
        while (!isDone()) {
            usleep(1);
        }
        set_as(as_addr);
        set_mode(static_cast<uint32_t>(Op::OP_LOAD_AS));
        start();
        while (!isDone()) {
            usleep(1);
        }
        set_b(b_addr);
        set_mode(static_cast<uint32_t>(Op::OP_PRELOAD_B));
        start();
        while (!isDone()) {
            usleep(1);
        }
        set_bs(bs_addr);
        set_mode(static_cast<uint32_t>(Op::OP_LOAD_BS));
        start();
        while (!isDone()) {
            usleep(1);
        }
    }

    void gemm_init_c_global() {
        set_mode(static_cast<uint32_t>(Op::OP_INIT_C_GLOBAL));
        start();
        while (!isDone()) {
            usleep(1);
        }
    }

    void gemm_run() {
        set_mode(static_cast<uint32_t>(Op::OP_COMPUTE));
        start();
        while (!isDone()) {
            usleep(1);
        }
    }

    void gemm_load(uint64_t c_addr, int c_offset, int c_stride) {
        set_c(c_addr);
        set_c_store_region(c_offset, c_stride);
        set_mode(static_cast<uint32_t>(Op::OP_STORE_C));
        start();
        while (!isDone()) {
            usleep(10);
        }
    }

    struct block_q8_0 {
        uint16_t d;
        int8_t qs[QK8_0];
    };

    struct WeightKind {
        bool is_vit_qkvo;
        bool is_vit_ffn_up;
        bool is_vit_ffn_down;
        bool is_lm_qo;
        bool is_lm_kv;
        bool is_lm_ffn_down;
        bool is_lm_ffn_up;
        bool is_lm_ffn_gate;
        bool is_lm_output;
    };

    WeightEntry weight_entry_at(
        WeightSlot weight_slot,
        WeightKind weight_kind,
        int idm, 
        int idk
    ) {
        WeightEntry weight_entry;
        if (weight_kind.is_vit_qkvo) {
            // The vit_qkvo weight has 6 tiles
            // 3x [512, 256] -> first block row
            // 3x [256, 256] -> second block row
            assert(idm < 2);
            assert(idk < 3);
            if (idm == 0) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + idk * 512 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + idk * (512/32) * 256 * sizeof(uint16_t);
                weight_entry.n_count = 512;
                return weight_entry;
            } 
            if (idm == 1) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + 512 * 256 * 3 * sizeof(int8_t)
                    + idk * 256 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + (512/32) * 256 * 3 * sizeof(uint16_t)
                    + idk * (256/32) * 256 * sizeof(uint16_t);
                weight_entry.n_count = 256;
                return weight_entry;
            }
        }
        if (weight_kind.is_vit_ffn_up) {
            // The vit_ffn_up weight has 18 tiles
            // 6x 3x [512, 256] -> first to sixth block row
            assert(idm < 6);
            assert(idk < 3);
            weight_entry.data_start_addr = weight_slot.data_offset 
                + idm * 512 * 256 * 3 * sizeof(int8_t)
                + idk * 512 * 256 * sizeof(int8_t);
            weight_entry.scale_start_addr = weight_slot.scale_offset 
                + idm * (512/32) * 256 * 3 * sizeof(uint16_t)
                + idk * (512/32) * 256 * sizeof(uint16_t);
            weight_entry.n_count = 512;
            return weight_entry;
        }
        if (weight_kind.is_vit_ffn_down) {
            // The vit_ffn_down weight has 24 tiles
            // 12x [512, 256] -> first block row
            // 12x [256, 256] -> second block row
            assert(idm < 2);
            assert(idk < 12);
            if (idm == 0) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + idk * 512 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + idk * (512/32) * 256 * sizeof(uint16_t);
                weight_entry.n_count = 512;
                return weight_entry;
            }
            if (idm == 1) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + 512 * 256 * 12 * sizeof(int8_t)
                    + idk * 256 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset
                    + 512 * (256/32) * 12 * sizeof(uint16_t)
                    + idk * (256/32) * 256 * sizeof(uint16_t);
                weight_entry.n_count = 256;
                return weight_entry;
            }
        }
        if (weight_kind.is_lm_qo) {
            // The lm_qo weight has 8 tiles
            // 4x [512, 256] -> first block row
            // 4x [448, 256] -> second block row
            assert(idm < 2);
            assert(idk < 4);
            if (idm == 0) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + idk * 512 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + idk * (512/32) * 256 * sizeof(uint16_t);
                weight_entry.n_count = 512;
                return weight_entry;
            }
            if (idm == 1) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + 512 * 256 * 4 * sizeof(int8_t)
                    + idk * 448 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + 512 * (256/32) * 4 * sizeof(uint16_t)
                    + idk * (512/32) * 256 * sizeof(uint16_t); /* scale is aligned to 512 now*/
                weight_entry.n_count = 448;
                return weight_entry;
            }
        }
        if (weight_kind.is_lm_kv) {
            // The lm_kv weight has 4 tiles
            // 4x [320, 256] -> first block row
            assert(idm < 1);
            assert(idk < 4);
            weight_entry.data_start_addr = weight_slot.data_offset 
                + idk * 320 * 256 * sizeof(int8_t);
            weight_entry.scale_start_addr = weight_slot.scale_offset 
                + idk * (512/32) * 256 * sizeof(uint16_t); /* scale is aligned to 512 now*/
            weight_entry.n_count = 320;
            return weight_entry;
        }
        if (weight_kind.is_lm_ffn_up || weight_kind.is_lm_ffn_gate) {
            // The lm_ffn_up and lm_ffn_gate weight have 20 tiles
            // 5x 4x [512, 256] -> first to fifth block row
            assert(idm < 5);
            assert(idk < 4);
            weight_entry.data_start_addr = weight_slot.data_offset 
                + idm * 512 * 256 * 4 * sizeof(int8_t)
                + idk * 512 * 256 * sizeof(int8_t);
            weight_entry.scale_start_addr = weight_slot.scale_offset 
                + idm * (512/32) * 256 * 4 * sizeof(uint16_t)
                + idk * (512/32) * 256 * sizeof(uint16_t);
            weight_entry.n_count = 512;
            return weight_entry;
        }
        if (weight_kind.is_lm_ffn_down) {
            // The lm_ffn_down weight has 20 tiles
            // 10x [512, 256] -> first block row
            // 10x [448, 256] -> second block row
            assert(idm < 2);
            assert(idk < 10);
            if (idm == 0) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + idk * 512 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + idk * (512/32) * 256 * sizeof(uint16_t);
                weight_entry.n_count = 512;
                return weight_entry;
            }
            if (idm == 1) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + 512 * 256 * 10 * sizeof(int8_t)
                    + idk * 448 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + 512 * (256/32) * 10 * sizeof(uint16_t)
                    + idk * (512/32) * 256 * sizeof(uint16_t); /* scale is aligned to 512 now*/
                weight_entry.n_count = 448;
                return weight_entry;
            }
        }
        if (weight_kind.is_lm_output) {
            // The lm_output weight has 97 tiles
            // 97x [4, 512] -> first block row
            assert(idk < 4);
            assert(idm < 97);
            if (idm < 96) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + idk * 512 * 256 * sizeof(int8_t);
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + idk * (512/32) * 256 * sizeof(uint16_t);
                weight_entry.n_count = 512;
                return weight_entry;
            }
            if (idm == 96) {
                weight_entry.data_start_addr = weight_slot.data_offset 
                    + 512 * 256 * 96 * sizeof(int8_t)
                    + idk * 512 * 256 * sizeof(int8_t); /* data is aligned to 512 now*/
                weight_entry.scale_start_addr = weight_slot.scale_offset 
                    + 512 * (256/32) * 96 * sizeof(uint16_t)
                    + idk * 512 * (256/32) * sizeof(uint16_t); /* scale is aligned to 512 now*/
                weight_entry.n_count = 128;
                return weight_entry;
            }
        }
        throw std::runtime_error("Unsupported weight kind");
    }

    struct WeightDim {
        int n_iter; 
        int k_iter;
        int n_num;
    };

    WeightDim weight_dim_at(
        WeightKind weight_kind
    ) {
        if (weight_kind.is_vit_qkvo) {
            return WeightDim{2, 3, 768};
        }
        if (weight_kind.is_vit_ffn_up) {
            return WeightDim{6, 3, 3072};
        }
        if (weight_kind.is_vit_ffn_down) {
            return WeightDim{2, 12, 768};
        }
        if (weight_kind.is_lm_qo) {
            return WeightDim{2, 4, 960};
        }
        if (weight_kind.is_lm_kv) {
            return WeightDim{1, 4, 320};
        }
        if (weight_kind.is_lm_ffn_up || weight_kind.is_lm_ffn_gate) {
            return WeightDim{5, 4, 2560};
        }
        if (weight_kind.is_lm_ffn_down) {
            return WeightDim{2, 10, 960};
        }
        if (weight_kind.is_lm_output) {
            return WeightDim{97, 4, 960};
        }
        throw std::runtime_error("Unsupported weight kind");
    }

    /// SmolVLM mmproj GGUF swaps `ffn_up` / `ffn_down` names vs logical matmul (see llama-server
    /// "ffn up/down are swapped"). Bins are tiled by shape (K_in=ne0, N_out=ne1): the tensor
    /// named ffn_up.weight is 3072->768 (down), ffn_down.weight is 768->3072 (up). Map GGUF
    /// name to the FPGA schedule that matches that shape.
    static void classify_vit_ffn_kind(const std::string& weight_name,
                                      uint64_t data_bytes,
                                      bool& is_vit_ffn_up,
                                      bool& is_vit_ffn_down) {
        is_vit_ffn_up = false;
        is_vit_ffn_down = false;
        if (data_bytes != 3072u * 768u * sizeof(int8_t))
            return;
        if (weight_name.find("v.blk.") == std::string::npos)
            return;
        const bool name_has_up = weight_name.find("ffn_up") != std::string::npos;
        const bool name_has_down = weight_name.find("ffn_down") != std::string::npos;
        if (!name_has_up && !name_has_down)
            return;
        is_vit_ffn_up = name_has_down;
        is_vit_ffn_down = name_has_up;
    }

    /// Classify LM tensors (blk.L.{attn_q,attn_k,attn_v,attn_output,ffn_*}.weight).
    /// Keys match load_weights_from_bin: filename minus .data.bin / .scale.bin.
    static void classify_lm_kind(const std::string& weight_name,
                                 uint64_t data_bytes,
                                 bool& is_lm_qo,
                                 bool& is_lm_kv,
                                 bool& is_lm_ffn_up,
                                 bool& is_lm_ffn_gate,
                                 bool& is_lm_ffn_down) {
        is_lm_qo = false;
        is_lm_kv = false;
        is_lm_ffn_up = false;
        is_lm_ffn_gate = false;
        is_lm_ffn_down = false;
        if (weight_name.find("blk.") == std::string::npos)
            return;

        if (data_bytes == kTiledLmQoDataBytes) {
            if (weight_name.find("attn_q.weight") != std::string::npos ||
                weight_name.find("attn_output.weight") != std::string::npos) {
                is_lm_qo = true;
            }
            return;
        }
        if (data_bytes == kTiledLmKvDataBytes) {
            if (weight_name.find("attn_k.weight") != std::string::npos ||
                weight_name.find("attn_v.weight") != std::string::npos) {
                is_lm_kv = true;
            }
            return;
        }
        if (data_bytes == kTiledLmFfnUpGateDataBytes) {
            if (weight_name.find("ffn_up.weight") != std::string::npos)
                is_lm_ffn_up = true;
            else if (weight_name.find("ffn_gate.weight") != std::string::npos)
                is_lm_ffn_gate = true;
            return;
        }
        if (data_bytes == kTiledLmFfnDownDataBytes &&
            weight_name.find("ffn_down.weight") != std::string::npos) {
            is_lm_ffn_down = true;
        }
    }


    // api to compute the gemm between the prepared activation and weight
    void gemm_compute(
        std::string weight_name
    ) {
        WeightSlot weight_slot = weight_slot_at(weight_name);
        // hard-coded logic for SmolVLM
        bool is_vit_qkvo = weight_slot.data_bytes == 768 * 768 * sizeof(int8_t);
        bool is_vit_ffn_up = false;
        bool is_vit_ffn_down = false;
        classify_vit_ffn_kind(weight_name, weight_slot.data_bytes, is_vit_ffn_up, is_vit_ffn_down);
        bool is_lm_qo = false;
        bool is_lm_kv = false;
        bool is_lm_ffn_up = false;
        bool is_lm_ffn_gate = false;
        bool is_lm_ffn_down = false;
        classify_lm_kind(weight_name,
                         weight_slot.data_bytes,
                         is_lm_qo,
                         is_lm_kv,
                         is_lm_ffn_up,
                         is_lm_ffn_gate,
                         is_lm_ffn_down);
        bool is_lm_output = weight_slot.data_bytes == 97 * 4 * 512 * 256 * sizeof(int8_t);
        
        if (!is_vit_qkvo && !is_vit_ffn_up && !is_vit_ffn_down && !is_lm_qo && !is_lm_kv && !is_lm_ffn_down && !is_lm_ffn_up && !is_lm_ffn_gate) {
            throw std::runtime_error("Unsupported weight: " + weight_name);
        }

        WeightKind weight_kind = {
            is_vit_qkvo,
            is_vit_ffn_up,
            is_vit_ffn_down,
            is_lm_qo,
            is_lm_kv,
            is_lm_ffn_down,
            is_lm_ffn_up,
            is_lm_ffn_gate,
            is_lm_output
        };

        WeightDim weight_dim = weight_dim_at(weight_kind);
        int m_iter = (int) (m_act_slots.size() / weight_dim.k_iter);
        int k_iter = weight_dim.k_iter;
        int n_iter = weight_dim.n_iter;

        // The C is 512 bit m_axi
        int stride = weight_dim.n_num * sizeof(float) / 64;
        // if the M>256 (heuristic), use cacheable
        // else use normal
        uint64_t c_start_addr = 0;
        // get the maximal act_slot.m_count
        int max_m_count = 0;
        for (int i = 0; i < m_act_slots.size(); i++) {
            max_m_count = std::max(max_m_count, m_act_slots[i].m_count);
        }
        if (max_m_count > 256) {
            c_start_addr = act_result_bo.address();
        } else {
            c_start_addr = act_result_decode_bo.address();
        }
        // test
        for (int idn = 0; idn < n_iter; idn++) {
            // base starting address for i
            for (int idm = 0; idm < m_iter; idm++) {
                gemm_init_c_global();
                int offset = 0;
                for (int idk = 0; idk < k_iter; idk++) {
                    WeightEntry weight_entry = weight_entry_at(
                        weight_slot, weight_kind, idn, idk
                    );
                    ActSlot act_slot = m_act_slots[
                        idm * k_iter + idk
                    ];
                    // set use_m and use_n
                    if (idk == 0) {
                        // we can safely assume that the offset on m dim is multiple of 512
                        offset = idm * stride * 512 + 
                            idn * 512 * sizeof(float) / 64;
                    }
                    // Note that due to legacy issue, 
                    // The FPGA's m/n are swapped
                    // Get the phys address
                    uint64_t weight_start_addr = weight_bo.address() + weight_entry.data_start_addr;
                    uint64_t weight_scale_start_addr = weight_scale_bo.address() + weight_entry.scale_start_addr;
                    {
                        const auto t_set0 = FpgaGemmInnerProf::Clock::now();
                        gemm_set(
                            weight_start_addr,
                            weight_scale_start_addr,
                            act_slot.data_start_addr,
                            act_slot.scale_start_addr,
                            weight_entry.n_count,
                            act_slot.m_count
                        );
                        FpgaGemmInnerProf::record_gemm_set(
                            FpgaGemmInnerProf::ms(FpgaGemmInnerProf::Clock::now() - t_set0));
                    }
                    {
                        const auto t_run0 = FpgaGemmInnerProf::Clock::now();
                        gemm_run();
                        FpgaGemmInnerProf::record_gemm_run(
                            FpgaGemmInnerProf::ms(FpgaGemmInnerProf::Clock::now() - t_run0));
                    }
                }
                {
                    const auto t_load0 = FpgaGemmInnerProf::Clock::now();
                    gemm_load(
                        c_start_addr,
                        offset,
                        stride
                    );
                    FpgaGemmInnerProf::record_gemm_load(
                        FpgaGemmInnerProf::ms(FpgaGemmInnerProf::Clock::now() - t_load0));
                }
            }
        }
        if (m_iter > 256) {
            act_result_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        } else {
            act_result_decode_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }
    }

    void gemm_copy_to_host(void * dst, size_t m_cnt, size_t n_cnt) {
        constexpr size_t CPY_LEN_BYTES = 256 * 1024; // 256 KB (KV260 L2 Cache 黄金尺寸)
        const size_t tot_len_bytes = static_cast<size_t>(m_cnt) * static_cast<size_t>(n_cnt) * sizeof(float);
        
        char* dst_bytes = reinterpret_cast<char*>(dst);
        char* src_bytes = (m_cnt > 256) ? act_result_bo.map<char*>() : act_result_decode_bo.map<char*>();

        // 安全检查：确保指针和长度在 KV260 上是 16 字节对齐的，触发 NEON 向量加速
        assert(reinterpret_cast<uintptr_t>(dst_bytes) % 16 == 0 && "dst is not 16-byte aligned!");
        assert(reinterpret_cast<uintptr_t>(src_bytes) % 16 == 0 && "src is not 16-byte aligned!");

        size_t bytes_copied = 0;
        while (bytes_copied < tot_len_bytes) {
            size_t chunk_size = std::min(CPY_LEN_BYTES, tot_len_bytes - bytes_copied);
            
            std::memcpy(dst_bytes + bytes_copied, src_bytes + bytes_copied, chunk_size);
            
            bytes_copied += chunk_size;
        }
    }

    void gemv_copy_to_host(void * dst, size_t n_cnt) {
        char* dst_bytes = reinterpret_cast<char*>(dst);
        char* src_bytes = act_result_decode_bo.map<char*>(); 
        std::memcpy(dst_bytes, src_bytes, n_cnt * sizeof(float));
    }

    void gemv_compute(
        std::string weight_name
    ) {

        WeightSlot weight_slot = weight_slot_at(weight_name);

        bool is_lm_qo = false;
        bool is_lm_kv = false;
        bool is_lm_ffn_up = false;
        bool is_lm_ffn_gate = false;
        bool is_lm_ffn_down = false;
        classify_lm_kind(weight_name,
                         weight_slot.data_bytes,
                         is_lm_qo,
                         is_lm_kv,
                         is_lm_ffn_up,
                         is_lm_ffn_gate,
                         is_lm_ffn_down);
        bool is_lm_output = weight_slot.data_bytes == 97 * 4 * 512 * 256 * sizeof(int8_t);
        // the decode stage cannot be other layers
        if (!is_lm_qo && !is_lm_kv && !is_lm_ffn_up && !is_lm_ffn_gate && !is_lm_ffn_down && !is_lm_output) {
            throw std::runtime_error("Unsupported weight: " + weight_name);
        }

        // setting the activation and meta parameters
        if (is_lm_qo) {
            gemv_set_n_block_num(8); // 512
            gemv_set_k_block_num(4); // 960 (1024)
        } else if (is_lm_kv) {
            gemv_set_n_block_num(5); // 320
            gemv_set_k_block_num(4); // 960 (1024)
        } else if (is_lm_ffn_up || is_lm_ffn_gate) {
            gemv_set_n_block_num(8); // 512
            gemv_set_k_block_num(4); // 960 (1024)
        } else if (is_lm_ffn_down) {
            gemv_set_n_block_num(8); // 512
            gemv_set_k_block_num(10); // 2560
        } else if (is_lm_output) {
            gemv_set_n_block_num(8); // 512
            gemv_set_k_block_num(4); // 960 (1024)
        }

        // send in a data
        gemv_set_a_in(act_decode_bo.address());
        gemv_set_mode(static_cast<uint32_t>(GemvOp::LOAD_A));
        start();
        while (!isDone()) {}

        // send in a scale
        gemv_set_scales1(act_scale_decode_bo.address());
        gemv_set_mode(static_cast<uint32_t>(GemvOp::LOAD_A_SCALE));
        start();
        while (!isDone()) {}

        // main loop for the gemv computation
        WeightKind weight_kind = {
            false,
            false,
            false,
            is_lm_qo,
            is_lm_kv,
            is_lm_ffn_down,
            is_lm_ffn_up,
            is_lm_ffn_gate,
            is_lm_output
        };

        WeightDim weight_dim = weight_dim_at(weight_kind);
        const int n_iter = weight_dim.n_iter;
        uint64_t c_start_addr = act_result_decode_bo.address();

        for (int idn = 0; idn < n_iter; idn++) {
            WeightEntry weight_entry = weight_entry_at(
                weight_slot, weight_kind, idn, 0
            );

            const uint64_t weight_start_addr = weight_bo.address() + weight_entry.data_start_addr;
            const uint64_t weight_scale_start_addr = weight_scale_bo.address() + weight_entry.scale_start_addr;

            if (idn != 0) {
                gemv_set_n_block_num(weight_entry.n_count / 64);
            }
            gemv_set_scales1(weight_scale_start_addr);
            const uint64_t scales2_addr = weight_scale_start_addr + 4096 * sizeof(char);
            gemv_set_scales2(scales2_addr);
            const uint64_t scales3_addr = weight_scale_start_addr + 8192 * sizeof(char);
            gemv_set_scales3(scales3_addr);
            const uint64_t scales4_addr = weight_scale_start_addr + 12288 * sizeof(char);
            gemv_set_scales4(scales4_addr);
            gemv_set_mode(static_cast<uint32_t>(GemvOp::LOAD_B_SCALE));
            start();
            while (!isDone()) {}

            gemv_set_b_in1(weight_start_addr);
            const uint64_t b_in2_addr = weight_start_addr + 4096 * sizeof(char);
            gemv_set_b_in2(b_in2_addr);
            const uint64_t b_in3_addr = weight_start_addr + 8192 * sizeof(char);
            gemv_set_b_in3(b_in3_addr);
            const uint64_t b_in4_addr = weight_start_addr + 12288 * sizeof(char);
            gemv_set_b_in4(b_in4_addr);
            gemv_set_c_out(c_start_addr + idn * 512 * sizeof(float));
            gemv_run();
        }
        act_result_decode_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    struct WeightPack {
        std::vector<int8_t> data;
        std::vector<uint16_t> scale;
    };

    void act_vector_prepare(
        const float* act_tensor_data,
        int K
    ) {
        auto q_tensor_data = act_decode_bo.map<int8_t*>();
        auto act_scale_ptr = act_scale_decode_bo.map<uint16_t*>();
        
        int K_ALIGNED = ((K + 511) / 512) * 512;
        int total_blocks = (K + 31) / 32; // 向上取整计算实际需要处理的 block 数
        int aligned_blocks = K_ALIGNED / 32;

        std::vector<float> scale_data(aligned_blocks);

        // 设置 OpenMP 4 线程并行化
        #pragma omp parallel num_threads(4)
        {
            // 1. 并行处理所有的 32-element blocks
            #pragma omp for schedule(static)
            for (int b = 0; b < total_blocks; b++) {
                int i = b * 32;
                
                // 处理特殊边缘情况：如果最后一个 block 不足 32 个元素，安全读取
                if (i + 32 > K) {
                    float amax = 0.0f;
                    for (int j = i; j < K; j++) {
                        amax = std::max(amax, std::abs(act_tensor_data[j]));
                    }
                    float d = amax / 127.0f;
                    float id = (d == 0.0f) ? 0.0f : 1.0f / d;
                    scale_data[b] = d;

                    for (int j = i; j < K; j++) {
                        float quantized = std::round(act_tensor_data[j] * id);
                        q_tensor_data[j] = static_cast<int8_t>(std::clamp(quantized, -127.0f, 127.0f));
                    }
                    continue;
                }

                // --- 标准 32 元素块：使用 NEON 展开与向量化 ---
                // 步骤 1: 寻找 32 个元素中的绝对值最大值
                float32x4_t v_max0 = vdupq_n_f32(0.0f);
                float32x4_t v_max1 = vdupq_n_f32(0.0f);

                // 每次循环处理 8 个 float，分 4 次处理完 32 个
                for (int j = 0; j < 32; j += 8) {
                    float32x4_t v_data0 = vld1q_f32(&act_tensor_data[i + j]);
                    float32x4_t v_data1 = vld1q_f32(&act_tensor_data[i + j + 4]);

                    // 取绝对值
                    float32x4_t v_abs0 = vabsq_f32(v_data0);
                    float32x4_t v_abs1 = vabsq_f32(v_data1);

                    // 累积最大值
                    v_max0 = vmaxq_f32(v_max0, v_abs0);
                    v_max1 = vmaxq_f32(v_max1, v_abs1);
                }

                // 汇总 8 个通道的最大值
                float32x4_t v_max = vmaxq_f32(v_max0, v_max1);
                // 树状两两比较汇总到标量
                float amax = std::max({
                    vgetq_lane_f32(v_max, 0), vgetq_lane_f32(v_max, 1),
                    vgetq_lane_f32(v_max, 2), vgetq_lane_f32(v_max, 3)
                });

                // 步骤 2: 计算缩放因子
                float d = amax / 127.0f;
                float id = (d == 0.0f) ? 0.0f : 1.0f / d;
                scale_data[b] = d;

                // 步骤 3: 使用 NEON 进行量化
                float32x4_t v_id = vdupq_n_f32(id);
                
                for (int j = 0; j < 32; j += 4) {
                    float32x4_t v_data = vld1q_f32(&act_tensor_data[i + j]);
                    // 乘上 1/d
                    float32x4_t v_scaled = vmulq_f32(v_data, v_id);

                    // 四舍五入到最近整数 (Cortex-A53 支持 ARMv8-A，vrndnq_f32 是硬件指令)
                    float32x4_t v_rounded = vrndnq_f32(v_scaled);

                    // 将 FP32 转换为 INT32 (带饱和保护)
                    int32x4_t v_int32 = vcvtq_s32_f32(v_rounded);

                    // 从 INT32 窄化到 INT16，再窄化到 INT8 (带饱和自动截断到 [-128, 127])
                    int16x4_t v_int16 = vqmovn_s32(v_int32);
                    
                    // 为了配合 vqmovn_s16 的输入，我们需要一个 8 元素的 16 位向量
                    // 这里用两个 4 元素拼一下，或者直接写入标量。为了写内存方便，转为标量写入：
                    alignas(16) int16_t tmp[4];
                    vst1_s16(tmp, v_int16);

                    for (int k = 0; k < 4; k++) {
                        // 裁剪到 [-127, 127]
                        int32_t val = tmp[k];
                        if (val > 127) val = 127;
                        if (val < -127) val = -127;
                        q_tensor_data[i + j + k] = static_cast<int8_t>(val);
                    }
                }
            }
        } // OpenMP 隐式同步结束

        // 2. 主线程：补全对齐所需的零和默认 scale
        for (int i = K; i < K_ALIGNED; i++) {
            q_tensor_data[i] = (int8_t)0;
        }
        for (int i = total_blocks; i < aligned_blocks; i++) {
            scale_data[i] = 1.0f;
        }

        // 3. 主线程：批量转换并拷贝 scale 到硬件 BO
        // (如果 float_to_half 能支持 NEON 批量转则更快，这里保持原本的逐个转换)
        for (int i = 0; i < aligned_blocks; i++) {
            act_scale_ptr[i] = float_to_half(scale_data[i]);
        }

        // 同步到设备
        act_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        act_scale_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    #if 0
    void act_vector_prepare(
        const float* act_tensor_data,
        int K
    ){
        // this one is ez! 
        auto q_tensor_data = act_bo.map<int8_t*>();
        auto act_scale_ptr = act_scale_bo.map<uint16_t*>();
        int K_ALIGNED = ((K + 511) / 512) * 512;
        // dequantize the act_tensor_data
        std::vector<float> scale_data(K_ALIGNED / 32);
        for (int i = 0; i < K; i += 32) {
            float amax = 0.0f;
            
            // 1. 寻找当前 32 个元素中的绝对值最大值
            for (int j = 0; j < 32; j++) {
                amax = std::max(amax, std::abs(act_tensor_data[i + j]));
            }
            
            // 2. 计算缩放因子 scale (d) 和 它的倒数 (id)
            float d = amax / 127.0f;
            float id = (d == 0.0f) ? 0.0f : 1.0f / d;
            
            // 存储当前 block 的 scale
            scale_data[i / 32] = d; 
            
            // 3. 补全量化部分：将 float 映射到 int8_t (-127 到 127)
            for (int j = 0; j < 32; j++) {
                // 使用 std::round 进行四舍五入，并使用 std::clamp 防止溢出
                float quantized = std::round(act_tensor_data[i + j] * id);
                q_tensor_data[i + j] = static_cast<int8_t>(std::clamp(quantized, -127.0f, 127.0f));
            }
        }
        // add some zeros to the q_tensor_data to align it
        for (int i = K; i < K_ALIGNED; i++) {
            q_tensor_data[i] = (int8_t) 0;
        }
        for (int i = K / 32; i < K_ALIGNED / 32; i++) {
            scale_data[i] = 1.0f;
        }
        // copy the scale_data to the act_scale_ptr
        for (int i = 0; i < K_ALIGNED / 32; i++) {
            act_scale_ptr[i] = float_to_half(scale_data[i]);
        }
        act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        act_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    #endif

    // Helper to convert a float ggml tensor to a q8_0 tensor
    // Meanwhile, we need to tile that also
    void act_tensor_prepare(
        const float* act_tensor_data,
        int K, 
        int M
    ) {
        set_act_l(M);
        // assuming row major on the act_tensor_data
        // on this model, the K only have some limited choices
        if (K == 768) {
            // ViT QKVO and FFN expansion input (768-dim), including misnamed ffn_down.weight
            act_prepare_768(act_tensor_data, M);    
        } else if (K == 3072) {
            // ViT FFN projection input (3072-dim), including misnamed ffn_up.weight
            act_prepare_3072(act_tensor_data, M);
        } else if (K == 960) {
            // LM QKVO/FFN-UP/FFN-GATE
            act_prepare_960(act_tensor_data, M);
        } else if (K == 2560) {
            // LM FFN-DOWN
            act_prepare_2560(act_tensor_data, M);
        } else {
            throw std::runtime_error("Unsupported K: " + std::to_string(K));
        }
    }

    void act_prepare_768(
        const float* act_tensor_data,
        int M
    ) {
        act_prepare_aligned(act_tensor_data, M, 768);
    }

    void act_prepare_3072(
        const float* act_tensor_data,
        int M
    ) {
        act_prepare_aligned(act_tensor_data, M, 3072);
    }

#if 0
    void act_prepare_960(
        const float* act_tensor_data,
        int M
    ) {
        // clear 
        m_act_slots.clear();

        // 原始输入的 K 维度
        const int K_ORIGIN = 960;
        
        // 动态计算 256 的对齐倍数：960 向上对齐到 256 得到 1024
        const int K_ALIGNMENT = 256;
        const int K_ALIGNED = ((K_ORIGIN + K_ALIGNMENT - 1) / K_ALIGNMENT) * K_ALIGNMENT;

        int8_t* qdata = act_bo.map<int8_t*>();
        uint16_t* qscale = act_scale_bo.map<uint16_t*>();
        
        int m_blocks = get_m_blocks(M);
        int data_block_start = 0;
        int scale_block_start = 0;
        int last_m_count = 0;

        for (int i = 0; i < m_blocks; i++) {
            const int m_count = std::min(GEMMShape::TOT_M, M - i * GEMMShape::TOT_M);
            
            // 原始 float 矩阵在显存/内存中是紧凑行优先的，每行跨度为 K_ORIGIN (960)
            int block_row_start = i * K_ORIGIN * GEMMShape::TOT_M;
            
            // 沿 K 轴切块时，依据对齐后的 K 维度进行切块计算 (例如 1024 / TOT_K)
            int block_cnt_per_m_row = K_ALIGNED / GEMMShape::TOT_K;

            for (int j = 0; j < block_cnt_per_m_row; j++) {
                // 当前分块在逻辑对齐大矩阵（1024列）下的横向起始列偏移
                int block_start_logical = j * GEMMShape::TOT_K * m_count;
                int n_sub_blocks = m_count / GEMMShape::M_TILE;

                // 保持物理存储的紧凑地址累加，清除越界 padding 块的空间
                if (i == 0 && j == 0) {
                    data_block_start = 0;
                    scale_block_start = 0;
                } else {
                    data_block_start += last_m_count * GEMMShape::TOT_K;
                    scale_block_start += last_m_count * GEMMShape::TILES;
                }

                uint64_t data_start_addr = act_bo.address() + data_block_start*sizeof(int8_t);
                uint64_t scale_start_addr = act_scale_bo.address() + scale_block_start*sizeof(uint16_t);
                m_act_slots.push_back(ActSlot{
                    data_start_addr,
                    scale_start_addr,
                    m_count
                });

                for (int k = 0; k < n_sub_blocks; k++) {
                    int sub_block_data_start = data_block_start + k * GEMMShape::M_TILE * GEMMShape::TOT_K;
                    int sub_block_scale_start = scale_block_start + k * GEMMShape::M_TILE * GEMMShape::TILES;

                    // 1. 局部子块读取：默认全部初始化为 0.0f（安全实现自动补零）
                    std::vector<float> sub_block_data(GEMMShape::M_TILE * GEMMShape::TOT_K, 0.0f);
                    
                    for (int l = 0; l < GEMMShape::M_TILE; l++) {
                        // 当前读取行在原始大矩阵中的绝对行索引
                        int src_row_idx = (i * GEMMShape::TOT_M) + (k * GEMMShape::M_TILE) + l;
                        if (src_row_idx >= M) continue; // M 越界保护

                        // 当前子块在逻辑对齐大矩阵（1024列）下的绝对列起始索引
                        int dst_col_start = j * GEMMShape::TOT_K;

                        for (int m = 0; m < GEMMShape::TOT_K; m++) {
                            int current_col = dst_col_start + m;
                            
                            // 边界控制：只有当前列在原始列宽（960）以内时，才从原 tensor 读值
                            // 当 current_col >= 960 时，维持初始化的 0.0f
                            if (current_col < K_ORIGIN) {
                                int src_idx = src_row_idx * K_ORIGIN + current_col;
                                sub_block_data[l * GEMMShape::TOT_K + m] = act_tensor_data[src_idx];
                            }
                        }
                    }

                    // 2. 局部量化：由于越界通道全是 0，amax 计算和缩放公式天然对齐
                    std::vector<uint16_t> sub_block_scale(GEMMShape::M_TILE * GEMMShape::TILES);
                    std::vector<int8_t> sub_block_data_q8(GEMMShape::M_TILE * GEMMShape::TOT_K);
                    for (int l = 0; l < GEMMShape::M_TILE; l++) {
                        for (int m = 0; m < GEMMShape::TILES; m++) {
                            float amax = 0.0f;
                            for (int bid = 0; bid < GEMMShape::K_BLK; ++bid) {
                                amax = std::max(amax, std::abs(
                                    sub_block_data[l * GEMMShape::TOT_K + m * GEMMShape::K_BLK + bid]
                                ));
                            }
                            const float d = amax / 127.0f;
                            sub_block_scale[l * GEMMShape::TILES + m] = float_to_half(d);
                            const float id = (d == 0.0f) ? 0.0f : 1.0f / d;
                            for (int bid = 0; bid < GEMMShape::K_BLK; ++bid) {
                                float val = sub_block_data[l * GEMMShape::TOT_K + m * GEMMShape::K_BLK + bid] * id;
                                int q = std::round(val);
                                q = std::max(-127, std::min(127, q));
                                sub_block_data_q8[l * GEMMShape::TOT_K + m * GEMMShape::K_BLK + bid] = static_cast<int8_t>(q);
                            }
                        }
                    }

                    // 3. 物理布局变换：qdata 强行转置 (列优先紧凑排列)
                    for (int idk = 0; idk < GEMMShape::TOT_K; idk++) {
                        for (int idm = 0; idm < GEMMShape::M_TILE; idm++) {
                            qdata[sub_block_data_start + idk * GEMMShape::M_TILE + idm] = 
                                sub_block_data_q8[idm * GEMMShape::TOT_K + idk];
                        }
                    }

                    // 4. 物理布局变换：qscale 交错重排
                    for (int idm_out = 0; idm_out < 2; idm_out++) {
                        for (int idk = 0; idk < GEMMShape::TILES; idk++) {
                            for (int idm = 0; idm < GEMMShape::M_TILE / 2; idm++) {
                                qscale[sub_block_scale_start + idm_out * GEMMShape::TILES * GEMMShape::M_TILE / 2 + idk * GEMMShape::M_TILE / 2 + idm] = 
                                    sub_block_scale[(idm + idm_out * GEMMShape::M_TILE / 2) * GEMMShape::TILES + idk];
                            }
                        }
                    }
                }
                last_m_count = m_count;
            }
        }
    }
#endif
    // gemini generated
    void act_prepare_960(
        const float* act_tensor_data,
        int M
    ) {
        // 1. 清空当前槽位
        m_act_slots.clear();

        const int K_ORIGIN = 960;
        const int K_ALIGNED = 1024; // (960 + 255) & ~255
        const int K_BLOCK_SIZE = 256; // GEMMShape::TOT_K = 256
        const int M_BLOCK_SIZE = 512; // GEMMShape::TOT_M = 512
        const int M_TILE_SIZE = 16;  // GEMMShape::M_TILE = 16

        // 【优化点 1】M 方向向上对齐到 16 的整数倍
        int M_aligned_16 = (M + 15) & ~15;

        // if the M>256 (heuristic), use cacheable
        // else use normal
        int8_t* qdata = (M > 256) ? act_bo.map<int8_t*>() : act_decode_bo.map<int8_t*>();
        uint16_t* qscale = (M > 256) ? act_scale_bo.map<uint16_t*>() : act_scale_decode_bo.map<uint16_t*>();
        
        int m_blocks = get_m_blocks(M_aligned_16);
        int block_cnt_per_m_row = K_ALIGNED / K_BLOCK_SIZE; // 1024 / 256 = 4

        // 2. 预先串行计算所有外层循环的 Offset，保证多线程无数据竞争
        int total_iterations = m_blocks * block_cnt_per_m_row;
        std::vector<int> data_starts(total_iterations);
        std::vector<int> scale_starts(total_iterations);
        std::vector<int> m_counts(total_iterations);

        int current_data_block_start = 0;
        int current_scale_block_start = 0;
        int last_m_count = 0;

        for (int i = 0; i < m_blocks; i++) {
            // 每个大 Block 的 m_count 也要向上对齐到 16 的整数倍
            int m_count_raw = std::min(M_BLOCK_SIZE, M - i * M_BLOCK_SIZE);
            int m_count = (m_count_raw + 15) & ~15; 
            
            for (int j = 0; j < block_cnt_per_m_row; j++) {
                int idx = i * block_cnt_per_m_row + j;
                if (i == 0 && j == 0) {
                    current_data_block_start = 0;
                    current_scale_block_start = 0;
                } else {
                    current_data_block_start += last_m_count * K_BLOCK_SIZE;
                    current_scale_block_start += last_m_count * 8; // GEMMShape::TILES = 8
                }
                
                data_starts[idx] = current_data_block_start;
                scale_starts[idx] = current_scale_block_start;
                m_counts[idx] = m_count;

                uint64_t data_start_addr = (M > 256) ? act_bo.address() + current_data_block_start * sizeof(int8_t) 
                        : act_decode_bo.address() + current_data_block_start * sizeof(int8_t);
                uint64_t scale_start_addr = (M > 256) ? act_scale_bo.address() + current_scale_block_start * sizeof(uint16_t) 
                        : act_scale_decode_bo.address() + current_scale_block_start * sizeof(uint16_t);
                m_act_slots.push_back(ActSlot{ data_start_addr, scale_start_addr, m_count });
                
                last_m_count = m_count;
            }
        }

        // 3. OpenMP 4 线程并行化外层打平后的循环
        #pragma omp parallel for num_threads(4) schedule(static)
        for (int idx = 0; idx < total_iterations; idx++) {
            int i = idx / block_cnt_per_m_row;
            int j = idx % block_cnt_per_m_row;
            int m_count = m_counts[idx]; 
            int data_block_start = data_starts[idx];
            int scale_block_start = scale_starts[idx];

            // 原始 float 矩阵是紧凑行优先，每行硬件物理跨度为 K_ORIGIN (960)
            int block_row_start = i * K_ORIGIN * M_BLOCK_SIZE;
            // 当前 K 切块在逻辑 1024 矩阵下的列起始轴位置
            int dst_col_start = j * K_BLOCK_SIZE;
            int n_sub_blocks = m_count / M_TILE_SIZE;            

            // 线程局部栈缓存 (Zero Allocation + Aligned)
            alignas(16) float sub_block_data[16 * 256];
            alignas(16) uint16_t sub_block_scale[16 * 8];
            alignas(16) int8_t sub_block_data_q8[16 * 256];

            for (int k = 0; k < n_sub_blocks; k++) {
                // 当前子分块在整个 M 维度的绝对物理起始行号
                int global_m_start = i * M_BLOCK_SIZE + k * M_TILE_SIZE;

                // 【优化点 2】结合 M 和 K(960) 的双重边界安全加载机制
                for (int l = 0; l < 16; l++) {
                    int src_row_idx = global_m_start + l;
                    float* dst_row_ptr = &sub_block_data[l * 256];

                    if (src_row_idx < M) {
                        // M 方向未越界，检查 K 方向
                        // 计算当前 256 大小的分块在原 960 矩阵内有多少列是有效的
                        int valid_k_cols = std::max(0, std::min(K_BLOCK_SIZE, K_ORIGIN - dst_col_start));
                        
                        if (valid_k_cols == K_BLOCK_SIZE) {
                            // 彻底没有越界（全在 960 以内），触发最速连续内存拷贝
                            int src_idx = src_row_idx * K_ORIGIN + dst_col_start;
                            std::copy_n(&act_tensor_data[src_idx], 256, dst_row_ptr);
                        } else if (valid_k_cols > 0) {
                            // 部分越界（比如第 4 个 K 切块，dst_col_start=768，有效 960-768=192 列，剩余 64 列需要补 0）
                            int src_idx = src_row_idx * K_ORIGIN + dst_col_start;
                            std::copy_n(&act_tensor_data[src_idx], valid_k_cols, dst_row_ptr);
                            std::fill_n(dst_row_ptr + valid_k_cols, K_BLOCK_SIZE - valid_k_cols, 0.0f);
                        } else {
                            // 完全超出 960 范围（安全防范，实际 1024 矩阵不会出现完全超出）
                            std::fill_n(dst_row_ptr, 256, 0.0f);
                        }
                    } else {
                        // M 方向彻底越界，整行无脑 Pad 0
                        std::fill_n(dst_row_ptr, 256, 0.0f);
                    }
                }

                // 4. 核心计算：寻找 amax 并进行 Q8 量化 (NEON SIMD)
                for (int l = 0; l < 16; l++) {
                    float* row_data = &sub_block_data[l * 256];

                    for (int m = 0; m < 8; m++) { // TILES = 8
                        float* blk_ptr = &row_data[m * 32]; // K_BLK = 32
                        
                        // 用 8 个 NEON 寄存器完全展开并行吞吐 32 个 float 的绝对值最大值
                        float32x4_t v_max0 = vabsq_f32(vld1q_f32(blk_ptr + 0));
                        float32x4_t v_max1 = vabsq_f32(vld1q_f32(blk_ptr + 4));
                        float32x4_t v_max2 = vabsq_f32(vld1q_f32(blk_ptr + 8));
                        float32x4_t v_max3 = vabsq_f32(vld1q_f32(blk_ptr + 12));
                        v_max0 = vmaxq_f32(v_max0, v_max1);
                        v_max2 = vmaxq_f32(v_max2, v_max3);
                        
                        float32x4_t v_max4 = vabsq_f32(vld1q_f32(blk_ptr + 16));
                        float32x4_t v_max5 = vabsq_f32(vld1q_f32(blk_ptr + 20));
                        float32x4_t v_max6 = vabsq_f32(vld1q_f32(blk_ptr + 24));
                        float32x4_t v_max7 = vabsq_f32(vld1q_f32(blk_ptr + 28));
                        v_max4 = vmaxq_f32(v_max4, v_max5);
                        v_max6 = vmaxq_f32(v_max6, v_max7);

                        v_max0 = vmaxq_f32(v_max0, v_max2);
                        v_max4 = vmaxq_f32(v_max4, v_max6);
                        float32x4_t v_max = vmaxq_f32(v_max0, v_max4);
                        
                        // 树状规约提取标量 amax
                        float amax = std::max({vgetq_lane_f32(v_max, 0), vgetq_lane_f32(v_max, 1), 
                                            vgetq_lane_f32(v_max, 2), vgetq_lane_f32(v_max, 3)});

                        const float d = amax / 127.0f;
                        sub_block_scale[l * 8 + m] = float_to_half(d);
                        const float id = (d == 0.0f) ? 0.0f : 1.0f / d;

                        float32x4_t v_id = vdupq_n_f32(id);
                        int8_t* q8_row_ptr = &sub_block_data_q8[l * 256 + m * 32];

                        for (int b = 0; b < 32; b += 8) {
                            float32x4_t v_val0 = vmulq_f32(vld1q_f32(blk_ptr + b), v_id);
                            float32x4_t v_val1 = vmulq_f32(vld1q_f32(blk_ptr + b + 4), v_id);
                            
                            int32x4_t v_q32_0 = vcvtnq_s32_f32(v_val0); 
                            int32x4_t v_q32_1 = vcvtnq_s32_f32(v_val1); 

                            int16x4_t v_q16_0 = vqmovn_s32(v_q32_0);
                            int16x4_t v_q16_1 = vqmovn_s32(v_q32_1);
                            int8x8_t v_q8 = vqmovn_s16(vcombine_s16(v_q16_0, v_q16_1));
                            
                            vst1_s8(q8_row_ptr + b, v_q8);
                        }
                    }
                }

                // 5. Data 转置与全局写回：将 16x256 行主序转置为 256x16 列主序
                int sub_block_data_offset = data_block_start + k * M_TILE_SIZE * K_BLOCK_SIZE;
                int8_t* global_qdata_ptr = &qdata[sub_block_data_offset];

                for (int cc = 0; cc < 256; cc += 16) {
                    for (int lx = 0; lx < 16; lx++) {
                        int8x16_t row_vec = vld1q_s8(&sub_block_data_q8[lx * 256 + cc]);
                        
                        global_qdata_ptr[(cc + 0) * 16 + lx] = vgetq_lane_s8(row_vec, 0);
                        global_qdata_ptr[(cc + 1) * 16 + lx] = vgetq_lane_s8(row_vec, 1);
                        global_qdata_ptr[(cc + 2) * 16 + lx] = vgetq_lane_s8(row_vec, 2);
                        global_qdata_ptr[(cc + 3) * 16 + lx] = vgetq_lane_s8(row_vec, 3);
                        global_qdata_ptr[(cc + 4) * 16 + lx] = vgetq_lane_s8(row_vec, 4);
                        global_qdata_ptr[(cc + 5) * 16 + lx] = vgetq_lane_s8(row_vec, 5);
                        global_qdata_ptr[(cc + 6) * 16 + lx] = vgetq_lane_s8(row_vec, 6);
                        global_qdata_ptr[(cc + 7) * 16 + lx] = vgetq_lane_s8(row_vec, 7);
                        global_qdata_ptr[(cc + 8) * 16 + lx] = vgetq_lane_s8(row_vec, 8);
                        global_qdata_ptr[(cc + 9) * 16 + lx] = vgetq_lane_s8(row_vec, 9);
                        global_qdata_ptr[(cc + 10) * 16 + lx] = vgetq_lane_s8(row_vec, 10);
                        global_qdata_ptr[(cc + 11) * 16 + lx] = vgetq_lane_s8(row_vec, 11);
                        global_qdata_ptr[(cc + 12) * 16 + lx] = vgetq_lane_s8(row_vec, 12);
                        global_qdata_ptr[(cc + 13) * 16 + lx] = vgetq_lane_s8(row_vec, 13);
                        global_qdata_ptr[(cc + 14) * 16 + lx] = vgetq_lane_s8(row_vec, 14);
                        global_qdata_ptr[(cc + 15) * 16 + lx] = vgetq_lane_s8(row_vec, 15);
                    }
                }

                // 6. Scale 尺度转置优化与全局写回
                // 原逻辑是交错重排（前 8 行和后 8 行分别做 8x8 转置并合并写回）
                int sub_block_scale_offset = scale_block_start + k * M_TILE_SIZE * 8;
                uint16_t* global_scale_ptr = &qscale[sub_block_scale_offset];

                for (int idm_out = 0; idm_out < 2; idm_out++) {
                    uint16_t* src_scale = &sub_block_scale[idm_out * 8 * 8];
                    int out_base = idm_out * 8 * 8; // 前8行与后8行在输出时的物理偏移（64元素）
                    
                    // 寄存器级 8x8 标量高速循环展开
                    for (int idk = 0; idk < 8; idk++) {
                        for (int idm = 0; idm < 8; idm++) {
                            global_scale_ptr[out_base + idk * 8 + idm] = src_scale[idm * 8 + idk];
                        }
                    }
                }
            }
        }

        if (M > 256) {
            act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            act_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else {
            act_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            act_scale_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    }

    void act_prepare_2560(
        const float* act_tensor_data,
        int M
    ) {
        if (M % 16 == 0) {
            act_prepare_aligned(act_tensor_data, M, 2560);
        } else {
            act_prepare_aligned_m_unaligned(act_tensor_data, M, 2560);
        }
    }    

#if 0
    void act_prepare_aligned(
        const float* act_tensor_data,
        int M, 
        int K
    ) {
        // clear 
        m_act_slots.clear();

        int8_t* qdata = act_bo.map<int8_t*>();
        uint16_t* qscale = act_scale_bo.map<uint16_t*>();
        int m_blocks = get_m_blocks(M);
        int data_block_start = 0;
        int scale_block_start = 0;
        int last_m_count = 0;
        for (int i = 0; i < m_blocks; i++) {
            const int m_count = std::min(GEMMShape::TOT_M, M - i * GEMMShape::TOT_M);
            // Starting pointer of the block row
            int block_row_start = i * K * GEMMShape::TOT_M;
            // 768 perfectly divide the 256
            int block_cnt_per_m_row = K / GEMMShape::TOT_K;
            for (int j = 0; j < block_cnt_per_m_row; j++) {
                // quantize the m_count x 256 block
                // Row-major act[row,col]: row in [0,m_count), col in [j*TOT_K, j*TOT_K+TOT_K)
                int block_start = block_row_start + j * GEMMShape::TOT_K;
                int n_sub_blocks = m_count / GEMMShape::M_TILE;
                // starting address for the blocks
                if (i==0 && j==0) {
                    // first block starts from 0
                    data_block_start = 0;
                    scale_block_start = 0;
                } else {
                    data_block_start += last_m_count * GEMMShape::TOT_K;
                    scale_block_start += last_m_count * GEMMShape::TILES;
                }
                uint64_t data_start_addr = act_bo.address() + data_block_start*sizeof(int8_t);
                uint64_t scale_start_addr = act_scale_bo.address() + scale_block_start*sizeof(uint16_t);
                m_act_slots.push_back(ActSlot{
                    data_start_addr,
                    scale_start_addr,
                    m_count
                });
                for (int k = 0; k < n_sub_blocks; k++) {
                    // quantize the M_TILE x 256 block (rows k*16..k*16+15 in this K-slab)
                    int sub_block_start = block_start + k * GEMMShape::M_TILE * K;
                    // Quantize the sub-block to Q8_0 format. 
                    // Store the quantized data to the act_bo and scale to act_scale_bo
                    // Use row major order for both the data and scale
                    std::vector<float> sub_block_data(GEMMShape::M_TILE * GEMMShape::TOT_K);
                    for (int l = 0; l < GEMMShape::M_TILE; l++) {
                        for (int m = 0; m < GEMMShape::TOT_K; m++) {
                            sub_block_data[l * GEMMShape::TOT_K + m] = 
                                act_tensor_data[sub_block_start + l * K + m];
                        }
                    }
                    std::vector<uint16_t> sub_block_scale(GEMMShape::M_TILE * GEMMShape::TILES);
                    std::vector<int8_t> sub_block_data_q8(GEMMShape::M_TILE * GEMMShape::TOT_K);
                    for (int l = 0; l < GEMMShape::M_TILE; l++) {
                        for (int m = 0; m < GEMMShape::TILES; m++) {
                            float amax = 0.0f;
                            for (int bid = 0; bid < GEMMShape::K_BLK; ++bid) {
                                amax = std::max(amax, std::abs(
                                    sub_block_data[l * GEMMShape::TOT_K + m * GEMMShape::K_BLK + bid]
                                ));
                            }
                            const float d = amax / 127.0f;
                            sub_block_scale[l * GEMMShape::TILES + m] = float_to_half(d);
                            const float id = (d == 0.0f) ? 0.0f : 1.0f / d;
                            for (int bid = 0; bid < GEMMShape::K_BLK; ++bid) {
                                float val = sub_block_data[l * GEMMShape::TOT_K + m * GEMMShape::K_BLK + bid] 
                                            * id;
                                
                                // Round to nearest integer and clamp to [-127, 127]
                                int q = std::lround(val);
                                q = std::max(-127, std::min(127, q));

                                sub_block_data_q8[l * GEMMShape::TOT_K + m * GEMMShape::K_BLK + bid] 
                                    = static_cast<int8_t>(q);
                            }
                        }
                    }
                    // storing the quantized data and scale to the act_bo and act_scale_bo
                    // starting address of each sub block
                    int sub_block_data_start = data_block_start + k * GEMMShape::M_TILE * GEMMShape::TOT_K;
                    int sub_block_scale_start = scale_block_start + k * GEMMShape::M_TILE * GEMMShape::TILES;
                    // transpose on the sub block for the data
                    for (int idk=0; idk<GEMMShape::TOT_K; idk++) {
                        for (int idm=0; idm<GEMMShape::M_TILE; idm++) {
                            qdata[sub_block_data_start + idk * GEMMShape::M_TILE + idm] = 
                                sub_block_data_q8[idm * GEMMShape::TOT_K + idk];
                        }
                    }
                    // transpose on the sub block for the scale
                    for (int idm_out=0; idm_out<2; idm_out++) {
                        for (int idk=0; idk<GEMMShape::TILES; idk++) {
                            for (int idm=0; idm<GEMMShape::M_TILE/2; idm++) {
                                qscale[sub_block_scale_start + idm_out * GEMMShape::TILES * GEMMShape::M_TILE/2 + idk * GEMMShape::M_TILE/2 + idm] = 
                                    sub_block_scale[(idm + idm_out * GEMMShape::M_TILE/2) * GEMMShape::TILES + idk];
                            }
                        }
                    }
                }
                last_m_count = m_count;
            }
        }
    }
#endif

    // gemini generated code
    void act_prepare_aligned_m_unaligned(
        const float* act_tensor_data,
        int M, 
        int K
    ) {
        // 1. 清空当前槽位
        m_act_slots.clear();

        // 【修改点 1】计算 M 方向向上对齐到 16 的总量，用于后续分块和写回地址计算
        int M_aligned_16 = (M + 15) & ~15; 

        // if the M>256 (heuristic), use cacheable
        // else use normal
        int8_t* qdata = (M > 256) ? act_bo.map<int8_t*>() : act_decode_bo.map<int8_t*>();
        uint16_t* qscale = (M > 256) ? act_scale_bo.map<uint16_t*>() : act_scale_decode_bo.map<uint16_t*>();
        
        // 注意：get_m_blocks 内部如果是基于原始 M 计算的，这里需要确保它能覆盖到 Padding 后的范围。
        // 如果 get_m_blocks 本身就是按 GEMMShape::TOT_M (512) 对齐，而 512 已经是 16 的倍数，则无需修改。
        int m_blocks = get_m_blocks(M_aligned_16); 
        int block_cnt_per_m_row = K / GEMMShape::TOT_K; // GEMMShape::TOT_K = 256
        
        // 2. 预先串行计算所有外层循环的 Offset（基于对齐后的 M 确保槽位大小正确）
        int total_iterations = m_blocks * block_cnt_per_m_row;
        std::vector<int> data_starts(total_iterations);
        std::vector<int> scale_starts(total_iterations);
        std::vector<int> m_counts(total_iterations);

        int current_data_block_start = 0;
        int current_scale_block_start = 0;
        int last_m_count = 0;

        for (int i = 0; i < m_blocks; i++) {
            // 【修改点 2】每个大 Block 的 m_count 也要向上对齐到 16 的整数倍
            // 例如原始最后剩下 319 行，对齐后该 Block 需要处理 320 行。
            int m_count_raw = std::min(GEMMShape::TOT_M, M - i * GEMMShape::TOT_M);
            int m_count = (m_count_raw + 15) & ~15; 
            
            for (int j = 0; j < block_cnt_per_m_row; j++) {
                int idx = i * block_cnt_per_m_row + j;
                if (i == 0 && j == 0) {
                    current_data_block_start = 0;
                    current_scale_block_start = 0;
                } else {
                    current_data_block_start += last_m_count * 256; // TOT_K = 256
                    current_scale_block_start += last_m_count * 8;  // TILES = 8
                }
                
                data_starts[idx] = current_data_block_start;
                scale_starts[idx] = current_scale_block_start;
                m_counts[idx] = m_count;

                uint64_t data_start_addr = (M > 256) ? act_bo.address() + current_data_block_start * sizeof(int8_t) 
                        : act_decode_bo.address() + current_data_block_start * sizeof(int8_t);
                uint64_t scale_start_addr = (M > 256) ? act_scale_bo.address() + current_scale_block_start * sizeof(uint16_t) 
                        : act_scale_decode_bo.address() + current_scale_block_start * sizeof(uint16_t);
                m_act_slots.push_back(ActSlot{ data_start_addr, scale_start_addr, m_count });
                
                last_m_count = m_count;
            }
        }

        // 3. OpenMP 4 线程并行化外层打平后的循环
        #pragma omp parallel for num_threads(4) schedule(static)
        for (int idx = 0; idx < total_iterations; idx++) {
            int i = idx / block_cnt_per_m_row;
            int j = idx % block_cnt_per_m_row;
            int m_count = m_counts[idx]; // 这里拿到的已经是 16 对齐后的 m_count 了
            int data_block_start = data_starts[idx];
            int scale_block_start = scale_starts[idx];

            // 这里的 block_row_start 依然基于原始输入矩阵的物理坐标映射
            int block_row_start = i * K * GEMMShape::TOT_M;
            int block_start = block_row_start + j * 256; // TOT_K = 256
            int n_sub_blocks = m_count / 16;             // M_TILE = 16

            // 线程局部栈缓存
            alignas(16) float sub_block_data[16 * 256];
            alignas(16) uint16_t sub_block_scale[16 * 8];
            alignas(16) int8_t sub_block_data_q8[16 * 256];

            for (int k = 0; k < n_sub_blocks; k++) {
                int sub_block_start = block_start + k * 16 * K;

                // 【修改点 3】判断当前 16x256 的子分块是否超出了原始 M 的物理边界
                // 当前子分块在整个 M 维度的起始行号：
                int global_m_start = i * GEMMShape::TOT_M + k * 16;

                if (global_m_start + 16 <= M) {
                    // 情况 A：完全在有效范围内，执行原本的最速连续内存拷贝
                    for (int l = 0; l < 16; l++) {
                        std::copy_n(&act_tensor_data[sub_block_start + l * K], 256, &sub_block_data[l * 256]);
                    }
                } else {
                    // 情况 B：触发边界 Padding。部分行有效，部分行需要填 0
                    int valid_rows = std::max(0, M - global_m_start);
                    
                    // 1. 拷贝有效行
                    for (int l = 0; l < valid_rows; l++) {
                        std::copy_n(&act_tensor_data[sub_block_start + l * K], 256, &sub_block_data[l * 256]);
                    }
                    // 2. 对越界的剩余行进行安全 Pad 0
                    for (int l = valid_rows; l < 16; l++) {
                        std::fill_n(&sub_block_data[l * 256], 256, 0.0f);
                    }
                }

                // 4. 核心计算：寻找 amax 并进行 Q8 量化（逻辑完全无需修改，SIMD 天生支持全 0 数据的量化）
                for (int l = 0; l < 16; l++) {
                    float* row_data = &sub_block_data[l * 256];

                    for (int m = 0; m < 8; m++) { // TILES = 8
                        float* blk_ptr = &row_data[m * 32]; // K_BLK = 32
                        
                        // 用 8 个 NEON 寄存器完全展开并行吞吐 32 个 float 的绝对值最大值
                        float32x4_t v_max0 = vabsq_f32(vld1q_f32(blk_ptr + 0));
                        float32x4_t v_max1 = vabsq_f32(vld1q_f32(blk_ptr + 4));
                        float32x4_t v_max2 = vabsq_f32(vld1q_f32(blk_ptr + 8));
                        float32x4_t v_max3 = vabsq_f32(vld1q_f32(blk_ptr + 12));
                        v_max0 = vmaxq_f32(v_max0, v_max1);
                        v_max2 = vmaxq_f32(v_max2, v_max3);
                        
                        float32x4_t v_max4 = vabsq_f32(vld1q_f32(blk_ptr + 16));
                        float32x4_t v_max5 = vabsq_f32(vld1q_f32(blk_ptr + 20));
                        float32x4_t v_max6 = vabsq_f32(vld1q_f32(blk_ptr + 24));
                        float32x4_t v_max7 = vabsq_f32(vld1q_f32(blk_ptr + 28));
                        v_max4 = vmaxq_f32(v_max4, v_max5);
                        v_max6 = vmaxq_f32(v_max6, v_max7);

                        v_max0 = vmaxq_f32(v_max0, v_max2);
                        v_max4 = vmaxq_f32(v_max4, v_max6);
                        float32x4_t v_max = vmaxq_f32(v_max0, v_max4);
                        
                        // 树状规约提取标量 amax
                        float amax = std::max({vgetq_lane_f32(v_max, 0), vgetq_lane_f32(v_max, 1), 
                                            vgetq_lane_f32(v_max, 2), vgetq_lane_f32(v_max, 3)});

                        const float d = amax / 127.0f;
                        sub_block_scale[l * 8 + m] = float_to_half(d);
                        const float id = (d == 0.0f) ? 0.0f : 1.0f / d;

                        float32x4_t v_id = vdupq_n_f32(id);
                        int8_t* q8_row_ptr = &sub_block_data_q8[l * 256 + m * 32];

                        // 对 32 个元素进行 NEON 硬件级四舍五入量化与饱和截断
                        for (int b = 0; b < 32; b += 8) {
                            float32x4_t v_val0 = vmulq_f32(vld1q_f32(blk_ptr + b), v_id);
                            float32x4_t v_val1 = vmulq_f32(vld1q_f32(blk_ptr + b + 4), v_id);
                            
                            int32x4_t v_q32_0 = vcvtnq_s32_f32(v_val0); 
                            int32x4_t v_q32_1 = vcvtnq_s32_f32(v_val1); 

                            int16x4_t v_q16_0 = vqmovn_s32(v_q32_0);
                            int16x4_t v_q16_1 = vqmovn_s32(v_q32_1);
                            int8x8_t v_q8 = vqmovn_s16(vcombine_s16(v_q16_0, v_q16_1));
                            
                            vst1_s8(q8_row_ptr + b, v_q8);
                        }
                    }
                }

                // 5. Data 转置与全局写回：逻辑完全不需要变，因为多出来的行在全局内存中会整齐地写在尾部
                int sub_block_data_start = data_block_start + k * 16 * 256;
                int8_t* global_qdata_ptr = &qdata[sub_block_data_start];

                for (int cc = 0; cc < 256; cc += 16) {
                    for (int lx = 0; lx < 16; lx++) {
                        int8x16_t row_vec = vld1q_s8(&sub_block_data_q8[lx * 256 + cc]);
                        
                        global_qdata_ptr[(cc + 0) * 16 + lx] = vgetq_lane_s8(row_vec, 0);
                        global_qdata_ptr[(cc + 1) * 16 + lx] = vgetq_lane_s8(row_vec, 1);
                        global_qdata_ptr[(cc + 2) * 16 + lx] = vgetq_lane_s8(row_vec, 2);
                        global_qdata_ptr[(cc + 3) * 16 + lx] = vgetq_lane_s8(row_vec, 3);
                        global_qdata_ptr[(cc + 4) * 16 + lx] = vgetq_lane_s8(row_vec, 4);
                        global_qdata_ptr[(cc + 5) * 16 + lx] = vgetq_lane_s8(row_vec, 5);
                        global_qdata_ptr[(cc + 6) * 16 + lx] = vgetq_lane_s8(row_vec, 6);
                        global_qdata_ptr[(cc + 7) * 16 + lx] = vgetq_lane_s8(row_vec, 7);
                        global_qdata_ptr[(cc + 8) * 16 + lx] = vgetq_lane_s8(row_vec, 8);
                        global_qdata_ptr[(cc + 9) * 16 + lx] = vgetq_lane_s8(row_vec, 9);
                        global_qdata_ptr[(cc + 10) * 16 + lx] = vgetq_lane_s8(row_vec, 10);
                        global_qdata_ptr[(cc + 11) * 16 + lx] = vgetq_lane_s8(row_vec, 11);
                        global_qdata_ptr[(cc + 12) * 16 + lx] = vgetq_lane_s8(row_vec, 12);
                        global_qdata_ptr[(cc + 13) * 16 + lx] = vgetq_lane_s8(row_vec, 13);
                        global_qdata_ptr[(cc + 14) * 16 + lx] = vgetq_lane_s8(row_vec, 14);
                        global_qdata_ptr[(cc + 15) * 16 + lx] = vgetq_lane_s8(row_vec, 15);
                    }
                }

                // 6. Scale 尺度转置优化与全局写回
                int sub_block_scale_start = scale_block_start + k * 16 * 8;
                uint16_t* global_scale_ptr = &qscale[sub_block_scale_start];

                for (int idm_out = 0; idm_out < 2; idm_out++) {
                    uint16_t* src_scale = &sub_block_scale[idm_out * 8 * 8];
                    int out_base = idm_out * 8 * 8;
                    
                    for (int idk = 0; idk < 8; idk++) {
                        for (int idm = 0; idm < 8; idm++) {
                            global_scale_ptr[out_base + idk * 8 + idm] = src_scale[idm * 8 + idk];
                        }
                    }
                }
            }
        }

        if (M > 256) {
            act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            act_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else {
            act_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            act_scale_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    }

    void act_prepare_aligned(
        const float* act_tensor_data,
        int M, 
        int K
    ) {
        // 1. 清空当前槽位
        m_act_slots.clear();

        // if the M>256 (heuristic), use cacheable
        // else use normal
        int8_t* qdata = (M > 256) ? act_bo.map<int8_t*>() : act_decode_bo.map<int8_t*>();
        uint16_t* qscale = (M > 256) ? act_scale_bo.map<uint16_t*>() : act_scale_decode_bo.map<uint16_t*>();

        int m_blocks = get_m_blocks(M);
        int block_cnt_per_m_row = K / GEMMShape::TOT_K; // GEMMShape::TOT_K = 256
        
        // 2. 预先串行计算所有外层循环的 Offset，保证多线程无数据竞争，并安全填充 m_act_slots
        int total_iterations = m_blocks * block_cnt_per_m_row;
        std::vector<int> data_starts(total_iterations);
        std::vector<int> scale_starts(total_iterations);
        std::vector<int> m_counts(total_iterations);

        int current_data_block_start = 0;
        int current_scale_block_start = 0;
        int last_m_count = 0;

        for (int i = 0; i < m_blocks; i++) {
            int m_count = std::min(GEMMShape::TOT_M, M - i * GEMMShape::TOT_M); // GEMMShape::TOT_M = 512
            for (int j = 0; j < block_cnt_per_m_row; j++) {
                int idx = i * block_cnt_per_m_row + j;
                if (i == 0 && j == 0) {
                    current_data_block_start = 0;
                    current_scale_block_start = 0;
                } else {
                    current_data_block_start += last_m_count * 256; // TOT_K = 256
                    current_scale_block_start += last_m_count * 8;  // TILES = 8
                }
                
                data_starts[idx] = current_data_block_start;
                scale_starts[idx] = current_scale_block_start;
                m_counts[idx] = m_count;

                if (M > 256) {
                    uint64_t data_start_addr = act_bo.address() + current_data_block_start * sizeof(int8_t);
                    uint64_t scale_start_addr = act_scale_bo.address() + current_scale_block_start * sizeof(uint16_t);
                    m_act_slots.push_back(ActSlot{ data_start_addr, scale_start_addr, m_count });
                } else {
                    uint64_t data_start_addr = act_decode_bo.address() + current_data_block_start * sizeof(int8_t);
                    uint64_t scale_start_addr = act_scale_decode_bo.address() + current_scale_block_start * sizeof(uint16_t);
                    m_act_slots.push_back(ActSlot{ data_start_addr, scale_start_addr, m_count });
                }
                
                last_m_count = m_count;
            }
        }

        // 3. OpenMP 4 线程并行化外层打平后的循环
        #pragma omp parallel for num_threads(4) schedule(static)
        for (int idx = 0; idx < total_iterations; idx++) {
            int i = idx / block_cnt_per_m_row;
            int j = idx % block_cnt_per_m_row;
            int m_count = m_counts[idx];
            int data_block_start = data_starts[idx];
            int scale_block_start = scale_starts[idx];

            int block_row_start = i * K * GEMMShape::TOT_M;
            int block_start = block_row_start + j * 256; // TOT_K = 256
            int n_sub_blocks = m_count / 16;             // M_TILE = 16

            // 线程局部栈缓存，实现零动态内存分配 (Zero Allocation)
            alignas(16) float sub_block_data[16 * 256];
            alignas(16) uint16_t sub_block_scale[16 * 8];
            alignas(16) int8_t sub_block_data_q8[16 * 256];

            for (int k = 0; k < n_sub_blocks; k++) {
                int sub_block_start = block_start + k * 16 * K;

                // 完美的流水线连续内存拷贝
                for (int l = 0; l < 16; l++) {
                    std::copy_n(&act_tensor_data[sub_block_start + l * K], 256, &sub_block_data[l * 256]);
                }

                // 4. 核心计算：寻找 amax 并进行 Q8 量化
                for (int l = 0; l < 16; l++) {
                    float* row_data = &sub_block_data[l * 256];

                    for (int m = 0; m < 8; m++) { // TILES = 8
                        float* blk_ptr = &row_data[m * 32]; // K_BLK = 32
                        
                        // 用 8 个 NEON 寄存器完全展开并行吞吐 32 个 float 的绝对值最大值
                        float32x4_t v_max0 = vabsq_f32(vld1q_f32(blk_ptr + 0));
                        float32x4_t v_max1 = vabsq_f32(vld1q_f32(blk_ptr + 4));
                        float32x4_t v_max2 = vabsq_f32(vld1q_f32(blk_ptr + 8));
                        float32x4_t v_max3 = vabsq_f32(vld1q_f32(blk_ptr + 12));
                        v_max0 = vmaxq_f32(v_max0, v_max1);
                        v_max2 = vmaxq_f32(v_max2, v_max3);
                        
                        float32x4_t v_max4 = vabsq_f32(vld1q_f32(blk_ptr + 16));
                        float32x4_t v_max5 = vabsq_f32(vld1q_f32(blk_ptr + 20));
                        float32x4_t v_max6 = vabsq_f32(vld1q_f32(blk_ptr + 24));
                        float32x4_t v_max7 = vabsq_f32(vld1q_f32(blk_ptr + 28));
                        v_max4 = vmaxq_f32(v_max4, v_max5);
                        v_max6 = vmaxq_f32(v_max6, v_max7);

                        v_max0 = vmaxq_f32(v_max0, v_max2);
                        v_max4 = vmaxq_f32(v_max4, v_max6);
                        float32x4_t v_max = vmaxq_f32(v_max0, v_max4);
                        
                        // 树状规约提取标量 amax
                        float amax = std::max({vgetq_lane_f32(v_max, 0), vgetq_lane_f32(v_max, 1), 
                                            vgetq_lane_f32(v_max, 2), vgetq_lane_f32(v_max, 3)});

                        const float d = amax / 127.0f;
                        sub_block_scale[l * 8 + m] = float_to_half(d);
                        const float id = (d == 0.0f) ? 0.0f : 1.0f / d;

                        float32x4_t v_id = vdupq_n_f32(id);
                        int8_t* q8_row_ptr = &sub_block_data_q8[l * 256 + m * 32];

                        // 对 32 个元素进行 NEON 硬件级四舍五入量化与饱和截断
                        for (int b = 0; b < 32; b += 8) {
                            float32x4_t v_val0 = vmulq_f32(vld1q_f32(blk_ptr + b), v_id);
                            float32x4_t v_val1 = vmulq_f32(vld1q_f32(blk_ptr + b + 4), v_id);
                            
                            // 硬件指令级四舍五入并转为 int32
                            int32x4_t v_q32_0 = vcvtnq_s32_f32(v_val0); 
                            int32x4_t v_q32_1 = vcvtnq_s32_f32(v_val1); 

                            // 饱和截断到 int16，再到 int8
                            int16x4_t v_q16_0 = vqmovn_s32(v_q32_0);
                            int16x4_t v_q16_1 = vqmovn_s32(v_q32_1);
                            int8x8_t v_q8 = vqmovn_s16(vcombine_s16(v_q16_0, v_q16_1));
                            
                            vst1_s8(q8_row_ptr + b, v_q8);
                        }
                    }
                }

                // 5. Data 转置与全局写回：将 16x256 的行主序转置为 256x16 写入全局内存
                // 采用 16x16 分块转置逻辑，最大化利用 L1 Cache 的单字节高速写性能
                int sub_block_data_start = data_block_start + k * 16 * 256;
                int8_t* global_qdata_ptr = &qdata[sub_block_data_start];

                for (int cc = 0; cc < 256; cc += 16) {
                    for (int lx = 0; lx < 16; lx++) {
                        int8x16_t row_vec = vld1q_s8(&sub_block_data_q8[lx * 256 + cc]);
                        
                        // 直接展开，规避内层跨步带来的指令依赖
                        global_qdata_ptr[(cc + 0) * 16 + lx] = vgetq_lane_s8(row_vec, 0);
                        global_qdata_ptr[(cc + 1) * 16 + lx] = vgetq_lane_s8(row_vec, 1);
                        global_qdata_ptr[(cc + 2) * 16 + lx] = vgetq_lane_s8(row_vec, 2);
                        global_qdata_ptr[(cc + 3) * 16 + lx] = vgetq_lane_s8(row_vec, 3);
                        global_qdata_ptr[(cc + 4) * 16 + lx] = vgetq_lane_s8(row_vec, 4);
                        global_qdata_ptr[(cc + 5) * 16 + lx] = vgetq_lane_s8(row_vec, 5);
                        global_qdata_ptr[(cc + 6) * 16 + lx] = vgetq_lane_s8(row_vec, 6);
                        global_qdata_ptr[(cc + 7) * 16 + lx] = vgetq_lane_s8(row_vec, 7);
                        global_qdata_ptr[(cc + 8) * 16 + lx] = vgetq_lane_s8(row_vec, 8);
                        global_qdata_ptr[(cc + 9) * 16 + lx] = vgetq_lane_s8(row_vec, 9);
                        global_qdata_ptr[(cc + 10) * 16 + lx] = vgetq_lane_s8(row_vec, 10);
                        global_qdata_ptr[(cc + 11) * 16 + lx] = vgetq_lane_s8(row_vec, 11);
                        global_qdata_ptr[(cc + 12) * 16 + lx] = vgetq_lane_s8(row_vec, 12);
                        global_qdata_ptr[(cc + 13) * 16 + lx] = vgetq_lane_s8(row_vec, 13);
                        global_qdata_ptr[(cc + 14) * 16 + lx] = vgetq_lane_s8(row_vec, 14);
                        global_qdata_ptr[(cc + 15) * 16 + lx] = vgetq_lane_s8(row_vec, 15);
                    }
                }

                // 6. Scale 尺度转置优化与全局写回
                // 原逻辑是分别处理前 8 行和后 8 行，产生两个标准的 8x8 矩阵转置写回
                int sub_block_scale_start = scale_block_start + k * 16 * 8;
                uint16_t* global_scale_ptr = &qscale[sub_block_scale_start];

                for (int idm_out = 0; idm_out < 2; idm_out++) {
                    uint16_t* src_scale = &sub_block_scale[idm_out * 8 * 8];
                    int out_base = idm_out * 8 * 8;
                    
                    // 寄存器级 8x8 标量高速循环展开（规避了多级不连续的深度循环嵌套）
                    for (int idk = 0; idk < 8; idk++) {
                        for (int idm = 0; idm < 8; idm++) {
                            global_scale_ptr[out_base + idk * 8 + idm] = src_scale[idm * 8 + idk];
                        }
                    }
                }
            }
        }

        if (M > 256) {
            act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            act_scale_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else {
            act_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            act_scale_decode_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    }

    // print out all the act slots
    // minus the address with the act_bo.address() or act_scale_bo.address()
    void print_act_slots() {
        for (int i = 0; i < m_act_slots.size(); i++) {
            printf("act slot %d: data_start_addr: %ld, scale_start_addr: %ld, m_count: %d\n", 
                i, m_act_slots[i].data_start_addr - act_bo.address(), 
                m_act_slots[i].scale_start_addr - act_scale_bo.address(), 
                m_act_slots[i].m_count);
        }
    }

    int get_m_blocks(int M) {
        return (M + GEMMShape::TOT_M - 1) / GEMMShape::TOT_M;
    }

    void set_act_l(int M) {
        int tmp = (M + 15) / 16;
        act_l = tmp * 16;
    }

    void set_act_n(int N) {
        int tmp = (N + 15) / 16;
        act_m = tmp * 16;
    }

    float get_result(int x) {
        float* result = act_result_bo.map<float*>();
        return result[x];
    }

    int8_t get_result_q8(int x) {
        int8_t* result = act_bo.map<int8_t*>();
        return result[x];
    }

    float get_result_scale(int x) {
        uint16_t* result = act_scale_bo.map<uint16_t*>();
        return half_to_float(result[x]);
    }

    void print_all_m_weights_keys() {
        for (auto& [key, value] : m_weights) {
            printf("weight key: %s\n", key.c_str());
        }
    }

private:
    xrt::device device;
    xrt::bo weight_bo;
    xrt::bo weight_scale_bo;
    xrt::bo act_bo;
    xrt::bo act_scale_bo;
    xrt::bo act_result_bo;
    xrt::bo act_decode_bo;
    xrt::bo act_scale_decode_bo;
    xrt::bo act_result_decode_bo;

    std::unordered_map<std::string, WeightSlot> m_weights;
    std::vector<ActSlot> m_act_slots;
    bool m_weights_loaded = false;
    size_t m_weight_data_used = 0;
    size_t m_weight_scale_used = 0;

    int act_l = 0;
    int act_m = 0;

    int m_fd = -1;
    volatile uint32_t* m_base = nullptr;
    size_t m_mapSize = 0;

    void shutdown_control() {
        if (m_base && m_base != MAP_FAILED) {
            auto* p = const_cast<uint32_t*>(m_base);
            munmap(static_cast<void*>(p), m_mapSize);
            m_base = nullptr;
        }
        if (m_fd >= 0) {
            close(m_fd);
            m_fd = -1;
        }
    }

    inline void wr(uint32_t off, uint32_t val) const {
        if (!m_base)
            return;
        m_base[off / 4] = val;
    }
    inline uint32_t rd(uint32_t off) const {
        if (!m_base)
            return 0;
        return m_base[off / 4];
    }

    inline void wr64(uint32_t off, uint64_t val) const {
        if (!m_base)
            return;
        m_base[off / 4]     = static_cast<uint32_t>(val);
        m_base[off / 4 + 1] = static_cast<uint32_t>(val >> 32);
    }
    inline uint64_t rd64(uint32_t off) const {
        if (!m_base)
            return 0;
        uint64_t lo = m_base[off / 4];
        uint64_t hi = m_base[off / 4 + 1];
        return lo | (hi << 32);
    }
};

#endif // FPGA_DRIVER_H