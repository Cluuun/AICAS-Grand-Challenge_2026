#define _POSIX_C_SOURCE 200809L

// Board smoke for the SDF2/SDF3 LINEAR_INT8_BATCH debug ABI.
//
// Layout:
//   activation: int8 row-major M x K, stride rounded to 16 bytes
//   weight:     int8 row-major N x K, stride rounded to 16 bytes
//   output:     int32 row-major M x N

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define REG_BASE_DEFAULT 0x96000000ULL
#define REG_SPAN 0x1000U
#define CMA_DEV_DEFAULT "/dev/smolvlm_llama_fpga_cma"

#define DBG_MAGIC_SDF2 0x53444632U
#define DBG_MAGIC_SDF3 0x53444633U
#define DBG_MAGIC_SDF4 0x53444634U
#define DBG_CAPS_ECHO (1U << 0)
#define DBG_CAPS_LINEAR_BATCH (1U << 4)
#define MODE_ECHO 0x0000e001U
#define MODE_LINEAR_Q8 0x0000e101U
#define MODE_LINEAR_BATCH 0x0000e301U

#define REG_DBG_MAGIC 0x154
#define REG_DBG_CAPS 0x15c
#define REG_DBG_CTRL 0x160
#define REG_DBG_STATUS 0x164
#define REG_DBG_SRC_LO 0x168
#define REG_DBG_SRC_HI 0x16c
#define REG_DBG_DST_LO 0x170
#define REG_DBG_DST_HI 0x174
#define REG_DBG_AUX_LO 0x178
#define REG_DBG_AUX_HI 0x17c
#define REG_DBG_LEN 0x180
#define REG_DBG_MODE 0x184
#define REG_DBG_ERROR 0x18c
#define REG_DBG_READ_BEATS 0x190
#define REG_DBG_WRITE_BEATS 0x194
#define REG_DBG_READ_BURSTS 0x198
#define REG_DBG_WRITE_BURSTS 0x19c
#define REG_DESC_M 0x1a0
#define REG_DESC_K 0x1a4
#define REG_DESC_N 0x1a8
#define REG_DESC_ACT_STRIDE 0x1ac
#define REG_DESC_WEIGHT_STRIDE 0x1b0
#define REG_DESC_DST_STRIDE 0x1b4
#define REG_CTR_CYCLES 0x148
#define REG_CTR_TILES 0x14c
#define REG_CTR_MAC_LO 0x150
#define REG_CTR_MAC_HI 0x1c4
#define REG_CTR_COMPUTE_CYCLES 0x1c8
#define REG_CTR_LOAD_CYCLES 0x1cc
#define REG_CTR_STORE_CYCLES 0x1d0
#define REG_CTR_STALL_CYCLES 0x1d4

#define SMOL_CMA_IOC_MAGIC 's'
struct smol_cma_info {
    uint64_t dma_addr;
    uint64_t size;
    uint32_t handle;
    uint32_t flags;
};
#define SMOL_CMA_IOC_GET_BUFFER _IOR(SMOL_CMA_IOC_MAGIC, 0x01, struct smol_cma_info)

static volatile uint32_t *regs;
static int g_quick_verify = 0;

static inline uint32_t reg_read(uint32_t off) {
    return regs[off >> 2];
}

static inline void reg_write(uint32_t off, uint32_t value) {
    regs[off >> 2] = value;
}

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int8_t pattern_s8(uint32_t x) {
    return (int8_t)((int)(x * 17U + 13U) % 17 - 8);
}

static uint32_t round16(uint32_t x) {
    return (x + 15U) & ~15U;
}

static void wr64(uint32_t off_lo, uint64_t value) {
    reg_write(off_lo, (uint32_t)value);
    reg_write(off_lo + 4U, (uint32_t)(value >> 32));
}

static int wait_done(uint32_t *status_out, uint64_t *ns_out) {
    uint64_t t0 = ns_now();
    uint32_t status = 0;
    for (uint32_t i = 0; i < 100000000U; ++i) {
        status = reg_read(REG_DBG_STATUS);
        if (status & 0x6U) break;
    }
    uint64_t t1 = ns_now();
    if (status_out) *status_out = status;
    if (ns_out) *ns_out = t1 - t0;
    return (status & 0x2U) ? 0 : 1;
}

static void fill_i8(uint8_t *base, uint32_t rows, uint32_t cols, uint32_t stride, uint32_t salt) {
    for (uint32_t r = 0; r < rows; ++r) {
        memset(base + (size_t)r * stride, 0, stride);
        for (uint32_t c = 0; c < cols; ++c) {
            base[(size_t)r * stride + c] = (uint8_t)pattern_s8(r * 131U + c * 7U + salt);
        }
    }
}

static int32_t ref_dot(const uint8_t *act, const uint8_t *w, uint32_t mi, uint32_t ni, uint32_t k, uint32_t a_stride, uint32_t w_stride) {
    int32_t acc = 0;
    const int8_t *arow = (const int8_t *)(act + (size_t)mi * a_stride);
    const int8_t *wrow = (const int8_t *)(w + (size_t)ni * w_stride);
    for (uint32_t ki = 0; ki < k; ++ki) {
        acc += (int32_t)arow[ki] * (int32_t)wrow[ki];
    }
    return acc;
}

static int run_echo(uint8_t *arena, uint64_t dma_base, uint64_t arena_size, uint32_t bytes) {
    const uint64_t src_off = 0x000000;
    const uint64_t dst_off = 0x200000;
    if (bytes == 0 || dst_off + bytes > arena_size) {
        fprintf(stderr, "arena too small for echo bytes=%u size=0x%llx\n",
                bytes, (unsigned long long)arena_size);
        return 2;
    }

    uint8_t *src = arena + src_off;
    uint8_t *dst = arena + dst_off;
    for (uint32_t i = 0; i < bytes; ++i) {
        src[i] = (uint8_t)((i * 37U + 11U) & 0xffU);
        dst[i] = 0xa5U;
    }

    wr64(REG_DBG_SRC_LO, dma_base + src_off);
    wr64(REG_DBG_DST_LO, dma_base + dst_off);
    reg_write(REG_DBG_LEN, bytes);
    reg_write(REG_DBG_MODE, MODE_ECHO);
    reg_write(REG_DBG_CTRL, 0x2);
    reg_write(REG_DBG_CTRL, 0x1);

    uint32_t status = 0;
    uint64_t ns = 0;
    int wait_rc = wait_done(&status, &ns);
    if (wait_rc != 0) {
        fprintf(stderr, "echo failed bytes=%u status=0x%08x error=0x%08x\n",
                bytes, status, reg_read(REG_DBG_ERROR));
        return 3;
    }
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < bytes; ++i) {
        if (dst[i] != src[i]) {
            if (mismatches < 8) {
                fprintf(stderr, "echo mismatch i=%u got=0x%02x exp=0x%02x\n", i, dst[i], src[i]);
            }
            ++mismatches;
        }
    }
    const double ms = (double)ns / 1.0e6;
    const double mb_s = ms > 0.0 ? (double)bytes / (ms * 1000.0) : 0.0;
    printf("SDF_ECHO bytes=%u %s wall_ms=%.3f MB_s=%.3f read_beats=%u write_beats=%u read_bursts=%u write_bursts=%u cycles=%u\n",
           bytes, mismatches ? "FAIL" : "PASS", ms, mb_s,
           reg_read(REG_DBG_READ_BEATS), reg_read(REG_DBG_WRITE_BEATS),
           reg_read(REG_DBG_READ_BURSTS), reg_read(REG_DBG_WRITE_BURSTS),
           reg_read(REG_CTR_CYCLES));
    return mismatches ? 4 : 0;
}

static int run_case_mode(uint8_t *arena, uint64_t dma_base, uint64_t arena_size,
                         const char *name, uint32_t mode, uint32_t m, uint32_t k, uint32_t n) {
    const uint32_t a_stride = round16(k);
    const uint32_t w_stride = round16(k);
    const uint32_t d_stride = n * 4U;
    const uint64_t act_off = 0x000000;
    const uint64_t w_off = 0x400000;
    uint64_t dst_off = 0x800000;
    const uint64_t act_bytes = (uint64_t)m * a_stride;
    const uint64_t w_bytes = (uint64_t)n * w_stride;
    const uint64_t dst_bytes = (uint64_t)m * d_stride;
    const uint64_t w_end = w_off + w_bytes;
    if (dst_off < w_end) dst_off = (w_end + 4095ULL) & ~4095ULL;
    if (dst_off + dst_bytes > arena_size || w_off + w_bytes > arena_size || act_off + act_bytes > arena_size) {
        fprintf(stderr, "arena too small for M=%u K=%u N=%u need dst_end=0x%llx size=0x%llx\n",
                m, k, n, (unsigned long long)(dst_off + dst_bytes), (unsigned long long)arena_size);
        return 2;
    }

    uint8_t *act = arena + act_off;
    uint8_t *w = arena + w_off;
    uint8_t *dst = arena + dst_off;
    fill_i8(act, m, k, a_stride, 1);
    fill_i8(w, n, k, w_stride, 9);
    memset(dst, 0xa5, dst_bytes);

    const uint64_t act_dma = dma_base + act_off;
    const uint64_t w_dma = dma_base + w_off;
    const uint64_t dst_dma = dma_base + dst_off;
    wr64(REG_DBG_SRC_LO, act_dma);
    wr64(REG_DBG_AUX_LO, w_dma);
    wr64(REG_DBG_DST_LO, dst_dma);
    reg_write(REG_DBG_MODE, mode);
    reg_write(REG_DESC_M, m);
    reg_write(REG_DESC_K, k);
    reg_write(REG_DESC_N, n);
    reg_write(REG_DESC_ACT_STRIDE, a_stride);
    reg_write(REG_DESC_WEIGHT_STRIDE, w_stride);
    reg_write(REG_DESC_DST_STRIDE, d_stride);
    reg_write(REG_DBG_CTRL, 0x2);

    reg_write(REG_DBG_CTRL, 0x1);
    uint32_t status = 0;
    uint64_t ns = 0;
    (void)wait_done(&status, &ns);
    if (!(status & 0x2U)) {
        fprintf(stderr, "case %s mode=0x%08x M=%u K=%u N=%u failed status=0x%08x error=0x%08x\n",
                name, mode, m, k, n, status, reg_read(REG_DBG_ERROR));
        return 3;
    }

    uint32_t mismatches = 0;
    if (g_quick_verify) {
        const uint32_t sample_m[5] = {0, m > 1 ? 1 : 0, m / 2, m > 0 ? m - 1 : 0, m > 3 ? m - 3 : 0};
        const uint32_t sample_n[5] = {0, n > 1 ? 1 : 0, n / 2, n > 0 ? n - 1 : 0, n > 5 ? n - 5 : 0};
        for (uint32_t si = 0; si < 5; ++si) {
            uint32_t mi = sample_m[si];
            uint32_t ni = sample_n[si];
            int32_t got;
            memcpy(&got, dst + (size_t)mi * d_stride + ni * 4U, sizeof(got));
            int32_t exp = ref_dot(act, w, mi, ni, k, a_stride, w_stride);
            if (got != exp) {
                fprintf(stderr, "mismatch %s quick mode=0x%08x M=%u K=%u N=%u mi=%u ni=%u got=%d exp=%d\n",
                        name, mode, m, k, n, mi, ni, got, exp);
                ++mismatches;
            }
        }
    } else {
        for (uint32_t mi = 0; mi < m; ++mi) {
            for (uint32_t ni = 0; ni < n; ++ni) {
                int32_t got;
                memcpy(&got, dst + (size_t)mi * d_stride + ni * 4U, sizeof(got));
                int32_t exp = ref_dot(act, w, mi, ni, k, a_stride, w_stride);
                if (got != exp) {
                    if (mismatches < 8) {
                        fprintf(stderr, "mismatch %s mode=0x%08x M=%u K=%u N=%u mi=%u ni=%u got=%d exp=%d\n",
                                name, mode, m, k, n, mi, ni, got, exp);
                    }
                    ++mismatches;
                }
            }
        }
    }
    const uint64_t macs = (uint64_t)m * k * n;
    const double ms = (double)ns / 1.0e6;
    const double gmac_s_wall = ms > 0.0 ? (double)macs / (ms * 1.0e6) : 0.0;
    printf("SDF_LINEAR name=%s mode=0x%08x M=%u K=%u N=%u %s wall_ms=%.3f wall_gmac_s=%.3f cycles=%u load=%u compute=%u store=%u stall=%u tiles=%u mac_lo=%u mac_hi=%u read_beats=%u write_beats=%u read_bursts=%u write_bursts=%u\n",
           name, mode, m, k, n, mismatches ? "FAIL" : (g_quick_verify ? "PASS_QUICK" : "PASS"), ms, gmac_s_wall,
           reg_read(REG_CTR_CYCLES), reg_read(REG_CTR_LOAD_CYCLES),
           reg_read(REG_CTR_COMPUTE_CYCLES), reg_read(REG_CTR_STORE_CYCLES),
           reg_read(REG_CTR_STALL_CYCLES), reg_read(REG_CTR_TILES),
           reg_read(REG_CTR_MAC_LO), reg_read(REG_CTR_MAC_HI),
           reg_read(REG_DBG_READ_BEATS), reg_read(REG_DBG_WRITE_BEATS),
           reg_read(REG_DBG_READ_BURSTS), reg_read(REG_DBG_WRITE_BURSTS));
    fflush(stdout);
    return mismatches ? 4 : 0;
}

static int run_case(uint8_t *arena, uint64_t dma_base, uint64_t arena_size, uint32_t m, uint32_t k, uint32_t n) {
    return run_case_mode(arena, dma_base, arena_size, "batch", MODE_LINEAR_BATCH, m, k, n);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    uint64_t reg_base = REG_BASE_DEFAULT;
    const char *reg_path = "/dev/mem";
    const char *cma_dev = CMA_DEV_DEFAULT;
    if (argc >= 2) {
        if (argv[1][0] == '/') {
            reg_path = argv[1];
        } else {
            reg_base = strtoull(argv[1], NULL, 0);
        }
    }
    if (argc >= 3) cma_dev = argv[2];

    int mem_fd = open(reg_path, O_RDWR | O_SYNC);
    if (mem_fd < 0) { perror("open regs"); return 1; }
    off_t reg_off = (strcmp(reg_path, "/dev/mem") == 0) ? (off_t)reg_base : 0;
    regs = mmap(NULL, REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, reg_off);
    if (regs == MAP_FAILED) { perror("mmap regs"); return 1; }
    printf("regs=%s", reg_path);
    if (strcmp(reg_path, "/dev/mem") == 0) {
        printf("@0x%016" PRIx64, reg_base);
    }
    printf("\n");

    int cma_fd = open(cma_dev, O_RDWR | O_SYNC);
    if (cma_fd < 0) { perror("open cma"); return 1; }
    struct smol_cma_info info;
    if (ioctl(cma_fd, SMOL_CMA_IOC_GET_BUFFER, &info) != 0) { perror("ioctl cma"); return 1; }
    uint8_t *arena = mmap(NULL, info.size, PROT_READ | PROT_WRITE, MAP_SHARED, cma_fd, 0);
    if (arena == MAP_FAILED) { perror("mmap cma"); return 1; }

    uint32_t magic = reg_read(REG_DBG_MAGIC);
    uint32_t caps = reg_read(REG_DBG_CAPS);
    printf("magic=0x%08x caps=0x%08x dma=0x%016" PRIx64 " size=%" PRIu64 "\n", magic, caps, info.dma_addr, info.size);
    if (magic != DBG_MAGIC_SDF2 && magic != DBG_MAGIC_SDF3 && magic != DBG_MAGIC_SDF4) { fprintf(stderr, "bad magic\n"); return 1; }
    if (!(caps & DBG_CAPS_LINEAR_BATCH)) { fprintf(stderr, "linear batch cap missing\n"); return 1; }

    int rc = 0;
    if (argc >= 4 && strcmp(argv[3], "probe") == 0) {
        return 0;
    }

    if (argc >= 4 && strcmp(argv[3], "echo") == 0) {
        uint32_t bytes = argc >= 5 ? (uint32_t)strtoul(argv[4], NULL, 0) : 4096U;
        if (!(caps & DBG_CAPS_ECHO)) { fprintf(stderr, "echo cap missing\n"); return 1; }
        return run_echo(arena, info.dma_addr, info.size, bytes) ? 1 : 0;
    }

    if (argc >= 6) {
        uint32_t m = (uint32_t)strtoul(argv[3], NULL, 0);
        uint32_t k = (uint32_t)strtoul(argv[4], NULL, 0);
        uint32_t n = (uint32_t)strtoul(argv[5], NULL, 0);
        if (argc >= 7 && strcmp(argv[6], "quick") == 0) g_quick_verify = 1;
        return run_case(arena, info.dma_addr, info.size, m, k, n) ? 1 : 0;
    }

    if (argc >= 4 && strcmp(argv[3], "legacy") == 0) {
        return run_case_mode(arena, info.dma_addr, info.size, "legacy_p1", MODE_LINEAR_Q8, 1, 960, 16) ? 1 : 0;
    }

    rc |= run_case(arena, info.dma_addr, info.size, 16, 64, 16);
    rc |= run_case(arena, info.dma_addr, info.size, 1, 960, 16);
    rc |= run_case_mode(arena, info.dma_addr, info.size, "legacy_p1", MODE_LINEAR_Q8, 1, 960, 16);
    rc |= run_case(arena, info.dma_addr, info.size, 17, 65, 19);
    rc |= run_case(arena, info.dma_addr, info.size, 80, 960, 320);
    return rc ? 1 : 0;
}
