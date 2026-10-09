#define _POSIX_C_SOURCE 200809L

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

#define DBG_MAGIC_SDF4 0x53444634U
#define DBG_VERSION_SDF44 0x00040400U
#define DBG_CAPS_ECHO (1U << 0)
#define DBG_CAPS_LINEAR_BATCH (1U << 4)

#define MODE_ECHO 0x0000e001U
#define MODE_LINEAR_BATCH 0x0000e301U

#define REG_MAGIC0 0x000U
#define REG_DBG_MAGIC 0x154U
#define REG_DBG_VERSION 0x158U
#define REG_DBG_CAPS 0x15cU

#define REG_LAST_AWADDR 0x280U
#define REG_LAST_WDATA 0x290U
#define REG_LAST_WSTRB 0x2a0U
#define REG_LAST_WOFF 0x2b0U
#define REG_LAST_WHIT 0x2c0U
#define REG_LANE0 0x300U
#define REG_LANE1 0x304U
#define REG_LANE2 0x308U
#define REG_LANE3 0x30cU
#define REG_LANE0_MIRROR 0x340U
#define REG_LANE1_MIRROR 0x350U
#define REG_LANE2_MIRROR 0x360U
#define REG_LANE3_MIRROR 0x370U

#define REG_A_CTRL 0x400U
#define REG_A_STATUS 0x410U
#define REG_A_SRC_LO 0x420U
#define REG_A_SRC_HI 0x430U
#define REG_A_DST_LO 0x440U
#define REG_A_DST_HI 0x450U
#define REG_A_AUX_LO 0x460U
#define REG_A_AUX_HI 0x470U
#define REG_A_LEN 0x480U
#define REG_A_MODE 0x490U
#define REG_A_TAG 0x4a0U
#define REG_A_ERROR 0x4b0U
#define REG_A_M 0x500U
#define REG_A_K 0x510U
#define REG_A_N 0x520U
#define REG_A_ACT_STRIDE 0x530U
#define REG_A_WEIGHT_STRIDE 0x540U
#define REG_A_DST_STRIDE 0x550U
#define REG_A_FLAGS 0x560U

#define REG_READ_BEATS 0x190U
#define REG_WRITE_BEATS 0x194U
#define REG_READ_BURSTS 0x198U
#define REG_WRITE_BURSTS 0x19cU
#define REG_CTR_CYCLES 0x148U
#define REG_CTR_TILES 0x14cU
#define REG_CTR_MAC_LO 0x150U
#define REG_CTR_MAC_HI 0x1c4U
#define REG_CTR_COMPUTE 0x1c8U
#define REG_CTR_LOAD 0x1ccU
#define REG_CTR_STORE 0x1d0U
#define REG_CTR_STALL 0x1d4U

#define SMOL_CMA_IOC_MAGIC 's'
struct smol_cma_info {
    uint64_t dma_addr;
    uint64_t size;
    uint32_t handle;
    uint32_t flags;
};
#define SMOL_CMA_IOC_GET_BUFFER _IOR(SMOL_CMA_IOC_MAGIC, 0x01, struct smol_cma_info)

static volatile uint32_t *regs;

static inline uint32_t rd32(uint32_t off) {
    return regs[off >> 2];
}

static inline void wr32(uint32_t off, uint32_t value) {
    regs[off >> 2] = value;
}

static void wr64_alias(uint32_t lo_off, uint32_t hi_off, uint64_t value) {
    wr32(lo_off, (uint32_t)value);
    wr32(hi_off, (uint32_t)(value >> 32));
}

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint32_t round16(uint32_t x) {
    return (x + 15U) & ~15U;
}

static int8_t pattern_s8(uint32_t x) {
    return (int8_t)((int)(x * 17U + 13U) % 17 - 8);
}

static void fill_i8(uint8_t *base, uint32_t rows, uint32_t cols, uint32_t stride, uint32_t salt) {
    for (uint32_t r = 0; r < rows; ++r) {
        memset(base + (size_t)r * stride, 0, stride);
        for (uint32_t c = 0; c < cols; ++c) {
            base[(size_t)r * stride + c] = (uint8_t)pattern_s8(r * 131U + c * 7U + salt);
        }
    }
}

static int32_t ref_dot(const uint8_t *act, const uint8_t *w, uint32_t mi, uint32_t ni,
                       uint32_t k, uint32_t a_stride, uint32_t w_stride) {
    int32_t acc = 0;
    const int8_t *arow = (const int8_t *)(act + (size_t)mi * a_stride);
    const int8_t *wrow = (const int8_t *)(w + (size_t)ni * w_stride);
    for (uint32_t ki = 0; ki < k; ++ki) {
        acc += (int32_t)arow[ki] * (int32_t)wrow[ki];
    }
    return acc;
}

static int wait_done(uint32_t *status_out, uint64_t *ns_out) {
    uint64_t t0 = ns_now();
    uint32_t status = 0;
    for (uint32_t i = 0; i < 200000000U; ++i) {
        status = rd32(REG_A_STATUS);
        if (status & 0x6U) break;
    }
    uint64_t t1 = ns_now();
    if (status_out) *status_out = status;
    if (ns_out) *ns_out = t1 - t0;
    return (status & 0x2U) ? 0 : 1;
}

static int map_regs(uint64_t base, const char *uio_path, int use_devmem) {
    if (use_devmem) {
        int fd = open("/dev/mem", O_RDWR | O_SYNC);
        if (fd < 0) {
            fprintf(stderr, "open /dev/mem failed: %s\n", strerror(errno));
            return -1;
        }
        void *map = mmap(NULL, REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
        if (map == MAP_FAILED) {
            fprintf(stderr, "mmap /dev/mem base=0x%llx failed: %s\n",
                    (unsigned long long)base, strerror(errno));
            close(fd);
            return -1;
        }
        regs = (volatile uint32_t *)map;
        printf("REGMAP /dev/mem base=0x%llx\n", (unsigned long long)base);
        return 0;
    }

    char path_buf[64] = {0};
    const char *path = uio_path;
    if (!path) {
        for (int i = 0; i < 32; ++i) {
            char sys_path[128];
            char addr_buf[64] = {0};
            snprintf(sys_path, sizeof(sys_path), "/sys/class/uio/uio%d/maps/map0/addr", i);
            int fd = open(sys_path, O_RDONLY);
            if (fd < 0) continue;
            ssize_t n = read(fd, addr_buf, sizeof(addr_buf) - 1);
            close(fd);
            if (n <= 0) continue;
            uint64_t addr = strtoull(addr_buf, NULL, 0);
            if (addr == base) {
                snprintf(path_buf, sizeof(path_buf), "/dev/uio%d", i);
                path = path_buf;
                break;
            }
        }
    }
    if (!path) {
        fprintf(stderr, "could not find UIO for base 0x%llx\n", (unsigned long long)base);
        return -1;
    }
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
        return -1;
    }
    void *map = mmap(NULL, REG_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap %s failed: %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    regs = (volatile uint32_t *)map;
    printf("REGMAP %s base=0x%llx\n", path, (unsigned long long)base);
    return 0;
}

static int run_lane_echo(void) {
    const uint32_t vals[4] = {0x11112222U, 0x33334444U, 0x55556666U, 0x77778888U};
    wr32(REG_LANE0, vals[0]);
    wr32(REG_LANE1, vals[1]);
    wr32(REG_LANE2, vals[2]);
    wr32(REG_LANE3, vals[3]);
    uint32_t got[4] = {
        rd32(REG_LANE0_MIRROR),
        rd32(REG_LANE1_MIRROR),
        rd32(REG_LANE2_MIRROR),
        rd32(REG_LANE3_MIRROR)
    };
    printf("LANE_ECHO direct=%08x/%08x/%08x/%08x mirror=%08x/%08x/%08x/%08x last_aw=0x%08x last_wdata=0x%08x last_wstrb=0x%08x last_woff=0x%08x last_whit=0x%08x\n",
           rd32(REG_LANE0), rd32(REG_LANE1), rd32(REG_LANE2), rd32(REG_LANE3),
           got[0], got[1], got[2], got[3],
           rd32(REG_LAST_AWADDR), rd32(REG_LAST_WDATA), rd32(REG_LAST_WSTRB),
           rd32(REG_LAST_WOFF), rd32(REG_LAST_WHIT));
    if (got[0] != vals[0]) {
        return 1;
    }
    if (got[1] != vals[1] || got[2] != vals[2] || got[3] != vals[3]) {
        printf("WARN non-lane0 direct writes are not usable on this KV260 path; alias tests use only 16B-aligned offsets\n");
    }
    return 0;
}

static int run_echo(uint8_t *arena, uint64_t dma_base, uint64_t arena_size, uint32_t bytes) {
    const uint64_t src_off = 0x000000;
    const uint64_t dst_off = 0x200000;
    if (dst_off + bytes > arena_size) return 2;
    uint8_t *src = arena + src_off;
    uint8_t *dst = arena + dst_off;
    for (uint32_t i = 0; i < bytes; ++i) {
        src[i] = (uint8_t)((i * 37U + 11U) & 0xffU);
        dst[i] = 0xa5U;
    }
    wr64_alias(REG_A_SRC_LO, REG_A_SRC_HI, dma_base + src_off);
    wr64_alias(REG_A_DST_LO, REG_A_DST_HI, dma_base + dst_off);
    wr32(REG_A_LEN, bytes);
    wr32(REG_A_MODE, MODE_ECHO);
    wr32(REG_A_CTRL, 0x2);
    wr32(REG_A_CTRL, 0x1);
    uint32_t status = 0;
    uint64_t ns = 0;
    if (wait_done(&status, &ns)) {
        printf("ECHO bytes=%u FAIL status=0x%08x error=0x%08x\n", bytes, status, rd32(REG_A_ERROR));
        return 3;
    }
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < bytes; ++i) {
        if (dst[i] != src[i]) {
            if (mismatches < 8) printf("echo mismatch i=%u got=0x%02x exp=0x%02x\n", i, dst[i], src[i]);
            ++mismatches;
        }
    }
    double ms = (double)ns / 1.0e6;
    printf("ECHO bytes=%u %s wall_ms=%.3f read_beats=%u write_beats=%u read_bursts=%u write_bursts=%u cycles=%u\n",
           bytes, mismatches ? "FAIL" : "PASS", ms, rd32(REG_READ_BEATS), rd32(REG_WRITE_BEATS),
           rd32(REG_READ_BURSTS), rd32(REG_WRITE_BURSTS), rd32(REG_CTR_CYCLES));
    return mismatches ? 4 : 0;
}

static int run_linear(uint8_t *arena, uint64_t dma_base, uint64_t arena_size,
                      const char *name, uint32_t m, uint32_t k, uint32_t n, int quick) {
    const uint32_t a_stride = round16(k);
    const uint32_t w_stride = round16(k);
    const uint32_t d_stride = n * 4U;
    const uint64_t act_off = 0x000000;
    const uint64_t w_off = 0x400000;
    uint64_t dst_off = 0x800000;
    uint64_t act_bytes = (uint64_t)m * a_stride;
    uint64_t w_bytes = (uint64_t)n * w_stride;
    uint64_t dst_bytes = (uint64_t)m * d_stride;
    uint64_t w_end = w_off + w_bytes;
    if (dst_off < w_end) dst_off = (w_end + 4095ULL) & ~4095ULL;
    if (act_off + act_bytes > arena_size || w_off + w_bytes > arena_size || dst_off + dst_bytes > arena_size) {
        printf("LINEAR %s SKIP arena too small need=0x%llx size=0x%llx\n",
               name, (unsigned long long)(dst_off + dst_bytes), (unsigned long long)arena_size);
        return 0;
    }
    uint8_t *act = arena + act_off;
    uint8_t *w = arena + w_off;
    uint8_t *dst = arena + dst_off;
    fill_i8(act, m, k, a_stride, 1);
    fill_i8(w, n, k, w_stride, 9);
    memset(dst, 0xa5, dst_bytes);
    wr64_alias(REG_A_SRC_LO, REG_A_SRC_HI, dma_base + act_off);
    wr64_alias(REG_A_AUX_LO, REG_A_AUX_HI, dma_base + w_off);
    wr64_alias(REG_A_DST_LO, REG_A_DST_HI, dma_base + dst_off);
    wr32(REG_A_MODE, MODE_LINEAR_BATCH);
    wr32(REG_A_M, m);
    wr32(REG_A_K, k);
    wr32(REG_A_N, n);
    wr32(REG_A_ACT_STRIDE, a_stride);
    wr32(REG_A_WEIGHT_STRIDE, w_stride);
    wr32(REG_A_DST_STRIDE, d_stride);
    wr32(REG_A_FLAGS, quick ? 1U : 0U);
    wr32(REG_A_CTRL, 0x2);
    wr32(REG_A_CTRL, 0x1);
    uint32_t status = 0;
    uint64_t ns = 0;
    if (wait_done(&status, &ns)) {
        printf("LINEAR %s M=%u K=%u N=%u FAIL status=0x%08x error=0x%08x\n",
               name, m, k, n, status, rd32(REG_A_ERROR));
        return 3;
    }
    uint32_t mismatches = 0;
    uint32_t checks = 0;
    for (uint32_t mi = 0; mi < m; ++mi) {
        for (uint32_t ni = 0; ni < n; ++ni) {
            if (quick && !((mi < 2 && ni < 8) || (mi + 2 >= m && ni + 8 >= n) || ((mi * 131U + ni * 17U) % 257U == 0))) {
                continue;
            }
            int32_t got = ((int32_t *)(dst + (size_t)mi * d_stride))[ni];
            int32_t exp = ref_dot(act, w, mi, ni, k, a_stride, w_stride);
            ++checks;
            if (got != exp) {
                if (mismatches < 12) {
                    printf("linear mismatch %s m=%u n=%u got=%d exp=%d\n", name, mi, ni, got, exp);
                }
                ++mismatches;
            }
        }
    }
    double ms = (double)ns / 1.0e6;
    double gmac = ((double)m * (double)n * (double)k) / ((double)ns);
    uint64_t mac = (uint64_t)m * (uint64_t)n * (uint64_t)k;
    printf("LINEAR %s M=%u K=%u N=%u %s checks=%u wall_ms=%.3f GMAC_s=%.3f mac=%llu read_beats=%u write_beats=%u tiles=%u cycles=%u compute=%u load=%u store=%u stall=%u\n",
           name, m, k, n, mismatches ? "FAIL" : "PASS", checks, ms, gmac,
           (unsigned long long)mac, rd32(REG_READ_BEATS), rd32(REG_WRITE_BEATS),
           rd32(REG_CTR_TILES), rd32(REG_CTR_CYCLES), rd32(REG_CTR_COMPUTE),
           rd32(REG_CTR_LOAD), rd32(REG_CTR_STORE), rd32(REG_CTR_STALL));
    return mismatches ? 4 : 0;
}

int main(int argc, char **argv) {
    uint64_t base = REG_BASE_DEFAULT;
    const char *uio = NULL;
    int use_devmem = 0;
    int quick = 0;
    int stress = 0;
    int mnn_only = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--stress")) stress = 1;
        else if (!strcmp(argv[i], "--mnn-only")) mnn_only = 1;
        else if (!strcmp(argv[i], "--devmem")) use_devmem = 1;
        else if (!strcmp(argv[i], "--uio") && i + 1 < argc) uio = argv[++i];
        else if (!strcmp(argv[i], "--base") && i + 1 < argc) base = strtoull(argv[++i], NULL, 0);
        else {
            fprintf(stderr, "usage: %s [--devmem] [--uio /dev/uioX] [--base 0x96000000] [--quick] [--stress] [--mnn-only]\n", argv[0]);
            return 2;
        }
    }
    if (map_regs(base, uio, use_devmem)) return 1;
    uint32_t magic0 = rd32(REG_MAGIC0);
    uint32_t magic = rd32(REG_DBG_MAGIC);
    uint32_t version = rd32(REG_DBG_VERSION);
    uint32_t caps = rd32(REG_DBG_CAPS);
    printf("IDENT magic0=0x%08x dbg_magic_legacy=0x%08x version_legacy=0x%08x caps_legacy=0x%08x\n",
           magic0, magic, version, caps);
    if (magic0 != DBG_MAGIC_SDF4) {
        fprintf(stderr, "unexpected SDF4.4 lane0 magic\n");
        return 1;
    }
    if (magic != DBG_MAGIC_SDF4 || version != DBG_VERSION_SDF44 ||
        !(caps & DBG_CAPS_ECHO) || !(caps & DBG_CAPS_LINEAR_BATCH)) {
        printf("WARN legacy non-lane0 identity/caps are not reliable on this KV260 path; continuing with aligned alias ABI\n");
    }
    if (run_lane_echo()) {
        fprintf(stderr, "lane echo failed\n");
        return 1;
    }

    int cma_fd = open(CMA_DEV_DEFAULT, O_RDWR | O_SYNC);
    if (cma_fd < 0) {
        perror("open cma");
        return 1;
    }
    struct smol_cma_info info;
    if (ioctl(cma_fd, SMOL_CMA_IOC_GET_BUFFER, &info) != 0) {
        perror("ioctl cma");
        return 1;
    }
    uint8_t *arena = mmap(NULL, info.size, PROT_READ | PROT_WRITE, MAP_SHARED, cma_fd, 0);
    if (arena == MAP_FAILED) {
        perror("mmap cma");
        return 1;
    }
    printf("CMA dma=0x%016" PRIx64 " size=%" PRIu64 "\n", info.dma_addr, info.size);

    int rc = 0;
    if (!mnn_only) {
        rc |= run_echo(arena, info.dma_addr, info.size, 4096);
        rc |= run_echo(arena, info.dma_addr, info.size, 65536);
        rc |= run_linear(arena, info.dma_addr, info.size, "small_16x64x16", 16, 64, 16, 0);
        rc |= run_linear(arena, info.dma_addr, info.size, "legacy_1x960x16", 1, 960, 16, 0);
        rc |= run_linear(arena, info.dma_addr, info.size, "edge_17x65x19", 17, 65, 19, 0);
    }
    rc |= run_linear(arena, info.dma_addr, info.size, "mnn_80x960x320", 80, 960, 320, quick);
    if (stress || mnn_only) {
        rc |= run_linear(arena, info.dma_addr, info.size, "mnn_80x960x960", 80, 960, 960, 1);
        rc |= run_linear(arena, info.dma_addr, info.size, "mnn_80x960x2560", 80, 960, 2560, 1);
        rc |= run_linear(arena, info.dma_addr, info.size, "mnn_80x2560x960", 80, 2560, 960, 1);
        rc |= run_linear(arena, info.dma_addr, info.size, "vision_1024x768x768", 1024, 768, 768, 1);
        rc |= run_linear(arena, info.dma_addr, info.size, "vision_1024x3072x768", 1024, 3072, 768, 1);
        rc |= run_linear(arena, info.dma_addr, info.size, "vision_1024x768x3072", 1024, 768, 3072, 1);
    }
    printf("SDF4_4_ALIAS_TEST %s\n", rc ? "FAIL" : "PASS");
    return rc ? 1 : 0;
}
