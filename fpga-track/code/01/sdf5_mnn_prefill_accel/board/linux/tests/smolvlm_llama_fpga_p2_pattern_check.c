// SPDX-License-Identifier: MIT
// Stronger P2 smoke for the SDF1 debug ABI: verify one-layer mode copies
// a non-zero 960-byte activation pattern through the PL AXI master path.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/ioctl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define REG_BASE_DEFAULT 0x96000000ULL
#define REG_MAP_SIZE 0x1000UL

#define REG_MAGIC 0x044U
#define REG_MODEL_BASE_LO 0x100U
#define REG_ACT_BASE_LO 0x108U
#define REG_KV_BASE_LO 0x110U
#define REG_CTX_LENGTH 0x118U
#define REG_TARGET_MODE 0x11cU
#define REG_DBG_MAGIC 0x154U
#define REG_DBG_VERSION 0x158U
#define REG_DBG_CAPS 0x15cU
#define REG_DBG_CTRL 0x160U
#define REG_DBG_STATUS 0x164U
#define REG_DBG_SRC_LO 0x168U
#define REG_DBG_DST_LO 0x170U
#define REG_DBG_AUX_LO 0x178U
#define REG_DBG_LEN 0x180U
#define REG_DBG_MODE 0x184U
#define REG_DBG_TAG 0x188U
#define REG_DBG_ERROR 0x18cU
#define REG_DBG_READ_BEATS 0x190U
#define REG_DBG_WRITE_BEATS 0x194U

#define AXI_MAGIC 0x0000004dU
#define DBG_MAGIC 0x53444631U
#define DBG_CAP_ONE_LAYER (1U << 3)
#define DBG_CTRL_START (1U << 0)
#define DBG_CTRL_CLEAR (1U << 1)
#define DBG_STATUS_BUSY (1U << 0)
#define DBG_STATUS_DONE (1U << 1)
#define DBG_STATUS_ERROR (1U << 2)
#define DBG_MODE_ONE_LAYER 0x0000e201U

#define P2_HIDDEN 960U
#define P2_KV_HEADS 5U
#define P2_HEAD_DIM 64U
#define P2_CTX 1024U
#define P2_MODEL_BYTES 21000000U
#define P2_KV_BYTES (2U * P2_KV_HEADS * P2_CTX * P2_HEAD_DIM * 2U)
#define P2_ACT_BYTES 32768U
#define P2_TOTAL_BYTES (P2_MODEL_BYTES + P2_KV_BYTES + P2_ACT_BYTES + 0x100000U)
#define P2_DST_OFFSET 0x4000U

#define SMOL_CMA_IOC_MAGIC 's'

struct smol_cma_info {
    uint64_t dma_addr;
    uint64_t size;
    uint32_t handle;
    uint32_t flags;
};

#define SMOL_CMA_IOC_GET_BUFFER _IOR(SMOL_CMA_IOC_MAGIC, 0x01, struct smol_cma_info)

static volatile uint32_t *regs32;

static uint32_t rd32(uint32_t off)
{
    return regs32[off >> 2];
}

static void wr32(uint32_t off, uint32_t value)
{
    regs32[off >> 2] = value;
}

static void wr64(uint32_t off_lo, uint64_t value)
{
    wr32(off_lo, (uint32_t)value);
    wr32(off_lo + 4U, (uint32_t)(value >> 32));
}

static uint64_t ms_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static int wait_done(unsigned timeout_ms)
{
    uint64_t deadline = ms_now() + timeout_ms;
    while (ms_now() < deadline) {
        uint32_t status = rd32(REG_DBG_STATUS);
        if (status & DBG_STATUS_ERROR) {
            return -2;
        }
        if ((status & DBG_STATUS_DONE) && !(status & DBG_STATUS_BUSY)) {
            return 0;
        }
        usleep(1000);
    }
    return -1;
}

int main(int argc, char **argv)
{
    uint64_t base = REG_BASE_DEFAULT;
    const char *cma_path = "/dev/smolvlm_llama_fpga_cma";
    int mem_fd = -1;
    int cma_fd = -1;
    void *reg_map = MAP_FAILED;
    uint8_t *cma = MAP_FAILED;
    struct smol_cma_info info;
    int rc = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) {
            base = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--cma-dev") == 0 && i + 1 < argc) {
            cma_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--base 0x96000000] [--cma-dev /dev/smolvlm_llama_fpga_cma]\n", argv[0]);
            return 2;
        }
    }

    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("open /dev/mem");
        goto out;
    }
    reg_map = mmap(NULL, REG_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, (off_t)base);
    if (reg_map == MAP_FAILED) {
        perror("mmap regs");
        goto out;
    }
    regs32 = (volatile uint32_t *)reg_map;

    printf("axi_magic=0x%08x\n", rd32(REG_MAGIC));
    if (rd32(REG_MAGIC) != AXI_MAGIC) {
        fprintf(stderr, "bad AXI-Lite magic\n");
        goto out;
    }

    printf("debug magic=0x%08x version=0x%08x caps=0x%08x\n",
           rd32(REG_DBG_MAGIC), rd32(REG_DBG_VERSION), rd32(REG_DBG_CAPS));
    if (rd32(REG_DBG_MAGIC) != DBG_MAGIC || !(rd32(REG_DBG_CAPS) & DBG_CAP_ONE_LAYER)) {
        fprintf(stderr, "SDF1 one-layer capability missing\n");
        goto out;
    }

    cma_fd = open(cma_path, O_RDWR | O_SYNC);
    if (cma_fd < 0) {
        perror("open cma");
        goto out;
    }
    memset(&info, 0, sizeof(info));
    if (ioctl(cma_fd, SMOL_CMA_IOC_GET_BUFFER, &info) != 0) {
        perror("ioctl cma");
        goto out;
    }
    printf("cma dma=0x%016" PRIx64 " size=%" PRIu64 "\n", info.dma_addr, info.size);
    if (info.size < P2_TOTAL_BYTES) {
        fprintf(stderr, "CMA arena too small: need %u have %" PRIu64 "\n", P2_TOTAL_BYTES, info.size);
        goto out;
    }
    cma = mmap(NULL, info.size, PROT_READ | PROT_WRITE, MAP_SHARED, cma_fd, 0);
    if (cma == MAP_FAILED) {
        perror("mmap cma");
        goto out;
    }

    const uint64_t model_base = info.dma_addr;
    const uint64_t kv_base = model_base + P2_MODEL_BYTES;
    const uint64_t act_base = kv_base + P2_KV_BYTES;
    const uint64_t dst_base = act_base + P2_DST_OFFSET;
    const size_t act_off = (size_t)(act_base - info.dma_addr);
    const size_t dst_off = (size_t)(dst_base - info.dma_addr);

    memset(cma + act_off, 0, P2_ACT_BYTES);
    for (unsigned i = 0; i < P2_HIDDEN; ++i) {
        cma[act_off + i] = (uint8_t)((i * 37U + 11U) & 0xffU);
    }
    memset(cma + dst_off, 0xcc, P2_HIDDEN);

    wr64(REG_MODEL_BASE_LO, model_base);
    wr64(REG_KV_BASE_LO, kv_base);
    wr64(REG_ACT_BASE_LO, act_base);
    wr32(REG_CTX_LENGTH, P2_CTX);
    wr32(REG_TARGET_MODE, DBG_MODE_ONE_LAYER);

    wr64(REG_DBG_SRC_LO, act_base);
    wr64(REG_DBG_DST_LO, dst_base);
    wr64(REG_DBG_AUX_LO, model_base);
    wr32(REG_DBG_LEN, P2_HIDDEN);
    wr32(REG_DBG_MODE, DBG_MODE_ONE_LAYER);
    wr32(REG_DBG_TAG, 0x5252U);
    wr32(REG_DBG_CTRL, DBG_CTRL_CLEAR);
    wr32(REG_DBG_CTRL, DBG_CTRL_START);

    int wait_rc = wait_done(30000);
    if (wait_rc != 0) {
        fprintf(stderr, "debug wait failed rc=%d status=0x%08x error=0x%08x\n",
                wait_rc, rd32(REG_DBG_STATUS), rd32(REG_DBG_ERROR));
        goto out;
    }

    unsigned mismatches = 0;
    for (unsigned i = 0; i < P2_HIDDEN; ++i) {
        if (cma[dst_off + i] != cma[act_off + i]) {
            if (mismatches < 8) {
                fprintf(stderr, "mismatch[%u] got=0x%02x exp=0x%02x\n",
                        i, cma[dst_off + i], cma[act_off + i]);
            }
            ++mismatches;
        }
    }

    printf("{\"test\":\"p2_one_layer_pattern_check\",\"status\":\"%s\",\"bytes\":%u,"
           "\"mismatches\":%u,\"read_beats\":%u,\"write_beats\":%u,"
           "\"src\":\"0x%016" PRIx64 "\",\"dst\":\"0x%016" PRIx64 "\"}\n",
           mismatches ? "fail" : "pass",
           P2_HIDDEN,
           mismatches,
           rd32(REG_DBG_READ_BEATS),
           rd32(REG_DBG_WRITE_BEATS),
           act_base,
           dst_base);
    rc = mismatches ? 1 : 0;

out:
    if (cma != MAP_FAILED) {
        munmap(cma, info.size);
    }
    if (reg_map != MAP_FAILED) {
        munmap(reg_map, REG_MAP_SIZE);
    }
    if (cma_fd >= 0) {
        close(cma_fd);
    }
    if (mem_fd >= 0) {
        close(mem_fd);
    }
    return rc;
}
