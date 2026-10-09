#define _GNU_SOURCE
#include <dirent.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_BASE      0x93000000ull
#define DEFAULT_MAP_SIZE  0x1000u
#define EXPECTED_MAGIC    0x0000004du

#define REG_TOKEN_CMD     0x00u
#define REG_STATUS        0x04u
#define REG_PREFILL       0x08u
#define REG_TEST          0x0cu
#define REG_DEST_TOKEN    0x10u
#define REG_LAYER         0x20u
#define REG_CMD_SEL       0x24u
#define REG_MAGIC         0x44u
#define REG_ARGMAX_CLEAR  0x80u
#define REG_SOFT_RESET    0xc0u

#define TOKEN_VLD         (1u << 16)
#define TOKEN_PREFILL     (1u << 17)
#define TOKEN_PREFILL_LAST (1u << 18)
#define TOKEN_DECODE      (1u << 19)

struct options {
    uint64_t base;
    size_t map_size;
    const char *uio_path;
    bool use_devmem;
    bool list_uio;
    bool soft_reset;
    bool token_smoke;
    uint32_t token;
    uint32_t token_mode;
    bool dest_token_valid;
    uint32_t dest_token;
    int poll_ms;
};

struct mapping {
    int fd;
    void *map;
    size_t map_len;
    volatile uint32_t *regs;
    const char *method;
};

static void usage(const char *argv0)
{
    printf("Usage: %s [options]\n", argv0);
    printf("\n");
    printf("Default: auto-find /dev/uioX mapped at 0x%08" PRIx64 " and run safe AXI-Lite smoke.\n",
           (uint64_t)DEFAULT_BASE);
    printf("\nOptions:\n");
    printf("  --base ADDR          Control base address, default 0x%08" PRIx64 "\n",
           (uint64_t)DEFAULT_BASE);
    printf("  --map-size SIZE      Mapping size, default 0x%x\n", DEFAULT_MAP_SIZE);
    printf("  --uio PATH           Use a specific /dev/uioX device\n");
    printf("  --devmem             Use /dev/mem instead of UIO\n");
    printf("  --list-uio           Print UIO map0 addresses and exit\n");
    printf("  --soft-reset         Pulse SOFT_RESET before register tests\n");
    printf("  --token-smoke TOKEN  Issue one token command after safe tests\n");
    printf("  --prefill            Token command mode: tokenVld + isPrefillToken\n");
    printf("  --prefill-last       Token command mode: tokenVld + isPrefillLastToken\n");
    printf("  --decode             Token command mode: tokenVld + isDecodeToken\n");
    printf("  --dest-token N       Write DEST_TOKEN_CNT before token smoke\n");
    printf("  --poll-ms N          Token smoke poll timeout, default 1000 ms\n");
    printf("  --help               Show this help\n");
}

static bool parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 0);
    while (end && *end && isspace((unsigned char)*end)) {
        end++;
    }
    if (errno || end == s || (end && *end != '\0')) {
        return false;
    }
    *out = (uint64_t)v;
    return true;
}

static bool parse_u32(const char *s, uint32_t *out)
{
    uint64_t v = 0;
    if (!parse_u64(s, &v) || v > UINT32_MAX) {
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

static int read_text_file(const char *path, char *buf, size_t cap)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    ssize_t n = read(fd, buf, cap - 1);
    int saved = errno;
    close(fd);
    errno = saved;
    if (n < 0) {
        return -1;
    }
    buf[n] = '\0';
    return 0;
}

static void list_uio_devices(void)
{
    DIR *dir = opendir("/sys/class/uio");
    if (!dir) {
        printf("No /sys/class/uio: %s\n", strerror(errno));
        return;
    }

    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        if (strncmp(de->d_name, "uio", 3) != 0) {
            continue;
        }
        char addr_path[PATH_MAX];
        char size_path[PATH_MAX];
        char name_path[PATH_MAX];
        char addr_buf[128] = {0};
        char size_buf[128] = {0};
        char name_buf[128] = {0};

        snprintf(addr_path, sizeof(addr_path), "/sys/class/uio/%s/maps/map0/addr", de->d_name);
        snprintf(size_path, sizeof(size_path), "/sys/class/uio/%s/maps/map0/size", de->d_name);
        snprintf(name_path, sizeof(name_path), "/sys/class/uio/%s/name", de->d_name);
        read_text_file(addr_path, addr_buf, sizeof(addr_buf));
        read_text_file(size_path, size_buf, sizeof(size_buf));
        read_text_file(name_path, name_buf, sizeof(name_buf));
        printf("/dev/%s addr=%s size=%s name=%s",
               de->d_name,
               addr_buf[0] ? addr_buf : "?",
               size_buf[0] ? size_buf : "?",
               name_buf[0] ? name_buf : "?\n");
        if (name_buf[0] && name_buf[strlen(name_buf) - 1] != '\n') {
            printf("\n");
        }
    }
    closedir(dir);
}

static int find_uio_for_base(uint64_t base, char *out_path, size_t out_cap)
{
    DIR *dir = opendir("/sys/class/uio");
    if (!dir) {
        return -1;
    }

    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        if (strncmp(de->d_name, "uio", 3) != 0) {
            continue;
        }

        char addr_path[PATH_MAX];
        char addr_buf[128] = {0};
        uint64_t addr = 0;
        snprintf(addr_path, sizeof(addr_path), "/sys/class/uio/%s/maps/map0/addr", de->d_name);
        if (read_text_file(addr_path, addr_buf, sizeof(addr_buf)) != 0) {
            continue;
        }
        if (!parse_u64(addr_buf, &addr)) {
            continue;
        }
        if (addr == base) {
            snprintf(out_path, out_cap, "/dev/%s", de->d_name);
            closedir(dir);
            return 0;
        }
    }

    closedir(dir);
    return -1;
}

static int map_uio(const char *path, size_t map_size, struct mapping *m)
{
    memset(m, 0, sizeof(*m));
    m->fd = open(path, O_RDWR | O_SYNC);
    if (m->fd < 0) {
        fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
        return -1;
    }

    m->map_len = map_size;
    m->map = mmap(NULL, m->map_len, PROT_READ | PROT_WRITE, MAP_SHARED, m->fd, 0);
    if (m->map == MAP_FAILED) {
        fprintf(stderr, "mmap %s failed: %s\n", path, strerror(errno));
        close(m->fd);
        return -1;
    }

    m->regs = (volatile uint32_t *)m->map;
    m->method = path;
    return 0;
}

static int map_devmem(uint64_t base, size_t map_size, struct mapping *m)
{
    memset(m, 0, sizeof(*m));
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        page_size = 4096;
    }

    uint64_t page_mask = (uint64_t)page_size - 1u;
    off_t page_base = (off_t)(base & ~page_mask);
    size_t page_off = (size_t)(base - (uint64_t)page_base);
    size_t raw_len = page_off + map_size;
    size_t map_len = (raw_len + (size_t)page_size - 1u) & ~((size_t)page_size - 1u);

    m->fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (m->fd < 0) {
        fprintf(stderr, "open /dev/mem failed: %s\n", strerror(errno));
        return -1;
    }

    m->map_len = map_len;
    m->map = mmap(NULL, m->map_len, PROT_READ | PROT_WRITE, MAP_SHARED, m->fd, page_base);
    if (m->map == MAP_FAILED) {
        fprintf(stderr, "mmap /dev/mem failed: %s\n", strerror(errno));
        close(m->fd);
        return -1;
    }

    m->regs = (volatile uint32_t *)((uint8_t *)m->map + page_off);
    m->method = "/dev/mem";
    return 0;
}

static void unmap_regs(struct mapping *m)
{
    if (m->map && m->map != MAP_FAILED) {
        munmap(m->map, m->map_len);
    }
    if (m->fd >= 0) {
        close(m->fd);
    }
}

static uint32_t reg_read(volatile uint32_t *regs, uint32_t off)
{
    uint32_t v = regs[off >> 2];
    __sync_synchronize();
    return v;
}

static void reg_write(volatile uint32_t *regs, uint32_t off, uint32_t v)
{
    regs[off >> 2] = v;
    __sync_synchronize();
}

static void print_status(volatile uint32_t *regs, const char *tag)
{
    uint32_t status = reg_read(regs, REG_STATUS);
    uint32_t prefill = reg_read(regs, REG_PREFILL);
    uint32_t layer = reg_read(regs, REG_LAYER);
    uint32_t timer = reg_read(regs, REG_DEST_TOKEN);
    uint32_t token_count = status & 0xffffu;
    uint32_t argmax = (status >> 16) & 0x7fffu;
    uint32_t argmax_valid = (status >> 31) & 0x1u;

    printf("%s STATUS=0x%08x token_count=%u argmax=%u argmax_valid=%u\n",
           tag, status, token_count, argmax, argmax_valid);
    printf("%s PREFILL=0x%08x LAYER=0x%08x TIMER/DEST=0x%08x\n",
           tag, prefill, layer, timer);
}

static int run_safe_tests(volatile uint32_t *regs, bool soft_reset)
{
    int errors = 0;
    printf("Running safe AXI-Lite register smoke\n");

    if (soft_reset) {
        printf("Pulsing SOFT_RESET\n");
        reg_write(regs, REG_SOFT_RESET, 1u);
        usleep(1000);
    }

    uint32_t magic = reg_read(regs, REG_MAGIC);
    printf("MAGIC=0x%08x expected=0x%08x\n", magic, EXPECTED_MAGIC);
    if (magic != EXPECTED_MAGIC) {
        fprintf(stderr, "ERROR: magic mismatch\n");
        errors++;
    }

    const uint32_t patterns[] = {
        0x00000000u,
        0x0000004du,
        0x12345678u,
        0xa5a55a5au,
    };
    for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); i++) {
        reg_write(regs, REG_TEST, patterns[i]);
        uint32_t got = reg_read(regs, REG_TEST);
        printf("TEST_REG write=0x%08x read=0x%08x\n", patterns[i], got);
        if (got != patterns[i]) {
            fprintf(stderr, "ERROR: TEST_REG mismatch at pattern %zu\n", i);
            errors++;
        }
    }

    print_status(regs, "before");

    for (uint32_t cmd_sel = 0; cmd_sel < 4; cmd_sel++) {
        reg_write(regs, REG_CMD_SEL, cmd_sel);
        usleep(100);
        printf("CMD_SEL write=%u\n", cmd_sel);
    }

    reg_write(regs, REG_ARGMAX_CLEAR, 1u);
    usleep(100);
    print_status(regs, "after ");

    return errors ? -1 : 0;
}

static int run_token_smoke(volatile uint32_t *regs, const struct options *opt)
{
    uint32_t cmd = (opt->token & 0xffffu) | TOKEN_VLD | opt->token_mode;

    if (opt->dest_token_valid) {
        printf("Writing DEST_TOKEN_CNT=0x%08x\n", opt->dest_token & 0xffffu);
        reg_write(regs, REG_DEST_TOKEN, opt->dest_token & 0xffffu);
    }

    printf("Issuing token command 0x%08x token=%u mode=0x%08x\n",
           cmd, opt->token & 0xffffu, opt->token_mode);
    reg_write(regs, REG_ARGMAX_CLEAR, 1u);
    usleep(100);
    reg_write(regs, REG_TOKEN_CMD, cmd);

    int elapsed = 0;
    while (elapsed <= opt->poll_ms) {
        uint32_t status = reg_read(regs, REG_STATUS);
        if (status & 0x80000000u) {
            print_status(regs, "token ");
            return 0;
        }
        usleep(1000);
        elapsed++;
    }

    print_status(regs, "token ");
    fprintf(stderr, "ERROR: token smoke timed out after %d ms without argmax valid\n",
            opt->poll_ms);
    return -1;
}

static int parse_args(int argc, char **argv, struct options *opt)
{
    opt->base = DEFAULT_BASE;
    opt->map_size = DEFAULT_MAP_SIZE;
    opt->uio_path = NULL;
    opt->use_devmem = false;
    opt->list_uio = false;
    opt->soft_reset = false;
    opt->token_smoke = false;
    opt->token = 0;
    opt->token_mode = TOKEN_PREFILL;
    opt->dest_token_valid = false;
    opt->dest_token = 0;
    opt->poll_ms = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) {
            if (!parse_u64(argv[++i], &opt->base)) {
                fprintf(stderr, "Invalid --base value\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--map-size") == 0 && i + 1 < argc) {
            uint64_t v = 0;
            if (!parse_u64(argv[++i], &v) || v == 0 || v > (1ull << 30)) {
                fprintf(stderr, "Invalid --map-size value\n");
                return -1;
            }
            opt->map_size = (size_t)v;
        } else if (strcmp(argv[i], "--uio") == 0 && i + 1 < argc) {
            opt->uio_path = argv[++i];
        } else if (strcmp(argv[i], "--devmem") == 0) {
            opt->use_devmem = true;
        } else if (strcmp(argv[i], "--list-uio") == 0) {
            opt->list_uio = true;
        } else if (strcmp(argv[i], "--soft-reset") == 0) {
            opt->soft_reset = true;
        } else if (strcmp(argv[i], "--token-smoke") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt->token)) {
                fprintf(stderr, "Invalid --token-smoke value\n");
                return -1;
            }
            opt->token_smoke = true;
        } else if (strcmp(argv[i], "--prefill") == 0) {
            opt->token_mode = TOKEN_PREFILL;
        } else if (strcmp(argv[i], "--prefill-last") == 0) {
            opt->token_mode = TOKEN_PREFILL_LAST;
        } else if (strcmp(argv[i], "--decode") == 0) {
            opt->token_mode = TOKEN_DECODE;
        } else if (strcmp(argv[i], "--dest-token") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &opt->dest_token)) {
                fprintf(stderr, "Invalid --dest-token value\n");
                return -1;
            }
            opt->dest_token_valid = true;
        } else if (strcmp(argv[i], "--poll-ms") == 0 && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_u32(argv[++i], &v) || v > 3600000u) {
                fprintf(stderr, "Invalid --poll-ms value\n");
                return -1;
            }
            opt->poll_ms = (int)v;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            return -1;
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    struct options opt;
    if (parse_args(argc, argv, &opt) != 0) {
        return 2;
    }

    if (opt.list_uio) {
        list_uio_devices();
        return 0;
    }

    printf("Control base=0x%08" PRIx64 " map_size=0x%zx\n", opt.base, opt.map_size);

    struct mapping m;
    int rc = -1;
    char auto_uio[PATH_MAX];

    if (opt.use_devmem) {
        rc = map_devmem(opt.base, opt.map_size, &m);
    } else {
        const char *uio = opt.uio_path;
        if (!uio) {
            if (find_uio_for_base(opt.base, auto_uio, sizeof(auto_uio)) == 0) {
                uio = auto_uio;
            }
        }
        if (uio) {
            rc = map_uio(uio, opt.map_size, &m);
        }
        if (rc != 0) {
            fprintf(stderr, "UIO mapping unavailable; trying /dev/mem fallback\n");
            rc = map_devmem(opt.base, opt.map_size, &m);
        }
    }

    if (rc != 0) {
        return 1;
    }

    printf("Mapped via %s\n", m.method);
    rc = run_safe_tests(m.regs, opt.soft_reset);
    if (rc == 0 && opt.token_smoke) {
        rc = run_token_smoke(m.regs, &opt);
    }

    unmap_regs(&m);
    if (rc == 0) {
        printf("PASS\n");
        return 0;
    }

    printf("FAIL\n");
    return 1;
}
