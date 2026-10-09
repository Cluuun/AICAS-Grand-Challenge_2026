#include "VSdf4DebugAxi.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t MODE_ECHO = 0x0000e001u;
static constexpr uint32_t MODE_LINEAR_Q8 = 0x0000e101u;
static constexpr uint32_t MODE_LINEAR_BATCH = 0x0000e301u;
static int g_failures = 0;

struct Sim {
    VSdf4DebugAxi top;
    uint64_t now = 0;
    std::vector<uint8_t> mem;
    unsigned read_latency_cycles = 0;
    unsigned bresp_latency_cycles = 0;

    bool rvalid = false;
    uint64_t raddr = 0;
    unsigned rbeats_left = 0;
    bool read_pending = false;
    uint64_t pending_raddr = 0;
    unsigned pending_rbeats = 0;
    unsigned read_delay_left = 0;
    bool bvalid = false;
    bool b_pending = false;
    unsigned b_delay_left = 0;
    bool aw_seen = false;
    uint64_t awaddr = 0;
    unsigned awbeats_total = 0;
    unsigned wbeats_seen = 0;

    explicit Sim(size_t mem_size, unsigned read_latency = 0, unsigned bresp_latency = 0)
        : mem(mem_size, 0),
          read_latency_cycles(read_latency),
          bresp_latency_cycles(bresp_latency) {
        top.ACLK = 0;
        top.ARESETN = 0;
        top.S_AXI_AWADDR = 0;
        top.S_AXI_AWPROT = 0;
        top.S_AXI_AWVALID = 0;
        top.S_AXI_WDATA = 0;
        top.S_AXI_WSTRB = 0;
        top.S_AXI_WVALID = 0;
        top.S_AXI_BREADY = 0;
        top.S_AXI_ARADDR = 0;
        top.S_AXI_ARPROT = 0;
        top.S_AXI_ARVALID = 0;
        top.S_AXI_RREADY = 0;
    }

    void drive_mem_inputs() {
        top.M_AXI_ARREADY = 1;
        top.M_AXI_AWREADY = 1;
        top.M_AXI_WREADY = 1;
        top.M_AXI_RVALID = rvalid ? 1 : 0;
        top.M_AXI_RRESP = 0;
        top.M_AXI_RID = 0;
        top.M_AXI_RLAST = (rvalid && rbeats_left == 1) ? 1 : 0;
        for (int i = 0; i < 4; ++i) {
            uint32_t word = 0;
            if (rvalid) {
                uint64_t a = raddr + static_cast<uint64_t>(i) * 4;
                if (a + 4 > mem.size()) throw std::runtime_error("read out of memory");
                word = static_cast<uint32_t>(mem[a]) |
                       (static_cast<uint32_t>(mem[a + 1]) << 8) |
                       (static_cast<uint32_t>(mem[a + 2]) << 16) |
                       (static_cast<uint32_t>(mem[a + 3]) << 24);
            }
            top.M_AXI_RDATA[i] = word;
        }
        top.M_AXI_BVALID = bvalid ? 1 : 0;
        top.M_AXI_BRESP = 0;
        top.M_AXI_BID = 0;
    }

    void tick() {
        drive_mem_inputs();
        top.ACLK = 0;
        top.eval();
        now++;

        const bool ar_hs = top.M_AXI_ARVALID && top.M_AXI_ARREADY;
        const bool r_hs = top.M_AXI_RVALID && top.M_AXI_RREADY;
        const bool aw_hs = top.M_AXI_AWVALID && top.M_AXI_AWREADY;
        const bool w_hs = top.M_AXI_WVALID && top.M_AXI_WREADY;
        const bool b_hs = top.M_AXI_BVALID && top.M_AXI_BREADY;
        const uint64_t araddr_now = top.M_AXI_ARADDR;
        const unsigned arbeats_now = static_cast<unsigned>(top.M_AXI_ARLEN) + 1u;
        const uint64_t awaddr_now = top.M_AXI_AWADDR;
        const unsigned awbeats_now = static_cast<unsigned>(top.M_AXI_AWLEN) + 1u;
        std::array<uint32_t, 4> wdata_now{};
        for (int i = 0; i < 4; ++i) wdata_now[i] = top.M_AXI_WDATA[i];
        const uint16_t wstrb_now = top.M_AXI_WSTRB;
        const bool wlast_now = top.M_AXI_WLAST;

        top.ACLK = 1;
        top.eval();
        now++;

        if (r_hs) {
            if (rbeats_left <= 1) {
                rvalid = false;
                rbeats_left = 0;
            } else {
                raddr += 16;
                --rbeats_left;
                rvalid = true;
            }
        }
        if (b_hs) bvalid = false;
        if (ar_hs) {
            if (((araddr_now & 0xfffull) + static_cast<uint64_t>(arbeats_now) * 16ull) > 4096ull) {
                throw std::runtime_error("M_AXI read burst crosses 4KB boundary");
            }
            pending_raddr = araddr_now;
            pending_rbeats = arbeats_now;
            read_delay_left = read_latency_cycles;
            read_pending = true;
        }
        if (aw_hs) {
            if (((awaddr_now & 0xfffull) + static_cast<uint64_t>(awbeats_now) * 16ull) > 4096ull) {
                throw std::runtime_error("M_AXI write burst crosses 4KB boundary");
            }
            awaddr = awaddr_now;
            awbeats_total = awbeats_now;
            wbeats_seen = 0;
            aw_seen = true;
        }
        if (w_hs) {
            uint64_t base = aw_seen ? awaddr : awaddr_now;
            uint64_t addr = base + static_cast<uint64_t>(wbeats_seen) * 16;
            if (addr + 16 > mem.size()) throw std::runtime_error("write out of memory");
            for (int lane = 0; lane < 16; ++lane) {
                if (wstrb_now & (1u << lane)) {
                    uint32_t word = wdata_now[lane / 4];
                    mem[addr + lane] = static_cast<uint8_t>((word >> ((lane % 4) * 8)) & 0xffu);
                }
            }
            ++wbeats_seen;
            if (wlast_now || (aw_seen && wbeats_seen >= awbeats_total)) {
                aw_seen = false;
                b_delay_left = bresp_latency_cycles;
                b_pending = true;
            }
        }
        if (!rvalid && read_pending) {
            if (read_delay_left == 0) {
                raddr = pending_raddr;
                rbeats_left = pending_rbeats;
                rvalid = true;
                read_pending = false;
            } else {
                --read_delay_left;
            }
        }
        if (!bvalid && b_pending) {
            if (b_delay_left == 0) {
                bvalid = true;
                b_pending = false;
            } else {
                --b_delay_left;
            }
        }
    }

    void reset() {
        top.ARESETN = 0;
        for (int i = 0; i < 8; ++i) tick();
        top.ARESETN = 1;
        for (int i = 0; i < 4; ++i) tick();
    }

    void axil_write(uint32_t addr, uint32_t data) {
        axil_write_split(addr, data, 0);
    }

    void axil_write_aw_then_w(uint32_t addr, uint32_t data) {
        axil_write_split(addr, data, 4);
    }

    void axil_write_w_then_aw(uint32_t addr, uint32_t data) {
        axil_write_split(addr, data, -4);
    }

    void axil_write_split(uint32_t addr, uint32_t data, int aw_before_w_gap) {
        top.S_AXI_AWADDR = addr;
        top.S_AXI_WDATA = data;
        top.S_AXI_WSTRB = 0xf;
        top.S_AXI_AWVALID = aw_before_w_gap >= 0 ? 1 : 0;
        top.S_AXI_WVALID = aw_before_w_gap <= 0 ? 1 : 0;
        bool aw_done = false, w_done = false;
        int gap = std::abs(aw_before_w_gap);
        for (int i = 0; i < 1000 && !(aw_done && w_done); ++i) {
            tick();
            if (gap > 0) {
                --gap;
                if (gap == 0) {
                    if (aw_before_w_gap > 0) top.S_AXI_WVALID = 1;
                    if (aw_before_w_gap < 0) top.S_AXI_AWVALID = 1;
                }
            }
            aw_done = aw_done || top.S_AXI_AWREADY;
            w_done = w_done || top.S_AXI_WREADY;
            if (aw_done) top.S_AXI_AWVALID = 0;
            if (w_done) top.S_AXI_WVALID = 0;
        }
        top.S_AXI_AWVALID = 0;
        top.S_AXI_WVALID = 0;
        top.S_AXI_BREADY = 1;
        for (int i = 0; i < 1000; ++i) {
            tick();
            if (top.S_AXI_BVALID) break;
        }
        tick();
        top.S_AXI_BREADY = 0;
    }

    uint32_t axil_read(uint32_t addr) {
        return axil_read_with_rready_delay(addr, 0);
    }

    uint32_t axil_read_with_rready_delay(uint32_t addr, int rready_delay) {
        top.S_AXI_ARADDR = addr;
        top.S_AXI_ARVALID = 1;
        top.S_AXI_RREADY = rready_delay == 0 ? 1 : 0;
        bool ar_done = false;
        for (int i = 0; i < 2000; ++i) {
            tick();
            if (top.S_AXI_ARREADY) {
                ar_done = true;
                top.S_AXI_ARVALID = 0;
            }
            if (ar_done && rready_delay > 0) {
                --rready_delay;
                if (rready_delay == 0) top.S_AXI_RREADY = 1;
            }
            if (top.S_AXI_RVALID) {
                uint32_t ret = top.S_AXI_RDATA;
                tick();
                top.S_AXI_RREADY = 0;
                return ret;
            }
        }
        top.S_AXI_ARVALID = 0;
        top.S_AXI_RREADY = 0;
        (void)ar_done;
        throw std::runtime_error("AXI-Lite read timeout");
    }

    void run_until_done(uint64_t max_ticks = 20000000) {
        for (uint64_t i = 0; i < max_ticks; ++i) {
            uint32_t st = axil_read(0x164);
            if (st & 0x2u) return;
            if (st & 0x4u) {
                uint32_t err = axil_read(0x18c);
                std::fprintf(stderr, "hardware error 0x%08x\n", err);
                std::exit(2);
            }
            for (int j = 0; j < 8; ++j) tick();
        }
        throw std::runtime_error("done timeout");
    }
};

static unsigned getenv_u32(const char *name, unsigned default_value) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw) return default_value;
    char *end = nullptr;
    unsigned long v = std::strtoul(raw, &end, 0);
    if (end == raw) return default_value;
    return static_cast<unsigned>(std::min<unsigned long>(v, 1000000ul));
}

static int8_t rand_s8(std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(-8, 7);
    return static_cast<int8_t>(dist(rng));
}

static int8_t s8(int v) {
    v &= 0xff;
    return static_cast<int8_t>(v >= 128 ? v - 256 : v);
}

static int8_t board_act_val(int mi, int kk) {
    return s8(mi * 13 + kk * 7 + 3);
}

static int8_t board_weight_val(int kk, int ni) {
    return s8(kk * 5 + ni * 11 - 9);
}

static uint64_t align_up(uint64_t v, uint64_t align) {
    return ((v + align - 1) / align) * align;
}

static void write_i8_matrix(std::vector<uint8_t> &mem, uint64_t base, int rows, int cols, int stride, std::mt19937 &rng, std::vector<int8_t> &out) {
    out.assign(static_cast<size_t>(rows) * cols, 0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int8_t v = rand_s8(rng);
            out[static_cast<size_t>(r) * cols + c] = v;
            mem[base + static_cast<uint64_t>(r) * stride + c] = static_cast<uint8_t>(v);
        }
    }
}

static int32_t read_i32(const std::vector<uint8_t> &mem, uint64_t addr) {
    uint32_t u = static_cast<uint32_t>(mem[addr]) |
                 (static_cast<uint32_t>(mem[addr + 1]) << 8) |
                 (static_cast<uint32_t>(mem[addr + 2]) << 16) |
                 (static_cast<uint32_t>(mem[addr + 3]) << 24);
    return static_cast<int32_t>(u);
}

static void fill_region(std::vector<uint8_t> &mem, uint64_t base, uint64_t size, uint8_t value) {
    if (base + size > mem.size()) throw std::runtime_error("fill out of memory");
    std::memset(mem.data() + base, value, static_cast<size_t>(size));
}

static void run_echo(Sim &sim) {
    std::printf("RUN ECHO\n");
    std::fflush(stdout);
    constexpr uint64_t src = 0x1000;
    constexpr uint64_t dst = 0x4000;
    for (int i = 0; i < 4096; ++i) sim.mem[src + i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
    sim.axil_write(0x168, static_cast<uint32_t>(src));
    sim.axil_write(0x16c, 0);
    sim.axil_write(0x170, static_cast<uint32_t>(dst));
    sim.axil_write(0x174, 0);
    sim.axil_write(0x180, 4096);
    sim.axil_write(0x184, MODE_ECHO);
    sim.axil_write(0x160, 0x1);
    sim.run_until_done();
    for (int i = 0; i < 4096; ++i) {
        if (sim.mem[src + i] != sim.mem[dst + i]) throw std::runtime_error("echo mismatch");
    }
    std::printf("ECHO PASS read_beats=%u write_beats=%u read_bursts=%u write_bursts=%u\n",
                sim.axil_read(0x190), sim.axil_read(0x194), sim.axil_read(0x198), sim.axil_read(0x19c));
    std::fflush(stdout);
}

static void run_echo_aligned_alias(Sim &sim) {
    std::printf("RUN ECHO_ALIGNED_ALIAS\n");
    std::fflush(stdout);
    constexpr uint64_t src = 0x18000;
    constexpr uint64_t dst = 0x1c000;
    for (int i = 0; i < 4096; ++i) sim.mem[src + i] = static_cast<uint8_t>((i * 31 + 9) & 0xff);
    sim.axil_write(0x420, static_cast<uint32_t>(src));
    sim.axil_write(0x430, 0);
    sim.axil_write(0x440, static_cast<uint32_t>(dst));
    sim.axil_write(0x450, 0);
    sim.axil_write(0x480, 4096);
    sim.axil_write(0x490, MODE_ECHO);
    sim.axil_write(0x400, 0x1);
    sim.run_until_done();
    for (int i = 0; i < 4096; ++i) {
        if (sim.mem[src + i] != sim.mem[dst + i]) throw std::runtime_error("aligned alias echo mismatch");
    }
    if (sim.axil_read(0x420) != static_cast<uint32_t>(src) ||
        sim.axil_read(0x440) != static_cast<uint32_t>(dst) ||
        sim.axil_read(0x480) != 4096 ||
        sim.axil_read(0x490) != MODE_ECHO) {
        throw std::runtime_error("aligned alias cfg readback mismatch");
    }
    std::printf("ECHO_ALIGNED_ALIAS PASS read_beats=%u write_beats=%u\n",
                sim.axil_read(0x190), sim.axil_read(0x194));
    std::fflush(stdout);
}

static void run_axil_protocol(Sim &sim) {
    std::printf("RUN AXIL_PROTOCOL\n");
    std::fflush(stdout);
    const uint32_t magic0 = sim.axil_read(0x000);
    const uint32_t magic154 = sim.axil_read_with_rready_delay(0x154, 5);
    if (magic0 != 0x53444634u || magic154 != 0x53444634u) {
        throw std::runtime_error("AXI-Lite magic mirror mismatch");
    }
    sim.axil_write(0x00c, 0x11223344u);
    if (sim.axil_read(0x00c) != 0x11223344u) throw std::runtime_error("combined AXI-Lite write failed");
    sim.axil_write_aw_then_w(0x010, 0x55667788u);
    if (sim.axil_read(0x010) != 0x55667788u) throw std::runtime_error("AW-before-W AXI-Lite write failed");
    sim.axil_write_w_then_aw(0x020, 0x99aabbccu);
    if (sim.axil_read(0x020) != 0x99aabbccu) throw std::runtime_error("W-before-AW AXI-Lite write failed");
    sim.axil_write(0x300, 0x11112222u);
    sim.axil_write(0x304, 0x33334444u);
    sim.axil_write(0x308, 0x55556666u);
    sim.axil_write(0x30c, 0x77778888u);
    if (sim.axil_read(0x340) != 0x11112222u ||
        sim.axil_read(0x350) != 0x33334444u ||
        sim.axil_read(0x360) != 0x55556666u ||
        sim.axil_read(0x370) != 0x77778888u) {
        throw std::runtime_error("SDF4.4 lane echo failed");
    }
    if (sim.axil_read(0x280) != 0x30cu ||
        sim.axil_read(0x290) != 0x77778888u ||
        sim.axil_read(0x2a0) != 0xfu ||
        sim.axil_read(0x2b0) != 0x30cu ||
        sim.axil_read(0x2c0) != 0x30cu) {
        throw std::runtime_error("SDF4.4 aligned AW/W mirrors failed");
    }
    const uint32_t awc = sim.axil_read(0x22c);
    const uint32_t wc = sim.axil_read(0x230);
    const uint32_t bc = sim.axil_read(0x234);
    const uint32_t arc = sim.axil_read(0x238);
    const uint32_t rc = sim.axil_read(0x23c);
    if (awc < 3 || wc < 3 || bc < 3 || arc < 6 || rc < 5) {
        throw std::runtime_error("AXI-Lite monitor counters too small");
    }
    std::printf("AXIL_PROTOCOL PASS aw=%u w=%u b=%u ar=%u r=%u alive=%u\n",
                awc, wc, bc, arc, rc, sim.axil_read(0x228));
    std::fflush(stdout);
}

static void run_linear_mode(Sim &sim, const char *name, int m, int k, int n,
                            uint64_t src, uint64_t weight, uint64_t dst, uint32_t mode) {
    std::printf("RUN %s mode=0x%08x M=%d K=%d N=%d\n", name, mode, m, k, n);
    std::fflush(stdout);
    const int k_stride = ((k + 15) / 16) * 16;
    const int dst_stride = static_cast<int>(align_up(static_cast<uint64_t>(n) * 4u, 16));
    std::mt19937 rng(static_cast<unsigned>(m * 1000003 + k * 9176 + n));
    std::vector<int8_t> act, w;
    write_i8_matrix(sim.mem, src, m, k, k_stride, rng, act);
    write_i8_matrix(sim.mem, weight, n, k, k_stride, rng, w);
    std::memset(sim.mem.data() + dst, 0, static_cast<size_t>(m) * dst_stride);

    sim.axil_write(0x168, static_cast<uint32_t>(src));
    sim.axil_write(0x16c, 0);
    sim.axil_write(0x170, static_cast<uint32_t>(dst));
    sim.axil_write(0x174, 0);
    sim.axil_write(0x178, static_cast<uint32_t>(weight));
    sim.axil_write(0x17c, 0);
    sim.axil_write(0x184, mode);
    sim.axil_write(0x1a0, static_cast<uint32_t>(m));
    sim.axil_write(0x1a4, static_cast<uint32_t>(k));
    sim.axil_write(0x1a8, static_cast<uint32_t>(n));
    sim.axil_write(0x1ac, static_cast<uint32_t>(k_stride));
    sim.axil_write(0x1b0, static_cast<uint32_t>(k_stride));
    sim.axil_write(0x1b4, static_cast<uint32_t>(dst_stride));
    sim.axil_write(0x160, 0x1);
    sim.run_until_done();

    for (int mi = 0; mi < m; ++mi) {
        for (int ni = 0; ni < n; ++ni) {
            int32_t exp = 0;
            for (int ki = 0; ki < k; ++ki) {
                exp += static_cast<int32_t>(act[static_cast<size_t>(mi) * k + ki]) *
                       static_cast<int32_t>(w[static_cast<size_t>(ni) * k + ki]);
            }
            int32_t got = read_i32(sim.mem, dst + static_cast<uint64_t>(mi) * dst_stride + ni * 4);
            if (got != exp) {
                std::fprintf(stderr, "%s mismatch m=%d n=%d got=%d exp=%d\n", name, mi, ni, got, exp);
                uint32_t exp_act0_lo = 0;
                uint32_t exp_w0_lo = 0;
                uint32_t exp_w8_lo = 0;
                for (int bi = 0; bi < 4 && bi < k; ++bi) {
                    exp_act0_lo |= static_cast<uint32_t>(static_cast<uint8_t>(act[bi])) << (8 * bi);
                    exp_w0_lo |= static_cast<uint32_t>(static_cast<uint8_t>(w[bi])) << (8 * bi);
                    if (n > 8) exp_w8_lo |= static_cast<uint32_t>(static_cast<uint8_t>(w[static_cast<size_t>(8) * k + bi])) << (8 * bi);
                }
                std::fprintf(stderr,
                             "%s hwdbg act0_lo=0x%08x exp=0x%08x w0_lo=0x%08x exp=0x%08x w8_lo=0x%08x exp=0x%08x first_sum00=%d\n",
                             name,
                             sim.axil_read(0x1d8), exp_act0_lo,
                             sim.axil_read(0x1dc), exp_w0_lo,
                             sim.axil_read(0x1e0), exp_w8_lo,
                             static_cast<int32_t>(sim.axil_read(0x1e4)));
                std::fprintf(stderr, "%s hwdbg trace acc:", name);
                for (int ti = 0; ti < 8; ++ti) {
                    std::fprintf(stderr, " %d", static_cast<int32_t>(sim.axil_read(0x1e8 + ti * 4)));
                }
                std::fprintf(stderr, "\n%s hwdbg trace delta:", name);
                for (int ti = 0; ti < 8; ++ti) {
                    std::fprintf(stderr, " %d", static_cast<int32_t>(sim.axil_read(0x208 + ti * 4)));
                }
                std::fprintf(stderr, "\n");
                if (mi == 0) {
                    std::fprintf(stderr, "%s row0 dump:", name);
                    for (int dj = 0; dj < n && dj < 32; ++dj) {
                        int32_t exp_j = 0;
                        for (int ki = 0; ki < k; ++ki) {
                            exp_j += static_cast<int32_t>(act[static_cast<size_t>(mi) * k + ki]) *
                                     static_cast<int32_t>(w[static_cast<size_t>(dj) * k + ki]);
                        }
                        int32_t got_j = read_i32(sim.mem, dst + static_cast<uint64_t>(mi) * dst_stride + dj * 4);
                        std::fprintf(stderr, " [%d got=%d exp=%d]", dj, got_j, exp_j);
                    }
                    std::fprintf(stderr, "\n");
                    std::fprintf(stderr, "%s row0 n0 k8 partials:", name);
                    for (int ks = 0; ks < k; ks += 8) {
                        int32_t part = 0;
                        for (int ski = ks; ski < std::min(k, ks + 8); ++ski) {
                            part += static_cast<int32_t>(act[static_cast<size_t>(mi) * k + ski]) *
                                    static_cast<int32_t>(w[static_cast<size_t>(0) * k + ski]);
                        }
                        std::fprintf(stderr, " [%d:%d=%d]", ks, std::min(k, ks + 8), part);
                    }
                    std::fprintf(stderr, "\n");
                    std::fprintf(stderr, "%s row0 n0 k16 partials:", name);
                    for (int ks = 0; ks < k; ks += 16) {
                        int32_t part = 0;
                        for (int ski = ks; ski < std::min(k, ks + 16); ++ski) {
                            part += static_cast<int32_t>(act[static_cast<size_t>(mi) * k + ski]) *
                                    static_cast<int32_t>(w[static_cast<size_t>(0) * k + ski]);
                        }
                        std::fprintf(stderr, " [%d:%d=%d]", ks, std::min(k, ks + 16), part);
                    }
                    std::fprintf(stderr, "\n");
                    std::fprintf(stderr, "%s row0 got-map:", name);
                    for (int dj = 0; dj < n && dj < 32; ++dj) {
                        int32_t got_j = read_i32(sim.mem, dst + static_cast<uint64_t>(mi) * dst_stride + dj * 4);
                        bool found = false;
                        for (int sn = 0; sn < n && sn < 64 && !found; ++sn) {
                            int32_t ref = 0;
                            for (int ski = 0; ski < k; ++ski) {
                                ref += static_cast<int32_t>(act[static_cast<size_t>(mi) * k + ski]) *
                                       static_cast<int32_t>(w[static_cast<size_t>(sn) * k + ski]);
                            }
                            if (ref == got_j) {
                                std::fprintf(stderr, " [%d->%d]", dj, sn);
                                found = true;
                            }
                        }
                        if (!found) std::fprintf(stderr, " [%d->?]", dj);
                    }
                    std::fprintf(stderr, "\n");
                }
                int shown = 0;
                std::fprintf(stderr, "%s got-value reference matches:", name);
                for (int sm = 0; sm < m && sm < 32 && shown < 16; ++sm) {
                    for (int sn = 0; sn < n && sn < 64 && shown < 16; ++sn) {
                        int32_t ref = 0;
                        for (int ski = 0; ski < k; ++ski) {
                            ref += static_cast<int32_t>(act[static_cast<size_t>(sm) * k + ski]) *
                                   static_cast<int32_t>(w[static_cast<size_t>(sn) * k + ski]);
                        }
                        if (ref == got) {
                            std::fprintf(stderr, " (%d,%d)", sm, sn);
                            ++shown;
                        }
                    }
                }
                if (shown == 0) std::fprintf(stderr, " none");
                std::fprintf(stderr, "\n");
                std::exit(3);
            }
        }
    }
    uint32_t cycles = sim.axil_read(0x148);
    uint32_t mac_lo = sim.axil_read(0x150);
    uint32_t compute_cycles = sim.axil_read(0x1c8);
    const double real_mac = static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
    const double peak_mac_per_cycle = 4.0 * 24.0 * 8.0;
    double lane_util = cycles ? static_cast<double>(mac_lo) / (static_cast<double>(cycles) * peak_mac_per_cycle) : 0.0;
    double real_util = cycles ? real_mac / (static_cast<double>(cycles) * peak_mac_per_cycle) : 0.0;
    double real_gmac_150 = cycles ? real_mac * 150.0e6 / static_cast<double>(cycles) / 1.0e9 : 0.0;
    std::printf("%s PASS cycles=%u tiles=%u real_mac=%.0f mac_lo=%u read_beats=%u write_beats=%u read_bursts=%u write_bursts=%u compute_cycles=%u load_cycles=%u store_cycles=%u stall_cycles=%u lane_util_est=%.6f real_util_est=%.6f real_gmac_150=%.3f\n",
                name,
                cycles,
                sim.axil_read(0x14c),
                real_mac,
                mac_lo,
                sim.axil_read(0x190),
                sim.axil_read(0x194),
                sim.axil_read(0x198),
                sim.axil_read(0x19c),
                compute_cycles,
                sim.axil_read(0x1cc),
                sim.axil_read(0x1d0),
                sim.axil_read(0x1d4),
                lane_util,
                real_util,
                real_gmac_150);
    std::fflush(stdout);
}

static void run_linear(Sim &sim, const char *name, int m, int k, int n,
                       uint64_t src, uint64_t weight, uint64_t dst) {
    run_linear_mode(sim, name, m, k, n, src, weight, dst, MODE_LINEAR_BATCH);
}

static void run_linear_board_pattern(Sim &sim, const char *name, int m, int k, int n,
                                     bool full_check) {
    const int k_stride = ((k + 15) / 16) * 16;
    const int dst_stride = static_cast<int>(align_up(static_cast<uint64_t>(n) * 4u, 16));
    const uint64_t src = 0;
    const uint64_t act_bytes = static_cast<uint64_t>(m) * k_stride;
    const uint64_t weight = align_up(src + act_bytes, 4096);
    const uint64_t weight_bytes = static_cast<uint64_t>(n) * k_stride;
    const uint64_t dst = align_up(weight + weight_bytes, 4096);
    const uint64_t dst_bytes = static_cast<uint64_t>(m) * dst_stride;
    if (dst + dst_bytes > sim.mem.size()) {
        throw std::runtime_error(std::string(name) + " does not fit in sim memory");
    }

    std::printf("RUN_BOARD %s M=%d K=%d N=%d layout=0x%llx/0x%llx/0x%llx full=%d\n",
                name, m, k, n,
                static_cast<unsigned long long>(src),
                static_cast<unsigned long long>(weight),
                static_cast<unsigned long long>(dst),
                full_check ? 1 : 0);
    std::fflush(stdout);

    fill_region(sim.mem, src, act_bytes, 0);
    fill_region(sim.mem, weight, weight_bytes, 0);
    fill_region(sim.mem, dst, dst_bytes, 0xa5);

    for (int mi = 0; mi < m; ++mi) {
        uint64_t base = src + static_cast<uint64_t>(mi) * k_stride;
        for (int kk = 0; kk < k; ++kk) {
            sim.mem[base + kk] = static_cast<uint8_t>(board_act_val(mi, kk));
        }
    }
    for (int ni = 0; ni < n; ++ni) {
        uint64_t base = weight + static_cast<uint64_t>(ni) * k_stride;
        for (int kk = 0; kk < k; ++kk) {
            sim.mem[base + kk] = static_cast<uint8_t>(board_weight_val(kk, ni));
        }
    }

    sim.axil_write(0x168, static_cast<uint32_t>(src));
    sim.axil_write(0x16c, 0);
    sim.axil_write(0x170, static_cast<uint32_t>(dst));
    sim.axil_write(0x174, 0);
    sim.axil_write(0x178, static_cast<uint32_t>(weight));
    sim.axil_write(0x17c, 0);
    sim.axil_write(0x184, MODE_LINEAR_BATCH);
    sim.axil_write(0x1a0, static_cast<uint32_t>(m));
    sim.axil_write(0x1a4, static_cast<uint32_t>(k));
    sim.axil_write(0x1a8, static_cast<uint32_t>(n));
    sim.axil_write(0x1ac, static_cast<uint32_t>(k_stride));
    sim.axil_write(0x1b0, static_cast<uint32_t>(k_stride));
    sim.axil_write(0x1b4, static_cast<uint32_t>(dst_stride));
    sim.axil_write(0x160, 0x2);
    sim.axil_write(0x160, 0x1);
    sim.run_until_done(100000000);

    std::vector<int> samples_m;
    std::vector<int> samples_n;
    auto push_unique = [](std::vector<int> &v, int x, int limit) {
        if (x < 0 || x >= limit) return;
        if (std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
    };
    if (full_check) {
        for (int mi = 0; mi < m; ++mi) samples_m.push_back(mi);
        for (int ni = 0; ni < n; ++ni) samples_n.push_back(ni);
    } else {
        for (int x : {0, 1, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, m / 2, m - 5, m - 1}) {
            push_unique(samples_m, x, m);
        }
        for (int base = 0; base < n; base += 96) {
            for (int d : {0, 1, 15, 16, 17, 21, 25, 31, 47, 63, 79, 95}) {
                push_unique(samples_n, base + d, n);
            }
        }
        for (int x : {0, 1, n / 2, n - 5, n - 1}) push_unique(samples_n, x, n);
        std::sort(samples_m.begin(), samples_m.end());
        std::sort(samples_n.begin(), samples_n.end());
    }

    int mismatches = 0;
    for (int mi : samples_m) {
        for (int ni : samples_n) {
            int32_t exp = 0;
            for (int kk = 0; kk < k; ++kk) {
                exp += static_cast<int32_t>(board_act_val(mi, kk)) *
                       static_cast<int32_t>(board_weight_val(kk, ni));
            }
            int32_t got = read_i32(sim.mem, dst + static_cast<uint64_t>(mi) * dst_stride + ni * 4);
            if (got != exp) {
                if (mismatches < 16) {
                    std::fprintf(stderr, "%s board-pattern mismatch m=%d n=%d got=%d exp=%d\n",
                                 name, mi, ni, got, exp);
                }
                ++mismatches;
            }
        }
    }

    const double real_mac = static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
    const uint32_t cycles = sim.axil_read(0x148);
    const double real_gmac_150 = cycles ? real_mac * 150.0e6 / static_cast<double>(cycles) / 1.0e9 : 0.0;
    std::printf("%s %s cycles=%u tiles=%u real_mac=%.0f read_beats=%u write_beats=%u read_bursts=%u write_bursts=%u compute_cycles=%u load_cycles=%u store_cycles=%u stall_cycles=%u real_gmac_150=%.3f samples=%zu/%zu mismatches=%d\n",
                name,
                mismatches == 0 ? "PASS" : "FAIL",
                cycles,
                sim.axil_read(0x14c),
                real_mac,
                sim.axil_read(0x190),
                sim.axil_read(0x194),
                sim.axil_read(0x198),
                sim.axil_read(0x19c),
                sim.axil_read(0x1c8),
                sim.axil_read(0x1cc),
                sim.axil_read(0x1d0),
                sim.axil_read(0x1d4),
                real_gmac_150,
                samples_m.size(),
                samples_n.size(),
                mismatches);
    std::fflush(stdout);
    if (mismatches != 0) ++g_failures;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    const unsigned read_latency = getenv_u32("SDF3_SIM_RLAT", 0);
    const unsigned bresp_latency = getenv_u32("SDF3_SIM_BRESP_LAT", 0);
    std::printf("SIM_LATENCY read=%u bresp=%u\n", read_latency, bresp_latency);
    Sim sim(128 * 1024 * 1024, read_latency, bresp_latency);
    sim.reset();
    uint32_t magic = sim.axil_read(0x154);
    if (magic != 0x53444634u) {
        std::fprintf(stderr, "bad magic 0x%08x\n", magic);
        return 1;
    }
    std::printf("MAGIC PASS 0x%08x\n", magic);
    std::fflush(stdout);
    run_axil_protocol(sim);
    run_echo(sim);
    run_echo_aligned_alias(sim);
    run_linear(sim, "SMALL_16x64x16", 16, 64, 16, 0x00100000, 0x00200000, 0x01000000);
    run_linear(sim, "SDF1_LIKE_1x960x16", 1, 960, 16, 0x00300000, 0x00400000, 0x01100000);
    run_linear_mode(sim, "SDF1_LEGACY_MODE_1x960x16", 1, 960, 16, 0x00700000, 0x00800000, 0x01300000, MODE_LINEAR_Q8);
    run_linear(sim, "EDGE_17x65x19", 17, 65, 19, 0x00500000, 0x00600000, 0x01200000);
    run_linear(sim, "MNN_BLOCK_1024x64x1024", 1024, 64, 1024, 0x01400000, 0x01800000, 0x02000000);
    run_linear(sim, "MNN_BLOCK_1024x1024x64", 1024, 1024, 64, 0x03000000, 0x03400000, 0x04000000);
    run_linear(sim, "MNN_LANG_80x960x320", 80, 960, 320, 0x05000000, 0x05200000, 0x05800000);
    run_linear(sim, "MNN_LANG_80x960x2560", 80, 960, 2560, 0x06000000, 0x06200000, 0x06800000);
    run_linear(sim, "MNN_LANG_80x2560x960", 80, 2560, 960, 0x07000000, 0x07300000, 0x07900000);
    run_linear_board_pattern(sim, "BOARD_LANG_80x960x96_FULL", 80, 960, 96, true);
    run_linear_board_pattern(sim, "BOARD_LANG_80x960x128_STRONG", 80, 960, 128, false);
    run_linear_board_pattern(sim, "BOARD_LANG_80x960x160_FULL", 80, 960, 160, true);
    run_linear_board_pattern(sim, "BOARD_LANG_80x960x192_STRONG", 80, 960, 192, false);
    run_linear_board_pattern(sim, "BOARD_LANG_80x960x320_STRONG", 80, 960, 320, false);
    run_linear_board_pattern(sim, "BOARD_LANG_80x960x960_STRONG", 80, 960, 960, false);
    run_linear_board_pattern(sim, "BOARD_LANG_80x960x2560_STRONG", 80, 960, 2560, false);
    run_linear_board_pattern(sim, "BOARD_LANG_80x2560x960_STRONG", 80, 2560, 960, false);
    run_linear_board_pattern(sim, "BOARD_VISION_1024x768x768_STRONG", 1024, 768, 768, false);
    run_linear_board_pattern(sim, "BOARD_VISION_1024x3072x768_STRONG", 1024, 3072, 768, false);
    run_linear_board_pattern(sim, "BOARD_VISION_1024x768x3072_STRONG", 1024, 768, 3072, false);
    return g_failures == 0 ? 0 : 4;
}
