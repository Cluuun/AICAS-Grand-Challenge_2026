#include "VSdf2DebugAxi.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t MODE_ECHO = 0x0000e001u;
static constexpr uint32_t MODE_LINEAR_Q8 = 0x0000e101u;
static constexpr uint32_t MODE_LINEAR_BATCH = 0x0000e301u;

struct Sim {
    VSdf2DebugAxi top;
    uint64_t now = 0;
    std::vector<uint8_t> mem;

    bool rvalid = false;
    uint64_t raddr = 0;
    bool bvalid = false;
    bool aw_seen = false;
    bool w_seen = false;
    uint64_t awaddr = 0;
    std::array<uint32_t, 4> wdata{};
    uint16_t wstrb = 0;

    explicit Sim(size_t mem_size) : mem(mem_size, 0) {
        top.ACLK = 0;
        top.ARESETN = 0;
        top.S_AXI_AWADDR = 0;
        top.S_AXI_AWVALID = 0;
        top.S_AXI_WDATA = 0;
        top.S_AXI_WSTRB = 0;
        top.S_AXI_WVALID = 0;
        top.S_AXI_BREADY = 0;
        top.S_AXI_ARADDR = 0;
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
        top.M_AXI_RLAST = 1;
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

    void commit_write_if_ready() {
        if (aw_seen && w_seen) {
            if (awaddr + 16 > mem.size()) throw std::runtime_error("write out of memory");
            for (int lane = 0; lane < 16; ++lane) {
                if (wstrb & (1u << lane)) {
                    uint32_t word = wdata[lane / 4];
                    mem[awaddr + lane] = static_cast<uint8_t>((word >> ((lane % 4) * 8)) & 0xffu);
                }
            }
            aw_seen = false;
            w_seen = false;
            bvalid = true;
        }
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
        const uint64_t awaddr_now = top.M_AXI_AWADDR;
        std::array<uint32_t, 4> wdata_now{};
        for (int i = 0; i < 4; ++i) wdata_now[i] = top.M_AXI_WDATA[i];
        const uint16_t wstrb_now = top.M_AXI_WSTRB;

        top.ACLK = 1;
        top.eval();
        now++;

        if (r_hs) rvalid = false;
        if (b_hs) bvalid = false;
        if (ar_hs) {
            raddr = araddr_now;
            rvalid = true;
        }
        if (aw_hs) {
            awaddr = awaddr_now;
            aw_seen = true;
        }
        if (w_hs) {
            wdata = wdata_now;
            wstrb = wstrb_now;
            w_seen = true;
        }
        commit_write_if_ready();
    }

    void reset() {
        top.ARESETN = 0;
        for (int i = 0; i < 8; ++i) tick();
        top.ARESETN = 1;
        for (int i = 0; i < 4; ++i) tick();
    }

    void axil_write(uint32_t addr, uint32_t data) {
        top.S_AXI_AWADDR = addr;
        top.S_AXI_WDATA = data;
        top.S_AXI_WSTRB = 0xf;
        top.S_AXI_AWVALID = 1;
        top.S_AXI_WVALID = 1;
        bool aw_done = false, w_done = false;
        for (int i = 0; i < 1000 && !(aw_done && w_done); ++i) {
            tick();
            aw_done = aw_done || top.S_AXI_AWREADY;
            w_done = w_done || top.S_AXI_WREADY;
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
        top.S_AXI_ARADDR = addr;
        top.S_AXI_ARVALID = 1;
        top.S_AXI_RREADY = 1;
        bool ar_done = false;
        for (int i = 0; i < 2000; ++i) {
            tick();
            if (top.S_AXI_ARREADY) {
                ar_done = true;
                top.S_AXI_ARVALID = 0;
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

static int8_t rand_s8(std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(-8, 7);
    return static_cast<int8_t>(dist(rng));
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
    std::printf("ECHO PASS read_beats=%u write_beats=%u\n", sim.axil_read(0x190), sim.axil_read(0x194));
    std::fflush(stdout);
}

static void run_linear_mode(Sim &sim, const char *name, int m, int k, int n,
                            uint64_t src, uint64_t weight, uint64_t dst, uint32_t mode) {
    std::printf("RUN %s mode=0x%08x M=%d K=%d N=%d\n", name, mode, m, k, n);
    std::fflush(stdout);
    const int k_stride = ((k + 15) / 16) * 16;
    const int dst_stride = n * 4;
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
                std::exit(3);
            }
        }
    }
    std::printf("%s PASS cycles=%u tiles=%u mac_lo=%u read_beats=%u write_beats=%u\n",
                name,
                sim.axil_read(0x148),
                sim.axil_read(0x14c),
                sim.axil_read(0x150),
                sim.axil_read(0x190),
                sim.axil_read(0x194));
    std::fflush(stdout);
}

static void run_linear(Sim &sim, const char *name, int m, int k, int n,
                       uint64_t src, uint64_t weight, uint64_t dst) {
    run_linear_mode(sim, name, m, k, n, src, weight, dst, MODE_LINEAR_BATCH);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Sim sim(128 * 1024 * 1024);
    sim.reset();
    uint32_t magic = sim.axil_read(0x154);
    if (magic != 0x53444632u) {
        std::fprintf(stderr, "bad magic 0x%08x\n", magic);
        return 1;
    }
    std::printf("MAGIC PASS 0x%08x\n", magic);
    std::fflush(stdout);
    run_echo(sim);
    run_linear(sim, "SMALL_16x64x16", 16, 64, 16, 0x00100000, 0x00200000, 0x01000000);
    run_linear(sim, "SDF1_LIKE_1x960x16", 1, 960, 16, 0x00300000, 0x00400000, 0x01100000);
    run_linear_mode(sim, "SDF1_LEGACY_MODE_1x960x16", 1, 960, 16, 0x00700000, 0x00800000, 0x01300000, MODE_LINEAR_Q8);
    run_linear(sim, "EDGE_17x65x19", 17, 65, 19, 0x00500000, 0x00600000, 0x01200000);
    return 0;
}
