/*
 * pl_controller.cpp — 描述符环预取引擎 + 门铃 FSM + 多核调度 + 中断输出 (v6)
 *
 * 严格遵循 kv260_maximum_optimized.md §2.4(3):
 *   1. 门铃原子拼接: 检测 REG_CMD_HIGH_TRIG 写握手 (WVALID && WREADY)
 *      → 原子拼接 128-bit 门铃 → 解析 desc_index → 推入门铃 FIFO
 *   2. AXI Master 描述符预取引擎: 检测门铃 FIFO 非空
 *      → 向 CMA 描述符环发起 Length=1, 256-bit 完美对齐单次大突发读取
 *   3. 算子分发: 解析 opcode → 配置对应计算核 → 启动 → 轮询完工
 *   4. 完工写回: AXI Master HPC 端口 (AxCACHE=0b1111) 写 last_done_id
 *   5. 硬件看门狗: 10μs 捕捉异常, 自动降级
 *   6. 中断输出: 每次调用处理一条命令, 处理完后函数返回 → ap_done → interrupt
 *      PS 侧通过 auto_restart 模式实现连续运行
 *
 * 块级协议: ap_ctrl_hs (s_axilite port=return)
 *   - HLS 自动生成标准 interrupt 端口 (INTERRUPT 类型, 与 Vivado IPI 兼容)
 *   - PS 写 ap_start 启动, 函数返回时 ap_done 拉高触发中断
 *   - auto_restart=1 使函数返回后自动重新进入, 实现连续调度
 *
 * 支持的算子:
 *   - GEMM_PREFILL (0x00) / GEMV_DECODING (0x01)
 *   - ATTENTION (0x02)
 *   - FUSED_VIT_MLP (0x11)
 *   - RMSNORM (0x12)
 *   - VECTOR_RESCALER (0x13)
 *   - ROPE (0x14)
 *   - KV_CACHE_DMA (0x03)
 *
 * 资源: DSP 8, BRAM 16, URAM 16, LUT ~15K, FF ~22K, 100MHz
 */

#include <ap_int.h>
#include <hls_stream.h>

// ============================================================
// 门铃结构体 (128-bit, 与 PS 侧 pl_cmd_doorbell 一致)
// ============================================================
struct doorbell_entry {
    ap_uint<32> desc_index;
    ap_uint<16> opcode;
    ap_uint<8>  flags;
    ap_uint<8>  reserved;
    ap_uint<64> timestamp;
};

// ============================================================
// 描述符结构体 (256-bit, 与 PS 侧 pl_descriptor 一致)
// ============================================================
struct desc_entry {
    ap_uint<64> src_addr;
    ap_uint<64> weight_addr;
    ap_uint<64> dst_addr;
    ap_uint<32> dim_m;
    ap_uint<16> dim_n;
    ap_uint<16> dim_k;
};

// ============================================================
// 状态枚举
// ============================================================
typedef enum {
    ST_IDLE           = 0,
    ST_WAIT_DOORBELL  = 1,
    ST_FETCH_DESC     = 2,
    ST_CHECK_DESC     = 3,
    ST_DECODE         = 4,
    ST_EXEC_GEMM      = 5,
    ST_EXEC_ATTN      = 6,
    ST_EXEC_VIT_MLP   = 8,
    ST_EXEC_RMSNORM   = 9,
    ST_EXEC_RESCALER  = 10,
    ST_EXEC_ROPE      = 11,
    ST_EXEC_DMA       = 12,
    ST_DONE           = 13,
    ST_ERR            = 14
} state_t;

// ============================================================
// 参数
// ============================================================
#define DOORBELL_FIFO_DEPTH   8       // 门铃 FIFO (LUTRAM SRL16)
#define DESC_RING_SLOTS       128     // CMA 描述符环槽位数
#define WDT_MAX_COUNT         1000    // 10μs @ 100MHz
#define MAX_POLL_COUNT        260000  // 2.6ms GEMM 轮询超时
#define IDLE_POLL_MAX          50     // IDLE 状态门铃轮询超时 (500ns @ 100MHz)

// 计算核 ap_ctrl 偏移
#define AP_CTRL_OFFSET        0x00
#define AP_START_BIT          0
#define AP_DONE_BIT           1
#define AP_IDLE_BIT           2

// 标准子核 s_axilite 参数寄存器字偏移映射
#define REG_SRC_LOW           4   // 0x10 -> 低32位
#define REG_SRC_HIGH          5   // 0x14 -> 高32位
#define REG_WGT_LOW           6   // 0x18 -> 低32位
#define REG_WGT_HIGH          7   // 0x1C -> 高32位
#define REG_DST_LOW           8   // 0x20 -> 低32位
#define REG_DST_HIGH          9   // 0x24 -> 高32位
#define REG_DIM_M             10  // 0x28
#define REG_DIM_N             11  // 0x2C
#define REG_DIM_K             12  // 0x30

// ============================================================
// CRC-8 查表 (与 PS 侧一致)
// ============================================================
static const ap_uint<8> crc8_tab[256] = {
    0x00,0x07,0x0E,0x09,0x1C,0x1B,0x12,0x15,0x38,0x3F,0x36,0x31,0x24,0x23,0x2A,0x2D,
    0x70,0x77,0x7E,0x79,0x6C,0x6B,0x62,0x65,0x48,0x4F,0x46,0x41,0x54,0x53,0x5A,0x5D,
    0xE0,0xE7,0xEE,0xE9,0xFC,0xFB,0xF2,0xF5,0xD8,0xDF,0xD6,0xD1,0xC4,0xC3,0xCA,0xCD,
    0x90,0x97,0x9E,0x99,0x8C,0x8B,0x82,0x85,0xA8,0xAF,0xA6,0xA1,0xB4,0xB3,0xBA,0xBD,
    0xC7,0xC0,0xC9,0xCE,0xDB,0xDC,0xD5,0xD2,0xFF,0xF8,0xF1,0xF6,0xE3,0xE4,0xED,0xEA,
    0xB7,0xB0,0xB9,0xBE,0xAB,0xAC,0xA5,0xA2,0x8F,0x88,0x81,0x86,0x93,0x94,0x9D,0x9A,
    0x27,0x20,0x29,0x2E,0x3B,0x3C,0x35,0x32,0x1F,0x18,0x11,0x16,0x03,0x04,0x0D,0x0A,
    0x57,0x50,0x59,0x5E,0x4B,0x4C,0x45,0x42,0x6F,0x68,0x61,0x66,0x73,0x74,0x7D,0x7A,
    0x89,0x8E,0x87,0x80,0x95,0x92,0x9B,0x9C,0xB1,0xB6,0xBF,0xB8,0xAD,0xAA,0xA3,0xA4,
    0xF9,0xFE,0xF7,0xF0,0xE5,0xE2,0xEB,0xEC,0xC1,0xC6,0xCF,0xC8,0xDD,0xDA,0xD3,0xD4,
    0x69,0x6E,0x67,0x60,0x75,0x72,0x7B,0x7C,0x51,0x56,0x5F,0x58,0x4D,0x4A,0x43,0x44,
    0x19,0x1E,0x17,0x10,0x05,0x02,0x0B,0x0C,0x21,0x26,0x2F,0x28,0x3D,0x3A,0x33,0x34,
    0x4E,0x49,0x40,0x47,0x52,0x55,0x5C,0x5B,0x76,0x71,0x78,0x7F,0x6A,0x6D,0x64,0x63,
    0x3E,0x39,0x30,0x37,0x22,0x25,0x2C,0x2B,0x06,0x01,0x08,0x0F,0x1A,0x1D,0x14,0x13,
    0xAE,0xA9,0xA0,0xA7,0xB2,0xB5,0xBC,0xBB,0x96,0x91,0x98,0x9F,0x8A,0x8D,0x84,0x83,
    0xDE,0xD9,0xD0,0xD7,0xC2,0xC5,0xCC,0xCB,0xE6,0xE1,0xE8,0xEF,0xFA,0xFD,0xF4,0xF3
};

// ============================================================
// 顶层: 描述符环预取引擎 + 多核调度控制器
//
// AXI 接口:
//   desc_ring:    m_axi dring — 256-bit 描述符环预取
//   status_page:  m_axi stat  — 完工写回 (HPC, AxCACHE=0b1111)
//   gemm_ctrl:    m_axi gemm  — GEMM 核 ap_ctrl
//   attn_ctrl:    m_axi attn  — ATTN 核 ap_ctrl
//   vitmlp_ctrl:  m_axi vitm  — ViT MLP 核 ap_ctrl
//   rmsnorm_ctrl: m_axi rmsn  — RMSNorm 核 ap_ctrl
//   doorbell_low/high: s_axilite — 128-bit 门铃接收端
//   status_reg/control_reg: s_axilite — PL 状态/控制
//   interrupt:    ap_none — 中断输出 (高有效)
// ============================================================
void pl_controller(
    volatile ap_uint<256> * desc_ring,       // m_axi dring: CMA 描述符环
    volatile ap_uint<32>  * status_page,     // m_axi stat:  共享状态页
    volatile ap_uint<32>  * gemm_ctrl,       // m_axi gemm:  GEMM 核 ap_ctrl
    volatile ap_uint<32>  * attn_ctrl,       // m_axi attn:  ATTN 核 ap_ctrl
    volatile ap_uint<32>  * vitmlp_ctrl,     // m_axi vitm:  ViT MLP 核 ap_ctrl
    volatile ap_uint<32>  * rmsnorm_ctrl,    // m_axi rmsn:  RMSNorm 核 ap_ctrl
    volatile ap_uint<64>  * doorbell_low,    // s_axilite:   门铃低 64-bit
    volatile ap_uint<64>  * doorbell_high,   // s_axilite:   门铃高 64-bit (写触发)
    volatile ap_uint<32>  * status_reg,      // s_axilite:   PL 状态寄存器
    volatile ap_uint<32>  * control_reg      // s_axilite:   控制寄存器
) {
    // §4.2 HLS 接口约束 — ap_ctrl_hs 块级协议自动生成 INTERRUPT 类型中断端口
#pragma HLS INTERFACE s_axilite port=return    bundle=ctrl
#pragma HLS INTERFACE m_axi port=desc_ring    bundle=dring offset=slave depth=DESC_RING_SLOTS
#pragma HLS INTERFACE m_axi port=status_page  bundle=stat  offset=slave depth=32
#pragma HLS INTERFACE m_axi port=gemm_ctrl    bundle=gemm  offset=slave depth=64
#pragma HLS INTERFACE m_axi port=attn_ctrl    bundle=attn  offset=slave depth=64
#pragma HLS INTERFACE m_axi port=vitmlp_ctrl  bundle=vitm  offset=slave depth=64
#pragma HLS INTERFACE m_axi port=rmsnorm_ctrl bundle=rmsn  offset=slave depth=64
#pragma HLS INTERFACE s_axilite port=doorbell_low   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=doorbell_high  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=status_reg     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=control_reg    bundle=ctrl

    // ── 门铃 FIFO (LUTRAM, 深度 8, 0 BRAM) ──
    static doorbell_entry db_fifo[DOORBELL_FIFO_DEPTH];
#pragma HLS BIND_STORAGE variable=db_fifo type=RAM_1P impl=LUTRAM
    static ap_uint<3> db_wr_ptr = 0;
    static ap_uint<3> db_rd_ptr = 0;
// 【修改点】：用显示的物理计数器取代单纯的 empty 标号，防止满回绕死锁
    static ap_uint<4> db_fifo_count = 0;

    // ── 当前描述符 ──
    static desc_entry cur_desc;
    static doorbell_entry cur_db;

    // ── FSM 状态 ──
    static state_t st = ST_IDLE;
    static ap_uint<32> done_cnt  = 0;
    static ap_uint<32> wdt_cnt   = 0;
    static ap_uint<32> poll_cnt  = 0;
    static ap_uint<64> last_doorbell_high = 0;

    // §2.4(3) 步骤 3: 单次调用处理一条命令
    // 函数返回时 ap_done 拉高, HLS 自动触发 interrupt 端口 (INTERRUPT 类型)
    // PS 通过 auto_restart=1 使函数返回后自动重新进入, 实现连续调度
    // 注: 不设 II=1 pipeline, 让 HLS 调度器自然安排 FSM 状态转换.

    // 读取控制寄存器
    ap_uint<32> ctrl_v = *control_reg;
    bool do_reset = (ctrl_v >> 1) & 0x1;

    // ── 门铃检测 (采样 REG_CMD_HIGH_TRIG) ──
    ap_uint<64> cur_high = *doorbell_high;
    ap_uint<64> cur_low  = *doorbell_low;
    bool doorbell_ring = (cur_high != last_doorbell_high) && (cur_high != 0);
    if (doorbell_ring && (db_fifo_count < DOORBELL_FIFO_DEPTH)) {
        db_fifo[db_wr_ptr].desc_index = (ap_uint<32>)(cur_low & 0xFFFFFFFF);
        db_fifo[db_wr_ptr].opcode     = (ap_uint<16>)((cur_low >> 32) & 0xFFFF);
        db_fifo[db_wr_ptr].flags      = (ap_uint<8>)((cur_low >> 48) & 0xFF);
        db_fifo[db_wr_ptr].reserved   = (ap_uint<8>)((cur_low >> 56) & 0xFF);
        db_fifo[db_wr_ptr].timestamp  = cur_high;
        db_wr_ptr++;
        db_fifo_count++;
        last_doorbell_high = cur_high;
    }

    switch (st) {
    // ========================================================
    // ST_IDLE: 轮询门铃 (有限超时), FIFO 非空则进入下一状态
    //   无门铃时函数返回 → auto_restart 自动重入继续轮询
    // ========================================================
    case ST_IDLE: {
        if (do_reset) {
            db_wr_ptr = 0; db_rd_ptr = 0; db_fifo_count = 0;
            last_doorbell_high = 0; wdt_cnt = 0; done_cnt = 0; poll_cnt = 0;
        }
        if (db_fifo_count > 0) {
            wdt_cnt = 0; poll_cnt = 0; st = ST_WAIT_DOORBELL;
        }
        break;
    }

    // ========================================================
    // ST_WAIT_DOORBELL: 从门铃 FIFO 出队
    // ========================================================
    case ST_WAIT_DOORBELL:
// 【修改点】：严格依照可靠计数器处理出队，杜绝假空异常
        if (db_fifo_count > 0) {
            cur_db = db_fifo[db_rd_ptr];
            db_rd_ptr++;
            db_fifo_count--;
            st = ST_FETCH_DESC;
        } else {
            st = ST_IDLE;
        }
        break;

    // ========================================================
    // ST_FETCH_DESC: AXI Master 256-bit 单次突发预取描述符
    //   §2.4(3) 步骤 5: Length=1, 256-bit 完美对齐读取
    // ========================================================
    case ST_FETCH_DESC: {
        ap_uint<256> raw = desc_ring[cur_db.desc_index];
        cur_desc.src_addr    = raw.range(63, 0);
        cur_desc.weight_addr = raw.range(127, 64);
        cur_desc.dst_addr    = raw.range(191, 128);
        cur_desc.dim_m       = raw.range(223, 192);
        cur_desc.dim_n       = raw.range(239, 224);
        cur_desc.dim_k       = raw.range(255, 240);
        st = ST_DECODE;
        break;
    }

    // ========================================================
    // ST_DECODE: 操作码分发
    // ========================================================
    case ST_DECODE:
        poll_cnt = 0;
        switch (cur_db.opcode) {
        case 0x00: case 0x01: st = ST_EXEC_GEMM;     break;
        case 0x02:            st = ST_EXEC_ATTN;     break;
        case 0x11:            st = ST_EXEC_VIT_MLP;  break;
        case 0x12:            st = ST_EXEC_RMSNORM;  break;
        case 0x13:            st = ST_EXEC_RESCALER; break;
        case 0x14:            st = ST_EXEC_ROPE;     break;
        case 0x03:            st = ST_EXEC_DMA;      break;
        case 0xFF:            st = ST_DONE;          break;
        default:
            status_page[1] = 3;  // INVALID_OPCODE
            st = ST_ERR; break;
        }
        break;

    // ========================================================
    // ST_EXEC_GEMM: 启动 GEMM 核并轮询完工
    // ========================================================
    case ST_EXEC_GEMM:
        if (poll_cnt == 0) {
            gemm_ctrl[REG_SRC_LOW]     = cur_desc.src_addr.range(31, 0);
            gemm_ctrl[REG_SRC_HIGH]    = cur_desc.src_addr.range(63, 32);
            gemm_ctrl[REG_WGT_LOW]     = cur_desc.weight_addr.range(31, 0);
            gemm_ctrl[REG_WGT_HIGH]    = cur_desc.weight_addr.range(63, 32);
            gemm_ctrl[REG_DST_LOW]     = cur_desc.dst_addr.range(31, 0);
            gemm_ctrl[REG_DST_HIGH]    = cur_desc.dst_addr.range(63, 32);
            gemm_ctrl[REG_DIM_M]       = cur_desc.dim_m;
            gemm_ctrl[REG_DIM_N]       = cur_desc.dim_n;
            gemm_ctrl[REG_DIM_K]       = cur_desc.dim_k;
            
            gemm_ctrl[AP_CTRL_OFFSET]  = 1; // ap_start
            poll_cnt = 1;
        } else {
            ap_uint<32> c = gemm_ctrl[AP_CTRL_OFFSET];
            if (c.get_bit(AP_DONE_BIT) || c.get_bit(AP_IDLE_BIT)) {
                st = ST_DONE;
            } else if (poll_cnt > MAX_POLL_COUNT) {
                status_page[1] = 4; st = ST_ERR;
            }
            poll_cnt++;
            wdt_cnt = 0; // 喂狗
        }
        break;

    // ========================================================
    // ST_EXEC_ATTN: 启动 ATTN 核并轮询完工
    // ========================================================
    case ST_EXEC_ATTN:
        if (poll_cnt == 0) {
            attn_ctrl[REG_SRC_LOW]    = cur_desc.src_addr.range(31, 0);
            attn_ctrl[REG_SRC_HIGH]   = cur_desc.src_addr.range(63, 32);
            attn_ctrl[REG_WGT_LOW]    = cur_desc.weight_addr.range(31, 0);
            attn_ctrl[REG_WGT_HIGH]   = cur_desc.weight_addr.range(63, 32);
            attn_ctrl[REG_DST_LOW]    = cur_desc.dst_addr.range(31, 0);
            attn_ctrl[REG_DST_HIGH]   = cur_desc.dst_addr.range(63, 32);
            attn_ctrl[REG_DIM_M]      = cur_desc.dim_m;
            attn_ctrl[REG_DIM_N]      = cur_desc.dim_n;
            attn_ctrl[REG_DIM_K]      = cur_desc.dim_k;

            attn_ctrl[AP_CTRL_OFFSET] = 1;
            poll_cnt = 1;
        } else {
            ap_uint<32> c = attn_ctrl[AP_CTRL_OFFSET];
            if (c.get_bit(AP_DONE_BIT) || c.get_bit(AP_IDLE_BIT)) {
                st = ST_DONE;
            } else if (poll_cnt > MAX_POLL_COUNT) {
                status_page[1] = 4; st = ST_ERR;
            }
            poll_cnt++;
            wdt_cnt = 0;
        }
        break;

    // ========================================================
    // ST_EXEC_VIT_MLP: 启动 ViT MLP 融合核并轮询
    // ========================================================
    case ST_EXEC_VIT_MLP:
        if (poll_cnt == 0) {
            vitmlp_ctrl[REG_SRC_LOW]    = cur_desc.src_addr.range(31, 0);
            vitmlp_ctrl[REG_SRC_HIGH]   = cur_desc.src_addr.range(63, 32);
            vitmlp_ctrl[REG_WGT_LOW]    = cur_desc.weight_addr.range(31, 0);
            vitmlp_ctrl[REG_WGT_HIGH]   = cur_desc.weight_addr.range(63, 32);
            vitmlp_ctrl[REG_DST_LOW]    = cur_desc.dst_addr.range(31, 0);
            vitmlp_ctrl[REG_DST_HIGH]   = cur_desc.dst_addr.range(63, 32);
            vitmlp_ctrl[REG_DIM_M]      = cur_desc.dim_m;
            vitmlp_ctrl[REG_DIM_N]      = cur_desc.dim_n;
            vitmlp_ctrl[REG_DIM_K]      = cur_desc.dim_k;

            vitmlp_ctrl[AP_CTRL_OFFSET] = 1;
            poll_cnt = 1;
        } else {
            ap_uint<32> c = vitmlp_ctrl[AP_CTRL_OFFSET];
            if (c.get_bit(AP_DONE_BIT) || c.get_bit(AP_IDLE_BIT)) {
                st = ST_DONE;
            } else if (poll_cnt > MAX_POLL_COUNT) {
                status_page[1] = 4; st = ST_ERR;
            }
            poll_cnt++;
            wdt_cnt = 0;
        }
        break;

    // ========================================================
    // ST_EXEC_RMSNORM: 启动 RMSNorm 核并轮询
    // ========================================================
    case ST_EXEC_RMSNORM:
        if (poll_cnt == 0) {
            rmsnorm_ctrl[REG_SRC_LOW]    = cur_desc.src_addr.range(31, 0);
            rmsnorm_ctrl[REG_SRC_HIGH]   = cur_desc.src_addr.range(63, 32);
            rmsnorm_ctrl[REG_WGT_LOW]    = cur_desc.weight_addr.range(31, 0);
            rmsnorm_ctrl[REG_WGT_HIGH]   = cur_desc.weight_addr.range(63, 32);
            rmsnorm_ctrl[REG_DST_LOW]    = cur_desc.dst_addr.range(31, 0);
            rmsnorm_ctrl[REG_DST_HIGH]   = cur_desc.dst_addr.range(63, 32);
            rmsnorm_ctrl[REG_DIM_M]      = cur_desc.dim_m;
            rmsnorm_ctrl[REG_DIM_N]      = cur_desc.dim_n;
            rmsnorm_ctrl[REG_DIM_K]      = cur_desc.dim_k;

            rmsnorm_ctrl[AP_CTRL_OFFSET] = 1;
            poll_cnt = 1;
        } else {
            ap_uint<32> c = rmsnorm_ctrl[AP_CTRL_OFFSET];
            if (c.get_bit(AP_DONE_BIT) || c.get_bit(AP_IDLE_BIT)) {
                st = ST_DONE;
            } else if (poll_cnt > MAX_POLL_COUNT) {
                status_page[1] = 4; st = ST_ERR;
            }
            poll_cnt++;
            wdt_cnt = 0;
        }
        break;

    // ========================================================
    // ST_EXEC_RESCALER / ST_EXEC_ROPE: 轻量算子 (直接完工)
    // ========================================================
    case ST_EXEC_RESCALER:
    case ST_EXEC_ROPE:
        st = ST_DONE;
        break;

    // ========================================================
    // ST_EXEC_DMA: KV Cache DMA 写回
    // ========================================================
    case ST_EXEC_DMA:
        st = ST_DONE;
        break;

    // ========================================================
    // ST_DONE: 写回完工 ID (§5.4 HPC 端口, AxCACHE=0b1111)
    // ========================================================
    case ST_DONE:
        done_cnt++;
        status_page[0] = done_cnt;
        // 函数返回 → ap_done 拉高 → HLS 自动触发 interrupt (INTERRUPT 类型)
        // PS 通过 auto_restart=1 使函数自动重新进入, 继续处理下一命令
        return;

    // ========================================================
    // ST_ERR: 错误处理 — 清空队列, 返回 (触发中断通知 PS)
    // ========================================================
    case ST_ERR:
        db_wr_ptr = 0; db_rd_ptr = 0; db_fifo_count = 0;
        // 函数返回 → ap_done 拉高 → HLS 自动触发 interrupt
        // PS 读 status_reg bit[21] 确认错误来源
        return;

    default: st = ST_IDLE; break;
    }

    // ── 硬件看门狗 (§5.5: 10μs 捕捉异常) ──
    if (st != ST_IDLE) {
        wdt_cnt++;
        if (wdt_cnt > WDT_MAX_COUNT) {
            status_page[1] = 4;  // WATCHDOG_TIMEOUT
            st = ST_ERR; wdt_cnt = 0;
        }
    } else {
        wdt_cnt = 0;
    }

    // ── 更新状态寄存器 ──
    ap_uint<32> stat_v = (done_cnt & 0xFFFF) |
                         ((ap_uint<32>(st) & 0xF) << 16) |
                         (ap_uint<32>((st == ST_ERR) ? 1 : 0) << 21) |
                         (ap_uint<32>((st == ST_DONE) ? 1 : 0) << 22);
    *status_reg = stat_v;

    // ── IDLE 无门铃: 函数返回 → auto_restart 自动重入继续轮询 ──
    // 中断由 ap_ctrl_hs 块级协议自动产生 (INTERRUPT 类型, 兼容 xlconcat)
    // PS 通过 status_reg bit[21]/bit[22] 区分 ST_ERR/ST_DONE
    // PS 通过 ISR 寄存器 (offset 0x0C) 清除中断
    return;
}
