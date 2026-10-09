// dense_engine.hpp --- V6 prefill dense / QKV-fused engine.
//
// Outer loop hierarchy (mt outermost so activation cache is reused):
//   for mt in m_tiles:
//     load_activation_block(K)            // ~5130 cycles for K=2560 (paid once / mt)
//     for out_idx in output_count:
//       for nt in n_tiles[out_idx]:
//         for kt in k_tiles:
//           load_weight_kt_dual + scales            (~134 cycles, DDR-bound)
//           drive 128 cycles of (a/w/ctrl)          (commit once per 128-K tile)
//           PE autonomously drains 128 psum beats   (overlaps next K-tile)
//           absorb_psum_drain (apply scale, fold)
//         store_out_acc                             (~256 cycles)
//
// Stream interfaces are AXIS ports the BD wires through to the pe_array_v6 IP.
#pragma once

#include "axi_io.hpp"
#include "act_cache.hpp"
#include "weight_loader.hpp"
#include "scale_loader.hpp"
#include "pe_driver.hpp"
#include "scale_drain.hpp"
#include "bias_loader.hpp"
#ifdef V6_CSIM
#include "../csim/pe_behavioral.hpp"
#else
#define V6_CSIM_PUMP_PE(a_stream, w0_stream, w1_stream, ctrl_stream, psum_stream) ((void)0)
#endif

namespace v6 {

/**
 * -------------------------------------------------------------------------------------
 * 函数：load_weight_scale_tile
 * 作用：从外部 DDR 中加载一个单独的权重矩阵 KxN-Tile 块及其对应的量化缩放因子(Scale)。
 *       加载的数据被放入局部的片上缓存（wbuf 和 sbuf），以备后续计算单元进行高速读取。
 * -------------------------------------------------------------------------------------
 */
static void load_weight_scale_tile(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        uint64_t w0_kt_base_words,
        uint64_t w1_kt_base_words,
        uint64_t w0_kt_scale_words,
        uint64_t w1_kt_scale_words,
        WeightBuf &wbuf,
        ScaleBuf &sbuf) {
    #pragma HLS INLINE off
    // 1. 先加载当前 K tile 的 8-bit 权重。w0/w1 两个 DDR port 分别对应 N tile 的低/高 64 列。
    load_weight_kt_dual(w0_arena, w1_arena, w0_kt_base_words, w1_kt_base_words, wbuf);

    // 2. 再加载同一个 K tile 的 scale metadata。后续 psum drain 需要它完成 Q22 反量化。
    load_weight_scale_kt(w0_arena, w1_arena, w0_kt_scale_words, w1_kt_scale_words, sbuf);
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：drive_compute_tile
 * 作用：核心的数据推流驱动。将放在片上的激活值(ActCache)与当前权重块(WeightBuf)结合，
 *       按照硬件脉动阵列（PE Array）要求的位宽进行拆解封装，利用 AXI-Stream 发送给计算引擎。
 *       同时控制 ctrl_stream 发送指令（区分开始、计算中和结束）。
 * -------------------------------------------------------------------------------------
 */
static void drive_compute_tile(
        const ActCache &act,
        const WeightBuf &wbuf,
        const ScaleBuf &sbuf,
        uint32_t act_word_base,
        uint32_t kt,
        bool final_kt,
        scale_stream_t &scale_stream,
        a_stream_t   &a_stream,
        w_stream_t   &w0_stream,
        w_stream_t   &w1_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=2

    // 1. 将量化组所需的比例因子(Scale Groups)推送给 Scale Drain 模块，因为反量化在 PE 阵列计算之后进行
    push_scale_groups(sbuf, scale_stream);

    // 2. 逐周期（k 维度）发送数据：对于每一个 K 维的元素，通过 Pipeline 流水线并发组装控制字与数据
    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        const uint32_t k_idx  = kt * kTileK + k;
        a_stream_word_t a_beat = 0;
        w_half_stream_word_t w0_beat = 0;
        w_half_stream_word_t w1_beat = 0;
        
        // 2.1 组装激活值心跳 (Activation Beat)：将 kSaM 行的所有 8-bit 或者所需宽度的数据打包到一个大位宽字中
        for (uint32_t r = 0; r < kSaM; ++r) {
            #pragma HLS UNROLL
            const ap_uint<64> a_word = act.buf[r][act_word_base + k_idx / 8];
            const uint32_t a_bidx = k_idx & 7u;
            a_beat(r * 8 + 7, r * 8) = a_word(a_bidx * 8 + 7, a_bidx * 8);
        }
        
        // 2.2 组装权重心跳 (Weight Beat)：双端口 w0_beat和 w1_beat 充分利用总线位宽拉取权重
        for (uint32_t w = 0; w < kWeightWordsPerKPerPort; ++w) {
            #pragma HLS UNROLL
            w0_beat(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) =
                wbuf.words[0][k][w];
            w1_beat(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) =
                wbuf.words[1][k][w];
        }
        
        // 2.3 生成 PE 阵列的控制信令 (Control Tokens)
        // Group 起点标识清除上一次计算的累加器，终点标识输出本 Group 结果
        const bool group_start = ((k & (kGroupSize - 1u)) == 0u);
        const bool group_end   = ((k & (kGroupSize - 1u)) == (kGroupSize - 1u));
        const bool kt_end      = (k == kTileK - 1u);
        
        // 2.4 将这一周期的所有就绪数据压入 AXI-Stream 被 PE Array 消耗
        a_stream.write(a_beat);
        w0_stream.write(w0_beat);
        w1_stream.write(w1_beat);
        ctrl_stream.write(make_ctrl(/*clear=*/group_start, true,
                                    /*commit=*/group_end,
                                    /*last=*/final_kt && kt_end));
    }
}

typedef ap_uint<2 * kAxiWidth> weight_pair_word_t;
typedef weight_pair_word_t scale_pair_word_t;
typedef hls::stream<weight_pair_word_t> weight_pair_stream_t;
typedef hls::stream<scale_pair_word_t> scale_pair_stream_t;

/**
 * -------------------------------------------------------------------------------------
 * 函数：pack_axi_pair
 * 作用：把两个 128-bit AXI word 合成一个 256-bit stream token。
 *       v6 的权重有两个 DDR/AXI port，w0 和 w1 分别提供 64 个输出列；
 *       loader stage 使用这个 token 保持两个 port 的数据同步进入 compute-owner。
 * -------------------------------------------------------------------------------------
 */
static inline weight_pair_word_t pack_axi_pair(axi_word_t lo, axi_word_t hi) {
    #pragma HLS INLINE
    weight_pair_word_t v = 0;
    v(kAxiWidth - 1, 0) = lo;
    v(2 * kAxiWidth - 1, kAxiWidth) = hi;
    return v;
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：pair_lo
 * 作用：取出 pair token 中低 128-bit 的 AXI word，对应 w0 port 数据。
 * -------------------------------------------------------------------------------------
 */
static inline axi_word_t pair_lo(weight_pair_word_t v) {
    #pragma HLS INLINE
    return (axi_word_t)v(kAxiWidth - 1, 0);
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：pair_hi
 * 作用：取出 pair token 中高 128-bit 的 AXI word，对应 w1 port 数据。
 * -------------------------------------------------------------------------------------
 */
static inline axi_word_t pair_hi(weight_pair_word_t v) {
    #pragma HLS INLINE
    return (axi_word_t)v(2 * kAxiWidth - 1, kAxiWidth);
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：store_weight_pair_token
 * 作用：把一个 256-bit weight pair token 写入 compute-owner 私有的 WeightBuf bank。
 *
 * 阶段说明：
 *   1. i 是当前 K tile 内的 pair-token 序号。
 *   2. i / kWeightWordsPerKPerPort 定位 K 行，也就是 PE 阵列第几个 k-cycle 会读取。
 *   3. i % kWeightWordsPerKPerPort 定位同一 K 行内的 128-bit word lane。
 *   4. token 的低/高 128-bit 分别落到 port0/port1，保持 8x128 PE 的列方向布局。
 * -------------------------------------------------------------------------------------
 */
static inline void store_weight_pair_token(
        weight_pair_word_t token,
        uint32_t i,
        WeightBuf &dst) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=dst.words complete dim=1
    const uint32_t k = i / kWeightWordsPerKPerPort;
    const uint32_t w = i % kWeightWordsPerKPerPort;
    dst.words[0][k][w] = pair_lo(token);
    dst.words[1][k][w] = pair_hi(token);
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：store_scale_pair_token
 * 作用：把一个 256-bit scale pair token 解包到 ScaleBuf。
 *
 * 阶段说明：
 *   1. v0/v1 分别对应两个 weight port 的 scale 字节流。
 *   2. scale layout 以 output column lane 和 group 为索引，当前只消费 e0/POT exp。
 *   3. 每个 AXI byte 被映射到 (lane, group)，再写入 sbuf.e0[group][lane]。
 *   4. w1 port 的列号整体偏移 kWeightColsPerPort，拼成完整 128 列 N tile。
 * -------------------------------------------------------------------------------------
 */
static inline void store_scale_pair_token(
        scale_pair_word_t token,
        uint32_t i,
        ScaleBuf &dst) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=dst.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dst.e0 complete dim=2
    const axi_word_t v0 = pair_lo(token);
    const axi_word_t v1 = pair_hi(token);
    for (uint32_t b = 0; b < kAxiBytes; ++b) {
        #pragma HLS UNROLL
        const uint32_t flat = i * kAxiBytes + b;
        const uint32_t entry = flat / kScaleBytesPerSlot;
        const uint32_t which_e = flat % kScaleBytesPerSlot;
        const uint32_t lane = entry / kGroupsPerKtile;
        const uint32_t group = entry % kGroupsPerKtile;
        if (which_e == 0u && lane < kWeightColsPerPort && group < kGroupsPerKtile) {
            const int8_t x0 = (int8_t)(uint8_t)v0(b * 8 + 7, b * 8);
            const int8_t x1 = (int8_t)(uint8_t)v1(b * 8 + 7, b * 8);
            dst.e0[group][lane]                      = x0;
            dst.e0[group][lane + kWeightColsPerPort] = x1;
        }

    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：weight_stream_load_stage
 * 作用：真正 ping-pong 的独立 DDR loader process。
 *
 * 阶段说明：
 *   1. 按 K tile 顺序计算当前 N tile 下的 weight/scale DDR word base。
 *   2. 先发 scale pair token，再发 weight pair token，保证 compute-owner 能先建好 sbuf。
 *   3. loader 只做 m_axi 读和 stream 写，不接触本地 weight bank。
 *   4. compute-owner 通过 stream 接收 token，因此 wbuf/sbuf 不跨 dataflow process 共享，
 *      避免 HLS 把数组降成 PIPO/FIFO 后造成 slab 间 token 被消费的问题。
 * -------------------------------------------------------------------------------------
 */
static void weight_stream_load_stage(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        scale_pair_stream_t &scale_pair_stream,
        weight_pair_stream_t &weight_pair_stream) {
    #pragma HLS INLINE off
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        const uint64_t w0_kt_base = w0_nt_base_words + (uint64_t)kt * kWeightWordsPerKtilePerPort;
        const uint64_t w1_kt_base = w1_nt_base_words + (uint64_t)kt * kWeightWordsPerKtilePerPort;
        const uint64_t w0_kt_scale = w0_nt_scale_words + (uint64_t)kt * kScaleWordsPerKtilePerPort;
        const uint64_t w1_kt_scale = w1_nt_scale_words + (uint64_t)kt * kScaleWordsPerKtilePerPort;

        for (uint32_t i = 0; i < kScaleWordsPerKtilePerPort; ++i) {
            #pragma HLS PIPELINE II=1
            scale_pair_stream.write(pack_axi_pair(w0_arena[w0_kt_scale + i],
                                                  w1_arena[w1_kt_scale + i]));
        }
        for (uint32_t i = 0; i < kWeightWordsPerKtilePerPort; ++i) {
            #pragma HLS PIPELINE II=1
            weight_pair_stream.write(pack_axi_pair(w0_arena[w0_kt_base + i],
                                                   w1_arena[w1_kt_base + i]));
        }
    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：try_fill_bank_from_stream_nb
 * 作用：非阻塞地从 scale/weight stream 中抓取 token，机会式填充 inactive bank。
 *
 * 阶段说明：
 *   1. 优先填 scale，因为 compute replay 需要先把当前 K tile 的 scale 推给 drain。
 *   2. scale 填满后再填 weight token。
 *   3. read_nb 失败时立即返回，不阻塞 PE 的 a/w/ctrl stream 推送。
 *   4. 这个函数适合“尽量预取但不能拖慢 compute”的路径。
 * -------------------------------------------------------------------------------------
 */
static inline void try_fill_bank_from_stream_nb(
        scale_pair_stream_t &scale_pair_stream,
        weight_pair_stream_t &weight_pair_stream,
        WeightBuf &wbuf,
        ScaleBuf &sbuf,
        uint32_t &scale_i,
        uint32_t &weight_i) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=2
    if (scale_i < kScaleWordsPerKtilePerPort) {
        scale_pair_word_t token;
        if (scale_pair_stream.read_nb(token)) {
            store_scale_pair_token(token, scale_i, sbuf);
            ++scale_i;
        }
    }
    if (scale_i >= kScaleWordsPerKtilePerPort && weight_i < kWeightWordsPerKtilePerPort) {
        weight_pair_word_t token;
        if (weight_pair_stream.read_nb(token)) {
            store_weight_pair_token(token, weight_i, wbuf);
            ++weight_i;
        }
    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：fill_bank_from_stream_blocking_step
 * 作用：在 full BM4 ping-pong 路径中，每个 compute k-cycle 同步消费一个预取 token。
 *
 * 阶段说明：
 *   1. scale 和 weight 是两个独立 stream；scale 先补，weight 每个 compute cycle 都补一个。
 *   2. 不能等 scale 全部读完才读 weight，否则 512 个 compute cycle 只能填 509/512 个 weight token。
 *   3. 这里允许阻塞，因为 full BM4 下 loader 应该已经和当前 active bank 的 compute 重叠；
 *      如果阻塞明显，Gate5 的 weight_compute overlap 会暴露问题。
 *   4. 这样可以让 next bank 在 4 个 slab 的 512 个 compute cycle 内完整填满。
 * -------------------------------------------------------------------------------------
 */
static inline void fill_bank_from_stream_blocking_step(
        scale_pair_stream_t &scale_pair_stream,
        weight_pair_stream_t &weight_pair_stream,
        WeightBuf &wbuf,
        ScaleBuf &sbuf,
        uint32_t &scale_i,
        uint32_t &weight_i) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=2
    if (scale_i < kScaleWordsPerKtilePerPort) {
        const scale_pair_word_t token = scale_pair_stream.read();
        store_scale_pair_token(token, scale_i, sbuf);
        ++scale_i;
    }
    if (weight_i < kWeightWordsPerKtilePerPort) {
        const weight_pair_word_t token = weight_pair_stream.read();
        store_weight_pair_token(token, weight_i, wbuf);
        ++weight_i;
    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：finish_bank_from_stream
 * 作用：补齐一个 bank 剩余的 scale/weight token，直到该 bank 可用于 compute replay。
 *
 * 阶段说明：
 *   1. 先补 scale token，再补 weight token，保持和 loader 发送协议一致。
 *   2. 每个补齐循环保持 II=1，避免在初始 bank0 或尾部残留 token 阶段退化。
 *   3. 初始 K tile 必须完整阻塞填满 bank0；后续 K tile 理想情况下只剩少量 token 需要补齐。
 * -------------------------------------------------------------------------------------
 */
static void finish_bank_from_stream(
        scale_pair_stream_t &scale_pair_stream,
        weight_pair_stream_t &weight_pair_stream,
        WeightBuf &wbuf,
        ScaleBuf &sbuf,
        uint32_t &scale_i,
        uint32_t &weight_i) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=2

    for (; scale_i < kScaleWordsPerKtilePerPort; ++scale_i) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=kScaleWordsPerKtilePerPort
        #pragma HLS PIPELINE II=1
        const scale_pair_word_t token = scale_pair_stream.read();
        store_scale_pair_token(token, scale_i, sbuf);
    }
    for (; weight_i < kWeightWordsPerKtilePerPort; ++weight_i) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=kWeightWordsPerKtilePerPort
        #pragma HLS PIPELINE II=1
        const weight_pair_word_t token = weight_pair_stream.read();
        store_weight_pair_token(token, weight_i, wbuf);
    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：drive_compute_tile_with_stream_prefetch
 * 作用：用 active bank 驱动一个 slab 的 K tile 计算，同时非阻塞预取 next bank。
 *
 * 阶段说明：
 *   1. push_scale_groups：把 active bank 的 scale exp 送往 psum absorb 侧。
 *   2. 逐 K cycle 从 ActCache 和 WeightBuf 组装 a/w0/w1 stream beat。
 *   3. 每个 group 的起止位置通过 ctrl_stream 标记 clear/commit/last。
 *   4. 如果 do_prefetch 为真，则每个 compute cycle 尝试从 loader stream 读取 token，
 *      但 read_nb 不允许阻塞 active bank 的计算流。
 * -------------------------------------------------------------------------------------
 */
static void drive_compute_tile_with_stream_prefetch(
        const ActCache &act,
        const WeightBuf &wbuf,
        const ScaleBuf &sbuf,
        uint32_t act_word_base,
        uint32_t kt,
        bool final_kt,
        bool do_prefetch,
        scale_pair_stream_t &scale_pair_stream_in,
        weight_pair_stream_t &weight_pair_stream,
        WeightBuf &next_wbuf,
        ScaleBuf &next_sbuf,
        uint32_t &next_scale_i,
        uint32_t &next_weight_i,
        scale_stream_t &scale_stream,
        a_stream_t &a_stream,
        w_stream_t &w0_stream,
        w_stream_t &w1_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=next_wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=2

    // 1. 发送量化缩放因子供下游 Scale Drain 使用
    push_scale_groups(sbuf, scale_stream);

    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        const uint32_t k_idx = kt * kTileK + k;
        a_stream_word_t a_beat = 0;
        w_half_stream_word_t w0_beat = 0;
        w_half_stream_word_t w1_beat = 0;
        
        // 2. 构造当前时钟周期的激活块
        for (uint32_t r = 0; r < kSaM; ++r) {
            #pragma HLS UNROLL
            const ap_uint<64> a_word = act.buf[r][act_word_base + k_idx / 8];
            const uint32_t a_bidx = k_idx & 7u;
            a_beat(r * 8 + 7, r * 8) = a_word(a_bidx * 8 + 7, a_bidx * 8);
        }
        
        // 3. 提取当前时刻参与运算的局部权重
        for (uint32_t w = 0; w < kWeightWordsPerKPerPort; ++w) {
            #pragma HLS UNROLL
            w0_beat(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) =
                wbuf.words[0][k][w];
            w1_beat(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) =
                wbuf.words[1][k][w];
        }
        
        // 4. 定好起始/结算边界，并通过 AXI-Stream 网终推入底层脉动阵列
        const bool group_start = ((k & (kGroupSize - 1u)) == 0u);
        const bool group_end   = ((k & (kGroupSize - 1u)) == (kGroupSize - 1u));
        const bool kt_end      = (k == kTileK - 1u);
        a_stream.write(a_beat);
        w0_stream.write(w0_beat);
        w1_stream.write(w1_beat);
        ctrl_stream.write(make_ctrl(/*clear=*/group_start, true,
                                    /*commit=*/group_end,
                                    /*last=*/final_kt && kt_end));
                                    
        if (do_prefetch) {
            try_fill_bank_from_stream_nb(scale_pair_stream_in, weight_pair_stream,
                                         next_wbuf, next_sbuf,
                                         next_scale_i, next_weight_i);
        }
    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：drive_compute_tile_with_stream_prefetch_blocking
 * 作用：full BM4 路径使用的主 compute replay。和非阻塞版本计算语义一致，
 *       但每个 k-cycle 会阻塞式推进 next bank 填充，使 load 更稳定地被 BM4 compute 覆盖。
 *
 * 阶段说明：
 *   1. active bank 的 scale 先进入 scale_stream，和随后 PE 输出的 psum 顺序对齐。
 *   2. 128 个 K cycle 以 II=1 推送 A/W/CTRL，PE 阵列每拍完成 8x128 MAC。
 *   3. 当前 tile 计算时，inactive bank 从 stream 中同步吸收下一 K tile token。
 *   4. 如果 do_prefetch 为假，表示最后一个 K tile，不再填充下一 bank。
 * -------------------------------------------------------------------------------------
 */
static void drive_compute_tile_with_stream_prefetch_blocking(
        const ActCache &act,
        const WeightBuf &wbuf,
        const ScaleBuf &sbuf,
        uint32_t act_word_base,
        uint32_t kt,
        bool final_kt,
        bool do_prefetch,
        scale_pair_stream_t &scale_pair_stream_in,
        weight_pair_stream_t &weight_pair_stream,
        WeightBuf &next_wbuf,
        ScaleBuf &next_sbuf,
        uint32_t &next_scale_i,
        uint32_t &next_weight_i,
        scale_stream_t &scale_stream,
        a_stream_t &a_stream,
        w_stream_t &w0_stream,
        w_stream_t &w1_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=next_wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=2

    push_scale_groups(sbuf, scale_stream);

    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        const uint32_t k_idx = kt * kTileK + k;
        a_stream_word_t a_beat = 0;
        w_half_stream_word_t w0_beat = 0;
        w_half_stream_word_t w1_beat = 0;

        for (uint32_t r = 0; r < kSaM; ++r) {
            #pragma HLS UNROLL
            const ap_uint<64> a_word = act.buf[r][act_word_base + k_idx / 8];
            const uint32_t a_bidx = k_idx & 7u;
            a_beat(r * 8 + 7, r * 8) = a_word(a_bidx * 8 + 7, a_bidx * 8);
        }

        for (uint32_t w = 0; w < kWeightWordsPerKPerPort; ++w) {
            #pragma HLS UNROLL
            w0_beat(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) =
                wbuf.words[0][k][w];
            w1_beat(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) =
                wbuf.words[1][k][w];
        }

        const bool group_start = ((k & (kGroupSize - 1u)) == 0u);
        const bool group_end   = ((k & (kGroupSize - 1u)) == (kGroupSize - 1u));
        const bool kt_end      = (k == kTileK - 1u);
        a_stream.write(a_beat);
        w0_stream.write(w0_beat);
        w1_stream.write(w1_beat);
        ctrl_stream.write(make_ctrl(/*clear=*/group_start, true,
                                    /*commit=*/group_end,
                                    /*last=*/final_kt && kt_end));

        if (do_prefetch) {
            fill_bank_from_stream_blocking_step(scale_pair_stream_in, weight_pair_stream,
                                                next_wbuf, next_sbuf,
                                                next_scale_i, next_weight_i);
        }
    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：compute_stage_pingpong_owner_bm4
 * 作用：真正拥有两个 weight/scale bank 的 compute process。
 *
 * 阶段说明：
 *   1. 初始阶段：阻塞填满 bank0，保证第一个 K tile 有完整 resident weight。
 *   2. K tile 循环：bank0/bank1 交替作为 active bank。
 *   3. slab 循环：同一个 active K tile 权重被 slabs_valid 个 8-row slab 复用。
 *      full BM4 时是 4 个 slab；decode/tail 时可以是 1/2/3 个 slab。
 *   4. 预取阶段：当前 active bank compute 时，inactive bank 从 loader stream 接收下一 K tile。
 *   5. full BM4 下 512 个 compute cycle 正好填满下一 bank 的 512 个 weight token。
 *      小 M/decode 下 compute 周期不足以填满下一 bank，因此 slab 循环后用 finish 补齐剩余 token。
 *   6. bank 数组只在本函数内部声明和传递，不跨 dataflow process 共享，是避免伪 ping-pong 的关键。
 * -------------------------------------------------------------------------------------
 */
static void compute_stage_pingpong_owner_bm4(
        const ActCache &act,
        const uint32_t act_word_base[kBatchMtiles],
        uint32_t k_tiles,
        uint32_t slabs_valid,
        scale_pair_stream_t &scale_pair_stream,
        weight_pair_stream_t &weight_pair_stream,
        scale_stream_t &scale_stream,
        a_stream_t &a_stream,
        w_stream_t &w0_stream,
        w_stream_t &w1_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    WeightBuf wbuf0;
    WeightBuf wbuf1;
    ScaleBuf sbuf0;
    ScaleBuf sbuf1;
    #pragma HLS BIND_STORAGE variable=wbuf0.words type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=wbuf1.words type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=wbuf0.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wbuf1.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 complete dim=2

    if (k_tiles == 0u) {
        return;
    }

    uint32_t init_scale_i = 0;
    uint32_t init_weight_i = 0;
    finish_bank_from_stream(scale_pair_stream, weight_pair_stream,
                            wbuf0, sbuf0, init_scale_i, init_weight_i);

    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        const bool cur_bank0 = ((kt & 1u) == 0u);
        const bool has_next = (kt + 1u < k_tiles);
        uint32_t next_scale_i = 0;
        uint32_t next_weight_i = 0;

        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            #pragma HLS UNROLL off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=kBatchMtiles
            if (cur_bank0) {
                drive_compute_tile_with_stream_prefetch_blocking(
                    act, wbuf0, sbuf0, act_word_base[slab],
                    kt, kt + 1u == k_tiles,
                    has_next,
                    scale_pair_stream, weight_pair_stream,
                    wbuf1, sbuf1,
                    next_scale_i, next_weight_i,
                    scale_stream, a_stream, w0_stream, w1_stream, ctrl_stream);
            } else {
                drive_compute_tile_with_stream_prefetch_blocking(
                    act, wbuf1, sbuf1, act_word_base[slab],
                    kt, kt + 1u == k_tiles,
                    has_next,
                    scale_pair_stream, weight_pair_stream,
                    wbuf0, sbuf0,
                                     next_scale_i, next_weight_i,
                                     scale_stream, a_stream, w0_stream, w1_stream, ctrl_stream);
            }
        }

        if (has_next) {
            if (cur_bank0) {
                finish_bank_from_stream(scale_pair_stream, weight_pair_stream,
                                        wbuf1, sbuf1,
                                        next_scale_i, next_weight_i);
            } else {
                finish_bank_from_stream(scale_pair_stream, weight_pair_stream,
                                        wbuf0, sbuf0,
                                        next_scale_i, next_weight_i);
            }
        }
    }
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：drive_all_ktiles_batch_pp
 * 作用：统一调度模块。综合路径使用独立 DDR loader + compute-owner 双 bank；
 *       C-sim 路径保留顺序实现，只用于数值回归。
 *
 * 阶段说明：
 *   1. V6_CSIM：走串行 load_weight_scale_tile + drive_compute_tile，方便 C 模型验证数值。
 *   2. Synth：创建 scale/weight pair stream，stream 只传 token，不传 bank 数组。
 *   3. Loader process：weight_stream_load_stage 顺序读 DDR 并写 stream。
 *   4. Compute-owner process：compute_stage_pingpong_owner_bm4 独占双 bank 并与 loader dataflow 并行。
 *   5. slabs_valid=1/2/3/4 都走同一套双 bank，避免 tail/decode 另综合一套单 bank。
 * -------------------------------------------------------------------------------------
 */
static void drive_all_ktiles_batch_pp(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        const ActCache &act,
        const uint32_t act_word_base[kBatchMtiles],
        uint32_t slabs_valid,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        scale_stream_t &scale_stream,
        a_stream_t &a_stream,
        w_stream_t &w0_stream,
        w_stream_t &w1_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off

#ifdef V6_CSIM
    WeightBuf wbuf0;
    ScaleBuf sbuf0;
    #pragma HLS BIND_STORAGE variable=wbuf0.words type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=wbuf0.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 complete dim=2

    // C-sim 只需要数值等价，不依赖 HLS dataflow 行为；这里保持最容易调试的串行路径。
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        const uint64_t w0_kt = w0_nt_base_words + (uint64_t)kt * kWeightWordsPerKtilePerPort;
        const uint64_t w1_kt = w1_nt_base_words + (uint64_t)kt * kWeightWordsPerKtilePerPort;
        const uint64_t s0_kt = w0_nt_scale_words + (uint64_t)kt * kScaleWordsPerKtilePerPort;
        const uint64_t s1_kt = w1_nt_scale_words + (uint64_t)kt * kScaleWordsPerKtilePerPort;
        load_weight_scale_tile(w0_arena, w1_arena, w0_kt, w1_kt, s0_kt, s1_kt, wbuf0, sbuf0);

        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=kBatchMtiles
            drive_compute_tile(act, wbuf0, sbuf0, act_word_base[slab],
                               kt, kt + 1u == k_tiles,
                               scale_stream, a_stream, w0_stream, w1_stream, ctrl_stream);
        }
    }
#else
    // 1. 使用小深度 SRL FIFO 承接 loader 和 compute-owner，避免 stream 默认推成 BRAM 导致资源膨胀。
    scale_pair_stream_t scale_pair_stream("v6_scale_pair_stream");
    weight_pair_stream_t weight_pair_stream("v6_weight_pair_stream");
    #pragma HLS STREAM variable=scale_pair_stream depth=8
    #pragma HLS STREAM variable=weight_pair_stream depth=16
    #pragma HLS BIND_STORAGE variable=scale_pair_stream type=fifo impl=srl
    #pragma HLS BIND_STORAGE variable=weight_pair_stream type=fifo impl=srl
    #pragma HLS DATAFLOW disable_start_propagation

    // 2. DDR loader：只读外存并写 token stream，和 compute-owner 并发。
    weight_stream_load_stage(w0_arena, w1_arena,
                             w0_nt_base_words, w1_nt_base_words,
                             w0_nt_scale_words, w1_nt_scale_words,
                             k_tiles,
                             scale_pair_stream, weight_pair_stream);

    // 3. Compute-owner：独占本地双 bank，执行 BM4 权重复用和 next-bank 预取。
    compute_stage_pingpong_owner_bm4(act, act_word_base, k_tiles, slabs_valid,
                                     scale_pair_stream, weight_pair_stream,
                                     scale_stream, a_stream, w0_stream, w1_stream, ctrl_stream);
#endif
}

/**
 * -------------------------------------------------------------------------------------
 * 函数：run_ntile_accumulate_batch
 * 作用：执行一组 Batch 的、基于 N-Tiling 小块的片端矩阵乘加（Accumulate）。
 *       在基于 CSIM 仿真时使用串行运行调试(pump/absorb)；
 *       在板上综合阶段 (#ifndef V6_CSIM) 则通过 HLS DATAFLOW 下发至 FPGA 内部的物理并发执行引擎：
 *       推流 (drive_all_ktiles_batch_pp) 和吸收/重塑 (absorb_all_ktiles_batch) 会完全并行工作不停机。
 *
 * 阶段说明：
 *   1. scale_stream 在 drive 侧写入 weight scale，在 absorb 侧按 PE psum 顺序读取。
 *   2. V6_CSIM：drive -> behavioral PE pump -> absorb，保持软件仿真确定性。
 *   3. Synth：drive_all_ktiles_batch_pp 与 absorb_all_ktiles_batch 在 DATAFLOW 中并行。
 *      full BM4 和 decode/tail 共用同一套 ping-pong driver，避免额外综合 tail wbuf。
 *   4. slabs_valid 控制真实 slab 数；小 M 只驱动有效 slab，且仍能部分隐藏下一 K tile load。
 *   5. absorb_all_ktiles_batch 在第一次 K tile 融合 bias init，后续 K tile 累加 scaled psum。
 * -------------------------------------------------------------------------------------
 */
static void run_ntile_accumulate_batch(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        const ActCache &act,
        const uint32_t act_word_base[kBatchMtiles],
        uint32_t slabs_valid,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        const int8_t act_row_exp_kt[kBatchMtiles][kMaxKtiles][kSaM][kGroupsPerKtile],
        a_stream_t &a_stream,
        w_stream_t &w0_stream,
        w_stream_t &w1_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream,
        OutAccBatch &out_batch,
        const out_acc_t bias_tile[kSaN]) {
    #pragma HLS INLINE off
    scale_stream_t scale_stream("batch_scale_stream");
    #pragma HLS STREAM variable=scale_stream depth=32
    #pragma HLS BIND_STORAGE variable=scale_stream type=fifo impl=srl
#ifdef V6_CSIM
    // 1. C-sim 下先生成所有 A/W/CTRL/scale token。
    drive_all_ktiles_batch_pp(w0_arena, w1_arena, act, act_word_base, slabs_valid,
                              w0_nt_base_words, w1_nt_base_words,
                              w0_nt_scale_words, w1_nt_scale_words, k_tiles,
                              scale_stream, a_stream, w0_stream, w1_stream, ctrl_stream);

    // 2. behavioral PE 消费 stream，产生 psum_stream，模拟 RTL PE array 的 commit/drain 行为。
    v6_csim::pump_ktiles_batch(k_tiles, slabs_valid,
                               a_stream, w0_stream, w1_stream, ctrl_stream, psum_stream);

    // 3. scale_drain 使用 act/weight exp 把 psum 还原到 Q22，并写入 work_batch。
    absorb_all_ktiles_batch(psum_stream, scale_stream, k_tiles, slabs_valid,
                            act_row_exp_kt, out_batch, bias_tile);
#else
    #pragma HLS DATAFLOW disable_start_propagation
    // 1. full BM4 和小 M/decode 共用一套 true ping-pong driver。
    //    slabs_valid=1 时只能隐藏当前 128 compute cycles 对应的下一 K tile load，
    //    但避免了旧 tail_seq 的完全串行 load->compute，同时省掉一套 tail wbuf BRAM。
    drive_all_ktiles_batch_pp(w0_arena, w1_arena, act, act_word_base, slabs_valid,
                              w0_nt_base_words, w1_nt_base_words,
                              w0_nt_scale_words, w1_nt_scale_words, k_tiles,
                              scale_stream, a_stream, w0_stream, w1_stream, ctrl_stream);

    // 2. psum absorb 在同一个 dataflow region 中持续消费 PE 输出。
    absorb_all_ktiles_batch(psum_stream, scale_stream, k_tiles, slabs_valid,
                            act_row_exp_kt, out_batch, bias_tile);
#endif
}

// QKV/O top-level: BM4 slabs outer, then output_count and n_tiles.
/**
 * 执行 Prefill 阶段的 Dense/Linear 层（如注意力机制的 QKV、O 投影，或普通的线性层）。
 * 
 * 核心架构与调度策略：
 * 1. Token 驻留复用 (M-Tiling Batching)：最外层先按批次加载 M(Token) 个激活用以复用极其昂贵的权重带宽。
 * 2. 多输出头处理 (Multi-output Fusion)：通过 output_count (如 Q,K,V 算子融合)，一次输入激活，连着投影出多个头的结果。
 * 3. 底层 Dataflow 驱动 (MAC Engine)：借助 run_ntile_accumulate_batch，自动使用 ping-pong buffer 和并行预取 (Prefetch) 完成 K 维度的计算。
 */
static void run_prefill_dense(
        const axi_word_t *w0,
        const axi_word_t *w1,
        const axi_word_t *act_arena,
        axi_word_t *out_arena,
        const task_t &task,
        ActCache &act_cache,
        OutAccBatch &work_batch,
        out_acc_t bias_cache[kBiasCacheCols],
        a_stream_t   &a_stream,
        w_stream_t   &w0_stream,
        w_stream_t   &w1_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream) {
    #pragma HLS INLINE

    const uint32_t M        = task.rows;
    const uint32_t K        = task.input_cols;
    const uint32_t k_tiles  = div_ceil(K, kTileK);
    const uint32_t k_groups = div_ceil(K, kGroupSize);
    const uint32_t m_tiles  = div_ceil(M, kSaM);
    const uint32_t k_words_valid = div_ceil(K, kAxiBytes);
    const uint32_t k_words_act = k_tiles * kActWordsPerKtile;

    static int8_t act_row_exp_kt[kBatchMtiles][kMaxKtiles][kSaM][kGroupsPerKtile];

    // =========================================================================================
    // 1. 偏置预存阶段 (Bias Pre-loading)：
    // 如果任意一个输出头（Q/K/V/O）带有偏置（通常 Vision 模块或 QKV 会配有偏置），
    // 预先扫取从 DDR 读取并按照任务头的顺序拼接存储到片上 BRAM (bias_cache) 备用。
    // =========================================================================================
    bool has_bias = false;
    for (uint32_t out_idx = 0; out_idx < VLM_W8A8_LINEAR_MAX_OUTPUTS; ++out_idx) {
        #pragma HLS UNROLL
        has_bias = has_bias || (task.bias_offset_bytes[out_idx] != 0);
    }
    if (has_bias) {
        uint32_t bias_base = 0;
        for (uint32_t out_idx = 0; out_idx < task.output_count; ++out_idx) {
            const uint32_t cols = task.out_cols[out_idx];
            load_bias_vector_q22_i40(act_arena,
                                     task.bias_offset_bytes[out_idx],
                                     cols,
                                     bias_base,
                                     bias_cache);
            bias_base += cols;
        }
    }

    // =========================================================================================
    // 2. M维度分批循环 (Token Batching Level)：
    // 按 kBatchMtiles 组将计算 M(Token流) 分块，使读入的输入激活能同时用于后面的计算而不用重复加载，最大化复用。
    // =========================================================================================
    for (uint32_t mt0 = 0; mt0 < m_tiles; mt0 += kBatchMtiles) {
        const uint32_t slabs_valid =
            (mt0 + kBatchMtiles <= m_tiles) ? kBatchMtiles : (m_tiles - mt0);
        uint32_t rows_valid[kBatchMtiles];
        uint32_t act_word_base[kBatchMtiles];
        
        // 2.1 将整批的输入特征（K通道层）从外存预取至 BRAM: act_cache 中。
        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            const uint32_t mt_row0 = (mt0 + slab) * kSaM;
            rows_valid[slab] = (mt_row0 + kSaM <= M) ? kSaM : (M - mt_row0);
            act_word_base[slab] = input_slab_word_base(slab);
            load_activation_block(act_arena,
                                  task.act_q_offset_bytes + (uint64_t)mt_row0 * task.act_row_stride,
                                  task.act_scale_offset_bytes + (uint64_t)mt_row0 * task.act_scale_row_stride,
                                  task.act_row_stride, task.act_scale_row_stride,
                                  rows_valid[slab], k_words_valid, k_words_act,
                                  k_groups, slab, act_cache);
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                for (uint32_t r = 0; r < kSaM; ++r) {
                    for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                        #pragma HLS UNROLL
                        const uint32_t glob_g = kt * kGroupsPerKtile + g;
                        act_row_exp_kt[slab][kt][r][g] =
                            (glob_g < k_groups) ? (int8_t)act_cache.row_exp[slab][r][glob_g] : (int8_t)0;
                    }
                }
            }
        }

        // =========================================================================================
        // 3. 多输出头融合映射 (Multi-head Output Iteration)：
        // 针对当前已驻留片上的批次数据，依次对其应用各个线性头的权重。
        // 例如当任务是一次算完 [Q, K, V] 三个投影大矩阵时，out_idx 将从 0 遍历到 2。
        // =========================================================================================
        for (uint32_t out_idx = 0; out_idx < task.output_count; ++out_idx) {
            const uint32_t N = task.out_cols[out_idx];
            const uint32_t n_tiles = div_ceil(N, kSaN);

            // -------------------------------------------------------------------------------------
            // 3.1 N 维度输出平铺遍历：将当前头的输出列再次进行芯片资源适配的硬平铺切分
            // -------------------------------------------------------------------------------------
            for (uint32_t nt = 0; nt < n_tiles; ++nt) {
                const uint64_t w0_nt_base = (task.weight_q_offset_bytes[out_idx][0]
                                             + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                const uint64_t w1_nt_base = (task.weight_q_offset_bytes[out_idx][1]
                                             + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                const uint64_t w0_nt_scl  = (task.weight_scale_offset_bytes[out_idx][0]
                                             + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                const uint64_t w1_nt_scl  = (task.weight_scale_offset_bytes[out_idx][1]
                                             + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;

                const uint64_t out_byte_off = task.dst_offset_bytes[out_idx]
                    + (uint64_t)mt0 * kSaM * task.output_row_stride_bytes[out_idx];
                const uint32_t out_col_byte_off = nt * kSaN * kOutBytesI32;
                const uint32_t cols_valid =
                    (nt + 1u) * kSaN <= N ? kSaN : (N - nt * kSaN);

                out_acc_t bias_tile[kSaN];
                #pragma HLS ARRAY_PARTITION variable=bias_tile cyclic factor=kScaleDrainColLanes dim=1
                if (has_bias && task.bias_offset_bytes[out_idx] != 0) {
                    uint32_t bias_cache_base = 0;
                    for (uint32_t bi = 0; bi < out_idx; ++bi) {
                        bias_cache_base += task.out_cols[bi];
                    }
                    load_bias_tile_from_cache(bias_cache,
                                              bias_cache_base + nt * kSaN,
                                              bias_tile);
                } else {
                    for (uint32_t c = 0; c < kSaN; ++c) {
                        #pragma HLS UNROLL
                        bias_tile[c] = 0;
                    }
                }

                // ---------------------------------------------------------------------------------
                // 3.2 调用底层脉动阵列引流器：
                // 使用驱动管线发送控制信号 (ctrl_stream/psum_stream/scale_stream)。
                // 将对应 M 和 nt 的矩阵乘积丢入芯片引擎内计算并自动收割尺度信息还原(Absorb Psum)。
                // ---------------------------------------------------------------------------------
                run_ntile_accumulate_batch(w0, w1, act_cache, act_word_base, slabs_valid,
                                     w0_nt_base, w1_nt_base, w0_nt_scl, w1_nt_scl,
                                     k_tiles,
                                     act_row_exp_kt,
                                     a_stream, w0_stream, w1_stream, ctrl_stream, psum_stream,
                                     work_batch,
                                     bias_tile);

                // ---------------------------------------------------------------------------------
                // 3.3 输出落盘：不同于 FFN 有极其复杂的后激活或子处理，
                // QKV 以及纯 Linear / Dense 层只做一次矩阵乘加后，直接将INT32计算结果导出写回 DDR。
                // ---------------------------------------------------------------------------------
                store_out_acc_batch_i32(out_arena, out_byte_off,
                                        task.output_row_stride_bytes[out_idx],
                                        slabs_valid, rows_valid, cols_valid,
                                        out_col_byte_off, work_batch);
            }
        }
    }
}

} // namespace v6
