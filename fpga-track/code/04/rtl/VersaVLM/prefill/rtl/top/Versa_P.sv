`default_nettype none

import npu_spm_pkg::*;

// Versa_P external AXI top-level shell.
//
// The user-configurable top-level parameters use literal defaults rather than
// package-derived defaults so Vivado IP packaging can expose them directly.
//
// M1 scope:
//   This module exposes the final AXI-Lite/AXI-master/IRQ boundary and
//   instantiates inst_ctrl. With M1_CONTROL_ONLY enabled, commands are accepted
//   by a small control-only backend and complete after a fixed latency. The
//   backend exists only so API/status/IRQ behavior can be verified through the
//   final top-level port shape before DMA/core integration lands.
//
// Normal working scope:
//   With M1_CONTROL_ONLY disabled, commands flow through api_dma_mapper into
//   the direct DMA backend. A/W/meta AXI reads feed Versa_P_core streams
//   directly, and O/MVOUT is written by the single m_axi2 AXI writer.
module Versa_P #(
    parameter int AXIL_ADDR_W = 32,
    parameter int AXIL_DATA_W = 64,
    parameter int AXI_ID_W    = 4,
    parameter int AXI_ADDR_W  = 32,
    parameter int AXI_DATA_W  = 128,
    parameter int BANK_DEPTH_WORDS = 16384,
    parameter int A_BANK_DEPTH_WORDS = 12288,
    parameter int W_BANK_DEPTH_WORDS = 12288,
    parameter int O_BANK_DEPTH_WORDS = 16384,
    parameter int META_WORD128_DEPTH = 2048,
    parameter int FIFO_DEPTH = 16,
    parameter int DMA_MAX_BURST_BEATS = 16,
    parameter bit M1_CONTROL_ONLY = 1'b0,
    parameter bit DISABLE_ERROR_CHECKS = 1'b1,
    parameter bit ENABLE_PROFILE = 1'b0,
    parameter bit O_SPLIT_BANK_HALVES = 1'b0,
    parameter int M1_DONE_LATENCY = 8,
    localparam int AXIL_STRB_W = AXIL_DATA_W / 8,
    localparam int AXI_STRB_W  = AXI_DATA_W / 8,
    localparam int DMA_NUM = 3,
    localparam int RF_DATA_W = 64,
    localparam int DMA_HALF_W = RF_DATA_W / 2,
    localparam int DMA_QUARTER_W = RF_DATA_W / 4,
    localparam int M1_DONE_COUNTER_W = (M1_DONE_LATENCY < 2) ? 1 : $clog2(M1_DONE_LATENCY + 1)
) (
    input  wire logic                           clk_i,
    input  wire logic                           rst_n_i,

    input  wire logic [AXIL_ADDR_W-1:0]         s_axil_awaddr_i,
    input  wire logic [2:0]                     s_axil_awprot_i,
    input  wire logic                           s_axil_awvalid_i,
    output logic                                s_axil_awready_o,
    input  wire logic [AXIL_DATA_W-1:0]         s_axil_wdata_i,
    input  wire logic [AXIL_STRB_W-1:0]         s_axil_wstrb_i,
    input  wire logic                           s_axil_wvalid_i,
    output logic                                s_axil_wready_o,
    output logic [1:0]                          s_axil_bresp_o,
    output logic                                s_axil_bvalid_o,
    input  wire logic                           s_axil_bready_i,
    input  wire logic [AXIL_ADDR_W-1:0]         s_axil_araddr_i,
    input  wire logic [2:0]                     s_axil_arprot_i,
    input  wire logic                           s_axil_arvalid_i,
    output logic                                s_axil_arready_o,
    output logic [AXIL_DATA_W-1:0]              s_axil_rdata_o,
    output logic [1:0]                          s_axil_rresp_o,
    output logic                                s_axil_rvalid_o,
    input  wire logic                           s_axil_rready_i,

    output logic [AXI_ID_W-1:0]                 m00_axi_awid_o,
    output logic [AXI_ADDR_W-1:0]               m00_axi_awaddr_o,
    output logic [7:0]                          m00_axi_awlen_o,
    output logic [2:0]                          m00_axi_awsize_o,
    output logic [1:0]                          m00_axi_awburst_o,
    output logic                                m00_axi_awvalid_o,
    input  wire logic                           m00_axi_awready_i,
    output logic [AXI_DATA_W-1:0]               m00_axi_wdata_o,
    output logic [AXI_STRB_W-1:0]               m00_axi_wstrb_o,
    output logic                                m00_axi_wlast_o,
    output logic                                m00_axi_wvalid_o,
    input  wire logic                           m00_axi_wready_i,
    input  wire logic [AXI_ID_W-1:0]            m00_axi_bid_i,
    input  wire logic [1:0]                     m00_axi_bresp_i,
    input  wire logic                           m00_axi_bvalid_i,
    output logic                                m00_axi_bready_o,
    output logic [AXI_ID_W-1:0]                 m00_axi_arid_o,
    output logic [AXI_ADDR_W-1:0]               m00_axi_araddr_o,
    output logic [7:0]                          m00_axi_arlen_o,
    output logic [2:0]                          m00_axi_arsize_o,
    output logic [1:0]                          m00_axi_arburst_o,
    output logic                                m00_axi_arvalid_o,
    input  wire logic                           m00_axi_arready_i,
    input  wire logic [AXI_ID_W-1:0]            m00_axi_rid_i,
    input  wire logic [AXI_DATA_W-1:0]          m00_axi_rdata_i,
    input  wire logic [1:0]                     m00_axi_rresp_i,
    input  wire logic                           m00_axi_rlast_i,
    input  wire logic                           m00_axi_rvalid_i,
    output logic                                m00_axi_rready_o,

    output logic [AXI_ID_W-1:0]                 m01_axi_awid_o,
    output logic [AXI_ADDR_W-1:0]               m01_axi_awaddr_o,
    output logic [7:0]                          m01_axi_awlen_o,
    output logic [2:0]                          m01_axi_awsize_o,
    output logic [1:0]                          m01_axi_awburst_o,
    output logic                                m01_axi_awvalid_o,
    input  wire logic                           m01_axi_awready_i,
    output logic [AXI_DATA_W-1:0]               m01_axi_wdata_o,
    output logic [AXI_STRB_W-1:0]               m01_axi_wstrb_o,
    output logic                                m01_axi_wlast_o,
    output logic                                m01_axi_wvalid_o,
    input  wire logic                           m01_axi_wready_i,
    input  wire logic [AXI_ID_W-1:0]            m01_axi_bid_i,
    input  wire logic [1:0]                     m01_axi_bresp_i,
    input  wire logic                           m01_axi_bvalid_i,
    output logic                                m01_axi_bready_o,
    output logic [AXI_ID_W-1:0]                 m01_axi_arid_o,
    output logic [AXI_ADDR_W-1:0]               m01_axi_araddr_o,
    output logic [7:0]                          m01_axi_arlen_o,
    output logic [2:0]                          m01_axi_arsize_o,
    output logic [1:0]                          m01_axi_arburst_o,
    output logic                                m01_axi_arvalid_o,
    input  wire logic                           m01_axi_arready_i,
    input  wire logic [AXI_ID_W-1:0]            m01_axi_rid_i,
    input  wire logic [AXI_DATA_W-1:0]          m01_axi_rdata_i,
    input  wire logic [1:0]                     m01_axi_rresp_i,
    input  wire logic                           m01_axi_rlast_i,
    input  wire logic                           m01_axi_rvalid_i,
    output logic                                m01_axi_rready_o,

    output logic [AXI_ID_W-1:0]                 m02_axi_awid_o,
    output logic [AXI_ADDR_W-1:0]               m02_axi_awaddr_o,
    output logic [7:0]                          m02_axi_awlen_o,
    output logic [2:0]                          m02_axi_awsize_o,
    output logic [1:0]                          m02_axi_awburst_o,
    output logic                                m02_axi_awvalid_o,
    input  wire logic                           m02_axi_awready_i,
    output logic [AXI_DATA_W-1:0]               m02_axi_wdata_o,
    output logic [AXI_STRB_W-1:0]               m02_axi_wstrb_o,
    output logic                                m02_axi_wlast_o,
    output logic                                m02_axi_wvalid_o,
    input  wire logic                           m02_axi_wready_i,
    input  wire logic [AXI_ID_W-1:0]            m02_axi_bid_i,
    input  wire logic [1:0]                     m02_axi_bresp_i,
    input  wire logic                           m02_axi_bvalid_i,
    output logic                                m02_axi_bready_o,
    output logic [AXI_ID_W-1:0]                 m02_axi_arid_o,
    output logic [AXI_ADDR_W-1:0]               m02_axi_araddr_o,
    output logic [7:0]                          m02_axi_arlen_o,
    output logic [2:0]                          m02_axi_arsize_o,
    output logic [1:0]                          m02_axi_arburst_o,
    output logic                                m02_axi_arvalid_o,
    input  wire logic                           m02_axi_arready_i,
    input  wire logic [AXI_ID_W-1:0]            m02_axi_rid_i,
    input  wire logic [AXI_DATA_W-1:0]          m02_axi_rdata_i,
    input  wire logic [1:0]                     m02_axi_rresp_i,
    input  wire logic                           m02_axi_rlast_i,
    input  wire logic                           m02_axi_rvalid_i,
    output logic                                m02_axi_rready_o,

    output logic                                irq_o,
    output logic                                busy_o
);
    localparam logic [2:0] CMD_MVIN_A    = 3'd0;
    localparam logic [2:0] CMD_MVIN_W    = 3'd1;
    localparam logic [2:0] CMD_META_MVIN = 3'd2;
    localparam logic [2:0] CMD_GEMM      = 3'd3;
    localparam logic [2:0] CMD_MVOUT     = 3'd4;
    localparam logic [2:0] CMD_ATTENTION_QK = 3'd5;

    logic rst;
    logic cmd_valid;
    logic cmd_ready;
    logic [2:0] cmd_opcode;
    logic cmd_a_bank;
    logic cmd_w_bank;
    logic cmd_o_bank;
    logic active_w_bank;
    logic backend_clear;
    logic profile_enable;
    logic profile_clear;

    logic [31:0] mvin_a_dram_base;
    logic [31:0] mvin_a_dram_row_stride_bytes;
    logic [15:0] mvin_a_m;
    logic [15:0] mvin_a_k;
    logic mvin_a_bank;
    logic mvin_a_u8_minus_128_en;
    logic [31:0] mvin_w_dram_base;
    logic [15:0] mvin_w_k;
    logic [15:0] mvin_w_n;
    logic mvin_w_bank;
    logic [31:0] mvin_meta_dram_base;
    logic [31:0] mvin_meta_offset_bytes;
    logic [31:0] mvin_meta_byte_count;
    logic [1:0] mvin_meta_type;
    logic [15:0] gemm_m;
    logic [15:0] gemm_n;
    logic [15:0] gemm_k;
    logic gemm_a_bank;
    logic gemm_w_bank;
    logic gemm_accumulate_en;
    logic gemm_add_bias_en;
    npu_spm_pkg::npu_mode_e gemm_mode;
    logic [31:0] gemm_bias_offset_bytes;
    logic [31:0] mvout_dram_base;
    logic [15:0] mvout_m;
    logic [15:0] mvout_n;
    logic [15:0] mvout_output_stride_n;
    npu_spm_pkg::npu_mvout_mode_e mvout_mode;
    logic mvout_per_channel_scale;
    logic [31:0] mvout_scale_param;
    logic [31:0] attention_qk_output_dram_base;
    logic [15:0] attention_qk_token_count;
    logic [31:0] attention_qk_gamma16_fix;
    logic attention_qk_mask_en;
    logic [4:0] attention_qk_q_block_start;
    logic [4:0] attention_qk_q_block_count_m1;

    logic mvin_a_done;
    logic mvin_w_done;
    logic mvin_meta_done;
    logic gemm_done;
    logic mvout_done;
    logic attention_qk_done;
    logic [5:0] api_error_valid;
    logic [5:0][3:0] api_error_code;

    logic backend_busy_q;
    logic [2:0] backend_opcode_q;
    logic [M1_DONE_COUNTER_W-1:0] backend_countdown_q;

    logic [DMA_NUM-1:0][AXI_ID_W-1:0] m_axi_awid;
    logic [DMA_NUM-1:0][AXI_ADDR_W-1:0] m_axi_awaddr;
    logic [DMA_NUM-1:0][7:0] m_axi_awlen;
    logic [DMA_NUM-1:0][2:0] m_axi_awsize;
    logic [DMA_NUM-1:0][1:0] m_axi_awburst;
    logic [DMA_NUM-1:0] m_axi_awvalid;
    logic [DMA_NUM-1:0] m_axi_awready;
    logic [DMA_NUM-1:0][AXI_DATA_W-1:0] m_axi_wdata;
    logic [DMA_NUM-1:0][AXI_STRB_W-1:0] m_axi_wstrb;
    logic [DMA_NUM-1:0] m_axi_wlast;
    logic [DMA_NUM-1:0] m_axi_wvalid;
    logic [DMA_NUM-1:0] m_axi_wready;
    logic [DMA_NUM-1:0][AXI_ID_W-1:0] m_axi_bid;
    logic [DMA_NUM-1:0][1:0] m_axi_bresp;
    logic [DMA_NUM-1:0] m_axi_bvalid;
    logic [DMA_NUM-1:0] m_axi_bready;
    logic [DMA_NUM-1:0][AXI_ID_W-1:0] m_axi_arid;
    logic [DMA_NUM-1:0][AXI_ADDR_W-1:0] m_axi_araddr;
    logic [DMA_NUM-1:0][7:0] m_axi_arlen;
    logic [DMA_NUM-1:0][2:0] m_axi_arsize;
    logic [DMA_NUM-1:0][1:0] m_axi_arburst;
    logic [DMA_NUM-1:0] m_axi_arvalid;
    logic [DMA_NUM-1:0] m_axi_arready;
    logic [DMA_NUM-1:0][AXI_ID_W-1:0] m_axi_rid;
    logic [DMA_NUM-1:0][AXI_DATA_W-1:0] m_axi_rdata;
    logic [DMA_NUM-1:0][1:0] m_axi_rresp;
    logic [DMA_NUM-1:0] m_axi_rlast;
    logic [DMA_NUM-1:0] m_axi_rvalid;
    logic [DMA_NUM-1:0] m_axi_rready;

    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvin_dram_addr;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvin_sram_addr;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvin_col_num;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvin_row_num;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvout_dram_addr;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvout_sram_addr;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvout_col_num;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] mvout_row_num;
    logic [DMA_NUM-1:0][1:0] cfg_mvin_input_type;
    logic [DMA_NUM-1:0][1:0] cfg_mvout_output_type;
    logic [DMA_NUM-1:0][1:0] cfg_mvin_input_precision;
    logic [DMA_NUM-1:0][1:0] cfg_mvout_output_precision;
    logic [DMA_NUM-1:0] cfg_mvin_is_quant;
    logic [DMA_NUM-1:0] cfg_mvout_is_quant;
    logic [DMA_NUM-1:0] cfg_mvin_dest;
    logic [DMA_NUM-1:0] cfg_mvout_source;
    logic [DMA_NUM-1:0][DMA_QUARTER_W-1:0] cfg_mvin_sram_stride;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] cfg_mvin_dram_stride;
    logic [DMA_NUM-1:0][DMA_QUARTER_W-1:0] cfg_mvout_sram_stride;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] cfg_mvout_dram_stride;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] cfg_mvin_input_zeropoint;
    logic [DMA_NUM-1:0][DMA_HALF_W-1:0] cfg_mvout_output_zeropoint;
    logic [DMA_NUM-1:0][DMA_QUARTER_W-1:0] cfg_mvin_input_scale;
    logic [DMA_NUM-1:0][DMA_QUARTER_W-1:0] cfg_mvout_output_scale;
    logic [DMA_NUM-1:0][DMA_QUARTER_W-1:0] cfg_mvin_input_scale_shift;
    logic [DMA_NUM-1:0][DMA_QUARTER_W-1:0] cfg_mvout_output_scale_shift;
    logic [DMA_NUM-1:0][1:0] cfg_mvout_mode;
    logic [DMA_NUM-1:0] cfg_mvout_per_channel;
    logic [DMA_NUM-1:0][31:0] cfg_mvout_scale_param;
    logic [DMA_NUM-1:0] dma_mvin_req_en;
    logic [DMA_NUM-1:0] dma_mvout_req_en;
    logic attention_qk_req_en;
    logic [DMA_NUM-1:0] loader_mvin_done;
    logic [DMA_NUM-1:0] dma_mvout_resp_done;
    logic [DMA_NUM-1:0] dma_mvin_busy;
    logic [DMA_NUM-1:0] dma_mvout_busy;
    logic o_mvout_stream_valid;
    logic o_mvout_stream_ready;
    logic [AXI_DATA_W-1:0] o_mvout_stream_data;
    logic [AXI_STRB_W-1:0] o_mvout_stream_keep;
    logic [63:0] o_mvout_stream_addr;
    logic o_mvout_stream_last;
    logic o_writer_busy;
    logic o_writer_done;
    logic o_writer_error;
    logic [AXI_ID_W-1:0] o_writer0_awid;
    logic [AXI_ADDR_W-1:0] o_writer0_awaddr;
    logic [7:0] o_writer0_awlen;
    logic [2:0] o_writer0_awsize;
    logic [1:0] o_writer0_awburst;
    logic o_writer0_awvalid;
    logic [AXI_DATA_W-1:0] o_writer0_wdata;
    logic [AXI_STRB_W-1:0] o_writer0_wstrb;
    logic o_writer0_wlast;
    logic o_writer0_wvalid;
    logic o_writer0_bready;
    logic mapper_sa_req_en;
    logic bridge_active_w_bank;
    logic core_busy;
    logic core_sa_busy;
    logic core_sa_done;
    logic core_attention_qk_done;
    logic core_error_sticky;
    npu_error_e core_last_error;
    logic [31:0] core_command_accepted_count;
    logic [31:0] core_conflict_count;
    logic dma_backpressure_error;
    logic dma_command_drop;
    logic mvout_bridge_underflow;
    logic [$clog2(FIFO_DEPTH + 1)-1:0] mvout_bridge_level;
    logic [63:0] profile_global_cycles;
    logic [63:0] profile_mvin_a_busy_cycles;
    logic [63:0] profile_mvin_w_busy_cycles;
    logic [63:0] profile_mvin_meta_busy_cycles;
    logic [63:0] profile_gemm_busy_cycles;
    logic [63:0] profile_mvout_busy_cycles;
    logic [63:0] profile_busy_any_cycles;
    logic [63:0] profile_busy_multi_cycles;
    logic [3:0][63:0] profile_axi_r_beats;
    logic [3:0][63:0] profile_axi_w_beats;
    logic [3:0][63:0] profile_axi_ar_stall_cycles;
    logic [3:0][63:0] profile_axi_r_stall_cycles;
    logic [3:0][63:0] profile_axi_aw_stall_cycles;
    logic [3:0][63:0] profile_axi_w_stall_cycles;
    logic [4:0] profile_busy_mask;
    logic [2:0] profile_busy_count;

    wire cmd_fire = cmd_valid && cmd_ready;

    assign rst = !rst_n_i;
    generate
        if (ENABLE_PROFILE) begin : gen_profile_counters
            assign profile_busy_mask = {
                dma_mvout_busy[2],
                core_sa_busy,
                dma_mvin_busy[2],
                dma_mvin_busy[1],
                dma_mvin_busy[0]
            };
            assign profile_busy_count = {2'd0, profile_busy_mask[0]} +
                                        {2'd0, profile_busy_mask[1]} +
                                        {2'd0, profile_busy_mask[2]} +
                                        {2'd0, profile_busy_mask[3]} +
                                        {2'd0, profile_busy_mask[4]};

            always_ff @(posedge clk_i) begin
                if (rst || profile_clear) begin
                    profile_global_cycles <= '0;
                    profile_mvin_a_busy_cycles <= '0;
                    profile_mvin_w_busy_cycles <= '0;
                    profile_mvin_meta_busy_cycles <= '0;
                    profile_gemm_busy_cycles <= '0;
                    profile_mvout_busy_cycles <= '0;
                    profile_busy_any_cycles <= '0;
                    profile_busy_multi_cycles <= '0;
                    profile_axi_r_beats <= '0;
                    profile_axi_w_beats <= '0;
                    profile_axi_ar_stall_cycles <= '0;
                    profile_axi_r_stall_cycles <= '0;
                    profile_axi_aw_stall_cycles <= '0;
                    profile_axi_w_stall_cycles <= '0;
                end else if (profile_enable) begin
                    profile_global_cycles <= profile_global_cycles + 64'd1;
                    if (dma_mvin_busy[0]) begin
                        profile_mvin_a_busy_cycles <= profile_mvin_a_busy_cycles + 64'd1;
                    end
                    if (dma_mvin_busy[1]) begin
                        profile_mvin_w_busy_cycles <= profile_mvin_w_busy_cycles + 64'd1;
                    end
                    if (dma_mvin_busy[2]) begin
                        profile_mvin_meta_busy_cycles <= profile_mvin_meta_busy_cycles + 64'd1;
                    end
                    if (core_sa_busy) begin
                        profile_gemm_busy_cycles <= profile_gemm_busy_cycles + 64'd1;
                    end
                    if (dma_mvout_busy[2]) begin
                        profile_mvout_busy_cycles <= profile_mvout_busy_cycles + 64'd1;
                    end
                    if (|profile_busy_mask) begin
                        profile_busy_any_cycles <= profile_busy_any_cycles + 64'd1;
                    end
                    if (profile_busy_count > 3'd1) begin
                        profile_busy_multi_cycles <= profile_busy_multi_cycles + 64'd1;
                    end
                    for (int axi_idx = 0; axi_idx < DMA_NUM; axi_idx++) begin
                        if (m_axi_rvalid[axi_idx] && m_axi_rready[axi_idx]) begin
                            profile_axi_r_beats[axi_idx] <= profile_axi_r_beats[axi_idx] + 64'd1;
                        end
                        if ((axi_idx == 2) ? (o_writer0_wvalid && m02_axi_wready_i)
                                           : (m_axi_wvalid[axi_idx] && m_axi_wready[axi_idx])) begin
                            profile_axi_w_beats[axi_idx] <= profile_axi_w_beats[axi_idx] + 64'd1;
                        end
                        if (m_axi_arvalid[axi_idx] && !m_axi_arready[axi_idx]) begin
                            profile_axi_ar_stall_cycles[axi_idx] <= profile_axi_ar_stall_cycles[axi_idx] + 64'd1;
                        end
                        if (m_axi_rvalid[axi_idx] && !m_axi_rready[axi_idx]) begin
                            profile_axi_r_stall_cycles[axi_idx] <= profile_axi_r_stall_cycles[axi_idx] + 64'd1;
                        end
                        if ((axi_idx == 2) ? (o_writer0_awvalid && !m02_axi_awready_i)
                                           : (m_axi_awvalid[axi_idx] && !m_axi_awready[axi_idx])) begin
                            profile_axi_aw_stall_cycles[axi_idx] <= profile_axi_aw_stall_cycles[axi_idx] + 64'd1;
                        end
                        if ((axi_idx == 2) ? (o_writer0_wvalid && !m02_axi_wready_i)
                                           : (m_axi_wvalid[axi_idx] && !m_axi_wready[axi_idx])) begin
                            profile_axi_w_stall_cycles[axi_idx] <= profile_axi_w_stall_cycles[axi_idx] + 64'd1;
                        end
                    end
                end
            end
        end else begin : gen_profile_disabled
            assign profile_busy_mask = '0;
            assign profile_busy_count = '0;
            assign profile_global_cycles = '0;
            assign profile_mvin_a_busy_cycles = '0;
            assign profile_mvin_w_busy_cycles = '0;
            assign profile_mvin_meta_busy_cycles = '0;
            assign profile_gemm_busy_cycles = '0;
            assign profile_mvout_busy_cycles = '0;
            assign profile_busy_any_cycles = '0;
            assign profile_busy_multi_cycles = '0;
            assign profile_axi_r_beats = '0;
            assign profile_axi_w_beats = '0;
            assign profile_axi_ar_stall_cycles = '0;
            assign profile_axi_r_stall_cycles = '0;
            assign profile_axi_aw_stall_cycles = '0;
            assign profile_axi_w_stall_cycles = '0;
        end
    endgenerate

    assign m00_axi_awid_o    = m_axi_awid[0];
    assign m00_axi_awaddr_o  = m_axi_awaddr[0];
    assign m00_axi_awlen_o   = m_axi_awlen[0];
    assign m00_axi_awsize_o  = m_axi_awsize[0];
    assign m00_axi_awburst_o = m_axi_awburst[0];
    assign m00_axi_awvalid_o = m_axi_awvalid[0];
    assign m00_axi_wdata_o   = m_axi_wdata[0];
    assign m00_axi_wstrb_o   = m_axi_wstrb[0];
    assign m00_axi_wlast_o   = m_axi_wlast[0];
    assign m00_axi_wvalid_o  = m_axi_wvalid[0];
    assign m00_axi_bready_o  = m_axi_bready[0];
    assign m00_axi_arid_o    = m_axi_arid[0];
    assign m00_axi_araddr_o  = m_axi_araddr[0];
    assign m00_axi_arlen_o   = m_axi_arlen[0];
    assign m00_axi_arsize_o  = m_axi_arsize[0];
    assign m00_axi_arburst_o = m_axi_arburst[0];
    assign m00_axi_arvalid_o = m_axi_arvalid[0];
    assign m00_axi_rready_o  = m_axi_rready[0];
    assign m_axi_awready[0]  = m00_axi_awready_i;
    assign m_axi_wready[0]   = m00_axi_wready_i;
    assign m_axi_bid[0]      = m00_axi_bid_i;
    assign m_axi_bresp[0]    = m00_axi_bresp_i;
    assign m_axi_bvalid[0]   = m00_axi_bvalid_i;
    assign m_axi_arready[0]  = m00_axi_arready_i;
    assign m_axi_rid[0]      = m00_axi_rid_i;
    assign m_axi_rdata[0]    = m00_axi_rdata_i;
    assign m_axi_rresp[0]    = m00_axi_rresp_i;
    assign m_axi_rlast[0]    = m00_axi_rlast_i;
    assign m_axi_rvalid[0]   = m00_axi_rvalid_i;

    assign m01_axi_awid_o    = m_axi_awid[1];
    assign m01_axi_awaddr_o  = m_axi_awaddr[1];
    assign m01_axi_awlen_o   = m_axi_awlen[1];
    assign m01_axi_awsize_o  = m_axi_awsize[1];
    assign m01_axi_awburst_o = m_axi_awburst[1];
    assign m01_axi_awvalid_o = m_axi_awvalid[1];
    assign m01_axi_wdata_o   = m_axi_wdata[1];
    assign m01_axi_wstrb_o   = m_axi_wstrb[1];
    assign m01_axi_wlast_o   = m_axi_wlast[1];
    assign m01_axi_wvalid_o  = m_axi_wvalid[1];
    assign m01_axi_bready_o  = m_axi_bready[1];
    assign m01_axi_arid_o    = m_axi_arid[1];
    assign m01_axi_araddr_o  = m_axi_araddr[1];
    assign m01_axi_arlen_o   = m_axi_arlen[1];
    assign m01_axi_arsize_o  = m_axi_arsize[1];
    assign m01_axi_arburst_o = m_axi_arburst[1];
    assign m01_axi_arvalid_o = m_axi_arvalid[1];
    assign m01_axi_rready_o  = m_axi_rready[1];
    assign m_axi_awready[1]  = m01_axi_awready_i;
    assign m_axi_wready[1]   = m01_axi_wready_i;
    assign m_axi_bid[1]      = m01_axi_bid_i;
    assign m_axi_bresp[1]    = m01_axi_bresp_i;
    assign m_axi_bvalid[1]   = m01_axi_bvalid_i;
    assign m_axi_arready[1]  = m01_axi_arready_i;
    assign m_axi_rid[1]      = m01_axi_rid_i;
    assign m_axi_rdata[1]    = m01_axi_rdata_i;
    assign m_axi_rresp[1]    = m01_axi_rresp_i;
    assign m_axi_rlast[1]    = m01_axi_rlast_i;
    assign m_axi_rvalid[1]   = m01_axi_rvalid_i;

    assign m02_axi_awid_o    = o_writer0_awid;
    assign m02_axi_awaddr_o  = o_writer0_awaddr;
    assign m02_axi_awlen_o   = o_writer0_awlen;
    assign m02_axi_awsize_o  = o_writer0_awsize;
    assign m02_axi_awburst_o = o_writer0_awburst;
    assign m02_axi_awvalid_o = o_writer0_awvalid;
    assign m02_axi_wdata_o   = o_writer0_wdata;
    assign m02_axi_wstrb_o   = o_writer0_wstrb;
    assign m02_axi_wlast_o   = o_writer0_wlast;
    assign m02_axi_wvalid_o  = o_writer0_wvalid;
    assign m02_axi_bready_o  = o_writer0_bready;
    assign m02_axi_arid_o    = m_axi_arid[2];
    assign m02_axi_araddr_o  = m_axi_araddr[2];
    assign m02_axi_arlen_o   = m_axi_arlen[2];
    assign m02_axi_arsize_o  = m_axi_arsize[2];
    assign m02_axi_arburst_o = m_axi_arburst[2];
    assign m02_axi_arvalid_o = m_axi_arvalid[2];
    assign m02_axi_rready_o  = m_axi_rready[2];
    assign m_axi_awready[2]  = 1'b0;
    assign m_axi_wready[2]   = 1'b0;
    assign m_axi_bid[2]      = m02_axi_bid_i;
    assign m_axi_bresp[2]    = m02_axi_bresp_i;
    assign m_axi_bvalid[2]   = 1'b0;
    assign m_axi_arready[2]  = m02_axi_arready_i;
    assign m_axi_rid[2]      = m02_axi_rid_i;
    assign m_axi_rdata[2]    = m02_axi_rdata_i;
    assign m_axi_rresp[2]    = m02_axi_rresp_i;
    assign m_axi_rlast[2]    = m02_axi_rlast_i;
    assign m_axi_rvalid[2]   = m02_axi_rvalid_i;

    axi_s2mm_writer #(
        .AXI_ID_WIDTH          (AXI_ID_W),
        .AXI_ADDR_WIDTH        (AXI_ADDR_W),
        .AXI_DATA_WIDTH        (AXI_DATA_W),
        .AXI_MAX_BURST_BEATS   (DMA_MAX_BURST_BEATS),
        .AXI_WR_OUTSTANDING    (16)
    ) u_o_s2mm_writer (
        .clk_i              (clk_i),
        .rst_i              (rst),
        .clear_i            (backend_clear),
        .start_i            (o_mvout_stream_valid && !o_writer_busy),
        .busy_o             (o_writer_busy),
        .done_o             (o_writer_done),
        .error_sticky_o     (o_writer_error),
        .s_valid_i          (o_mvout_stream_valid),
        .s_ready_o          (o_mvout_stream_ready),
        .s_data_i           (o_mvout_stream_data),
        .s_keep_i           (o_mvout_stream_keep),
        .s_addr_i           (o_mvout_stream_addr[AXI_ADDR_W-1:0]),
        .s_last_i           (o_mvout_stream_last),
        .m_axi_awid_o       (o_writer0_awid),
        .m_axi_awaddr_o     (o_writer0_awaddr),
        .m_axi_awlen_o      (o_writer0_awlen),
        .m_axi_awsize_o     (o_writer0_awsize),
        .m_axi_awburst_o    (o_writer0_awburst),
        .m_axi_awvalid_o    (o_writer0_awvalid),
        .m_axi_awready_i    (m02_axi_awready_i),
        .m_axi_wdata_o      (o_writer0_wdata),
        .m_axi_wstrb_o      (o_writer0_wstrb),
        .m_axi_wlast_o      (o_writer0_wlast),
        .m_axi_wvalid_o     (o_writer0_wvalid),
        .m_axi_wready_i     (m02_axi_wready_i),
        .m_axi_bresp_i      (m02_axi_bresp_i),
        .m_axi_bvalid_i     (m02_axi_bvalid_i),
        .m_axi_bready_o     (o_writer0_bready)
    );

    inst_ctrl #(
        .AXIL_ADDR_W (AXIL_ADDR_W),
        .AXIL_DATA_W (AXIL_DATA_W),
        .BANK_DEPTH_WORDS (BANK_DEPTH_WORDS),
        .META_WORD128_DEPTH (META_WORD128_DEPTH),
        .DISABLE_ERROR_CHECKS (DISABLE_ERROR_CHECKS),
        .ENABLE_PROFILE (ENABLE_PROFILE)
    ) u_inst_ctrl (
        .clk_i                            (clk_i),
        .rst_i                            (rst),
        .s_axi_awaddr_i                   (s_axil_awaddr_i),
        .s_axi_awprot_i                   (s_axil_awprot_i),
        .s_axi_awvalid_i                  (s_axil_awvalid_i),
        .s_axi_awready_o                  (s_axil_awready_o),
        .s_axi_wdata_i                    (s_axil_wdata_i),
        .s_axi_wstrb_i                    (s_axil_wstrb_i),
        .s_axi_wvalid_i                   (s_axil_wvalid_i),
        .s_axi_wready_o                   (s_axil_wready_o),
        .s_axi_bresp_o                    (s_axil_bresp_o),
        .s_axi_bvalid_o                   (s_axil_bvalid_o),
        .s_axi_bready_i                   (s_axil_bready_i),
        .s_axi_araddr_i                   (s_axil_araddr_i),
        .s_axi_arprot_i                   (s_axil_arprot_i),
        .s_axi_arvalid_i                  (s_axil_arvalid_i),
        .s_axi_arready_o                  (s_axil_arready_o),
        .s_axi_rdata_o                    (s_axil_rdata_o),
        .s_axi_rresp_o                    (s_axil_rresp_o),
        .s_axi_rvalid_o                   (s_axil_rvalid_o),
        .s_axi_rready_i                   (s_axil_rready_i),
        .irq_o                            (irq_o),
        .cmd_valid_o                      (cmd_valid),
        .cmd_ready_i                      (cmd_ready),
        .cmd_opcode_o                     (cmd_opcode),
        .cmd_a_bank_o                     (cmd_a_bank),
        .cmd_w_bank_o                     (cmd_w_bank),
        .cmd_o_bank_o                     (cmd_o_bank),
        .active_w_bank_o                  (active_w_bank),
        .backend_clear_o                  (backend_clear),
        .profile_enable_o                 (profile_enable),
        .profile_clear_o                  (profile_clear),
        .mvin_a_dram_base_o               (mvin_a_dram_base),
        .mvin_a_dram_row_stride_bytes_o   (mvin_a_dram_row_stride_bytes),
        .mvin_a_m_o                       (mvin_a_m),
        .mvin_a_k_o                       (mvin_a_k),
        .mvin_a_bank_o                    (mvin_a_bank),
        .mvin_a_u8_minus_128_en_o         (mvin_a_u8_minus_128_en),
        .mvin_w_dram_base_o               (mvin_w_dram_base),
        .mvin_w_k_o                       (mvin_w_k),
        .mvin_w_n_o                       (mvin_w_n),
        .mvin_w_bank_o                    (mvin_w_bank),
        .mvin_meta_dram_base_o            (mvin_meta_dram_base),
        .mvin_meta_offset_bytes_o         (mvin_meta_offset_bytes),
        .mvin_meta_byte_count_o           (mvin_meta_byte_count),
        .mvin_meta_type_o                 (mvin_meta_type),
        .gemm_m_o                         (gemm_m),
        .gemm_n_o                         (gemm_n),
        .gemm_k_o                         (gemm_k),
        .gemm_a_bank_o                    (gemm_a_bank),
        .gemm_w_bank_o                    (gemm_w_bank),
        .gemm_accumulate_en_o             (gemm_accumulate_en),
        .gemm_add_bias_en_o               (gemm_add_bias_en),
        .gemm_mode_o                      (gemm_mode),
        .gemm_bias_offset_bytes_o         (gemm_bias_offset_bytes),
        .mvout_dram_base_o                (mvout_dram_base),
        .mvout_m_o                        (mvout_m),
        .mvout_n_o                        (mvout_n),
        .mvout_output_stride_n_o          (mvout_output_stride_n),
        .mvout_mode_o                     (mvout_mode),
        .mvout_per_channel_scale_o        (mvout_per_channel_scale),
        .mvout_scale_param_o              (mvout_scale_param),
        .attention_qk_output_dram_base_o     (attention_qk_output_dram_base),
        .attention_qk_token_count_o          (attention_qk_token_count),
        .attention_qk_gamma16_fix_o          (attention_qk_gamma16_fix),
        .attention_qk_mask_en_o              (attention_qk_mask_en),
        .attention_qk_q_block_start_o        (attention_qk_q_block_start),
        .attention_qk_q_block_count_m1_o     (attention_qk_q_block_count_m1),
        .mvin_a_done_i                    (mvin_a_done),
        .mvin_w_done_i                    (mvin_w_done),
        .mvin_meta_done_i                 (mvin_meta_done),
        .gemm_done_i                      (gemm_done),
        .mvout_done_i                     (mvout_done),
        .attention_qk_done_i                 (attention_qk_done),
        .api_error_valid_i                (api_error_valid),
        .api_error_code_i                 (api_error_code),
        .profile_global_cycles_i          (profile_global_cycles),
        .profile_mvin_a_busy_cycles_i     (profile_mvin_a_busy_cycles),
        .profile_mvin_w_busy_cycles_i     (profile_mvin_w_busy_cycles),
        .profile_mvin_meta_busy_cycles_i  (profile_mvin_meta_busy_cycles),
        .profile_gemm_busy_cycles_i       (profile_gemm_busy_cycles),
        .profile_mvout_busy_cycles_i      (profile_mvout_busy_cycles),
        .profile_busy_any_cycles_i        (profile_busy_any_cycles),
        .profile_busy_multi_cycles_i      (profile_busy_multi_cycles),
        .profile_axi_r_beats_i            (profile_axi_r_beats),
        .profile_axi_w_beats_i            (profile_axi_w_beats),
        .profile_axi_ar_stall_cycles_i    (profile_axi_ar_stall_cycles),
        .profile_axi_r_stall_cycles_i     (profile_axi_r_stall_cycles),
        .profile_axi_aw_stall_cycles_i    (profile_axi_aw_stall_cycles),
        .profile_axi_w_stall_cycles_i     (profile_axi_w_stall_cycles)
    );

    if (M1_CONTROL_ONLY) begin : gen_m1_control_only
        assign cmd_ready = !backend_busy_q;
        assign api_error_valid = '0;
        assign api_error_code = '0;
        assign loader_mvin_done = '0;
        assign dma_mvin_busy = '0;
        assign dma_mvout_busy = '0;
        assign core_busy = backend_busy_q;
        assign core_sa_busy = backend_busy_q &&
                              ((backend_opcode_q == CMD_GEMM) ||
                               (backend_opcode_q == CMD_ATTENTION_QK));
        assign core_sa_done = gemm_done;
        assign core_attention_qk_done = attention_qk_done;
        assign core_error_sticky = 1'b0;
        assign core_last_error = NPU_ERR_NONE;
        assign core_command_accepted_count = '0;
        assign core_conflict_count = '0;
        assign dma_backpressure_error = 1'b0;
        assign dma_command_drop = 1'b0;
        assign mvout_bridge_underflow = 1'b0;
        assign mvout_bridge_level = '0;
        assign bridge_active_w_bank = active_w_bank;
        assign dma_mvout_resp_done = '0;
        assign m_axi_awid = '0;
        assign m_axi_awaddr = '0;
        assign m_axi_awlen = '0;
        assign m_axi_awsize = '0;
        assign m_axi_awburst = '0;
        assign m_axi_awvalid = '0;
        assign m_axi_wdata = '0;
        assign m_axi_wstrb = '0;
        assign m_axi_wlast = '0;
        assign m_axi_wvalid = '0;
        assign m_axi_bready = '0;
        assign m_axi_arid = '0;
        assign m_axi_araddr = '0;
        assign m_axi_arlen = '0;
        assign m_axi_arsize = '0;
        assign m_axi_arburst = '0;
        assign m_axi_arvalid = '0;
        assign m_axi_rready = '0;
        assign o_mvout_stream_valid = 1'b0;
        assign o_mvout_stream_data = '0;
        assign o_mvout_stream_keep = '0;
        assign o_mvout_stream_addr = '0;
        assign o_mvout_stream_last = 1'b0;

        always_ff @(posedge clk_i) begin
            if (rst) begin
                backend_busy_q      <= 1'b0;
                backend_opcode_q    <= CMD_MVIN_A;
                backend_countdown_q <= '0;
                mvin_a_done         <= 1'b0;
                mvin_w_done         <= 1'b0;
                mvin_meta_done      <= 1'b0;
                gemm_done           <= 1'b0;
                mvout_done          <= 1'b0;
                attention_qk_done      <= 1'b0;
            end else begin
                mvin_a_done    <= 1'b0;
                mvin_w_done    <= 1'b0;
                mvin_meta_done <= 1'b0;
                gemm_done      <= 1'b0;
                mvout_done     <= 1'b0;
                attention_qk_done <= 1'b0;

                if (cmd_fire) begin
                    backend_busy_q      <= 1'b1;
                    backend_opcode_q    <= cmd_opcode;
                    backend_countdown_q <= M1_DONE_COUNTER_W'(M1_DONE_LATENCY);
                end else if (backend_busy_q) begin
                    if (backend_countdown_q == '0) begin
                        backend_busy_q <= 1'b0;
                        unique case (backend_opcode_q)
                            CMD_MVIN_A:    mvin_a_done    <= 1'b1;
                            CMD_MVIN_W:    mvin_w_done    <= 1'b1;
                            CMD_META_MVIN: mvin_meta_done <= 1'b1;
                            CMD_GEMM:      gemm_done      <= 1'b1;
                            CMD_MVOUT:     mvout_done     <= 1'b1;
                            CMD_ATTENTION_QK: attention_qk_done <= 1'b1;
                            default: begin
                            end
                        endcase
                    end else begin
                        backend_countdown_q <= backend_countdown_q - 4'd1;
                    end
                end
            end
        end
    end else begin : gen_m2_real_backend
        assign bridge_active_w_bank = active_w_bank;

        api_dma_mapper #(
            .RF_DATA_WIDTH (RF_DATA_W),
            .DMA_NUM       (DMA_NUM),
            .A_DMA_IDX     (0),
            .W_DMA_IDX     (1),
            .OC_DMA_IDX    (2),
            .DISABLE_ERROR_CHECKS (DISABLE_ERROR_CHECKS)
        ) u_api_dma_mapper (
            .clk_i                          (clk_i),
            .rst_i                          (rst),
            .cmd_valid_i                    (cmd_valid),
            .cmd_ready_o                    (cmd_ready),
            .cmd_opcode_i                   (cmd_opcode),
            .mvin_a_dram_base_i             (mvin_a_dram_base),
            .mvin_a_dram_row_stride_bytes_i (mvin_a_dram_row_stride_bytes),
            .mvin_a_m_i                     (mvin_a_m),
            .mvin_a_k_i                     (mvin_a_k),
            .mvin_w_dram_base_i             (mvin_w_dram_base),
            .mvin_w_k_i                     (mvin_w_k),
            .mvin_w_n_i                     (mvin_w_n),
            .mvin_meta_dram_base_i          (mvin_meta_dram_base),
            .mvin_meta_offset_bytes_i       (mvin_meta_offset_bytes),
            .mvin_meta_byte_count_i         (mvin_meta_byte_count),
            .mvout_dram_base_i              (mvout_dram_base),
            .mvout_m_i                      (mvout_m),
            .mvout_n_i                      (mvout_n),
            .mvout_output_stride_n_i        (mvout_output_stride_n),
            .mvout_mode_i                   (mvout_mode),
            .mvout_per_channel_scale_i      (mvout_per_channel_scale),
            .mvout_scale_param_i            (mvout_scale_param),
            .attention_qk_output_dram_base_i   (attention_qk_output_dram_base),
            .attention_qk_token_count_i        (attention_qk_token_count),
            .attention_qk_gamma16_fix_i        (attention_qk_gamma16_fix),
            .attention_qk_mask_en_i            (attention_qk_mask_en),
            .attention_qk_q_block_start_i      (attention_qk_q_block_start),
            .attention_qk_q_block_count_m1_i   (attention_qk_q_block_count_m1),
            .dma_mvin_busy_i                (dma_mvin_busy),
            .dma_mvout_busy_i               (dma_mvout_busy),
            .loader_mvin_done_i             (loader_mvin_done),
            .dma_mvout_done_i               (dma_mvout_resp_done),
            .core_sa_done_i                  (core_sa_done),
            .core_attention_qk_done_i           (core_attention_qk_done),
            .core_error_sticky_i            (core_error_sticky),
            .core_last_error_i              (core_last_error),
            .mvin_dram_addr_o               (mvin_dram_addr),
            .mvin_sram_addr_o               (mvin_sram_addr),
            .mvin_col_num_o                 (mvin_col_num),
            .mvin_row_num_o                 (mvin_row_num),
            .mvout_dram_addr_o              (mvout_dram_addr),
            .mvout_sram_addr_o              (mvout_sram_addr),
            .mvout_col_num_o                (mvout_col_num),
            .mvout_row_num_o                (mvout_row_num),
            .cfg_mvin_input_type_o          (cfg_mvin_input_type),
            .cfg_mvout_output_type_o        (cfg_mvout_output_type),
            .cfg_mvin_input_precision_o     (cfg_mvin_input_precision),
            .cfg_mvout_output_precision_o   (cfg_mvout_output_precision),
            .cfg_mvin_is_quant_o            (cfg_mvin_is_quant),
            .cfg_mvout_is_quant_o           (cfg_mvout_is_quant),
            .cfg_mvin_dest_o                (cfg_mvin_dest),
            .cfg_mvout_source_o             (cfg_mvout_source),
            .cfg_mvin_sram_stride_o         (cfg_mvin_sram_stride),
            .cfg_mvin_dram_stride_o         (cfg_mvin_dram_stride),
            .cfg_mvout_sram_stride_o        (cfg_mvout_sram_stride),
            .cfg_mvout_dram_stride_o        (cfg_mvout_dram_stride),
            .cfg_mvin_input_zeropoint_o     (cfg_mvin_input_zeropoint),
            .cfg_mvout_output_zeropoint_o   (cfg_mvout_output_zeropoint),
            .cfg_mvin_input_scale_o         (cfg_mvin_input_scale),
            .cfg_mvout_output_scale_o       (cfg_mvout_output_scale),
            .cfg_mvin_input_scale_shift_o   (cfg_mvin_input_scale_shift),
            .cfg_mvout_output_scale_shift_o (cfg_mvout_output_scale_shift),
            .cfg_mvout_mode_o               (cfg_mvout_mode),
            .cfg_mvout_per_channel_o        (cfg_mvout_per_channel),
            .cfg_mvout_scale_param_o        (cfg_mvout_scale_param),
            .dma_mvin_req_en_o              (dma_mvin_req_en),
            .dma_mvout_req_en_o             (dma_mvout_req_en),
            .sa_req_en_o                    (mapper_sa_req_en),
            .attention_qk_req_en_o              (attention_qk_req_en),
            .mvin_a_done_o                  (mvin_a_done),
            .mvin_w_done_o                  (mvin_w_done),
            .mvin_meta_done_o               (mvin_meta_done),
            .gemm_done_o                    (gemm_done),
            .mvout_done_o                   (mvout_done),
            .attention_qk_done_o                (attention_qk_done),
            .api_error_valid_o              (api_error_valid),
            .api_error_code_o               (api_error_code)
        );

        versa_p_direct_dma_backend #(
            .RF_DATA_WIDTH        (RF_DATA_W),
            .DMA_NUM              (DMA_NUM),
            .AXI_ID_WIDTH         (AXI_ID_W),
            .AXI_ADDR_WIDTH       (AXI_ADDR_W),
            .AXI_DATA_WIDTH       (AXI_DATA_W),
            .AXI_MAX_BURST_BEATS  (DMA_MAX_BURST_BEATS),
            .AXI_RD_OUTSTANDING   (16),
            .BANK_DEPTH_WORDS     (BANK_DEPTH_WORDS),
            .A_BANK_DEPTH_WORDS   (A_BANK_DEPTH_WORDS),
            .W_BANK_DEPTH_WORDS   (W_BANK_DEPTH_WORDS),
            .O_BANK_DEPTH_WORDS   (O_BANK_DEPTH_WORDS),
            .META_WORD128_DEPTH   (META_WORD128_DEPTH),
            .FIFO_DEPTH           (FIFO_DEPTH),
            .O_SPLIT_BANK_HALVES  (O_SPLIT_BANK_HALVES),
            .DISABLE_ERROR_CHECKS (DISABLE_ERROR_CHECKS)
        ) u_direct_dma_backend (
            .clk_i                         (clk_i),
            .rst_i                         (rst),
            .clear_i                       (backend_clear),
            .active_w_bank_i               (active_w_bank),
            .cmd_a_bank_i                  (cmd_a_bank),
            .cmd_w_bank_i                  (cmd_w_bank),
            .cmd_o_bank_i                  (cmd_o_bank),
            .mvin_a_u8_minus_128_en_i      (mvin_a_u8_minus_128_en),
            .mvin_w_k_i                    (mvin_w_k),
            .mvin_w_n_i                    (mvin_w_n),
            .gemm_accumulate_en_i          (gemm_accumulate_en),
            .gemm_add_bias_en_i            (gemm_add_bias_en),
            .gemm_mode_i                   (gemm_mode),
            .gemm_bias_offset_bytes_i      (gemm_bias_offset_bytes),
            .attention_qk_output_dram_base_i  (attention_qk_output_dram_base),
            .attention_qk_token_count_i       (attention_qk_token_count),
            .attention_qk_gamma16_fix_i       (attention_qk_gamma16_fix),
            .attention_qk_mask_en_i           (attention_qk_mask_en),
            .attention_qk_q_block_start_i     (attention_qk_q_block_start),
            .attention_qk_q_block_count_m1_i  (attention_qk_q_block_count_m1),
            .mvin_dram_addr_i              (mvin_dram_addr),
            .mvin_sram_addr_i              (mvin_sram_addr),
            .mvin_col_num_i                (mvin_col_num),
            .mvin_row_num_i                (mvin_row_num),
            .mvout_dram_addr_i             (mvout_dram_addr),
            .mvout_sram_addr_i             (mvout_sram_addr),
            .mvout_col_num_i               (mvout_col_num),
            .mvout_row_num_i               (mvout_row_num),
            .cfg_mvin_sram_stride_i        (cfg_mvin_sram_stride),
            .cfg_mvin_dram_stride_i        (cfg_mvin_dram_stride),
            .cfg_mvout_sram_stride_i       (cfg_mvout_sram_stride),
            .cfg_mvout_dram_stride_i       (cfg_mvout_dram_stride),
            .cfg_mvout_mode_i              (cfg_mvout_mode),
            .cfg_mvout_per_channel_i       (cfg_mvout_per_channel),
            .cfg_mvout_scale_param_i       (cfg_mvout_scale_param),
            .dma_mvin_req_en_i             (dma_mvin_req_en),
            .dma_mvout_req_en_i            (dma_mvout_req_en),
            .sa_req_en_i                   (mapper_sa_req_en),
            .attention_qk_req_en_i            (attention_qk_req_en),
            .dma_mvin_busy_o               (dma_mvin_busy),
            .loader_mvin_done_o            (loader_mvin_done),
            .dma_mvout_busy_o              (dma_mvout_busy),
            .dma_mvout_done_o              (dma_mvout_resp_done),
            .mvout_writer_busy_i           (o_writer_busy),
            .mvout_writer_done_i           (o_writer_done),
            .m_axi_awid_o                  (m_axi_awid),
            .m_axi_awaddr_o                (m_axi_awaddr),
            .m_axi_awlen_o                 (m_axi_awlen),
            .m_axi_awsize_o                (m_axi_awsize),
            .m_axi_awburst_o               (m_axi_awburst),
            .m_axi_awvalid_o               (m_axi_awvalid),
            .m_axi_wdata_o                 (m_axi_wdata),
            .m_axi_wstrb_o                 (m_axi_wstrb),
            .m_axi_wlast_o                 (m_axi_wlast),
            .m_axi_wvalid_o                (m_axi_wvalid),
            .m_axi_bready_o                (m_axi_bready),
            .m_axi_arid_o                  (m_axi_arid),
            .m_axi_araddr_o                (m_axi_araddr),
            .m_axi_arlen_o                 (m_axi_arlen),
            .m_axi_arsize_o                (m_axi_arsize),
            .m_axi_arburst_o               (m_axi_arburst),
            .m_axi_arvalid_o               (m_axi_arvalid),
            .m_axi_arready_i               (m_axi_arready),
            .m_axi_rid_i                   (m_axi_rid),
            .m_axi_rdata_i                 (m_axi_rdata),
            .m_axi_rresp_i                 (m_axi_rresp),
            .m_axi_rlast_i                 (m_axi_rlast),
            .m_axi_rvalid_i                (m_axi_rvalid),
            .m_axi_rready_o                (m_axi_rready),
            .mvout_dma_valid_o             (o_mvout_stream_valid),
            .mvout_dma_ready_i             (o_mvout_stream_ready),
            .mvout_dma_data_o              (o_mvout_stream_data),
            .mvout_dma_keep_o              (o_mvout_stream_keep),
            .mvout_dma_addr_o              (o_mvout_stream_addr),
            .mvout_dma_last_o              (o_mvout_stream_last),
            .core_busy_o                   (core_busy),
            .core_sa_busy_o                (core_sa_busy),
            .core_sa_done_o                (core_sa_done),
            .core_attention_qk_done_o         (core_attention_qk_done),
            .core_error_sticky_o           (core_error_sticky),
            .core_last_error_o             (core_last_error),
            .command_accepted_count_o      (core_command_accepted_count),
            .conflict_count_o              (core_conflict_count),
            .dma_backpressure_error_o      (dma_backpressure_error),
            .dma_command_drop_o            (dma_command_drop),
            .mvout_bridge_underflow_o      (mvout_bridge_underflow),
            .mvout_bridge_level_o          (mvout_bridge_level)
        );

    end

    assign busy_o = core_busy || (|dma_mvin_busy) || (|dma_mvout_busy);

    wire unused_ctrl_outputs = cmd_a_bank ^ cmd_w_bank ^ bridge_active_w_bank ^
                               core_sa_busy ^ core_sa_done ^
                               ^core_command_accepted_count ^ ^core_conflict_count ^
                               dma_backpressure_error ^ dma_command_drop ^
                               mvout_bridge_underflow ^ ^mvout_bridge_level ^
                               ^mvin_meta_type ^ ^gemm_m ^ ^gemm_n ^ ^gemm_k ^
                               mvin_a_bank ^ gemm_a_bank ^ gemm_w_bank ^ gemm_accumulate_en ^
                               gemm_add_bias_en ^ ^gemm_mode ^ ^gemm_bias_offset_bytes ^
                               mvout_per_channel_scale;
endmodule

`default_nettype wire
