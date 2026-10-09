`default_nettype none

import npu_spm_pkg::*;

// Direct DMA backend for the Phase-1 five-command API.
//
// This replaces the legacy DMA_TOP -> dma_core_bridge -> dma_core_adapter
// datapath. The API mapper still normalizes descriptors, but A/W/meta traffic
// is read directly from AXI into Versa_P_core streams, and O/MVOUT exits the
// core stream directly into the two-lane writer in the top shell.
module versa_p_direct_dma_backend #(
    parameter int RF_DATA_WIDTH        = 64,
    parameter int DMA_NUM              = 3,
    parameter int AXI_ID_WIDTH         = 4,
    parameter int AXI_ADDR_WIDTH       = 32,
    parameter int AXI_DATA_WIDTH       = 128,
    parameter int AXI_MAX_BURST_BEATS  = 16,
    parameter int AXI_RD_OUTSTANDING   = 16,
    parameter int BANK_DEPTH_WORDS     = 16384,
    parameter int A_BANK_DEPTH_WORDS   = 12288,
    parameter int W_BANK_DEPTH_WORDS   = 12288,
    parameter int O_BANK_DEPTH_WORDS   = 16384,
    parameter int META_WORD128_DEPTH   = 2048,
    parameter int FIFO_DEPTH           = 16,
    parameter bit O_SPLIT_BANK_HALVES  = 1'b0,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0,
    localparam int HALF_W              = RF_DATA_WIDTH / 2,
    localparam int QUARTER_W           = RF_DATA_WIDTH / 4,
    localparam int AXI_STRB_W          = AXI_DATA_WIDTH / 8,
    localparam int O_PHYS_DEPTH_WORDS  = O_SPLIT_BANK_HALVES ? (2 * O_BANK_DEPTH_WORDS) : O_BANK_DEPTH_WORDS,
    localparam int MAX_AW_DEPTH_WORDS  = (A_BANK_DEPTH_WORDS > W_BANK_DEPTH_WORDS) ? A_BANK_DEPTH_WORDS : W_BANK_DEPTH_WORDS,
    localparam int MAX_BANK_DEPTH_WORDS = (MAX_AW_DEPTH_WORDS > O_PHYS_DEPTH_WORDS) ? MAX_AW_DEPTH_WORDS : O_PHYS_DEPTH_WORDS,
    localparam int ADDR_W              = (MAX_BANK_DEPTH_WORDS <= 1) ? 1 : $clog2(MAX_BANK_DEPTH_WORDS),
    localparam int META_ADDR_W         = (META_WORD128_DEPTH <= 1) ? 1 : $clog2(META_WORD128_DEPTH),
    localparam int META_BYTE_ADDR_W    = META_ADDR_W + 4
) (
    input  wire logic                                  clk_i,
    input  wire logic                                  rst_i,
    input  wire logic                                  clear_i,

    input  wire logic                                  active_w_bank_i,
    input  wire logic                                  cmd_a_bank_i,
    input  wire logic                                  cmd_w_bank_i,
    input  wire logic                                  cmd_o_bank_i,
    input  wire logic                                  mvin_a_u8_minus_128_en_i,
    input  wire logic [15:0]                           mvin_w_k_i,
    input  wire logic [15:0]                           mvin_w_n_i,
    input  wire logic                                  gemm_accumulate_en_i,
    input  wire logic                                  gemm_add_bias_en_i,
    input  wire npu_mode_e                             gemm_mode_i,
    input  wire logic [31:0]                           gemm_bias_offset_bytes_i,
    input  wire logic [31:0]                           attention_qk_output_dram_base_i,
    input  wire logic [15:0]                           attention_qk_token_count_i,
    input  wire logic [31:0]                           attention_qk_gamma16_fix_i,
    input  wire logic                                  attention_qk_mask_en_i,
    input  wire logic [4:0]                            attention_qk_q_block_start_i,
    input  wire logic [4:0]                            attention_qk_q_block_count_m1_i,

    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvin_dram_addr_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvin_sram_addr_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvin_col_num_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvin_row_num_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvout_dram_addr_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvout_sram_addr_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvout_col_num_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        mvout_row_num_i,
    input  wire logic [DMA_NUM-1:0][QUARTER_W-1:0]     cfg_mvin_sram_stride_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        cfg_mvin_dram_stride_i,
    input  wire logic [DMA_NUM-1:0][QUARTER_W-1:0]     cfg_mvout_sram_stride_i,
    input  wire logic [DMA_NUM-1:0][HALF_W-1:0]        cfg_mvout_dram_stride_i,
    input  wire logic [DMA_NUM-1:0][1:0]               cfg_mvout_mode_i,
    input  wire logic [DMA_NUM-1:0]                    cfg_mvout_per_channel_i,
    input  wire logic [DMA_NUM-1:0][31:0]              cfg_mvout_scale_param_i,
    input  wire logic [DMA_NUM-1:0]                    dma_mvin_req_en_i,
    input  wire logic [DMA_NUM-1:0]                    dma_mvout_req_en_i,
    input  wire logic                                  sa_req_en_i,
    input  wire logic                                  attention_qk_req_en_i,

    output logic [DMA_NUM-1:0]                         dma_mvin_busy_o,
    output logic [DMA_NUM-1:0]                         loader_mvin_done_o,
    output logic [DMA_NUM-1:0]                         dma_mvout_busy_o,
    output logic [DMA_NUM-1:0]                         dma_mvout_done_o,
    input  wire logic                                  mvout_writer_busy_i,
    input  wire logic                                  mvout_writer_done_i,

    output logic [DMA_NUM-1:0][AXI_ID_WIDTH-1:0]       m_axi_awid_o,
    output logic [DMA_NUM-1:0][AXI_ADDR_WIDTH-1:0]     m_axi_awaddr_o,
    output logic [DMA_NUM-1:0][7:0]                    m_axi_awlen_o,
    output logic [DMA_NUM-1:0][2:0]                    m_axi_awsize_o,
    output logic [DMA_NUM-1:0][1:0]                    m_axi_awburst_o,
    output logic [DMA_NUM-1:0]                         m_axi_awvalid_o,
    output logic [DMA_NUM-1:0][AXI_DATA_WIDTH-1:0]     m_axi_wdata_o,
    output logic [DMA_NUM-1:0][AXI_STRB_W-1:0]         m_axi_wstrb_o,
    output logic [DMA_NUM-1:0]                         m_axi_wlast_o,
    output logic [DMA_NUM-1:0]                         m_axi_wvalid_o,
    output logic [DMA_NUM-1:0]                         m_axi_bready_o,
    output logic [DMA_NUM-1:0][AXI_ID_WIDTH-1:0]       m_axi_arid_o,
    output logic [DMA_NUM-1:0][AXI_ADDR_WIDTH-1:0]     m_axi_araddr_o,
    output logic [DMA_NUM-1:0][7:0]                    m_axi_arlen_o,
    output logic [DMA_NUM-1:0][2:0]                    m_axi_arsize_o,
    output logic [DMA_NUM-1:0][1:0]                    m_axi_arburst_o,
    output logic [DMA_NUM-1:0]                         m_axi_arvalid_o,
    input  wire logic [DMA_NUM-1:0]                    m_axi_arready_i,
    input  wire logic [DMA_NUM-1:0][AXI_ID_WIDTH-1:0]  m_axi_rid_i,
    input  wire logic [DMA_NUM-1:0][AXI_DATA_WIDTH-1:0] m_axi_rdata_i,
    input  wire logic [DMA_NUM-1:0][1:0]               m_axi_rresp_i,
    input  wire logic [DMA_NUM-1:0]                    m_axi_rlast_i,
    input  wire logic [DMA_NUM-1:0]                    m_axi_rvalid_i,
    output logic [DMA_NUM-1:0]                         m_axi_rready_o,

    output logic                                       mvout_dma_valid_o,
    input  wire logic                                  mvout_dma_ready_i,
    output logic [AXI_DATA_WIDTH-1:0]                  mvout_dma_data_o,
    output logic [AXI_STRB_W-1:0]                      mvout_dma_keep_o,
    output logic [63:0]                                mvout_dma_addr_o,
    output logic                                       mvout_dma_last_o,

    output logic                                       core_busy_o,
    output logic                                       core_sa_busy_o,
    output logic                                       core_sa_done_o,
    output logic                                       core_attention_qk_done_o,
    output logic                                       core_error_sticky_o,
    output npu_error_e                                 core_last_error_o,
    output logic [31:0]                                command_accepted_count_o,
    output logic [31:0]                                conflict_count_o,
    output logic                                       dma_backpressure_error_o,
    output logic                                       dma_command_drop_o,
    output logic                                       mvout_bridge_underflow_o,
    output logic [$clog2(FIFO_DEPTH + 1)-1:0]          mvout_bridge_level_o
);
    localparam int A_DMA_IDX = 0;
    localparam int W_DMA_IDX = 1;
    localparam int OC_DMA_IDX = 2;
    localparam logic [2:0] CMD_MVIN_A    = 3'd0;
    localparam logic [2:0] CMD_MVIN_W    = 3'd1;
    localparam logic [2:0] CMD_META_MVIN = 3'd2;
    localparam logic [2:0] CMD_GEMM      = 3'd3;
    localparam logic [2:0] CMD_MVOUT     = 3'd4;
    localparam logic [2:0] CMD_ATTENTION_QK = 3'd5;

    logic a_pending_q;
    logic w_pending_q;
    logic meta_pending_q;
    logic gemm_pending_q;
    logic mvout_pending_q;
    logic attention_qk_pending_q;
    logic [2:0] selected_cmd;
    logic selected_valid;
    logic core_cmd_ready;
    logic core_cmd_fire;
    logic core_cmd_a_bank;
    logic core_cmd_w_bank;
    logic core_cmd_o_bank;

    logic a_start;
    logic w_start;
    logic meta_start;
    logic a_reader_busy;
    logic w_reader_busy;
    logic meta_reader_busy;
    logic a_reader_done;
    logic w_reader_done;
    logic meta_reader_done;
    logic a_reader_error;
    logic w_reader_error;
    logic meta_reader_error;

    logic a_dma_valid;
    logic a_dma_ready;
    logic [AXI_DATA_WIDTH-1:0] a_dma_data;
    logic [AXI_STRB_W-1:0] a_dma_keep;
    logic a_dma_last;
    logic w_dma_valid;
    logic w_dma_ready;
    logic [AXI_DATA_WIDTH-1:0] w_dma_data;
    logic [AXI_STRB_W-1:0] w_dma_keep;
    logic w_dma_last;
    logic meta_dma_valid;
    logic meta_dma_ready;
    logic [AXI_DATA_WIDTH-1:0] meta_dma_data;
    logic [AXI_STRB_W-1:0] meta_dma_keep;
    logic meta_dma_last;
    logic [META_ADDR_W-1:0] meta_word_addr_q;

    logic [ADDR_W-1:0] a_base_word_q;
    logic a_bank_q;
    logic gemm_a_bank_q;
    logic [15:0] a_rows_q;
    logic [15:0] a_row_bytes_q;
    logic [15:0] a_stride_words_q;
    logic a_u8_minus_128_q;
    logic [AXI_ADDR_WIDTH-1:0] a_dram_base_q;
    logic [31:0] a_dram_stride_q;

    logic [ADDR_W-1:0] w_base_word_q;
    logic w_bank_q;
    logic [15:0] w_k_loaded_q;
    logic [15:0] w_n_loaded_q;
    logic [AXI_ADDR_WIDTH-1:0] w_dram_base_q;
    logic w_prepare_q;
    logic [15:0] w_rows_q;
    logic [15:0] w_row_bytes_q;
    logic [31:0] w_total_bytes_q;
    logic [31:0] w_dram_stride_q;

    logic [AXI_ADDR_WIDTH-1:0] meta_dram_base_q;
    logic [31:0] meta_byte_count_q;
    logic [META_ADDR_W-1:0] meta_start_word_addr_q;

    logic [ADDR_W-1:0] mvout_req_word_addr_q;
    logic [15:0] mvout_m_count_q;
    logic [15:0] mvout_n_count_q;
    logic [63:0] mvout_dram_base_addr_q;
    logic [31:0] mvout_dram_row_stride_bytes_q;
    logic mvout_o_bank_q;
    npu_mvout_mode_e mvout_mode_q;
    logic mvout_per_channel_scale_q;
    logic [31:0] mvout_tensor_scale_q8_24_q;
    logic [META_BYTE_ADDR_W-1:0] mvout_scale_addr_q;

    logic core_a_busy;
    logic core_a0_busy;
    logic core_a1_busy;
    logic core_w0_busy;
    logic core_w1_busy;
    logic core_co_busy;
    logic core_co0_busy;
    logic core_co1_busy;
    logic core_busy_raw;
    logic core_selected_a_busy;
    logic core_selected_w_busy;
    logic core_selected_gemm_busy;
    logic selected_gemm_attention_qk;
    logic core_error_sticky_raw;
    npu_error_e core_last_error_raw;
    logic dma_reader_error;
    logic dma_error_sticky_q;
    logic mvout_fifo_almost_full;
    logic core_attention_qk_done;
    logic attention_qk_inflight_q;
    logic attention_qk_core_done_q;
    logic attention_qk_writer_seen_busy_q;
    logic attention_qk_q_bank_q;
    logic attention_qk_k_bank_q;
    logic attention_qk_o_bank_q;
    logic [15:0] attention_qk_token_count_q;
    logic [31:0] attention_qk_gamma16_fix_q;
    logic attention_qk_mask_en_q;
    logic [31:0] attention_qk_output_dram_base_q;
    logic [4:0] attention_qk_q_block_start_q;
    logic [4:0] attention_qk_q_block_count_m1_q;
    logic a_done_pending_q;
    logic a_seen_busy_q;
    logic w_done_pending_q;
    logic w_seen_busy_q;
    logic meta_done_pending_q;
    logic gemm_inflight_q;
    logic gemm_seen_busy_q;
    logic gemm_o_bank_q;
    logic gemm_accumulate_en_q;
    logic gemm_add_bias_en_q;
    npu_mode_e gemm_mode_q;
    logic [31:0] gemm_bias_offset_bytes_q;

    function automatic logic [15:0] plus_one16(input logic [HALF_W-1:0] value);
        plus_one16 = 16'(value[15:0] + 16'd1);
    endfunction

    function automatic logic [15:0] ceil_div32_16(input logic [15:0] value);
        ceil_div32_16 = (value + 16'd31) >> 5;
    endfunction

    function automatic logic [15:0] ceil_div16_16(input logic [15:0] value);
        ceil_div16_16 = (value + 16'd15) >> 4;
    endfunction

    function automatic logic [15:0] max16(input logic [15:0] a, input logic [15:0] b);
        max16 = (a > b) ? a : b;
    endfunction

    function automatic npu_mvout_mode_e cast_mvout_mode(input logic [1:0] mode);
        cast_mvout_mode = npu_mvout_mode_e'(mode);
    endfunction

    always_comb begin
        selected_cmd = CMD_MVIN_A;
        selected_valid = 1'b0;
        if (a_pending_q) begin
            selected_cmd = CMD_MVIN_A;
            selected_valid = 1'b1;
        end else if (w_pending_q) begin
            selected_cmd = CMD_MVIN_W;
            selected_valid = 1'b1;
        end else if (meta_pending_q) begin
            selected_cmd = CMD_META_MVIN;
            selected_valid = 1'b1;
        end else if (gemm_pending_q) begin
            selected_cmd = CMD_GEMM;
            selected_valid = 1'b1;
        end else if (mvout_pending_q) begin
            selected_cmd = CMD_MVOUT;
            selected_valid = 1'b1;
        end else if (attention_qk_pending_q) begin
            selected_cmd = CMD_ATTENTION_QK;
            selected_valid = 1'b1;
        end
    end

    assign core_cmd_fire = selected_valid && core_cmd_ready;
    assign a_start = core_cmd_fire && (selected_cmd == CMD_MVIN_A);
    assign w_start = core_cmd_fire && (selected_cmd == CMD_MVIN_W);
    assign meta_start = core_cmd_fire && (selected_cmd == CMD_META_MVIN);
    assign selected_gemm_attention_qk =
        (selected_cmd == CMD_GEMM) && (gemm_mode_q == NPU_MODE_ATTENTION_QK);
    assign core_cmd_a_bank = (selected_cmd == CMD_MVIN_A) ? a_bank_q :
                             ((selected_cmd == CMD_GEMM) ? gemm_a_bank_q :
                              ((selected_cmd == CMD_ATTENTION_QK) ? attention_qk_q_bank_q : 1'b0));
    assign core_cmd_w_bank = (selected_cmd == CMD_MVIN_W) ? w_bank_q :
                             (((selected_cmd == CMD_ATTENTION_QK) || selected_gemm_attention_qk)
                              ? attention_qk_k_bank_q : active_w_bank_i);
    assign core_cmd_o_bank = (selected_cmd == CMD_GEMM) ? gemm_o_bank_q :
                             ((selected_cmd == CMD_MVOUT) ? mvout_o_bank_q :
                              ((selected_cmd == CMD_ATTENTION_QK) ? attention_qk_o_bank_q : 1'b0));

    assign dma_mvin_busy_o[A_DMA_IDX] = a_pending_q || a_reader_busy || a_done_pending_q;
    assign dma_mvin_busy_o[W_DMA_IDX] = w_prepare_q || w_pending_q || w_reader_busy || w_done_pending_q;
    assign dma_mvin_busy_o[OC_DMA_IDX] = meta_pending_q || meta_reader_busy || meta_done_pending_q;
    assign dma_mvout_busy_o = {mvout_writer_busy_i || mvout_pending_q || attention_qk_inflight_q, 2'b00};
    assign dma_mvout_done_o = {mvout_writer_done_i && !attention_qk_inflight_q, 2'b00};
    assign core_busy_o = core_busy_raw || selected_valid || w_prepare_q ||
                         a_reader_busy || w_reader_busy || meta_reader_busy;
    assign core_sa_busy_o = gemm_pending_q || gemm_inflight_q ||
                            attention_qk_pending_q || attention_qk_inflight_q;
    assign core_selected_a_busy = a_bank_q ? core_a1_busy : core_a0_busy;
    assign core_selected_w_busy = w_bank_q ? core_w1_busy : core_w0_busy;
    assign core_selected_gemm_busy =
        (gemm_a_bank_q ? core_a1_busy : core_a0_busy) ||
        (gemm_o_bank_q ? core_co1_busy : core_co0_busy);
    assign dma_reader_error = a_reader_error || w_reader_error || meta_reader_error;
    assign dma_backpressure_error_o = dma_error_sticky_q || dma_reader_error;
    assign core_error_sticky_o = core_error_sticky_raw || dma_error_sticky_q || dma_reader_error;
    assign core_last_error_o = (dma_error_sticky_q || dma_reader_error) ? NPU_ERR_ALIGN : core_last_error_raw;
    assign dma_command_drop_o = (|dma_mvin_req_en_i && (a_pending_q || w_prepare_q || w_pending_q || meta_pending_q)) ||
                                (|dma_mvout_req_en_i && mvout_pending_q) ||
                                (sa_req_en_i && gemm_pending_q) ||
                                (attention_qk_req_en_i && attention_qk_pending_q);
    assign mvout_bridge_underflow_o = 1'b0;
    assign mvout_bridge_level_o = '0;

    assign m_axi_awid_o = '0;
    assign m_axi_awaddr_o = '0;
    assign m_axi_awlen_o = '0;
    assign m_axi_awsize_o = '0;
    assign m_axi_awburst_o = '0;
    assign m_axi_awvalid_o = '0;
    assign m_axi_wdata_o = '0;
    assign m_axi_wstrb_o = '0;
    assign m_axi_wlast_o = '0;
    assign m_axi_wvalid_o = '0;
    assign m_axi_bready_o = '0;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            a_pending_q <= 1'b0;
            w_pending_q <= 1'b0;
            meta_pending_q <= 1'b0;
            gemm_pending_q <= 1'b0;
            mvout_pending_q <= 1'b0;
            attention_qk_pending_q <= 1'b0;
            loader_mvin_done_o <= '0;
            core_sa_done_o <= 1'b0;
            core_attention_qk_done_o <= 1'b0;
            a_done_pending_q <= 1'b0;
            a_seen_busy_q <= 1'b0;
            w_done_pending_q <= 1'b0;
            w_seen_busy_q <= 1'b0;
            meta_done_pending_q <= 1'b0;
            gemm_inflight_q <= 1'b0;
            gemm_seen_busy_q <= 1'b0;
            meta_word_addr_q <= '0;
            a_base_word_q <= '0;
            a_bank_q <= 1'b0;
            gemm_a_bank_q <= 1'b0;
            a_rows_q <= '0;
            a_row_bytes_q <= '0;
            a_stride_words_q <= '0;
            a_u8_minus_128_q <= 1'b0;
            a_dram_base_q <= '0;
            a_dram_stride_q <= '0;
            w_base_word_q <= '0;
            w_bank_q <= 1'b0;
            w_k_loaded_q <= '0;
            w_n_loaded_q <= '0;
            w_dram_base_q <= '0;
            w_prepare_q <= 1'b0;
            w_rows_q <= '0;
            w_row_bytes_q <= '0;
            w_total_bytes_q <= '0;
            w_dram_stride_q <= '0;
            meta_dram_base_q <= '0;
            meta_byte_count_q <= '0;
            meta_start_word_addr_q <= '0;
            mvout_req_word_addr_q <= '0;
            mvout_m_count_q <= '0;
            mvout_n_count_q <= '0;
            mvout_dram_base_addr_q <= '0;
            mvout_dram_row_stride_bytes_q <= '0;
            mvout_o_bank_q <= 1'b0;
            mvout_mode_q <= NPU_MVOUT_RAW_I32;
            mvout_per_channel_scale_q <= 1'b0;
            mvout_tensor_scale_q8_24_q <= '0;
            mvout_scale_addr_q <= '0;
            gemm_o_bank_q <= 1'b0;
            gemm_accumulate_en_q <= 1'b0;
            gemm_add_bias_en_q <= 1'b0;
            gemm_mode_q <= NPU_MODE_INT8;
            gemm_bias_offset_bytes_q <= '0;
            attention_qk_inflight_q <= 1'b0;
            attention_qk_core_done_q <= 1'b0;
            attention_qk_writer_seen_busy_q <= 1'b0;
            attention_qk_q_bank_q <= 1'b0;
            attention_qk_k_bank_q <= 1'b0;
            attention_qk_o_bank_q <= 1'b0;
            attention_qk_token_count_q <= '0;
            attention_qk_gamma16_fix_q <= '0;
            attention_qk_mask_en_q <= 1'b0;
            attention_qk_output_dram_base_q <= '0;
            attention_qk_q_block_start_q <= '0;
            attention_qk_q_block_count_m1_q <= '0;
            dma_error_sticky_q <= 1'b0;
        end else begin
            loader_mvin_done_o <= '0;
            core_sa_done_o <= 1'b0;
            core_attention_qk_done_o <= 1'b0;

            if (clear_i) begin
                a_pending_q <= 1'b0;
                w_pending_q <= 1'b0;
                w_prepare_q <= 1'b0;
                meta_pending_q <= 1'b0;
                gemm_pending_q <= 1'b0;
                mvout_pending_q <= 1'b0;
                attention_qk_pending_q <= 1'b0;
                a_done_pending_q <= 1'b0;
                a_seen_busy_q <= 1'b0;
                w_done_pending_q <= 1'b0;
                w_seen_busy_q <= 1'b0;
                meta_done_pending_q <= 1'b0;
                gemm_inflight_q <= 1'b0;
                gemm_seen_busy_q <= 1'b0;
                gemm_o_bank_q <= 1'b0;
                gemm_accumulate_en_q <= 1'b0;
                gemm_add_bias_en_q <= 1'b0;
                gemm_mode_q <= NPU_MODE_INT8;
                gemm_bias_offset_bytes_q <= '0;
                attention_qk_inflight_q <= 1'b0;
                attention_qk_core_done_q <= 1'b0;
                attention_qk_writer_seen_busy_q <= 1'b0;
                dma_error_sticky_q <= 1'b0;
            end else begin
                if (dma_reader_error) begin
                    dma_error_sticky_q <= 1'b1;
                end
                if (dma_mvin_req_en_i[A_DMA_IDX]) begin
                    a_pending_q <= 1'b1;
                    a_rows_q <= plus_one16(mvin_row_num_i[A_DMA_IDX]);
                    a_row_bytes_q <= plus_one16(mvin_col_num_i[A_DMA_IDX]);
                    a_stride_words_q <= max16(
                        ceil_div32_16(plus_one16(mvin_col_num_i[A_DMA_IDX])),
                        ceil_div32_16(cfg_mvin_sram_stride_i[A_DMA_IDX])
                    );
                    a_base_word_q <= ADDR_W'(mvin_sram_addr_i[A_DMA_IDX] >> 1);
                    a_bank_q <= cmd_a_bank_i;
                    a_u8_minus_128_q <= mvin_a_u8_minus_128_en_i;
                    a_dram_base_q <= AXI_ADDR_WIDTH'(mvin_dram_addr_i[A_DMA_IDX]);
                    a_dram_stride_q <= cfg_mvin_dram_stride_i[A_DMA_IDX];
                end
                if (dma_mvin_req_en_i[W_DMA_IDX]) begin
                    w_prepare_q <= 1'b1;
                    w_base_word_q <= ADDR_W'(mvin_sram_addr_i[W_DMA_IDX] >> 1);
                    w_bank_q <= cmd_w_bank_i;
                    w_k_loaded_q <= mvin_w_k_i;
                    w_n_loaded_q <= mvin_w_n_i;
                    w_dram_base_q <= AXI_ADDR_WIDTH'(mvin_dram_addr_i[W_DMA_IDX]);
                    w_rows_q <= plus_one16(mvin_row_num_i[W_DMA_IDX]);
                    w_row_bytes_q <= plus_one16(mvin_col_num_i[W_DMA_IDX]);
                    w_dram_stride_q <= cfg_mvin_dram_stride_i[W_DMA_IDX];
                end
                if (dma_mvin_req_en_i[OC_DMA_IDX]) begin
                    meta_pending_q <= 1'b1;
                    meta_dram_base_q <= AXI_ADDR_WIDTH'(mvin_dram_addr_i[OC_DMA_IDX]);
                    meta_byte_count_q <= {16'd0, plus_one16(mvin_col_num_i[OC_DMA_IDX])};
                    meta_start_word_addr_q <= META_ADDR_W'(mvin_sram_addr_i[OC_DMA_IDX][META_ADDR_W-1:0]);
                    meta_word_addr_q <= META_ADDR_W'(mvin_sram_addr_i[OC_DMA_IDX][META_ADDR_W-1:0]);
                end
                if (sa_req_en_i) begin
                    gemm_pending_q <= 1'b1;
                    gemm_a_bank_q <= cmd_a_bank_i;
                    gemm_o_bank_q <= cmd_o_bank_i;
                    gemm_accumulate_en_q <= gemm_accumulate_en_i;
                    gemm_add_bias_en_q <= gemm_add_bias_en_i;
                    gemm_mode_q <= gemm_mode_i;
                    gemm_bias_offset_bytes_q <= gemm_bias_offset_bytes_i;
                    if (gemm_mode_i == NPU_MODE_ATTENTION_QK) begin
                        attention_qk_q_bank_q <= cmd_a_bank_i;
                        attention_qk_k_bank_q <= cmd_w_bank_i;
                        attention_qk_o_bank_q <= cmd_o_bank_i;
                        attention_qk_token_count_q <= attention_qk_token_count_i;
                        attention_qk_gamma16_fix_q <= attention_qk_gamma16_fix_i;
                        attention_qk_mask_en_q <= attention_qk_mask_en_i;
                        attention_qk_output_dram_base_q <= '0;
                        attention_qk_q_block_start_q <= attention_qk_q_block_start_i;
                        attention_qk_q_block_count_m1_q <= attention_qk_q_block_count_m1_i;
                    end
                end
                if (dma_mvout_req_en_i[OC_DMA_IDX]) begin
                    mvout_pending_q <= 1'b1;
                    mvout_req_word_addr_q <= ADDR_W'(mvout_sram_addr_i[OC_DMA_IDX] >> 1);
                    mvout_m_count_q <= plus_one16(mvout_row_num_i[OC_DMA_IDX]);
                    mvout_n_count_q <= plus_one16(mvout_col_num_i[OC_DMA_IDX]);
                    mvout_dram_base_addr_q <= {32'd0, mvout_dram_addr_i[OC_DMA_IDX]};
                    if (cfg_mvout_mode_i[OC_DMA_IDX] == NPU_MVOUT_ATTENTION_QK_LOGP) begin
                        mvout_dram_row_stride_bytes_q <= (cfg_mvout_dram_stride_i[OC_DMA_IDX] == '0)
                            ? {16'd0, plus_one16(mvout_col_num_i[OC_DMA_IDX])}
                            : {16'd0, cfg_mvout_dram_stride_i[OC_DMA_IDX]};
                    end else begin
                        mvout_dram_row_stride_bytes_q <= {16'd0, cfg_mvout_dram_stride_i[OC_DMA_IDX]} << 2;
                    end
                    mvout_o_bank_q <= cmd_o_bank_i;
                    mvout_mode_q <= cast_mvout_mode(cfg_mvout_mode_i[OC_DMA_IDX]);
                    mvout_per_channel_scale_q <= cfg_mvout_per_channel_i[OC_DMA_IDX];
                    mvout_tensor_scale_q8_24_q <= cfg_mvout_scale_param_i[OC_DMA_IDX];
                    mvout_scale_addr_q <= META_BYTE_ADDR_W'(cfg_mvout_scale_param_i[OC_DMA_IDX]);
                end
                if (attention_qk_req_en_i) begin
                    attention_qk_pending_q <= 1'b1;
                    attention_qk_q_bank_q <= cmd_a_bank_i;
                    attention_qk_k_bank_q <= cmd_w_bank_i;
                    attention_qk_o_bank_q <= cmd_o_bank_i;
                    attention_qk_token_count_q <= attention_qk_token_count_i;
                    attention_qk_gamma16_fix_q <= attention_qk_gamma16_fix_i;
                    attention_qk_mask_en_q <= attention_qk_mask_en_i;
                    attention_qk_output_dram_base_q <= attention_qk_output_dram_base_i;
                    attention_qk_q_block_start_q <= attention_qk_q_block_start_i;
                    attention_qk_q_block_count_m1_q <= attention_qk_q_block_count_m1_i;
                end

                if (w_prepare_q) begin
                    w_prepare_q <= 1'b0;
                    w_pending_q <= 1'b1;
                    w_total_bytes_q <= {16'd0, w_rows_q} * {16'd0, w_row_bytes_q};
                end

                if (a_start) begin
                    a_pending_q <= 1'b0;
                    a_done_pending_q <= 1'b1;
                    a_seen_busy_q <= 1'b0;
                end
                if (w_start) begin
                    w_pending_q <= 1'b0;
                    w_done_pending_q <= 1'b1;
                    w_seen_busy_q <= 1'b0;
                end
                if (meta_start) begin
                    meta_pending_q <= 1'b0;
                    meta_done_pending_q <= 1'b1;
                    meta_word_addr_q <= meta_start_word_addr_q;
                end
                if (core_cmd_fire && (selected_cmd == CMD_GEMM)) begin
                    gemm_pending_q <= 1'b0;
                    gemm_inflight_q <= 1'b1;
                    gemm_seen_busy_q <= 1'b0;
                end
                if (core_cmd_fire && (selected_cmd == CMD_MVOUT)) begin
                    mvout_pending_q <= 1'b0;
                end
                if (core_cmd_fire && (selected_cmd == CMD_ATTENTION_QK)) begin
                    attention_qk_pending_q <= 1'b0;
                    attention_qk_inflight_q <= 1'b1;
                    attention_qk_core_done_q <= 1'b0;
                    attention_qk_writer_seen_busy_q <= 1'b0;
                end

                if (a_done_pending_q) begin
                    if (core_selected_a_busy) begin
                        a_seen_busy_q <= 1'b1;
                    end else if (a_seen_busy_q && !a_reader_busy) begin
                        a_done_pending_q <= 1'b0;
                        loader_mvin_done_o[A_DMA_IDX] <= 1'b1;
                    end
                end
                if (w_done_pending_q) begin
                    if (core_selected_w_busy) begin
                        w_seen_busy_q <= 1'b1;
                    end else if (w_seen_busy_q && !w_reader_busy) begin
                        w_done_pending_q <= 1'b0;
                        loader_mvin_done_o[W_DMA_IDX] <= 1'b1;
                    end
                end
                if (meta_done_pending_q && meta_reader_done) begin
                    meta_done_pending_q <= 1'b0;
                    loader_mvin_done_o[OC_DMA_IDX] <= 1'b1;
                end
                if (gemm_inflight_q) begin
                    if (core_selected_gemm_busy) begin
                        gemm_seen_busy_q <= 1'b1;
                    end else if (gemm_seen_busy_q) begin
                        gemm_inflight_q <= 1'b0;
                        core_sa_done_o <= 1'b1;
                    end
                end
                if (attention_qk_inflight_q) begin
                    if (mvout_writer_busy_i) begin
                        attention_qk_writer_seen_busy_q <= 1'b1;
                    end
                    if (core_attention_qk_done) begin
                        attention_qk_core_done_q <= 1'b1;
                    end
                    if (mvout_writer_done_i &&
                        (attention_qk_core_done_q || core_attention_qk_done) &&
                        attention_qk_writer_seen_busy_q) begin
                        attention_qk_inflight_q <= 1'b0;
                        attention_qk_core_done_q <= 1'b0;
                        attention_qk_writer_seen_busy_q <= 1'b0;
                        core_attention_qk_done_o <= 1'b1;
                    end
                end
                if (meta_dma_valid && meta_dma_ready) begin
                    meta_word_addr_q <= meta_word_addr_q + META_ADDR_W'(1);
                end
            end
        end
    end

    axi_aligned_read_stream #(
        .AXI_ID_WIDTH        (AXI_ID_WIDTH),
        .AXI_ADDR_WIDTH      (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH      (AXI_DATA_WIDTH),
        .AXI_MAX_BURST_BEATS (AXI_MAX_BURST_BEATS),
        .AXI_RD_OUTSTANDING  (AXI_RD_OUTSTANDING),
        .FIFO_DEPTH          (512)
    ) u_a_reader (
        .clk_i              (clk_i),
        .rst_i              (rst_i),
        .clear_i            (clear_i),
        .start_i            (a_start),
        .busy_o             (a_reader_busy),
        .done_o             (a_reader_done),
        .error_sticky_o     (a_reader_error),
        .base_addr_i        (a_dram_base_q),
        .row_count_i        (a_rows_q),
        .row_bytes_i        ({16'd0, a_row_bytes_q}),
        .row_stride_bytes_i (a_dram_stride_q),
        .m_axi_arid_o       (m_axi_arid_o[A_DMA_IDX]),
        .m_axi_araddr_o     (m_axi_araddr_o[A_DMA_IDX]),
        .m_axi_arlen_o      (m_axi_arlen_o[A_DMA_IDX]),
        .m_axi_arsize_o     (m_axi_arsize_o[A_DMA_IDX]),
        .m_axi_arburst_o    (m_axi_arburst_o[A_DMA_IDX]),
        .m_axi_arvalid_o    (m_axi_arvalid_o[A_DMA_IDX]),
        .m_axi_arready_i    (m_axi_arready_i[A_DMA_IDX]),
        .m_axi_rid_i        (m_axi_rid_i[A_DMA_IDX]),
        .m_axi_rdata_i      (m_axi_rdata_i[A_DMA_IDX]),
        .m_axi_rresp_i      (m_axi_rresp_i[A_DMA_IDX]),
        .m_axi_rlast_i      (m_axi_rlast_i[A_DMA_IDX]),
        .m_axi_rvalid_i     (m_axi_rvalid_i[A_DMA_IDX]),
        .m_axi_rready_o     (m_axi_rready_o[A_DMA_IDX]),
        .stream_valid_o     (a_dma_valid),
        .stream_ready_i     (a_dma_ready),
        .stream_data_o      (a_dma_data),
        .stream_keep_o      (a_dma_keep),
        .stream_last_o      (a_dma_last)
    );

    axi_aligned_read_stream #(
        .AXI_ID_WIDTH        (AXI_ID_WIDTH),
        .AXI_ADDR_WIDTH      (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH      (AXI_DATA_WIDTH),
        .AXI_MAX_BURST_BEATS (AXI_MAX_BURST_BEATS),
        .AXI_RD_OUTSTANDING  (AXI_RD_OUTSTANDING),
        .FIFO_DEPTH          (512)
    ) u_w_reader (
        .clk_i              (clk_i),
        .rst_i              (rst_i),
        .clear_i            (clear_i),
        .start_i            (w_start),
        .busy_o             (w_reader_busy),
        .done_o             (w_reader_done),
        .error_sticky_o     (w_reader_error),
        .base_addr_i        (w_dram_base_q),
        .row_count_i        (16'd1),
        .row_bytes_i        (w_total_bytes_q),
        .row_stride_bytes_i (w_total_bytes_q),
        .m_axi_arid_o       (m_axi_arid_o[W_DMA_IDX]),
        .m_axi_araddr_o     (m_axi_araddr_o[W_DMA_IDX]),
        .m_axi_arlen_o      (m_axi_arlen_o[W_DMA_IDX]),
        .m_axi_arsize_o     (m_axi_arsize_o[W_DMA_IDX]),
        .m_axi_arburst_o    (m_axi_arburst_o[W_DMA_IDX]),
        .m_axi_arvalid_o    (m_axi_arvalid_o[W_DMA_IDX]),
        .m_axi_arready_i    (m_axi_arready_i[W_DMA_IDX]),
        .m_axi_rid_i        (m_axi_rid_i[W_DMA_IDX]),
        .m_axi_rdata_i      (m_axi_rdata_i[W_DMA_IDX]),
        .m_axi_rresp_i      (m_axi_rresp_i[W_DMA_IDX]),
        .m_axi_rlast_i      (m_axi_rlast_i[W_DMA_IDX]),
        .m_axi_rvalid_i     (m_axi_rvalid_i[W_DMA_IDX]),
        .m_axi_rready_o     (m_axi_rready_o[W_DMA_IDX]),
        .stream_valid_o     (w_dma_valid),
        .stream_ready_i     (w_dma_ready),
        .stream_data_o      (w_dma_data),
        .stream_keep_o      (w_dma_keep),
        .stream_last_o      (w_dma_last)
    );

    axi_aligned_read_stream #(
        .AXI_ID_WIDTH        (AXI_ID_WIDTH),
        .AXI_ADDR_WIDTH      (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH      (AXI_DATA_WIDTH),
        .AXI_MAX_BURST_BEATS (AXI_MAX_BURST_BEATS),
        .AXI_RD_OUTSTANDING  (AXI_RD_OUTSTANDING),
        .FIFO_DEPTH          (256)
    ) u_meta_reader (
        .clk_i              (clk_i),
        .rst_i              (rst_i),
        .clear_i            (clear_i),
        .start_i            (meta_start),
        .busy_o             (meta_reader_busy),
        .done_o             (meta_reader_done),
        .error_sticky_o     (meta_reader_error),
        .base_addr_i        (meta_dram_base_q),
        .row_count_i        (16'd1),
        .row_bytes_i        (meta_byte_count_q),
        .row_stride_bytes_i (32'd16),
        .m_axi_arid_o       (m_axi_arid_o[OC_DMA_IDX]),
        .m_axi_araddr_o     (m_axi_araddr_o[OC_DMA_IDX]),
        .m_axi_arlen_o      (m_axi_arlen_o[OC_DMA_IDX]),
        .m_axi_arsize_o     (m_axi_arsize_o[OC_DMA_IDX]),
        .m_axi_arburst_o    (m_axi_arburst_o[OC_DMA_IDX]),
        .m_axi_arvalid_o    (m_axi_arvalid_o[OC_DMA_IDX]),
        .m_axi_arready_i    (m_axi_arready_i[OC_DMA_IDX]),
        .m_axi_rid_i        (m_axi_rid_i[OC_DMA_IDX]),
        .m_axi_rdata_i      (m_axi_rdata_i[OC_DMA_IDX]),
        .m_axi_rresp_i      (m_axi_rresp_i[OC_DMA_IDX]),
        .m_axi_rlast_i      (m_axi_rlast_i[OC_DMA_IDX]),
        .m_axi_rvalid_i     (m_axi_rvalid_i[OC_DMA_IDX]),
        .m_axi_rready_o     (m_axi_rready_o[OC_DMA_IDX]),
        .stream_valid_o     (meta_dma_valid),
        .stream_ready_i     (meta_dma_ready),
        .stream_data_o      (meta_dma_data),
        .stream_keep_o      (meta_dma_keep),
        .stream_last_o      (meta_dma_last)
    );

    Versa_P_core #(
        .BANK_DEPTH_WORDS    (BANK_DEPTH_WORDS),
        .A_BANK_DEPTH_WORDS  (A_BANK_DEPTH_WORDS),
        .W_BANK_DEPTH_WORDS  (W_BANK_DEPTH_WORDS),
        .O_BANK_DEPTH_WORDS  (O_BANK_DEPTH_WORDS),
        .META_WORD128_DEPTH  (META_WORD128_DEPTH),
        .FIFO_DEPTH          (FIFO_DEPTH),
        .O_SPLIT_BANK_HALVES (O_SPLIT_BANK_HALVES),
        .DISABLE_ERROR_CHECKS(DISABLE_ERROR_CHECKS)
    ) u_core (
        .clk_i                         (clk_i),
        .rst_i                         (rst_i),
        .clear_error_i                 (clear_i),
        .cmd_valid_i                   (selected_valid),
        .cmd_ready_o                   (core_cmd_ready),
        .cmd_opcode_i                  (selected_cmd),
        .cmd_a_bank_i                  (core_cmd_a_bank),
        .cmd_w_bank_i                  (core_cmd_w_bank),
        .cmd_o_bank_i                  (core_cmd_o_bank),
        .active_w_bank_i               (active_w_bank_i),
        .co_done_i                     (1'b0),
        .meta_done_i                   (loader_mvin_done_o[OC_DMA_IDX]),
        .mvout_done_i                  (1'b0),
        .a_cfg_base_word_addr_i        (a_base_word_q),
        .a_cfg_row_count_i             (a_rows_q),
        .a_cfg_row_bytes_i             (a_row_bytes_q),
        .a_cfg_row_stride_words_i      (a_stride_words_q),
        .a_cfg_u8_minus_128_en_i       (a_u8_minus_128_q),
        .a_dma_valid_i                 (a_dma_valid),
        .a_dma_ready_o                 (a_dma_ready),
        .a_dma_data_i                  (a_dma_data),
        .a_dma_keep_i                  (a_dma_keep),
        .a_dma_last_i                  (a_dma_last),
        .w_cfg_base_word_addr_i        (w_base_word_q),
        .w_cfg_k_loaded_i              (w_k_loaded_q),
        .w_cfg_n_loaded_i              (w_n_loaded_q),
        .w_dma_valid_i                 (w_dma_valid),
        .w_dma_ready_o                 (w_dma_ready),
        .w_dma_data_i                  (w_dma_data),
        .w_dma_keep_i                  (w_dma_keep),
        .w_dma_last_i                  (w_dma_last),
        .meta_mvin_valid_i             (meta_dma_valid),
        .meta_mvin_ready_o             (meta_dma_ready),
        .meta_mvin_word128_addr_i      (meta_word_addr_q),
        .meta_mvin_data_i              (meta_dma_data),
        .meta_mvin_keep_i              (meta_dma_keep),
        .co_wr_start_i                 (1'b0),
        .co_wr_base_word_addr_i        ('0),
        .co_accumulate_en_i            (gemm_accumulate_en_q),
        .co_add_bias_en_i              (gemm_add_bias_en_q),
        .co_bias_addr_i                (META_BYTE_ADDR_W'(gemm_bias_offset_bytes_q)),
        .gemm_mode_i                   (gemm_mode_q),
        .mvout_req_word_addr_i         (mvout_req_word_addr_q),
        .mvout_m_count_i               (mvout_m_count_q),
        .mvout_n_count_i               (mvout_n_count_q),
        .mvout_dram_base_addr_i        (mvout_dram_base_addr_q),
        .mvout_dram_row_stride_bytes_i (mvout_dram_row_stride_bytes_q),
        .mvout_mode_i                  (mvout_mode_q),
        .mvout_per_channel_scale_i     (mvout_per_channel_scale_q),
        .mvout_tensor_scale_q8_24_i    (mvout_tensor_scale_q8_24_q),
        .mvout_scale_addr_i            (mvout_scale_addr_q),
        .attention_qk_token_count_i       (attention_qk_token_count_q),
        .attention_qk_gamma16_fix_i       (attention_qk_gamma16_fix_q),
        .attention_qk_mask_en_i           (attention_qk_mask_en_q),
        .attention_qk_output_base_word_addr_i ('0),
        .attention_qk_output_dram_base_addr_i (attention_qk_output_dram_base_q),
        .attention_qk_q_block_start_i         (attention_qk_q_block_start_q),
        .attention_qk_q_block_count_m1_i      (attention_qk_q_block_count_m1_q),
        .mvout_dma_valid_o             (mvout_dma_valid_o),
        .mvout_dma_ready_i             (mvout_dma_ready_i),
        .mvout_dma_data_o              (mvout_dma_data_o),
        .mvout_dma_keep_o              (mvout_dma_keep_o),
        .mvout_dma_addr_o              (mvout_dma_addr_o),
        .mvout_dma_last_o              (mvout_dma_last_o),
        .mvout_fifo_almost_full_o      (mvout_fifo_almost_full),
        .attention_qk_done_o              (core_attention_qk_done),
        .a_busy_o                      (core_a_busy),
        .a0_busy_o                     (core_a0_busy),
        .a1_busy_o                     (core_a1_busy),
        .w0_busy_o                     (core_w0_busy),
        .w1_busy_o                     (core_w1_busy),
        .co_busy_o                     (core_co_busy),
        .co0_busy_o                    (core_co0_busy),
        .co1_busy_o                    (core_co1_busy),
        .busy_o                        (core_busy_raw),
        .error_sticky_o                (core_error_sticky_raw),
        .last_error_o                  (core_last_error_raw),
        .command_accepted_count_o      (command_accepted_count_o),
        .conflict_count_o              (conflict_count_o),
        .gemm_error_sticky_o           (),
        .gemm_last_error_o             ()
    );

    wire unused_direct_dma_backend = ^cfg_mvout_sram_stride_i ^ meta_dma_last ^
                                     a_reader_done ^ w_reader_done ^
                                     mvout_fifo_almost_full;
endmodule

`default_nettype wire
