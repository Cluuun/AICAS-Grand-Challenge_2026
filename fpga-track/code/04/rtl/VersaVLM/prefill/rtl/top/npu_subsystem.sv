`default_nettype none

import npu_spm_pkg::*;

// Phase-1 SPM subsystem integration wrapper.
//
// This is the command-facing composition point for the owner shell and the
// local A/W/C/O datapath. It keeps command acceptance separate from DMA/data
// streaming so the old top-level control can wire one layer at a time.
module npu_subsystem #(
    parameter int BANK_DEPTH_WORDS   = 16384,
    parameter int A_BANK_DEPTH_WORDS = 12288,
    parameter int W_BANK_DEPTH_WORDS = 12288,
    parameter int O_BANK_DEPTH_WORDS = 16384,
    parameter int READ_LATENCY       = 4,
    parameter int META_WORD128_DEPTH = 2048,
    parameter int FIFO_DEPTH         = 8,
    parameter bit O_SPLIT_BANK_HALVES = 1'b0,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0,
    localparam int O_PHYS_DEPTH_WORDS = O_SPLIT_BANK_HALVES ? (2 * O_BANK_DEPTH_WORDS) : O_BANK_DEPTH_WORDS,
    localparam int MAX_AW_DEPTH_WORDS = (A_BANK_DEPTH_WORDS > W_BANK_DEPTH_WORDS) ? A_BANK_DEPTH_WORDS : W_BANK_DEPTH_WORDS,
    localparam int MAX_BANK_DEPTH_WORDS = (MAX_AW_DEPTH_WORDS > O_PHYS_DEPTH_WORDS) ? MAX_AW_DEPTH_WORDS : O_PHYS_DEPTH_WORDS,
    localparam int ADDR_W            = (MAX_BANK_DEPTH_WORDS <= 1) ? 1 : $clog2(MAX_BANK_DEPTH_WORDS),
    localparam int META_ADDR_W       = (META_WORD128_DEPTH <= 1) ? 1 : $clog2(META_WORD128_DEPTH),
    localparam int META_BYTE_ADDR_W  = META_ADDR_W + 4
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,
    input  wire logic              clear_error_i,

    input  wire logic              cmd_valid_i,
    output logic                   cmd_ready_o,
    input  wire logic [2:0]        cmd_opcode_i,
    input  wire logic              cmd_a_bank_i,
    input  wire logic              cmd_w_bank_i,
    input  wire logic              cmd_o_bank_i,
    input  wire logic              active_w_bank_i,
    input  wire logic              co_done_i,
    input  wire logic              meta_done_i,
    input  wire logic              mvout_done_i,

    input  wire logic [ADDR_W-1:0] a_cfg_base_word_addr_i,
    input  wire logic [15:0]       a_cfg_row_count_i,
    input  wire logic [15:0]       a_cfg_row_bytes_i,
    input  wire logic [15:0]       a_cfg_row_stride_words_i,
    input  wire logic              a_cfg_u8_minus_128_en_i,
    input  wire logic              a_dma_valid_i,
    output logic                   a_dma_ready_o,
    input  wire logic [127:0]      a_dma_data_i,
    input  wire logic [15:0]       a_dma_keep_i,
    input  wire logic              a_dma_last_i,
    input  wire logic              a_rd_valid_i,
    input  wire logic [ADDR_W-1:0] a_rd_word_addr_i,
    output logic                   a_rsp_valid_o,
    output logic [255:0]           a_rsp_data_o,

    input  wire logic [ADDR_W-1:0] w_cfg_base_word_addr_i,
    input  wire logic [15:0]       w_cfg_k_loaded_i,
    input  wire logic [15:0]       w_cfg_n_loaded_i,
    input  wire logic              w_dma_valid_i,
    output logic                   w_dma_ready_o,
    input  wire logic [127:0]      w_dma_data_i,
    input  wire logic [15:0]       w_dma_keep_i,
    input  wire logic              w_dma_last_i,
    input  wire logic              w_rd_valid_i,
    input  wire logic [ADDR_W-1:0] w_rd_word_addr_i,
    output logic                   w_rsp_valid_o,
    output logic [255:0]           w_rsp_data_o,

    input  wire logic              meta_mvin_valid_i,
    output logic                   meta_mvin_ready_o,
    input  wire logic [META_ADDR_W-1:0] meta_mvin_word128_addr_i,
    input  wire logic [127:0]      meta_mvin_data_i,
    input  wire logic [15:0]       meta_mvin_keep_i,
    input  wire logic              meta_read256_valid_i,
    input  wire logic [META_ADDR_W-1:0] meta_read256_base_word128_i,
    input  wire logic [META_ADDR_W-1:0] meta_read256_word_idx_i,
    output logic                   meta_read256_rsp_valid_o,
    output logic [255:0]           meta_read256_rsp_data_o,

    input  wire logic              co_wr_start_i,
    input  wire logic [ADDR_W-1:0] co_wr_base_word_addr_i,
    input  wire logic              co_accumulate_en_i,
    input  wire logic              co_add_bias_en_i,
    input  wire logic              co_in_valid_i,
    output logic                   co_in_ready_o,
    input  wire logic [255:0]      co_partial_data_i,
    input  wire logic [7:0]        co_lane_mask_i,
    input  wire logic [META_BYTE_ADDR_W-1:0] co_bias_addr_i,
    input  wire npu_mode_e         gemm_mode_i,

    input  wire logic [ADDR_W-1:0] mvout_req_word_addr_i,
    input  wire logic [15:0]       mvout_m_count_i,
    input  wire logic [15:0]       mvout_n_count_i,
    input  wire logic [63:0]       mvout_dram_base_addr_i,
    input  wire logic [31:0]       mvout_dram_row_stride_bytes_i,
    input  wire npu_mvout_mode_e   mvout_mode_i,
    input  wire logic              mvout_per_channel_scale_i,
    input  wire logic [31:0]       mvout_tensor_scale_q8_24_i,
    input  wire logic [META_BYTE_ADDR_W-1:0] mvout_scale_addr_i,

    input  wire logic [15:0]       attention_qk_token_count_i,
    input  wire logic [31:0]       attention_qk_gamma16_fix_i,
    input  wire logic              attention_qk_mask_en_i,
    input  wire logic [ADDR_W-1:0] attention_qk_output_base_word_addr_i,
    input  wire logic [31:0]       attention_qk_output_dram_base_addr_i,
    input  wire logic [4:0]        attention_qk_q_block_start_i,
    input  wire logic [4:0]        attention_qk_q_block_count_m1_i,

    output logic                   mvout_dma_valid_o,
    input  wire logic              mvout_dma_ready_i,
    output logic [127:0]           mvout_dma_data_o,
    output logic [15:0]            mvout_dma_keep_o,
    output logic [63:0]            mvout_dma_addr_o,
    output logic                   mvout_dma_last_o,
    output logic                   mvout_fifo_almost_full_o,

    output logic                   a_busy_o,
    output logic                   a0_busy_o,
    output logic                   a1_busy_o,
    output logic                   w0_busy_o,
    output logic                   w1_busy_o,
    output logic                   co_busy_o,
    output logic                   co0_busy_o,
    output logic                   co1_busy_o,
    output logic                   owner_conflict_sticky_o,
    output npu_error_e             owner_last_error_o,
    output logic [31:0]            command_accepted_count_o,
    output logic [31:0]            conflict_count_o,
    output logic                   a_error_sticky_o,
    output logic [7:0]             a_error_code_o,
    output logic                   w_error_sticky_o,
    output logic [7:0]             w_error_code_o,
    output logic                   meta_conflict_sticky_o,
    output npu_error_e             meta_last_error_o,
    output logic                   co_acc_error_sticky_o,
    output npu_error_e             co_acc_last_error_o,
    output logic                   gemm_error_sticky_o,
    output npu_error_e             gemm_last_error_o,
    output logic                   mvout_error_sticky_o,
    output npu_error_e             mvout_last_error_o,
    output logic                   attention_qk_done_o,
    output logic                   attention_qk_error_sticky_o,
    output npu_error_e             attention_qk_last_error_o,
    output logic                   a_bank_conflict_o,
    output logic                   w0_bank_conflict_o,
    output logic                   w1_bank_conflict_o,
    output logic                   co_bank_conflict_o
);
    localparam logic [2:0] CMD_META_MVIN = 3'd2;
    localparam logic [2:0] CMD_GEMM      = 3'd3;
    localparam logic [2:0] CMD_ATTENTION_QK = 3'd5;

    logic start_mvin_a;
    logic start_mvin_w;
    logic start_metadata_mvin;
    logic start_gemm;
    logic start_mvout;
    logic start_attention_qk;
    logic a_loader_busy;
    logic a_loader_done;
    logic w_loader_busy;
    logic w_loader_done;
    logic datapath_a_ready;
    logic datapath_w_ready;
    logic datapath_gemm_ready;
    logic datapath_mvout_ready;
    logic datapath_attention_qk_ready;
    logic command_datapath_ready;
    logic owner_cmd_valid;
    logic owner_cmd_ready;
    logic [ADDR_W-1:0] mvout_req_word_addr_q;
    logic [15:0] mvout_m_count_q;
    logic [15:0] mvout_n_count_q;
    logic [63:0] mvout_dram_base_addr_q;
    logic [31:0] mvout_dram_row_stride_bytes_q;
    npu_mvout_mode_e mvout_mode_q;
    logic mvout_per_channel_scale_q;
    logic [31:0] mvout_tensor_scale_q8_24_q;
    logic [META_BYTE_ADDR_W-1:0] mvout_scale_addr_q;
    logic mvin_a_bank_q;
    logic gemm_a_bank_q;
    logic mvin_w_bank_q;
    logic gemm_o_bank_q;
    npu_mode_e gemm_mode_q;
    logic mvout_o_bank_q;
    logic attention_qk_q_bank_q;
    logic attention_qk_k_bank_q;
    logic attention_qk_o_bank_q;
    logic datapath_mvout_done;
    logic datapath_attention_qk_done;

    assign attention_qk_done_o = 1'b0;
    logic gemm_busy;
    logic gemm_done;
    assign owner_cmd_valid = cmd_valid_i && command_datapath_ready;
    assign cmd_ready_o = owner_cmd_ready;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            mvout_req_word_addr_q <= '0;
            mvout_m_count_q <= '0;
            mvout_n_count_q <= '0;
            mvout_dram_base_addr_q <= '0;
            mvout_dram_row_stride_bytes_q <= '0;
            mvout_mode_q <= NPU_MVOUT_RAW_I32;
            mvout_per_channel_scale_q <= 1'b0;
            mvout_tensor_scale_q8_24_q <= '0;
            mvout_scale_addr_q <= '0;
            mvin_a_bank_q <= 1'b0;
            gemm_a_bank_q <= 1'b0;
            mvin_w_bank_q <= 1'b0;
            gemm_o_bank_q <= 1'b0;
            gemm_mode_q <= NPU_MODE_INT8;
            mvout_o_bank_q <= 1'b0;
            attention_qk_q_bank_q <= 1'b0;
            attention_qk_k_bank_q <= 1'b0;
            attention_qk_o_bank_q <= 1'b0;
        end
        else if (owner_cmd_valid && owner_cmd_ready) begin
            if (cmd_opcode_i == 3'd0) begin
                mvin_a_bank_q <= cmd_a_bank_i;
            end
            if (cmd_opcode_i == 3'd1) begin
                mvin_w_bank_q <= cmd_w_bank_i;
            end
            if (cmd_opcode_i == 3'd3) begin
                gemm_a_bank_q <= cmd_a_bank_i;
                gemm_o_bank_q <= cmd_o_bank_i;
                gemm_mode_q <= gemm_mode_i;
                if (gemm_mode_i == NPU_MODE_ATTENTION_QK) begin
                    attention_qk_q_bank_q <= cmd_a_bank_i;
                    attention_qk_k_bank_q <= cmd_w_bank_i;
                    attention_qk_o_bank_q <= cmd_o_bank_i;
                end
            end
            if (cmd_opcode_i == 3'd4) begin
                mvout_o_bank_q <= cmd_o_bank_i;
                mvout_req_word_addr_q <= mvout_req_word_addr_i;
                mvout_m_count_q <= mvout_m_count_i;
                mvout_n_count_q <= mvout_n_count_i;
                mvout_dram_base_addr_q <= mvout_dram_base_addr_i;
                mvout_dram_row_stride_bytes_q <= mvout_dram_row_stride_bytes_i;
                mvout_mode_q <= mvout_mode_i;
                mvout_per_channel_scale_q <= mvout_per_channel_scale_i;
                mvout_tensor_scale_q8_24_q <= mvout_tensor_scale_q8_24_i;
                mvout_scale_addr_q <= mvout_scale_addr_i;
            end
            if (cmd_opcode_i == 3'd5) begin
                attention_qk_q_bank_q <= cmd_a_bank_i;
                attention_qk_k_bank_q <= cmd_w_bank_i;
                attention_qk_o_bank_q <= cmd_o_bank_i;
            end
        end
    end

    always_comb begin
        command_datapath_ready = 1'b1;
        unique case (cmd_opcode_i)
            3'd0: command_datapath_ready = datapath_a_ready;
            3'd1: command_datapath_ready = datapath_w_ready;
            3'd2: command_datapath_ready = meta_mvin_ready_o;
            3'd3: command_datapath_ready = datapath_gemm_ready;
            3'd4: command_datapath_ready = datapath_mvout_ready;
            3'd5: command_datapath_ready = 1'b0;
            default: command_datapath_ready = 1'b1;
        endcase
    end

    bank_owner u_owner (
        .clk_i                    (clk_i),
        .rst_i                    (rst_i),
        .clear_error_i            (clear_error_i),
        .cmd_valid_i              (owner_cmd_valid),
        .cmd_ready_o              (owner_cmd_ready),
        .cmd_opcode_i             (cmd_opcode_i),
        .cmd_a_bank_i             (cmd_a_bank_i),
        .cmd_w_bank_i             (cmd_w_bank_i),
        .cmd_o_bank_i             (cmd_o_bank_i),
        .cmd_gemm_attention_qk_mode_i ((cmd_opcode_i == CMD_GEMM) &&
                                       (gemm_mode_i == NPU_MODE_ATTENTION_QK)),
        .active_w_bank_i          (active_w_bank_i),
        .mvin_a_done_i            (a_loader_done),
        .w_done_i                 (w_loader_done),
        .meta_done_i              (meta_done_i),
        .gemm_done_i              (gemm_done),
        .mvout_done_i             (mvout_done_i || datapath_mvout_done),
        .attention_qk_done_i         (1'b0),
        .co_done_i                (co_done_i),
        .start_mvin_a_o           (start_mvin_a),
        .start_mvin_w_o           (start_mvin_w),
        .start_metadata_mvin_o    (start_metadata_mvin),
        .start_gemm_o             (start_gemm),
        .start_mvout_o            (start_mvout),
        .start_attention_qk_o        (start_attention_qk),
        .a_busy_o                 (a_busy_o),
        .a0_busy_o                (a0_busy_o),
        .a1_busy_o                (a1_busy_o),
        .w0_busy_o                (w0_busy_o),
        .w1_busy_o                (w1_busy_o),
        .co_busy_o                (co_busy_o),
        .co0_busy_o               (co0_busy_o),
        .co1_busy_o               (co1_busy_o),
        .conflict_sticky_o        (owner_conflict_sticky_o),
        .last_error_o             (owner_last_error_o),
        .command_accepted_count_o (command_accepted_count_o),
        .conflict_count_o         (conflict_count_o)
    );

    npu_datapath #(
        .BANK_DEPTH_WORDS   (BANK_DEPTH_WORDS),
        .A_BANK_DEPTH_WORDS (A_BANK_DEPTH_WORDS),
        .W_BANK_DEPTH_WORDS (W_BANK_DEPTH_WORDS),
        .O_BANK_DEPTH_WORDS (O_BANK_DEPTH_WORDS),
        .READ_LATENCY       (READ_LATENCY),
        .META_WORD128_DEPTH (META_WORD128_DEPTH),
        .FIFO_DEPTH         (FIFO_DEPTH),
        .O_SPLIT_BANK_HALVES(O_SPLIT_BANK_HALVES),
        .DISABLE_ERROR_CHECKS(DISABLE_ERROR_CHECKS)
    ) u_datapath (
        .clk_i                         (clk_i),
        .rst_i                         (rst_i),
        .clear_error_i                 (clear_error_i),
        .a_cfg_start_i                 (start_mvin_a),
        .a_cfg_ready_o                 (datapath_a_ready),
        .a_cfg_base_word_addr_i        (a_cfg_base_word_addr_i),
        .a_cfg_row_count_i             (a_cfg_row_count_i),
        .a_cfg_row_bytes_i             (a_cfg_row_bytes_i),
        .a_cfg_row_stride_words_i      (a_cfg_row_stride_words_i),
        .a_cfg_selected_bank_i         (mvin_a_bank_q),
        .a_cfg_u8_minus_128_en_i       (a_cfg_u8_minus_128_en_i),
        .a_dma_valid_i                 (a_dma_valid_i),
        .a_dma_ready_o                 (a_dma_ready_o),
        .a_dma_data_i                  (a_dma_data_i),
        .a_dma_keep_i                  (a_dma_keep_i),
        .a_dma_last_i                  (a_dma_last_i),
        .a_rd_valid_i                  (a_rd_valid_i),
        .a_rd_word_addr_i              (a_rd_word_addr_i),
        .a_rsp_valid_o                 (a_rsp_valid_o),
        .a_rsp_data_o                  (a_rsp_data_o),
        .a_busy_o                      (a_loader_busy),
        .a_done_o                      (a_loader_done),
        .a_error_sticky_o              (a_error_sticky_o),
        .a_error_code_o                (a_error_code_o),
        .w_cfg_start_i                 (start_mvin_w),
        .w_cfg_ready_o                 (datapath_w_ready),
        .w_cfg_base_word_addr_i        (w_cfg_base_word_addr_i),
        .w_cfg_k_loaded_i              (w_cfg_k_loaded_i),
        .w_cfg_n_loaded_i              (w_cfg_n_loaded_i),
        .w_cfg_selected_bank_i         (mvin_w_bank_q),
        .active_w_bank_i               (active_w_bank_i),
        .w_dma_valid_i                 (w_dma_valid_i),
        .w_dma_ready_o                 (w_dma_ready_o),
        .w_dma_data_i                  (w_dma_data_i),
        .w_dma_keep_i                  (w_dma_keep_i),
        .w_dma_last_i                  (w_dma_last_i),
        .w_rd_valid_i                  (w_rd_valid_i),
        .w_rd_word_addr_i              (w_rd_word_addr_i),
        .w_rsp_valid_o                 (w_rsp_valid_o),
        .w_rsp_data_o                  (w_rsp_data_o),
        .w_busy_o                      (w_loader_busy),
        .w_done_o                      (w_loader_done),
        .w_error_sticky_o              (w_error_sticky_o),
        .w_error_code_o                (w_error_code_o),
        .gemm_cfg_start_i              (start_gemm),
        .gemm_cfg_ready_o              (datapath_gemm_ready),
        .gemm_a_base_word_addr_i       (a_cfg_base_word_addr_i),
        .gemm_a_k_word_offset_i        ('0),
        .gemm_a_row_stride_words_i     (a_cfg_row_stride_words_i[ADDR_W-1:0]),
        .gemm_w_base_word_addr_i       (w_cfg_base_word_addr_i),
        .gemm_a_bank_i                 (gemm_a_bank_q),
        .gemm_o_bank_i                 (gemm_o_bank_q),
        .gemm_mode_i                   (gemm_mode_q),
        .gemm_m_count_i                (a_cfg_row_count_i),
        .gemm_n_count_i                (w_cfg_n_loaded_i),
        .gemm_k_count_i                (w_cfg_k_loaded_i[12:0]),
        .gemm_busy_o                   (gemm_busy),
        .gemm_done_o                   (gemm_done),
        .gemm_error_sticky_o           (gemm_error_sticky_o),
        .gemm_last_error_o             (gemm_last_error_o),
        .meta_mvin_valid_i             (meta_mvin_valid_i && (co_busy_o || start_metadata_mvin)),
        .meta_mvin_ready_o             (meta_mvin_ready_o),
        .meta_mvin_word128_addr_i      (meta_mvin_word128_addr_i),
        .meta_mvin_data_i              (meta_mvin_data_i),
        .meta_mvin_keep_i              (meta_mvin_keep_i),
        .meta_read256_valid_i          (meta_read256_valid_i),
        .meta_read256_base_word128_i   (meta_read256_base_word128_i),
        .meta_read256_word_idx_i       (meta_read256_word_idx_i),
        .meta_read256_rsp_valid_o      (meta_read256_rsp_valid_o),
        .meta_read256_rsp_data_o       (meta_read256_rsp_data_o),
        .meta_conflict_sticky_o        (meta_conflict_sticky_o),
        .meta_last_error_o             (meta_last_error_o),
        .co_wr_start_i                 (co_wr_start_i),
        .co_wr_base_word_addr_i        (co_wr_base_word_addr_i),
        .co_accumulate_en_i            (co_accumulate_en_i),
        .co_add_bias_en_i              (co_add_bias_en_i),
        .co_in_valid_i                 (co_in_valid_i),
        .co_in_ready_o                 (co_in_ready_o),
        .co_partial_data_i             (co_partial_data_i),
        .co_lane_mask_i                (co_lane_mask_i),
        .co_bias_addr_i                (co_bias_addr_i),
        .co_acc_error_sticky_o         (co_acc_error_sticky_o),
        .co_acc_last_error_o           (co_acc_last_error_o),
        .mvout_req_valid_i             (start_mvout),
        .mvout_req_ready_o             (datapath_mvout_ready),
        .mvout_req_word_addr_i         (mvout_req_word_addr_q),
        .mvout_o_bank_i                (mvout_o_bank_q),
        .mvout_m_count_i               (mvout_m_count_q),
        .mvout_n_count_i               (mvout_n_count_q),
        .mvout_dram_base_addr_i        (mvout_dram_base_addr_q),
        .mvout_dram_row_stride_bytes_i (mvout_dram_row_stride_bytes_q),
        .mvout_mode_i                  (mvout_mode_q),
        .mvout_per_channel_scale_i     (mvout_per_channel_scale_q),
        .mvout_tensor_scale_q8_24_i    (mvout_tensor_scale_q8_24_q),
        .mvout_scale_addr_i            (mvout_scale_addr_q),
        .attention_qk_req_valid_i         (start_attention_qk),
        .attention_qk_req_ready_o         (datapath_attention_qk_ready),
        .attention_qk_token_count_i       (attention_qk_token_count_i),
        .attention_qk_gamma16_fix_i       (attention_qk_gamma16_fix_i),
        .attention_qk_mask_en_i           (attention_qk_mask_en_i),
        .attention_qk_q_bank_i            (attention_qk_q_bank_q),
        .attention_qk_k_bank_i            (attention_qk_k_bank_q),
        .attention_qk_o_bank_i            (attention_qk_o_bank_q),
        .attention_qk_output_base_word_addr_i (attention_qk_output_base_word_addr_i),
        .attention_qk_output_dram_base_addr_i (attention_qk_output_dram_base_addr_i),
        .attention_qk_q_block_start_i         (attention_qk_q_block_start_i),
        .attention_qk_q_block_count_m1_i      (attention_qk_q_block_count_m1_i),
        .mvout_dma_valid_o             (mvout_dma_valid_o),
        .mvout_dma_ready_i             (mvout_dma_ready_i),
        .mvout_dma_data_o              (mvout_dma_data_o),
        .mvout_dma_keep_o              (mvout_dma_keep_o),
        .mvout_dma_addr_o              (mvout_dma_addr_o),
        .mvout_dma_last_o              (mvout_dma_last_o),
        .mvout_fifo_almost_full_o      (mvout_fifo_almost_full_o),
        .mvout_done_o                  (datapath_mvout_done),
        .mvout_error_sticky_o          (mvout_error_sticky_o),
        .mvout_last_error_o            (mvout_last_error_o),
        .attention_qk_done_o              (datapath_attention_qk_done),
        .attention_qk_error_sticky_o      (attention_qk_error_sticky_o),
        .attention_qk_last_error_o        (attention_qk_last_error_o),
        .a_bank_conflict_o             (a_bank_conflict_o),
        .w0_bank_conflict_o            (w0_bank_conflict_o),
        .w1_bank_conflict_o            (w1_bank_conflict_o),
        .co_bank_conflict_o            (co_bank_conflict_o)
    );

endmodule

`default_nettype wire
