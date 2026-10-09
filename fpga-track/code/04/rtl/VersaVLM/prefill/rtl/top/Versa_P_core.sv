`default_nettype none

import npu_spm_pkg::*;

// Phase-1 INT8 NPU top-level integration shell.
//
// This top intentionally exposes command and local DMA-stream style ports rather
// than the legacy AXI-Lite/AXI4 shell. It is the first integration boundary for
// the new A/W/C/O banked subsystem and keeps descriptor-driven address
// generation inside local blocks instead of routing CSR fields to URAM ports.
module Versa_P_core #(
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
    input  wire logic                  clk_i,
    input  wire logic                  rst_i,
    input  wire logic                  clear_error_i,

    input  wire logic                  cmd_valid_i,
    output logic                       cmd_ready_o,
    input  wire logic [2:0]            cmd_opcode_i,
    input  wire logic                  cmd_a_bank_i,
    input  wire logic                  cmd_w_bank_i,
    input  wire logic                  cmd_o_bank_i,
    input  wire logic                  active_w_bank_i,
    input  wire logic                  co_done_i,
    input  wire logic                  meta_done_i,
    input  wire logic                  mvout_done_i,

    input  wire logic [ADDR_W-1:0]     a_cfg_base_word_addr_i,
    input  wire logic [15:0]           a_cfg_row_count_i,
    input  wire logic [15:0]           a_cfg_row_bytes_i,
    input  wire logic [15:0]           a_cfg_row_stride_words_i,
    input  wire logic                  a_cfg_u8_minus_128_en_i,
    input  wire logic                  a_dma_valid_i,
    output logic                       a_dma_ready_o,
    input  wire logic [127:0]          a_dma_data_i,
    input  wire logic [15:0]           a_dma_keep_i,
    input  wire logic                  a_dma_last_i,

    input  wire logic [ADDR_W-1:0]     w_cfg_base_word_addr_i,
    input  wire logic [15:0]           w_cfg_k_loaded_i,
    input  wire logic [15:0]           w_cfg_n_loaded_i,
    input  wire logic                  w_dma_valid_i,
    output logic                       w_dma_ready_o,
    input  wire logic [127:0]          w_dma_data_i,
    input  wire logic [15:0]           w_dma_keep_i,
    input  wire logic                  w_dma_last_i,

    input  wire logic                  meta_mvin_valid_i,
    output logic                       meta_mvin_ready_o,
    input  wire logic [META_ADDR_W-1:0] meta_mvin_word128_addr_i,
    input  wire logic [127:0]          meta_mvin_data_i,
    input  wire logic [15:0]           meta_mvin_keep_i,

    input  wire logic                  co_wr_start_i,
    input  wire logic [ADDR_W-1:0]     co_wr_base_word_addr_i,
    input  wire logic                  co_accumulate_en_i,
    input  wire logic                  co_add_bias_en_i,
    input  wire logic [META_BYTE_ADDR_W-1:0] co_bias_addr_i,
    input  wire npu_mode_e             gemm_mode_i,

    input  wire logic [ADDR_W-1:0]     mvout_req_word_addr_i,
    input  wire logic [15:0]           mvout_m_count_i,
    input  wire logic [15:0]           mvout_n_count_i,
    input  wire logic [63:0]           mvout_dram_base_addr_i,
    input  wire logic [31:0]           mvout_dram_row_stride_bytes_i,
    input  wire npu_mvout_mode_e       mvout_mode_i,
    input  wire logic                  mvout_per_channel_scale_i,
    input  wire logic [31:0]           mvout_tensor_scale_q8_24_i,
    input  wire logic [META_BYTE_ADDR_W-1:0] mvout_scale_addr_i,
    input  wire logic [15:0]           attention_qk_token_count_i,
    input  wire logic [31:0]           attention_qk_gamma16_fix_i,
    input  wire logic                  attention_qk_mask_en_i,
    input  wire logic [ADDR_W-1:0]     attention_qk_output_base_word_addr_i,
    input  wire logic [31:0]           attention_qk_output_dram_base_addr_i,
    input  wire logic [4:0]            attention_qk_q_block_start_i,
    input  wire logic [4:0]            attention_qk_q_block_count_m1_i,
    output logic                       mvout_dma_valid_o,
    input  wire logic                  mvout_dma_ready_i,
    output logic [127:0]               mvout_dma_data_o,
    output logic [15:0]                mvout_dma_keep_o,
    output logic [63:0]                mvout_dma_addr_o,
    output logic                       mvout_dma_last_o,
    output logic                       mvout_fifo_almost_full_o,
    output logic                       attention_qk_done_o,

    output logic                       a_busy_o,
    output logic                       a0_busy_o,
    output logic                       a1_busy_o,
    output logic                       w0_busy_o,
    output logic                       w1_busy_o,
    output logic                       co_busy_o,
    output logic                       co0_busy_o,
    output logic                       co1_busy_o,
    output logic                       busy_o,
    output logic                       error_sticky_o,
    output npu_error_e                 last_error_o,
    output logic [31:0]                command_accepted_count_o,
    output logic [31:0]                conflict_count_o,
    output logic                       gemm_error_sticky_o,
    output npu_error_e                 gemm_last_error_o
);
    logic owner_conflict_sticky;
    npu_error_e owner_last_error;
    logic a_error_sticky;
    logic [7:0] a_error_code;
    logic w_error_sticky;
    logic [7:0] w_error_code;
    logic meta_conflict_sticky;
    npu_error_e meta_last_error;
    logic co_acc_error_sticky;
    npu_error_e co_acc_last_error;
    logic mvout_error_sticky;
    npu_error_e mvout_last_error;
    logic attention_qk_error_sticky;
    npu_error_e attention_qk_last_error;
    logic a_bank_conflict;
    logic w0_bank_conflict;
    logic w1_bank_conflict;
    logic co_bank_conflict;

    assign busy_o = a_busy_o || w0_busy_o || w1_busy_o || co_busy_o;
    assign error_sticky_o = DISABLE_ERROR_CHECKS ? 1'b0 :
                            (owner_conflict_sticky || a_error_sticky ||
                             w_error_sticky || meta_conflict_sticky ||
                             co_acc_error_sticky || gemm_error_sticky_o ||
                             mvout_error_sticky || attention_qk_error_sticky ||
                             a_bank_conflict ||
                             w0_bank_conflict || w1_bank_conflict ||
                             co_bank_conflict);

    always_comb begin
        last_error_o = NPU_ERR_NONE;
        if (DISABLE_ERROR_CHECKS) begin
            last_error_o = NPU_ERR_NONE;
        end else if (owner_conflict_sticky) begin
            last_error_o = owner_last_error;
        end else if (meta_conflict_sticky) begin
            last_error_o = meta_last_error;
        end else if (co_acc_error_sticky) begin
            last_error_o = co_acc_last_error;
        end else if (gemm_error_sticky_o) begin
            last_error_o = gemm_last_error_o;
        end else if (mvout_error_sticky) begin
            last_error_o = mvout_last_error;
        end else if (attention_qk_error_sticky) begin
            last_error_o = attention_qk_last_error;
        end else if (a_error_sticky || w_error_sticky ||
                     a_bank_conflict || w0_bank_conflict ||
                     w1_bank_conflict || co_bank_conflict) begin
            last_error_o = NPU_ERR_BANK_CONFLICT;
        end
    end

    npu_subsystem #(
        .BANK_DEPTH_WORDS   (BANK_DEPTH_WORDS),
        .A_BANK_DEPTH_WORDS (A_BANK_DEPTH_WORDS),
        .W_BANK_DEPTH_WORDS (W_BANK_DEPTH_WORDS),
        .O_BANK_DEPTH_WORDS (O_BANK_DEPTH_WORDS),
        .READ_LATENCY       (READ_LATENCY),
        .META_WORD128_DEPTH (META_WORD128_DEPTH),
        .FIFO_DEPTH         (FIFO_DEPTH),
        .O_SPLIT_BANK_HALVES(O_SPLIT_BANK_HALVES),
        .DISABLE_ERROR_CHECKS(DISABLE_ERROR_CHECKS)
    ) u_spm (
        .clk_i                         (clk_i),
        .rst_i                         (rst_i),
        .clear_error_i                 (clear_error_i),
        .cmd_valid_i                   (cmd_valid_i),
        .cmd_ready_o                   (cmd_ready_o),
        .cmd_opcode_i                  (cmd_opcode_i),
        .cmd_a_bank_i                  (cmd_a_bank_i),
        .cmd_w_bank_i                  (cmd_w_bank_i),
        .cmd_o_bank_i                  (cmd_o_bank_i),
        .active_w_bank_i               (active_w_bank_i),
        .co_done_i                     (co_done_i),
        .meta_done_i                   (meta_done_i),
        .mvout_done_i                  (mvout_done_i),
        .a_cfg_base_word_addr_i        (a_cfg_base_word_addr_i),
        .a_cfg_row_count_i             (a_cfg_row_count_i),
        .a_cfg_row_bytes_i             (a_cfg_row_bytes_i),
        .a_cfg_row_stride_words_i      (a_cfg_row_stride_words_i),
        .a_cfg_u8_minus_128_en_i       (a_cfg_u8_minus_128_en_i),
        .a_dma_valid_i                 (a_dma_valid_i),
        .a_dma_ready_o                 (a_dma_ready_o),
        .a_dma_data_i                  (a_dma_data_i),
        .a_dma_keep_i                  (a_dma_keep_i),
        .a_dma_last_i                  (a_dma_last_i),
        .a_rd_valid_i                  (1'b0),
        .a_rd_word_addr_i              ('0),
        .a_rsp_valid_o                 (),
        .a_rsp_data_o                  (),
        .w_cfg_base_word_addr_i        (w_cfg_base_word_addr_i),
        .w_cfg_k_loaded_i              (w_cfg_k_loaded_i),
        .w_cfg_n_loaded_i              (w_cfg_n_loaded_i),
        .w_dma_valid_i                 (w_dma_valid_i),
        .w_dma_ready_o                 (w_dma_ready_o),
        .w_dma_data_i                  (w_dma_data_i),
        .w_dma_keep_i                  (w_dma_keep_i),
        .w_dma_last_i                  (w_dma_last_i),
        .w_rd_valid_i                  (1'b0),
        .w_rd_word_addr_i              ('0),
        .w_rsp_valid_o                 (),
        .w_rsp_data_o                  (),
        .meta_mvin_valid_i             (meta_mvin_valid_i),
        .meta_mvin_ready_o             (meta_mvin_ready_o),
        .meta_mvin_word128_addr_i      (meta_mvin_word128_addr_i),
        .meta_mvin_data_i              (meta_mvin_data_i),
        .meta_mvin_keep_i              (meta_mvin_keep_i),
        .meta_read256_valid_i          (1'b0),
        .meta_read256_base_word128_i   ('0),
        .meta_read256_word_idx_i       ('0),
        .meta_read256_rsp_valid_o      (),
        .meta_read256_rsp_data_o       (),
        .co_wr_start_i                 (co_wr_start_i),
        .co_wr_base_word_addr_i        (co_wr_base_word_addr_i),
        .co_accumulate_en_i            (co_accumulate_en_i),
        .co_add_bias_en_i              (co_add_bias_en_i),
        .co_in_valid_i                 (1'b0),
        .co_in_ready_o                 (),
        .co_partial_data_i             ('0),
        .co_lane_mask_i                (8'hff),
        .co_bias_addr_i                (co_bias_addr_i),
        .gemm_mode_i                   (gemm_mode_i),
        .mvout_req_word_addr_i         (mvout_req_word_addr_i),
        .mvout_m_count_i               (mvout_m_count_i),
        .mvout_n_count_i               (mvout_n_count_i),
        .mvout_dram_base_addr_i        (mvout_dram_base_addr_i),
        .mvout_dram_row_stride_bytes_i (mvout_dram_row_stride_bytes_i),
        .mvout_mode_i                  (mvout_mode_i),
        .mvout_per_channel_scale_i     (mvout_per_channel_scale_i),
        .mvout_tensor_scale_q8_24_i    (mvout_tensor_scale_q8_24_i),
        .mvout_scale_addr_i            (mvout_scale_addr_i),
        .attention_qk_token_count_i       (attention_qk_token_count_i),
        .attention_qk_gamma16_fix_i       (attention_qk_gamma16_fix_i),
        .attention_qk_mask_en_i           (attention_qk_mask_en_i),
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
        .attention_qk_done_o              (attention_qk_done_o),
        .a_busy_o                      (a_busy_o),
        .a0_busy_o                     (a0_busy_o),
        .a1_busy_o                     (a1_busy_o),
        .w0_busy_o                     (w0_busy_o),
        .w1_busy_o                     (w1_busy_o),
        .co_busy_o                     (co_busy_o),
        .co0_busy_o                    (co0_busy_o),
        .co1_busy_o                    (co1_busy_o),
        .owner_conflict_sticky_o       (owner_conflict_sticky),
        .owner_last_error_o            (owner_last_error),
        .command_accepted_count_o      (command_accepted_count_o),
        .conflict_count_o              (conflict_count_o),
        .a_error_sticky_o              (a_error_sticky),
        .a_error_code_o                (a_error_code),
        .w_error_sticky_o              (w_error_sticky),
        .w_error_code_o                (w_error_code),
        .meta_conflict_sticky_o        (meta_conflict_sticky),
        .meta_last_error_o             (meta_last_error),
        .co_acc_error_sticky_o         (co_acc_error_sticky),
        .co_acc_last_error_o           (co_acc_last_error),
        .gemm_error_sticky_o           (gemm_error_sticky_o),
        .gemm_last_error_o             (gemm_last_error_o),
        .mvout_error_sticky_o          (mvout_error_sticky),
        .mvout_last_error_o            (mvout_last_error),
        .attention_qk_error_sticky_o      (attention_qk_error_sticky),
        .attention_qk_last_error_o        (attention_qk_last_error),
        .a_bank_conflict_o             (a_bank_conflict),
        .w0_bank_conflict_o            (w0_bank_conflict),
        .w1_bank_conflict_o            (w1_bank_conflict),
        .co_bank_conflict_o            (co_bank_conflict)
    );
endmodule

`default_nettype wire
