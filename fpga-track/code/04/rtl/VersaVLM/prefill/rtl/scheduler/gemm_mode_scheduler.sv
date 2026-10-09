`default_nettype none

import npu_spm_pkg::*;

// Unified GEMM macro-mode scheduler front end.
//
// This module owns the QK shared-core launch/drain path. The regular INT8/PV
// tile walker remains in npu_datapath for this step and is muxed with these
// QK core requests there.
module gemm_mode_scheduler #(
    parameter int ADDR_W = 14,
    parameter int META_ADDR_W = 11,
    parameter int O_BANK_DEPTH_WORDS = 16384,
    parameter int GAMMA16_FRAC = 24,
    parameter bit DEBUG_PRINT = 1'b0
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,
    input  wire logic              clear_error_i,

    input  wire logic              qk_start_i,
    output logic                   qk_ready_o,
    output logic                   qk_busy_o,
    output logic                   qk_done_o,
    input  wire logic [15:0]       qk_token_count_i,
    input  wire logic [31:0]       qk_gamma16_fix_i,
    input  wire logic              qk_mask_en_i,
    input  wire logic [4:0]        qk_q_block_start_i,
    input  wire logic [4:0]        qk_q_block_count_m1_i,

    output logic                   core_cfg_start_o,
    input  wire logic              core_cfg_ready_i,
    output logic [ADDR_W-1:0]      core_a_base_word_addr_o,
    output logic [ADDR_W-1:0]      core_a_k_word_offset_o,
    output logic [ADDR_W-1:0]      core_a_row_stride_words_o,
    output logic [ADDR_W-1:0]      core_w_base_word_addr_o,
    output logic [5:0]             core_m_count_o,
    output logic [5:0]             core_n_count_o,
    output logic [12:0]            core_k_count_o,
    output npu_mode_e              core_mode_o,
    output logic                   core_snapshot_ready_o,
    input  wire logic              core_busy_i,
    input  wire logic              core_done_i,
    input  wire logic              core_error_sticky_i,
    input  wire npu_error_e        core_last_error_i,
    input  wire logic              core_co_valid_i,
    output logic                   core_co_ready_o,
    input  wire logic [255:0]      core_co_data_i,
    input  wire logic [7:0]        core_co_lane_mask_i,

    output logic                   qk_o_wr_req_valid_o,
    input  wire logic              qk_o_wr_req_ready_i,
    output logic [ADDR_W-1:0]      qk_o_wr_req_addr_o,
    output logic [255:0]           qk_o_wr_req_data_o,
    output logic [31:0]            qk_o_wr_req_byte_en_o,

    output logic                   qk_meta_wr_valid_o,
    input  wire logic              qk_meta_wr_ready_i,
    output logic [META_ADDR_W-1:0] qk_meta_wr_addr_o,
    output logic [127:0]           qk_meta_wr_data_o,
    output logic [15:0]            qk_meta_wr_keep_o,

    output logic                   qk_m_global_wr_valid_o,
    output logic [8:0]             qk_m_global_wr_row_o,
    output logic signed [15:0]     qk_m_global_wr_data_o,

    output logic                   qk_error_sticky_o,
    output npu_error_e             qk_last_error_o
);

    attention_qk_scheduler #(
        .ADDR_W             (ADDR_W),
        .META_ADDR_W        (META_ADDR_W),
        .O_BANK_DEPTH_WORDS (O_BANK_DEPTH_WORDS),
        .GAMMA16_FRAC       (GAMMA16_FRAC),
        .DEBUG_PRINT        (DEBUG_PRINT)
    ) u_qk_scheduler (
        .clk_i                       (clk_i),
        .rst_i                       (rst_i),
        .clear_error_i               (clear_error_i),
        .start_i                     (qk_start_i),
        .ready_o                     (qk_ready_o),
        .busy_o                      (qk_busy_o),
        .done_o                      (qk_done_o),
        .token_count_i               (qk_token_count_i),
        .gamma16_fix_i               (qk_gamma16_fix_i),
        .mask_en_i                   (qk_mask_en_i),
        .q_block_start_i             (qk_q_block_start_i),
        .q_block_count_m1_i          (qk_q_block_count_m1_i),
        .core_cfg_start_o            (core_cfg_start_o),
        .core_cfg_ready_i            (core_cfg_ready_i),
        .core_a_base_word_addr_o     (core_a_base_word_addr_o),
        .core_a_k_word_offset_o      (core_a_k_word_offset_o),
        .core_a_row_stride_words_o   (core_a_row_stride_words_o),
        .core_w_base_word_addr_o     (core_w_base_word_addr_o),
        .core_m_count_o              (core_m_count_o),
        .core_n_count_o              (core_n_count_o),
        .core_k_count_o              (core_k_count_o),
        .core_mode_o                 (core_mode_o),
        .core_snapshot_ready_o       (core_snapshot_ready_o),
        .core_busy_i                 (core_busy_i),
        .core_done_i                 (core_done_i),
        .core_error_sticky_i         (core_error_sticky_i),
        .core_last_error_i           (core_last_error_i),
        .core_co_valid_i             (core_co_valid_i),
        .core_co_ready_o             (core_co_ready_o),
        .core_co_data_i              (core_co_data_i),
        .core_co_lane_mask_i         (core_co_lane_mask_i),
        .o0_wr_req_valid_o           (qk_o_wr_req_valid_o),
        .o0_wr_req_ready_i           (qk_o_wr_req_ready_i),
        .o0_wr_req_word_addr_o       (qk_o_wr_req_addr_o),
        .o0_wr_req_data_o            (qk_o_wr_req_data_o),
        .o0_wr_req_byte_en_o         (qk_o_wr_req_byte_en_o),
        .meta_wr_valid_o             (qk_meta_wr_valid_o),
        .meta_wr_ready_i             (qk_meta_wr_ready_i),
        .meta_wr_word128_addr_o      (qk_meta_wr_addr_o),
        .meta_wr_data_o              (qk_meta_wr_data_o),
        .meta_wr_keep_o              (qk_meta_wr_keep_o),
        .m_global_wr_valid_o         (qk_m_global_wr_valid_o),
        .m_global_wr_row_o           (qk_m_global_wr_row_o),
        .m_global_wr_data_o          (qk_m_global_wr_data_o),
        .error_sticky_o              (qk_error_sticky_o),
        .last_error_o                (qk_last_error_o)
    );

endmodule

`default_nettype wire
