`default_nettype none

import npu_spm_pkg::*;

// Standalone existing-DMA integration boundary for the Versa_P core.
//
// This wrapper composes:
//   * dma_core_adapter: existing inst_ctrl/DMA write beats -> Versa_P_core
//   * dma_mvout_bridge: Versa_P_core mvout stream -> existing DMA SPM read
//
// W bank ownership is supplied by inst_ctrl. The bridge must not switch the
// active W bank implicitly.
module dma_core_bridge #(
    parameter int RF_DATA_WIDTH      = 64,
    parameter int DMA_NUM            = 3,
    parameter int SPM_DATA_WIDTH     = 128,
    parameter int ACC_DATA_WIDTH     = 512,
    parameter int BANK_DEPTH_WORDS   = 16384,
    parameter int A_BANK_DEPTH_WORDS = 12288,
    parameter int W_BANK_DEPTH_WORDS = 12288,
    parameter int O_BANK_DEPTH_WORDS = 16384,
    parameter int READ_LATENCY       = 4,
    parameter int META_WORD128_DEPTH = 2048,
    parameter int FIFO_DEPTH         = 8,
    parameter int MVOUT_PREFILL_LEVEL = FIFO_DEPTH,
    parameter int A_DMA_IDX          = 0,
    parameter int W_DMA_IDX          = 1,
    parameter int OC_DMA_IDX         = 2,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0,
    parameter bit O_SPLIT_BANK_HALVES = 1'b0,
    localparam int LEGACY_HALF_W     = RF_DATA_WIDTH / 2,
    localparam int LEGACY_QUARTER_W  = RF_DATA_WIDTH / 4,
    localparam int MVOUT_COUNT_W     = $clog2(FIFO_DEPTH + 1),
    localparam int MVOUT_PREFILL_TARGET =
        (MVOUT_PREFILL_LEVEL <= 0) ? 1 :
        ((MVOUT_PREFILL_LEVEL > FIFO_DEPTH) ? FIFO_DEPTH : MVOUT_PREFILL_LEVEL)
) (
    input  wire logic                                      clk_i,
    input  wire logic                                      rst_n_i,
    input  wire logic                                      clear_error_i,

    input  wire logic [DMA_NUM-1:0]                        dma_mvin_req_en_vec_i,
    input  wire logic [DMA_NUM-1:0]                        dma_mvout_req_en_vec_i,
    output logic [DMA_NUM-1:0]                              dma_mvin_req_en_gated_vec_o,
    output logic [DMA_NUM-1:0]                              dma_mvout_req_en_gated_vec_o,
    input  wire logic                                      dma_mvout_consume_i,
    input  wire logic                                      sa_req_en_i,
    input  wire logic                                      active_w_bank_i,
    input  wire logic                                      mvin_w_bank_i,
    input  wire logic                                      cmd_o_bank_i,

    input  wire logic [DMA_NUM-1:0][LEGACY_HALF_W-1:0]     mvin_row_num_vec_i,
    input  wire logic [DMA_NUM-1:0][LEGACY_HALF_W-1:0]     mvin_col_num_vec_i,
    input  wire logic [DMA_NUM-1:0][LEGACY_HALF_W-1:0]     mvin_sram_addr_vec_i,
    input  wire logic [DMA_NUM-1:0][LEGACY_QUARTER_W-1:0]  cfg_mvin_sram_stride_vec_i,
    input  wire logic [DMA_NUM-1:0][LEGACY_HALF_W-1:0]     mvout_row_num_vec_i,
    input  wire logic [DMA_NUM-1:0][LEGACY_HALF_W-1:0]     mvout_col_num_vec_i,
    input  wire logic [DMA_NUM-1:0][LEGACY_HALF_W-1:0]     mvout_sram_addr_vec_i,
    input  wire logic [DMA_NUM-1:0][1:0]                   cfg_mvout_mode_vec_i,
    input  wire logic [DMA_NUM-1:0]                        cfg_mvout_per_channel_vec_i,
    input  wire logic [DMA_NUM-1:0][31:0]                  cfg_mvout_scale_param_vec_i,
    input  wire logic [DMA_NUM-1:0][LEGACY_HALF_W-1:0]     cfg_mvout_dram_stride_vec_i,
    input  wire logic [31:0]                               mvout_dram_base_addr_i,
    input  wire logic                                      co_accumulate_en_i,
    input  wire logic                                      co_add_bias_en_i,
    input  wire logic [31:0]                               co_bias_addr_i,

    input  wire logic [15:0]                               sa_input_a_row_num_i,
    input  wire logic [15:0]                               sa_input_a_col_num_i,
    input  wire logic [15:0]                               sa_input_b_col_num_i,
    input  wire logic [15:0]                               sa_input_b_row_num_i,
    input  wire logic                                      cfg_compute_asymmetric_activations_i,

    input  wire logic [DMA_NUM-1:0]                        dma_spm_wr_en_vec_i,
    input  wire logic [DMA_NUM-1:0][SPM_DATA_WIDTH-1:0]    dma_spm_din_vec_i,
    input  wire logic [DMA_NUM-1:0][SPM_DATA_WIDTH/8-1:0]  dma_spm_wr_mask_vec_i,
    input  wire logic [DMA_NUM-1:0]                        dma_spm_rd_en_vec_i,
    output logic [DMA_NUM-1:0][SPM_DATA_WIDTH-1:0]         dma_spm_dout_vec_o,

    input  wire logic [DMA_NUM-1:0]                        dma_acc_wr_en_vec_i,
    input  wire logic [DMA_NUM-1:0][ACC_DATA_WIDTH-1:0]    dma_acc_din_vec_i,
    input  wire logic [DMA_NUM-1:0][ACC_DATA_WIDTH/32-1:0] dma_acc_wr_mask_vec_i,

    output logic                                           active_w_bank_o,
    output logic                                           mvout_dma_valid_o,
    input  wire logic                                      mvout_dma_ready_i,
    output logic [127:0]                                   mvout_dma_data_o,
    output logic [15:0]                                    mvout_dma_keep_o,
    output logic [63:0]                                    mvout_dma_addr_o,
    output logic                                           mvout_dma_last_o,
    output logic                                           core_busy_o,
    output logic                                           core_sa_busy_o,
    output logic                                           core_sa_done_o,
    output logic                                           core_error_sticky_o,
    output npu_error_e                                     core_last_error_o,
    output logic [31:0]                                    command_accepted_count_o,
    output logic [31:0]                                    conflict_count_o,
    output logic                                           dma_backpressure_error_o,
    output logic                                           dma_command_drop_o,
    output logic                                           mvout_bridge_underflow_o,
    output logic [MVOUT_COUNT_W-1:0]                       mvout_bridge_level_o,
    output logic                                           mvin_a_done_o,
    output logic                                           mvin_w_done_o,
    output logic                                           mvin_meta_done_o
);
    logic rst;
    logic mvout_stream_valid;
    logic mvout_stream_ready;
    logic [127:0] mvout_stream_data;
    logic [15:0] mvout_stream_keep;
    logic [63:0] mvout_stream_addr;
    logic mvout_stream_last;
    logic [SPM_DATA_WIDTH-1:0] mvout_bridge_dout;
    logic bridge_empty;
    logic bridge_full;
    logic mvout_prefetch_pending;
    logic [MVOUT_COUNT_W-1:0] mvout_prefetch_target;
    logic core_error_sticky;
    npu_error_e core_last_error;

    assign rst = !rst_n_i;
    assign active_w_bank_o = active_w_bank_i;
    assign core_error_sticky_o = DISABLE_ERROR_CHECKS ? 1'b0 :
                                 (core_error_sticky || mvout_bridge_underflow_o);
    assign core_last_error_o = DISABLE_ERROR_CHECKS ? NPU_ERR_NONE :
                               (core_error_sticky ? core_last_error :
                                (mvout_bridge_underflow_o ? NPU_ERR_BANK_CONFLICT :
                                                            NPU_ERR_NONE));
    assign mvout_dma_valid_o = mvout_stream_valid;
    assign mvout_dma_data_o = mvout_stream_data;
    assign mvout_dma_keep_o = mvout_stream_keep;
    assign mvout_dma_addr_o = mvout_stream_addr;
    assign mvout_dma_last_o = mvout_stream_last;

    function automatic logic [15:0] plus_one16(input logic [LEGACY_HALF_W-1:0] value);
        plus_one16 = value[15:0] + 16'd1;
    endfunction

    function automatic logic [31:0] ceil_div16_32(input logic [15:0] value);
        ceil_div16_32 = (value + 16'd15) >> 4;
    endfunction

    function automatic logic [MVOUT_COUNT_W-1:0] mvout_prefill_target(
        input logic [LEGACY_HALF_W-1:0] rows_m1,
        input logic [LEGACY_HALF_W-1:0] cols_m1
    );
        logic [15:0] row_count;
        logic [15:0] col_count;
        logic [15:0] beats_per_row;
        logic [MVOUT_COUNT_W-1:0] row_clip;
        logic [MVOUT_COUNT_W-1:0] beats_clip;
        logic [2*MVOUT_COUNT_W-1:0] total_clip;
        logic [MVOUT_COUNT_W-1:0] target;
        begin
            row_count = plus_one16(rows_m1);
            col_count = plus_one16(cols_m1);
            beats_per_row = (col_count + 32'd3) >> 2;
            target = MVOUT_COUNT_W'(MVOUT_PREFILL_TARGET);
            row_clip = (row_count >= 16'(MVOUT_PREFILL_TARGET)) ?
                       target : MVOUT_COUNT_W'(row_count[MVOUT_COUNT_W-1:0]);
            beats_clip = (beats_per_row >= 16'(MVOUT_PREFILL_TARGET)) ?
                         target : MVOUT_COUNT_W'(beats_per_row[MVOUT_COUNT_W-1:0]);
            total_clip = row_clip * beats_clip;
            mvout_prefill_target = (total_clip >= (2*MVOUT_COUNT_W)'(MVOUT_PREFILL_TARGET)) ?
                                   target : MVOUT_COUNT_W'(total_clip[MVOUT_COUNT_W-1:0]);
            if (mvout_prefill_target == '0) begin
                mvout_prefill_target = MVOUT_COUNT_W'(1);
            end
        end
    endfunction

    assign mvout_prefetch_target = mvout_prefill_target(mvout_row_num_vec_i[OC_DMA_IDX],
                                                        mvout_col_num_vec_i[OC_DMA_IDX]);

    always_comb begin
        dma_spm_dout_vec_o = '0;
        dma_spm_dout_vec_o[OC_DMA_IDX] = mvout_bridge_dout;
    end

    dma_mvout_prefetch_gate #(
        .DMA_NUM       (DMA_NUM),
        .OC_DMA_IDX    (OC_DMA_IDX),
        .FIFO_DEPTH    (FIFO_DEPTH),
        .PREFILL_LEVEL (MVOUT_PREFILL_LEVEL)
    ) u_mvout_prefetch_gate (
        .clk_i                  (clk_i),
        .rst_i                  (rst),
        .clear_i                (clear_error_i),
        .dma_mvout_req_en_i     (dma_mvout_req_en_vec_i),
        .oc_prefetch_target_i   (mvout_prefetch_target),
        .oc_fifo_level_i        (mvout_bridge_level_o),
        .dma_mvout_req_en_o     (dma_mvout_req_en_gated_vec_o),
        .oc_pending_o           (mvout_prefetch_pending)
    );

    dma_core_adapter #(
        .RF_DATA_WIDTH      (RF_DATA_WIDTH),
        .DMA_NUM            (DMA_NUM),
        .SPM_DATA_WIDTH     (SPM_DATA_WIDTH),
        .ACC_DATA_WIDTH     (ACC_DATA_WIDTH),
        .BANK_DEPTH_WORDS   (BANK_DEPTH_WORDS),
        .A_BANK_DEPTH_WORDS (A_BANK_DEPTH_WORDS),
        .W_BANK_DEPTH_WORDS (W_BANK_DEPTH_WORDS),
        .O_BANK_DEPTH_WORDS (O_BANK_DEPTH_WORDS),
        .READ_LATENCY       (READ_LATENCY),
        .META_WORD128_DEPTH (META_WORD128_DEPTH),
        .FIFO_DEPTH         (FIFO_DEPTH),
        .O_SPLIT_BANK_HALVES(O_SPLIT_BANK_HALVES),
        .A_DMA_IDX          (A_DMA_IDX),
        .W_DMA_IDX          (W_DMA_IDX),
        .OC_DMA_IDX         (OC_DMA_IDX),
        .DISABLE_ERROR_CHECKS (DISABLE_ERROR_CHECKS)
    ) u_DMA_adapter (
        .clk_i                              (clk_i),
        .rst_n_i                            (rst_n_i),
        .clear_error_i                      (clear_error_i),
        .dma_mvin_req_en_vec_i              (dma_mvin_req_en_vec_i),
        .dma_mvout_req_en_vec_i             (dma_mvout_req_en_vec_i),
        .dma_mvin_req_en_gated_vec_o        (dma_mvin_req_en_gated_vec_o),
        .sa_req_en_i                        (sa_req_en_i),
        .active_w_bank_i                    (active_w_bank_i),
        .mvin_w_bank_i                      (mvin_w_bank_i),
        .cmd_o_bank_i                       (cmd_o_bank_i),
        .mvin_row_num_vec_i                 (mvin_row_num_vec_i),
        .mvin_col_num_vec_i                 (mvin_col_num_vec_i),
        .mvin_sram_addr_vec_i               (mvin_sram_addr_vec_i),
        .cfg_mvin_sram_stride_vec_i         (cfg_mvin_sram_stride_vec_i),
        .mvout_row_num_vec_i                (mvout_row_num_vec_i),
        .mvout_col_num_vec_i                (mvout_col_num_vec_i),
        .mvout_sram_addr_vec_i              (mvout_sram_addr_vec_i),
        .cfg_mvout_mode_vec_i               (cfg_mvout_mode_vec_i),
        .cfg_mvout_per_channel_vec_i        (cfg_mvout_per_channel_vec_i),
        .cfg_mvout_scale_param_vec_i        (cfg_mvout_scale_param_vec_i),
        .cfg_mvout_dram_stride_vec_i        (cfg_mvout_dram_stride_vec_i),
        .mvout_dram_base_addr_i             (mvout_dram_base_addr_i),
        .co_accumulate_en_i                 (co_accumulate_en_i),
        .co_add_bias_en_i                   (co_add_bias_en_i),
        .co_bias_addr_i                     (co_bias_addr_i),
        .sa_input_a_row_num_i               (sa_input_a_row_num_i),
        .sa_input_a_col_num_i               (sa_input_a_col_num_i),
        .sa_input_b_col_num_i               (sa_input_b_col_num_i),
        .sa_input_b_row_num_i               (sa_input_b_row_num_i),
        .cfg_compute_asymmetric_activations_i(cfg_compute_asymmetric_activations_i),
        .dma_spm_wr_en_vec_i                (dma_spm_wr_en_vec_i),
        .dma_spm_din_vec_i                  (dma_spm_din_vec_i),
        .dma_spm_wr_mask_vec_i              (dma_spm_wr_mask_vec_i),
        .dma_acc_wr_en_vec_i                (dma_acc_wr_en_vec_i),
        .dma_acc_din_vec_i                  (dma_acc_din_vec_i),
        .dma_acc_wr_mask_vec_i              (dma_acc_wr_mask_vec_i),
        .mvout_dma_valid_o                  (mvout_stream_valid),
        .mvout_dma_ready_i                  (mvout_dma_ready_i),
        .mvout_dma_data_o                   (mvout_stream_data),
        .mvout_dma_keep_o                   (mvout_stream_keep),
        .mvout_dma_addr_o                   (mvout_stream_addr),
        .mvout_dma_last_o                   (mvout_stream_last),
        .core_busy_o                      (core_busy_o),
        .core_sa_busy_o                   (core_sa_busy_o),
        .core_sa_done_o                   (core_sa_done_o),
        .core_error_sticky_o              (core_error_sticky),
        .core_last_error_o                (core_last_error),
        .command_accepted_count_o           (command_accepted_count_o),
        .conflict_count_o                   (conflict_count_o),
        .dma_backpressure_error_o        (dma_backpressure_error_o),
        .dma_command_drop_o              (dma_command_drop_o),
        .mvin_a_done_o                   (mvin_a_done_o),
        .mvin_w_done_o                   (mvin_w_done_o),
        .mvin_meta_done_o                (mvin_meta_done_o)
    );

    dma_mvout_bridge #(
        .DATA_WIDTH (SPM_DATA_WIDTH),
        .FIFO_DEPTH (FIFO_DEPTH)
    ) u_mvout_bridge (
        .clk_i                  (clk_i),
        .rst_i                  (rst),
        .clear_error_i          (clear_error_i),
        .core_mvout_valid_i   (mvout_stream_valid),
        .core_mvout_ready_o   (mvout_stream_ready),
        .core_mvout_data_i    (mvout_stream_data),
        .dma_spm_rd_en_i        (dma_spm_rd_en_vec_i[OC_DMA_IDX]),
        .dma_consume_i          (dma_mvout_consume_i),
        .dma_spm_dout_o         (mvout_bridge_dout),
        .fifo_empty_o           (bridge_empty),
        .fifo_full_o            (bridge_full),
        .fifo_level_o           (mvout_bridge_level_o),
        .underflow_sticky_o     (mvout_bridge_underflow_o)
    );
endmodule

`default_nettype wire
