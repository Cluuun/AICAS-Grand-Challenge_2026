`default_nettype none

import npu_spm_pkg::*;

// Bridges the existing inst_ctrl/DMA_TOP style control surface to the Versa_P
// local-stream NPU core.  It is intentionally a narrow integration boundary:
// existing DMA memory-port writes are converted into A/W/metadata streams, while
// Versa_P mvout remains exposed as a 128-bit stream for the next AXI S2MM
// integration step.
module dma_core_adapter #(
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
    parameter int A_DMA_IDX          = 0,
    parameter int W_DMA_IDX          = 1,
    parameter int OC_DMA_IDX         = 2,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0,
    parameter bit O_SPLIT_BANK_HALVES = 1'b0,
    localparam int LEGACY_HALF_W     = RF_DATA_WIDTH / 2,
    localparam int LEGACY_QUARTER_W  = RF_DATA_WIDTH / 4,
    localparam int O_PHYS_DEPTH_WORDS = O_SPLIT_BANK_HALVES ? (2 * O_BANK_DEPTH_WORDS) : O_BANK_DEPTH_WORDS,
    localparam int MAX_AW_DEPTH_WORDS = (A_BANK_DEPTH_WORDS > W_BANK_DEPTH_WORDS) ? A_BANK_DEPTH_WORDS : W_BANK_DEPTH_WORDS,
    localparam int MAX_BANK_DEPTH_WORDS = (MAX_AW_DEPTH_WORDS > O_PHYS_DEPTH_WORDS) ? MAX_AW_DEPTH_WORDS : O_PHYS_DEPTH_WORDS,
    localparam int ADDR_W            = (MAX_BANK_DEPTH_WORDS <= 1) ? 1 : $clog2(MAX_BANK_DEPTH_WORDS),
    localparam int META_ADDR_W       = (META_WORD128_DEPTH <= 1) ? 1 : $clog2(META_WORD128_DEPTH),
    localparam int META_BYTE_ADDR_W  = META_ADDR_W + 4
) (
    input  wire logic                                      clk_i,
    input  wire logic                                      rst_n_i,
    input  wire logic                                      clear_error_i,

    input  wire logic [DMA_NUM-1:0]                        dma_mvin_req_en_vec_i,
    input  wire logic [DMA_NUM-1:0]                        dma_mvout_req_en_vec_i,
    output logic [DMA_NUM-1:0]                              dma_mvin_req_en_gated_vec_o,
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
    input  wire logic [DMA_NUM-1:0]                        dma_acc_wr_en_vec_i,
    input  wire logic [DMA_NUM-1:0][ACC_DATA_WIDTH-1:0]    dma_acc_din_vec_i,
    input  wire logic [DMA_NUM-1:0][ACC_DATA_WIDTH/32-1:0] dma_acc_wr_mask_vec_i,

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
    output logic                                           mvin_a_done_o,
    output logic                                           mvin_w_done_o,
    output logic                                           mvin_meta_done_o
);
    localparam logic [2:0] CMD_MVIN_A    = 3'd0;
    localparam logic [2:0] CMD_MVIN_W    = 3'd1;
    localparam logic [2:0] CMD_META_MVIN = 3'd2;
    localparam logic [2:0] CMD_GEMM      = 3'd3;
    localparam logic [2:0] CMD_MVOUT     = 3'd4;

    logic rst;
    logic dma_cmd_valid;
    logic cmd_valid;
    logic cmd_ready;
    logic [2:0] cmd_opcode;
    logic cmd_w_bank;
    logic cmd_o_bank;
    logic current_dma_cmd_valid;
    logic [2:0] current_dma_cmd_opcode;
    logic current_dma_cmd_w_bank;
    logic current_dma_cmd_o_bank;
    logic dma_cmd_pending_q;
    logic [2:0] dma_cmd_pending_opcode_q;
    logic dma_cmd_pending_w_bank_q;
    logic dma_cmd_pending_o_bank_q;
    logic [ADDR_W-1:0] dma_cmd_pending_mvout_word_addr_q;
    logic [31:0] dma_cmd_pending_mvout_word_count_q;
    logic cmd_fire;
    logic cmd_from_pending;
    logic cmd_from_dma;
    logic dma_mvout_cmd_fire;
    logic [ADDR_W-1:0] dma_mvout_fire_word_addr;
    logic [31:0] dma_mvout_fire_word_count;

    logic [ADDR_W-1:0] a_base_word;
    logic [15:0]       a_rows;
    logic [15:0]       a_row_bytes;
    logic [15:0]       a_stride_words;
    logic              a_dma_valid;
    logic              a_dma_ready;
    logic [127:0]      a_dma_data;
    logic [15:0]       a_dma_keep;
    logic              a_dma_last;
    logic [31:0]       a_expected_beats;
    logic [31:0]       a_expected_beats_q;
    logic              a_req_pending_q;
    logic [31:0]       a_push_remaining;

    logic [ADDR_W-1:0] w_base_word;
    logic [15:0]       w_k_loaded;
    logic [15:0]       w_n_loaded;
    logic              w_dma_valid;
    logic              w_dma_ready;
    logic [127:0]      w_dma_data;
    logic [15:0]       w_dma_keep;
    logic              w_dma_last;
    logic [31:0]       w_expected_beats;
    logic [31:0]       w_expected_beats_q;
    logic              w_req_pending_q;
    logic [31:0]       w_push_remaining;

    logic              meta_mvin_valid;
    logic              meta_mvin_ready;
    logic [META_ADDR_W-1:0] meta_word_addr;
    logic [127:0]      meta_mvin_data;
    logic [15:0]       meta_mvin_keep;

    logic [ADDR_W-1:0] mvout_req_word_addr;
    logic [ADDR_W-1:0] mvout_dma_req_word_addr;
    logic [31:0]       mvout_dram_row_stride_bytes;
    npu_mvout_mode_e   mvout_mode;
    logic              mvout_fifo_almost_full;
    logic              a_busy;
    logic              w0_busy;
    logic              w1_busy;
    logic              co_busy;
    logic              core_busy;
    logic              core_error_sticky;
    npu_error_e        core_last_error;
    logic              gemm_error_sticky;
    npu_error_e        gemm_last_error;

    logic [31:0] a_beats_remaining_q;
    logic [31:0] w_beats_remaining_q;
    logic [META_ADDR_W-1:0] meta_word_addr_q;
    logic gemm_inflight_q;
    logic gemm_seen_core_busy_q;
    logic command_drop_q;
    logic backpressure_error_q;
    logic a_done_pending_q;
    logic a_seen_busy_q;
    logic w_done_pending_q;
    logic w_seen_busy_q;
    logic meta_done_pending_q;
    logic [31:0] meta_beats_remaining_q;
    logic meta_beat_fire;
    logic [31:0] meta_expected_beats;

    assign rst = !rst_n_i;

    function automatic logic [15:0] plus_one16(input logic [LEGACY_HALF_W-1:0] value);
        plus_one16 = value[15:0] + 16'd1;
    endfunction

    function automatic logic [15:0] ceil_div32_16(input logic [15:0] value);
        ceil_div32_16 = (value + 16'd31) >> 5;
    endfunction

    function automatic logic [15:0] max16(input logic [15:0] a, input logic [15:0] b);
        max16 = (a > b) ? a : b;
    endfunction

    function automatic logic [31:0] ceil_div16_32(input logic [15:0] value);
        ceil_div16_32 = (value + 16'd15) >> 4;
    endfunction

    function automatic logic [31:0] mvout_word_count(
        input logic [LEGACY_HALF_W-1:0] rows_m1,
        input logic [LEGACY_HALF_W-1:0] cols_m1
    );
        logic [31:0] elem_count;
        begin
            elem_count = {16'b0, plus_one16(rows_m1)} * {16'b0, plus_one16(cols_m1)};
            mvout_word_count = (elem_count + 32'd7) >> 3;
        end
    endfunction

    always_comb begin
        a_rows         = plus_one16(mvin_row_num_vec_i[A_DMA_IDX]);
        a_row_bytes    = plus_one16(mvin_col_num_vec_i[A_DMA_IDX]);
        a_base_word    = ADDR_W'(mvin_sram_addr_vec_i[A_DMA_IDX][ADDR_W-1:0] >> 1);
        a_stride_words = max16(ceil_div32_16(a_row_bytes),
                               ceil_div32_16(16'(cfg_mvin_sram_stride_vec_i[A_DMA_IDX])));
        a_expected_beats = {16'b0, a_rows} * ceil_div16_32(a_row_bytes);

        w_k_loaded = sa_input_b_row_num_i + 16'd1;
        w_n_loaded = sa_input_b_col_num_i + 16'd1;
        w_base_word = ADDR_W'(mvin_sram_addr_vec_i[W_DMA_IDX][ADDR_W-1:0] >> 1);
        w_expected_beats = {16'b0, plus_one16(mvin_row_num_vec_i[W_DMA_IDX])} *
                           ceil_div16_32(plus_one16(mvin_col_num_vec_i[W_DMA_IDX]));

        mvout_dma_req_word_addr = ADDR_W'(mvout_sram_addr_vec_i[OC_DMA_IDX][ADDR_W-1:0] >> 1);
        mvout_req_word_addr = mvout_dma_req_word_addr;
        mvout_dram_row_stride_bytes = cfg_mvout_dram_stride_vec_i[OC_DMA_IDX] << 2;
        mvout_mode = npu_mvout_mode_e'(cfg_mvout_mode_vec_i[OC_DMA_IDX]);

        current_dma_cmd_valid = |dma_mvin_req_en_vec_i || |dma_mvout_req_en_vec_i || sa_req_en_i;
        current_dma_cmd_opcode = CMD_MVOUT;
        current_dma_cmd_w_bank = 1'b0;
        current_dma_cmd_o_bank = 1'b0;
        if (dma_mvin_req_en_vec_i[A_DMA_IDX]) begin
            current_dma_cmd_opcode = CMD_MVIN_A;
        end
        else if (dma_mvin_req_en_vec_i[W_DMA_IDX]) begin
            current_dma_cmd_opcode = CMD_MVIN_W;
            current_dma_cmd_w_bank = mvin_w_bank_i;
        end
        else if (dma_mvin_req_en_vec_i[OC_DMA_IDX]) begin
            current_dma_cmd_opcode = CMD_META_MVIN;
        end
        else if (sa_req_en_i) begin
            current_dma_cmd_opcode = CMD_GEMM;
            current_dma_cmd_o_bank = cmd_o_bank_i;
        end
        else if (dma_mvout_req_en_vec_i[OC_DMA_IDX]) begin
            current_dma_cmd_o_bank = cmd_o_bank_i;
        end

        dma_cmd_valid = dma_cmd_pending_q || current_dma_cmd_valid;
        cmd_valid = dma_cmd_valid;
        cmd_opcode = CMD_MVOUT;
        cmd_w_bank = 1'b0;
        cmd_o_bank = 1'b0;
        if (dma_cmd_pending_q) begin
            cmd_opcode = dma_cmd_pending_opcode_q;
            cmd_w_bank = dma_cmd_pending_w_bank_q;
            cmd_o_bank = dma_cmd_pending_o_bank_q;
        end
        else if (current_dma_cmd_valid) begin
            cmd_opcode = current_dma_cmd_opcode;
            cmd_w_bank = current_dma_cmd_w_bank;
            cmd_o_bank = current_dma_cmd_o_bank;
        end
    end

    assign cmd_fire = cmd_valid && cmd_ready;
    assign cmd_from_pending = dma_cmd_pending_q;
    assign cmd_from_dma = !dma_cmd_pending_q && current_dma_cmd_valid;
    assign dma_mvout_cmd_fire = cmd_fire &&
                                   (cmd_opcode == CMD_MVOUT) &&
                                   (cmd_from_pending || cmd_from_dma);
    assign dma_mvout_fire_word_addr = cmd_from_pending ? dma_cmd_pending_mvout_word_addr_q :
                                                            mvout_dma_req_word_addr;
    assign dma_mvout_fire_word_count = cmd_from_pending ? dma_cmd_pending_mvout_word_count_q :
                                                             mvout_word_count(mvout_row_num_vec_i[OC_DMA_IDX],
                                                                              mvout_col_num_vec_i[OC_DMA_IDX]);

    always_comb begin
        dma_mvin_req_en_gated_vec_o = '0;
        if (cmd_fire) begin
            unique case (cmd_opcode)
                CMD_MVIN_A: begin
                    dma_mvin_req_en_gated_vec_o[A_DMA_IDX] = 1'b1;
                end
                CMD_MVIN_W: begin
                    dma_mvin_req_en_gated_vec_o[W_DMA_IDX] = 1'b1;
                end
                CMD_META_MVIN: begin
                    dma_mvin_req_en_gated_vec_o[OC_DMA_IDX] = 1'b1;
                end
                default: begin
                end
            endcase
        end
    end

    assign a_push_remaining = (a_beats_remaining_q != 32'd0) ? a_beats_remaining_q :
                              (a_req_pending_q ? a_expected_beats_q : 32'd0);
    assign a_dma_valid = dma_spm_wr_en_vec_i[A_DMA_IDX];
    assign a_dma_data  = dma_spm_din_vec_i[A_DMA_IDX];
    assign a_dma_keep  = dma_spm_wr_mask_vec_i[A_DMA_IDX];
    assign a_dma_last  = (a_push_remaining == 32'd1);

    assign w_push_remaining = (w_beats_remaining_q != 32'd0) ? w_beats_remaining_q :
                              (w_req_pending_q ? w_expected_beats_q : 32'd0);
    assign w_dma_valid = dma_spm_wr_en_vec_i[W_DMA_IDX];
    assign w_dma_data  = dma_spm_din_vec_i[W_DMA_IDX];
    assign w_dma_keep  = dma_spm_wr_mask_vec_i[W_DMA_IDX];
    assign w_dma_last  = (w_push_remaining == 32'd1);

    assign meta_mvin_valid = dma_spm_wr_en_vec_i[OC_DMA_IDX] || dma_acc_wr_en_vec_i[OC_DMA_IDX];
    assign meta_mvin_data  = dma_spm_wr_en_vec_i[OC_DMA_IDX] ? dma_spm_din_vec_i[OC_DMA_IDX] :
                                                               dma_acc_din_vec_i[OC_DMA_IDX][127:0];
    assign meta_mvin_keep  = dma_spm_wr_en_vec_i[OC_DMA_IDX] ? dma_spm_wr_mask_vec_i[OC_DMA_IDX] :
                                                               {8'b0, dma_acc_wr_mask_vec_i[OC_DMA_IDX]};
    assign meta_word_addr  = meta_word_addr_q;

    assign core_busy_o = core_busy || gemm_inflight_q;
    assign core_sa_busy_o = gemm_inflight_q;
    assign core_error_sticky_o = DISABLE_ERROR_CHECKS ? 1'b0 :
                                 (core_error_sticky || command_drop_q || backpressure_error_q);
    assign core_last_error_o = DISABLE_ERROR_CHECKS ? NPU_ERR_NONE : core_last_error;
    assign dma_command_drop_o = command_drop_q;
    assign dma_backpressure_error_o = backpressure_error_q;
    assign meta_beat_fire = meta_mvin_valid && meta_mvin_ready;
    assign meta_expected_beats = ceil_div16_32(plus_one16(mvin_col_num_vec_i[OC_DMA_IDX]));

    always_ff @(posedge clk_i) begin
        if (rst) begin
            a_beats_remaining_q    <= '0;
            a_expected_beats_q      <= '0;
            a_req_pending_q         <= 1'b0;
            w_beats_remaining_q    <= '0;
            w_expected_beats_q      <= '0;
            w_req_pending_q         <= 1'b0;
            meta_word_addr_q       <= '0;
            gemm_inflight_q        <= 1'b0;
            gemm_seen_core_busy_q  <= 1'b0;
            core_sa_done_o       <= 1'b0;
            command_drop_q         <= 1'b0;
            backpressure_error_q   <= 1'b0;
            a_done_pending_q       <= 1'b0;
            a_seen_busy_q          <= 1'b0;
            w_done_pending_q       <= 1'b0;
            w_seen_busy_q          <= 1'b0;
            meta_done_pending_q    <= 1'b0;
            meta_beats_remaining_q <= '0;
            mvin_a_done_o          <= 1'b0;
            mvin_w_done_o          <= 1'b0;
            mvin_meta_done_o       <= 1'b0;
            dma_cmd_pending_q   <= 1'b0;
            dma_cmd_pending_opcode_q <= CMD_MVOUT;
            dma_cmd_pending_w_bank_q <= 1'b0;
            dma_cmd_pending_o_bank_q <= 1'b0;
            dma_cmd_pending_mvout_word_addr_q <= '0;
            dma_cmd_pending_mvout_word_count_q <= '0;
        end
        else begin
            core_sa_done_o <= 1'b0;
            mvin_a_done_o <= 1'b0;
            mvin_w_done_o <= 1'b0;
            mvin_meta_done_o <= 1'b0;

            if (clear_error_i) begin
                command_drop_q       <= 1'b0;
                backpressure_error_q <= 1'b0;
                a_done_pending_q     <= 1'b0;
                a_seen_busy_q        <= 1'b0;
                w_done_pending_q     <= 1'b0;
                w_seen_busy_q        <= 1'b0;
                meta_done_pending_q  <= 1'b0;
                meta_beats_remaining_q <= '0;
            end

            if (dma_cmd_pending_q && current_dma_cmd_valid) begin
                command_drop_q <= 1'b1;
            end
            else if (current_dma_cmd_valid && !cmd_ready) begin
                dma_cmd_pending_q <= 1'b1;
                dma_cmd_pending_opcode_q <= current_dma_cmd_opcode;
                dma_cmd_pending_w_bank_q <= current_dma_cmd_w_bank;
                dma_cmd_pending_o_bank_q <= current_dma_cmd_o_bank;
                dma_cmd_pending_mvout_word_addr_q <= mvout_dma_req_word_addr;
                dma_cmd_pending_mvout_word_count_q <= mvout_word_count(mvout_row_num_vec_i[OC_DMA_IDX],
                                                                           mvout_col_num_vec_i[OC_DMA_IDX]);
            end

            if (cmd_fire && dma_cmd_pending_q) begin
                dma_cmd_pending_q <= 1'b0;
            end

            if (dma_mvin_req_en_vec_i[A_DMA_IDX]) begin
                a_expected_beats_q <= a_expected_beats;
                a_req_pending_q <= 1'b1;
            end
            else if (a_req_pending_q) begin
                a_req_pending_q <= 1'b0;
                if (dma_spm_wr_en_vec_i[A_DMA_IDX] && (a_expected_beats_q != 32'd0)) begin
                    a_beats_remaining_q <= a_expected_beats_q - 32'd1;
                end
                else begin
                    a_beats_remaining_q <= a_expected_beats_q;
                end
            end
            else if (dma_spm_wr_en_vec_i[A_DMA_IDX]) begin
                if (!a_dma_ready) begin
                    backpressure_error_q <= 1'b1;
                end
                if (a_beats_remaining_q != 0) begin
                    a_beats_remaining_q <= a_beats_remaining_q - 32'd1;
                end
            end

            if (dma_mvin_req_en_vec_i[W_DMA_IDX]) begin
                w_expected_beats_q <= w_expected_beats;
                w_req_pending_q <= 1'b1;
            end
            else if (w_req_pending_q) begin
                w_req_pending_q <= 1'b0;
                if (dma_spm_wr_en_vec_i[W_DMA_IDX] && (w_expected_beats_q != 32'd0)) begin
                    w_beats_remaining_q <= w_expected_beats_q - 32'd1;
                end
                else begin
                    w_beats_remaining_q <= w_expected_beats_q;
                end
            end
            else if (dma_spm_wr_en_vec_i[W_DMA_IDX]) begin
                if (!w_dma_ready) begin
                    backpressure_error_q <= 1'b1;
                end
                if (w_beats_remaining_q != 0) begin
                    w_beats_remaining_q <= w_beats_remaining_q - 32'd1;
                end
            end
            if (cmd_fire && (cmd_opcode == CMD_META_MVIN)) begin
                meta_word_addr_q <= META_ADDR_W'(mvin_sram_addr_vec_i[OC_DMA_IDX][META_ADDR_W-1:0]);
                meta_done_pending_q <= 1'b1;
                meta_beats_remaining_q <= meta_expected_beats;
            end
            else if (meta_mvin_valid) begin
                if (!meta_mvin_ready) begin
                    backpressure_error_q <= 1'b1;
                end
                if (meta_mvin_ready) begin
                    meta_word_addr_q <= meta_word_addr_q + META_ADDR_W'(1);
                end
            end

            if (meta_done_pending_q && meta_beat_fire) begin
                if (meta_beats_remaining_q <= 32'd1) begin
                    meta_done_pending_q <= 1'b0;
                    meta_beats_remaining_q <= '0;
                    mvin_meta_done_o <= 1'b1;
                end else begin
                    meta_beats_remaining_q <= meta_beats_remaining_q - 32'd1;
                end
            end

            if (cmd_fire && (cmd_opcode == CMD_MVIN_A)) begin
                a_done_pending_q <= 1'b1;
                a_seen_busy_q <= 1'b0;
            end else if (a_done_pending_q) begin
                if (a_busy) begin
                    a_seen_busy_q <= 1'b1;
                end else if (a_seen_busy_q) begin
                    a_done_pending_q <= 1'b0;
                    a_seen_busy_q <= 1'b0;
                    mvin_a_done_o <= 1'b1;
                end
            end

            if (cmd_fire && (cmd_opcode == CMD_MVIN_W)) begin
                w_done_pending_q <= 1'b1;
                w_seen_busy_q <= 1'b0;
            end else if (w_done_pending_q) begin
                if (w0_busy || w1_busy) begin
                    w_seen_busy_q <= 1'b1;
                end else if (w_seen_busy_q) begin
                    w_done_pending_q <= 1'b0;
                    w_seen_busy_q <= 1'b0;
                    mvin_w_done_o <= 1'b1;
                end
            end

            if (cmd_fire && (cmd_opcode == CMD_GEMM)) begin
                gemm_inflight_q       <= 1'b1;
                gemm_seen_core_busy_q <= 1'b0;
            end
            else if (gemm_inflight_q) begin
                if (core_busy) begin
                    gemm_seen_core_busy_q <= 1'b1;
                end
                else if (gemm_seen_core_busy_q) begin
                    gemm_inflight_q       <= 1'b0;
                    gemm_seen_core_busy_q <= 1'b0;
                    core_sa_done_o      <= 1'b1;
                end
            end
        end
    end

    Versa_P_core #(
        .BANK_DEPTH_WORDS   (BANK_DEPTH_WORDS),
        .A_BANK_DEPTH_WORDS (A_BANK_DEPTH_WORDS),
        .W_BANK_DEPTH_WORDS (W_BANK_DEPTH_WORDS),
        .O_BANK_DEPTH_WORDS (O_BANK_DEPTH_WORDS),
        .READ_LATENCY       (READ_LATENCY),
        .META_WORD128_DEPTH (META_WORD128_DEPTH),
        .FIFO_DEPTH         (FIFO_DEPTH),
        .O_SPLIT_BANK_HALVES(O_SPLIT_BANK_HALVES)
    ) u_Versa_P_core (
        .clk_i                         (clk_i),
        .rst_i                         (rst),
        .clear_error_i                 (clear_error_i),
        .cmd_valid_i                   (cmd_valid),
        .cmd_ready_o                   (cmd_ready),
        .cmd_opcode_i                  (cmd_opcode),
        .cmd_a_bank_i                  (1'b0),
        .cmd_w_bank_i                  (cmd_w_bank),
        .cmd_o_bank_i                  (cmd_o_bank),
        .active_w_bank_i               (active_w_bank_i),
        .co_done_i                     (1'b0),
        .meta_done_i                   (mvin_meta_done_o),
        .mvout_done_i                  (1'b0),
        .a_cfg_base_word_addr_i        (a_base_word),
        .a_cfg_row_count_i             (a_rows),
        .a_cfg_row_bytes_i             (a_row_bytes),
        .a_cfg_row_stride_words_i      (a_stride_words),
        .a_cfg_u8_minus_128_en_i       (cfg_compute_asymmetric_activations_i),
        .a_dma_valid_i                 (a_dma_valid),
        .a_dma_ready_o                 (a_dma_ready),
        .a_dma_data_i                  (a_dma_data),
        .a_dma_keep_i                  (a_dma_keep),
        .a_dma_last_i                  (a_dma_last),
        .w_cfg_base_word_addr_i        (w_base_word),
        .w_cfg_k_loaded_i              (w_k_loaded),
        .w_cfg_n_loaded_i              (w_n_loaded),
        .w_dma_valid_i                 (w_dma_valid),
        .w_dma_ready_o                 (w_dma_ready),
        .w_dma_data_i                  (w_dma_data),
        .w_dma_keep_i                  (w_dma_keep),
        .w_dma_last_i                  (w_dma_last),
        .meta_mvin_valid_i             (meta_mvin_valid),
        .meta_mvin_ready_o             (meta_mvin_ready),
        .meta_mvin_word128_addr_i      (meta_word_addr),
        .meta_mvin_data_i              (meta_mvin_data),
        .meta_mvin_keep_i              (meta_mvin_keep),
        .co_wr_start_i                 (1'b0),
        .co_wr_base_word_addr_i        ('0),
        .co_accumulate_en_i            (co_accumulate_en_i),
        .co_add_bias_en_i              (co_add_bias_en_i),
        .co_bias_addr_i                (META_BYTE_ADDR_W'(co_bias_addr_i)),
        .gemm_mode_i                   (NPU_MODE_INT8),
        .mvout_req_word_addr_i         (mvout_req_word_addr),
        .mvout_m_count_i               (plus_one16(mvout_row_num_vec_i[OC_DMA_IDX])),
        .mvout_n_count_i               (plus_one16(mvout_col_num_vec_i[OC_DMA_IDX])),
        .mvout_dram_base_addr_i        ({32'd0, mvout_dram_base_addr_i}),
        .mvout_dram_row_stride_bytes_i (mvout_dram_row_stride_bytes),
        .mvout_mode_i                  (mvout_mode),
        .mvout_per_channel_scale_i     (cfg_mvout_per_channel_vec_i[OC_DMA_IDX]),
        .mvout_tensor_scale_q8_24_i    (cfg_mvout_scale_param_vec_i[OC_DMA_IDX]),
        .mvout_scale_addr_i            (META_BYTE_ADDR_W'(cfg_mvout_scale_param_vec_i[OC_DMA_IDX])),
        .mvout_dma_valid_o             (mvout_dma_valid_o),
        .mvout_dma_ready_i             (mvout_dma_ready_i),
        .mvout_dma_data_o              (mvout_dma_data_o),
        .mvout_dma_keep_o              (mvout_dma_keep_o),
        .mvout_dma_addr_o              (mvout_dma_addr_o),
        .mvout_dma_last_o              (mvout_dma_last_o),
        .mvout_fifo_almost_full_o      (mvout_fifo_almost_full),
        .a_busy_o                      (a_busy),
        .w0_busy_o                     (w0_busy),
        .w1_busy_o                     (w1_busy),
        .co_busy_o                     (co_busy),
        .busy_o                        (core_busy),
        .error_sticky_o                (core_error_sticky),
        .last_error_o                  (core_last_error),
        .command_accepted_count_o      (command_accepted_count_o),
        .conflict_count_o              (conflict_count_o),
        .gemm_error_sticky_o           (gemm_error_sticky),
        .gemm_last_error_o             (gemm_last_error)
    );

endmodule

`default_nettype wire
