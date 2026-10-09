`default_nettype none

import npu_spm_pkg::*;

// Phase-1 aligned-only 32x32 SA tile producer.
//
// The core drives a direct 32x32 systolic array. In INT8 mode it sign-extends
// A bytes to the internal 16-bit lane. In PV mode it decodes log8 P bytes to
// unsigned U15 probabilities and zero-extends them to the same 16-bit lane.
module spm_sa_gemm_i8_core #(
    parameter int ADDR_W       = 14,
    parameter int K_COUNT_W    = 13,
    parameter int FLUSH_CYCLES = 64,
    parameter int CO_WORDS     = 128,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0
) (
    input  wire logic                 clk_i,
    input  wire logic                 rst_i,
    input  wire logic                 clear_error_i,

    input  wire logic                 cfg_start_i,
    output logic                      cfg_ready_o,
    input  wire logic [ADDR_W-1:0]    cfg_a_base_word_addr_i,
    input  wire logic [ADDR_W-1:0]    cfg_a_k_word_offset_i,
    input  wire logic [ADDR_W-1:0]    cfg_a_row_stride_words_i,
    input  wire logic [ADDR_W-1:0]    cfg_w_base_word_addr_i,
    input  wire logic [5:0]           cfg_m_count_i,
    input  wire logic [5:0]           cfg_n_count_i,
    input  wire logic [K_COUNT_W-1:0] cfg_k_count_i,
    input  wire npu_mode_e            cfg_mode_i,
    input  wire logic                 snapshot_ready_i,

    output logic                      a_rd_req_valid_o,
    input  wire logic                 a_rd_req_ready_i,
    output logic [ADDR_W-1:0]         a_rd_req_word_addr_o,
    input  wire logic                 a_rd_rsp_valid_i,
    input  wire logic [255:0]         a_rd_rsp_data_i,

    output logic                      w_rd_req_valid_o,
    input  wire logic                 w_rd_req_ready_i,
    output logic [ADDR_W-1:0]         w_rd_req_word_addr_o,
    input  wire logic                 w_rd_rsp_valid_i,
    input  wire logic [255:0]         w_rd_rsp_data_i,

    output logic                      co_out_valid_o,
    input  wire logic                 co_out_ready_i,
    output logic [255:0]              co_out_data_o,
    output logic [7:0]                co_out_lane_mask_o,

    output logic                      pv_row_sum_valid_o,
    output logic [NPU_SA_ROWS-1:0][31:0] pv_row_sum_o,

    output logic                      busy_o,
    output logic                      done_o,
    output logic                      error_sticky_o,
    output npu_error_e                last_error_o
);
    localparam int A_META_DEPTH = 64;
    localparam int A_META_W     = $clog2(A_META_DEPTH);

    typedef enum logic [1:0] {
        STATE_IDLE,
        STATE_RUN,
        STATE_FLUSH,
        STATE_WAIT_DRAIN
    } state_e;

    state_e state_q;

    logic [ADDR_W-1:0]    a_base_word_addr_q;
    logic [ADDR_W-1:0]    a_k_word_offset_q;
    logic [ADDR_W-1:0]    a_row_stride_words_q;
    logic [ADDR_W-1:0]    w_base_word_addr_q;
    logic [5:0]           m_count_q;
    logic [5:0]           n_count_q;
    logic [K_COUNT_W-1:0] k_count_q;
    logic [K_COUNT_W-1:0] k_word_count_q;
    npu_mode_e            mode_q;
    logic                 cfg_pending_valid_q;
    logic [ADDR_W-1:0]    cfg_pending_a_base_word_addr_q;
    logic [ADDR_W-1:0]    cfg_pending_a_k_word_offset_q;
    logic [ADDR_W-1:0]    cfg_pending_a_row_stride_words_q;
    logic [ADDR_W-1:0]    cfg_pending_w_base_word_addr_q;
    logic [5:0]           cfg_pending_m_count_q;
    logic [5:0]           cfg_pending_n_count_q;
    logic [K_COUNT_W-1:0] cfg_pending_k_count_q;
    logic [K_COUNT_W-1:0] cfg_pending_k_word_count_q;
    npu_mode_e            cfg_pending_mode_q;

    logic [K_COUNT_W-1:0] issue_cycle_q;
    logic [K_COUNT_W-1:0] w_issue_idx_q;
    logic [ADDR_W-1:0]    a_issue_word_base_q;
    logic [ADDR_W-1:0]    a_issue_addr_q;
    logic [K_COUNT_W-1:0] top_cycle_q;
    logic                 top_started_q;
    logic [7:0]           flush_count_q;
    logic [7:0]           co_word_count_q;
    logic                 sa_drain_done_seen_q;
    logic                 drain_start_pending_q;
    logic [5:0]           drain_pending_m_count_q;
    logic [5:0]           drain_pending_n_count_q;
    logic [5:0]           drain_active_m_count_q;
    logic [5:0]           drain_active_n_count_q;

    logic [1:0][255:0] a_word_q [0:NPU_SA_ROWS-1];

    logic [5:0]           a_meta_row_fifo_q [0:A_META_DEPTH-1];
    logic [K_COUNT_W-1:0] a_meta_word_fifo_q [0:A_META_DEPTH-1];
    logic [A_META_W-1:0]  a_meta_wr_ptr_q;
    logic [A_META_W-1:0]  a_meta_rd_ptr_q;
    logic [A_META_W:0]    a_meta_queued_count_q;
    logic                 a_meta_head_valid_q;
    logic [5:0]           a_meta_head_row_q;
    logic [K_COUNT_W-1:0] a_meta_head_word_q;

    logic [1:0]   w_edge_valid_pipe_q;
    logic [255:0] w_edge_data_pipe_q [0:1];

    logic [NPU_SA_ROWS-1:0]       a_lane_valid;
    logic [NPU_SA_ROWS-1:0][7:0]  a_lane_byte;
    logic [NPU_SA_ROWS-1:0][15:0] a_lane_data;
    logic [NPU_SA_ROWS*8-1:0]     a_lane_log8_code;
    logic [NPU_SA_ROWS*8-1:0]     a_lane_log8_decode_code;
    logic [NPU_SA_ROWS*15-1:0]    a_lane_p_u15;
    logic [NPU_SA_ROWS-1:0]       a_lane_valid_d;
    logic [NPU_SA_ROWS-1:0][15:0] a_lane_data_d;
    logic                         w_common_valid_d;
    logic [255:0]                 w_common_data_d;
    logic [NPU_SA_ROWS-1:0]       pv_row_sum_add_valid_q;
    logic [NPU_SA_ROWS-1:0][14:0] pv_row_sum_add_data_q;
    logic [NPU_SA_ROWS-1:0][31:0] pv_row_sum_q;
    logic [NPU_SA_COLS-1:0]      w_lane_valid;
    logic [NPU_SA_COLS-1:0][7:0] w_lane_data;

    logic sa_step_fire;
    logic sa_clear_acc;
    logic sa_snapshot_acc;
    logic sa_drain_start;
    logic sa_drain_valid;
    logic sa_drain_ready;
    logic [4:0] sa_drain_row_id;
    logic [1:0] sa_drain_group_id;
    logic [255:0] sa_drain_data;
    logic sa_drain_done;
    logic sa_drain_busy;
    logic sa_mode_error;
    logic [7:0] co_out_lane_mask;

    initial begin
        if (FLUSH_CYCLES < 32) begin
            $error("spm_sa_gemm_i8_core: FLUSH_CYCLES below 32 is not valid for the full 32x32 path");
        end
        if (CO_WORDS != 128) begin
            $error("spm_sa_gemm_i8_core: aligned 32x32 path expects 128 C/O words");
        end
    end

    wire cfg_pending_shape_legal =
        (cfg_pending_m_count_q != '0) &&
        (cfg_pending_m_count_q <= 6'd32) &&
        (cfg_pending_n_count_q != '0) &&
        (cfg_pending_n_count_q <= 6'd32) &&
        (cfg_pending_k_count_q != '0) &&
        (cfg_pending_k_count_q <= K_COUNT_W'(4096)) &&
        (cfg_pending_k_count_q[4:0] == 5'd0) &&
        (cfg_pending_a_row_stride_words_q != '0);

    wire [4:0] a_issue_row = issue_cycle_q[4:0];
    wire [K_COUNT_W-1:0] a_issue_word = issue_cycle_q >> 5;
    wire a_issue_in_range = (a_issue_word < k_word_count_q);
    wire a_issue_row_active = ({1'b0, a_issue_row} < m_count_q);
    wire a_rd_req_fire = a_rd_req_valid_o && a_rd_req_ready_i;
    wire w_rd_req_fire = w_rd_req_valid_o && w_rd_req_ready_i;
    wire a_meta_pop_queue =
        a_rd_rsp_valid_i && a_meta_head_valid_q && (a_meta_queued_count_q != '0);
    wire a_meta_req_becomes_head =
        a_rd_req_fire &&
        (!a_meta_head_valid_q ||
         (a_rd_rsp_valid_i && a_meta_head_valid_q && (a_meta_queued_count_q == '0)));
    wire a_meta_enqueue = a_rd_req_fire && !a_meta_req_becomes_head;

    wire top_active = top_started_q || w_edge_valid_pipe_q[1];
    wire [K_COUNT_W-1:0] top_cycle = top_started_q ? top_cycle_q : '0;
    wire w_common_valid = w_edge_valid_pipe_q[1];
    wire [255:0] w_common_data = w_edge_data_pipe_q[1];
    wire pv_log8_mode = (mode_q == NPU_MODE_PV_LOG8_U16I8);
    wire run_inputs_done = top_active && (top_cycle >= (k_count_q + K_COUNT_W'(32)));
    wire co_out_fire = co_out_valid_o && co_out_ready_i;
    wire co_last_fire = co_out_fire && (co_word_count_q == (CO_WORDS - 1));
    wire snapshot_ready = snapshot_ready_i;

    assign cfg_ready_o = (state_q == STATE_IDLE) && !cfg_pending_valid_q &&
                         !drain_start_pending_q &&
                         (DISABLE_ERROR_CHECKS || !error_sticky_o);
    assign busy_o      = cfg_pending_valid_q || (state_q != STATE_IDLE) || sa_drain_busy;

    assign a_rd_req_valid_o = (state_q == STATE_RUN) && a_issue_in_range &&
                              a_issue_row_active;
    assign a_rd_req_word_addr_o = a_issue_addr_q;

    assign w_rd_req_valid_o = (state_q == STATE_RUN) && (w_issue_idx_q < k_count_q);
    assign w_rd_req_word_addr_o = w_base_word_addr_q + ADDR_W'(w_issue_idx_q);

    assign sa_step_fire   = (state_q == STATE_RUN) || (state_q == STATE_FLUSH);
    assign sa_clear_acc   = (state_q == STATE_RUN) && (issue_cycle_q == '0);
    assign sa_snapshot_acc =
        ((state_q == STATE_FLUSH) && (flush_count_q == (FLUSH_CYCLES - 1)) && snapshot_ready) ||
        ((state_q == STATE_WAIT_DRAIN) && snapshot_ready);
    assign sa_drain_start = drain_start_pending_q && !sa_drain_busy;
    assign sa_drain_ready = co_out_ready_i;
    assign co_out_valid_o = sa_drain_valid;
    assign co_out_data_o = sa_drain_data;
    assign co_out_lane_mask_o = co_out_lane_mask;
    assign a_lane_log8_decode_code = pv_log8_mode ? a_lane_log8_code : '0;
    assign pv_row_sum_o = pv_row_sum_q;

    always_comb begin
        logic [5:0] drain_col_base;

        drain_col_base = {1'b0, sa_drain_group_id, 3'b000};
        co_out_lane_mask = '0;
        for (int lane = 0; lane < NPU_ACC_LANES_I32; lane++) begin
            co_out_lane_mask[lane] =
                sa_drain_valid &&
                ({1'b0, sa_drain_row_id} < drain_active_m_count_q) &&
                ((drain_col_base + 6'(lane)) < drain_active_n_count_q);
        end
    end

    always_comb begin
        for (int row = 0; row < NPU_SA_ROWS; row++) begin
            logic [K_COUNT_W-1:0] row_delay;
            logic [K_COUNT_W-1:0] row_k;

            row_delay = K_COUNT_W'(row);
            row_k = top_cycle - row_delay;
            a_lane_valid[row] = top_active &&
                                (6'(row) < m_count_q) &&
                                (top_cycle >= row_delay) &&
                                (row_k < k_count_q);
            a_lane_byte[row] = a_word_q[row][row_k[5]][row_k[4:0]*8 +: 8];
            if (a_rd_rsp_valid_i &&
                a_meta_head_valid_q &&
                (a_meta_head_row_q == 6'(row)) &&
                (a_meta_head_word_q == (row_k >> 5))) begin
                a_lane_byte[row] = a_rd_rsp_data_i[row_k[4:0]*8 +: 8];
            end
            a_lane_log8_code[row*8 +: 8] = a_lane_byte[row];
            a_lane_data[row] = pv_log8_mode
                ? {1'b0, a_lane_p_u15[row*15 +: 15]}
                : {{8{a_lane_byte[row][7]}}, a_lane_byte[row]};
        end
    end

    log8_u15_vector_decoder #(
        .LANES           (NPU_SA_ROWS),
        .USE_MANT15      (1'b0),
        .REGISTER_INPUT  (1'b0),
        .REGISTER_OUTPUT (1'b0)
    ) u_log8_p_decoder (
        .clk_i  (clk_i),
        .code_i (a_lane_log8_decode_code),
        .p_o    (a_lane_p_u15)
    );

    genvar col;
    generate
        for (col = 0; col < NPU_SA_COLS; col++) begin : gen_w_col_skew
            logic [7:0] w_lane;

            assign w_lane = w_common_data_d[col*8 +: 8];
            skew_delay_line #(
                .DATA_W (8),
                .DELAY  (col)
            ) u_w_col_skew (
                .clk_i   (clk_i),
                .rst_i   (rst_i),
                .valid_i (w_common_valid_d),
                .data_i  (w_lane),
                .valid_o (w_lane_valid[col]),
                .data_o  (w_lane_data[col])
            );
        end
    endgenerate

    systolic_array_32x32 u_sa (
        .clk_i                    (clk_i),
        .rst_i                    (rst_i),
        .step_fire_i              (sa_step_fire),
        .mode_i                   (mode_q),
        .clear_acc_i              (sa_clear_acc),
        .snapshot_acc_i           (sa_snapshot_acc),
        .a_in_valid_i             (a_lane_valid_d),
        .a_in_data_i              (a_lane_data_d),
        .w_in_valid_i             (w_lane_valid),
        .w_in_data_i              (w_lane_data),
        .drain_start_i            (sa_drain_start),
        .drain_valid_o            (sa_drain_valid),
        .drain_ready_i            (sa_drain_ready),
        .drain_row_id_o           (sa_drain_row_id),
        .drain_group_id_o         (sa_drain_group_id),
        .drain_data_o             (sa_drain_data),
        .drain_done_o             (sa_drain_done),
        .drain_busy_o             (sa_drain_busy),
        .unsupported_mode_error_o (sa_mode_error)
    );

    always_ff @(posedge clk_i) begin
        logic [5:0] rsp_row;

        if (rst_i) begin
            state_q              <= STATE_IDLE;
            a_base_word_addr_q   <= '0;
            a_k_word_offset_q    <= '0;
            a_row_stride_words_q <= '0;
            w_base_word_addr_q   <= '0;
            m_count_q            <= '0;
            n_count_q            <= '0;
            k_count_q            <= '0;
            k_word_count_q       <= '0;
            mode_q               <= NPU_MODE_INT8;
            cfg_pending_valid_q   <= 1'b0;
            issue_cycle_q        <= '0;
            w_issue_idx_q        <= '0;
            a_issue_word_base_q  <= '0;
            a_issue_addr_q       <= '0;
            top_cycle_q          <= '0;
            top_started_q        <= 1'b0;
            flush_count_q        <= '0;
            co_word_count_q      <= '0;
            sa_drain_done_seen_q <= 1'b0;
            drain_start_pending_q <= 1'b0;
            drain_pending_m_count_q <= '0;
            drain_pending_n_count_q <= '0;
            drain_active_m_count_q <= '0;
            drain_active_n_count_q <= '0;
            a_meta_wr_ptr_q      <= '0;
            a_meta_rd_ptr_q      <= '0;
            a_meta_queued_count_q <= '0;
            a_meta_head_valid_q  <= 1'b0;
            a_meta_head_row_q    <= '0;
            a_meta_head_word_q   <= '0;
            w_edge_valid_pipe_q  <= '0;
            w_common_valid_d     <= 1'b0;
            w_common_data_d      <= '0;
            a_lane_valid_d       <= '0;
            for (int row = 0; row < NPU_SA_ROWS; row++) begin
                a_lane_data_d[row] <= '0;
            end
            done_o               <= 1'b0;
            pv_row_sum_valid_o   <= 1'b0;
            pv_row_sum_add_valid_q <= '0;
            error_sticky_o       <= 1'b0;
            last_error_o         <= NPU_ERR_NONE;
            for (int row = 0; row < NPU_SA_ROWS; row++) begin
                a_word_q[row][0] <= '0;
                a_word_q[row][1] <= '0;
                pv_row_sum_q[row] <= '0;
            end
            for (int idx = 0; idx < A_META_DEPTH; idx++) begin
                a_meta_row_fifo_q[idx] <= '0;
                a_meta_word_fifo_q[idx] <= '0;
            end
            for (int idx = 0; idx < 2; idx++) begin
                w_edge_data_pipe_q[idx] <= '0;
            end
        end else begin
            done_o <= 1'b0;
            pv_row_sum_valid_o <= 1'b0;
            pv_row_sum_add_valid_q <= '0;

            if (pv_log8_mode) begin
                for (int row = 0; row < NPU_SA_ROWS; row++) begin
                    if (pv_row_sum_add_valid_q[row]) begin
                        pv_row_sum_q[row] <= pv_row_sum_q[row] +
                            {17'd0, pv_row_sum_add_data_q[row]};
                    end
                end
            end

            if (sa_drain_start) begin
                drain_start_pending_q <= 1'b0;
                drain_active_m_count_q <= drain_pending_m_count_q;
                drain_active_n_count_q <= drain_pending_n_count_q;
            end

            w_edge_valid_pipe_q[0] <= w_rd_rsp_valid_i;
            w_edge_valid_pipe_q[1] <= w_edge_valid_pipe_q[0];
            w_edge_data_pipe_q[0]  <= w_rd_rsp_data_i;
            w_edge_data_pipe_q[1]  <= w_edge_data_pipe_q[0];
            w_common_valid_d       <= w_common_valid;
            w_common_data_d        <= w_common_data;
            a_lane_valid_d         <= a_lane_valid;
            for (int row = 0; row < NPU_SA_ROWS; row++) begin
                a_lane_data_d[row] <= a_lane_data[row];
            end

            if (clear_error_i) begin
                error_sticky_o <= 1'b0;
                last_error_o   <= NPU_ERR_NONE;
            end else if (!DISABLE_ERROR_CHECKS && sa_mode_error) begin
                error_sticky_o <= 1'b1;
                last_error_o   <= NPU_ERR_UNSUPPORTED_MODE;
            end

            if (sa_snapshot_acc) begin
                co_word_count_q       <= '0;
                sa_drain_done_seen_q  <= 1'b0;
                drain_start_pending_q <= 1'b1;
                drain_pending_m_count_q <= m_count_q;
                drain_pending_n_count_q <= n_count_q;
                done_o                <= 1'b1;
                pv_row_sum_valid_o    <= pv_log8_mode;
            end else if (co_out_fire && !co_last_fire) begin
                co_word_count_q <= co_word_count_q + 8'd1;
            end

            if (sa_drain_done) begin
                sa_drain_done_seen_q <= 1'b1;
            end

            if (a_meta_enqueue) begin
                a_meta_row_fifo_q[a_meta_wr_ptr_q] <= {1'b0, a_issue_row};
                a_meta_word_fifo_q[a_meta_wr_ptr_q] <= a_issue_word;
                a_meta_wr_ptr_q <= a_meta_wr_ptr_q + A_META_W'(1);
            end
            if (a_meta_req_becomes_head) begin
                a_meta_head_valid_q <= 1'b1;
                a_meta_head_row_q   <= {1'b0, a_issue_row};
                a_meta_head_word_q  <= a_issue_word;
            end

            if (a_rd_rsp_valid_i && a_meta_head_valid_q) begin
                rsp_row = a_meta_head_row_q;
                if (a_meta_pop_queue) begin
                    a_meta_head_valid_q <= 1'b1;
                    a_meta_head_row_q   <= a_meta_row_fifo_q[a_meta_rd_ptr_q];
                    a_meta_head_word_q  <= a_meta_word_fifo_q[a_meta_rd_ptr_q];
                    a_meta_rd_ptr_q     <= a_meta_rd_ptr_q + A_META_W'(1);
                end else if (!a_meta_req_becomes_head) begin
                    a_meta_head_valid_q <= 1'b0;
                end
                a_word_q[rsp_row[4:0]][a_meta_head_word_q[0]] <= a_rd_rsp_data_i;
            end
            unique case ({a_meta_enqueue, a_meta_pop_queue})
                2'b10: a_meta_queued_count_q <= a_meta_queued_count_q + (A_META_W+1)'(1);
                2'b01: a_meta_queued_count_q <= a_meta_queued_count_q - (A_META_W+1)'(1);
                default: begin
                end
            endcase

            unique case (state_q)
                STATE_IDLE: begin
                    top_started_q        <= 1'b0;
                    sa_drain_done_seen_q <= 1'b0;
                    w_edge_valid_pipe_q  <= '0;
                    w_common_valid_d     <= 1'b0;
                    a_lane_valid_d       <= '0;
                    if (cfg_pending_valid_q) begin
                        cfg_pending_valid_q <= 1'b0;
                        if (!DISABLE_ERROR_CHECKS &&
                            !mode_supported(cfg_pending_mode_q)) begin
                            error_sticky_o <= 1'b1;
                            last_error_o   <= NPU_ERR_UNSUPPORTED_MODE;
                        end else if (!DISABLE_ERROR_CHECKS &&
                                     !cfg_pending_shape_legal) begin
                            error_sticky_o <= 1'b1;
                            last_error_o   <= NPU_ERR_ILLEGAL_SHAPE;
                        end else begin
                            a_base_word_addr_q   <= cfg_pending_a_base_word_addr_q;
                            a_k_word_offset_q    <= cfg_pending_a_k_word_offset_q;
                            a_row_stride_words_q <= cfg_pending_a_row_stride_words_q;
                            w_base_word_addr_q   <= cfg_pending_w_base_word_addr_q;
                            m_count_q            <= cfg_pending_m_count_q;
                            n_count_q            <= cfg_pending_n_count_q;
                            k_count_q            <= cfg_pending_k_count_q;
                            k_word_count_q       <= cfg_pending_k_word_count_q;
                            mode_q               <= cfg_pending_mode_q;
                            issue_cycle_q        <= '0;
                            w_issue_idx_q        <= '0;
                            a_issue_word_base_q  <= cfg_pending_a_base_word_addr_q +
                                                    cfg_pending_a_k_word_offset_q;
                            a_issue_addr_q       <= cfg_pending_a_base_word_addr_q +
                                                    cfg_pending_a_k_word_offset_q;
                            top_cycle_q          <= '0;
                            flush_count_q        <= '0;
                            co_word_count_q      <= '0;
                            a_meta_wr_ptr_q      <= '0;
                            a_meta_rd_ptr_q      <= '0;
                            a_meta_queued_count_q <= '0;
                            a_meta_head_valid_q  <= 1'b0;
                            if (cfg_pending_mode_q == NPU_MODE_PV_LOG8_U16I8) begin
                                for (int row = 0; row < NPU_SA_ROWS; row++) begin
                                    pv_row_sum_q[row] <= '0;
                                end
                            end
                            state_q              <= STATE_RUN;
                        end
                    end else if (cfg_start_i && cfg_ready_o) begin
                        cfg_pending_valid_q             <= 1'b1;
                        cfg_pending_a_base_word_addr_q  <= cfg_a_base_word_addr_i;
                        cfg_pending_a_k_word_offset_q   <= cfg_a_k_word_offset_i;
                        cfg_pending_a_row_stride_words_q <= cfg_a_row_stride_words_i;
                        cfg_pending_w_base_word_addr_q  <= cfg_w_base_word_addr_i;
                        cfg_pending_m_count_q           <= cfg_m_count_i;
                        cfg_pending_n_count_q           <= cfg_n_count_i;
                        cfg_pending_k_count_q           <= cfg_k_count_i;
                        cfg_pending_k_word_count_q      <= cfg_k_count_i >> 5;
                        cfg_pending_mode_q              <= cfg_mode_i;
                    end
                end

                STATE_RUN: begin
                    if (pv_log8_mode) begin
                        for (int row = 0; row < NPU_SA_ROWS; row++) begin
                            if (a_lane_valid_d[row]) begin
                                pv_row_sum_add_valid_q[row] <= 1'b1;
                                pv_row_sum_add_data_q[row] <= a_lane_data_d[row][14:0];
                            end
                        end
                    end
                    issue_cycle_q <= issue_cycle_q + K_COUNT_W'(1);
                    if (a_issue_row == 5'd31) begin
                        a_issue_word_base_q <= a_issue_word_base_q + ADDR_W'(1);
                        a_issue_addr_q      <= a_issue_word_base_q + ADDR_W'(1);
                    end else begin
                        a_issue_addr_q <= a_issue_addr_q + a_row_stride_words_q;
                    end
                    if (w_rd_req_fire) begin
                        w_issue_idx_q <= w_issue_idx_q + K_COUNT_W'(1);
                    end
                    if (w_edge_valid_pipe_q[1] && !top_started_q) begin
                        top_started_q <= 1'b1;
                        top_cycle_q   <= K_COUNT_W'(1);
                    end else if (top_started_q) begin
                        top_cycle_q <= top_cycle_q + K_COUNT_W'(1);
                    end
                    if (run_inputs_done) begin
                        flush_count_q <= '0;
                        state_q       <= STATE_FLUSH;
                    end
                end

                STATE_FLUSH: begin
                    if (flush_count_q == (FLUSH_CYCLES - 1)) begin
                        state_q <= snapshot_ready ? STATE_IDLE : STATE_WAIT_DRAIN;
                    end else begin
                        flush_count_q <= flush_count_q + 8'd1;
                    end
                end

                STATE_WAIT_DRAIN: begin
                    if (snapshot_ready) begin
                        state_q <= STATE_IDLE;
                    end
                end

                default: begin
                    state_q <= STATE_IDLE;
                end
            endcase
        end
    end

    wire unused_sa_ids = ^sa_drain_row_id ^ ^sa_drain_group_id ^
                         sa_drain_done_seen_q ^ ^a_base_word_addr_q ^
                         ^a_k_word_offset_q;

endmodule

`default_nettype wire
