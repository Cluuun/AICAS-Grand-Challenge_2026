`default_nettype none

import npu_spm_pkg::*;

// Thin Phase-1 SPM subsystem owner shell.
//
// Purpose:
//   Accept command-level starts from the scheduler/control layer and fan them
//   out to bank-local loaders or C/O blocks. This shell owns only phase/conflict
//   policy; it intentionally contains no shared A/W/C/O data crossbar.
//
// Clock/reset:
//   Synchronous active-high reset. Only owner/busy/status registers reset.
//
// Valid/ready:
//   Commands are accepted only when their required phase resources are idle.
//   Rejected commands set sticky error status; Phase 1 does not queue them.
module bank_owner (
    input  wire logic        clk_i,
    input  wire logic        rst_i,
    input  wire logic        clear_error_i,

    input  wire logic        cmd_valid_i,
    output logic             cmd_ready_o,
    input  wire logic [2:0]  cmd_opcode_i,
    input  wire logic        cmd_a_bank_i,
    input  wire logic        cmd_w_bank_i,
    input  wire logic        cmd_o_bank_i,
    input  wire logic        cmd_gemm_attention_qk_mode_i,
    input  wire logic        active_w_bank_i,

    input  wire logic        mvin_a_done_i,
    input  wire logic        w_done_i,
    input  wire logic        meta_done_i,
    input  wire logic        gemm_done_i,
    input  wire logic        mvout_done_i,
    input  wire logic        attention_qk_done_i,
    input  wire logic        co_done_i,

    output logic             start_mvin_a_o,
    output logic             start_mvin_w_o,
    output logic             start_metadata_mvin_o,
    output logic             start_gemm_o,
    output logic             start_mvout_o,
    output logic             start_attention_qk_o,

    output logic             a_busy_o,
    output logic             a0_busy_o,
    output logic             a1_busy_o,
    output logic             w0_busy_o,
    output logic             w1_busy_o,
    output logic             co_busy_o,
    output logic             co0_busy_o,
    output logic             co1_busy_o,
    output logic             conflict_sticky_o,
    output npu_error_e       last_error_o,
    output logic [31:0]      command_accepted_count_o,
    output logic [31:0]      conflict_count_o
);
    localparam logic [2:0] CMD_MVIN_A    = 3'd0;
    localparam logic [2:0] CMD_MVIN_W    = 3'd1;
    localparam logic [2:0] CMD_META_MVIN = 3'd2;
    localparam logic [2:0] CMD_GEMM      = 3'd3;
    localparam logic [2:0] CMD_MVOUT     = 3'd4;
    localparam logic [2:0] CMD_ATTENTION_QK = 3'd5;

    logic [1:0] a_busy_q;
    logic [1:0] co_busy_q;
    logic w0_busy_q;
    logic w1_busy_q;
    logic w_load_bank_q;
    logic mvin_a_bank_q;
    logic gemm_a_bank_q;
    logic gemm_o_bank_q;
    logic mvout_o_bank_q;
    logic gemm_uses_w_bank_q;
    logic gemm_w_bank_q;
    logic attention_qk_a_bank_q;
    logic attention_qk_w_bank_q;
    logic attention_qk_o_bank_q;
    logic meta_busy_q;
    logic command_conflict;
    logic command_supported;
    logic selected_w_busy;
    logic selected_a_busy;
    logic selected_co_busy;

    assign a_busy_o  = |a_busy_q;
    assign a0_busy_o = a_busy_q[0];
    assign a1_busy_o = a_busy_q[1];
    assign w0_busy_o = w0_busy_q;
    assign w1_busy_o = w1_busy_q;
    assign co_busy_o = (|co_busy_q) || meta_busy_q;
    assign co0_busy_o = co_busy_q[0];
    assign co1_busy_o = co_busy_q[1];

    assign selected_w_busy = cmd_w_bank_i ? w1_busy_q : w0_busy_q;
    assign selected_a_busy = cmd_a_bank_i ? a_busy_q[1] : a_busy_q[0];
    assign selected_co_busy = cmd_o_bank_i ? co_busy_q[1] : co_busy_q[0];
    assign command_supported = (cmd_opcode_i <= CMD_ATTENTION_QK);

    always_comb begin
        command_conflict = 1'b0;
        unique case (cmd_opcode_i)
            CMD_MVIN_A: begin
                command_conflict = selected_a_busy;
            end
            CMD_MVIN_W: begin
                command_conflict = selected_w_busy || (cmd_w_bank_i == active_w_bank_i);
            end
            CMD_META_MVIN: begin
                command_conflict = meta_busy_q || (|co_busy_q);
            end
            CMD_GEMM: begin
                command_conflict = selected_a_busy || selected_co_busy || meta_busy_q ||
                                   (cmd_gemm_attention_qk_mode_i && selected_w_busy);
            end
            CMD_MVOUT: begin
                command_conflict = selected_co_busy || meta_busy_q;
            end
            CMD_ATTENTION_QK: begin
                command_conflict = selected_a_busy || selected_w_busy ||
                                   selected_co_busy || meta_busy_q;
            end
            default: begin
                command_conflict = 1'b1;
            end
        endcase
    end

    assign cmd_ready_o = cmd_valid_i && command_supported && !command_conflict;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            a_busy_q                 <= '0;
            w0_busy_q                <= 1'b0;
            w1_busy_q                <= 1'b0;
            w_load_bank_q            <= 1'b0;
            co_busy_q                <= '0;
            mvin_a_bank_q            <= 1'b0;
            gemm_a_bank_q            <= 1'b0;
            gemm_o_bank_q            <= 1'b0;
            gemm_uses_w_bank_q       <= 1'b0;
            gemm_w_bank_q            <= 1'b0;
            mvout_o_bank_q           <= 1'b0;
            attention_qk_a_bank_q       <= 1'b0;
            attention_qk_w_bank_q       <= 1'b0;
            attention_qk_o_bank_q       <= 1'b0;
            meta_busy_q              <= 1'b0;
            start_mvin_a_o           <= 1'b0;
            start_mvin_w_o           <= 1'b0;
            start_metadata_mvin_o    <= 1'b0;
            start_gemm_o             <= 1'b0;
            start_mvout_o            <= 1'b0;
            start_attention_qk_o        <= 1'b0;
            conflict_sticky_o        <= 1'b0;
            last_error_o             <= NPU_ERR_NONE;
            command_accepted_count_o <= '0;
            conflict_count_o         <= '0;
        end else begin
            start_mvin_a_o        <= 1'b0;
            start_mvin_w_o        <= 1'b0;
            start_metadata_mvin_o <= 1'b0;
            start_gemm_o          <= 1'b0;
            start_mvout_o         <= 1'b0;
            start_attention_qk_o     <= 1'b0;

            if (clear_error_i) begin
                a_busy_q          <= '0;
                w0_busy_q         <= 1'b0;
                w1_busy_q         <= 1'b0;
                w_load_bank_q     <= 1'b0;
                co_busy_q         <= '0;
                meta_busy_q       <= 1'b0;
                gemm_uses_w_bank_q <= 1'b0;
                conflict_sticky_o <= 1'b0;
                last_error_o      <= NPU_ERR_NONE;
            end

            if (mvin_a_done_i) begin
                a_busy_q[mvin_a_bank_q] <= 1'b0;
            end
            if (w_done_i) begin
                if (w_load_bank_q) begin
                    w1_busy_q <= 1'b0;
                end else begin
                    w0_busy_q <= 1'b0;
                end
            end
            if (meta_done_i) begin
                meta_busy_q <= 1'b0;
            end
            if (gemm_done_i) begin
                a_busy_q[gemm_a_bank_q]  <= 1'b0;
                co_busy_q[gemm_o_bank_q] <= 1'b0;
                if (gemm_uses_w_bank_q) begin
                    if (gemm_w_bank_q) begin
                        w1_busy_q <= 1'b0;
                    end else begin
                        w0_busy_q <= 1'b0;
                    end
                    gemm_uses_w_bank_q <= 1'b0;
                end
            end
            if (mvout_done_i) begin
                co_busy_q[mvout_o_bank_q] <= 1'b0;
            end
            if (attention_qk_done_i) begin
                a_busy_q[attention_qk_a_bank_q]  <= 1'b0;
                co_busy_q[attention_qk_o_bank_q] <= 1'b0;
                if (attention_qk_w_bank_q) begin
                    w1_busy_q <= 1'b0;
                end else begin
                    w0_busy_q <= 1'b0;
                end
            end
            if (co_done_i) begin
                co_busy_q   <= '0;
                meta_busy_q <= 1'b0;
            end

            if (cmd_valid_i && !cmd_ready_o) begin
                conflict_sticky_o <= 1'b1;
                last_error_o      <= command_supported ? NPU_ERR_BANK_CONFLICT : NPU_ERR_UNSUPPORTED_MODE;
                conflict_count_o  <= conflict_count_o + 32'd1;
            end else if (cmd_ready_o) begin
                command_accepted_count_o <= command_accepted_count_o + 32'd1;
                unique case (cmd_opcode_i)
                    CMD_MVIN_A: begin
                        a_busy_q[cmd_a_bank_i] <= 1'b1;
                        mvin_a_bank_q          <= cmd_a_bank_i;
                        start_mvin_a_o         <= 1'b1;
                    end
                    CMD_MVIN_W: begin
                        if (cmd_w_bank_i) begin
                            w1_busy_q <= 1'b1;
                        end else begin
                            w0_busy_q <= 1'b1;
                        end
                        w_load_bank_q  <= cmd_w_bank_i;
                        start_mvin_w_o <= 1'b1;
                    end
                    CMD_META_MVIN: begin
                        meta_busy_q           <= 1'b1;
                        start_metadata_mvin_o <= 1'b1;
                    end
                    CMD_GEMM: begin
                        a_busy_q[cmd_a_bank_i]  <= 1'b1;
                        co_busy_q[cmd_o_bank_i] <= 1'b1;
                        gemm_a_bank_q           <= cmd_a_bank_i;
                        gemm_o_bank_q           <= cmd_o_bank_i;
                        gemm_uses_w_bank_q      <= cmd_gemm_attention_qk_mode_i;
                        gemm_w_bank_q           <= cmd_w_bank_i;
                        if (cmd_gemm_attention_qk_mode_i) begin
                            if (cmd_w_bank_i) begin
                                w1_busy_q <= 1'b1;
                            end else begin
                                w0_busy_q <= 1'b1;
                            end
                        end
                        start_gemm_o            <= 1'b1;
                    end
                    CMD_MVOUT: begin
                        co_busy_q[cmd_o_bank_i] <= 1'b1;
                        mvout_o_bank_q          <= cmd_o_bank_i;
                        start_mvout_o           <= 1'b1;
                    end
                    CMD_ATTENTION_QK: begin
                        a_busy_q[cmd_a_bank_i]  <= 1'b1;
                        co_busy_q[cmd_o_bank_i] <= 1'b1;
                        if (cmd_w_bank_i) begin
                            w1_busy_q <= 1'b1;
                        end else begin
                            w0_busy_q <= 1'b1;
                        end
                        attention_qk_a_bank_q <= cmd_a_bank_i;
                        attention_qk_w_bank_q <= cmd_w_bank_i;
                        attention_qk_o_bank_q <= cmd_o_bank_i;
                        start_attention_qk_o  <= 1'b1;
                    end
                    default: begin
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
