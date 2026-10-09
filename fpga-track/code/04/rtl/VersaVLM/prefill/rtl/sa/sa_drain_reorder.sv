`default_nettype none

import npu_spm_pkg::*;

// SA drain reorder for Phase-1 INT8.
//
// Purpose:
//   Convert the physical PE drain scan order into C/O row-major 256-bit words.
//   Each input group contains four PE payloads. Each PE payload is
//   `{acc1_i32, acc0_i32}`. INT8 output order is:
//       C[0] word0..3, C[1] word0..3, C[2] word0..3, ...
//
// Clock/reset:
//   Synchronous active-high reset. Only state and valid registers reset.
//
// Valid/ready:
//   Supports downstream backpressure. Input is accepted only when enough local
//   buffering is available for the group pair being assembled.
module sa_drain_reorder (
    input  wire logic         clk_i,
    input  wire logic         rst_i,

    input  wire logic         in_valid_i,
    output logic              in_ready_o,
    input  wire npu_mode_e    in_mode_i,
    input  wire logic [3:0]   in_row_id_i,
    input  wire logic [2:0]   in_group_id_i,
    input  wire logic [255:0] in_data_i,

    output logic              out_valid_o,
    input  wire logic         out_ready_i,
    output npu_mode_e         out_mode_o,
    output logic [255:0]      out_data_o,
    output logic [7:0]        out_lane_mask_o,

    output logic              error_sticky_o,
    output npu_error_e        last_error_o
);
    typedef enum logic [1:0] {
        RD_IDLE,
        RD_HAVE_EVEN_GROUP,
        RD_EMIT_ODD_ROW
    } reorder_state_e;

    reorder_state_e state_q;
    logic [127:0] even_lo_q;
    logic [127:0] odd_lo_q;
    logic [255:0] odd_words_q [0:3];
    logic [1:0]   odd_emit_idx_q;
    logic [255:0] out_data_q;
    logic [7:0]   out_lane_mask_q;
    logic         out_valid_q;

    wire out_fire = out_valid_q && out_ready_i;
    wire int8_mode = (in_mode_i == NPU_MODE_INT8);
    wire group_even = (in_group_id_i[0] == 1'b0);
    wire [1:0] word_idx = in_group_id_i[2:1];

    assign out_valid_o     = out_valid_q;
    assign out_data_o      = out_data_q;
    assign out_mode_o      = NPU_MODE_INT8;
    assign out_lane_mask_o = out_lane_mask_q;
    assign in_ready_o      = !error_sticky_o &&
                             ((state_q == RD_IDLE) ||
                              (state_q == RD_HAVE_EVEN_GROUP && !out_valid_q));

    function automatic logic [127:0] select_acc0(input logic [255:0] group_data);
        select_acc0 = {
            group_data[3*64 +: 32],
            group_data[2*64 +: 32],
            group_data[1*64 +: 32],
            group_data[0*64 +: 32]
        };
    endfunction

    function automatic logic [127:0] select_acc1(input logic [255:0] group_data);
        select_acc1 = {
            group_data[3*64 + 32 +: 32],
            group_data[2*64 + 32 +: 32],
            group_data[1*64 + 32 +: 32],
            group_data[0*64 + 32 +: 32]
        };
    endfunction

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            state_q         <= RD_IDLE;
            even_lo_q       <= '0;
            odd_lo_q        <= '0;
            odd_emit_idx_q  <= '0;
            out_data_q      <= '0;
            out_lane_mask_q <= '0;
            out_valid_q     <= 1'b0;
            error_sticky_o  <= 1'b0;
            last_error_o    <= NPU_ERR_NONE;
            for (int idx = 0; idx < 4; idx++) begin
                odd_words_q[idx] <= '0;
            end
        end else begin
            if (out_fire) begin
                out_valid_q <= 1'b0;
            end

            unique case (state_q)
                RD_IDLE: begin
                    if (in_valid_i && in_ready_o) begin
                        if (!int8_mode) begin
                            error_sticky_o <= 1'b1;
                            last_error_o   <= NPU_ERR_UNSUPPORTED_MODE;
                        end else if (!group_even) begin
                            error_sticky_o <= 1'b1;
                            last_error_o   <= NPU_ERR_ILLEGAL_SHAPE;
                        end else begin
                            even_lo_q <= select_acc0(in_data_i);
                            odd_lo_q  <= select_acc1(in_data_i);
                            state_q   <= RD_HAVE_EVEN_GROUP;
                        end
                    end
                end

                RD_HAVE_EVEN_GROUP: begin
                    if (in_valid_i && in_ready_o) begin
                        if (!int8_mode || group_even) begin
                            error_sticky_o <= 1'b1;
                            last_error_o   <= NPU_ERR_ILLEGAL_SHAPE;
                        end else begin
                            out_data_q           <= {select_acc0(in_data_i), even_lo_q};
                            out_lane_mask_q      <= 8'hff;
                            out_valid_q          <= 1'b1;
                            odd_words_q[word_idx] <= {select_acc1(in_data_i), odd_lo_q};
                            if (in_group_id_i == 3'd7) begin
                                odd_emit_idx_q <= '0;
                                state_q        <= RD_EMIT_ODD_ROW;
                            end else begin
                                state_q <= RD_IDLE;
                            end
                        end
                    end
                end

                RD_EMIT_ODD_ROW: begin
                    if (!out_valid_q || out_fire) begin
                        out_data_q      <= odd_words_q[odd_emit_idx_q];
                        out_lane_mask_q <= 8'hff;
                        out_valid_q     <= 1'b1;
                        if (odd_emit_idx_q == 2'd3) begin
                            odd_emit_idx_q <= '0;
                            state_q        <= RD_IDLE;
                        end else begin
                            odd_emit_idx_q <= odd_emit_idx_q + 2'd1;
                        end
                    end
                end

                default: begin
                    state_q <= RD_IDLE;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
