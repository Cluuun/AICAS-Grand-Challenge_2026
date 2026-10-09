`default_nettype none

import npu_spm_pkg::*;

// C/O int32 accumulation pipeline.
//
// Purpose:
//   Combine SA drain partial sums with optional old accumulator data or bias
//   metadata, then emit a 256-bit C/O-bank write word plus byte enable.
//
// Clock/reset:
//   Synchronous active-high reset. Only valid/status pipeline state resets.
//
// Valid/ready:
//   One-stage ready/valid pipeline. `accumulate_en_i && add_bias_en_i` is an
//   illegal Phase-1 descriptor combination and is rejected deterministically.
module co_accumulate (
    input  wire logic         clk_i,
    input  wire logic         rst_i,

    input  wire logic         clear_error_i,
    input  wire logic         accumulate_en_i,
    input  wire logic         add_bias_en_i,

    input  wire logic         in_valid_i,
    output logic         in_ready_o,
    input  wire logic [255:0] partial_data_i,
    input  wire logic [7:0]   lane_mask_i,
    input  wire logic [255:0] old_acc_data_i,
    input  wire logic [255:0] bias_data_i,

    output logic         out_valid_o,
    input  wire logic         out_ready_i,
    output logic [255:0] out_data_o,
    output logic [31:0]  out_byte_en_o,

    output logic         error_sticky_o,
    output npu_error_e   last_error_o
);
    logic illegal_flags;
    logic out_valid_q;
    logic [255:0] out_data_q;
    logic [31:0] out_byte_en_q;

    assign illegal_flags = accumulate_en_i && add_bias_en_i;
    assign in_ready_o    = !out_valid_q || out_ready_i;
    assign out_valid_o   = out_valid_q;
    assign out_data_o    = out_data_q;
    assign out_byte_en_o = out_byte_en_q;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            out_valid_q    <= 1'b0;
            out_data_q     <= '0;
            out_byte_en_q  <= '0;
            error_sticky_o <= 1'b0;
            last_error_o   <= NPU_ERR_NONE;
        end else begin
            if (clear_error_i) begin
                error_sticky_o <= 1'b0;
                last_error_o   <= NPU_ERR_NONE;
            end

            if (out_valid_q && out_ready_i) begin
                out_valid_q <= 1'b0;
            end

            if (in_valid_i && in_ready_o) begin
                if (illegal_flags) begin
                    error_sticky_o <= 1'b1;
                    last_error_o   <= NPU_ERR_ILLEGAL_FLAGS;
                    out_valid_q    <= 1'b0;
                end else begin
                    out_valid_q   <= 1'b1;
                    out_byte_en_q <= lane_mask_to_byte_strobe(lane_mask_i);
                    for (int lane = 0; lane < NPU_ACC_LANES_I32; lane++) begin
                        logic signed [31:0] partial_lane;
                        logic signed [31:0] addend_lane;
                        partial_lane = partial_data_i[lane*32 +: 32];
                        if (accumulate_en_i) begin
                            addend_lane = old_acc_data_i[lane*32 +: 32];
                        end else if (add_bias_en_i) begin
                            addend_lane = bias_data_i[lane*32 +: 32];
                        end else begin
                            addend_lane = 32'sd0;
                        end
                        out_data_q[lane*32 +: 32] <= partial_lane + addend_lane;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
