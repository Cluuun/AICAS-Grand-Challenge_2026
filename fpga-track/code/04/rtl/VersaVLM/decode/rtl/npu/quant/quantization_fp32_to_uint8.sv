`timescale 1ns / 1ps

module quantization_fp32_to_uint8 #(
    parameter INPUT_NUMBER = 8,
    parameter int LATENCY = 4
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          in_valid,
    input  logic [INPUT_NUMBER-1:0][31:0] input_data,
    input  logic [31:0]                   quant_inv_scale,
    input  logic [7:0]                    zero_point,

    output logic                          out_valid,
    output logic [INPUT_NUMBER-1:0][7:0]  output_data
);

localparam logic [1:0] SPECIAL_NORMAL   = 2'b00;
localparam logic [1:0] SPECIAL_USE_ZP   = 2'b01;
localparam logic [1:0] SPECIAL_SAT_ZERO = 2'b10;
localparam logic [1:0] SPECIAL_SAT_MAX  = 2'b11;

logic                                           backend_out_valid;
logic signed [INPUT_NUMBER-1:0][31:0]           backend_output_data;
logic [LATENCY-1:0]                     valid_pipe;
logic [LATENCY-1:0][7:0]                zero_point_pipe;
logic [LATENCY-1:0][INPUT_NUMBER-1:0][1:0] special_pipe;
logic signed [INPUT_NUMBER-1:0][32:0]           biased_output;

function automatic logic fp_is_nan(input logic [31:0] value_bits);
    fp_is_nan = (value_bits[30:23] == 8'hFF) && (value_bits[22:0] != 23'd0);
endfunction

function automatic logic fp_is_zero(input logic [31:0] value_bits);
    fp_is_zero = (value_bits[30:23] == 8'h00) && (value_bits[22:0] == 23'd0);
endfunction

function automatic logic fp_is_pos_inf(input logic [31:0] value_bits);
    fp_is_pos_inf = (value_bits == 32'h7F80_0000);
endfunction

function automatic logic fp_is_neg_inf(input logic [31:0] value_bits);
    fp_is_neg_inf = (value_bits == 32'hFF80_0000);
endfunction

function automatic logic fp_is_inf(input logic [31:0] value_bits);
    fp_is_inf = (value_bits[30:23] == 8'hFF) && (value_bits[22:0] == 23'd0);
endfunction

function automatic logic [1:0] classify_special(
    input logic [31:0] value_bits,
    input logic [31:0] inv_scale_bits
);
    logic inv_scale_invalid;
    begin
        inv_scale_invalid = fp_is_nan(inv_scale_bits) || inv_scale_bits[31] || fp_is_inf(inv_scale_bits);

        if (fp_is_nan(value_bits) || inv_scale_invalid) begin
            classify_special = SPECIAL_USE_ZP;
        end
        else if (fp_is_pos_inf(value_bits)) begin
            classify_special = SPECIAL_SAT_MAX;
        end
        else if (fp_is_neg_inf(value_bits)) begin
            classify_special = SPECIAL_SAT_ZERO;
        end
        else begin
            classify_special = SPECIAL_NORMAL;
        end
    end
endfunction

quantization_fp32_to_uint8_fpga_backend #(
    .INPUT_NUMBER(INPUT_NUMBER),
    .LATENCY     (LATENCY)
) u_quantization_fp32_to_uint8_fpga_backend (
    .clk         (clk),
    .rst_n       (rst_n),
    .in_valid    (in_valid),
    .input_data  (input_data),
    .quant_inv_scale (quant_inv_scale),
    .out_valid   (backend_out_valid),
    .output_data (backend_output_data)
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_pipe      <= '0;
        zero_point_pipe <= '0;
        special_pipe    <= '0;
    end
    else begin
        valid_pipe[0] <= in_valid;
        if (in_valid) begin
            zero_point_pipe[0] <= zero_point;
            for (int lane = 0; lane < INPUT_NUMBER; lane++) begin
                special_pipe[0][lane] <= classify_special(input_data[lane], quant_inv_scale);
            end
        end

        for (int stage = 1; stage < LATENCY; stage++) begin
            valid_pipe[stage] <= valid_pipe[stage-1];
            if (valid_pipe[stage-1]) begin
                zero_point_pipe[stage] <= zero_point_pipe[stage-1];
                special_pipe[stage]    <= special_pipe[stage-1];
            end
        end
    end
end

always_comb begin
    output_data   = '0;
    biased_output = '0;

    for (int lane = 0; lane < INPUT_NUMBER; lane++) begin
        biased_output[lane] =
            $signed({backend_output_data[lane][31], backend_output_data[lane]}) +
            $signed({25'd0, zero_point_pipe[LATENCY-1]});

        unique case (special_pipe[LATENCY-1][lane])
            SPECIAL_USE_ZP: begin
                output_data[lane] = zero_point_pipe[LATENCY-1];
            end

            SPECIAL_SAT_ZERO: begin
                output_data[lane] = 8'd0;
            end

            SPECIAL_SAT_MAX: begin
                output_data[lane] = 8'hFF;
            end

            default: begin
                if ($signed(biased_output[lane]) < 33'sd0) begin
                    output_data[lane] = 8'd0;
                end
                else if ($signed(biased_output[lane]) > 33'sd255) begin
                    output_data[lane] = 8'hFF;
                end
                else begin
                    output_data[lane] = biased_output[lane][7:0];
                end
            end
        endcase
    end
end

assign out_valid = rst_n && backend_out_valid;

`ifndef SYNTHESIS
always_ff @(posedge clk) begin
    if (rst_n && (backend_out_valid !== valid_pipe[LATENCY-1])) begin
        $error("quantization_fp32_to_uint8 valid pipe mismatch: backend=%0b sideband=%0b",
               backend_out_valid, valid_pipe[LATENCY-1]);
    end
end
`endif

endmodule
