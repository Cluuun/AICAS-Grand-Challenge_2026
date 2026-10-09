// pe_mac_i8_v6.v --- single signed int8 x int8 -> int32 MAC processing element.
`timescale 1ns/1ps
module pe_mac_i8_v6 (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 clear,
    input  wire                 en,
    input  wire signed [7:0]    a_in,
    input  wire signed [7:0]    w_in,
    output wire signed [31:0]   acc_out
);
    reg signed [31:0] acc_r;
    reg signed [31:0] prod_pipe_r;
    reg               clear_pipe_r;
    reg               en_pipe_r;

    wire signed [8:0] a_ext = {a_in[7], a_in};
    wire signed [8:0] w_ext = {w_in[7], w_in};
    (* use_dsp = "yes" *) wire signed [17:0] prod = a_ext * w_ext;
    wire signed [31:0] prod32 = {{14{prod[17]}}, prod};

    always @(posedge clk) begin
        if (!rst_n) begin
            prod_pipe_r  <= 32'sd0;
            clear_pipe_r <= 1'b0;
            en_pipe_r    <= 1'b0;
            acc_r        <= 32'sd0;
        end else begin
            // Stage 0: register the multiplier result and its control. This
            // breaks the full a*w+acc DSP path that was the integrated
            // 300 MHz WNS limiter while preserving one accepted MAC per cycle.
            prod_pipe_r  <= prod32;
            clear_pipe_r <= clear;
            en_pipe_r    <= en;

            // Stage 1: apply the previous product to the accumulator. A clear
            // beat starts a new K tile by replacing acc with the first product;
            // a clear without en still zeros the accumulator, matching the
            // original PE behavior.
            if (clear_pipe_r) begin
                acc_r <= en_pipe_r ? prod_pipe_r : 32'sd0;
            end else if (en_pipe_r) begin
                acc_r <= acc_r + prod_pipe_r;
            end
        end
    end

    assign acc_out = acc_r;
endmodule
