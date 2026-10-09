// pe_mac_i8.v --- single signed int8 x int8 -> int32 MAC processing element.
// One DSP48E2 maps to (a_in * w_in + acc) per cycle. Drain uses a snapshot
// outside the PE, so the live accumulator can immediately start the next group.
// Latency: 1 cycle (acc registered).
`timescale 1ns/1ps
module pe_mac_i8 (
    input  wire                 clk,
    input  wire                 rst_n,    // sync active-low reset
    input  wire                 clear,    // sync zero (acc <= 0)
    input  wire                 en,       // accumulate enable
    input  wire signed [7:0]    a_in,
    input  wire signed [7:0]    w_in,
    output wire signed [31:0]   acc_out
);
    reg signed [31:0] acc_r;
    wire signed [8:0] a_ext = {a_in[7], a_in};
    wire signed [8:0] w_ext = {w_in[7], w_in};
    (* use_dsp = "yes" *) wire signed [17:0] prod = a_ext * w_ext;
    wire signed [31:0] prod32 = {{14{prod[17]}}, prod};

    always @(posedge clk) begin
        if (!rst_n)      acc_r <= 32'sd0;
        else if (clear)  acc_r <= en ? prod32 : 32'sd0;
        else if (en)     acc_r <= acc_r + prod32;
    end

    assign acc_out = acc_r;
endmodule
