// pe_mac_i8.v — Single INT8 MAC Processing Element for systolic array.
// One DSP48E2 + minimal LUT (~15). Controlled by global signals.
module pe_mac_i8 (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire        clear,
    input  wire [7:0]  a_in,
    input  wire [7:0]  w_in,
    output wire [31:0] acc_out
);

    reg signed [31:0] acc;
    wire signed [7:0] a_s = a_in;
    wire signed [7:0] w_s = w_in;
    wire signed [15:0] prod = a_s * w_s;

    always @(posedge clk) begin
        if (rst | clear)
            acc <= 32'd0;
        else if (en)
            acc <= acc + {{16{prod[15]}}, prod};
    end

    assign acc_out = acc;

endmodule
