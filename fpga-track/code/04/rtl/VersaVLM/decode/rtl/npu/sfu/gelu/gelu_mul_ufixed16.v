module gelu_mul_ufixed16 #(
    parameter DOUT_WIDTH         = 32,
    parameter DIN0_INTEGER_WIDTH = 16,
    parameter DIN1_INTEGER_WIDTH = 16,
    parameter DOUT_INTEGER_WIDTH = 16
)(
    clk,
    rst_n,
    din0,
    din1,
    dout
);
input                       clk;
input                       rst_n;
input  [16 - 1 : 0] din0;
input  [16 - 1 : 0] din1;
output [DOUT_WIDTH - 1 : 0] dout;

reg   [31 : 0] dout_tmp;

assign dout = dout_tmp[32 - DIN0_INTEGER_WIDTH - DIN1_INTEGER_WIDTH + DOUT_INTEGER_WIDTH - 1 : 
                          32 - DIN0_INTEGER_WIDTH - DIN1_INTEGER_WIDTH + DOUT_INTEGER_WIDTH - DOUT_WIDTH];

// multi_16 u_multi_16 (
// 	.clk        (clk),
// 	.rst_n      (rst_n),
// 	.a_in       (din0),
// 	.b_in       (din1),
// 	.mode       (2'b00),
// 	.res_out    (dout_tmp)
// );

always @(posedge clk) begin
    dout_tmp <= $unsigned(din0) * $unsigned(din1);
end

endmodule

