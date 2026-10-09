//------------------------------------------------------------------------------
// vmath_fp32_add.sv
//------------------------------------------------------------------------------
// Thin FP32 adder wrapper that reuses the existing add_sub implementation.
//------------------------------------------------------------------------------
module vmath_fp32_add (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic [31:0] z_o
);

  add_sub #(
    .K(32),
    .P(24),
    .E(8)
  ) u_add_sub (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    .FP_A    (a_i),
    .FP_B    (b_i),
    .add_sub (1'b1),
    .FP_Z    (z_o)
  );

endmodule
