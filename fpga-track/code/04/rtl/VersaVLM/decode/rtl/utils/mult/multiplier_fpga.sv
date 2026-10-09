module multiplier_fpga #(
	parameter MUL_CYCLE = 3
) (
	input  logic    		clk_i,
  input  logic    		rst_n_i,

	input  logic [31:0] operand_a_i,
	input  logic [31:0] operand_b_i,

	input  logic        operand_a_signed_i,
	input  logic        operand_b_signed_i,

	output logic [31:0] res_low_32_o,
	output logic [31:0] res_high_32_o,

	input  logic    		mul_start_i,
	output logic        mul_finish_o
);
	// multiplier state
	logic [2:0] mul_cnt;
	
	always @(posedge clk_i or negedge rst_n_i) begin
	  if (!rst_n_i) begin
	    mul_cnt <= '0;
	  end else begin
	    if (mul_start_i) begin
	      mul_cnt <= 3'd1;        // start count
	    end else begin
	      if (mul_cnt == MUL_CYCLE - 1)
	        mul_cnt <= '0;          // multiplier finish
	      else if (mul_cnt != '0)
	        mul_cnt <= mul_cnt + 1; // multiplier calculating
	      else
	        mul_cnt <= '0;          // multiplier idle
	    end
	  end
	end

  assign mul_finish_o = mul_cnt == MUL_CYCLE - 1;

  // multiplier
	logic [63:0] mul_res_ss, mul_res_su, mul_res_uu;

  mult_gen_ss u_multiplier_fpga_32_ss (
    .CLK ( clk_i       ),  // input wire CLK
    .A   ( operand_a_i ),  // input wire [31 : 0] A
    .B   ( operand_b_i ),  // input wire [31 : 0] B
    .P   ( mul_res_ss  )   // output wire [63 : 0] P
  );
  
  mult_gen_su u_multiplier_fpga_32_su (
    .CLK ( clk_i       ),  // input wire CLK
    .A   ( operand_a_i ),  // input wire [31 : 0] A
    .B   ( operand_b_i ),  // input wire [31 : 0] B
    .P   ( mul_res_su  )   // output wire [63 : 0] P
  );
  
  mult_gen_uu u_multiplier_fpga_32_uu (
    .CLK ( clk_i       ),  // input wire CLK
    .A   ( operand_a_i ),  // input wire [31 : 0] A
    .B   ( operand_b_i ),  // input wire [31 : 0] B
    .P   ( mul_res_uu  )   // output wire [63 : 0] P
  );
  
  always @(*) begin
      if (operand_a_signed_i & operand_b_signed_i) begin
          res_low_32_o  = mul_res_ss[31:0];
          res_high_32_o = mul_res_ss[63:32];
      end else if (operand_a_signed_i & ~operand_b_signed_i) begin
          res_low_32_o  = mul_res_su[31:0];
          res_high_32_o = mul_res_su[63:32];
      end else begin
          res_low_32_o  = mul_res_uu[31:0];
          res_high_32_o = mul_res_uu[63:32];
      end
  end

endmodule