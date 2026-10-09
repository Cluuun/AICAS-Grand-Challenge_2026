/////////////////////////////////////
//                                 
//  Caculate the 16x16 partial product   
//                                 
/////////////////////////////////////

module cal_pp_16 (
	input  [15:0] a_in,   // multiplicand
	input  [2:0]  b_in,   // 3bit of multiplier 
	input         sign,   // sign = 1, a signed,    sign = 0 , a unsigned
	output [16:0] res_out	
);

	wire a_sign;
	reg [16:0] res_reg;

	assign a_sign = (sign) ?   a_in[15] : 0;

	always @(*)
	begin
		case (b_in)
			3'b001  : res_reg = {a_sign,a_in};
			3'b010  : res_reg = {a_sign,a_in};
			3'b011  : res_reg = {a_in[15],a_in<<1};
			3'b100  : res_reg = ~{a_in[15],a_in<<1} + 1'b1;
			3'b101  : res_reg = ~{a_sign,a_in} + 1'b1;
			3'b110  : res_reg = ~{a_sign,a_in} + 1'b1;
			default : res_reg = 0;
		endcase
	end

	assign res_out = res_reg;

endmodule