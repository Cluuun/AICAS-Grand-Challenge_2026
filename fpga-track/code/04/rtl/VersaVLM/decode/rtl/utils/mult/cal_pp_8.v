/////////////////////////////////////
//                                 
//  Caculate the 8x8 partial product   
//                                 
/////////////////////////////////////

module cal_pp_8 (
	input  [7:0]  a_in,   // multiplicand
	input  [2:0]  b_in,   // 3bit of multiplier 
	input         sign,   // sign = 1, a signed,    sign = 0 , a unsigned
	output [8:0]  res_out	
);
	wire       a_sign;
	reg  [8:0] res_reg;

	assign a_sign = (sign) ?   a_in[7] : 0;

/*	assign res_out = (b_in == 3'b001) ?  {a_sign,a_in}             :
					 (b_in == 3'b010) ?  {a_sign,a_in}             :
					 (b_in == 3'b011) ?  {a_in[7],a_in<<1}         :
					 (b_in == 3'b100) ?  ~{a_in[7],a_in<<1} + 1'b1 :
					 (b_in == 3'b101) ?  ~{a_sign,a_in} + 1'b1     :
					 (b_in == 3'b110) ?  ~{a_sign,a_in} + 1'b1     :
					 0;*/

	assign res_out = res_reg;

	always @(*)
	begin
		case (b_in)
			3'b001  : res_reg = {a_sign,a_in};
			3'b010  : res_reg = {a_sign,a_in};
			3'b011  : res_reg = {a_in[7],a_in<<1};
			3'b100  : res_reg = ~{a_in[7],a_in<<1} + 1'b1;
			3'b101  : res_reg = ~{a_sign,a_in} + 1'b1;
			3'b110  : res_reg = ~{a_sign,a_in} + 1'b1;
			default : res_reg = 0;
		endcase
	end

endmodule