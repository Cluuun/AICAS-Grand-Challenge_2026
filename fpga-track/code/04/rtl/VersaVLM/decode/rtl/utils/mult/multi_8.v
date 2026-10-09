/////////////////////////////////////
//                                 //
//        8x8 Multiplier           //
//                                 //
/////////////////////////////////////

module multi_8 (
	input    [7:0]  a_in,
	input    [7:0]  b_in,
	input           sign, // sign = 1, a b signed, sign = 0 , a b unsigned
	output   [15:0] res_out
);


//////////// partial products ////////////
	wire  [11:0]  pp0;
	wire  [10:0]  pp1,pp2;
	wire  [9:0]   pp3;
	wire  [7:0]   pp4;

	booth2_8 booth(
		.a_in(a_in),
		.b_in(b_in),
		.sign_a(sign),
		.pp0 (pp0),
		.pp1 (pp1),
		.pp2 (pp2),
		.pp3 (pp3)
		);

	assign pp4 = (sign)?      0 : 
				 (b_in[7]) ?  a_in : 0;


///////////// wallace tree ////////////////
	wire [15:0] adda;
	wire [15:0] addb;

	wallace_8 tree(
		.pp0 (pp0),
		.pp1 (pp1),
		.pp2 (pp2),
		.pp3 (pp3),
		.pp4 (pp4),
		.adda(adda),
		.addb(addb)
		);

///////////// 16+16 adder ////////////////

	wire [15:0] res_adda;
	
	adder16 adder0(
		.a(adda),
		.b(addb),
		.sum(res_adda),
		.cout()
		);

	assign res_out = (a_in == 0) ? 0   : res_adda;

endmodule