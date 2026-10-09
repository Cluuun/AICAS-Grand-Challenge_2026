/////////////////////////////////////
//                                 
//        16x16 Multiplier         
//                                 
/////////////////////////////////////

module multi_16 (
	input	 		 clk,
	input			 rst_n,
	input    [15:0]  a_in,
	input    [15:0]  b_in,
	input    [1:0]   mode,  // mode = 'b10, sign a b            mode = 'b00 , unsign a b     
						    // mode = 'b01, sign a   unsign b , mode = 'b11  illegal input
	output   [31:0]  res_out
);


//////////// partial products ////////////
	wire  [19:0]  pp0;
	wire  [18:0]  pp1,pp2,pp3,pp4,pp5,pp6;
	wire  [17:0]  pp7;
	wire  [15:0]  pp8;
 	wire 		  sign_a;

	assign sign_a = mode[1] | mode[0];   

	booth2_16 booth(
		.a_in(a_in),
		.b_in(b_in),
		.sign_a(sign_a),
		.pp0 (pp0),
		.pp1 (pp1),
		.pp2 (pp2),
		.pp3 (pp3),
		.pp4 (pp4),
		.pp5 (pp5),
		.pp6 (pp6),
		.pp7 (pp7)
		);

	assign pp8 = (mode[1])?   0 : 
				 (b_in[15])?  a_in : 0;

reg [19:0] rpp0;
reg [18:0] rpp1, rpp2, rpp3, rpp4, rpp5, rpp6;
reg [17:0] rpp7;
reg [15:0] rpp8;

always @(posedge clk or negedge rst_n) begin
	if(!rst_n) begin
		rpp0 <= 'b0;
		rpp1 <= 'b0;
		rpp2 <= 'b0;
		rpp3 <= 'b0;
		rpp4 <= 'b0;
		rpp5 <= 'b0;
		rpp6 <= 'b0;
		rpp7 <= 'b0;
		rpp8 <= 'b0;
	end	else begin
		rpp0 <= pp0;
		rpp1 <= pp1;
		rpp2 <= pp2;
		rpp3 <= pp3;
		rpp4 <= pp4;
		rpp5 <= pp5;
		rpp6 <= pp6;
		rpp7 <= pp7;
		rpp8 <= pp8;
	end
end

///////////// wallace tree ////////////////
	wire [31:0] adda;
	wire [31:0] addb;


	wallace_16 tree(
		.pp0 (rpp0),
		.pp1 (rpp1),
		.pp2 (rpp2),
		.pp3 (rpp3),
		.pp4 (rpp4),
		.pp5 (rpp5),
		.pp6 (rpp6),
		.pp7 (rpp7),
		.pp8 (rpp8),
		.adda(adda),
		.addb(addb)
		);

///////////// 32+32 adder ////////////////

	wire [31:0] res_adda;
	
	adder32 adder(
		.a(adda),
		.b(addb),
		.sum(res_adda),
		.cout()
		);

	assign res_out = (a_in == 0) ?  0 : res_adda;

endmodule