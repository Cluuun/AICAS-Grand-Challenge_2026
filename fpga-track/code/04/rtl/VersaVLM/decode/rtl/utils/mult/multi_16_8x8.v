///////////////////////////////////////////////////////////
//                                 
//     4 8x8 Multipliers combine as a 16x16 Multiplier         
//                                 
/////////////////////////////////////////////////////////// 

module multi_16_8x8(
	input     [31:0]  opa,
	input     [31:0]  opb,
	input     [2:0]   mode,     // mode = 3'b000 mul   , mode = 3'b001 mulh , 
								// mode = 3'b010 mulhu , mode = 3'b011 mulhsu , 
								// mode = 3'b101 8-bit dot product , mode = 3'b110 16-bit dot product , else illegal
	output    [31:0]  res_16,
	output    [31:0]  res_dot
	);


	wire [15:0] res_mult8_3;
	wire [15:0] res_mult8_2;
	wire [15:0] res_mult8_1;
	wire [15:0] res_mult8_0;
	wire [7:0]  mult8_3_a,mult8_3_b,mult8_2_a,mult8_2_b;
	wire [7:0]  mult8_1_a;
	wire        sign_mult8;

	assign sign_mult8 = (mode[2])? 1 : 0;

	assign mult8_3_a = (mode[2])? opa[31:24] : opa[15:8];
	assign mult8_3_b = (mode[2])? opb[31:24] : opb[15:8];

	multi_8 multi8_3(
		.a_in(mult8_3_a),
		.b_in(mult8_3_b),
		.sign   (sign_mult8),
		.res_out(res_mult8_3)
		);

	assign mult8_2_a = (mode[2])? opa[23:16] : opa[15:8];
	assign mult8_2_b = (mode[2])? opb[23:16] : opb[7:0]; 

	multi_8 multi8_2(
		.a_in(mult8_2_a),
		.b_in(mult8_2_b),
		.sign   (sign_mult8),
		.res_out(res_mult8_2)
		); 

	assign mult8_1_a = (mode[2])? opa[15:8] : opa[7:0];

	multi_8 multi8_1(
		.a_in   (mult8_1_a),
		.b_in   (opb[15:8]),
		.sign   (sign_mult8),
		.res_out(res_mult8_1)
		); 

	multi_8 multi8_0(
		.a_in(opa[7:0]),
		.b_in(opb[7:0]),
		.sign   (sign_mult8),
		.res_out(res_mult8_0)
		); 	

//////////// caculate 8-bit dot product ////////////////

	wire [15:0] res_add17_1;
	wire [15:0] res_add17_0;
	wire        cout_add1;
	wire        cout_add0;

	adder16 adder1(
		.a   (res_mult8_3),
		.b   (res_mult8_0),
		.sum (res_add17_1),
		.cout(cout_add1)
		);

	adder16 adder0(
		.a(res_mult8_1),
		.b(res_mult8_2),
		.sum(res_add17_0),
		.cout(cout_add0)
		);

	wire [17:0] res_add18;

	adder18 adder2(
		.a({res_add17_0[15],res_add17_0[15],res_add17_0}),
		.b({res_add17_1[15],res_add17_1[15],res_add17_1}),
		.sum(res_add18)
		);

	assign res_dot[18] = res_add18[17];
	assign res_dot[19] = res_add18[17];
	assign res_dot[20] = res_add18[17];
	assign res_dot[21] = res_add18[17];
	assign res_dot[22] = res_add18[17];
	assign res_dot[23] = res_add18[17];
	assign res_dot[24] = res_add18[17];
	assign res_dot[25] = res_add18[17];
	assign res_dot[26] = res_add18[17];
	assign res_dot[27] = res_add18[17];
	assign res_dot[28] = res_add18[17];
	assign res_dot[29] = res_add18[17];
	assign res_dot[30] = res_add18[17];
	assign res_dot[31] = res_add18[17];
	assign res_dot[17:0] = res_add18;

//////////// caculate 16x16 multiply result  ////////////////

	adder32 adder3(
		.a   ({res_mult8_3,res_mult8_0}),
		.b   ({7'b0,cout_add0,res_add17_0,8'b0}),
		.sum (res_16),
		.cout()
		);

endmodule