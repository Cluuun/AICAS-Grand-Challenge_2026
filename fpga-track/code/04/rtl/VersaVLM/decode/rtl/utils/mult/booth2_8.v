/////////////////////////////////////////////////
//                                             //
//   The partial products generator based on   //
//         Booth2 algorithm 8x8                //
//                                             //
/////////////////////////////////////////////////

module booth2_8(
	input    [7:0]  a_in,        // multiplicand
	input    [7:0]  b_in,        // multiplier
	input           sign_a,      // sign = 1, a signed,    sign = 0 , a unsigned
	output   [11:0] pp0,         // partial peoduct 0
	output   [10:0] pp1,         // partial peoduct 1
	output   [10:0] pp2,         // partial peoduct 2
	output   [9:0]  pp3          // partial peoduct 3
);

	wire [2:0] part [3:0];       // part of multiplier, 0 is added at LSB 
	assign part[0] = {b_in[1:0],1'b0};
	assign part[1] = {b_in[3:1]};
	assign part[2] = {b_in[5:3]};
	assign part[3] = {b_in[7:5]};

	wire [3:0] sign;
	assign sign[0] = ((&part[0]) | ~(|part[0]))?   1 :
					 (sign_a)?  ~(b_in[1] ^ a_in[7]) :
					 ~b_in[1];
	assign sign[1] = ((&part[1]) | ~(|part[1]))?   1 :
					 (sign_a)?  ~(b_in[3] ^ a_in[7]) :
					 ~b_in[3];
	assign sign[2] = ((&part[2]) | ~(|part[2]))?   1 :
					 (sign_a)?  ~(b_in[5] ^ a_in[7]) :
					 ~b_in[5];
	assign sign[3] = ((&part[3]) | ~(|part[3]))?   1 :
					 (sign_a)?  ~(b_in[7] ^ a_in[7]) :
					 ~b_in[7];				 
//	assign sign[3] = (part[3] == 3'b000)?   1        :
//					 (part[3] == 3'b111)?   1        :
//					 (sign_a)?  ~(b_in[7] ^ a_in[7]) :
//					 ~b_in[7];

///////////// caculate pp0 ////////////////
	cal_pp_8 pp0_cal(
		.a_in(a_in),
		.b_in(part[0]),
		.sign   (sign_a),
		.res_out(pp0[8:0])
		);

	assign pp0[11:9] = {sign[0],~sign[0],~sign[0]};

///////////// caculate pp1 ////////////////
	cal_pp_8 pp1_cal(
		.a_in(a_in),
		.b_in(part[1]),
		.sign   (sign_a),
		.res_out(pp1[8:0])
		);

	assign pp1[10:9] = {1'b1,sign[1]};

///////////// caculate pp2 ////////////////
	cal_pp_8 pp2_cal(
		.a_in(a_in),
		.b_in(part[2]),
		.sign   (sign_a),
		.res_out(pp2[8:0])
		);

	assign pp2[10:9] = {1'b1,sign[2]};

///////////// caculate pp3 ////////////////
	cal_pp_8 pp3_cal(
		.a_in(a_in),
		.b_in(part[3]),
		.sign   (sign_a),
		.res_out(pp3[8:0])
		);

	assign pp3[9] = {sign[3]};


endmodule