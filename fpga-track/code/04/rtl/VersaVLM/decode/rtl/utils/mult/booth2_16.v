/////////////////////////////////////////////////
//                                             //
//   The partial products generator based on   //
//         Booth2 algorithm 16x16              //
//                                             //
/////////////////////////////////////////////////

module booth2_16(
	input    [15:0]  a_in,        // multiplicand
	input    [15:0]  b_in,        // multiplier
	input            sign_a,      // sign = 1, a signed,    sign = 0 , a unsigned
	output   [19:0]  pp0,         // partial peoduct 0
	output   [18:0]  pp1,         // partial peoduct 1
	output   [18:0]  pp2,         // partial peoduct 2
	output   [18:0]  pp3,         // partial peoduct 3
	output   [18:0]  pp4,         // partial peoduct 4
	output   [18:0]  pp5,         // partial peoduct 5
	output   [18:0]  pp6,         // partial peoduct 6
	output   [17:0]  pp7          // partial peoduct 7
);

	wire [2:0] part [7:0];       // part of multiplier, 0 is added at LSB 
	assign part[0] = {b_in[1:0],1'b0};
	assign part[1] = {b_in[3:1]};
	assign part[2] = {b_in[5:3]};
	assign part[3] = {b_in[7:5]};
	assign part[4] = {b_in[9:7]};
	assign part[5] = {b_in[11:9]};
	assign part[6] = {b_in[13:11]};
	assign part[7] = {b_in[15:13]};

	wire [7:0] sign;
	assign sign[0] = ((&part[0]) | ~(|part[0]))?    1 :
					 (sign_a)?  ~(b_in[1] ^ a_in[15]) :
					 ~b_in[1];
	assign sign[1] = ((&part[1]) | ~(|part[1]))?    1 :
					 (sign_a)?  ~(b_in[3] ^ a_in[15]) :
					 ~b_in[3];
	assign sign[2] = ((&part[2]) | ~(|part[2]))?    1 :
					 (sign_a)?  ~(b_in[5] ^ a_in[15]) :
					 ~b_in[5];
	assign sign[3] = ((&part[3]) | ~(|part[3]))?    1 :
					 (sign_a)?  ~(b_in[7] ^ a_in[15]) :
					 ~b_in[7];
	assign sign[4] = ((&part[4]) | ~(|part[4]))?    1 :
					 (sign_a)?  ~(b_in[9] ^ a_in[15]) :
					 ~b_in[9];
	assign sign[5] = ((&part[5]) | ~(|part[5]))?    1 :
					 (sign_a)?  ~(b_in[11] ^ a_in[15]):
					 ~b_in[11];
	assign sign[6] = ((&part[6]) | ~(|part[6]))?    1 :
					 (sign_a)?  ~(b_in[13] ^ a_in[15]):
					 ~b_in[13];
	assign sign[7] = ((&part[7]) | ~(|part[7]))?    1 :
					 (sign_a)?  ~(b_in[15] ^ a_in[15]):
					 ~b_in[15];

///////////// caculate pp0 ////////////////
	cal_pp_16 pp0_cal(
		.a_in(a_in),
		.b_in(part[0]),
		.sign   (sign_a),
		.res_out(pp0[16:0])
		);

	assign pp0[19:17] = {sign[0],~sign[0],~sign[0]};

///////////// caculate pp1 ////////////////
	cal_pp_16 pp1_cal(
		.a_in(a_in),
		.b_in(part[1]),
		.sign   (sign_a),
		.res_out(pp1[16:0])
		);

	assign pp1[18:17] = {1'b1,sign[1]};

///////////// caculate pp2 ////////////////
	cal_pp_16 pp2_cal(
		.a_in(a_in),
		.b_in(part[2]),
		.sign   (sign_a),
		.res_out(pp2[16:0])
		);

	assign pp2[18:17] = {1'b1,sign[2]};

///////////// caculate pp3 ////////////////
	cal_pp_16 pp3_cal(
		.a_in(a_in),
		.b_in(part[3]),
		.sign   (sign_a),
		.res_out(pp3[16:0])
		);

	assign pp3[18:17] = {1'b1,sign[3]};

///////////// caculate pp4 ////////////////
	cal_pp_16 pp4_cal(
		.a_in(a_in),
		.b_in(part[4]),
		.sign   (sign_a),
		.res_out(pp4[16:0])
		);

	assign pp4[18:17] = {1'b1,sign[4]};

///////////// caculate pp5 ////////////////
	cal_pp_16 pp5_cal(
		.a_in(a_in),
		.b_in(part[5]),
		.sign   (sign_a),
		.res_out(pp5[16:0])
		);

	assign pp5[18:17] = {1'b1,sign[5]};

///////////// caculate pp6 ////////////////
	cal_pp_16 pp6_cal(
		.a_in(a_in),
		.b_in(part[6]),
		.sign   (sign_a),
		.res_out(pp6[16:0])
		);

	assign pp6[18:17] = {1'b1,sign[6]};

///////////// caculate pp7 ////////////////
	cal_pp_16 pp7_cal(
		.a_in(a_in),
		.b_in(part[7]),
		.sign   (sign_a),
		.res_out(pp7[16:0])
		);

	assign pp7[17] = {sign[7]};


endmodule