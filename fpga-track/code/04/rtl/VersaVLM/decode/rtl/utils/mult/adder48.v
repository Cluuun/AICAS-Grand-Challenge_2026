/////////////////////////////////////
//                                 //
//   48 bit carry select Adder     //
//                                 //
/////////////////////////////////////

module adder48 (
	input   [47:0]    a,
	input   [47:0]    b,
	output  [47:0]    sum,
	output            cout
	);

	wire 	[3:0]		p2, g2, c2_s0, c2;
	wire 	[4:0]		p3, g3, c3_s0, c3_s1, c3;
 	wire 	[5:0]		p4, g4, c4_s0, c4_s1, c4;
 	wire 	[6:0]		p5, g5, c5_s0, c5_s1, c5;
 	wire 	[7:0]		p6, g6, c6_s0, c6_s1, c6; 
 	wire 	[8:0]		p7, g7, c7_s0, c7_s1, c7;
 	wire 	[8:0]		p8, g8, c8_s0, c8_s1, c8;   

///////////////////// 3:0 bit adder ///////////////////////////////
	assign	g2 = a[3 : 0] & b[3 : 0];
	assign	p2 = a[3 : 0] ^ b[3 : 0];
	assign	c2_s0[0] = g2[0];
	assign 	c2_s0[1] = g2[1] | (p2[1] & c2_s0[0]);
	assign 	c2_s0[2] = g2[2] | (p2[2] & c2_s0[1]);
	assign	c2_s0[3] = g2[3] | (p2[3] & c2_s0[2]);
	assign	c2 = c2_s0;
	assign	sum[3 : 0] = p2 ^ {c2[2:0], 1'b0};

///////////////////// 8:4 bit adder ///////////////////////////////
	assign	g3 = a[8 : 4] & b[8 : 4];
	assign	p3 = a[8 : 4] ^ b[8 : 4];
	assign	c3_s0[0] = g3[0];
	assign 	c3_s0[1] = g3[1] | (p3[1] & c3_s0[0]);
	assign 	c3_s0[2] = g3[2] | (p3[2] & c3_s0[1]);
	assign	c3_s0[3] = g3[3] | (p3[3] & c3_s0[2]);
	assign	c3_s0[4] = g3[4] | (p3[4] & c3_s0[3]);
	assign	c3_s1[0] = g3[0] | p3[0];
	assign	c3_s1[1] = g3[1] | (p3[1] & c3_s1[0]);
	assign	c3_s1[2] = g3[2] | (p3[2] & c3_s1[1]);
	assign	c3_s1[3] = g3[3] | (p3[3] & c3_s1[2]);
	assign	c3_s1[4] = g3[4] | (p3[4] & c3_s1[3]);
	assign	c3 = c2[3] ? c3_s1 : c3_s0;
	assign	sum[8 : 4] = p3 ^ {c3[3:0], c2[3]};

///////////////////// 14:9 bit adder ///////////////////////////////
	assign	g4 = a[14 : 9] & b[14 : 9];
	assign	p4 = a[14 : 9] ^ b[14 : 9];
	assign	c4_s0[0] = g4[0];
	assign 	c4_s0[1] = g4[1] | (p4[1] & c4_s0[0]);
	assign 	c4_s0[2] = g4[2] | (p4[2] & c4_s0[1]);
	assign	c4_s0[3] = g4[3] | (p4[3] & c4_s0[2]);
	assign	c4_s0[4] = g4[4] | (p4[4] & c4_s0[3]);
	assign	c4_s0[5] = g4[5] | (p4[5] & c4_s0[4]);
	assign	c4_s1[0] = g4[0] | p4[0];
	assign	c4_s1[1] = g4[1] | (p4[1] & c4_s1[0]);
	assign	c4_s1[2] = g4[2] | (p4[2] & c4_s1[1]);
	assign	c4_s1[3] = g4[3] | (p4[3] & c4_s1[2]);
	assign	c4_s1[4] = g4[4] | (p4[4] & c4_s1[3]);
	assign	c4_s1[5] = g4[5] | (p4[5] & c4_s1[4]);
	assign	c4 = c3[4] ? c4_s1 : c4_s0;
	assign	sum[14 : 9] = p4 ^ {c4[4:0], c3[4]};

///////////////////// 21:15 bit adder ///////////////////////////////
	assign	g5 = a[21 : 15] & b[21 : 15];
	assign	p5 = a[21 : 15] ^ b[21 : 15];
	assign	c5_s0[0] = g5[0];
	assign 	c5_s0[1] = g5[1] | (p5[1] & c5_s0[0]);
	assign 	c5_s0[2] = g5[2] | (p5[2] & c5_s0[1]);
	assign	c5_s0[3] = g5[3] | (p5[3] & c5_s0[2]);
	assign	c5_s0[4] = g5[4] | (p5[4] & c5_s0[3]);
	assign	c5_s0[5] = g5[5] | (p5[5] & c5_s0[4]);
	assign	c5_s0[6] = g5[6] | (p5[6] & c5_s0[5]);
	assign	c5_s1[0] = g5[0] | p5[0];
	assign	c5_s1[1] = g5[1] | (p5[1] & c5_s1[0]);
	assign	c5_s1[2] = g5[2] | (p5[2] & c5_s1[1]);
	assign	c5_s1[3] = g5[3] | (p5[3] & c5_s1[2]);
	assign	c5_s1[4] = g5[4] | (p5[4] & c5_s1[3]);
	assign	c5_s1[5] = g5[5] | (p5[5] & c5_s1[4]);
	assign	c5_s1[6] = g5[6] | (p5[6] & c5_s1[5]);
	assign	c5 = c4[5] ? c5_s1 : c5_s0;
	assign	sum[21 : 15] = p5 ^ {c5[5:0], c4[5]};	

///////////////////// 29:22 bit adder ///////////////////////////////
	assign	g6 = a[29 : 22] & b[29 : 22];
	assign	p6 = a[29 : 22] ^ b[29 : 22];
	assign	c6_s0[0] = g6[0];
	assign 	c6_s0[1] = g6[1] | (p6[1] & c6_s0[0]);
	assign 	c6_s0[2] = g6[2] | (p6[2] & c6_s0[1]);
	assign	c6_s0[3] = g6[3] | (p6[3] & c6_s0[2]);
	assign	c6_s0[4] = g6[4] | (p6[4] & c6_s0[3]);
	assign	c6_s0[5] = g6[5] | (p6[5] & c6_s0[4]);
	assign	c6_s0[6] = g6[6] | (p6[6] & c6_s0[5]);
	assign	c6_s0[7] = g6[7] | (p6[7] & c6_s0[6]);
	assign	c6_s1[0] = g6[0] | p6[0];
	assign	c6_s1[1] = g6[1] | (p6[1] & c6_s1[0]);
	assign	c6_s1[2] = g6[2] | (p6[2] & c6_s1[1]);
	assign	c6_s1[3] = g6[3] | (p6[3] & c6_s1[2]);
	assign	c6_s1[4] = g6[4] | (p6[4] & c6_s1[3]);
	assign	c6_s1[5] = g6[5] | (p6[5] & c6_s1[4]);
	assign	c6_s1[6] = g6[6] | (p6[6] & c6_s1[5]);
	assign	c6_s1[7] = g6[7] | (p6[7] & c6_s1[6]);
	assign	c6 = c5[6] ? c6_s1 : c6_s0;
	assign	sum[29 : 22] = p6 ^ {c6[6:0], c5[6]};

///////////////////// 38:30 bit adder ///////////////////////////////
	assign	g7 = a[38 : 30] & b[38 : 30];
	assign	p7 = a[38 : 30] ^ b[38 : 30];
	assign	c7_s0[0] = g7[0];
	assign 	c7_s0[1] = g7[1] | (p7[1] & c7_s0[0]);
	assign 	c7_s0[2] = g7[2] | (p7[2] & c7_s0[1]);
	assign	c7_s0[3] = g7[3] | (p7[3] & c7_s0[2]);
	assign	c7_s0[4] = g7[4] | (p7[4] & c7_s0[3]);
	assign	c7_s0[5] = g7[5] | (p7[5] & c7_s0[4]);
	assign	c7_s0[6] = g7[6] | (p7[6] & c7_s0[5]);
	assign	c7_s0[7] = g7[7] | (p7[7] & c7_s0[6]);
	assign	c7_s0[8] = g7[8] | (p7[8] & c7_s0[7]);
	assign	c7_s1[0] = g7[0] | p7[0];
	assign	c7_s1[1] = g7[1] | (p7[1] & c7_s1[0]);
	assign	c7_s1[2] = g7[2] | (p7[2] & c7_s1[1]);
	assign	c7_s1[3] = g7[3] | (p7[3] & c7_s1[2]);
	assign	c7_s1[4] = g7[4] | (p7[4] & c7_s1[3]);
	assign	c7_s1[5] = g7[5] | (p7[5] & c7_s1[4]);
	assign	c7_s1[6] = g7[6] | (p7[6] & c7_s1[5]);
	assign	c7_s1[7] = g7[7] | (p7[7] & c7_s1[6]);
	assign	c7_s1[8] = g7[8] | (p7[8] & c7_s1[7]);
	assign	c7 = c6[7] ? c7_s1 : c7_s0;
	assign	sum[38 : 30] = p7 ^ {c7[7:0], c6[7]};

///////////////////// 47:39 bit adder ///////////////////////////////
	assign	g8 = a[47 : 39] & b[47 : 39];
	assign	p8 = a[47 : 39] ^ b[47 : 39];
	assign	c8_s0[0] = g8[0];
	assign 	c8_s0[1] = g8[1] | (p8[1] & c8_s0[0]);
	assign 	c8_s0[2] = g8[2] | (p8[2] & c8_s0[1]);
	assign	c8_s0[3] = g8[3] | (p8[3] & c8_s0[2]);
	assign	c8_s0[4] = g8[4] | (p8[4] & c8_s0[3]);
	assign	c8_s0[5] = g8[5] | (p8[5] & c8_s0[4]);
	assign	c8_s0[6] = g8[6] | (p8[6] & c8_s0[5]);
	assign	c8_s0[7] = g8[7] | (p8[7] & c8_s0[6]);
	assign	c8_s0[8] = g8[8] | (p8[8] & c8_s0[7]);
	assign	c8_s1[0] = g8[0] | p8[0];
	assign	c8_s1[1] = g8[1] | (p8[1] & c8_s1[0]);
	assign	c8_s1[2] = g8[2] | (p8[2] & c8_s1[1]);
	assign	c8_s1[3] = g8[3] | (p8[3] & c8_s1[2]);
	assign	c8_s1[4] = g8[4] | (p8[4] & c8_s1[3]);
	assign	c8_s1[5] = g8[5] | (p8[5] & c8_s1[4]);
	assign	c8_s1[6] = g8[6] | (p8[6] & c8_s1[5]);
	assign	c8_s1[7] = g8[7] | (p8[7] & c8_s1[6]);
	assign	c8_s1[8] = g8[8] | (p8[8] & c8_s1[7]);
	assign	c8 = c7[8] ? c8_s1 : c8_s0;
	assign	sum[47 : 39] = p8 ^ {c8[7:0], c7[8]};

	assign cout = c8[8];	
		
endmodule