/////////////////////////////////////
//                                 //
//   18 bit carry select Adder     //
//                                 //
/////////////////////////////////////

module adder18 (
	input   [17:0]    a,
	input   [17:0]    b,
	output  [17:0]    sum
	);

	wire	[2 : 0]		p1, g1, c1_s0, c1;          //c1_s1,
	wire 	[3 : 0]		p2, g2, c2_s0, c2_s1, c2;
	wire 	[4 : 0]		p3, g3, c3_s0, c3_s1, c3;
//	wire 	[5 : 0]		p4, g4, c4_s0, c4_s1, c4;

///////////////////// 2:0 bit adder ///////////////////////////////
	assign	g1 = a[2:0] & b[2:0];
	assign	p1 = a[2:0]	^ b[2:0];
	assign	c1_s0[0] = g1[0];
	assign 	c1_s0[1] = g1[1] | (p1[1] & c1_s0[0]);
	assign 	c1_s0[2] = g1[2] | (p1[2] & c1_s0[1]);
//	assign	c1_s1[0] = g1[0] | p1[0];
//	assign	c1_s1[1] = g1[1] | (p1[1] & c1_s1[0]);
//	assign	c1_s1[2] = g1[2] | (p1[2] & c1_s1[1]);
//	assign	c1 = cin ? c1_s1 : c1_s0;
	assign	c1 = c1_s0;
	assign	sum[2 : 0] = p1 ^ {c1[1:0], 1'b0};
//	assign	sum[2 : 0] = p1 ^ {c1[1:0], cin};
///////////////////// 6:3 bit adder ///////////////////////////////
	assign	g2 = a[6 : 3] & b[6 : 3];
	assign	p2 = a[6 : 3] ^ b[6 : 3];
	assign	c2_s0[0] = g2[0];
	assign 	c2_s0[1] = g2[1] | (p2[1] & c2_s0[0]);
	assign 	c2_s0[2] = g2[2] | (p2[2] & c2_s0[1]);
	assign	c2_s0[3] = g2[3] | (p2[3] & c2_s0[2]);
	assign	c2_s1[0] = g2[0] | p2[0];
	assign	c2_s1[1] = g2[1] | (p2[1] & c2_s1[0]);
	assign	c2_s1[2] = g2[2] | (p2[2] & c2_s1[1]);
	assign	c2_s1[3] = g2[3] | (p2[3] & c2_s1[2]);
	assign	c2 = c1[2] ? c2_s1 : c2_s0;
	assign	sum[6 : 3] = p2 ^ {c2[2:0], c1[2]};
///////////////////// 11:7 bit adder ///////////////////////////////
	assign	g3 = a[11 : 7] & b[11 : 7];
	assign	p3 = a[11 : 7] ^ b[11 : 7];
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
	assign	sum[11 : 7] = p3 ^ {c3[3:0], c2[3]};
///////////////////// 17:12 bit adder ///////////////////////////////

	wire 	[5:0]		p4;
	wire    [4:0]       g4, c4_s0, c4_s1, c4;

	assign	g4 = a[17 : 12] & b[17 : 12];
	assign	p4 = a[17 : 12] ^ b[17 : 12];
	assign	c4_s0[0] = g4[0];
	assign 	c4_s0[1] = g4[1] | (p4[1] & c4_s0[0]);
	assign 	c4_s0[2] = g4[2] | (p4[2] & c4_s0[1]);
	assign	c4_s0[3] = g4[3] | (p4[3] & c4_s0[2]);
	assign	c4_s0[4] = g4[4] | (p4[4] & c4_s0[3]);
//	assign	c4_s0[5] = g4[5] | (p4[5] & c4_s0[4]);
	assign	c4_s1[0] = g4[0] | p4[0];
	assign	c4_s1[1] = g4[1] | (p4[1] & c4_s1[0]);
	assign	c4_s1[2] = g4[2] | (p4[2] & c4_s1[1]);
	assign	c4_s1[3] = g4[3] | (p4[3] & c4_s1[2]);
	assign	c4_s1[4] = g4[4] | (p4[4] & c4_s1[3]);
//	assign	c4_s1[5] = g4[5] | (p4[5] & c4_s1[4]);
	assign	c4 = c3[4] ? c4_s1 : c4_s0;
	assign	sum[17 : 12] = p4 ^ {c4, c3[4]};	
		
endmodule