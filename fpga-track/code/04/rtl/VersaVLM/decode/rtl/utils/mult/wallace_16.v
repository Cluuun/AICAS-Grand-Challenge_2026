/////////////////////////////////////
//                                 
//       16x16 Wallace Tree         
//                                 
/////////////////////////////////////

module wallace_16 (
	input    [19:0] pp0,
	input    [18:0] pp1,
	input    [18:0] pp2,
	input    [18:0] pp3,
	input    [18:0] pp4,
	input    [18:0] pp5,
	input    [18:0] pp6,
	input    [17:0] pp7,
	input    [15:0] pp8,
	output   [31:0] adda,
	output   [31:0] addb
);

///////////////// first stage ////////////////////
	wire [26:0] fir_sum_1;          // sum output of adder, layer 1
	wire [26:0] fir_cout_1;         // cout output of adder, layer 1
	fulladder firadd1_0  (.a(pp0[4]), .b(pp1[2]), .cin(pp2[0]), .sum(fir_sum_1[0]), .cout(fir_cout_1[0]));
	fulladder firadd1_1  (.a(pp0[5]), .b(pp1[3]), .cin(pp2[1]), .sum(fir_sum_1[1]), .cout(fir_cout_1[1]));
	fulladder firadd1_2  (.a(pp0[6]), .b(pp1[4]), .cin(pp2[2]), .sum(fir_sum_1[2]), .cout(fir_cout_1[2]));
	fulladder firadd1_3  (.a(pp0[7]), .b(pp1[5]), .cin(pp2[3]), .sum(fir_sum_1[3]), .cout(fir_cout_1[3]));
	fulladder firadd1_4  (.a(pp0[8]), .b(pp1[6]), .cin(pp2[4]), .sum(fir_sum_1[4]), .cout(fir_cout_1[4]));
	fulladder firadd1_5  (.a(pp0[9]), .b(pp1[7]), .cin(pp2[5]), .sum(fir_sum_1[5]), .cout(fir_cout_1[5]));
	fulladder firadd1_6  (.a(pp0[10]), .b(pp1[8]), .cin(pp2[6]), .sum(fir_sum_1[6]), .cout(fir_cout_1[6]));
	fulladder firadd1_7  (.a(pp0[11]), .b(pp1[9]), .cin(pp2[7]), .sum(fir_sum_1[7]), .cout(fir_cout_1[7]));
	fulladder firadd1_8  (.a(pp0[12]), .b(pp1[10]), .cin(pp2[8]), .sum(fir_sum_1[8]), .cout(fir_cout_1[8]));
	fulladder firadd1_9  (.a(pp0[13]), .b(pp1[11]), .cin(pp2[9]), .sum(fir_sum_1[9]), .cout(fir_cout_1[9]));
	fulladder firadd1_10 (.a(pp0[14]), .b(pp1[12]), .cin(pp2[10]), .sum(fir_sum_1[10]), .cout(fir_cout_1[10]));
	fulladder firadd1_11 (.a(pp0[15]), .b(pp1[13]), .cin(pp2[11]), .sum(fir_sum_1[11]), .cout(fir_cout_1[11]));
	fulladder firadd1_12 (.a(pp0[16]), .b(pp1[14]), .cin(pp2[12]), .sum(fir_sum_1[12]), .cout(fir_cout_1[12]));
	fulladder firadd1_13 (.a(pp0[17]), .b(pp1[15]), .cin(pp2[13]), .sum(fir_sum_1[13]), .cout(fir_cout_1[13]));
	fulladder firadd1_14 (.a(pp0[18]), .b(pp1[16]), .cin(pp2[14]), .sum(fir_sum_1[14]), .cout(fir_cout_1[14]));
	fulladder firadd1_15 (.a(pp0[19]), .b(pp1[17]), .cin(pp2[15]), .sum(fir_sum_1[15]), .cout(fir_cout_1[15]));

	fulladder firadd1_16 (.a(pp1[18]), .b(pp2[16]), .cin(pp3[14]), .sum(fir_sum_1[16]), .cout(fir_cout_1[16]));

	fulladder firadd1_17 (.a(pp2[17]), .b(pp3[15]), .cin(pp4[13]), .sum(fir_sum_1[17]), .cout(fir_cout_1[17]));
	fulladder firadd1_18 (.a(pp2[18]), .b(pp3[16]), .cin(pp4[14]), .sum(fir_sum_1[18]), .cout(fir_cout_1[18]));

	fulladder firadd1_19 (.a(pp3[17]), .b(pp4[15]), .cin(pp5[13]), .sum(fir_sum_1[19]), .cout(fir_cout_1[19]));
	fulladder firadd1_20 (.a(pp3[18]), .b(pp4[16]), .cin(pp5[14]), .sum(fir_sum_1[20]), .cout(fir_cout_1[20]));

	fulladder firadd1_21 (.a(pp4[17]), .b(pp5[15]), .cin(pp6[13]), .sum(fir_sum_1[21]), .cout(fir_cout_1[21]));
	fulladder firadd1_22 (.a(pp4[18]), .b(pp5[16]), .cin(pp6[14]), .sum(fir_sum_1[22]), .cout(fir_cout_1[22]));

	fulladder firadd1_23 (.a(pp5[17]), .b(pp6[15]), .cin(pp7[13]), .sum(fir_sum_1[23]), .cout(fir_cout_1[23]));
	fulladder firadd1_24 (.a(pp5[18]), .b(pp6[16]), .cin(pp7[14]), .sum(fir_sum_1[24]), .cout(fir_cout_1[24]));

	fulladder firadd1_25 (.a(pp6[17]), .b(pp7[15]), .cin(pp8[13]), .sum(fir_sum_1[25]), .cout(fir_cout_1[25]));
	fulladder firadd1_26 (.a(pp6[18]), .b(pp7[16]), .cin(pp8[14]), .sum(fir_sum_1[26]), .cout(fir_cout_1[26]));


	wire [14:0] fir_sum_2;          // sum output of adder, layer 2
	wire [14:0] fir_cout_2;         // cout output of adder, layer 2
	fulladder firadd2_0  (.a(pp3[4]), .b(pp4[2]), .cin(pp5[0]), .sum(fir_sum_2[0]), .cout(fir_cout_2[0]));
	fulladder firadd2_1  (.a(pp3[5]), .b(pp4[3]), .cin(pp5[1]), .sum(fir_sum_2[1]), .cout(fir_cout_2[1]));
	fulladder firadd2_2  (.a(pp3[6]), .b(pp4[4]), .cin(pp5[2]), .sum(fir_sum_2[2]), .cout(fir_cout_2[2]));
	fulladder firadd2_3  (.a(pp3[7]), .b(pp4[5]), .cin(pp5[3]), .sum(fir_sum_2[3]), .cout(fir_cout_2[3]));
	fulladder firadd2_4  (.a(pp3[8]), .b(pp4[6]), .cin(pp5[4]), .sum(fir_sum_2[4]), .cout(fir_cout_2[4]));
	fulladder firadd2_5  (.a(pp3[9]), .b(pp4[7]), .cin(pp5[5]), .sum(fir_sum_2[5]), .cout(fir_cout_2[5]));
	fulladder firadd2_6  (.a(pp3[10]), .b(pp4[8]), .cin(pp5[6]), .sum(fir_sum_2[6]), .cout(fir_cout_2[6]));
	fulladder firadd2_7  (.a(pp3[11]), .b(pp4[9]), .cin(pp5[7]), .sum(fir_sum_2[7]), .cout(fir_cout_2[7]));
	fulladder firadd2_8  (.a(pp3[12]), .b(pp4[10]), .cin(pp5[8]), .sum(fir_sum_2[8]), .cout(fir_cout_2[8]));
	fulladder firadd2_9  (.a(pp3[13]), .b(pp4[11]), .cin(pp5[9]), .sum(fir_sum_2[9]), .cout(fir_cout_2[9]));

	fulladder firadd2_10 (.a(pp4[12]), .b(pp5[10]), .cin(pp6[8]), .sum(fir_sum_2[10]), .cout(fir_cout_2[10]));

	fulladder firadd2_11 (.a(pp5[11]), .b(pp6[9]), .cin(pp7[7]), .sum(fir_sum_2[11]), .cout(fir_cout_2[11]));
	fulladder firadd2_12 (.a(pp5[12]), .b(pp6[10]), .cin(pp7[8]), .sum(fir_sum_2[12]), .cout(fir_cout_2[12]));

	fulladder firadd2_13 (.a(pp6[11]), .b(pp7[9]), .cin(pp8[7]), .sum(fir_sum_2[13]), .cout(fir_cout_2[13]));
	fulladder firadd2_14 (.a(pp6[12]), .b(pp7[10]), .cin(pp8[8]), .sum(fir_sum_2[14]), .cout(fir_cout_2[14]));

	wire [4:0] fir_sum_3;          // sum output of adder, layer 3
	wire [4:0] fir_cout_3;         // cout output of adder, layer 3
	fulladder firadd3_0  (.a(pp6[4]), .b(pp7[2]), .cin(pp8[0]), .sum(fir_sum_3[0]), .cout(fir_cout_3[0]));
	fulladder firadd3_1  (.a(pp6[5]), .b(pp7[3]), .cin(pp8[1]), .sum(fir_sum_3[1]), .cout(fir_cout_3[1]));
	fulladder firadd3_2  (.a(pp6[6]), .b(pp7[4]), .cin(pp8[2]), .sum(fir_sum_3[2]), .cout(fir_cout_3[2]));
	fulladder firadd3_3  (.a(pp6[7]), .b(pp7[5]), .cin(pp8[3]), .sum(fir_sum_3[3]), .cout(fir_cout_3[3]));

	halfadder firadd3_4  (.a(pp7[6]), .b(pp8[4]), .sum(fir_sum_3[4]), .cout(fir_cout_3[4]));


///////////////// second stage ////////////////////
	wire [23:0] sec_sum_1;          // sum output of adder, layer 1
	wire [22:0] sec_cout_1;         // cout output of adder, layer 1
	fulladder secadd1_0  (.a(fir_sum_1[2]), .b(fir_cout_1[1]), .cin(pp3[0]), .sum(sec_sum_1[0]), .cout(sec_cout_1[0]));
	fulladder secadd1_1  (.a(fir_sum_1[3]), .b(fir_cout_1[2]), .cin(pp3[1]), .sum(sec_sum_1[1]), .cout(sec_cout_1[1]));
	fulladder secadd1_2  (.a(fir_sum_1[4]), .b(fir_cout_1[3]), .cin(pp3[2]), .sum(sec_sum_1[2]), .cout(sec_cout_1[2]));
	fulladder secadd1_3  (.a(fir_sum_1[5]), .b(fir_cout_1[4]), .cin(pp3[3]), .sum(sec_sum_1[3]), .cout(sec_cout_1[3]));

	fulladder secadd1_4  (.a(fir_sum_1[6]), .b(fir_cout_1[5]), .cin(fir_sum_2[0]), .sum(sec_sum_1[4]), .cout(sec_cout_1[4]));
	fulladder secadd1_5  (.a(fir_sum_1[7]), .b(fir_cout_1[6]), .cin(fir_sum_2[1]), .sum(sec_sum_1[5]), .cout(sec_cout_1[5]));
	fulladder secadd1_6  (.a(fir_sum_1[8]), .b(fir_cout_1[7]), .cin(fir_sum_2[2]), .sum(sec_sum_1[6]), .cout(sec_cout_1[6]));
	fulladder secadd1_7  (.a(fir_sum_1[9]), .b(fir_cout_1[8]), .cin(fir_sum_2[3]), .sum(sec_sum_1[7]), .cout(sec_cout_1[7]));
	fulladder secadd1_8  (.a(fir_sum_1[10]), .b(fir_cout_1[9]), .cin(fir_sum_2[4]), .sum(sec_sum_1[8]), .cout(sec_cout_1[8]));
	fulladder secadd1_9  (.a(fir_sum_1[11]), .b(fir_cout_1[10]), .cin(fir_sum_2[5]), .sum(sec_sum_1[9]), .cout(sec_cout_1[9]));
	fulladder secadd1_10 (.a(fir_sum_1[12]), .b(fir_cout_1[11]), .cin(fir_sum_2[6]), .sum(sec_sum_1[10]), .cout(sec_cout_1[10]));
	fulladder secadd1_11 (.a(fir_sum_1[13]), .b(fir_cout_1[12]), .cin(fir_sum_2[7]), .sum(sec_sum_1[11]), .cout(sec_cout_1[11]));
	fulladder secadd1_12 (.a(fir_sum_1[14]), .b(fir_cout_1[13]), .cin(fir_sum_2[8]), .sum(sec_sum_1[12]), .cout(sec_cout_1[12]));
	fulladder secadd1_13 (.a(fir_sum_1[15]), .b(fir_cout_1[14]), .cin(fir_sum_2[9]), .sum(sec_sum_1[13]), .cout(sec_cout_1[13]));

	fulladder secadd1_14 (.a(fir_sum_1[16]), .b(fir_cout_1[15]), .cin(fir_sum_2[10]), .sum(sec_sum_1[14]), .cout(sec_cout_1[14]));     //col 20

	fulladder secadd1_15 (.a(fir_sum_1[17]), .b(fir_cout_1[16]), .cin(fir_sum_2[11]), .sum(sec_sum_1[15]), .cout(sec_cout_1[15]));
	fulladder secadd1_16 (.a(fir_sum_1[18]), .b(fir_cout_1[17]), .cin(fir_sum_2[12]), .sum(sec_sum_1[16]), .cout(sec_cout_1[16]));

	fulladder secadd1_17 (.a(fir_sum_1[19]), .b(fir_cout_1[18]), .cin(fir_sum_2[13]), .sum(sec_sum_1[17]), .cout(sec_cout_1[17]));
	fulladder secadd1_18 (.a(fir_sum_1[20]), .b(fir_cout_1[19]), .cin(fir_sum_2[14]), .sum(sec_sum_1[18]), .cout(sec_cout_1[18]));

	fulladder secadd1_19 (.a(fir_sum_1[21]), .b(fir_cout_1[20]), .cin(pp7[11]), .sum(sec_sum_1[19]), .cout(sec_cout_1[19]));
	fulladder secadd1_20 (.a(fir_sum_1[22]), .b(fir_cout_1[21]), .cin(pp7[12]), .sum(sec_sum_1[20]), .cout(sec_cout_1[20]));

	fulladder secadd1_21 (.a(fir_sum_1[23]), .b(fir_cout_1[22]), .cin(pp8[11]), .sum(sec_sum_1[21]), .cout(sec_cout_1[21]));
	fulladder secadd1_22 (.a(fir_sum_1[24]), .b(fir_cout_1[23]), .cin(pp8[12]), .sum(sec_sum_1[22]), .cout(sec_cout_1[22]));

	fulladder secadd1_23 (.a(pp7[17]), .b(pp8[15]), .cin(fir_cout_1[26]), .sum(sec_sum_1[23]), .cout());                     //col 31

	wire [8:0] sec_sum_2;          // sum output of adder, layer 2
	wire [8:0] sec_cout_2;         // cout output of adder, layer 2
	fulladder secadd2_0  (.a(pp6[2]), .b(pp7[0]), .cin(fir_cout_2[3]), .sum(sec_sum_2[0]), .cout(sec_cout_2[0]));
	fulladder secadd2_1  (.a(pp6[3]), .b(pp7[1]), .cin(fir_cout_2[4]), .sum(sec_sum_2[1]), .cout(sec_cout_2[1]));

	halfadder secadd2_2  (.a(fir_sum_3[0]), .b(fir_cout_2[5]), .sum(sec_sum_2[2]), .cout(sec_cout_2[2]));

	fulladder secadd2_3  (.a(fir_sum_3[1]), .b(fir_cout_3[0]), .cin(fir_cout_2[6]), .sum(sec_sum_2[3]), .cout(sec_cout_2[3]));
	fulladder secadd2_4  (.a(fir_sum_3[2]), .b(fir_cout_3[1]), .cin(fir_cout_2[7]), .sum(sec_sum_2[4]), .cout(sec_cout_2[4]));
	fulladder secadd2_5  (.a(fir_sum_3[3]), .b(fir_cout_3[2]), .cin(fir_cout_2[8]), .sum(sec_sum_2[5]), .cout(sec_cout_2[5]));
	fulladder secadd2_6  (.a(fir_sum_3[4]), .b(fir_cout_3[3]), .cin(fir_cout_2[9]), .sum(sec_sum_2[6]), .cout(sec_cout_2[6]));

	fulladder secadd2_7  (.a(pp8[5]), .b(fir_cout_3[4]), .cin(fir_cout_2[10]), .sum(sec_sum_2[7]), .cout(sec_cout_2[7]));

	halfadder secadd2_8  (.a(pp8[6]), .b(fir_cout_2[11]), .sum(sec_sum_2[8]), .cout(sec_cout_2[8]));



///////////////// third stage ////////////////////
	wire [18:0] thr_sum;          // sum output of adder
	wire [18:0] thr_cout;         // cout output of adder
	fulladder thradd0  (.a(sec_sum_1[2]), .b(sec_cout_1[1]), .cin(pp4[0]), .sum(thr_sum[0]), .cout(thr_cout[0]));
	fulladder thradd1  (.a(sec_sum_1[3]), .b(sec_cout_1[2]), .cin(pp4[1]), .sum(thr_sum[1]), .cout(thr_cout[1]));

	fulladder thradd2  (.a(sec_sum_1[5]), .b(sec_cout_1[4]), .cin(fir_cout_2[0]), .sum(thr_sum[2]), .cout(thr_cout[2]));

	fulladder thradd3  (.a(sec_sum_1[6]), .b(sec_cout_1[5]), .cin(pp6[0]), .sum(thr_sum[3]), .cout(thr_cout[3]));
	fulladder thradd4  (.a(sec_sum_1[7]), .b(sec_cout_1[6]), .cin(pp6[1]), .sum(thr_sum[4]), .cout(thr_cout[4]));

	fulladder thradd5  (.a(sec_sum_1[8]), .b(sec_cout_1[7]), .cin(sec_sum_2[0]), .sum(thr_sum[5]), .cout(thr_cout[5]));
	fulladder thradd6  (.a(sec_sum_1[9]), .b(sec_cout_1[8]), .cin(sec_sum_2[1]), .sum(thr_sum[6]), .cout(thr_cout[6]));
	fulladder thradd7  (.a(sec_sum_1[10]), .b(sec_cout_1[9]), .cin(sec_sum_2[2]), .sum(thr_sum[7]), .cout(thr_cout[7]));
	fulladder thradd8  (.a(sec_sum_1[11]), .b(sec_cout_1[10]), .cin(sec_sum_2[3]), .sum(thr_sum[8]), .cout(thr_cout[8]));
	fulladder thradd9  (.a(sec_sum_1[12]), .b(sec_cout_1[11]), .cin(sec_sum_2[4]), .sum(thr_sum[9]), .cout(thr_cout[9]));
	fulladder thradd10 (.a(sec_sum_1[13]), .b(sec_cout_1[12]), .cin(sec_sum_2[5]), .sum(thr_sum[10]), .cout(thr_cout[10]));
	fulladder thradd11 (.a(sec_sum_1[14]), .b(sec_cout_1[13]), .cin(sec_sum_2[6]), .sum(thr_sum[11]), .cout(thr_cout[11]));
	fulladder thradd12 (.a(sec_sum_1[15]), .b(sec_cout_1[14]), .cin(sec_sum_2[7]), .sum(thr_sum[12]), .cout(thr_cout[12]));
	fulladder thradd13 (.a(sec_sum_1[16]), .b(sec_cout_1[15]), .cin(sec_sum_2[8]), .sum(thr_sum[13]), .cout(thr_cout[13]));

	fulladder thradd14 (.a(sec_sum_1[17]), .b(sec_cout_1[16]), .cin(fir_cout_2[12]), .sum(thr_sum[14]), .cout(thr_cout[14]));
	fulladder thradd15 (.a(sec_sum_1[18]), .b(sec_cout_1[17]), .cin(fir_cout_2[13]), .sum(thr_sum[15]), .cout(thr_cout[15]));

	fulladder thradd16 (.a(sec_sum_1[19]), .b(sec_cout_1[18]), .cin(pp8[9]), .sum(thr_sum[16]), .cout(thr_cout[16]));
	fulladder thradd17 (.a(sec_sum_1[20]), .b(sec_cout_1[19]), .cin(pp8[10]), .sum(thr_sum[17]), .cout(thr_cout[17]));

	fulladder thradd18 (.a(fir_sum_1[25]), .b(fir_cout_1[24]), .cin(sec_cout_1[22]), .sum(thr_sum[18]), .cout(thr_cout[18]));


///////////////// forth stage ////////////////////
	wire [18:0] for_sum;          // sum output of adder
	wire [18:0] for_cout;         // cout output of adder
	fulladder foradd0  (.a(sec_sum_1[4]), .b(sec_cout_1[3]), .cin(thr_cout[1]), .sum(for_sum[0]), .cout(for_cout[0]));

	fulladder foradd1  (.a(thr_sum[3]), .b(thr_cout[2]), .cin(fir_cout_2[1]), .sum(for_sum[1]), .cout(for_cout[1]));
	fulladder foradd2  (.a(thr_sum[4]), .b(thr_cout[3]), .cin(fir_cout_2[2]), .sum(for_sum[2]), .cout(for_cout[2]));

	halfadder foradd3  (.a(thr_sum[5]), .b(thr_cout[4]), .sum(for_sum[3]), .cout(for_cout[3]));

	fulladder foradd4  (.a(thr_sum[6]), .b(thr_cout[5]), .cin(sec_cout_2[0]), .sum(for_sum[4]), .cout(for_cout[4]));
	fulladder foradd5  (.a(thr_sum[7]), .b(thr_cout[6]), .cin(sec_cout_2[1]), .sum(for_sum[5]), .cout(for_cout[5]));
	fulladder foradd6  (.a(thr_sum[8]), .b(thr_cout[7]), .cin(sec_cout_2[2]), .sum(for_sum[6]), .cout(for_cout[6]));
	fulladder foradd7  (.a(thr_sum[9]), .b(thr_cout[8]), .cin(sec_cout_2[3]), .sum(for_sum[7]), .cout(for_cout[7]));
	fulladder foradd8  (.a(thr_sum[10]), .b(thr_cout[9]), .cin(sec_cout_2[4]), .sum(for_sum[8]), .cout(for_cout[8]));
	fulladder foradd9  (.a(thr_sum[11]), .b(thr_cout[10]), .cin(sec_cout_2[5]), .sum(for_sum[9]), .cout(for_cout[9]));
	fulladder foradd10 (.a(thr_sum[12]), .b(thr_cout[11]), .cin(sec_cout_2[6]), .sum(for_sum[10]), .cout(for_cout[10]));
	fulladder foradd11 (.a(thr_sum[13]), .b(thr_cout[12]), .cin(sec_cout_2[7]), .sum(for_sum[11]), .cout(for_cout[11]));
	fulladder foradd12 (.a(thr_sum[14]), .b(thr_cout[13]), .cin(sec_cout_2[8]), .sum(for_sum[12]), .cout(for_cout[12]));

	halfadder foradd13 (.a(thr_sum[15]), .b(thr_cout[14]), .sum(for_sum[13]), .cout(for_cout[13]));

	fulladder foradd14 (.a(thr_sum[16]), .b(thr_cout[15]), .cin(fir_cout_2[14]), .sum(for_sum[14]), .cout(for_cout[14]));

	halfadder foradd15 (.a(thr_sum[17]), .b(thr_cout[16]), .sum(for_sum[15]), .cout(for_cout[15]));

	fulladder foradd16 (.a(sec_sum_1[21]), .b(thr_cout[17]), .cin(sec_cout_1[20]), .sum(for_sum[16]), .cout(for_cout[16]));

	halfadder foradd17 (.a(sec_sum_1[22]), .b(sec_cout_1[21]), .sum(for_sum[17]), .cout(for_cout[17]));

	fulladder foradd18 (.a(fir_sum_1[26]), .b(fir_cout_1[25]), .cin(thr_cout[18]), .sum(for_sum[18]), .cout(for_cout[18]));


	assign adda = {sec_sum_1[23],for_sum[18],thr_sum[18],for_sum[17:1],thr_sum[2],for_sum[0],thr_sum[1:0],sec_sum_1[1:0],fir_sum_1[1:0],pp0[3:0]};
	assign addb = {for_cout[18],1'b0,for_cout[17:1],1'b0,for_cout[0],1'b0,thr_cout[0],1'b0,sec_cout_1[0],1'b0,fir_cout_1[0],1'b0,pp1[1:0],2'b00};

endmodule