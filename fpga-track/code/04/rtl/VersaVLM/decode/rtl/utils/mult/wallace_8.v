/////////////////////////////////////
//                                 //
//       8x8 Wallace Tree          //
//                                 //
/////////////////////////////////////

module wallace_8 (
	input    [11:0] pp0,
	input    [10:0] pp1,
	input    [10:0] pp2,
	input    [9:0]  pp3,
	input    [7:0]  pp4,
	output   [15:0] adda,
	output   [15:0] addb
);

///////////////// first stage ////////////////////
	wire [10:0] fir_sum;          // sum output of adder
	wire [10:0] fir_cout;         // cout output of adder
	fulladder firadd_0 (.a(pp0[4]), .b(pp1[2]), .cin(pp2[0]), .sum(fir_sum[0]), .cout(fir_cout[0]));
	fulladder firadd_1 (.a(pp0[5]), .b(pp1[3]), .cin(pp2[1]), .sum(fir_sum[1]), .cout(fir_cout[1]));
	fulladder firadd_2 (.a(pp0[6]), .b(pp1[4]), .cin(pp2[2]), .sum(fir_sum[2]), .cout(fir_cout[2]));
	fulladder firadd_3 (.a(pp0[7]), .b(pp1[5]), .cin(pp2[3]), .sum(fir_sum[3]), .cout(fir_cout[3]));
	fulladder firadd_4 (.a(pp0[8]), .b(pp1[6]), .cin(pp2[4]), .sum(fir_sum[4]), .cout(fir_cout[4]));
	fulladder firadd_5 (.a(pp0[9]), .b(pp1[7]), .cin(pp2[5]), .sum(fir_sum[5]), .cout(fir_cout[5]));
	fulladder firadd_6 (.a(pp0[10]), .b(pp1[8]), .cin(pp2[6]), .sum(fir_sum[6]), .cout(fir_cout[6]));
	fulladder firadd_7 (.a(pp0[11]), .b(pp1[9]), .cin(pp2[7]), .sum(fir_sum[7]), .cout(fir_cout[7]));
	fulladder firadd_8 (.a(pp1[10]), .b(pp2[8]), .cin(pp3[6]), .sum(fir_sum[8]), .cout(fir_cout[8]));
	fulladder firadd_9 (.a(pp2[9]), .b(pp3[7]), .cin(pp4[5]), .sum(fir_sum[9]), .cout(fir_cout[9]));
	fulladder firadd_10 (.a(pp2[10]), .b(pp3[8]), .cin(pp4[6]), .sum(fir_sum[10]), .cout(fir_cout[10]));

///////////////// second stage ////////////////////
	wire [7:0] sec_sum;          // sum output of adder
	wire [6:0] sec_cout;         // cout output of adder
	fulladder secadd0 (.a(fir_sum[2]), .b(fir_cout[1]), .cin(pp3[0]), .sum(sec_sum[0]), .cout(sec_cout[0]));
	fulladder secadd1 (.a(fir_sum[3]), .b(fir_cout[2]), .cin(pp3[1]), .sum(sec_sum[1]), .cout(sec_cout[1]));
	fulladder secadd2 (.a(fir_sum[4]), .b(fir_cout[3]), .cin(pp3[2]), .sum(sec_sum[2]), .cout(sec_cout[2]));
	fulladder secadd3 (.a(fir_sum[5]), .b(fir_cout[4]), .cin(pp3[3]), .sum(sec_sum[3]), .cout(sec_cout[3]));
	fulladder secadd4 (.a(fir_sum[6]), .b(fir_cout[5]), .cin(pp3[4]), .sum(sec_sum[4]), .cout(sec_cout[4]));
	fulladder secadd5 (.a(fir_sum[7]), .b(fir_cout[6]), .cin(pp3[5]), .sum(sec_sum[5]), .cout(sec_cout[5]));
	fulladder secadd6 (.a(fir_sum[8]), .b(fir_cout[7]), .cin(pp4[4]), .sum(sec_sum[6]), .cout(sec_cout[6]));
	fulladder secadd7 (.a(pp3[9]), .b(pp4[7]), .cin(fir_cout[10]), .sum(sec_sum[7]), .cout());
	

///////////////// third stage ////////////////////
	wire [6:0] thr_sum;          // sum output of adder
	wire [6:0] thr_cout;         // cout output of adder
	fulladder thradd0 (.a(sec_sum[2]), .b(sec_cout[1]), .cin(pp4[0]), .sum(thr_sum[0]), .cout(thr_cout[0]));
	fulladder thradd1 (.a(sec_sum[3]), .b(sec_cout[2]), .cin(pp4[1]), .sum(thr_sum[1]), .cout(thr_cout[1]));
	fulladder thradd2 (.a(sec_sum[4]), .b(sec_cout[3]), .cin(pp4[2]), .sum(thr_sum[2]), .cout(thr_cout[2]));
	fulladder thradd3 (.a(sec_sum[5]), .b(sec_cout[4]), .cin(pp4[3]), .sum(thr_sum[3]), .cout(thr_cout[3]));
	halfadder thradd4 (.a(sec_sum[6]), .b(sec_cout[5]), .sum(thr_sum[4]), .cout(thr_cout[4]));
	fulladder thradd5 (.a(fir_sum[9]), .b(sec_cout[6]), .cin(fir_cout[8]), .sum(thr_sum[5]), .cout(thr_cout[5]));
	halfadder thradd6 (.a(fir_sum[10]), .b(fir_cout[9]), .sum(thr_sum[6]), .cout(thr_cout[6]));

	assign adda = {sec_sum[7],thr_sum,sec_sum[1:0],fir_sum[1:0],pp0[3:0]};
	assign addb = {thr_cout,1'b0,sec_cout[0],1'b0,fir_cout[0],1'b0,pp1[1:0],2'b00};

endmodule