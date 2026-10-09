////////////////////////////////////////////////////////////////////////////////////
//
// ZeroSoC version 1.2
//
// Module: Mul -- Multiplier Top Module
//
////////////////////////////////////////////////////////////////////////////////////

module multiplier(
	input 			  	clk_i,
    input 			  	rst_n_i,
	input     [31:0]  	opa,
	input     [31:0]  	opb,
	input     [2:0]   	mode,     // mode = 3'b000 mul   , mode = 3'b010 mulh , 
								// mode = 3'b011 mulhu , mode = 3'b001 mulhsu , sign a    unsign b
								// mode = 3'b101 8-bit dot product , mode = 3'b110 16-bit dot product , else illegal
	output    [31:0]  	res_low_32,   // low 32 bit of the result or dot product
	output    [31:0]  	res_high_32,   // high 32 bit of the result
	input     		  	mul_start,
	output    reg 		mul_finish
	);


	reg [1:0]  mult16_mode_1;
	reg [1:0]  mult16_mode_2;
	reg [1:0]  mult16_mode_3;
	reg [1:0]  mult16_mode_4;
	reg [31:0] out_reg;
	wire [31:0] res_high;     // high 32 bit of the result
	wire [32:0] add_mid;
	wire [47:0] res_mulh;
	// wire [31:0] res_dot_8;       // the result of 8-bit dot product
	wire [31:0] res_low;         // low 32 bit of the result
	reg         sign_add_33_a;   // sign = 1, sign a, sign = 0, unsign a
	reg         sign_add_33_b;   // sign = 1, sign b, sign = 0, unsign b


	always @(posedge clk_i or negedge rst_n_i) begin
		if (!rst_n_i) begin
			mul_finish  <= 0;
		end else begin
			mul_finish <= mul_start;
		end	
	end

	assign res_low_32  = {res_mulh[15:0], res_low[15:0]};
	assign res_high_32 = res_mulh[47:16]; 

	always @(*) begin
		case (mode)
			3'b000 : begin
				mult16_mode_1 = 2'b00;
				mult16_mode_2 = 2'b00;
				mult16_mode_3 = 2'b00;
				mult16_mode_4 = 2'b00;
				sign_add_33_a = 0;
				sign_add_33_b = 0;
			end
			3'b001 : begin   //   sign a , unsign b
				mult16_mode_1 = 2'b01;
				mult16_mode_2 = 2'b01;
				mult16_mode_3 = 2'b00;
				mult16_mode_4 = 2'b00;
				sign_add_33_a = 1;
				sign_add_33_b = 0;
			end
			3'b010 : begin
				mult16_mode_1 = 2'b10;
				mult16_mode_2 = 2'b01;
				mult16_mode_3 = 2'b01;
				mult16_mode_4 = 2'b00;				
				sign_add_33_a = 1;
				sign_add_33_b = 1;

			end
			3'b011 : begin
				mult16_mode_1 = 2'b00;
				mult16_mode_2 = 2'b00;
				mult16_mode_3 = 2'b00;
				mult16_mode_4 = 2'b00;				
				sign_add_33_a = 0;
				sign_add_33_b = 0;
			end
			default : begin
				mult16_mode_1 = 2'b00;
				mult16_mode_2 = 2'b00;
				mult16_mode_3 = 2'b00;
				mult16_mode_4 = 2'b00;				
				out_reg       = 0;
				sign_add_33_a = 0;
				sign_add_33_b = 0;
			end
		endcase
	end

/////////////// 3 16x 16 multiplier  /////////////////////

	multi_16 mult1(
		.clk	(clk_i),
		.rst_n	(rst_n_i),
		.a_in   (opa[31:16]),
		.b_in   (opb[31:16]),
		.mode   (mult16_mode_1),
		.res_out(res_high)
		);

	wire [31:0] res_middle_1;     // 47:16 bit of the result

	multi_16 mult2(
		.clk	(clk_i),
		.rst_n	(rst_n_i),
		.a_in   (opa[31:16]),
		.b_in   (opb[15:0]),
		.mode   (mult16_mode_2),
		.res_out(res_middle_1)
		);

	wire [31:0] res_middle_0;     // 47:16 bit of the result


	multi_16 mult3(
		.clk	(clk_i),
		.rst_n	(rst_n_i),
		.a_in   (opb[31:16]),
		.b_in   (opa[15:0]),
		.mode   (mult16_mode_3),
		.res_out(res_middle_0)
		);


	wire  sign_add_33_a_wire;
	wire  sign_add_33_b_wire;
	assign sign_add_33_a_wire = (sign_add_33_a) ? res_middle_1[31] : 0;
	assign sign_add_33_b_wire = (sign_add_33_b) ? res_middle_0[31] : 0;

	adder33 adder_middle (
		.a   ({sign_add_33_a_wire,res_middle_1}),
		.b   ({sign_add_33_b_wire,res_middle_0}),
		.sum (add_mid)
		);

//////////////// caculate the result of mulh //////////////////

	wire [14:0] sign_add;

	generate
		genvar i;
		for (i=0;i<15;i=i+1) begin: sign_add_blcok
			assign sign_add[i] = (sign_add_33_a | sign_add_33_b)? add_mid[32] : 0;
		end
	endgenerate

	adder48 adder_mulh(
		.a  ({res_high,res_low[31:16]}),
		.b  ({sign_add,add_mid}),
		.sum(res_mulh),
		.cout()
		);

	multi_16 mult4(
	.clk	(clk_i),
	.rst_n	(rst_n_i),
	.a_in   (opa[15:0]),
	.b_in   (opb[15:0]),
	.mode   (mult16_mode_4),
	.res_out(res_low)
	);

endmodule