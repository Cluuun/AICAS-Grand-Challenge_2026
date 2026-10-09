module gelu_cordic #(
    parameter DATA_WIDTH = 16,
    parameter K_WIDTH    = 4
)(
    clk,
    rst_n,
    xin,
    yin,
    zin,
    kin,
    start,
    tan_out,
    done
    //xout,
    //yout
    //zout
);
localparam CORDIC_WIDTH = DATA_WIDTH + 2;
localparam NUM_STAGES = 12;
input                           clk;
input                           rst_n;
input   [CORDIC_WIDTH - 1 : 0]  xin; //<2,16>
input   [CORDIC_WIDTH - 1 : 0]  yin;
input   [CORDIC_WIDTH - 1 : 0]  zin;
input   [  K_WIDTH - 1    : 0]  kin; //<4,0>
input                           start;
output  [DATA_WIDTH -1    : 0]  tan_out; //not real tanh, but tanh + 1
output                          done;


// input   [18 - 1 : 0]  xin;
// input   [18 - 1 : 0]  yin;
// input   [18 - 1 : 0]  zin;
// input   [ 4 - 1 : 0]  kin;
// input                 start;
// output  [16 - 1 : 0]  tan_out;
// output                done;

// wire d;
// wire [18 - 1 : 0] x_w;
// wire [18 - 1 : 0] y_w;
// wire [18 - 1 : 0] z_w;
// reg  [18 - 1 : 0] x;
// reg  [18 - 1 : 0] y;
// reg  [18 - 1 : 0] z;
// reg  [ 4 - 1 : 0] k;

// assign d = MODE == 1 ? ~yin[DATA_WIDTH - 1] : zin[DATA_WIDTH - 1];
// assign 

// always @(posedge clk or negedge rst_n) begin
//     if (~rst_n) begin
//         x <= 18'b0;
//         y <= 18'b0;
//         z <= 18'b0;
//         k <=  4'b0;
//     end
//     else if (start) begin
//         x <= xin;
//         y <= yin;
//         z <= zin;
//         k <= kin;
//     end
//     else if (start_d1) begin
//         x <= x_w;
// end

reg [ 5 : 0] tau_table     [34 : 0];
reg [33 : 0] arctanh_table [34 : 0];

wire [                NUM_STAGES + 1  - 1 : 0] en_pipe;
wire [CORDIC_WIDTH * (NUM_STAGES + 1) - 1 : 0] x_pipe;
wire [CORDIC_WIDTH * (NUM_STAGES + 1) - 1 : 0] y_pipe;
wire [CORDIC_WIDTH * (NUM_STAGES + 1) - 1 : 0] z_pipe;
wire [     K_WIDTH * (NUM_STAGES + 1) - 1 : 0] k_pipe;
wire [CORDIC_WIDTH - 1 : 0]  x_result;
wire [CORDIC_WIDTH - 1 : 0]  y_result;
wire [CORDIC_WIDTH     : 0]  er_tmp;
wire [CORDIC_WIDTH     : 0]  e_r_tmp;
wire [32 - 1 : 0]  er; //the width is 32 for div
wire [32 - 1 : 0]  e_r; //<16,16>
wire [32 - 1 : 0]  quotient;
reg  [32 - 1 : 0]  dividend;
reg  [32 - 1 : 0]  divisor;
reg  div_start;
wire vld_result;

assign en_pipe[0] = start;
assign x_pipe[CORDIC_WIDTH - 1 : 0] = xin;
assign y_pipe[CORDIC_WIDTH - 1 : 0] = yin;
assign z_pipe[CORDIC_WIDTH - 1 : 0] = zin;
assign k_pipe[     K_WIDTH - 1 : 0] = kin;

assign x_result = x_pipe[CORDIC_WIDTH * (NUM_STAGES + 1) - 1 : CORDIC_WIDTH * NUM_STAGES];
assign y_result = y_pipe[CORDIC_WIDTH * (NUM_STAGES + 1) - 1 : CORDIC_WIDTH * NUM_STAGES];
assign vld_result = en_pipe[NUM_STAGES];
genvar i;
generate
    for (i = 0; i < NUM_STAGES; i = i + 1) begin : pipe_gen
        gelu_cordic_pipeline #(
            .DATA_WIDTH   (CORDIC_WIDTH),
            .K_WIDTH      (K_WIDTH),
            .MODE         (0)
            // .K            (tau_table[i]),
            // .ARCTANH      (arctanh_table[i][33 : 33 - DATA_WIDTH + 1])
        ) u_cordic_pipe (
            .clk          (clk),
            .rst_n        (rst_n),
            .en           (en_pipe[i]),
            .tau          (tau_table[i]),
            .arctanh      (arctanh_table[i][33 : 33 - CORDIC_WIDTH + 1]),
            .xin          (x_pipe[CORDIC_WIDTH * (i + 1) - 1 : CORDIC_WIDTH * i]),
            .yin          (y_pipe[CORDIC_WIDTH * (i + 1) - 1 : CORDIC_WIDTH * i]),
            .zin          (z_pipe[CORDIC_WIDTH * (i + 1) - 1 : CORDIC_WIDTH * i]),
            .kin          (k_pipe[     K_WIDTH * (i + 1) - 1 :      K_WIDTH * i]),
            .vld          (en_pipe[i + 1]),
            .xout         (x_pipe[CORDIC_WIDTH * (i + 2) - 1 : CORDIC_WIDTH * (i + 1)]),
            .yout         (y_pipe[CORDIC_WIDTH * (i + 2) - 1 : CORDIC_WIDTH * (i + 1)]),
            .zout         (z_pipe[CORDIC_WIDTH * (i + 2) - 1 : CORDIC_WIDTH * (i + 1)]),
            .kout         (k_pipe[     K_WIDTH * (i + 2) - 1 :      K_WIDTH * (i + 1)])
        );
    end
endgenerate

assign er_tmp = x_result + y_result;
assign e_r_tmp = x_result - y_result;
assign er = er_tmp <<< kin;
assign e_r = e_r_tmp >>> kin;
// assign divisor_w = er + e_r;

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        dividend <= 32'b0;
        divisor  <= 32'b0;
        div_start <= 1'b0;
    end
    else if (vld_result) begin
        dividend <= er <<< (1 + 5);
        divisor  <= (er + e_r) >>> 7;
        div_start <= 1'b1;
    end
    else begin
        dividend <= 32'b0;
        divisor  <= 32'b0;
        div_start <= 1'b0;
    end
end

div u_gelu_div (
    .clk_i(clk),
    .rst_n_i(rst_n),

    .start_i(div_start),
    .signed_flag_i(1'b1),
    .dividend_i(dividend),
    .divisor_i(divisor),

    .busy_o(),
    .finish_o(done),
    .quotient_o(quotient),
    .remainder_o()
);

assign tan_out = quotient[16:0];

always @(posedge clk) begin
    tau_table[ 0] = 6'h01;
    tau_table[ 1] = 6'h02;
    tau_table[ 2] = 6'h03;
    tau_table[ 3] = 6'h04;
    tau_table[ 4] = 6'h04;
    tau_table[ 5] = 6'h05;
    tau_table[ 6] = 6'h06;
    tau_table[ 7] = 6'h07;
    tau_table[ 8] = 6'h08;
    tau_table[ 9] = 6'h09;
    tau_table[10] = 6'h0A;
    tau_table[11] = 6'h0B;
    tau_table[12] = 6'h0C;
    tau_table[13] = 6'h0D;
    tau_table[14] = 6'h0D;
    tau_table[15] = 6'h0E;
    tau_table[16] = 6'h0F;
    tau_table[17] = 6'h10;
    tau_table[18] = 6'h11;
    tau_table[19] = 6'h12;
    tau_table[20] = 6'h13;
    tau_table[21] = 6'h14;
    tau_table[22] = 6'h15;
    tau_table[23] = 6'h16;
    tau_table[24] = 6'h17;
    tau_table[25] = 6'h18;
    tau_table[26] = 6'h19;
    tau_table[27] = 6'h1A;
    tau_table[28] = 6'h1B;
    tau_table[29] = 6'h1C;
    tau_table[30] = 6'h1D;
    tau_table[31] = 6'h1E;
    tau_table[32] = 6'h1F;
    tau_table[33] = 6'h20;
    tau_table[34] = 6'h21;
    arctanh_table[ 0] = 34'h8C9F53D5;
    arctanh_table[ 1] = 34'h4162BBEA;
    arctanh_table[ 2] = 34'h202B1239;
    arctanh_table[ 3] = 34'h1005588A;
    arctanh_table[ 4] = 34'h1005588A;
    arctanh_table[ 5] = 34'h0800AAC4;
    arctanh_table[ 6] = 34'h04001556;
    arctanh_table[ 7] = 34'h020002AA;
    arctanh_table[ 8] = 34'h01000055;
    arctanh_table[ 9] = 34'h0080000A;
    arctanh_table[10] = 34'h00400001;
    arctanh_table[11] = 34'h00200000;
    arctanh_table[12] = 34'h00100000;
    arctanh_table[13] = 34'h00080000;
    arctanh_table[14] = 34'h00080000;
    arctanh_table[15] = 34'h00040000;
    arctanh_table[16] = 34'h00020000;
    arctanh_table[17] = 34'h00010000;
    arctanh_table[18] = 34'h00008000;
    arctanh_table[19] = 34'h00004000;
    arctanh_table[20] = 34'h00002000;
    arctanh_table[21] = 34'h00001000;
    arctanh_table[22] = 34'h00000800;
    arctanh_table[23] = 34'h00000400;
    arctanh_table[24] = 34'h00000200;
    arctanh_table[25] = 34'h00000100;
    arctanh_table[26] = 34'h00000080;
    arctanh_table[27] = 34'h00000040;
    arctanh_table[28] = 34'h00000020;
    arctanh_table[29] = 34'h00000010;
    arctanh_table[30] = 34'h00000008;
    arctanh_table[31] = 34'h00000004;
    arctanh_table[32] = 34'h00000002;
    arctanh_table[33] = 34'h00000001;
    arctanh_table[34] = 34'h00000000;
end

// hyperb_table_tau_128 u_tau_table #(
//      .DataWidth(7)
//     ,.AddressRange(128)
//     ,.AddressWidth(7)
// )(
//      .clk(clk)
//     ,.ce0()
//     ,.address0()
//     ,.q0()
// );
// hyperb_table_arctanh_128 u_arctanh_table #(
//      .DataWidth(124)
//     ,.AddressRange(128)
//     ,.AddressWidth(7)
// )(
//      .clk(clk)
//     ,.ce0()
//     ,.address0()
//     ,.q0()
// );




endmodule
