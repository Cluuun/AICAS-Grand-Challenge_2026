module gelu # (
    parameter DATA_WIDTH    = 16,
    parameter FRAC_WIDTH    = 12 
)(
    clk,
    rst_n,
    din,
    start,
    dout,
    busy,
    done
);
input                        clk;
input                        rst_n;
input   [DATA_WIDTH -1 : 0]  din;
input                        start;
output  [DATA_WIDTH -1 : 0]  dout;
output                       busy;
output                       done;


wire    [DATA_WIDTH*2 -1 : 0]  din_squared;
wire    [DATA_WIDTH*3 -1 : 0]  din_cubed;
wire    [DATA_WIDTH -1 : 0]  mul_coeff;
wire    [DATA_WIDTH -1 : 0]  add_coeff;
wire    [DATA_WIDTH    : 0]  add_coeff_tmp;
wire    [DATA_WIDTH -1 : 0]  term;
wire    [DATA_WIDTH -1 : 0]  coeff;
wire    [DATA_WIDTH -1 : 0]  sqrt2pi;
reg     [DATA_WIDTH -1 : 0]  din_r;
reg                          busy_r;

wire [18 - 1 : 0] xin_value = 18'h1351E;
wire [16 - 1 : 0] tanh_value;
wire              tanh_vld;
reg  [ 4 - 1 : 0] k_d1;
reg  start_d1, start_d2, start_d3, start_d4, start_d5, start_d6, start_d7, done_r;

assign coeff = 16'h5b9;
assign sqrt2pi = 16'h6621;
assign add_coeff_tmp = mul_coeff + din_r;
assign add_coeff = add_coeff_tmp[DATA_WIDTH : 1];

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        din_r  <= {DATA_WIDTH{1'b0}};
        busy_r <= 1'b0;
    end
    else if (start & ~busy_r) begin
        din_r  <= din;
        busy_r <= 1'b1;
    end
    else if (tanh_vld) begin
        // din_r  <= {DATA_WIDTH{1'b0}};
        busy_r <= 1'b0;
    end
end

gelu_mul_fixed16 #(
    .DIN0_WIDTH         (DATA_WIDTH),
    .DIN1_WIDTH         (DATA_WIDTH),
    .DOUT_WIDTH         (DATA_WIDTH*2),
    .DIN0_INTEGER_WIDTH (DATA_WIDTH - FRAC_WIDTH),
    .DIN1_INTEGER_WIDTH (DATA_WIDTH - FRAC_WIDTH),
    .DOUT_INTEGER_WIDTH (DATA_WIDTH*2 - FRAC_WIDTH*2)
) u_mul_01 (
    .clk                (clk),
    .rst_n              (rst_n),
    .din0               (din_r),
    .din1               (din_r),
    .dout               (din_squared)
);
gelu_mul_fixed16 #(
    .DIN0_WIDTH         (DATA_WIDTH),
    .DIN1_WIDTH         (DATA_WIDTH*2),
    .DOUT_WIDTH         (DATA_WIDTH*3),
    .DIN0_INTEGER_WIDTH (DATA_WIDTH - FRAC_WIDTH),
    .DIN1_INTEGER_WIDTH (DATA_WIDTH*2 - FRAC_WIDTH*2),
    .DOUT_INTEGER_WIDTH (DATA_WIDTH*3 - FRAC_WIDTH*3)
) u_mul_02 (
    .clk                (clk),
    .rst_n              (rst_n),
    .din0               (din_r),
    .din1               (din_squared),
    .dout               (din_cubed)
);
wire [23:0] din_cubed_trunc;
assign din_cubed_trunc = din_cubed[DATA_WIDTH*3-1 : DATA_WIDTH*3-24];

gelu_mul_fixed16 #(
    .DIN0_WIDTH         (24),
    .DIN1_WIDTH         (16),
    .DOUT_WIDTH         (DATA_WIDTH),
    .DIN0_INTEGER_WIDTH (DATA_WIDTH*3 - FRAC_WIDTH*3),
    .DIN1_INTEGER_WIDTH (1),
    .DOUT_INTEGER_WIDTH (4)
) u_mul_03 (
    .clk                (clk),
    .rst_n              (rst_n),
    .din0               (din_cubed_trunc),
    .din1               (coeff),
    .dout               (mul_coeff)
);
gelu_mul_fixed16 #(
    .DIN0_WIDTH         (16),
    .DIN1_WIDTH         (16),
    .DOUT_WIDTH         (DATA_WIDTH),
    .DIN0_INTEGER_WIDTH (5),
    .DIN1_INTEGER_WIDTH (1),
    .DOUT_INTEGER_WIDTH (4)
) u_mul_04 (
    .clk                (clk),
    .rst_n              (rst_n),
    .din0               (add_coeff),
    .din1               (sqrt2pi),
    .dout               (term)
);

wire  [DATA_WIDTH - 1 : 0]  inabs;
wire  [DATA_WIDTH - 1 : 0]  inabs_tmp;
wire  [ 4 - 1         : 0]  k;
wire  [DATA_WIDTH - 1 : 0]  r;
wire  [DATA_WIDTH - 1 : 0]  inv_ln2;
wire  [DATA_WIDTH - 1 : 0]  prod;
wire  [12 - 1         : 0]  prod_dec;
wire  [DATA_WIDTH - 1 : 0]  ln2;

assign inabs_tmp = term[15] == 1'b1 ? -term : term;
assign inabs = {inabs_tmp[14 : 0], 1'b0};
assign inv_ln2 = 16'hb8aa;
assign k = prod[15 : 12];
assign prod_dec = prod[11 : 0];
assign ln2 = 16'hb172;

gelu_mul_ufixed16 #(
    .DOUT_WIDTH         (16),
    .DIN0_INTEGER_WIDTH (3),
    .DIN1_INTEGER_WIDTH (1),
    .DOUT_INTEGER_WIDTH (4)
) u_mul_05 (
    .clk                (clk),
    .rst_n              (rst_n),
    .din0               (inabs),
    .din1               (inv_ln2),
    .dout               (prod)
);
gelu_mul_ufixed16 #(
    .DOUT_WIDTH         (16),
    .DIN0_INTEGER_WIDTH (0),
    .DIN1_INTEGER_WIDTH (0),
    .DOUT_INTEGER_WIDTH (0)
) u_mul_06 (
    .clk                (clk),
    .rst_n              (rst_n),
    .din0               ({prod_dec, {4{1'b0}}}),
    .din1               (ln2),
    .dout               (r)
);

// wire [DATA_WIDTH + 1 : 0] x;
// wire [DATA_WIDTH + 1 : 0] y;

assign done = done_r;
assign busy = busy_r;

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        start_d1 <= 1'b0;
        start_d2 <= 1'b0;
        start_d3 <= 1'b0;
        start_d4 <= 1'b0;
        start_d5 <= 1'b0;
        start_d6 <= 1'b0;
        start_d7 <= 1'b0;
        done_r   <= 1'b0;
    end
    else begin
        start_d1 <= start;
        start_d2 <= start_d1;
        start_d3 <= start_d2;
        start_d4 <= start_d3;
        start_d5 <= start_d4;
        start_d6 <= start_d5;
        start_d7 <= start_d6;
        done_r   <= tanh_vld;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        k_d1 <= 4'b0;
    end
    else if (start_d6)begin
        k_d1 <= k;
    end
end

gelu_cordic #(
    .DATA_WIDTH    (DATA_WIDTH),
    .K_WIDTH       (4)
)u_gelu_cordic(
    .clk(clk),
    .rst_n(rst_n),
    .xin(xin_value),
    .yin(18'b0),
    .zin({{2{1'b0}},r}),
    .kin(k_d1),
    .start(start_d7),
    .tan_out(tanh_value),
    .done(tanh_vld)
);

gelu_mul_fixed16 #(
    .DIN0_WIDTH         (16),
    .DIN1_WIDTH         (16),
    .DOUT_WIDTH         (DATA_WIDTH),
    .DIN0_INTEGER_WIDTH (3), //din_r / 2
    .DIN1_INTEGER_WIDTH (4),
    .DOUT_INTEGER_WIDTH (4)
) u_mul_07 (
    .clk                (clk),
    .rst_n              (rst_n),
    .din0               (din_r),
    .din1               (din_r[15] ? 16'h2000 - tanh_value : tanh_value),
    .dout               (dout)
);


endmodule
