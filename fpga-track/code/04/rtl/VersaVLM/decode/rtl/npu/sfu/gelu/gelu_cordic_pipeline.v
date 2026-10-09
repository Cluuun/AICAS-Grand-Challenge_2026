module gelu_cordic_pipeline #(
    parameter DATA_WIDTH = 34,
    parameter K_WIDTH = 17,
    parameter MODE = 0         //0 for ratation mode, 1 for vectoring mode
    // parameter K = 0,
    // parameter ARCTANH = 0
)(
    clk,
    rst_n,
    en,
    tau,
    arctanh,
    xin,
    yin,
    zin,
    kin,
    vld,
    xout,
    yout,
    zout,
    kout
);
input                        clk;
input                        rst_n;
input                        en;
input   [            5 : 0]  tau;
input   [DATA_WIDTH -1 : 0]  arctanh;
input   [DATA_WIDTH -1 : 0]  xin;
input   [DATA_WIDTH -1 : 0]  yin;
input   [DATA_WIDTH -1 : 0]  zin;
input   [   K_WIDTH -1 : 0]  kin;
output                       vld;
output  [DATA_WIDTH -1 : 0]  xout;
output  [DATA_WIDTH -1 : 0]  yout;
output  [DATA_WIDTH -1 : 0]  zout;
output  [   K_WIDTH -1 : 0]  kout;

wire                         d;
wire    [DATA_WIDTH -1 : 0]  x_s;
wire    [DATA_WIDTH -1 : 0]  y_s;
wire    [DATA_WIDTH -1 : 0]  xout_w;
wire    [DATA_WIDTH -1 : 0]  yout_w;
wire    [DATA_WIDTH -1 : 0]  zout_w;
reg     [DATA_WIDTH -1 : 0]  xout_r;
reg     [DATA_WIDTH -1 : 0]  yout_r;
reg     [DATA_WIDTH -1 : 0]  zout_r;  
reg     [   K_WIDTH -1 : 0]  kout_r;  
reg                          vld_r;


assign d = MODE == 1 ? ~yin[DATA_WIDTH - 1] : zin[DATA_WIDTH - 1];
assign x_s = xin >>> tau;
assign y_s = yin >>> tau;
assign xout_w = d == 1'b1 ? xin - y_s : xin + y_s;
assign yout_w = d == 1'b1 ? yin - x_s : yin + x_s;
assign zout_w = d == 1'b1 ? zin + arctanh : zin - arctanh;

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        xout_r <= {DATA_WIDTH{1'b0}};
        yout_r <= {DATA_WIDTH{1'b0}};
        zout_r <= {DATA_WIDTH{1'b0}};
        kout_r <= {K_WIDTH{1'b0}};
        vld_r <= 1'b0;
    end
    else if (en) begin
        xout_r <= xout_w;
        yout_r <= yout_w;
        zout_r <= zout_w;
        kout_r <= kin;
        vld_r <= en;
    end
    else begin
        xout_r <= {DATA_WIDTH{1'b0}};
        yout_r <= {DATA_WIDTH{1'b0}};
        zout_r <= {DATA_WIDTH{1'b0}};
        kout_r <= {K_WIDTH{1'b0}};
        vld_r <= 1'b0;
    end
end

assign xout = xout_r;
assign yout = yout_r;
assign zout = zout_r;
assign kout = kout_r;
assign vld = vld_r;

endmodule
