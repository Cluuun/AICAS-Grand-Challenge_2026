module pow2_approximation #(
    parameter IN_DATA_WIDTH  = 16,
    parameter IN_FRAC_WIDTH  = 10,
    parameter OUT_DATA_WIDTH = 10,
    parameter OUT_FRAC_WIDTH = 6 
) (
    input  logic                      clk_i      ,
    input  logic                      rstn_i     ,

    input  logic                             in_valid   ,
    input  logic signed [IN_DATA_WIDTH-1:0]  in_data_i  ,

    output logic                             out_valid  ,
    output logic        [OUT_DATA_WIDTH-1:0] out_data_o 
);

localparam INT_WIDTH  = IN_DATA_WIDTH - IN_FRAC_WIDTH;
localparam SLOPE_WIDTH = 10;
localparam INTERCEPT_WIDTH = 11;
localparam MUL_WIDTH = IN_FRAC_WIDTH + SLOPE_WIDTH; // 乘法结果小数宽度
localparam LPW_WIDTH = MUL_WIDTH + OUT_DATA_WIDTH - OUT_FRAC_WIDTH; // 加法结果总宽度

logic signed [INT_WIDTH    -1:0]   in_int;
logic        [IN_FRAC_WIDTH-1:0]   in_frac;

// 分段查表索引（使用最高2位）
logic [1:0] frac_sel;

// LUT：斜率 + 截距
logic [SLOPE_WIDTH    :0] slope; //int part is 1 bit
logic [INTERCEPT_WIDTH:0] intercept;

logic [MUL_WIDTH:0] mul_result;
logic [LPW_WIDTH-1:0] lpw_result;

logic [LPW_WIDTH-1:0] shifted_result;
logic                 lpw_valid;
// 拆解
//assign in_int = $signed(in_data_i[IN_DATA_WIDTH-1:IN_FRAC_WIDTH]);
always_ff @(posedge clk_i or negedge rstn_i) begin
    if(!rstn_i) begin       
        in_int <= '0;
        lpw_valid <= '0;
        lpw_result <= '0;
    end else if(in_valid) begin
        in_int <= in_data_i[IN_DATA_WIDTH-1:IN_FRAC_WIDTH];
        lpw_valid <= '1;
        lpw_result <= mul_result + (intercept << (MUL_WIDTH - INTERCEPT_WIDTH));
    end
    else begin
        in_int <= in_int;
        lpw_valid <= '0;
        lpw_result <= lpw_result;
    end
end
assign in_frac = in_data_i[IN_FRAC_WIDTH-1:0];

assign frac_sel = in_frac[IN_FRAC_WIDTH-1 -: 2];

always_comb begin
    case (frac_sel)
        2'b00: begin slope = 11'h307; intercept = 12'h800; end
        2'b01: begin slope = 11'h39a; intercept = 12'h7b7; end
        2'b10: begin slope = 11'h448; intercept = 12'h708; end
        2'b11: begin slope = 11'h517; intercept = 12'h5d1; end
    endcase
end

// 2^frac ≈ slope * frac + intercept
assign mul_result = in_frac * slope;
//assign lpw_result = mul_result + (intercept << (MUL_WIDTH - INTERCEPT_WIDTH));

// 2^int
assign shifted_result = in_int<0? (lpw_result >> (-in_int)):(lpw_result<<in_int); // in_int <= 0
always_ff @(posedge clk_i or negedge rstn_i) begin
    if(rstn_i==0)begin
        out_data_o<='0;
        out_valid<=1'b0;
    end
    else if(lpw_valid)begin
        out_data_o<= shifted_result[LPW_WIDTH-1 -: OUT_DATA_WIDTH];
        out_valid<=1'b1;
    end
    else begin
        out_data_o<=out_data_o;
        out_valid<=1'b0;
    end
end
//assign out_data_o  = shifted_result[LPW_WIDTH-1 -: OUT_DATA_WIDTH];

endmodule
