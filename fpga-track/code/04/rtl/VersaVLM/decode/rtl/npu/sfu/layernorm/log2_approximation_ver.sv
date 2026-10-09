module log2_approximation_ver #(
    parameter VER_WIDTH   = 128,
    parameter IDATA_WIDTH = 16 ,
    parameter IFRAC_WIDTH = 8  ,
    parameter ODATA_WIDTH = 23 ,
    parameter OFRAC_WIDTH = 16 
) (
    input  logic                          clk_i  ,
    input  logic                          rstn_i ,
    input  logic        [IDATA_WIDTH-1:0] data_i  [VER_WIDTH-1:0],
    input  logic                          valid_i,

    output logic signed [ODATA_WIDTH-1:0] data_o  [VER_WIDTH-1:0],
    output logic                          valid_o
);

localparam SHIFT_WIDTH = $clog2(IDATA_WIDTH);

logic                   msb_not_found [VER_WIDTH-1:0];
logic [SHIFT_WIDTH-1:0] shift_amt [VER_WIDTH-1:0];

logic [IDATA_WIDTH-1:0] shift_data [VER_WIDTH-1:0];
logic signed [ODATA_WIDTH-OFRAC_WIDTH-1:0] int_val [VER_WIDTH-1:0];

genvar vi;
generate
    for (vi = 0; vi < VER_WIDTH; vi++) begin
        always_comb begin   //find the first one in data
            msb_not_found[vi] = 1'b1;
            shift_amt[vi]     = '0;
            for (int i = IDATA_WIDTH-1; i >= 0; i--) begin
                if (data_i[vi][i] && msb_not_found[vi]) begin
                    shift_amt[vi]     = IDATA_WIDTH - 1 - i;
                    msb_not_found[vi] = 1'b0;
                end
            end
        end
        assign shift_data[vi] = data_i[vi] << shift_amt[vi];  //shift the 1 to the position [IDATAWIDTH-1]
        assign int_val[vi]    = $signed(IDATA_WIDTH - 1 - shift_amt[vi] - IFRAC_WIDTH);  //cal the exponent value
    end
endgenerate

always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
        for (int vi = 0; vi < VER_WIDTH; vi++) begin
            data_o[vi] <= '0;
        end
        valid_o <= 1'b0;
    end else if (valid_i) begin
        for (int vi = 0; vi < VER_WIDTH; vi++) begin  //log2(M*2^k)=k+log2M,when M in [1,2),log2M=M-1,also is shift-data frac part
            data_o[vi] <= msb_not_found[vi] ? - (1 <<< (ODATA_WIDTH-1)) : {int_val[vi], shift_data[vi][IDATA_WIDTH-2 -: OFRAC_WIDTH]};
        end
        valid_o <= 1'b1;
    end else begin
        valid_o <= 1'b0;
    end
end

endmodule