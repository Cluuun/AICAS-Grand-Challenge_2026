module log2_approximation #(
    parameter IDATA_WIDTH = 16 ,
    parameter IFRAC_WIDTH = 8  ,
    parameter ODATA_WIDTH = 16 ,
    parameter OFRAC_WIDTH = 10 
) (
    input  logic                          clk_i  ,
    input  logic                          rstn_i ,
    input  logic        [IDATA_WIDTH-1:0] data_i ,
    input  logic                          valid_i,

    output logic signed [ODATA_WIDTH-1:0] data_o ,
    output logic                          valid_o
);

localparam SHIFT_WIDTH = $clog2(IDATA_WIDTH);

logic                   msb_not_found;
logic [SHIFT_WIDTH-1:0] shift_amt;

logic [IDATA_WIDTH-1:0] shift_data;
logic signed [ODATA_WIDTH-OFRAC_WIDTH-1:0] int_val;

genvar vi;
generate
    always_comb begin
        msb_not_found = 1'b1;
        shift_amt     = '0;
        for (int i = IDATA_WIDTH-1; i >= 0; i--) begin
            if (data_i[i] && msb_not_found) begin
                shift_amt     = IDATA_WIDTH - 1 - i;
                msb_not_found = 1'b0;
            end
        end
    end
    assign shift_data = data_i << shift_amt;
    assign int_val    = $signed(IDATA_WIDTH - 1 - shift_amt - IFRAC_WIDTH);
endgenerate

always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
        data_o <= 'b0;
        valid_o <= 1'b0;
    end else if (valid_i) begin
        data_o <= msb_not_found ? - (1 <<< (ODATA_WIDTH-1)) : {int_val, shift_data[IDATA_WIDTH-2 -: OFRAC_WIDTH]};
        valid_o <= 1'b1;
    end else begin
        valid_o <= 1'b0;
    end
end

endmodule