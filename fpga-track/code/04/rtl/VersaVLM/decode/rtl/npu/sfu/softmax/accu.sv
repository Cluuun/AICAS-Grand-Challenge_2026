module accu 
#(
    parameter SHIFT_DATA_WIDTH = 8,
    parameter GROUP_SIZE = 32,
    parameter MAX_LENGTH =1024,
    parameter REAL_IDX_WIDTH = 6,
    parameter DATA_WIDTH =16,
    parameter  IN_WIDTH       = SHIFT_DATA_WIDTH+$clog2(GROUP_SIZE),
    localparam OUT_WIDTH      = $clog2(MAX_LENGTH/GROUP_SIZE)+IN_WIDTH
) (
    input  logic                          clk     ,
    input  logic                          rst_n   ,
    input  logic [IN_WIDTH-1:0]           data_in ,
    input  logic                          in_valid,
    input  logic [REAL_IDX_WIDTH-1:0]     in_num  ,   //from 1 to $clog2(MAX_LENGTH/GROUP_SIZE)
    input  logic [$clog2(DATA_WIDTH)-1:0] mmsub,
    output logic [OUT_WIDTH-1:0]          data_out,
    output logic                          out_valid
);
    logic [REAL_IDX_WIDTH-1:0]count;
    always_ff @(posedge clk or negedge rst_n) begin
        if(rst_n==0)
        data_out<='0;
        else if(count == in_num)
        data_out<='0;
        else if(in_valid)
        data_out<=(data_out>>mmsub)+data_in;
        else
        data_out<=data_out;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if(rst_n==0)
        count<='0;
        else if(count == in_num)
        count<='0;
        else if(in_valid)
        count<=count+1;
        else
        count<=count;  
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if(rst_n==0)
        out_valid<=1'b0;
        else if(in_valid && count == in_num-1)
        out_valid<=1'b1;
        else 
        out_valid<=1'b0;
    end
endmodule