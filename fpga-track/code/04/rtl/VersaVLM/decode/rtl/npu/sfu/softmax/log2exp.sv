module log2exp
#(
    parameter GROUP_SIZE =32,
    parameter DATA_WIDTH =16,
    parameter FRAC_WIDTH=8,
    parameter SHIFT_WIDTH = 3,
    parameter SHIFT_DATA_WIDTH = 8
) (
    input  logic                                       clk,
    input  logic                                       rst_n,
    input  logic                                       in_valid,
    input  logic [GROUP_SIZE-1:0][DATA_WIDTH-1:0 ]     in_num,
    output logic [GROUP_SIZE-1:0][SHIFT_WIDTH-1  :0 ]  out_num,
    output logic [GROUP_SIZE-1:0][SHIFT_DATA_WIDTH-1:0 ]shift_out,                      
    output logic                                       out_valid  
);
    logic  [GROUP_SIZE-1:0][DATA_WIDTH:0 ]             in_1_4375_num;
    logic  [GROUP_SIZE-1:0][DATA_WIDTH-FRAC_WIDTH:0]   out_round;
    logic  [GROUP_SIZE-1:0][SHIFT_WIDTH-1:0 ]          out_data;  
    always_comb begin 
        for(int i=0;i<GROUP_SIZE;i++)begin
            in_1_4375_num[i] = in_num[i]+(in_num[i]>>1)-(in_num[i]>>4);
            out_round[i]     = in_1_4375_num[i][FRAC_WIDTH-1]?(in_1_4375_num[i][DATA_WIDTH:FRAC_WIDTH]+1):(in_1_4375_num[i][DATA_WIDTH:FRAC_WIDTH]);
            if(DATA_WIDTH-FRAC_WIDTH<SHIFT_WIDTH)
            out_data[i]     = out_round[i];
            else
            out_data[i]      = (|out_round[i][DATA_WIDTH-FRAC_WIDTH:SHIFT_WIDTH])?{(SHIFT_WIDTH){1'b1}}:out_round[i][SHIFT_WIDTH-1:0];
        end
    end
    always_ff@( posedge clk or negedge rst_n )begin
        if(rst_n==0)begin
        shift_out   <='0;
        out_num     <='0;
        end
        else if(in_valid)begin
            for (int i=0;i<GROUP_SIZE;i++)begin
                shift_out[i]   <= {1'b1,{(SHIFT_DATA_WIDTH-1){1'b0}}}>>out_data[i];
                out_num[i]     <= out_data[i];
            end
        end
        else begin
        shift_out   <= shift_out;
        out_num     <= out_num;
        end
    end    
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(rst_n==0)
        out_valid<=1'b0;
        else 
        out_valid<=in_valid;    
    end
endmodule
