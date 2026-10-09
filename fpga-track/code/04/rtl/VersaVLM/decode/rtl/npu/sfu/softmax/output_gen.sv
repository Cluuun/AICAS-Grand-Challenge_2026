module output_gen 
#(
    parameter GROUP_SIZE = 32,
    parameter SHIFT_WIDTH = 3,
    parameter IN_DATA_WIDTH = 16,
    parameter CODE_DATA_WIDTH = 16,
    parameter OUT_DATA_WIDTH = 16
) (
    input  logic      clk,
    input  logic      rst_n,
    input  logic  [GROUP_SIZE-1:0][SHIFT_WIDTH-1:0]       mx_shift   ,
    input  logic  [IN_DATA_WIDTH-1:0]                     local_max  ,
    input  logic  [IN_DATA_WIDTH-1:0]                     global_max ,  
    input  logic  [CODE_DATA_WIDTH-1:0]                   code       ,
    input  logic                                          in_valid   ,
    output logic                                          out_valid  ,
    output logic  [GROUP_SIZE-1:0][OUT_DATA_WIDTH-1:0]    out_data
);
    localparam IN_FRAC_WIDTH = 4;
    logic  [IN_DATA_WIDTH+IN_FRAC_WIDTH-1:0] ml_sub; //global-max   -  local-max
    logic  [IN_DATA_WIDTH+IN_FRAC_WIDTH:0] ml_1_4375_num;  
    logic  [IN_DATA_WIDTH:0] ml_round;
    logic  [SHIFT_WIDTH-1  :0] ml_data;
    logic  out_part_valid;
    logic  [GROUP_SIZE-1:0][SHIFT_WIDTH:0]total_shift;
    always_ff @(posedge clk) begin
        out_part_valid <= in_valid;
        out_valid      <= out_part_valid;
    end
    always_comb begin :ml_data_gen
        ml_sub[IN_DATA_WIDTH+IN_FRAC_WIDTH-1:IN_FRAC_WIDTH]= global_max - local_max;
        ml_sub[IN_FRAC_WIDTH-1:0] = '0;
        ml_1_4375_num = ml_sub+(ml_sub>>1)-(ml_sub>>4);
        ml_round      = ml_1_4375_num[IN_FRAC_WIDTH-1]?(ml_1_4375_num[IN_DATA_WIDTH+IN_FRAC_WIDTH:IN_FRAC_WIDTH]+1):(ml_1_4375_num[IN_DATA_WIDTH+IN_FRAC_WIDTH:IN_FRAC_WIDTH]); 
        if(IN_DATA_WIDTH<SHIFT_WIDTH)
        ml_data       = ml_round;
        else
        ml_data       = (|ml_round[IN_DATA_WIDTH:SHIFT_WIDTH])?{(SHIFT_WIDTH){1'b1}}:ml_round[SHIFT_WIDTH-1:0]; 
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(rst_n==0)begin
            total_shift  <= '0;
        end
        else if(in_valid)begin
            for (int j=0;j<GROUP_SIZE;j++)begin
                if(mx_shift[j]+ml_data=={(SHIFT_WIDTH){1'b1}})
                    total_shift[j]<={{1'b1},{(SHIFT_WIDTH){1'b0}}};
                else
                    total_shift[j]<=mx_shift[j]+ml_data;
            end
                
        end
        else begin
            total_shift  <='0;
        end
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(rst_n==0)begin
            out_data  <= '0;
        end
        else if(out_part_valid)begin
            for (int j=0;j<GROUP_SIZE;j++)begin
                out_data[j]<=code>>(total_shift[j]);
            end
                
        end
        else begin
            out_data  <='0;
        end
    end
endmodule
