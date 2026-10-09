module quantization_fxX_to_intY #(
    parameter SCALE_DATA_WIDTH       = 16,
    parameter INPUT_DATA_WIDTH       = 16,  //X
    parameter OUTPUT_DATA_WIDTH      = 8 ,   //Y
    parameter INPUT_NUMBER           = 16,
    parameter SCALE_SHIFT_WIDTH      = SCALE_DATA_WIDTH,
    parameter FRAC_WIDTH             = 8  )
(  
    input  logic                                                      clk,
    input  logic                                                      rst_n,
    input  logic                                                      in_valid,  
    input  logic signed [INPUT_NUMBER-1:0] [INPUT_DATA_WIDTH-1:0]     quantized_input   ,
    input  logic signed [SCALE_DATA_WIDTH-1:0]                        quant_scale       ,
    input  logic signed [SCALE_DATA_WIDTH-1:0]                        scale_shift       ,

    output logic                                                      out_valid         ,
    output logic signed [INPUT_NUMBER-1:0][OUTPUT_DATA_WIDTH-1:0]     quantized_output
);

localparam signed [OUTPUT_DATA_WIDTH-1:0] MIN_OUTPUT = {1'b1,{(OUTPUT_DATA_WIDTH-1){1'b0}}};
localparam signed [OUTPUT_DATA_WIDTH-1:0] MAX_OUTPUT = {1'b0,{(OUTPUT_DATA_WIDTH-1){1'b1}}};
localparam MUL_WIDTH = INPUT_DATA_WIDTH+SCALE_DATA_WIDTH>OUTPUT_DATA_WIDTH+3? INPUT_DATA_WIDTH+SCALE_DATA_WIDTH:OUTPUT_DATA_WIDTH+3;  

//----------caculate  *scale--------------------------//
logic signed [INPUT_NUMBER-1:0][MUL_WIDTH-1:0] mul_scale_result ;
logic signed [INPUT_NUMBER-1:0][MUL_WIDTH+FRAC_WIDTH-1:0] extended_result;
logic signed [INPUT_NUMBER-1:0][OUTPUT_DATA_WIDTH+2:0] quant_noclip ;
logic signed[SCALE_SHIFT_WIDTH-1:0]scale_shift_delay;
always_ff @( posedge clk ) begin 
    out_valid<=in_valid;
end
always_ff @( posedge clk or negedge rst_n ) begin 
    if(rst_n==0)
    scale_shift_delay<='0;
    else if(in_valid)
    scale_shift_delay<=scale_shift;
    else 
    scale_shift_delay<=scale_shift_delay;
end
genvar i;
generate
for(i=0; i<INPUT_NUMBER; i++) begin
   always_ff @(posedge clk or negedge rst_n) begin
        if(rst_n==0)
        mul_scale_result[i]<='0;
        else if(in_valid)
        mul_scale_result[i]<= $signed(quant_scale) * $signed(quantized_input[i]);
        else
        mul_scale_result[i]<=mul_scale_result[i];
    end
    always_comb begin
        //mul_scale_result[i] = $signed(quant_scale) * $signed(quantized_input[i]);

        extended_result[i] = { {FRAC_WIDTH{mul_scale_result[i][MUL_WIDTH-1]}}, mul_scale_result[i] };
        // quant_noclip[i] = (scale_shift + FRAC_WIDTH >= 0) ?
        //         (extended_result[i] >>> (scale_shift + FRAC_WIDTH)):
        //         (extended_result[i] << (-scale_shift - FRAC_WIDTH));

        quant_noclip[i] = (-scale_shift_delay + FRAC_WIDTH >= 0) ?
                ($signed(extended_result[i]) >>> (-scale_shift_delay + FRAC_WIDTH)):
                ($signed(extended_result[i]) << (scale_shift_delay - FRAC_WIDTH));       

        if ($signed(quant_noclip[i]) > MAX_OUTPUT) begin
            quantized_output[i] = MAX_OUTPUT;
        end
        else if ($signed(quant_noclip[i]) < MIN_OUTPUT) begin
            quantized_output[i] = MIN_OUTPUT;                                                               
        end
        else begin
            quantized_output[i] = quant_noclip[i][OUTPUT_DATA_WIDTH-1:0];
        end
    end
    // always_ff @(posedge clk or negedge rst_n) begin
    //     if(rst_n==0)begin
    //         quantized_output[i]<='0;
    //     end else if(in_valid)begin
    //         if ($signed(quant_noclip[i]) > MAX_OUTPUT) begin
    //             quantized_output[i] <= MAX_OUTPUT;
    //         end
    //         else if ($signed(quant_noclip[i]) < MIN_OUTPUT) begin
    //             quantized_output[i] <= MIN_OUTPUT;
    //         end
    //         else begin
    //             quantized_output[i] <= quant_noclip[i][OUTPUT_DATA_WIDTH-1:0];
    //         end
    //     end
    //     else begin
    //         quantized_output[i]<=quantized_output[i];
    //     end
    // end
end
endgenerate
endmodule