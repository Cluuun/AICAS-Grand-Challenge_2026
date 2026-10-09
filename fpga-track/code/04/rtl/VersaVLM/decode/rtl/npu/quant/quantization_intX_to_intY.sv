module quantization_intX_to_intY #(
    parameter INPUT_DATA_WIDTH  = 8,
    parameter OUTPUT_DATA_WIDTH = 32,
    parameter SCALE_DATA_WIDTH  = 16,
    parameter SCALE_SHIFT_WIDTH = SCALE_DATA_WIDTH,
    parameter INPUT_NUMBER      = 32)
(  
    input  logic                                                      clk,
    input  logic                                                      rst_n,
    input  logic                                                      in_valid,
    input  logic signed [INPUT_NUMBER-1:0][INPUT_DATA_WIDTH-1:0]      unquantized_input,
    input  logic signed [SCALE_DATA_WIDTH-1:0]                        quant_scale,
    input  logic signed [SCALE_SHIFT_WIDTH-1:0]                       scale_shift,

    output logic                                                      out_valid,
    output logic signed [INPUT_NUMBER-1:0][OUTPUT_DATA_WIDTH-1:0]     quantized_output
);


localparam signed [OUTPUT_DATA_WIDTH-1:0] MIN_OUTPUT = {1'b1,{(OUTPUT_DATA_WIDTH-1){1'b0}}};
localparam signed [OUTPUT_DATA_WIDTH-1:0] MAX_OUTPUT = {1'b0,{(OUTPUT_DATA_WIDTH-1){1'b1}}};
localparam MUL_WIDTH = INPUT_DATA_WIDTH+SCALE_DATA_WIDTH>OUTPUT_DATA_WIDTH+3? INPUT_DATA_WIDTH+SCALE_DATA_WIDTH:OUTPUT_DATA_WIDTH+3;

//----------caculate  *scale--------------------------//
logic signed [INPUT_NUMBER-1:0][MUL_WIDTH-1:0] mul_scale_result ;
logic signed [INPUT_NUMBER-1:0][OUTPUT_DATA_WIDTH+2:0] quant_noclip ;
logic signed [SCALE_SHIFT_WIDTH-1:0]scale_shift_delay;
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
for(i=0; i<INPUT_NUMBER; i=i+1)begin
    always_ff @(posedge clk or negedge rst_n) begin
        if(rst_n==0)
        mul_scale_result[i]<='0;
        else if(in_valid)
        mul_scale_result[i]<= $signed(quant_scale) * $signed(unquantized_input[i]);
        else
        mul_scale_result[i]<=mul_scale_result[i];
    end
    always_comb  begin
        // if(INPUT_DATA_WIDTH>OUTPUT_DATA_WIDTH)begin
        //     quant_noclip[i] =  mul_scale_result[i] >>> scale_shift_delay;
        // end
        // else begin
        //     quant_noclip[i] =  mul_scale_result[i] <<< scale_shift_delay;
        // end

        quant_noclip[i] = (scale_shift_delay <= 0) ? 
                  ($signed(mul_scale_result[i]) >>> (-scale_shift_delay)) :  
                  ($signed(mul_scale_result[i]) <<< scale_shift_delay);

        if ($signed(quant_noclip[i]) > $signed(MAX_OUTPUT)) begin
            quantized_output[i] = MAX_OUTPUT;
        end
        else if ($signed(quant_noclip[i]) < $signed(MIN_OUTPUT)) begin
            quantized_output[i] = MIN_OUTPUT;
        end
        else begin
            quantized_output[i] = quant_noclip[i][OUTPUT_DATA_WIDTH-1:0];
        end
        //mul_scale_result[i] = $signed(quant_scale) * $signed(unquantized_input[i]);
        

        // if ((~quant_noclip[i][OUTPUT_DATA_WIDTH+2]) && (|quant_noclip[i][OUTPUT_DATA_WIDTH+2:OUTPUT_DATA_WIDTH])) begin
        //     quantized_output[i] = MAX_OUTPUT;
        // end
        // else if ((quant_noclip[i][OUTPUT_DATA_WIDTH+2]) && (~&quant_noclip[i][OUTPUT_DATA_WIDTH+2:OUTPUT_DATA_WIDTH])) begin
        //     quantized_output[i] = MIN_OUTPUT;
        // end
        // else begin
        //     quantized_output[i] = quant_noclip[i][OUTPUT_DATA_WIDTH-1:0];
        // end
    end      
end
endgenerate
endmodule
