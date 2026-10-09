module quantization_intX_to_fxY #(
    parameter SCALE_DATA_WIDTH       = 16 ,
    parameter INPUT_DATA_WIDTH       = 8  ,
    parameter INPUT_NUMBER           = 32 ,
    parameter OUTPUT_DATA_WIDTH      = 16 ,
    parameter SCALE_SHIFT_WIDTH      = SCALE_DATA_WIDTH,
    parameter FRAC_WIDTH             = 4)
(  
    input  logic                                                      clk,
    input  logic                                                      rst_n,
    input  logic                                                      in_valid,
    input  logic signed [INPUT_NUMBER-1:0][INPUT_DATA_WIDTH-1:0]      quantized_input   ,
    input  logic signed [SCALE_DATA_WIDTH-1:0]                        quant_scale       ,
    input  logic signed [SCALE_DATA_WIDTH-1:0]                        scale_shift       ,

    output logic                                                      out_valid         ,
    output logic signed [INPUT_NUMBER-1:0][OUTPUT_DATA_WIDTH-1:0]     quantized_output
);

localparam signed [OUTPUT_DATA_WIDTH-FRAC_WIDTH-1:0] MIN_OUTPUT = {1'b1,{(OUTPUT_DATA_WIDTH-FRAC_WIDTH-1){1'b0}}};
localparam signed [OUTPUT_DATA_WIDTH-FRAC_WIDTH-1:0] MAX_OUTPUT = {1'b0,{(OUTPUT_DATA_WIDTH-FRAC_WIDTH-1){1'b1}}};
localparam MUL_WIDTH = INPUT_DATA_WIDTH+SCALE_DATA_WIDTH>OUTPUT_DATA_WIDTH+3? INPUT_DATA_WIDTH+SCALE_DATA_WIDTH:OUTPUT_DATA_WIDTH+3;

//----------caculate  *scale--------------------------//
logic signed [INPUT_NUMBER-1:0][MUL_WIDTH-1:0] mul_scale_result;
logic signed [INPUT_NUMBER-1:0][OUTPUT_DATA_WIDTH+2:0] quant_noclip ;
logic signed[SCALE_SHIFT_WIDTH-1:0]scale_shift_delay;
logic [INPUT_NUMBER-1:0][FRAC_WIDTH-1:0]frac_part;
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
for(i=0; i<INPUT_NUMBER; i=i+1) begin
    always_ff @(posedge clk or negedge rst_n) begin
        if(rst_n==0)
        mul_scale_result[i]<='0;
        else if(in_valid)
        mul_scale_result[i]<= $signed(quant_scale) * $signed(quantized_input[i]);
        else
        mul_scale_result[i]<=mul_scale_result[i];
    end
    always_comb begin
            //mul_scale_result[i]  = $signed(quant_scale) * $signed(quantized_input[i]);// caculate int type result
            //systolic_array_quant_noclip = number_multed_scale_result >>> systolic_array_quant_scale_shift;
            // if(scale_shift >= FRAC_WIDTH) begin
            //     quantized_fx16 [i][FRAC_WIDTH-1:0] = mul_scale_result[i][scale_shift-1 -:FRAC_WIDTH];
            // end else if (scale_shift > 0) begin
            //     quantized_fx16 [i][FRAC_WIDTH-1:0] = mul_scale_result[i]<<(FRAC_WIDTH-scale_shift);
            // end else begin
            //     quantized_fx16 [i][FRAC_WIDTH-1:0] = {FRAC_WIDTH{1'b0}};;
            // end
            if(-scale_shift_delay >= FRAC_WIDTH) begin
                frac_part [i] = mul_scale_result[i][-scale_shift_delay-1 -:FRAC_WIDTH];
            end else if (-scale_shift_delay > 0) begin
                frac_part [i] = mul_scale_result[i]<<(FRAC_WIDTH+scale_shift_delay);
            end else begin
                frac_part [i] = {FRAC_WIDTH{1'b0}};
            end

            //quant_noclip[i] = mul_scale_result[i] >>> scale_shift;
            quant_noclip[i] = (-scale_shift_delay >= 0) ? ($signed(mul_scale_result[i]) >>> (-scale_shift_delay)) :
                    ($signed(mul_scale_result[i]) <<< (scale_shift_delay));    

            //The comparison process is not parameterized, and will be added later
            if ($signed(quant_noclip[i]) > $signed(MAX_OUTPUT)) begin
                quantized_output[i]  = {MAX_OUTPUT,{(FRAC_WIDTH){1'b1}}};
            end
            else if ($signed(quant_noclip[i]) < $signed(MIN_OUTPUT)) begin
                quantized_output[i] = {MIN_OUTPUT,{(FRAC_WIDTH){1'b0}}};
            end
            else begin
                quantized_output[i]  = {quant_noclip[i][OUTPUT_DATA_WIDTH-FRAC_WIDTH-1:0],frac_part[i]};
            end
    end
    // always_ff @(posedge clk or negedge rst_n) begin
    //     if(rst_n==0)begin
    //         quantized_output[i]<='0;
    //     end else if(in_valid)begin
    //         if ($signed(quant_noclip[i]) > $signed(MAX_OUTPUT)) begin
    //             quantized_output[i] [OUTPUT_DATA_WIDTH-1:FRAC_WIDTH] <= MAX_OUTPUT;
    //         end
    //         else if ($signed(quant_noclip[i]) < $signed(MIN_OUTPUT)) begin
    //             quantized_output[i] [OUTPUT_DATA_WIDTH-1:FRAC_WIDTH] <= MIN_OUTPUT;
    //         end
    //         else begin
    //             quantized_output[i] [OUTPUT_DATA_WIDTH-1:FRAC_WIDTH] <= quant_noclip[i][OUTPUT_DATA_WIDTH-FRAC_WIDTH-1:0];
    //         end

    //         if(-scale_shift >= FRAC_WIDTH) begin
    //             quantized_output [i][FRAC_WIDTH-1:0] <= mul_scale_result[i][-scale_shift-1 -:FRAC_WIDTH];
    //         end else if (-scale_shift > 0) begin
    //             quantized_output [i][FRAC_WIDTH-1:0] <= mul_scale_result[i]<<(FRAC_WIDTH+scale_shift);
    //         end else begin
    //             quantized_output [i][FRAC_WIDTH-1:0] <= {FRAC_WIDTH{1'b0}};
    //         end
    //     end
    //     else begin
    //         quantized_output[i]<=quantized_output[i];
    //     end
    // end
end
endgenerate
endmodule