`ifndef FP16_TO_FP32_SV
`define FP16_TO_FP32_SV

module fp16_to_fp32 (
    input  logic [15:0] fp16_i,
    output logic [31:0] fp32_o
);

    logic sign_w;
    logic [4:0] exp_w;
    logic [9:0] frac_w;
    logic [7:0] exp32_w;

    always_comb begin
        sign_w = fp16_i[15];
        exp_w  = fp16_i[14:10];
        frac_w = fp16_i[9:0];
        exp32_w = '0;
        fp32_o = {sign_w, 31'd0};

        if (exp_w == 5'h1f) begin
            fp32_o = {sign_w, 8'hff, (frac_w == '0) ? 23'd0 : {1'b1, frac_w, 12'd0}};
        end else if (exp_w != 5'd0) begin
            exp32_w = 8'(int'(exp_w) - 15 + 127);
            fp32_o = {sign_w, exp32_w, frac_w, 13'd0};
        end
    end

endmodule

`endif
