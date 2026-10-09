`ifndef FP32_TO_FP16_SV
`define FP32_TO_FP16_SV

module fp32_to_fp16 (
    input  logic [31:0] fp32_i,
    output logic [15:0] fp16_o
);

    logic sign_w;
    logic [7:0] exp_w;
    logic [22:0] frac_w;
    int exp16_w;
    logic [10:0] mant_round_w;

    always_comb begin
        sign_w = fp32_i[31];
        exp_w = fp32_i[30:23];
        frac_w = fp32_i[22:0];
        exp16_w = int'(exp_w) - 127 + 15;
        mant_round_w = '0;
        fp16_o = {sign_w, 15'd0};

        if (exp_w == 8'hff) begin
            fp16_o = {sign_w, 5'h1f, (frac_w == '0) ? 10'd0 : 10'h200};
        end else if (exp_w != 8'd0) begin
            if (exp16_w >= 31) begin
                fp16_o = {sign_w, 5'h1f, 10'd0};
            end else if (exp16_w > 0) begin
                mant_round_w = {1'b0, frac_w[22:13]} +
                               ((frac_w[12] && ((|frac_w[11:0]) || frac_w[13])) ? 11'd1 : 11'd0);
                if (mant_round_w[10]) begin
                    if (exp16_w == 30)
                        fp16_o = {sign_w, 5'h1f, 10'd0};
                    else
                        fp16_o = {sign_w, exp16_w[4:0] + 5'd1, 10'd0};
                end else begin
                    fp16_o = {sign_w, exp16_w[4:0], mant_round_w[9:0]};
                end
            end
        end
    end

endmodule

`endif
