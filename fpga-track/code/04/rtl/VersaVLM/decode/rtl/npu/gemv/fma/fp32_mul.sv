`ifndef FP32_MUL_SV
`define FP32_MUL_SV

module fp32_mul (
    input  logic [31:0] a_i,
    input  logic [31:0] b_i,
    output logic [31:0] z_o
);

    logic sign_w;
    logic [7:0] exp_a_w;
    logic [7:0] exp_b_w;
    logic [22:0] frac_a_w;
    logic [22:0] frac_b_w;
    logic [23:0] mant_a_w;
    logic [23:0] mant_b_w;
    logic [47:0] prod_w;
    logic [22:0] frac_norm_w;
    logic round_bit_w;
    logic sticky_w;
    logic [23:0] frac_round_w;
    int exp_w;

    always_comb begin
        sign_w = a_i[31] ^ b_i[31];
        exp_a_w = a_i[30:23];
        exp_b_w = b_i[30:23];
        frac_a_w = a_i[22:0];
        frac_b_w = b_i[22:0];
        mant_a_w = {1'b1, frac_a_w};
        mant_b_w = {1'b1, frac_b_w};
        prod_w = mant_a_w * mant_b_w;
        exp_w = int'(exp_a_w) + int'(exp_b_w) - 127;
        frac_norm_w = '0;
        round_bit_w = 1'b0;
        sticky_w = 1'b0;
        frac_round_w = '0;
        z_o = {sign_w, 31'd0};

        if ((exp_a_w == 8'd0) || (exp_b_w == 8'd0)) begin
            z_o = {sign_w, 31'd0};
        end else if ((exp_a_w == 8'hff) || (exp_b_w == 8'hff)) begin
            z_o = {sign_w, 8'hff, 23'd0};
        end else begin
            if (prod_w[47]) begin
                exp_w = exp_w + 1;
                frac_norm_w = prod_w[46:24];
                round_bit_w = prod_w[23];
                sticky_w = |prod_w[22:0];
            end else begin
                frac_norm_w = prod_w[45:23];
                round_bit_w = prod_w[22];
                sticky_w = |prod_w[21:0];
            end

            frac_round_w = {1'b0, frac_norm_w} +
                           ((round_bit_w && (sticky_w || frac_norm_w[0])) ? 24'd1 : 24'd0);
            if (frac_round_w[23]) begin
                exp_w = exp_w + 1;
                frac_norm_w = 23'd0;
            end else begin
                frac_norm_w = frac_round_w[22:0];
            end

            if (exp_w >= 255)
                z_o = {sign_w, 8'hff, 23'd0};
            else if (exp_w <= 0)
                z_o = {sign_w, 31'd0};
            else
                z_o = {sign_w, 8'(exp_w), frac_norm_w};
        end
    end

endmodule

`endif
