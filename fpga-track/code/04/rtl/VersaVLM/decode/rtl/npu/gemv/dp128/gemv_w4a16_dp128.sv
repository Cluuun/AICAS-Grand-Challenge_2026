`ifndef GEMV_W4A16_DP128_SV
`define GEMV_W4A16_DP128_SV

package gemv_fp16_pkg;
    localparam int FP16_GUARD_BITS = 6;
    localparam int FP16_ADD_W      = 18;
    localparam int FP16_SUM_W      = FP16_ADD_W + 1;
    localparam int FP16_TARGET_BIT = FP16_GUARD_BITS + 10;

function automatic logic fp16_is_normal(input logic [15:0] fp16);
    begin
        fp16_is_normal = (fp16[14:10] != 5'd0) && (fp16[14:10] != 5'h1f);
    end
endfunction

    function automatic logic [15:0] fp16_zero_if_non_normal(input logic [15:0] fp16);
        begin
            fp16_zero_if_non_normal = fp16_is_normal(fp16) ? fp16 : 16'h0000;
        end
    endfunction

    function automatic logic [4:0] leading_one9(input logic [8:0] value);
        logic found;
        begin
            leading_one9 = '0;
            found = 1'b0;
            for (int i = 8; i >= 0; i--) begin
                if (!found && value[i]) begin
                    leading_one9 = i[4:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [4:0] leading_one_sum(input logic [FP16_SUM_W-1:0] value);
        logic found;
        begin
            leading_one_sum = '0;
            found = 1'b0;
            for (int i = FP16_SUM_W-1; i >= 0; i--) begin
                if (!found && value[i]) begin
                    leading_one_sum = i[4:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [FP16_SUM_W-1:0] rshift_sticky_sum(
        input logic [FP16_SUM_W-1:0] value,
        input int                    shift
    );
        logic sticky;
        begin
            sticky = 1'b0;
            if (shift <= 0) begin
                rshift_sticky_sum = value;
            end else if (shift >= FP16_SUM_W) begin
                rshift_sticky_sum = '0;
                rshift_sticky_sum[0] = |value;
            end else begin
                rshift_sticky_sum = value >> shift;
                for (int i = 0; i < FP16_SUM_W; i++) begin
                    if (i < shift)
                        sticky |= value[i];
                end
                if (sticky)
                    rshift_sticky_sum[0] = 1'b1;
            end
        end
    endfunction

    function automatic logic [15:0] fp16_from_int9(input logic signed [8:0] value);
        logic sign;
        logic [8:0] abs_value;
        logic [18:0] shifted_abs;
        logic [4:0] lead;
        logic [4:0] exp16;
        logic [9:0] frac;
        begin
            sign = value[8];
            abs_value = sign ? (~value + 9'd1) : value[8:0];
            lead = leading_one9(abs_value);
            exp16 = lead + 5'd15;
            shifted_abs = '0;
            frac = '0;

            if (abs_value == 9'd0) begin
                fp16_from_int9 = 16'h0000;
            end else begin
                shifted_abs = abs_value << (10 - int'(lead));
                frac = shifted_abs[9:0];
                fp16_from_int9 = {sign, exp16, frac};
            end
        end
    endfunction

    function automatic logic [15:0] fp16_mul(
        input logic [15:0] a,
        input logic [15:0] b
    );
        logic sign;
        logic [4:0] exp_a;
        logic [4:0] exp_b;
        logic [10:0] mant_a;
        logic [10:0] mant_b;
        logic [21:0] prod;
        logic [10:0] mant_pre;
        logic [11:0] mant_round;
        logic round_bit;
        logic sticky_bit;
        int exp_unbiased;
        begin
            sign = a[15] ^ b[15];
            exp_a = a[14:10];
            exp_b = b[14:10];
            mant_a = {1'b1, a[9:0]};
            mant_b = {1'b1, b[9:0]};
            prod = mant_a * mant_b;
            mant_pre = '0;
            mant_round = '0;
            round_bit = 1'b0;
            sticky_bit = 1'b0;
            exp_unbiased = int'(exp_a) + int'(exp_b) - 15;
            fp16_mul = 16'h0000;

            if (fp16_is_normal(a) && fp16_is_normal(b)) begin
                if (prod[21]) begin
                    exp_unbiased++;
                    mant_pre = prod[21:11];
                    round_bit = prod[10];
                    sticky_bit = |prod[9:0];
                end else begin
                    mant_pre = prod[20:10];
                    round_bit = prod[9];
                    sticky_bit = |prod[8:0];
                end

                mant_round = {1'b0, mant_pre} +
                             ((round_bit && (sticky_bit || mant_pre[0])) ? 12'd1 : 12'd0);
                if (mant_round[11]) begin
                    exp_unbiased++;
                    mant_pre = 11'b100_0000_0000;
                end else begin
                    mant_pre = mant_round[10:0];
                end

                if (exp_unbiased >= 31)
                    fp16_mul = {sign, 5'h1f, 10'd0};
                else if (exp_unbiased <= 0)
                    fp16_mul = 16'h0000;
                else
                    fp16_mul = {sign, exp_unbiased[4:0], mant_pre[9:0]};
            end
        end
    endfunction

    function automatic logic [15:0] fp16_add(
        input logic [15:0] a,
        input logic [15:0] b
    );
        logic [15:0] a_n;
        logic [15:0] b_n;
        logic sign_a;
        logic sign_b;
        logic sign_z;
        logic [4:0] exp_a;
        logic [4:0] exp_b;
        logic [4:0] exp_max;
        logic [10:0] mant_a;
        logic [10:0] mant_b;
        logic [FP16_SUM_W-1:0] ext_a;
        logic [FP16_SUM_W-1:0] ext_b;
        logic signed [FP16_SUM_W:0] signed_a;
        logic signed [FP16_SUM_W:0] signed_b;
        logic signed [FP16_SUM_W:0] signed_sum;
        logic [FP16_SUM_W-1:0] abs_sum;
        logic [FP16_SUM_W-1:0] norm_sum;
        logic [4:0] lead;
        logic [10:0] mant_pre;
        logic [11:0] mant_round;
        logic round_bit;
        logic sticky_bit;
        int exp_norm;
        int shift_amt;
        begin
            a_n = fp16_zero_if_non_normal(a);
            b_n = fp16_zero_if_non_normal(b);
            sign_a = a_n[15];
            sign_b = b_n[15];
            exp_a = a_n[14:10];
            exp_b = b_n[14:10];
            mant_a = {1'b1, a_n[9:0]};
            mant_b = {1'b1, b_n[9:0]};
            exp_max = (exp_a > exp_b) ? exp_a : exp_b;
            ext_a = '0;
            ext_b = '0;
            signed_a = '0;
            signed_b = '0;
            signed_sum = '0;
            abs_sum = '0;
            norm_sum = '0;
            lead = '0;
            mant_pre = '0;
            mant_round = '0;
            round_bit = 1'b0;
            sticky_bit = 1'b0;
            exp_norm = int'(exp_max);
            shift_amt = 0;
            fp16_add = 16'h0000;

            if (a_n == 16'h0000) begin
                fp16_add = b_n;
            end else if (b_n == 16'h0000) begin
                fp16_add = a_n;
            end else begin
                ext_a = {{(FP16_SUM_W-(11+FP16_GUARD_BITS)){1'b0}}, mant_a, {FP16_GUARD_BITS{1'b0}}};
                ext_b = {{(FP16_SUM_W-(11+FP16_GUARD_BITS)){1'b0}}, mant_b, {FP16_GUARD_BITS{1'b0}}};
                ext_a = rshift_sticky_sum(ext_a, int'(exp_max) - int'(exp_a));
                ext_b = rshift_sticky_sum(ext_b, int'(exp_max) - int'(exp_b));

                signed_a = sign_a ? -$signed({1'b0, ext_a}) : $signed({1'b0, ext_a});
                signed_b = sign_b ? -$signed({1'b0, ext_b}) : $signed({1'b0, ext_b});
                signed_sum = signed_a + signed_b;

                if (signed_sum != '0) begin
                    sign_z = signed_sum[FP16_SUM_W];
                    abs_sum = sign_z ? (~signed_sum[FP16_SUM_W-1:0] + 1'b1) :
                                       signed_sum[FP16_SUM_W-1:0];
                    lead = leading_one_sum(abs_sum);

                    if (int'(lead) > FP16_TARGET_BIT) begin
                        shift_amt = int'(lead) - FP16_TARGET_BIT;
                        norm_sum = rshift_sticky_sum(abs_sum, shift_amt);
                        exp_norm = int'(exp_max) + shift_amt;
                    end else begin
                        shift_amt = FP16_TARGET_BIT - int'(lead);
                        norm_sum = abs_sum << shift_amt;
                        exp_norm = int'(exp_max) - shift_amt;
                    end

                    if (exp_norm <= 0) begin
                        fp16_add = 16'h0000;
                    end else if (exp_norm >= 31) begin
                        fp16_add = {sign_z, 5'h1f, 10'd0};
                    end else begin
                        mant_pre = norm_sum[FP16_TARGET_BIT -: 11];
                        round_bit = norm_sum[FP16_GUARD_BITS-1];
                        sticky_bit = |norm_sum[FP16_GUARD_BITS-2:0];
                        mant_round = {1'b0, mant_pre} +
                                     ((round_bit && (sticky_bit || mant_pre[0])) ? 12'd1 : 12'd0);

                        if (mant_round[11]) begin
                            exp_norm++;
                            if (exp_norm >= 31)
                                fp16_add = {sign_z, 5'h1f, 10'd0};
                            else
                                fp16_add = {sign_z, exp_norm[4:0], 10'd0};
                        end else begin
                            fp16_add = {sign_z, exp_norm[4:0], mant_round[9:0]};
                        end
                    end
                end
            end
        end
    endfunction
endpackage

module gemv_fp16_add_dsp (
    input  logic [15:0] a_i,
    input  logic [15:0] b_i,
    output logic [15:0] z_o
);
    import gemv_fp16_pkg::*;

    logic [15:0] a_n_w;
    logic [15:0] b_n_w;
    logic        sign_a_w;
    logic        sign_b_w;
    logic [4:0]  exp_a_w;
    logic [4:0]  exp_b_w;
    logic [4:0]  exp_max_w;
    logic [10:0] mant_a_w;
    logic [10:0] mant_b_w;
    logic [FP16_SUM_W-1:0] ext_a_base_w;
    logic [FP16_SUM_W-1:0] ext_b_base_w;
    logic [FP16_SUM_W-1:0] ext_a_w;
    logic [FP16_SUM_W-1:0] ext_b_w;
    logic signed [FP16_SUM_W:0] signed_a_w;
    logic signed [FP16_SUM_W:0] signed_b_w;
    (* use_dsp = "yes" *) wire signed [FP16_SUM_W:0] signed_sum_w;

    assign a_n_w = fp16_zero_if_non_normal(a_i);
    assign b_n_w = fp16_zero_if_non_normal(b_i);
    assign sign_a_w = a_n_w[15];
    assign sign_b_w = b_n_w[15];
    assign exp_a_w = a_n_w[14:10];
    assign exp_b_w = b_n_w[14:10];
    assign exp_max_w = (exp_a_w > exp_b_w) ? exp_a_w : exp_b_w;
    assign mant_a_w = {1'b1, a_n_w[9:0]};
    assign mant_b_w = {1'b1, b_n_w[9:0]};
    assign ext_a_base_w = {{(FP16_SUM_W-(11+FP16_GUARD_BITS)){1'b0}}, mant_a_w,
                           {FP16_GUARD_BITS{1'b0}}};
    assign ext_b_base_w = {{(FP16_SUM_W-(11+FP16_GUARD_BITS)){1'b0}}, mant_b_w,
                           {FP16_GUARD_BITS{1'b0}}};
    assign ext_a_w = rshift_sticky_sum(ext_a_base_w, int'(exp_max_w) - int'(exp_a_w));
    assign ext_b_w = rshift_sticky_sum(ext_b_base_w, int'(exp_max_w) - int'(exp_b_w));
    assign signed_a_w = sign_a_w ? -$signed({1'b0, ext_a_w}) : $signed({1'b0, ext_a_w});
    assign signed_b_w = sign_b_w ? -$signed({1'b0, ext_b_w}) : $signed({1'b0, ext_b_w});
    assign signed_sum_w = signed_a_w + signed_b_w;

    always_comb begin
        logic sign_z;
        logic [FP16_SUM_W-1:0] abs_sum;
        logic [FP16_SUM_W-1:0] norm_sum;
        logic [4:0] lead;
        logic [10:0] mant_pre;
        logic [11:0] mant_round;
        logic round_bit;
        logic sticky_bit;
        int exp_norm;
        int shift_amt;

        sign_z = 1'b0;
        abs_sum = '0;
        norm_sum = '0;
        lead = '0;
        mant_pre = '0;
        mant_round = '0;
        round_bit = 1'b0;
        sticky_bit = 1'b0;
        exp_norm = int'(exp_max_w);
        shift_amt = 0;
        z_o = 16'h0000;

        if (a_n_w == 16'h0000) begin
            z_o = b_n_w;
        end else if (b_n_w == 16'h0000) begin
            z_o = a_n_w;
        end else if (signed_sum_w != '0) begin
            sign_z = signed_sum_w[FP16_SUM_W];
            abs_sum = sign_z ? (~signed_sum_w[FP16_SUM_W-1:0] + 1'b1) :
                               signed_sum_w[FP16_SUM_W-1:0];
            lead = leading_one_sum(abs_sum);

            if (int'(lead) > FP16_TARGET_BIT) begin
                shift_amt = int'(lead) - FP16_TARGET_BIT;
                norm_sum = rshift_sticky_sum(abs_sum, shift_amt);
                exp_norm = int'(exp_max_w) + shift_amt;
            end else begin
                shift_amt = FP16_TARGET_BIT - int'(lead);
                norm_sum = abs_sum << shift_amt;
                exp_norm = int'(exp_max_w) - shift_amt;
            end

            if (exp_norm <= 0) begin
                z_o = 16'h0000;
            end else if (exp_norm >= 31) begin
                z_o = {sign_z, 5'h1f, 10'd0};
            end else begin
                mant_pre = norm_sum[FP16_TARGET_BIT -: 11];
                round_bit = norm_sum[FP16_GUARD_BITS-1];
                sticky_bit = |norm_sum[FP16_GUARD_BITS-2:0];
                mant_round = {1'b0, mant_pre} +
                             ((round_bit && (sticky_bit || mant_pre[0])) ? 12'd1 : 12'd0);

                if (mant_round[11]) begin
                    exp_norm++;
                    if (exp_norm >= 31)
                        z_o = {sign_z, 5'h1f, 10'd0};
                    else
                        z_o = {sign_z, exp_norm[4:0], 10'd0};
                end else begin
                    z_o = {sign_z, exp_norm[4:0], mant_round[9:0]};
                end
            end
        end
    end
endmodule

module gemv_fp16_add_pipe3 (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        valid_i,
    input  logic        add_en_i,
    input  logic [15:0] a_i,
    input  logic [15:0] b_i,
    output logic [15:0] z_o,
    output logic        valid_o
);
    import gemv_fp16_pkg::*;

    logic        valid_s1_q;
    logic        valid_s2_q;
    logic [15:0] pass_s1_q;
    logic [15:0] pass_s2_q;
    logic        special_s1_q;
    logic        special_s2_q;
    logic [4:0]  exp_max_s1_q;
    logic [4:0]  exp_max_s2_q;
    logic signed [FP16_SUM_W:0] signed_sum_s1_q;
    logic        sign_s2_q;
    logic [FP16_SUM_W-1:0] abs_sum_s2_q;
    logic [4:0]  lead_s2_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_s1_q <= 1'b0;
            valid_s2_q <= 1'b0;
            valid_o <= 1'b0;
            z_o <= 16'h0000;
            pass_s1_q <= 16'h0000;
            pass_s2_q <= 16'h0000;
            special_s1_q <= 1'b0;
            special_s2_q <= 1'b0;
            exp_max_s1_q <= 5'd0;
            exp_max_s2_q <= 5'd0;
            signed_sum_s1_q <= '0;
            sign_s2_q <= 1'b0;
            abs_sum_s2_q <= '0;
            lead_s2_q <= '0;
        end else begin
            logic [15:0] a_n;
            logic [15:0] b_n;
            logic [4:0] exp_max;
            logic [10:0] mant_a;
            logic [10:0] mant_b;
            logic [FP16_SUM_W-1:0] ext_a;
            logic [FP16_SUM_W-1:0] ext_b;
            logic signed [FP16_SUM_W:0] signed_a;
            logic signed [FP16_SUM_W:0] signed_b;
            logic sign_z;
            logic [FP16_SUM_W-1:0] abs_sum;
            logic [FP16_SUM_W-1:0] norm_sum;
            logic [10:0] mant_pre;
            logic [11:0] mant_round;
            logic round_bit;
            logic sticky_bit;
            int exp_norm;
            int shift_amt;

            valid_s1_q <= valid_i;
            valid_s2_q <= valid_s1_q;
            valid_o <= valid_s2_q;

            a_n = fp16_zero_if_non_normal(a_i);
            b_n = fp16_zero_if_non_normal(b_i);
            exp_max = (a_n[14:10] > b_n[14:10]) ? a_n[14:10] : b_n[14:10];
            mant_a = {1'b1, a_n[9:0]};
            mant_b = {1'b1, b_n[9:0]};
            ext_a = {{(FP16_SUM_W-(11+FP16_GUARD_BITS)){1'b0}}, mant_a, {FP16_GUARD_BITS{1'b0}}};
            ext_b = {{(FP16_SUM_W-(11+FP16_GUARD_BITS)){1'b0}}, mant_b, {FP16_GUARD_BITS{1'b0}}};
            ext_a = rshift_sticky_sum(ext_a, int'(exp_max) - int'(a_n[14:10]));
            ext_b = rshift_sticky_sum(ext_b, int'(exp_max) - int'(b_n[14:10]));
            signed_a = a_n[15] ? -$signed({1'b0, ext_a}) : $signed({1'b0, ext_a});
            signed_b = b_n[15] ? -$signed({1'b0, ext_b}) : $signed({1'b0, ext_b});

            special_s1_q <= 1'b0;
            pass_s1_q <= 16'h0000;
            signed_sum_s1_q <= '0;
            exp_max_s1_q <= exp_max;
            if (!add_en_i) begin
                special_s1_q <= 1'b1;
                pass_s1_q <= a_n;
            end else if (a_n == 16'h0000) begin
                special_s1_q <= 1'b1;
                pass_s1_q <= b_n;
            end else if (b_n == 16'h0000) begin
                special_s1_q <= 1'b1;
                pass_s1_q <= a_n;
            end else begin
                signed_sum_s1_q <= signed_a + signed_b;
            end

            special_s2_q <= special_s1_q || (signed_sum_s1_q == '0);
            pass_s2_q <= special_s1_q ? pass_s1_q : 16'h0000;
            exp_max_s2_q <= exp_max_s1_q;
            sign_s2_q <= signed_sum_s1_q[FP16_SUM_W];
            abs_sum_s2_q <= signed_sum_s1_q[FP16_SUM_W] ?
                (~signed_sum_s1_q[FP16_SUM_W-1:0] + 1'b1) :
                signed_sum_s1_q[FP16_SUM_W-1:0];
            lead_s2_q <= leading_one_sum(
                signed_sum_s1_q[FP16_SUM_W] ?
                (~signed_sum_s1_q[FP16_SUM_W-1:0] + 1'b1) :
                signed_sum_s1_q[FP16_SUM_W-1:0]
            );

            sign_z = sign_s2_q;
            abs_sum = abs_sum_s2_q;
            norm_sum = '0;
            mant_pre = '0;
            mant_round = '0;
            round_bit = 1'b0;
            sticky_bit = 1'b0;
            exp_norm = int'(exp_max_s2_q);
            shift_amt = 0;
            z_o <= 16'h0000;
            if (valid_s2_q) begin
                if (special_s2_q) begin
                    z_o <= pass_s2_q;
                end else begin
                    if (int'(lead_s2_q) > FP16_TARGET_BIT) begin
                        shift_amt = int'(lead_s2_q) - FP16_TARGET_BIT;
                        norm_sum = rshift_sticky_sum(abs_sum, shift_amt);
                        exp_norm = int'(exp_max_s2_q) + shift_amt;
                    end else begin
                        shift_amt = FP16_TARGET_BIT - int'(lead_s2_q);
                        norm_sum = abs_sum << shift_amt;
                        exp_norm = int'(exp_max_s2_q) - shift_amt;
                    end

                    if (exp_norm <= 0) begin
                        z_o <= 16'h0000;
                    end else if (exp_norm >= 31) begin
                        z_o <= {sign_z, 5'h1f, 10'd0};
                    end else begin
                        mant_pre = norm_sum[FP16_TARGET_BIT -: 11];
                        round_bit = norm_sum[FP16_GUARD_BITS-1];
                        sticky_bit = |norm_sum[FP16_GUARD_BITS-2:0];
                        mant_round = {1'b0, mant_pre} +
                                     ((round_bit && (sticky_bit || mant_pre[0])) ? 12'd1 : 12'd0);
                        if (mant_round[11]) begin
                            exp_norm++;
                            if (exp_norm >= 31)
                                z_o <= {sign_z, 5'h1f, 10'd0};
                            else
                                z_o <= {sign_z, exp_norm[4:0], 10'd0};
                        end else begin
                            z_o <= {sign_z, exp_norm[4:0], mant_round[9:0]};
                        end
                    end
                end
            end
        end
    end
endmodule

module gemv_fp16_mul_pipe2 #(
    parameter bit SUPPORT_INPUT_SUBNORMAL = 1'b0,
    parameter bit ENABLE_FP16_OUT = 1'b1,
    parameter bit ENABLE_UQ0P24_OUT = 1'b1
) (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        valid_i,
    input  logic        mul_en_i,
    input  logic        preserve_subnormal_i,
    input  logic [15:0] a_i,
    input  logic [15:0] b_i,
    output logic [15:0] z_o,
    output logic [23:0] uq0p24_o,
    output logic        valid_o
);
    gemv_fp16_mul_ip_pipe #(
        .IP_LATENCY (4),
        .CONV_LATENCY (3),
        .SUPPORT_INPUT_SUBNORMAL (SUPPORT_INPUT_SUBNORMAL),
        .ENABLE_FP16_OUT (ENABLE_FP16_OUT),
        .ENABLE_UQ0P24_OUT (ENABLE_UQ0P24_OUT)
    ) u_mul_ip (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .valid_i              (valid_i),
        .mul_en_i             (mul_en_i),
        .preserve_subnormal_i (preserve_subnormal_i),
        .a_i                  (a_i),
        .b_i                  (b_i),
        .z_o                  (z_o),
        .uq0p24_o             (uq0p24_o),
        .valid_o              (valid_o)
    );
endmodule

module gemv_dsp48e2_add6_chain3 (
    input  logic                    clk_i,
    input  logic                    rst_ni,
    input  logic signed [47:0]      in0_i,
    input  logic signed [47:0]      in1_i,
    input  logic signed [47:0]      in2_i,
    input  logic signed [47:0]      in3_i,
    input  logic signed [47:0]      in4_i,
    input  logic signed [47:0]      in5_i,
    output logic signed [47:0]      sum_o
);

    logic [47:0] p0_w;
    logic [47:0] p1_w;
    logic [47:0] p2_w;
    logic [47:0] pc0_w;
    logic [47:0] pc1_w;
    logic signed [47:0] in2_q;
    logic signed [47:0] in3_q;
    logic signed [47:0] in4_q [0:1];
    logic signed [47:0] in5_q [0:1];

    assign sum_o = p2_w;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            in2_q <= '0;
            in3_q <= '0;
            for (int i = 0; i < 2; i++) begin
                in4_q[i] <= '0;
                in5_q[i] <= '0;
            end
        end else begin
            in2_q <= in2_i;
            in3_q <= in3_i;
            in4_q[0] <= in4_i;
            in5_q[0] <= in5_i;
            in4_q[1] <= in4_q[0];
            in5_q[1] <= in5_q[0];
        end
    end

    DSP48E2 #(
        .ACASCREG      (0),
        .ADREG         (0),
        .ALUMODEREG    (0),
        .AREG          (0),
        .BCASCREG      (0),
        .BREG          (0),
        .CARRYINREG    (0),
        .CARRYINSELREG (0),
        .CREG          (0),
        .DREG          (0),
        .INMODEREG     (0),
        .MREG          (0),
        .OPMODEREG     (0),
        .PREG          (1),
        .USE_MULT      ("NONE"),
        .USE_SIMD      ("ONE48")
    ) u_add0 (
        .A            (in0_i[47:18]),
        .ACIN         ('0),
        .ALUMODE      (4'b0000),
        .B            (in0_i[17:0]),
        .BCIN         ('0),
        .C            (in1_i),
        .CARRYCASCIN  (1'b0),
        .CARRYIN      (1'b0),
        .CARRYINSEL   (3'b000),
        .CEA1         (1'b1),
        .CEA2         (1'b1),
        .CEAD         (1'b1),
        .CEALUMODE    (1'b1),
        .CEB1         (1'b1),
        .CEB2         (1'b1),
        .CEC          (1'b1),
        .CECARRYIN    (1'b1),
        .CECTRL       (1'b1),
        .CED          (1'b1),
        .CEINMODE     (1'b1),
        .CEM          (1'b1),
        .CEP          (1'b1),
        .CLK          (clk_i),
        .D            ('0),
        .INMODE       (5'b00000),
        .MULTSIGNIN   (1'b0),
        .OPMODE       (9'b110000011),
        .PCIN         ('0),
        .RSTA         (!rst_ni),
        .RSTALLCARRYIN(!rst_ni),
        .RSTALUMODE   (!rst_ni),
        .RSTB         (!rst_ni),
        .RSTC         (!rst_ni),
        .RSTCTRL      (!rst_ni),
        .RSTD         (!rst_ni),
        .RSTINMODE    (!rst_ni),
        .RSTM         (!rst_ni),
        .RSTP         (!rst_ni),
        .ACOUT        (),
        .BCOUT        (),
        .CARRYCASCOUT (),
        .CARRYOUT     (),
        .MULTSIGNOUT  (),
        .OVERFLOW     (),
        .P            (p0_w),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT        (pc0_w),
        .UNDERFLOW    (),
        .XOROUT       ()
    );

    DSP48E2 #(
        .ACASCREG      (0),
        .ADREG         (0),
        .ALUMODEREG    (0),
        .AREG          (0),
        .BCASCREG      (0),
        .BREG          (0),
        .CARRYINREG    (0),
        .CARRYINSELREG (0),
        .CREG          (0),
        .DREG          (0),
        .INMODEREG     (0),
        .MREG          (0),
        .OPMODEREG     (0),
        .PREG          (1),
        .USE_MULT      ("NONE"),
        .USE_SIMD      ("ONE48")
    ) u_add1 (
        .A            (in2_q[47:18]),
        .ACIN         ('0),
        .ALUMODE      (4'b0000),
        .B            (in2_q[17:0]),
        .BCIN         ('0),
        .C            (in3_q),
        .CARRYCASCIN  (1'b0),
        .CARRYIN      (1'b0),
        .CARRYINSEL   (3'b000),
        .CEA1         (1'b1),
        .CEA2         (1'b1),
        .CEAD         (1'b1),
        .CEALUMODE    (1'b1),
        .CEB1         (1'b1),
        .CEB2         (1'b1),
        .CEC          (1'b1),
        .CECARRYIN    (1'b1),
        .CECTRL       (1'b1),
        .CED          (1'b1),
        .CEINMODE     (1'b1),
        .CEM          (1'b1),
        .CEP          (1'b1),
        .CLK          (clk_i),
        .D            ('0),
        .INMODE       (5'b00000),
        .MULTSIGNIN   (1'b0),
        .OPMODE       (9'b110010011),
        .PCIN         (pc0_w),
        .RSTA         (!rst_ni),
        .RSTALLCARRYIN(!rst_ni),
        .RSTALUMODE   (!rst_ni),
        .RSTB         (!rst_ni),
        .RSTC         (!rst_ni),
        .RSTCTRL      (!rst_ni),
        .RSTD         (!rst_ni),
        .RSTINMODE    (!rst_ni),
        .RSTM         (!rst_ni),
        .RSTP         (!rst_ni),
        .ACOUT        (),
        .BCOUT        (),
        .CARRYCASCOUT (),
        .CARRYOUT     (),
        .MULTSIGNOUT  (),
        .OVERFLOW     (),
        .P            (p1_w),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT        (pc1_w),
        .UNDERFLOW    (),
        .XOROUT       ()
    );

    DSP48E2 #(
        .ACASCREG      (0),
        .ADREG         (0),
        .ALUMODEREG    (0),
        .AREG          (0),
        .BCASCREG      (0),
        .BREG          (0),
        .CARRYINREG    (0),
        .CARRYINSELREG (0),
        .CREG          (0),
        .DREG          (0),
        .INMODEREG     (0),
        .MREG          (0),
        .OPMODEREG     (0),
        .PREG          (1),
        .USE_MULT      ("NONE"),
        .USE_SIMD      ("ONE48")
    ) u_add2 (
        .A            (in4_q[1][47:18]),
        .ACIN         ('0),
        .ALUMODE      (4'b0000),
        .B            (in4_q[1][17:0]),
        .BCIN         ('0),
        .C            (in5_q[1]),
        .CARRYCASCIN  (1'b0),
        .CARRYIN      (1'b0),
        .CARRYINSEL   (3'b000),
        .CEA1         (1'b1),
        .CEA2         (1'b1),
        .CEAD         (1'b1),
        .CEALUMODE    (1'b1),
        .CEB1         (1'b1),
        .CEB2         (1'b1),
        .CEC          (1'b1),
        .CECARRYIN    (1'b1),
        .CECTRL       (1'b1),
        .CED          (1'b1),
        .CEINMODE     (1'b1),
        .CEM          (1'b1),
        .CEP          (1'b1),
        .CLK          (clk_i),
        .D            ('0),
        .INMODE       (5'b00000),
        .MULTSIGNIN   (1'b0),
        .OPMODE       (9'b110010011),
        .PCIN         (pc1_w),
        .RSTA         (!rst_ni),
        .RSTALLCARRYIN(!rst_ni),
        .RSTALUMODE   (!rst_ni),
        .RSTB         (!rst_ni),
        .RSTC         (!rst_ni),
        .RSTCTRL      (!rst_ni),
        .RSTD         (!rst_ni),
        .RSTINMODE    (!rst_ni),
        .RSTM         (!rst_ni),
        .RSTP         (!rst_ni),
        .ACOUT        (),
        .BCOUT        (),
        .CARRYCASCOUT (),
        .CARRYOUT     (),
        .MULTSIGNOUT  (),
        .OVERFLOW     (),
        .P            (p2_w),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT        (),
        .UNDERFLOW    (),
        .XOROUT       ()
    );

endmodule

module gemv_dsp48e2_mac4_chain #(
    parameter int ACT_W    = 27,
    parameter int WEIGHT_W = 18,
    parameter int SUM_W    = 48
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         valid_i,
    input  logic signed [ACT_W-1:0]      act0_i,
    input  logic signed [ACT_W-1:0]      act1_i,
    input  logic signed [ACT_W-1:0]      act2_i,
    input  logic signed [ACT_W-1:0]      act3_i,
    input  logic signed [WEIGHT_W-1:0]   weight0_i,
    input  logic signed [WEIGHT_W-1:0]   weight1_i,
    input  logic signed [WEIGHT_W-1:0]   weight2_i,
    input  logic signed [WEIGHT_W-1:0]   weight3_i,
    output logic signed [SUM_W-1:0]      sum_o,
    output logic                         valid_o
);
    localparam int CHAIN_LEN         = 4;
    localparam int FIRST_DSP_LATENCY = 2;
    localparam int CHAIN_LATENCY     = FIRST_DSP_LATENCY + CHAIN_LEN - 1;

    localparam logic [8:0] OPMODE_M          = 9'b000000101;
    localparam logic [8:0] OPMODE_PCIN_PLUS_M = 9'b000010101;

    logic signed [ACT_W-1:0]    act1_q;
    logic signed [ACT_W-1:0]    act2_q [0:1];
    logic signed [ACT_W-1:0]    act3_q [0:2];
    logic signed [WEIGHT_W-1:0] weight1_q;
    logic signed [WEIGHT_W-1:0] weight2_q [0:1];
    logic signed [WEIGHT_W-1:0] weight3_q [0:2];
    logic [CHAIN_LATENCY:0]     valid_pipe_q;

    logic [47:0] p0_w;
    logic [47:0] p1_w;
    logic [47:0] p2_w;
    logic [47:0] p3_w;
    logic [47:0] pc0_w;
    logic [47:0] pc1_w;
    logic [47:0] pc2_w;

    function automatic logic [29:0] dsp_a(input logic signed [ACT_W-1:0] value);
        begin
            dsp_a = {{(30-ACT_W){value[ACT_W-1]}}, value};
        end
    endfunction

    function automatic logic [17:0] dsp_b(input logic signed [WEIGHT_W-1:0] value);
        begin
            dsp_b = {{(18-WEIGHT_W){value[WEIGHT_W-1]}}, value};
        end
    endfunction

    assign sum_o = p3_w;
    assign valid_o = valid_pipe_q[CHAIN_LATENCY];

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            act1_q <= '0;
            weight1_q <= '0;
            for (int i = 0; i < 2; i++) begin
                act2_q[i] <= '0;
                weight2_q[i] <= '0;
            end
            for (int i = 0; i < 3; i++) begin
                act3_q[i] <= '0;
                weight3_q[i] <= '0;
            end
            valid_pipe_q <= '0;
        end else begin
            act1_q <= valid_i ? act1_i : '0;
            weight1_q <= valid_i ? weight1_i : '0;

            act2_q[0] <= valid_i ? act2_i : '0;
            weight2_q[0] <= valid_i ? weight2_i : '0;
            act2_q[1] <= act2_q[0];
            weight2_q[1] <= weight2_q[0];

            act3_q[0] <= valid_i ? act3_i : '0;
            weight3_q[0] <= valid_i ? weight3_i : '0;
            act3_q[1] <= act3_q[0];
            weight3_q[1] <= weight3_q[0];
            act3_q[2] <= act3_q[1];
            weight3_q[2] <= weight3_q[1];

            valid_pipe_q[0] <= valid_i;
            for (int i = 1; i <= CHAIN_LATENCY; i++) begin
                valid_pipe_q[i] <= valid_pipe_q[i-1];
            end
        end
    end

    DSP48E2 #(
        .ACASCREG      (1),
        .ADREG         (0),
        .ALUMODEREG    (0),
        .AREG          (1),
        .BCASCREG      (1),
        .BREG          (1),
        .CARRYINREG    (0),
        .CARRYINSELREG (0),
        .CREG          (0),
        .DREG          (0),
        .INMODEREG     (0),
        .MREG          (1),
        .OPMODEREG     (0),
        .PREG          (1),
        .USE_MULT      ("MULTIPLY"),
        .USE_SIMD      ("ONE48")
    ) u_mac0 (
        .A            (dsp_a(valid_i ? act0_i : '0)),
        .ACIN         ('0),
        .ALUMODE      (4'b0000),
        .B            (dsp_b(valid_i ? weight0_i : '0)),
        .BCIN         ('0),
        .C            ('0),
        .CARRYCASCIN  (1'b0),
        .CARRYIN      (1'b0),
        .CARRYINSEL   (3'b000),
        .CEA1         (1'b1),
        .CEA2         (1'b1),
        .CEAD         (1'b1),
        .CEALUMODE    (1'b1),
        .CEB1         (1'b1),
        .CEB2         (1'b1),
        .CEC          (1'b1),
        .CECARRYIN    (1'b1),
        .CECTRL       (1'b1),
        .CED          (1'b1),
        .CEINMODE     (1'b1),
        .CEM          (1'b1),
        .CEP          (1'b1),
        .CLK          (clk_i),
        .D            ('0),
        .INMODE       (5'b00000),
        .MULTSIGNIN   (1'b0),
        .OPMODE       (OPMODE_M),
        .PCIN         ('0),
        .RSTA         (!rst_ni),
        .RSTALLCARRYIN(!rst_ni),
        .RSTALUMODE   (!rst_ni),
        .RSTB         (!rst_ni),
        .RSTC         (!rst_ni),
        .RSTCTRL      (!rst_ni),
        .RSTD         (!rst_ni),
        .RSTINMODE    (!rst_ni),
        .RSTM         (!rst_ni),
        .RSTP         (!rst_ni),
        .ACOUT        (),
        .BCOUT        (),
        .CARRYCASCOUT (),
        .CARRYOUT     (),
        .MULTSIGNOUT  (),
        .OVERFLOW     (),
        .P            (p0_w),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT        (pc0_w),
        .UNDERFLOW    (),
        .XOROUT       ()
    );

    DSP48E2 #(
        .ACASCREG      (1),
        .ADREG         (0),
        .ALUMODEREG    (0),
        .AREG          (1),
        .BCASCREG      (1),
        .BREG          (1),
        .CARRYINREG    (0),
        .CARRYINSELREG (0),
        .CREG          (0),
        .DREG          (0),
        .INMODEREG     (0),
        .MREG          (1),
        .OPMODEREG     (0),
        .PREG          (1),
        .USE_MULT      ("MULTIPLY"),
        .USE_SIMD      ("ONE48")
    ) u_mac1 (
        .A            (dsp_a(act1_q)),
        .ACIN         ('0),
        .ALUMODE      (4'b0000),
        .B            (dsp_b(weight1_q)),
        .BCIN         ('0),
        .C            ('0),
        .CARRYCASCIN  (1'b0),
        .CARRYIN      (1'b0),
        .CARRYINSEL   (3'b000),
        .CEA1         (1'b1),
        .CEA2         (1'b1),
        .CEAD         (1'b1),
        .CEALUMODE    (1'b1),
        .CEB1         (1'b1),
        .CEB2         (1'b1),
        .CEC          (1'b1),
        .CECARRYIN    (1'b1),
        .CECTRL       (1'b1),
        .CED          (1'b1),
        .CEINMODE     (1'b1),
        .CEM          (1'b1),
        .CEP          (1'b1),
        .CLK          (clk_i),
        .D            ('0),
        .INMODE       (5'b00000),
        .MULTSIGNIN   (1'b0),
        .OPMODE       (OPMODE_PCIN_PLUS_M),
        .PCIN         (pc0_w),
        .RSTA         (!rst_ni),
        .RSTALLCARRYIN(!rst_ni),
        .RSTALUMODE   (!rst_ni),
        .RSTB         (!rst_ni),
        .RSTC         (!rst_ni),
        .RSTCTRL      (!rst_ni),
        .RSTD         (!rst_ni),
        .RSTINMODE    (!rst_ni),
        .RSTM         (!rst_ni),
        .RSTP         (!rst_ni),
        .ACOUT        (),
        .BCOUT        (),
        .CARRYCASCOUT (),
        .CARRYOUT     (),
        .MULTSIGNOUT  (),
        .OVERFLOW     (),
        .P            (p1_w),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT        (pc1_w),
        .UNDERFLOW    (),
        .XOROUT       ()
    );

    DSP48E2 #(
        .ACASCREG      (1),
        .ADREG         (0),
        .ALUMODEREG    (0),
        .AREG          (1),
        .BCASCREG      (1),
        .BREG          (1),
        .CARRYINREG    (0),
        .CARRYINSELREG (0),
        .CREG          (0),
        .DREG          (0),
        .INMODEREG     (0),
        .MREG          (1),
        .OPMODEREG     (0),
        .PREG          (1),
        .USE_MULT      ("MULTIPLY"),
        .USE_SIMD      ("ONE48")
    ) u_mac2 (
        .A            (dsp_a(act2_q[1])),
        .ACIN         ('0),
        .ALUMODE      (4'b0000),
        .B            (dsp_b(weight2_q[1])),
        .BCIN         ('0),
        .C            ('0),
        .CARRYCASCIN  (1'b0),
        .CARRYIN      (1'b0),
        .CARRYINSEL   (3'b000),
        .CEA1         (1'b1),
        .CEA2         (1'b1),
        .CEAD         (1'b1),
        .CEALUMODE    (1'b1),
        .CEB1         (1'b1),
        .CEB2         (1'b1),
        .CEC          (1'b1),
        .CECARRYIN    (1'b1),
        .CECTRL       (1'b1),
        .CED          (1'b1),
        .CEINMODE     (1'b1),
        .CEM          (1'b1),
        .CEP          (1'b1),
        .CLK          (clk_i),
        .D            ('0),
        .INMODE       (5'b00000),
        .MULTSIGNIN   (1'b0),
        .OPMODE       (OPMODE_PCIN_PLUS_M),
        .PCIN         (pc1_w),
        .RSTA         (!rst_ni),
        .RSTALLCARRYIN(!rst_ni),
        .RSTALUMODE   (!rst_ni),
        .RSTB         (!rst_ni),
        .RSTC         (!rst_ni),
        .RSTCTRL      (!rst_ni),
        .RSTD         (!rst_ni),
        .RSTINMODE    (!rst_ni),
        .RSTM         (!rst_ni),
        .RSTP         (!rst_ni),
        .ACOUT        (),
        .BCOUT        (),
        .CARRYCASCOUT (),
        .CARRYOUT     (),
        .MULTSIGNOUT  (),
        .OVERFLOW     (),
        .P            (p2_w),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT        (pc2_w),
        .UNDERFLOW    (),
        .XOROUT       ()
    );

    DSP48E2 #(
        .ACASCREG      (1),
        .ADREG         (0),
        .ALUMODEREG    (0),
        .AREG          (1),
        .BCASCREG      (1),
        .BREG          (1),
        .CARRYINREG    (0),
        .CARRYINSELREG (0),
        .CREG          (0),
        .DREG          (0),
        .INMODEREG     (0),
        .MREG          (1),
        .OPMODEREG     (0),
        .PREG          (1),
        .USE_MULT      ("MULTIPLY"),
        .USE_SIMD      ("ONE48")
    ) u_mac3 (
        .A            (dsp_a(act3_q[2])),
        .ACIN         ('0),
        .ALUMODE      (4'b0000),
        .B            (dsp_b(weight3_q[2])),
        .BCIN         ('0),
        .C            ('0),
        .CARRYCASCIN  (1'b0),
        .CARRYIN      (1'b0),
        .CARRYINSEL   (3'b000),
        .CEA1         (1'b1),
        .CEA2         (1'b1),
        .CEAD         (1'b1),
        .CEALUMODE    (1'b1),
        .CEB1         (1'b1),
        .CEB2         (1'b1),
        .CEC          (1'b1),
        .CECARRYIN    (1'b1),
        .CECTRL       (1'b1),
        .CED          (1'b1),
        .CEINMODE     (1'b1),
        .CEM          (1'b1),
        .CEP          (1'b1),
        .CLK          (clk_i),
        .D            ('0),
        .INMODE       (5'b00000),
        .MULTSIGNIN   (1'b0),
        .OPMODE       (OPMODE_PCIN_PLUS_M),
        .PCIN         (pc2_w),
        .RSTA         (!rst_ni),
        .RSTALLCARRYIN(!rst_ni),
        .RSTALUMODE   (!rst_ni),
        .RSTB         (!rst_ni),
        .RSTC         (!rst_ni),
        .RSTCTRL      (!rst_ni),
        .RSTD         (!rst_ni),
        .RSTINMODE    (!rst_ni),
        .RSTM         (!rst_ni),
        .RSTP         (!rst_ni),
        .ACOUT        (),
        .BCOUT        (),
        .CARRYCASCOUT (),
        .CARRYOUT     (),
        .MULTSIGNOUT  (),
        .OVERFLOW     (),
        .P            (p3_w),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT        (),
        .UNDERFLOW    (),
        .XOROUT       ()
    );
endmodule

module gemv_fp16_dp128 #(
    parameter int TILE_ELEMS = 128,
    parameter int VEC_MUL_ELEMS = 32
) (
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       valid_i,
    input  logic                       act_valid_i,
    input  logic                       vec_mul_valid_i,
    input  logic [7:0]                 act_frac_cfg_i,
    input  logic                       act_uq24_packed_i,
    input  logic [1:0]                 mode_i,
    input  logic [TILE_ELEMS*16-1:0]   weight_i,
    input  logic [TILE_ELEMS*16-1:0]   act_vec_i,
    input  logic [VEC_MUL_ELEMS*16-1:0] vec_mul_a_i,
    input  logic [VEC_MUL_ELEMS*16-1:0] vec_mul_b_i,
    input  logic [15:0]                scale_fp16_i,
    input  logic [15:0]                old_acc_i,
    input  logic                       acc_mode_i,
    output logic [VEC_MUL_ELEMS*16-1:0] vec_mul_o,
    output logic [VEC_MUL_ELEMS*24-1:0] vec_mul_uq24_o,
    output logic                       vec_mul_valid_o,
    output logic [15:0]                dot_o,
    output logic                       valid_o
);
    import gemv_fp16_pkg::*;

    localparam int CHAIN_LEN          = 4;
    localparam int CHAIN_NUM          = (TILE_ELEMS + CHAIN_LEN - 1) / CHAIN_LEN;
    localparam int REDUCE_RADIX       = 6;
    localparam int REDUCE_L1_NUM      = (CHAIN_NUM + REDUCE_RADIX - 1) / REDUCE_RADIX;
    localparam int REDUCE_L2_NUM      = (REDUCE_L1_NUM + REDUCE_RADIX - 1) / REDUCE_RADIX;
    localparam int REDUCE_LEVELS      = (REDUCE_L1_NUM <= 1) ? 1 : 2;
    localparam int DSP_REDUCE_LATENCY = REDUCE_LEVELS * 3;
    localparam int MAC_CHAIN_LATENCY  = 5;
    localparam int FP_MUL_IP_LATENCY  = 4;
    localparam int FP_MUL_CONV_LATENCY = 3;
    localparam int FP_MUL_RESULT_LATENCY = FP_MUL_IP_LATENCY + FP_MUL_CONV_LATENCY;
    localparam int INPUT_PIPE_STAGES  = 2;
    localparam int REDUCE_CAPTURE_STAGE = INPUT_PIPE_STAGES + MAC_CHAIN_LATENCY + 1 + DSP_REDUCE_LATENCY;
    localparam int TILE_ABS_STAGE     = REDUCE_CAPTURE_STAGE + 2;
    localparam int TILE_SUM_STAGE     = TILE_ABS_STAGE + 1;
    localparam int SUM_W              = 48;
    localparam int SCALE_MUL_IN_STAGE = TILE_SUM_STAGE + 1;
    localparam int FINAL_ADD_IN_STAGE = SCALE_MUL_IN_STAGE + FP_MUL_RESULT_LATENCY;
    localparam int OUTPUT_LATENCY     = FINAL_ADD_IN_STAGE + 3;
    localparam int VEC_MUL_TAIL_STAGE = 2;
    localparam int DOT_ABS_STAGE      = TILE_ABS_STAGE - REDUCE_CAPTURE_STAGE;
    localparam int W8_TILE_ELEMS      = TILE_ELEMS / 2;
    localparam int ACT_FRAC_BITS      = 16;
    localparam int ACT_FRAC_UQ24_BITS = 24;
    localparam logic [7:0] ACT_FRAC_UQ24_CFG = 8'h98;
    localparam logic [4:0] ACT_FRAC_BITS_W = 5'(ACT_FRAC_BITS);
    localparam logic [4:0] ACT_FRAC_BIAS_W = 5'd25 - ACT_FRAC_BITS_W;

    logic [OUTPUT_LATENCY:0] valid_pipe_q;
    logic [INPUT_PIPE_STAGES-1:0] compute_valid_pipe_q;
    logic [1:0]              mode_pipe_q [0:INPUT_PIPE_STAGES-1];
    logic [TILE_ELEMS*16-1:0] weight_pipe_q [0:INPUT_PIPE_STAGES-1];
    logic [VEC_MUL_TAIL_STAGE:0] vec_mul_valid_pipe_q;
    logic [VEC_MUL_ELEMS*16-1:0] vec_mul_pipe_q [0:VEC_MUL_TAIL_STAGE];
    logic [VEC_MUL_ELEMS*16-1:0] vec_mul_lane_z_w;
    logic [VEC_MUL_ELEMS*24-1:0] vec_mul_uq24_pipe_q [0:VEC_MUL_TAIL_STAGE];
    logic [VEC_MUL_ELEMS*24-1:0] vec_mul_lane_uq24_w;
    logic [VEC_MUL_ELEMS-1:0]    vec_mul_lane_valid_w;
    logic [15:0]             scale_pipe_q [0:SCALE_MUL_IN_STAGE];
    logic [15:0]             old_acc_pipe_q [0:FINAL_ADD_IN_STAGE];
    logic                    acc_mode_pipe_q [0:FINAL_ADD_IN_STAGE];
    logic [TILE_ELEMS*16-1:0] act_load_vec_q;
    logic                    act_uq24_mode_q;
    logic                    act_uq24_packed_q;
    logic [23:0]             qact_q [0:TILE_ELEMS-1];
    logic signed [SUM_W-1:0] dot_sum_w;
    logic signed [SUM_W-1:0] dot_sum_pipe_q [0:DOT_ABS_STAGE];
    logic signed [26:0]      mac_act_w [0:CHAIN_NUM-1][0:CHAIN_LEN-1];
    logic signed [17:0]      mac_weight_w [0:CHAIN_NUM-1][0:CHAIN_LEN-1];
    logic signed [SUM_W-1:0] mac_sum_w [0:CHAIN_NUM-1];
    logic [CHAIN_NUM-1:0]    mac_valid_w;
    logic signed [SUM_W-1:0] reduce_l1_in_w [0:REDUCE_L1_NUM*REDUCE_RADIX-1];
    logic signed [SUM_W-1:0] reduce_l1_sum_w [0:REDUCE_L1_NUM-1];
    logic signed [SUM_W-1:0] reduce_l2_in_w [0:REDUCE_L2_NUM*REDUCE_RADIX-1];
    logic signed [SUM_W-1:0] reduce_l2_sum_w [0:REDUCE_L2_NUM-1];
    logic signed [SUM_W-1:0] reduce_final_w;
    logic                        tile_sign_q;
    logic [SUM_W-1:0]            tile_abs_sum_q;
    logic [15:0]                 tile_sum_q;
    logic [15:0]                 scaled_tile_sum_w;
    logic                        scale_mul_valid_w;
    logic [15:0] dot_q;
    logic        final_add_valid_w;

    assign dot_o   = dot_q;
    assign valid_o = final_add_valid_w;
    assign vec_mul_valid_o = vec_mul_valid_pipe_q[VEC_MUL_TAIL_STAGE];
    assign vec_mul_o = vec_mul_pipe_q[VEC_MUL_TAIL_STAGE];
    assign vec_mul_uq24_o = vec_mul_uq24_pipe_q[VEC_MUL_TAIL_STAGE];

    function automatic logic [23:0] round_mant_right_to_u24(
        input logic [10:0] mant,
        input logic [4:0]  shift
    );
        logic [10:0] shifted;
        logic round_bit;
        logic sticky_bit;
        logic lsb;
        begin
            if (shift == '0) begin
                round_mant_right_to_u24 = {13'd0, mant};
            end else begin
                shifted = mant >> shift;
                round_mant_right_to_u24 = {13'd0, shifted};
                round_bit = mant[shift - 5'd1];
                sticky_bit = 1'b0;
                for (int i = 0; i < 11; i++) begin
                    if (i < (shift - 1))
                        sticky_bit |= mant[i];
                end
                lsb = shifted[0];
                if (round_bit && (sticky_bit || lsb))
                    round_mant_right_to_u24 = round_mant_right_to_u24 + 24'd1;
            end
        end
    endfunction

    function automatic logic signed [23:0] fp16_to_s24_fixed(
        input logic [15:0] fp16
    );
        logic [10:0] mant;
        logic [4:0]  exp;
        logic [4:0]  lshift;
        logic [4:0]  rshift;
        logic [23:0] mag;
        logic        sat_mag;
        begin
            fp16_to_s24_fixed = '0;
            if (fp16_is_normal(fp16)) begin
                mant = {1'b1, fp16[9:0]};
                exp = fp16[14:10];
                sat_mag = 1'b0;

                if (exp < ACT_FRAC_BIAS_W) begin
                    rshift = ACT_FRAC_BIAS_W - exp;
                    mag = round_mant_right_to_u24(mant, rshift);
                end else begin
                    lshift = exp - ACT_FRAC_BIAS_W;
                    if (lshift >= 5'd13) begin
                        mag = '0;
                        sat_mag = 1'b1;
                    end else begin
                        mag = {13'd0, mant} << lshift;
                    end
                end

                if (fp16[15]) begin
                    if (sat_mag || (mag >= 24'h800000))
                        fp16_to_s24_fixed = 24'sh800000;
                    else
                        fp16_to_s24_fixed = -$signed({1'b0, mag});
                end else begin
                    if (sat_mag || (mag >= 24'h7fffff))
                        fp16_to_s24_fixed = 24'sh7fffff;
                    else
                        fp16_to_s24_fixed = $signed(mag);
                end
            end
        end
    endfunction

    function automatic logic [23:0] fp16_to_uq0p24_fixed(
        input logic [15:0] fp16
    );
        logic [63:0] mag;
        int shift;
        begin
            fp16_to_uq0p24_fixed = '0;
            if (!fp16[15]) begin
                if (fp16[14:10] == 5'd0) begin
                    fp16_to_uq0p24_fixed = {14'd0, fp16[9:0]};
                end else if (fp16[14:10] >= 5'd15) begin
                    fp16_to_uq0p24_fixed = 24'hff_ffff;
                end else if (fp16[14:10] != 5'h1f) begin
                    mag = {53'd0, 1'b1, fp16[9:0]};
                    shift = int'(fp16[14:10]) - 1;
                    mag = mag << shift;
                    fp16_to_uq0p24_fixed =
                        (mag >= 64'h00ff_ffff) ? 24'hff_ffff : mag[23:0];
                end
            end
        end
    endfunction

    function automatic logic [5:0] leading_one_reduce(input logic [SUM_W-1:0] value);
        logic found;
        begin
            leading_one_reduce = '0;
            found = 1'b0;
            for (int i = SUM_W-1; i >= 0; i--) begin
                if (!found && value[i]) begin
                    leading_one_reduce = i[5:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [SUM_W-1:0] rshift_sticky_reduce(
        input logic [SUM_W-1:0] value,
        input int               shift
    );
        logic sticky;
        begin
            sticky = 1'b0;
            if (shift <= 0) begin
                rshift_sticky_reduce = value;
            end else if (shift >= SUM_W) begin
                rshift_sticky_reduce = '0;
                rshift_sticky_reduce[0] = |value;
            end else begin
                rshift_sticky_reduce = value >> shift;
                for (int i = 0; i < SUM_W; i++) begin
                    if (i < shift)
                        sticky |= value[i];
                end
                if (sticky)
                    rshift_sticky_reduce[0] = 1'b1;
            end
        end
    endfunction

    function automatic logic [15:0] fp16_from_aligned_sum(
        input logic              sign,
        input logic [SUM_W-1:0]  abs_sum,
        input int                frac_bits
    );
        logic [5:0]         lead;
        logic [SUM_W-1:0]   norm_sum;
        logic [10:0]        mant_pre;
        logic [11:0]        mant_round;
        logic               round_bit;
        logic               sticky_bit;
        int                 shift_amt;
        int                 exp_norm;
        begin
            fp16_from_aligned_sum = 16'h0000;
            lead = leading_one_reduce(abs_sum);
            norm_sum = '0;
            mant_pre = '0;
            mant_round = '0;
            round_bit = 1'b0;
            sticky_bit = 1'b0;
            shift_amt = 0;
            exp_norm = 15 + int'(lead) - frac_bits;

            if (abs_sum != '0) begin
                if (int'(lead) > FP16_TARGET_BIT) begin
                    shift_amt = int'(lead) - FP16_TARGET_BIT;
                    norm_sum = rshift_sticky_reduce(abs_sum, shift_amt);
                end else begin
                    shift_amt = FP16_TARGET_BIT - int'(lead);
                    norm_sum = abs_sum << shift_amt;
                end

                if (exp_norm <= 0) begin
                    fp16_from_aligned_sum = 16'h0000;
                end else if (exp_norm >= 31) begin
                    fp16_from_aligned_sum = {sign, 5'h1f, 10'd0};
                end else begin
                    mant_pre = norm_sum[FP16_TARGET_BIT -: 11];
                    round_bit = norm_sum[FP16_GUARD_BITS-1];
                    sticky_bit = |norm_sum[FP16_GUARD_BITS-2:0];
                    mant_round = {1'b0, mant_pre} +
                                 ((round_bit && (sticky_bit || mant_pre[0])) ? 12'd1 : 12'd0);
                    if (mant_round[11]) begin
                        exp_norm++;
                        if (exp_norm >= 31)
                            fp16_from_aligned_sum = {sign, 5'h1f, 10'd0};
                        else
                            fp16_from_aligned_sum = {sign, exp_norm[4:0], 10'd0};
                    end else begin
                        fp16_from_aligned_sum = {sign, exp_norm[4:0], mant_round[9:0]};
                    end
                end
            end
        end
    endfunction

    always_comb begin
        for (int chain = 0; chain < CHAIN_NUM; chain++) begin
            for (int lane_in_chain = 0; lane_in_chain < CHAIN_LEN; lane_in_chain++) begin
                int lane;
                logic signed [7:0] weight_s8;
                logic signed [3:0] weight_s4;
                logic lane_active;
                lane = chain * CHAIN_LEN + lane_in_chain;
                lane_active = mode_pipe_q[INPUT_PIPE_STAGES-1][0] ?
                              (lane < W8_TILE_ELEMS) : (lane < TILE_ELEMS);
                mac_act_w[chain][lane_in_chain] = '0;
                mac_weight_w[chain][lane_in_chain] = '0;
                if (lane_active) begin
                    mac_act_w[chain][lane_in_chain] =
                        act_uq24_mode_q ? {3'd0, qact_q[lane]} :
                        {{3{qact_q[lane][23]}}, qact_q[lane]};
                    if (mode_pipe_q[INPUT_PIPE_STAGES-1][0]) begin
                        weight_s8 = weight_pipe_q[INPUT_PIPE_STAGES-1][lane*8 +: 8];
                        mac_weight_w[chain][lane_in_chain] =
                            {{10{weight_s8[7]}}, weight_s8};
                    end else begin
                        weight_s4 = weight_pipe_q[INPUT_PIPE_STAGES-1][lane*4 +: 4];
                        mac_weight_w[chain][lane_in_chain] =
                            {{14{weight_s4[3]}}, weight_s4};
                    end
                end
            end
        end
        for (int i = 0; i < REDUCE_L1_NUM * REDUCE_RADIX; i++) begin
            if (i < CHAIN_NUM)
                reduce_l1_in_w[i] = mac_sum_w[i];
            else
                reduce_l1_in_w[i] = '0;
        end
        for (int i = 0; i < REDUCE_L2_NUM * REDUCE_RADIX; i++) begin
            if (i < REDUCE_L1_NUM)
                reduce_l2_in_w[i] = reduce_l1_sum_w[i];
            else
                reduce_l2_in_w[i] = '0;
        end
        reduce_final_w = reduce_l2_sum_w[0];
        dot_sum_w = reduce_final_w;
    end

    generate
        for (genvar lane = 0; lane < VEC_MUL_ELEMS; lane++) begin : gen_vec_mul
            gemv_fp16_mul_pipe2 #(
                .SUPPORT_INPUT_SUBNORMAL (1'b0)
            ) u_vec_mul (
                .clk_i    (clk_i),
                .rst_ni   (rst_ni),
                .valid_i  (vec_mul_valid_i),
                .mul_en_i (1'b1),
                .preserve_subnormal_i (1'b0),
                .a_i      (vec_mul_a_i[lane*16 +: 16]),
                .b_i      (vec_mul_b_i[lane*16 +: 16]),
                .z_o      (vec_mul_lane_z_w[lane*16 +: 16]),
                .uq0p24_o (vec_mul_lane_uq24_w[lane*24 +: 24]),
                .valid_o  (vec_mul_lane_valid_w[lane])
            );
        end
        for (genvar chain = 0; chain < CHAIN_NUM; chain++) begin : gen_mac_chain
            gemv_dsp48e2_mac4_chain u_mac4 (
                .clk_i     (clk_i),
                .rst_ni    (rst_ni),
                .valid_i   (compute_valid_pipe_q[INPUT_PIPE_STAGES-1]),
                .act0_i    (mac_act_w[chain][0]),
                .act1_i    (mac_act_w[chain][1]),
                .act2_i    (mac_act_w[chain][2]),
                .act3_i    (mac_act_w[chain][3]),
                .weight0_i (mac_weight_w[chain][0]),
                .weight1_i (mac_weight_w[chain][1]),
                .weight2_i (mac_weight_w[chain][2]),
                .weight3_i (mac_weight_w[chain][3]),
                .sum_o     (mac_sum_w[chain]),
                .valid_o   (mac_valid_w[chain])
            );
        end
        for (genvar grp = 0; grp < REDUCE_L1_NUM; grp++) begin : gen_reduce_l1
            gemv_dsp48e2_add6_chain3 u_add6 (
                .clk_i  (clk_i),
                .rst_ni (rst_ni),
                .in0_i  (reduce_l1_in_w[grp*REDUCE_RADIX + 0]),
                .in1_i  (reduce_l1_in_w[grp*REDUCE_RADIX + 1]),
                .in2_i  (reduce_l1_in_w[grp*REDUCE_RADIX + 2]),
                .in3_i  (reduce_l1_in_w[grp*REDUCE_RADIX + 3]),
                .in4_i  (reduce_l1_in_w[grp*REDUCE_RADIX + 4]),
                .in5_i  (reduce_l1_in_w[grp*REDUCE_RADIX + 5]),
                .sum_o  (reduce_l1_sum_w[grp])
            );
        end
        for (genvar grp = 0; grp < REDUCE_L2_NUM; grp++) begin : gen_reduce_l2
            gemv_dsp48e2_add6_chain3 u_add6 (
                .clk_i  (clk_i),
                .rst_ni (rst_ni),
                .in0_i  (reduce_l2_in_w[grp*REDUCE_RADIX + 0]),
                .in1_i  (reduce_l2_in_w[grp*REDUCE_RADIX + 1]),
                .in2_i  (reduce_l2_in_w[grp*REDUCE_RADIX + 2]),
                .in3_i  (reduce_l2_in_w[grp*REDUCE_RADIX + 3]),
                .in4_i  (reduce_l2_in_w[grp*REDUCE_RADIX + 4]),
                .in5_i  (reduce_l2_in_w[grp*REDUCE_RADIX + 5]),
                .sum_o  (reduce_l2_sum_w[grp])
            );
        end
    endgenerate

    gemv_fp16_mul_pipe2 #(
        .SUPPORT_INPUT_SUBNORMAL (1'b0),
        .ENABLE_FP16_OUT (1'b1),
        .ENABLE_UQ0P24_OUT (1'b0)
    ) u_scale_mul (
        .clk_i    (clk_i),
        .rst_ni   (rst_ni),
        .valid_i  (valid_pipe_q[SCALE_MUL_IN_STAGE]),
        .mul_en_i (1'b1),
        .preserve_subnormal_i (1'b0),
        .a_i      (tile_sum_q),
        .b_i      (scale_pipe_q[SCALE_MUL_IN_STAGE]),
        .z_o      (scaled_tile_sum_w),
        .uq0p24_o (),
        .valid_o  (scale_mul_valid_w)
    );

    gemv_fp16_add_pipe3 u_final_add (
        .clk_i    (clk_i),
        .rst_ni   (rst_ni),
        .valid_i  (scale_mul_valid_w),
        .add_en_i (acc_mode_pipe_q[FINAL_ADD_IN_STAGE]),
        .a_i      (scaled_tile_sum_w),
        .b_i      (old_acc_pipe_q[FINAL_ADD_IN_STAGE]),
        .z_o      (dot_q),
        .valid_o  (final_add_valid_w)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_pipe_q <= '0;
            compute_valid_pipe_q <= '0;
            vec_mul_valid_pipe_q <= '0;
            for (int i = 0; i <= VEC_MUL_TAIL_STAGE; i++) begin
                vec_mul_pipe_q[i] <= '0;
                vec_mul_uq24_pipe_q[i] <= '0;
            end
            for (int i = 0; i < INPUT_PIPE_STAGES; i++) begin
                mode_pipe_q[i] <= '0;
            end
            for (int i = 0; i <= FINAL_ADD_IN_STAGE; i++) begin
                acc_mode_pipe_q[i] <= 1'b0;
            end
        end else begin
            valid_pipe_q[0] <= valid_i;
            for (int i = 1; i <= OUTPUT_LATENCY; i++) begin
                valid_pipe_q[i] <= valid_pipe_q[i-1];
            end
            compute_valid_pipe_q[0] <= valid_i;
            for (int i = 1; i < INPUT_PIPE_STAGES; i++) begin
                compute_valid_pipe_q[i] <= compute_valid_pipe_q[i-1];
            end
            mode_pipe_q[0] <= mode_i;
            for (int i = 1; i < INPUT_PIPE_STAGES; i++) begin
                mode_pipe_q[i] <= mode_pipe_q[i-1];
            end
            vec_mul_valid_pipe_q[0] <= vec_mul_lane_valid_w[0];
            for (int i = 1; i <= VEC_MUL_TAIL_STAGE; i++) begin
                vec_mul_valid_pipe_q[i] <= vec_mul_valid_pipe_q[i-1];
            end

            vec_mul_pipe_q[0] <= vec_mul_lane_valid_w[0] ? vec_mul_lane_z_w : '0;
            vec_mul_uq24_pipe_q[0] <= vec_mul_lane_valid_w[0] ? vec_mul_lane_uq24_w : '0;
            for (int i = 1; i <= VEC_MUL_TAIL_STAGE; i++) begin
                vec_mul_pipe_q[i] <= vec_mul_pipe_q[i-1];
                vec_mul_uq24_pipe_q[i] <= vec_mul_uq24_pipe_q[i-1];
            end

            acc_mode_pipe_q[0] <= acc_mode_i;
            for (int i = 1; i <= FINAL_ADD_IN_STAGE; i++) begin
                acc_mode_pipe_q[i] <= acc_mode_pipe_q[i-1];
            end
        end
    end

    always_ff @(posedge clk_i) begin
        weight_pipe_q[0] <= weight_i;
        for (int i = 1; i < INPUT_PIPE_STAGES; i++) begin
            weight_pipe_q[i] <= weight_pipe_q[i-1];
        end

        scale_pipe_q[0] <= scale_fp16_i;
        for (int i = 1; i <= SCALE_MUL_IN_STAGE; i++) begin
            scale_pipe_q[i] <= scale_pipe_q[i-1];
        end
        old_acc_pipe_q[0] <= old_acc_i;
        for (int i = 1; i <= FINAL_ADD_IN_STAGE; i++) begin
            old_acc_pipe_q[i] <= old_acc_pipe_q[i-1];
        end

        if (act_valid_i) begin
            act_load_vec_q <= act_vec_i;
        end
        if (act_uq24_mode_q) begin
            for (int lane = 0; lane < W8_TILE_ELEMS; lane++) begin
                qact_q[lane] <= act_uq24_packed_q ?
                                act_load_vec_q[lane*24 +: 24] :
                                fp16_to_uq0p24_fixed(act_load_vec_q[lane*16 +: 16]);
            end
            for (int lane = W8_TILE_ELEMS; lane < TILE_ELEMS; lane++) begin
                qact_q[lane] <= '0;
            end
        end else begin
            for (int lane = 0; lane < TILE_ELEMS; lane++) begin
                qact_q[lane] <= fp16_to_s24_fixed(act_load_vec_q[lane*16 +: 16]);
            end
        end
        dot_sum_pipe_q[0] <= dot_sum_w;
        for (int stage = 1; stage <= DOT_ABS_STAGE; stage++) begin
            dot_sum_pipe_q[stage] <= dot_sum_pipe_q[stage-1];
        end

        tile_sign_q <= dot_sum_pipe_q[DOT_ABS_STAGE][SUM_W-1];
        tile_abs_sum_q <= dot_sum_pipe_q[DOT_ABS_STAGE][SUM_W-1] ?
                          (~dot_sum_pipe_q[DOT_ABS_STAGE][SUM_W-1:0] + 1'b1) :
                          dot_sum_pipe_q[DOT_ABS_STAGE][SUM_W-1:0];
        tile_sum_q <= fp16_from_aligned_sum(
            tile_sign_q, tile_abs_sum_q,
            act_uq24_mode_q ? ACT_FRAC_UQ24_BITS : ACT_FRAC_BITS);
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            act_uq24_mode_q <= 1'b0;
            act_uq24_packed_q <= 1'b0;
        end else if (act_valid_i) begin
            act_uq24_mode_q <= (act_frac_cfg_i == ACT_FRAC_UQ24_CFG) &&
                               (mode_i == 2'b01);
            act_uq24_packed_q <= act_uq24_packed_i;
        end
    end

`ifndef SYNTHESIS
    property p_output_valid_latency;
        @(posedge clk_i) disable iff (!rst_ni) valid_i |=> ##OUTPUT_LATENCY valid_o;
    endproperty
    assert property (p_output_valid_latency);

    property p_no_dual_issue;
        @(posedge clk_i) disable iff (!rst_ni) !(valid_i && vec_mul_valid_i);
    endproperty
    assert property (p_no_dual_issue);

    property p_no_act_convert_during_dot;
        @(posedge clk_i) disable iff (!rst_ni) !(valid_i && act_valid_i);
    endproperty
    assert property (p_no_act_convert_during_dot);
`endif

endmodule

module gemv_w4a16_dp128 #(
    parameter int TILE_ELEMS = 128,
    parameter int FIXED_FRAC = 12
) (
    input  logic                     clk_i,
    input  logic                     rst_ni,
    input  logic                     act_valid_i,
    input  logic                     valid_i,
    input  logic [TILE_ELEMS*4-1:0]  weight_i,
    input  logic [TILE_ELEMS*16-1:0] act_vec_i,
    input  logic [15:0]              scale_fp16_i,
    input  logic [31:0]              old_acc_i,
    input  logic                     acc_mode_i,
    output logic [31:0]              dot_o,
    output logic                     valid_o
);
    logic [TILE_ELEMS*16-1:0] weight_ext_w;
    logic [15:0] dot_fp16_w;

    always_comb begin
        weight_ext_w = '0;
        weight_ext_w[TILE_ELEMS*4-1:0] = weight_i;
    end

    gemv_fp16_dp128 #(
        .TILE_ELEMS     (TILE_ELEMS),
        .VEC_MUL_ELEMS  (32)
    ) u_fp16_dp128 (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (valid_i),
        .act_valid_i  (act_valid_i),
        .vec_mul_valid_i (1'b0),
        .act_frac_cfg_i (8'd0),
        .act_uq24_packed_i (1'b0),
        .mode_i       (2'b00),
        .weight_i     (weight_ext_w),
        .act_vec_i    (act_vec_i),
        .vec_mul_a_i  ('0),
        .vec_mul_b_i  ('0),
        .scale_fp16_i (scale_fp16_i),
        .old_acc_i    (old_acc_i[15:0]),
        .acc_mode_i   (acc_mode_i),
        .vec_mul_o    (),
        .vec_mul_uq24_o (),
        .vec_mul_valid_o (),
        .dot_o        (dot_fp16_w),
        .valid_o      (valid_o)
    );

    assign dot_o = {16'h0000, dot_fp16_w};

    logic [31:0] unused_fixed_frac;
    assign unused_fixed_frac = FIXED_FRAC;
endmodule

`endif
