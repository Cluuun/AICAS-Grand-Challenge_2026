`ifndef GEMV_FP16_MUL_IP_PIPE_SV
`define GEMV_FP16_MUL_IP_PIPE_SV

module gemv_fp16_mul_ip_pipe #(
    parameter int IP_LATENCY = 4,
    parameter int CONV_LATENCY = 3,
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
    logic [31:0] a_fp32_w;
    logic [31:0] b_fp32_w;
    logic [31:0] product_fp32_w;
    logic        product_valid_w;
    logic        a_ready_unused;
    logic        b_ready_unused;
    logic        fp16_conv_valid_w;
    logic [15:0] fp16_conv_w;
    logic [15:0] fp16_conv_norm_w;
    logic        uq0p24_conv_valid_w;
    logic [31:0] uq0p24_conv_w;

    localparam int RESULT_LATENCY = IP_LATENCY + CONV_LATENCY;
    logic [RESULT_LATENCY:0] valid_pipe_q;
    logic [RESULT_LATENCY:0] mul_en_pipe_q;
    logic [RESULT_LATENCY:0] preserve_pipe_q;
    logic [15:0]             pass_pipe_q [0:RESULT_LATENCY];

    function automatic logic [31:0] fp16_to_fp32_bits(
        input logic [15:0] fp16,
        input logic        support_subnormal
    );
        logic       sign;
        logic [4:0] exp;
        logic [9:0] frac;
        logic [7:0] exp32;
        logic [9:0] frac_shift;
        int         lead;
        begin
            sign = fp16[15];
            exp = fp16[14:10];
            frac = fp16[9:0];
            fp16_to_fp32_bits = {sign, 31'd0};

            if (exp == 5'h1f) begin
                fp16_to_fp32_bits = {sign, 8'hff,
                                     (frac == '0) ? 23'd0 : {1'b1, frac, 12'd0}};
            end else if (exp != 5'd0) begin
                exp32 = {3'd0, exp} + 8'd112;
                fp16_to_fp32_bits = {sign, exp32, frac, 13'd0};
            end else if (support_subnormal && (frac != 10'd0)) begin
                lead = 0;
                for (int i = 0; i < 10; i++) begin
                    if (frac[i])
                        lead = i;
                end
                frac_shift = frac << (10 - lead);
                fp16_to_fp32_bits = {sign, 8'(lead + 103), frac_shift[9:0], 13'd0};
            end
        end
    endfunction

    function automatic logic [23:0] rshift_round_even_u24(
        input logic [23:0] value,
        input logic [4:0]  shift
    );
        logic [23:0] shifted;
        logic round_bit;
        logic sticky_bit;
        logic lsb;
        begin
            rshift_round_even_u24 = '0;
            shifted = '0;
            round_bit = 1'b0;
            sticky_bit = 1'b0;

            case (shift)
                5'd0:  begin shifted = value;       round_bit = 1'b0;      sticky_bit = 1'b0; end
                5'd1:  begin shifted = value >> 1;  round_bit = value[0];  sticky_bit = 1'b0; end
                5'd2:  begin shifted = value >> 2;  round_bit = value[1];  sticky_bit = |value[0:0]; end
                5'd3:  begin shifted = value >> 3;  round_bit = value[2];  sticky_bit = |value[1:0]; end
                5'd4:  begin shifted = value >> 4;  round_bit = value[3];  sticky_bit = |value[2:0]; end
                5'd5:  begin shifted = value >> 5;  round_bit = value[4];  sticky_bit = |value[3:0]; end
                5'd6:  begin shifted = value >> 6;  round_bit = value[5];  sticky_bit = |value[4:0]; end
                5'd7:  begin shifted = value >> 7;  round_bit = value[6];  sticky_bit = |value[5:0]; end
                5'd8:  begin shifted = value >> 8;  round_bit = value[7];  sticky_bit = |value[6:0]; end
                5'd9:  begin shifted = value >> 9;  round_bit = value[8];  sticky_bit = |value[7:0]; end
                5'd10: begin shifted = value >> 10; round_bit = value[9];  sticky_bit = |value[8:0]; end
                5'd11: begin shifted = value >> 11; round_bit = value[10]; sticky_bit = |value[9:0]; end
                5'd12: begin shifted = value >> 12; round_bit = value[11]; sticky_bit = |value[10:0]; end
                5'd13: begin shifted = value >> 13; round_bit = value[12]; sticky_bit = |value[11:0]; end
                5'd14: begin shifted = value >> 14; round_bit = value[13]; sticky_bit = |value[12:0]; end
                5'd15: begin shifted = value >> 15; round_bit = value[14]; sticky_bit = |value[13:0]; end
                5'd16: begin shifted = value >> 16; round_bit = value[15]; sticky_bit = |value[14:0]; end
                5'd17: begin shifted = value >> 17; round_bit = value[16]; sticky_bit = |value[15:0]; end
                5'd18: begin shifted = value >> 18; round_bit = value[17]; sticky_bit = |value[16:0]; end
                5'd19: begin shifted = value >> 19; round_bit = value[18]; sticky_bit = |value[17:0]; end
                5'd20: begin shifted = value >> 20; round_bit = value[19]; sticky_bit = |value[18:0]; end
                5'd21: begin shifted = value >> 21; round_bit = value[20]; sticky_bit = |value[19:0]; end
                5'd22: begin shifted = value >> 22; round_bit = value[21]; sticky_bit = |value[20:0]; end
                5'd23: begin shifted = value >> 23; round_bit = value[22]; sticky_bit = |value[21:0]; end
                default: begin shifted = '0;       round_bit = 1'b0;      sticky_bit = 1'b0; end
            endcase

            if (shift < 5'd24) begin
                rshift_round_even_u24 = shifted;
                lsb = shifted[0];
                if (round_bit && (sticky_bit || lsb))
                    rshift_round_even_u24++;
            end
        end
    endfunction

    function automatic logic [23:0] fp32_to_uq0p24_fixed(
        input logic [31:0] fp32
    );
        logic [7:0]  exp;
        logic [23:0] mant;
        logic [4:0]  shift;
        begin
            exp = fp32[30:23];
            mant = {1'b1, fp32[22:0]};
            fp32_to_uq0p24_fixed = '0;

            if (!fp32[31]) begin
                if (exp == 8'hff) begin
                    fp32_to_uq0p24_fixed = 24'hff_ffff;
                end else if (exp >= 8'd127) begin
                    fp32_to_uq0p24_fixed = 24'hff_ffff;
                end else if (exp >= 8'd103) begin
                    shift = 5'(8'd126 - exp);
                    fp32_to_uq0p24_fixed = rshift_round_even_u24(mant, shift);
                end
            end
        end
    endfunction

    function automatic logic [15:0] fp32_to_fp16_bits(
        input logic [31:0] fp32,
        input logic        preserve_subnormal
    );
        logic        sign;
        logic [7:0]  exp;
        logic [22:0] frac;
        logic [4:0]  exp16;
        logic [10:0] mant_round;
        logic [24:0] mant25;
        logic [24:0] sub_shifted;
        logic        round_bit;
        logic        sticky_bit;
        logic        lsb;
        int          sub_shift;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            frac = fp32[22:0];
            exp16 = exp[4:0] - 5'd16;
            mant_round = '0;
            mant25 = {1'b0, 1'b1, frac};
            sub_shifted = '0;
            round_bit = 1'b0;
            sticky_bit = 1'b0;
            lsb = 1'b0;
            sub_shift = 0;
            fp32_to_fp16_bits = 16'h0000;

            if (exp == 8'hff) begin
                fp32_to_fp16_bits = {sign, 5'h1f,
                                     (frac == '0) ? 10'd0 : 10'h200};
            end else if (exp > 8'd142) begin
                fp32_to_fp16_bits = {sign, 5'h1f, 10'd0};
            end else if (exp >= 8'd113) begin
                mant_round = {1'b0, frac[22:13]} +
                             ((frac[12] && ((|frac[11:0]) || frac[13])) ?
                              11'd1 : 11'd0);
                if (mant_round[10]) begin
                    if (exp16 == 5'd30)
                        fp32_to_fp16_bits = {sign, 5'h1f, 10'd0};
                    else
                        fp32_to_fp16_bits = {sign, exp16 + 5'd1, 10'd0};
                end else begin
                    fp32_to_fp16_bits = {sign, exp16, mant_round[9:0]};
                end
            end else if (preserve_subnormal && (exp >= 8'd103)) begin
                sub_shift = 126 - int'(exp);
                sub_shifted = mant25 >> sub_shift;
                round_bit = mant25[sub_shift - 1];
                sticky_bit = 1'b0;
                for (int i = 0; i < 25; i++) begin
                    if (i < (sub_shift - 1))
                        sticky_bit |= mant25[i];
                end
                lsb = sub_shifted[0];
                if (round_bit && (sticky_bit || lsb))
                    sub_shifted++;

                if (sub_shifted >= 25'd1024)
                    fp32_to_fp16_bits = {sign, 5'd1, 10'd0};
                else if (sub_shifted != 25'd0)
                    fp32_to_fp16_bits = {sign, 5'd0, sub_shifted[9:0]};
            end
        end
    endfunction

    assign a_fp32_w = fp16_to_fp32_bits(a_i, SUPPORT_INPUT_SUBNORMAL);
    assign b_fp32_w = fp16_to_fp32_bits(b_i, SUPPORT_INPUT_SUBNORMAL);
    assign fp16_conv_norm_w = (fp16_conv_w[14:0] == 15'd0) ? 16'h0000 : fp16_conv_w;
    assign valid_o = ENABLE_FP16_OUT ? fp16_conv_valid_w :
                     (ENABLE_UQ0P24_OUT ? uq0p24_conv_valid_w :
                      valid_pipe_q[RESULT_LATENCY]);
    generate
        if (ENABLE_FP16_OUT) begin : gen_fp16_out
            assign z_o = mul_en_pipe_q[RESULT_LATENCY] ?
                         fp16_conv_norm_w :
                         pass_pipe_q[RESULT_LATENCY];

            fp32_to_fp16_lat3 u_fp32_to_fp16 (
                .aclk                 (clk_i),
                .s_axis_a_tvalid      (product_valid_w),
                .s_axis_a_tdata       (product_fp32_w),
                .m_axis_result_tvalid (fp16_conv_valid_w),
                .m_axis_result_tdata  (fp16_conv_w)
            );
        end else begin : gen_no_fp16_out
            assign z_o = '0;
            assign fp16_conv_valid_w = 1'b0;
            assign fp16_conv_w = '0;
        end

        if (ENABLE_UQ0P24_OUT) begin : gen_uq0p24_out
            assign uq0p24_o = uq0p24_conv_w[23:0];

            fp32_to_q1p24_lat3 u_fp32_to_q1p24 (
                .aclk                 (clk_i),
                .s_axis_a_tvalid      (product_valid_w),
                .s_axis_a_tdata       (product_fp32_w),
                .m_axis_result_tvalid (uq0p24_conv_valid_w),
                .m_axis_result_tdata  (uq0p24_conv_w)
            );
        end else begin : gen_no_uq0p24_out
            assign uq0p24_o = '0;
            assign uq0p24_conv_valid_w = 1'b0;
            assign uq0p24_conv_w = '0;
        end
    endgenerate

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_pipe_q <= '0;
            mul_en_pipe_q <= '0;
            preserve_pipe_q <= '0;
            for (int i = 0; i <= RESULT_LATENCY; i++) begin
                pass_pipe_q[i] <= 16'h0000;
            end
        end else begin
            valid_pipe_q[0] <= valid_i;
            mul_en_pipe_q[0] <= mul_en_i;
            preserve_pipe_q[0] <= preserve_subnormal_i;
            pass_pipe_q[0] <= a_i;
            for (int i = 1; i <= RESULT_LATENCY; i++) begin
                valid_pipe_q[i] <= valid_pipe_q[i-1];
                mul_en_pipe_q[i] <= mul_en_pipe_q[i-1];
                preserve_pipe_q[i] <= preserve_pipe_q[i-1];
                pass_pipe_q[i] <= pass_pipe_q[i-1];
            end
        end
    end

    fp_mul_single_lat2_rate1 u_fp32_mul (
        .aclk                 (clk_i),
        .s_axis_a_tvalid      (valid_i),
        .s_axis_a_tready      (a_ready_unused),
        .s_axis_a_tdata       (a_fp32_w),
        .s_axis_b_tvalid      (valid_i),
        .s_axis_b_tready      (b_ready_unused),
        .s_axis_b_tdata       (b_fp32_w),
        .m_axis_result_tvalid (product_valid_w),
        .m_axis_result_tdata  (product_fp32_w)
    );

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (valid_i) begin
            assert (a_ready_unused && b_ready_unused)
                else $error("fp_mul_single_lat2_rate1 unexpectedly back-pressured");
        end
    end
`endif
endmodule

`ifndef SYNTHESIS
module fp_mul_single_lat2_rate1 (
    input  logic        aclk,
    input  logic        s_axis_a_tvalid,
    output logic        s_axis_a_tready,
    input  logic [31:0] s_axis_a_tdata,
    input  logic        s_axis_b_tvalid,
    output logic        s_axis_b_tready,
    input  logic [31:0] s_axis_b_tdata,
    output logic        m_axis_result_tvalid,
    output logic [31:0] m_axis_result_tdata
);
    localparam int LATENCY = 3;

    logic [LATENCY:0] valid_pipe_q;
    logic [31:0]      data_pipe_q [0:LATENCY];

    function automatic logic [31:0] fp32_mul_bits(
        input logic [31:0] a,
        input logic [31:0] b
    );
        shortreal ar;
        shortreal br;
        shortreal zr;
        begin
            ar = $bitstoshortreal(a);
            br = $bitstoshortreal(b);
            zr = ar * br;
            fp32_mul_bits = $shortrealtobits(zr);
        end
    endfunction

    assign s_axis_a_tready = 1'b1;
    assign s_axis_b_tready = 1'b1;
    assign m_axis_result_tvalid = valid_pipe_q[LATENCY];
    assign m_axis_result_tdata = data_pipe_q[LATENCY];

    always_ff @(posedge aclk) begin
        valid_pipe_q[0] <= s_axis_a_tvalid && s_axis_b_tvalid;
        data_pipe_q[0] <= fp32_mul_bits(s_axis_a_tdata, s_axis_b_tdata);
        for (int i = 1; i <= LATENCY; i++) begin
            valid_pipe_q[i] <= valid_pipe_q[i-1];
            data_pipe_q[i] <= data_pipe_q[i-1];
        end
    end
endmodule
`endif

`endif
