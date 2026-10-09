`timescale 1ns / 1ps

module quantization_int32_to_fp32 #(
    parameter INPUT_NUMBER = 8
) (
    input  logic                                 clk,
    input  logic                                 rst_n,
    input  logic                                 in_valid,
    input  logic signed [INPUT_NUMBER-1:0][31:0] input_data,
    input  logic signed [INPUT_NUMBER-1:0][31:0] dequant_scale,

    output logic                                 out_valid,
    output logic [INPUT_NUMBER-1:0][31:0]        output_data
);

logic                                                   stage0_valid = 1'b0;
logic                                                   stage1_valid = 1'b0;
logic                                                   stage2_valid = 1'b0;
logic                                                   stage3_valid = 1'b0;
logic                                                   stage4_valid = 1'b0;

logic signed [INPUT_NUMBER-1:0][31:0]                   input_data_stage0 = '0;
logic signed [INPUT_NUMBER-1:0][31:0]                   dequant_scale_stage0 = '0;
logic signed [INPUT_NUMBER-1:0][63:0]                   fixed_mul_stage0 = '0;

logic [INPUT_NUMBER-1:0]                                fp_sign_stage1 = '0;
logic [INPUT_NUMBER-1:0]                                fp_zero_stage1 = '0;
logic [INPUT_NUMBER-1:0][63:0]                          fp_abs_stage1 = '0;
logic [INPUT_NUMBER-1:0][5:0]                           fp_msb_idx_stage1 = '0;

logic [INPUT_NUMBER-1:0]                                fp_sign_stage2 = '0;
logic [INPUT_NUMBER-1:0]                                fp_zero_stage2 = '0;
logic [INPUT_NUMBER-1:0][7:0]                           fp_exponent_stage2 = '0;
logic [INPUT_NUMBER-1:0][23:0]                          fp_mantissa_stage2 = '0;
logic [INPUT_NUMBER-1:0]                                fp_round_up_stage2 = '0;

logic [INPUT_NUMBER-1:0][31:0]                          fp_data_stage3 = '0;

function automatic logic signed [63:0] fixed_mul_i32_q8_24(
    input logic signed [31:0] int_value,
    input logic signed [31:0] scale_value
);
    logic signed [63:0] product;
    begin
        product = $signed(int_value) * $signed(scale_value);
        fixed_mul_i32_q8_24 = product;
    end
endfunction

function automatic logic [5:0] find_msb_idx64(
    input logic [63:0] value
);
    logic [5:0] msb_idx;
    logic       found_msb;
    begin
        msb_idx   = 6'd0;
        found_msb = 1'b0;
        for (int bit_idx = 63; bit_idx >= 0; bit_idx--) begin
            if (!found_msb && value[bit_idx]) begin
                msb_idx   = bit_idx[5:0];
                found_msb = 1'b1;
            end
        end
        find_msb_idx64 = msb_idx;
    end
endfunction

function automatic logic should_round_up(
    input logic [63:0] abs_value,
    input logic [5:0]  right_shift,
    input logic        mantissa_lsb
);
    logic [63:0] remainder_mask;
    logic [63:0] remainder_bits;
    logic [63:0] half_ulp;
    begin
        if (right_shift == 6'd0) begin
            should_round_up = 1'b0;
        end
        else begin
            remainder_mask = (64'h1 << right_shift) - 64'h1;
            remainder_bits = abs_value & remainder_mask;
            half_ulp       = 64'h1 << (right_shift - 6'd1);

            should_round_up =
                (remainder_bits > half_ulp) ||
                ((remainder_bits == half_ulp) && mantissa_lsb);
        end
    end
endfunction

always_ff @(posedge clk) begin
    if (!rst_n) begin
        stage0_valid <= 1'b0;
        stage1_valid <= 1'b0;
        stage2_valid <= 1'b0;
        stage3_valid <= 1'b0;
        stage4_valid <= 1'b0;
        input_data_stage0 <= '0;
        dequant_scale_stage0 <= '0;
        fixed_mul_stage0 <= '0;
        fp_sign_stage1   <= '0;
        fp_zero_stage1   <= '0;
        fp_abs_stage1    <= '0;
        fp_msb_idx_stage1 <= '0;
        fp_sign_stage2   <= '0;
        fp_zero_stage2   <= '0;
        fp_exponent_stage2 <= '0;
        fp_mantissa_stage2 <= '0;
        fp_round_up_stage2 <= '0;
        fp_data_stage3   <= '0;
    end
    else begin
        stage0_valid <= in_valid;
        stage1_valid <= stage0_valid;
        stage2_valid <= stage1_valid;
        stage3_valid <= stage2_valid;
        stage4_valid <= stage3_valid;

        if (in_valid) begin
            input_data_stage0 <= input_data;
            dequant_scale_stage0 <= dequant_scale;
        end

        if (stage0_valid) begin
            for (int lane = 0; lane < INPUT_NUMBER; lane++) begin
                fixed_mul_stage0[lane] <= fixed_mul_i32_q8_24(input_data_stage0[lane], dequant_scale_stage0[lane]);
            end
        end

        if (stage1_valid) begin
            for (int lane = 0; lane < INPUT_NUMBER; lane++) begin
                logic signed [63:0] fixed_value;
                logic [63:0]        abs_value;

                fixed_value = fixed_mul_stage0[lane];
                abs_value   = fixed_value[63] ? $unsigned(-fixed_value) : $unsigned(fixed_value);

                fp_sign_stage1[lane]     <= fixed_value[63];
                fp_zero_stage1[lane]     <= (fixed_value == 64'sd0);
                fp_abs_stage1[lane]      <= abs_value;
                fp_msb_idx_stage1[lane]  <= find_msb_idx64(abs_value);
            end
        end

        if (stage2_valid) begin
            for (int lane = 0; lane < INPUT_NUMBER; lane++) begin
                logic [5:0]  shift_amount;
                logic [23:0] mantissa_24;
                logic        round_up;

                fp_sign_stage2[lane]     <= fp_sign_stage1[lane];
                fp_zero_stage2[lane]     <= fp_zero_stage1[lane];
                fp_exponent_stage2[lane] <= {2'b00, fp_msb_idx_stage1[lane]} + 8'd103;

                if (fp_zero_stage1[lane]) begin
                    fp_mantissa_stage2[lane] <= 24'd0;
                    fp_round_up_stage2[lane] <= 1'b0;
                end
                else if (fp_msb_idx_stage1[lane] <= 6'd23) begin
                    shift_amount = 6'd23 - fp_msb_idx_stage1[lane];
                    mantissa_24  = fp_abs_stage1[lane] << shift_amount;
                    fp_mantissa_stage2[lane] <= mantissa_24;
                    fp_round_up_stage2[lane] <= 1'b0;
                end
                else begin
                    shift_amount = fp_msb_idx_stage1[lane] - 6'd23;
                    mantissa_24  = fp_abs_stage1[lane] >> shift_amount;
                    round_up     = should_round_up(fp_abs_stage1[lane], shift_amount, mantissa_24[0]);
                    fp_mantissa_stage2[lane] <= mantissa_24;
                    fp_round_up_stage2[lane] <= round_up;
                end
            end
        end

        if (stage3_valid) begin
            for (int lane = 0; lane < INPUT_NUMBER; lane++) begin
                logic [23:0] mantissa_24;
                logic [7:0]  exponent_bits;

                mantissa_24   = fp_mantissa_stage2[lane];
                exponent_bits = fp_exponent_stage2[lane];

                if (fp_zero_stage2[lane]) begin
                    fp_data_stage3[lane] <= 32'h0000_0000;
                end
                else begin
                    if (fp_round_up_stage2[lane]) begin
                        if (mantissa_24 == 24'hFF_FFFF) begin
                            mantissa_24   = 24'h800000;
                            exponent_bits = exponent_bits + 8'd1;
                        end
                        else begin
                            mantissa_24 = mantissa_24 + 24'd1;
                        end
                    end

                    fp_data_stage3[lane] <= {fp_sign_stage2[lane], exponent_bits, mantissa_24[22:0]};
                end
            end
        end
    end
end

assign out_valid   = stage4_valid;
assign output_data = fp_data_stage3;

endmodule
