`timescale 1ns / 1ps

module quantization_fp32_to_uint8_fpga_backend #(
    parameter INPUT_NUMBER = 8,
    parameter int LATENCY = 4
) (
    input  logic                                     clk,
    input  logic                                     rst_n,
    input  logic                                     in_valid,
    input  logic [INPUT_NUMBER-1:0][31:0]            input_data,
    input  logic [31:0]                              quant_inv_scale,

    output logic                                     out_valid,
    output logic signed [INPUT_NUMBER-1:0][31:0]     output_data
);

localparam int TOTAL_LATENCY = LATENCY;

logic [TOTAL_LATENCY-1:0]                        valid_pipe;
logic signed [TOTAL_LATENCY-1:0][INPUT_NUMBER-1:0][31:0] data_pipe;

function automatic logic signed [31:0] fp32_mul_to_i32(
    input logic [31:0] lhs_bits,
    input logic [31:0] rhs_bits
);
    logic                   lhs_sign;
    logic [7:0]             lhs_exp;
    logic [22:0]            lhs_frac;
    logic                   rhs_sign;
    logic [7:0]             rhs_exp;
    logic [22:0]            rhs_frac;
    logic                   out_sign;
    logic                   lhs_is_zero;
    logic                   rhs_is_zero;
    logic                   lhs_is_inf;
    logic                   rhs_is_inf;
    logic                   lhs_is_nan;
    logic                   rhs_is_nan;
    logic [23:0]            lhs_mant;
    logic [23:0]            rhs_mant;
    int                     lhs_exp_unbiased;
    int                     rhs_exp_unbiased;
    int                     shift_amount;
    logic [47:0]            prod_mant;
    longint unsigned        abs_value;
    longint unsigned        rounded_abs;
    longint unsigned        lower_mask;
    longint unsigned        remainder;
    longint unsigned        half_ulp;
    logic signed [31:0]     sat_value;
    begin
        lhs_sign = lhs_bits[31];
        lhs_exp  = lhs_bits[30:23];
        lhs_frac = lhs_bits[22:0];

        rhs_sign = rhs_bits[31];
        rhs_exp  = rhs_bits[30:23];
        rhs_frac = rhs_bits[22:0];

        lhs_is_zero = (lhs_exp == 8'h00) && (lhs_frac == 23'd0);
        rhs_is_zero = (rhs_exp == 8'h00) && (rhs_frac == 23'd0);
        lhs_is_inf  = (lhs_exp == 8'hFF) && (lhs_frac == 23'd0);
        rhs_is_inf  = (rhs_exp == 8'hFF) && (rhs_frac == 23'd0);
        lhs_is_nan  = (lhs_exp == 8'hFF) && (lhs_frac != 23'd0);
        rhs_is_nan  = (rhs_exp == 8'hFF) && (rhs_frac != 23'd0);

        out_sign = lhs_sign ^ rhs_sign;

        if (lhs_is_nan || rhs_is_nan) begin
            fp32_mul_to_i32 = 32'sd0;
        end
        else if (lhs_is_inf || rhs_is_inf) begin
            if (lhs_is_zero || rhs_is_zero) begin
                fp32_mul_to_i32 = 32'sd0;
            end
            else if (out_sign) begin
                fp32_mul_to_i32 = 32'sh8000_0000;
            end
            else begin
                fp32_mul_to_i32 = 32'sh7FFF_FFFF;
            end
        end
        else if (lhs_is_zero || rhs_is_zero) begin
            fp32_mul_to_i32 = 32'sd0;
        end
        else begin
            lhs_mant         = (lhs_exp == 8'h00) ? {1'b0, lhs_frac} : {1'b1, lhs_frac};
            rhs_mant         = (rhs_exp == 8'h00) ? {1'b0, rhs_frac} : {1'b1, rhs_frac};
            lhs_exp_unbiased = (lhs_exp == 8'h00) ? -126 : (int'(lhs_exp) - 127);
            rhs_exp_unbiased = (rhs_exp == 8'h00) ? -126 : (int'(rhs_exp) - 127);
            prod_mant        = lhs_mant * rhs_mant;
            shift_amount     = lhs_exp_unbiased + rhs_exp_unbiased - 46;

            if (shift_amount >= 0) begin
                if (shift_amount >= 16) begin
                    abs_value = 64'hFFFF_FFFF_FFFF_FFFF;
                end
                else begin
                    abs_value = longint'(prod_mant) <<< shift_amount;
                end
                rounded_abs = abs_value;
            end
            else begin
                if (-shift_amount >= 64) begin
                    rounded_abs = 64'd0;
                end
                else begin
                    abs_value   = longint'(prod_mant) >> (-shift_amount);
                    if (-shift_amount == 1) begin
                        lower_mask = 64'h1;
                        half_ulp   = 64'h1;
                    end
                    else begin
                        lower_mask = (64'h1 << (-shift_amount)) - 64'h1;
                        half_ulp   = 64'h1 << ((-shift_amount) - 1);
                    end
                    remainder = longint'(prod_mant) & lower_mask;

                    rounded_abs = abs_value;
                    if ((remainder > half_ulp) || ((remainder == half_ulp) && abs_value[0])) begin
                        rounded_abs = abs_value + 64'd1;
                    end
                end
            end

            if (!out_sign) begin
                if (rounded_abs > 64'd2147483647) begin
                    sat_value = 32'sh7FFF_FFFF;
                end
                else begin
                    sat_value = $signed(rounded_abs[31:0]);
                end
            end
            else begin
                if (rounded_abs >= 64'd2147483648) begin
                    sat_value = 32'sh8000_0000;
                end
                else begin
                    sat_value = -$signed(rounded_abs[31:0]);
                end
            end

            fp32_mul_to_i32 = sat_value;
        end
    end
endfunction

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_pipe <= '0;
        data_pipe  <= '0;
    end
    else begin
        valid_pipe[0] <= in_valid;
        for (int lane = 0; lane < INPUT_NUMBER; lane++) begin
            data_pipe[0][lane] <= fp32_mul_to_i32(input_data[lane], quant_inv_scale);
        end

        for (int stage = 1; stage < TOTAL_LATENCY; stage++) begin
            valid_pipe[stage] <= valid_pipe[stage-1];
            data_pipe[stage]  <= data_pipe[stage-1];
        end
    end
end

assign out_valid   = valid_pipe[TOTAL_LATENCY-1];
assign output_data = data_pipe[TOTAL_LATENCY-1];

endmodule
