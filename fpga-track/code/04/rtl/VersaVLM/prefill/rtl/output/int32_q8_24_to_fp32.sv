`default_nettype none

// INT32 accumulator times Q8.24 scale to IEEE-754 single precision.
//
// Purpose:
//   Provide the Phase-1 INT8 mvout conversion primitive. The arithmetic model
//   is deterministic: `fp32 = float(acc_i32 * scale_q8_24 / 2^24)`.
//
// Clock/reset:
//   Synchronous active-high reset. Valid/data pass through a three-stage
//   pipeline: multiply, normalize metadata, and final FP32 pack.
//
// Rounding:
//   Phase-1 uses truncation toward zero after normalization. This is stable for
//   reference comparison but is not a correctly-rounded IEEE converter.
module int32_q8_24_to_fp32 (
    input  wire logic        clk_i,
    input  wire logic        rst_i,
    input  wire logic        valid_i,
    input  wire logic signed [31:0] acc_i,
    input  wire logic signed [31:0] scale_q8_24_i,
    output logic        valid_o,
    output logic [31:0] fp32_o
);
    function automatic logic [5:0] leading_one_idx(input logic [63:0] value);
        int msb_idx;
        begin
            msb_idx = 0;
            for (int bit_idx = 0; bit_idx < 64; bit_idx++) begin
                if (value[bit_idx]) begin
                    msb_idx = bit_idx;
                end
            end
            leading_one_idx = msb_idx[5:0];
        end
    endfunction

    logic              valid_s0;
    logic signed [63:0] product_s0;

    logic              valid_s1;
    logic              sign_s1;
    logic              zero_s1;
    logic [5:0]        msb_idx_s1;
    logic [7:0]        exponent_s1;
    logic [63:0]       abs_product_s1;

    logic              valid_s2;
    logic              sign_s2;
    logic              zero_s2;
    logic [7:0]        exponent_s2;
    logic [63:0]       mantissa_s2;

    logic [63:0] abs_product_s0;
    logic [5:0]  msb_idx_s0;
    logic [8:0]  exponent_biased_s0;

    assign abs_product_s0     = product_s0[63] ? (64'd0 - product_s0) : product_s0;
    assign msb_idx_s0         = leading_one_idx(abs_product_s0);
    assign exponent_biased_s0 = {3'b000, msb_idx_s0} + 9'd103;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            valid_s0       <= 1'b0;
            product_s0     <= 64'sd0;
            valid_s1       <= 1'b0;
            sign_s1        <= 1'b0;
            zero_s1        <= 1'b1;
            msb_idx_s1     <= 6'd0;
            exponent_s1    <= 8'd0;
            abs_product_s1 <= 64'd0;
            valid_s2       <= 1'b0;
            sign_s2        <= 1'b0;
            zero_s2        <= 1'b1;
            exponent_s2    <= 8'd0;
            mantissa_s2    <= 64'd0;
            valid_o        <= 1'b0;
            fp32_o         <= 32'h0000_0000;
        end else begin
            valid_s0   <= valid_i;
            product_s0 <= acc_i * scale_q8_24_i;

            valid_s1       <= valid_s0;
            sign_s1        <= product_s0[63];
            zero_s1        <= (product_s0 == 64'sd0);
            msb_idx_s1     <= msb_idx_s0;
            exponent_s1    <= exponent_biased_s0[7:0];
            abs_product_s1 <= abs_product_s0;

            valid_s2    <= valid_s1;
            sign_s2     <= sign_s1;
            zero_s2     <= zero_s1;
            exponent_s2 <= exponent_s1;
            if (msb_idx_s1 >= 6'd23) begin
                mantissa_s2 <= abs_product_s1 >> (msb_idx_s1 - 6'd23);
            end else begin
                mantissa_s2 <= abs_product_s1 << (6'd23 - msb_idx_s1);
            end

            valid_o <= valid_s2;
            fp32_o  <= zero_s2
                ? 32'h0000_0000
                : {sign_s2, exponent_s2, mantissa_s2[22:0]};
        end
    end
endmodule

`default_nettype wire
