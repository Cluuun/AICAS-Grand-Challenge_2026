`default_nettype none

// Four-cycle approximate reciprocal for PV row denominators.
//
// The unit avoids variable division. It normalizes the unsigned denominator to
// x in [1, 2), builds a linear seed for 1/x, and applies one Newton step:
//     y1 = y0 * (2 - x*y0)
// y1 is Q1.20 and the final reciprocal is y1 >> exp.
module pv_recip_approx_q20 (
    input  wire logic        clk_i,
    input  wire logic        rst_i,

    input  wire logic        valid_i,
    input  wire logic [4:0]  tag_i,
    input  wire logic [31:0] den_i,

    output logic             valid_o,
    output logic [4:0]       tag_o,
    output logic             zero_o,
    output logic [20:0]      recip_mant_o,
    output logic [5:0]       recip_shift_o
);
    localparam int FRAC_BITS = 20;
    localparam logic [20:0] ONE_Q20 = 21'd1048576;
    localparam logic [21:0] TWO_Q20 = 22'd2097152;
    localparam logic [21:0] SEED_A_Q20 = 22'd1480343; // round((24/17) * 2^20)
    localparam logic [20:0] SEED_B_Q20 = 21'd493448;  // round((8/17) * 2^20)

    logic        valid_s0_q;
    logic [4:0]  tag_s0_q;
    logic        zero_s0_q;
    logic [4:0]  exp_s0_q;
    logic [20:0] x_s0_q;

    logic        valid_s1_q;
    logic [4:0]  tag_s1_q;
    logic        zero_s1_q;
    logic [4:0]  exp_s1_q;
    logic [20:0] x_s1_q;
    logic [20:0] y0_s1_q;

    logic        valid_s2_q;
    logic [4:0]  tag_s2_q;
    logic        zero_s2_q;
    logic [4:0]  exp_s2_q;
    logic [20:0] y0_s2_q;
    logic [21:0] xy_s2_q;

    logic        valid_s3_q;
    logic [4:0]  tag_s3_q;
    logic        zero_s3_q;
    logic [4:0]  exp_s3_q;
    logic [20:0] y0_s3_q;
    logic [21:0] term_s3_q;

    function automatic logic [4:0] msb_index(input logic [31:0] value);
        logic [4:0] idx;
        begin
            idx = '0;
            for (int bit_idx = 0; bit_idx < 32; bit_idx++) begin
                if (value[bit_idx]) begin
                    idx = 5'(bit_idx);
                end
            end
            return idx;
        end
    endfunction

    function automatic logic [20:0] normalize_q20(input logic [31:0] value,
                                                  input logic [4:0] exp);
        logic [31:0] shifted;
        begin
            if (value == '0) begin
                normalize_q20 = '0;
            end else if (exp >= 5'd20) begin
                shifted = value >> (exp - 5'd20);
                normalize_q20 = shifted[20:0];
            end else begin
                normalize_q20 = 21'(value << (5'd20 - exp));
            end
        end
    endfunction

    function automatic logic [20:0] seed_y0(input logic [20:0] x_q20);
        logic [41:0] bx;
        logic [21:0] bx_q20;
        begin
            bx = SEED_B_Q20 * x_q20;
            bx_q20 = 22'(bx >> FRAC_BITS);
            seed_y0 = 21'(SEED_A_Q20 - bx_q20);
        end
    endfunction

    always_ff @(posedge clk_i) begin
        logic [4:0] exp_next;
        logic [41:0] xy_product;
        logic [42:0] y_product;

        if (rst_i) begin
            valid_s0_q    <= 1'b0;
            tag_s0_q      <= '0;
            zero_s0_q     <= 1'b0;
            exp_s0_q      <= '0;
            x_s0_q        <= '0;
            valid_s1_q    <= 1'b0;
            tag_s1_q      <= '0;
            zero_s1_q     <= 1'b0;
            exp_s1_q      <= '0;
            x_s1_q        <= '0;
            y0_s1_q       <= '0;
            valid_s2_q    <= 1'b0;
            tag_s2_q      <= '0;
            zero_s2_q     <= 1'b0;
            exp_s2_q      <= '0;
            y0_s2_q       <= '0;
            xy_s2_q       <= '0;
            valid_s3_q    <= 1'b0;
            tag_s3_q      <= '0;
            zero_s3_q     <= 1'b0;
            exp_s3_q      <= '0;
            y0_s3_q       <= '0;
            term_s3_q     <= '0;
            valid_o       <= 1'b0;
            tag_o         <= '0;
            zero_o        <= 1'b0;
            recip_mant_o  <= '0;
            recip_shift_o <= '0;
        end else begin
            exp_next = msb_index(den_i);

            valid_s0_q <= valid_i;
            tag_s0_q   <= tag_i;
            zero_s0_q  <= (den_i == '0);
            exp_s0_q   <= exp_next;
            x_s0_q     <= normalize_q20(den_i, exp_next);

            valid_s1_q <= valid_s0_q;
            tag_s1_q   <= tag_s0_q;
            zero_s1_q  <= zero_s0_q;
            exp_s1_q   <= exp_s0_q;
            x_s1_q     <= x_s0_q;
            y0_s1_q    <= zero_s0_q ? '0 : seed_y0(x_s0_q);

            xy_product = x_s1_q * y0_s1_q;
            valid_s2_q <= valid_s1_q;
            tag_s2_q   <= tag_s1_q;
            zero_s2_q  <= zero_s1_q;
            exp_s2_q   <= exp_s1_q;
            y0_s2_q    <= y0_s1_q;
            xy_s2_q    <= 22'(xy_product >> FRAC_BITS);

            valid_s3_q <= valid_s2_q;
            tag_s3_q   <= tag_s2_q;
            zero_s3_q  <= zero_s2_q;
            exp_s3_q   <= exp_s2_q;
            y0_s3_q    <= y0_s2_q;
            term_s3_q  <= zero_s2_q ? '0 : (TWO_Q20 - xy_s2_q);

            y_product = y0_s3_q * term_s3_q;
            valid_o <= valid_s3_q;
            tag_o <= tag_s3_q;
            zero_o <= zero_s3_q;
            recip_mant_o <= zero_s3_q ? '0 : 21'(y_product >> FRAC_BITS);
            recip_shift_o <= zero_s3_q ? '0 : ({1'b0, exp_s3_q} + 6'(FRAC_BITS));
        end
    end
endmodule

`default_nettype wire
