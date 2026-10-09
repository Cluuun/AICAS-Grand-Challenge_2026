`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// rope_rotate_pe.sv — Single-pair rotation PE for the RoPE Rotary Engine.
//
// Applies one 2x2 rotation matrix to an (x_even, x_odd) pair:
//
//     [ x_even' ]   [ cos  -sin ] [ x_even ]
//     [ x_odd'  ] = [ sin   cos ] [ x_odd  ]
//
// Datapath:
//     Stage 1 : latch inputs
//     Stage 2 : four signed multiplies, vector data is Q6.14 and cos/sin is Q1.15
//     Stage 3 : subtract / add in 32-bit signed accumulator
//     Stage 4 : arithmetic right-shift by 15, saturate back to Q6.14
//
// The wider Q6.14 vector format covers model-observed Q/K projection values
// around +/-16 without saturating before RoPE.
// -----------------------------------------------------------------------------
`default_nettype none

module rope_rotate_pe #(
    parameter int DATA_WIDTH = 20,     // Q6.14 signed
    parameter int COEF_WIDTH = 16,     // Q1.15 signed
    parameter int COEF_FRAC  = 15
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         valid_in,
    input  wire signed [DATA_WIDTH-1:0] x_even_in,
    input  wire signed [DATA_WIDTH-1:0] x_odd_in,
    input  wire signed [COEF_WIDTH-1:0] cos_in,
    input  wire signed [COEF_WIDTH-1:0] sin_in,
    output reg                          valid_out,
    output reg  signed [DATA_WIDTH-1:0] x_even_out,
    output reg  signed [DATA_WIDTH-1:0] x_odd_out
);

    localparam int PROD_W  = DATA_WIDTH + COEF_WIDTH;
    localparam int ACC_W   = PROD_W + 1;
    localparam int SHIFT_W = ACC_W - COEF_FRAC;

    // ---------------- Stage 1: register inputs ----------------
    reg                         s1_valid;
    reg signed [DATA_WIDTH-1:0] s1_xe, s1_xo;
    reg signed [COEF_WIDTH-1:0] s1_cos, s1_sin;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
        end else begin
            s1_valid <= valid_in;
            s1_xe    <= x_even_in;
            s1_xo    <= x_odd_in;
            s1_cos   <= cos_in;
            s1_sin   <= sin_in;
        end
    end

    // ---------------- Stage 2: four parallel multiplies ----------------
    // Inference of DSP48E2 (UltraScale+): signed * signed, registered.
    reg                    s2_valid;
    (* use_dsp = "yes" *)
    reg signed [PROD_W-1:0] s2_p_ec, s2_p_os, s2_p_es, s2_p_oc;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
        end else begin
            s2_valid <= s1_valid;
            s2_p_ec  <= s1_xe * s1_cos;   // x_even * cos
            s2_p_os  <= s1_xo * s1_sin;   // x_odd  * sin
            s2_p_es  <= s1_xe * s1_sin;   // x_even * sin
            s2_p_oc  <= s1_xo * s1_cos;   // x_odd  * cos
        end
    end

    // ---------------- Stage 3: add / subtract ----------------
    reg                   s3_valid;
    reg signed [ACC_W-1:0] s3_even_acc, s3_odd_acc;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s3_valid <= 1'b0;
        end else begin
            s3_valid    <= s2_valid;
            s3_even_acc <= $signed({s2_p_ec[PROD_W-1], s2_p_ec})
                         - $signed({s2_p_os[PROD_W-1], s2_p_os});
            s3_odd_acc  <= $signed({s2_p_es[PROD_W-1], s2_p_es})
                         + $signed({s2_p_oc[PROD_W-1], s2_p_oc});
        end
    end

    // ---------------- Stage 4: arithmetic shift + saturate ----------------
    wire signed [SHIFT_W-1:0] s3_even_shift = s3_even_acc >>> COEF_FRAC;
    wire signed [SHIFT_W-1:0] s3_odd_shift  = s3_odd_acc  >>> COEF_FRAC;

    function automatic signed [DATA_WIDTH-1:0] sat_fixed(input signed [SHIFT_W-1:0] v);
        localparam signed [DATA_WIDTH-1:0] POS_MAX = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        localparam signed [DATA_WIDTH-1:0] NEG_MIN = {1'b1, {(DATA_WIDTH-1){1'b0}}};
        if      (v >  $signed(POS_MAX)) sat_fixed = POS_MAX;
        else if (v <  $signed(NEG_MIN)) sat_fixed = NEG_MIN;
        else                            sat_fixed = v[DATA_WIDTH-1:0];
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            valid_out  <= 1'b0;
        end else begin
            valid_out  <= s3_valid;
            x_even_out <= sat_fixed(s3_even_shift);
            x_odd_out  <= sat_fixed(s3_odd_shift);
        end
    end

endmodule

`default_nettype wire
