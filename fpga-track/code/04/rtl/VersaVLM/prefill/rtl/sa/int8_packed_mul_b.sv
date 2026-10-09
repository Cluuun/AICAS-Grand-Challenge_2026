`default_nettype none

// Phase-1 INT8 single-lane DSP MAC.
//
// The pipeline preserves the PE_MAC_LAT contract used by the SA scheduler:
// input samples are delayed PIPE_STAGES cycles before updating the live P
// accumulator. clear_i resets only the live accumulator, matching the old PE
// behavior where the multiply pipeline was independent from clear_acc_i.
module int8_dsp_mac_lane #(
    parameter int PIPE_STAGES = 1,
    parameter int ACC_W       = 32
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,

    input  wire logic              valid_i,
    input  wire logic signed [7:0] a_i,
    input  wire logic signed [7:0] w_i,
    input  wire logic              clear_i,

    output logic signed [ACC_W-1:0] acc_o
);

    localparam int PIPE_W = (PIPE_STAGES <= 0) ? 1 : PIPE_STAGES;

    logic [PIPE_W-1:0] valid_pipe_q;
    logic signed [7:0] a_pipe_q [PIPE_W-1:0];
    logic signed [7:0] w_pipe_q [PIPE_W-1:0];
    logic signed [7:0] mac_a;
    logic signed [7:0] mac_w;
    logic              mac_valid;
    logic signed [26:0] mac_a_ext;
    logic signed [17:0] mac_w_ext;
    (* use_dsp = "yes" *) logic signed [47:0] mac_p_q;
    (* use_dsp = "yes" *) logic signed [47:0] mac_next;

    assign mac_a_ext = {{19{mac_a[7]}}, mac_a};
    assign mac_w_ext = {{10{mac_w[7]}}, mac_w};
    assign mac_next  = mac_p_q + (mac_a_ext * mac_w_ext);

    generate
        if (PIPE_STAGES <= 0) begin : gen_no_pipe
            assign mac_valid = valid_i;
            assign mac_a = a_i;
            assign mac_w = w_i;

            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    valid_pipe_q <= '0;
                    for (int stage = 0; stage < PIPE_W; stage++) begin
                        a_pipe_q[stage] <= '0;
                        w_pipe_q[stage] <= '0;
                    end
                end
            end
        end else begin : gen_pipe
            assign mac_valid = valid_pipe_q[PIPE_STAGES-1];
            assign mac_a = a_pipe_q[PIPE_STAGES-1];
            assign mac_w = w_pipe_q[PIPE_STAGES-1];

            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    valid_pipe_q <= '0;
                    for (int stage = 0; stage < PIPE_STAGES; stage++) begin
                        a_pipe_q[stage] <= '0;
                        w_pipe_q[stage] <= '0;
                    end
                end else begin
                    valid_pipe_q[0] <= valid_i;
                    a_pipe_q[0] <= a_i;
                    w_pipe_q[0] <= w_i;
                    for (int stage = 1; stage < PIPE_STAGES; stage++) begin
                        valid_pipe_q[stage] <= valid_pipe_q[stage-1];
                        a_pipe_q[stage] <= a_pipe_q[stage-1];
                        w_pipe_q[stage] <= w_pipe_q[stage-1];
                    end
                end
            end
        end
    endgenerate

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            mac_p_q <= '0;
        end else if (clear_i) begin
            mac_p_q <= '0;
        end else if (mac_valid) begin
            mac_p_q <= mac_next;
        end
    end

    assign acc_o = mac_p_q[ACC_W-1:0];

endmodule

// Phase-1 INT8 shared-W multiplier.
//
// This block implements the INT8 B scheme used by one physical PE:
// two signed A lanes share one signed W lane. This exploration intentionally
// does not pack both lanes into one DSP; each logical PE lane uses its own
// signed INT8 multiply while the surrounding PE/SA interface stays unchanged.
module int8_packed_mul_b #(
    parameter int PIPE_STAGES = 1
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,

    input  wire logic              valid_i,
    input  wire logic [15:0]       a_pair_i,
    input  wire logic [7:0]        w_i,

    output logic              valid_o,
    output logic signed [17:0] prod_even_o,
    output logic signed [17:0] prod_odd_o
);

    localparam int PRODUCT_PIPE_STAGES = (PIPE_STAGES > 1) ? (PIPE_STAGES - 1) : PIPE_STAGES;

    logic              mul_valid;
    logic [15:0]       mul_a_pair;
    logic [7:0]        mul_w;
    logic signed [7:0] a_even_s;
    logic signed [7:0] a_odd_s;
    logic signed [7:0] w_s;
    (* use_dsp = "yes" *) logic signed [15:0] product_even_comb;
    (* use_dsp = "yes" *) logic signed [15:0] product_odd_comb;
    logic signed [17:0] prod_even_comb;
    logic signed [17:0] prod_odd_comb;

    generate
        if (PIPE_STAGES > 1) begin : gen_input_pipe
            logic        valid_q;
            logic [15:0] a_pair_q;
            logic [7:0]  w_q;

            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    valid_q  <= 1'b0;
                    a_pair_q <= '0;
                    w_q      <= '0;
                end else begin
                    valid_q  <= valid_i;
                    a_pair_q <= a_pair_i;
                    w_q      <= w_i;
                end
            end

            assign mul_valid  = valid_q;
            assign mul_a_pair = a_pair_q;
            assign mul_w      = w_q;
        end else begin : gen_no_input_pipe
            assign mul_valid  = valid_i;
            assign mul_a_pair = a_pair_i;
            assign mul_w      = w_i;
        end
    endgenerate

    assign a_even_s = mul_a_pair[7:0];
    assign a_odd_s  = mul_a_pair[15:8];
    assign w_s      = mul_w;

    assign product_even_comb = a_even_s * w_s;
    assign product_odd_comb  = a_odd_s * w_s;
    assign prod_even_comb    = {{2{product_even_comb[15]}}, product_even_comb};
    assign prod_odd_comb     = {{2{product_odd_comb[15]}},  product_odd_comb};

    generate
        if (PIPE_STAGES >= 3) begin : gen_split_product_pipe
            localparam int OUTPUT_PIPE_STAGES = PIPE_STAGES - 2;

            integer stage;
            logic valid_product_q;
            (* keep = "true", dont_touch = "true" *) logic signed [15:0] product_even_q;
            (* keep = "true", dont_touch = "true" *) logic signed [15:0] product_odd_q;
            logic signed [17:0] prod_even_split_comb;
            logic signed [17:0] prod_odd_split_comb;
            logic [OUTPUT_PIPE_STAGES-1:0] valid_q;
            logic signed [17:0] prod_even_q [OUTPUT_PIPE_STAGES-1:0];
            logic signed [17:0] prod_odd_q  [OUTPUT_PIPE_STAGES-1:0];

            assign prod_even_split_comb = {{2{product_even_q[15]}}, product_even_q};
            assign prod_odd_split_comb  = {{2{product_odd_q[15]}},  product_odd_q};

            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    valid_product_q  <= 1'b0;
                    product_even_q   <= '0;
                    product_odd_q    <= '0;
                    valid_q          <= '0;
                    for (stage = 0; stage < OUTPUT_PIPE_STAGES; stage = stage + 1) begin
                        prod_even_q[stage] <= '0;
                        prod_odd_q[stage]  <= '0;
                    end
                end else begin
                    valid_product_q  <= mul_valid;
                    product_even_q   <= product_even_comb;
                    product_odd_q    <= product_odd_comb;

                    valid_q[0]     <= valid_product_q;
                    prod_even_q[0] <= prod_even_split_comb;
                    prod_odd_q[0]  <= prod_odd_split_comb;

                    for (stage = 1; stage < OUTPUT_PIPE_STAGES; stage = stage + 1) begin
                        valid_q[stage]      <= valid_q[stage-1];
                        prod_even_q[stage] <= prod_even_q[stage-1];
                        prod_odd_q[stage]  <= prod_odd_q[stage-1];
                    end
                end
            end

            assign valid_o     = valid_q[OUTPUT_PIPE_STAGES-1];
            assign prod_even_o = prod_even_q[OUTPUT_PIPE_STAGES-1];
            assign prod_odd_o  = prod_odd_q[OUTPUT_PIPE_STAGES-1];
        end else if (PRODUCT_PIPE_STAGES == 0) begin : gen_no_product_pipe
            assign valid_o     = mul_valid;
            assign prod_even_o = prod_even_comb;
            assign prod_odd_o  = prod_odd_comb;
        end else begin : gen_product_pipe
            integer stage;
            logic [PRODUCT_PIPE_STAGES-1:0] valid_q;
            logic signed [17:0] prod_even_q [PRODUCT_PIPE_STAGES-1:0];
            logic signed [17:0] prod_odd_q  [PRODUCT_PIPE_STAGES-1:0];

            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    valid_q <= '0;
                    for (stage = 0; stage < PRODUCT_PIPE_STAGES; stage = stage + 1) begin
                        prod_even_q[stage] <= '0;
                        prod_odd_q[stage]  <= '0;
                    end
                end else begin
                    valid_q[0]     <= mul_valid;
                    prod_even_q[0] <= prod_even_comb;
                    prod_odd_q[0]  <= prod_odd_comb;

                    for (stage = 1; stage < PRODUCT_PIPE_STAGES; stage = stage + 1) begin
                        valid_q[stage]      <= valid_q[stage-1];
                        prod_even_q[stage] <= prod_even_q[stage-1];
                        prod_odd_q[stage]  <= prod_odd_q[stage-1];
                    end
                end
            end

            assign valid_o     = valid_q[PRODUCT_PIPE_STAGES-1];
            assign prod_even_o = prod_even_q[PRODUCT_PIPE_STAGES-1];
            assign prod_odd_o  = prod_odd_q[PRODUCT_PIPE_STAGES-1];
        end
    endgenerate

endmodule

`default_nettype wire
