`default_nettype none

// Phase-1 INT8 single-lane DSP MAC.
//
// The pipeline preserves the PE_MAC_LAT contract used by the SA scheduler:
// input samples are delayed PIPE_STAGES cycles before updating the live P
// accumulator. clear_i resets only the live accumulator.
module int8_dsp_mac_lane #(
    parameter int PIPE_STAGES = 1,
    parameter int A_W         = 8,
    parameter int W_W         = 8,
    parameter int ACC_W       = 32
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,

    input  wire logic              valid_i,
    input  wire logic signed [A_W-1:0] a_i,
    input  wire logic signed [W_W-1:0] w_i,
    input  wire logic              clear_i,

    output logic signed [ACC_W-1:0] acc_o
);

    localparam int PIPE_W = (PIPE_STAGES <= 0) ? 1 : PIPE_STAGES;

    logic [PIPE_W-1:0] valid_pipe_q;
    logic signed [A_W-1:0] a_pipe_q [PIPE_W-1:0];
    logic signed [W_W-1:0] w_pipe_q [PIPE_W-1:0];
    logic signed [A_W-1:0] mac_a;
    logic signed [W_W-1:0] mac_w;
    logic              mac_valid;
    logic signed [26:0] mac_a_ext;
    logic signed [17:0] mac_w_ext;
    (* use_dsp = "yes" *) logic signed [47:0] mac_p_q;
    (* use_dsp = "yes" *) logic signed [47:0] mac_next;

    assign mac_a_ext = $signed(mac_a);
    assign mac_w_ext = $signed(mac_w);
    assign mac_next  = mac_p_q + (mac_a_ext * mac_w_ext);

    initial begin
        if ((A_W < 1) || (A_W > 27)) begin
            $error("int8_dsp_mac_lane: A_W must fit in the 27-bit DSP A input");
        end
        if ((W_W < 1) || (W_W > 18)) begin
            $error("int8_dsp_mac_lane: W_W must fit in the 18-bit DSP B input");
        end
    end

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

`default_nettype wire
