`default_nettype none

// Normalize one 256-bit PV numerator drain beat with one row denominator.
module pv_i32_row_normalize #(
    parameter int ADDR_W = 14
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,
    input  wire logic              clear_i,

    input  wire logic              in_valid_i,
    output logic                   in_ready_o,
    input  wire logic [255:0]      numerator_i,
    input  wire logic [31:0]       denominator_i,
    input  wire logic [7:0]        lane_mask_i,
    input  wire logic [ADDR_W-1:0] word_addr_i,
    input  wire logic [1:0]        word_group_idx_i,

    output logic                   out_valid_o,
    input  wire logic              out_ready_i,
    output logic [255:0]           normalized_o,
    output logic [7:0]             lane_mask_o,
    output logic [ADDR_W-1:0]      word_addr_o,
    output logic [1:0]             word_group_idx_o,
    output logic                   busy_o
);
    localparam int LANES = 8;
    localparam int DIV_STAGES = 8;

    logic [DIV_STAGES-1:0] stage_valid_q;
    logic stage_can_load [0:DIV_STAGES-1];
    logic [39:0] rem_q [0:DIV_STAGES-1][0:LANES-1];
    logic [7:0] quot_q [0:DIV_STAGES-1][0:LANES-1];
    logic sign_q [0:DIV_STAGES-1][0:LANES-1];
    logic [31:0] den_q [0:DIV_STAGES-1];
    logic den_zero_q [0:DIV_STAGES-1];
    logic [7:0] lane_mask_pipe_q [0:DIV_STAGES-1];
    logic [ADDR_W-1:0] word_addr_pipe_q [0:DIV_STAGES-1];
    logic [1:0] word_group_idx_pipe_q [0:DIV_STAGES-1];

    logic out_valid_q;
    logic [255:0] normalized_q;
    logic [7:0] lane_mask_q;
    logic [ADDR_W-1:0] word_addr_q;
    logic [1:0] word_group_idx_q;

    wire out_can_load = !out_valid_q || out_ready_i;

    function automatic logic [32:0] abs_i32(input logic signed [31:0] value);
        begin
            if (value[31]) begin
                abs_i32 = {1'b0, (~value + 32'd1)};
            end else begin
                abs_i32 = {1'b0, value};
            end
        end
    endfunction

    function automatic logic [39:0] trial_for_bit(
        input logic [31:0] den,
        input int unsigned bit_idx
    );
        begin
            trial_for_bit = {8'd0, den} << bit_idx;
        end
    endfunction

    function automatic logic [39:0] div_next_rem(
        input logic [39:0] rem,
        input logic [31:0] den,
        input int unsigned bit_idx
    );
        logic [39:0] trial;
        begin
            trial = trial_for_bit(den, bit_idx);
            if ((den != 32'd0) && (rem >= trial)) begin
                div_next_rem = rem - trial;
            end else begin
                div_next_rem = rem;
            end
        end
    endfunction

    function automatic logic [7:0] div_next_quot(
        input logic [7:0] quot,
        input logic [39:0] rem,
        input logic [31:0] den,
        input int unsigned bit_idx
    );
        logic [39:0] trial;
        begin
            trial = trial_for_bit(den, bit_idx);
            div_next_quot = quot;
            if ((den != 32'd0) && (rem >= trial)) begin
                div_next_quot[bit_idx] = 1'b1;
            end
        end
    endfunction

    function automatic logic [8:0] rounded_abs_quot(
        input logic [7:0] quot_floor,
        input logic [39:0] rem,
        input logic [31:0] den,
        input logic den_zero
    );
        logic [40:0] rem_x2;
        begin
            if (den_zero) begin
                rounded_abs_quot = 9'd0;
            end else begin
                rem_x2 = {rem, 1'b0};
                rounded_abs_quot = {1'b0, quot_floor} +
                    ((rem_x2 >= {9'd0, den}) ? 9'd1 : 9'd0);
            end
        end
    endfunction

    function automatic logic signed [31:0] restore_sat_i8(
        input logic sign,
        input logic [8:0] abs_quot,
        input logic den_zero
    );
        begin
            if (den_zero || (abs_quot == 9'd0)) begin
                restore_sat_i8 = 32'sd0;
            end else if (sign) begin
                if (abs_quot > 9'd127) begin
                    restore_sat_i8 = -32'sd127;
                end else begin
                    restore_sat_i8 = -$signed({24'd0, abs_quot[7:0]});
                end
            end else begin
                if (abs_quot > 9'd127) begin
                    restore_sat_i8 = 32'sd127;
                end else begin
                    restore_sat_i8 = $signed({24'd0, abs_quot[7:0]});
                end
            end
        end
    endfunction

    always_comb begin
        stage_can_load[DIV_STAGES-1] =
            !stage_valid_q[DIV_STAGES-1] || out_can_load;
        for (int stage = DIV_STAGES - 2; stage >= 0; stage--) begin
            stage_can_load[stage] =
                !stage_valid_q[stage] || stage_can_load[stage + 1];
        end
    end

    assign in_ready_o = stage_can_load[0];
    assign out_valid_o = out_valid_q;
    assign normalized_o = normalized_q;
    assign lane_mask_o = lane_mask_q;
    assign word_addr_o = word_addr_q;
    assign word_group_idx_o = word_group_idx_q;
    assign busy_o = (|stage_valid_q) || out_valid_q;

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (!rst_i && !clear_i && in_valid_i && in_ready_o &&
            (denominator_i == 32'd0)) begin
            $warning("pv_i32_row_normalize accepted denominator zero; output beat is forced to zero");
        end
    end
`endif

    always_ff @(posedge clk_i) begin
        logic signed [31:0] lane_value;
        logic [32:0] abs_value;
        logic [39:0] initial_rem;
        logic [8:0] abs_rounded;

        if (rst_i || clear_i) begin
            stage_valid_q <= '0;
            out_valid_q <= 1'b0;
        end else begin
            if (out_can_load) begin
                out_valid_q <= stage_valid_q[DIV_STAGES-1];
                if (stage_valid_q[DIV_STAGES-1]) begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        abs_rounded = rounded_abs_quot(
                            quot_q[DIV_STAGES-1][lane],
                            rem_q[DIV_STAGES-1][lane],
                            den_q[DIV_STAGES-1],
                            den_zero_q[DIV_STAGES-1]
                        );
                        normalized_q[lane*32 +: 32] <= restore_sat_i8(
                            sign_q[DIV_STAGES-1][lane],
                            abs_rounded,
                            den_zero_q[DIV_STAGES-1]
                        );
                    end
                    lane_mask_q <= lane_mask_pipe_q[DIV_STAGES-1];
                    word_addr_q <= word_addr_pipe_q[DIV_STAGES-1];
                    word_group_idx_q <= word_group_idx_pipe_q[DIV_STAGES-1];
                end
            end

            for (int stage = DIV_STAGES - 1; stage > 0; stage--) begin
                if (stage_can_load[stage]) begin
                    stage_valid_q[stage] <= stage_valid_q[stage - 1];
                    if (stage_valid_q[stage - 1]) begin
                        for (int lane = 0; lane < LANES; lane++) begin
                            rem_q[stage][lane] <= div_next_rem(
                                rem_q[stage - 1][lane],
                                den_q[stage - 1],
                                7 - stage
                            );
                            quot_q[stage][lane] <= div_next_quot(
                                quot_q[stage - 1][lane],
                                rem_q[stage - 1][lane],
                                den_q[stage - 1],
                                7 - stage
                            );
                            sign_q[stage][lane] <= sign_q[stage - 1][lane];
                        end
                        den_q[stage] <= den_q[stage - 1];
                        den_zero_q[stage] <= den_zero_q[stage - 1];
                        lane_mask_pipe_q[stage] <= lane_mask_pipe_q[stage - 1];
                        word_addr_pipe_q[stage] <= word_addr_pipe_q[stage - 1];
                        word_group_idx_pipe_q[stage] <= word_group_idx_pipe_q[stage - 1];
                    end
                end
            end

            if (stage_can_load[0]) begin
                stage_valid_q[0] <= in_valid_i;
                if (in_valid_i) begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        lane_value = numerator_i[lane*32 +: 32];
                        abs_value = abs_i32(lane_value);
                        initial_rem = {7'd0, abs_value};
                        rem_q[0][lane] <= div_next_rem(
                            initial_rem,
                            denominator_i,
                            7
                        );
                        quot_q[0][lane] <= div_next_quot(
                            8'd0,
                            initial_rem,
                            denominator_i,
                            7
                        );
                        sign_q[0][lane] <= lane_value[31];
                    end
                    den_q[0] <= denominator_i;
                    den_zero_q[0] <= (denominator_i == 32'd0);
                    lane_mask_pipe_q[0] <= lane_mask_i;
                    word_addr_pipe_q[0] <= word_addr_i;
                    word_group_idx_pipe_q[0] <= word_group_idx_i;
                end
            end
        end
    end
endmodule

`default_nettype wire
