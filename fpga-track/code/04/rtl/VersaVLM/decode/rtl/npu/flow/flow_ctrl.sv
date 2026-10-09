`ifndef FLOW_CTRL_SV
`define FLOW_CTRL_SV

module flow_ctrl (
    input  logic       clk,
    input  logic       rst_n,

    input  logic [1:0] cfg_flow_mode_i,
    input  logic       flow_req_en_i,
    input  logic       flow_has_work_i,

    input  logic       gemv_busy_i,
    input  logic       gemv_done_i,
    input  logic       writer_busy_i,
    input  logic       writer_done_i,

    output logic       gemv_req_en_o,
    output logic       writer_req_en_o,
    output logic       sfu_silu_en_o,
    output logic       writer_dst_act_o,
    output logic       flow_busy_o,
    output logic       flow_done_o
);

    localparam logic [1:0] FLOW_GEMV_TO_SPM      = 2'b00;
    localparam logic [1:0] FLOW_GEMV_SILU_TO_SPM = 2'b01;
    localparam logic [1:0] FLOW_GEMV_TO_ACT      = 2'b10;
    localparam logic [1:0] FLOW_GEMV_SILU_TO_ACT = 2'b11;

    logic active_q;
    logic need_writer_q;
    logic gemv_done_seen_q;
    logic writer_done_seen_q;
    logic done_hit_w;
    logic [1:0] flow_mode_q;
    logic [1:0] flow_mode_w;

    assign flow_mode_w      = active_q ? flow_mode_q : cfg_flow_mode_i;
    assign gemv_req_en_o    = flow_req_en_i && flow_has_work_i;
    assign writer_req_en_o  = flow_req_en_i && flow_has_work_i;
    assign sfu_silu_en_o    = (flow_mode_w == FLOW_GEMV_SILU_TO_SPM) ||
                              (flow_mode_w == FLOW_GEMV_SILU_TO_ACT);
    assign writer_dst_act_o = (flow_mode_w == FLOW_GEMV_TO_ACT) ||
                              (flow_mode_w == FLOW_GEMV_SILU_TO_ACT);
    assign flow_busy_o      = active_q || gemv_busy_i || writer_busy_i;
    assign done_hit_w       = active_q &&
                              (gemv_done_seen_q || gemv_done_i) &&
                              (!need_writer_q || writer_done_seen_q || writer_done_i);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_q           <= 1'b0;
            need_writer_q      <= 1'b0;
            gemv_done_seen_q   <= 1'b0;
            writer_done_seen_q <= 1'b0;
            flow_mode_q        <= FLOW_GEMV_TO_SPM;
            flow_done_o        <= 1'b0;
        end else begin
            flow_done_o <= 1'b0;

            if (flow_req_en_i) begin
                active_q           <= flow_has_work_i;
                need_writer_q      <= flow_has_work_i;
                gemv_done_seen_q   <= 1'b0;
                writer_done_seen_q <= !flow_has_work_i;
                flow_mode_q        <= cfg_flow_mode_i;
                flow_done_o        <= !flow_has_work_i;
            end else begin
                if (gemv_done_i) begin
                    gemv_done_seen_q <= 1'b1;
                end
                if (writer_done_i) begin
                    writer_done_seen_q <= 1'b1;
                end
                if (done_hit_w) begin
                    flow_done_o        <= 1'b1;
                    active_q           <= 1'b0;
                    need_writer_q      <= 1'b0;
                    gemv_done_seen_q   <= 1'b0;
                    writer_done_seen_q <= 1'b0;
                end
            end
        end
    end

`ifndef SYNTHESIS
    property p_no_req_while_active;
        @(posedge clk) disable iff (!rst_n) active_q |-> !flow_req_en_i;
    endproperty
    assert property (p_no_req_while_active);
`endif

endmodule

`endif
