// pe_array_axis.v --- AXI-Stream wrapper around pe_array_32x32.
//
// Streams: A/W are 256-bit, psum is 256-bit, ctrl is 16-bit.
//   s_axis_a    256b   one beat = 32 activation int8s for one cycle
//   s_axis_w    256b   one beat = 32 weight int8s for one cycle
//   s_axis_ctrl 16b    sideband: {last, commit, accumulate_en, clear}
//                      bit0 = clear, bit1 = accumulate_en (en),
//                      bit2 = commit final 32-K group beat and start drain,
//                      bit3 = final tile marker propagated as TLAST
//   m_axis_psum 256b   8 x int32 per beat, 4 beats per row,
//                      4 groups x 32 rows x 4 quarters = 512 beats per drain
//
// Protocol:
//   * Compute phase: drive a/w/ctrl in lockstep. Each accepted (a,w,ctrl) beat
//     with ctrl[1]=1 latches one MAC step. ctrl[0]=1 zeros the array.
//   * Drain phase: final compute beat carries ctrl[2]=1. The wrapper then
//     autonomously emits the row segments; no extra ctrl beats are required.
//     Rows are emitted one group at a time. Compute is held while a just-committed
//     group is being snapshotted, then non-commit compute may overlap the active
//     snapshot drain. The next commit is backpressured until the current group
//     drain completes.
//
// Backpressure: m_axis_psum.tready stalls autonomous drain.
`timescale 1ns/1ps
module pe_array_axis (
    input  wire             ap_clk,
    input  wire             ap_rst_n,

    // a stream (32 int8 activations / cycle)
    input  wire [255:0]     s_axis_a_tdata,
    input  wire             s_axis_a_tvalid,
    output wire             s_axis_a_tready,

    // w stream (32 int8 weights / cycle)
    input  wire [255:0]     s_axis_w_tdata,
    input  wire             s_axis_w_tvalid,
    output wire             s_axis_w_tready,

    // ctrl sideband
    input  wire [15:0]      s_axis_ctrl_tdata,
    input  wire             s_axis_ctrl_tvalid,
    output wire             s_axis_ctrl_tready,

    // psum output stream (8 int32 / beat)
    output wire [255:0]     m_axis_psum_tdata,
    output wire             m_axis_psum_tvalid,
    input  wire             m_axis_psum_tready,
    output wire             m_axis_psum_tlast
);
    // ----- Decode ctrl -----
    wire ctrl_clear  = s_axis_ctrl_tdata[0];
    wire ctrl_en     = s_axis_ctrl_tdata[1];
    wire ctrl_commit = s_axis_ctrl_tdata[2];
    wire ctrl_last   = s_axis_ctrl_tdata[3];

    reg       pending_valid;
    reg       pending_last;
    reg       pending_delay;
    reg [1:0] commit_group_idx;

    reg       drain_phase_active;
    reg       drain_last_reg;
    reg [4:0] drain_row_idx;
    reg [1:0] drain_half_idx;

    wire compute_can_accept = !pending_valid && !(ctrl_commit & drain_phase_active);
    wire compute_req = s_axis_a_tvalid & s_axis_w_tvalid & s_axis_ctrl_tvalid & ctrl_en;
    wire compute_fire = compute_req & compute_can_accept;
    wire commit_fire = compute_fire & ctrl_commit;

    wire start_drain = pending_valid & ~pending_delay & ~drain_phase_active;
    wire drain_fire = drain_phase_active & m_axis_psum_tready;
    wire final_drain_beat = (drain_row_idx == 5'd31) & (drain_half_idx == 2'd3);
    wire row_done = (drain_half_idx == 2'd3);

    assign s_axis_a_tready =
        s_axis_w_tvalid & s_axis_ctrl_tvalid & ctrl_en & compute_can_accept;
    assign s_axis_w_tready =
        s_axis_a_tvalid & s_axis_ctrl_tvalid & ctrl_en & compute_can_accept;
    assign s_axis_ctrl_tready =
        ctrl_en & s_axis_a_tvalid & s_axis_w_tvalid & compute_can_accept;

    // ----- Drive PE array -----
    wire [1023:0] row0;

    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            pending_valid     <= 1'b0;
            pending_last      <= 1'b0;
            pending_delay     <= 1'b0;
            commit_group_idx  <= 2'd0;
            drain_phase_active<= 1'b0;
            drain_last_reg    <= 1'b0;
            drain_row_idx     <= 5'd0;
            drain_half_idx    <= 2'd0;
        end else begin
            if (commit_fire) begin
                pending_valid  <= 1'b1;
                pending_last   <= ctrl_last & (commit_group_idx == 2'd3);
                pending_delay  <= 1'b1;
                commit_group_idx <= commit_group_idx + 2'd1;
            end else if (pending_valid && pending_delay) begin
                pending_delay <= 1'b0;
            end else if (start_drain) begin
                drain_phase_active <= 1'b1;
                drain_last_reg     <= pending_last;
                drain_row_idx      <= 5'd0;
                drain_half_idx     <= 2'd0;
                pending_valid      <= 1'b0;
            end

            if (drain_fire) begin
                if (final_drain_beat) begin
                    drain_phase_active <= 1'b0;
                    drain_row_idx      <= 5'd0;
                    drain_half_idx     <= 2'd0;
                end else if (row_done) begin
                    drain_row_idx <= drain_row_idx + 5'd1;
                    drain_half_idx <= 2'd0;
                end else begin
                    drain_half_idx <= drain_half_idx + 2'd1;
                end
            end
        end
    end

    pe_array_32x32 u_array (
        .clk         (ap_clk),
        .rst_n       (ap_rst_n),
        .clear       (compute_fire & ctrl_clear),
        .en          (compute_fire & ctrl_en),
        .a_row_in    (s_axis_a_tdata),
        .w_col_in    (s_axis_w_tdata),
        .drain_load  (start_drain),
        .drain_active(drain_phase_active),
        .drain_shift (drain_fire & row_done),
        .drain_row0  (row0)
    );

    assign m_axis_psum_tdata  =
        (drain_half_idx == 2'd0) ? row0[255:0] :
        (drain_half_idx == 2'd1) ? row0[511:256] :
        (drain_half_idx == 2'd2) ? row0[767:512] :
                                   row0[1023:768];
    assign m_axis_psum_tvalid = drain_phase_active;
    assign m_axis_psum_tlast  = drain_phase_active &
                                drain_last_reg & final_drain_beat;

endmodule
