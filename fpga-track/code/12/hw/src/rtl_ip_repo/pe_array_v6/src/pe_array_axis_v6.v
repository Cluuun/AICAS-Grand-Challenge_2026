// pe_array_axis_v6.v --- AXI-Stream wrapper around pe_array_8x128_v6.
//
// Streams:
//   s_axis_a    64b    one beat = 8 activation int8s for one cycle
//   s_axis_w0   512b   one beat = 64 weight int8s for columns 0..63
//   s_axis_w1   512b   one beat = 64 weight int8s for columns 64..127
//   s_axis_ctrl 16b    bit0 clear, bit1 en, bit2 commit, bit3 TLAST marker
//   m_axis_psum 256b   8 x int32 per beat, 16 beats per row, 8 rows
`timescale 1ns/1ps
module pe_array_axis_v6 (
    input  wire             ap_clk,
    input  wire             ap_rst_n,

    input  wire [63:0]      s_axis_a_tdata,
    input  wire             s_axis_a_tvalid,
    output wire             s_axis_a_tready,

    input  wire [511:0]     s_axis_w0_tdata,
    input  wire             s_axis_w0_tvalid,
    output wire             s_axis_w0_tready,

    input  wire [511:0]     s_axis_w1_tdata,
    input  wire             s_axis_w1_tvalid,
    output wire             s_axis_w1_tready,

    input  wire [15:0]      s_axis_ctrl_tdata,
    input  wire             s_axis_ctrl_tvalid,
    output wire             s_axis_ctrl_tready,

    output wire [255:0]     m_axis_psum_tdata,
    output wire             m_axis_psum_tvalid,
    input  wire             m_axis_psum_tready,
    output wire             m_axis_psum_tlast
);
    // The accepted AXIS compute beat is registered in this wrapper, registered
    // once more as a 4-column leaf in pe_array_8x128_v6, then passes through the
    // PE product pipeline before it reaches the accumulator. Keep the drain
    // snapshot four cycles after commit so the final product has definitely
    // been accumulated before psum capture starts.
    localparam [2:0] COMMIT_TO_DRAIN_DELAY = 3'd4;

    wire ctrl_clear  = s_axis_ctrl_tdata[0];
    wire ctrl_en     = s_axis_ctrl_tdata[1];
    wire ctrl_commit = s_axis_ctrl_tdata[2];
    wire ctrl_last   = s_axis_ctrl_tdata[3];

    reg       pending_valid;
    reg       pending_last;
    reg [2:0] pending_delay;
    reg       drain_start_pending;
    reg       drain_phase_active;
    reg       drain_last_reg;
    reg [2:0] drain_row_idx;
    reg [3:0] drain_seg_idx;
    reg [63:0]  array_a_q;
    reg [511:0] array_w0_q;
    reg [511:0] array_w1_q;
    // Drive PE control per 8-column group instead of as one scalar.  The
    // 8x128 array is route-limited in the full KV260 design; feeding scalar
    // clear/en into 128 leaf registers makes Vivado build long replicated
    // nets across the DSP columns.  Registering 32 column-group controls here
    // keeps the fanout local without changing the stream protocol.
    reg [31:0]  array_clear_group_q;
    reg [31:0]  array_en_group_q;
    // Same idea for drain snapshot. A scalar drain_load drives 8x128 snapshot
    // enables in the array and became a routing hotspot in the full design.
    // Registering one enable per output column adds one drain-start cycle but
    // limits each routed bit to the 8 row snapshots for that column.
    reg [127:0] array_drain_load_col_q;

    wire compute_can_accept = !pending_valid && !drain_start_pending && !(ctrl_commit & drain_phase_active);
    wire compute_req = s_axis_a_tvalid & s_axis_w0_tvalid & s_axis_w1_tvalid & s_axis_ctrl_tvalid & ctrl_en;
    wire compute_fire = compute_req & compute_can_accept;
    wire commit_fire = compute_fire & ctrl_commit;
    wire start_drain = pending_valid & (pending_delay == 3'd0) & ~drain_phase_active & ~drain_start_pending;
    wire drain_fire = drain_phase_active & m_axis_psum_tready;
    wire final_drain_beat = (drain_row_idx == 3'd7) & (drain_seg_idx == 4'd15);
    wire row_done = (drain_seg_idx == 4'd15);

    assign s_axis_a_tready =
        s_axis_w0_tvalid & s_axis_w1_tvalid & s_axis_ctrl_tvalid & ctrl_en & compute_can_accept;
    assign s_axis_w0_tready =
        s_axis_a_tvalid & s_axis_w1_tvalid & s_axis_ctrl_tvalid & ctrl_en & compute_can_accept;
    assign s_axis_w1_tready =
        s_axis_a_tvalid & s_axis_w0_tvalid & s_axis_ctrl_tvalid & ctrl_en & compute_can_accept;
    assign s_axis_ctrl_tready =
        ctrl_en & s_axis_a_tvalid & s_axis_w0_tvalid & s_axis_w1_tvalid & compute_can_accept;

    always @(posedge ap_clk) begin
        if (!ap_rst_n) begin
            pending_valid      <= 1'b0;
            pending_last       <= 1'b0;
            pending_delay      <= 3'd0;
            drain_start_pending <= 1'b0;
            drain_phase_active <= 1'b0;
            drain_last_reg     <= 1'b0;
            drain_row_idx      <= 3'd0;
            drain_seg_idx      <= 4'd0;
            array_a_q          <= 64'd0;
            array_w0_q         <= 512'd0;
            array_w1_q         <= 512'd0;
            array_clear_group_q <= 32'd0;
            array_en_group_q    <= 32'd0;
            array_drain_load_col_q <= 128'd0;
        end else begin
            array_en_group_q    <= {32{compute_fire}};
            array_clear_group_q <= {32{compute_fire & ctrl_clear}};
            array_drain_load_col_q <= {128{start_drain}};
            if (compute_fire) begin
                array_a_q  <= s_axis_a_tdata;
                array_w0_q <= s_axis_w0_tdata;
                array_w1_q <= s_axis_w1_tdata;
            end

            if (commit_fire) begin
                pending_valid <= 1'b1;
                pending_last  <= ctrl_last;
                pending_delay <= COMMIT_TO_DRAIN_DELAY;
            end else if (pending_valid && (pending_delay != 3'd0)) begin
                pending_delay <= pending_delay - 3'd1;
            end else if (start_drain) begin
                drain_start_pending <= 1'b1;
                drain_last_reg      <= pending_last;
                pending_valid       <= 1'b0;
            end else if (drain_start_pending) begin
                drain_start_pending <= 1'b0;
                drain_phase_active <= 1'b1;
                drain_row_idx      <= 3'd0;
                drain_seg_idx      <= 4'd0;
            end

            if (drain_fire) begin
                if (final_drain_beat) begin
                    drain_phase_active <= 1'b0;
                    drain_row_idx      <= 3'd0;
                    drain_seg_idx      <= 4'd0;
                end else if (row_done) begin
                    drain_row_idx <= drain_row_idx + 3'd1;
                    drain_seg_idx <= 4'd0;
                end else begin
                    drain_seg_idx <= drain_seg_idx + 4'd1;
                end
            end
        end
    end

    wire [255:0] psum_segment;
    pe_array_8x128_v6 u_array (
        .clk          (ap_clk),
        .rst_n        (ap_rst_n),
        .clear_group  (array_clear_group_q),
        .en_group     (array_en_group_q),
        .a_row_in     (array_a_q),
        .w_col0_in    (array_w0_q),
        .w_col1_in    (array_w1_q),
        .drain_load_col(array_drain_load_col_q),
        .drain_row_idx(drain_row_idx),
        .drain_seg_idx(drain_seg_idx),
        .drain_segment(psum_segment)
    );

    assign m_axis_psum_tdata  = psum_segment;
    assign m_axis_psum_tvalid = drain_phase_active;
    assign m_axis_psum_tlast  = drain_phase_active & drain_last_reg & final_drain_beat;
endmodule
