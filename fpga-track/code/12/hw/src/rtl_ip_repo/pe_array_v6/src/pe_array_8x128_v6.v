// pe_array_8x128_v6.v --- 8x128 decode-friendly int8 MAC array.
//
// Geometry:
//   * a_row[63:0]    = 8 activation bytes, one per row.
//   * w_col0[511:0]  = 64 weight bytes for output columns 0..63.
//   * w_col1[511:0]  = 64 weight bytes for output columns 64..127.
//   * 8 x 128 = 1024 DSP MACs, matching the v5 DSP budget.
//
// Fanout:
//   * A bytes are replicated into 4-column leaves. The extra leaf registers
//     are cheaper than letting phys_opt create long, congested A-net replicas
//     across the full 128-column array.
//   * W bytes are split into two 512b domains, then registered once per output
//     column. Each registered byte fans out to 8 row MACs, which is cheaper than
//     replicating the full 128-column vector per row.
//
// Drain:
//   * Snapshot is 8 rows x 128 cols. The wrapper selects one 8-column segment
//     at a time, avoiding a 4096b row bus and the old row-shift network.
`timescale 1ns/1ps
module pe_array_8x128_v6 (
    input  wire             clk,
    input  wire             rst_n,
    input  wire [31:0]      clear_group,
    input  wire [31:0]      en_group,
    input  wire [63:0]      a_row_in,
    input  wire [511:0]     w_col0_in,
    input  wire [511:0]     w_col1_in,
    input  wire [127:0]     drain_load_col,
    input  wire [2:0]       drain_row_idx,
    input  wire [3:0]       drain_seg_idx,
    output wire [255:0]     drain_segment
);
    localparam integer M = 8;
    localparam integer N = 128;
    localparam integer COL_GROUPS = 32;
    localparam integer COLS_PER_GROUP = 4;

    // Do not mark these registers KEEP. The full KV260 design is congestion
    // limited, and Vivado needs freedom to duplicate/move the leaf registers
    // during place/phys_opt while respecting the requested fanout caps.
    (* max_fanout = 4  *) reg [7:0] a_leaf_q [0:M-1][0:COL_GROUPS-1];
    (* max_fanout = 8  *) reg [7:0] w_col_q  [0:N-1];
    (* max_fanout = 4  *) reg clear_leaf_q [0:M-1][0:COL_GROUPS-1];
    (* max_fanout = 4  *) reg en_leaf_q    [0:M-1][0:COL_GROUPS-1];

    integer rr, cc, gg;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (rr = 0; rr < M; rr = rr + 1) begin
                for (gg = 0; gg < COL_GROUPS; gg = gg + 1) begin
                    a_leaf_q[rr][gg] <= 8'd0;
                    clear_leaf_q[rr][gg] <= 1'b0;
                    en_leaf_q[rr][gg] <= 1'b0;
                end
            end
            for (cc = 0; cc < N; cc = cc + 1) begin
                w_col_q[cc] <= 8'd0;
            end
        end else begin
            for (rr = 0; rr < M; rr = rr + 1) begin
                for (gg = 0; gg < COL_GROUPS; gg = gg + 1) begin
                    a_leaf_q[rr][gg] <= a_row_in[rr*8 +: 8];
                    clear_leaf_q[rr][gg] <= clear_group[gg];
                    en_leaf_q[rr][gg] <= en_group[gg];
                end
            end
            for (cc = 0; cc < N; cc = cc + 1) begin
                if (cc < 64) begin
                    w_col_q[cc] <= w_col0_in[cc*8 +: 8];
                end else begin
                    w_col_q[cc] <= w_col1_in[(cc-64)*8 +: 8];
                end
            end
        end
    end

    wire signed [31:0] acc_w [0:M-1][0:N-1];
    genvar m, n;
    generate
        for (m = 0; m < M; m = m + 1) begin : ROW
            for (n = 0; n < N; n = n + 1) begin : COL
                localparam integer CG = n / COLS_PER_GROUP;
                pe_mac_i8_v6 u_pe (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .clear   (clear_leaf_q[m][CG]),
                    .en      (en_leaf_q[m][CG]),
                    .a_in    (a_leaf_q[m][CG]),
                    .w_in    (w_col_q[n]),
                    .acc_out (acc_w[m][n])
                );
            end
        end
    endgenerate

    // No reset is needed: the snapshot is only read after drain_load captures a
    // completed tile. Dropping reset and row-shift control avoids thousands of
    // high-fanout control routes in the full KV260 implementation.
    reg [4095:0] psum_snap_row_r [0:M-1];
    genvar pr, pc;
    generate
        for (pr = 0; pr < M; pr = pr + 1) begin : PSUM_ROW
            for (pc = 0; pc < N; pc = pc + 1) begin : PSUM_COL
                always @(posedge clk) begin
                    if (drain_load_col[pc]) begin
                        psum_snap_row_r[pr][pc*32 +: 32] <= acc_w[pr][pc];
                    end
                end
            end
        end

        wire [255:0] row_seg0 = psum_snap_row_r[0][drain_seg_idx*256 +: 256];
        wire [255:0] row_seg1 = psum_snap_row_r[1][drain_seg_idx*256 +: 256];
        wire [255:0] row_seg2 = psum_snap_row_r[2][drain_seg_idx*256 +: 256];
        wire [255:0] row_seg3 = psum_snap_row_r[3][drain_seg_idx*256 +: 256];
        wire [255:0] row_seg4 = psum_snap_row_r[4][drain_seg_idx*256 +: 256];
        wire [255:0] row_seg5 = psum_snap_row_r[5][drain_seg_idx*256 +: 256];
        wire [255:0] row_seg6 = psum_snap_row_r[6][drain_seg_idx*256 +: 256];
        wire [255:0] row_seg7 = psum_snap_row_r[7][drain_seg_idx*256 +: 256];
    endgenerate

    assign drain_segment =
        (drain_row_idx == 3'd0) ? row_seg0 :
        (drain_row_idx == 3'd1) ? row_seg1 :
        (drain_row_idx == 3'd2) ? row_seg2 :
        (drain_row_idx == 3'd3) ? row_seg3 :
        (drain_row_idx == 3'd4) ? row_seg4 :
        (drain_row_idx == 3'd5) ? row_seg5 :
        (drain_row_idx == 3'd6) ? row_seg6 : row_seg7;
endmodule
