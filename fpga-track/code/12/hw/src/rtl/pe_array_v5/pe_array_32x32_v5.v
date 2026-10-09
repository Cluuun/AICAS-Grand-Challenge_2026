// pe_array_32x32_v5.v --- 32x32 weight-broadcast int8 MAC array.
//
// Topology:
//   * a_row[255:0]  = 32 activation bytes, one per row, broadcast across columns.
//   * w_col[255:0]  = 32 weight bytes,     one per column, broadcast across rows.
//   * Each PE (m,n) computes acc[m,n] += a_row[m] * w_col[n].
//
// Fanout relief:
//   * Two local pipeline stages replicate a_row / w_col into 8x8 PE leaves.
//   * Each registered data byte sees fan-out = 8 instead of 32.
//   * Latency: 2 broadcast registers + 1 acc register = 3 cycles per (a,w) pair.
//
// Drain:
//   * The wrapper snapshots the current accumulators into a row shift chain.
//     The shift chain is isolated from live accumulators after drain_load, so
//     the next compute group can overlap the previous group's drain.
//   * read_row_data[1023:0] = {col31_acc, col30_acc, ..., col0_acc}.
//   * One 128-K tile drains as 32 rows; the AXIS wrapper emits four 256b
//     quarter-row beats per row.
`timescale 1ns/1ps
module pe_array_32x32_v5 (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             clear,         // zero all PE accumulators (sync)
    input  wire             en,            // accumulate cycle
    input  wire [255:0]     a_row_in,      // 32 x int8
    input  wire [255:0]     w_col_in,      // 32 x int8
    input  wire             drain_load,    // snapshot selected bank before drain
    input  wire             drain_active,  // hold snapshot while valid
    input  wire             drain_shift,   // shift one row after accepted beat
    output wire [1023:0]    drain_row0     // row 0 PSUM, 32 x int32
);
    // --- 8x8 leaf pipeline registers on broadcast buses ---
    // Stage 0 breaks the top-level 256-bit stream into four 64-bit stripes.
    // Stage 1 duplicates each stripe into the 4x4 PE leaves. Keeping these
    // stages separate prevents physopt from using one far-away leaf register as
    // the source for the other replicated leaves in the full BD route.
    (* keep = "true", max_fanout = 4 *) reg [63:0] a_stripe_q [0:3];
    (* keep = "true", max_fanout = 4 *) reg [63:0] w_stripe_q [0:3];
    (* keep = "true", max_fanout = 4 *) reg clear_quad_q [0:3];
    (* keep = "true", max_fanout = 4 *) reg en_quad_q [0:3];

    (* keep = "true", max_fanout = 8 *) reg [63:0] a_leaf_q [0:3][0:3];
    (* keep = "true", max_fanout = 8 *) reg [63:0] w_leaf_q [0:3][0:3];
    (* keep = "true", max_fanout = 64 *) reg clear_leaf_q [0:3][0:3];
    (* keep = "true", max_fanout = 64 *) reg en_leaf_q [0:3][0:3];
    integer leaf_r, leaf_c;

    always @(posedge clk) begin
        if (!rst_n) begin
            for (leaf_r = 0; leaf_r < 4; leaf_r = leaf_r + 1) begin
                a_stripe_q[leaf_r] <= 64'd0;
                w_stripe_q[leaf_r] <= 64'd0;
                clear_quad_q[leaf_r] <= 1'b0;
                en_quad_q[leaf_r] <= 1'b0;
            end
            for (leaf_r = 0; leaf_r < 4; leaf_r = leaf_r + 1)
                for (leaf_c = 0; leaf_c < 4; leaf_c = leaf_c + 1) begin
                    a_leaf_q[leaf_r][leaf_c] <= 64'd0;
                    w_leaf_q[leaf_c][leaf_r] <= 64'd0;
                    clear_leaf_q[leaf_r][leaf_c] <= 1'b0;
                    en_leaf_q[leaf_r][leaf_c] <= 1'b0;
                end
        end else begin
            for (leaf_r = 0; leaf_r < 4; leaf_r = leaf_r + 1) begin
                a_stripe_q[leaf_r] <= a_row_in[leaf_r*64 +: 64];
                w_stripe_q[leaf_r] <= w_col_in[leaf_r*64 +: 64];
                clear_quad_q[leaf_r] <= clear;
                en_quad_q[leaf_r] <= en;
            end
            for (leaf_r = 0; leaf_r < 4; leaf_r = leaf_r + 1)
                for (leaf_c = 0; leaf_c < 4; leaf_c = leaf_c + 1) begin
                    a_leaf_q[leaf_r][leaf_c] <= a_stripe_q[leaf_r];
                    w_leaf_q[leaf_c][leaf_r] <= w_stripe_q[leaf_c];
                    clear_leaf_q[leaf_r][leaf_c] <= clear_quad_q[leaf_r];
                    en_leaf_q[leaf_r][leaf_c] <= en_quad_q[leaf_r];
                end
        end
    end

    // --- 32x32 PE generate ---
    // PE accumulator wires live in a 2-D array.
    wire signed [31:0] acc_w [0:31][0:31];

    genvar m, n;
    generate
        for (m = 0; m < 32; m = m + 1) begin : ROW
            for (n = 0; n < 32; n = n + 1) begin : COL
                wire [7:0] a_byte = a_leaf_q[m/8][n/8][(m%8)*8 +: 8];
                wire [7:0] w_byte = w_leaf_q[n/8][m/8][(n%8)*8 +: 8];
                pe_mac_i8_v5 u_pe (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .clear   (clear_leaf_q[m/8][n/8]),
                    .en      (en_leaf_q[m/8][n/8]),
                    .a_in    (a_byte),
                    .w_in    (w_byte),
                    .acc_out (acc_w[m][n])
                );
            end
        end
    endgenerate

    (* keep = "true", max_fanout = 32 *) wire drain_load_col [0:31];
    (* keep = "true", max_fanout = 32 *) wire drain_shift_col [0:31];
    (* keep = "true", max_fanout = 32 *) wire drain_active_col [0:31];

    genvar k;
    generate
        for (k = 0; k < 32; k = k + 1) begin : CTRL_FANOUT
            assign drain_load_col[k] = drain_load;
            assign drain_shift_col[k] = drain_shift;
            assign drain_active_col[k] = drain_active;
        end
    endgenerate

    reg signed [31:0] psum_row_r [0:31][0:31];

    genvar pr, pc;
    generate
        for (pc = 0; pc < 32; pc = pc + 1) begin : PSUM_COL
            for (pr = 0; pr < 31; pr = pr + 1) begin : PSUM_ROW
                always @(posedge clk) begin
                    if (!rst_n) begin
                        psum_row_r[pr][pc] <= 32'sd0;
                    end else if (drain_load_col[pc]) begin
                        psum_row_r[pr][pc] <= acc_w[pr][pc];
                    end else if (drain_shift_col[pc]) begin
                        psum_row_r[pr][pc] <= psum_row_r[pr+1][pc];
                    end else if (drain_active_col[pc]) begin
                        // Hold the snapshot under output backpressure.
                    end else begin
                        psum_row_r[pr][pc] <= acc_w[pr][pc];
                    end
                end
            end
            always @(posedge clk) begin : PSUM_LAST_ROW
                if (!rst_n) begin
                    psum_row_r[31][pc] <= 32'sd0;
                end else if (drain_load_col[pc]) begin
                    psum_row_r[31][pc] <= acc_w[31][pc];
                end else if (drain_shift_col[pc]) begin
                    psum_row_r[31][pc] <= 32'sd0;
                end else if (drain_active_col[pc]) begin
                    // Hold the snapshot under output backpressure.
                end else begin
                    psum_row_r[31][pc] <= acc_w[31][pc];
                end
            end
        end

        for (k = 0; k < 32; k = k + 1) begin : OUT
            assign drain_row0[k*32 +: 32] = psum_row_r[0][k];
        end
    endgenerate

endmodule
