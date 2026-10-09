// pe_array_32x32.v --- 32x32 weight-broadcast int8 MAC array.
//
// Topology:
//   * a_row[255:0]  = 32 activation bytes, one per row, broadcast across columns.
//   * w_col[255:0]  = 32 weight bytes,     one per column, broadcast across rows.
//   * Each PE (m,n) computes acc[m,n] += a_row[m] * w_col[n].
//
// Fanout relief:
//   * One pipeline stage on a_row / w_col splitting array into 4 quadrants.
//   * Each quadrant sees fan-out = 16 instead of 32 -> closes 300 MHz.
//   * Latency: 1 broadcast register + 1 acc register = 2 cycles per (a,w) pair.
//
// Drain:
//   * The wrapper snapshots the current accumulators into a row shift chain.
//     The shift chain is isolated from live accumulators after drain_load, so
//     the next compute group can overlap the previous group's drain.
//   * read_row_data[1023:0] = {col31_acc, col30_acc, ..., col0_acc}.
//   * One 32-K group drains in 32 cycles; four commits produce a 128-beat K-tile.
`timescale 1ns/1ps
module pe_array_32x32 (
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
    // --- Quadrant pipeline registers on broadcast buses ---
    reg [127:0] a_lo_q, a_hi_q;   // a_row split low/high 16 bytes
    reg [127:0] w_lo_q, w_hi_q;
    reg         clear_q, en_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            a_lo_q  <= 128'd0; a_hi_q <= 128'd0;
            w_lo_q  <= 128'd0; w_hi_q <= 128'd0;
            clear_q <= 1'b0;   en_q   <= 1'b0;
        end else begin
            a_lo_q  <= a_row_in[127:0];
            a_hi_q  <= a_row_in[255:128];
            w_lo_q  <= w_col_in[127:0];
            w_hi_q  <= w_col_in[255:128];
            clear_q <= clear;
            en_q    <= en;
        end
    end

    // --- 32x32 PE generate ---
    // PE accumulator wires live in a 2-D array.
    wire signed [31:0] acc_w [0:31][0:31];

    genvar m, n;
    generate
        for (m = 0; m < 32; m = m + 1) begin : ROW
            wire [7:0] a_byte = (m < 16) ? a_lo_q[m*8 +: 8]
                                          : a_hi_q[(m-16)*8 +: 8];
            for (n = 0; n < 32; n = n + 1) begin : COL
                wire [7:0] w_byte = (n < 16) ? w_lo_q[n*8 +: 8]
                                              : w_hi_q[(n-16)*8 +: 8];
                pe_mac_i8 u_pe (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .clear   (clear_q),
                    .en      (en_q),
                    .a_in    (a_byte),
                    .w_in    (w_byte),
                    .acc_out (acc_w[m][n])
                );
            end
        end
    endgenerate

    reg signed [31:0] psum_row_r [0:31][0:31];
    integer i, j;

    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < 32; i = i + 1)
                for (j = 0; j < 32; j = j + 1)
                    psum_row_r[i][j] <= 32'sd0;
        end else if (drain_load) begin
            for (i = 0; i < 32; i = i + 1)
                for (j = 0; j < 32; j = j + 1)
                    psum_row_r[i][j] <= acc_w[i][j];
        end else if (drain_shift) begin
            for (i = 0; i < 31; i = i + 1)
                for (j = 0; j < 32; j = j + 1)
                    psum_row_r[i][j] <= psum_row_r[i+1][j];
            for (j = 0; j < 32; j = j + 1)
                psum_row_r[31][j] <= 32'sd0;
        end else if (drain_active) begin
            // Hold the snapshot under output backpressure.
        end else begin
            for (i = 0; i < 32; i = i + 1)
                for (j = 0; j < 32; j = j + 1)
                    psum_row_r[i][j] <= acc_w[i][j];
        end
    end

    genvar k;
    generate
        for (k = 0; k < 32; k = k + 1) begin : OUT
            assign drain_row0[k*32 +: 32] = psum_row_r[0][k];
        end
    endgenerate

endmodule
