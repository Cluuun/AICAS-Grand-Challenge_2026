// pe_array_32x32.v — 32×32 Weight-Broadcast INT8 MAC Array.
// Activation broadcast per-row, weight broadcast per-column.
// All PEs share global en/clear. Accumulator outputs exposed as flat bus.
module pe_array_32x32 (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire        clear,
    input  wire [255:0] a_row,   // 32 × 8-bit activations (one per row)
    input  wire [255:0] w_col,   // 32 × 8-bit weights (one per column)
    output wire [32767:0] acc_flat // 32×32 × 32-bit = 32768 bits
);

    genvar m, n;
    generate
        for (m = 0; m < 32; m = m + 1) begin : ROW
            for (n = 0; n < 32; n = n + 1) begin : COL
                pe_mac_i8 u_pe (
                    .clk(clk),
                    .rst(rst),
                    .en(en),
                    .clear(clear),
                    .a_in(a_row[m*8 +: 8]),
                    .w_in(w_col[n*8 +: 8]),
                    .acc_out(acc_flat[(m*32+n)*32 +: 32])
                );
            end
        end
    endgenerate

endmodule
