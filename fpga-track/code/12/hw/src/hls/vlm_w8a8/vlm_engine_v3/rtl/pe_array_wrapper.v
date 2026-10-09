// pe_array_wrapper.v — HLS-compatible blackbox wrapper for pe_array_32x32.
// Exposes ap_ctrl_hs-like interface for Vitis HLS integration.
// HLS sees this as a function with ap_none ports + ap_ctrl_hs protocol.
module pe_array_wrapper (
    input  wire        ap_clk,
    input  wire        ap_rst,
    input  wire        ap_start,
    output wire        ap_done,
    output wire        ap_idle,
    output wire        ap_ready,
    input  wire        pe_en,
    input  wire        pe_clear,
    input  wire [255:0] a_row,
    input  wire [255:0] w_col,
    // Accumulator readout: row-by-row via mux
    input  wire [4:0]  rd_row_sel,
    input  wire [4:0]  rd_col_sel,
    output wire [31:0] rd_acc_val
);

    wire [32767:0] acc_flat;

    pe_array_32x32 u_array (
        .clk(ap_clk),
        .rst(ap_rst),
        .en(pe_en),
        .clear(pe_clear),
        .a_row(a_row),
        .w_col(w_col),
        .acc_flat(acc_flat)
    );

    // Mux for reading individual accumulator
    wire [9:0] rd_idx = {rd_row_sel, rd_col_sel};
    assign rd_acc_val = acc_flat[rd_idx * 32 +: 32];

    // Always ready (combinational array, no pipeline latency beyond 1 cycle)
    assign ap_done  = ap_start;
    assign ap_idle  = ~ap_start;
    assign ap_ready = ap_start;

endmodule
