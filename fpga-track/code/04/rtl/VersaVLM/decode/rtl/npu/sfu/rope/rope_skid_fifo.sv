// -----------------------------------------------------------------------------
// rope_skid_fifo.sv — Synchronous FIFO with AXI4-Stream-style handshakes.
//
// A clean, reusable elastic buffer that absorbs bursty producers and stalling
// consumers around the rope_engine PE bank.
//
//     producer ──valid──▶┌────────────┐──valid──▶ consumer
//     producer ◀─ready───┤ skid_fifo  │◀──ready── consumer
//     producer ──data───▶└────────────┘──data────▶ consumer
//
// Contract:
//   * A beat is transferred IFF (valid && ready) on that cycle.
//   * `i_ready` drops when the FIFO is full.
//   * `o_valid` asserts when the FIFO is non-empty.
//   * When `rst_n` deasserts, the FIFO returns to empty in one cycle.
//
// Synthesis notes:
//   * DEPTH rounded up to power-of-two internally for a clean wrap counter.
//   * Uses distributed RAM for small depths (<=32); BRAM for larger via
//     synthesis inference.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module rope_skid_fifo #(
    parameter int WIDTH = 145,                           // payload bits
    parameter int DEPTH = 32                             // number of entries (power of two)
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // Producer side (slave)
    input  wire                 i_valid,
    output wire                 i_ready,
    input  wire [WIDTH-1:0]     i_data,

    // Consumer side (master)
    output wire                 o_valid,
    input  wire                 o_ready,
    output wire [WIDTH-1:0]     o_data,

    // Flags (optional debug)
    output wire                 full,
    output wire                 empty
);

    // ----- Sanity: require power-of-two depth for simple wrap -----
    initial begin
        if ((DEPTH & (DEPTH - 1)) != 0)
            $fatal(1, "rope_skid_fifo DEPTH must be a power of two, got %0d", DEPTH);
    end

    localparam int PTR_W = $clog2(DEPTH);

    // ----- Storage -----
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    // ----- Pointers: one extra bit to disambiguate full vs empty -----
    reg [PTR_W:0] wr_ptr;
    reg [PTR_W:0] rd_ptr;

    wire [PTR_W-1:0] wr_idx = wr_ptr[PTR_W-1:0];
    wire [PTR_W-1:0] rd_idx = rd_ptr[PTR_W-1:0];

    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[PTR_W]     != rd_ptr[PTR_W]) &&
                   (wr_ptr[PTR_W-1:0] == rd_ptr[PTR_W-1:0]);

    // ----- Handshakes -----
    assign i_ready = ~full;
    assign o_valid = ~empty;
    wire   do_wr   = i_valid & i_ready;
    wire   do_rd   = o_valid & o_ready;

    // ----- Write port -----
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (do_wr) begin
            mem[wr_idx] <= i_data;
            wr_ptr      <= wr_ptr + 1'b1;
        end
    end

    // ----- Read port (registered output for clean timing) -----
    reg [WIDTH-1:0] o_data_r;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rd_ptr   <= '0;
            o_data_r <= '0;
        end else if (do_rd) begin
            o_data_r <= mem[(rd_ptr + 1'b1) & {PTR_W{1'b1}}];  // lookahead for 1-cycle read latency
            rd_ptr   <= rd_ptr + 1'b1;
        end else if (o_valid && !do_rd) begin
            // Keep output register stable when not popping
            o_data_r <= mem[rd_idx];
        end
    end

    // Simpler, equivalent combinational read (what most FPGA flows infer cleanly):
    // Use combinational read so that o_data follows rd_ptr without a 1-cycle skew;
    // this matches AXI4-Stream "data valid with valid" semantics exactly.
    assign o_data = mem[rd_idx];

endmodule

`default_nettype wire
