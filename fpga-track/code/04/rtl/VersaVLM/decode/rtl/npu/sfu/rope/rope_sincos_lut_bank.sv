`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// rope_sincos_lut_bank.sv — Banked pre-computed cos/sin lookup for RoPE.
//
// One instance per rotation PE. Bank `g` stores cos/sin for pair indices
// {g, g+PPC, g+2*PPC, ...} across all supported positions. This cuts per-PE
// BRAM footprint by PAIRS_PER_CYCLE compared to a full-depth LUT.
//
// Layout (matches python/rope_golden.py::gen_vectors banked output):
//     addr = position * LOCAL_PAIRS + local_pair_idx
//     where LOCAL_PAIRS = (HEAD_DIM / 2) / PAIRS_PER_CYCLE
//
// Hex-file naming: cos_lut_bank{g}.hex / sin_lut_bank{g}.hex (see BANK_ID).
// -----------------------------------------------------------------------------
`default_nettype none

module rope_sincos_lut_bank #(
    parameter int DATA_WIDTH  = 16,
    parameter int MAX_SEQ_LEN = 2048,
    parameter int LOCAL_PAIRS = 16,                    // N_PAIRS / PAIRS_PER_CYCLE
    parameter int ADDR_WIDTH  = $clog2(MAX_SEQ_LEN * LOCAL_PAIRS),
    parameter int READ_LATENCY = 2,
    parameter     COS_PATH    = "cos_lut_bank0.hex",
    parameter     SIN_PATH    = "sin_lut_bank0.hex"
) (
    input  wire                          clk,
    input  wire                          rd_en,
    input  wire [ADDR_WIDTH-1:0]         rd_addr,
    output reg  signed [DATA_WIDTH-1:0]  cos_out,
    output reg  signed [DATA_WIDTH-1:0]  sin_out
);

    localparam int DEPTH = MAX_SEQ_LEN * LOCAL_PAIRS;

    // KV260 UltraRAM cannot preserve non-zero $readmemh ROM initialization in
    // Vivado 2025.1, so these tables must remain in BRAM unless the contents
    // are generated or loaded at runtime.
    (* rom_style = "block", ram_style = "block" *) reg [DATA_WIDTH-1:0] cos_mem [0:DEPTH-1];
    (* rom_style = "block", ram_style = "block" *) reg [DATA_WIDTH-1:0] sin_mem [0:DEPTH-1];
    logic [DATA_WIDTH-1:0] cos_mem_q;
    logic [DATA_WIDTH-1:0] sin_mem_q;

    // Vivado synthesis requires the $readmemh path to be a literal or parameter
    // string at elaboration — so we pass the filename in from the parent
    // instead of composing it with $sformatf.
    initial begin
        $readmemh(COS_PATH, cos_mem);
        $readmemh(SIN_PATH, sin_mem);
    end

    always_ff @(posedge clk) begin
        if (rd_en) begin
            cos_mem_q <= cos_mem[rd_addr];
            sin_mem_q <= sin_mem[rd_addr];
        end
    end

    generate
        if (READ_LATENCY == 1) begin : gen_latency1
            always_comb begin
                cos_out = cos_mem_q;
                sin_out = sin_mem_q;
            end
        end else begin : gen_pipelined
            logic [READ_LATENCY-2:0][DATA_WIDTH-1:0] cos_pipe_q;
            logic [READ_LATENCY-2:0][DATA_WIDTH-1:0] sin_pipe_q;

            always_ff @(posedge clk) begin
                cos_pipe_q[0] <= cos_mem_q;
                sin_pipe_q[0] <= sin_mem_q;
                for (int stage = 1; stage < READ_LATENCY - 1; stage++) begin
                    cos_pipe_q[stage] <= cos_pipe_q[stage-1];
                    sin_pipe_q[stage] <= sin_pipe_q[stage-1];
                end
            end

            always_comb begin
                cos_out = cos_pipe_q[READ_LATENCY-2];
                sin_out = sin_pipe_q[READ_LATENCY-2];
            end
        end
    endgenerate

    initial begin
        if (READ_LATENCY < 1)
            $fatal(1, "rope_sincos_lut_bank READ_LATENCY must be >= 1");
    end

endmodule

`default_nettype wire
