`default_nettype none

// Purpose:
//   Phase-1 256-bit SPM bank wrapper built from four depth slices and four
//   72-bit lanes per slice. Each logical word stores 256 data bits; lane spare
//   bits are written as zero and ignored on read in Phase 1.
// Clock/reset:
//   All state uses clk_i. rst_i is synchronous, active high. Memory contents
//   are not reset.
// Interface latency:
//   Accepted reads on either port return rsp_valid_o after READ_LATENCY cycles.
// Valid/ready:
//   c_req_ready_o and d_req_ready_o are always high. Writes do not generate
//   responses. The wrapper does not arbitrate between clients.
// Supported modes:
//   256-bit aligned word access, optional byte write enables for C/O lane masks.
// Unsupported/error behavior:
//   The wrapper exposes same-address read/write conflict flags but does not
//   resolve the conflict. Phase-level ownership must prevent the hazard.
//
// Address example for DEPTH_WORDS=16384 and DEPTH_SLICES=4:
//   word_addr[13:12] selects one of four depth slices.
//   word_addr[11:0]  selects the local address inside that slice.
module uram256_bank #(
    parameter int DATA_W              = 256,
    parameter int DEPTH_WORDS         = 16384,
    parameter int READ_LATENCY        = 4,
    parameter int URAM_LANE_W         = 72,
    parameter int URAM_DATA_W         = 64,
    parameter int URAM_SPARE_W        = 8,
    parameter int URAM_LANES_PER_WORD = 4,
    parameter int DEPTH_SLICES        = 4,
    parameter bit ENABLE_BYTE_WRITE   = 1'b0,
    parameter bit USE_SPARE_BITS      = 1'b0,
    localparam int ADDR_W             = $clog2(DEPTH_WORDS),
    localparam int WORD_BYTES         = DATA_W / 8,
    localparam int SLICE_DEPTH_WORDS  = DEPTH_WORDS / DEPTH_SLICES,
    localparam int SLICE_ADDR_W       = $clog2(SLICE_DEPTH_WORDS),
    localparam int SLICE_SEL_W        = $clog2(DEPTH_SLICES)
) (
    input  wire logic                  clk_i,
    input  wire logic                  rst_i,

    input  wire logic                  c_req_valid_i,
    output logic                  c_req_ready_o,
    input  wire logic                  c_req_write_i,
    input  wire logic [ADDR_W-1:0]     c_req_word_addr_i,
    input  wire logic [DATA_W-1:0]     c_req_wdata_i,
    input  wire logic [WORD_BYTES-1:0] c_req_wstrb_i,
    output logic                  c_rsp_valid_o,
    output logic [DATA_W-1:0]     c_rsp_rdata_o,

    input  wire logic                  d_req_valid_i,
    output logic                  d_req_ready_o,
    input  wire logic                  d_req_write_i,
    input  wire logic [ADDR_W-1:0]     d_req_word_addr_i,
    input  wire logic [DATA_W-1:0]     d_req_wdata_i,
    input  wire logic [WORD_BYTES-1:0] d_req_wstrb_i,
    output logic                  d_rsp_valid_o,
    output logic [DATA_W-1:0]     d_rsp_rdata_o,

    output logic                  same_addr_read_write_conflict_o
);

    localparam int LANE_BYTE_COUNT = URAM_LANE_W / 8;

    logic [SLICE_SEL_W-1:0] c_slice_sel;
    logic [SLICE_SEL_W-1:0] d_slice_sel;
    logic [SLICE_ADDR_W-1:0] c_slice_addr;
    logic [SLICE_ADDR_W-1:0] d_slice_addr;

    logic [DEPTH_SLICES-1:0] c_slice_rsp_valid;
    logic [DEPTH_SLICES-1:0] d_slice_rsp_valid;
    logic [DEPTH_SLICES-1:0][DATA_W-1:0] c_slice_rsp_data;
    logic [DEPTH_SLICES-1:0][DATA_W-1:0] d_slice_rsp_data;

    assign c_req_ready_o = 1'b1;
    assign d_req_ready_o = 1'b1;

    assign c_slice_sel  = c_req_word_addr_i[SLICE_ADDR_W +: SLICE_SEL_W];
    assign d_slice_sel  = d_req_word_addr_i[SLICE_ADDR_W +: SLICE_SEL_W];
    assign c_slice_addr = c_req_word_addr_i[SLICE_ADDR_W-1:0];
    assign d_slice_addr = d_req_word_addr_i[SLICE_ADDR_W-1:0];

    // Conflict is reported only when one side reads and the other side writes
    // the same logical 256-bit word in the same cycle.
    assign same_addr_read_write_conflict_o =
        c_req_valid_i && d_req_valid_i &&
        (c_req_word_addr_i == d_req_word_addr_i) &&
        (c_req_write_i != d_req_write_i);

    always_comb begin : rsp_mux_comb
        c_rsp_valid_o = 1'b0;
        d_rsp_valid_o = 1'b0;
        c_rsp_rdata_o = '0;
        d_rsp_rdata_o = '0;
        for (int slice_idx = 0; slice_idx < DEPTH_SLICES; slice_idx++) begin
            c_rsp_valid_o |= c_slice_rsp_valid[slice_idx];
            d_rsp_valid_o |= d_slice_rsp_valid[slice_idx];
            if (c_slice_rsp_valid[slice_idx]) begin
                c_rsp_rdata_o = c_slice_rsp_data[slice_idx];
            end
            if (d_slice_rsp_valid[slice_idx]) begin
                d_rsp_rdata_o = d_slice_rsp_data[slice_idx];
            end
        end
    end

    generate
        genvar slice_idx;
        genvar lane_idx;

        for (slice_idx = 0; slice_idx < DEPTH_SLICES; slice_idx++) begin : gen_depth_slice
            localparam logic [SLICE_SEL_W-1:0] SLICE_ID = slice_idx;

            logic [URAM_LANES_PER_WORD-1:0]                  c_lane_rsp_valid;
            logic [URAM_LANES_PER_WORD-1:0]                  d_lane_rsp_valid;
            logic [URAM_LANES_PER_WORD-1:0][URAM_LANE_W-1:0] c_lane_rsp_data;
            logic [URAM_LANES_PER_WORD-1:0][URAM_LANE_W-1:0] d_lane_rsp_data;

            assign c_slice_rsp_valid[slice_idx] = |c_lane_rsp_valid;
            assign d_slice_rsp_valid[slice_idx] = |d_lane_rsp_valid;

            for (lane_idx = 0; lane_idx < URAM_LANES_PER_WORD; lane_idx++) begin : gen_lane
                logic [URAM_LANE_W-1:0] c_lane_wdata;
                logic [URAM_LANE_W-1:0] d_lane_wdata;
                logic [LANE_BYTE_COUNT-1:0] c_lane_wstrb;
                logic [LANE_BYTE_COUNT-1:0] d_lane_wstrb;

                assign c_lane_wdata = {
                    {URAM_SPARE_W{1'b0}},
                    c_req_wdata_i[lane_idx*URAM_DATA_W +: URAM_DATA_W]
                };
                assign d_lane_wdata = {
                    {URAM_SPARE_W{1'b0}},
                    d_req_wdata_i[lane_idx*URAM_DATA_W +: URAM_DATA_W]
                };
                assign c_lane_wstrb = {
                    {(URAM_SPARE_W/8){USE_SPARE_BITS}},
                    c_req_wstrb_i[lane_idx*(URAM_DATA_W/8) +: (URAM_DATA_W/8)]
                };
                assign d_lane_wstrb = {
                    {(URAM_SPARE_W/8){USE_SPARE_BITS}},
                    d_req_wstrb_i[lane_idx*(URAM_DATA_W/8) +: (URAM_DATA_W/8)]
                };

                assign c_slice_rsp_data[slice_idx][lane_idx*URAM_DATA_W +: URAM_DATA_W] =
                    c_lane_rsp_data[lane_idx][URAM_DATA_W-1:0];
                assign d_slice_rsp_data[slice_idx][lane_idx*URAM_DATA_W +: URAM_DATA_W] =
                    d_lane_rsp_data[lane_idx][URAM_DATA_W-1:0];

                uram_lane72 #(
                    .DEPTH_WORDS       (SLICE_DEPTH_WORDS),
                    .READ_LATENCY      (READ_LATENCY),
                    .DATA_W            (URAM_LANE_W),
                    .ENABLE_BYTE_WRITE (ENABLE_BYTE_WRITE)
                ) u_uram_lane72 (
                    .clk_i             (clk_i),
                    .rst_i             (rst_i),

                    .a_req_valid_i     (c_req_valid_i && (c_slice_sel == SLICE_ID)),
                    .a_req_ready_o     (),
                    .a_req_write_i     (c_req_write_i),
                    .a_req_addr_i      (c_slice_addr),
                    .a_req_wdata_i     (c_lane_wdata),
                    .a_req_wstrb_i     (c_lane_wstrb),
                    .a_rsp_valid_o     (c_lane_rsp_valid[lane_idx]),
                    .a_rsp_rdata_o     (c_lane_rsp_data[lane_idx]),

                    .b_req_valid_i     (d_req_valid_i && (d_slice_sel == SLICE_ID)),
                    .b_req_ready_o     (),
                    .b_req_write_i     (d_req_write_i),
                    .b_req_addr_i      (d_slice_addr),
                    .b_req_wdata_i     (d_lane_wdata),
                    .b_req_wstrb_i     (d_lane_wstrb),
                    .b_rsp_valid_o     (d_lane_rsp_valid[lane_idx]),
                    .b_rsp_rdata_o     (d_lane_rsp_data[lane_idx])
                );
            end
        end
    endgenerate

endmodule

`default_nettype wire
