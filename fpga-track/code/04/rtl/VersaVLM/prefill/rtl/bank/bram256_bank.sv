`default_nettype none

// 256-bit block-RAM bank with one write port and one read port.
// The interface intentionally matches uram256_bank so C/O can use BRAM for
// the resource-search-selected O0 bank while keeping the existing datapath
// arbitration contract. The C port is the write side in the C/O datapath; D is
// the read side. D writes and C reads are intentionally unsupported.
module bram256_bank #(
    parameter int DATA_W            = 256,
    parameter int DEPTH_WORDS       = 16384,
    parameter int READ_LATENCY      = 2,
    parameter bit ENABLE_BYTE_WRITE = 1'b1,
    localparam int ADDR_W           = $clog2(DEPTH_WORDS),
    localparam int WORD_BYTES       = DATA_W / 8
) (
    input  wire logic                  clk_i,
    input  wire logic                  rst_i,

    input  wire logic                  c_req_valid_i,
    output logic                       c_req_ready_o,
    input  wire logic                  c_req_write_i,
    input  wire logic [ADDR_W-1:0]     c_req_word_addr_i,
    input  wire logic [DATA_W-1:0]     c_req_wdata_i,
    input  wire logic [WORD_BYTES-1:0] c_req_wstrb_i,
    output logic                       c_rsp_valid_o,
    output logic [DATA_W-1:0]          c_rsp_rdata_o,

    input  wire logic                  d_req_valid_i,
    output logic                       d_req_ready_o,
    input  wire logic                  d_req_write_i,
    input  wire logic [ADDR_W-1:0]     d_req_word_addr_i,
    input  wire logic [DATA_W-1:0]     d_req_wdata_i,
    input  wire logic [WORD_BYTES-1:0] d_req_wstrb_i,
    output logic                       d_rsp_valid_o,
    output logic [DATA_W-1:0]          d_rsp_rdata_o,

    output logic                       same_addr_read_write_conflict_o
);
    localparam int LATENCY = (READ_LATENCY < 1) ? 1 : READ_LATENCY;

    (* ram_style = "block" *) logic [DATA_W-1:0] mem [0:DEPTH_WORDS-1];

    logic [LATENCY-1:0] d_valid_pipe_q;
    logic [DATA_W-1:0] d_data_pipe_q [0:LATENCY-1];

    assign c_req_ready_o = 1'b1;
    assign d_req_ready_o = 1'b1;
    assign c_rsp_valid_o = 1'b0;
    assign d_rsp_valid_o = d_valid_pipe_q[LATENCY-1];
    assign c_rsp_rdata_o = '0;
    assign d_rsp_rdata_o = d_data_pipe_q[LATENCY-1];

    assign same_addr_read_write_conflict_o =
        c_req_valid_i && d_req_valid_i &&
        (c_req_word_addr_i == d_req_word_addr_i) &&
        c_req_write_i && !d_req_write_i;

    always_ff @(posedge clk_i) begin
        if (c_req_valid_i && c_req_write_i) begin
            if (ENABLE_BYTE_WRITE) begin
                for (int byte_idx = 0; byte_idx < WORD_BYTES; byte_idx++) begin
                    if (c_req_wstrb_i[byte_idx]) begin
                        mem[c_req_word_addr_i][byte_idx*8 +: 8] <=
                            c_req_wdata_i[byte_idx*8 +: 8];
                    end
                end
            end else begin
                mem[c_req_word_addr_i] <= c_req_wdata_i;
            end
        end

        if (rst_i) begin
            d_valid_pipe_q <= '0;
            for (int i = 0; i < LATENCY; i++) begin
                d_data_pipe_q[i] <= '0;
            end
        end else begin
            d_valid_pipe_q[0] <= d_req_valid_i && !d_req_write_i;
            d_data_pipe_q[0] <= mem[d_req_word_addr_i];
            for (int i = 1; i < LATENCY; i++) begin
                d_valid_pipe_q[i] <= d_valid_pipe_q[i-1];
                d_data_pipe_q[i] <= d_data_pipe_q[i-1];
            end
        end
    end

    wire unused_inputs = d_req_write_i ^ ^d_req_wdata_i ^ ^d_req_wstrb_i;
endmodule

`default_nettype wire
