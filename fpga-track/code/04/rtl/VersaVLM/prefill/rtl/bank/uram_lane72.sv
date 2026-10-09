`default_nettype none

// Purpose:
//   Behavioral 72-bit true-dual-port memory lane for the Phase-1 SPM bank.
// Clock/reset:
//   All state uses clk_i. rst_i is synchronous, active high, and only clears
//   response-valid pipeline state. The memory array is intentionally not reset.
// Interface latency:
//   Accepted read requests return rsp_valid_o after READ_LATENCY cycles.
// Valid/ready:
//   Both ports are always ready. Writes are accepted when req_valid_i and
//   req_write_i are high. Writes do not generate responses.
// Supported modes:
//   Full-lane writes and optional byte-write enables.
// Unsupported/error behavior:
//   Same-address or same-port read/write semantics are not defined here; bank
//   owners must avoid those hazards at a higher level.
module uram_lane72 #(
    parameter int DEPTH_WORDS       = 4096,
    parameter int READ_LATENCY      = 4,
    parameter int DATA_W            = 72,
    parameter int BYTE_W            = 8,
    parameter bit ENABLE_BYTE_WRITE = 1'b1,
    localparam int ADDR_W           = $clog2(DEPTH_WORDS),
    localparam int BYTE_COUNT       = (DATA_W + BYTE_W - 1) / BYTE_W
) (
    input  wire logic                    clk_i,
    input  wire logic                    rst_i,

    input  wire logic                    a_req_valid_i,
    output logic                    a_req_ready_o,
    input  wire logic                    a_req_write_i,
    input  wire logic [ADDR_W-1:0]       a_req_addr_i,
    input  wire logic [DATA_W-1:0]       a_req_wdata_i,
    input  wire logic [BYTE_COUNT-1:0]   a_req_wstrb_i,
    output logic                    a_rsp_valid_o,
    output logic [DATA_W-1:0]       a_rsp_rdata_o,

    input  wire logic                    b_req_valid_i,
    output logic                    b_req_ready_o,
    input  wire logic                    b_req_write_i,
    input  wire logic [ADDR_W-1:0]       b_req_addr_i,
    input  wire logic [DATA_W-1:0]       b_req_wdata_i,
    input  wire logic [BYTE_COUNT-1:0]   b_req_wstrb_i,
    output logic                    b_rsp_valid_o,
    output logic [DATA_W-1:0]       b_rsp_rdata_o
);

    localparam int LATENCY = (READ_LATENCY < 1) ? 1 : READ_LATENCY;

    (* ram_style = "ultra" *) logic [DATA_W-1:0] mem [0:DEPTH_WORDS-1];

    logic [LATENCY-1:0]              a_rsp_valid_pipe;
    logic [LATENCY-1:0]              b_rsp_valid_pipe;
    logic [LATENCY-1:0][DATA_W-1:0]  a_rsp_data_pipe;
    logic [LATENCY-1:0][DATA_W-1:0]  b_rsp_data_pipe;

    assign a_req_ready_o = 1'b1;
    assign b_req_ready_o = 1'b1;
    assign a_rsp_valid_o = a_rsp_valid_pipe[LATENCY-1];
    assign b_rsp_valid_o = b_rsp_valid_pipe[LATENCY-1];

`ifdef SYNTHESIS
    localparam int XPM_WE_W = ENABLE_BYTE_WRITE ? BYTE_COUNT : 1;

    logic [XPM_WE_W-1:0] a_xpm_we;
    logic [XPM_WE_W-1:0] b_xpm_we;

    generate
        if (ENABLE_BYTE_WRITE) begin : gen_xpm_byte_we
            assign a_xpm_we = (a_req_valid_i && a_req_write_i) ? a_req_wstrb_i : '0;
            assign b_xpm_we = (b_req_valid_i && b_req_write_i) ? b_req_wstrb_i : '0;
        end
        else begin : gen_xpm_word_we
            assign a_xpm_we = (a_req_valid_i && a_req_write_i) ? {XPM_WE_W{1'b1}} : '0;
            assign b_xpm_we = (b_req_valid_i && b_req_write_i) ? {XPM_WE_W{1'b1}} : '0;
        end
    endgenerate

    xpm_memory_tdpram #(
        .MEMORY_SIZE        (DEPTH_WORDS * DATA_W),
        .MEMORY_PRIMITIVE   ("ultra"),
        .CLOCKING_MODE      ("common_clock"),
        .ECC_MODE           ("no_ecc"),
        .MEMORY_INIT_FILE   ("none"),
        .MEMORY_INIT_PARAM  ("0"),
        .USE_MEM_INIT       (0),
        .MESSAGE_CONTROL    (0),
        .WRITE_DATA_WIDTH_A (DATA_W),
        .READ_DATA_WIDTH_A  (DATA_W),
        .BYTE_WRITE_WIDTH_A (ENABLE_BYTE_WRITE ? BYTE_W : DATA_W),
        .ADDR_WIDTH_A       (ADDR_W),
        .READ_RESET_VALUE_A ("0"),
        .READ_LATENCY_A     (LATENCY),
        .WRITE_MODE_A       ("no_change"),
        .RST_MODE_A         ("SYNC"),
        .WRITE_DATA_WIDTH_B (DATA_W),
        .READ_DATA_WIDTH_B  (DATA_W),
        .BYTE_WRITE_WIDTH_B (ENABLE_BYTE_WRITE ? BYTE_W : DATA_W),
        .ADDR_WIDTH_B       (ADDR_W),
        .READ_RESET_VALUE_B ("0"),
        .READ_LATENCY_B     (LATENCY),
        .WRITE_MODE_B       ("no_change"),
        .RST_MODE_B         ("SYNC")
    ) u_xpm_memory_tdpram (
        .sleep          (1'b0),
        .clka           (clk_i),
        .rsta           (rst_i),
        .ena            (a_req_valid_i),
        .regcea         (1'b1),
        .wea            (a_xpm_we),
        .addra          (a_req_addr_i),
        .dina           (a_req_wdata_i),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .douta          (a_rsp_rdata_o),
        .sbiterra       (),
        .dbiterra       (),
        .clkb           (clk_i),
        .rstb           (rst_i),
        .enb            (b_req_valid_i),
        .regceb         (1'b1),
        .web            (b_xpm_we),
        .addrb          (b_req_addr_i),
        .dinb           (b_req_wdata_i),
        .injectsbiterrb (1'b0),
        .injectdbiterrb (1'b0),
        .doutb          (b_rsp_rdata_o),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    always_ff @(posedge clk_i) begin : xpm_valid_ff
        if (rst_i) begin
            a_rsp_valid_pipe <= '0;
            b_rsp_valid_pipe <= '0;
        end
        else begin
            a_rsp_valid_pipe[0] <= a_req_valid_i && !a_req_write_i;
            b_rsp_valid_pipe[0] <= b_req_valid_i && !b_req_write_i;
            for (int stage_idx = 1; stage_idx < LATENCY; stage_idx++) begin
                a_rsp_valid_pipe[stage_idx] <= a_rsp_valid_pipe[stage_idx-1];
                b_rsp_valid_pipe[stage_idx] <= b_rsp_valid_pipe[stage_idx-1];
            end
        end
    end
`else
    assign a_rsp_rdata_o = a_rsp_data_pipe[LATENCY-1];
    assign b_rsp_rdata_o = b_rsp_data_pipe[LATENCY-1];

    always_ff @(posedge clk_i) begin : port_a_ff
        if (rst_i) begin
            a_rsp_valid_pipe <= '0;
            a_rsp_data_pipe  <= '0;
        end
        else begin
            a_rsp_valid_pipe[0] <= a_req_valid_i && !a_req_write_i;
            if (a_req_valid_i) begin
                a_rsp_data_pipe[0] <= mem[a_req_addr_i];
            end
            else begin
                a_rsp_data_pipe[0] <= '0;
            end

            if (a_req_valid_i && a_req_write_i) begin
                if (ENABLE_BYTE_WRITE) begin
                    for (int byte_idx = 0; byte_idx < BYTE_COUNT; byte_idx++) begin
                        if (a_req_wstrb_i[byte_idx]) begin
                            mem[a_req_addr_i][byte_idx*BYTE_W +: BYTE_W] <=
                                a_req_wdata_i[byte_idx*BYTE_W +: BYTE_W];
                        end
                    end
                end
                else begin
                    mem[a_req_addr_i] <= a_req_wdata_i;
                end
            end

            for (int stage_idx = 1; stage_idx < LATENCY; stage_idx++) begin
                a_rsp_valid_pipe[stage_idx] <= a_rsp_valid_pipe[stage_idx-1];
                a_rsp_data_pipe[stage_idx]  <= a_rsp_data_pipe[stage_idx-1];
            end
        end
    end

    always_ff @(posedge clk_i) begin : port_b_ff
        if (rst_i) begin
            b_rsp_valid_pipe <= '0;
            b_rsp_data_pipe  <= '0;
        end
        else begin
            b_rsp_valid_pipe[0] <= b_req_valid_i && !b_req_write_i;
            if (b_req_valid_i) begin
                b_rsp_data_pipe[0] <= mem[b_req_addr_i];
            end
            else begin
                b_rsp_data_pipe[0] <= '0;
            end

            if (b_req_valid_i && b_req_write_i) begin
                if (ENABLE_BYTE_WRITE) begin
                    for (int byte_idx = 0; byte_idx < BYTE_COUNT; byte_idx++) begin
                        if (b_req_wstrb_i[byte_idx]) begin
                            mem[b_req_addr_i][byte_idx*BYTE_W +: BYTE_W] <=
                                b_req_wdata_i[byte_idx*BYTE_W +: BYTE_W];
                        end
                    end
                end
                else begin
                    mem[b_req_addr_i] <= b_req_wdata_i;
                end
            end

            for (int stage_idx = 1; stage_idx < LATENCY; stage_idx++) begin
                b_rsp_valid_pipe[stage_idx] <= b_rsp_valid_pipe[stage_idx-1];
                b_rsp_data_pipe[stage_idx]  <= b_rsp_data_pipe[stage_idx-1];
            end
        end
    end
`endif

endmodule

`default_nettype wire
