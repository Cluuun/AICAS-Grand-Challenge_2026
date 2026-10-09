module sram_fpga #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 4,
    parameter RAM_STYLE  = "block",
    parameter READ_PIPELINE = 0,
    parameter PORT_B_WRITE = 1,
    parameter BYTE_WRITE_ENABLE = 1
) (
    input  wire                  clk,
    input  wire                  we_a,
    input  wire [ADDR_WIDTH-1:0] addr_a,
    input  wire [DATA_WIDTH-1:0] din_a,
    input  wire [DATA_WIDTH/8-1:0] wr_mask_a,
    output reg  [DATA_WIDTH-1:0] dout_a,
    input  wire                  we_b,
    input  wire [ADDR_WIDTH-1:0] addr_b,
    input  wire [DATA_WIDTH-1:0] din_b,
    input  wire [DATA_WIDTH/8-1:0] wr_mask_b,
    output reg  [DATA_WIDTH-1:0] dout_b
);

    localparam DEPTH = 1 << ADDR_WIDTH;

    generate
        if ((RAM_STYLE == "ultra") && (PORT_B_WRITE == 0) && (BYTE_WRITE_ENABLE == 0)) begin : gen_ultra_sdp_full
            (* ram_style = "ultra" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

            if (READ_PIPELINE > 0) begin : gen_pipe
                reg [DATA_WIDTH-1:0] dout_b_mem;
                reg [DATA_WIDTH-1:0] dout_b_pipe [0:READ_PIPELINE-1];
                integer pipe_idx;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b_mem = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                    for (pipe_idx = 0; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] = {DATA_WIDTH{1'b0}};
                    end
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        mem[addr_a] <= din_a;
                    end
                end

                always @(posedge clk) begin
                    dout_b_mem <= mem[addr_b];
                end

                always @(posedge clk) begin
                    dout_b_pipe[0] <= dout_b_mem;
                    for (pipe_idx = 1; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] <= dout_b_pipe[pipe_idx - 1];
                    end
                    dout_b <= dout_b_pipe[READ_PIPELINE - 1];
                end
            end else begin : gen_no_pipe
`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        mem[addr_a] <= din_a;
                    end
                    dout_b <= mem[addr_b];
                end
            end
        end else if ((RAM_STYLE == "block") && (PORT_B_WRITE == 0) && (BYTE_WRITE_ENABLE == 0)) begin : gen_block_sdp_full
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

            if (READ_PIPELINE > 0) begin : gen_pipe
                reg [DATA_WIDTH-1:0] dout_b_mem;
                reg [DATA_WIDTH-1:0] dout_b_pipe [0:READ_PIPELINE-1];
                integer pipe_idx;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b_mem = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                    for (pipe_idx = 0; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] = {DATA_WIDTH{1'b0}};
                    end
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        mem[addr_a] <= din_a;
                    end
                end

                always @(posedge clk) begin
                    dout_b_mem <= mem[addr_b];
                end

                always @(posedge clk) begin
                    dout_b_pipe[0] <= dout_b_mem;
                    for (pipe_idx = 1; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] <= dout_b_pipe[pipe_idx - 1];
                    end
                    dout_b <= dout_b_pipe[READ_PIPELINE - 1];
                end
            end else begin : gen_no_pipe
`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        mem[addr_a] <= din_a;
                    end
                    dout_b <= mem[addr_b];
                end
            end
        end else if ((RAM_STYLE == "ultra") && (PORT_B_WRITE == 0)) begin : gen_ultra_sdp
            (* ram_style = "ultra" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

            if (READ_PIPELINE > 0) begin : gen_pipe
                reg [DATA_WIDTH-1:0] dout_b_mem;
                reg [DATA_WIDTH-1:0] dout_b_pipe [0:READ_PIPELINE-1];
                integer pipe_idx;
                integer byte_idx_a;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b_mem = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                    for (pipe_idx = 0; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] = {DATA_WIDTH{1'b0}};
                    end
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        for (byte_idx_a = 0; byte_idx_a < DATA_WIDTH / 8; byte_idx_a = byte_idx_a + 1) begin
                            if (wr_mask_a[byte_idx_a]) begin
                                mem[addr_a][byte_idx_a * 8 +: 8] <= din_a[byte_idx_a * 8 +: 8];
                            end
                        end
                    end
                end

                always @(posedge clk) begin
                    dout_b_mem <= mem[addr_b];
                end

                always @(posedge clk) begin
                    dout_b_pipe[0] <= dout_b_mem;
                    for (pipe_idx = 1; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] <= dout_b_pipe[pipe_idx - 1];
                    end
                    dout_b <= dout_b_pipe[READ_PIPELINE - 1];
                end
            end else begin : gen_no_pipe
                integer byte_idx_a;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        for (byte_idx_a = 0; byte_idx_a < DATA_WIDTH / 8; byte_idx_a = byte_idx_a + 1) begin
                            if (wr_mask_a[byte_idx_a]) begin
                                mem[addr_a][byte_idx_a * 8 +: 8] <= din_a[byte_idx_a * 8 +: 8];
                            end
                        end
                    end
                    dout_b <= mem[addr_b];
                end
            end
        end else if ((RAM_STYLE == "block") && (PORT_B_WRITE == 0)) begin : gen_block_sdp
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

            if (READ_PIPELINE > 0) begin : gen_pipe
                reg [DATA_WIDTH-1:0] dout_b_mem;
                reg [DATA_WIDTH-1:0] dout_b_pipe [0:READ_PIPELINE-1];
                integer pipe_idx;
                integer byte_idx_a;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b_mem = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                    for (pipe_idx = 0; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] = {DATA_WIDTH{1'b0}};
                    end
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        for (byte_idx_a = 0; byte_idx_a < DATA_WIDTH / 8; byte_idx_a = byte_idx_a + 1) begin
                            if (wr_mask_a[byte_idx_a]) begin
                                mem[addr_a][byte_idx_a * 8 +: 8] <= din_a[byte_idx_a * 8 +: 8];
                            end
                        end
                    end
                end

                always @(posedge clk) begin
                    dout_b_mem <= mem[addr_b];
                end

                always @(posedge clk) begin
                    dout_b_pipe[0] <= dout_b_mem;
                    for (pipe_idx = 1; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_b_pipe[pipe_idx] <= dout_b_pipe[pipe_idx - 1];
                    end
                    dout_b <= dout_b_pipe[READ_PIPELINE - 1];
                end
            end else begin : gen_no_pipe
                integer byte_idx_a;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                end
`endif

                always @(posedge clk) begin
                    dout_a <= {DATA_WIDTH{1'b0}};
                    if (we_a) begin
                        for (byte_idx_a = 0; byte_idx_a < DATA_WIDTH / 8; byte_idx_a = byte_idx_a + 1) begin
                            if (wr_mask_a[byte_idx_a]) begin
                                mem[addr_a][byte_idx_a * 8 +: 8] <= din_a[byte_idx_a * 8 +: 8];
                            end
                        end
                    end
                    dout_b <= mem[addr_b];
                end
            end
        end else if (RAM_STYLE == "ultra") begin : gen_ultra
            (* ram_style = "ultra" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

            if (READ_PIPELINE > 0) begin : gen_pipe
                reg [DATA_WIDTH-1:0] dout_a_mem;
                reg [DATA_WIDTH-1:0] dout_b_mem;
                reg [DATA_WIDTH-1:0] dout_a_pipe [0:READ_PIPELINE-1];
                reg [DATA_WIDTH-1:0] dout_b_pipe [0:READ_PIPELINE-1];
                integer pipe_idx;
                integer byte_idx_a;
                integer byte_idx_b;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a_mem = {DATA_WIDTH{1'b0}};
                    dout_b_mem = {DATA_WIDTH{1'b0}};
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                    for (pipe_idx = 0; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_a_pipe[pipe_idx] = {DATA_WIDTH{1'b0}};
                        dout_b_pipe[pipe_idx] = {DATA_WIDTH{1'b0}};
                    end
                end
`endif

                always @(posedge clk) begin
                    if (we_a) begin
                        mem[addr_a] <= din_a;
                    end else begin
                        dout_a_mem <= mem[addr_a];
                    end
                end

                always @(posedge clk) begin
                    if (we_b) begin
                        mem[addr_b] <= din_b;
                    end else begin
                        dout_b_mem <= mem[addr_b];
                    end
                end

                always @(posedge clk) begin
                    dout_a_pipe[0] <= dout_a_mem;
                    dout_b_pipe[0] <= dout_b_mem;
                    for (pipe_idx = 1; pipe_idx < READ_PIPELINE; pipe_idx = pipe_idx + 1) begin
                        dout_a_pipe[pipe_idx] <= dout_a_pipe[pipe_idx - 1];
                        dout_b_pipe[pipe_idx] <= dout_b_pipe[pipe_idx - 1];
                    end
                    dout_a <= dout_a_pipe[READ_PIPELINE - 1];
                    dout_b <= dout_b_pipe[READ_PIPELINE - 1];
                end
            end else begin : gen_no_pipe
                integer byte_idx_a;
                integer byte_idx_b;

`ifdef INITIALIZE_MEMORY
                integer init_idx;
                initial begin
                    for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                        mem[init_idx] = {DATA_WIDTH{1'b0}};
                    end
                    dout_a = {DATA_WIDTH{1'b0}};
                    dout_b = {DATA_WIDTH{1'b0}};
                end
`endif

                always @(posedge clk) begin
                    if (we_a) begin
                        mem[addr_a] <= din_a;
                    end else begin
                        dout_a <= mem[addr_a];
                    end
                end

                always @(posedge clk) begin
                    if (we_b) begin
                        mem[addr_b] <= din_b;
                    end else begin
                        dout_b <= mem[addr_b];
                    end
                end
            end
        end else if (RAM_STYLE == "distributed") begin : gen_distributed
            (* ram_style = "distributed" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
            integer byte_idx_a;
            integer byte_idx_b;

`ifdef INITIALIZE_MEMORY
            integer init_idx;
            initial begin
                for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                    mem[init_idx] = {DATA_WIDTH{1'b0}};
                end
                dout_a = {DATA_WIDTH{1'b0}};
                dout_b = {DATA_WIDTH{1'b0}};
            end
`endif

            always @(posedge clk) begin
                if (we_a) begin
                    mem[addr_a] <= din_a;
                end else begin
                    dout_a <= mem[addr_a];
                end
            end

            always @(posedge clk) begin
                if (we_b) begin
                    mem[addr_b] <= din_b;
                end else begin
                    dout_b <= mem[addr_b];
                end
            end
        end else begin : gen_block
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
            integer byte_idx_a;
            integer byte_idx_b;

`ifdef INITIALIZE_MEMORY
            integer init_idx;
            initial begin
                for (init_idx = 0; init_idx < DEPTH; init_idx = init_idx + 1) begin
                    mem[init_idx] = {DATA_WIDTH{1'b0}};
                end
                dout_a = {DATA_WIDTH{1'b0}};
                dout_b = {DATA_WIDTH{1'b0}};
            end
`endif

            always @(posedge clk) begin
                if (we_a) begin
                    mem[addr_a] <= din_a;
                end else begin
                    dout_a <= mem[addr_a];
                end
            end

            always @(posedge clk) begin
                if (we_b) begin
                    mem[addr_b] <= din_b;
                end else begin
                    dout_b <= mem[addr_b];
                end
            end
        end
    endgenerate

endmodule
