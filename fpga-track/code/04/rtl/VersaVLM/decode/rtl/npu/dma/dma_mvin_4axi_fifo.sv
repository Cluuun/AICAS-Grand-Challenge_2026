`ifndef DMA_MVIN_4AXI_FIFO_V
`define DMA_MVIN_4AXI_FIFO_V

module dma_mvin_4axi_fifo #(
    parameter int SPM_FPGA_SRAM = 0,
    parameter int DATA_WIDTH    = 128,
    parameter int DEPTH         = 16,
    parameter int BANKS         = 4,
    parameter RAM_STYLE         = "ultra"
) (
    input  logic                                   clk,
    input  logic                                   rst_n,
    input  logic                                   push,
    input  logic [DATA_WIDTH-1:0]                  push_data,
    output logic                                   full,
    output logic                                   row_ready,
    input  logic                                   pop_row,
    output logic [BANKS-1:0][DATA_WIDTH-1:0]       pop_row_data,
    output logic                                   pop_row_valid,
    output logic [$clog2(DEPTH + 1)-1:0]           count
);

localparam int CNT_WIDTH   = $clog2(DEPTH + 1);
localparam int ROWS        = DEPTH / BANKS;
localparam int PTR_WIDTH   = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
localparam int ROW_W       = (ROWS <= 1) ? 1 : $clog2(ROWS);
localparam int BANK_SEL_W  = (BANKS <= 1) ? 1 : $clog2(BANKS);

logic [PTR_WIDTH-1:0]                          wr_ptr_q;
logic [ROW_W-1:0]                              rd_row_ptr_q;
logic [CNT_WIDTH-1:0]                          count_q;
logic [BANKS-1:0][ROWS-1:0][DATA_WIDTH-1:0]    bank_mem;
logic [BANKS-1:0][DATA_WIDTH-1:0]              fpga_row_data;
logic [BANKS-1:0][DATA_WIDTH-1:0]              sim_row_data_q;
logic                                          pop_row_valid_q;

logic                                          push_fire;
logic                                          pop_fire;
logic [BANK_SEL_W-1:0]                         wr_bank_sel;
logic [ROW_W-1:0]                              wr_row_addr;

assign push_fire = push && !full;
assign pop_fire = pop_row && row_ready;
assign full = (count_q == CNT_WIDTH'(DEPTH));
assign row_ready = (count_q >= CNT_WIDTH'(BANKS));
assign count = count_q;
assign pop_row_valid = pop_row_valid_q;
assign wr_bank_sel = BANK_SEL_W'(wr_ptr_q % BANKS);
assign wr_row_addr = ROW_W'(wr_ptr_q / BANKS);

generate
    if (SPM_FPGA_SRAM == 1) begin : gen_fpga_ram
        genvar bank;
        for (bank = 0; bank < BANKS; bank++) begin : gen_bank
            wire [DATA_WIDTH-1:0] unused_dout_a;
            sram_fpga #(
                .DATA_WIDTH (DATA_WIDTH),
                .ADDR_WIDTH (ROW_W),
                .RAM_STYLE  (RAM_STYLE),
                .PORT_B_WRITE (0),
                .BYTE_WRITE_ENABLE (0)
            ) u_sram_fpga (
                .clk    (clk),
                .we_a   (push_fire && (wr_bank_sel == BANK_SEL_W'(bank))),
                .addr_a (wr_row_addr),
                .din_a  (push_data),
                .wr_mask_a ({DATA_WIDTH/8{1'b1}}),
                .dout_a (unused_dout_a),
                .we_b   (1'b0),
                .addr_b (rd_row_ptr_q),
                .din_b  ('0),
                .wr_mask_b ({DATA_WIDTH/8{1'b1}}),
                .dout_b (fpga_row_data[bank])
            );
        end

        assign pop_row_data = fpga_row_data;
    end else begin : gen_sim_ram
        assign pop_row_data = sim_row_data_q;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                sim_row_data_q <= '0;
            end else begin
                if (push_fire) begin
                    bank_mem[wr_bank_sel][wr_row_addr] <= push_data;
                end

                if (pop_fire) begin
                    for (int bank = 0; bank < BANKS; bank++) begin
                        sim_row_data_q[bank] <= bank_mem[bank][rd_row_ptr_q];
                    end
                end
            end
        end
    end
endgenerate

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_ptr_q <= '0;
        rd_row_ptr_q <= '0;
        count_q <= '0;
        pop_row_valid_q <= 1'b0;
    end else begin
        pop_row_valid_q <= pop_fire;

        if (push_fire) begin
            wr_ptr_q <= wr_ptr_q + 1'b1;
        end

        if (pop_fire) begin
            rd_row_ptr_q <= rd_row_ptr_q + 1'b1;
        end

        case ({push_fire, pop_fire})
            2'b10: count_q <= count_q + CNT_WIDTH'(1);
            2'b01: count_q <= count_q - CNT_WIDTH'(BANKS);
            2'b11: count_q <= count_q + CNT_WIDTH'(1) - CNT_WIDTH'(BANKS);
            default: count_q <= count_q;
        endcase
    end
end

`ifndef SYNTHESIS
initial begin
    if ((DEPTH % BANKS) != 0) begin
        $error("dma_mvin_4axi_fifo expects DEPTH (%0d) to be a multiple of BANKS (%0d)", DEPTH, BANKS);
    end
end
`endif

endmodule

`endif
