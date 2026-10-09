`default_nettype none

// Converts the Versa_P mvout stream into the existing DMA SPM-read data shape.
// The bridge is intentionally small: DMA read enable pops one 128-bit beat, and
// the Versa_P producer is backpressured when the local FIFO is full.
module dma_mvout_bridge #(
    parameter int DATA_WIDTH = 128,
    parameter int FIFO_DEPTH = 8,
    localparam int PTR_W     = (FIFO_DEPTH <= 1) ? 1 : $clog2(FIFO_DEPTH),
    localparam int COUNT_W   = $clog2(FIFO_DEPTH + 1)
) (
    input  wire logic                  clk_i,
    input  wire logic                  rst_i,
    input  wire logic                  clear_error_i,

    input  wire logic                  core_mvout_valid_i,
    output logic                       core_mvout_ready_o,
    input  wire logic [DATA_WIDTH-1:0] core_mvout_data_i,

    input  wire logic                  dma_spm_rd_en_i,
    input  wire logic                  dma_consume_i,
    output logic [DATA_WIDTH-1:0]      dma_spm_dout_o,

    output logic                       fifo_empty_o,
    output logic                       fifo_full_o,
    output logic [COUNT_W-1:0]         fifo_level_o,
    output logic                       underflow_sticky_o
);
    logic [DATA_WIDTH-1:0] fifo_mem [FIFO_DEPTH];
    logic [PTR_W-1:0] wr_ptr_q;
    logic [PTR_W-1:0] rd_ptr_q;
    logic [COUNT_W-1:0] fifo_count_q;

    logic fifo_empty;
    logic fifo_full;
    logic bypass_take;
    logic fifo_pop;
    logic fifo_push;

    function automatic logic [PTR_W-1:0] ptr_next(input logic [PTR_W-1:0] ptr);
        if (ptr == PTR_W'(FIFO_DEPTH - 1)) begin
            ptr_next = '0;
        end
        else begin
            ptr_next = ptr + PTR_W'(1);
        end
    endfunction

    assign fifo_empty = (fifo_count_q == '0);
    assign fifo_full  = (fifo_count_q == COUNT_W'(FIFO_DEPTH));

    assign bypass_take = dma_consume_i && fifo_empty && core_mvout_valid_i;
    assign fifo_pop    = dma_consume_i && !fifo_empty;
    assign fifo_push   = core_mvout_valid_i && core_mvout_ready_o && !bypass_take;

    assign core_mvout_ready_o = core_mvout_valid_i && (bypass_take || !fifo_full || fifo_pop);
    assign dma_spm_dout_o       = fifo_empty ? (core_mvout_valid_i ? core_mvout_data_i : '0) :
                                               fifo_mem[rd_ptr_q];
    assign fifo_empty_o         = fifo_empty;
    assign fifo_full_o          = fifo_full;
    assign fifo_level_o         = fifo_count_q;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            wr_ptr_q           <= '0;
            rd_ptr_q           <= '0;
            fifo_count_q       <= '0;
            underflow_sticky_o <= 1'b0;
        end
        else begin
            if (clear_error_i) begin
                underflow_sticky_o <= 1'b0;
            end

            if (dma_consume_i && fifo_empty && !core_mvout_valid_i) begin
                underflow_sticky_o <= 1'b1;
            end

            if (fifo_push) begin
                fifo_mem[wr_ptr_q] <= core_mvout_data_i;
                wr_ptr_q <= ptr_next(wr_ptr_q);
            end

            if (fifo_pop) begin
                rd_ptr_q <= ptr_next(rd_ptr_q);
            end

            unique case ({fifo_push, fifo_pop})
                2'b10: fifo_count_q <= fifo_count_q + COUNT_W'(1);
                2'b01: fifo_count_q <= fifo_count_q - COUNT_W'(1);
                default: fifo_count_q <= fifo_count_q;
            endcase
        end
    end

    wire unused_dma_spm_rd_en = dma_spm_rd_en_i;
endmodule

`default_nettype wire
