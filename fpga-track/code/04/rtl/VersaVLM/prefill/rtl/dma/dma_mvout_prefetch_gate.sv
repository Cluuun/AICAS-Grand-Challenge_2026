`default_nettype none

// Delays the DMA_O mvout request until the Versa_P mvout stream has prefetched
// a small number of 128-bit beats into the bridge FIFO. Non-OC DMA requests pass
// through unchanged.
module dma_mvout_prefetch_gate #(
    parameter int DMA_NUM        = 3,
    parameter int OC_DMA_IDX     = 2,
    parameter int FIFO_DEPTH     = 8,
    parameter int PREFILL_LEVEL  = FIFO_DEPTH,
    localparam int FIFO_COUNT_W  = $clog2(FIFO_DEPTH + 1)
) (
    input  wire logic                         clk_i,
    input  wire logic                         rst_i,
    input  wire logic                         clear_i,

    input  wire logic [DMA_NUM-1:0]           dma_mvout_req_en_i,
    input  wire logic [FIFO_COUNT_W-1:0]      oc_prefetch_target_i,
    input  wire logic [FIFO_COUNT_W-1:0]      oc_fifo_level_i,

    output logic [DMA_NUM-1:0]                dma_mvout_req_en_o,
    output logic                              oc_pending_o
);
    logic pending_q;
    logic [FIFO_COUNT_W-1:0] target_level_q;
    logic release_oc;

    assign release_oc = pending_q && (oc_fifo_level_i >= target_level_q);
    assign oc_pending_o = pending_q;

    always_comb begin
        dma_mvout_req_en_o = dma_mvout_req_en_i;
        dma_mvout_req_en_o[OC_DMA_IDX] = release_oc;
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            pending_q      <= 1'b0;
            target_level_q <= FIFO_COUNT_W'(1);
        end
        else begin
            if (clear_i) begin
                pending_q      <= 1'b0;
                target_level_q <= FIFO_COUNT_W'(1);
            end
            else begin
                if (dma_mvout_req_en_i[OC_DMA_IDX]) begin
                    pending_q      <= 1'b1;
                    target_level_q <= oc_prefetch_target_i;
                end
                if (release_oc) begin
                    pending_q <= 1'b0;
                end
            end
        end
    end
endmodule

`default_nettype wire
