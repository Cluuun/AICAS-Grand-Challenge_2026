`default_nettype none

// Small synchronous output FIFO for DMA_OC mvout.
//
// Purpose:
//   Absorb DMA backpressure after C/O output conversion. This FIFO is used by
//   both FP32 and raw INT32 output modes so DMA_OC never reads the C/O bank
//   directly.
//
// Clock/reset:
//   Synchronous active-high reset. Only pointers, count, and outputs reset.
//
// Valid/ready:
//   Standard single-clock ready/valid. Data remains stable while valid is high
//   and ready is low.
module co_output_fifo #(
    parameter int DATA_W = 128,
    parameter int DEPTH  = 8,
    localparam int PTR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    localparam int CNT_W = $clog2(DEPTH + 1)
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,

    input  wire logic              push_valid_i,
    output logic              push_ready_o,
    input  wire logic [DATA_W-1:0] push_data_i,

    output logic              pop_valid_o,
    input  wire logic              pop_ready_i,
    output logic [DATA_W-1:0] pop_data_o,

    output logic              almost_full_o,
    output logic              overflow_error_o,
    output logic              underflow_error_o
);
    logic [DATA_W-1:0] mem [0:DEPTH-1];
    logic [PTR_W-1:0] wr_ptr_q;
    logic [PTR_W-1:0] rd_ptr_q;
    logic [CNT_W-1:0] count_q;
    localparam logic [CNT_W-1:0] DEPTH_COUNT = DEPTH;
    localparam logic [CNT_W-1:0] ALMOST_FULL_COUNT = (DEPTH - 1);

    wire push_fire = push_valid_i && push_ready_o;
    wire pop_fire  = pop_valid_o && pop_ready_i;

    assign push_ready_o  = (count_q < DEPTH_COUNT);
    assign pop_valid_o   = (count_q != '0);
    assign pop_data_o    = mem[rd_ptr_q];
    assign almost_full_o = (count_q >= ALMOST_FULL_COUNT);

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            wr_ptr_q          <= '0;
            rd_ptr_q          <= '0;
            count_q           <= '0;
            overflow_error_o  <= 1'b0;
            underflow_error_o <= 1'b0;
        end else begin
            if (push_fire) begin
                mem[wr_ptr_q] <= push_data_i;
                if (wr_ptr_q == PTR_W'(DEPTH - 1)) begin
                    wr_ptr_q <= '0;
                end else begin
                    wr_ptr_q <= wr_ptr_q + PTR_W'(1);
                end
            end

            if (pop_fire) begin
                if (rd_ptr_q == PTR_W'(DEPTH - 1)) begin
                    rd_ptr_q <= '0;
                end else begin
                    rd_ptr_q <= rd_ptr_q + PTR_W'(1);
                end
            end

            unique case ({push_fire, pop_fire})
                2'b10: count_q <= count_q + CNT_W'(1);
                2'b01: count_q <= count_q - CNT_W'(1);
                default: count_q <= count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
