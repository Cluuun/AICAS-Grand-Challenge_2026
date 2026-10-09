`ifndef STREAM_BUFFER_SV
`define STREAM_BUFFER_SV

module stream_buffer #(
    parameter int DATA_WIDTH = 16,
    parameter int USER_WIDTH = 16,
    parameter int DEPTH      = 4096,
    parameter string FIFO_MEMORY_TYPE = "block",

    localparam int COUNT_WIDTH = $clog2(DEPTH + 1),
    localparam int ENTRY_WIDTH = DATA_WIDTH + USER_WIDTH
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  clear_i,
    input  logic                  stats_clear_i,

    input  logic                  push_valid_i,
    input  logic [DATA_WIDTH-1:0] push_data_i,
    input  logic [USER_WIDTH-1:0] push_user_i,
    output logic                  push_ready_o,

    input  logic                  pop_en_i,
    output logic                  pop_valid_o,
    output logic [DATA_WIDTH-1:0] pop_data_o,
    output logic [USER_WIDTH-1:0] pop_user_o,

    output logic [COUNT_WIDTH-1:0] count_o,
    output logic                   empty_o,
    output logic                   full_o,
    output logic                   overflow_o,
    output logic                   underflow_o,
    output logic [COUNT_WIDTH-1:0] push_count_o,
    output logic [COUNT_WIDTH-1:0] pop_count_o
);

    typedef logic [ENTRY_WIDTH-1:0] stream_entry_t;

    localparam logic [COUNT_WIDTH-1:0] DEPTH_COUNT = COUNT_WIDTH'(DEPTH);

    logic [3:0] rst_pipe_q;
    logic fifo_rst_w;
    logic fifo_full_w;
    logic fifo_empty_w;
    logic wr_rst_busy_w;
    logic rd_rst_busy_w;
    logic push_fire_w;
    logic pop_fire_w;
    stream_entry_t fifo_din_w;
    stream_entry_t fifo_dout_w;
    logic [COUNT_WIDTH-1:0] count_q;
    logic [COUNT_WIDTH-1:0] push_count_q;
    logic [COUNT_WIDTH-1:0] pop_count_q;
    logic overflow_q;
    logic underflow_q;

    assign fifo_rst_w = |rst_pipe_q;

    assign fifo_din_w = {push_user_i, push_data_i};
    assign push_ready_o = !fifo_rst_w && !wr_rst_busy_w && !rd_rst_busy_w && !fifo_full_w;
    assign pop_valid_o  = !fifo_rst_w && !rd_rst_busy_w && !fifo_empty_w;
    assign push_fire_w  = push_valid_i && push_ready_o;
    assign pop_fire_w   = pop_en_i && pop_valid_o;

    assign pop_data_o = fifo_dout_w[DATA_WIDTH-1:0];
    assign pop_user_o = fifo_dout_w[DATA_WIDTH +: USER_WIDTH];
    assign count_o = count_q;
    assign empty_o = (count_q == '0);
    assign full_o = (count_q == DEPTH_COUNT);
    assign overflow_o = overflow_q;
    assign underflow_o = underflow_q;
    assign push_count_o = push_count_q;
    assign pop_count_o = pop_count_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rst_pipe_q <= 4'hf;
        end else if (clear_i) begin
            rst_pipe_q <= 4'hf;
        end else begin
            rst_pipe_q <= {1'b0, rst_pipe_q[3:1]};
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_q      <= '0;
            push_count_q <= '0;
            pop_count_q  <= '0;
            overflow_q   <= 1'b0;
            underflow_q  <= 1'b0;
        end else if (fifo_rst_w) begin
            count_q      <= '0;
            push_count_q <= '0;
            pop_count_q  <= '0;
            overflow_q   <= 1'b0;
            underflow_q  <= 1'b0;
        end else begin
            if (stats_clear_i) begin
                push_count_q <= push_fire_w ? COUNT_WIDTH'(1) : '0;
                pop_count_q  <= pop_fire_w ? COUNT_WIDTH'(1) : '0;
                overflow_q   <= push_valid_i && !push_ready_o;
                underflow_q  <= pop_en_i && !pop_valid_o;
            end else begin
                if (push_fire_w)
                    push_count_q <= push_count_q + COUNT_WIDTH'(1);
                if (pop_fire_w)
                    pop_count_q <= pop_count_q + COUNT_WIDTH'(1);
                overflow_q  <= push_valid_i && !push_ready_o;
                underflow_q <= pop_en_i && !pop_valid_o;
            end

            unique case ({push_fire_w, pop_fire_w})
                2'b10: count_q <= count_q + COUNT_WIDTH'(1);
                2'b01: count_q <= count_q - COUNT_WIDTH'(1);
                default: count_q <= count_q;
            endcase
        end
    end

    xpm_fifo_sync #(
        .FIFO_MEMORY_TYPE    (FIFO_MEMORY_TYPE),
        .ECC_MODE            ("no_ecc"),
        .SIM_ASSERT_CHK      (0),
        .FIFO_WRITE_DEPTH    (DEPTH),
        .WRITE_DATA_WIDTH    (ENTRY_WIDTH),
        .WR_DATA_COUNT_WIDTH (COUNT_WIDTH),
        .PROG_FULL_THRESH    (DEPTH - 4),
        .FULL_RESET_VALUE    (0),
        .USE_ADV_FEATURES    ("0000"),
        .READ_MODE           ("fwft"),
        .FIFO_READ_LATENCY   (1),
        .READ_DATA_WIDTH     (ENTRY_WIDTH),
        .RD_DATA_COUNT_WIDTH (COUNT_WIDTH),
        .PROG_EMPTY_THRESH   (4),
        .DOUT_RESET_VALUE    ("0"),
        .WAKEUP_TIME         (0)
    ) u_xpm_fifo (
        .sleep         (1'b0),
        .rst           (fifo_rst_w),
        .wr_clk        (clk),
        .wr_en         (push_fire_w),
        .din           (fifo_din_w),
        .full          (fifo_full_w),
        .prog_full     (),
        .wr_data_count (),
        .overflow      (),
        .wr_rst_busy   (wr_rst_busy_w),
        .almost_full   (),
        .wr_ack        (),
        .rd_en         (pop_fire_w),
        .dout          (fifo_dout_w),
        .empty         (fifo_empty_w),
        .prog_empty    (),
        .rd_data_count (),
        .underflow     (),
        .rd_rst_busy   (rd_rst_busy_w),
        .almost_empty  (),
        .data_valid    (),
        .injectsbiterr (1'b0),
        .injectdbiterr (1'b0),
        .sbiterr       (),
        .dbiterr       ()
    );

`ifndef SYNTHESIS
    initial begin
        if (DEPTH <= 16)
            $fatal(1, "stream_buffer DEPTH must be greater than 16 for XPM FIFO thresholds");
    end

    property p_no_push_when_full;
        @(posedge clk) disable iff (!rst_n) push_valid_i && full_o |=> overflow_o;
    endproperty
    assert property (p_no_push_when_full);

    property p_no_pop_when_empty;
        @(posedge clk) disable iff (!rst_n) pop_en_i && empty_o |=> underflow_o;
    endproperty
    assert property (p_no_pop_when_empty);
`endif

endmodule

`endif
