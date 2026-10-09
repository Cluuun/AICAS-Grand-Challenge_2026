`ifndef DMA_MVOUT_SPM_1AXI_V
`define DMA_MVOUT_SPM_1AXI_V

module DMA_MVOUT_S2MM_WRITER_1AXI #(
    parameter int AXI_ID_WIDTH        = 4,
    parameter int AXI_ADDR_WIDTH      = 32,
    parameter int AXI_DATA_WIDTH      = 128,
    parameter int AXI_MAX_BURST_BEATS = 16,
    parameter int AXI_WR_OUTSTANDING  = 16,
    parameter int BEAT_FIFO_DEPTH     = 64,
    parameter int DESC_FIFO_DEPTH     = 32,
    localparam int AXI_STRB_W         = AXI_DATA_WIDTH / 8,
    localparam int BURST_CNT_W        = $clog2(AXI_MAX_BURST_BEATS + 1),
    localparam int OUTSTANDING_CNT_W  = $clog2(AXI_WR_OUTSTANDING + 1),
    localparam int DESC_PTR_W         = (DESC_FIFO_DEPTH <= 2) ? 1 : $clog2(DESC_FIFO_DEPTH),
    localparam int DESC_CNT_W         = $clog2(DESC_FIFO_DEPTH + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         start,
    output logic                         busy,
    output logic                         done,
    output logic                         error_sticky,

    input  logic                         s_valid,
    output logic                         s_ready,
    input  logic [AXI_DATA_WIDTH-1:0]    s_data,
    input  logic [AXI_STRB_W-1:0]        s_keep,
    input  logic [AXI_ADDR_WIDTH-1:0]    s_addr,
    input  logic                         s_last,

    output logic [AXI_ID_WIDTH-1:0]      m_axi_awid,
    output logic [AXI_ADDR_WIDTH-1:0]    m_axi_awaddr,
    output logic [7:0]                   m_axi_awlen,
    output logic [2:0]                   m_axi_awsize,
    output logic [1:0]                   m_axi_awburst,
    output logic                         m_axi_awvalid,
    input  logic                         m_axi_awready,
    output logic [AXI_DATA_WIDTH-1:0]    m_axi_wdata,
    output logic [AXI_STRB_W-1:0]        m_axi_wstrb,
    output logic                         m_axi_wlast,
    output logic                         m_axi_wvalid,
    input  logic                         m_axi_wready,
    input  logic [1:0]                   m_axi_bresp,
    input  logic                         m_axi_bvalid,
    output logic                         m_axi_bready
);
    localparam int BEAT_BYTES = AXI_STRB_W;
    localparam logic [2:0] AXI_SIZE = $clog2(AXI_STRB_W);
    localparam int BEAT_FIFO_WIDTH = AXI_DATA_WIDTH + AXI_STRB_W;

    logic [BEAT_FIFO_WIDTH-1:0] beat_fifo_din_w;
    logic [BEAT_FIFO_WIDTH-1:0] beat_fifo_dout_w;
    logic                       beat_fifo_full_w;
    logic                       beat_fifo_empty_w;

    logic [AXI_ADDR_WIDTH-1:0] desc_addr_q [DESC_FIFO_DEPTH];
    logic [BURST_CNT_W-1:0]    desc_len_q [DESC_FIFO_DEPTH];
    logic [DESC_PTR_W-1:0]     desc_wr_ptr_q;
    logic [DESC_PTR_W-1:0]     desc_rd_ptr_q;
    logic [DESC_CNT_W-1:0]     desc_count_q;

    logic [BURST_CNT_W-1:0]    issued_len_q [DESC_FIFO_DEPTH];
    logic [DESC_PTR_W-1:0]     issued_wr_ptr_q;
    logic [DESC_PTR_W-1:0]     issued_rd_ptr_q;
    logic [DESC_CNT_W-1:0]     issued_count_q;

    logic                      open_valid_q;
    logic [AXI_ADDR_WIDTH-1:0] open_addr_q;
    logic [AXI_ADDR_WIDTH-1:0] open_next_addr_q;
    logic [BURST_CNT_W-1:0]    open_len_q;
    logic                      last_seen_q;

    logic [BURST_CNT_W-1:0]    w_beats_left_q;
    logic [OUTSTANDING_CNT_W-1:0] outstanding_q;

    logic accept_w;
    logic push_beat_w;
    logic break_before_push_w;
    logic close_after_push_w;
    logic [BURST_CNT_W-1:0] new_open_len_w;
    logic aw_fire_w;
    logic w_fire_w;
    logic b_fire_w;
    logic [BURST_CNT_W-1:0] w_current_len_w;

    function automatic logic [DESC_PTR_W-1:0] desc_ptr_inc(input logic [DESC_PTR_W-1:0] ptr);
        if (ptr == DESC_PTR_W'(DESC_FIFO_DEPTH - 1)) begin
            desc_ptr_inc = '0;
        end else begin
            desc_ptr_inc = ptr + DESC_PTR_W'(1);
        end
    endfunction

    assign s_ready = busy &&
                     !beat_fifo_full_w &&
                     (desc_count_q <= DESC_CNT_W'(DESC_FIFO_DEPTH - 3));
    assign accept_w = s_valid && s_ready;
    assign push_beat_w = accept_w && (s_keep != '0);
    assign break_before_push_w = push_beat_w && open_valid_q &&
                                 ((open_len_q == BURST_CNT_W'(AXI_MAX_BURST_BEATS)) ||
                                  (s_addr != open_next_addr_q) ||
                                  (s_addr[11:0] == 12'h000));
    assign new_open_len_w = break_before_push_w ? BURST_CNT_W'(1) :
                            (open_valid_q ? (open_len_q + BURST_CNT_W'(1)) : BURST_CNT_W'(1));
    assign close_after_push_w = push_beat_w &&
                                (s_last || (new_open_len_w == BURST_CNT_W'(AXI_MAX_BURST_BEATS)));

    assign m_axi_awid    = '0;
    assign m_axi_awaddr  = desc_addr_q[desc_rd_ptr_q];
    assign m_axi_awlen   = 8'(desc_len_q[desc_rd_ptr_q] - BURST_CNT_W'(1));
    assign m_axi_awsize  = AXI_SIZE;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = busy && (desc_count_q != '0) &&
                           (issued_count_q < DESC_CNT_W'(DESC_FIFO_DEPTH)) &&
                           (outstanding_q < OUTSTANDING_CNT_W'(AXI_WR_OUTSTANDING));

    assign w_current_len_w = (w_beats_left_q != '0) ? w_beats_left_q : issued_len_q[issued_rd_ptr_q];
    assign m_axi_wdata  = beat_fifo_dout_w[AXI_DATA_WIDTH-1:0];
    assign m_axi_wstrb  = beat_fifo_dout_w[AXI_DATA_WIDTH +: AXI_STRB_W];
    assign m_axi_wlast  = (w_current_len_w == BURST_CNT_W'(1));
    assign m_axi_wvalid = busy && !beat_fifo_empty_w &&
                          ((w_beats_left_q != '0) || (issued_count_q != '0));
    assign m_axi_bready = 1'b1;

    assign beat_fifo_din_w = {s_keep, s_data};

    xpm_fifo_sync #(
        .CASCADE_HEIGHT      (0),
        .DOUT_RESET_VALUE    ("0"),
        .ECC_MODE            ("no_ecc"),
        .FIFO_MEMORY_TYPE    ("block"),
        .FIFO_READ_LATENCY   (0),
        .FIFO_WRITE_DEPTH    (BEAT_FIFO_DEPTH),
        .FULL_RESET_VALUE    (0),
        .PROG_EMPTY_THRESH   (10),
        .PROG_FULL_THRESH    (10),
        .RD_DATA_COUNT_WIDTH (1),
        .READ_DATA_WIDTH     (BEAT_FIFO_WIDTH),
        .READ_MODE           ("fwft"),
        .SIM_ASSERT_CHK      (0),
        .USE_ADV_FEATURES    ("0000"),
        .WAKEUP_TIME         (0),
        .WRITE_DATA_WIDTH    (BEAT_FIFO_WIDTH),
        .WR_DATA_COUNT_WIDTH (1)
    ) u_beat_fifo (
        .almost_empty  (),
        .almost_full   (),
        .data_valid    (),
        .dbiterr       (),
        .dout          (beat_fifo_dout_w),
        .empty         (beat_fifo_empty_w),
        .full          (beat_fifo_full_w),
        .overflow      (),
        .prog_empty    (),
        .prog_full     (),
        .rd_data_count (),
        .rd_rst_busy   (),
        .sbiterr       (),
        .underflow     (),
        .wr_ack        (),
        .wr_data_count (),
        .wr_rst_busy   (),
        .din           (beat_fifo_din_w),
        .injectdbiterr (1'b0),
        .injectsbiterr (1'b0),
        .rd_en         (w_fire_w),
        .rst           (!rst_n),
        .sleep         (1'b0),
        .wr_clk        (clk),
        .wr_en         (push_beat_w)
    );

    assign aw_fire_w = m_axi_awvalid && m_axi_awready;
    assign w_fire_w = m_axi_wvalid && m_axi_wready;
    assign b_fire_w = m_axi_bvalid && m_axi_bready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            error_sticky <= 1'b0;
            desc_wr_ptr_q <= '0;
            desc_rd_ptr_q <= '0;
            desc_count_q <= '0;
            issued_wr_ptr_q <= '0;
            issued_rd_ptr_q <= '0;
            issued_count_q <= '0;
            open_valid_q <= 1'b0;
            open_addr_q <= '0;
            open_next_addr_q <= '0;
            open_len_q <= '0;
            last_seen_q <= 1'b0;
            w_beats_left_q <= '0;
            outstanding_q <= '0;
        end else begin
            done <= 1'b0;

            begin
                logic [DESC_PTR_W-1:0] desc_wr_next;
                logic [DESC_PTR_W-1:0] desc_rd_next;
                logic [DESC_CNT_W-1:0] desc_count_next;
                logic [DESC_PTR_W-1:0] issued_wr_next;
                logic [DESC_PTR_W-1:0] issued_rd_next;
                logic [DESC_CNT_W-1:0] issued_count_next;
                logic open_valid_next;
                logic [AXI_ADDR_WIDTH-1:0] open_addr_next;
                logic [AXI_ADDR_WIDTH-1:0] open_next_addr_next;
                logic [BURST_CNT_W-1:0] open_len_next;
                logic [BURST_CNT_W-1:0] w_left_next;
                logic [OUTSTANDING_CNT_W-1:0] outstanding_next;

                desc_wr_next = desc_wr_ptr_q;
                desc_rd_next = desc_rd_ptr_q;
                desc_count_next = desc_count_q;
                issued_wr_next = issued_wr_ptr_q;
                issued_rd_next = issued_rd_ptr_q;
                issued_count_next = issued_count_q;
                open_valid_next = open_valid_q;
                open_addr_next = open_addr_q;
                open_next_addr_next = open_next_addr_q;
                open_len_next = open_len_q;
                w_left_next = w_beats_left_q;
                outstanding_next = outstanding_q;

                if (start && !busy) begin
                    busy <= 1'b1;
                    error_sticky <= 1'b0;
                    desc_wr_next = '0;
                    desc_rd_next = '0;
                    desc_count_next = '0;
                    issued_wr_next = '0;
                    issued_rd_next = '0;
                    issued_count_next = '0;
                    open_valid_next = 1'b0;
                    open_addr_next = '0;
                    open_next_addr_next = '0;
                    open_len_next = '0;
                    last_seen_q <= 1'b0;
                    w_left_next = '0;
                    outstanding_next = '0;
                end

                if (push_beat_w) begin
                    if (break_before_push_w) begin
                        desc_addr_q[desc_wr_next] <= open_addr_q;
                        desc_len_q[desc_wr_next] <= open_len_q;
                        desc_wr_next = desc_ptr_inc(desc_wr_next);
                        desc_count_next = desc_count_next + DESC_CNT_W'(1);
                        open_valid_next = 1'b0;
                        open_len_next = '0;
                    end

                    if (close_after_push_w) begin
                        desc_addr_q[desc_wr_next] <= (break_before_push_w || !open_valid_q) ? s_addr : open_addr_q;
                        desc_len_q[desc_wr_next] <= new_open_len_w;
                        desc_wr_next = desc_ptr_inc(desc_wr_next);
                        desc_count_next = desc_count_next + DESC_CNT_W'(1);
                        open_valid_next = 1'b0;
                        open_len_next = '0;
                    end else begin
                        open_valid_next = 1'b1;
                        if (break_before_push_w || !open_valid_q) begin
                            open_addr_next = s_addr;
                        end
                        open_next_addr_next = s_addr + AXI_ADDR_WIDTH'(BEAT_BYTES);
                        open_len_next = new_open_len_w;
                    end

                    if (s_last) begin
                        last_seen_q <= 1'b1;
                    end
                end

                if (aw_fire_w) begin
                    desc_rd_next = desc_ptr_inc(desc_rd_next);
                    desc_count_next = desc_count_next - DESC_CNT_W'(1);
                    issued_len_q[issued_wr_next] <= desc_len_q[desc_rd_ptr_q];
                    issued_wr_next = desc_ptr_inc(issued_wr_next);
                    issued_count_next = issued_count_next + DESC_CNT_W'(1);
                    outstanding_next = outstanding_next + OUTSTANDING_CNT_W'(1);
                end

                if (w_fire_w) begin
                    if (w_beats_left_q == '0) begin
                        issued_rd_next = desc_ptr_inc(issued_rd_next);
                        issued_count_next = issued_count_next - DESC_CNT_W'(1);
                        if (issued_len_q[issued_rd_ptr_q] > BURST_CNT_W'(1)) begin
                            w_left_next = issued_len_q[issued_rd_ptr_q] - BURST_CNT_W'(1);
                        end else begin
                            w_left_next = '0;
                        end
                    end else if (w_beats_left_q > BURST_CNT_W'(1)) begin
                        w_left_next = w_beats_left_q - BURST_CNT_W'(1);
                    end else begin
                        w_left_next = '0;
                    end
                end

                if (b_fire_w) begin
                    outstanding_next = outstanding_next - OUTSTANDING_CNT_W'(1);
                    if (m_axi_bresp != 2'b00) begin
                        error_sticky <= 1'b1;
                    end
                end

                desc_wr_ptr_q <= desc_wr_next;
                desc_rd_ptr_q <= desc_rd_next;
                desc_count_q <= desc_count_next;
                issued_wr_ptr_q <= issued_wr_next;
                issued_rd_ptr_q <= issued_rd_next;
                issued_count_q <= issued_count_next;
                open_valid_q <= open_valid_next;
                open_addr_q <= open_addr_next;
                open_next_addr_q <= open_next_addr_next;
                open_len_q <= open_len_next;
                w_beats_left_q <= w_left_next;
                outstanding_q <= outstanding_next;

                if (busy && last_seen_q && !open_valid_next &&
                    beat_fifo_empty_w && (desc_count_next == '0) &&
                    (issued_count_next == '0) && (w_left_next == '0) &&
                    (outstanding_next == '0)) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    last_seen_q <= 1'b0;
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (AXI_DATA_WIDTH != 128) begin
            $error("DMA_MVOUT_S2MM_WRITER_1AXI expects 128-bit AXI data");
        end
        if (AXI_MAX_BURST_BEATS != 16) begin
            $error("DMA_MVOUT_S2MM_WRITER_1AXI expects 16-beat bursts");
        end
    end
`endif
endmodule

module DMA_MVOUT_SPM_1AXI
import npu_config_pkg::*;
#(
    parameter int AXI_ID_WIDTH        = npu_config_pkg::AXI_ID_WIDTH,
    parameter int AXI_ADDR_WIDTH      = npu_config_pkg::AXI_ADDR_WIDTH,
    parameter int AXI_DATA_WIDTH      = npu_config_pkg::AXI_DATA_WIDTH,
    parameter int AXI_MAX_BURST_BEATS = 16,
    parameter int AXI_MAX_OUTSTANDING = 16,
    parameter int SPM_BANKS           = npu_config_pkg::SPM_BANK_NUM,
    parameter int SPM_BANK_DATA_WIDTH = npu_config_pkg::SPM_BANK_DATA_WIDTH,
    parameter int SPM_DEPTH           = npu_config_pkg::SPM_SIZE / (npu_config_pkg::SPM_BANK_NUM * (npu_config_pkg::SPM_BANK_DATA_WIDTH / 8)),
    parameter int SPM_READ_LATENCY    = 1,
    parameter int LINE_FIFO_DEPTH     = 8
) (
    input  logic                                                   clk,
    input  logic                                                   rst_n,

    input  logic [AXI_ADDR_WIDTH-1:0]                              mvout_dram_base_addr,
    input  logic [$clog2(SPM_DEPTH)-1:0]                           mvout_spm_addr,
    input  logic [15:0]                                            mvout_line_num,
    input  logic [1:0]                                             mvout_output_precision,
    input  logic                                                   dma_mvout_req_en,
    output logic                                                   dma_mvout_resp_done,
    output logic                                                   dma_mvout_busy,

    output logic                                                   spm_rd_en,
    output logic [$clog2(SPM_DEPTH)-1:0]                           spm_rd_addr,
    input  logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]               spm_rd_data,

    output logic [AXI_ID_WIDTH-1:0]                                m_axi_awid,
    output logic [AXI_ADDR_WIDTH-1:0]                              m_axi_awaddr,
    output logic [7:0]                                             m_axi_awlen,
    output logic [2:0]                                             m_axi_awsize,
    output logic [1:0]                                             m_axi_awburst,
    output logic                                                   m_axi_awvalid,
    input  logic                                                   m_axi_awready,

    output logic [AXI_DATA_WIDTH-1:0]                              m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0]                            m_axi_wstrb,
    output logic                                                   m_axi_wlast,
    output logic                                                   m_axi_wvalid,
    input  logic                                                   m_axi_wready,

    input  logic [AXI_ID_WIDTH-1:0]                                m_axi_bid,
    input  logic [1:0]                                             m_axi_bresp,
    input  logic                                                   m_axi_bvalid,
    output logic                                                   m_axi_bready
);

localparam int AXI_STRB_WIDTH       = AXI_DATA_WIDTH / 8;
localparam int AXI_ADDR_LSB         = $clog2(AXI_STRB_WIDTH);
localparam int LINE_DATA_WIDTH      = SPM_BANKS * SPM_BANK_DATA_WIDTH;
localparam int LINE_BEATS           = LINE_DATA_WIDTH / AXI_DATA_WIDTH;
localparam int FP16_PER_LINE        = LINE_DATA_WIDTH / 16;
localparam int FP32_LINE_DATA_WIDTH = FP16_PER_LINE * 32;
localparam int FP32_LINE_BEATS      = FP32_LINE_DATA_WIDTH / AXI_DATA_WIDTH;
localparam int MAX_LINE_DATA_WIDTH  = FP32_LINE_DATA_WIDTH;
localparam int LINE_FIFO_PTR_W      = (LINE_FIFO_DEPTH <= 2) ? 1 : $clog2(LINE_FIFO_DEPTH);
localparam int LINE_FIFO_CNT_W      = $clog2(LINE_FIFO_DEPTH + 1);
localparam int RD_PIPE_W            = (SPM_READ_LATENCY < 1) ? 1 : SPM_READ_LATENCY;
localparam int BEAT_IDX_W           = $clog2(FP32_LINE_BEATS);

localparam logic [1:0] PREC_FP16 = 2'b01;
localparam logic [1:0] PREC_FP32 = 2'b11;
localparam logic [1:0] PREC_RAW8 = 2'b00;

logic producer_active_q;
logic fp32_mode_q;
logic [15:0] active_line_beats_q;
logic [15:0] lines_to_issue_q;
logic [$clog2(SPM_DEPTH)-1:0] next_spm_addr_q;
logic [31:0] beats_total_q;
logic [31:0] beats_sent_q;
logic [AXI_ADDR_WIDTH-1:0] stream_addr_q;

logic [RD_PIPE_W-1:0] rd_valid_pipe_q;
logic [LINE_FIFO_CNT_W-1:0] rd_inflight_q;

logic [MAX_LINE_DATA_WIDTH-1:0] line_fifo_q [LINE_FIFO_DEPTH];
logic [LINE_FIFO_PTR_W-1:0] line_wr_ptr_q;
logic [LINE_FIFO_PTR_W-1:0] line_rd_ptr_q;
logic [LINE_FIFO_CNT_W-1:0] line_count_q;
logic [BEAT_IDX_W-1:0] emit_beat_idx_q;

logic writer_start_w;
logic writer_busy_w;
logic writer_done_w;
logic writer_error_w;
logic stream_valid_w;
logic stream_ready_w;
logic stream_fire_w;
logic stream_last_w;
logic issue_read_w;
logic capture_read_w;
logic pop_line_w;
logic [LINE_FIFO_CNT_W-1:0] reserved_lines_w;

function automatic logic [LINE_FIFO_PTR_W-1:0] fifo_ptr_inc(input logic [LINE_FIFO_PTR_W-1:0] ptr);
    if (ptr == LINE_FIFO_PTR_W'(LINE_FIFO_DEPTH - 1)) begin
        fifo_ptr_inc = '0;
    end else begin
        fifo_ptr_inc = ptr + LINE_FIFO_PTR_W'(1);
    end
endfunction

function automatic logic [31:0] fp16_to_fp32_bits(input logic [15:0] fp16);
    logic sign;
    logic [4:0] exp16;
    logic [9:0] frac16;
    logic [7:0] exp32;
    begin
        sign = fp16[15];
        exp16 = fp16[14:10];
        frac16 = fp16[9:0];
        exp32 = '0;
        fp16_to_fp32_bits = {sign, 31'd0};

        if (exp16 == 5'h1f) begin
            fp16_to_fp32_bits = {sign, 8'hff, (frac16 == '0) ? 23'd0 : {1'b1, frac16, 12'd0}};
        end else if (exp16 != 5'd0) begin
            exp32 = 8'(int'(exp16) - 15 + 127);
            fp16_to_fp32_bits = {sign, exp32, frac16, 13'd0};
        end
    end
endfunction

function automatic logic [MAX_LINE_DATA_WIDTH-1:0] expand_line_data(
    input logic [LINE_DATA_WIDTH-1:0] fp16_line,
    input logic fp32_mode
);
    begin
        expand_line_data = '0;
        if (fp32_mode) begin
            for (int elem = 0; elem < FP16_PER_LINE; elem++) begin
                expand_line_data[elem*32 +: 32] = fp16_to_fp32_bits(fp16_line[elem*16 +: 16]);
            end
        end else begin
            expand_line_data[LINE_DATA_WIDTH-1:0] = fp16_line;
        end
    end
endfunction

assign reserved_lines_w = line_count_q + rd_inflight_q;
assign issue_read_w = producer_active_q &&
                      (lines_to_issue_q != 16'd0) &&
                      (reserved_lines_w < LINE_FIFO_CNT_W'(LINE_FIFO_DEPTH));
assign capture_read_w = rd_valid_pipe_q[RD_PIPE_W-1];

assign spm_rd_en = issue_read_w;
assign spm_rd_addr = next_spm_addr_q;

assign stream_valid_w = producer_active_q && writer_busy_w && (line_count_q != '0);
assign stream_fire_w = stream_valid_w && stream_ready_w;
assign stream_last_w = (beats_sent_q + 32'd1) == beats_total_q;
assign pop_line_w = stream_fire_w &&
                    (emit_beat_idx_q == BEAT_IDX_W'(active_line_beats_q - 16'd1));

assign dma_mvout_busy = producer_active_q || writer_busy_w;
assign writer_start_w = dma_mvout_req_en && !producer_active_q && !writer_busy_w && (mvout_line_num != 16'd0);

DMA_MVOUT_S2MM_WRITER_1AXI #(
    .AXI_ID_WIDTH        (AXI_ID_WIDTH),
    .AXI_ADDR_WIDTH      (AXI_ADDR_WIDTH),
    .AXI_DATA_WIDTH      (AXI_DATA_WIDTH),
    .AXI_MAX_BURST_BEATS (AXI_MAX_BURST_BEATS),
    .AXI_WR_OUTSTANDING  (AXI_MAX_OUTSTANDING)
) u_writer (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (writer_start_w),
    .busy           (writer_busy_w),
    .done           (writer_done_w),
    .error_sticky   (writer_error_w),
    .s_valid        (stream_valid_w),
    .s_ready        (stream_ready_w),
    .s_data         (line_fifo_q[line_rd_ptr_q][emit_beat_idx_q*AXI_DATA_WIDTH +: AXI_DATA_WIDTH]),
    .s_keep         ('1),
    .s_addr         (stream_addr_q),
    .s_last         (stream_last_w),
    .m_axi_awid     (m_axi_awid),
    .m_axi_awaddr   (m_axi_awaddr),
    .m_axi_awlen    (m_axi_awlen),
    .m_axi_awsize   (m_axi_awsize),
    .m_axi_awburst  (m_axi_awburst),
    .m_axi_awvalid  (m_axi_awvalid),
    .m_axi_awready  (m_axi_awready),
    .m_axi_wdata    (m_axi_wdata),
    .m_axi_wstrb    (m_axi_wstrb),
    .m_axi_wlast    (m_axi_wlast),
    .m_axi_wvalid   (m_axi_wvalid),
    .m_axi_wready   (m_axi_wready),
    .m_axi_bresp    (m_axi_bresp),
    .m_axi_bvalid   (m_axi_bvalid),
    .m_axi_bready   (m_axi_bready)
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        producer_active_q <= 1'b0;
        fp32_mode_q <= 1'b0;
        active_line_beats_q <= '0;
        lines_to_issue_q <= '0;
        next_spm_addr_q <= '0;
        beats_total_q <= '0;
        beats_sent_q <= '0;
        stream_addr_q <= '0;
        rd_valid_pipe_q <= '0;
        rd_inflight_q <= '0;
        line_wr_ptr_q <= '0;
        line_rd_ptr_q <= '0;
        line_count_q <= '0;
        emit_beat_idx_q <= '0;
        dma_mvout_resp_done <= 1'b0;
    end else begin
        dma_mvout_resp_done <= 1'b0;

        if (dma_mvout_req_en && !producer_active_q && !writer_busy_w) begin
            if (mvout_line_num == 16'd0) begin
                dma_mvout_resp_done <= 1'b1;
            end else begin
`ifndef SYNTHESIS
                if ((mvout_output_precision != PREC_RAW8) &&
                    (mvout_output_precision != PREC_FP16) &&
                    (mvout_output_precision != PREC_FP32)) begin
                    $fatal(1, "DMA_MVOUT_SPM_1AXI supports RAW8(00), FP16(01), or FP32(11) output precision");
                end
`endif
                producer_active_q <= 1'b1;
                fp32_mode_q <= (mvout_output_precision == PREC_FP32);
                active_line_beats_q <= (mvout_output_precision == PREC_FP32) ?
                                       16'(FP32_LINE_BEATS) : 16'(LINE_BEATS);
                lines_to_issue_q <= mvout_line_num;
                next_spm_addr_q <= mvout_spm_addr;
                beats_total_q <= {16'd0, mvout_line_num} *
                                 ((mvout_output_precision == PREC_FP32) ?
                                  32'(FP32_LINE_BEATS) : 32'(LINE_BEATS));
                beats_sent_q <= 32'd0;
                stream_addr_q <= mvout_dram_base_addr;
                rd_valid_pipe_q <= '0;
                rd_inflight_q <= '0;
                line_wr_ptr_q <= '0;
                line_rd_ptr_q <= '0;
                line_count_q <= '0;
                emit_beat_idx_q <= '0;
            end
        end else begin
            if (issue_read_w) begin
                lines_to_issue_q <= lines_to_issue_q - 16'd1;
                next_spm_addr_q <= next_spm_addr_q + 1'b1;
            end

            rd_valid_pipe_q <= (rd_valid_pipe_q << 1) | RD_PIPE_W'(issue_read_w);

            case ({issue_read_w, capture_read_w})
                2'b10: rd_inflight_q <= rd_inflight_q + LINE_FIFO_CNT_W'(1);
                2'b01: rd_inflight_q <= rd_inflight_q - LINE_FIFO_CNT_W'(1);
                default: begin
                end
            endcase

            if (capture_read_w) begin
                line_fifo_q[line_wr_ptr_q] <= expand_line_data(spm_rd_data, fp32_mode_q);
                line_wr_ptr_q <= fifo_ptr_inc(line_wr_ptr_q);
            end

            if (pop_line_w) begin
                line_rd_ptr_q <= fifo_ptr_inc(line_rd_ptr_q);
            end

            case ({capture_read_w, pop_line_w})
                2'b10: line_count_q <= line_count_q + LINE_FIFO_CNT_W'(1);
                2'b01: line_count_q <= line_count_q - LINE_FIFO_CNT_W'(1);
                default: begin
                end
            endcase

            if (stream_fire_w) begin
                beats_sent_q <= beats_sent_q + 32'd1;
                stream_addr_q <= stream_addr_q + AXI_ADDR_WIDTH'(AXI_STRB_WIDTH);
                if (pop_line_w) begin
                    emit_beat_idx_q <= '0;
                end else begin
                    emit_beat_idx_q <= emit_beat_idx_q + BEAT_IDX_W'(1);
                end
            end

            if (writer_done_w) begin
                producer_active_q <= 1'b0;
                dma_mvout_resp_done <= 1'b1;
            end
        end
    end
end

`ifndef SYNTHESIS
initial begin
    if (AXI_DATA_WIDTH != 128) begin
        $error("DMA_MVOUT_SPM_1AXI expects 128-bit AXI data");
    end
    if (AXI_MAX_BURST_BEATS != 16) begin
        $error("DMA_MVOUT_SPM_1AXI expects 16-beat bursts");
    end
    if (AXI_MAX_OUTSTANDING != 16) begin
        $error("DMA_MVOUT_SPM_1AXI expects 16 outstanding writes");
    end
    if (SPM_READ_LATENCY < 1) begin
        $error("DMA_MVOUT_SPM_1AXI expects SPM_READ_LATENCY >= 1");
    end
    if (LINE_BEATS != 4) begin
        $error("DMA_MVOUT_SPM_1AXI expects 512-bit output SPM lines on a 128-bit AXI port");
    end
    if (FP32_LINE_BEATS != 8) begin
        $error("DMA_MVOUT_SPM_1AXI expects FP32 expanded lines to be 8 AXI beats");
    end
    if (LINE_FIFO_DEPTH < 4) begin
        $error("DMA_MVOUT_SPM_1AXI expects LINE_FIFO_DEPTH >= 4");
    end
end
`endif

wire unused_bid = |m_axi_bid;
wire unused_error = writer_error_w;
wire unused_addr_lsb = ^AXI_ADDR_LSB;

endmodule

`endif
