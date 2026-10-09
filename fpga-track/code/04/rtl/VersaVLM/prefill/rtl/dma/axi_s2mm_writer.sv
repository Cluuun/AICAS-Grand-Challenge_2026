`default_nettype none

// Single-AXI 128-bit stream-to-memory writer for O/MVOUT.
//
// Input beats already carry byte addresses. The writer groups contiguous beats
// into 16-beat AXI INCR bursts, splits at 4KB boundaries, and keeps AW/W/B
// decoupled so up to AXI_WR_OUTSTANDING writes can be in flight.
module axi_s2mm_writer #(
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
    localparam int BEAT_PTR_W         = (BEAT_FIFO_DEPTH <= 2) ? 1 : $clog2(BEAT_FIFO_DEPTH),
    localparam int BEAT_CNT_W         = $clog2(BEAT_FIFO_DEPTH + 1),
    localparam int DESC_PTR_W         = (DESC_FIFO_DEPTH <= 2) ? 1 : $clog2(DESC_FIFO_DEPTH),
    localparam int DESC_CNT_W         = $clog2(DESC_FIFO_DEPTH + 1)
) (
    input  wire logic                         clk_i,
    input  wire logic                         rst_i,
    input  wire logic                         clear_i,

    input  wire logic                         start_i,
    output logic                              busy_o,
    output logic                              done_o,
    output logic                              error_sticky_o,

    input  wire logic                         s_valid_i,
    output logic                              s_ready_o,
    input  wire logic [AXI_DATA_WIDTH-1:0]    s_data_i,
    input  wire logic [AXI_STRB_W-1:0]        s_keep_i,
    input  wire logic [AXI_ADDR_WIDTH-1:0]    s_addr_i,
    input  wire logic                         s_last_i,

    output logic [AXI_ID_WIDTH-1:0]           m_axi_awid_o,
    output logic [AXI_ADDR_WIDTH-1:0]         m_axi_awaddr_o,
    output logic [7:0]                        m_axi_awlen_o,
    output logic [2:0]                        m_axi_awsize_o,
    output logic [1:0]                        m_axi_awburst_o,
    output logic                              m_axi_awvalid_o,
    input  wire logic                         m_axi_awready_i,
    output logic [AXI_DATA_WIDTH-1:0]         m_axi_wdata_o,
    output logic [AXI_STRB_W-1:0]             m_axi_wstrb_o,
    output logic                              m_axi_wlast_o,
    output logic                              m_axi_wvalid_o,
    input  wire logic                         m_axi_wready_i,
    input  wire logic [1:0]                   m_axi_bresp_i,
    input  wire logic                         m_axi_bvalid_i,
    output logic                              m_axi_bready_o
);
    localparam int BEAT_BYTES = AXI_STRB_W;
    localparam logic [2:0] AXI_SIZE = $clog2(AXI_STRB_W);

    typedef logic [AXI_DATA_WIDTH-1:0] beat_data_t;
    typedef logic [AXI_STRB_W-1:0] beat_strb_t;

    beat_data_t beat_data_q [BEAT_FIFO_DEPTH];
    beat_strb_t beat_strb_q [BEAT_FIFO_DEPTH];
    logic [BEAT_PTR_W-1:0] beat_wr_ptr_q;
    logic [BEAT_PTR_W-1:0] beat_rd_ptr_q;
    logic [BEAT_CNT_W-1:0] beat_count_q;

    logic [AXI_ADDR_WIDTH-1:0] desc_addr_q [DESC_FIFO_DEPTH];
    logic [BURST_CNT_W-1:0] desc_len_q [DESC_FIFO_DEPTH];
    logic [DESC_PTR_W-1:0] desc_wr_ptr_q;
    logic [DESC_PTR_W-1:0] desc_rd_ptr_q;
    logic [DESC_CNT_W-1:0] desc_count_q;

    logic [BURST_CNT_W-1:0] issued_len_q [DESC_FIFO_DEPTH];
    logic [DESC_PTR_W-1:0] issued_wr_ptr_q;
    logic [DESC_PTR_W-1:0] issued_rd_ptr_q;
    logic [DESC_CNT_W-1:0] issued_count_q;

    logic open_valid_q;
    logic [AXI_ADDR_WIDTH-1:0] open_addr_q;
    logic [AXI_ADDR_WIDTH-1:0] open_next_addr_q;
    logic [BURST_CNT_W-1:0] open_len_q;
    logic last_seen_q;

    logic [BURST_CNT_W-1:0] w_beats_left_q;
    logic [OUTSTANDING_CNT_W-1:0] outstanding_q;

    logic accept;
    logic push_beat;
    logic break_before_push;
    logic close_after_push;
    logic [BURST_CNT_W-1:0] new_open_len;
    logic aw_fire;
    logic w_fire;
    logic b_fire;
    logic [BURST_CNT_W-1:0] w_current_len;

    function automatic logic [BEAT_PTR_W-1:0] beat_ptr_inc(input logic [BEAT_PTR_W-1:0] ptr);
        begin
            beat_ptr_inc = (ptr == BEAT_PTR_W'(BEAT_FIFO_DEPTH - 1)) ? '0 : (ptr + BEAT_PTR_W'(1));
        end
    endfunction

    function automatic logic [DESC_PTR_W-1:0] desc_ptr_inc(input logic [DESC_PTR_W-1:0] ptr);
        begin
            desc_ptr_inc = (ptr == DESC_PTR_W'(DESC_FIFO_DEPTH - 1)) ? '0 : (ptr + DESC_PTR_W'(1));
        end
    endfunction

    assign s_ready_o = busy_o &&
                       (beat_count_q < BEAT_CNT_W'(BEAT_FIFO_DEPTH)) &&
                       (desc_count_q <= DESC_CNT_W'(DESC_FIFO_DEPTH - 3));
    assign accept = s_valid_i && s_ready_o;
    assign push_beat = accept && (s_keep_i != '0);
    assign break_before_push = push_beat && open_valid_q &&
                               ((open_len_q == BURST_CNT_W'(AXI_MAX_BURST_BEATS)) ||
                                (s_addr_i != open_next_addr_q) ||
                                (s_addr_i[11:0] == 12'h000));
    assign new_open_len = break_before_push ? BURST_CNT_W'(1) :
                          (open_valid_q ? (open_len_q + BURST_CNT_W'(1)) : BURST_CNT_W'(1));
    assign close_after_push = push_beat && (s_last_i || (new_open_len == BURST_CNT_W'(AXI_MAX_BURST_BEATS)));

    assign m_axi_awid_o = '0;
    assign m_axi_awaddr_o = desc_addr_q[desc_rd_ptr_q];
    assign m_axi_awlen_o = 8'(desc_len_q[desc_rd_ptr_q] - BURST_CNT_W'(1));
    assign m_axi_awsize_o = AXI_SIZE;
    assign m_axi_awburst_o = 2'b01;
    assign m_axi_awvalid_o = busy_o && (desc_count_q != '0) &&
                             (issued_count_q < DESC_CNT_W'(DESC_FIFO_DEPTH)) &&
                             (outstanding_q < OUTSTANDING_CNT_W'(AXI_WR_OUTSTANDING));

    assign w_current_len = (w_beats_left_q != '0) ? w_beats_left_q : issued_len_q[issued_rd_ptr_q];
    assign m_axi_wdata_o = beat_data_q[beat_rd_ptr_q];
    assign m_axi_wstrb_o = beat_strb_q[beat_rd_ptr_q];
    assign m_axi_wlast_o = (w_current_len == BURST_CNT_W'(1));
    assign m_axi_wvalid_o = busy_o && (beat_count_q != '0) &&
                            ((w_beats_left_q != '0) || (issued_count_q != '0));
    assign m_axi_bready_o = 1'b1;

    assign aw_fire = m_axi_awvalid_o && m_axi_awready_i;
    assign w_fire = m_axi_wvalid_o && m_axi_wready_i;
    assign b_fire = m_axi_bvalid_i && m_axi_bready_o;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            busy_o <= 1'b0;
            done_o <= 1'b0;
            error_sticky_o <= 1'b0;
            beat_wr_ptr_q <= '0;
            beat_rd_ptr_q <= '0;
            beat_count_q <= '0;
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
            done_o <= 1'b0;

            if (clear_i) begin
                busy_o <= 1'b0;
                error_sticky_o <= 1'b0;
                beat_wr_ptr_q <= '0;
                beat_rd_ptr_q <= '0;
                beat_count_q <= '0;
                desc_wr_ptr_q <= '0;
                desc_rd_ptr_q <= '0;
                desc_count_q <= '0;
                issued_wr_ptr_q <= '0;
                issued_rd_ptr_q <= '0;
                issued_count_q <= '0;
                open_valid_q <= 1'b0;
                open_len_q <= '0;
                last_seen_q <= 1'b0;
                w_beats_left_q <= '0;
                outstanding_q <= '0;
            end else begin
                automatic logic [BEAT_PTR_W-1:0] beat_wr_next;
                automatic logic [BEAT_PTR_W-1:0] beat_rd_next;
                automatic logic [BEAT_CNT_W-1:0] beat_count_next;
                automatic logic [DESC_PTR_W-1:0] desc_wr_next;
                automatic logic [DESC_PTR_W-1:0] desc_rd_next;
                automatic logic [DESC_CNT_W-1:0] desc_count_next;
                automatic logic [DESC_PTR_W-1:0] issued_wr_next;
                automatic logic [DESC_PTR_W-1:0] issued_rd_next;
                automatic logic [DESC_CNT_W-1:0] issued_count_next;
                automatic logic open_valid_next;
                automatic logic [AXI_ADDR_WIDTH-1:0] open_addr_next;
                automatic logic [AXI_ADDR_WIDTH-1:0] open_next_addr_next;
                automatic logic [BURST_CNT_W-1:0] open_len_next;
                automatic logic [BURST_CNT_W-1:0] w_left_next;
                automatic logic [OUTSTANDING_CNT_W-1:0] outstanding_next;

                beat_wr_next = beat_wr_ptr_q;
                beat_rd_next = beat_rd_ptr_q;
                beat_count_next = beat_count_q;
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

                if (start_i && !busy_o) begin
                    busy_o <= 1'b1;
                    last_seen_q <= 1'b0;
                    open_valid_next = 1'b0;
                    open_len_next = '0;
                end

                if (push_beat) begin
                    beat_data_q[beat_wr_next] <= s_data_i;
                    beat_strb_q[beat_wr_next] <= s_keep_i;
                    beat_wr_next = beat_ptr_inc(beat_wr_next);
                    beat_count_next = beat_count_next + BEAT_CNT_W'(1);

                    if (break_before_push) begin
                        desc_addr_q[desc_wr_next] <= open_addr_q;
                        desc_len_q[desc_wr_next] <= open_len_q;
                        desc_wr_next = desc_ptr_inc(desc_wr_next);
                        desc_count_next = desc_count_next + DESC_CNT_W'(1);
                        open_valid_next = 1'b0;
                        open_len_next = '0;
                    end

                    if (close_after_push) begin
                        desc_addr_q[desc_wr_next] <= (break_before_push || !open_valid_q) ? s_addr_i : open_addr_q;
                        desc_len_q[desc_wr_next] <= new_open_len;
                        desc_wr_next = desc_ptr_inc(desc_wr_next);
                        desc_count_next = desc_count_next + DESC_CNT_W'(1);
                        open_valid_next = 1'b0;
                        open_len_next = '0;
                    end else begin
                        open_valid_next = 1'b1;
                        if (break_before_push || !open_valid_q) begin
                            open_addr_next = s_addr_i;
                        end
                        open_next_addr_next = s_addr_i + AXI_ADDR_WIDTH'(BEAT_BYTES);
                        open_len_next = new_open_len;
                    end

                    if (s_last_i) begin
                        last_seen_q <= 1'b1;
                    end
                end

                if (aw_fire) begin
                    desc_rd_next = desc_ptr_inc(desc_rd_next);
                    desc_count_next = desc_count_next - DESC_CNT_W'(1);
                    issued_len_q[issued_wr_next] <= desc_len_q[desc_rd_ptr_q];
                    issued_wr_next = desc_ptr_inc(issued_wr_next);
                    issued_count_next = issued_count_next + DESC_CNT_W'(1);
                    outstanding_next = outstanding_next + OUTSTANDING_CNT_W'(1);
                end

                if (w_fire) begin
                    beat_rd_next = beat_ptr_inc(beat_rd_next);
                    beat_count_next = beat_count_next - BEAT_CNT_W'(1);
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

                if (b_fire) begin
                    outstanding_next = outstanding_next - OUTSTANDING_CNT_W'(1);
                    if (m_axi_bresp_i != 2'b00) begin
                        error_sticky_o <= 1'b1;
                    end
                end

                beat_wr_ptr_q <= beat_wr_next;
                beat_rd_ptr_q <= beat_rd_next;
                beat_count_q <= beat_count_next;
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

                if (busy_o && last_seen_q && !open_valid_next &&
                    (beat_count_next == '0) && (desc_count_next == '0) &&
                    (issued_count_next == '0) && (w_left_next == '0) &&
                    (outstanding_next == '0)) begin
                    busy_o <= 1'b0;
                    done_o <= 1'b1;
                    last_seen_q <= 1'b0;
                end
            end
        end
    end

    initial begin
        if (AXI_DATA_WIDTH != 128) begin
            $error("axi_s2mm_writer expects 128-bit AXI data");
        end
        if (AXI_MAX_BURST_BEATS != 16) begin
            $error("axi_s2mm_writer target expects 16-beat bursts");
        end
        if (AXI_WR_OUTSTANDING != 16) begin
            $error("axi_s2mm_writer target expects 16 outstanding writes");
        end
    end
endmodule

`default_nettype wire
