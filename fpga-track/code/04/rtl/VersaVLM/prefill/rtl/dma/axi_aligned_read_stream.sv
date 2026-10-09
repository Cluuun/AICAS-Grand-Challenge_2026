`default_nettype none

// Aligned AXI4 memory-to-stream reader.
//
// The engine issues INCR read bursts up to 16 beats, keeps up to 16 bursts
// outstanding, and preserves one 128-bit output stream beat per AXI R beat.
// Addresses must be 16-byte aligned. Row byte counts may end on a partial
// final beat, reflected in the output keep mask.
module axi_aligned_read_stream #(
    parameter int AXI_ID_WIDTH        = 4,
    parameter int AXI_ADDR_WIDTH      = 32,
    parameter int AXI_DATA_WIDTH      = 128,
    parameter int AXI_MAX_BURST_BEATS = 16,
    parameter int AXI_RD_OUTSTANDING  = 16,
    parameter int FIFO_DEPTH          = 512,
    localparam int AXI_STRB_W         = AXI_DATA_WIDTH / 8,
    localparam int FIFO_PTR_W         = (FIFO_DEPTH <= 1) ? 1 : $clog2(FIFO_DEPTH),
    localparam int FIFO_COUNT_W       = $clog2(FIFO_DEPTH + 1),
    localparam int OUTSTANDING_W      = $clog2(AXI_RD_OUTSTANDING + 1)
) (
    input  wire logic                         clk_i,
    input  wire logic                         rst_i,
    input  wire logic                         clear_i,

    input  wire logic                         start_i,
    output logic                              busy_o,
    output logic                              done_o,
    output logic                              error_sticky_o,

    input  wire logic [AXI_ADDR_WIDTH-1:0]    base_addr_i,
    input  wire logic [15:0]                  row_count_i,
    input  wire logic [31:0]                  row_bytes_i,
    input  wire logic [31:0]                  row_stride_bytes_i,

    output logic [AXI_ID_WIDTH-1:0]           m_axi_arid_o,
    output logic [AXI_ADDR_WIDTH-1:0]         m_axi_araddr_o,
    output logic [7:0]                        m_axi_arlen_o,
    output logic [2:0]                        m_axi_arsize_o,
    output logic [1:0]                        m_axi_arburst_o,
    output logic                              m_axi_arvalid_o,
    input  wire logic                         m_axi_arready_i,
    input  wire logic [AXI_ID_WIDTH-1:0]      m_axi_rid_i,
    input  wire logic [AXI_DATA_WIDTH-1:0]    m_axi_rdata_i,
    input  wire logic [1:0]                   m_axi_rresp_i,
    input  wire logic                         m_axi_rlast_i,
    input  wire logic                         m_axi_rvalid_i,
    output logic                              m_axi_rready_o,

    output logic                              stream_valid_o,
    input  wire logic                         stream_ready_i,
    output logic [AXI_DATA_WIDTH-1:0]         stream_data_o,
    output logic [AXI_STRB_W-1:0]             stream_keep_o,
    output logic                              stream_last_o
);
    localparam int BEAT_BYTES    = AXI_STRB_W;
    localparam int BEAT_BYTE_LG2 = $clog2(BEAT_BYTES);
    localparam int BURST_CNT_W   = $clog2(AXI_MAX_BURST_BEATS + 1);
    localparam int BURST_PTR_W   = (AXI_RD_OUTSTANDING <= 1) ? 1 : $clog2(AXI_RD_OUTSTANDING);
    localparam logic [2:0] AXI_SIZE = $clog2(AXI_STRB_W);

    typedef logic [AXI_STRB_W-1:0] keep_t;

    keep_t burst_last_keep_q [AXI_RD_OUTSTANDING];
    logic burst_transfer_last_q [AXI_RD_OUTSTANDING];
    logic [BURST_CNT_W-1:0] burst_beats_q [AXI_RD_OUTSTANDING];
    logic [BURST_PTR_W-1:0] burst_wr_ptr_q;
    logic [BURST_PTR_W-1:0] burst_rd_ptr_q;
    logic [OUTSTANDING_W-1:0] burst_meta_count_q;
    logic cur_meta_valid_q;
    logic [BURST_CNT_W-1:0] cur_beats_left_q;
    keep_t cur_last_keep_q;
    logic cur_transfer_last_q;
    logic hold_valid_q;
    logic [AXI_DATA_WIDTH-1:0] hold_data_q;
    keep_t hold_keep_q;
    logic hold_last_q;

    logic [AXI_ADDR_WIDTH-1:0] issue_addr_q;
    logic [AXI_ADDR_WIDTH-1:0] row_base_addr_q;
    logic [15:0] rows_left_q;
    logic [31:0] row_bytes_left_q;
    logic [31:0] row_stride_bytes_q;
    logic [31:0] row_bytes_total_q;
    logic issue_done_q;

    logic [OUTSTANDING_W-1:0] outstanding_q;
    logic arvalid_q;
    logic [AXI_ADDR_WIDTH-1:0] araddr_q;
    logic [BURST_CNT_W-1:0] ar_beats_q;
    logic [31:0] ar_bytes_q;
    logic ar_row_last_q;
    logic ar_transfer_last_q;

    logic stream_fire;
    logic r_fire;
    logic ar_fire;
    logic rlast_fire;
    logic burst_done_fire;
    logic can_issue;
    logic load_cur_meta;
    logic cur_last_beat;
    logic [31:0] beats_left_in_row;
    logic [31:0] beats_to_4k;
    logic [31:0] issue_beats;
    logic [31:0] issue_bytes;
    logic issue_row_last;
    logic issue_transfer_last;
    logic start_align_error;
    logic [31:0] ar_last_beat_bytes;

    assign stream_valid_o = hold_valid_q;
    assign stream_data_o = hold_data_q;
    assign stream_keep_o = hold_keep_q;
    assign stream_last_o = hold_last_q;
    assign stream_fire = stream_valid_o && stream_ready_i;

    assign m_axi_arid_o = '0;
    assign m_axi_araddr_o = araddr_q;
    assign m_axi_arlen_o = 8'(ar_beats_q - BURST_CNT_W'(1));
    assign m_axi_arsize_o = AXI_SIZE;
    assign m_axi_arburst_o = 2'b01;
    assign m_axi_arvalid_o = arvalid_q;
    assign ar_fire = m_axi_arvalid_o && m_axi_arready_i;

    assign m_axi_rready_o = busy_o && cur_meta_valid_q && (!hold_valid_q || stream_ready_i);
    assign r_fire = m_axi_rvalid_i && m_axi_rready_o;
    assign rlast_fire = r_fire && m_axi_rlast_i;
    assign cur_last_beat = cur_meta_valid_q && (cur_beats_left_q <= BURST_CNT_W'(1));
    assign burst_done_fire = r_fire && (m_axi_rlast_i || cur_last_beat);
    assign load_cur_meta = !cur_meta_valid_q && (burst_meta_count_q != '0);

    function automatic logic [31:0] ceil_div_beat(input logic [31:0] bytes);
        ceil_div_beat = (bytes + 32'(BEAT_BYTES - 1)) >> BEAT_BYTE_LG2;
    endfunction

    function automatic keep_t keep_for_bytes(input logic [31:0] bytes);
        keep_t keep;
        begin
            keep = '0;
            for (int lane = 0; lane < AXI_STRB_W; lane++) begin
                keep[lane] = (bytes > 32'(lane));
            end
            return keep;
        end
    endfunction

    function automatic logic [31:0] min32(input logic [31:0] a, input logic [31:0] b);
        min32 = (a < b) ? a : b;
    endfunction

    always_comb begin
        beats_left_in_row = ceil_div_beat(row_bytes_left_q);
        beats_to_4k = (32'h1000 - {20'd0, issue_addr_q[11:0]}) >> BEAT_BYTE_LG2;
        if (beats_to_4k == 32'd0) begin
            beats_to_4k = 32'd256;
        end
        issue_beats = min32(32'(AXI_MAX_BURST_BEATS), min32(beats_left_in_row, beats_to_4k));
        issue_bytes = min32(row_bytes_left_q, issue_beats << BEAT_BYTE_LG2);
        issue_row_last = (issue_bytes == row_bytes_left_q);
        issue_transfer_last = issue_row_last && (rows_left_q <= 16'd1);
        start_align_error = (base_addr_i[BEAT_BYTE_LG2-1:0] != '0) ||
                            (row_stride_bytes_i[BEAT_BYTE_LG2-1:0] != '0);
        can_issue = busy_o && !issue_done_q && !arvalid_q &&
                    !error_sticky_o &&
                    (row_bytes_left_q != 32'd0) &&
                    (issue_beats != 32'd0) &&
                    (outstanding_q < OUTSTANDING_W'(AXI_RD_OUTSTANDING));
        ar_last_beat_bytes = ar_bytes_q - ((32'(ar_beats_q) - 32'd1) << BEAT_BYTE_LG2);
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            busy_o <= 1'b0;
            done_o <= 1'b0;
            error_sticky_o <= 1'b0;
            burst_wr_ptr_q <= '0;
            burst_rd_ptr_q <= '0;
            burst_meta_count_q <= '0;
            cur_meta_valid_q <= 1'b0;
            cur_beats_left_q <= '0;
            cur_last_keep_q <= '0;
            cur_transfer_last_q <= 1'b0;
            hold_valid_q <= 1'b0;
            hold_data_q <= '0;
            hold_keep_q <= '0;
            hold_last_q <= 1'b0;
            issue_addr_q <= '0;
            row_base_addr_q <= '0;
            rows_left_q <= '0;
            row_bytes_left_q <= '0;
            row_stride_bytes_q <= '0;
            row_bytes_total_q <= '0;
            issue_done_q <= 1'b0;
            outstanding_q <= '0;
            arvalid_q <= 1'b0;
            araddr_q <= '0;
            ar_beats_q <= '0;
            ar_bytes_q <= '0;
            ar_row_last_q <= 1'b0;
            ar_transfer_last_q <= 1'b0;
        end else begin
            done_o <= 1'b0;

            if (clear_i) begin
                busy_o <= 1'b0;
                error_sticky_o <= 1'b0;
                burst_wr_ptr_q <= '0;
                burst_rd_ptr_q <= '0;
                burst_meta_count_q <= '0;
                cur_meta_valid_q <= 1'b0;
                cur_beats_left_q <= '0;
                cur_last_keep_q <= '0;
                cur_transfer_last_q <= 1'b0;
                hold_valid_q <= 1'b0;
                issue_done_q <= 1'b0;
                outstanding_q <= '0;
                arvalid_q <= 1'b0;
            end else begin
                if (start_i && !busy_o) begin
                    busy_o <= 1'b1;
                    issue_addr_q <= base_addr_i;
                    row_base_addr_q <= base_addr_i;
                    rows_left_q <= row_count_i;
                    row_bytes_left_q <= row_bytes_i;
                    row_stride_bytes_q <= row_stride_bytes_i;
                    row_bytes_total_q <= row_bytes_i;
                    issue_done_q <= (row_count_i == 16'd0) || (row_bytes_i == 32'd0) || start_align_error;
                    burst_wr_ptr_q <= '0;
                    burst_rd_ptr_q <= '0;
                    burst_meta_count_q <= '0;
                    cur_meta_valid_q <= 1'b0;
                    hold_valid_q <= 1'b0;
                    outstanding_q <= '0;
                    arvalid_q <= 1'b0;
                    if (start_align_error) begin
                        error_sticky_o <= 1'b1;
                    end
                end

                if (can_issue) begin
                    arvalid_q <= 1'b1;
                    araddr_q <= issue_addr_q;
                    ar_beats_q <= BURST_CNT_W'(issue_beats);
                    ar_bytes_q <= issue_bytes;
                    ar_row_last_q <= issue_row_last;
                    ar_transfer_last_q <= issue_transfer_last;
                end

                if (ar_fire) begin
                    arvalid_q <= 1'b0;
                    burst_beats_q[burst_wr_ptr_q] <= ar_beats_q;
                    burst_last_keep_q[burst_wr_ptr_q] <= keep_for_bytes(min32(ar_last_beat_bytes, 32'(BEAT_BYTES)));
                    burst_transfer_last_q[burst_wr_ptr_q] <= ar_transfer_last_q;
                    burst_wr_ptr_q <= burst_wr_ptr_q + BURST_PTR_W'(1);
                    if (ar_row_last_q) begin
                        if (rows_left_q <= 16'd1) begin
                            issue_done_q <= 1'b1;
                            rows_left_q <= '0;
                            row_bytes_left_q <= '0;
                        end else begin
                            rows_left_q <= rows_left_q - 16'd1;
                            row_base_addr_q <= row_base_addr_q + AXI_ADDR_WIDTH'(row_stride_bytes_q);
                            issue_addr_q <= row_base_addr_q + AXI_ADDR_WIDTH'(row_stride_bytes_q);
                            row_bytes_left_q <= row_bytes_total_q;
                        end
                    end else begin
                        issue_addr_q <= issue_addr_q + AXI_ADDR_WIDTH'(ar_bytes_q);
                        row_bytes_left_q <= row_bytes_left_q - ar_bytes_q;
                    end
                end

                if (load_cur_meta) begin
                    cur_meta_valid_q <= 1'b1;
                    cur_beats_left_q <= burst_beats_q[burst_rd_ptr_q];
                    cur_last_keep_q <= burst_last_keep_q[burst_rd_ptr_q];
                    cur_transfer_last_q <= burst_transfer_last_q[burst_rd_ptr_q];
                    burst_rd_ptr_q <= burst_rd_ptr_q + BURST_PTR_W'(1);
                end

                if (r_fire) begin
                    hold_valid_q <= 1'b1;
                    hold_data_q <= m_axi_rdata_i;
                    hold_keep_q <= cur_last_beat ? cur_last_keep_q : {AXI_STRB_W{1'b1}};
                    hold_last_q <= cur_last_beat && cur_transfer_last_q;
                    if (cur_last_beat || m_axi_rlast_i) begin
                        cur_meta_valid_q <= 1'b0;
                        cur_beats_left_q <= '0;
                    end else begin
                        cur_beats_left_q <= cur_beats_left_q - BURST_CNT_W'(1);
                    end
                    if (m_axi_rresp_i != 2'b00) begin
                        error_sticky_o <= 1'b1;
                        issue_done_q <= 1'b1;
                    end
                    if (m_axi_rlast_i != cur_last_beat) begin
                        error_sticky_o <= 1'b1;
                        issue_done_q <= 1'b1;
                    end
                end

                unique case ({ar_fire, burst_done_fire})
                    2'b10: outstanding_q <= outstanding_q + OUTSTANDING_W'(1);
                    2'b01: outstanding_q <= outstanding_q - OUTSTANDING_W'(1);
                    default: outstanding_q <= outstanding_q;
                endcase

                unique case ({ar_fire, load_cur_meta})
                    2'b10: burst_meta_count_q <= burst_meta_count_q + OUTSTANDING_W'(1);
                    2'b01: burst_meta_count_q <= burst_meta_count_q - OUTSTANDING_W'(1);
                    default: burst_meta_count_q <= burst_meta_count_q;
                endcase

                if (stream_fire && !r_fire) begin
                    hold_valid_q <= 1'b0;
                end

                if (busy_o && issue_done_q && !arvalid_q &&
                    (outstanding_q == '0) && (burst_meta_count_q == '0) &&
                    !cur_meta_valid_q && !hold_valid_q) begin
                    busy_o <= 1'b0;
                    done_o <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (AXI_DATA_WIDTH != 128) begin
            $error("axi_aligned_read_stream expects 128-bit AXI data");
        end
        if (AXI_MAX_BURST_BEATS != 16) begin
            $error("axi_aligned_read_stream target expects 16-beat bursts");
        end
        if (AXI_RD_OUTSTANDING != 16) begin
            $error("axi_aligned_read_stream target expects 16 outstanding reads");
        end
        if ((FIFO_DEPTH & (FIFO_DEPTH - 1)) != 0) begin
            $error("axi_aligned_read_stream FIFO_DEPTH must be a power of two");
        end
    end
endmodule

`default_nettype wire
