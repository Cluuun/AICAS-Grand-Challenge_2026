`ifndef DMA_MVIN_4AXI_V
`define DMA_MVIN_4AXI_V

module DMA_MVIN_4AXI #(
    parameter int SPM_FPGA_SRAM        = 0,
    parameter int AXI_PORTS            = 4,
    parameter int AXI_ID_WIDTH         = 4,
    parameter int AXI_ADDR_WIDTH       = 32,
    parameter int AXI_DATA_WIDTH       = 128,
    parameter int AXI_MAX_BURST_BEATS  = 16,
    parameter int AXI_MAX_OUTSTANDING  = 16,
    parameter int SPM_BANKS            = 4,
    parameter int SPM_BANK_DATA_WIDTH  = 128,
    parameter int SPM_DEPTH            = 1024
) (
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  logic [AXI_ADDR_WIDTH-1:0]                               mvin_dram_base_addr,
    input  logic [$clog2(SPM_DEPTH)-1:0]                            mvin_spm_addr,
    input  logic [15:0]                                             mvin_line_num,
    input  logic                                                    stream_ring_mode,
    input  logic [15:0]                                             stream_ring_free_lines,
    output logic                                                    stream_reserve_valid,
    output logic [4:0]                                              stream_reserve_lines,
    input  logic                                                    dma_mvin_req_en,
    output logic                                                    dma_mvin_resp_done,
    output logic                                                    dma_mvin_busy,

    output logic                                                    spm_wr_en,
    output logic [$clog2(SPM_DEPTH)-1:0]                            spm_wr_addr,
    output logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]           spm_wr_data,
    output logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH/8-1:0]         spm_wr_mask,
    input  logic                                                    spm_wr_ready,
    input  logic                                                    spm_wr_linear_order,

    output logic [AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]                  m_axi_arid,
    output logic [AXI_PORTS-1:0][AXI_ADDR_WIDTH-1:0]                m_axi_araddr,
    output logic [AXI_PORTS-1:0][7:0]                               m_axi_arlen,
    output logic [AXI_PORTS-1:0][2:0]                               m_axi_arsize,
    output logic [AXI_PORTS-1:0][1:0]                               m_axi_arburst,
    output logic [AXI_PORTS-1:0]                                    m_axi_arvalid,
    input  logic [AXI_PORTS-1:0]                                    m_axi_arready,

    input  logic [AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]                  m_axi_rid,
    input  logic [AXI_PORTS-1:0][AXI_DATA_WIDTH-1:0]                m_axi_rdata,
    input  logic [AXI_PORTS-1:0][1:0]                               m_axi_rresp,
    input  logic [AXI_PORTS-1:0]                                    m_axi_rlast,
    input  logic [AXI_PORTS-1:0]                                    m_axi_rvalid,
    output logic [AXI_PORTS-1:0]                                    m_axi_rready
);

localparam int AXI_STRB_WIDTH         = AXI_DATA_WIDTH / 8;
localparam int AXI_ADDR_LSB           = $clog2(AXI_STRB_WIDTH);
localparam int SPM_ADDR_WIDTH         = $clog2(SPM_DEPTH);
localparam int FIFO_DEPTH             = AXI_MAX_BURST_BEATS * AXI_MAX_OUTSTANDING;
localparam int PORT_SEL_WIDTH         = (AXI_PORTS <= 1) ? 1 : $clog2(AXI_PORTS);
localparam int FIFO_CNT_WIDTH         = $clog2(FIFO_DEPTH + 1);
localparam int OUTSTANDING_CNT_WIDTH  = (AXI_MAX_OUTSTANDING <= 1) ? 1 : $clog2(AXI_MAX_OUTSTANDING + 1);
localparam int LINE_BYTES             = SPM_BANKS * AXI_STRB_WIDTH;
localparam int BLOCK_LINES            = AXI_PORTS * AXI_MAX_BURST_BEATS / SPM_BANKS;
localparam int PORT_LINES_PER_BLOCK   = AXI_MAX_BURST_BEATS / SPM_BANKS;
localparam int BLOCK_LINE_SHIFT       = $clog2(BLOCK_LINES);
localparam int PORT_LINE_SHIFT        = $clog2(PORT_LINES_PER_BLOCK);
localparam int LINE_BYTE_SHIFT        = $clog2(LINE_BYTES);
localparam int PORT_BYTE_SHIFT        = $clog2(PORT_LINES_PER_BLOCK * LINE_BYTES);

typedef enum logic [0:0] {
    ST_IDLE   = 1'b0,
    ST_ACTIVE = 1'b1
} dma_state_t;

dma_state_t                                      state_q;
logic [SPM_ADDR_WIDTH-1:0]                      spm_addr_q;
logic [SPM_ADDR_WIDTH-1:0]                      stream_spm_wr_ptr_q;
logic [15:0]                                    total_lines_q;
logic [15:0]                                    written_rows_total_q;
logic [15:0]                                    pop_rows_issued_q;
logic [15:0]                                    pop_line_q;
logic                                           stream_mode_q;
logic [15:0]                                    stream_issue_group_q;
logic                                           stream_group_active_q;
logic [AXI_PORTS-1:0]                           stream_group_pending_q;
logic [AXI_PORTS-1:0][15:0]                     issued_beats_q;
logic [AXI_PORTS-1:0][15:0]                     received_beats_q;
logic [AXI_PORTS-1:0][15:0]                     port_total_beats_q;
logic [AXI_PORTS-1:0][15:0]                     reserved_beats_q;
logic [AXI_PORTS-1:0][OUTSTANDING_CNT_WIDTH-1:0] outstanding_bursts_q;
logic [PORT_SEL_WIDTH-1:0]                      fifo_pop_port_q;

logic [AXI_PORTS-1:0]                           ar_can_issue;
logic [AXI_PORTS-1:0]                           ar_fire;
logic [AXI_PORTS-1:0]                           fifo_full;
logic [AXI_PORTS-1:0]                           r_fire;
logic [AXI_PORTS-1:0]                           fifo_row_ready;
logic [AXI_PORTS-1:0]                           fifo_push_q;
logic [AXI_PORTS-1:0][AXI_DATA_WIDTH-1:0]       fifo_push_data_q;
logic                                           fifo_pop_fire_w;
logic [PORT_SEL_WIDTH-1:0]                      fifo_pop_port_w;
logic [AXI_PORTS-1:0]                           fifo_pop_sel_w;
logic [AXI_PORTS-1:0]                           fifo_pop_valid;
logic                                           fifo_write_fire;
logic [PORT_SEL_WIDTH-1:0]                      fifo_write_port;
logic [AXI_PORTS-1:0][FIFO_CNT_WIDTH-1:0]       fifo_count;
logic [AXI_PORTS-1:0][SPM_BANKS-1:0][AXI_DATA_WIDTH-1:0] fifo_pop_data;
logic [AXI_PORTS-1:0][15:0]                     remaining_beats_w;
logic [AXI_PORTS-1:0][15:0]                     ar_burst_beats_w;
logic [AXI_PORTS-1:0][15:0]                     reserved_next_w;
logic [AXI_PORTS-1:0][OUTSTANDING_CNT_WIDTH-1:0] outstanding_bursts_next_w;
logic [AXI_PORTS-1:0][AXI_ADDR_WIDTH-1:0]       next_ar_addr_w;
logic [AXI_PORTS-1:0][15:0]                     stream_group_port_beats_w;
logic [AXI_PORTS-1:0][15:0]                     start_port_total_beats_w;
logic                                           all_received_next_w;
logic                                           transfer_done_w;
logic [15:0]                                    written_rows_total_next_w;
logic [PORT_SEL_WIDTH-1:0]                      expected_pop_port_w;
logic [4:0]                                     stream_next_group_lines_w;
logic                                           stream_can_reserve_group_w;
logic                                           stream_group_done_w;

function automatic [15:0] min16(input logic [15:0] a, input logic [15:0] b);
    min16 = (a < b) ? a : b;
endfunction

function automatic [15:0] port_total_beats(
    input int unsigned port_idx,
    input logic [15:0] total_lines
);
    logic [15:0] full_blocks;
    logic [3:0]  tail_lines;
    logic [15:0] port_tail_base;
    logic [15:0] port_tail_lines;
    begin
        full_blocks = total_lines >> BLOCK_LINE_SHIFT;
        tail_lines = total_lines[BLOCK_LINE_SHIFT-1:0];
        port_tail_base = 16'(port_idx << PORT_LINE_SHIFT);
        if ({12'd0, tail_lines} <= port_tail_base) begin
            port_tail_lines = 16'd0;
        end else if (({12'd0, tail_lines} - port_tail_base) > 16'(PORT_LINES_PER_BLOCK)) begin
            port_tail_lines = 16'(PORT_LINES_PER_BLOCK);
        end else begin
            port_tail_lines = {12'd0, tail_lines} - port_tail_base;
        end
        port_total_beats = (full_blocks << $clog2(AXI_MAX_BURST_BEATS)) +
                           (port_tail_lines << $clog2(SPM_BANKS));
    end
endfunction

function automatic [AXI_ADDR_WIDTH-1:0] port_ar_addr(
    input int unsigned port_idx,
    input logic [15:0] issued_beats
);
    logic [15:0] group_idx;
    logic [3:0]  beat_in_group;
    logic [AXI_ADDR_WIDTH-1:0] group_byte_offset;
    logic [AXI_ADDR_WIDTH-1:0] port_byte_offset;
    logic [AXI_ADDR_WIDTH-1:0] beat_byte_offset;
    begin
        group_idx = issued_beats >> $clog2(AXI_MAX_BURST_BEATS);
        beat_in_group = issued_beats[$clog2(AXI_MAX_BURST_BEATS)-1:0];
        group_byte_offset = AXI_ADDR_WIDTH'(group_idx) << (BLOCK_LINE_SHIFT + LINE_BYTE_SHIFT);
        port_byte_offset = AXI_ADDR_WIDTH'(port_idx) << PORT_BYTE_SHIFT;
        beat_byte_offset = AXI_ADDR_WIDTH'(beat_in_group) << AXI_ADDR_LSB;
        port_ar_addr = mvin_dram_base_addr + group_byte_offset + port_byte_offset + beat_byte_offset;
    end
endfunction

function automatic [4:0] stream_group_lines(
    input logic [15:0] group_idx,
    input logic [15:0] total_lines
);
    logic [15:0] group_base;
    logic [15:0] remaining;
    begin
        group_base = group_idx << BLOCK_LINE_SHIFT;
        if (total_lines <= group_base) begin
            stream_group_lines = 5'd0;
        end else begin
            remaining = total_lines - group_base;
            stream_group_lines = (remaining > 16'(BLOCK_LINES)) ?
                                 5'(BLOCK_LINES) : remaining[4:0];
        end
    end
endfunction

function automatic [15:0] stream_group_port_beats(
    input logic [15:0] group_idx,
    input int unsigned port_idx,
    input logic [15:0] total_lines
);
    logic [15:0] line_base;
    logic [15:0] remaining;
    logic [15:0] port_lines;
    begin
        line_base = (group_idx << BLOCK_LINE_SHIFT) +
                    16'(port_idx << PORT_LINE_SHIFT);
        if (total_lines <= line_base) begin
            stream_group_port_beats = 16'd0;
        end else begin
            remaining = total_lines - line_base;
            port_lines = (remaining > 16'(PORT_LINES_PER_BLOCK)) ?
                         16'(PORT_LINES_PER_BLOCK) : remaining;
            stream_group_port_beats = port_lines << $clog2(SPM_BANKS);
        end
    end
endfunction

always_comb begin
    all_received_next_w = 1'b1;
    written_rows_total_next_w = written_rows_total_q;
    fifo_pop_fire_w = 1'b0;
    fifo_pop_port_w = '0;
    fifo_pop_sel_w = '0;
    expected_pop_port_w = PORT_SEL_WIDTH'(pop_line_q[PORT_LINE_SHIFT +: PORT_SEL_WIDTH]);
    stream_next_group_lines_w = stream_group_lines(stream_issue_group_q, total_lines_q);
    stream_can_reserve_group_w = (state_q == ST_ACTIVE) &&
                                 stream_mode_q &&
                                 !stream_group_active_q &&
                                 (stream_next_group_lines_w != 5'd0) &&
                                 (stream_ring_free_lines >= {11'd0, stream_next_group_lines_w});

    for (int i = 0; i < AXI_PORTS; i++) begin
        start_port_total_beats_w[i] = port_total_beats(i, mvin_line_num);
        remaining_beats_w[i] = port_total_beats_q[i] - issued_beats_q[i];
        next_ar_addr_w[i] = port_ar_addr(i, issued_beats_q[i]);
        stream_group_port_beats_w[i] =
            stream_group_port_beats(stream_issue_group_q, i, total_lines_q);
        ar_burst_beats_w[i] = stream_mode_q ?
                              stream_group_port_beats_w[i] :
                              min16(remaining_beats_w[i],
                                    16'(AXI_MAX_BURST_BEATS) -
                                    {12'd0, issued_beats_q[i][$clog2(AXI_MAX_BURST_BEATS)-1:0]});
        ar_fire[i] = m_axi_arvalid[i] && m_axi_arready[i];
        r_fire[i] = (state_q == ST_ACTIVE) && m_axi_rvalid[i] && m_axi_rready[i];
        reserved_next_w[i] = reserved_beats_q[i] +
                             (ar_fire[i] ? ar_burst_beats_w[i] : 16'd0) -
                             16'(r_fire[i]);
        outstanding_bursts_next_w[i] = outstanding_bursts_q[i] +
                                       OUTSTANDING_CNT_WIDTH'(ar_fire[i]) -
                                       OUTSTANDING_CNT_WIDTH'(r_fire[i] && m_axi_rlast[i]);
        ar_can_issue[i] = (state_q == ST_ACTIVE) &&
                          (stream_mode_q ?
                           (stream_group_active_q &&
                            stream_group_pending_q[i] &&
                            (stream_group_port_beats_w[i] != 16'd0)) :
                           (remaining_beats_w[i] != 16'd0)) &&
                          (ar_burst_beats_w[i] != 16'd0) &&
                          (outstanding_bursts_q[i] < OUTSTANDING_CNT_WIDTH'(AXI_MAX_OUTSTANDING)) &&
                          (({1'b0, fifo_count[i]} + {1'b0, reserved_beats_q[i]} +
                            (FIFO_CNT_WIDTH+1)'(ar_burst_beats_w[i])) <=
                           (FIFO_CNT_WIDTH+1)'(FIFO_DEPTH));
        all_received_next_w = all_received_next_w &&
                              ((received_beats_q[i] + 16'(r_fire[i])) == port_total_beats_q[i]);
    end

    if (state_q == ST_ACTIVE &&
        spm_wr_ready &&
        (pop_rows_issued_q < total_lines_q) &&
        fifo_row_ready[expected_pop_port_w]) begin
        fifo_pop_fire_w = 1'b1;
        fifo_pop_port_w = expected_pop_port_w;
        fifo_pop_sel_w[expected_pop_port_w] = 1'b1;
    end

    fifo_write_fire = 1'b0;
    fifo_write_port = '0;
    for (int i = 0; i < AXI_PORTS; i++) begin
        if (!fifo_write_fire && fifo_pop_valid[i]) begin
            fifo_write_fire = 1'b1;
            fifo_write_port = PORT_SEL_WIDTH'(i);
        end
    end
    if (fifo_write_fire) begin
        written_rows_total_next_w = written_rows_total_q + 16'd1;
    end

    transfer_done_w = (state_q == ST_ACTIVE) &&
                      all_received_next_w &&
                      (written_rows_total_next_w == total_lines_q);
    stream_group_done_w = stream_mode_q &&
                          stream_group_active_q &&
                          ((stream_group_pending_q & ~ar_fire) == '0);
end

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_q <= ST_IDLE;
        spm_addr_q <= '0;
        stream_spm_wr_ptr_q <= '0;
        total_lines_q <= '0;
        written_rows_total_q <= '0;
        pop_rows_issued_q <= '0;
        pop_line_q <= '0;
        stream_mode_q <= 1'b0;
        stream_issue_group_q <= '0;
        stream_group_active_q <= 1'b0;
        stream_group_pending_q <= '0;
        issued_beats_q <= '0;
        received_beats_q <= '0;
        port_total_beats_q <= '0;
        reserved_beats_q <= '0;
        outstanding_bursts_q <= '0;
        fifo_pop_port_q <= '0;
        dma_mvin_resp_done <= 1'b0;
        dma_mvin_busy <= 1'b0;
        spm_wr_en <= 1'b0;
        spm_wr_addr <= '0;
        spm_wr_data <= '0;
        spm_wr_mask <= '0;
        m_axi_arid <= '0;
        m_axi_araddr <= '0;
        m_axi_arlen <= '0;
        m_axi_arsize <= '0;
        m_axi_arburst <= '0;
        m_axi_arvalid <= '0;
        m_axi_rready <= '0;
        fifo_push_q <= '0;
        fifo_push_data_q <= '0;
        stream_reserve_valid <= 1'b0;
        stream_reserve_lines <= '0;
    end else begin
        dma_mvin_resp_done <= 1'b0;
        spm_wr_en <= 1'b0;
        stream_reserve_valid <= 1'b0;
        stream_reserve_lines <= '0;

        case (state_q)
            ST_IDLE: begin
                dma_mvin_busy <= 1'b0;
                m_axi_arvalid <= '0;
                m_axi_rready <= '0;
                fifo_push_q <= '0;
                stream_group_active_q <= 1'b0;
                stream_group_pending_q <= '0;
                if (dma_mvin_req_en) begin
                    if (mvin_line_num == '0) begin
                        dma_mvin_resp_done <= 1'b1;
                    end else begin
`ifndef SYNTHESIS
                        if (!stream_ring_mode && ((mvin_line_num % SPM_BANKS) != 0)) begin
                            $fatal(1, "DMA_MVIN_4AXI expects mvin_line_num to be a multiple of %0d", SPM_BANKS);
                        end
                        if (mvin_dram_base_addr[7:0] != 8'd0) begin
                            $fatal(1, "DMA_MVIN_4AXI block-interleaved mode expects 256B-aligned DRAM base");
                        end
`endif
                        spm_addr_q <= mvin_spm_addr;
                        stream_spm_wr_ptr_q <= mvin_spm_addr;
                        total_lines_q <= mvin_line_num;
                        written_rows_total_q <= '0;
                        pop_rows_issued_q <= '0;
                        pop_line_q <= '0;
                        stream_mode_q <= stream_ring_mode;
                        stream_issue_group_q <= '0;
                        stream_group_active_q <= 1'b0;
                        stream_group_pending_q <= '0;
                        issued_beats_q <= '0;
                        received_beats_q <= '0;
                        port_total_beats_q <= start_port_total_beats_w;
                        reserved_beats_q <= '0;
                        outstanding_bursts_q <= '0;
                        fifo_pop_port_q <= '0;
                        fifo_push_data_q <= '0;
                        dma_mvin_busy <= 1'b1;
                        state_q <= ST_ACTIVE;
                    end
                end
            end

            ST_ACTIVE: begin
                dma_mvin_busy <= 1'b1;

                if (stream_can_reserve_group_w) begin
                    stream_group_active_q <= 1'b1;
                    stream_reserve_valid <= 1'b1;
                    stream_reserve_lines <= stream_next_group_lines_w;
                    for (int i = 0; i < AXI_PORTS; i++) begin
                        stream_group_pending_q[i] <=
                            (stream_group_port_beats(stream_issue_group_q, i, total_lines_q) != 16'd0);
                    end
                end

                for (int i = 0; i < AXI_PORTS; i++) begin
                    fifo_push_q[i] <= r_fire[i];
                    if (r_fire[i]) begin
                        fifo_push_data_q[i] <= m_axi_rdata[i];
                    end

                    m_axi_rready[i] <= (received_beats_q[i] < port_total_beats_q[i]);

                    if (ar_fire[i]) begin
                        m_axi_arvalid[i] <= 1'b0;
                        if (stream_mode_q) begin
                            stream_group_pending_q[i] <= 1'b0;
                        end
                    end
                    if (!m_axi_arvalid[i] && ar_can_issue[i]) begin
                        m_axi_arid[i]    <= '0;
                        m_axi_araddr[i]  <= next_ar_addr_w[i];
                        m_axi_arlen[i]   <= ar_burst_beats_w[i][7:0] - 8'd1;
                        m_axi_arsize[i]  <= $clog2(AXI_STRB_WIDTH);
                        m_axi_arburst[i] <= 2'b01;
                        m_axi_arvalid[i] <= 1'b1;
                    end

                    if (ar_fire[i]) begin
                        issued_beats_q[i] <= issued_beats_q[i] + ar_burst_beats_w[i];
                    end
                    if (r_fire[i]) begin
                        received_beats_q[i] <= received_beats_q[i] + 1'b1;
                    end
                    reserved_beats_q[i] <= reserved_next_w[i];
                    outstanding_bursts_q[i] <= outstanding_bursts_next_w[i];
                end

                if (stream_group_done_w) begin
                    stream_group_active_q <= 1'b0;
                    stream_issue_group_q <= stream_issue_group_q + 1'b1;
                end

                if (transfer_done_w) begin
                    dma_mvin_busy <= 1'b0;
                    dma_mvin_resp_done <= 1'b1;
                    m_axi_rready <= '0;
                    fifo_push_q <= '0;
                    state_q <= ST_IDLE;
                end
            end

            default: begin
                state_q <= ST_IDLE;
                fifo_push_q <= '0;
            end
        endcase

        if (fifo_pop_fire_w) begin
            fifo_pop_port_q <= fifo_pop_port_w;
            pop_rows_issued_q <= pop_rows_issued_q + 16'd1;
            pop_line_q <= pop_line_q + 16'd1;
        end

        if (fifo_write_fire) begin
            spm_wr_en   <= 1'b1;
            spm_wr_addr <= stream_mode_q ?
                           stream_spm_wr_ptr_q :
                           (spm_addr_q + written_rows_total_q[SPM_ADDR_WIDTH-1:0]);
            spm_wr_mask <= '1;
            for (int bank = 0; bank < SPM_BANKS; bank++) begin
                spm_wr_data[bank] <= fifo_pop_data[fifo_write_port][bank];
            end
            if (stream_mode_q) begin
                if (stream_spm_wr_ptr_q == SPM_ADDR_WIDTH'(SPM_DEPTH - 1)) begin
                    stream_spm_wr_ptr_q <= spm_addr_q;
                end else begin
                    stream_spm_wr_ptr_q <= stream_spm_wr_ptr_q + 1'b1;
                end
            end
            written_rows_total_q <= written_rows_total_next_w;
        end
    end
end

`ifndef SYNTHESIS
initial begin
    if (AXI_PORTS != 4) begin
        $error("DMA_MVIN_4AXI expects AXI_PORTS == 4");
    end
    if (SPM_BANKS != 4) begin
        $error("DMA_MVIN_4AXI expects SPM_BANKS == 4");
    end
    if (AXI_DATA_WIDTH != SPM_BANK_DATA_WIDTH) begin
        $error("DMA_MVIN_4AXI expects AXI_DATA_WIDTH == SPM_BANK_DATA_WIDTH");
    end
    if (AXI_MAX_BURST_BEATS != 16) begin
        $error("DMA_MVIN_4AXI block-interleaved mode expects 16-beat bursts");
    end
    if (AXI_MAX_OUTSTANDING != 16) begin
        $warning("DMA_MVIN_4AXI block-interleaved mode is targeted for 16 outstanding bursts");
    end
    if ((FIFO_DEPTH % SPM_BANKS) != 0) begin
        $error("DMA_MVIN_4AXI expects FIFO_DEPTH to be a multiple of SPM_BANKS");
    end
end

always_ff @(posedge clk) begin
    if (rst_n) begin
        for (int i = 0; i < AXI_PORTS; i++) begin
            assert (({1'b0, fifo_count[i]} + {1'b0, reserved_beats_q[i]}) <=
                    (FIFO_CNT_WIDTH+1)'(FIFO_DEPTH))
                else $fatal(1, "DMA_MVIN_4AXI FIFO reservation overflow port=%0d", i);
            assert (!(m_axi_rvalid[i] && !m_axi_rready[i] && (received_beats_q[i] < port_total_beats_q[i])))
                else $fatal(1, "DMA_MVIN_4AXI R channel backpressure port=%0d", i);
            if (m_axi_arvalid[i] && m_axi_arready[i]) begin
                assert ({1'b0, m_axi_araddr[i][11:0]} +
                        ({4'd0, m_axi_arlen[i]} + 13'd1) * 13'(AXI_STRB_WIDTH) <= 13'h1000)
                    else $fatal(1, "DMA_MVIN_4AXI burst crosses 4KB boundary port=%0d addr=0x%08h len=%0d",
                                i, m_axi_araddr[i], m_axi_arlen[i]);
            end
        end
        assert (!stream_reserve_valid ||
                (stream_ring_free_lines >= {11'd0, stream_reserve_lines}))
            else $fatal(1, "DMA_MVIN_4AXI stream reserve without ring credit");
    end
end
`endif

generate
    genvar port_idx;
    for (port_idx = 0; port_idx < AXI_PORTS; port_idx++) begin : gen_dma_fifo
        dma_mvin_4axi_fifo #(
            .SPM_FPGA_SRAM (SPM_FPGA_SRAM),
            .DATA_WIDTH    (AXI_DATA_WIDTH),
            .DEPTH         (FIFO_DEPTH),
            .BANKS         (SPM_BANKS),
            .RAM_STYLE     ("block")
        ) u_dma_mvin_4axi_fifo (
            .clk           (clk),
            .rst_n         (rst_n),
            .push          (fifo_push_q[port_idx]),
            .push_data     (fifo_push_data_q[port_idx]),
            .full          (fifo_full[port_idx]),
            .row_ready     (fifo_row_ready[port_idx]),
            .pop_row       (fifo_pop_sel_w[port_idx]),
            .pop_row_data  (fifo_pop_data[port_idx]),
            .pop_row_valid (fifo_pop_valid[port_idx]),
            .count         (fifo_count[port_idx])
        );
    end
endgenerate

endmodule

`endif
