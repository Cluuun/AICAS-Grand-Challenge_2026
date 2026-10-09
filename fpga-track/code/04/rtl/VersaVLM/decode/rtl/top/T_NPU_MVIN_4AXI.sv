`ifndef T_NPU_MVIN_4AXI_SV
`define T_NPU_MVIN_4AXI_SV

module T_NPU_MVIN_4AXI #(
    parameter int AXI_PORTS           = 4,
    parameter int AXI_ID_WIDTH        = 4,
    parameter int AXI_ADDR_WIDTH      = 32,
    parameter int AXI_DATA_WIDTH      = 128,
    parameter int AXI_MAX_BURST_BEATS = 16,
    parameter int AXI_MAX_OUTSTANDING = 16,
    parameter int SPM_FPGA_SRAM       = 0,
    parameter int SPM_BANKS           = 4,
    parameter int SPM_BANK_DATA_WIDTH = 128,
    parameter int SPM_DEPTH           = 1024,
    parameter int SCALE_SPM_DEPTH     = 4096,
    parameter int OUTPUT_SPM_DEPTH    = SPM_DEPTH,
    parameter int ACC_DATA_WIDTH      = 512,
    parameter int ACC_ADDR_WIDTH      = 19
) (
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  logic [AXI_ADDR_WIDTH-1:0]                               mvin_dram_base_addr,
    input  logic [$clog2(SPM_DEPTH)-1:0]                            mvin_spm_addr,
    input  logic [15:0]                                             mvin_line_num,
    input  logic                                                    mvin_stream_fifo_fill,
    input  logic                                                     dma_mvin_req_en,
    input  logic                                                     dma_scale_mvin_req_en,
    input  logic                                                     dma_act_mvin_req_en,
    output logic                                                     dma_mvin_resp_done,
    output logic                                                     dma_mvin_busy,

    input  logic [AXI_ADDR_WIDTH-1:0]                               mvout_dram_base_addr,
    input  logic [$clog2(OUTPUT_SPM_DEPTH)-1:0]                     mvout_spm_addr,
    input  logic [15:0]                                             mvout_row_num,
    input  logic [1:0]                                              mvout_output_precision,
    input  logic                                                    dma_mvout_req_en,
    output logic                                                    dma_mvout_resp_done,
    output logic                                                    dma_mvout_busy,

    input  logic                                                     spm_beat_rd_en,
    input  logic [$clog2(SPM_DEPTH * SPM_BANKS * (SPM_BANK_DATA_WIDTH / 8))-1:0] spm_beat_rd_addr,
    output logic [SPM_BANK_DATA_WIDTH-1:0]                          spm_beat_rd_data,
    input  logic                                                    gemv_spm_line_rd_en,
    input  logic [$clog2(SPM_DEPTH * SPM_BANKS * (SPM_BANK_DATA_WIDTH / 8))-1:0] gemv_spm_line_rd_addr,
    output logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]                gemv_spm_line_rd_data,
    input  logic                                                    gemv_weight_stream_enable,
    output logic                                                    gemv_weight_stream_valid,
    output logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]                gemv_weight_stream_data,
    output logic                                                    gemv_weight_stream_last,
    output logic                                                    gemv_weight_stream_block_ready,
    input  logic                                                    gemv_weight_stream_ready,
    input  logic                                                    gemv_scale_line_rd_en,
    input  logic [$clog2(SPM_DEPTH * SPM_BANKS * (SPM_BANK_DATA_WIDTH / 8))-1:0] gemv_scale_line_rd_addr,
    output logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]                gemv_scale_line_rd_data,
    input  logic                                                    gemv_act_line_rd_en,
    input  logic [$clog2(SPM_DEPTH * SPM_BANKS * (SPM_BANK_DATA_WIDTH / 8))-1:0] gemv_act_line_rd_addr,
    output logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]                gemv_act_line_rd_data,
    input  logic                                                    gemv_spm_line_wr_en,
    input  logic [$clog2(SPM_DEPTH * SPM_BANKS * (SPM_BANK_DATA_WIDTH / 8))-1:0] gemv_spm_line_wr_addr,
    input  logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]                gemv_spm_line_wr_data,
    input  logic [(SPM_BANKS*SPM_BANK_DATA_WIDTH)/8-1:0]            gemv_spm_line_wr_mask,
    input  logic                                                    flow_act_line_wr_en,
    input  logic [$clog2(SPM_DEPTH * SPM_BANKS * (SPM_BANK_DATA_WIDTH / 8))-1:0] flow_act_line_wr_addr,
    input  logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]                flow_act_line_wr_data,
    input  logic [(SPM_BANKS*SPM_BANK_DATA_WIDTH)/8-1:0]            flow_act_line_wr_mask,
    output logic                                                    acc_rd_en,
    output logic [ACC_ADDR_WIDTH-1:0]                               acc_rd_addr,
    input  logic [ACC_DATA_WIDTH-1:0]                               acc_rd_data,
    input  logic                                                    acc_rd_valid,

    output logic [AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]                  m_axi_awid,
    output logic [AXI_PORTS-1:0][AXI_ADDR_WIDTH-1:0]                m_axi_awaddr,
    output logic [AXI_PORTS-1:0][7:0]                               m_axi_awlen,
    output logic [AXI_PORTS-1:0][2:0]                               m_axi_awsize,
    output logic [AXI_PORTS-1:0][1:0]                               m_axi_awburst,
    output logic [AXI_PORTS-1:0]                                    m_axi_awvalid,
    input  logic [AXI_PORTS-1:0]                                    m_axi_awready,

    output logic [AXI_PORTS-1:0][AXI_DATA_WIDTH-1:0]                m_axi_wdata,
    output logic [AXI_PORTS-1:0][AXI_DATA_WIDTH/8-1:0]              m_axi_wstrb,
    output logic [AXI_PORTS-1:0]                                    m_axi_wlast,
    output logic [AXI_PORTS-1:0]                                    m_axi_wvalid,
    input  logic [AXI_PORTS-1:0]                                    m_axi_wready,

    input  logic [AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]                  m_axi_bid,
    input  logic [AXI_PORTS-1:0][1:0]                               m_axi_bresp,
    input  logic [AXI_PORTS-1:0]                                    m_axi_bvalid,
    output logic [AXI_PORTS-1:0]                                    m_axi_bready,

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

localparam int SPM_ROW_BYTES      = SPM_BANKS * (SPM_BANK_DATA_WIDTH / 8);
localparam int SPM_ADDR_WIDTH     = $clog2(SPM_DEPTH);
localparam int SPM_BANK_BYTE_LSB  = $clog2(SPM_BANK_DATA_WIDTH / 8);
localparam int SPM_ROW_ADDR_LSB   = $clog2(SPM_ROW_BYTES);
localparam int ACT_SPM_DEPTH      = 256;
localparam int ACT_ADDR_WIDTH     = $clog2(ACT_SPM_DEPTH);
localparam int SCALE_ADDR_WIDTH   = $clog2(SCALE_SPM_DEPTH);
localparam int OUTPUT_ADDR_WIDTH  = $clog2(OUTPUT_SPM_DEPTH);
localparam int STREAM_GROUP_LINES    = AXI_PORTS * AXI_MAX_BURST_BEATS / SPM_BANKS;
localparam int STREAM_LAST_LEGAL_BASE = (SPM_DEPTH >= STREAM_GROUP_LINES) ?
                                        (SPM_DEPTH - STREAM_GROUP_LINES) : 0;
localparam int STREAM_SKID_DEPTH     = 8;
localparam int STREAM_SKID_PTR_WIDTH = $clog2(STREAM_SKID_DEPTH);
localparam int STREAM_SKID_CNT_WIDTH = $clog2(STREAM_SKID_DEPTH + 1);
localparam int STREAM_RING_CNT_WIDTH = $clog2(SPM_DEPTH + 1);
localparam int OUTPUT_SPM_READ_PIPELINE = 1;
localparam int OUTPUT_SPM_READ_LATENCY  = (SPM_FPGA_SRAM == 1) ?
                                          (OUTPUT_SPM_READ_PIPELINE + 2) : 1;
localparam int OUTPUT_SPM_BYTES         = OUTPUT_SPM_DEPTH * SPM_ROW_BYTES;

logic                                           spm_wr_en;
logic [$clog2(SPM_DEPTH)-1:0]                   spm_wr_addr;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]  spm_wr_data;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH/8-1:0] spm_wr_mask;
logic                                           raw_mvin_done;
logic                                           raw_mvin_busy;
logic                                           raw_spm_wr_ready;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]  input_spm_rd_data_int;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]  scale_spm_rd_data_int;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]  act_spm_rd_data_int;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]  output_spm_rd_data_int;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]  output_spm_wr_data;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH/8-1:0] output_spm_wr_mask;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH-1:0]  flow_act_wr_data;
logic [SPM_BANKS-1:0][SPM_BANK_DATA_WIDTH/8-1:0] flow_act_wr_mask;
logic                                           stream_reject_done_q;
logic [$clog2(SPM_BANKS)-1:0]                   spm_beat_bank_sel_q;
logic                                           gemv_line_rd_active_q;
logic                                           stream_line_rd_active_q;
logic                                           gemv_scale_line_rd_active_q;
logic                                           gemv_act_line_rd_active_q;
logic                                           mvin_target_act_q;
logic                                           mvin_target_scale_q;
logic                                           mvin_target_stream_q;
logic [$clog2(SPM_DEPTH)-1:0]                   mvin_spm_addr_q;
logic                                           stream_active_q;
logic [15:0]                                    stream_total_lines_q;
logic [$clog2(SPM_DEPTH)-1:0]                   stream_base_q;
logic [$clog2(SPM_DEPTH)-1:0]                   stream_rd_ptr_q;
logic [STREAM_RING_CNT_WIDTH-1:0]               stream_fifo_capacity_q;
logic [STREAM_RING_CNT_WIDTH-1:0]               stream_ring_occupied_q;
logic [STREAM_RING_CNT_WIDTH-1:0]               stream_write_reserved_q;
logic [STREAM_SKID_CNT_WIDTH-1:0]               stream_read_inflight_q;
logic [15:0]                                    stream_read_line_count_q;
logic [15:0]                                    stream_consume_count_q;
logic                                           stream_rd_pending_q;
logic                                           stream_rd_pending_last_q;
logic                                           stream_raw_done_q;
logic                                           stream_done_w;
logic                                           stream_reader_rd_en_w;
logic [$clog2(SPM_DEPTH)-1:0]                   stream_reader_rd_addr_w;
logic                                           stream_spm_write_fire_w;
logic [15:0]                                    stream_fifo_free_lines_w;
logic [15:0]                                    stream_start_threshold_w;
logic [15:0]                                    stream_skid_future_used_w;
logic                                           stream_skid_push_w;
logic                                           stream_skid_pop_w;
logic                                           stream_skid_full_w;
logic                                           stream_skid_empty_w;
logic [STREAM_SKID_DEPTH-1:0][SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0] stream_skid_data_q;
logic [STREAM_SKID_DEPTH-1:0]                                    stream_skid_last_q;
logic [STREAM_SKID_PTR_WIDTH-1:0]                                stream_skid_wr_ptr_q;
logic [STREAM_SKID_PTR_WIDTH-1:0]                                stream_skid_rd_ptr_q;
logic [STREAM_SKID_CNT_WIDTH-1:0]                                stream_skid_count_q;
logic                                           stream_dma_reserve_valid;
logic [4:0]                                     stream_dma_reserve_lines;
logic                                           output_spm_rd_en;
logic [OUTPUT_ADDR_WIDTH-1:0]                   output_spm_rd_addr;
logic [SPM_BANKS*SPM_BANK_DATA_WIDTH-1:0]       output_spm_rd_data_flat;
logic [AXI_ID_WIDTH-1:0]                        mvout_axi_awid;
logic [AXI_ADDR_WIDTH-1:0]                      mvout_axi_awaddr;
logic [7:0]                                     mvout_axi_awlen;
logic [2:0]                                     mvout_axi_awsize;
logic [1:0]                                     mvout_axi_awburst;
logic                                           mvout_axi_awvalid;
logic                                           mvout_axi_awready;
logic [AXI_DATA_WIDTH-1:0]                      mvout_axi_wdata;
logic [AXI_DATA_WIDTH/8-1:0]                    mvout_axi_wstrb;
logic                                           mvout_axi_wlast;
logic                                           mvout_axi_wvalid;
logic                                           mvout_axi_wready;
logic [AXI_ID_WIDTH-1:0]                        mvout_axi_bid;
logic [1:0]                                     mvout_axi_bresp;
logic                                           mvout_axi_bvalid;
logic                                           mvout_axi_bready;

wire spm_rd_en_int = spm_beat_rd_en;
wire [$clog2(SPM_DEPTH)-1:0] spm_rd_addr_int = spm_beat_rd_addr[SPM_ROW_ADDR_LSB +: $clog2(SPM_DEPTH)];
wire [$clog2(SPM_DEPTH)-1:0] gemv_spm_line_rd_addr_int = gemv_spm_line_rd_addr[SPM_ROW_ADDR_LSB +: $clog2(SPM_DEPTH)];
wire [SCALE_ADDR_WIDTH-1:0] gemv_scale_line_rd_addr_int = gemv_scale_line_rd_addr[SPM_ROW_ADDR_LSB +: SCALE_ADDR_WIDTH];
wire [$clog2(SPM_DEPTH)-1:0] gemv_act_line_rd_addr_int = gemv_act_line_rd_addr[SPM_ROW_ADDR_LSB +: $clog2(SPM_DEPTH)];
wire [OUTPUT_ADDR_WIDTH-1:0] gemv_spm_line_wr_addr_int = gemv_spm_line_wr_addr[SPM_ROW_ADDR_LSB +: OUTPUT_ADDR_WIDTH];
wire [$clog2(SPM_DEPTH)-1:0] flow_act_line_wr_addr_int = flow_act_line_wr_addr[SPM_ROW_ADDR_LSB +: $clog2(SPM_DEPTH)];
wire [SCALE_ADDR_WIDTH-1:0] scale_spm_wr_addr_int = spm_wr_addr[SCALE_ADDR_WIDTH-1:0];
wire [ACT_ADDR_WIDTH-1:0] act_spm_wr_addr_int = spm_wr_addr[ACT_ADDR_WIDTH-1:0];
wire [ACT_ADDR_WIDTH-1:0] flow_act_wr_addr = flow_act_line_wr_addr_int[ACT_ADDR_WIDTH-1:0];
wire [ACT_ADDR_WIDTH-1:0] gemv_act_line_rd_addr_narrow = gemv_act_line_rd_addr_int[ACT_ADDR_WIDTH-1:0];
wire dma_mvin_req_any = dma_mvin_req_en | dma_scale_mvin_req_en |
                        dma_act_mvin_req_en;
wire stream_exclusive_req_w = dma_mvin_req_en && mvin_stream_fifo_fill &&
                              !dma_scale_mvin_req_en &&
                              !dma_act_mvin_req_en;
wire stream_ring_base_valid_w = (SPM_DEPTH >= STREAM_GROUP_LINES) &&
                                (mvin_spm_addr <= SPM_ADDR_WIDTH'(STREAM_LAST_LEGAL_BASE));
wire stream_reject_req_w = stream_exclusive_req_w && !stream_ring_base_valid_w;
wire stream_mode_req_w = stream_exclusive_req_w && stream_ring_base_valid_w;
wire stream_mvin_start_w = stream_mode_req_w &&
                            (mvin_line_num != 16'd0) && !dma_mvin_busy;
wire raw_mvin_req_w = dma_mvin_req_any && !stream_reject_req_w;
assign raw_spm_wr_ready = 1'b1;
assign dma_mvin_resp_done = stream_reject_done_q ? 1'b1 :
                            (mvin_target_stream_q ?
                             ((stream_raw_done_q || raw_mvin_done) &&
                              (stream_done_w || !stream_active_q)) :
                             raw_mvin_done);
assign dma_mvin_busy = raw_mvin_busy ||
                       (mvin_target_stream_q && stream_active_q);
assign spm_beat_rd_data = input_spm_rd_data_int[spm_beat_bank_sel_q];
assign acc_rd_en = 1'b0;
assign acc_rd_addr = '0;
assign stream_skid_full_w = stream_skid_count_q == STREAM_SKID_CNT_WIDTH'(STREAM_SKID_DEPTH);
assign stream_skid_empty_w = stream_skid_count_q == '0;
assign stream_skid_push_w = stream_rd_pending_q && !stream_skid_full_w;
assign stream_skid_pop_w = gemv_weight_stream_valid && gemv_weight_stream_ready;
assign stream_done_w = stream_active_q && stream_skid_pop_w &&
                       ((stream_consume_count_q + 16'd1) == stream_total_lines_q);
assign gemv_weight_stream_valid = !stream_skid_empty_w;
assign gemv_weight_stream_data = stream_skid_data_q[stream_skid_rd_ptr_q];
assign gemv_weight_stream_last = stream_skid_last_q[stream_skid_rd_ptr_q];
assign stream_spm_write_fire_w = stream_active_q && spm_wr_en && mvin_target_stream_q;
assign stream_fifo_free_lines_w =
    (stream_fifo_capacity_q > (stream_ring_occupied_q + stream_write_reserved_q)) ?
    16'(stream_fifo_capacity_q - stream_ring_occupied_q - stream_write_reserved_q) :
    16'd0;
assign stream_start_threshold_w =
    (stream_total_lines_q < 16'd16) ? stream_total_lines_q : 16'd16;
assign stream_skid_future_used_w = 16'(stream_skid_count_q) +
                                   16'(stream_read_inflight_q) -
                                   16'(stream_skid_pop_w);
assign gemv_weight_stream_block_ready = stream_active_q &&
                                        (stream_start_threshold_w != 16'd0) &&
                                        ((stream_skid_count_q != '0) ||
                                         (stream_ring_occupied_q >=
                                          STREAM_RING_CNT_WIDTH'(stream_start_threshold_w)));
assign stream_reader_rd_en_w = gemv_weight_stream_enable &&
                               stream_active_q &&
                               (stream_ring_occupied_q > STREAM_RING_CNT_WIDTH'(stream_read_inflight_q)) &&
                               (stream_skid_future_used_w < 16'(STREAM_SKID_DEPTH)) &&
                               (stream_read_line_count_q < stream_total_lines_q) &&
                               !gemv_spm_line_rd_en &&
                               !spm_rd_en_int;
assign stream_reader_rd_addr_w = stream_rd_ptr_q;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        spm_beat_bank_sel_q  <= '0;
        gemv_line_rd_active_q <= 1'b0;
        gemv_scale_line_rd_active_q <= 1'b0;
        gemv_act_line_rd_active_q <= 1'b0;
        mvin_target_act_q <= 1'b0;
        mvin_target_scale_q <= 1'b0;
        mvin_target_stream_q <= 1'b0;
        mvin_spm_addr_q <= '0;
        stream_active_q <= 1'b0;
        stream_total_lines_q <= '0;
        stream_base_q <= '0;
        stream_rd_ptr_q <= '0;
        stream_fifo_capacity_q <= '0;
        stream_ring_occupied_q <= '0;
        stream_write_reserved_q <= '0;
        stream_read_inflight_q <= '0;
        stream_read_line_count_q <= '0;
        stream_consume_count_q <= '0;
        stream_rd_pending_q <= 1'b0;
        stream_rd_pending_last_q <= 1'b0;
        stream_raw_done_q <= 1'b0;
        stream_line_rd_active_q <= 1'b0;
        stream_skid_wr_ptr_q <= '0;
        stream_skid_rd_ptr_q <= '0;
        stream_skid_count_q <= '0;
        stream_skid_data_q <= '0;
        stream_skid_last_q <= '0;
        stream_reject_done_q <= 1'b0;
    end else begin
        if (spm_rd_en_int) begin
            spm_beat_bank_sel_q  <= spm_beat_rd_addr[SPM_BANK_BYTE_LSB +: $clog2(SPM_BANKS)];
        end
        gemv_line_rd_active_q <= gemv_spm_line_rd_en;
        stream_line_rd_active_q <= stream_reader_rd_en_w;
        gemv_scale_line_rd_active_q <= gemv_scale_line_rd_en;
        gemv_act_line_rd_active_q <= gemv_act_line_rd_en;
        stream_reject_done_q <= stream_reject_req_w && !dma_mvin_busy;

        if (dma_mvin_req_any && !dma_mvin_busy) begin
            mvin_target_act_q <= dma_act_mvin_req_en;
            mvin_target_scale_q <= dma_scale_mvin_req_en;
            mvin_target_stream_q <= stream_mode_req_w;
            mvin_spm_addr_q <= mvin_spm_addr;
        end

        if (stream_mvin_start_w) begin
            mvin_target_act_q <= 1'b0;
            mvin_target_scale_q <= 1'b0;
            mvin_target_stream_q <= 1'b1;
            stream_active_q <= 1'b1;
            stream_total_lines_q <= mvin_line_num;
            stream_base_q <= mvin_spm_addr;
            stream_rd_ptr_q <= mvin_spm_addr;
            stream_fifo_capacity_q <= STREAM_RING_CNT_WIDTH'(SPM_DEPTH) -
                                      STREAM_RING_CNT_WIDTH'(mvin_spm_addr);
            stream_ring_occupied_q <= '0;
            stream_write_reserved_q <= '0;
            stream_read_inflight_q <= '0;
            stream_read_line_count_q <= '0;
            stream_consume_count_q <= '0;
            stream_rd_pending_q <= 1'b0;
            stream_rd_pending_last_q <= 1'b0;
            stream_raw_done_q <= 1'b0;
            stream_skid_wr_ptr_q <= '0;
            stream_skid_rd_ptr_q <= '0;
            stream_skid_count_q <= '0;
            stream_skid_data_q <= '0;
            stream_skid_last_q <= '0;
        end else begin
            if (mvin_target_stream_q && raw_mvin_done) begin
                stream_raw_done_q <= 1'b1;
            end

            stream_rd_pending_q <= stream_reader_rd_en_w;
            stream_rd_pending_last_q <= stream_reader_rd_en_w &&
                                        (stream_read_line_count_q ==
                                         (stream_total_lines_q - 16'd1));

            stream_read_inflight_q <= stream_read_inflight_q +
                                      STREAM_SKID_CNT_WIDTH'(stream_reader_rd_en_w) -
                                      STREAM_SKID_CNT_WIDTH'(stream_skid_push_w);

            stream_ring_occupied_q <= stream_ring_occupied_q +
                                      STREAM_RING_CNT_WIDTH'(stream_spm_write_fire_w) -
                                      STREAM_RING_CNT_WIDTH'(stream_skid_push_w);
            stream_write_reserved_q <= stream_write_reserved_q +
                                       STREAM_RING_CNT_WIDTH'(stream_dma_reserve_valid ?
                                                             stream_dma_reserve_lines :
                                                             5'd0) -
                                       STREAM_RING_CNT_WIDTH'(stream_spm_write_fire_w);

            if (stream_reader_rd_en_w) begin
                if (stream_rd_ptr_q == SPM_ADDR_WIDTH'(SPM_DEPTH - 1)) begin
                    stream_rd_ptr_q <= stream_base_q;
                end else begin
                    stream_rd_ptr_q <= stream_rd_ptr_q + 1'b1;
                end
                stream_read_line_count_q <= stream_read_line_count_q + 1'b1;
            end

            if (stream_skid_push_w) begin
                for (int bank = 0; bank < SPM_BANKS; bank++) begin
                    stream_skid_data_q[stream_skid_wr_ptr_q][bank*SPM_BANK_DATA_WIDTH +: SPM_BANK_DATA_WIDTH] <=
                        input_spm_rd_data_int[bank];
                end
                stream_skid_last_q[stream_skid_wr_ptr_q] <= stream_rd_pending_last_q;
                stream_skid_wr_ptr_q <= stream_skid_wr_ptr_q + 1'b1;
            end

            if (stream_skid_pop_w) begin
                stream_skid_rd_ptr_q <= stream_skid_rd_ptr_q + 1'b1;
                stream_consume_count_q <= stream_consume_count_q + 1'b1;
                if (stream_done_w) begin
                    stream_active_q <= 1'b0;
                    stream_raw_done_q <= 1'b0;
                end
            end

            unique case ({stream_skid_push_w, stream_skid_pop_w})
                2'b10: stream_skid_count_q <= stream_skid_count_q + 1'b1;
                2'b01: stream_skid_count_q <= stream_skid_count_q - 1'b1;
                default: stream_skid_count_q <= stream_skid_count_q;
            endcase
        end
    end
end

always_comb begin
    gemv_spm_line_rd_data = '0;
    if (gemv_line_rd_active_q) begin
        for (int bank = 0; bank < SPM_BANKS; bank++) begin
            gemv_spm_line_rd_data[bank*SPM_BANK_DATA_WIDTH +: SPM_BANK_DATA_WIDTH] = input_spm_rd_data_int[bank];
        end
    end
end

always_comb begin
    gemv_scale_line_rd_data = '0;
    if (gemv_scale_line_rd_active_q) begin
        for (int bank = 0; bank < SPM_BANKS; bank++) begin
            gemv_scale_line_rd_data[bank*SPM_BANK_DATA_WIDTH +: SPM_BANK_DATA_WIDTH] = scale_spm_rd_data_int[bank];
        end
    end
end

always_comb begin
    gemv_act_line_rd_data = '0;
    if (gemv_act_line_rd_active_q) begin
        for (int bank = 0; bank < SPM_BANKS; bank++) begin
            gemv_act_line_rd_data[bank*SPM_BANK_DATA_WIDTH +: SPM_BANK_DATA_WIDTH] = act_spm_rd_data_int[bank];
        end
    end
end

always_comb begin
    output_spm_rd_data_flat = '0;
    for (int bank = 0; bank < SPM_BANKS; bank++) begin
        output_spm_rd_data_flat[bank*SPM_BANK_DATA_WIDTH +: SPM_BANK_DATA_WIDTH] = output_spm_rd_data_int[bank];
    end
end

always_comb begin
    output_spm_wr_data = '0;
    output_spm_wr_mask = '0;
    flow_act_wr_data = '0;
    flow_act_wr_mask = '0;
    for (int bank = 0; bank < SPM_BANKS; bank++) begin
        output_spm_wr_data[bank] = gemv_spm_line_wr_data[bank*SPM_BANK_DATA_WIDTH +: SPM_BANK_DATA_WIDTH];
        output_spm_wr_mask[bank] = gemv_spm_line_wr_mask[bank*(SPM_BANK_DATA_WIDTH/8) +: (SPM_BANK_DATA_WIDTH/8)];
        flow_act_wr_data[bank] = flow_act_line_wr_data[bank*SPM_BANK_DATA_WIDTH +: SPM_BANK_DATA_WIDTH];
        flow_act_wr_mask[bank] = flow_act_line_wr_mask[bank*(SPM_BANK_DATA_WIDTH/8) +: (SPM_BANK_DATA_WIDTH/8)];
    end
end

assign mvout_axi_awready = m_axi_awready[0];
assign mvout_axi_wready = m_axi_wready[0];
assign mvout_axi_bid = m_axi_bid[0];
assign mvout_axi_bresp = m_axi_bresp[0];
assign mvout_axi_bvalid = m_axi_bvalid[0];

always_comb begin
    m_axi_awid = '0;
    m_axi_awaddr = '0;
    m_axi_awlen = '0;
    m_axi_awsize = '0;
    m_axi_awburst = '0;
    m_axi_awvalid = '0;
    m_axi_wdata = '0;
    m_axi_wstrb = '0;
    m_axi_wlast = '0;
    m_axi_wvalid = '0;
    m_axi_bready = '0;

    m_axi_awid[0] = mvout_axi_awid;
    m_axi_awaddr[0] = mvout_axi_awaddr;
    m_axi_awlen[0] = mvout_axi_awlen;
    m_axi_awsize[0] = mvout_axi_awsize;
    m_axi_awburst[0] = mvout_axi_awburst;
    m_axi_awvalid[0] = mvout_axi_awvalid;
    m_axi_wdata[0] = mvout_axi_wdata;
    m_axi_wstrb[0] = mvout_axi_wstrb;
    m_axi_wlast[0] = mvout_axi_wlast;
    m_axi_wvalid[0] = mvout_axi_wvalid;
    m_axi_bready[0] = mvout_axi_bready;
end

DMA_MVIN_4AXI #(
    .SPM_FPGA_SRAM       (SPM_FPGA_SRAM),
    .AXI_PORTS           (AXI_PORTS),
    .AXI_ID_WIDTH        (AXI_ID_WIDTH),
    .AXI_ADDR_WIDTH      (AXI_ADDR_WIDTH),
    .AXI_DATA_WIDTH      (AXI_DATA_WIDTH),
    .AXI_MAX_BURST_BEATS (AXI_MAX_BURST_BEATS),
    .AXI_MAX_OUTSTANDING (AXI_MAX_OUTSTANDING),
    .SPM_BANKS           (SPM_BANKS),
    .SPM_BANK_DATA_WIDTH (SPM_BANK_DATA_WIDTH),
    .SPM_DEPTH           (SPM_DEPTH)
) u_dma_mvin_4axi (
    .clk                (clk),
    .rst_n              (rst_n),
    .mvin_dram_base_addr(mvin_dram_base_addr),
    .mvin_spm_addr      (mvin_spm_addr),
    .mvin_line_num      (mvin_line_num),
    .stream_ring_mode   (stream_mode_req_w),
    .stream_ring_free_lines(stream_fifo_free_lines_w),
    .stream_reserve_valid(stream_dma_reserve_valid),
    .stream_reserve_lines(stream_dma_reserve_lines),
    .dma_mvin_req_en    (raw_mvin_req_w),
    .dma_mvin_resp_done (raw_mvin_done),
    .dma_mvin_busy      (raw_mvin_busy),
    .spm_wr_en          (spm_wr_en),
    .spm_wr_addr        (spm_wr_addr),
    .spm_wr_data        (spm_wr_data),
    .spm_wr_mask        (spm_wr_mask),
    .spm_wr_ready       (raw_spm_wr_ready),
    .spm_wr_linear_order(1'b0),
    .m_axi_arid         (m_axi_arid),
    .m_axi_araddr       (m_axi_araddr),
    .m_axi_arlen        (m_axi_arlen),
    .m_axi_arsize       (m_axi_arsize),
    .m_axi_arburst      (m_axi_arburst),
    .m_axi_arvalid      (m_axi_arvalid),
    .m_axi_arready      (m_axi_arready),
    .m_axi_rid          (m_axi_rid),
    .m_axi_rdata        (m_axi_rdata),
    .m_axi_rresp        (m_axi_rresp),
    .m_axi_rlast        (m_axi_rlast),
    .m_axi_rvalid       (m_axi_rvalid),
    .m_axi_rready       (m_axi_rready)
);

DMA_MVOUT_SPM_1AXI #(
    .AXI_ID_WIDTH        (AXI_ID_WIDTH),
    .AXI_ADDR_WIDTH      (AXI_ADDR_WIDTH),
    .AXI_DATA_WIDTH      (AXI_DATA_WIDTH),
    .AXI_MAX_BURST_BEATS (AXI_MAX_BURST_BEATS),
    .SPM_BANKS           (SPM_BANKS),
    .SPM_BANK_DATA_WIDTH (SPM_BANK_DATA_WIDTH),
    .SPM_DEPTH           (OUTPUT_SPM_DEPTH),
    .SPM_READ_LATENCY    (OUTPUT_SPM_READ_LATENCY)
) u_dma_mvout_spm_1axi (
    .clk                (clk),
    .rst_n              (rst_n),
    .mvout_dram_base_addr(mvout_dram_base_addr),
    .mvout_spm_addr     (mvout_spm_addr),
    .mvout_line_num     (mvout_row_num),
    .mvout_output_precision(mvout_output_precision),
    .dma_mvout_req_en   (dma_mvout_req_en),
    .dma_mvout_resp_done(dma_mvout_resp_done),
    .dma_mvout_busy     (dma_mvout_busy),
    .spm_rd_en          (output_spm_rd_en),
    .spm_rd_addr        (output_spm_rd_addr),
    .spm_rd_data        (output_spm_rd_data_flat),
    .m_axi_awid         (mvout_axi_awid),
    .m_axi_awaddr       (mvout_axi_awaddr),
    .m_axi_awlen        (mvout_axi_awlen),
    .m_axi_awsize       (mvout_axi_awsize),
    .m_axi_awburst      (mvout_axi_awburst),
    .m_axi_awvalid      (mvout_axi_awvalid),
    .m_axi_awready      (mvout_axi_awready),
    .m_axi_wdata        (mvout_axi_wdata),
    .m_axi_wstrb        (mvout_axi_wstrb),
    .m_axi_wlast        (mvout_axi_wlast),
    .m_axi_wvalid       (mvout_axi_wvalid),
    .m_axi_wready       (mvout_axi_wready),
    .m_axi_bid          (mvout_axi_bid),
    .m_axi_bresp        (mvout_axi_bresp),
    .m_axi_bvalid       (mvout_axi_bvalid),
    .m_axi_bready       (mvout_axi_bready)
);

scratchpad_pingpong_8bank #(
    .SPM_FPGA_SRAM   (SPM_FPGA_SRAM),
    .BANKS           (SPM_BANKS),
    .BUFFER_NUM      (2),
    .BANK_DATA_WIDTH (SPM_BANK_DATA_WIDTH),
    .DEPTH           (SPM_DEPTH),
    .BYTE_WRITE_ENABLE (0)
) u_input_scratchpad_4bank (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_en   (spm_wr_en && !mvin_target_act_q && !mvin_target_scale_q),
    .wr_addr (spm_wr_addr),
    .wr_data (spm_wr_data),
    .wr_mask (spm_wr_mask),
    .rd_en   (spm_rd_en_int | gemv_spm_line_rd_en | stream_reader_rd_en_w),
    .rd_addr (gemv_spm_line_rd_en ? gemv_spm_line_rd_addr_int :
              (stream_reader_rd_en_w ? stream_reader_rd_addr_w : spm_rd_addr_int)),
    .rd_data (input_spm_rd_data_int)
);

scratchpad_4bank #(
    .SPM_FPGA_SRAM   (SPM_FPGA_SRAM),
    .BANKS           (SPM_BANKS),
    .BANK_DATA_WIDTH (SPM_BANK_DATA_WIDTH),
    .DEPTH           (SCALE_SPM_DEPTH),
    .RAM_STYLE       ("ultra"),
    .BYTE_WRITE_ENABLE (1)
) u_scale_scratchpad_4bank (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_en   (spm_wr_en && mvin_target_scale_q),
    .wr_addr (scale_spm_wr_addr_int),
    .wr_data (spm_wr_data),
    .wr_mask (spm_wr_mask),
    .rd_en   (gemv_scale_line_rd_en),
    .rd_addr (gemv_scale_line_rd_addr_int),
    .rd_data (scale_spm_rd_data_int)
);

scratchpad_4bank #(
    .SPM_FPGA_SRAM   (SPM_FPGA_SRAM),
    .BANKS           (SPM_BANKS),
    .BANK_DATA_WIDTH (SPM_BANK_DATA_WIDTH),
    .DEPTH           (ACT_SPM_DEPTH),
    .RAM_STYLE       ("block"),
    .BYTE_WRITE_ENABLE (1)
) u_act_scratchpad_4bank (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_en   (flow_act_line_wr_en || (spm_wr_en && mvin_target_act_q)),
    .wr_addr (flow_act_line_wr_en ? flow_act_wr_addr : act_spm_wr_addr_int),
    .wr_data (flow_act_line_wr_en ? flow_act_wr_data : spm_wr_data),
    .wr_mask (flow_act_line_wr_en ? flow_act_wr_mask : spm_wr_mask),
    .rd_en   (gemv_act_line_rd_en),
    .rd_addr (gemv_act_line_rd_addr_narrow),
    .rd_data (act_spm_rd_data_int)
);

`ifndef SYNTHESIS
    property p_no_dma_and_flow_act_write_conflict;
        @(posedge clk) disable iff (!rst_n)
            flow_act_line_wr_en |-> !(spm_wr_en && mvin_target_act_q);
    endproperty
    assert property (p_no_dma_and_flow_act_write_conflict);

    property p_gemv_output_write_in_range;
        @(posedge clk) disable iff (!rst_n)
            gemv_spm_line_wr_en |-> (gemv_spm_line_wr_addr < OUTPUT_SPM_BYTES);
    endproperty
    assert property (p_gemv_output_write_in_range);

    property p_scale_mvin_write_in_range;
        @(posedge clk) disable iff (!rst_n)
            (spm_wr_en && mvin_target_scale_q) |-> (spm_wr_addr < SCALE_SPM_DEPTH);
    endproperty
    assert property (p_scale_mvin_write_in_range);

    property p_stream_credit_bound;
        @(posedge clk) disable iff (!rst_n)
            stream_active_q |-> ((stream_ring_occupied_q + stream_write_reserved_q) <=
                                 stream_fifo_capacity_q);
    endproperty
    assert property (p_stream_credit_bound);

    property p_stream_spm_write_has_reservation;
        @(posedge clk) disable iff (!rst_n)
            stream_spm_write_fire_w |-> (stream_write_reserved_q != '0);
    endproperty
    assert property (p_stream_spm_write_has_reservation);

    property p_stream_read_has_occupancy;
        @(posedge clk) disable iff (!rst_n)
            stream_reader_rd_en_w |-> (stream_ring_occupied_q >
                                       STREAM_RING_CNT_WIDTH'(stream_read_inflight_q));
    endproperty
    assert property (p_stream_read_has_occupancy);

    property p_stream_skid_no_overflow;
        @(posedge clk) disable iff (!rst_n)
            stream_skid_push_w |-> !stream_skid_full_w;
    endproperty
    assert property (p_stream_skid_no_overflow);

    property p_stream_no_read_port_conflict;
        @(posedge clk) disable iff (!rst_n)
            stream_reader_rd_en_w |-> (!gemv_spm_line_rd_en && !spm_rd_en_int);
    endproperty
    assert property (p_stream_no_read_port_conflict);

    property p_stream_start_in_range;
        @(posedge clk) disable iff (!rst_n)
            stream_mvin_start_w |-> (mvin_spm_addr < SPM_ADDR_WIDTH'(SPM_DEPTH));
    endproperty
    assert property (p_stream_start_in_range);

    property p_stream_request_exclusive;
        @(posedge clk) disable iff (!rst_n)
            (mvin_stream_fifo_fill && dma_mvin_req_any) |->
                (dma_mvin_req_en &&
                 !dma_scale_mvin_req_en &&
                 !dma_act_mvin_req_en);
    endproperty
    assert property (p_stream_request_exclusive);

    property p_stream_ring_has_group_capacity;
        @(posedge clk) disable iff (!rst_n)
            stream_mvin_start_w |->
                (mvin_spm_addr <= SPM_ADDR_WIDTH'(STREAM_LAST_LEGAL_BASE));
    endproperty
    assert property (p_stream_ring_has_group_capacity);

    property p_stream_reject_completes_without_raw_dma;
        @(posedge clk) disable iff (!rst_n)
            (stream_reject_req_w && !dma_mvin_busy) |=>
                (stream_reject_done_q && !raw_mvin_busy && !stream_active_q);
    endproperty
    assert property (p_stream_reject_completes_without_raw_dma);
`endif

scratchpad_4bank #(
    .SPM_FPGA_SRAM   (SPM_FPGA_SRAM),
    .BANKS           (SPM_BANKS),
    .BANK_DATA_WIDTH (SPM_BANK_DATA_WIDTH),
    .DEPTH           (OUTPUT_SPM_DEPTH),
    .RAM_STYLE       ("ultra"),
    .READ_PIPELINE   (OUTPUT_SPM_READ_PIPELINE),
    .BYTE_WRITE_ENABLE (1)
) u_output_scratchpad_4bank (
    .clk     (clk),
    .rst_n   (rst_n),
    .wr_en   (gemv_spm_line_wr_en),
    .wr_addr (gemv_spm_line_wr_addr_int),
    .wr_data (output_spm_wr_data),
    .wr_mask (output_spm_wr_mask),
    .rd_en   (output_spm_rd_en),
    .rd_addr (output_spm_rd_addr),
    .rd_data (output_spm_rd_data_int)
);

endmodule

`endif
