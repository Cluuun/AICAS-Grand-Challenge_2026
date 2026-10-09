// addr都是字节地址
// 不能同时读写，仲裁
// 最大支持512*8bit位宽（512byte），宽度需要是2的幂
// 必须地址对齐
// burst支持incr，最大burst长度65535
module axi_dram # (
    parameter int unsigned      ADDR_WIDTH = 32'd64,
    parameter int unsigned      DATA_WIDTH = 32'd512,
    parameter longint unsigned  ADDR_BASE  = 64'h80000000,
    parameter               DRAMType       = "DDR4",    //DRAM type
    parameter               CustomerDRAM   = "none",    //DRAM type
    parameter               InitPath       = "none",    //mem path
	parameter time          ClkPeriod	   = 1ns,
    parameter int unsigned      AXI_WR_OUTSTANDING = 32'd8,
    parameter int unsigned      AXI_RD_OUTSTANDING = 32'd8,
    
    localparam int unsigned     DRAM_LINEWIDTH  = `DRAM_WIDTH // ddr config
) (
    input    clk_i,
    input    rst_ni,

    input                           axi_mem_awvalid_i,
    output                          axi_mem_awready_o,
    input [ADDR_WIDTH-1:0]          axi_mem_awaddr_i,
    input [3:0]                     axi_mem_awid_i,
    input [15:0]                    axi_mem_awlen_i,
    input [1:0]                     axi_mem_awburst_i,

    input                           axi_mem_wvalid_i,
    output                          axi_mem_wready_o,
    input [DATA_WIDTH-1:0]          axi_mem_wdata_i,
    input [DATA_WIDTH/8-1:0]        axi_mem_wstrb_i,
    input                           axi_mem_wlast_i,

    output                          axi_mem_bvalid_o,
    input                           axi_mem_bready_i,
    output [1:0]                    axi_mem_bresp_o,
    output [3:0]                    axi_mem_bid_o,
    
    input                           axi_mem_arvalid_i,
    output                          axi_mem_arready_o,
    input [ADDR_WIDTH-1:0]          axi_mem_araddr_i,
    input [3:0]                     axi_mem_arid_i,
    input [15:0]                    axi_mem_arlen_i,
    input [1:0]                     axi_mem_arburst_i,

    output                          axi_mem_rvalid_o,
    input                           axi_mem_rready_i,
    output [DATA_WIDTH-1:0]         axi_mem_rdata_o,
    output [1:0]                    axi_mem_rresp_o,
    output [3:0]                    axi_mem_rid_o,
    output                          axi_mem_rlast_o
);

typedef struct packed {
    logic [ADDR_WIDTH-1:0]          awaddr;
    logic [3:0]                     awid;
    logic [15:0]                    awlen;
    logic [1:0]                     awburst;
} axi_write_req;

typedef struct packed {
    logic [ADDR_WIDTH-1:0]          araddr;
    logic [3:0]                     arid;
    logic [15:0]                    arlen;
    logic [1:0]                     arburst;
} axi_read_req;

localparam axi_write_req_len = $bits(axi_write_req);
localparam axi_read_req_len  = $bits(axi_read_req);

logic                       axi_awvalid;
logic [ADDR_WIDTH-1:0]      axi_awaddr;
logic [3:0]                 axi_awid;
logic [15:0]                axi_awlen;
logic [1:0]                 axi_awburst;
logic                       axi_awready;

logic                       axi_wvalid;
logic [DATA_WIDTH-1:0]      axi_wdata;
logic [DATA_WIDTH/8-1:0]    axi_wstrb;
logic                       axi_wlast;
logic                       axi_wready;
        
logic                       axi_bready;
logic                       axi_bvalid;
logic [1:0]                 axi_bresp;
logic [3:0]                 axi_bid;

logic                       axi_arvalid;
logic [ADDR_WIDTH-1:0]      axi_araddr;
logic [3:0]                 axi_arid;
logic [15:0]                axi_arlen;
logic [1:0]                 axi_arburst;
logic                       axi_arready;

logic                       axi_rready;
logic                       axi_rvalid;
logic [DATA_WIDTH-1:0]      axi_rdata;
logic [1:0]                 axi_rresp;
logic [3:0]                 axi_rid;
logic                       axi_rlast;

assign axi_wvalid       = axi_mem_wvalid_i;
assign axi_wdata        = axi_mem_wdata_i;
assign axi_wstrb        = axi_mem_wstrb_i;
assign axi_wlast        = axi_mem_wlast_i;
assign axi_mem_wready_o = axi_wready;

assign axi_bready       = axi_mem_bready_i;
assign axi_mem_bvalid_o = axi_bvalid;
assign axi_mem_bresp_o  = axi_bresp;
assign axi_mem_bid_o    = axi_bid;

assign axi_rready       = axi_mem_rready_i;
assign axi_mem_rvalid_o = axi_rvalid;
assign axi_mem_rdata_o  = axi_rdata;
assign axi_mem_rresp_o  = axi_rresp;
assign axi_mem_rid_o    = axi_rid;
assign axi_mem_rlast_o  = axi_rlast;

////////////////////////////////////////////////
// write                                      //
////////////////////////////////////////////////
logic               wr_fifo_valid_i;
logic               wr_fifo_ready;
logic               wr_fifo_valid_o;
logic               wr_fifo_read;
axi_write_req       wr_fifo_data_i;
axi_write_req       wr_fifo_data_o;

assign wr_fifo_valid_i = axi_mem_awvalid_i & wr_fifo_ready;
assign wr_fifo_data_i = '{
        awaddr      : axi_mem_awaddr_i,
        awid        : axi_mem_awid_i,
        awlen       : axi_mem_awlen_i,
        awburst     : axi_mem_awburst_i
};
assign wr_fifo_read = wr_fifo_valid_o & axi_awready;

assign axi_mem_awready_o    = wr_fifo_ready;
assign axi_awvalid          = wr_fifo_valid_o;
assign axi_awaddr           = wr_fifo_data_o.awaddr;
assign axi_awid             = wr_fifo_data_o.awid;
assign axi_awlen            = wr_fifo_data_o.awlen;
assign axi_awburst          = wr_fifo_data_o.awburst;

fifo_sync #(
    .width_p ( axi_write_req_len        ),
    .depth_p ( AXI_WR_OUTSTANDING       )
) write_outstanding_buffer (
    .clk_i   ( clk_i                    ),
    .rst_ni  ( rst_ni                   ),
    .valid_i ( wr_fifo_valid_i          ),
    .data_i  ( wr_fifo_data_i           ),
    .ready_o ( wr_fifo_ready            ),
    .valid_o ( wr_fifo_valid_o          ),
    .data_o  ( wr_fifo_data_o           ),
    .read_i  ( wr_fifo_read             )
);

////////////////////////////////////////////////
// read                                       //
////////////////////////////////////////////////
logic               rd_fifo_valid_i;
logic               rd_fifo_ready;
logic               rd_fifo_valid_o;
logic               rd_fifo_read;
axi_read_req        rd_fifo_data_i;
axi_read_req        rd_fifo_data_o; 

assign rd_fifo_valid_i = axi_mem_arvalid_i & rd_fifo_ready;
assign rd_fifo_data_i = '{
        araddr      : axi_mem_araddr_i,
        arid        : axi_mem_arid_i,
        arlen       : axi_mem_arlen_i,
        arburst     : axi_mem_arburst_i
};
assign rd_fifo_read = rd_fifo_valid_o & axi_arready;

assign axi_mem_arready_o    = rd_fifo_ready;
assign axi_arvalid          = rd_fifo_valid_o;
assign axi_araddr           = rd_fifo_data_o.araddr;
assign axi_arid             = rd_fifo_data_o.arid;
assign axi_arlen            = rd_fifo_data_o.arlen;
assign axi_arburst          = rd_fifo_data_o.arburst;

fifo_sync #(
    .width_p ( axi_read_req_len        ),
    .depth_p ( AXI_RD_OUTSTANDING       )
) read_outstanding_buffer (
    .clk_i   ( clk_i                    ),
    .rst_ni  ( rst_ni                   ),
    .valid_i ( rd_fifo_valid_i          ),
    .data_i  ( rd_fifo_data_i           ),
    .ready_o ( rd_fifo_ready            ),
    .valid_o ( rd_fifo_valid_o          ),
    .data_o  ( rd_fifo_data_o           ),
    .read_i  ( rd_fifo_read             )
);

////////////////////////////////////////////////
// axi interface                              //
////////////////////////////////////////////////
axi_interface #(
    .ADDR_WIDTH             ( ADDR_WIDTH            ),
    .DATA_WIDTH             ( DATA_WIDTH            ),
    .ADDR_BASE              ( ADDR_BASE             ),
    .DRAMType               ( DRAMType              ),
    .CustomerDRAM           ( CustomerDRAM          ),
    .InitPath               ( InitPath              ),
    .ClkPeriod              ( ClkPeriod             ),
    .DRAM_LINEWIDTH         ( DRAM_LINEWIDTH        )
) axi_if (
    .clk_i                  ( clk_i                 ),
    .rst_ni                 ( rst_ni                ),
    .axi_mem_awvalid_i      ( axi_awvalid           ),
    .axi_mem_awaddr_i       ( axi_awaddr            ),
    .axi_mem_awid_i         ( axi_awid              ),
    .axi_mem_awlen_i        ( axi_awlen             ),
    .axi_mem_awburst_i      ( axi_awburst           ),
    .axi_mem_awready_o      ( axi_awready           ),
    .axi_mem_wvalid_i       ( axi_wvalid            ),
    .axi_mem_wdata_i        ( axi_wdata             ),
    .axi_mem_wstrb_i        ( axi_wstrb             ),
    .axi_mem_wlast_i        ( axi_wlast             ),
    .axi_mem_wready_o       ( axi_wready            ),
    .axi_mem_bready_i       ( axi_bready            ),
    .axi_mem_bvalid_o       ( axi_bvalid            ),
    .axi_mem_bresp_o        ( axi_bresp             ),
    .axi_mem_bid_o          ( axi_bid               ),
    .axi_mem_arvalid_i      ( axi_arvalid           ),
    .axi_mem_araddr_i       ( axi_araddr            ),
    .axi_mem_arid_i         ( axi_arid              ),
    .axi_mem_arlen_i        ( axi_arlen             ),
    .axi_mem_arburst_i      ( axi_arburst           ),
    .axi_mem_arready_o      ( axi_arready           ),
    .axi_mem_rready_i       ( axi_rready            ),
    .axi_mem_rvalid_o       ( axi_rvalid            ),
    .axi_mem_rdata_o        ( axi_rdata             ),
    .axi_mem_rresp_o        ( axi_rresp             ),
    .axi_mem_rid_o          ( axi_rid               ),
    .axi_mem_rlast_o        ( axi_rlast             )
);

endmodule