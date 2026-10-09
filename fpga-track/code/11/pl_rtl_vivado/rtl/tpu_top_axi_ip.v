module tpu_top_axi_ip #(
    parameter AXI_ID_WIDTH         = 4             ,
    parameter AXI_ADDR_WIDTH       = 32            ,
    parameter AXI_DATA_WIDTH       = 64            ,
    parameter AXI_AWUSER_WIDTH     = 8             ,
    parameter AXI_WUSER_WIDTH      = 8             ,
    parameter AXI_BUSER_WIDTH      = 8             ,

    parameter CSR_DATA_WIDTH       = 32            ,
    parameter CSR_ADDR_WIDTH       = 10            ,

    parameter PE_SIZE              = 8             ,
    parameter INT8_DOT_FACTOR      = 4             ,
    parameter AB_DUAL_INPUT        = 0             ,
    parameter CMAT_D_FULL_WORD_PACK = 0            ,
    parameter CMAT_STREAMING_C_ADDER = 0           ,
    parameter MATRIX_DIM_WIDTH     = 8             ,
    parameter PRECISION_MODE_WIDTH = 4             ,
    parameter RAM_ADDR_WIDTH       = 8             ,
    parameter RAM_C_ADDR_WIDTH     = 10            ,
    parameter RAM_D_ADDR_WIDTH     = 10            ,
    parameter RAM_A_LOCAL_ADDR_WIDTH = RAM_ADDR_WIDTH,
    parameter RAM_B_LOCAL_ADDR_WIDTH = 0,
    parameter RAM_D_LOCAL_ADDR_WIDTH = RAM_D_ADDR_WIDTH,
    parameter RAM_DATA_WIDTH       = 64            ,
    parameter FIFO_DATA_WIDTH      = 32            ,
    parameter FIFO_DEPTH           = 512           ,
    parameter MATRIX_A_BASE_ADDR   = 32'h0000_0000 ,
    parameter MATRIX_B_BASE_ADDR   = 32'h0000_4000 ,
    parameter MATRIX_C_BASE_ADDR   = 32'h0000_8000 ,
    parameter MATRIX_D_BASE_ADDR   = 32'h0000_C000 ,

    parameter DATA_WIDTH           = 32
) (
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK, ASSOCIATED_BUSIF S_AXI:S_AXIL:M_AXI_RD:M_AXI_RD1:M_AXI, ASSOCIATED_RESET rst_n, INSERT_VIP 0" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK CLK" *)
    input  wire                              clk,
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME RST_N, POLARITY ACTIVE_LOW, INSERT_VIP 0" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 RST_N RST" *)
    input  wire                              rst_n,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, READ_WRITE_MODE WRITE_ONLY, PROTOCOL AXI4, ADDR_WIDTH 32, ID_WIDTH 4, AWUSER_WIDTH 8, WUSER_WIDTH 8, BUSER_WIDTH 8, ARUSER_WIDTH 0, RUSER_WIDTH 0, HAS_BURST 1, HAS_LOCK 1, HAS_PROT 1, HAS_CACHE 1, HAS_QOS 1, HAS_REGION 1, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 0, SUPPORTS_NARROW_BURST 1, NUM_WRITE_OUTSTANDING 1, NUM_READ_OUTSTANDING 0, MAX_BURST_LENGTH 256, INSERT_VIP 0" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWID" *)
    input  wire [       AXI_ID_WIDTH-1:0]    s_axi_awid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [     AXI_ADDR_WIDTH-1:0]    s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWLEN" *)
    input  wire [                    7:0]    s_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWSIZE" *)
    input  wire [                    2:0]    s_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWBURST" *)
    input  wire [                    1:0]    s_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWLOCK" *)
    input  wire                             s_axi_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWCACHE" *)
    input  wire [                    3:0]   s_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire [                    2:0]   s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWQOS" *)
    input  wire [                    3:0]   s_axi_awqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREGION" *)
    input  wire [                    3:0]   s_axi_awregion,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWUSER" *)
    input  wire [   AXI_AWUSER_WIDTH-1:0]   s_axi_awuser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire                             s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire                             s_axi_awready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [       AXI_DATA_WIDTH-1:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [(   AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WLAST" *)
    input  wire                             s_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WUSER" *)
    input  wire [      AXI_WUSER_WIDTH-1:0] s_axi_wuser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire                             s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire                             s_axi_wready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BID" *)
    output wire [      AXI_ID_WIDTH-1:0]    s_axi_bid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [                   1:0]    s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BUSER" *)
    output wire [   AXI_BUSER_WIDTH-1:0]    s_axi_buser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire                             s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire                             s_axi_bready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIL, PROTOCOL AXI4LITE, READ_WRITE_MODE READ_WRITE, DATA_WIDTH 32, ADDR_WIDTH 10, ID_WIDTH 0, AWUSER_WIDTH 0, ARUSER_WIDTH 0, WUSER_WIDTH 0, RUSER_WIDTH 0, BUSER_WIDTH 0, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 1, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, SUPPORTS_NARROW_BURST 0, NUM_WRITE_OUTSTANDING 1, NUM_READ_OUTSTANDING 1, MAX_BURST_LENGTH 1, INSERT_VIP 0" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL AWADDR" *)
    input  wire [   CSR_ADDR_WIDTH-1:0]     s_axil_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL AWPROT" *)
    input  wire [                    2:0]   s_axil_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL AWVALID" *)
    input  wire                             s_axil_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL AWREADY" *)
    output wire                             s_axil_awready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL WDATA" *)
    input  wire [   CSR_DATA_WIDTH-1:0]     s_axil_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL WSTRB" *)
    input  wire [   CSR_DATA_WIDTH/8-1:0]   s_axil_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL WVALID" *)
    input  wire                             s_axil_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL WREADY" *)
    output wire                             s_axil_wready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL BRESP" *)
    output wire [                   1:0]    s_axil_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL BVALID" *)
    output wire                             s_axil_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL BREADY" *)
    input  wire                             s_axil_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL ARADDR" *)
    input  wire [   CSR_ADDR_WIDTH-1:0]     s_axil_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL ARPROT" *)
    input  wire [                    2:0]   s_axil_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL ARVALID" *)
    input  wire                             s_axil_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL ARREADY" *)
    output wire                             s_axil_arready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL RDATA" *)
    output wire [   CSR_DATA_WIDTH-1:0]     s_axil_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL RRESP" *)
    output wire [                   1:0]    s_axil_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL RVALID" *)
    output wire                             s_axil_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXIL RREADY" *)
    input  wire                             s_axil_rready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_RD, READ_WRITE_MODE READ_ONLY, PROTOCOL AXI4, ADDR_WIDTH 32, ID_WIDTH 4, AWUSER_WIDTH 0, WUSER_WIDTH 0, BUSER_WIDTH 0, ARUSER_WIDTH 0, RUSER_WIDTH 0, HAS_BURST 1, HAS_LOCK 1, HAS_PROT 1, HAS_CACHE 1, HAS_QOS 1, HAS_REGION 1, HAS_WSTRB 0, HAS_BRESP 0, HAS_RRESP 1, SUPPORTS_NARROW_BURST 1, NUM_WRITE_OUTSTANDING 0, NUM_READ_OUTSTANDING 1, MAX_BURST_LENGTH 256, INSERT_VIP 0" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARID" *)
    output wire [      AXI_ID_WIDTH-1:0]    m_axi_rd_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARADDR" *)
    output wire [    AXI_ADDR_WIDTH-1:0]    m_axi_rd_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARLEN" *)
    output wire [                   7:0]    m_axi_rd_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARSIZE" *)
    output wire [                   2:0]    m_axi_rd_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARBURST" *)
    output wire [                   1:0]    m_axi_rd_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARLOCK" *)
    output wire                            m_axi_rd_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARCACHE" *)
    output wire [                   3:0]    m_axi_rd_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARPROT" *)
    output wire [                   2:0]    m_axi_rd_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARQOS" *)
    output wire [                   3:0]    m_axi_rd_arqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARREGION" *)
    output wire [                   3:0]    m_axi_rd_arregion,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARVALID" *)
    output wire                            m_axi_rd_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD ARREADY" *)
    input  wire                            m_axi_rd_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD RID" *)
    input  wire [      AXI_ID_WIDTH-1:0]    m_axi_rd_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD RDATA" *)
    input  wire [      AXI_DATA_WIDTH-1:0]  m_axi_rd_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD RRESP" *)
    input  wire [                   1:0]    m_axi_rd_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD RLAST" *)
    input  wire                            m_axi_rd_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD RVALID" *)
    input  wire                            m_axi_rd_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD RREADY" *)
    output wire                            m_axi_rd_rready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_RD1, READ_WRITE_MODE READ_ONLY, PROTOCOL AXI4, ADDR_WIDTH 32, ID_WIDTH 4, AWUSER_WIDTH 0, WUSER_WIDTH 0, BUSER_WIDTH 0, ARUSER_WIDTH 0, RUSER_WIDTH 0, HAS_BURST 1, HAS_LOCK 1, HAS_PROT 1, HAS_CACHE 1, HAS_QOS 1, HAS_REGION 1, HAS_WSTRB 0, HAS_BRESP 0, HAS_RRESP 1, SUPPORTS_NARROW_BURST 1, NUM_WRITE_OUTSTANDING 0, NUM_READ_OUTSTANDING 1, MAX_BURST_LENGTH 256, INSERT_VIP 0" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARID" *)
    output wire [      AXI_ID_WIDTH-1:0]    m_axi_rd1_arid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARADDR" *)
    output wire [    AXI_ADDR_WIDTH-1:0]    m_axi_rd1_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARLEN" *)
    output wire [                   7:0]    m_axi_rd1_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARSIZE" *)
    output wire [                   2:0]    m_axi_rd1_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARBURST" *)
    output wire [                   1:0]    m_axi_rd1_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARLOCK" *)
    output wire                            m_axi_rd1_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARCACHE" *)
    output wire [                   3:0]    m_axi_rd1_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARPROT" *)
    output wire [                   2:0]    m_axi_rd1_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARQOS" *)
    output wire [                   3:0]    m_axi_rd1_arqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARREGION" *)
    output wire [                   3:0]    m_axi_rd1_arregion,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARVALID" *)
    output wire                            m_axi_rd1_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 ARREADY" *)
    input  wire                            m_axi_rd1_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 RID" *)
    input  wire [      AXI_ID_WIDTH-1:0]    m_axi_rd1_rid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 RDATA" *)
    input  wire [      AXI_DATA_WIDTH-1:0]  m_axi_rd1_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 RRESP" *)
    input  wire [                   1:0]    m_axi_rd1_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 RLAST" *)
    input  wire                            m_axi_rd1_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 RVALID" *)
    input  wire                            m_axi_rd1_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_RD1 RREADY" *)
    output wire                            m_axi_rd1_rready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI, READ_WRITE_MODE WRITE_ONLY, PROTOCOL AXI4, ADDR_WIDTH 32, ID_WIDTH 4, AWUSER_WIDTH 8, WUSER_WIDTH 8, BUSER_WIDTH 8, ARUSER_WIDTH 0, RUSER_WIDTH 0, HAS_BURST 1, HAS_LOCK 1, HAS_PROT 1, HAS_CACHE 1, HAS_QOS 1, HAS_REGION 1, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 0, SUPPORTS_NARROW_BURST 1, NUM_WRITE_OUTSTANDING 1, NUM_READ_OUTSTANDING 0, MAX_BURST_LENGTH 256, INSERT_VIP 0" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWID" *)
    output wire [      AXI_ID_WIDTH-1:0]    m_axi_awid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *)
    output wire [    AXI_ADDR_WIDTH-1:0]    m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *)
    output wire [                   7:0]    m_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *)
    output wire [                   2:0]    m_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *)
    output wire [                   1:0]    m_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLOCK" *)
    output wire                            m_axi_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWCACHE" *)
    output wire [                   3:0]    m_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWPROT" *)
    output wire [                   2:0]    m_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWQOS" *)
    output wire [                   3:0]    m_axi_awqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREGION" *)
    output wire [                   3:0]    m_axi_awregion,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWUSER" *)
    output wire [  AXI_AWUSER_WIDTH-1:0]    m_axi_awuser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *)
    output wire                            m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *)
    input  wire                            m_axi_awready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WDATA" *)
    output wire [      AXI_DATA_WIDTH-1:0]  m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *)
    output wire [  (AXI_DATA_WIDTH/8)-1:0]  m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WLAST" *)
    output wire                            m_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WUSER" *)
    output wire [     AXI_WUSER_WIDTH-1:0]  m_axi_wuser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WVALID" *)
    output wire                            m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WREADY" *)
    input  wire                            m_axi_wready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BID" *)
    input  wire [      AXI_ID_WIDTH-1:0]    m_axi_bid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BRESP" *)
    input  wire [                   1:0]    m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BUSER" *)
    input  wire [   AXI_BUSER_WIDTH-1:0]    m_axi_buser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BVALID" *)
    input  wire                            m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BREADY" *)
    output wire                            m_axi_bready,

    output wire [                  31:0]   ila_prevd_ctrl,
    output wire [                  31:0]   ila_prevd_data
);

    wire unused_s_axil_awprot;
    wire unused_s_axil_arprot;

    assign unused_s_axil_awprot = ^s_axil_awprot;
    assign unused_s_axil_arprot = ^s_axil_arprot;

    tpu_top #(
        .AXI_ID_WIDTH         (AXI_ID_WIDTH),
        .AXI_ADDR_WIDTH       (AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH       (AXI_DATA_WIDTH),
        .AXI_AWUSER_WIDTH     (AXI_AWUSER_WIDTH),
        .AXI_WUSER_WIDTH      (AXI_WUSER_WIDTH),
        .AXI_BUSER_WIDTH      (AXI_BUSER_WIDTH),
        .CSR_DATA_WIDTH       (CSR_DATA_WIDTH),
        .CSR_ADDR_WIDTH       (CSR_ADDR_WIDTH),
        .PE_SIZE              (PE_SIZE),
        .INT8_DOT_FACTOR      (INT8_DOT_FACTOR),
        .AB_DUAL_INPUT        (AB_DUAL_INPUT),
        .CMAT_D_FULL_WORD_PACK(CMAT_D_FULL_WORD_PACK),
        .CMAT_STREAMING_C_ADDER(CMAT_STREAMING_C_ADDER),
        .MATRIX_DIM_WIDTH     (MATRIX_DIM_WIDTH),
        .PRECISION_MODE_WIDTH (PRECISION_MODE_WIDTH),
        .RAM_ADDR_WIDTH       (RAM_ADDR_WIDTH),
        .RAM_C_ADDR_WIDTH     (RAM_C_ADDR_WIDTH),
        .RAM_D_ADDR_WIDTH     (RAM_D_ADDR_WIDTH),
        .RAM_A_LOCAL_ADDR_WIDTH(RAM_A_LOCAL_ADDR_WIDTH),
        .RAM_B_LOCAL_ADDR_WIDTH(RAM_B_LOCAL_ADDR_WIDTH),
        .RAM_D_LOCAL_ADDR_WIDTH(RAM_D_LOCAL_ADDR_WIDTH),
        .RAM_DATA_WIDTH       (RAM_DATA_WIDTH),
        .FIFO_DATA_WIDTH      (FIFO_DATA_WIDTH),
        .FIFO_DEPTH           (FIFO_DEPTH),
        .MATRIX_A_BASE_ADDR   (MATRIX_A_BASE_ADDR),
        .MATRIX_B_BASE_ADDR   (MATRIX_B_BASE_ADDR),
        .MATRIX_C_BASE_ADDR   (MATRIX_C_BASE_ADDR),
        .MATRIX_D_BASE_ADDR   (MATRIX_D_BASE_ADDR),
        .DATA_WIDTH           (DATA_WIDTH)
    ) tpu_top_inst (
        .clk            (clk),
        .rst_n          (rst_n),

        .s_axi_awid     (s_axi_awid),
        .s_axi_awaddr   (s_axi_awaddr),
        .s_axi_awlen    (s_axi_awlen),
        .s_axi_awsize   (s_axi_awsize),
        .s_axi_awburst  (s_axi_awburst),
        .s_axi_awlock   (s_axi_awlock),
        .s_axi_awcache  (s_axi_awcache),
        .s_axi_awprot   (s_axi_awprot),
        .s_axi_awqos    (s_axi_awqos),
        .s_axi_awregion (s_axi_awregion),
        .s_axi_awuser   (s_axi_awuser),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wlast    (s_axi_wlast),
        .s_axi_wuser    (s_axi_wuser),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),
        .s_axi_bid      (s_axi_bid),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_buser    (s_axi_buser),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),

        .s_axil_awaddr  (s_axil_awaddr),
        .s_axil_awvalid (s_axil_awvalid),
        .s_axil_awready (s_axil_awready),
        .s_axil_wdata   (s_axil_wdata),
        .s_axil_wstrb   (s_axil_wstrb),
        .s_axil_wvalid  (s_axil_wvalid),
        .s_axil_wready  (s_axil_wready),
        .s_axil_bresp   (s_axil_bresp),
        .s_axil_bvalid  (s_axil_bvalid),
        .s_axil_bready  (s_axil_bready),
        .s_axil_araddr  (s_axil_araddr),
        .s_axil_arvalid (s_axil_arvalid),
        .s_axil_arready (s_axil_arready),
        .s_axil_rdata   (s_axil_rdata),
        .s_axil_rresp   (s_axil_rresp),
        .s_axil_rvalid  (s_axil_rvalid),
        .s_axil_rready  (s_axil_rready),

        .m_axi_rd_arid  (m_axi_rd_arid),
        .m_axi_rd_araddr(m_axi_rd_araddr),
        .m_axi_rd_arlen (m_axi_rd_arlen),
        .m_axi_rd_arsize(m_axi_rd_arsize),
        .m_axi_rd_arburst(m_axi_rd_arburst),
        .m_axi_rd_arlock(m_axi_rd_arlock),
        .m_axi_rd_arcache(m_axi_rd_arcache),
        .m_axi_rd_arprot(m_axi_rd_arprot),
        .m_axi_rd_arqos (m_axi_rd_arqos),
        .m_axi_rd_arregion(m_axi_rd_arregion),
        .m_axi_rd_arvalid(m_axi_rd_arvalid),
        .m_axi_rd_arready(m_axi_rd_arready),
        .m_axi_rd_rid   (m_axi_rd_rid),
        .m_axi_rd_rdata (m_axi_rd_rdata),
        .m_axi_rd_rresp (m_axi_rd_rresp),
        .m_axi_rd_rlast (m_axi_rd_rlast),
        .m_axi_rd_rvalid(m_axi_rd_rvalid),
        .m_axi_rd_rready(m_axi_rd_rready),
        .m_axi_rd1_arid (m_axi_rd1_arid),
        .m_axi_rd1_araddr(m_axi_rd1_araddr),
        .m_axi_rd1_arlen(m_axi_rd1_arlen),
        .m_axi_rd1_arsize(m_axi_rd1_arsize),
        .m_axi_rd1_arburst(m_axi_rd1_arburst),
        .m_axi_rd1_arlock(m_axi_rd1_arlock),
        .m_axi_rd1_arcache(m_axi_rd1_arcache),
        .m_axi_rd1_arprot(m_axi_rd1_arprot),
        .m_axi_rd1_arqos(m_axi_rd1_arqos),
        .m_axi_rd1_arregion(m_axi_rd1_arregion),
        .m_axi_rd1_arvalid(m_axi_rd1_arvalid),
        .m_axi_rd1_arready(m_axi_rd1_arready),
        .m_axi_rd1_rid  (m_axi_rd1_rid),
        .m_axi_rd1_rdata(m_axi_rd1_rdata),
        .m_axi_rd1_rresp(m_axi_rd1_rresp),
        .m_axi_rd1_rlast(m_axi_rd1_rlast),
        .m_axi_rd1_rvalid(m_axi_rd1_rvalid),
        .m_axi_rd1_rready(m_axi_rd1_rready),

        .m_axi_awid     (m_axi_awid),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awlock   (m_axi_awlock),
        .m_axi_awcache  (m_axi_awcache),
        .m_axi_awprot   (m_axi_awprot),
        .m_axi_awqos    (m_axi_awqos),
        .m_axi_awregion (m_axi_awregion),
        .m_axi_awuser   (m_axi_awuser),
        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),
        .m_axi_wuser    (m_axi_wuser),
        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_bid      (m_axi_bid),
        .m_axi_bresp    (m_axi_bresp),
        .m_axi_buser    (m_axi_buser),
        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready),
        .ila_prevd_ctrl (ila_prevd_ctrl),
        .ila_prevd_data (ila_prevd_data)
    );

endmodule
