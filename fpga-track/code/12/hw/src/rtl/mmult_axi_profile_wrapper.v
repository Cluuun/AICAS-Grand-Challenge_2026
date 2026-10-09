`timescale 1 ns / 1 ps

`define AXIL_SLAVE_PORTS(prefix, ADDR_W, DATA_W) \
    input  wire                              prefix``_AWVALID, \
    output wire                              prefix``_AWREADY, \
    input  wire [(ADDR_W)-1:0]               prefix``_AWADDR, \
    input  wire                              prefix``_WVALID, \
    output wire                              prefix``_WREADY, \
    input  wire [(DATA_W)-1:0]               prefix``_WDATA, \
    input  wire [((DATA_W)/8)-1:0]           prefix``_WSTRB, \
    input  wire                              prefix``_ARVALID, \
    output wire                              prefix``_ARREADY, \
    input  wire [(ADDR_W)-1:0]               prefix``_ARADDR, \
    output wire                              prefix``_RVALID, \
    input  wire                              prefix``_RREADY, \
    output wire [(DATA_W)-1:0]               prefix``_RDATA, \
    output wire [1:0]                        prefix``_RRESP, \
    output wire                              prefix``_BVALID, \
    input  wire                              prefix``_BREADY, \
    output wire [1:0]                        prefix``_BRESP

`define AXIL_MASTER_PORTS(prefix, ADDR_W, DATA_W) \
    output wire                              prefix``_AWVALID, \
    input  wire                              prefix``_AWREADY, \
    output wire [(ADDR_W)-1:0]               prefix``_AWADDR, \
    output wire                              prefix``_WVALID, \
    input  wire                              prefix``_WREADY, \
    output wire [(DATA_W)-1:0]               prefix``_WDATA, \
    output wire [((DATA_W)/8)-1:0]           prefix``_WSTRB, \
    output wire                              prefix``_ARVALID, \
    input  wire                              prefix``_ARREADY, \
    output wire [(ADDR_W)-1:0]               prefix``_ARADDR, \
    input  wire                              prefix``_RVALID, \
    output wire                              prefix``_RREADY, \
    input  wire [(DATA_W)-1:0]               prefix``_RDATA, \
    input  wire [1:0]                        prefix``_RRESP, \
    input  wire                              prefix``_BVALID, \
    output wire                              prefix``_BREADY, \
    input  wire [1:0]                        prefix``_BRESP

`define AXIMM_SLAVE_PORTS(prefix, ID_W, ADDR_W, DATA_W, WSTRB_W, USER_W) \
    input  wire                              prefix``_AWVALID, \
    output wire                              prefix``_AWREADY, \
    input  wire [(ADDR_W)-1:0]               prefix``_AWADDR, \
    input  wire [(ID_W)-1:0]                 prefix``_AWID, \
    input  wire [7:0]                        prefix``_AWLEN, \
    input  wire [2:0]                        prefix``_AWSIZE, \
    input  wire [1:0]                        prefix``_AWBURST, \
    input  wire [1:0]                        prefix``_AWLOCK, \
    input  wire [3:0]                        prefix``_AWCACHE, \
    input  wire [2:0]                        prefix``_AWPROT, \
    input  wire [3:0]                        prefix``_AWQOS, \
    input  wire [3:0]                        prefix``_AWREGION, \
    input  wire [(USER_W)-1:0]               prefix``_AWUSER, \
    input  wire                              prefix``_WVALID, \
    output wire                              prefix``_WREADY, \
    input  wire [(DATA_W)-1:0]               prefix``_WDATA, \
    input  wire [(WSTRB_W)-1:0]              prefix``_WSTRB, \
    input  wire                              prefix``_WLAST, \
    input  wire [(ID_W)-1:0]                 prefix``_WID, \
    input  wire [(USER_W)-1:0]               prefix``_WUSER, \
    input  wire                              prefix``_ARVALID, \
    output wire                              prefix``_ARREADY, \
    input  wire [(ADDR_W)-1:0]               prefix``_ARADDR, \
    input  wire [(ID_W)-1:0]                 prefix``_ARID, \
    input  wire [7:0]                        prefix``_ARLEN, \
    input  wire [2:0]                        prefix``_ARSIZE, \
    input  wire [1:0]                        prefix``_ARBURST, \
    input  wire [1:0]                        prefix``_ARLOCK, \
    input  wire [3:0]                        prefix``_ARCACHE, \
    input  wire [2:0]                        prefix``_ARPROT, \
    input  wire [3:0]                        prefix``_ARQOS, \
    input  wire [3:0]                        prefix``_ARREGION, \
    input  wire [(USER_W)-1:0]               prefix``_ARUSER, \
    output wire                              prefix``_RVALID, \
    input  wire                              prefix``_RREADY, \
    output wire [(DATA_W)-1:0]               prefix``_RDATA, \
    output wire                              prefix``_RLAST, \
    output wire [(ID_W)-1:0]                 prefix``_RID, \
    output wire [(USER_W)-1:0]               prefix``_RUSER, \
    output wire [1:0]                        prefix``_RRESP, \
    output wire                              prefix``_BVALID, \
    input  wire                              prefix``_BREADY, \
    output wire [1:0]                        prefix``_BRESP, \
    output wire [(ID_W)-1:0]                 prefix``_BID, \
    output wire [(USER_W)-1:0]               prefix``_BUSER

`define AXIMM_MASTER_PORTS(prefix, ID_W, ADDR_W, DATA_W, WSTRB_W, USER_W) \
    output wire                              prefix``_AWVALID, \
    input  wire                              prefix``_AWREADY, \
    output wire [(ADDR_W)-1:0]               prefix``_AWADDR, \
    output wire [(ID_W)-1:0]                 prefix``_AWID, \
    output wire [7:0]                        prefix``_AWLEN, \
    output wire [2:0]                        prefix``_AWSIZE, \
    output wire [1:0]                        prefix``_AWBURST, \
    output wire [1:0]                        prefix``_AWLOCK, \
    output wire [3:0]                        prefix``_AWCACHE, \
    output wire [2:0]                        prefix``_AWPROT, \
    output wire [3:0]                        prefix``_AWQOS, \
    output wire [3:0]                        prefix``_AWREGION, \
    output wire [(USER_W)-1:0]               prefix``_AWUSER, \
    output wire                              prefix``_WVALID, \
    input  wire                              prefix``_WREADY, \
    output wire [(DATA_W)-1:0]               prefix``_WDATA, \
    output wire [(WSTRB_W)-1:0]              prefix``_WSTRB, \
    output wire                              prefix``_WLAST, \
    output wire [(ID_W)-1:0]                 prefix``_WID, \
    output wire [(USER_W)-1:0]               prefix``_WUSER, \
    output wire                              prefix``_ARVALID, \
    input  wire                              prefix``_ARREADY, \
    output wire [(ADDR_W)-1:0]               prefix``_ARADDR, \
    output wire [(ID_W)-1:0]                 prefix``_ARID, \
    output wire [7:0]                        prefix``_ARLEN, \
    output wire [2:0]                        prefix``_ARSIZE, \
    output wire [1:0]                        prefix``_ARBURST, \
    output wire [1:0]                        prefix``_ARLOCK, \
    output wire [3:0]                        prefix``_ARCACHE, \
    output wire [2:0]                        prefix``_ARPROT, \
    output wire [3:0]                        prefix``_ARQOS, \
    output wire [3:0]                        prefix``_ARREGION, \
    output wire [(USER_W)-1:0]               prefix``_ARUSER, \
    input  wire                              prefix``_RVALID, \
    output wire                              prefix``_RREADY, \
    input  wire [(DATA_W)-1:0]               prefix``_RDATA, \
    input  wire                              prefix``_RLAST, \
    input  wire [(ID_W)-1:0]                 prefix``_RID, \
    input  wire [(USER_W)-1:0]               prefix``_RUSER, \
    input  wire [1:0]                        prefix``_RRESP, \
    input  wire                              prefix``_BVALID, \
    output wire                              prefix``_BREADY, \
    input  wire [1:0]                        prefix``_BRESP, \
    input  wire [(ID_W)-1:0]                 prefix``_BID, \
    input  wire [(USER_W)-1:0]               prefix``_BUSER

`define AXIL_PASS_THROUGH(slave, master) \
    assign master``_AWVALID = slave``_AWVALID; \
    assign slave``_AWREADY = master``_AWREADY; \
    assign master``_AWADDR  = slave``_AWADDR;  \
    assign master``_WVALID  = slave``_WVALID;  \
    assign slave``_WREADY   = master``_WREADY; \
    assign master``_WDATA   = slave``_WDATA;   \
    assign master``_WSTRB   = slave``_WSTRB;   \
    assign master``_ARVALID = slave``_ARVALID; \
    assign slave``_ARREADY  = master``_ARREADY; \
    assign master``_ARADDR  = slave``_ARADDR;  \
    assign slave``_RVALID   = master``_RVALID; \
    assign master``_RREADY  = slave``_RREADY;  \
    assign slave``_RDATA    = master``_RDATA;  \
    assign slave``_RRESP    = master``_RRESP;  \
    assign slave``_BVALID   = master``_BVALID; \
    assign master``_BREADY  = slave``_BREADY;  \
    assign slave``_BRESP    = master``_BRESP

`define AXIMM_PASS_THROUGH(slave, master) \
    assign master``_AWVALID = slave``_AWVALID; \
    assign slave``_AWREADY  = master``_AWREADY; \
    assign master``_AWADDR  = slave``_AWADDR; \
    assign master``_AWID    = slave``_AWID; \
    assign master``_AWLEN   = slave``_AWLEN; \
    assign master``_AWSIZE  = slave``_AWSIZE; \
    assign master``_AWBURST = slave``_AWBURST; \
    assign master``_AWLOCK  = slave``_AWLOCK; \
    assign master``_AWCACHE = slave``_AWCACHE; \
    assign master``_AWPROT  = slave``_AWPROT; \
    assign master``_AWQOS   = slave``_AWQOS; \
    assign master``_AWREGION = slave``_AWREGION; \
    assign master``_AWUSER  = slave``_AWUSER; \
    assign master``_WVALID  = slave``_WVALID; \
    assign slave``_WREADY   = master``_WREADY; \
    assign master``_WDATA   = slave``_WDATA; \
    assign master``_WSTRB   = slave``_WSTRB; \
    assign master``_WLAST   = slave``_WLAST; \
    assign master``_WID     = slave``_WID; \
    assign master``_WUSER   = slave``_WUSER; \
    assign master``_ARVALID = slave``_ARVALID; \
    assign slave``_ARREADY  = master``_ARREADY; \
    assign master``_ARADDR  = slave``_ARADDR; \
    assign master``_ARID    = slave``_ARID; \
    assign master``_ARLEN   = slave``_ARLEN; \
    assign master``_ARSIZE  = slave``_ARSIZE; \
    assign master``_ARBURST = slave``_ARBURST; \
    assign master``_ARLOCK  = slave``_ARLOCK; \
    assign master``_ARCACHE = slave``_ARCACHE; \
    assign master``_ARPROT  = slave``_ARPROT; \
    assign master``_ARQOS   = slave``_ARQOS; \
    assign master``_ARREGION = slave``_ARREGION; \
    assign master``_ARUSER  = slave``_ARUSER; \
    assign slave``_RVALID   = master``_RVALID; \
    assign master``_RREADY  = slave``_RREADY; \
    assign slave``_RDATA    = master``_RDATA; \
    assign slave``_RLAST    = master``_RLAST; \
    assign slave``_RID      = master``_RID; \
    assign slave``_RUSER    = master``_RUSER; \
    assign slave``_RRESP    = master``_RRESP; \
    assign slave``_BVALID   = master``_BVALID; \
    assign master``_BREADY  = slave``_BREADY; \
    assign slave``_BRESP    = master``_BRESP; \
    assign slave``_BID      = master``_BID; \
    assign slave``_BUSER    = master``_BUSER

module mmult_axi_profile_wrapper #
(
    parameter integer C_S_AXI_CONTROL_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_CONTROL_DATA_WIDTH = 32,
    parameter integer C_S_AXI_PROFILE_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_PROFILE_DATA_WIDTH = 32,
    parameter integer C_M_AXI_CONTROL_ADDR_WIDTH = 8,
    parameter integer C_M_AXI_CONTROL_DATA_WIDTH = 32,
    parameter integer C_AXI_GMEM_ID_WIDTH = 1,
    parameter integer C_AXI_GMEM_ADDR_WIDTH = 64,
    parameter integer C_AXI_GMEM_DATA_WIDTH = 512,
    parameter integer C_AXI_GMEM_WSTRB_WIDTH = 64,
    parameter integer C_AXI_GMEM_USER_WIDTH = 1
)
(
    // Pass-through wrapper around the HLS AXI interfaces. It keeps the
    // functional datapath untouched and only observes protocol-level activity.
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_clk, ASSOCIATED_BUSIF s_axi_control:m_axi_control:s_axi_profile:s_axi_gmem_a:m_axi_gmem_a:s_axi_gmem_b:m_axi_gmem_b:s_axi_gmem_c:m_axi_gmem_c, ASSOCIATED_RESET ap_rst_n, FREQ_HZ 250000000" *)
    input  wire ap_clk,
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_rst_n, POLARITY ACTIVE_LOW" *)
    input  wire ap_rst_n,
    `AXIL_SLAVE_PORTS(s_axi_control, C_S_AXI_CONTROL_ADDR_WIDTH, C_S_AXI_CONTROL_DATA_WIDTH),
    `AXIL_MASTER_PORTS(m_axi_control, C_M_AXI_CONTROL_ADDR_WIDTH, C_M_AXI_CONTROL_DATA_WIDTH),
    `AXIL_SLAVE_PORTS(s_axi_profile, C_S_AXI_PROFILE_ADDR_WIDTH, C_S_AXI_PROFILE_DATA_WIDTH),
    `AXIMM_SLAVE_PORTS(s_axi_gmem_a, C_AXI_GMEM_ID_WIDTH, C_AXI_GMEM_ADDR_WIDTH, C_AXI_GMEM_DATA_WIDTH, C_AXI_GMEM_WSTRB_WIDTH, C_AXI_GMEM_USER_WIDTH),
    `AXIMM_MASTER_PORTS(m_axi_gmem_a, C_AXI_GMEM_ID_WIDTH, C_AXI_GMEM_ADDR_WIDTH, C_AXI_GMEM_DATA_WIDTH, C_AXI_GMEM_WSTRB_WIDTH, C_AXI_GMEM_USER_WIDTH),
    `AXIMM_SLAVE_PORTS(s_axi_gmem_b, C_AXI_GMEM_ID_WIDTH, C_AXI_GMEM_ADDR_WIDTH, C_AXI_GMEM_DATA_WIDTH, C_AXI_GMEM_WSTRB_WIDTH, C_AXI_GMEM_USER_WIDTH),
    `AXIMM_MASTER_PORTS(m_axi_gmem_b, C_AXI_GMEM_ID_WIDTH, C_AXI_GMEM_ADDR_WIDTH, C_AXI_GMEM_DATA_WIDTH, C_AXI_GMEM_WSTRB_WIDTH, C_AXI_GMEM_USER_WIDTH),
    `AXIMM_SLAVE_PORTS(s_axi_gmem_c, C_AXI_GMEM_ID_WIDTH, C_AXI_GMEM_ADDR_WIDTH, C_AXI_GMEM_DATA_WIDTH, C_AXI_GMEM_WSTRB_WIDTH, C_AXI_GMEM_USER_WIDTH),
    `AXIMM_MASTER_PORTS(m_axi_gmem_c, C_AXI_GMEM_ID_WIDTH, C_AXI_GMEM_ADDR_WIDTH, C_AXI_GMEM_DATA_WIDTH, C_AXI_GMEM_WSTRB_WIDTH, C_AXI_GMEM_USER_WIDTH)
);

`AXIL_PASS_THROUGH(s_axi_control, m_axi_control);
`AXIMM_PASS_THROUGH(s_axi_gmem_a, m_axi_gmem_a);
`AXIMM_PASS_THROUGH(s_axi_gmem_b, m_axi_gmem_b);
`AXIMM_PASS_THROUGH(s_axi_gmem_c, m_axi_gmem_c);

mmult_axi_profile_regs #(
    .C_S_AXI_PROFILE_ADDR_WIDTH(C_S_AXI_PROFILE_ADDR_WIDTH),
    .C_S_AXI_PROFILE_DATA_WIDTH(C_S_AXI_PROFILE_DATA_WIDTH)
) profile_regs_i (
    .ap_clk(ap_clk),
    .ap_rst_n(ap_rst_n),
    .ctrl_awvalid(s_axi_control_AWVALID),
    .ctrl_awready(s_axi_control_AWREADY),
    .ctrl_awaddr(s_axi_control_AWADDR),
    .ctrl_wvalid(s_axi_control_WVALID),
    .ctrl_wready(s_axi_control_WREADY),
    .ctrl_wdata(s_axi_control_WDATA),
    .a_arvalid(s_axi_gmem_a_ARVALID),
    .a_arready(m_axi_gmem_a_ARREADY),
    .a_arlen(s_axi_gmem_a_ARLEN),
    .a_rvalid(m_axi_gmem_a_RVALID),
    .a_rready(s_axi_gmem_a_RREADY),
    .b_arvalid(s_axi_gmem_b_ARVALID),
    .b_arready(m_axi_gmem_b_ARREADY),
    .b_arlen(s_axi_gmem_b_ARLEN),
    .b_rvalid(m_axi_gmem_b_RVALID),
    .b_rready(s_axi_gmem_b_RREADY),
    .c_awvalid(s_axi_gmem_c_AWVALID),
    .c_awready(m_axi_gmem_c_AWREADY),
    .c_awlen(s_axi_gmem_c_AWLEN),
    .c_wvalid(s_axi_gmem_c_WVALID),
    .c_wready(m_axi_gmem_c_WREADY),
    .c_bvalid(m_axi_gmem_c_BVALID),
    .c_bready(s_axi_gmem_c_BREADY),
    .s_axi_profile_AWADDR(s_axi_profile_AWADDR),
    .s_axi_profile_AWVALID(s_axi_profile_AWVALID),
    .s_axi_profile_AWREADY(s_axi_profile_AWREADY),
    .s_axi_profile_WDATA(s_axi_profile_WDATA),
    .s_axi_profile_WSTRB(s_axi_profile_WSTRB),
    .s_axi_profile_WVALID(s_axi_profile_WVALID),
    .s_axi_profile_WREADY(s_axi_profile_WREADY),
    .s_axi_profile_BRESP(s_axi_profile_BRESP),
    .s_axi_profile_BVALID(s_axi_profile_BVALID),
    .s_axi_profile_BREADY(s_axi_profile_BREADY),
    .s_axi_profile_ARADDR(s_axi_profile_ARADDR),
    .s_axi_profile_ARVALID(s_axi_profile_ARVALID),
    .s_axi_profile_ARREADY(s_axi_profile_ARREADY),
    .s_axi_profile_RDATA(s_axi_profile_RDATA),
    .s_axi_profile_RRESP(s_axi_profile_RRESP),
    .s_axi_profile_RVALID(s_axi_profile_RVALID),
    .s_axi_profile_RREADY(s_axi_profile_RREADY)
);

endmodule

`undef AXIL_SLAVE_PORTS
`undef AXIL_MASTER_PORTS
`undef AXIMM_SLAVE_PORTS
`undef AXIMM_MASTER_PORTS
`undef AXIL_PASS_THROUGH
`undef AXIMM_PASS_THROUGH
