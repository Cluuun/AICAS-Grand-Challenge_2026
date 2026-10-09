// Generator : SpinalHDL v1.10.1    git head : 2527c7c6b0fb0f95e5e1a5722a0be732b364ce43
// Component : TryoTop

`timescale 1ns/1ps

module TryoTop (
  input  wire          resetn,
  input  wire          clk,
  input  wire          axilite_awvalid,
  output wire          axilite_awready,
  input  wire [15:0]   axilite_awaddr,
  input  wire [2:0]    axilite_awprot,
  input  wire          axilite_wvalid,
  output wire          axilite_wready,
  input  wire [63:0]   axilite_wdata,
  input  wire [7:0]    axilite_wstrb,
  output wire          axilite_bvalid,
  input  wire          axilite_bready,
  output wire [1:0]    axilite_bresp,
  input  wire          axilite_arvalid,
  output wire          axilite_arready,
  input  wire [15:0]   axilite_araddr,
  input  wire [2:0]    axilite_arprot,
  output wire          axilite_rvalid,
  input  wire          axilite_rready,
  output wire [63:0]   axilite_rdata,
  output wire [1:0]    axilite_rresp,
  output wire          gmem1_awvalid,
  input  wire          gmem1_awready,
  output wire [63:0]   gmem1_awaddr,
  output wire [0:0]    gmem1_awid,
  output wire [3:0]    gmem1_awregion,
  output wire [7:0]    gmem1_awlen,
  output wire [2:0]    gmem1_awsize,
  output wire [1:0]    gmem1_awburst,
  output wire [0:0]    gmem1_awlock,
  output wire [3:0]    gmem1_awcache,
  output wire [3:0]    gmem1_awqos,
  output wire [0:0]    gmem1_awuser,
  output wire [2:0]    gmem1_awprot,
  output wire          gmem1_wvalid,
  input  wire          gmem1_wready,
  output wire [127:0]  gmem1_wdata,
  output wire [15:0]   gmem1_wstrb,
  output wire [0:0]    gmem1_wuser,
  output wire          gmem1_wlast,
  input  wire          gmem1_bvalid,
  output wire          gmem1_bready,
  input  wire [0:0]    gmem1_bid,
  input  wire [1:0]    gmem1_bresp,
  input  wire [0:0]    gmem1_buser,
  output wire          gmem1_arvalid,
  input  wire          gmem1_arready,
  output wire [63:0]   gmem1_araddr,
  output wire [0:0]    gmem1_arid,
  output wire [3:0]    gmem1_arregion,
  output wire [7:0]    gmem1_arlen,
  output wire [2:0]    gmem1_arsize,
  output wire [1:0]    gmem1_arburst,
  output wire [0:0]    gmem1_arlock,
  output wire [3:0]    gmem1_arcache,
  output wire [3:0]    gmem1_arqos,
  output wire [0:0]    gmem1_aruser,
  output wire [2:0]    gmem1_arprot,
  input  wire          gmem1_rvalid,
  output wire          gmem1_rready,
  input  wire [127:0]  gmem1_rdata,
  input  wire [0:0]    gmem1_rid,
  input  wire [1:0]    gmem1_rresp,
  input  wire          gmem1_rlast,
  input  wire [0:0]    gmem1_ruser,
  output wire          gmem2_awvalid,
  input  wire          gmem2_awready,
  output wire [63:0]   gmem2_awaddr,
  output wire [0:0]    gmem2_awid,
  output wire [3:0]    gmem2_awregion,
  output wire [7:0]    gmem2_awlen,
  output wire [2:0]    gmem2_awsize,
  output wire [1:0]    gmem2_awburst,
  output wire [0:0]    gmem2_awlock,
  output wire [3:0]    gmem2_awcache,
  output wire [3:0]    gmem2_awqos,
  output wire [0:0]    gmem2_awuser,
  output wire [2:0]    gmem2_awprot,
  output wire          gmem2_wvalid,
  input  wire          gmem2_wready,
  output wire [127:0]  gmem2_wdata,
  output wire [15:0]   gmem2_wstrb,
  output wire [0:0]    gmem2_wuser,
  output wire          gmem2_wlast,
  input  wire          gmem2_bvalid,
  output wire          gmem2_bready,
  input  wire [0:0]    gmem2_bid,
  input  wire [1:0]    gmem2_bresp,
  input  wire [0:0]    gmem2_buser,
  output wire          gmem2_arvalid,
  input  wire          gmem2_arready,
  output wire [63:0]   gmem2_araddr,
  output wire [0:0]    gmem2_arid,
  output wire [3:0]    gmem2_arregion,
  output wire [7:0]    gmem2_arlen,
  output wire [2:0]    gmem2_arsize,
  output wire [1:0]    gmem2_arburst,
  output wire [0:0]    gmem2_arlock,
  output wire [3:0]    gmem2_arcache,
  output wire [3:0]    gmem2_arqos,
  output wire [0:0]    gmem2_aruser,
  output wire [2:0]    gmem2_arprot,
  input  wire          gmem2_rvalid,
  output wire          gmem2_rready,
  input  wire [127:0]  gmem2_rdata,
  input  wire [0:0]    gmem2_rid,
  input  wire [1:0]    gmem2_rresp,
  input  wire          gmem2_rlast,
  input  wire [0:0]    gmem2_ruser,
  output wire          gmem3_awvalid,
  input  wire          gmem3_awready,
  output wire [62:0]   gmem3_awaddr,
  output wire [0:0]    gmem3_awid,
  output wire [3:0]    gmem3_awregion,
  output wire [7:0]    gmem3_awlen,
  output wire [2:0]    gmem3_awsize,
  output wire [1:0]    gmem3_awburst,
  output wire [0:0]    gmem3_awlock,
  output wire [3:0]    gmem3_awcache,
  output wire [3:0]    gmem3_awqos,
  output wire [0:0]    gmem3_awuser,
  output wire [2:0]    gmem3_awprot,
  output wire          gmem3_wvalid,
  input  wire          gmem3_wready,
  output wire [127:0]  gmem3_wdata,
  output wire [15:0]   gmem3_wstrb,
  output wire [0:0]    gmem3_wuser,
  output wire          gmem3_wlast,
  input  wire          gmem3_bvalid,
  output wire          gmem3_bready,
  input  wire [0:0]    gmem3_bid,
  input  wire [1:0]    gmem3_bresp,
  input  wire [0:0]    gmem3_buser,
  output wire          gmem3_arvalid,
  input  wire          gmem3_arready,
  output wire [62:0]   gmem3_araddr,
  output wire [0:0]    gmem3_arid,
  output wire [3:0]    gmem3_arregion,
  output wire [7:0]    gmem3_arlen,
  output wire [2:0]    gmem3_arsize,
  output wire [1:0]    gmem3_arburst,
  output wire [0:0]    gmem3_arlock,
  output wire [3:0]    gmem3_arcache,
  output wire [3:0]    gmem3_arqos,
  output wire [0:0]    gmem3_aruser,
  output wire [2:0]    gmem3_arprot,
  input  wire          gmem3_rvalid,
  output wire          gmem3_rready,
  input  wire [127:0]  gmem3_rdata,
  input  wire [0:0]    gmem3_rid,
  input  wire [1:0]    gmem3_rresp,
  input  wire          gmem3_rlast,
  input  wire [0:0]    gmem3_ruser,
  output wire          idle
);

  wire                toplevel_inst_m_axi_wq_stream_fifo_io_flush;
  wire                toplevel_inst_m_axi_ws1_stream_fifo_io_flush;
  wire                toplevel_inst_m_axi_ws2_stream_fifo_io_flush;
  wire                toplevel_mux_1_s_stream_fifo_io_flush;
  wire                toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_flush;
  wire                toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_flush;
  wire                toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_flush;
  wire                toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_flush;
  wire       [31:0]   inst_m_axi_signals_O_L_BEGIN;
  wire       [31:0]   inst_m_axi_signals_O_L_CLOSE;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   inst_m_axi_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   inst_m_axi_signals_O_POS_ID;
  wire                inst_m_axi_signals_O_T;
  wire                inst_m_axi_gmem1_arvalid;
  wire       [63:0]   inst_m_axi_gmem1_araddr;
  wire       [0:0]    inst_m_axi_gmem1_arid;
  wire       [3:0]    inst_m_axi_gmem1_arregion;
  wire       [7:0]    inst_m_axi_gmem1_arlen;
  wire       [2:0]    inst_m_axi_gmem1_arsize;
  wire       [1:0]    inst_m_axi_gmem1_arburst;
  wire       [0:0]    inst_m_axi_gmem1_arlock;
  wire       [3:0]    inst_m_axi_gmem1_arcache;
  wire       [3:0]    inst_m_axi_gmem1_arqos;
  wire       [0:0]    inst_m_axi_gmem1_aruser;
  wire       [2:0]    inst_m_axi_gmem1_arprot;
  wire                inst_m_axi_gmem1_awvalid;
  wire       [63:0]   inst_m_axi_gmem1_awaddr;
  wire       [0:0]    inst_m_axi_gmem1_awid;
  wire       [3:0]    inst_m_axi_gmem1_awregion;
  wire       [7:0]    inst_m_axi_gmem1_awlen;
  wire       [2:0]    inst_m_axi_gmem1_awsize;
  wire       [1:0]    inst_m_axi_gmem1_awburst;
  wire       [0:0]    inst_m_axi_gmem1_awlock;
  wire       [3:0]    inst_m_axi_gmem1_awcache;
  wire       [3:0]    inst_m_axi_gmem1_awqos;
  wire       [0:0]    inst_m_axi_gmem1_awuser;
  wire       [2:0]    inst_m_axi_gmem1_awprot;
  wire                inst_m_axi_gmem1_wvalid;
  wire       [127:0]  inst_m_axi_gmem1_wdata;
  wire       [15:0]   inst_m_axi_gmem1_wstrb;
  wire       [0:0]    inst_m_axi_gmem1_wuser;
  wire                inst_m_axi_gmem1_wlast;
  wire                inst_m_axi_gmem1_rready;
  wire                inst_m_axi_gmem1_bready;
  wire                inst_m_axi_gmem2_arvalid;
  wire       [63:0]   inst_m_axi_gmem2_araddr;
  wire       [0:0]    inst_m_axi_gmem2_arid;
  wire       [3:0]    inst_m_axi_gmem2_arregion;
  wire       [7:0]    inst_m_axi_gmem2_arlen;
  wire       [2:0]    inst_m_axi_gmem2_arsize;
  wire       [1:0]    inst_m_axi_gmem2_arburst;
  wire       [0:0]    inst_m_axi_gmem2_arlock;
  wire       [3:0]    inst_m_axi_gmem2_arcache;
  wire       [3:0]    inst_m_axi_gmem2_arqos;
  wire       [0:0]    inst_m_axi_gmem2_aruser;
  wire       [2:0]    inst_m_axi_gmem2_arprot;
  wire                inst_m_axi_gmem2_awvalid;
  wire       [63:0]   inst_m_axi_gmem2_awaddr;
  wire       [0:0]    inst_m_axi_gmem2_awid;
  wire       [3:0]    inst_m_axi_gmem2_awregion;
  wire       [7:0]    inst_m_axi_gmem2_awlen;
  wire       [2:0]    inst_m_axi_gmem2_awsize;
  wire       [1:0]    inst_m_axi_gmem2_awburst;
  wire       [0:0]    inst_m_axi_gmem2_awlock;
  wire       [3:0]    inst_m_axi_gmem2_awcache;
  wire       [3:0]    inst_m_axi_gmem2_awqos;
  wire       [0:0]    inst_m_axi_gmem2_awuser;
  wire       [2:0]    inst_m_axi_gmem2_awprot;
  wire                inst_m_axi_gmem2_wvalid;
  wire       [127:0]  inst_m_axi_gmem2_wdata;
  wire       [15:0]   inst_m_axi_gmem2_wstrb;
  wire       [0:0]    inst_m_axi_gmem2_wuser;
  wire                inst_m_axi_gmem2_wlast;
  wire                inst_m_axi_gmem2_rready;
  wire                inst_m_axi_gmem2_bready;
  wire                inst_m_axi_x_stream_TVALID;
  wire       [199:0]  inst_m_axi_x_stream_TDATA;
  wire                inst_m_axi_wq_stream_TVALID;
  wire       [319:0]  inst_m_axi_wq_stream_TDATA;
  wire                inst_m_axi_ws1_stream_TVALID;
  wire       [39:0]   inst_m_axi_ws1_stream_TDATA;
  wire                inst_m_axi_ws2_stream_TVALID;
  wire       [39:0]   inst_m_axi_ws2_stream_TDATA;
  wire                inst_m_axi_y_stream_TREADY;
  wire                inst_m_axi_logit_stream_TREADY;
  wire                inst_m_axi_idle;
  wire       [31:0]   inst_kv_cache_signals_O_L_BEGIN;
  wire       [31:0]   inst_kv_cache_signals_O_L_CLOSE;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   inst_kv_cache_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   inst_kv_cache_signals_O_POS_ID;
  wire                inst_kv_cache_signals_O_T;
  wire                inst_kv_cache_gmem1_arvalid;
  wire       [62:0]   inst_kv_cache_gmem1_araddr;
  wire       [0:0]    inst_kv_cache_gmem1_arid;
  wire       [3:0]    inst_kv_cache_gmem1_arregion;
  wire       [7:0]    inst_kv_cache_gmem1_arlen;
  wire       [2:0]    inst_kv_cache_gmem1_arsize;
  wire       [1:0]    inst_kv_cache_gmem1_arburst;
  wire       [0:0]    inst_kv_cache_gmem1_arlock;
  wire       [3:0]    inst_kv_cache_gmem1_arcache;
  wire       [3:0]    inst_kv_cache_gmem1_arqos;
  wire       [0:0]    inst_kv_cache_gmem1_aruser;
  wire       [2:0]    inst_kv_cache_gmem1_arprot;
  wire                inst_kv_cache_gmem1_awvalid;
  wire       [62:0]   inst_kv_cache_gmem1_awaddr;
  wire       [0:0]    inst_kv_cache_gmem1_awid;
  wire       [3:0]    inst_kv_cache_gmem1_awregion;
  wire       [7:0]    inst_kv_cache_gmem1_awlen;
  wire       [2:0]    inst_kv_cache_gmem1_awsize;
  wire       [1:0]    inst_kv_cache_gmem1_awburst;
  wire       [0:0]    inst_kv_cache_gmem1_awlock;
  wire       [3:0]    inst_kv_cache_gmem1_awcache;
  wire       [3:0]    inst_kv_cache_gmem1_awqos;
  wire       [0:0]    inst_kv_cache_gmem1_awuser;
  wire       [2:0]    inst_kv_cache_gmem1_awprot;
  wire                inst_kv_cache_gmem1_wvalid;
  wire       [127:0]  inst_kv_cache_gmem1_wdata;
  wire       [15:0]   inst_kv_cache_gmem1_wstrb;
  wire       [0:0]    inst_kv_cache_gmem1_wuser;
  wire                inst_kv_cache_gmem1_wlast;
  wire                inst_kv_cache_gmem1_rready;
  wire                inst_kv_cache_gmem1_bready;
  wire                inst_kv_cache_kq_cache_i_stream_TVALID;
  wire       [63:0]   inst_kv_cache_kq_cache_i_stream_TDATA;
  wire                inst_kv_cache_kq_cache_o_stream_TREADY;
  wire                inst_kv_cache_ks_cache_i_stream_TVALID;
  wire       [7:0]    inst_kv_cache_ks_cache_i_stream_TDATA;
  wire                inst_kv_cache_ks_cache_o_stream_TREADY;
  wire                inst_kv_cache_vq_cache_i_stream_TVALID;
  wire       [63:0]   inst_kv_cache_vq_cache_i_stream_TDATA;
  wire                inst_kv_cache_vq_cache_o_stream_TREADY;
  wire                inst_kv_cache_vs_cache_i_stream_TVALID;
  wire       [7:0]    inst_kv_cache_vs_cache_i_stream_TDATA;
  wire                inst_kv_cache_vs_cache_o_stream_TREADY;
  wire                inst_kv_cache_idle;
  wire                inst_controller_axilite_awready;
  wire                inst_controller_axilite_wready;
  wire                inst_controller_axilite_bvalid;
  wire       [1:0]    inst_controller_axilite_bresp;
  wire                inst_controller_axilite_arready;
  wire                inst_controller_axilite_rvalid;
  wire       [63:0]   inst_controller_axilite_rdata;
  wire       [1:0]    inst_controller_axilite_rresp;
  wire       [31:0]   inst_controller_signals_L_BEGIN;
  wire       [31:0]   inst_controller_signals_L_CLOSE;
  wire       [63:0]   inst_controller_signals_MEMORY_DECODER_X;
  wire       [63:0]   inst_controller_signals_MEMORY_DECODER_Y;
  wire       [63:0]   inst_controller_signals_MEMORY_LOGIT_Y;
  wire       [63:0]   inst_controller_signals_MEMORY_DECODER_W_LO;
  wire       [63:0]   inst_controller_signals_MEMORY_DECODER_W_HI;
  wire       [63:0]   inst_controller_signals_MEMORY_LOGIT_W_LO;
  wire       [63:0]   inst_controller_signals_MEMORY_LOGIT_W_HI;
  wire       [63:0]   inst_controller_signals_MEMORY_KV_CACHE;
  wire       [31:0]   inst_controller_signals_POS_ID;
  wire                inst_controller_signals_T;
  wire       [31:0]   gemm_signals_O_L_BEGIN;
  wire       [31:0]   gemm_signals_O_L_CLOSE;
  wire       [63:0]   gemm_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   gemm_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   gemm_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   gemm_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   gemm_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   gemm_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   gemm_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   gemm_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   gemm_signals_O_POS_ID;
  wire                gemm_signals_O_T;
  wire                gemm_i_stream_TREADY;
  wire                gemm_w_stream_TREADY;
  wire                gemm_s_stream_TREADY;
  wire                gemm_s1_stream_TREADY;
  wire                gemm_s2_stream_TREADY;
  wire                gemm_o_stream_TVALID;
  wire       [239:0]  gemm_o_stream_TDATA;
  wire       [31:0]   demux_1_signals_O_L_BEGIN;
  wire       [31:0]   demux_1_signals_O_L_CLOSE;
  wire       [63:0]   demux_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   demux_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   demux_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   demux_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   demux_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   demux_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   demux_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   demux_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   demux_1_signals_O_POS_ID;
  wire                demux_1_signals_O_T;
  wire                demux_1_gemm_stream_TREADY;
  wire                demux_1_qk_stream_TVALID;
  wire       [183:0]  demux_1_qk_stream_TDATA;
  wire                demux_1_v_stream_TVALID;
  wire       [183:0]  demux_1_v_stream_TDATA;
  wire                demux_1_ug_stream_TVALID;
  wire       [167:0]  demux_1_ug_stream_TDATA;
  wire                demux_1_od_stream_TVALID;
  wire       [199:0]  demux_1_od_stream_TDATA;
  wire                demux_1_logit_stream_valid;
  wire       [255:0]  demux_1_logit_stream_payload_data;
  wire       [31:0]   rope_qk_quant_1_signals_O_L_BEGIN;
  wire       [31:0]   rope_qk_quant_1_signals_O_L_CLOSE;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   rope_qk_quant_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   rope_qk_quant_1_signals_O_POS_ID;
  wire                rope_qk_quant_1_signals_O_T;
  wire                rope_qk_quant_1_qk_stream_TREADY;
  wire                rope_qk_quant_1_rot_q_stream_TVALID;
  wire       [63:0]   rope_qk_quant_1_rot_q_stream_TDATA;
  wire                rope_qk_quant_1_rot_s_stream_TVALID;
  wire       [7:0]    rope_qk_quant_1_rot_s_stream_TDATA;
  wire       [31:0]   qk_gemm_1_signals_O_L_BEGIN;
  wire       [31:0]   qk_gemm_1_signals_O_L_CLOSE;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   qk_gemm_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   qk_gemm_1_signals_O_POS_ID;
  wire                qk_gemm_1_signals_O_T;
  wire                qk_gemm_1_qk_q_stream_TREADY;
  wire                qk_gemm_1_qk_s_stream_TREADY;
  wire                qk_gemm_1_kq_cache_i_stream_TREADY;
  wire                qk_gemm_1_ks_cache_i_stream_TREADY;
  wire                qk_gemm_1_kq_cache_o_stream_TVALID;
  wire       [63:0]   qk_gemm_1_kq_cache_o_stream_TDATA;
  wire                qk_gemm_1_ks_cache_o_stream_TVALID;
  wire       [7:0]    qk_gemm_1_ks_cache_o_stream_TDATA;
  wire                qk_gemm_1_r_stream_TVALID;
  wire       [247:0]  qk_gemm_1_r_stream_TDATA;
  wire       [31:0]   softmax_quant_1_signals_O_L_BEGIN;
  wire       [31:0]   softmax_quant_1_signals_O_L_CLOSE;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   softmax_quant_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   softmax_quant_1_signals_O_POS_ID;
  wire                softmax_quant_1_signals_O_T;
  wire                softmax_quant_1_r_stream_TREADY;
  wire                softmax_quant_1_rq_stream_TVALID;
  wire       [63:0]   softmax_quant_1_rq_stream_TDATA;
  wire                softmax_quant_1_rs_stream_TVALID;
  wire       [7:0]    softmax_quant_1_rs_stream_TDATA;
  wire       [31:0]   rv_gemm_1_signals_O_L_BEGIN;
  wire       [31:0]   rv_gemm_1_signals_O_L_CLOSE;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   rv_gemm_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   rv_gemm_1_signals_O_POS_ID;
  wire                rv_gemm_1_signals_O_T;
  wire                rv_gemm_1_rq_stream_TREADY;
  wire                rv_gemm_1_rs_stream_TREADY;
  wire                rv_gemm_1_v_stream_TREADY;
  wire                rv_gemm_1_vq_cache_i_stream_TREADY;
  wire                rv_gemm_1_vs_cache_i_stream_TREADY;
  wire                rv_gemm_1_vq_cache_o_stream_TVALID;
  wire       [63:0]   rv_gemm_1_vq_cache_o_stream_TDATA;
  wire                rv_gemm_1_vs_cache_o_stream_TVALID;
  wire       [7:0]    rv_gemm_1_vs_cache_o_stream_TDATA;
  wire                rv_gemm_1_aq_stream_TVALID;
  wire       [63:0]   rv_gemm_1_aq_stream_TDATA;
  wire                rv_gemm_1_as_stream_TVALID;
  wire       [7:0]    rv_gemm_1_as_stream_TDATA;
  wire       [31:0]   silu_em_quant_1_signals_O_L_BEGIN;
  wire       [31:0]   silu_em_quant_1_signals_O_L_CLOSE;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   silu_em_quant_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   silu_em_quant_1_signals_O_POS_ID;
  wire                silu_em_quant_1_signals_O_T;
  wire                silu_em_quant_1_ug_stream_TREADY;
  wire                silu_em_quant_1_q_stream_TVALID;
  wire       [63:0]   silu_em_quant_1_q_stream_TDATA;
  wire                silu_em_quant_1_s_stream_TVALID;
  wire       [7:0]    silu_em_quant_1_s_stream_TDATA;
  wire       [31:0]   residual_1_signals_O_L_BEGIN;
  wire       [31:0]   residual_1_signals_O_L_CLOSE;
  wire       [63:0]   residual_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   residual_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   residual_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   residual_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   residual_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   residual_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   residual_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   residual_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   residual_1_signals_O_POS_ID;
  wire                residual_1_signals_O_T;
  wire                residual_1_x_stream_TREADY;
  wire                residual_1_res_i_stream_TREADY;
  wire                residual_1_res_o_stream_TVALID;
  wire       [199:0]  residual_1_res_o_stream_TDATA;
  wire                residual_1_y_stream_TVALID;
  wire       [199:0]  residual_1_y_stream_TDATA;
  wire       [31:0]   rmsnorm_quant_1_signals_O_L_BEGIN;
  wire       [31:0]   rmsnorm_quant_1_signals_O_L_CLOSE;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   rmsnorm_quant_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   rmsnorm_quant_1_signals_O_POS_ID;
  wire                rmsnorm_quant_1_signals_O_T;
  wire                rmsnorm_quant_1_x_stream_TREADY;
  wire                rmsnorm_quant_1_xlnq_stream_TVALID;
  wire       [63:0]   rmsnorm_quant_1_xlnq_stream_TDATA;
  wire                rmsnorm_quant_1_xlns_stream_TVALID;
  wire       [7:0]    rmsnorm_quant_1_xlns_stream_TDATA;
  wire       [31:0]   mux_1_signals_O_L_BEGIN;
  wire       [31:0]   mux_1_signals_O_L_CLOSE;
  wire       [63:0]   mux_1_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   mux_1_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   mux_1_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   mux_1_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   mux_1_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   mux_1_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   mux_1_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   mux_1_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   mux_1_signals_O_POS_ID;
  wire                mux_1_signals_O_T;
  wire                mux_1_xlnq_stream_TREADY;
  wire                mux_1_xlns_stream_TREADY;
  wire                mux_1_aq_stream_TREADY;
  wire                mux_1_as_stream_TREADY;
  wire                mux_1_xmq_stream_TREADY;
  wire                mux_1_xms_stream_TREADY;
  wire                mux_1_q_stream_TVALID;
  wire       [511:0]  mux_1_q_stream_TDATA;
  wire                mux_1_s_stream_TVALID;
  wire       [39:0]   mux_1_s_stream_TDATA;
  wire                toplevel_inst_m_axi_wq_stream_fifo_io_push_ready;
  wire                toplevel_inst_m_axi_wq_stream_fifo_io_pop_valid;
  wire       [319:0]  toplevel_inst_m_axi_wq_stream_fifo_io_pop_payload_data;
  wire       [11:0]   toplevel_inst_m_axi_wq_stream_fifo_io_occupancy;
  wire       [11:0]   toplevel_inst_m_axi_wq_stream_fifo_io_availability;
  wire                toplevel_inst_m_axi_ws1_stream_fifo_io_push_ready;
  wire                toplevel_inst_m_axi_ws1_stream_fifo_io_pop_valid;
  wire       [39:0]   toplevel_inst_m_axi_ws1_stream_fifo_io_pop_payload_data;
  wire       [11:0]   toplevel_inst_m_axi_ws1_stream_fifo_io_occupancy;
  wire       [11:0]   toplevel_inst_m_axi_ws1_stream_fifo_io_availability;
  wire                toplevel_inst_m_axi_ws2_stream_fifo_io_push_ready;
  wire                toplevel_inst_m_axi_ws2_stream_fifo_io_pop_valid;
  wire       [39:0]   toplevel_inst_m_axi_ws2_stream_fifo_io_pop_payload_data;
  wire       [11:0]   toplevel_inst_m_axi_ws2_stream_fifo_io_occupancy;
  wire       [11:0]   toplevel_inst_m_axi_ws2_stream_fifo_io_availability;
  wire                toplevel_mux_1_s_stream_fifo_io_push_ready;
  wire                toplevel_mux_1_s_stream_fifo_io_pop_valid;
  wire       [39:0]   toplevel_mux_1_s_stream_fifo_io_pop_payload_data;
  wire       [4:0]    toplevel_mux_1_s_stream_fifo_io_occupancy;
  wire       [4:0]    toplevel_mux_1_s_stream_fifo_io_availability;
  wire                toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_push_ready;
  wire                toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_pop_valid;
  wire       [63:0]   toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_pop_payload_data;
  wire       [12:0]   toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_occupancy;
  wire       [12:0]   toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_availability;
  wire                toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_push_ready;
  wire                toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_pop_valid;
  wire       [7:0]    toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_pop_payload_data;
  wire       [12:0]   toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_occupancy;
  wire       [12:0]   toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_availability;
  wire                toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_push_ready;
  wire                toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_pop_valid;
  wire       [63:0]   toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_pop_payload_data;
  wire       [12:0]   toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_occupancy;
  wire       [12:0]   toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_availability;
  wire                toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_push_ready;
  wire                toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_pop_valid;
  wire       [7:0]    toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_pop_payload_data;
  wire       [12:0]   toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_occupancy;
  wire       [12:0]   toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_availability;

  tryo_mmu_wrapper inst_m_axi (
    .resetn                        (resetn                                           ), //i
    .clk                           (clk                                              ), //i
    .signals_I_L_BEGIN             (inst_controller_signals_L_BEGIN[31:0]            ), //i
    .signals_I_L_CLOSE             (inst_controller_signals_L_CLOSE[31:0]            ), //i
    .signals_I_MEMORY_DECODER_X    (inst_controller_signals_MEMORY_DECODER_X[63:0]   ), //i
    .signals_I_MEMORY_DECODER_Y    (inst_controller_signals_MEMORY_DECODER_Y[63:0]   ), //i
    .signals_I_MEMORY_LOGIT_Y      (inst_controller_signals_MEMORY_LOGIT_Y[63:0]     ), //i
    .signals_I_MEMORY_DECODER_W_LO (inst_controller_signals_MEMORY_DECODER_W_LO[63:0]), //i
    .signals_I_MEMORY_DECODER_W_HI (inst_controller_signals_MEMORY_DECODER_W_HI[63:0]), //i
    .signals_I_MEMORY_LOGIT_W_LO   (inst_controller_signals_MEMORY_LOGIT_W_LO[63:0]  ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (inst_controller_signals_MEMORY_LOGIT_W_HI[63:0]  ), //i
    .signals_I_MEMORY_KV_CACHE     (inst_controller_signals_MEMORY_KV_CACHE[63:0]    ), //i
    .signals_I_POS_ID              (inst_controller_signals_POS_ID[31:0]             ), //i
    .signals_I_T                   (inst_controller_signals_T                        ), //i
    .signals_O_L_BEGIN             (inst_m_axi_signals_O_L_BEGIN[31:0]               ), //o
    .signals_O_L_CLOSE             (inst_m_axi_signals_O_L_CLOSE[31:0]               ), //o
    .signals_O_MEMORY_DECODER_X    (inst_m_axi_signals_O_MEMORY_DECODER_X[63:0]      ), //o
    .signals_O_MEMORY_DECODER_Y    (inst_m_axi_signals_O_MEMORY_DECODER_Y[63:0]      ), //o
    .signals_O_MEMORY_LOGIT_Y      (inst_m_axi_signals_O_MEMORY_LOGIT_Y[63:0]        ), //o
    .signals_O_MEMORY_DECODER_W_LO (inst_m_axi_signals_O_MEMORY_DECODER_W_LO[63:0]   ), //o
    .signals_O_MEMORY_DECODER_W_HI (inst_m_axi_signals_O_MEMORY_DECODER_W_HI[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_W_LO   (inst_m_axi_signals_O_MEMORY_LOGIT_W_LO[63:0]     ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (inst_m_axi_signals_O_MEMORY_LOGIT_W_HI[63:0]     ), //o
    .signals_O_MEMORY_KV_CACHE     (inst_m_axi_signals_O_MEMORY_KV_CACHE[63:0]       ), //o
    .signals_O_POS_ID              (inst_m_axi_signals_O_POS_ID[31:0]                ), //o
    .signals_O_T                   (inst_m_axi_signals_O_T                           ), //o
    .gmem1_awvalid                 (inst_m_axi_gmem1_awvalid                         ), //o
    .gmem1_awready                 (gmem1_awready                                    ), //i
    .gmem1_awaddr                  (inst_m_axi_gmem1_awaddr[63:0]                    ), //o
    .gmem1_awid                    (inst_m_axi_gmem1_awid                            ), //o
    .gmem1_awregion                (inst_m_axi_gmem1_awregion[3:0]                   ), //o
    .gmem1_awlen                   (inst_m_axi_gmem1_awlen[7:0]                      ), //o
    .gmem1_awsize                  (inst_m_axi_gmem1_awsize[2:0]                     ), //o
    .gmem1_awburst                 (inst_m_axi_gmem1_awburst[1:0]                    ), //o
    .gmem1_awlock                  (inst_m_axi_gmem1_awlock                          ), //o
    .gmem1_awcache                 (inst_m_axi_gmem1_awcache[3:0]                    ), //o
    .gmem1_awqos                   (inst_m_axi_gmem1_awqos[3:0]                      ), //o
    .gmem1_awuser                  (inst_m_axi_gmem1_awuser                          ), //o
    .gmem1_awprot                  (inst_m_axi_gmem1_awprot[2:0]                     ), //o
    .gmem1_wvalid                  (inst_m_axi_gmem1_wvalid                          ), //o
    .gmem1_wready                  (gmem1_wready                                     ), //i
    .gmem1_wdata                   (inst_m_axi_gmem1_wdata[127:0]                    ), //o
    .gmem1_wstrb                   (inst_m_axi_gmem1_wstrb[15:0]                     ), //o
    .gmem1_wuser                   (inst_m_axi_gmem1_wuser                           ), //o
    .gmem1_wlast                   (inst_m_axi_gmem1_wlast                           ), //o
    .gmem1_bvalid                  (gmem1_bvalid                                     ), //i
    .gmem1_bready                  (inst_m_axi_gmem1_bready                          ), //o
    .gmem1_bid                     (gmem1_bid                                        ), //i
    .gmem1_bresp                   (gmem1_bresp[1:0]                                 ), //i
    .gmem1_buser                   (gmem1_buser                                      ), //i
    .gmem1_arvalid                 (inst_m_axi_gmem1_arvalid                         ), //o
    .gmem1_arready                 (gmem1_arready                                    ), //i
    .gmem1_araddr                  (inst_m_axi_gmem1_araddr[63:0]                    ), //o
    .gmem1_arid                    (inst_m_axi_gmem1_arid                            ), //o
    .gmem1_arregion                (inst_m_axi_gmem1_arregion[3:0]                   ), //o
    .gmem1_arlen                   (inst_m_axi_gmem1_arlen[7:0]                      ), //o
    .gmem1_arsize                  (inst_m_axi_gmem1_arsize[2:0]                     ), //o
    .gmem1_arburst                 (inst_m_axi_gmem1_arburst[1:0]                    ), //o
    .gmem1_arlock                  (inst_m_axi_gmem1_arlock                          ), //o
    .gmem1_arcache                 (inst_m_axi_gmem1_arcache[3:0]                    ), //o
    .gmem1_arqos                   (inst_m_axi_gmem1_arqos[3:0]                      ), //o
    .gmem1_aruser                  (inst_m_axi_gmem1_aruser                          ), //o
    .gmem1_arprot                  (inst_m_axi_gmem1_arprot[2:0]                     ), //o
    .gmem1_rvalid                  (gmem1_rvalid                                     ), //i
    .gmem1_rready                  (inst_m_axi_gmem1_rready                          ), //o
    .gmem1_rdata                   (gmem1_rdata[127:0]                               ), //i
    .gmem1_rid                     (gmem1_rid                                        ), //i
    .gmem1_rresp                   (gmem1_rresp[1:0]                                 ), //i
    .gmem1_rlast                   (gmem1_rlast                                      ), //i
    .gmem1_ruser                   (gmem1_ruser                                      ), //i
    .gmem2_awvalid                 (inst_m_axi_gmem2_awvalid                         ), //o
    .gmem2_awready                 (gmem2_awready                                    ), //i
    .gmem2_awaddr                  (inst_m_axi_gmem2_awaddr[63:0]                    ), //o
    .gmem2_awid                    (inst_m_axi_gmem2_awid                            ), //o
    .gmem2_awregion                (inst_m_axi_gmem2_awregion[3:0]                   ), //o
    .gmem2_awlen                   (inst_m_axi_gmem2_awlen[7:0]                      ), //o
    .gmem2_awsize                  (inst_m_axi_gmem2_awsize[2:0]                     ), //o
    .gmem2_awburst                 (inst_m_axi_gmem2_awburst[1:0]                    ), //o
    .gmem2_awlock                  (inst_m_axi_gmem2_awlock                          ), //o
    .gmem2_awcache                 (inst_m_axi_gmem2_awcache[3:0]                    ), //o
    .gmem2_awqos                   (inst_m_axi_gmem2_awqos[3:0]                      ), //o
    .gmem2_awuser                  (inst_m_axi_gmem2_awuser                          ), //o
    .gmem2_awprot                  (inst_m_axi_gmem2_awprot[2:0]                     ), //o
    .gmem2_wvalid                  (inst_m_axi_gmem2_wvalid                          ), //o
    .gmem2_wready                  (gmem2_wready                                     ), //i
    .gmem2_wdata                   (inst_m_axi_gmem2_wdata[127:0]                    ), //o
    .gmem2_wstrb                   (inst_m_axi_gmem2_wstrb[15:0]                     ), //o
    .gmem2_wuser                   (inst_m_axi_gmem2_wuser                           ), //o
    .gmem2_wlast                   (inst_m_axi_gmem2_wlast                           ), //o
    .gmem2_bvalid                  (gmem2_bvalid                                     ), //i
    .gmem2_bready                  (inst_m_axi_gmem2_bready                          ), //o
    .gmem2_bid                     (gmem2_bid                                        ), //i
    .gmem2_bresp                   (gmem2_bresp[1:0]                                 ), //i
    .gmem2_buser                   (gmem2_buser                                      ), //i
    .gmem2_arvalid                 (inst_m_axi_gmem2_arvalid                         ), //o
    .gmem2_arready                 (gmem2_arready                                    ), //i
    .gmem2_araddr                  (inst_m_axi_gmem2_araddr[63:0]                    ), //o
    .gmem2_arid                    (inst_m_axi_gmem2_arid                            ), //o
    .gmem2_arregion                (inst_m_axi_gmem2_arregion[3:0]                   ), //o
    .gmem2_arlen                   (inst_m_axi_gmem2_arlen[7:0]                      ), //o
    .gmem2_arsize                  (inst_m_axi_gmem2_arsize[2:0]                     ), //o
    .gmem2_arburst                 (inst_m_axi_gmem2_arburst[1:0]                    ), //o
    .gmem2_arlock                  (inst_m_axi_gmem2_arlock                          ), //o
    .gmem2_arcache                 (inst_m_axi_gmem2_arcache[3:0]                    ), //o
    .gmem2_arqos                   (inst_m_axi_gmem2_arqos[3:0]                      ), //o
    .gmem2_aruser                  (inst_m_axi_gmem2_aruser                          ), //o
    .gmem2_arprot                  (inst_m_axi_gmem2_arprot[2:0]                     ), //o
    .gmem2_rvalid                  (gmem2_rvalid                                     ), //i
    .gmem2_rready                  (inst_m_axi_gmem2_rready                          ), //o
    .gmem2_rdata                   (gmem2_rdata[127:0]                               ), //i
    .gmem2_rid                     (gmem2_rid                                        ), //i
    .gmem2_rresp                   (gmem2_rresp[1:0]                                 ), //i
    .gmem2_rlast                   (gmem2_rlast                                      ), //i
    .gmem2_ruser                   (gmem2_ruser                                      ), //i
    .x_stream_TVALID               (inst_m_axi_x_stream_TVALID                       ), //o
    .x_stream_TREADY               (residual_1_x_stream_TREADY                       ), //i
    .x_stream_TDATA                (inst_m_axi_x_stream_TDATA[199:0]                 ), //o
    .wq_stream_TVALID              (inst_m_axi_wq_stream_TVALID                      ), //o
    .wq_stream_TREADY              (toplevel_inst_m_axi_wq_stream_fifo_io_push_ready ), //i
    .wq_stream_TDATA               (inst_m_axi_wq_stream_TDATA[319:0]                ), //o
    .ws1_stream_TVALID             (inst_m_axi_ws1_stream_TVALID                     ), //o
    .ws1_stream_TREADY             (toplevel_inst_m_axi_ws1_stream_fifo_io_push_ready), //i
    .ws1_stream_TDATA              (inst_m_axi_ws1_stream_TDATA[39:0]                ), //o
    .ws2_stream_TVALID             (inst_m_axi_ws2_stream_TVALID                     ), //o
    .ws2_stream_TREADY             (toplevel_inst_m_axi_ws2_stream_fifo_io_push_ready), //i
    .ws2_stream_TDATA              (inst_m_axi_ws2_stream_TDATA[39:0]                ), //o
    .y_stream_TVALID               (residual_1_y_stream_TVALID                       ), //i
    .y_stream_TREADY               (inst_m_axi_y_stream_TREADY                       ), //o
    .y_stream_TDATA                (residual_1_y_stream_TDATA[199:0]                 ), //i
    .logit_stream_TVALID           (demux_1_logit_stream_valid                       ), //i
    .logit_stream_TREADY           (inst_m_axi_logit_stream_TREADY                   ), //o
    .logit_stream_TDATA            (demux_1_logit_stream_payload_data[255:0]         ), //i
    .idle                          (inst_m_axi_idle                                  )  //o
  );
  tryo_kvbank_wrapper inst_kv_cache (
    .resetn                        (resetn                                                             ), //i
    .clk                           (clk                                                                ), //i
    .signals_I_L_BEGIN             (inst_m_axi_signals_O_L_BEGIN[31:0]                                 ), //i
    .signals_I_L_CLOSE             (inst_m_axi_signals_O_L_CLOSE[31:0]                                 ), //i
    .signals_I_MEMORY_DECODER_X    (inst_m_axi_signals_O_MEMORY_DECODER_X[63:0]                        ), //i
    .signals_I_MEMORY_DECODER_Y    (inst_m_axi_signals_O_MEMORY_DECODER_Y[63:0]                        ), //i
    .signals_I_MEMORY_LOGIT_Y      (inst_m_axi_signals_O_MEMORY_LOGIT_Y[63:0]                          ), //i
    .signals_I_MEMORY_DECODER_W_LO (inst_m_axi_signals_O_MEMORY_DECODER_W_LO[63:0]                     ), //i
    .signals_I_MEMORY_DECODER_W_HI (inst_m_axi_signals_O_MEMORY_DECODER_W_HI[63:0]                     ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (inst_m_axi_signals_O_MEMORY_LOGIT_W_LO[63:0]                       ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (inst_m_axi_signals_O_MEMORY_LOGIT_W_HI[63:0]                       ), //i
    .signals_I_MEMORY_KV_CACHE     (inst_m_axi_signals_O_MEMORY_KV_CACHE[63:0]                         ), //i
    .signals_I_POS_ID              (inst_m_axi_signals_O_POS_ID[31:0]                                  ), //i
    .signals_I_T                   (inst_m_axi_signals_O_T                                             ), //i
    .signals_O_L_BEGIN             (inst_kv_cache_signals_O_L_BEGIN[31:0]                              ), //o
    .signals_O_L_CLOSE             (inst_kv_cache_signals_O_L_CLOSE[31:0]                              ), //o
    .signals_O_MEMORY_DECODER_X    (inst_kv_cache_signals_O_MEMORY_DECODER_X[63:0]                     ), //o
    .signals_O_MEMORY_DECODER_Y    (inst_kv_cache_signals_O_MEMORY_DECODER_Y[63:0]                     ), //o
    .signals_O_MEMORY_LOGIT_Y      (inst_kv_cache_signals_O_MEMORY_LOGIT_Y[63:0]                       ), //o
    .signals_O_MEMORY_DECODER_W_LO (inst_kv_cache_signals_O_MEMORY_DECODER_W_LO[63:0]                  ), //o
    .signals_O_MEMORY_DECODER_W_HI (inst_kv_cache_signals_O_MEMORY_DECODER_W_HI[63:0]                  ), //o
    .signals_O_MEMORY_LOGIT_W_LO   (inst_kv_cache_signals_O_MEMORY_LOGIT_W_LO[63:0]                    ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (inst_kv_cache_signals_O_MEMORY_LOGIT_W_HI[63:0]                    ), //o
    .signals_O_MEMORY_KV_CACHE     (inst_kv_cache_signals_O_MEMORY_KV_CACHE[63:0]                      ), //o
    .signals_O_POS_ID              (inst_kv_cache_signals_O_POS_ID[31:0]                               ), //o
    .signals_O_T                   (inst_kv_cache_signals_O_T                                          ), //o
    .gmem1_awvalid                 (inst_kv_cache_gmem1_awvalid                                        ), //o
    .gmem1_awready                 (gmem3_awready                                                      ), //i
    .gmem1_awaddr                  (inst_kv_cache_gmem1_awaddr[62:0]                                   ), //o
    .gmem1_awid                    (inst_kv_cache_gmem1_awid                                           ), //o
    .gmem1_awregion                (inst_kv_cache_gmem1_awregion[3:0]                                  ), //o
    .gmem1_awlen                   (inst_kv_cache_gmem1_awlen[7:0]                                     ), //o
    .gmem1_awsize                  (inst_kv_cache_gmem1_awsize[2:0]                                    ), //o
    .gmem1_awburst                 (inst_kv_cache_gmem1_awburst[1:0]                                   ), //o
    .gmem1_awlock                  (inst_kv_cache_gmem1_awlock                                         ), //o
    .gmem1_awcache                 (inst_kv_cache_gmem1_awcache[3:0]                                   ), //o
    .gmem1_awqos                   (inst_kv_cache_gmem1_awqos[3:0]                                     ), //o
    .gmem1_awuser                  (inst_kv_cache_gmem1_awuser                                         ), //o
    .gmem1_awprot                  (inst_kv_cache_gmem1_awprot[2:0]                                    ), //o
    .gmem1_wvalid                  (inst_kv_cache_gmem1_wvalid                                         ), //o
    .gmem1_wready                  (gmem3_wready                                                       ), //i
    .gmem1_wdata                   (inst_kv_cache_gmem1_wdata[127:0]                                   ), //o
    .gmem1_wstrb                   (inst_kv_cache_gmem1_wstrb[15:0]                                    ), //o
    .gmem1_wuser                   (inst_kv_cache_gmem1_wuser                                          ), //o
    .gmem1_wlast                   (inst_kv_cache_gmem1_wlast                                          ), //o
    .gmem1_bvalid                  (gmem3_bvalid                                                       ), //i
    .gmem1_bready                  (inst_kv_cache_gmem1_bready                                         ), //o
    .gmem1_bid                     (gmem3_bid                                                          ), //i
    .gmem1_bresp                   (gmem3_bresp[1:0]                                                   ), //i
    .gmem1_buser                   (gmem3_buser                                                        ), //i
    .gmem1_arvalid                 (inst_kv_cache_gmem1_arvalid                                        ), //o
    .gmem1_arready                 (gmem3_arready                                                      ), //i
    .gmem1_araddr                  (inst_kv_cache_gmem1_araddr[62:0]                                   ), //o
    .gmem1_arid                    (inst_kv_cache_gmem1_arid                                           ), //o
    .gmem1_arregion                (inst_kv_cache_gmem1_arregion[3:0]                                  ), //o
    .gmem1_arlen                   (inst_kv_cache_gmem1_arlen[7:0]                                     ), //o
    .gmem1_arsize                  (inst_kv_cache_gmem1_arsize[2:0]                                    ), //o
    .gmem1_arburst                 (inst_kv_cache_gmem1_arburst[1:0]                                   ), //o
    .gmem1_arlock                  (inst_kv_cache_gmem1_arlock                                         ), //o
    .gmem1_arcache                 (inst_kv_cache_gmem1_arcache[3:0]                                   ), //o
    .gmem1_arqos                   (inst_kv_cache_gmem1_arqos[3:0]                                     ), //o
    .gmem1_aruser                  (inst_kv_cache_gmem1_aruser                                         ), //o
    .gmem1_arprot                  (inst_kv_cache_gmem1_arprot[2:0]                                    ), //o
    .gmem1_rvalid                  (gmem3_rvalid                                                       ), //i
    .gmem1_rready                  (inst_kv_cache_gmem1_rready                                         ), //o
    .gmem1_rdata                   (gmem3_rdata[127:0]                                                 ), //i
    .gmem1_rid                     (gmem3_rid                                                          ), //i
    .gmem1_rresp                   (gmem3_rresp[1:0]                                                   ), //i
    .gmem1_rlast                   (gmem3_rlast                                                        ), //i
    .gmem1_ruser                   (gmem3_ruser                                                        ), //i
    .kq_cache_i_stream_TVALID      (inst_kv_cache_kq_cache_i_stream_TVALID                             ), //o
    .kq_cache_i_stream_TREADY      (qk_gemm_1_kq_cache_i_stream_TREADY                                 ), //i
    .kq_cache_i_stream_TDATA       (inst_kv_cache_kq_cache_i_stream_TDATA[63:0]                        ), //o
    .kq_cache_o_stream_TVALID      (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_pop_valid             ), //i
    .kq_cache_o_stream_TREADY      (inst_kv_cache_kq_cache_o_stream_TREADY                             ), //o
    .kq_cache_o_stream_TDATA       (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_pop_payload_data[63:0]), //i
    .ks_cache_i_stream_TVALID      (inst_kv_cache_ks_cache_i_stream_TVALID                             ), //o
    .ks_cache_i_stream_TREADY      (qk_gemm_1_ks_cache_i_stream_TREADY                                 ), //i
    .ks_cache_i_stream_TDATA       (inst_kv_cache_ks_cache_i_stream_TDATA[7:0]                         ), //o
    .ks_cache_o_stream_TVALID      (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_pop_valid             ), //i
    .ks_cache_o_stream_TREADY      (inst_kv_cache_ks_cache_o_stream_TREADY                             ), //o
    .ks_cache_o_stream_TDATA       (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_pop_payload_data[7:0] ), //i
    .vq_cache_i_stream_TVALID      (inst_kv_cache_vq_cache_i_stream_TVALID                             ), //o
    .vq_cache_i_stream_TREADY      (rv_gemm_1_vq_cache_i_stream_TREADY                                 ), //i
    .vq_cache_i_stream_TDATA       (inst_kv_cache_vq_cache_i_stream_TDATA[63:0]                        ), //o
    .vq_cache_o_stream_TVALID      (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_pop_valid             ), //i
    .vq_cache_o_stream_TREADY      (inst_kv_cache_vq_cache_o_stream_TREADY                             ), //o
    .vq_cache_o_stream_TDATA       (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_pop_payload_data[63:0]), //i
    .vs_cache_i_stream_TVALID      (inst_kv_cache_vs_cache_i_stream_TVALID                             ), //o
    .vs_cache_i_stream_TREADY      (rv_gemm_1_vs_cache_i_stream_TREADY                                 ), //i
    .vs_cache_i_stream_TDATA       (inst_kv_cache_vs_cache_i_stream_TDATA[7:0]                         ), //o
    .vs_cache_o_stream_TVALID      (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_pop_valid             ), //i
    .vs_cache_o_stream_TREADY      (inst_kv_cache_vs_cache_o_stream_TREADY                             ), //o
    .vs_cache_o_stream_TDATA       (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_pop_payload_data[7:0] ), //i
    .idle                          (inst_kv_cache_idle                                                 )  //o
  );
  Controller inst_controller (
    .axilite_awvalid             (axilite_awvalid                                  ), //i
    .axilite_awready             (inst_controller_axilite_awready                  ), //o
    .axilite_awaddr              (axilite_awaddr[15:0]                             ), //i
    .axilite_awprot              (axilite_awprot[2:0]                              ), //i
    .axilite_wvalid              (axilite_wvalid                                   ), //i
    .axilite_wready              (inst_controller_axilite_wready                   ), //o
    .axilite_wdata               (axilite_wdata[63:0]                              ), //i
    .axilite_wstrb               (axilite_wstrb[7:0]                               ), //i
    .axilite_bvalid              (inst_controller_axilite_bvalid                   ), //o
    .axilite_bready              (axilite_bready                                   ), //i
    .axilite_bresp               (inst_controller_axilite_bresp[1:0]               ), //o
    .axilite_arvalid             (axilite_arvalid                                  ), //i
    .axilite_arready             (inst_controller_axilite_arready                  ), //o
    .axilite_araddr              (axilite_araddr[15:0]                             ), //i
    .axilite_arprot              (axilite_arprot[2:0]                              ), //i
    .axilite_rvalid              (inst_controller_axilite_rvalid                   ), //o
    .axilite_rready              (axilite_rready                                   ), //i
    .axilite_rdata               (inst_controller_axilite_rdata[63:0]              ), //o
    .axilite_rresp               (inst_controller_axilite_rresp[1:0]               ), //o
    .signals_L_BEGIN             (inst_controller_signals_L_BEGIN[31:0]            ), //o
    .signals_L_CLOSE             (inst_controller_signals_L_CLOSE[31:0]            ), //o
    .signals_MEMORY_DECODER_X    (inst_controller_signals_MEMORY_DECODER_X[63:0]   ), //o
    .signals_MEMORY_DECODER_Y    (inst_controller_signals_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_MEMORY_LOGIT_Y      (inst_controller_signals_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_MEMORY_DECODER_W_LO (inst_controller_signals_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_MEMORY_DECODER_W_HI (inst_controller_signals_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_MEMORY_LOGIT_W_LO   (inst_controller_signals_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_MEMORY_LOGIT_W_HI   (inst_controller_signals_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_MEMORY_KV_CACHE     (inst_controller_signals_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_POS_ID              (inst_controller_signals_POS_ID[31:0]             ), //o
    .signals_T                   (inst_controller_signals_T                        ), //o
    .idle                        (inst_m_axi_idle                                  ), //i
    .clk                         (clk                                              ), //i
    .resetn                      (resetn                                           )  //i
  );
  tryo_vpu_wrapper gemm (
    .resetn                        (resetn                                                       ), //i
    .clk                           (clk                                                          ), //i
    .signals_I_L_BEGIN             (mux_1_signals_O_L_BEGIN[31:0]                                ), //i
    .signals_I_L_CLOSE             (mux_1_signals_O_L_CLOSE[31:0]                                ), //i
    .signals_I_MEMORY_DECODER_X    (mux_1_signals_O_MEMORY_DECODER_X[63:0]                       ), //i
    .signals_I_MEMORY_DECODER_Y    (mux_1_signals_O_MEMORY_DECODER_Y[63:0]                       ), //i
    .signals_I_MEMORY_LOGIT_Y      (mux_1_signals_O_MEMORY_LOGIT_Y[63:0]                         ), //i
    .signals_I_MEMORY_DECODER_W_LO (mux_1_signals_O_MEMORY_DECODER_W_LO[63:0]                    ), //i
    .signals_I_MEMORY_DECODER_W_HI (mux_1_signals_O_MEMORY_DECODER_W_HI[63:0]                    ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (mux_1_signals_O_MEMORY_LOGIT_W_LO[63:0]                      ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (mux_1_signals_O_MEMORY_LOGIT_W_HI[63:0]                      ), //i
    .signals_I_MEMORY_KV_CACHE     (mux_1_signals_O_MEMORY_KV_CACHE[63:0]                        ), //i
    .signals_I_POS_ID              (mux_1_signals_O_POS_ID[31:0]                                 ), //i
    .signals_I_T                   (mux_1_signals_O_T                                            ), //i
    .signals_O_L_BEGIN             (gemm_signals_O_L_BEGIN[31:0]                                 ), //o
    .signals_O_L_CLOSE             (gemm_signals_O_L_CLOSE[31:0]                                 ), //o
    .signals_O_MEMORY_DECODER_X    (gemm_signals_O_MEMORY_DECODER_X[63:0]                        ), //o
    .signals_O_MEMORY_DECODER_Y    (gemm_signals_O_MEMORY_DECODER_Y[63:0]                        ), //o
    .signals_O_MEMORY_LOGIT_Y      (gemm_signals_O_MEMORY_LOGIT_Y[63:0]                          ), //o
    .signals_O_MEMORY_DECODER_W_LO (gemm_signals_O_MEMORY_DECODER_W_LO[63:0]                     ), //o
    .signals_O_MEMORY_DECODER_W_HI (gemm_signals_O_MEMORY_DECODER_W_HI[63:0]                     ), //o
    .signals_O_MEMORY_LOGIT_W_LO   (gemm_signals_O_MEMORY_LOGIT_W_LO[63:0]                       ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (gemm_signals_O_MEMORY_LOGIT_W_HI[63:0]                       ), //o
    .signals_O_MEMORY_KV_CACHE     (gemm_signals_O_MEMORY_KV_CACHE[63:0]                         ), //o
    .signals_O_POS_ID              (gemm_signals_O_POS_ID[31:0]                                  ), //o
    .signals_O_T                   (gemm_signals_O_T                                             ), //o
    .i_stream_TVALID               (mux_1_q_stream_TVALID                                        ), //i
    .i_stream_TREADY               (gemm_i_stream_TREADY                                         ), //o
    .i_stream_TDATA                (mux_1_q_stream_TDATA[511:0]                                  ), //i
    .w_stream_TVALID               (toplevel_inst_m_axi_wq_stream_fifo_io_pop_valid              ), //i
    .w_stream_TREADY               (gemm_w_stream_TREADY                                         ), //o
    .w_stream_TDATA                (toplevel_inst_m_axi_wq_stream_fifo_io_pop_payload_data[319:0]), //i
    .s_stream_TVALID               (toplevel_mux_1_s_stream_fifo_io_pop_valid                    ), //i
    .s_stream_TREADY               (gemm_s_stream_TREADY                                         ), //o
    .s_stream_TDATA                (toplevel_mux_1_s_stream_fifo_io_pop_payload_data[39:0]       ), //i
    .s1_stream_TVALID              (toplevel_inst_m_axi_ws1_stream_fifo_io_pop_valid             ), //i
    .s1_stream_TREADY              (gemm_s1_stream_TREADY                                        ), //o
    .s1_stream_TDATA               (toplevel_inst_m_axi_ws1_stream_fifo_io_pop_payload_data[39:0]), //i
    .s2_stream_TVALID              (toplevel_inst_m_axi_ws2_stream_fifo_io_pop_valid             ), //i
    .s2_stream_TREADY              (gemm_s2_stream_TREADY                                        ), //o
    .s2_stream_TDATA               (toplevel_inst_m_axi_ws2_stream_fifo_io_pop_payload_data[39:0]), //i
    .o_stream_TVALID               (gemm_o_stream_TVALID                                         ), //o
    .o_stream_TREADY               (demux_1_gemm_stream_TREADY                                   ), //i
    .o_stream_TDATA                (gemm_o_stream_TDATA[239:0]                                   )  //o
  );
  tryo_route_wrapper demux_1 (
    .resetn                        (resetn                                     ), //i
    .clk                           (clk                                        ), //i
    .signals_I_L_BEGIN             (gemm_signals_O_L_BEGIN[31:0]               ), //i
    .signals_I_L_CLOSE             (gemm_signals_O_L_CLOSE[31:0]               ), //i
    .signals_I_MEMORY_DECODER_X    (gemm_signals_O_MEMORY_DECODER_X[63:0]      ), //i
    .signals_I_MEMORY_DECODER_Y    (gemm_signals_O_MEMORY_DECODER_Y[63:0]      ), //i
    .signals_I_MEMORY_LOGIT_Y      (gemm_signals_O_MEMORY_LOGIT_Y[63:0]        ), //i
    .signals_I_MEMORY_DECODER_W_LO (gemm_signals_O_MEMORY_DECODER_W_LO[63:0]   ), //i
    .signals_I_MEMORY_DECODER_W_HI (gemm_signals_O_MEMORY_DECODER_W_HI[63:0]   ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (gemm_signals_O_MEMORY_LOGIT_W_LO[63:0]     ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (gemm_signals_O_MEMORY_LOGIT_W_HI[63:0]     ), //i
    .signals_I_MEMORY_KV_CACHE     (gemm_signals_O_MEMORY_KV_CACHE[63:0]       ), //i
    .signals_I_POS_ID              (gemm_signals_O_POS_ID[31:0]                ), //i
    .signals_I_T                   (gemm_signals_O_T                           ), //i
    .signals_O_L_BEGIN             (demux_1_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (demux_1_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (demux_1_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (demux_1_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (demux_1_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (demux_1_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (demux_1_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (demux_1_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (demux_1_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (demux_1_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (demux_1_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (demux_1_signals_O_T                        ), //o
    .gemm_stream_TVALID            (gemm_o_stream_TVALID                       ), //i
    .gemm_stream_TREADY            (demux_1_gemm_stream_TREADY                 ), //o
    .gemm_stream_TDATA             (gemm_o_stream_TDATA[239:0]                 ), //i
    .qk_stream_TVALID              (demux_1_qk_stream_TVALID                   ), //o
    .qk_stream_TREADY              (rope_qk_quant_1_qk_stream_TREADY           ), //i
    .qk_stream_TDATA               (demux_1_qk_stream_TDATA[183:0]             ), //o
    .v_stream_TVALID               (demux_1_v_stream_TVALID                    ), //o
    .v_stream_TREADY               (rv_gemm_1_v_stream_TREADY                  ), //i
    .v_stream_TDATA                (demux_1_v_stream_TDATA[183:0]              ), //o
    .ug_stream_TVALID              (demux_1_ug_stream_TVALID                   ), //o
    .ug_stream_TREADY              (silu_em_quant_1_ug_stream_TREADY           ), //i
    .ug_stream_TDATA               (demux_1_ug_stream_TDATA[167:0]             ), //o
    .od_stream_TVALID              (demux_1_od_stream_TVALID                   ), //o
    .od_stream_TREADY              (residual_1_res_i_stream_TREADY             ), //i
    .od_stream_TDATA               (demux_1_od_stream_TDATA[199:0]             ), //o
    .logit_stream_valid            (demux_1_logit_stream_valid                 ), //o
    .logit_stream_ready            (inst_m_axi_logit_stream_TREADY             ), //i
    .logit_stream_payload_data     (demux_1_logit_stream_payload_data[255:0]   )  //o
  );
  tryo_rot_wrapper rope_qk_quant_1 (
    .resetn                        (resetn                                             ), //i
    .clk                           (clk                                                ), //i
    .signals_I_L_BEGIN             (demux_1_signals_O_L_BEGIN[31:0]                    ), //i
    .signals_I_L_CLOSE             (demux_1_signals_O_L_CLOSE[31:0]                    ), //i
    .signals_I_MEMORY_DECODER_X    (demux_1_signals_O_MEMORY_DECODER_X[63:0]           ), //i
    .signals_I_MEMORY_DECODER_Y    (demux_1_signals_O_MEMORY_DECODER_Y[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_Y      (demux_1_signals_O_MEMORY_LOGIT_Y[63:0]             ), //i
    .signals_I_MEMORY_DECODER_W_LO (demux_1_signals_O_MEMORY_DECODER_W_LO[63:0]        ), //i
    .signals_I_MEMORY_DECODER_W_HI (demux_1_signals_O_MEMORY_DECODER_W_HI[63:0]        ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (demux_1_signals_O_MEMORY_LOGIT_W_LO[63:0]          ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (demux_1_signals_O_MEMORY_LOGIT_W_HI[63:0]          ), //i
    .signals_I_MEMORY_KV_CACHE     (demux_1_signals_O_MEMORY_KV_CACHE[63:0]            ), //i
    .signals_I_POS_ID              (demux_1_signals_O_POS_ID[31:0]                     ), //i
    .signals_I_T                   (demux_1_signals_O_T                                ), //i
    .signals_O_L_BEGIN             (rope_qk_quant_1_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (rope_qk_quant_1_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (rope_qk_quant_1_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (rope_qk_quant_1_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (rope_qk_quant_1_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (rope_qk_quant_1_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (rope_qk_quant_1_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (rope_qk_quant_1_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (rope_qk_quant_1_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (rope_qk_quant_1_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (rope_qk_quant_1_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (rope_qk_quant_1_signals_O_T                        ), //o
    .qk_stream_TVALID              (demux_1_qk_stream_TVALID                           ), //i
    .qk_stream_TREADY              (rope_qk_quant_1_qk_stream_TREADY                   ), //o
    .qk_stream_TDATA               (demux_1_qk_stream_TDATA[183:0]                     ), //i
    .rot_q_stream_TVALID           (rope_qk_quant_1_rot_q_stream_TVALID                ), //o
    .rot_q_stream_TREADY           (qk_gemm_1_qk_q_stream_TREADY                       ), //i
    .rot_q_stream_TDATA            (rope_qk_quant_1_rot_q_stream_TDATA[63:0]           ), //o
    .rot_s_stream_TVALID           (rope_qk_quant_1_rot_s_stream_TVALID                ), //o
    .rot_s_stream_TREADY           (qk_gemm_1_qk_s_stream_TREADY                       ), //i
    .rot_s_stream_TDATA            (rope_qk_quant_1_rot_s_stream_TDATA[7:0]            )  //o
  );
  tryo_score_wrapper qk_gemm_1 (
    .resetn                        (resetn                                                 ), //i
    .clk                           (clk                                                    ), //i
    .signals_I_L_BEGIN             (rope_qk_quant_1_signals_O_L_BEGIN[31:0]                ), //i
    .signals_I_L_CLOSE             (rope_qk_quant_1_signals_O_L_CLOSE[31:0]                ), //i
    .signals_I_MEMORY_DECODER_X    (rope_qk_quant_1_signals_O_MEMORY_DECODER_X[63:0]       ), //i
    .signals_I_MEMORY_DECODER_Y    (rope_qk_quant_1_signals_O_MEMORY_DECODER_Y[63:0]       ), //i
    .signals_I_MEMORY_LOGIT_Y      (rope_qk_quant_1_signals_O_MEMORY_LOGIT_Y[63:0]         ), //i
    .signals_I_MEMORY_DECODER_W_LO (rope_qk_quant_1_signals_O_MEMORY_DECODER_W_LO[63:0]    ), //i
    .signals_I_MEMORY_DECODER_W_HI (rope_qk_quant_1_signals_O_MEMORY_DECODER_W_HI[63:0]    ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (rope_qk_quant_1_signals_O_MEMORY_LOGIT_W_LO[63:0]      ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (rope_qk_quant_1_signals_O_MEMORY_LOGIT_W_HI[63:0]      ), //i
    .signals_I_MEMORY_KV_CACHE     (rope_qk_quant_1_signals_O_MEMORY_KV_CACHE[63:0]        ), //i
    .signals_I_POS_ID              (rope_qk_quant_1_signals_O_POS_ID[31:0]                 ), //i
    .signals_I_T                   (rope_qk_quant_1_signals_O_T                            ), //i
    .signals_O_L_BEGIN             (qk_gemm_1_signals_O_L_BEGIN[31:0]                      ), //o
    .signals_O_L_CLOSE             (qk_gemm_1_signals_O_L_CLOSE[31:0]                      ), //o
    .signals_O_MEMORY_DECODER_X    (qk_gemm_1_signals_O_MEMORY_DECODER_X[63:0]             ), //o
    .signals_O_MEMORY_DECODER_Y    (qk_gemm_1_signals_O_MEMORY_DECODER_Y[63:0]             ), //o
    .signals_O_MEMORY_LOGIT_Y      (qk_gemm_1_signals_O_MEMORY_LOGIT_Y[63:0]               ), //o
    .signals_O_MEMORY_DECODER_W_LO (qk_gemm_1_signals_O_MEMORY_DECODER_W_LO[63:0]          ), //o
    .signals_O_MEMORY_DECODER_W_HI (qk_gemm_1_signals_O_MEMORY_DECODER_W_HI[63:0]          ), //o
    .signals_O_MEMORY_LOGIT_W_LO   (qk_gemm_1_signals_O_MEMORY_LOGIT_W_LO[63:0]            ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (qk_gemm_1_signals_O_MEMORY_LOGIT_W_HI[63:0]            ), //o
    .signals_O_MEMORY_KV_CACHE     (qk_gemm_1_signals_O_MEMORY_KV_CACHE[63:0]              ), //o
    .signals_O_POS_ID              (qk_gemm_1_signals_O_POS_ID[31:0]                       ), //o
    .signals_O_T                   (qk_gemm_1_signals_O_T                                  ), //o
    .qk_q_stream_TVALID            (rope_qk_quant_1_rot_q_stream_TVALID                    ), //i
    .qk_q_stream_TREADY            (qk_gemm_1_qk_q_stream_TREADY                           ), //o
    .qk_q_stream_TDATA             (rope_qk_quant_1_rot_q_stream_TDATA[63:0]               ), //i
    .qk_s_stream_TVALID            (rope_qk_quant_1_rot_s_stream_TVALID                    ), //i
    .qk_s_stream_TREADY            (qk_gemm_1_qk_s_stream_TREADY                           ), //o
    .qk_s_stream_TDATA             (rope_qk_quant_1_rot_s_stream_TDATA[7:0]                ), //i
    .kq_cache_i_stream_TVALID      (inst_kv_cache_kq_cache_i_stream_TVALID                 ), //i
    .kq_cache_i_stream_TREADY      (qk_gemm_1_kq_cache_i_stream_TREADY                     ), //o
    .kq_cache_i_stream_TDATA       (inst_kv_cache_kq_cache_i_stream_TDATA[63:0]            ), //i
    .ks_cache_i_stream_TVALID      (inst_kv_cache_ks_cache_i_stream_TVALID                 ), //i
    .ks_cache_i_stream_TREADY      (qk_gemm_1_ks_cache_i_stream_TREADY                     ), //o
    .ks_cache_i_stream_TDATA       (inst_kv_cache_ks_cache_i_stream_TDATA[7:0]             ), //i
    .kq_cache_o_stream_TVALID      (qk_gemm_1_kq_cache_o_stream_TVALID                     ), //o
    .kq_cache_o_stream_TREADY      (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_push_ready), //i
    .kq_cache_o_stream_TDATA       (qk_gemm_1_kq_cache_o_stream_TDATA[63:0]                ), //o
    .ks_cache_o_stream_TVALID      (qk_gemm_1_ks_cache_o_stream_TVALID                     ), //o
    .ks_cache_o_stream_TREADY      (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_push_ready), //i
    .ks_cache_o_stream_TDATA       (qk_gemm_1_ks_cache_o_stream_TDATA[7:0]                 ), //o
    .r_stream_TVALID               (qk_gemm_1_r_stream_TVALID                              ), //o
    .r_stream_TREADY               (softmax_quant_1_r_stream_TREADY                        ), //i
    .r_stream_TDATA                (qk_gemm_1_r_stream_TDATA[247:0]                        )  //o
  );
  tryo_prob_wrapper softmax_quant_1 (
    .resetn                        (resetn                                             ), //i
    .clk                           (clk                                                ), //i
    .signals_I_L_BEGIN             (qk_gemm_1_signals_O_L_BEGIN[31:0]                  ), //i
    .signals_I_L_CLOSE             (qk_gemm_1_signals_O_L_CLOSE[31:0]                  ), //i
    .signals_I_MEMORY_DECODER_X    (qk_gemm_1_signals_O_MEMORY_DECODER_X[63:0]         ), //i
    .signals_I_MEMORY_DECODER_Y    (qk_gemm_1_signals_O_MEMORY_DECODER_Y[63:0]         ), //i
    .signals_I_MEMORY_LOGIT_Y      (qk_gemm_1_signals_O_MEMORY_LOGIT_Y[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_LO (qk_gemm_1_signals_O_MEMORY_DECODER_W_LO[63:0]      ), //i
    .signals_I_MEMORY_DECODER_W_HI (qk_gemm_1_signals_O_MEMORY_DECODER_W_HI[63:0]      ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (qk_gemm_1_signals_O_MEMORY_LOGIT_W_LO[63:0]        ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (qk_gemm_1_signals_O_MEMORY_LOGIT_W_HI[63:0]        ), //i
    .signals_I_MEMORY_KV_CACHE     (qk_gemm_1_signals_O_MEMORY_KV_CACHE[63:0]          ), //i
    .signals_I_POS_ID              (qk_gemm_1_signals_O_POS_ID[31:0]                   ), //i
    .signals_I_T                   (qk_gemm_1_signals_O_T                              ), //i
    .signals_O_L_BEGIN             (softmax_quant_1_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (softmax_quant_1_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (softmax_quant_1_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (softmax_quant_1_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (softmax_quant_1_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (softmax_quant_1_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (softmax_quant_1_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (softmax_quant_1_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (softmax_quant_1_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (softmax_quant_1_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (softmax_quant_1_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (softmax_quant_1_signals_O_T                        ), //o
    .r_stream_TVALID               (qk_gemm_1_r_stream_TVALID                          ), //i
    .r_stream_TREADY               (softmax_quant_1_r_stream_TREADY                    ), //o
    .r_stream_TDATA                (qk_gemm_1_r_stream_TDATA[247:0]                    ), //i
    .rq_stream_TVALID              (softmax_quant_1_rq_stream_TVALID                   ), //o
    .rq_stream_TREADY              (rv_gemm_1_rq_stream_TREADY                         ), //i
    .rq_stream_TDATA               (softmax_quant_1_rq_stream_TDATA[63:0]              ), //o
    .rs_stream_TVALID              (softmax_quant_1_rs_stream_TVALID                   ), //o
    .rs_stream_TREADY              (rv_gemm_1_rs_stream_TREADY                         ), //i
    .rs_stream_TDATA               (softmax_quant_1_rs_stream_TDATA[7:0]               )  //o
  );
  tryo_mix_wrapper rv_gemm_1 (
    .resetn                        (resetn                                                 ), //i
    .clk                           (clk                                                    ), //i
    .signals_I_L_BEGIN             (softmax_quant_1_signals_O_L_BEGIN[31:0]                ), //i
    .signals_I_L_CLOSE             (softmax_quant_1_signals_O_L_CLOSE[31:0]                ), //i
    .signals_I_MEMORY_DECODER_X    (softmax_quant_1_signals_O_MEMORY_DECODER_X[63:0]       ), //i
    .signals_I_MEMORY_DECODER_Y    (softmax_quant_1_signals_O_MEMORY_DECODER_Y[63:0]       ), //i
    .signals_I_MEMORY_LOGIT_Y      (softmax_quant_1_signals_O_MEMORY_LOGIT_Y[63:0]         ), //i
    .signals_I_MEMORY_DECODER_W_LO (softmax_quant_1_signals_O_MEMORY_DECODER_W_LO[63:0]    ), //i
    .signals_I_MEMORY_DECODER_W_HI (softmax_quant_1_signals_O_MEMORY_DECODER_W_HI[63:0]    ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (softmax_quant_1_signals_O_MEMORY_LOGIT_W_LO[63:0]      ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (softmax_quant_1_signals_O_MEMORY_LOGIT_W_HI[63:0]      ), //i
    .signals_I_MEMORY_KV_CACHE     (softmax_quant_1_signals_O_MEMORY_KV_CACHE[63:0]        ), //i
    .signals_I_POS_ID              (softmax_quant_1_signals_O_POS_ID[31:0]                 ), //i
    .signals_I_T                   (softmax_quant_1_signals_O_T                            ), //i
    .signals_O_L_BEGIN             (rv_gemm_1_signals_O_L_BEGIN[31:0]                      ), //o
    .signals_O_L_CLOSE             (rv_gemm_1_signals_O_L_CLOSE[31:0]                      ), //o
    .signals_O_MEMORY_DECODER_X    (rv_gemm_1_signals_O_MEMORY_DECODER_X[63:0]             ), //o
    .signals_O_MEMORY_DECODER_Y    (rv_gemm_1_signals_O_MEMORY_DECODER_Y[63:0]             ), //o
    .signals_O_MEMORY_LOGIT_Y      (rv_gemm_1_signals_O_MEMORY_LOGIT_Y[63:0]               ), //o
    .signals_O_MEMORY_DECODER_W_LO (rv_gemm_1_signals_O_MEMORY_DECODER_W_LO[63:0]          ), //o
    .signals_O_MEMORY_DECODER_W_HI (rv_gemm_1_signals_O_MEMORY_DECODER_W_HI[63:0]          ), //o
    .signals_O_MEMORY_LOGIT_W_LO   (rv_gemm_1_signals_O_MEMORY_LOGIT_W_LO[63:0]            ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (rv_gemm_1_signals_O_MEMORY_LOGIT_W_HI[63:0]            ), //o
    .signals_O_MEMORY_KV_CACHE     (rv_gemm_1_signals_O_MEMORY_KV_CACHE[63:0]              ), //o
    .signals_O_POS_ID              (rv_gemm_1_signals_O_POS_ID[31:0]                       ), //o
    .signals_O_T                   (rv_gemm_1_signals_O_T                                  ), //o
    .rq_stream_TVALID              (softmax_quant_1_rq_stream_TVALID                       ), //i
    .rq_stream_TREADY              (rv_gemm_1_rq_stream_TREADY                             ), //o
    .rq_stream_TDATA               (softmax_quant_1_rq_stream_TDATA[63:0]                  ), //i
    .rs_stream_TVALID              (softmax_quant_1_rs_stream_TVALID                       ), //i
    .rs_stream_TREADY              (rv_gemm_1_rs_stream_TREADY                             ), //o
    .rs_stream_TDATA               (softmax_quant_1_rs_stream_TDATA[7:0]                   ), //i
    .v_stream_TVALID               (demux_1_v_stream_TVALID                                ), //i
    .v_stream_TREADY               (rv_gemm_1_v_stream_TREADY                              ), //o
    .v_stream_TDATA                (demux_1_v_stream_TDATA[183:0]                          ), //i
    .vq_cache_i_stream_TVALID      (inst_kv_cache_vq_cache_i_stream_TVALID                 ), //i
    .vq_cache_i_stream_TREADY      (rv_gemm_1_vq_cache_i_stream_TREADY                     ), //o
    .vq_cache_i_stream_TDATA       (inst_kv_cache_vq_cache_i_stream_TDATA[63:0]            ), //i
    .vs_cache_i_stream_TVALID      (inst_kv_cache_vs_cache_i_stream_TVALID                 ), //i
    .vs_cache_i_stream_TREADY      (rv_gemm_1_vs_cache_i_stream_TREADY                     ), //o
    .vs_cache_i_stream_TDATA       (inst_kv_cache_vs_cache_i_stream_TDATA[7:0]             ), //i
    .vq_cache_o_stream_TVALID      (rv_gemm_1_vq_cache_o_stream_TVALID                     ), //o
    .vq_cache_o_stream_TREADY      (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_push_ready), //i
    .vq_cache_o_stream_TDATA       (rv_gemm_1_vq_cache_o_stream_TDATA[63:0]                ), //o
    .vs_cache_o_stream_TVALID      (rv_gemm_1_vs_cache_o_stream_TVALID                     ), //o
    .vs_cache_o_stream_TREADY      (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_push_ready), //i
    .vs_cache_o_stream_TDATA       (rv_gemm_1_vs_cache_o_stream_TDATA[7:0]                 ), //o
    .aq_stream_TVALID              (rv_gemm_1_aq_stream_TVALID                             ), //o
    .aq_stream_TREADY              (mux_1_aq_stream_TREADY                                 ), //i
    .aq_stream_TDATA               (rv_gemm_1_aq_stream_TDATA[63:0]                        ), //o
    .as_stream_TVALID              (rv_gemm_1_as_stream_TVALID                             ), //o
    .as_stream_TREADY              (mux_1_as_stream_TREADY                                 ), //i
    .as_stream_TDATA               (rv_gemm_1_as_stream_TDATA[7:0]                         )  //o
  );
  tryo_gate_wrapper silu_em_quant_1 (
    .resetn                        (resetn                                             ), //i
    .clk                           (clk                                                ), //i
    .signals_I_L_BEGIN             (rv_gemm_1_signals_O_L_BEGIN[31:0]                  ), //i
    .signals_I_L_CLOSE             (rv_gemm_1_signals_O_L_CLOSE[31:0]                  ), //i
    .signals_I_MEMORY_DECODER_X    (rv_gemm_1_signals_O_MEMORY_DECODER_X[63:0]         ), //i
    .signals_I_MEMORY_DECODER_Y    (rv_gemm_1_signals_O_MEMORY_DECODER_Y[63:0]         ), //i
    .signals_I_MEMORY_LOGIT_Y      (rv_gemm_1_signals_O_MEMORY_LOGIT_Y[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_LO (rv_gemm_1_signals_O_MEMORY_DECODER_W_LO[63:0]      ), //i
    .signals_I_MEMORY_DECODER_W_HI (rv_gemm_1_signals_O_MEMORY_DECODER_W_HI[63:0]      ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (rv_gemm_1_signals_O_MEMORY_LOGIT_W_LO[63:0]        ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (rv_gemm_1_signals_O_MEMORY_LOGIT_W_HI[63:0]        ), //i
    .signals_I_MEMORY_KV_CACHE     (rv_gemm_1_signals_O_MEMORY_KV_CACHE[63:0]          ), //i
    .signals_I_POS_ID              (rv_gemm_1_signals_O_POS_ID[31:0]                   ), //i
    .signals_I_T                   (rv_gemm_1_signals_O_T                              ), //i
    .signals_O_L_BEGIN             (silu_em_quant_1_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (silu_em_quant_1_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (silu_em_quant_1_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (silu_em_quant_1_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (silu_em_quant_1_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (silu_em_quant_1_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (silu_em_quant_1_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (silu_em_quant_1_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (silu_em_quant_1_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (silu_em_quant_1_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (silu_em_quant_1_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (silu_em_quant_1_signals_O_T                        ), //o
    .ug_stream_TVALID              (demux_1_ug_stream_TVALID                           ), //i
    .ug_stream_TREADY              (silu_em_quant_1_ug_stream_TREADY                   ), //o
    .ug_stream_TDATA               (demux_1_ug_stream_TDATA[167:0]                     ), //i
    .q_stream_TVALID               (silu_em_quant_1_q_stream_TVALID                    ), //o
    .q_stream_TREADY               (mux_1_xmq_stream_TREADY                            ), //i
    .q_stream_TDATA                (silu_em_quant_1_q_stream_TDATA[63:0]               ), //o
    .s_stream_TVALID               (silu_em_quant_1_s_stream_TVALID                    ), //o
    .s_stream_TREADY               (mux_1_xms_stream_TREADY                            ), //i
    .s_stream_TDATA                (silu_em_quant_1_s_stream_TDATA[7:0]                )  //o
  );
  tryo_add_wrapper residual_1 (
    .resetn                        (resetn                                             ), //i
    .clk                           (clk                                                ), //i
    .signals_I_L_BEGIN             (silu_em_quant_1_signals_O_L_BEGIN[31:0]            ), //i
    .signals_I_L_CLOSE             (silu_em_quant_1_signals_O_L_CLOSE[31:0]            ), //i
    .signals_I_MEMORY_DECODER_X    (silu_em_quant_1_signals_O_MEMORY_DECODER_X[63:0]   ), //i
    .signals_I_MEMORY_DECODER_Y    (silu_em_quant_1_signals_O_MEMORY_DECODER_Y[63:0]   ), //i
    .signals_I_MEMORY_LOGIT_Y      (silu_em_quant_1_signals_O_MEMORY_LOGIT_Y[63:0]     ), //i
    .signals_I_MEMORY_DECODER_W_LO (silu_em_quant_1_signals_O_MEMORY_DECODER_W_LO[63:0]), //i
    .signals_I_MEMORY_DECODER_W_HI (silu_em_quant_1_signals_O_MEMORY_DECODER_W_HI[63:0]), //i
    .signals_I_MEMORY_LOGIT_W_LO   (silu_em_quant_1_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (silu_em_quant_1_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //i
    .signals_I_MEMORY_KV_CACHE     (silu_em_quant_1_signals_O_MEMORY_KV_CACHE[63:0]    ), //i
    .signals_I_POS_ID              (silu_em_quant_1_signals_O_POS_ID[31:0]             ), //i
    .signals_I_T                   (silu_em_quant_1_signals_O_T                        ), //i
    .signals_O_L_BEGIN             (residual_1_signals_O_L_BEGIN[31:0]                 ), //o
    .signals_O_L_CLOSE             (residual_1_signals_O_L_CLOSE[31:0]                 ), //o
    .signals_O_MEMORY_DECODER_X    (residual_1_signals_O_MEMORY_DECODER_X[63:0]        ), //o
    .signals_O_MEMORY_DECODER_Y    (residual_1_signals_O_MEMORY_DECODER_Y[63:0]        ), //o
    .signals_O_MEMORY_LOGIT_Y      (residual_1_signals_O_MEMORY_LOGIT_Y[63:0]          ), //o
    .signals_O_MEMORY_DECODER_W_LO (residual_1_signals_O_MEMORY_DECODER_W_LO[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_HI (residual_1_signals_O_MEMORY_DECODER_W_HI[63:0]     ), //o
    .signals_O_MEMORY_LOGIT_W_LO   (residual_1_signals_O_MEMORY_LOGIT_W_LO[63:0]       ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (residual_1_signals_O_MEMORY_LOGIT_W_HI[63:0]       ), //o
    .signals_O_MEMORY_KV_CACHE     (residual_1_signals_O_MEMORY_KV_CACHE[63:0]         ), //o
    .signals_O_POS_ID              (residual_1_signals_O_POS_ID[31:0]                  ), //o
    .signals_O_T                   (residual_1_signals_O_T                             ), //o
    .x_stream_TVALID               (inst_m_axi_x_stream_TVALID                         ), //i
    .x_stream_TREADY               (residual_1_x_stream_TREADY                         ), //o
    .x_stream_TDATA                (inst_m_axi_x_stream_TDATA[199:0]                   ), //i
    .res_i_stream_TVALID           (demux_1_od_stream_TVALID                           ), //i
    .res_i_stream_TREADY           (residual_1_res_i_stream_TREADY                     ), //o
    .res_i_stream_TDATA            (demux_1_od_stream_TDATA[199:0]                     ), //i
    .res_o_stream_TVALID           (residual_1_res_o_stream_TVALID                     ), //o
    .res_o_stream_TREADY           (rmsnorm_quant_1_x_stream_TREADY                    ), //i
    .res_o_stream_TDATA            (residual_1_res_o_stream_TDATA[199:0]               ), //o
    .y_stream_TVALID               (residual_1_y_stream_TVALID                         ), //o
    .y_stream_TREADY               (inst_m_axi_y_stream_TREADY                         ), //i
    .y_stream_TDATA                (residual_1_y_stream_TDATA[199:0]                   )  //o
  );
  tryo_norm_wrapper rmsnorm_quant_1 (
    .resetn                        (resetn                                             ), //i
    .clk                           (clk                                                ), //i
    .signals_I_L_BEGIN             (residual_1_signals_O_L_BEGIN[31:0]                 ), //i
    .signals_I_L_CLOSE             (residual_1_signals_O_L_CLOSE[31:0]                 ), //i
    .signals_I_MEMORY_DECODER_X    (residual_1_signals_O_MEMORY_DECODER_X[63:0]        ), //i
    .signals_I_MEMORY_DECODER_Y    (residual_1_signals_O_MEMORY_DECODER_Y[63:0]        ), //i
    .signals_I_MEMORY_LOGIT_Y      (residual_1_signals_O_MEMORY_LOGIT_Y[63:0]          ), //i
    .signals_I_MEMORY_DECODER_W_LO (residual_1_signals_O_MEMORY_DECODER_W_LO[63:0]     ), //i
    .signals_I_MEMORY_DECODER_W_HI (residual_1_signals_O_MEMORY_DECODER_W_HI[63:0]     ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (residual_1_signals_O_MEMORY_LOGIT_W_LO[63:0]       ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (residual_1_signals_O_MEMORY_LOGIT_W_HI[63:0]       ), //i
    .signals_I_MEMORY_KV_CACHE     (residual_1_signals_O_MEMORY_KV_CACHE[63:0]         ), //i
    .signals_I_POS_ID              (residual_1_signals_O_POS_ID[31:0]                  ), //i
    .signals_I_T                   (residual_1_signals_O_T                             ), //i
    .signals_O_L_BEGIN             (rmsnorm_quant_1_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (rmsnorm_quant_1_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (rmsnorm_quant_1_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (rmsnorm_quant_1_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (rmsnorm_quant_1_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (rmsnorm_quant_1_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (rmsnorm_quant_1_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (rmsnorm_quant_1_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (rmsnorm_quant_1_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (rmsnorm_quant_1_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (rmsnorm_quant_1_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (rmsnorm_quant_1_signals_O_T                        ), //o
    .x_stream_TVALID               (residual_1_res_o_stream_TVALID                     ), //i
    .x_stream_TREADY               (rmsnorm_quant_1_x_stream_TREADY                    ), //o
    .x_stream_TDATA                (residual_1_res_o_stream_TDATA[199:0]               ), //i
    .xlnq_stream_TVALID            (rmsnorm_quant_1_xlnq_stream_TVALID                 ), //o
    .xlnq_stream_TREADY            (mux_1_xlnq_stream_TREADY                           ), //i
    .xlnq_stream_TDATA             (rmsnorm_quant_1_xlnq_stream_TDATA[63:0]            ), //o
    .xlns_stream_TVALID            (rmsnorm_quant_1_xlns_stream_TVALID                 ), //o
    .xlns_stream_TREADY            (mux_1_xlns_stream_TREADY                           ), //i
    .xlns_stream_TDATA             (rmsnorm_quant_1_xlns_stream_TDATA[7:0]             )  //o
  );
  tryo_flow_mux_wrapper mux_1 (
    .resetn                        (resetn                                           ), //i
    .clk                           (clk                                              ), //i
    .signals_I_L_BEGIN             (inst_kv_cache_signals_O_L_BEGIN[31:0]            ), //i
    .signals_I_L_CLOSE             (inst_kv_cache_signals_O_L_CLOSE[31:0]            ), //i
    .signals_I_MEMORY_DECODER_X    (inst_kv_cache_signals_O_MEMORY_DECODER_X[63:0]   ), //i
    .signals_I_MEMORY_DECODER_Y    (inst_kv_cache_signals_O_MEMORY_DECODER_Y[63:0]   ), //i
    .signals_I_MEMORY_LOGIT_Y      (inst_kv_cache_signals_O_MEMORY_LOGIT_Y[63:0]     ), //i
    .signals_I_MEMORY_DECODER_W_LO (inst_kv_cache_signals_O_MEMORY_DECODER_W_LO[63:0]), //i
    .signals_I_MEMORY_DECODER_W_HI (inst_kv_cache_signals_O_MEMORY_DECODER_W_HI[63:0]), //i
    .signals_I_MEMORY_LOGIT_W_LO   (inst_kv_cache_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (inst_kv_cache_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //i
    .signals_I_MEMORY_KV_CACHE     (inst_kv_cache_signals_O_MEMORY_KV_CACHE[63:0]    ), //i
    .signals_I_POS_ID              (inst_kv_cache_signals_O_POS_ID[31:0]             ), //i
    .signals_I_T                   (inst_kv_cache_signals_O_T                        ), //i
    .signals_O_L_BEGIN             (mux_1_signals_O_L_BEGIN[31:0]                    ), //o
    .signals_O_L_CLOSE             (mux_1_signals_O_L_CLOSE[31:0]                    ), //o
    .signals_O_MEMORY_DECODER_X    (mux_1_signals_O_MEMORY_DECODER_X[63:0]           ), //o
    .signals_O_MEMORY_DECODER_Y    (mux_1_signals_O_MEMORY_DECODER_Y[63:0]           ), //o
    .signals_O_MEMORY_LOGIT_Y      (mux_1_signals_O_MEMORY_LOGIT_Y[63:0]             ), //o
    .signals_O_MEMORY_DECODER_W_LO (mux_1_signals_O_MEMORY_DECODER_W_LO[63:0]        ), //o
    .signals_O_MEMORY_DECODER_W_HI (mux_1_signals_O_MEMORY_DECODER_W_HI[63:0]        ), //o
    .signals_O_MEMORY_LOGIT_W_LO   (mux_1_signals_O_MEMORY_LOGIT_W_LO[63:0]          ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (mux_1_signals_O_MEMORY_LOGIT_W_HI[63:0]          ), //o
    .signals_O_MEMORY_KV_CACHE     (mux_1_signals_O_MEMORY_KV_CACHE[63:0]            ), //o
    .signals_O_POS_ID              (mux_1_signals_O_POS_ID[31:0]                     ), //o
    .signals_O_T                   (mux_1_signals_O_T                                ), //o
    .xlnq_stream_TVALID            (rmsnorm_quant_1_xlnq_stream_TVALID               ), //i
    .xlnq_stream_TREADY            (mux_1_xlnq_stream_TREADY                         ), //o
    .xlnq_stream_TDATA             (rmsnorm_quant_1_xlnq_stream_TDATA[63:0]          ), //i
    .xlns_stream_TVALID            (rmsnorm_quant_1_xlns_stream_TVALID               ), //i
    .xlns_stream_TREADY            (mux_1_xlns_stream_TREADY                         ), //o
    .xlns_stream_TDATA             (rmsnorm_quant_1_xlns_stream_TDATA[7:0]           ), //i
    .aq_stream_TVALID              (rv_gemm_1_aq_stream_TVALID                       ), //i
    .aq_stream_TREADY              (mux_1_aq_stream_TREADY                           ), //o
    .aq_stream_TDATA               (rv_gemm_1_aq_stream_TDATA[63:0]                  ), //i
    .as_stream_TVALID              (rv_gemm_1_as_stream_TVALID                       ), //i
    .as_stream_TREADY              (mux_1_as_stream_TREADY                           ), //o
    .as_stream_TDATA               (rv_gemm_1_as_stream_TDATA[7:0]                   ), //i
    .xmq_stream_TVALID             (silu_em_quant_1_q_stream_TVALID                  ), //i
    .xmq_stream_TREADY             (mux_1_xmq_stream_TREADY                          ), //o
    .xmq_stream_TDATA              (silu_em_quant_1_q_stream_TDATA[63:0]             ), //i
    .xms_stream_TVALID             (silu_em_quant_1_s_stream_TVALID                  ), //i
    .xms_stream_TREADY             (mux_1_xms_stream_TREADY                          ), //o
    .xms_stream_TDATA              (silu_em_quant_1_s_stream_TDATA[7:0]              ), //i
    .q_stream_TVALID               (mux_1_q_stream_TVALID                            ), //o
    .q_stream_TREADY               (gemm_i_stream_TREADY                             ), //i
    .q_stream_TDATA                (mux_1_q_stream_TDATA[511:0]                      ), //o
    .s_stream_TVALID               (mux_1_s_stream_TVALID                            ), //o
    .s_stream_TREADY               (toplevel_mux_1_s_stream_fifo_io_push_ready       ), //i
    .s_stream_TDATA                (mux_1_s_stream_TDATA[39:0]                       )  //o
  );
  StreamFifo toplevel_inst_m_axi_wq_stream_fifo (
    .io_push_valid        (inst_m_axi_wq_stream_TVALID                                  ), //i
    .io_push_ready        (toplevel_inst_m_axi_wq_stream_fifo_io_push_ready             ), //o
    .io_push_payload_data (inst_m_axi_wq_stream_TDATA[319:0]                            ), //i
    .io_pop_valid         (toplevel_inst_m_axi_wq_stream_fifo_io_pop_valid              ), //o
    .io_pop_ready         (gemm_w_stream_TREADY                                         ), //i
    .io_pop_payload_data  (toplevel_inst_m_axi_wq_stream_fifo_io_pop_payload_data[319:0]), //o
    .io_flush             (toplevel_inst_m_axi_wq_stream_fifo_io_flush                  ), //i
    .io_occupancy         (toplevel_inst_m_axi_wq_stream_fifo_io_occupancy[11:0]        ), //o
    .io_availability      (toplevel_inst_m_axi_wq_stream_fifo_io_availability[11:0]     ), //o
    .clk                  (clk                                                          ), //i
    .resetn               (resetn                                                       )  //i
  );
  StreamFifo_1 toplevel_inst_m_axi_ws1_stream_fifo (
    .io_push_valid        (inst_m_axi_ws1_stream_TVALID                                 ), //i
    .io_push_ready        (toplevel_inst_m_axi_ws1_stream_fifo_io_push_ready            ), //o
    .io_push_payload_data (inst_m_axi_ws1_stream_TDATA[39:0]                            ), //i
    .io_pop_valid         (toplevel_inst_m_axi_ws1_stream_fifo_io_pop_valid             ), //o
    .io_pop_ready         (gemm_s1_stream_TREADY                                        ), //i
    .io_pop_payload_data  (toplevel_inst_m_axi_ws1_stream_fifo_io_pop_payload_data[39:0]), //o
    .io_flush             (toplevel_inst_m_axi_ws1_stream_fifo_io_flush                 ), //i
    .io_occupancy         (toplevel_inst_m_axi_ws1_stream_fifo_io_occupancy[11:0]       ), //o
    .io_availability      (toplevel_inst_m_axi_ws1_stream_fifo_io_availability[11:0]    ), //o
    .clk                  (clk                                                          ), //i
    .resetn               (resetn                                                       )  //i
  );
  StreamFifo_1 toplevel_inst_m_axi_ws2_stream_fifo (
    .io_push_valid        (inst_m_axi_ws2_stream_TVALID                                 ), //i
    .io_push_ready        (toplevel_inst_m_axi_ws2_stream_fifo_io_push_ready            ), //o
    .io_push_payload_data (inst_m_axi_ws2_stream_TDATA[39:0]                            ), //i
    .io_pop_valid         (toplevel_inst_m_axi_ws2_stream_fifo_io_pop_valid             ), //o
    .io_pop_ready         (gemm_s2_stream_TREADY                                        ), //i
    .io_pop_payload_data  (toplevel_inst_m_axi_ws2_stream_fifo_io_pop_payload_data[39:0]), //o
    .io_flush             (toplevel_inst_m_axi_ws2_stream_fifo_io_flush                 ), //i
    .io_occupancy         (toplevel_inst_m_axi_ws2_stream_fifo_io_occupancy[11:0]       ), //o
    .io_availability      (toplevel_inst_m_axi_ws2_stream_fifo_io_availability[11:0]    ), //o
    .clk                  (clk                                                          ), //i
    .resetn               (resetn                                                       )  //i
  );
  StreamFifo_3 toplevel_mux_1_s_stream_fifo (
    .io_push_valid        (mux_1_s_stream_TVALID                                 ), //i
    .io_push_ready        (toplevel_mux_1_s_stream_fifo_io_push_ready            ), //o
    .io_push_payload_data (mux_1_s_stream_TDATA[39:0]                            ), //i
    .io_pop_valid         (toplevel_mux_1_s_stream_fifo_io_pop_valid             ), //o
    .io_pop_ready         (gemm_s_stream_TREADY                                  ), //i
    .io_pop_payload_data  (toplevel_mux_1_s_stream_fifo_io_pop_payload_data[39:0]), //o
    .io_flush             (toplevel_mux_1_s_stream_fifo_io_flush                 ), //i
    .io_occupancy         (toplevel_mux_1_s_stream_fifo_io_occupancy[4:0]        ), //o
    .io_availability      (toplevel_mux_1_s_stream_fifo_io_availability[4:0]     ), //o
    .clk                  (clk                                                   ), //i
    .resetn               (resetn                                                )  //i
  );
  StreamFifo_4 toplevel_qk_gemm_1_kq_cache_o_stream_fifo (
    .io_push_valid        (qk_gemm_1_kq_cache_o_stream_TVALID                                 ), //i
    .io_push_ready        (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_push_ready            ), //o
    .io_push_payload_data (qk_gemm_1_kq_cache_o_stream_TDATA[63:0]                            ), //i
    .io_pop_valid         (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_pop_valid             ), //o
    .io_pop_ready         (inst_kv_cache_kq_cache_o_stream_TREADY                             ), //i
    .io_pop_payload_data  (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_pop_payload_data[63:0]), //o
    .io_flush             (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_flush                 ), //i
    .io_occupancy         (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_occupancy[12:0]       ), //o
    .io_availability      (toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_availability[12:0]    ), //o
    .clk                  (clk                                                                ), //i
    .resetn               (resetn                                                             )  //i
  );
  StreamFifo_5 toplevel_qk_gemm_1_ks_cache_o_stream_fifo (
    .io_push_valid        (qk_gemm_1_ks_cache_o_stream_TVALID                                ), //i
    .io_push_ready        (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_push_ready           ), //o
    .io_push_payload_data (qk_gemm_1_ks_cache_o_stream_TDATA[7:0]                            ), //i
    .io_pop_valid         (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_pop_valid            ), //o
    .io_pop_ready         (inst_kv_cache_ks_cache_o_stream_TREADY                            ), //i
    .io_pop_payload_data  (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_pop_payload_data[7:0]), //o
    .io_flush             (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_flush                ), //i
    .io_occupancy         (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_occupancy[12:0]      ), //o
    .io_availability      (toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_availability[12:0]   ), //o
    .clk                  (clk                                                               ), //i
    .resetn               (resetn                                                            )  //i
  );
  StreamFifo_4 toplevel_rv_gemm_1_vq_cache_o_stream_fifo (
    .io_push_valid        (rv_gemm_1_vq_cache_o_stream_TVALID                                 ), //i
    .io_push_ready        (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_push_ready            ), //o
    .io_push_payload_data (rv_gemm_1_vq_cache_o_stream_TDATA[63:0]                            ), //i
    .io_pop_valid         (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_pop_valid             ), //o
    .io_pop_ready         (inst_kv_cache_vq_cache_o_stream_TREADY                             ), //i
    .io_pop_payload_data  (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_pop_payload_data[63:0]), //o
    .io_flush             (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_flush                 ), //i
    .io_occupancy         (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_occupancy[12:0]       ), //o
    .io_availability      (toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_availability[12:0]    ), //o
    .clk                  (clk                                                                ), //i
    .resetn               (resetn                                                             )  //i
  );
  StreamFifo_5 toplevel_rv_gemm_1_vs_cache_o_stream_fifo (
    .io_push_valid        (rv_gemm_1_vs_cache_o_stream_TVALID                                ), //i
    .io_push_ready        (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_push_ready           ), //o
    .io_push_payload_data (rv_gemm_1_vs_cache_o_stream_TDATA[7:0]                            ), //i
    .io_pop_valid         (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_pop_valid            ), //o
    .io_pop_ready         (inst_kv_cache_vs_cache_o_stream_TREADY                            ), //i
    .io_pop_payload_data  (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_pop_payload_data[7:0]), //o
    .io_flush             (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_flush                ), //i
    .io_occupancy         (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_occupancy[12:0]      ), //o
    .io_availability      (toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_availability[12:0]   ), //o
    .clk                  (clk                                                               ), //i
    .resetn               (resetn                                                            )  //i
  );
  assign axilite_awready = inst_controller_axilite_awready;
  assign axilite_wready = inst_controller_axilite_wready;
  assign axilite_bvalid = inst_controller_axilite_bvalid;
  assign axilite_bresp = inst_controller_axilite_bresp;
  assign axilite_arready = inst_controller_axilite_arready;
  assign axilite_rvalid = inst_controller_axilite_rvalid;
  assign axilite_rdata = inst_controller_axilite_rdata;
  assign axilite_rresp = inst_controller_axilite_rresp;
  assign gmem1_awvalid = inst_m_axi_gmem1_awvalid;
  assign gmem1_awaddr = inst_m_axi_gmem1_awaddr;
  assign gmem1_awid = inst_m_axi_gmem1_awid;
  assign gmem1_awregion = inst_m_axi_gmem1_awregion;
  assign gmem1_awlen = inst_m_axi_gmem1_awlen;
  assign gmem1_awsize = inst_m_axi_gmem1_awsize;
  assign gmem1_awburst = inst_m_axi_gmem1_awburst;
  assign gmem1_awlock = inst_m_axi_gmem1_awlock;
  assign gmem1_awcache = inst_m_axi_gmem1_awcache;
  assign gmem1_awqos = inst_m_axi_gmem1_awqos;
  assign gmem1_awuser = inst_m_axi_gmem1_awuser;
  assign gmem1_awprot = inst_m_axi_gmem1_awprot;
  assign gmem1_wvalid = inst_m_axi_gmem1_wvalid;
  assign gmem1_wdata = inst_m_axi_gmem1_wdata;
  assign gmem1_wstrb = inst_m_axi_gmem1_wstrb;
  assign gmem1_wuser = inst_m_axi_gmem1_wuser;
  assign gmem1_wlast = inst_m_axi_gmem1_wlast;
  assign gmem1_bready = inst_m_axi_gmem1_bready;
  assign gmem1_arvalid = inst_m_axi_gmem1_arvalid;
  assign gmem1_araddr = inst_m_axi_gmem1_araddr;
  assign gmem1_arid = inst_m_axi_gmem1_arid;
  assign gmem1_arregion = inst_m_axi_gmem1_arregion;
  assign gmem1_arlen = inst_m_axi_gmem1_arlen;
  assign gmem1_arsize = inst_m_axi_gmem1_arsize;
  assign gmem1_arburst = inst_m_axi_gmem1_arburst;
  assign gmem1_arlock = inst_m_axi_gmem1_arlock;
  assign gmem1_arcache = inst_m_axi_gmem1_arcache;
  assign gmem1_arqos = inst_m_axi_gmem1_arqos;
  assign gmem1_aruser = inst_m_axi_gmem1_aruser;
  assign gmem1_arprot = inst_m_axi_gmem1_arprot;
  assign gmem1_rready = inst_m_axi_gmem1_rready;
  assign gmem2_awvalid = inst_m_axi_gmem2_awvalid;
  assign gmem2_awaddr = inst_m_axi_gmem2_awaddr;
  assign gmem2_awid = inst_m_axi_gmem2_awid;
  assign gmem2_awregion = inst_m_axi_gmem2_awregion;
  assign gmem2_awlen = inst_m_axi_gmem2_awlen;
  assign gmem2_awsize = inst_m_axi_gmem2_awsize;
  assign gmem2_awburst = inst_m_axi_gmem2_awburst;
  assign gmem2_awlock = inst_m_axi_gmem2_awlock;
  assign gmem2_awcache = inst_m_axi_gmem2_awcache;
  assign gmem2_awqos = inst_m_axi_gmem2_awqos;
  assign gmem2_awuser = inst_m_axi_gmem2_awuser;
  assign gmem2_awprot = inst_m_axi_gmem2_awprot;
  assign gmem2_wvalid = inst_m_axi_gmem2_wvalid;
  assign gmem2_wdata = inst_m_axi_gmem2_wdata;
  assign gmem2_wstrb = inst_m_axi_gmem2_wstrb;
  assign gmem2_wuser = inst_m_axi_gmem2_wuser;
  assign gmem2_wlast = inst_m_axi_gmem2_wlast;
  assign gmem2_bready = inst_m_axi_gmem2_bready;
  assign gmem2_arvalid = inst_m_axi_gmem2_arvalid;
  assign gmem2_araddr = inst_m_axi_gmem2_araddr;
  assign gmem2_arid = inst_m_axi_gmem2_arid;
  assign gmem2_arregion = inst_m_axi_gmem2_arregion;
  assign gmem2_arlen = inst_m_axi_gmem2_arlen;
  assign gmem2_arsize = inst_m_axi_gmem2_arsize;
  assign gmem2_arburst = inst_m_axi_gmem2_arburst;
  assign gmem2_arlock = inst_m_axi_gmem2_arlock;
  assign gmem2_arcache = inst_m_axi_gmem2_arcache;
  assign gmem2_arqos = inst_m_axi_gmem2_arqos;
  assign gmem2_aruser = inst_m_axi_gmem2_aruser;
  assign gmem2_arprot = inst_m_axi_gmem2_arprot;
  assign gmem2_rready = inst_m_axi_gmem2_rready;
  assign gmem3_awvalid = inst_kv_cache_gmem1_awvalid;
  assign gmem3_awaddr = inst_kv_cache_gmem1_awaddr;
  assign gmem3_awid = inst_kv_cache_gmem1_awid;
  assign gmem3_awregion = inst_kv_cache_gmem1_awregion;
  assign gmem3_awlen = inst_kv_cache_gmem1_awlen;
  assign gmem3_awsize = inst_kv_cache_gmem1_awsize;
  assign gmem3_awburst = inst_kv_cache_gmem1_awburst;
  assign gmem3_awlock = inst_kv_cache_gmem1_awlock;
  assign gmem3_awcache = inst_kv_cache_gmem1_awcache;
  assign gmem3_awqos = inst_kv_cache_gmem1_awqos;
  assign gmem3_awuser = inst_kv_cache_gmem1_awuser;
  assign gmem3_awprot = inst_kv_cache_gmem1_awprot;
  assign gmem3_wvalid = inst_kv_cache_gmem1_wvalid;
  assign gmem3_wdata = inst_kv_cache_gmem1_wdata;
  assign gmem3_wstrb = inst_kv_cache_gmem1_wstrb;
  assign gmem3_wuser = inst_kv_cache_gmem1_wuser;
  assign gmem3_wlast = inst_kv_cache_gmem1_wlast;
  assign gmem3_bready = inst_kv_cache_gmem1_bready;
  assign gmem3_arvalid = inst_kv_cache_gmem1_arvalid;
  assign gmem3_araddr = inst_kv_cache_gmem1_araddr;
  assign gmem3_arid = inst_kv_cache_gmem1_arid;
  assign gmem3_arregion = inst_kv_cache_gmem1_arregion;
  assign gmem3_arlen = inst_kv_cache_gmem1_arlen;
  assign gmem3_arsize = inst_kv_cache_gmem1_arsize;
  assign gmem3_arburst = inst_kv_cache_gmem1_arburst;
  assign gmem3_arlock = inst_kv_cache_gmem1_arlock;
  assign gmem3_arcache = inst_kv_cache_gmem1_arcache;
  assign gmem3_arqos = inst_kv_cache_gmem1_arqos;
  assign gmem3_aruser = inst_kv_cache_gmem1_aruser;
  assign gmem3_arprot = inst_kv_cache_gmem1_arprot;
  assign gmem3_rready = inst_kv_cache_gmem1_rready;
  assign idle = inst_m_axi_idle;
  assign toplevel_inst_m_axi_wq_stream_fifo_io_flush = 1'b0;
  assign toplevel_inst_m_axi_ws1_stream_fifo_io_flush = 1'b0;
  assign toplevel_inst_m_axi_ws2_stream_fifo_io_flush = 1'b0;
  assign toplevel_mux_1_s_stream_fifo_io_flush = 1'b0;
  assign toplevel_qk_gemm_1_kq_cache_o_stream_fifo_io_flush = 1'b0;
  assign toplevel_qk_gemm_1_ks_cache_o_stream_fifo_io_flush = 1'b0;
  assign toplevel_rv_gemm_1_vq_cache_o_stream_fifo_io_flush = 1'b0;
  assign toplevel_rv_gemm_1_vs_cache_o_stream_fifo_io_flush = 1'b0;

endmodule

//StreamFifo_7 replaced by StreamFifo_5

//StreamFifo_6 replaced by StreamFifo_4

module StreamFifo_5 (
  input  wire          io_push_valid,
  output wire          io_push_ready,
  input  wire [7:0]    io_push_payload_data,
  output wire          io_pop_valid,
  input  wire          io_pop_ready,
  output wire [7:0]    io_pop_payload_data,
  input  wire          io_flush,
  output wire [12:0]   io_occupancy,
  output wire [12:0]   io_availability,
  input  wire          clk,
  input  wire          resetn
);

  reg        [7:0]    _zz_logic_ram_port1;
  reg                 _zz_1;
  wire                logic_ptr_doPush;
  wire                logic_ptr_doPop;
  wire                logic_ptr_full;
  wire                logic_ptr_empty;
  reg        [12:0]   logic_ptr_push;
  reg        [12:0]   logic_ptr_pop;
  wire       [12:0]   logic_ptr_occupancy;
  wire       [12:0]   logic_ptr_popOnIo;
  wire                when_Stream_l1205;
  reg                 logic_ptr_wentUp;
  wire                io_push_fire;
  wire                logic_push_onRam_write_valid;
  wire       [11:0]   logic_push_onRam_write_payload_address;
  wire       [7:0]    logic_push_onRam_write_payload_data_data;
  wire                logic_pop_addressGen_valid;
  reg                 logic_pop_addressGen_ready;
  wire       [11:0]   logic_pop_addressGen_payload;
  wire                logic_pop_addressGen_fire;
  wire                logic_pop_sync_readArbitation_valid;
  wire                logic_pop_sync_readArbitation_ready;
  wire       [11:0]   logic_pop_sync_readArbitation_payload;
  reg                 logic_pop_addressGen_rValid;
  reg        [11:0]   logic_pop_addressGen_rData;
  wire                when_Stream_l369;
  wire                logic_pop_sync_readPort_cmd_valid;
  wire       [11:0]   logic_pop_sync_readPort_cmd_payload;
  wire       [7:0]    logic_pop_sync_readPort_rsp_data;
  wire                logic_pop_sync_readArbitation_translated_valid;
  wire                logic_pop_sync_readArbitation_translated_ready;
  wire       [7:0]    logic_pop_sync_readArbitation_translated_payload_data;
  wire                logic_pop_sync_readArbitation_fire;
  reg        [12:0]   logic_pop_sync_popReg;
  reg [7:0] logic_ram [0:4095];

  always @(posedge clk) begin
    if(_zz_1) begin
      logic_ram[logic_push_onRam_write_payload_address] <= logic_push_onRam_write_payload_data_data;
    end
  end

  always @(posedge clk) begin
    if(logic_pop_sync_readPort_cmd_valid) begin
      _zz_logic_ram_port1 <= logic_ram[logic_pop_sync_readPort_cmd_payload];
    end
  end

  always @(*) begin
    _zz_1 = 1'b0;
    if(logic_push_onRam_write_valid) begin
      _zz_1 = 1'b1;
    end
  end

  assign when_Stream_l1205 = (logic_ptr_doPush != logic_ptr_doPop);
  assign logic_ptr_full = (((logic_ptr_push ^ logic_ptr_popOnIo) ^ 13'h1000) == 13'h0000);
  assign logic_ptr_empty = (logic_ptr_push == logic_ptr_pop);
  assign logic_ptr_occupancy = (logic_ptr_push - logic_ptr_popOnIo);
  assign io_push_ready = (! logic_ptr_full);
  assign io_push_fire = (io_push_valid && io_push_ready);
  assign logic_ptr_doPush = io_push_fire;
  assign logic_push_onRam_write_valid = io_push_fire;
  assign logic_push_onRam_write_payload_address = logic_ptr_push[11:0];
  assign logic_push_onRam_write_payload_data_data = io_push_payload_data;
  assign logic_pop_addressGen_valid = (! logic_ptr_empty);
  assign logic_pop_addressGen_payload = logic_ptr_pop[11:0];
  assign logic_pop_addressGen_fire = (logic_pop_addressGen_valid && logic_pop_addressGen_ready);
  assign logic_ptr_doPop = logic_pop_addressGen_fire;
  always @(*) begin
    logic_pop_addressGen_ready = logic_pop_sync_readArbitation_ready;
    if(when_Stream_l369) begin
      logic_pop_addressGen_ready = 1'b1;
    end
  end

  assign when_Stream_l369 = (! logic_pop_sync_readArbitation_valid);
  assign logic_pop_sync_readArbitation_valid = logic_pop_addressGen_rValid;
  assign logic_pop_sync_readArbitation_payload = logic_pop_addressGen_rData;
  assign logic_pop_sync_readPort_rsp_data = _zz_logic_ram_port1[7 : 0];
  assign logic_pop_sync_readPort_cmd_valid = logic_pop_addressGen_fire;
  assign logic_pop_sync_readPort_cmd_payload = logic_pop_addressGen_payload;
  assign logic_pop_sync_readArbitation_translated_valid = logic_pop_sync_readArbitation_valid;
  assign logic_pop_sync_readArbitation_ready = logic_pop_sync_readArbitation_translated_ready;
  assign logic_pop_sync_readArbitation_translated_payload_data = logic_pop_sync_readPort_rsp_data;
  assign io_pop_valid = logic_pop_sync_readArbitation_translated_valid;
  assign logic_pop_sync_readArbitation_translated_ready = io_pop_ready;
  assign io_pop_payload_data = logic_pop_sync_readArbitation_translated_payload_data;
  assign logic_pop_sync_readArbitation_fire = (logic_pop_sync_readArbitation_valid && logic_pop_sync_readArbitation_ready);
  assign logic_ptr_popOnIo = logic_pop_sync_popReg;
  assign io_occupancy = logic_ptr_occupancy;
  assign io_availability = (13'h1000 - logic_ptr_occupancy);
  always @(posedge clk) begin
    if(!resetn) begin
      logic_ptr_push <= 13'h0000;
      logic_ptr_pop <= 13'h0000;
      logic_ptr_wentUp <= 1'b0;
      logic_pop_addressGen_rValid <= 1'b0;
      logic_pop_sync_popReg <= 13'h0000;
    end else begin
      if(when_Stream_l1205) begin
        logic_ptr_wentUp <= logic_ptr_doPush;
      end
      if(io_flush) begin
        logic_ptr_wentUp <= 1'b0;
      end
      if(logic_ptr_doPush) begin
        logic_ptr_push <= (logic_ptr_push + 13'h0001);
      end
      if(logic_ptr_doPop) begin
        logic_ptr_pop <= (logic_ptr_pop + 13'h0001);
      end
      if(io_flush) begin
        logic_ptr_push <= 13'h0000;
        logic_ptr_pop <= 13'h0000;
      end
      if(logic_pop_addressGen_ready) begin
        logic_pop_addressGen_rValid <= logic_pop_addressGen_valid;
      end
      if(io_flush) begin
        logic_pop_addressGen_rValid <= 1'b0;
      end
      if(logic_pop_sync_readArbitation_fire) begin
        logic_pop_sync_popReg <= logic_ptr_pop;
      end
      if(io_flush) begin
        logic_pop_sync_popReg <= 13'h0000;
      end
    end
  end

  always @(posedge clk) begin
    if(logic_pop_addressGen_ready) begin
      logic_pop_addressGen_rData <= logic_pop_addressGen_payload;
    end
  end


endmodule

module StreamFifo_4 (
  input  wire          io_push_valid,
  output wire          io_push_ready,
  input  wire [63:0]   io_push_payload_data,
  output wire          io_pop_valid,
  input  wire          io_pop_ready,
  output wire [63:0]   io_pop_payload_data,
  input  wire          io_flush,
  output wire [12:0]   io_occupancy,
  output wire [12:0]   io_availability,
  input  wire          clk,
  input  wire          resetn
);

  reg        [63:0]   _zz_logic_ram_port1;
  reg                 _zz_1;
  wire                logic_ptr_doPush;
  wire                logic_ptr_doPop;
  wire                logic_ptr_full;
  wire                logic_ptr_empty;
  reg        [12:0]   logic_ptr_push;
  reg        [12:0]   logic_ptr_pop;
  wire       [12:0]   logic_ptr_occupancy;
  wire       [12:0]   logic_ptr_popOnIo;
  wire                when_Stream_l1205;
  reg                 logic_ptr_wentUp;
  wire                io_push_fire;
  wire                logic_push_onRam_write_valid;
  wire       [11:0]   logic_push_onRam_write_payload_address;
  wire       [63:0]   logic_push_onRam_write_payload_data_data;
  wire                logic_pop_addressGen_valid;
  reg                 logic_pop_addressGen_ready;
  wire       [11:0]   logic_pop_addressGen_payload;
  wire                logic_pop_addressGen_fire;
  wire                logic_pop_sync_readArbitation_valid;
  wire                logic_pop_sync_readArbitation_ready;
  wire       [11:0]   logic_pop_sync_readArbitation_payload;
  reg                 logic_pop_addressGen_rValid;
  reg        [11:0]   logic_pop_addressGen_rData;
  wire                when_Stream_l369;
  wire                logic_pop_sync_readPort_cmd_valid;
  wire       [11:0]   logic_pop_sync_readPort_cmd_payload;
  wire       [63:0]   logic_pop_sync_readPort_rsp_data;
  wire                logic_pop_sync_readArbitation_translated_valid;
  wire                logic_pop_sync_readArbitation_translated_ready;
  wire       [63:0]   logic_pop_sync_readArbitation_translated_payload_data;
  wire                logic_pop_sync_readArbitation_fire;
  reg        [12:0]   logic_pop_sync_popReg;
  reg [63:0] logic_ram [0:4095];

  always @(posedge clk) begin
    if(_zz_1) begin
      logic_ram[logic_push_onRam_write_payload_address] <= logic_push_onRam_write_payload_data_data;
    end
  end

  always @(posedge clk) begin
    if(logic_pop_sync_readPort_cmd_valid) begin
      _zz_logic_ram_port1 <= logic_ram[logic_pop_sync_readPort_cmd_payload];
    end
  end

  always @(*) begin
    _zz_1 = 1'b0;
    if(logic_push_onRam_write_valid) begin
      _zz_1 = 1'b1;
    end
  end

  assign when_Stream_l1205 = (logic_ptr_doPush != logic_ptr_doPop);
  assign logic_ptr_full = (((logic_ptr_push ^ logic_ptr_popOnIo) ^ 13'h1000) == 13'h0000);
  assign logic_ptr_empty = (logic_ptr_push == logic_ptr_pop);
  assign logic_ptr_occupancy = (logic_ptr_push - logic_ptr_popOnIo);
  assign io_push_ready = (! logic_ptr_full);
  assign io_push_fire = (io_push_valid && io_push_ready);
  assign logic_ptr_doPush = io_push_fire;
  assign logic_push_onRam_write_valid = io_push_fire;
  assign logic_push_onRam_write_payload_address = logic_ptr_push[11:0];
  assign logic_push_onRam_write_payload_data_data = io_push_payload_data;
  assign logic_pop_addressGen_valid = (! logic_ptr_empty);
  assign logic_pop_addressGen_payload = logic_ptr_pop[11:0];
  assign logic_pop_addressGen_fire = (logic_pop_addressGen_valid && logic_pop_addressGen_ready);
  assign logic_ptr_doPop = logic_pop_addressGen_fire;
  always @(*) begin
    logic_pop_addressGen_ready = logic_pop_sync_readArbitation_ready;
    if(when_Stream_l369) begin
      logic_pop_addressGen_ready = 1'b1;
    end
  end

  assign when_Stream_l369 = (! logic_pop_sync_readArbitation_valid);
  assign logic_pop_sync_readArbitation_valid = logic_pop_addressGen_rValid;
  assign logic_pop_sync_readArbitation_payload = logic_pop_addressGen_rData;
  assign logic_pop_sync_readPort_rsp_data = _zz_logic_ram_port1[63 : 0];
  assign logic_pop_sync_readPort_cmd_valid = logic_pop_addressGen_fire;
  assign logic_pop_sync_readPort_cmd_payload = logic_pop_addressGen_payload;
  assign logic_pop_sync_readArbitation_translated_valid = logic_pop_sync_readArbitation_valid;
  assign logic_pop_sync_readArbitation_ready = logic_pop_sync_readArbitation_translated_ready;
  assign logic_pop_sync_readArbitation_translated_payload_data = logic_pop_sync_readPort_rsp_data;
  assign io_pop_valid = logic_pop_sync_readArbitation_translated_valid;
  assign logic_pop_sync_readArbitation_translated_ready = io_pop_ready;
  assign io_pop_payload_data = logic_pop_sync_readArbitation_translated_payload_data;
  assign logic_pop_sync_readArbitation_fire = (logic_pop_sync_readArbitation_valid && logic_pop_sync_readArbitation_ready);
  assign logic_ptr_popOnIo = logic_pop_sync_popReg;
  assign io_occupancy = logic_ptr_occupancy;
  assign io_availability = (13'h1000 - logic_ptr_occupancy);
  always @(posedge clk) begin
    if(!resetn) begin
      logic_ptr_push <= 13'h0000;
      logic_ptr_pop <= 13'h0000;
      logic_ptr_wentUp <= 1'b0;
      logic_pop_addressGen_rValid <= 1'b0;
      logic_pop_sync_popReg <= 13'h0000;
    end else begin
      if(when_Stream_l1205) begin
        logic_ptr_wentUp <= logic_ptr_doPush;
      end
      if(io_flush) begin
        logic_ptr_wentUp <= 1'b0;
      end
      if(logic_ptr_doPush) begin
        logic_ptr_push <= (logic_ptr_push + 13'h0001);
      end
      if(logic_ptr_doPop) begin
        logic_ptr_pop <= (logic_ptr_pop + 13'h0001);
      end
      if(io_flush) begin
        logic_ptr_push <= 13'h0000;
        logic_ptr_pop <= 13'h0000;
      end
      if(logic_pop_addressGen_ready) begin
        logic_pop_addressGen_rValid <= logic_pop_addressGen_valid;
      end
      if(io_flush) begin
        logic_pop_addressGen_rValid <= 1'b0;
      end
      if(logic_pop_sync_readArbitation_fire) begin
        logic_pop_sync_popReg <= logic_ptr_pop;
      end
      if(io_flush) begin
        logic_pop_sync_popReg <= 13'h0000;
      end
    end
  end

  always @(posedge clk) begin
    if(logic_pop_addressGen_ready) begin
      logic_pop_addressGen_rData <= logic_pop_addressGen_payload;
    end
  end


endmodule

module StreamFifo_3 (
  input  wire          io_push_valid,
  output wire          io_push_ready,
  input  wire [39:0]   io_push_payload_data,
  output wire          io_pop_valid,
  input  wire          io_pop_ready,
  output wire [39:0]   io_pop_payload_data,
  input  wire          io_flush,
  output wire [4:0]    io_occupancy,
  output wire [4:0]    io_availability,
  input  wire          clk,
  input  wire          resetn
);

  reg        [39:0]   _zz_logic_ram_port1;
  reg                 _zz_1;
  wire                logic_ptr_doPush;
  wire                logic_ptr_doPop;
  wire                logic_ptr_full;
  wire                logic_ptr_empty;
  reg        [4:0]    logic_ptr_push;
  reg        [4:0]    logic_ptr_pop;
  wire       [4:0]    logic_ptr_occupancy;
  wire       [4:0]    logic_ptr_popOnIo;
  wire                when_Stream_l1205;
  reg                 logic_ptr_wentUp;
  wire                io_push_fire;
  wire                logic_push_onRam_write_valid;
  wire       [3:0]    logic_push_onRam_write_payload_address;
  wire       [39:0]   logic_push_onRam_write_payload_data_data;
  wire                logic_pop_addressGen_valid;
  reg                 logic_pop_addressGen_ready;
  wire       [3:0]    logic_pop_addressGen_payload;
  wire                logic_pop_addressGen_fire;
  wire                logic_pop_sync_readArbitation_valid;
  wire                logic_pop_sync_readArbitation_ready;
  wire       [3:0]    logic_pop_sync_readArbitation_payload;
  reg                 logic_pop_addressGen_rValid;
  reg        [3:0]    logic_pop_addressGen_rData;
  wire                when_Stream_l369;
  wire                logic_pop_sync_readPort_cmd_valid;
  wire       [3:0]    logic_pop_sync_readPort_cmd_payload;
  wire       [39:0]   logic_pop_sync_readPort_rsp_data;
  wire                logic_pop_sync_readArbitation_translated_valid;
  wire                logic_pop_sync_readArbitation_translated_ready;
  wire       [39:0]   logic_pop_sync_readArbitation_translated_payload_data;
  wire                logic_pop_sync_readArbitation_fire;
  reg        [4:0]    logic_pop_sync_popReg;
  reg [39:0] logic_ram [0:15];

  always @(posedge clk) begin
    if(_zz_1) begin
      logic_ram[logic_push_onRam_write_payload_address] <= logic_push_onRam_write_payload_data_data;
    end
  end

  always @(posedge clk) begin
    if(logic_pop_sync_readPort_cmd_valid) begin
      _zz_logic_ram_port1 <= logic_ram[logic_pop_sync_readPort_cmd_payload];
    end
  end

  always @(*) begin
    _zz_1 = 1'b0;
    if(logic_push_onRam_write_valid) begin
      _zz_1 = 1'b1;
    end
  end

  assign when_Stream_l1205 = (logic_ptr_doPush != logic_ptr_doPop);
  assign logic_ptr_full = (((logic_ptr_push ^ logic_ptr_popOnIo) ^ 5'h10) == 5'h00);
  assign logic_ptr_empty = (logic_ptr_push == logic_ptr_pop);
  assign logic_ptr_occupancy = (logic_ptr_push - logic_ptr_popOnIo);
  assign io_push_ready = (! logic_ptr_full);
  assign io_push_fire = (io_push_valid && io_push_ready);
  assign logic_ptr_doPush = io_push_fire;
  assign logic_push_onRam_write_valid = io_push_fire;
  assign logic_push_onRam_write_payload_address = logic_ptr_push[3:0];
  assign logic_push_onRam_write_payload_data_data = io_push_payload_data;
  assign logic_pop_addressGen_valid = (! logic_ptr_empty);
  assign logic_pop_addressGen_payload = logic_ptr_pop[3:0];
  assign logic_pop_addressGen_fire = (logic_pop_addressGen_valid && logic_pop_addressGen_ready);
  assign logic_ptr_doPop = logic_pop_addressGen_fire;
  always @(*) begin
    logic_pop_addressGen_ready = logic_pop_sync_readArbitation_ready;
    if(when_Stream_l369) begin
      logic_pop_addressGen_ready = 1'b1;
    end
  end

  assign when_Stream_l369 = (! logic_pop_sync_readArbitation_valid);
  assign logic_pop_sync_readArbitation_valid = logic_pop_addressGen_rValid;
  assign logic_pop_sync_readArbitation_payload = logic_pop_addressGen_rData;
  assign logic_pop_sync_readPort_rsp_data = _zz_logic_ram_port1[39 : 0];
  assign logic_pop_sync_readPort_cmd_valid = logic_pop_addressGen_fire;
  assign logic_pop_sync_readPort_cmd_payload = logic_pop_addressGen_payload;
  assign logic_pop_sync_readArbitation_translated_valid = logic_pop_sync_readArbitation_valid;
  assign logic_pop_sync_readArbitation_ready = logic_pop_sync_readArbitation_translated_ready;
  assign logic_pop_sync_readArbitation_translated_payload_data = logic_pop_sync_readPort_rsp_data;
  assign io_pop_valid = logic_pop_sync_readArbitation_translated_valid;
  assign logic_pop_sync_readArbitation_translated_ready = io_pop_ready;
  assign io_pop_payload_data = logic_pop_sync_readArbitation_translated_payload_data;
  assign logic_pop_sync_readArbitation_fire = (logic_pop_sync_readArbitation_valid && logic_pop_sync_readArbitation_ready);
  assign logic_ptr_popOnIo = logic_pop_sync_popReg;
  assign io_occupancy = logic_ptr_occupancy;
  assign io_availability = (5'h10 - logic_ptr_occupancy);
  always @(posedge clk) begin
    if(!resetn) begin
      logic_ptr_push <= 5'h00;
      logic_ptr_pop <= 5'h00;
      logic_ptr_wentUp <= 1'b0;
      logic_pop_addressGen_rValid <= 1'b0;
      logic_pop_sync_popReg <= 5'h00;
    end else begin
      if(when_Stream_l1205) begin
        logic_ptr_wentUp <= logic_ptr_doPush;
      end
      if(io_flush) begin
        logic_ptr_wentUp <= 1'b0;
      end
      if(logic_ptr_doPush) begin
        logic_ptr_push <= (logic_ptr_push + 5'h01);
      end
      if(logic_ptr_doPop) begin
        logic_ptr_pop <= (logic_ptr_pop + 5'h01);
      end
      if(io_flush) begin
        logic_ptr_push <= 5'h00;
        logic_ptr_pop <= 5'h00;
      end
      if(logic_pop_addressGen_ready) begin
        logic_pop_addressGen_rValid <= logic_pop_addressGen_valid;
      end
      if(io_flush) begin
        logic_pop_addressGen_rValid <= 1'b0;
      end
      if(logic_pop_sync_readArbitation_fire) begin
        logic_pop_sync_popReg <= logic_ptr_pop;
      end
      if(io_flush) begin
        logic_pop_sync_popReg <= 5'h00;
      end
    end
  end

  always @(posedge clk) begin
    if(logic_pop_addressGen_ready) begin
      logic_pop_addressGen_rData <= logic_pop_addressGen_payload;
    end
  end


endmodule

//StreamFifo_2 replaced by StreamFifo_1

module StreamFifo_1 (
  input  wire          io_push_valid,
  output wire          io_push_ready,
  input  wire [39:0]   io_push_payload_data,
  output wire          io_pop_valid,
  input  wire          io_pop_ready,
  output wire [39:0]   io_pop_payload_data,
  input  wire          io_flush,
  output wire [11:0]   io_occupancy,
  output wire [11:0]   io_availability,
  input  wire          clk,
  input  wire          resetn
);

  reg        [39:0]   _zz_logic_ram_port1;
  wire       [11:0]   _zz_logic_ptr_notPow2_counter;
  wire       [11:0]   _zz_logic_ptr_notPow2_counter_1;
  wire       [0:0]    _zz_logic_ptr_notPow2_counter_2;
  wire       [11:0]   _zz_logic_ptr_notPow2_counter_3;
  wire       [0:0]    _zz_logic_ptr_notPow2_counter_4;
  reg                 _zz_1;
  wire                logic_ptr_doPush;
  wire                logic_ptr_doPop;
  wire                logic_ptr_full;
  wire                logic_ptr_empty;
  reg        [11:0]   logic_ptr_push;
  reg        [11:0]   logic_ptr_pop;
  wire       [11:0]   logic_ptr_occupancy;
  wire       [11:0]   logic_ptr_popOnIo;
  wire                when_Stream_l1205;
  reg                 logic_ptr_wentUp;
  wire                when_Stream_l1240;
  wire                when_Stream_l1244;
  reg        [11:0]   logic_ptr_notPow2_counter;
  wire                io_push_fire;
  wire                io_pop_fire;
  wire                logic_push_onRam_write_valid;
  wire       [11:0]   logic_push_onRam_write_payload_address;
  wire       [39:0]   logic_push_onRam_write_payload_data_data;
  wire                logic_pop_addressGen_valid;
  reg                 logic_pop_addressGen_ready;
  wire       [11:0]   logic_pop_addressGen_payload;
  wire                logic_pop_addressGen_fire;
  wire                logic_pop_sync_readArbitation_valid;
  wire                logic_pop_sync_readArbitation_ready;
  wire       [11:0]   logic_pop_sync_readArbitation_payload;
  reg                 logic_pop_addressGen_rValid;
  reg        [11:0]   logic_pop_addressGen_rData;
  wire                when_Stream_l369;
  wire                logic_pop_sync_readPort_cmd_valid;
  wire       [11:0]   logic_pop_sync_readPort_cmd_payload;
  wire       [39:0]   logic_pop_sync_readPort_rsp_data;
  wire                logic_pop_sync_readArbitation_translated_valid;
  wire                logic_pop_sync_readArbitation_translated_ready;
  wire       [39:0]   logic_pop_sync_readArbitation_translated_payload_data;
  wire                logic_pop_sync_readArbitation_fire;
  reg        [11:0]   logic_pop_sync_popReg;
  reg [39:0] logic_ram [0:3999];

  assign _zz_logic_ptr_notPow2_counter = (logic_ptr_notPow2_counter + _zz_logic_ptr_notPow2_counter_1);
  assign _zz_logic_ptr_notPow2_counter_2 = io_push_fire;
  assign _zz_logic_ptr_notPow2_counter_1 = {11'd0, _zz_logic_ptr_notPow2_counter_2};
  assign _zz_logic_ptr_notPow2_counter_4 = io_pop_fire;
  assign _zz_logic_ptr_notPow2_counter_3 = {11'd0, _zz_logic_ptr_notPow2_counter_4};
  always @(posedge clk) begin
    if(_zz_1) begin
      logic_ram[logic_push_onRam_write_payload_address] <= logic_push_onRam_write_payload_data_data;
    end
  end

  always @(posedge clk) begin
    if(logic_pop_sync_readPort_cmd_valid) begin
      _zz_logic_ram_port1 <= logic_ram[logic_pop_sync_readPort_cmd_payload];
    end
  end

  always @(*) begin
    _zz_1 = 1'b0;
    if(logic_push_onRam_write_valid) begin
      _zz_1 = 1'b1;
    end
  end

  assign when_Stream_l1205 = (logic_ptr_doPush != logic_ptr_doPop);
  assign logic_ptr_full = ((logic_ptr_push == logic_ptr_popOnIo) && logic_ptr_wentUp);
  assign logic_ptr_empty = ((logic_ptr_push == logic_ptr_pop) && (! logic_ptr_wentUp));
  assign when_Stream_l1240 = (logic_ptr_push == 12'hf9f);
  assign when_Stream_l1244 = (logic_ptr_pop == 12'hf9f);
  assign io_push_fire = (io_push_valid && io_push_ready);
  assign io_pop_fire = (io_pop_valid && io_pop_ready);
  assign logic_ptr_occupancy = logic_ptr_notPow2_counter;
  assign io_push_ready = (! logic_ptr_full);
  assign logic_ptr_doPush = io_push_fire;
  assign logic_push_onRam_write_valid = io_push_fire;
  assign logic_push_onRam_write_payload_address = logic_ptr_push;
  assign logic_push_onRam_write_payload_data_data = io_push_payload_data;
  assign logic_pop_addressGen_valid = (! logic_ptr_empty);
  assign logic_pop_addressGen_payload = logic_ptr_pop;
  assign logic_pop_addressGen_fire = (logic_pop_addressGen_valid && logic_pop_addressGen_ready);
  assign logic_ptr_doPop = logic_pop_addressGen_fire;
  always @(*) begin
    logic_pop_addressGen_ready = logic_pop_sync_readArbitation_ready;
    if(when_Stream_l369) begin
      logic_pop_addressGen_ready = 1'b1;
    end
  end

  assign when_Stream_l369 = (! logic_pop_sync_readArbitation_valid);
  assign logic_pop_sync_readArbitation_valid = logic_pop_addressGen_rValid;
  assign logic_pop_sync_readArbitation_payload = logic_pop_addressGen_rData;
  assign logic_pop_sync_readPort_rsp_data = _zz_logic_ram_port1[39 : 0];
  assign logic_pop_sync_readPort_cmd_valid = logic_pop_addressGen_fire;
  assign logic_pop_sync_readPort_cmd_payload = logic_pop_addressGen_payload;
  assign logic_pop_sync_readArbitation_translated_valid = logic_pop_sync_readArbitation_valid;
  assign logic_pop_sync_readArbitation_ready = logic_pop_sync_readArbitation_translated_ready;
  assign logic_pop_sync_readArbitation_translated_payload_data = logic_pop_sync_readPort_rsp_data;
  assign io_pop_valid = logic_pop_sync_readArbitation_translated_valid;
  assign logic_pop_sync_readArbitation_translated_ready = io_pop_ready;
  assign io_pop_payload_data = logic_pop_sync_readArbitation_translated_payload_data;
  assign logic_pop_sync_readArbitation_fire = (logic_pop_sync_readArbitation_valid && logic_pop_sync_readArbitation_ready);
  assign logic_ptr_popOnIo = logic_pop_sync_popReg;
  assign io_occupancy = logic_ptr_occupancy;
  assign io_availability = (12'hfa0 - logic_ptr_occupancy);
  always @(posedge clk) begin
    if(!resetn) begin
      logic_ptr_push <= 12'h000;
      logic_ptr_pop <= 12'h000;
      logic_ptr_wentUp <= 1'b0;
      logic_ptr_notPow2_counter <= 12'h000;
      logic_pop_addressGen_rValid <= 1'b0;
      logic_pop_sync_popReg <= 12'h000;
    end else begin
      if(when_Stream_l1205) begin
        logic_ptr_wentUp <= logic_ptr_doPush;
      end
      if(io_flush) begin
        logic_ptr_wentUp <= 1'b0;
      end
      if(logic_ptr_doPush) begin
        logic_ptr_push <= (logic_ptr_push + 12'h001);
        if(when_Stream_l1240) begin
          logic_ptr_push <= 12'h000;
        end
      end
      if(logic_ptr_doPop) begin
        logic_ptr_pop <= (logic_ptr_pop + 12'h001);
        if(when_Stream_l1244) begin
          logic_ptr_pop <= 12'h000;
        end
      end
      if(io_flush) begin
        logic_ptr_push <= 12'h000;
        logic_ptr_pop <= 12'h000;
      end
      logic_ptr_notPow2_counter <= (_zz_logic_ptr_notPow2_counter - _zz_logic_ptr_notPow2_counter_3);
      if(io_flush) begin
        logic_ptr_notPow2_counter <= 12'h000;
      end
      if(logic_pop_addressGen_ready) begin
        logic_pop_addressGen_rValid <= logic_pop_addressGen_valid;
      end
      if(io_flush) begin
        logic_pop_addressGen_rValid <= 1'b0;
      end
      if(logic_pop_sync_readArbitation_fire) begin
        logic_pop_sync_popReg <= logic_ptr_pop;
      end
      if(io_flush) begin
        logic_pop_sync_popReg <= 12'h000;
      end
    end
  end

  always @(posedge clk) begin
    if(logic_pop_addressGen_ready) begin
      logic_pop_addressGen_rData <= logic_pop_addressGen_payload;
    end
  end


endmodule

module StreamFifo (
  input  wire          io_push_valid,
  output wire          io_push_ready,
  input  wire [319:0]  io_push_payload_data,
  output wire          io_pop_valid,
  input  wire          io_pop_ready,
  output wire [319:0]  io_pop_payload_data,
  input  wire          io_flush,
  output wire [11:0]   io_occupancy,
  output wire [11:0]   io_availability,
  input  wire          clk,
  input  wire          resetn
);

  reg        [319:0]  _zz_logic_ram_port1;
  wire       [11:0]   _zz_logic_ptr_notPow2_counter;
  wire       [11:0]   _zz_logic_ptr_notPow2_counter_1;
  wire       [0:0]    _zz_logic_ptr_notPow2_counter_2;
  wire       [11:0]   _zz_logic_ptr_notPow2_counter_3;
  wire       [0:0]    _zz_logic_ptr_notPow2_counter_4;
  reg                 _zz_1;
  wire                logic_ptr_doPush;
  wire                logic_ptr_doPop;
  wire                logic_ptr_full;
  wire                logic_ptr_empty;
  reg        [11:0]   logic_ptr_push;
  reg        [11:0]   logic_ptr_pop;
  wire       [11:0]   logic_ptr_occupancy;
  wire       [11:0]   logic_ptr_popOnIo;
  wire                when_Stream_l1205;
  reg                 logic_ptr_wentUp;
  wire                when_Stream_l1240;
  wire                when_Stream_l1244;
  reg        [11:0]   logic_ptr_notPow2_counter;
  wire                io_push_fire;
  wire                io_pop_fire;
  wire                logic_push_onRam_write_valid;
  wire       [11:0]   logic_push_onRam_write_payload_address;
  wire       [319:0]  logic_push_onRam_write_payload_data_data;
  wire                logic_pop_addressGen_valid;
  reg                 logic_pop_addressGen_ready;
  wire       [11:0]   logic_pop_addressGen_payload;
  wire                logic_pop_addressGen_fire;
  wire                logic_pop_sync_readArbitation_valid;
  wire                logic_pop_sync_readArbitation_ready;
  wire       [11:0]   logic_pop_sync_readArbitation_payload;
  reg                 logic_pop_addressGen_rValid;
  reg        [11:0]   logic_pop_addressGen_rData;
  wire                when_Stream_l369;
  wire                logic_pop_sync_readPort_cmd_valid;
  wire       [11:0]   logic_pop_sync_readPort_cmd_payload;
  wire       [319:0]  logic_pop_sync_readPort_rsp_data;
  wire                logic_pop_sync_readArbitation_translated_valid;
  wire                logic_pop_sync_readArbitation_translated_ready;
  wire       [319:0]  logic_pop_sync_readArbitation_translated_payload_data;
  wire                logic_pop_sync_readArbitation_fire;
  reg        [11:0]   logic_pop_sync_popReg;
  reg [319:0] logic_ram [0:3999];

  assign _zz_logic_ptr_notPow2_counter = (logic_ptr_notPow2_counter + _zz_logic_ptr_notPow2_counter_1);
  assign _zz_logic_ptr_notPow2_counter_2 = io_push_fire;
  assign _zz_logic_ptr_notPow2_counter_1 = {11'd0, _zz_logic_ptr_notPow2_counter_2};
  assign _zz_logic_ptr_notPow2_counter_4 = io_pop_fire;
  assign _zz_logic_ptr_notPow2_counter_3 = {11'd0, _zz_logic_ptr_notPow2_counter_4};
  always @(posedge clk) begin
    if(_zz_1) begin
      logic_ram[logic_push_onRam_write_payload_address] <= logic_push_onRam_write_payload_data_data;
    end
  end

  always @(posedge clk) begin
    if(logic_pop_sync_readPort_cmd_valid) begin
      _zz_logic_ram_port1 <= logic_ram[logic_pop_sync_readPort_cmd_payload];
    end
  end

  always @(*) begin
    _zz_1 = 1'b0;
    if(logic_push_onRam_write_valid) begin
      _zz_1 = 1'b1;
    end
  end

  assign when_Stream_l1205 = (logic_ptr_doPush != logic_ptr_doPop);
  assign logic_ptr_full = ((logic_ptr_push == logic_ptr_popOnIo) && logic_ptr_wentUp);
  assign logic_ptr_empty = ((logic_ptr_push == logic_ptr_pop) && (! logic_ptr_wentUp));
  assign when_Stream_l1240 = (logic_ptr_push == 12'hf9f);
  assign when_Stream_l1244 = (logic_ptr_pop == 12'hf9f);
  assign io_push_fire = (io_push_valid && io_push_ready);
  assign io_pop_fire = (io_pop_valid && io_pop_ready);
  assign logic_ptr_occupancy = logic_ptr_notPow2_counter;
  assign io_push_ready = (! logic_ptr_full);
  assign logic_ptr_doPush = io_push_fire;
  assign logic_push_onRam_write_valid = io_push_fire;
  assign logic_push_onRam_write_payload_address = logic_ptr_push;
  assign logic_push_onRam_write_payload_data_data = io_push_payload_data;
  assign logic_pop_addressGen_valid = (! logic_ptr_empty);
  assign logic_pop_addressGen_payload = logic_ptr_pop;
  assign logic_pop_addressGen_fire = (logic_pop_addressGen_valid && logic_pop_addressGen_ready);
  assign logic_ptr_doPop = logic_pop_addressGen_fire;
  always @(*) begin
    logic_pop_addressGen_ready = logic_pop_sync_readArbitation_ready;
    if(when_Stream_l369) begin
      logic_pop_addressGen_ready = 1'b1;
    end
  end

  assign when_Stream_l369 = (! logic_pop_sync_readArbitation_valid);
  assign logic_pop_sync_readArbitation_valid = logic_pop_addressGen_rValid;
  assign logic_pop_sync_readArbitation_payload = logic_pop_addressGen_rData;
  assign logic_pop_sync_readPort_rsp_data = _zz_logic_ram_port1[319 : 0];
  assign logic_pop_sync_readPort_cmd_valid = logic_pop_addressGen_fire;
  assign logic_pop_sync_readPort_cmd_payload = logic_pop_addressGen_payload;
  assign logic_pop_sync_readArbitation_translated_valid = logic_pop_sync_readArbitation_valid;
  assign logic_pop_sync_readArbitation_ready = logic_pop_sync_readArbitation_translated_ready;
  assign logic_pop_sync_readArbitation_translated_payload_data = logic_pop_sync_readPort_rsp_data;
  assign io_pop_valid = logic_pop_sync_readArbitation_translated_valid;
  assign logic_pop_sync_readArbitation_translated_ready = io_pop_ready;
  assign io_pop_payload_data = logic_pop_sync_readArbitation_translated_payload_data;
  assign logic_pop_sync_readArbitation_fire = (logic_pop_sync_readArbitation_valid && logic_pop_sync_readArbitation_ready);
  assign logic_ptr_popOnIo = logic_pop_sync_popReg;
  assign io_occupancy = logic_ptr_occupancy;
  assign io_availability = (12'hfa0 - logic_ptr_occupancy);
  always @(posedge clk) begin
    if(!resetn) begin
      logic_ptr_push <= 12'h000;
      logic_ptr_pop <= 12'h000;
      logic_ptr_wentUp <= 1'b0;
      logic_ptr_notPow2_counter <= 12'h000;
      logic_pop_addressGen_rValid <= 1'b0;
      logic_pop_sync_popReg <= 12'h000;
    end else begin
      if(when_Stream_l1205) begin
        logic_ptr_wentUp <= logic_ptr_doPush;
      end
      if(io_flush) begin
        logic_ptr_wentUp <= 1'b0;
      end
      if(logic_ptr_doPush) begin
        logic_ptr_push <= (logic_ptr_push + 12'h001);
        if(when_Stream_l1240) begin
          logic_ptr_push <= 12'h000;
        end
      end
      if(logic_ptr_doPop) begin
        logic_ptr_pop <= (logic_ptr_pop + 12'h001);
        if(when_Stream_l1244) begin
          logic_ptr_pop <= 12'h000;
        end
      end
      if(io_flush) begin
        logic_ptr_push <= 12'h000;
        logic_ptr_pop <= 12'h000;
      end
      logic_ptr_notPow2_counter <= (_zz_logic_ptr_notPow2_counter - _zz_logic_ptr_notPow2_counter_3);
      if(io_flush) begin
        logic_ptr_notPow2_counter <= 12'h000;
      end
      if(logic_pop_addressGen_ready) begin
        logic_pop_addressGen_rValid <= logic_pop_addressGen_valid;
      end
      if(io_flush) begin
        logic_pop_addressGen_rValid <= 1'b0;
      end
      if(logic_pop_sync_readArbitation_fire) begin
        logic_pop_sync_popReg <= logic_ptr_pop;
      end
      if(io_flush) begin
        logic_pop_sync_popReg <= 12'h000;
      end
    end
  end

  always @(posedge clk) begin
    if(logic_pop_addressGen_ready) begin
      logic_pop_addressGen_rData <= logic_pop_addressGen_payload;
    end
  end


endmodule

module tryo_flow_mux_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          xlnq_stream_TVALID,
  output wire          xlnq_stream_TREADY,
  input  wire [63:0]   xlnq_stream_TDATA,
  input  wire          xlns_stream_TVALID,
  output wire          xlns_stream_TREADY,
  input  wire [7:0]    xlns_stream_TDATA,
  input  wire          aq_stream_TVALID,
  output wire          aq_stream_TREADY,
  input  wire [63:0]   aq_stream_TDATA,
  input  wire          as_stream_TVALID,
  output wire          as_stream_TREADY,
  input  wire [7:0]    as_stream_TDATA,
  input  wire          xmq_stream_TVALID,
  output wire          xmq_stream_TREADY,
  input  wire [63:0]   xmq_stream_TDATA,
  input  wire          xms_stream_TVALID,
  output wire          xms_stream_TREADY,
  input  wire [7:0]    xms_stream_TDATA,
  output wire          q_stream_TVALID,
  input  wire          q_stream_TREADY,
  output wire [511:0]  q_stream_TDATA,
  output wire          s_stream_TVALID,
  input  wire          s_stream_TREADY,
  output wire [39:0]   s_stream_TDATA
);

  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_xlnq_stream_TREADY;
  wire                black_box_xlns_stream_TREADY;
  wire                black_box_aq_stream_TREADY;
  wire                black_box_as_stream_TREADY;
  wire                black_box_xmq_stream_TREADY;
  wire                black_box_xms_stream_TREADY;
  wire       [511:0]  black_box_q_stream_TDATA;
  wire                black_box_q_stream_TVALID;
  wire       [39:0]   black_box_s_stream_TDATA;
  wire                black_box_s_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;

  MUX black_box (
    .ap_clk             (clk                               ), //i
    .ap_rst_n           (resetn                            ), //i
    .l_begin            (manager_12_signals_O_L_BEGIN[31:0]), //i
    .l_close            (manager_12_signals_O_L_CLOSE[31:0]), //i
    .ap_start           (manager_12_ap_ctrl_ap_start       ), //i
    .ap_continue        (manager_12_ap_ctrl_ap_continue    ), //i
    .ap_idle            (black_box_ap_idle                 ), //o
    .ap_ready           (black_box_ap_ready                ), //o
    .ap_done            (black_box_ap_done                 ), //o
    .xlnq_stream_TDATA  (xlnq_stream_TDATA[63:0]           ), //i
    .xlnq_stream_TVALID (xlnq_stream_TVALID                ), //i
    .xlnq_stream_TREADY (black_box_xlnq_stream_TREADY      ), //o
    .xlns_stream_TDATA  (xlns_stream_TDATA[7:0]            ), //i
    .xlns_stream_TVALID (xlns_stream_TVALID                ), //i
    .xlns_stream_TREADY (black_box_xlns_stream_TREADY      ), //o
    .aq_stream_TDATA    (aq_stream_TDATA[63:0]             ), //i
    .aq_stream_TVALID   (aq_stream_TVALID                  ), //i
    .aq_stream_TREADY   (black_box_aq_stream_TREADY        ), //o
    .as_stream_TDATA    (as_stream_TDATA[7:0]              ), //i
    .as_stream_TVALID   (as_stream_TVALID                  ), //i
    .as_stream_TREADY   (black_box_as_stream_TREADY        ), //o
    .xmq_stream_TDATA   (xmq_stream_TDATA[63:0]            ), //i
    .xmq_stream_TVALID  (xmq_stream_TVALID                 ), //i
    .xmq_stream_TREADY  (black_box_xmq_stream_TREADY       ), //o
    .xms_stream_TDATA   (xms_stream_TDATA[7:0]             ), //i
    .xms_stream_TVALID  (xms_stream_TVALID                 ), //i
    .xms_stream_TREADY  (black_box_xms_stream_TREADY       ), //o
    .q_stream_TDATA     (black_box_q_stream_TDATA[511:0]   ), //o
    .q_stream_TVALID    (black_box_q_stream_TVALID         ), //o
    .q_stream_TREADY    (q_stream_TREADY                   ), //i
    .s_stream_TDATA     (black_box_s_stream_TDATA[39:0]    ), //o
    .s_stream_TVALID    (black_box_s_stream_TVALID         ), //o
    .s_stream_TREADY    (s_stream_TREADY                   )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign xlnq_stream_TREADY = black_box_xlnq_stream_TREADY;
  assign xlns_stream_TREADY = black_box_xlns_stream_TREADY;
  assign aq_stream_TREADY = black_box_aq_stream_TREADY;
  assign as_stream_TREADY = black_box_as_stream_TREADY;
  assign xmq_stream_TREADY = black_box_xmq_stream_TREADY;
  assign xms_stream_TREADY = black_box_xms_stream_TREADY;
  assign q_stream_TDATA = black_box_q_stream_TDATA;
  assign q_stream_TVALID = black_box_q_stream_TVALID;
  assign s_stream_TDATA = black_box_s_stream_TDATA;
  assign s_stream_TVALID = black_box_s_stream_TVALID;

endmodule

module tryo_norm_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          x_stream_TVALID,
  output wire          x_stream_TREADY,
  input  wire [199:0]  x_stream_TDATA,
  output wire          xlnq_stream_TVALID,
  input  wire          xlnq_stream_TREADY,
  output wire [63:0]   xlnq_stream_TDATA,
  output wire          xlns_stream_TVALID,
  input  wire          xlns_stream_TREADY,
  output wire [7:0]    xlns_stream_TDATA
);

  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_x_stream_TREADY;
  wire       [63:0]   black_box_xlnq_stream_TDATA;
  wire                black_box_xlnq_stream_TVALID;
  wire       [7:0]    black_box_xlns_stream_TDATA;
  wire                black_box_xlns_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;

  RMSNORM_QUANT black_box (
    .ap_clk             (clk                               ), //i
    .ap_rst_n           (resetn                            ), //i
    .l_begin            (manager_12_signals_O_L_BEGIN[31:0]), //i
    .l_close            (manager_12_signals_O_L_CLOSE[31:0]), //i
    .ap_start           (manager_12_ap_ctrl_ap_start       ), //i
    .ap_continue        (manager_12_ap_ctrl_ap_continue    ), //i
    .ap_idle            (black_box_ap_idle                 ), //o
    .ap_ready           (black_box_ap_ready                ), //o
    .ap_done            (black_box_ap_done                 ), //o
    .x_stream_TDATA     (x_stream_TDATA[199:0]             ), //i
    .x_stream_TVALID    (x_stream_TVALID                   ), //i
    .x_stream_TREADY    (black_box_x_stream_TREADY         ), //o
    .xlnq_stream_TDATA  (black_box_xlnq_stream_TDATA[63:0] ), //o
    .xlnq_stream_TVALID (black_box_xlnq_stream_TVALID      ), //o
    .xlnq_stream_TREADY (xlnq_stream_TREADY                ), //i
    .xlns_stream_TDATA  (black_box_xlns_stream_TDATA[7:0]  ), //o
    .xlns_stream_TVALID (black_box_xlns_stream_TVALID      ), //o
    .xlns_stream_TREADY (xlns_stream_TREADY                )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign x_stream_TREADY = black_box_x_stream_TREADY;
  assign xlnq_stream_TDATA = black_box_xlnq_stream_TDATA;
  assign xlnq_stream_TVALID = black_box_xlnq_stream_TVALID;
  assign xlns_stream_TDATA = black_box_xlns_stream_TDATA;
  assign xlns_stream_TVALID = black_box_xlns_stream_TVALID;

endmodule

module tryo_add_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          x_stream_TVALID,
  output wire          x_stream_TREADY,
  input  wire [199:0]  x_stream_TDATA,
  input  wire          res_i_stream_TVALID,
  output wire          res_i_stream_TREADY,
  input  wire [199:0]  res_i_stream_TDATA,
  output wire          res_o_stream_TVALID,
  input  wire          res_o_stream_TREADY,
  output wire [199:0]  res_o_stream_TDATA,
  output wire          y_stream_TVALID,
  input  wire          y_stream_TREADY,
  output wire [199:0]  y_stream_TDATA
);

  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_x_stream_TREADY;
  wire                black_box_res_i_stream_TREADY;
  wire       [199:0]  black_box_res_o_stream_TDATA;
  wire                black_box_res_o_stream_TVALID;
  wire       [199:0]  black_box_y_stream_TDATA;
  wire                black_box_y_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;

  RESIDUAL black_box (
    .ap_clk              (clk                                ), //i
    .ap_rst_n            (resetn                             ), //i
    .l_begin             (manager_12_signals_O_L_BEGIN[31:0] ), //i
    .l_close             (manager_12_signals_O_L_CLOSE[31:0] ), //i
    .ap_start            (manager_12_ap_ctrl_ap_start        ), //i
    .ap_continue         (manager_12_ap_ctrl_ap_continue     ), //i
    .ap_idle             (black_box_ap_idle                  ), //o
    .ap_ready            (black_box_ap_ready                 ), //o
    .ap_done             (black_box_ap_done                  ), //o
    .x_stream_TDATA      (x_stream_TDATA[199:0]              ), //i
    .x_stream_TVALID     (x_stream_TVALID                    ), //i
    .x_stream_TREADY     (black_box_x_stream_TREADY          ), //o
    .res_i_stream_TDATA  (res_i_stream_TDATA[199:0]          ), //i
    .res_i_stream_TVALID (res_i_stream_TVALID                ), //i
    .res_i_stream_TREADY (black_box_res_i_stream_TREADY      ), //o
    .res_o_stream_TDATA  (black_box_res_o_stream_TDATA[199:0]), //o
    .res_o_stream_TVALID (black_box_res_o_stream_TVALID      ), //o
    .res_o_stream_TREADY (res_o_stream_TREADY                ), //i
    .y_stream_TDATA      (black_box_y_stream_TDATA[199:0]    ), //o
    .y_stream_TVALID     (black_box_y_stream_TVALID          ), //o
    .y_stream_TREADY     (y_stream_TREADY                    )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign x_stream_TREADY = black_box_x_stream_TREADY;
  assign res_i_stream_TREADY = black_box_res_i_stream_TREADY;
  assign res_o_stream_TDATA = black_box_res_o_stream_TDATA;
  assign res_o_stream_TVALID = black_box_res_o_stream_TVALID;
  assign y_stream_TDATA = black_box_y_stream_TDATA;
  assign y_stream_TVALID = black_box_y_stream_TVALID;

endmodule

module tryo_gate_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          ug_stream_TVALID,
  output wire          ug_stream_TREADY,
  input  wire [167:0]  ug_stream_TDATA,
  output wire          q_stream_TVALID,
  input  wire          q_stream_TREADY,
  output wire [63:0]   q_stream_TDATA,
  output wire          s_stream_TVALID,
  input  wire          s_stream_TREADY,
  output wire [7:0]    s_stream_TDATA
);

  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_ug_stream_TREADY;
  wire       [63:0]   black_box_q_stream_TDATA;
  wire                black_box_q_stream_TVALID;
  wire       [7:0]    black_box_s_stream_TDATA;
  wire                black_box_s_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;

  SILU_EM_QUANT black_box (
    .ap_clk           (clk                               ), //i
    .ap_rst_n         (resetn                            ), //i
    .l_begin          (manager_12_signals_O_L_BEGIN[31:0]), //i
    .l_close          (manager_12_signals_O_L_CLOSE[31:0]), //i
    .ap_start         (manager_12_ap_ctrl_ap_start       ), //i
    .ap_continue      (manager_12_ap_ctrl_ap_continue    ), //i
    .ap_idle          (black_box_ap_idle                 ), //o
    .ap_ready         (black_box_ap_ready                ), //o
    .ap_done          (black_box_ap_done                 ), //o
    .ug_stream_TDATA  (ug_stream_TDATA[167:0]            ), //i
    .ug_stream_TVALID (ug_stream_TVALID                  ), //i
    .ug_stream_TREADY (black_box_ug_stream_TREADY        ), //o
    .q_stream_TDATA   (black_box_q_stream_TDATA[63:0]    ), //o
    .q_stream_TVALID  (black_box_q_stream_TVALID         ), //o
    .q_stream_TREADY  (q_stream_TREADY                   ), //i
    .s_stream_TDATA   (black_box_s_stream_TDATA[7:0]     ), //o
    .s_stream_TVALID  (black_box_s_stream_TVALID         ), //o
    .s_stream_TREADY  (s_stream_TREADY                   )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign ug_stream_TREADY = black_box_ug_stream_TREADY;
  assign q_stream_TDATA = black_box_q_stream_TDATA;
  assign q_stream_TVALID = black_box_q_stream_TVALID;
  assign s_stream_TDATA = black_box_s_stream_TDATA;
  assign s_stream_TVALID = black_box_s_stream_TVALID;

endmodule

module tryo_mix_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          rq_stream_TVALID,
  output wire          rq_stream_TREADY,
  input  wire [63:0]   rq_stream_TDATA,
  input  wire          rs_stream_TVALID,
  output wire          rs_stream_TREADY,
  input  wire [7:0]    rs_stream_TDATA,
  input  wire          v_stream_TVALID,
  output wire          v_stream_TREADY,
  input  wire [183:0]  v_stream_TDATA,
  input  wire          vq_cache_i_stream_TVALID,
  output wire          vq_cache_i_stream_TREADY,
  input  wire [63:0]   vq_cache_i_stream_TDATA,
  input  wire          vs_cache_i_stream_TVALID,
  output wire          vs_cache_i_stream_TREADY,
  input  wire [7:0]    vs_cache_i_stream_TDATA,
  output wire          vq_cache_o_stream_TVALID,
  input  wire          vq_cache_o_stream_TREADY,
  output wire [63:0]   vq_cache_o_stream_TDATA,
  output wire          vs_cache_o_stream_TVALID,
  input  wire          vs_cache_o_stream_TREADY,
  output wire [7:0]    vs_cache_o_stream_TDATA,
  output wire          aq_stream_TVALID,
  input  wire          aq_stream_TREADY,
  output wire [63:0]   aq_stream_TDATA,
  output wire          as_stream_TVALID,
  input  wire          as_stream_TREADY,
  output wire [7:0]    as_stream_TDATA
);

  wire       [31:0]   black_box_chunk;
  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_rq_stream_TREADY;
  wire                black_box_rs_stream_TREADY;
  wire                black_box_v_stream_TREADY;
  wire                black_box_vq_cache_i_stream_TREADY;
  wire                black_box_vs_cache_i_stream_TREADY;
  wire       [63:0]   black_box_vq_cache_o_stream_TDATA;
  wire                black_box_vq_cache_o_stream_TVALID;
  wire       [7:0]    black_box_vs_cache_o_stream_TDATA;
  wire                black_box_vs_cache_o_stream_TVALID;
  wire       [63:0]   black_box_aq_stream_TDATA;
  wire                black_box_aq_stream_TVALID;
  wire       [7:0]    black_box_as_stream_TDATA;
  wire                black_box_as_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;
  wire       [28:0]   _zz_chunk;

  assign _zz_chunk = (manager_12_signals_O_POS_ID >>> 2'd3);
  RV_GEMM black_box (
    .ap_clk                   (clk                                    ), //i
    .ap_rst_n                 (resetn                                 ), //i
    .l_begin                  (manager_12_signals_O_L_BEGIN[31:0]     ), //i
    .l_close                  (manager_12_signals_O_L_CLOSE[31:0]     ), //i
    .chunk                    (black_box_chunk[31:0]                  ), //i
    .ap_start                 (manager_12_ap_ctrl_ap_start            ), //i
    .ap_continue              (manager_12_ap_ctrl_ap_continue         ), //i
    .ap_idle                  (black_box_ap_idle                      ), //o
    .ap_ready                 (black_box_ap_ready                     ), //o
    .ap_done                  (black_box_ap_done                      ), //o
    .rq_stream_TDATA          (rq_stream_TDATA[63:0]                  ), //i
    .rq_stream_TVALID         (rq_stream_TVALID                       ), //i
    .rq_stream_TREADY         (black_box_rq_stream_TREADY             ), //o
    .rs_stream_TDATA          (rs_stream_TDATA[7:0]                   ), //i
    .rs_stream_TVALID         (rs_stream_TVALID                       ), //i
    .rs_stream_TREADY         (black_box_rs_stream_TREADY             ), //o
    .v_stream_TDATA           (v_stream_TDATA[183:0]                  ), //i
    .v_stream_TVALID          (v_stream_TVALID                        ), //i
    .v_stream_TREADY          (black_box_v_stream_TREADY              ), //o
    .vq_cache_i_stream_TDATA  (vq_cache_i_stream_TDATA[63:0]          ), //i
    .vq_cache_i_stream_TVALID (vq_cache_i_stream_TVALID               ), //i
    .vq_cache_i_stream_TREADY (black_box_vq_cache_i_stream_TREADY     ), //o
    .vs_cache_i_stream_TDATA  (vs_cache_i_stream_TDATA[7:0]           ), //i
    .vs_cache_i_stream_TVALID (vs_cache_i_stream_TVALID               ), //i
    .vs_cache_i_stream_TREADY (black_box_vs_cache_i_stream_TREADY     ), //o
    .vq_cache_o_stream_TDATA  (black_box_vq_cache_o_stream_TDATA[63:0]), //o
    .vq_cache_o_stream_TVALID (black_box_vq_cache_o_stream_TVALID     ), //o
    .vq_cache_o_stream_TREADY (vq_cache_o_stream_TREADY               ), //i
    .vs_cache_o_stream_TDATA  (black_box_vs_cache_o_stream_TDATA[7:0] ), //o
    .vs_cache_o_stream_TVALID (black_box_vs_cache_o_stream_TVALID     ), //o
    .vs_cache_o_stream_TREADY (vs_cache_o_stream_TREADY               ), //i
    .aq_stream_TDATA          (black_box_aq_stream_TDATA[63:0]        ), //o
    .aq_stream_TVALID         (black_box_aq_stream_TVALID             ), //o
    .aq_stream_TREADY         (aq_stream_TREADY                       ), //i
    .as_stream_TDATA          (black_box_as_stream_TDATA[7:0]         ), //o
    .as_stream_TVALID         (black_box_as_stream_TVALID             ), //o
    .as_stream_TREADY         (as_stream_TREADY                       )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign black_box_chunk = {3'd0, _zz_chunk};
  assign rq_stream_TREADY = black_box_rq_stream_TREADY;
  assign rs_stream_TREADY = black_box_rs_stream_TREADY;
  assign v_stream_TREADY = black_box_v_stream_TREADY;
  assign vq_cache_i_stream_TREADY = black_box_vq_cache_i_stream_TREADY;
  assign vs_cache_i_stream_TREADY = black_box_vs_cache_i_stream_TREADY;
  assign vq_cache_o_stream_TDATA = black_box_vq_cache_o_stream_TDATA;
  assign vq_cache_o_stream_TVALID = black_box_vq_cache_o_stream_TVALID;
  assign vs_cache_o_stream_TDATA = black_box_vs_cache_o_stream_TDATA;
  assign vs_cache_o_stream_TVALID = black_box_vs_cache_o_stream_TVALID;
  assign aq_stream_TDATA = black_box_aq_stream_TDATA;
  assign aq_stream_TVALID = black_box_aq_stream_TVALID;
  assign as_stream_TDATA = black_box_as_stream_TDATA;
  assign as_stream_TVALID = black_box_as_stream_TVALID;

endmodule

module tryo_prob_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          r_stream_TVALID,
  output wire          r_stream_TREADY,
  input  wire [247:0]  r_stream_TDATA,
  output wire          rq_stream_TVALID,
  input  wire          rq_stream_TREADY,
  output wire [63:0]   rq_stream_TDATA,
  output wire          rs_stream_TVALID,
  input  wire          rs_stream_TREADY,
  output wire [7:0]    rs_stream_TDATA
);

  wire       [31:0]   black_box_chunk;
  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_r_stream_TREADY;
  wire       [63:0]   black_box_rq_stream_TDATA;
  wire                black_box_rq_stream_TVALID;
  wire       [7:0]    black_box_rs_stream_TDATA;
  wire                black_box_rs_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;
  wire       [28:0]   _zz_chunk;

  assign _zz_chunk = (manager_12_signals_O_POS_ID >>> 2'd3);
  SOFTMAX_QUANT black_box (
    .ap_clk           (clk                               ), //i
    .ap_rst_n         (resetn                            ), //i
    .l_begin          (manager_12_signals_O_L_BEGIN[31:0]), //i
    .l_close          (manager_12_signals_O_L_CLOSE[31:0]), //i
    .chunk            (black_box_chunk[31:0]             ), //i
    .ap_start         (manager_12_ap_ctrl_ap_start       ), //i
    .ap_continue      (manager_12_ap_ctrl_ap_continue    ), //i
    .ap_idle          (black_box_ap_idle                 ), //o
    .ap_ready         (black_box_ap_ready                ), //o
    .ap_done          (black_box_ap_done                 ), //o
    .r_stream_TDATA   (r_stream_TDATA[247:0]             ), //i
    .r_stream_TVALID  (r_stream_TVALID                   ), //i
    .r_stream_TREADY  (black_box_r_stream_TREADY         ), //o
    .rq_stream_TDATA  (black_box_rq_stream_TDATA[63:0]   ), //o
    .rq_stream_TVALID (black_box_rq_stream_TVALID        ), //o
    .rq_stream_TREADY (rq_stream_TREADY                  ), //i
    .rs_stream_TDATA  (black_box_rs_stream_TDATA[7:0]    ), //o
    .rs_stream_TVALID (black_box_rs_stream_TVALID        ), //o
    .rs_stream_TREADY (rs_stream_TREADY                  )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign black_box_chunk = {3'd0, _zz_chunk};
  assign r_stream_TREADY = black_box_r_stream_TREADY;
  assign rq_stream_TDATA = black_box_rq_stream_TDATA;
  assign rq_stream_TVALID = black_box_rq_stream_TVALID;
  assign rs_stream_TDATA = black_box_rs_stream_TDATA;
  assign rs_stream_TVALID = black_box_rs_stream_TVALID;

endmodule

module tryo_score_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          qk_q_stream_TVALID,
  output wire          qk_q_stream_TREADY,
  input  wire [63:0]   qk_q_stream_TDATA,
  input  wire          qk_s_stream_TVALID,
  output wire          qk_s_stream_TREADY,
  input  wire [7:0]    qk_s_stream_TDATA,
  input  wire          kq_cache_i_stream_TVALID,
  output wire          kq_cache_i_stream_TREADY,
  input  wire [63:0]   kq_cache_i_stream_TDATA,
  input  wire          ks_cache_i_stream_TVALID,
  output wire          ks_cache_i_stream_TREADY,
  input  wire [7:0]    ks_cache_i_stream_TDATA,
  output wire          kq_cache_o_stream_TVALID,
  input  wire          kq_cache_o_stream_TREADY,
  output wire [63:0]   kq_cache_o_stream_TDATA,
  output wire          ks_cache_o_stream_TVALID,
  input  wire          ks_cache_o_stream_TREADY,
  output wire [7:0]    ks_cache_o_stream_TDATA,
  output wire          r_stream_TVALID,
  input  wire          r_stream_TREADY,
  output wire [247:0]  r_stream_TDATA
);

  wire       [31:0]   black_box_chunk;
  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_qk_q_stream_TREADY;
  wire                black_box_qk_s_stream_TREADY;
  wire                black_box_kq_cache_i_stream_TREADY;
  wire                black_box_ks_cache_i_stream_TREADY;
  wire       [63:0]   black_box_kq_cache_o_stream_TDATA;
  wire                black_box_kq_cache_o_stream_TVALID;
  wire       [7:0]    black_box_ks_cache_o_stream_TDATA;
  wire                black_box_ks_cache_o_stream_TVALID;
  wire       [247:0]  black_box_r_stream_TDATA;
  wire                black_box_r_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;
  wire       [28:0]   _zz_chunk;

  assign _zz_chunk = (manager_12_signals_O_POS_ID >>> 2'd3);
  QK_GEMM black_box (
    .ap_clk                   (clk                                    ), //i
    .ap_rst_n                 (resetn                                 ), //i
    .l_begin                  (manager_12_signals_O_L_BEGIN[31:0]     ), //i
    .l_close                  (manager_12_signals_O_L_CLOSE[31:0]     ), //i
    .chunk                    (black_box_chunk[31:0]                  ), //i
    .ap_start                 (manager_12_ap_ctrl_ap_start            ), //i
    .ap_continue              (manager_12_ap_ctrl_ap_continue         ), //i
    .ap_idle                  (black_box_ap_idle                      ), //o
    .ap_ready                 (black_box_ap_ready                     ), //o
    .ap_done                  (black_box_ap_done                      ), //o
    .qk_q_stream_TDATA        (qk_q_stream_TDATA[63:0]                ), //i
    .qk_q_stream_TVALID       (qk_q_stream_TVALID                     ), //i
    .qk_q_stream_TREADY       (black_box_qk_q_stream_TREADY           ), //o
    .qk_s_stream_TDATA        (qk_s_stream_TDATA[7:0]                 ), //i
    .qk_s_stream_TVALID       (qk_s_stream_TVALID                     ), //i
    .qk_s_stream_TREADY       (black_box_qk_s_stream_TREADY           ), //o
    .kq_cache_i_stream_TDATA  (kq_cache_i_stream_TDATA[63:0]          ), //i
    .kq_cache_i_stream_TVALID (kq_cache_i_stream_TVALID               ), //i
    .kq_cache_i_stream_TREADY (black_box_kq_cache_i_stream_TREADY     ), //o
    .ks_cache_i_stream_TDATA  (ks_cache_i_stream_TDATA[7:0]           ), //i
    .ks_cache_i_stream_TVALID (ks_cache_i_stream_TVALID               ), //i
    .ks_cache_i_stream_TREADY (black_box_ks_cache_i_stream_TREADY     ), //o
    .kq_cache_o_stream_TDATA  (black_box_kq_cache_o_stream_TDATA[63:0]), //o
    .kq_cache_o_stream_TVALID (black_box_kq_cache_o_stream_TVALID     ), //o
    .kq_cache_o_stream_TREADY (kq_cache_o_stream_TREADY               ), //i
    .ks_cache_o_stream_TDATA  (black_box_ks_cache_o_stream_TDATA[7:0] ), //o
    .ks_cache_o_stream_TVALID (black_box_ks_cache_o_stream_TVALID     ), //o
    .ks_cache_o_stream_TREADY (ks_cache_o_stream_TREADY               ), //i
    .r_stream_TDATA           (black_box_r_stream_TDATA[247:0]        ), //o
    .r_stream_TVALID          (black_box_r_stream_TVALID              ), //o
    .r_stream_TREADY          (r_stream_TREADY                        )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign black_box_chunk = {3'd0, _zz_chunk};
  assign qk_q_stream_TREADY = black_box_qk_q_stream_TREADY;
  assign qk_s_stream_TREADY = black_box_qk_s_stream_TREADY;
  assign kq_cache_i_stream_TREADY = black_box_kq_cache_i_stream_TREADY;
  assign ks_cache_i_stream_TREADY = black_box_ks_cache_i_stream_TREADY;
  assign kq_cache_o_stream_TDATA = black_box_kq_cache_o_stream_TDATA;
  assign kq_cache_o_stream_TVALID = black_box_kq_cache_o_stream_TVALID;
  assign ks_cache_o_stream_TDATA = black_box_ks_cache_o_stream_TDATA;
  assign ks_cache_o_stream_TVALID = black_box_ks_cache_o_stream_TVALID;
  assign r_stream_TDATA = black_box_r_stream_TDATA;
  assign r_stream_TVALID = black_box_r_stream_TVALID;

endmodule

module tryo_rot_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          qk_stream_TVALID,
  output wire          qk_stream_TREADY,
  input  wire [183:0]  qk_stream_TDATA,
  output wire          rot_q_stream_TVALID,
  input  wire          rot_q_stream_TREADY,
  output wire [63:0]   rot_q_stream_TDATA,
  output wire          rot_s_stream_TVALID,
  input  wire          rot_s_stream_TREADY,
  output wire [7:0]    rot_s_stream_TDATA
);

  wire       [31:0]   black_box_chunk;
  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_qk_stream_TREADY;
  wire       [63:0]   black_box_rot_q_stream_TDATA;
  wire                black_box_rot_q_stream_TVALID;
  wire       [7:0]    black_box_rot_s_stream_TDATA;
  wire                black_box_rot_s_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;
  wire       [28:0]   _zz_chunk;

  assign _zz_chunk = (manager_12_signals_O_POS_ID >>> 2'd3);
  ROPE_QK_QUANT black_box (
    .ap_clk              (clk                               ), //i
    .ap_rst_n            (resetn                            ), //i
    .chunk               (black_box_chunk[31:0]             ), //i
    .l_begin             (manager_12_signals_O_L_BEGIN[31:0]), //i
    .l_close             (manager_12_signals_O_L_CLOSE[31:0]), //i
    .ap_start            (manager_12_ap_ctrl_ap_start       ), //i
    .ap_continue         (manager_12_ap_ctrl_ap_continue    ), //i
    .ap_idle             (black_box_ap_idle                 ), //o
    .ap_ready            (black_box_ap_ready                ), //o
    .ap_done             (black_box_ap_done                 ), //o
    .qk_stream_TDATA     (qk_stream_TDATA[183:0]            ), //i
    .qk_stream_TVALID    (qk_stream_TVALID                  ), //i
    .qk_stream_TREADY    (black_box_qk_stream_TREADY        ), //o
    .rot_q_stream_TDATA  (black_box_rot_q_stream_TDATA[63:0]), //o
    .rot_q_stream_TVALID (black_box_rot_q_stream_TVALID     ), //o
    .rot_q_stream_TREADY (rot_q_stream_TREADY               ), //i
    .rot_s_stream_TDATA  (black_box_rot_s_stream_TDATA[7:0] ), //o
    .rot_s_stream_TVALID (black_box_rot_s_stream_TVALID     ), //o
    .rot_s_stream_TREADY (rot_s_stream_TREADY               )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign black_box_chunk = {3'd0, _zz_chunk};
  assign qk_stream_TREADY = black_box_qk_stream_TREADY;
  assign rot_q_stream_TDATA = black_box_rot_q_stream_TDATA;
  assign rot_q_stream_TVALID = black_box_rot_q_stream_TVALID;
  assign rot_s_stream_TDATA = black_box_rot_s_stream_TDATA;
  assign rot_s_stream_TVALID = black_box_rot_s_stream_TVALID;

endmodule

module tryo_route_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          gemm_stream_TVALID,
  output wire          gemm_stream_TREADY,
  input  wire [239:0]  gemm_stream_TDATA,
  output wire          qk_stream_TVALID,
  input  wire          qk_stream_TREADY,
  output wire [183:0]  qk_stream_TDATA,
  output wire          v_stream_TVALID,
  input  wire          v_stream_TREADY,
  output wire [183:0]  v_stream_TDATA,
  output wire          ug_stream_TVALID,
  input  wire          ug_stream_TREADY,
  output wire [167:0]  ug_stream_TDATA,
  output wire          od_stream_TVALID,
  input  wire          od_stream_TREADY,
  output wire [199:0]  od_stream_TDATA,
  output wire          logit_stream_valid,
  input  wire          logit_stream_ready,
  output wire [255:0]  logit_stream_payload_data
);

  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_gemm_stream_TREADY;
  wire       [183:0]  black_box_qk_stream_TDATA;
  wire                black_box_qk_stream_TVALID;
  wire       [183:0]  black_box_v_stream_TDATA;
  wire                black_box_v_stream_TVALID;
  wire       [167:0]  black_box_ug_stream_TDATA;
  wire                black_box_ug_stream_TVALID;
  wire       [199:0]  black_box_od_stream_TDATA;
  wire                black_box_od_stream_TVALID;
  wire       [255:0]  black_box_cls_index_stream_TDATA;
  wire                black_box_cls_index_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;

  DEMUX black_box (
    .ap_clk                  (clk                                    ), //i
    .ap_rst_n                (resetn                                 ), //i
    .l_begin                 (manager_12_signals_O_L_BEGIN[31:0]     ), //i
    .l_close                 (manager_12_signals_O_L_CLOSE[31:0]     ), //i
    .pos_r                   (manager_12_signals_O_POS_ID[31:0]      ), //i
    .ap_start                (manager_12_ap_ctrl_ap_start            ), //i
    .ap_continue             (manager_12_ap_ctrl_ap_continue         ), //i
    .ap_idle                 (black_box_ap_idle                      ), //o
    .ap_ready                (black_box_ap_ready                     ), //o
    .ap_done                 (black_box_ap_done                      ), //o
    .gemm_stream_TDATA       (gemm_stream_TDATA[239:0]               ), //i
    .gemm_stream_TVALID      (gemm_stream_TVALID                     ), //i
    .gemm_stream_TREADY      (black_box_gemm_stream_TREADY           ), //o
    .qk_stream_TDATA         (black_box_qk_stream_TDATA[183:0]       ), //o
    .qk_stream_TVALID        (black_box_qk_stream_TVALID             ), //o
    .qk_stream_TREADY        (qk_stream_TREADY                       ), //i
    .v_stream_TDATA          (black_box_v_stream_TDATA[183:0]        ), //o
    .v_stream_TVALID         (black_box_v_stream_TVALID              ), //o
    .v_stream_TREADY         (v_stream_TREADY                        ), //i
    .ug_stream_TDATA         (black_box_ug_stream_TDATA[167:0]       ), //o
    .ug_stream_TVALID        (black_box_ug_stream_TVALID             ), //o
    .ug_stream_TREADY        (ug_stream_TREADY                       ), //i
    .od_stream_TDATA         (black_box_od_stream_TDATA[199:0]       ), //o
    .od_stream_TVALID        (black_box_od_stream_TVALID             ), //o
    .od_stream_TREADY        (od_stream_TREADY                       ), //i
    .cls_index_stream_TDATA  (black_box_cls_index_stream_TDATA[255:0]), //o
    .cls_index_stream_TVALID (black_box_cls_index_stream_TVALID      ), //o
    .cls_index_stream_TREADY (logit_stream_ready                     )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign gemm_stream_TREADY = black_box_gemm_stream_TREADY;
  assign qk_stream_TDATA = black_box_qk_stream_TDATA;
  assign qk_stream_TVALID = black_box_qk_stream_TVALID;
  assign v_stream_TDATA = black_box_v_stream_TDATA;
  assign v_stream_TVALID = black_box_v_stream_TVALID;
  assign ug_stream_TDATA = black_box_ug_stream_TDATA;
  assign ug_stream_TVALID = black_box_ug_stream_TVALID;
  assign od_stream_TDATA = black_box_od_stream_TDATA;
  assign od_stream_TVALID = black_box_od_stream_TVALID;
  assign logit_stream_payload_data = black_box_cls_index_stream_TDATA;
  assign logit_stream_valid = black_box_cls_index_stream_TVALID;

endmodule

module tryo_vpu_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  input  wire          i_stream_TVALID,
  output wire          i_stream_TREADY,
  input  wire [511:0]  i_stream_TDATA,
  input  wire          w_stream_TVALID,
  output wire          w_stream_TREADY,
  input  wire [319:0]  w_stream_TDATA,
  input  wire          s_stream_TVALID,
  output wire          s_stream_TREADY,
  input  wire [39:0]   s_stream_TDATA,
  input  wire          s1_stream_TVALID,
  output wire          s1_stream_TREADY,
  input  wire [39:0]   s1_stream_TDATA,
  input  wire          s2_stream_TVALID,
  output wire          s2_stream_TREADY,
  input  wire [39:0]   s2_stream_TDATA,
  output wire          o_stream_TVALID,
  input  wire          o_stream_TREADY,
  output wire [239:0]  o_stream_TDATA
);

  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_i_stream_TREADY;
  wire                black_box_w_stream_TREADY;
  wire                black_box_s_stream_TREADY;
  wire                black_box_s1_stream_TREADY;
  wire                black_box_s2_stream_TREADY;
  wire       [239:0]  black_box_o_stream_TDATA;
  wire                black_box_o_stream_TVALID;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;

  GEMM_PERMUTE black_box (
    .ap_clk           (clk                               ), //i
    .ap_rst_n         (resetn                            ), //i
    .l_begin          (manager_12_signals_O_L_BEGIN[31:0]), //i
    .l_close          (manager_12_signals_O_L_CLOSE[31:0]), //i
    .ap_start         (manager_12_ap_ctrl_ap_start       ), //i
    .ap_continue      (manager_12_ap_ctrl_ap_continue    ), //i
    .ap_idle          (black_box_ap_idle                 ), //o
    .ap_ready         (black_box_ap_ready                ), //o
    .ap_done          (black_box_ap_done                 ), //o
    .i_stream_TDATA   (i_stream_TDATA[511:0]             ), //i
    .i_stream_TVALID  (i_stream_TVALID                   ), //i
    .i_stream_TREADY  (black_box_i_stream_TREADY         ), //o
    .w_stream_TDATA   (w_stream_TDATA[319:0]             ), //i
    .w_stream_TVALID  (w_stream_TVALID                   ), //i
    .w_stream_TREADY  (black_box_w_stream_TREADY         ), //o
    .s_stream_TDATA   (s_stream_TDATA[39:0]              ), //i
    .s_stream_TVALID  (s_stream_TVALID                   ), //i
    .s_stream_TREADY  (black_box_s_stream_TREADY         ), //o
    .s1_stream_TDATA  (s1_stream_TDATA[39:0]             ), //i
    .s1_stream_TVALID (s1_stream_TVALID                  ), //i
    .s1_stream_TREADY (black_box_s1_stream_TREADY        ), //o
    .s2_stream_TDATA  (s2_stream_TDATA[39:0]             ), //i
    .s2_stream_TVALID (s2_stream_TVALID                  ), //i
    .s2_stream_TREADY (black_box_s2_stream_TREADY        ), //o
    .o_stream_TDATA   (black_box_o_stream_TDATA[239:0]   ), //o
    .o_stream_TVALID  (black_box_o_stream_TVALID         ), //o
    .o_stream_TREADY  (o_stream_TREADY                   )  //i
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign i_stream_TREADY = black_box_i_stream_TREADY;
  assign w_stream_TREADY = black_box_w_stream_TREADY;
  assign s_stream_TREADY = black_box_s_stream_TREADY;
  assign s1_stream_TREADY = black_box_s1_stream_TREADY;
  assign s2_stream_TREADY = black_box_s2_stream_TREADY;
  assign o_stream_TDATA = black_box_o_stream_TDATA;
  assign o_stream_TVALID = black_box_o_stream_TVALID;

endmodule

module Controller (
  input  wire          axilite_awvalid,
  output wire          axilite_awready,
  input  wire [15:0]   axilite_awaddr,
  input  wire [2:0]    axilite_awprot,
  input  wire          axilite_wvalid,
  output wire          axilite_wready,
  input  wire [63:0]   axilite_wdata,
  input  wire [7:0]    axilite_wstrb,
  output wire          axilite_bvalid,
  input  wire          axilite_bready,
  output wire [1:0]    axilite_bresp,
  input  wire          axilite_arvalid,
  output reg           axilite_arready,
  input  wire [15:0]   axilite_araddr,
  input  wire [2:0]    axilite_arprot,
  output wire          axilite_rvalid,
  input  wire          axilite_rready,
  output wire [63:0]   axilite_rdata,
  output wire [1:0]    axilite_rresp,
  output wire [31:0]   signals_L_BEGIN,
  output wire [31:0]   signals_L_CLOSE,
  output wire [63:0]   signals_MEMORY_DECODER_X,
  output wire [63:0]   signals_MEMORY_DECODER_Y,
  output wire [63:0]   signals_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_MEMORY_KV_CACHE,
  output wire [31:0]   signals_POS_ID,
  output reg           signals_T,
  input  wire          idle,
  input  wire          clk,
  input  wire          resetn
);

  wire                busCtrl_readErrorFlag;
  wire                busCtrl_writeErrorFlag;
  wire                busCtrl_readHaltRequest;
  wire                busCtrl_writeHaltRequest;
  wire                busCtrl_writeJoinEvent_valid;
  wire                busCtrl_writeJoinEvent_ready;
  wire                busCtrl_writeOccur;
  reg        [1:0]    busCtrl_writeRsp_resp;
  wire                busCtrl_writeJoinEvent_translated_valid;
  wire                busCtrl_writeJoinEvent_translated_ready;
  wire       [1:0]    busCtrl_writeJoinEvent_translated_payload_resp;
  wire                _zz_busCtrl_writeJoinEvent_translated_ready;
  reg                 _zz_busCtrl_writeJoinEvent_translated_ready_1;
  wire                _zz_axilite_bvalid;
  reg                 _zz_axilite_bvalid_1;
  reg        [1:0]    _zz_axilite_bresp;
  wire                when_Stream_l369;
  wire                busCtrl_readDataStage_valid;
  wire                busCtrl_readDataStage_ready;
  wire       [15:0]   busCtrl_readDataStage_payload_addr;
  wire       [2:0]    busCtrl_readDataStage_payload_prot;
  reg                 axilite_ar_rValid;
  reg        [15:0]   axilite_ar_rData_addr;
  reg        [2:0]    axilite_ar_rData_prot;
  wire                when_Stream_l369_1;
  reg        [63:0]   busCtrl_readRsp_data;
  reg        [1:0]    busCtrl_readRsp_resp;
  wire                _zz_axilite_rvalid;
  wire       [15:0]   busCtrl_readAddressMasked;
  wire       [15:0]   busCtrl_writeAddressMasked;
  wire                busCtrl_readOccur;
  reg        [31:0]   signals_L_BEGIN_driver;
  reg        [31:0]   signals_L_CLOSE_driver;
  reg        [31:0]   signals_POS_ID_driver;
  reg        [63:0]   signals_MEMORY_DECODER_X_driver;
  reg        [63:0]   signals_MEMORY_DECODER_Y_driver;
  reg        [63:0]   signals_MEMORY_LOGIT_Y_driver;
  reg        [63:0]   signals_MEMORY_DECODER_W_LO_driver;
  reg        [63:0]   signals_MEMORY_DECODER_W_HI_driver;
  reg        [63:0]   signals_MEMORY_LOGIT_W_LO_driver;
  reg        [63:0]   signals_MEMORY_LOGIT_W_HI_driver;
  reg        [63:0]   signals_MEMORY_KV_CACHE_driver;

  assign busCtrl_readErrorFlag = 1'b0;
  assign busCtrl_writeErrorFlag = 1'b0;
  assign busCtrl_readHaltRequest = 1'b0;
  assign busCtrl_writeHaltRequest = 1'b0;
  assign busCtrl_writeOccur = (busCtrl_writeJoinEvent_valid && busCtrl_writeJoinEvent_ready);
  assign busCtrl_writeJoinEvent_valid = (axilite_awvalid && axilite_wvalid);
  assign axilite_awready = busCtrl_writeOccur;
  assign axilite_wready = busCtrl_writeOccur;
  assign busCtrl_writeJoinEvent_translated_valid = busCtrl_writeJoinEvent_valid;
  assign busCtrl_writeJoinEvent_ready = busCtrl_writeJoinEvent_translated_ready;
  assign busCtrl_writeJoinEvent_translated_payload_resp = busCtrl_writeRsp_resp;
  assign _zz_busCtrl_writeJoinEvent_translated_ready = (! busCtrl_writeHaltRequest);
  assign busCtrl_writeJoinEvent_translated_ready = (_zz_busCtrl_writeJoinEvent_translated_ready_1 && _zz_busCtrl_writeJoinEvent_translated_ready);
  always @(*) begin
    _zz_busCtrl_writeJoinEvent_translated_ready_1 = axilite_bready;
    if(when_Stream_l369) begin
      _zz_busCtrl_writeJoinEvent_translated_ready_1 = 1'b1;
    end
  end

  assign when_Stream_l369 = (! _zz_axilite_bvalid);
  assign _zz_axilite_bvalid = _zz_axilite_bvalid_1;
  assign axilite_bvalid = _zz_axilite_bvalid;
  assign axilite_bresp = _zz_axilite_bresp;
  always @(*) begin
    axilite_arready = busCtrl_readDataStage_ready;
    if(when_Stream_l369_1) begin
      axilite_arready = 1'b1;
    end
  end

  assign when_Stream_l369_1 = (! busCtrl_readDataStage_valid);
  assign busCtrl_readDataStage_valid = axilite_ar_rValid;
  assign busCtrl_readDataStage_payload_addr = axilite_ar_rData_addr;
  assign busCtrl_readDataStage_payload_prot = axilite_ar_rData_prot;
  assign _zz_axilite_rvalid = (! busCtrl_readHaltRequest);
  assign busCtrl_readDataStage_ready = (axilite_rready && _zz_axilite_rvalid);
  assign axilite_rvalid = (busCtrl_readDataStage_valid && _zz_axilite_rvalid);
  assign axilite_rdata = busCtrl_readRsp_data;
  assign axilite_rresp = busCtrl_readRsp_resp;
  always @(*) begin
    if(busCtrl_writeErrorFlag) begin
      busCtrl_writeRsp_resp = 2'b10;
    end else begin
      busCtrl_writeRsp_resp = 2'b00;
    end
  end

  always @(*) begin
    if(busCtrl_readErrorFlag) begin
      busCtrl_readRsp_resp = 2'b10;
    end else begin
      busCtrl_readRsp_resp = 2'b00;
    end
  end

  always @(*) begin
    busCtrl_readRsp_data = 64'h0000000000000000;
    case(busCtrl_readAddressMasked)
      16'h0000 : begin
        busCtrl_readRsp_data[31 : 0] = signals_L_BEGIN_driver;
      end
      16'h0010 : begin
        busCtrl_readRsp_data[31 : 0] = signals_L_CLOSE_driver;
      end
      16'h00a0 : begin
        busCtrl_readRsp_data[31 : 0] = signals_POS_ID_driver;
      end
      16'h0020 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_DECODER_X_driver;
      end
      16'h0030 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_DECODER_Y_driver;
      end
      16'h0040 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_LOGIT_Y_driver;
      end
      16'h0050 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_DECODER_W_LO_driver;
      end
      16'h0060 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_DECODER_W_HI_driver;
      end
      16'h0070 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_LOGIT_W_LO_driver;
      end
      16'h0080 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_LOGIT_W_HI_driver;
      end
      16'h0090 : begin
        busCtrl_readRsp_data[63 : 0] = signals_MEMORY_KV_CACHE_driver;
      end
      16'h00c0 : begin
        busCtrl_readRsp_data[0 : 0] = idle;
      end
      default : begin
      end
    endcase
  end

  assign busCtrl_readAddressMasked = (busCtrl_readDataStage_payload_addr & (~ 16'h0007));
  assign busCtrl_writeAddressMasked = (axilite_awaddr & (~ 16'h0007));
  assign busCtrl_readOccur = (axilite_rvalid && axilite_rready);
  always @(*) begin
    signals_T = 1'b0;
    case(busCtrl_writeAddressMasked)
      16'h00b0 : begin
        if(busCtrl_writeOccur) begin
          signals_T = axilite_wdata[0];
        end
      end
      default : begin
      end
    endcase
  end

  assign signals_L_BEGIN = signals_L_BEGIN_driver;
  assign signals_L_CLOSE = signals_L_CLOSE_driver;
  assign signals_POS_ID = signals_POS_ID_driver;
  assign signals_MEMORY_DECODER_X = signals_MEMORY_DECODER_X_driver;
  assign signals_MEMORY_DECODER_Y = signals_MEMORY_DECODER_Y_driver;
  assign signals_MEMORY_LOGIT_Y = signals_MEMORY_LOGIT_Y_driver;
  assign signals_MEMORY_DECODER_W_LO = signals_MEMORY_DECODER_W_LO_driver;
  assign signals_MEMORY_DECODER_W_HI = signals_MEMORY_DECODER_W_HI_driver;
  assign signals_MEMORY_LOGIT_W_LO = signals_MEMORY_LOGIT_W_LO_driver;
  assign signals_MEMORY_LOGIT_W_HI = signals_MEMORY_LOGIT_W_HI_driver;
  assign signals_MEMORY_KV_CACHE = signals_MEMORY_KV_CACHE_driver;
  always @(posedge clk) begin
    if(!resetn) begin
      _zz_axilite_bvalid_1 <= 1'b0;
      axilite_ar_rValid <= 1'b0;
    end else begin
      if(_zz_busCtrl_writeJoinEvent_translated_ready_1) begin
        _zz_axilite_bvalid_1 <= (busCtrl_writeJoinEvent_translated_valid && _zz_busCtrl_writeJoinEvent_translated_ready);
      end
      if(axilite_arready) begin
        axilite_ar_rValid <= axilite_arvalid;
      end
    end
  end

  always @(posedge clk) begin
    if(_zz_busCtrl_writeJoinEvent_translated_ready_1) begin
      _zz_axilite_bresp <= busCtrl_writeJoinEvent_translated_payload_resp;
    end
    if(axilite_arready) begin
      axilite_ar_rData_addr <= axilite_araddr;
      axilite_ar_rData_prot <= axilite_arprot;
    end
    case(busCtrl_writeAddressMasked)
      16'h0000 : begin
        if(busCtrl_writeOccur) begin
          signals_L_BEGIN_driver <= axilite_wdata[31 : 0];
        end
      end
      16'h0010 : begin
        if(busCtrl_writeOccur) begin
          signals_L_CLOSE_driver <= axilite_wdata[31 : 0];
        end
      end
      16'h00a0 : begin
        if(busCtrl_writeOccur) begin
          signals_POS_ID_driver <= axilite_wdata[31 : 0];
        end
      end
      16'h0020 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_DECODER_X_driver <= axilite_wdata[63 : 0];
        end
      end
      16'h0030 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_DECODER_Y_driver <= axilite_wdata[63 : 0];
        end
      end
      16'h0040 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_LOGIT_Y_driver <= axilite_wdata[63 : 0];
        end
      end
      16'h0050 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_DECODER_W_LO_driver <= axilite_wdata[63 : 0];
        end
      end
      16'h0060 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_DECODER_W_HI_driver <= axilite_wdata[63 : 0];
        end
      end
      16'h0070 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_LOGIT_W_LO_driver <= axilite_wdata[63 : 0];
        end
      end
      16'h0080 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_LOGIT_W_HI_driver <= axilite_wdata[63 : 0];
        end
      end
      16'h0090 : begin
        if(busCtrl_writeOccur) begin
          signals_MEMORY_KV_CACHE_driver <= axilite_wdata[63 : 0];
        end
      end
      default : begin
      end
    endcase
  end


endmodule

module tryo_kvbank_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  output wire          gmem1_awvalid,
  input  wire          gmem1_awready,
  output wire [62:0]   gmem1_awaddr,
  output wire [0:0]    gmem1_awid,
  output wire [3:0]    gmem1_awregion,
  output wire [7:0]    gmem1_awlen,
  output wire [2:0]    gmem1_awsize,
  output wire [1:0]    gmem1_awburst,
  output wire [0:0]    gmem1_awlock,
  output wire [3:0]    gmem1_awcache,
  output wire [3:0]    gmem1_awqos,
  output wire [0:0]    gmem1_awuser,
  output wire [2:0]    gmem1_awprot,
  output wire          gmem1_wvalid,
  input  wire          gmem1_wready,
  output wire [127:0]  gmem1_wdata,
  output wire [15:0]   gmem1_wstrb,
  output wire [0:0]    gmem1_wuser,
  output wire          gmem1_wlast,
  input  wire          gmem1_bvalid,
  output wire          gmem1_bready,
  input  wire [0:0]    gmem1_bid,
  input  wire [1:0]    gmem1_bresp,
  input  wire [0:0]    gmem1_buser,
  output wire          gmem1_arvalid,
  input  wire          gmem1_arready,
  output wire [62:0]   gmem1_araddr,
  output wire [0:0]    gmem1_arid,
  output wire [3:0]    gmem1_arregion,
  output wire [7:0]    gmem1_arlen,
  output wire [2:0]    gmem1_arsize,
  output wire [1:0]    gmem1_arburst,
  output wire [0:0]    gmem1_arlock,
  output wire [3:0]    gmem1_arcache,
  output wire [3:0]    gmem1_arqos,
  output wire [0:0]    gmem1_aruser,
  output wire [2:0]    gmem1_arprot,
  input  wire          gmem1_rvalid,
  output wire          gmem1_rready,
  input  wire [127:0]  gmem1_rdata,
  input  wire [0:0]    gmem1_rid,
  input  wire [1:0]    gmem1_rresp,
  input  wire          gmem1_rlast,
  input  wire [0:0]    gmem1_ruser,
  output wire          kq_cache_i_stream_TVALID,
  input  wire          kq_cache_i_stream_TREADY,
  output wire [63:0]   kq_cache_i_stream_TDATA,
  input  wire          kq_cache_o_stream_TVALID,
  output wire          kq_cache_o_stream_TREADY,
  input  wire [63:0]   kq_cache_o_stream_TDATA,
  output wire          ks_cache_i_stream_TVALID,
  input  wire          ks_cache_i_stream_TREADY,
  output wire [7:0]    ks_cache_i_stream_TDATA,
  input  wire          ks_cache_o_stream_TVALID,
  output wire          ks_cache_o_stream_TREADY,
  input  wire [7:0]    ks_cache_o_stream_TDATA,
  output wire          vq_cache_i_stream_TVALID,
  input  wire          vq_cache_i_stream_TREADY,
  output wire [63:0]   vq_cache_i_stream_TDATA,
  input  wire          vq_cache_o_stream_TVALID,
  output wire          vq_cache_o_stream_TREADY,
  input  wire [63:0]   vq_cache_o_stream_TDATA,
  output wire          vs_cache_i_stream_TVALID,
  input  wire          vs_cache_i_stream_TREADY,
  output wire [7:0]    vs_cache_i_stream_TDATA,
  input  wire          vs_cache_o_stream_TVALID,
  output wire          vs_cache_o_stream_TREADY,
  input  wire [7:0]    vs_cache_o_stream_TDATA,
  output wire          idle
);

  wire       [31:0]   black_box_chunk;
  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_m_axi_gmem1_AWVALID;
  wire       [62:0]   black_box_m_axi_gmem1_AWADDR;
  wire       [0:0]    black_box_m_axi_gmem1_AWID;
  wire       [0:0]    black_box_m_axi_gmem1_AWUSER;
  wire       [3:0]    black_box_m_axi_gmem1_AWREGION;
  wire       [7:0]    black_box_m_axi_gmem1_AWLEN;
  wire       [2:0]    black_box_m_axi_gmem1_AWSIZE;
  wire       [1:0]    black_box_m_axi_gmem1_AWBURST;
  wire       [0:0]    black_box_m_axi_gmem1_AWLOCK;
  wire       [3:0]    black_box_m_axi_gmem1_AWCACHE;
  wire       [3:0]    black_box_m_axi_gmem1_AWQOS;
  wire       [2:0]    black_box_m_axi_gmem1_AWPROT;
  wire                black_box_m_axi_gmem1_WVALID;
  wire       [127:0]  black_box_m_axi_gmem1_WDATA;
  wire       [15:0]   black_box_m_axi_gmem1_WSTRB;
  wire                black_box_m_axi_gmem1_WLAST;
  wire       [0:0]    black_box_m_axi_gmem1_WID;
  wire       [0:0]    black_box_m_axi_gmem1_WUSER;
  wire                black_box_m_axi_gmem1_ARVALID;
  wire       [62:0]   black_box_m_axi_gmem1_ARADDR;
  wire       [0:0]    black_box_m_axi_gmem1_ARID;
  wire       [0:0]    black_box_m_axi_gmem1_ARUSER;
  wire       [3:0]    black_box_m_axi_gmem1_ARREGION;
  wire       [7:0]    black_box_m_axi_gmem1_ARLEN;
  wire       [2:0]    black_box_m_axi_gmem1_ARSIZE;
  wire       [1:0]    black_box_m_axi_gmem1_ARBURST;
  wire       [0:0]    black_box_m_axi_gmem1_ARLOCK;
  wire       [3:0]    black_box_m_axi_gmem1_ARCACHE;
  wire       [3:0]    black_box_m_axi_gmem1_ARQOS;
  wire       [2:0]    black_box_m_axi_gmem1_ARPROT;
  wire                black_box_m_axi_gmem1_RREADY;
  wire                black_box_m_axi_gmem1_BREADY;
  wire       [63:0]   black_box_kq_cache_i_stream_TDATA;
  wire                black_box_kq_cache_i_stream_TVALID;
  wire                black_box_kq_cache_o_stream_TREADY;
  wire       [7:0]    black_box_ks_cache_i_stream_TDATA;
  wire                black_box_ks_cache_i_stream_TVALID;
  wire                black_box_ks_cache_o_stream_TREADY;
  wire       [63:0]   black_box_vq_cache_i_stream_TDATA;
  wire                black_box_vq_cache_i_stream_TVALID;
  wire                black_box_vq_cache_o_stream_TREADY;
  wire       [7:0]    black_box_vs_cache_i_stream_TDATA;
  wire                black_box_vs_cache_i_stream_TVALID;
  wire                black_box_vs_cache_o_stream_TREADY;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;
  wire       [28:0]   _zz_chunk;

  assign _zz_chunk = (manager_12_signals_O_POS_ID >>> 2'd3);
  KV_CACHE black_box (
    .ap_clk                   (clk                                       ), //i
    .ap_rst_n                 (resetn                                    ), //i
    .l_begin                  (manager_12_signals_O_L_BEGIN[31:0]        ), //i
    .l_close                  (manager_12_signals_O_L_CLOSE[31:0]        ), //i
    .chunk                    (black_box_chunk[31:0]                     ), //i
    .memory_k_cache           (manager_12_signals_O_MEMORY_KV_CACHE[63:0]), //i
    .ap_start                 (manager_12_ap_ctrl_ap_start               ), //i
    .ap_continue              (manager_12_ap_ctrl_ap_continue            ), //i
    .ap_idle                  (black_box_ap_idle                         ), //o
    .ap_ready                 (black_box_ap_ready                        ), //o
    .ap_done                  (black_box_ap_done                         ), //o
    .m_axi_gmem1_AWVALID      (black_box_m_axi_gmem1_AWVALID             ), //o
    .m_axi_gmem1_AWREADY      (gmem1_awready                             ), //i
    .m_axi_gmem1_AWADDR       (black_box_m_axi_gmem1_AWADDR[62:0]        ), //o
    .m_axi_gmem1_AWID         (black_box_m_axi_gmem1_AWID                ), //o
    .m_axi_gmem1_AWUSER       (black_box_m_axi_gmem1_AWUSER              ), //o
    .m_axi_gmem1_AWREGION     (black_box_m_axi_gmem1_AWREGION[3:0]       ), //o
    .m_axi_gmem1_AWLEN        (black_box_m_axi_gmem1_AWLEN[7:0]          ), //o
    .m_axi_gmem1_AWSIZE       (black_box_m_axi_gmem1_AWSIZE[2:0]         ), //o
    .m_axi_gmem1_AWBURST      (black_box_m_axi_gmem1_AWBURST[1:0]        ), //o
    .m_axi_gmem1_AWLOCK       (black_box_m_axi_gmem1_AWLOCK              ), //o
    .m_axi_gmem1_AWCACHE      (black_box_m_axi_gmem1_AWCACHE[3:0]        ), //o
    .m_axi_gmem1_AWQOS        (black_box_m_axi_gmem1_AWQOS[3:0]          ), //o
    .m_axi_gmem1_AWPROT       (black_box_m_axi_gmem1_AWPROT[2:0]         ), //o
    .m_axi_gmem1_WVALID       (black_box_m_axi_gmem1_WVALID              ), //o
    .m_axi_gmem1_WREADY       (gmem1_wready                              ), //i
    .m_axi_gmem1_WDATA        (black_box_m_axi_gmem1_WDATA[127:0]        ), //o
    .m_axi_gmem1_WSTRB        (black_box_m_axi_gmem1_WSTRB[15:0]         ), //o
    .m_axi_gmem1_WLAST        (black_box_m_axi_gmem1_WLAST               ), //o
    .m_axi_gmem1_WID          (black_box_m_axi_gmem1_WID                 ), //o
    .m_axi_gmem1_WUSER        (black_box_m_axi_gmem1_WUSER               ), //o
    .m_axi_gmem1_ARVALID      (black_box_m_axi_gmem1_ARVALID             ), //o
    .m_axi_gmem1_ARREADY      (gmem1_arready                             ), //i
    .m_axi_gmem1_ARADDR       (black_box_m_axi_gmem1_ARADDR[62:0]        ), //o
    .m_axi_gmem1_ARID         (black_box_m_axi_gmem1_ARID                ), //o
    .m_axi_gmem1_ARUSER       (black_box_m_axi_gmem1_ARUSER              ), //o
    .m_axi_gmem1_ARREGION     (black_box_m_axi_gmem1_ARREGION[3:0]       ), //o
    .m_axi_gmem1_ARLEN        (black_box_m_axi_gmem1_ARLEN[7:0]          ), //o
    .m_axi_gmem1_ARSIZE       (black_box_m_axi_gmem1_ARSIZE[2:0]         ), //o
    .m_axi_gmem1_ARBURST      (black_box_m_axi_gmem1_ARBURST[1:0]        ), //o
    .m_axi_gmem1_ARLOCK       (black_box_m_axi_gmem1_ARLOCK              ), //o
    .m_axi_gmem1_ARCACHE      (black_box_m_axi_gmem1_ARCACHE[3:0]        ), //o
    .m_axi_gmem1_ARQOS        (black_box_m_axi_gmem1_ARQOS[3:0]          ), //o
    .m_axi_gmem1_ARPROT       (black_box_m_axi_gmem1_ARPROT[2:0]         ), //o
    .m_axi_gmem1_RVALID       (gmem1_rvalid                              ), //i
    .m_axi_gmem1_RREADY       (black_box_m_axi_gmem1_RREADY              ), //o
    .m_axi_gmem1_RDATA        (gmem1_rdata[127:0]                        ), //i
    .m_axi_gmem1_RLAST        (gmem1_rlast                               ), //i
    .m_axi_gmem1_RID          (gmem1_rid                                 ), //i
    .m_axi_gmem1_RUSER        (gmem1_ruser                               ), //i
    .m_axi_gmem1_RRESP        (gmem1_rresp[1:0]                          ), //i
    .m_axi_gmem1_BVALID       (gmem1_bvalid                              ), //i
    .m_axi_gmem1_BREADY       (black_box_m_axi_gmem1_BREADY              ), //o
    .m_axi_gmem1_BID          (gmem1_bid                                 ), //i
    .m_axi_gmem1_BUSER        (gmem1_buser                               ), //i
    .m_axi_gmem1_BRESP        (gmem1_bresp[1:0]                          ), //i
    .kq_cache_i_stream_TDATA  (black_box_kq_cache_i_stream_TDATA[63:0]   ), //o
    .kq_cache_i_stream_TVALID (black_box_kq_cache_i_stream_TVALID        ), //o
    .kq_cache_i_stream_TREADY (kq_cache_i_stream_TREADY                  ), //i
    .kq_cache_o_stream_TDATA  (kq_cache_o_stream_TDATA[63:0]             ), //i
    .kq_cache_o_stream_TVALID (kq_cache_o_stream_TVALID                  ), //i
    .kq_cache_o_stream_TREADY (black_box_kq_cache_o_stream_TREADY        ), //o
    .ks_cache_i_stream_TDATA  (black_box_ks_cache_i_stream_TDATA[7:0]    ), //o
    .ks_cache_i_stream_TVALID (black_box_ks_cache_i_stream_TVALID        ), //o
    .ks_cache_i_stream_TREADY (ks_cache_i_stream_TREADY                  ), //i
    .ks_cache_o_stream_TDATA  (ks_cache_o_stream_TDATA[7:0]              ), //i
    .ks_cache_o_stream_TVALID (ks_cache_o_stream_TVALID                  ), //i
    .ks_cache_o_stream_TREADY (black_box_ks_cache_o_stream_TREADY        ), //o
    .vq_cache_i_stream_TDATA  (black_box_vq_cache_i_stream_TDATA[63:0]   ), //o
    .vq_cache_i_stream_TVALID (black_box_vq_cache_i_stream_TVALID        ), //o
    .vq_cache_i_stream_TREADY (vq_cache_i_stream_TREADY                  ), //i
    .vq_cache_o_stream_TDATA  (vq_cache_o_stream_TDATA[63:0]             ), //i
    .vq_cache_o_stream_TVALID (vq_cache_o_stream_TVALID                  ), //i
    .vq_cache_o_stream_TREADY (black_box_vq_cache_o_stream_TREADY        ), //o
    .vs_cache_i_stream_TDATA  (black_box_vs_cache_i_stream_TDATA[7:0]    ), //o
    .vs_cache_i_stream_TVALID (black_box_vs_cache_i_stream_TVALID        ), //o
    .vs_cache_i_stream_TREADY (vs_cache_i_stream_TREADY                  ), //i
    .vs_cache_o_stream_TDATA  (vs_cache_o_stream_TDATA[7:0]              ), //i
    .vs_cache_o_stream_TVALID (vs_cache_o_stream_TVALID                  ), //i
    .vs_cache_o_stream_TREADY (black_box_vs_cache_o_stream_TREADY        )  //o
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign idle = black_box_ap_idle;
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign black_box_chunk = {3'd0, _zz_chunk};
  assign gmem1_awvalid = black_box_m_axi_gmem1_AWVALID;
  assign gmem1_awaddr = black_box_m_axi_gmem1_AWADDR;
  assign gmem1_awid = black_box_m_axi_gmem1_AWID;
  assign gmem1_awlen = black_box_m_axi_gmem1_AWLEN;
  assign gmem1_awsize = black_box_m_axi_gmem1_AWSIZE;
  assign gmem1_awburst = black_box_m_axi_gmem1_AWBURST;
  assign gmem1_awlock = black_box_m_axi_gmem1_AWLOCK;
  assign gmem1_awcache = black_box_m_axi_gmem1_AWCACHE;
  assign gmem1_awprot = black_box_m_axi_gmem1_AWPROT;
  assign gmem1_awqos = black_box_m_axi_gmem1_AWQOS;
  assign gmem1_awregion = black_box_m_axi_gmem1_AWREGION;
  assign gmem1_awuser = black_box_m_axi_gmem1_AWUSER;
  assign gmem1_wvalid = black_box_m_axi_gmem1_WVALID;
  assign gmem1_wdata = black_box_m_axi_gmem1_WDATA;
  assign gmem1_wstrb = black_box_m_axi_gmem1_WSTRB;
  assign gmem1_wlast = black_box_m_axi_gmem1_WLAST;
  assign gmem1_wuser = black_box_m_axi_gmem1_WUSER;
  assign gmem1_arvalid = black_box_m_axi_gmem1_ARVALID;
  assign gmem1_araddr = black_box_m_axi_gmem1_ARADDR;
  assign gmem1_arid = black_box_m_axi_gmem1_ARID;
  assign gmem1_arlen = black_box_m_axi_gmem1_ARLEN;
  assign gmem1_arsize = black_box_m_axi_gmem1_ARSIZE;
  assign gmem1_arburst = black_box_m_axi_gmem1_ARBURST;
  assign gmem1_arlock = black_box_m_axi_gmem1_ARLOCK;
  assign gmem1_arcache = black_box_m_axi_gmem1_ARCACHE;
  assign gmem1_arprot = black_box_m_axi_gmem1_ARPROT;
  assign gmem1_arqos = black_box_m_axi_gmem1_ARQOS;
  assign gmem1_arregion = black_box_m_axi_gmem1_ARREGION;
  assign gmem1_aruser = black_box_m_axi_gmem1_ARUSER;
  assign gmem1_rready = black_box_m_axi_gmem1_RREADY;
  assign gmem1_bready = black_box_m_axi_gmem1_BREADY;
  assign kq_cache_i_stream_TDATA = black_box_kq_cache_i_stream_TDATA;
  assign kq_cache_i_stream_TVALID = black_box_kq_cache_i_stream_TVALID;
  assign kq_cache_o_stream_TREADY = black_box_kq_cache_o_stream_TREADY;
  assign ks_cache_i_stream_TDATA = black_box_ks_cache_i_stream_TDATA;
  assign ks_cache_i_stream_TVALID = black_box_ks_cache_i_stream_TVALID;
  assign ks_cache_o_stream_TREADY = black_box_ks_cache_o_stream_TREADY;
  assign vq_cache_i_stream_TDATA = black_box_vq_cache_i_stream_TDATA;
  assign vq_cache_i_stream_TVALID = black_box_vq_cache_i_stream_TVALID;
  assign vq_cache_o_stream_TREADY = black_box_vq_cache_o_stream_TREADY;
  assign vs_cache_i_stream_TDATA = black_box_vs_cache_i_stream_TDATA;
  assign vs_cache_i_stream_TVALID = black_box_vs_cache_i_stream_TVALID;
  assign vs_cache_o_stream_TREADY = black_box_vs_cache_o_stream_TREADY;

endmodule

module tryo_mmu_wrapper (
  input  wire          resetn,
  input  wire          clk,
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  output wire          gmem1_awvalid,
  input  wire          gmem1_awready,
  output wire [63:0]   gmem1_awaddr,
  output wire [0:0]    gmem1_awid,
  output wire [3:0]    gmem1_awregion,
  output wire [7:0]    gmem1_awlen,
  output wire [2:0]    gmem1_awsize,
  output wire [1:0]    gmem1_awburst,
  output wire [0:0]    gmem1_awlock,
  output wire [3:0]    gmem1_awcache,
  output wire [3:0]    gmem1_awqos,
  output wire [0:0]    gmem1_awuser,
  output wire [2:0]    gmem1_awprot,
  output wire          gmem1_wvalid,
  input  wire          gmem1_wready,
  output wire [127:0]  gmem1_wdata,
  output wire [15:0]   gmem1_wstrb,
  output wire [0:0]    gmem1_wuser,
  output wire          gmem1_wlast,
  input  wire          gmem1_bvalid,
  output wire          gmem1_bready,
  input  wire [0:0]    gmem1_bid,
  input  wire [1:0]    gmem1_bresp,
  input  wire [0:0]    gmem1_buser,
  output wire          gmem1_arvalid,
  input  wire          gmem1_arready,
  output wire [63:0]   gmem1_araddr,
  output wire [0:0]    gmem1_arid,
  output wire [3:0]    gmem1_arregion,
  output wire [7:0]    gmem1_arlen,
  output wire [2:0]    gmem1_arsize,
  output wire [1:0]    gmem1_arburst,
  output wire [0:0]    gmem1_arlock,
  output wire [3:0]    gmem1_arcache,
  output wire [3:0]    gmem1_arqos,
  output wire [0:0]    gmem1_aruser,
  output wire [2:0]    gmem1_arprot,
  input  wire          gmem1_rvalid,
  output wire          gmem1_rready,
  input  wire [127:0]  gmem1_rdata,
  input  wire [0:0]    gmem1_rid,
  input  wire [1:0]    gmem1_rresp,
  input  wire          gmem1_rlast,
  input  wire [0:0]    gmem1_ruser,
  output wire          gmem2_awvalid,
  input  wire          gmem2_awready,
  output wire [63:0]   gmem2_awaddr,
  output wire [0:0]    gmem2_awid,
  output wire [3:0]    gmem2_awregion,
  output wire [7:0]    gmem2_awlen,
  output wire [2:0]    gmem2_awsize,
  output wire [1:0]    gmem2_awburst,
  output wire [0:0]    gmem2_awlock,
  output wire [3:0]    gmem2_awcache,
  output wire [3:0]    gmem2_awqos,
  output wire [0:0]    gmem2_awuser,
  output wire [2:0]    gmem2_awprot,
  output wire          gmem2_wvalid,
  input  wire          gmem2_wready,
  output wire [127:0]  gmem2_wdata,
  output wire [15:0]   gmem2_wstrb,
  output wire [0:0]    gmem2_wuser,
  output wire          gmem2_wlast,
  input  wire          gmem2_bvalid,
  output wire          gmem2_bready,
  input  wire [0:0]    gmem2_bid,
  input  wire [1:0]    gmem2_bresp,
  input  wire [0:0]    gmem2_buser,
  output wire          gmem2_arvalid,
  input  wire          gmem2_arready,
  output wire [63:0]   gmem2_araddr,
  output wire [0:0]    gmem2_arid,
  output wire [3:0]    gmem2_arregion,
  output wire [7:0]    gmem2_arlen,
  output wire [2:0]    gmem2_arsize,
  output wire [1:0]    gmem2_arburst,
  output wire [0:0]    gmem2_arlock,
  output wire [3:0]    gmem2_arcache,
  output wire [3:0]    gmem2_arqos,
  output wire [0:0]    gmem2_aruser,
  output wire [2:0]    gmem2_arprot,
  input  wire          gmem2_rvalid,
  output wire          gmem2_rready,
  input  wire [127:0]  gmem2_rdata,
  input  wire [0:0]    gmem2_rid,
  input  wire [1:0]    gmem2_rresp,
  input  wire          gmem2_rlast,
  input  wire [0:0]    gmem2_ruser,
  output wire          x_stream_TVALID,
  input  wire          x_stream_TREADY,
  output wire [199:0]  x_stream_TDATA,
  output wire          wq_stream_TVALID,
  input  wire          wq_stream_TREADY,
  output wire [319:0]  wq_stream_TDATA,
  output wire          ws1_stream_TVALID,
  input  wire          ws1_stream_TREADY,
  output wire [39:0]   ws1_stream_TDATA,
  output wire          ws2_stream_TVALID,
  input  wire          ws2_stream_TREADY,
  output wire [39:0]   ws2_stream_TDATA,
  input  wire          y_stream_TVALID,
  output wire          y_stream_TREADY,
  input  wire [199:0]  y_stream_TDATA,
  input  wire          logit_stream_TVALID,
  output wire          logit_stream_TREADY,
  input  wire [255:0]  logit_stream_TDATA,
  output wire          idle
);

  wire                black_box_ap_idle;
  wire                black_box_ap_ready;
  wire                black_box_ap_done;
  wire                black_box_m_axi_gmem1_AWVALID;
  wire       [63:0]   black_box_m_axi_gmem1_AWADDR;
  wire       [0:0]    black_box_m_axi_gmem1_AWID;
  wire       [0:0]    black_box_m_axi_gmem1_AWUSER;
  wire       [3:0]    black_box_m_axi_gmem1_AWREGION;
  wire       [7:0]    black_box_m_axi_gmem1_AWLEN;
  wire       [2:0]    black_box_m_axi_gmem1_AWSIZE;
  wire       [1:0]    black_box_m_axi_gmem1_AWBURST;
  wire       [0:0]    black_box_m_axi_gmem1_AWLOCK;
  wire       [3:0]    black_box_m_axi_gmem1_AWCACHE;
  wire       [3:0]    black_box_m_axi_gmem1_AWQOS;
  wire       [2:0]    black_box_m_axi_gmem1_AWPROT;
  wire                black_box_m_axi_gmem1_WVALID;
  wire       [127:0]  black_box_m_axi_gmem1_WDATA;
  wire       [15:0]   black_box_m_axi_gmem1_WSTRB;
  wire                black_box_m_axi_gmem1_WLAST;
  wire       [0:0]    black_box_m_axi_gmem1_WID;
  wire       [0:0]    black_box_m_axi_gmem1_WUSER;
  wire                black_box_m_axi_gmem1_ARVALID;
  wire       [63:0]   black_box_m_axi_gmem1_ARADDR;
  wire       [0:0]    black_box_m_axi_gmem1_ARID;
  wire       [0:0]    black_box_m_axi_gmem1_ARUSER;
  wire       [3:0]    black_box_m_axi_gmem1_ARREGION;
  wire       [7:0]    black_box_m_axi_gmem1_ARLEN;
  wire       [2:0]    black_box_m_axi_gmem1_ARSIZE;
  wire       [1:0]    black_box_m_axi_gmem1_ARBURST;
  wire       [0:0]    black_box_m_axi_gmem1_ARLOCK;
  wire       [3:0]    black_box_m_axi_gmem1_ARCACHE;
  wire       [3:0]    black_box_m_axi_gmem1_ARQOS;
  wire       [2:0]    black_box_m_axi_gmem1_ARPROT;
  wire                black_box_m_axi_gmem1_RREADY;
  wire                black_box_m_axi_gmem1_BREADY;
  wire                black_box_m_axi_gmem2_AWVALID;
  wire       [63:0]   black_box_m_axi_gmem2_AWADDR;
  wire       [0:0]    black_box_m_axi_gmem2_AWID;
  wire       [0:0]    black_box_m_axi_gmem2_AWUSER;
  wire       [3:0]    black_box_m_axi_gmem2_AWREGION;
  wire       [7:0]    black_box_m_axi_gmem2_AWLEN;
  wire       [2:0]    black_box_m_axi_gmem2_AWSIZE;
  wire       [1:0]    black_box_m_axi_gmem2_AWBURST;
  wire       [0:0]    black_box_m_axi_gmem2_AWLOCK;
  wire       [3:0]    black_box_m_axi_gmem2_AWCACHE;
  wire       [3:0]    black_box_m_axi_gmem2_AWQOS;
  wire       [2:0]    black_box_m_axi_gmem2_AWPROT;
  wire                black_box_m_axi_gmem2_WVALID;
  wire       [127:0]  black_box_m_axi_gmem2_WDATA;
  wire       [15:0]   black_box_m_axi_gmem2_WSTRB;
  wire                black_box_m_axi_gmem2_WLAST;
  wire       [0:0]    black_box_m_axi_gmem2_WID;
  wire       [0:0]    black_box_m_axi_gmem2_WUSER;
  wire                black_box_m_axi_gmem2_ARVALID;
  wire       [63:0]   black_box_m_axi_gmem2_ARADDR;
  wire       [0:0]    black_box_m_axi_gmem2_ARID;
  wire       [0:0]    black_box_m_axi_gmem2_ARUSER;
  wire       [3:0]    black_box_m_axi_gmem2_ARREGION;
  wire       [7:0]    black_box_m_axi_gmem2_ARLEN;
  wire       [2:0]    black_box_m_axi_gmem2_ARSIZE;
  wire       [1:0]    black_box_m_axi_gmem2_ARBURST;
  wire       [0:0]    black_box_m_axi_gmem2_ARLOCK;
  wire       [3:0]    black_box_m_axi_gmem2_ARCACHE;
  wire       [3:0]    black_box_m_axi_gmem2_ARQOS;
  wire       [2:0]    black_box_m_axi_gmem2_ARPROT;
  wire                black_box_m_axi_gmem2_RREADY;
  wire                black_box_m_axi_gmem2_BREADY;
  wire       [199:0]  black_box_x_stream_TDATA;
  wire                black_box_x_stream_TVALID;
  wire       [319:0]  black_box_wq_stream_TDATA;
  wire                black_box_wq_stream_TVALID;
  wire       [39:0]   black_box_ws1_stream_TDATA;
  wire                black_box_ws1_stream_TVALID;
  wire       [39:0]   black_box_ws2_stream_TDATA;
  wire                black_box_ws2_stream_TVALID;
  wire                black_box_y_stream_TREADY;
  wire                black_box_cls_stream_TREADY;
  wire       [31:0]   manager_12_signals_O_L_BEGIN;
  wire       [31:0]   manager_12_signals_O_L_CLOSE;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_X;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_Y;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_DECODER_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_LO;
  wire       [63:0]   manager_12_signals_O_MEMORY_LOGIT_W_HI;
  wire       [63:0]   manager_12_signals_O_MEMORY_KV_CACHE;
  wire       [31:0]   manager_12_signals_O_POS_ID;
  wire                manager_12_signals_O_T;
  wire                manager_12_ap_ctrl_ap_start;
  wire                manager_12_ap_ctrl_ap_continue;

  M_AXI black_box (
    .ap_clk               (clk                                           ), //i
    .ap_rst_n             (resetn                                        ), //i
    .l_begin              (manager_12_signals_O_L_BEGIN[31:0]            ), //i
    .l_close              (manager_12_signals_O_L_CLOSE[31:0]            ), //i
    .memory_decoder_x     (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //i
    .memory_decoder_y     (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //i
    .memory_cls_y         (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //i
    .memory_decoder_w_lo  (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //i
    .memory_decoder_w_hi  (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //i
    .memory_cls_w_lo      (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //i
    .memory_cls_w_hi      (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //i
    .ap_start             (manager_12_ap_ctrl_ap_start                   ), //i
    .ap_continue          (manager_12_ap_ctrl_ap_continue                ), //i
    .ap_idle              (black_box_ap_idle                             ), //o
    .ap_ready             (black_box_ap_ready                            ), //o
    .ap_done              (black_box_ap_done                             ), //o
    .m_axi_gmem1_AWVALID  (black_box_m_axi_gmem1_AWVALID                 ), //o
    .m_axi_gmem1_AWREADY  (gmem1_awready                                 ), //i
    .m_axi_gmem1_AWADDR   (black_box_m_axi_gmem1_AWADDR[63:0]            ), //o
    .m_axi_gmem1_AWID     (black_box_m_axi_gmem1_AWID                    ), //o
    .m_axi_gmem1_AWUSER   (black_box_m_axi_gmem1_AWUSER                  ), //o
    .m_axi_gmem1_AWREGION (black_box_m_axi_gmem1_AWREGION[3:0]           ), //o
    .m_axi_gmem1_AWLEN    (black_box_m_axi_gmem1_AWLEN[7:0]              ), //o
    .m_axi_gmem1_AWSIZE   (black_box_m_axi_gmem1_AWSIZE[2:0]             ), //o
    .m_axi_gmem1_AWBURST  (black_box_m_axi_gmem1_AWBURST[1:0]            ), //o
    .m_axi_gmem1_AWLOCK   (black_box_m_axi_gmem1_AWLOCK                  ), //o
    .m_axi_gmem1_AWCACHE  (black_box_m_axi_gmem1_AWCACHE[3:0]            ), //o
    .m_axi_gmem1_AWQOS    (black_box_m_axi_gmem1_AWQOS[3:0]              ), //o
    .m_axi_gmem1_AWPROT   (black_box_m_axi_gmem1_AWPROT[2:0]             ), //o
    .m_axi_gmem1_WVALID   (black_box_m_axi_gmem1_WVALID                  ), //o
    .m_axi_gmem1_WREADY   (gmem1_wready                                  ), //i
    .m_axi_gmem1_WDATA    (black_box_m_axi_gmem1_WDATA[127:0]            ), //o
    .m_axi_gmem1_WSTRB    (black_box_m_axi_gmem1_WSTRB[15:0]             ), //o
    .m_axi_gmem1_WLAST    (black_box_m_axi_gmem1_WLAST                   ), //o
    .m_axi_gmem1_WID      (black_box_m_axi_gmem1_WID                     ), //o
    .m_axi_gmem1_WUSER    (black_box_m_axi_gmem1_WUSER                   ), //o
    .m_axi_gmem1_ARVALID  (black_box_m_axi_gmem1_ARVALID                 ), //o
    .m_axi_gmem1_ARREADY  (gmem1_arready                                 ), //i
    .m_axi_gmem1_ARADDR   (black_box_m_axi_gmem1_ARADDR[63:0]            ), //o
    .m_axi_gmem1_ARID     (black_box_m_axi_gmem1_ARID                    ), //o
    .m_axi_gmem1_ARUSER   (black_box_m_axi_gmem1_ARUSER                  ), //o
    .m_axi_gmem1_ARREGION (black_box_m_axi_gmem1_ARREGION[3:0]           ), //o
    .m_axi_gmem1_ARLEN    (black_box_m_axi_gmem1_ARLEN[7:0]              ), //o
    .m_axi_gmem1_ARSIZE   (black_box_m_axi_gmem1_ARSIZE[2:0]             ), //o
    .m_axi_gmem1_ARBURST  (black_box_m_axi_gmem1_ARBURST[1:0]            ), //o
    .m_axi_gmem1_ARLOCK   (black_box_m_axi_gmem1_ARLOCK                  ), //o
    .m_axi_gmem1_ARCACHE  (black_box_m_axi_gmem1_ARCACHE[3:0]            ), //o
    .m_axi_gmem1_ARQOS    (black_box_m_axi_gmem1_ARQOS[3:0]              ), //o
    .m_axi_gmem1_ARPROT   (black_box_m_axi_gmem1_ARPROT[2:0]             ), //o
    .m_axi_gmem1_RVALID   (gmem1_rvalid                                  ), //i
    .m_axi_gmem1_RREADY   (black_box_m_axi_gmem1_RREADY                  ), //o
    .m_axi_gmem1_RDATA    (gmem1_rdata[127:0]                            ), //i
    .m_axi_gmem1_RLAST    (gmem1_rlast                                   ), //i
    .m_axi_gmem1_RID      (gmem1_rid                                     ), //i
    .m_axi_gmem1_RUSER    (gmem1_ruser                                   ), //i
    .m_axi_gmem1_RRESP    (gmem1_rresp[1:0]                              ), //i
    .m_axi_gmem1_BVALID   (gmem1_bvalid                                  ), //i
    .m_axi_gmem1_BREADY   (black_box_m_axi_gmem1_BREADY                  ), //o
    .m_axi_gmem1_BID      (gmem1_bid                                     ), //i
    .m_axi_gmem1_BUSER    (gmem1_buser                                   ), //i
    .m_axi_gmem1_BRESP    (gmem1_bresp[1:0]                              ), //i
    .m_axi_gmem2_AWVALID  (black_box_m_axi_gmem2_AWVALID                 ), //o
    .m_axi_gmem2_AWREADY  (gmem2_awready                                 ), //i
    .m_axi_gmem2_AWADDR   (black_box_m_axi_gmem2_AWADDR[63:0]            ), //o
    .m_axi_gmem2_AWID     (black_box_m_axi_gmem2_AWID                    ), //o
    .m_axi_gmem2_AWUSER   (black_box_m_axi_gmem2_AWUSER                  ), //o
    .m_axi_gmem2_AWREGION (black_box_m_axi_gmem2_AWREGION[3:0]           ), //o
    .m_axi_gmem2_AWLEN    (black_box_m_axi_gmem2_AWLEN[7:0]              ), //o
    .m_axi_gmem2_AWSIZE   (black_box_m_axi_gmem2_AWSIZE[2:0]             ), //o
    .m_axi_gmem2_AWBURST  (black_box_m_axi_gmem2_AWBURST[1:0]            ), //o
    .m_axi_gmem2_AWLOCK   (black_box_m_axi_gmem2_AWLOCK                  ), //o
    .m_axi_gmem2_AWCACHE  (black_box_m_axi_gmem2_AWCACHE[3:0]            ), //o
    .m_axi_gmem2_AWQOS    (black_box_m_axi_gmem2_AWQOS[3:0]              ), //o
    .m_axi_gmem2_AWPROT   (black_box_m_axi_gmem2_AWPROT[2:0]             ), //o
    .m_axi_gmem2_WVALID   (black_box_m_axi_gmem2_WVALID                  ), //o
    .m_axi_gmem2_WREADY   (gmem2_wready                                  ), //i
    .m_axi_gmem2_WDATA    (black_box_m_axi_gmem2_WDATA[127:0]            ), //o
    .m_axi_gmem2_WSTRB    (black_box_m_axi_gmem2_WSTRB[15:0]             ), //o
    .m_axi_gmem2_WLAST    (black_box_m_axi_gmem2_WLAST                   ), //o
    .m_axi_gmem2_WID      (black_box_m_axi_gmem2_WID                     ), //o
    .m_axi_gmem2_WUSER    (black_box_m_axi_gmem2_WUSER                   ), //o
    .m_axi_gmem2_ARVALID  (black_box_m_axi_gmem2_ARVALID                 ), //o
    .m_axi_gmem2_ARREADY  (gmem2_arready                                 ), //i
    .m_axi_gmem2_ARADDR   (black_box_m_axi_gmem2_ARADDR[63:0]            ), //o
    .m_axi_gmem2_ARID     (black_box_m_axi_gmem2_ARID                    ), //o
    .m_axi_gmem2_ARUSER   (black_box_m_axi_gmem2_ARUSER                  ), //o
    .m_axi_gmem2_ARREGION (black_box_m_axi_gmem2_ARREGION[3:0]           ), //o
    .m_axi_gmem2_ARLEN    (black_box_m_axi_gmem2_ARLEN[7:0]              ), //o
    .m_axi_gmem2_ARSIZE   (black_box_m_axi_gmem2_ARSIZE[2:0]             ), //o
    .m_axi_gmem2_ARBURST  (black_box_m_axi_gmem2_ARBURST[1:0]            ), //o
    .m_axi_gmem2_ARLOCK   (black_box_m_axi_gmem2_ARLOCK                  ), //o
    .m_axi_gmem2_ARCACHE  (black_box_m_axi_gmem2_ARCACHE[3:0]            ), //o
    .m_axi_gmem2_ARQOS    (black_box_m_axi_gmem2_ARQOS[3:0]              ), //o
    .m_axi_gmem2_ARPROT   (black_box_m_axi_gmem2_ARPROT[2:0]             ), //o
    .m_axi_gmem2_RVALID   (gmem2_rvalid                                  ), //i
    .m_axi_gmem2_RREADY   (black_box_m_axi_gmem2_RREADY                  ), //o
    .m_axi_gmem2_RDATA    (gmem2_rdata[127:0]                            ), //i
    .m_axi_gmem2_RLAST    (gmem2_rlast                                   ), //i
    .m_axi_gmem2_RID      (gmem2_rid                                     ), //i
    .m_axi_gmem2_RUSER    (gmem2_ruser                                   ), //i
    .m_axi_gmem2_RRESP    (gmem2_rresp[1:0]                              ), //i
    .m_axi_gmem2_BVALID   (gmem2_bvalid                                  ), //i
    .m_axi_gmem2_BREADY   (black_box_m_axi_gmem2_BREADY                  ), //o
    .m_axi_gmem2_BID      (gmem2_bid                                     ), //i
    .m_axi_gmem2_BUSER    (gmem2_buser                                   ), //i
    .m_axi_gmem2_BRESP    (gmem2_bresp[1:0]                              ), //i
    .x_stream_TDATA       (black_box_x_stream_TDATA[199:0]               ), //o
    .x_stream_TVALID      (black_box_x_stream_TVALID                     ), //o
    .x_stream_TREADY      (x_stream_TREADY                               ), //i
    .wq_stream_TDATA      (black_box_wq_stream_TDATA[319:0]              ), //o
    .wq_stream_TVALID     (black_box_wq_stream_TVALID                    ), //o
    .wq_stream_TREADY     (wq_stream_TREADY                              ), //i
    .ws1_stream_TDATA     (black_box_ws1_stream_TDATA[39:0]              ), //o
    .ws1_stream_TVALID    (black_box_ws1_stream_TVALID                   ), //o
    .ws1_stream_TREADY    (ws1_stream_TREADY                             ), //i
    .ws2_stream_TDATA     (black_box_ws2_stream_TDATA[39:0]              ), //o
    .ws2_stream_TVALID    (black_box_ws2_stream_TVALID                   ), //o
    .ws2_stream_TREADY    (ws2_stream_TREADY                             ), //i
    .y_stream_TDATA       (y_stream_TDATA[199:0]                         ), //i
    .y_stream_TVALID      (y_stream_TVALID                               ), //i
    .y_stream_TREADY      (black_box_y_stream_TREADY                     ), //o
    .cls_stream_TDATA     (logit_stream_TDATA[255:0]                     ), //i
    .cls_stream_TVALID    (logit_stream_TVALID                           ), //i
    .cls_stream_TREADY    (black_box_cls_stream_TREADY                   )  //o
  );
  Manager_11 manager_12 (
    .signals_I_L_BEGIN             (signals_I_L_BEGIN[31:0]                       ), //i
    .signals_I_L_CLOSE             (signals_I_L_CLOSE[31:0]                       ), //i
    .signals_I_MEMORY_DECODER_X    (signals_I_MEMORY_DECODER_X[63:0]              ), //i
    .signals_I_MEMORY_DECODER_Y    (signals_I_MEMORY_DECODER_Y[63:0]              ), //i
    .signals_I_MEMORY_LOGIT_Y      (signals_I_MEMORY_LOGIT_Y[63:0]                ), //i
    .signals_I_MEMORY_DECODER_W_LO (signals_I_MEMORY_DECODER_W_LO[63:0]           ), //i
    .signals_I_MEMORY_DECODER_W_HI (signals_I_MEMORY_DECODER_W_HI[63:0]           ), //i
    .signals_I_MEMORY_LOGIT_W_LO   (signals_I_MEMORY_LOGIT_W_LO[63:0]             ), //i
    .signals_I_MEMORY_LOGIT_W_HI   (signals_I_MEMORY_LOGIT_W_HI[63:0]             ), //i
    .signals_I_MEMORY_KV_CACHE     (signals_I_MEMORY_KV_CACHE[63:0]               ), //i
    .signals_I_POS_ID              (signals_I_POS_ID[31:0]                        ), //i
    .signals_I_T                   (signals_I_T                                   ), //i
    .signals_O_L_BEGIN             (manager_12_signals_O_L_BEGIN[31:0]            ), //o
    .signals_O_L_CLOSE             (manager_12_signals_O_L_CLOSE[31:0]            ), //o
    .signals_O_MEMORY_DECODER_X    (manager_12_signals_O_MEMORY_DECODER_X[63:0]   ), //o
    .signals_O_MEMORY_DECODER_Y    (manager_12_signals_O_MEMORY_DECODER_Y[63:0]   ), //o
    .signals_O_MEMORY_LOGIT_Y      (manager_12_signals_O_MEMORY_LOGIT_Y[63:0]     ), //o
    .signals_O_MEMORY_DECODER_W_LO (manager_12_signals_O_MEMORY_DECODER_W_LO[63:0]), //o
    .signals_O_MEMORY_DECODER_W_HI (manager_12_signals_O_MEMORY_DECODER_W_HI[63:0]), //o
    .signals_O_MEMORY_LOGIT_W_LO   (manager_12_signals_O_MEMORY_LOGIT_W_LO[63:0]  ), //o
    .signals_O_MEMORY_LOGIT_W_HI   (manager_12_signals_O_MEMORY_LOGIT_W_HI[63:0]  ), //o
    .signals_O_MEMORY_KV_CACHE     (manager_12_signals_O_MEMORY_KV_CACHE[63:0]    ), //o
    .signals_O_POS_ID              (manager_12_signals_O_POS_ID[31:0]             ), //o
    .signals_O_T                   (manager_12_signals_O_T                        ), //o
    .ap_ctrl_ap_start              (manager_12_ap_ctrl_ap_start                   ), //o
    .ap_ctrl_ap_continue           (manager_12_ap_ctrl_ap_continue                ), //o
    .ap_ctrl_ap_idle               (black_box_ap_idle                             ), //i
    .ap_ctrl_ap_ready              (black_box_ap_ready                            ), //i
    .ap_ctrl_ap_done               (black_box_ap_done                             ), //i
    .clk                           (clk                                           ), //i
    .resetn                        (resetn                                        )  //i
  );
  assign idle = black_box_ap_idle;
  assign signals_O_L_BEGIN = manager_12_signals_O_L_BEGIN;
  assign signals_O_L_CLOSE = manager_12_signals_O_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = manager_12_signals_O_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = manager_12_signals_O_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = manager_12_signals_O_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = manager_12_signals_O_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = manager_12_signals_O_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = manager_12_signals_O_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = manager_12_signals_O_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = manager_12_signals_O_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = manager_12_signals_O_POS_ID;
  assign signals_O_T = manager_12_signals_O_T;
  assign gmem1_awvalid = black_box_m_axi_gmem1_AWVALID;
  assign gmem1_awaddr = black_box_m_axi_gmem1_AWADDR;
  assign gmem1_awid = black_box_m_axi_gmem1_AWID;
  assign gmem1_awlen = black_box_m_axi_gmem1_AWLEN;
  assign gmem1_awsize = black_box_m_axi_gmem1_AWSIZE;
  assign gmem1_awburst = black_box_m_axi_gmem1_AWBURST;
  assign gmem1_awlock = black_box_m_axi_gmem1_AWLOCK;
  assign gmem1_awcache = black_box_m_axi_gmem1_AWCACHE;
  assign gmem1_awprot = black_box_m_axi_gmem1_AWPROT;
  assign gmem1_awqos = black_box_m_axi_gmem1_AWQOS;
  assign gmem1_awregion = black_box_m_axi_gmem1_AWREGION;
  assign gmem1_awuser = black_box_m_axi_gmem1_AWUSER;
  assign gmem1_wvalid = black_box_m_axi_gmem1_WVALID;
  assign gmem1_wdata = black_box_m_axi_gmem1_WDATA;
  assign gmem1_wstrb = black_box_m_axi_gmem1_WSTRB;
  assign gmem1_wlast = black_box_m_axi_gmem1_WLAST;
  assign gmem1_wuser = black_box_m_axi_gmem1_WUSER;
  assign gmem1_arvalid = black_box_m_axi_gmem1_ARVALID;
  assign gmem1_araddr = black_box_m_axi_gmem1_ARADDR;
  assign gmem1_arid = black_box_m_axi_gmem1_ARID;
  assign gmem1_arlen = black_box_m_axi_gmem1_ARLEN;
  assign gmem1_arsize = black_box_m_axi_gmem1_ARSIZE;
  assign gmem1_arburst = black_box_m_axi_gmem1_ARBURST;
  assign gmem1_arlock = black_box_m_axi_gmem1_ARLOCK;
  assign gmem1_arcache = black_box_m_axi_gmem1_ARCACHE;
  assign gmem1_arprot = black_box_m_axi_gmem1_ARPROT;
  assign gmem1_arqos = black_box_m_axi_gmem1_ARQOS;
  assign gmem1_arregion = black_box_m_axi_gmem1_ARREGION;
  assign gmem1_aruser = black_box_m_axi_gmem1_ARUSER;
  assign gmem1_rready = black_box_m_axi_gmem1_RREADY;
  assign gmem1_bready = black_box_m_axi_gmem1_BREADY;
  assign gmem2_awvalid = black_box_m_axi_gmem2_AWVALID;
  assign gmem2_awaddr = black_box_m_axi_gmem2_AWADDR;
  assign gmem2_awid = black_box_m_axi_gmem2_AWID;
  assign gmem2_awlen = black_box_m_axi_gmem2_AWLEN;
  assign gmem2_awsize = black_box_m_axi_gmem2_AWSIZE;
  assign gmem2_awburst = black_box_m_axi_gmem2_AWBURST;
  assign gmem2_awlock = black_box_m_axi_gmem2_AWLOCK;
  assign gmem2_awcache = black_box_m_axi_gmem2_AWCACHE;
  assign gmem2_awprot = black_box_m_axi_gmem2_AWPROT;
  assign gmem2_awqos = black_box_m_axi_gmem2_AWQOS;
  assign gmem2_awregion = black_box_m_axi_gmem2_AWREGION;
  assign gmem2_awuser = black_box_m_axi_gmem2_AWUSER;
  assign gmem2_wvalid = black_box_m_axi_gmem2_WVALID;
  assign gmem2_wdata = black_box_m_axi_gmem2_WDATA;
  assign gmem2_wstrb = black_box_m_axi_gmem2_WSTRB;
  assign gmem2_wlast = black_box_m_axi_gmem2_WLAST;
  assign gmem2_wuser = black_box_m_axi_gmem2_WUSER;
  assign gmem2_arvalid = black_box_m_axi_gmem2_ARVALID;
  assign gmem2_araddr = black_box_m_axi_gmem2_ARADDR;
  assign gmem2_arid = black_box_m_axi_gmem2_ARID;
  assign gmem2_arlen = black_box_m_axi_gmem2_ARLEN;
  assign gmem2_arsize = black_box_m_axi_gmem2_ARSIZE;
  assign gmem2_arburst = black_box_m_axi_gmem2_ARBURST;
  assign gmem2_arlock = black_box_m_axi_gmem2_ARLOCK;
  assign gmem2_arcache = black_box_m_axi_gmem2_ARCACHE;
  assign gmem2_arprot = black_box_m_axi_gmem2_ARPROT;
  assign gmem2_arqos = black_box_m_axi_gmem2_ARQOS;
  assign gmem2_arregion = black_box_m_axi_gmem2_ARREGION;
  assign gmem2_aruser = black_box_m_axi_gmem2_ARUSER;
  assign gmem2_rready = black_box_m_axi_gmem2_RREADY;
  assign gmem2_bready = black_box_m_axi_gmem2_BREADY;
  assign x_stream_TDATA = black_box_x_stream_TDATA;
  assign x_stream_TVALID = black_box_x_stream_TVALID;
  assign wq_stream_TDATA = black_box_wq_stream_TDATA;
  assign wq_stream_TVALID = black_box_wq_stream_TVALID;
  assign ws1_stream_TDATA = black_box_ws1_stream_TDATA;
  assign ws1_stream_TVALID = black_box_ws1_stream_TVALID;
  assign ws2_stream_TDATA = black_box_ws2_stream_TDATA;
  assign ws2_stream_TVALID = black_box_ws2_stream_TVALID;
  assign y_stream_TREADY = black_box_y_stream_TREADY;
  assign logit_stream_TREADY = black_box_cls_stream_TREADY;

endmodule

//Manager replaced by Manager_11

//Manager_1 replaced by Manager_11

//Manager_2 replaced by Manager_11

//Manager_3 replaced by Manager_11

//Manager_4 replaced by Manager_11

//Manager_5 replaced by Manager_11

//Manager_6 replaced by Manager_11

//Manager_7 replaced by Manager_11

//Manager_8 replaced by Manager_11

//Manager_9 replaced by Manager_11

//Manager_10 replaced by Manager_11

module Manager_11 (
  input  wire [31:0]   signals_I_L_BEGIN,
  input  wire [31:0]   signals_I_L_CLOSE,
  input  wire [63:0]   signals_I_MEMORY_DECODER_X,
  input  wire [63:0]   signals_I_MEMORY_DECODER_Y,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_Y,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_LO,
  input  wire [63:0]   signals_I_MEMORY_DECODER_W_HI,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_LO,
  input  wire [63:0]   signals_I_MEMORY_LOGIT_W_HI,
  input  wire [63:0]   signals_I_MEMORY_KV_CACHE,
  input  wire [31:0]   signals_I_POS_ID,
  input  wire          signals_I_T,
  output wire [31:0]   signals_O_L_BEGIN,
  output wire [31:0]   signals_O_L_CLOSE,
  output wire [63:0]   signals_O_MEMORY_DECODER_X,
  output wire [63:0]   signals_O_MEMORY_DECODER_Y,
  output wire [63:0]   signals_O_MEMORY_LOGIT_Y,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_LO,
  output wire [63:0]   signals_O_MEMORY_DECODER_W_HI,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_LO,
  output wire [63:0]   signals_O_MEMORY_LOGIT_W_HI,
  output wire [63:0]   signals_O_MEMORY_KV_CACHE,
  output wire [31:0]   signals_O_POS_ID,
  output wire          signals_O_T,
  output reg           ap_ctrl_ap_start,
  output wire          ap_ctrl_ap_continue,
  input  wire          ap_ctrl_ap_idle,
  input  wire          ap_ctrl_ap_ready,
  input  wire          ap_ctrl_ap_done,
  input  wire          clk,
  input  wire          resetn
);
  localparam fsm_enumDef_BOOT = 2'd0;
  localparam fsm_enumDef_s_idle = 2'd1;
  localparam fsm_enumDef_s_work = 2'd2;

  reg        [31:0]   signals_I_regNext_L_BEGIN;
  reg        [31:0]   signals_I_regNext_L_CLOSE;
  reg        [63:0]   signals_I_regNext_MEMORY_DECODER_X;
  reg        [63:0]   signals_I_regNext_MEMORY_DECODER_Y;
  reg        [63:0]   signals_I_regNext_MEMORY_LOGIT_Y;
  reg        [63:0]   signals_I_regNext_MEMORY_DECODER_W_LO;
  reg        [63:0]   signals_I_regNext_MEMORY_DECODER_W_HI;
  reg        [63:0]   signals_I_regNext_MEMORY_LOGIT_W_LO;
  reg        [63:0]   signals_I_regNext_MEMORY_LOGIT_W_HI;
  reg        [63:0]   signals_I_regNext_MEMORY_KV_CACHE;
  reg        [31:0]   signals_I_regNext_POS_ID;
  reg                 signals_I_regNext_T;
  wire                fsm_wantExit;
  reg                 fsm_wantStart;
  wire                fsm_wantKill;
  reg        [1:0]    fsm_stateReg;
  reg        [1:0]    fsm_stateNext;
  `ifndef SYNTHESIS
  reg [47:0] fsm_stateReg_string;
  reg [47:0] fsm_stateNext_string;
  `endif


  `ifndef SYNTHESIS
  always @(*) begin
    case(fsm_stateReg)
      fsm_enumDef_BOOT : fsm_stateReg_string = "BOOT  ";
      fsm_enumDef_s_idle : fsm_stateReg_string = "s_idle";
      fsm_enumDef_s_work : fsm_stateReg_string = "s_work";
      default : fsm_stateReg_string = "??????";
    endcase
  end
  always @(*) begin
    case(fsm_stateNext)
      fsm_enumDef_BOOT : fsm_stateNext_string = "BOOT  ";
      fsm_enumDef_s_idle : fsm_stateNext_string = "s_idle";
      fsm_enumDef_s_work : fsm_stateNext_string = "s_work";
      default : fsm_stateNext_string = "??????";
    endcase
  end
  `endif

  assign signals_O_L_BEGIN = signals_I_regNext_L_BEGIN;
  assign signals_O_L_CLOSE = signals_I_regNext_L_CLOSE;
  assign signals_O_MEMORY_DECODER_X = signals_I_regNext_MEMORY_DECODER_X;
  assign signals_O_MEMORY_DECODER_Y = signals_I_regNext_MEMORY_DECODER_Y;
  assign signals_O_MEMORY_LOGIT_Y = signals_I_regNext_MEMORY_LOGIT_Y;
  assign signals_O_MEMORY_DECODER_W_LO = signals_I_regNext_MEMORY_DECODER_W_LO;
  assign signals_O_MEMORY_DECODER_W_HI = signals_I_regNext_MEMORY_DECODER_W_HI;
  assign signals_O_MEMORY_LOGIT_W_LO = signals_I_regNext_MEMORY_LOGIT_W_LO;
  assign signals_O_MEMORY_LOGIT_W_HI = signals_I_regNext_MEMORY_LOGIT_W_HI;
  assign signals_O_MEMORY_KV_CACHE = signals_I_regNext_MEMORY_KV_CACHE;
  assign signals_O_POS_ID = signals_I_regNext_POS_ID;
  assign signals_O_T = signals_I_regNext_T;
  assign ap_ctrl_ap_continue = 1'b1;
  always @(*) begin
    ap_ctrl_ap_start = 1'b0;
    case(fsm_stateReg)
      fsm_enumDef_s_idle : begin
      end
      fsm_enumDef_s_work : begin
        ap_ctrl_ap_start = 1'b1;
      end
      default : begin
      end
    endcase
  end

  assign fsm_wantExit = 1'b0;
  always @(*) begin
    fsm_wantStart = 1'b0;
    case(fsm_stateReg)
      fsm_enumDef_s_idle : begin
      end
      fsm_enumDef_s_work : begin
      end
      default : begin
        fsm_wantStart = 1'b1;
      end
    endcase
  end

  assign fsm_wantKill = 1'b0;
  always @(*) begin
    fsm_stateNext = fsm_stateReg;
    case(fsm_stateReg)
      fsm_enumDef_s_idle : begin
        if(signals_I_T) begin
          fsm_stateNext = fsm_enumDef_s_work;
        end
      end
      fsm_enumDef_s_work : begin
        if(ap_ctrl_ap_ready) begin
          fsm_stateNext = fsm_enumDef_s_idle;
        end
      end
      default : begin
      end
    endcase
    if(fsm_wantStart) begin
      fsm_stateNext = fsm_enumDef_s_idle;
    end
    if(fsm_wantKill) begin
      fsm_stateNext = fsm_enumDef_BOOT;
    end
  end

  always @(posedge clk) begin
    signals_I_regNext_L_BEGIN <= signals_I_L_BEGIN;
    signals_I_regNext_L_CLOSE <= signals_I_L_CLOSE;
    signals_I_regNext_MEMORY_DECODER_X <= signals_I_MEMORY_DECODER_X;
    signals_I_regNext_MEMORY_DECODER_Y <= signals_I_MEMORY_DECODER_Y;
    signals_I_regNext_MEMORY_LOGIT_Y <= signals_I_MEMORY_LOGIT_Y;
    signals_I_regNext_MEMORY_DECODER_W_LO <= signals_I_MEMORY_DECODER_W_LO;
    signals_I_regNext_MEMORY_DECODER_W_HI <= signals_I_MEMORY_DECODER_W_HI;
    signals_I_regNext_MEMORY_LOGIT_W_LO <= signals_I_MEMORY_LOGIT_W_LO;
    signals_I_regNext_MEMORY_LOGIT_W_HI <= signals_I_MEMORY_LOGIT_W_HI;
    signals_I_regNext_MEMORY_KV_CACHE <= signals_I_MEMORY_KV_CACHE;
    signals_I_regNext_POS_ID <= signals_I_POS_ID;
    signals_I_regNext_T <= signals_I_T;
  end

  always @(posedge clk) begin
    if(!resetn) begin
      fsm_stateReg <= fsm_enumDef_BOOT;
    end else begin
      fsm_stateReg <= fsm_stateNext;
    end
  end


endmodule
