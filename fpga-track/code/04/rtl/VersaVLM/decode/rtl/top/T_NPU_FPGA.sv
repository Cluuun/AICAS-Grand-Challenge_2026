//////////////////////////////////////////////////////////////////////////////////
// Copyright by FuxionLab 
//
// Designer     : Sihao Fu
// Create Date  : 2024/10/29
// Project Name : T_NPU
// File Name    : T_NPU.v 
//
// Description  : Top Module for T_NPU, connecting CPU with AXI-Lite. and DDR with AXI4
//
// Revision:  
// Revision 2.0 - File Created
// Additional Comments:     
//
//////////////////////////////////////////////////////////////////////////////////

`ifndef T_NPU_FPGA_SV
`define T_NPU_FPGA_SV

module T_NPU_FPGA #(
    parameter int RF_DATA_WIDTH        = 64,
    parameter int MMIO_AXI_DATA_WIDTH  = 64,
    parameter int MMIO_AXI_ADDR_WIDTH  = 32,
    parameter int AXI_ID_WIDTH         = 4,
    parameter int AXI_ADDR_WIDTH       = 32,
    parameter int AXI_DATA_WIDTH       = 128,
    parameter int PE_WIDTH             = 16,
    parameter int PE_DATA_WIDTH        = 8,
    parameter int PE_DATA_WIDTH_IN     = 8,
    parameter int PE_DATA_WIDTH_OUT    = 32,
    parameter int SA_MAX_LENGTH        = 4096,
    parameter int SPM_SIZE             = 1048576,
    parameter int RD_PORTS             = 2,
    parameter int ACC_SIZE             = 524288,
    parameter logic signed [31:0] ALPHA_1 = 32'sd6656,
    parameter logic signed [31:0] ALPHA_2 = 32'sd13312,
    parameter logic signed [31:0] ALPHA_3 = 32'sd656,
    parameter integer PE_FPGA_DSP       = 1,
    parameter integer DSP_PE_NUM        = 256,
    parameter integer SPM_FPGA_SRAM     = 1,
    parameter GELU_NUM                  = 4,
    parameter DISABLE_SYSTOLIC_ARRAY    = 0,
    parameter DISABLE_ACCUMULATOR       = 1,
    parameter DISABLE_GEMV              = 0,
    parameter DISABLE_SFU               = 1,
    parameter DISABLE_GELU              = 1,
    parameter DISABLE_SOFTMAX           = 1,//模型中需要softmax以及layernorm，但是暂时先disable
    parameter DISABLE_LAYERNORM         = 1,//模型中需要softmax以及layernorm，但是暂时先disable
    parameter DISABLE_RESAMPLE          = 1,
    parameter DISABLE_TRANSPOSE         = 1,
    parameter DISABLE_IM2COL            = 1,
    parameter int AXI_MAX_BURST_BEATS   = 16,
    parameter int AXI_MAX_OUTSTANDING   = 16,
    parameter DISABLE_MVIN_INT8_TO_INT32 = 1,
    parameter DISABLE_ACC_INT32_TO_INT8  = 1,
    parameter DISABLE_ACC_TO_SPM_PATH    = 1,
    parameter DISABLE_DECODE_ROPE        = 0,
    parameter DISABLE_DECODE_SOFTMAX     = 0,
    parameter DISABLE_DECODE_KV_QUANT    = 0,
    parameter int ENABLE_PROFILE_COUNTERS = 1,
    parameter int ENABLE_DETAILED_ERROR_STATUS = 1
)(
    input  logic    clk     ,
    input  logic    rst_n   ,
    
    //-------------------------------------
    // AXI-Lite interface
    //-------------------------------------
    input  logic [MMIO_AXI_ADDR_WIDTH-1 : 0]     s_axi_awaddr      , // Write address 
    input  logic [2 : 0]                         s_axi_awprot      , // Write channel Protection type.not used in this module
    input  logic                                 s_axi_awvalid     , // Write address valid.
    output logic                                 s_axi_awready     , // Write address ready.
    
    input  logic [MMIO_AXI_DATA_WIDTH-1 : 0]     s_axi_wdata       , // Write data  
    input  logic [(MMIO_AXI_DATA_WIDTH/8)-1 : 0] s_axi_wstrb       , // Write strobes.
    input  logic                                 s_axi_wvalid      , // Write valid.
    output logic                                 s_axi_wready      , // Write ready.
    
    output logic [1 : 0]                         s_axi_bresp       , // Write response.
    output logic                                 s_axi_bvalid      , // Write response valid.
    input  logic                                 s_axi_bready      , // Response ready.
    
    input  logic [MMIO_AXI_ADDR_WIDTH-1 : 0]     s_axi_araddr      , // Read address 
    input  logic [2 : 0]                         s_axi_arprot      , // Protection type.not used in this module
    input  logic                                 s_axi_arvalid     , // Read address valid.
    output logic                                 s_axi_arready     , // Read address ready.

    output logic [MMIO_AXI_DATA_WIDTH-1 : 0]     s_axi_rdata       , // Read data (issued by slave)
    output logic [1 : 0]                         s_axi_rresp       , // Read response.
    output logic                                 s_axi_rvalid      , // Read valid.
    input  logic                                 s_axi_rready      ,  // Read ready.

    output logic                                 irq                ,
    //-------------------------------------
    // DMA0 AXI-4 master interface
    //-------------------------------------
    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi0_awid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi0_awaddr              ,
    output  logic    [7 : 0]                                                m_axi0_awlen               ,
    output  logic    [2 : 0]                                                m_axi0_awsize              ,
    output  logic    [1 : 0]                                                m_axi0_awburst             ,
    output  logic                                                            m_axi0_awvalid             ,
    input   logic                                                            m_axi0_awready             ,

    output  logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi0_wdata               ,
    output  logic    [AXI_DATA_WIDTH/8-1 : 0]                               m_axi0_wstrb               ,
    output  logic                                                            m_axi0_wlast               ,
    output  logic                                                            m_axi0_wvalid              ,
    input   logic                                                            m_axi0_wready              ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi0_bid                 ,
    input   logic    [1 : 0]                                                m_axi0_bresp               ,
    input   logic                                                            m_axi0_bvalid              ,
    output  logic                                                            m_axi0_bready              ,

    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi0_arid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi0_araddr              ,
    output  logic    [7 : 0]                                                m_axi0_arlen               ,
    output  logic    [2 : 0]                                                m_axi0_arsize              ,
    output  logic    [1 : 0]                                                m_axi0_arburst             ,
    output  logic                                                            m_axi0_arvalid             ,
    input   logic                                                            m_axi0_arready             ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi0_rid                 ,
    input   logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi0_rdata               ,
    input   logic    [1 : 0]                                                m_axi0_rresp               ,
    input   logic                                                            m_axi0_rlast               ,
    input   logic                                                            m_axi0_rvalid              ,
    output  logic                                                            m_axi0_rready              ,

    // DMA1 AXI-4 master interface
    //-------------------------------------
    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi1_awid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi1_awaddr              ,
    output  logic    [7 : 0]                                                m_axi1_awlen               ,
    output  logic    [2 : 0]                                                m_axi1_awsize              ,
    output  logic    [1 : 0]                                                m_axi1_awburst             ,
    output  logic                                                            m_axi1_awvalid             ,
    input   logic                                                            m_axi1_awready             ,

    output  logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi1_wdata               ,
    output  logic    [AXI_DATA_WIDTH/8-1 : 0]                               m_axi1_wstrb               ,
    output  logic                                                            m_axi1_wlast               ,
    output  logic                                                            m_axi1_wvalid              ,
    input   logic                                                            m_axi1_wready              ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi1_bid                 ,
    input   logic    [1 : 0]                                                m_axi1_bresp               ,
    input   logic                                                            m_axi1_bvalid              ,
    output  logic                                                            m_axi1_bready              ,

    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi1_arid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi1_araddr              ,
    output  logic    [7 : 0]                                                m_axi1_arlen               ,
    output  logic    [2 : 0]                                                m_axi1_arsize              ,
    output  logic    [1 : 0]                                                m_axi1_arburst             ,
    output  logic                                                            m_axi1_arvalid             ,
    input   logic                                                            m_axi1_arready             ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi1_rid                 ,
    input   logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi1_rdata               ,
    input   logic    [1 : 0]                                                m_axi1_rresp               ,
    input   logic                                                            m_axi1_rlast               ,
    input   logic                                                            m_axi1_rvalid              ,
    output  logic                                                            m_axi1_rready              ,

    // DMA2 AXI-4 master interface
    //-------------------------------------
    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi2_awid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi2_awaddr              ,
    output  logic    [7 : 0]                                                m_axi2_awlen               ,
    output  logic    [2 : 0]                                                m_axi2_awsize              ,
    output  logic    [1 : 0]                                                m_axi2_awburst             ,
    output  logic                                                            m_axi2_awvalid             ,
    input   logic                                                            m_axi2_awready             ,

    output  logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi2_wdata               ,
    output  logic    [AXI_DATA_WIDTH/8-1 : 0]                               m_axi2_wstrb               ,
    output  logic                                                            m_axi2_wlast               ,
    output  logic                                                            m_axi2_wvalid              ,
    input   logic                                                            m_axi2_wready              ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi2_bid                 ,
    input   logic    [1 : 0]                                                m_axi2_bresp               ,
    input   logic                                                            m_axi2_bvalid              ,
    output  logic                                                            m_axi2_bready              ,

    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi2_arid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi2_araddr              ,
    output  logic    [7 : 0]                                                m_axi2_arlen               ,
    output  logic    [2 : 0]                                                m_axi2_arsize              ,
    output  logic    [1 : 0]                                                m_axi2_arburst             ,
    output  logic                                                            m_axi2_arvalid             ,
    input   logic                                                            m_axi2_arready             ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi2_rid                 ,
    input   logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi2_rdata               ,
    input   logic    [1 : 0]                                                m_axi2_rresp               ,
    input   logic                                                            m_axi2_rlast               ,
    input   logic                                                            m_axi2_rvalid              ,
    output  logic                                                            m_axi2_rready              ,

    // DMA3 AXI-4 master interface
    //-------------------------------------
    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi3_awid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi3_awaddr              ,
    output  logic    [7 : 0]                                                m_axi3_awlen               ,
    output  logic    [2 : 0]                                                m_axi3_awsize              ,
    output  logic    [1 : 0]                                                m_axi3_awburst             ,
    output  logic                                                            m_axi3_awvalid             ,
    input   logic                                                            m_axi3_awready             ,

    output  logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi3_wdata               ,
    output  logic    [AXI_DATA_WIDTH/8-1 : 0]                               m_axi3_wstrb               ,
    output  logic                                                            m_axi3_wlast               ,
    output  logic                                                            m_axi3_wvalid              ,
    input   logic                                                            m_axi3_wready              ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi3_bid                 ,
    input   logic    [1 : 0]                                                m_axi3_bresp               ,
    input   logic                                                            m_axi3_bvalid              ,
    output  logic                                                            m_axi3_bready              ,

    output  logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi3_arid                ,
    output  logic    [AXI_ADDR_WIDTH-1 : 0]                                 m_axi3_araddr              ,
    output  logic    [7 : 0]                                                m_axi3_arlen               ,
    output  logic    [2 : 0]                                                m_axi3_arsize              ,
    output  logic    [1 : 0]                                                m_axi3_arburst             ,
    output  logic                                                            m_axi3_arvalid             ,
    input   logic                                                            m_axi3_arready             ,

    input   logic    [AXI_ID_WIDTH-1 : 0]                                   m_axi3_rid                 ,
    input   logic    [AXI_DATA_WIDTH-1 : 0]                                 m_axi3_rdata               ,
    input   logic    [1 : 0]                                                m_axi3_rresp               ,
    input   logic                                                            m_axi3_rlast               ,
    input   logic                                                            m_axi3_rvalid              ,
    output  logic                                                            m_axi3_rready              ,
    output  logic                                                            busy
);
localparam int WR_PORTS      = 1;
localparam int ARRAY_WIDTH    = PE_WIDTH;
localparam int SPM_DATA_WIDTH = PE_DATA_WIDTH_IN * PE_WIDTH;
localparam int SPM_ADDR_WIDTH = $clog2(SPM_SIZE);
localparam int ACC_DATA_WIDTH = PE_DATA_WIDTH_OUT * PE_WIDTH;
localparam int ACC_ADDR_WIDTH = $clog2(ACC_SIZE);
localparam int ACC_DMA_IDX    = 0;
localparam int MVIN_4AXI_PORTS = 4;
localparam int MVIN_4AXI_LINE_BYTES = npu_config_pkg::SPM_LINE_DATA_WIDTH / 8;
localparam int MVIN_4AXI_SCALE_SPM_BYTES = 256 * 1024;
localparam int MVIN_4AXI_OUTPUT_SPM_BYTES = 512 * 1024;
localparam int MVIN_4AXI_SPM_DEPTH = SPM_SIZE / MVIN_4AXI_LINE_BYTES;
localparam int MVIN_4AXI_SCALE_SPM_DEPTH = MVIN_4AXI_SCALE_SPM_BYTES / MVIN_4AXI_LINE_BYTES;
localparam int MVIN_4AXI_OUTPUT_SPM_DEPTH = MVIN_4AXI_OUTPUT_SPM_BYTES / MVIN_4AXI_LINE_BYTES;
localparam int GEMV_SPM_DATA_WIDTH = npu_config_pkg::SPM_LINE_DATA_WIDTH;
localparam int GEMV_TILE_ELEMS = 128;
//dma signals

logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvin_dram_addr          ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvin_row_num            ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvin_sram_addr          ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvin_col_num            ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvout_dram_addr         ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvout_row_num           ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvout_sram_addr         ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_mvout_col_num           ;
logic    [0:0][1:0]                 dma_cfg_mvin_input_type     ;
logic    [0:0][1:0]                 dma_cfg_mvout_output_type   ;
logic    [0:0][1:0]                 dma_cfg_mvin_input_precision;
logic    [0:0][1:0]                 dma_cfg_mvout_output_precision;
logic    [0:0]                      dma_cfg_mvin_is_quant       ;
logic    [0:0]                      dma_cfg_mvout_is_quant      ;
logic    [0:0]                      dma_cfg_mvin_dest           ;
logic    [0:0]                      dma_cfg_mvout_source        ;
logic    [0:0]                      dma_cfg_mvout_per_channel   ;
logic    [0:0]                      dma_cfg_mvin_isbias         ;
logic    [0:0]                      dma_cfg_mvin_scale_target   ;
logic    [0:0]                      dma_cfg_mvin_stream_fifo_fill;
logic    [0:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvin_sram_stride    ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvin_dram_stride    ;
logic    [0:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvout_sram_stride   ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvout_dram_stride   ;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvin_input_zeropoint;
logic    [0:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvout_output_zeropoint;
logic    [0:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvin_input_scale    ;
logic    [0:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvout_output_scale  ;
logic    [0:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvin_input_scale_shift;
logic    [0:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvout_output_scale_shift;
logic                                             cfg_mvin_isbias;
logic    [1:0]                                    cfg_mvout_output_precision;
logic                                             cfg_mvout_per_channel;
logic    [RF_DATA_WIDTH/4-1:0]                    cfg_mvout_output_scale;
logic    [RF_DATA_WIDTH/4-1:0]                    cfg_mvout_output_scale_shift;
logic    [0:0]                      dma_mvin_req_en         ;
logic    [0:0]                      dma_mvout_req_en        ;
logic    [0:0]                      dma_mvin_busy_status           ;
logic    [0:0]                      dma_mvout_busy_status          ;
logic    [0:0]                      dma_mvin_resp_done_status      ;
logic    [0:0]                      dma_mvout_resp_done_status     ;
logic    [0:0]                      dma_mvin_req_en_dly     ;
logic    [0:0]                      dma_mvout_req_en_dly    ;
logic                                             dma_mvin_resp_done_any      ;
logic                                             dma_mvout_resp_done_any     ;

logic                                             dma_acc_wr_en              ;
logic   [$clog2(ACC_SIZE)-1:0]                    dma_acc_wr_addr            ;
logic   [ACC_DATA_WIDTH/32-1:0]                   dma_acc_wr_mask            ;
logic   [ACC_DATA_WIDTH-1:0]                      dma_acc_din                ;
logic                                             dma_acc_rd_en              ;
logic   [$clog2(ACC_SIZE)-1:0]                    dma_acc_rd_addr            ;
logic   [ACC_DATA_WIDTH-1:0]                      dma_acc_dout               ;
logic                                             dma_acc_rd_valid           ;
logic   [AXI_ADDR_WIDTH-1:0]                      mvin4_dram_base_addr_cfg   ;
logic   [$clog2(MVIN_4AXI_SPM_DEPTH)-1:0]         mvin4_spm_addr_cfg        ;
logic   [15:0]                                    mvin4_line_num_cfg        ;
logic                                             mvin4_stream_fifo_fill_cfg;
logic                                             mvin4_req_en_cfg          ;
logic                                             mvin4_scale_req_en_cfg    ;
logic                                             mvin4_act_req_en_cfg      ;
logic                                             mvin4_resp_done           ;
logic                                             mvin4_busy                ;
logic   [AXI_ADDR_WIDTH-1:0]                      mvout1_dram_base_addr_cfg ;
logic   [$clog2(MVIN_4AXI_OUTPUT_SPM_DEPTH)-1:0]  mvout1_spm_addr_cfg      ;
logic   [15:0]                                    mvout1_row_num_cfg       ;
logic   [1:0]                                     mvout1_output_precision_cfg;
logic                                             mvout1_req_en_cfg        ;
logic                                             mvout1_resp_done         ;
logic                                             mvout1_busy              ;
logic                                             mvout1_cfg_pending_q     ;
logic   [31:0]                                    mvout1_elem_count_q      ;
logic                                             mvin4_spm_beat_rd_en      ;
logic   [SPM_ADDR_WIDTH-1:0]                      mvin4_spm_beat_rd_addr    ;
logic   [AXI_DATA_WIDTH-1:0]                      mvin4_spm_beat_rd_data    ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]   mvin4_axi_awid            ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_ADDR_WIDTH-1:0] mvin4_axi_awaddr          ;
logic   [MVIN_4AXI_PORTS-1:0][7:0]                mvin4_axi_awlen           ;
logic   [MVIN_4AXI_PORTS-1:0][2:0]                mvin4_axi_awsize          ;
logic   [MVIN_4AXI_PORTS-1:0][1:0]                mvin4_axi_awburst         ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_awvalid         ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_awready         ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_DATA_WIDTH-1:0] mvin4_axi_wdata           ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_DATA_WIDTH/8-1:0] mvin4_axi_wstrb         ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_wlast           ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_wvalid          ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_wready          ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]   mvin4_axi_bid             ;
logic   [MVIN_4AXI_PORTS-1:0][1:0]                mvin4_axi_bresp           ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_bvalid          ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_bready          ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]   mvin4_axi_arid            ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_ADDR_WIDTH-1:0] mvin4_axi_araddr          ;
logic   [MVIN_4AXI_PORTS-1:0][7:0]                mvin4_axi_arlen           ;
logic   [MVIN_4AXI_PORTS-1:0][2:0]                mvin4_axi_arsize          ;
logic   [MVIN_4AXI_PORTS-1:0][1:0]                mvin4_axi_arburst         ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_arvalid         ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_arready         ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_ID_WIDTH-1:0]   mvin4_axi_rid             ;
logic   [MVIN_4AXI_PORTS-1:0][AXI_DATA_WIDTH-1:0] mvin4_axi_rdata           ;
logic   [MVIN_4AXI_PORTS-1:0][1:0]                mvin4_axi_rresp           ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_rlast           ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_rvalid          ;
logic   [MVIN_4AXI_PORTS-1:0]                     mvin4_axi_rready          ;
logic                                             mvin4_acc_rd_en          ;
logic   [ACC_ADDR_WIDTH-1:0]                      mvin4_acc_rd_addr        ;

localparam int MVIN_4AXI_SPM_ADDR_LSB  = $clog2(MVIN_4AXI_LINE_BYTES);
localparam int GEMV_MVOUT_ELEM_BYTES   = 2;
localparam logic [1:0] GEMV_MVOUT_PREC_RAW8 = 2'b00;

assign dma_mvin_resp_done_any = |dma_mvin_resp_done_status;
assign dma_mvout_resp_done_any = |dma_mvout_resp_done_status;

//sfu signals
logic    [5:0]                           cfg_sfu_op                  ;   // operation of SFU
logic    [1:0]                           cfg_sfu_int_type            ;   // datatype range of SFU input data, 00 for int8, 01 for int16, 10 for int32, 11 for int64  
logic                                    cfg_sfu_is_quant            ;   // is / not quant process
logic                                    cfg_trans_out_ispad_row     ;   //for transpose,output row is/not need padding 0 to align to 32
logic                                    cfg_trans_out_ispad_col     ;   //for transpose,output col is/not need padding 0 to align to 32
logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_zeropoint    ;  // zero point of SFU output data
logic    [RF_DATA_WIDTH/2-1:0]           cfg_sfu_input_zeropoint     ;  // zero point of SFU input data
logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_input_scale         ;   // scale of SFU input data 
logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_scale        ;   // scale of SFU output data 
logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_input_scale_shift   ;   // scale shift of SFU input data 
logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_scale_shift  ;   // scale shift of SFU output data 
logic    [RF_DATA_WIDTH/2-1:0]           sfu_input_sram_addr         ;   // address for input data in SPM /ACC  
logic    [RF_DATA_WIDTH/4-1:0]           sfu_input_col_num           ;   // width of input matrix data in SPM
logic    [RF_DATA_WIDTH/4-1:0]           sfu_input_row_num           ;   // height of input matrix data in SPM
logic    [RF_DATA_WIDTH/2-1:0]           sfu_output_spm_addr         ;   // address for output data in SPM
logic                                    sfu_req_en                  ;
logic                                    sfu_busy                    ;
logic                                    sfu_comp_done               ;   
//sfu spm signals
logic                                            sfu_spm_rd_en       ;   
logic    [SPM_ADDR_WIDTH-1:0]                    sfu_spm_rd_addr     ;   
logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       sfu_spm_rd_data_in  ;   

logic                                            sfu_spm_wr_en       ;   
logic    [SPM_ADDR_WIDTH-1:0]                    sfu_spm_wr_addr     ;   
logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       sfu_spm_wr_data_out ;  
logic    [PE_WIDTH-1:0]                          sfu_spm_wr_mask     ;   
//sa spm signals
logic                                            sa_spm_rd1_en       ;   
logic    [SPM_ADDR_WIDTH-1:0]                    sa_spm_rd1_addr     ;
logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       sa_spm_rd1_data_in  ;   

logic                                            sa_spm_rd2_en       ;
logic    [SPM_ADDR_WIDTH-1:0]                    sa_spm_rd2_addr     ;
logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       sa_spm_rd2_data_in  ;   

logic                                            accu_spm_wr_en      ;
logic                  [SPM_ADDR_WIDTH-1:0]      accu_spm_wr_addr    ;
logic    [PE_WIDTH-1:0]                          accu_spm_wr_mask    ;
logic    [SPM_DATA_WIDTH-1:0]                    accu_spm_wr_data    ;     
logic                                            accu_spm_wr_en_gate ;
logic                  [SPM_ADDR_WIDTH-1:0]      accu_spm_wr_addr_gate;
logic    [PE_WIDTH-1:0]                          accu_spm_wr_mask_gate;
logic    [SPM_DATA_WIDTH-1:0]                    accu_spm_wr_data_gate;
logic                                            gemv_spm_rd_en      ;
logic                  [SPM_ADDR_WIDTH-1:0]      gemv_spm_rd_addr    ;
logic                  [GEMV_SPM_DATA_WIDTH-1:0] gemv_spm_rd_data    ;
logic                                            gemv_scale_rd_en    ;
logic                  [SPM_ADDR_WIDTH-1:0]      gemv_scale_rd_addr  ;
logic                  [GEMV_SPM_DATA_WIDTH-1:0] gemv_scale_rd_data  ;
logic                                            gemv_act_rd_en      ;
logic                  [SPM_ADDR_WIDTH-1:0]      gemv_act_rd_addr    ;
logic                  [GEMV_SPM_DATA_WIDTH-1:0] gemv_act_rd_data    ;
logic                                            gemv_weight_stream_valid;
logic                  [GEMV_SPM_DATA_WIDTH-1:0] gemv_weight_stream_data;
logic                                            gemv_weight_stream_last;
logic                                            gemv_weight_stream_block_ready;
logic                                            gemv_weight_stream_ready;
logic                                            matvec_weight_stream_mode;
logic                                            matvec_act_scale_enable;
logic                                            matvec_act_scale2_enable;
logic                                            gemv_spm_wr_en      ;
logic                  [SPM_ADDR_WIDTH-1:0]      gemv_spm_wr_addr    ;
logic                  [GEMV_SPM_DATA_WIDTH-1:0] gemv_spm_wr_data    ;
logic                  [GEMV_SPM_DATA_WIDTH/8-1:0] gemv_spm_wr_mask  ;
logic                                            flow_act_wr_en       ;
logic                  [SPM_ADDR_WIDTH-1:0]      flow_act_wr_addr     ;
logic                  [GEMV_SPM_DATA_WIDTH-1:0] flow_act_wr_data     ;
logic                  [GEMV_SPM_DATA_WIDTH/8-1:0] flow_act_wr_mask   ;

assign accu_spm_wr_en_gate   = DISABLE_ACC_TO_SPM_PATH ? 1'b0 : accu_spm_wr_en;
assign accu_spm_wr_addr_gate = DISABLE_ACC_TO_SPM_PATH ? '0   : accu_spm_wr_addr;
assign accu_spm_wr_mask_gate = DISABLE_ACC_TO_SPM_PATH ? '0   : accu_spm_wr_mask;
assign accu_spm_wr_data_gate = DISABLE_ACC_TO_SPM_PATH ? '0   : accu_spm_wr_data;

//sa signals
logic                                    cfg_compute_dataflow        ;   // 0 for weight stationary, 1 for output stationary
logic    [1:0]                           cfg_compute_padding_left    ;   // Number of left padding data columns for the input
logic    [1:0]                           cfg_compute_padding_right   ;   // Number of right padding data columns for the input
logic    [1:0]                           cfg_compute_padding_top     ;   // Number of top padding data columns for the input
logic    [1:0]                           cfg_compute_padding_bottom  ;   // Number of bottom padding data columns for the input
logic    [1:0]                           cfg_compute_padding_mode    ;   // padding data type, 00 for zero padding, 01 for reflect padding,
                                                                     ;       // 10 for replicate padding, 11 for circular padding, only zero padding is supported currently    
logic    [3:0]                           cfg_compute_weight_shape    ;    // the width/height for weight, 0 represent width/height is 1
logic    [1:0]                           cfg_compute_weight_stride   ;    // the stride of weight , 0 represent 1
logic    [4:0]                           cfg_compute_weight_dilation ;   // the dilation of weight ,0 represent 1
logic                                    cfg_compute_is_groupconv    ;   //  0 for groups=1; 1 for groups = C_in(depthwise)
logic    [1:0]                           cfg_compute_int_type        ;   // SA datatype; for GEMV: 00=W4A16, 01=W8A16, 1x reserved/unsupported
logic    [1:0]                           cfg_compute_optype          ;   // operation type ,00 for GEMM,01 for CONV,10 for GEMV,11 for ADD
logic                                    cfg_compute_accout_dest     ;   // write back destination for single instruction accumulate,0:SPM,1:ACC
logic                                    cfg_compute_asymmetric_activations; // only for GEMM: 1 means subtract 128 from input A before SA
logic    [RF_DATA_WIDTH-1:0]             cfg_decode_flow             ;   // GEMV decode flow recipe fields
logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_inputa_zeropoint;   //quant zero point for input feature/input matrix A 
logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_inputb_zeropoint;   //quant zero point for input weight/input matrix B
logic    [RF_DATA_WIDTH/2-1:0]           cfg_compute_output_zeropoint;   //quant zero point when writing the accumulated output from Acc back to SPM​
logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_output_scale    ;   // scale of output data from PE in OS dataflow
logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_output_scale_shift;   // scale shift of output data from PE in OS dataflow
logic    [RF_DATA_WIDTH/2-1:0]           cfg_accu_biaspsum_addr      ;   //bias/psum data  start address in ACC
logic    [RF_DATA_WIDTH/4-1:0]           cfg_accu_biaspsum_stride    ;   //bias/psum data  stride in ACC
logic    [RF_DATA_WIDTH/8-1:0]           cfg_accu_biaspsum_width     ;   //bias/psum data width
logic    [RF_DATA_WIDTH/8-1:0]           cfg_accu_biaspsum_height    ;   //bias/psum data height
logic    [RF_DATA_WIDTH/2-1:0]           cfg_accu_output_addr        ;   // output data in ACC/SPM start address
logic    [RF_DATA_WIDTH/4-1:0]           cfg_accu_output_stride      ;   // output data in ACC/SPM stride
logic                                    cfg_accu_isaccu             ;   // 0:the sa result direct save to ACC/SPM;1:save to ACC/SPM after accumulate 
logic                                    cfg_accu_relu               ;   // 0 for no relu,1 for relu
logic    [2:0]                           cfg_accu_relu_type          ;   // 000 for relu, 001 for relu6, 010 for leaky relu a=0.1
                                                                         // 011 for leaky relu a=0.2 ,100 for leaky relu a=0.01
logic                                    cfg_accu_isbias             ;   // 0:part sum, 1:bias
logic    [RF_DATA_WIDTH/2-1:0]           sa_input_a_spm_addr         ;   // address for input feature(WS)/input matrix A(OS) data in SPM   
logic    [$clog2(SA_MAX_LENGTH)-1:0]     sa_input_a_col_num          ;   // column number of input feature(WS)/input matrix A(OS) in systolic array
logic    [$clog2(ARRAY_WIDTH)-1:0]       sa_input_a_row_num          ;   // row number of input feature(WS)/input matrix A(OS) in systolic array
logic    [RF_DATA_WIDTH/4-1:0]           sa_input_a_stride           ;   //stride for input feature(WS)/input matrix A(OS) data, for GEMM not valid(default W)
                                                                         //for dilated conv(not support now),default is W-1
logic    [RF_DATA_WIDTH/2-1:0]           sa_input_b_spm_addr         ;   // address for input matrix B(OS) data in SPM
logic    [$clog2(ARRAY_WIDTH)-1:0]       sa_input_b_col_num          ;   // column number of input matrix B(OS) in systolic array
logic    [$clog2(SA_MAX_LENGTH)-1:0]     sa_input_b_row_num          ;   // row number of input matrix B(OS) in systolic array
logic    [RF_DATA_WIDTH/4-1:0]           sa_input_b_stride           ;   // same as a_stride,when unit matrix is B:stride=32,for conv:invalid
logic                                    sa_req_en                   ;
logic                                    systolic_array_busy         ;
logic                                    systolic_array_done         ;
logic                                    sa_acc_valid                ;
logic   [ARRAY_WIDTH-1:0]                sa_acc_mask                 ;
logic   [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_OUT-1:0]     sa_acc_data_out ;
logic                                    sa_acc_start                ;
logic   [RF_DATA_WIDTH/8-1:0]            sa_acc_row_num              ;
logic   [RF_DATA_WIDTH/8-1:0]            sa_acc_ofm_row_num          ;
logic   [RF_DATA_WIDTH/8-1:0]            sa_acc_ofm_col_num          ;
logic                                    sa_comp_done                ;
logic                                    sa_busy                     ;
logic                                    accadd_busy                 ;
//matadd signals
logic   [RF_DATA_WIDTH/2-1:0]           matadd_input_a_addr          ; //input matrix A start address in ACC
logic   [RF_DATA_WIDTH/2-1:0]           matadd_input_b_addr          ; //input matrix B start address in ACC
logic   [RF_DATA_WIDTH/8-1:0]           matadd_input_col_num         ; //input matrix A/B column number
logic   [RF_DATA_WIDTH/8-1:0]           matadd_input_row_num         ; //input matrix A/B row number
logic   [RF_DATA_WIDTH/2-1:0]           matadd_output_addr           ; //output matrix start address in SPM
logic                                   matadd_req_en                ;
logic                                   matadd_busy                  ;
logic                                   matadd_comp_done             ;    
//matvec signals
logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_mat_addr        ; //input matrix  start address in SPM
logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_vec_addr        ; //input vector  start address in SPM
logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_act_scale_addr  ; //input act scale start address in SPM
logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_act_scale2_addr ; //second input act scale start address in SPM
logic   [RF_DATA_WIDTH/4-1:0]           matvec_cache_cell_idx        ; //KV cache window cell index
logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_act_group_stride_bytes; //activation group stride bytes
logic   [7:0]                           matvec_act_frac_cfg         ; //GEMV activation frac cfg
logic   [RF_DATA_WIDTH/4-1:0]           matvec_input_mat_width       ; //input matrix width
logic   [RF_DATA_WIDTH/4-1:0]           matvec_input_mat_height      ; //input matrix height
logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_mat_stride      ; //GEMV output byte address
logic   [RF_DATA_WIDTH/4-1:0]           matvec_input_vec_stride      ; //input vector stride
logic                                   matvec_req_en                ;
logic                                   matvec_busy                  ;
logic                                   matvec_comp_done             ;
logic                                   kv_scale_commit_valid        ;
logic                                   kv_scale_commit_is_v         ;
logic   [4:0][15:0]                     kv_scale_commit_values       ;
logic   [7:0]                           kv_scale_commit_count        ;
always_comb begin 
    sa_busy=systolic_array_busy || accadd_busy;
end

assign busy = mvin4_busy || mvout1_busy || sa_busy || matadd_busy || matvec_busy || sfu_busy;

always_ff @(posedge clk or negedge rst_n) begin : mvin4_req_collect
    if (!rst_n) begin
        mvin4_dram_base_addr_cfg <= '0;
        mvin4_spm_addr_cfg  <= '0;
        mvin4_line_num_cfg  <= '0;
        mvin4_stream_fifo_fill_cfg <= 1'b0;
        mvin4_req_en_cfg    <= 1'b0;
        mvin4_scale_req_en_cfg <= 1'b0;
        mvin4_act_req_en_cfg <= 1'b0;
        mvout1_dram_base_addr_cfg <= '0;
        mvout1_spm_addr_cfg <= '0;
        mvout1_row_num_cfg <= '0;
        mvout1_output_precision_cfg <= 2'b01;
        mvout1_req_en_cfg <= 1'b0;
        mvout1_cfg_pending_q <= 1'b0;
        mvout1_elem_count_q <= '0;
        dma_mvin_req_en_dly <= '0;
        dma_mvout_req_en_dly <= '0;
    end else begin
        mvin4_req_en_cfg <= 1'b0;
        mvin4_stream_fifo_fill_cfg <= 1'b0;
        mvin4_scale_req_en_cfg <= 1'b0;
        mvin4_act_req_en_cfg <= 1'b0;
        mvout1_req_en_cfg <= 1'b0;
        dma_mvin_req_en_dly <= dma_mvin_req_en;
        dma_mvout_req_en_dly <= dma_mvout_req_en;

        if (dma_mvin_req_en_dly[0] && !mvin4_busy) begin
            mvin4_dram_base_addr_cfg <= dma_mvin_dram_addr[0][AXI_ADDR_WIDTH-1:0];
            mvin4_spm_addr_cfg <= dma_mvin_sram_addr[0][MVIN_4AXI_SPM_ADDR_LSB +: $clog2(MVIN_4AXI_SPM_DEPTH)];
            mvin4_line_num_cfg <= (dma_mvin_col_num[0] + 32'd1) / MVIN_4AXI_LINE_BYTES;
            mvin4_stream_fifo_fill_cfg <= dma_cfg_mvin_stream_fifo_fill[0];
            mvin4_req_en_cfg <= (dma_cfg_mvin_input_type[0] != 2'b11) &&
                                 (dma_cfg_mvin_input_type[0] != 2'b10);
            mvin4_scale_req_en_cfg <= dma_cfg_mvin_scale_target[0];
            mvin4_act_req_en_cfg <= dma_cfg_mvin_input_type[0] == 2'b11;
`ifndef SYNTHESIS
            if (dma_mvin_row_num[0] != '0) begin
                $fatal(1, "4AXI DMA expects mvin_row_num == 0 and mvin_col_num to encode total byte count - 1");
            end
            if (((dma_mvin_col_num[0] + 32'd1) % (MVIN_4AXI_LINE_BYTES * npu_config_pkg::SPM_BANK_NUM)) != 0) begin
                $fatal(
                    1,
                    "4AXI DMA expects total byte count to be a multiple of %0d",
                    (MVIN_4AXI_LINE_BYTES * npu_config_pkg::SPM_BANK_NUM)
                );
            end
            if (dma_cfg_mvin_dest[0] != 1'b0) begin
                $fatal(1, "4AXI DMA currently supports mvin to SPM only");
            end
`endif
        end

        if (mvout1_cfg_pending_q && !mvout1_busy) begin
            mvout1_row_num_cfg <= (mvout1_elem_count_q +
                                   ((mvout1_output_precision_cfg == GEMV_MVOUT_PREC_RAW8) ?
                                    MVIN_4AXI_LINE_BYTES :
                                    (MVIN_4AXI_LINE_BYTES / GEMV_MVOUT_ELEM_BYTES)) - 1)
                                  / ((mvout1_output_precision_cfg == GEMV_MVOUT_PREC_RAW8) ?
                                     MVIN_4AXI_LINE_BYTES :
                                     (MVIN_4AXI_LINE_BYTES / GEMV_MVOUT_ELEM_BYTES));
            mvout1_req_en_cfg <= 1'b1;
            mvout1_cfg_pending_q <= 1'b0;
        end

        if (dma_mvout_req_en_dly[ACC_DMA_IDX] && !mvout1_busy && !mvout1_cfg_pending_q) begin
            mvout1_dram_base_addr_cfg <= dma_mvout_dram_addr[ACC_DMA_IDX][AXI_ADDR_WIDTH-1:0];
            mvout1_spm_addr_cfg <= dma_mvout_sram_addr[ACC_DMA_IDX][MVIN_4AXI_SPM_ADDR_LSB +: $clog2(MVIN_4AXI_OUTPUT_SPM_DEPTH)];
            mvout1_output_precision_cfg <= dma_cfg_mvout_output_precision[ACC_DMA_IDX];
            mvout1_elem_count_q <= (dma_mvout_row_num[ACC_DMA_IDX] + 32'd1) *
                                   (dma_mvout_col_num[ACC_DMA_IDX] + 32'd1);
            mvout1_cfg_pending_q <= 1'b1;
`ifndef SYNTHESIS
            if (dma_cfg_mvout_source[ACC_DMA_IDX] != 1'b0) begin
                $fatal(1, "4AXI DMA expects GEMV mvout request on DMA%0d to source from output SPM", ACC_DMA_IDX);
            end
            if ((dma_mvout_sram_addr[ACC_DMA_IDX] +
                 (((dma_mvout_row_num[ACC_DMA_IDX] + 32'd1) *
                   (dma_mvout_col_num[ACC_DMA_IDX] + 32'd1)) *
                  ((dma_cfg_mvout_output_precision[ACC_DMA_IDX] == GEMV_MVOUT_PREC_RAW8) ?
                   32'd1 : GEMV_MVOUT_ELEM_BYTES))) >
                MVIN_4AXI_OUTPUT_SPM_BYTES) begin
                $fatal(1, "4AXI DMA GEMV mvout exceeds output SPM capacity");
            end
            if (((dma_cfg_mvout_output_precision[ACC_DMA_IDX] != GEMV_MVOUT_PREC_RAW8) &&
                 (dma_cfg_mvout_output_precision[ACC_DMA_IDX] != 2'b01) &&
                 (dma_cfg_mvout_output_precision[ACC_DMA_IDX] != 2'b11)) ||
                (dma_cfg_mvout_is_quant[ACC_DMA_IDX] &&
                 (dma_cfg_mvout_output_precision[ACC_DMA_IDX] != GEMV_MVOUT_PREC_RAW8)) ||
                dma_cfg_mvout_per_channel[ACC_DMA_IDX]) begin
                $fatal(1, "4AXI DMA GEMV mvout supports raw int8 or raw fp16/fp32 output only");
            end
`endif
        end
    end
end

always_comb begin : dma_status_mux
    dma_mvin_busy_status          = '0;
    dma_mvout_busy_status         = '0;
    dma_mvin_resp_done_status     = '0;
    dma_mvout_resp_done_status    = '0;

    dma_mvin_busy_status[0]            = mvin4_busy;
    dma_mvin_resp_done_status[0]       = mvin4_resp_done;
    dma_mvout_busy_status[ACC_DMA_IDX]      = mvout1_busy;
    dma_mvout_resp_done_status[ACC_DMA_IDX] = mvout1_resp_done;
end

assign dma_acc_wr_en              = 1'b0;
assign dma_acc_wr_addr            = '0;
assign dma_acc_wr_mask            = '0;
assign dma_acc_din                = '0;
assign dma_acc_rd_en              = mvin4_acc_rd_en;
assign dma_acc_rd_addr            = mvin4_acc_rd_addr;
assign sa_spm_rd1_data_in         = '0;
assign sa_spm_rd2_data_in         = '0;
assign sfu_spm_rd_data_in         = '0;

always_comb begin : dma_axi_direct_connect
    m_axi0_awid    = mvin4_axi_awid[0];
    m_axi0_awaddr  = mvin4_axi_awaddr[0];
    m_axi0_awlen   = mvin4_axi_awlen[0];
    m_axi0_awsize  = mvin4_axi_awsize[0];
    m_axi0_awburst = mvin4_axi_awburst[0];
    m_axi0_awvalid = mvin4_axi_awvalid[0];
    m_axi0_wdata   = mvin4_axi_wdata[0];
    m_axi0_wstrb   = mvin4_axi_wstrb[0];
    m_axi0_wlast   = mvin4_axi_wlast[0];
    m_axi0_wvalid  = mvin4_axi_wvalid[0];
    m_axi0_bready  = mvin4_axi_bready[0];
    m_axi0_arid    = mvin4_axi_arid[0];
    m_axi0_araddr  = mvin4_axi_araddr[0];
    m_axi0_arlen   = mvin4_axi_arlen[0];
    m_axi0_arsize  = mvin4_axi_arsize[0];
    m_axi0_arburst = mvin4_axi_arburst[0];
    m_axi0_arvalid = mvin4_axi_arvalid[0];
    m_axi0_rready  = mvin4_axi_rready[0];

    m_axi1_awid    = mvin4_axi_awid[1];
    m_axi1_awaddr  = mvin4_axi_awaddr[1];
    m_axi1_awlen   = mvin4_axi_awlen[1];
    m_axi1_awsize  = mvin4_axi_awsize[1];
    m_axi1_awburst = mvin4_axi_awburst[1];
    m_axi1_awvalid = mvin4_axi_awvalid[1];
    m_axi1_wdata   = mvin4_axi_wdata[1];
    m_axi1_wstrb   = mvin4_axi_wstrb[1];
    m_axi1_wlast   = mvin4_axi_wlast[1];
    m_axi1_wvalid  = mvin4_axi_wvalid[1];
    m_axi1_bready  = mvin4_axi_bready[1];
    m_axi1_arid    = mvin4_axi_arid[1];
    m_axi1_araddr  = mvin4_axi_araddr[1];
    m_axi1_arlen   = mvin4_axi_arlen[1];
    m_axi1_arsize  = mvin4_axi_arsize[1];
    m_axi1_arburst = mvin4_axi_arburst[1];
    m_axi1_arvalid = mvin4_axi_arvalid[1];
    m_axi1_rready  = mvin4_axi_rready[1];

    m_axi2_awid    = mvin4_axi_awid[2];
    m_axi2_awaddr  = mvin4_axi_awaddr[2];
    m_axi2_awlen   = mvin4_axi_awlen[2];
    m_axi2_awsize  = mvin4_axi_awsize[2];
    m_axi2_awburst = mvin4_axi_awburst[2];
    m_axi2_awvalid = mvin4_axi_awvalid[2];
    m_axi2_wdata   = mvin4_axi_wdata[2];
    m_axi2_wstrb   = mvin4_axi_wstrb[2];
    m_axi2_wlast   = mvin4_axi_wlast[2];
    m_axi2_wvalid  = mvin4_axi_wvalid[2];
    m_axi2_bready  = mvin4_axi_bready[2];
    m_axi2_arid    = mvin4_axi_arid[2];
    m_axi2_araddr  = mvin4_axi_araddr[2];
    m_axi2_arlen   = mvin4_axi_arlen[2];
    m_axi2_arsize  = mvin4_axi_arsize[2];
    m_axi2_arburst = mvin4_axi_arburst[2];
    m_axi2_arvalid = mvin4_axi_arvalid[2];
    m_axi2_rready  = mvin4_axi_rready[2];

    m_axi3_awid    = mvin4_axi_awid[3];
    m_axi3_awaddr  = mvin4_axi_awaddr[3];
    m_axi3_awlen   = mvin4_axi_awlen[3];
    m_axi3_awsize  = mvin4_axi_awsize[3];
    m_axi3_awburst = mvin4_axi_awburst[3];
    m_axi3_awvalid = mvin4_axi_awvalid[3];
    m_axi3_wdata   = mvin4_axi_wdata[3];
    m_axi3_wstrb   = mvin4_axi_wstrb[3];
    m_axi3_wlast   = mvin4_axi_wlast[3];
    m_axi3_wvalid  = mvin4_axi_wvalid[3];
    m_axi3_bready  = mvin4_axi_bready[3];
    m_axi3_arid    = mvin4_axi_arid[3];
    m_axi3_araddr  = mvin4_axi_araddr[3];
    m_axi3_arlen   = mvin4_axi_arlen[3];
    m_axi3_arsize  = mvin4_axi_arsize[3];
    m_axi3_arburst = mvin4_axi_arburst[3];
    m_axi3_arvalid = mvin4_axi_arvalid[3];
    m_axi3_rready  = mvin4_axi_rready[3];

    mvin4_axi_awready = '0;
    mvin4_axi_wready  = '0;
    mvin4_axi_bid     = '0;
    mvin4_axi_bresp   = '0;
    mvin4_axi_bvalid  = '0;
    mvin4_axi_arready = '0;
    mvin4_axi_rid     = '0;
    mvin4_axi_rdata   = '0;
    mvin4_axi_rresp   = '0;
    mvin4_axi_rlast   = '0;
    mvin4_axi_rvalid  = '0;

    mvin4_axi_awready[0] = m_axi0_awready;
    mvin4_axi_wready[0]  = m_axi0_wready;
    mvin4_axi_bid[0]     = m_axi0_bid;
    mvin4_axi_bresp[0]   = m_axi0_bresp;
    mvin4_axi_bvalid[0]  = m_axi0_bvalid;
    mvin4_axi_arready[0] = m_axi0_arready;
    mvin4_axi_rid[0]     = m_axi0_rid;
    mvin4_axi_rdata[0]   = m_axi0_rdata;
    mvin4_axi_rresp[0]   = m_axi0_rresp;
    mvin4_axi_rlast[0]   = m_axi0_rlast;
    mvin4_axi_rvalid[0]  = m_axi0_rvalid;

    mvin4_axi_awready[1] = m_axi1_awready;
    mvin4_axi_wready[1]  = m_axi1_wready;
    mvin4_axi_bid[1]     = m_axi1_bid;
    mvin4_axi_bresp[1]   = m_axi1_bresp;
    mvin4_axi_bvalid[1]  = m_axi1_bvalid;
    mvin4_axi_arready[1] = m_axi1_arready;
    mvin4_axi_rid[1]     = m_axi1_rid;
    mvin4_axi_rdata[1]   = m_axi1_rdata;
    mvin4_axi_rresp[1]   = m_axi1_rresp;
    mvin4_axi_rlast[1]   = m_axi1_rlast;
    mvin4_axi_rvalid[1]  = m_axi1_rvalid;

    mvin4_axi_awready[2] = m_axi2_awready;
    mvin4_axi_wready[2]  = m_axi2_wready;
    mvin4_axi_bid[2]     = m_axi2_bid;
    mvin4_axi_bresp[2]   = m_axi2_bresp;
    mvin4_axi_bvalid[2]  = m_axi2_bvalid;
    mvin4_axi_arready[2] = m_axi2_arready;
    mvin4_axi_rid[2]     = m_axi2_rid;
    mvin4_axi_rdata[2]   = m_axi2_rdata;
    mvin4_axi_rresp[2]   = m_axi2_rresp;
    mvin4_axi_rlast[2]   = m_axi2_rlast;
    mvin4_axi_rvalid[2]  = m_axi2_rvalid;

    mvin4_axi_awready[3] = m_axi3_awready;
    mvin4_axi_wready[3]  = m_axi3_wready;
    mvin4_axi_bid[3]     = m_axi3_bid;
    mvin4_axi_bresp[3]   = m_axi3_bresp;
    mvin4_axi_bvalid[3]  = m_axi3_bvalid;
    mvin4_axi_arready[3] = m_axi3_arready;
    mvin4_axi_rid[3]     = m_axi3_rid;
    mvin4_axi_rdata[3]   = m_axi3_rdata;
    mvin4_axi_rresp[3]   = m_axi3_rresp;
    mvin4_axi_rlast[3]   = m_axi3_rlast;
    mvin4_axi_rvalid[3]  = m_axi3_rvalid;
end

T_NPU_MVIN_4AXI #(
    .AXI_PORTS           (MVIN_4AXI_PORTS),
    .AXI_ID_WIDTH        (AXI_ID_WIDTH),
    .AXI_ADDR_WIDTH      (AXI_ADDR_WIDTH),
    .AXI_DATA_WIDTH      (AXI_DATA_WIDTH),
    .AXI_MAX_BURST_BEATS (AXI_MAX_BURST_BEATS),
    .AXI_MAX_OUTSTANDING (AXI_MAX_OUTSTANDING),
    .SPM_FPGA_SRAM       (SPM_FPGA_SRAM),
    .SPM_BANKS           (npu_config_pkg::SPM_BANK_NUM),
    .SPM_BANK_DATA_WIDTH (npu_config_pkg::SPM_BANK_DATA_WIDTH),
    .SPM_DEPTH           (MVIN_4AXI_SPM_DEPTH),
    .SCALE_SPM_DEPTH     (MVIN_4AXI_SCALE_SPM_DEPTH),
    .OUTPUT_SPM_DEPTH    (MVIN_4AXI_OUTPUT_SPM_DEPTH)
) u_mvin_4axi_top (
    .clk                (clk),
    .rst_n              (rst_n),
    .mvin_dram_base_addr(mvin4_dram_base_addr_cfg),
    .mvin_spm_addr      (mvin4_spm_addr_cfg),
    .mvin_line_num      (mvin4_line_num_cfg),
    .mvin_stream_fifo_fill(mvin4_stream_fifo_fill_cfg),
    .dma_mvin_req_en    (mvin4_req_en_cfg),
    .dma_scale_mvin_req_en(mvin4_scale_req_en_cfg),
    .dma_act_mvin_req_en(mvin4_act_req_en_cfg),
    .dma_mvin_resp_done (mvin4_resp_done),
    .dma_mvin_busy      (mvin4_busy),
    .mvout_dram_base_addr(mvout1_dram_base_addr_cfg),
    .mvout_spm_addr     (mvout1_spm_addr_cfg),
    .mvout_row_num      (mvout1_row_num_cfg),
    .mvout_output_precision(mvout1_output_precision_cfg),
    .dma_mvout_req_en   (mvout1_req_en_cfg),
    .dma_mvout_resp_done(mvout1_resp_done),
    .dma_mvout_busy     (mvout1_busy),
    .spm_beat_rd_en     (1'b0),
    .spm_beat_rd_addr   ('0),
    .spm_beat_rd_data   (mvin4_spm_beat_rd_data),
    .gemv_spm_line_rd_en(gemv_spm_rd_en),
    .gemv_spm_line_rd_addr(gemv_spm_rd_addr),
    .gemv_spm_line_rd_data(gemv_spm_rd_data),
    .gemv_weight_stream_enable(matvec_weight_stream_mode),
    .gemv_weight_stream_valid(gemv_weight_stream_valid),
    .gemv_weight_stream_data(gemv_weight_stream_data),
    .gemv_weight_stream_last(gemv_weight_stream_last),
    .gemv_weight_stream_block_ready(gemv_weight_stream_block_ready),
    .gemv_weight_stream_ready(gemv_weight_stream_ready),
    .gemv_scale_line_rd_en(gemv_scale_rd_en),
    .gemv_scale_line_rd_addr(gemv_scale_rd_addr),
    .gemv_scale_line_rd_data(gemv_scale_rd_data),
    .gemv_act_line_rd_en(gemv_act_rd_en),
    .gemv_act_line_rd_addr(gemv_act_rd_addr),
    .gemv_act_line_rd_data(gemv_act_rd_data),
    .gemv_spm_line_wr_en(gemv_spm_wr_en),
    .gemv_spm_line_wr_addr(gemv_spm_wr_addr),
    .gemv_spm_line_wr_data(gemv_spm_wr_data),
    .gemv_spm_line_wr_mask(gemv_spm_wr_mask),
    .flow_act_line_wr_en(flow_act_wr_en),
    .flow_act_line_wr_addr(flow_act_wr_addr),
    .flow_act_line_wr_data(flow_act_wr_data),
    .flow_act_line_wr_mask(flow_act_wr_mask),
    .acc_rd_en          (mvin4_acc_rd_en),
    .acc_rd_addr        (mvin4_acc_rd_addr),
    .acc_rd_data        (dma_acc_dout),
    .acc_rd_valid       (dma_acc_rd_valid),
    .m_axi_awid         (mvin4_axi_awid),
    .m_axi_awaddr       (mvin4_axi_awaddr),
    .m_axi_awlen        (mvin4_axi_awlen),
    .m_axi_awsize       (mvin4_axi_awsize),
    .m_axi_awburst      (mvin4_axi_awburst),
    .m_axi_awvalid      (mvin4_axi_awvalid),
    .m_axi_awready      (mvin4_axi_awready),
    .m_axi_wdata        (mvin4_axi_wdata),
    .m_axi_wstrb        (mvin4_axi_wstrb),
    .m_axi_wlast        (mvin4_axi_wlast),
    .m_axi_wvalid       (mvin4_axi_wvalid),
    .m_axi_wready       (mvin4_axi_wready),
    .m_axi_bid          (mvin4_axi_bid),
    .m_axi_bresp        (mvin4_axi_bresp),
    .m_axi_bvalid       (mvin4_axi_bvalid),
    .m_axi_bready       (mvin4_axi_bready),
    .m_axi_arid         (mvin4_axi_arid),
    .m_axi_araddr       (mvin4_axi_araddr),
    .m_axi_arlen        (mvin4_axi_arlen),
    .m_axi_arsize       (mvin4_axi_arsize),
    .m_axi_arburst      (mvin4_axi_arburst),
    .m_axi_arvalid      (mvin4_axi_arvalid),
    .m_axi_arready      (mvin4_axi_arready),
    .m_axi_rid          (mvin4_axi_rid),
    .m_axi_rdata        (mvin4_axi_rdata),
    .m_axi_rresp        (mvin4_axi_rresp),
    .m_axi_rlast        (mvin4_axi_rlast),
    .m_axi_rvalid       (mvin4_axi_rvalid),
    .m_axi_rready       (mvin4_axi_rready)
);

inst_ctrl #(
        .RF_DATA_WIDTH       ( RF_DATA_WIDTH       ),
        .MMIO_AXI_DATA_WIDTH ( MMIO_AXI_DATA_WIDTH ),
        .MMIO_AXI_ADDR_WIDTH ( MMIO_AXI_ADDR_WIDTH ),
        .DMA_NUM             ( 1                   ),
        .INPUT_WIDTH_MAX     ( SA_MAX_LENGTH       ),
        .INPUT_HEIGHT_MAX    ( SA_MAX_LENGTH       ),
        .ARRAY_WIDTH         ( ARRAY_WIDTH         ),
        .ARRAY_HEIGHT        ( ARRAY_WIDTH         ),
        .ENABLE_PROFILE_COUNTERS( ENABLE_PROFILE_COUNTERS ),
        .ENABLE_DETAILED_ERROR_STATUS( ENABLE_DETAILED_ERROR_STATUS )
) inst_ctrl_u0 (
        .clk                             (clk),
        .rst_n                           (rst_n), 
        // ===== AXI-Lite =====
        .s_axi_awaddr                   (s_axi_awaddr       ),
        .s_axi_awprot                   (s_axi_awprot       ),
        .s_axi_awvalid                  (s_axi_awvalid      ),
        .s_axi_awready                  (s_axi_awready      ),
     
        .s_axi_wdata                    (s_axi_wdata        ),
        .s_axi_wstrb                    (s_axi_wstrb        ),
        .s_axi_wvalid                   (s_axi_wvalid       ),
        .s_axi_wready                   (s_axi_wready       ),
       
        .s_axi_bresp                    (s_axi_bresp        ),
        .s_axi_bvalid                   (s_axi_bvalid       ),
        .s_axi_bready                   (s_axi_bready       ),
 
        .s_axi_araddr                   (s_axi_araddr       ),
        .s_axi_arprot                   (s_axi_arprot       ),
        .s_axi_arvalid                  (s_axi_arvalid      ),
        .s_axi_arready                  (s_axi_arready      ),
   
        .s_axi_rdata                    (s_axi_rdata        ),
        .s_axi_rresp                    (s_axi_rresp        ),
        .s_axi_rvalid                   (s_axi_rvalid       ),
        .s_axi_rready                   (s_axi_rready       ),
        //dma
        // .dma_mvin_busy                  (1'b0                    ),
        // .dma_mvout_busy                 (1'b0                    ),
        // .dma_mvin_resp_done             (1'b0                    ),
        // .dma_mvout_resp_done            (1'b0                    ),
        .cfg_mvin_isbias                (cfg_mvin_isbias            ),             
        .dma_mvin_dram_addr             (dma_mvin_dram_addr         ),
        .dma_mvin_row_num               (dma_mvin_row_num           ),
        .dma_mvin_sram_addr             (dma_mvin_sram_addr         ),
        .dma_mvin_col_num               (dma_mvin_col_num           ),
        .dma_mvout_dram_addr            (dma_mvout_dram_addr        ),
        .dma_mvout_row_num              (dma_mvout_row_num          ),
        .dma_mvout_sram_addr            (dma_mvout_sram_addr        ),
        .dma_mvout_col_num              (dma_mvout_col_num          ),
        .dma_cfg_mvin_input_type        (dma_cfg_mvin_input_type    ),
        .dma_cfg_mvout_output_type      (dma_cfg_mvout_output_type  ),
        .dma_cfg_mvin_input_precision   (dma_cfg_mvin_input_precision),
        .dma_cfg_mvout_output_precision (dma_cfg_mvout_output_precision),
        .dma_cfg_mvin_is_quant          (dma_cfg_mvin_is_quant      ),
        .dma_cfg_mvout_is_quant         (dma_cfg_mvout_is_quant     ),
        .dma_cfg_mvin_dest              (dma_cfg_mvin_dest          ),
        .dma_cfg_mvout_source           (dma_cfg_mvout_source       ),
        .dma_cfg_mvout_per_channel      (dma_cfg_mvout_per_channel  ),
        .cfg_mvout_output_precision     (cfg_mvout_output_precision ),
        .cfg_mvout_per_channel          (cfg_mvout_per_channel      ),
        .cfg_mvout_output_scale         (cfg_mvout_output_scale     ),
        .cfg_mvout_output_scale_shift   (cfg_mvout_output_scale_shift),
        .dma_cfg_mvin_isbias            (dma_cfg_mvin_isbias        ),
        .dma_cfg_mvin_scale_target      (dma_cfg_mvin_scale_target  ),
        .dma_cfg_mvin_stream_fifo_fill  (dma_cfg_mvin_stream_fifo_fill),
        .dma_cfg_mvin_sram_stride       (dma_cfg_mvin_sram_stride   ),
        .dma_cfg_mvin_dram_stride       (dma_cfg_mvin_dram_stride   ),
        .dma_cfg_mvout_sram_stride      (dma_cfg_mvout_sram_stride  ),
        .dma_cfg_mvout_dram_stride      (dma_cfg_mvout_dram_stride  ),
        .dma_cfg_mvin_input_zeropoint   (dma_cfg_mvin_input_zeropoint),
        .dma_cfg_mvout_output_zeropoint (dma_cfg_mvout_output_zeropoint),
        .dma_cfg_mvin_input_scale       (dma_cfg_mvin_input_scale   ),
        .dma_cfg_mvout_output_scale     (dma_cfg_mvout_output_scale ),
        .dma_cfg_mvin_input_scale_shift (dma_cfg_mvin_input_scale_shift),
        .dma_cfg_mvout_output_scale_shift(dma_cfg_mvout_output_scale_shift),
        .matvec_weight_stream_mode      (matvec_weight_stream_mode),
        .matvec_act_scale_enable        (matvec_act_scale_enable),
        .matvec_act_scale2_enable       (matvec_act_scale2_enable),
        .dma_mvin_req_en            (dma_mvin_req_en            ),
        .dma_mvout_req_en           (dma_mvout_req_en           ),
        .dma_mvin_busy_status              (dma_mvin_busy_status       ),
        .dma_mvout_busy_status             (dma_mvout_busy_status      ),
        .dma_mvin_resp_done_status         (dma_mvin_resp_done_status  ),
        .dma_mvout_resp_done_status        (dma_mvout_resp_done_status ),
        //sa
        .cfg_compute_dataflow          (cfg_compute_dataflow        ),          
        .cfg_compute_padding_left      (cfg_compute_padding_left    ),      
        .cfg_compute_padding_right     (cfg_compute_padding_right   ),     
        .cfg_compute_padding_top       (cfg_compute_padding_top     ),       
        .cfg_compute_padding_bottom    (cfg_compute_padding_bottom  ),    
        .cfg_compute_padding_mode      (cfg_compute_padding_mode    ),      
        .cfg_compute_weight_shape      (cfg_compute_weight_shape    ),      
        .cfg_compute_weight_stride     (cfg_compute_weight_stride   ),     
        .cfg_compute_weight_dilation   (cfg_compute_weight_dilation ),   
        .cfg_compute_is_groupconv      (cfg_compute_is_groupconv    ),      
        .cfg_compute_int_type          (cfg_compute_int_type        ),          
        .cfg_compute_optype            (cfg_compute_optype          ),            
        .cfg_compute_accout_dest       (cfg_compute_accout_dest     ),       
        .cfg_compute_asymmetric_activations(cfg_compute_asymmetric_activations),
        .cfg_decode_flow               (cfg_decode_flow             ),
        .cfg_compute_inputa_zeropoint  (cfg_compute_inputa_zeropoint),  
        .cfg_compute_inputb_zeropoint  (cfg_compute_inputb_zeropoint),  
        .cfg_compute_output_zeropoint  (cfg_compute_output_zeropoint),  
        .cfg_compute_output_scale      (cfg_compute_output_scale     ),      
        .cfg_compute_output_scale_shift(cfg_compute_output_scale_shift),
        //acc
        .cfg_accu_biaspsum_addr        (cfg_accu_biaspsum_addr      ),       
        .cfg_accu_biaspsum_stride      (cfg_accu_biaspsum_stride    ),     
        .cfg_accu_biaspsum_width       (cfg_accu_biaspsum_width     ),      
        .cfg_accu_biaspsum_height      (cfg_accu_biaspsum_height    ),     
        .cfg_accu_output_addr          (cfg_accu_output_addr        ),         
        .cfg_accu_output_stride        (cfg_accu_output_stride      ),
        .cfg_accu_isaccu               (cfg_accu_isaccu             ),       
        .cfg_accu_relu                 (cfg_accu_relu               ),
        .cfg_accu_relu_type            (cfg_accu_relu_type          ),                
        .cfg_accu_isbias               (cfg_accu_isbias             ),              
        //sa
        .sa_input_a_spm_addr            (sa_input_a_spm_addr        ),  
        .sa_input_a_col_num             (sa_input_a_col_num         ),   
        .sa_input_a_row_num             (sa_input_a_row_num         ),   
        .sa_input_a_stride              (sa_input_a_stride          ),    
        .sa_input_b_spm_addr            (sa_input_b_spm_addr        ),  
        .sa_input_b_col_num             (sa_input_b_col_num         ),   
        .sa_input_b_row_num             (sa_input_b_row_num         ),   
        .sa_input_b_stride              (sa_input_b_stride          ),    
        .sa_req_en                      (sa_req_en                  ),            
        .sa_busy                        (sa_busy                    ),              
        .sa_comp_done                   (sa_comp_done               ),         
        //matadd
        .matadd_input_a_addr            (matadd_input_a_addr        ),   
        .matadd_input_b_addr            (matadd_input_b_addr        ),   
        .matadd_input_col_num           (matadd_input_col_num       ),  
        .matadd_input_row_num           (matadd_input_row_num       ),  
        .matadd_output_addr             (matadd_output_addr         ), 
        .matadd_req_en                  (matadd_req_en              ),         
        .matadd_busy                    (matadd_busy                ),           
        .matadd_comp_done               (matadd_comp_done           ),      
        //matvec
        .matvec_input_mat_addr          (matvec_input_mat_addr      ), 
        .matvec_input_vec_addr          (matvec_input_vec_addr      ), 
        .matvec_input_act_scale_addr    (matvec_input_act_scale_addr),
        .matvec_input_act_scale2_addr   (matvec_input_act_scale2_addr),
        .matvec_cache_cell_idx          (matvec_cache_cell_idx      ),
        .matvec_input_act_group_stride_bytes(matvec_input_act_group_stride_bytes),
        .matvec_act_frac_cfg            (matvec_act_frac_cfg        ),
        .matvec_input_mat_width         (matvec_input_mat_width     ),
        .matvec_input_mat_height        (matvec_input_mat_height    ),
        .matvec_input_mat_stride        (matvec_input_mat_stride    ),
        .matvec_input_vec_stride        (matvec_input_vec_stride    ),
        .matvec_req_en                  (matvec_req_en              ),        
        .matvec_busy                    (matvec_busy                ),          
        .matvec_comp_done               (matvec_comp_done           ),     
        .kv_scale_commit_valid          (kv_scale_commit_valid      ),
        .kv_scale_commit_is_v           (kv_scale_commit_is_v       ),
        .kv_scale_commit_values         (kv_scale_commit_values     ),
        .kv_scale_commit_count          (kv_scale_commit_count      ),
        //sfu
        .cfg_sfu_op                     (cfg_sfu_op                 ),            
        .cfg_sfu_int_type               (cfg_sfu_int_type           ),      
        .cfg_sfu_is_quant               (cfg_sfu_is_quant           ),      
        .cfg_trans_out_ispad_row        (cfg_trans_out_ispad_row    ),
        .cfg_trans_out_ispad_col        (cfg_trans_out_ispad_col    ),
        .cfg_sfu_output_zeropoint       (cfg_sfu_output_zeropoint   ),
        .cfg_sfu_input_zeropoint        (cfg_sfu_input_zeropoint    ),  
        .cfg_sfu_input_scale            (cfg_sfu_input_scale        ),   
        .cfg_sfu_output_scale           (cfg_sfu_output_scale       ),  
        .cfg_sfu_input_scale_shift      (cfg_sfu_input_scale_shift  ),
        .cfg_sfu_output_scale_shift     (cfg_sfu_output_scale_shift ),
        .sfu_input_sram_addr            (sfu_input_sram_addr        ),  
        .sfu_input_col_num              (sfu_input_col_num          ),    
        .sfu_input_row_num              (sfu_input_row_num          ),    
        .sfu_output_spm_addr            (sfu_output_spm_addr        ),    
        .sfu_req_en                     (sfu_req_en                 ),           
        .sfu_busy                       (sfu_busy                   ),             
        .sfu_comp_done                  (sfu_comp_done              ),    
        //irq            
        .irq                            (irq                        )                   
    );

    generate
        if (DISABLE_GEMV == 1) begin : gen_empty_gemv
            gemv_empty #(
                .RF_DATA_WIDTH  ( RF_DATA_WIDTH   ),
                .SPM_SIZE       ( SPM_SIZE        ),
                .SPM_DATA_WIDTH ( GEMV_SPM_DATA_WIDTH ),
                .GEMV_ACC_NUM   ( 4               ),
                .TILE_ELEMS     ( GEMV_TILE_ELEMS )
            ) u_gemv_top (
                .clk                     ( clk                    ),
                .rst_n                   ( rst_n                  ),
                .matvec_input_mat_addr   ( matvec_input_mat_addr  ),
                .matvec_input_vec_addr   ( matvec_input_vec_addr  ),
                .matvec_input_act_scale_addr ( matvec_input_act_scale_addr ),
                .matvec_input_act_scale2_addr ( 32'd0 ),
                .matvec_input_act_group_stride_bytes ( matvec_input_act_group_stride_bytes ),
                .matvec_input_mat_width  ( matvec_input_mat_width ),
                .matvec_input_mat_height ( matvec_input_mat_height),
                .matvec_output_addr      ( matvec_input_mat_stride),
                .matvec_input_scale_addr ( matvec_input_vec_stride),
                .gemv_mode               ( cfg_compute_int_type   ),
                .cfg_decode_flow         ( cfg_decode_flow        ),
                .matvec_req_en           ( matvec_req_en          ),
                .matvec_busy             ( matvec_busy            ),
                .matvec_comp_done        ( matvec_comp_done       ),
                .spm_rd_en               ( gemv_spm_rd_en         ),
                .spm_rd_addr             ( gemv_spm_rd_addr       ),
                .spm_rd_data             ( gemv_spm_rd_data       ),
                .scale_rd_en             ( gemv_scale_rd_en       ),
                .scale_rd_addr           ( gemv_scale_rd_addr     ),
                .scale_rd_data           ( gemv_scale_rd_data     ),
                .act_rd_en               ( gemv_act_rd_en         ),
                .act_rd_addr             ( gemv_act_rd_addr       ),
                .act_rd_data             ( gemv_act_rd_data       ),
                .spm_wr_en               ( gemv_spm_wr_en         ),
                .spm_wr_addr             ( gemv_spm_wr_addr       ),
                .spm_wr_data             ( gemv_spm_wr_data       ),
                .spm_wr_mask             ( gemv_spm_wr_mask       ),
                .act_wr_en               ( flow_act_wr_en         ),
                .act_wr_addr             ( flow_act_wr_addr       ),
                .act_wr_data             ( flow_act_wr_data       ),
                .act_wr_mask             ( flow_act_wr_mask       ),
                .preload_acc_en_i        ( 1'b0                   ),
                .preload_acc_id_i        ( '0                     ),
                .preload_acc_row_i       ( '0                     ),
                .preload_acc_data_i      ( '0                     )
            );
            assign kv_scale_commit_valid = 1'b0;
            assign kv_scale_commit_is_v = 1'b0;
            assign kv_scale_commit_values = '0;
            assign kv_scale_commit_count = '0;
            assign gemv_weight_stream_ready = 1'b0;
        end
        else begin : gen_gemv
            smolvlm2_decode #(
                .RF_DATA_WIDTH  ( RF_DATA_WIDTH   ),
                .SPM_SIZE       ( SPM_SIZE        ),
                .SPM_DATA_WIDTH ( GEMV_SPM_DATA_WIDTH ),
                .GEMV_ACC_NUM   ( 4               ),
                .TILE_ELEMS     ( GEMV_TILE_ELEMS ),
                .DISABLE_DECODE_ROPE     ( DISABLE_DECODE_ROPE     ),
                .DISABLE_DECODE_SOFTMAX  ( DISABLE_DECODE_SOFTMAX  ),
                .DISABLE_DECODE_KV_QUANT ( DISABLE_DECODE_KV_QUANT )
            ) u_smolvlm2_decode (
                .clk                     ( clk                    ),
                .rst_n                   ( rst_n                  ),
                .matvec_input_mat_addr   ( matvec_input_mat_addr  ),
                .matvec_input_vec_addr   ( matvec_input_vec_addr  ),
                .matvec_input_act_scale_addr ( matvec_input_act_scale_addr ),
                .matvec_input_act_scale2_addr ( matvec_input_act_scale2_addr ),
                .matvec_cache_cell_idx ( matvec_cache_cell_idx ),
                .matvec_input_act_group_stride_bytes ( matvec_input_act_group_stride_bytes ),
                .matvec_act_frac_cfg    ( matvec_act_frac_cfg ),
                .matvec_input_mat_width  ( matvec_input_mat_width ),
                .matvec_input_mat_height ( matvec_input_mat_height),
                .matvec_output_addr      ( matvec_input_mat_stride),
                .matvec_input_scale_addr ( matvec_input_vec_stride),
                .gemv_mode               ( cfg_compute_int_type   ),
                .cfg_decode_flow         ( cfg_decode_flow        ),
                .weight_stream_mode      ( matvec_weight_stream_mode),
                .act_scale_enable        ( matvec_act_scale_enable),
                .act_scale2_enable       ( matvec_act_scale2_enable),
                .matvec_req_en           ( matvec_req_en          ),
                .matvec_busy             ( matvec_busy            ),
                .matvec_comp_done        ( matvec_comp_done       ),
                .kv_scale_commit_valid_o ( kv_scale_commit_valid  ),
                .kv_scale_commit_is_v_o  ( kv_scale_commit_is_v   ),
                .kv_scale_commit_values_o( kv_scale_commit_values ),
                .kv_scale_commit_count_o ( kv_scale_commit_count  ),
                .spm_rd_en               ( gemv_spm_rd_en         ),
                .spm_rd_addr             ( gemv_spm_rd_addr       ),
                .spm_rd_data             ( gemv_spm_rd_data       ),
                .scale_rd_en             ( gemv_scale_rd_en       ),
                .scale_rd_addr           ( gemv_scale_rd_addr     ),
                .scale_rd_data           ( gemv_scale_rd_data     ),
                .act_rd_en               ( gemv_act_rd_en         ),
                .act_rd_addr             ( gemv_act_rd_addr       ),
                .act_rd_data             ( gemv_act_rd_data       ),
                .weight_stream_valid     ( gemv_weight_stream_valid),
                .weight_stream_data      ( gemv_weight_stream_data),
                .weight_stream_last      ( gemv_weight_stream_last),
                .weight_stream_ready     ( gemv_weight_stream_ready),
                .spm_wr_en               ( gemv_spm_wr_en         ),
                .spm_wr_addr             ( gemv_spm_wr_addr       ),
                .spm_wr_data             ( gemv_spm_wr_data       ),
                .spm_wr_mask             ( gemv_spm_wr_mask       ),
                .act_wr_en               ( flow_act_wr_en         ),
                .act_wr_addr             ( flow_act_wr_addr       ),
                .act_wr_data             ( flow_act_wr_data       ),
                .act_wr_mask             ( flow_act_wr_mask       ),
                .preload_acc_en_i        ( 1'b0                   ),
                .preload_acc_id_i        ( '0                     ),
                .preload_acc_row_i       ( '0                     ),
                .preload_acc_data_i      ( '0                     )
            );
        end
    endgenerate

    generate
        if (DISABLE_SYSTOLIC_ARRAY == 1) begin : gen_empty_systolic_array
            systolic_array_top_empty #(
                .RF_DATA_WIDTH     ( RF_DATA_WIDTH      ),
                .PE_DATA_WIDTH_IN  ( PE_DATA_WIDTH_IN   ),
                .PE_DATA_WIDTH_OUT ( PE_DATA_WIDTH_OUT  ),
                .INPUT_WIDTH_MAX   ( SA_MAX_LENGTH      ),
                .INPUT_HEIGHT_MAX  ( SA_MAX_LENGTH      ),
                .ARRAY_WIDTH       ( ARRAY_WIDTH        ),
                .ARRAY_HEIGHT      ( ARRAY_WIDTH        ),
                .SPM_SIZE          ( SPM_SIZE           ),
                .SPM_ADDR_WIDTH    ( SPM_ADDR_WIDTH     )
            ) u_systolic_array_top (
                .clk                        (clk    ),
                .rstn                       (rst_n   ),
                
                .cfg_compute_dataflow           (cfg_compute_dataflow       ),
                .cfg_compute_padding_left       (cfg_compute_padding_left   ),
                .cfg_compute_padding_right      (cfg_compute_padding_right  ),
                .cfg_compute_padding_top        (cfg_compute_padding_top    ),
                .cfg_compute_padding_bottom     (cfg_compute_padding_bottom ),
                .cfg_compute_padding_mode       (cfg_compute_padding_mode   ),
                .cfg_compute_weight_shape_m1    (cfg_compute_weight_shape   ),
                .cfg_compute_weight_stride_m1   (cfg_compute_weight_stride  ),
                .cfg_compute_weight_dilation_m1 (cfg_compute_weight_dilation),
                .cfg_compute_is_groupconv       (cfg_compute_is_groupconv   ),
                .cfg_compute_int_type           (cfg_compute_int_type       ),
                .cfg_compute_optype             (cfg_compute_optype         ),
                .cfg_compute_asymmetric_activations(cfg_compute_asymmetric_activations),
                .sa_input_a_stride              (sa_input_a_stride          ),
                .sa_input_b_stride              (sa_input_b_stride          ),
                .sa_input_a_spm_addr            (sa_input_a_spm_addr        ),
                .sa_input_a_col_num_sub1        (sa_input_a_col_num         ),
                .sa_input_a_row_num_sub1        (sa_input_a_row_num         ),
                .sa_input_b_spm_addr            (sa_input_b_spm_addr        ),
                .sa_input_b_col_num_sub1        (sa_input_b_col_num         ),
                .sa_input_b_row_num_sub1        (sa_input_b_row_num         ),
                .cfg_accu_biaspsum_height       (cfg_accu_biaspsum_height   ),
                .cfg_accu_biaspsum_width        (cfg_accu_biaspsum_width    ),

                .sa_req_en                      (sa_req_en                  ),
                .sa_busy                        (systolic_array_busy        ),
                .sa_comp_done                   (systolic_array_done        ),
                .sa_spm_rd1_en                  (sa_spm_rd1_en              ),
                .sa_spm_rd1_addr                (sa_spm_rd1_addr            ),
                .sa_spm_rd1_data_in             (sa_spm_rd1_data_in         ),
                .sa_spm_rd2_en                  (sa_spm_rd2_en              ),
                .sa_spm_rd2_addr                (sa_spm_rd2_addr            ),
                .sa_spm_rd2_data_in             (sa_spm_rd2_data_in         ),
                
                .sa_acc_valid                   (sa_acc_valid               ),
                .sa_acc_mask                    (sa_acc_mask                ),
                .sa_acc_data_out                (sa_acc_data_out            ),
                .sa_acc_start                   (sa_acc_start               )
            );
        end
        else begin : gen_systolic_array
            systolic_array_top #(
                .RF_DATA_WIDTH ( RF_DATA_WIDTH ),
                .PE_DATA_WIDTH_IN( PE_DATA_WIDTH_IN ),
                .PE_DATA_WIDTH_OUT( PE_DATA_WIDTH_OUT ),
                .INPUT_WIDTH_MAX( SA_MAX_LENGTH ),
                .INPUT_HEIGHT_MAX( SA_MAX_LENGTH ),
                .ARRAY_WIDTH( ARRAY_WIDTH ),
                .ARRAY_HEIGHT( ARRAY_WIDTH ),
                .SPM_SIZE( SPM_SIZE ),
                .SPM_ADDR_WIDTH( SPM_ADDR_WIDTH ),
                .PE_FPGA_DSP   ( PE_FPGA_DSP   ),
                .DSP_PE_NUM    ( DSP_PE_NUM    ),
                .DISABLE_IM2COL( DISABLE_IM2COL)
            ) u_systolic_array_top (
                .clk                        (clk    ),
                .rstn                       (rst_n   ),
                
                .cfg_compute_dataflow           (cfg_compute_dataflow       ),
                .cfg_compute_padding_left       (cfg_compute_padding_left   ),
                .cfg_compute_padding_right      (cfg_compute_padding_right  ),
                .cfg_compute_padding_top        (cfg_compute_padding_top    ),
                .cfg_compute_padding_bottom     (cfg_compute_padding_bottom ),
                .cfg_compute_padding_mode       (cfg_compute_padding_mode   ),
                .cfg_compute_weight_shape_m1    (cfg_compute_weight_shape   ),
                .cfg_compute_weight_stride_m1   (cfg_compute_weight_stride  ),
                .cfg_compute_weight_dilation_m1 (cfg_compute_weight_dilation),
                .cfg_compute_is_groupconv       (cfg_compute_is_groupconv   ),
                .cfg_compute_int_type           (cfg_compute_int_type       ),
                .cfg_compute_optype             (cfg_compute_optype         ),
                .cfg_compute_asymmetric_activations(cfg_compute_asymmetric_activations),
                .sa_input_a_stride              (sa_input_a_stride          ),
                .sa_input_b_stride              (sa_input_b_stride          ),
                .sa_input_a_spm_addr            (sa_input_a_spm_addr        ),
                .sa_input_a_col_num_sub1        (sa_input_a_col_num         ),
                .sa_input_a_row_num_sub1        (sa_input_a_row_num         ),
                .sa_input_b_spm_addr            (sa_input_b_spm_addr        ),
                .sa_input_b_col_num_sub1        (sa_input_b_col_num         ),
                .sa_input_b_row_num_sub1        (sa_input_b_row_num         ),
                .cfg_accu_biaspsum_height       (cfg_accu_biaspsum_height   ),
                .cfg_accu_biaspsum_width        (cfg_accu_biaspsum_width    ),

                .sa_req_en                      (sa_req_en                  ),
                .sa_busy                        (systolic_array_busy        ),
                .sa_comp_done                   (systolic_array_done        ),
                .sa_spm_rd1_en                  (sa_spm_rd1_en              ),
                .sa_spm_rd1_addr                (sa_spm_rd1_addr            ),
                .sa_spm_rd1_data_in             (sa_spm_rd1_data_in         ),
                .sa_spm_rd2_en                  (sa_spm_rd2_en              ),
                .sa_spm_rd2_addr                (sa_spm_rd2_addr            ),
                .sa_spm_rd2_data_in             (sa_spm_rd2_data_in         ),
                
                .sa_acc_valid                   (sa_acc_valid               ),
                .sa_acc_mask                    (sa_acc_mask                ),
                .sa_acc_data_out                (sa_acc_data_out            ),
                .sa_acc_start                   (sa_acc_start               )
            );
        end
    endgenerate

    generate
        if (DISABLE_ACCUMULATOR == 1) begin : gen_empty_accumulator
            accumulator_empty # (
                .AXI_DATA_WIDTH       ( AXI_DATA_WIDTH       ),
                .PE_WIDTH             ( PE_WIDTH             ),
                .ACCU_DATA_WIDTH      ( PE_DATA_WIDTH_OUT    ),
                .SPM_DATA_WIDTH       ( PE_DATA_WIDTH_IN     ),
                .SPM_ADDR_WIDTH       ( SPM_ADDR_WIDTH       ),
                .ACCU_SIZE            ( ACC_SIZE             ),
                .RD_PORTS             ( RD_PORTS             ),
                .WR_PORTS             ( WR_PORTS             ),
                .RF_DATA_WIDTH        ( RF_DATA_WIDTH        ),
                .ALPHA_1              ( ALPHA_1              ),
                .ALPHA_2              ( ALPHA_2              ),
                .ALPHA_3              ( ALPHA_3              ),
                .DISABLE_ACC_INT32_TO_INT8 ( DISABLE_ACC_INT32_TO_INT8 )
            ) u_accu (
                .clk                      ( clk                          ),
                .rst_n                    ( rst_n                        ),
                .cfg_mvin_isbias          ( cfg_mvin_isbias              ),
                .cfg_compute_optype       ( cfg_compute_optype           ),     
                .cfg_compute_accout_dest  ( cfg_compute_accout_dest      ),
                .cfg_compute_output_scale ( cfg_compute_output_scale     ),
                .cfg_compute_output_scale_shift( cfg_compute_output_scale_shift ),
                .sa_input_b_col_num       ( sa_input_b_col_num           ),     
                .cfg_accu_biaspsum_addr   ( cfg_accu_biaspsum_addr       ),
                .cfg_accu_biaspsum_stride ( cfg_accu_biaspsum_stride     ),
                .cfg_accu_biaspsum_height ( cfg_accu_biaspsum_height     ),
                .cfg_accu_biaspsum_width  ( cfg_accu_biaspsum_width      ),
                .cfg_accu_output_addr     ( cfg_accu_output_addr         ),
                .cfg_accu_output_stride   ( cfg_accu_output_stride       ),
                .cfg_accu_relu            ( cfg_accu_relu                ),
                .cfg_accu_relu_type       ( cfg_accu_relu_type           ),
                .cfg_accu_isaccu          ( cfg_accu_isaccu              ),
                .cfg_accu_isbias          ( cfg_accu_isbias              ),
                .matadd_input_a_addr      ( matadd_input_a_addr          ),
                .matadd_input_b_addr      ( matadd_input_b_addr          ),
                .matadd_input_col_num     ( matadd_input_col_num         ),
                .matadd_input_row_num     ( matadd_input_row_num         ),
                .matadd_output_addr       ( matadd_output_addr           ),
                .accadd_start             ( sa_acc_start                 ),
                .accadd_busy              ( accadd_busy                  ),
                .accadd_done              ( sa_comp_done                 ),
                .sa_wr_en                 ( sa_acc_valid                 ),
                .sa_wr_mask               ( sa_acc_mask                  ),
                .sa_wr_data               ( sa_acc_data_out              ),
                .dma_accu_wr_en           ( dma_acc_wr_en                ),
                .dma_accu_wr_addr         ( dma_acc_wr_addr              ),
                .dma_accu_wr_mask         ( dma_acc_wr_mask              ),
                .dma_accu_wr_data         ( dma_acc_din                  ),
                .dma_mvin_resp_done       ( dma_mvin_resp_done_any       ),
                .dma_mvout_resp_done      ( dma_mvout_resp_done_any      ),
                .dma_accu_rd_en           ( dma_acc_rd_en                ),
                .dma_accu_rd_addr         ( dma_acc_rd_addr              ),
                .dma_accu_rd_data         ( dma_acc_dout                 ),
                .dma_accu_rd_valid        ( dma_acc_rd_valid             ),
                .cfg_mvout_output_precision(cfg_mvout_output_precision   ),
                .cfg_mvout_per_channel     (cfg_mvout_per_channel        ),
                .mvout_col_num             (dma_mvout_col_num[ACC_DMA_IDX]),
                .cfg_mvout_output_scale   (cfg_mvout_output_scale        ),
                .cfg_mvout_output_scale_shift(cfg_mvout_output_scale_shift),
                .accu_spm_wr_en           ( accu_spm_wr_en               ),
                .accu_spm_wr_addr         ( accu_spm_wr_addr             ),
                .accu_spm_wr_mask         ( accu_spm_wr_mask             ),
                .accu_spm_wr_data         ( accu_spm_wr_data             ),
                .matadd_req_en            ( matadd_req_en                ),
                .matadd_busy              ( matadd_busy                  ),
                .matadd_comp_done         ( matadd_comp_done             )
            );
        end
        else begin : gen_accumulator
            accumulator # (
                .SPM_FPGA_SRAM        ( SPM_FPGA_SRAM        ),
                .AXI_DATA_WIDTH       ( AXI_DATA_WIDTH       ),
                .PE_WIDTH             ( PE_WIDTH             ),
                .ACCU_DATA_WIDTH      ( PE_DATA_WIDTH_OUT    ),
                .SPM_DATA_WIDTH       ( PE_DATA_WIDTH_IN     ),
                .SPM_ADDR_WIDTH       ( SPM_ADDR_WIDTH       ),
                .ACCU_SIZE            ( ACC_SIZE             ),
                .RD_PORTS             ( RD_PORTS             ),
                .WR_PORTS             ( WR_PORTS             ),
                .RF_DATA_WIDTH        ( RF_DATA_WIDTH        ),
                .DISABLE_ACC_INT32_TO_INT8 ( DISABLE_ACC_INT32_TO_INT8 ),
                .DISABLE_ACC_TO_SPM_PATH ( DISABLE_ACC_TO_SPM_PATH ),
                .ALPHA_1              ( ALPHA_1              ),
                .ALPHA_2              ( ALPHA_2              ),
                .ALPHA_3              ( ALPHA_3              )
            ) u_accu (
                .clk                      ( clk                          ),
                .rst_n                    ( rst_n                        ),
                .cfg_mvin_isbias          ( cfg_mvin_isbias              ),
                .cfg_compute_optype       ( cfg_compute_optype           ),     
                .cfg_compute_accout_dest  ( cfg_compute_accout_dest      ),
                .cfg_compute_output_scale ( cfg_compute_output_scale     ),
                .cfg_compute_output_scale_shift( cfg_compute_output_scale_shift ),
                .sa_input_b_col_num       ( sa_input_b_col_num           ),     
                .cfg_accu_biaspsum_addr   ( cfg_accu_biaspsum_addr       ),
                .cfg_accu_biaspsum_stride ( cfg_accu_biaspsum_stride     ),
                .cfg_accu_biaspsum_height ( cfg_accu_biaspsum_height     ),
                .cfg_accu_biaspsum_width  ( cfg_accu_biaspsum_width      ),
                .cfg_accu_output_addr     ( cfg_accu_output_addr         ),
                .cfg_accu_output_stride   ( cfg_accu_output_stride       ),
                .cfg_accu_relu            ( cfg_accu_relu                ),
                .cfg_accu_relu_type       ( cfg_accu_relu_type           ),
                .cfg_accu_isaccu          ( cfg_accu_isaccu              ),
                .cfg_accu_isbias          ( cfg_accu_isbias              ),
                .matadd_input_a_addr      ( matadd_input_a_addr          ),
                .matadd_input_b_addr      ( matadd_input_b_addr          ),
                .matadd_input_col_num     ( matadd_input_col_num         ),
                .matadd_input_row_num     ( matadd_input_row_num         ),
                .matadd_output_addr       ( matadd_output_addr           ),
                .accadd_start             ( sa_acc_start                 ),
                .accadd_busy              ( accadd_busy                  ),
                .accadd_done              ( sa_comp_done                 ),
                .sa_wr_en                 ( sa_acc_valid                 ),
                .sa_wr_mask               ( sa_acc_mask                  ),
                .sa_wr_data               ( sa_acc_data_out              ),
                .dma_accu_wr_en           ( dma_acc_wr_en                ),
                .dma_accu_wr_addr         ( dma_acc_wr_addr              ),
                .dma_accu_wr_mask         ( dma_acc_wr_mask              ),
                .dma_accu_wr_data         ( dma_acc_din                  ),
                .dma_mvin_resp_done       ( dma_mvin_resp_done_any       ),
                .dma_mvout_resp_done      ( dma_mvout_resp_done_any      ),
                .dma_accu_rd_en           ( dma_acc_rd_en                ),
                .dma_accu_rd_addr         ( dma_acc_rd_addr              ),
                .dma_accu_rd_data         ( dma_acc_dout                 ),
                .dma_accu_rd_valid        ( dma_acc_rd_valid             ),
                .cfg_mvout_output_precision(cfg_mvout_output_precision   ),
                .cfg_mvout_per_channel     (cfg_mvout_per_channel        ),
                .mvout_col_num             (dma_mvout_col_num[ACC_DMA_IDX]),
                .cfg_mvout_output_scale   (cfg_mvout_output_scale        ),
                .cfg_mvout_output_scale_shift(cfg_mvout_output_scale_shift),
                .accu_spm_wr_en           ( accu_spm_wr_en               ),
                .accu_spm_wr_addr         ( accu_spm_wr_addr             ),
                .accu_spm_wr_mask         ( accu_spm_wr_mask             ),
                .accu_spm_wr_data         ( accu_spm_wr_data             ),
                .matadd_req_en            ( matadd_req_en                ),
                .matadd_busy              ( matadd_busy                  ),
                .matadd_comp_done         ( matadd_comp_done             )
            );
        end
    endgenerate

    generate
        if (DISABLE_SFU == 1) begin : gen_empty_sfu
            sfu_empty #(
                .RF_DATA_WIDTH    ( RF_DATA_WIDTH    ),
                .PE_WIDTH         ( PE_WIDTH         ),
                .PE_DATA_WIDTH    ( PE_DATA_WIDTH    ),
                .SPM_SIZE         ( SPM_SIZE         ),
                .DISABLE_SOFTMAX  ( DISABLE_SOFTMAX  ),
                .DISABLE_GELU     ( DISABLE_GELU     ),
                .DISABLE_LAYERNORM( DISABLE_LAYERNORM),
                .DISABLE_RESAMPLE ( DISABLE_RESAMPLE ),
                .DISABLE_TRANSPOSE( DISABLE_TRANSPOSE),
                .GELU_NUM         ( GELU_NUM         )
            ) u_sfu (
                .clk                       ( clk                     ),
                .rst_n                     ( rst_n                   ),

                .cfg_sfu_op                ( cfg_sfu_op              ),
                .cfg_sfu_int_type          ( cfg_sfu_int_type        ),
                .cfg_sfu_is_quant          ( cfg_sfu_is_quant        ),
                .cfg_trans_out_ispad_row   ( cfg_trans_out_ispad_row ),
                .cfg_trans_out_ispad_col   ( cfg_trans_out_ispad_col ),
                .cfg_sfu_output_zeropoint  ( cfg_sfu_output_zeropoint),
                .cfg_sfu_input_zeropoint   ( cfg_sfu_input_zeropoint ),
                .cfg_sfu_input_scale       ( cfg_sfu_input_scale     ),
                .cfg_sfu_output_scale      ( cfg_sfu_output_scale    ),
                .cfg_sfu_input_scale_shift ( cfg_sfu_input_scale_shift),
                .cfg_sfu_output_scale_shift( cfg_sfu_output_scale_shift),
                .sfu_input_sram_addr       ( sfu_input_sram_addr     ),
                .sfu_input_col_num         ( sfu_input_col_num       ),
                .sfu_input_row_num         ( sfu_input_row_num       ),
                .sfu_output_spm_addr       ( sfu_output_spm_addr     ),
                
                .sfu_req_en                ( sfu_req_en              ),
                .sfu_busy                  ( sfu_busy                ),
                .sfu_comp_done             ( sfu_comp_done           ),

                .sfu_spm_rd_en             ( sfu_spm_rd_en           ),
                .sfu_spm_rd_addr           ( sfu_spm_rd_addr         ),
                .sfu_spm_rd_data_in        ( sfu_spm_rd_data_in      ),
                .sfu_spm_wr_en             ( sfu_spm_wr_en           ),
                .sfu_spm_wr_addr           ( sfu_spm_wr_addr         ),
                .sfu_spm_wr_data_out       ( sfu_spm_wr_data_out     ),
                .sfu_spm_wr_mask           ( sfu_spm_wr_mask         )
            );
        end
        else begin : gen_sfu
            sfu #(
                .GELU_NUM           (GELU_NUM),
                .DISABLE_SOFTMAX    (DISABLE_SOFTMAX),
                .DISABLE_GELU       (DISABLE_GELU),
                .DISABLE_LAYERNORM  (DISABLE_LAYERNORM),
                .DISABLE_RESAMPLE   (DISABLE_RESAMPLE),
                .DISABLE_TRANSPOSE  (DISABLE_TRANSPOSE)
            ) u_sfu (
                .clk                       ( clk                     ),
                .rst_n                     ( rst_n                   ),

                .cfg_sfu_op                ( cfg_sfu_op              ),
                .cfg_sfu_int_type          ( cfg_sfu_int_type        ),
                .cfg_sfu_is_quant          ( cfg_sfu_is_quant        ),
                .cfg_trans_out_ispad_row   (cfg_trans_out_ispad_row  ),
                .cfg_trans_out_ispad_col   (cfg_trans_out_ispad_col  ),
                .cfg_sfu_output_zeropoint  ( cfg_sfu_output_zeropoint),
                .cfg_sfu_input_zeropoint   ( cfg_sfu_input_zeropoint ),
                .cfg_sfu_input_scale       ( cfg_sfu_input_scale     ),
                .cfg_sfu_output_scale      ( cfg_sfu_output_scale    ),
                .cfg_sfu_input_scale_shift ( cfg_sfu_input_scale_shift),
                .cfg_sfu_output_scale_shift( cfg_sfu_output_scale_shift),
                .sfu_input_sram_addr       ( sfu_input_sram_addr     ),
                .sfu_input_col_num         ( sfu_input_col_num       ),
                .sfu_input_row_num         ( sfu_input_row_num       ),
                .sfu_output_spm_addr       ( sfu_output_spm_addr     ),
                
                .sfu_req_en                ( sfu_req_en              ),
                .sfu_busy                  ( sfu_busy                ),
                .sfu_comp_done             ( sfu_comp_done           ),

                .sfu_spm_rd_en             ( sfu_spm_rd_en           ),
                .sfu_spm_rd_addr           ( sfu_spm_rd_addr         ),
                .sfu_spm_rd_data_in        ( sfu_spm_rd_data_in      ),
                .sfu_spm_wr_en             ( sfu_spm_wr_en           ),
                .sfu_spm_wr_addr           ( sfu_spm_wr_addr         ),
                .sfu_spm_wr_data_out       ( sfu_spm_wr_data_out     ),
                .sfu_spm_wr_mask           ( sfu_spm_wr_mask         )
            );
        end
    endgenerate








endmodule

`endif 
