//*****************************************************************
// 'Empty' stub module for FPGA verification.
//
// This module has the exact same interface as 'DMA'
// but all outputs are tied to 0, simulating an inactive state.
//*****************************************************************
module dma_empty
    import npu_config_pkg::*;
#(
    //fit for:SPM_DATA_WIDTH=AXI_DATA_WIDTH
) (
    input  logic                                   clk,
    input  logic                                   rst_n,

    //------------------------------------- 
    // DMA Control Signals 
    //------------------------------------- 

    input  logic   [RF_DATA_WIDTH/2-1:0]           mvin_dram_addr,
    input  logic   [RF_DATA_WIDTH/2-1:0]           mvin_sram_addr,
    input  logic   [RF_DATA_WIDTH/2-1:0]           mvin_col_num,
    input  logic   [RF_DATA_WIDTH/2-1:0]           mvin_row_num,
    input  logic   [RF_DATA_WIDTH/2-1:0]           mvout_dram_addr,
    input  logic   [RF_DATA_WIDTH/2-1:0]           mvout_sram_addr,
    input  logic   [RF_DATA_WIDTH/2-1:0]           mvout_col_num,
    input  logic   [RF_DATA_WIDTH/2-1:0]           mvout_row_num,

    input  logic   [1:0]                           cfg_mvin_input_type,
    input  logic   [1:0]                           cfg_mvin_input_precision,
    input  logic   [1:0]                           cfg_mvout_output_precision,
    input  logic                                   cfg_mvin_is_quant,
    input  logic                                   cfg_mvout_is_quant,
    input  logic                                   cfg_mvin_dest,
    input  logic                                   cfg_mvout_source,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_mvin_sram_stride,
    input  logic   [RF_DATA_WIDTH/2-1:0]           cfg_mvin_dram_stride,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_mvout_sram_stride,
    input  logic   [RF_DATA_WIDTH/2-1:0]           cfg_mvout_dram_stride,
    input  logic   [RF_DATA_WIDTH/2-1:0]           cfg_mvin_input_zeropoint,
    input  logic   [RF_DATA_WIDTH/2-1:0]           cfg_mvout_output_zeropoint,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_mvin_input_scale,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_mvout_output_scale,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_mvin_input_scale_shift,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_mvout_output_scale_shift,

    input  logic                                   dma_mvin_req_en,
    input  logic                                   dma_mvout_req_en,
    output logic                                   dma_mvin_resp_done,  // Output
    output logic                                   dma_mvout_resp_done, // Output
    output logic                                   dma_mvin_busy,       // Output
    output logic                                   dma_mvout_busy,      // Output
    
    //------------------------------------- 
    // SPM Control Signals 
    //------------------------------------- 

    output logic   [SPM_DATA_WIDTH-1:0]            spm_din,        // Output
    output logic                                   spm_wr_en,      // Output
    output logic   [$clog2(SPM_SIZE)-1:0]          spm_wr_addr,    // Output
    output logic   [$clog2(SPM_SIZE)-1:0]          spm_rd_addr,    // Output
    output logic                                   spm_rd_en,      // Output
    output logic   [SPM_DATA_WIDTH/8-1:0]          spm_wr_mask,    // Output
    input  logic   [SPM_DATA_WIDTH-1:0]            spm_dout,

    //------------------------------------- 
    // ACC Control Signals 
    //------------------------------------- 

    output logic   [ACC_DATA_WIDTH-1:0]            acc_din,        // Output
    output logic                                   acc_wr_en,      // Output
    output logic   [$clog2(ACC_SIZE)-1:0]          acc_wr_addr,    // Output
    output logic   [$clog2(ACC_SIZE)-1:0]          acc_rd_addr,    // Output
    output logic                                   acc_rd_en,      // Output
    output logic   [ACC_DATA_WIDTH/32-1:0]         acc_wr_mask,    // Output
    input  logic   [ACC_DATA_WIDTH-1:0]            acc_dout,
    
    //-------------------------------------    
    // AXI Control Signals (Master)    
    //-------------------------------------    
    output logic   [AXI_ID_WIDTH-1 : 0]            m_axi_awid,    // Output
    output logic   [AXI_ADDR_WIDTH-1 : 0]          m_axi_awaddr,  // Output
    output logic   [7 : 0]                         m_axi_awlen,   // Output
    output logic   [2 : 0]                         m_axi_awsize,  // Output
    output logic   [1 : 0]                         m_axi_awburst, // Output
    output logic                                   m_axi_awvalid, // Output
    input  logic                                   m_axi_awready,

    output logic   [AXI_DATA_WIDTH-1 : 0]          m_axi_wdata,   // Output
    output logic   [AXI_DATA_WIDTH/8-1 : 0]        m_axi_wstrb,   // Output
    output logic                                   m_axi_wlast,   // Output
    output logic                                   m_axi_wvalid,  // Output
    input  logic                                   m_axi_wready,
    
    input  logic   [AXI_ID_WIDTH-1 : 0]            m_axi_bid,
    input  logic   [1 : 0]                         m_axi_bresp,
    input  logic                                   m_axi_bvalid,
    output logic                                   m_axi_bready,  // Output
    
    output logic   [AXI_ID_WIDTH-1 : 0]            m_axi_arid,    // Output
    output logic   [AXI_ADDR_WIDTH-1 : 0]          m_axi_araddr,  // Output
    output logic   [7 : 0]                         m_axi_arlen,   // Output
    output logic   [2 : 0]                         m_axi_arsize,  // Output
    output logic   [1 : 0]                         m_axi_arburst, // Output
    output logic                                   m_axi_arvalid, // Output
    input  logic                                   m_axi_arready,
    
    input  logic   [AXI_ID_WIDTH-1 : 0]            m_axi_rid,
    input  logic   [AXI_DATA_WIDTH-1 : 0]          m_axi_rdata,
    input  logic   [1 : 0]                         m_axi_rresp,
    input  logic                                   m_axi_rlast,
    input  logic                                   m_axi_rvalid,
    output logic                                   m_axi_rready   // Output
);

    // ------------------------------------- 
    // 1. DMA Control Outputs (set to inactive/done)
    // ------------------------------------- 
    assign dma_mvin_resp_done = 1'b1; // Immediately signal done for read requests
    assign dma_mvout_resp_done = 1'b1; // Immediately signal done for write requests
    assign dma_mvin_busy = 1'b0;
    assign dma_mvout_busy = 1'b0;

    // ------------------------------------- 
    // 2. SPM Control Outputs (set to 0 / inactive)
    // ------------------------------------- 
    assign spm_din     = '0;
    assign spm_wr_en   = 1'b0;
    assign spm_wr_addr = '0;
    assign spm_rd_addr = '0;
    assign spm_rd_en   = 1'b0;
    assign spm_wr_mask = '0;

    // ------------------------------------- 
    // 3. ACC Control Outputs (set to 0 / inactive)
    // ------------------------------------- 
    assign acc_din     = '0;
    assign acc_wr_en   = 1'b0;
    assign acc_wr_addr = '0;
    assign acc_rd_addr = '0;
    assign acc_rd_en   = 1'b0;
    assign acc_wr_mask = '0;
    
    // ------------------------------------- 
    // 4. AXI Master Outputs (set to 0 / inactive)
    // ------------------------------------- 
    
    // Write Address Channel (AW)
    assign m_axi_awid    = '0;
    assign m_axi_awaddr  = '0;
    assign m_axi_awlen   = '0;
    assign m_axi_awsize  = '0;
    assign m_axi_awburst = '0;
    assign m_axi_awvalid = 1'b0; // Never issue a write request
    
    // Write Data Channel (W)
    assign m_axi_wdata   = '0;
    assign m_axi_wstrb   = '0;
    assign m_axi_wlast   = 1'b0;
    assign m_axi_wvalid  = 1'b0; // Never issue write data
    
    // Write Response Channel (B)
    assign m_axi_bready  = 1'b1; // Always ready to accept a response (simplifies AXI connection)

    // Read Address Channel (AR)
    assign m_axi_arid    = '0;
    assign m_axi_araddr  = '0;
    assign m_axi_arlen   = '0;
    assign m_axi_arsize  = '0;
    assign m_axi_arburst = '0;
    assign m_axi_arvalid = 1'b0; // Never issue a read request
    
    // Read Data Channel (R)
    assign m_axi_rready  = 1'b1; // Always ready to accept data (simplifies AXI connection)

endmodule