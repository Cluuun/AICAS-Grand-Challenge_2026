`ifndef DMA_TOP_V
`define DMA_TOP_V

module DMA_TOP
import npu_config_pkg::*;
#(
    //fit for:SPM_DATA_WIDTH=AXI_DATA_WIDTH
    parameter int RF_DATA_WIDTH   = npu_config_pkg::RF_DATA_WIDTH,
    parameter int DMA_NUM         = npu_config_pkg::DMA_NUM,
    parameter int AXI_ID_WIDTH    = npu_config_pkg::AXI_ID_WIDTH,
    parameter int AXI_ADDR_WIDTH  = npu_config_pkg::AXI_ADDR_WIDTH,
    parameter int AXI_DATA_WIDTH  = npu_config_pkg::AXI_DATA_WIDTH,
    parameter int AXI_MAX_BURST_BEATS = 16,
    parameter int SPM_SIZE        = npu_config_pkg::SPM_SIZE,
    parameter int SPM_DATA_WIDTH  = npu_config_pkg::SPM_DATA_WIDTH,
    parameter int ACC_SIZE        = npu_config_pkg::ACC_SIZE,
    parameter int ACC_DATA_WIDTH  = npu_config_pkg::ACC_DATA_WIDTH,
    parameter integer DISABLE_MVIN_INT8_TO_INT32 = 0
) (
    input  logic                               clk                         ,
    input  logic                               rst_n                       ,

    //-------------------------------------
    // DMA Control Signals
    //-------------------------------------
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvin_dram_addr              ,
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvin_sram_addr              ,
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvin_col_num                ,
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvin_row_num                ,
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvout_dram_addr             ,
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvout_sram_addr             ,
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvout_col_num               ,
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     mvout_row_num               ,

    input  logic [DMA_NUM-1:0][1:0]                     cfg_mvin_input_type         ,   // data type of the mvin data, 00 for input feature/A, 01 for weight/B
                                                                                        // 10 for bias
    input  logic [DMA_NUM-1:0][1:0]                     cfg_mvout_output_type       ,   // data type of the mvout data, 00 for final output, 01 for part sum
    input  logic [DMA_NUM-1:0][1:0]                     cfg_mvin_input_precision    ,   // data precision, 00/01/10/11 for int4/int8/fp16/fp32
                                                                                        // for bias :int16/int32/fp16/fp32
    input  logic [DMA_NUM-1:0][1:0]                     cfg_mvout_output_precision  ,
    input  logic [DMA_NUM-1:0]                          cfg_mvin_is_quant           ,   // mvin/mvout is/not quant
    input  logic [DMA_NUM-1:0]                          cfg_mvout_is_quant          ,
    input  logic [DMA_NUM-1:0]                          cfg_mvin_dest               ,   // mvin data destination, 0 for SPM, 1 for ACC
    input  logic [DMA_NUM-1:0]                          cfg_mvout_source            ,   // mvout data source, 0 for SPM, 1 for ACC
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0]     cfg_mvin_sram_stride        ,   // Stride for mvin SPM/ACC address increment
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     cfg_mvin_dram_stride        ,   // Stride for mvin DRAM address increment
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0]     cfg_mvout_sram_stride       ,   // Stride for mvout SPM/ACC address increment
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     cfg_mvout_dram_stride       ,   // Stride for mvout DRAM address increment
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     cfg_mvin_input_zeropoint    ,   // for matrix add, the input quant zero point
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0]     cfg_mvout_output_zeropoint  ,   // for matrix add, the output quant zero point
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0]     cfg_mvin_input_scale        ,   // for matrix add, the input quant scale
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0]     cfg_mvout_output_scale      ,   // for matrix add, the output quant scale
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0]     cfg_mvin_input_scale_shift  ,   // for matrix add, the input quant scale shift
    input  logic [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0]     cfg_mvout_output_scale_shift,   // for matrix add, the output quant scale shift

    input  logic [DMA_NUM-1:0]                          dma_mvin_req_en             ,
    input  logic [DMA_NUM-1:0]                          dma_mvout_req_en            ,
    output logic [DMA_NUM-1:0]                          dma_mvin_resp_done          ,   // Response done for each DMA
    output logic [DMA_NUM-1:0]                          dma_mvout_resp_done         ,
    output logic [DMA_NUM-1:0]                          dma_mvin_busy               ,
    output logic [DMA_NUM-1:0]                          dma_mvout_busy              ,

    //-------------------------------------
    // SPM Control Signals
    //-------------------------------------
    output logic  [DMA_NUM-1:0]  [SPM_DATA_WIDTH-1:0]      spm_din                      ,
    output logic  [DMA_NUM-1:0]                            spm_wr_en                    ,
    output logic  [DMA_NUM-1:0]  [$clog2(SPM_SIZE)-1:0]    spm_wr_addr                  ,
    output logic  [DMA_NUM-1:0]  [$clog2(SPM_SIZE)-1:0]    spm_rd_addr                  ,
    output logic  [DMA_NUM-1:0]                            spm_rd_en                    ,
    output logic  [DMA_NUM-1:0]  [SPM_DATA_WIDTH/8-1:0]    spm_wr_mask                  ,
    input  logic  [DMA_NUM-1:0]  [SPM_DATA_WIDTH-1:0]      spm_dout                     ,

    //-------------------------------------
    // ACC Control Signals
    //-------------------------------------
    output logic  [DMA_NUM-1:0]  [ACC_DATA_WIDTH-1:0]      acc_din                      ,
    output logic  [DMA_NUM-1:0]                            acc_wr_en                    ,
    output logic  [DMA_NUM-1:0]  [$clog2(ACC_SIZE)-1:0]    acc_wr_addr                  ,
    output logic  [DMA_NUM-1:0]  [$clog2(ACC_SIZE)-1:0]    acc_rd_addr                  ,
    output logic  [DMA_NUM-1:0]                            acc_rd_en                    ,
    output logic  [DMA_NUM-1:0]  [ACC_DATA_WIDTH/32-1:0]   acc_wr_mask                  ,
    input  logic  [DMA_NUM-1:0]  [ACC_DATA_WIDTH-1:0]      acc_dout                     ,
    input  logic  [DMA_NUM-1:0]                            acc_dout_valid               ,

    //-------------------------------------
    // AXI Control Signals
    //-------------------------------------
    output  logic  [DMA_NUM-1:0]  [AXI_ID_WIDTH-1 : 0]        m_axi_awid                 ,
	output  logic  [DMA_NUM-1:0]  [AXI_ADDR_WIDTH-1 : 0]      m_axi_awaddr               ,
	output  logic  [DMA_NUM-1:0]  [7 : 0]                     m_axi_awlen                ,
	output  logic  [DMA_NUM-1:0]  [2 : 0]                     m_axi_awsize               ,
	output  logic  [DMA_NUM-1:0]  [1 : 0]                     m_axi_awburst              ,
	output  logic  [DMA_NUM-1:0]                              m_axi_awvalid              ,
	input   logic  [DMA_NUM-1:0]                              m_axi_awready              ,
	output  logic  [DMA_NUM-1:0]  [AXI_DATA_WIDTH-1 : 0]      m_axi_wdata                ,
	output  logic  [DMA_NUM-1:0]  [AXI_DATA_WIDTH/8-1 : 0]    m_axi_wstrb                ,
	output  logic  [DMA_NUM-1:0]                              m_axi_wlast                ,
	output  logic  [DMA_NUM-1:0]                              m_axi_wvalid               ,
	input   logic  [DMA_NUM-1:0]                              m_axi_wready               ,
	input   logic  [DMA_NUM-1:0]  [AXI_ID_WIDTH-1 : 0]        m_axi_bid                  ,
	input   logic  [DMA_NUM-1:0]  [1 : 0]                     m_axi_bresp                ,
	input   logic  [DMA_NUM-1:0]                              m_axi_bvalid               ,
	output  logic  [DMA_NUM-1:0]                              m_axi_bready               ,
	output  logic  [DMA_NUM-1:0]  [AXI_ID_WIDTH-1 : 0]        m_axi_arid                 ,
	output  logic  [DMA_NUM-1:0]  [AXI_ADDR_WIDTH-1 : 0]      m_axi_araddr               ,
	output  logic  [DMA_NUM-1:0]  [7 : 0]                     m_axi_arlen                ,
	output  logic  [DMA_NUM-1:0]  [2 : 0]                     m_axi_arsize               ,
	output  logic  [DMA_NUM-1:0]  [1 : 0]                     m_axi_arburst              ,
	output  logic  [DMA_NUM-1:0]                              m_axi_arvalid              ,
	input   logic  [DMA_NUM-1:0]                              m_axi_arready              ,
	input   logic  [DMA_NUM-1:0]  [AXI_ID_WIDTH-1 : 0]        m_axi_rid                  ,
	input   logic  [DMA_NUM-1:0]  [AXI_DATA_WIDTH-1 : 0]      m_axi_rdata                ,
	input   logic  [DMA_NUM-1:0]  [1 : 0]                     m_axi_rresp                ,
	input   logic  [DMA_NUM-1:0]                              m_axi_rlast                ,
	input   logic  [DMA_NUM-1:0]                              m_axi_rvalid               ,
	output  logic  [DMA_NUM-1:0]                              m_axi_rready
);

// Generate DMA instances
genvar i;
generate
    for (i = 0; i < DMA_NUM; i++) begin : dma_instances
        DMA #(
            .RF_DATA_WIDTH                 ( RF_DATA_WIDTH   ),
            .AXI_ID_WIDTH                  ( AXI_ID_WIDTH    ),
            .AXI_ADDR_WIDTH                ( AXI_ADDR_WIDTH  ),
            .AXI_DATA_WIDTH                ( AXI_DATA_WIDTH  ),
            .AXI_MAX_BURST_BEATS           ( AXI_MAX_BURST_BEATS ),
            .SPM_SIZE                      ( SPM_SIZE        ),
            .SPM_DATA_WIDTH                ( SPM_DATA_WIDTH  ),
            .ACC_SIZE                      ( ACC_SIZE        ),
            .ACC_DATA_WIDTH                ( ACC_DATA_WIDTH  ),
            .DISABLE_MVIN_INT8_TO_INT32 ( DISABLE_MVIN_INT8_TO_INT32 )
        ) u_dma (
            .clk                            (clk),
            .rst_n                          (rst_n),

            .mvin_dram_addr                 (mvin_dram_addr[i]),
            .mvin_sram_addr                 (mvin_sram_addr[i]),
            .mvin_col_num                   (mvin_col_num[i]),
            .mvin_row_num                   (mvin_row_num[i]),
            .mvout_dram_addr                (mvout_dram_addr[i]),
            .mvout_sram_addr                (mvout_sram_addr[i]),
            .mvout_col_num                  (mvout_col_num[i]),
            .mvout_row_num                  (mvout_row_num[i]),
            .cfg_mvin_input_type            (cfg_mvin_input_type[i]),
            .cfg_mvout_output_type          (cfg_mvout_output_type[i]),
            .cfg_mvin_input_precision       (cfg_mvin_input_precision[i]),
            .cfg_mvout_output_precision     (cfg_mvout_output_precision[i]),
            .cfg_mvin_is_quant              (cfg_mvin_is_quant[i]),
            .cfg_mvout_is_quant             (cfg_mvout_is_quant[i]),
            .cfg_mvin_dest                  (cfg_mvin_dest[i]),
            .cfg_mvout_source               (cfg_mvout_source[i]),
            .cfg_mvin_sram_stride           (cfg_mvin_sram_stride[i]),
            .cfg_mvin_dram_stride           (cfg_mvin_dram_stride[i]),
            .cfg_mvout_sram_stride          (cfg_mvout_sram_stride[i]),
            .cfg_mvout_dram_stride          (cfg_mvout_dram_stride[i]),
            .cfg_mvin_input_zeropoint       (cfg_mvin_input_zeropoint[i]),
            .cfg_mvout_output_zeropoint     (cfg_mvout_output_zeropoint[i]),
            .cfg_mvin_input_scale           (cfg_mvin_input_scale[i]),
            .cfg_mvout_output_scale         (cfg_mvout_output_scale[i]),
            .cfg_mvin_input_scale_shift     (cfg_mvin_input_scale_shift[i]),
            .cfg_mvout_output_scale_shift   (cfg_mvout_output_scale_shift[i]),
            .dma_mvin_req_en                (dma_mvin_req_en[i]),
            .dma_mvout_req_en               (dma_mvout_req_en[i]),
            .dma_mvin_resp_done             (dma_mvin_resp_done[i]),
            .dma_mvout_resp_done            (dma_mvout_resp_done[i]),
            .dma_mvin_busy                  (dma_mvin_busy[i]),
            .dma_mvout_busy                 (dma_mvout_busy[i]),

            .spm_din                        (spm_din[i]),
            .spm_wr_en                      (spm_wr_en[i]),
            .spm_wr_addr                    (spm_wr_addr[i]),
            .spm_rd_addr                    (spm_rd_addr[i]),
            .spm_rd_en                      (spm_rd_en[i]),
            .spm_wr_mask                    (spm_wr_mask[i]),
            .spm_dout                       (spm_dout[i]),

            .acc_din                        (acc_din[i]),
            .acc_wr_en                      (acc_wr_en[i]),
            .acc_wr_addr                    (acc_wr_addr[i]),
            .acc_rd_addr                    (acc_rd_addr[i]),
            .acc_rd_en                      (acc_rd_en[i]),
            .acc_wr_mask                    (acc_wr_mask[i]),
            .acc_dout                       (acc_dout[i]),
            .acc_dout_valid                 (acc_dout_valid[i]),

            .m_axi_awid                     (m_axi_awid[i]),
            .m_axi_awaddr                   (m_axi_awaddr[i]),
            .m_axi_awlen                    (m_axi_awlen[i]),
            .m_axi_awsize                   (m_axi_awsize[i]),
            .m_axi_awburst                  (m_axi_awburst[i]),
            .m_axi_awvalid                  (m_axi_awvalid[i]),
            .m_axi_awready                  (m_axi_awready[i]),
            .m_axi_wdata                    (m_axi_wdata[i]),
            .m_axi_wstrb                    (m_axi_wstrb[i]),
            .m_axi_wlast                    (m_axi_wlast[i]),
            .m_axi_wvalid                   (m_axi_wvalid[i]),
            .m_axi_wready                   (m_axi_wready[i]),
            .m_axi_bid                      (m_axi_bid[i]),
            .m_axi_bresp                    (m_axi_bresp[i]),
            .m_axi_bvalid                   (m_axi_bvalid[i]),
            .m_axi_bready                   (m_axi_bready[i]),
            .m_axi_arid                     (m_axi_arid[i]),
            .m_axi_araddr                   (m_axi_araddr[i]),
            .m_axi_arlen                    (m_axi_arlen[i]),
            .m_axi_arsize                   (m_axi_arsize[i]),
            .m_axi_arburst                  (m_axi_arburst[i]),
            .m_axi_arvalid                  (m_axi_arvalid[i]),
            .m_axi_arready                  (m_axi_arready[i]),
            .m_axi_rid                      (m_axi_rid[i]),
            .m_axi_rdata                    (m_axi_rdata[i]),
            .m_axi_rresp                    (m_axi_rresp[i]),
            .m_axi_rlast                    (m_axi_rlast[i]),
            .m_axi_rvalid                   (m_axi_rvalid[i]),
            .m_axi_rready                   (m_axi_rready[i])
        );
    end
endgenerate
endmodule



`endif
