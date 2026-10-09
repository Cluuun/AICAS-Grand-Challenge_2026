//*****************************************************************
// 'Empty' stub module for FPGA verification.
//
// This module has the exact same interface as 'accumulator'
// but all outputs are tied to 0, simulating an inactive state.
//*****************************************************************
module accumulator_empty #(
    parameter                 AXI_DATA_WIDTH   = 128                                    ,
    parameter                 PE_WIDTH         = 16                                     ,
    parameter                 ACCU_DATA_WIDTH  = 32                                     ,  //INT32
    parameter                 SPM_DATA_WIDTH   = 8                                      ,  //INT8
    parameter                 SPM_ADDR_WIDTH   = 20                                     ,

    parameter                 ACCU_SIZE        = 1 << 20                                ,
    parameter                 RD_PORTS         = 1                                      ,
    parameter                 WR_PORTS         = 1                                      ,

    parameter                 RF_DATA_WIDTH    = 64                                     ,
    parameter integer         DISABLE_ACC_INT32_TO_INT8 = 1                            ,

    parameter  signed [31:0]  ALPHA_1          = 32'd6553                               ,
    parameter  signed [31:0]  ALPHA_2          = 32'd13107                              ,
    parameter  signed [31:0]  ALPHA_3          = 32'd655                                ,

    localparam                ACCU_ADDR_WIDTH  = $clog2(ACCU_SIZE)                      ,  //20 bit address
    localparam                PE_IDX           = $clog2(PE_WIDTH)                       ,  //5 bit
    localparam                PE_DATA_SIZE     = $clog2(ACCU_DATA_WIDTH/8)              ,  //2 bit
    localparam                PE_DATA_SIZE_SPM = $clog2(SPM_DATA_WIDTH/8)
)(
    input  logic                                        clk,
    input  logic                                        rst_n,

    //----------------------------------  
    // Accumulator control signals
    //----------------------------------
    input  logic                                        cfg_mvin_isbias,                // 0:mvin to sram, 1:mvin to bias register

    input  logic               [1:0]                    cfg_compute_optype,             //operation type ,00 for GEMM,01 for CONV,10 for GEMV,11 for ADD
    input  logic                                        cfg_compute_accout_dest,        //write back destination for single instruction accumulate,0:SPM,1:ACC
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_compute_output_scale,       //scale of output data from PE in OS dataflow
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_compute_output_scale_shift, //scale shift of output data from PE in OS dataflow
    // quantization from accu to spm

    input  logic               [RF_DATA_WIDTH/8-1:0]    sa_input_b_col_num,             // column number of input matrix B(OS) in systolic array

    input  logic               [RF_DATA_WIDTH/2-1:0]    cfg_accu_biaspsum_addr,         //bias/psum data  start address in ACC
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_accu_biaspsum_stride,       //bias/psum data  stride in ACC
    input  logic               [RF_DATA_WIDTH/8-1:0]    cfg_accu_biaspsum_width,        //bias/psum data width
    input  logic               [RF_DATA_WIDTH/8-1:0]    cfg_accu_biaspsum_height,       //bias/psum data height
    input  logic               [RF_DATA_WIDTH/2-1:0]    cfg_accu_output_addr,           // output data in ACC/SPM start address
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_accu_output_stride,         // output data in ACC/SPM stride
    input  logic                                        cfg_accu_relu,                  // 0 for no relu,1 for relu
    input  logic               [2:0]                    cfg_accu_relu_type,             // 000 for relu, 001 for relu6, 010 for leaky relu a=0.1
                                                                                        // 011 for leaky relu a=0.2 ,100 for leaky relu a=0.01
    input  logic                                        cfg_accu_isaccu,                // 0:the sa result direct save to ACC/SPM;1:save to ACC/SPM after accumulate
    input  logic                                        cfg_accu_isbias,                // 0:part sum, 1:bias

    input  logic               [RF_DATA_WIDTH/2-1:0]    matadd_input_a_addr,            //input matrix A start address in ACC
    input  logic               [RF_DATA_WIDTH/2-1:0]    matadd_input_b_addr,            //input matrix B start address in ACC
    input  logic               [RF_DATA_WIDTH/8-1:0]    matadd_input_col_num,           //input matrix A/B column number
    input  logic               [RF_DATA_WIDTH/8-1:0]    matadd_input_row_num,           //input matrix A/B row number
    input  logic               [RF_DATA_WIDTH/2-1:0]    matadd_output_addr,             //output matrix address in ACC

    //----------------------------------  
    // Systolic Array → Accumulator  INT32 → INT32
    //----------------------------------

    input  logic                                        accadd_start,
    output logic                                        accadd_busy,
    output logic                                        accadd_done,

    input  logic                                        sa_wr_en,                       //sa_acc_valid
    input  logic [PE_WIDTH-1:0]                         sa_wr_mask,
    input  logic [PE_WIDTH-1:0][ACCU_DATA_WIDTH-1:0]    sa_wr_data,

    //----------------------------------  
    // DMA ↔ Accumulator  INT32 ↔ INT32
    //----------------------------------

    input  logic                                        dma_accu_wr_en,
    input  logic               [ACCU_ADDR_WIDTH-1:0]    dma_accu_wr_addr,
    input  logic [PE_WIDTH-1:0]                         dma_accu_wr_mask,
    input  logic [PE_WIDTH-1:0][ACCU_DATA_WIDTH-1:0]    dma_accu_wr_data,               // 32 × 32, res_add: add mask
    input  logic                                        dma_mvin_resp_done,             // done signal for mvin instructions
    input  logic                                        dma_mvout_resp_done,            // done signal for mvout instructions

    input  logic                                        dma_accu_rd_en,
    input  logic               [ACCU_ADDR_WIDTH-1:0]    dma_accu_rd_addr,
    output logic [PE_WIDTH-1:0][ACCU_DATA_WIDTH-1:0]    dma_accu_rd_data,
    output logic                                        dma_accu_rd_valid,

    input  logic               [1:0]                    cfg_mvout_output_precision,
    input  logic                                        cfg_mvout_per_channel,
    input  logic               [RF_DATA_WIDTH/2-1:0]    mvout_col_num,
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_mvout_output_scale,
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_mvout_output_scale_shift,

    //----------------------------------  
    // Accumulator → SPM  INT32 → INT8
    //----------------------------------

    output logic                                        accu_spm_wr_en,
    output logic               [SPM_ADDR_WIDTH-1:0]     accu_spm_wr_addr,
    output logic [PE_WIDTH-1:0]                         accu_spm_wr_mask,
    output logic [PE_WIDTH-1:0][SPM_DATA_WIDTH-1:0]     accu_spm_wr_data,

    //----------------------------------  
    // Matadd
    //----------------------------------

    input  logic                                        matadd_req_en,
    output logic                                        matadd_busy,
    output logic                                        matadd_comp_done
);


    // ------------------------------------- 
    // Tie all outputs to a default, inactive state (0)
    // ------------------------------------- 

    // Systolic Array Outputs
    assign accadd_busy = 1'b0;
    assign accadd_done = 1'b0; // Signal completion instantly (can be 1'b0 too, depending on test harness need)

    // DMA Read Data Output
    assign dma_accu_rd_data  = '0;
    assign dma_accu_rd_valid = 1'b0;

    // Accumulator -> SPM Outputs
    assign accu_spm_wr_en   = 1'b0;
    assign accu_spm_wr_addr = '0;
    assign accu_spm_wr_mask = '0;
    assign accu_spm_wr_data = '0;

    // Matadd Outputs
    assign matadd_busy = 1'b0;
    assign matadd_comp_done = 1'b0; // Signal completion instantly

endmodule
