//////////////////////////////////////////////////////////////////////////////////
// Copyright by FuxionLab 
//
// Designer     : Sihao Fu & Lanqi Ma
// Create Date  : 2024/10/29
// Update Date  : 2025/09/10
// Project Name : T_NPU
// File Name    : inst_ctrl.sv 
//
// Description  : v1:Decode ROCC custom instructions from Core, and set config into Config Regfile and send control signals to DMA / Systolic Array / SFU.
//              : v2:Change ROCC to AXI-Lite
// Revision:  
// Revision 2.0 - File Created
// Additional Comments:     
//
//////////////////////////////////////////////////////////////////////////////////

`ifndef INST_CTRL_SV
`define INST_CTRL_SV

module inst_ctrl 
    import npu_config_pkg::*;
#(
    parameter int RF_DATA_WIDTH        = npu_config_pkg::RF_DATA_WIDTH,
    parameter int MMIO_AXI_DATA_WIDTH  = npu_config_pkg::MMIO_AXI_DATA_WIDTH,
    parameter int MMIO_AXI_ADDR_WIDTH  = npu_config_pkg::MMIO_AXI_ADDR_WIDTH,
    parameter int MMIO_REG_NUMBER      = npu_config_pkg::MMIO_REG_NUMBER,
    parameter int MMIO_REG_WOEND       = npu_config_pkg::MMIO_REG_WOEND,
    parameter int MMIO_REG_ROSTART     = npu_config_pkg::MMIO_REG_ROSTART,
    parameter int DMA_NUM              = 1,
    parameter int INPUT_WIDTH_MAX      = npu_config_pkg::INPUT_WIDTH_MAX,
    parameter int INPUT_HEIGHT_MAX     = npu_config_pkg::INPUT_HEIGHT_MAX,
    parameter int ARRAY_WIDTH          = npu_config_pkg::ARRAY_WIDTH,
    parameter int ARRAY_HEIGHT         = npu_config_pkg::ARRAY_HEIGHT,
    parameter int ENABLE_PROFILE_COUNTERS = 1,
    parameter int ENABLE_DETAILED_ERROR_STATUS = 1
)
( 
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

    //-------------------------------------             
    // DMA Control Signals              
    //-------------------------------------             
    output logic    [RF_DATA_WIDTH/2-1:0]           mvin_dram_addr               ,   // DDR physical address for DMA request
    output logic    [RF_DATA_WIDTH/2-1:0]           mvin_row_num                 ,   // number of rows for DMA request
    output logic    [RF_DATA_WIDTH/2-1:0]           mvin_sram_addr               ,   // ScratchPad physical address for DMA request
    output logic    [RF_DATA_WIDTH/2-1:0]           mvin_col_num                 ,    //number of column for DMA request
    output logic    [RF_DATA_WIDTH/2-1:0]           mvout_dram_addr              ,   // DDR physical address for DMA request
    output logic    [RF_DATA_WIDTH/2-1:0]           mvout_row_num                ,   // number of rows for DMA request
    output logic    [RF_DATA_WIDTH/2-1:0]           mvout_sram_addr              ,   // ScratchPad physical address for DMA request
    output logic    [RF_DATA_WIDTH/2-1:0]           mvout_col_num                ,    //number of column for DMA request

    output logic    [1:0]                           cfg_mvin_input_type         ,   // data type of the mvin data, 00 for input feature/A, 01 for weight/B
                                                                                    // 10 for bias
    output logic    [1:0]                           cfg_mvout_output_type       ,   // data type of the mvout data, 00 for final output, 01 for part sum                                                                               
    output logic    [1:0]                           cfg_mvin_input_precision    ,   //data precision,00/01/10/11 for int4/int8/fp16/fp32
                                                                                    // for bias :int16/int32/fp16/fp32
    output logic    [1:0]                           cfg_mvout_output_precision  ,
    output logic                                    cfg_mvin_is_quant           ,    //mvin/mvout is/not quant
    output logic                                    cfg_mvout_is_quant          ,
    output logic                                    cfg_mvin_dest               ,    //mvin data destination ,0 for SPM ,1 for ACC
    output logic                                    cfg_mvout_source            ,    //mvout data source,0 for SPM ,1 for ACC
    output logic                                    cfg_mvout_per_channel       ,
    output logic                                    cfg_mvin_isbias             ,   // marks ACC MVIN payload as bias data
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvin_sram_stride        ,   // Stride for mvin SPM address increment
    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvin_dram_stride        ,   // Stride for mvin DRAM address increment
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvout_sram_stride       ,   // Stride for mvout SPM address increment
    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvout_dram_stride       ,   // Stride for mvout DRAM address increment
    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvin_input_zeropoint    ,   // for matrix add,the input quant zero point
    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvout_output_zeropoint  ,   // for matrix add,the output quant zero point
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvin_input_scale        ,   // for matrix add,the input quant scale
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvout_output_scale      ,   // for matrix add,the output quant scale
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvin_input_scale_shift  ,   // for matrix add,the input quant scale shift
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvout_output_scale_shift,   // for matrix add,the output quant scale shift

    // output logic                                    dma_mvin_req_en             ,   // enable signal for mvin instructions, generate AXI read req when high
    // output logic                                    dma_mvout_req_en            ,   // enable signal for mvout instructions, generate AXI write req when high
    // input  logic                                    dma_mvin_busy               ,
    // input  logic                                    dma_mvout_busy              ,
    // input  logic                                    dma_mvin_resp_done          ,
    // input  logic                                    dma_mvout_resp_done         ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvin_dram_addr       ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvin_row_num         ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvin_sram_addr       ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvin_col_num         ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvout_dram_addr      ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvout_row_num        ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvout_sram_addr      ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_mvout_col_num        ,
    output logic    [DMA_NUM-1:0][1:0]                 dma_cfg_mvin_input_type  ,
    output logic    [DMA_NUM-1:0][1:0]                 dma_cfg_mvout_output_type,
    output logic    [DMA_NUM-1:0][1:0]                 dma_cfg_mvin_input_precision,
    output logic    [DMA_NUM-1:0][1:0]                 dma_cfg_mvout_output_precision,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvin_is_quant    ,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvout_is_quant   ,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvin_dest        ,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvout_source     ,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvout_per_channel,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvin_isbias      ,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvin_scale_target,
    output logic    [DMA_NUM-1:0]                      dma_cfg_mvin_stream_fifo_fill,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvin_sram_stride ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvin_dram_stride ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvout_sram_stride,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvout_dram_stride,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvin_input_zeropoint,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/2-1:0] dma_cfg_mvout_output_zeropoint,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvin_input_scale ,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvout_output_scale,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvin_input_scale_shift,
    output logic    [DMA_NUM-1:0][RF_DATA_WIDTH/4-1:0] dma_cfg_mvout_output_scale_shift,
    output logic    [DMA_NUM-1:0]                      dma_mvin_req_en      ,
    output logic    [DMA_NUM-1:0]                      dma_mvout_req_en     ,
    input  logic    [DMA_NUM-1:0]                      dma_mvin_busy_status        ,
    input  logic    [DMA_NUM-1:0]                      dma_mvout_busy_status       ,
    input  logic    [DMA_NUM-1:0]                      dma_mvin_resp_done_status   ,
    input  logic    [DMA_NUM-1:0]                      dma_mvout_resp_done_status  ,

    //------------------------------------          
    // Systolic Array Control Signals           
    //------------------------------------          

    output logic                                    cfg_compute_dataflow        ,   // 0 for weight stationary, 1 for output stationary
    output logic    [1:0]                           cfg_compute_padding_left    ,   // Number of left padding data columns for the input
    output logic    [1:0]                           cfg_compute_padding_right   ,   // Number of right padding data columns for the input
    output logic    [1:0]                           cfg_compute_padding_top     ,   // Number of top padding data columns for the input
    output logic    [1:0]                           cfg_compute_padding_bottom  ,   // Number of bottom padding data columns for the input
    output logic    [1:0]                           cfg_compute_padding_mode    ,   // padding data type, 00 for zero padding, 01 for reflect padding,
                                                                                        // 10 for replicate padding, 11 for circular padding, only zero padding is supported currently    
    output logic    [3:0]                           cfg_compute_weight_shape    ,    // the width/height for weight, 0 represent width/height is 1
    output logic    [1:0]                           cfg_compute_weight_stride   ,    // the stride of weight , 0 represent 1
    output logic    [4:0]                           cfg_compute_weight_dilation ,   // the dilation of weight ,0 represent 1
    output logic                                    cfg_compute_is_groupconv    ,   //  0 for groups=1; 1 for groups = C_in(depthwise)
    output logic    [1:0]                           cfg_compute_int_type        ,   // SA datatype; for GEMV: 00=W4A16, 01=W8A16, 1x reserved/unsupported
    output logic    [1:0]                           cfg_compute_optype          ,   // operation type ,00 for GEMM,01 for CONV,10 for GEMV,11 for ADD
    output logic                                    cfg_compute_accout_dest     ,   // write back destination for single instruction accumulate,0:SPM,1:ACC
    output logic                                    cfg_compute_asymmetric_activations, // only for GEMM: 1 means subtract 128 from input A before SA
    output logic    [RF_DATA_WIDTH-1:0]             cfg_decode_flow             ,   // GEMV decode flow recipe fields
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_inputa_zeropoint,   //quant zero point for input feature/input matrix A 
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_inputb_zeropoint,   //quant zero point for input weight/input matrix B
    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_compute_output_zeropoint,   //quant zero point when writing the accumulated output from Acc back to SPM​
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_output_scale    ,   // scale of output data from PE in OS dataflow
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_compute_output_scale_shift ,   // scale shift of output data from PE in OS dataflow

    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_accu_biaspsum_addr      ,   //bias/psum data  start address in ACC
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_accu_biaspsum_stride    ,   //bias/psum data  stride in ACC
    output logic    [RF_DATA_WIDTH/8-1:0]           cfg_accu_biaspsum_width     ,   //bias/psum data width
    output logic    [RF_DATA_WIDTH/8-1:0]           cfg_accu_biaspsum_height    ,   //bias/psum data height
    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_accu_output_addr        ,   // output data in ACC/SPM start address
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_accu_output_stride      ,   // output data in ACC/SPM stride
    output logic                                    cfg_accu_isaccu             ,   // 0:the sa result direct save to ACC/SPM;1:save to ACC/SPM after accumulate
    output logic                                    cfg_accu_relu               ,   // 0 for no relu,1 for relu
    output logic    [2:0]                           cfg_accu_relu_type          ,   // 000 for relu, 001 for relu6, 010 for leaky relu a=0.1
                                                                                    // 011 for leaky relu a=0.2 ,100 for leaky relu a=0.01
    output logic                                    cfg_accu_isbias             ,   // 0:part sum, 1:bias

    output logic    [RF_DATA_WIDTH/2-1:0]           sa_input_a_spm_addr         ,   // address for input feature(WS)/input matrix A(OS) data in SPM   
    output logic    [$clog2(INPUT_WIDTH_MAX)-1:0]   sa_input_a_col_num          ,   // column number of input feature(WS)/input matrix A(OS) in systolic array
    output logic    [$clog2(ARRAY_HEIGHT)-1:0]      sa_input_a_row_num          ,   // row number of input feature(WS)/input matrix A(OS) in systolic array
    output logic    [RF_DATA_WIDTH/4-1:0]           sa_input_a_stride           ,   //stride for input feature(WS)/input matrix A(OS) data, for GEMM not valid(default W)
                                                                                    //for dilated conv(not support now),default is W-1

    output logic    [RF_DATA_WIDTH/2-1:0]           sa_input_b_spm_addr         ,   // address for input matrix B(OS) data in SPM
    output logic    [$clog2(ARRAY_WIDTH)-1:0]       sa_input_b_col_num          ,   // column number of input matrix B(OS) in systolic array
    output logic    [$clog2(INPUT_HEIGHT_MAX)-1:0]  sa_input_b_row_num          ,   // row number of input matrix B(OS) in systolic array
    output logic    [RF_DATA_WIDTH/4-1:0]           sa_input_b_stride           ,   // same as a_stride,when unit matrix is B:stride=32,for conv:invalid
    
    output logic                                    sa_req_en                   ,
    input  logic                                    sa_busy                     ,
    input  logic                                    sa_comp_done                ,
    
    //------------------------------------    
    // MATADD Control Signals    
    //------------------------------------    

    output logic   [RF_DATA_WIDTH/2-1:0]           matadd_input_a_addr          , //input matrix A start address in ACC
    output logic   [RF_DATA_WIDTH/2-1:0]           matadd_input_b_addr          , //input matrix B start address in ACC
    output logic   [RF_DATA_WIDTH/8-1:0]           matadd_input_col_num         , //input matrix A/B column number
    output logic   [RF_DATA_WIDTH/8-1:0]           matadd_input_row_num         , //input matrix A/B row number
    output logic   [RF_DATA_WIDTH/2-1:0]           matadd_output_addr           , //output matrix start address in SPM

    output logic                                    matadd_req_en               ,
    input  logic                                    matadd_busy                 ,
    input  logic                                    matadd_comp_done            ,
    
    //------------------------------------    
    // MATVEC Control Signals          
    //------------------------------------    
    output logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_mat_addr        , //input matrix  start address in SPM
    output logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_vec_addr        , //input vector  start address in SPM
    output logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_act_scale_addr  , //input act scale start address in SPM
    output logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_act_scale2_addr , //second input act scale start address in SPM
    output logic   [RF_DATA_WIDTH/4-1:0]           matvec_cache_cell_idx        , //KV cache window cell index for GEMV decode writer
    output logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_act_group_stride_bytes, //activation group stride in bytes
    output logic   [7:0]                            matvec_act_frac_cfg         , //GEMV activation fixed-point frac cfg
    output logic   [RF_DATA_WIDTH/4-1:0]           matvec_input_mat_width       , //input matrix width
    output logic   [RF_DATA_WIDTH/4-1:0]           matvec_input_mat_height      , //input matrix height
    output logic   [RF_DATA_WIDTH/2-1:0]           matvec_input_mat_stride      , //GEMV output byte address
    output logic   [RF_DATA_WIDTH/4-1:0]           matvec_input_vec_stride      , //input vector stride
    output logic                                    matvec_weight_stream_mode    ,
    output logic                                    matvec_act_scale_enable      ,
    output logic                                    matvec_act_scale2_enable     ,

    output logic                                    matvec_req_en               ,
    input  logic                                    matvec_busy                 ,
    input  logic                                    matvec_comp_done            ,
    input  logic                                    kv_scale_commit_valid       ,
    input  logic                                    kv_scale_commit_is_v        ,
    input  logic [4:0][15:0]                       kv_scale_commit_values      ,
    input  logic [7:0]                              kv_scale_commit_count       ,
    
    
    //------------------------------------          
    // SFU Control Signals          
    //------------------------------------          

    output logic    [5:0]                           cfg_sfu_op                  ,   // operation of SFU
    output logic    [1:0]                           cfg_sfu_int_type            ,   // datatype range of SFU input data, 00 for int8, 01 for int16, 10 for int32, 11 for int64  
    output logic                                    cfg_sfu_is_quant            ,   // is / not quant process
    output logic                                    cfg_trans_out_ispad_row     ,   //for transpose,output row is/not need padding 0 to align to 32
    output logic                                    cfg_trans_out_ispad_col     ,   //for transpose,output col is/not need padding 0 to align to 32
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_zeropoint    ,  // zero point of SFU output data
    output logic    [RF_DATA_WIDTH/2-1:0]           cfg_sfu_input_zeropoint     ,  // zero point of SFU input data
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_input_scale         ,   // scale of SFU input data 
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_scale        ,   // scale of SFU output data 
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_input_scale_shift   ,   // scale shift of SFU input data 
    output logic    [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_scale_shift  ,   // scale shift of SFU output data 

    output logic    [RF_DATA_WIDTH/2-1:0]           sfu_input_sram_addr         ,   // address for input data in SPM /ACC  
    output logic    [RF_DATA_WIDTH/4-1:0]           sfu_input_col_num           ,   // width of input matrix data in SPM,real col=col num+1
    output logic    [RF_DATA_WIDTH/4-1:0]           sfu_input_row_num           ,   // height of input matrix data in SPM,real row=row num+1
    output logic    [RF_DATA_WIDTH/2-1:0]           sfu_output_spm_addr         ,   // address for output data in SPM
    //output logic                                    sfu_input_source            ,   // input data source ,0 for SPM ,1 for ACC
    
    

    output logic                                    sfu_req_en                  ,
    input  logic                                    sfu_busy                    ,
    input  logic                                    sfu_comp_done               ,
    //------------------------------------          
    // Interrupt signals         
    //------------------------------------ 
    output logic                                    irq                              
);
    localparam logic [1:0] AXI_RESP_OKAY   = 2'b00;
    localparam logic [1:0] AXI_RESP_SLVERR = 2'b10;

    localparam logic [15:0] REG_IAR          = 16'h0100;
    localparam logic [15:0] REG_MER          = 16'h0108;
    localparam logic [15:0] REG_IER          = 16'h0110;
    localparam logic [15:0] REG_ISR          = 16'h0118;
    localparam logic [15:0] REG_IPR          = 16'h0120;
    localparam logic [15:0] REG_GLOBAL_CLEAR = 16'h0128;
    localparam logic [15:0] REG_PROFILE0     = 16'h0130;

    localparam logic [15:0] REG_GEMV_DESC0   = 16'h0300;
    localparam logic [15:0] REG_GEMV_DESC1   = 16'h0308;
    localparam logic [15:0] REG_GEMV_DESC2   = 16'h0310;
    localparam logic [15:0] REG_GEMV_DESC3   = 16'h0318;
    localparam logic [15:0] REG_GEMV_CTRL    = 16'h0320;
    localparam logic [15:0] REG_GEMV_STATUS  = 16'h0328;
    // DESC2[58] enables W8 PV fixed-stride row tiles; DESC4[31:0]=stride bytes,
    // DESC4[47:32]=capacity tokens, DESC4[63:48] reserved/ignored.
    localparam logic [15:0] REG_GEMV_DESC4   = 16'h0330;
    // DESC5[31:0]=activation group stride bytes for grouped PV/GEMV,
    // DESC5[63:32] reserved.
    localparam logic [15:0] REG_GEMV_DESC5   = 16'h0338;
    // DESC6[31:0]=weight scale DMA base. DESC2[59] selects cached SPM
    // weight mode instead of stream FIFO weight mode.
    localparam logic [15:0] REG_GEMV_DESC6   = 16'h0370;
    // DESC7[31:0]=second act scale DMA base. DESC2[60] enables the second
    // serialized act-scale pass for stream GEMV.
    localparam logic [15:0] REG_GEMV_DESC7   = 16'h0378;

    localparam logic [15:0] REG_MLP_DESC0    = 16'h0340;
    localparam logic [15:0] REG_MLP_DESC1    = 16'h0348;
    localparam logic [15:0] REG_MLP_DESC2    = 16'h0350;
    localparam logic [15:0] REG_MLP_DESC3    = 16'h0358;
    localparam logic [15:0] REG_MLP_CTRL     = 16'h0360;
    localparam logic [15:0] REG_MLP_STATUS   = 16'h0368;

    localparam logic [15:0] REG_ATTN_DESC0   = 16'h0380;
    localparam logic [15:0] REG_ATTN_DESC1   = 16'h0388;
    localparam logic [15:0] REG_ATTN_DESC2   = 16'h0390;
    localparam logic [15:0] REG_ATTN_DESC3   = 16'h0398;
    localparam logic [15:0] REG_ATTN_DESC4   = 16'h03a0;
    localparam logic [15:0] REG_ATTN_DESC5   = 16'h03a8;
    localparam logic [15:0] REG_ATTN_DESC6   = 16'h03b0;
    localparam logic [15:0] REG_ATTN_CTRL    = 16'h03b8;
    localparam logic [15:0] REG_ATTN_STATUS  = 16'h03c0;

    localparam logic [15:0] REG_DBG_MVIN_DESC0   = 16'h0400;
    localparam logic [15:0] REG_DBG_MVIN_DESC1   = 16'h0408;
    localparam logic [15:0] REG_DBG_MVIN_CTRL    = 16'h0410;
    localparam logic [15:0] REG_DBG_MVIN_STATUS  = 16'h0418;
    localparam logic [15:0] REG_DBG_MVOUT_DESC0  = 16'h0420;
    localparam logic [15:0] REG_DBG_MVOUT_DESC1  = 16'h0428;
    localparam logic [15:0] REG_DBG_MVOUT_CTRL   = 16'h0430;
    localparam logic [15:0] REG_DBG_MVOUT_STATUS = 16'h0438;
    localparam logic [15:0] REG_DBG_GEMV_DESC0   = 16'h0440;
    localparam logic [15:0] REG_DBG_GEMV_DESC1   = 16'h0448;
    localparam logic [15:0] REG_DBG_GEMV_DESC2   = 16'h0450;
    localparam logic [15:0] REG_DBG_GEMV_DESC3   = 16'h0458;
    localparam logic [15:0] REG_DBG_GEMV_DESC4   = 16'h0460;
    localparam logic [15:0] REG_DBG_GEMV_CTRL    = 16'h0468;
    localparam logic [15:0] REG_DBG_GEMV_STATUS  = 16'h0470;
    localparam logic [15:0] REG_KV_K_SCALE0      = 16'h04a0;
    localparam logic [15:0] REG_KV_K_SCALE1      = 16'h04a8;
    localparam logic [15:0] REG_KV_V_SCALE0      = 16'h04b0;
    localparam logic [15:0] REG_KV_V_SCALE1      = 16'h04b8;

    localparam logic [15:0] REG_DECODE_IAR_ALIAS = 16'h00b8;
    localparam logic [15:0] REG_DECODE_MER_ALIAS = 16'h00c0;
    localparam logic [15:0] REG_DECODE_IER_ALIAS = 16'h00c8;
    localparam logic [15:0] REG_DECODE_ISR_ALIAS = 16'h00d0;
    localparam logic [15:0] REG_DECODE_IPR_ALIAS = 16'h00d8;

    localparam logic [15:0] REG_GENERIC_MAGIC   = 16'hff00;
    localparam logic [15:0] REG_GENERIC_VERSION = 16'hff08;
    localparam logic [15:0] REG_GENERIC_MODE    = 16'hff10;
    localparam logic [15:0] REG_GENERIC_CAPS    = 16'hff18;
    localparam logic [15:0] REG_GENERIC_CONTROL = 16'hff20;
    localparam logic [15:0] REG_GENERIC_STATUS  = 16'hff28;
    localparam logic [15:0] REG_GENERIC_ERROR   = 16'hff30;
    localparam logic [15:0] REG_GENERIC_MAGIC_ALIAS   = 16'h0f00;
    localparam logic [15:0] REG_GENERIC_VERSION_ALIAS = 16'h0f08;
    localparam logic [15:0] REG_GENERIC_MODE_ALIAS    = 16'h0f10;
    localparam logic [15:0] REG_GENERIC_CAPS_ALIAS    = 16'h0f18;
    localparam logic [15:0] REG_GENERIC_CONTROL_ALIAS = 16'h0f20;
    localparam logic [15:0] REG_GENERIC_STATUS_ALIAS  = 16'h0f28;
    localparam logic [15:0] REG_GENERIC_ERROR_ALIAS   = 16'h0f30;

    localparam logic [31:0] GENERIC_MAGIC       = 32'h4e505547;
    localparam logic [31:0] GENERIC_ABI_VERSION = 32'd2;
    localparam logic [31:0] GENERIC_MODE_DECODE = 32'd2;
    localparam logic [31:0] GENERIC_CAPS        = 32'h0000_00b2;

    localparam int API_COUNT       = 6;
    localparam int API_GEMV_BLOCK  = 0;
    localparam int API_MLP         = 1;
    localparam int API_ATTN        = 2;
    localparam int API_DBG_MVIN    = 3;
    localparam int API_DBG_MVOUT   = 4;
    localparam int API_DBG_GEMV    = 5;
    localparam int IRQ_ERROR       = 7;

    localparam logic [3:0] ERR_NONE              = 4'd0;
    localparam logic [3:0] ERR_ILLEGAL_SHAPE     = 4'd1;
    localparam logic [3:0] ERR_START_WHILE_BUSY  = 4'd2;
    localparam logic [3:0] ERR_ALIGNMENT_ERROR   = 4'd7;
    localparam logic [3:0] ERR_UNSUPPORTED_MODE  = 4'd8;

    localparam logic [31:0] GEMV_TILE_ELEMS      = 32'd128;
    localparam logic [31:0] GEMV_ROW_TILE_ELEMS  = 32'd32;
    localparam logic [31:0] GEMV_LINE_BYTES      = 32'd64;
    localparam logic [31:0] GEMV_MVIN_ALIGN      = 32'd256;
    localparam logic [31:0] GEMV_ACT_TILE_BYTES  = 32'd256;
    localparam logic [31:0] GEMV_SCALE_TILE_BYTES = 32'd64;
    localparam logic [31:0] GEMV_SPM_BYTES       = 32'd524288;
    localparam logic [31:0] GEMV_ACT_PAYLOAD_LIMIT = 32'h0000_3f00;
    localparam logic [31:0] GEMV_ROPE_LUT_BASE  = 32'h0000_3f00;
    localparam logic [31:0] GEMV_ROPE_LUT_BYTES = 32'd256;
    localparam logic [31:0] GEMV_PING_BASE       = 32'h0001_0000;
    localparam logic [31:0] GEMV_PONG_BASE       = 32'h0004_0000;
    localparam logic [31:0] GEMV_ACT_BASE        = 32'h0000_0000;
    localparam logic [31:0] GEMV_OUTPUT_BASE     = 32'h0000_0000;
    localparam logic [15:0] GEMV_CACHE_INVALIDATE_TAG = 16'h8000;
    localparam logic [31:0] GEMV_START_CAPACITY =
        ((GEMV_PONG_BASE - GEMV_PING_BASE) < (GEMV_SPM_BYTES - GEMV_PONG_BASE)) ?
        (GEMV_PONG_BASE - GEMV_PING_BASE) : (GEMV_SPM_BYTES - GEMV_PONG_BASE);
    localparam logic [31:0] GEMV_WEIGHT_TILE_BYTES_MODE0 =
        GEMV_SCALE_TILE_BYTES + (GEMV_ROW_TILE_ELEMS * 32'd64);
    localparam logic [31:0] GEMV_WEIGHT_TILE_BYTES_MODE1 =
        GEMV_SCALE_TILE_BYTES + (GEMV_ROW_TILE_ELEMS * 32'd128);
    localparam logic [31:0] GEMV_WEIGHT_PAYLOAD_TILE_BYTES_MODE0 =
        GEMV_ROW_TILE_ELEMS * 32'd64;
    localparam logic [31:0] GEMV_WEIGHT_PAYLOAD_TILE_BYTES_MODE1 =
        GEMV_ROW_TILE_ELEMS * 32'd128;
    localparam logic [31:0] GEMV_MAX_MVIN_BYTES = 32'd4194240;
`ifdef GEMV_TARGET_ROW_TILES_PER_BLOCK_1
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd1;
`elsif GEMV_TARGET_ROW_TILES_PER_BLOCK_2
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd2;
`elsif GEMV_TARGET_ROW_TILES_PER_BLOCK_3
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd3;
`elsif GEMV_TARGET_ROW_TILES_PER_BLOCK_4
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd4;
`elsif GEMV_TARGET_ROW_TILES_PER_BLOCK_5
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd5;
`elsif GEMV_TARGET_ROW_TILES_PER_BLOCK_6
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd6;
`elsif GEMV_TARGET_ROW_TILES_PER_BLOCK_32
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd32;
`else
    localparam logic [31:0] GEMV_TARGET_ROW_TILES_PER_BLOCK = 32'd4;
`endif
    localparam int GEMV_TARGET_ROW_TILES_PER_BLOCK_INT = int'(GEMV_TARGET_ROW_TILES_PER_BLOCK);

    localparam int DMA_SEL_WIDTH = (DMA_NUM > 1) ? $clog2(DMA_NUM) : 1;
    localparam int ACC_DMA_IDX   = DMA_NUM - 1;
    localparam logic [DMA_SEL_WIDTH-1:0] ACC_DMA_ID = ACC_DMA_IDX;
    localparam bit PROFILE_COUNTERS_ON = (ENABLE_PROFILE_COUNTERS != 0);
    localparam bit DETAILED_ERROR_STATUS_ON = (ENABLE_DETAILED_ERROR_STATUS != 0);

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_GEMV_MVIN_ACT,
        ST_GEMV_WAIT_ACT,
        ST_GEMV_MVIN_ACT_SCALE,
        ST_GEMV_WAIT_ACT_SCALE,
        ST_GEMV_MVIN_ACT_SCALE2,
        ST_GEMV_WAIT_ACT_SCALE2,
        ST_GEMV_MVIN_ROPE_LUT,
        ST_GEMV_WAIT_ROPE_LUT,
        ST_GEMV_MVIN_WEIGHT_SCALE,
        ST_GEMV_WAIT_WEIGHT_SCALE,
        ST_GEMV_MVIN_BLOCK0,
        ST_GEMV_WAIT_BLOCK0,
        ST_GEMV_WAIT_BLOCK_PREP,
        ST_GEMV_START_BLOCK,
        ST_GEMV_WAIT_BLOCK,
        ST_GEMV_MVOUT,
        ST_GEMV_WAIT_MVOUT
    } engine_state_t;

    typedef struct packed {
        logic [15:0] row_tile_base;
        logic [31:0] block_index;
        logic [31:0] weight_offset;
        logic        buf_select;
    } gemv_block_seed_t;

    typedef struct packed {
        logic [31:0] row_base;
        logic [31:0] block_index;
        logic [31:0] weight_offset;
        logic [31:0] current_rows;
        logic [31:0] next_row_base;
        logic [31:0] next_rows;
        logic [5:0]  current_row_tiles;
        logic [5:0]  next_row_tiles;
        logic        has_next;
        logic        buf_select;
    } gemv_block_prep_a_t;

    typedef struct packed {
        logic [31:0] row_base;
        logic [31:0] block_index;
        logic [31:0] weight_offset;
        logic [31:0] current_rows;
        logic [31:0] next_row_base;
        logic [31:0] current_weight_bytes;
        logic [31:0] next_weight_bytes;
        logic [31:0] current_buf_base;
        logic [31:0] next_buf_base;
        logic [31:0] current_weight_dma_addr;
        logic [31:0] output_addr;
        logic [31:0] output_dram_addr;
        logic [63:0] bypass_flow;
        logic        has_next;
        logic        buf_select;
    } gemv_block_prep_b_t;

    typedef struct packed {
        logic [31:0] row_base;
        logic [31:0] block_index;
        logic [31:0] current_rows;
        logic [31:0] next_row_base;
        logic [31:0] next_weight_offset;
        logic [31:0] current_weight_bytes;
        logic [31:0] next_weight_bytes;
        logic [31:0] current_buf_base;
        logic [31:0] next_buf_base;
        logic [31:0] current_weight_dma_addr;
        logic [31:0] next_weight_dma_addr;
        logic [31:0] output_addr;
        logic [31:0] output_dram_addr;
        logic [63:0] bypass_flow;
        logic        has_next;
        logic        buf_select;
    } gemv_block_t;

    logic [MMIO_AXI_ADDR_WIDTH-1:0]   axi_awaddr;
    logic [MMIO_AXI_DATA_WIDTH-1:0]   axi_wdata;
    logic [MMIO_AXI_DATA_WIDTH/8-1:0] axi_wstrb;
    logic                             axi_awaddr_valid;
    logic                             axi_wdata_valid;
    logic                             write_fire;
    logic                             awaddr_legal;
    logic                             araddr_legal;
    logic [MMIO_AXI_ADDR_WIDTH-1:0]   axi_araddr;

    logic [63:0] gemv_desc0;
    logic [63:0] gemv_desc1;
    logic [63:0] gemv_desc2;
    logic [63:0] gemv_desc3;
    logic [63:0] gemv_desc4;
    logic [63:0] gemv_desc5;
    logic [63:0] gemv_desc6;
    logic [63:0] gemv_desc7;
    logic [63:0] mlp_desc0;
    logic [63:0] mlp_desc1;
    logic [63:0] mlp_desc2;
    logic [63:0] mlp_desc3;
    logic [63:0] attn_desc0;
    logic [63:0] attn_desc1;
    logic [63:0] attn_desc2;
    logic [63:0] attn_desc3;
    logic [63:0] attn_desc4;
    logic [63:0] attn_desc5;
    logic [63:0] attn_desc6;
    logic [63:0] dbg_mvin_desc0;
    logic [63:0] dbg_mvin_desc1;
    logic [63:0] dbg_mvout_desc0;
    logic [63:0] dbg_mvout_desc1;
    logic [63:0] dbg_gemv_desc0;
    logic [63:0] dbg_gemv_desc1;
    logic [63:0] dbg_gemv_desc2;
    logic [63:0] dbg_gemv_desc3;
    logic [63:0] dbg_gemv_desc4;
    logic [API_COUNT-1:0]       api_busy;
    logic [API_COUNT-1:0]       api_done;
    logic [API_COUNT-1:0]       api_error;
    logic [API_COUNT-1:0][3:0]  api_error_code;
    logic [API_COUNT-1:0][31:0] api_accepted_count;
    logic [7:0]                 irq_enable;
    logic [7:0]                 irq_status;
    logic                       master_enable;

    engine_state_t              engine_state;
    logic                       command_active;
    logic                       gemv_start_accept_pending;
    logic [2:0]                 active_api;
    logic [31:0]                cmd_act_dma_base;
    logic [31:0]                cmd_act_scale_dma_base;
    logic [31:0]                cmd_act_scale2_dma_base;
    logic [31:0]                cmd_rope_lut_dma_base;
    logic [31:0]                cmd_weight_dma_base;
    logic [31:0]                cmd_weight_scale_dma_base;
    logic [31:0]                cmd_output_dma_base;
    logic [15:0]                cmd_mat_width;
    logic [15:0]                cmd_mat_height;
    logic [7:0]                 cmd_gemv_mode;
    logic [7:0]                 cmd_output_precision;
    logic [31:0]                cmd_col_tiles;
    logic [31:0]                cmd_act_data_bytes;
    logic [31:0]                cmd_act_payload_bytes;
    logic [31:0]                cmd_act_scale_payload_bytes;
    logic [31:0]                cmd_act_scale2_payload_bytes;
    logic [31:0]                cmd_act_group_stride;
    logic [31:0]                cmd_mvout_rows;
    logic [31:0]                cmd_row_tile_bytes;
    logic [31:0]                cmd_weight_payload_bytes;
    logic [15:0]                cmd_cache_cell;
    logic                       cmd_enable_act_scale;
    logic                       cmd_enable_act_scale2;
    logic                       cmd_fixed_stride_enable;
    logic                       cmd_cached_group_enable;
    logic                       cmd_uses_weight_scale;
    logic                       cmd_uses_rope_lut;
    logic [31:0]                cmd_weight_row_tile_stride_bytes;
    logic [15:0]                cmd_weight_capacity_tokens;
    logic [31:0]                cmd_fixed_stride_active_row_tile_bytes;
    logic [31:0]                cmd_weight_scale_payload_bytes;
    logic [3:0]                 cmd_fixed_stride_segments_issued;
    logic [3:0]                 cmd_fixed_stride_segment_count;
    logic [63:0]                cmd_decode_flow;
    logic [31:0]                cmd_rows_per_block;
    logic [15:0]                cmd_total_row_tiles;
    logic [5:0]                 cmd_row_tiles_per_block;
    logic [31:0]                cmd_row_base;
    logic [31:0]                cmd_block_index;
    logic                       cmd_buf_select;
    logic [1:0]                 cmd_group_count_m1;
    logic [1:0]                 cmd_group_index;
    logic                       cmd_act_slot_select;
    logic                       prefetch_active;
    logic                       prefetch_done_seen;
    logic                       gemv_done_seen;
    logic                       gemv_mvin_done_q;
    logic                       gemv_wait_mvin_done_q;
    logic                       gemv_wait_matvec_done_q;
    logic                       mvout_active;
    logic                       overlap_seen;
    logic                       mvout_overlap_seen;
    logic                       last_pingpong_buffer;
    logic [31:0]                prefetch_wait_cycles;
    logic [31:0]                mvout_wait_cycles;
    logic [31:0]                completed_block_count;


    logic                       dma_mvin_resp_done_any;
    logic                       dma_mvout_resp_done_any;
    logic                       npu_busy_any;
    logic                       gemv_start_s0_valid;
    logic                       gemv_start_s1_valid;
    logic                       gemv_start_s2_valid;
    logic                       gemv_start_s3_valid;
    logic [31:0]                gemv_start_act_dma_base_s1;
    logic [31:0]                gemv_start_act_scale_dma_base_s1;
    logic [31:0]                gemv_start_act_scale2_dma_base_s1;
    logic [31:0]                gemv_start_rope_lut_dma_base_s1;
    logic [31:0]                gemv_start_weight_dma_base_s1;
    logic [31:0]                gemv_start_weight_scale_dma_base_s1;
    logic [31:0]                gemv_start_output_dma_base_s1;
    logic [15:0]                gemv_start_mat_width_s1;
    logic [15:0]                gemv_start_mat_height_s1;
    logic [15:0]                gemv_start_cache_cell_s1;
    logic [7:0]                 gemv_start_gemv_mode_s1;
    logic [7:0]                 gemv_start_output_precision_s1;
    logic                       gemv_start_enable_act_scale_s1;
    logic                       gemv_start_enable_act_scale2_s1;
    logic                       gemv_start_fixed_stride_enable_s1;
    logic                       gemv_start_cached_group_enable_s1;
    logic                       gemv_start_uses_weight_scale_s1;
    logic                       gemv_start_uses_rope_lut_s1;
    logic [31:0]                gemv_start_weight_row_tile_stride_bytes_s1;
    logic [15:0]                gemv_start_weight_capacity_tokens_s1;
    logic [31:0]                gemv_start_act_group_stride_s1;
    logic [63:0]                gemv_start_decode_flow_s1;
    logic                       gemv_start_zero_shape_s1;
    logic                       gemv_start_unsupported_mode_s1;
    logic                       gemv_start_alignment_error_s1;
    logic                       gemv_start_height_overflow_s1;
    logic [31:0]                gemv_start_col_tiles_calc;
    logic [31:0]                gemv_start_act_bytes_calc;
    logic [31:0]                gemv_start_group_act_bytes_calc;
    logic [31:0]                gemv_start_output_rows_calc;
    logic [31:0]                gemv_start_payload_bytes_calc;
    logic [31:0]                gemv_start_act_scale_payload_bytes_calc;
    logic [31:0]                gemv_start_row_tile_bytes_calc;
    logic [31:0]                gemv_start_weight_payload_bytes_calc;
    logic [31:0]                gemv_start_weight_scale_payload_bytes_calc;
    logic [31:0]                gemv_start_fixed_stride_min_stride_calc;
    logic [31:0]                gemv_start_fixed_stride_active_row_tile_bytes_calc;
    logic [31:0]                gemv_start_rows_per_block_calc;
    logic [3:0]                 gemv_start_error_code_calc;
    logic [31:0]                gemv_start_act_dma_base_s2;
    logic [31:0]                gemv_start_act_scale_dma_base_s2;
    logic [31:0]                gemv_start_act_scale2_dma_base_s2;
    logic [31:0]                gemv_start_rope_lut_dma_base_s2;
    logic [31:0]                gemv_start_weight_dma_base_s2;
    logic [31:0]                gemv_start_weight_scale_dma_base_s2;
    logic [31:0]                gemv_start_output_dma_base_s2;
    logic [15:0]                gemv_start_mat_width_s2;
    logic [15:0]                gemv_start_mat_height_s2;
    logic [15:0]                gemv_start_cache_cell_s2;
    logic [7:0]                 gemv_start_gemv_mode_s2;
    logic [7:0]                 gemv_start_output_precision_s2;
    logic                       gemv_start_enable_act_scale_s2;
    logic                       gemv_start_enable_act_scale2_s2;
    logic                       gemv_start_fixed_stride_enable_s2;
    logic                       gemv_start_cached_group_enable_s2;
    logic                       gemv_start_uses_weight_scale_s2;
    logic                       gemv_start_uses_rope_lut_s2;
    logic [31:0]                gemv_start_weight_row_tile_stride_bytes_s2;
    logic [15:0]                gemv_start_weight_capacity_tokens_s2;
    logic [31:0]                gemv_start_act_group_stride_s2;
    logic [63:0]                gemv_start_decode_flow_s2;
    logic [31:0]                gemv_start_col_tiles_s2;
    logic [31:0]                gemv_start_act_bytes_s2;
    logic [31:0]                gemv_start_group_act_bytes_s2;
    logic [31:0]                gemv_start_output_rows_s2;
    logic [31:0]                gemv_start_payload_bytes_s2;
    logic [31:0]                gemv_start_act_scale_payload_bytes_s2;
    logic [31:0]                gemv_start_act_scale2_payload_bytes_s2;
    logic [31:0]                gemv_start_row_tile_bytes_s2;
    logic [31:0]                gemv_start_weight_payload_bytes_s2;
    logic [31:0]                gemv_start_weight_scale_payload_bytes_s2;
    logic [31:0]                gemv_start_fixed_stride_active_row_tile_bytes_s2;
    logic [31:0]                gemv_start_rows_per_block_s2;
    logic [3:0]                 gemv_start_error_code_s2;
    logic [31:0]                gemv_start_act_dma_base_s3;
    logic [31:0]                gemv_start_act_scale_dma_base_s3;
    logic [31:0]                gemv_start_act_scale2_dma_base_s3;
    logic [31:0]                gemv_start_rope_lut_dma_base_s3;
    logic [31:0]                gemv_start_weight_dma_base_s3;
    logic [31:0]                gemv_start_weight_scale_dma_base_s3;
    logic [31:0]                gemv_start_output_dma_base_s3;
    logic [15:0]                gemv_start_mat_width_s3;
    logic [15:0]                gemv_start_mat_height_s3;
    logic [15:0]                gemv_start_cache_cell_s3;
    logic [7:0]                 gemv_start_gemv_mode_s3;
    logic [7:0]                 gemv_start_output_precision_s3;
    logic                       gemv_start_enable_act_scale_s3;
    logic                       gemv_start_enable_act_scale2_s3;
    logic                       gemv_start_fixed_stride_enable_s3;
    logic                       gemv_start_cached_group_enable_s3;
    logic                       gemv_start_uses_weight_scale_s3;
    logic                       gemv_start_uses_rope_lut_s3;
    logic [31:0]                gemv_start_weight_row_tile_stride_bytes_s3;
    logic [15:0]                gemv_start_weight_capacity_tokens_s3;
    logic [31:0]                gemv_start_act_group_stride_s3;
    logic [63:0]                gemv_start_decode_flow_s3;
    logic [31:0]                gemv_start_col_tiles_s3;
    logic [31:0]                gemv_start_act_bytes_s3;
    logic [31:0]                gemv_start_group_act_bytes_s3;
    logic [31:0]                gemv_start_output_rows_s3;
    logic [31:0]                gemv_start_payload_bytes_s3;
    logic [31:0]                gemv_start_act_scale_payload_bytes_s3;
    logic [31:0]                gemv_start_act_scale2_payload_bytes_s3;
    logic [31:0]                gemv_start_row_tile_bytes_s3;
    logic [31:0]                gemv_start_weight_payload_bytes_s3;
    logic [31:0]                gemv_start_weight_scale_payload_bytes_s3;
    logic [31:0]                gemv_start_fixed_stride_active_row_tile_bytes_s3;
    logic [31:0]                gemv_start_rows_per_block_s3;
    logic [3:0]                 gemv_start_error_code_s3;
    logic                       gemv_start_ok_s3;
    logic                       gemv_start_req_fire;
    logic                       gemv_start_commit_valid;
    logic                       gemv_start_commit_ok;

    gemv_block_seed_t           gemv_prep_seed;
    gemv_block_prep_a_t         gemv_block_prep_a;
    gemv_block_prep_b_t         gemv_block_prep_b;
    gemv_block_t                gemv_block;
    gemv_block_t                gemv_active_block;
    logic                       gemv_prep_req;
    logic                       gemv_block_prep_a_valid;
    logic                       gemv_block_prep_b_valid;
    logic                       gemv_block_valid;
    logic [31:0]                gemv_prep_row_base_calc;
    logic [15:0]                gemv_prep_remaining_row_tiles_calc;
    logic [31:0]                gemv_prep_current_rows_calc;
    logic [31:0]                gemv_prep_next_row_base_calc;
    logic [15:0]                gemv_prep_next_row_tile_base_calc;
    logic [15:0]                gemv_prep_next_remaining_row_tiles_calc;
    logic [31:0]                gemv_prep_next_rows_calc;
    logic [5:0]                 gemv_prep_current_row_tiles_calc;
    logic [5:0]                 gemv_prep_next_row_tiles_calc;
    logic [31:0]                gemv_prep_current_weight_bytes_calc;
    logic [31:0]                gemv_prep_next_weight_bytes_calc;

    logic [API_COUNT-1:0]       api_start_event;
    logic [API_COUNT-1:0]       api_done_event;
    logic [API_COUNT-1:0]       api_error_event;
    logic [API_COUNT-1:0][3:0]  api_error_code_event;
    logic [API_COUNT-1:0]       api_clear_done_event;
    logic [API_COUNT-1:0]       api_clear_error_event;
    logic                       api_clear_all_event;
    logic [7:0]                 irq_clear_mask_event;
    logic [4:0][15:0]           kv_k_scale_value;
    logic [4:0][15:0]           kv_v_scale_value;
    logic [7:0]                 kv_k_scale_count;
    logic [7:0]                 kv_v_scale_count;
    logic [7:0]                 kv_k_scale_seq;
    logic [7:0]                 kv_v_scale_seq;
    logic                       kv_k_scale_valid;
    logic                       kv_v_scale_valid;

    function automatic [63:0] apply_wstrb(
        input [63:0] old_val,
        input [63:0] new_val,
        input [7:0]  strb
    );
        apply_wstrb = old_val;
        for (int i = 0; i < 8; i++) begin
            if (strb[i]) begin
                apply_wstrb[i*8 +: 8] = new_val[i*8 +: 8];
            end
        end
    endfunction

    function automatic [31:0] ceil_div_u32(input [31:0] a, input [31:0] b);
        ceil_div_u32 = (b == 0) ? 32'd0 : ((a + b - 32'd1) / b);
    endfunction

    function automatic [31:0] min_u32(input [31:0] a, input [31:0] b);
        min_u32 = (a < b) ? a : b;
    endfunction

    function automatic [63:0] pack_kv_scale0(input logic [4:0][15:0] value);
        pack_kv_scale0 = {value[3], value[2], value[1], value[0]};
    endfunction

    function automatic [63:0] pack_kv_scale1(
        input logic [15:0] value4,
        input logic [7:0]  count,
        input logic [7:0]  seq,
        input logic        valid
    );
        pack_kv_scale1 = {31'd0, valid, seq, count, value4};
    endfunction

    function automatic [31:0] weight_tile_bytes(input [7:0] mode);
        case (mode)
            8'd0:    weight_tile_bytes = GEMV_WEIGHT_TILE_BYTES_MODE0;
            8'd1:    weight_tile_bytes = GEMV_WEIGHT_TILE_BYTES_MODE1;
            default: weight_tile_bytes = 32'd0;
        endcase
    endfunction

    function automatic [31:0] gemv_col_tiles_for_mode(
        input [15:0] mat_width,
        input [7:0]  mode
    );
        begin
            gemv_col_tiles_for_mode = mode[0] ?
                (({16'd0, mat_width} + 32'd63) >> 6) :
                (({16'd0, mat_width} + (GEMV_TILE_ELEMS - 32'd1)) >> 7);
        end
    endfunction

    function automatic [31:0] gemv_act_bytes_for_mode(
        input [31:0] col_tiles,
        input [7:0]  mode
    );
        begin
            gemv_act_bytes_for_mode = mode[0] ? (col_tiles << 7) :
                                                 (col_tiles << 8);
        end
    endfunction

    function automatic [31:0] gemv_act_scale_bytes_for(
        input [31:0] col_tiles,
        input [7:0]  mode,
        input [63:0] decode_flow,
        input        enable
    );
        logic [31:0] base_bytes;
        begin
            base_bytes = mode[0] ? (col_tiles << 7) : (col_tiles << 8);
            if (!enable) begin
                gemv_act_scale_bytes_for = 32'd0;
            end else begin
                gemv_act_scale_bytes_for = base_bytes;
            end
        end
    endfunction

    function automatic [31:0] gemv_group_count_scale_u32(
        input [31:0] value,
        input [63:0] decode_flow
    );
        begin
            unique case (decode_flow[26:25])
                2'd0: gemv_group_count_scale_u32 = value;
                2'd1: gemv_group_count_scale_u32 = value << 1;
                2'd2: gemv_group_count_scale_u32 = (value << 1) + value;
                default: gemv_group_count_scale_u32 = value << 2;
            endcase
        end
    endfunction

    function automatic [63:0] gemv_single_group_flow(input [63:0] decode_flow);
        begin
            gemv_single_group_flow = decode_flow;
            gemv_single_group_flow[26:25] = 2'd0;
        end
    endfunction

    function automatic logic gemv_uses_weight_scale(
        input [7:0]  mode,
        input [63:0] decode_flow
    );
        begin
            gemv_uses_weight_scale = ((mode == 8'd0) || (mode == 8'd1)) &&
                                     !decode_flow[29];
        end
    endfunction

    function automatic logic gemv_uses_rope_lut(input [63:0] decode_flow);
        begin
            gemv_uses_rope_lut = (decode_flow[9:8] == 2'd2);
        end
    endfunction

    function automatic [3:0] gemv_group_count_segments_x2(input [63:0] decode_flow);
        begin
            unique case (decode_flow[26:25])
                2'd0: gemv_group_count_segments_x2 = 4'd2;
                2'd1: gemv_group_count_segments_x2 = 4'd4;
                2'd2: gemv_group_count_segments_x2 = 4'd6;
                default: gemv_group_count_segments_x2 = 4'd8;
            endcase
        end
    endfunction

    function automatic [31:0] gemv_group_act_bytes_for(
        input [31:0] act_bytes,
        input [31:0] act_group_stride_bytes,
        input [63:0] decode_flow,
        input        cached_group_enable
    );
        begin
            if (cached_group_enable || decode_flow[26:25] == 2'd0) begin
                gemv_group_act_bytes_for = act_bytes;
            end else if (act_group_stride_bytes == 32'd0) begin
                gemv_group_act_bytes_for =
                    gemv_group_count_scale_u32(act_bytes, decode_flow);
            end else begin
                unique case (decode_flow[26:25])
                    2'd1: gemv_group_act_bytes_for = act_group_stride_bytes + act_bytes;
                    2'd2: gemv_group_act_bytes_for = (act_group_stride_bytes << 1) + act_bytes;
                    default: gemv_group_act_bytes_for = (act_group_stride_bytes << 1) +
                                                        act_group_stride_bytes + act_bytes;
                endcase
            end
        end
    endfunction

    function automatic [31:0] align_gemv_mvin_bytes(input [31:0] byte_count);
        align_gemv_mvin_bytes = (byte_count + (GEMV_MVIN_ALIGN - 32'd1)) &
                                ~(GEMV_MVIN_ALIGN - 32'd1);
    endfunction

    function automatic [31:0] gemv_weight_payload_row_tile_bytes_for(
        input [31:0] col_tiles,
        input [7:0]  mode
    );
        begin
            gemv_weight_payload_row_tile_bytes_for = col_tiles << 11;
        end
    endfunction

    function automatic [31:0] gemv_rows_per_block_for(input [31:0] row_tile_bytes);
        logic [31:0] fit_tiles;
        begin
            fit_tiles = 32'd0;
            if (row_tile_bytes != 32'd0) begin
                for (int i = 1; i <= GEMV_TARGET_ROW_TILES_PER_BLOCK_INT; i++) begin
                    if (row_tile_bytes <= (GEMV_START_CAPACITY / i)) begin
                        fit_tiles = i;
                    end
                end
            end
            gemv_rows_per_block_for = fit_tiles << 5;
        end
    endfunction

    function automatic [3:0] gemv_start_error_for(
        input        zero_shape,
        input        unsupported_mode,
        input        alignment_error,
        input        height_overflow,
        input        fixed_stride_shape_error,
        input        fixed_stride_alignment_error,
        input [31:0] payload_bytes,
        input [31:0] act_scale_payload_bytes,
        input [31:0] act_scale2_payload_bytes,
        input [31:0] weight_payload_bytes
    );
        begin
            if (zero_shape) begin
                gemv_start_error_for = ERR_ILLEGAL_SHAPE;
            end else if (unsupported_mode) begin
                gemv_start_error_for = ERR_UNSUPPORTED_MODE;
            end else if (fixed_stride_shape_error ||
                         (payload_bytes + act_scale_payload_bytes + act_scale2_payload_bytes) >
                         GEMV_ACT_PAYLOAD_LIMIT ||
                         weight_payload_bytes == 32'd0 ||
                         weight_payload_bytes > GEMV_MAX_MVIN_BYTES ||
                         weight_payload_bytes[5:0] != 6'd0 ||
                         height_overflow) begin
                gemv_start_error_for = ERR_ILLEGAL_SHAPE;
            end else if (alignment_error || fixed_stride_alignment_error) begin
                gemv_start_error_for = ERR_ALIGNMENT_ERROR;
            end else begin
                gemv_start_error_for = ERR_NONE;
            end
        end
    endfunction

    function automatic logic default_matvec_act_scale_enable(
        input [7:0]  mode,
        input [63:0] decode_flow
    );
        begin
            default_matvec_act_scale_enable =
                (mode == 8'd0) ||
                ((mode == 8'd1) && decode_flow[27]);
        end
    endfunction

    function automatic [31:0] block_rows_for(
        input [31:0] row_base,
        input [31:0] mat_height,
        input [31:0] rows_per_block
    );
        logic [31:0] remaining;
        begin
            remaining = (row_base < mat_height) ? (mat_height - row_base) : 32'd0;
            block_rows_for = min_u32(remaining, rows_per_block);
        end
    endfunction

    function automatic [31:0] row_tiles_for_rows(input [31:0] rows);
        row_tiles_for_rows = (rows + (GEMV_ROW_TILE_ELEMS - 32'd1)) >> 5;
    endfunction

    function automatic [31:0] row_tile_bytes_times(
        input [5:0]  row_tiles,
        input [31:0] row_tile_bytes
    );
        logic [31:0] part0;
        logic [31:0] part1;
        begin
            part0 = (row_tiles[0] ? row_tile_bytes        : 32'd0) +
                    (row_tiles[1] ? (row_tile_bytes << 1) : 32'd0) +
                    (row_tiles[2] ? (row_tile_bytes << 2) : 32'd0);
            part1 = (row_tiles[3] ? (row_tile_bytes << 3) : 32'd0) +
                    (row_tiles[4] ? (row_tile_bytes << 4) : 32'd0) +
                    (row_tiles[5] ? (row_tile_bytes << 5) : 32'd0);
            row_tile_bytes_times = part0 + part1;
        end
    endfunction

    function automatic [31:0] row_tile_bytes_times16(
        input [15:0] row_tiles,
        input [31:0] row_tile_bytes
    );
        logic [31:0] sum;
        begin
            sum = 32'd0;
            for (int i = 0; i < 16; i++) begin
                if (row_tiles[i]) begin
                    sum = sum + (row_tile_bytes << i);
                end
            end
            row_tile_bytes_times16 = sum;
        end
    endfunction

    function automatic [31:0] small_index_times_u32(
        input [3:0]  index,
        input [31:0] value
    );
        logic [31:0] part0;
        logic [31:0] part1;
        begin
            part0 = (index[0] ? value        : 32'd0) +
                    (index[1] ? (value << 1) : 32'd0);
            part1 = (index[2] ? (value << 2) : 32'd0) +
                    (index[3] ? (value << 3) : 32'd0);
            small_index_times_u32 = part0 + part1;
        end
    endfunction

    function automatic [31:0] gemv_group_output_byte_offset(
        input [1:0]  group_index,
        input [15:0] mat_height
    );
        begin
            gemv_group_output_byte_offset =
                small_index_times_u32({2'd0, group_index}, {15'd0, mat_height, 1'b0});
        end
    endfunction

    function automatic [31:0] gemv_group_output_byte_offset_for_flow(
        input [1:0]  group_index,
        input [15:0] mat_height,
        input [63:0] decode_flow
    );
        logic [31:0] group_stride_bytes;
        begin
            if (decode_flow[31] && (decode_flow[12:11] == 2'd1) &&
                (decode_flow[15:13] == 3'd2)) begin
                group_stride_bytes = ((({16'd0, mat_height} + 32'd63) >> 6)
                                      << 6) << 2;
                gemv_group_output_byte_offset_for_flow =
                    small_index_times_u32({2'd0, group_index},
                                          group_stride_bytes);
            end else begin
                gemv_group_output_byte_offset_for_flow =
                    gemv_group_output_byte_offset(group_index, mat_height);
            end
        end
    endfunction

    function automatic [15:0] gemv_group_output_addr16(
        input [1:0]  group_index,
        input [15:0] mat_height
    );
        logic [31:0] offset;
        begin
            offset = gemv_group_output_byte_offset(group_index, mat_height);
            gemv_group_output_addr16 =
                GEMV_OUTPUT_BASE[15:0] + offset[15:0];
        end
    endfunction

    function automatic [15:0] gemv_group_output_addr16_for_flow(
        input [1:0]  group_index,
        input [15:0] mat_height,
        input [63:0] decode_flow
    );
        logic [31:0] offset;
        begin
            offset = gemv_group_output_byte_offset_for_flow(group_index,
                                                            mat_height,
                                                            decode_flow);
            gemv_group_output_addr16_for_flow =
                GEMV_OUTPUT_BASE[15:0] + offset[15:0];
        end
    endfunction

    function automatic [63:0] make_bypass_output_flow(input [15:0] elem_count);
        make_bypass_output_flow = (64'd1 << 0) | (64'd1 << 13) |
                                  ({48'd0, elem_count} << 32);
    endfunction

    function automatic logic valid_gemv_act_frac_cfg(
        input logic [7:0] gemv_mode,
        input logic [7:0] act_frac_cfg
    );
        logic [4:0] frac_bits;
        begin
            frac_bits = act_frac_cfg[4:0];
            valid_gemv_act_frac_cfg =
                (act_frac_cfg == 8'd0) ||
                ((gemv_mode == 8'd1) &&
                 act_frac_cfg[7] &&
                 (act_frac_cfg[6:5] == 2'b00) &&
                 ((frac_bits == 5'd16) || (act_frac_cfg == 8'h98)));
        end
    endfunction

    function automatic [1:0] legacy_output_precision(input [7:0] output_precision);
        legacy_output_precision = output_precision[0] ? 2'd3 : 2'd1;
    endfunction

    function automatic logic kv_quant_scratch_flow(input logic [63:0] flow);
        kv_quant_scratch_flow = flow[24] && flow[30];
    endfunction

    function automatic [1:0] gemv_mvout_precision(
        input logic [63:0] flow,
        input logic [7:0]  output_precision
    );
        gemv_mvout_precision = kv_quant_scratch_flow(flow) ?
                               2'b00 : legacy_output_precision(output_precision);
    endfunction

    function automatic logic gemv_mvout_is_quant(input logic [63:0] flow);
        gemv_mvout_is_quant = kv_quant_scratch_flow(flow);
    endfunction

    function automatic [DMA_SEL_WIDTH-1:0] legal_dma_id(input [7:0] dma_id);
        legal_dma_id = (dma_id < DMA_NUM) ? dma_id[DMA_SEL_WIDTH-1:0] : '0;
    endfunction

    function automatic logic is_write_addr_legal(input [15:0] addr);
        unique case (addr)
            REG_IAR, REG_MER, REG_IER, REG_ISR, REG_GLOBAL_CLEAR,
            REG_DECODE_IAR_ALIAS, REG_DECODE_MER_ALIAS, REG_DECODE_IER_ALIAS, REG_DECODE_ISR_ALIAS,
            REG_GEMV_DESC0, REG_GEMV_DESC1, REG_GEMV_DESC2, REG_GEMV_DESC3,
            REG_GEMV_DESC4, REG_GEMV_DESC5, REG_GEMV_DESC6, REG_GEMV_DESC7,
            REG_GEMV_CTRL, REG_GEMV_STATUS,
            REG_MLP_DESC0, REG_MLP_DESC1, REG_MLP_DESC2, REG_MLP_DESC3,
            REG_MLP_CTRL, REG_MLP_STATUS,
            REG_ATTN_DESC0, REG_ATTN_DESC1, REG_ATTN_DESC2, REG_ATTN_DESC3,
            REG_ATTN_DESC4, REG_ATTN_DESC5, REG_ATTN_DESC6, REG_ATTN_CTRL, REG_ATTN_STATUS,
            REG_DBG_MVIN_DESC0, REG_DBG_MVIN_DESC1, REG_DBG_MVIN_CTRL, REG_DBG_MVIN_STATUS,
            REG_DBG_MVOUT_DESC0, REG_DBG_MVOUT_DESC1, REG_DBG_MVOUT_CTRL, REG_DBG_MVOUT_STATUS,
            REG_DBG_GEMV_DESC0, REG_DBG_GEMV_DESC1, REG_DBG_GEMV_DESC2,
            REG_DBG_GEMV_DESC3, REG_DBG_GEMV_DESC4, REG_DBG_GEMV_CTRL, REG_DBG_GEMV_STATUS,
            REG_GENERIC_CONTROL, REG_GENERIC_CONTROL_ALIAS:
                is_write_addr_legal = 1'b1;
            default:
                is_write_addr_legal = 1'b0;
        endcase
    endfunction

    function automatic logic is_read_addr_legal(input [15:0] addr);
        unique case (addr)
            REG_IAR, REG_MER, REG_IER, REG_ISR, REG_IPR, REG_PROFILE0,
            REG_DECODE_IAR_ALIAS, REG_DECODE_MER_ALIAS, REG_DECODE_IER_ALIAS,
            REG_DECODE_ISR_ALIAS, REG_DECODE_IPR_ALIAS,
            REG_GEMV_DESC0, REG_GEMV_DESC1, REG_GEMV_DESC2, REG_GEMV_DESC3,
            REG_GEMV_DESC4, REG_GEMV_DESC5, REG_GEMV_DESC6, REG_GEMV_DESC7,
            REG_GEMV_STATUS,
            REG_MLP_DESC0, REG_MLP_DESC1, REG_MLP_DESC2, REG_MLP_DESC3, REG_MLP_STATUS,
            REG_ATTN_DESC0, REG_ATTN_DESC1, REG_ATTN_DESC2, REG_ATTN_DESC3,
            REG_ATTN_DESC4, REG_ATTN_DESC5, REG_ATTN_DESC6, REG_ATTN_STATUS,
            REG_DBG_MVIN_DESC0, REG_DBG_MVIN_DESC1, REG_DBG_MVIN_STATUS,
            REG_DBG_MVOUT_DESC0, REG_DBG_MVOUT_DESC1, REG_DBG_MVOUT_STATUS,
            REG_DBG_GEMV_DESC0, REG_DBG_GEMV_DESC1, REG_DBG_GEMV_DESC2,
            REG_DBG_GEMV_DESC3, REG_DBG_GEMV_DESC4, REG_DBG_GEMV_STATUS,
            REG_KV_K_SCALE0, REG_KV_K_SCALE1, REG_KV_V_SCALE0, REG_KV_V_SCALE1,
            REG_GENERIC_MAGIC, REG_GENERIC_VERSION, REG_GENERIC_MODE, REG_GENERIC_CAPS,
            REG_GENERIC_STATUS, REG_GENERIC_ERROR,
            REG_GENERIC_MAGIC_ALIAS, REG_GENERIC_VERSION_ALIAS, REG_GENERIC_MODE_ALIAS, REG_GENERIC_CAPS_ALIAS,
            REG_GENERIC_STATUS_ALIAS, REG_GENERIC_ERROR_ALIAS:
                is_read_addr_legal = 1'b1;
            default:
                is_read_addr_legal = 1'b0;
        endcase
    endfunction

    function automatic [63:0] status_value(input int api_idx);
        status_value = {(PROFILE_COUNTERS_ON ? api_accepted_count[api_idx] : 32'd0),
                        24'd0,
                        (DETAILED_ERROR_STATUS_ON ? api_error_code[api_idx] : 4'd0),
                        1'b0,
                        api_error[api_idx],
                        api_done[api_idx],
                        api_busy[api_idx]};
    endfunction

    function automatic [7:0] generic_error_value;
        logic [7:0] result;
        begin
            result = 8'd0;
            for (int i = 0; i < API_COUNT; i++) begin
                if (api_error[i]) begin
                    result = result | (DETAILED_ERROR_STATUS_ON ?
                                       {4'd0, api_error_code[i]} :
                                       8'h01);
                end
            end
            generic_error_value = result;
        end
    endfunction

    function automatic [63:0] read_data_for_addr(input [15:0] addr);
        unique case (addr)
            REG_IAR, REG_DECODE_IAR_ALIAS:
                                read_data_for_addr = {56'd0, irq_status};
            REG_MER, REG_DECODE_MER_ALIAS:
                                read_data_for_addr = {63'd0, master_enable};
            REG_IER, REG_DECODE_IER_ALIAS:
                                read_data_for_addr = {56'd0, irq_enable};
            REG_ISR, REG_DECODE_ISR_ALIAS:
                                read_data_for_addr = {56'd0, irq_status};
            REG_IPR, REG_DECODE_IPR_ALIAS:
                                read_data_for_addr = {56'd0, (irq_status & irq_enable)};
            REG_PROFILE0:       read_data_for_addr = PROFILE_COUNTERS_ON ?
                                {completed_block_count,
                                 prefetch_wait_cycles[15:0],
                                 mvout_wait_cycles[7:0],
                                 engine_state,
                                 mvout_overlap_seen,
                                 last_pingpong_buffer,
                                 overlap_seen} : 64'd0;
            REG_GEMV_DESC0:     read_data_for_addr = gemv_desc0;
            REG_GEMV_DESC1:     read_data_for_addr = gemv_desc1;
            REG_GEMV_DESC2:     read_data_for_addr = gemv_desc2;
            REG_GEMV_DESC3:     read_data_for_addr = gemv_desc3;
            REG_GEMV_DESC4:     read_data_for_addr = gemv_desc4;
            REG_GEMV_DESC5:     read_data_for_addr = gemv_desc5;
            REG_GEMV_DESC6:     read_data_for_addr = gemv_desc6;
            REG_GEMV_DESC7:     read_data_for_addr = gemv_desc7;
            REG_GEMV_STATUS:    read_data_for_addr = status_value(API_GEMV_BLOCK);
            REG_MLP_DESC0:      read_data_for_addr = mlp_desc0;
            REG_MLP_DESC1:      read_data_for_addr = mlp_desc1;
            REG_MLP_DESC2:      read_data_for_addr = mlp_desc2;
            REG_MLP_DESC3:      read_data_for_addr = mlp_desc3;
            REG_MLP_STATUS:     read_data_for_addr = status_value(API_MLP);
            REG_ATTN_DESC0:     read_data_for_addr = attn_desc0;
            REG_ATTN_DESC1:     read_data_for_addr = attn_desc1;
            REG_ATTN_DESC2:     read_data_for_addr = attn_desc2;
            REG_ATTN_DESC3:     read_data_for_addr = attn_desc3;
            REG_ATTN_DESC4:     read_data_for_addr = attn_desc4;
            REG_ATTN_DESC5:     read_data_for_addr = attn_desc5;
            REG_ATTN_DESC6:     read_data_for_addr = attn_desc6;
            REG_ATTN_STATUS:    read_data_for_addr = status_value(API_ATTN);
            REG_DBG_MVIN_DESC0: read_data_for_addr = dbg_mvin_desc0;
            REG_DBG_MVIN_DESC1: read_data_for_addr = dbg_mvin_desc1;
            REG_DBG_MVIN_STATUS: read_data_for_addr = status_value(API_DBG_MVIN);
            REG_DBG_MVOUT_DESC0: read_data_for_addr = dbg_mvout_desc0;
            REG_DBG_MVOUT_DESC1: read_data_for_addr = dbg_mvout_desc1;
            REG_DBG_MVOUT_STATUS: read_data_for_addr = status_value(API_DBG_MVOUT);
            REG_DBG_GEMV_DESC0: read_data_for_addr = dbg_gemv_desc0;
            REG_DBG_GEMV_DESC1: read_data_for_addr = dbg_gemv_desc1;
            REG_DBG_GEMV_DESC2: read_data_for_addr = dbg_gemv_desc2;
            REG_DBG_GEMV_DESC3: read_data_for_addr = dbg_gemv_desc3;
            REG_DBG_GEMV_DESC4: read_data_for_addr = dbg_gemv_desc4;
            REG_DBG_GEMV_STATUS: read_data_for_addr = status_value(API_DBG_GEMV);
            REG_KV_K_SCALE0:    read_data_for_addr = pack_kv_scale0(kv_k_scale_value);
            REG_KV_K_SCALE1:    read_data_for_addr = pack_kv_scale1(kv_k_scale_value[4],
                                                                     kv_k_scale_count,
                                                                     kv_k_scale_seq,
                                                                     kv_k_scale_valid);
            REG_KV_V_SCALE0:    read_data_for_addr = pack_kv_scale0(kv_v_scale_value);
            REG_KV_V_SCALE1:    read_data_for_addr = pack_kv_scale1(kv_v_scale_value[4],
                                                                     kv_v_scale_count,
                                                                     kv_v_scale_seq,
                                                                     kv_v_scale_valid);
            REG_GENERIC_MAGIC, REG_GENERIC_MAGIC_ALIAS:
                                read_data_for_addr = {32'd0, GENERIC_MAGIC};
            REG_GENERIC_VERSION, REG_GENERIC_VERSION_ALIAS:
                                read_data_for_addr = {32'd0, GENERIC_ABI_VERSION};
            REG_GENERIC_MODE, REG_GENERIC_MODE_ALIAS:
                                read_data_for_addr = {32'd0, GENERIC_MODE_DECODE};
            REG_GENERIC_CAPS, REG_GENERIC_CAPS_ALIAS:
                                read_data_for_addr = {32'd0, GENERIC_CAPS};
            REG_GENERIC_STATUS, REG_GENERIC_STATUS_ALIAS:
                                read_data_for_addr = {61'd0, |api_error, npu_busy_any, !npu_busy_any};
            REG_GENERIC_ERROR, REG_GENERIC_ERROR_ALIAS:
                                read_data_for_addr = {56'd0, generic_error_value()};
            default:             read_data_for_addr = 64'd0;
        endcase
    endfunction

    task automatic clear_sticky_status;
        begin
            api_done       <= '0;
            api_error      <= '0;
            if (DETAILED_ERROR_STATUS_ON) begin
                api_error_code <= '0;
            end
            irq_status     <= '0;
        end
    endtask

    task automatic clear_api_status_bit(input int api_idx, input [63:0] data, input [7:0] strb);
        logic [API_COUNT-1:0] remaining_errors;
        begin
            if (strb[0] && data[1]) begin
                api_done[api_idx] <= 1'b0;
                irq_status[api_idx] <= 1'b0;
            end
            if (strb[0] && data[2]) begin
                remaining_errors = api_error;
                remaining_errors[api_idx] = 1'b0;
                api_error[api_idx] <= 1'b0;
                if (DETAILED_ERROR_STATUS_ON) begin
                    api_error_code[api_idx] <= ERR_NONE;
                end
                if (remaining_errors == '0) begin
                    irq_status[IRQ_ERROR] <= 1'b0;
                end
            end
        end
    endtask

    task automatic raise_api_error(input int api_idx, input [3:0] err_code);
        begin
            api_error[api_idx]      <= 1'b1;
            if (DETAILED_ERROR_STATUS_ON) begin
                api_error_code[api_idx] <= err_code;
            end
            irq_status[IRQ_ERROR]   <= 1'b1;
        end
    endtask

    task automatic complete_api(input int api_idx);
        begin
            api_busy[api_idx]     <= 1'b0;
            api_done[api_idx]     <= 1'b1;
            irq_status[api_idx]   <= 1'b1;
            command_active        <= 1'b0;
            engine_state          <= ST_IDLE;
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kv_k_scale_value <= '0;
            kv_v_scale_value <= '0;
            kv_k_scale_count <= '0;
            kv_v_scale_count <= '0;
            kv_k_scale_seq <= '0;
            kv_v_scale_seq <= '0;
            kv_k_scale_valid <= 1'b0;
            kv_v_scale_valid <= 1'b0;
        end else if (api_clear_all_event) begin
            kv_k_scale_value <= '0;
            kv_v_scale_value <= '0;
            kv_k_scale_count <= '0;
            kv_v_scale_count <= '0;
            kv_k_scale_seq <= '0;
            kv_v_scale_seq <= '0;
            kv_k_scale_valid <= 1'b0;
            kv_v_scale_valid <= 1'b0;
        end else if (kv_scale_commit_valid) begin
            if (kv_scale_commit_is_v) begin
                kv_v_scale_value <= kv_scale_commit_values;
                kv_v_scale_count <= kv_scale_commit_count;
                if (kv_scale_commit_count == 8'd5) begin
                    kv_v_scale_seq <= kv_v_scale_seq + 1'b1;
                end
                kv_v_scale_valid <= (kv_scale_commit_count == 8'd5);
            end else begin
                kv_k_scale_value <= kv_scale_commit_values;
                kv_k_scale_count <= kv_scale_commit_count;
                if (kv_scale_commit_count == 8'd5) begin
                    kv_k_scale_seq <= kv_k_scale_seq + 1'b1;
                end
                kv_k_scale_valid <= (kv_scale_commit_count == 8'd5);
            end
        end
    end

    task automatic issue_mvin_req(
        input [31:0] dram_addr_i,
        input [31:0] sram_addr_i,
        input [31:0] byte_count_i,
        input [1:0]  input_type_i,
        input [1:0]  precision_i,
        input        dest_i,
        input        is_bias_i,
        input        is_quant_i,
        input        scale_target_i,
        input        stream_fifo_fill_i,
        input [7:0]  dma_id_i
    );
        logic [DMA_SEL_WIDTH-1:0] target;
        logic [31:0] byte_count_m1;
        begin
            target = dest_i ? ACC_DMA_ID : legal_dma_id(dma_id_i);
            byte_count_m1 = byte_count_i - 32'd1;
            mvin_dram_addr             <= dram_addr_i[RF_DATA_WIDTH/2-1:0];
            mvin_row_num               <= '0;
            mvin_sram_addr             <= sram_addr_i[RF_DATA_WIDTH/2-1:0];
            mvin_col_num               <= byte_count_m1[RF_DATA_WIDTH/2-1:0];
            cfg_mvin_input_type        <= input_type_i;
            cfg_mvin_input_precision   <= precision_i;
            cfg_mvin_is_quant          <= is_quant_i;
            cfg_mvin_dest              <= dest_i;
            cfg_mvin_isbias            <= is_bias_i;
            cfg_mvin_sram_stride       <= '0;
            cfg_mvin_dram_stride       <= '0;
            cfg_mvin_input_zeropoint   <= '0;
            cfg_mvin_input_scale       <= '0;
            cfg_mvin_input_scale_shift <= '0;

            dma_mvin_dram_addr[target]               <= dram_addr_i[RF_DATA_WIDTH/2-1:0];
            dma_mvin_row_num[target]                 <= '0;
            dma_mvin_sram_addr[target]               <= sram_addr_i[RF_DATA_WIDTH/2-1:0];
            dma_mvin_col_num[target]                 <= byte_count_m1[RF_DATA_WIDTH/2-1:0];
            dma_cfg_mvin_input_type[target]          <= input_type_i;
            dma_cfg_mvin_input_precision[target]     <= precision_i;
            dma_cfg_mvin_is_quant[target]            <= is_quant_i;
            dma_cfg_mvin_dest[target]                <= dest_i;
            dma_cfg_mvin_isbias[target]              <= is_bias_i;
            dma_cfg_mvin_scale_target[target]        <= scale_target_i;
            dma_cfg_mvin_stream_fifo_fill[target]    <= stream_fifo_fill_i;
            dma_cfg_mvin_sram_stride[target]         <= '0;
            dma_cfg_mvin_dram_stride[target]         <= '0;
            dma_cfg_mvin_input_zeropoint[target]     <= '0;
            dma_cfg_mvin_input_scale[target]         <= '0;
            dma_cfg_mvin_input_scale_shift[target]   <= '0;
            dma_mvin_req_en[target]                  <= 1'b1;
        end
    endtask

    task automatic issue_mvout_req(
        input [31:0] dram_addr_i,
        input [31:0] sram_addr_i,
        input [31:0] elem_count_i,
        input [1:0]  output_type_i,
        input [1:0]  precision_i,
        input        source_i,
        input        is_quant_i,
        input        per_channel_i,
        input [7:0]  dma_id_i
    );
        logic [DMA_SEL_WIDTH-1:0] target;
        logic [31:0] elem_count_m1;
        begin
            target = source_i ? ACC_DMA_ID : legal_dma_id(dma_id_i);
            elem_count_m1 = elem_count_i - 32'd1;
            mvout_dram_addr              <= dram_addr_i[RF_DATA_WIDTH/2-1:0];
            mvout_row_num                <= elem_count_m1[RF_DATA_WIDTH/2-1:0];
            mvout_sram_addr              <= sram_addr_i[RF_DATA_WIDTH/2-1:0];
            mvout_col_num                <= '0;
            cfg_mvout_output_type        <= output_type_i;
            cfg_mvout_output_precision   <= precision_i;
            cfg_mvout_is_quant           <= is_quant_i;
            cfg_mvout_source             <= source_i;
            cfg_mvout_per_channel        <= per_channel_i;
            cfg_mvout_sram_stride        <= {{(RF_DATA_WIDTH/4-1){1'b0}}, 1'b1};
            cfg_mvout_dram_stride        <= {{(RF_DATA_WIDTH/2-1){1'b0}}, 1'b1};
            cfg_mvout_output_zeropoint   <= '0;
            cfg_mvout_output_scale       <= '0;
            cfg_mvout_output_scale_shift <= '0;

            dma_mvout_dram_addr[target]              <= dram_addr_i[RF_DATA_WIDTH/2-1:0];
            dma_mvout_row_num[target]                <= elem_count_m1[RF_DATA_WIDTH/2-1:0];
            dma_mvout_sram_addr[target]              <= sram_addr_i[RF_DATA_WIDTH/2-1:0];
            dma_mvout_col_num[target]                <= '0;
            dma_cfg_mvout_output_type[target]        <= output_type_i;
            dma_cfg_mvout_output_precision[target]   <= precision_i;
            dma_cfg_mvout_is_quant[target]           <= is_quant_i;
            dma_cfg_mvout_source[target]             <= source_i;
            dma_cfg_mvout_per_channel[target]        <= per_channel_i;
            dma_cfg_mvout_sram_stride[target]        <= {{(RF_DATA_WIDTH/4-1){1'b0}}, 1'b1};
            dma_cfg_mvout_dram_stride[target]        <= {{(RF_DATA_WIDTH/2-1){1'b0}}, 1'b1};
            dma_cfg_mvout_output_zeropoint[target]   <= '0;
            dma_cfg_mvout_output_scale[target]       <= '0;
            dma_cfg_mvout_output_scale_shift[target] <= '0;
            dma_mvout_req_en[target]                 <= 1'b1;
        end
    endtask

    task automatic issue_matvec_req(
        input [31:0] mat_addr_i,
        input [31:0] vec_addr_i,
        input [15:0] mat_width_i,
        input [15:0] mat_height_i,
        input [15:0] output_addr_i,
        input [15:0] scale_addr_i,
        input [31:0] act_scale_addr_i,
        input [31:0] act_scale2_addr_i,
        input [15:0] cache_cell_i,
        input [31:0] act_group_stride_i,
        input [7:0]  act_frac_cfg_i,
        input [7:0]  gemv_mode_i,
        input [63:0] decode_flow_i,
        input        weight_stream_mode_i,
        input        act_scale_enable_i,
        input        act_scale2_enable_i
    );
        begin
            cfg_compute_dataflow     <= 1'b0;
            cfg_compute_padding_left <= '0;
            cfg_compute_padding_right <= '0;
            cfg_compute_padding_top <= '0;
            cfg_compute_padding_bottom <= '0;
            cfg_compute_padding_mode <= '0;
            cfg_compute_weight_shape <= '0;
            cfg_compute_weight_stride <= '0;
            cfg_compute_weight_dilation <= '0;
            cfg_compute_is_groupconv <= 1'b0;
            cfg_compute_int_type     <= {1'b0, gemv_mode_i[0]};
            cfg_compute_optype       <= 2'd2;
            cfg_compute_accout_dest  <= 1'b0;
            cfg_compute_asymmetric_activations <= 1'b0;
            cfg_decode_flow          <= decode_flow_i;
            cfg_compute_inputa_zeropoint <= '0;
            cfg_compute_inputb_zeropoint <= '0;
            cfg_compute_output_zeropoint <= '0;
            cfg_compute_output_scale <= '0;
            cfg_compute_output_scale_shift <= '0;

            matvec_input_mat_addr       <= mat_addr_i[RF_DATA_WIDTH/2-1:0];
            matvec_input_vec_addr       <= vec_addr_i[RF_DATA_WIDTH/2-1:0];
            matvec_input_act_scale_addr <= act_scale_addr_i[RF_DATA_WIDTH/2-1:0];
            matvec_input_act_scale2_addr <= act_scale2_addr_i[RF_DATA_WIDTH/2-1:0];
            matvec_cache_cell_idx       <= cache_cell_i[RF_DATA_WIDTH/4-1:0];
            matvec_input_act_group_stride_bytes <= act_group_stride_i[RF_DATA_WIDTH/2-1:0];
            matvec_act_frac_cfg         <= act_frac_cfg_i;
            matvec_input_mat_width      <= mat_width_i;
            matvec_input_mat_height     <= mat_height_i;
            matvec_input_mat_stride     <= {{(RF_DATA_WIDTH/2-16){1'b0}}, output_addr_i};
            matvec_input_vec_stride     <= scale_addr_i[RF_DATA_WIDTH/4-1:0];
            matvec_weight_stream_mode   <= weight_stream_mode_i;
            matvec_act_scale_enable     <= act_scale_enable_i;
            matvec_act_scale2_enable    <= act_scale2_enable_i;
            matvec_req_en               <= 1'b1;
        end
    endtask

    assign write_fire = axi_awaddr_valid && axi_wdata_valid && !s_axi_bvalid;
    assign s_axi_awready = !axi_awaddr_valid && !s_axi_bvalid;
    assign s_axi_wready = !axi_wdata_valid && !s_axi_bvalid;
    assign irq = master_enable && |(irq_status & irq_enable);

    always_comb begin
        dma_mvin_resp_done_any = |dma_mvin_resp_done_status;
        dma_mvout_resp_done_any = |dma_mvout_resp_done_status;
        gemv_start_accept_pending = gemv_start_s0_valid || gemv_start_s1_valid ||
                                    gemv_start_s2_valid || gemv_start_s3_valid;
        npu_busy_any = command_active || gemv_start_accept_pending ||
                       (|dma_mvin_busy_status) || (|dma_mvout_busy_status) ||
                       matvec_busy || sa_busy || matadd_busy || sfu_busy;

        gemv_start_req_fire = 1'b0;
        api_start_event = '0;
        api_done_event = '0;
        api_error_event = '0;
        api_error_code_event = '0;
        api_clear_done_event = '0;
        api_clear_error_event = '0;
        api_clear_all_event = 1'b0;
        irq_clear_mask_event = '0;

        gemv_prep_row_base_calc = {11'd0, gemv_prep_seed.row_tile_base, 5'd0};
        gemv_prep_remaining_row_tiles_calc =
            (gemv_prep_seed.row_tile_base < cmd_total_row_tiles) ?
            (cmd_total_row_tiles - gemv_prep_seed.row_tile_base) : 16'd0;
        gemv_prep_current_row_tiles_calc =
            (gemv_prep_remaining_row_tiles_calc > {10'd0, cmd_row_tiles_per_block}) ?
            cmd_row_tiles_per_block : gemv_prep_remaining_row_tiles_calc[5:0];
        gemv_prep_current_rows_calc =
            min_u32(({26'd0, gemv_prep_current_row_tiles_calc} << 5),
                    ({16'd0, cmd_mat_height} > gemv_prep_row_base_calc) ?
                    ({16'd0, cmd_mat_height} - gemv_prep_row_base_calc) : 32'd0);
        gemv_prep_next_row_tile_base_calc =
            gemv_prep_seed.row_tile_base + {10'd0, gemv_prep_current_row_tiles_calc};
        gemv_prep_next_row_base_calc =
            {11'd0, gemv_prep_next_row_tile_base_calc, 5'd0};
        gemv_prep_next_remaining_row_tiles_calc =
            (gemv_prep_next_row_tile_base_calc < cmd_total_row_tiles) ?
            (cmd_total_row_tiles - gemv_prep_next_row_tile_base_calc) : 16'd0;
        gemv_prep_next_row_tiles_calc =
            (gemv_prep_next_remaining_row_tiles_calc > {10'd0, cmd_row_tiles_per_block}) ?
            cmd_row_tiles_per_block : gemv_prep_next_remaining_row_tiles_calc[5:0];
        gemv_prep_next_rows_calc =
            min_u32(({26'd0, gemv_prep_next_row_tiles_calc} << 5),
                    ({16'd0, cmd_mat_height} > gemv_prep_next_row_base_calc) ?
                    ({16'd0, cmd_mat_height} - gemv_prep_next_row_base_calc) : 32'd0);
        gemv_prep_current_weight_bytes_calc =
            row_tile_bytes_times(gemv_block_prep_a.current_row_tiles, cmd_row_tile_bytes);
        gemv_prep_next_weight_bytes_calc =
            row_tile_bytes_times(gemv_block_prep_a.next_row_tiles, cmd_row_tile_bytes);

        gemv_start_col_tiles_calc =
            gemv_col_tiles_for_mode(gemv_start_mat_width_s1, gemv_start_gemv_mode_s1);
        gemv_start_act_bytes_calc =
            gemv_act_bytes_for_mode(gemv_start_col_tiles_calc, gemv_start_gemv_mode_s1);
        gemv_start_group_act_bytes_calc =
            gemv_group_act_bytes_for(gemv_start_act_bytes_calc,
                                     gemv_start_act_group_stride_s1,
                                     gemv_start_decode_flow_s1,
                                     gemv_start_cached_group_enable_s1);
        gemv_start_output_rows_calc =
            gemv_group_count_scale_u32({16'd0, gemv_start_mat_height_s1},
                                       gemv_start_decode_flow_s1);
        gemv_start_payload_bytes_calc =
            align_gemv_mvin_bytes(gemv_start_group_act_bytes_calc);
        gemv_start_act_scale_payload_bytes_calc =
            align_gemv_mvin_bytes(
                gemv_act_scale_bytes_for(gemv_start_col_tiles_calc,
                                         gemv_start_gemv_mode_s1,
                                         gemv_start_decode_flow_s1,
                                         gemv_start_enable_act_scale_s1));
        gemv_start_row_tile_bytes_calc =
            gemv_weight_payload_row_tile_bytes_for(gemv_start_col_tiles_calc,
                                                   gemv_start_gemv_mode_s1);
        gemv_start_weight_payload_bytes_calc =
            row_tile_bytes_times16(16'(row_tiles_for_rows({16'd0, gemv_start_mat_height_s1})),
                                   gemv_start_row_tile_bytes_calc);
        gemv_start_weight_scale_payload_bytes_calc =
            align_gemv_mvin_bytes(
                row_tile_bytes_times16(
                    16'(row_tiles_for_rows({16'd0, gemv_start_mat_height_s1})),
                    gemv_start_col_tiles_calc << 6));
        gemv_start_fixed_stride_min_stride_calc =
            (({16'd0, gemv_start_weight_capacity_tokens_s1} + 32'd63) >> 6) << 11;
        gemv_start_fixed_stride_active_row_tile_bytes_calc =
            gemv_start_col_tiles_calc << 11;
        gemv_start_rows_per_block_calc =
            32'(row_tiles_for_rows({16'd0, gemv_start_mat_height_s1})) << 5;
        gemv_start_error_code_calc =
            gemv_start_error_for(gemv_start_zero_shape_s1,
                                 gemv_start_unsupported_mode_s1,
                                 gemv_start_alignment_error_s1,
                                 gemv_start_height_overflow_s1 ||
                                     ((gemv_start_output_rows_calc << 1) > GEMV_SPM_BYTES),
                                 gemv_start_fixed_stride_enable_s1 &&
                                     ((gemv_start_mat_height_s1 != 16'd64) ||
                                      ({16'd0, gemv_start_mat_width_s1} >
                                       {16'd0, gemv_start_weight_capacity_tokens_s1}) ||
                                      (gemv_start_weight_capacity_tokens_s1 == 16'd0) ||
                                      (gemv_start_weight_row_tile_stride_bytes_s1 <
                                       gemv_start_fixed_stride_min_stride_calc)),
                                 gemv_start_fixed_stride_enable_s1 &&
                                     (gemv_start_weight_row_tile_stride_bytes_s1[7:0] != 8'd0),
                                 gemv_start_payload_bytes_calc,
                                 gemv_start_act_scale_payload_bytes_calc,
                                 gemv_start_enable_act_scale2_s1 ?
                                     gemv_start_act_scale_payload_bytes_calc : 32'd0,
                                 gemv_start_weight_payload_bytes_calc);
        gemv_start_commit_valid = gemv_start_s3_valid;
        gemv_start_commit_ok = gemv_start_s3_valid && gemv_start_ok_s3;

        if (gemv_start_commit_valid) begin
            if (gemv_start_ok_s3) begin
                api_start_event[API_GEMV_BLOCK] = 1'b1;
            end else begin
                api_error_event[API_GEMV_BLOCK] = 1'b1;
                api_error_code_event[API_GEMV_BLOCK] = gemv_start_error_code_s3;
            end
        end

        if (engine_state == ST_GEMV_WAIT_MVOUT &&
            (dma_mvout_resp_done_any || !mvout_active)) begin
            api_done_event[API_GEMV_BLOCK] = 1'b1;
        end
        if (write_fire && awaddr_legal) begin
            unique case (axi_awaddr[15:0])
                REG_IAR, REG_ISR, REG_DECODE_IAR_ALIAS, REG_DECODE_ISR_ALIAS: begin
                    if (axi_wstrb[0]) begin
                        irq_clear_mask_event = axi_wdata[7:0];
                    end
                end
                REG_GLOBAL_CLEAR: begin
                    api_clear_all_event = axi_wstrb[0] && axi_wdata[0];
                end
                REG_GENERIC_CONTROL, REG_GENERIC_CONTROL_ALIAS: begin
                    api_clear_all_event = axi_wstrb[0] &&
                                          (axi_wdata[0] || axi_wdata[1] || axi_wdata[2]);
                end
                REG_GEMV_STATUS: begin
                    api_clear_done_event[API_GEMV_BLOCK] = axi_wstrb[0] && axi_wdata[1];
                    api_clear_error_event[API_GEMV_BLOCK] = axi_wstrb[0] && axi_wdata[2];
                end
                REG_MLP_STATUS: begin
                    api_clear_done_event[API_MLP] = axi_wstrb[0] && axi_wdata[1];
                    api_clear_error_event[API_MLP] = axi_wstrb[0] && axi_wdata[2];
                end
                REG_ATTN_STATUS: begin
                    api_clear_done_event[API_ATTN] = axi_wstrb[0] && axi_wdata[1];
                    api_clear_error_event[API_ATTN] = axi_wstrb[0] && axi_wdata[2];
                end
                REG_DBG_MVIN_STATUS: begin
                    api_clear_done_event[API_DBG_MVIN] = axi_wstrb[0] && axi_wdata[1];
                    api_clear_error_event[API_DBG_MVIN] = axi_wstrb[0] && axi_wdata[2];
                end
                REG_DBG_MVOUT_STATUS: begin
                    api_clear_done_event[API_DBG_MVOUT] = axi_wstrb[0] && axi_wdata[1];
                    api_clear_error_event[API_DBG_MVOUT] = axi_wstrb[0] && axi_wdata[2];
                end
                REG_DBG_GEMV_STATUS: begin
                    api_clear_done_event[API_DBG_GEMV] = axi_wstrb[0] && axi_wdata[1];
                    api_clear_error_event[API_DBG_GEMV] = axi_wstrb[0] && axi_wdata[2];
                end
                REG_GEMV_CTRL: begin
                    if (axi_wstrb[0] && axi_wdata[0]) begin
                        if (command_active || gemv_start_accept_pending ||
                            (|dma_mvin_busy_status)) begin
                            api_error_event[API_GEMV_BLOCK] = 1'b1;
                            api_error_code_event[API_GEMV_BLOCK] = ERR_START_WHILE_BUSY;
                        end else begin
                            gemv_start_req_fire = 1'b1;
                        end
                    end
                end
                REG_MLP_CTRL: begin
                    if (axi_wstrb[0] && axi_wdata[0]) begin
                        api_error_event[API_MLP] = 1'b1;
                        api_error_code_event[API_MLP] =
                            (command_active || gemv_start_accept_pending) ?
                            ERR_START_WHILE_BUSY : ERR_UNSUPPORTED_MODE;
                    end
                end
                REG_ATTN_CTRL: begin
                    if (axi_wstrb[0] && axi_wdata[0]) begin
                        api_error_event[API_ATTN] = 1'b1;
                        api_error_code_event[API_ATTN] =
                            (command_active || gemv_start_accept_pending) ?
                            ERR_START_WHILE_BUSY : ERR_UNSUPPORTED_MODE;
                    end
                end
                REG_DBG_MVIN_CTRL: begin
                    if (axi_wstrb[0] && axi_wdata[0]) begin
                        api_error_event[API_DBG_MVIN] = 1'b1;
                        api_error_code_event[API_DBG_MVIN] = ERR_UNSUPPORTED_MODE;
                    end
                end
                REG_DBG_MVOUT_CTRL: begin
                    if (axi_wstrb[0] && axi_wdata[0]) begin
                        api_error_event[API_DBG_MVOUT] = 1'b1;
                        api_error_code_event[API_DBG_MVOUT] = ERR_UNSUPPORTED_MODE;
                    end
                end
                REG_DBG_GEMV_CTRL: begin
                    if (axi_wstrb[0] && axi_wdata[0]) begin
                        api_error_event[API_DBG_GEMV] = 1'b1;
                        api_error_code_event[API_DBG_GEMV] = ERR_UNSUPPORTED_MODE;
                    end
                end
                default: begin
                end
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_awaddr <= '0;
            axi_awaddr_valid <= 1'b0;
        end else if (write_fire) begin
            axi_awaddr_valid <= 1'b0;
        end else if (s_axi_awready && s_axi_awvalid) begin
            axi_awaddr <= s_axi_awaddr;
            axi_awaddr_valid <= 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_wdata <= '0;
            axi_wstrb <= '0;
            axi_wdata_valid <= 1'b0;
        end else if (write_fire) begin
            axi_wdata_valid <= 1'b0;
        end else if (s_axi_wready && s_axi_wvalid) begin
            axi_wdata <= s_axi_wdata;
            axi_wstrb <= s_axi_wstrb;
            axi_wdata_valid <= 1'b1;
        end
    end

    always_comb begin
        awaddr_legal = is_write_addr_legal(axi_awaddr[15:0]);
        araddr_legal = is_read_addr_legal(axi_araddr[15:0]);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= AXI_RESP_OKAY;
        end else if (write_fire) begin
            s_axi_bvalid <= 1'b1;
            s_axi_bresp  <= awaddr_legal ? AXI_RESP_OKAY : AXI_RESP_SLVERR;
        end else if (s_axi_bready && s_axi_bvalid) begin
            s_axi_bvalid <= 1'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b0;
            axi_araddr <= '0;
        end else if (!s_axi_arready && s_axi_arvalid) begin
            s_axi_arready <= 1'b1;
            axi_araddr <= s_axi_araddr;
        end else begin
            s_axi_arready <= 1'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rresp  <= AXI_RESP_OKAY;
            s_axi_rdata  <= '0;
        end else begin
            if (s_axi_arready && s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= araddr_legal ? AXI_RESP_OKAY : AXI_RESP_SLVERR;
                s_axi_rdata  <= araddr_legal ? read_data_for_addr(axi_araddr[15:0]) : 64'd0;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gemv_desc0 <= '0;
            gemv_desc1 <= '0;
            gemv_desc2 <= '0;
            gemv_desc3 <= '0;
            gemv_desc4 <= '0;
            gemv_desc5 <= '0;
            gemv_desc6 <= '0;
            gemv_desc7 <= '0;
            mlp_desc0 <= '0;
            mlp_desc1 <= '0;
            mlp_desc2 <= '0;
            mlp_desc3 <= '0;
            attn_desc0 <= '0;
            attn_desc1 <= '0;
            attn_desc2 <= '0;
            attn_desc3 <= '0;
            attn_desc4 <= '0;
            attn_desc5 <= '0;
            attn_desc6 <= '0;
            dbg_mvin_desc0 <= '0;
            dbg_mvin_desc1 <= '0;
            dbg_mvout_desc0 <= '0;
            dbg_mvout_desc1 <= '0;
            dbg_gemv_desc0 <= '0;
            dbg_gemv_desc1 <= '0;
            dbg_gemv_desc2 <= '0;
            dbg_gemv_desc3 <= '0;
            dbg_gemv_desc4 <= '0;
            irq_enable <= '0;
            master_enable <= 1'b0;
        end else if (write_fire && awaddr_legal) begin
            unique case (axi_awaddr[15:0])
                REG_GEMV_DESC0: gemv_desc0 <= apply_wstrb(gemv_desc0, axi_wdata, axi_wstrb);
                REG_GEMV_DESC1: gemv_desc1 <= apply_wstrb(gemv_desc1, axi_wdata, axi_wstrb);
                REG_GEMV_DESC2: gemv_desc2 <= apply_wstrb(gemv_desc2, axi_wdata, axi_wstrb);
                REG_GEMV_DESC3: gemv_desc3 <= apply_wstrb(gemv_desc3, axi_wdata, axi_wstrb);
                REG_GEMV_DESC4: gemv_desc4 <= apply_wstrb(gemv_desc4, axi_wdata, axi_wstrb);
                REG_GEMV_DESC5: gemv_desc5 <= apply_wstrb(gemv_desc5, axi_wdata, axi_wstrb);
                REG_GEMV_DESC6: gemv_desc6 <= apply_wstrb(gemv_desc6, axi_wdata, axi_wstrb);
                REG_GEMV_DESC7: gemv_desc7 <= apply_wstrb(gemv_desc7, axi_wdata, axi_wstrb);
                REG_MLP_DESC0:  mlp_desc0 <= apply_wstrb(mlp_desc0, axi_wdata, axi_wstrb);
                REG_MLP_DESC1:  mlp_desc1 <= apply_wstrb(mlp_desc1, axi_wdata, axi_wstrb);
                REG_MLP_DESC2:  mlp_desc2 <= apply_wstrb(mlp_desc2, axi_wdata, axi_wstrb);
                REG_MLP_DESC3:  mlp_desc3 <= apply_wstrb(mlp_desc3, axi_wdata, axi_wstrb);
                REG_ATTN_DESC0: attn_desc0 <= apply_wstrb(attn_desc0, axi_wdata, axi_wstrb);
                REG_ATTN_DESC1: attn_desc1 <= apply_wstrb(attn_desc1, axi_wdata, axi_wstrb);
                REG_ATTN_DESC2: attn_desc2 <= apply_wstrb(attn_desc2, axi_wdata, axi_wstrb);
                REG_ATTN_DESC3: attn_desc3 <= apply_wstrb(attn_desc3, axi_wdata, axi_wstrb);
                REG_ATTN_DESC4: attn_desc4 <= apply_wstrb(attn_desc4, axi_wdata, axi_wstrb);
                REG_ATTN_DESC5: attn_desc5 <= apply_wstrb(attn_desc5, axi_wdata, axi_wstrb);
                REG_ATTN_DESC6: attn_desc6 <= apply_wstrb(attn_desc6, axi_wdata, axi_wstrb);
                REG_DBG_MVIN_DESC0: dbg_mvin_desc0 <= apply_wstrb(dbg_mvin_desc0, axi_wdata, axi_wstrb);
                REG_DBG_MVIN_DESC1: dbg_mvin_desc1 <= apply_wstrb(dbg_mvin_desc1, axi_wdata, axi_wstrb);
                REG_DBG_MVOUT_DESC0: dbg_mvout_desc0 <= apply_wstrb(dbg_mvout_desc0, axi_wdata, axi_wstrb);
                REG_DBG_MVOUT_DESC1: dbg_mvout_desc1 <= apply_wstrb(dbg_mvout_desc1, axi_wdata, axi_wstrb);
                REG_DBG_GEMV_DESC0: dbg_gemv_desc0 <= apply_wstrb(dbg_gemv_desc0, axi_wdata, axi_wstrb);
                REG_DBG_GEMV_DESC1: dbg_gemv_desc1 <= apply_wstrb(dbg_gemv_desc1, axi_wdata, axi_wstrb);
                REG_DBG_GEMV_DESC2: dbg_gemv_desc2 <= apply_wstrb(dbg_gemv_desc2, axi_wdata, axi_wstrb);
                REG_DBG_GEMV_DESC3: dbg_gemv_desc3 <= apply_wstrb(dbg_gemv_desc3, axi_wdata, axi_wstrb);
                REG_DBG_GEMV_DESC4: dbg_gemv_desc4 <= apply_wstrb(dbg_gemv_desc4, axi_wdata, axi_wstrb);
                REG_MER, REG_DECODE_MER_ALIAS: begin
                    if (axi_wstrb[0]) begin
                        master_enable <= axi_wdata[0];
                    end
                end
                REG_IER, REG_DECODE_IER_ALIAS: begin
                    if (axi_wstrb[0]) begin
                        irq_enable <= axi_wdata[7:0];
                    end
                end
                default: begin
                end
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gemv_start_s0_valid <= 1'b0;
            gemv_start_s1_valid <= 1'b0;
            gemv_start_s2_valid <= 1'b0;
            gemv_start_s3_valid <= 1'b0;
            gemv_start_act_dma_base_s1 <= '0;
            gemv_start_act_scale_dma_base_s1 <= '0;
            gemv_start_act_scale2_dma_base_s1 <= '0;
            gemv_start_rope_lut_dma_base_s1 <= '0;
            gemv_start_weight_dma_base_s1 <= '0;
            gemv_start_weight_scale_dma_base_s1 <= '0;
            gemv_start_output_dma_base_s1 <= '0;
            gemv_start_mat_width_s1 <= '0;
            gemv_start_mat_height_s1 <= '0;
            gemv_start_cache_cell_s1 <= '0;
            gemv_start_gemv_mode_s1 <= '0;
            gemv_start_output_precision_s1 <= '0;
            gemv_start_enable_act_scale_s1 <= 1'b0;
            gemv_start_enable_act_scale2_s1 <= 1'b0;
            gemv_start_fixed_stride_enable_s1 <= 1'b0;
            gemv_start_cached_group_enable_s1 <= 1'b0;
            gemv_start_uses_weight_scale_s1 <= 1'b0;
            gemv_start_uses_rope_lut_s1 <= 1'b0;
            gemv_start_weight_row_tile_stride_bytes_s1 <= '0;
            gemv_start_weight_capacity_tokens_s1 <= '0;
            gemv_start_act_group_stride_s1 <= '0;
            gemv_start_decode_flow_s1 <= '0;
            gemv_start_zero_shape_s1 <= 1'b0;
            gemv_start_unsupported_mode_s1 <= 1'b0;
            gemv_start_alignment_error_s1 <= 1'b0;
            gemv_start_height_overflow_s1 <= 1'b0;
            gemv_start_act_dma_base_s2 <= '0;
            gemv_start_act_scale_dma_base_s2 <= '0;
            gemv_start_act_scale2_dma_base_s2 <= '0;
            gemv_start_rope_lut_dma_base_s2 <= '0;
            gemv_start_weight_dma_base_s2 <= '0;
            gemv_start_weight_scale_dma_base_s2 <= '0;
            gemv_start_output_dma_base_s2 <= '0;
            gemv_start_mat_width_s2 <= '0;
            gemv_start_mat_height_s2 <= '0;
            gemv_start_cache_cell_s2 <= '0;
            gemv_start_gemv_mode_s2 <= '0;
            gemv_start_output_precision_s2 <= '0;
            gemv_start_enable_act_scale_s2 <= 1'b0;
            gemv_start_enable_act_scale2_s2 <= 1'b0;
            gemv_start_fixed_stride_enable_s2 <= 1'b0;
            gemv_start_cached_group_enable_s2 <= 1'b0;
            gemv_start_uses_weight_scale_s2 <= 1'b0;
            gemv_start_uses_rope_lut_s2 <= 1'b0;
            gemv_start_weight_row_tile_stride_bytes_s2 <= '0;
            gemv_start_weight_capacity_tokens_s2 <= '0;
            gemv_start_act_group_stride_s2 <= '0;
            gemv_start_decode_flow_s2 <= '0;
            gemv_start_col_tiles_s2 <= '0;
            gemv_start_act_bytes_s2 <= '0;
            gemv_start_group_act_bytes_s2 <= '0;
            gemv_start_output_rows_s2 <= '0;
            gemv_start_payload_bytes_s2 <= '0;
            gemv_start_act_scale_payload_bytes_s2 <= '0;
            gemv_start_act_scale2_payload_bytes_s2 <= '0;
            gemv_start_row_tile_bytes_s2 <= '0;
            gemv_start_weight_payload_bytes_s2 <= '0;
            gemv_start_weight_scale_payload_bytes_s2 <= '0;
            gemv_start_fixed_stride_active_row_tile_bytes_s2 <= '0;
            gemv_start_rows_per_block_s2 <= '0;
            gemv_start_error_code_s2 <= ERR_NONE;
            gemv_start_act_dma_base_s3 <= '0;
            gemv_start_act_scale_dma_base_s3 <= '0;
            gemv_start_act_scale2_dma_base_s3 <= '0;
            gemv_start_rope_lut_dma_base_s3 <= '0;
            gemv_start_weight_dma_base_s3 <= '0;
            gemv_start_weight_scale_dma_base_s3 <= '0;
            gemv_start_output_dma_base_s3 <= '0;
            gemv_start_mat_width_s3 <= '0;
            gemv_start_mat_height_s3 <= '0;
            gemv_start_cache_cell_s3 <= '0;
            gemv_start_gemv_mode_s3 <= '0;
            gemv_start_output_precision_s3 <= '0;
            gemv_start_enable_act_scale_s3 <= 1'b0;
            gemv_start_enable_act_scale2_s3 <= 1'b0;
            gemv_start_fixed_stride_enable_s3 <= 1'b0;
            gemv_start_cached_group_enable_s3 <= 1'b0;
            gemv_start_uses_weight_scale_s3 <= 1'b0;
            gemv_start_uses_rope_lut_s3 <= 1'b0;
            gemv_start_weight_row_tile_stride_bytes_s3 <= '0;
            gemv_start_weight_capacity_tokens_s3 <= '0;
            gemv_start_act_group_stride_s3 <= '0;
            gemv_start_decode_flow_s3 <= '0;
            gemv_start_col_tiles_s3 <= '0;
            gemv_start_act_bytes_s3 <= '0;
            gemv_start_group_act_bytes_s3 <= '0;
            gemv_start_output_rows_s3 <= '0;
            gemv_start_payload_bytes_s3 <= '0;
            gemv_start_act_scale_payload_bytes_s3 <= '0;
            gemv_start_act_scale2_payload_bytes_s3 <= '0;
            gemv_start_row_tile_bytes_s3 <= '0;
            gemv_start_weight_payload_bytes_s3 <= '0;
            gemv_start_weight_scale_payload_bytes_s3 <= '0;
            gemv_start_fixed_stride_active_row_tile_bytes_s3 <= '0;
            gemv_start_rows_per_block_s3 <= '0;
            gemv_start_error_code_s3 <= ERR_NONE;
            gemv_start_ok_s3 <= 1'b0;
        end else begin
            gemv_start_s0_valid <= gemv_start_req_fire;
            gemv_start_s1_valid <= gemv_start_s0_valid;
            gemv_start_s2_valid <= gemv_start_s1_valid;
            gemv_start_s3_valid <= gemv_start_s2_valid;

            if (gemv_start_s0_valid) begin
                gemv_start_act_dma_base_s1 <= gemv_desc0[31:0];
                gemv_start_weight_dma_base_s1 <= gemv_desc0[63:32];
                gemv_start_act_scale_dma_base_s1 <= gemv_desc1[31:0];
                gemv_start_act_scale2_dma_base_s1 <= gemv_desc7[31:0];
                gemv_start_weight_scale_dma_base_s1 <= gemv_desc6[31:0];
                gemv_start_rope_lut_dma_base_s1 <= gemv_desc6[63:32];
                gemv_start_output_dma_base_s1 <= gemv_desc1[63:32];
                gemv_start_mat_width_s1 <= gemv_desc2[15:0];
                gemv_start_mat_height_s1 <= gemv_desc2[31:16];
                gemv_start_cache_cell_s1 <= gemv_desc2[47:32];
                gemv_start_gemv_mode_s1 <= gemv_desc2[55:48];
                gemv_start_output_precision_s1 <= {7'd0, gemv_desc2[56]};
                gemv_start_enable_act_scale_s1 <= gemv_desc2[57];
                gemv_start_enable_act_scale2_s1 <= gemv_desc2[60];
                gemv_start_fixed_stride_enable_s1 <= gemv_desc2[58];
                gemv_start_cached_group_enable_s1 <= gemv_desc2[59];
                gemv_start_weight_row_tile_stride_bytes_s1 <= gemv_desc4[31:0];
                gemv_start_weight_capacity_tokens_s1 <= gemv_desc4[47:32];
                gemv_start_act_group_stride_s1 <= gemv_desc5[31:0];
                gemv_start_decode_flow_s1 <= gemv_desc3;
                gemv_start_uses_weight_scale_s1 <=
                    gemv_uses_weight_scale(gemv_desc2[55:48], gemv_desc3);
                gemv_start_uses_rope_lut_s1 <= gemv_uses_rope_lut(gemv_desc3);
                gemv_start_zero_shape_s1 <= (gemv_desc2[15:0] == 16'd0) ||
                                            (gemv_desc2[31:16] == 16'd0);
                gemv_start_unsupported_mode_s1 <= (gemv_desc2[55:48] > 8'd1) ||
                                                  (gemv_desc2[63:61] != 3'd0) ||
                                                  (gemv_desc2[60] && !gemv_desc2[57]) ||
                                                  (gemv_desc2[60] && gemv_desc2[59]) ||
                                                  (gemv_desc2[59] && (gemv_desc3[26:25] == 2'd0)) ||
                                                  (gemv_desc2[58] && (gemv_desc2[55:48] != 8'd1));
                gemv_start_alignment_error_s1 <= (gemv_desc0[7:0] != 8'd0) ||
                                                 (gemv_desc0[39:32] != 8'd0) ||
                                                 (gemv_desc1[39:32] != 8'd0) ||
                                                 (gemv_uses_weight_scale(gemv_desc2[55:48], gemv_desc3) &&
                                                  (gemv_desc6[7:0] != 8'd0)) ||
                                                 (gemv_uses_rope_lut(gemv_desc3) &&
                                                  (gemv_desc6[39:32] != 8'd0)) ||
                                                 (gemv_desc2[57] && (gemv_desc1[7:0] != 8'd0)) ||
                                                 (gemv_desc2[60] && (gemv_desc7[7:0] != 8'd0)) ||
                                                 ((gemv_desc5[31:0] != 32'd0) &&
                                                  (gemv_desc2[59] ? (gemv_desc5[7:0] != 8'd0) :
                                                                    (gemv_desc5[5:0] != 6'd0)));
                gemv_start_height_overflow_s1 <= ({15'd0, gemv_desc2[31:16], 1'b0} > GEMV_SPM_BYTES);
            end

            if (gemv_start_s1_valid) begin
                gemv_start_act_dma_base_s2 <= gemv_start_act_dma_base_s1;
                gemv_start_act_scale_dma_base_s2 <= gemv_start_act_scale_dma_base_s1;
                gemv_start_act_scale2_dma_base_s2 <= gemv_start_act_scale2_dma_base_s1;
                gemv_start_rope_lut_dma_base_s2 <= gemv_start_rope_lut_dma_base_s1;
                gemv_start_weight_dma_base_s2 <= gemv_start_weight_dma_base_s1;
                gemv_start_weight_scale_dma_base_s2 <= gemv_start_weight_scale_dma_base_s1;
                gemv_start_output_dma_base_s2 <= gemv_start_output_dma_base_s1;
                gemv_start_mat_width_s2 <= gemv_start_mat_width_s1;
                gemv_start_mat_height_s2 <= gemv_start_mat_height_s1;
                gemv_start_cache_cell_s2 <= gemv_start_cache_cell_s1;
                gemv_start_gemv_mode_s2 <= gemv_start_gemv_mode_s1;
                gemv_start_output_precision_s2 <= gemv_start_output_precision_s1;
                gemv_start_enable_act_scale_s2 <= gemv_start_enable_act_scale_s1;
                gemv_start_enable_act_scale2_s2 <= gemv_start_enable_act_scale2_s1;
                gemv_start_fixed_stride_enable_s2 <= gemv_start_fixed_stride_enable_s1;
                gemv_start_cached_group_enable_s2 <= gemv_start_cached_group_enable_s1;
                gemv_start_uses_weight_scale_s2 <= gemv_start_uses_weight_scale_s1;
                gemv_start_uses_rope_lut_s2 <= gemv_start_uses_rope_lut_s1;
                gemv_start_weight_row_tile_stride_bytes_s2 <=
                    gemv_start_weight_row_tile_stride_bytes_s1;
                gemv_start_weight_capacity_tokens_s2 <= gemv_start_weight_capacity_tokens_s1;
                gemv_start_act_group_stride_s2 <= gemv_start_act_group_stride_s1;
                gemv_start_decode_flow_s2 <= gemv_start_decode_flow_s1;
                gemv_start_col_tiles_s2 <= gemv_start_col_tiles_calc;
                gemv_start_act_bytes_s2 <= gemv_start_act_bytes_calc;
                gemv_start_group_act_bytes_s2 <= gemv_start_group_act_bytes_calc;
                gemv_start_output_rows_s2 <= gemv_start_output_rows_calc;
                gemv_start_payload_bytes_s2 <= gemv_start_payload_bytes_calc;
                gemv_start_act_scale_payload_bytes_s2 <= gemv_start_act_scale_payload_bytes_calc;
                gemv_start_act_scale2_payload_bytes_s2 <=
                    gemv_start_enable_act_scale2_s1 ? gemv_start_act_scale_payload_bytes_calc : 32'd0;
                gemv_start_row_tile_bytes_s2 <= gemv_start_row_tile_bytes_calc;
                gemv_start_weight_payload_bytes_s2 <= gemv_start_weight_payload_bytes_calc;
                gemv_start_weight_scale_payload_bytes_s2 <=
                    gemv_start_weight_scale_payload_bytes_calc;
                gemv_start_fixed_stride_active_row_tile_bytes_s2 <=
                    gemv_start_fixed_stride_active_row_tile_bytes_calc;
                gemv_start_rows_per_block_s2 <= gemv_start_rows_per_block_calc;
                gemv_start_error_code_s2 <= gemv_start_error_code_calc;
            end

            if (gemv_start_s2_valid) begin
                gemv_start_act_dma_base_s3 <= gemv_start_act_dma_base_s2;
                gemv_start_act_scale_dma_base_s3 <= gemv_start_act_scale_dma_base_s2;
                gemv_start_act_scale2_dma_base_s3 <= gemv_start_act_scale2_dma_base_s2;
                gemv_start_rope_lut_dma_base_s3 <= gemv_start_rope_lut_dma_base_s2;
                gemv_start_weight_dma_base_s3 <= gemv_start_weight_dma_base_s2;
                gemv_start_weight_scale_dma_base_s3 <= gemv_start_weight_scale_dma_base_s2;
                gemv_start_output_dma_base_s3 <= gemv_start_output_dma_base_s2;
                gemv_start_mat_width_s3 <= gemv_start_mat_width_s2;
                gemv_start_mat_height_s3 <= gemv_start_mat_height_s2;
                gemv_start_cache_cell_s3 <= gemv_start_cache_cell_s2;
                gemv_start_gemv_mode_s3 <= gemv_start_gemv_mode_s2;
                gemv_start_output_precision_s3 <= gemv_start_output_precision_s2;
                gemv_start_enable_act_scale_s3 <= gemv_start_enable_act_scale_s2;
                gemv_start_enable_act_scale2_s3 <= gemv_start_enable_act_scale2_s2;
                gemv_start_fixed_stride_enable_s3 <= gemv_start_fixed_stride_enable_s2;
                gemv_start_cached_group_enable_s3 <= gemv_start_cached_group_enable_s2;
                gemv_start_uses_weight_scale_s3 <= gemv_start_uses_weight_scale_s2;
                gemv_start_uses_rope_lut_s3 <= gemv_start_uses_rope_lut_s2;
                gemv_start_weight_row_tile_stride_bytes_s3 <=
                    gemv_start_weight_row_tile_stride_bytes_s2;
                gemv_start_weight_capacity_tokens_s3 <= gemv_start_weight_capacity_tokens_s2;
                gemv_start_act_group_stride_s3 <= gemv_start_act_group_stride_s2;
                gemv_start_decode_flow_s3 <= gemv_start_decode_flow_s2;
                gemv_start_col_tiles_s3 <= gemv_start_col_tiles_s2;
                gemv_start_act_bytes_s3 <= gemv_start_act_bytes_s2;
                gemv_start_group_act_bytes_s3 <= gemv_start_group_act_bytes_s2;
                gemv_start_output_rows_s3 <= gemv_start_output_rows_s2;
                gemv_start_payload_bytes_s3 <= gemv_start_payload_bytes_s2;
                gemv_start_act_scale_payload_bytes_s3 <= gemv_start_act_scale_payload_bytes_s2;
                gemv_start_act_scale2_payload_bytes_s3 <= gemv_start_act_scale2_payload_bytes_s2;
                gemv_start_row_tile_bytes_s3 <= gemv_start_row_tile_bytes_s2;
                gemv_start_weight_payload_bytes_s3 <= gemv_start_weight_payload_bytes_s2;
                gemv_start_weight_scale_payload_bytes_s3 <=
                    gemv_start_weight_scale_payload_bytes_s2;
                gemv_start_fixed_stride_active_row_tile_bytes_s3 <=
                    gemv_start_fixed_stride_active_row_tile_bytes_s2;
                gemv_start_rows_per_block_s3 <= gemv_start_rows_per_block_s2;
                gemv_start_error_code_s3 <= gemv_start_error_code_s2;
                gemv_start_ok_s3 <= (gemv_start_error_code_s2 == ERR_NONE);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gemv_block_prep_a <= '0;
            gemv_block_prep_b <= '0;
            gemv_block <= '0;
            gemv_block_prep_a_valid <= 1'b0;
            gemv_block_prep_b_valid <= 1'b0;
            gemv_block_valid <= 1'b0;
        end else begin
            gemv_block_prep_a_valid <= 1'b0;
            gemv_block_prep_b_valid <= 1'b0;
            if (gemv_prep_req) begin
                gemv_block_prep_a_valid <= 1'b1;
                gemv_block_prep_b_valid <= 1'b0;
                gemv_block_valid <= 1'b0;
                gemv_block_prep_a.row_base <= gemv_prep_row_base_calc;
                gemv_block_prep_a.block_index <= gemv_prep_seed.block_index;
                gemv_block_prep_a.weight_offset <= gemv_prep_seed.weight_offset;
                gemv_block_prep_a.buf_select <= gemv_prep_seed.buf_select;
                gemv_block_prep_a.current_rows <= gemv_prep_current_rows_calc;
                gemv_block_prep_a.next_row_base <= gemv_prep_next_row_base_calc;
                gemv_block_prep_a.next_rows <= gemv_prep_next_rows_calc;
                gemv_block_prep_a.current_row_tiles <= gemv_prep_current_row_tiles_calc;
                gemv_block_prep_a.next_row_tiles <= gemv_prep_next_row_tiles_calc;
                gemv_block_prep_a.has_next <= (gemv_prep_next_row_tile_base_calc < cmd_total_row_tiles);
            end

            if (gemv_block_prep_a_valid) begin
                gemv_block_prep_b_valid <= 1'b1;
                gemv_block_prep_b.row_base <= gemv_block_prep_a.row_base;
                gemv_block_prep_b.block_index <= gemv_block_prep_a.block_index;
                gemv_block_prep_b.weight_offset <= gemv_block_prep_a.weight_offset;
                gemv_block_prep_b.current_rows <= gemv_block_prep_a.current_rows;
                gemv_block_prep_b.next_row_base <= gemv_block_prep_a.next_row_base;
                gemv_block_prep_b.current_weight_bytes <= gemv_prep_current_weight_bytes_calc;
                gemv_block_prep_b.next_weight_bytes <= gemv_prep_next_weight_bytes_calc;
                gemv_block_prep_b.current_buf_base <=
                    gemv_block_prep_a.buf_select ? GEMV_PONG_BASE : GEMV_PING_BASE;
                gemv_block_prep_b.next_buf_base <=
                    gemv_block_prep_a.buf_select ? GEMV_PING_BASE : GEMV_PONG_BASE;
                gemv_block_prep_b.current_weight_dma_addr <=
                    cmd_weight_dma_base + gemv_block_prep_a.weight_offset;
                gemv_block_prep_b.output_addr <= GEMV_OUTPUT_BASE + (gemv_block_prep_a.row_base << 1);
                gemv_block_prep_b.output_dram_addr <= cmd_output_dma_base +
                    (gemv_block_prep_a.row_base << (cmd_output_precision[0] ? 2 : 1));
                gemv_block_prep_b.bypass_flow <= make_bypass_output_flow(gemv_block_prep_a.current_rows[15:0]);
                gemv_block_prep_b.has_next <= gemv_block_prep_a.has_next;
                gemv_block_prep_b.buf_select <= gemv_block_prep_a.buf_select;
            end

            if (gemv_block_prep_b_valid) begin
                gemv_block.row_base <= gemv_block_prep_b.row_base;
                gemv_block.block_index <= gemv_block_prep_b.block_index;
                gemv_block.current_rows <= gemv_block_prep_b.current_rows;
                gemv_block.next_row_base <= gemv_block_prep_b.next_row_base;
                gemv_block.next_weight_offset <=
                    gemv_block_prep_b.weight_offset + gemv_block_prep_b.current_weight_bytes;
                gemv_block.current_weight_bytes <= gemv_block_prep_b.current_weight_bytes;
                gemv_block.next_weight_bytes <= gemv_block_prep_b.next_weight_bytes;
                gemv_block.current_buf_base <= gemv_block_prep_b.current_buf_base;
                gemv_block.next_buf_base <= gemv_block_prep_b.next_buf_base;
                gemv_block.current_weight_dma_addr <= gemv_block_prep_b.current_weight_dma_addr;
                gemv_block.next_weight_dma_addr <=
                    gemv_block_prep_b.current_weight_dma_addr + gemv_block_prep_b.current_weight_bytes;
                gemv_block.output_addr <= gemv_block_prep_b.output_addr;
                gemv_block.output_dram_addr <= gemv_block_prep_b.output_dram_addr;
                gemv_block.bypass_flow <= gemv_block_prep_b.bypass_flow;
                gemv_block.has_next <= gemv_block_prep_b.has_next;
                gemv_block.buf_select <= gemv_block_prep_b.buf_select;
                gemv_block_valid <= 1'b1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            api_busy <= '0;
            api_done <= '0;
            api_error <= '0;
            if (DETAILED_ERROR_STATUS_ON) begin
                api_error_code <= '0;
            end
            if (PROFILE_COUNTERS_ON) begin
                api_accepted_count <= '0;
            end
            irq_status <= '0;
        end else begin
            if (api_clear_all_event) begin
                api_done <= '0;
                api_error <= '0;
                if (DETAILED_ERROR_STATUS_ON) begin
                    api_error_code <= '0;
                end
                irq_status <= '0;
            end else begin
                irq_status <= irq_status & ~irq_clear_mask_event;
                for (int i = 0; i < API_COUNT; i++) begin
                    if (api_clear_done_event[i]) begin
                        api_done[i] <= 1'b0;
                        irq_status[i] <= 1'b0;
                    end
                    if (api_clear_error_event[i]) begin
                        api_error[i] <= 1'b0;
                        if (DETAILED_ERROR_STATUS_ON) begin
                            api_error_code[i] <= ERR_NONE;
                        end
                    end
                end
                if ((api_error & ~api_clear_error_event) == '0) begin
                    irq_status[IRQ_ERROR] <= 1'b0;
                end
            end

            for (int i = 0; i < API_COUNT; i++) begin
                if (api_start_event[i]) begin
                    api_busy[i] <= 1'b1;
                    api_done[i] <= 1'b0;
                    api_error[i] <= 1'b0;
                    if (DETAILED_ERROR_STATUS_ON) begin
                        api_error_code[i] <= ERR_NONE;
                    end
                    if (PROFILE_COUNTERS_ON) begin
                        api_accepted_count[i] <= api_accepted_count[i] + 1'b1;
                    end
                end
                if (api_error_event[i]) begin
                    api_error[i] <= 1'b1;
                    if (DETAILED_ERROR_STATUS_ON) begin
                        api_error_code[i] <= api_error_code_event[i];
                    end
                    irq_status[IRQ_ERROR] <= 1'b1;
                end
                if (api_done_event[i]) begin
                    api_busy[i] <= 1'b0;
                    api_done[i] <= 1'b1;
                    irq_status[i] <= 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            engine_state <= ST_IDLE;
            command_active <= 1'b0;
            active_api <= '0;
            cmd_act_dma_base <= '0;
            cmd_act_scale_dma_base <= '0;
            cmd_rope_lut_dma_base <= '0;
            cmd_weight_dma_base <= '0;
            cmd_weight_scale_dma_base <= '0;
            cmd_act_scale2_dma_base <= '0;
            cmd_output_dma_base <= '0;
            cmd_mat_width <= '0;
            cmd_mat_height <= '0;
            cmd_gemv_mode <= '0;
            cmd_output_precision <= '0;
            cmd_col_tiles <= '0;
            cmd_act_data_bytes <= '0;
            cmd_act_payload_bytes <= '0;
            cmd_act_scale_payload_bytes <= '0;
            cmd_act_scale2_payload_bytes <= '0;
            cmd_act_group_stride <= '0;
            cmd_mvout_rows <= '0;
            cmd_row_tile_bytes <= '0;
            cmd_weight_payload_bytes <= '0;
            cmd_cache_cell <= '0;
            cmd_enable_act_scale <= 1'b0;
            cmd_enable_act_scale2 <= 1'b0;
            cmd_fixed_stride_enable <= 1'b0;
            cmd_cached_group_enable <= 1'b0;
            cmd_uses_weight_scale <= 1'b0;
            cmd_uses_rope_lut <= 1'b0;
            cmd_weight_row_tile_stride_bytes <= '0;
            cmd_weight_capacity_tokens <= '0;
            cmd_fixed_stride_active_row_tile_bytes <= '0;
            cmd_weight_scale_payload_bytes <= '0;
            cmd_fixed_stride_segments_issued <= '0;
            cmd_fixed_stride_segment_count <= '0;
            cmd_decode_flow <= '0;
            cmd_rows_per_block <= '0;
            cmd_total_row_tiles <= '0;
            cmd_row_tiles_per_block <= '0;
            cmd_row_base <= '0;
            cmd_block_index <= '0;
            cmd_buf_select <= 1'b0;
            cmd_group_count_m1 <= '0;
            cmd_group_index <= '0;
            cmd_act_slot_select <= 1'b0;
            prefetch_active <= 1'b0;
            prefetch_done_seen <= 1'b0;
            gemv_done_seen <= 1'b0;
            gemv_mvin_done_q <= 1'b0;
            gemv_wait_mvin_done_q <= 1'b0;
            gemv_wait_matvec_done_q <= 1'b0;
            mvout_active <= 1'b0;
            overlap_seen <= 1'b0;
            mvout_overlap_seen <= 1'b0;
            last_pingpong_buffer <= 1'b0;
            if (PROFILE_COUNTERS_ON) begin
                prefetch_wait_cycles <= '0;
                mvout_wait_cycles <= '0;
                completed_block_count <= '0;
            end
            gemv_prep_req <= 1'b0;
            gemv_prep_seed <= '0;
            gemv_active_block <= '0;

            mvin_dram_addr <= '0;
            mvin_row_num <= '0;
            mvin_sram_addr <= '0;
            mvin_col_num <= '0;
            mvout_dram_addr <= '0;
            mvout_row_num <= '0;
            mvout_sram_addr <= '0;
            mvout_col_num <= '0;
            cfg_mvin_input_type <= '0;
            cfg_mvout_output_type <= '0;
            cfg_mvin_input_precision <= '0;
            cfg_mvout_output_precision <= '0;
            cfg_mvin_is_quant <= 1'b0;
            cfg_mvout_is_quant <= 1'b0;
            cfg_mvin_dest <= 1'b0;
            cfg_mvout_source <= 1'b0;
            cfg_mvout_per_channel <= 1'b0;
            cfg_mvin_isbias <= 1'b0;
            cfg_mvin_sram_stride <= '0;
            cfg_mvin_dram_stride <= '0;
            cfg_mvout_sram_stride <= '0;
            cfg_mvout_dram_stride <= '0;
            cfg_mvin_input_zeropoint <= '0;
            cfg_mvout_output_zeropoint <= '0;
            cfg_mvin_input_scale <= '0;
            cfg_mvout_output_scale <= '0;
            cfg_mvin_input_scale_shift <= '0;
            cfg_mvout_output_scale_shift <= '0;
            dma_mvin_dram_addr <= '0;
            dma_mvin_row_num <= '0;
            dma_mvin_sram_addr <= '0;
            dma_mvin_col_num <= '0;
            dma_mvout_dram_addr <= '0;
            dma_mvout_row_num <= '0;
            dma_mvout_sram_addr <= '0;
            dma_mvout_col_num <= '0;
            dma_cfg_mvin_input_type <= '0;
            dma_cfg_mvout_output_type <= '0;
            dma_cfg_mvin_input_precision <= '0;
            dma_cfg_mvout_output_precision <= '0;
            dma_cfg_mvin_is_quant <= '0;
            dma_cfg_mvout_is_quant <= '0;
            dma_cfg_mvin_dest <= '0;
            dma_cfg_mvout_source <= '0;
            dma_cfg_mvout_per_channel <= '0;
            dma_cfg_mvin_isbias <= '0;
            dma_cfg_mvin_scale_target <= '0;
            dma_cfg_mvin_stream_fifo_fill <= '0;
            dma_cfg_mvin_sram_stride <= '0;
            dma_cfg_mvin_dram_stride <= '0;
            dma_cfg_mvout_sram_stride <= '0;
            dma_cfg_mvout_dram_stride <= '0;
            dma_cfg_mvin_input_zeropoint <= '0;
            dma_cfg_mvout_output_zeropoint <= '0;
            dma_cfg_mvin_input_scale <= '0;
            dma_cfg_mvout_output_scale <= '0;
            dma_cfg_mvin_input_scale_shift <= '0;
            dma_cfg_mvout_output_scale_shift <= '0;
            dma_mvin_req_en <= '0;
            dma_mvout_req_en <= '0;

            cfg_compute_dataflow <= 1'b0;
            cfg_compute_padding_left <= '0;
            cfg_compute_padding_right <= '0;
            cfg_compute_padding_top <= '0;
            cfg_compute_padding_bottom <= '0;
            cfg_compute_padding_mode <= '0;
            cfg_compute_weight_shape <= '0;
            cfg_compute_weight_stride <= '0;
            cfg_compute_weight_dilation <= '0;
            cfg_compute_is_groupconv <= 1'b0;
            cfg_compute_int_type <= '0;
            cfg_compute_optype <= '0;
            cfg_compute_accout_dest <= 1'b0;
            cfg_compute_asymmetric_activations <= 1'b0;
            cfg_decode_flow <= '0;
            cfg_compute_inputa_zeropoint <= '0;
            cfg_compute_inputb_zeropoint <= '0;
            cfg_compute_output_zeropoint <= '0;
            cfg_compute_output_scale <= '0;
            cfg_compute_output_scale_shift <= '0;
            cfg_accu_biaspsum_addr <= '0;
            cfg_accu_biaspsum_stride <= '0;
            cfg_accu_biaspsum_width <= '0;
            cfg_accu_biaspsum_height <= '0;
            cfg_accu_output_addr <= '0;
            cfg_accu_output_stride <= '0;
            cfg_accu_isaccu <= 1'b0;
            cfg_accu_relu <= 1'b0;
            cfg_accu_relu_type <= '0;
            cfg_accu_isbias <= 1'b0;
            sa_input_a_spm_addr <= '0;
            sa_input_a_col_num <= '0;
            sa_input_a_row_num <= '0;
            sa_input_a_stride <= '0;
            sa_input_b_spm_addr <= '0;
            sa_input_b_col_num <= '0;
            sa_input_b_row_num <= '0;
            sa_input_b_stride <= '0;
            sa_req_en <= 1'b0;
            matadd_input_a_addr <= '0;
            matadd_input_b_addr <= '0;
            matadd_input_col_num <= '0;
            matadd_input_row_num <= '0;
            matadd_output_addr <= '0;
            matadd_req_en <= 1'b0;
            matvec_input_mat_addr <= '0;
            matvec_input_vec_addr <= '0;
            matvec_input_act_scale_addr <= '0;
            matvec_input_act_scale2_addr <= '0;
            matvec_cache_cell_idx <= '0;
            matvec_input_act_group_stride_bytes <= '0;
            matvec_act_frac_cfg <= '0;
            matvec_input_mat_width <= '0;
            matvec_input_mat_height <= '0;
            matvec_input_mat_stride <= '0;
            matvec_input_vec_stride <= '0;
            matvec_weight_stream_mode <= 1'b0;
            matvec_act_scale_enable <= 1'b0;
            matvec_act_scale2_enable <= 1'b0;
            matvec_req_en <= 1'b0;
            cfg_sfu_op <= '0;
            cfg_sfu_int_type <= '0;
            cfg_sfu_is_quant <= 1'b0;
            cfg_trans_out_ispad_row <= 1'b0;
            cfg_trans_out_ispad_col <= 1'b0;
            cfg_sfu_output_zeropoint <= '0;
            cfg_sfu_input_zeropoint <= '0;
            cfg_sfu_input_scale <= '0;
            cfg_sfu_output_scale <= '0;
            cfg_sfu_input_scale_shift <= '0;
            cfg_sfu_output_scale_shift <= '0;
            sfu_input_sram_addr <= '0;
            sfu_input_col_num <= '0;
            sfu_input_row_num <= '0;
            sfu_output_spm_addr <= '0;
            sfu_req_en <= 1'b0;
        end else begin
            dma_mvin_req_en <= '0;
            dma_mvout_req_en <= '0;
            sa_req_en <= 1'b0;
            matadd_req_en <= 1'b0;
            matvec_req_en <= 1'b0;
            sfu_req_en <= 1'b0;
            gemv_prep_req <= 1'b0;
            gemv_mvin_done_q <= ((engine_state == ST_GEMV_WAIT_ACT) ||
                                 (engine_state == ST_GEMV_WAIT_ACT_SCALE) ||
                                 (engine_state == ST_GEMV_WAIT_ACT_SCALE2) ||
                                 (engine_state == ST_GEMV_WAIT_ROPE_LUT) ||
                                 (engine_state == ST_GEMV_WAIT_WEIGHT_SCALE) ||
                                 (engine_state == ST_GEMV_WAIT_BLOCK0)) &&
                                dma_mvin_resp_done_any;
            gemv_wait_mvin_done_q <= (engine_state == ST_GEMV_WAIT_BLOCK) &&
                                     dma_mvin_resp_done_any;
            gemv_wait_matvec_done_q <= (engine_state == ST_GEMV_WAIT_BLOCK) &&
                                       matvec_comp_done;

            if (command_active && matvec_busy && (|dma_mvin_busy_status)) begin
                overlap_seen <= 1'b1;
            end
            if (command_active && matvec_busy && (|dma_mvout_busy_status)) begin
                mvout_overlap_seen <= 1'b1;
            end
            if (mvout_active && dma_mvout_resp_done_any) begin
                mvout_active <= 1'b0;
            end

            if (gemv_start_commit_ok) begin
                command_active <= 1'b1;
                active_api <= API_GEMV_BLOCK[2:0];
                engine_state <= gemv_start_cached_group_enable_s3 ?
                                (gemv_start_uses_weight_scale_s3 ?
                                 ST_GEMV_MVIN_WEIGHT_SCALE : ST_GEMV_MVIN_BLOCK0) :
                ST_GEMV_MVIN_ACT;
                cmd_act_dma_base <= gemv_start_act_dma_base_s3;
                cmd_act_scale_dma_base <= gemv_start_act_scale_dma_base_s3;
                cmd_act_scale2_dma_base <= gemv_start_act_scale2_dma_base_s3;
                cmd_rope_lut_dma_base <= gemv_start_rope_lut_dma_base_s3;
                cmd_weight_dma_base <= gemv_start_weight_dma_base_s3;
                cmd_weight_scale_dma_base <= gemv_start_weight_scale_dma_base_s3;
                cmd_output_dma_base <= gemv_start_output_dma_base_s3;
                cmd_mat_width <= gemv_start_mat_width_s3;
                cmd_mat_height <= gemv_start_mat_height_s3;
                cmd_gemv_mode <= gemv_start_gemv_mode_s3;
                cmd_output_precision <= gemv_start_output_precision_s3;
                cmd_col_tiles <= gemv_start_col_tiles_s3;
                cmd_act_data_bytes <= gemv_start_group_act_bytes_s3;
                cmd_act_payload_bytes <= gemv_start_payload_bytes_s3;
                cmd_act_scale_payload_bytes <= gemv_start_act_scale_payload_bytes_s3;
                cmd_act_scale2_payload_bytes <= gemv_start_act_scale2_payload_bytes_s3;
                cmd_act_group_stride <= gemv_start_act_group_stride_s3;
                cmd_mvout_rows <= gemv_start_output_rows_s3;
                cmd_row_tile_bytes <= gemv_start_row_tile_bytes_s3;
                cmd_weight_payload_bytes <= gemv_start_weight_payload_bytes_s3;
                cmd_cache_cell <= gemv_start_cache_cell_s3;
                cmd_enable_act_scale <= gemv_start_enable_act_scale_s3;
                cmd_enable_act_scale2 <= gemv_start_enable_act_scale2_s3;
                cmd_fixed_stride_enable <= gemv_start_fixed_stride_enable_s3;
                cmd_cached_group_enable <= gemv_start_cached_group_enable_s3;
                cmd_uses_weight_scale <= gemv_start_uses_weight_scale_s3;
                cmd_uses_rope_lut <= gemv_start_uses_rope_lut_s3;
                cmd_weight_row_tile_stride_bytes <= gemv_start_weight_row_tile_stride_bytes_s3;
                cmd_weight_capacity_tokens <= gemv_start_weight_capacity_tokens_s3;
                cmd_fixed_stride_active_row_tile_bytes <=
                    gemv_start_fixed_stride_active_row_tile_bytes_s3;
                cmd_weight_scale_payload_bytes <= gemv_start_weight_scale_payload_bytes_s3;
                cmd_fixed_stride_segments_issued <= 4'd0;
                cmd_fixed_stride_segment_count <=
                    gemv_start_cached_group_enable_s3 ?
                    4'(row_tiles_for_rows({16'd0, gemv_start_mat_height_s3})) :
                    gemv_group_count_segments_x2(gemv_start_decode_flow_s3);
                cmd_decode_flow <= gemv_start_decode_flow_s3;
                cmd_rows_per_block <= gemv_start_rows_per_block_s3;
                cmd_total_row_tiles <= 16'(row_tiles_for_rows({16'd0, gemv_start_mat_height_s3}));
                cmd_row_tiles_per_block <= gemv_start_rows_per_block_s3[10:5];
                cmd_row_base <= 32'd0;
                cmd_block_index <= 32'd0;
                cmd_buf_select <= 1'b0;
                cmd_group_count_m1 <= gemv_start_decode_flow_s3[26:25];
                cmd_group_index <= 2'd0;
                cmd_act_slot_select <= 1'b0;
                prefetch_active <= 1'b0;
                prefetch_done_seen <= 1'b0;
                gemv_done_seen <= 1'b0;
                mvout_active <= 1'b0;
                overlap_seen <= 1'b0;
                mvout_overlap_seen <= 1'b0;
                last_pingpong_buffer <= 1'b0;
                if (PROFILE_COUNTERS_ON) begin
                    prefetch_wait_cycles <= 32'd0;
                    mvout_wait_cycles <= 32'd0;
                    completed_block_count <= 32'd0;
                end
                gemv_prep_seed.row_tile_base <= 16'd0;
                gemv_prep_seed.block_index <= 32'd0;
                gemv_prep_seed.weight_offset <= 32'd0;
                gemv_prep_seed.buf_select <= 1'b0;
                gemv_prep_req <= 1'b0;
            end

            unique case (engine_state)
                ST_IDLE: begin
                end
                ST_GEMV_MVIN_ACT: begin
                    issue_mvin_req(cmd_act_dma_base +
                                   small_index_times_u32(
                                       {2'd0, cmd_group_index},
                                       (cmd_act_group_stride != 32'd0) ?
                                       cmd_act_group_stride : cmd_act_payload_bytes),
                                   GEMV_ACT_BASE +
                                   (cmd_cached_group_enable && cmd_act_slot_select ?
                                    cmd_act_payload_bytes : 32'd0),
                                   cmd_act_payload_bytes,
                                   2'd3, 2'd2, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 8'd0);
                    engine_state <= ST_GEMV_WAIT_ACT;
                end
                ST_GEMV_WAIT_ACT: begin
                    if (gemv_mvin_done_q) begin
                        if (cmd_enable_act_scale &&
                            (!cmd_cached_group_enable || cmd_group_index == 2'd0)) begin
                            engine_state <= ST_GEMV_MVIN_ACT_SCALE;
                        end else if (cmd_cached_group_enable) begin
                            engine_state <= cmd_uses_rope_lut ?
                                            ST_GEMV_MVIN_ROPE_LUT :
                                            ST_GEMV_START_BLOCK;
                        end else begin
                            engine_state <= cmd_uses_rope_lut ?
                                            ST_GEMV_MVIN_ROPE_LUT :
                                            (cmd_uses_weight_scale ?
                                             ST_GEMV_MVIN_WEIGHT_SCALE :
                                             ST_GEMV_MVIN_BLOCK0);
                        end
                    end
                end
                ST_GEMV_MVIN_ACT_SCALE: begin
                    issue_mvin_req(cmd_act_scale_dma_base,
                                   GEMV_ACT_BASE +
                                   (cmd_cached_group_enable ?
                                    (cmd_act_payload_bytes << 1) :
                                    cmd_act_data_bytes),
                                   cmd_act_scale_payload_bytes,
                                   2'd3, 2'd2, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 8'd0);
                    engine_state <= ST_GEMV_WAIT_ACT_SCALE;
                end
                ST_GEMV_WAIT_ACT_SCALE: begin
                    if (gemv_mvin_done_q) begin
                        if (cmd_enable_act_scale2) begin
                            engine_state <= ST_GEMV_MVIN_ACT_SCALE2;
                        end else begin
                            engine_state <= cmd_uses_rope_lut ?
                                            ST_GEMV_MVIN_ROPE_LUT :
                                            (cmd_cached_group_enable ?
                                             ST_GEMV_START_BLOCK :
                                             (cmd_uses_weight_scale ?
                                              ST_GEMV_MVIN_WEIGHT_SCALE :
                                              ST_GEMV_MVIN_BLOCK0));
                        end
                    end
                end
                ST_GEMV_MVIN_ACT_SCALE2: begin
                    issue_mvin_req(cmd_act_scale2_dma_base,
                                   GEMV_ACT_BASE +
                                   cmd_act_data_bytes +
                                   cmd_act_scale_payload_bytes,
                                   cmd_act_scale2_payload_bytes,
                                   2'd3, 2'd2, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 8'd0);
                    engine_state <= ST_GEMV_WAIT_ACT_SCALE2;
                end
                ST_GEMV_WAIT_ACT_SCALE2: begin
                    if (gemv_mvin_done_q) begin
                        engine_state <= cmd_uses_rope_lut ?
                                        ST_GEMV_MVIN_ROPE_LUT :
                                        (cmd_uses_weight_scale ?
                                         ST_GEMV_MVIN_WEIGHT_SCALE :
                                         ST_GEMV_MVIN_BLOCK0);
                    end
                end
                ST_GEMV_MVIN_ROPE_LUT: begin
                    issue_mvin_req(cmd_rope_lut_dma_base,
                                   GEMV_ROPE_LUT_BASE,
                                   GEMV_ROPE_LUT_BYTES,
                                   2'd3, 2'd1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 8'd0);
                    engine_state <= ST_GEMV_WAIT_ROPE_LUT;
                end
                ST_GEMV_WAIT_ROPE_LUT: begin
                    if (gemv_mvin_done_q) begin
                        engine_state <= cmd_cached_group_enable ?
                                        ST_GEMV_START_BLOCK :
                                        (cmd_uses_weight_scale ?
                                         ST_GEMV_MVIN_WEIGHT_SCALE :
                                         ST_GEMV_MVIN_BLOCK0);
                    end
                end
                ST_GEMV_MVIN_WEIGHT_SCALE: begin
                    issue_mvin_req(cmd_weight_scale_dma_base,
                                   32'd0,
                                   cmd_weight_scale_payload_bytes,
                                   2'd0, 2'd2, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 8'd0);
                    engine_state <= ST_GEMV_WAIT_WEIGHT_SCALE;
                end
                ST_GEMV_WAIT_WEIGHT_SCALE: begin
                    if (gemv_mvin_done_q) begin
                        engine_state <= ST_GEMV_MVIN_BLOCK0;
                    end
                end
                ST_GEMV_MVIN_BLOCK0: begin
                    if (cmd_cached_group_enable) begin
                        issue_mvin_req(cmd_weight_dma_base,
                                       GEMV_PING_BASE,
                                       cmd_fixed_stride_enable ?
                                           cmd_fixed_stride_active_row_tile_bytes :
                                           cmd_weight_payload_bytes,
                                       2'd1, 2'd1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 8'd0);
                        cmd_fixed_stride_segments_issued <= cmd_fixed_stride_enable ? 4'd1 : 4'd0;
                        engine_state <= ST_GEMV_WAIT_BLOCK0;
                    end else begin
                        issue_mvin_req(cmd_weight_dma_base,
                                       GEMV_ACT_BASE,
                                       cmd_fixed_stride_enable ?
                                           cmd_fixed_stride_active_row_tile_bytes :
                                           cmd_weight_payload_bytes,
                                       2'd1, 2'd1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 8'd0);
                        prefetch_active <= 1'b1;
                        prefetch_done_seen <= 1'b0;
                        cmd_fixed_stride_segments_issued <= cmd_fixed_stride_enable ? 4'd1 : 4'd0;
                        engine_state <= ST_GEMV_START_BLOCK;
                    end
                end
                ST_GEMV_WAIT_BLOCK0: begin
                    if (gemv_mvin_done_q) begin
                        if (cmd_cached_group_enable && cmd_fixed_stride_enable &&
                            (cmd_fixed_stride_segments_issued < cmd_fixed_stride_segment_count)) begin
                            issue_mvin_req(cmd_weight_dma_base +
                                           small_index_times_u32(
                                               cmd_fixed_stride_segments_issued,
                                               cmd_weight_row_tile_stride_bytes),
                                           GEMV_PING_BASE +
                                           small_index_times_u32(
                                               cmd_fixed_stride_segments_issued,
                                               cmd_fixed_stride_active_row_tile_bytes),
                                           cmd_fixed_stride_active_row_tile_bytes,
                                           2'd1, 2'd1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 8'd0);
                            cmd_fixed_stride_segments_issued <=
                                cmd_fixed_stride_segments_issued + 1'b1;
                        end else begin
                            engine_state <= ST_GEMV_MVIN_ACT;
                        end
                    end
                end
                ST_GEMV_WAIT_BLOCK_PREP: begin
                    if (gemv_block_valid) begin
                        engine_state <= ST_GEMV_START_BLOCK;
                    end
                end
                ST_GEMV_START_BLOCK: begin
                    gemv_active_block.row_base <= 32'd0;
                    gemv_active_block.block_index <= 32'd0;
                    gemv_active_block.current_rows <= {16'd0, cmd_mat_height};
                    gemv_active_block.next_row_base <= {16'd0, cmd_mat_height};
                    gemv_active_block.next_weight_offset <= cmd_weight_payload_bytes;
                    gemv_active_block.current_weight_bytes <= cmd_weight_payload_bytes;
                    gemv_active_block.next_weight_bytes <= 32'd0;
                    gemv_active_block.current_buf_base <= GEMV_ACT_BASE;
                    gemv_active_block.next_buf_base <= GEMV_ACT_BASE;
                    gemv_active_block.current_weight_dma_addr <= cmd_weight_dma_base;
                    gemv_active_block.next_weight_dma_addr <= cmd_weight_dma_base + cmd_weight_payload_bytes;
                    gemv_active_block.output_addr <= GEMV_OUTPUT_BASE;
                    gemv_active_block.output_dram_addr <= cmd_output_dma_base;
                    gemv_active_block.bypass_flow <= make_bypass_output_flow(cmd_mat_height);
                    gemv_active_block.has_next <= 1'b0;
                    gemv_active_block.buf_select <= 1'b0;
                    cmd_row_base <= 32'd0;
                    cmd_block_index <= 32'd0;
                    cmd_buf_select <= 1'b0;
                    issue_matvec_req(cmd_cached_group_enable ? GEMV_PING_BASE : GEMV_ACT_BASE,
                                     GEMV_ACT_BASE +
                                      (cmd_cached_group_enable && cmd_act_slot_select ?
                                       cmd_act_payload_bytes : 32'd0),
                                     cmd_mat_width,
                                     cmd_mat_height,
                                     (cmd_cached_group_enable ?
                                         gemv_group_output_addr16_for_flow(
                                             cmd_group_index, cmd_mat_height,
                                             cmd_decode_flow) :
                                         GEMV_OUTPUT_BASE[15:0]),
                                     16'd0,
                                     GEMV_ACT_BASE +
                                     (cmd_cached_group_enable ?
                                      (cmd_act_payload_bytes << 1) :
                                      cmd_act_data_bytes),
                                     GEMV_ACT_BASE +
                                     cmd_act_data_bytes +
                                     cmd_act_scale_payload_bytes,
                                     cmd_cache_cell,
                                     cmd_cached_group_enable ? 32'd0 :
                                     cmd_act_group_stride,
                                     cmd_decode_flow[31] ? 8'h98 : 8'd0,
                                     cmd_gemv_mode,
                                     (cmd_decode_flow == 64'd0) ?
                                         make_bypass_output_flow(cmd_mat_height) :
                                         (cmd_cached_group_enable ?
                                          gemv_single_group_flow(cmd_decode_flow) :
                                          cmd_decode_flow),
                                     !cmd_cached_group_enable,
                                     cmd_enable_act_scale,
                                     cmd_enable_act_scale2);
                    if (cmd_cached_group_enable &&
                        (cmd_group_index < cmd_group_count_m1)) begin
                        issue_mvin_req(cmd_act_dma_base +
                                       small_index_times_u32(
                                           {2'd0, (cmd_group_index + 2'd1)},
                                           (cmd_act_group_stride != 32'd0) ?
                                           cmd_act_group_stride : cmd_act_payload_bytes),
                                       GEMV_ACT_BASE +
                                       (cmd_act_slot_select ? 32'd0 : cmd_act_payload_bytes),
                                       cmd_act_payload_bytes,
                                       2'd3, 2'd2, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 8'd0);
                        prefetch_active <= 1'b1;
                        prefetch_done_seen <= 1'b0;
                    end else if (cmd_cached_group_enable) begin
                        prefetch_active <= 1'b0;
                        prefetch_done_seen <= 1'b1;
                    end
                    gemv_done_seen <= 1'b0;
                    gemv_wait_mvin_done_q <= 1'b0;
                    gemv_wait_matvec_done_q <= 1'b0;
                    last_pingpong_buffer <= 1'b0;
                    engine_state <= ST_GEMV_WAIT_BLOCK;
                end
                ST_GEMV_WAIT_BLOCK: begin
                    if (gemv_wait_matvec_done_q) begin
                        gemv_done_seen <= 1'b1;
                    end
                    if (cmd_cached_group_enable && prefetch_active &&
                        gemv_wait_mvin_done_q) begin
                        prefetch_done_seen <= 1'b1;
                    end else if (prefetch_active && gemv_wait_mvin_done_q &&
                        cmd_fixed_stride_enable &&
                        (cmd_fixed_stride_segments_issued < cmd_fixed_stride_segment_count)) begin
                        issue_mvin_req(cmd_weight_dma_base +
                                       (cmd_fixed_stride_segments_issued[0] ?
                                        cmd_weight_row_tile_stride_bytes : 32'd0),
                                       GEMV_ACT_BASE,
                                       cmd_fixed_stride_active_row_tile_bytes,
                                       2'd1, 2'd1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 8'd0);
                        cmd_fixed_stride_segments_issued <=
                            cmd_fixed_stride_segments_issued + 1'b1;
                    end else if (prefetch_active && gemv_wait_mvin_done_q) begin
                        prefetch_done_seen <= 1'b1;
                    end
                    if ((gemv_done_seen || gemv_wait_matvec_done_q) &&
                        prefetch_active && !(prefetch_done_seen || gemv_wait_mvin_done_q)) begin
                        if (PROFILE_COUNTERS_ON) begin
                            prefetch_wait_cycles <= prefetch_wait_cycles + 1'b1;
                        end
                    end
                    if ((gemv_done_seen || gemv_wait_matvec_done_q) &&
                        (!prefetch_active || prefetch_done_seen || gemv_wait_mvin_done_q)) begin
                        if (cmd_cached_group_enable &&
                            (cmd_group_index < cmd_group_count_m1)) begin
                            cmd_group_index <= cmd_group_index + 1'b1;
                            cmd_act_slot_select <= ~cmd_act_slot_select;
                            gemv_done_seen <= 1'b0;
                            prefetch_active <= 1'b0;
                            prefetch_done_seen <= 1'b0;
                            engine_state <= ST_GEMV_START_BLOCK;
                        end else if (mvout_active && !dma_mvout_resp_done_any) begin
                            if (PROFILE_COUNTERS_ON) begin
                                mvout_wait_cycles <= mvout_wait_cycles + 1'b1;
                            end
                        end else begin
                            issue_mvout_req(gemv_active_block.output_dram_addr,
                                            gemv_active_block.output_addr,
                                            cmd_mvout_rows,
                                            2'd0,
                                            gemv_mvout_precision(cmd_decode_flow,
                                                                 cmd_output_precision),
                                            1'b0, gemv_mvout_is_quant(cmd_decode_flow),
                                            1'b0, 8'd0);
                            mvout_active <= 1'b1;
                            if (PROFILE_COUNTERS_ON) begin
                                completed_block_count <= completed_block_count + 1'b1;
                            end
                            if (gemv_active_block.has_next) begin
                                engine_state <= ST_GEMV_WAIT_BLOCK_PREP;
                            end else begin
                                engine_state <= ST_GEMV_WAIT_MVOUT;
                            end
                        end
                    end
                end
                ST_GEMV_MVOUT: begin
                    issue_mvout_req(cmd_output_dma_base, GEMV_OUTPUT_BASE, cmd_mvout_rows,
                                    2'd0,
                                    gemv_mvout_precision(cmd_decode_flow,
                                                         cmd_output_precision),
                                    1'b0, gemv_mvout_is_quant(cmd_decode_flow),
                                    1'b0, 8'd0);
                    engine_state <= ST_GEMV_WAIT_MVOUT;
                end
                ST_GEMV_WAIT_MVOUT: begin
                    if (dma_mvout_resp_done_any || !mvout_active) begin
                        command_active <= 1'b0;
                        engine_state <= ST_IDLE;
                    end
                end
                default: begin
                    engine_state <= ST_IDLE;
                    command_active <= 1'b0;
                end
            endcase
        end
    end


   

   
endmodule

`endif
