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

`ifndef T_NPU_SV
`define T_NPU_SV

module T_NPU 
    import npu_config_pkg::*;
#(
    parameter integer DSP_PE_NUM        = (PE_FPGA_DSP ? PE_WIDTH * PE_WIDTH : 0),
    parameter integer DISABLE_IM2COL    = 0,
    parameter int AXI_MAX_BURST_BEATS   = 16,
    parameter integer DISABLE_MVIN_INT8_TO_INT32 = 0,
    parameter integer DISABLE_ACC_INT32_TO_INT8  = 0
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

    output logic                                 irq                ,
    //-------------------------------------
    // DMA AXI-4 interface
    //------------------------------------- 
	output  logic    [AXI_ID_WIDTH-1 : 0]        m_axi_awid                 ,
	output  logic    [AXI_ADDR_WIDTH-1 : 0]      m_axi_awaddr               ,
	output  logic    [7 : 0]                     m_axi_awlen                ,
	output  logic    [2 : 0]                     m_axi_awsize               ,
	output  logic    [1 : 0]                     m_axi_awburst              ,
	output  logic                                m_axi_awvalid              ,
	input   logic                                m_axi_awready              ,

	output  logic    [AXI_DATA_WIDTH-1 : 0]      m_axi_wdata                ,
	output  logic    [AXI_DATA_WIDTH/8-1 : 0]    m_axi_wstrb                ,
	output  logic                                m_axi_wlast                ,
	output  logic                                m_axi_wvalid               ,
	input   logic                                m_axi_wready               ,
   
	input   logic    [AXI_ID_WIDTH-1 : 0]        m_axi_bid                  ,
	input   logic    [1 : 0]                     m_axi_bresp                ,
	input   logic                                m_axi_bvalid               ,
	output  logic                                m_axi_bready               ,
   
	output  logic    [AXI_ID_WIDTH-1 : 0]        m_axi_arid                 ,
	output  logic    [AXI_ADDR_WIDTH-1 : 0]      m_axi_araddr               ,
	output  logic    [7 : 0]                     m_axi_arlen                ,
	output  logic    [2 : 0]                     m_axi_arsize               ,
	output  logic    [1 : 0]                     m_axi_arburst              ,
	output  logic                                m_axi_arvalid              ,
	input   logic                                m_axi_arready              ,
   
	input   logic    [AXI_ID_WIDTH-1 : 0]        m_axi_rid                  ,
	input   logic    [AXI_DATA_WIDTH-1 : 0]      m_axi_rdata                ,
	input   logic    [1 : 0]                     m_axi_rresp                ,
	input   logic                                m_axi_rlast                ,
	input   logic                                m_axi_rvalid               ,
	output  logic                                m_axi_rready            

);
//localparam SPM_ADDR_WIDTH = $clog2(SPM_SIZE);
//dma signals

logic    [RF_DATA_WIDTH/2-1:0]           mvin_dram_addr               ;  // DDR physical address for DMA request
logic    [RF_DATA_WIDTH/2-1:0]           mvin_row_num                 ;  // number of rows for DMA request
logic    [RF_DATA_WIDTH/2-1:0]           mvin_sram_addr               ;  // ScratchPad physical address for DMA request
logic    [RF_DATA_WIDTH/2-1:0]           mvin_col_num                 ;   //number of column for DMA request
logic    [RF_DATA_WIDTH/2-1:0]           mvout_dram_addr              ;  // DDR physical address for DMA request
logic    [RF_DATA_WIDTH/2-1:0]           mvout_row_num                ;  // number of rows for DMA request
logic    [RF_DATA_WIDTH/2-1:0]           mvout_sram_addr              ;  // ScratchPad physical address for DMA request
logic    [RF_DATA_WIDTH/2-1:0]           mvout_col_num                ;   //number of column for DMA request
logic    [1:0]                           cfg_mvin_input_type          ;   // data type of the mvin data, 00 for input feature/A, 01 for weight/B
                                                                          // 10 for bias
logic    [1:0]                           cfg_mvout_output_type        ;
logic    [1:0]                           cfg_mvin_input_precision     ;   //data precision,00/01/10/11 for int4/int8/fp16/fp32
                                                                         // for bias :int16/int32/fp16/fp32
logic    [1:0]                           cfg_mvout_output_precision   ;
logic                                    cfg_mvin_is_quant            ;    //mvin/mvout is/not quant
logic                                    cfg_mvout_is_quant           ;
logic                                    cfg_mvin_dest                ;    //mvin data destination ,0 for SPM ,1 for ACC
logic                                    cfg_mvout_source             ;    //mvout data source,0 for SPM ,1 for ACC
logic                                    cfg_mvout_per_channel        ;
logic                                    cfg_mvin_isbias              ;   // marks ACC MVIN payload as bias data
logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvin_sram_stride         ;   // Stride for mvin SPM address increment
logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvin_dram_stride         ;   // Stride for mvin DRAM address increment
logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvout_sram_stride        ;   // Stride for mvout SPM address increment
logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvout_dram_stride        ;   // Stride for mvout DRAM address increment
logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvin_input_zeropoint     ;   // for matrix add,the input quant zero point
logic    [RF_DATA_WIDTH/2-1:0]           cfg_mvout_output_zeropoint   ;   // for matrix add,the output quant zero point
logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvin_input_scale         ;   // for matrix add,the input quant scale
logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvout_output_scale       ;   // for matrix add,the output quant scale
logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvin_input_scale_shift   ;   // for matrix add,the input quant scale shift
logic    [RF_DATA_WIDTH/4-1:0]           cfg_mvout_output_scale_shift ;   // for matrix add,the output quant scale shift
logic                                    dma_mvin_req_en              ;   // enable signal for mvin instructions, generate AXI read req when high
logic                                    dma_mvout_req_en             ;   // enable signal for mvout instructions, generate AXI write req when high
logic                                    dma_mvin_busy                ;
logic                                    dma_mvout_busy               ;
logic                                    dma_mvin_resp_done           ;
logic                                    dma_mvout_resp_done          ;
//dma spm singals
logic   [SPM_DATA_WIDTH-1:0]             dma_spm_din          ;  
logic                                    dma_spm_wr_en        ;
logic   [$clog2(SPM_SIZE)-1:0]           dma_spm_wr_addr      ;
logic   [$clog2(SPM_SIZE)-1:0]           dma_spm_rd_addr      ;
logic                                    dma_spm_rd_en        ;
logic   [PE_WIDTH-1:0]                   dma_spm_wr_mask      ;
logic   [SPM_DATA_WIDTH-1:0]             dma_spm_dout         ;
//dma acc signals
logic   [ACC_DATA_WIDTH-1:0]             dma_acc_din          ;  
logic                                    dma_acc_wr_en        ;
logic   [$clog2(ACC_SIZE)-1:0]           dma_acc_wr_addr      ;
logic   [$clog2(ACC_SIZE)-1:0]           dma_acc_rd_addr      ;
logic                                    dma_acc_rd_en        ;
logic   [ACC_DATA_WIDTH/32-1:0]          dma_acc_wr_mask      ;
logic   [ACC_DATA_WIDTH-1:0]             dma_acc_dout         ;
logic                                    dma_acc_rd_valid     ;
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
logic    [$clog2(INPUT_WIDTH_MAX)-1:0]   sa_input_a_col_num          ;   // column number of input feature(WS)/input matrix A(OS) in systolic array
logic    [$clog2(ARRAY_HEIGHT)-1:0]      sa_input_a_row_num          ;   // row number of input feature(WS)/input matrix A(OS) in systolic array
logic    [RF_DATA_WIDTH/4-1:0]           sa_input_a_stride           ;   //stride for input feature(WS)/input matrix A(OS) data, for GEMM not valid(default W)
                                                                         //for dilated conv(not support now),default is W-1
logic    [RF_DATA_WIDTH/2-1:0]           sa_input_b_spm_addr         ;   // address for input matrix B(OS) data in SPM
logic    [$clog2(ARRAY_WIDTH)-1:0]       sa_input_b_col_num          ;   // column number of input matrix B(OS) in systolic array
logic    [$clog2(INPUT_HEIGHT_MAX)-1:0]  sa_input_b_row_num          ;   // row number of input matrix B(OS) in systolic array
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
logic                                   matvec_weight_stream_mode    ;
logic                                   matvec_act_scale_enable      ;
logic                                   matvec_act_scale2_enable     ;
logic                                   matvec_req_en                ;
logic                                   matvec_busy                  ;
logic                                   matvec_comp_done             ;
//spm signals
logic               [WR_PORTS-1:0]                       wr_en   ;
logic [PE_WIDTH-1:0][WR_PORTS-1:0]                       wr_mask ;
logic               [WR_PORTS-1:0][SPM_ADDR_WIDTH-1:0]   wr_addr ;
logic [PE_WIDTH-1:0][WR_PORTS-1:0][PE_DATA_WIDTH-1:0]    din     ;
logic               [RD_PORTS-1:0]                       rd_en   ;
logic               [RD_PORTS-1:0][SPM_ADDR_WIDTH-1:0]   rd_addr ;
logic [PE_WIDTH-1:0][RD_PORTS-1:0][PE_DATA_WIDTH-1:0]    dout    ;

logic [WR_PORTS-1:0][PE_WIDTH-1:0]                       port_wr_mask ;
logic [WR_PORTS-1:0][PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]    port_din     ;
logic [RD_PORTS-1:0][PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]    port_dout    ;
always_comb begin 
    sa_busy=systolic_array_busy || accadd_busy;
    matvec_busy = '0;
    matvec_comp_done = '0;
end
always_comb begin : port_switch
    for (int idx = 0; idx < PE_WIDTH; idx++)begin
        for (int p = 0; p < WR_PORTS; p++)begin
            wr_mask[idx][p]  = port_wr_mask[p][idx];
            din [idx][p]     = port_din[p][idx] ;
        end
        for (int p = 0; p < RD_PORTS; p++)begin
            port_dout[p][idx]     = dout [idx][p];
        end
    end
end
always_comb begin : spm_control    
    rd_en        = '0;
    rd_addr      = '0;
    dma_spm_dout = '0; 
    sfu_spm_rd_data_in = '0;
    sa_spm_rd1_data_in = '0;
    sa_spm_rd2_data_in = '0;
    wr_en        = '0;
    port_wr_mask = '0;
    port_din     = '0;
    wr_addr      = '0;
    case({dma_mvin_busy|dma_mvout_busy,sfu_busy,sa_busy,matadd_busy})  //3 bit signal
    4'b1000:begin
        rd_en  [0]       = dma_spm_rd_en;
        rd_addr[0]       = dma_spm_rd_addr;
        dma_spm_dout     = port_dout[0]; 
        wr_en [0]        = dma_spm_wr_en;
        port_wr_mask [0] = dma_spm_wr_mask;
        port_din[0]      = dma_spm_din;
        wr_addr[0]       = dma_spm_wr_addr;
    end
    4'b0100:begin
       rd_en  [1]          = sfu_spm_rd_en;
       rd_addr[1]          = sfu_spm_rd_addr;
       sfu_spm_rd_data_in  = port_dout[1]; 
       wr_en [0]           = sfu_spm_wr_en;
       port_wr_mask [0]    = sfu_spm_wr_mask;
       port_din[0]         = sfu_spm_wr_data_out;
       wr_addr[0]          = sfu_spm_wr_addr; 
    end
    4'b0010:begin
       rd_en  [0]          = sa_spm_rd1_en;
       rd_addr[0]          = sa_spm_rd1_addr;
       sa_spm_rd1_data_in  = port_dout[0]; 
       rd_en  [1]          = sa_spm_rd2_en;
       rd_addr[1]          = sa_spm_rd2_addr;
       sa_spm_rd2_data_in  = port_dout[1]; 
       wr_en [0]           = accu_spm_wr_en;
       port_wr_mask [0]    = accu_spm_wr_mask;
       port_din[0]         = accu_spm_wr_data;
       wr_addr[0]          = accu_spm_wr_addr;
    end
    4'b0001:begin
       wr_en [0]           = accu_spm_wr_en;
       port_wr_mask [0]    = accu_spm_wr_mask;
       port_din[0]         = accu_spm_wr_data;
       wr_addr[0]          = accu_spm_wr_addr;
    end
    default:begin
       rd_en        = '0;
       rd_addr      = '0;
       dma_spm_dout = '0; 
       sfu_spm_rd_data_in = '0;
       sa_spm_rd1_data_in = '0;
       sa_spm_rd2_data_in = '0;
       wr_en        = '0;
       port_wr_mask = '0;
       port_din     = '0;
       wr_addr      = '0;
    end
    endcase
end
inst_ctrl inst_ctrl_u0 (
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
        .mvin_dram_addr                 (mvin_dram_addr     ),        
        .mvin_row_num                   (mvin_row_num       ),          
        .mvin_sram_addr                 (mvin_sram_addr     ),        
        .mvin_col_num                   (mvin_col_num       ),             
        .mvout_dram_addr                (mvout_dram_addr    ),       
        .mvout_row_num                  (mvout_row_num      ),         
        .mvout_sram_addr                (mvout_sram_addr    ),       
        .mvout_col_num                  (mvout_col_num      ),         
        .cfg_mvin_input_type            (cfg_mvin_input_type        ), 
        .cfg_mvout_output_type          (cfg_mvout_output_type      ),          
        .cfg_mvin_input_precision       (cfg_mvin_input_precision   ),      
        .cfg_mvout_output_precision     (cfg_mvout_output_precision ),    
        .cfg_mvin_is_quant              (cfg_mvin_is_quant          ),             
        .cfg_mvout_is_quant             (cfg_mvout_is_quant         ),            
        .cfg_mvin_dest                  (cfg_mvin_dest              ),                 
        .cfg_mvout_source               (cfg_mvout_source           ),             
        .cfg_mvout_per_channel          (cfg_mvout_per_channel      ),
        .cfg_mvin_isbias                (cfg_mvin_isbias            ),
        .cfg_mvin_sram_stride           (cfg_mvin_sram_stride       ),          
        .cfg_mvin_dram_stride           (cfg_mvin_dram_stride       ),          
        .cfg_mvout_sram_stride          (cfg_mvout_sram_stride      ),         
        .cfg_mvout_dram_stride          (cfg_mvout_dram_stride      ),         
        .cfg_mvin_input_zeropoint       (cfg_mvin_input_zeropoint   ),      
        .cfg_mvout_output_zeropoint     (cfg_mvout_output_zeropoint ),    
        .cfg_mvin_input_scale           (cfg_mvin_input_scale       ),          
        .cfg_mvout_output_scale         (cfg_mvout_output_scale     ),        
        .cfg_mvin_input_scale_shift     (cfg_mvin_input_scale_shift ),    
        .cfg_mvout_output_scale_shift   (cfg_mvout_output_scale_shift),  
        .dma_mvin_req_en                (dma_mvin_req_en            ),       
        .dma_mvout_req_en               (dma_mvout_req_en           ),      
        .dma_mvin_busy                  (dma_mvin_busy              ),         
        .dma_mvout_busy                 (dma_mvout_busy             ),        
        .dma_mvin_resp_done             (dma_mvin_resp_done         ),    
        .dma_mvout_resp_done            (dma_mvout_resp_done        ),   
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
        .matvec_weight_stream_mode      (matvec_weight_stream_mode  ),
        .matvec_act_scale_enable        (matvec_act_scale_enable    ),
        .matvec_act_scale2_enable       (matvec_act_scale2_enable   ),
        .matvec_req_en                  (matvec_req_en              ),        
        .matvec_busy                    (matvec_busy                ),          
        .matvec_comp_done               (matvec_comp_done           ),     
        .kv_scale_commit_valid          (1'b0                       ),
        .kv_scale_commit_is_v           (1'b0                       ),
        .kv_scale_commit_values         ('0                         ),
        .kv_scale_commit_count          ('0                         ),
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

`ifdef VCS_SRAM_ENABLE
scratchpad #(
        .PE_WIDTH       ( PE_WIDTH      ),
        .PE_DATA_WIDTH  ( PE_DATA_WIDTH ),
        .SPM_SIZE       ( SPM_SIZE      ),
        .RD_PORTS       ( 2             ),
        .WR_PORTS       ( 1             )
    )u_scratchpad(
        .clk            ( clk           ),
        .rst_n          ( rst_n         ),
        .wr_en          ( wr_en         ),
        .wr_mask        ( wr_mask       ),
        .wr_addr        ( wr_addr       ),
        .din            ( din           ),
        .rd_en          ( rd_en         ),
        .rd_addr        ( rd_addr       ),
        .dout           ( dout          )
);
`else 
scratchpad_dp_sram #(
        .PE_WIDTH       ( PE_WIDTH      ),
        .PE_DATA_WIDTH  ( PE_DATA_WIDTH ),
        .SPM_SIZE       ( SPM_SIZE      ),
        .RD_PORTS       ( 2             ),
        .WR_PORTS       ( 1             )
    )u_scratchpad(
        .clk            ( clk           ),
        .rst_n          ( rst_n         ),
        .wr_en          ( wr_en         ),
        .wr_mask        ( wr_mask       ),
        .wr_addr        ( wr_addr       ),
        .din            ( din           ),
        .rd_en          ( rd_en         ),
        .rd_addr        ( rd_addr       ),
        .dout           ( dout          )
);
`endif
DMA #(
        .AXI_MAX_BURST_BEATS      ( AXI_MAX_BURST_BEATS ),
        .DISABLE_MVIN_INT8_TO_INT32 ( DISABLE_MVIN_INT8_TO_INT32 )
) u_dma(
        .clk                    ( clk                   ),
        .rst_n                  ( rst_n                 ),

        .mvin_dram_addr              ( mvin_dram_addr            ),          
        .mvin_sram_addr              ( mvin_sram_addr            ),          
        .mvin_col_num                ( mvin_col_num              ),            
        .mvin_row_num                ( mvin_row_num              ),            
        .mvout_dram_addr             ( mvout_dram_addr           ),         
        .mvout_sram_addr             ( mvout_sram_addr           ),         
        .mvout_col_num               ( mvout_col_num             ),           
        .mvout_row_num               ( mvout_row_num             ),           
        .cfg_mvin_input_type         ( cfg_mvin_input_type       ),  
        .cfg_mvout_output_type       ( cfg_mvout_output_type     ),     
        .cfg_mvin_input_precision    ( cfg_mvin_input_precision  ),  
        .cfg_mvout_output_precision  ( cfg_mvout_output_precision),
        .cfg_mvin_is_quant           ( cfg_mvin_is_quant         ),         
        .cfg_mvout_is_quant          ( cfg_mvout_is_quant        ),        
        .cfg_mvin_dest               ( cfg_mvin_dest             ),          
        .cfg_mvout_source            ( cfg_mvout_source          ),       
        .cfg_mvin_sram_stride        ( cfg_mvin_sram_stride      ),    
        .cfg_mvin_dram_stride        ( cfg_mvin_dram_stride      ),    
        .cfg_mvout_sram_stride       ( cfg_mvout_sram_stride     ),   
        .cfg_mvout_dram_stride       ( cfg_mvout_dram_stride     ),   
        .cfg_mvin_input_zeropoint    ( cfg_mvin_input_zeropoint  ),      
        .cfg_mvout_output_zeropoint  ( cfg_mvout_output_zeropoint),    
        .cfg_mvin_input_scale        ( cfg_mvin_input_scale      ),          
        .cfg_mvout_output_scale      ( cfg_mvout_output_scale    ),        
        .cfg_mvin_input_scale_shift  ( cfg_mvin_input_scale_shift),    
        .cfg_mvout_output_scale_shift( cfg_mvout_output_scale_shift), 
        .dma_mvin_req_en             ( dma_mvin_req_en           ),        
        .dma_mvout_req_en            ( dma_mvout_req_en          ),       
        .dma_mvin_resp_done          ( dma_mvin_resp_done        ),     
        .dma_mvout_resp_done         ( dma_mvout_resp_done       ),   
        .dma_mvin_busy               ( dma_mvin_busy             ),        
        .dma_mvout_busy              ( dma_mvout_busy            ), 

        .spm_din                ( dma_spm_din           ),  
        .spm_wr_en              ( dma_spm_wr_en         ),  
        .spm_wr_addr            ( dma_spm_wr_addr       ),  
        .spm_rd_addr            ( dma_spm_rd_addr       ),  
        .spm_rd_en              ( dma_spm_rd_en         ),  
        .spm_wr_mask            ( dma_spm_wr_mask       ),  
        .spm_dout               ( dma_spm_dout          ),  

        .acc_din                ( dma_acc_din           ),            
        .acc_wr_en              ( dma_acc_wr_en         ),            
        .acc_wr_addr            ( dma_acc_wr_addr       ),            
        .acc_rd_addr            ( dma_acc_rd_addr       ),            
        .acc_rd_en              ( dma_acc_rd_en         ),            
        .acc_wr_mask            ( dma_acc_wr_mask       ),            
        .acc_dout               ( dma_acc_dout          ),            
        .acc_dout_valid         ( dma_acc_rd_valid      ),

        .m_axi_awid             ( m_axi_awid            ), 
        .m_axi_awaddr           ( m_axi_awaddr          ), 
        .m_axi_awlen            ( m_axi_awlen           ), 
        .m_axi_awsize           ( m_axi_awsize          ), 
        .m_axi_awburst          ( m_axi_awburst         ), 
        .m_axi_awvalid          ( m_axi_awvalid         ), 
        .m_axi_awready          ( m_axi_awready         ), 
        .m_axi_wdata            ( m_axi_wdata           ), 
        .m_axi_wstrb            ( m_axi_wstrb           ), 
        .m_axi_wlast            ( m_axi_wlast           ), 
        .m_axi_wvalid           ( m_axi_wvalid          ), 
        .m_axi_wready           ( m_axi_wready          ), 
        .m_axi_bid              ( m_axi_bid             ), 
        .m_axi_bresp            ( m_axi_bresp           ), 
        .m_axi_bvalid           ( m_axi_bvalid          ), 
        .m_axi_bready           ( m_axi_bready          ), 
        .m_axi_arid             ( m_axi_arid            ), 
        .m_axi_araddr           ( m_axi_araddr          ), 
        .m_axi_arlen            ( m_axi_arlen           ), 
        .m_axi_arsize           ( m_axi_arsize          ), 
        .m_axi_arburst          ( m_axi_arburst         ), 
        .m_axi_arvalid          ( m_axi_arvalid         ), 
        .m_axi_arready          ( m_axi_arready         ), 
        .m_axi_rid              ( m_axi_rid             ), 
        .m_axi_rdata            ( m_axi_rdata           ), 
        .m_axi_rresp            ( m_axi_rresp           ), 
        .m_axi_rlast            ( m_axi_rlast           ), 
        .m_axi_rvalid           ( m_axi_rvalid          ), 
        .m_axi_rready           ( m_axi_rready          )
);

systolic_array_top #(
        .DSP_PE_NUM              ( DSP_PE_NUM ),
        .DISABLE_IM2COL          ( DISABLE_IM2COL )
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

accumulator # (
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
        .cfg_accu_isaccu          ( cfg_accu_isaccu              ),
        .cfg_accu_relu            ( cfg_accu_relu                ),
        .cfg_accu_relu_type       ( cfg_accu_relu_type           ),
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
        .dma_mvin_resp_done       ( dma_mvin_resp_done           ),
        .dma_mvout_resp_done      ( dma_mvout_resp_done          ),
        .dma_accu_rd_en           ( dma_acc_rd_en                ),
        .dma_accu_rd_addr         ( dma_acc_rd_addr              ),
        .dma_accu_rd_data         ( dma_acc_dout                 ),
        .dma_accu_rd_valid        ( dma_acc_rd_valid             ),
        .cfg_mvout_output_precision(cfg_mvout_output_precision   ),
        .cfg_mvout_per_channel     (cfg_mvout_per_channel        ),
        .mvout_col_num             (mvout_col_num                ),
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

   sfu  u_sfu (
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








endmodule

`endif 
