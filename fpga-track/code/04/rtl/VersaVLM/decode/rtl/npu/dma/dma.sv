//////////////////////////////////////////////////////////////////
//////////////// 
// Copyright by FuxionLab 
//  
// Designer     : Lanqi Ma 
// Create Date  : 2024/11/08 
// Project Name : NPU 
// File Name    : dma.sv 
// 
// Description  : dma code  
// 
// Revision:  
// Additional Comments:  
// 
//////////////////////////////////////////////////////////////////
////////////////
`ifndef DMA_V
`define DMA_V
module DMA 
import npu_config_pkg::*;
#(
    //fit for:SPM_DATA_WIDTH=AXI_DATA_WIDTH
    parameter int RF_DATA_WIDTH   = npu_config_pkg::RF_DATA_WIDTH,
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

    input   logic    [RF_DATA_WIDTH/2-1:0]      mvin_dram_addr              ,   // DDR physical address for DMA request
    input   logic    [RF_DATA_WIDTH/2-1:0]      mvin_sram_addr              ,   // ScratchPad physical address for DMA request
    input   logic    [RF_DATA_WIDTH/2-1:0]      mvin_col_num                ,   // Burst length for DMA request bytes   ,real = len+1
    input   logic    [RF_DATA_WIDTH/2-1:0]      mvin_row_num                ,   // number of rows for DMA request     ,real = row_num+1
    input   logic    [RF_DATA_WIDTH/2-1:0]      mvout_dram_addr             ,   // DDR physical address for DMA request
    input   logic    [RF_DATA_WIDTH/2-1:0]      mvout_sram_addr             ,   // ScratchPad physical address for DMA request
    input   logic    [RF_DATA_WIDTH/2-1:0]      mvout_col_num               ,   // Burst length for DMA request bytes   ,real = len+1
    input   logic    [RF_DATA_WIDTH/2-1:0]      mvout_row_num               ,   // number of rows for DMA request     ,real = row_num+1

    input   logic    [1:0]                      cfg_mvin_input_type         ,   // data type of the mvin data, 00 for input feature/A, 01 for weight/B
                                                                                    // 10 for bias
    input   logic    [1:0]                      cfg_mvout_output_type       ,   // data type of the mvout data, 00 for final output, 01 for part sum
    input   logic    [1:0]                      cfg_mvin_input_precision    ,   //data precision,00/01/10/11 for int4/int8/fp16/fp32
                                                                                     // for bias :int16/int32/fp16/fp32
    input   logic    [1:0]                      cfg_mvout_output_precision  ,
    input   logic                               cfg_mvin_is_quant           ,    //mvin/mvout is/not quant
    input   logic                               cfg_mvout_is_quant          ,
    input   logic                               cfg_mvin_dest               ,    //mvin data destination ,0 for SPM ,1 for ACC
    input   logic                               cfg_mvout_source            ,    //mvout data source,0 for SPM ,1 for ACC
    input   logic    [RF_DATA_WIDTH/4-1:0]      cfg_mvin_sram_stride        ,   // Stride for mvin SPM/ACC address increment
    input   logic    [RF_DATA_WIDTH/2-1:0]      cfg_mvin_dram_stride        ,   // Stride for mvin DRAM address increment
    input   logic    [RF_DATA_WIDTH/4-1:0]      cfg_mvout_sram_stride       ,   // Stride for mvout SPM/ACC address increment
    input   logic    [RF_DATA_WIDTH/2-1:0]      cfg_mvout_dram_stride       ,   // Stride for mvout DRAM address increment
    input   logic    [RF_DATA_WIDTH/2-1:0]      cfg_mvin_input_zeropoint    ,   // for matrix add,the input quant zero point
    input   logic    [RF_DATA_WIDTH/2-1:0]      cfg_mvout_output_zeropoint  ,   // for matrix add,the output quant zero point
    input   logic    [RF_DATA_WIDTH/4-1:0]      cfg_mvin_input_scale        ,   // for matrix add,the input quant scale
    input   logic    [RF_DATA_WIDTH/4-1:0]      cfg_mvout_output_scale      ,   // for matrix add,the output quant scale
    input   logic    [RF_DATA_WIDTH/4-1:0]      cfg_mvin_input_scale_shift  ,   // for matrix add,the input quant scale shift
    input   logic    [RF_DATA_WIDTH/4-1:0]      cfg_mvout_output_scale_shift,   // for matrix add,the output quant scale shift

    input   logic                               dma_mvin_req_en             ,   // enable signal for mvin instructions, generate AXI read req when high
    input   logic                               dma_mvout_req_en            ,   // enable signal for mvout instructions, generate AXI write req when high
    output  logic                               dma_mvin_resp_done          ,   // done signal for mvin instructions 
    output  logic                               dma_mvout_resp_done         ,
    output  logic                               dma_mvin_busy               ,
    output  logic                               dma_mvout_busy              ,
    
    //------------------------------------- 
    // SPM Control Signals  
    //------------------------------------- 

   
    output  logic    [SPM_DATA_WIDTH-1:0]      spm_din                      ,  
    output  logic                              spm_wr_en                    ,
    output  logic    [$clog2(SPM_SIZE)-1:0]    spm_wr_addr                  ,
    output  logic    [$clog2(SPM_SIZE)-1:0]    spm_rd_addr                  ,
    output  logic                              spm_rd_en                    ,
    output  logic    [SPM_DATA_WIDTH/8-1:0]    spm_wr_mask                  ,
    input   logic    [SPM_DATA_WIDTH-1:0]      spm_dout                     ,

    //------------------------------------- 
    // ACC Control Signals  
    //------------------------------------- 

   
    output  logic    [ACC_DATA_WIDTH-1:0]      acc_din                      ,  
    output  logic                              acc_wr_en                    ,
    output  logic    [$clog2(ACC_SIZE)-1:0]    acc_wr_addr                  ,
    output  logic    [$clog2(ACC_SIZE)-1:0]    acc_rd_addr                  ,
    output  logic                              acc_rd_en                    ,
    output  logic    [ACC_DATA_WIDTH/32-1:0]   acc_wr_mask                  ,
    input   logic    [ACC_DATA_WIDTH-1:0]      acc_dout                     ,
    input   logic                              acc_dout_valid               ,
   
    //-------------------------------------    
    // AXI Control Signals     
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
localparam      IDLE                 = 2'b00                                     ;   //FSM state IDLE
localparam      S_READ               = 2'b01                                     ;   //FSM state S_READ
localparam      S_WRITE              = 2'b01                                     ;   //FSM state S_WRITE
localparam      S_WAIT               = 2'b11                                     ;   //FSM state S_WAIT
localparam      AXI_AxSIZE           = $clog2(AXI_DATA_WIDTH/8)                  ;   //burst size
localparam      BURST_ADDR_INCR      = AXI_DATA_WIDTH/8                          ;   //each transfer,the axi addr self incr
localparam      SPM_ADDR_INCR        = SPM_DATA_WIDTH/8                          ;   //each transfer,the spm addr self incr
localparam      BURST_ALIGN_BITWID   = $clog2(AXI_DATA_WIDTH/8)                  ;   //the bitwidth of unaligned transfer process need  
localparam      DATA_SIZE            = 32                                        ;   //the max data width 
localparam      DATA_IDX             = $clog2(DATA_SIZE/8)                       ;   //the bytewidth of DATASIZE
localparam      SRAM_ADDR_WIDTH      = SPM_ADDR_WIDTH>ACC_ADDR_WIDTH?SPM_ADDR_WIDTH:ACC_ADDR_WIDTH; //the max addr width 
localparam      DMA_DIM_WIDTH        = RF_DATA_WIDTH/2                           ;   // row/col bitwidth from control register
localparam      BEATS_WIDTH          = DMA_DIM_WIDTH+DATA_IDX-BURST_ALIGN_BITWID+1;   //the width of beats in one row
localparam      REAL_NUM_WIDTH       = DMA_DIM_WIDTH+DATA_IDX+1;
localparam      REAL_STRIDE_WIDTH    = RF_DATA_WIDTH/2+DATA_IDX;
localparam int  RD_META_DEPTH        = AXI_RD_OUTSTANDING;
localparam int  RD_META_PTR_WIDTH    = (AXI_RD_OUTSTANDING <= 1) ? 1 : $clog2(AXI_RD_OUTSTANDING);

logic    [1:0]                               mvin_dram_data_size_log2        ;    //data size of dram,clog2(dram-data-width/8)
logic    [1:0]                               mvin_sram_data_size_log2        ;    //data size of spm/acc,clog2(spm/acc-data-width/8)
logic    [1:0]                               mvout_dram_data_size_log2       ;    //data size of dram,clog2(dram-data-width/8)
logic    [1:0]                               mvout_sram_data_size_log2       ;    //data size of spm/acc,clog2(spm/acc-data-width/8)
logic                                        sram_wr_en                      ;
logic    [SRAM_ADDR_WIDTH-1:0]               sram_wr_addr                    ;
logic    [SRAM_ADDR_WIDTH-1:0]               sram_wr_addr_delay              ;
logic    [SRAM_ADDR_WIDTH-1:0]               sram_rd_addr                    ;
logic                                        sram_rd_en                      ;
logic                                        sram_wr_en_delay                ;
logic    [AXI_DATA_WIDTH/(8)-1:0]            sram_wr_mask                    ;
logic    [AXI_DATA_WIDTH/(8)-1:0]            sram_wr_mask_delay              ;

logic    [REAL_NUM_WIDTH-1:0]                reg_read_row_num                ; 
logic    [REAL_NUM_WIDTH-1:0]                reg_read_col_num                ;
logic    [REAL_NUM_WIDTH-1:0]                reg_write_col_num               ;
logic    [REAL_NUM_WIDTH-1:0]                reg_write_row_num               ;
logic    [REAL_STRIDE_WIDTH-1:0]             reg_cfg_mvin_sram_stride        ;
logic    [REAL_STRIDE_WIDTH-1:0]             reg_cfg_mvin_dram_stride        ;
logic    [REAL_STRIDE_WIDTH-1:0]             reg_cfg_mvout_sram_stride       ;
logic    [REAL_STRIDE_WIDTH-1:0]             reg_cfg_mvout_dram_stride       ;      
logic    [1:0]                               reg_dma_mvin_req_en             ;   
logic    [2:0]                               reg_dma_mvout_req_en            ;  
logic    [REAL_NUM_WIDTH-1:0]                mvin_rows_real                  ;
logic    [REAL_NUM_WIDTH-1:0]                mvin_cols_real                  ;
logic    [REAL_NUM_WIDTH-1:0]                mvout_rows_real                 ;
logic    [REAL_NUM_WIDTH-1:0]                mvout_cols_real                 ;
logic                                        mvin_contiguous_2d              ;
logic                                        mvout_contiguous_2d             ;

//the logic of some axi signal

logic    [SPM_DATA_WIDTH-1:0]                fore_sram_dout                  ;   //the forward data for spm read 
logic                                        fore_flag                       ;   // use  fore data flag 
logic    [AXI_DATA_WIDTH/8-1:0]              sram_wr_mask_bias               ;    

logic    [ACC_DATA_WIDTH-1:0]                dequant_data                    ;  
logic    [SPM_DATA_WIDTH-1:0]                sram_din_pre                    ;
logic    [SPM_DATA_WIDTH-1:0]                sram_din                        ;
logic    [AXI_DATA_WIDTH-1:0]                quant_data                      ;  
logic    [SPM_DATA_WIDTH-1 : 0]              axi_wdata                       ;
logic                                        cfg_mvin_is_quant_effective     ;
logic                                        mvout_w_fire                    ;
logic                                        mvout_wlast_fire                ;

logic    [1 : 0]                             current_read_state              ;  //read channel state control
logic    [1 : 0]                             next_read_state                 ;
logic    [1 : 0]                             current_write_state             ;
logic    [1 : 0]                             next_write_state                ;
logic    [1 : 0]                             current_raddr_state             ;  //read address state control
logic    [1 : 0]                             next_raddr_state                ;
logic    [1 : 0]                             current_waddr_state             ;
logic    [1 : 0]                             next_waddr_state                ;

logic    [12:0]                              aw_bytes_to_4k                  ; //the bytes number to 4K boundary
logic    [12-BURST_ALIGN_BITWID:0]           aw_beats_to_4k                  ; //the beats number to 4K boundary, one beat represent one transfer,
logic    [BEATS_WIDTH-1:0]                   aw_remain_beats                 ; //the beats number remain to transfer
logic    [8:0]                               aw_remain_beats_clamped         ; //clamped by AXI_MAX_BURST_BEATS
logic    [8:0]                               aw_final_beats                  ; //chose beats from beats-to-4k and remain-beats-clamped
logic    [12:0]                              w_bytes_to_4k                   ; //for w channel, cal the split logic again
logic    [12-BURST_ALIGN_BITWID:0]           w_beats_to_4k                   ;
logic    [BEATS_WIDTH-1:0]                   w_remain_beats                  ; //change while wlast is high,also is the end of one burst
logic    [BEATS_WIDTH-1:0]                   w_total_beats                   ;
logic    [BEATS_WIDTH-1:0]                   w_total_beats_pre               ; //cal the next total beats for sram pre fetch data 
logic    [BEATS_WIDTH-1:0]                   w_remain_beats_runtime          ; //runtime remain beats,reduce by w handshake
logic    [8:0]                               w_remain_beats_clamped          ;
logic    [8:0]                               w_final_beats                   ;
logic    [AXI_ADDR_WIDTH-1:0]                w_shadow_addr                   ;
logic    [$clog2(AXI_WR_OUTSTANDING):0]      w_outstanding_cnt               ; //cnt for aw handshake and b handshake
logic    [DMA_DIM_WIDTH:0]                   wdata_row_cnt                   ;  //counter for row change
logic    [$clog2(AXI_RD_OUTSTANDING):0]      r_outstanding_cnt               ; //cnt for ar handshake and rlast handshake

logic    [12:0]                              ar_bytes_to_4k                  ; //the bytes number to 4K boundary
logic    [12-BURST_ALIGN_BITWID:0]           ar_beats_to_4k                  ; //the beats number to 4K boundary, one beat represent one transfer,
logic    [BEATS_WIDTH-1:0]                   ar_remain_beats                 ; //the beats number remain to transfer
logic    [BEATS_WIDTH-1:0]                   ar_total_beats                  ;
logic    [8:0]                               ar_remain_beats_clamped         ; //clamped by AXI_MAX_BURST_BEATS
logic    [8:0]                               ar_final_beats                  ; //chose beats from beats-to-4k and remain-beats-clamped
logic    [BEATS_WIDTH-1:0]                   r_remain_beats                  ; //change when r handshake
logic    [BEATS_WIDTH-1:0]                   r_total_beats                   ;
logic    [RD_META_PTR_WIDTH-1:0]             rd_meta_wr_ptr                  ;
logic    [RD_META_PTR_WIDTH-1:0]             rd_meta_rd_ptr                  ;
logic    [$clog2(AXI_RD_OUTSTANDING+1)-1:0]  rd_meta_count                   ;
logic    [RD_META_DEPTH-1:0]                 rd_meta_row_first               ;
logic    [RD_META_DEPTH-1:0]                 rd_meta_row_last                ;
logic    [RD_META_DEPTH-1:0]                 rd_meta_row_single              ;
logic    [RD_META_DEPTH-1:0][8:0]            rd_meta_burst_beats             ;
logic    [RD_META_DEPTH-1:0][BURST_ALIGN_BITWID-1:0] rd_meta_head_offset      ;
logic    [RD_META_DEPTH-1:0][BURST_ALIGN_BITWID-1:0] rd_meta_tail_bytes       ;
logic    [8:0]                               rd_burst_beat_cnt               ;
logic                                        rd_meta_push                    ;
logic                                        rd_meta_pop                     ;
logic                                        rd_meta_empty                   ;
logic                                        rd_meta_cur_first_beat          ;
logic                                        rd_meta_cur_last_beat           ;
logic                                        rd_row_done_fire                ;
logic    [DMA_DIM_WIDTH:0]                   rd_done_row_cnt                 ;

logic    [RF_DATA_WIDTH/2-1:0]               araddr_row_first                ;  //each row ,the first read addr
logic    [DMA_DIM_WIDTH:0]                   araddr_row_cnt                  ;  //counter for row change
logic    [DMA_DIM_WIDTH:0]                   rdata_row_cnt                   ;  //counter for row change
logic    [BURST_ALIGN_BITWID-1:0]            rdata_head_offset               ;  //each row, the first transfer's offset
logic    [BURST_ALIGN_BITWID-1:0]            rdata_tail_bytes                ;  //each row, the last transfer's bytes
logic    [RF_DATA_WIDTH/2-1:0]               rdata_row_first                 ;

logic    [RF_DATA_WIDTH/2-1:0]               awaddr_row_first                ;  //each row ,the first read addr
logic    [DMA_DIM_WIDTH:0]                   awaddr_row_cnt                  ;  //counter for row change
logic    [8 : 0]                             wdata_burst_cnt                 ;   //the cnt for each burst
logic    [BURST_ALIGN_BITWID-1:0]            wdata_head_offset               ;  //each row, the first transfer's offset
logic    [BURST_ALIGN_BITWID-1:0]            wdata_head_offset_pre           ;  //for sram pre fetch
logic    [BURST_ALIGN_BITWID-1:0]            wdata_tail_bytes                ;  //each row, the last transfer's bytes
logic    [BURST_ALIGN_BITWID-1:0]            wdata_tail_bytes_comb           ;
logic    [RF_DATA_WIDTH/2-1:0]               wdata_row_first                 ;
logic    [RF_DATA_WIDTH/2-1:0]               wdata_row_first_pre             ;  
logic    [AXI_DATA_WIDTH/8-1:0]              w_mask_head                     ;
logic    [AXI_DATA_WIDTH/8-1:0]              w_mask_tail                     ;

always_ff@(posedge clk or negedge rst_n) begin 
    if(~rst_n)begin
        mvin_dram_data_size_log2 <= '0; 
        mvin_sram_data_size_log2 <= '0; 
    end
    else if(dma_mvin_req_en)begin
        case ({cfg_mvin_input_type[1],cfg_mvin_input_precision,cfg_mvin_is_quant_effective})
            4'b0010:begin
                mvin_dram_data_size_log2 <= 0; //int8
                mvin_sram_data_size_log2 <= 0;  //int8
            end 
            4'b0100,4'b1100:begin
                mvin_dram_data_size_log2 <= 1; //fp16
                mvin_sram_data_size_log2 <= 1;  //fp16
            end 
            4'b1010:begin
                mvin_dram_data_size_log2 <= 2; //int32
                mvin_sram_data_size_log2 <= 2;  //int32
            end 
            4'b0011:begin  //quant:int8 -> int32
                mvin_dram_data_size_log2 <= 0;  //int8
                mvin_sram_data_size_log2 <= 2;   //int32
            end
            default: begin
                mvin_dram_data_size_log2 <= 0;
                mvin_sram_data_size_log2 <= 0;
            end
        endcase
    end
    else begin
        mvin_dram_data_size_log2 <=mvin_dram_data_size_log2;
        mvin_sram_data_size_log2 <=mvin_sram_data_size_log2;
    end
end
logic quant_out_valid;
assign cfg_mvin_is_quant_effective = (DISABLE_MVIN_INT8_TO_INT32 == 0) && cfg_mvin_is_quant;

generate
    if (DISABLE_MVIN_INT8_TO_INT32 == 1) begin : gen_disable_dma_quant
        assign dequant_data   = '0;
        assign quant_out_valid = 1'b0;
    end
    else begin : gen_dma_quant
        quantization_intX_to_intY #( 
           .INPUT_DATA_WIDTH   (8),   
           .OUTPUT_DATA_WIDTH  (32), 
           .SCALE_DATA_WIDTH   (SCALE_DATA_WIDTH), 
           .INPUT_NUMBER       (AXI_DATA_WIDTH/8)  
        )u_dequantization_int8_to_int32(
            .clk                   (clk),
            .rst_n                 (rst_n),
            .in_valid              (cfg_mvin_is_quant_effective && sram_wr_en),
            .out_valid             (quant_out_valid),
            .unquantized_input     (sram_din_pre),    
            .quant_scale           (cfg_mvin_input_scale),       
            .scale_shift           (cfg_mvin_input_scale_shift),       
            .quantized_output      (dequant_data)  
        );
    end
endgenerate

always_ff@(posedge clk or negedge rst_n) begin 
    if(~rst_n)begin
        mvout_dram_data_size_log2 <= '0; 
        mvout_sram_data_size_log2 <= '0; 
    end
    else if(dma_mvout_req_en)begin
        case ({cfg_mvout_output_type[0],cfg_mvout_output_precision})
            3'b010:begin
                mvout_dram_data_size_log2 <= 1; //fp16  not used now
                mvout_sram_data_size_log2 <= 1;  //fp16
            end 
            3'b001:begin
                mvout_dram_data_size_log2 <= 0;  //int8
                mvout_sram_data_size_log2 <= 0;  //int8
            end 
            3'b101:begin
                mvout_dram_data_size_log2 <= 2;  //int32
                mvout_sram_data_size_log2 <= 2;  //int32
            end
            3'b011,
            3'b111:begin
                mvout_dram_data_size_log2 <= 2;  //fp32
                mvout_sram_data_size_log2 <= 2;  //fp32
            end
            default: begin
                mvout_dram_data_size_log2 <= 0;
                mvout_sram_data_size_log2 <= 0;
            end
        endcase
    end
    else begin
        mvout_dram_data_size_log2 <= mvout_dram_data_size_log2; 
        mvout_sram_data_size_log2 <= mvout_sram_data_size_log2;
    end
end
assign  m_axi_arid           = 0                       ;
assign  m_axi_arsize         = AXI_AxSIZE              ;
assign  m_axi_arburst        = 2'b01                   ;     //INCR  burst type
assign  m_axi_awid           = 0                       ;
assign  m_axi_awsize         = AXI_AxSIZE              ; 
assign  m_axi_awburst        = 2'b01                   ;

assign  mvin_rows_real       = {{(REAL_NUM_WIDTH-DMA_DIM_WIDTH){1'b0}}, mvin_row_num} + 1'b1;
assign  mvin_cols_real       = {{(REAL_NUM_WIDTH-DMA_DIM_WIDTH){1'b0}}, mvin_col_num} + 1'b1;
assign  mvout_rows_real      = {{(REAL_NUM_WIDTH-DMA_DIM_WIDTH){1'b0}}, mvout_row_num} + 1'b1;
assign  mvout_cols_real      = {{(REAL_NUM_WIDTH-DMA_DIM_WIDTH){1'b0}}, mvout_col_num} + 1'b1;
assign  mvin_contiguous_2d   = (mvin_row_num != '0) &&
                               (cfg_mvin_sram_stride == mvin_cols_real[RF_DATA_WIDTH/4-1:0]) &&
                               (cfg_mvin_dram_stride == mvin_cols_real[RF_DATA_WIDTH/2-1:0]);
assign  mvout_contiguous_2d  = (mvout_row_num != '0) &&
                               (cfg_mvout_sram_stride == mvout_cols_real[RF_DATA_WIDTH/4-1:0]) &&
                               (cfg_mvout_dram_stride == mvout_cols_real[RF_DATA_WIDTH/2-1:0]);


// MVOUT ACC FP32
// Pre-buffer a full AXI burst before issuing AW/W for ACC->DRAM fp32 writes.
localparam int FP32_STREAM_FIFO_DEPTH = 32;
localparam int FP32_STREAM_FIFO_PTR_W = $clog2(FP32_STREAM_FIFO_DEPTH);

logic                                 mvout_acc_fp32_mode;
logic                                 mvout_acc_fp32_mode_req;
logic                                 mvout_acc_fp32_active;
logic                                 mvout_prefetch_kick;
logic                                 mvout_acc_rd_fire;

logic [RF_DATA_WIDTH/2-1:0]           fp32_rd_row_first_pre;
logic [RF_DATA_WIDTH/2-1:0]           fp32_rd_row_cnt;
logic [RF_DATA_WIDTH/2-1:0]           fp32_rd_total_beats;
logic [RF_DATA_WIDTH/2-1:0]           fp32_rd_remain_beats;

logic [BURST_ALIGN_BITWID-1:0]        fp32_rd_head_offset_pre;
logic [BURST_ALIGN_BITWID:0]          fp32_rd_tail_bytes_comb;
logic [RF_DATA_WIDTH/2-1:0]           fp32_rd_next_row_first;
logic [RF_DATA_WIDTH/2-1:0]           fp32_rd_next_row_beats;

logic                                 fp32_fifo_push;
logic                                 fp32_fifo_pop;
logic                                 fp32_can_issue_rd;
logic [SPM_DATA_WIDTH-1:0]            fp32_fifo_rdata;
logic                                 fp32_axi_data_load;
logic                                 fp32_axi_data_take;
logic                                 fp32_axi_data_valid;
logic                                 fp32_axi_data_valid_nxt;
logic [SPM_DATA_WIDTH-1:0]            fp32_axi_data;

logic [FP32_STREAM_FIFO_PTR_W-1:0]    fp32_fifo_wr_ptr;
logic [FP32_STREAM_FIFO_PTR_W-1:0]    fp32_fifo_rd_ptr;
logic [$clog2(FP32_STREAM_FIFO_DEPTH+1)-1:0] fp32_fifo_count;
logic [$clog2(FP32_STREAM_FIFO_DEPTH+1)-1:0] fp32_inflight_cnt;
logic [$clog2(FP32_STREAM_FIFO_DEPTH+1)-1:0] fp32_fifo_count_nxt;
logic                                        m_axi_wvalid_nxt;
logic [AXI_DATA_WIDTH-1:0]                   m_axi_wdata_pre;
logic [AXI_DATA_WIDTH/8-1:0]                 m_axi_wstrb_pre;

logic [SPM_DATA_WIDTH-1:0]            fp32_wdata_fifo [0:FP32_STREAM_FIFO_DEPTH-1];

assign mvout_acc_fp32_mode_req = cfg_mvout_source && (cfg_mvout_output_precision == 2'b11);
assign mvout_acc_fp32_active   = mvout_acc_fp32_mode || mvout_acc_fp32_mode_req;
assign mvout_prefetch_kick   = reg_dma_mvout_req_en[2] && !mvout_acc_fp32_mode;

assign fp32_rd_head_offset_pre = fp32_rd_row_first_pre[BURST_ALIGN_BITWID-1:0];
assign fp32_rd_tail_bytes_comb = fp32_rd_row_first_pre[BURST_ALIGN_BITWID-1:0] +
                                 reg_write_col_num[BURST_ALIGN_BITWID-1:0];
assign fp32_rd_next_row_first  = fp32_rd_row_first_pre + reg_cfg_mvout_dram_stride;
assign fp32_rd_next_row_beats  = (fp32_rd_next_row_first[BURST_ALIGN_BITWID-1:0] +
                                  (BURST_ADDR_INCR - 1) + reg_write_col_num) >> BURST_ALIGN_BITWID;

assign fp32_fifo_push    = mvout_acc_fp32_mode && acc_dout_valid;
assign fp32_axi_data_take = mvout_acc_fp32_mode && m_axi_wvalid && m_axi_wready;
assign fp32_axi_data_load = mvout_acc_fp32_mode &&
                            (!fp32_axi_data_valid || fp32_axi_data_take) &&
                            (fp32_fifo_count != 0);
assign fp32_fifo_pop     = fp32_axi_data_load;
assign fp32_can_issue_rd = (fp32_fifo_count + fp32_inflight_cnt) < FP32_STREAM_FIFO_DEPTH;
assign fp32_fifo_rdata   = fp32_wdata_fifo[fp32_fifo_rd_ptr];
always_comb begin
    fp32_fifo_count_nxt = fp32_fifo_count;

    case ({fp32_fifo_push, fp32_fifo_pop})
        2'b10: fp32_fifo_count_nxt = fp32_fifo_count + 1'b1;
        2'b01: fp32_fifo_count_nxt = fp32_fifo_count - 1'b1;
        default: fp32_fifo_count_nxt = fp32_fifo_count;
    endcase
end

always_comb begin
    fp32_axi_data_valid_nxt = fp32_axi_data_valid;

    if(fp32_axi_data_take)
        fp32_axi_data_valid_nxt = 1'b0;

    if(fp32_axi_data_load)
        fp32_axi_data_valid_nxt = 1'b1;
end

assign mvout_acc_rd_fire =
    mvout_acc_fp32_mode &&
    current_write_state == S_WRITE &&
    fp32_can_issue_rd &&
    (fp32_rd_row_cnt < reg_write_row_num) &&
    (fp32_rd_remain_beats != 0);

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        mvout_acc_fp32_mode <= 1'b0;
    end else if (reg_dma_mvout_req_en[0]) begin
        mvout_acc_fp32_mode <= mvout_acc_fp32_mode_req;
    end else if (dma_mvout_resp_done) begin
        mvout_acc_fp32_mode <= 1'b0;
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n || dma_mvout_resp_done) begin
        fp32_rd_row_first_pre <= '0;
        fp32_rd_row_cnt       <= '0;
        fp32_rd_total_beats   <= '0;
        fp32_rd_remain_beats  <= '0;
    end
    else if(reg_dma_mvout_req_en[1]) begin
        fp32_rd_row_first_pre <= mvout_dram_addr;
        fp32_rd_row_cnt       <= '0;
        fp32_rd_total_beats   <= (mvout_dram_addr[BURST_ALIGN_BITWID-1:0] + (BURST_ADDR_INCR - 1) + reg_write_col_num)
                                 >> BURST_ALIGN_BITWID;
        fp32_rd_remain_beats  <= (mvout_dram_addr[BURST_ALIGN_BITWID-1:0] + (BURST_ADDR_INCR - 1) + reg_write_col_num)
                                 >> BURST_ALIGN_BITWID;
    end
    else if(mvout_acc_rd_fire) begin
        if(fp32_rd_total_beats == 1 || fp32_rd_remain_beats == 1) begin
            fp32_rd_row_cnt <= fp32_rd_row_cnt + 1'b1;
            if(fp32_rd_row_cnt + 1'b1 == reg_write_row_num) begin
                fp32_rd_row_first_pre <= '0;
                fp32_rd_total_beats   <= '0;
                fp32_rd_remain_beats  <= '0;
            end
            else begin
                fp32_rd_row_first_pre <= fp32_rd_next_row_first;
                fp32_rd_total_beats   <= fp32_rd_next_row_beats;
                fp32_rd_remain_beats  <= fp32_rd_next_row_beats;
            end
        end
        else begin
            fp32_rd_remain_beats <= fp32_rd_remain_beats - 1'b1;
        end
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n || dma_mvout_resp_done) begin
        fp32_fifo_wr_ptr <= '0;
        fp32_fifo_rd_ptr <= '0;
        fp32_fifo_count  <= '0;
        fp32_inflight_cnt <= '0;
        fp32_axi_data_valid <= 1'b0;
        fp32_axi_data <= '0;
    end
    else begin
        if(fp32_fifo_push) begin
            fp32_wdata_fifo[fp32_fifo_wr_ptr] <= acc_dout[SPM_DATA_WIDTH-1:0];
        end

        if(fp32_axi_data_load) begin
            fp32_axi_data <= fp32_fifo_rdata;
        end
        fp32_axi_data_valid <= fp32_axi_data_valid_nxt;

        case ({fp32_fifo_push, fp32_fifo_pop})
            2'b10: begin
                fp32_fifo_wr_ptr <= fp32_fifo_wr_ptr + 1'b1;
                fp32_fifo_count  <= fp32_fifo_count + 1'b1;
            end
            2'b01: begin
                fp32_fifo_rd_ptr <= fp32_fifo_rd_ptr + 1'b1;
                fp32_fifo_count  <= fp32_fifo_count - 1'b1;
            end
            2'b11: begin
                fp32_fifo_wr_ptr <= fp32_fifo_wr_ptr + 1'b1;
                fp32_fifo_rd_ptr <= fp32_fifo_rd_ptr + 1'b1;
            end
            default: begin
                fp32_fifo_wr_ptr <= fp32_fifo_wr_ptr;
                fp32_fifo_rd_ptr <= fp32_fifo_rd_ptr;
                fp32_fifo_count  <= fp32_fifo_count;
            end
        endcase

        case ({mvout_acc_rd_fire, fp32_fifo_push})
            2'b10: fp32_inflight_cnt <= fp32_inflight_cnt + 1'b1;
            2'b01: fp32_inflight_cnt <= fp32_inflight_cnt - 1'b1;
            default: fp32_inflight_cnt <= fp32_inflight_cnt;
        endcase
    end
end


//---------------------------------------------    
// write address channel signal generate     
//--------------------------------------------- 
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    m_axi_awaddr <= 0;
    else if(dma_mvout_req_en) //mvout en, awaddr = the initial addr after align process 
    m_axi_awaddr <= {mvout_dram_addr[RF_DATA_WIDTH/2-1:BURST_ALIGN_BITWID],{(BURST_ALIGN_BITWID){1'b0}}};
    else if(((m_axi_awlen+1)>=aw_remain_beats) && m_axi_awvalid && m_axi_awready)  //each row finish,addr=addr+stride
    m_axi_awaddr <= (((awaddr_row_first + reg_cfg_mvout_dram_stride)>>BURST_ALIGN_BITWID)<<BURST_ALIGN_BITWID);
    else if(m_axi_awvalid && m_axi_awready)  //handshake, addr+incr*len
    m_axi_awaddr <= m_axi_awaddr + (m_axi_awlen+1)*BURST_ADDR_INCR;
    else 
    m_axi_awaddr <= m_axi_awaddr;
end

always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    m_axi_awvalid <= 0;
    else if(((m_axi_awlen+1)>=aw_remain_beats) && m_axi_awready && m_axi_awvalid) //each row ,the last handshake finish,valid=0
    m_axi_awvalid<= 0;
    else if((w_outstanding_cnt>=(AXI_WR_OUTSTANDING-1)) && m_axi_awvalid && m_axi_awready && !(m_axi_bvalid && m_axi_bready))
    m_axi_awvalid <= 0;
    else if(w_outstanding_cnt>=AXI_WR_OUTSTANDING)
    m_axi_awvalid <= 0;
    else if(current_waddr_state==S_WRITE && aw_remain_beats>0 && w_outstanding_cnt<AXI_WR_OUTSTANDING ) //state correct,has data to transfer,valid=1
    m_axi_awvalid <= 1;
    else 
    m_axi_awvalid <= m_axi_awvalid;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    awaddr_row_cnt <= 0;
    else if(awaddr_row_cnt == reg_write_row_num && w_outstanding_cnt == 0)   //the all trans finish ,row cnt set to 0
    awaddr_row_cnt <= 0;
    else if(aw_remain_beats==0 && current_waddr_state != IDLE) //each row, the last finish,row cnt+1
    awaddr_row_cnt <= awaddr_row_cnt + 1;
    else 
    awaddr_row_cnt <= awaddr_row_cnt;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    awaddr_row_first <= 0;
    else if(dma_mvout_req_en) //initial row first addr =  ini addr
    awaddr_row_first <= mvout_dram_addr;
    else if(awaddr_row_cnt == reg_write_row_num)  //write finish, set to 0
    awaddr_row_first <= 0;
    else if(((m_axi_awlen+1)>=aw_remain_beats)&& m_axi_awvalid && m_axi_awready)  //row change ,cal the next row first addr
    awaddr_row_first <= awaddr_row_first + reg_cfg_mvout_dram_stride ;
    else 
    awaddr_row_first <= awaddr_row_first;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    aw_remain_beats<= '1;
    else if(reg_dma_mvout_req_en[1] || aw_remain_beats==0) //cal the beats number for each row
    aw_remain_beats<= (awaddr_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_write_col_num)>>BURST_ALIGN_BITWID;
    else if(m_axi_awvalid && m_axi_awready )  //aw handshake ,remain-beats reduce
    aw_remain_beats<=aw_remain_beats-m_axi_awlen-1;
    else 
    aw_remain_beats<=aw_remain_beats;
end
always_comb begin
    aw_bytes_to_4k          = 13'h1000-{1'b0,m_axi_awaddr[11:0]};   //the distance to 4k boundry
    aw_beats_to_4k          = aw_bytes_to_4k[12:BURST_ALIGN_BITWID];
    aw_remain_beats_clamped = (aw_remain_beats > AXI_MAX_BURST_BEATS) ? AXI_MAX_BURST_BEATS[8:0] : aw_remain_beats[8:0];
    aw_final_beats          = (aw_beats_to_4k<aw_remain_beats_clamped)?aw_beats_to_4k:aw_remain_beats_clamped;  
    m_axi_awlen             = (aw_final_beats>0)?aw_final_beats-1:0;
end

//---------------------------------------------    
// write data channel signal generate     
//---------------------------------------------  
assign  m_axi_wlast             = m_axi_wvalid && (wdata_burst_cnt == w_final_beats - 1);
assign  mvout_w_fire            = m_axi_wvalid && m_axi_wready;
assign  mvout_wlast_fire        = mvout_w_fire && (wdata_burst_cnt == w_final_beats - 1);
assign  wdata_head_offset_pre   = wdata_row_first_pre[BURST_ALIGN_BITWID-1:0] ;
assign  wdata_tail_bytes_comb   = wdata_row_first[BURST_ALIGN_BITWID-1:0] + reg_write_col_num[BURST_ALIGN_BITWID-1:0];
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    wdata_row_first <= 0;
    else if(dma_mvout_req_en) //initial row first addr =  ini addr
    wdata_row_first <= mvout_dram_addr;
    else if(dma_mvout_resp_done)  //write finish, set to 0
    wdata_row_first <= 0;
    else if(w_remain_beats_runtime == 1 && m_axi_wready && m_axi_wvalid)  //row change ,cal the next row first addr
    wdata_row_first <= wdata_row_first + reg_cfg_mvout_dram_stride ;
    else 
    wdata_row_first <= wdata_row_first;
end
always_ff @(posedge clk or negedge rst_n) begin  //for sram prefetch data, cal the next row first addr in advance
    if(rst_n == 0)
    wdata_row_first_pre <= 0;
    else if(dma_mvout_req_en) //initial row first addr =  ini addr
    wdata_row_first_pre <= mvout_dram_addr;
    else if(dma_mvout_resp_done)  //write finish, set to 0
    wdata_row_first_pre <= 0;
    else if(w_total_beats_pre==1 && (reg_dma_mvout_req_en[2]||m_axi_wready && m_axi_wvalid))//for total==1,row change 
    wdata_row_first_pre <= wdata_row_first_pre + reg_cfg_mvout_dram_stride ;
    else if(w_remain_beats_runtime == 2 && m_axi_wready && m_axi_wvalid)  //remain ==2, row change in advance ,cal the next row first addr
    wdata_row_first_pre <= wdata_row_first_pre + reg_cfg_mvout_dram_stride ;
    else 
    wdata_row_first_pre <= wdata_row_first_pre;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)begin
        w_mask_head         <='0;
        wdata_head_offset   <='0;
    end
    else if((w_remain_beats_runtime == 0 && current_write_state == S_WRITE) || (reg_dma_mvout_req_en[0]))begin
        w_mask_head         <= {(AXI_DATA_WIDTH/8){1'b1}}<<(wdata_row_first[BURST_ALIGN_BITWID-1:0]);
        wdata_head_offset   <= wdata_row_first[BURST_ALIGN_BITWID-1:0] ;
    end
    else begin
        w_mask_head         <=w_mask_head;
        wdata_head_offset   <=wdata_head_offset;
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)begin
        w_mask_tail         <='0;
        wdata_tail_bytes    <='0;
    end
    else if((w_remain_beats_runtime == 0 && current_write_state == S_WRITE) || (reg_dma_mvout_req_en[1]))begin
        wdata_tail_bytes    <= wdata_row_first[BURST_ALIGN_BITWID-1:0] + reg_write_col_num[BURST_ALIGN_BITWID-1:0];
        w_mask_tail         <= wdata_tail_bytes_comb==0?{(AXI_DATA_WIDTH/8){1'b1}}:(~({(AXI_DATA_WIDTH/8){1'b1}}<<(wdata_tail_bytes_comb)));
    end
    else begin
        w_mask_tail         <=w_mask_tail;
        wdata_tail_bytes    <=wdata_tail_bytes;
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    wdata_burst_cnt <= 0;
    else if(mvout_wlast_fire)
    wdata_burst_cnt <= 0;
    else if(m_axi_wready && m_axi_wvalid) //in burst ,the handshake cnt
    wdata_burst_cnt <= wdata_burst_cnt + 1;
    else 
    wdata_burst_cnt <= wdata_burst_cnt;    
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    wdata_row_cnt <= 0;
    else if(dma_mvout_resp_done)
    wdata_row_cnt <= 0;
    else if(w_remain_beats==0 && current_write_state==S_WRITE )
    wdata_row_cnt <= wdata_row_cnt + 1;
    else 
    wdata_row_cnt <= wdata_row_cnt;    
end
always_comb begin
    if(w_total_beats == 1)
        m_axi_wstrb_pre = w_mask_head & w_mask_tail;
    else if(w_remain_beats_runtime == w_total_beats)
        m_axi_wstrb_pre = w_mask_head;
    else if(w_remain_beats_runtime == 1)
        m_axi_wstrb_pre = w_mask_tail;
    else
        m_axi_wstrb_pre = {(AXI_DATA_WIDTH/8){1'b1}};
end
always_comb begin
    if(mvout_acc_fp32_mode)
        m_axi_wvalid_nxt = fp32_axi_data_valid_nxt;
    else if(w_remain_beats_runtime == 1 && m_axi_wvalid && m_axi_wready)
        m_axi_wvalid_nxt = 1'b0;
    else if(w_remain_beats_runtime > 0 && current_write_state == S_WRITE && wdata_row_cnt < reg_write_row_num)
        m_axi_wvalid_nxt = 1'b1;
    else
        m_axi_wvalid_nxt = m_axi_wvalid;
end

always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
        m_axi_wvalid <= 1'b0;
    else
        m_axi_wvalid <= m_axi_wvalid_nxt;
end

always_comb begin
    m_axi_wdata = m_axi_wdata_pre;
    m_axi_wstrb = m_axi_wstrb_pre;
end


always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    w_remain_beats<='1;
    else if(reg_dma_mvout_req_en[1] || w_remain_beats==0)
    w_remain_beats<= (wdata_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_write_col_num)>>BURST_ALIGN_BITWID;
    else if(mvout_wlast_fire)
    w_remain_beats<=w_remain_beats - w_final_beats;
    else 
    w_remain_beats<=w_remain_beats;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    w_remain_beats_runtime<='1;
    else if(reg_dma_mvout_req_en[1] || w_remain_beats_runtime==0)
    w_remain_beats_runtime<= (wdata_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_write_col_num)>>BURST_ALIGN_BITWID;
    else if(m_axi_wvalid && m_axi_wready)
    w_remain_beats_runtime<=w_remain_beats_runtime - 1;
    else 
    w_remain_beats_runtime<=w_remain_beats_runtime;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    w_total_beats<='0;
    else if(reg_dma_mvout_req_en[1] || w_remain_beats==0)
    w_total_beats<= (wdata_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_write_col_num)>>BURST_ALIGN_BITWID;
    else 
    w_total_beats<=w_total_beats;
end
assign w_total_beats_pre = (wdata_row_first_pre[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_write_col_num)>>BURST_ALIGN_BITWID;
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    w_shadow_addr<='0;
    else if(dma_mvout_req_en)  
    w_shadow_addr <= {mvout_dram_addr[RF_DATA_WIDTH/2-1:BURST_ALIGN_BITWID],{(BURST_ALIGN_BITWID){1'b0}}};
    else if(w_remain_beats_runtime == 1 && m_axi_wvalid && m_axi_wready)  //next row
    w_shadow_addr <= (((wdata_row_first + reg_cfg_mvout_dram_stride)>>BURST_ALIGN_BITWID)<<BURST_ALIGN_BITWID);
    else if(mvout_wlast_fire)  
    w_shadow_addr <= w_shadow_addr + w_final_beats*BURST_ADDR_INCR;
    else 
    w_shadow_addr <= w_shadow_addr;
end
always_comb begin
    w_bytes_to_4k          = 13'h1000-{1'b0,w_shadow_addr[11:0]};
    w_beats_to_4k          = w_bytes_to_4k[12:BURST_ALIGN_BITWID];
    w_remain_beats_clamped = (w_remain_beats > AXI_MAX_BURST_BEATS) ? AXI_MAX_BURST_BEATS[8:0] : w_remain_beats[8:0];
    w_final_beats          = (w_beats_to_4k<w_remain_beats_clamped)?w_beats_to_4k:w_remain_beats_clamped;
end

//---------------------------------------------    
// write response channel signal generate     
//--------------------------------------------- 
assign m_axi_bready = 1'b1;
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)
    w_outstanding_cnt<='0;
    else begin
        case({m_axi_awvalid && m_axi_awready,m_axi_bvalid && m_axi_bready})
            2'b10:w_outstanding_cnt<=w_outstanding_cnt+1;
            2'b01:w_outstanding_cnt<=w_outstanding_cnt-1;
            default:w_outstanding_cnt<=w_outstanding_cnt;
        endcase
    end
end
always_ff @(posedge clk or negedge rst_n) begin        //write done signal generate 
    if(rst_n == 0)
    dma_mvout_resp_done <= 0;
    else if(awaddr_row_cnt == reg_write_row_num && w_outstanding_cnt == 0) //bresp row cnt == row num,represent the finish of mvout
    dma_mvout_resp_done <= 1;
    else
    dma_mvout_resp_done <= 0;    
end
//---------------------------------------------    
//  mvout spm signal generate     
//--------------------------------------------- 

always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    fore_flag <= 1;
    else if(fore_flag == 1)begin
        if((m_axi_wready && m_axi_wvalid)||mvout_prefetch_kick)
        fore_flag <= 0;
        else 
        fore_flag <= 1;
    end
    else begin
        if(!(m_axi_wready && m_axi_wvalid))
        fore_flag <= 1;
        else 
        fore_flag <= 0; 
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    fore_sram_dout <= 0;
    else if(fore_flag == 0)
    fore_sram_dout <= (cfg_mvout_source==1'b0)?spm_dout:acc_dout[SPM_DATA_WIDTH-1:0];
    else 
    fore_sram_dout <= fore_sram_dout;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    sram_rd_addr <= 0;
    else if(reg_dma_mvout_req_en[1])
    sram_rd_addr <= mvout_sram_addr;
    else if(mvout_acc_fp32_mode && mvout_acc_rd_fire) begin
        if(fp32_rd_total_beats == 1)
        sram_rd_addr <= sram_rd_addr + reg_cfg_mvout_sram_stride +
                        (cfg_mvout_source ? reg_write_col_num :
                         (cfg_mvout_sram_stride << (mvout_sram_data_size_log2-mvout_dram_data_size_log2)));
        else if(fp32_rd_remain_beats == 1)
        sram_rd_addr <= sram_rd_addr + ((BURST_ADDR_INCR - fp32_rd_head_offset_pre)
                        << (mvout_sram_data_size_log2-mvout_dram_data_size_log2));
        else if(fp32_rd_remain_beats == 2)
        sram_rd_addr <= sram_rd_addr + (BURST_ADDR_INCR << (mvout_sram_data_size_log2-mvout_dram_data_size_log2));
        else
        sram_rd_addr <= sram_rd_addr + (BURST_ADDR_INCR << (mvout_sram_data_size_log2-mvout_dram_data_size_log2));
    end
    else if(!mvout_acc_fp32_mode && w_total_beats_pre==1 && ((m_axi_wvalid && m_axi_wready)||mvout_prefetch_kick))
    sram_rd_addr <= sram_rd_addr + reg_cfg_mvout_sram_stride +
                    (cfg_mvout_source ? reg_write_col_num :
                     (cfg_mvout_sram_stride<<(mvout_sram_data_size_log2-mvout_dram_data_size_log2)));
    else if(!mvout_acc_fp32_mode && ((w_remain_beats_runtime== 1 && m_axi_wready && m_axi_wvalid)||mvout_prefetch_kick))
    sram_rd_addr <= sram_rd_addr + ((BURST_ADDR_INCR -wdata_head_offset_pre)<<(mvout_sram_data_size_log2-mvout_dram_data_size_log2));
    else if(!mvout_acc_fp32_mode && w_remain_beats_runtime== 2 && m_axi_wready && m_axi_wvalid)
    sram_rd_addr <= sram_rd_addr + reg_cfg_mvout_sram_stride+((((|wdata_tail_bytes)?wdata_tail_bytes:BURST_ADDR_INCR))<<(mvout_sram_data_size_log2-mvout_dram_data_size_log2));
    else if(!mvout_acc_fp32_mode && (m_axi_wready && m_axi_wvalid ))
    sram_rd_addr <= sram_rd_addr + (BURST_ADDR_INCR<<(mvout_sram_data_size_log2-mvout_dram_data_size_log2));
    else
    sram_rd_addr <= sram_rd_addr;
end
always_comb begin
    if(w_remain_beats_runtime == w_total_beats)
        m_axi_wdata_pre = axi_wdata << (wdata_head_offset * 8);
    else
        m_axi_wdata_pre = axi_wdata;
end
always_comb begin 
    if(cfg_mvout_source==1) begin
        if(mvout_acc_fp32_mode)
            axi_wdata = fp32_axi_data;
        else
            axi_wdata = fore_flag ? fore_sram_dout : acc_dout[SPM_DATA_WIDTH-1:0];
    end else begin
        axi_wdata = fore_flag ? fore_sram_dout : spm_dout;
    end
end
always_comb begin
 sram_rd_en   = current_write_state==S_WRITE;
 spm_rd_en    = cfg_mvout_source == 0 && sram_rd_en;
 acc_rd_en    = cfg_mvout_source == 1 &&
                (mvout_acc_fp32_mode ? mvout_acc_rd_fire : sram_rd_en);
 spm_rd_addr  = cfg_mvout_source == 0 ? sram_rd_addr[SPM_ADDR_WIDTH-1:0] : '0;
 acc_rd_addr  = cfg_mvout_source == 1 ? sram_rd_addr[ACC_ADDR_WIDTH-1:0] : '0;
end
//---------------------------------------------    
// read address channel signal generate     
//---------------------------------------------  
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    m_axi_araddr <= 0;
    else if(dma_mvin_req_en)
    m_axi_araddr <= {mvin_dram_addr[RF_DATA_WIDTH/2-1:BURST_ALIGN_BITWID],{(BURST_ALIGN_BITWID){1'b0}}};
    else if(((m_axi_arlen+1)>=ar_remain_beats) && m_axi_arvalid && m_axi_arready)
    m_axi_araddr <= (((araddr_row_first + reg_cfg_mvin_dram_stride )>>BURST_ALIGN_BITWID)<<BURST_ALIGN_BITWID);
    else if(m_axi_arvalid && m_axi_arready)
    m_axi_araddr <= m_axi_araddr + (m_axi_arlen+1)*BURST_ADDR_INCR;
    else 
    m_axi_araddr <= m_axi_araddr;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    m_axi_arvalid <= 0;
    else if(((m_axi_arlen+1)>=ar_remain_beats) && m_axi_arready && m_axi_arvalid)
    m_axi_arvalid<= 0;
    else if((r_outstanding_cnt>=(AXI_RD_OUTSTANDING-1)) && m_axi_arvalid && m_axi_arready && !(m_axi_rlast && m_axi_rvalid && m_axi_rready))
    m_axi_arvalid <= 0;
    else if(r_outstanding_cnt>=AXI_RD_OUTSTANDING)
    m_axi_arvalid <= 0;
    else if(current_raddr_state==S_READ &&  ar_remain_beats>0 && r_outstanding_cnt<AXI_RD_OUTSTANDING)
    m_axi_arvalid <= 1;
    else 
    m_axi_arvalid <= m_axi_arvalid;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)
    r_outstanding_cnt<='0;
    else begin
        case({m_axi_arvalid && m_axi_arready,m_axi_rlast&&m_axi_rvalid&&m_axi_rready})
            2'b10:r_outstanding_cnt<=r_outstanding_cnt+1;
            2'b01:r_outstanding_cnt<=r_outstanding_cnt-1;
            default:r_outstanding_cnt<=r_outstanding_cnt;
        endcase
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    araddr_row_cnt <= 0;
    else if(dma_mvin_resp_done)
    araddr_row_cnt <= 0;
    else if(ar_remain_beats==0 && current_raddr_state!=IDLE)
    araddr_row_cnt <= araddr_row_cnt + 1;
    else 
    araddr_row_cnt <= araddr_row_cnt;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    araddr_row_first <= 0;
    else if(dma_mvin_req_en)
    araddr_row_first <= mvin_dram_addr;
    else if(araddr_row_cnt == reg_read_row_num)
    araddr_row_first <= 0;
    else if(((m_axi_arlen+1)>=ar_remain_beats) && m_axi_arvalid && m_axi_arready)
    araddr_row_first <= araddr_row_first + reg_cfg_mvin_dram_stride ;
    else 
    araddr_row_first <= araddr_row_first;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    ar_remain_beats<= '1;
    else if(reg_dma_mvin_req_en[1] || ar_remain_beats==0)
    ar_remain_beats<= (araddr_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_read_col_num)>>BURST_ALIGN_BITWID;
    else if(m_axi_arvalid && m_axi_arready )
    ar_remain_beats<=ar_remain_beats-m_axi_arlen-1;
    else 
    ar_remain_beats<=ar_remain_beats;
end
always_comb begin
    ar_bytes_to_4k          = 13'h1000-{1'b0,m_axi_araddr[11:0]};
    ar_beats_to_4k          = ar_bytes_to_4k[12:BURST_ALIGN_BITWID];
    ar_total_beats          = (araddr_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_read_col_num)>>BURST_ALIGN_BITWID;
    ar_remain_beats_clamped = (ar_remain_beats > AXI_MAX_BURST_BEATS) ? AXI_MAX_BURST_BEATS[8:0] : ar_remain_beats[8:0];
    ar_final_beats          = (ar_beats_to_4k<ar_remain_beats_clamped)?ar_beats_to_4k:ar_remain_beats_clamped;
    m_axi_arlen             = (ar_final_beats>0)?ar_final_beats-1:0;
end

assign rd_meta_push           = m_axi_arvalid && m_axi_arready;
assign rd_meta_pop            = m_axi_rvalid && m_axi_rready && m_axi_rlast;
assign rd_meta_empty          = rd_meta_count == '0;
assign rd_meta_cur_first_beat = rd_burst_beat_cnt == 0;
assign rd_meta_cur_last_beat  = rd_burst_beat_cnt == (rd_meta_burst_beats[rd_meta_rd_ptr] - 1);
assign rd_row_done_fire       = sram_wr_en && rd_meta_cur_last_beat && rd_meta_row_last[rd_meta_rd_ptr];

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        rd_meta_wr_ptr <= '0;
        rd_meta_rd_ptr <= '0;
        rd_meta_count  <= '0;
    end
    else if(dma_mvin_req_en) begin
        rd_meta_wr_ptr <= '0;
        rd_meta_rd_ptr <= '0;
        rd_meta_count  <= '0;
    end
    else begin
        if(rd_meta_push)
            rd_meta_wr_ptr <= rd_meta_wr_ptr + 1'b1;
        if(rd_meta_pop)
            rd_meta_rd_ptr <= rd_meta_rd_ptr + 1'b1;

        case({rd_meta_push, rd_meta_pop})
            2'b10: rd_meta_count <= rd_meta_count + 1'b1;
            2'b01: rd_meta_count <= rd_meta_count - 1'b1;
            default: rd_meta_count <= rd_meta_count;
        endcase
    end
end

always_ff @(posedge clk) begin
    if(rd_meta_push) begin
        rd_meta_burst_beats[rd_meta_wr_ptr] <= {1'b0, m_axi_arlen} + 1'b1;
        rd_meta_row_first[rd_meta_wr_ptr]   <= ar_remain_beats == ar_total_beats;
        rd_meta_row_last[rd_meta_wr_ptr]    <= (m_axi_arlen + 1'b1) >= ar_remain_beats;
        rd_meta_row_single[rd_meta_wr_ptr]  <= ar_total_beats == 1;
        rd_meta_head_offset[rd_meta_wr_ptr] <= (ar_remain_beats == ar_total_beats) ? araddr_row_first[BURST_ALIGN_BITWID-1:0] : '0;
        rd_meta_tail_bytes[rd_meta_wr_ptr]  <= (((m_axi_arlen + 1'b1) >= ar_remain_beats) ?
                                                (araddr_row_first[BURST_ALIGN_BITWID-1:0] + reg_read_col_num[BURST_ALIGN_BITWID-1:0]) : '0);
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        rd_burst_beat_cnt <= '0;
    else if(dma_mvin_req_en || rd_meta_pop)
        rd_burst_beat_cnt <= '0;
    else if(m_axi_rvalid && m_axi_rready)
        rd_burst_beat_cnt <= rd_burst_beat_cnt + 1'b1;
    else
        rd_burst_beat_cnt <= rd_burst_beat_cnt;
end

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        rd_done_row_cnt <= '0;
    else if(dma_mvin_req_en || dma_mvin_resp_done)
        rd_done_row_cnt <= '0;
    else if(rd_row_done_fire)
        rd_done_row_cnt <= rd_done_row_cnt + 1'b1;
    else
        rd_done_row_cnt <= rd_done_row_cnt;
end

//---------------------------------------------    
// read data channel signal generate     
//---------------------------------------------  
assign m_axi_rready = 1'b1;
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    rdata_row_first <= 0;
    else if(dma_mvin_req_en) //initial row first addr =  ini addr
    rdata_row_first <= mvin_dram_addr;
    else if(rdata_row_cnt == reg_read_row_num)  //write finish, set to 0
    rdata_row_first <= 0;
    else if(r_remain_beats==1 && m_axi_rvalid && m_axi_rready)  //row change ,cal the next row first addr
    rdata_row_first <= rdata_row_first + reg_cfg_mvin_dram_stride;
    else 
    rdata_row_first <= rdata_row_first;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    rdata_row_cnt <= 0;
    else if(dma_mvin_req_en || dma_mvin_resp_done)
    rdata_row_cnt <= 0;
    else if(rd_row_done_fire)
    rdata_row_cnt <= rd_done_row_cnt + 1'b1;
    else 
    rdata_row_cnt <= rdata_row_cnt;    
end
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    r_remain_beats<='1;
    else if(reg_dma_mvin_req_en[1] || r_remain_beats==0)
    r_remain_beats<= (rdata_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_read_col_num)>>BURST_ALIGN_BITWID;
    else if(m_axi_rvalid && m_axi_rready)
    r_remain_beats<=r_remain_beats - 1;
    else 
    r_remain_beats<=r_remain_beats;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    r_total_beats<='0;
    else if(reg_dma_mvin_req_en[1] || r_remain_beats==0)
    r_total_beats<= (rdata_row_first[BURST_ALIGN_BITWID-1:0]+(BURST_ADDR_INCR - 1)+reg_read_col_num)>>BURST_ALIGN_BITWID;
    else 
    r_total_beats<=r_total_beats;
end
assign  rdata_head_offset  = rdata_row_first[BURST_ALIGN_BITWID-1:0] ;
assign  rdata_tail_bytes   = rdata_row_first[BURST_ALIGN_BITWID-1:0] + reg_read_col_num[BURST_ALIGN_BITWID-1:0];
//---------------------------------------------    
// mvin spm  signal generate     
//--------------------------------------------- 
assign  sram_wr_en  = m_axi_rvalid && m_axi_rready && m_axi_rresp==2'b00 && m_axi_rid== m_axi_arid && !rd_meta_empty;  //axi read ok ,then spm write
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)
    sram_wr_addr <= 0;
    else if(dma_mvin_req_en)
    sram_wr_addr <= mvin_sram_addr;
    else if(rd_meta_row_single[rd_meta_rd_ptr] && rd_meta_cur_first_beat && sram_wr_en)
    sram_wr_addr <= sram_wr_addr + ((cfg_mvin_sram_stride)<<(mvin_sram_data_size_log2-mvin_dram_data_size_log2));
    else if(rd_meta_row_first[rd_meta_rd_ptr] && rd_meta_cur_first_beat && sram_wr_en)
    sram_wr_addr <= sram_wr_addr +( (BURST_ADDR_INCR -rd_meta_head_offset[rd_meta_rd_ptr])<<(mvin_sram_data_size_log2-mvin_dram_data_size_log2));
    else if(rd_meta_row_last[rd_meta_rd_ptr] && rd_meta_cur_last_beat && sram_wr_en)
    sram_wr_addr <= sram_wr_addr + (((|rd_meta_tail_bytes[rd_meta_rd_ptr])?rd_meta_tail_bytes[rd_meta_rd_ptr]:BURST_ADDR_INCR)<<(mvin_sram_data_size_log2-mvin_dram_data_size_log2))+reg_cfg_mvin_sram_stride;
    else if(sram_wr_en)
    sram_wr_addr <= sram_wr_addr + (BURST_ADDR_INCR<<(mvin_sram_data_size_log2-mvin_dram_data_size_log2));
    else 
    sram_wr_addr <= sram_wr_addr;
end


always_comb begin
    if(rd_meta_row_single[rd_meta_rd_ptr] && rd_meta_cur_first_beat)
    sram_wr_mask = (~({{AXI_DATA_WIDTH/(8)}{1'b1}}<<((((|rd_meta_tail_bytes[rd_meta_rd_ptr])?rd_meta_tail_bytes[rd_meta_rd_ptr]:BURST_ADDR_INCR)-rd_meta_head_offset[rd_meta_rd_ptr]))));
    else if(rd_meta_row_first[rd_meta_rd_ptr] && rd_meta_cur_first_beat)
    sram_wr_mask = {(AXI_DATA_WIDTH/8){1'b1}}>>(rd_meta_head_offset[rd_meta_rd_ptr]);
    else if(rd_meta_row_last[rd_meta_rd_ptr] && rd_meta_cur_last_beat)
    sram_wr_mask = rd_meta_tail_bytes[rd_meta_rd_ptr]==0?{(AXI_DATA_WIDTH/8){1'b1}}:(~({(AXI_DATA_WIDTH/8){1'b1}}<<(rd_meta_tail_bytes[rd_meta_rd_ptr])));
    else 
    sram_wr_mask = {{AXI_DATA_WIDTH/(8)}{1'b1}};
end
always_ff@(posedge clk)begin
    sram_wr_en_delay  <=sram_wr_en;
    sram_wr_addr_delay<=sram_wr_addr;
    sram_wr_mask_delay<=sram_wr_mask;
    sram_din          <=sram_din_pre;
    for(int i=0;i<AXI_DATA_WIDTH/8;i++)begin
        sram_wr_mask_bias[i]<= i<AXI_DATA_WIDTH/32?sram_wr_mask[i<<2]:0;
    end
end
always_comb begin 
    spm_wr_en   = cfg_mvin_dest==0?sram_wr_en_delay:'0;
    spm_wr_mask = cfg_mvin_dest==0?sram_wr_mask_delay:'0;
    spm_din     = cfg_mvin_dest==0?sram_din:'0;
    spm_wr_addr = cfg_mvin_dest==0?sram_wr_addr_delay[SPM_ADDR_WIDTH-1:0]:'0;
    acc_wr_en   = cfg_mvin_dest==1?sram_wr_en_delay:1'b0;
    acc_din     = cfg_mvin_dest==1?(cfg_mvin_is_quant_effective?dequant_data:{{(ACC_DATA_WIDTH-SPM_DATA_WIDTH){1'b0}},sram_din}):'0;
    acc_wr_mask = cfg_mvin_dest==1?(cfg_mvin_is_quant_effective?sram_wr_mask_delay:sram_wr_mask_bias):'0;   
    acc_wr_addr = cfg_mvin_dest==1?sram_wr_addr_delay[ACC_ADDR_WIDTH-1:0]:'0;
end
always_comb begin 
    if(rd_meta_row_first[rd_meta_rd_ptr] && rd_meta_cur_first_beat && m_axi_rvalid && m_axi_rready)
    sram_din_pre = m_axi_rdata>>(rd_meta_head_offset[rd_meta_rd_ptr]*8)   ;
    else
    sram_din_pre = m_axi_rdata;
end

always_ff @(posedge clk or negedge rst_n) begin        //read done signal generate 
    if(rst_n == 0)
    dma_mvin_resp_done <= 0;
    else if(rd_row_done_fire && rd_done_row_cnt == reg_read_row_num - 1)
    dma_mvin_resp_done <= 1;
    else
    dma_mvin_resp_done <= 0;    
end


//---------------------------------------------    
// read & write FSM
//--------------------------------------------- 
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)begin
    current_read_state  <= IDLE;
    current_write_state <= IDLE;
    current_raddr_state <= IDLE;
    current_waddr_state <= IDLE;
    end
    else begin
    current_read_state  <= next_read_state;
    current_write_state <= next_write_state;
    current_raddr_state <= next_raddr_state;
    current_waddr_state <= next_waddr_state;
    end
end
always_comb begin
    case (current_waddr_state)
        IDLE: begin 
            if(reg_dma_mvout_req_en[0])
            next_waddr_state = S_WRITE ;
            else 
            next_waddr_state = IDLE;
        end
        S_WRITE: begin
            if(((m_axi_awlen+1)>=aw_remain_beats) && m_axi_awready && m_axi_awvalid)
            next_waddr_state = S_WAIT;
            else if(awaddr_row_cnt == reg_write_row_num)
            next_waddr_state = IDLE;
            else
            next_waddr_state = S_WRITE ;
        end
        S_WAIT: begin
            if(awaddr_row_cnt == reg_write_row_num -1)
            next_waddr_state = IDLE ;
            else 
            next_waddr_state = S_WRITE ;
        end
        default:next_waddr_state=IDLE; 
    endcase
end
always_comb begin
    case (current_write_state)
        IDLE: begin 
            if(reg_dma_mvout_req_en[1])
            next_write_state = S_WRITE ;
            else 
            next_write_state = IDLE;
        end
        S_WRITE: begin
            if(awaddr_row_cnt == reg_write_row_num && w_outstanding_cnt == 0)
            next_write_state = IDLE ;
            else 
            next_write_state = S_WRITE ;
        end
        default:next_write_state=IDLE; 
    endcase
end
always_comb begin
    case (current_read_state)
        IDLE: begin 
            if(reg_dma_mvin_req_en[0])
            next_read_state = S_READ ;
            else 
            next_read_state = IDLE;
        end
        S_READ: begin
            if(dma_mvin_resp_done)
            next_read_state = IDLE ;
            else 
            next_read_state = S_READ ;
        end
        default:next_read_state=IDLE; 
    endcase
end
always_comb begin
    case (current_raddr_state)
        IDLE: begin 
            if(reg_dma_mvin_req_en[0])
            next_raddr_state = S_READ ;
            else 
            next_raddr_state = IDLE;
        end
        S_READ: begin
            if(((m_axi_arlen+1)>=ar_remain_beats) && m_axi_arready && m_axi_arvalid)
            next_raddr_state = S_WAIT;
            else if(araddr_row_cnt == reg_read_row_num)
            next_raddr_state = IDLE;
            else
            next_raddr_state = S_READ ;
        end
        S_WAIT: begin
            if(araddr_row_cnt == reg_read_row_num -1)
            next_raddr_state = IDLE ;
            else 
            next_raddr_state = S_READ ;
        end
        default:next_raddr_state=IDLE; 
    endcase
end


//---------------------------------------------    
// cfg info save reg
//--------------------------------------------- 
always_ff @(posedge clk or negedge rst_n)begin
    if(rst_n == 0)begin
    reg_dma_mvin_req_en  <= 0;
    reg_dma_mvout_req_en <= 0;
    end
    else begin
    reg_dma_mvin_req_en  <= {reg_dma_mvin_req_en[0] ,dma_mvin_req_en }   ;
    reg_dma_mvout_req_en <= {reg_dma_mvout_req_en[1],reg_dma_mvout_req_en[0],dma_mvout_req_en} ;    
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n == 0)begin
        reg_read_col_num           <= 0 ;
        reg_write_col_num          <= 0 ;
        reg_cfg_mvin_sram_stride   <= 0 ;
        reg_cfg_mvin_dram_stride   <= 0 ;
        reg_cfg_mvout_sram_stride  <= 0 ;
        reg_cfg_mvout_dram_stride  <= 0 ;
        reg_read_row_num           <= {RF_DATA_WIDTH/2{1'b1}} ;
        reg_write_row_num          <= {RF_DATA_WIDTH/2{1'b1}} ;
    end
    else if(reg_dma_mvin_req_en[0])begin
        if(mvin_contiguous_2d) begin
            reg_cfg_mvin_sram_stride   <=  0 ;
            reg_cfg_mvin_dram_stride   <=  0 ;
            reg_read_row_num           <=  1 ;
            reg_read_col_num           <=  (mvin_rows_real * mvin_cols_real)<<mvin_dram_data_size_log2 ;
        end
        else begin
            reg_cfg_mvin_sram_stride   <=  (cfg_mvin_sram_stride-mvin_col_num-1)<<mvin_sram_data_size_log2  ;
            reg_cfg_mvin_dram_stride   <=  cfg_mvin_dram_stride<<mvin_dram_data_size_log2 ;
            reg_read_row_num           <=  mvin_row_num+1 ; 
            reg_read_col_num           <=  (mvin_col_num + 1)<<mvin_dram_data_size_log2 ;
        end
    end
    else if(reg_dma_mvout_req_en[0])begin
        if(mvout_contiguous_2d) begin
            reg_cfg_mvout_sram_stride  <=  0 ;
            reg_cfg_mvout_dram_stride  <=  0 ;
            reg_write_row_num          <=  1 ;
            reg_write_col_num          <=  (mvout_rows_real * mvout_cols_real)<<mvout_dram_data_size_log2 ;
        end
        else begin
            reg_cfg_mvout_sram_stride  <=  (cfg_mvout_sram_stride-mvout_col_num-1)<<mvout_sram_data_size_log2 ; 
            reg_cfg_mvout_dram_stride  <=  cfg_mvout_dram_stride<<mvout_dram_data_size_log2;
            reg_write_row_num          <=  mvout_row_num+1        ;
            reg_write_col_num          <=  (mvout_col_num + 1)<<mvout_dram_data_size_log2 ;
        end
    end
    else begin
        reg_read_col_num           <= reg_read_col_num          ;
        reg_write_col_num          <= reg_write_col_num         ;
        reg_read_row_num           <= reg_read_row_num          ;
        reg_write_row_num          <= reg_write_row_num         ;
        reg_cfg_mvin_sram_stride   <= reg_cfg_mvin_sram_stride  ;
        reg_cfg_mvin_dram_stride   <= reg_cfg_mvin_dram_stride  ;
        reg_cfg_mvout_sram_stride  <= reg_cfg_mvout_sram_stride ;
        reg_cfg_mvout_dram_stride  <= reg_cfg_mvout_dram_stride ;
    end
end

always_ff @( posedge clk or negedge rst_n) begin : dma_mvin_busy_ff
        if(~rst_n)begin
            dma_mvin_busy       <= 1'b0 ;
        end
        else begin
            if(dma_mvin_req_en)begin
                dma_mvin_busy <= 1'b1;
            end
            else if(dma_mvin_resp_done)begin
                dma_mvin_busy <= 1'b0;
            end
            else begin
                dma_mvin_busy <= dma_mvin_busy;
            end
        end
    end

    always_ff @( posedge clk or negedge rst_n) begin : dma_mvout_busy_ff
        if(~rst_n)begin
            dma_mvout_busy      <= 1'b0 ;
        end
        else begin
            if(dma_mvout_req_en)begin
                dma_mvout_busy <= 1'b1;
            end
            else if(dma_mvout_resp_done)begin
                dma_mvout_busy <= 1'b0;
            end
            else begin
                dma_mvout_busy <= dma_mvout_busy;
            end
        end
    end




endmodule






`endif
