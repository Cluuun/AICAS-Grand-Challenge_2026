module accumulator #(
    parameter                 SPM_FPGA_SRAM    = 0                                      ,
    parameter                 AXI_DATA_WIDTH   = 128                                    ,
    parameter                 PE_WIDTH         = 16                                     ,
    parameter                 ACCU_DATA_WIDTH  = 32                                     ,  //INT32
    parameter                 SPM_DATA_WIDTH   = 8                                      ,  //INT8
    parameter                 SPM_ADDR_WIDTH   = 18                                     ,

    parameter                 ACCU_SIZE        = 1 << 18                                ,
    parameter                 RD_PORTS         = 2                                      ,
    parameter                 WR_PORTS         = 1                                      ,

    parameter                 RF_DATA_WIDTH    = 64                                     ,
    parameter integer         DISABLE_ACC_INT32_TO_INT8 = 1                            ,
    parameter integer         DISABLE_ACC_TO_SPM_PATH = 0                              ,

    parameter  signed [31:0]  ALPHA_1          = 16'd6656                               ,  //approximation for 0.1
    parameter  signed [31:0]  ALPHA_2          = 16'd13312                              ,  //approximation for 0.2
    parameter  signed [31:0]  ALPHA_3          = 16'd656                                ,  //approximation for 0.01

    localparam                ACCU_ADDR_WIDTH  = $clog2(ACCU_SIZE)                      ,  //20 bit address
    localparam                PE_IDX           = $clog2(PE_WIDTH)                       ,  //5 bit
    localparam                PE_DATA_SIZE     = $clog2(ACCU_DATA_WIDTH/8)              ,  //2 bit
    localparam                PE_DATA_SIZE_SPM = $clog2(SPM_DATA_WIDTH/8)               ,
    localparam                MVIN_MAX_NUM     = AXI_DATA_WIDTH / ACCU_DATA_WIDTH
)(
    input  logic                                        clk,
    input  logic                                        rst_n,

    //----------------------------------  
    // Accumulator control signals
    //----------------------------------
    input  logic                                        cfg_mvin_isbias,                // 0:mvin to normal ACC data, 1:mvin to ACC bias data

    input  logic               [1:0]                    cfg_compute_optype,             //operation type ,00 for GEMM,01 for CONV,10 for GEMV,11 for ADD
    input  logic                                        cfg_compute_accout_dest,        //write back destination for single instruction accumulate,0:SPM,1:ACC
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_compute_output_scale,       //scale of output data from PE in OS dataflow
    input  logic               [RF_DATA_WIDTH/4-1:0]    cfg_compute_output_scale_shift, //scale shift of output data from PE in OS dataflow
    // quantization from accu to spm

    input  logic               [$clog2(PE_WIDTH)-1:0]   sa_input_b_col_num,             // column number of input matrix B(OS) in systolic array

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
    input  logic               [RF_DATA_WIDTH/2-1:0]    matadd_output_addr,             //output matrix start address in SPM

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


localparam                 IDLE             = 2'b00;
localparam                 ACCADD           = 2'b01;
localparam                 MATADD           = 2'b01;
localparam                 DONE             = 2'b10;
localparam SRAM_ADDR_WIDTH = SPM_ADDR_WIDTH>ACCU_ADDR_WIDTH?SPM_ADDR_WIDTH:ACCU_ADDR_WIDTH;

localparam int             FP32_DMA_LANES     = AXI_DATA_WIDTH / ACCU_DATA_WIDTH;
localparam int             FP32_QUANT_LATENCY = 5;

// dma
logic signed                          [PE_WIDTH-1:0]       [ACCU_DATA_WIDTH-1:0]    dma_accu_rd_data_pre;
logic signed                          [FP32_DMA_LANES-1:0] [ACCU_DATA_WIDTH-1:0]    dma_accu_rd_data_fp32_lo;
logic signed                          [FP32_DMA_LANES-1:0] [ACCU_DATA_WIDTH-1:0]    fp32_dequant_scale;
logic signed                          [PE_WIDTH-1:0]       [ACCU_DATA_WIDTH-1:0]    fp32_scale_rd_data;
// logic signed [FP32_QUANT_LATENCY-1:0] [PE_WIDTH-1:0]       [ACCU_DATA_WIDTH-1:0]    dma_accu_rd_data_raw_dly;

logic                                                         fp32_quant_in_valid;
logic                                                         dma_accu_rd_fp32_valid;
logic                                                         fp32_scale_rd_en;
logic                                                         fp32_scale_rd_en_dly;
logic                                [ACCU_ADDR_WIDTH-1:0]    fp32_scale_rd_addr;
logic                                [ACCU_ADDR_WIDTH-1:0]    fp32_scale_base_addr;
logic signed                         [ACCU_DATA_WIDTH-1:0]    fp32_scale_immediate;
logic                                [RF_DATA_WIDTH/2-1:0]    fp32_scale_col_byte;
logic                                [RF_DATA_WIDTH/2:0]      fp32_scale_col_next;
logic                                [RF_DATA_WIDTH/2:0]      fp32_mvout_row_bytes;

//accumulator_scratchpad
logic                  [WR_PORTS-1:0]                         wr_en;
logic                  [WR_PORTS-1:0][ACCU_ADDR_WIDTH-1:0]    wr_addr;
logic    [PE_WIDTH-1:0][WR_PORTS-1:0]                         wr_mask;
logic    [PE_WIDTH-1:0][WR_PORTS-1:0][ACCU_DATA_WIDTH-1:0]    din;

logic                  [RD_PORTS-1:0]                         rd_en;
logic                  [RD_PORTS-1:0][ACCU_ADDR_WIDTH-1:0]    rd_addr;
logic    [PE_WIDTH-1:0][RD_PORTS-1:0][ACCU_DATA_WIDTH-1:0]    dout;

logic                                                         biaspsum_rd_en_dly;
logic                                                         dma_accu_rd_en_dly;
logic                                                         matadd_rd_en_1_dly;
logic                                                         matadd_rd_en_2_dly;

//32×32bit adder
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    adder_a;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    adder_b;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    adder_sum;

logic                               [ACCU_DATA_WIDTH+15:0]    product_2;
  

logic                                                         sa_sum_en;
logic    [PE_WIDTH-1:0]                                       sa_sum_mask;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    sa_sum_data;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    sa_sum_data_dly;

logic                                                         biaspsum_wr_en;
logic    [PE_WIDTH-1:0]                                       biaspsum_wr_mask;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    biaspsum_wr_data_pre_relu;
//sram:for acc & spm
logic                                                         sram_wr_en_pre;
logic                                [SRAM_ADDR_WIDTH-1:0]    sram_wr_addr_pre;
logic    [PE_WIDTH-1:0]                                       sram_wr_mask_pre;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    sram_wr_data_32_pre_relu;
//sa_noaccu
logic                                                         sa_wr_en_dly;
logic    [PE_WIDTH-1:0]                                       sa_wr_mask_dly;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    sa_wr_data_dly;

//sa_accu
logic                                                         sa_accu_wr_en;
logic                                [ACCU_ADDR_WIDTH-1:0]    sa_accu_wr_addr;
logic    [PE_WIDTH-1:0]                                       sa_accu_wr_mask;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    sa_accu_wr_data;

//accu_spm
logic                                                         quant_in_valid;
logic                                                         quant_out_valid;
logic [PE_WIDTH-1:0][SPM_DATA_WIDTH-1:0]                     accu_spm_wr_data_quant;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    sram_wr_data_32;
logic                                                         sram_wr_en_relu;
logic                                [SPM_ADDR_WIDTH-1:0]     sram_wr_addr_relu;
logic    [PE_WIDTH-1:0]                                       sram_wr_mask_relu;

//matadd
logic                                [RF_DATA_WIDTH/8  :0]    matadd_input_col_num_real;
logic                                [RF_DATA_WIDTH/8  :0]    matadd_input_row_num_real;
logic                                [RF_DATA_WIDTH/2  :0]    matadd_total_elems;         // total elements, at least 18-bit to avoid truncation

logic                                [1:0]                    matadd_status;
logic                                [RF_DATA_WIDTH/4-1:0]    matadd_count;
logic                                [RF_DATA_WIDTH/4-1:0]    matadd_count_dly;
logic    [PE_WIDTH-1:0]                                       matadd_wr_mask_pre;

logic                                                         matadd_rd_en_1;
logic                                                         matadd_rd_en_2;
logic                                [ACCU_ADDR_WIDTH-1:0]    matadd_rd_addr_1;
logic                                [ACCU_ADDR_WIDTH-1:0]    matadd_rd_addr_2;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    matadd_rd_data_1;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    matadd_rd_data_2;

logic                                [ACCU_ADDR_WIDTH-1:0]    matadd_output_addr_dly;
logic                                                         matadd_wr_en;
logic                                [ACCU_ADDR_WIDTH-1:0]    matadd_wr_addr;
logic    [PE_WIDTH-1:0]                                       matadd_wr_mask;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    matadd_wr_data_pre_relu;

//accumulate
logic                                                         biaspsum_rd_en;
logic                                [ACCU_ADDR_WIDTH-1:0]    biaspsum_rd_addr;
logic    [PE_WIDTH-1:0]              [ACCU_DATA_WIDTH-1:0]    biaspsum_rd_data;

logic                                [ACCU_ADDR_WIDTH-1:0]    biaspsum_wr_addr_acc;
logic                                [SPM_ADDR_WIDTH-1:0]     biaspsum_wr_addr_spm;

logic                                [RF_DATA_WIDTH/8-1:0]    gemm_count;
logic                                [RF_DATA_WIDTH/8-1:0]    conv_col_count;
logic                                [RF_DATA_WIDTH/8-1:0]    conv_row_count;
logic                                [RF_DATA_WIDTH/2-1:0]    gemm_addr_add_rd;
logic                                [RF_DATA_WIDTH/2-1:0]    conv_addr_add_rd;
logic                                [RF_DATA_WIDTH*3/8-1:0]  conv_addr_leap;                         
logic                                [RF_DATA_WIDTH/2-1:0]    gemm_addr_add_wr;
logic                                [RF_DATA_WIDTH/2-1:0]    conv_addr_add_wr;
logic                                [RF_DATA_WIDTH/2-1:0]    addr_add_rd;
logic                                [RF_DATA_WIDTH/2-1:0]    addr_add_wr;
logic                                [RF_DATA_WIDTH/2-1:0]    row_add_rd;
logic                                [PE_IDX - 1       :0]    col_add_rd;
logic                                [RF_DATA_WIDTH/2-1:0]    row_add_wr;
logic                                [PE_IDX - 1       :0]    col_add_wr;

// Accumulator to DMA Quantization
assign fp32_scale_base_addr = {cfg_mvout_output_scale_shift, cfg_mvout_output_scale};
assign fp32_scale_immediate = $signed({cfg_mvout_output_scale_shift, cfg_mvout_output_scale});
assign fp32_mvout_row_bytes = ({1'b0, mvout_col_num} + {{RF_DATA_WIDTH/2{1'b0}}, 1'b1}) << PE_DATA_SIZE;
assign fp32_scale_col_next  = {1'b0, fp32_scale_col_byte} + AXI_DATA_WIDTH/8;
assign fp32_scale_rd_en     = dma_accu_rd_en && (cfg_mvout_output_precision == 2'b11) && cfg_mvout_per_channel;
assign fp32_scale_rd_addr   = cfg_mvout_per_channel ?
                              (fp32_scale_base_addr + fp32_scale_col_byte[ACCU_ADDR_WIDTH-1:0]) :
                              fp32_scale_base_addr;

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        fp32_scale_col_byte <= '0;
    end else if(dma_mvout_resp_done) begin
        fp32_scale_col_byte <= '0;
    end else if(fp32_scale_rd_en) begin
        if(fp32_mvout_row_bytes == '0 || fp32_scale_col_next >= fp32_mvout_row_bytes) begin
            fp32_scale_col_byte <= '0;
        end else begin
            fp32_scale_col_byte <= fp32_scale_col_next[RF_DATA_WIDTH/2-1:0];
        end
    end
end

always_comb begin
    for(int idx = 0; idx < FP32_DMA_LANES; idx++) begin
        fp32_dequant_scale[idx] = cfg_mvout_per_channel ? fp32_scale_rd_data[idx] : fp32_scale_immediate;
    end
end

quantization_int32_to_fp32 #(
    .INPUT_NUMBER(FP32_DMA_LANES)
) u_quantization_int32_to_fp32(
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(fp32_quant_in_valid),
    .input_data(dma_accu_rd_data_pre[FP32_DMA_LANES-1:0]),
    .dequant_scale(fp32_dequant_scale),
    .out_valid(dma_accu_rd_fp32_valid),
    .output_data(dma_accu_rd_data_fp32_lo)
);

assign fp32_quant_in_valid = dma_accu_rd_en_dly && (cfg_mvout_output_precision == 2'b11);

assign dma_accu_rd_valid = (cfg_mvout_output_precision == 2'b11) ? dma_accu_rd_fp32_valid : dma_accu_rd_en_dly;

// always_ff @(posedge clk or negedge rst_n) begin
//     if(!rst_n) begin
//         dma_accu_rd_data_raw_dly <= '0;
//     end else if (fp32_quant_in_valid) begin
//         dma_accu_rd_data_raw_dly[0] <= dma_accu_rd_data_pre;
//         for (int stage = 1; stage < FP32_QUANT_LATENCY; stage++) begin
//             dma_accu_rd_data_raw_dly[stage] <= dma_accu_rd_data_raw_dly[stage-1];
//         end
//     end
// end

always_comb begin
    dma_accu_rd_data = dma_accu_rd_data_pre;

    if(cfg_mvout_output_precision == 2'b11) begin
        dma_accu_rd_data = '0;
        if(dma_accu_rd_fp32_valid) begin
            dma_accu_rd_data[FP32_DMA_LANES-1:0] = dma_accu_rd_data_fp32_lo;
            // for(int idx = FP32_DMA_LANES; idx < PE_WIDTH; idx++) begin
            //     dma_accu_rd_data[idx] = dma_accu_rd_data_raw_dly[FP32_QUANT_LATENCY-1][idx];
            // end
        end
    end
end

// Accumulator to SPM Quantization
generate
    if(DISABLE_ACC_INT32_TO_INT8 == 1) begin : gen_disable_acc_int32_to_int8
        assign quant_out_valid = quant_in_valid;
        always_comb begin
            for(int idx = 0; idx < PE_WIDTH; idx++) begin
                accu_spm_wr_data_quant[idx] = sram_wr_data_32[idx][SPM_DATA_WIDTH-1:0];
            end
        end
    end
    else begin : gen_enable_acc_int32_to_int8
        quantization_intX_to_intY #(
            .INPUT_DATA_WIDTH(ACCU_DATA_WIDTH),
            .OUTPUT_DATA_WIDTH(SPM_DATA_WIDTH),
            .SCALE_DATA_WIDTH(RF_DATA_WIDTH/4),
            .INPUT_NUMBER(PE_WIDTH)
        ) u_quantization_intX_to_intY_accu_spm(
            .clk(clk),
            .rst_n(rst_n),
            .in_valid(quant_in_valid),
            .out_valid(quant_out_valid),
            .unquantized_input(sram_wr_data_32),
            .quant_scale(cfg_compute_output_scale),
            .scale_shift(cfg_compute_output_scale_shift),
            .quantized_output(accu_spm_wr_data_quant)
        );
    end
endgenerate

assign accu_spm_wr_en   = DISABLE_ACC_TO_SPM_PATH ? 1'b0 : quant_out_valid;
assign accu_spm_wr_data = DISABLE_ACC_TO_SPM_PATH ? '0 : accu_spm_wr_data_quant;
assign quant_in_valid   = DISABLE_ACC_TO_SPM_PATH ? 1'b0 : (sram_wr_en_relu&&((~cfg_compute_accout_dest)||(matadd_busy)));
always_ff @( posedge clk ) begin 
    if(DISABLE_ACC_TO_SPM_PATH)begin
        accu_spm_wr_addr<='0;
        accu_spm_wr_mask<='0;
    end
    else if(cfg_compute_accout_dest && (~matadd_busy))begin
        accu_spm_wr_addr<='0;
        accu_spm_wr_mask<='0;
    end
    else begin
        accu_spm_wr_addr<=sram_wr_addr_relu;
        accu_spm_wr_mask<=sram_wr_mask_relu;
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        sa_wr_en_dly   <= 1'b0;
        sa_wr_mask_dly <= {PE_WIDTH{1'b1}};
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            sa_wr_data_dly[idx] <= {ACCU_DATA_WIDTH{1'b0}};
        end
    end else begin
        sa_wr_en_dly   <= sa_wr_en;
        sa_wr_mask_dly <= sa_wr_mask;
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            sa_wr_data_dly[idx] <= sa_wr_data[idx];
        end
    end
end


always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        biaspsum_wr_en              <= 1'b0;
        biaspsum_wr_mask            <= {PE_WIDTH{1'b0}};
        matadd_output_addr_dly      <= {ACCU_ADDR_WIDTH{1'b0}};
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            sa_sum_data_dly[idx]    <= {ACCU_DATA_WIDTH{1'b0}};
        end
    end else begin
        biaspsum_wr_en              <= sa_sum_en;
        biaspsum_wr_mask            <= sa_sum_mask;
        matadd_output_addr_dly      <= matadd_output_addr;
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            sa_sum_data_dly[idx]    <= sa_sum_data[idx];
        end
    end
end


always_comb begin  //sa to accu dataflow control
    biaspsum_rd_en          = cfg_accu_isaccu ? sa_wr_en                                       : 1'b0;
    sa_sum_en               = cfg_accu_isaccu ? sa_wr_en                                       : 1'b0;
    sa_sum_mask             = cfg_accu_isaccu ? sa_wr_mask                                     : {PE_WIDTH{1'b1}};
    for(int idx = 0; idx < PE_WIDTH; idx++) begin
        sa_sum_data[idx]    = cfg_accu_isaccu ? sa_wr_data[idx]                                : {ACCU_DATA_WIDTH{1'b0}};
    end
end


always_ff@(posedge clk or negedge rst_n) begin  //sram write control
    if(rst_n==0)begin
        sram_wr_en_pre           <='0;  
        sram_wr_addr_pre         <='0;    
        sram_wr_mask_pre         <='0;  
        sram_wr_data_32_pre_relu <='0; 
    end
    else begin
        if(matadd_busy) begin
            sram_wr_en_pre    <= matadd_wr_en;
            sram_wr_addr_pre  <= matadd_wr_addr;
            sram_wr_mask_pre  <= matadd_wr_mask;
            sram_wr_data_32_pre_relu <= matadd_wr_data_pre_relu;
         
        end else begin
            if(cfg_accu_isaccu) begin
                sram_wr_en_pre   <= biaspsum_wr_en;
                if(cfg_compute_accout_dest) begin
                    sram_wr_addr_pre <= biaspsum_wr_addr_acc;
                end else begin
                    sram_wr_addr_pre <= biaspsum_wr_addr_spm;
                end
                sram_wr_mask_pre <= biaspsum_wr_mask;
                sram_wr_data_32_pre_relu <= biaspsum_wr_data_pre_relu;
                
            end else begin
                sram_wr_en_pre    <= sa_wr_en_dly;
                if(cfg_compute_accout_dest) begin
                    sram_wr_addr_pre <= biaspsum_wr_addr_acc;
                end else begin
                    sram_wr_addr_pre <= biaspsum_wr_addr_spm;
                end
                sram_wr_mask_pre  <= sa_wr_mask_dly;
                sram_wr_data_32_pre_relu <= sa_wr_data_dly;
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n) begin  //gemm read and write address
    if(!rst_n) begin
        gemm_count       <= {RF_DATA_WIDTH/8{1'b0}};
        gemm_addr_add_rd <= {RF_DATA_WIDTH/2{1'b0}};
    end else begin
        if(!sa_wr_en || (sa_wr_en && (cfg_compute_optype != 2'b00))) begin
            gemm_count       <= gemm_count;
            gemm_addr_add_rd <= gemm_addr_add_rd;
        end else begin
            if(gemm_count == cfg_accu_biaspsum_height - 1'b1) begin
                gemm_count       <= {RF_DATA_WIDTH/8{1'b0}};
                gemm_addr_add_rd <= {RF_DATA_WIDTH/2{1'b0}};
            end else begin
                gemm_count       <= gemm_count + 1'b1;
                gemm_addr_add_rd <= gemm_addr_add_rd + cfg_accu_biaspsum_stride;
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        gemm_addr_add_wr <= {RF_DATA_WIDTH/2{1'b0}};
    end else begin
        gemm_addr_add_wr <= gemm_count * cfg_accu_output_stride;
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        conv_addr_leap<='0;
    else 
        conv_addr_leap<=(cfg_accu_biaspsum_stride - cfg_accu_biaspsum_width + 1) * (sa_input_b_col_num + 1);
end
always_ff @(posedge clk or negedge rst_n) begin  //conv read and write address
    if(!rst_n) begin
        conv_col_count   <= {RF_DATA_WIDTH/8{1'b0}};
        conv_row_count   <= {RF_DATA_WIDTH/8{1'b0}};
        conv_addr_add_rd <= {RF_DATA_WIDTH/2{1'b0}};
    end else begin
        if(!sa_wr_en || (sa_wr_en && (cfg_compute_optype != 2'b01))) begin
            conv_col_count   <= conv_col_count;
            conv_row_count   <= conv_row_count;
            conv_addr_add_rd <= conv_addr_add_rd;
        end else begin
            if(conv_col_count == cfg_accu_biaspsum_width - 1'b1) begin
                conv_col_count <= {RF_DATA_WIDTH/8{1'b0}};
                if(conv_row_count == cfg_accu_biaspsum_height - 1'b1) begin
                    conv_row_count   <= {RF_DATA_WIDTH/8{1'b0}};
                    conv_addr_add_rd <= {RF_DATA_WIDTH/2{1'b0}};
                end else begin
                    conv_row_count   <= conv_row_count + 1'b1;
                    //conv_addr_add_rd <= conv_addr_add_rd + (cfg_accu_biaspsum_stride - cfg_accu_biaspsum_width + 1) * (sa_input_b_col_num + 1);
                    conv_addr_add_rd <= conv_addr_add_rd + conv_addr_leap;
                end
            end else begin
                conv_col_count   <= conv_col_count + 1'b1;
                conv_addr_add_rd <= conv_addr_add_rd + sa_input_b_col_num + 1;
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        conv_addr_add_wr <= {RF_DATA_WIDTH/2{1'b0}};
    end else begin
        conv_addr_add_wr <= (sa_input_b_col_num + 1) * (cfg_accu_output_stride * conv_row_count + conv_col_count);
    end
end


always_comb begin  //biaspsum read and write address
    addr_add_rd = cfg_accu_isbias ? {RF_DATA_WIDTH/2{1'b0}} :
                  (cfg_compute_optype == 2'b00) ? gemm_addr_add_rd :
                  (cfg_compute_optype == 2'b01) ? conv_addr_add_rd :
                  {RF_DATA_WIDTH/2{1'b0}};
    addr_add_wr = (cfg_compute_optype == 2'b00) ? gemm_addr_add_wr : (cfg_compute_optype == 2'b01) ? conv_addr_add_wr : {RF_DATA_WIDTH/2{1'b0}};

    row_add_rd  = addr_add_rd >> PE_IDX;
    col_add_rd  = addr_add_rd - row_add_rd * PE_WIDTH;
    row_add_wr  = addr_add_wr >> PE_IDX;
    col_add_wr  = addr_add_wr - row_add_wr * PE_WIDTH;

    biaspsum_wr_addr_acc = '0;
    biaspsum_wr_addr_spm = '0;

    if(col_add_rd < PE_WIDTH - cfg_accu_biaspsum_addr[PE_DATA_SIZE +: PE_IDX]) begin
        biaspsum_rd_addr[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] = cfg_accu_biaspsum_addr[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] + row_add_rd;
        biaspsum_rd_addr[PE_DATA_SIZE +: PE_IDX]                      = cfg_accu_biaspsum_addr[PE_DATA_SIZE +: PE_IDX] + col_add_rd;
        biaspsum_rd_addr[PE_DATA_SIZE - 1 : 0]                        = cfg_accu_biaspsum_addr[PE_DATA_SIZE - 1 : 0];
    end else begin
        biaspsum_rd_addr[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] = cfg_accu_biaspsum_addr[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] + row_add_rd + 1;
        biaspsum_rd_addr[PE_DATA_SIZE +: PE_IDX]                      = cfg_accu_biaspsum_addr[PE_DATA_SIZE +: PE_IDX] + col_add_rd - PE_WIDTH;
        biaspsum_rd_addr[PE_DATA_SIZE - 1 : 0]                        = cfg_accu_biaspsum_addr[PE_DATA_SIZE - 1 : 0];
    end

    if(cfg_compute_accout_dest) begin
        if(col_add_wr < PE_WIDTH - cfg_accu_output_addr[PE_DATA_SIZE +: PE_IDX]) begin
            biaspsum_wr_addr_acc[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] = cfg_accu_output_addr[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] + row_add_wr;
            biaspsum_wr_addr_acc[PE_DATA_SIZE +: PE_IDX]                      = cfg_accu_output_addr[PE_DATA_SIZE +: PE_IDX] + col_add_wr;
            biaspsum_wr_addr_acc[PE_DATA_SIZE - 1 : 0]                        = cfg_accu_output_addr[PE_DATA_SIZE - 1 : 0];
        end else begin
            biaspsum_wr_addr_acc[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] = cfg_accu_output_addr[ACCU_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE] + row_add_wr + 1;
            biaspsum_wr_addr_acc[PE_DATA_SIZE +: PE_IDX]                      = cfg_accu_output_addr[PE_DATA_SIZE +: PE_IDX] + col_add_wr - PE_WIDTH;
            biaspsum_wr_addr_acc[PE_DATA_SIZE - 1 : 0]                        = cfg_accu_output_addr[PE_DATA_SIZE - 1 : 0];
        end
    end else begin
        if(col_add_wr < PE_WIDTH - cfg_accu_output_addr[PE_DATA_SIZE_SPM +: PE_IDX]) begin
            biaspsum_wr_addr_spm[SPM_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE_SPM] = cfg_accu_output_addr[SPM_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE_SPM] + row_add_wr;
            biaspsum_wr_addr_spm[PE_DATA_SIZE_SPM +: PE_IDX]                      = cfg_accu_output_addr[PE_DATA_SIZE_SPM +: PE_IDX] + col_add_wr;
        end else begin
            biaspsum_wr_addr_spm[SPM_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE_SPM] = cfg_accu_output_addr[SPM_ADDR_WIDTH - 1 : PE_IDX + PE_DATA_SIZE_SPM] + row_add_wr + 1;
            biaspsum_wr_addr_spm[PE_DATA_SIZE_SPM +: PE_IDX]                      = cfg_accu_output_addr[PE_DATA_SIZE_SPM +: PE_IDX] + col_add_wr - PE_WIDTH;
        end
    end
end


generate
    if(PE_DATA_SIZE_SPM > 0) begin
        always_comb begin
            if(!cfg_compute_accout_dest) begin
                biaspsum_wr_addr_spm[PE_DATA_SIZE_SPM - 1 : 0]                = cfg_accu_output_addr[PE_DATA_SIZE_SPM - 1 : 0];
            end
        end
    end
endgenerate


always_ff @(posedge clk or negedge rst_n) begin  //accadd_busy control
    if(!rst_n) begin
        accadd_busy      <= 1'b0;
    end else begin
        if(accadd_start) begin
            accadd_busy  <= 1'b1;
        end else if(accadd_done) begin
            accadd_busy  <= 1'b0;
        end else begin
            accadd_busy  <= accadd_busy;
        end
    end
end

always_comb begin
    if(cfg_compute_accout_dest) //acc
        accadd_done = accadd_busy&&sa_accu_wr_en && (~sram_wr_en_pre);
    else
        accadd_done = accadd_busy&&accu_spm_wr_en && (~sram_wr_en_relu);
end


always_comb begin  //sa write back to acc
    if(!cfg_compute_accout_dest) begin
        sa_accu_wr_en   = '0;
        sa_accu_wr_addr = '0;
        sa_accu_wr_mask = '0;
        sa_accu_wr_data = '0; 
    end else begin
        sa_accu_wr_en   = sram_wr_en_relu;
        sa_accu_wr_addr = sram_wr_addr_relu;
        sa_accu_wr_mask = sram_wr_mask_relu;
        sa_accu_wr_data = sram_wr_data_32;
    end
end



//matadd

always_comb begin
    matadd_input_col_num_real = matadd_input_col_num + 1;
    matadd_input_row_num_real = matadd_input_row_num + 1;
    // Explicit calculation of total elements to avoid implicit truncation
    matadd_total_elems = ({1'b0, matadd_input_col_num_real}) * ({1'b0, matadd_input_row_num_real});
end

always_ff @(posedge clk or negedge rst_n) begin  //write back
    if(!rst_n) begin
        matadd_wr_en     <= 1'b0;
        matadd_wr_addr   <= {ACCU_ADDR_WIDTH{1'b0}};
        matadd_wr_mask   <= {PE_WIDTH{1'b1}};
        matadd_count_dly <= {RF_DATA_WIDTH/4{1'b0}};
    end else begin
        matadd_wr_en     <= matadd_rd_en_1;
        matadd_wr_addr[ACCU_ADDR_WIDTH - 1 : PE_DATA_SIZE_SPM + PE_IDX] <= matadd_output_addr_dly[ACCU_ADDR_WIDTH - 1 : PE_DATA_SIZE_SPM + PE_IDX] + matadd_count_dly;
        matadd_wr_addr[PE_DATA_SIZE_SPM + PE_IDX - 1 : 0]               <= matadd_output_addr_dly[PE_DATA_SIZE_SPM + PE_IDX - 1 : 0];
        matadd_wr_mask   <= matadd_wr_mask_pre;
        matadd_count_dly <= matadd_count;
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        matadd_status      <= IDLE;
        matadd_count       <= {RF_DATA_WIDTH/4{1'b0}};
        matadd_rd_en_1     <= 1'b0;
        matadd_rd_en_2     <= 1'b0;
        matadd_rd_addr_1   <= {ACCU_ADDR_WIDTH{1'b0}};
        matadd_rd_addr_2   <= {ACCU_ADDR_WIDTH{1'b0}};
        matadd_wr_mask_pre <= {PE_WIDTH{1'b1}};
    end else begin
        case(matadd_status)
        IDLE:begin
            if(!matadd_req_en) begin
                matadd_status      <= IDLE;
                matadd_count       <= {RF_DATA_WIDTH/4{1'b0}};


                matadd_rd_en_1     <= 1'b0;
                matadd_rd_en_2     <= 1'b0;
                matadd_rd_addr_1   <= {ACCU_ADDR_WIDTH{1'b0}};
                matadd_rd_addr_2   <= {ACCU_ADDR_WIDTH{1'b0}};
                matadd_wr_mask_pre <= {PE_WIDTH{1'b1}};
            end else begin
                matadd_status    <= MATADD;
            end
        end

        MATADD:begin
            matadd_rd_en_1    <= 1'b1;
            matadd_rd_en_2    <= 1'b1;

            matadd_rd_addr_1[ACCU_ADDR_WIDTH - 1 : PE_DATA_SIZE + PE_IDX] <= matadd_input_a_addr[ACCU_ADDR_WIDTH - 1 : PE_DATA_SIZE + PE_IDX] + matadd_count;
            matadd_rd_addr_1[PE_DATA_SIZE + PE_IDX - 1 : 0]               <= matadd_input_a_addr[PE_DATA_SIZE + PE_IDX - 1 : 0];
            matadd_rd_addr_2[ACCU_ADDR_WIDTH - 1 : PE_DATA_SIZE + PE_IDX] <= matadd_input_b_addr[ACCU_ADDR_WIDTH - 1 : PE_DATA_SIZE + PE_IDX] + matadd_count;
            matadd_rd_addr_2[PE_DATA_SIZE + PE_IDX - 1 : 0]               <= matadd_input_b_addr[PE_DATA_SIZE + PE_IDX - 1 : 0];

            // Use explicit total_elems to avoid boundary truncation issues at large dimensions
            if(({1'b0, matadd_count} + 1'b1) * PE_WIDTH >= matadd_total_elems) begin
                matadd_wr_mask_pre <= {PE_WIDTH{1'b1}} >> (({1'b0, matadd_count} + 1'b1) * PE_WIDTH - matadd_total_elems);

                matadd_status      <= DONE;
                matadd_count       <= {RF_DATA_WIDTH/4{1'b0}};
            end else begin
                matadd_wr_mask_pre <= {PE_WIDTH{1'b1}};
                matadd_count       <= matadd_count + 1'b1;
            end
        end

        DONE:begin
            matadd_rd_en_1   <= 1'b0;
            matadd_rd_en_2   <= 1'b0;

            matadd_status    <= IDLE;
        end

        default: begin
            matadd_status    <= IDLE;
        end
        endcase
    end
end

always_ff @(posedge clk or negedge rst_n) begin  //matadd_busy control
    if(!rst_n) begin
        matadd_busy      <= 1'b0;
    end else begin
        if(matadd_req_en) begin
            matadd_busy  <= 1'b1;
        end else if(matadd_comp_done) begin
            matadd_busy  <= 1'b0;
        end else begin
            matadd_busy  <= matadd_busy;
        end
    end
end
assign matadd_comp_done = matadd_busy && accu_spm_wr_en && (~sram_wr_en_relu);
//adder

always_comb begin
    for(int idx = 0; idx < PE_WIDTH; idx++) begin
        adder_a[idx]                   = matadd_busy ? matadd_rd_data_1[idx]  : biaspsum_rd_data[idx];
        adder_b[idx]                   = matadd_busy ? matadd_rd_data_2[idx]  : sa_sum_data_dly[idx];
        biaspsum_wr_data_pre_relu[idx] = matadd_busy ? {ACCU_DATA_WIDTH{1'b0}}: adder_sum[idx];
        matadd_wr_data_pre_relu[idx]   = matadd_busy ? adder_sum[idx]         : {ACCU_DATA_WIDTH{1'b0}};
    end
end

generate
    genvar idx_adder;
    for(idx_adder = 0; idx_adder < PE_WIDTH; idx_adder++) begin : gen_adder
        assign adder_sum[idx_adder]=$signed(adder_a[idx_adder])+$signed(adder_b[idx_adder]);
    end
endgenerate


//relu

always_comb begin 
    if(cfg_accu_relu && accadd_busy) begin
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            case(cfg_accu_relu_type)
                3'b010:product_2 = $signed(sram_wr_data_32_pre_relu[idx])*ALPHA_1;
                3'b011:product_2 = $signed(sram_wr_data_32_pre_relu[idx])*ALPHA_2;
                3'b100:product_2 = $signed(sram_wr_data_32_pre_relu[idx])*ALPHA_3;
                default:product_2 = '0;
            endcase
        end
    end
    else begin
        product_2 = '0;
    end
end
always_ff@(posedge clk or negedge rst_n) begin
    if(rst_n==0)begin
        sram_wr_en_relu  <='0;
        sram_wr_addr_relu<='0;
        sram_wr_mask_relu<='0;
        sram_wr_data_32  <='0;
    end
    else begin
        sram_wr_en_relu  <=sram_wr_en_pre;
        sram_wr_addr_relu<=sram_wr_addr_pre;
        sram_wr_mask_relu<=sram_wr_mask_pre;
        if(cfg_accu_relu && accadd_busy) begin
            for(int idx = 0; idx < PE_WIDTH; idx++) begin
                case(cfg_accu_relu_type)
                3'b000: sram_wr_data_32[idx] <= ($signed(sram_wr_data_32_pre_relu[idx]) > 0) ? sram_wr_data_32_pre_relu[idx] : {ACCU_DATA_WIDTH{1'b0}};
                3'b001: begin
                    if($signed(sram_wr_data_32_pre_relu[idx]) > 'd6) sram_wr_data_32[idx] <= 'd6;
                    else if($signed(sram_wr_data_32_pre_relu[idx]) > 0) sram_wr_data_32[idx] <= sram_wr_data_32_pre_relu[idx];
                    else sram_wr_data_32[idx] <= {ACCU_DATA_WIDTH{1'b0}};
                end
                3'b010,3'b011,3'b100: begin
                    sram_wr_data_32[idx] <= ($signed(sram_wr_data_32_pre_relu[idx]) > 0) ? sram_wr_data_32_pre_relu[idx] : (product_2 >>> 16);
                end
                default: sram_wr_data_32[idx] <= sram_wr_data_32_pre_relu[idx];
                endcase
            end
        end else begin
            sram_wr_data_32 <= sram_wr_data_32_pre_relu;
        end
    end
end

//sram

always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        biaspsum_rd_en_dly <= 1'b0;
        dma_accu_rd_en_dly <= 1'b0;
        fp32_scale_rd_en_dly <= 1'b0;
        matadd_rd_en_1_dly <= 1'b0;
        matadd_rd_en_2_dly <= 1'b0;
    end else begin
        biaspsum_rd_en_dly <= biaspsum_rd_en;
        dma_accu_rd_en_dly <= dma_accu_rd_en;
        fp32_scale_rd_en_dly <= fp32_scale_rd_en;
        matadd_rd_en_1_dly <= matadd_rd_en_1;
        matadd_rd_en_2_dly <= matadd_rd_en_2;
    end
end


always_comb begin  // read ports control
    case({biaspsum_rd_en, dma_accu_rd_en, matadd_rd_en_1, matadd_rd_en_2})
    4'b1000:begin
        rd_en[1]         = biaspsum_rd_en;
        rd_en[0]         = 1'b0;
        rd_addr[1]       = biaspsum_rd_addr;
        rd_addr[0]       = {ACCU_ADDR_WIDTH{1'b0}};
    end
    4'b0100:begin
        rd_en[0]         = dma_accu_rd_en;
        rd_en[1]         = fp32_scale_rd_en;
        rd_addr[0]       = dma_accu_rd_addr;
        rd_addr[1]       = fp32_scale_rd_addr;
    end
    4'b0011:begin
        rd_en[0]         = matadd_rd_en_1;
        rd_en[1]         = matadd_rd_en_2;
        rd_addr[0]       = matadd_rd_addr_1;
        rd_addr[1]       = matadd_rd_addr_2;
    end
    default:begin
        rd_en[0]         = 1'b0;
        rd_en[1]         = 1'b0;
        rd_addr[0]       = {ACCU_ADDR_WIDTH{1'b0}};
        rd_addr[1]       = {ACCU_ADDR_WIDTH{1'b0}};
    end
    endcase

    case({biaspsum_rd_en_dly, dma_accu_rd_en_dly, matadd_rd_en_1_dly, matadd_rd_en_2_dly})
    4'b1000:begin
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            biaspsum_rd_data[idx]    = dout[idx][1];
            dma_accu_rd_data_pre[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            fp32_scale_rd_data[idx]   = {ACCU_DATA_WIDTH{1'b0}};
            matadd_rd_data_1[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            matadd_rd_data_2[idx]    = {ACCU_DATA_WIDTH{1'b0}};
        end
    end
    4'b0100:begin
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            biaspsum_rd_data[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            dma_accu_rd_data_pre[idx]    = dout[idx][0];
            fp32_scale_rd_data[idx]   = fp32_scale_rd_en_dly ? dout[idx][1] : {ACCU_DATA_WIDTH{1'b0}};
            matadd_rd_data_1[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            matadd_rd_data_2[idx]    = {ACCU_DATA_WIDTH{1'b0}};
        end
    end
    4'b0011:begin
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            biaspsum_rd_data[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            dma_accu_rd_data_pre[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            fp32_scale_rd_data[idx]   = {ACCU_DATA_WIDTH{1'b0}};
            matadd_rd_data_1[idx]    = dout[idx][0];
            matadd_rd_data_2[idx]    = dout[idx][1];
        end
    end
    default:begin
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            biaspsum_rd_data[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            dma_accu_rd_data_pre[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            fp32_scale_rd_data[idx]   = {ACCU_DATA_WIDTH{1'b0}};
            matadd_rd_data_1[idx]    = {ACCU_DATA_WIDTH{1'b0}};
            matadd_rd_data_2[idx]    = {ACCU_DATA_WIDTH{1'b0}};
        end
    end
    endcase
end


always_comb begin  // write ports control
    case({sa_accu_wr_en, dma_accu_wr_en})
    2'b10:begin
        wr_en[0]   = sa_accu_wr_en;
        wr_addr[0] = sa_accu_wr_addr;
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            wr_mask[idx][0] = sa_accu_wr_mask[idx];
            din[idx][0]     = sa_accu_wr_data[idx];
        end
    end
    2'b01:begin
        wr_en[0]   = dma_accu_wr_en;
        wr_addr[0] = dma_accu_wr_addr;
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            wr_mask[idx][0] = dma_accu_wr_mask[idx];
            din[idx][0]     = dma_accu_wr_data[idx];
        end
    end
    default:begin
        wr_en[0]   = 1'b0;
        wr_addr[0] = {ACCU_ADDR_WIDTH{1'b0}};
        for(int idx = 0; idx < PE_WIDTH; idx++) begin
            wr_mask[idx][0] = 1'b1;
            din[idx][0]     = {ACCU_DATA_WIDTH{1'b0}};
        end
    end
    endcase
end

/*
scratchpad #(
    .PE_WIDTH(PE_WIDTH),
    .PE_DATA_WIDTH(ACCU_DATA_WIDTH),
    .SPM_SIZE(ACCU_SIZE),
    .RD_PORTS(RD_PORTS),
    .WR_PORTS(WR_PORTS)
) u_scratchpad_accumulator(
    .clk        (clk),
    .rst_n      (rst_n),

    .wr_en      (wr_en),
    .wr_mask    (wr_mask),
    .wr_addr    (wr_addr),
    .din        (din),

    .rd_en      (rd_en),
    .rd_addr    (rd_addr),
    .dout       (dout)
);
*/
generate
    if (SPM_FPGA_SRAM == 1) begin : gen_fpga_sram
        scratchpad #(
            .SPM_FPGA_SRAM  ( 1               ),
            .PE_WIDTH       ( PE_WIDTH        ),
            .PE_DATA_WIDTH  ( ACCU_DATA_WIDTH ),
            .SPM_SIZE       ( ACCU_SIZE       ),
            .RD_PORTS       ( RD_PORTS        ),
            .WR_PORTS       ( WR_PORTS        )
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
    end
    else begin : gen_asic_sram
        `ifdef VCS_SRAM_ENABLE
            scratchpad #(
                .SPM_FPGA_SRAM  ( 0               ),
                .PE_WIDTH       ( PE_WIDTH        ),
                .PE_DATA_WIDTH  ( ACCU_DATA_WIDTH ),
                .SPM_SIZE       ( ACCU_SIZE       ),
                .RD_PORTS       ( RD_PORTS        ),
                .WR_PORTS       ( WR_PORTS        )
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
                .PE_WIDTH       ( PE_WIDTH        ),
                .PE_DATA_WIDTH  ( ACCU_DATA_WIDTH ),
                .SPM_SIZE       ( ACCU_SIZE       ),
                .RD_PORTS       ( RD_PORTS        ),
                .WR_PORTS       ( WR_PORTS        )
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
    end
endgenerate
endmodule

// module leakyrelu#(
//     parameter INPUT_NUMBER = 32,
//     parameter INPUT_DATA_WIDTH =32,
// )(
//     input  logic signed [INPUT_NUMBER-1:0][INPUT_DATA_WIDTH-1:0] data_in,
//     input  logic        [2:0]                                    cfg_type,
//     output logic signed [INPUT_NUMBER-1:0][INPUT_DATA_WIDTH-1:0] data_out
// );

// endmodule
