module sfu 
    import npu_config_pkg::*;
#(
    parameter integer DISABLE_SOFTMAX     = npu_config_pkg::DISABLE_SOFTMAX,
    parameter integer DISABLE_GELU        = npu_config_pkg::DISABLE_GELU,
    parameter integer DISABLE_LAYERNORM   = npu_config_pkg::DISABLE_LAYERNORM,
    parameter integer DISABLE_RESAMPLE    = npu_config_pkg::DISABLE_RESAMPLE,
    parameter integer DISABLE_TRANSPOSE   = npu_config_pkg::DISABLE_TRANSPOSE,
    parameter integer GELU_NUM            = npu_config_pkg::GELU_NUM,
    localparam ADDR_WIDTH                 = $clog2(SPM_SIZE)
) (
    input  logic    clk     ,
    input  logic    rst_n   ,

    //------------------------------------          
    // SFU Control Signals          
    //------------------------------------          

    input  logic    [5:0]                                   cfg_sfu_op                  ,   // operation of SFU
    input  logic    [1:0]                                   cfg_sfu_int_type            ,   // datatype range of SFU input data, 00 for int8, 01 for int16, 10 for int32, 11 for int64  
    input  logic                                            cfg_sfu_is_quant            ,   // is / not quant process
    input  logic                                            cfg_trans_out_ispad_row     ,   //for transpose,output row is/not need padding 0 to align to 32
    input  logic                                            cfg_trans_out_ispad_col     ,   //for transpose,output col is/not need padding 0 to align to 32
    input  logic    [RF_DATA_WIDTH/4-1:0]                   cfg_sfu_output_zeropoint    ,  // zero point of SFU output data
    input  logic    [RF_DATA_WIDTH/2-1:0]                   cfg_sfu_input_zeropoint     ,  // zero point of SFU input data
    input  logic    [RF_DATA_WIDTH/4-1:0]                   cfg_sfu_input_scale         ,   // scale of SFU input data 
    input  logic    [RF_DATA_WIDTH/4-1:0]                   cfg_sfu_output_scale        ,   // scale of SFU output data 
    input  logic    [RF_DATA_WIDTH/4-1:0]                   cfg_sfu_input_scale_shift   ,   // scale shift of SFU input data 
    input  logic    [RF_DATA_WIDTH/4-1:0]                   cfg_sfu_output_scale_shift  ,   // scale shift of SFU output data 

    input  logic    [RF_DATA_WIDTH/2-1:0]                   sfu_input_sram_addr         ,   // address for input data in SPM   
    input  logic    [RF_DATA_WIDTH/4-1:0]                   sfu_input_col_num           ,   // width of input matrix data in SPM ,real col=col num+1 
    input logic     [RF_DATA_WIDTH/4-1:0]                   sfu_input_row_num           ,   // height of input matrix data in SPM ,real row =row num +1

    input  logic    [RF_DATA_WIDTH/2-1:0]                   sfu_output_spm_addr         ,   // address for output data in SPM
    //input  logic                                            sfu_input_source            ,   // input data source ,0 for SPM ,1 for ACC
    
    input  logic                                            sfu_req_en                  ,
    output logic                                            sfu_busy                    ,
    output logic                                            sfu_comp_done               ,
    //------------------------------------
    // ScratchPad 
    //------------------------------------
    
    output logic                                            sfu_spm_rd_en               ,   // SPM read enable
    output logic    [ADDR_WIDTH-1:0]                        sfu_spm_rd_addr             ,   // SPM read address
    input  logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       sfu_spm_rd_data_in          ,   // read data from SPM

    output logic                                            sfu_spm_wr_en               ,   // SPM write enable
    output logic    [ADDR_WIDTH-1:0]                        sfu_spm_wr_addr             ,   // SPM write address
    output logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       sfu_spm_wr_data_out         ,   // write data into SPM
    output logic    [PE_WIDTH-1:0]                          sfu_spm_wr_mask                 // SPM write mask

);
    localparam  PE_WIDTH_CLOG2      = $clog2(PE_WIDTH);          
    localparam  MAX_TOTAL_WIDTH     = SFU_MAX_INPUT_LENGTH*SFU_MAX_INPUT_LENGTH;   // max width*height of input matrix
    localparam  SPM_ADDR_INCR       = PE_WIDTH*PE_DATA_WIDTH/8;
    localparam  RESAMPLE_IDX_WIDTH  = $clog2(INPUT_WIDTH_MAX);
    localparam  TRANS_COUNT_WIDTH   = $clog2(TRANSPOSE_MAX_LENGTH/PE_WIDTH);
    localparam  PE_DATA_SIZE        = $clog2(PE_DATA_WIDTH/8);
    localparam  VECTOR_WIDTH_CLOG2  = $clog2(SFU_MAX_INPUT_LENGTH/PE_WIDTH); 
    localparam  LAYERNORM_NUM_WIDTH = $clog2(SFU_MAX_INPUT_LENGTH);
    localparam  SOFTMAX_NUM_WIDTH   = $clog2(SFU_MAX_INPUT_LENGTH/PE_WIDTH)+1;
    localparam  TRANS_DIM_WIDTH     = $clog2(TRANSPOSE_MAX_LENGTH);
    localparam  GELU_NUM_CLOG2      = GELU_NUM==1?1:$clog2(GELU_NUM);

    logic  [RF_DATA_WIDTH/4:0]                             input_real_col               ;
    logic  [RF_DATA_WIDTH/4:0]                             input_real_row               ;
    logic  [RF_DATA_WIDTH/2-1:0]                           input_len                    ;
    logic  [RF_DATA_WIDTH/2:0]                             input_real_len               ;              
    logic  [RF_DATA_WIDTH/2-PE_WIDTH_CLOG2:0]              input_len_spm                ;
    logic  [RF_DATA_WIDTH/2-GELU_NUM_CLOG2:0]              input_len_gelu               ;
    logic  [RF_DATA_WIDTH/4-PE_WIDTH_CLOG2:0]              input_col_spm                ;
    logic  [PE_WIDTH-1:0]                                  prepared_mask                ;
    // softmax
    logic  [SOFTMAX_NUM_WIDTH-1:0]                         softmax_in_num               ;
    logic  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]               softmax_in_data              ;
    logic                                                  softmax_in_valid             ;
    logic  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]               softmax_out_data             ;
    logic                                                  softmax_out_valid            ;
    logic                                                  softmax_quant_valid          ;
    logic                                                  softmax_spm_rd_en            ;
    
    // gelu
    logic  [GELU_NUM-1:0][FX_DATA_WIDTH-1:0]                gelu_din                    ;
    logic                                                   gelu_start                  ;
    logic  [GELU_NUM-1:0][FX_DATA_WIDTH-1:0]                gelu_dout                   ;
    logic  [GELU_NUM-1:0][FX_DATA_WIDTH-1:0]                gelu_dout_raw               ;
    logic                                                   gelu_quant_valid            ;
    logic  [GELU_NUM-1:0]                                   gelu_busy                   ;
    logic  [GELU_NUM-1:0]                                   gelu_done_raw               ;
    logic  [GELU_NUM-1:0]                                   gelu_done                   ;
    logic                                                   gelu_spm_rd_en              ;

    logic signed  [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]         softmax_quant_data_out       ;
    //logic signed  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]         softmax_quant_data_in        ;
    //logic signed  [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]         softmax_dequant_data_in      ;
    //logic signed  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]         softmax_dequant_data_out     ;
    logic signed  [GELU_NUM-1:0][PE_DATA_WIDTH-1:0]         gelu_quant_data_out          ;
    //logic signed  [GELU_NUM-1:0][FX_DATA_WIDTH-1:0]         gelu_quant_data_in           ;
    //logic signed  [GELU_NUM-1:0][PE_DATA_WIDTH-1:0]         gelu_dequant_data_in         ;
    //logic signed  [GELU_NUM-1:0][FX_DATA_WIDTH-1:0]         gelu_dequant_data_out        ;
    logic signed  [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]         layernorm_quant_data_out     ;
    //logic signed  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]         layernorm_quant_data_in      ;
    //logic signed  [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]         layernorm_dequant_data_in    ;
    //logic signed  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]         layernorm_dequant_data_out   ;
    //transpose
    logic  [PE_WIDTH*PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       trans_in_data                ;
    logic                                                   trans_in_valid               ;
    logic                                                   trans_in_ready               ;
    logic  [PE_WIDTH*PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       trans_out_data               ;
    logic  [PE_WIDTH*PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       trans_out_data_save          ;
    logic                                                   trans_out_valid              ; 
    logic                                                   trans_out_ready              ; 
    logic   [ADDR_WIDTH-1:0]                                trans_spm_rd_addr            ;
    logic   [ADDR_WIDTH-1:0]                                trans_spm_wr_addr            ;
    logic   [PE_WIDTH-1:0]                                  trans_spm_wr_mask            ;
    logic                                                   trans_spm_rd_en              ;
    logic                                                   trans_spm_wr_en              ;
    logic                                                   trans_out_flag               ;
    logic [TRANS_COUNT_WIDTH-1:0]                           trans_in_real_depth0         ;
    logic [TRANS_COUNT_WIDTH-1:0]                           trans_in_real_depth1         ; 
    logic [TRANS_COUNT_WIDTH-1:0]                           trans_in_row_count           ;
    logic [TRANS_COUNT_WIDTH-1:0]                           trans_in_col_count           ;
    logic [TRANS_COUNT_WIDTH-1:0]                           trans_out_row_count          ;
    logic [TRANS_COUNT_WIDTH-1:0]                           trans_out_col_count          ;
    logic [TRANS_COUNT_WIDTH-1:0]                           out_row_count                ;
    logic [TRANS_COUNT_WIDTH-1:0]                           out_col_count                ;
    logic [$clog2(PE_WIDTH):0]                              trans_spm_rd_count           ;
    logic [$clog2(PE_WIDTH):0]                              trans_spm_wr_count           ;   
    logic                                                   trans_padding_row            ; //in one cycle,transpose output in row need padding 
    logic                                                   trans_padding_col            ;
    logic                                                   padding_mask                 ; //when padding is 1,else 0
    logic [RF_DATA_WIDTH/4:0]                               trans_wr_stride              ;
    // layernorm                                    
    logic  [LAYERNORM_NUM_WIDTH:0]                          layernorm_in_num            ;
    logic  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]                norm_din                    ;
    logic                                                   norm_in_valid               ;
    logic  [PE_WIDTH-1:0][FX_DATA_WIDTH-1:0]                norm_dout                   ;
    logic                                                   norm_out_valid              ;
    logic                                                   layernorm_quant_valid       ;
    
    logic                                                   norm_spm_rd_en              ;

    // resample
    logic   [1:0]                                           resample_type               ; 
    logic                                                   resample_op                 ;  
    logic                                                   resample_input_en           ;
    logic   [ADDR_WIDTH-1:0]                                resample_input_addr         ;
    logic   [ADDR_WIDTH-1:0]                                resample_output_addr        ;
    logic   [RESAMPLE_IDX_WIDTH-1:0]                        resample_input_row_num      ;
    logic   [RESAMPLE_IDX_WIDTH-1:0]                        resample_input_col_num      ;
    logic                                                   resample_busy               ;
    logic                                                   resample_done               ;
    logic                                                   resample_spm_rd_en          ;
    logic   [ADDR_WIDTH-1:0]                                resample_spm_rd_addr        ;
    logic   [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]               resample_spm_rd_data_in     ;
    logic                                                   resample_spm_wr_en          ;
    logic   [ADDR_WIDTH-1:0]                                resample_spm_wr_addr        ;
    logic   [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]               resample_spm_wr_data_out    ;
    logic   [PE_WIDTH-1:0]                                  resample_spm_wr_mask        ;

    logic                                                   sfu_spm_rd_en_dly           ;
    logic                                                   sfu_spm_rd_en_dly2          ;
    logic                                                   sfu_req_en_dly              ;

    logic   [RF_DATA_WIDTH/2-PE_WIDTH_CLOG2:0]              sfu_wr_count                ;
    logic   [RF_DATA_WIDTH/4-1:0]                           sfu_wr_row_count            ;
    logic   [RF_DATA_WIDTH/2-PE_WIDTH_CLOG2:0]              sfu_rd_count                ;
    logic   [RF_DATA_WIDTH/2-PE_WIDTH_CLOG2:0]              sfu_rd_count_dly            ;
    logic   [ADDR_WIDTH-1:0]                                spm_rd_addr                 ;
    logic   [ADDR_WIDTH-1:0]                                spm_wr_addr                 ;
    logic   [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]               rd_data_masked              ; 
    


 
    typedef enum logic[2:0] {
        IDLE                = 3'd0,
        SOFTMAX             = 3'd1,
        GELU                = 3'd2,
        LAYERNORM           = 3'd3,
        RESAMPLE            = 3'd4,
        TRANSPOSE           = 3'd5
    }sfu_state_e;
     
    sfu_state_e                                             sfu_current_state           ;
    sfu_state_e                                             sfu_next_state              ;    

    always_ff @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            input_real_len <= '0;
            input_real_col <= '0; 
            input_real_row <= '0; 
            trans_wr_stride<= '0;
        end else if(sfu_req_en)begin
            input_real_len <= sfu_input_col_num * sfu_input_row_num + sfu_input_col_num + sfu_input_row_num +1;
            input_real_col <= sfu_input_col_num +1;
            input_real_row <= sfu_input_row_num +1;
            trans_wr_stride<= cfg_trans_out_ispad_col ? ((sfu_input_row_num + PE_WIDTH ) & ~(PE_WIDTH - 1)) : (sfu_input_row_num + 1);
        end
        else begin
            input_real_len <=input_real_len;
            input_real_col <=input_real_col;
            input_real_row <=input_real_row;
            trans_wr_stride<=trans_wr_stride;
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n)begin
            input_len_spm <='0; 
            input_len_gelu<='0; 
            input_col_spm <='0; 
            prepared_mask <='0;
        end else if(sfu_req_en_dly)begin
            input_len_spm   <= input_real_len[RF_DATA_WIDTH/2:PE_WIDTH_CLOG2]+(|input_real_len[PE_WIDTH_CLOG2-1:0]);
            input_len_gelu  <= input_real_len[RF_DATA_WIDTH/2:GELU_NUM_CLOG2]+(|input_real_len[GELU_NUM_CLOG2-1:0]);
            input_col_spm   <= input_real_col[RF_DATA_WIDTH/4:PE_WIDTH_CLOG2]+(|input_real_col[PE_WIDTH_CLOG2-1:0]);
            case (sfu_current_state)
            TRANSPOSE:begin
                prepared_mask    <= (|input_real_row[PE_WIDTH_CLOG2-1:0])?~({(PE_WIDTH){1'b1}}<<(input_real_row[PE_WIDTH_CLOG2-1:0])):{(PE_WIDTH){1'b1}};
            end
            SOFTMAX,LAYERNORM:begin
                prepared_mask    <= (|input_real_col[PE_WIDTH_CLOG2-1:0])?~({(PE_WIDTH){1'b1}}<<(input_real_col[PE_WIDTH_CLOG2-1:0])):{(PE_WIDTH){1'b1}};
            end
            GELU:begin
                prepared_mask    <= (|input_real_len[GELU_NUM_CLOG2-1:0])?~({(PE_WIDTH){1'b1}}<<(input_real_len[GELU_NUM_CLOG2-1:0])):{{(PE_WIDTH-GELU_NUM){1'b0}},{(GELU_NUM){1'b1}}};
            end
            default:prepared_mask <= '0;
        endcase
        end
        else begin
            input_len_spm   <= input_len_spm ;
            input_len_gelu  <= input_len_gelu;
            input_col_spm   <= input_col_spm ;
            prepared_mask   <= prepared_mask ;
        end

    end
    always_comb begin : comb
        sfu_busy         = sfu_current_state != IDLE;
    end 
    always_ff@(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            rd_data_masked<='0;
        end
        else if(sfu_spm_rd_en_dly)begin
            for (int idx=0;idx<PE_WIDTH;idx++)begin
                case (sfu_current_state)
                    SOFTMAX:begin
                        rd_data_masked[idx]<=(idx<input_real_col[PE_WIDTH_CLOG2-1:0]||input_real_col[PE_WIDTH_CLOG2-1:0]==0||sfu_rd_count<input_col_spm)?sfu_spm_rd_data_in[idx]:{1'b1,{(PE_DATA_WIDTH-1){1'b0}}};
                    end
                    LAYERNORM:begin                 
                        rd_data_masked[idx]<=(idx<input_real_col[PE_WIDTH_CLOG2-1:0]||input_real_col[PE_WIDTH_CLOG2-1:0]==0||sfu_rd_count<input_col_spm)?sfu_spm_rd_data_in[idx]:{1'b0,{(PE_DATA_WIDTH-1){1'b0}}};
                    end
                    GELU:begin
                        rd_data_masked[idx]<=(idx<input_real_len[GELU_NUM_CLOG2-1:0]||input_real_len[GELU_NUM_CLOG2-1:0]==0||sfu_rd_count<input_len_gelu)?sfu_spm_rd_data_in[idx]:{1'b0,{(PE_DATA_WIDTH-1){1'b0}}};
                    end
                    default:rd_data_masked[idx]<='0;
                endcase
            end
        end
        else begin
            rd_data_masked<=rd_data_masked;
        end
    end
    always_comb begin 
        case(sfu_current_state)
            RESAMPLE:sfu_comp_done          = resample_done;
            SOFTMAX,LAYERNORM: sfu_comp_done= (sfu_wr_count==input_col_spm)&&(sfu_wr_row_count==sfu_input_row_num)&& sfu_wr_count!=0;
            GELU:sfu_comp_done              = sfu_wr_count==(input_len_gelu) && sfu_wr_count!=0;
            TRANSPOSE :sfu_comp_done        = trans_out_flag && trans_spm_wr_count == PE_WIDTH;
            IDLE: sfu_comp_done             = 0;
            default:sfu_comp_done           = 0;
        endcase
    end
    always_ff@(posedge clk) begin :delay
        sfu_spm_rd_en_dly  <= sfu_spm_rd_en  ;
        sfu_spm_rd_en_dly2 <= sfu_spm_rd_en_dly;
        sfu_req_en_dly     <= sfu_req_en     ;
        sfu_rd_count_dly   <= sfu_rd_count   ;
    end
    genvar idx;
    generate
        for(idx=0;idx<GELU_NUM;idx++)begin
            always_ff@(posedge clk or negedge rst_n)begin
                if(~rst_n)begin
                    gelu_dout[idx]<={(FX_DATA_WIDTH){1'b0}};
                    gelu_done[idx]<=1'b0;
                end
                else if(&gelu_done)begin
                    gelu_dout[idx]<={(FX_DATA_WIDTH){1'b0}};
                    gelu_done[idx]<=1'b0;
                end
                else if(gelu_done_raw[idx])begin
                    gelu_done[idx]<=1'b1;
                    gelu_dout[idx]<=gelu_dout_raw[idx];
                end
                else begin
                    gelu_done[idx]<=gelu_done[idx];
                    gelu_dout[idx]<=gelu_dout[idx];
                end
            end
        end
    endgenerate
    always_ff@(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            sfu_current_state <= IDLE;
        end
        else begin
            sfu_current_state <= sfu_next_state;
        end
    end

    always_comb begin : next_state
        case (sfu_current_state)
            IDLE : begin
                if(sfu_req_en)begin
                    case(cfg_sfu_op)
                        SOFTMAX_OP : begin
                            sfu_next_state = SOFTMAX; 
                        end
                        GELU_OP : begin
                            sfu_next_state = GELU;    
                        end
                        LAYERNORM_OP : begin
                            sfu_next_state = LAYERNORM; 
                        end
                        DOWNSAMPLE_MAX_OP, DOWNSAMPLE_AVG_OP, UPSAMPLE_NEAREST_OP, POOL_MAX_OP, POOL_AVG_OP : begin
                            sfu_next_state = RESAMPLE;
                        end
                        TRANSPOSE_OP:begin
                            sfu_next_state = TRANSPOSE;
                        end
                        default : begin
                            sfu_next_state = IDLE;
                        end 
                    endcase
                end
                else begin
                    sfu_next_state = IDLE;
                end
            end

            SOFTMAX : begin
                if(sfu_comp_done)begin
                    sfu_next_state = IDLE ;
                end
                else begin
                    sfu_next_state = SOFTMAX ;
                end
            end

            GELU : begin
                if(sfu_comp_done)begin
                    sfu_next_state = IDLE ;
                end
                else begin
                    sfu_next_state = GELU ;
                end
            end

            LAYERNORM : begin
                if(sfu_comp_done)begin
                    sfu_next_state = IDLE ;
                end
                else begin
                    sfu_next_state = LAYERNORM ;
                end
            end

            RESAMPLE : begin
                if(sfu_comp_done)begin
                    sfu_next_state = IDLE ;
                end
                else begin
                    sfu_next_state = RESAMPLE ;
                end
            end
            TRANSPOSE : begin
                if(sfu_comp_done)begin
                    sfu_next_state = IDLE ;
                end
                else begin
                    sfu_next_state = TRANSPOSE ;
                end
            end
            default: sfu_next_state = IDLE;
        endcase
    end


   
    always_comb begin : comb_case
        sfu_spm_wr_data_out     = '0;
        sfu_spm_wr_en           = '0;
        sfu_spm_wr_addr         = '0;
        sfu_spm_wr_mask         = '0;
        sfu_spm_rd_addr         = '0;
        sfu_spm_rd_en           = '0;
        resample_spm_rd_data_in = '0;
        resample_input_addr     = '0;
        resample_output_addr    = '0;
        resample_input_en       = '0;
        resample_type           = 2'b11;
        resample_op             = '0;
        resample_input_row_num  = '1;
        resample_input_col_num  = '1;
        layernorm_in_num        = '0;
        softmax_in_num          = '0;
        trans_in_valid          = '0;
        trans_out_ready         = '0;
        case (sfu_current_state)
        IDLE:begin
           sfu_spm_wr_data_out     = '0;
           sfu_spm_wr_en           = '0;
           sfu_spm_wr_addr         = '0;
           sfu_spm_wr_mask         = '0;
           sfu_spm_rd_addr         = '0;
           sfu_spm_rd_en           = '0; 
           resample_spm_rd_data_in = '0;
           resample_input_addr     = '0;
           resample_output_addr    = '0;
           resample_input_en       = '0;
           resample_type           = 2'b11;
           resample_op             = '0;
           resample_input_row_num  = '1;
           resample_input_col_num  = '1;
           layernorm_in_num        = '0; 
           softmax_in_num          = '0;     
           trans_in_valid          = '0; 
           trans_out_ready         = '0; 
        end 
        GELU:begin
           //gelu_din             = gelu_dequant_data_out;  
           //gelu_start           = sfu_spm_rd_en_dly  ;
           //gelu_quant_data_in   = gelu_dout          ;   
           sfu_spm_wr_en        = gelu_quant_valid   ;
           sfu_spm_wr_addr      = spm_wr_addr        ;
           sfu_spm_rd_addr      = spm_rd_addr        ;
           sfu_spm_rd_en        = gelu_spm_rd_en     ;
           sfu_spm_wr_data_out  = {{(PE_WIDTH*PE_DATA_WIDTH-GELU_NUM*PE_DATA_WIDTH){1'b0}},gelu_quant_data_out};
           sfu_spm_wr_mask      = sfu_spm_wr_en?(sfu_wr_count==input_len_gelu-1?prepared_mask:{{(PE_WIDTH-GELU_NUM){1'b0}},{(GELU_NUM){1'b1}}}):0 ;
        //    for(int idx=0;idx<GELU_NUM;idx++)begin
        //        gelu_dequant_data_in[idx]     =  sfu_rd_count==input_len_gelu?rd_data_masked[idx]:sfu_spm_rd_data_in[idx] ;
        //    end
        end
        LAYERNORM:begin
            layernorm_in_num    =  input_real_col[LAYERNORM_NUM_WIDTH-1:0];     
            //norm_in_valid       =  sfu_spm_rd_en_dly  ; 
            sfu_spm_wr_en       =  layernorm_quant_valid ;
            sfu_spm_wr_addr     =  spm_wr_addr        ;
            sfu_spm_rd_addr     =  spm_rd_addr        ;
            sfu_spm_rd_en       =  norm_spm_rd_en     ;
            sfu_spm_wr_mask     =  sfu_spm_wr_en?(sfu_wr_count==input_col_spm-1?prepared_mask:{(PE_WIDTH){1'b1}}):0 ;
            //layernorm_quant_data_in  =  norm_dout          ;
            sfu_spm_wr_data_out =  layernorm_quant_data_out;
            //norm_din            =  layernorm_dequant_data_out;
            //layernorm_dequant_data_in =  sfu_rd_count==input_col_spm?rd_data_masked:sfu_spm_rd_data_in;
        end
        TRANSPOSE:begin
            trans_in_valid          =  trans_spm_rd_count==PE_WIDTH;
            sfu_spm_wr_en           =  trans_spm_wr_en     ;
            trans_out_ready         =  trans_spm_wr_count == 0;
            sfu_spm_wr_addr         =  trans_spm_wr_addr   ;
            sfu_spm_rd_addr         =  trans_spm_rd_addr   ;
            sfu_spm_rd_en           =  trans_spm_rd_en     ;
            sfu_spm_wr_mask         =  trans_spm_wr_mask |{(PE_WIDTH){padding_mask}} ;
            for(int idx=0;idx<PE_WIDTH;idx++)begin
                if(trans_spm_wr_mask[idx])begin
                    sfu_spm_wr_data_out[idx] =  trans_out_data_save[((trans_spm_wr_count>0?trans_spm_wr_count-1:trans_spm_wr_count)<<(PE_WIDTH_CLOG2))+idx]   ;
                end else begin
                    sfu_spm_wr_data_out[idx] = '0;
                end
            end
        end
        SOFTMAX:begin
            softmax_in_num          =  input_col_spm[SOFTMAX_NUM_WIDTH-1:0];
            //softmax_in_valid        =  sfu_spm_rd_en_dly ;  
            sfu_spm_wr_en           =  softmax_quant_valid ;
            sfu_spm_wr_addr         =  spm_wr_addr       ;
            sfu_spm_rd_addr         =  spm_rd_addr       ;
            sfu_spm_rd_en           =  softmax_spm_rd_en ;
            sfu_spm_wr_data_out     =  softmax_quant_data_out;
            //softmax_quant_data_in   =  softmax_out_data  ;
            //softmax_in_data         =  softmax_dequant_data_out;
            sfu_spm_wr_mask         =  sfu_spm_wr_en?(sfu_wr_count==input_col_spm-1?prepared_mask:{(PE_WIDTH){1'b1}}):0 ;
           // softmax_dequant_data_in = sfu_rd_count==input_col_spm?rd_data_masked:sfu_spm_rd_data_in;
        end
        RESAMPLE:begin
            sfu_spm_wr_data_out     = resample_spm_wr_data_out;
            sfu_spm_wr_en           = resample_spm_wr_en      ;
            sfu_spm_wr_addr         = resample_spm_wr_addr    ;
            sfu_spm_wr_mask         = resample_spm_wr_mask    ;
            sfu_spm_rd_addr         = resample_spm_rd_addr    ;
            sfu_spm_rd_en           = resample_spm_rd_en      ;
            resample_spm_rd_data_in = sfu_spm_rd_data_in      ;
            resample_input_row_num  = sfu_input_row_num       ;
            resample_input_col_num  = sfu_input_col_num       ;
            resample_input_addr     = sfu_input_sram_addr     ;
            resample_output_addr    = sfu_output_spm_addr     ;
            resample_input_en       = sfu_req_en_dly          ;
            resample_type           = cfg_sfu_op>4?(cfg_sfu_op==5?2'b01:2'b10):2'b00;
            resample_op             = cfg_sfu_op==4 || cfg_sfu_op==7;
        end
        default:begin
        ;    
        end
        endcase
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if(rst_n==0)
            trans_in_data<='0;
        else if(sfu_current_state==TRANSPOSE && sfu_spm_rd_en_dly)
            trans_in_data[(trans_spm_rd_count*PE_WIDTH)+:PE_WIDTH]<=sfu_spm_rd_data_in;
        else 
            trans_in_data<=trans_in_data;
    end
    always_ff @( posedge clk or negedge rst_n ) begin
        if(~rst_n)begin
            trans_spm_rd_count<=0;
            trans_spm_wr_count<=0;
        end
        else begin
            if((trans_spm_rd_count==PE_WIDTH) || sfu_comp_done)
            trans_spm_rd_count<=0;
            else if(sfu_spm_rd_en_dly && trans_spm_rd_count!=PE_WIDTH)
            trans_spm_rd_count<=trans_spm_rd_count+1;
            else 
            trans_spm_rd_count<=trans_spm_rd_count;

            if(trans_spm_wr_count==PE_WIDTH || sfu_comp_done)
            trans_spm_wr_count<=0;
            else if(sfu_spm_wr_en || (trans_out_valid && trans_out_ready))
            trans_spm_wr_count<=trans_spm_wr_count+1;
            else 
            trans_spm_wr_count<=trans_spm_wr_count;
        end
    end
    always_ff@(posedge clk or negedge rst_n)begin 
        if(~rst_n)begin
            sfu_wr_count  <= 0;
        end
        else if(sfu_current_state!=RESAMPLE && sfu_current_state!=TRANSPOSE)begin
            if(((sfu_current_state==SOFTMAX||sfu_current_state==LAYERNORM)&&(sfu_wr_count == input_col_spm )) || sfu_comp_done)
                sfu_wr_count <= '0 ;
            else if(sfu_spm_wr_en)
                sfu_wr_count <= sfu_wr_count + 1;
            else
                sfu_wr_count <= sfu_wr_count ;
        end
        else 
            sfu_wr_count <= sfu_wr_count ;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if(~rst_n)begin
            sfu_wr_row_count<='0;
        end
        else if(sfu_comp_done)begin
            sfu_wr_row_count<='0;
        end
        else if((sfu_current_state==SOFTMAX||sfu_current_state==LAYERNORM)&&(sfu_wr_count == input_col_spm&& input_col_spm!=0))begin
            sfu_wr_row_count<=sfu_wr_row_count+1;
        end
        else begin
            sfu_wr_row_count<=sfu_wr_row_count;
        end
    end
    always_ff@(posedge clk or negedge rst_n)begin 
        if(~rst_n)begin
            sfu_rd_count  <= 0;
        end
        else if(sfu_current_state!=RESAMPLE && sfu_current_state!=TRANSPOSE)begin
            if(((sfu_current_state==SOFTMAX||sfu_current_state==LAYERNORM)&&(sfu_rd_count == input_col_spm)) || sfu_comp_done)
                sfu_rd_count <= 0 ;
            else if(sfu_spm_rd_en)
                sfu_rd_count <= sfu_rd_count + 1;
            else
                sfu_rd_count <= sfu_rd_count ;
        end
        else 
            sfu_rd_count <= sfu_rd_count ;
    end
    always_ff @(posedge clk or negedge rst_n) begin : wr_addr
        if(~rst_n)
           spm_wr_addr<=0;
        else if(sfu_req_en)
            spm_wr_addr <= sfu_output_spm_addr;     
        else if(sfu_spm_wr_en && sfu_wr_count==input_col_spm-1 &&(sfu_current_state==SOFTMAX || sfu_current_state==LAYERNORM))
            spm_wr_addr <= spm_wr_addr + ((input_real_col[PE_WIDTH_CLOG2-1:0]==0)?SPM_ADDR_INCR:(input_real_col[PE_WIDTH_CLOG2-1:0]<<PE_DATA_SIZE));
        else if(sfu_spm_wr_en)
            spm_wr_addr <= spm_wr_addr + (sfu_current_state==GELU?GELU_NUM*PE_DATA_WIDTH/8:SPM_ADDR_INCR);
        else 
            spm_wr_addr <= spm_wr_addr;
    end
    always_ff @(posedge clk or negedge rst_n) begin : rd_addr
        if(~rst_n)
           spm_rd_addr<=0;
        else if(sfu_req_en)
            spm_rd_addr <= sfu_input_sram_addr;
        else if(sfu_spm_rd_en && sfu_rd_count==input_col_spm-1&&(sfu_current_state==SOFTMAX||sfu_current_state==LAYERNORM))
            spm_rd_addr <= spm_rd_addr + ((input_real_col[PE_WIDTH_CLOG2-1:0]==0)?SPM_ADDR_INCR:(input_real_col[PE_WIDTH_CLOG2-1:0]<<PE_DATA_SIZE));
        else if(sfu_spm_rd_en )
            spm_rd_addr <= spm_rd_addr + (sfu_current_state==GELU?GELU_NUM*PE_DATA_WIDTH/8:SPM_ADDR_INCR);
        else 
            spm_rd_addr <= spm_rd_addr;
    end


    always_ff@(posedge clk or negedge rst_n)begin : sfu_fsm
        if(~rst_n)begin
            norm_spm_rd_en      <= 0;
            gelu_spm_rd_en      <= 0;
            softmax_spm_rd_en   <= 0;
            trans_spm_rd_en     <= 0;
        end
        else begin
            case(sfu_current_state)
            IDLE:begin
                norm_spm_rd_en      <= 0;
                gelu_spm_rd_en      <= 0;
                softmax_spm_rd_en   <= 0;
                trans_spm_rd_en     <= 0;
            end
            LAYERNORM:begin
                if(sfu_req_en_dly)begin
                norm_spm_rd_en   <= 1 ;
                end
                else if( norm_spm_rd_en && sfu_rd_count == input_col_spm -1)begin
                norm_spm_rd_en  <=0;
                end
                else if(sfu_wr_count==input_col_spm)
                norm_spm_rd_en  <='1;
                else 
                norm_spm_rd_en <= norm_spm_rd_en ;
            end
            TRANSPOSE:begin
                if(sfu_req_en_dly)
                trans_spm_rd_en <=1;
                else if(trans_spm_rd_count==PE_WIDTH-2)
                trans_spm_rd_en <= '0;
                else if(trans_in_ready&&trans_in_valid&&(!trans_out_flag))
                trans_spm_rd_en <=1'b1;
                // else if(trans_in_row_count==trans_in_real_depth1 - 1 && trans_in_col_count == trans_in_real_depth0 - 1 && trans_spm_rd_count==PE_WIDTH-2)
                // trans_spm_rd_en <=0;
                else
                trans_spm_rd_en <=trans_spm_rd_en ;
            end
            SOFTMAX:begin
                if(sfu_req_en_dly)begin
                softmax_spm_rd_en   <= 1 ;
                end
                else if( softmax_spm_rd_en && sfu_rd_count == input_col_spm -1)begin
                softmax_spm_rd_en  <=0;
                end
                else if(sfu_wr_count==input_col_spm)
                softmax_spm_rd_en  <='1;
                else 
                softmax_spm_rd_en <= softmax_spm_rd_en ;
            end
            GELU:begin
                if(sfu_req_en_dly)
                gelu_spm_rd_en <= 1;
                else if(gelu_spm_rd_en == 1)
                gelu_spm_rd_en <= 0;
                else if(gelu_spm_rd_en == 0 && (&gelu_done)&&sfu_rd_count!=input_len_gelu)
                gelu_spm_rd_en <= 1;
                else 
                gelu_spm_rd_en <= gelu_spm_rd_en;
            end
            default:begin
                norm_spm_rd_en       <= 0;
                gelu_spm_rd_en       <= 0;
                softmax_spm_rd_en    <= 0;
                trans_spm_rd_en      <= 0;
            end
            endcase
        end
    end 
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(~rst_n)
            trans_out_data_save<='0;
        else if(trans_out_ready && trans_out_valid)
            trans_out_data_save<=trans_out_data;
        else 
            trans_out_data_save<=trans_out_data_save;
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(~rst_n)
        trans_spm_wr_en <= 0;
        else if(trans_spm_wr_count == PE_WIDTH)
        trans_spm_wr_en <=0;
        else if(trans_out_ready && trans_out_valid)
        trans_spm_wr_en <=1;
        else 
        trans_spm_wr_en <= trans_spm_wr_en;
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(~rst_n)
        trans_spm_rd_addr<=0;
        else if(sfu_current_state == TRANSPOSE || sfu_current_state==IDLE) begin
            if(sfu_req_en)
            trans_spm_rd_addr <= sfu_input_sram_addr;
            else if(trans_spm_rd_count==PE_WIDTH-2 && sfu_spm_rd_en && trans_in_col_count == trans_in_real_depth0 - 1)
            trans_spm_rd_addr <= trans_spm_rd_addr + ((input_real_col[PE_WIDTH_CLOG2-1:0]==0)?SPM_ADDR_INCR:((input_real_col[PE_WIDTH_CLOG2-1:0])<<PE_DATA_SIZE));
            else if(trans_spm_rd_count==PE_WIDTH-2 && sfu_spm_rd_en)
            trans_spm_rd_addr <= trans_spm_rd_addr - (input_real_col<<(PE_WIDTH_CLOG2+PE_DATA_SIZE))+(input_real_col<<PE_DATA_SIZE)+SPM_ADDR_INCR;
            else if(sfu_spm_rd_en)
            trans_spm_rd_addr <= trans_spm_rd_addr + (input_real_col<<PE_DATA_SIZE);
            else 
            trans_spm_rd_addr <= trans_spm_rd_addr;
        end
        else
        trans_spm_rd_addr <= trans_spm_rd_addr;
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(~rst_n)
        trans_spm_wr_addr<=0;
        else if(sfu_current_state == TRANSPOSE || sfu_current_state==IDLE)begin
            if(sfu_req_en)
            trans_spm_wr_addr <= sfu_output_spm_addr;
            //else if(trans_spm_wr_count==PE_WIDTH && sfu_spm_wr_en && trans_out_col_count == trans_in_real_depth1 - 1)
            //trans_spm_wr_addr <= trans_spm_wr_addr + ((input_real_row[PE_WIDTH_CLOG2-1:0]==0)?SPM_ADDR_INCR:((input_real_row[PE_WIDTH_CLOG2-1:0])<<PE_DATA_SIZE));
            else if(trans_spm_wr_count==PE_WIDTH && sfu_spm_wr_en && trans_out_row_count == trans_in_real_depth0-1)
            trans_spm_wr_addr <= sfu_output_spm_addr + ((trans_out_col_count+1)<<($clog2(SPM_ADDR_INCR)));
            else if(sfu_spm_wr_en)
            trans_spm_wr_addr <= trans_spm_wr_addr + (trans_wr_stride<<PE_DATA_SIZE);
            else 
            trans_spm_wr_addr <= trans_spm_wr_addr;
        end
        else 
        trans_spm_wr_addr <= trans_spm_wr_addr;
    end
    always_comb begin
        if(trans_out_row_count == trans_in_real_depth0 -1 && trans_out_col_count == trans_in_real_depth1-1 && sfu_spm_wr_en)
        trans_spm_wr_mask = (trans_spm_wr_count-1<input_real_col[PE_WIDTH_CLOG2-1:0]|| input_real_col[PE_WIDTH_CLOG2-1:0]==0)?prepared_mask:{(PE_WIDTH){1'b0}};
        else if(trans_out_row_count == trans_in_real_depth0 -1 && sfu_spm_wr_en)
        trans_spm_wr_mask = (trans_spm_wr_count-1<input_real_col[PE_WIDTH_CLOG2-1:0] || input_real_col[PE_WIDTH_CLOG2-1:0]==0)?{(PE_WIDTH){1'b1}}:{(PE_WIDTH){1'b0}};
        else if(trans_out_col_count == trans_in_real_depth1-1 && sfu_spm_wr_en)
        trans_spm_wr_mask = prepared_mask;
        else if(sfu_spm_wr_en)
        trans_spm_wr_mask = {(PE_WIDTH){1'b1}};
        else 
        trans_spm_wr_mask = {(PE_WIDTH){1'b0}};
    end
    always_comb begin
        trans_padding_row = cfg_trans_out_ispad_row && (trans_out_row_count == trans_in_real_depth0 - 1)&&(trans_spm_wr_count>input_real_col[PE_WIDTH_CLOG2-1:0])&&(input_real_col[PE_WIDTH_CLOG2-1:0]!=0);
        trans_padding_col = cfg_trans_out_ispad_col && (trans_out_col_count == trans_in_real_depth1 - 1);
        padding_mask      = sfu_spm_wr_en &&(trans_padding_col || trans_padding_row);
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(~rst_n)
        trans_out_flag<=0;
        else if(sfu_comp_done)
        trans_out_flag <=0;
        else if(trans_spm_rd_count==PE_WIDTH-1 && trans_in_row_count == trans_in_real_depth1 -1 && trans_in_col_count == trans_in_real_depth0-1)
        trans_out_flag <=1;
        else 
        trans_out_flag <= trans_out_flag;
    end
    always_ff @( posedge clk or negedge rst_n ) begin 
        if(~rst_n)begin
        trans_out_col_count <= 0;
        trans_out_row_count <= 0;
        end
        else if(trans_spm_wr_count==PE_WIDTH)begin
        trans_out_col_count <= out_col_count;
        trans_out_row_count <= out_row_count;
        end
        else begin
        trans_out_col_count <= trans_out_col_count;
        trans_out_row_count <= trans_out_row_count;
        end

    end 
    

    
    // GELU only support FP16 input and output, quant required
    generate
        if (DISABLE_GELU == 1) begin : gen_no_gelu
            for(idx = 0; idx < GELU_NUM; idx++)begin
                assign gelu_dout_raw[idx] = '0;
                assign gelu_busy[idx] = 1'b0;
                assign gelu_done_raw[idx] = 1'b0;
            end
        end
        else begin : gen_gelu
            for(idx = 0; idx < GELU_NUM; idx++)begin
                gelu_top  #(
                .DATA_WIDTH(FX_DATA_WIDTH),
                .FRAC_WIDTH(GELU_FRAC_WIDTH)
                ) 
                u_gelu(
                    .clk    ( clk                        ),   
                    .rst_n  ( rst_n                      ),       
                    .din    ( gelu_din[idx]              ),   
                    .start  ( gelu_start                 ),       
                    .dout   ( gelu_dout_raw[idx]         ),       
                    .busy   ( gelu_busy[idx]             ),       
                    .done   ( gelu_done_raw[idx]         )             
                );
            end
        end
    endgenerate

    generate
        if (DISABLE_SOFTMAX == 1) begin : gen_no_softmax
            assign softmax_out_data = '0;
            assign softmax_out_valid = 1'b0;
        end
        else begin : gen_softmax
            softmax_approx #(
                .GROUP_SIZE      (PE_WIDTH            ),
                .IN_DATA_WIDTH   (FX_DATA_WIDTH       ),
                .IN_FRAC_WIDTH   (SOFTMAX_FRAC_WIDTH  ),
                .OUT_DATA_WIDTH  (FX_DATA_WIDTH       ),
                .OUT_FRAC_WIDTH  (FX_DATA_WIDTH-1     ),
                .MAX_LENGTH      (SFU_MAX_INPUT_LENGTH),
                .SHIFT_WIDTH     (4                   ),
                .SHIFT_DATA_WIDTH(8                   )
            ) u_softmax (
                .clk      (clk                        ),
                .rst_n    (rst_n                      ),
                .data_in  (softmax_in_data            ),   
                .in_valid (softmax_in_valid           ),
                .data_out (softmax_out_data           ),      
                .out_valid(softmax_out_valid          ),
                .in_num   (softmax_in_num             )
            );
        end
    endgenerate
   
    generate
        if (DISABLE_LAYERNORM == 1) begin : gen_no_layernorm
            assign norm_out_valid = 1'b0;
            assign norm_dout = '0;
        end
        else begin : gen_layernorm
            layernorm_1d #(
                .MAX_LENGTH         ( SFU_MAX_INPUT_LENGTH),
                .GROUP_SIZE         ( PE_WIDTH             ),
                .IDATA_WIDTH        ( FX_DATA_WIDTH        ),
                .IFRAC_WIDTH        ( LAYERNORM_FRAC_WIDTH ),
                .ODATA_WIDTH        ( FX_DATA_WIDTH        ),
                .OFRAC_WIDTH        ( LAYERNORM_FRAC_WIDTH )
            )u_layernorm(
               .clk         ( clk                        ),   
               .rst_n       ( rst_n                      ),  
               .in_real_num ( layernorm_in_num           ), 
               .in_valid_i  ( norm_in_valid              ),
               .in_data_i   ( norm_din                   ),
               .out_valid_o ( norm_out_valid             ),
               .out_data_o  ( norm_dout                  )
            );
        end
    endgenerate

    generate
        if (DISABLE_RESAMPLE == 1) begin : gen_no_resample
            assign resample_busy = 1'b0;
            assign resample_done = 1'b0;
            assign resample_spm_rd_en = 1'b0;
            assign resample_spm_rd_addr = '0;
            assign resample_spm_wr_en = 1'b0;
            assign resample_spm_wr_addr = '0;
            assign resample_spm_wr_data_out = '0;
            assign resample_spm_wr_mask = '0;
        end
        else begin : gen_resample
            resample_unit u_resample_unit(
                .clk                        ( clk                       ),
                .rst_n                      ( rst_n                     ), 

                .resample_type              ( resample_type             ),   
                .resample_op                ( resample_op               ),   

                .resample_input_en          ( resample_input_en         ),   
                .resample_input_addr        ( resample_input_addr       ),   
                .resample_output_addr       ( resample_output_addr      ),   

                .resample_input_row_num     ( resample_input_row_num    ),   
                .resample_input_col_num     ( resample_input_col_num    ),   

                .resample_busy              ( resample_busy             ),
                .resample_done_out          ( resample_done             ),   

                .resample_spm_rd_en         ( resample_spm_rd_en        ),   
                .resample_spm_rd_addr       ( resample_spm_rd_addr      ),   
                .resample_spm_rd_data_in    ( resample_spm_rd_data_in   ),   

                .resample_spm_wr_en         ( resample_spm_wr_en        ),   
                .resample_spm_wr_addr       ( resample_spm_wr_addr      ),   
                .resample_spm_wr_data_out   ( resample_spm_wr_data_out  ),   
                .resample_spm_wr_mask       ( resample_spm_wr_mask      )    
            );
        end
    endgenerate

    generate
        if (DISABLE_TRANSPOSE == 1) begin : gen_no_transpose
            assign trans_in_ready = 1'b0;
            assign trans_out_data = '0;
            assign trans_out_valid = 1'b0;
            assign out_row_count = '0;
            assign out_col_count = '0;
        end
        else begin : gen_transpose
            matrix_stream_transpose_new #(
            .MAX_DIM0     ( TRANSPOSE_MAX_LENGTH   ),
            .MAX_DIM1     ( TRANSPOSE_MAX_LENGTH   ),
            .COMPUTE_DIM0 ( PE_WIDTH          ),
            .COMPUTE_DIM1 ( PE_WIDTH          ),
            .DATA_WIDTH   ( PE_DATA_WIDTH     ))
            u_matrix_stream_transpose (
            .clk            ( clk              ),
            .rst            ( ~rst_n           ),
            .in_real_dim0   ( input_real_col[TRANS_DIM_WIDTH:0]   ),
            .in_real_dim1   ( input_real_row[TRANS_DIM_WIDTH:0]   ),
            .in_data        ( trans_in_data    ),
            .in_valid       ( trans_in_valid   ),
            .out_ready      ( trans_out_ready  ),
             
            .in_ready       ( trans_in_ready   ),
            .out_data       ( trans_out_data   ),
            .out_valid      ( trans_out_valid  ),
            .in_row_count   ( trans_in_row_count  ),
            .in_col_count   ( trans_in_col_count  ),
            .out_row_count  (     out_row_count ),
            .out_col_count  (     out_col_count ),
            .in_real_depth0 (trans_in_real_depth0),
            .in_real_depth1 (trans_in_real_depth1)
            );
        end
    endgenerate
    
    // quant, after layernorm 
    generate
        if (DISABLE_LAYERNORM == 1) begin : gen_no_quant_layernorm
            assign layernorm_quant_data_out = '0;
        end
        else begin : gen_quant_layernorm
            quantization_fxX_to_intY #(
            .SCALE_DATA_WIDTH ( RF_DATA_WIDTH/4  ),
            .INPUT_DATA_WIDTH ( FX_DATA_WIDTH    ),
            .OUTPUT_DATA_WIDTH( PE_DATA_WIDTH    ),
            .INPUT_NUMBER     ( PE_WIDTH         ),
            .FRAC_WIDTH       ( LAYERNORM_FRAC_WIDTH   ))
            u_quantization_fx16_to_int8_layernorm (
            .clk               ( clk                         ),
            .rst_n             ( rst_n                       ),
            .in_valid          ( norm_out_valid              ),
            .quantized_input   ( norm_dout                   ),
            .quant_scale       ( cfg_sfu_output_scale        ),
            .scale_shift       ( cfg_sfu_output_scale_shift  ),
            .out_valid         ( layernorm_quant_valid       ),
            .quantized_output  ( layernorm_quant_data_out    )
            );
        end
    endgenerate

    // quant, after softmax
    generate
        if (DISABLE_SOFTMAX == 1) begin : gen_no_quant_softmax
            assign softmax_quant_data_out = '0;
        end
        else begin : gen_quant_softmax
            quantization_fxX_to_intY #(
            .SCALE_DATA_WIDTH ( RF_DATA_WIDTH/4  ),
            .INPUT_DATA_WIDTH ( FX_DATA_WIDTH    ),
            .INPUT_NUMBER     ( PE_WIDTH         ),
            .OUTPUT_DATA_WIDTH( PE_DATA_WIDTH    ),
            .FRAC_WIDTH       ( FX_DATA_WIDTH-1  ))
            u_quantization_fx16_to_int8_softmax (
            .clk              ( clk                         ),
            .rst_n            ( rst_n                       ),
            .in_valid         ( softmax_out_valid           ),    
            .quantized_input  ( softmax_out_data            ),
            .quant_scale      ( cfg_sfu_output_scale        ),
            .scale_shift      ( cfg_sfu_output_scale_shift  ),
            .out_valid        ( softmax_quant_valid         ),
            .quantized_output ( softmax_quant_data_out      )
            );
        end
    endgenerate

    //quant after gelu
    generate
        if (DISABLE_GELU == 1) begin : gen_no_quant_gelu
            assign gelu_quant_data_out = '0;
        end
        else begin : gen_quant_gelu
            quantization_fxX_to_intY #(
            .SCALE_DATA_WIDTH ( RF_DATA_WIDTH/4  ),
            .INPUT_DATA_WIDTH ( FX_DATA_WIDTH    ),
            .INPUT_NUMBER     ( GELU_NUM         ),
            .OUTPUT_DATA_WIDTH( PE_DATA_WIDTH    ),
            .FRAC_WIDTH       ( GELU_FRAC_WIDTH  ))
            u_quantization_fx16_to_int8_gelu (
            .clk              ( clk                          ),
            .rst_n            ( rst_n                        ),
            .in_valid         ( &gelu_done                   ),
            .quantized_input  ( gelu_dout                    ),
            .quant_scale      ( cfg_sfu_output_scale         ),
            .scale_shift      ( cfg_sfu_output_scale_shift   ),
            .out_valid        ( gelu_quant_valid             ),
            .quantized_output ( gelu_quant_data_out          )
            );
        end
    endgenerate

    //dequant before softmax
    generate
        if (DISABLE_SOFTMAX == 1) begin : gen_no_dequant_softmax
            assign softmax_in_data = '0;
        end
        else begin : gen_dequant_softmax
            quantization_intX_to_fxY #(
            .SCALE_DATA_WIDTH ( RF_DATA_WIDTH/4  ),
            .INPUT_DATA_WIDTH ( PE_DATA_WIDTH    ),
            .OUTPUT_DATA_WIDTH( FX_DATA_WIDTH    ),
            .INPUT_NUMBER     ( PE_WIDTH         ),
            .FRAC_WIDTH       ( SOFTMAX_FRAC_WIDTH ))
            u_quantization_int8_to_fx16_softmax (
            .clk              ( clk                          ),
            .rst_n            ( rst_n                        ),
            .in_valid         ( sfu_spm_rd_en_dly2&&sfu_current_state==SOFTMAX),
            .quantized_input  ( rd_data_masked               ),
            .quant_scale      ( cfg_sfu_input_scale          ),
            .scale_shift      ( cfg_sfu_input_scale_shift    ),

            .out_valid        ( softmax_in_valid             ),
            .quantized_output ( softmax_in_data              )
            );
        end
    endgenerate

    //dequant before gelu
    generate
        if (DISABLE_GELU == 1) begin : gen_no_dequant_gelu
            assign gelu_din = '0;
        end
        else begin : gen_dequant_gelu
            quantization_intX_to_fxY #(
            .SCALE_DATA_WIDTH ( RF_DATA_WIDTH/4  ),
            .INPUT_DATA_WIDTH ( PE_DATA_WIDTH    ),
            .OUTPUT_DATA_WIDTH( FX_DATA_WIDTH    ),
            .INPUT_NUMBER     ( GELU_NUM         ),
            .FRAC_WIDTH       ( GELU_FRAC_WIDTH  ))
            u_quantization_int8_to_fx16_gelu (
            .clk              ( clk                          ),
            .rst_n            ( rst_n                        ),
            .in_valid         ( sfu_spm_rd_en_dly2&&sfu_current_state==GELU),
            .quantized_input  ( rd_data_masked[GELU_NUM-1:0] ),
            .quant_scale      ( cfg_sfu_input_scale          ),
            .scale_shift      ( cfg_sfu_input_scale_shift    ),
            .out_valid        ( gelu_start                   ),
            .quantized_output ( gelu_din                     )
            );
        end
    endgenerate

    //dequant before layernorm
    generate
        if (DISABLE_LAYERNORM == 1) begin : gen_no_dequant_layernorm
            assign norm_din = '0;
        end
        else begin : gen_dequant_layernorm
            quantization_intX_to_fxY #(
            .SCALE_DATA_WIDTH ( RF_DATA_WIDTH/4  ),
            .INPUT_DATA_WIDTH ( PE_DATA_WIDTH    ),
            .INPUT_NUMBER     ( PE_WIDTH         ),
            .OUTPUT_DATA_WIDTH( FX_DATA_WIDTH    ),
            .FRAC_WIDTH       ( LAYERNORM_FRAC_WIDTH))
            u_quantization_int8_to_fx16_layernorm (
            .clk              ( clk                          ),
            .rst_n            ( rst_n                        ),
            .in_valid         ( sfu_spm_rd_en_dly2&&sfu_current_state==LAYERNORM),
            .quantized_input  ( rd_data_masked               ),
            .quant_scale      ( cfg_sfu_input_scale          ),
            .scale_shift      ( cfg_sfu_input_scale_shift    ),
            .out_valid        ( norm_in_valid                ),
            .quantized_output ( norm_din                     )
            );
        end
    endgenerate
    
endmodule