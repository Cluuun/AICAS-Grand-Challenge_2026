//////////////////////////////////////////////////////////////////////////////////
// Copyright by FuxionLab 
//
// Designer     : Sihao Fu
// Create Date  : 2024/11/20
// Project Name : T_NPU
// File Name    : resample.sv 
//
// Description  : 2x Upsample/Downsample the output from systolic array(scratchpad) and store the resampled result into scratchpad.
//
// Revision:  
// Revision 2.0 - File Created
// Additional Comments:     
//
//////////////////////////////////////////////////////////////////////////////////

module resample_unit_base 
    import npu_config_pkg::*;
#(
    localparam  ADDR_WIDTH          = $clog2(SPM_SIZE)   ,
    localparam  PE_IDX              = $clog2(PE_WIDTH)  ,
    localparam  RESAMPLE_IDX_WIDTH  = RF_DATA_WIDTH/4   ,    // index for the image to resample, height and width limited by 2^(RF_DATA_WIDTH/8)
    localparam  PE_DATA_SIZE        = $clog2(PE_DATA_WIDTH/8)   
) (
    input  logic                                            clk                     ,
    input  logic                                            rst_n                   , 
    
    // Instuction Control
    input  logic    [1:0]                                   resample_type           ,   // 00 for downsample, 01 for upsample, 10 for pooling,
                                                         
                                                                                            // keep whiling resampling
    input  logic                                            resample_op             ,   // operation of resampling, keep whiling resampling
                                                                                            // for downsampling, 0 for max, 1 for average, 
                                                                                            // for upsampling, 0 for nearest, 1 for bilinear(bilinear not supported now)
                                                                                            // for pooling, 0 for max, 1 for average
                                                                                            // other mode not supported currently, 
                                                                                            // especially bicubic cannot be supported under current architecture
    
    input  logic                                            resample_input_en       ,   // enable signal for resample unit to start working
    input  logic    [ADDR_WIDTH-1:0]                        resample_input_addr     ,   // initial SPM address of input data to be resampled
    input  logic    [ADDR_WIDTH-1:0]                        resample_output_addr    ,   // initial SPM address of already resampled output data

    input  logic    [RESAMPLE_IDX_WIDTH-1:0]                resample_input_row_num  ,   // number of rows of input data, resample_input_row_num = real row number -1, keep while resampling
    input  logic    [RESAMPLE_IDX_WIDTH-1:0]                resample_input_col_num  ,   // number of columns of input data, resample_input_col_num = real column number -1, keep while resampling
    
    output logic                                            resample_busy           ,
    output logic                                            resample_done_out       ,   // resample operation complete

    // ScratchPad 
    output logic                                            resample_spm_rd_en      ,   // SPM read enable
    output logic    [ADDR_WIDTH-1:0]                        resample_spm_rd_addr    ,   // SPM read address
    input  logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       resample_spm_rd_data_in ,   // read data from SPM

    output logic                                            resample_spm_wr_en      ,   // SPM write enable
    output logic    [ADDR_WIDTH-1:0]                        resample_spm_wr_addr    ,   // SPM write address
    output logic    [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]       resample_spm_wr_data_out,   // write data into SPM
    output logic    [PE_WIDTH-1:0]                          resample_spm_wr_mask        // SPM write mask
      
);
    //-------------------------------------------------------------------
    // resample direction
    //
    //  ↓         → ↓         →        
    //  ↓        /  ↓        /        
    //  ↓       /   ↓       /        
    //  ↓      /    ↓      /        
    //  ↓     /     ↓     /    
    //  ↓    /      ↓    /    
    //  ↓   /       ↓   /    
    //  ↓  /        ↓  /    
    //  ↓ /         ↓ /
    //     
    //-------------------------------------------------------------------
    localparam  TYPE_IDLE=2'b11;
    localparam  RESAMPLE_MAX_ROW = 1<<RESAMPLE_IDX_WIDTH;
    logic   [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]               temp_data_row           ;   // store temporary row data for downsample and pooling
    logic   [RESAMPLE_MAX_ROW+1:0][PE_DATA_WIDTH-1:0]       temp_data_col           ;   // store temporary column data for pooling
    logic   [RESAMPLE_IDX_WIDTH:0]                          temp_col_rd_ptr         ;   // rd pointer temp data col 
    logic   [RESAMPLE_IDX_WIDTH:0]                          temp_col_wr_ptr         ;   // wr pointer temp data col


    logic   [RESAMPLE_IDX_WIDTH-1:0]                        row_ptr                 ;   // row pointer for current input feature row position
    logic   [RESAMPLE_IDX_WIDTH-1:0]                        col_ptr                 ;   // column pointer for current input feature row position

    logic                                                   vector_resample_done    ;   // current vector in a row resample complete
    logic                                                   col_resample_done       ;   // current column resample complete
    logic                                                   reach_last_col          ;   // col_ptr reach the last vector of a row, namely the last column
    logic                                                   resample_rd_busy        ;

    logic   [RESAMPLE_IDX_WIDTH:0]                          resample_input_col_num_ceil;   // ceil(resample_input_col_num << 1) >> 1, even number
    logic   [PE_IDX:0]                                      resample_input_col_num_ceil_half;   //(ceil[PE_IDX-1:0]+1)/2                   

    logic   [1:0]                                           upsample_cnt            ;   // counter for generating downsample results, 1 input data for 4 output data

    logic   [ADDR_WIDTH-1:0]                                resample_next_col_rd_addr  ;   // read address for the first row in next column of input
    logic   [ADDR_WIDTH-1:0]                                resample_next_col_wr_addr  ;   // write address for the first row in next column of output
    
    logic                                                   resample_done           ;
    logic                                                   resample_done_delay1    ;
    logic                                                   resample_done_delay2    ;

    logic   [RESAMPLE_IDX_WIDTH-1:0]                        row_ptr_delay           ;   // delay row_ptr for 1 cycle to align timing   
    logic   [RESAMPLE_IDX_WIDTH-1:0]                        col_ptr_delay           ;   // delay col_ptr for 1 cycle to align timing
    logic   [RESAMPLE_IDX_WIDTH-1:0]                        col_ptr_delay_delay     ;   // delay col_ptr for 2 cycle to align timing
    logic   [1:0]                                           upsample_cnt_delay      ;   // counter for generating downsample results, 1 input data for 4 output data
    logic                                                   resample_spm_rd_en_delay;   // delay resample_spm_rd_en for 1 cycle to align timing
    logic                                                   col_resample_done_delay ;   // delay col_resample_done for 1 cycle to align timing
    logic                                                   reach_last_col_delay    ;   // delay reach_last_col for 1 cycle to align timing
    logic                                                   resample_input_en_delay ;
 
    //-------------------------------------------------------------------
    // delay signals
    //-------------------------------------------------------------------
    always_ff@(posedge clk) begin : delay_signals
        row_ptr_delay               <= row_ptr;
        col_ptr_delay               <= col_ptr;
        upsample_cnt_delay          <= upsample_cnt;
        resample_spm_rd_en_delay    <= resample_spm_rd_en;
        reach_last_col_delay        <= reach_last_col;
        resample_input_en_delay     <= resample_input_en;
        col_ptr_delay_delay         <= col_ptr_delay;
    end
    always_ff@(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
            resample_done_delay1        <= 0;
            resample_done_delay2        <= 0;
        end
        else if(resample_done_out)begin
            resample_done_delay1        <= 0;
            resample_done_delay2        <= 0;
        end
        else begin
            resample_done_delay1        <= resample_done;
            resample_done_delay2        <= resample_done_delay1;
        end

    end
    //-------------------------------------------------------------------
    // calculate and update pointers
    //-------------------------------------------------------------------
    always_ff@(posedge clk or negedge rst_n)begin : row_ptr_ff
        if(~rst_n)begin
            row_ptr <= '0;     
        end
        else begin
            if(resample_input_en || resample_done_out)begin
                row_ptr <= '0;
            end
            else begin
                if(resample_spm_rd_en)begin
                    row_ptr <= (row_ptr != resample_input_row_num) ? (row_ptr + 1'b1) : '0;
                end    
                else begin
                    row_ptr <= row_ptr;
                end
            end
        end
    end

    always_ff@(posedge clk or negedge rst_n)begin : col_ptr_ff
        if(~rst_n)begin
            col_ptr <= '0;     
        end
        else begin
            if(resample_input_en |resample_done_out)begin
                col_ptr <= '0;
            end
            else begin
                if(row_ptr == resample_input_row_num && resample_type!=TYPE_IDLE)begin
                    if(resample_type == UPSAMPLE_TYPE)begin
                        if(upsample_cnt == 2'b11)begin
                            col_ptr <= (resample_done) ? '0 : (col_ptr + PE_WIDTH);
                        end
                        else begin
                            col_ptr <= col_ptr;
                        end
                    end
                    else begin
                        col_ptr <= (resample_done) ? '0 : (col_ptr + PE_WIDTH);
                    end
                end    
                else begin
                    col_ptr <= col_ptr;
                end
            end
        end
    end

    //-------------------------------------------------------------------
    // store temporate row/column boundary data
    //-------------------------------------------------------------------
    always_ff@(posedge clk or negedge rst_n)begin : temp_data_row_ff
        if(~rst_n)begin
            temp_data_row <= '0;
        end
        else begin
            if(resample_spm_rd_en_delay)begin
                if(resample_type == DOWNSAMPLE_TYPE)begin
                    temp_data_row <= ((row_ptr == resample_input_row_num) | (row_ptr[0] == 1'b1)) ? resample_spm_rd_data_in : temp_data_row;
                end
                else if(resample_type == POOLING_TYPE)begin
                    temp_data_row <= resample_spm_rd_data_in;
                end
                // else if(resample_type == UPSAMPLE_TYPE)begin    
                // end
                else begin
                    temp_data_row <= temp_data_row;
                end
            end
            else begin
                temp_data_row <= temp_data_row;
            end
        end
    end

    always_ff@(posedge clk or negedge rst_n)begin : temp_data_col_ff
        if(~rst_n)begin
            temp_data_col   <= '0;
        end
        else begin
            if(resample_spm_rd_en_delay && resample_type == POOLING_TYPE)begin
                temp_data_col[temp_col_wr_ptr] <= resample_spm_rd_data_in[PE_WIDTH-1];   // read the last bit of the vector 
            end 
            else if(resample_done_out)begin
                temp_data_col   <= '0;
            end
            else begin
                temp_data_col   <= temp_data_col;
            end
        end
    end
    
    always_ff@(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
        temp_col_wr_ptr <= 0;
        end
        else if(temp_col_wr_ptr == RESAMPLE_MAX_ROW+1 || resample_done_out)
        temp_col_wr_ptr <=0;
        else if(resample_spm_rd_en_delay&& resample_type == POOLING_TYPE)
        temp_col_wr_ptr <= temp_col_wr_ptr +1;
        else 
        temp_col_wr_ptr <= temp_col_wr_ptr;
    end

    always_ff@(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
        temp_col_rd_ptr <= 0;
        end
        else if(temp_col_rd_ptr == RESAMPLE_MAX_ROW+1 || resample_done_out)
        temp_col_rd_ptr <=0;
        else if((temp_col_rd_ptr == 0 && col_ptr_delay_delay !=0) || (resample_input_row_num==0 && resample_spm_wr_en))
        temp_col_rd_ptr <= temp_col_rd_ptr+1;
        else if((col_ptr_delay_delay!= 0) && resample_spm_wr_en && (resample_type == POOLING_TYPE))
        temp_col_rd_ptr <= temp_col_rd_ptr + (row_ptr_delay==0?0:(row_ptr==0?2:1));
        else 
        temp_col_rd_ptr <= temp_col_rd_ptr;
    end



    always_comb begin : vector_resample_done_ff
        case(resample_type)
            DOWNSAMPLE_TYPE : begin
                vector_resample_done = ((row_ptr == resample_input_row_num) || (row_ptr[0] == 1'b1)) ? 1'b1 : 1'b0;  // odd row num, last row only downsample horizontally
            end

            UPSAMPLE_TYPE : begin
                vector_resample_done = upsample_cnt == (2'b11) ? 1'b1 : 1'b0;
            end

            POOLING_TYPE : begin
                vector_resample_done = (row_ptr == '0) ? 1'b0 : 1'b1; 
            end

            default : begin
                vector_resample_done = 1'b0;
            end
        endcase  
    end

    always_ff@(posedge clk or negedge rst_n)begin : upsample_cnt_ff
        if(~rst_n)begin
            upsample_cnt <= '0;
        end
        else begin
            if(resample_done |resample_done_out)begin
                upsample_cnt <= '0;
            end
            else if(resample_input_en)begin
                upsample_cnt <=(reach_last_col&&resample_input_col_num[PE_IDX-1] == 1'b0)?2'b01:2'b00;
            end
            else begin
                if(resample_type == UPSAMPLE_TYPE)begin
                    if(reach_last_col&& (resample_input_col_num[PE_IDX-1] == 1'b0))begin
                        upsample_cnt <= (upsample_cnt == 2'b11) ? 2'b01 : (upsample_cnt + 2'b10);
                    end
                    else if((col_ptr[RESAMPLE_IDX_WIDTH-1 : PE_IDX] == resample_input_col_num[RESAMPLE_IDX_WIDTH-1 : PE_IDX]-1)&&
                    (row_ptr==resample_input_row_num)&&(upsample_cnt==2'b11)&&(resample_input_col_num[PE_IDX-1] == 1'b0))
                        upsample_cnt<=2'b01;
                    else begin
                        upsample_cnt <= (upsample_cnt == 2'b11) ? '0 : (upsample_cnt + 1'b1);
                    end
                end
                else begin
                    upsample_cnt <= upsample_cnt;
                end
            end
        end
    end

    //-------------------------------------------------------------------
    // busy and done signals
    //-------------------------------------------------------------------
    always_comb begin  
        // col_resample_done = (col_ptr[RESAMPLE_IDX_WIDTH-1 : PE_IDX] == resample_input_col_num[RESAMPLE_IDX_WIDTH-1 : PE_IDX]);
        if(row_ptr == resample_input_row_num  && resample_type!=TYPE_IDLE)begin
            if(resample_type == UPSAMPLE_TYPE)begin
                col_resample_done = (upsample_cnt == 2'b11);
            end
            else begin
                col_resample_done = 1'b1;
            end
        end
        else begin
            col_resample_done = 1'b0;
        end
        
        reach_last_col = (col_ptr[RESAMPLE_IDX_WIDTH-1 : PE_IDX] == resample_input_col_num[RESAMPLE_IDX_WIDTH-1 : PE_IDX]);
        resample_done = reach_last_col && col_resample_done && (~resample_input_en);
        resample_done_out = resample_type == UPSAMPLE_TYPE?(resample_done_delay1):(resample_type==TYPE_IDLE?0:resample_done_delay2);
    end

    always_ff @( posedge clk or negedge rst_n) begin : resample_rd_busy_ff
        if(~rst_n)begin
            resample_rd_busy <= 1'b0 ;
        end
        else begin
            if(resample_input_en)begin
                resample_rd_busy <= 1'b1;
            end
            else if(resample_done)begin
                resample_rd_busy <= 1'b0;
            end
            else begin
                resample_rd_busy <= resample_rd_busy;
            end
        end
    end

    always_ff @( posedge clk or negedge rst_n) begin : resample_busy_ff
        if(~rst_n)begin
            resample_busy <= 1'b0 ;
        end
        else begin
            if(resample_input_en)begin
                resample_busy <= 1'b1;
            end
            else if(resample_done_out)begin
                resample_busy <= 1'b0;
            end
            else begin
                resample_busy <= resample_busy;
            end
        end
    end

    //-------------------------------------------------------------------
    // calculate resample input read enable and address
    //-------------------------------------------------------------------
    always_comb begin : resample_spm_rd_en_comb
        case(resample_type)
        UPSAMPLE_TYPE:begin
            resample_spm_rd_en = (resample_input_en | vector_resample_done)&&(!resample_done);
        end
        DOWNSAMPLE_TYPE:begin
            resample_spm_rd_en = resample_rd_busy;
        end
        POOLING_TYPE :begin
            resample_spm_rd_en = resample_rd_busy;
        end
        default:resample_spm_rd_en = 0;
        endcase
    end

    always_ff@(posedge clk or negedge rst_n)begin : resample_spm_rd_addr_ff
        if(~rst_n)begin
            resample_spm_rd_addr <= '0;
        end
        else if(resample_input_en)
            resample_spm_rd_addr <= resample_input_addr;
        else begin
            case(resample_type) // calculate next read address
            UPSAMPLE_TYPE:begin
                if(resample_input_row_num==0 && resample_spm_rd_en)begin
                    resample_spm_rd_addr <= resample_spm_rd_addr+(PE_WIDTH<<PE_DATA_SIZE);
                end 
                else if(resample_spm_rd_en)begin
                    resample_spm_rd_addr <= ((row_ptr == resample_input_row_num-1)&&(~resample_input_en)) ?
                        resample_next_col_rd_addr : (resample_spm_rd_addr + ((resample_input_col_num + 1) << PE_DATA_SIZE));
                end 
                else begin
                    resample_spm_rd_addr <= resample_spm_rd_addr;
                end
            end
            DOWNSAMPLE_TYPE,POOLING_TYPE:begin
                if(resample_input_row_num==0 && resample_spm_rd_en)begin
                    resample_spm_rd_addr <= resample_spm_rd_addr+(PE_WIDTH<<PE_DATA_SIZE);
                end 
                else if(resample_spm_rd_en)begin
                    resample_spm_rd_addr <= (row_ptr == resample_input_row_num) ?
                        resample_next_col_rd_addr : (resample_spm_rd_addr + ((resample_input_col_num + 1) << PE_DATA_SIZE));
                end 
                else begin
                    resample_spm_rd_addr <= resample_spm_rd_addr;
                end
            end
            default:begin
                resample_spm_rd_addr<= resample_input_addr;
            end
            endcase
        end
    end

    always_ff@(posedge clk or negedge rst_n)begin : resample_next_col_rd_addr_ff
        if(~rst_n)begin
            resample_next_col_rd_addr <= '0;
        end
        else begin
            if(resample_input_en)begin
                resample_next_col_rd_addr <= resample_input_addr+ (PE_WIDTH << PE_DATA_SIZE) ;
            end
            else if(resample_spm_rd_en && (row_ptr == '0)&&(col_ptr!=0))begin
                resample_next_col_rd_addr <= resample_next_col_rd_addr + (PE_WIDTH << PE_DATA_SIZE);
            end
            else begin
                resample_next_col_rd_addr <= resample_next_col_rd_addr;
            end
        end
    end
    

    //-------------------------------------------------------------------
    // calculate resample output write enable, data, mask and address 
    //-------------------------------------------------------------------

    always_ff@(posedge clk or negedge rst_n)begin : resample_spm_wr_addranden
        if(~rst_n)begin
            resample_spm_wr_addr    <= '0;
            resample_spm_wr_en      <= '0;
        end
        else begin
            if(resample_input_en)begin
                resample_spm_wr_addr <= resample_output_addr;
            end
            else if(resample_busy)begin
                case(resample_type)

                    DOWNSAMPLE_TYPE : begin
                    
                        
                        if(resample_input_row_num==0 )begin
                            resample_spm_wr_addr    <= resample_spm_wr_addr + (resample_spm_wr_en?((PE_WIDTH>>1) << (PE_DATA_SIZE)):'0); 
                        end     
                        else if(row_ptr_delay == 1'b1 )begin   // next column
                            resample_spm_wr_addr    <= col_ptr_delay==0?resample_spm_wr_addr:resample_next_col_wr_addr;
                        end
                        else if(((row_ptr_delay[0] == 1'b1) | (row_ptr_delay == resample_input_row_num)))begin
                            resample_spm_wr_addr    <= resample_spm_wr_addr + (((resample_input_col_num_ceil + 1)>>1) << (PE_DATA_SIZE));
                        end
                        else begin
                            resample_spm_wr_addr    <= resample_spm_wr_addr;
                        end
                         
                        if(resample_input_row_num==0)
                            resample_spm_wr_en <= resample_spm_rd_en_delay;
                        else if((row_ptr_delay[0] == 1'b1) | (row_ptr_delay == resample_input_row_num))
                            resample_spm_wr_en <= 1'b1;
                        else 
                            resample_spm_wr_en <= 1'b0;

                    end

                    UPSAMPLE_TYPE : begin
                        if(resample_input_en_delay)begin  // keep
                            resample_spm_wr_addr <= resample_spm_wr_addr;
                        end
                        else if((upsample_cnt_delay == 2'b11) && (row_ptr_delay == resample_input_row_num))begin  // next column
                            resample_spm_wr_addr <= resample_next_col_wr_addr;
                        end
                        else if(upsample_cnt_delay[0] == 1'b0)begin  // same row, next vector
                            resample_spm_wr_addr <= resample_spm_wr_addr + (PE_WIDTH << PE_DATA_SIZE);    
                        end
                        else if(reach_last_col && (resample_input_col_num[PE_IDX-1] == 1'b0))begin
                            resample_spm_wr_addr <= resample_spm_wr_addr+ ((resample_input_col_num + 1) << (PE_DATA_SIZE+1));
                        end                        
                        else begin  // next row
                            resample_spm_wr_addr <= resample_spm_wr_addr+ ((resample_input_col_num + 1) << (PE_DATA_SIZE+1))-(PE_WIDTH << PE_DATA_SIZE);
                        end
                        

                        if(~((row_ptr_delay == resample_input_row_num) && (reach_last_col_delay) && (upsample_cnt_delay == 2'b11)))begin
                            resample_spm_wr_en <= 1'b1;
                        end
                        else begin
                            resample_spm_wr_en <= 1'b0;
                        end
                    end

                    POOLING_TYPE : begin
                    
                        if(resample_input_row_num==0 )begin
                            resample_spm_wr_addr    <= resample_spm_wr_addr + (resample_spm_wr_en?
                            (col_ptr_delay==PE_WIDTH?((PE_WIDTH-1) << (PE_DATA_SIZE)):PE_WIDTH << (PE_DATA_SIZE)):'0); 
                        end     
                        else if(row_ptr_delay == 1)begin
                            resample_spm_wr_addr <= col_ptr_delay==0?resample_spm_wr_addr:resample_next_col_wr_addr;
                        end
                        else if(row_ptr_delay!=0) begin
                            resample_spm_wr_addr <= resample_spm_wr_addr + ((resample_input_col_num) << (PE_DATA_SIZE));
                        end   
                        else begin
                            resample_spm_wr_addr <= resample_spm_wr_addr ;
                        end
                         

                        if(resample_input_row_num==0)
                            resample_spm_wr_en <= resample_spm_rd_en_delay;
                        else if(row_ptr_delay == '0)begin
                            resample_spm_wr_en <= 1'b0;
                        end
                        else begin
                            resample_spm_wr_en <= 1'b1;
                        end
                    end

                    default : begin
                        resample_spm_wr_addr <= resample_spm_wr_addr;
                        resample_spm_wr_en <= '0;
                    end
                endcase
            end  
            else begin
                resample_spm_wr_addr <= resample_spm_wr_addr;
                resample_spm_wr_en <= '0;
            end
        end  
    end

    always_ff@(posedge clk or negedge rst_n)begin : resample_next_col_wr_addr_ff
        if(~rst_n)begin
            resample_next_col_wr_addr <= '0;
        end
        else begin
             // update right after the first row is calculated
            case(resample_type)
            
                DOWNSAMPLE_TYPE : begin
                    if(resample_input_en)begin
                        resample_next_col_wr_addr <= resample_output_addr+ ((PE_WIDTH>>1) << (PE_DATA_SIZE));
                    end
                    else if(row_ptr_delay == 1 && col_ptr_delay!=0)begin
                        resample_next_col_wr_addr <= resample_next_col_wr_addr + ((PE_WIDTH>>1) << (PE_DATA_SIZE));
                    end
                    else begin
                        resample_next_col_wr_addr <= resample_next_col_wr_addr;
                    end
                end

                UPSAMPLE_TYPE : begin
                    if(resample_input_en)begin
                        resample_next_col_wr_addr <= resample_output_addr + (PE_WIDTH << (PE_DATA_SIZE + 1));
                    end
                    else if(upsample_cnt==2'b1 && row_ptr == '0 && col_ptr!='0)begin
                        resample_next_col_wr_addr <= resample_next_col_wr_addr + (PE_WIDTH << (PE_DATA_SIZE + 1));
                    end
                    else begin
                        resample_next_col_wr_addr <= resample_next_col_wr_addr; 
                    end
                end

                POOLING_TYPE : begin
                    if(resample_input_en)  begin
                        resample_next_col_wr_addr <= resample_output_addr + ((PE_WIDTH-1) << (PE_DATA_SIZE));
                    end
                    else if(row_ptr_delay == 1 && col_ptr_delay != 0)begin
                        resample_next_col_wr_addr <= resample_next_col_wr_addr + (PE_WIDTH << (PE_DATA_SIZE));
                    end
                    else begin
                        resample_next_col_wr_addr <= resample_next_col_wr_addr;
                    end
                end

                default : begin
                    resample_next_col_wr_addr <= resample_next_col_wr_addr;
                end
            endcase
        end
    end

    always_ff@(posedge clk or negedge rst_n)begin : resample_spm_wr_data_outandmask
        if(~rst_n)begin
            resample_spm_wr_data_out    <= '0;
            resample_spm_wr_mask        <= '0;
        end
        else begin
            if(resample_busy) begin
                case(resample_type)
                    
                    DOWNSAMPLE_TYPE : begin

                        if(row_ptr_delay[0] == 1'b1)begin   // only downsample when reach odd rows or the last row
                                // max
                            if(resample_op == 1'b0)begin
                                for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                    if(reach_last_col_delay && resample_input_col_num[0]==1'b0 && idx==(resample_input_col_num_ceil_half-1))begin
                                        resample_spm_wr_data_out[idx] <= 
                                            max2to1({resample_spm_rd_data_in[idx*2], temp_data_row[idx*2]});
                                    end
                                    else begin
                                        resample_spm_wr_data_out[idx] <= 
                                            max4to1({resample_spm_rd_data_in[idx*2], resample_spm_rd_data_in[idx*2+1],
                                            temp_data_row[idx*2], temp_data_row[idx*2+1]});
                                    end
                                end 
                            end

                            // average
                            else begin
                                for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                    if(reach_last_col_delay && resample_input_col_num[0]==1'b0 && idx==(resample_input_col_num_ceil_half-1))begin
                                        resample_spm_wr_data_out[idx] <= 
                                            avg2to1({resample_spm_rd_data_in[idx*2],temp_data_row[idx*2]});
                                    end
                                    else begin
                                        resample_spm_wr_data_out[idx] <= 
                                            // avg4to1({resample_spm_rd_data_in[idx*2 +: 2],temp_data_row[idx*2 +: 2]});
                                            avg4to1({resample_spm_rd_data_in[idx*2], resample_spm_rd_data_in[idx*2+1],
                                            temp_data_row[idx*2], temp_data_row[idx*2+1]});
                                    end
                                end
                            end

                            // not the last columns

                            if(~reach_last_col_delay)begin
                                resample_spm_wr_mask <= {{(PE_WIDTH/2){1'b0}}, {(PE_WIDTH/2){1'b1}}};
                            end

                            // reach the last column
                            else begin  
                                resample_spm_wr_mask <= {(PE_WIDTH){1'b1}}>>(PE_WIDTH-resample_input_col_num_ceil_half);
                            end
                                        
                        end

                        else if(row_ptr_delay == resample_input_row_num)begin
                            // max
                            if(resample_op == 1'b0)begin
                                for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                    if(reach_last_col_delay && resample_input_col_num[0]==1'b0 && idx==(resample_input_col_num_ceil_half-1))begin
                                        resample_spm_wr_data_out[idx] <= resample_spm_rd_data_in[idx*2];
                                    end
                                    else begin
                                        resample_spm_wr_data_out[idx] <= max2to1({resample_spm_rd_data_in[idx*2], 
                                            resample_spm_rd_data_in[idx*2+1]});
                                    end
                                end
                            end

                            // average
                            else begin
                                for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                    if(reach_last_col_delay && resample_input_col_num[0]==1'b0 && idx==(resample_input_col_num_ceil_half-1))begin
                                        resample_spm_wr_data_out[idx] <= resample_spm_rd_data_in[idx*2];
                                    end
                                    else begin
                                        resample_spm_wr_data_out[idx] <= avg2to1({resample_spm_rd_data_in[idx*2], 
                                            resample_spm_rd_data_in[idx*2+1] });
                                    end
                                end
                            end

                            // not the last columns
                            if(~reach_last_col_delay)begin   
                                resample_spm_wr_mask <= {{(PE_WIDTH/2){1'b0}}, {(PE_WIDTH/2){1'b1}}};
                            end

                            // reach the last column
                            else begin  
                                resample_spm_wr_mask <= {(PE_WIDTH){1'b1}}>>(PE_WIDTH-resample_input_col_num_ceil_half);
                                
                            end
                            end
                        else begin
                            resample_spm_wr_data_out    <= resample_spm_wr_data_out;
                            resample_spm_wr_mask        <= '0;
                        end

                    end

                    UPSAMPLE_TYPE : begin   // only support nearest now, bilinear not supported
                        
                        // nearest
                        if(resample_op == 1'b0)begin
                            if(upsample_cnt[0] == 1'b0 || (reach_last_col&&upsample_cnt[0] == 1'b1&&(resample_input_col_num[PE_IDX-1] == 1'b0)))begin
                                for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                    resample_spm_wr_data_out[(idx<<1)] <= {resample_spm_rd_data_in[idx]};
                                    resample_spm_wr_data_out[(idx<<1)+1] <= {resample_spm_rd_data_in[idx]};
                                end
                            end
                            else begin
                                for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                    resample_spm_wr_data_out[(idx<<1)] <= {resample_spm_rd_data_in[idx + PE_WIDTH/2]};
                                    resample_spm_wr_data_out[(idx<<1)+1] <= {resample_spm_rd_data_in[idx + PE_WIDTH/2]};
                                end
                            end

                             // not the last columns
                            if(~(reach_last_col|reach_last_col_delay))begin
                                resample_spm_wr_mask <= {PE_WIDTH{1'b1}};
                            end

                            // reach the last column
                            else begin
                                if(resample_input_col_num[PE_IDX-1] == 1'b1)begin   // long tail, upsample 4 times
                                    if(upsample_cnt[0] == 1'b0)begin
                                        resample_spm_wr_mask <= {PE_WIDTH{1'b1}};
                                    end
                                    else begin
                                        for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                            if(idx <= resample_input_col_num[PE_IDX-2:0])begin
                                                resample_spm_wr_mask[(idx<<1) +: 2] <= 2'b11;
                                            end
                                            else begin
                                                resample_spm_wr_mask[(idx<<1) +: 2] <= 2'b0;
                                            end
                                        end
                                    end
                                end
                                else begin  // short tail
                                    if(upsample_cnt[0]==1'b1)begin
                                        for(int idx = 0; idx < PE_WIDTH/2; idx++)begin
                                            if(idx <= resample_input_col_num[PE_IDX-2:0])begin
                                                resample_spm_wr_mask[(idx<<1) +: 2] <= 2'b11;
                                            end
                                            else begin
                                                resample_spm_wr_mask[(idx<<1) +: 2] <= 2'b0;
                                            end
                                        end
                                    end
                                    else begin
                                        resample_spm_wr_mask <= {PE_WIDTH{1'b0}};
                                    end
                                end
                            end
                        end

                        // bilienar, not supported currently
                        else begin
                            resample_spm_wr_data_out    <= '0;
                            resample_spm_wr_mask        <= '0;
                        end
                    end

                    POOLING_TYPE : begin
                        // max
                        if(resample_op == 1'b0)begin
                            if(row_ptr_delay != '0 && resample_input_row_num!=0)begin    // not the first row and row number !=0 
                                if(col_ptr_delay == '0)begin     // first column
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx] <= 
                                            max4to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1],
                                            temp_data_row[idx], temp_data_row[idx+1]});
                                    end
                                    resample_spm_wr_data_out[PE_WIDTH-1] <= resample_spm_wr_data_out[PE_WIDTH-1];
                                end
                                else begin  // middle or last column
                                    resample_spm_wr_data_out[0] <= 
                                        // max4to1({temp_data_col[temp_col_rd_ptr +: 2],
                                        // resample_spm_rd_data_in[0], temp_data_row[0]});
                                        max4to1({temp_data_col[temp_col_rd_ptr], temp_data_col[temp_col_rd_ptr+1], resample_spm_rd_data_in[0], temp_data_row[0]});
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx+1] <= 
                                            max4to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1],
                                            temp_data_row[idx], temp_data_row[idx+1]});
                                    end
                                end
                            end
                            else if(resample_input_row_num==0)begin
                                if(col_ptr_delay == '0)begin     // first column
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx] <= 
                                            max2to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1]});
                                    end
                                    resample_spm_wr_data_out[PE_WIDTH-1] <= resample_spm_wr_data_out[PE_WIDTH-1];
                                end
                                else begin  // middle or last column
                                    resample_spm_wr_data_out[0] <= 
                                        // max2to1({temp_data_col[temp_col_rd_ptr +: 1],
                                        // resample_spm_rd_data_in[0]});
                                        max2to1({temp_data_col[temp_col_rd_ptr],resample_spm_rd_data_in[0]});
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx+1] <= 
                                            max2to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1]});
                                    end
                                end
                            end
                            else begin
                                resample_spm_wr_data_out <= resample_spm_wr_data_out;
                            end
                        end

                        // average
                        else begin
                            if(row_ptr_delay != '0 && resample_input_row_num!=0)begin    // not the first row
                                if(col_ptr_delay == '0)begin     // first column
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx] <= 
                                            avg4to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1], 
                                            temp_data_row[idx], temp_data_row[idx+1]});
                                    end
                                    resample_spm_wr_data_out[PE_WIDTH-1] <= resample_spm_wr_data_out[PE_WIDTH-1];
                                end
                                else begin  // middle or last column
                                    resample_spm_wr_data_out[0] <= 
                                        // avg4to1({temp_data_col[temp_col_rd_ptr +: 2],
                                        avg4to1({temp_data_col[temp_col_rd_ptr], temp_data_col[temp_col_rd_ptr+1],
                                        resample_spm_rd_data_in[0], temp_data_row[0]});
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx+1] <= 
                                            avg4to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1],
                                            temp_data_row[idx], temp_data_row[idx+1]});
                                    end
                                end
                            end
                            else if(resample_input_row_num==0)begin
                                if(col_ptr_delay == '0)begin     // first column
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx] <= 
                                            avg2to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1]});
                                    end
                                    resample_spm_wr_data_out[PE_WIDTH-1] <= resample_spm_wr_data_out[PE_WIDTH-1];
                                end
                                else begin  // middle or last column
                                    resample_spm_wr_data_out[0] <= 
                                        avg2to1({temp_data_col[temp_col_rd_ptr],
                                        resample_spm_rd_data_in[0]});
                                    for(int idx = 0; idx < PE_WIDTH-1; idx++)begin
                                        resample_spm_wr_data_out[idx+1] <= 
                                            avg2to1({resample_spm_rd_data_in[idx], resample_spm_rd_data_in[idx+1]});
                                    end
                                end
                            end
                            else begin
                                resample_spm_wr_data_out <= resample_spm_wr_data_out;
                            end
                        end

                    
                        
                        // not the last columns
                        if(~reach_last_col_delay)begin   
                            if(col_ptr_delay == '0)begin
                                resample_spm_wr_mask <= {1'b0,{(PE_WIDTH-1){1'b1}}};
                            end
                            else begin
                                resample_spm_wr_mask <= {(PE_WIDTH){1'b1}};
                            end
                        end

                        // reach the last column
                        else begin  
                            for(int idx = 0; idx < PE_WIDTH; idx++)begin
                                if(idx < (resample_input_col_num[PE_IDX-1:0]+(col_ptr_delay==0?0:1)))begin
                                    resample_spm_wr_mask[idx] <= 1'b1;
                                end
                                else begin
                                    resample_spm_wr_mask[idx] <= 1'b0;
                                end
                            end
                        end
                    end

                    default : begin
                        resample_spm_wr_data_out    <= '0;
                        resample_spm_wr_mask        <= '0;
                    end

                endcase
                end
            else begin
                resample_spm_wr_data_out    <= '0;
                resample_spm_wr_mask        <= '0;
            end
        end
    end

    always_comb begin
        if(resample_input_col_num[0] == 1'b0)begin
            resample_input_col_num_ceil = resample_input_col_num + 1'b1;
        end
        else begin
            resample_input_col_num_ceil = resample_input_col_num;
        end
    end
    always_comb begin
        resample_input_col_num_ceil_half=(resample_input_col_num_ceil[PE_IDX-1:0]+1)>>1;
    end

    function logic [PE_DATA_WIDTH-1:0] max4to1(
        logic [3:0][PE_DATA_WIDTH-1:0] data_in
    );

        logic [1:0][PE_DATA_WIDTH-1:0] max_temp;
        
        // if (data_in[0] >= data_in[1]) begin 
        //     max_temp[0] = data_in[0];
        // end
        // else begin
        //     max_temp[0] = data_in[1];
        // end

        // if (data_in[2] >= data_in[3]) begin
        //     max_temp[1] = data_in[2];
        // end
        // else begin
        //     max_temp[1] = data_in[3];
        // end
        
        // if (max_temp[0] >= max_temp[1]) begin
        //     return max_temp[0];
        // end
        // else begin
        //     return max_temp[1];
        // end   

        max_temp[0] = max2to1(data_in[1:0]);
        max_temp[1] = max2to1(data_in[3:2]);
        return max2to1(max_temp[1:0]);

    endfunction : max4to1

    function logic [PE_DATA_WIDTH-1:0] avg4to1(
        logic [3:0][PE_DATA_WIDTH-1:0] data_in
    );
        logic [PE_DATA_WIDTH+1:0] sum;

        sum = (data_in[0] + data_in[1]) + (data_in[2] + data_in[3]);

        if(sum[1] == 1'b0)begin
            return (sum >> 2);
        end
        else begin
            return ((sum >> 2) + 1'b1);
        end

    endfunction : avg4to1

    function logic [PE_DATA_WIDTH-1:0] max2to1(
        logic [1:0][PE_DATA_WIDTH-1:0] data_in
    );
        
        if (data_in[0] >= data_in[1]) begin 
            return  data_in[0];
        end
        else begin
            return  data_in[1];
        end

    endfunction : max2to1

    function logic [PE_DATA_WIDTH-1:0] avg2to1(
        logic [1:0][PE_DATA_WIDTH-1:0] data_in
    );
        logic [PE_DATA_WIDTH:0] sum;

        sum = (data_in[0] + data_in[1]);

        if(sum[0] == 1'b0)begin
            return (sum >> 1);
        end
        else begin
            return ((sum >> 1) + 1'b1);
        end

    endfunction :avg2to1

endmodule