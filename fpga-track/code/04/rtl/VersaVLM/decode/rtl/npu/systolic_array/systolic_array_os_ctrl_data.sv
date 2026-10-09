module systolic_array_os_ctrl_data #(
    parameter PE_FPGA_DSP       = 0,
    parameter PE_DATA_WIDTH_IN  = 8,
    parameter PE_DATA_WIDTH_OUT = 32,
    parameter INPUT_WIDTH_MAX   = 4096,
    parameter INPUT_HEIGHT_MAX  = 4096,
    parameter ARRAY_WIDTH       = 16,
    parameter ARRAY_HEIGHT      = 16,
    parameter integer DSP_PE_NUM = (PE_FPGA_DSP ? ARRAY_WIDTH * ARRAY_HEIGHT : 0),
    localparam ARRAY_WIDTH_CLOG2 =$clog2(ARRAY_WIDTH),
    localparam ARRAY_HEIGHT_CLOG2=$clog2(ARRAY_HEIGHT)       
)(
    input   logic   clk_i,          
    
    input   logic   rstn_i,    
    input   logic   systolic_array_start_i,  
 
    input   logic signed [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] active_row,
    input   logic signed [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0] weight_column_i,

    input   logic [ARRAY_HEIGHT_CLOG2:0]       input_a_row_num,
    input   logic [$clog2(INPUT_WIDTH_MAX):0]  input_a_col_num,
    input   logic [$clog2(INPUT_HEIGHT_MAX):0] input_b_row_num,
    input   logic [ARRAY_WIDTH_CLOG2:0]        input_b_col_num,

    output  logic addr_leap_enable,
    output  logic [$clog2(ARRAY_WIDTH+INPUT_HEIGHT_MAX+ARRAY_WIDTH):0]addr_leap,

    output  logic signed [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_OUT-1:0] result_os_o       ,
    output  logic systolic_array_busy_o,             //the systolic array starts to calculate and clear the mac_res_o in every PE
    output  logic systolic_array_os_wr_en_o,
    output  logic systolic_array_done_o          //the systolic array has finished the calculation
);
           
    
    logic  [$clog2(ARRAY_WIDTH+INPUT_HEIGHT_MAX+ARRAY_WIDTH):0] cycle_cnt;
    logic  [$clog2(ARRAY_WIDTH+INPUT_HEIGHT_MAX+ARRAY_WIDTH)-ARRAY_WIDTH_CLOG2:0] cycle_cnt_high_active;
    logic [ARRAY_WIDTH_CLOG2-1:0] active_cycle_cnt;
    logic [ARRAY_WIDTH_CLOG2:0] input_a_col_num_clamped; //min(col_num,array-width)
    logic [$clog2(INPUT_WIDTH_MAX):0] active_row_valid_num;
    logic [ARRAY_HEIGHT_CLOG2-1:0] weight_cycle_cnt;
    logic [ARRAY_HEIGHT_CLOG2:0] input_b_row_num_clamped; //min(row_num,array-height)     

    logic signed [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0] active_i ;
    logic signed [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0]  weight_i ;
    logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] active_reg ;
    logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] weight_reg ;


    logic signed [ARRAY_HEIGHT-1:0][ARRAY_WIDTH-1:0][PE_DATA_WIDTH_OUT-1:0] result_o_w ;
    logic signed [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] active_row_i;
`ifndef SYNTHESIS
    logic [15:0] dbg_wr_cnt;
`endif

    always_comb begin :gemmconvset
        active_cycle_cnt        = cycle_cnt[ARRAY_WIDTH_CLOG2-1:0];
        cycle_cnt_high_active   = cycle_cnt[$clog2(ARRAY_WIDTH+INPUT_HEIGHT_MAX+ARRAY_WIDTH):ARRAY_WIDTH_CLOG2];
        input_a_col_num_clamped = input_a_col_num<ARRAY_WIDTH?input_a_col_num:ARRAY_WIDTH;
        weight_cycle_cnt        = cycle_cnt[ARRAY_HEIGHT_CLOG2-1:0];
        input_b_row_num_clamped = input_b_row_num<ARRAY_HEIGHT?input_b_row_num:ARRAY_HEIGHT;
        if(input_a_col_num[ARRAY_WIDTH_CLOG2-1:0]!=0 && input_a_col_num[ARRAY_WIDTH_CLOG2-1:0]<input_a_row_num)begin
            active_row_valid_num= (input_a_col_num[$clog2(INPUT_WIDTH_MAX):ARRAY_WIDTH_CLOG2]<<ARRAY_WIDTH_CLOG2)+input_a_row_num;
        end
        else begin
            active_row_valid_num = input_a_col_num;
        end
        for(int k=0;k<ARRAY_WIDTH;k++)begin
            if(cycle_cnt_high_active==input_a_col_num[$clog2(INPUT_WIDTH_MAX):ARRAY_WIDTH_CLOG2]&& input_a_col_num[ARRAY_WIDTH_CLOG2-1:0]!=0)
                active_row_i[k] = k<input_a_col_num[ARRAY_WIDTH_CLOG2-1:0]?active_row[k]:'0;
            else 
                active_row_i[k] = active_row[k];
        end
        addr_leap_enable        = active_cycle_cnt==ARRAY_WIDTH-2 && input_a_col_num>ARRAY_WIDTH;
        addr_leap               = (cycle_cnt_high_active+1)<<($clog2(ARRAY_WIDTH));
    end
    systolic_array_os #(
        .PE_FPGA_DSP       ( PE_FPGA_DSP       ),
        .DSP_PE_NUM        ( DSP_PE_NUM        ),
        .PE_DATA_WIDTH_IN  ( PE_DATA_WIDTH_IN  ),
        .PE_DATA_WIDTH_OUT ( PE_DATA_WIDTH_OUT ),
        .ARRAY_WIDTH       ( ARRAY_WIDTH       ),
        .ARRAY_HEIGHT      ( ARRAY_HEIGHT      ))
     u_systolic_array (
        .clk_i        ( clk_i        ),
        .rstn_i       ( rstn_i       ),

        .ctrl_start_i ( systolic_array_start_i),
       
        .active_i     ( active_i     ),
        .weight_i     ( weight_i     ),

        .result_o_w   ( result_o_w   )
    );

    

    typedef enum logic [1:0] {
        IDLE                  = 2'd0,     
        WAIT_FOR_PREPARE      = 2'd1,
        ROLLING               = 2'd3  
    } state_t;

    state_t current_state, next_state;

    
    always @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    always_comb begin
        case (current_state)
            IDLE: begin
                if (systolic_array_start_i)
                    next_state = WAIT_FOR_PREPARE;
                else
                    next_state = IDLE;
            end

            WAIT_FOR_PREPARE: begin
                    next_state = ROLLING;
            end

            ROLLING: begin        
                if (systolic_array_done_o)  
                    next_state = IDLE;
                else
                    next_state = ROLLING;
            end
            default: next_state = IDLE;
        endcase
    end

   
    integer i,j;
     
    always @(posedge clk_i or negedge rstn_i) begin
        if (~rstn_i) begin
            active_i         <= '0;
            weight_i         <= '0;
            active_reg       <= '0;
            weight_reg       <= '0;            
            result_os_o      <= '0;           
            cycle_cnt        <= '0;          
            systolic_array_done_o       <= 1'b0;
            systolic_array_os_wr_en_o   <= 1'b0;    
`ifndef SYNTHESIS
            dbg_wr_cnt                  <= '0;
`endif
           

        end else begin
            case (next_state)
            //case (current_state)
                IDLE: begin
                    active_i         <= '0;
                    weight_i         <= '0;
                    active_reg       <= '0;
                    weight_reg       <= '0;            
                    result_os_o      <= '0;           
                    cycle_cnt        <= '0;       
                    systolic_array_done_o       <= 1'b0;
                    systolic_array_os_wr_en_o   <= 1'b0;   
`ifndef SYNTHESIS
                    dbg_wr_cnt                  <= '0;
`endif
                   
                end



                WAIT_FOR_PREPARE: begin   //in this state,pe clear the reg in it
                end
                
                ROLLING: begin
                    cycle_cnt <= cycle_cnt+1;//when pe_os_enable_i == 1, cycle_cnt is same to cycle_cnt,
                  

                    for (int i = 0; i < ARRAY_HEIGHT; i=i+1)  begin 
                        // each cycle, load one row to active-reg, in this row,all data is active-row-i
                        if (active_cycle_cnt == i && active_cycle_cnt<input_a_row_num )
                            active_reg[i] <= cycle_cnt<active_row_valid_num?active_row_i:'0;
                    end
             

                    for (int i = 0; i < ARRAY_WIDTH; i=i+1) begin
                        if (weight_cycle_cnt == i)
                            weight_reg[i] <= cycle_cnt<input_b_row_num?weight_column_i:'0; 
                    end    
                    

                    for (int i = 0; i < ARRAY_HEIGHT; i=i+1) begin
                        if ((active_cycle_cnt == i) && (active_cycle_cnt < input_a_row_num))begin   
                            active_i[i] <= cycle_cnt<active_row_valid_num?active_row_i[0]:'0;
                        end else if ((i < active_cycle_cnt) && (active_cycle_cnt-i <input_a_col_num_clamped))begin
                            active_i[i] <= active_reg[i][active_cycle_cnt-i];
                        end else if (i > active_cycle_cnt )begin
                            active_i[i] <= active_reg[i][active_cycle_cnt+ARRAY_WIDTH-i];
                        end else begin
                            active_i[i] <= 'b0;
                        end
                    end
                    for (int i = 0; i < ARRAY_WIDTH; i=i+1) begin
                        if (i < input_b_col_num) begin                
                            if ( i == 0 )begin    
                                weight_i[i] <= cycle_cnt<input_b_row_num?weight_column_i[0]:'0;
                            end else if((weight_cycle_cnt - i >= 0) && (weight_cycle_cnt - i < input_b_row_num_clamped))begin
                                weight_i[i] <= weight_reg[weight_cycle_cnt-i][i];
                            end else if( i>weight_cycle_cnt )begin
                                weight_i[i]<=weight_reg[weight_cycle_cnt+ARRAY_HEIGHT-i][i];
                            end
                            else begin
                                weight_i[i] <= 'b0;
                            end
                        end
                    end 

                    
                    //load the output of each line of systolic array into the output
                    for (i = 0; i < ARRAY_HEIGHT; i=i+1)  begin
                        if ( i < input_a_row_num && cycle_cnt == i+input_b_col_num+input_a_col_num) begin
                            result_os_o  <= result_o_w[i];
                        end
                    end

                            
                   
                    if ((input_b_col_num+input_a_col_num-1 <= cycle_cnt)&&(cycle_cnt< input_a_row_num+input_b_col_num+input_a_col_num-1 ))
                        systolic_array_os_wr_en_o<='b1;
                    else
                        systolic_array_os_wr_en_o<='b0;

`ifndef SYNTHESIS
                    if ((input_b_col_num+input_a_col_num-1 <= cycle_cnt)&&(cycle_cnt< input_a_row_num+input_b_col_num+input_a_col_num-1 ))
                        dbg_wr_cnt <= dbg_wr_cnt + 1'b1;
`endif
                    
                    
                    if (cycle_cnt == input_a_row_num+input_b_col_num+input_a_col_num-1) begin
                            systolic_array_done_o <= 1'b1;
                    end
                    else
                            systolic_array_done_o <= 1'b0;
                    end
                default: begin
                    cycle_cnt <= '0;
                    systolic_array_done_o <= 1'b0;
                end
            endcase
    
        end
    end
    assign    systolic_array_busy_o     = (current_state != IDLE);

endmodule
