module softmax_approx #(
    parameter GROUP_SIZE     = 32,
    parameter IN_DATA_WIDTH  = 16,
    parameter IN_FRAC_WIDTH  = 8,
    parameter OUT_DATA_WIDTH = 16,
    parameter OUT_FRAC_WIDTH = 15 ,
    parameter MAX_LENGTH     = 1024,
    parameter SHIFT_WIDTH    = 4,
    parameter SHIFT_DATA_WIDTH =8,
    localparam REAL_IDX_WIDTH = $clog2(MAX_LENGTH/GROUP_SIZE)+1
)(
    input  logic                                          clk,
    input  logic                                          rst_n,
    input  logic [GROUP_SIZE-1:0][IN_DATA_WIDTH-1:0]      data_in,
    input  logic                                          in_valid,
    output logic [GROUP_SIZE-1:0][OUT_DATA_WIDTH-1:0]     data_out,
    output logic                                          out_valid,
    //input  logic                                          flush,
    input  logic [REAL_IDX_WIDTH-1:0]                     in_num          //in_data_num/group size                      
);
localparam GROUP_FIFO_SIZE = MAX_LENGTH/GROUP_SIZE;
localparam MAXLEN_IDX = $clog2(MAX_LENGTH);
localparam MG_DIV_IDX = $clog2(MAX_LENGTH/GROUP_SIZE);
localparam FIFO_SIZE_IDX = $clog2(GROUP_FIFO_SIZE);
localparam GROUP_IDX = $clog2(GROUP_SIZE);
localparam ACCU_OUT_WIDTH = MG_DIV_IDX+SHIFT_WIDTH+GROUP_IDX;
localparam MASKSUM_WIDTH =GROUP_IDX+MG_DIV_IDX+1;
localparam MAXCOMP_WIDTH = IN_DATA_WIDTH-IN_FRAC_WIDTH ;
localparam CODE_DATA_WIDTH = 16;
localparam MAX_PART_NUM = 8;
localparam MAX_PART = GROUP_SIZE/MAX_PART_NUM;
logic [IN_DATA_WIDTH-1:0] global_max_pre;
logic [IN_DATA_WIDTH-1:0] global_max;
logic [IN_DATA_WIDTH-1:0] group_max;
logic [MAXCOMP_WIDTH-1:0] local_max;
logic [2:0][GROUP_SIZE-1:0][IN_DATA_WIDTH-1:0]data_in_buf;
logic [MAX_PART_NUM-1:0][MAXCOMP_WIDTH-1:0]  max_part;
logic [MAX_PART_NUM-1:0][MAXCOMP_WIDTH-1:0]  max_part_q;
logic [MAX_PART_NUM/MAX_PART-1:0][MAXCOMP_WIDTH-1:0]max_temp_q;
logic [MAXCOMP_WIDTH-1:0]                    group_max_comb;
logic                                        part_max_valid;
logic                                        group_max_valid;
logic                                        mxsub_valid;
logic [GROUP_SIZE-1:0][IN_DATA_WIDTH-1:0]    mxsub           ;
logic [IN_DATA_WIDTH-1:0]                    mm_sub          ;
logic [IN_DATA_WIDTH:0]                      mmsub_log2e     ;
logic [$clog2(IN_DATA_WIDTH)-1:0]            mmsub_shift     ;
logic [IN_DATA_WIDTH-IN_FRAC_WIDTH:0]        mmsub_round     ;
logic [$clog2(IN_DATA_WIDTH)-1:0]            mmsub_accu      ;
logic [GROUP_SIZE-1:0][SHIFT_DATA_WIDTH-1:0] shift_out       ;
logic [GROUP_SIZE-1:0][SHIFT_WIDTH-1:0 ]     shift_amount    ;
logic                                        log2exp_out_valid ; 
logic                                        adder_out_valid;
logic [SHIFT_WIDTH+GROUP_IDX-1:0]            adder_data_out_comb;
logic [SHIFT_WIDTH+GROUP_IDX-1:0]            adder_data_out;
logic [ACCU_OUT_WIDTH-1:0]                   accu_data_out;
logic                                        accu_out_valid;
logic                                        lod_out_valid;
logic [$clog2(ACCU_OUT_WIDTH)-1:0]           pos_out;
logic [IN_DATA_WIDTH-1:0]                    code;
logic [REAL_IDX_WIDTH-2:0]                   out_count;
logic                                        output_gen_in_valid;
logic                                        fifo_pop;
logic  [GROUP_SIZE-1:0][SHIFT_WIDTH-1:0]     mx_shift;
logic                                        buf_valid;

always_ff @(posedge clk ) begin
    buf_valid      <=in_valid;
    part_max_valid <=buf_valid;
    group_max_valid<=part_max_valid;
    mxsub_valid    <=group_max_valid;
    adder_out_valid<=log2exp_out_valid;
    output_gen_in_valid<=fifo_pop;
end
always_comb begin
    for(int p=0;p<MAX_PART_NUM;p++)begin
        max_part[p] = data_in_buf[0][p*MAX_PART][IN_DATA_WIDTH-1:IN_FRAC_WIDTH];
        for(int i=1;i<MAX_PART;i++)begin        //reduce compare width
            if($signed(data_in_buf[0][p*MAX_PART+i][IN_DATA_WIDTH-1:IN_FRAC_WIDTH])>$signed(max_part[p]))begin
                max_part[p] = data_in_buf[0][p*MAX_PART+i][IN_DATA_WIDTH-1:IN_FRAC_WIDTH];
            end
        end
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)
    max_part_q<={1'b1,{(IN_DATA_WIDTH-1){1'b0}}};
    else if(buf_valid)
    max_part_q<=max_part;
    else 
    max_part_q<=max_part_q;
end
always_comb begin
    for(int p=0;p<MAX_PART_NUM/MAX_PART;p++)begin
        max_temp_q[p] = max_part_q[p*MAX_PART];
        for(int i=1;i<MAX_PART;i++)begin        //reduce compare width
            if($signed(max_part_q[p*MAX_PART+i])>$signed(max_temp_q[p]))begin
                max_temp_q[p] = max_part_q[p*MAX_PART+i];
            end
        end
    end
    group_max_comb = max_temp_q[0];
    for(int i=1;i<MAX_PART_NUM/MAX_PART;i++)begin        //reduce compare width
        if($signed(max_temp_q[i])>$signed(group_max_comb))begin
            group_max_comb = max_temp_q[i];
        end
    end
    group_max = {group_max_comb,{(IN_FRAC_WIDTH){1'b1}}};
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)
    global_max<={1'b1,{(IN_DATA_WIDTH-1){1'b0}}};
    else if(out_valid && (~output_gen_in_valid))
    global_max<={1'b1,{(IN_DATA_WIDTH-1){1'b0}}};
    else if(part_max_valid)begin
    global_max<=$signed(global_max)>$signed(group_max)?global_max:group_max;
    end
    else 
    global_max<=global_max;
end
always_ff @(posedge clk or negedge rst_n)begin
    if(rst_n==0)begin
        data_in_buf<='0;
    end
    else begin 
        if(in_valid) 
            data_in_buf[0]<=data_in;
        else 
            data_in_buf[0]<=data_in_buf[0];
        if(buf_valid) 
            data_in_buf[1]<=data_in_buf[0];
        else 
            data_in_buf[1]<=data_in_buf[1];
        if(part_max_valid) 
            data_in_buf[2]<=data_in_buf[1];
        else 
            data_in_buf[2]<=data_in_buf[2];
    end 
end
always_ff @( posedge clk or negedge rst_n ) begin 
    if(rst_n==0)
    global_max_pre<={1'b1,{(IN_DATA_WIDTH-1){1'b0}}};
    else if(group_max_valid)
    global_max_pre<=global_max;
    else 
    global_max_pre<={1'b1,{(IN_DATA_WIDTH-1){1'b0}}};
end
assign mm_sub = global_max_pre=={1'b1,{(IN_DATA_WIDTH-1){1'b0}}}?'0:$signed(global_max)-$signed(global_max_pre);
always_ff @( posedge clk or negedge rst_n ) begin 
    if(rst_n==0)begin
    mxsub           <='0;
    mmsub_log2e     <='0;
    end
    else if(group_max_valid)begin
        for(int i = 0;i<GROUP_SIZE;i++)begin
            mxsub[i]<=$signed(global_max)-$signed(data_in_buf[2][i]);
        end
    mmsub_log2e     <= mm_sub+(mm_sub>>1)-(mm_sub>>4);
    end
    else begin
    mxsub           <=mxsub;
    mmsub_log2e     <=mmsub_log2e;
    end
end
log2exp #(
    .FRAC_WIDTH(IN_FRAC_WIDTH),  
    .DATA_WIDTH(IN_DATA_WIDTH),
    .GROUP_SIZE(GROUP_SIZE),
    .SHIFT_WIDTH(SHIFT_WIDTH),
    .SHIFT_DATA_WIDTH(SHIFT_DATA_WIDTH)  
) u_log2exp (
    .clk         (clk),        
    .rst_n       (rst_n),      
    .in_valid    (mxsub_valid),    
    .in_num      (mxsub),   
    .out_num     (shift_amount),        
    .shift_out   (shift_out),       
    .out_valid   (log2exp_out_valid)   
);
always_ff@(posedge clk or negedge rst_n) begin 
    if(rst_n==0)
    mmsub_round <='0;
    else if(mxsub_valid)
    mmsub_round <= mmsub_log2e[IN_FRAC_WIDTH-1]?(mmsub_log2e[IN_DATA_WIDTH:IN_FRAC_WIDTH]+1):(mmsub_log2e[IN_DATA_WIDTH:IN_FRAC_WIDTH]); 
    else 
    mmsub_round <= mmsub_round;
end
always_ff@( posedge clk or negedge rst_n )begin
    if(rst_n==0)
    mmsub_shift   <='0;
    else if(log2exp_out_valid)begin
        if(IN_DATA_WIDTH-IN_FRAC_WIDTH<$clog2(IN_DATA_WIDTH))
        mmsub_shift <= mmsub_round;
        else 
        mmsub_shift   <=(|mmsub_round[IN_DATA_WIDTH-IN_FRAC_WIDTH:$clog2(IN_DATA_WIDTH)])?{($clog2(IN_DATA_WIDTH)){1'b1}}:mmsub_round[$clog2(IN_DATA_WIDTH)-1:0];
    end
    else 
    mmsub_shift   <= mmsub_shift;
end
always_comb begin
    adder_data_out_comb = '0;
    for(int i=0;i<GROUP_SIZE;i++)begin
        adder_data_out_comb = adder_data_out_comb + (log2exp_out_valid?shift_out[i]:'0);
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)
    adder_data_out<='0;
    else if(log2exp_out_valid)
    adder_data_out<=adder_data_out_comb;
    else 
    adder_data_out<=adder_data_out;
end
accu #(
    .GROUP_SIZE(GROUP_SIZE),
    .SHIFT_DATA_WIDTH(SHIFT_DATA_WIDTH),
    .MAX_LENGTH(MAX_LENGTH),
    .REAL_IDX_WIDTH(REAL_IDX_WIDTH),
    .DATA_WIDTH(IN_DATA_WIDTH),
    .IN_WIDTH(SHIFT_WIDTH+GROUP_IDX)
)u_accu (
    .clk       (clk),        
    .rst_n     (rst_n),      
    .data_in   (adder_data_out),    
    .in_valid  (adder_out_valid), 
    .mmsub     (mmsub_shift) , 
    .in_num    (in_num),     
    .data_out  (accu_data_out),   
    .out_valid (accu_out_valid)   
);
lod #(
    .IN_WIDTH(ACCU_OUT_WIDTH),
    .GROUP_SIZE(GROUP_SIZE),
    .SHIFT_DATA_WIDTH(SHIFT_DATA_WIDTH),
    .DATA_WIDTH(CODE_DATA_WIDTH)
) u_lod (
    .clk       (clk),        
    .rst_n     (rst_n),      
    .in_valid  (accu_out_valid),   
    .data_in   (accu_data_out),    
    .code      (code),       //(1,15)(int,frac)
    .out_valid (lod_out_valid)   
);

fifo_g #(
    .DATA_WIDTH(MAXCOMP_WIDTH),
    .DEPTH(GROUP_FIFO_SIZE)
) u_fifo_max (
    .clk_i            (clk),
    .rst_n_i          (rst_n),
    .fifo_push_data_i (global_max[IN_DATA_WIDTH-1:IN_FRAC_WIDTH]),
    .fifo_push_i      (group_max_valid),
    .fifo_pop_data_o  (local_max),
    .fifo_pop_i       (fifo_pop),
    .fifo_ready_o     (),
    .fifo_valid_o     ()
);

fifo_g #(
    .DATA_WIDTH(SHIFT_WIDTH*GROUP_SIZE),
    .DEPTH(GROUP_FIFO_SIZE)
) u_fifo_shift (
    .clk_i            (clk),
    .rst_n_i          (rst_n),
    .fifo_push_data_i (shift_amount),
    .fifo_push_i      (log2exp_out_valid),
    .fifo_pop_data_o  (mx_shift),
    .fifo_pop_i       (fifo_pop),
    .fifo_ready_o     (),
    .fifo_valid_o     ()
);
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)
    out_count<='0;
    else if(out_count == in_num)
    out_count<='0;
    else if(fifo_pop)
    out_count<=out_count+1;
    else
    out_count<=out_count;  
end
always_ff @(posedge clk or negedge rst_n) begin
    if(rst_n==0)
    fifo_pop<='0;
    else if(accu_out_valid)
    fifo_pop<='1;
    else if(out_count==in_num-1)
    fifo_pop<='0;
    else 
    fifo_pop<=fifo_pop;
end
output_gen #(
    .GROUP_SIZE      (GROUP_SIZE      )  ,
    .SHIFT_WIDTH     (SHIFT_WIDTH     )  ,
    .IN_DATA_WIDTH   (MAXCOMP_WIDTH   )  ,
    .CODE_DATA_WIDTH (CODE_DATA_WIDTH )  ,
    .OUT_DATA_WIDTH  (OUT_DATA_WIDTH  ) 
) u_output_gen (
    .clk          (clk),          
    .rst_n        (rst_n),        
    .mx_shift     (mx_shift),     
    .local_max    (local_max),    
    .global_max   (global_max[IN_DATA_WIDTH-1:IN_FRAC_WIDTH]),   
    .code         (code),           
    .in_valid     (output_gen_in_valid),  
    .out_valid    (out_valid),    
    .out_data     (data_out)      
);

endmodule
