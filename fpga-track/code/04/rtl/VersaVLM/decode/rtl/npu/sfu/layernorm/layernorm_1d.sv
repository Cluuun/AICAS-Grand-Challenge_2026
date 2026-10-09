module layernorm_1d #(
    parameter MAX_LENGTH  = 1024,
    parameter GROUP_SIZE  = 32,
    parameter IDATA_WIDTH = 16,
    parameter IFRAC_WIDTH = 10 ,
    parameter ODATA_WIDTH = 16,
    parameter OFRAC_WIDTH = 10,
    localparam NUM_WIDTH = $clog2(MAX_LENGTH)+1
) (
    input  logic                          clk      ,
    input  logic                          rst_n     ,

    input  logic [NUM_WIDTH-1:0]          in_real_num,
    input  logic                          in_valid_i ,
    input  logic [GROUP_SIZE-1:0][IDATA_WIDTH-1:0] in_data_i ,

    output logic                          out_valid_o,
    output logic [GROUP_SIZE-1:0][ODATA_WIDTH-1:0] out_data_o 
    //output logic                          out_done_o 
);

localparam NUM_GROUPS = MAX_LENGTH / GROUP_SIZE;
localparam MEAN_ADDER_TREE_WIDTH = IDATA_WIDTH+ $clog2(GROUP_SIZE) ;
localparam SUM_WIDTH = IDATA_WIDTH + $clog2(MAX_LENGTH);
localparam MEAN_DIV_WIDTH = SUM_WIDTH>32?64:32;
localparam MEAN_WIDTH = IDATA_WIDTH;
localparam SQR_WIDTH = IDATA_WIDTH * 2 ;
localparam SQR_ADDER_TREE_WIDTH = SQR_WIDTH+ $clog2(GROUP_SIZE) ;
localparam SQR_SUM_WIDTH = SQR_WIDTH + $clog2(MAX_LENGTH);
localparam VAR_DIV_WIDTH = SQR_SUM_WIDTH>32?64:32;
localparam VAR_WIDTH = IDATA_WIDTH * 2  ;
localparam VAR_FRAC_WIDTH = IFRAC_WIDTH * 2;
localparam LOG2_FRAC_WIDTH = IDATA_WIDTH-1;
localparam LOG2_WIDTH = $clog2(VAR_WIDTH) + 1 + LOG2_FRAC_WIDTH;
localparam POW2_WIDTH = LOG2_WIDTH + 1;
localparam NUM_GROUPS_WIDTH = $clog2(NUM_GROUPS)+1;
logic [NUM_GROUPS_WIDTH-1:0]   in_group_num;

assign in_group_num=in_real_num[NUM_WIDTH-1:$clog2(GROUP_SIZE)]+(|(in_real_num[$clog2(GROUP_SIZE)-1:0]));


logic                          in_valid_dly;
logic                          in_valid_dly2;
logic signed [IDATA_WIDTH-1:0] in_data_buf [NUM_GROUPS-1:0][GROUP_SIZE-1:0];
logic signed [IDATA_WIDTH-1:0] in_data_dly  [GROUP_SIZE-1:0];
logic [NUM_GROUPS_WIDTH-1:0]   inbuf_cnt;

logic signed [MEAN_ADDER_TREE_WIDTH-1:0] in_accum_sum;
logic signed [SUM_WIDTH-1:0] in_accum;
logic [NUM_GROUPS_WIDTH-1:0] add_cnt;
logic [NUM_GROUPS_WIDTH-1:0] sqradd_cnt;
logic signed [MEAN_WIDTH-1:0] mean;
logic [SQR_WIDTH-1:0]mean_sqr;
logic signed [MEAN_DIV_WIDTH-1:0]mean_quotient;
logic mean_add_valid;
logic signed [SQR_WIDTH -1:0] sqr_accin [GROUP_SIZE-1:0];
logic [SQR_ADDER_TREE_WIDTH -1:0] sqr_sum;
logic [SQR_SUM_WIDTH -1:0] sqr_accum;
logic [VAR_DIV_WIDTH-1:0]sqr_quotient;
logic sqr_add_valid;
logic [VAR_WIDTH -1:0] sqr_div;
logic valid_sum;
logic valid_sqrsum;
logic div_ready;


logic [VAR_WIDTH -1:0] var_data;
logic [MEAN_WIDTH-1:0] diff_data [GROUP_SIZE-1:0];
logic [GROUP_SIZE-1:0] sign [NUM_GROUPS-1:0];
logic [$clog2(NUM_GROUPS)-1:0] diff_cnt;
logic valid_diff;
logic valid_mean;
logic valid_sqr_div;

logic signed [LOG2_WIDTH-1:0] log2_mean [GROUP_SIZE-1:0];
logic signed [LOG2_WIDTH-1:0] log2_var;
logic valid_mean_approx;
logic valid_sqr_div_approx;
logic valid_log2;

logic signed [POW2_WIDTH-1:0] pow2_data [GROUP_SIZE-1:0];
logic [ODATA_WIDTH-2:0] out_data [GROUP_SIZE-1:0];
logic [$clog2(NUM_GROUPS)-1:0] out_cnt;
logic valid_pow2;
logic [GROUP_SIZE-1:0]valid_outgen;

//------- In buffer -------
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        in_valid_dly <= 1'b0;
        in_valid_dly2<=1'b0;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            in_data_dly[i] <= '0;
        end
    end else begin
        in_valid_dly <= in_valid_i;
        in_valid_dly2<=in_valid_dly;
        if(in_valid_i)begin
            for (int i = 0; i < GROUP_SIZE; i++) begin
                in_data_dly[i] <= in_data_i[i];
            end
        end
        else begin
            for (int i = 0; i < GROUP_SIZE; i++) begin
                in_data_dly[i] <= in_data_dly[i];
            end
        end
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        inbuf_cnt <= 'b0;
        for (int i = 0; i < NUM_GROUPS; i++) begin
            for (int j = 0; j < GROUP_SIZE; j++) begin
                in_data_buf[i][j] <= 'b0;
            end
        end
    end else if (in_valid_i) begin
        inbuf_cnt <= inbuf_cnt + 1;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            in_data_buf[inbuf_cnt][i] <= in_data_i[i];
        end
    end else begin
        inbuf_cnt <= 'b0;
    end
end

//------- In Add Tree -------
layernorm_add #(
    .DATA_NUM   (GROUP_SIZE    ),
    .DATA_WIDTH (IDATA_WIDTH   ),
    .SUM_WIDTH  (MEAN_ADDER_TREE_WIDTH )
) u_mean_add (
    .clk_i    (clk               ),
    .rstn_i   (rst_n             ),
    .valid_i  (in_valid_dly      ),
    .data_i   (in_data_dly       ),
    .valid_o  (mean_add_valid    ),
    .data_o   (in_accum_sum      )
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        in_accum <= 'b0;
        add_cnt <= 'b0;
    end else if (add_cnt == in_group_num) begin
        in_accum <= 'b0;
        add_cnt <= 'b0;
    end else if (mean_add_valid) begin
        in_accum <= in_accum + in_accum_sum;
        add_cnt <= add_cnt + 1;
    end
    else begin
        in_accum <= in_accum;
        add_cnt <= add_cnt;
    end
end

always_ff@(posedge clk or negedge rst_n) begin
    if(rst_n==0)begin
        for (int i = 0; i < GROUP_SIZE; i++) begin
            sqr_accin[i] <= 'b0;
        end
    end else if (in_valid_dly) begin
        for (int i = 0; i < GROUP_SIZE; i++) begin
            sqr_accin[i] <= $signed(in_data_dly[i]) * $signed(in_data_dly[i]);
        end
    end else begin
        for (int i = 0; i < GROUP_SIZE; i++) begin
            sqr_accin[i] <= 'b0;
        end
    end
end

layernorm_add #(
    .DATA_NUM   (GROUP_SIZE    ),
    .DATA_WIDTH (SQR_WIDTH     ),
    .SUM_WIDTH  (SQR_ADDER_TREE_WIDTH)
) u_sqr_add (
    .clk_i    (clk           ),
    .rstn_i   (rst_n         ),
    .valid_i  (in_valid_dly2 ),
    .data_i   (sqr_accin     ),
    .valid_o  (sqr_add_valid ),
    .data_o   (sqr_sum       )
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sqr_accum <= 'b0;
        sqradd_cnt<='0;
    end else if (sqradd_cnt == in_group_num) begin
        sqr_accum <= 'b0;
        sqradd_cnt<='0;
    end else if (sqr_add_valid) begin
        sqr_accum <= sqr_accum + sqr_sum;
        sqradd_cnt <= sqradd_cnt + 1;
    end
    else begin
        sqr_accum <= sqr_accum;
        sqradd_cnt <= sqradd_cnt;
    end
end

//------- Mean & Variance -------
assign valid_sum = add_cnt == in_group_num && in_group_num!=0;
assign valid_sqrsum = sqradd_cnt == in_group_num && in_group_num!=0;
int_div_radix_16_v4 #(
    .WIDTH(MEAN_DIV_WIDTH)  
) u_divider_mean (
    .div_start_valid_i  (valid_sum),    
    .div_start_ready_o  (),    
    .flush_i            (1'b0),          
    .signed_op_i        (1'b1),      

    .dividend_i         ({{(MEAN_DIV_WIDTH-SUM_WIDTH){in_accum[SUM_WIDTH-1]}},in_accum}),      
    .divisor_i          ({{(MEAN_DIV_WIDTH-NUM_WIDTH){1'b0}},in_real_num}),       

    .div_finish_valid_o (valid_mean),   
    .div_finish_ready_i (div_ready),   
    .quotient_o         (mean_quotient),       
    .remainder_o        (),      
    .divisor_is_zero_o  (),      

    .clk                (clk),            
    .rst_n              (rst_n)           
);
int_div_radix_16_v4 #(
    .WIDTH(VAR_DIV_WIDTH)  
) u_divider_var (
    .div_start_valid_i  (valid_sqrsum),    
    .div_start_ready_o  (),    
    .flush_i            (1'b0),          
    .signed_op_i        (1'b0),      

    .dividend_i         ({{(VAR_DIV_WIDTH-SQR_SUM_WIDTH){1'b0}},sqr_accum}),      
    .divisor_i          ({{(VAR_DIV_WIDTH-NUM_WIDTH){1'b0}},in_real_num}),       

    .div_finish_valid_o (valid_sqr_div),   
    .div_finish_ready_i (div_ready),   
    .quotient_o         (sqr_quotient),       
    .remainder_o        (),      
    .divisor_is_zero_o  (),      

    .clk                (clk),            
    .rst_n              (rst_n)           
);
assign sqr_div = sqr_quotient[VAR_WIDTH-1:0];
always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)
    mean<='b0;
    else if(valid_mean)
    mean<=mean_quotient[MEAN_WIDTH-1:0];
    else
    mean<=mean;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)
    mean_sqr<='b0;
    else if(valid_mean)
    mean_sqr<=$signed(mean_quotient[MEAN_WIDTH-1:0])*$signed(mean_quotient[MEAN_WIDTH-1:0]);
    else
    mean_sqr<=mean_sqr;
end
always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)
    div_ready<=1'b0;
    else if(valid_mean && valid_sqr_div)
    div_ready<=1'b1;
    else
    div_ready<=1'b0;
end
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_diff <= 1'b0;
    end else if (diff_cnt == in_group_num - 1 && valid_diff==1) begin
        valid_diff <= 1'b0;
    end else if (valid_mean && valid_sqr_div) begin
        valid_diff <= 1'b1;
    end else begin
        valid_diff <=valid_diff;
    end
end
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        var_data <= 'b0;
    end else if (valid_sqr_div && valid_mean) begin
        var_data <=sqr_div-mean_sqr;
    end
    else begin
        var_data<= var_data;
    end
end

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_log2 <= 1'b0;
        diff_cnt <= 'b0;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            diff_data[i] <= 'b0;
        end
    end else if (valid_diff) begin
        valid_log2 <= 1'b1;
        diff_cnt <= diff_cnt + 1;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            if (in_data_buf[diff_cnt][i] > mean) begin
                diff_data[i] <= in_data_buf[diff_cnt][i] -mean;
                sign[diff_cnt][i] <= 1'b0;
            end else begin
                diff_data[i] <= mean - in_data_buf[diff_cnt][i];
                sign[diff_cnt][i] <= 1'b1;
            end
        end
    end else begin
        valid_log2 <= 1'b0;
        diff_cnt <= 'b0;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            diff_data[i] <= diff_data[i];
        end
    end
end

log2_approximation_ver #(
    .VER_WIDTH   (GROUP_SIZE     ),
    .IDATA_WIDTH (MEAN_WIDTH     ),
    .IFRAC_WIDTH (IFRAC_WIDTH    ),
    .ODATA_WIDTH (LOG2_WIDTH     ),
    .OFRAC_WIDTH (LOG2_FRAC_WIDTH)
) u_log2_mean_approximation (
    .clk_i   (clk            ),
    .rstn_i  (rst_n          ),
    .data_i  (diff_data       ),
    .valid_i (valid_log2     ),
    .data_o  (log2_mean      ),
    .valid_o (valid_mean_approx)
);

log2_approximation #(
    .IDATA_WIDTH (VAR_WIDTH      ),
    .IFRAC_WIDTH (VAR_FRAC_WIDTH ),
    .ODATA_WIDTH (LOG2_WIDTH     ),
    .OFRAC_WIDTH (LOG2_FRAC_WIDTH)
) u_log2_var_approximation (
    .clk_i   (clk           ),
    .rstn_i  (rst_n          ),
    .data_i  (var_data        ),
    .valid_i (valid_log2        ),
    .data_o  (log2_var        ),
    .valid_o (valid_sqr_div_approx)
);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_pow2  <= 1'b0;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            pow2_data[i] <= 'b0;
        end
    end else if (valid_mean_approx && valid_sqr_div_approx) begin
        valid_pow2  <= 1'b1;
        for (int i = 0; i < GROUP_SIZE; i++) begin
            pow2_data[i] <= log2_mean[i] - (log2_var >>> 1);
        end
    end else begin
        valid_pow2  <= 1'b0;
    end
end

generate
    for (genvar i = 0; i < GROUP_SIZE; i = i + 1) begin
        pow2_approximation #(
            .IN_DATA_WIDTH  (POW2_WIDTH     ),
            .IN_FRAC_WIDTH  (LOG2_FRAC_WIDTH),
            .OUT_DATA_WIDTH (ODATA_WIDTH-1  ),
            .OUT_FRAC_WIDTH (OFRAC_WIDTH    )
        ) u_pow2_approximation (
            .clk_i      (clk               ),
            .rstn_i     (rst_n             ),
            .in_valid   (valid_pow2        ),
            .in_data_i  (pow2_data[i]      ),
            .out_valid  (valid_outgen[i]      ),
            .out_data_o (out_data[i]       )
        );
    end
endgenerate

generate
  for (genvar i = 0; i < GROUP_SIZE; i++) begin : SIGN_EXTEND
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data_o[i] <= '0;
        end else if (&valid_outgen) begin
            if (sign[out_cnt][i])
                out_data_o[i] <= -$signed({1'b0, out_data[i]});
            else
                out_data_o[i] <=  $signed({1'b0, out_data[i]});
        end else begin
            out_data_o[i] <= '0;
        end
    end
  end
endgenerate

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_cnt <= 'b0;
        out_valid_o <= 1'b0;
    end else if (&valid_outgen) begin
        out_cnt <= out_cnt + 1;
        out_valid_o <= 1'b1;
    end else begin
        out_cnt <= 'b0;
        out_valid_o <= 1'b0;
    end
end

// always_ff @(posedge clk or negedge rst_n) begin
//     if (!rst_n) begin
//         out_done_o <= 1'b0;
//     end else if (out_cnt == NUM_GROUPS - 1) begin
//         out_done_o <= 1'b1;
//     end else begin
//         out_done_o <= 1'b0;
//     end
// end

endmodule