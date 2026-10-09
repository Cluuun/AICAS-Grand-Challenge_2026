module systolic_array_top
    import npu_config_pkg::*;
#(
    parameter int RF_DATA_WIDTH        = npu_config_pkg::RF_DATA_WIDTH,
    parameter int PE_DATA_WIDTH_IN     = npu_config_pkg::PE_DATA_WIDTH_IN,
    parameter int PE_DATA_WIDTH_OUT    = npu_config_pkg::PE_DATA_WIDTH_OUT,
    parameter int INPUT_WIDTH_MAX      = npu_config_pkg::INPUT_WIDTH_MAX,
    parameter int INPUT_HEIGHT_MAX     = npu_config_pkg::INPUT_HEIGHT_MAX,
    parameter int ARRAY_WIDTH          = npu_config_pkg::ARRAY_WIDTH,
    parameter int ARRAY_HEIGHT         = npu_config_pkg::ARRAY_HEIGHT,
    parameter int SPM_SIZE             = npu_config_pkg::SPM_SIZE,
    parameter int SPM_ADDR_WIDTH       = $clog2(SPM_SIZE),
    parameter PE_FPGA_DSP = npu_config_pkg::PE_FPGA_DSP,
    parameter integer DSP_PE_NUM = (PE_FPGA_DSP ? ARRAY_WIDTH * ARRAY_HEIGHT : 0),
    parameter integer DISABLE_IM2COL = 0
)
(
    input   logic   clk,          
    
    input   logic   rstn,    
    //----------------------------------  
    // Systolic array control signals
    //----------------------------------
    input   logic                                    cfg_compute_dataflow           ,  // 0 for weight stationary, 1 for output stationary
    input   logic    [1:0]                           cfg_compute_padding_left       ,  // Number of left padding data columns for the input
    input   logic    [1:0]                           cfg_compute_padding_right      ,  // Number of right padding data columns for the input
    input   logic    [1:0]                           cfg_compute_padding_top        ,  // Number of top padding data columns for the input
    input   logic    [1:0]                           cfg_compute_padding_bottom     ,  // Number of bottom padding data columns for the input
    input   logic    [1:0]                           cfg_compute_padding_mode       ,  // Only zero padding supported currently
    input   logic    [3:0]                           cfg_compute_weight_shape_m1    ,  // 0 => 1
    input   logic    [1:0]                           cfg_compute_weight_stride_m1   ,  // 0 => 1
    input   logic    [4:0]                           cfg_compute_weight_dilation_m1 ,  // 0 => 1
    input   logic                                    cfg_compute_is_groupconv       ,
    input   logic    [1:0]                           cfg_compute_optype             ,
    input   logic                                    cfg_compute_asymmetric_activations,

    input   logic    [1:0]                           cfg_compute_int_type        ,   // datatype range of input data, input feature/A should be the same precision as weight/B,
                                                                                     // 00 for int8, 01 for int16, 10 for int32, 11 for int64
    input   logic    [RF_DATA_WIDTH/4-1:0]           sa_input_a_stride           ,   // address interval for systolic array inputA or input feature data
                                                                                     //the stride indicates the delta address to next row
    input   logic    [RF_DATA_WIDTH/4-1:0]           sa_input_b_stride           ,   // address interval for systolic array inputB  data
                                                                                     //the stride indicates the delta address to next row                
    input   logic    [RF_DATA_WIDTH/2-1:0]           sa_input_a_spm_addr         ,   // address for input feature(WS)/input matrix A(OS) data in SPM   
    input   logic    [$clog2(INPUT_WIDTH_MAX)-1:0]   sa_input_a_col_num_sub1     ,   // column number of input feature(WS)/input matrix A(OS) in systolic array
    input   logic    [$clog2(ARRAY_HEIGHT)-1:0]      sa_input_a_row_num_sub1     ,   // row number of input feature(WS)/input matrix A(OS) in systolic array
   
     
    input   logic    [RF_DATA_WIDTH/2-1:0]           sa_input_b_spm_addr         ,   // address for input matrix B(OS) data in SPM
    input   logic    [$clog2(ARRAY_WIDTH)-1:0]       sa_input_b_col_num_sub1     ,   // column number of input matrix B(OS) in systolic array
    input   logic    [$clog2(INPUT_HEIGHT_MAX)-1:0]  sa_input_b_row_num_sub1     ,   // row number of input matrix B(OS) in systolic array

    input   logic    [RF_DATA_WIDTH/8-1:0]           cfg_accu_biaspsum_width     ,   //bias/psum data width  ,in conv is ofm-col-num
    input   logic    [RF_DATA_WIDTH/8-1:0]           cfg_accu_biaspsum_height    ,   //bias/psum data height ,in conv is ofm-row-num
        
    input   logic                                    sa_req_en                   ,
    output  logic                                    sa_busy                     ,  //the systolic array starts to calculate
    output  logic                                    sa_comp_done                ,  //the systolic array has finished the calculation
    //----------------------------------
    // Scratchpad
    //----------------------------------
    output logic                                                sa_spm_rd1_en         ,   // SPM read enable  //weight
    output logic [SPM_ADDR_WIDTH-1:0]                           sa_spm_rd1_addr       ,   // SPM read address
    input  logic [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0]        sa_spm_rd1_data_in ,

    output logic                                                sa_spm_rd2_en         ,   // SPM read enable //active
    output logic [SPM_ADDR_WIDTH-1:0]                           sa_spm_rd2_addr       ,   // SPM read address
    input  logic [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0]       sa_spm_rd2_data_in    ,
    //output for acc
    output logic                                                sa_acc_start         ,
    output logic                                                sa_acc_valid         ,   // ACC  enable
    output logic [ARRAY_WIDTH-1:0]                              sa_acc_mask          ,      // ACC  mask
    output logic [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_OUT-1:0]       sa_acc_data_out      
    
);
    logic   sa_req_en_delay;
    logic   im2col_req_en;
    logic   weightfeeder_valid;
    logic   im2col_valid;
    logic   sa_ready;
    logic   systolic_array_start;
    logic   systolic_array_os_wr_en_o;
 
    logic                                                conv_spm_rd1_en      ;  
    logic [SPM_ADDR_WIDTH-1:0]                           conv_spm_rd1_addr    ;  
    logic                                                conv_spm_rd2_en      ;  
    logic [SPM_ADDR_WIDTH-1:0]                           conv_spm_rd2_addr    ;
    logic                                                gemm_spm_rd1_en      ;  
    logic [SPM_ADDR_WIDTH-1:0]                           gemm_spm_rd1_addr    ;  
    logic                                                gemm_spm_rd2_en      ;  
    logic [SPM_ADDR_WIDTH-1:0]                           gemm_spm_rd2_addr    ;  
    logic signed [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] weight_feeder_sa_data;
    logic signed [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0]im2col_sa_data       ;
    logic        [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0]active_row_pre_sa    ;
    logic signed [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0]active_row_raw       ;
    logic signed [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN:0]  active_row_adjusted  ;
    logic signed [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0]active_row           ;
    logic signed [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] weight_col           ;

    logic [$clog2(ARRAY_HEIGHT):0]     input_a_row_num;
    logic [$clog2(INPUT_WIDTH_MAX):0]  input_a_col_num;
    logic [$clog2(INPUT_HEIGHT_MAX):0] input_b_row_num;
    logic [$clog2(ARRAY_WIDTH):0]      input_b_col_num;
    logic [$clog2(ARRAY_HEIGHT):0]     sa_a_row_num;
    logic [$clog2(INPUT_WIDTH_MAX):0]  sa_a_col_num;
    logic [$clog2(INPUT_HEIGHT_MAX):0] sa_b_row_num;
    logic [$clog2(ARRAY_WIDTH):0]      sa_b_col_num;
    logic [RF_DATA_WIDTH/8-1:0]        sa_input_ofm_col_num_m1;
	    logic [RF_DATA_WIDTH/8-1:0]        sa_input_ofm_row_num_m1;
	    logic [RF_DATA_WIDTH/4-1:0]        sa_ofm_size;
        logic                              gemm_active_path;
	    logic                              addr_leap_enable;
    logic [$clog2(ARRAY_WIDTH+INPUT_HEIGHT_MAX+ARRAY_WIDTH):0]addr_leap;

   
    
    always_ff@(posedge clk or negedge rstn) begin
        if(~rstn)begin
            input_a_col_num            <= '0;
            input_a_row_num            <= '0;
            input_b_col_num            <= '0;
            input_b_row_num            <= '0;
            sa_input_ofm_col_num_m1    <= '0;
            sa_input_ofm_row_num_m1    <= '0;
        end else if (sa_req_en)begin
            input_a_col_num            <= sa_input_a_col_num_sub1 + 1'b1;
            input_a_row_num            <= sa_input_a_row_num_sub1 + 1'b1;
            input_b_col_num            <= sa_input_b_col_num_sub1 + 1'b1;
            input_b_row_num            <= sa_input_b_row_num_sub1 + 1'b1;
            sa_input_ofm_col_num_m1    <= cfg_accu_biaspsum_width - 1'b1;
            sa_input_ofm_row_num_m1    <= cfg_accu_biaspsum_height - 1'b1;
        end
        else begin
            input_a_col_num            <= input_a_col_num  ;
            input_a_row_num            <= input_a_row_num  ;
            input_b_col_num            <= input_b_col_num  ;
            input_b_row_num            <= input_b_row_num  ;
            sa_input_ofm_col_num_m1    <= '0;
            sa_input_ofm_row_num_m1    <= '0;
        end
    end
    always_ff @( posedge clk) begin 
        sa_req_en_delay   <= sa_req_en;
        //systolic_array_start <= (im2col_valid && weightfeeder_valid &&(~sa_ready)&& cfg_compute_dataflow==0)||(cfg_compute_dataflow==1&&sa_req_en);
    end
		    always_comb begin : convgemmselect
		        sa_acc_mask         = ~({ARRAY_WIDTH{1'b1}}<<input_b_col_num);
	            active_row_pre_sa  = '0;
                gemm_active_path   = 1'b0;
		        if(cfg_compute_dataflow || (DISABLE_IM2COL == 1))begin //gemm / bypass im2col path
                    gemm_active_path = 1'b1;
		            sa_spm_rd1_en   = gemm_spm_rd1_en;
		            sa_spm_rd1_addr = gemm_spm_rd1_addr;
	            sa_spm_rd2_en   = gemm_spm_rd2_en;
	            sa_spm_rd2_addr = gemm_spm_rd2_addr;
	            active_row_pre_sa = sa_spm_rd2_data_in;
	            active_row_raw  = $signed(sa_spm_rd2_data_in);
	            weight_col      = sa_spm_rd1_data_in;
	            sa_a_row_num    = input_a_row_num;
	            sa_a_col_num    = input_a_col_num;
	            sa_b_row_num    = input_b_row_num;
	        end
	        else begin  //conv
	            sa_spm_rd1_en   = conv_spm_rd1_en;
	            sa_spm_rd1_addr = conv_spm_rd1_addr;
	            sa_spm_rd2_en   = conv_spm_rd2_en;
	            sa_spm_rd2_addr = conv_spm_rd2_addr;
	            active_row_raw  = im2col_sa_data;
	            weight_col      = weight_feeder_sa_data;
	            sa_a_row_num    = sa_ofm_size[$clog2(ARRAY_HEIGHT):0];
	            sa_a_col_num    = ((cfg_compute_weight_shape_m1+1)*(cfg_compute_weight_shape_m1+1))<<$clog2(ARRAY_WIDTH);
	            sa_b_row_num    = sa_a_col_num;
	        end
        sa_b_col_num        = input_b_col_num;
        sa_ofm_size         = (cfg_accu_biaspsum_width) * (cfg_accu_biaspsum_height);
        im2col_req_en       = sa_req_en_delay && (~cfg_compute_dataflow) && (DISABLE_IM2COL == 0);
	        sa_acc_start        = systolic_array_os_wr_en_o && (~sa_acc_valid);
	        systolic_array_start= (im2col_valid && weightfeeder_valid &&(~sa_ready) && (cfg_compute_dataflow==0) && (DISABLE_IM2COL==0))
	                              || (((cfg_compute_dataflow==1) || (DISABLE_IM2COL==1)) && sa_req_en_delay);
	    end
	        always_comb begin : adjust_gemm_asymmetric_activations
	            active_row = active_row_raw;
	            for (int lane = 0; lane < ARRAY_HEIGHT; lane++) begin
	                active_row_adjusted[lane] = $signed({1'b0, active_row_pre_sa[lane]}) - 9'sd128;
	                if (cfg_compute_asymmetric_activations && gemm_active_path && (cfg_compute_optype == 2'b00)) begin
	                    active_row[lane] = active_row_adjusted[lane][PE_DATA_WIDTH_IN-1:0];
	                end
	            end
	        end
	    always_ff @( posedge clk or negedge rstn ) begin 
	        if(!rstn)
	        sa_ready<=1'b0;
        else if(sa_comp_done)
        sa_ready<=1'b0;
        else if(im2col_valid && weightfeeder_valid)
        sa_ready<=1'b1;
        else 
        sa_ready<=sa_ready;
    end
    
    //         
    //             [input_b]
    //            -----------
    // [input_a]  |   SA    |
    //            -----------
    //after Im2col  input_a:ofmsize*Cin,input_b:Cin*Cout
    systolic_array_os_ctrl_data #(
        .PE_FPGA_DSP     ( PE_FPGA_DSP     ),
        .DSP_PE_NUM      ( DSP_PE_NUM      ),
        .PE_DATA_WIDTH_IN( PE_DATA_WIDTH_IN),
        .PE_DATA_WIDTH_OUT(PE_DATA_WIDTH_OUT),
        .INPUT_WIDTH_MAX ( INPUT_WIDTH_MAX ),
        .INPUT_HEIGHT_MAX( INPUT_HEIGHT_MAX),
        .ARRAY_WIDTH     ( ARRAY_WIDTH     ),
        .ARRAY_HEIGHT    ( ARRAY_HEIGHT    )
    ) u_systolic_array_os_ctrl_data (
        .clk_i(clk),
        .rstn_i(rstn),
        .systolic_array_start_i(systolic_array_start),
        .active_row(active_row),
        .weight_column_i(weight_col),
        .input_a_row_num(sa_a_row_num),
        .input_a_col_num(sa_a_col_num),
        .input_b_row_num(sa_b_row_num),   
        .input_b_col_num(sa_b_col_num),   
        .result_os_o(sa_acc_data_out),
        .systolic_array_busy_o(),
        .systolic_array_os_wr_en_o(systolic_array_os_wr_en_o),
        .systolic_array_done_o(sa_comp_done),
        .addr_leap(addr_leap),
        .addr_leap_enable(addr_leap_enable)
    );
    generate
        if (DISABLE_IM2COL == 1) begin : gen_no_im2col
            assign conv_spm_rd2_en   = 1'b0;
            assign conv_spm_rd2_addr = '0;
            assign im2col_valid      = 1'b0;
            assign im2col_sa_data    = '0;
        end
        else begin : gen_im2col
            im2col_top #(
                .RF_DATA_WIDTH   ( RF_DATA_WIDTH    ),
                .PE_DATA_WIDTH_IN( PE_DATA_WIDTH_IN ),
                .INPUT_WIDTH_MAX ( INPUT_WIDTH_MAX  ),
                .ARRAY_HEIGHT    ( ARRAY_HEIGHT     ),
                .SPM_SIZE        ( SPM_SIZE         ),
                .SPM_ADDR_WIDTH  ( SPM_ADDR_WIDTH   )
            ) u_im2col (
                .clk(clk),
                .rst_n(rstn),
                
                .cfg_compute_dataflow(cfg_compute_dataflow),
                .cfg_compute_padding_left(cfg_compute_padding_left),
                .cfg_compute_padding_right(cfg_compute_padding_right),
                .cfg_compute_padding_top(cfg_compute_padding_top),
                .cfg_compute_padding_bottom(cfg_compute_padding_bottom),
                .cfg_compute_padding_mode(cfg_compute_padding_mode),
                .cfg_compute_weight_shape_m1(cfg_compute_weight_shape_m1),
                .cfg_compute_weight_stride_m1(cfg_compute_weight_stride_m1),
                .cfg_compute_weight_dilation_m1(cfg_compute_weight_dilation_m1),
                .cfg_compute_is_groupconv(cfg_compute_is_groupconv),
                .cfg_compute_int_type(cfg_compute_int_type),
                .cfg_compute_optype(cfg_compute_optype),
                
                .sa_input_ifm_spm_addr(sa_input_a_spm_addr),
                .sa_input_ifm_col_num_m1(sa_input_a_col_num_sub1),
                .sa_input_ifm_row_num_m1(sa_input_a_row_num_sub1),
                .sa_input_ifm_stride(sa_input_a_stride),
                .sa_input_ifm_channel_m1(sa_input_b_row_num_sub1[$clog2(ARRAY_HEIGHT)-1:0]),
                
                .sa_input_ofm_col_num_m1(sa_input_ofm_col_num_m1),
                .sa_input_ofm_row_num_m1(sa_input_ofm_row_num_m1),
                
                .sa_spm_rd_en(conv_spm_rd2_en),
                .sa_spm_rd_addr(conv_spm_rd2_addr),
                .sa_spm_rd_data_in(sa_spm_rd2_data_in),
                
                .im2col_sa_ready(sa_ready),
                .im2col_sa_valid(im2col_valid),
                .im2col_sa_data(im2col_sa_data),
                
                .im2col_req_en(im2col_req_en),
                .im2col_busy(),
                .im2col_done()
            );
        end
    endgenerate
    weight_feeder_top #(
        .RF_DATA_WIDTH   ( RF_DATA_WIDTH    ),
        .PE_DATA_WIDTH_IN( PE_DATA_WIDTH_IN ),
        .ARRAY_WIDTH     ( ARRAY_WIDTH      ),
        .ARRAY_HEIGHT    ( ARRAY_HEIGHT     ),
        .SPM_SIZE        ( SPM_SIZE         ),
        .SPM_ADDR_WIDTH  ( SPM_ADDR_WIDTH   )
    ) u_weight_feeder (
        .clk(clk),
        .rst_n(rstn),
        
        .cfg_compute_weight_shape_m1(cfg_compute_weight_shape_m1),
        .sa_input_weight_spm_addr(sa_input_b_spm_addr),
        .cfg_compute_weight_channel_out_m1(sa_input_b_col_num_sub1),
        .cfg_compute_weight_channel_in_m1(sa_input_b_row_num_sub1[$clog2(ARRAY_HEIGHT)-1:0]),
        
        .weight_spm_rd_en(conv_spm_rd1_en),
        .weight_spm_rd_addr(conv_spm_rd1_addr),
        .weight_spm_rd_data_in(sa_spm_rd1_data_in),
        
        .weight_feeder_sa_ready(sa_ready),
        .weight_feeder_sa_valid(weightfeeder_valid),
        .weight_feeder_sa_data(weight_feeder_sa_data),
        
        .weight_feeder_req_en(im2col_req_en),
        .weight_feeder_busy(),
        .weight_feeder_done()
    );
 
     
    
    always_ff @( posedge clk or negedge rstn ) begin 
        if(~rstn) 
        sa_busy<=1'b0;
        else if(sa_comp_done)
        sa_busy<=1'b0;
        else if(sa_req_en)
        sa_busy<=1'b1;
        else 
        sa_busy<=sa_busy;      
    end
    always_ff @(posedge clk or negedge rstn) begin : wr_mask
        if(~rstn)begin
            sa_acc_valid <= 0;
        end else begin
            sa_acc_valid <= systolic_array_os_wr_en_o;
        end
    end
    // --------------------------------------------------------------------------
    // Read SPM
    // --------------------------------------------------------------------------
    // read spm
    always_ff@(posedge clk or negedge rstn)begin 
        if(~rstn)begin
            gemm_spm_rd1_en    <= 'b0;
        end
        else if (sa_req_en && ((cfg_compute_dataflow==1) || (DISABLE_IM2COL==1)))begin
            gemm_spm_rd1_en    <= 'b1;
        end
        else if (sa_acc_start)begin
            gemm_spm_rd1_en    <= 'b0;
        end
        else begin
            gemm_spm_rd1_en    <= gemm_spm_rd1_en;
        end
    end
    always_ff@(posedge clk or negedge rstn)begin 
        if(~rstn)begin
            gemm_spm_rd2_en    <= 'b0;
        end
        else if (sa_req_en && ((cfg_compute_dataflow==1) || (DISABLE_IM2COL==1)))begin
            gemm_spm_rd2_en    <= 'b1;
        end
        else if (sa_comp_done)begin
            gemm_spm_rd2_en    <= 'b0;
        end
        else begin
            gemm_spm_rd2_en    <= gemm_spm_rd2_en;
        end
    end

    // read spm 1 port (up)  weight
    always_ff@(posedge clk or negedge rstn)begin 
        if(~rstn)begin
            gemm_spm_rd1_addr    <= 'b0;
        // init load input B
        end else if (sa_req_en)begin
            gemm_spm_rd1_addr    <= sa_input_b_spm_addr;
        // loading
        end else if (gemm_spm_rd1_en)begin 
           gemm_spm_rd1_addr     <= gemm_spm_rd1_addr + sa_input_b_stride*PE_DATA_WIDTH_IN/8;
        end
    end

    // read spm 2 port (left)   active
    always_ff@(posedge clk or negedge rstn)begin 
        if(~rstn)begin
            gemm_spm_rd2_addr    <= 'b0;
        // init load input A
        end else if (sa_req_en)begin
            gemm_spm_rd2_addr    <= sa_input_a_spm_addr ;
        // loading
        end else if (gemm_spm_rd2_en)begin 
            if(addr_leap_enable)
            gemm_spm_rd2_addr    <= sa_input_a_spm_addr + addr_leap;
            else
            gemm_spm_rd2_addr    <= gemm_spm_rd2_addr + sa_input_a_stride*PE_DATA_WIDTH_IN/8;
        end
    end
    
endmodule
