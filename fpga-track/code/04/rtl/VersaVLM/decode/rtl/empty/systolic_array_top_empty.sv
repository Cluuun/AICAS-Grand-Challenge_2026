//*****************************************************************
// 'Empty' stub module for FPGA verification.
//
// This module has the exact same interface as 'systolic_array_top'
// but all outputs are tied to 0. This allows it to be swapped in
// when testing other parts of the design that interface with it.
//*****************************************************************
module systolic_array_top_empty
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
    parameter int SPM_ADDR_WIDTH       = $clog2(SPM_SIZE)
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
    input   logic    [RF_DATA_WIDTH/8-1:0]           sa_input_a_col_num_sub1     ,   // column number of input feature(WS)/input matrix A(OS) in systolic array
    input   logic    [RF_DATA_WIDTH/8-1:0]           sa_input_a_row_num_sub1     ,   // row number of input feature(WS)/input matrix A(OS) in systolic array
   
     
    input   logic    [RF_DATA_WIDTH/2-1:0]           sa_input_b_spm_addr         ,   // address for input matrix B(OS) data in SPM
    input   logic    [RF_DATA_WIDTH/8-1:0]           sa_input_b_col_num_sub1     ,   // column number of input matrix B(OS) in systolic array
    input   logic    [RF_DATA_WIDTH/8-1:0]           sa_input_b_row_num_sub1     ,   // row number of input matrix B(OS) in systolic array

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

    // Tie all outputs to a default, inactive state (0)
    assign sa_busy          = 1'b0;
    assign sa_comp_done     = 1'b0;
    
    assign sa_spm_rd1_en    = 1'b0;
    assign sa_spm_rd1_addr  = '0; // Assigns all bits to 0
    
    assign sa_spm_rd2_en    = 1'b0;
    assign sa_spm_rd2_addr  = '0;
    
    assign sa_acc_start     = 1'b0;
    assign sa_acc_valid     = 1'b0;
    assign sa_acc_mask      = '0;
    assign sa_acc_data_out  = '0;

endmodule
