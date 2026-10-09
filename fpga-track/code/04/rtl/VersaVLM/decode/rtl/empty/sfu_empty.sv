//*****************************************************************
// 'Empty' stub module for FPGA verification.
//
// This module has the exact same interface as 'sfu'
// but all outputs are tied to 0 or an inactive state.
//*****************************************************************
module sfu_empty import npu_config_pkg::*; #(
    parameter int RF_DATA_WIDTH        = npu_config_pkg::RF_DATA_WIDTH,
    parameter int PE_WIDTH             = npu_config_pkg::PE_WIDTH,
    parameter int PE_DATA_WIDTH        = npu_config_pkg::PE_DATA_WIDTH,
    parameter int SPM_SIZE             = npu_config_pkg::SPM_SIZE,
    parameter integer DISABLE_SOFTMAX     = npu_config_pkg::DISABLE_SOFTMAX,
    parameter integer DISABLE_GELU        = npu_config_pkg::DISABLE_GELU,
    parameter integer DISABLE_LAYERNORM   = npu_config_pkg::DISABLE_LAYERNORM,
    parameter integer DISABLE_RESAMPLE    = npu_config_pkg::DISABLE_RESAMPLE,
    parameter integer DISABLE_TRANSPOSE   = npu_config_pkg::DISABLE_TRANSPOSE,
    parameter integer GELU_NUM            = npu_config_pkg::GELU_NUM,
    localparam ADDR_WIDTH                 = $clog2(SPM_SIZE)
) (
    input  logic   clk,
    input  logic   rst_n,

    //------------------------------------         
    // SFU Control Signals           
    //------------------------------------         

    input  logic   [5:0]                           cfg_sfu_op,
    input  logic   [1:0]                           cfg_sfu_int_type,
    input  logic                                   cfg_sfu_is_quant,
    input  logic                                   cfg_trans_out_ispad_row,
    input  logic                                   cfg_trans_out_ispad_col,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_zeropoint,
    input  logic   [RF_DATA_WIDTH/2-1:0]           cfg_sfu_input_zeropoint,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_sfu_input_scale,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_scale,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_sfu_input_scale_shift,
    input  logic   [RF_DATA_WIDTH/4-1:0]           cfg_sfu_output_scale_shift,

    input  logic   [RF_DATA_WIDTH/2-1:0]           sfu_input_sram_addr,
    input  logic   [RF_DATA_WIDTH/4-1:0]           sfu_input_col_num,
    input  logic   [RF_DATA_WIDTH/4-1:0]           sfu_input_row_num,

    input  logic   [RF_DATA_WIDTH/2-1:0]           sfu_output_spm_addr,
    
    input  logic                                   sfu_req_en,
    output logic                                   sfu_busy,          // Output
    output logic                                   sfu_comp_done,     // Output

    //------------------------------------
    // ScratchPad (SPM) Interface
    //------------------------------------
    
    output logic                                   sfu_spm_rd_en,     // Output
    output logic   [ADDR_WIDTH-1:0]                sfu_spm_rd_addr,   // Output
    input  logic   [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0] sfu_spm_rd_data_in,

    output logic                                   sfu_spm_wr_en,     // Output
    output logic   [ADDR_WIDTH-1:0]                sfu_spm_wr_addr,   // Output
    output logic   [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0] sfu_spm_wr_data_out, // Output
    output logic   [PE_WIDTH-1:0]                  sfu_spm_wr_mask    // Output

);

    // ------------------------------------- 
    // SFU Control Outputs
    // ------------------------------------- 
    // Always report not busy and instantly done
    assign sfu_busy      = 1'b0;
    assign sfu_comp_done = 1'b0;

    // ------------------------------------- 
    // SPM Read Interface (Inactive)
    // ------------------------------------- 
    // Never issue a read request
    assign sfu_spm_rd_en   = 1'b0;
    assign sfu_spm_rd_addr = '0;

    // ------------------------------------- 
    // SPM Write Interface (Inactive)
    // ------------------------------------- 
    // Never issue a write request
    assign sfu_spm_wr_en       = 1'b0;
    assign sfu_spm_wr_addr     = '0;
    assign sfu_spm_wr_data_out = '0;
    assign sfu_spm_wr_mask     = '0;

endmodule
