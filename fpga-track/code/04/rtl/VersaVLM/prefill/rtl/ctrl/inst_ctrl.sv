`default_nettype none

import npu_spm_pkg::*;

// Versa_P API register file and command issuer.
//
// Purpose:
//   Expose the five accepted INT8 APIs through a compact AXI-Lite register
//   surface and translate write-one-to-start descriptors into local core
//   commands. This module owns descriptor latching, sticky status, IRQ status,
//   and W resident-bank selection state. It does not generate bank addresses.
//
// Clock/reset:
//   All state uses clk_i. rst_i is synchronous, active high.
module inst_ctrl #(
    parameter int AXIL_ADDR_W = 32,
    parameter int AXIL_DATA_W = 64,
    parameter int BANK_DEPTH_WORDS = 16384,
    parameter int META_WORD128_DEPTH = 2048,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0,
    parameter bit ENABLE_PROFILE = 1'b0,
    localparam int AXIL_STRB_W = AXIL_DATA_W / 8
) (
    input  wire logic                      clk_i,
    input  wire logic                      rst_i,

    input  wire logic [AXIL_ADDR_W-1:0]    s_axi_awaddr_i,
    input  wire logic [2:0]                s_axi_awprot_i,
    input  wire logic                      s_axi_awvalid_i,
    output logic                           s_axi_awready_o,
    input  wire logic [AXIL_DATA_W-1:0]    s_axi_wdata_i,
    input  wire logic [AXIL_STRB_W-1:0]    s_axi_wstrb_i,
    input  wire logic                      s_axi_wvalid_i,
    output logic                           s_axi_wready_o,
    output logic [1:0]                     s_axi_bresp_o,
    output logic                           s_axi_bvalid_o,
    input  wire logic                      s_axi_bready_i,
    input  wire logic [AXIL_ADDR_W-1:0]    s_axi_araddr_i,
    input  wire logic [2:0]                s_axi_arprot_i,
    input  wire logic                      s_axi_arvalid_i,
    output logic                           s_axi_arready_o,
    output logic [AXIL_DATA_W-1:0]         s_axi_rdata_o,
    output logic [1:0]                     s_axi_rresp_o,
    output logic                           s_axi_rvalid_o,
    input  wire logic                      s_axi_rready_i,
    output logic                           irq_o,

    output logic                           cmd_valid_o,
    input  wire logic                      cmd_ready_i,
    output logic [2:0]                     cmd_opcode_o,
    output logic                           cmd_a_bank_o,
    output logic                           cmd_w_bank_o,
    output logic                           cmd_o_bank_o,
    output logic                           active_w_bank_o,
    output logic                           backend_clear_o,
    output logic                           profile_enable_o,
    output logic                           profile_clear_o,

    output logic [31:0]                    mvin_a_dram_base_o,
    output logic [31:0]                    mvin_a_dram_row_stride_bytes_o,
    output logic [15:0]                    mvin_a_m_o,
    output logic [15:0]                    mvin_a_k_o,
    output logic                           mvin_a_bank_o,
    output logic                           mvin_a_u8_minus_128_en_o,

    output logic [31:0]                    mvin_w_dram_base_o,
    output logic [15:0]                    mvin_w_k_o,
    output logic [15:0]                    mvin_w_n_o,
    output logic                           mvin_w_bank_o,

    output logic [31:0]                    mvin_meta_dram_base_o,
    output logic [31:0]                    mvin_meta_offset_bytes_o,
    output logic [31:0]                    mvin_meta_byte_count_o,
    output logic [1:0]                     mvin_meta_type_o,

    output logic [15:0]                    gemm_m_o,
    output logic [15:0]                    gemm_n_o,
    output logic [15:0]                    gemm_k_o,
    output logic                           gemm_a_bank_o,
    output logic                           gemm_w_bank_o,
    output logic                           gemm_accumulate_en_o,
    output logic                           gemm_add_bias_en_o,
    output npu_mode_e                      gemm_mode_o,
    output logic [31:0]                    gemm_bias_offset_bytes_o,

    output logic [31:0]                    mvout_dram_base_o,
    output logic [15:0]                    mvout_m_o,
    output logic [15:0]                    mvout_n_o,
    output logic [15:0]                    mvout_output_stride_n_o,
    output npu_mvout_mode_e                mvout_mode_o,
    output logic                           mvout_per_channel_scale_o,
    output logic [31:0]                    mvout_scale_param_o,

    output logic [31:0]                    attention_qk_output_dram_base_o,
    output logic [15:0]                    attention_qk_token_count_o,
    output logic [31:0]                    attention_qk_gamma16_fix_o,
    output logic                           attention_qk_mask_en_o,
    output logic [4:0]                     attention_qk_q_block_start_o,
    output logic [4:0]                     attention_qk_q_block_count_m1_o,

    input  wire logic                      mvin_a_done_i,
    input  wire logic                      mvin_w_done_i,
    input  wire logic                      mvin_meta_done_i,
    input  wire logic                      gemm_done_i,
    input  wire logic                      mvout_done_i,
    input  wire logic                      attention_qk_done_i,
    input  wire logic [5:0]                api_error_valid_i,
    input  wire logic [5:0][3:0]           api_error_code_i,

    input  wire logic [63:0]               profile_global_cycles_i,
    input  wire logic [63:0]               profile_mvin_a_busy_cycles_i,
    input  wire logic [63:0]               profile_mvin_w_busy_cycles_i,
    input  wire logic [63:0]               profile_mvin_meta_busy_cycles_i,
    input  wire logic [63:0]               profile_gemm_busy_cycles_i,
    input  wire logic [63:0]               profile_mvout_busy_cycles_i,
    input  wire logic [63:0]               profile_busy_any_cycles_i,
    input  wire logic [63:0]               profile_busy_multi_cycles_i,
    input  wire logic [3:0][63:0]          profile_axi_r_beats_i,
    input  wire logic [3:0][63:0]          profile_axi_w_beats_i,
    input  wire logic [3:0][63:0]          profile_axi_ar_stall_cycles_i,
    input  wire logic [3:0][63:0]          profile_axi_r_stall_cycles_i,
    input  wire logic [3:0][63:0]          profile_axi_aw_stall_cycles_i,
    input  wire logic [3:0][63:0]          profile_axi_w_stall_cycles_i
);
    localparam logic [2:0] CMD_MVIN_A    = 3'd0;
    localparam logic [2:0] CMD_MVIN_W    = 3'd1;
    localparam logic [2:0] CMD_META_MVIN = 3'd2;
    localparam logic [2:0] CMD_GEMM      = 3'd3;
    localparam logic [2:0] CMD_MVOUT     = 3'd4;
    localparam logic [2:0] CMD_ATTENTION_QK = 3'd5;

    localparam logic [3:0] ERR_NONE              = 4'd0;
    localparam logic [3:0] ERR_ILLEGAL_SHAPE     = 4'd1;
    localparam logic [3:0] ERR_START_WHILE_BUSY  = 4'd2;
    localparam logic [3:0] ERR_RESOURCE_CONFLICT = 4'd3;
    localparam logic [3:0] ERR_BANK_CONFLICT     = 4'd4;
    localparam logic [3:0] ERR_BANK_NOT_VALID    = 4'd5;
    localparam logic [3:0] ERR_SHAPE_MISMATCH    = 4'd6;
    localparam logic [3:0] ERR_ALIGNMENT         = 4'd7;
    localparam logic [3:0] ERR_UNSUPPORTED_MODE  = 4'd8;
    localparam logic [3:0] ERR_INTERNAL_ASSERT   = 4'd10;
    localparam logic [3:0] ERR_ILLEGAL_FLAGS     = 4'd11;

    localparam logic [11:0] REG_MVIN_A_DESC0    = 12'h000;
    localparam logic [11:0] REG_MVIN_A_DESC1    = 12'h008;
    localparam logic [11:0] REG_MVIN_A_STATUS   = 12'h010;
    localparam logic [11:0] REG_MVIN_W_DESC0    = 12'h020;
    localparam logic [11:0] REG_MVIN_W_DESC1    = 12'h028;
    localparam logic [11:0] REG_MVIN_W_STATUS   = 12'h030;
    localparam logic [11:0] REG_MVIN_META_DESC0 = 12'h040;
    localparam logic [11:0] REG_MVIN_META_DESC1 = 12'h048;
    localparam logic [11:0] REG_MVIN_META_STATUS = 12'h050;
    localparam logic [11:0] REG_GEMM_DESC0      = 12'h060;
    localparam logic [11:0] REG_GEMM_DESC1      = 12'h068;
    localparam logic [11:0] REG_GEMM_STATUS     = 12'h070;
    localparam logic [11:0] REG_MVOUT_DESC0     = 12'h080;
    localparam logic [11:0] REG_MVOUT_DESC1     = 12'h088;
    localparam logic [11:0] REG_MVOUT_STATUS    = 12'h090;
    localparam logic [11:0] REG_ATTENTION_QK_DESC0 = 12'h0a0;
    localparam logic [11:0] REG_ATTENTION_QK_DESC1 = 12'h0a8;
    localparam logic [11:0] REG_ATTENTION_QK_STATUS = 12'h0b0;
    localparam logic [11:0] REG_IAR             = 12'h100;
    localparam logic [11:0] REG_MER             = 12'h108;
    localparam logic [11:0] REG_IER             = 12'h110;
    localparam logic [11:0] REG_ISR             = 12'h118;
    localparam logic [11:0] REG_IPR             = 12'h120;
    localparam logic [11:0] REG_GLOBAL_CLEAR    = 12'h128;
    localparam logic [11:0] REG_PROFILE_CTRL    = 12'h130;
    localparam logic [11:0] REG_PROFILE_GLOBAL  = 12'h138;
    localparam logic [11:0] REG_PROFILE_A_BUSY  = 12'h140;
    localparam logic [11:0] REG_PROFILE_W_BUSY  = 12'h148;
    localparam logic [11:0] REG_PROFILE_META_BUSY = 12'h150;
    localparam logic [11:0] REG_PROFILE_GEMM_BUSY = 12'h158;
    localparam logic [11:0] REG_PROFILE_MVOUT_BUSY = 12'h160;
    localparam logic [11:0] REG_PROFILE_BUSY_ANY = 12'h168;
    localparam logic [11:0] REG_PROFILE_BUSY_MULTI = 12'h170;
    localparam logic [11:0] REG_PROFILE_AXI0_R_BEATS = 12'h178;
    localparam logic [11:0] REG_PROFILE_AXI1_R_BEATS = 12'h180;
    localparam logic [11:0] REG_PROFILE_AXI2_R_BEATS = 12'h188;
    localparam logic [11:0] REG_PROFILE_AXI3_R_BEATS = 12'h190;
    localparam logic [11:0] REG_PROFILE_AXI0_W_BEATS = 12'h198;
    localparam logic [11:0] REG_PROFILE_AXI1_W_BEATS = 12'h1a0;
    localparam logic [11:0] REG_PROFILE_AXI2_W_BEATS = 12'h1a8;
    localparam logic [11:0] REG_PROFILE_AXI3_W_BEATS = 12'h1b0;
    localparam logic [11:0] REG_PROFILE_AXI0_AR_STALL = 12'h1b8;
    localparam logic [11:0] REG_PROFILE_AXI1_AR_STALL = 12'h1c0;
    localparam logic [11:0] REG_PROFILE_AXI2_AR_STALL = 12'h1c8;
    localparam logic [11:0] REG_PROFILE_AXI3_AR_STALL = 12'h1d0;
    localparam logic [11:0] REG_PROFILE_AXI0_R_STALL = 12'h1d8;
    localparam logic [11:0] REG_PROFILE_AXI1_R_STALL = 12'h1e0;
    localparam logic [11:0] REG_PROFILE_AXI2_R_STALL = 12'h1e8;
    localparam logic [11:0] REG_PROFILE_AXI3_R_STALL = 12'h1f0;
    localparam logic [11:0] REG_PROFILE_AXI0_AW_STALL = 12'h1f8;
    localparam logic [11:0] REG_PROFILE_AXI1_AW_STALL = 12'h200;
    localparam logic [11:0] REG_PROFILE_AXI2_AW_STALL = 12'h208;
    localparam logic [11:0] REG_PROFILE_AXI3_AW_STALL = 12'h210;
    localparam logic [11:0] REG_PROFILE_AXI0_W_STALL = 12'h218;
    localparam logic [11:0] REG_PROFILE_AXI1_W_STALL = 12'h220;
    localparam logic [11:0] REG_PROFILE_AXI2_W_STALL = 12'h228;
    localparam logic [11:0] REG_PROFILE_AXI3_W_STALL = 12'h230;
    localparam logic [11:0] REG_GENERIC_MAGIC   = 12'hf00;
    localparam logic [11:0] REG_GENERIC_VERSION = 12'hf08;
    localparam logic [11:0] REG_GENERIC_MODE    = 12'hf10;
    localparam logic [11:0] REG_GENERIC_CAPS    = 12'hf18;
    localparam logic [11:0] REG_GENERIC_CONTROL = 12'hf20;
    localparam logic [11:0] REG_GENERIC_STATUS  = 12'hf28;
    localparam logic [11:0] REG_GENERIC_ERROR   = 12'hf30;

    localparam logic [31:0] GENERIC_MAGIC       = 32'h4e505547;
    localparam logic [31:0] GENERIC_ABI_VERSION = 32'd1;
    localparam logic [31:0] GENERIC_MODE_PREFILL = 32'd1;
    localparam logic [31:0] GENERIC_CAPS        = 32'h0000_0001;

    logic aw_valid_q;
    logic [AXIL_ADDR_W-1:0] awaddr_q;
    logic w_valid_q;
    logic [AXIL_DATA_W-1:0] wdata_q;
    logic [AXIL_STRB_W-1:0] wstrb_q;
    logic bvalid_q;
    logic rvalid_q;
    logic [AXIL_DATA_W-1:0] rdata_q;

    logic [63:0] mvin_a_desc0_q;
    logic [63:0] mvin_a_desc1_q;
    logic [63:0] mvin_w_desc0_q;
    logic [63:0] mvin_w_desc1_q;
    logic [63:0] mvin_meta_desc0_q;
    logic [63:0] mvin_meta_desc1_q;
    logic [63:0] gemm_desc0_q;
    logic [63:0] gemm_desc1_q;
    logic [63:0] mvout_desc0_q;
    logic [63:0] mvout_desc1_q;
    logic [63:0] attention_qk_desc0_q;
    logic [63:0] attention_qk_desc1_q;

    logic [5:0] status_busy_q;
    logic [5:0] status_done_q;
    logic [5:0] status_error_q;
    logic [5:0][3:0] status_error_code_q;
    logic [5:0][31:0] accepted_count_q;

    logic [6:0] isr_q;
    logic [6:0] ier_q;
    logic mer_master_enable_q;
    logic profile_enable_q;
    logic active_w_bank_q;
    logic [1:0] w_bank_loaded_q;
    logic [15:0] w_bank_k_q [0:1];
    logic [15:0] w_bank_n_q [0:1];
    logic mvin_w_busy_bank_q;
    logic mvin_a_busy_bank_q;
    logic gemm_busy_a_bank_q;
    logic gemm_busy_o_bank_q;
    logic mvout_busy_o_bank_q;
    logic attention_qk_busy_a_bank_q;
    logic attention_qk_busy_w_bank_q;
    logic attention_qk_busy_o_bank_q;
    logic gemm_busy_uses_w_bank_q;
    logic gemm_busy_w_bank_q;
    logic [15:0] cmd_mvin_w_k_q;
    logic [15:0] cmd_mvin_w_n_q;
    logic [15:0] mvin_w_busy_k_q;
    logic [15:0] mvin_w_busy_n_q;
    logic [11:0] gemm_attention_qk_m_blocks_m1;

    logic cmd_valid_q;
    logic [2:0] cmd_opcode_q;
    logic cmd_a_bank_q;
    logic cmd_w_bank_q;
    logic cmd_o_bank_q;
    logic [2:0] cmd_api_idx_q;
    logic cmd_gemm_attention_qk_q;
    logic pending_start_q;
    logic [2:0] pending_opcode_q;
    logic pending_a_bank_q;
    logic pending_w_bank_q;
    logic pending_o_bank_q;
    logic [2:0] pending_api_idx_q;
    logic [63:0] pending_desc0_q;
    logic [63:0] pending_desc1_q;
    logic unused_prot;

    wire write_fire = aw_valid_q && w_valid_q && !bvalid_q;
    wire read_fire = s_axi_arvalid_i && s_axi_arready_o;
    wire cmd_fire = cmd_valid_q && cmd_ready_i;

    assign s_axi_awready_o = !aw_valid_q && !pending_start_q;
    assign s_axi_wready_o  = !w_valid_q && !pending_start_q;
    assign s_axi_bvalid_o  = bvalid_q;
    assign s_axi_bresp_o   = 2'b00;
    assign s_axi_arready_o = !rvalid_q;
    assign s_axi_rvalid_o  = rvalid_q;
    assign s_axi_rdata_o   = rdata_q;
    assign s_axi_rresp_o   = 2'b00;
    assign cmd_valid_o     = cmd_valid_q;
    assign cmd_opcode_o    = cmd_opcode_q;
    assign cmd_a_bank_o    = cmd_a_bank_q;
    assign cmd_w_bank_o    = cmd_w_bank_q;
    assign cmd_o_bank_o    = cmd_o_bank_q;
    assign active_w_bank_o = active_w_bank_q;
    assign profile_enable_o = ENABLE_PROFILE ? profile_enable_q : 1'b0;
    assign irq_o           = mer_master_enable_q && |(isr_q & ier_q);
    assign unused_prot     = ^s_axi_awprot_i ^ ^s_axi_arprot_i;

    assign mvin_a_dram_base_o = mvin_a_desc0_q[31:0];
    assign mvin_a_dram_row_stride_bytes_o = mvin_a_desc0_q[63:32];
    assign mvin_a_m_o = mvin_a_desc1_q[15:0];
    assign mvin_a_k_o = mvin_a_desc1_q[31:16];
    assign mvin_a_u8_minus_128_en_o = mvin_a_desc1_q[32];
    assign mvin_a_bank_o = mvin_a_desc1_q[34];

    assign mvin_w_dram_base_o = mvin_w_desc0_q[31:0];
    assign mvin_w_k_o = mvin_w_desc1_q[15:0];
    assign mvin_w_n_o = mvin_w_desc1_q[31:16];
    assign mvin_w_bank_o = mvin_w_desc1_q[32];

    assign mvin_meta_dram_base_o = mvin_meta_desc0_q[31:0];
    assign mvin_meta_offset_bytes_o = mvin_meta_desc0_q[63:32];
    assign mvin_meta_byte_count_o = mvin_meta_desc1_q[31:0];
    assign mvin_meta_type_o = mvin_meta_desc1_q[33:32];

    assign gemm_m_o = gemm_desc0_q[15:0];
    assign gemm_n_o = gemm_desc0_q[31:16];
    assign gemm_k_o = gemm_desc0_q[47:32];
    assign gemm_w_bank_o = gemm_desc0_q[48];
    assign gemm_a_bank_o = gemm_desc0_q[52];
    assign gemm_accumulate_en_o = gemm_desc0_q[49];
    assign gemm_add_bias_en_o = gemm_desc0_q[50] &&
                                (gemm_desc0_q[55:54] != NPU_MODE_ATTENTION_QK);
    assign gemm_mode_o = npu_mode_e'(gemm_desc0_q[55:54]);
    assign gemm_bias_offset_bytes_o = gemm_desc1_q[31:0];
    assign gemm_attention_qk_m_blocks_m1 = {1'b0, gemm_desc0_q[15:5]} - 12'd1;

    assign mvout_dram_base_o = mvout_desc0_q[31:0];
    assign mvout_scale_param_o = mvout_desc0_q[63:32];
    assign mvout_m_o = mvout_desc1_q[15:0];
    assign mvout_n_o = mvout_desc1_q[31:16];
    assign mvout_output_stride_n_o = mvout_desc1_q[47:32];
    assign mvout_per_channel_scale_o = (mvout_desc1_q[49:48] == 2'd2);
    assign mvout_mode_o = (mvout_desc1_q[49:48] == 2'd0) ? NPU_MVOUT_RAW_I32 :
                          ((mvout_desc1_q[49:48] == 2'd1) ||
                           (mvout_desc1_q[49:48] == 2'd2)) ? NPU_MVOUT_FP32_Q8_24 :
                                                              NPU_MVOUT_ATTENTION_QK_LOGP;
    assign attention_qk_output_dram_base_o = 32'd0;
    assign attention_qk_token_count_o      = gemm_desc0_q[31:16];
    assign attention_qk_gamma16_fix_o      = gemm_desc1_q[31:0];
    assign attention_qk_mask_en_o          =
        (gemm_desc0_q[55:54] == NPU_MODE_ATTENTION_QK) && gemm_desc0_q[50];
    assign attention_qk_q_block_start_o    = gemm_desc1_q[41:37];
    assign attention_qk_q_block_count_m1_o =
        (gemm_desc0_q[15:5] == 11'd0) ? 5'd0 :
        gemm_attention_qk_m_blocks_m1[4:0];

    function automatic logic [63:0] apply_wstrb(
        input logic [63:0] old_value,
        input logic [63:0] new_value,
        input logic [7:0] strobe
    );
        logic [63:0] merged;
        begin
            merged = old_value;
            for (int byte_idx = 0; byte_idx < 8; byte_idx++) begin
                if (strobe[byte_idx]) begin
                    merged[byte_idx*8 +: 8] = new_value[byte_idx*8 +: 8];
                end
            end
            return merged;
        end
    endfunction

    function automatic logic [63:0] wstrb_mask64(input logic [7:0] strobe);
        logic [63:0] mask;
        begin
            mask = '0;
            for (int byte_idx = 0; byte_idx < 8; byte_idx++) begin
                if (strobe[byte_idx]) begin
                    mask[byte_idx*8 +: 8] = 8'hff;
                end
            end
            return mask;
        end
    endfunction

    function automatic logic [63:0] make_status(input int idx);
        logic [63:0] value;
        begin
            value = '0;
            value[0] = status_busy_q[idx];
            value[1] = status_done_q[idx];
            value[2] = status_error_q[idx];
            value[7:4] = status_error_code_q[idx];
            value[63:32] = accepted_count_q[idx];
            return value;
        end
    endfunction

    function automatic logic [3:0] map_core_error_code(input logic [3:0] core_code);
        logic [3:0] api_code;
        begin
            unique case (core_code)
                NPU_ERR_NONE: begin
                    api_code = ERR_NONE;
                end
                NPU_ERR_UNSUPPORTED_MODE: begin
                    api_code = ERR_UNSUPPORTED_MODE;
                end
                NPU_ERR_ILLEGAL_SHAPE,
                NPU_ERR_RANGE: begin
                    api_code = ERR_ILLEGAL_SHAPE;
                end
                NPU_ERR_BANK_CONFLICT: begin
                    api_code = ERR_BANK_CONFLICT;
                end
                NPU_ERR_ALIGN,
                NPU_ERR_ODD_DMA_BEAT,
                NPU_ERR_BAD_KEEP: begin
                    api_code = ERR_ALIGNMENT;
                end
                NPU_ERR_FIFO: begin
                    api_code = ERR_INTERNAL_ASSERT;
                end
                NPU_ERR_ILLEGAL_FLAGS: begin
                    api_code = ERR_ILLEGAL_FLAGS;
                end
                default: begin
                    api_code = ERR_INTERNAL_ASSERT;
                end
            endcase
            return api_code;
        end
    endfunction

    function automatic logic shape_error(
        input logic [2:0] opcode,
        input logic [63:0] desc0,
        input logic [63:0] desc1
    );
        logic err;
        logic gemm_attention_qk_mode;
        logic [31:0] gemm_attention_qk_row_end;
        begin
            err = 1'b0;
            gemm_attention_qk_mode =
                (opcode == CMD_GEMM) && (desc0[55:54] == NPU_MODE_ATTENTION_QK);
            gemm_attention_qk_row_end = {16'd0, desc1[47:32]} + {16'd0, desc0[15:0]};
            unique case (opcode)
                CMD_MVIN_A: begin
                    err = (desc1[15:0] == 16'd0) ||
                          (desc1[31:16] == 16'd0) ||
                          (desc1[31:16] > 16'd4096) ||
                          (desc0[63:32] < {16'd0, desc1[31:16]});
                end
                CMD_MVIN_W: begin
                    err = (desc1[15:0] == 16'd0) ||
                          (desc1[31:16] == 16'd0) ||
                          (desc1[15:0] > 16'd4096);
                end
                CMD_META_MVIN: begin
                    err = (desc1[31:0] == 32'd0);
                end
                CMD_GEMM: begin
                    if (gemm_attention_qk_mode) begin
                        err = (desc0[15:0] == 16'd0) ||
                              (desc0[4:0] != 5'd0) ||
                              (desc0[31:16] < 16'd32) ||
                              (desc0[31:16] > 16'd1024) ||
                              (desc0[20:16] != 5'd0) ||
                              (desc0[47:32] != 16'd64) ||
                              (desc1[31:0] == 32'd0) ||
                              (desc1[36:32] != 5'd0) ||
                              (gemm_attention_qk_row_end > {16'd0, desc0[31:16]});
                    end else begin
                        err = (desc0[15:0] == 16'd0) ||
                              (desc0[31:16] == 16'd0) ||
                              (desc0[47:32] == 16'd0) ||
                              (desc0[47:32] > 16'd4096);
                    end
                end
                CMD_MVOUT: begin
                    err = (desc1[15:0] == 16'd0) ||
                          (desc1[31:16] == 16'd0) ||
                          ((desc1[47:32] != 16'd0) &&
                           (desc1[47:32] < desc1[31:16]));
                end
                CMD_ATTENTION_QK: begin
                    err = (desc0[47:32] < 16'd32) ||
                          (desc0[47:32] > 16'd1024) ||
                          (desc0[36:32] != 5'd0) ||
                          (desc1[31:0] == 32'd0);
                end
                default: begin
                    err = 1'b1;
                end
            endcase
            return err;
        end
    endfunction

    function automatic logic [31:0] ceil_div32(input logic [15:0] value);
        ceil_div32 = ({16'd0, value} + 32'd31) >> 5;
    endfunction

    function automatic logic [31:0] ceil_div16_32(input logic [31:0] value);
        ceil_div16_32 = (value + 32'd15) >> 4;
    endfunction

    function automatic logic capacity_error(
        input logic [2:0] opcode,
        input logic [63:0] desc0,
        input logic [63:0] desc1
    );
        logic [31:0] words_needed;
        logic [31:0] meta_first_word;
        logic [31:0] meta_word_count;
        begin
            words_needed = 32'd0;
            meta_first_word = 32'd0;
            meta_word_count = 32'd0;
            capacity_error = 1'b0;
            unique case (opcode)
                CMD_MVIN_A: begin
                    words_needed = {16'd0, desc1[15:0]} * ceil_div32(desc1[31:16]);
                    capacity_error = words_needed > 32'(BANK_DEPTH_WORDS);
                end
                CMD_MVIN_W: begin
                    words_needed = {16'd0, desc1[15:0]} * ceil_div32(desc1[31:16]);
                    capacity_error = words_needed > 32'(BANK_DEPTH_WORDS);
                end
                CMD_META_MVIN: begin
                    meta_first_word = desc0[63:32] >> 4;
                    meta_word_count = ceil_div16_32(desc1[31:0]);
                    capacity_error = (meta_first_word + meta_word_count) > 32'(META_WORD128_DEPTH);
                end
                CMD_GEMM: begin
                    if (desc0[55:54] == NPU_MODE_ATTENTION_QK) begin
                        words_needed = {16'd0, desc0[15:0]} *
                                       ({16'd0, desc0[31:16]} >> 5);
                    end else begin
                        words_needed = {16'd0, desc0[15:0]} *
                                       (({16'd0, desc0[31:16]} + 32'd7) >> 3);
                    end
                    capacity_error = words_needed > 32'(BANK_DEPTH_WORDS);
                end
                CMD_MVOUT: begin
                    words_needed = {16'd0, desc1[15:0]} *
                                   (({16'd0, desc1[31:16]} + 32'd7) >> 3);
                    capacity_error = words_needed > 32'(BANK_DEPTH_WORDS);
                end
                CMD_ATTENTION_QK: begin
                    words_needed = {16'd0, desc0[47:32]};
                    capacity_error = words_needed > 32'(BANK_DEPTH_WORDS);
                end
                default: begin
                    capacity_error = 1'b1;
                end
            endcase
        end
    endfunction

    function automatic logic flag_error(
        input logic [2:0] opcode,
        input logic [63:0] desc0,
        input logic [63:0] desc1
    );
        logic err;
        begin
            err = 1'b0;
            if (opcode == CMD_GEMM) begin
                if (desc0[55:54] == NPU_MODE_ATTENTION_QK) begin
                    err = desc0[49];
                end else begin
                    err = (desc0[49] && desc0[50]) ||
                          ((desc0[55:54] == NPU_MODE_PV_LOG8_U16I8) && desc0[50]);
                end
            end
            return err;
        end
    endfunction

    function automatic logic unsupported_mode_error(
        input logic [2:0] opcode,
        input logic [63:0] desc0,
        input logic [63:0] desc1
    );
        logic err;
        begin
            err = 1'b0;
            unique case (opcode)
                CMD_GEMM: begin
                    err = (desc0[55:54] == NPU_MODE_BFP16M);
                end
                CMD_MVOUT: begin
                    err = 1'b0;
                end
                default: begin
                    err = 1'b0;
                end
            endcase
            return err;
        end
    endfunction

    function automatic logic align_error(
        input logic [2:0] opcode,
        input logic [63:0] desc0
    );
        logic err;
        begin
            err = 1'b0;
            unique case (opcode)
                CMD_MVIN_A: begin
                    err = (desc0[3:0] != 4'd0);
                end
                CMD_MVIN_W: begin
                    err = (desc0[3:0] != 4'd0);
                end
                CMD_META_MVIN: begin
                    err = (desc0[3:0] != 4'd0) ||
                          (desc0[35:32] != 4'd0);
                end
                CMD_MVOUT: begin
                    err = (desc0[3:0] != 4'd0);
                end
                CMD_ATTENTION_QK: begin
                    err = (desc0[3:0] != 4'd0);
                end
                default: begin
                    err = 1'b0;
                end
            endcase
            return err;
        end
    endfunction

    function automatic logic resource_conflict(
        input logic [2:0] opcode,
        input logic       a_bank,
        input logic       w_bank,
        input logic       o_bank,
        input logic [63:0] desc0
    );
        logic err;
        logic gemm_attention_qk_mode;
        begin
            err = 1'b0;
            gemm_attention_qk_mode =
                (opcode == CMD_GEMM) && (desc0[55:54] == NPU_MODE_ATTENTION_QK);
            unique case (opcode)
                CMD_MVIN_A: begin
                    err = (status_busy_q[3] && (gemm_busy_a_bank_q == a_bank)) ||
                          (status_busy_q[5] && (attention_qk_busy_a_bank_q == a_bank));
                end
                CMD_MVIN_W: begin
                    err = (status_busy_q[5] && (attention_qk_busy_w_bank_q == w_bank)) ||
                          (status_busy_q[3] && gemm_busy_uses_w_bank_q &&
                           (gemm_busy_w_bank_q == w_bank));
                end
                CMD_META_MVIN: begin
                    err = status_busy_q[3] || status_busy_q[4] || status_busy_q[5];
                end
                CMD_GEMM: begin
                    err = (status_busy_q[0] && (mvin_a_busy_bank_q == a_bank)) ||
                          (gemm_attention_qk_mode &&
                           status_busy_q[1] && (mvin_w_busy_bank_q == w_bank)) ||
                          status_busy_q[2] ||
                          (status_busy_q[4] && (mvout_busy_o_bank_q == o_bank)) ||
                          (status_busy_q[5] &&
                           ((attention_qk_busy_a_bank_q == a_bank) ||
                            (attention_qk_busy_o_bank_q == o_bank) ||
                            (attention_qk_busy_w_bank_q == w_bank)));
                end
                CMD_MVOUT: begin
                    err = status_busy_q[2] ||
                          (status_busy_q[3] && (gemm_busy_o_bank_q == o_bank)) ||
                          (status_busy_q[5] && (attention_qk_busy_o_bank_q == o_bank));
                end
                CMD_ATTENTION_QK: begin
                    err = (status_busy_q[0] && (mvin_a_busy_bank_q == a_bank)) ||
                          status_busy_q[2] ||
                          (status_busy_q[3] &&
                           ((gemm_busy_a_bank_q == a_bank) ||
                            (gemm_busy_o_bank_q == o_bank))) ||
                          (status_busy_q[4] && (mvout_busy_o_bank_q == o_bank));
                end
                default: begin
                    err = 1'b0;
                end
            endcase
            return err;
        end
    endfunction

    function automatic logic bank_conflict(
        input logic [2:0] opcode,
        input logic       w_bank
    );
        logic err;
        begin
            err = 1'b0;
            if (opcode == CMD_MVIN_W) begin
                err = (w_bank == active_w_bank_q);
            end
            return err;
        end
    endfunction

    function automatic logic bank_not_valid(
        input logic [2:0] opcode,
        input logic       w_bank
    );
        logic err;
        begin
            err = 1'b0;
            if ((opcode == CMD_GEMM) || (opcode == CMD_ATTENTION_QK)) begin
                err = !w_bank_loaded_q[w_bank];
            end
            return err;
        end
    endfunction

    function automatic logic bank_shape_mismatch(
        input logic [2:0]  opcode,
        input logic        w_bank,
        input logic [63:0] desc0
    );
        logic err;
        begin
            err = 1'b0;
            if ((opcode == CMD_GEMM) && w_bank_loaded_q[w_bank] &&
                (desc0[55:54] == NPU_MODE_ATTENTION_QK)) begin
                err = (w_bank_k_q[w_bank] != 16'd64) ||
                      (desc0[31:16] > w_bank_n_q[w_bank]);
            end else if ((opcode == CMD_GEMM) && w_bank_loaded_q[w_bank]) begin
                err = (desc0[47:32] != w_bank_k_q[w_bank]) ||
                      (desc0[31:16] > w_bank_n_q[w_bank]);
            end else if ((opcode == CMD_ATTENTION_QK) && w_bank_loaded_q[w_bank]) begin
                err = (w_bank_k_q[w_bank] != 16'd64) ||
                      (desc0[47:32] > w_bank_n_q[w_bank]);
            end
            return err;
        end
    endfunction

    function automatic logic [63:0] read_reg(input logic [11:0] addr);
        logic [63:0] value;
        begin
            value = '0;
            unique case (addr)
                REG_MVIN_A_DESC0: value = mvin_a_desc0_q;
                REG_MVIN_A_DESC1: value = mvin_a_desc1_q;
                REG_MVIN_A_STATUS: value = make_status(0);
                REG_MVIN_W_DESC0: value = mvin_w_desc0_q;
                REG_MVIN_W_DESC1: value = mvin_w_desc1_q;
                REG_MVIN_W_STATUS: value = make_status(1);
                REG_MVIN_META_DESC0: value = mvin_meta_desc0_q;
                REG_MVIN_META_DESC1: value = mvin_meta_desc1_q;
                REG_MVIN_META_STATUS: value = make_status(2);
                REG_GEMM_DESC0: value = gemm_desc0_q;
                REG_GEMM_DESC1: value = gemm_desc1_q;
                REG_GEMM_STATUS: value = make_status(3);
                REG_MVOUT_DESC0: value = mvout_desc0_q;
                REG_MVOUT_DESC1: value = mvout_desc1_q;
                REG_MVOUT_STATUS: value = make_status(4);
                REG_ATTENTION_QK_DESC0: value = attention_qk_desc0_q;
                REG_ATTENTION_QK_DESC1: value = attention_qk_desc1_q;
                REG_ATTENTION_QK_STATUS: value = make_status(5);
                REG_MER: value = {63'd0, mer_master_enable_q};
                REG_IER: value = {57'd0, ier_q};
                REG_ISR: value = {57'd0, isr_q};
                REG_IPR: value = {57'd0, isr_q & ier_q};
                REG_PROFILE_CTRL: value = ENABLE_PROFILE ? {62'd0, profile_enable_q, 1'b0} : '0;
                REG_PROFILE_GLOBAL: value = ENABLE_PROFILE ? profile_global_cycles_i : '0;
                REG_PROFILE_A_BUSY: value = ENABLE_PROFILE ? profile_mvin_a_busy_cycles_i : '0;
                REG_PROFILE_W_BUSY: value = ENABLE_PROFILE ? profile_mvin_w_busy_cycles_i : '0;
                REG_PROFILE_META_BUSY: value = ENABLE_PROFILE ? profile_mvin_meta_busy_cycles_i : '0;
                REG_PROFILE_GEMM_BUSY: value = ENABLE_PROFILE ? profile_gemm_busy_cycles_i : '0;
                REG_PROFILE_MVOUT_BUSY: value = ENABLE_PROFILE ? profile_mvout_busy_cycles_i : '0;
                REG_PROFILE_BUSY_ANY: value = ENABLE_PROFILE ? profile_busy_any_cycles_i : '0;
                REG_PROFILE_BUSY_MULTI: value = ENABLE_PROFILE ? profile_busy_multi_cycles_i : '0;
                REG_PROFILE_AXI0_R_BEATS: value = ENABLE_PROFILE ? profile_axi_r_beats_i[0] : '0;
                REG_PROFILE_AXI1_R_BEATS: value = ENABLE_PROFILE ? profile_axi_r_beats_i[1] : '0;
                REG_PROFILE_AXI2_R_BEATS: value = ENABLE_PROFILE ? profile_axi_r_beats_i[2] : '0;
                REG_PROFILE_AXI3_R_BEATS: value = ENABLE_PROFILE ? profile_axi_r_beats_i[3] : '0;
                REG_PROFILE_AXI0_W_BEATS: value = ENABLE_PROFILE ? profile_axi_w_beats_i[0] : '0;
                REG_PROFILE_AXI1_W_BEATS: value = ENABLE_PROFILE ? profile_axi_w_beats_i[1] : '0;
                REG_PROFILE_AXI2_W_BEATS: value = ENABLE_PROFILE ? profile_axi_w_beats_i[2] : '0;
                REG_PROFILE_AXI3_W_BEATS: value = ENABLE_PROFILE ? profile_axi_w_beats_i[3] : '0;
                REG_PROFILE_AXI0_AR_STALL: value = ENABLE_PROFILE ? profile_axi_ar_stall_cycles_i[0] : '0;
                REG_PROFILE_AXI1_AR_STALL: value = ENABLE_PROFILE ? profile_axi_ar_stall_cycles_i[1] : '0;
                REG_PROFILE_AXI2_AR_STALL: value = ENABLE_PROFILE ? profile_axi_ar_stall_cycles_i[2] : '0;
                REG_PROFILE_AXI3_AR_STALL: value = ENABLE_PROFILE ? profile_axi_ar_stall_cycles_i[3] : '0;
                REG_PROFILE_AXI0_R_STALL: value = ENABLE_PROFILE ? profile_axi_r_stall_cycles_i[0] : '0;
                REG_PROFILE_AXI1_R_STALL: value = ENABLE_PROFILE ? profile_axi_r_stall_cycles_i[1] : '0;
                REG_PROFILE_AXI2_R_STALL: value = ENABLE_PROFILE ? profile_axi_r_stall_cycles_i[2] : '0;
                REG_PROFILE_AXI3_R_STALL: value = ENABLE_PROFILE ? profile_axi_r_stall_cycles_i[3] : '0;
                REG_PROFILE_AXI0_AW_STALL: value = ENABLE_PROFILE ? profile_axi_aw_stall_cycles_i[0] : '0;
                REG_PROFILE_AXI1_AW_STALL: value = ENABLE_PROFILE ? profile_axi_aw_stall_cycles_i[1] : '0;
                REG_PROFILE_AXI2_AW_STALL: value = ENABLE_PROFILE ? profile_axi_aw_stall_cycles_i[2] : '0;
                REG_PROFILE_AXI3_AW_STALL: value = ENABLE_PROFILE ? profile_axi_aw_stall_cycles_i[3] : '0;
                REG_PROFILE_AXI0_W_STALL: value = ENABLE_PROFILE ? profile_axi_w_stall_cycles_i[0] : '0;
                REG_PROFILE_AXI1_W_STALL: value = ENABLE_PROFILE ? profile_axi_w_stall_cycles_i[1] : '0;
                REG_PROFILE_AXI2_W_STALL: value = ENABLE_PROFILE ? profile_axi_w_stall_cycles_i[2] : '0;
                REG_PROFILE_AXI3_W_STALL: value = ENABLE_PROFILE ? profile_axi_w_stall_cycles_i[3] : '0;
                REG_GENERIC_MAGIC: value = {32'd0, GENERIC_MAGIC};
                REG_GENERIC_VERSION: value = {32'd0, GENERIC_ABI_VERSION};
                REG_GENERIC_MODE: value = {32'd0, GENERIC_MODE_PREFILL};
                REG_GENERIC_CAPS: value = {32'd0, GENERIC_CAPS};
                REG_GENERIC_STATUS: value = {61'd0, |status_error_q,
                                             |status_busy_q || cmd_valid_q || pending_start_q,
                                             !(|status_busy_q || cmd_valid_q || pending_start_q)};
                REG_GENERIC_ERROR: value = {57'd0, |status_error_q, status_error_q};
                default: value = '0;
            endcase
            return value;
        end
    endfunction

    task automatic start_api(
        input logic [2:0] opcode,
        input logic       a_bank,
        input logic       w_bank,
        input logic       o_bank,
        input int         idx,
        input logic [63:0] desc0,
        input logic [63:0] desc1
    );
        if (DISABLE_ERROR_CHECKS) begin
            cmd_valid_q <= 1'b1;
            cmd_opcode_q <= opcode;
            cmd_a_bank_q <= a_bank;
            cmd_w_bank_q <= w_bank;
            cmd_o_bank_q <= o_bank;
            cmd_api_idx_q <= idx[2:0];
            cmd_gemm_attention_qk_q <=
                (opcode == CMD_GEMM) && (desc0[55:54] == NPU_MODE_ATTENTION_QK);
            if (opcode == CMD_MVIN_W) begin
                cmd_mvin_w_k_q <= desc1[15:0];
                cmd_mvin_w_n_q <= desc1[31:16];
            end
        end else if (cmd_valid_q || status_busy_q[idx]) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_START_WHILE_BUSY;
            isr_q[6] <= 1'b1;
        end else if (shape_error(opcode, desc0, desc1)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_ILLEGAL_SHAPE;
            isr_q[6] <= 1'b1;
        end else if (capacity_error(opcode, desc0, desc1)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_ILLEGAL_SHAPE;
            isr_q[6] <= 1'b1;
        end else if (flag_error(opcode, desc0, desc1)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_ILLEGAL_FLAGS;
            isr_q[6] <= 1'b1;
        end else if (unsupported_mode_error(opcode, desc0, desc1)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_UNSUPPORTED_MODE;
            isr_q[6] <= 1'b1;
        end else if (align_error(opcode, desc0)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_ALIGNMENT;
            isr_q[6] <= 1'b1;
        end else if (resource_conflict(opcode, a_bank, w_bank, o_bank, desc0)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_RESOURCE_CONFLICT;
            isr_q[6] <= 1'b1;
        end else if (bank_conflict(opcode, w_bank)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_BANK_CONFLICT;
            isr_q[6] <= 1'b1;
        end else if (bank_not_valid(opcode, w_bank)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_BANK_NOT_VALID;
            isr_q[6] <= 1'b1;
        end else if (bank_shape_mismatch(opcode, w_bank, desc0)) begin
            status_error_q[idx] <= 1'b1;
            status_error_code_q[idx] <= ERR_SHAPE_MISMATCH;
            isr_q[6] <= 1'b1;
        end else begin
            cmd_valid_q <= 1'b1;
            cmd_opcode_q <= opcode;
            cmd_a_bank_q <= a_bank;
            cmd_w_bank_q <= w_bank;
            cmd_o_bank_q <= o_bank;
            cmd_api_idx_q <= idx[2:0];
            cmd_gemm_attention_qk_q <=
                (opcode == CMD_GEMM) && (desc0[55:54] == NPU_MODE_ATTENTION_QK);
            if (opcode == CMD_MVIN_W) begin
                cmd_mvin_w_k_q <= desc1[15:0];
                cmd_mvin_w_n_q <= desc1[31:16];
            end
        end
    endtask

    task automatic queue_start_api(
        input logic [2:0] opcode,
        input logic       a_bank,
        input logic       w_bank,
        input logic       o_bank,
        input logic [2:0] idx,
        input logic [63:0] desc0,
        input logic [63:0] desc1
    );
        pending_start_q <= 1'b1;
        pending_opcode_q <= opcode;
        pending_a_bank_q <= a_bank;
        pending_w_bank_q <= w_bank;
        pending_o_bank_q <= o_bank;
        pending_api_idx_q <= idx;
        pending_desc0_q <= desc0;
        pending_desc1_q <= desc1;
    endtask

    task automatic clear_status_bits(input int idx, input logic [63:0] value);
        if (value[1]) begin
            status_done_q[idx] <= 1'b0;
        end
        if (value[2]) begin
            status_error_q[idx] <= 1'b0;
            status_error_code_q[idx] <= ERR_NONE;
        end
    endtask

    always_ff @(posedge clk_i) begin
        logic [11:0] write_addr;
        logic [63:0] write_data;
        logic [63:0] write_w1c;

        if (rst_i) begin
            aw_valid_q <= 1'b0;
            awaddr_q <= '0;
            w_valid_q <= 1'b0;
            wdata_q <= '0;
            wstrb_q <= '0;
            bvalid_q <= 1'b0;
            rvalid_q <= 1'b0;
            rdata_q <= '0;
            mvin_a_desc0_q <= '0;
            mvin_a_desc1_q <= '0;
            mvin_w_desc0_q <= '0;
            mvin_w_desc1_q <= '0;
            mvin_meta_desc0_q <= '0;
            mvin_meta_desc1_q <= '0;
            gemm_desc0_q <= '0;
            gemm_desc1_q <= '0;
            mvout_desc0_q <= '0;
            mvout_desc1_q <= '0;
            attention_qk_desc0_q <= '0;
            attention_qk_desc1_q <= '0;
            status_busy_q <= '0;
            status_done_q <= '0;
            status_error_q <= '0;
            status_error_code_q <= '0;
            accepted_count_q <= '0;
            isr_q <= '0;
            ier_q <= '0;
            mer_master_enable_q <= 1'b0;
            profile_enable_q <= 1'b0;
            profile_clear_o <= 1'b0;
            active_w_bank_q <= 1'b1;
            w_bank_loaded_q <= '0;
            for (int bank_idx = 0; bank_idx < 2; bank_idx++) begin
                w_bank_k_q[bank_idx] <= '0;
                w_bank_n_q[bank_idx] <= '0;
            end
            mvin_w_busy_bank_q <= 1'b0;
            mvin_a_busy_bank_q <= 1'b0;
            gemm_busy_a_bank_q <= 1'b0;
            gemm_busy_o_bank_q <= 1'b0;
            gemm_busy_uses_w_bank_q <= 1'b0;
            gemm_busy_w_bank_q <= 1'b0;
            mvout_busy_o_bank_q <= 1'b0;
            attention_qk_busy_a_bank_q <= 1'b0;
            attention_qk_busy_w_bank_q <= 1'b0;
            attention_qk_busy_o_bank_q <= 1'b0;
            cmd_mvin_w_k_q <= '0;
            cmd_mvin_w_n_q <= '0;
            mvin_w_busy_k_q <= '0;
            mvin_w_busy_n_q <= '0;
            cmd_valid_q <= 1'b0;
            cmd_opcode_q <= CMD_MVIN_A;
            cmd_a_bank_q <= 1'b0;
            cmd_w_bank_q <= 1'b0;
            cmd_o_bank_q <= 1'b0;
            cmd_api_idx_q <= '0;
            cmd_gemm_attention_qk_q <= 1'b0;
            pending_start_q <= 1'b0;
            pending_opcode_q <= CMD_MVIN_A;
            pending_a_bank_q <= 1'b0;
            pending_w_bank_q <= 1'b0;
            pending_o_bank_q <= 1'b0;
            pending_api_idx_q <= '0;
            pending_desc0_q <= '0;
            pending_desc1_q <= '0;
            backend_clear_o <= 1'b0;
        end else begin
            backend_clear_o <= 1'b0;
            profile_clear_o <= 1'b0;

            if (s_axi_awvalid_i && s_axi_awready_o) begin
                aw_valid_q <= 1'b1;
                awaddr_q <= s_axi_awaddr_i;
            end
            if (s_axi_wvalid_i && s_axi_wready_o) begin
                w_valid_q <= 1'b1;
                wdata_q <= s_axi_wdata_i;
                wstrb_q <= s_axi_wstrb_i;
            end
            if (bvalid_q && s_axi_bready_i) begin
                bvalid_q <= 1'b0;
            end
            if (rvalid_q && s_axi_rready_i) begin
                rvalid_q <= 1'b0;
            end
            if (read_fire) begin
                rvalid_q <= 1'b1;
                rdata_q <= read_reg(s_axi_araddr_i[11:0]);
            end

            if (cmd_fire) begin
                cmd_valid_q <= 1'b0;
                status_busy_q[cmd_api_idx_q] <= 1'b1;
                accepted_count_q[cmd_api_idx_q] <= accepted_count_q[cmd_api_idx_q] + 32'd1;
                if (cmd_opcode_q == CMD_GEMM) begin
                    if (!cmd_gemm_attention_qk_q) begin
                        active_w_bank_q <= cmd_w_bank_q;
                    end
                    gemm_busy_a_bank_q <= cmd_a_bank_q;
                    gemm_busy_o_bank_q <= cmd_o_bank_q;
                    gemm_busy_uses_w_bank_q <= cmd_gemm_attention_qk_q;
                    gemm_busy_w_bank_q <= cmd_w_bank_q;
                end
                if (cmd_opcode_q == CMD_MVIN_A) begin
                    mvin_a_busy_bank_q <= cmd_a_bank_q;
                end
                if (cmd_opcode_q == CMD_MVIN_W) begin
                    w_bank_loaded_q[cmd_w_bank_q] <= 1'b0;
                    mvin_w_busy_bank_q <= cmd_w_bank_q;
                    mvin_w_busy_k_q <= cmd_mvin_w_k_q;
                    mvin_w_busy_n_q <= cmd_mvin_w_n_q;
                end
                if (cmd_opcode_q == CMD_MVOUT) begin
                    mvout_busy_o_bank_q <= cmd_o_bank_q;
                end
                if (cmd_opcode_q == CMD_ATTENTION_QK) begin
                    attention_qk_busy_a_bank_q <= cmd_a_bank_q;
                    attention_qk_busy_w_bank_q <= cmd_w_bank_q;
                    attention_qk_busy_o_bank_q <= cmd_o_bank_q;
                end
            end

            if (pending_start_q && (!DISABLE_ERROR_CHECKS || !cmd_valid_q || cmd_fire)) begin
                start_api(
                    pending_opcode_q,
                    pending_a_bank_q,
                    pending_w_bank_q,
                    pending_o_bank_q,
                    pending_api_idx_q,
                    pending_desc0_q,
                    pending_desc1_q
                );
                pending_start_q <= 1'b0;
            end

            if (write_fire) begin
                write_addr = awaddr_q[11:0];
                write_data = apply_wstrb(read_reg(write_addr), wdata_q, wstrb_q);
                write_w1c = wdata_q & wstrb_mask64(wstrb_q);
                aw_valid_q <= 1'b0;
                w_valid_q <= 1'b0;
                bvalid_q <= 1'b1;

                unique case (write_addr)
                    REG_MVIN_A_DESC0: begin
                        mvin_a_desc0_q <= write_data;
                    end
                    REG_MVIN_A_DESC1: begin
                        mvin_a_desc1_q <= write_data & ~(64'(1) << 33);
                        if (write_data[33]) begin
                            queue_start_api(CMD_MVIN_A, write_data[34], 1'b0, 1'b0, 3'd0, mvin_a_desc0_q, write_data);
                        end
                    end
                    REG_MVIN_A_STATUS: begin
                        clear_status_bits(0, write_w1c);
                    end
                    REG_MVIN_W_DESC0: begin
                        mvin_w_desc0_q <= write_data;
                    end
                    REG_MVIN_W_DESC1: begin
                        mvin_w_desc1_q <= write_data & ~(64'(1) << 33);
                        if (write_data[33]) begin
                            queue_start_api(CMD_MVIN_W, 1'b0, write_data[32], 1'b0, 3'd1, mvin_w_desc0_q, write_data);
                        end
                    end
                    REG_MVIN_W_STATUS: begin
                        clear_status_bits(1, write_w1c);
                    end
                    REG_MVIN_META_DESC0: begin
                        mvin_meta_desc0_q <= write_data;
                    end
                    REG_MVIN_META_DESC1: begin
                        mvin_meta_desc1_q <= write_data & ~(64'(1) << 34);
                        if (write_data[34]) begin
                            queue_start_api(CMD_META_MVIN, 1'b0, 1'b0, 1'b0, 3'd2, mvin_meta_desc0_q, write_data);
                        end
                    end
                    REG_MVIN_META_STATUS: begin
                        clear_status_bits(2, write_w1c);
                    end
                    REG_GEMM_DESC0: begin
                        gemm_desc0_q <= write_data & ~(64'(1) << 51);
                        if (write_data[51]) begin
                            queue_start_api(CMD_GEMM, write_data[52], write_data[48], write_data[53], 3'd3, write_data, gemm_desc1_q);
                        end
                    end
                    REG_GEMM_DESC1: begin
                        gemm_desc1_q <= write_data;
                    end
                    REG_GEMM_STATUS: begin
                        clear_status_bits(3, write_w1c);
                    end
                    REG_MVOUT_DESC0: begin
                        mvout_desc0_q <= write_data;
                    end
                    REG_MVOUT_DESC1: begin
                        mvout_desc1_q <= write_data & ~(64'(1) << 50);
                        if (write_data[50]) begin
                            queue_start_api(CMD_MVOUT, 1'b0, 1'b0, write_data[51], 3'd4, mvout_desc0_q, write_data);
                        end
                    end
                    REG_MVOUT_STATUS: begin
                        clear_status_bits(4, write_w1c);
                    end
                    REG_ATTENTION_QK_DESC0: begin
                        attention_qk_desc0_q <= write_data & ~(64'(1) << 63);
                        if (write_data[63]) begin
                            status_error_q[5] <= 1'b1;
                            status_error_code_q[5] <= ERR_UNSUPPORTED_MODE;
                            isr_q[6] <= 1'b1;
                        end
                    end
                    REG_ATTENTION_QK_DESC1: begin
                        attention_qk_desc1_q <= write_data;
                    end
                    REG_ATTENTION_QK_STATUS: begin
                        clear_status_bits(5, write_w1c);
                    end
                    REG_IAR, REG_ISR: begin
                        isr_q <= isr_q & ~write_w1c[6:0];
                    end
                    REG_MER: begin
                        mer_master_enable_q <= write_data[0];
                    end
                    REG_IER: begin
                        ier_q <= write_data[6:0];
                    end
                    REG_PROFILE_CTRL: begin
                        if (ENABLE_PROFILE) begin
                            profile_enable_q <= write_data[1];
                            if (write_w1c[0]) begin
                                profile_clear_o <= 1'b1;
                            end
                        end
                    end
                    REG_GENERIC_CONTROL: begin
                        if (write_w1c[0]) begin
                            status_busy_q <= '0;
                            status_done_q <= '0;
                            status_error_q <= '0;
                            status_error_code_q <= '0;
                            active_w_bank_q <= 1'b1;
                            w_bank_loaded_q <= '0;
                            mvin_w_busy_bank_q <= 1'b0;
                            mvin_a_busy_bank_q <= 1'b0;
                            gemm_busy_a_bank_q <= 1'b0;
                            gemm_busy_o_bank_q <= 1'b0;
                            gemm_busy_uses_w_bank_q <= 1'b0;
                            gemm_busy_w_bank_q <= 1'b0;
                            mvout_busy_o_bank_q <= 1'b0;
                            attention_qk_busy_a_bank_q <= 1'b0;
                            attention_qk_busy_w_bank_q <= 1'b0;
                            attention_qk_busy_o_bank_q <= 1'b0;
                            cmd_valid_q <= 1'b0;
                            pending_start_q <= 1'b0;
                            isr_q <= '0;
                            backend_clear_o <= 1'b1;
                            if (ENABLE_PROFILE) begin
                                profile_clear_o <= 1'b1;
                            end
                        end else begin
                            if (write_w1c[1]) begin
                                isr_q <= '0;
                            end
                            if (write_w1c[2]) begin
                                status_error_q <= '0;
                                status_error_code_q <= '0;
                            end
                        end
                    end
                    REG_GLOBAL_CLEAR: begin
                        if (write_w1c[0]) begin
                            status_busy_q <= '0;
                            status_done_q <= '0;
                            status_error_q <= '0;
                            status_error_code_q <= '0;
                            active_w_bank_q <= 1'b1;
                            w_bank_loaded_q <= '0;
                            mvin_w_busy_bank_q <= 1'b0;
                            mvin_a_busy_bank_q <= 1'b0;
                            gemm_busy_a_bank_q <= 1'b0;
                            gemm_busy_o_bank_q <= 1'b0;
                            mvout_busy_o_bank_q <= 1'b0;
                            attention_qk_busy_a_bank_q <= 1'b0;
                            attention_qk_busy_w_bank_q <= 1'b0;
                            attention_qk_busy_o_bank_q <= 1'b0;
                            cmd_valid_q <= 1'b0;
                            pending_start_q <= 1'b0;
                            isr_q <= '0;
                            backend_clear_o <= 1'b1;
                            if (ENABLE_PROFILE) begin
                                profile_clear_o <= 1'b1;
                            end
                        end
                    end
                    default: begin
                    end
                endcase
            end

            if (mvin_a_done_i) begin
                status_busy_q[0] <= 1'b0;
                status_done_q[0] <= 1'b1;
                isr_q[0] <= 1'b1;
            end
            if (mvin_w_done_i) begin
                status_busy_q[1] <= 1'b0;
                status_done_q[1] <= 1'b1;
                if (mvin_w_busy_bank_q) begin
                    w_bank_loaded_q[1] <= 1'b1;
                    w_bank_k_q[1] <= mvin_w_busy_k_q;
                    w_bank_n_q[1] <= mvin_w_busy_n_q;
                end else begin
                    w_bank_loaded_q[0] <= 1'b1;
                    w_bank_k_q[0] <= mvin_w_busy_k_q;
                    w_bank_n_q[0] <= mvin_w_busy_n_q;
                end
                isr_q[1] <= 1'b1;
            end
            if (mvin_meta_done_i) begin
                status_busy_q[2] <= 1'b0;
                status_done_q[2] <= 1'b1;
                isr_q[2] <= 1'b1;
            end
            if (gemm_done_i) begin
                status_busy_q[3] <= 1'b0;
                status_done_q[3] <= 1'b1;
                gemm_busy_uses_w_bank_q <= 1'b0;
                isr_q[3] <= 1'b1;
            end
            if (mvout_done_i) begin
                status_busy_q[4] <= 1'b0;
                status_done_q[4] <= 1'b1;
                isr_q[4] <= 1'b1;
            end
            if (attention_qk_done_i) begin
                status_busy_q[5] <= 1'b0;
                status_done_q[5] <= 1'b1;
                isr_q[5] <= 1'b1;
            end
            if (!DISABLE_ERROR_CHECKS) begin
                for (int api_idx = 0; api_idx < 6; api_idx++) begin
                    if (api_error_valid_i[api_idx]) begin
                        status_busy_q[api_idx] <= 1'b0;
                        status_error_q[api_idx] <= 1'b1;
                        status_error_code_q[api_idx] <= map_core_error_code(api_error_code_i[api_idx]);
                        isr_q[6] <= 1'b1;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
