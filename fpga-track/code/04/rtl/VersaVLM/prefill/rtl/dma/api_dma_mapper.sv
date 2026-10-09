`default_nettype none

import npu_spm_pkg::*;

// Maps the final Versa_P API descriptors onto the three DMA command lanes.
// This module does not implement DMA traffic; it only emits request pulses and
// stable descriptor vectors for the direct DMA backend.
module api_dma_mapper #(
    parameter int RF_DATA_WIDTH = 64,
    parameter int DMA_NUM = 3,
    parameter int A_DMA_IDX = 0,
    parameter int W_DMA_IDX = 1,
    parameter int OC_DMA_IDX = 2,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0,
    localparam int HALF_W = RF_DATA_WIDTH / 2,
    localparam int QUARTER_W = RF_DATA_WIDTH / 4
) (
    input  wire logic                                  clk_i,
    input  wire logic                                  rst_i,

    input  wire logic                                  cmd_valid_i,
    output logic                                       cmd_ready_o,
    input  wire logic [2:0]                            cmd_opcode_i,

    input  wire logic [31:0]                           mvin_a_dram_base_i,
    input  wire logic [31:0]                           mvin_a_dram_row_stride_bytes_i,
    input  wire logic [15:0]                           mvin_a_m_i,
    input  wire logic [15:0]                           mvin_a_k_i,

    input  wire logic [31:0]                           mvin_w_dram_base_i,
    input  wire logic [15:0]                           mvin_w_k_i,
    input  wire logic [15:0]                           mvin_w_n_i,

    input  wire logic [31:0]                           mvin_meta_dram_base_i,
    input  wire logic [31:0]                           mvin_meta_offset_bytes_i,
    input  wire logic [31:0]                           mvin_meta_byte_count_i,

    input  wire logic [31:0]                           mvout_dram_base_i,
    input  wire logic [15:0]                           mvout_m_i,
    input  wire logic [15:0]                           mvout_n_i,
    input  wire logic [15:0]                           mvout_output_stride_n_i,
    input  wire npu_mvout_mode_e                       mvout_mode_i,
    input  wire logic                                  mvout_per_channel_scale_i,
    input  wire logic [31:0]                           mvout_scale_param_i,

    input  wire logic [31:0]                           attention_qk_output_dram_base_i,
    input  wire logic [15:0]                           attention_qk_token_count_i,
    input  wire logic [31:0]                           attention_qk_gamma16_fix_i,
    input  wire logic                                  attention_qk_mask_en_i,
    input  wire logic [4:0]                            attention_qk_q_block_start_i,
    input  wire logic [4:0]                            attention_qk_q_block_count_m1_i,

    input  wire logic [DMA_NUM-1:0]                    dma_mvin_busy_i,
    input  wire logic [DMA_NUM-1:0]                    dma_mvout_busy_i,
    input  wire logic [DMA_NUM-1:0]                    loader_mvin_done_i,
    input  wire logic [DMA_NUM-1:0]                    dma_mvout_done_i,
    input  wire logic                                  core_sa_done_i,
    input  wire logic                                  core_attention_qk_done_i,

    input  wire logic                                  core_error_sticky_i,
    input  wire npu_error_e                            core_last_error_i,

    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvin_dram_addr_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvin_sram_addr_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvin_col_num_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvin_row_num_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvout_dram_addr_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvout_sram_addr_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvout_col_num_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             mvout_row_num_o,
    output logic [DMA_NUM-1:0][1:0]                    cfg_mvin_input_type_o,
    output logic [DMA_NUM-1:0][1:0]                    cfg_mvout_output_type_o,
    output logic [DMA_NUM-1:0][1:0]                    cfg_mvin_input_precision_o,
    output logic [DMA_NUM-1:0][1:0]                    cfg_mvout_output_precision_o,
    output logic [DMA_NUM-1:0]                         cfg_mvin_is_quant_o,
    output logic [DMA_NUM-1:0]                         cfg_mvout_is_quant_o,
    output logic [DMA_NUM-1:0]                         cfg_mvin_dest_o,
    output logic [DMA_NUM-1:0]                         cfg_mvout_source_o,
    output logic [DMA_NUM-1:0][QUARTER_W-1:0]          cfg_mvin_sram_stride_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             cfg_mvin_dram_stride_o,
    output logic [DMA_NUM-1:0][QUARTER_W-1:0]          cfg_mvout_sram_stride_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             cfg_mvout_dram_stride_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             cfg_mvin_input_zeropoint_o,
    output logic [DMA_NUM-1:0][HALF_W-1:0]             cfg_mvout_output_zeropoint_o,
    output logic [DMA_NUM-1:0][QUARTER_W-1:0]          cfg_mvin_input_scale_o,
    output logic [DMA_NUM-1:0][QUARTER_W-1:0]          cfg_mvout_output_scale_o,
    output logic [DMA_NUM-1:0][QUARTER_W-1:0]          cfg_mvin_input_scale_shift_o,
    output logic [DMA_NUM-1:0][QUARTER_W-1:0]          cfg_mvout_output_scale_shift_o,
    output logic [DMA_NUM-1:0][1:0]                   cfg_mvout_mode_o,
    output logic [DMA_NUM-1:0]                         cfg_mvout_per_channel_o,
    output logic [DMA_NUM-1:0][31:0]                   cfg_mvout_scale_param_o,
    output logic [DMA_NUM-1:0]                         dma_mvin_req_en_o,
    output logic [DMA_NUM-1:0]                         dma_mvout_req_en_o,
    output logic                                       sa_req_en_o,
    output logic                                       attention_qk_req_en_o,

    output logic                                       mvin_a_done_o,
    output logic                                       mvin_w_done_o,
    output logic                                       mvin_meta_done_o,
    output logic                                       gemm_done_o,
    output logic                                       mvout_done_o,
    output logic                                       attention_qk_done_o,
    output logic [5:0]                                 api_error_valid_o,
    output logic [5:0][3:0]                            api_error_code_o
);
    localparam logic [2:0] CMD_MVIN_A    = 3'd0;
    localparam logic [2:0] CMD_MVIN_W    = 3'd1;
    localparam logic [2:0] CMD_META_MVIN = 3'd2;
    localparam logic [2:0] CMD_GEMM      = 3'd3;
    localparam logic [2:0] CMD_MVOUT     = 3'd4;
    localparam logic [2:0] CMD_ATTENTION_QK = 3'd5;

    logic [5:0] inflight_q;
    logic core_error_seen_q;
    logic core_error_sticky_q;
    logic [3:0] core_last_error_q;
    logic [5:0] api_error_valid_q;
    logic [5:0][3:0] api_error_code_q;
    logic cmd_supported_dma;
    logic cmd_target_idle;
    logic cmd_fire;

    assign cmd_supported_dma = (cmd_opcode_i == CMD_MVIN_A) ||
                               (cmd_opcode_i == CMD_MVIN_W) ||
                               (cmd_opcode_i == CMD_META_MVIN) ||
                               (cmd_opcode_i == CMD_GEMM) ||
                               (cmd_opcode_i == CMD_MVOUT);

    always_comb begin
        cmd_target_idle = 1'b0;
        unique case (cmd_opcode_i)
            CMD_MVIN_A: begin
                cmd_target_idle = !inflight_q[0] && !dma_mvin_busy_i[A_DMA_IDX];
            end
            CMD_MVIN_W: begin
                cmd_target_idle = !inflight_q[1] && !dma_mvin_busy_i[W_DMA_IDX];
            end
            CMD_META_MVIN: begin
                cmd_target_idle = !inflight_q[2] && !dma_mvin_busy_i[OC_DMA_IDX];
            end
            CMD_MVOUT: begin
                cmd_target_idle = !inflight_q[4] && !dma_mvout_busy_i[OC_DMA_IDX];
            end
            CMD_GEMM: begin
                cmd_target_idle = !inflight_q[3];
            end
            default: begin
                cmd_target_idle = 1'b0;
            end
        endcase
    end

    assign cmd_ready_o = cmd_valid_i && cmd_target_idle;
    assign cmd_fire = cmd_valid_i && cmd_ready_o;

    function automatic logic [15:0] ceil_div32_16(input logic [15:0] value);
        ceil_div32_16 = (value + 16'd31) >> 5;
    endfunction

    function automatic logic [31:0] ceil_div32_32(input logic [31:0] value);
        ceil_div32_32 = (value + 32'd31) >> 5;
    endfunction

    function automatic logic [31:0] ceil_div16_32(input logic [31:0] value);
        ceil_div16_32 = (value + 32'd15) >> 4;
    endfunction

    function automatic logic [QUARTER_W-1:0] tight_a_stride(input logic [15:0] k);
        tight_a_stride = QUARTER_W'(ceil_div32_16(k) << 5);
    endfunction

    always_comb begin
        mvin_dram_addr_o = '0;
        mvin_sram_addr_o = '0;
        mvin_col_num_o = '0;
        mvin_row_num_o = '0;
        mvout_dram_addr_o = '0;
        mvout_sram_addr_o = '0;
        mvout_col_num_o = '0;
        mvout_row_num_o = '0;
        cfg_mvin_input_type_o = '0;
        cfg_mvout_output_type_o = '0;
        cfg_mvin_input_precision_o = '0;
        cfg_mvout_output_precision_o = '0;
        cfg_mvin_is_quant_o = '0;
        cfg_mvout_is_quant_o = '0;
        cfg_mvin_dest_o = '0;
        cfg_mvout_source_o = '0;
        cfg_mvin_sram_stride_o = '0;
        cfg_mvin_dram_stride_o = '0;
        cfg_mvout_sram_stride_o = '0;
        cfg_mvout_dram_stride_o = '0;
        cfg_mvin_input_zeropoint_o = '0;
        cfg_mvout_output_zeropoint_o = '0;
        cfg_mvin_input_scale_o = '0;
        cfg_mvout_output_scale_o = '0;
        cfg_mvin_input_scale_shift_o = '0;
        cfg_mvout_output_scale_shift_o = '0;
        cfg_mvout_per_channel_o = '0;
        cfg_mvout_scale_param_o = '0;
        for (int dma_idx = 0; dma_idx < DMA_NUM; dma_idx++) begin
            cfg_mvout_mode_o[dma_idx] = NPU_MVOUT_RAW_I32;
        end
        dma_mvin_req_en_o = '0;
        dma_mvout_req_en_o = '0;
        sa_req_en_o = 1'b0;
        attention_qk_req_en_o = 1'b0;

        cfg_mvin_input_precision_o[A_DMA_IDX] = 2'b01;
        cfg_mvin_input_precision_o[W_DMA_IDX] = 2'b01;
        cfg_mvin_input_precision_o[OC_DMA_IDX] = 2'b01;

        mvin_dram_addr_o[A_DMA_IDX] = HALF_W'(mvin_a_dram_base_i);
        mvin_col_num_o[A_DMA_IDX] = HALF_W'({16'd0, mvin_a_k_i} - 32'd1);
        mvin_row_num_o[A_DMA_IDX] = HALF_W'({16'd0, mvin_a_m_i} - 32'd1);
        cfg_mvin_sram_stride_o[A_DMA_IDX] = tight_a_stride(mvin_a_k_i);
        cfg_mvin_dram_stride_o[A_DMA_IDX] = HALF_W'(mvin_a_dram_row_stride_bytes_i);

        mvin_dram_addr_o[W_DMA_IDX] = HALF_W'(mvin_w_dram_base_i);
        mvin_col_num_o[W_DMA_IDX] = HALF_W'(32'd31);
        mvin_row_num_o[W_DMA_IDX] =
            HALF_W'(({16'd0, mvin_w_k_i} * {16'd0, ceil_div32_16(mvin_w_n_i)}) - 32'd1);
        cfg_mvin_sram_stride_o[W_DMA_IDX] = QUARTER_W'(16'd32);
        cfg_mvin_dram_stride_o[W_DMA_IDX] = HALF_W'(32'd32);

        mvin_dram_addr_o[OC_DMA_IDX] = HALF_W'(mvin_meta_dram_base_i);
        mvin_sram_addr_o[OC_DMA_IDX] = HALF_W'(mvin_meta_offset_bytes_i >> 4);
        mvin_col_num_o[OC_DMA_IDX] = HALF_W'(mvin_meta_byte_count_i - 32'd1);

        mvout_dram_addr_o[OC_DMA_IDX] = HALF_W'(mvout_dram_base_i);
        mvout_col_num_o[OC_DMA_IDX] = HALF_W'({16'd0, mvout_n_i} - 32'd1);
        mvout_row_num_o[OC_DMA_IDX] = HALF_W'({16'd0, mvout_m_i} - 32'd1);
        cfg_mvout_dram_stride_o[OC_DMA_IDX] =
            HALF_W'((mvout_output_stride_n_i == 16'd0) ? {16'd0, mvout_n_i} :
                                                         {16'd0, mvout_output_stride_n_i});
        cfg_mvout_sram_stride_o[OC_DMA_IDX] = QUARTER_W'((({16'd0, mvout_n_i} + 32'd3) >> 2) << 2);
        cfg_mvout_output_precision_o[OC_DMA_IDX] = 2'b11;
        cfg_mvout_output_type_o[OC_DMA_IDX] = (mvout_mode_i == NPU_MVOUT_RAW_I32) ? 2'b01 : 2'b11;
        cfg_mvout_output_scale_o[OC_DMA_IDX] = QUARTER_W'(mvout_scale_param_i[QUARTER_W-1:0]);
        cfg_mvout_mode_o[OC_DMA_IDX] = mvout_mode_i;
        cfg_mvout_per_channel_o[OC_DMA_IDX] =
            (mvout_mode_i == NPU_MVOUT_FP32_Q8_24) && mvout_per_channel_scale_i;
        cfg_mvout_scale_param_o[OC_DMA_IDX] = mvout_scale_param_i;

        if (cmd_fire && cmd_supported_dma) begin
            unique case (cmd_opcode_i)
                CMD_MVIN_A: begin
                    dma_mvin_req_en_o[A_DMA_IDX] = 1'b1;
                end
                CMD_MVIN_W: begin
                    dma_mvin_req_en_o[W_DMA_IDX] = 1'b1;
                end
                CMD_META_MVIN: begin
                    dma_mvin_req_en_o[OC_DMA_IDX] = 1'b1;
                end
                CMD_MVOUT: begin
                    dma_mvout_req_en_o[OC_DMA_IDX] = 1'b1;
                end
                CMD_GEMM: begin
                    sa_req_en_o = 1'b1;
                end
                default: begin
                end
            endcase
        end
    end

    assign mvin_a_done_o = loader_mvin_done_i[A_DMA_IDX];
    assign mvin_w_done_o = loader_mvin_done_i[W_DMA_IDX];
    assign mvin_meta_done_o = loader_mvin_done_i[OC_DMA_IDX];
    assign gemm_done_o = core_sa_done_i;
    assign mvout_done_o = dma_mvout_done_i[OC_DMA_IDX];
    assign attention_qk_done_o = 1'b0;

    assign api_error_valid_o = DISABLE_ERROR_CHECKS ? '0 : api_error_valid_q;
    assign api_error_code_o = DISABLE_ERROR_CHECKS ? '0 : api_error_code_q;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            inflight_q <= '0;
            core_error_seen_q <= 1'b0;
            core_error_sticky_q <= 1'b0;
            core_last_error_q <= '0;
            api_error_valid_q <= '0;
            api_error_code_q <= '0;
        end
        else begin
            api_error_valid_q <= '0;
            api_error_code_q <= '0;

            if (cmd_fire) begin
                unique case (cmd_opcode_i)
                    CMD_MVIN_A: begin
                        inflight_q[0] <= 1'b1;
                    end
                    CMD_MVIN_W: begin
                        inflight_q[1] <= 1'b1;
                    end
                    CMD_META_MVIN: begin
                        inflight_q[2] <= 1'b1;
                    end
                    CMD_GEMM: begin
                        inflight_q[3] <= 1'b1;
                    end
                    CMD_MVOUT: begin
                        inflight_q[4] <= 1'b1;
                    end
                    CMD_ATTENTION_QK: begin
                        inflight_q[5] <= 1'b1;
                    end
                    default: begin
                    end
                endcase
            end

            if (mvin_a_done_o) begin
                inflight_q[0] <= 1'b0;
            end
            if (mvin_w_done_o) begin
                inflight_q[1] <= 1'b0;
            end
            if (mvin_meta_done_o) begin
                inflight_q[2] <= 1'b0;
            end
            if (gemm_done_o) begin
                inflight_q[3] <= 1'b0;
            end
            if (mvout_done_o) begin
                inflight_q[4] <= 1'b0;
            end
            if (attention_qk_done_o) begin
                inflight_q[5] <= 1'b0;
            end

            if (!DISABLE_ERROR_CHECKS) begin
                if (core_error_sticky_q && !core_error_seen_q) begin
                    for (int api_idx = 0; api_idx < 6; api_idx++) begin
                        if (inflight_q[api_idx]) begin
                            api_error_valid_q[api_idx] <= 1'b1;
                            api_error_code_q[api_idx] <= core_last_error_q;
                            inflight_q[api_idx] <= 1'b0;
                        end
                    end
                end

                core_error_sticky_q <= core_error_sticky_i;
                core_last_error_q <= core_last_error_i;
                core_error_seen_q <= core_error_sticky_q;
            end
        end
    end

    wire unused_attention_qk_descriptor_inputs =
        ^attention_qk_output_dram_base_i ^ ^attention_qk_token_count_i ^
        ^attention_qk_gamma16_fix_i ^ attention_qk_mask_en_i ^
        ^attention_qk_q_block_start_i ^ ^attention_qk_q_block_count_m1_i;
endmodule

`default_nettype wire
