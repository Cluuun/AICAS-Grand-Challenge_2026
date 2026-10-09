`default_nettype none

import npu_spm_pkg::*;

// Phase-1 SPM/SA local datapath wrapper.
//
// This module is intentionally narrower than the legacy top-level scratchpad:
// it instantiates the Phase-1 A/W/C/O banks and local loader/output blocks, but
// it does not perform beat-level arbitration between unrelated clients. The
// scheduler/owner layer must still enforce phase exclusivity.
module npu_datapath #(
    parameter int BANK_DEPTH_WORDS   = 16384,
    parameter int A_BANK_DEPTH_WORDS = 12288,
    parameter int W_BANK_DEPTH_WORDS = 12288,
    parameter int O_BANK_DEPTH_WORDS = 16384,
    parameter int READ_LATENCY       = 4,
    parameter int META_WORD128_DEPTH = 2048,
    parameter int FIFO_DEPTH         = 8,
    parameter bit O_SPLIT_BANK_HALVES = 1'b0,
    parameter bit DISABLE_ERROR_CHECKS = 1'b0,
    localparam int O_PHYS_DEPTH_WORDS = O_SPLIT_BANK_HALVES ? (2 * O_BANK_DEPTH_WORDS) : O_BANK_DEPTH_WORDS,
    localparam int MAX_AW_DEPTH_WORDS = (A_BANK_DEPTH_WORDS > W_BANK_DEPTH_WORDS) ? A_BANK_DEPTH_WORDS : W_BANK_DEPTH_WORDS,
    localparam int MAX_BANK_DEPTH_WORDS = (MAX_AW_DEPTH_WORDS > O_PHYS_DEPTH_WORDS) ? MAX_AW_DEPTH_WORDS : O_PHYS_DEPTH_WORDS,
    localparam int A_URAM_DEPTH_SLICES = (A_BANK_DEPTH_WORDS + 4095) / 4096,
    localparam int W_URAM_DEPTH_SLICES = (W_BANK_DEPTH_WORDS + 4095) / 4096,
    localparam int O_URAM_DEPTH_SLICES = (O_BANK_DEPTH_WORDS + 4095) / 4096,
    localparam int O_PHYS_URAM_DEPTH_SLICES = (O_PHYS_DEPTH_WORDS + 4095) / 4096,
    localparam int ADDR_W            = (MAX_BANK_DEPTH_WORDS <= 1) ? 1 : $clog2(MAX_BANK_DEPTH_WORDS),
    localparam int META_ADDR_W       = (META_WORD128_DEPTH <= 1) ? 1 : $clog2(META_WORD128_DEPTH),
    localparam int META_BYTE_ADDR_W  = META_ADDR_W + 4
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,
    input  wire logic              clear_error_i,

    input  wire logic              a_cfg_start_i,
    output logic                   a_cfg_ready_o,
    input  wire logic [ADDR_W-1:0] a_cfg_base_word_addr_i,
    input  wire logic [15:0]       a_cfg_row_count_i,
    input  wire logic [15:0]       a_cfg_row_bytes_i,
    input  wire logic [15:0]       a_cfg_row_stride_words_i,
    input  wire logic              a_cfg_selected_bank_i,
    input  wire logic              a_cfg_u8_minus_128_en_i,
    input  wire logic              a_dma_valid_i,
    output logic                   a_dma_ready_o,
    input  wire logic [127:0]      a_dma_data_i,
    input  wire logic [15:0]       a_dma_keep_i,
    input  wire logic              a_dma_last_i,
    input  wire logic              a_rd_valid_i,
    input  wire logic [ADDR_W-1:0] a_rd_word_addr_i,
    output logic                   a_rsp_valid_o,
    output logic [255:0]           a_rsp_data_o,
    output logic                   a_busy_o,
    output logic                   a_done_o,
    output logic                   a_error_sticky_o,
    output logic [7:0]             a_error_code_o,

    input  wire logic              w_cfg_start_i,
    output logic                   w_cfg_ready_o,
    input  wire logic [ADDR_W-1:0] w_cfg_base_word_addr_i,
    input  wire logic [15:0]       w_cfg_k_loaded_i,
    input  wire logic [15:0]       w_cfg_n_loaded_i,
    input  wire logic              w_cfg_selected_bank_i,
    input  wire logic              active_w_bank_i,
    input  wire logic              w_dma_valid_i,
    output logic                   w_dma_ready_o,
    input  wire logic [127:0]      w_dma_data_i,
    input  wire logic [15:0]       w_dma_keep_i,
    input  wire logic              w_dma_last_i,
    input  wire logic              w_rd_valid_i,
    input  wire logic [ADDR_W-1:0] w_rd_word_addr_i,
    output logic                   w_rsp_valid_o,
    output logic [255:0]           w_rsp_data_o,
    output logic                   w_busy_o,
    output logic                   w_done_o,
    output logic                   w_error_sticky_o,
    output logic [7:0]             w_error_code_o,

    input  wire logic              gemm_cfg_start_i,
    output logic                   gemm_cfg_ready_o,
    input  wire logic [ADDR_W-1:0] gemm_a_base_word_addr_i,
    input  wire logic [ADDR_W-1:0] gemm_a_k_word_offset_i,
    input  wire logic [ADDR_W-1:0] gemm_a_row_stride_words_i,
    input  wire logic [ADDR_W-1:0] gemm_w_base_word_addr_i,
    input  wire logic              gemm_a_bank_i,
    input  wire logic              gemm_o_bank_i,
    input  wire npu_mode_e         gemm_mode_i,
    input  wire logic [15:0]       gemm_m_count_i,
    input  wire logic [15:0]       gemm_n_count_i,
    input  wire logic [12:0]       gemm_k_count_i,
    output logic                   gemm_busy_o,
    output logic                   gemm_done_o,
    output logic                   gemm_error_sticky_o,
    output npu_error_e             gemm_last_error_o,

    input  wire logic              meta_mvin_valid_i,
    output logic                   meta_mvin_ready_o,
    input  wire logic [META_ADDR_W-1:0] meta_mvin_word128_addr_i,
    input  wire logic [127:0]      meta_mvin_data_i,
    input  wire logic [15:0]       meta_mvin_keep_i,
    input  wire logic              meta_read256_valid_i,
    input  wire logic [META_ADDR_W-1:0] meta_read256_base_word128_i,
    input  wire logic [META_ADDR_W-1:0] meta_read256_word_idx_i,
    output logic                   meta_read256_rsp_valid_o,
    output logic [255:0]           meta_read256_rsp_data_o,
    output logic                   meta_conflict_sticky_o,
    output npu_error_e             meta_last_error_o,

    input  wire logic              co_wr_start_i,
    input  wire logic [ADDR_W-1:0] co_wr_base_word_addr_i,
    input  wire logic              co_accumulate_en_i,
    input  wire logic              co_add_bias_en_i,
    input  wire logic              co_in_valid_i,
    output logic                   co_in_ready_o,
    input  wire logic [255:0]      co_partial_data_i,
    input  wire logic [7:0]        co_lane_mask_i,
    input  wire logic [META_BYTE_ADDR_W-1:0] co_bias_addr_i,
    output logic                   co_acc_error_sticky_o,
    output npu_error_e             co_acc_last_error_o,

    input  wire logic              mvout_req_valid_i,
    output logic                   mvout_req_ready_o,
    input  wire logic [ADDR_W-1:0] mvout_req_word_addr_i,
    input  wire logic              mvout_o_bank_i,
    input  wire logic [15:0]       mvout_m_count_i,
    input  wire logic [15:0]       mvout_n_count_i,
    input  wire logic [63:0]       mvout_dram_base_addr_i,
    input  wire logic [31:0]       mvout_dram_row_stride_bytes_i,
    input  wire npu_mvout_mode_e   mvout_mode_i,
    input  wire logic              mvout_per_channel_scale_i,
    input  wire logic [31:0]       mvout_tensor_scale_q8_24_i,
    input  wire logic [META_BYTE_ADDR_W-1:0] mvout_scale_addr_i,

    input  wire logic              attention_qk_req_valid_i,
    output logic                   attention_qk_req_ready_o,
    input  wire logic [15:0]       attention_qk_token_count_i,
    input  wire logic [31:0]       attention_qk_gamma16_fix_i,
    input  wire logic              attention_qk_mask_en_i,
    input  wire logic              attention_qk_q_bank_i,
    input  wire logic              attention_qk_k_bank_i,
    input  wire logic              attention_qk_o_bank_i,
    input  wire logic [ADDR_W-1:0] attention_qk_output_base_word_addr_i,
    input  wire logic [31:0]       attention_qk_output_dram_base_addr_i,
    input  wire logic [4:0]        attention_qk_q_block_start_i,
    input  wire logic [4:0]        attention_qk_q_block_count_m1_i,

    output logic                   mvout_dma_valid_o,
    input  wire logic              mvout_dma_ready_i,
    output logic [127:0]           mvout_dma_data_o,
    output logic [15:0]            mvout_dma_keep_o,
    output logic [63:0]            mvout_dma_addr_o,
    output logic                   mvout_dma_last_o,
    output logic                   mvout_fifo_almost_full_o,
    output logic                   mvout_done_o,
    output logic                   mvout_error_sticky_o,
    output npu_error_e             mvout_last_error_o,
    output logic                   attention_qk_done_o,
    output logic                   attention_qk_error_sticky_o,
    output npu_error_e             attention_qk_last_error_o,

    output logic                   a_bank_conflict_o,
    output logic                   w0_bank_conflict_o,
    output logic                   w1_bank_conflict_o,
    output logic                   co_bank_conflict_o
);

    logic a_wr_valid;
    logic a_wr_ready;
    logic [ADDR_W-1:0] a_wr_word_addr;
    logic [255:0] a_wr_data;
    logic [31:0] a_wr_byte_en;
    logic a_load_bank_q;
    logic gemm_a_bank_q;
    logic a0_wr_ready;
    logic a1_wr_ready;
    logic a0_bank_d_req_valid;
    logic a1_bank_d_req_valid;
    logic a0_bank_rsp_valid;
    logic a1_bank_rsp_valid;
    logic [255:0] a0_bank_rsp_data;
    logic [255:0] a1_bank_rsp_data;
    logic a0_bank_conflict;
    logic a1_bank_conflict;

    logic w0_wr_valid;
    logic w0_wr_ready;
    logic [ADDR_W-1:0] w0_wr_word_addr;
    logic [255:0] w0_wr_data;
    logic [31:0] w0_wr_byte_en;
    logic w1_wr_valid;
    logic w1_wr_ready;
    logic [ADDR_W-1:0] w1_wr_word_addr;
    logic [255:0] w1_wr_data;
    logic [31:0] w1_wr_byte_en;
    logic w0_rsp_valid;
    logic [255:0] w0_rsp_data;
    logic w1_rsp_valid;
    logic [255:0] w1_rsp_data;
    logic gemm_a_rd_req_valid;
    logic gemm_a_rd_req_ready;
    logic [ADDR_W-1:0] gemm_a_rd_req_addr;
    logic gemm_a_rd_rsp_valid;
    logic [255:0] gemm_a_rd_rsp_data;
    logic gemm_w_rd_req_valid;
    logic gemm_w_rd_req_ready;
    logic [ADDR_W-1:0] gemm_w_rd_req_addr;
    logic gemm_w_rd_rsp_valid;
    logic [255:0] gemm_w_rd_rsp_data;
    logic gemm_co_valid;
    logic gemm_co_ready;
    logic [255:0] gemm_co_data;
    logic [7:0] gemm_co_lane_mask;
    logic gemm_pv_row_sum_valid;
    logic [NPU_SA_ROWS-1:0][31:0] gemm_pv_row_sum;
    logic [NPU_SA_ROWS-1:0][31:0] gemm_pv_row_sum_q;
    logic gemm_core_cfg_start;
    logic gemm_core_cfg_ready;
    logic gemm_core_snapshot_ready;
    logic gemm_core_snapshot_ready_q;
    logic gemm_core_busy;
    logic gemm_core_done;
    logic gemm_core_done_for_macro;
    logic gemm_core_error_sticky;
    npu_error_e gemm_core_last_error;
    logic gemm_cfg_attention_qk_mode;
    logic gemm_attention_qk_start_fire;
    logic gemm_core_qk_owner;
    logic qk_core_cfg_start;
    logic qk_core_cfg_ready;
    logic [ADDR_W-1:0] qk_core_a_base_word_addr;
    logic [ADDR_W-1:0] qk_core_a_k_word_offset;
    logic [ADDR_W-1:0] qk_core_a_row_stride_words;
    logic [ADDR_W-1:0] qk_core_w_base_word_addr;
    logic [5:0] qk_core_m_count;
    logic [5:0] qk_core_n_count;
    logic [12:0] qk_core_k_count;
    npu_mode_e qk_core_mode;
    logic qk_core_snapshot_ready;
    logic qk_core_co_ready;
    logic shared_core_cfg_start;
    logic [ADDR_W-1:0] shared_core_a_base;
    logic [ADDR_W-1:0] shared_core_a_k_word_offset;
    logic [ADDR_W-1:0] shared_core_a_row_stride_words;
    logic [ADDR_W-1:0] shared_core_w_base;
    logic [5:0] shared_core_m_count;
    logic [5:0] shared_core_n_count;
    logic [12:0] shared_core_k_count;
    npu_mode_e shared_core_mode;
    logic shared_core_snapshot_ready;
    logic shared_core_a_bank;
    logic shared_core_w_bank;
    logic shared_launch_req;
    logic shared_launch_req_qk;
    logic shared_launch_can_accept;
    logic shared_launch_fire;
    logic shared_launch_valid_q;
    logic shared_launch_owner_q;
    logic [ADDR_W-1:0] shared_launch_a_base_q;
    logic [ADDR_W-1:0] shared_launch_a_k_word_offset_q;
    logic [ADDR_W-1:0] shared_launch_a_row_stride_words_q;
    logic [ADDR_W-1:0] shared_launch_w_base_q;
    logic [5:0] shared_launch_m_count_q;
    logic [5:0] shared_launch_n_count_q;
    logic [12:0] shared_launch_k_count_q;
    npu_mode_e shared_launch_mode_q;
    logic shared_launch_a_bank_q;
    logic shared_launch_w_bank_q;
    logic shared_core_active_qk_q;
    logic shared_core_active_a_bank_q;
    logic shared_core_active_w_bank_q;
    logic [4:0] gemm_attention_qk_count_m1;
    logic [11:0] gemm_attention_qk_m_blocks_m1;
    logic [1:0] gemm_core_done_pipe_q;
    logic gemm_macro_busy_q;
    logic gemm_macro_done_q;
    logic gemm_all_tiles_done_q;
    logic gemm_core_done_pending_q;
    logic [ADDR_W-1:0] gemm_a_base_word_addr_q;
    logic [ADDR_W-1:0] gemm_a_row_stride_words_q;
    logic [ADDR_W-1:0] gemm_w_base_word_addr_q;
    logic [ADDR_W-1:0] gemm_co_base_word_addr_q;
    logic [ADDR_W-1:0] gemm_co_row_stride_words_q;
    logic [15:0] gemm_m_total_q;
    logic [15:0] gemm_n_total_q;
    logic [12:0] gemm_k_count_q;
    logic [15:0] gemm_m_base_q;
    logic [15:0] gemm_n_group_q;
    logic gemm_add_bias_q;
    logic gemm_accumulate_q;
    npu_mode_e gemm_mode_q;
    logic gemm_o_bank_q;
    logic co_write_bank_q;
    logic mvout_o_bank_q;
    logic [7:0] gemm_co_word_idx_q;
    logic [7:0] gemm_co_accept_idx_q;
    logic gemm_drain_active_q;
    logic gemm_pv_drain_prepare_q;
    logic [15:0] gemm_drain_m_base_q;
    logic [15:0] gemm_drain_n_group_q;
    logic [ADDR_W-1:0] gemm_drain_word_addr_q;
    logic [ADDR_W-1:0] gemm_core_a_base;
    logic [ADDR_W-1:0] gemm_core_w_base;
    logic [5:0] gemm_core_m_count;
    logic [5:0] gemm_core_n_count;
    logic [ADDR_W-1:0] co_bank_wr_addr;
    logic [ADDR_W-1:0] co_src_word_addr;
    logic [ADDR_W-1:0] co_acc_out_word_addr_q;
    logic [READ_LATENCY-1:0] a_rd_gemm_pipe_q;
    logic [READ_LATENCY-1:0] w_rd_gemm_pipe_q;
    logic a_bank_d_req_valid;
    logic [ADDR_W-1:0] a_bank_d_req_addr;
    logic a_bank_rsp_valid;
    logic [255:0] a_bank_rsp_data;
    logic w0_bank_d_req_valid;
    logic w1_bank_d_req_valid;
    logic [ADDR_W-1:0] w_bank_d_req_addr;
    logic w_rsp_for_gemm;
    logic co_src_valid;
    logic gemm_drain_to_co_valid;
    logic co_src_ready;
    logic co_src_accumulate_en;
    logic co_src_add_bias_en;
    logic co_src_needs_old_acc;
    logic co_src_needs_bias;
    logic [255:0] co_src_partial_data;
    logic [7:0] co_src_lane_mask;
    logic co_direct_fire;
    logic co_acc_stage_load_valid;
    logic co_acc_stage_load_ready;
    logic co_acc_stage_load_accumulate_en;
    logic co_acc_stage_load_add_bias_en;
    logic [255:0] co_acc_stage_load_partial_data;
    logic [7:0] co_acc_stage_load_lane_mask;
    logic [255:0] co_acc_stage_load_old_acc_data;
    logic [255:0] co_acc_stage_load_bias_data;
    logic [ADDR_W-1:0] co_acc_stage_load_word_addr;
    logic co_acc_stage_valid_q;
    logic co_acc_stage_accumulate_en_q;
    logic co_acc_stage_add_bias_en_q;
    logic [255:0] co_acc_stage_partial_data_q;
    logic [7:0] co_acc_stage_lane_mask_q;
    logic [255:0] co_acc_stage_old_acc_data_q;
    logic [255:0] co_acc_stage_bias_data_q;
    logic [ADDR_W-1:0] co_acc_stage_word_addr_q;
    logic co_acc_in_valid;
    logic co_acc_in_ready;
    logic co_acc_accumulate_en;
    logic co_acc_add_bias_en;
    logic [255:0] co_acc_partial_data;
    logic [7:0] co_acc_lane_mask;
    logic [255:0] co_acc_old_acc_data;
    logic [255:0] co_acc_bias_data;
    logic bias_pending_q;
    logic bias_rsp_valid_q;
    logic bias_accumulate_q;
    logic bias_add_bias_q;
    logic [255:0] bias_partial_data_q;
    logic [7:0] bias_lane_mask_q;
    logic [255:0] bias_data_q;
    logic [ADDR_W-1:0] bias_word_addr_q;
    logic [META_ADDR_W-1:0] bias_base_word128;
    logic [META_ADDR_W-1:0] bias_word_idx;
    logic old_acc_pending_q;
    logic old_acc_rsp_valid_q;
    logic [255:0] old_acc_partial_data_q;
    logic [7:0] old_acc_lane_mask_q;
    logic [255:0] old_acc_data_q;
    logic [ADDR_W-1:0] old_acc_word_addr_q;

    logic co_acc_out_valid;
    logic co_acc_out_ready;
    logic [255:0] co_acc_out_data;
    logic [31:0] co_acc_out_byte_en;
    logic [ADDR_W-1:0] co_wr_word_addr_q;
    logic co_rsp_valid;
    logic [255:0] co_rsp_data;
    logic co0_acc_out_ready;
    logic co1_acc_out_ready;
    logic co0_rsp_valid;
    logic co1_rsp_valid;
    logic [255:0] co0_rsp_data;
    logic [255:0] co1_rsp_data;
    logic co0_bank_d_req_valid;
    logic co1_bank_d_req_valid;
    logic [ADDR_W-1:0] co0_bank_d_req_addr;
    logic [ADDR_W-1:0] co1_bank_d_req_addr;
    logic co0_bank_conflict;
    logic co1_bank_conflict;
    logic [ADDR_W-1:0] co_split_wr_addr;
    logic [ADDR_W-1:0] co_split_rd_addr;
    logic              co_read_bank;
    logic attention_qk_o_wr_req_valid;
    logic attention_qk_o_wr_req_ready;
    logic [ADDR_W-1:0] attention_qk_o_wr_req_addr;
    logic [255:0] attention_qk_o_wr_req_data;
    logic [31:0] attention_qk_o_wr_req_byte_en;
    logic attention_qk_meta_wr_valid;
    logic attention_qk_meta_wr_ready;
    logic [META_ADDR_W-1:0] attention_qk_meta_wr_addr;
    logic [127:0] attention_qk_meta_wr_data;
    logic [15:0] attention_qk_meta_wr_keep;
    logic attention_qk_m_global_wr_valid;
    logic [8:0] attention_qk_m_global_wr_row;
    logic signed [15:0] attention_qk_m_global_wr_data;
    logic attention_qk_busy;
    logic signed [15:0] attention_qk_m_global_q [0:511];
    logic attention_qk_chunk_valid_q;
    logic attention_qk_chunk_o_bank_q;
    logic [15:0] attention_qk_chunk_token_count_q;
    logic [4:0] attention_qk_chunk_q_block_start_q;
    logic [5:0] attention_qk_chunk_q_block_count_q;
    logic attention_qk_chunk_mask_en_q;
    localparam int CO_HALF_DEPTH_WORDS = O_BANK_DEPTH_WORDS;
    localparam int MVOUT_PREFETCH_DEPTH = (FIFO_DEPTH < (READ_LATENCY + 2)) ? (READ_LATENCY + 2) : FIFO_DEPTH;
    localparam int MVOUT_Q_PTR_W = (MVOUT_PREFETCH_DEPTH <= 1) ? 1 : $clog2(MVOUT_PREFETCH_DEPTH);
    localparam int MVOUT_Q_CNT_W = $clog2(MVOUT_PREFETCH_DEPTH + 1);
    localparam logic [MVOUT_Q_CNT_W-1:0] MVOUT_PREFETCH_DEPTH_COUNT = MVOUT_PREFETCH_DEPTH;

    logic [255:0] mvout_resp_data_q [0:MVOUT_PREFETCH_DEPTH-1];
    logic [255:0] mvout_resp_scale_q [0:MVOUT_PREFETCH_DEPTH-1];
    logic [MVOUT_Q_PTR_W-1:0] mvout_resp_wr_ptr_q;
    logic [MVOUT_Q_PTR_W-1:0] mvout_resp_rd_ptr_q;
    logic [MVOUT_Q_CNT_W-1:0] mvout_resp_count_q;
    logic [MVOUT_Q_CNT_W-1:0] mvout_outstanding_count_q;
    logic [255:0] mvout_scale_q [0:MVOUT_PREFETCH_DEPTH-1];
    logic [MVOUT_Q_PTR_W-1:0] mvout_scale_wr_ptr_q;
    logic [MVOUT_Q_PTR_W-1:0] mvout_scale_rd_ptr_q;
    logic [MVOUT_Q_CNT_W-1:0] mvout_scale_count_q;
    logic [MVOUT_Q_CNT_W-1:0] mvout_scale_outstanding_count_q;
    logic [MVOUT_Q_CNT_W:0] mvout_prefetch_pending_count;
    logic [MVOUT_Q_CNT_W:0] mvout_scale_pending_count;
    logic scale_pending_q;
    logic mvout_issue_done_q;
    logic mvout_read_path_active;
    logic mvout_blocks_co_write;
    logic mvout_prefetch_candidate;
    logic mvout_prefetch_fire;
    logic [ADDR_W-1:0] mvout_prefetch_addr;
    logic [ADDR_W-1:0] mvout_prefetch_addr_q;
    logic mvout_resp_enqueue;
    logic mvout_resp_dequeue;
    logic mvout_scale_rsp_push;
    logic mvout_scale_pop_for_rsp;
    logic [255:0] mvout_scale_for_rsp;
    logic [META_ADDR_W-1:0] mvout_scale_base_word128;
    logic [META_ADDR_W-1:0] mvout_scale_word_idx;
    logic mvout_co_ready;
    logic mvout_convert_co_ready;
    logic mvout_co_valid;
    logic [127:0] mvout_co_data;
    logic [127:0] mvout_co_scale_data;
    logic mvout_convert_dma_valid;
    logic mvout_convert_dma_ready;
    logic [127:0] mvout_convert_dma_data;
    logic mvout_convert_input_fire;
    logic mvout_req_fire;
    logic mvout_req_attn_legal;
    logic mvout_convert_error_sticky;
    npu_error_e mvout_convert_last_error;
    logic mvout_local_error_sticky_q;
    npu_error_e mvout_local_last_error_q;
    logic mvout_scale_req_fire;
    logic mvout_active_q;
    logic [ADDR_W-1:0] mvout_base_word_addr_q;
    logic [15:0] mvout_m_count_q;
    logic [15:0] mvout_n_count_q;
    logic [15:0] mvout_row_q;
    logic [15:0] mvout_word_col_q;
    logic [ADDR_W-1:0] mvout_co_row_stride_words_q;
    logic [31:0] mvout_row_bytes_q;
    logic [15:0] mvout_words_per_row_q;
    logic [15:0] mvout_in_row_q;
    logic [15:0] mvout_in_word_col_q;
    logic mvout_in_half_q;
    logic [15:0] mvout_out_row_q;
    logic [15:0] mvout_out_word_col_q;
    logic mvout_out_half_q;
    logic [63:0] mvout_dram_base_addr_q;
    logic [31:0] mvout_dram_row_stride_bytes_q;
    npu_mvout_mode_e mvout_mode_q;
    logic mvout_per_channel_scale_q;
    logic [31:0] mvout_tensor_scale_q8_24_q;
    logic [META_BYTE_ADDR_W-1:0] mvout_scale_addr_q;
    logic [31:0] mvout_in_half_col_byte;
    logic [31:0] mvout_out_half_col_byte;
    logic [31:0] mvout_output_remaining_bytes;
    logic [15:0] mvout_current_keep;
    logic mvout_input_half_valid;
    logic mvout_input_word_done;
    logic mvout_output_half_valid;
    logic mvout_output_word_done;
    logic mvout_output_last_valid_half;
    logic mvout_convert_pop;
    logic mvout_attn_mode;
    logic signed [15:0] mvout_attn_m_block;
    logic signed [15:0] mvout_attn_m_global;
    logic signed [32:0] mvout_attn_correction;
    logic [8:0] mvout_attn_row_index;
    logic [15:0] mvout_attn_global_row;
    logic [15:0] mvout_attn_global_col_base;
    logic mvout_attn_meta_req_fire;
    logic [14:0] mvout_attn_meta_half_index;
    logic [14:0] mvout_attn_output_meta_half_index;
    logic [3:0] mvout_attn_meta_half_lane;
    logic [META_ADDR_W-1:0] mvout_attn_meta_word_idx;
    logic mvout_attn_s0_ready;
    logic mvout_attn_s0_fire;
    logic mvout_attn_s0_valid_q;
    logic [255:0] mvout_attn_s0_d_word_q;
    logic signed [32:0] mvout_attn_s0_correction_q;
    logic [15:0] mvout_attn_s0_global_row_q;
    logic [15:0] mvout_attn_s0_col_base_q;
    logic mvout_attn_s0_half_q;
    logic [15:0] mvout_attn_s0_keep_q;
    logic [63:0] mvout_attn_s0_addr_q;
    logic mvout_attn_s0_last_q;
    logic mvout_attn_s1_ready;
    logic mvout_attn_s1_to_dma_fire;
    logic mvout_attn_s1_valid_q;
    logic [127:0] mvout_attn_s1_data_q;
    logic [15:0] mvout_attn_s1_keep_q;
    logic [63:0] mvout_attn_s1_addr_q;
    logic mvout_attn_s1_last_q;
    logic [127:0] mvout_attn_s1_data_comb;
    logic mvout_internal_last_pop;
    logic mvout_dma_load_fire;
    logic mvout_dma_reg_ready;
    logic mvout_dma_valid_q;
    logic [127:0] mvout_dma_data_q;
    logic [15:0] mvout_dma_keep_q;
    logic [63:0] mvout_dma_addr_q;
    logic mvout_dma_last_q;
    logic regular_mvout_dma_ready;
    logic mvout_last_read_word;
    logic mvout_last_row_word;
    logic old_acc_read_fire;
    logic co_bank_d_idle_for_old_acc;
    logic co_bank_d_idle_for_write;
    logic co_bank_d_req_valid;
    logic [ADDR_W-1:0] co_bank_d_req_addr;
    logic co_acc_pipeline_idle;
    logic gemm_microtile_done_ready;
    logic gemm_compute_done_available;
    logic gemm_drain_valid;
    logic gemm_drain_slot_available;
    logic gemm_drain_input_last_fire;
    logic gemm_drain_write_last_fire;
    logic gemm_drain_bias_prefetch_fire;
    logic gemm_drain_input_done_q;
    logic gemm_drain_write_done_q;
    logic gemm_drain_input_done_next;
    logic gemm_drain_write_done_next;
    logic gemm_drain_done_next;
    logic gemm_pv_mode;
    logic gemm_pv_drain_mode;
    logic gemm_non_pv_drain_to_co_valid;
    logic gemm_pv_norm_to_co_valid;
    logic gemm_pv_row_last_fire;
    logic [4:0] gemm_pv_drain_row;
    logic [1:0] co_src_word_group_idx;
    logic co_external_allowed;

    logic pv_norm_in_valid;
    logic pv_norm_in_ready;
    logic [31:0] pv_norm_denominator;
    logic pv_norm_out_valid;
    logic pv_norm_out_ready;
    logic [255:0] pv_norm_out_data;
    logic [7:0] pv_norm_out_lane_mask;
    logic [ADDR_W-1:0] pv_norm_out_word_addr;
    logic [1:0] pv_norm_out_word_group_idx;
    logic pv_norm_busy;

    typedef enum logic [2:0] {
        META_READ_NONE,
        META_READ_EXTERNAL,
        META_READ_BIAS,
        META_READ_SCALE,
        META_READ_BIAS_CACHE,
        META_READ_ATTENTION_QK_MBLOCK
    } meta_read_owner_e;

    meta_read_owner_e meta_read_owner_q;
    meta_read_owner_e meta_read_owner_d;
    logic metadata_read_valid;
    logic [META_ADDR_W-1:0] metadata_read_base_word128;
    logic [META_ADDR_W-1:0] metadata_read_word_idx;
    logic metadata_rsp_valid;
    logic [255:0] metadata_rsp_data;
    logic metadata_mvin_valid;
    logic metadata_mvin_ready;
    logic [META_ADDR_W-1:0] metadata_mvin_addr;
    logic [127:0] metadata_mvin_data;
    logic [15:0] metadata_mvin_keep;
    logic bias_read_fire;
    logic bias_cache_prefetch_fire;
    logic bias_cache_usable;
    logic bias_cache_ready;
    logic bias_cache_hit;
    logic bias_cache_fire;
    logic bias_cache_prefetch_active_q;
    logic [1:0] bias_cache_issue_idx_q;
    logic [1:0] bias_cache_rsp_idx_q;
    logic [15:0] bias_cache_n_group_q;
    logic [15:0] bias_cache_next_n_group;
    logic [3:0] bias_cache_valid_q;
    logic [255:0] bias_cache_q [0:3];

    assign w_rsp_for_gemm = w_rd_gemm_pipe_q[READ_LATENCY-1];
    assign w_rsp_valid_o = (w0_rsp_valid || w1_rsp_valid) &&
                           !w_rsp_for_gemm;
    assign w_rsp_data_o  = w1_rsp_valid ? w1_rsp_data : w0_rsp_data;

    assign a_bank_d_req_valid = gemm_a_rd_req_valid || a_rd_valid_i;
    assign a_bank_d_req_addr  = gemm_a_rd_req_valid ? gemm_a_rd_req_addr : a_rd_word_addr_i;
    assign gemm_a_rd_req_ready = 1'b1;
    assign a0_bank_d_req_valid = (gemm_a_rd_req_valid && !shared_core_active_a_bank_q) ||
                                 (a_rd_valid_i && !gemm_a_bank_q);
    assign a1_bank_d_req_valid = (gemm_a_rd_req_valid && shared_core_active_a_bank_q) ||
                                 (a_rd_valid_i && gemm_a_bank_q);
    assign a_bank_rsp_valid    = a0_bank_rsp_valid || a1_bank_rsp_valid;
    assign a_bank_rsp_data     = a1_bank_rsp_valid ? a1_bank_rsp_data : a0_bank_rsp_data;
    assign gemm_a_rd_rsp_valid = a_bank_rsp_valid && a_rd_gemm_pipe_q[READ_LATENCY-1];
    assign gemm_a_rd_rsp_data  = a_bank_rsp_data;
    assign a_rsp_valid_o       = a_bank_rsp_valid &&
                                 !a_rd_gemm_pipe_q[READ_LATENCY-1];
    assign a_rsp_data_o        = a_bank_rsp_data;

    assign w0_bank_d_req_valid = (gemm_w_rd_req_valid && !shared_core_active_w_bank_q) ||
                                 (w_rd_valid_i && !active_w_bank_i);
    assign w1_bank_d_req_valid = (gemm_w_rd_req_valid && shared_core_active_w_bank_q) ||
                                 (w_rd_valid_i && active_w_bank_i);
    assign w_bank_d_req_addr   = gemm_w_rd_req_valid ? gemm_w_rd_req_addr : w_rd_word_addr_i;
    assign gemm_w_rd_req_ready = 1'b1;
    assign gemm_w_rd_rsp_valid = (w0_rsp_valid || w1_rsp_valid) && w_rsp_for_gemm;
    assign gemm_w_rd_rsp_data  = w1_rsp_valid ? w1_rsp_data : w0_rsp_data;

    assign gemm_pv_mode         = (gemm_mode_q == NPU_MODE_PV_LOG8_U16I8);
    assign gemm_pv_drain_mode   = gemm_drain_active_q && gemm_pv_mode;
    assign gemm_pv_drain_row    = gemm_co_accept_idx_q[6:2];
    assign pv_norm_denominator  = gemm_pv_row_sum_q[gemm_pv_drain_row];
    assign gemm_drain_valid     = !gemm_core_qk_owner && gemm_drain_active_q && gemm_co_valid;
    assign pv_norm_in_valid     = gemm_pv_drain_mode && gemm_drain_valid;
    assign gemm_non_pv_drain_to_co_valid = gemm_drain_valid && !gemm_pv_drain_mode;
    assign gemm_pv_norm_to_co_valid = pv_norm_out_valid;
    assign gemm_drain_to_co_valid = gemm_non_pv_drain_to_co_valid || gemm_pv_norm_to_co_valid;
    assign co_external_allowed  = !gemm_drain_active_q &&
                                  !gemm_pv_drain_prepare_q &&
                                  !pv_norm_busy;
    assign co_src_valid         = gemm_drain_to_co_valid ||
                                  (co_external_allowed && co_in_valid_i);
    assign co_src_accumulate_en = gemm_drain_to_co_valid ? gemm_accumulate_q : co_accumulate_en_i;
    assign co_src_add_bias_en   = gemm_drain_to_co_valid ? gemm_add_bias_q : co_add_bias_en_i;
    assign co_src_needs_old_acc = co_src_accumulate_en && !co_src_add_bias_en;
    assign co_src_needs_bias    = co_src_add_bias_en && !co_src_accumulate_en;
    assign co_src_partial_data  = gemm_pv_norm_to_co_valid ? pv_norm_out_data :
                                  gemm_non_pv_drain_to_co_valid ? gemm_co_data :
                                  co_partial_data_i;
    assign co_src_lane_mask     = gemm_pv_norm_to_co_valid ? pv_norm_out_lane_mask :
                                  gemm_non_pv_drain_to_co_valid ? gemm_co_lane_mask :
                                  co_lane_mask_i;
    assign co_src_word_group_idx = gemm_pv_norm_to_co_valid ? pv_norm_out_word_group_idx :
                                   gemm_non_pv_drain_to_co_valid ? gemm_co_accept_idx_q[1:0] :
                                   co_wr_word_addr_q[1:0];
    assign mvout_read_path_active = mvout_active_q ||
                                    (mvout_outstanding_count_q != '0) ||
                                    (mvout_resp_count_q != '0) ||
                                    (mvout_scale_outstanding_count_q != '0) ||
                                    (mvout_scale_count_q != '0);
    assign mvout_blocks_co_write = mvout_read_path_active &&
                                   (mvout_o_bank_q == co_write_bank_q);
    assign co_bank_d_idle_for_write = !mvout_blocks_co_write;
    assign co_bank_d_idle_for_old_acc = co_bank_d_idle_for_write;
    assign bias_cache_usable    = gemm_drain_to_co_valid && gemm_add_bias_q && !gemm_accumulate_q;
    assign bias_cache_ready     = (&bias_cache_valid_q) && !bias_cache_prefetch_active_q &&
                                  (bias_cache_n_group_q == gemm_drain_n_group_q);
    assign bias_cache_hit       = bias_cache_usable && bias_cache_ready;
    assign co_src_ready         = (!bias_pending_q && !bias_rsp_valid_q &&
                                   !old_acc_pending_q && !old_acc_rsp_valid_q &&
                                   co_bank_d_idle_for_write) &&
                                  (co_src_needs_old_acc
                                      ? (!co_acc_out_valid && co_bank_d_idle_for_old_acc)
                                      : (co_src_needs_bias
                                             ? (bias_cache_hit ? co_acc_stage_load_ready : !co_acc_out_valid)
                                             : co_acc_stage_load_ready));
    assign gemm_co_ready        = gemm_core_qk_owner
        ? qk_core_co_ready
        : (gemm_pv_drain_mode
              ? pv_norm_in_ready
              : (gemm_drain_active_q && co_src_ready));
    assign pv_norm_out_ready    = co_src_ready;
    assign co_in_ready_o        = co_external_allowed && co_src_ready;
    assign bias_cache_fire      = co_src_valid && co_src_ready &&
                                  co_src_needs_bias && bias_cache_hit;
    assign bias_read_fire       = co_src_valid && co_src_ready &&
                                  co_src_needs_bias && !bias_cache_hit;
    assign old_acc_read_fire    = co_src_valid && co_src_ready && co_src_needs_old_acc;
    assign co_direct_fire       = co_src_valid && co_src_ready &&
                                  !co_src_needs_bias && !co_src_needs_old_acc;

    assign co_acc_stage_load_valid = bias_cache_fire || bias_rsp_valid_q ||
                                     old_acc_rsp_valid_q || co_direct_fire;
    assign co_acc_stage_load_ready = !co_acc_stage_valid_q || co_acc_in_ready;
    assign co_acc_stage_load_accumulate_en = bias_cache_fire ? co_src_accumulate_en :
                                             bias_rsp_valid_q ? bias_accumulate_q :
                                             old_acc_rsp_valid_q ? 1'b1 :
                                             co_src_accumulate_en;
    assign co_acc_stage_load_add_bias_en   = bias_cache_fire ? co_src_add_bias_en :
                                             bias_rsp_valid_q ? bias_add_bias_q :
                                             old_acc_rsp_valid_q ? 1'b0 :
                                             co_src_add_bias_en;
    assign co_acc_stage_load_partial_data  = bias_cache_fire ? co_src_partial_data :
                                             bias_rsp_valid_q ? bias_partial_data_q :
                                             old_acc_rsp_valid_q ? old_acc_partial_data_q :
                                             co_src_partial_data;
    assign co_acc_stage_load_lane_mask     = bias_cache_fire ? co_src_lane_mask :
                                             bias_rsp_valid_q ? bias_lane_mask_q :
                                             old_acc_rsp_valid_q ? old_acc_lane_mask_q :
                                             co_src_lane_mask;
    assign co_acc_stage_load_old_acc_data  = old_acc_rsp_valid_q ? old_acc_data_q : '0;
    assign co_acc_stage_load_bias_data     = bias_cache_fire ? bias_cache_q[co_src_word_group_idx] :
                                             bias_rsp_valid_q ? bias_data_q : '0;
    assign gemm_cfg_attention_qk_mode = (gemm_mode_i == NPU_MODE_ATTENTION_QK);
    assign gemm_attention_qk_start_fire =
        gemm_cfg_start_i && gemm_cfg_ready_o && gemm_cfg_attention_qk_mode;
    assign gemm_attention_qk_m_blocks_m1 = {1'b0, gemm_m_count_i[15:5]} - 12'd1;
    assign gemm_attention_qk_count_m1 =
        (gemm_m_count_i[15:5] == 11'd0) ? 5'd0 :
        gemm_attention_qk_m_blocks_m1[4:0];

    assign shared_launch_req_qk = qk_core_cfg_start;
    assign shared_launch_req = qk_core_cfg_start || gemm_core_cfg_start;
    assign shared_launch_can_accept = !shared_launch_valid_q && !gemm_core_error_sticky;
    assign shared_launch_fire = shared_launch_valid_q && gemm_core_cfg_ready;

    assign qk_core_cfg_ready = shared_launch_can_accept && !gemm_macro_busy_q &&
                               !gemm_drain_active_q && !gemm_pv_drain_prepare_q &&
                               co_acc_pipeline_idle;
    assign gemm_core_qk_owner = shared_core_active_qk_q;
    assign gemm_core_done_for_macro = gemm_core_done && !shared_core_active_qk_q;
    assign shared_core_cfg_start = shared_launch_fire;
    assign shared_core_a_base = shared_launch_req_qk ? qk_core_a_base_word_addr : gemm_core_a_base;
    assign shared_core_a_k_word_offset = shared_launch_req_qk ? qk_core_a_k_word_offset : gemm_a_k_word_offset_i;
    assign shared_core_a_row_stride_words = shared_launch_req_qk ? qk_core_a_row_stride_words : gemm_a_row_stride_words_q;
    assign shared_core_w_base = shared_launch_req_qk ? qk_core_w_base_word_addr : gemm_core_w_base;
    assign shared_core_m_count = shared_launch_req_qk ? qk_core_m_count : gemm_core_m_count;
    assign shared_core_n_count = shared_launch_req_qk ? qk_core_n_count : gemm_core_n_count;
    assign shared_core_k_count = shared_launch_req_qk ? qk_core_k_count : gemm_k_count_q;
    assign shared_core_mode = shared_launch_req_qk ? qk_core_mode : gemm_mode_q;
    assign shared_core_snapshot_ready = shared_core_active_qk_q ? qk_core_snapshot_ready : gemm_core_snapshot_ready;
    assign shared_core_a_bank = shared_launch_req_qk ? attention_qk_q_bank_i : gemm_a_bank_q;
    assign shared_core_w_bank = shared_launch_req_qk ? attention_qk_k_bank_i : active_w_bank_i;

    assign gemm_cfg_ready_o    = gemm_cfg_attention_qk_mode
        ? attention_qk_req_ready_o
        : (!gemm_macro_busy_q && !attention_qk_busy &&
           shared_launch_can_accept && !gemm_core_error_sticky);
    assign gemm_busy_o         = gemm_macro_busy_q || gemm_core_busy || attention_qk_busy;
    assign gemm_done_o         = gemm_macro_done_q || attention_qk_done_o;
    assign gemm_error_sticky_o = gemm_core_error_sticky || attention_qk_error_sticky_o;
    assign gemm_last_error_o   = gemm_core_error_sticky ? gemm_core_last_error :
                                 attention_qk_error_sticky_o ? attention_qk_last_error_o :
                                 NPU_ERR_NONE;
    assign co_acc_pipeline_idle = !co_src_valid &&
                                  !co_acc_stage_load_valid &&
                                  !co_acc_stage_valid_q && !co_acc_out_valid &&
                                  !pv_norm_busy &&
                                  !bias_pending_q && !bias_rsp_valid_q &&
                                  !bias_cache_prefetch_active_q &&
                                  !old_acc_pending_q && !old_acc_rsp_valid_q;
    assign gemm_compute_done_available = gemm_core_done_pending_q || gemm_core_done_for_macro;
    assign gemm_drain_slot_available =
        (!gemm_drain_active_q && !gemm_pv_drain_prepare_q) ||
        gemm_drain_done_next;
    assign gemm_core_snapshot_ready = gemm_core_snapshot_ready_q;
    // The SA shadow is the single result slot. A following tile may compute
    // while the previous tile drains, but its final snapshot must wait until
    // the active C/O drain context has retired.
    assign gemm_microtile_done_ready = gemm_macro_busy_q &&
                                       gemm_compute_done_available &&
                                       shared_launch_can_accept;
    assign gemm_drain_input_last_fire = gemm_drain_valid && gemm_co_ready &&
                                        (gemm_co_accept_idx_q == 8'd127);
    assign gemm_pv_row_last_fire = gemm_pv_drain_mode && gemm_drain_valid && gemm_co_ready &&
                                   (gemm_co_accept_idx_q[1:0] == 2'd3);

    function automatic logic [7:0] attention_qk_add_correction_u8(
        input logic [7:0] local_d_i,
        input logic signed [32:0] correction_i
    );
        logic [8:0] sum;
        begin
            if (local_d_i == 8'hff) begin
                attention_qk_add_correction_u8 = 8'hff;
            end else if (correction_i <= 33'sd0) begin
                attention_qk_add_correction_u8 = local_d_i;
            end else if (correction_i >= 33'sd255) begin
                attention_qk_add_correction_u8 = 8'hff;
            end else begin
                sum = {1'b0, local_d_i} + correction_i[8:0];
                attention_qk_add_correction_u8 = sum[8] ? 8'hff : sum[7:0];
            end
        end
    endfunction

    function automatic logic [7:0] attention_qk_distance_to_logp_u8(
        input logic valid_i,
        input logic [7:0] d_i
    );
        logic [16:0] scaled;
        begin
            if (!valid_i) begin
                attention_qk_distance_to_logp_u8 = 8'd0;
            end else begin
                scaled = ({9'd0, d_i} * 17'd254 + 17'd128) >> 8;
                attention_qk_distance_to_logp_u8 = 8'd255 - scaled[7:0];
            end
        end
    endfunction

    assign gemm_drain_write_last_fire = gemm_drain_active_q &&
                                        co_acc_out_valid && co_acc_out_ready &&
                                        (gemm_co_word_idx_q == 8'd127);
    assign gemm_drain_bias_prefetch_fire = gemm_pv_drain_mode
        ? gemm_drain_write_last_fire
        : gemm_drain_input_last_fire;
    assign gemm_drain_input_done_next = gemm_drain_input_done_q ||
                                        gemm_drain_input_last_fire;
    assign gemm_drain_write_done_next = gemm_drain_write_done_q ||
                                        gemm_drain_write_last_fire;
    assign gemm_drain_done_next = gemm_drain_input_done_next &&
                                  gemm_drain_write_done_next;

    assign mvout_req_ready_o = !mvout_active_q &&
                               (mvout_outstanding_count_q == '0) &&
                               (mvout_resp_count_q == '0) &&
                               !old_acc_pending_q &&
                               !old_acc_rsp_valid_q &&
                               !old_acc_read_fire &&
                               !scale_pending_q &&
                               (mvout_scale_count_q == '0) &&
                               !mvout_attn_s0_valid_q &&
                               !mvout_attn_s1_valid_q &&
                               !mvout_dma_valid_q &&
                               !mvout_convert_dma_valid &&
                               mvout_co_ready;
    assign mvout_attn_mode = (mvout_mode_q == NPU_MVOUT_ATTENTION_QK_LOGP);
    assign mvout_prefetch_pending_count =
        {1'b0, mvout_outstanding_count_q} + {1'b0, mvout_resp_count_q};
    assign mvout_scale_pending_count =
        {1'b0, mvout_scale_outstanding_count_q} + {1'b0, mvout_scale_count_q};
    assign scale_pending_q = (mvout_scale_outstanding_count_q != '0);
    assign mvout_prefetch_candidate =
        mvout_active_q &&
        !mvout_issue_done_q &&
        (mvout_prefetch_pending_count < {1'b0, MVOUT_PREFETCH_DEPTH_COUNT}) &&
        (((mvout_mode_q != NPU_MVOUT_FP32_Q8_24) && !mvout_attn_mode) ||
         ((mvout_mode_q == NPU_MVOUT_FP32_Q8_24) && !mvout_per_channel_scale_q) ||
         (mvout_scale_pending_count < {1'b0, MVOUT_PREFETCH_DEPTH_COUNT}));
    assign mvout_prefetch_fire = mvout_prefetch_candidate && !old_acc_read_fire;
    assign mvout_prefetch_addr = mvout_prefetch_addr_q;
    assign mvout_resp_enqueue = co_rsp_valid && (mvout_outstanding_count_q != '0);
    assign mvout_convert_input_fire = mvout_co_valid && mvout_co_ready;
    assign mvout_resp_dequeue = mvout_convert_input_fire && mvout_input_word_done;
    assign mvout_scale_rsp_push = metadata_rsp_valid &&
                                  ((meta_read_owner_q == META_READ_SCALE) ||
                                   (meta_read_owner_q == META_READ_ATTENTION_QK_MBLOCK));
    assign mvout_scale_pop_for_rsp = mvout_resp_enqueue &&
                                     (((mvout_mode_q == NPU_MVOUT_FP32_Q8_24) &&
                                       mvout_per_channel_scale_q) ||
                                      mvout_attn_mode);
    assign mvout_scale_for_rsp = (mvout_scale_count_q != '0)
        ? mvout_scale_q[mvout_scale_rd_ptr_q]
        : metadata_rsp_data;
    assign mvout_in_half_col_byte = ({16'b0, mvout_in_word_col_q} << 5) +
                                    (mvout_in_half_q ? 32'd16 : 32'd0);
    assign mvout_input_half_valid = mvout_in_half_col_byte < mvout_row_bytes_q;
    assign mvout_input_word_done = mvout_in_half_q ||
                                   ((mvout_in_half_col_byte + 32'd16) >= mvout_row_bytes_q);
    assign mvout_co_valid = (mvout_resp_count_q != '0) && mvout_input_half_valid;
    assign mvout_co_ready = mvout_attn_mode ? mvout_attn_s0_ready : mvout_convert_co_ready;
    assign mvout_co_data = mvout_in_half_q
        ? mvout_resp_data_q[mvout_resp_rd_ptr_q][255:128]
        : mvout_resp_data_q[mvout_resp_rd_ptr_q][127:0];
    assign mvout_co_scale_data = mvout_in_half_q
        ? mvout_resp_scale_q[mvout_resp_rd_ptr_q][255:128]
        : mvout_resp_scale_q[mvout_resp_rd_ptr_q][127:0];
    assign mvout_req_fire = mvout_req_valid_i && mvout_req_ready_o;
    assign mvout_req_attn_legal =
        (mvout_mode_i != NPU_MVOUT_ATTENTION_QK_LOGP) ||
        (attention_qk_chunk_valid_q &&
         (mvout_o_bank_i == attention_qk_chunk_o_bank_q) &&
         (mvout_n_count_i == attention_qk_chunk_token_count_q) &&
         (mvout_m_count_i == ({10'd0, attention_qk_chunk_q_block_count_q} << 5)) &&
         (mvout_tensor_scale_q8_24_i[15:0] ==
          ({11'd0, attention_qk_chunk_q_block_start_q} << 5)) &&
         (mvout_tensor_scale_q8_24_i[16] == attention_qk_chunk_mask_en_q));
    assign mvout_scale_req_fire = mvout_prefetch_fire &&
                                  (mvout_mode_q == NPU_MVOUT_FP32_Q8_24) &&
                                  mvout_per_channel_scale_q;
    assign mvout_attn_meta_half_index =
        (15'(mvout_row_q) * 15'(mvout_words_per_row_q)) +
        15'(mvout_word_col_q);
    assign mvout_attn_meta_word_idx = mvout_attn_meta_half_index[META_ADDR_W+3:4];
    assign mvout_attn_meta_req_fire = mvout_prefetch_fire && mvout_attn_mode;
    assign mvout_last_row_word = (mvout_word_col_q + 16'd1) >= mvout_words_per_row_q;
    assign mvout_last_read_word = ((mvout_row_q + 16'd1) >= mvout_m_count_q) &&
                                  mvout_last_row_word;
    assign mvout_out_half_col_byte = ({16'b0, mvout_out_word_col_q} << 5) +
                                     (mvout_out_half_q ? 32'd16 : 32'd0);
    assign mvout_output_remaining_bytes = (mvout_out_half_col_byte < mvout_row_bytes_q)
        ? (mvout_row_bytes_q - mvout_out_half_col_byte)
        : 32'd0;
    assign mvout_output_half_valid = mvout_out_half_col_byte < mvout_row_bytes_q;
    assign mvout_output_word_done = mvout_out_half_q ||
                                    ((mvout_out_half_col_byte + 32'd16) >= mvout_row_bytes_q);
    always_comb begin
        mvout_current_keep = 16'h0000;
        for (int byte_idx = 0; byte_idx < 16; byte_idx++) begin
            if (mvout_output_remaining_bytes > {27'b0, byte_idx[4:0]}) begin
                mvout_current_keep[byte_idx] = 1'b1;
            end
        end
    end
    assign mvout_output_last_valid_half = ((mvout_out_row_q + 16'd1) >= mvout_m_count_q) &&
                                          mvout_output_half_valid &&
                                          ((mvout_out_half_col_byte + 32'd16) >= mvout_row_bytes_q);
    assign mvout_attn_row_index = mvout_out_row_q[8:0];
    assign mvout_attn_global_row = mvout_tensor_scale_q8_24_q[15:0] + mvout_out_row_q;
    assign mvout_attn_global_col_base = mvout_out_word_col_q << 5;
    assign mvout_attn_output_meta_half_index =
        (15'(mvout_out_row_q) * 15'(mvout_words_per_row_q)) +
        15'(mvout_out_word_col_q);
    assign mvout_attn_meta_half_lane = mvout_attn_output_meta_half_index[3:0];
    assign mvout_attn_m_block =
        mvout_resp_scale_q[mvout_resp_rd_ptr_q][mvout_attn_meta_half_lane*16 +: 16];
    assign mvout_attn_m_global = attention_qk_m_global_q[mvout_attn_row_index];
    assign mvout_attn_correction =
        {{17{mvout_attn_m_global[15]}}, mvout_attn_m_global} -
        {{17{mvout_attn_m_block[15]}}, mvout_attn_m_block};
    always_comb begin
        logic [7:0] d_global;
        logic valid_lane;
        int byte_idx;

        mvout_attn_s1_data_comb = '0;
        for (int lane = 0; lane < 16; lane++) begin
            byte_idx = (mvout_attn_s0_half_q ? 16 : 0) + lane;
            valid_lane = !mvout_tensor_scale_q8_24_q[16] ||
                         ((mvout_attn_s0_col_base_q + 16'(byte_idx)) <=
                          mvout_attn_s0_global_row_q);
            d_global = attention_qk_add_correction_u8(
                mvout_attn_s0_d_word_q[byte_idx*8 +: 8],
                mvout_attn_s0_correction_q);
            mvout_attn_s1_data_comb[lane*8 +: 8] =
                attention_qk_distance_to_logp_u8(valid_lane, d_global);
        end
    end
    assign regular_mvout_dma_ready = mvout_dma_ready_i;
    assign mvout_dma_reg_ready = !mvout_dma_valid_q || regular_mvout_dma_ready;
    assign mvout_attn_s1_ready = !mvout_attn_s1_valid_q || mvout_dma_reg_ready;
    assign mvout_attn_s0_ready = !mvout_attn_s0_valid_q || mvout_attn_s1_ready;
    assign mvout_attn_s0_fire = mvout_attn_mode && mvout_co_valid && mvout_attn_s0_ready;
    assign mvout_attn_s1_to_dma_fire = mvout_attn_s1_valid_q && mvout_dma_reg_ready;
    assign mvout_convert_pop = mvout_attn_mode
        ? mvout_attn_s0_fire
        : (mvout_convert_dma_valid && mvout_convert_dma_ready);
    assign mvout_convert_dma_ready = (!mvout_attn_mode && mvout_output_half_valid)
        ? mvout_dma_reg_ready
        : 1'b1;
    assign mvout_dma_load_fire =
        (mvout_attn_mode && mvout_attn_s1_to_dma_fire) ||
        (!mvout_attn_mode && mvout_convert_pop && mvout_output_half_valid);
    assign mvout_internal_last_pop = mvout_attn_mode
        ? (mvout_attn_s1_to_dma_fire && mvout_attn_s1_last_q)
        : (mvout_convert_pop && mvout_output_last_valid_half);
    assign mvout_dma_valid_o = mvout_dma_valid_q;
    assign mvout_dma_data_o = mvout_dma_data_q;
    assign mvout_dma_keep_o = mvout_dma_keep_q;
    assign mvout_dma_addr_o = mvout_dma_addr_q;
    assign mvout_dma_last_o = mvout_dma_last_q;
    assign mvout_error_sticky_o = mvout_local_error_sticky_q || mvout_convert_error_sticky;
    assign mvout_last_error_o = mvout_local_error_sticky_q
        ? mvout_local_last_error_q
        : mvout_convert_last_error;

    a_loader_128_to_256 #(
        .ADDR_W (ADDR_W),
        .LEN_W  (16)
    ) u_a_loader (
        .clk_i                   (clk_i),
        .rst_i                   (rst_i),
        .cfg_start_i             (a_cfg_start_i),
        .cfg_ready_o             (a_cfg_ready_o),
        .cfg_base_word_addr_i    (a_cfg_base_word_addr_i),
        .cfg_row_count_i         (a_cfg_row_count_i),
        .cfg_row_bytes_i         (a_cfg_row_bytes_i),
        .cfg_row_stride_words_i  (a_cfg_row_stride_words_i),
        .cfg_a_u8_minus_128_en_i (a_cfg_u8_minus_128_en_i),
        .clear_error_i           (clear_error_i),
        .dma_valid_i             (a_dma_valid_i),
        .dma_ready_o             (a_dma_ready_o),
        .dma_data_i              (a_dma_data_i),
        .dma_keep_i              (a_dma_keep_i),
        .dma_last_i              (a_dma_last_i),
        .a_wr_valid_o            (a_wr_valid),
        .a_wr_ready_i            (a_wr_ready),
        .a_wr_word_addr_o        (a_wr_word_addr),
        .a_wr_data_o             (a_wr_data),
        .a_wr_byte_en_o          (a_wr_byte_en),
        .busy_o                  (a_busy_o),
        .done_o                  (a_done_o),
        .error_sticky_o          (a_error_sticky_o),
        .error_code_o            (a_error_code_o)
    );

    assign a_wr_ready = a_load_bank_q ? a1_wr_ready : a0_wr_ready;
    assign a_bank_conflict_o = a0_bank_conflict || a1_bank_conflict;

    uram256_bank #(
        .DEPTH_WORDS       (A_BANK_DEPTH_WORDS),
        .READ_LATENCY      (READ_LATENCY),
        .DEPTH_SLICES      (A_URAM_DEPTH_SLICES),
        .ENABLE_BYTE_WRITE (1'b1)
    ) u_a0_bank (
        .clk_i                           (clk_i),
        .rst_i                           (rst_i),
        .c_req_valid_i                   (a_wr_valid && !a_load_bank_q),
        .c_req_ready_o                   (a0_wr_ready),
        .c_req_write_i                   (1'b1),
        .c_req_word_addr_i               (a_wr_word_addr),
        .c_req_wdata_i                   (a_wr_data),
        .c_req_wstrb_i                   (a_wr_byte_en),
        .c_rsp_valid_o                   (),
        .c_rsp_rdata_o                   (),
        .d_req_valid_i                   (a0_bank_d_req_valid),
        .d_req_ready_o                   (),
        .d_req_write_i                   (1'b0),
        .d_req_word_addr_i               (a_bank_d_req_addr),
        .d_req_wdata_i                   ('0),
        .d_req_wstrb_i                   ('0),
        .d_rsp_valid_o                   (a0_bank_rsp_valid),
        .d_rsp_rdata_o                   (a0_bank_rsp_data),
        .same_addr_read_write_conflict_o (a0_bank_conflict)
    );

    uram256_bank #(
        .DEPTH_WORDS       (A_BANK_DEPTH_WORDS),
        .READ_LATENCY      (READ_LATENCY),
        .DEPTH_SLICES      (A_URAM_DEPTH_SLICES),
        .ENABLE_BYTE_WRITE (1'b1)
    ) u_a1_bank (
        .clk_i                           (clk_i),
        .rst_i                           (rst_i),
        .c_req_valid_i                   (a_wr_valid && a_load_bank_q),
        .c_req_ready_o                   (a1_wr_ready),
        .c_req_write_i                   (1'b1),
        .c_req_word_addr_i               (a_wr_word_addr),
        .c_req_wdata_i                   (a_wr_data),
        .c_req_wstrb_i                   (a_wr_byte_en),
        .c_rsp_valid_o                   (),
        .c_rsp_rdata_o                   (),
        .d_req_valid_i                   (a1_bank_d_req_valid),
        .d_req_ready_o                   (),
        .d_req_write_i                   (1'b0),
        .d_req_word_addr_i               (a_bank_d_req_addr),
        .d_req_wdata_i                   ('0),
        .d_req_wstrb_i                   ('0),
        .d_rsp_valid_o                   (a1_bank_rsp_valid),
        .d_rsp_rdata_o                   (a1_bank_rsp_data),
        .same_addr_read_write_conflict_o (a1_bank_conflict)
    );

    w_loader_128_to_256 #(
        .ADDR_W (ADDR_W),
        .LEN_W  (16)
    ) u_w_loader (
        .clk_i                 (clk_i),
        .rst_i                 (rst_i),
        .cfg_start_i           (w_cfg_start_i),
        .cfg_ready_o           (w_cfg_ready_o),
        .cfg_base_word_addr_i  (w_cfg_base_word_addr_i),
        .cfg_k_loaded_i        (w_cfg_k_loaded_i),
        .cfg_n_loaded_i        (w_cfg_n_loaded_i),
        .cfg_selected_w_bank_i (w_cfg_selected_bank_i),
        .active_w_bank_i       (active_w_bank_i),
        .clear_error_i         (clear_error_i),
        .dma_valid_i           (w_dma_valid_i),
        .dma_ready_o           (w_dma_ready_o),
        .dma_data_i            (w_dma_data_i),
        .dma_keep_i            (w_dma_keep_i),
        .dma_last_i            (w_dma_last_i),
        .w0_wr_valid_o         (w0_wr_valid),
        .w0_wr_ready_i         (w0_wr_ready),
        .w0_wr_word_addr_o     (w0_wr_word_addr),
        .w0_wr_data_o          (w0_wr_data),
        .w0_wr_byte_en_o       (w0_wr_byte_en),
        .w1_wr_valid_o         (w1_wr_valid),
        .w1_wr_ready_i         (w1_wr_ready),
        .w1_wr_word_addr_o     (w1_wr_word_addr),
        .w1_wr_data_o          (w1_wr_data),
        .w1_wr_byte_en_o       (w1_wr_byte_en),
        .busy_o                (w_busy_o),
        .done_o                (w_done_o),
        .error_sticky_o        (w_error_sticky_o),
        .error_code_o          (w_error_code_o)
    );

    uram256_bank #(
        .DEPTH_WORDS       (W_BANK_DEPTH_WORDS),
        .READ_LATENCY      (READ_LATENCY),
        .DEPTH_SLICES      (W_URAM_DEPTH_SLICES),
        .ENABLE_BYTE_WRITE (1'b0)
    ) u_w0_bank (
        .clk_i                           (clk_i),
        .rst_i                           (rst_i),
        .c_req_valid_i                   (w0_wr_valid),
        .c_req_ready_o                   (w0_wr_ready),
        .c_req_write_i                   (1'b1),
        .c_req_word_addr_i               (w0_wr_word_addr),
        .c_req_wdata_i                   (w0_wr_data),
        .c_req_wstrb_i                   (w0_wr_byte_en),
        .c_rsp_valid_o                   (),
        .c_rsp_rdata_o                   (),
        .d_req_valid_i                   (w0_bank_d_req_valid),
        .d_req_ready_o                   (),
        .d_req_write_i                   (1'b0),
        .d_req_word_addr_i               (w_bank_d_req_addr),
        .d_req_wdata_i                   ('0),
        .d_req_wstrb_i                   ('0),
        .d_rsp_valid_o                   (w0_rsp_valid),
        .d_rsp_rdata_o                   (w0_rsp_data),
        .same_addr_read_write_conflict_o (w0_bank_conflict_o)
    );

    uram256_bank #(
        .DEPTH_WORDS       (W_BANK_DEPTH_WORDS),
        .READ_LATENCY      (READ_LATENCY),
        .DEPTH_SLICES      (W_URAM_DEPTH_SLICES),
        .ENABLE_BYTE_WRITE (1'b0)
    ) u_w1_bank (
        .clk_i                           (clk_i),
        .rst_i                           (rst_i),
        .c_req_valid_i                   (w1_wr_valid),
        .c_req_ready_o                   (w1_wr_ready),
        .c_req_write_i                   (1'b1),
        .c_req_word_addr_i               (w1_wr_word_addr),
        .c_req_wdata_i                   (w1_wr_data),
        .c_req_wstrb_i                   (w1_wr_byte_en),
        .c_rsp_valid_o                   (),
        .c_rsp_rdata_o                   (),
        .d_req_valid_i                   (w1_bank_d_req_valid),
        .d_req_ready_o                   (),
        .d_req_write_i                   (1'b0),
        .d_req_word_addr_i               (w_bank_d_req_addr),
        .d_req_wdata_i                   ('0),
        .d_req_wstrb_i                   ('0),
        .d_rsp_valid_o                   (w1_rsp_valid),
        .d_rsp_rdata_o                   (w1_rsp_data),
        .same_addr_read_write_conflict_o (w1_bank_conflict_o)
    );

    always_comb begin
        bias_base_word128 = co_bias_addr_i[META_BYTE_ADDR_W-1:4];
        if (gemm_drain_to_co_valid) begin
            bias_word_idx = (META_ADDR_W'(gemm_drain_n_group_q) << 2) +
                            META_ADDR_W'(co_src_word_group_idx);
        end else begin
            bias_word_idx = META_ADDR_W'(co_wr_word_addr_q[1:0]);
        end

        mvout_scale_base_word128 = mvout_scale_addr_q[META_BYTE_ADDR_W-1:4];
        mvout_scale_word_idx = META_ADDR_W'(mvout_word_col_q);
    end

    always_comb begin
        logic [15:0] next_n_group_for_bias;
        logic [15:0] next_n_base_for_bias;

        next_n_group_for_bias = gemm_drain_n_group_q + 16'd1;
        next_n_base_for_bias = next_n_group_for_bias << 5;
        bias_cache_next_n_group = (next_n_base_for_bias < gemm_n_total_q)
            ? next_n_group_for_bias
            : 16'd0;
    end

    assign bias_cache_prefetch_fire = bias_cache_prefetch_active_q &&
                                      !mvout_scale_req_fire &&
                                      !bias_read_fire;

    always_comb begin
        metadata_read_valid = 1'b0;
        metadata_read_base_word128 = '0;
        metadata_read_word_idx = '0;
        meta_read_owner_d = META_READ_NONE;

        if (mvout_scale_req_fire) begin
            metadata_read_valid = 1'b1;
            metadata_read_base_word128 = mvout_scale_base_word128;
            metadata_read_word_idx = mvout_scale_word_idx;
            meta_read_owner_d = META_READ_SCALE;
        end else if (mvout_attn_meta_req_fire) begin
            metadata_read_valid = 1'b1;
            metadata_read_base_word128 = '0;
            metadata_read_word_idx = mvout_attn_meta_word_idx;
            meta_read_owner_d = META_READ_ATTENTION_QK_MBLOCK;
        end else if (bias_read_fire) begin
            metadata_read_valid = 1'b1;
            metadata_read_base_word128 = bias_base_word128;
            metadata_read_word_idx = bias_word_idx;
            meta_read_owner_d = META_READ_BIAS;
        end else if (bias_cache_prefetch_fire) begin
            metadata_read_valid = 1'b1;
            metadata_read_base_word128 = bias_base_word128;
            metadata_read_word_idx = (META_ADDR_W'(bias_cache_n_group_q) << 2) +
                                     META_ADDR_W'(bias_cache_issue_idx_q);
            meta_read_owner_d = META_READ_BIAS_CACHE;
        end else if (meta_read256_valid_i) begin
            metadata_read_valid = 1'b1;
            metadata_read_base_word128 = meta_read256_base_word128_i;
            metadata_read_word_idx = meta_read256_word_idx_i;
            meta_read_owner_d = META_READ_EXTERNAL;
        end
    end

    assign meta_read256_rsp_valid_o = metadata_rsp_valid &&
                                      (meta_read_owner_q == META_READ_EXTERNAL);
    assign meta_read256_rsp_data_o  = metadata_rsp_data;
    assign metadata_mvin_valid = attention_qk_meta_wr_valid ? 1'b1 : meta_mvin_valid_i;
    assign metadata_mvin_addr = attention_qk_meta_wr_valid
        ? attention_qk_meta_wr_addr
        : meta_mvin_word128_addr_i;
    assign metadata_mvin_data = attention_qk_meta_wr_valid
        ? attention_qk_meta_wr_data
        : meta_mvin_data_i;
    assign metadata_mvin_keep = attention_qk_meta_wr_valid
        ? attention_qk_meta_wr_keep
        : meta_mvin_keep_i;
    assign attention_qk_meta_wr_ready = attention_qk_meta_wr_valid && metadata_mvin_ready;
    assign meta_mvin_ready_o = !attention_qk_meta_wr_valid && metadata_mvin_ready;

    co_metadata_buffer #(
        .WORD128_DEPTH (META_WORD128_DEPTH)
    ) u_metadata (
        .clk_i                  (clk_i),
        .rst_i                  (rst_i),
        .co_active_i            (gemm_macro_busy_q || co_src_valid ||
                                  co_acc_in_valid || co_acc_out_valid ||
                                  pv_norm_busy ||
                                  bias_pending_q || bias_rsp_valid_q ||
                                  old_acc_pending_q || old_acc_rsp_valid_q ||
                                  mvout_active_q ||
                                  (mvout_outstanding_count_q != '0) ||
                                  (mvout_resp_count_q != '0) ||
                                  (mvout_scale_count_q != '0) ||
                                  scale_pending_q),
        .clear_error_i          (clear_error_i),
        .mvin_valid_i           (metadata_mvin_valid),
        .mvin_ready_o           (metadata_mvin_ready),
        .mvin_word128_addr_i    (metadata_mvin_addr),
        .mvin_data_i            (metadata_mvin_data),
        .mvin_keep_i            (metadata_mvin_keep),
        .read256_valid_i        (metadata_read_valid),
        .read256_base_word128_i (metadata_read_base_word128),
        .read256_word_idx_i     (metadata_read_word_idx),
        .read256_rsp_valid_o    (metadata_rsp_valid),
        .read256_rsp_data_o     (metadata_rsp_data),
        .conflict_sticky_o      (meta_conflict_sticky_o),
        .last_error_o           (meta_last_error_o)
    );

    function automatic logic [15:0] min_u16(input logic [15:0] a, input logic [15:0] b);
        min_u16 = (a < b) ? a : b;
    endfunction

    always_comb begin
        logic [15:0] m_remaining;
        logic [15:0] n_base;
        logic [15:0] n_remaining;
        logic [15:0] m_cur;
        logic [15:0] n_cur;

        m_remaining = gemm_m_total_q - gemm_m_base_q;
        n_base      = gemm_n_group_q << 5;
        n_remaining = gemm_n_total_q - n_base;
        m_cur       = min_u16(m_remaining, 16'd32);
        n_cur       = min_u16(n_remaining, 16'd32);

        gemm_core_a_base  = gemm_a_base_word_addr_q +
                            ADDR_W'(gemm_m_base_q) * gemm_a_row_stride_words_q;
        gemm_core_w_base  = gemm_w_base_word_addr_q +
                            ADDR_W'(gemm_n_group_q) * ADDR_W'(gemm_k_count_q);
        gemm_core_m_count = m_cur[5:0];
        gemm_core_n_count = n_cur[5:0];
    end

    always_comb begin
        co_src_word_addr = co_wr_word_addr_q;
        if (gemm_pv_norm_to_co_valid) begin
            co_src_word_addr = pv_norm_out_word_addr;
        end else if (gemm_non_pv_drain_to_co_valid) begin
            co_src_word_addr = gemm_drain_word_addr_q;
        end
    end

    always_comb begin
        co_acc_stage_load_word_addr = co_src_word_addr;
        if (bias_rsp_valid_q) begin
            co_acc_stage_load_word_addr = bias_word_addr_q;
        end else if (old_acc_rsp_valid_q) begin
            co_acc_stage_load_word_addr = old_acc_word_addr_q;
        end
    end

    assign co_bank_wr_addr = co_acc_out_word_addr_q;
    assign co_read_bank = old_acc_read_fire ? co_write_bank_q : mvout_o_bank_q;
    assign co_split_wr_addr = co_bank_wr_addr +
                              (co_write_bank_q ? ADDR_W'(CO_HALF_DEPTH_WORDS) : '0);
    assign co_split_rd_addr = co_bank_d_req_addr +
                              (co_read_bank ? ADDR_W'(CO_HALF_DEPTH_WORDS) : '0);
    assign co_acc_in_valid = co_acc_stage_valid_q;
    assign co_acc_accumulate_en = co_acc_stage_accumulate_en_q;
    assign co_acc_add_bias_en = co_acc_stage_add_bias_en_q;
    assign co_acc_partial_data = co_acc_stage_partial_data_q;
    assign co_acc_lane_mask = co_acc_stage_lane_mask_q;
    assign co_acc_old_acc_data = co_acc_stage_old_acc_data_q;
    assign co_acc_bias_data = co_acc_stage_bias_data_q;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            shared_launch_valid_q <= 1'b0;
            shared_launch_owner_q <= 1'b0;
            shared_core_active_qk_q <= 1'b0;
            shared_core_active_a_bank_q <= 1'b0;
            shared_core_active_w_bank_q <= 1'b0;
        end else begin
            if (clear_error_i) begin
                shared_launch_valid_q <= 1'b0;
                shared_core_active_qk_q <= 1'b0;
            end else begin
                if (attention_qk_done_o || attention_qk_error_sticky_o) begin
                    shared_core_active_qk_q <= 1'b0;
                end

                if (shared_launch_fire) begin
                    shared_launch_valid_q <= 1'b0;
                    shared_core_active_qk_q <= shared_launch_owner_q;
                    shared_core_active_a_bank_q <= shared_launch_a_bank_q;
                    shared_core_active_w_bank_q <= shared_launch_w_bank_q;
                end

                if (shared_launch_req && shared_launch_can_accept) begin
                    shared_launch_valid_q <= 1'b1;
                    shared_launch_owner_q <= shared_launch_req_qk;
                    shared_launch_a_base_q <= shared_core_a_base;
                    shared_launch_a_k_word_offset_q <= shared_core_a_k_word_offset;
                    shared_launch_a_row_stride_words_q <= shared_core_a_row_stride_words;
                    shared_launch_w_base_q <= shared_core_w_base;
                    shared_launch_m_count_q <= shared_core_m_count;
                    shared_launch_n_count_q <= shared_core_n_count;
                    shared_launch_k_count_q <= shared_core_k_count;
                    shared_launch_mode_q <= shared_core_mode;
                    shared_launch_a_bank_q <= shared_core_a_bank;
                    shared_launch_w_bank_q <= shared_core_w_bank;
                end
            end
        end
    end

    spm_sa_gemm_i8_core #(
        .ADDR_W (ADDR_W),
        .DISABLE_ERROR_CHECKS (DISABLE_ERROR_CHECKS)
    ) u_gemm_core (
        .clk_i                    (clk_i),
        .rst_i                    (rst_i),
        .clear_error_i            (clear_error_i),
        .cfg_start_i              (shared_core_cfg_start),
        .cfg_ready_o              (gemm_core_cfg_ready),
        .cfg_a_base_word_addr_i   (shared_launch_a_base_q),
        .cfg_a_k_word_offset_i    (shared_launch_a_k_word_offset_q),
        .cfg_a_row_stride_words_i (shared_launch_a_row_stride_words_q),
        .cfg_w_base_word_addr_i   (shared_launch_w_base_q),
        .cfg_m_count_i            (shared_launch_m_count_q),
        .cfg_n_count_i            (shared_launch_n_count_q),
        .cfg_k_count_i            (shared_launch_k_count_q),
        .cfg_mode_i               (shared_launch_mode_q),
        .snapshot_ready_i         (shared_core_snapshot_ready),
        .a_rd_req_valid_o         (gemm_a_rd_req_valid),
        .a_rd_req_ready_i         (gemm_a_rd_req_ready),
        .a_rd_req_word_addr_o     (gemm_a_rd_req_addr),
        .a_rd_rsp_valid_i         (gemm_a_rd_rsp_valid),
        .a_rd_rsp_data_i          (gemm_a_rd_rsp_data),
        .w_rd_req_valid_o         (gemm_w_rd_req_valid),
        .w_rd_req_ready_i         (gemm_w_rd_req_ready),
        .w_rd_req_word_addr_o     (gemm_w_rd_req_addr),
        .w_rd_rsp_valid_i         (gemm_w_rd_rsp_valid),
        .w_rd_rsp_data_i          (gemm_w_rd_rsp_data),
        .co_out_valid_o           (gemm_co_valid),
        .co_out_ready_i           (gemm_co_ready),
        .co_out_data_o            (gemm_co_data),
        .co_out_lane_mask_o       (gemm_co_lane_mask),
        .pv_row_sum_valid_o       (gemm_pv_row_sum_valid),
        .pv_row_sum_o             (gemm_pv_row_sum),
        .busy_o                   (gemm_core_busy),
        .done_o                   (gemm_core_done),
        .error_sticky_o           (gemm_core_error_sticky),
        .last_error_o             (gemm_core_last_error)
    );

    pv_i32_row_normalize #(
        .ADDR_W (ADDR_W)
    ) u_pv_normalize (
        .clk_i              (clk_i),
        .rst_i              (rst_i),
        .clear_i            (clear_error_i),
        .in_valid_i         (pv_norm_in_valid),
        .in_ready_o         (pv_norm_in_ready),
        .numerator_i        (gemm_co_data),
        .denominator_i      (pv_norm_denominator),
        .lane_mask_i        (gemm_co_lane_mask),
        .word_addr_i        (gemm_drain_word_addr_q),
        .word_group_idx_i   (gemm_co_accept_idx_q[1:0]),
        .out_valid_o        (pv_norm_out_valid),
        .out_ready_i        (pv_norm_out_ready),
        .normalized_o       (pv_norm_out_data),
        .lane_mask_o        (pv_norm_out_lane_mask),
        .word_addr_o        (pv_norm_out_word_addr),
        .word_group_idx_o   (pv_norm_out_word_group_idx),
        .busy_o             (pv_norm_busy)
    );

    gemm_mode_scheduler #(
        .ADDR_W             (ADDR_W),
        .META_ADDR_W        (META_ADDR_W),
        .O_BANK_DEPTH_WORDS (O_BANK_DEPTH_WORDS),
        .GAMMA16_FRAC       (24),
        .DEBUG_PRINT        (1'b0)
    ) u_gemm_mode_scheduler (
        .clk_i                       (clk_i),
        .rst_i                       (rst_i),
        .clear_error_i               (clear_error_i),
        .qk_start_i                  (gemm_attention_qk_start_fire),
        .qk_ready_o                  (attention_qk_req_ready_o),
        .qk_busy_o                   (attention_qk_busy),
        .qk_done_o                   (attention_qk_done_o),
        .qk_token_count_i            (gemm_n_count_i),
        .qk_gamma16_fix_i            (attention_qk_gamma16_fix_i),
        .qk_mask_en_i                (attention_qk_mask_en_i),
        .qk_q_block_start_i          (attention_qk_q_block_start_i),
        .qk_q_block_count_m1_i       (attention_qk_q_block_count_m1_i),
        .core_cfg_start_o            (qk_core_cfg_start),
        .core_cfg_ready_i            (qk_core_cfg_ready),
        .core_a_base_word_addr_o     (qk_core_a_base_word_addr),
        .core_a_k_word_offset_o      (qk_core_a_k_word_offset),
        .core_a_row_stride_words_o   (qk_core_a_row_stride_words),
        .core_w_base_word_addr_o     (qk_core_w_base_word_addr),
        .core_m_count_o              (qk_core_m_count),
        .core_n_count_o              (qk_core_n_count),
        .core_k_count_o              (qk_core_k_count),
        .core_mode_o                 (qk_core_mode),
        .core_snapshot_ready_o       (qk_core_snapshot_ready),
        .core_busy_i                 (gemm_core_qk_owner && gemm_core_busy),
        .core_done_i                 (gemm_core_qk_owner && gemm_core_done),
        .core_error_sticky_i         (gemm_core_qk_owner && gemm_core_error_sticky),
        .core_last_error_i           (gemm_core_last_error),
        .core_co_valid_i             (gemm_core_qk_owner && gemm_co_valid),
        .core_co_ready_o             (qk_core_co_ready),
        .core_co_data_i              (gemm_co_data),
        .core_co_lane_mask_i         (gemm_co_lane_mask),
        .qk_o_wr_req_valid_o         (attention_qk_o_wr_req_valid),
        .qk_o_wr_req_ready_i         (attention_qk_o_wr_req_ready),
        .qk_o_wr_req_addr_o          (attention_qk_o_wr_req_addr),
        .qk_o_wr_req_data_o          (attention_qk_o_wr_req_data),
        .qk_o_wr_req_byte_en_o       (attention_qk_o_wr_req_byte_en),
        .qk_meta_wr_valid_o          (attention_qk_meta_wr_valid),
        .qk_meta_wr_ready_i          (attention_qk_meta_wr_ready),
        .qk_meta_wr_addr_o           (attention_qk_meta_wr_addr),
        .qk_meta_wr_data_o           (attention_qk_meta_wr_data),
        .qk_meta_wr_keep_o           (attention_qk_meta_wr_keep),
        .qk_m_global_wr_valid_o      (attention_qk_m_global_wr_valid),
        .qk_m_global_wr_row_o        (attention_qk_m_global_wr_row),
        .qk_m_global_wr_data_o       (attention_qk_m_global_wr_data),
        .qk_error_sticky_o           (attention_qk_error_sticky_o),
        .qk_last_error_o             (attention_qk_last_error_o)
    );

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            attention_qk_chunk_valid_q <= 1'b0;
            attention_qk_chunk_o_bank_q <= 1'b0;
            attention_qk_chunk_token_count_q <= '0;
            attention_qk_chunk_q_block_start_q <= '0;
            attention_qk_chunk_q_block_count_q <= '0;
            attention_qk_chunk_mask_en_q <= 1'b0;
            for (int row = 0; row < 512; row++) begin
                attention_qk_m_global_q[row] <= '0;
            end
        end else begin
            if (clear_error_i) begin
                attention_qk_chunk_valid_q <= 1'b0;
            end

            if (attention_qk_m_global_wr_valid) begin
                attention_qk_m_global_q[attention_qk_m_global_wr_row] <=
                    attention_qk_m_global_wr_data;
            end

            if (gemm_attention_qk_start_fire) begin
                attention_qk_chunk_valid_q <= 1'b0;
            end

            if (attention_qk_done_o && !attention_qk_error_sticky_o) begin
                attention_qk_chunk_valid_q <= 1'b1;
                attention_qk_chunk_o_bank_q <= gemm_o_bank_i;
                attention_qk_chunk_token_count_q <= gemm_n_count_i;
                attention_qk_chunk_q_block_start_q <= attention_qk_q_block_start_i;
                attention_qk_chunk_q_block_count_q <=
                    {1'b0, attention_qk_q_block_count_m1_i} + 6'd1;
                attention_qk_chunk_mask_en_q <= attention_qk_mask_en_i;
            end

            if ((meta_mvin_valid_i && meta_mvin_ready_o) ||
                (gemm_cfg_start_i && gemm_cfg_ready_o &&
                 attention_qk_chunk_valid_q &&
                 (gemm_o_bank_i == attention_qk_chunk_o_bank_q))) begin
                attention_qk_chunk_valid_q <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk_i) begin
        logic [15:0] next_n_group;
        logic [15:0] next_n_base;
        logic [15:0] next_m_base;

        if (rst_i) begin
            gemm_core_cfg_start       <= 1'b0;
            gemm_core_done_pipe_q      <= '0;
            gemm_macro_busy_q         <= 1'b0;
            gemm_macro_done_q         <= 1'b0;
            gemm_all_tiles_done_q     <= 1'b0;
            gemm_core_done_pending_q  <= 1'b0;
            gemm_core_snapshot_ready_q <= 1'b1;
            gemm_a_base_word_addr_q   <= '0;
            gemm_a_row_stride_words_q <= '0;
            gemm_w_base_word_addr_q   <= '0;
            gemm_co_base_word_addr_q  <= '0;
            gemm_co_row_stride_words_q <= '0;
            gemm_m_total_q            <= '0;
            gemm_n_total_q            <= '0;
            gemm_k_count_q            <= '0;
            gemm_m_base_q             <= '0;
            gemm_n_group_q            <= '0;
            gemm_add_bias_q           <= 1'b0;
            gemm_accumulate_q         <= 1'b0;
            gemm_mode_q               <= NPU_MODE_INT8;
            gemm_a_bank_q             <= 1'b0;
            gemm_o_bank_q             <= 1'b0;
            gemm_co_accept_idx_q      <= '0;
            gemm_drain_active_q       <= 1'b0;
            gemm_pv_drain_prepare_q   <= 1'b0;
            gemm_drain_m_base_q       <= '0;
            gemm_drain_n_group_q      <= '0;
            gemm_drain_word_addr_q    <= '0;
            gemm_drain_input_done_q   <= 1'b0;
            gemm_drain_write_done_q   <= 1'b0;
        end else begin
            gemm_core_cfg_start <= 1'b0;
            gemm_core_done_pipe_q <= {gemm_core_done_pipe_q[0], gemm_core_done_for_macro};
            gemm_macro_done_q   <= 1'b0;
            gemm_core_snapshot_ready_q <= gemm_drain_slot_available;
            if (gemm_core_done_for_macro && !gemm_microtile_done_ready) begin
                gemm_core_done_pending_q <= 1'b1;
            end

            if (gemm_pv_drain_prepare_q && gemm_pv_row_sum_valid) begin
                for (int row = 0; row < NPU_SA_ROWS; row++) begin
                    gemm_pv_row_sum_q[row] <= gemm_pv_row_sum[row];
                end
                gemm_pv_drain_prepare_q <= 1'b0;
                gemm_drain_active_q     <= 1'b1;
            end

            if (gemm_core_done_for_macro) begin
                gemm_drain_active_q  <= !gemm_pv_mode ||
                    (gemm_pv_mode && gemm_pv_row_sum_valid);
                gemm_pv_drain_prepare_q <= gemm_pv_mode && !gemm_pv_row_sum_valid;
                gemm_drain_m_base_q  <= gemm_m_base_q;
                gemm_drain_n_group_q <= gemm_n_group_q;
                gemm_drain_word_addr_q <= gemm_co_base_word_addr_q +
                    ADDR_W'(gemm_m_base_q) * gemm_co_row_stride_words_q +
                    ADDR_W'(gemm_n_group_q) * ADDR_W'(4);
                gemm_co_accept_idx_q <= '0;
                gemm_drain_input_done_q <= 1'b0;
                gemm_drain_write_done_q <= 1'b0;
                if (gemm_pv_mode && gemm_pv_row_sum_valid) begin
                    for (int row = 0; row < NPU_SA_ROWS; row++) begin
                        gemm_pv_row_sum_q[row] <= gemm_pv_row_sum[row];
                    end
                end
            end else begin
                gemm_drain_input_done_q <= gemm_drain_input_done_next;
                gemm_drain_write_done_q <= gemm_drain_write_done_next;
            end

            if (gemm_drain_done_next && !gemm_core_done_for_macro) begin
                gemm_drain_active_q <= 1'b0;
                gemm_pv_drain_prepare_q <= 1'b0;
            end

            if (gemm_all_tiles_done_q && !gemm_drain_active_q &&
                !gemm_pv_drain_prepare_q &&
                co_acc_pipeline_idle && !gemm_core_busy) begin
                gemm_macro_busy_q     <= 1'b0;
                gemm_macro_done_q     <= 1'b1;
                gemm_all_tiles_done_q <= 1'b0;
            end

            if (gemm_cfg_start_i && gemm_cfg_ready_o && !gemm_cfg_attention_qk_mode) begin
                gemm_a_base_word_addr_q    <= gemm_a_base_word_addr_i;
                gemm_a_row_stride_words_q  <= gemm_a_row_stride_words_i;
                gemm_w_base_word_addr_q    <= gemm_w_base_word_addr_i;
                gemm_co_base_word_addr_q   <= co_wr_base_word_addr_i;
                gemm_co_row_stride_words_q <= ADDR_W'((gemm_n_count_i + 16'd7) >> 3);
                gemm_m_total_q             <= gemm_m_count_i;
                gemm_n_total_q             <= gemm_n_count_i;
                gemm_k_count_q             <= gemm_k_count_i;
                gemm_m_base_q              <= '0;
                gemm_n_group_q             <= '0;
                gemm_add_bias_q            <= co_add_bias_en_i;
                gemm_accumulate_q          <= co_accumulate_en_i;
                gemm_mode_q                <= gemm_mode_i;
                gemm_a_bank_q              <= gemm_a_bank_i;
                gemm_o_bank_q              <= gemm_o_bank_i;
                gemm_macro_busy_q          <= 1'b1;
                gemm_all_tiles_done_q      <= 1'b0;
                gemm_core_cfg_start        <= 1'b1;
                gemm_core_done_pipe_q      <= '0;
                gemm_core_done_pending_q   <= 1'b0;
                gemm_co_accept_idx_q       <= '0;
                gemm_drain_active_q        <= 1'b0;
                gemm_pv_drain_prepare_q    <= 1'b0;
                gemm_drain_word_addr_q     <= '0;
                gemm_drain_input_done_q    <= 1'b0;
                gemm_drain_write_done_q    <= 1'b0;
            end else if (gemm_microtile_done_ready) begin
                gemm_core_done_pending_q <= 1'b0;
                next_n_group = gemm_n_group_q + 16'd1;
                next_n_base = next_n_group << 5;
                if (next_n_base < gemm_n_total_q) begin
                    gemm_n_group_q      <= next_n_group;
                    gemm_core_cfg_start <= 1'b1;
                end else begin
                    next_m_base = gemm_m_base_q + 16'd32;
                    if (next_m_base < gemm_m_total_q) begin
                        gemm_m_base_q       <= next_m_base;
                        gemm_n_group_q      <= '0;
                        gemm_core_cfg_start <= 1'b1;
                    end else begin
                        gemm_all_tiles_done_q <= 1'b1;
                    end
                end
            end

            if (gemm_drain_valid && gemm_co_ready) begin
                gemm_co_accept_idx_q <= gemm_co_accept_idx_q + 8'd1;
                if (gemm_co_accept_idx_q[1:0] == 2'd3) begin
                    gemm_drain_word_addr_q <=
                        gemm_drain_word_addr_q + gemm_co_row_stride_words_q - ADDR_W'(3);
                end else begin
                    gemm_drain_word_addr_q <= gemm_drain_word_addr_q + ADDR_W'(1);
                end
            end
        end
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            a_rd_gemm_pipe_q <= '0;
            w_rd_gemm_pipe_q <= '0;
        end else begin
            a_rd_gemm_pipe_q <= {a_rd_gemm_pipe_q[READ_LATENCY-2:0], gemm_a_rd_req_valid};
            w_rd_gemm_pipe_q <= {w_rd_gemm_pipe_q[READ_LATENCY-2:0], gemm_w_rd_req_valid};
        end
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            meta_read_owner_q <= META_READ_NONE;
            bias_pending_q <= 1'b0;
            bias_rsp_valid_q <= 1'b0;
            bias_accumulate_q <= 1'b0;
            bias_add_bias_q <= 1'b0;
            bias_partial_data_q <= '0;
            bias_lane_mask_q <= '0;
            bias_data_q <= '0;
            bias_word_addr_q <= '0;
            bias_cache_prefetch_active_q <= 1'b0;
            bias_cache_issue_idx_q <= '0;
            bias_cache_rsp_idx_q <= '0;
            bias_cache_n_group_q <= '0;
            bias_cache_valid_q <= '0;
            old_acc_pending_q <= 1'b0;
            old_acc_rsp_valid_q <= 1'b0;
            old_acc_partial_data_q <= '0;
            old_acc_lane_mask_q <= '0;
            old_acc_data_q <= '0;
            old_acc_word_addr_q <= '0;
            co_acc_stage_valid_q <= 1'b0;
            co_acc_stage_accumulate_en_q <= 1'b0;
            co_acc_stage_add_bias_en_q <= 1'b0;
            co_acc_stage_partial_data_q <= '0;
            co_acc_stage_lane_mask_q <= '0;
            co_acc_stage_old_acc_data_q <= '0;
            co_acc_stage_bias_data_q <= '0;
            co_acc_stage_word_addr_q <= '0;
            mvout_scale_wr_ptr_q <= '0;
            mvout_scale_rd_ptr_q <= '0;
            mvout_scale_count_q <= '0;
            mvout_scale_outstanding_count_q <= '0;
        end else begin
            meta_read_owner_q <= meta_read_owner_d;

            if (gemm_cfg_start_i && gemm_cfg_ready_o) begin
                bias_cache_valid_q <= '0;
                bias_cache_issue_idx_q <= '0;
                bias_cache_n_group_q <= '0;
                bias_cache_prefetch_active_q <= co_add_bias_en_i && !co_accumulate_en_i;
            end else if (gemm_drain_bias_prefetch_fire &&
                         gemm_add_bias_q && !gemm_accumulate_q &&
                         gemm_macro_busy_q && !gemm_all_tiles_done_q) begin
                bias_cache_valid_q <= '0;
                bias_cache_issue_idx_q <= '0;
                bias_cache_n_group_q <= bias_cache_next_n_group;
                bias_cache_prefetch_active_q <= 1'b1;
            end else if (bias_cache_prefetch_fire) begin
                bias_cache_rsp_idx_q <= bias_cache_issue_idx_q;
                if (bias_cache_issue_idx_q == 2'd3) begin
                    bias_cache_prefetch_active_q <= 1'b0;
                end else begin
                    bias_cache_issue_idx_q <= bias_cache_issue_idx_q + 2'd1;
                end
            end

            if (metadata_rsp_valid && (meta_read_owner_q == META_READ_BIAS_CACHE)) begin
                bias_cache_q[bias_cache_rsp_idx_q] <= metadata_rsp_data;
                bias_cache_valid_q[bias_cache_rsp_idx_q] <= 1'b1;
            end

            if (bias_read_fire) begin
                bias_pending_q <= 1'b1;
                bias_accumulate_q <= co_src_accumulate_en;
                bias_add_bias_q <= co_src_add_bias_en;
                bias_partial_data_q <= co_src_partial_data;
                bias_lane_mask_q <= co_src_lane_mask;
                bias_word_addr_q <= co_src_word_addr;
            end

            if (metadata_rsp_valid && (meta_read_owner_q == META_READ_BIAS)) begin
                bias_pending_q <= 1'b0;
                bias_rsp_valid_q <= 1'b1;
                bias_data_q <= metadata_rsp_data;
            end

            if (bias_rsp_valid_q && co_acc_stage_load_ready) begin
                bias_rsp_valid_q <= 1'b0;
            end

            if (old_acc_read_fire) begin
                old_acc_pending_q <= 1'b1;
                old_acc_partial_data_q <= co_src_partial_data;
                old_acc_lane_mask_q <= co_src_lane_mask;
                old_acc_word_addr_q <= co_src_word_addr;
            end

            if (co_rsp_valid && old_acc_pending_q) begin
                old_acc_pending_q <= 1'b0;
                old_acc_rsp_valid_q <= 1'b1;
                old_acc_data_q <= co_rsp_data;
            end

            if (old_acc_rsp_valid_q && co_acc_stage_load_ready) begin
                old_acc_rsp_valid_q <= 1'b0;
            end

            if (co_acc_in_valid && co_acc_in_ready) begin
                co_acc_stage_valid_q <= 1'b0;
            end

            if (co_acc_stage_load_valid && co_acc_stage_load_ready) begin
                co_acc_stage_valid_q <= 1'b1;
                co_acc_stage_accumulate_en_q <= co_acc_stage_load_accumulate_en;
                co_acc_stage_add_bias_en_q <= co_acc_stage_load_add_bias_en;
                co_acc_stage_partial_data_q <= co_acc_stage_load_partial_data;
                co_acc_stage_lane_mask_q <= co_acc_stage_load_lane_mask;
                co_acc_stage_old_acc_data_q <= co_acc_stage_load_old_acc_data;
                co_acc_stage_bias_data_q <= co_acc_stage_load_bias_data;
                co_acc_stage_word_addr_q <= co_acc_stage_load_word_addr;
            end

            if (mvout_req_fire) begin
                mvout_scale_wr_ptr_q <= '0;
                mvout_scale_rd_ptr_q <= '0;
                mvout_scale_count_q <= '0;
                mvout_scale_outstanding_count_q <= '0;
            end else begin
                if (mvout_scale_rsp_push &&
                    !(mvout_scale_pop_for_rsp && (mvout_scale_count_q == '0))) begin
                    mvout_scale_q[mvout_scale_wr_ptr_q] <= metadata_rsp_data;
                    if (mvout_scale_wr_ptr_q == MVOUT_Q_PTR_W'(MVOUT_PREFETCH_DEPTH - 1)) begin
                        mvout_scale_wr_ptr_q <= '0;
                    end else begin
                        mvout_scale_wr_ptr_q <= mvout_scale_wr_ptr_q + MVOUT_Q_PTR_W'(1);
                    end
                end

                if (mvout_scale_pop_for_rsp && (mvout_scale_count_q != '0)) begin
                    if (mvout_scale_rd_ptr_q == MVOUT_Q_PTR_W'(MVOUT_PREFETCH_DEPTH - 1)) begin
                        mvout_scale_rd_ptr_q <= '0;
                    end else begin
                        mvout_scale_rd_ptr_q <= mvout_scale_rd_ptr_q + MVOUT_Q_PTR_W'(1);
                    end
                end

                unique case ({mvout_scale_rsp_push &&
                              !(mvout_scale_pop_for_rsp && (mvout_scale_count_q == '0)),
                              mvout_scale_pop_for_rsp && (mvout_scale_count_q != '0)})
                    2'b10: mvout_scale_count_q <= mvout_scale_count_q + MVOUT_Q_CNT_W'(1);
                    2'b01: mvout_scale_count_q <= mvout_scale_count_q - MVOUT_Q_CNT_W'(1);
                    default: mvout_scale_count_q <= mvout_scale_count_q;
                endcase

                unique case ({(mvout_scale_req_fire || mvout_attn_meta_req_fire), mvout_scale_rsp_push})
                    2'b10: mvout_scale_outstanding_count_q <=
                        mvout_scale_outstanding_count_q + MVOUT_Q_CNT_W'(1);
                    2'b01: mvout_scale_outstanding_count_q <=
                        mvout_scale_outstanding_count_q - MVOUT_Q_CNT_W'(1);
                    default: mvout_scale_outstanding_count_q <= mvout_scale_outstanding_count_q;
                endcase
            end
        end
    end

    co_accumulate u_co_accumulate (
        .clk_i            (clk_i),
        .rst_i            (rst_i),
        .clear_error_i    (clear_error_i),
        .accumulate_en_i  (co_acc_accumulate_en),
        .add_bias_en_i    (co_acc_add_bias_en),
        .in_valid_i       (co_acc_in_valid),
        .in_ready_o       (co_acc_in_ready),
        .partial_data_i   (co_acc_partial_data),
        .lane_mask_i      (co_acc_lane_mask),
        .old_acc_data_i   (co_acc_old_acc_data),
        .bias_data_i      (co_acc_bias_data),
        .out_valid_o      (co_acc_out_valid),
        .out_ready_i      (co_acc_out_ready),
        .out_data_o       (co_acc_out_data),
        .out_byte_en_o    (co_acc_out_byte_en),
        .error_sticky_o   (co_acc_error_sticky_o),
        .last_error_o     (co_acc_last_error_o)
    );

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            co_wr_word_addr_q <= '0;
            gemm_co_word_idx_q <= '0;
            co_acc_out_word_addr_q <= '0;
            co_write_bank_q <= 1'b0;
        end else if (co_wr_start_i) begin
            co_wr_word_addr_q <= co_wr_base_word_addr_i;
            gemm_co_word_idx_q <= '0;
            co_write_bank_q <= 1'b0;
        end else if (gemm_core_done_for_macro) begin
            co_wr_word_addr_q <= gemm_co_base_word_addr_q;
            gemm_co_word_idx_q <= '0;
            co_write_bank_q <= gemm_o_bank_q;
        end else begin
            if (co_acc_in_valid && co_acc_in_ready) begin
                co_acc_out_word_addr_q <= co_acc_stage_word_addr_q;
            end
            if (co_acc_out_valid && co_acc_out_ready) begin
                if (gemm_drain_active_q) begin
                    gemm_co_word_idx_q <= gemm_co_word_idx_q + 8'd1;
                end else begin
                    co_wr_word_addr_q <= co_wr_word_addr_q + ADDR_W'(1);
                end
            end
        end
    end

    generate
        if (O_SPLIT_BANK_HALVES) begin : gen_co_split_bank
            uram256_bank #(
                .DEPTH_WORDS       (O_PHYS_DEPTH_WORDS),
                .READ_LATENCY      (READ_LATENCY),
                .DEPTH_SLICES      (O_PHYS_URAM_DEPTH_SLICES),
                .ENABLE_BYTE_WRITE (1'b1)
            ) u_co_bank (
                .clk_i                           (clk_i),
                .rst_i                           (rst_i),
                .c_req_valid_i                   (attention_qk_o_wr_req_valid || co_acc_out_valid),
                .c_req_ready_o                   (co_acc_out_ready),
                .c_req_write_i                   (1'b1),
                .c_req_word_addr_i               (attention_qk_o_wr_req_valid
                    ? (attention_qk_o_wr_req_addr + (gemm_o_bank_i ? ADDR_W'(CO_HALF_DEPTH_WORDS) : '0))
                    : co_split_wr_addr),
                .c_req_wdata_i                   (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_data : co_acc_out_data),
                .c_req_wstrb_i                   (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_byte_en : co_acc_out_byte_en),
                .c_rsp_valid_o                   (),
                .c_rsp_rdata_o                   (),
                .d_req_valid_i                   (co_bank_d_req_valid),
                .d_req_ready_o                   (),
                .d_req_write_i                   (1'b0),
                .d_req_word_addr_i               (co_split_rd_addr),
                .d_req_wdata_i                   ('0),
                .d_req_wstrb_i                   ('0),
                .d_rsp_valid_o                   (co_rsp_valid),
                .d_rsp_rdata_o                   (co_rsp_data),
                .same_addr_read_write_conflict_o (co_bank_conflict_o)
            );
            assign attention_qk_o_wr_req_ready = co_acc_out_ready;
        end else begin : gen_co_physical_banks
            assign co0_bank_d_req_valid = co_bank_d_req_valid && !co_read_bank;
            assign co1_bank_d_req_valid = co_bank_d_req_valid && co_read_bank;
            assign co0_bank_d_req_addr  = co_bank_d_req_addr;
            assign co1_bank_d_req_addr  = co_bank_d_req_addr;
            assign co_acc_out_ready     = co_write_bank_q ? co1_acc_out_ready : co0_acc_out_ready;
            assign attention_qk_o_wr_req_ready = gemm_o_bank_i ? co1_acc_out_ready : co0_acc_out_ready;
            assign co_rsp_valid         = co0_rsp_valid || co1_rsp_valid;
            assign co_rsp_data          = co1_rsp_valid ? co1_rsp_data : co0_rsp_data;
            assign co_bank_conflict_o   = co0_bank_conflict || co1_bank_conflict;

            bram256_bank #(
                .DEPTH_WORDS       (O_BANK_DEPTH_WORDS),
                .READ_LATENCY      (READ_LATENCY),
                .ENABLE_BYTE_WRITE (1'b1)
            ) u_co0_bank (
                .clk_i                           (clk_i),
                .rst_i                           (rst_i),
                .c_req_valid_i                   ((attention_qk_o_wr_req_valid && !gemm_o_bank_i) ||
                                                  (co_acc_out_valid && !co_write_bank_q)),
                .c_req_ready_o                   (co0_acc_out_ready),
                .c_req_write_i                   (1'b1),
                .c_req_word_addr_i               (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_addr : co_bank_wr_addr),
                .c_req_wdata_i                   (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_data : co_acc_out_data),
                .c_req_wstrb_i                   (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_byte_en : co_acc_out_byte_en),
                .c_rsp_valid_o                   (),
                .c_rsp_rdata_o                   (),
                .d_req_valid_i                   (co0_bank_d_req_valid),
                .d_req_ready_o                   (),
                .d_req_write_i                   (1'b0),
                .d_req_word_addr_i               (co0_bank_d_req_addr),
                .d_req_wdata_i                   ('0),
                .d_req_wstrb_i                   ('0),
                .d_rsp_valid_o                   (co0_rsp_valid),
                .d_rsp_rdata_o                   (co0_rsp_data),
                .same_addr_read_write_conflict_o (co0_bank_conflict)
            );

            uram256_bank #(
                .DEPTH_WORDS       (O_BANK_DEPTH_WORDS),
                .READ_LATENCY      (READ_LATENCY),
                .ENABLE_BYTE_WRITE (1'b1)
            ) u_co1_bank (
                .clk_i                           (clk_i),
                .rst_i                           (rst_i),
                .c_req_valid_i                   ((attention_qk_o_wr_req_valid && gemm_o_bank_i) ||
                                                  (co_acc_out_valid && co_write_bank_q)),
                .c_req_ready_o                   (co1_acc_out_ready),
                .c_req_write_i                   (1'b1),
                .c_req_word_addr_i               (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_addr : co_bank_wr_addr),
                .c_req_wdata_i                   (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_data : co_acc_out_data),
                .c_req_wstrb_i                   (attention_qk_o_wr_req_valid ? attention_qk_o_wr_req_byte_en : co_acc_out_byte_en),
                .c_rsp_valid_o                   (),
                .c_rsp_rdata_o                   (),
                .d_req_valid_i                   (co1_bank_d_req_valid),
                .d_req_ready_o                   (),
                .d_req_write_i                   (1'b0),
                .d_req_word_addr_i               (co1_bank_d_req_addr),
                .d_req_wdata_i                   ('0),
                .d_req_wstrb_i                   ('0),
                .d_rsp_valid_o                   (co1_rsp_valid),
                .d_rsp_rdata_o                   (co1_rsp_data),
                .same_addr_read_write_conflict_o (co1_bank_conflict)
            );
        end
    endgenerate

    assign co_bank_d_req_valid = old_acc_read_fire || mvout_prefetch_fire;
    assign co_bank_d_req_addr  = old_acc_read_fire ? co_src_word_addr : mvout_prefetch_addr;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            a_load_bank_q <= 1'b0;
        end else if (a_cfg_start_i && a_cfg_ready_o) begin
            a_load_bank_q <= a_cfg_selected_bank_i;
        end
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            mvout_active_q           <= 1'b0;
            mvout_o_bank_q           <= 1'b0;
            mvout_base_word_addr_q   <= '0;
            mvout_m_count_q          <= '0;
            mvout_n_count_q          <= '0;
            mvout_row_q              <= '0;
            mvout_word_col_q         <= '0;
            mvout_prefetch_addr_q     <= '0;
            mvout_co_row_stride_words_q <= '0;
            mvout_row_bytes_q        <= '0;
            mvout_words_per_row_q    <= '0;
            mvout_in_row_q           <= '0;
            mvout_in_word_col_q      <= '0;
            mvout_in_half_q          <= 1'b0;
            mvout_out_row_q          <= '0;
            mvout_out_word_col_q     <= '0;
            mvout_out_half_q         <= 1'b0;
            mvout_dram_base_addr_q   <= '0;
            mvout_dram_row_stride_bytes_q <= '0;
            mvout_mode_q             <= NPU_MVOUT_RAW_I32;
            mvout_per_channel_scale_q <= 1'b0;
            mvout_tensor_scale_q8_24_q <= '0;
            mvout_scale_addr_q       <= '0;
            mvout_done_o             <= 1'b0;
            mvout_issue_done_q       <= 1'b0;
            mvout_resp_wr_ptr_q      <= '0;
            mvout_resp_rd_ptr_q      <= '0;
            mvout_resp_count_q       <= '0;
            mvout_outstanding_count_q <= '0;
            mvout_attn_s0_valid_q    <= 1'b0;
            mvout_attn_s1_valid_q    <= 1'b0;
            mvout_dma_valid_q        <= 1'b0;
            mvout_local_error_sticky_q <= 1'b0;
            mvout_local_last_error_q <= NPU_ERR_NONE;
        end else begin
            mvout_done_o <= 1'b0;
            if (clear_error_i) begin
                mvout_local_error_sticky_q <= 1'b0;
                mvout_local_last_error_q <= NPU_ERR_NONE;
            end

            if (mvout_internal_last_pop) begin
                mvout_active_q <= 1'b0;
                mvout_done_o <= 1'b1;
            end

            if (regular_mvout_dma_ready && mvout_dma_valid_q && !mvout_dma_load_fire) begin
                mvout_dma_valid_q <= 1'b0;
            end

            if (mvout_req_fire) begin
                if (!mvout_req_attn_legal) begin
                    mvout_active_q <= 1'b0;
                    mvout_done_o <= 1'b1;
                    mvout_local_error_sticky_q <= 1'b1;
                    mvout_local_last_error_q <= NPU_ERR_ILLEGAL_SHAPE;
                end else begin
                    mvout_active_q           <= (mvout_m_count_i != 16'd0) &&
                                                (mvout_n_count_i != 16'd0);
                end
                mvout_o_bank_q           <= mvout_o_bank_i;
                mvout_base_word_addr_q   <= (mvout_mode_i == NPU_MVOUT_ATTENTION_QK_LOGP)
                                            ? '0
                                            : mvout_req_word_addr_i;
                mvout_m_count_q          <= mvout_m_count_i;
                mvout_n_count_q          <= mvout_n_count_i;
                mvout_row_q              <= '0;
                mvout_word_col_q         <= '0;
                mvout_prefetch_addr_q     <= (mvout_mode_i == NPU_MVOUT_ATTENTION_QK_LOGP)
                                            ? '0
                                            : mvout_req_word_addr_i;
                mvout_co_row_stride_words_q <= (mvout_mode_i == NPU_MVOUT_ATTENTION_QK_LOGP)
                    ? ADDR_W'((mvout_n_count_i + 16'd31) >> 5)
                    : ADDR_W'((mvout_n_count_i + 16'd7) >> 3);
                mvout_row_bytes_q        <= (mvout_mode_i == NPU_MVOUT_ATTENTION_QK_LOGP)
                    ? {16'b0, mvout_n_count_i}
                    : ({16'b0, mvout_n_count_i} << 2);
                mvout_words_per_row_q    <= (mvout_mode_i == NPU_MVOUT_ATTENTION_QK_LOGP)
                    ? ((mvout_n_count_i + 16'd31) >> 5)
                    : ((mvout_n_count_i + 16'd7) >> 3);
                mvout_in_row_q           <= '0;
                mvout_in_word_col_q      <= '0;
                mvout_in_half_q          <= 1'b0;
                mvout_out_row_q          <= '0;
                mvout_out_word_col_q     <= '0;
                mvout_out_half_q         <= 1'b0;
                mvout_dram_base_addr_q   <= mvout_dram_base_addr_i;
                mvout_dram_row_stride_bytes_q <= mvout_dram_row_stride_bytes_i;
                mvout_mode_q             <= mvout_mode_i;
                mvout_per_channel_scale_q <= mvout_per_channel_scale_i;
                mvout_tensor_scale_q8_24_q <= mvout_tensor_scale_q8_24_i;
                mvout_scale_addr_q       <= mvout_scale_addr_i;
                mvout_issue_done_q       <= 1'b0;
                mvout_resp_wr_ptr_q      <= '0;
                mvout_resp_rd_ptr_q      <= '0;
                mvout_resp_count_q       <= '0;
                mvout_outstanding_count_q <= '0;
                mvout_attn_s0_valid_q    <= 1'b0;
                mvout_attn_s1_valid_q    <= 1'b0;
                if ((mvout_m_count_i != 16'd0) && (mvout_n_count_i != 16'd0)) begin
                    mvout_issue_done_q       <= 1'b0;
                end else begin
                    mvout_done_o             <= 1'b1;
                end
            end else begin
                if (mvout_prefetch_fire) begin
                    if (mvout_last_read_word) begin
                        mvout_issue_done_q <= 1'b1;
                    end
                    if (mvout_last_row_word) begin
                        mvout_row_q <= mvout_row_q + 16'd1;
                        mvout_word_col_q <= '0;
                        mvout_prefetch_addr_q <=
                            mvout_prefetch_addr_q + mvout_co_row_stride_words_q -
                            ADDR_W'(mvout_words_per_row_q - 16'd1);
                    end else begin
                        mvout_word_col_q <= mvout_word_col_q + 16'd1;
                        mvout_prefetch_addr_q <= mvout_prefetch_addr_q + ADDR_W'(1);
                    end
                end

                if (mvout_resp_enqueue) begin
                    mvout_resp_data_q[mvout_resp_wr_ptr_q] <= co_rsp_data;
                    mvout_resp_scale_q[mvout_resp_wr_ptr_q] <= mvout_scale_pop_for_rsp
                        ? mvout_scale_for_rsp
                        : '0;
                    if (mvout_resp_wr_ptr_q == MVOUT_Q_PTR_W'(MVOUT_PREFETCH_DEPTH - 1)) begin
                        mvout_resp_wr_ptr_q <= '0;
                    end else begin
                        mvout_resp_wr_ptr_q <= mvout_resp_wr_ptr_q + MVOUT_Q_PTR_W'(1);
                    end
                end

                if (mvout_resp_dequeue) begin
                    if (mvout_resp_rd_ptr_q == MVOUT_Q_PTR_W'(MVOUT_PREFETCH_DEPTH - 1)) begin
                        mvout_resp_rd_ptr_q <= '0;
                    end else begin
                        mvout_resp_rd_ptr_q <= mvout_resp_rd_ptr_q + MVOUT_Q_PTR_W'(1);
                    end
                end

                if (mvout_convert_input_fire) begin
                    if (mvout_input_word_done) begin
                        mvout_in_half_q <= 1'b0;
                        if ((mvout_in_word_col_q + 16'd1) >= mvout_words_per_row_q) begin
                            mvout_in_word_col_q <= '0;
                            if ((mvout_in_row_q + 16'd1) < mvout_m_count_q) begin
                                mvout_in_row_q <= mvout_in_row_q + 16'd1;
                            end
                        end else begin
                            mvout_in_word_col_q <= mvout_in_word_col_q + 16'd1;
                        end
                    end else begin
                        mvout_in_half_q <= 1'b1;
                    end
                end

                unique case ({mvout_resp_enqueue, mvout_resp_dequeue})
                    2'b10: mvout_resp_count_q <= mvout_resp_count_q + MVOUT_Q_CNT_W'(1);
                    2'b01: mvout_resp_count_q <= mvout_resp_count_q - MVOUT_Q_CNT_W'(1);
                    default: mvout_resp_count_q <= mvout_resp_count_q;
                endcase

                unique case ({mvout_prefetch_fire, mvout_resp_enqueue})
                    2'b10: mvout_outstanding_count_q <=
                        mvout_outstanding_count_q + MVOUT_Q_CNT_W'(1);
                    2'b01: mvout_outstanding_count_q <=
                        mvout_outstanding_count_q - MVOUT_Q_CNT_W'(1);
                    default: mvout_outstanding_count_q <= mvout_outstanding_count_q;
                endcase

                if (mvout_attn_s1_to_dma_fire) begin
                    mvout_dma_valid_q <= 1'b1;
                    mvout_dma_data_q  <= mvout_attn_s1_data_q;
                    mvout_dma_keep_q  <= mvout_attn_s1_keep_q;
                    mvout_dma_addr_q  <= mvout_attn_s1_addr_q;
                    mvout_dma_last_q  <= mvout_attn_s1_last_q;
                end else if (!mvout_attn_mode && mvout_convert_pop && mvout_output_half_valid) begin
                    mvout_dma_valid_q <= 1'b1;
                    mvout_dma_data_q  <= mvout_convert_dma_data;
                    mvout_dma_keep_q  <= mvout_current_keep;
                    mvout_dma_addr_q  <= mvout_dram_base_addr_q +
                        ({48'b0, mvout_out_row_q} * {32'b0, mvout_dram_row_stride_bytes_q}) +
                        {32'b0, mvout_out_half_col_byte};
                    mvout_dma_last_q  <= mvout_output_last_valid_half;
                end

                if (mvout_attn_s1_ready) begin
                    mvout_attn_s1_valid_q <= mvout_attn_s0_valid_q;
                    if (mvout_attn_s0_valid_q) begin
                        mvout_attn_s1_data_q <= mvout_attn_s1_data_comb;
                        mvout_attn_s1_keep_q <= mvout_attn_s0_keep_q;
                        mvout_attn_s1_addr_q <= mvout_attn_s0_addr_q;
                        mvout_attn_s1_last_q <= mvout_attn_s0_last_q;
                    end
                end

                if (mvout_attn_s0_ready) begin
                    mvout_attn_s0_valid_q <= mvout_attn_s0_fire;
                    if (mvout_attn_s0_fire) begin
                        mvout_attn_s0_d_word_q <= mvout_resp_data_q[mvout_resp_rd_ptr_q];
                        mvout_attn_s0_correction_q <= mvout_attn_correction;
                        mvout_attn_s0_global_row_q <= mvout_attn_global_row;
                        mvout_attn_s0_col_base_q <= mvout_attn_global_col_base;
                        mvout_attn_s0_half_q <= mvout_out_half_q;
                        mvout_attn_s0_keep_q <= mvout_current_keep;
                        mvout_attn_s0_addr_q <= mvout_dram_base_addr_q +
                            ({48'b0, mvout_out_row_q} * {32'b0, mvout_dram_row_stride_bytes_q}) +
                            {32'b0, mvout_out_half_col_byte};
                        mvout_attn_s0_last_q <= mvout_output_last_valid_half;
                    end
                end

                if (mvout_convert_pop) begin
                    if (mvout_output_word_done) begin
                        mvout_out_half_q <= 1'b0;
                        if ((mvout_out_word_col_q + 16'd1) >= mvout_words_per_row_q) begin
                            mvout_out_word_col_q <= '0;
                            if ((mvout_out_row_q + 16'd1) < mvout_m_count_q) begin
                                mvout_out_row_q <= mvout_out_row_q + 16'd1;
                            end
                        end else begin
                            mvout_out_word_col_q <= mvout_out_word_col_q + 16'd1;
                        end
                    end else begin
                        mvout_out_half_q <= 1'b1;
                    end
                end
            end
        end
    end

    co_output_convert #(
        .FIFO_DEPTH (FIFO_DEPTH)
    ) u_output_convert (
        .clk_i                (clk_i),
        .rst_i                (rst_i),
        .clear_error_i        (clear_error_i),
        .mvout_mode_i         (mvout_mode_q),
        .per_channel_scale_i  (mvout_per_channel_scale_q),
        .tensor_scale_q8_24_i (mvout_tensor_scale_q8_24_q),
        .co_valid_i           (mvout_co_valid && !mvout_attn_mode),
        .co_ready_o           (mvout_convert_co_ready),
        .co_acc_data_i        (mvout_co_data),
        .scale_data_i         (mvout_co_scale_data),
        .dma_valid_o          (mvout_convert_dma_valid),
        .dma_ready_i          (mvout_convert_dma_ready),
        .dma_data_o           (mvout_convert_dma_data),
        .fifo_almost_full_o   (mvout_fifo_almost_full_o),
        .error_sticky_o       (mvout_convert_error_sticky),
        .last_error_o         (mvout_convert_last_error)
    );

endmodule

`default_nettype wire
