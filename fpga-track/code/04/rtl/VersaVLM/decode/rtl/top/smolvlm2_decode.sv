`ifndef SMOLVLM2_DECODE_SV
`define SMOLVLM2_DECODE_SV

module smolvlm2_decode #(
    parameter int RF_DATA_WIDTH      = 64,
    parameter int SPM_SIZE           = 1 << 19,
    parameter int SPM_DATA_WIDTH     = 512,
    parameter int GEMV_ACC_NUM       = 4,
    parameter int TILE_ELEMS         = 128,
    parameter int ROW_TILE_ELEMS     = 32,
    parameter int FIXED_FRAC         = 12,
    parameter int GEMV_STREAM_BUFFER_DEPTH = 64,
    parameter int STREAM_BUFFER_DEPTH = 4096,
    parameter int ROPE_HEAD_DIM      = 64,
    parameter int ROPE_MAX_SEQ_LEN   = 4096,
    parameter int DISABLE_DECODE_ROPE = 0,
    parameter int DISABLE_DECODE_SOFTMAX = 0,
    parameter int DISABLE_DECODE_KV_QUANT = 0,
    parameter int KQ_SCALE_POW2_SHIFT = 3,

    localparam int SPM_ADDR_WIDTH = $clog2(SPM_SIZE)
) (
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_mat_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_vec_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_scale_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_scale2_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_group_stride_bytes,
    input  logic [7:0]                        matvec_act_frac_cfg,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_mat_width,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_mat_height,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_output_addr,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_scale_addr,
    input  logic [15:0]                       matvec_cache_cell_idx,
    input  logic [1:0]                        gemv_mode,
    input  logic [63:0]                       cfg_decode_flow,
    input  logic                              weight_stream_mode,
    input  logic                              act_scale_enable,
    input  logic                              act_scale2_enable,
    input  logic                              matvec_req_en,
    output logic                              matvec_busy,
    output logic                              matvec_comp_done,
    output logic                              kv_scale_commit_valid_o,
    output logic                              kv_scale_commit_is_v_o,
    output logic [4:0][15:0]                 kv_scale_commit_values_o,
    output logic [7:0]                       kv_scale_commit_count_o,

    output logic                              spm_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         spm_rd_data,

    output logic                              scale_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         scale_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         scale_rd_data,

    output logic                              act_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         act_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         act_rd_data,

    input  logic                              weight_stream_valid,
    input  logic [SPM_DATA_WIDTH-1:0]         weight_stream_data,
    input  logic                              weight_stream_last,
    output logic                              weight_stream_ready,

    output logic                              spm_wr_en,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_wr_addr,
    output logic [SPM_DATA_WIDTH-1:0]         spm_wr_data,
    output logic [SPM_DATA_WIDTH/8-1:0]       spm_wr_mask,

    output logic                              act_wr_en,
    output logic [SPM_ADDR_WIDTH-1:0]         act_wr_addr,
    output logic [SPM_DATA_WIDTH-1:0]         act_wr_data,
    output logic [SPM_DATA_WIDTH/8-1:0]       act_wr_mask,

    input  logic                              preload_acc_en_i,
    input  logic [((GEMV_ACC_NUM <= 1) ? 1 : $clog2(GEMV_ACC_NUM))-1:0] preload_acc_id_i,
    input  logic [((ROW_TILE_ELEMS <= 1) ? 1 : $clog2(ROW_TILE_ELEMS))-1:0] preload_acc_row_i,
    input  logic [31:0]                       preload_acc_data_i
);

    localparam int USER_WIDTH = RF_DATA_WIDTH / 4;
    localparam int W8_TILE_ELEMS = TILE_ELEMS / 2;
    localparam int BUFFER_COUNT_WIDTH = $clog2(STREAM_BUFFER_DEPTH + 1);
    localparam int GEMV_BUFFER_COUNT_WIDTH = $clog2(GEMV_STREAM_BUFFER_DEPTH + 1);
    localparam int GEMV_BUFFER_RESERVE = ROW_TILE_ELEMS + 4;
    localparam int GEMV_BUFFER_READY_LEVEL =
        (GEMV_STREAM_BUFFER_DEPTH > GEMV_BUFFER_RESERVE) ?
        (GEMV_STREAM_BUFFER_DEPTH - GEMV_BUFFER_RESERVE) :
        (GEMV_STREAM_BUFFER_DEPTH - 1);

    logic                       raw_valid_w;
    logic [15:0]                raw_data_w;
    logic [USER_WIDTH-1:0]      raw_row_w;
    logic                       gemv_buffer_ready_w;
    logic                       gemv_buffer_push_ready_w;
    logic                       gemv_buffer_pop_en_w;
    logic                       gemv_buffer_pop_valid_w;
    logic [15:0]                gemv_buffer_pop_data_w;
    logic [USER_WIDTH-1:0]      gemv_buffer_pop_user_w;
    logic [GEMV_BUFFER_COUNT_WIDTH-1:0] gemv_buffer_count_w;
    logic [GEMV_BUFFER_COUNT_WIDTH-1:0] gemv_buffer_push_count_w;
    logic [GEMV_BUFFER_COUNT_WIDTH-1:0] gemv_buffer_pop_count_w;
    logic                       gemv_buffer_empty_w;
    logic                       gemv_buffer_full_w;
    logic                       gemv_buffer_overflow_w;
    logic                       gemv_buffer_underflow_w;
    logic                       post_gemv_pop_w;
    logic                       gemv_raw_ready_w;
    logic                       post_out_valid_w;
    logic [15:0]                post_out_data_w;
    logic [USER_WIDTH-1:0]      post_out_row_w;
    logic                       post_valid_w;
    logic [15:0]                post_data_w;
    logic [USER_WIDTH-1:0]      post_row_w;
    logic                       core_req_en_w;
    logic                       writer_req_en_w;
    logic                       stream_writer_req_en_w;
    logic                       kv_writer_req_en_w;
    logic                       core_busy_w;
    logic                       core_done_w;
    logic                       gemv_act_rd_en_w;
    logic [SPM_ADDR_WIDTH-1:0]  gemv_act_rd_addr_w;
    logic [SPM_DATA_WIDTH-1:0]  gemv_act_rd_data_w;
    logic                       post_req_en_w;
    logic                       post_busy_w;
    logic                       post_done_w;
    logic                       post_error_w;
    logic                       rope_lut_rd_en_w;
    logic [SPM_ADDR_WIDTH-1:0]  rope_lut_rd_addr_w;
    logic                       writer_busy_w;
    logic                       writer_done_w;
    logic                       stream_writer_busy_w;
    logic                       stream_writer_done_w;
    logic                       kv_writer_busy_w;
    logic                       kv_writer_done_w;
    logic                       sfu_silu_en_w;
    logic                       writer_dst_act_w;
    logic                       flow_has_work_w;
    logic [BUFFER_COUNT_WIDTH-1:0] buffer_count_w;
    logic [BUFFER_COUNT_WIDTH-1:0] buffer_push_count_w;
    logic [BUFFER_COUNT_WIDTH-1:0] buffer_pop_count_w;
    logic                       buffer_overflow_w;
    logic                       buffer_underflow_w;
    logic                       mul_pair_miss_w;
    logic [USER_WIDTH:0]        mul_pair_count_w;
    logic [USER_WIDTH:0]        mul_output_count_w;
    logic                       stream_writer_mode_w;
    logic                       stream_writer_group_stride_w;
    logic                       writer_valid_w;
    logic [15:0]                writer_data_w;
    logic [31:0]                writer_data32_w;
    logic [USER_WIDTH-1:0]      writer_row_w;
    logic                       writer_valid_q;
    logic [15:0]                writer_data_q;
    logic [31:0]                writer_data32_q;
    logic [USER_WIDTH-1:0]      writer_row_q;
    logic                       flow_error_w;
    logic                       flow_busy_w;
    logic                       local_req_q;
    logic [RF_DATA_WIDTH/2-1:0] matvec_input_mat_addr_q;
    logic [RF_DATA_WIDTH/2-1:0] matvec_input_vec_addr_q;
    logic [RF_DATA_WIDTH/2-1:0] matvec_input_act_scale_addr_q;
    logic [RF_DATA_WIDTH/2-1:0] matvec_input_act_scale2_addr_q;
    logic [RF_DATA_WIDTH/2-1:0] matvec_input_act_group_stride_bytes_q;
    logic [RF_DATA_WIDTH/4-1:0] matvec_input_mat_width_q;
    logic [RF_DATA_WIDTH/4-1:0] matvec_input_mat_height_q;
    logic [RF_DATA_WIDTH/2-1:0] matvec_output_addr_q;
    logic [RF_DATA_WIDTH/4-1:0] matvec_input_scale_addr_q;
    logic [15:0]                matvec_cache_cell_idx_q;
    logic [15:0]                matvec_cache_cell_idx_w;
    logic [7:0]                 matvec_act_frac_cfg_q;
    logic [1:0]                 gemv_mode_q;
    logic                       weight_stream_mode_q;
    logic                       act_scale_enable_q;
    logic                       act_scale2_enable_q;
    logic [63:0]                cfg_decode_flow_q;
    logic                       act_buffer_dirty_q;
    logic                       delay_rope_flow_w;
    logic [3:0]                 cfg_src0_w;
    logic [3:0]                 cfg_src1_w;
    logic [1:0]                 cfg_unary_op_w;
    logic                       cfg_binary_op_w;
    logic [1:0]                 cfg_reduce_op_w;
    logic [2:0]                 cfg_dst_w;
    logic [USER_WIDTH-1:0]      cfg_elem_count_w;
    logic [1:0]                 cfg_group_count_m1_w;
    logic [USER_WIDTH-1:0]      cfg_group_count_w;
    logic [USER_WIDTH-1:0]      cfg_total_elem_count_w;
    logic [USER_WIDTH-1:0]      cfg_group_stride_elem_count_w;
    logic [15:0]                cfg_position_w;
    logic                       cfg_kv_quant_en_w;
    logic                       cfg_kv_col_scale_en_w;
    logic                       cfg_kv_v_separated_w;
    logic                       cfg_unit_weight_scale_w;
    logic                       cfg_kv_quant_scratch_w;
    logic                       cfg_kv_quant_active_w;
    logic                       kv_scale_valid_w;
    logic [15:0]                kv_scale_fp16_w;
    logic [USER_WIDTH-1:0]      kv_scale_user_w;
    logic [4:0][15:0]           kv_scale_capture_q;
    logic [4:0]                 kv_scale_capture_seen_q;
    logic [7:0]                 kv_scale_capture_count_q;
    logic                       kv_scale_capture_active_q;
    logic                       kv_scale_capture_is_v_q;
    logic                       kv_scale_capture_fire_w;
    logic                       kv_scale_commit_pending_q;
    logic [2:0]                 kv_scale_capture_head_w;
    logic                       kv_scale_capture_head_valid_w;
    logic [7:0]                 kv_scale_capture_seen_count_w;
    logic                       flow_done_w;
    logic                       kv_quant_valid_w;
    logic signed [7:0]          kv_quant_i8_w;
    logic [USER_WIDTH-1:0]      kv_quant_user_w;
    logic                       stream_spm_wr_en_w;
    logic [SPM_ADDR_WIDTH-1:0]  stream_spm_wr_addr_w;
    logic [SPM_DATA_WIDTH-1:0]  stream_spm_wr_data_w;
    logic [SPM_DATA_WIDTH/8-1:0] stream_spm_wr_mask_w;
    logic                       stream_act_wr_en_w;
    logic [SPM_ADDR_WIDTH-1:0]  stream_act_wr_addr_w;
    logic [SPM_DATA_WIDTH-1:0]  stream_act_wr_data_w;
    logic [SPM_DATA_WIDTH/8-1:0] stream_act_wr_mask_w;
    logic                       kv_spm_wr_en_w;
    logic [SPM_ADDR_WIDTH-1:0]  kv_spm_wr_addr_w;
    logic [SPM_DATA_WIDTH-1:0]  kv_spm_wr_data_w;
    logic [SPM_DATA_WIDTH/8-1:0] kv_spm_wr_mask_w;

    localparam logic [3:0] SRC_GEMV_STREAM = 4'd1;
    localparam logic [1:0] UNARY_ROPE      = 2'd2;
    localparam logic [2:0] DST_OUTPUT_SPM  = 3'd1;
    localparam logic [2:0] DST_ACT_BUFFER  = 3'd2;
    localparam int CACHE_INVALIDATE_BIT = (RF_DATA_WIDTH / 4) - 1;

    function automatic logic [7:0] popcount5(input logic [4:0] value);
        begin
            popcount5 = {7'd0, value[0]} + {7'd0, value[1]} +
                        {7'd0, value[2]} + {7'd0, value[3]} +
                        {7'd0, value[4]};
        end
    endfunction

    assign matvec_busy = matvec_req_en || local_req_q || flow_busy_w;
    assign matvec_cache_cell_idx_w =
        matvec_cache_cell_idx_q |
        (act_buffer_dirty_q ? (16'(1) << CACHE_INVALIDATE_BIT) : 16'd0);

    assign cfg_src0_w = cfg_decode_flow_q[3:0];
    assign cfg_src1_w = cfg_decode_flow_q[7:4];
    assign cfg_unary_op_w = cfg_decode_flow_q[9:8];
    assign cfg_binary_op_w = cfg_decode_flow_q[10];
    assign cfg_reduce_op_w = cfg_decode_flow_q[12:11];
    assign cfg_dst_w = cfg_decode_flow_q[15:13];
    assign cfg_kv_quant_en_w = cfg_decode_flow_q[24];
    assign cfg_group_count_m1_w = cfg_decode_flow_q[26:25];
    assign cfg_kv_col_scale_en_w = cfg_decode_flow_q[27];
    assign cfg_kv_v_separated_w = cfg_decode_flow_q[28];
    assign cfg_unit_weight_scale_w = cfg_decode_flow_q[29];
    assign cfg_kv_quant_scratch_w = cfg_decode_flow_q[30];
    assign cfg_group_count_w = USER_WIDTH'(cfg_group_count_m1_w) + USER_WIDTH'(1);
    assign cfg_kv_quant_active_w = cfg_kv_quant_en_w && (DISABLE_DECODE_KV_QUANT == 0);
    assign flow_has_work_w = ((cfg_src0_w == SRC_GEMV_STREAM) ||
                              (cfg_src1_w == SRC_GEMV_STREAM)) ?
                             ((matvec_input_mat_width_q != '0) &&
                              (matvec_input_mat_height_q != '0)) :
                             (cfg_decode_flow_q[47:32] != '0);
    assign cfg_elem_count_w = cfg_decode_flow_q[47:32];
    assign cfg_total_elem_count_w = cfg_elem_count_w * cfg_group_count_w;
    assign cfg_group_stride_elem_count_w =
        (cfg_elem_count_w + USER_WIDTH'(W8_TILE_ELEMS - 1)) &
        ~USER_WIDTH'(W8_TILE_ELEMS - 1);
    assign cfg_position_w = cfg_decode_flow_q[63:48];
    assign delay_rope_flow_w = ((cfg_src0_w == SRC_GEMV_STREAM) ||
                                (cfg_src1_w == SRC_GEMV_STREAM)) &&
                               (cfg_unary_op_w == UNARY_ROPE);
    assign gemv_buffer_pop_en_w = post_gemv_pop_w;
    assign gemv_buffer_ready_w = gemv_buffer_count_w <=
                                 GEMV_BUFFER_COUNT_WIDTH'(GEMV_BUFFER_READY_LEVEL);
    assign gemv_raw_ready_w = gemv_buffer_push_ready_w &&
                              gemv_buffer_ready_w &&
                              !gemv_buffer_full_w;
    assign stream_writer_mode_w = !cfg_kv_quant_active_w &&
                                  ((cfg_dst_w == DST_OUTPUT_SPM) ||
                                   (cfg_dst_w == DST_ACT_BUFFER));
    assign stream_writer_group_stride_w =
        (cfg_reduce_op_w == 2'd1) &&
        (cfg_dst_w == DST_ACT_BUFFER) &&
        (cfg_group_count_m1_w != 2'd0);
    assign stream_writer_req_en_w = writer_req_en_w && stream_writer_mode_w;
    assign kv_writer_req_en_w = writer_req_en_w && cfg_kv_quant_active_w;
    assign writer_busy_w = stream_writer_busy_w || kv_writer_busy_w;
    assign writer_done_w = stream_writer_done_w || kv_writer_done_w;
    assign kv_scale_capture_fire_w = kv_scale_valid_w && kv_scale_capture_active_q;
    assign kv_scale_capture_head_w = kv_scale_user_w[8:6];
    assign kv_scale_capture_head_valid_w = kv_scale_capture_head_w < 3'd5;
    assign kv_scale_capture_seen_count_w = popcount5(kv_scale_capture_seen_q);
    assign matvec_comp_done = flow_done_w;

    assign spm_wr_en   = stream_spm_wr_en_w || kv_spm_wr_en_w;
    assign spm_wr_addr = kv_spm_wr_en_w ? kv_spm_wr_addr_w : stream_spm_wr_addr_w;
    assign spm_wr_data = kv_spm_wr_en_w ? kv_spm_wr_data_w : stream_spm_wr_data_w;
    assign spm_wr_mask = kv_spm_wr_en_w ? kv_spm_wr_mask_w : stream_spm_wr_mask_w;
    assign act_wr_en   = stream_act_wr_en_w;
    assign act_wr_addr = stream_act_wr_addr_w;
    assign act_wr_data = stream_act_wr_data_w;
    assign act_wr_mask = stream_act_wr_mask_w;
    assign act_rd_en = rope_lut_rd_en_w ? 1'b1 : gemv_act_rd_en_w;
    assign act_rd_addr = rope_lut_rd_en_w ? rope_lut_rd_addr_w :
                         gemv_act_rd_addr_w;
    assign gemv_act_rd_data_w = act_rd_data;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_req_q <= 1'b0;
            matvec_input_mat_addr_q <= '0;
            matvec_input_vec_addr_q <= '0;
            matvec_input_act_scale_addr_q <= '0;
            matvec_input_act_scale2_addr_q <= '0;
            matvec_input_act_group_stride_bytes_q <= '0;
            matvec_input_mat_width_q <= '0;
            matvec_input_mat_height_q <= '0;
            matvec_output_addr_q <= '0;
            matvec_input_scale_addr_q <= '0;
            matvec_cache_cell_idx_q <= '0;
            matvec_act_frac_cfg_q <= '0;
            gemv_mode_q <= '0;
            weight_stream_mode_q <= 1'b0;
            act_scale_enable_q <= 1'b0;
            act_scale2_enable_q <= 1'b0;
            cfg_decode_flow_q <= '0;
            act_buffer_dirty_q <= 1'b0;
            kv_scale_capture_q <= '0;
            kv_scale_capture_seen_q <= '0;
            kv_scale_capture_count_q <= '0;
            kv_scale_capture_active_q <= 1'b0;
            kv_scale_capture_is_v_q <= 1'b0;
            kv_scale_commit_pending_q <= 1'b0;
            kv_scale_commit_valid_o <= 1'b0;
            kv_scale_commit_is_v_o <= 1'b0;
            kv_scale_commit_values_o <= '0;
            kv_scale_commit_count_o <= '0;
        end else begin
            local_req_q <= 1'b0;
            kv_scale_commit_valid_o <= 1'b0;
            if (matvec_req_en && !local_req_q && !flow_busy_w) begin
                local_req_q <= 1'b1;
                matvec_input_mat_addr_q <= matvec_input_mat_addr;
                matvec_input_vec_addr_q <= matvec_input_vec_addr;
                matvec_input_act_scale_addr_q <= matvec_input_act_scale_addr;
                matvec_input_act_scale2_addr_q <= matvec_input_act_scale2_addr;
                matvec_input_act_group_stride_bytes_q <= matvec_input_act_group_stride_bytes;
                matvec_input_mat_width_q <= matvec_input_mat_width;
                matvec_input_mat_height_q <= matvec_input_mat_height;
                matvec_output_addr_q <= matvec_output_addr;
                matvec_input_scale_addr_q <= matvec_input_scale_addr;
                matvec_cache_cell_idx_q <= matvec_cache_cell_idx;
                matvec_act_frac_cfg_q <= matvec_act_frac_cfg;
                gemv_mode_q <= gemv_mode;
                weight_stream_mode_q <= weight_stream_mode;
                act_scale_enable_q <= act_scale_enable;
                act_scale2_enable_q <= act_scale2_enable;
                cfg_decode_flow_q <= cfg_decode_flow;
                kv_scale_capture_q <= '0;
                kv_scale_capture_seen_q <= '0;
                kv_scale_capture_count_q <= '0;
                kv_scale_commit_pending_q <= 1'b0;
                kv_scale_capture_active_q <= cfg_decode_flow[24] &&
                                             (DISABLE_DECODE_KV_QUANT == 0) &&
                                             (cfg_decode_flow[15:13] == DST_OUTPUT_SPM);
                kv_scale_capture_is_v_q <= cfg_decode_flow[28];
            end else if (kv_scale_capture_fire_w && kv_scale_capture_head_valid_w) begin
                kv_scale_capture_q[kv_scale_capture_head_w] <= kv_scale_fp16_w;
                kv_scale_capture_seen_q[kv_scale_capture_head_w] <= 1'b1;
                kv_scale_capture_count_q <= popcount5(kv_scale_capture_seen_q |
                                                      (5'b00001 << kv_scale_capture_head_w));
            end
            if (flow_done_w && kv_scale_capture_active_q) begin
                kv_scale_commit_pending_q <= 1'b1;
                kv_scale_capture_active_q <= 1'b0;
            end
            if (kv_scale_commit_pending_q) begin
                kv_scale_commit_valid_o <= 1'b1;
                kv_scale_commit_is_v_o <= kv_scale_capture_is_v_q;
                kv_scale_commit_values_o <= kv_scale_capture_q;
                kv_scale_commit_count_o <= kv_scale_capture_seen_count_w;
                kv_scale_commit_pending_q <= 1'b0;
            end
            if (stream_act_wr_en_w) begin
                act_buffer_dirty_q <= 1'b1;
            end else if (core_req_en_w && act_buffer_dirty_q) begin
                act_buffer_dirty_q <= 1'b0;
            end
        end
    end

    decode_flow_scheduler #(
        .USER_WIDTH          (USER_WIDTH),
        .STREAM_BUFFER_DEPTH (STREAM_BUFFER_DEPTH),
        .ROPE_HEAD_DIM       (ROPE_HEAD_DIM),
        .ROPE_MAX_SEQ_LEN    (ROPE_MAX_SEQ_LEN),
        .DISABLE_DECODE_ROPE (DISABLE_DECODE_ROPE),
        .DISABLE_DECODE_SOFTMAX (DISABLE_DECODE_SOFTMAX),
        .DISABLE_DECODE_KV_QUANT (DISABLE_DECODE_KV_QUANT)
    ) u_flow_ctrl (
        .clk                  (clk),
        .rst_n                (rst_n),
        .cfg_decode_flow_i    (cfg_decode_flow_q),
        .flow_req_en_i        (local_req_q),
        .flow_has_work_i      (flow_has_work_w),
        .gemv_busy_i          (core_busy_w),
        .gemv_done_i          (core_done_w),
        .post_busy_i          (post_busy_w),
        .post_done_i          (post_done_w),
        .post_error_i         (post_error_w),
        .writer_busy_i        (writer_busy_w),
        .writer_done_i        (writer_done_w),
        .buffer_count_i       (buffer_count_w),
        .buffer_push_count_i  (buffer_push_count_w),
        .buffer_pop_count_i   (buffer_pop_count_w),
        .buffer_overflow_i    (buffer_overflow_w),
        .buffer_underflow_i   (buffer_underflow_w),
        .gemv_buffer_push_count_i (BUFFER_COUNT_WIDTH'(gemv_buffer_push_count_w)),
        .gemv_buffer_pop_count_i (BUFFER_COUNT_WIDTH'(gemv_buffer_pop_count_w)),
        .gemv_buffer_overflow_i  (gemv_buffer_overflow_w),
        .gemv_buffer_underflow_i (gemv_buffer_underflow_w),
        .mul_pair_count_i     (mul_pair_count_w),
        .mul_output_count_i   (mul_output_count_w),
        .mul_pair_miss_i      (mul_pair_miss_w),
        .gemv_req_en_o        (core_req_en_w),
        .post_req_en_o        (post_req_en_w),
        .writer_req_en_o      (writer_req_en_w),
        .sfu_silu_en_o        (sfu_silu_en_w),
        .writer_dst_act_o     (writer_dst_act_w),
        .buffer_clear_o       (),
        .buffer_stats_clear_o (),
        .buffer_push_en_o     (),
        .mul_enable_o         (),
        .writer_sel_mul_o     (),
        .flow_busy_o          (flow_busy_w),
        .flow_done_o          (flow_done_w),
        .flow_error_o         (flow_error_w)
    );

    gemv_compute_top #(
        .RF_DATA_WIDTH  (RF_DATA_WIDTH),
        .SPM_SIZE       (SPM_SIZE),
        .SPM_DATA_WIDTH (SPM_DATA_WIDTH),
        .GEMV_ACC_NUM   (GEMV_ACC_NUM),
        .TILE_ELEMS     (TILE_ELEMS),
        .ROW_TILE_ELEMS (ROW_TILE_ELEMS),
        .FIXED_FRAC     (FIXED_FRAC)
    ) u_gemv_core (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .matvec_input_mat_addr   (matvec_input_mat_addr_q),
        .matvec_input_vec_addr   (matvec_input_vec_addr_q),
        .matvec_input_act_scale_addr (matvec_input_act_scale_addr_q),
        .matvec_input_act_scale2_addr (matvec_input_act_scale2_addr_q),
        .matvec_input_act_group_stride_bytes (matvec_input_act_group_stride_bytes_q),
        .matvec_input_mat_width  (matvec_input_mat_width_q),
        .matvec_input_mat_height (matvec_input_mat_height_q),
        .matvec_input_scale_addr (matvec_input_scale_addr_q),
        .matvec_cache_cell_idx   (matvec_cache_cell_idx_w),
        .act_frac_cfg_i          (matvec_act_frac_cfg_q),
        .gemv_mode_i             (gemv_mode_q),
        .gemv_group_count_m1_i   (cfg_group_count_m1_w),
        .kv_col_scale_en_i       (cfg_kv_col_scale_en_w),
        .unit_weight_scale_i      (cfg_unit_weight_scale_w),
        .matvec_req_en           (core_req_en_w),
        .matvec_busy             (core_busy_w),
        .matvec_comp_done        (core_done_w),
        .spm_rd_en               (spm_rd_en),
        .spm_rd_addr             (spm_rd_addr),
        .spm_rd_data             (spm_rd_data),
        .scale_rd_en             (scale_rd_en),
        .scale_rd_addr           (scale_rd_addr),
        .scale_rd_data           (scale_rd_data),
        .act_rd_en               (gemv_act_rd_en_w),
        .act_rd_addr             (gemv_act_rd_addr_w),
        .act_rd_data             (gemv_act_rd_data_w),
        .weight_stream_mode_i    (weight_stream_mode_q),
        .act_scale_enable_i      (act_scale_enable_q),
        .act_scale2_enable_i     (act_scale2_enable_q),
        .weight_stream_valid_i   (weight_stream_valid),
        .weight_stream_data_i    (weight_stream_data),
        .weight_stream_last_i    (weight_stream_last),
        .weight_stream_ready_o   (weight_stream_ready),
        .raw_valid_o             (raw_valid_w),
        .raw_data_o              (raw_data_w),
        .raw_row_o               (raw_row_w),
        .raw_ready_i             (gemv_raw_ready_w),
        .preload_acc_en_i        (preload_acc_en_i),
        .preload_acc_id_i        (preload_acc_id_i),
        .preload_acc_row_i       (preload_acc_row_i),
        .preload_acc_data_i      (preload_acc_data_i)
    );

    stream_buffer #(
        .DATA_WIDTH (16),
        .USER_WIDTH (USER_WIDTH),
        .DEPTH      (GEMV_STREAM_BUFFER_DEPTH),
        .FIFO_MEMORY_TYPE ("distributed")
    ) u_gemv_stream_buffer (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear_i         (core_req_en_w),
        .stats_clear_i   (core_req_en_w),
        .push_valid_i    (raw_valid_w),
        .push_data_i     (raw_data_w),
        .push_user_i     (raw_row_w),
        .push_ready_o    (gemv_buffer_push_ready_w),
        .pop_en_i        (gemv_buffer_pop_en_w),
        .pop_valid_o     (gemv_buffer_pop_valid_w),
        .pop_data_o      (gemv_buffer_pop_data_w),
        .pop_user_o      (gemv_buffer_pop_user_w),
        .count_o         (gemv_buffer_count_w),
        .empty_o         (gemv_buffer_empty_w),
        .full_o          (gemv_buffer_full_w),
        .overflow_o      (gemv_buffer_overflow_w),
        .underflow_o     (gemv_buffer_underflow_w),
        .push_count_o    (gemv_buffer_push_count_w),
        .pop_count_o     (gemv_buffer_pop_count_w)
    );

    postprocess_top #(
        .USER_WIDTH        (USER_WIDTH),
        .POST_BUFFER_DEPTH (STREAM_BUFFER_DEPTH),
        .SOFTMAX_MAX_LEN   (STREAM_BUFFER_DEPTH),
        .ROPE_HEAD_DIM     (ROPE_HEAD_DIM),
        .ROPE_MAX_SEQ_LEN  (ROPE_MAX_SEQ_LEN),
        .SPM_DATA_WIDTH    (SPM_DATA_WIDTH),
        .SPM_ADDR_WIDTH    (SPM_ADDR_WIDTH),
        .ROPE_LUT_BASE_ADDR(32'h0000_3f00),
        .DISABLE_DECODE_ROPE (DISABLE_DECODE_ROPE),
        .DISABLE_DECODE_SOFTMAX (DISABLE_DECODE_SOFTMAX),
        .DISABLE_DECODE_KV_QUANT (DISABLE_DECODE_KV_QUANT),
        .SCORE_SCALE_POW2_SHIFT (KQ_SCALE_POW2_SHIFT)
    ) u_postprocess_top (
        .clk                  (clk),
        .rst_n                (rst_n),
        .cfg_src0_i           (cfg_src0_w),
        .cfg_src1_i           (cfg_src1_w),
        .cfg_unary_op_i       (cfg_unary_op_w),
        .cfg_binary_op_i      (cfg_binary_op_w),
        .cfg_reduce_op_i      (cfg_reduce_op_w),
        .cfg_dst_i            (cfg_dst_w),
        .cfg_kv_quant_en_i    (cfg_kv_quant_active_w),
        .cfg_group_count_m1_i  (cfg_group_count_m1_w),
        .req_en_i             (post_req_en_w),
        .cfg_elem_count_i     (cfg_elem_count_w),
        .cfg_position_i       (cfg_position_w),
        .busy_o               (post_busy_w),
        .done_o               (post_done_w),
        .error_o              (post_error_w),
        .rope_lut_rd_en_o     (rope_lut_rd_en_w),
        .rope_lut_rd_addr_o   (rope_lut_rd_addr_w),
        .rope_lut_rd_data_i   (act_rd_data),
        .gemv_valid_i         (gemv_buffer_pop_valid_w),
        .gemv_data_i          (gemv_buffer_pop_data_w),
        .gemv_user_i          (gemv_buffer_pop_user_w),
        .gemv_pop_o           (post_gemv_pop_w),
        .valid_o              (post_out_valid_w),
        .data_o               (post_out_data_w),
        .user_o               (post_out_row_w),
        .kv_scale_valid_o     (kv_scale_valid_w),
        .kv_scale_fp16_o      (kv_scale_fp16_w),
        .kv_scale_user_o      (kv_scale_user_w),
        .kv_scale_ready_i     (1'b1),
        .kv_quant_valid_o     (kv_quant_valid_w),
        .kv_quant_i8_o        (kv_quant_i8_w),
        .kv_quant_user_o      (kv_quant_user_w),
        .kv_quant_ready_i     (1'b1),
        .observe_valid_o      (post_valid_w),
        .observe_data_o       (post_data_w),
        .observe_user_o       (post_row_w),
        .postbuf_count_o      (buffer_count_w),
        .postbuf_push_count_o (buffer_push_count_w),
        .postbuf_pop_count_o  (buffer_pop_count_w),
        .postbuf_overflow_o   (buffer_overflow_w),
        .postbuf_underflow_o  (buffer_underflow_w),
        .mul_pair_count_o     (mul_pair_count_w),
        .mul_output_count_o   (mul_output_count_w),
        .mul_pair_miss_o      (mul_pair_miss_w)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            writer_valid_q <= 1'b0;
        end else begin
            writer_valid_q <= stream_writer_mode_w && post_out_valid_w;
        end
    end

    always_ff @(posedge clk) begin
        if (stream_writer_mode_w && post_out_valid_w) begin
            writer_data_q <= post_out_data_w;
            writer_data32_q <= {16'd0, post_out_data_w};
            writer_row_q  <= post_out_row_w;
        end
    end

    assign writer_valid_w = writer_valid_q;
    assign writer_data_w  = writer_data_q;
    assign writer_data32_w = writer_data32_q;
    assign writer_row_w   = writer_row_q;

    stream_line_writer #(
        .SPM_SIZE       (SPM_SIZE),
        .SPM_DATA_WIDTH (SPM_DATA_WIDTH),
        .USER_WIDTH     (USER_WIDTH)
    ) u_stream_writer (
        .clk              (clk),
        .rst_n            (rst_n),
        .cfg_dst_act_i    (writer_dst_act_w),
        .cfg_word32_i     (1'b0),
        .cfg_base_addr_i  (matvec_output_addr_q[SPM_ADDR_WIDTH-1:0]),
        .cfg_elem_count_i (cfg_total_elem_count_w),
        .cfg_group_elem_count_i (cfg_elem_count_w),
        .cfg_group_stride_elems_i (cfg_group_stride_elem_count_w),
        .cfg_group_stride_en_i (stream_writer_group_stride_w),
        .req_en_i         (stream_writer_req_en_w),
        .busy_o           (stream_writer_busy_w),
        .done_o           (stream_writer_done_w),
        .valid_i          (writer_valid_w),
        .data_i           (writer_data_w),
        .data32_i         (writer_data32_w),
        .user_i           (writer_row_w),
        .spm_wr_en_o      (stream_spm_wr_en_w),
        .spm_wr_addr_o    (stream_spm_wr_addr_w),
        .spm_wr_data_o    (stream_spm_wr_data_w),
        .spm_wr_mask_o    (stream_spm_wr_mask_w),
        .act_wr_en_o      (stream_act_wr_en_w),
        .act_wr_addr_o    (stream_act_wr_addr_w),
        .act_wr_data_o    (stream_act_wr_data_w),
        .act_wr_mask_o    (stream_act_wr_mask_w)
    );

    generate
        if (DISABLE_DECODE_KV_QUANT != 0) begin : gen_no_kv_cache_writer
            assign kv_writer_busy_w = 1'b0;
            assign kv_writer_done_w = 1'b0;
            assign kv_spm_wr_en_w = 1'b0;
            assign kv_spm_wr_addr_w = '0;
            assign kv_spm_wr_data_w = '0;
            assign kv_spm_wr_mask_w = '0;
        end else begin : gen_kv_cache_writer
            kv_cache_writer #(
                .SPM_SIZE       (SPM_SIZE),
                .SPM_DATA_WIDTH (SPM_DATA_WIDTH),
                .USER_WIDTH     (USER_WIDTH),
                .CACHE_HEAD_DIM  (ROPE_HEAD_DIM),
                .CACHE_ROW_BYTES (ROPE_HEAD_DIM),
                .GEMV_TILE_ELEMS (TILE_ELEMS),
                .CACHE_HEAD_COUNT(5),
                .ROW_TILE_ELEMS  (ROW_TILE_ELEMS),
                .KV_HEAD_STRIDE_BYTES (65536)
            ) u_kv_cache_writer (
                .clk              (clk),
                .rst_n            (rst_n),
                .cfg_base_addr_i  (matvec_output_addr_q[SPM_ADDR_WIDTH-1:0]),
                .cfg_elem_count_i (cfg_elem_count_w),
                .cfg_cache_cell_i (matvec_cache_cell_idx_q),
                .cfg_v_separated_i(cfg_kv_v_separated_w),
                .cfg_scratch_i    (cfg_kv_quant_scratch_w),
                .req_en_i         (kv_writer_req_en_w),
                .busy_o           (kv_writer_busy_w),
                .done_o           (kv_writer_done_w),
                .scale_valid_i    (kv_scale_valid_w),
                .scale_fp16_i     (kv_scale_fp16_w),
                .scale_user_i     (kv_scale_user_w),
                .quant_valid_i    (kv_quant_valid_w),
                .quant_i8_i       (kv_quant_i8_w),
                .quant_user_i     (kv_quant_user_w),
                .spm_wr_en_o      (kv_spm_wr_en_w),
                .spm_wr_addr_o    (kv_spm_wr_addr_w),
                .spm_wr_data_o    (kv_spm_wr_data_w),
                .spm_wr_mask_o    (kv_spm_wr_mask_w)
            );
        end
    endgenerate

endmodule

`endif
