`ifndef GEMV_COMPUTE_TOP_SV
`define GEMV_COMPUTE_TOP_SV

module gemv_compute_top #(
    parameter int RF_DATA_WIDTH  = 64,
    parameter int SPM_SIZE       = 1 << 19,
    parameter int SPM_DATA_WIDTH = 512,
    parameter int GEMV_ACC_NUM   = 4,
    parameter int TILE_ELEMS     = 128,
    parameter int ROW_TILE_ELEMS = 32,
    parameter int FIXED_FRAC     = 12,

    localparam int SPM_ADDR_WIDTH = $clog2(SPM_SIZE)
) (
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_mat_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_vec_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_scale_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_scale2_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_group_stride_bytes,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_mat_width,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_mat_height,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_scale_addr,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_cache_cell_idx,
    input  logic [7:0]                        act_frac_cfg_i,
    input  logic [1:0]                        gemv_mode_i,
    input  logic [1:0]                        gemv_group_count_m1_i,
    input  logic                              kv_col_scale_en_i,
    input  logic                              unit_weight_scale_i,
    input  logic                              matvec_req_en,
    output logic                              matvec_busy,
    output logic                              matvec_comp_done,

    output logic                              spm_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         spm_rd_data,

    output logic                              scale_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         scale_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         scale_rd_data,

    output logic                              act_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         act_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         act_rd_data,

    input  logic                              weight_stream_mode_i,
    input  logic                              act_scale_enable_i,
    input  logic                              act_scale2_enable_i,
    input  logic                              weight_stream_valid_i,
    input  logic [SPM_DATA_WIDTH-1:0]         weight_stream_data_i,
    input  logic                              weight_stream_last_i,
    output logic                              weight_stream_ready_o,

    output logic                              raw_valid_o,
    output logic [15:0]                       raw_data_o,
    output logic [RF_DATA_WIDTH/4-1:0]        raw_row_o,
    input  logic                              raw_ready_i,

    input  logic                              preload_acc_en_i,
    input  logic [((GEMV_ACC_NUM <= 1) ? 1 : $clog2(GEMV_ACC_NUM))-1:0] preload_acc_id_i,
    input  logic [((ROW_TILE_ELEMS <= 1) ? 1 : $clog2(ROW_TILE_ELEMS))-1:0] preload_acc_row_i,
    input  logic [31:0]                       preload_acc_data_i
);

    localparam int TILE_IDX_WIDTH = (TILE_ELEMS <= 1) ? 1 : $clog2(TILE_ELEMS);
    // Matches gemv_fp16_dp128 OUTPUT_LATENCY with the latency-4 FP multiply IP plus 3-cycle conversion IP.
    localparam int DP_LATENCY     = 28;

    logic [TILE_ELEMS*16-1:0] dp_weight_w;
    logic [TILE_ELEMS*16-1:0] dp_act_vec_w;
    logic                     dp_act_valid_w;
    logic                     dp_valid_w;
    logic [1:0]               dp_mode_w;
    logic [TILE_IDX_WIDTH-1:0] dp_row_w;
    logic [RF_DATA_WIDTH/4-1:0] dp_global_row_w;
    logic                     dp_last_col_w;
    logic [15:0]              fma_scale_w;
    logic [15:0]              fma_old_acc_w;
    logic                     fma_acc_mode_w;
    logic [15:0]              fma_result_fp16_w;
    logic                     dot_valid_w;
    logic [SPM_DATA_WIDTH-1:0] awq_pre_mul_a_w;
    logic [SPM_DATA_WIDTH-1:0] awq_pre_mul_b_w;
    logic [SPM_DATA_WIDTH-1:0] awq_pre_mul_z_w;
    logic [(SPM_DATA_WIDTH/16)*24-1:0] awq_pre_mul_uq24_z_w;
    logic                     awq_pre_mul_valid_w;
    logic [SPM_DATA_WIDTH-1:0] awq_pre_mul_a_q;
    logic [SPM_DATA_WIDTH-1:0] awq_pre_mul_b_q;
    logic                     awq_pre_mul_valid_q;
    logic                     awq_pre_mul_done_w;
    logic [TILE_IDX_WIDTH-1:0] fma_row_pipe_q [0:DP_LATENCY];
    logic [RF_DATA_WIDTH/4-1:0] fma_global_row_pipe_q [0:DP_LATENCY];
    logic                     fma_last_col_pipe_q [0:DP_LATENCY];

    gemv_ctrl #(
        .RF_DATA_WIDTH  (RF_DATA_WIDTH),
        .SPM_SIZE       (SPM_SIZE),
        .SPM_DATA_WIDTH (SPM_DATA_WIDTH),
        .GEMV_ACC_NUM   (GEMV_ACC_NUM),
        .TILE_ELEMS     (TILE_ELEMS),
        .ROW_TILE_ELEMS (ROW_TILE_ELEMS)
    ) u_ctrl (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .matvec_input_mat_addr   (matvec_input_mat_addr),
        .matvec_input_vec_addr   (matvec_input_vec_addr),
        .matvec_input_act_scale_addr (matvec_input_act_scale_addr),
        .matvec_input_act_scale2_addr (matvec_input_act_scale2_addr),
        .matvec_input_act_group_stride_bytes (matvec_input_act_group_stride_bytes),
        .matvec_input_mat_width  (matvec_input_mat_width),
        .matvec_input_mat_height (matvec_input_mat_height),
        .matvec_input_scale_addr (matvec_input_scale_addr),
        .matvec_cache_cell_idx   (matvec_cache_cell_idx),
        .act_frac_cfg_i          (act_frac_cfg_i),
        .gemv_mode_i             (gemv_mode_i),
        .gemv_group_count_m1_i   (gemv_group_count_m1_i),
        .kv_col_scale_en_i       (kv_col_scale_en_i),
        .unit_weight_scale_i      (unit_weight_scale_i),
        .matvec_req_en           (matvec_req_en),
        .matvec_busy             (matvec_busy),
        .matvec_comp_done        (matvec_comp_done),
        .spm_rd_en               (spm_rd_en),
        .spm_rd_addr             (spm_rd_addr),
        .spm_rd_data             (spm_rd_data),
        .scale_rd_en             (scale_rd_en),
        .scale_rd_addr           (scale_rd_addr),
        .scale_rd_data           (scale_rd_data),
        .act_rd_en               (act_rd_en),
        .act_rd_addr             (act_rd_addr),
        .act_rd_data             (act_rd_data),
        .weight_stream_mode_i    (weight_stream_mode_i),
        .act_scale_enable_i      (act_scale_enable_i),
        .act_scale2_enable_i     (act_scale2_enable_i),
        .weight_stream_valid_i   (weight_stream_valid_i),
        .weight_stream_data_i    (weight_stream_data_i),
        .weight_stream_last_i    (weight_stream_last_i),
        .weight_stream_ready_o   (weight_stream_ready_o),
        .dp_weight_o             (dp_weight_w),
        .dp_act_vec_o            (dp_act_vec_w),
        .dp_act_valid_o          (dp_act_valid_w),
        .dp_valid_o              (dp_valid_w),
        .dp_mode_o               (dp_mode_w),
        .dp_row_o                (dp_row_w),
        .dp_global_row_o         (dp_global_row_w),
        .dp_last_col_o           (dp_last_col_w),
        .fma_scale_o             (fma_scale_w),
        .fma_old_acc_o           (fma_old_acc_w),
        .fma_acc_mode_o          (fma_acc_mode_w),
        .awq_pre_mul_a_o         (awq_pre_mul_a_w),
        .awq_pre_mul_b_o         (awq_pre_mul_b_w),
        .awq_pre_mul_valid_o     (awq_pre_mul_valid_w),
        .awq_pre_mul_done_i      (awq_pre_mul_done_w),
        .awq_pre_mul_z_i         (awq_pre_mul_z_w),
        .awq_pre_mul_uq24_z_i    (awq_pre_mul_uq24_z_w),
        .preload_acc_en_i        (preload_acc_en_i),
        .preload_acc_id_i        (preload_acc_id_i),
        .preload_acc_row_i       (preload_acc_row_i),
        .preload_acc_data_i      (preload_acc_data_i),
        .fma_valid_i             (dot_valid_w),
        .fma_row_i               (fma_row_pipe_q[DP_LATENCY]),
        .fma_global_row_i        (fma_global_row_pipe_q[DP_LATENCY]),
        .fma_last_col_i          (fma_last_col_pipe_q[DP_LATENCY]),
        .fma_result_i            (fma_result_fp16_w),
        .raw_ready_i             (raw_ready_i),
        .raw_valid_o             (raw_valid_o),
        .raw_data_o              (raw_data_o),
        .raw_row_o               (raw_row_o)
    );

    gemv_fp16_dp128 #(
        .TILE_ELEMS     (TILE_ELEMS),
        .VEC_MUL_ELEMS  (SPM_DATA_WIDTH / 16)
    ) u_dp128 (
        .clk_i        (clk),
        .rst_ni       (rst_n),
        .valid_i      (dp_valid_w),
        .act_valid_i  (dp_act_valid_w),
        .act_frac_cfg_i (act_frac_cfg_i),
        .act_uq24_packed_i ((act_frac_cfg_i == 8'h98) &&
                            act_scale_enable_i &&
                            !act_scale2_enable_i &&
                            (dp_mode_w == 2'b01)),
        .mode_i       (dp_mode_w),
        .weight_i     (dp_weight_w),
        .act_vec_i    (dp_act_vec_w),
        .scale_fp16_i (fma_scale_w),
        .old_acc_i    (fma_old_acc_w),
        .acc_mode_i   (fma_acc_mode_w),
        .vec_mul_valid_i (awq_pre_mul_valid_q),
        .vec_mul_a_i     (awq_pre_mul_a_q),
        .vec_mul_b_i     (awq_pre_mul_b_q),
        .vec_mul_o       (awq_pre_mul_z_w),
        .vec_mul_uq24_o  (awq_pre_mul_uq24_z_w),
        .vec_mul_valid_o (awq_pre_mul_done_w),
        .dot_o        (fma_result_fp16_w),
        .valid_o      (dot_valid_w)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            awq_pre_mul_a_q <= '0;
            awq_pre_mul_b_q <= '0;
            awq_pre_mul_valid_q <= 1'b0;
            for (int i = 0; i <= DP_LATENCY; i++) begin
                fma_row_pipe_q[i]        <= '0;
                fma_global_row_pipe_q[i] <= '0;
                fma_last_col_pipe_q[i]   <= 1'b0;
            end
        end else begin
            awq_pre_mul_a_q <= awq_pre_mul_a_w;
            awq_pre_mul_b_q <= awq_pre_mul_b_w;
            awq_pre_mul_valid_q <= awq_pre_mul_valid_w;
            fma_row_pipe_q[0]        <= dp_row_w;
            fma_global_row_pipe_q[0] <= dp_global_row_w;
            fma_last_col_pipe_q[0]   <= dp_last_col_w;
            for (int i = 1; i <= DP_LATENCY; i++) begin
                fma_row_pipe_q[i]        <= fma_row_pipe_q[i-1];
                fma_global_row_pipe_q[i] <= fma_global_row_pipe_q[i-1];
                fma_last_col_pipe_q[i]   <= fma_last_col_pipe_q[i-1];
            end
        end
    end

endmodule

`endif
