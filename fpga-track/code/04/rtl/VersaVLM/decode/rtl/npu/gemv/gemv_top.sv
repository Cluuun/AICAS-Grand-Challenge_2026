`ifndef GEMV_TOP_SV
`define GEMV_TOP_SV

module gemv_top #(
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
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_output_addr,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_scale_addr,
    input  logic [1:0]                        gemv_mode,
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

    output logic                              spm_wr_en,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_wr_addr,
    output logic [SPM_DATA_WIDTH-1:0]         spm_wr_data,
    output logic [SPM_DATA_WIDTH/8-1:0]       spm_wr_mask,

    output logic                              raw_valid_o,
    output logic [15:0]                       raw_data_o,
    output logic [RF_DATA_WIDTH/4-1:0]        raw_row_o,
    input  logic                              post_valid_i,
    input  logic [15:0]                       post_data_i,
    input  logic [RF_DATA_WIDTH/4-1:0]        post_row_i,

    input  logic                              preload_acc_en_i,
    input  logic [((GEMV_ACC_NUM <= 1) ? 1 : $clog2(GEMV_ACC_NUM))-1:0] preload_acc_id_i,
    input  logic [((ROW_TILE_ELEMS <= 1) ? 1 : $clog2(ROW_TILE_ELEMS))-1:0] preload_acc_row_i,
    input  logic [31:0]                       preload_acc_data_i
);

    logic core_req_en_w;
    logic writer_req_en_w;
    logic core_busy_w;
    logic core_done_w;
    logic writer_busy_w;
    logic writer_done_w;
    logic unused_writer_dst_act_w;
    logic unused_sfu_silu_en_w;
    logic [1:0] legacy_flow_mode_w;
    logic flow_has_work_w;

    assign legacy_flow_mode_w = 2'b00;
    assign flow_has_work_w = (matvec_input_mat_width != '0) &&
                             (matvec_input_mat_height != '0);

    flow_ctrl u_flow_ctrl (
        .clk              (clk),
        .rst_n            (rst_n),
        .cfg_flow_mode_i  (legacy_flow_mode_w),
        .flow_req_en_i    (matvec_req_en),
        .flow_has_work_i  (flow_has_work_w),
        .gemv_busy_i      (core_busy_w),
        .gemv_done_i      (core_done_w),
        .writer_busy_i    (writer_busy_w),
        .writer_done_i    (writer_done_w),
        .gemv_req_en_o    (core_req_en_w),
        .writer_req_en_o  (writer_req_en_w),
        .sfu_silu_en_o    (unused_sfu_silu_en_w),
        .writer_dst_act_o (unused_writer_dst_act_w),
        .flow_busy_o      (matvec_busy),
        .flow_done_o      (matvec_comp_done)
    );

    gemv_compute_top #(
        .RF_DATA_WIDTH  (RF_DATA_WIDTH),
        .SPM_SIZE       (SPM_SIZE),
        .SPM_DATA_WIDTH (SPM_DATA_WIDTH),
        .GEMV_ACC_NUM   (GEMV_ACC_NUM),
        .TILE_ELEMS     (TILE_ELEMS),
        .ROW_TILE_ELEMS (ROW_TILE_ELEMS),
        .FIXED_FRAC     (FIXED_FRAC)
    ) u_compute (
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
        .matvec_cache_cell_idx   ('0),
        .act_frac_cfg_i          (8'd0),
        .gemv_mode_i             (gemv_mode),
        .gemv_group_count_m1_i   (2'b00),
        .kv_col_scale_en_i       (1'b0),
        .unit_weight_scale_i      (1'b0),
        .matvec_req_en           (core_req_en_w),
        .matvec_busy             (core_busy_w),
        .matvec_comp_done        (core_done_w),
        .spm_rd_en               (spm_rd_en),
        .spm_rd_addr             (spm_rd_addr),
        .spm_rd_data             (spm_rd_data),
        .scale_rd_en             (scale_rd_en),
        .scale_rd_addr           (scale_rd_addr),
        .scale_rd_data           (scale_rd_data),
        .act_rd_en               (act_rd_en),
        .act_rd_addr             (act_rd_addr),
        .act_rd_data             (act_rd_data),
        .weight_stream_mode_i    (1'b0),
        .act_scale_enable_i      (1'b1),
        .act_scale2_enable_i     (1'b0),
        .weight_stream_valid_i   (1'b0),
        .weight_stream_data_i    ('0),
        .weight_stream_last_i    (1'b0),
        .weight_stream_ready_o   (),
        .raw_valid_o             (raw_valid_o),
        .raw_data_o              (raw_data_o),
        .raw_row_o               (raw_row_o),
        .raw_ready_i             (1'b1),
        .preload_acc_en_i        (preload_acc_en_i),
        .preload_acc_id_i        (preload_acc_id_i),
        .preload_acc_row_i       (preload_acc_row_i),
        .preload_acc_data_i      (preload_acc_data_i)
    );

    stream_line_writer #(
        .SPM_SIZE       (SPM_SIZE),
        .SPM_DATA_WIDTH (SPM_DATA_WIDTH),
        .USER_WIDTH     (RF_DATA_WIDTH/4),
        .ENABLE_GROUP_STRIDE (1'b0)
    ) u_writer (
        .clk              (clk),
        .rst_n            (rst_n),
        .cfg_dst_act_i    (1'b0),
        .cfg_word32_i     (1'b0),
        .cfg_base_addr_i  (matvec_output_addr[SPM_ADDR_WIDTH-1:0]),
        .cfg_elem_count_i (matvec_input_mat_height),
        .cfg_group_elem_count_i ('0),
        .cfg_group_stride_elems_i ('0),
        .cfg_group_stride_en_i (1'b0),
        .req_en_i         (writer_req_en_w),
        .busy_o           (writer_busy_w),
        .done_o           (writer_done_w),
        .valid_i          (post_valid_i),
        .data_i           (post_data_i),
        .data32_i         ('0),
        .user_i           (post_row_i),
        .spm_wr_en_o      (spm_wr_en),
        .spm_wr_addr_o    (spm_wr_addr),
        .spm_wr_data_o    (spm_wr_data),
        .spm_wr_mask_o    (spm_wr_mask),
        .act_wr_en_o      (),
        .act_wr_addr_o    (),
        .act_wr_data_o    (),
        .act_wr_mask_o    ()
    );

endmodule

`endif
