`ifndef GEMV_CTRL_SV
`define GEMV_CTRL_SV

module gemv_ctrl #(
    parameter int RF_DATA_WIDTH  = 64,
    parameter int SPM_SIZE       = 1 << 19,
    parameter int SPM_DATA_WIDTH = 512,
    parameter int GEMV_ACC_NUM   = 4,
    parameter int TILE_ELEMS     = 128,
    parameter int ROW_TILE_ELEMS = 32,

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

    output logic [TILE_ELEMS*16-1:0]          dp_weight_o,
    output logic [TILE_ELEMS*16-1:0]          dp_act_vec_o,
    output logic                              dp_act_valid_o,
    output logic                              dp_valid_o,
    output logic [1:0]                        dp_mode_o,
    output logic [((TILE_ELEMS <= 1) ? 1 : $clog2(TILE_ELEMS))-1:0] dp_row_o,
    output logic [RF_DATA_WIDTH/4-1:0]        dp_global_row_o,
    output logic                              dp_last_col_o,
    output logic [15:0]                       fma_scale_o,
    output logic [15:0]                       fma_old_acc_o,
    output logic                              fma_acc_mode_o,
    output logic [SPM_DATA_WIDTH-1:0]         awq_pre_mul_a_o,
    output logic [SPM_DATA_WIDTH-1:0]         awq_pre_mul_b_o,
    output logic                              awq_pre_mul_valid_o,
    input  logic                              awq_pre_mul_done_i,
    input  logic [SPM_DATA_WIDTH-1:0]         awq_pre_mul_z_i,
    input  logic [(SPM_DATA_WIDTH/16)*24-1:0] awq_pre_mul_uq24_z_i,
    input  logic                              preload_acc_en_i,
    input  logic [((GEMV_ACC_NUM <= 1) ? 1 : $clog2(GEMV_ACC_NUM))-1:0] preload_acc_id_i,
    input  logic [((ROW_TILE_ELEMS <= 1) ? 1 : $clog2(ROW_TILE_ELEMS))-1:0] preload_acc_row_i,
    input  logic [31:0]                       preload_acc_data_i,
    input  logic                              fma_valid_i,
    input  logic [((TILE_ELEMS <= 1) ? 1 : $clog2(TILE_ELEMS))-1:0] fma_row_i,
    input  logic [RF_DATA_WIDTH/4-1:0]        fma_global_row_i,
    input  logic                              fma_last_col_i,
    input  logic [15:0]                       fma_result_i,
    input  logic                              raw_ready_i,

    output logic                              raw_valid_o,
    output logic [15:0]                       raw_data_o,
    output logic [RF_DATA_WIDTH/4-1:0]        raw_row_o
);

    localparam int FP16_PER_BEAT         = SPM_DATA_WIDTH / 16;
    localparam int ACT_LOAD_BEATS        = (TILE_ELEMS + FP16_PER_BEAT - 1) / FP16_PER_BEAT;
    localparam int SCALE_LOAD_BEATS      = (ROW_TILE_ELEMS + FP16_PER_BEAT - 1) / FP16_PER_BEAT;
    localparam int BYTE_PER_BEAT         = SPM_DATA_WIDTH / 8;
    localparam int FP16_BEAT_SHIFT       = $clog2(FP16_PER_BEAT);
    localparam int BYTE_BEAT_SHIFT       = $clog2(BYTE_PER_BEAT);
    localparam int ACT_TILE_BYTES        = TILE_ELEMS * 2;
    localparam int SCALE_TILE_BYTES      = ROW_TILE_ELEMS * 2;
    localparam int MAX_WEIGHT_ROW_BYTES  = TILE_ELEMS * 2;
    localparam int MAX_WEIGHT_ROW_BEATS  = (MAX_WEIGHT_ROW_BYTES + BYTE_PER_BEAT - 1) / BYTE_PER_BEAT;
    localparam int DIM_WIDTH             = RF_DATA_WIDTH / 4;
    localparam int TILE_IDX_WIDTH        = (TILE_ELEMS <= 1) ? 1 : $clog2(TILE_ELEMS);
    localparam int ROW_IDX_WIDTH         = (ROW_TILE_ELEMS <= 1) ? 1 : $clog2(ROW_TILE_ELEMS);
    localparam int TILE_SHIFT            = $clog2(TILE_ELEMS);
    localparam int W8_TILE_ELEMS         = TILE_ELEMS / 2;
    localparam int W8_TILE_SHIFT         = TILE_SHIFT - 1;
    localparam int ROW_TILE_SHIFT        = $clog2(ROW_TILE_ELEMS);
    localparam int ACC_DEPTH             = ROW_TILE_ELEMS;
    localparam int ACC_ADDR_WIDTH        = (ACC_DEPTH <= 1) ? 1 : $clog2(ACC_DEPTH);
    localparam int COUNT_WIDTH           = $clog2(TILE_ELEMS + 1);
    localparam int WEIGHT_BEAT_IDX_WIDTH = (MAX_WEIGHT_ROW_BEATS <= 1) ? 1 : $clog2(MAX_WEIGHT_ROW_BEATS);
    localparam int ACT_BEAT_IDX_WIDTH    = (ACT_LOAD_BEATS <= 1) ? 1 : $clog2(ACT_LOAD_BEATS);
    localparam int ACT_CACHE_WAIT        = 4;
    localparam int ACT_COL_CACHE_HIT_WAIT = 1;
    localparam int WAIT_WIDTH            = $clog2(ACT_CACHE_WAIT + 1);
    localparam int AWQ_LOAD_BEATS        = (ACT_LOAD_BEATS > SCALE_LOAD_BEATS) ?
                                           ACT_LOAD_BEATS : SCALE_LOAD_BEATS;
    localparam int AWQ_VEC_MUL_LATENCY   = 10;
    localparam logic [7:0] ACT_FRAC_UQ24_CFG = 8'h98;
    // 16 col tiles cover W8 attention up to 1024 tokens.
    localparam int ACT_COL_CACHE_DEPTH    = 16;
    localparam int ACT_COL_CACHE_IDX_WIDTH =
        (ACT_COL_CACHE_DEPTH <= 1) ? 1 : $clog2(ACT_COL_CACHE_DEPTH);
    localparam int ACT_COL_CACHE_MIN_ROW_TILES = 1;
`ifdef DISABLE_ACT_COL_CACHE
    localparam bit ACT_COL_CACHE_FORCE_DISABLE = 1'b1;
`else
    localparam bit ACT_COL_CACHE_FORCE_DISABLE = 1'b0;
`endif

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD_A,
        ST_LOAD_AWQ_SCALE,
        ST_LOAD_S,
        ST_WAIT_ACT,
        ST_STREAM_W,
        ST_WAIT_DRAIN
    } state_e;

    typedef enum logic [1:0] {
        RD_NONE,
        RD_ACT,
        RD_SCALE,
        RD_WEIGHT
    } rd_kind_e;

    state_e state_q;
    logic [COUNT_WIDTH-1:0] issue_count_q;
    logic [WEIGHT_BEAT_IDX_WIDTH-1:0] weight_beat_q;
    logic stream_warmup_q;
    logic stream_warmup_flush_q;
    logic [WAIT_WIDTH-1:0] wait_count_q;
    logic complete_pending_q;
    logic act_valid_pending_q;
    logic fma_complete_q;

    logic [1:0] gemv_mode_q;
    logic [31:0] base_vec_q;
    logic [31:0] base_act_scale_q;
    logic [31:0] base_act_scale2_q;
    logic [31:0] act_group_stride_bytes_q;
    logic [31:0] weight_tile_start_q;
    logic [31:0] scale_tile_start_q;
    logic [31:0] weight_tile_base_q;
    logic [31:0] scale_tile_base_q;
    logic [31:0] stream_weight_row_addr_q;
    logic [31:0] stream_weight_beat_addr_q;
    logic [31:0] stream_weight_row_stride_bytes_q;
    logic [WEIGHT_BEAT_IDX_WIDTH-1:0] stream_weight_last_beat_q;
    logic stream_last_col_q;
    logic [DIM_WIDTH-1:0] mat_width_q;
    logic [DIM_WIDTH-1:0] mat_height_q;
    logic [DIM_WIDTH-1:0] col_tiles_q;
    logic [DIM_WIDTH-1:0] row_tiles_q;
    logic [DIM_WIDTH-1:0] col_tile_q;
    logic [DIM_WIDTH-1:0] row_tile_q;
    logic [DIM_WIDTH-1:0] col_tiles_left_q;
    logic [DIM_WIDTH-1:0] row_tiles_left_q;
    logic col_tile_last_q;
    logic row_tile_last_q;
    logic [DIM_WIDTH-1:0] tile_valid_cols_w;
    logic [1:0] group_count_m1_q;
    logic [1:0] group_q;
    logic [RF_DATA_WIDTH/4-1:0] total_output_count_q;
    logic kv_col_scale_en_q;

    logic rd_valid_q;
    rd_kind_e rd_kind_q;
    logic [COUNT_WIDTH-1:0] rd_index_q;
    logic [WEIGHT_BEAT_IDX_WIDTH-1:0] rd_weight_beat_q;
    logic [WEIGHT_BEAT_IDX_WIDTH-1:0] rd_weight_last_beat_q;
    logic [DIM_WIDTH-1:0] rd_col_tile_q;
    logic [DIM_WIDTH-1:0] rd_row_tile_q;
    logic rd_last_col_q;
    logic [1:0] rd_group_q;
    logic rd_pending_valid_q;
    rd_kind_e rd_pending_kind_q;
    logic [COUNT_WIDTH-1:0] rd_pending_index_q;
    logic [WEIGHT_BEAT_IDX_WIDTH-1:0] rd_pending_weight_beat_q;
    logic [WEIGHT_BEAT_IDX_WIDTH-1:0] rd_pending_weight_last_beat_q;
    logic [DIM_WIDTH-1:0] rd_pending_col_tile_q;
    logic [DIM_WIDTH-1:0] rd_pending_row_tile_q;
    logic rd_pending_last_col_q;
    logic [1:0] rd_pending_group_q;
    logic rd_warmup_q;
    logic rd_pending_warmup_q;
    logic rd_stream_weight_q;
    logic rd_pending_stream_weight_q;
    logic act_scale_rd_valid_q;
    logic act_scale_rd_pending_valid_q;
    logic [COUNT_WIDTH-1:0] act_scale_rd_index_q;
    logic [COUNT_WIDTH-1:0] act_scale_rd_pending_index_q;
    logic [DIM_WIDTH-1:0] act_scale_rd_col_tile_q;
    logic [DIM_WIDTH-1:0] act_scale_rd_pending_col_tile_q;
    logic [ACT_BEAT_IDX_WIDTH-1:0] awq_pre_mul_beat_pipe_q [0:AWQ_VEC_MUL_LATENCY];
    logic [COUNT_WIDTH-1:0] awq_pre_mul_done_count_q;

    logic [TILE_ELEMS*16-1:0] act_vec_q;
    logic [TILE_ELEMS*16-1:0] stream_act_vec_q;
    logic [ROW_TILE_ELEMS*16-1:0] scale_vec_q;
    logic [ROW_TILE_ELEMS*16-1:0] scale_prefetch_vec_q;
    logic scale_buf_valid_q;
    logic scale_prefetch_valid_q;
    logic scale_prefetch_active_q;
    logic scale_prefetch_busy_q;
    logic scale_apply_pending_q;
    logic [1:0] scale_apply_wait_q;
    logic [COUNT_WIDTH-1:0] scale_prefetch_count_q;
    logic [31:0] scale_prefetch_base_q;
    logic [DIM_WIDTH-1:0] scale_prefetch_row_tile_q;
    logic scale_prefetch_rd_pending_valid_q;
    logic scale_prefetch_rd_valid_q;
    logic [COUNT_WIDTH-1:0] scale_prefetch_rd_pending_index_q;
    logic [COUNT_WIDTH-1:0] scale_prefetch_rd_index_q;
    logic [DIM_WIDTH-1:0] scale_prefetch_rd_pending_row_tile_q;
    logic [DIM_WIDTH-1:0] scale_prefetch_rd_row_tile_q;
    logic [SPM_DATA_WIDTH-1:0] weight_beat_insert_w;
    logic [SPM_DATA_WIDTH-1:0] weight_read_data_w;
    logic [SPM_DATA_WIDTH-1:0] weight_stream_data_q;
    logic [SPM_DATA_WIDTH-1:0] weight_stream_data_pending_q;
    logic weight_stream_mode_q;
    logic act_scale_enable_q;
    logic act_scale2_enable_q;
    logic act_scale_pass_q;
    logic unit_weight_scale_q;
    logic weight_issue_slot_w;
    logic weight_issue_fire_w;
    logic unused_weight_stream_last_w;
    (* ram_style = "distributed" *) logic [15:0] acc_mem [0:ACC_DEPTH-1];
    (* ram_style = "block" *) logic [SPM_DATA_WIDTH-1:0]
        act_col_cache_mem [0:ACT_LOAD_BEATS-1][0:ACT_COL_CACHE_DEPTH-1];
    logic [ACT_COL_CACHE_DEPTH-1:0] act_col_cache_valid_q;
    logic [1:0] act_col_cache_group_q [0:ACT_COL_CACHE_DEPTH-1];
    logic act_col_cache_store_pending_q;
    logic [ACT_COL_CACHE_IDX_WIDTH-1:0] act_col_cache_store_idx_q;
    logic [1:0] act_col_cache_store_group_q;
    logic act_col_cache_hit_active_q;
    logic act_col_cache_rd_en_q;
    logic [ACT_COL_CACHE_IDX_WIDTH-1:0] act_col_cache_rd_idx_q;
    logic [TILE_ELEMS*16-1:0] act_col_cache_rd_data_q;
    logic act_col_cache_prefetch_pending_q;
    logic act_col_cache_prefetch_valid_q;
    logic [ACT_COL_CACHE_IDX_WIDTH-1:0] act_col_cache_prefetch_pending_idx_q;
    logic [ACT_COL_CACHE_IDX_WIDTH-1:0] act_col_cache_prefetch_idx_q;
    logic [1:0] act_col_cache_prefetch_pending_group_q;
    logic [1:0] act_col_cache_prefetch_group_q;
    logic act_col_cache_key_valid_q;
    logic [RF_DATA_WIDTH/4-1:0] act_col_cache_key_cell_q;
    logic [RF_DATA_WIDTH/4-1:0] act_col_cache_key_mat_width_q;
    logic [RF_DATA_WIDTH/2-1:0] act_col_cache_key_vec_addr_q;
    logic [RF_DATA_WIDTH/2-1:0] act_col_cache_key_act_scale_addr_q;
    logic [RF_DATA_WIDTH/2-1:0] act_col_cache_key_group_stride_q;
    logic [1:0] act_col_cache_key_mode_q;
    logic act_col_cache_key_kv_col_scale_en_q;
    logic act_col_cache_key_act_scale_enable_q;
    logic act_col_cache_key_uq24_packed_q;

    logic [ACC_ADDR_WIDTH-1:0] acc_rd_addr_w;
    logic [15:0] acc_rd_data_w;
    logic [ACC_ADDR_WIDTH-1:0] fma_acc_addr_w;
    logic acc_wr_en_w;
    logic [ACC_ADDR_WIDTH-1:0] acc_wr_addr_w;
    logic [15:0] acc_wr_data_w;

    logic [SPM_DATA_WIDTH-1:0] act_insert_beat_w;
    logic [SPM_DATA_WIDTH-1:0] act_scale_insert_beat_w;
    logic awq_act_scale_en_w;
    logic [SPM_DATA_WIDTH-1:0] scale_insert_beat_w;
    logic [SPM_DATA_WIDTH-1:0] scale_prefetch_insert_beat_w;
    logic raw_out_valid_w;
    logic raw_issue_stall_w;
    logic fma_complete_hit_w;
    logic task_complete_w;
    logic [TILE_ELEMS*16-1:0] dp_weight_q;
    logic [TILE_ELEMS*16-1:0] dp_act_vec_q;
    logic dp_act_valid_q;
    logic dp_valid_q;
    logic [TILE_IDX_WIDTH-1:0] dp_row_q;
    logic [RF_DATA_WIDTH/4-1:0] dp_global_row_q;
    logic dp_last_col_q;
    logic [15:0] fma_scale_q;
    logic [15:0] fma_old_acc_q;
    logic fma_acc_mode_q;
    logic [31:0] act_rd_addr_w;
    logic [31:0] act_scale_base_w;
    logic [31:0] act_scale_rd_addr_w;
    logic [31:0] scale_rd_addr_w;
    logic [31:0] stream_weight_start_addr_w;
    logic [31:0] next_scale_tile_base_w;
    logic [31:0] scale_prefetch_rd_addr_w;
    logic [DIM_WIDTH-1:0] next_col_tile_w;
    logic [DIM_WIDTH-1:0] next_scale_row_tile_w;
    logic [1:0] next_group_w;
    logic [31:0] act_load_beats_w;
    logic [31:0] act_scale_load_beats_w;
    logic [31:0] awq_done_beats_w;
    logic [31:0] awq_load_beats_w;
    logic [31:0] awq_state_load_beats_w;
    logic [1:0] gemv_mode_req_w;
    logic has_next_tile_w;
    logic scale_prefetch_complete_w;
    logic act_col_cache_key_match_w;
    logic act_col_cache_invalidate_req_w;
    logic [RF_DATA_WIDTH/4-1:0] act_col_cache_req_cell_w;
    logic [31:0] weight_row_bytes_w;
    logic [31:0] weight_row_stride_bytes_w;
    logic [31:0] weight_row_beats_w;
    logic [DIM_WIDTH-1:0] rd_tile_valid_cols_w;
    logic [31:0] weight_tile_stride_bytes_w;
    logic [ROW_IDX_WIDTH-1:0] row_index_w;
    logic row_valid_w;
    logic weight_read_last_w;
    logic [ROW_IDX_WIDTH-1:0] fma_ret_row_q;
    logic [DIM_WIDTH-1:0] fma_ret_col_tile_q;
    logic [DIM_WIDTH-1:0] fma_ret_row_tile_q;
    logic [DIM_WIDTH-1:0] fma_ret_row_tiles_left_q;
    logic fma_ret_row_tile_last_q;
    logic [1:0] fma_ret_group_q;
    logic [RF_DATA_WIDTH/4-1:0] fma_ret_output_row_w;
    logic fma_ret_last_col_w;
    logic fma_ret_row_valid_w;
    logic fma_ret_row_countable_w;
    logic unused_preload_acc_id;
    logic unused_act_frac_cfg;
    logic [31:0] act_group_stride_bytes_w;
    logic [RF_DATA_WIDTH/4-1:0] output_row_w;
    logic act_col_cache_enabled_w;
    logic act_col_cache_index_valid_w;
    logic act_col_cache_next_index_valid_w;
    logic [ACT_COL_CACHE_IDX_WIDTH-1:0] act_col_cache_idx_w;
    logic [ACT_COL_CACHE_IDX_WIDTH-1:0] act_col_cache_next_idx_w;
    logic act_col_cache_hit_w;
    logic act_col_cache_prefetch_request_w;
    logic act_col_cache_prefetch_hit_w;
    logic awq_pre_mul_uq24_mode_w;
    logic [WAIT_WIDTH-1:0] cache_hit_wait_limit_w;
    logic [WAIT_WIDTH-1:0] non_awq_wait_limit_w;

    function automatic logic [DIM_WIDTH-1:0] k_tile_elems_for_mode(input logic [1:0] mode);
        begin
            k_tile_elems_for_mode = mode[0] ? DIM_WIDTH'(W8_TILE_ELEMS) :
                                              DIM_WIDTH'(TILE_ELEMS);
        end
    endfunction

    function automatic logic [4:0] k_tile_shift_for_mode(input logic [1:0] mode);
        begin
            k_tile_shift_for_mode = mode[0] ? 5'(W8_TILE_SHIFT) : 5'(TILE_SHIFT);
        end
    endfunction

    function automatic logic [31:0] k_tile_act_bytes_for_mode(input logic [1:0] mode);
        begin
            k_tile_act_bytes_for_mode = mode[0] ? 32'(W8_TILE_ELEMS * 2) :
                                                  32'(TILE_ELEMS * 2);
        end
    endfunction

    function automatic logic [DIM_WIDTH-1:0] ceil_div_k_tile_mode(
        input logic [1:0] mode,
        input logic [DIM_WIDTH-1:0] value
    );
        begin
            ceil_div_k_tile_mode = (value == '0) ? '0 :
                ((value + k_tile_elems_for_mode(mode) - 1'b1) >> k_tile_shift_for_mode(mode));
        end
    endfunction

    function automatic logic [DIM_WIDTH-1:0] ceil_div_row_tile(input logic [DIM_WIDTH-1:0] value);
        begin
            ceil_div_row_tile = (value == '0) ? '0 :
                ((value + DIM_WIDTH'(ROW_TILE_ELEMS - 1)) >> ROW_TILE_SHIFT);
        end
    endfunction

    function automatic logic [31:0] weight_row_bytes_for_mode(input logic [1:0] mode);
        begin
            weight_row_bytes_for_mode = mode[0] ? 32'(W8_TILE_ELEMS) :
                                                   32'(TILE_ELEMS >> 1);
        end
    endfunction

    function automatic logic [31:0] weight_row_bytes_for_tile(
        input logic [1:0] mode,
        input logic [DIM_WIDTH-1:0] valid_cols
    );
        begin
            unique case (mode)
                2'b01: weight_row_bytes_for_tile =
                    (32'(valid_cols) + 32'(BYTE_PER_BEAT - 1)) &
                    ~(32'(BYTE_PER_BEAT - 1));
                default: weight_row_bytes_for_tile = weight_row_bytes_for_mode(mode);
            endcase
        end
    endfunction

    function automatic logic [DIM_WIDTH-1:0] dim_scale_by_group(
        input logic [1:0] group,
        input logic [DIM_WIDTH-1:0] value
    );
        begin
            unique case (group)
                2'd0: dim_scale_by_group = '0;
                2'd1: dim_scale_by_group = value;
                2'd2: dim_scale_by_group = value << 1;
                default: dim_scale_by_group = value + (value << 1);
            endcase
        end
    endfunction

    function automatic logic [DIM_WIDTH-1:0] dim_scale_by_group_count_m1(
        input logic [1:0] group_count_m1,
        input logic [DIM_WIDTH-1:0] value
    );
        begin
            unique case (group_count_m1)
                2'd0: dim_scale_by_group_count_m1 = value;
                2'd1: dim_scale_by_group_count_m1 = value << 1;
                2'd2: dim_scale_by_group_count_m1 = value + (value << 1);
                default: dim_scale_by_group_count_m1 = value << 2;
            endcase
        end
    endfunction

    function automatic logic [31:0] word_scale_by_group(
        input logic [1:0] group,
        input logic [31:0] value
    );
        begin
            unique case (group)
                2'd0: word_scale_by_group = 32'd0;
                2'd1: word_scale_by_group = value;
                2'd2: word_scale_by_group = value << 1;
                default: word_scale_by_group = value + (value << 1);
            endcase
        end
    endfunction

    function automatic logic [SPM_DATA_WIDTH-1:0] mask_act_beat(
        input logic [SPM_DATA_WIDTH-1:0] beat,
        input logic [COUNT_WIDTH-1:0] beat_index,
        input logic [1:0] mode,
        input logic [DIM_WIDTH-1:0] col_tile
    );
        begin
            mask_act_beat = '0;
            for (int lane = 0; lane < FP16_PER_BEAT; lane++) begin
                if (((col_tile << k_tile_shift_for_mode(mode)) +
                     (DIM_WIDTH'(beat_index) << FP16_BEAT_SHIFT) + lane) < mat_width_q)
                    mask_act_beat[lane*16 +: 16] = beat[lane*16 +: 16];
            end
        end
    endfunction

    function automatic logic [SPM_DATA_WIDTH-1:0] mask_scale_beat(
        input logic [SPM_DATA_WIDTH-1:0] beat,
        input logic [COUNT_WIDTH-1:0] beat_index,
        input logic [DIM_WIDTH-1:0] row_tile
    );
        begin
            mask_scale_beat = '0;
            for (int lane = 0; lane < FP16_PER_BEAT; lane++) begin
                if (((row_tile << ROW_TILE_SHIFT) + (DIM_WIDTH'(beat_index) << FP16_BEAT_SHIFT) + lane) < mat_height_q)
                    mask_scale_beat[lane*16 +: 16] = beat[lane*16 +: 16];
            end
        end
    endfunction

    function automatic logic [SPM_DATA_WIDTH-1:0] mask_weight_beat(
        input logic [SPM_DATA_WIDTH-1:0] beat,
        input logic [WEIGHT_BEAT_IDX_WIDTH-1:0] beat_index,
        input logic [1:0] mode,
        input logic [DIM_WIDTH-1:0] valid_cols
    );
        begin
            mask_weight_beat = beat;
            if (mode == 2'b01) begin
                mask_weight_beat = '0;
                for (int lane = 0; lane < BYTE_PER_BEAT; lane++) begin
                    if (((DIM_WIDTH'(beat_index) << BYTE_BEAT_SHIFT) + lane) < valid_cols)
                        mask_weight_beat[lane*8 +: 8] = beat[lane*8 +: 8];
                end
            end
        end
    endfunction

    assign act_insert_beat_w = mask_act_beat(act_rd_data, rd_index_q, gemv_mode_q, rd_col_tile_q);
    assign act_scale_insert_beat_w = mask_act_beat(act_rd_data,
                                                   act_scale_rd_index_q,
                                                   gemv_mode_q,
                                                   act_scale_rd_col_tile_q);
    assign scale_insert_beat_w = mask_scale_beat(scale_rd_data, rd_index_q, rd_row_tile_q);
    assign scale_prefetch_insert_beat_w =
        mask_scale_beat(scale_rd_data,
                        scale_prefetch_rd_index_q,
                        scale_prefetch_rd_row_tile_q);
    assign awq_act_scale_en_w =
        act_scale_enable_q && ((gemv_mode_q == 2'b00) || kv_col_scale_en_q);
    assign act_col_cache_enabled_w =
        !ACT_COL_CACHE_FORCE_DISABLE &&
        !act_scale2_enable_q &&
        awq_act_scale_en_w &&
        (col_tiles_q <= DIM_WIDTH'(ACT_COL_CACHE_DEPTH)) &&
        (row_tiles_q >= DIM_WIDTH'(ACT_COL_CACHE_MIN_ROW_TILES));
    assign act_col_cache_index_valid_w = col_tile_q < DIM_WIDTH'(ACT_COL_CACHE_DEPTH);
    assign act_col_cache_idx_w = col_tile_q[ACT_COL_CACHE_IDX_WIDTH-1:0];
    assign act_col_cache_next_index_valid_w =
        next_col_tile_w < DIM_WIDTH'(ACT_COL_CACHE_DEPTH);
    assign act_col_cache_next_idx_w =
        next_col_tile_w[ACT_COL_CACHE_IDX_WIDTH-1:0];
    assign act_col_cache_hit_w =
        act_col_cache_enabled_w &&
        act_col_cache_index_valid_w &&
        act_col_cache_valid_q[act_col_cache_idx_w] &&
        (act_col_cache_group_q[act_col_cache_idx_w] == group_q);
    assign act_col_cache_prefetch_request_w =
        act_col_cache_enabled_w &&
        has_next_tile_w &&
        (next_group_w == group_q) &&
        act_col_cache_next_index_valid_w &&
        act_col_cache_valid_q[act_col_cache_next_idx_w] &&
        (act_col_cache_group_q[act_col_cache_next_idx_w] == next_group_w);
    assign act_col_cache_prefetch_hit_w =
        act_col_cache_prefetch_valid_q &&
        (act_col_cache_prefetch_idx_q == act_col_cache_idx_w) &&
        (act_col_cache_prefetch_group_q == group_q);
    assign gemv_mode_req_w = {1'b0, gemv_mode_i[0]};
    // The MSB is an inst_ctrl-internal one-shot invalidate tag, not a cache key bit.
    assign act_col_cache_invalidate_req_w = matvec_cache_cell_idx[RF_DATA_WIDTH/4-1];
    assign act_col_cache_req_cell_w = {1'b0, matvec_cache_cell_idx[RF_DATA_WIDTH/4-2:0]};
    assign act_col_cache_key_match_w =
        !act_col_cache_invalidate_req_w &&
        act_col_cache_key_valid_q &&
        (act_col_cache_key_cell_q == act_col_cache_req_cell_w) &&
        (act_col_cache_key_mat_width_q == matvec_input_mat_width) &&
        (act_col_cache_key_vec_addr_q == matvec_input_vec_addr) &&
        (act_col_cache_key_act_scale_addr_q == matvec_input_act_scale_addr) &&
        (act_col_cache_key_group_stride_q == matvec_input_act_group_stride_bytes) &&
        (act_col_cache_key_mode_q == gemv_mode_req_w) &&
        (act_col_cache_key_act_scale_enable_q == act_scale_enable_i) &&
        (act_col_cache_key_uq24_packed_q ==
         ((act_frac_cfg_i == ACT_FRAC_UQ24_CFG) &&
          (gemv_mode_req_w == 2'b01) &&
          act_scale_enable_i &&
          !act_scale2_enable_i)) &&
        (act_col_cache_key_kv_col_scale_en_q ==
         (kv_col_scale_en_i && (gemv_mode_req_w == 2'b01)));

    assign weight_read_data_w = rd_stream_weight_q ? weight_stream_data_q : spm_rd_data;
    assign weight_beat_insert_w =
        row_valid_w ? mask_weight_beat(weight_read_data_w, rd_weight_beat_q,
                                       gemv_mode_q, rd_tile_valid_cols_w) : '0;
    assign weight_issue_slot_w =
        (state_q == ST_STREAM_W) &&
        !stream_warmup_flush_q &&
        (issue_count_q < ROW_TILE_ELEMS) &&
        !raw_issue_stall_w;
    assign weight_stream_ready_o =
        weight_stream_mode_q && weight_issue_slot_w && !stream_warmup_q;
    assign weight_issue_fire_w =
        weight_issue_slot_w &&
        (!weight_stream_mode_q || stream_warmup_q || weight_stream_valid_i);
    assign unused_weight_stream_last_w = weight_stream_last_i;

    assign dp_weight_o     = dp_weight_q;
    assign dp_act_vec_o    = dp_act_valid_q ? act_vec_q : dp_act_vec_q;
    assign dp_act_valid_o  = dp_act_valid_q;
    assign dp_valid_o      = dp_valid_q;
    assign dp_mode_o       = gemv_mode_q;
    assign dp_row_o        = dp_row_q;
    assign dp_global_row_o = dp_global_row_q;
    assign dp_last_col_o   = dp_last_col_q;
    assign fma_scale_o     = fma_scale_q;
    assign fma_old_acc_o   = fma_old_acc_q;
    assign fma_acc_mode_o  = fma_acc_mode_q;
    assign awq_pre_mul_a_o = act_vec_q[act_scale_rd_index_q*SPM_DATA_WIDTH +: SPM_DATA_WIDTH];
    assign awq_pre_mul_b_o = act_scale_insert_beat_w;
    assign awq_pre_mul_valid_o = awq_act_scale_en_w && act_scale_rd_valid_q;
    assign awq_pre_mul_uq24_mode_w =
        (act_frac_cfg_i == ACT_FRAC_UQ24_CFG) &&
        (gemv_mode_q == 2'b01) &&
        act_scale_enable_q &&
        !act_scale2_enable_q;

    assign row_index_w       = rd_index_q[ROW_IDX_WIDTH-1:0];
    assign row_valid_w       = rd_index_q < COUNT_WIDTH'(ROW_TILE_ELEMS);
    assign output_row_w      = dim_scale_by_group(rd_group_q, mat_height_q) +
                               ((rd_row_tile_q << ROW_TILE_SHIFT) + row_index_w);
    assign weight_read_last_w = rd_weight_beat_q == rd_weight_last_beat_q;
    assign acc_rd_addr_w     = row_index_w;
    assign fma_acc_addr_w    = fma_row_i[ACC_ADDR_WIDTH-1:0];
    assign acc_wr_en_w       = fma_valid_i && fma_ret_row_valid_w;
    assign acc_wr_addr_w     = fma_row_i[ACC_ADDR_WIDTH-1:0];
    assign acc_wr_data_w     = fma_result_i;
    assign acc_rd_data_w     =
        (acc_wr_en_w && (acc_wr_addr_w == acc_rd_addr_w)) ?
        acc_wr_data_w : acc_mem[acc_rd_addr_w];

    assign fma_ret_output_row_w = fma_global_row_i;
    assign fma_ret_last_col_w = fma_last_col_i;
    assign fma_ret_row_valid_w = fma_global_row_i < total_output_count_q;
    assign fma_ret_row_countable_w = fma_ret_row_valid_w ||
                                     (fma_global_row_i == total_output_count_q);
    assign raw_out_valid_w   = fma_valid_i && fma_ret_last_col_w && fma_ret_row_valid_w;
    assign raw_issue_stall_w = stream_last_col_q && !raw_ready_i;
    assign raw_valid_o       = raw_out_valid_w;
    assign raw_data_o        = fma_result_i;
    assign raw_row_o         = fma_ret_output_row_w;

    assign fma_complete_hit_w =
        fma_valid_i &&
        fma_ret_row_countable_w &&
        (fma_ret_row_q == ROW_IDX_WIDTH'(ROW_TILE_ELEMS - 1)) &&
        fma_ret_last_col_w &&
        fma_ret_row_tile_last_q &&
        (fma_ret_group_q == group_count_m1_q);
    assign task_complete_w   = fma_complete_q || fma_complete_hit_w;
    assign tile_valid_cols_w =
        col_tile_last_q ?
        (mat_width_q - (col_tile_q << k_tile_shift_for_mode(gemv_mode_q))) :
        k_tile_elems_for_mode(gemv_mode_q);
    assign rd_tile_valid_cols_w =
        rd_last_col_q ?
        (mat_width_q - (rd_col_tile_q << k_tile_shift_for_mode(gemv_mode_q))) :
        k_tile_elems_for_mode(gemv_mode_q);
    assign weight_row_bytes_w = weight_row_bytes_for_tile(gemv_mode_q, tile_valid_cols_w);
    assign weight_row_stride_bytes_w = weight_row_bytes_for_mode(gemv_mode_q);
    assign weight_row_beats_w = (weight_row_bytes_w + 32'(BYTE_PER_BEAT - 1)) >> BYTE_BEAT_SHIFT;
    assign weight_tile_stride_bytes_w = weight_row_stride_bytes_w << ROW_TILE_SHIFT;
    assign act_load_beats_w = (gemv_mode_q == 2'b01) ?
                              ((32'(tile_valid_cols_w) + 32'(FP16_PER_BEAT - 1)) >> FP16_BEAT_SHIFT) :
                              32'(ACT_LOAD_BEATS);
    assign act_scale_load_beats_w = (gemv_mode_q == 2'b01) ?
                                    ((32'(tile_valid_cols_w) + 32'(FP16_PER_BEAT - 1)) >> FP16_BEAT_SHIFT) :
                                    32'(ACT_LOAD_BEATS);
    assign awq_done_beats_w = act_load_beats_w;
    assign awq_load_beats_w = (act_scale_load_beats_w > SCALE_LOAD_BEATS) ?
                              act_scale_load_beats_w : SCALE_LOAD_BEATS;
    assign awq_state_load_beats_w = scale_buf_valid_q ? act_scale_load_beats_w :
                                                        awq_load_beats_w;
    assign scale_prefetch_complete_w =
        scale_prefetch_rd_valid_q &&
        (scale_prefetch_rd_index_q == COUNT_WIDTH'(SCALE_LOAD_BEATS - 1));
    assign act_group_stride_bytes_w = (act_group_stride_bytes_q != 32'd0) ?
                                      act_group_stride_bytes_q :
                                      ((gemv_mode_q == 2'b01) ?
                                       (32'(mat_width_q) << 1) :
                                       (32'(col_tiles_q) << 8));
    assign act_rd_addr_w     = base_vec_q + word_scale_by_group(group_q, act_group_stride_bytes_w) +
                               (32'(col_tile_q) * k_tile_act_bytes_for_mode(gemv_mode_q)) +
                               (32'(issue_count_q) << BYTE_BEAT_SHIFT);
    assign act_scale_base_w = (act_scale_pass_q && act_scale2_enable_q) ?
                              base_act_scale2_q : base_act_scale_q;
    assign act_scale_rd_addr_w = act_scale_base_w +
                                 (32'(col_tile_q) * k_tile_act_bytes_for_mode(gemv_mode_q)) +
                                 (32'(issue_count_q) << BYTE_BEAT_SHIFT);
    assign scale_rd_addr_w   = scale_tile_base_q + (32'(issue_count_q) << BYTE_BEAT_SHIFT);
    assign stream_weight_start_addr_w = weight_tile_base_q;
    assign scale_prefetch_rd_addr_w =
        scale_prefetch_base_q + (32'(scale_prefetch_count_q) << BYTE_BEAT_SHIFT);
    assign cache_hit_wait_limit_w = WAIT_WIDTH'(ACT_COL_CACHE_HIT_WAIT);
    assign non_awq_wait_limit_w = '0;
    assign unused_preload_acc_id = |preload_acc_id_i;
    assign unused_act_frac_cfg = |(act_frac_cfg_i & ~8'h98);

    always_comb begin
        has_next_tile_w = 1'b1;
        next_scale_tile_base_w = scale_tile_base_q + SCALE_TILE_BYTES;
        next_col_tile_w = col_tile_q + 1'b1;
        next_scale_row_tile_w = row_tile_q;
        next_group_w = group_q;

        if (col_tile_last_q && row_tile_last_q) begin
            if (group_q == group_count_m1_q) begin
                has_next_tile_w = 1'b0;
                next_scale_tile_base_w = scale_tile_base_q;
                next_col_tile_w = col_tile_q;
                next_scale_row_tile_w = row_tile_q;
                next_group_w = group_q;
            end else begin
                next_scale_tile_base_w = scale_tile_start_q;
                next_col_tile_w = '0;
                next_scale_row_tile_w = '0;
                next_group_w = group_q + 1'b1;
            end
        end else if (col_tile_last_q) begin
            next_col_tile_w = '0;
            next_scale_row_tile_w = row_tile_q + 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            issue_count_q <= '0;
            weight_beat_q <= '0;
            stream_warmup_q <= 1'b0;
            stream_warmup_flush_q <= 1'b0;
            wait_count_q <= '0;
            complete_pending_q <= 1'b0;
            act_valid_pending_q <= 1'b0;
            fma_complete_q <= 1'b0;
            gemv_mode_q <= '0;
            base_vec_q <= '0;
            base_act_scale_q <= '0;
            base_act_scale2_q <= '0;
            act_group_stride_bytes_q <= '0;
            weight_tile_start_q <= '0;
            scale_tile_start_q <= '0;
            weight_tile_base_q <= '0;
            scale_tile_base_q <= '0;
            stream_weight_row_addr_q <= '0;
            stream_weight_beat_addr_q <= '0;
            stream_weight_row_stride_bytes_q <= 32'(TILE_ELEMS >> 1);
            stream_weight_last_beat_q <= '0;
            stream_last_col_q <= 1'b0;
            mat_width_q <= '0;
            mat_height_q <= '0;
            col_tiles_q <= '0;
            row_tiles_q <= '0;
            col_tile_q <= '0;
            row_tile_q <= '0;
            col_tiles_left_q <= '0;
            row_tiles_left_q <= '0;
            col_tile_last_q <= 1'b0;
            row_tile_last_q <= 1'b0;
            group_count_m1_q <= '0;
            group_q <= '0;
            total_output_count_q <= '0;
            kv_col_scale_en_q <= 1'b0;
            rd_valid_q <= 1'b0;
            rd_kind_q <= RD_NONE;
            rd_index_q <= '0;
            rd_weight_beat_q <= '0;
            rd_weight_last_beat_q <= '0;
            rd_col_tile_q <= '0;
            rd_row_tile_q <= '0;
            rd_last_col_q <= 1'b0;
            rd_group_q <= '0;
            rd_pending_valid_q <= 1'b0;
            rd_pending_kind_q <= RD_NONE;
            rd_pending_index_q <= '0;
            rd_pending_weight_beat_q <= '0;
            rd_pending_weight_last_beat_q <= '0;
            rd_pending_col_tile_q <= '0;
            rd_pending_row_tile_q <= '0;
            rd_pending_last_col_q <= 1'b0;
            rd_pending_group_q <= '0;
            rd_warmup_q <= 1'b0;
            rd_pending_warmup_q <= 1'b0;
            rd_stream_weight_q <= 1'b0;
            rd_pending_stream_weight_q <= 1'b0;
            act_scale_rd_valid_q <= 1'b0;
            act_scale_rd_pending_valid_q <= 1'b0;
            act_scale_rd_index_q <= '0;
            act_scale_rd_pending_index_q <= '0;
            act_scale_rd_col_tile_q <= '0;
            act_scale_rd_pending_col_tile_q <= '0;
            awq_pre_mul_done_count_q <= '0;
            for (int i = 0; i <= AWQ_VEC_MUL_LATENCY; i++) begin
                awq_pre_mul_beat_pipe_q[i] <= '0;
            end
            matvec_busy <= 1'b0;
            matvec_comp_done <= 1'b0;
            spm_rd_en <= 1'b0;
            spm_rd_addr <= '0;
            scale_rd_en <= 1'b0;
            scale_rd_addr <= '0;
            act_rd_en <= 1'b0;
            act_rd_addr <= '0;
            act_vec_q <= '0;
            stream_act_vec_q <= '0;
            scale_vec_q <= '0;
            scale_prefetch_vec_q <= '0;
            scale_buf_valid_q <= 1'b0;
            scale_prefetch_valid_q <= 1'b0;
            scale_prefetch_active_q <= 1'b0;
            scale_prefetch_busy_q <= 1'b0;
            scale_apply_pending_q <= 1'b0;
            scale_apply_wait_q <= 1'b0;
            scale_prefetch_count_q <= '0;
            scale_prefetch_base_q <= '0;
            scale_prefetch_row_tile_q <= '0;
            scale_prefetch_rd_pending_valid_q <= 1'b0;
            scale_prefetch_rd_valid_q <= 1'b0;
            scale_prefetch_rd_pending_index_q <= '0;
            scale_prefetch_rd_index_q <= '0;
            scale_prefetch_rd_pending_row_tile_q <= '0;
            scale_prefetch_rd_row_tile_q <= '0;
            weight_stream_data_q <= '0;
            weight_stream_data_pending_q <= '0;
            weight_stream_mode_q <= 1'b0;
            act_scale_enable_q <= 1'b0;
            act_scale2_enable_q <= 1'b0;
            act_scale_pass_q <= 1'b0;
            unit_weight_scale_q <= 1'b0;
            dp_weight_q <= '0;
            dp_act_vec_q <= '0;
            dp_act_valid_q <= 1'b0;
            dp_valid_q <= 1'b0;
            dp_row_q <= '0;
            dp_global_row_q <= '0;
            dp_last_col_q <= 1'b0;
            fma_scale_q <= '0;
            fma_old_acc_q <= '0;
            fma_acc_mode_q <= 1'b0;
            fma_ret_row_q <= '0;
            fma_ret_col_tile_q <= '0;
            fma_ret_row_tile_q <= '0;
            fma_ret_row_tiles_left_q <= '0;
            fma_ret_row_tile_last_q <= 1'b0;
            fma_ret_group_q <= '0;
            act_col_cache_valid_q <= '0;
            act_col_cache_store_pending_q <= 1'b0;
            act_col_cache_store_idx_q <= '0;
            act_col_cache_store_group_q <= '0;
            act_col_cache_hit_active_q <= 1'b0;
            act_col_cache_rd_en_q <= 1'b0;
            act_col_cache_rd_idx_q <= '0;
            act_col_cache_prefetch_pending_q <= 1'b0;
            act_col_cache_prefetch_valid_q <= 1'b0;
            act_col_cache_prefetch_pending_idx_q <= '0;
            act_col_cache_prefetch_idx_q <= '0;
            act_col_cache_prefetch_pending_group_q <= '0;
            act_col_cache_prefetch_group_q <= '0;
            act_col_cache_key_valid_q <= 1'b0;
            act_col_cache_key_cell_q <= '0;
            act_col_cache_key_mat_width_q <= '0;
            act_col_cache_key_vec_addr_q <= '0;
            act_col_cache_key_act_scale_addr_q <= '0;
            act_col_cache_key_group_stride_q <= '0;
            act_col_cache_key_mode_q <= '0;
            act_col_cache_key_kv_col_scale_en_q <= 1'b0;
            act_col_cache_key_act_scale_enable_q <= 1'b0;
            act_col_cache_key_uq24_packed_q <= 1'b0;
            for (int i = 0; i < ACT_COL_CACHE_DEPTH; i++) begin
                act_col_cache_group_q[i] <= '0;
            end
        end else begin
            matvec_comp_done <= 1'b0;
            spm_rd_en <= 1'b0;
            scale_rd_en <= 1'b0;
            act_rd_en <= 1'b0;
            dp_act_valid_q <= 1'b0;
            dp_valid_q <= 1'b0;
            act_col_cache_rd_en_q <= 1'b0;

            if (act_col_cache_store_pending_q) begin
                act_col_cache_valid_q[act_col_cache_store_idx_q] <= 1'b1;
                act_col_cache_group_q[act_col_cache_store_idx_q] <=
                    act_col_cache_store_group_q;
                act_col_cache_store_pending_q <= 1'b0;
            end

            if (act_valid_pending_q) begin
                dp_act_valid_q <= 1'b1;
                act_valid_pending_q <= 1'b0;
            end

            if (complete_pending_q) begin
                matvec_busy <= 1'b0;
                matvec_comp_done <= 1'b1;
                complete_pending_q <= 1'b0;
                state_q <= ST_IDLE;
                rd_valid_q <= 1'b0;
                rd_kind_q <= RD_NONE;
                rd_pending_valid_q <= 1'b0;
                rd_pending_kind_q <= RD_NONE;
                rd_warmup_q <= 1'b0;
                rd_pending_warmup_q <= 1'b0;
                rd_stream_weight_q <= 1'b0;
                rd_pending_stream_weight_q <= 1'b0;
                weight_stream_mode_q <= 1'b0;
                unit_weight_scale_q <= 1'b0;
                scale_prefetch_rd_pending_valid_q <= 1'b0;
                scale_prefetch_rd_valid_q <= 1'b0;
                act_scale_rd_valid_q <= 1'b0;
                act_scale_rd_pending_valid_q <= 1'b0;
                awq_pre_mul_done_count_q <= '0;
                fma_complete_q <= 1'b0;
                scale_buf_valid_q <= 1'b0;
                scale_prefetch_valid_q <= 1'b0;
                scale_prefetch_active_q <= 1'b0;
                scale_prefetch_busy_q <= 1'b0;
                scale_apply_pending_q <= 1'b0;
                scale_apply_wait_q <= 1'b0;
                scale_prefetch_count_q <= '0;
                act_col_cache_store_pending_q <= 1'b0;
                act_col_cache_hit_active_q <= 1'b0;
                act_col_cache_rd_en_q <= 1'b0;
                act_col_cache_prefetch_pending_q <= 1'b0;
                act_col_cache_prefetch_valid_q <= 1'b0;
                stream_warmup_q <= 1'b0;
                stream_warmup_flush_q <= 1'b0;
            end else begin
                if (act_col_cache_prefetch_pending_q) begin
                    act_col_cache_prefetch_valid_q <= 1'b1;
                    act_col_cache_prefetch_idx_q <= act_col_cache_prefetch_pending_idx_q;
                    act_col_cache_prefetch_group_q <=
                        act_col_cache_prefetch_pending_group_q;
                    act_col_cache_prefetch_pending_q <= 1'b0;
                end

                if (rd_valid_q) begin
                    unique case (rd_kind_q)
                        RD_ACT: begin
                            if (rd_index_q == '0) begin
                                act_vec_q <= '0;
                            end
                            act_vec_q[rd_index_q*SPM_DATA_WIDTH +: SPM_DATA_WIDTH] <=
                                act_insert_beat_w;
                            if ((rd_index_q == (act_load_beats_w - 1)) && !awq_act_scale_en_w) begin
                                act_valid_pending_q <= 1'b1;
                            end
                        end
                        RD_SCALE: begin
                            scale_vec_q[rd_index_q*SPM_DATA_WIDTH +: SPM_DATA_WIDTH] <=
                                scale_insert_beat_w[ROW_TILE_ELEMS*16-1:0];
                            if (rd_index_q == COUNT_WIDTH'(SCALE_LOAD_BEATS - 1)) begin
                                scale_buf_valid_q <= 1'b1;
                            end
                        end
                        RD_WEIGHT: begin
                            dp_weight_q[rd_weight_beat_q*SPM_DATA_WIDTH +: SPM_DATA_WIDTH] <=
                                weight_beat_insert_w;
                            if (weight_read_last_w) begin
                                dp_act_vec_q <= stream_act_vec_q;
                                dp_valid_q <= 1'b1;
                                dp_row_q <= {{(TILE_IDX_WIDTH-ROW_IDX_WIDTH){1'b0}}, row_index_w};
                                dp_global_row_q <= rd_warmup_q ? (total_output_count_q + 1'b1) :
                                                   (row_valid_w &&
                                                    (((rd_row_tile_q << ROW_TILE_SHIFT) + row_index_w) < mat_height_q)) ?
                                                   output_row_w : total_output_count_q;
                                dp_last_col_q <= rd_last_col_q;
                                fma_scale_q <= row_valid_w ?
                                               (unit_weight_scale_q ? 16'h3c00 :
                                                scale_vec_q[row_index_w*16 +: 16]) :
                                               16'h0000;
                                fma_old_acc_q <= acc_rd_data_w;
                                fma_acc_mode_q <= (rd_col_tile_q != '0);
                            end
                        end
                        default: begin end
                    endcase
                end

                if (scale_prefetch_rd_valid_q) begin
                    scale_prefetch_vec_q[scale_prefetch_rd_index_q*SPM_DATA_WIDTH +: SPM_DATA_WIDTH] <=
                        scale_prefetch_insert_beat_w[ROW_TILE_ELEMS*16-1:0];
                    if (scale_prefetch_rd_index_q == COUNT_WIDTH'(SCALE_LOAD_BEATS - 1)) begin
                        scale_prefetch_valid_q <= 1'b1;
                        scale_prefetch_busy_q <= 1'b0;
                    end
                end

                if (scale_apply_pending_q) begin
                    if (scale_apply_wait_q != '0) begin
                        scale_apply_wait_q <= scale_apply_wait_q - 1'b1;
                    end else begin
                        scale_vec_q <= scale_prefetch_vec_q;
                        scale_apply_pending_q <= 1'b0;
                    end
                end

                awq_pre_mul_beat_pipe_q[0] <= act_scale_rd_index_q[ACT_BEAT_IDX_WIDTH-1:0];
                for (int i = 1; i <= AWQ_VEC_MUL_LATENCY; i++) begin
                    awq_pre_mul_beat_pipe_q[i] <= awq_pre_mul_beat_pipe_q[i-1];
                end

                if (awq_pre_mul_done_i) begin
                    if (awq_pre_mul_uq24_mode_w) begin
                        for (int lane = 0; lane < FP16_PER_BEAT; lane++) begin
                            if (awq_pre_mul_beat_pipe_q[AWQ_VEC_MUL_LATENCY] == '0) begin
                                act_vec_q[lane*24 +: 24] <=
                                    awq_pre_mul_uq24_z_i[lane*24 +: 24];
                            end else if (awq_pre_mul_beat_pipe_q[AWQ_VEC_MUL_LATENCY] ==
                                         ACT_BEAT_IDX_WIDTH'(1)) begin
                                act_vec_q[(FP16_PER_BEAT + lane)*24 +: 24] <=
                                    awq_pre_mul_uq24_z_i[lane*24 +: 24];
                            end
                        end
                    end else begin
                        act_vec_q[awq_pre_mul_beat_pipe_q[AWQ_VEC_MUL_LATENCY]*SPM_DATA_WIDTH +: SPM_DATA_WIDTH] <=
                            awq_pre_mul_z_i;
                    end
                    if (awq_pre_mul_done_count_q == (awq_done_beats_w - 1)) begin
                        awq_pre_mul_done_count_q <= '0;
                        act_valid_pending_q <= 1'b1;
                    end else begin
                        awq_pre_mul_done_count_q <= awq_pre_mul_done_count_q + 1'b1;
                    end
                end

                rd_valid_q <= rd_pending_valid_q;
                rd_kind_q <= rd_pending_kind_q;
                rd_index_q <= rd_pending_index_q;
                rd_weight_beat_q <= rd_pending_weight_beat_q;
                rd_weight_last_beat_q <= rd_pending_weight_last_beat_q;
                rd_col_tile_q <= rd_pending_col_tile_q;
                rd_row_tile_q <= rd_pending_row_tile_q;
                rd_last_col_q <= rd_pending_last_col_q;
                rd_group_q <= rd_pending_group_q;
                rd_warmup_q <= rd_pending_warmup_q;
                rd_stream_weight_q <= rd_pending_stream_weight_q;
                weight_stream_data_q <= weight_stream_data_pending_q;
                rd_pending_valid_q <= 1'b0;
                rd_pending_kind_q <= RD_NONE;
                rd_pending_index_q <= '0;
                rd_pending_weight_beat_q <= '0;
                rd_pending_weight_last_beat_q <= '0;
                rd_pending_col_tile_q <= '0;
                rd_pending_row_tile_q <= '0;
                rd_pending_last_col_q <= 1'b0;
                rd_pending_group_q <= '0;
                rd_pending_warmup_q <= 1'b0;
                rd_pending_stream_weight_q <= 1'b0;
                scale_prefetch_rd_valid_q <= scale_prefetch_rd_pending_valid_q;
                scale_prefetch_rd_index_q <= scale_prefetch_rd_pending_index_q;
                scale_prefetch_rd_row_tile_q <= scale_prefetch_rd_pending_row_tile_q;
                scale_prefetch_rd_pending_valid_q <= 1'b0;
                scale_prefetch_rd_pending_index_q <= '0;
                scale_prefetch_rd_pending_row_tile_q <= '0;
                act_scale_rd_valid_q <= act_scale_rd_pending_valid_q;
                act_scale_rd_index_q <= act_scale_rd_pending_index_q;
                act_scale_rd_col_tile_q <= act_scale_rd_pending_col_tile_q;
                act_scale_rd_pending_valid_q <= 1'b0;
                act_scale_rd_pending_index_q <= '0;
                act_scale_rd_pending_col_tile_q <= '0;

                if (fma_valid_i && fma_ret_row_countable_w) begin
                    if (fma_ret_row_q == (ROW_TILE_ELEMS - 1)) begin
                        fma_ret_row_q <= '0;
                        if (fma_ret_last_col_w) begin
                            fma_ret_col_tile_q <= '0;
                            if (fma_ret_row_tile_last_q) begin
                                fma_ret_row_tile_q <= '0;
                                fma_ret_row_tiles_left_q <= row_tiles_q;
                                fma_ret_row_tile_last_q <= (row_tiles_q == DIM_WIDTH'(1));
                                if (fma_ret_group_q == group_count_m1_q)
                                    fma_ret_group_q <= '0;
                                else
                                    fma_ret_group_q <= fma_ret_group_q + 1'b1;
                            end else begin
                                fma_ret_row_tile_q <= fma_ret_row_tile_q + 1'b1;
                                fma_ret_row_tiles_left_q <= fma_ret_row_tiles_left_q - 1'b1;
                                fma_ret_row_tile_last_q <=
                                    (fma_ret_row_tiles_left_q == DIM_WIDTH'(2));
                            end
                        end else begin
                            fma_ret_col_tile_q <= fma_ret_col_tile_q + 1'b1;
                        end
                    end else begin
                        fma_ret_row_q <= fma_ret_row_q + 1'b1;
                    end
                    if (fma_complete_hit_w) begin
                        fma_complete_q <= 1'b1;
                    end
                end

                if (task_complete_w) begin
                    complete_pending_q <= 1'b1;
                end

                unique case (state_q)
                    ST_IDLE: begin
                        matvec_busy <= 1'b0;
                        if (matvec_req_en) begin
                            logic [1:0] mode_next;

                            mode_next = gemv_mode_req_w;

                            gemv_mode_q <= mode_next;
                            weight_stream_mode_q <= weight_stream_mode_i;
                            act_scale_enable_q <= act_scale_enable_i;
                            act_scale2_enable_q <= act_scale2_enable_i;
                            act_scale_pass_q <= 1'b0;
                            unit_weight_scale_q <= unit_weight_scale_i;
                            kv_col_scale_en_q <= kv_col_scale_en_i && (mode_next == 2'b01);
                            base_vec_q <= matvec_input_vec_addr;
                            base_act_scale_q <= matvec_input_act_scale_addr;
                            base_act_scale2_q <= matvec_input_act_scale2_addr;
                            act_group_stride_bytes_q <= matvec_input_act_group_stride_bytes;
                            weight_tile_start_q <= matvec_input_mat_addr;
                            weight_tile_base_q <= matvec_input_mat_addr;
                            scale_tile_start_q <= {{(32-DIM_WIDTH){1'b0}},
                                                   matvec_input_scale_addr};
                            scale_tile_base_q <= {{(32-DIM_WIDTH){1'b0}},
                                                  matvec_input_scale_addr};
                            scale_buf_valid_q <= 1'b0;
                            if (unit_weight_scale_i) begin
                                scale_buf_valid_q <= 1'b1;
                            end
                            scale_prefetch_valid_q <= 1'b0;
                            scale_prefetch_active_q <= 1'b0;
                            scale_prefetch_busy_q <= 1'b0;
                            scale_apply_pending_q <= 1'b0;
                            scale_apply_wait_q <= 1'b0;
                            scale_prefetch_count_q <= '0;
                            scale_prefetch_base_q <= {{(32-DIM_WIDTH){1'b0}},
                                                      matvec_input_scale_addr};
                            scale_prefetch_row_tile_q <= '0;
                            mat_width_q <= matvec_input_mat_width;
                            mat_height_q <= matvec_input_mat_height;
                            col_tiles_q <= ceil_div_k_tile_mode(mode_next, matvec_input_mat_width);
                            row_tiles_q <= ceil_div_row_tile(matvec_input_mat_height);
                            col_tile_q <= '0;
                            row_tile_q <= '0;
                            col_tiles_left_q <= ceil_div_k_tile_mode(mode_next, matvec_input_mat_width);
                            row_tiles_left_q <= ceil_div_row_tile(matvec_input_mat_height);
                            col_tile_last_q <=
                                (ceil_div_k_tile_mode(mode_next, matvec_input_mat_width) == DIM_WIDTH'(1));
                            row_tile_last_q <=
                                (ceil_div_row_tile(matvec_input_mat_height) == DIM_WIDTH'(1));
                            group_count_m1_q <= gemv_group_count_m1_i;
                            group_q <= '0;
                            total_output_count_q <=
                                dim_scale_by_group_count_m1(gemv_group_count_m1_i,
                                                            matvec_input_mat_height);
                            fma_complete_q <= 1'b0;
                            fma_ret_row_q <= '0;
                            fma_ret_col_tile_q <= '0;
                            fma_ret_row_tile_q <= '0;
                            fma_ret_row_tiles_left_q <= ceil_div_row_tile(matvec_input_mat_height);
                            fma_ret_row_tile_last_q <=
                                (ceil_div_row_tile(matvec_input_mat_height) == DIM_WIDTH'(1));
                            fma_ret_group_q <= '0;
                            awq_pre_mul_done_count_q <= '0;
                            act_col_cache_valid_q <= act_col_cache_key_match_w ?
                                                     act_col_cache_valid_q : '0;
                            act_col_cache_store_pending_q <= 1'b0;
                            act_col_cache_hit_active_q <= 1'b0;
                            act_col_cache_prefetch_pending_q <= 1'b0;
                            act_col_cache_prefetch_valid_q <= 1'b0;
                            act_col_cache_key_valid_q <= 1'b1;
                            act_col_cache_key_cell_q <= act_col_cache_req_cell_w;
                            act_col_cache_key_mat_width_q <= matvec_input_mat_width;
                            act_col_cache_key_vec_addr_q <= matvec_input_vec_addr;
                            act_col_cache_key_act_scale_addr_q <= matvec_input_act_scale_addr;
                            act_col_cache_key_group_stride_q <= matvec_input_act_group_stride_bytes;
                            act_col_cache_key_mode_q <= mode_next;
                            act_col_cache_key_act_scale_enable_q <= act_scale_enable_i;
                            act_col_cache_key_uq24_packed_q <=
                                (act_frac_cfg_i == ACT_FRAC_UQ24_CFG) &&
                                (mode_next == 2'b01) &&
                                act_scale_enable_i &&
                                !act_scale2_enable_i;
                            act_col_cache_key_kv_col_scale_en_q <=
                                kv_col_scale_en_i && (mode_next == 2'b01);

                            if ((matvec_input_mat_width == '0) || (matvec_input_mat_height == '0)) begin
                                matvec_comp_done <= 1'b1;
                                weight_stream_mode_q <= 1'b0;
                                unit_weight_scale_q <= 1'b0;
                            end else begin
                                matvec_busy <= 1'b1;
                                state_q <= ST_LOAD_A;
                                act_scale_pass_q <= 1'b0;
                                issue_count_q <= '0;
                                weight_beat_q <= '0;
                            end
                        end
                    end

                    ST_LOAD_A: begin
                        if ((issue_count_q == '0) && act_col_cache_hit_w) begin
                            wait_count_q <= '0;
                            if (scale_buf_valid_q && act_col_cache_prefetch_hit_w) begin
                                act_col_cache_prefetch_valid_q <= 1'b0;
                                act_col_cache_hit_active_q <= 1'b1;
                                wait_count_q <= cache_hit_wait_limit_w;
                                state_q <= ST_WAIT_ACT;
                            end else begin
                                act_col_cache_rd_en_q <= 1'b1;
                                act_col_cache_rd_idx_q <= act_col_cache_idx_w;
                                act_col_cache_hit_active_q <= 1'b1;
                                state_q <= scale_buf_valid_q ? ST_WAIT_ACT : ST_LOAD_S;
                            end
                        end else if (issue_count_q < act_load_beats_w) begin
                            act_rd_en <= 1'b1;
                            act_rd_addr <= act_rd_addr_w[SPM_ADDR_WIDTH-1:0];
                            rd_pending_valid_q <= 1'b1;
                            rd_pending_kind_q <= RD_ACT;
                            rd_pending_index_q <= issue_count_q;
                            rd_pending_weight_beat_q <= '0;
                            rd_pending_weight_last_beat_q <= '0;
                            rd_pending_col_tile_q <= col_tile_q;
                            rd_pending_row_tile_q <= row_tile_q;
                            rd_pending_last_col_q <= col_tile_last_q;
                            rd_pending_group_q <= group_q;
                            if (issue_count_q == (act_load_beats_w - 1)) begin
                                issue_count_q <= '0;
                                if (awq_act_scale_en_w) begin
                                    state_q <= ST_LOAD_AWQ_SCALE;
                                end else if (scale_buf_valid_q) begin
                                    wait_count_q <= '0;
                                    state_q <= ST_WAIT_ACT;
                                end else begin
                                    state_q <= ST_LOAD_S;
                                end
                            end else begin
                                issue_count_q <= issue_count_q + 1'b1;
                            end
                        end
                    end

                    ST_LOAD_AWQ_SCALE: begin
                        if (issue_count_q < awq_state_load_beats_w) begin
                            if (issue_count_q < act_scale_load_beats_w) begin
                                act_rd_en <= 1'b1;
                                act_rd_addr <= act_scale_rd_addr_w[SPM_ADDR_WIDTH-1:0];
                                act_scale_rd_pending_valid_q <= 1'b1;
                                act_scale_rd_pending_index_q <= issue_count_q;
                                act_scale_rd_pending_col_tile_q <= col_tile_q;
                            end
                            if (!scale_buf_valid_q && (issue_count_q < SCALE_LOAD_BEATS)) begin
                                scale_rd_en <= 1'b1;
                                scale_rd_addr <= scale_rd_addr_w[SPM_ADDR_WIDTH-1:0];
                                rd_pending_valid_q <= 1'b1;
                                rd_pending_kind_q <= RD_SCALE;
                                rd_pending_index_q <= issue_count_q;
                                rd_pending_weight_beat_q <= '0;
                                rd_pending_weight_last_beat_q <= '0;
                                rd_pending_col_tile_q <= col_tile_q;
                                rd_pending_row_tile_q <= row_tile_q;
                                rd_pending_last_col_q <= col_tile_last_q;
                                rd_pending_group_q <= group_q;
                            end
                            if (issue_count_q == (awq_state_load_beats_w - 1)) begin
                                issue_count_q <= '0;
                                wait_count_q <= '0;
                                state_q <= ST_WAIT_ACT;
                            end else begin
                                issue_count_q <= issue_count_q + 1'b1;
                            end
                        end
                    end

                    ST_LOAD_S: begin
                        if (scale_buf_valid_q) begin
                            issue_count_q <= '0;
                            wait_count_q <= '0;
                            state_q <= ST_WAIT_ACT;
                        end else if (issue_count_q < SCALE_LOAD_BEATS) begin
                            scale_rd_en <= 1'b1;
                            scale_rd_addr <= scale_rd_addr_w[SPM_ADDR_WIDTH-1:0];
                            rd_pending_valid_q <= 1'b1;
                            rd_pending_kind_q <= RD_SCALE;
                            rd_pending_index_q <= issue_count_q;
                            rd_pending_weight_beat_q <= '0;
                            rd_pending_weight_last_beat_q <= '0;
                            rd_pending_col_tile_q <= col_tile_q;
                            rd_pending_row_tile_q <= row_tile_q;
                            rd_pending_last_col_q <= col_tile_last_q;
                            rd_pending_group_q <= group_q;
                            if (issue_count_q == (SCALE_LOAD_BEATS - 1)) begin
                                issue_count_q <= '0;
                                wait_count_q <= '0;
                                state_q <= ST_WAIT_ACT;
                            end else begin
                                issue_count_q <= issue_count_q + 1'b1;
                            end
                        end
                    end

                    ST_WAIT_ACT: begin
                        if (act_col_cache_hit_active_q) begin
                            if (wait_count_q == '0) begin
                                act_vec_q <= act_col_cache_rd_data_q;
                            end
                            if (wait_count_q == cache_hit_wait_limit_w) begin
                                act_vec_q <= act_col_cache_rd_data_q;
                                wait_count_q <= '0;
                                issue_count_q <= '0;
                                weight_beat_q <= '0;
                                act_valid_pending_q <= 1'b1;
                                act_col_cache_hit_active_q <= 1'b0;
                                stream_weight_row_addr_q <= stream_weight_start_addr_w;
                                stream_weight_beat_addr_q <= stream_weight_start_addr_w;
                                stream_weight_row_stride_bytes_q <= weight_row_stride_bytes_w;
                                stream_weight_last_beat_q <=
                                    WEIGHT_BEAT_IDX_WIDTH'(weight_row_beats_w - 1'b1);
                                stream_last_col_q <= col_tile_last_q;
                                stream_warmup_q <= 1'b1;
                                state_q <= ST_STREAM_W;
                            end else begin
                                wait_count_q <= wait_count_q + 1'b1;
                            end
                        end else if (awq_act_scale_en_w) begin
                            if (awq_pre_mul_done_i &&
                                (awq_pre_mul_done_count_q == (awq_done_beats_w - 1))) begin
                                if (act_scale2_enable_q && !act_scale_pass_q) begin
                                    act_scale_pass_q <= 1'b1;
                                    awq_pre_mul_done_count_q <= '0;
                                    wait_count_q <= '0;
                                    issue_count_q <= '0;
                                    state_q <= ST_LOAD_AWQ_SCALE;
                                end else begin
                                    act_scale_pass_q <= 1'b0;
                                    wait_count_q <= '0;
                                    issue_count_q <= '0;
                                    weight_beat_q <= '0;
                                    stream_weight_row_addr_q <= stream_weight_start_addr_w;
                                    stream_weight_beat_addr_q <= stream_weight_start_addr_w;
                                    stream_weight_row_stride_bytes_q <= weight_row_stride_bytes_w;
                                    stream_weight_last_beat_q <=
                                        WEIGHT_BEAT_IDX_WIDTH'(weight_row_beats_w - 1'b1);
                                    stream_last_col_q <= col_tile_last_q;
                                    stream_warmup_q <= 1'b1;
                                    state_q <= ST_STREAM_W;
                                end
                            end
                        end else begin
                            if (wait_count_q == non_awq_wait_limit_w) begin
                                wait_count_q <= '0;
                                issue_count_q <= '0;
                                weight_beat_q <= '0;
                                stream_weight_row_addr_q <= stream_weight_start_addr_w;
                                stream_weight_beat_addr_q <= stream_weight_start_addr_w;
                                stream_weight_row_stride_bytes_q <= weight_row_stride_bytes_w;
                                stream_weight_last_beat_q <=
                                    WEIGHT_BEAT_IDX_WIDTH'(weight_row_beats_w - 1'b1);
                                stream_last_col_q <= col_tile_last_q;
                                stream_warmup_q <= 1'b1;
                                state_q <= ST_STREAM_W;
                            end else begin
                                wait_count_q <= wait_count_q + 1'b1;
                            end
                        end
                    end

                    ST_STREAM_W: begin
                        if (scale_prefetch_active_q) begin
                            scale_rd_en <= 1'b1;
                            scale_rd_addr <= scale_prefetch_rd_addr_w[SPM_ADDR_WIDTH-1:0];
                            scale_prefetch_rd_pending_valid_q <= 1'b1;
                            scale_prefetch_rd_pending_index_q <= scale_prefetch_count_q;
                            scale_prefetch_rd_pending_row_tile_q <= scale_prefetch_row_tile_q;
                            if (scale_prefetch_count_q == COUNT_WIDTH'(SCALE_LOAD_BEATS - 1)) begin
                                scale_prefetch_active_q <= 1'b0;
                                scale_prefetch_count_q <= '0;
                            end else begin
                                scale_prefetch_count_q <= scale_prefetch_count_q + 1'b1;
                            end
                        end else if (has_next_tile_w && !scale_prefetch_valid_q &&
                                     !scale_prefetch_busy_q) begin
                            scale_rd_en <= 1'b1;
                            scale_rd_addr <= next_scale_tile_base_w[SPM_ADDR_WIDTH-1:0];
                            scale_prefetch_valid_q <= 1'b0;
                            scale_prefetch_busy_q <= 1'b1;
                            scale_prefetch_base_q <= next_scale_tile_base_w;
                            scale_prefetch_row_tile_q <= next_scale_row_tile_w;
                            scale_prefetch_rd_pending_valid_q <= 1'b1;
                            scale_prefetch_rd_pending_index_q <= '0;
                            scale_prefetch_rd_pending_row_tile_q <= next_scale_row_tile_w;
                            if (SCALE_LOAD_BEATS == 1) begin
                                scale_prefetch_active_q <= 1'b0;
                                scale_prefetch_count_q <= '0;
                            end else begin
                                scale_prefetch_active_q <= 1'b1;
                                scale_prefetch_count_q <= COUNT_WIDTH'(1);
                            end
                        end

                        if (act_col_cache_prefetch_request_w &&
                            !act_col_cache_prefetch_pending_q &&
                            !act_col_cache_prefetch_valid_q) begin
                            act_col_cache_rd_en_q <= 1'b1;
                            act_col_cache_rd_idx_q <= act_col_cache_next_idx_w;
                            act_col_cache_prefetch_pending_q <= 1'b1;
                            act_col_cache_prefetch_pending_idx_q <=
                                act_col_cache_next_idx_w;
                            act_col_cache_prefetch_pending_group_q <= next_group_w;
                        end

                        if (stream_warmup_flush_q) begin
                            stream_warmup_flush_q <= 1'b0;
                        end else if (weight_issue_fire_w) begin
                            if ((issue_count_q == '0) && (weight_beat_q == '0)) begin
                                stream_act_vec_q <= act_vec_q;
                            end
                            if (weight_stream_mode_q) begin
                                weight_stream_data_pending_q <=
                                    stream_warmup_q ? '0 : weight_stream_data_i;
                            end else begin
                                spm_rd_en <= 1'b1;
                                spm_rd_addr <= stream_weight_beat_addr_q[SPM_ADDR_WIDTH-1:0];
                            end
                            rd_pending_valid_q <= 1'b1;
                            rd_pending_kind_q <= RD_WEIGHT;
                            rd_pending_index_q <= issue_count_q;
                            rd_pending_weight_beat_q <= weight_beat_q;
                            rd_pending_weight_last_beat_q <= stream_weight_last_beat_q;
                            rd_pending_col_tile_q <= col_tile_q;
                            rd_pending_row_tile_q <= row_tile_q;
                            rd_pending_last_col_q <= stream_last_col_q;
                            rd_pending_group_q <= group_q;
                            rd_pending_warmup_q <= stream_warmup_q;
                            rd_pending_stream_weight_q <= weight_stream_mode_q;

                            if (weight_beat_q == stream_weight_last_beat_q) begin
                                weight_beat_q <= '0;
                                if (stream_warmup_q) begin
                                    stream_warmup_q <= 1'b0;
                                    stream_warmup_flush_q <= (stream_weight_last_beat_q != '0);
                                    stream_weight_beat_addr_q <= stream_weight_row_addr_q;
                                end else begin
                                    stream_weight_row_addr_q <=
                                        stream_weight_row_addr_q + stream_weight_row_stride_bytes_q;
                                    stream_weight_beat_addr_q <=
                                        stream_weight_row_addr_q + stream_weight_row_stride_bytes_q;
                                end
                                if (!stream_warmup_q && (issue_count_q == (ROW_TILE_ELEMS - 1))) begin
                                    issue_count_q <= '0;
                                    if (act_col_cache_enabled_w &&
                                        act_col_cache_index_valid_w &&
                                        (!act_col_cache_valid_q[act_col_cache_idx_w] ||
                                         (act_col_cache_group_q[act_col_cache_idx_w] != group_q))) begin
                                        act_col_cache_store_pending_q <= 1'b1;
                                        act_col_cache_store_idx_q <= act_col_cache_idx_w;
                                        act_col_cache_store_group_q <= group_q;
                                    end
                                    if (col_tile_last_q && row_tile_last_q) begin
                                        if (group_q == group_count_m1_q) begin
                                            scale_buf_valid_q <= 1'b0;
                                            scale_prefetch_valid_q <= 1'b0;
                                            scale_prefetch_active_q <= 1'b0;
                                            scale_prefetch_busy_q <= 1'b0;
                                            scale_apply_pending_q <= 1'b0;
                                            scale_apply_wait_q <= 1'b0;
                                            scale_prefetch_count_q <= '0;
                                            act_col_cache_prefetch_pending_q <= 1'b0;
                                            act_col_cache_prefetch_valid_q <= 1'b0;
                                            state_q <= ST_WAIT_DRAIN;
                                        end else begin
                                            group_q <= group_q + 1'b1;
                                            col_tile_q <= '0;
                                            row_tile_q <= '0;
                                            col_tiles_left_q <= col_tiles_q;
                                            row_tiles_left_q <= row_tiles_q;
                                            col_tile_last_q <= (col_tiles_q == DIM_WIDTH'(1));
                                            row_tile_last_q <= (row_tiles_q == DIM_WIDTH'(1));
                                            weight_tile_base_q <= weight_tile_start_q;
                                            scale_tile_base_q <= scale_tile_start_q;
                                            if (scale_prefetch_valid_q || scale_prefetch_complete_w) begin
                                                scale_buf_valid_q <= 1'b1;
                                                scale_prefetch_valid_q <= 1'b0;
                                                scale_apply_pending_q <= 1'b1;
                                                scale_apply_wait_q <= 2'd2;
                                            end else begin
                                                scale_buf_valid_q <= 1'b0;
                                                scale_apply_pending_q <= 1'b0;
                                            end
                                            scale_prefetch_active_q <= 1'b0;
                                            scale_prefetch_busy_q <= 1'b0;
                                            scale_prefetch_count_q <= '0;
                                            awq_pre_mul_done_count_q <= '0;
                                            act_scale_pass_q <= 1'b0;
                                            act_col_cache_valid_q <= '0;
                                            act_col_cache_prefetch_pending_q <= 1'b0;
                                            act_col_cache_prefetch_valid_q <= 1'b0;
                                            state_q <= ST_LOAD_A;
                                        end
                                    end else begin
                                        if (col_tile_last_q) begin
                                            col_tile_q <= '0;
                                            row_tile_q <= row_tile_q + 1'b1;
                                            col_tiles_left_q <= col_tiles_q;
                                            row_tiles_left_q <= row_tiles_left_q - 1'b1;
                                            col_tile_last_q <= (col_tiles_q == DIM_WIDTH'(1));
                                            row_tile_last_q <= (row_tiles_left_q == DIM_WIDTH'(2));
                                        end else begin
                                            col_tile_q <= col_tile_q + 1'b1;
                                            col_tiles_left_q <= col_tiles_left_q - 1'b1;
                                            col_tile_last_q <= (col_tiles_left_q == DIM_WIDTH'(2));
                                        end
                                        weight_tile_base_q <= weight_tile_base_q + weight_tile_stride_bytes_w;
                                        scale_tile_base_q <= scale_tile_base_q + SCALE_TILE_BYTES;
                                        if (scale_prefetch_valid_q || scale_prefetch_complete_w) begin
                                            scale_buf_valid_q <= 1'b1;
                                            scale_prefetch_valid_q <= 1'b0;
                                            scale_apply_pending_q <= 1'b1;
                                            scale_apply_wait_q <= 2'd2;
                                        end else begin
                                            scale_buf_valid_q <= 1'b0;
                                            scale_apply_pending_q <= 1'b0;
                                        end
                                        scale_prefetch_active_q <= 1'b0;
                                        scale_prefetch_busy_q <= 1'b0;
                                        scale_prefetch_count_q <= '0;
                                        awq_pre_mul_done_count_q <= '0;
                                        act_scale_pass_q <= 1'b0;
                                        state_q <= ST_LOAD_A;
                                    end
                                end else if (!stream_warmup_q) begin
                                    issue_count_q <= issue_count_q + 1'b1;
                                end
                            end else begin
                                weight_beat_q <= weight_beat_q + 1'b1;
                                stream_weight_beat_addr_q <=
                                    stream_weight_beat_addr_q + BYTE_PER_BEAT;
                            end
                        end
                    end

                    ST_WAIT_DRAIN: begin
                        issue_count_q <= '0;
                    end

                    default: state_q <= ST_IDLE;
                endcase
                if (unit_weight_scale_q) begin
                    scale_buf_valid_q <= 1'b1;
                    scale_prefetch_valid_q <= 1'b0;
                    scale_prefetch_active_q <= 1'b0;
                    scale_prefetch_busy_q <= 1'b0;
                    scale_apply_pending_q <= 1'b0;
                    scale_apply_wait_q <= 1'b0;
                    scale_prefetch_count_q <= '0;
                    scale_prefetch_rd_pending_valid_q <= 1'b0;
                    scale_prefetch_rd_valid_q <= 1'b0;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!ACT_COL_CACHE_FORCE_DISABLE) begin
            if (act_col_cache_store_pending_q) begin
                for (int beat = 0; beat < ACT_LOAD_BEATS; beat++) begin
                    act_col_cache_mem[beat][act_col_cache_store_idx_q] <=
                        act_vec_q[beat*SPM_DATA_WIDTH +: SPM_DATA_WIDTH];
                end
            end
            if (act_col_cache_rd_en_q) begin
                for (int beat = 0; beat < ACT_LOAD_BEATS; beat++) begin
                    act_col_cache_rd_data_q[beat*SPM_DATA_WIDTH +: SPM_DATA_WIDTH] <=
                        act_col_cache_mem[beat][act_col_cache_rd_idx_q];
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (acc_wr_en_w) begin
            acc_mem[acc_wr_addr_w] <= acc_wr_data_w;
        end
        if (preload_acc_en_i) begin
            acc_mem[preload_acc_row_i] <= preload_acc_data_i[15:0];
        end
    end

`ifndef SYNTHESIS
    property p_no_req_while_busy;
        @(posedge clk) disable iff (!rst_n) matvec_busy |-> !matvec_req_en;
    endproperty
    assert property (p_no_req_while_busy);

    property p_spm_read_in_range;
        @(posedge clk) disable iff (!rst_n) spm_rd_en |-> (spm_rd_addr < SPM_SIZE);
    endproperty
    assert property (p_spm_read_in_range);

    property p_act_read_in_range;
        @(posedge clk) disable iff (!rst_n) act_rd_en |-> (act_rd_addr < SPM_SIZE);
    endproperty
    assert property (p_act_read_in_range);

    property p_scale_read_in_range;
        @(posedge clk) disable iff (!rst_n) scale_rd_en |-> (scale_rd_addr < SPM_SIZE);
    endproperty
    assert property (p_scale_read_in_range);

    property p_valid_gemv_mode;
        @(posedge clk) disable iff (!rst_n) matvec_req_en |-> !gemv_mode_i[1];
    endproperty
    assert property (p_valid_gemv_mode);
`endif

endmodule

`endif
