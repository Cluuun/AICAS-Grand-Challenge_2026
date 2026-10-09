`ifndef POSTPROCESS_TOP_SV
`define POSTPROCESS_TOP_SV

module postprocess_top #(
    parameter int USER_WIDTH        = 16,
    parameter int POST_BUFFER_DEPTH = 4096,
    parameter int SOFTMAX_MAX_LEN   = 4096,
    parameter int ROPE_HEAD_DIM     = 64,
    parameter int ROPE_MAX_SEQ_LEN  = 4096,
    parameter int ROPE_POS_WIDTH    = 16,
    parameter int SPM_DATA_WIDTH    = 512,
    parameter int SPM_ADDR_WIDTH    = 19,
    parameter int ROPE_LUT_BASE_ADDR = 32'h0000_3f00,
    parameter int DISABLE_DECODE_ROPE = 0,
    parameter int DISABLE_DECODE_SOFTMAX = 0,
    parameter int DISABLE_DECODE_KV_QUANT = 0,
    parameter int SCORE_SCALE_POW2_SHIFT = 0,

    localparam int COUNT_WIDTH = $clog2(POST_BUFFER_DEPTH + 1)
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic [3:0]              cfg_src0_i,
    input  logic [3:0]              cfg_src1_i,
    input  logic [1:0]              cfg_unary_op_i,
    input  logic                    cfg_binary_op_i,
    input  logic [1:0]              cfg_reduce_op_i,
    input  logic [2:0]              cfg_dst_i,
    input  logic                    cfg_kv_quant_en_i,
    input  logic [1:0]              cfg_group_count_m1_i,
    input  logic                    req_en_i,
    input  logic [USER_WIDTH-1:0]   cfg_elem_count_i,
    input  logic [ROPE_POS_WIDTH-1:0] cfg_position_i,
    output logic                    busy_o,
    output logic                    done_o,
    output logic                    error_o,

    output logic                    rope_lut_rd_en_o,
    output logic [SPM_ADDR_WIDTH-1:0] rope_lut_rd_addr_o,
    input  logic [SPM_DATA_WIDTH-1:0] rope_lut_rd_data_i,

    input  logic                    gemv_valid_i,
    input  logic [15:0]             gemv_data_i,
    input  logic [USER_WIDTH-1:0]   gemv_user_i,
    output logic                    gemv_pop_o,

    output logic                    valid_o,
    output logic [15:0]             data_o,
    output logic [USER_WIDTH-1:0]   user_o,

    output logic                    kv_scale_valid_o,
    output logic [15:0]             kv_scale_fp16_o,
    output logic [USER_WIDTH-1:0]   kv_scale_user_o,
    input  logic                    kv_scale_ready_i,
    output logic                    kv_quant_valid_o,
    output logic signed [7:0]       kv_quant_i8_o,
    output logic [USER_WIDTH-1:0]   kv_quant_user_o,
    input  logic                    kv_quant_ready_i,

    output logic                    observe_valid_o,
    output logic [15:0]             observe_data_o,
    output logic [USER_WIDTH-1:0]   observe_user_o,

    output logic [COUNT_WIDTH-1:0]  postbuf_count_o,
    output logic [COUNT_WIDTH-1:0]  postbuf_push_count_o,
    output logic [COUNT_WIDTH-1:0]  postbuf_pop_count_o,
    output logic                    postbuf_overflow_o,
    output logic                    postbuf_underflow_o,

    output logic [USER_WIDTH:0]     mul_pair_count_o,
    output logic [USER_WIDTH:0]     mul_output_count_o,
    output logic                    mul_pair_miss_o
);

    localparam logic [3:0] SRC_NONE        = 4'd0;
    localparam logic [3:0] SRC_GEMV_STREAM = 4'd1;
    localparam logic [3:0] SRC_POST_BUFFER = 4'd2;

    localparam logic [1:0] UNARY_BYPASS = 2'd0;
    localparam logic [1:0] UNARY_SILU   = 2'd1;
    localparam logic [1:0] UNARY_ROPE   = 2'd2;

    localparam logic       BINARY_BYPASS  = 1'b0;
    localparam logic       BINARY_FP16_MUL = 1'b1;

    localparam logic [1:0] REDUCE_BYPASS  = 2'd0;
    localparam logic [1:0] REDUCE_SOFTMAX = 2'd1;

    localparam logic [2:0] DST_OUTPUT_SPM  = 3'd1;
    localparam logic [2:0] DST_ACT_BUFFER  = 3'd2;
    localparam logic [2:0] DST_POST_BUFFER = 3'd3;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_RUN,
        ST_DONE
    } state_t;

    state_t state_q;

    logic [3:0] src0_q;
    logic [3:0] src1_q;
    logic [1:0] unary_q;
    logic binary_q;
    logic [1:0] reduce_q;
    logic [2:0] dst_q;
    logic kv_quant_en_q;
    logic [USER_WIDTH-1:0] elem_count_q;
    logic [1:0] group_count_m1_q;
    logic [COUNT_WIDTH-1:0] total_elem_count_count_q;
    logic [COUNT_WIDTH-1:0] elem_count_count_q;

    logic dst_postbuf_w;
    logic dst_writer_w;
    logic dst_kv_w;
    logic use_softmax_w;
    logic use_rope_w;
    logic use_silu_w;
    logic use_mul_w;
    logic use_postbuf_src0_w;
    logic use_postbuf_src1_w;
    logic tail_ready_w;
    logic route_source_valid_w;
    logic route_valid_w;
    logic route_fire_w;
    logic [15:0] route_data_w;
    logic [USER_WIDTH-1:0] route_user_w;
    logic kv_ready_w;
    logic kv_done_w;
    logic kv_block_done_w;

    logic postbuf_clear_w;
    logic postbuf_stats_clear_w;
    logic postbuf_push_valid_w;
    logic postbuf_push_ready_w;
    logic postbuf_stage_ready_w;
    logic postbuf_stage_fire_w;
    logic postbuf_stage_valid_q;
    logic [15:0] postbuf_stage_data_q;
    logic [USER_WIDTH-1:0] postbuf_stage_user_q;
    logic postbuf_pop_en_w;
    logic postbuf_pop_src0_w;
    logic postbuf_pop_mul_w;
    logic postbuf_overflow_w;
    logic postbuf_underflow_w;
    logic postbuf_pop_valid_w;
    logic [15:0] postbuf_pop_data_w;
    logic [USER_WIDTH-1:0] postbuf_pop_user_w;
    logic [COUNT_WIDTH-1:0] postbuf_count_w;
    logic [COUNT_WIDTH-1:0] postbuf_push_count_w;
    logic [COUNT_WIDTH-1:0] postbuf_pop_count_w;

    logic silu_input_fire_w;
    logic silu_valid_w;
    logic [15:0] silu_data_w;
    logic [USER_WIDTH-1:0] silu_user_w;

    logic rope_valid_w;
    logic [15:0] rope_data_w;
    logic [USER_WIDTH-1:0] rope_user_w;
    logic rope_busy_w;
    logic rope_done_w;
    logic rope_error_w;
    logic rope_gemv_pop_w;
    logic rope_ready_w;

    logic mul_valid_w;
    logic [15:0] mul_data_w;
    logic [USER_WIDTH-1:0] mul_user_w;
    logic mul_pair_miss_w;
    logic [USER_WIDTH:0] mul_pair_count_w;
    logic [USER_WIDTH:0] mul_output_count_w;
    logic gemv_pop_mul_w;

    logic softmax_valid_w;
    logic [15:0] softmax_data_w;
    logic [USER_WIDTH-1:0] softmax_user_w;
    logic softmax_busy_w;
    logic softmax_done_w;
    logic softmax_error_w;
    logic softmax_score_pop_w;
    logic softmax_score_valid_w;
    logic [15:0] softmax_score_data_w;

    logic [COUNT_WIDTH-1:0] output_count_q;
    logic illegal_req_w;
    logic disabled_req_w;
    logic done_hit_w;

    assign busy_o = (state_q != ST_IDLE);

    assign dst_postbuf_w      = (dst_q == DST_POST_BUFFER);
    assign dst_kv_w           = kv_quant_en_q && (dst_q == DST_OUTPUT_SPM);
    assign dst_writer_w       = ((dst_q == DST_OUTPUT_SPM) && !dst_kv_w) ||
                                (dst_q == DST_ACT_BUFFER);
    assign use_softmax_w      = (reduce_q == REDUCE_SOFTMAX);
    assign use_rope_w         = (unary_q == UNARY_ROPE);
    assign use_silu_w         = (unary_q == UNARY_SILU);
    assign use_mul_w          = (binary_q == BINARY_FP16_MUL);
    assign use_postbuf_src0_w = (src0_q == SRC_POST_BUFFER);
    assign use_postbuf_src1_w = (src1_q == SRC_POST_BUFFER);
    assign tail_ready_w       = dst_postbuf_w ? postbuf_stage_ready_w :
                                dst_kv_w ? kv_ready_w :
                                1'b1;

    assign postbuf_clear_w       = req_en_i && (cfg_dst_i == DST_POST_BUFFER);
    assign postbuf_stats_clear_w = req_en_i;

    assign postbuf_stage_ready_w = !postbuf_stage_valid_q ||
                                   (postbuf_push_ready_w && postbuf_stage_valid_q);
    assign postbuf_stage_fire_w = postbuf_stage_valid_q && postbuf_push_ready_w;
    assign postbuf_push_valid_w = postbuf_stage_fire_w;
    assign postbuf_pop_en_w = postbuf_pop_src0_w || postbuf_pop_mul_w;

    assign postbuf_count_o       = postbuf_count_w;
    assign postbuf_push_count_o  = postbuf_push_count_w;
    assign postbuf_pop_count_o   = postbuf_pop_count_w;
    assign postbuf_overflow_o    = postbuf_overflow_w;
    assign postbuf_underflow_o   = postbuf_underflow_w;
    assign mul_pair_count_o      = mul_pair_count_w;
    assign mul_output_count_o    = mul_output_count_w;
    assign mul_pair_miss_o       = mul_pair_miss_w;

    assign disabled_req_w = ((DISABLE_DECODE_ROPE != 0) &&
                             (cfg_unary_op_i == UNARY_ROPE)) ||
                            ((DISABLE_DECODE_SOFTMAX != 0) &&
                             (cfg_reduce_op_i == REDUCE_SOFTMAX)) ||
                            ((DISABLE_DECODE_KV_QUANT != 0) &&
                             cfg_kv_quant_en_i);
    assign illegal_req_w = req_en_i &&
                           ((cfg_elem_count_i == '0) || disabled_req_w ||
                            ((COUNT_WIDTH'(cfg_elem_count_i) *
                              (COUNT_WIDTH'(cfg_group_count_m1_i) + COUNT_WIDTH'(1))) >
                             COUNT_WIDTH'(POST_BUFFER_DEPTH)));

    always_comb begin
        gemv_pop_o = 1'b0;
        postbuf_pop_src0_w = 1'b0;
        silu_input_fire_w = 1'b0;
        rope_ready_w = 1'b0;
        softmax_score_valid_w = 1'b0;
        softmax_score_data_w = '0;
        route_source_valid_w = 1'b0;
        route_data_w = '0;
        route_user_w = '0;

        if (state_q == ST_RUN) begin
            if (use_mul_w) begin
                gemv_pop_o = gemv_pop_mul_w;
                route_source_valid_w = mul_valid_w && !use_softmax_w;
                route_data_w = mul_data_w;
                route_user_w = mul_user_w;
            end else if (use_silu_w) begin
                silu_input_fire_w = gemv_valid_i && !use_softmax_w && tail_ready_w;
                gemv_pop_o = silu_input_fire_w;
                route_source_valid_w = silu_valid_w && !use_softmax_w;
                route_data_w = silu_data_w;
                route_user_w = silu_user_w;
            end else if (use_rope_w) begin
                rope_ready_w = use_softmax_w ? softmax_score_pop_w : tail_ready_w;
                gemv_pop_o = rope_gemv_pop_w;
                if (use_softmax_w) begin
                    softmax_score_valid_w = rope_valid_w;
                    softmax_score_data_w = rope_data_w;
                    route_source_valid_w = softmax_valid_w;
                    route_data_w = softmax_data_w;
                    route_user_w = softmax_user_w;
                end else begin
                    route_source_valid_w = rope_valid_w;
                    route_data_w = rope_data_w;
                    route_user_w = rope_user_w;
                end
            end else if (use_softmax_w) begin
                if (use_postbuf_src0_w) begin
                    softmax_score_valid_w = postbuf_pop_valid_w;
                    softmax_score_data_w = postbuf_pop_data_w;
                    postbuf_pop_src0_w = softmax_score_pop_w;
                end else begin
                    softmax_score_valid_w = gemv_valid_i;
                    softmax_score_data_w = gemv_data_i;
                    gemv_pop_o = softmax_score_pop_w;
                end
                route_source_valid_w = softmax_valid_w;
                route_data_w = softmax_data_w;
                route_user_w = softmax_user_w;
            end else begin
                if (use_postbuf_src0_w) begin
                    postbuf_pop_src0_w = postbuf_pop_valid_w && tail_ready_w;
                    route_source_valid_w = postbuf_pop_valid_w;
                    route_data_w = postbuf_pop_data_w;
                    route_user_w = postbuf_pop_user_w;
                end else begin
                    gemv_pop_o = gemv_valid_i && tail_ready_w;
                    route_source_valid_w = gemv_valid_i;
                    route_data_w = gemv_data_i;
                    route_user_w = gemv_user_i;
                end
            end
        end
    end

    assign route_valid_w = route_source_valid_w && tail_ready_w;
    assign route_fire_w = route_valid_w;
    assign valid_o = dst_writer_w && route_fire_w;
    assign data_o  = route_data_w;
    assign user_o  = route_user_w;

    always_comb begin
        observe_valid_o = route_fire_w;
        observe_data_o = route_data_w;
        observe_user_o = route_user_w;

        if (use_silu_w) begin
            observe_valid_o = silu_valid_w;
            observe_data_o = silu_data_w;
            observe_user_o = silu_user_w;
        end else if (use_rope_w) begin
            observe_valid_o = rope_valid_w;
            observe_data_o = rope_data_w;
            observe_user_o = rope_user_w;
        end
    end

    stream_buffer #(
        .DATA_WIDTH (16),
        .USER_WIDTH (USER_WIDTH),
        .DEPTH      (POST_BUFFER_DEPTH)
    ) u_post_stream_buffer (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear_i         (postbuf_clear_w),
        .stats_clear_i   (postbuf_stats_clear_w),
        .push_valid_i    (postbuf_push_valid_w),
        .push_data_i     (postbuf_stage_data_q),
        .push_user_i     (postbuf_stage_user_q),
        .push_ready_o    (postbuf_push_ready_w),
        .pop_en_i        (postbuf_pop_en_w),
        .pop_valid_o     (postbuf_pop_valid_w),
        .pop_data_o      (postbuf_pop_data_w),
        .pop_user_o      (postbuf_pop_user_w),
        .count_o         (postbuf_count_w),
        .empty_o         (),
        .full_o          (),
        .overflow_o      (postbuf_overflow_w),
        .underflow_o     (postbuf_underflow_w),
        .push_count_o    (postbuf_push_count_w),
        .pop_count_o     (postbuf_pop_count_w)
    );

    silu_top #(
        .USER_WIDTH (USER_WIDTH)
    ) u_silu (
        .clk      (clk),
        .rst_n    (rst_n),
        .enable_i (use_silu_w && (state_q == ST_RUN)),
        .valid_i  (silu_input_fire_w),
        .fp16_i   (gemv_data_i),
        .user_i   (gemv_user_i),
        .valid_o  (silu_valid_w),
        .fp16_o   (silu_data_w),
        .user_o   (silu_user_w)
    );

    generate
        if (DISABLE_DECODE_ROPE != 0) begin : gen_no_rope
            assign rope_busy_w = 1'b0;
            assign rope_done_w = 1'b0;
            assign rope_error_w = 1'b0;
            assign rope_gemv_pop_w = 1'b0;
            assign rope_valid_w = 1'b0;
            assign rope_data_w = '0;
            assign rope_user_w = '0;
            assign rope_lut_rd_en_o = 1'b0;
            assign rope_lut_rd_addr_o = '0;
        end else begin : gen_rope
            rope_stream_unit #(
                .USER_WIDTH      (USER_WIDTH),
                .HEAD_DIM        (ROPE_HEAD_DIM),
                .MAX_SEQ_LEN     (ROPE_MAX_SEQ_LEN),
                .POS_WIDTH       (ROPE_POS_WIDTH),
                .SPM_DATA_WIDTH  (SPM_DATA_WIDTH),
                .SPM_ADDR_WIDTH  (SPM_ADDR_WIDTH),
                .ROPE_LUT_BASE_ADDR (ROPE_LUT_BASE_ADDR)
            ) u_rope (
                .clk          (clk),
                .rst_n        (rst_n),
                .start_i      (req_en_i && (cfg_unary_op_i == UNARY_ROPE)),
                .elem_count_i (cfg_elem_count_i),
                .position_i   (cfg_position_i),
                .busy_o       (rope_busy_w),
                .done_o       (rope_done_w),
                .error_o      (rope_error_w),
                .lut_rd_en_o  (rope_lut_rd_en_o),
                .lut_rd_addr_o(rope_lut_rd_addr_o),
                .lut_rd_data_i(rope_lut_rd_data_i),
                .in_valid_i   (gemv_valid_i),
                .in_data_i    (gemv_data_i),
                .in_pop_o     (rope_gemv_pop_w),
                .valid_o      (rope_valid_w),
                .data_o       (rope_data_w),
                .user_o       (rope_user_w),
                .ready_i      (rope_ready_w)
            );
        end
    endgenerate

    fp16_mul_stream #(
        .USER_WIDTH (USER_WIDTH)
    ) u_fp16_mul_stream (
        .clk            (clk),
        .rst_n          (rst_n),
        .enable_i       (use_mul_w && (state_q == ST_RUN)),
        .src0_valid_i   (gemv_valid_i),
        .src0_data_i    (gemv_data_i),
        .src0_user_i    (gemv_user_i),
        .src0_pop_o     (gemv_pop_mul_w),
        .src1_valid_i   (postbuf_pop_valid_w),
        .src1_data_i    (postbuf_pop_data_w),
        .src1_pop_o     (postbuf_pop_mul_w),
        .valid_o        (mul_valid_w),
        .data_o         (mul_data_w),
        .user_o         (mul_user_w),
        .pair_miss_o    (mul_pair_miss_w),
        .pair_count_o   (mul_pair_count_w),
        .output_count_o (mul_output_count_w)
    );

    generate
        if (DISABLE_DECODE_SOFTMAX != 0) begin : gen_no_softmax
            assign softmax_valid_w = 1'b0;
            assign softmax_data_w = '0;
            assign softmax_user_w = '0;
            assign softmax_busy_w = 1'b0;
            assign softmax_done_w = 1'b0;
            assign softmax_error_w = 1'b0;
            assign softmax_score_pop_w = 1'b0;
        end else begin : gen_softmax
            softmax_stream_unit #(
                .USER_WIDTH              (USER_WIDTH),
                .MAX_LEN                 (SOFTMAX_MAX_LEN),
                .SCORE_SCALE_POW2_SHIFT  (SCORE_SCALE_POW2_SHIFT)
            ) u_softmax (
                .clk           (clk),
                .rst_n         (rst_n),
                .start_i       (req_en_i && (cfg_reduce_op_i == REDUCE_SOFTMAX)),
                .elem_count_i  (cfg_elem_count_i),
                .group_count_m1_i (cfg_group_count_m1_i),
                .busy_o        (softmax_busy_w),
                .done_o        (softmax_done_w),
                .error_o       (softmax_error_w),
                .score_valid_i (softmax_score_valid_w),
                .score_data_i  (softmax_score_data_w),
                .score_pop_o   (softmax_score_pop_w),
                .valid_o       (softmax_valid_w),
                .data_o        (softmax_data_w),
                .user_o        (softmax_user_w)
            );
        end
    endgenerate

    generate
        if (DISABLE_DECODE_KV_QUANT != 0) begin : gen_no_kv_quant
            assign kv_ready_w = 1'b0;
            assign kv_scale_valid_o = 1'b0;
            assign kv_scale_fp16_o = '0;
            assign kv_scale_user_o = '0;
            assign kv_quant_valid_o = 1'b0;
            assign kv_quant_i8_o = '0;
            assign kv_quant_user_o = '0;
            assign kv_block_done_w = 1'b0;
            assign kv_done_w = 1'b0;
        end else begin : gen_kv_quant
            kv_quant #(
                .USER_WIDTH  (USER_WIDTH),
                .BLOCK_ELEMS (ROPE_HEAD_DIM),
                .BANK_NUM    (5)
            ) u_kv_quant (
                .clk           (clk),
                .rst_n         (rst_n),
                .req_en_i      (req_en_i && cfg_kv_quant_en_i && (cfg_dst_i == DST_OUTPUT_SPM)),
                .clear_i       (1'b0),
                .elem_count_i  (cfg_elem_count_i),
                .valid_i       (dst_kv_w && route_fire_w),
                .fp16_i        (route_data_w),
                .user_i        (route_user_w),
                .ready_o       (kv_ready_w),
                .scale_valid_o (kv_scale_valid_o),
                .scale_fp16_o  (kv_scale_fp16_o),
                .scale_user_o  (kv_scale_user_o),
                .scale_ready_i (kv_scale_ready_i),
                .quant_valid_o (kv_quant_valid_o),
                .quant_i8_o    (kv_quant_i8_o),
                .quant_user_o  (kv_quant_user_o),
                .quant_ready_i (kv_quant_ready_i),
                .busy_o        (),
                .block_done_o  (kv_block_done_w),
                .done_o        (kv_done_w)
            );
        end
    endgenerate

    assign done_hit_w = dst_kv_w ? kv_done_w :
                        dst_postbuf_w ? ((output_count_q >= total_elem_count_count_q) &&
                                         !postbuf_stage_valid_q) :
                        (((reduce_q == REDUCE_SOFTMAX) && softmax_done_w) ||
                         ((reduce_q == REDUCE_BYPASS) &&
                          (output_count_q >= total_elem_count_count_q)));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q            <= ST_IDLE;
            src0_q             <= SRC_NONE;
            src1_q             <= SRC_NONE;
            unary_q            <= UNARY_BYPASS;
            binary_q           <= BINARY_BYPASS;
            reduce_q           <= REDUCE_BYPASS;
            dst_q              <= DST_OUTPUT_SPM;
            kv_quant_en_q      <= 1'b0;
            elem_count_q       <= '0;
            group_count_m1_q   <= '0;
            elem_count_count_q <= '0;
            total_elem_count_count_q <= '0;
            output_count_q     <= '0;
            postbuf_stage_valid_q <= 1'b0;
            postbuf_stage_data_q  <= '0;
            postbuf_stage_user_q  <= '0;
            done_o             <= 1'b0;
            error_o            <= 1'b0;
        end else begin
            done_o  <= 1'b0;
            error_o <= 1'b0;

            if (req_en_i) begin
                src0_q             <= cfg_src0_i;
                src1_q             <= cfg_src1_i;
                unary_q            <= cfg_unary_op_i;
                binary_q           <= cfg_binary_op_i;
                reduce_q           <= cfg_reduce_op_i;
                dst_q              <= cfg_dst_i;
                kv_quant_en_q      <= cfg_kv_quant_en_i;
                elem_count_q       <= cfg_elem_count_i;
                group_count_m1_q   <= cfg_group_count_m1_i;
                elem_count_count_q <= COUNT_WIDTH'(cfg_elem_count_i);
                total_elem_count_count_q <= COUNT_WIDTH'(cfg_elem_count_i) *
                                            (COUNT_WIDTH'(cfg_group_count_m1_i) + COUNT_WIDTH'(1));
                output_count_q     <= '0;
                postbuf_stage_valid_q <= 1'b0;
                postbuf_stage_data_q  <= '0;
                postbuf_stage_user_q  <= '0;
                error_o            <= illegal_req_w;
                state_q            <= illegal_req_w ? ST_DONE : ST_RUN;
            end else begin
                if (postbuf_stage_fire_w && !route_fire_w) begin
                    postbuf_stage_valid_q <= 1'b0;
                end

                if (dst_postbuf_w && route_fire_w) begin
                    postbuf_stage_valid_q <= 1'b1;
                    postbuf_stage_data_q  <= route_data_w;
                    postbuf_stage_user_q  <= route_user_w;
                end

            if (route_fire_w)
                output_count_q <= output_count_q + COUNT_WIDTH'(1);

                if (postbuf_overflow_w || postbuf_underflow_w ||
                    mul_pair_miss_w || softmax_error_w || rope_error_w) begin
                    error_o <= 1'b1;
                    done_o  <= 1'b1;
                    state_q <= ST_IDLE;
                end else if (state_q == ST_DONE) begin
                    done_o  <= 1'b1;
                    state_q <= ST_IDLE;
                end else if ((state_q == ST_RUN) && done_hit_w) begin
                    done_o  <= 1'b1;
                    state_q <= ST_IDLE;
                end
            end
        end
    end

`ifndef SYNTHESIS
    property p_no_req_while_busy;
        @(posedge clk) disable iff (!rst_n) (state_q != ST_IDLE) |-> !req_en_i;
    endproperty
    assert property (p_no_req_while_busy);
`endif

endmodule

`endif
