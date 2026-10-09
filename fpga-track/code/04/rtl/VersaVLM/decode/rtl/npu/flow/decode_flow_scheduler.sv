`ifndef DECODE_FLOW_SCHEDULER_SV
`define DECODE_FLOW_SCHEDULER_SV

module decode_flow_scheduler #(
    parameter int USER_WIDTH          = 16,
    parameter int STREAM_BUFFER_DEPTH = 4096,
    parameter int ROPE_HEAD_DIM       = 64,
    parameter int ROPE_MAX_SEQ_LEN    = 4096,
    parameter int DISABLE_DECODE_ROPE = 0,
    parameter int DISABLE_DECODE_SOFTMAX = 0,
    parameter int DISABLE_DECODE_KV_QUANT = 0,

    localparam int COUNT_WIDTH = $clog2(STREAM_BUFFER_DEPTH + 1)
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic [63:0]             cfg_decode_flow_i,
    input  logic                    flow_req_en_i,
    input  logic                    flow_has_work_i,

    input  logic                    gemv_busy_i,
    input  logic                    gemv_done_i,
    input  logic                    post_busy_i,
    input  logic                    post_done_i,
    input  logic                    post_error_i,
    input  logic                    writer_busy_i,
    input  logic                    writer_done_i,

    input  logic [COUNT_WIDTH-1:0]  buffer_count_i,
    input  logic [COUNT_WIDTH-1:0]  buffer_push_count_i,
    input  logic [COUNT_WIDTH-1:0]  buffer_pop_count_i,
    input  logic                    buffer_overflow_i,
    input  logic                    buffer_underflow_i,
    input  logic [COUNT_WIDTH-1:0]  gemv_buffer_push_count_i,
    input  logic [COUNT_WIDTH-1:0]  gemv_buffer_pop_count_i,
    input  logic                    gemv_buffer_overflow_i,
    input  logic                    gemv_buffer_underflow_i,
    input  logic [USER_WIDTH:0]     mul_pair_count_i,
    input  logic [USER_WIDTH:0]     mul_output_count_i,
    input  logic                    mul_pair_miss_i,

    output logic                    gemv_req_en_o,
    output logic                    post_req_en_o,
    output logic                    writer_req_en_o,
    output logic                    sfu_silu_en_o,
    output logic                    writer_dst_act_o,
    output logic                    buffer_clear_o,
    output logic                    buffer_stats_clear_o,
    output logic                    buffer_push_en_o,
    output logic                    mul_enable_o,
    output logic                    writer_sel_mul_o,
    output logic                    flow_busy_o,
    output logic                    flow_done_o,
    output logic                    flow_error_o
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

    localparam logic [2:0] DST_NONE        = 3'd0;
    localparam logic [2:0] DST_OUTPUT_SPM  = 3'd1;
    localparam logic [2:0] DST_ACT_BUFFER  = 3'd2;
    localparam logic [2:0] DST_POST_BUFFER = 3'd3;

    localparam logic [COUNT_WIDTH-1:0] BUFFER_DEPTH_COUNT = COUNT_WIDTH'(STREAM_BUFFER_DEPTH);

    logic [3:0] cfg_src0_w;
    logic [3:0] cfg_src1_w;
    logic [1:0] cfg_unary_w;
    logic       cfg_binary_w;
    logic [1:0] cfg_reduce_w;
    logic [2:0] cfg_dst_w;
    logic [3:0] cfg_src_buffer_id_w;
    logic [3:0] cfg_dst_buffer_id_w;
    logic       cfg_kv_quant_en_w;
    logic [1:0] cfg_group_count_m1_w;
    logic [USER_WIDTH-1:0] cfg_elem_count_w;
    logic [COUNT_WIDTH-1:0] cfg_elem_count_count_w;
    logic [COUNT_WIDTH-1:0] cfg_total_elem_count_count_w;
    logic [15:0] cfg_position_w;

    logic uses_gemv_w;
    logic uses_postbuf_r_w;
    logic uses_postbuf_w_w;
    logic uses_writer_w;
    logic has_unary_w;
    logic has_binary_w;
    logic has_softmax_w;
    logic delay_post_w;
    logic source_legal_w;
    logic stage_legal_w;
    logic dst_legal_w;
    logic buffer_legal_w;
    logic rope_legal_w;
    logic softmax_legal_w;
    logic kv_legal_w;
    logic legal_start_w;
    logic illegal_start_w;
    logic delayed_post_start_ready_w;

    logic active_q;
    logic uses_gemv_q;
    logic uses_writer_q;
    logic delay_post_q;
    logic post_started_q;
    logic gemv_done_seen_q;
    logic post_done_seen_q;
    logic writer_done_seen_q;
    logic active_error_w;
    logic done_hit_w;

    assign cfg_src0_w            = cfg_decode_flow_i[3:0];
    assign cfg_src1_w            = cfg_decode_flow_i[7:4];
    assign cfg_unary_w           = cfg_decode_flow_i[9:8];
    assign cfg_binary_w          = cfg_decode_flow_i[10];
    assign cfg_reduce_w          = cfg_decode_flow_i[12:11];
    assign cfg_dst_w             = cfg_decode_flow_i[15:13];
    assign cfg_src_buffer_id_w   = cfg_decode_flow_i[19:16];
    assign cfg_dst_buffer_id_w   = cfg_decode_flow_i[23:20];
    assign cfg_kv_quant_en_w     = cfg_decode_flow_i[24];
    assign cfg_group_count_m1_w   = cfg_decode_flow_i[26:25];
    assign cfg_elem_count_w      = cfg_decode_flow_i[47:32];
    assign cfg_elem_count_count_w = COUNT_WIDTH'(cfg_decode_flow_i[47:32]);
    assign cfg_total_elem_count_count_w =
        COUNT_WIDTH'(cfg_decode_flow_i[47:32]) *
        (COUNT_WIDTH'(cfg_decode_flow_i[26:25]) + COUNT_WIDTH'(1));
    assign cfg_position_w        = cfg_decode_flow_i[63:48];

    assign uses_gemv_w      = (cfg_src0_w == SRC_GEMV_STREAM) ||
                              (cfg_src1_w == SRC_GEMV_STREAM);
    assign uses_postbuf_r_w = (cfg_src0_w == SRC_POST_BUFFER) ||
                              (cfg_src1_w == SRC_POST_BUFFER);
    assign uses_postbuf_w_w = (cfg_dst_w == DST_POST_BUFFER);
    assign uses_writer_w    = (cfg_dst_w == DST_OUTPUT_SPM) ||
                              (cfg_dst_w == DST_ACT_BUFFER);
    assign has_unary_w      = (cfg_unary_w != UNARY_BYPASS);
    assign has_binary_w     = (cfg_binary_w != BINARY_BYPASS);
    assign has_softmax_w    = (cfg_reduce_w == REDUCE_SOFTMAX);
    assign delay_post_w     = uses_gemv_w && (cfg_unary_w == UNARY_ROPE);

    always_comb begin
        source_legal_w = 1'b0;
        if (has_binary_w) begin
            source_legal_w = (cfg_src0_w == SRC_GEMV_STREAM) &&
                             (cfg_src1_w == SRC_POST_BUFFER);
        end else begin
            source_legal_w = ((cfg_src0_w == SRC_GEMV_STREAM) ||
                              (cfg_src0_w == SRC_POST_BUFFER)) &&
                             (cfg_src1_w == SRC_NONE);
        end
    end

    always_comb begin
        stage_legal_w = 1'b0;

        unique case (cfg_unary_w)
            UNARY_BYPASS: stage_legal_w = 1'b1;
            UNARY_SILU: begin
                stage_legal_w = (cfg_src0_w == SRC_GEMV_STREAM) &&
                                !has_binary_w &&
                                (cfg_reduce_w == REDUCE_BYPASS);
            end
            UNARY_ROPE: begin
                stage_legal_w = (cfg_src0_w == SRC_GEMV_STREAM) &&
                                !has_binary_w;
            end
            default: stage_legal_w = 1'b0;
        endcase

        if (has_binary_w) begin
            stage_legal_w = stage_legal_w &&
                            (cfg_unary_w == UNARY_BYPASS) &&
                            (cfg_reduce_w == REDUCE_BYPASS);
        end

        if (!((cfg_reduce_w == REDUCE_BYPASS) ||
              (cfg_reduce_w == REDUCE_SOFTMAX))) begin
            stage_legal_w = 1'b0;
        end
    end

    assign dst_legal_w = (cfg_dst_w == DST_OUTPUT_SPM) ||
                         (cfg_dst_w == DST_ACT_BUFFER) ||
                         (cfg_dst_w == DST_POST_BUFFER);

    assign buffer_legal_w = (!uses_postbuf_r_w ||
                             ((cfg_src_buffer_id_w == 4'd0) &&
                              (buffer_count_i >= cfg_total_elem_count_count_w))) &&
                            (!uses_postbuf_w_w ||
                             (cfg_dst_buffer_id_w == 4'd0)) &&
                            (!(uses_postbuf_r_w && uses_postbuf_w_w));

    assign rope_legal_w = (cfg_unary_w != UNARY_ROPE) ||
                          ((!DISABLE_DECODE_ROPE) &&
                           ((cfg_elem_count_w % USER_WIDTH'(ROPE_HEAD_DIM)) == '0) &&
                           (cfg_position_w < 16'(ROPE_MAX_SEQ_LEN)));

    assign softmax_legal_w = !has_softmax_w ||
                             ((!DISABLE_DECODE_SOFTMAX) &&
                              !has_binary_w &&
                              ((cfg_unary_w == UNARY_BYPASS) ||
                               (cfg_unary_w == UNARY_ROPE)) &&
                              (cfg_total_elem_count_count_w <= BUFFER_DEPTH_COUNT));
    assign kv_legal_w = !cfg_kv_quant_en_w ||
                        ((!DISABLE_DECODE_KV_QUANT) &&
                         (cfg_dst_w == DST_OUTPUT_SPM) &&
                        (!has_binary_w &&
                         !has_softmax_w &&
                         ((cfg_unary_w == UNARY_BYPASS) ||
                          (cfg_unary_w == UNARY_ROPE))));

    assign legal_start_w = source_legal_w &&
                           stage_legal_w &&
                           dst_legal_w &&
                           buffer_legal_w &&
                           rope_legal_w &&
                           softmax_legal_w &&
                           kv_legal_w &&
                           (cfg_elem_count_w != '0) &&
                           (cfg_total_elem_count_count_w <= BUFFER_DEPTH_COUNT);
    assign illegal_start_w = flow_req_en_i && flow_has_work_i && !legal_start_w;

    assign delayed_post_start_ready_w = (gemv_buffer_push_count_i != '0) ||
                                        gemv_done_seen_q || gemv_done_i;

    assign gemv_req_en_o        = flow_req_en_i && flow_has_work_i &&
                                  legal_start_w && uses_gemv_w;
    assign post_req_en_o        = (flow_req_en_i && flow_has_work_i &&
                                   legal_start_w && !delay_post_w) ||
                                  (active_q && delay_post_q && !post_started_q &&
                                   delayed_post_start_ready_w);
    assign writer_req_en_o      = (flow_req_en_i && flow_has_work_i &&
                                  legal_start_w && uses_writer_w && !delay_post_w) ||
                                  (active_q && delay_post_q && !post_started_q &&
                                   uses_writer_q && delayed_post_start_ready_w);
    assign writer_dst_act_o     = active_q ? (uses_writer_q && (cfg_decode_flow_i[15:13] == DST_ACT_BUFFER))
                                           : (cfg_dst_w == DST_ACT_BUFFER);
    assign flow_busy_o          = active_q || gemv_busy_i || post_busy_i || writer_busy_i;

    assign buffer_clear_o       = 1'b0;
    assign buffer_stats_clear_o = 1'b0;
    assign sfu_silu_en_o        = active_q && has_unary_w;
    assign buffer_push_en_o     = active_q && uses_postbuf_w_w;
    assign mul_enable_o         = active_q && has_binary_w;
    assign writer_sel_mul_o     = active_q && has_binary_w;

    assign active_error_w = buffer_overflow_i || buffer_underflow_i ||
                            gemv_buffer_overflow_i || gemv_buffer_underflow_i ||
                            mul_pair_miss_i || post_error_i;

    assign done_hit_w = active_q &&
                        (!uses_gemv_q || gemv_done_seen_q || gemv_done_i) &&
                        (post_done_seen_q || post_done_i) &&
                        (!uses_writer_q || writer_done_seen_q || writer_done_i);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_q           <= 1'b0;
            uses_gemv_q        <= 1'b0;
            uses_writer_q      <= 1'b0;
            delay_post_q       <= 1'b0;
            post_started_q     <= 1'b0;
            gemv_done_seen_q   <= 1'b0;
            post_done_seen_q   <= 1'b0;
            writer_done_seen_q <= 1'b0;
            flow_done_o        <= 1'b0;
            flow_error_o       <= 1'b0;
        end else begin
            flow_done_o  <= 1'b0;
            flow_error_o <= 1'b0;

            if (flow_req_en_i) begin
                active_q           <= flow_has_work_i && legal_start_w;
                uses_gemv_q        <= uses_gemv_w;
                uses_writer_q      <= uses_writer_w;
                delay_post_q       <= delay_post_w;
                post_started_q     <= flow_has_work_i && legal_start_w && !delay_post_w;
                gemv_done_seen_q   <= 1'b0;
                post_done_seen_q   <= !(flow_has_work_i && legal_start_w);
                writer_done_seen_q <= !(flow_has_work_i && legal_start_w && uses_writer_w);
                flow_done_o        <= !flow_has_work_i || illegal_start_w;
                flow_error_o       <= illegal_start_w;
            end else begin
                if (active_q && delay_post_q && !post_started_q &&
                    delayed_post_start_ready_w) begin
                    post_started_q <= 1'b1;
                end
                if (gemv_done_i)
                    gemv_done_seen_q <= 1'b1;
                if (post_done_i)
                    post_done_seen_q <= 1'b1;
                if (writer_done_i)
                    writer_done_seen_q <= 1'b1;

                if (active_error_w) begin
                    flow_done_o        <= 1'b1;
                    flow_error_o       <= 1'b1;
                    active_q           <= 1'b0;
                    uses_gemv_q        <= 1'b0;
                    uses_writer_q      <= 1'b0;
                    delay_post_q       <= 1'b0;
                    post_started_q     <= 1'b0;
                    gemv_done_seen_q   <= 1'b0;
                    post_done_seen_q   <= 1'b0;
                    writer_done_seen_q <= 1'b0;
                end else if (done_hit_w) begin
                    flow_done_o        <= 1'b1;
                    active_q           <= 1'b0;
                    uses_gemv_q        <= 1'b0;
                    uses_writer_q      <= 1'b0;
                    delay_post_q       <= 1'b0;
                    post_started_q     <= 1'b0;
                    gemv_done_seen_q   <= 1'b0;
                    post_done_seen_q   <= 1'b0;
                    writer_done_seen_q <= 1'b0;
                end
            end
        end
    end

`ifndef SYNTHESIS
    property p_no_req_while_active;
        @(posedge clk) disable iff (!rst_n) active_q |-> !flow_req_en_i;
    endproperty
    assert property (p_no_req_while_active);
`endif

endmodule

`endif
