`default_nettype none

import npu_spm_pkg::*;

// C/O mvout converter and unified DMA output FIFO.
//
// Purpose:
//   Convert one 128-bit C/O accumulator half-word into one 128-bit DMA_OC beat.
//   Raw INT32 and Q8.24-scaled FP32 both use the same output register queue.
//
// Clock/reset:
//   Synchronous active-high reset. Control/FIFO state resets; no memories are
//   cleared beyond FIFO pointers.
//
// Valid/ready:
//   A new C/O half-word can be accepted every cycle when queue space is available.
//   FP32 scale mode does not wait for the conversion result before accepting
//   the next half-word; in-flight conversions and completed 128-bit output words are
//   tracked by a small register queue.
module co_output_convert #(
    parameter int FIFO_DEPTH = 8,
    parameter int OUT_WORD_Q_DEPTH = (FIFO_DEPTH < 4) ? 4 : FIFO_DEPTH
) (
    input  wire logic             clk_i,
    input  wire logic             rst_i,

    input  wire logic             clear_error_i,
    input  wire npu_mvout_mode_e  mvout_mode_i,
    input  wire logic             per_channel_scale_i,
    input  wire logic [31:0]      tensor_scale_q8_24_i,

    input  wire logic             co_valid_i,
    output logic             co_ready_o,
    input  wire logic [127:0]     co_acc_data_i,
    input  wire logic [127:0]     scale_data_i,

    output logic             dma_valid_o,
    input  wire logic             dma_ready_i,
    output logic [127:0]     dma_data_o,

    output logic             fifo_almost_full_o,
    output logic             error_sticky_o,
    output npu_error_e       last_error_o
);
    logic         fp32_launch;
    logic         fp32_stage_load;

    localparam int CONVERT_LANES = 4;

    logic [CONVERT_LANES-1:0] fp32_valid;
    logic [CONVERT_LANES-1:0][31:0] fp32_lane;

    localparam int Q_PTR_W = (OUT_WORD_Q_DEPTH <= 1) ? 1 : $clog2(OUT_WORD_Q_DEPTH);
    localparam int Q_CNT_W = $clog2(OUT_WORD_Q_DEPTH + 1);
    localparam logic [Q_CNT_W-1:0] OUT_WORD_Q_DEPTH_COUNT = OUT_WORD_Q_DEPTH;
    localparam logic [Q_CNT_W-1:0] OUT_WORD_Q_ALMOST_COUNT =
        (OUT_WORD_Q_DEPTH > 1) ? Q_CNT_W'(OUT_WORD_Q_DEPTH - 1) : '0;

    logic [127:0] out_word_q [0:OUT_WORD_Q_DEPTH-1];
    logic [Q_PTR_W-1:0] out_wr_ptr_q;
    logic [Q_PTR_W-1:0] out_rd_ptr_q;
    logic [Q_CNT_W-1:0] out_count_q;
    logic [Q_CNT_W-1:0] fp32_inflight_count_q;
    logic [Q_CNT_W:0] fp32_total_pending;
    logic fp32_stage_valid_q;
    logic fp32_stage_per_channel_scale_q;
    logic [31:0] fp32_stage_tensor_scale_q8_24_q;
    logic [127:0] fp32_stage_acc_data_q;
    logic [127:0] fp32_stage_scale_data_q;

    logic input_fire;
    logic mode_raw;
    logic mode_fp32;
    logic fp32_done;
    logic fp32_done_q;
    logic [127:0] fp32_done_word_q;
    logic output_word_enqueue;
    logic [127:0] output_word_enqueue_data;
    logic output_word_dequeue;
    logic unsupported_input_fire;

    assign mode_raw = (mvout_mode_i == NPU_MVOUT_RAW_I32);
    assign mode_fp32 = (mvout_mode_i == NPU_MVOUT_FP32_Q8_24);
    assign fp32_done = &fp32_valid;
    assign fp32_total_pending = {1'b0, out_count_q} +
                                {1'b0, fp32_inflight_count_q} +
                                {{Q_CNT_W{1'b0}}, fp32_stage_valid_q} +
                                {{Q_CNT_W{1'b0}}, fp32_done_q};
    assign co_ready_o = !(fp32_done && mode_raw) &&
                        (fp32_total_pending < {1'b0, OUT_WORD_Q_DEPTH_COUNT}) &&
                        !error_sticky_o;
    assign input_fire = co_valid_i && co_ready_o;
    assign unsupported_input_fire = input_fire && !mode_raw && !mode_fp32;
    assign fp32_stage_load = input_fire && mode_fp32;
    assign fp32_launch = fp32_stage_valid_q;

    genvar lane;
    generate
        for (lane = 0; lane < CONVERT_LANES; lane = lane + 1) begin : gen_converters
            logic [31:0] selected_scale;
            assign selected_scale = fp32_stage_per_channel_scale_q
                ? fp32_stage_scale_data_q[lane*32 +: 32]
                : fp32_stage_tensor_scale_q8_24_q;

            int32_q8_24_to_fp32 u_convert (
                .clk_i           (clk_i),
                .rst_i           (rst_i),
                .valid_i         (fp32_launch),
                .acc_i           (fp32_stage_acc_data_q[lane*32 +: 32]),
                .scale_q8_24_i   (selected_scale),
                .valid_o         (fp32_valid[lane]),
                .fp32_o          (fp32_lane[lane])
            );
        end
    endgenerate

    assign output_word_enqueue = (input_fire && mode_raw) || fp32_done_q;
    assign output_word_enqueue_data = (input_fire && mode_raw)
        ? co_acc_data_i
        : fp32_done_word_q;

    assign dma_valid_o = (out_count_q != '0);
    assign dma_data_o = out_word_q[out_rd_ptr_q];
    assign output_word_dequeue = dma_valid_o && dma_ready_i;
    assign fifo_almost_full_o = (fp32_total_pending >= {1'b0, OUT_WORD_Q_ALMOST_COUNT});

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            out_wr_ptr_q          <= '0;
            out_rd_ptr_q          <= '0;
            out_count_q           <= '0;
            fp32_inflight_count_q <= '0;
            fp32_stage_valid_q    <= 1'b0;
            fp32_stage_per_channel_scale_q <= 1'b0;
            fp32_stage_tensor_scale_q8_24_q <= '0;
            fp32_stage_acc_data_q <= '0;
            fp32_stage_scale_data_q <= '0;
            fp32_done_q           <= 1'b0;
            fp32_done_word_q      <= '0;
            error_sticky_o        <= 1'b0;
            last_error_o          <= NPU_ERR_NONE;
        end else begin
            fp32_stage_valid_q <= fp32_stage_load;
            if (fp32_stage_load) begin
                fp32_stage_per_channel_scale_q <= per_channel_scale_i;
                fp32_stage_tensor_scale_q8_24_q <= tensor_scale_q8_24_i;
                fp32_stage_acc_data_q <= co_acc_data_i;
                fp32_stage_scale_data_q <= scale_data_i;
            end

            fp32_done_q <= fp32_done;
            if (fp32_done) begin
                fp32_done_word_q <= {fp32_lane[3], fp32_lane[2], fp32_lane[1], fp32_lane[0]};
            end

            if (clear_error_i) begin
                error_sticky_o <= 1'b0;
                last_error_o   <= NPU_ERR_NONE;
            end

            if ((output_word_enqueue && (out_count_q >= OUT_WORD_Q_DEPTH_COUNT) &&
                 !output_word_dequeue)) begin
                error_sticky_o <= 1'b1;
                last_error_o   <= NPU_ERR_FIFO;
            end

            if (unsupported_input_fire) begin
                error_sticky_o <= 1'b1;
                last_error_o   <= NPU_ERR_UNSUPPORTED_MODE;
            end

            if (output_word_enqueue) begin
                out_word_q[out_wr_ptr_q] <= output_word_enqueue_data;
                if (out_wr_ptr_q == Q_PTR_W'(OUT_WORD_Q_DEPTH - 1)) begin
                    out_wr_ptr_q <= '0;
                end else begin
                    out_wr_ptr_q <= out_wr_ptr_q + Q_PTR_W'(1);
                end
            end

            if (output_word_dequeue) begin
                if (out_rd_ptr_q == Q_PTR_W'(OUT_WORD_Q_DEPTH - 1)) begin
                    out_rd_ptr_q <= '0;
                end else begin
                    out_rd_ptr_q <= out_rd_ptr_q + Q_PTR_W'(1);
                end
            end

            unique case ({output_word_enqueue, output_word_dequeue})
                2'b10: out_count_q <= out_count_q + Q_CNT_W'(1);
                2'b01: out_count_q <= out_count_q - Q_CNT_W'(1);
                default: out_count_q <= out_count_q;
            endcase

            unique case ({fp32_launch, fp32_done})
                2'b10: fp32_inflight_count_q <= fp32_inflight_count_q + Q_CNT_W'(1);
                2'b01: fp32_inflight_count_q <= fp32_inflight_count_q - Q_CNT_W'(1);
                default: fp32_inflight_count_q <= fp32_inflight_count_q;
            endcase
        end
    end
endmodule

`default_nettype wire
