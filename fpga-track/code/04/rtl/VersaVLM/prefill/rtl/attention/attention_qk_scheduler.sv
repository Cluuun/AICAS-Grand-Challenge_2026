`default_nettype none

import npu_spm_pkg::*;

// Attention-QK phase-1 scheduler prototype.
//
// Scope of this version:
//   - Computes a configured Q-row chunk against all K blocks in the sequence.
//   - Produces QK score/logP codes only; PV/value aggregation is handled by a
//     separate path and is not part of this instruction.
//   - Q is read from A0 starting at word address 0.
//   - K is read from W0 starting at word address 0. W0 must already use the
//     GEMM W layout: for each K block, word r contains K[block*32+0..31][r].
//   - Each 32x32 QK tile is streamed from the SA shadow registers, scaled to
//     a signed 8-bit log-domain score code S_q, row-maxed, and converted to
//     local distance codes D_local = m_block - S_q.
//   - O bank stores D_local scratch at row_in_chunk*num_k_blocks+kb.
//   - Metadata stores one signed int8 m_block byte per row/k-block.
//   - m_global is emitted as a datapath register write. Final correction/logP
//     conversion is a separate MVOUT mode.
module attention_qk_scheduler #(
    parameter int ADDR_W          = 14,
    parameter int META_ADDR_W     = 11,
    parameter int O_BANK_DEPTH_WORDS = 16384,
    parameter int GAMMA16_FRAC    = 24,
    parameter bit DEBUG_PRINT     = 1'b0
) (
    input  wire logic              clk_i,
    input  wire logic              rst_i,
    input  wire logic              clear_error_i,

    input  wire logic              start_i,
    output logic                   ready_o,
    output logic                   busy_o,
    output logic                   done_o,

    input  wire logic [15:0]       token_count_i,
    input  wire logic [31:0]       gamma16_fix_i,
    input  wire logic              mask_en_i,
    input  wire logic [4:0]        q_block_start_i,
    input  wire logic [4:0]        q_block_count_m1_i,

    output logic                   core_cfg_start_o,
    input  wire logic              core_cfg_ready_i,
    output logic [ADDR_W-1:0]      core_a_base_word_addr_o,
    output logic [ADDR_W-1:0]      core_a_k_word_offset_o,
    output logic [ADDR_W-1:0]      core_a_row_stride_words_o,
    output logic [ADDR_W-1:0]      core_w_base_word_addr_o,
    output logic [5:0]             core_m_count_o,
    output logic [5:0]             core_n_count_o,
    output logic [12:0]            core_k_count_o,
    output npu_mode_e              core_mode_o,
    output logic                   core_snapshot_ready_o,
    input  wire logic              core_busy_i,
    input  wire logic              core_done_i,
    input  wire logic              core_error_sticky_i,
    input  wire npu_error_e        core_last_error_i,
    input  wire logic              core_co_valid_i,
    output logic                   core_co_ready_o,
    input  wire logic [255:0]      core_co_data_i,
    input  wire logic [7:0]        core_co_lane_mask_i,

    output logic                   o0_wr_req_valid_o,
    input  wire logic              o0_wr_req_ready_i,
    output logic [ADDR_W-1:0]      o0_wr_req_word_addr_o,
    output logic [255:0]           o0_wr_req_data_o,
    output logic [31:0]            o0_wr_req_byte_en_o,

    output logic                   meta_wr_valid_o,
    input  wire logic              meta_wr_ready_i,
    output logic [META_ADDR_W-1:0] meta_wr_word128_addr_o,
    output logic [127:0]           meta_wr_data_o,
    output logic [15:0]            meta_wr_keep_o,

    output logic                   m_global_wr_valid_o,
    output logic [8:0]             m_global_wr_row_o,
    output logic signed [15:0]     m_global_wr_data_o,

    output logic                   error_sticky_o,
    output npu_error_e             last_error_o
);
    localparam int TILE_M       = 32;
    localparam int TILE_N       = 32;
    localparam int TILE_K       = 64;
    localparam int LANES_I32    = 8;
    localparam int DRAIN_WORDS  = 128;

    localparam logic [63:0] GAMMA_ROUND_HALF = 64'd1 << (GAMMA16_FRAC - 1);
    localparam logic signed [15:0] SCORE_MIN = 16'sh8000;

    typedef enum logic [1:0] {
        PH_IDLE,
        PH_VALIDATE,
        PH_RUN
    } phase_e;

    initial begin
        if (GAMMA16_FRAC <= 0 || GAMMA16_FRAC >= 63) begin
            $error("attention_qk_scheduler: GAMMA16_FRAC must be in 1..62");
        end
    end

    logic gemm_cfg_start_q;
    logic gemm_cfg_ready;
    logic [ADDR_W-1:0] gemm_a_base_word_addr_q;
    logic [ADDR_W-1:0] gemm_w_base_word_addr_q;
    logic gemm_snapshot_ready;
    logic gemm_co_valid;
    logic gemm_co_ready;
    logic [255:0] gemm_co_data;
    logic [7:0] gemm_co_lane_mask;
    logic gemm_core_busy;
    logic gemm_core_done;
    logic gemm_error_sticky;
    npu_error_e gemm_last_error;

    phase_e phase_q;
    logic [15:0] token_count_q;
    logic [5:0] num_q_blocks_q;
    logic [5:0] num_k_blocks_q;
    logic [31:0] gamma16_fix_q;
    logic mask_en_q;
    logic [4:0] q_block_start_q;
    logic [5:0] q_block_count_q;

    logic compute_inflight_q;
    logic [4:0] active_qb_q;
    logic [4:0] compute_qb_q;
    logic [4:0] compute_kb_q;
    logic [5:0] next_launch_kb_q;
    logic [5:0] completed_tiles_q;
    logic shadow_busy_q;
    logic [4:0] drain_qb_q;
    logic [4:0] drain_kb_q;
    logic [7:0] stream_word_idx_q;
    logic [14:0] output_write_count_q;
    logic [10:0] qblock_output_write_count_q;
    logic [10:0] total_output_words_q;
    logic [14:0] chunk_total_output_words_q;

    logic row_out_valid_q;
    logic [ADDR_W-1:0] row_out_addr_q;
    logic [255:0] row_out_data_q;
    logic [META_ADDR_W-1:0] meta_out_addr_q;
    logic [127:0] meta_out_data_q;
    logic [15:0] meta_out_keep_q;
    logic [8:0] m_global_out_row_q;
    logic signed [15:0] m_global_out_data_q;
    logic signed [15:0] row_score_q [0:TILE_N-1];
    logic signed [15:0] m_global_q [0:TILE_M-1];

    logic s0_valid_q;
    logic signed [15:0] s0_score_q [0:TILE_N-1];
    logic [4:0] s0_row_q;
    logic [4:0] s0_qb_q;
    logic [4:0] s0_kb_q;
    logic [8:0] s0_in_chunk_q;
    logic [14:0] s0_meta_byte_index_q;

    logic s1_valid_q;
    logic signed [15:0] s1_score_q [0:TILE_N-1];
    logic [5:0] s1_valid_count_q;
    logic [4:0] s1_row_q;
    logic [4:0] s1_kb_q;
    logic [8:0] s1_in_chunk_q;
    logic [14:0] s1_meta_byte_index_q;

    logic s2_valid_q;
    logic signed [15:0] s2_score_q [0:TILE_N-1];
    logic [5:0] s2_valid_count_q;
    logic signed [15:0] s2_group_max_q [0:7];
    logic s2_group_valid_q [0:7];
    logic [4:0] s2_row_q;
    logic [4:0] s2_kb_q;
    logic [8:0] s2_in_chunk_q;
    logic [14:0] s2_meta_byte_index_q;

    logic s3_valid_q;
    logic signed [15:0] s3_score_q [0:TILE_N-1];
    logic [5:0] s3_valid_count_q;
    logic signed [15:0] s3_pair_max_q [0:3];
    logic s3_pair_valid_q [0:3];
    logic [4:0] s3_row_q;
    logic [4:0] s3_kb_q;
    logic [8:0] s3_in_chunk_q;
    logic [14:0] s3_meta_byte_index_q;

    logic s4_valid_q;
    logic signed [15:0] s4_score_q [0:TILE_N-1];
    logic [5:0] s4_valid_count_q;
    logic signed [15:0] s4_m_block_q;
    logic s4_has_valid_q;
    logic [4:0] s4_row_q;
    logic [4:0] s4_kb_q;
    logic [8:0] s4_in_chunk_q;
    logic [14:0] s4_meta_byte_index_q;

    logic post_valid_q;
    logic [ADDR_W-1:0] post_addr_q;
    logic [255:0] post_data_q;
    logic [META_ADDR_W-1:0] post_meta_addr_q;
    logic [127:0] post_meta_data_q;
    logic [15:0] post_meta_keep_q;
    logic [8:0] post_m_global_row_q;
    logic signed [15:0] post_m_global_data_q;

    logic local_error_sticky_q;
    npu_error_e local_last_error_q;

    logic signed [31:0] lane_dot [0:LANES_I32-1];
    logic signed [15:0] lane_score [0:LANES_I32-1];
    logic score_s1_valid_q;
    logic signed [31:0] score_s1_dot_q [0:LANES_I32-1];
    logic signed [63:0] score_s1_product_q [0:LANES_I32-1];
    logic [1:0] score_s1_group_id_q;
    logic [4:0] score_s1_row_id_q;
    logic [4:0] score_s1_qb_q;
    logic [4:0] score_s1_kb_q;
    logic score_s1_row_complete_q;
    logic score_s1_last_q;
    logic [8:0] score_s1_row_in_chunk_q;
    logic [14:0] score_s1_meta_byte_index_q;
    logic score_round_valid_q;
    logic signed [15:0] score_round_lane_q [0:LANES_I32-1];
    logic [1:0] score_round_group_id_q;
    logic [4:0] score_round_row_id_q;
    logic [4:0] score_round_qb_q;
    logic [4:0] score_round_kb_q;
    logic score_round_row_complete_q;
    logic [8:0] score_round_row_in_chunk_q;
    logic [14:0] score_round_meta_byte_index_q;
    logic score_s1_can_load;
    logic score_s1_consume;
    logic score_round_can_load;
    logic score_round_consume;
    logic signed [15:0] row_score_work [0:TILE_N-1];
    logic signed [16:0] row_d_work [0:TILE_N-1];
    logic [255:0] row_d_data;
    logic [8:0] stream_row_in_chunk;
    logic [14:0] stream_meta_byte_index;
    logic [5:0] s1_valid_count_comb;
    logic signed [15:0] s2_group_max_comb [0:7];
    logic s2_group_valid_comb [0:7];
    logic signed [15:0] s3_pair_max_comb [0:3];
    logic s3_pair_valid_comb [0:3];
    logic signed [15:0] s4_m_block_comb;
    logic s4_has_valid_comb;
    logic signed [15:0] post_m_global_next;

    wire [5:0] token_num_k_blocks = token_count_i[10:5];
    wire [5:0] requested_q_block_count = {1'b0, q_block_count_m1_i} + 6'd1;
    wire cfg_token_count_legal =
        (token_count_q >= 16'd32) &&
        (token_count_q <= 16'd1024) &&
        (token_count_q[4:0] == 5'd0);
    wire [5:0] cfg_q_block_end = {1'b0, q_block_start_q} + q_block_count_q;
    wire [9:0] cfg_output_tile_words =
        {4'd0, q_block_count_q} * {4'd0, num_k_blocks_q};
    wire cfg_chunk_shape_legal =
        cfg_token_count_legal &&
        (cfg_q_block_end <= num_k_blocks_q) &&
        (cfg_output_tile_words <= 10'(O_BANK_DEPTH_WORDS >> 5));
    wire start_fire = start_i && ready_o;
    wire row_out_fire = row_out_valid_q && o0_wr_req_ready_i && meta_wr_ready_i;
    wire row_out_can_load = !row_out_valid_q || row_out_fire;
    wire pipe_can_advance = !post_valid_q || row_out_can_load;
    wire stream_can_accept = (phase_q == PH_RUN) &&
                             pipe_can_advance;
    wire co_fire = gemm_co_valid && gemm_co_ready;
    wire co_last_fire = co_fire && (stream_word_idx_q == 8'(DRAIN_WORDS - 1));
    wire [4:0] stream_row_id = stream_word_idx_q[6:2];
    wire [1:0] stream_group_id = stream_word_idx_q[1:0];
    wire stream_row_complete = co_fire && (stream_group_id == 2'd3);
    wire [5:0] active_qb_rel = {1'b0, active_qb_q} - {1'b0, q_block_start_q};
    wire last_output_write_fire =
        row_out_fire && (output_write_count_q == (chunk_total_output_words_q - 15'd1));
    wire last_qblock_write_fire =
        row_out_fire && (qblock_output_write_count_q == (total_output_words_q - 11'd1));
    wire launch_next_tile =
        (phase_q == PH_RUN) &&
        !compute_inflight_q &&
        (next_launch_kb_q < num_k_blocks_q) &&
        gemm_cfg_ready;

    assign gemm_snapshot_ready = !shadow_busy_q || co_last_fire;

    assign ready_o = (phase_q == PH_IDLE) &&
                     !local_error_sticky_q;
    assign busy_o = (phase_q != PH_IDLE) ||
                    shadow_busy_q || row_out_valid_q || post_valid_q ||
                    score_s1_valid_q || score_round_valid_q ||
                    s0_valid_q || s1_valid_q || s2_valid_q || s3_valid_q ||
                    s4_valid_q;
    assign error_sticky_o = local_error_sticky_q || gemm_error_sticky;
    assign last_error_o = local_error_sticky_q ? local_last_error_q : gemm_last_error;

    assign o0_wr_req_valid_o = row_out_valid_q && meta_wr_ready_i;
    assign o0_wr_req_word_addr_o = row_out_addr_q;
    assign o0_wr_req_data_o = row_out_data_q;
    assign o0_wr_req_byte_en_o = 32'hffff_ffff;

    assign meta_wr_valid_o = row_out_valid_q && o0_wr_req_ready_i;
    assign meta_wr_word128_addr_o = meta_out_addr_q;
    assign meta_wr_data_o = meta_out_data_q;
    assign meta_wr_keep_o = meta_out_keep_q;

    assign m_global_wr_valid_o = row_out_fire;
    assign m_global_wr_row_o = m_global_out_row_q;
    assign m_global_wr_data_o = m_global_out_data_q;

    assign score_round_consume = score_round_valid_q && stream_can_accept;
    assign score_round_can_load = !score_round_valid_q || score_round_consume;
    assign score_s1_consume = score_s1_valid_q && score_round_can_load;
    assign score_s1_can_load = !score_s1_valid_q || score_s1_consume;
    assign gemm_co_ready = stream_can_accept && score_s1_can_load;
    assign core_cfg_start_o = gemm_cfg_start_q;
    assign gemm_cfg_ready = core_cfg_ready_i;
    assign core_a_base_word_addr_o = gemm_a_base_word_addr_q;
    assign core_a_k_word_offset_o = '0;
    assign core_a_row_stride_words_o = ADDR_W'(2);
    assign core_w_base_word_addr_o = gemm_w_base_word_addr_q;
    assign core_m_count_o = 6'd32;
    assign core_n_count_o = 6'd32;
    assign core_k_count_o = 13'd64;
    assign core_mode_o = NPU_MODE_INT8;
    assign core_snapshot_ready_o = gemm_snapshot_ready;
    assign gemm_core_busy = core_busy_i;
    assign gemm_core_done = core_done_i;
    assign gemm_error_sticky = core_error_sticky_i;
    assign gemm_last_error = core_last_error_i;
    assign gemm_co_valid = core_co_valid_i;
    assign core_co_ready_o = gemm_co_ready;
    assign gemm_co_data = core_co_data_i;
    assign gemm_co_lane_mask = core_co_lane_mask_i;

    function automatic logic signed [63:0] score_gamma_product(
        input logic signed [31:0] dot_i,
        input logic [31:0] gamma16_fix_i
    );
        logic signed [63:0] dot_ext;
        logic signed [63:0] gamma_ext;
        begin
            dot_ext = {{32{dot_i[31]}}, dot_i};
            gamma_ext = {32'd0, gamma16_fix_i};
            score_gamma_product = dot_ext * gamma_ext;
        end
    endfunction

    function automatic logic signed [15:0] round_score_product(
        input logic signed [63:0] product
    );
        logic signed [63:0] neg_product;
        logic [63:0] abs_product;
        logic [63:0] rounded_abs;
        begin
            if (product[63]) begin
                neg_product = -product;
                abs_product = neg_product[63:0];
                rounded_abs = (abs_product + GAMMA_ROUND_HALF) >> GAMMA16_FRAC;
                if (rounded_abs >= 64'd32768) begin
                    round_score_product = 16'sh8000;
                end else begin
                    round_score_product = -$signed(rounded_abs[15:0]);
                end
            end else begin
                rounded_abs = (product[63:0] + GAMMA_ROUND_HALF) >> GAMMA16_FRAC;
                if (rounded_abs >= 64'd32768) begin
                    round_score_product = 16'sh7fff;
                end else begin
                    round_score_product = $signed(rounded_abs[15:0]);
                end
            end
        end
    endfunction

    function automatic logic [7:0] d_to_u8(
        input logic valid_i,
        input logic signed [31:0] d_i
    );
        begin
            if (!valid_i) begin
                d_to_u8 = 8'hff;
            end else if (d_i <= 32'sd0) begin
                d_to_u8 = 8'd0;
            end else if (d_i >= 32'sd255) begin
                d_to_u8 = 8'hff;
            end else begin
                d_to_u8 = d_i[7:0];
            end
        end
    endfunction

    always_comb begin
        for (int lane = 0; lane < LANES_I32; lane++) begin
            lane_dot[lane] = $signed(gemm_co_data[lane*32 +: 32]);
            lane_score[lane] = round_score_product(score_s1_product_q[lane]);
        end

        for (int col = 0; col < TILE_N; col++) begin
            row_score_work[col] = row_score_q[col];
        end

        if (score_round_consume) begin
            for (int lane = 0; lane < LANES_I32; lane++) begin
                row_score_work[{score_round_group_id_q, 3'b000} + lane] =
                    score_round_lane_q[lane];
            end
        end

        stream_row_in_chunk =
            ({3'd0, (drain_qb_q - q_block_start_q)} << 5) + {4'd0, stream_row_id};
        stream_meta_byte_index =
            ({6'd0, stream_row_in_chunk} * {9'd0, num_k_blocks_q}) +
            {10'd0, drain_kb_q};

        if (!mask_en_q) begin
            s1_valid_count_comb = 6'd32;
        end else if ((({11'd0, s0_qb_q} << 5) + {11'd0, s0_row_q}) <
                     ({11'd0, s0_kb_q} << 5)) begin
            s1_valid_count_comb = 6'd0;
        end else if ((({11'd0, s0_qb_q} << 5) + {11'd0, s0_row_q}) >=
                     (({11'd0, s0_kb_q} << 5) + 16'd31)) begin
            s1_valid_count_comb = 6'd32;
        end else begin
            s1_valid_count_comb =
                6'((({11'd0, s0_qb_q} << 5) + {11'd0, s0_row_q}) -
                   ({11'd0, s0_kb_q} << 5) + 16'd1);
        end

        for (int group = 0; group < 8; group++) begin
            s2_group_max_comb[group] = SCORE_MIN;
            s2_group_valid_comb[group] = 1'b0;
            for (int lane = 0; lane < 4; lane++) begin
                int col;
                col = group * 4 + lane;
                if ((6'(col) < s1_valid_count_q) &&
                    (!s2_group_valid_comb[group] ||
                     (s1_score_q[col] > s2_group_max_comb[group]))) begin
                    s2_group_max_comb[group] = s1_score_q[col];
                    s2_group_valid_comb[group] = 1'b1;
                end
            end
        end

        for (int pair = 0; pair < 4; pair++) begin
            s3_pair_max_comb[pair] = SCORE_MIN;
            s3_pair_valid_comb[pair] =
                s2_group_valid_q[pair*2] || s2_group_valid_q[pair*2 + 1];
            if (s2_group_valid_q[pair*2] &&
                (!s2_group_valid_q[pair*2 + 1] ||
                 (s2_group_max_q[pair*2] >= s2_group_max_q[pair*2 + 1]))) begin
                s3_pair_max_comb[pair] = s2_group_max_q[pair*2];
            end else if (s2_group_valid_q[pair*2 + 1]) begin
                s3_pair_max_comb[pair] = s2_group_max_q[pair*2 + 1];
            end
        end

        s4_m_block_comb = SCORE_MIN;
        s4_has_valid_comb = 1'b0;
        for (int pair = 0; pair < 4; pair++) begin
            if (s3_pair_valid_q[pair] &&
                (!s4_has_valid_comb || (s3_pair_max_q[pair] > s4_m_block_comb))) begin
                s4_m_block_comb = s3_pair_max_q[pair];
                s4_has_valid_comb = 1'b1;
            end
        end

        row_d_data = '0;
        for (int col = 0; col < TILE_N; col++) begin
            row_d_work[col] = (s4_has_valid_q && (6'(col) < s4_valid_count_q))
                ? ({s4_m_block_q[15], s4_m_block_q} -
                   {s4_score_q[col][15], s4_score_q[col]})
                : 17'sd255;
            row_d_data[col*8 +: 8] =
                d_to_u8(s4_has_valid_q && (6'(col) < s4_valid_count_q),
                        row_d_work[col]);
        end

        post_m_global_next = (s4_m_block_q > m_global_q[s4_row_q])
            ? s4_m_block_q
            : m_global_q[s4_row_q];
    end

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            phase_q <= PH_IDLE;
            done_o <= 1'b0;
            gemm_cfg_start_q <= 1'b0;
            token_count_q <= '0;
            num_q_blocks_q <= '0;
            num_k_blocks_q <= '0;
            gamma16_fix_q <= '0;
            mask_en_q <= 1'b0;
            q_block_start_q <= '0;
            q_block_count_q <= '0;
            compute_inflight_q <= 1'b0;
            active_qb_q <= '0;
            compute_qb_q <= '0;
            compute_kb_q <= '0;
            next_launch_kb_q <= '0;
            completed_tiles_q <= '0;
            shadow_busy_q <= 1'b0;
            drain_qb_q <= '0;
            drain_kb_q <= '0;
            stream_word_idx_q <= '0;
            output_write_count_q <= '0;
            qblock_output_write_count_q <= '0;
            total_output_words_q <= '0;
            chunk_total_output_words_q <= '0;
            row_out_valid_q <= 1'b0;
            score_s1_valid_q <= 1'b0;
            score_round_valid_q <= 1'b0;
            score_s1_row_complete_q <= 1'b0;
            score_s1_last_q <= 1'b0;
            score_round_row_complete_q <= 1'b0;
            s0_valid_q <= 1'b0;
            s1_valid_q <= 1'b0;
            s2_valid_q <= 1'b0;
            s3_valid_q <= 1'b0;
            s4_valid_q <= 1'b0;
            s4_has_valid_q <= 1'b0;
            post_valid_q <= 1'b0;
            local_error_sticky_q <= 1'b0;
            local_last_error_q <= NPU_ERR_NONE;
            for (int group = 0; group < 8; group++) begin
                s2_group_valid_q[group] <= 1'b0;
            end
            for (int pair = 0; pair < 4; pair++) begin
                s3_pair_valid_q[pair] <= 1'b0;
            end
        end else begin
            done_o <= 1'b0;
            gemm_cfg_start_q <= 1'b0;

            if (clear_error_i) begin
                local_error_sticky_q <= 1'b0;
                local_last_error_q <= NPU_ERR_NONE;
            end

            if (row_out_fire) begin
                row_out_valid_q <= 1'b0;
                output_write_count_q <= output_write_count_q + 11'd1;
                if (last_output_write_fire) begin
                    phase_q <= PH_IDLE;
                    done_o <= 1'b1;
                    qblock_output_write_count_q <= '0;
                end else if (last_qblock_write_fire) begin
                    qblock_output_write_count_q <= '0;
                    phase_q <= PH_RUN;
                    active_qb_q <= active_qb_q + 5'd1;
                    gemm_cfg_start_q <= 1'b1;
                    gemm_a_base_word_addr_q <=
                        ADDR_W'(({6'd0, active_qb_q} + 11'd1) * 11'(TILE_M * 2));
                    gemm_w_base_word_addr_q <= '0;
                    compute_inflight_q <= 1'b1;
                    compute_qb_q <= active_qb_q + 5'd1;
                    compute_kb_q <= '0;
                    next_launch_kb_q <= 6'd1;
                    completed_tiles_q <= '0;
                    shadow_busy_q <= 1'b0;
                    drain_qb_q <= active_qb_q + 5'd1;
                    drain_kb_q <= '0;
                    stream_word_idx_q <= '0;
                    score_s1_valid_q <= 1'b0;
                    score_round_valid_q <= 1'b0;
                    s0_valid_q <= 1'b0;
                    s1_valid_q <= 1'b0;
                    s2_valid_q <= 1'b0;
                    s3_valid_q <= 1'b0;
                    s4_valid_q <= 1'b0;
                    post_valid_q <= 1'b0;
                    for (int col = 0; col < TILE_N; col++) begin
                        row_score_q[col] <= '0;
                    end
                    for (int row = 0; row < TILE_M; row++) begin
                        m_global_q[row] <= SCORE_MIN;
                    end
                end else begin
                    qblock_output_write_count_q <= qblock_output_write_count_q + 11'd1;
                end
            end

            if (post_valid_q && row_out_can_load) begin
                row_out_valid_q <= 1'b1;
                row_out_addr_q <= post_addr_q;
                row_out_data_q <= post_data_q;
                meta_out_addr_q <= post_meta_addr_q;
                meta_out_data_q <= post_meta_data_q;
                meta_out_keep_q <= post_meta_keep_q;
                m_global_out_row_q <= post_m_global_row_q;
                m_global_out_data_q <= post_m_global_data_q;
                m_global_q[post_m_global_row_q[4:0]] <= post_m_global_data_q;
            end

            if (pipe_can_advance) begin
                post_valid_q <= s4_valid_q;
                if (s4_valid_q) begin
                    post_addr_q <=
                        ADDR_W'(({6'd0, s4_in_chunk_q} * {5'd0, num_k_blocks_q}) +
                                {6'd0, s4_kb_q});
                    post_data_q <= row_d_data;
                    post_meta_addr_q <= s4_meta_byte_index_q[META_ADDR_W+2:3];
                    post_meta_data_q <= '0;
                    post_meta_data_q[s4_meta_byte_index_q[2:0]*16 +: 16] <= s4_m_block_q;
                    post_meta_keep_q <= 16'(16'h0003 << {s4_meta_byte_index_q[2:0], 1'b0});
                    post_m_global_row_q <= s4_in_chunk_q;
                    post_m_global_data_q <= post_m_global_next;
                end

                s4_valid_q <= s3_valid_q;
                s4_valid_count_q <= s3_valid_count_q;
                s4_m_block_q <= s4_m_block_comb;
                s4_has_valid_q <= s4_has_valid_comb;
                s4_row_q <= s3_row_q;
                s4_kb_q <= s3_kb_q;
                s4_in_chunk_q <= s3_in_chunk_q;
                s4_meta_byte_index_q <= s3_meta_byte_index_q;
                for (int col = 0; col < TILE_N; col++) begin
                    s4_score_q[col] <= s3_score_q[col];
                end

                s3_valid_q <= s2_valid_q;
                s3_valid_count_q <= s2_valid_count_q;
                s3_row_q <= s2_row_q;
                s3_kb_q <= s2_kb_q;
                s3_in_chunk_q <= s2_in_chunk_q;
                s3_meta_byte_index_q <= s2_meta_byte_index_q;
                for (int pair = 0; pair < 4; pair++) begin
                    s3_pair_max_q[pair] <= s3_pair_max_comb[pair];
                    s3_pair_valid_q[pair] <= s3_pair_valid_comb[pair];
                end
                for (int col = 0; col < TILE_N; col++) begin
                    s3_score_q[col] <= s2_score_q[col];
                end

                s2_valid_q <= s1_valid_q;
                s2_valid_count_q <= s1_valid_count_q;
                s2_row_q <= s1_row_q;
                s2_kb_q <= s1_kb_q;
                s2_in_chunk_q <= s1_in_chunk_q;
                s2_meta_byte_index_q <= s1_meta_byte_index_q;
                for (int group = 0; group < 8; group++) begin
                    s2_group_max_q[group] <= s2_group_max_comb[group];
                    s2_group_valid_q[group] <= s2_group_valid_comb[group];
                end
                for (int col = 0; col < TILE_N; col++) begin
                    s2_score_q[col] <= s1_score_q[col];
                end

                s1_valid_q <= s0_valid_q;
                s1_valid_count_q <= s1_valid_count_comb;
                s1_row_q <= s0_row_q;
                s1_kb_q <= s0_kb_q;
                s1_in_chunk_q <= s0_in_chunk_q;
                s1_meta_byte_index_q <= s0_meta_byte_index_q;
                for (int col = 0; col < TILE_N; col++) begin
                    s1_score_q[col] <= s0_score_q[col];
                end

                s0_valid_q <= 1'b0;
            end

            if (score_round_consume) begin
                for (int lane = 0; lane < LANES_I32; lane++) begin
                    row_score_q[{score_round_group_id_q, 3'b000} + lane] <=
                        score_round_lane_q[lane];
                end
                if (score_round_row_complete_q) begin
                    for (int col = 0; col < TILE_N; col++) begin
                        if ((col / LANES_I32) == int'(score_round_group_id_q)) begin
                            s0_score_q[col] <= score_round_lane_q[col % LANES_I32];
                        end else begin
                            s0_score_q[col] <= row_score_q[col];
                        end
                    end
                    s0_valid_q <= 1'b1;
                    s0_row_q <= score_round_row_id_q;
                    s0_qb_q <= score_round_qb_q;
                    s0_kb_q <= score_round_kb_q;
                    s0_in_chunk_q <= score_round_row_in_chunk_q;
                    s0_meta_byte_index_q <= score_round_meta_byte_index_q;
                end
                score_round_valid_q <= 1'b0;
            end

            if (score_s1_consume) begin
                score_round_valid_q <= 1'b1;
                for (int lane = 0; lane < LANES_I32; lane++) begin
                    score_round_lane_q[lane] <= lane_score[lane];
                end
                score_round_group_id_q <= score_s1_group_id_q;
                score_round_row_id_q <= score_s1_row_id_q;
                score_round_qb_q <= score_s1_qb_q;
                score_round_kb_q <= score_s1_kb_q;
                score_round_row_complete_q <= score_s1_row_complete_q;
                score_round_row_in_chunk_q <= score_s1_row_in_chunk_q;
                score_round_meta_byte_index_q <= score_s1_meta_byte_index_q;
                score_s1_valid_q <= 1'b0;
            end

            if (co_fire) begin
                score_s1_valid_q <= 1'b1;
                for (int lane = 0; lane < LANES_I32; lane++) begin
                    score_s1_dot_q[lane] <= $signed(gemm_co_data[lane*32 +: 32]);
                    score_s1_product_q[lane] <=
                        score_gamma_product($signed(gemm_co_data[lane*32 +: 32]),
                                            gamma16_fix_q);
                end
                score_s1_group_id_q <= stream_group_id;
                score_s1_row_id_q <= stream_row_id;
                score_s1_qb_q <= drain_qb_q;
                score_s1_kb_q <= drain_kb_q;
                score_s1_row_complete_q <= stream_row_complete;
                score_s1_last_q <= co_last_fire;
                score_s1_row_in_chunk_q <= stream_row_in_chunk;
                score_s1_meta_byte_index_q <= stream_meta_byte_index;
                if (co_last_fire) begin
                    stream_word_idx_q <= '0;
                end else begin
                    stream_word_idx_q <= stream_word_idx_q + 8'd1;
                end
            end

            if (gemm_core_done) begin
                compute_inflight_q <= 1'b0;
                completed_tiles_q <= completed_tiles_q + 6'd1;
                drain_qb_q <= compute_qb_q;
                drain_kb_q <= compute_kb_q;
                stream_word_idx_q <= '0;
            end

            unique case ({gemm_core_done, co_last_fire})
                2'b10: shadow_busy_q <= 1'b1;
                2'b01: shadow_busy_q <= 1'b0;
                2'b11: shadow_busy_q <= 1'b1;
                default: begin
                end
            endcase

            if (start_fire) begin
                phase_q <= PH_VALIDATE;
                token_count_q <= token_count_i;
                num_q_blocks_q <= token_num_k_blocks;
                num_k_blocks_q <= token_num_k_blocks;
                gamma16_fix_q <= gamma16_fix_i;
                mask_en_q <= mask_en_i;
                q_block_start_q <= q_block_start_i;
                q_block_count_q <= requested_q_block_count;
                total_output_words_q <= {5'd0, token_num_k_blocks} << 5;
                chunk_total_output_words_q <=
                    {9'd0, requested_q_block_count} *
                    {4'd0, ({5'd0, token_num_k_blocks} << 5)};
            end else if (phase_q == PH_VALIDATE) begin
                if (!cfg_chunk_shape_legal) begin
                    phase_q <= PH_IDLE;
                    local_error_sticky_q <= 1'b1;
                    local_last_error_q <= NPU_ERR_ILLEGAL_SHAPE;
                end else if (gemm_cfg_ready) begin
                    phase_q <= PH_RUN;
                    gemm_cfg_start_q <= 1'b1;
                    gemm_a_base_word_addr_q <=
                        ADDR_W'({6'd0, q_block_start_q} * 11'(TILE_M * 2));
                    gemm_w_base_word_addr_q <= '0;
                    compute_inflight_q <= 1'b1;
                    active_qb_q <= q_block_start_q;
                    compute_qb_q <= q_block_start_q;
                    compute_kb_q <= '0;
                    next_launch_kb_q <= 6'd1;
                    completed_tiles_q <= '0;
                    shadow_busy_q <= 1'b0;
                    drain_qb_q <= '0;
                    drain_kb_q <= '0;
                    stream_word_idx_q <= '0;
                    score_s1_valid_q <= 1'b0;
                    score_round_valid_q <= 1'b0;
                    output_write_count_q <= '0;
                    qblock_output_write_count_q <= '0;
                    row_out_valid_q <= 1'b0;
                    m_global_out_row_q <= '0;
                    m_global_out_data_q <= '0;
                    score_s1_valid_q <= 1'b0;
                    score_round_valid_q <= 1'b0;
                    s0_valid_q <= 1'b0;
                    s1_valid_q <= 1'b0;
                    s2_valid_q <= 1'b0;
                    s3_valid_q <= 1'b0;
                    s4_valid_q <= 1'b0;
                    post_valid_q <= 1'b0;
                    s0_row_q <= '0;
                    s0_qb_q <= '0;
                    s0_kb_q <= '0;
                    s0_in_chunk_q <= '0;
                    s0_meta_byte_index_q <= '0;
                    s1_valid_count_q <= '0;
                    s1_row_q <= '0;
                    s1_kb_q <= '0;
                    s1_in_chunk_q <= '0;
                    s1_meta_byte_index_q <= '0;
                    s2_valid_count_q <= '0;
                    s2_row_q <= '0;
                    s2_kb_q <= '0;
                    s2_in_chunk_q <= '0;
                    s2_meta_byte_index_q <= '0;
                    s3_valid_count_q <= '0;
                    s3_row_q <= '0;
                    s3_kb_q <= '0;
                    s3_in_chunk_q <= '0;
                    s3_meta_byte_index_q <= '0;
                    s4_valid_count_q <= '0;
                    s4_m_block_q <= SCORE_MIN;
                    s4_has_valid_q <= 1'b0;
                    s4_row_q <= '0;
                    s4_kb_q <= '0;
                    s4_in_chunk_q <= '0;
                    s4_meta_byte_index_q <= '0;
                    post_m_global_row_q <= '0;
                    post_m_global_data_q <= SCORE_MIN;
                    for (int col = 0; col < TILE_N; col++) begin
                        row_score_q[col] <= '0;
                        s0_score_q[col] <= '0;
                        s1_score_q[col] <= '0;
                        s2_score_q[col] <= '0;
                        s3_score_q[col] <= '0;
                        s4_score_q[col] <= '0;
                    end
                    for (int group = 0; group < 8; group++) begin
                        s2_group_max_q[group] <= SCORE_MIN;
                        s2_group_valid_q[group] <= 1'b0;
                    end
                    for (int pair = 0; pair < 4; pair++) begin
                        s3_pair_max_q[pair] <= SCORE_MIN;
                        s3_pair_valid_q[pair] <= 1'b0;
                    end
                    for (int row = 0; row < TILE_M; row++) begin
                        m_global_q[row] <= SCORE_MIN;
                    end
                end
            end else if (launch_next_tile) begin
                gemm_cfg_start_q <= 1'b1;
                gemm_w_base_word_addr_q <=
                    ADDR_W'({6'd0, next_launch_kb_q} * 11'(TILE_K));
                compute_inflight_q <= 1'b1;
                compute_qb_q <= active_qb_q;
                compute_kb_q <= next_launch_kb_q[4:0];
                next_launch_kb_q <= next_launch_kb_q + 6'd1;
            end

`ifndef SYNTHESIS
            if (DEBUG_PRINT && gemm_cfg_start_q) begin
                $display("[%0t] ATTN launch Qblock=%0d Kblock=%0d A_base_word=%0d W_base_word=%0d shadow_busy=%0d",
                         $time, compute_qb_q, compute_kb_q,
                         gemm_a_base_word_addr_q, gemm_w_base_word_addr_q, shadow_busy_q);
            end

            if (DEBUG_PRINT && gemm_core_done) begin
                $display("[%0t] ATTN snapshot done Qblock=%0d Kblock=%0d, shadow drain can stream",
                         $time, compute_qb_q, compute_kb_q);
            end

            if (DEBUG_PRINT && co_fire) begin
                $write("[%0t] ATTN drain A qb=%0d kb=%0d row=%0d group=%0d:",
                       $time, drain_qb_q, drain_kb_q, stream_row_id, stream_group_id);
                for (int lane = 0; lane < LANES_I32; lane++) begin
                    $write(" %0d", lane_dot[lane]);
                end
                $write("\n");

                $write("[%0t] ATTN scale S_q qb=%0d kb=%0d row=%0d group=%0d:",
                       $time, drain_qb_q, drain_kb_q, stream_row_id, stream_group_id);
                for (int lane = 0; lane < LANES_I32; lane++) begin
                    $write(" %0d", lane_score[lane]);
                end
                $write("\n");

                if (stream_row_complete) begin
                    $write("[%0t] ATTN row complete qb=%0d kb=%0d row=%0d S_q:",
                           $time, drain_qb_q, drain_kb_q, stream_row_id);
                    for (int col = 0; col < TILE_N; col++) begin
                        $write(" %0d", row_score_work[col]);
                    end
                    $write("\n");

                    $display("[%0t] ATTN row postprocess enqueued qb=%0d kb=%0d row=%0d",
                             $time, drain_qb_q, drain_kb_q, stream_row_id);
                end
            end

            if (DEBUG_PRINT && row_out_fire) begin
                $write("[%0t] ATTN O0 write addr=%0d data:", $time, row_out_addr_q);
                for (int col = 0; col < TILE_N; col++) begin
                    $write(" %0d", row_out_data_q[col*8 +: 8]);
                end
                $write("\n");
            end

`endif
        end
    end

    wire unused_core_outputs = gemm_core_busy ^ ^gemm_co_lane_mask ^ gemm_core_done ^
                               ^token_count_q ^ ^num_q_blocks_q ^ ^completed_tiles_q;

endmodule

`default_nettype wire
