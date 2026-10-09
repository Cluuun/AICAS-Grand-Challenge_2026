`ifndef SOFTMAX_STREAM_UNIT_SV
`define SOFTMAX_STREAM_UNIT_SV

module softmax_stream_unit #(
    parameter int USER_WIDTH = 16,
    parameter int MAX_LEN    = 4096,
    // Power-of-two score scale applied before softmax. 0 keeps raw score
    // semantics; SmolVLM2 head_dim=64 uses 3 for kq_scale=1/8.
    parameter int SCORE_SCALE_POW2_SHIFT = 0,

    localparam int LEN_WIDTH = $clog2(MAX_LEN + 1),
    localparam int IDX_WIDTH = (MAX_LEN <= 1) ? 1 : $clog2(MAX_LEN)
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  start_i,
    input  logic [USER_WIDTH-1:0] elem_count_i,
    input  logic [1:0]            group_count_m1_i,
    output logic                  busy_o,
    output logic                  done_o,
    output logic                  error_o,

    input  logic                  score_valid_i,
    input  logic [15:0]           score_data_i,
    output logic                  score_pop_o,

    output logic                  valid_o,
    output logic [15:0]           data_o,
    output logic [USER_WIDTH-1:0] user_o
);

    localparam logic [16:0] Q_ONE    = 17'd65536;
    localparam int          PROB_FRAC_BITS = 24;
    localparam int          INV_FRAC_BITS = 40;
    localparam int          PROB_MUL_SHIFT = INV_FRAC_BITS - PROB_FRAC_BITS;
    localparam logic [PROB_FRAC_BITS:0] Q_ONE_PROB = 25'd16777216;
    localparam int          DIV_BITS = 49;

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_READ_MAX,
        ST_SUM_READ,
        ST_INV_DIV_START,
        ST_INV_DIV,
        ST_INV_ROUND,
        ST_EMIT_READ,
        ST_DONE
    } state_t;

    state_t state_q;

    (* ram_style = "block" *) logic signed [15:0] score_mem [0:MAX_LEN-1];

    logic [LEN_WIDTH-1:0] elem_count_q;
    logic [1:0]           group_count_m1_q;
    logic [1:0]           group_q;
    logic [LEN_WIDTH-1:0] read_count_q;
    logic [LEN_WIDTH-1:0] read_write_count_q;
    logic [LEN_WIDTH-1:0] sum_issue_count_q;
    logic [LEN_WIDTH-1:0] sum_done_count_q;
    logic [LEN_WIDTH-1:0] emit_count_q;
    logic signed [15:0]   max_q;
    logic [31:0]          sum_q16_q;
    logic                 error_q;

    logic [15:0] score_data_q;
    logic signed [15:0] score_q8_8_q;
    logic               read_capture_valid_q;
    logic               read_shift_valid_q;
    logic               read_saturate_valid_q;
    logic               read_convert_valid_q;
    logic               read_fp16_sign_q;
    logic [4:0]         read_fp16_exp_q;
    logic [10:0]        read_fp16_mant_q;
    logic signed [31:0] read_fp16_shift_q;
    logic signed [31:0] read_q8_8_value_q;

    logic signed [15:0] score_mem_dout_q;

    logic [48:0] div_dividend_q;
    logic [48:0] div_quotient_q;
    logic [49:0] div_remainder_q;
    logic [31:0] div_divisor_q;
    logic [5:0]  div_count_q;
    logic [PROB_FRAC_BITS:0] inv_sum_q;
    logic                    sum_issue_valid_q;
    logic                    sum_delta_valid_q;
    logic signed [15:0]      sum_delta_q;
    logic                    sum_log_valid_q;
    logic                    sum_log_one_q;
    logic [7:0]              sum_log_int_q;
    logic [7:0]              sum_log_frac_q;
    logic                    sum_lut_valid_q;
    logic                    sum_lut_one_q;
    logic [16:0]             sum_lut_base_q;
    logic [16:0]             sum_lut_frac_q;
    logic                    sum_exp_valid_q;
    logic [16:0]             sum_exp_value_q;
    logic                    emit_issue_valid_q;
    logic [LEN_WIDTH-1:0]    emit_issue_user_q;
    logic                    emit_delta_valid_q;
    logic [LEN_WIDTH-1:0]    emit_delta_user_q;
    logic signed [15:0]      emit_delta_q;
    logic                    emit_log_valid_q;
    logic [LEN_WIDTH-1:0]    emit_log_user_q;
    logic                    emit_log_one_q;
    logic [7:0]              emit_log_int_q;
    logic [7:0]              emit_log_frac_q;
    logic                    emit_lut_valid_q;
    logic [LEN_WIDTH-1:0]    emit_lut_user_q;
    logic                    emit_lut_one_q;
    logic [16:0]             emit_lut_base_q;
    logic [16:0]             emit_lut_frac_q;
    logic                    emit_exp_valid_q;
    logic [LEN_WIDTH-1:0]    emit_exp_user_q;
    logic [16:0]             emit_exp_value_q;
    logic                    emit_prob_valid_q;
    logic [LEN_WIDTH-1:0]    emit_prob_user_q;
    logic [PROB_FRAC_BITS:0] emit_prob_q;
    logic                    emit_fp_pos_valid_q;
    logic [LEN_WIDTH-1:0]    emit_fp_pos_user_q;
    logic [PROB_FRAC_BITS:0] emit_fp_pos_prob_q;
    logic [4:0]              emit_fp_pos_q;
    logic                    emit_fp16_valid_q;
    logic [LEN_WIDTH-1:0]    emit_fp16_user_q;
    logic [15:0]             emit_fp16_data_q;

    logic [48:0]        inv_dividend_w;
    logic [49:0]        div_remainder_shift_w;
    logic [49:0]        div_remainder_sub_w;
    logic [49:0]        div_remainder_next_w;
    logic [48:0]        div_quotient_next_w;
    logic               div_subtract_w;
    logic [PROB_FRAC_BITS:0] inv_sum_next_w;
    logic               div_round_up_w;
    logic [PROB_FRAC_BITS+1:0] inv_sum_rounded_w;
    logic               score_mem_we_w;
    logic               read_fire_w;
    logic               sum_issue_w;
    logic               emit_issue_w;
    logic [IDX_WIDTH-1:0] score_mem_wr_addr_w;
    logic [IDX_WIDTH-1:0] score_mem_rd_addr_w;
    logic signed [15:0]   score_mem_din_w;
    logic [15:0]           emit_delta_abs_w;
    logic [31:0]           emit_log2_w;
    logic [33:0]           emit_exp_product_w;
    logic [15:0]           sum_delta_abs_w;
    logic [31:0]           sum_log2_w;
    logic [33:0]           sum_exp_product_w;

    assign busy_o      = (state_q != ST_IDLE);
    assign error_o     = error_q;
    assign score_pop_o = read_fire_w;

    assign inv_dividend_w = 49'd1 << INV_FRAC_BITS;

    assign div_remainder_shift_w = {div_remainder_q[48:0], div_dividend_q[48]};
    assign div_subtract_w        = div_remainder_shift_w >= {18'd0, div_divisor_q};
    assign div_remainder_sub_w   = div_remainder_shift_w - {18'd0, div_divisor_q};
    assign div_remainder_next_w  = div_subtract_w ? div_remainder_sub_w : div_remainder_shift_w;
    assign div_quotient_next_w   = {div_quotient_q[47:0], div_subtract_w};
    assign div_round_up_w = ({div_remainder_q[48:0], 1'b0} >= {18'd0, div_divisor_q});
    assign inv_sum_rounded_w = {1'b0, div_quotient_q[PROB_FRAC_BITS:0]} +
                               {{PROB_FRAC_BITS+1{1'b0}}, div_round_up_w};
    assign inv_sum_next_w = inv_sum_rounded_w[PROB_FRAC_BITS+1] ?
                            Q_ONE_PROB : inv_sum_rounded_w[PROB_FRAC_BITS:0];
    assign read_fire_w           = (state_q == ST_READ_MAX) &&
                                   score_valid_i &&
                                   (read_count_q < elem_count_q);
    assign sum_issue_w           = (state_q == ST_SUM_READ) &&
                                   (sum_issue_count_q < elem_count_q);
    assign emit_issue_w          = (state_q == ST_EMIT_READ) &&
                                   (emit_count_q < elem_count_q);
    assign score_mem_we_w        = (state_q == ST_READ_MAX) && read_convert_valid_q;
    assign score_mem_din_w       = score_q8_8_q;
    assign score_mem_wr_addr_w   = read_write_count_q[IDX_WIDTH-1:0];
    assign score_mem_rd_addr_w   = sum_issue_w ? sum_issue_count_q[IDX_WIDTH-1:0] :
                                   (emit_issue_w ? emit_count_q[IDX_WIDTH-1:0] : '0);
    assign emit_delta_abs_w      = 16'(-emit_delta_q);
    assign emit_log2_w           = (32'(emit_delta_abs_w) * 32'd369) >> 8;
    assign emit_exp_product_w    = emit_lut_base_q * emit_lut_frac_q;
    assign sum_delta_abs_w       = 16'(-sum_delta_q);
    assign sum_log2_w            = (32'(sum_delta_abs_w) * 32'd369) >> 8;
    assign sum_exp_product_w     = sum_lut_base_q * sum_lut_frac_q;

    function automatic int signed fp16_to_q8_8_shifted_value(
        input logic       sign,
        input logic [4:0] exp,
        input logic [10:0] mant,
        input logic signed [31:0] shift
    );
        int signed value;
        begin
            value = 0;

            if (exp == 5'h1f) begin
                value = sign ? -32768 : 32767;
            end else if (exp != 5'd0) begin
                if (shift >= 0)
                    value = int'(mant) <<< shift;
                else
                    value = int'(mant) >>> (-shift);

                if (sign)
                    value = -value;
            end

            fp16_to_q8_8_shifted_value = value;
        end
    endfunction

    function automatic logic signed [15:0] saturate_q8_8(input logic signed [31:0] value);
        begin
            if (value > 32767)
                saturate_q8_8 = 16'sh7fff;
            else if (value < -32768)
                saturate_q8_8 = 16'sh8000;
            else
                saturate_q8_8 = 16'(value);
        end
    endfunction

`ifndef SYNTHESIS
    initial begin
        if (SCORE_SCALE_POW2_SHIFT < 0)
            $fatal(1, "SCORE_SCALE_POW2_SHIFT must be non-negative");
    end
`endif

    function automatic logic [16:0] exp2_frac_lut_q16(input logic [7:0] frac_idx);
        logic [16:0] coarse_q16;
        logic [16:0] fine_q16;
        logic [33:0] product;
        begin
            unique case (frac_idx[7:4])
                4'd0:  coarse_q16 = 17'd65536;
                4'd1:  coarse_q16 = 17'd62757;
                4'd2:  coarse_q16 = 17'd60097;
                4'd3:  coarse_q16 = 17'd57549;
                4'd4:  coarse_q16 = 17'd55109;
                4'd5:  coarse_q16 = 17'd52773;
                4'd6:  coarse_q16 = 17'd50535;
                4'd7:  coarse_q16 = 17'd48393;
                4'd8:  coarse_q16 = 17'd46341;
                4'd9:  coarse_q16 = 17'd44376;
                4'd10: coarse_q16 = 17'd42495;
                4'd11: coarse_q16 = 17'd40693;
                4'd12: coarse_q16 = 17'd38968;
                4'd13: coarse_q16 = 17'd37316;
                4'd14: coarse_q16 = 17'd35734;
                default: coarse_q16 = 17'd34219;
            endcase
            unique case (frac_idx[3:0])
                4'd0:  fine_q16 = 17'd65536;
                4'd1:  fine_q16 = 17'd65359;
                4'd2:  fine_q16 = 17'd65182;
                4'd3:  fine_q16 = 17'd65006;
                4'd4:  fine_q16 = 17'd64830;
                4'd5:  fine_q16 = 17'd64655;
                4'd6:  fine_q16 = 17'd64480;
                4'd7:  fine_q16 = 17'd64306;
                4'd8:  fine_q16 = 17'd64132;
                4'd9:  fine_q16 = 17'd63958;
                4'd10: fine_q16 = 17'd63785;
                4'd11: fine_q16 = 17'd63613;
                4'd12: fine_q16 = 17'd63441;
                4'd13: fine_q16 = 17'd63269;
                4'd14: fine_q16 = 17'd63098;
                default: fine_q16 = 17'd62928;
            endcase
            product = coarse_q16 * fine_q16;
            exp2_frac_lut_q16 = (product + 34'd32768) >> 16;
        end
    endfunction

    function automatic logic [16:0] round_product_q16(input logic [33:0] product);
        logic [33:0] rounded;
        begin
            rounded = product + 34'd32768;
            round_product_q16 = (rounded[33:16] > 18'(Q_ONE)) ?
                                Q_ONE : rounded[32:16];
        end
    endfunction

    function automatic logic [16:0] exp_neg_q8_8_to_q16(input logic signed [15:0] value);
        logic [15:0] local_abs;
        logic [31:0] local_log2;
        logic [7:0]  local_int;
        logic [7:0]  local_frac;
        logic [16:0] local_base;
        logic [16:0] local_frac_lut;
        logic [33:0] local_product;
        begin
            if (value >= 0) begin
                exp_neg_q8_8_to_q16 = Q_ONE;
            end else begin
                local_abs = 16'(-value);
                local_log2 = (32'(local_abs) * 32'd369) >> 8;
                local_int = local_log2[15:8];
                local_frac = local_log2[7:0];
                local_base = (local_int >= 8'd16) ? 17'd0 : (Q_ONE >> local_int);
                local_frac_lut = exp2_frac_lut_q16(local_frac);
                local_product = local_base * local_frac_lut;
                exp_neg_q8_8_to_q16 = round_product_q16(local_product);
            end
        end
    endfunction

    function automatic logic [PROB_FRAC_BITS:0] prob_from_exp_inv(
        input logic [16:0] exp_value,
        input logic [PROB_FRAC_BITS:0] inv_sum
    );
        logic [41:0] product;
        logic [41:0] rounded;
        begin
            product = exp_value * inv_sum;
            rounded = product + (42'd1 << (PROB_MUL_SHIFT - 1));
            prob_from_exp_inv = (rounded[41:PROB_MUL_SHIFT] > 26'(Q_ONE_PROB)) ?
                                Q_ONE_PROB :
                                rounded[PROB_FRAC_BITS+PROB_MUL_SHIFT:PROB_MUL_SHIFT];
        end
    endfunction

    function automatic logic [15:0] q0_24_to_fp16(input logic [PROB_FRAC_BITS:0] q);
        int p;
        int i;
        logic found;
        logic [4:0] exp16;
        logic [9:0] frac16;
        logic [PROB_FRAC_BITS:0] leading;
        logic [34:0] frac_calc_w;
        logic [10:0] mant_round;
        begin
            q0_24_to_fp16 = 16'h0000;
            p = 0;
            found = 1'b0;
            exp16 = '0;
            frac16 = '0;
            leading = '0;
            frac_calc_w = '0;
            mant_round = '0;

            for (i = PROB_FRAC_BITS; i >= 0; i = i - 1) begin
                if (q[i] && !found) begin
                    p = i;
                    found = 1'b1;
                end
            end

            if (q != '0) begin
                if (p <= (PROB_FRAC_BITS - 15)) begin
                    q0_24_to_fp16 = {1'b0, 5'd0, q[9:0]};
                end else begin
                    exp16 = 5'(p + 15 - PROB_FRAC_BITS);
                    leading = (25'd1) << p;
                    frac_calc_w = 35'(q - leading) << 10;
                    mant_round = (frac_calc_w + (35'd1 << (p - 1))) >> p;
                    if (mant_round[10]) begin
                        exp16 = exp16 + 5'd1;
                        frac16 = 10'd0;
                    end else begin
                        frac16 = mant_round[9:0];
                    end
                    q0_24_to_fp16 = {1'b0, exp16, frac16};
                end
            end
        end
    endfunction

    function automatic logic [4:0] q0_24_leading_pos(input logic [PROB_FRAC_BITS:0] q);
        int i;
        begin
            q0_24_leading_pos = '0;
            for (i = 0; i <= PROB_FRAC_BITS; i = i + 1) begin
                if (q[i])
                    q0_24_leading_pos = 5'(i);
            end
        end
    endfunction

    function automatic logic [15:0] q0_24_to_fp16_pos(
        input logic [PROB_FRAC_BITS:0] q,
        input logic [4:0]              p
    );
        logic [4:0] exp16;
        logic [9:0] frac16;
        logic [PROB_FRAC_BITS:0] leading;
        logic [34:0] frac_calc_w;
        logic [10:0] mant_round;
        begin
            q0_24_to_fp16_pos = 16'h0000;
            exp16 = '0;
            frac16 = '0;
            leading = '0;
            frac_calc_w = '0;
            mant_round = '0;

            if (q != '0) begin
                if (p <= 5'(PROB_FRAC_BITS - 15)) begin
                    q0_24_to_fp16_pos = {1'b0, 5'd0, q[9:0]};
                end else begin
                    exp16 = 5'(p + 15 - PROB_FRAC_BITS);
                    leading = (25'd1) << p;
                    frac_calc_w = 35'(q - leading) << 10;
                    mant_round = (frac_calc_w + (35'd1 << (p - 1))) >> p;
                    if (mant_round[10]) begin
                        exp16 = exp16 + 5'd1;
                        frac16 = 10'd0;
                    end else begin
                        frac16 = mant_round[9:0];
                    end
                    q0_24_to_fp16_pos = {1'b0, exp16, frac16};
                end
            end
        end
    endfunction

    always_ff @(posedge clk) begin
        if (score_mem_we_w)
            score_mem[score_mem_wr_addr_w] <= score_mem_din_w;
        score_mem_dout_q <= score_mem[score_mem_rd_addr_w];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q           <= ST_IDLE;
            elem_count_q      <= '0;
            group_count_m1_q  <= '0;
            group_q           <= '0;
            read_count_q      <= '0;
            read_write_count_q <= '0;
            sum_issue_count_q <= '0;
            sum_done_count_q  <= '0;
            emit_count_q      <= '0;
            max_q             <= '0;
            sum_q16_q         <= '0;
            error_q           <= 1'b0;
            score_data_q      <= '0;
            score_q8_8_q      <= '0;
            read_capture_valid_q <= 1'b0;
            read_shift_valid_q <= 1'b0;
            read_saturate_valid_q <= 1'b0;
            read_convert_valid_q <= 1'b0;
            read_fp16_sign_q <= 1'b0;
            read_fp16_exp_q <= '0;
            read_fp16_mant_q <= '0;
            read_fp16_shift_q <= '0;
            read_q8_8_value_q <= '0;
            div_dividend_q    <= '0;
            div_quotient_q    <= '0;
            div_remainder_q   <= '0;
            div_divisor_q     <= '0;
            div_count_q       <= '0;
            inv_sum_q         <= '0;
            sum_issue_valid_q <= 1'b0;
            sum_delta_valid_q <= 1'b0;
            sum_delta_q <= '0;
            sum_log_valid_q <= 1'b0;
            sum_log_one_q <= 1'b0;
            sum_log_int_q <= '0;
            sum_log_frac_q <= '0;
            sum_lut_valid_q <= 1'b0;
            sum_lut_one_q <= 1'b0;
            sum_lut_base_q <= '0;
            sum_lut_frac_q <= '0;
            sum_exp_valid_q <= 1'b0;
            sum_exp_value_q <= '0;
            emit_issue_valid_q <= 1'b0;
            emit_issue_user_q <= '0;
            emit_delta_valid_q <= 1'b0;
            emit_delta_user_q <= '0;
            emit_delta_q <= '0;
            emit_log_valid_q <= 1'b0;
            emit_log_user_q <= '0;
            emit_log_one_q <= 1'b0;
            emit_log_int_q <= '0;
            emit_log_frac_q <= '0;
            emit_lut_valid_q <= 1'b0;
            emit_lut_user_q <= '0;
            emit_lut_one_q <= 1'b0;
            emit_lut_base_q <= '0;
            emit_lut_frac_q <= '0;
            emit_exp_valid_q <= 1'b0;
            emit_exp_user_q <= '0;
            emit_exp_value_q <= '0;
            emit_prob_valid_q <= 1'b0;
            emit_prob_user_q <= '0;
            emit_prob_q <= '0;
            emit_fp_pos_valid_q <= 1'b0;
            emit_fp_pos_user_q <= '0;
            emit_fp_pos_prob_q <= '0;
            emit_fp_pos_q <= '0;
            emit_fp16_valid_q <= 1'b0;
            emit_fp16_user_q <= '0;
            emit_fp16_data_q <= '0;
            done_o            <= 1'b0;
            valid_o           <= 1'b0;
            data_o            <= '0;
            user_o            <= '0;
        end else begin
            done_o  <= 1'b0;
            valid_o <= 1'b0;
            data_o  <= '0;
            user_o  <= '0;
            sum_issue_valid_q <= 1'b0;
            sum_delta_valid_q <= 1'b0;
            sum_log_valid_q <= 1'b0;
            sum_lut_valid_q <= 1'b0;
            sum_exp_valid_q <= 1'b0;
            emit_issue_valid_q <= 1'b0;
            emit_delta_valid_q <= 1'b0;
            emit_log_valid_q <= 1'b0;
            emit_lut_valid_q <= 1'b0;
            emit_exp_valid_q <= 1'b0;
            emit_prob_valid_q <= 1'b0;
            emit_fp_pos_valid_q <= 1'b0;
            emit_fp16_valid_q <= 1'b0;

            unique case (state_q)
                ST_IDLE: begin
                    if (start_i) begin
                        error_q      <= (elem_count_i == '0) || (elem_count_i > USER_WIDTH'(MAX_LEN));
                        elem_count_q <= LEN_WIDTH'(elem_count_i);
                        group_count_m1_q <= group_count_m1_i;
                        group_q      <= '0;
                        read_count_q <= '0;
                        read_write_count_q <= '0;
                        sum_issue_count_q <= '0;
                        sum_done_count_q <= '0;
                        emit_count_q <= '0;
                        max_q        <= '0;
                        sum_q16_q    <= '0;
                        inv_sum_q    <= '0;
                        read_capture_valid_q <= 1'b0;
                        read_shift_valid_q <= 1'b0;
                        read_saturate_valid_q <= 1'b0;
                        read_convert_valid_q <= 1'b0;
                        sum_issue_valid_q <= 1'b0;
                        sum_delta_valid_q <= 1'b0;
                        sum_log_valid_q <= 1'b0;
                        sum_lut_valid_q <= 1'b0;
                        sum_exp_valid_q <= 1'b0;
                        emit_issue_valid_q <= 1'b0;
                        emit_delta_valid_q <= 1'b0;
                        emit_log_valid_q <= 1'b0;
                        emit_lut_valid_q <= 1'b0;
                        emit_exp_valid_q <= 1'b0;
                        emit_prob_valid_q <= 1'b0;
                        emit_fp_pos_valid_q <= 1'b0;
                        emit_fp16_valid_q <= 1'b0;
                        if ((elem_count_i == '0) || (elem_count_i > USER_WIDTH'(MAX_LEN)))
                            state_q <= ST_DONE;
                        else
                            state_q <= ST_READ_MAX;
                    end
                end

                ST_READ_MAX: begin
                    read_capture_valid_q <= read_fire_w;
                    read_shift_valid_q <= read_capture_valid_q;
                    read_saturate_valid_q <= read_shift_valid_q;
                    read_convert_valid_q <= read_saturate_valid_q;

                    if (read_fire_w) begin
                        score_data_q <= score_data_i;
                        read_count_q <= read_count_q + LEN_WIDTH'(1);
                    end

                    if (read_capture_valid_q) begin
                        read_fp16_sign_q  <= score_data_q[15];
                        read_fp16_exp_q   <= score_data_q[14:10];
                        read_fp16_mant_q  <= {1'b1, score_data_q[9:0]};
                        read_fp16_shift_q <= int'(score_data_q[14:10]) - 17 - SCORE_SCALE_POW2_SHIFT;
                    end

                    if (read_shift_valid_q)
                        read_q8_8_value_q <= fp16_to_q8_8_shifted_value(
                            read_fp16_sign_q,
                            read_fp16_exp_q,
                            read_fp16_mant_q,
                            read_fp16_shift_q
                        );

                    if (read_saturate_valid_q)
                        score_q8_8_q <= saturate_q8_8(read_q8_8_value_q);

                    if (read_convert_valid_q) begin
                        if ((read_write_count_q == '0) || (score_q8_8_q > max_q))
                            max_q <= score_q8_8_q;

                        read_write_count_q <= read_write_count_q + LEN_WIDTH'(1);
                        if (read_write_count_q == (elem_count_q - LEN_WIDTH'(1))) begin
                            read_capture_valid_q <= 1'b0;
                            read_shift_valid_q <= 1'b0;
                            read_saturate_valid_q <= 1'b0;
                            read_convert_valid_q <= 1'b0;
                            sum_issue_count_q <= '0;
                            sum_done_count_q <= '0;
                            sum_q16_q <= '0;
                            sum_issue_valid_q <= 1'b0;
                            sum_delta_valid_q <= 1'b0;
                            sum_log_valid_q <= 1'b0;
                            sum_lut_valid_q <= 1'b0;
                            sum_exp_valid_q <= 1'b0;
                            state_q <= ST_SUM_READ;
                        end
                    end
                end

                ST_SUM_READ: begin
                    if (sum_issue_w) begin
                        sum_issue_valid_q <= 1'b1;
                        sum_issue_count_q <= sum_issue_count_q + LEN_WIDTH'(1);
                    end

                    if (sum_issue_valid_q) begin
                        sum_delta_valid_q <= 1'b1;
                        sum_delta_q       <= score_mem_dout_q - max_q;
                    end

                    if (sum_delta_valid_q) begin
                        sum_log_valid_q <= 1'b1;
                        sum_log_one_q   <= (sum_delta_q >= 0);
                        sum_log_int_q   <= sum_log2_w[15:8];
                        sum_log_frac_q  <= sum_log2_w[7:0];
                    end

                    if (sum_log_valid_q) begin
                        sum_lut_valid_q <= 1'b1;
                        sum_lut_one_q   <= sum_log_one_q;
                        sum_lut_base_q  <= (sum_log_int_q >= 8'd16) ? 17'd0 : (Q_ONE >> sum_log_int_q);
                        sum_lut_frac_q  <= exp2_frac_lut_q16(sum_log_frac_q);
                    end

                    if (sum_lut_valid_q) begin
                        sum_exp_valid_q <= 1'b1;
                        sum_exp_value_q <= sum_lut_one_q ? Q_ONE : round_product_q16(sum_exp_product_w);
                    end

                    if (sum_exp_valid_q) begin
                        sum_q16_q <= sum_q16_q + 32'(sum_exp_value_q);
                        sum_done_count_q <= sum_done_count_q + LEN_WIDTH'(1);
                        if (sum_done_count_q == (elem_count_q - LEN_WIDTH'(1))) begin
                            sum_issue_valid_q <= 1'b0;
                            sum_delta_valid_q <= 1'b0;
                            sum_log_valid_q <= 1'b0;
                            sum_lut_valid_q <= 1'b0;
                            sum_exp_valid_q <= 1'b0;
                            state_q <= ST_INV_DIV_START;
                        end
                    end
                end

                ST_INV_DIV_START: begin
                    if (sum_q16_q == '0) begin
                        error_q <= 1'b1;
                        state_q <= ST_DONE;
                    end else begin
                        div_dividend_q  <= inv_dividend_w;
                        div_quotient_q  <= '0;
                        div_remainder_q <= '0;
                        div_divisor_q   <= sum_q16_q;
                        div_count_q     <= 6'(DIV_BITS);
                        state_q         <= ST_INV_DIV;
                    end
                end

                ST_INV_DIV: begin
                    div_remainder_q <= div_remainder_next_w;
                    div_dividend_q  <= {div_dividend_q[47:0], 1'b0};
                    div_quotient_q  <= div_quotient_next_w;
                    div_count_q     <= div_count_q - 6'd1;

                    if (div_count_q == 6'd1) begin
                        state_q   <= ST_INV_ROUND;
                    end
                end

                ST_INV_ROUND: begin
                    inv_sum_q <= inv_sum_next_w;
                    state_q   <= ST_EMIT_READ;
                end

                ST_EMIT_READ: begin
                    if (emit_issue_w) begin
                        emit_issue_valid_q <= 1'b1;
                        emit_issue_user_q  <= emit_count_q;
                        emit_count_q       <= emit_count_q + LEN_WIDTH'(1);
                    end

                    if (emit_issue_valid_q) begin
                        emit_delta_valid_q <= 1'b1;
                        emit_delta_user_q  <= emit_issue_user_q;
                        emit_delta_q       <= score_mem_dout_q - max_q;
                    end

                    if (emit_delta_valid_q) begin
                        emit_log_valid_q <= 1'b1;
                        emit_log_user_q  <= emit_delta_user_q;
                        emit_log_one_q   <= (emit_delta_q >= 0);
                        emit_log_int_q   <= emit_log2_w[15:8];
                        emit_log_frac_q  <= emit_log2_w[7:0];
                    end

                    if (emit_log_valid_q) begin
                        emit_lut_valid_q <= 1'b1;
                        emit_lut_user_q  <= emit_log_user_q;
                        emit_lut_one_q   <= emit_log_one_q;
                        emit_lut_base_q  <= (emit_log_int_q >= 8'd16) ? 17'd0 : (Q_ONE >> emit_log_int_q);
                        emit_lut_frac_q  <= exp2_frac_lut_q16(emit_log_frac_q);
                    end

                    if (emit_lut_valid_q) begin
                        emit_exp_valid_q <= 1'b1;
                        emit_exp_user_q  <= emit_lut_user_q;
                        emit_exp_value_q <= emit_lut_one_q ? Q_ONE : round_product_q16(emit_exp_product_w);
                    end

                    if (emit_exp_valid_q) begin
                        emit_prob_valid_q <= 1'b1;
                        emit_prob_user_q  <= emit_exp_user_q;
                        emit_prob_q       <= prob_from_exp_inv(emit_exp_value_q, inv_sum_q);
                    end

                    if (emit_prob_valid_q) begin
                        emit_fp_pos_valid_q <= 1'b1;
                        emit_fp_pos_user_q  <= emit_prob_user_q;
                        emit_fp_pos_prob_q  <= emit_prob_q;
                        emit_fp_pos_q       <= q0_24_leading_pos(emit_prob_q);
                    end

                    if (emit_fp_pos_valid_q) begin
                        emit_fp16_valid_q <= 1'b1;
                        emit_fp16_user_q  <= emit_fp_pos_user_q;
                        emit_fp16_data_q  <= q0_24_to_fp16_pos(emit_fp_pos_prob_q, emit_fp_pos_q);
                    end

                    if (emit_fp16_valid_q) begin
                        valid_o <= 1'b1;
                        data_o  <= emit_fp16_data_q;
                        user_o  <= USER_WIDTH'((LEN_WIDTH'(group_q) * elem_count_q) +
                                                emit_fp16_user_q);

                        if (emit_fp16_user_q == (elem_count_q - LEN_WIDTH'(1))) begin
                            if (group_q == group_count_m1_q) begin
                                state_q <= ST_DONE;
                            end else begin
                                group_q      <= group_q + 1'b1;
                                read_count_q <= '0;
                                read_write_count_q <= '0;
                                read_capture_valid_q <= 1'b0;
                                read_shift_valid_q <= 1'b0;
                                read_saturate_valid_q <= 1'b0;
                                read_convert_valid_q <= 1'b0;
                                sum_issue_count_q <= '0;
                                sum_done_count_q <= '0;
                                emit_count_q <= '0;
                                max_q        <= '0;
                                sum_q16_q    <= '0;
                                inv_sum_q    <= '0;
                                sum_issue_valid_q <= 1'b0;
                                sum_delta_valid_q <= 1'b0;
                                sum_log_valid_q <= 1'b0;
                                sum_lut_valid_q <= 1'b0;
                                sum_exp_valid_q <= 1'b0;
                                emit_issue_valid_q <= 1'b0;
                                emit_delta_valid_q <= 1'b0;
                                emit_log_valid_q <= 1'b0;
                                emit_lut_valid_q <= 1'b0;
                                emit_exp_valid_q <= 1'b0;
                                emit_prob_valid_q <= 1'b0;
                                emit_fp_pos_valid_q <= 1'b0;
                                emit_fp16_valid_q <= 1'b0;
                                state_q      <= ST_READ_MAX;
                            end
                        end
                    end
                end

                ST_DONE: begin
                    done_o  <= 1'b1;
                    state_q <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    property p_no_start_while_busy;
        @(posedge clk) disable iff (!rst_n) busy_o |-> !start_i;
    endproperty
    assert property (p_no_start_while_busy);
`endif

endmodule

`endif
