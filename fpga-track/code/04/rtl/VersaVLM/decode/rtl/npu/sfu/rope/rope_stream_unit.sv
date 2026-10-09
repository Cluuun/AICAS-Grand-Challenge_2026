`ifndef ROPE_STREAM_UNIT_SV
`define ROPE_STREAM_UNIT_SV

module rope_stream_unit #(
    parameter int USER_WIDTH       = 16,
    parameter int DATA_WIDTH       = 20,
    parameter int COEF_WIDTH       = 16,
    parameter int DATA_FRAC        = 14,
    parameter int HEAD_DIM         = 64,
    parameter int MAX_SEQ_LEN      = 4096,
    parameter int PAIRS_PER_CYCLE  = 4,
    parameter int POS_WIDTH        = 16,
    parameter int SPM_DATA_WIDTH   = 512,
    parameter int SPM_ADDR_WIDTH   = 19,
    parameter int ROPE_LUT_BASE_ADDR = 32'h0000_3f00,

    localparam int LANES           = PAIRS_PER_CYCLE * 2,
    localparam int BEAT_WIDTH      = LANES * DATA_WIDTH,
    localparam int PACK_CNT_WIDTH  = $clog2(LANES + 1),
    localparam int OUT_LANE_WIDTH  = (LANES <= 1) ? 1 : $clog2(LANES),
    localparam int DATA_ABS_WIDTH  = DATA_WIDTH + 1
) (
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    start_i,
    input  logic [USER_WIDTH-1:0]   elem_count_i,
    input  logic [POS_WIDTH-1:0]    position_i,
    output logic                    busy_o,
    output logic                    done_o,
    output logic                    error_o,

    output logic                    lut_rd_en_o,
    output logic [SPM_ADDR_WIDTH-1:0] lut_rd_addr_o,
    input  logic [SPM_DATA_WIDTH-1:0] lut_rd_data_i,

    input  logic                    in_valid_i,
    input  logic [15:0]             in_data_i,
    output logic                    in_pop_o,

    output logic                    valid_o,
    output logic [15:0]             data_o,
    output logic [USER_WIDTH-1:0]   user_o,
    input  logic                    ready_i
);

    localparam logic [USER_WIDTH-1:0] HEAD_DIM_COUNT = USER_WIDTH'(HEAD_DIM);
    localparam int N_PAIRS = HEAD_DIM / 2;
    localparam int LUT_LINE_BYTES = SPM_DATA_WIDTH / 8;
    localparam int LUT_SLOT_BYTES = 2 * LUT_LINE_BYTES;

    typedef enum logic [1:0] {
        LUT_IDLE,
        LUT_LOAD_COS,
        LUT_LOAD_SIN,
        LUT_WAIT_LAST
    } lut_state_t;

    logic active_q;
    lut_state_t lut_state_q;
    logic error_q;
    logic [USER_WIDTH-1:0] elem_count_q;
    logic [POS_WIDTH-1:0] position_q;
    logic lut_capture_valid_q;
    logic lut_capture_sin_q;
    logic [N_PAIRS*COEF_WIDTH-1:0] cos_cache_q;
    logic [N_PAIRS*COEF_WIDTH-1:0] sin_cache_q;

    logic [BEAT_WIDTH-1:0] pack_data_q;
    logic [PACK_CNT_WIDTH-1:0] pack_count_q;
    logic pack_valid_q;
    logic pack_last_q;
    logic [USER_WIDTH-1:0] input_count_q;

    logic [BEAT_WIDTH-1:0] unpack_data_q;
    logic unpack_valid_q;
    logic [OUT_LANE_WIDTH-1:0] unpack_lane_q;
    logic [USER_WIDTH-1:0] output_count_q;
    logic scalar_valid_q;
    logic scalar_sign_q;
    logic [DATA_ABS_WIDTH-1:0] scalar_abs_q;
    logic [5:0] scalar_lead_q;
    logic [USER_WIDTH-1:0] scalar_user_q;
    logic emit_valid_q;
    logic [15:0] emit_data_q;
    logic [USER_WIDTH-1:0] emit_user_q;

    logic engine_s_valid_w;
    logic engine_s_ready_w;
    logic [BEAT_WIDTH-1:0] engine_s_data_w;
    logic engine_s_last_w;
    logic engine_m_valid_w;
    logic engine_m_ready_w;
    logic [BEAT_WIDTH-1:0] engine_m_data_w;
    logic engine_m_last_w;

    logic accept_scalar_w;
    logic engine_in_fire_w;
    logic engine_out_fire_w;
    logic output_fire_w;
    logic emit_slot_ready_w;
    logic scalar_slot_ready_w;
    logic scalar_load_w;
    logic emit_load_w;
    logic emit_last_fire_w;
    logic last_scalar_in_w;
    logic [USER_WIDTH-1:0] input_elem_in_head_w;

    assign busy_o = active_q || (lut_state_q != LUT_IDLE);
    assign error_o = error_q;

    assign lut_rd_en_o = (lut_state_q == LUT_LOAD_COS) ||
                         (lut_state_q == LUT_LOAD_SIN);
    assign lut_rd_addr_o =
        SPM_ADDR_WIDTH'(ROPE_LUT_BASE_ADDR) +
        (position_q[0] ? SPM_ADDR_WIDTH'(LUT_SLOT_BYTES) : '0) +
        ((lut_state_q == LUT_LOAD_SIN) ? SPM_ADDR_WIDTH'(LUT_LINE_BYTES) : '0);

    assign input_elem_in_head_w = input_count_q % HEAD_DIM_COUNT;
    assign in_pop_o = active_q &&
                      !pack_valid_q &&
                      (input_count_q < elem_count_q) &&
                      in_valid_i;
    assign accept_scalar_w = in_pop_o;
    assign last_scalar_in_w = (input_elem_in_head_w == (HEAD_DIM_COUNT - USER_WIDTH'(1)));
    assign scalar_slot_ready_w = !scalar_valid_q || (emit_slot_ready_w && scalar_valid_q);
    assign scalar_load_w = active_q && scalar_slot_ready_w && unpack_valid_q;

    assign engine_s_valid_w = pack_valid_q;
    assign engine_s_data_w = pack_data_q;
    assign engine_s_last_w = pack_last_q;
    assign engine_in_fire_w = engine_s_valid_w && engine_s_ready_w;

    assign engine_m_ready_w = active_q && !unpack_valid_q;
    assign engine_out_fire_w = engine_m_valid_w && engine_m_ready_w;

    assign valid_o = emit_valid_q;
    assign data_o = emit_data_q;
    assign user_o = emit_user_q;
    assign output_fire_w = valid_o && ready_i;
    assign emit_slot_ready_w = !emit_valid_q || output_fire_w;
    assign emit_load_w = active_q && emit_slot_ready_w && scalar_valid_q;
    assign emit_last_fire_w = output_fire_w && (emit_user_q == (elem_count_q - USER_WIDTH'(1)));

    rope_engine #(
        .DATA_WIDTH      (DATA_WIDTH),
        .COEF_WIDTH      (COEF_WIDTH),
        .HEAD_DIM        (HEAD_DIM),
        .MAX_SEQ_LEN     (MAX_SEQ_LEN),
        .PAIRS_PER_CYCLE (PAIRS_PER_CYCLE),
        .POS_WIDTH       (POS_WIDTH)
    ) u_rope_engine (
        .clk        (clk),
        .rst_n      (rst_n),
        .s_valid    (engine_s_valid_w),
        .s_ready    (engine_s_ready_w),
        .s_data     (engine_s_data_w),
        .s_position (position_q),
        .s_last     (engine_s_last_w),
        .cos_lut_i  (cos_cache_q),
        .sin_lut_i  (sin_cache_q),
        .m_valid    (engine_m_valid_w),
        .m_ready    (engine_m_ready_w),
        .m_data     (engine_m_data_w),
        .m_last     (engine_m_last_w)
    );

    function automatic logic signed [DATA_WIDTH-1:0] fp16_to_q6_14(input logic [15:0] fp16);
        logic sign;
        logic [4:0] exp;
        logic [9:0] frac;
        logic [10:0] mant;
        int shift;
        int unsigned mag;
        int unsigned round_bit;
        int signed signed_value;
        begin
            sign = fp16[15];
            exp = fp16[14:10];
            frac = fp16[9:0];
            mant = (exp == 5'd0) ? {1'b0, frac} : {1'b1, frac};
            mag = 0;

            if (exp == 5'h1f) begin
                mag = 1 << (DATA_WIDTH - 1);
            end else if (exp == 5'd0) begin
                if (frac != 10'd0) begin
                    // subnormal: frac / 2^10 * 2^-14, then scaled by 2^14
                    mag = (int'(frac) + 512) >> 10;
                end
            end else begin
                // (1.frac * 2^(exp-15)) * 2^DATA_FRAC.
                shift = int'(exp) - 15 + DATA_FRAC - 10;
                if (shift >= 0) begin
                    mag = int'(mant) << shift;
                end else begin
                    round_bit = 1 << ((-shift) - 1);
                    mag = (int'(mant) + round_bit) >> (-shift);
                end
            end

            if (sign) begin
                if (mag >= (1 << (DATA_WIDTH - 1)))
                    fp16_to_q6_14 = {1'b1, {(DATA_WIDTH-1){1'b0}}};
                else begin
                    signed_value = -int'(mag);
                    fp16_to_q6_14 = DATA_WIDTH'(signed_value);
                end
            end else begin
                if (mag >= ((1 << (DATA_WIDTH - 1)) - 1))
                    fp16_to_q6_14 = {1'b0, {(DATA_WIDTH-1){1'b1}}};
                else
                    fp16_to_q6_14 = DATA_WIDTH'(mag);
            end
        end
    endfunction

    function automatic logic [5:0] leading_pos_fixed(input logic [DATA_ABS_WIDTH-1:0] value);
        int i;
        logic found;
        begin
            leading_pos_fixed = '0;
            found = 1'b0;
            for (i = DATA_ABS_WIDTH - 1; i >= 0; i = i - 1) begin
                if (value[i] && !found) begin
                    leading_pos_fixed = 6'(i);
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic logic [DATA_ABS_WIDTH-1:0] fixed_abs(input logic signed [DATA_WIDTH-1:0] q);
        begin
            fixed_abs = q[DATA_WIDTH-1] ?
                        (DATA_ABS_WIDTH'(~q) + DATA_ABS_WIDTH'(1)) :
                        DATA_ABS_WIDTH'(q);
        end
    endfunction

    function automatic logic [15:0] q6_14_parts_to_fp16(
        input logic        sign,
        input logic [DATA_ABS_WIDTH-1:0] abs_q,
        input logic [5:0]  p
    );
        logic [4:0] exp16;
        logic [9:0] frac16;
        logic [DATA_ABS_WIDTH-1:0] leading;
        logic [31:0] frac_calc;
        logic [11:0] mant_round;
        int signed exp_i;
        int shift;
        begin
            q6_14_parts_to_fp16 = {sign, 15'd0};
            exp16 = '0;
            frac16 = '0;
            leading = '0;
            frac_calc = '0;
            mant_round = '0;
            exp_i = 0;
            shift = 0;

            if (abs_q != '0) begin
                exp_i = int'(p) - DATA_FRAC + 15;
                if (exp_i <= 0) begin
                    q6_14_parts_to_fp16 = {sign, 15'd0};
                end else begin
                    exp16 = exp_i[4:0];
                    if (p >= 6'd10) begin
                        shift = int'(p) - 10;
                        if (shift == 0) begin
                            mant_round = 12'(abs_q);
                        end else begin
                            mant_round =
                                12'((abs_q + (DATA_ABS_WIDTH'(1) << (shift - 1))) >> shift);
                        end
                    end else begin
                        mant_round = 12'(abs_q << (10 - int'(p)));
                    end
                    if (mant_round[11]) begin
                        exp16 = exp16 + 5'd1;
                        frac16 = 10'd0;
                    end else begin
                        frac16 = mant_round[9:0];
                    end
                    if (exp16 >= 5'h1f)
                        q6_14_parts_to_fp16 = {sign, 5'h1f, 10'd0};
                    else
                        q6_14_parts_to_fp16 = {sign, exp16, frac16};
                end
            end
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_q       <= 1'b0;
            lut_state_q    <= LUT_IDLE;
            error_q        <= 1'b0;
            elem_count_q   <= '0;
            position_q     <= '0;
            lut_capture_valid_q <= 1'b0;
            lut_capture_sin_q <= 1'b0;
            cos_cache_q    <= '0;
            sin_cache_q    <= '0;
            pack_data_q    <= '0;
            pack_count_q   <= '0;
            pack_valid_q   <= 1'b0;
            pack_last_q    <= 1'b0;
            input_count_q  <= '0;
            unpack_data_q  <= '0;
            unpack_valid_q <= 1'b0;
            unpack_lane_q  <= '0;
            output_count_q <= '0;
            scalar_valid_q <= 1'b0;
            scalar_sign_q  <= 1'b0;
            scalar_abs_q   <= '0;
            scalar_lead_q  <= '0;
            scalar_user_q  <= '0;
            emit_valid_q   <= 1'b0;
            emit_data_q    <= '0;
            emit_user_q    <= '0;
            done_o         <= 1'b0;
        end else begin
            done_o <= 1'b0;
            lut_capture_valid_q <= lut_rd_en_o;
            lut_capture_sin_q <= (lut_state_q == LUT_LOAD_SIN);

            if (lut_capture_valid_q) begin
                if (lut_capture_sin_q)
                    sin_cache_q <= lut_rd_data_i[N_PAIRS*COEF_WIDTH-1:0];
                else
                    cos_cache_q <= lut_rd_data_i[N_PAIRS*COEF_WIDTH-1:0];
            end

            if (start_i) begin
                active_q       <= 1'b0;
                lut_state_q    <= ((elem_count_i != '0) &&
                                   ((elem_count_i % HEAD_DIM_COUNT) == '0) &&
                                   (position_i < POS_WIDTH'(MAX_SEQ_LEN)) &&
                                   !active_q &&
                                   (lut_state_q == LUT_IDLE)) ? LUT_LOAD_COS : LUT_IDLE;
                error_q        <= (elem_count_i == '0) ||
                                  ((elem_count_i % HEAD_DIM_COUNT) != '0) ||
                                  (position_i >= POS_WIDTH'(MAX_SEQ_LEN)) ||
                                  active_q ||
                                  (lut_state_q != LUT_IDLE);
                elem_count_q   <= elem_count_i;
                position_q     <= position_i;
                lut_capture_valid_q <= 1'b0;
                lut_capture_sin_q <= 1'b0;
                pack_data_q    <= '0;
                pack_count_q   <= '0;
                pack_valid_q   <= 1'b0;
                pack_last_q    <= 1'b0;
                input_count_q  <= '0;
                unpack_data_q  <= '0;
                unpack_valid_q <= 1'b0;
                unpack_lane_q  <= '0;
                output_count_q <= '0;
                scalar_valid_q <= 1'b0;
                scalar_sign_q  <= 1'b0;
                scalar_abs_q   <= '0;
                scalar_lead_q  <= '0;
                scalar_user_q  <= '0;
                emit_valid_q   <= 1'b0;
                emit_data_q    <= '0;
                emit_user_q    <= '0;
                if ((elem_count_i == '0) ||
                    ((elem_count_i % HEAD_DIM_COUNT) != '0) ||
                    (position_i >= POS_WIDTH'(MAX_SEQ_LEN)) ||
                    active_q ||
                    (lut_state_q != LUT_IDLE))
                    done_o <= 1'b1;
            end else begin
                unique case (lut_state_q)
                    LUT_LOAD_COS: begin
                        lut_state_q <= LUT_LOAD_SIN;
                    end
                    LUT_LOAD_SIN: begin
                        lut_state_q <= LUT_WAIT_LAST;
                    end
                    LUT_WAIT_LAST: begin
                        if (lut_capture_valid_q && lut_capture_sin_q) begin
                            lut_state_q <= LUT_IDLE;
                            active_q <= 1'b1;
                        end
                    end
                    default: begin
                        lut_state_q <= LUT_IDLE;
                    end
                endcase

                if (engine_in_fire_w)
                    pack_valid_q <= 1'b0;

                if (accept_scalar_w) begin
                    pack_data_q[pack_count_q * DATA_WIDTH +: DATA_WIDTH] <= fp16_to_q6_14(in_data_i);
                    input_count_q <= input_count_q + USER_WIDTH'(1);
                    if (pack_count_q == PACK_CNT_WIDTH'(LANES - 1)) begin
                        pack_count_q <= '0;
                        pack_valid_q <= 1'b1;
                        pack_last_q  <= last_scalar_in_w;
                    end else begin
                        pack_count_q <= pack_count_q + PACK_CNT_WIDTH'(1);
                    end
                end

                if (engine_out_fire_w) begin
                    unpack_data_q  <= engine_m_data_w;
                    unpack_valid_q <= 1'b1;
                    unpack_lane_q  <= '0;
                end

                if (output_fire_w && !emit_load_w)
                    emit_valid_q <= 1'b0;

                if (emit_load_w && !scalar_load_w)
                    scalar_valid_q <= 1'b0;

                if (scalar_load_w) begin
                    logic signed [DATA_WIDTH-1:0] scalar_sample;
                    logic [DATA_ABS_WIDTH-1:0] scalar_abs;
                    scalar_sample = $signed(unpack_data_q[unpack_lane_q * DATA_WIDTH +: DATA_WIDTH]);
                    scalar_abs = fixed_abs(scalar_sample);
                    scalar_valid_q <= 1'b1;
                    scalar_sign_q  <= scalar_sample[DATA_WIDTH-1];
                    scalar_abs_q   <= scalar_abs;
                    scalar_lead_q  <= leading_pos_fixed(scalar_abs);
                    scalar_user_q  <= output_count_q;
                    output_count_q <= output_count_q + USER_WIDTH'(1);
                    if (unpack_lane_q == OUT_LANE_WIDTH'(LANES - 1)) begin
                        unpack_valid_q <= 1'b0;
                        unpack_lane_q  <= '0;
                    end else begin
                        unpack_lane_q <= unpack_lane_q + OUT_LANE_WIDTH'(1);
                    end
                end

                if (emit_load_w) begin
                    emit_valid_q <= 1'b1;
                    emit_data_q  <= q6_14_parts_to_fp16(scalar_sign_q, scalar_abs_q, scalar_lead_q);
                    emit_user_q  <= scalar_user_q;
                end

                if (emit_last_fire_w) begin
                    active_q <= 1'b0;
                    done_o   <= 1'b1;
                end
            end
        end
    end

`ifndef SYNTHESIS
    property p_no_start_while_busy;
        @(posedge clk) disable iff (!rst_n) (active_q || (lut_state_q != LUT_IDLE)) |-> !start_i;
    endproperty
    assert property (p_no_start_while_busy);
`endif

    initial begin
        if (HEAD_DIM % 2 != 0)
            $error("rope_stream_unit expects even HEAD_DIM");
        if ((HEAD_DIM / 2) % PAIRS_PER_CYCLE != 0)
            $error("rope_stream_unit expects HEAD_DIM/2 divisible by PAIRS_PER_CYCLE");
        if ((HEAD_DIM / 2) * COEF_WIDTH > SPM_DATA_WIDTH)
            $error("rope_stream_unit expects one external cos/sin line per position");
    end

endmodule

`endif
