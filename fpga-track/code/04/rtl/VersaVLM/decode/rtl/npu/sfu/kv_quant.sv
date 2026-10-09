`ifndef KV_QUANT_SV
`define KV_QUANT_SV

module kv_quant #(
    parameter int USER_WIDTH  = 16,
    parameter int BLOCK_ELEMS = 128,
    parameter int BANK_NUM    = 2,

    localparam int INDEX_WIDTH = (BLOCK_ELEMS <= 1) ? 1 : $clog2(BLOCK_ELEMS),
    localparam int BANK_INDEX_WIDTH = (BANK_NUM <= 1) ? 1 : $clog2(BANK_NUM)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         req_en_i,
    input  logic                         clear_i,
    input  logic [USER_WIDTH-1:0]        elem_count_i,

    input  logic                         valid_i,
    input  logic [15:0]                  fp16_i,
    input  logic [USER_WIDTH-1:0]        user_i,
    output logic                         ready_o,

    output logic                         scale_valid_o,
    output logic [15:0]                  scale_fp16_o,
    output logic [USER_WIDTH-1:0]        scale_user_o,
    input  logic                         scale_ready_i,

    output logic                         quant_valid_o,
    output logic signed [7:0]            quant_i8_o,
    output logic [USER_WIDTH-1:0]        quant_user_o,
    input  logic                         quant_ready_i,

    output logic                         busy_o,
    output logic                         block_done_o,
    output logic                         done_o
);

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_SCALE,
        ST_EMIT
    } out_state_t;

    localparam int MEM_DEPTH = BANK_NUM * BLOCK_ELEMS;
    localparam int MEM_ADDR_WIDTH = (MEM_DEPTH <= 1) ? 1 : $clog2(MEM_DEPTH);
    localparam int BLOCK_COUNT_WIDTH = $clog2(BLOCK_ELEMS + 1);
    localparam int RECIP_FRAC_BITS = 32;
    localparam int RECIP_WIDTH = 23;
    localparam int NUM_WIDTH = 18;
    localparam int PRODUCT_WIDTH = NUM_WIDTH + RECIP_WIDTH;
    localparam int PIPE_STAGES = 5;

    localparam logic [BLOCK_COUNT_WIDTH-1:0] FULL_BLOCK_ELEMS =
        BLOCK_COUNT_WIDTH'(BLOCK_ELEMS);

    out_state_t state_q;

    (* ram_style = "block" *) logic [15:0] sample_mem_q [0:MEM_DEPTH-1];
    (* ram_style = "block" *) logic [USER_WIDTH-1:0] user_mem_q [0:MEM_DEPTH-1];
    (* rom_style = "block" *) logic [RECIP_WIDTH-1:0] recip_rom_q [0:1023];

    logic [BANK_NUM-1:0] bank_full_q;
    logic [BANK_NUM-1:0] bank_pending_q;
    logic [BANK_INDEX_WIDTH-1:0] write_bank_q;
    logic [BANK_INDEX_WIDTH-1:0] read_bank_q;
    logic [BANK_INDEX_WIDTH-1:0] read_bank_next_w;

    logic [INDEX_WIDTH-1:0] collect_count_q;
    logic [BLOCK_COUNT_WIDTH-1:0] emit_count_q;
    logic [BLOCK_COUNT_WIDTH-1:0] collect_target_q;
    logic [BLOCK_COUNT_WIDTH-1:0] bank_elem_count_q [0:BANK_NUM-1];
    logic [14:0] bank_max_abs_q [0:BANK_NUM-1];
    logic [15:0] bank_scale_q [0:BANK_NUM-1];
    logic [USER_WIDTH-1:0] bank_user_q [0:BANK_NUM-1];

    logic [USER_WIDTH-1:0] elem_count_q;
    logic [USER_WIDTH-1:0] accepted_count_q;
    logic [USER_WIDTH-1:0] emitted_count_q;
    logic [14:0] collect_max_abs_q;
    logic scale_calc_valid_q;
    logic [BANK_INDEX_WIDTH-1:0] scale_calc_bank_q;
    logic [14:0] scale_calc_max_abs_q;
    logic scale_calc_s1_valid_q;
    logic [BANK_INDEX_WIDTH-1:0] scale_calc_s1_bank_q;
    logic scale_calc_s1_zero_q;
    logic scale_calc_s1_inf_q;
    logic [11:0] scale_calc_s1_mant_scaled_q;
    logic signed [6:0] scale_calc_s1_exp_out_q;

    logic input_fire_w;
    logic scale_fire_w;
    logic quant_fire_w;
    logic issue_fire_w;
    logic pipe_advance_w;
    logic [14:0] input_abs_mag_w;
    logic [14:0] max_abs_next_w;
    logic [USER_WIDTH-1:0] remaining_count_w;
    logic [BLOCK_COUNT_WIDTH-1:0] next_block_elems_w;
    logic [BLOCK_COUNT_WIDTH-1:0] active_block_elems_w;
    logic collect_last_w;
    logic issue_last_w;
    logic output_last_w;
    logic final_output_w;
    logic next_read_bank_full_after_w;
    logic has_full_bank_w;
    logic [MEM_ADDR_WIDTH-1:0] write_addr_w;
    logic [MEM_ADDR_WIDTH-1:0] read_addr_w;
    logic [4:0] scale_calc_exp_w;
    logic [10:0] scale_calc_mant_w;
    logic [31:0] scale_calc_recip_product_w;
    logic [11:0] scale_calc_mant_scaled_w;
    logic signed [6:0] scale_calc_exp_out_w;

    logic pipe_valid_q [0:PIPE_STAGES-1];
    logic pipe_last_q [0:PIPE_STAGES-1];
    logic [USER_WIDTH-1:0] pipe_user_q [0:PIPE_STAGES-1];

    logic [15:0] pipe0_sample_q;
    logic [14:0] pipe0_max_abs_q;

    logic pipe1_sign_q;
    logic pipe1_zero_q;
    logic [NUM_WIDTH-1:0] pipe1_num_round_q;
    logic [9:0] pipe1_recip_idx_q;

    logic pipe2_sign_q;
    logic pipe2_zero_q;
    logic [NUM_WIDTH-1:0] pipe2_num_round_q;
    logic [RECIP_WIDTH-1:0] pipe2_recip_q;

    logic pipe3_sign_q;
    logic pipe3_zero_q;
    logic [PRODUCT_WIDTH-1:0] pipe3_product_q;

    logic signed [7:0] pipe4_quant_q;
    logic issue_valid_d_q;
    logic issue_last_d_q;
    logic [15:0] sample_rd_q;
    logic [USER_WIDTH-1:0] user_rd_q;
    logic [14:0] max_abs_rd_q;

    initial begin
        for (int idx = 0; idx < 1024; idx++) begin
            recip_rom_q[idx] = RECIP_WIDTH'(((64'h1 << RECIP_FRAC_BITS) +
                                             64'(idx + 1024) - 1) /
                                            64'(idx + 1024));
        end
    end

    function automatic logic [15:0] fp16_div_pow2(
        input logic [15:0] value,
        input int unsigned shift
    );
        logic [4:0] exp;
        logic [10:0] mant;
        logic [10:0] rounded_mant;
        int unsigned sub_shift;
        begin
            exp = value[14:10];
            mant = {1'b1, value[9:0]};
            fp16_div_pow2 = value;
            fp16_div_pow2[15] = 1'b0;

            if (value[14:0] == 15'h0000) begin
                fp16_div_pow2 = 16'h0000;
            end else if (exp == 5'h1f) begin
                fp16_div_pow2 = {1'b0, 5'h1f, 10'h000};
            end else if (exp <= shift[4:0]) begin
                sub_shift = shift - int'(exp) + 1;
                if (sub_shift >= 11) begin
                    fp16_div_pow2 = 16'h0000;
                end else begin
                    rounded_mant = (mant + (11'(1) << (sub_shift - 1))) >> sub_shift;
                    fp16_div_pow2 = rounded_mant[10] ?
                                    {1'b0, 5'h01, 10'h000} :
                                    {1'b0, 5'h00, rounded_mant[9:0]};
                end
            end else begin
                fp16_div_pow2[14:10] = exp - shift[4:0];
            end
        end
    endfunction

    function automatic logic [15:0] fp16_div_127(input logic [15:0] value);
        logic [4:0] exp;
        logic [10:0] mant;
        logic [11:0] mant_scaled;
        logic [11:0] mant_norm;
        logic [11:0] sub_round;
        logic [11:0] sub_mant;
        logic [31:0] recip_product;
        int signed exp_out;
        int unsigned sub_shift;
        begin
            exp = value[14:10];
            mant = {1'b1, value[9:0]};
            fp16_div_127 = {1'b0, value[14:0]};

            if (value[14:0] == 15'h0000) begin
                fp16_div_127 = 16'h0000;
            end else if (exp == 5'h1f) begin
                fp16_div_127 = {1'b0, 5'h1f, 10'h000};
            end else if (exp == 5'h00) begin
                fp16_div_127 = 16'h0000;
            end else begin
                recip_product = ((32'(mant) << 7) * 32'd129) + 32'd8192;
                mant_scaled = 12'(recip_product >> 14);
                exp_out = int'(exp) - 7;
                if (mant_scaled >= 12'd2048) begin
                    mant_norm = (mant_scaled + 12'd1) >> 1;
                    exp_out = exp_out + 1;
                end else begin
                    mant_norm = mant_scaled;
                end

                if (exp_out >= 31) begin
                    fp16_div_127 = {1'b0, 5'h1f, 10'h000};
                end else if (exp_out >= 1) begin
                    fp16_div_127 = {1'b0, 5'(exp_out), mant_norm[9:0]};
                end else begin
                    sub_shift = int'(1 - exp_out);
                    if (sub_shift >= 12) begin
                        fp16_div_127 = 16'h0000;
                    end else begin
                        sub_round = mant_norm + (12'd1 << (sub_shift - 1));
                        sub_mant = sub_round >> sub_shift;
                        fp16_div_127 = sub_mant[10] ?
                                        {1'b0, 5'h01, 10'h000} :
                                        {1'b0, 5'h00, sub_mant[9:0]};
                    end
                end
            end
        end
    endfunction

    function automatic logic [15:0] fp16_div_127_finish(
        input logic zero,
        input logic inf,
        input logic [11:0] mant_scaled,
        input logic signed [6:0] exp_out_base
    );
        logic [11:0] mant_norm;
        logic [11:0] sub_round;
        logic [11:0] sub_mant;
        logic signed [6:0] exp_norm;
        int unsigned sub_shift;
        begin
            if (zero) begin
                fp16_div_127_finish = 16'h0000;
            end else if (inf) begin
                fp16_div_127_finish = {1'b0, 5'h1f, 10'h000};
            end else begin
                if (mant_scaled >= 12'd2048) begin
                    mant_norm = (mant_scaled + 12'd1) >> 1;
                    exp_norm = exp_out_base + 7'sd1;
                end else begin
                    mant_norm = mant_scaled;
                    exp_norm = exp_out_base;
                end

                if (exp_norm >= 7'sd31) begin
                    fp16_div_127_finish = {1'b0, 5'h1f, 10'h000};
                end else if (exp_norm >= 7'sd1) begin
                    fp16_div_127_finish = {1'b0, exp_norm[4:0], mant_norm[9:0]};
                end else begin
                    sub_shift = int'(7'sd1 - exp_norm);
                    if (sub_shift >= 12) begin
                        fp16_div_127_finish = 16'h0000;
                    end else begin
                        sub_round = mant_norm + (12'd1 << (sub_shift - 1));
                        sub_mant = sub_round >> sub_shift;
                        fp16_div_127_finish = sub_mant[10] ?
                                              {1'b0, 5'h01, 10'h000} :
                                              {1'b0, 5'h00, sub_mant[9:0]};
                    end
                end
            end
        end
    endfunction

    function automatic logic [NUM_WIDTH-1:0] shifted_numerator(
        input logic [10:0] value_mant,
        input int signed exp_diff
    );
        logic [31:0] numer;
        int unsigned shift_amount;
        begin
            numer = 32'(value_mant) << 7;
            if (exp_diff > 0) begin
                shift_amount = int'(exp_diff);
                numer = (shift_amount >= 14) ? 32'h0003_ffff : (numer << shift_amount);
                if (numer > 32'h0003_ffff)
                    numer = 32'h0003_ffff;
            end else if (exp_diff < 0) begin
                shift_amount = int'(-exp_diff);
                if (shift_amount >= NUM_WIDTH) begin
                    numer = '0;
                end else begin
                    numer = (numer + (32'h1 << (shift_amount - 1))) >> shift_amount;
                end
            end
            shifted_numerator = NUM_WIDTH'(numer);
        end
    endfunction

    function automatic logic [NUM_WIDTH-1:0] rounded_numerator(
        input logic [15:0] value,
        input logic [14:0] max_abs
    );
        logic [4:0] value_exp;
        logic [4:0] max_exp;
        logic [10:0] value_mant;
        logic [10:0] max_mant;
        logic [NUM_WIDTH-1:0] numer;
        int signed exp_diff;
        begin
            value_exp = value[14:10];
            max_exp = max_abs[14:10];
            value_mant = {1'b1, value[9:0]};
            max_mant = {1'b1, max_abs[9:0]};
            exp_diff = int'(value_exp) - int'(max_exp);
            numer = shifted_numerator(value_mant, exp_diff);
            rounded_numerator = numer + NUM_WIDTH'({1'b0, max_mant[10:1]});
        end
    endfunction

    function automatic logic signed [7:0] saturate_quant(
        input logic sign,
        input logic zero,
        input logic [PRODUCT_WIDTH-1:0] product
    );
        logic [8:0] q_abs;
        logic [6:0] q_sat;
        begin
            q_abs = product[RECIP_FRAC_BITS +: 9];
            if (zero) begin
                q_sat = 7'd0;
            end else if (q_abs > 9'd127) begin
                q_sat = 7'd127;
            end else begin
                q_sat = q_abs[6:0];
            end

            if (sign && (q_sat != 7'd0))
                saturate_quant = -$signed({1'b0, q_sat});
            else
                saturate_quant = $signed({1'b0, q_sat});
        end
    endfunction

    function automatic logic [BANK_INDEX_WIDTH-1:0] bank_next(
        input logic [BANK_INDEX_WIDTH-1:0] bank
    );
        begin
            bank_next = (bank == BANK_INDEX_WIDTH'(BANK_NUM - 1)) ?
                        '0 : (bank + BANK_INDEX_WIDTH'(1));
        end
    endfunction

    assign busy_o = (elem_count_q != '0) &&
                    ((accepted_count_q < elem_count_q) ||
                     (emitted_count_q < elem_count_q) ||
                     (state_q != ST_IDLE) ||
                     scale_calc_valid_q ||
                     scale_calc_s1_valid_q ||
                     (|bank_pending_q) ||
                     pipe_valid_q[0] || pipe_valid_q[1] ||
                     pipe_valid_q[2] || pipe_valid_q[3] ||
                     pipe_valid_q[4]);

    assign remaining_count_w = elem_count_q - accepted_count_q;
    assign next_block_elems_w = (remaining_count_w > USER_WIDTH'(BLOCK_ELEMS)) ?
                                FULL_BLOCK_ELEMS :
                                BLOCK_COUNT_WIDTH'(remaining_count_w);
    assign active_block_elems_w = (collect_count_q == '0) ? next_block_elems_w : collect_target_q;

    assign ready_o = busy_o &&
                     (accepted_count_q < elem_count_q) &&
                     !bank_full_q[write_bank_q] &&
                     !bank_pending_q[write_bank_q] &&
                     (active_block_elems_w != '0);

    assign input_fire_w  = valid_i && ready_o;
    assign scale_fire_w  = scale_valid_o && scale_ready_i;
    assign quant_fire_w  = quant_valid_o && quant_ready_i;
    assign input_abs_mag_w = fp16_i[14:0];
    assign max_abs_next_w =
        (input_abs_mag_w > collect_max_abs_q) ? input_abs_mag_w : collect_max_abs_q;
    assign collect_last_w = input_fire_w &&
                            ((BLOCK_COUNT_WIDTH'(collect_count_q) + BLOCK_COUNT_WIDTH'(1)) ==
                             active_block_elems_w);
    assign pipe_advance_w = quant_ready_i || !pipe_valid_q[PIPE_STAGES-1];
    assign issue_fire_w = (state_q == ST_EMIT) &&
                          pipe_advance_w &&
                          (BLOCK_COUNT_WIDTH'(emit_count_q) < bank_elem_count_q[read_bank_q]);
    assign issue_last_w = issue_fire_w &&
                          ((BLOCK_COUNT_WIDTH'(emit_count_q) + BLOCK_COUNT_WIDTH'(1)) ==
                           bank_elem_count_q[read_bank_q]);
    assign output_last_w = quant_fire_w && pipe_last_q[PIPE_STAGES-1];
    assign final_output_w = quant_fire_w &&
                            ((emitted_count_q + USER_WIDTH'(1)) == elem_count_q);
    assign has_full_bank_w = |bank_full_q;
    assign read_bank_next_w = bank_next(read_bank_q);
    assign next_read_bank_full_after_w =
        bank_full_q[read_bank_next_w] ||
        (scale_calc_s1_valid_q && (scale_calc_s1_bank_q == read_bank_next_w));
    assign write_addr_w = {write_bank_q, collect_count_q};
    assign read_addr_w = {read_bank_q, emit_count_q[INDEX_WIDTH-1:0]};
    assign scale_calc_exp_w = scale_calc_max_abs_q[14:10];
    assign scale_calc_mant_w = {1'b1, scale_calc_max_abs_q[9:0]};
    assign scale_calc_recip_product_w =
        ((32'(scale_calc_mant_w) << 7) * 32'd129) + 32'd8192;
    assign scale_calc_mant_scaled_w = 12'(scale_calc_recip_product_w >> 14);
    assign scale_calc_exp_out_w = $signed({2'b00, scale_calc_exp_w}) - 7'sd7;

    assign scale_valid_o = (state_q == ST_SCALE);
    assign scale_fp16_o  = bank_scale_q[read_bank_q];
    assign scale_user_o  = bank_user_q[read_bank_q];

    assign quant_valid_o = pipe_valid_q[PIPE_STAGES-1];
    assign quant_i8_o    = pipe4_quant_q;
    assign quant_user_o  = pipe_user_q[PIPE_STAGES-1];

    always_ff @(posedge clk) begin
        if (input_fire_w) begin
            sample_mem_q[write_addr_w] <= fp16_i;
            user_mem_q[write_addr_w]   <= user_i;
        end

        if (pipe_advance_w) begin
            sample_rd_q <= sample_mem_q[read_addr_w];
            user_rd_q   <= user_mem_q[read_addr_w];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q             <= ST_IDLE;
            bank_full_q         <= '0;
            bank_pending_q      <= '0;
            write_bank_q        <= 1'b0;
            read_bank_q         <= 1'b0;
            collect_count_q     <= '0;
            emit_count_q        <= '0;
            collect_target_q    <= '0;
            elem_count_q        <= '0;
            accepted_count_q    <= '0;
            emitted_count_q     <= '0;
            collect_max_abs_q   <= '0;
            scale_calc_valid_q  <= 1'b0;
            scale_calc_bank_q   <= '0;
            scale_calc_max_abs_q <= '0;
            scale_calc_s1_valid_q <= 1'b0;
            scale_calc_s1_bank_q  <= '0;
            scale_calc_s1_zero_q  <= 1'b1;
            scale_calc_s1_inf_q   <= 1'b0;
            scale_calc_s1_mant_scaled_q <= '0;
            scale_calc_s1_exp_out_q <= '0;
            block_done_o        <= 1'b0;
            done_o              <= 1'b0;
            pipe0_sample_q      <= '0;
            pipe0_max_abs_q     <= '0;
            pipe1_sign_q        <= 1'b0;
            pipe1_zero_q        <= 1'b1;
            pipe1_num_round_q   <= '0;
            pipe1_recip_idx_q   <= '0;
            pipe2_sign_q        <= 1'b0;
            pipe2_zero_q        <= 1'b1;
            pipe2_num_round_q   <= '0;
            pipe2_recip_q       <= '0;
            pipe3_sign_q        <= 1'b0;
            pipe3_zero_q        <= 1'b1;
            pipe3_product_q     <= '0;
            pipe4_quant_q       <= '0;
            issue_valid_d_q     <= 1'b0;
            issue_last_d_q      <= 1'b0;
            max_abs_rd_q        <= '0;
            for (int st = 0; st < PIPE_STAGES; st++) begin
                pipe_valid_q[st] <= 1'b0;
                pipe_last_q[st]  <= 1'b0;
                pipe_user_q[st]  <= '0;
            end
            for (int bank = 0; bank < BANK_NUM; bank++) begin
                bank_elem_count_q[bank] <= '0;
                bank_max_abs_q[bank]    <= '0;
                bank_scale_q[bank]      <= '0;
                bank_user_q[bank]       <= '0;
            end
        end else begin
            block_done_o <= 1'b0;
            done_o       <= 1'b0;

            if (clear_i) begin
                state_q             <= ST_IDLE;
                bank_full_q         <= '0;
                bank_pending_q      <= '0;
                write_bank_q        <= 1'b0;
                read_bank_q         <= 1'b0;
                collect_count_q     <= '0;
                emit_count_q        <= '0;
                collect_target_q    <= '0;
                elem_count_q        <= '0;
                accepted_count_q    <= '0;
                emitted_count_q     <= '0;
                collect_max_abs_q   <= '0;
                scale_calc_valid_q  <= 1'b0;
                scale_calc_bank_q   <= '0;
                scale_calc_max_abs_q <= '0;
                scale_calc_s1_valid_q <= 1'b0;
                scale_calc_s1_bank_q  <= '0;
                scale_calc_s1_zero_q  <= 1'b1;
                scale_calc_s1_inf_q   <= 1'b0;
                scale_calc_s1_mant_scaled_q <= '0;
                scale_calc_s1_exp_out_q <= '0;
                issue_valid_d_q     <= 1'b0;
                issue_last_d_q      <= 1'b0;
                for (int st = 0; st < PIPE_STAGES; st++)
                    pipe_valid_q[st] <= 1'b0;
            end else if (req_en_i) begin
                state_q             <= ST_IDLE;
                bank_full_q         <= '0;
                bank_pending_q      <= '0;
                write_bank_q        <= 1'b0;
                read_bank_q         <= 1'b0;
                collect_count_q     <= '0;
                emit_count_q        <= '0;
                collect_target_q    <= '0;
                elem_count_q        <= elem_count_i;
                accepted_count_q    <= '0;
                emitted_count_q     <= '0;
                collect_max_abs_q   <= '0;
                scale_calc_valid_q  <= 1'b0;
                scale_calc_bank_q   <= '0;
                scale_calc_max_abs_q <= '0;
                scale_calc_s1_valid_q <= 1'b0;
                scale_calc_s1_bank_q  <= '0;
                scale_calc_s1_zero_q  <= 1'b1;
                scale_calc_s1_inf_q   <= 1'b0;
                scale_calc_s1_mant_scaled_q <= '0;
                scale_calc_s1_exp_out_q <= '0;
                done_o              <= (elem_count_i == '0);
                issue_valid_d_q     <= 1'b0;
                issue_last_d_q      <= 1'b0;
                for (int st = 0; st < PIPE_STAGES; st++)
                    pipe_valid_q[st] <= 1'b0;
            end else begin
                if (scale_calc_valid_q) begin
                    scale_calc_s1_bank_q        <= scale_calc_bank_q;
                    scale_calc_s1_zero_q        <= (scale_calc_max_abs_q == 15'h0000) ||
                                                   (scale_calc_exp_w == 5'h00);
                    scale_calc_s1_inf_q         <= (scale_calc_exp_w == 5'h1f);
                    scale_calc_s1_mant_scaled_q <= scale_calc_mant_scaled_w;
                    scale_calc_s1_exp_out_q     <= scale_calc_exp_out_w;
                end
                scale_calc_s1_valid_q <= scale_calc_valid_q;

                if (scale_calc_s1_valid_q) begin
                    bank_scale_q[scale_calc_s1_bank_q] <=
                        fp16_div_127_finish(scale_calc_s1_zero_q,
                                            scale_calc_s1_inf_q,
                                            scale_calc_s1_mant_scaled_q,
                                            scale_calc_s1_exp_out_q);
                    bank_full_q[scale_calc_s1_bank_q]    <= 1'b1;
                    bank_pending_q[scale_calc_s1_bank_q] <= 1'b0;
                end
                scale_calc_valid_q <= 1'b0;

                if (input_fire_w) begin
                    collect_max_abs_q <= max_abs_next_w;
                    accepted_count_q  <= accepted_count_q + USER_WIDTH'(1);

                    if (collect_count_q == '0) begin
                        bank_user_q[write_bank_q] <= user_i;
                        collect_target_q          <= next_block_elems_w;
                    end

                    if (collect_last_w) begin
                        bank_pending_q[write_bank_q]    <= 1'b1;
                        bank_elem_count_q[write_bank_q] <= active_block_elems_w;
                        bank_max_abs_q[write_bank_q]    <= max_abs_next_w;
                        scale_calc_valid_q              <= 1'b1;
                        scale_calc_bank_q               <= write_bank_q;
                        scale_calc_max_abs_q            <= max_abs_next_w;
                        collect_count_q                 <= '0;
                        collect_target_q                <= '0;
                        collect_max_abs_q               <= '0;
                        write_bank_q                    <= bank_next(write_bank_q);
                    end else begin
                        collect_count_q <= collect_count_q + 1'b1;
                    end
                end

                if (pipe_advance_w) begin
                    issue_valid_d_q <= issue_fire_w;
                    issue_last_d_q  <= issue_last_w;
                    max_abs_rd_q    <= bank_max_abs_q[read_bank_q];

                    pipe_valid_q[0] <= issue_valid_d_q;
                    pipe_last_q[0]  <= issue_last_d_q;
                    pipe_user_q[0]  <= user_rd_q;
                    pipe0_sample_q  <= sample_rd_q;
                    pipe0_max_abs_q <= max_abs_rd_q;

                    pipe_valid_q[1] <= pipe_valid_q[0];
                    pipe_last_q[1]  <= pipe_last_q[0];
                    pipe_user_q[1]  <= pipe_user_q[0];
                    pipe1_sign_q    <= pipe0_sample_q[15];
                    pipe1_zero_q    <= (pipe0_max_abs_q == 15'h0000) ||
                                       (pipe0_sample_q[14:0] == 15'h0000) ||
                                       (pipe0_sample_q[14:10] == 5'h00) ||
                                       (pipe0_max_abs_q[14:10] == 5'h00);
                    pipe1_num_round_q <= rounded_numerator(pipe0_sample_q, pipe0_max_abs_q);
                    pipe1_recip_idx_q <= pipe0_max_abs_q[9:0];

                    pipe_valid_q[2]   <= pipe_valid_q[1];
                    pipe_last_q[2]    <= pipe_last_q[1];
                    pipe_user_q[2]    <= pipe_user_q[1];
                    pipe2_sign_q      <= pipe1_sign_q;
                    pipe2_zero_q      <= pipe1_zero_q;
                    pipe2_num_round_q <= pipe1_num_round_q;
                    pipe2_recip_q     <= recip_rom_q[pipe1_recip_idx_q];

                    pipe_valid_q[3] <= pipe_valid_q[2];
                    pipe_last_q[3]  <= pipe_last_q[2];
                    pipe_user_q[3]  <= pipe_user_q[2];
                    pipe3_sign_q    <= pipe2_sign_q;
                    pipe3_zero_q    <= pipe2_zero_q;
                    pipe3_product_q <= pipe2_num_round_q * pipe2_recip_q;

                    pipe_valid_q[4] <= pipe_valid_q[3];
                    pipe_last_q[4]  <= pipe_last_q[3];
                    pipe_user_q[4]  <= pipe_user_q[3];
                    pipe4_quant_q   <= saturate_quant(pipe3_sign_q,
                                                       pipe3_zero_q,
                                                       pipe3_product_q);
                end

                if (issue_fire_w)
                    emit_count_q <= emit_count_q + 1'b1;

                unique case (state_q)
                    ST_IDLE: begin
                        if (bank_full_q[read_bank_q]) begin
                            state_q     <= ST_SCALE;
                        end
                    end

                    ST_SCALE: begin
                        if (scale_fire_w) begin
                            emit_count_q <= '0;
                            state_q      <= ST_EMIT;
                        end
                    end

                    ST_EMIT: begin
                        if (quant_fire_w)
                            emitted_count_q <= emitted_count_q + USER_WIDTH'(1);

                        if (output_last_w) begin
                            block_done_o <= 1'b1;
                            bank_full_q[read_bank_q] <= 1'b0;
                            emit_count_q <= '0;

                            if (final_output_w) begin
                                done_o  <= 1'b1;
                                state_q <= ST_IDLE;
                            end else if (next_read_bank_full_after_w) begin
                                read_bank_q <= read_bank_next_w;
                                state_q     <= ST_SCALE;
                            end else begin
                                read_bank_q <= read_bank_next_w;
                                state_q <= ST_IDLE;
                            end
                        end
                    end

                    default: begin
                        state_q <= ST_IDLE;
                    end
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (BLOCK_ELEMS <= 0)
            $fatal(1, "kv_quant BLOCK_ELEMS must be positive");
        if (BANK_NUM <= 1)
            $fatal(1, "kv_quant BANK_NUM must be greater than one");
        if ((1 << INDEX_WIDTH) != BLOCK_ELEMS)
            $fatal(1, "kv_quant BLOCK_ELEMS must be a power of two");
        if (MEM_DEPTH > (1 << MEM_ADDR_WIDTH))
            $fatal(1, "kv_quant memory address width is too small");
    end
`endif

endmodule

`endif
