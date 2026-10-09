`ifndef KV_CACHE_WRITER_SV
`define KV_CACHE_WRITER_SV

module kv_cache_writer #(
    parameter int SPM_SIZE       = 1 << 19,
    parameter int SPM_DATA_WIDTH = 512,
    parameter int USER_WIDTH     = 16,
    parameter int CACHE_HEAD_DIM  = 64,
    parameter int CACHE_ROW_BYTES = 64,
    parameter int GEMV_TILE_ELEMS = 128,
    parameter int CACHE_HEAD_COUNT = 5,
    parameter int ROW_TILE_ELEMS = 32,
    parameter int KV_HEAD_STRIDE_BYTES = 65536,

    localparam int SPM_ADDR_WIDTH = $clog2(SPM_SIZE)
) (
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic [SPM_ADDR_WIDTH-1:0]         cfg_base_addr_i,
    input  logic [USER_WIDTH-1:0]             cfg_elem_count_i,
    input  logic [15:0]                       cfg_cache_cell_i,
    input  logic                              cfg_v_separated_i,
    input  logic                              cfg_scratch_i,
    input  logic                              req_en_i,
    output logic                              busy_o,
    output logic                              done_o,

    input  logic                              scale_valid_i,
    input  logic [15:0]                       scale_fp16_i,
    input  logic [USER_WIDTH-1:0]             scale_user_i,

    input  logic                              quant_valid_i,
    input  logic signed [7:0]                 quant_i8_i,
    input  logic [USER_WIDTH-1:0]             quant_user_i,

    output logic                              spm_wr_en_o,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_wr_addr_o,
    output logic [SPM_DATA_WIDTH-1:0]         spm_wr_data_o,
    output logic [SPM_DATA_WIDTH/8-1:0]       spm_wr_mask_o
);

    localparam int BYTE_PER_BEAT       = SPM_DATA_WIDTH / 8;
    localparam int QUANT_ELEMS_PER_BEAT = BYTE_PER_BEAT;
    localparam int K_CACHE_TILE_BYTES = BYTE_PER_BEAT + (ROW_TILE_ELEMS * CACHE_ROW_BYTES);
    localparam int K_CACHE_TILES_PER_HEAD =
        (KV_HEAD_STRIDE_BYTES / K_CACHE_TILE_BYTES) > 0 ?
        (KV_HEAD_STRIDE_BYTES / K_CACHE_TILE_BYTES) : 1;
    localparam int CACHE_WINDOW_TOKENS = K_CACHE_TILES_PER_HEAD * ROW_TILE_ELEMS;
    localparam int V_QUANT_TILE_BYTES = ROW_TILE_ELEMS * CACHE_ROW_BYTES;
    localparam int V_SCALE_REGION_BYTES =
        (((CACHE_WINDOW_TOKENS * 2) + 255) / 256) * 256;
    localparam int ROW_TILE_SHIFT = $clog2(ROW_TILE_ELEMS);
    localparam int CACHE_HEAD_DIM_SHIFT = $clog2(CACHE_HEAD_DIM);
    localparam int SCALE_LINE_SHIFT = $clog2(BYTE_PER_BEAT / 2);

    logic [SPM_ADDR_WIDTH-1:0]         base_addr_q;
    logic [USER_WIDTH-1:0]             elem_count_q;
    logic [SPM_ADDR_WIDTH-1:0]         v_quant_tile_offset_q;
    logic [SPM_ADDR_WIDTH-1:0]         quant_row_offset_q;
    logic [SPM_DATA_WIDTH-1:0]         quant_pack_q;
    logic [SPM_DATA_WIDTH/8-1:0]       quant_mask_q;
    logic                              busy_q;
    logic                              done_pending_q;

    logic                              quant_in_range_w;
    logic                              quant_last_elem_w;
    logic                              quant_flush_w;
    logic [SPM_DATA_WIDTH-1:0]         quant_pack_next_w;
    logic [SPM_DATA_WIDTH/8-1:0]       quant_mask_next_w;
    logic [SPM_ADDR_WIDTH-1:0]         quant_addr_w;
    logic [SPM_ADDR_WIDTH-1:0]         scratch_quant_addr_w;

    logic [USER_WIDTH-1:0]             cfg_cache_cell_w;
    logic [USER_WIDTH-1:0]             cfg_cache_tile_raw_w;
    logic [USER_WIDTH-1:0]             cfg_cache_row_tile_w;
    logic [4:0]                        cfg_cache_row_in_tile_w;
    logic [SPM_ADDR_WIDTH-1:0]         cfg_v_quant_tile_offset_w;
    logic [SPM_ADDR_WIDTH-1:0]         cfg_quant_row_offset_w;

    logic [USER_WIDTH-1:0] quant_head_idx_w;
    logic [USER_WIDTH-1:0] quant_elem_in_head_w;
    logic [5:0]            quant_lane_w;
    logic [SPM_ADDR_WIDTH-1:0] quant_head_offset_w;
    logic unused_cfg_v_separated_w;
    logic unused_scale_inputs_w;

    function automatic logic [USER_WIDTH-1:0] mod_cache_tile_count(
        input logic [USER_WIDTH-1:0] tile_idx
    );
        logic [6:0] folded;
        logic [6:0] red1;
        begin
            // Default layout has 31 row tiles per head. Since 2^5 == 1 (mod 31),
            // fold 5-bit groups instead of inferring a runtime modulo divider.
            folded = {2'b0, tile_idx[4:0]} +
                     {2'b0, tile_idx[9:5]} +
                     {6'b0, tile_idx[10]};
            red1 = (folded >= 7'd31) ? (folded - 7'd31) : folded;
            mod_cache_tile_count = USER_WIDTH'((red1 >= 7'd31) ? (red1 - 7'd31) : red1);
        end
    endfunction

    assign busy_o = busy_q;

    // Cache-cell layout terms are request-static; register them before writes.
    assign cfg_cache_tile_raw_w   = USER_WIDTH'(cfg_cache_cell_i >> ROW_TILE_SHIFT);
    assign cfg_cache_row_tile_w   = mod_cache_tile_count(cfg_cache_tile_raw_w);
    assign cfg_cache_cell_w       = {cfg_cache_row_tile_w[USER_WIDTH-ROW_TILE_SHIFT-1:0],
                                     cfg_cache_cell_i[ROW_TILE_SHIFT-1:0]};
    assign cfg_cache_row_in_tile_w = cfg_cache_cell_w[4:0];
    assign cfg_v_quant_tile_offset_w = SPM_ADDR_WIDTH'(cfg_cache_row_tile_w) << 11;
    assign cfg_quant_row_offset_w  = SPM_ADDR_WIDTH'(cfg_cache_row_in_tile_w) << 6;

    assign unused_cfg_v_separated_w = cfg_v_separated_i;
    assign unused_scale_inputs_w = scale_valid_i | scale_fp16_i[0] | scale_user_i[0] |
                                   unused_cfg_v_separated_w;

    assign quant_head_idx_w       = quant_user_i >> CACHE_HEAD_DIM_SHIFT;
    assign quant_elem_in_head_w   = {{(USER_WIDTH-CACHE_HEAD_DIM_SHIFT){1'b0}},
                                     quant_user_i[CACHE_HEAD_DIM_SHIFT-1:0]};
    assign quant_lane_w           = quant_elem_in_head_w[5:0];
    assign quant_head_offset_w    = SPM_ADDR_WIDTH'(quant_head_idx_w) << 16;
    assign quant_in_range_w        = quant_valid_i && busy_q && (quant_user_i < elem_count_q);
    assign quant_last_elem_w       = quant_in_range_w && (quant_user_i == (elem_count_q - 1'b1));
    assign quant_flush_w           = quant_in_range_w &&
                                     ((quant_lane_w == (QUANT_ELEMS_PER_BEAT - 1)) ||
                                      (quant_elem_in_head_w == USER_WIDTH'(CACHE_HEAD_DIM - 1)) ||
                                      quant_last_elem_w);
    assign scratch_quant_addr_w    = base_addr_q +
                                     (SPM_ADDR_WIDTH'(quant_user_i >> CACHE_HEAD_DIM_SHIFT)
                                      << CACHE_HEAD_DIM_SHIFT);
    assign quant_addr_w            = cfg_scratch_i ?
                                     scratch_quant_addr_w :
                                     (base_addr_q +
                                      quant_head_offset_w +
                                      SPM_ADDR_WIDTH'(V_SCALE_REGION_BYTES) +
                                      v_quant_tile_offset_q +
                                      quant_row_offset_q);

    always_comb begin
        quant_pack_next_w = quant_pack_q;
        quant_mask_next_w = quant_mask_q;
        if (quant_in_range_w) begin
            quant_pack_next_w[quant_lane_w*8 +: 8] = quant_i8_i;
            quant_mask_next_w[quant_lane_w] = 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            base_addr_q    <= '0;
            elem_count_q   <= '0;
            v_quant_tile_offset_q <= '0;
            quant_row_offset_q    <= '0;
            quant_mask_q   <= '0;
            busy_q         <= 1'b0;
            done_pending_q <= 1'b0;
            done_o         <= 1'b0;
            spm_wr_en_o    <= 1'b0;
            spm_wr_addr_o  <= '0;
            spm_wr_mask_o  <= '0;
        end else begin
            done_o        <= 1'b0;
            spm_wr_en_o   <= 1'b0;
            spm_wr_mask_o <= '0;

            if (req_en_i) begin
                base_addr_q    <= cfg_base_addr_i;
                elem_count_q   <= cfg_elem_count_i;
                v_quant_tile_offset_q <= cfg_v_quant_tile_offset_w;
                quant_row_offset_q    <= cfg_quant_row_offset_w;
                quant_mask_q   <= '0;
                busy_q         <= (cfg_elem_count_i != '0);
                done_pending_q <= (cfg_elem_count_i == '0);
            end else if (done_pending_q) begin
                done_o         <= 1'b1;
                busy_q         <= 1'b0;
                done_pending_q <= 1'b0;
                quant_mask_q   <= '0;
            end else if (quant_in_range_w) begin
                if (quant_flush_w) begin
                    spm_wr_en_o   <= 1'b1;
                    spm_wr_addr_o <= quant_addr_w;
                    spm_wr_data_o <= quant_pack_next_w;
                    spm_wr_mask_o <= quant_mask_next_w;
                    quant_mask_q  <= '0;
                    done_pending_q <= quant_last_elem_w;
                end else begin
                    quant_pack_q <= quant_pack_next_w;
                    quant_mask_q <= quant_mask_next_w;
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (SPM_DATA_WIDTH != 512)
            $fatal(1, "kv_cache_writer expects 512-bit SPM beats");
        if (CACHE_ROW_BYTES != 64)
            $fatal(1, "kv_cache_writer expects CACHE_ROW_BYTES == 64");
        if (CACHE_HEAD_DIM > CACHE_ROW_BYTES)
            $fatal(1, "kv_cache_writer CACHE_HEAD_DIM must fit in CACHE_ROW_BYTES");
        if (CACHE_HEAD_DIM > GEMV_TILE_ELEMS)
            $fatal(1, "kv_cache_writer CACHE_HEAD_DIM must fit in GEMV_TILE_ELEMS");
        if ((CACHE_ROW_BYTES % QUANT_ELEMS_PER_BEAT) != 0)
            $fatal(1, "kv_cache_writer CACHE_ROW_BYTES must be a multiple of beat bytes");
        if (KV_HEAD_STRIDE_BYTES < K_CACHE_TILE_BYTES)
            $fatal(1, "kv_cache_writer KV_HEAD_STRIDE_BYTES must fit at least one cache tile");
        if (CACHE_WINDOW_TOKENS <= 0)
            $fatal(1, "kv_cache_writer expects a non-empty cache window");
        if (K_CACHE_TILES_PER_HEAD != 31)
            $fatal(1, "kv_cache_writer modulo fold assumes 31 cache row tiles per head");
        if (ROW_TILE_ELEMS != 32 || CACHE_HEAD_DIM != 64 || BYTE_PER_BEAT != 64 ||
            KV_HEAD_STRIDE_BYTES != 65536 || K_CACHE_TILE_BYTES != 2112)
            $fatal(1, "kv_cache_writer shift layout assumes the decode KV cache geometry");
        if ((V_SCALE_REGION_BYTES + (K_CACHE_TILES_PER_HEAD * V_QUANT_TILE_BYTES)) >
            KV_HEAD_STRIDE_BYTES)
            $fatal(1, "kv_cache_writer V separated layout exceeds head stride");
    end

    property p_no_start_while_busy;
        @(posedge clk) disable iff (!rst_n) busy_q |-> !req_en_i;
    endproperty
    assert property (p_no_start_while_busy);

    property p_output_write_has_mask;
        @(posedge clk) disable iff (!rst_n) spm_wr_en_o |-> (spm_wr_mask_o != '0);
    endproperty
    assert property (p_output_write_has_mask);
`endif

endmodule

`endif
