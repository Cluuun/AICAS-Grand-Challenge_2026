`ifndef STREAM_LINE_WRITER_SV
`define STREAM_LINE_WRITER_SV

module stream_line_writer #(
    parameter int SPM_SIZE       = 1 << 19,
    parameter int SPM_DATA_WIDTH = 512,
    parameter int USER_WIDTH     = 16,
    parameter bit ENABLE_GROUP_STRIDE = 1'b1,

    localparam int SPM_ADDR_WIDTH = $clog2(SPM_SIZE)
) (
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic                              cfg_dst_act_i,
    input  logic                              cfg_word32_i,
    input  logic [SPM_ADDR_WIDTH-1:0]         cfg_base_addr_i,
    input  logic [USER_WIDTH-1:0]             cfg_elem_count_i,
    input  logic [USER_WIDTH-1:0]             cfg_group_elem_count_i,
    input  logic [USER_WIDTH-1:0]             cfg_group_stride_elems_i,
    input  logic                              cfg_group_stride_en_i,
    input  logic                              req_en_i,
    output logic                              busy_o,
    output logic                              done_o,

    input  logic                              valid_i,
    input  logic [15:0]                       data_i,
    input  logic [31:0]                       data32_i,
    input  logic [USER_WIDTH-1:0]             user_i,

    output logic                              spm_wr_en_o,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_wr_addr_o,
    output logic [SPM_DATA_WIDTH-1:0]         spm_wr_data_o,
    output logic [SPM_DATA_WIDTH/8-1:0]       spm_wr_mask_o,

    output logic                              act_wr_en_o,
    output logic [SPM_ADDR_WIDTH-1:0]         act_wr_addr_o,
    output logic [SPM_DATA_WIDTH-1:0]         act_wr_data_o,
    output logic [SPM_DATA_WIDTH/8-1:0]       act_wr_mask_o
);

    localparam int FP16_PER_BEAT       = SPM_DATA_WIDTH / 16;
    localparam int WORD32_PER_BEAT     = SPM_DATA_WIDTH / 32;
    localparam int BYTE_PER_BEAT       = SPM_DATA_WIDTH / 8;
    localparam int OUT_BEAT_ADDR_SHIFT = $clog2(BYTE_PER_BEAT);
    localparam int FP16_BEAT_IDX_WIDTH = $clog2(FP16_PER_BEAT);
    localparam int WORD32_BEAT_IDX_WIDTH = $clog2(WORD32_PER_BEAT);
    localparam int PHYS_ELEM_WIDTH     = USER_WIDTH + 1;

    logic                              dst_act_q;
    logic                              word32_q;
    logic [SPM_ADDR_WIDTH-1:0]         base_addr_q;
    logic [SPM_ADDR_WIDTH-1:0]         base_line_addr_q;
    logic [FP16_BEAT_IDX_WIDTH-1:0]    base_lane_q;
    logic [WORD32_BEAT_IDX_WIDTH-1:0]  base_lane32_q;
    logic [USER_WIDTH-1:0]             elem_count_q;
    logic                              group_stride_en_q;
    logic [USER_WIDTH-1:0]             group_bound1_q;
    logic [USER_WIDTH-1:0]             group_bound2_q;
    logic [USER_WIDTH-1:0]             group_bound3_q;
    logic [USER_WIDTH-1:0]             group_base1_q;
    logic [USER_WIDTH-1:0]             group_base2_q;
    logic [USER_WIDTH-1:0]             group_base3_q;
    logic [USER_WIDTH-1:0]             group_last_lane_q;
    logic [SPM_DATA_WIDTH-1:0]         pack_q;
    logic [SPM_DATA_WIDTH/8-1:0]       mask_q;
    logic                              busy_q;
    logic                              done_pending_q;

    logic [SPM_DATA_WIDTH-1:0]         pack_next_w;
    logic [SPM_DATA_WIDTH/8-1:0]       mask_next_w;
    logic                              in_range_w;
    logic                              flush_w;
    logic                              last_elem_w;
    logic [USER_WIDTH-1:0]             group_lane_w;
    logic [USER_WIDTH-1:0]             group_base_elem_w;
    logic [USER_WIDTH-1:0]             addr_elem_w;
    logic [PHYS_ELEM_WIDTH-1:0]        phys_elem_w;
    logic [PHYS_ELEM_WIDTH-1:0]        phys_elem32_w;
    logic                              group_stride_active_w;
    logic [FP16_BEAT_IDX_WIDTH-1:0]    phys_lane_w;
    logic [WORD32_BEAT_IDX_WIDTH-1:0]  phys_lane32_w;
    logic [SPM_ADDR_WIDTH-1:0]         flush_addr_w;

    assign busy_o = busy_q;

    assign in_range_w = valid_i && busy_q && (user_i < elem_count_q);
    assign group_stride_active_w = group_stride_en_q;
    assign addr_elem_w = group_stride_active_w ? (group_base_elem_w + group_lane_w) : user_i;
    assign phys_elem_w = PHYS_ELEM_WIDTH'(addr_elem_w) +
                         PHYS_ELEM_WIDTH'(base_lane_q);
    assign phys_elem32_w = PHYS_ELEM_WIDTH'(addr_elem_w) +
                           PHYS_ELEM_WIDTH'(base_lane32_q);
    assign phys_lane_w = phys_elem_w[FP16_BEAT_IDX_WIDTH-1:0];
    assign phys_lane32_w = phys_elem32_w[WORD32_BEAT_IDX_WIDTH-1:0];
    assign last_elem_w = in_range_w && (user_i == (elem_count_q - 1'b1));
    assign flush_w = in_range_w &&
                     ((word32_q ? (&phys_lane32_w) : (&phys_lane_w)) ||
                      (group_stride_active_w &&
                       (group_lane_w == group_last_lane_q)) ||
                      last_elem_w);
    assign flush_addr_w = base_line_addr_q +
                          (word32_q ?
                           (SPM_ADDR_WIDTH'(phys_elem32_w >> WORD32_BEAT_IDX_WIDTH)
                            << OUT_BEAT_ADDR_SHIFT) :
                           (SPM_ADDR_WIDTH'(phys_elem_w >> FP16_BEAT_IDX_WIDTH)
                            << OUT_BEAT_ADDR_SHIFT));

    always_comb begin
        group_lane_w = user_i;
        group_base_elem_w = '0;
        if (group_stride_active_w && (group_bound1_q != '0)) begin
            if (user_i >= group_bound3_q) begin
                group_lane_w = user_i - group_bound3_q;
                group_base_elem_w = group_base3_q;
            end else if (user_i >= group_bound2_q) begin
                group_lane_w = user_i - group_bound2_q;
                group_base_elem_w = group_base2_q;
            end else if (user_i >= group_bound1_q) begin
                group_lane_w = user_i - group_bound1_q;
                group_base_elem_w = group_base1_q;
            end
        end
    end

    always_comb begin
        pack_next_w = pack_q;
        mask_next_w = mask_q;
        if (in_range_w) begin
            if (word32_q) begin
                pack_next_w[phys_lane32_w * 32 +: 32] = data32_i;
                mask_next_w[phys_lane32_w * 4 +: 4] = 4'b1111;
            end else begin
                pack_next_w[phys_lane_w * 16 +: 16] = data_i;
                mask_next_w[phys_lane_w * 2 +: 2] = 2'b11;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dst_act_q      <= 1'b0;
            word32_q       <= 1'b0;
            base_addr_q    <= '0;
            base_line_addr_q <= '0;
            base_lane_q    <= '0;
            base_lane32_q  <= '0;
            elem_count_q   <= '0;
            group_stride_en_q <= 1'b0;
            group_bound1_q <= '0;
            group_bound2_q <= '0;
            group_bound3_q <= '0;
            group_base1_q <= '0;
            group_base2_q <= '0;
            group_base3_q <= '0;
            group_last_lane_q <= '0;
            pack_q         <= '0;
            mask_q         <= '0;
            busy_q         <= 1'b0;
            done_pending_q <= 1'b0;
            done_o         <= 1'b0;
            spm_wr_en_o    <= 1'b0;
            spm_wr_mask_o  <= '0;
            act_wr_en_o    <= 1'b0;
            act_wr_mask_o  <= '0;
        end else begin
            done_o        <= 1'b0;
            spm_wr_en_o   <= 1'b0;
            spm_wr_mask_o <= '0;
            act_wr_en_o   <= 1'b0;
            act_wr_mask_o <= '0;

            if (req_en_i) begin
                dst_act_q      <= cfg_dst_act_i;
                word32_q       <= cfg_word32_i;
                base_addr_q    <= cfg_base_addr_i;
                base_line_addr_q <= {cfg_base_addr_i[SPM_ADDR_WIDTH-1:OUT_BEAT_ADDR_SHIFT],
                                     {OUT_BEAT_ADDR_SHIFT{1'b0}}};
                base_lane_q    <= cfg_base_addr_i[OUT_BEAT_ADDR_SHIFT-1:1];
                base_lane32_q  <= cfg_base_addr_i[OUT_BEAT_ADDR_SHIFT-1:2];
                elem_count_q   <= cfg_elem_count_i;
                group_stride_en_q <= ENABLE_GROUP_STRIDE && cfg_group_stride_en_i;
                group_bound1_q <= ENABLE_GROUP_STRIDE ? cfg_group_elem_count_i : '0;
                group_bound2_q <= ENABLE_GROUP_STRIDE ? (cfg_group_elem_count_i << 1) : '0;
                group_bound3_q <= ENABLE_GROUP_STRIDE ?
                                  (cfg_group_elem_count_i + (cfg_group_elem_count_i << 1)) : '0;
                group_base1_q <= ENABLE_GROUP_STRIDE ? cfg_group_stride_elems_i : '0;
                group_base2_q <= ENABLE_GROUP_STRIDE ? (cfg_group_stride_elems_i << 1) : '0;
                group_base3_q <= ENABLE_GROUP_STRIDE ?
                                 (cfg_group_stride_elems_i + (cfg_group_stride_elems_i << 1)) : '0;
                group_last_lane_q <= (ENABLE_GROUP_STRIDE && (cfg_group_elem_count_i != '0)) ?
                                     (cfg_group_elem_count_i - 1'b1) : '0;
                pack_q         <= '0;
                mask_q         <= '0;
                busy_q         <= (cfg_elem_count_i != '0);
                done_pending_q <= (cfg_elem_count_i == '0);
            end else if (done_pending_q) begin
                done_o         <= 1'b1;
                busy_q         <= 1'b0;
                done_pending_q <= 1'b0;
                pack_q         <= '0;
                mask_q         <= '0;
            end else if (in_range_w) begin
                if (flush_w) begin
                    if (dst_act_q) begin
                        act_wr_en_o   <= 1'b1;
                        act_wr_addr_o <= flush_addr_w;
                        act_wr_data_o <= pack_next_w;
                        act_wr_mask_o <= mask_next_w;
                    end else begin
                        spm_wr_en_o   <= 1'b1;
                        spm_wr_addr_o <= flush_addr_w;
                        spm_wr_data_o <= pack_next_w;
                        spm_wr_mask_o <= mask_next_w;
                    end
                    pack_q         <= '0;
                    mask_q         <= '0;
                    done_pending_q <= last_elem_w;
                end else begin
                    pack_q <= pack_next_w;
                    mask_q <= mask_next_w;
                end
            end
        end
    end

`ifndef SYNTHESIS
    property p_no_start_while_busy;
        @(posedge clk) disable iff (!rst_n) busy_q |-> !req_en_i;
    endproperty
    assert property (p_no_start_while_busy);

    property p_base_addr_fp16_aligned;
        @(posedge clk) disable iff (!rst_n) req_en_i |-> (cfg_base_addr_i[0] == 1'b0);
    endproperty
    assert property (p_base_addr_fp16_aligned);

    property p_base_addr_word32_aligned;
        @(posedge clk) disable iff (!rst_n) (req_en_i && cfg_word32_i) |-> (cfg_base_addr_i[1:0] == 2'b00);
    endproperty
    assert property (p_base_addr_word32_aligned);

    property p_output_write_has_mask;
        @(posedge clk) disable iff (!rst_n) spm_wr_en_o |-> (spm_wr_mask_o != '0);
    endproperty
    assert property (p_output_write_has_mask);

    property p_act_write_has_mask;
        @(posedge clk) disable iff (!rst_n) act_wr_en_o |-> (act_wr_mask_o != '0);
    endproperty
    assert property (p_act_write_has_mask);
`endif

endmodule

`endif
