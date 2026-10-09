`default_nettype none

// Purpose:
//   Phase-1 weight loader. It accepts one 128-bit DMA_W stream, packs every two
//   full beats into one 256-bit W-bank word, and writes the selected inactive
//   W bank using the frozen INT8 resident layout.
// Clock/reset:
//   All state uses clk_i. rst_i is synchronous, active high.
// Interface latency:
//   In the common case where the selected bank write port is ready, the second
//   128-bit beat of a 256-bit word writes through to the bank in the same
//   cycle. A generated bank write is stored only when the bank applies
//   backpressure.
// Valid/ready:
//   cfg_start_i is a command-level pulse accepted when cfg_ready_o is high.
//   dma_ready_o is asserted only during an active command with no pending write.
// Supported modes:
//   INT8 signed-payload W load, K-major inside each 32-column group.
// Unsupported/error behavior:
//   Non-full keep, odd beat count, last mismatch, zero descriptor dimensions,
//   and preload-to-active-bank are reported as sticky errors. No weight reorder
//   or zeropoint correction is implemented.
module w_loader_128_to_256 #(
    parameter int ADDR_W = 14,
    parameter int LEN_W  = 16
) (
    input  wire logic                    clk_i,
    input  wire logic                    rst_i,

    input  wire logic                    cfg_start_i,
    output logic                    cfg_ready_o,
    input  wire logic [ADDR_W-1:0]       cfg_base_word_addr_i,
    input  wire logic [LEN_W-1:0]        cfg_k_loaded_i,
    input  wire logic [LEN_W-1:0]        cfg_n_loaded_i,
    input  wire logic                    cfg_selected_w_bank_i,
    input  wire logic                    active_w_bank_i,
    input  wire logic                    clear_error_i,

    input  wire logic                    dma_valid_i,
    output logic                    dma_ready_o,
    input  wire logic [127:0]            dma_data_i,
    input  wire logic [15:0]             dma_keep_i,
    input  wire logic                    dma_last_i,

    output logic                    w0_wr_valid_o,
    input  wire logic                    w0_wr_ready_i,
    output logic [ADDR_W-1:0]       w0_wr_word_addr_o,
    output logic [255:0]            w0_wr_data_o,
    output logic [31:0]             w0_wr_byte_en_o,

    output logic                    w1_wr_valid_o,
    input  wire logic                    w1_wr_ready_i,
    output logic [ADDR_W-1:0]       w1_wr_word_addr_o,
    output logic [255:0]            w1_wr_data_o,
    output logic [31:0]             w1_wr_byte_en_o,

    output logic                    busy_o,
    output logic                    done_o,
    output logic                    error_sticky_o,
    output logic [7:0]              error_code_o
);

    localparam logic [7:0] ERROR_NONE        = 8'h00;
    localparam logic [7:0] ERROR_BUSY        = 8'h01;
    localparam logic [7:0] ERROR_DESCRIPTOR  = 8'h02;
    localparam logic [7:0] ERROR_ACTIVE_BANK = 8'h03;
    localparam logic [7:0] ERROR_KEEP        = 8'h04;
    localparam logic [7:0] ERROR_ODD_BEAT    = 8'h05;
    localparam logic [7:0] ERROR_LENGTH      = 8'h06;

    logic [LEN_W-1:0]  k_loaded_q;
    logic [LEN_W-1:0]  group_count_q;
    logic              selected_w_bank_q;

    logic [LEN_W-1:0]  k_idx_q;
    logic [LEN_W-1:0]  group_idx_q;
    logic [ADDR_W-1:0] current_group_base_word_addr_q;

    logic              half_full_q;
    logic [127:0]      half_data_q;

    logic              wr_pending_q;
    logic              wr_pending_last_q;
    logic [ADDR_W-1:0] wr_addr_q;
    logic [255:0]      wr_data_q;

    logic selected_bank_ready;
    logic beat_fire;
    logic word_is_last;
    logic beat_error;
    logic generated_wr_valid;
    logic [ADDR_W-1:0] generated_wr_addr;
    logic [255:0] generated_wr_data;

    assign cfg_ready_o         = !busy_o;
    assign selected_bank_ready = selected_w_bank_q ? w1_wr_ready_i : w0_wr_ready_i;
    assign dma_ready_o         = busy_o && !wr_pending_q && !error_sticky_o;
    assign beat_fire           = dma_valid_i && dma_ready_o;
    assign word_is_last        = (group_idx_q == (group_count_q - LEN_W'(1))) &&
                                 (k_idx_q == (k_loaded_q - LEN_W'(1)));
    assign beat_error          = (dma_keep_i != 16'hffff) ||
                                 (!half_full_q && dma_last_i) ||
                                 (half_full_q && (dma_last_i != word_is_last));
    assign generated_wr_valid  = beat_fire && half_full_q && !beat_error;
    assign generated_wr_addr   = current_group_base_word_addr_q + ADDR_W'(k_idx_q);
    assign generated_wr_data   = {dma_data_i, half_data_q};

    assign w0_wr_valid_o       = (wr_pending_q || generated_wr_valid) && !selected_w_bank_q;
    assign w1_wr_valid_o       = (wr_pending_q || generated_wr_valid) && selected_w_bank_q;
    assign w0_wr_word_addr_o   = wr_pending_q ? wr_addr_q : generated_wr_addr;
    assign w1_wr_word_addr_o   = wr_pending_q ? wr_addr_q : generated_wr_addr;
    assign w0_wr_data_o        = wr_pending_q ? wr_data_q : generated_wr_data;
    assign w1_wr_data_o        = wr_pending_q ? wr_data_q : generated_wr_data;
    assign w0_wr_byte_en_o     = 32'hffff_ffff;
    assign w1_wr_byte_en_o     = 32'hffff_ffff;

    function automatic logic [LEN_W-1:0] ceil_div_32(input logic [LEN_W-1:0] value);
        ceil_div_32 = (value + LEN_W'(31)) >> 5;
    endfunction

    always_ff @(posedge clk_i) begin : loader_ff
        if (rst_i) begin
            k_loaded_q                    <= '0;
            group_count_q                 <= '0;
            selected_w_bank_q             <= 1'b0;
            k_idx_q                       <= '0;
            group_idx_q                   <= '0;
            current_group_base_word_addr_q <= '0;
            half_full_q                   <= 1'b0;
            half_data_q                   <= '0;
            wr_pending_q                  <= 1'b0;
            wr_pending_last_q             <= 1'b0;
            wr_addr_q                     <= '0;
            wr_data_q                     <= '0;
            busy_o                        <= 1'b0;
            done_o                        <= 1'b0;
            error_sticky_o                <= 1'b0;
            error_code_o                  <= ERROR_NONE;
        end
        else begin
            done_o <= 1'b0;

            if (clear_error_i) begin
                error_sticky_o <= 1'b0;
                error_code_o   <= ERROR_NONE;
            end

            if (wr_pending_q && selected_bank_ready) begin
                wr_pending_q <= 1'b0;
                if (wr_pending_last_q) begin
                    busy_o            <= 1'b0;
                    done_o            <= 1'b1;
                    wr_pending_last_q <= 1'b0;
                end
            end

            if (cfg_start_i && cfg_ready_o) begin
                if ((cfg_k_loaded_i == '0) || (cfg_n_loaded_i == '0)) begin
                    error_sticky_o <= 1'b1;
                    error_code_o   <= ERROR_DESCRIPTOR;
                end
                else if (cfg_selected_w_bank_i == active_w_bank_i) begin
                    error_sticky_o <= 1'b1;
                    error_code_o   <= ERROR_ACTIVE_BANK;
                end
                else begin
                    k_loaded_q                    <= cfg_k_loaded_i;
                    group_count_q                 <= ceil_div_32(cfg_n_loaded_i);
                    selected_w_bank_q             <= cfg_selected_w_bank_i;
                    k_idx_q                       <= '0;
                    group_idx_q                   <= '0;
                    current_group_base_word_addr_q <= cfg_base_word_addr_i;
                    half_full_q                   <= 1'b0;
                    half_data_q                   <= '0;
                    wr_pending_last_q             <= 1'b0;
                    busy_o                        <= 1'b1;
                end
            end
            else if (cfg_start_i && !cfg_ready_o) begin
                error_sticky_o <= 1'b1;
                error_code_o   <= ERROR_BUSY;
            end

            if (beat_fire) begin
                if (dma_keep_i != 16'hffff) begin
                    error_sticky_o <= 1'b1;
                    error_code_o   <= ERROR_KEEP;
                    busy_o         <= 1'b0;
                    half_full_q    <= 1'b0;
                end
                else if (!half_full_q && dma_last_i) begin
                    error_sticky_o <= 1'b1;
                    error_code_o   <= ERROR_ODD_BEAT;
                    busy_o         <= 1'b0;
                    half_full_q    <= 1'b0;
                end
                else if (!half_full_q) begin
                    half_full_q <= 1'b1;
                    half_data_q <= dma_data_i;
                end
                else begin
                    if (dma_last_i != word_is_last) begin
                        error_sticky_o <= 1'b1;
                        error_code_o   <= ERROR_LENGTH;
                        busy_o         <= 1'b0;
                        half_full_q    <= 1'b0;
                    end
                    else begin
                        if (!selected_bank_ready) begin
                            wr_pending_q      <= 1'b1;
                            wr_pending_last_q <= word_is_last;
                            wr_addr_q         <= generated_wr_addr;
                            wr_data_q         <= generated_wr_data;
                        end
                        else if (word_is_last) begin
                            busy_o            <= 1'b0;
                            done_o            <= 1'b1;
                            wr_pending_q      <= 1'b0;
                            wr_pending_last_q <= 1'b0;
                        end
                        half_full_q       <= 1'b0;
                        if (k_idx_q == (k_loaded_q - LEN_W'(1))) begin
                            k_idx_q                       <= '0;
                            group_idx_q                   <= group_idx_q + LEN_W'(1);
                            current_group_base_word_addr_q <=
                                current_group_base_word_addr_q + ADDR_W'(k_loaded_q);
                        end
                        else begin
                            k_idx_q <= k_idx_q + LEN_W'(1);
                        end
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
