`default_nettype none

// Purpose:
//   Phase-1 activation loader. It accepts one 128-bit DMA_A stream, packs up
//   to two beats into one 256-bit A-bank word, and optionally converts uint8
//   asymmetric activation bytes into signed int8 payload bytes.
// Clock/reset:
//   All state uses clk_i. rst_i is synchronous, active high.
// Interface latency:
//   In the common case where the bank write port is ready, the second 128-bit
//   beat of a 256-bit word writes through to the bank in the same cycle. A
//   generated bank write is stored only when the bank applies backpressure.
// Valid/ready:
//   cfg_start_i is a command-level pulse accepted when cfg_ready_o is high.
//   dma_ready_o applies backpressure while a bank write is pending.
// Supported modes:
//   Row-major 2D A load with 256-bit word addresses, full 256-bit words, and
//   optional uint8 minus 128 conversion.
// Unsupported/error behavior:
//   Non-full keep masks and non-32B row descriptors are errors.
//   There is no deep FIFO in this Phase-1 standalone loader.
module a_loader_128_to_256 #(
    parameter int ADDR_W = 14,
    parameter int LEN_W  = 16
) (
    input  wire logic                    clk_i,
    input  wire logic                    rst_i,

    input  wire logic                    cfg_start_i,
    output logic                    cfg_ready_o,
    input  wire logic [ADDR_W-1:0]       cfg_base_word_addr_i,
    input  wire logic [LEN_W-1:0]        cfg_row_count_i,
    input  wire logic [LEN_W-1:0]        cfg_row_bytes_i,
    input  wire logic [LEN_W-1:0]        cfg_row_stride_words_i,
    input  wire logic                    cfg_a_u8_minus_128_en_i,
    input  wire logic                    clear_error_i,

    input  wire logic                    dma_valid_i,
    output logic                    dma_ready_o,
    input  wire logic [127:0]            dma_data_i,
    input  wire logic [15:0]             dma_keep_i,
    input  wire logic                    dma_last_i,

    output logic                    a_wr_valid_o,
    input  wire logic                    a_wr_ready_i,
    output logic [ADDR_W-1:0]       a_wr_word_addr_o,
    output logic [255:0]            a_wr_data_o,
    output logic [31:0]             a_wr_byte_en_o,

    output logic                    busy_o,
    output logic                    done_o,
    output logic                    error_sticky_o,
    output logic [7:0]              error_code_o
);

    localparam logic [7:0] ERROR_NONE       = 8'h00;
    localparam logic [7:0] ERROR_BUSY       = 8'h01;
    localparam logic [7:0] ERROR_DESCRIPTOR = 8'h02;
    localparam logic [7:0] ERROR_KEEP       = 8'h03;
    localparam logic [7:0] ERROR_LENGTH     = 8'h04;

    logic [LEN_W-1:0]  row_count_q;
    logic [LEN_W-1:0]  row_bytes_q;
    logic [LEN_W-1:0]  row_stride_words_q;
    logic              a_u8_minus_128_en_q;

    logic [ADDR_W-1:0] current_row_base_word_addr_q;
    logic [LEN_W-1:0]  row_idx_q;
    logic [LEN_W-1:0]  row_byte_idx_q;
    logic [LEN_W-1:0]  row_word_idx_q;

    logic              half_full_q;
    logic [127:0]      half_data_q;
    logic [15:0]       half_keep_q;

    logic              wr_pending_q;
    logic              wr_pending_last_q;
    logic [ADDR_W-1:0] wr_addr_q;
    logic [255:0]      wr_data_q;
    logic [31:0]       wr_byte_en_q;

    logic [LEN_W:0]  row_remaining_bytes;
    logic [5:0]      expected_bytes;
    logic [15:0]     expected_keep;
    logic            beat_fire;
    logic            word_complete;
    logic            row_complete;
    logic            packet_complete;
    logic [127:0]    beat_data;
    logic            beat_error;
    logic            generated_wr_valid;
    logic [ADDR_W-1:0] generated_wr_addr;
    logic [255:0]    generated_wr_data;
    logic [31:0]     generated_wr_byte_en;

    assign cfg_ready_o      = !busy_o;
    assign dma_ready_o      = busy_o && !wr_pending_q && !error_sticky_o;
    assign beat_fire        = dma_valid_i && dma_ready_o;

    assign row_remaining_bytes = {1'b0, row_bytes_q} - {1'b0, row_byte_idx_q};
    assign expected_bytes      = 6'd16;
    assign expected_keep       = 16'hffff;
    assign beat_data           = transform_beat(dma_data_i, dma_keep_i, a_u8_minus_128_en_q);
    assign word_complete       = half_full_q || (row_remaining_bytes <= 16);
    assign row_complete        = row_remaining_bytes <= 16;
    assign packet_complete     = row_complete && (row_idx_q == (row_count_q - LEN_W'(1)));
    assign beat_error          = (dma_keep_i != expected_keep) || (dma_last_i != packet_complete);
    assign generated_wr_valid  = beat_fire && word_complete && !beat_error;
    assign generated_wr_addr   = current_row_base_word_addr_q + ADDR_W'(row_word_idx_q);
    assign generated_wr_data   = half_full_q ? {beat_data, half_data_q} : {128'b0, beat_data};
    assign generated_wr_byte_en = 32'hffff_ffff;

    assign a_wr_valid_o     = wr_pending_q || generated_wr_valid;
    assign a_wr_word_addr_o = wr_pending_q ? wr_addr_q : generated_wr_addr;
    assign a_wr_data_o      = wr_pending_q ? wr_data_q : generated_wr_data;
    assign a_wr_byte_en_o   = wr_pending_q ? wr_byte_en_q : generated_wr_byte_en;

    function automatic logic [15:0] keep_for_bytes(input logic [5:0] byte_count);
        if (byte_count == 6'd0) begin
            keep_for_bytes = 16'h0000;
        end
        else if (byte_count >= 6'd16) begin
            keep_for_bytes = 16'hffff;
        end
        else begin
            keep_for_bytes = 16'hffff >> (6'd16 - byte_count);
        end
    endfunction

    function automatic logic [127:0] transform_beat(
        input logic [127:0] data,
        input logic [15:0]  keep,
        input logic         minus_128_en
    );
        logic signed [8:0] signed_byte;
        begin
            transform_beat = '0;
            for (int byte_idx = 0; byte_idx < 16; byte_idx++) begin
                if (keep[byte_idx]) begin
                    if (minus_128_en) begin
                        signed_byte = $signed({1'b0, data[byte_idx*8 +: 8]}) - 9'sd128;
                        transform_beat[byte_idx*8 +: 8] = signed_byte[7:0];
                    end
                    else begin
                        transform_beat[byte_idx*8 +: 8] = data[byte_idx*8 +: 8];
                    end
                end
            end
        end
    endfunction

    always_ff @(posedge clk_i) begin : loader_ff
        if (rst_i) begin
            row_count_q           <= '0;
            row_bytes_q           <= '0;
            row_stride_words_q    <= '0;
            a_u8_minus_128_en_q   <= 1'b0;
            current_row_base_word_addr_q <= '0;
            row_idx_q             <= '0;
            row_byte_idx_q        <= '0;
            row_word_idx_q        <= '0;
            half_full_q           <= 1'b0;
            half_data_q           <= '0;
            half_keep_q           <= '0;
            wr_pending_q          <= 1'b0;
            wr_pending_last_q     <= 1'b0;
            wr_addr_q             <= '0;
            wr_data_q             <= '0;
            wr_byte_en_q          <= '0;
            busy_o                <= 1'b0;
            done_o                <= 1'b0;
            error_sticky_o        <= 1'b0;
            error_code_o          <= ERROR_NONE;
        end
        else begin
            done_o <= 1'b0;

            if (clear_error_i) begin
                error_sticky_o <= 1'b0;
                error_code_o   <= ERROR_NONE;
            end

            if (wr_pending_q && a_wr_ready_i) begin
                wr_pending_q <= 1'b0;
                if (wr_pending_last_q) begin
                    busy_o            <= 1'b0;
                    done_o            <= 1'b1;
                    wr_pending_last_q <= 1'b0;
                end
            end

            if (cfg_start_i && cfg_ready_o) begin
                if ((cfg_row_count_i == '0) || (cfg_row_bytes_i == '0) ||
                    (cfg_row_bytes_i[4:0] != 5'd0) ||
                    (cfg_row_stride_words_i < ((cfg_row_bytes_i + LEN_W'(31)) >> 5))) begin
                    error_sticky_o <= 1'b1;
                    error_code_o   <= ERROR_DESCRIPTOR;
                end
                else begin
                    row_count_q                  <= cfg_row_count_i;
                    row_bytes_q                  <= cfg_row_bytes_i;
                    row_stride_words_q           <= cfg_row_stride_words_i;
                    a_u8_minus_128_en_q          <= cfg_a_u8_minus_128_en_i;
                    current_row_base_word_addr_q <= cfg_base_word_addr_i;
                    row_idx_q                    <= '0;
                    row_byte_idx_q               <= '0;
                    row_word_idx_q               <= '0;
                    half_full_q                  <= 1'b0;
                    half_data_q                  <= '0;
                    half_keep_q                  <= '0;
                    wr_pending_last_q            <= 1'b0;
                    busy_o                       <= 1'b1;
                end
            end
            else if (cfg_start_i && !cfg_ready_o) begin
                error_sticky_o <= 1'b1;
                error_code_o   <= ERROR_BUSY;
            end

            else if (beat_fire) begin
                if (dma_keep_i != expected_keep) begin
                    error_sticky_o <= 1'b1;
                    error_code_o   <= ERROR_KEEP;
                    busy_o         <= 1'b0;
                    half_full_q    <= 1'b0;
                end
                else if (dma_last_i != packet_complete) begin
                    error_sticky_o <= 1'b1;
                    error_code_o   <= ERROR_LENGTH;
                    busy_o         <= 1'b0;
                    half_full_q    <= 1'b0;
                end
                else if (!word_complete) begin
                    half_full_q    <= 1'b1;
                    half_data_q    <= beat_data;
                    half_keep_q    <= dma_keep_i;
                    row_byte_idx_q <= row_byte_idx_q + LEN_W'(16);
                end
                else begin
                    if (!a_wr_ready_i) begin
                        wr_pending_q      <= 1'b1;
                        wr_pending_last_q <= packet_complete;
                        wr_addr_q         <= generated_wr_addr;
                        wr_data_q         <= generated_wr_data;
                        wr_byte_en_q      <= generated_wr_byte_en;
                    end
                    else if (packet_complete) begin
                        busy_o            <= 1'b0;
                        done_o            <= 1'b1;
                        wr_pending_q      <= 1'b0;
                        wr_pending_last_q <= 1'b0;
                    end

                    half_full_q <= 1'b0;
                    if (row_complete) begin
                        row_idx_q                    <= row_idx_q + LEN_W'(1);
                        row_byte_idx_q               <= '0;
                        row_word_idx_q               <= '0;
                        current_row_base_word_addr_q <=
                            current_row_base_word_addr_q + ADDR_W'(row_stride_words_q);
                    end
                    else begin
                        row_byte_idx_q <= row_byte_idx_q + LEN_W'(16);
                        row_word_idx_q <= row_word_idx_q + LEN_W'(1);
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
