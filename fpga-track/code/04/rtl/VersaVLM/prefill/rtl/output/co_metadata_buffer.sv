`default_nettype none

import npu_spm_pkg::*;

// C/O metadata buffer.
//
// Purpose:
//   Store Q8.24 scale and INT32 bias metadata written by DMA_OC mvin. Read side
//   provides 256-bit words for bias/co_accumulate and scale/co_output_convert.
//
// Clock/reset:
//   Synchronous active-high reset. The BRAM-like metadata array is not reset.
//
// Valid/ready:
//   DMA writes are 128-bit beats with byte strobes. 256-bit reads return after
//   one cycle. Metadata mvin is rejected while C/O is active.
module co_metadata_buffer #(
    parameter int WORD128_DEPTH = 1024,
    localparam int ADDR_W = (WORD128_DEPTH <= 1) ? 1 : $clog2(WORD128_DEPTH)
) (
    input  wire logic                  clk_i,
    input  wire logic                  rst_i,

    input  wire logic                  co_active_i,
    input  wire logic                  clear_error_i,

    input  wire logic                  mvin_valid_i,
    output logic                  mvin_ready_o,
    input  wire logic [ADDR_W-1:0]     mvin_word128_addr_i,
    input  wire logic [127:0]          mvin_data_i,
    input  wire logic [15:0]           mvin_keep_i,

    input  wire logic                  read256_valid_i,
    input  wire logic [ADDR_W-1:0]     read256_base_word128_i,
    input  wire logic [ADDR_W-1:0]     read256_word_idx_i,
    output logic                  read256_rsp_valid_o,
    output logic [255:0]          read256_rsp_data_o,

    output logic                  conflict_sticky_o,
    output npu_error_e            last_error_o
);
    logic [127:0] mem [0:WORD128_DEPTH-1];
    assign mvin_ready_o = !co_active_i;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            read256_rsp_valid_o <= 1'b0;
            read256_rsp_data_o  <= '0;
            conflict_sticky_o   <= 1'b0;
            last_error_o        <= NPU_ERR_NONE;
        end else begin
            read256_rsp_valid_o <= read256_valid_i;
            if (read256_valid_i) begin
                read256_rsp_data_o <= {
                    mem[read256_base_word128_i + (read256_word_idx_i << 1) + ADDR_W'(1)],
                    mem[read256_base_word128_i + (read256_word_idx_i << 1)]
                };
            end

            if (clear_error_i) begin
                conflict_sticky_o <= 1'b0;
                last_error_o      <= NPU_ERR_NONE;
            end

            if (mvin_valid_i && co_active_i) begin
                conflict_sticky_o <= 1'b1;
                last_error_o      <= NPU_ERR_BANK_CONFLICT;
            end else if (mvin_valid_i) begin
                for (int byte_idx = 0; byte_idx < 16; byte_idx++) begin
                    if (mvin_keep_i[byte_idx]) begin
                        mem[mvin_word128_addr_i][byte_idx*8 +: 8] <=
                            mvin_data_i[byte_idx*8 +: 8];
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
