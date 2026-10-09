`ifndef SCRATCHPAD_4BANK_V
`define SCRATCHPAD_4BANK_V

module scratchpad_4bank
import npu_config_pkg::*;
#(
    parameter int SPM_FPGA_SRAM   = npu_config_pkg::SPM_FPGA_SRAM,
    parameter int BANKS           = npu_config_pkg::SPM_BANK_NUM,
    parameter int BANK_DATA_WIDTH = npu_config_pkg::SPM_BANK_DATA_WIDTH,
    parameter int DEPTH           = npu_config_pkg::SPM_SIZE / (npu_config_pkg::SPM_BANK_NUM * (npu_config_pkg::SPM_BANK_DATA_WIDTH / 8)),
    parameter string RAM_STYLE    = "ultra",
    parameter int READ_PIPELINE   = 0,
    parameter int BYTE_WRITE_ENABLE = 1
) (
    input  logic                                            clk,
    input  logic                                            rst_n,

    input  logic                                            wr_en,
    input  logic [$clog2(DEPTH)-1:0]                        wr_addr,
    input  logic [BANKS-1:0][BANK_DATA_WIDTH-1:0]           wr_data,
    input  logic [BANKS-1:0][BANK_DATA_WIDTH/8-1:0]         wr_mask,

    input  logic                                            rd_en,
    input  logic [$clog2(DEPTH)-1:0]                        rd_addr,
    output logic [BANKS-1:0][BANK_DATA_WIDTH-1:0]           rd_data
);

// `wr_addr`/`rd_addr` index one full 4-bank group. In the linear scratchpad
// address view, the ordering inside one group is:
// bank0[group] -> bank1[group] -> bank2[group] -> bank3[group].
logic [BANKS-1:0][BANK_DATA_WIDTH-1:0] fpga_rd_data;

generate
    if (SPM_FPGA_SRAM == 1) begin : gen_fpga_uram
        genvar bank;
        for (bank = 0; bank < BANKS; bank++) begin : gen_bank
            wire [BANK_DATA_WIDTH-1:0] unused_dout_a;
            sram_fpga #(
                .DATA_WIDTH    (BANK_DATA_WIDTH),
                .ADDR_WIDTH    ($clog2(DEPTH)),
                .RAM_STYLE     (RAM_STYLE),
                .READ_PIPELINE (READ_PIPELINE),
                .PORT_B_WRITE  (0),
                .BYTE_WRITE_ENABLE (BYTE_WRITE_ENABLE)
            ) u_sram_fpga (
                .clk    (clk),
                .we_a   (wr_en),
                .addr_a (wr_addr),
                .din_a  (wr_data[bank]),
                .wr_mask_a (wr_mask[bank]),
                .dout_a (unused_dout_a),
                .we_b   (1'b0),
                .addr_b (rd_addr),
                .din_b  ('0),
                .wr_mask_b ({BANK_DATA_WIDTH/8{1'b1}}),
                .dout_b (fpga_rd_data[bank])
            );
        end

        assign rd_data = fpga_rd_data;
    end else begin : gen_sim_bank_mem
        logic [BANKS-1:0][DEPTH-1:0][BANK_DATA_WIDTH-1:0] bank_mem;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                rd_data <= '0;
            end else begin
                if (wr_en) begin
                    for (int bank = 0; bank < BANKS; bank++) begin
                        for (int byte_idx = 0; byte_idx < BANK_DATA_WIDTH / 8; byte_idx++) begin
                            if (wr_mask[bank][byte_idx]) begin
                                bank_mem[bank][wr_addr][byte_idx * 8 +: 8] <= wr_data[bank][byte_idx * 8 +: 8];
                            end
                        end
                    end
                end

                if (rd_en) begin
                    for (int bank = 0; bank < BANKS; bank++) begin
                        rd_data[bank] <= bank_mem[bank][rd_addr];
                    end
                end
            end
        end
    end
endgenerate

endmodule

`endif
