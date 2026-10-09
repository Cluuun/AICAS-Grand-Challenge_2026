module multibank_sram #(
    parameter int DATA_WIDTH = 256,
    parameter int BANK_DEPTH = 128 ,
    parameter int NUM_BANKS  = 8
)(
    input  logic                                         clk_i ,
    input  logic [NUM_BANKS-1:0]                         ena_i ,
    input  logic [NUM_BANKS-1:0]                         wen_i ,
    input  logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] din_i ,
    input  logic [NUM_BANKS-1:0][$clog2(BANK_DEPTH)-1:0] addr_i,
    output logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] dout_o
);

generate
    for (genvar i = 0; i < NUM_BANKS; i++) begin : sram_bank_array
        sram_bank #(
            .DATA_WIDTH(DATA_WIDTH),
            .DEPTH(BANK_DEPTH)
        ) u_sram_bank (
            .clk_i (clk_i    ),
            .ena   (ena_i[i] ),
            .wen   (wen_i[i] ),
            .addr  (addr_i[i]),
            .din   (din_i[i] ),
            .dout  (dout_o[i])
        );
    end
endgenerate

endmodule
