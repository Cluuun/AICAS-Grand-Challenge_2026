module multibank_ocm_system #(
    parameter ADDR_WIDTH = 32 ,
    parameter DATA_WIDTH = 256,
    parameter BANK_WIDTH = 256,
    parameter BANK_DEPTH = 128,
    parameter NUM_BANKS  = 8
)(
    input  logic                           clk_i        ,
    input  logic                           gdma_ena_i   ,
    input  logic                           gdma_wen_i   ,
    input  logic [ADDR_WIDTH        -1:0]  gdma_addr_i  ,
    input  logic [DATA_WIDTH        -1:0]  gdma_din_i   ,
    output logic [DATA_WIDTH        -1:0]  gdma_dout_o  ,
    input  logic [1:0]                     ce_ena_i     ,
    input  logic [1:0]                     ce_wen_i     ,
    input  logic [1:0][ADDR_WIDTH   -1:0]  ce_addr_i    ,
    input  logic [1:0][DATA_WIDTH   -1:0]  ce_din_i     ,
    output logic [1:0][DATA_WIDTH   -1:0]  ce_dout_o    
);

logic [NUM_BANKS-1:0]                         ena_bus ;
logic [NUM_BANKS-1:0]                         wen_bus ;
logic [NUM_BANKS-1:0][$clog2(BANK_DEPTH)-1:0] addr_bus;
logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] din_bus ;
logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] dout_bus;

crossbar #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(DATA_WIDTH),
    .BANK_WIDTH(BANK_WIDTH),
    .BANK_DEPTH(BANK_DEPTH),
    .NUM_BANKS (NUM_BANKS )
) u_crossbar (
    .clk_i      (clk_i      ),
    .gdma_ena_i (gdma_ena_i ),
    .gdma_wen_i (gdma_wen_i ),
    .gdma_addr_i(gdma_addr_i),
    .gdma_din_i (gdma_din_i ),
    .gdma_dout_o(gdma_dout_o),
    .ce_ena_i   (ce_ena_i   ),
    .ce_wen_i   (ce_wen_i   ),
    .ce_addr_i  (ce_addr_i  ),
    .ce_din_i   (ce_din_i   ),
    .ce_dout_o  (ce_dout_o  ),
    .ena_o      (ena_bus    ),
    .wen_o      (wen_bus    ),
    .addr_o     (addr_bus   ),
    .din_o      (din_bus    ),
    .dout_i     (dout_bus   )
);

multibank_sram #(
    .DATA_WIDTH(DATA_WIDTH),
    .BANK_DEPTH(BANK_DEPTH),
    .NUM_BANKS (NUM_BANKS )
) u_multibank_sram (
    .clk_i  (clk_i   ),
    .ena_i  (ena_bus ),
    .wen_i  (wen_bus ),
    .addr_i (addr_bus),
    .din_i  (din_bus ),
    .dout_o (dout_bus)
);

endmodule
