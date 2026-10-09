/*
双端口，同时调用两个bank
注意两个bank_sel地址相同时会发生冲突覆盖
*/

module crossbar #(
    parameter ADDR_WIDTH = 32 ,
    parameter DATA_WIDTH = 256,
    parameter BANK_WIDTH = 256,
    parameter BANK_DEPTH = 128,
    parameter NUM_BANKS  = 8
)(
    input  logic                                         clk_i      ,

    input  logic                                         gdma_ena_i ,
    input  logic                                         gdma_wen_i ,
    input  logic [ADDR_WIDTH             -1:0]           gdma_addr_i,
    input  logic [DATA_WIDTH             -1:0]           gdma_din_i ,
    output logic [DATA_WIDTH             -1:0]           gdma_dout_o,

    input  logic [1:0]                                   ce_ena_i   ,
    input  logic [1:0]                                   ce_wen_i   ,
    input  logic [1:0][ADDR_WIDTH        -1:0]           ce_addr_i  ,
    input  logic [1:0][DATA_WIDTH        -1:0]           ce_din_i   ,
    output logic [1:0][DATA_WIDTH        -1:0]           ce_dout_o  ,

    output logic [NUM_BANKS-1:0]                         ena_o      ,
    output logic [NUM_BANKS-1:0]                         wen_o      ,
    output logic [NUM_BANKS-1:0][$clog2(BANK_DEPTH)-1:0] addr_o     ,
    output logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] din_o      ,
    input  logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] dout_i     
);

localparam BANK_ADDR_SPACE = BANK_DEPTH * (BANK_WIDTH/8);
localparam BANK_ADDR_SHIFT = $clog2(BANK_ADDR_SPACE);

logic [$clog2(NUM_BANKS)-1:0] gdma_bank_sel;
logic [$clog2(BANK_DEPTH)-1:0] gdma_bank_addr;
logic [NUM_BANKS-1:0]                         gdma_ena ;
logic [NUM_BANKS-1:0]                         gdma_wen ;
logic [NUM_BANKS-1:0][$clog2(BANK_DEPTH)-1:0] gdma_addr;
logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] gdma_din ;

logic [$clog2(NUM_BANKS)-1:0] ce_bank_sel [1:0];
logic [$clog2(BANK_DEPTH)-1:0] ce_bank_addr [1:0];
logic [NUM_BANKS-1:0]                         ce_ena ;
logic [NUM_BANKS-1:0]                         ce_wen ;
logic [NUM_BANKS-1:0][$clog2(BANK_DEPTH)-1:0] ce_addr;
logic [NUM_BANKS-1:0][DATA_WIDTH        -1:0] ce_din ;

// Bank decoding and address translation
assign gdma_bank_sel  = (gdma_addr_i - `BASE_ADDR_OCM) >> BANK_ADDR_SHIFT;
assign gdma_bank_addr = ((gdma_addr_i - `BASE_ADDR_OCM) & (BANK_ADDR_SPACE - 1)) >> $clog2(DATA_WIDTH/8);// (gdma_addr_i % BANK_ADDR_SPACE) >> $clog2(DATA_WIDTH/8);

always_comb begin
    for (int i = 0; i < 2; i++) begin
        ce_bank_sel[i]  = (ce_addr_i[i] - `BASE_ADDR_OCM) >> BANK_ADDR_SHIFT;
        ce_bank_addr[i] = ((ce_addr_i[i] - `BASE_ADDR_OCM) & (BANK_ADDR_SPACE - 1)) >> $clog2(DATA_WIDTH/8);
    end
end

always_comb begin
    gdma_ena  = '0;
    gdma_wen  = '0;
    gdma_addr = '0;
    gdma_din  = '0;
    gdma_dout_o = '0;
    if (gdma_ena_i) begin
        gdma_ena[gdma_bank_sel]  = 1'b1;
        gdma_wen[gdma_bank_sel]  = gdma_wen_i;
        gdma_addr[gdma_bank_sel] = gdma_bank_addr;
        gdma_din[gdma_bank_sel]  = gdma_din_i;
        gdma_dout_o              = dout_i[gdma_bank_sel];
    end
end

always_comb begin
    ce_ena  = '0;
    ce_wen  = '0;
    ce_addr = '0;
    ce_din  = '0;
    ce_dout_o = '0;
    for (int i = 0; i < 2; i++) begin
        if (ce_ena_i[i]) begin
            ce_ena[ce_bank_sel[i]]  = 1'b1;
            ce_wen[ce_bank_sel[i]]  = ce_wen_i[i];
            ce_addr[ce_bank_sel[i]] = ce_bank_addr[i];
            ce_din[ce_bank_sel[i]]  = ce_din_i[i];
            ce_dout_o[i]            = dout_i[ce_bank_sel[i]];
        end
    end
end

always_comb begin
    for (int i = 0; i < NUM_BANKS; i++) begin
        ena_o [i] = gdma_ena [i] | ce_ena [i];
        wen_o [i] = gdma_wen [i] | ce_wen [i];
        addr_o[i] = gdma_addr[i] | ce_addr[i];
        din_o [i] = gdma_din [i] | ce_din [i];
    end
end

endmodule
