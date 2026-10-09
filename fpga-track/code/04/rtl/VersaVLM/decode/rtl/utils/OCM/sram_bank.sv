module sram_bank #(
    parameter int DATA_WIDTH = 256,
    parameter int DEPTH = 128
)(
    input  logic                      clk_i ,
    input  logic                      ena   ,
    input  logic                      wen   ,
    input  logic [$clog2(DEPTH)-1:0]  addr  ,
    input  logic [DATA_WIDTH   -1:0]  din   ,
    output logic [DATA_WIDTH   -1:0]  dout  
);

    // logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    sram_128_128 u1_sram_128_128 (
        .Q     (dout[DATA_WIDTH/2-1:0]),
        .CLK   (clk_i                 ),
        .CEN   (~ena                  ),
        .WEN   ({128{~wen}}           ),
        .A     (addr                  ),
        .D     (din[DATA_WIDTH/2-1:0] ),
        .EMA   (3'b000                ),
        .EMAW  (2'b00                 ),
        .EMAS  (1'b0                  ),
        .GWEN  (~wen                  ),
        .RET1N (1'b1                  )
    );

    sram_128_128 u2_sram_128_128 (
        .Q     (dout[DATA_WIDTH-1:DATA_WIDTH/2]),
        .CLK   (clk_i                          ),
        .CEN   (~ena                           ),
        .WEN   ({128{~wen}}                    ),
        .A     (addr                           ),
        .D     (din[DATA_WIDTH-1:DATA_WIDTH/2] ),
        .EMA   (3'b000                         ),
        .EMAW  (2'b00                          ),
        .EMAS  (1'b0                           ),
        .GWEN  (~wen                           ),
        .RET1N (1'b1                           )
    );

    // always_ff @(posedge clk_i) begin
        // if(ena && wen)
            // mem[addr] <= din;
    // end

    // always_ff @(posedge clk_i) begin
        // if(ena && !wen)
            // dout <= mem[addr];
        // else
            // dout <= 'b0;
    // end

endmodule
