//////////////////////////////////////////////////////////////////////////////////
// Copyright by FuxionLab
// 
// Designer     : Sihao Fu
// Create Date  : 2024/11/11
// Project Name : ZeroCore
// File Name    : dp_sram_masked.v
//
// Description  : Behavioral model for dual-port sram with write mask.

//
// Revision: 
// 
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////

module dp_sram_masked#(
    parameter WIDTH   = 32  ,
    parameter SIZE    = 256 ,
    parameter RSTABLE = 1   ,
    parameter RST_VAL = 0
)(
    input   wire                            clk         ,
    input   wire                            rstn        ,
    input   wire    [WIDTH-1:0]             din         ,
    input   wire                            wr_en       ,
    input   wire    [WIDTH/8-1:0]           wr_mask     ,
    input   wire    [$clog2(SIZE)-1:0]      wr_addr     ,
    input   wire    [$clog2(SIZE)-1:0]      rd_addr     ,
    input   wire                            rd_en       ,
    output  wire    [WIDTH-1:0]             dout
);

    reg [WIDTH-1:0] mem [0:SIZE-1];
    reg [WIDTH-1:0] rdat          ;

    assign dout = rdat;

    integer ram_index;
    initial begin
      for (ram_index = 0; ram_index < SIZE; ram_index = ram_index + 1)
        mem[ram_index] = {WIDTH{1'b0}};
    end
    
    generate
        if(RSTABLE) begin
            //Create Resetable SRAM
            integer i0;
            always @(posedge clk or negedge rstn) begin : Update
                if(!rstn) begin
                    for (i0 = 0; i0 < SIZE; i0=i0+1) begin
                        mem[i0] <= RST_VAL;
                    end
                    rdat <= 'd0;
                end
                else begin
                    if (rd_en) begin
                         rdat <= mem[rd_addr];
                    end 
                    if (wr_en) 
                        mem[wr_addr] <= din;
                end
            end
        end else begin
            //Create Non-Resetable SRAM
            integer i0;
            always @(posedge clk) begin : Update
                if (rd_en) begin
                    rdat <= mem[rd_addr];
                end 
                if (wr_en) begin
                    for (i0 = 0; i0 < WIDTH/8; i0=i0+1) begin
                        mem[wr_addr][i0*8 +: 8] <= din[i0*8 +: 8];
                    end
                end 
            end
        end
    endgenerate
endmodule