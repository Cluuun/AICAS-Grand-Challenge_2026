//*****************************************************************
// 'Empty' stub module for FPGA verification.
//
// This module has the exact same interface as 'scratchpad'
// but all outputs (data out) are tied to 0.
//*****************************************************************
module scratchpad_empty #(
    parameter   PE_WIDTH        = 16    ,
    parameter   PE_DATA_WIDTH   = 16    ,

    parameter   SPM_SIZE        = 1 << 20,
    parameter   RD_PORTS        = 2     ,    
    parameter   WR_PORTS        = 1     ,

    localparam  PE_IDX          = $clog2(PE_WIDTH)     ,
    localparam  PE_DATA_SIZE    = $clog2(PE_DATA_WIDTH/8),
    localparam  SPM_ADDR_WIDTH  = $clog2(SPM_SIZE)     
    
) (
    input  logic                                        clk,
    input  logic                                        rst_n,

    // Write Ports (Inputs - unused in this stub)
    input  logic               [WR_PORTS-1:0]           wr_en,
    input  logic [PE_WIDTH-1:0][WR_PORTS-1:0]           wr_mask,
    input  logic               [WR_PORTS-1:0][SPM_ADDR_WIDTH-1:0] wr_addr,
    input  logic [PE_WIDTH-1:0][WR_PORTS-1:0][PE_DATA_WIDTH-1:0] din,    
    
    // Read Ports (Inputs & Outputs)
    input  logic               [RD_PORTS-1:0]           rd_en, 
    input  logic               [RD_PORTS-1:0][SPM_ADDR_WIDTH-1:0] rd_addr,
    output logic [PE_WIDTH-1:0][RD_PORTS-1:0][PE_DATA_WIDTH-1:0] dout // Output
);

    // ------------------------------------- 
    // Tie all outputs to zero.
    // ------------------------------------- 

    // Read Data Output
    // This assigns all bits of the entire multi-dimensional output vector to 0.
    assign dout = '0; 

endmodule
