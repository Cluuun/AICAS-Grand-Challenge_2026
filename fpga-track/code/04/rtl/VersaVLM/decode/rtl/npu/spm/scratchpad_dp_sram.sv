//////////////////////////////////////////////////////////////////////////////////
// Copyright by FuxionLab
// 
// Designer     : Sihao Fu
// Create Date  : 2024/12/09
// Project Name : ZeroCore
// File Name    : scratchpad.sv
//
// Description  : Multi-banked scratchpad sram for NPU, enabling continuous address access for unaligned address,
//                so that the systolic array and sfu can always process complete vector input/output,
//                using dual-port sram ip instead of behavioral description.
//
// Revision: 
// 
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////

module scratchpad_dp_sram #(
    parameter   PE_WIDTH        = 16        ,
    parameter   PE_DATA_WIDTH   = 8         ,

    parameter   SPM_SIZE        = 1 << 20   ,
    parameter   RD_PORTS        = 2         ,   
    parameter   WR_PORTS        = 1         ,

    localparam  PE_IDX          = $clog2(PE_WIDTH)          ,
    localparam  PE_DATA_SIZE    = $clog2(PE_DATA_WIDTH/8)   ,
    localparam  SPM_ADDR_WIDTH  = $clog2(SPM_SIZE)          ,
    localparam  DUAL_PORTS      = 2                         
    
) (
    input  logic                                                    clk     ,
    input  logic                                                    rst_n   ,

    input  logic               [WR_PORTS-1:0]                       wr_en   ,
    
    input  logic [PE_WIDTH-1:0][WR_PORTS-1:0]                       wr_mask ,
    input  logic               [WR_PORTS-1:0][SPM_ADDR_WIDTH-1:0]   wr_addr ,
    input  logic [PE_WIDTH-1:0][WR_PORTS-1:0][PE_DATA_WIDTH-1:0]    din     ,    
    
    input  logic               [RD_PORTS-1:0]                       rd_en   ,  
    input  logic               [RD_PORTS-1:0][SPM_ADDR_WIDTH-1:0]   rd_addr ,
    output logic [PE_WIDTH-1:0][RD_PORTS-1:0][PE_DATA_WIDTH-1:0]    dout                   
    
);

    // TODO: current multi-bank architecture is actually crossbar(bank_*_sel), 
    //       should be updated to other architecture with lower area, e.g. real dual port and data preload

    // bank select signals
    // bank_num = PE_WIDTH
    localparam BANK_DEPTH   =  1024;
    localparam SPM_PARTS    = (SPM_SIZE/(PE_WIDTH*(PE_DATA_WIDTH/8)*BANK_DEPTH))   ;  // SPM seperated into 5 parts
    localparam BANK_ADDR_WIDTH = SPM_ADDR_WIDTH-PE_IDX-PE_DATA_SIZE;
    logic   [PE_WIDTH-1:0][WR_PORTS-1:0]                                bank_wr_sel     ;
    logic   [PE_WIDTH-1:0][RD_PORTS-1:0]                                bank_rd_sel     ;

    logic   [PE_WIDTH-1:0][WR_PORTS-1:0][BANK_ADDR_WIDTH-1:0]           bank_wr_addr    ;
    logic   [PE_WIDTH-1:0][RD_PORTS-1:0][BANK_ADDR_WIDTH-1:0]           bank_rd_addr    ; 

    logic   [PE_WIDTH-1:0][WR_PORTS-1:0][PE_DATA_WIDTH-1:0]             wr_data         ;
    logic   [PE_WIDTH-1:0][RD_PORTS-1:0][PE_DATA_WIDTH-1:0]             rd_data         ;

    logic   [PE_WIDTH-1:0][WR_PORTS*PE_DATA_WIDTH-1:0]                  new_data        ;
    logic   [PE_WIDTH-1:0][RD_PORTS*PE_DATA_WIDTH-1:0]                  data_out        ;

    logic   [WR_PORTS-1:0][PE_WIDTH-1:0][PE_IDX-1:0]                    wr_idx          ;
    logic   [RD_PORTS-1:0][PE_WIDTH-1:0][PE_IDX-1:0]                    rd_idx          ;
    logic   [RD_PORTS-1:0][SPM_ADDR_WIDTH-1:0]                          rd_addr_dly     ;

    logic   [PE_WIDTH-1:0][WR_PORTS-1:0][SPM_PARTS-1:0]                       bank_part_wr_sel   ;
    logic   [PE_WIDTH-1:0][RD_PORTS-1:0][SPM_PARTS-1:0]                       bank_part_rd_sel   ;
    logic   [PE_WIDTH-1:0][DUAL_PORTS-1:0][SPM_PARTS-1:0]                       bank_part_sel    ;
    logic   [PE_WIDTH-1:0][DUAL_PORTS-1:0][SPM_PARTS-1:0][$clog2(BANK_DEPTH)-1:0]  bank_part_addr;
    // logic   [PE_WIDTH-1:0][DUAL_PORTS-1:0][SPM_PARTS-1:0][PE_DATA_WIDTH-1:0]    wr_part_data    ;
    logic   [PE_WIDTH-1:0][DUAL_PORTS-1:0][SPM_PARTS-1:0][PE_DATA_WIDTH-1:0]    rd_part_data     ;
    localparam SPM_PARTS_WIDTH = SPM_PARTS==1?1:$clog2(SPM_PARTS);
    logic   [PE_WIDTH-1:0][DUAL_PORTS-1:0][SPM_PARTS_WIDTH-1:0]             rd_part_data_sel;
   
    always_ff@(posedge clk or negedge rst_n)begin
        if(~rst_n)begin
        rd_addr_dly <=  '0;
        rd_part_data_sel <= '0;
        end
        else if(|rd_en)begin
        rd_addr_dly <=  rd_addr;
         for (int idx = 0; idx < PE_WIDTH; idx++)begin
            for (int p = 0; p < RD_PORTS; p++)begin
                if(SPM_PARTS==1)
                    rd_part_data_sel[idx][p] <='0;
                else 
                    rd_part_data_sel[idx][p] <= bank_rd_addr[idx][p][BANK_ADDR_WIDTH-1 -: SPM_PARTS_WIDTH];
            end
         end
        end
        else begin
        rd_addr_dly <=  rd_addr_dly;
        rd_part_data_sel <= rd_part_data_sel;
        end
    end

    always_comb begin : bank_ctrl
        dout = '0;
        wr_data = '0;
        for (int idx = 0; idx < PE_WIDTH; idx++)begin 
            for (int p = 0; p < WR_PORTS; p++)begin 
                wr_idx[p][idx]          = idx - wr_addr[p][PE_DATA_SIZE +: PE_IDX];
                bank_wr_sel[idx][p]     = wr_mask[wr_idx[p][idx]][p] && wr_en[p]; 
                bank_wr_addr[idx][p]    = (wr_addr[p][PE_DATA_SIZE +: PE_IDX] > idx) ?(wr_addr[p][SPM_ADDR_WIDTH-1 : PE_IDX+PE_DATA_SIZE] + 1):
                (wr_addr[p][SPM_ADDR_WIDTH-1: PE_IDX+PE_DATA_SIZE]) ;
                wr_data[idx][p]         = din[wr_idx[p][idx]][p] ;
            end
            for (int p = 0; p < RD_PORTS; p++)begin
                rd_idx[p][idx]          = idx - rd_addr_dly[p][PE_DATA_SIZE +: PE_IDX];
                bank_rd_sel[idx][p]     = rd_en[p];
                bank_rd_addr[idx][p]    = (rd_addr[p][PE_DATA_SIZE +: PE_IDX] > idx) ? (rd_addr[p][SPM_ADDR_WIDTH-1 : PE_IDX+PE_DATA_SIZE]+1) : 
                    (rd_addr[p][SPM_ADDR_WIDTH-1: PE_IDX+PE_DATA_SIZE] );
                dout[rd_idx[p][idx]][p] = rd_data[idx][p] ;
            end
        end
    end

    always_comb begin : array_switch
        for (int idx = 0; idx < PE_WIDTH; idx++)begin
            for (int p = 0; p < WR_PORTS; p++)begin
                new_data[idx][p*PE_DATA_WIDTH +: PE_DATA_WIDTH] = wr_data[idx][p];
            end
            for (int p = 0; p < RD_PORTS; p++)begin
                // rd_data[idx][p] = data_out[idx][p*PE_DATA_WIDTH +: PE_DATA_WIDTH];
                rd_data[idx][p] = rd_part_data[idx][p][rd_part_data_sel[idx][p]];
            end
        end
    end

    always_comb begin : bank_part
        for (int idx = 0; idx < PE_WIDTH; idx++)begin
            for (int port = 0; port < WR_PORTS; port++)begin
                for (int part = 0; part < SPM_PARTS; part++)begin
                    if(SPM_PARTS==1)
                        bank_part_wr_sel[idx][port][part] = bank_wr_sel[idx][port];
                    else
                        bank_part_wr_sel[idx][port][part] = (bank_wr_addr[idx][port][BANK_ADDR_WIDTH-1 -: SPM_PARTS_WIDTH] == part) ? 
                        bank_wr_sel[idx][port] : 1'b0;
                end
            end
            for (int port = 0; port < RD_PORTS; port++)begin
                for (int part = 0; part < SPM_PARTS; part++)begin
                    if(SPM_PARTS==1)
                        bank_part_rd_sel[idx][port][part] = bank_rd_sel[idx][port];
                    else
                        bank_part_rd_sel[idx][port][part] = (bank_rd_addr[idx][port][BANK_ADDR_WIDTH-1 -: SPM_PARTS_WIDTH] == part) ? 
                        bank_rd_sel[idx][port] : 1'b0;
                end
            end
            for (int port = 0; port < DUAL_PORTS; port++)begin
                for (int part = 0; part < SPM_PARTS; part++)begin
                    bank_part_sel[idx][port][part] = (port == 0) ? (bank_part_wr_sel[idx][port][part] | bank_part_rd_sel[idx][port][part]) : bank_part_rd_sel[idx][port][part];
                    bank_part_addr[idx][port][part] = bank_part_rd_sel[idx][port][part] ? bank_rd_addr[idx][port][$clog2(BANK_DEPTH)-1:0] : bank_wr_addr[idx][0][$clog2(BANK_DEPTH)-1:0];
                end
            end
        end

    end


    generate
    genvar idx;
    genvar part;
    if(PE_DATA_WIDTH == 8)begin : spm_gen
        for (idx = 0; idx < PE_WIDTH; idx++) begin: scratchpad
            for (part = 0; part < SPM_PARTS; part++) begin
                sram_8_1024_dp u_sram_dp
                (
                    // unused, for BIST test
                    .CENYA      (),
                    .WENYA      (),
                    .AYA        (),
                    .CENYB      (),
                    .WENYB      (),
                    .AYB        (),
                    .GWENYA     (),
                    .GWENYB     (),
                    .SOA        (),
                    .SOB        (),

                    // used
                    .QA         ( rd_part_data[idx][0][part] ),
                    .QB         ( rd_part_data[idx][1][part] ),
                    
                    .CLKA       ( clk ),
                    .CENA       ( ~bank_part_sel[idx][0][part] ),
                    .WENA       ( ~{(PE_DATA_WIDTH){bank_part_wr_sel[idx][0][part]}} ),
                    .AA         ( bank_part_addr[idx][0][part] ),
                    .DA         ( wr_data[idx][0]),
                    .CLKB       ( clk ),
                    .CENB       ( ~bank_part_sel[idx][1][part]),
                    .WENB       ( ~{(PE_DATA_WIDTH){1'b0}} ),
                    .AB         ( bank_part_addr[idx][1][part]),
                    .DB         ( '0),

                    .EMAA       ( '0 ),
                    .EMAWA      ( '0 ),
                    .EMASA      ( '0 ),
                    .EMAB       ( '0 ),
                    .EMAWB      ( '0 ),
                    .EMASB      ( '0 ),

                    .TENA       ( '1 ),
                    .TCENA      ( '0 ),
                    .TWENA      ( '0 ),
                    .TAA        ( '0 ),
                    .TDA        ( '0 ),
                    .TENB       ( '1 ),
                    .TCENB      ( '0 ),
                    .TWENB      ( '0 ),
                    .TAB        ( '0 ),
                    .TDB        ( '0 ),

                    .GWENA      ( ~bank_part_wr_sel[idx][0][part] ),
                    .GWENB      ( '1 ),
                    .TGWENA     ( '0 ),
                    .TGWENB     ( '0 ),
                    .RET1N      ( '1 ),
                    .SIA        ( '0 ),
                    .SEA        ( '0 ),
                    .DFTRAMBYP  ( '0 ),
                    .SIB        ( '0 ),
                    .SEB        ( '0 ),
                    .COLLDISN   ( '0 )
                );
            end
        end
    end
    else begin : acc_gen
        for (idx = 0; idx < PE_WIDTH; idx++) begin: scratchpad
            for (part = 0; part < SPM_PARTS; part++) begin
                sram_32_1024_dp u_sram_dp
                (
                    // unused, for BIST test
                    .CENYA      (),
                    .WENYA      (),
                    .AYA        (),
                    .CENYB      (),
                    .WENYB      (),
                    .AYB        (),
                    .GWENYA     (),
                    .GWENYB     (),
                    .SOA        (),
                    .SOB        (),

                    // used
                    .QA         ( rd_part_data[idx][0][part] ),
                    .QB         ( rd_part_data[idx][1][part] ),
                    
                    .CLKA       ( clk ),
                    .CENA       ( ~bank_part_sel[idx][0][part] ),
                    .WENA       ( ~{(PE_DATA_WIDTH){bank_part_wr_sel[idx][0][part]}} ),
                    .AA         ( bank_part_addr[idx][0][part] ),
                    .DA         ( wr_data[idx][0]),
                    .CLKB       ( clk ),
                    .CENB       ( ~bank_part_sel[idx][1][part]),
                    .WENB       ( ~{(PE_DATA_WIDTH){1'b0}} ),
                    .AB         ( bank_part_addr[idx][1][part]),
                    .DB         ( '0),

                    .EMAA       ( '0 ),
                    .EMAWA      ( '0 ),
                    .EMASA      ( '0 ),
                    .EMAB       ( '0 ),
                    .EMAWB      ( '0 ),
                    .EMASB      ( '0 ),

                    .TENA       ( '1 ),
                    .TCENA      ( '0 ),
                    .TWENA      ( '0 ),
                    .TAA        ( '0 ),
                    .TDA        ( '0 ),
                    .TENB       ( '1 ),
                    .TCENB      ( '0 ),
                    .TWENB      ( '0 ),
                    .TAB        ( '0 ),
                    .TDB        ( '0 ),

                    .GWENA      ( ~bank_part_wr_sel[idx][0][part] ),
                    .GWENB      ( '1 ),
                    .TGWENA     ( '0 ),
                    .TGWENB     ( '0 ),
                    .RET1N      ( '1 ),
                    .SIA        ( '0 ),
                    .SEA        ( '0 ),
                    .DFTRAMBYP  ( '0 ),
                    .SIB        ( '0 ),
                    .SEB        ( '0 ),
                    .COLLDISN   ( '0 )
                );
            end
        end
    end
    endgenerate

endmodule
