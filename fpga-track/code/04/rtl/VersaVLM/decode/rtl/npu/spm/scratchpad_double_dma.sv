//////////////////////////////////////////////////////////////////////////////////
// Copyright by FuxionLab
// 
// Designer     : Sihao Fu
// Create Date  : 2024/12/09
// Project Name : ZeroCore
// File Name    : scratchpad.sv
//
// Description  : Multi-banked scratchpad sram for NPU, enabling continuous address access for unaligned address,
//                so that the systolic array and sfu can always process complete vector input/output. 
//
// Revision: 
// 
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////

module scratchpad_double_dma #(
    parameter   SPM_FPGA_SRAM   = 0         ,
    parameter   PE_WIDTH        = 16        ,
    parameter   PE_DATA_WIDTH   = 16        ,

    parameter   SPM_SIZE        = 1 << 20   ,
    parameter   RD_PORTS        = 2         ,   
    parameter   WR_PORTS        = 2         ,

    localparam  PE_IDX          = $clog2(PE_WIDTH)          ,
    localparam  PE_DATA_SIZE    = $clog2(PE_DATA_WIDTH/8)   ,
    localparam  SPM_ADDR_WIDTH  = $clog2(SPM_SIZE)          
    
) (
    input  logic                                                    clk     ,
    input  logic                                                    rst_n   ,

    input  logic                                                    dma0_wr_en   ,
    input  logic [PE_WIDTH-1:0]                                     dma0_wr_mask ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 dma0_wr_addr ,
    input  logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  dma0_din     ,    

    input  logic                                                    dma1_wr_en   ,
    input  logic [PE_WIDTH-1:0]                                     dma1_wr_mask ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 dma1_wr_addr ,
    input  logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  dma1_din     , 

    input  logic                                                    sfu_wr_en    ,
    input  logic [PE_WIDTH-1:0]                                     sfu_wr_mask  ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 sfu_wr_addr  ,
    input  logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  sfu_din      , 

    input  logic                                                    accu_wr_en   ,
    input  logic [PE_WIDTH-1:0]                                     accu_wr_mask ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 accu_wr_addr ,
    input  logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  accu_din     , 
    
    // DMA read port (uses rd port 0)
    input  logic                                                    dma0_rd_en       ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 dma0_rd_addr     ,
    output logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  dma0_rd_data     ,
    
    // DMA read port 1 (uses rd port 1)
    input  logic                                                    dma1_rd_en       ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 dma1_rd_addr     ,
    output logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  dma1_rd_data     ,

    // GEMV read port (shares rd port 0 with DMA0/SA1)
    input  logic                                                    gemv_rd_en       ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 gemv_rd_addr     ,
    output logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  gemv_rd_data     ,
    
    // SA read port 1 (uses rd port 0)
    input  logic                                                    sa_rd1_en       ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 sa_rd1_addr     ,
    output logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  sa_rd1_data     ,
    
    // SA read port 2 (uses rd port 1)
    input  logic                                                    sa_rd2_en       ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 sa_rd2_addr     ,
    output logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  sa_rd2_data     ,
    
    // SFU read port (uses rd port 1)
    input  logic                                                    sfu_rd_en       ,
    input  logic               [SPM_ADDR_WIDTH-1:0]                 sfu_rd_addr     ,
    output logic [PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]                  sfu_rd_data      
    
);

    // TODO: current multi-bank architecture is actually crossbar(bank_*_sel), 
    //       should be updated to other architecture with lower area, e.g. real dual port and data preload

    // SPM write port control
    logic               [WR_PORTS-1:0]                       wr_en  ;
    logic [PE_WIDTH-1:0][WR_PORTS-1:0]                       wr_mask;
    logic               [WR_PORTS-1:0][SPM_ADDR_WIDTH-1:0]   wr_addr;
    logic [PE_WIDTH-1:0][WR_PORTS-1:0][PE_DATA_WIDTH-1:0]    din    ;
    logic [WR_PORTS-1:0][PE_WIDTH-1:0]                       port_wr_mask ;
    logic [WR_PORTS-1:0][PE_WIDTH-1:0][PE_DATA_WIDTH-1:0]    port_din     ;
    always_comb begin : spm_wr_control    
        wr_en                   = '0;
        port_wr_mask            = '0;
        port_din                = '0;
        wr_addr                 = '0;

        // Write port0 muxing based on active module
        if (dma0_wr_en) begin
            wr_en [0]           = dma0_wr_en;
            port_wr_mask [0]    = dma0_wr_mask;
            port_din[0]         = dma0_din;
            wr_addr[0]          = dma0_wr_addr;
        end else if (sfu_wr_en) begin
            wr_en [0]           = sfu_wr_en;
            port_wr_mask [0]    = sfu_wr_mask;
            port_din[0]         = sfu_din;
            wr_addr[0]          = sfu_wr_addr; 
        end else if (accu_wr_en) begin
            wr_en [0]           = accu_wr_en;
            port_wr_mask [0]    = accu_wr_mask;
            port_din[0]         = accu_din;
            wr_addr[0]          = accu_wr_addr;
        end

        if (dma1_wr_en) begin
            wr_en [1]           = dma1_wr_en;
            port_wr_mask [1]    = dma1_wr_mask;
            port_din[1]         = dma1_din;
            wr_addr[1]          = dma1_wr_addr;
        end
    end

    always_comb begin : port_switch
        for (int idx = 0; idx < PE_WIDTH; idx++) begin
            for (int p = 0; p < WR_PORTS; p++) begin
                wr_mask[idx][p]  = port_wr_mask[p][idx];
                din [idx][p]     = port_din[p][idx] ;
            end
        end
    end

    // Internal read port signals after muxing
    logic               [RD_PORTS-1:0]                       rd_en   ;
    logic               [RD_PORTS-1:0][SPM_ADDR_WIDTH-1:0]   rd_addr ;
    logic [PE_WIDTH-1:0][RD_PORTS-1:0][PE_DATA_WIDTH-1:0]    dout    ;
    
    reg dma0_rd_en_dly;
    reg dma1_rd_en_dly;
    reg gemv_rd_en_dly;
    reg sa_rd1_en_dly;
    reg sa_rd2_en_dly;
    reg sfu_rd_en_dly;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma0_rd_en_dly <= 0;
            dma1_rd_en_dly <= 0;
            gemv_rd_en_dly <= 0;
            sa_rd1_en_dly <= 0;
            sa_rd2_en_dly <= 0;
            sfu_rd_en_dly <= 0;
        end else begin
            dma0_rd_en_dly <= dma0_rd_en;
            dma1_rd_en_dly <= dma1_rd_en;
            gemv_rd_en_dly <= gemv_rd_en;
            sa_rd1_en_dly <= sa_rd1_en;
            sa_rd2_en_dly <= sa_rd2_en;
            sfu_rd_en_dly <= sfu_rd_en;
        end
    end
    // Read port 0 selection: DMA0 or GEMV or SA_RD1 (priority: DMA0 > GEMV > SA)
    // Read port 1 selection: DMA1 or SA_RD2 or SFU (priority: DMA1 > SA > SFU)
    always_comb begin
        // Read port 0: DMA0 has priority over GEMV and SA_RD1
        if (dma0_rd_en) begin
            rd_en[0]   = dma0_rd_en;
            rd_addr[0] = dma0_rd_addr;
        end else if (gemv_rd_en) begin
            rd_en[0]   = gemv_rd_en;
            rd_addr[0] = gemv_rd_addr;
        end else begin
            rd_en[0]   = sa_rd1_en;
            rd_addr[0] = sa_rd1_addr;
        end
        
        // Read port 1: DMA1 has priority over SA_RD2 and SFU
        if (dma1_rd_en) begin
            rd_en[1]   = dma1_rd_en;
            rd_addr[1] = dma1_rd_addr;
        end else if (sa_rd2_en) begin
            rd_en[1]   = sa_rd2_en;
            rd_addr[1] = sa_rd2_addr;
        end else begin
            rd_en[1]   = sfu_rd_en;
            rd_addr[1] = sfu_rd_addr;
        end
    end
    
    // Output data routing
    always_comb begin
        // Port 0 outputs
        dma0_rd_data = '0;
        gemv_rd_data = '0;
        dma1_rd_data = '0;
        sa_rd1_data = '0;
        if (dma0_rd_en_dly) begin
            for (int i = 0; i < PE_WIDTH; i++) begin
                dma0_rd_data[i] = dout[i][0];
            end
        end else if (gemv_rd_en_dly) begin
            for (int i = 0; i < PE_WIDTH; i++) begin
                gemv_rd_data[i] = dout[i][0];
            end
        end else if (sa_rd1_en_dly) begin
            for (int i = 0; i < PE_WIDTH; i++) begin
                sa_rd1_data[i] = dout[i][0];
            end
        end else begin
            sa_rd1_data = '0;
        end
        
        // Port 1 outputs
        sa_rd2_data = '0;
        sfu_rd_data = '0;
        if (dma1_rd_en_dly) begin
            for (int i = 0; i < PE_WIDTH; i++) begin
                dma1_rd_data[i] = dout[i][1];
            end
        end else if (sa_rd2_en_dly) begin
            for (int i = 0; i < PE_WIDTH; i++) begin
                sa_rd2_data[i] = dout[i][1];
            end
        end else if (sfu_rd_en_dly) begin
            for (int i = 0; i < PE_WIDTH; i++) begin
                sfu_rd_data[i] = dout[i][1];
            end
        end
    end

    // bank select signals
    // bank_num = PE_WIDTH
    logic   [PE_WIDTH-1:0][WR_PORTS-1:0]                                bank_wr_sel     ;
    logic   [PE_WIDTH-1:0][RD_PORTS-1:0]                                bank_rd_sel     ;

    logic   [PE_WIDTH-1:0][WR_PORTS-1:0][SPM_ADDR_WIDTH-PE_IDX-PE_DATA_SIZE-1:0]     bank_wr_addr    ;
    logic   [PE_WIDTH-1:0][RD_PORTS-1:0][SPM_ADDR_WIDTH-PE_IDX-PE_DATA_SIZE-1:0]     bank_rd_addr    ; 

    logic   [PE_WIDTH-1:0][WR_PORTS-1:0][PE_DATA_WIDTH-1:0]             wr_data         ;
    logic   [PE_WIDTH-1:0][RD_PORTS-1:0][PE_DATA_WIDTH-1:0]             rd_data         ;

    logic   [PE_WIDTH-1:0][WR_PORTS*PE_DATA_WIDTH-1:0]                  new_data        ;
    logic   [PE_WIDTH-1:0][RD_PORTS*PE_DATA_WIDTH-1:0]                  data_out        ;

    logic   [WR_PORTS-1:0][PE_WIDTH-1:0][PE_IDX-1:0]wr_idx;
    logic   [RD_PORTS-1:0][PE_WIDTH-1:0][PE_IDX-1:0]rd_idx;
    logic   [RD_PORTS-1:0][SPM_ADDR_WIDTH-1:0]                          rd_addr_dly    ;
    always_ff@(posedge clk or negedge rst_n)begin
        if(~rst_n)
        rd_addr_dly<=0;
        else if(|rd_en)
        rd_addr_dly<=rd_addr;
        else 
        rd_addr_dly<=rd_addr_dly;
    end

    always_comb begin : bank_ctrl
        dout= '0;
        for (int idx = 0; idx < PE_WIDTH; idx++)begin 
            for (int p = 0; p < WR_PORTS; p++)begin 
                wr_idx[p][idx]=idx -wr_addr[p][PE_DATA_SIZE +: PE_IDX];
                bank_wr_sel[idx][p]     = wr_mask[wr_idx[p][idx]][p] && wr_en[p]; 
                bank_wr_addr[idx][p]    = (wr_addr[p][PE_DATA_SIZE +: PE_IDX] > idx) ?(wr_addr[p][SPM_ADDR_WIDTH-1 : PE_IDX+PE_DATA_SIZE] + 1):
                (wr_addr[p][SPM_ADDR_WIDTH-1: PE_IDX+PE_DATA_SIZE]) ;
                wr_data[idx][p]         = din[wr_idx[p][idx]][p] ;
            end
            for (int p = 0; p < RD_PORTS; p++)begin
                rd_idx[p][idx]=idx -rd_addr_dly[p][PE_DATA_SIZE +: PE_IDX];
                bank_rd_sel[idx][p]     = rd_en[p];
                bank_rd_addr[idx][p]    = (rd_addr[p][PE_DATA_SIZE +: PE_IDX] > idx) ? (rd_addr[p][SPM_ADDR_WIDTH-1 : PE_IDX+PE_DATA_SIZE]+1) : 
                    (rd_addr[p][SPM_ADDR_WIDTH-1: PE_IDX+PE_DATA_SIZE] ) ;
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
                rd_data[idx][p] = data_out[idx][p*PE_DATA_WIDTH +: PE_DATA_WIDTH];
            end
        end
    end
        
    generate
        genvar idx;
        for (idx = 0; idx < PE_WIDTH; idx++) begin: scratchpad
            if (SPM_FPGA_SRAM == 1) begin : gen_sram_fpga
                sram_fpga#(
                .DATA_WIDTH     (PE_DATA_WIDTH            ),
                .ADDR_WIDTH     (SPM_ADDR_WIDTH - PE_IDX - PE_DATA_SIZE        ),
                .RAM_STYLE      ("block"        )
                )
                u_sram_fpga(
                    .clk                                (clk                       ),
                // Port A
                    .we_a                               (bank_wr_sel[idx][0]                      ),
                    .addr_a                             (bank_wr_sel[idx][0] ? bank_wr_addr[idx][0] :  bank_rd_addr[idx][0]                   ),
                    .din_a                              (wr_data[idx][0]                 ),
                    .wr_mask_a                          ({PE_DATA_WIDTH/8{1'b1}}),
                    .dout_a                             (data_out[idx][PE_DATA_WIDTH-1:0]               ),
                // Port B
                    .we_b                               (bank_wr_sel[idx][1]                      ),
                    .addr_b                             (bank_wr_sel[idx][1] ? bank_wr_addr[idx][1] :  bank_rd_addr[idx][1]                   ),
                    .din_b                              (wr_data[idx][1]                 ),
                    .wr_mask_b                          ({PE_DATA_WIDTH/8{1'b1}}),
                    .dout_b                             (data_out[idx][PE_DATA_WIDTH*2-1:PE_DATA_WIDTH]                    )
                );
            end
            else begin : gen_sram_asic
                sram_mp #(
                    .TYPE           (1),
                    .SIZE           ((SPM_SIZE >> PE_IDX)>>PE_DATA_SIZE),
                    .DATA_WIDTH     (PE_DATA_WIDTH),
                    .RD_PORTS       (RD_PORTS),     
                    .WR_PORTS       (WR_PORTS),         
                    .RESETVAL       (0),
                    .RESETABLE      (1)
                )
                u_sram_mp
                (   .clk           (clk),
                    .rst_n         (rst_n),

                    .wr_en         (bank_wr_sel[idx]    ),
                    .write_address (bank_wr_addr[idx]   ),
                    .new_data      (new_data[idx]       ),

                    .rd_en         (bank_rd_sel[idx]    ),
                    .read_address  (bank_rd_addr[idx]   ),
                    .data_out      (data_out[idx]       )
                );
            end
        end
    endgenerate

endmodule
