module axi_interface # (
    parameter int unsigned      ADDR_WIDTH = 32'd64,
    parameter int unsigned      DATA_WIDTH = 32'd512,
    parameter longint unsigned  ADDR_BASE  = 64'h80000000,
    parameter               DRAMType       = "DDR4",    //DRAM type
    parameter               CustomerDRAM   = "none",    //DRAM type
    parameter               InitPath       = "none",    //mem path
	parameter time          ClkPeriod	   = 1ns,
    
    parameter int unsigned  DRAM_LINEWIDTH, // ddr config

    localparam int unsigned DATA_BYTE       = DATA_WIDTH/8,
    localparam int unsigned DRAM_MASK_LEN   = $clog2(DRAM_LINEWIDTH/8),

    localparam int unsigned DRAM_ADDRWIDTH  = 64,  // ddr config
    localparam int unsigned DRAM_DATAWIDTH  = DATA_WIDTH > DRAM_LINEWIDTH ? DATA_WIDTH : DRAM_LINEWIDTH,
    localparam int unsigned LINE_PER_DATA   = DATA_WIDTH > DRAM_LINEWIDTH ? DATA_WIDTH/DRAM_LINEWIDTH : 'd1
) (
    input    clk_i,
    input    rst_ni,

    input                           axi_mem_awvalid_i,
    input [ADDR_WIDTH-1:0]          axi_mem_awaddr_i,
    input [3:0]                     axi_mem_awid_i,
    input [15:0]                    axi_mem_awlen_i,
    input [1:0]                     axi_mem_awburst_i,
    input                           axi_mem_wvalid_i,
    input [DATA_WIDTH-1:0]          axi_mem_wdata_i,
    input [DATA_WIDTH/8-1:0]        axi_mem_wstrb_i,
    input                           axi_mem_wlast_i,
    input                           axi_mem_bready_i,
    input                           axi_mem_arvalid_i,
    input [ADDR_WIDTH-1:0]          axi_mem_araddr_i,
    input [3:0]                     axi_mem_arid_i,
    input [15:0]                    axi_mem_arlen_i,
    input [1:0]                     axi_mem_arburst_i,
    input                           axi_mem_rready_i,

    output                      axi_mem_awready_o,
    output                      axi_mem_wready_o,
    output                      axi_mem_bvalid_o,
    output [1:0]                axi_mem_bresp_o,
    output [3:0]                axi_mem_bid_o,
    output                      axi_mem_arready_o,
    output                      axi_mem_rvalid_o,
    output [DATA_WIDTH-1:0]     axi_mem_rdata_o,
    output [1:0]                axi_mem_rresp_o,
    output [3:0]                axi_mem_rid_o,
    output                      axi_mem_rlast_o
);

    // requests ports
    logic                           req_valid_i;// request valid
    logic                           req_ready_o;// request ready
    logic                           we_i;       // write enable
    logic [DRAM_ADDRWIDTH-1:0]      addr_i;     // request address
    logic [DRAM_DATAWIDTH-1:0]      wdata_i;    // write data
    logic [DRAM_DATAWIDTH/8-1:0]    wstrb_i;    // write strb
    // read response
    logic                           rsp_valid_o;
    logic                           rsp_ready_i;
    logic [DRAM_DATAWIDTH-1:0]      rdata_o;     // read data
    // write response
    logic                           b_valid_o;
    logic                           b_ready_i;

    clock_engine #(
        .ClkPeriod(ClkPeriod)
    ) i_clock_engine (
        .*
    );

    sim_dram #(
        .DataWidth(DRAM_DATAWIDTH), 
        .AddrWidth(DRAM_ADDRWIDTH), 
        .DRAMType(DRAMType), 
        .CustomerDRAM(CustomerDRAM), 
        .InitPath(InitPath),
        .BASE(64'h0)
    ) i_sim_dram (
        .*
    );
    
    localparam NO_BURST     = 2'd0;
    localparam BURST_INCAR  = 2'd1;
    localparam BURST_WRAP   = 2'd2;

    localparam IDLE    = 3'b000;
    localparam RADDR   = 3'b001;
    localparam RDATA   = 3'b010;
    localparam WADDR   = 3'b011;
    localparam WDATA   = 3'b100;
    localparam WRESP   = 3'b101;

    localparam RESP_OK = 2'b00;

    logic [15:0] len;
    logic [15:0] rsp_cnt, req_cnt;
    logic [3:0]  arid;
    logic [3:0]  awid;
    logic [1:0]  burst;

    logic  [2:0]                current_state;
    logic  [2:0]                next_state;
    logic  [ADDR_WIDTH-1:0]     raddr_start;
    logic  [ADDR_WIDTH-1:0]     waddr_start;
    logic  [ADDR_WIDTH-1:0]     rd_addr;
    logic  [ADDR_WIDTH-1:0]     wr_addr;
    
    logic  [DRAM_ADDRWIDTH-1:0]     req_addr;
    logic  [DRAM_ADDRWIDTH-1:0]     addr_dram;
    logic  [DRAM_DATAWIDTH/8-1:0]   wstrb_dram;
    logic  [DRAM_DATAWIDTH-1:0]     wdata_dram;
    logic  [DRAM_MASK_LEN-1:0]      rd_mask;
    logic  [DRAM_MASK_LEN-1:0]      wr_mask;

    logic is_last_line;
        
    always_comb begin
        case (current_state)
            RDATA : begin
                req_addr = rd_addr;
            end
            WDATA : begin
                req_addr = wr_addr;
            end 
            default: begin
                req_addr = 0;
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if(!rst_ni) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end

if (LINE_PER_DATA == 1) begin

    logic  [DRAM_ADDRWIDTH-1:0]     line_addr;

    assign line_addr = req_addr[DRAM_ADDRWIDTH-1:DRAM_MASK_LEN] << DRAM_MASK_LEN;

    always_comb begin
        if (current_state == RDATA) begin
            case (burst)
                BURST_INCAR : begin
                    rd_mask = raddr_start[DRAM_MASK_LEN-1:0] + rsp_cnt * DATA_BYTE;
                end
                default: begin
                    rd_mask = raddr_start[DRAM_MASK_LEN-1:0];
                end
            endcase
        end
        else begin
            rd_mask = 0;
        end
    end

    assign wr_mask = req_addr[DRAM_MASK_LEN-1:0];
    assign is_last_line = 1;
    assign addr_dram = line_addr;
    assign wdata_dram = axi_mem_wdata_i << (wr_mask*8);
    assign wstrb_dram = axi_mem_wstrb_i << wr_mask;

end

else begin
    
    logic [3:0] line_cnt;

    always_ff @( posedge clk_i or negedge rst_ni ) begin
        if (!rst_ni) begin
            line_cnt <= 'b0;
        end
        else if (b_valid_o & b_ready_i) begin
            line_cnt <= (line_cnt == LINE_PER_DATA - 1) ? 'b0 : (line_cnt + 1); 
        end
    end

    assign rd_mask = 0;
    assign wr_mask = 0;
    assign is_last_line = line_cnt == LINE_PER_DATA - 1;
    assign addr_dram = req_addr;
    assign wdata_dram = axi_mem_wdata_i;
    assign wstrb_dram = axi_mem_wstrb_i;
        
end

    // DRAM ports
    always_comb begin
        case (current_state)
            RDATA : begin
                req_valid_i = (len != req_cnt);
                we_i = 0;
            end
            WDATA : begin
                req_valid_i = axi_mem_wvalid_i;
                we_i = axi_mem_wvalid_i;
            end 
            default: begin
                req_valid_i = 0;
                we_i = 0;
            end
        endcase
    end
    assign addr_i = addr_dram;
    assign wdata_i = wdata_dram;
    assign wstrb_i = wstrb_dram;
    assign rsp_ready_i = axi_mem_rready_i;
    assign b_ready_i = axi_mem_bready_i | (~is_last_line);

    // AR
    assign axi_mem_arready_o = (current_state == RADDR) ? 1 : 0;
    
    // AW
    assign axi_mem_awready_o = (current_state == WADDR) ? 1 : 0;

    // R/W addr INCR
    always_comb begin
        case (burst)
            BURST_INCAR : begin
                rd_addr = raddr_start + req_cnt * DATA_BYTE;
                wr_addr = waddr_start + req_cnt * DATA_BYTE;
            end
            default: begin
                rd_addr = raddr_start;
                wr_addr = waddr_start;
            end
        endcase
    end

    // R
    assign axi_mem_rdata_o  = axi_mem_rvalid_o ? rdata_o >> (rd_mask*8) : 0;
    assign axi_mem_rresp_o  =  RESP_OK;
    assign axi_mem_rvalid_o =  rsp_valid_o;
    assign axi_mem_rlast_o  =  (rsp_cnt == len - 1) && rsp_valid_o && rsp_ready_i;
    assign axi_mem_rid_o    =  axi_mem_rvalid_o ? arid : 0; 

    // W
    assign axi_mem_wready_o = (req_cnt != len) & (current_state == WDATA) & req_ready_o;

    // B
    assign axi_mem_bvalid_o = (len - 1 == rsp_cnt) && b_valid_o && is_last_line;
    assign axi_mem_bid_o    = axi_mem_bvalid_o ? awid : 0;
    assign axi_mem_bresp_o  = RESP_OK;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (~rst_ni) begin
            rsp_cnt <= 0;
            req_cnt <= 0;
        end
        else begin
            if (current_state == IDLE) begin
                req_cnt <= 0;
            end
            else if(req_valid_i && req_ready_o) begin
                req_cnt <= req_cnt + 1;
            end
            case (current_state)
                RDATA : begin
                    if(rsp_valid_o && rsp_ready_i) begin
                        rsp_cnt <= rsp_cnt + 1;
                    end
                end
                WDATA : begin   
                    if(b_valid_o && b_ready_i && is_last_line) begin
                        rsp_cnt <= rsp_cnt + 1;
                    end                  
                end
                WRESP : begin   
                    if(b_valid_o && b_ready_i && is_last_line) begin
                        rsp_cnt <= rsp_cnt + 1;
                    end                  
                end
                default: begin 
                    rsp_cnt <= 0;
                end
            endcase 
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : CHANNEL_SAMPLE
        if(!rst_ni) begin
            waddr_start <= 'b0;
            raddr_start <= 'b0;
            burst <= 'b0;
            len   <= 'b0;
            arid  <= 'b0;
            awid  <= 'b0; 
        end else begin
            case (current_state) 
                RADDR : begin
                    raddr_start <= axi_mem_araddr_i - ADDR_BASE; // byte
                    len   <= (axi_mem_arburst_i == 2'b00) ? 1 : axi_mem_arlen_i;
                    burst <= axi_mem_arburst_i;
                    arid  <= axi_mem_arid_i;
                end 
                WADDR : begin
                    waddr_start <= axi_mem_awaddr_i - ADDR_BASE;
                    len   <= (axi_mem_awburst_i == 2'b00) ? 1 : axi_mem_awlen_i;
                    burst <= axi_mem_awburst_i; 
                    awid  <= axi_mem_awid_i;
                end
                default : begin
                end
            endcase
        end  
    end
    
    logic rd_grant, wr_grant;
    logic [1:0] reqs;

    assign reqs = (current_state == IDLE && req_ready_o) ? {axi_mem_arvalid_i, axi_mem_awvalid_i} : 2'b00;

    arb_round_robin #(
        .width_p  (2)
    ) arb_rr (
        .clk_i    ( clk_i                   ),
        .rst_ni   ( rst_ni                  ),

        .reqs_i   ( reqs                    ),
        .grants_o ( {rd_grant, wr_grant}    ),
        .read_i   ( |reqs                   )
    );

    always_comb begin
        case (current_state)
            IDLE : next_state = rd_grant ? RADDR : wr_grant ? WADDR : IDLE;
            RADDR : if (axi_mem_arvalid_i && axi_mem_arready_o) next_state = RDATA;
            RDATA : if (axi_mem_rlast_o) next_state = IDLE;
            WADDR : if (axi_mem_awvalid_i && axi_mem_awready_o) next_state = WDATA; 
            // WDATA : if (axi_mem_bvalid_o  && axi_mem_bready_i) next_state = IDLE; 
            WDATA : if (axi_mem_wlast_i & axi_mem_wvalid_i & axi_mem_wready_o) next_state = WRESP; 
            WRESP : if (axi_mem_bvalid_o  && axi_mem_bready_i) next_state = IDLE; 
            default: next_state = IDLE;
        endcase   
    end
    
endmodule
