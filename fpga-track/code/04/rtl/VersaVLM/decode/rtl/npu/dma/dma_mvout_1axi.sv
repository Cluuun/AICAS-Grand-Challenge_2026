`ifndef DMA_MVOUT_1AXI_V
`define DMA_MVOUT_1AXI_V

module DMA_MVOUT_1AXI
import npu_config_pkg::*;
#(
    parameter int AXI_ID_WIDTH          = npu_config_pkg::AXI_ID_WIDTH,
    parameter int AXI_ADDR_WIDTH        = npu_config_pkg::AXI_ADDR_WIDTH,
    parameter int AXI_DATA_WIDTH        = npu_config_pkg::AXI_DATA_WIDTH,
    parameter int AXI_MAX_BURST_BEATS   = 16,
    parameter int SPM_BANKS             = npu_config_pkg::SPM_BANK_NUM,
    parameter int SPM_BANK_DATA_WIDTH   = npu_config_pkg::SPM_BANK_DATA_WIDTH,
    parameter int ACC_DATA_WIDTH        = npu_config_pkg::ACC_DATA_WIDTH,
    parameter int ACC_ADDR_WIDTH        = npu_config_pkg::ACC_ADDR_WIDTH
) (
    input  logic                                                     clk,
    input  logic                                                     rst_n,

    input  logic [AXI_ADDR_WIDTH-1:0]                               mvout_dram_base_addr,
    input  logic [ACC_ADDR_WIDTH-1:0]                               mvout_acc_addr,
    input  logic [15:0]                                             mvout_row_num,
    input  logic                                                    dma_mvout_req_en,
    output logic                                                    dma_mvout_resp_done,
    output logic                                                    dma_mvout_busy,

    output logic                                                    acc_rd_en,
    output logic [ACC_ADDR_WIDTH-1:0]                               acc_rd_addr,
    input  logic [ACC_DATA_WIDTH-1:0]                               acc_rd_data,
    input  logic                                                    acc_rd_valid,

    output logic [AXI_ID_WIDTH-1:0]                                 m_axi_awid,
    output logic [AXI_ADDR_WIDTH-1:0]                               m_axi_awaddr,
    output logic [7:0]                                              m_axi_awlen,
    output logic [2:0]                                              m_axi_awsize,
    output logic [1:0]                                              m_axi_awburst,
    output logic                                                    m_axi_awvalid,
    input  logic                                                    m_axi_awready,

    output logic [AXI_DATA_WIDTH-1:0]                               m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0]                             m_axi_wstrb,
    output logic                                                    m_axi_wlast,
    output logic                                                    m_axi_wvalid,
    input  logic                                                    m_axi_wready,

    input  logic [AXI_ID_WIDTH-1:0]                                 m_axi_bid,
    input  logic [1:0]                                              m_axi_bresp,
    input  logic                                                    m_axi_bvalid,
    output logic                                                    m_axi_bready
);

localparam int AXI_STRB_WIDTH      = AXI_DATA_WIDTH / 8;
localparam int BURST_ROWS_MAX      = AXI_MAX_BURST_BEATS / SPM_BANKS;
localparam int BURST_ROW_IDX_WIDTH = (BURST_ROWS_MAX <= 1) ? 1 : $clog2(BURST_ROWS_MAX);
localparam int BANK_IDX_WIDTH      = (SPM_BANKS <= 1) ? 1 : $clog2(SPM_BANKS);
localparam int ROW_DATA_WIDTH      = SPM_BANKS * AXI_DATA_WIDTH;

typedef enum logic [2:0] {
    ST_IDLE         = 3'd0,
    ST_LOAD_REQ     = 3'd1,
    ST_LOAD_CAPTURE = 3'd2,
    ST_SEND_AW      = 3'd3,
    ST_WRITE        = 3'd4,
    ST_WAIT_B       = 3'd5
} mvout_state_t;

mvout_state_t                                   state_q;
logic [AXI_ADDR_WIDTH-1:0]                      dram_addr_q;
logic [ACC_ADDR_WIDTH-1:0]                      acc_addr_q;
logic [15:0]                                    remaining_rows_q;
logic [15:0]                                    burst_rows_q;
logic [BURST_ROW_IDX_WIDTH-1:0]                 load_row_idx_q;
logic [BANK_IDX_WIDTH-1:0]                      acc_load_bank_idx_q;
logic [BURST_ROW_IDX_WIDTH-1:0]                 write_row_idx_q;
logic [BANK_IDX_WIDTH-1:0]                      write_bank_idx_q;
logic [BURST_ROWS_MAX-1:0][SPM_BANKS-1:0][AXI_DATA_WIDTH-1:0] burst_buf_q;

function automatic [15:0] clamp_burst_rows(input logic [15:0] row_num);
    if (row_num > BURST_ROWS_MAX[15:0]) begin
        clamp_burst_rows = BURST_ROWS_MAX[15:0];
    end else begin
        clamp_burst_rows = row_num;
    end
endfunction

always_comb begin
    acc_rd_en    = 1'b0;
    acc_rd_addr  = acc_addr_q + (load_row_idx_q * ROW_DATA_WIDTH / 8) + (acc_load_bank_idx_q * AXI_STRB_WIDTH);
    m_axi_awid    = '0;
    m_axi_awaddr  = dram_addr_q;
    m_axi_awlen   = (burst_rows_q * SPM_BANKS) - 1'b1;
    m_axi_awsize  = $clog2(AXI_STRB_WIDTH);
    m_axi_awburst = 2'b01;
    m_axi_awvalid = 1'b0;
    m_axi_wdata   = burst_buf_q[write_row_idx_q][write_bank_idx_q];
    m_axi_wstrb   = '1;
    m_axi_wlast   = (write_row_idx_q == (burst_rows_q - 1'b1)) && (write_bank_idx_q == (SPM_BANKS - 1));
    m_axi_wvalid  = 1'b0;
    m_axi_bready  = 1'b0;

    case (state_q)
        ST_LOAD_REQ: begin
            acc_rd_en = 1'b1;
        end

        ST_SEND_AW: begin
            m_axi_awvalid = 1'b1;
        end

        ST_WRITE: begin
            m_axi_wvalid = 1'b1;
        end

        ST_WAIT_B: begin
            m_axi_bready = 1'b1;
        end

        default: begin
        end
    endcase
end

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_q <= ST_IDLE;
        dram_addr_q <= '0;
        acc_addr_q <= '0;
        remaining_rows_q <= '0;
        burst_rows_q <= '0;
        load_row_idx_q <= '0;
        acc_load_bank_idx_q <= '0;
        write_row_idx_q <= '0;
        write_bank_idx_q <= '0;
        dma_mvout_resp_done <= 1'b0;
        dma_mvout_busy <= 1'b0;
        burst_buf_q <= '0;
    end else begin
        dma_mvout_resp_done <= 1'b0;

        case (state_q)
            ST_IDLE: begin
                dma_mvout_busy <= 1'b0;
                if (dma_mvout_req_en) begin
                    if (mvout_row_num == '0) begin
                        dma_mvout_resp_done <= 1'b1;
                    end else begin
`ifndef SYNTHESIS
                        if ((AXI_MAX_BURST_BEATS % SPM_BANKS) != 0) begin
                            $fatal(1, "DMA_MVOUT_1AXI expects AXI_MAX_BURST_BEATS to be a multiple of SPM_BANKS");
                        end
`endif
                        dram_addr_q <= mvout_dram_base_addr;
                        acc_addr_q <= mvout_acc_addr;
                        remaining_rows_q <= mvout_row_num;
                        burst_rows_q <= clamp_burst_rows(mvout_row_num);
                        load_row_idx_q <= '0;
                        acc_load_bank_idx_q <= '0;
                        write_row_idx_q <= '0;
                        write_bank_idx_q <= '0;
                        dma_mvout_busy <= 1'b1;
                        state_q <= ST_LOAD_REQ;
                    end
                end
            end

            ST_LOAD_REQ: begin
                state_q <= ST_LOAD_CAPTURE;
            end

            ST_LOAD_CAPTURE: begin
                if (acc_rd_valid) begin
                    burst_buf_q[load_row_idx_q][acc_load_bank_idx_q] <= acc_rd_data[AXI_DATA_WIDTH-1:0];
                    if (acc_load_bank_idx_q == (SPM_BANKS - 1)) begin
                        acc_load_bank_idx_q <= '0;
                        if ((load_row_idx_q + 1'b1) == burst_rows_q[BURST_ROW_IDX_WIDTH-1:0]) begin
                            load_row_idx_q <= '0;
                            state_q <= ST_SEND_AW;
                        end else begin
                            load_row_idx_q <= load_row_idx_q + 1'b1;
                            state_q <= ST_LOAD_REQ;
                        end
                    end else begin
                        acc_load_bank_idx_q <= acc_load_bank_idx_q + 1'b1;
                        state_q <= ST_LOAD_REQ;
                    end
                end
            end

            ST_SEND_AW: begin
                if (m_axi_awready) begin
                    write_row_idx_q <= '0;
                    write_bank_idx_q <= '0;
                    state_q <= ST_WRITE;
                end
            end

            ST_WRITE: begin
                if (m_axi_wready) begin
                    if (write_bank_idx_q == (SPM_BANKS - 1)) begin
                        write_bank_idx_q <= '0;
                        if (write_row_idx_q == (burst_rows_q - 1'b1)) begin
                            write_row_idx_q <= '0;
                            state_q <= ST_WAIT_B;
                        end else begin
                            write_row_idx_q <= write_row_idx_q + 1'b1;
                        end
                    end else begin
                        write_bank_idx_q <= write_bank_idx_q + 1'b1;
                    end
                end
            end

            ST_WAIT_B: begin
                if (m_axi_bvalid) begin
                    if (remaining_rows_q == burst_rows_q) begin
                        dma_mvout_busy <= 1'b0;
                        dma_mvout_resp_done <= 1'b1;
                        state_q <= ST_IDLE;
                    end else begin
                        dram_addr_q <= dram_addr_q + (burst_rows_q * SPM_BANKS * AXI_STRB_WIDTH);
                        acc_addr_q <= acc_addr_q + (burst_rows_q * ROW_DATA_WIDTH / 8);
                        remaining_rows_q <= remaining_rows_q - burst_rows_q;
                        burst_rows_q <= clamp_burst_rows(remaining_rows_q - burst_rows_q);
                        load_row_idx_q <= '0;
                        acc_load_bank_idx_q <= '0;
                        write_row_idx_q <= '0;
                        write_bank_idx_q <= '0;
                        state_q <= ST_LOAD_REQ;
                    end
                end
            end

            default: begin
                state_q <= ST_IDLE;
            end
        endcase
    end
end

`ifndef SYNTHESIS
initial begin
    if (SPM_BANKS != 4) begin
        $error("DMA_MVOUT_1AXI expects SPM_BANKS == 4");
    end
    if (AXI_DATA_WIDTH != SPM_BANK_DATA_WIDTH) begin
        $error("DMA_MVOUT_1AXI expects AXI_DATA_WIDTH == SPM_BANK_DATA_WIDTH");
    end
end
`endif

endmodule

`endif
