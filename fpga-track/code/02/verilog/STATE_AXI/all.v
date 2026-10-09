/* verilator lint_off PINMISSING */
/* verilator lint_off CASEINCOMPLETE */
/* verilator lint_off COMBDLY */
/* verilator lint_off CASEX */
/* verilator lint_off CASEOVERLAP */
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1ns/1ps

module STATE_AXI_regslice_both
#(parameter 
    DataWidth=32
)(
    input ap_clk ,
    input ap_rst,

    input [DataWidth-1:0] data_in , 
    input vld_in , 
    output ack_in ,
    output [DataWidth-1:0] data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);
 

reg   [1:0] B_V_data_1_state;
wire   [DataWidth-1:0] B_V_data_1_data_in;
reg   [DataWidth-1:0] B_V_data_1_data_out;
wire    B_V_data_1_vld_reg;
wire    B_V_data_1_vld_in;
wire    B_V_data_1_vld_out;
reg   [DataWidth-1:0] B_V_data_1_payload_A;
reg   [DataWidth-1:0] B_V_data_1_payload_B;
reg    B_V_data_1_sel_rd;
reg    B_V_data_1_sel_wr;
wire    B_V_data_1_sel;
wire    B_V_data_1_load_A;
wire    B_V_data_1_load_B;
wire    B_V_data_1_state_cmp_full;
wire    B_V_data_1_ack_in;
wire    B_V_data_1_ack_out;

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        B_V_data_1_sel_rd <= 1'b0;
    end else begin
        if (((1'b1 == B_V_data_1_vld_out) & (1'b1 == B_V_data_1_ack_out))) begin
            B_V_data_1_sel_rd <= ~B_V_data_1_sel_rd;
        end else begin
            B_V_data_1_sel_rd <= B_V_data_1_sel_rd;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        B_V_data_1_sel_wr <= 1'b0;
    end else begin
        if (((1'b1 == B_V_data_1_vld_in) & (1'b1 == B_V_data_1_ack_in))) begin
            B_V_data_1_sel_wr <= ~B_V_data_1_sel_wr;
        end else begin
            B_V_data_1_sel_wr <= B_V_data_1_sel_wr;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        B_V_data_1_state <= 2'd0;
    end else begin
        if ((((2'd3 == B_V_data_1_state) & (1'b0 == B_V_data_1_vld_in) & (1'b1 == B_V_data_1_ack_out)) | ((2'd2 == B_V_data_1_state) & (1'b0 == B_V_data_1_vld_in)))) begin
            B_V_data_1_state <= 2'd2;
        end else if ((((2'd1 == B_V_data_1_state) & (1'b0 == B_V_data_1_ack_out)) | ((2'd3 == B_V_data_1_state) & (1'b0 == B_V_data_1_ack_out) & (1'b1 == B_V_data_1_vld_in)))) begin
            B_V_data_1_state <= 2'd1;
        end else if ((((2'd1 == B_V_data_1_state) & (1'b1 == B_V_data_1_ack_out)) | (~((1'b0 == B_V_data_1_ack_out) & (1'b1 == B_V_data_1_vld_in)) & ~((1'b0 == B_V_data_1_vld_in) & (1'b1 == B_V_data_1_ack_out)) & (2'd3 == B_V_data_1_state)) | ((2'd2 == B_V_data_1_state) & (1'b1 == B_V_data_1_vld_in)))) begin
            B_V_data_1_state <= 2'd3;
        end else begin
            B_V_data_1_state <= 2'd2;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == B_V_data_1_load_A)) begin
        B_V_data_1_payload_A <= B_V_data_1_data_in;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == B_V_data_1_load_B)) begin
        B_V_data_1_payload_B <= B_V_data_1_data_in;
    end
end

always @ (*) begin
    if ((1'b1 == B_V_data_1_sel)) begin
        B_V_data_1_data_out = B_V_data_1_payload_B;
    end else begin
        B_V_data_1_data_out = B_V_data_1_payload_A;
    end
end

assign B_V_data_1_ack_in = B_V_data_1_state[1'd1];
assign B_V_data_1_load_A = (~B_V_data_1_sel_wr & B_V_data_1_state_cmp_full);
assign B_V_data_1_load_B = (B_V_data_1_state_cmp_full & B_V_data_1_sel_wr);
assign B_V_data_1_sel = B_V_data_1_sel_rd;
assign B_V_data_1_state_cmp_full = ((B_V_data_1_state != 2'd1) ? 1'b1 : 1'b0);
assign B_V_data_1_vld_out = B_V_data_1_state[1'd0];

assign ack_in = B_V_data_1_ack_in;
assign B_V_data_1_data_in = data_in;
assign B_V_data_1_vld_in = vld_in;

assign vld_out = B_V_data_1_vld_out;
assign data_out = B_V_data_1_data_out;
assign B_V_data_1_ack_out = ack_out;

assign apdone_blk = ((B_V_data_1_state == 2'd3 && ack_out == 1'b0) | (B_V_data_1_state == 2'd1));

endmodule // both

module STATE_AXI_regslice_both_w1
#(parameter 
    DataWidth=1
)(
    input ap_clk ,
    input ap_rst,

    input data_in , 
    input vld_in , 
    output ack_in ,
    output data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);

reg     [1:0] B_V_data_1_state;
wire    B_V_data_1_data_in;
reg     B_V_data_1_data_out;
wire    B_V_data_1_vld_reg;
wire    B_V_data_1_vld_in;
wire    B_V_data_1_vld_out;
reg     B_V_data_1_payload_A;
reg     B_V_data_1_payload_B;
reg     B_V_data_1_sel_rd;
reg     B_V_data_1_sel_wr;
wire    B_V_data_1_sel;
wire    B_V_data_1_load_A;
wire    B_V_data_1_load_B;
wire    B_V_data_1_state_cmp_full;
wire    B_V_data_1_ack_in;
wire    B_V_data_1_ack_out;

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        B_V_data_1_sel_rd <= 1'b0;
    end else begin
        if (((1'b1 == B_V_data_1_vld_out) & (1'b1 == B_V_data_1_ack_out))) begin
            B_V_data_1_sel_rd <= ~B_V_data_1_sel_rd;
        end else begin
            B_V_data_1_sel_rd <= B_V_data_1_sel_rd;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        B_V_data_1_sel_wr <= 1'b0;
    end else begin
        if (((1'b1 == B_V_data_1_vld_in) & (1'b1 == B_V_data_1_ack_in))) begin
            B_V_data_1_sel_wr <= ~B_V_data_1_sel_wr;
        end else begin
            B_V_data_1_sel_wr <= B_V_data_1_sel_wr;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        B_V_data_1_state <= 2'd0;
    end else begin
        if ((((2'd3 == B_V_data_1_state) & (1'b0 == B_V_data_1_vld_in) & (1'b1 == B_V_data_1_ack_out)) | ((2'd2 == B_V_data_1_state) & (1'b0 == B_V_data_1_vld_in)))) begin
            B_V_data_1_state <= 2'd2;
        end else if ((((2'd1 == B_V_data_1_state) & (1'b0 == B_V_data_1_ack_out)) | ((2'd3 == B_V_data_1_state) & (1'b0 == B_V_data_1_ack_out) & (1'b1 == B_V_data_1_vld_in)))) begin
            B_V_data_1_state <= 2'd1;
        end else if ((((2'd1 == B_V_data_1_state) & (1'b1 == B_V_data_1_ack_out)) | (~((1'b0 == B_V_data_1_ack_out) & (1'b1 == B_V_data_1_vld_in)) & ~((1'b0 == B_V_data_1_vld_in) & (1'b1 == B_V_data_1_ack_out)) & (2'd3 == B_V_data_1_state)) | ((2'd2 == B_V_data_1_state) & (1'b1 == B_V_data_1_vld_in)))) begin
            B_V_data_1_state <= 2'd3;
        end else begin
            B_V_data_1_state <= 2'd2;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == B_V_data_1_load_A)) begin
        B_V_data_1_payload_A <= B_V_data_1_data_in;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == B_V_data_1_load_B)) begin
        B_V_data_1_payload_B <= B_V_data_1_data_in;
    end
end

always @ (*) begin
    if ((1'b1 == B_V_data_1_sel)) begin
        B_V_data_1_data_out = B_V_data_1_payload_B;
    end else begin
        B_V_data_1_data_out = B_V_data_1_payload_A;
    end
end

assign B_V_data_1_ack_in = B_V_data_1_state[1'd1];
assign B_V_data_1_load_A = (~B_V_data_1_sel_wr & B_V_data_1_state_cmp_full);
assign B_V_data_1_load_B = (B_V_data_1_state_cmp_full & B_V_data_1_sel_wr);
assign B_V_data_1_sel = B_V_data_1_sel_rd;
assign B_V_data_1_state_cmp_full = ((B_V_data_1_state != 2'd1) ? 1'b1 : 1'b0);
assign B_V_data_1_vld_out = B_V_data_1_state[1'd0];

assign ack_in = B_V_data_1_ack_in;
assign B_V_data_1_data_in = data_in;
assign B_V_data_1_vld_in = vld_in;

assign vld_out = B_V_data_1_vld_out;
assign data_out = B_V_data_1_data_out;
assign B_V_data_1_ack_out = ack_out;

assign apdone_blk = ((B_V_data_1_state == 2'd3 && ack_out == 1'b0) | (B_V_data_1_state == 2'd1));

endmodule // both


// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_llm_delta_order_indexed (
        ap_clk,
        ap_rst,
        ap_start,
        start_full_n,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        start_out,
        start_write,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state,
        x_stream_TDATA,
        x_stream_TVALID,
        x_stream_TREADY,
        tile_idx_stream_din,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_full_n,
        tile_idx_stream_write,
        memory_state_c_din,
        memory_state_c_num_data_valid,
        memory_state_c_fifo_cap,
        memory_state_c_full_n,
        memory_state_c_write
);

parameter    ap_ST_fsm_state1 = 3'd1;
parameter    ap_ST_fsm_state2 = 3'd2;
parameter    ap_ST_fsm_state3 = 3'd4;

input   ap_clk;
input   ap_rst;
input   ap_start;
input   start_full_n;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   start_out;
output   start_write;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;
input   x_stream_TREADY;
output  [31:0] tile_idx_stream_din;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_full_n;
output   tile_idx_stream_write;
output  [63:0] memory_state_c_din;
input  [2:0] memory_state_c_num_data_valid;
input  [2:0] memory_state_c_fifo_cap;
input   memory_state_c_full_n;
output   memory_state_c_write;

reg ap_done;
reg ap_idle;
reg start_write;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg tile_idx_stream_write;
reg memory_state_c_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    memory_state_c_blk_n;
reg    ap_block_state1;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_done;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_idle;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_ready;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWVALID;
wire   [63:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWADDR;
wire   [0:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWID;
wire   [31:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWLEN;
wire   [2:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWBURST;
wire   [1:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWPROT;
wire   [3:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWQOS;
wire   [3:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWREGION;
wire   [0:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWUSER;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WVALID;
wire   [127:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WDATA;
wire   [15:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WSTRB;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WLAST;
wire   [0:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WID;
wire   [0:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WUSER;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARVALID;
wire   [63:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARADDR;
wire   [0:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARID;
wire   [31:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARLEN;
wire   [2:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARBURST;
wire   [1:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARPROT;
wire   [3:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARQOS;
wire   [3:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARREGION;
wire   [0:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARUSER;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_RREADY;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_BREADY;
wire   [31:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_tile_idx_stream_din;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_tile_idx_stream_write;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TREADY;
wire   [199:0] grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TDATA;
wire    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TVALID;
reg    grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start_reg;
wire    ap_CS_fsm_state2;
wire    ap_CS_fsm_state3;
reg   [2:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 start_once_reg = 1'b0;
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 3'd1;
//#0 grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start_reg = 1'b0;
end

STATE_AXI_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2 grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start),
    .ap_done(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_done),
    .ap_idle(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_idle),
    .ap_ready(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_ready),
    .m_axi_gmem1_AWVALID(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(m_axi_gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(m_axi_gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(m_axi_gmem1_RDATA),
    .m_axi_gmem1_RLAST(m_axi_gmem1_RLAST),
    .m_axi_gmem1_RID(m_axi_gmem1_RID),
    .m_axi_gmem1_RFIFONUM(m_axi_gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(m_axi_gmem1_RUSER),
    .m_axi_gmem1_RRESP(m_axi_gmem1_RRESP),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .tile_idx_stream_din(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_tile_idx_stream_din),
    .tile_idx_stream_num_data_valid(7'd0),
    .tile_idx_stream_fifo_cap(7'd0),
    .tile_idx_stream_full_n(tile_idx_stream_full_n),
    .tile_idx_stream_write(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_tile_idx_stream_write),
    .x_stream_TREADY(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TREADY),
    .memory_state(memory_state),
    .x_stream_TDATA(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TDATA),
    .x_stream_TVALID(grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TVALID)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_done == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start_reg <= 1'b1;
        end else if ((grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_ready == 1'b1)) begin
            grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        start_once_reg <= 1'b0;
    end else begin
        if (((real_start == 1'b1) & (internal_ap_ready == 1'b0))) begin
            start_once_reg <= 1'b1;
        end else if ((internal_ap_ready == 1'b1)) begin
            start_once_reg <= 1'b0;
        end
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_done == 1'b1))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((real_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_done == 1'b1))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_ARVALID = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARVALID;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_RREADY = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_RREADY;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        memory_state_c_blk_n = memory_state_c_full_n;
    end else begin
        memory_state_c_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
        memory_state_c_write = 1'b1;
    end else begin
        memory_state_c_write = 1'b0;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (start_full_n == 1'b0))) begin
        real_start = 1'b0;
    end else begin
        real_start = ap_start;
    end
end

always @ (*) begin
    if (((real_start == 1'b1) & (start_once_reg == 1'b0))) begin
        start_write = 1'b1;
    end else begin
        start_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        tile_idx_stream_write = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_tile_idx_stream_write;
    end else begin
        tile_idx_stream_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (ap_done_reg == 1'b1) | (memory_state_c_full_n == 1'b0));
end

assign ap_ready = internal_ap_ready;

assign grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_ap_start_reg;

assign grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TREADY = (x_stream_TREADY & ap_CS_fsm_state3);

assign m_axi_gmem1_ARADDR = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARADDR;

assign m_axi_gmem1_ARBURST = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARBURST;

assign m_axi_gmem1_ARCACHE = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARCACHE;

assign m_axi_gmem1_ARID = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARID;

assign m_axi_gmem1_ARLEN = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARLEN;

assign m_axi_gmem1_ARLOCK = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARLOCK;

assign m_axi_gmem1_ARPROT = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARPROT;

assign m_axi_gmem1_ARQOS = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARQOS;

assign m_axi_gmem1_ARREGION = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARREGION;

assign m_axi_gmem1_ARSIZE = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARSIZE;

assign m_axi_gmem1_ARUSER = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_m_axi_gmem1_ARUSER;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign memory_state_c_din = memory_state;

assign start_out = real_start;

assign tile_idx_stream_din = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_tile_idx_stream_din;

assign x_stream_TDATA = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TDATA;

assign x_stream_TVALID = grp_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2_fu_64_x_stream_TVALID;

endmodule //STATE_AXI_replay_llm_delta_order_indexed
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        tile_idx_stream_din,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_full_n,
        tile_idx_stream_write,
        x_stream_TREADY,
        memory_state,
        x_stream_TDATA,
        x_stream_TVALID
);

parameter    ap_ST_iter0_fsm_state1 = 2'd1;
parameter    ap_ST_iter0_fsm_state2 = 2'd2;
parameter    ap_ST_iter1_fsm_state3 = 3'd2;
parameter    ap_ST_iter1_fsm_state4 = 3'd4;
parameter    ap_ST_iter2_fsm_state5 = 3'd2;
parameter    ap_ST_iter2_fsm_state6 = 3'd4;
parameter    ap_ST_iter3_fsm_state7 = 3'd2;
parameter    ap_ST_iter3_fsm_state8 = 3'd4;
parameter    ap_ST_iter4_fsm_state9 = 3'd2;
parameter    ap_ST_iter4_fsm_state10 = 3'd4;
parameter    ap_ST_iter5_fsm_state11 = 3'd2;
parameter    ap_ST_iter5_fsm_state12 = 3'd4;
parameter    ap_ST_iter6_fsm_state13 = 3'd2;
parameter    ap_ST_iter6_fsm_state14 = 3'd4;
parameter    ap_ST_iter7_fsm_state15 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 3'd1;
parameter    ap_ST_iter2_fsm_state0 = 3'd1;
parameter    ap_ST_iter3_fsm_state0 = 3'd1;
parameter    ap_ST_iter4_fsm_state0 = 3'd1;
parameter    ap_ST_iter5_fsm_state0 = 3'd1;
parameter    ap_ST_iter6_fsm_state0 = 3'd1;
parameter    ap_ST_iter7_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
output  [31:0] tile_idx_stream_din;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_full_n;
output   tile_idx_stream_write;
input   x_stream_TREADY;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;

reg ap_idle;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg tile_idx_stream_write;
reg x_stream_TVALID;

reg   [1:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [2:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [2:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [2:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
reg   [2:0] ap_CS_iter4_fsm;
wire    ap_CS_iter4_fsm_state0;
reg   [2:0] ap_CS_iter5_fsm;
wire    ap_CS_iter5_fsm_state0;
reg   [2:0] ap_CS_iter6_fsm;
wire    ap_CS_iter6_fsm_state0;
reg   [1:0] ap_CS_iter7_fsm;
wire    ap_CS_iter7_fsm_state0;
wire    ap_CS_iter0_fsm_state2;
wire    ap_CS_iter1_fsm_state3;
wire    ap_CS_iter1_fsm_state4;
reg   [0:0] icmp_ln84_reg_633;
reg   [0:0] icmp_ln84_reg_633_pp0_iter1_reg;
reg    ap_block_state5_io;
wire    ap_CS_iter2_fsm_state5;
wire    ap_CS_iter2_fsm_state6;
wire    ap_CS_iter3_fsm_state7;
wire    ap_CS_iter3_fsm_state8;
wire    ap_CS_iter4_fsm_state9;
wire    ap_CS_iter4_fsm_state10;
wire    ap_CS_iter5_fsm_state11;
wire    ap_CS_iter5_fsm_state12;
reg   [0:0] icmp_ln84_reg_633_pp0_iter5_reg;
reg    ap_block_state13_pp0_stage0_iter6;
wire    ap_CS_iter6_fsm_state13;
reg    ap_block_state14_pp0_stage1_iter6;
wire    ap_CS_iter6_fsm_state14;
reg   [0:0] icmp_ln84_reg_633_pp0_iter6_reg;
reg    ap_block_state15_pp0_stage0_iter7;
reg    ap_block_state15_io;
wire    ap_CS_iter7_fsm_state15;
reg    ap_condition_exit_pp0_iter0_stage1;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    tile_idx_stream_blk_n;
reg    gmem1_blk_n_AR;
reg    gmem1_blk_n_R;
reg    x_stream_TDATA_blk_n;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln84_fu_250_p2;
reg   [0:0] icmp_ln84_reg_633_pp0_iter0_reg;
reg   [0:0] icmp_ln84_reg_633_pp0_iter2_reg;
reg   [0:0] icmp_ln84_reg_633_pp0_iter3_reg;
reg   [0:0] icmp_ln84_reg_633_pp0_iter4_reg;
reg   [3:0] tp_load_reg_637;
wire   [0:0] icmp_ln85_fu_271_p2;
reg   [0:0] icmp_ln85_reg_642;
wire   [0:0] and_ln84_fu_289_p2;
reg   [0:0] and_ln84_reg_649;
wire   [7:0] select_ln84_1_fu_301_p3;
reg   [7:0] select_ln84_1_reg_655;
wire   [6:0] empty_fu_309_p1;
reg   [6:0] empty_reg_660;
wire   [10:0] add_ln85_1_fu_313_p2;
reg   [10:0] add_ln85_1_reg_667;
wire   [6:0] select_ln85_1_fu_351_p3;
reg   [6:0] select_ln85_1_reg_672;
reg   [6:0] select_ln85_1_reg_672_pp0_iter1_reg;
reg   [6:0] select_ln85_1_reg_672_pp0_iter2_reg;
reg   [6:0] select_ln85_1_reg_672_pp0_iter3_reg;
reg   [6:0] select_ln85_1_reg_672_pp0_iter4_reg;
reg   [6:0] select_ln85_1_reg_672_pp0_iter5_reg;
reg   [6:0] select_ln85_1_reg_672_pp0_iter6_reg;
wire   [11:0] sub_ln88_fu_425_p2;
reg   [11:0] sub_ln88_reg_679;
reg   [10:0] tmp_3_reg_684;
wire   [9:0] add_ln89_fu_441_p2;
reg   [9:0] add_ln89_reg_689;
reg   [9:0] add_ln89_reg_689_pp0_iter1_reg;
reg   [9:0] add_ln89_reg_689_pp0_iter2_reg;
reg   [9:0] add_ln89_reg_689_pp0_iter3_reg;
reg   [9:0] add_ln89_reg_689_pp0_iter4_reg;
reg   [9:0] add_ln89_reg_689_pp0_iter5_reg;
reg   [9:0] add_ln89_reg_689_pp0_iter6_reg;
wire   [18:0] add_ln88_fu_484_p2;
reg   [18:0] add_ln88_reg_695;
reg   [59:0] trunc_ln7_reg_700;
wire   [24:0] trunc_ln210_fu_530_p1;
reg   [24:0] trunc_ln210_reg_711;
reg   [24:0] trunc_ln210_reg_711_pp0_iter6_reg;
reg   [24:0] tmp_reg_716;
reg   [24:0] tmp_reg_716_pp0_iter6_reg;
reg   [24:0] tmp_7_reg_721;
reg   [24:0] tmp_7_reg_721_pp0_iter6_reg;
reg   [24:0] tmp_8_reg_726;
reg   [24:0] tmp_8_reg_726_pp0_iter6_reg;
wire   [24:0] trunc_ln91_fu_534_p1;
reg   [24:0] trunc_ln91_reg_731;
reg   [24:0] tmp_s_reg_736;
reg   [24:0] tmp_1_reg_741;
reg   [24:0] tmp_2_reg_746;
wire  signed [63:0] sext_ln206_fu_520_p1;
reg   [3:0] tp_fu_140;
wire   [3:0] add_ln86_fu_447_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_tp_load;
reg   [6:0] ct_fu_144;
reg   [10:0] indvar_flatten_fu_148;
wire   [10:0] select_ln85_2_fu_453_p3;
reg   [10:0] ap_sig_allocacmp_indvar_flatten_load;
reg   [7:0] tt_d_fu_152;
reg   [7:0] ap_sig_allocacmp_tt_d_load;
reg   [16:0] indvar_flatten12_fu_156;
wire   [16:0] add_ln84_fu_256_p2;
reg   [16:0] ap_sig_allocacmp_indvar_flatten12_load;
wire   [0:0] icmp_ln86_fu_283_p2;
wire   [0:0] xor_ln84_fu_277_p2;
wire   [7:0] add_ln84_1_fu_295_p2;
wire   [6:0] select_ln84_fu_327_p3;
wire   [0:0] or_ln85_fu_340_p2;
wire   [6:0] add_ln85_fu_334_p2;
wire   [17:0] p_shl_fu_358_p3;
wire   [15:0] p_shl1_fu_369_p3;
wire   [18:0] p_shl_cast_fu_365_p1;
wire   [18:0] p_shl1_cast_fu_376_p1;
wire   [3:0] select_ln85_fu_344_p3;
wire   [2:0] trunc_ln88_fu_397_p1;
wire   [10:0] shl_ln_fu_401_p3;
wire   [8:0] shl_ln88_1_fu_413_p3;
wire   [11:0] zext_ln88_fu_409_p1;
wire   [11:0] zext_ln88_1_fu_421_p1;
wire   [18:0] empty_44_fu_380_p2;
wire   [9:0] tmp_4_fu_386_p3;
wire   [9:0] zext_ln86_fu_393_p1;
wire   [18:0] or_ln_fu_476_p4;
wire  signed [18:0] sext_ln88_fu_473_p1;
wire   [22:0] tmp_5_fu_494_p3;
wire  signed [63:0] sext_ln88_1_fu_501_p1;
wire   [63:0] add_ln88_1_fu_505_p2;
wire   [16:0] shl_ln3_fu_541_p3;
wire   [14:0] shl_ln89_1_fu_552_p3;
wire   [17:0] zext_ln89_1_fu_548_p1;
wire   [17:0] zext_ln89_2_fu_559_p1;
wire   [17:0] sub_ln89_fu_563_p2;
wire   [17:0] zext_ln89_fu_538_p1;
wire   [17:0] add_ln90_fu_569_p2;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg    ap_loop_exit_ready_pp0_iter6_reg;
reg    ap_loop_exit_ready_pp0_iter7_reg;
reg   [1:0] ap_NS_iter0_fsm;
reg   [2:0] ap_NS_iter1_fsm;
reg   [2:0] ap_NS_iter2_fsm;
reg   [2:0] ap_NS_iter3_fsm;
reg   [2:0] ap_NS_iter4_fsm;
reg   [2:0] ap_NS_iter5_fsm;
reg   [2:0] ap_NS_iter6_fsm;
reg   [1:0] ap_NS_iter7_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter1_fsm_state3_blk;
wire    ap_ST_iter1_fsm_state4_blk;
reg    ap_ST_iter2_fsm_state5_blk;
wire    ap_ST_iter2_fsm_state6_blk;
wire    ap_ST_iter3_fsm_state7_blk;
wire    ap_ST_iter3_fsm_state8_blk;
wire    ap_ST_iter4_fsm_state9_blk;
wire    ap_ST_iter4_fsm_state10_blk;
wire    ap_ST_iter5_fsm_state11_blk;
wire    ap_ST_iter5_fsm_state12_blk;
reg    ap_ST_iter6_fsm_state13_blk;
reg    ap_ST_iter6_fsm_state14_blk;
reg    ap_ST_iter7_fsm_state15_blk;
wire    ap_start_int;
reg    ap_condition_284;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 2'd1;
//#0 ap_CS_iter1_fsm = 3'd1;
//#0 ap_CS_iter2_fsm = 3'd1;
//#0 ap_CS_iter3_fsm = 3'd1;
//#0 ap_CS_iter4_fsm = 3'd1;
//#0 ap_CS_iter5_fsm = 3'd1;
//#0 ap_CS_iter6_fsm = 3'd1;
//#0 ap_CS_iter7_fsm = 2'd1;
//#0 tp_fu_140 = 4'd0;
//#0 ct_fu_144 = 7'd0;
//#0 indvar_flatten_fu_148 = 11'd0;
//#0 tt_d_fu_152 = 8'd0;
//#0 indvar_flatten12_fu_156 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage1),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter3_fsm <= ap_ST_iter3_fsm_state0;
    end else begin
        ap_CS_iter3_fsm <= ap_NS_iter3_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter4_fsm <= ap_ST_iter4_fsm_state0;
    end else begin
        ap_CS_iter4_fsm <= ap_NS_iter4_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter5_fsm <= ap_ST_iter5_fsm_state0;
    end else begin
        ap_CS_iter5_fsm <= ap_NS_iter5_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter6_fsm <= ap_ST_iter6_fsm_state0;
    end else begin
        ap_CS_iter6_fsm <= ap_NS_iter6_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter7_fsm <= ap_ST_iter7_fsm_state0;
    end else begin
        ap_CS_iter7_fsm <= ap_NS_iter7_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (ap_loop_exit_ready_pp0_iter6_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ct_fu_144 <= 7'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3) & (icmp_ln84_reg_633_pp0_iter0_reg == 1'd0))) begin
        ct_fu_144 <= select_ln85_1_reg_672;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_284)) begin
        if ((icmp_ln84_fu_250_p2 == 1'd0)) begin
            indvar_flatten12_fu_156 <= add_ln84_fu_256_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten12_fu_156 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        indvar_flatten_fu_148 <= 11'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln84_reg_633 == 1'd0))) begin
        indvar_flatten_fu_148 <= select_ln85_2_fu_453_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        tp_fu_140 <= 4'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln84_reg_633 == 1'd0))) begin
        tp_fu_140 <= add_ln86_fu_447_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        tt_d_fu_152 <= 8'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln84_reg_633 == 1'd0))) begin
        tt_d_fu_152 <= select_ln84_1_reg_655;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        add_ln85_1_reg_667 <= add_ln85_1_fu_313_p2;
        and_ln84_reg_649 <= and_ln84_fu_289_p2;
        empty_reg_660 <= empty_fu_309_p1;
        icmp_ln84_reg_633 <= icmp_ln84_fu_250_p2;
        icmp_ln85_reg_642 <= icmp_ln85_fu_271_p2;
        select_ln84_1_reg_655 <= select_ln84_1_fu_301_p3;
        tp_load_reg_637 <= ap_sig_allocacmp_tp_load;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
        add_ln88_reg_695[18 : 1] <= add_ln88_fu_484_p2[18 : 1];
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        add_ln89_reg_689 <= add_ln89_fu_441_p2;
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln84_reg_633_pp0_iter0_reg <= icmp_ln84_reg_633;
        select_ln85_1_reg_672 <= select_ln85_1_fu_351_p3;
        sub_ln88_reg_679[11 : 6] <= sub_ln88_fu_425_p2[11 : 6];
        tmp_3_reg_684 <= {{empty_44_fu_380_p2[18:8]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
        add_ln89_reg_689_pp0_iter1_reg <= add_ln89_reg_689;
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln84_reg_633_pp0_iter1_reg <= icmp_ln84_reg_633_pp0_iter0_reg;
        select_ln85_1_reg_672_pp0_iter1_reg <= select_ln85_1_reg_672;
        trunc_ln7_reg_700 <= {{add_ln88_1_fu_505_p2[63:4]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
        add_ln89_reg_689_pp0_iter2_reg <= add_ln89_reg_689_pp0_iter1_reg;
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln84_reg_633_pp0_iter2_reg <= icmp_ln84_reg_633_pp0_iter1_reg;
        select_ln85_1_reg_672_pp0_iter2_reg <= select_ln85_1_reg_672_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
        add_ln89_reg_689_pp0_iter3_reg <= add_ln89_reg_689_pp0_iter2_reg;
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln84_reg_633_pp0_iter3_reg <= icmp_ln84_reg_633_pp0_iter2_reg;
        select_ln85_1_reg_672_pp0_iter3_reg <= select_ln85_1_reg_672_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        add_ln89_reg_689_pp0_iter4_reg <= add_ln89_reg_689_pp0_iter3_reg;
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        icmp_ln84_reg_633_pp0_iter4_reg <= icmp_ln84_reg_633_pp0_iter3_reg;
        select_ln85_1_reg_672_pp0_iter4_reg <= select_ln85_1_reg_672_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
        add_ln89_reg_689_pp0_iter5_reg <= add_ln89_reg_689_pp0_iter4_reg;
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        icmp_ln84_reg_633_pp0_iter5_reg <= icmp_ln84_reg_633_pp0_iter4_reg;
        select_ln85_1_reg_672_pp0_iter5_reg <= select_ln85_1_reg_672_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
        add_ln89_reg_689_pp0_iter6_reg <= add_ln89_reg_689_pp0_iter5_reg;
        icmp_ln84_reg_633_pp0_iter6_reg <= icmp_ln84_reg_633_pp0_iter5_reg;
        select_ln85_1_reg_672_pp0_iter6_reg <= select_ln85_1_reg_672_pp0_iter5_reg;
        tmp_1_reg_741 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_2_reg_746 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_7_reg_721_pp0_iter6_reg <= tmp_7_reg_721;
        tmp_8_reg_726_pp0_iter6_reg <= tmp_8_reg_726;
        tmp_reg_716_pp0_iter6_reg <= tmp_reg_716;
        tmp_s_reg_736 <= {{m_axi_gmem1_RDATA[56:32]}};
        trunc_ln210_reg_711_pp0_iter6_reg <= trunc_ln210_reg_711;
        trunc_ln91_reg_731 <= trunc_ln91_fu_534_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state13_pp0_stage0_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state13))) begin
        tmp_7_reg_721 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_8_reg_726 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_reg_716 <= {{m_axi_gmem1_RDATA[56:32]}};
        trunc_ln210_reg_711 <= trunc_ln210_fu_530_p1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter0_fsm_state2_blk = 1'b0;

assign ap_ST_iter1_fsm_state3_blk = 1'b0;

assign ap_ST_iter1_fsm_state4_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state5_io)) begin
        ap_ST_iter2_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state5_blk = 1'b0;
    end
end

assign ap_ST_iter2_fsm_state6_blk = 1'b0;

assign ap_ST_iter3_fsm_state7_blk = 1'b0;

assign ap_ST_iter3_fsm_state8_blk = 1'b0;

assign ap_ST_iter4_fsm_state10_blk = 1'b0;

assign ap_ST_iter4_fsm_state9_blk = 1'b0;

assign ap_ST_iter5_fsm_state11_blk = 1'b0;

assign ap_ST_iter5_fsm_state12_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state13_pp0_stage0_iter6)) begin
        ap_ST_iter6_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_iter6_fsm_state13_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state14_pp0_stage1_iter6)) begin
        ap_ST_iter6_fsm_state14_blk = 1'b1;
    end else begin
        ap_ST_iter6_fsm_state14_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) begin
        ap_ST_iter7_fsm_state15_blk = 1'b1;
    end else begin
        ap_ST_iter7_fsm_state15_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln84_reg_633 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state0) & (1'b1 == ap_CS_iter6_fsm_state0) & (1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten12_load = 17'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten12_load = indvar_flatten12_fu_156;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 11'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_148;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_tp_load = 4'd0;
    end else begin
        ap_sig_allocacmp_tp_load = tp_fu_140;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_tt_d_load = 8'd0;
    end else begin
        ap_sig_allocacmp_tt_d_load = tt_d_fu_152;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln84_reg_633_pp0_iter1_reg == 1'd0))) begin
        gmem1_blk_n_AR = m_axi_gmem1_ARREADY;
    end else begin
        gmem1_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter6_fsm_state14) & (icmp_ln84_reg_633_pp0_iter5_reg == 1'd0)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (icmp_ln84_reg_633_pp0_iter5_reg == 1'd0)))) begin
        gmem1_blk_n_R = m_axi_gmem1_RVALID;
    end else begin
        gmem1_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln84_reg_633_pp0_iter1_reg == 1'd0))) begin
        m_axi_gmem1_ARVALID = 1'b1;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state13_pp0_stage0_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state13) & (icmp_ln84_reg_633_pp0_iter5_reg == 1'd0)) | (~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14) & (icmp_ln84_reg_633_pp0_iter5_reg == 1'd0)))) begin
        m_axi_gmem1_RREADY = 1'b1;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln84_reg_633_pp0_iter6_reg == 1'd0))) begin
        tile_idx_stream_blk_n = tile_idx_stream_full_n;
    end else begin
        tile_idx_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln84_reg_633_pp0_iter6_reg == 1'd0))) begin
        tile_idx_stream_write = 1'b1;
    end else begin
        tile_idx_stream_write = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln84_reg_633_pp0_iter6_reg == 1'd0))) begin
        x_stream_TDATA_blk_n = x_stream_TREADY;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln84_reg_633_pp0_iter6_reg == 1'd0))) begin
        x_stream_TVALID = 1'b1;
    end else begin
        x_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if (~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end
        end
        ap_ST_iter1_fsm_state4 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b0 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state5 : begin
            if ((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state5))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end
        end
        ap_ST_iter2_fsm_state6 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter3_fsm)
        ap_ST_iter3_fsm_state7 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state7))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end
        end
        ap_ST_iter3_fsm_state8 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter3_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter4_fsm)
        ap_ST_iter4_fsm_state9 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state9))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end
        end
        ap_ST_iter4_fsm_state10 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter4_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter5_fsm)
        ap_ST_iter5_fsm_state11 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter5_fsm_state11))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state12;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end
        end
        ap_ST_iter5_fsm_state12 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state12;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter5_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter6_fsm)
        ap_ST_iter6_fsm_state13 : begin
            if ((~((1'b1 == ap_block_state13_pp0_stage0_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state13))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state14;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end
        end
        ap_ST_iter6_fsm_state14 : begin
            if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end else if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b0 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state14;
            end
        end
        ap_ST_iter6_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter6_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter7_fsm)
        ap_ST_iter7_fsm_state15 : begin
            if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & ((1'b0 == ap_CS_iter6_fsm_state14) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6))))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end else if (((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln84_reg_633_pp0_iter6_reg == 1'd1)) | (~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter6_fsm_state14) & (1'b0 == ap_block_state14_pp0_stage1_iter6)))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state15;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state15;
            end
        end
        ap_ST_iter7_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state15;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter7_fsm = 'bx;
        end
    endcase
end

assign add_ln84_1_fu_295_p2 = (ap_sig_allocacmp_tt_d_load + 8'd1);

assign add_ln84_fu_256_p2 = (ap_sig_allocacmp_indvar_flatten12_load + 17'd1);

assign add_ln85_1_fu_313_p2 = (ap_sig_allocacmp_indvar_flatten_load + 11'd1);

assign add_ln85_fu_334_p2 = (select_ln84_fu_327_p3 + 7'd1);

assign add_ln86_fu_447_p2 = (select_ln85_fu_344_p3 + 4'd1);

assign add_ln88_1_fu_505_p2 = ($signed(sext_ln88_1_fu_501_p1) + $signed(memory_state));

assign add_ln88_fu_484_p2 = ($signed(or_ln_fu_476_p4) + $signed(sext_ln88_fu_473_p1));

assign add_ln89_fu_441_p2 = (tmp_4_fu_386_p3 + zext_ln86_fu_393_p1);

assign add_ln90_fu_569_p2 = (sub_ln89_fu_563_p2 + zext_ln89_fu_538_p1);

assign and_ln84_fu_289_p2 = (xor_ln84_fu_277_p2 & icmp_ln86_fu_283_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state3 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state4 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state5 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter2_fsm_state6 = ap_CS_iter2_fsm[32'd2];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state7 = ap_CS_iter3_fsm[32'd1];

assign ap_CS_iter3_fsm_state8 = ap_CS_iter3_fsm[32'd2];

assign ap_CS_iter4_fsm_state0 = ap_CS_iter4_fsm[32'd0];

assign ap_CS_iter4_fsm_state10 = ap_CS_iter4_fsm[32'd2];

assign ap_CS_iter4_fsm_state9 = ap_CS_iter4_fsm[32'd1];

assign ap_CS_iter5_fsm_state0 = ap_CS_iter5_fsm[32'd0];

assign ap_CS_iter5_fsm_state11 = ap_CS_iter5_fsm[32'd1];

assign ap_CS_iter5_fsm_state12 = ap_CS_iter5_fsm[32'd2];

assign ap_CS_iter6_fsm_state0 = ap_CS_iter6_fsm[32'd0];

assign ap_CS_iter6_fsm_state13 = ap_CS_iter6_fsm[32'd1];

assign ap_CS_iter6_fsm_state14 = ap_CS_iter6_fsm[32'd2];

assign ap_CS_iter7_fsm_state0 = ap_CS_iter7_fsm[32'd0];

assign ap_CS_iter7_fsm_state15 = ap_CS_iter7_fsm[32'd1];

always @ (*) begin
    ap_block_state13_pp0_stage0_iter6 = ((icmp_ln84_reg_633_pp0_iter5_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state14_pp0_stage1_iter6 = ((icmp_ln84_reg_633_pp0_iter5_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state15_io = ((x_stream_TREADY == 1'b0) & (icmp_ln84_reg_633_pp0_iter6_reg == 1'd0));
end

always @ (*) begin
    ap_block_state15_pp0_stage0_iter7 = (((x_stream_TREADY == 1'b0) & (icmp_ln84_reg_633_pp0_iter6_reg == 1'd0)) | ((icmp_ln84_reg_633_pp0_iter6_reg == 1'd0) & (tile_idx_stream_full_n == 1'b0)));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state5_io = ((icmp_ln84_reg_633_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_ARREADY == 1'b0));
end

always @ (*) begin
    ap_condition_284 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage1;

assign empty_44_fu_380_p2 = (p_shl_cast_fu_365_p1 - p_shl1_cast_fu_376_p1);

assign empty_fu_309_p1 = select_ln84_1_fu_301_p3[6:0];

assign icmp_ln84_fu_250_p2 = ((ap_sig_allocacmp_indvar_flatten12_load == 17'd98304) ? 1'b1 : 1'b0);

assign icmp_ln85_fu_271_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 11'd768) ? 1'b1 : 1'b0);

assign icmp_ln86_fu_283_p2 = ((ap_sig_allocacmp_tp_load == 4'd8) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = sext_ln206_fu_520_p1;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd2;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign or_ln85_fu_340_p2 = (icmp_ln85_reg_642 | and_ln84_reg_649);

assign or_ln_fu_476_p4 = {{{tmp_3_reg_684}, {select_ln85_1_reg_672}}, {1'd0}};

assign p_shl1_cast_fu_376_p1 = p_shl1_fu_369_p3;

assign p_shl1_fu_369_p3 = {{empty_reg_660}, {9'd0}};

assign p_shl_cast_fu_365_p1 = p_shl_fu_358_p3;

assign p_shl_fu_358_p3 = {{empty_reg_660}, {11'd0}};

assign select_ln84_1_fu_301_p3 = ((icmp_ln85_fu_271_p2[0:0] == 1'b1) ? add_ln84_1_fu_295_p2 : ap_sig_allocacmp_tt_d_load);

assign select_ln84_fu_327_p3 = ((icmp_ln85_reg_642[0:0] == 1'b1) ? 7'd0 : ct_fu_144);

assign select_ln85_1_fu_351_p3 = ((and_ln84_reg_649[0:0] == 1'b1) ? add_ln85_fu_334_p2 : select_ln84_fu_327_p3);

assign select_ln85_2_fu_453_p3 = ((icmp_ln85_reg_642[0:0] == 1'b1) ? 11'd1 : add_ln85_1_reg_667);

assign select_ln85_fu_344_p3 = ((or_ln85_fu_340_p2[0:0] == 1'b1) ? 4'd0 : tp_load_reg_637);

assign sext_ln206_fu_520_p1 = $signed(trunc_ln7_reg_700);

assign sext_ln88_1_fu_501_p1 = $signed(tmp_5_fu_494_p3);

assign sext_ln88_fu_473_p1 = $signed(sub_ln88_reg_679);

assign shl_ln3_fu_541_p3 = {{add_ln89_reg_689_pp0_iter6_reg}, {7'd0}};

assign shl_ln88_1_fu_413_p3 = {{trunc_ln88_fu_397_p1}, {6'd0}};

assign shl_ln89_1_fu_552_p3 = {{add_ln89_reg_689_pp0_iter6_reg}, {5'd0}};

assign shl_ln_fu_401_p3 = {{trunc_ln88_fu_397_p1}, {8'd0}};

assign sub_ln88_fu_425_p2 = (zext_ln88_fu_409_p1 - zext_ln88_1_fu_421_p1);

assign sub_ln89_fu_563_p2 = (zext_ln89_1_fu_548_p1 - zext_ln89_2_fu_559_p1);

assign tile_idx_stream_din = $signed(add_ln90_fu_569_p2);

assign tmp_4_fu_386_p3 = {{empty_reg_660}, {3'd0}};

assign tmp_5_fu_494_p3 = {{add_ln88_reg_695}, {4'd0}};

assign trunc_ln210_fu_530_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln88_fu_397_p1 = select_ln85_fu_344_p3[2:0];

assign trunc_ln91_fu_534_p1 = m_axi_gmem1_RDATA[24:0];

assign x_stream_TDATA = {{{{{{{{tmp_2_reg_746}, {tmp_1_reg_741}}, {tmp_s_reg_736}}, {trunc_ln91_reg_731}}, {tmp_8_reg_726_pp0_iter6_reg}}, {tmp_7_reg_721_pp0_iter6_reg}}, {tmp_reg_716_pp0_iter6_reg}}, {trunc_ln210_reg_711_pp0_iter6_reg}};

assign xor_ln84_fu_277_p2 = (icmp_ln85_fu_271_p2 ^ 1'd1);

assign zext_ln86_fu_393_p1 = select_ln85_fu_344_p3;

assign zext_ln88_1_fu_421_p1 = shl_ln88_1_fu_413_p3;

assign zext_ln88_fu_409_p1 = shl_ln_fu_401_p3;

assign zext_ln89_1_fu_548_p1 = shl_ln3_fu_541_p3;

assign zext_ln89_2_fu_559_p1 = shl_ln89_1_fu_552_p3;

assign zext_ln89_fu_538_p1 = select_ln85_1_reg_672_pp0_iter6_reg;

always @ (posedge ap_clk) begin
    sub_ln88_reg_679[5:0] <= 6'b000000;
    add_ln88_reg_695[0] <= 1'b0;
end

endmodule //STATE_AXI_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_writeback_indexed_order (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state_dout,
        memory_state_num_data_valid,
        memory_state_fifo_cap,
        memory_state_empty_n,
        memory_state_read,
        y_stream_TDATA,
        y_stream_TVALID,
        y_stream_TREADY,
        tile_idx_stream_dout,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_empty_n,
        tile_idx_stream_read
);

parameter    ap_ST_fsm_state1 = 3'd1;
parameter    ap_ST_fsm_state2 = 3'd2;
parameter    ap_ST_fsm_state3 = 3'd4;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state_dout;
input  [2:0] memory_state_num_data_valid;
input  [2:0] memory_state_fifo_cap;
input   memory_state_empty_n;
output   memory_state_read;
input  [199:0] y_stream_TDATA;
input   y_stream_TVALID;
output   y_stream_TREADY;
input  [31:0] tile_idx_stream_dout;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_empty_n;
output   tile_idx_stream_read;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg m_axi_gmem1_AWVALID;
reg m_axi_gmem1_WVALID;
reg m_axi_gmem1_BREADY;
reg memory_state_read;
reg y_stream_TREADY;
reg tile_idx_stream_read;

reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    memory_state_blk_n;
reg   [63:0] memory_state_1_reg_63;
reg    ap_block_state1;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_idle;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_ready;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_tile_idx_stream_read;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWVALID;
wire   [63:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWADDR;
wire   [0:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWID;
wire   [31:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLEN;
wire   [2:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWBURST;
wire   [1:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWPROT;
wire   [3:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWQOS;
wire   [3:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWREGION;
wire   [0:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWUSER;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WVALID;
wire   [127:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WDATA;
wire   [15:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WSTRB;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WLAST;
wire   [0:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WID;
wire   [0:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WUSER;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARVALID;
wire   [63:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARADDR;
wire   [0:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARID;
wire   [31:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLEN;
wire   [2:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARBURST;
wire   [1:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARPROT;
wire   [3:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARQOS;
wire   [3:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARREGION;
wire   [0:0] grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARUSER;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_RREADY;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_BREADY;
wire    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_y_stream_TREADY;
reg    grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg;
wire    ap_CS_fsm_state2;
wire    ap_CS_fsm_state3;
reg   [2:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 3'd1;
//#0 grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg = 1'b0;
end

STATE_AXI_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1 grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start),
    .ap_done(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done),
    .ap_idle(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_idle),
    .ap_ready(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_ready),
    .tile_idx_stream_dout(tile_idx_stream_dout),
    .tile_idx_stream_num_data_valid(7'd0),
    .tile_idx_stream_fifo_cap(7'd0),
    .tile_idx_stream_empty_n(tile_idx_stream_empty_n),
    .tile_idx_stream_read(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_tile_idx_stream_read),
    .y_stream_TVALID(y_stream_TVALID),
    .m_axi_gmem1_AWVALID(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(m_axi_gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(m_axi_gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(9'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(m_axi_gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(m_axi_gmem1_BRESP),
    .m_axi_gmem1_BID(m_axi_gmem1_BID),
    .m_axi_gmem1_BUSER(m_axi_gmem1_BUSER),
    .y_stream_TDATA(y_stream_TDATA),
    .y_stream_TREADY(grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_y_stream_TREADY),
    .memory_state_1(memory_state_1_reg_63)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg <= 1'b1;
        end else if ((grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_ready == 1'b1)) begin
            grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
        memory_state_1_reg_63 <= memory_state_dout;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWVALID = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWVALID;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_BREADY = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_BREADY;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_WVALID = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WVALID;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        memory_state_blk_n = memory_state_empty_n;
    end else begin
        memory_state_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
        memory_state_read = 1'b1;
    end else begin
        memory_state_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        tile_idx_stream_read = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_tile_idx_stream_read;
    end else begin
        tile_idx_stream_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        y_stream_TREADY = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_y_stream_TREADY;
    end else begin
        y_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

always @ (*) begin
    ap_block_state1 = ((ap_start == 1'b0) | (memory_state_empty_n == 1'b0) | (ap_done_reg == 1'b1));
end

assign grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg;

assign m_axi_gmem1_ARADDR = 64'd0;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd0;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_ARVALID = 1'b0;

assign m_axi_gmem1_AWADDR = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWADDR;

assign m_axi_gmem1_AWBURST = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWBURST;

assign m_axi_gmem1_AWCACHE = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWCACHE;

assign m_axi_gmem1_AWID = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWID;

assign m_axi_gmem1_AWLEN = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLEN;

assign m_axi_gmem1_AWLOCK = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLOCK;

assign m_axi_gmem1_AWPROT = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWPROT;

assign m_axi_gmem1_AWQOS = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWQOS;

assign m_axi_gmem1_AWREGION = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWREGION;

assign m_axi_gmem1_AWSIZE = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWSIZE;

assign m_axi_gmem1_AWUSER = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWUSER;

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WDATA = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WDATA;

assign m_axi_gmem1_WID = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WID;

assign m_axi_gmem1_WLAST = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WLAST;

assign m_axi_gmem1_WSTRB = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WSTRB;

assign m_axi_gmem1_WUSER = grp_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WUSER;

endmodule //STATE_AXI_writeback_indexed_order
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 2 

module STATE_AXI_fifo_w32_d64_A_x
#(parameter
    MEM_STYLE    = "auto",
    DATA_WIDTH   = 32,
    ADDR_WIDTH   = 6,
    DEPTH        = 63)
(
    // system signal
    input  wire                  clk,
    input  wire                  reset,

    // write
    output wire                  if_full_n,
    input  wire                  if_write_ce,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,
    
    // read 
    output wire [ADDR_WIDTH:0]   if_num_data_valid, // for FRP
    output wire [ADDR_WIDTH:0]   if_fifo_cap,       // for FRP

    output wire                  if_empty_n,
    input  wire                  if_read_ce,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout
);
//------------------------Parameter----------------------

//------------------------Local signal-------------------
    reg  [ADDR_WIDTH-1:0] waddr;
    reg  [ADDR_WIDTH-1:0] raddr;
    wire [ADDR_WIDTH-1:0] wnext;
    wire [ADDR_WIDTH-1:0] rnext;
    wire                  push;
    wire                  pop;
    reg  [ADDR_WIDTH:0]   mOutPtr;
    reg                   empty_n = 1'b0;
    reg                   full_n = 1'b1;
    // has num_data_valid? 
    wire                  num_extra_words;//yes
    reg  [ADDR_WIDTH:0]   num_data_valid; //yes 

    wire                  pop_dout;
    reg  [ADDR_WIDTH:0]   num_data_cnt;
    reg                   dout_vld = 1'b0;

//------------------------Instantiation------------------
    STATE_AXI_fifo_w32_d64_A_x_ram 
    #(  .MEM_STYLE  (MEM_STYLE),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH)
    ) U_STATE_AXI_fifo_w32_d64_A_x_ram (
        .clk        (clk),
        .reset      (reset),
        .we         (push),
        .waddr      (waddr),
        .din        (if_din),
        .raddr      (raddr),
        .rden       (pop),
        .dout       (if_dout)
    );

//------------------------Task and function--------------

//------------------------Body---------------------------
    // num_data_valid 
    assign if_num_data_valid = num_data_valid;
    assign if_fifo_cap = DEPTH + 1;

    // almost full/empty 

    // program full/empty 

    assign if_full_n  = full_n;
    assign if_empty_n = dout_vld;

    assign push       = full_n & if_write_ce & if_write;
    assign pop        = empty_n & (pop_dout | ~dout_vld);
    assign pop_dout   = dout_vld & if_read_ce & if_read;
    
    assign wnext      = !push                ? waddr :
                        (waddr == DEPTH - 1) ? 1'b0  :
                        waddr + 1'b1;
    assign rnext      = !pop                 ? raddr :
                        (raddr == DEPTH - 1) ? 1'b0  :
                        raddr + 1'b1;

    // waddr
    always @(posedge clk) begin
        if (reset)
            waddr <= {ADDR_WIDTH{1'b0}};
        else
            waddr <= wnext;
    end

    // raddr
    always @(posedge clk) begin
        if (reset)
            raddr <= {ADDR_WIDTH{1'b0}};
        else
            raddr <= rnext;
    end

    // mOutPtr
    always @(posedge clk) begin
        if (reset)
            mOutPtr <= {ADDR_WIDTH+1{1'b0}};
        else if (push & ~pop)
            mOutPtr <= mOutPtr + 1'b1;
        else if (~push & pop)
            mOutPtr <= mOutPtr - 1'b1;
    end

    // full_n 
    always @(posedge clk) begin
        if (reset)
            full_n <= 1'b1;
        else if ((push & ~pop_dout) && (num_data_cnt == DEPTH))
            full_n <= 1'b0;
        else if (~push & pop_dout)
            full_n <= 1'b1;
    end

    // empty_n
    always @(posedge clk) begin
        if (reset)
            empty_n <= 1'b0;
        else if (push & ~pop)
            empty_n <= 1'b1;
        else if ((~push & pop) && (mOutPtr == 1))
            empty_n <= 1'b0;
    end

    // almost_full_n 

    // almost_empty_n 

    // prog_full_n 

    // prog_empty_n 

    // num_data_cnt
    always @(posedge clk) begin
        if (reset)
            num_data_cnt <= {ADDR_WIDTH+1{1'b0}};
        else if ( push & ~pop_dout)
            num_data_cnt <= num_data_cnt + 1'b1;
        else if (~push & pop_dout)
            num_data_cnt <= num_data_cnt - 1'b1;
    end

    // num_data_valid 
    assign num_extra_words = (dout_vld & ~pop_dout) ? 1 : 0;
                             
    always @(posedge clk) begin
        if (reset)
            num_data_valid <= {ADDR_WIDTH+1{1'b0}};
        else if (empty_n | (dout_vld & ~pop_dout))
            num_data_valid <= push + mOutPtr + num_extra_words;
        else
            num_data_valid <= num_extra_words;
    end // 

    // dout_vld
    always @(posedge clk) begin
        if (reset)
            dout_vld <= 1'b0;
        else if (pop)
            dout_vld <= 1'b1;
        else if (pop_dout)
            dout_vld <= 1'b0;
    end

endmodule


module STATE_AXI_fifo_w32_d64_A_x_ram
#(parameter
    MEM_STYLE   = "auto",
    DATA_WIDTH  = 32,
    ADDR_WIDTH  = 6,
    DEPTH       = 63)
(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [DATA_WIDTH-1:0] din,
    input  wire [ADDR_WIDTH-1:0] raddr,
    input  wire                  rden,
    output wire [DATA_WIDTH-1:0] dout
);

    (* ram_style = MEM_STYLE *)
    reg  [DATA_WIDTH-1:0] mem[0:DEPTH-1];
    reg  [DATA_WIDTH-1:0] mem_reg;

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= din;
    end

    always @(posedge clk) begin
        if (reset)
            mem_reg <= 0;
        else if (rden)
            mem_reg <= mem[raddr];
    end
    
    assign dout = mem_reg;

endmodule// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 2 

module STATE_AXI_fifo_w32_d64_A
#(parameter
    MEM_STYLE    = "auto",
    DATA_WIDTH   = 32,
    ADDR_WIDTH   = 6,
    DEPTH        = 63)
(
    // system signal
    input  wire                  clk,
    input  wire                  reset,

    // write
    output wire                  if_full_n,
    input  wire                  if_write_ce,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,
    
    // read 
    output wire [ADDR_WIDTH:0]   if_num_data_valid, // for FRP
    output wire [ADDR_WIDTH:0]   if_fifo_cap,       // for FRP

    output wire                  if_empty_n,
    input  wire                  if_read_ce,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout
);
//------------------------Parameter----------------------

//------------------------Local signal-------------------
    reg  [ADDR_WIDTH-1:0] waddr;
    reg  [ADDR_WIDTH-1:0] raddr;
    wire [ADDR_WIDTH-1:0] wnext;
    wire [ADDR_WIDTH-1:0] rnext;
    wire                  push;
    wire                  pop;
    reg  [ADDR_WIDTH:0]   mOutPtr;
    reg                   empty_n = 1'b0;
    reg                   full_n = 1'b1;
    // has num_data_valid? 
    wire                  num_extra_words;//yes
    reg  [ADDR_WIDTH:0]   num_data_valid; //yes 

    wire                  pop_dout;
    reg  [ADDR_WIDTH:0]   num_data_cnt;
    reg                   dout_vld = 1'b0;

//------------------------Instantiation------------------
    STATE_AXI_fifo_w32_d64_A_ram 
    #(  .MEM_STYLE  (MEM_STYLE),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH)
    ) U_STATE_AXI_fifo_w32_d64_A_ram (
        .clk        (clk),
        .reset      (reset),
        .we         (push),
        .waddr      (waddr),
        .din        (if_din),
        .raddr      (raddr),
        .rden       (pop),
        .dout       (if_dout)
    );

//------------------------Task and function--------------

//------------------------Body---------------------------
    // num_data_valid 
    assign if_num_data_valid = num_data_valid;
    assign if_fifo_cap = DEPTH + 1;

    // almost full/empty 

    // program full/empty 

    assign if_full_n  = full_n;
    assign if_empty_n = dout_vld;

    assign push       = full_n & if_write_ce & if_write;
    assign pop        = empty_n & (pop_dout | ~dout_vld);
    assign pop_dout   = dout_vld & if_read_ce & if_read;
    
    assign wnext      = !push                ? waddr :
                        (waddr == DEPTH - 1) ? 1'b0  :
                        waddr + 1'b1;
    assign rnext      = !pop                 ? raddr :
                        (raddr == DEPTH - 1) ? 1'b0  :
                        raddr + 1'b1;

    // waddr
    always @(posedge clk) begin
        if (reset)
            waddr <= {ADDR_WIDTH{1'b0}};
        else
            waddr <= wnext;
    end

    // raddr
    always @(posedge clk) begin
        if (reset)
            raddr <= {ADDR_WIDTH{1'b0}};
        else
            raddr <= rnext;
    end

    // mOutPtr
    always @(posedge clk) begin
        if (reset)
            mOutPtr <= {ADDR_WIDTH+1{1'b0}};
        else if (push & ~pop)
            mOutPtr <= mOutPtr + 1'b1;
        else if (~push & pop)
            mOutPtr <= mOutPtr - 1'b1;
    end

    // full_n 
    always @(posedge clk) begin
        if (reset)
            full_n <= 1'b1;
        else if ((push & ~pop_dout) && (num_data_cnt == DEPTH))
            full_n <= 1'b0;
        else if (~push & pop_dout)
            full_n <= 1'b1;
    end

    // empty_n
    always @(posedge clk) begin
        if (reset)
            empty_n <= 1'b0;
        else if (push & ~pop)
            empty_n <= 1'b1;
        else if ((~push & pop) && (mOutPtr == 1))
            empty_n <= 1'b0;
    end

    // almost_full_n 

    // almost_empty_n 

    // prog_full_n 

    // prog_empty_n 

    // num_data_cnt
    always @(posedge clk) begin
        if (reset)
            num_data_cnt <= {ADDR_WIDTH+1{1'b0}};
        else if ( push & ~pop_dout)
            num_data_cnt <= num_data_cnt + 1'b1;
        else if (~push & pop_dout)
            num_data_cnt <= num_data_cnt - 1'b1;
    end

    // num_data_valid 
    assign num_extra_words = (dout_vld & ~pop_dout) ? 1 : 0;
                             
    always @(posedge clk) begin
        if (reset)
            num_data_valid <= {ADDR_WIDTH+1{1'b0}};
        else if (empty_n | (dout_vld & ~pop_dout))
            num_data_valid <= push + mOutPtr + num_extra_words;
        else
            num_data_valid <= num_extra_words;
    end // 

    // dout_vld
    always @(posedge clk) begin
        if (reset)
            dout_vld <= 1'b0;
        else if (pop)
            dout_vld <= 1'b1;
        else if (pop_dout)
            dout_vld <= 1'b0;
    end

endmodule


module STATE_AXI_fifo_w32_d64_A_ram
#(parameter
    MEM_STYLE   = "auto",
    DATA_WIDTH  = 32,
    ADDR_WIDTH  = 6,
    DEPTH       = 63)
(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [DATA_WIDTH-1:0] din,
    input  wire [ADDR_WIDTH-1:0] raddr,
    input  wire                  rden,
    output wire [DATA_WIDTH-1:0] dout
);

    (* ram_style = MEM_STYLE *)
    reg  [DATA_WIDTH-1:0] mem[0:DEPTH-1];
    reg  [DATA_WIDTH-1:0] mem_reg;

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= din;
    end

    always @(posedge clk) begin
        if (reset)
            mem_reg <= 0;
        else if (rden)
            mem_reg <= mem[raddr];
    end
    
    assign dout = mem_reg;

endmodule// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module STATE_AXI_start_for_writeback_indexed_order_U0
#(parameter
    MEM_STYLE    = "shiftReg",
    DATA_WIDTH   = 1,
    ADDR_WIDTH   = 1,
    DEPTH        = 2)
(
    // system signal
    input  wire                  clk,
    input  wire                  reset,

    // write
    output wire                  if_full_n,
    input  wire                  if_write_ce,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,
    
    // read 

    output wire                  if_empty_n,
    input  wire                  if_read_ce,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout
);
//------------------------Parameter----------------------

//------------------------Local signal-------------------
    wire [ADDR_WIDTH-1:0] addr;
    wire                  push;
    wire                  pop;
    reg signed [ADDR_WIDTH:0] mOutPtr;
    reg                   empty_n = 1'b0;
    reg                   full_n = 1'b1; 
    // has num_data_valid?  no 
//------------------------Instantiation------------------
    STATE_AXI_start_for_writeback_indexed_order_U0_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_STATE_AXI_start_for_writeback_indexed_order_U0_ShiftReg (
        .clk        (clk),
        .we         (push),
        .addr       (addr),
        .din        (if_din),
        .dout       (if_dout)
    );
//------------------------Task and function--------------

//------------------------Body---------------------------
    // num_data_valid 

    // almost full/empty 

    // program full/empty 

    assign if_full_n  = full_n; 
    assign if_empty_n = empty_n;

    assign push       = full_n & if_write_ce & if_write;
    assign pop        = empty_n & if_read_ce & if_read;

    assign addr       = mOutPtr[ADDR_WIDTH] == 1'b0 ? mOutPtr[ADDR_WIDTH-1:0] : {ADDR_WIDTH{1'b0}};

    // mOutPtr
    always @(posedge clk) begin
        if (reset)
            mOutPtr <= {ADDR_WIDTH+1{1'b1}};
        else if (push & ~pop)
            mOutPtr <= mOutPtr + 1'b1;
        else if (~push & pop)
            mOutPtr <= mOutPtr - 1'b1;
    end

    // full_n
    always @(posedge clk) begin
        if (reset)
            full_n <= 1'b1;
        else if ((push & ~pop) && (mOutPtr == DEPTH - 2))
            full_n <= 1'b0;
        else if (~push & pop)
            full_n <= 1'b1;
    end

    // empty_n
    always @(posedge clk) begin
        if (reset)
            empty_n <= 1'b0;
        else if (push & ~pop)
            empty_n <= 1'b1;
        else if ((~push & pop) && (mOutPtr == 0))
            empty_n <= 1'b0;
    end

    // almost_full_n 

    // almost_empty_n 

    // prog_full_n 
 
    // prog_empty_n 

    // num_data_valid 

endmodule  


module STATE_AXI_start_for_writeback_indexed_order_U0_ShiftReg
#(parameter
    DATA_WIDTH  = 1,
    ADDR_WIDTH  = 1,
    DEPTH       = 2)
(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] din,
    output wire [DATA_WIDTH-1:0] dout
);

    reg [DATA_WIDTH-1:0] SRL_SIG [0:DEPTH-1];
    integer i;

    always @ (posedge clk) begin
        if (we) begin
            for (i=0; i<DEPTH-1; i=i+1)
                SRL_SIG[i+1] <= SRL_SIG[i];
            SRL_SIG[0] <= din;
        end
    end

    assign dout = SRL_SIG[addr];

endmodule// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        tile_idx_stream_dout,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_empty_n,
        tile_idx_stream_read,
        y_stream_TVALID,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        y_stream_TDATA,
        y_stream_TREADY,
        memory_state_load
);

parameter    ap_ST_iter0_fsm_state1 = 2'd1;
parameter    ap_ST_iter0_fsm_state2 = 2'd2;
parameter    ap_ST_iter1_fsm_state3 = 3'd2;
parameter    ap_ST_iter1_fsm_state4 = 3'd4;
parameter    ap_ST_iter2_fsm_state5 = 3'd2;
parameter    ap_ST_iter2_fsm_state6 = 3'd4;
parameter    ap_ST_iter3_fsm_state7 = 3'd2;
parameter    ap_ST_iter3_fsm_state8 = 3'd4;
parameter    ap_ST_iter4_fsm_state9 = 3'd2;
parameter    ap_ST_iter4_fsm_state10 = 3'd4;
parameter    ap_ST_iter5_fsm_state11 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 3'd1;
parameter    ap_ST_iter2_fsm_state0 = 3'd1;
parameter    ap_ST_iter3_fsm_state0 = 3'd1;
parameter    ap_ST_iter4_fsm_state0 = 3'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [31:0] tile_idx_stream_dout;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_empty_n;
output   tile_idx_stream_read;
input   y_stream_TVALID;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [199:0] y_stream_TDATA;
output   y_stream_TREADY;
input  [63:0] memory_state_load;

reg ap_idle;
reg tile_idx_stream_read;
reg m_axi_gmem1_AWVALID;
reg m_axi_gmem1_WVALID;
reg[127:0] m_axi_gmem1_WDATA;
reg m_axi_gmem1_BREADY;
reg y_stream_TREADY;

reg   [1:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [2:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [2:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [2:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
reg   [2:0] ap_CS_iter4_fsm;
wire    ap_CS_iter4_fsm_state0;
reg   [1:0] ap_CS_iter5_fsm;
wire    ap_CS_iter5_fsm_state0;
wire    ap_CS_iter0_fsm_state2;
reg   [0:0] icmp_ln137_reg_355;
reg    ap_block_state2_pp0_stage1_iter0;
wire    ap_CS_iter1_fsm_state3;
reg   [0:0] icmp_ln137_reg_355_pp0_iter0_reg;
reg    ap_block_state4_io;
wire    ap_CS_iter1_fsm_state4;
reg   [0:0] icmp_ln137_reg_355_pp0_iter1_reg;
reg    ap_block_state5_io;
wire    ap_CS_iter2_fsm_state5;
reg    ap_block_state6_io;
wire    ap_CS_iter2_fsm_state6;
wire    ap_CS_iter3_fsm_state7;
wire    ap_CS_iter3_fsm_state8;
wire    ap_CS_iter4_fsm_state9;
wire    ap_CS_iter4_fsm_state10;
reg   [0:0] icmp_ln137_reg_355_pp0_iter4_reg;
reg    ap_block_state11_pp0_stage0_iter5;
wire    ap_CS_iter5_fsm_state11;
reg    ap_condition_exit_pp0_iter0_stage1;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    tile_idx_stream_blk_n;
reg    y_stream_TDATA_blk_n;
reg    gmem1_blk_n_AW;
reg    gmem1_blk_n_W;
reg    gmem1_blk_n_B;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln137_fu_162_p2;
reg   [0:0] icmp_ln137_reg_355_pp0_iter2_reg;
reg   [0:0] icmp_ln137_reg_355_pp0_iter3_reg;
wire   [24:0] trunc_ln140_fu_179_p1;
reg   [24:0] trunc_ln140_reg_359;
reg   [24:0] trunc_ln140_reg_359_pp0_iter1_reg;
reg   [24:0] trunc_ln140_1_reg_364;
reg   [24:0] trunc_ln140_1_reg_364_pp0_iter1_reg;
reg   [24:0] trunc_ln140_2_reg_369;
reg   [24:0] trunc_ln140_2_reg_369_pp0_iter1_reg;
reg   [24:0] packet_reg_374;
reg   [24:0] packet_reg_374_pp0_iter1_reg;
reg   [24:0] trunc_ln140_4_reg_379;
reg   [24:0] trunc_ln140_4_reg_379_pp0_iter1_reg;
reg   [24:0] trunc_ln140_5_reg_384;
reg   [24:0] trunc_ln140_5_reg_384_pp0_iter1_reg;
reg   [24:0] trunc_ln140_6_reg_389;
reg   [24:0] trunc_ln140_6_reg_389_pp0_iter1_reg;
reg   [24:0] packet_5_reg_394;
reg   [24:0] packet_5_reg_394_pp0_iter1_reg;
wire   [30:0] empty_fu_253_p1;
reg   [30:0] empty_reg_399;
reg   [59:0] trunc_ln2_reg_404;
wire  signed [63:0] sext_ln137_fu_283_p1;
wire  signed [127:0] sext_ln225_15_fu_313_p1;
wire  signed [127:0] sext_ln225_16_fu_338_p1;
reg   [16:0] i_fu_108;
wire   [16:0] i_4_fu_168_p2;
wire    ap_loop_init;
reg   [16:0] ap_sig_allocacmp_i_3;
wire   [35:0] shl_ln_fu_257_p3;
wire  signed [63:0] sext_ln227_fu_264_p1;
wire   [63:0] add_ln227_fu_268_p2;
wire  signed [31:0] sext_ln225_fu_293_p1;
wire  signed [31:0] sext_ln225_6_fu_296_p1;
wire  signed [31:0] sext_ln225_7_fu_299_p1;
wire   [120:0] packet_7_fu_302_p5;
wire  signed [31:0] sext_ln225_8_fu_318_p1;
wire  signed [31:0] sext_ln225_9_fu_321_p1;
wire  signed [31:0] sext_ln225_10_fu_324_p1;
wire   [120:0] packet_8_fu_327_p5;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg   [1:0] ap_NS_iter0_fsm;
reg   [2:0] ap_NS_iter1_fsm;
reg   [2:0] ap_NS_iter2_fsm;
reg   [2:0] ap_NS_iter3_fsm;
reg   [2:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter1_fsm_state3_blk;
reg    ap_ST_iter1_fsm_state4_blk;
reg    ap_ST_iter2_fsm_state5_blk;
reg    ap_ST_iter2_fsm_state6_blk;
wire    ap_ST_iter3_fsm_state7_blk;
wire    ap_ST_iter3_fsm_state8_blk;
wire    ap_ST_iter4_fsm_state9_blk;
wire    ap_ST_iter4_fsm_state10_blk;
reg    ap_ST_iter5_fsm_state11_blk;
wire    ap_start_int;
reg    ap_condition_248;
reg    ap_condition_563;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 2'd1;
//#0 ap_CS_iter1_fsm = 3'd1;
//#0 ap_CS_iter2_fsm = 3'd1;
//#0 ap_CS_iter3_fsm = 3'd1;
//#0 ap_CS_iter4_fsm = 3'd1;
//#0 ap_CS_iter5_fsm = 2'd1;
//#0 i_fu_108 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage1),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter3_fsm <= ap_ST_iter3_fsm_state0;
    end else begin
        ap_CS_iter3_fsm <= ap_NS_iter3_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter4_fsm <= ap_ST_iter4_fsm_state0;
    end else begin
        ap_CS_iter4_fsm <= ap_NS_iter4_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter5_fsm <= ap_ST_iter5_fsm_state0;
    end else begin
        ap_CS_iter5_fsm <= ap_NS_iter5_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b0) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_248)) begin
        if ((icmp_ln137_fu_162_p2 == 1'd0)) begin
            i_fu_108 <= i_4_fu_168_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            i_fu_108 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        empty_reg_399 <= empty_fu_253_p1;
        icmp_ln137_reg_355_pp0_iter0_reg <= icmp_ln137_reg_355;
        packet_5_reg_394 <= {{y_stream_TDATA[199:175]}};
        packet_reg_374 <= {{y_stream_TDATA[99:75]}};
        trunc_ln140_1_reg_364 <= {{y_stream_TDATA[49:25]}};
        trunc_ln140_2_reg_369 <= {{y_stream_TDATA[74:50]}};
        trunc_ln140_4_reg_379 <= {{y_stream_TDATA[124:100]}};
        trunc_ln140_5_reg_384 <= {{y_stream_TDATA[149:125]}};
        trunc_ln140_6_reg_389 <= {{y_stream_TDATA[174:150]}};
        trunc_ln140_reg_359 <= trunc_ln140_fu_179_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln137_reg_355_pp0_iter1_reg <= icmp_ln137_reg_355_pp0_iter0_reg;
        packet_5_reg_394_pp0_iter1_reg <= packet_5_reg_394;
        packet_reg_374_pp0_iter1_reg <= packet_reg_374;
        trunc_ln140_1_reg_364_pp0_iter1_reg <= trunc_ln140_1_reg_364;
        trunc_ln140_2_reg_369_pp0_iter1_reg <= trunc_ln140_2_reg_369;
        trunc_ln140_4_reg_379_pp0_iter1_reg <= trunc_ln140_4_reg_379;
        trunc_ln140_5_reg_384_pp0_iter1_reg <= trunc_ln140_5_reg_384;
        trunc_ln140_6_reg_389_pp0_iter1_reg <= trunc_ln140_6_reg_389;
        trunc_ln140_reg_359_pp0_iter1_reg <= trunc_ln140_reg_359;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln137_reg_355_pp0_iter2_reg <= icmp_ln137_reg_355_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln137_reg_355_pp0_iter3_reg <= icmp_ln137_reg_355_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln137_reg_355 <= icmp_ln137_fu_162_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        icmp_ln137_reg_355_pp0_iter4_reg <= icmp_ln137_reg_355_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
        trunc_ln2_reg_404 <= {{add_ln227_fu_268_p2[63:4]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state2_pp0_stage1_iter0)) begin
        ap_ST_iter0_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state2_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_io)) begin
        ap_ST_iter1_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state5_io)) begin
        ap_ST_iter2_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state6_io)) begin
        ap_ST_iter2_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state6_blk = 1'b0;
    end
end

assign ap_ST_iter3_fsm_state7_blk = 1'b0;

assign ap_ST_iter3_fsm_state8_blk = 1'b0;

assign ap_ST_iter4_fsm_state10_blk = 1'b0;

assign ap_ST_iter4_fsm_state9_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state11_pp0_stage0_iter5)) begin
        ap_ST_iter5_fsm_state11_blk = 1'b1;
    end else begin
        ap_ST_iter5_fsm_state11_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln137_reg_355 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_i_3 = 17'd0;
    end else begin
        ap_sig_allocacmp_i_3 = i_fu_108;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state4) & (icmp_ln137_reg_355_pp0_iter0_reg == 1'd0))) begin
        gmem1_blk_n_AW = m_axi_gmem1_AWREADY;
    end else begin
        gmem1_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (icmp_ln137_reg_355_pp0_iter4_reg == 1'd0))) begin
        gmem1_blk_n_B = m_axi_gmem1_BVALID;
    end else begin
        gmem1_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter2_fsm_state6) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)))) begin
        gmem1_blk_n_W = m_axi_gmem1_WREADY;
    end else begin
        gmem1_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4) & (icmp_ln137_reg_355_pp0_iter0_reg == 1'd0))) begin
        m_axi_gmem1_AWVALID = 1'b1;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (1'b0 == ap_block_state11_pp0_stage0_iter5) & (icmp_ln137_reg_355_pp0_iter4_reg == 1'd0))) begin
        m_axi_gmem1_BREADY = 1'b1;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_condition_563)) begin
        if ((1'b1 == ap_CS_iter2_fsm_state6)) begin
            m_axi_gmem1_WDATA = sext_ln225_16_fu_338_p1;
        end else if ((1'b1 == ap_CS_iter2_fsm_state5)) begin
            m_axi_gmem1_WDATA = sext_ln225_15_fu_313_p1;
        end else begin
            m_axi_gmem1_WDATA = 'bx;
        end
    end else begin
        m_axi_gmem1_WDATA = 'bx;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)) | (~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)))) begin
        m_axi_gmem1_WVALID = 1'b1;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        tile_idx_stream_blk_n = tile_idx_stream_empty_n;
    end else begin
        tile_idx_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        tile_idx_stream_read = 1'b1;
    end else begin
        tile_idx_stream_read = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        y_stream_TDATA_blk_n = y_stream_TVALID;
    end else begin
        y_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        y_stream_TREADY = 1'b1;
    end else begin
        y_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if (~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end
        end
        ap_ST_iter1_fsm_state4 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (1'b0 == ap_block_state2_pp0_stage1_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & ((1'b0 == ap_CS_iter0_fsm_state2) | ((1'b1 == ap_CS_iter0_fsm_state2) & (1'b1 == ap_block_state2_pp0_stage1_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state5 : begin
            if ((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state5))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end
        end
        ap_ST_iter2_fsm_state6 : begin
            if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state4) & (1'b0 == ap_block_state4_io))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & ((1'b0 == ap_CS_iter1_fsm_state4) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter3_fsm)
        ap_ST_iter3_fsm_state7 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state7))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end
        end
        ap_ST_iter3_fsm_state8 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state6) & (1'b0 == ap_block_state6_io))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & ((1'b0 == ap_CS_iter2_fsm_state6) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io))))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter3_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter4_fsm)
        ap_ST_iter4_fsm_state9 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state9))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end
        end
        ap_ST_iter4_fsm_state10 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter4_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter5_fsm)
        ap_ST_iter5_fsm_state11 : begin
            if (((1'b0 == ap_CS_iter4_fsm_state10) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter5_fsm_state11) & (1'b0 == ap_block_state11_pp0_stage0_iter5) & (icmp_ln137_reg_355_pp0_iter4_reg == 1'd1)) | ((1'b1 == ap_CS_iter4_fsm_state10) & (1'b0 == ap_block_state11_pp0_stage0_iter5)))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter5_fsm = 'bx;
        end
    endcase
end

assign add_ln227_fu_268_p2 = ($signed(memory_state_load) + $signed(sext_ln227_fu_264_p1));

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state3 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state4 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state5 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter2_fsm_state6 = ap_CS_iter2_fsm[32'd2];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state7 = ap_CS_iter3_fsm[32'd1];

assign ap_CS_iter3_fsm_state8 = ap_CS_iter3_fsm[32'd2];

assign ap_CS_iter4_fsm_state0 = ap_CS_iter4_fsm[32'd0];

assign ap_CS_iter4_fsm_state10 = ap_CS_iter4_fsm[32'd2];

assign ap_CS_iter4_fsm_state9 = ap_CS_iter4_fsm[32'd1];

assign ap_CS_iter5_fsm_state0 = ap_CS_iter5_fsm[32'd0];

assign ap_CS_iter5_fsm_state11 = ap_CS_iter5_fsm[32'd1];

always @ (*) begin
    ap_block_state11_pp0_stage0_iter5 = ((icmp_ln137_reg_355_pp0_iter4_reg == 1'd0) & (m_axi_gmem1_BVALID == 1'b0));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state2_pp0_stage1_iter0 = (((y_stream_TVALID == 1'b0) & (icmp_ln137_reg_355 == 1'd0)) | ((icmp_ln137_reg_355 == 1'd0) & (tile_idx_stream_empty_n == 1'b0)));
end

always @ (*) begin
    ap_block_state4_io = ((icmp_ln137_reg_355_pp0_iter0_reg == 1'd0) & (m_axi_gmem1_AWREADY == 1'b0));
end

always @ (*) begin
    ap_block_state5_io = ((icmp_ln137_reg_355_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_block_state6_io = ((icmp_ln137_reg_355_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_condition_248 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

always @ (*) begin
    ap_condition_563 = (~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage1;

assign empty_fu_253_p1 = tile_idx_stream_dout[30:0];

assign i_4_fu_168_p2 = (ap_sig_allocacmp_i_3 + 17'd1);

assign icmp_ln137_fu_162_p2 = ((ap_sig_allocacmp_i_3 == 17'd98304) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = 64'd0;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd0;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_ARVALID = 1'b0;

assign m_axi_gmem1_AWADDR = sext_ln137_fu_283_p1;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd2;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd65535;

assign m_axi_gmem1_WUSER = 1'd0;

assign packet_7_fu_302_p5 = {{{{packet_reg_374_pp0_iter1_reg}, {sext_ln225_fu_293_p1}}, {sext_ln225_6_fu_296_p1}}, {sext_ln225_7_fu_299_p1}};

assign packet_8_fu_327_p5 = {{{{packet_5_reg_394_pp0_iter1_reg}, {sext_ln225_8_fu_318_p1}}, {sext_ln225_9_fu_321_p1}}, {sext_ln225_10_fu_324_p1}};

assign sext_ln137_fu_283_p1 = $signed(trunc_ln2_reg_404);

assign sext_ln225_10_fu_324_p1 = $signed(trunc_ln140_4_reg_379_pp0_iter1_reg);

assign sext_ln225_15_fu_313_p1 = $signed(packet_7_fu_302_p5);

assign sext_ln225_16_fu_338_p1 = $signed(packet_8_fu_327_p5);

assign sext_ln225_6_fu_296_p1 = $signed(trunc_ln140_1_reg_364_pp0_iter1_reg);

assign sext_ln225_7_fu_299_p1 = $signed(trunc_ln140_reg_359_pp0_iter1_reg);

assign sext_ln225_8_fu_318_p1 = $signed(trunc_ln140_6_reg_389_pp0_iter1_reg);

assign sext_ln225_9_fu_321_p1 = $signed(trunc_ln140_5_reg_384_pp0_iter1_reg);

assign sext_ln225_fu_293_p1 = $signed(trunc_ln140_2_reg_369_pp0_iter1_reg);

assign sext_ln227_fu_264_p1 = $signed(shl_ln_fu_257_p3);

assign shl_ln_fu_257_p3 = {{empty_reg_399}, {5'd0}};

assign trunc_ln140_fu_179_p1 = y_stream_TDATA[24:0];

endmodule //STATE_AXI_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module STATE_AXI_fifo_w64_d2_S
#(parameter
    MEM_STYLE    = "shiftReg",
    DATA_WIDTH   = 64,
    ADDR_WIDTH   = 2,
    DEPTH        = 2)
(
    // system signal
    input  wire                  clk,
    input  wire                  reset,

    // write
    output wire                  if_full_n,
    input  wire                  if_write_ce,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,
    
    // read 
    output wire [ADDR_WIDTH:0]   if_num_data_valid, // for FRP
    output wire [ADDR_WIDTH:0]   if_fifo_cap,       // for FRP

    output wire                  if_empty_n,
    input  wire                  if_read_ce,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout
);
//------------------------Parameter----------------------

//------------------------Local signal-------------------
    wire [ADDR_WIDTH-1:0] addr;
    wire                  push;
    wire                  pop;
    reg signed [ADDR_WIDTH:0] mOutPtr;
    reg                   empty_n = 1'b0;
    reg                   full_n = 1'b1; 
    // has num_data_valid? 
    reg  [ADDR_WIDTH:0]   num_data_valid; //yes 
//------------------------Instantiation------------------
    STATE_AXI_fifo_w64_d2_S_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_STATE_AXI_fifo_w64_d2_S_ShiftReg (
        .clk        (clk),
        .we         (push),
        .addr       (addr),
        .din        (if_din),
        .dout       (if_dout)
    );
//------------------------Task and function--------------

//------------------------Body---------------------------
    // num_data_valid 
    assign if_num_data_valid = num_data_valid;
    assign if_fifo_cap       = DEPTH;

    // almost full/empty 

    // program full/empty 

    assign if_full_n  = full_n; 
    assign if_empty_n = empty_n;

    assign push       = full_n & if_write_ce & if_write;
    assign pop        = empty_n & if_read_ce & if_read;

    assign addr       = mOutPtr[ADDR_WIDTH] == 1'b0 ? mOutPtr[ADDR_WIDTH-1:0] : {ADDR_WIDTH{1'b0}};

    // mOutPtr
    always @(posedge clk) begin
        if (reset)
            mOutPtr <= {ADDR_WIDTH+1{1'b1}};
        else if (push & ~pop)
            mOutPtr <= mOutPtr + 1'b1;
        else if (~push & pop)
            mOutPtr <= mOutPtr - 1'b1;
    end

    // full_n
    always @(posedge clk) begin
        if (reset)
            full_n <= 1'b1;
        else if ((push & ~pop) && (mOutPtr == DEPTH - 2))
            full_n <= 1'b0;
        else if (~push & pop)
            full_n <= 1'b1;
    end

    // empty_n
    always @(posedge clk) begin
        if (reset)
            empty_n <= 1'b0;
        else if (push & ~pop)
            empty_n <= 1'b1;
        else if ((~push & pop) && (mOutPtr == 0))
            empty_n <= 1'b0;
    end

    // almost_full_n 

    // almost_empty_n 

    // prog_full_n 
 
    // prog_empty_n 

    // num_data_valid 
    always @(posedge clk) begin
        if (reset)
            num_data_valid <= {ADDR_WIDTH+1{1'b0}};
        else if ( push & ~pop)
            num_data_valid <= num_data_valid + 1;
        else if (~push & pop)
            num_data_valid <= num_data_valid - 1;
    end // 

endmodule  


module STATE_AXI_fifo_w64_d2_S_ShiftReg
#(parameter
    DATA_WIDTH  = 64,
    ADDR_WIDTH  = 2,
    DEPTH       = 2)
(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] din,
    output wire [DATA_WIDTH-1:0] dout
);

    reg [DATA_WIDTH-1:0] SRL_SIG [0:DEPTH-1];
    integer i;

    always @ (posedge clk) begin
        if (we) begin
            for (i=0; i<DEPTH-1; i=i+1)
                SRL_SIG[i+1] <= SRL_SIG[i];
            SRL_SIG[0] <= din;
        end
    end

    assign dout = SRL_SIG[addr];

endmodule// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        y_stream_TVALID,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_decoder_state,
        y_stream_TDATA,
        y_stream_TREADY
);

parameter    ap_ST_iter0_fsm_state1 = 2'd1;
parameter    ap_ST_iter0_fsm_state2 = 2'd2;
parameter    ap_ST_iter1_fsm_state3 = 3'd2;
parameter    ap_ST_iter1_fsm_state4 = 3'd4;
parameter    ap_ST_iter2_fsm_state5 = 3'd2;
parameter    ap_ST_iter2_fsm_state6 = 3'd4;
parameter    ap_ST_iter3_fsm_state7 = 3'd2;
parameter    ap_ST_iter3_fsm_state8 = 3'd4;
parameter    ap_ST_iter4_fsm_state9 = 3'd2;
parameter    ap_ST_iter4_fsm_state10 = 3'd4;
parameter    ap_ST_iter5_fsm_state11 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 3'd1;
parameter    ap_ST_iter2_fsm_state0 = 3'd1;
parameter    ap_ST_iter3_fsm_state0 = 3'd1;
parameter    ap_ST_iter4_fsm_state0 = 3'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   y_stream_TVALID;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_decoder_state;
input  [199:0] y_stream_TDATA;
output   y_stream_TREADY;

reg ap_idle;
reg m_axi_gmem1_AWVALID;
reg m_axi_gmem1_WVALID;
reg[127:0] m_axi_gmem1_WDATA;
reg m_axi_gmem1_BREADY;
reg y_stream_TREADY;

reg   [1:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [2:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [2:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [2:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
reg   [2:0] ap_CS_iter4_fsm;
wire    ap_CS_iter4_fsm_state0;
reg   [1:0] ap_CS_iter5_fsm;
wire    ap_CS_iter5_fsm_state0;
wire    ap_CS_iter0_fsm_state2;
reg   [0:0] icmp_ln102_reg_490;
reg    ap_block_state2_pp0_stage1_iter0;
wire    ap_CS_iter1_fsm_state3;
reg   [0:0] icmp_ln102_reg_490_pp0_iter0_reg;
reg    ap_block_state4_io;
wire    ap_CS_iter1_fsm_state4;
reg   [0:0] icmp_ln102_reg_490_pp0_iter1_reg;
reg    ap_block_state5_io;
wire    ap_CS_iter2_fsm_state5;
reg    ap_block_state6_io;
wire    ap_CS_iter2_fsm_state6;
wire    ap_CS_iter3_fsm_state7;
wire    ap_CS_iter3_fsm_state8;
wire    ap_CS_iter4_fsm_state9;
wire    ap_CS_iter4_fsm_state10;
reg   [0:0] icmp_ln102_reg_490_pp0_iter4_reg;
reg    ap_block_state11_pp0_stage0_iter5;
wire    ap_CS_iter5_fsm_state11;
reg    ap_condition_exit_pp0_iter0_stage1;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem1_blk_n_AW;
reg    gmem1_blk_n_W;
reg    gmem1_blk_n_B;
reg    y_stream_TDATA_blk_n;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln102_fu_188_p2;
reg   [0:0] icmp_ln102_reg_490_pp0_iter2_reg;
reg   [0:0] icmp_ln102_reg_490_pp0_iter3_reg;
reg   [3:0] t_load_reg_494;
wire   [0:0] icmp_ln103_fu_206_p2;
reg   [0:0] icmp_ln103_reg_499;
wire   [6:0] select_ln102_1_fu_218_p3;
reg   [6:0] select_ln102_1_reg_504;
wire   [11:0] add_ln105_fu_287_p2;
reg   [11:0] add_ln105_reg_509;
wire   [24:0] trunc_ln105_3_fu_293_p1;
reg   [24:0] trunc_ln105_3_reg_514;
reg   [24:0] trunc_ln105_3_reg_514_pp0_iter1_reg;
reg   [24:0] trunc_ln105_1_reg_519;
reg   [24:0] trunc_ln105_1_reg_519_pp0_iter1_reg;
reg   [24:0] trunc_ln105_2_reg_524;
reg   [24:0] trunc_ln105_2_reg_524_pp0_iter1_reg;
reg   [24:0] packet_reg_529;
reg   [24:0] packet_reg_529_pp0_iter1_reg;
reg   [24:0] trunc_ln105_4_reg_534;
reg   [24:0] trunc_ln105_4_reg_534_pp0_iter1_reg;
reg   [24:0] trunc_ln105_5_reg_539;
reg   [24:0] trunc_ln105_5_reg_539_pp0_iter1_reg;
reg   [24:0] trunc_ln105_6_reg_544;
reg   [24:0] trunc_ln105_6_reg_544_pp0_iter1_reg;
reg   [24:0] packet_1_reg_549;
reg   [24:0] packet_1_reg_549_pp0_iter1_reg;
reg   [59:0] trunc_ln6_reg_554;
wire  signed [63:0] sext_ln103_fu_404_p1;
wire  signed [127:0] sext_ln225_21_fu_434_p1;
wire  signed [127:0] sext_ln225_22_fu_459_p1;
reg   [3:0] t_fu_122;
wire   [3:0] add_ln103_fu_367_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_t_load;
reg   [6:0] ct_fu_126;
reg   [6:0] ap_sig_allocacmp_ct_load;
reg   [9:0] indvar_flatten6_fu_130;
wire   [9:0] add_ln102_fu_194_p2;
reg   [9:0] ap_sig_allocacmp_indvar_flatten6_load;
wire   [6:0] add_ln102_1_fu_212_p2;
wire   [3:0] select_ln102_fu_236_p3;
wire   [2:0] trunc_ln105_fu_242_p1;
wire   [10:0] shl_ln2_fu_246_p3;
wire   [6:0] shl_ln105_1_fu_258_p3;
wire   [11:0] zext_ln105_fu_254_p1;
wire   [11:0] zext_ln105_1_fu_266_p1;
wire   [7:0] shl_ln105_2_fu_276_p3;
wire   [11:0] sub_ln105_fu_270_p2;
wire   [11:0] zext_ln105_2_fu_283_p1;
wire   [15:0] tmp_s_fu_378_p3;
wire  signed [63:0] sext_ln105_fu_385_p1;
wire   [63:0] add_ln105_1_fu_389_p2;
wire  signed [31:0] sext_ln225_fu_414_p1;
wire  signed [31:0] sext_ln225_16_fu_417_p1;
wire  signed [31:0] sext_ln225_17_fu_420_p1;
wire   [120:0] packet_10_fu_423_p5;
wire  signed [31:0] sext_ln225_18_fu_439_p1;
wire  signed [31:0] sext_ln225_19_fu_442_p1;
wire  signed [31:0] sext_ln225_20_fu_445_p1;
wire   [120:0] packet_11_fu_448_p5;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg   [1:0] ap_NS_iter0_fsm;
reg   [2:0] ap_NS_iter1_fsm;
reg   [2:0] ap_NS_iter2_fsm;
reg   [2:0] ap_NS_iter3_fsm;
reg   [2:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter1_fsm_state3_blk;
reg    ap_ST_iter1_fsm_state4_blk;
reg    ap_ST_iter2_fsm_state5_blk;
reg    ap_ST_iter2_fsm_state6_blk;
wire    ap_ST_iter3_fsm_state7_blk;
wire    ap_ST_iter3_fsm_state8_blk;
wire    ap_ST_iter4_fsm_state9_blk;
wire    ap_ST_iter4_fsm_state10_blk;
reg    ap_ST_iter5_fsm_state11_blk;
wire    ap_start_int;
reg    ap_condition_239;
reg    ap_condition_621;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 2'd1;
//#0 ap_CS_iter1_fsm = 3'd1;
//#0 ap_CS_iter2_fsm = 3'd1;
//#0 ap_CS_iter3_fsm = 3'd1;
//#0 ap_CS_iter4_fsm = 3'd1;
//#0 ap_CS_iter5_fsm = 2'd1;
//#0 t_fu_122 = 4'd0;
//#0 ct_fu_126 = 7'd0;
//#0 indvar_flatten6_fu_130 = 10'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage1),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter3_fsm <= ap_ST_iter3_fsm_state0;
    end else begin
        ap_CS_iter3_fsm <= ap_NS_iter3_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter4_fsm <= ap_ST_iter4_fsm_state0;
    end else begin
        ap_CS_iter4_fsm <= ap_NS_iter4_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter5_fsm <= ap_ST_iter5_fsm_state0;
    end else begin
        ap_CS_iter5_fsm <= ap_NS_iter5_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b0) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_239)) begin
        if ((icmp_ln102_fu_188_p2 == 1'd0)) begin
            ct_fu_126 <= select_ln102_1_fu_218_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            ct_fu_126 <= 7'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_239)) begin
        if ((icmp_ln102_fu_188_p2 == 1'd0)) begin
            indvar_flatten6_fu_130 <= add_ln102_fu_194_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten6_fu_130 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        t_fu_122 <= 4'd0;
    end else if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln102_reg_490 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        t_fu_122 <= add_ln103_fu_367_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        add_ln105_reg_509[11 : 1] <= add_ln105_fu_287_p2[11 : 1];
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln102_reg_490_pp0_iter0_reg <= icmp_ln102_reg_490;
        packet_1_reg_549 <= {{y_stream_TDATA[199:175]}};
        packet_reg_529 <= {{y_stream_TDATA[99:75]}};
        trunc_ln105_1_reg_519 <= {{y_stream_TDATA[49:25]}};
        trunc_ln105_2_reg_524 <= {{y_stream_TDATA[74:50]}};
        trunc_ln105_3_reg_514 <= trunc_ln105_3_fu_293_p1;
        trunc_ln105_4_reg_534 <= {{y_stream_TDATA[124:100]}};
        trunc_ln105_5_reg_539 <= {{y_stream_TDATA[149:125]}};
        trunc_ln105_6_reg_544 <= {{y_stream_TDATA[174:150]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln102_reg_490_pp0_iter1_reg <= icmp_ln102_reg_490_pp0_iter0_reg;
        packet_1_reg_549_pp0_iter1_reg <= packet_1_reg_549;
        packet_reg_529_pp0_iter1_reg <= packet_reg_529;
        trunc_ln105_1_reg_519_pp0_iter1_reg <= trunc_ln105_1_reg_519;
        trunc_ln105_2_reg_524_pp0_iter1_reg <= trunc_ln105_2_reg_524;
        trunc_ln105_3_reg_514_pp0_iter1_reg <= trunc_ln105_3_reg_514;
        trunc_ln105_4_reg_534_pp0_iter1_reg <= trunc_ln105_4_reg_534;
        trunc_ln105_5_reg_539_pp0_iter1_reg <= trunc_ln105_5_reg_539;
        trunc_ln105_6_reg_544_pp0_iter1_reg <= trunc_ln105_6_reg_544;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln102_reg_490_pp0_iter2_reg <= icmp_ln102_reg_490_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln102_reg_490_pp0_iter3_reg <= icmp_ln102_reg_490_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln102_reg_490 <= icmp_ln102_fu_188_p2;
        icmp_ln103_reg_499 <= icmp_ln103_fu_206_p2;
        select_ln102_1_reg_504 <= select_ln102_1_fu_218_p3;
        t_load_reg_494 <= ap_sig_allocacmp_t_load;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        icmp_ln102_reg_490_pp0_iter4_reg <= icmp_ln102_reg_490_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
        trunc_ln6_reg_554 <= {{add_ln105_1_fu_389_p2[63:4]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state2_pp0_stage1_iter0)) begin
        ap_ST_iter0_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state2_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_io)) begin
        ap_ST_iter1_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state5_io)) begin
        ap_ST_iter2_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state6_io)) begin
        ap_ST_iter2_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state6_blk = 1'b0;
    end
end

assign ap_ST_iter3_fsm_state7_blk = 1'b0;

assign ap_ST_iter3_fsm_state8_blk = 1'b0;

assign ap_ST_iter4_fsm_state10_blk = 1'b0;

assign ap_ST_iter4_fsm_state9_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state11_pp0_stage0_iter5)) begin
        ap_ST_iter5_fsm_state11_blk = 1'b1;
    end else begin
        ap_ST_iter5_fsm_state11_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln102_reg_490 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_ct_load = 7'd0;
    end else begin
        ap_sig_allocacmp_ct_load = ct_fu_126;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten6_load = 10'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten6_load = indvar_flatten6_fu_130;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_t_load = 4'd0;
    end else begin
        ap_sig_allocacmp_t_load = t_fu_122;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state4) & (icmp_ln102_reg_490_pp0_iter0_reg == 1'd0))) begin
        gmem1_blk_n_AW = m_axi_gmem1_AWREADY;
    end else begin
        gmem1_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (icmp_ln102_reg_490_pp0_iter4_reg == 1'd0))) begin
        gmem1_blk_n_B = m_axi_gmem1_BVALID;
    end else begin
        gmem1_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter2_fsm_state6) & (icmp_ln102_reg_490_pp0_iter1_reg == 1'd0)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln102_reg_490_pp0_iter1_reg == 1'd0)))) begin
        gmem1_blk_n_W = m_axi_gmem1_WREADY;
    end else begin
        gmem1_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4) & (icmp_ln102_reg_490_pp0_iter0_reg == 1'd0))) begin
        m_axi_gmem1_AWVALID = 1'b1;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (1'b0 == ap_block_state11_pp0_stage0_iter5) & (icmp_ln102_reg_490_pp0_iter4_reg == 1'd0))) begin
        m_axi_gmem1_BREADY = 1'b1;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_condition_621)) begin
        if ((1'b1 == ap_CS_iter2_fsm_state6)) begin
            m_axi_gmem1_WDATA = sext_ln225_22_fu_459_p1;
        end else if ((1'b1 == ap_CS_iter2_fsm_state5)) begin
            m_axi_gmem1_WDATA = sext_ln225_21_fu_434_p1;
        end else begin
            m_axi_gmem1_WDATA = 'bx;
        end
    end else begin
        m_axi_gmem1_WDATA = 'bx;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln102_reg_490_pp0_iter1_reg == 1'd0)) | (~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6) & (icmp_ln102_reg_490_pp0_iter1_reg == 1'd0)))) begin
        m_axi_gmem1_WVALID = 1'b1;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln102_reg_490 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        y_stream_TDATA_blk_n = y_stream_TVALID;
    end else begin
        y_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln102_reg_490 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        y_stream_TREADY = 1'b1;
    end else begin
        y_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if (~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end
        end
        ap_ST_iter1_fsm_state4 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (1'b0 == ap_block_state2_pp0_stage1_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & ((1'b0 == ap_CS_iter0_fsm_state2) | ((1'b1 == ap_CS_iter0_fsm_state2) & (1'b1 == ap_block_state2_pp0_stage1_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state5 : begin
            if ((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state5))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end
        end
        ap_ST_iter2_fsm_state6 : begin
            if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state4) & (1'b0 == ap_block_state4_io))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & ((1'b0 == ap_CS_iter1_fsm_state4) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter3_fsm)
        ap_ST_iter3_fsm_state7 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state7))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end
        end
        ap_ST_iter3_fsm_state8 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state6) & (1'b0 == ap_block_state6_io))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & ((1'b0 == ap_CS_iter2_fsm_state6) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io))))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter3_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter4_fsm)
        ap_ST_iter4_fsm_state9 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state9))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end
        end
        ap_ST_iter4_fsm_state10 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter4_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter5_fsm)
        ap_ST_iter5_fsm_state11 : begin
            if (((1'b0 == ap_CS_iter4_fsm_state10) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter5_fsm_state11) & (1'b0 == ap_block_state11_pp0_stage0_iter5) & (icmp_ln102_reg_490_pp0_iter4_reg == 1'd1)) | ((1'b1 == ap_CS_iter4_fsm_state10) & (1'b0 == ap_block_state11_pp0_stage0_iter5)))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter5_fsm = 'bx;
        end
    endcase
end

assign add_ln102_1_fu_212_p2 = (ap_sig_allocacmp_ct_load + 7'd1);

assign add_ln102_fu_194_p2 = (ap_sig_allocacmp_indvar_flatten6_load + 10'd1);

assign add_ln103_fu_367_p2 = (select_ln102_fu_236_p3 + 4'd1);

assign add_ln105_1_fu_389_p2 = ($signed(sext_ln105_fu_385_p1) + $signed(memory_decoder_state));

assign add_ln105_fu_287_p2 = (sub_ln105_fu_270_p2 + zext_ln105_2_fu_283_p1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state3 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state4 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state5 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter2_fsm_state6 = ap_CS_iter2_fsm[32'd2];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state7 = ap_CS_iter3_fsm[32'd1];

assign ap_CS_iter3_fsm_state8 = ap_CS_iter3_fsm[32'd2];

assign ap_CS_iter4_fsm_state0 = ap_CS_iter4_fsm[32'd0];

assign ap_CS_iter4_fsm_state10 = ap_CS_iter4_fsm[32'd2];

assign ap_CS_iter4_fsm_state9 = ap_CS_iter4_fsm[32'd1];

assign ap_CS_iter5_fsm_state0 = ap_CS_iter5_fsm[32'd0];

assign ap_CS_iter5_fsm_state11 = ap_CS_iter5_fsm[32'd1];

always @ (*) begin
    ap_block_state11_pp0_stage0_iter5 = ((icmp_ln102_reg_490_pp0_iter4_reg == 1'd0) & (m_axi_gmem1_BVALID == 1'b0));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state2_pp0_stage1_iter0 = ((icmp_ln102_reg_490 == 1'd0) & (y_stream_TVALID == 1'b0));
end

always @ (*) begin
    ap_block_state4_io = ((m_axi_gmem1_AWREADY == 1'b0) & (icmp_ln102_reg_490_pp0_iter0_reg == 1'd0));
end

always @ (*) begin
    ap_block_state5_io = ((icmp_ln102_reg_490_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_block_state6_io = ((icmp_ln102_reg_490_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_condition_239 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

always @ (*) begin
    ap_condition_621 = (~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (icmp_ln102_reg_490_pp0_iter1_reg == 1'd0));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage1;

assign icmp_ln102_fu_188_p2 = ((ap_sig_allocacmp_indvar_flatten6_load == 10'd960) ? 1'b1 : 1'b0);

assign icmp_ln103_fu_206_p2 = ((ap_sig_allocacmp_t_load == 4'd8) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = 64'd0;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd0;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_ARVALID = 1'b0;

assign m_axi_gmem1_AWADDR = sext_ln103_fu_404_p1;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd2;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd65535;

assign m_axi_gmem1_WUSER = 1'd0;

assign packet_10_fu_423_p5 = {{{{packet_reg_529_pp0_iter1_reg}, {sext_ln225_fu_414_p1}}, {sext_ln225_16_fu_417_p1}}, {sext_ln225_17_fu_420_p1}};

assign packet_11_fu_448_p5 = {{{{packet_1_reg_549_pp0_iter1_reg}, {sext_ln225_18_fu_439_p1}}, {sext_ln225_19_fu_442_p1}}, {sext_ln225_20_fu_445_p1}};

assign select_ln102_1_fu_218_p3 = ((icmp_ln103_fu_206_p2[0:0] == 1'b1) ? add_ln102_1_fu_212_p2 : ap_sig_allocacmp_ct_load);

assign select_ln102_fu_236_p3 = ((icmp_ln103_reg_499[0:0] == 1'b1) ? 4'd0 : t_load_reg_494);

assign sext_ln103_fu_404_p1 = $signed(trunc_ln6_reg_554);

assign sext_ln105_fu_385_p1 = $signed(tmp_s_fu_378_p3);

assign sext_ln225_16_fu_417_p1 = $signed(trunc_ln105_1_reg_519_pp0_iter1_reg);

assign sext_ln225_17_fu_420_p1 = $signed(trunc_ln105_3_reg_514_pp0_iter1_reg);

assign sext_ln225_18_fu_439_p1 = $signed(trunc_ln105_6_reg_544_pp0_iter1_reg);

assign sext_ln225_19_fu_442_p1 = $signed(trunc_ln105_5_reg_539_pp0_iter1_reg);

assign sext_ln225_20_fu_445_p1 = $signed(trunc_ln105_4_reg_534_pp0_iter1_reg);

assign sext_ln225_21_fu_434_p1 = $signed(packet_10_fu_423_p5);

assign sext_ln225_22_fu_459_p1 = $signed(packet_11_fu_448_p5);

assign sext_ln225_fu_414_p1 = $signed(trunc_ln105_2_reg_524_pp0_iter1_reg);

assign shl_ln105_1_fu_258_p3 = {{trunc_ln105_fu_242_p1}, {4'd0}};

assign shl_ln105_2_fu_276_p3 = {{select_ln102_1_reg_504}, {1'd0}};

assign shl_ln2_fu_246_p3 = {{trunc_ln105_fu_242_p1}, {8'd0}};

assign sub_ln105_fu_270_p2 = (zext_ln105_fu_254_p1 - zext_ln105_1_fu_266_p1);

assign tmp_s_fu_378_p3 = {{add_ln105_reg_509}, {4'd0}};

assign trunc_ln105_3_fu_293_p1 = y_stream_TDATA[24:0];

assign trunc_ln105_fu_242_p1 = select_ln102_fu_236_p3[2:0];

assign zext_ln105_1_fu_266_p1 = shl_ln105_1_fu_258_p3;

assign zext_ln105_2_fu_283_p1 = shl_ln105_2_fu_276_p3;

assign zext_ln105_fu_254_p1 = shl_ln2_fu_246_p3;

always @ (posedge ap_clk) begin
    add_ln105_reg_509[0] <= 1'b0;
end

endmodule //STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_208_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        empty,
        memory_cls_y
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter4_fsm_state5 = 2'd2;
parameter    ap_ST_iter5_fsm_state6 = 2'd2;
parameter    ap_ST_iter6_fsm_state7 = 2'd2;
parameter    ap_ST_iter7_fsm_state8 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;
parameter    ap_ST_iter4_fsm_state0 = 2'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;
parameter    ap_ST_iter6_fsm_state0 = 2'd1;
parameter    ap_ST_iter7_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [255:0] empty;
input  [63:0] memory_cls_y;

reg ap_idle;
reg m_axi_gmem1_AWVALID;
reg m_axi_gmem1_WVALID;
reg m_axi_gmem1_BREADY;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [1:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [1:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
reg   [1:0] ap_CS_iter4_fsm;
wire    ap_CS_iter4_fsm_state0;
reg   [1:0] ap_CS_iter5_fsm;
wire    ap_CS_iter5_fsm_state0;
reg   [1:0] ap_CS_iter6_fsm;
wire    ap_CS_iter6_fsm_state0;
reg   [1:0] ap_CS_iter7_fsm;
wire    ap_CS_iter7_fsm_state0;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln208_reg_234;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg   [0:0] icmp_ln208_reg_234_pp0_iter1_reg;
reg    ap_block_state3_io;
wire    ap_CS_iter2_fsm_state3;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter4_fsm_state5;
wire    ap_CS_iter5_fsm_state6;
wire    ap_CS_iter6_fsm_state7;
reg   [0:0] icmp_ln208_reg_234_pp0_iter6_reg;
reg    ap_block_state8_pp0_stage0_iter7;
wire    ap_CS_iter7_fsm_state8;
wire   [0:0] icmp_ln208_fu_113_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem1_blk_n_AW;
reg    gmem1_blk_n_W;
reg    gmem1_blk_n_B;
reg   [0:0] icmp_ln208_reg_234_pp0_iter2_reg;
reg   [0:0] icmp_ln208_reg_234_pp0_iter3_reg;
reg   [0:0] icmp_ln208_reg_234_pp0_iter4_reg;
reg   [0:0] icmp_ln208_reg_234_pp0_iter5_reg;
wire   [31:0] trunc_ln210_1_fu_147_p1;
reg   [31:0] trunc_ln210_1_reg_238;
wire   [3:0] trunc_ln210_2_fu_169_p1;
reg   [3:0] trunc_ln210_2_reg_243;
reg   [59:0] trunc_ln210_3_reg_249;
wire   [15:0] shl_ln210_fu_194_p2;
reg   [15:0] shl_ln210_reg_254;
wire   [127:0] shl_ln210_2_fu_211_p2;
reg   [127:0] shl_ln210_2_reg_259;
wire  signed [63:0] sext_ln210_fu_217_p1;
reg   [3:0] t_fu_74;
wire   [3:0] add_ln208_fu_119_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_t_1;
wire   [2:0] trunc_ln210_fu_125_p1;
wire   [7:0] shl_ln_fu_129_p3;
wire   [255:0] zext_ln210_fu_137_p1;
wire   [255:0] lshr_ln210_fu_141_p2;
wire   [4:0] shl_ln210_1_fu_151_p3;
wire   [63:0] zext_ln210_3_fu_159_p1;
wire   [63:0] add_ln210_fu_163_p2;
wire   [15:0] zext_ln210_2_fu_191_p1;
wire   [6:0] shl_ln210_3_fu_200_p3;
wire   [127:0] zext_ln210_1_fu_188_p1;
wire   [127:0] zext_ln210_4_fu_207_p1;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg    ap_loop_exit_ready_pp0_iter6_reg;
reg    ap_loop_exit_ready_pp0_iter7_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg   [1:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg   [1:0] ap_NS_iter6_fsm;
reg   [1:0] ap_NS_iter7_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter1_fsm_state2_blk;
reg    ap_ST_iter2_fsm_state3_blk;
wire    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
wire    ap_ST_iter5_fsm_state6_blk;
wire    ap_ST_iter6_fsm_state7_blk;
reg    ap_ST_iter7_fsm_state8_blk;
wire    ap_start_int;
reg    ap_condition_199;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 ap_CS_iter2_fsm = 2'd1;
//#0 ap_CS_iter3_fsm = 2'd1;
//#0 ap_CS_iter4_fsm = 2'd1;
//#0 ap_CS_iter5_fsm = 2'd1;
//#0 ap_CS_iter6_fsm = 2'd1;
//#0 ap_CS_iter7_fsm = 2'd1;
//#0 t_fu_74 = 4'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter3_fsm <= ap_ST_iter3_fsm_state0;
    end else begin
        ap_CS_iter3_fsm <= ap_NS_iter3_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter4_fsm <= ap_ST_iter4_fsm_state0;
    end else begin
        ap_CS_iter4_fsm <= ap_NS_iter4_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter5_fsm <= ap_ST_iter5_fsm_state0;
    end else begin
        ap_CS_iter5_fsm <= ap_NS_iter5_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter6_fsm <= ap_ST_iter6_fsm_state0;
    end else begin
        ap_CS_iter6_fsm <= ap_NS_iter6_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter7_fsm <= ap_ST_iter7_fsm_state0;
    end else begin
        ap_CS_iter7_fsm <= ap_NS_iter7_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter7_fsm_state8) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b1) & (1'b0 == ap_block_state8_pp0_stage0_iter7))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter7_fsm_state8) & (ap_loop_exit_ready_pp0_iter6_reg == 1'b0) & (1'b0 == ap_block_state8_pp0_stage0_iter7))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_199)) begin
        if ((icmp_ln208_fu_113_p2 == 1'd0)) begin
            t_fu_74 <= add_ln208_fu_119_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            t_fu_74 <= 4'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io)) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln208_reg_234 <= icmp_ln208_fu_113_p2;
        trunc_ln210_1_reg_238 <= trunc_ln210_1_fu_147_p1;
        trunc_ln210_2_reg_243 <= trunc_ln210_2_fu_169_p1;
        trunc_ln210_3_reg_249 <= {{add_ln210_fu_163_p2[63:4]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln208_reg_234_pp0_iter1_reg <= icmp_ln208_reg_234;
        shl_ln210_2_reg_259 <= shl_ln210_2_fu_211_p2;
        shl_ln210_reg_254 <= shl_ln210_fu_194_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state3_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln208_reg_234_pp0_iter2_reg <= icmp_ln208_reg_234_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln208_reg_234_pp0_iter3_reg <= icmp_ln208_reg_234_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        icmp_ln208_reg_234_pp0_iter4_reg <= icmp_ln208_reg_234_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        icmp_ln208_reg_234_pp0_iter5_reg <= icmp_ln208_reg_234_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
        icmp_ln208_reg_234_pp0_iter6_reg <= icmp_ln208_reg_234_pp0_iter5_reg;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state2_io)) begin
        ap_ST_iter1_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state2_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state3_io)) begin
        ap_ST_iter2_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state3_blk = 1'b0;
    end
end

assign ap_ST_iter3_fsm_state4_blk = 1'b0;

assign ap_ST_iter4_fsm_state5_blk = 1'b0;

assign ap_ST_iter5_fsm_state6_blk = 1'b0;

assign ap_ST_iter6_fsm_state7_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state8_pp0_stage0_iter7)) begin
        ap_ST_iter7_fsm_state8_blk = 1'b1;
    end else begin
        ap_ST_iter7_fsm_state8_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io)) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (icmp_ln208_fu_113_p2 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state8) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b1) & (1'b0 == ap_block_state8_pp0_stage0_iter7))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state0) & (1'b1 == ap_CS_iter6_fsm_state0) & (1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io)) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_t_1 = 4'd0;
    end else begin
        ap_sig_allocacmp_t_1 = t_fu_74;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln208_reg_234 == 1'd0))) begin
        gmem1_blk_n_AW = m_axi_gmem1_AWREADY;
    end else begin
        gmem1_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state8) & (icmp_ln208_reg_234_pp0_iter6_reg == 1'd0))) begin
        gmem1_blk_n_B = m_axi_gmem1_BVALID;
    end else begin
        gmem1_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln208_reg_234_pp0_iter1_reg == 1'd0))) begin
        gmem1_blk_n_W = m_axi_gmem1_WREADY;
    end else begin
        gmem1_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io))) & (1'b1 == ap_CS_iter1_fsm_state2) & (icmp_ln208_reg_234 == 1'd0))) begin
        m_axi_gmem1_AWVALID = 1'b1;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state8) & (1'b0 == ap_block_state8_pp0_stage0_iter7) & (icmp_ln208_reg_234_pp0_iter6_reg == 1'd0))) begin
        m_axi_gmem1_BREADY = 1'b1;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state3_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7))) & (1'b1 == ap_CS_iter2_fsm_state3) & (icmp_ln208_reg_234_pp0_iter1_reg == 1'd0))) begin
        m_axi_gmem1_WVALID = 1'b1;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state2 : begin
            if ((~((1'b1 == ap_block_state2_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~((1'b1 == ap_block_state2_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io)) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state3 : begin
            if ((~((1'b1 == ap_block_state3_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7))) & (1'b1 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state2_io))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~((1'b1 == ap_block_state3_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7))) & ((1'b0 == ap_CS_iter1_fsm_state2) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state2_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter3_fsm)
        ap_ST_iter3_fsm_state4 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter2_fsm_state3) & (1'b0 == ap_block_state3_io))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & ((1'b0 == ap_CS_iter2_fsm_state3) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io))))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state3_io) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter3_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter4_fsm)
        ap_ST_iter4_fsm_state5 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b0 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter4_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter5_fsm)
        ap_ST_iter5_fsm_state6 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b0 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter5_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter6_fsm)
        ap_ST_iter6_fsm_state7 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end else if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b0 == ap_CS_iter5_fsm_state6))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end
        end
        ap_ST_iter6_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter6_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter7_fsm)
        ap_ST_iter7_fsm_state8 : begin
            if (((1'b0 == ap_CS_iter6_fsm_state7) & (1'b0 == ap_block_state8_pp0_stage0_iter7))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter7_fsm_state8) & (1'b0 == ap_block_state8_pp0_stage0_iter7) & (icmp_ln208_reg_234_pp0_iter6_reg == 1'd1)) | ((1'b1 == ap_CS_iter6_fsm_state7) & (1'b0 == ap_block_state8_pp0_stage0_iter7)))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end
        end
        ap_ST_iter7_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter7_fsm = 'bx;
        end
    endcase
end

assign add_ln208_fu_119_p2 = (ap_sig_allocacmp_t_1 + 4'd1);

assign add_ln210_fu_163_p2 = (zext_ln210_3_fu_159_p1 + memory_cls_y);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state3 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state4 = ap_CS_iter3_fsm[32'd1];

assign ap_CS_iter4_fsm_state0 = ap_CS_iter4_fsm[32'd0];

assign ap_CS_iter4_fsm_state5 = ap_CS_iter4_fsm[32'd1];

assign ap_CS_iter5_fsm_state0 = ap_CS_iter5_fsm[32'd0];

assign ap_CS_iter5_fsm_state6 = ap_CS_iter5_fsm[32'd1];

assign ap_CS_iter6_fsm_state0 = ap_CS_iter6_fsm[32'd0];

assign ap_CS_iter6_fsm_state7 = ap_CS_iter6_fsm[32'd1];

assign ap_CS_iter7_fsm_state0 = ap_CS_iter7_fsm[32'd0];

assign ap_CS_iter7_fsm_state8 = ap_CS_iter7_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state2_io = ((m_axi_gmem1_AWREADY == 1'b0) & (icmp_ln208_reg_234 == 1'd0));
end

always @ (*) begin
    ap_block_state3_io = ((icmp_ln208_reg_234_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_block_state8_pp0_stage0_iter7 = ((icmp_ln208_reg_234_pp0_iter6_reg == 1'd0) & (m_axi_gmem1_BVALID == 1'b0));
end

always @ (*) begin
    ap_condition_199 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state8) & (1'b1 == ap_block_state8_pp0_stage0_iter7)) | ((1'b1 == ap_CS_iter2_fsm_state3) & (1'b1 == ap_block_state3_io)) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln208_fu_113_p2 = ((ap_sig_allocacmp_t_1 == 4'd8) ? 1'b1 : 1'b0);

assign lshr_ln210_fu_141_p2 = empty >> zext_ln210_fu_137_p1;

assign m_axi_gmem1_ARADDR = 64'd0;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd0;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_ARVALID = 1'b0;

assign m_axi_gmem1_AWADDR = sext_ln210_fu_217_p1;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd1;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WDATA = shl_ln210_2_reg_259;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = shl_ln210_reg_254;

assign m_axi_gmem1_WUSER = 1'd0;

assign sext_ln210_fu_217_p1 = $signed(trunc_ln210_3_reg_249);

assign shl_ln210_1_fu_151_p3 = {{trunc_ln210_fu_125_p1}, {2'd0}};

assign shl_ln210_2_fu_211_p2 = zext_ln210_1_fu_188_p1 << zext_ln210_4_fu_207_p1;

assign shl_ln210_3_fu_200_p3 = {{trunc_ln210_2_reg_243}, {3'd0}};

assign shl_ln210_fu_194_p2 = 16'd15 << zext_ln210_2_fu_191_p1;

assign shl_ln_fu_129_p3 = {{trunc_ln210_fu_125_p1}, {5'd0}};

assign trunc_ln210_1_fu_147_p1 = lshr_ln210_fu_141_p2[31:0];

assign trunc_ln210_2_fu_169_p1 = add_ln210_fu_163_p2[3:0];

assign trunc_ln210_fu_125_p1 = ap_sig_allocacmp_t_1[2:0];

assign zext_ln210_1_fu_188_p1 = trunc_ln210_1_reg_238;

assign zext_ln210_2_fu_191_p1 = trunc_ln210_2_reg_243;

assign zext_ln210_3_fu_159_p1 = shl_ln210_1_fu_151_p3;

assign zext_ln210_4_fu_207_p1 = shl_ln210_3_fu_200_p3;

assign zext_ln210_fu_137_p1 = shl_ln_fu_129_p3;

endmodule //STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_208_1
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="STATE_AXI_STATE_AXI,hls_ip_2023_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xczu5ev-fbvb900-1L-i,HLS_INPUT_CLOCK=2.500000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=2.295250,HLS_SYN_LAT=193698,HLS_SYN_TPT=none,HLS_SYN_MEM=20,HLS_SYN_DSP=0,HLS_SYN_FF=6478,HLS_SYN_LUT=8076,HLS_VERSION=2023_2}" *)

module STATE_AXI (
        ap_clk,
        ap_rst_n,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        mode,
        l_begin,
        l_close,
        op,
        memory_decoder_state,
        memory_vit_state,
        memory_cls_y,
        x_stream_TDATA,
        x_stream_TVALID,
        x_stream_TREADY,
        y_stream_TDATA,
        y_stream_TVALID,
        y_stream_TREADY,
        cls_stream_TDATA,
        cls_stream_TVALID,
        cls_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 22'd1;
parameter    ap_ST_fsm_state2 = 22'd2;
parameter    ap_ST_fsm_state3 = 22'd4;
parameter    ap_ST_fsm_state4 = 22'd8;
parameter    ap_ST_fsm_state5 = 22'd16;
parameter    ap_ST_fsm_state6 = 22'd32;
parameter    ap_ST_fsm_state7 = 22'd64;
parameter    ap_ST_fsm_state8 = 22'd128;
parameter    ap_ST_fsm_state9 = 22'd256;
parameter    ap_ST_fsm_state10 = 22'd512;
parameter    ap_ST_fsm_state11 = 22'd1024;
parameter    ap_ST_fsm_state12 = 22'd2048;
parameter    ap_ST_fsm_state13 = 22'd4096;
parameter    ap_ST_fsm_state14 = 22'd8192;
parameter    ap_ST_fsm_state15 = 22'd16384;
parameter    ap_ST_fsm_state16 = 22'd32768;
parameter    ap_ST_fsm_state17 = 22'd65536;
parameter    ap_ST_fsm_state18 = 22'd131072;
parameter    ap_ST_fsm_state19 = 22'd262144;
parameter    ap_ST_fsm_state20 = 22'd524288;
parameter    ap_ST_fsm_state21 = 22'd1048576;
parameter    ap_ST_fsm_state22 = 22'd2097152;
parameter    C_M_AXI_GMEM1_ID_WIDTH = 1;
parameter    C_M_AXI_GMEM1_ADDR_WIDTH = 63;
parameter    C_M_AXI_GMEM1_DATA_WIDTH = 128;
parameter    C_M_AXI_GMEM1_AWUSER_WIDTH = 1;
parameter    C_M_AXI_GMEM1_ARUSER_WIDTH = 1;
parameter    C_M_AXI_GMEM1_WUSER_WIDTH = 1;
parameter    C_M_AXI_GMEM1_RUSER_WIDTH = 1;
parameter    C_M_AXI_GMEM1_BUSER_WIDTH = 1;
parameter    C_M_AXI_GMEM1_USER_VALUE = 0;
parameter    C_M_AXI_GMEM1_PROT_VALUE = 0;
parameter    C_M_AXI_GMEM1_CACHE_VALUE = 3;
parameter    C_M_AXI_DATA_WIDTH = 32;

parameter C_M_AXI_GMEM1_WSTRB_WIDTH = (128 / 8);
parameter C_M_AXI_WSTRB_WIDTH = (32 / 8);

input   ap_clk;
input   ap_rst_n;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [C_M_AXI_GMEM1_ADDR_WIDTH - 1:0] m_axi_gmem1_AWADDR;
output  [C_M_AXI_GMEM1_ID_WIDTH - 1:0] m_axi_gmem1_AWID;
output  [7:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [C_M_AXI_GMEM1_AWUSER_WIDTH - 1:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [C_M_AXI_GMEM1_DATA_WIDTH - 1:0] m_axi_gmem1_WDATA;
output  [C_M_AXI_GMEM1_WSTRB_WIDTH - 1:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [C_M_AXI_GMEM1_ID_WIDTH - 1:0] m_axi_gmem1_WID;
output  [C_M_AXI_GMEM1_WUSER_WIDTH - 1:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [C_M_AXI_GMEM1_ADDR_WIDTH - 1:0] m_axi_gmem1_ARADDR;
output  [C_M_AXI_GMEM1_ID_WIDTH - 1:0] m_axi_gmem1_ARID;
output  [7:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [C_M_AXI_GMEM1_ARUSER_WIDTH - 1:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [C_M_AXI_GMEM1_DATA_WIDTH - 1:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [C_M_AXI_GMEM1_ID_WIDTH - 1:0] m_axi_gmem1_RID;
input  [C_M_AXI_GMEM1_RUSER_WIDTH - 1:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [C_M_AXI_GMEM1_ID_WIDTH - 1:0] m_axi_gmem1_BID;
input  [C_M_AXI_GMEM1_BUSER_WIDTH - 1:0] m_axi_gmem1_BUSER;
input  [0:0] mode;
input  [31:0] l_begin;
input  [31:0] l_close;
input  [2:0] op;
input  [63:0] memory_decoder_state;
input  [63:0] memory_vit_state;
input  [63:0] memory_cls_y;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;
input   x_stream_TREADY;
input  [199:0] y_stream_TDATA;
input   y_stream_TVALID;
output   y_stream_TREADY;
input  [255:0] cls_stream_TDATA;
input   cls_stream_TVALID;
output   cls_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;

 reg    ap_rst_n_inv;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [21:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    cls_stream_TDATA_blk_n;
wire   [0:0] icmp_ln243_fu_272_p2;
wire   [0:0] mode_read_read_fu_174_p2;
reg    ap_predicate_op76_read_state1;
reg    ap_block_state1;
wire   [0:0] icmp_ln265_1_fu_334_p2;
reg   [0:0] icmp_ln265_1_reg_434;
wire   [0:0] and_ln265_fu_340_p2;
reg   [0:0] and_ln265_reg_438;
reg   [255:0] cls_stream_read_reg_442;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_done;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_idle;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_ready;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWVALID;
wire   [63:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWADDR;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWID;
wire   [31:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWLEN;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWBURST;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWPROT;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWQOS;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWREGION;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WVALID;
wire   [127:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WDATA;
wire   [15:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WSTRB;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WLAST;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WID;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARVALID;
wire   [63:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARADDR;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARID;
wire   [31:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARLEN;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARBURST;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARPROT;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARQOS;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARREGION;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_RREADY;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_BREADY;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TREADY;
wire   [199:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TDATA;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TVALID;
wire    grp_replay_vit_delta_order_fu_196_ap_start;
wire    grp_replay_vit_delta_order_fu_196_ap_done;
wire    grp_replay_vit_delta_order_fu_196_ap_idle;
wire    grp_replay_vit_delta_order_fu_196_ap_ready;
wire    grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWVALID;
wire   [63:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWADDR;
wire   [0:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWID;
wire   [31:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWLEN;
wire   [2:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWBURST;
wire   [1:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWPROT;
wire   [3:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWQOS;
wire   [3:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWREGION;
wire   [0:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWUSER;
wire    grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WVALID;
wire   [127:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WDATA;
wire   [15:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WSTRB;
wire    grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WLAST;
wire   [0:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WID;
wire   [0:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WUSER;
wire    grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARVALID;
wire   [63:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARADDR;
wire   [0:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARID;
wire   [31:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARLEN;
wire   [2:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARBURST;
wire   [1:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARPROT;
wire   [3:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARQOS;
wire   [3:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARREGION;
wire   [0:0] grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARUSER;
wire    grp_replay_vit_delta_order_fu_196_m_axi_gmem1_RREADY;
wire    grp_replay_vit_delta_order_fu_196_m_axi_gmem1_BREADY;
wire    grp_replay_vit_delta_order_fu_196_x_stream_TREADY;
wire   [199:0] grp_replay_vit_delta_order_fu_196_x_stream_TDATA;
wire    grp_replay_vit_delta_order_fu_196_x_stream_TVALID;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_done;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_idle;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_ready;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWVALID;
wire   [63:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWADDR;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWID;
wire   [31:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWLEN;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWBURST;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWPROT;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWQOS;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWREGION;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WVALID;
wire   [127:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WDATA;
wire   [15:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WSTRB;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WLAST;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WID;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARVALID;
wire   [63:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARADDR;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARID;
wire   [31:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARLEN;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARBURST;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARPROT;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARQOS;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARREGION;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_RREADY;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_BREADY;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_y_stream_TREADY;
wire    grp_replay_token_major_fu_215_ap_start;
wire    grp_replay_token_major_fu_215_ap_done;
wire    grp_replay_token_major_fu_215_ap_idle;
wire    grp_replay_token_major_fu_215_ap_ready;
wire    grp_replay_token_major_fu_215_m_axi_gmem1_AWVALID;
wire   [63:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWADDR;
wire   [0:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWID;
wire   [31:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWLEN;
wire   [2:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWBURST;
wire   [1:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWPROT;
wire   [3:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWQOS;
wire   [3:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWREGION;
wire   [0:0] grp_replay_token_major_fu_215_m_axi_gmem1_AWUSER;
wire    grp_replay_token_major_fu_215_m_axi_gmem1_WVALID;
wire   [127:0] grp_replay_token_major_fu_215_m_axi_gmem1_WDATA;
wire   [15:0] grp_replay_token_major_fu_215_m_axi_gmem1_WSTRB;
wire    grp_replay_token_major_fu_215_m_axi_gmem1_WLAST;
wire   [0:0] grp_replay_token_major_fu_215_m_axi_gmem1_WID;
wire   [0:0] grp_replay_token_major_fu_215_m_axi_gmem1_WUSER;
wire    grp_replay_token_major_fu_215_m_axi_gmem1_ARVALID;
wire   [63:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARADDR;
wire   [0:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARID;
wire   [31:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARLEN;
wire   [2:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARBURST;
wire   [1:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARPROT;
wire   [3:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARQOS;
wire   [3:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARREGION;
wire   [0:0] grp_replay_token_major_fu_215_m_axi_gmem1_ARUSER;
wire    grp_replay_token_major_fu_215_m_axi_gmem1_RREADY;
wire    grp_replay_token_major_fu_215_m_axi_gmem1_BREADY;
wire   [199:0] grp_replay_token_major_fu_215_x_stream_TDATA;
wire    grp_replay_token_major_fu_215_x_stream_TVALID;
wire    grp_replay_token_major_fu_215_x_stream_TREADY;
wire    grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWVALID;
wire   [63:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWADDR;
wire   [0:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWID;
wire   [31:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWLEN;
wire   [2:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWBURST;
wire   [1:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWPROT;
wire   [3:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWQOS;
wire   [3:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWREGION;
wire   [0:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWUSER;
wire    grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WVALID;
wire   [127:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WDATA;
wire   [15:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WSTRB;
wire    grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WLAST;
wire   [0:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WID;
wire   [0:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WUSER;
wire    grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARVALID;
wire   [63:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARADDR;
wire   [0:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARID;
wire   [31:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARLEN;
wire   [2:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARBURST;
wire   [1:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARPROT;
wire   [3:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARQOS;
wire   [3:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARREGION;
wire   [0:0] grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARUSER;
wire    grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_RREADY;
wire    grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_BREADY;
wire   [199:0] grp_replay_writeback_llm_delta_order_fu_224_x_stream_TDATA;
wire    grp_replay_writeback_llm_delta_order_fu_224_x_stream_TVALID;
wire    grp_replay_writeback_llm_delta_order_fu_224_x_stream_TREADY;
wire    grp_replay_writeback_llm_delta_order_fu_224_ap_start;
wire    grp_replay_writeback_llm_delta_order_fu_224_ap_done;
wire    grp_replay_writeback_llm_delta_order_fu_224_y_stream_TREADY;
wire    grp_replay_writeback_llm_delta_order_fu_224_ap_ready;
wire    grp_replay_writeback_llm_delta_order_fu_224_ap_idle;
reg    grp_replay_writeback_llm_delta_order_fu_224_ap_continue;
wire    grp_replay_token_major_1_fu_235_ap_start;
wire    grp_replay_token_major_1_fu_235_ap_done;
wire    grp_replay_token_major_1_fu_235_ap_idle;
wire    grp_replay_token_major_1_fu_235_ap_ready;
wire    grp_replay_token_major_1_fu_235_m_axi_gmem1_AWVALID;
wire   [63:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWADDR;
wire   [0:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWID;
wire   [31:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWLEN;
wire   [2:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWBURST;
wire   [1:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWPROT;
wire   [3:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWQOS;
wire   [3:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWREGION;
wire   [0:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_AWUSER;
wire    grp_replay_token_major_1_fu_235_m_axi_gmem1_WVALID;
wire   [127:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_WDATA;
wire   [15:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_WSTRB;
wire    grp_replay_token_major_1_fu_235_m_axi_gmem1_WLAST;
wire   [0:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_WID;
wire   [0:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_WUSER;
wire    grp_replay_token_major_1_fu_235_m_axi_gmem1_ARVALID;
wire   [63:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARADDR;
wire   [0:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARID;
wire   [31:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARLEN;
wire   [2:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARBURST;
wire   [1:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARPROT;
wire   [3:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARQOS;
wire   [3:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARREGION;
wire   [0:0] grp_replay_token_major_1_fu_235_m_axi_gmem1_ARUSER;
wire    grp_replay_token_major_1_fu_235_m_axi_gmem1_RREADY;
wire    grp_replay_token_major_1_fu_235_m_axi_gmem1_BREADY;
wire   [199:0] grp_replay_token_major_1_fu_235_x_stream_TDATA;
wire    grp_replay_token_major_1_fu_235_x_stream_TVALID;
wire    grp_replay_token_major_1_fu_235_x_stream_TREADY;
wire    grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWVALID;
wire   [63:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWADDR;
wire   [0:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWID;
wire   [31:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWLEN;
wire   [2:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWBURST;
wire   [1:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWPROT;
wire   [3:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWQOS;
wire   [3:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWREGION;
wire   [0:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWUSER;
wire    grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WVALID;
wire   [127:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WDATA;
wire   [15:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WSTRB;
wire    grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WLAST;
wire   [0:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WID;
wire   [0:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WUSER;
wire    grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARVALID;
wire   [63:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARADDR;
wire   [0:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARID;
wire   [31:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARLEN;
wire   [2:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARBURST;
wire   [1:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARPROT;
wire   [3:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARQOS;
wire   [3:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARREGION;
wire   [0:0] grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARUSER;
wire    grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_RREADY;
wire    grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_BREADY;
wire   [199:0] grp_replay_writeback_vit_delta_order_fu_244_x_stream_TDATA;
wire    grp_replay_writeback_vit_delta_order_fu_244_x_stream_TVALID;
wire    grp_replay_writeback_vit_delta_order_fu_244_x_stream_TREADY;
wire    grp_replay_writeback_vit_delta_order_fu_244_ap_start;
wire    grp_replay_writeback_vit_delta_order_fu_244_ap_done;
wire    grp_replay_writeback_vit_delta_order_fu_244_y_stream_TREADY;
wire    grp_replay_writeback_vit_delta_order_fu_244_ap_ready;
wire    grp_replay_writeback_vit_delta_order_fu_244_ap_idle;
reg    grp_replay_writeback_vit_delta_order_fu_244_ap_continue;
wire    grp_writeback_vit_delta_order_fu_255_ap_start;
wire    grp_writeback_vit_delta_order_fu_255_ap_done;
wire    grp_writeback_vit_delta_order_fu_255_ap_idle;
wire    grp_writeback_vit_delta_order_fu_255_ap_ready;
wire    grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWVALID;
wire   [63:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWADDR;
wire   [0:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWID;
wire   [31:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWLEN;
wire   [2:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWBURST;
wire   [1:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWPROT;
wire   [3:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWQOS;
wire   [3:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWREGION;
wire   [0:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWUSER;
wire    grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WVALID;
wire   [127:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WDATA;
wire   [15:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WSTRB;
wire    grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WLAST;
wire   [0:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WID;
wire   [0:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WUSER;
wire    grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARVALID;
wire   [63:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARADDR;
wire   [0:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARID;
wire   [31:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARLEN;
wire   [2:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARBURST;
wire   [1:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARPROT;
wire   [3:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARQOS;
wire   [3:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARREGION;
wire   [0:0] grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARUSER;
wire    grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_RREADY;
wire    grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_BREADY;
wire    grp_writeback_vit_delta_order_fu_255_y_stream_TREADY;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_done;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_idle;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_ready;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWVALID;
wire   [63:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWADDR;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWID;
wire   [31:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWLEN;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWBURST;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWPROT;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWQOS;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWREGION;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WVALID;
wire   [127:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WDATA;
wire   [15:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WSTRB;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WLAST;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WID;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARVALID;
wire   [63:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARADDR;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARID;
wire   [31:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARLEN;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARBURST;
wire   [1:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARPROT;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARQOS;
wire   [3:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARREGION;
wire   [0:0] grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARUSER;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_RREADY;
wire    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_BREADY;
reg    gmem1_AWVALID;
wire    gmem1_AWREADY;
reg   [63:0] gmem1_AWADDR;
reg   [31:0] gmem1_AWLEN;
reg    gmem1_WVALID;
wire    gmem1_WREADY;
reg   [127:0] gmem1_WDATA;
reg   [15:0] gmem1_WSTRB;
reg    gmem1_ARVALID;
wire    gmem1_ARREADY;
reg   [63:0] gmem1_ARADDR;
reg   [31:0] gmem1_ARLEN;
wire    gmem1_RVALID;
reg    gmem1_RREADY;
wire   [127:0] gmem1_RDATA;
wire   [8:0] gmem1_RFIFONUM;
wire    gmem1_BVALID;
reg    gmem1_BREADY;
reg    grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start_reg;
reg    ap_block_state1_ignore_call1;
wire   [0:0] icmp_ln250_fu_278_p2;
wire   [0:0] and_ln55_fu_312_p2;
wire    ap_CS_fsm_state4;
reg    grp_replay_vit_delta_order_fu_196_ap_start_reg;
reg    ap_block_state1_ignore_call0;
wire    ap_CS_fsm_state11;
reg    grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start_reg;
wire    ap_CS_fsm_state2;
wire    ap_CS_fsm_state3;
reg    grp_replay_token_major_fu_215_ap_start_reg;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state7;
wire    ap_CS_fsm_state8;
reg    grp_replay_writeback_llm_delta_order_fu_224_ap_start_reg;
wire    ap_CS_fsm_state9;
wire    ap_CS_fsm_state10;
wire    ap_sync_grp_replay_writeback_llm_delta_order_fu_224_ap_ready;
wire    ap_sync_grp_replay_writeback_llm_delta_order_fu_224_ap_done;
reg    ap_block_state10_on_subcall_done;
reg    ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_ready;
reg    ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_done;
reg    grp_replay_token_major_1_fu_235_ap_start_reg;
wire    ap_CS_fsm_state12;
wire    ap_CS_fsm_state15;
wire    ap_CS_fsm_state13;
wire    ap_CS_fsm_state16;
reg    grp_replay_writeback_vit_delta_order_fu_244_ap_start_reg;
wire    ap_CS_fsm_state17;
wire    ap_CS_fsm_state18;
wire    ap_sync_grp_replay_writeback_vit_delta_order_fu_244_ap_ready;
wire    ap_sync_grp_replay_writeback_vit_delta_order_fu_244_ap_done;
reg    ap_block_state18_on_subcall_done;
reg    ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_ready;
reg    ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_done;
reg    grp_writeback_vit_delta_order_fu_255_ap_start_reg;
wire    ap_CS_fsm_state19;
reg    grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start_reg;
wire    ap_CS_fsm_state20;
wire    ap_CS_fsm_state21;
reg   [1:0] pass_2_fu_130;
wire   [1:0] pass_6_fu_355_p2;
wire    ap_CS_fsm_state6;
reg    ap_predicate_op79_call_state3;
reg    ap_predicate_op81_call_state3;
reg    ap_block_state3_on_subcall_done;
wire   [0:0] icmp_ln184_fu_349_p2;
reg   [1:0] pass_fu_134;
wire   [1:0] pass_4_fu_380_p2;
wire    ap_CS_fsm_state14;
reg    ap_predicate_op117_call_state13;
reg    ap_block_state13_on_subcall_done;
wire   [0:0] icmp_ln196_fu_374_p2;
wire   [0:0] tmp_fu_292_p3;
wire   [31:0] select_ln55_fu_284_p3;
wire   [0:0] icmp_ln55_fu_306_p2;
wire   [0:0] xor_ln55_fu_300_p2;
wire   [26:0] tmp_1_fu_324_p4;
wire   [0:0] icmp_ln265_fu_318_p2;
wire    ap_CS_fsm_state22;
wire    regslice_both_x_stream_U_apdone_blk;
reg   [21:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
reg    ap_ST_fsm_state8_blk;
wire    ap_ST_fsm_state9_blk;
reg    ap_ST_fsm_state10_blk;
reg    ap_ST_fsm_state11_blk;
wire    ap_ST_fsm_state12_blk;
reg    ap_ST_fsm_state13_blk;
wire    ap_ST_fsm_state14_blk;
wire    ap_ST_fsm_state15_blk;
reg    ap_ST_fsm_state16_blk;
wire    ap_ST_fsm_state17_blk;
reg    ap_ST_fsm_state18_blk;
wire    ap_ST_fsm_state19_blk;
wire    ap_ST_fsm_state20_blk;
reg    ap_block_state21_on_subcall_done;
reg    ap_ST_fsm_state21_blk;
reg    ap_ST_fsm_state22_blk;
reg   [199:0] x_stream_TDATA_int_regslice;
reg    x_stream_TVALID_int_regslice;
wire    x_stream_TREADY_int_regslice;
wire    regslice_both_x_stream_U_vld_out;
wire    regslice_both_y_stream_U_apdone_blk;
wire   [199:0] y_stream_TDATA_int_regslice;
wire    y_stream_TVALID_int_regslice;
reg    y_stream_TREADY_int_regslice;
wire    regslice_both_y_stream_U_ack_in;
wire    regslice_both_cls_stream_U_apdone_blk;
wire   [255:0] cls_stream_TDATA_int_regslice;
wire    cls_stream_TVALID_int_regslice;
reg    cls_stream_TREADY_int_regslice;
wire    regslice_both_cls_stream_U_ack_in;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 22'd1;
//#0 grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start_reg = 1'b0;
//#0 grp_replay_vit_delta_order_fu_196_ap_start_reg = 1'b0;
//#0 grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start_reg = 1'b0;
//#0 grp_replay_token_major_fu_215_ap_start_reg = 1'b0;
//#0 grp_replay_writeback_llm_delta_order_fu_224_ap_start_reg = 1'b0;
//#0 ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_ready = 1'b0;
//#0 ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_done = 1'b0;
//#0 grp_replay_token_major_1_fu_235_ap_start_reg = 1'b0;
//#0 grp_replay_writeback_vit_delta_order_fu_244_ap_start_reg = 1'b0;
//#0 ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_ready = 1'b0;
//#0 ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_done = 1'b0;
//#0 grp_writeback_vit_delta_order_fu_255_ap_start_reg = 1'b0;
//#0 grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start_reg = 1'b0;
//#0 pass_2_fu_130 = 2'd0;
//#0 pass_fu_134 = 2'd0;
end

STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2 grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start),
    .ap_done(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_done),
    .ap_idle(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_idle),
    .ap_ready(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_ready),
    .m_axi_gmem1_AWVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(gmem1_RDATA),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .x_stream_TREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TREADY),
    .memory_decoder_state(memory_decoder_state),
    .x_stream_TDATA(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TDATA),
    .x_stream_TVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TVALID)
);

STATE_AXI_replay_vit_delta_order grp_replay_vit_delta_order_fu_196(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_replay_vit_delta_order_fu_196_ap_start),
    .ap_done(grp_replay_vit_delta_order_fu_196_ap_done),
    .ap_idle(grp_replay_vit_delta_order_fu_196_ap_idle),
    .ap_ready(grp_replay_vit_delta_order_fu_196_ap_ready),
    .m_axi_gmem1_AWVALID(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(gmem1_RDATA),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(grp_replay_vit_delta_order_fu_196_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .x_stream_TREADY(grp_replay_vit_delta_order_fu_196_x_stream_TREADY),
    .memory_state(memory_vit_state),
    .x_stream_TDATA(grp_replay_vit_delta_order_fu_196_x_stream_TDATA),
    .x_stream_TVALID(grp_replay_vit_delta_order_fu_196_x_stream_TVALID)
);

STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2 grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start),
    .ap_done(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_done),
    .ap_idle(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_idle),
    .ap_ready(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_ready),
    .y_stream_TVALID(y_stream_TVALID_int_regslice),
    .m_axi_gmem1_AWVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(9'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_decoder_state(memory_decoder_state),
    .y_stream_TDATA(y_stream_TDATA_int_regslice),
    .y_stream_TREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_y_stream_TREADY)
);

STATE_AXI_replay_token_major grp_replay_token_major_fu_215(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_replay_token_major_fu_215_ap_start),
    .ap_done(grp_replay_token_major_fu_215_ap_done),
    .ap_idle(grp_replay_token_major_fu_215_ap_idle),
    .ap_ready(grp_replay_token_major_fu_215_ap_ready),
    .m_axi_gmem1_AWVALID(grp_replay_token_major_fu_215_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(grp_replay_token_major_fu_215_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_replay_token_major_fu_215_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_replay_token_major_fu_215_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_replay_token_major_fu_215_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_replay_token_major_fu_215_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_replay_token_major_fu_215_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_replay_token_major_fu_215_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_replay_token_major_fu_215_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_replay_token_major_fu_215_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_replay_token_major_fu_215_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_replay_token_major_fu_215_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_replay_token_major_fu_215_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(grp_replay_token_major_fu_215_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_replay_token_major_fu_215_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_replay_token_major_fu_215_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_replay_token_major_fu_215_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_replay_token_major_fu_215_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_replay_token_major_fu_215_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_replay_token_major_fu_215_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_replay_token_major_fu_215_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_replay_token_major_fu_215_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_replay_token_major_fu_215_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_replay_token_major_fu_215_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_replay_token_major_fu_215_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_replay_token_major_fu_215_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_replay_token_major_fu_215_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_replay_token_major_fu_215_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_replay_token_major_fu_215_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_replay_token_major_fu_215_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_replay_token_major_fu_215_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(gmem1_RDATA),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(grp_replay_token_major_fu_215_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_state(memory_decoder_state),
    .x_stream_TDATA(grp_replay_token_major_fu_215_x_stream_TDATA),
    .x_stream_TVALID(grp_replay_token_major_fu_215_x_stream_TVALID),
    .x_stream_TREADY(grp_replay_token_major_fu_215_x_stream_TREADY)
);

STATE_AXI_replay_writeback_llm_delta_order grp_replay_writeback_llm_delta_order_fu_224(
    .m_axi_gmem1_AWVALID(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(gmem1_RDATA),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_state(memory_decoder_state),
    .x_stream_TDATA(grp_replay_writeback_llm_delta_order_fu_224_x_stream_TDATA),
    .y_stream_TDATA(y_stream_TDATA_int_regslice),
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .memory_state_ap_vld(1'b1),
    .x_stream_TVALID(grp_replay_writeback_llm_delta_order_fu_224_x_stream_TVALID),
    .x_stream_TREADY(grp_replay_writeback_llm_delta_order_fu_224_x_stream_TREADY),
    .ap_start(grp_replay_writeback_llm_delta_order_fu_224_ap_start),
    .ap_done(grp_replay_writeback_llm_delta_order_fu_224_ap_done),
    .y_stream_TVALID(y_stream_TVALID_int_regslice),
    .y_stream_TREADY(grp_replay_writeback_llm_delta_order_fu_224_y_stream_TREADY),
    .ap_ready(grp_replay_writeback_llm_delta_order_fu_224_ap_ready),
    .ap_idle(grp_replay_writeback_llm_delta_order_fu_224_ap_idle),
    .ap_continue(grp_replay_writeback_llm_delta_order_fu_224_ap_continue)
);

STATE_AXI_replay_token_major_1 grp_replay_token_major_1_fu_235(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_replay_token_major_1_fu_235_ap_start),
    .ap_done(grp_replay_token_major_1_fu_235_ap_done),
    .ap_idle(grp_replay_token_major_1_fu_235_ap_idle),
    .ap_ready(grp_replay_token_major_1_fu_235_ap_ready),
    .m_axi_gmem1_AWVALID(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_replay_token_major_1_fu_235_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_replay_token_major_1_fu_235_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(grp_replay_token_major_1_fu_235_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_replay_token_major_1_fu_235_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_replay_token_major_1_fu_235_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_replay_token_major_1_fu_235_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_replay_token_major_1_fu_235_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_replay_token_major_1_fu_235_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_replay_token_major_1_fu_235_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(gmem1_RDATA),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(grp_replay_token_major_1_fu_235_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_state(memory_vit_state),
    .x_stream_TDATA(grp_replay_token_major_1_fu_235_x_stream_TDATA),
    .x_stream_TVALID(grp_replay_token_major_1_fu_235_x_stream_TVALID),
    .x_stream_TREADY(grp_replay_token_major_1_fu_235_x_stream_TREADY)
);

STATE_AXI_replay_writeback_vit_delta_order grp_replay_writeback_vit_delta_order_fu_244(
    .m_axi_gmem1_AWVALID(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(gmem1_RDATA),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_state(memory_vit_state),
    .x_stream_TDATA(grp_replay_writeback_vit_delta_order_fu_244_x_stream_TDATA),
    .y_stream_TDATA(y_stream_TDATA_int_regslice),
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .memory_state_ap_vld(1'b1),
    .x_stream_TVALID(grp_replay_writeback_vit_delta_order_fu_244_x_stream_TVALID),
    .x_stream_TREADY(grp_replay_writeback_vit_delta_order_fu_244_x_stream_TREADY),
    .ap_start(grp_replay_writeback_vit_delta_order_fu_244_ap_start),
    .ap_done(grp_replay_writeback_vit_delta_order_fu_244_ap_done),
    .y_stream_TVALID(y_stream_TVALID_int_regslice),
    .y_stream_TREADY(grp_replay_writeback_vit_delta_order_fu_244_y_stream_TREADY),
    .ap_ready(grp_replay_writeback_vit_delta_order_fu_244_ap_ready),
    .ap_idle(grp_replay_writeback_vit_delta_order_fu_244_ap_idle),
    .ap_continue(grp_replay_writeback_vit_delta_order_fu_244_ap_continue)
);

STATE_AXI_writeback_vit_delta_order grp_writeback_vit_delta_order_fu_255(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_writeback_vit_delta_order_fu_255_ap_start),
    .ap_done(grp_writeback_vit_delta_order_fu_255_ap_done),
    .ap_idle(grp_writeback_vit_delta_order_fu_255_ap_idle),
    .ap_ready(grp_writeback_vit_delta_order_fu_255_ap_ready),
    .y_stream_TVALID(y_stream_TVALID_int_regslice),
    .m_axi_gmem1_AWVALID(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(9'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_state(memory_vit_state),
    .y_stream_TDATA(y_stream_TDATA_int_regslice),
    .y_stream_TREADY(grp_writeback_vit_delta_order_fu_255_y_stream_TREADY)
);

STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_208_1 grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start),
    .ap_done(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_done),
    .ap_idle(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_idle),
    .ap_ready(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_ready),
    .m_axi_gmem1_AWVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(9'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .empty(cls_stream_read_reg_442),
    .memory_cls_y(memory_cls_y)
);

STATE_AXI_gmem1_m_axi #(
    .CONSERVATIVE( 1 ),
    .USER_MAXREQS( 7 ),
    .MAX_READ_BURST_LENGTH( 16 ),
    .MAX_WRITE_BURST_LENGTH( 16 ),
    .C_M_AXI_ID_WIDTH( C_M_AXI_GMEM1_ID_WIDTH ),
    .C_M_AXI_ADDR_WIDTH( C_M_AXI_GMEM1_ADDR_WIDTH ),
    .C_M_AXI_DATA_WIDTH( C_M_AXI_GMEM1_DATA_WIDTH ),
    .C_M_AXI_AWUSER_WIDTH( C_M_AXI_GMEM1_AWUSER_WIDTH ),
    .C_M_AXI_ARUSER_WIDTH( C_M_AXI_GMEM1_ARUSER_WIDTH ),
    .C_M_AXI_WUSER_WIDTH( C_M_AXI_GMEM1_WUSER_WIDTH ),
    .C_M_AXI_RUSER_WIDTH( C_M_AXI_GMEM1_RUSER_WIDTH ),
    .C_M_AXI_BUSER_WIDTH( C_M_AXI_GMEM1_BUSER_WIDTH ),
    .C_USER_VALUE( C_M_AXI_GMEM1_USER_VALUE ),
    .C_PROT_VALUE( C_M_AXI_GMEM1_PROT_VALUE ),
    .C_CACHE_VALUE( C_M_AXI_GMEM1_CACHE_VALUE ),
    .USER_RFIFONUM_WIDTH( 9 ),
    .USER_DW( 128 ),
    .USER_AW( 64 ),
    .NUM_READ_OUTSTANDING( 16 ),
    .NUM_WRITE_OUTSTANDING( 16 ))
gmem1_m_axi_U(
    .AWVALID(m_axi_gmem1_AWVALID),
    .AWREADY(m_axi_gmem1_AWREADY),
    .AWADDR(m_axi_gmem1_AWADDR),
    .AWID(m_axi_gmem1_AWID),
    .AWLEN(m_axi_gmem1_AWLEN),
    .AWSIZE(m_axi_gmem1_AWSIZE),
    .AWBURST(m_axi_gmem1_AWBURST),
    .AWLOCK(m_axi_gmem1_AWLOCK),
    .AWCACHE(m_axi_gmem1_AWCACHE),
    .AWPROT(m_axi_gmem1_AWPROT),
    .AWQOS(m_axi_gmem1_AWQOS),
    .AWREGION(m_axi_gmem1_AWREGION),
    .AWUSER(m_axi_gmem1_AWUSER),
    .WVALID(m_axi_gmem1_WVALID),
    .WREADY(m_axi_gmem1_WREADY),
    .WDATA(m_axi_gmem1_WDATA),
    .WSTRB(m_axi_gmem1_WSTRB),
    .WLAST(m_axi_gmem1_WLAST),
    .WID(m_axi_gmem1_WID),
    .WUSER(m_axi_gmem1_WUSER),
    .ARVALID(m_axi_gmem1_ARVALID),
    .ARREADY(m_axi_gmem1_ARREADY),
    .ARADDR(m_axi_gmem1_ARADDR),
    .ARID(m_axi_gmem1_ARID),
    .ARLEN(m_axi_gmem1_ARLEN),
    .ARSIZE(m_axi_gmem1_ARSIZE),
    .ARBURST(m_axi_gmem1_ARBURST),
    .ARLOCK(m_axi_gmem1_ARLOCK),
    .ARCACHE(m_axi_gmem1_ARCACHE),
    .ARPROT(m_axi_gmem1_ARPROT),
    .ARQOS(m_axi_gmem1_ARQOS),
    .ARREGION(m_axi_gmem1_ARREGION),
    .ARUSER(m_axi_gmem1_ARUSER),
    .RVALID(m_axi_gmem1_RVALID),
    .RREADY(m_axi_gmem1_RREADY),
    .RDATA(m_axi_gmem1_RDATA),
    .RLAST(m_axi_gmem1_RLAST),
    .RID(m_axi_gmem1_RID),
    .RUSER(m_axi_gmem1_RUSER),
    .RRESP(m_axi_gmem1_RRESP),
    .BVALID(m_axi_gmem1_BVALID),
    .BREADY(m_axi_gmem1_BREADY),
    .BRESP(m_axi_gmem1_BRESP),
    .BID(m_axi_gmem1_BID),
    .BUSER(m_axi_gmem1_BUSER),
    .ACLK(ap_clk),
    .ARESET(ap_rst_n_inv),
    .ACLK_EN(1'b1),
    .I_ARVALID(gmem1_ARVALID),
    .I_ARREADY(gmem1_ARREADY),
    .I_ARADDR(gmem1_ARADDR),
    .I_ARLEN(gmem1_ARLEN),
    .I_RVALID(gmem1_RVALID),
    .I_RREADY(gmem1_RREADY),
    .I_RDATA(gmem1_RDATA),
    .I_RFIFONUM(gmem1_RFIFONUM),
    .I_AWVALID(gmem1_AWVALID),
    .I_AWREADY(gmem1_AWREADY),
    .I_AWADDR(gmem1_AWADDR),
    .I_AWLEN(gmem1_AWLEN),
    .I_WVALID(gmem1_WVALID),
    .I_WREADY(gmem1_WREADY),
    .I_WDATA(gmem1_WDATA),
    .I_WSTRB(gmem1_WSTRB),
    .I_BVALID(gmem1_BVALID),
    .I_BREADY(gmem1_BREADY)
);

STATE_AXI_regslice_both #(
    .DataWidth( 200 ))
regslice_both_x_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(x_stream_TDATA_int_regslice),
    .vld_in(x_stream_TVALID_int_regslice),
    .ack_in(x_stream_TREADY_int_regslice),
    .data_out(x_stream_TDATA),
    .vld_out(regslice_both_x_stream_U_vld_out),
    .ack_out(x_stream_TREADY),
    .apdone_blk(regslice_both_x_stream_U_apdone_blk)
);

STATE_AXI_regslice_both #(
    .DataWidth( 200 ))
regslice_both_y_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(y_stream_TDATA),
    .vld_in(y_stream_TVALID),
    .ack_in(regslice_both_y_stream_U_ack_in),
    .data_out(y_stream_TDATA_int_regslice),
    .vld_out(y_stream_TVALID_int_regslice),
    .ack_out(y_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_y_stream_U_apdone_blk)
);

STATE_AXI_regslice_both #(
    .DataWidth( 256 ))
regslice_both_cls_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(cls_stream_TDATA),
    .vld_in(cls_stream_TVALID),
    .ack_in(regslice_both_cls_stream_U_ack_in),
    .data_out(cls_stream_TDATA_int_regslice),
    .vld_out(cls_stream_TVALID_int_regslice),
    .ack_out(cls_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_cls_stream_U_apdone_blk)
);

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_state22) & (regslice_both_x_stream_U_apdone_blk == 1'b0))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_done <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state10) & (1'b0 == ap_block_state10_on_subcall_done))) begin
            ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_done <= 1'b0;
        end else if ((grp_replay_writeback_llm_delta_order_fu_224_ap_done == 1'b1)) begin
            ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_done <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_ready <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state10) & (1'b0 == ap_block_state10_on_subcall_done))) begin
            ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_ready <= 1'b0;
        end else if ((grp_replay_writeback_llm_delta_order_fu_224_ap_ready == 1'b1)) begin
            ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_ready <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_done <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state18) & (1'b0 == ap_block_state18_on_subcall_done))) begin
            ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_done <= 1'b0;
        end else if ((grp_replay_writeback_vit_delta_order_fu_244_ap_done == 1'b1)) begin
            ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_done <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_ready <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state18) & (1'b0 == ap_block_state18_on_subcall_done))) begin
            ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_ready <= 1'b0;
        end else if ((grp_replay_writeback_vit_delta_order_fu_244_ap_ready == 1'b1)) begin
            ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_ready <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_replay_token_major_1_fu_235_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state15) | (1'b1 == ap_CS_fsm_state12))) begin
            grp_replay_token_major_1_fu_235_ap_start_reg <= 1'b1;
        end else if ((grp_replay_token_major_1_fu_235_ap_ready == 1'b1)) begin
            grp_replay_token_major_1_fu_235_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_replay_token_major_fu_215_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state7) | (1'b1 == ap_CS_fsm_state5))) begin
            grp_replay_token_major_fu_215_ap_start_reg <= 1'b1;
        end else if ((grp_replay_token_major_fu_215_ap_ready == 1'b1)) begin
            grp_replay_token_major_fu_215_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_replay_vit_delta_order_fu_196_ap_start_reg <= 1'b0;
    end else begin
        if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd1) & (1'b0 == ap_block_state1_ignore_call0) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
            grp_replay_vit_delta_order_fu_196_ap_start_reg <= 1'b1;
        end else if ((grp_replay_vit_delta_order_fu_196_ap_ready == 1'b1)) begin
            grp_replay_vit_delta_order_fu_196_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_replay_writeback_llm_delta_order_fu_224_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state9) | ((ap_sync_grp_replay_writeback_llm_delta_order_fu_224_ap_ready == 1'b0) & (1'b1 == ap_CS_fsm_state10)))) begin
            grp_replay_writeback_llm_delta_order_fu_224_ap_start_reg <= 1'b1;
        end else if ((grp_replay_writeback_llm_delta_order_fu_224_ap_ready == 1'b1)) begin
            grp_replay_writeback_llm_delta_order_fu_224_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_replay_writeback_vit_delta_order_fu_244_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state17) | ((ap_sync_grp_replay_writeback_vit_delta_order_fu_244_ap_ready == 1'b0) & (1'b1 == ap_CS_fsm_state18)))) begin
            grp_replay_writeback_vit_delta_order_fu_244_ap_start_reg <= 1'b1;
        end else if ((grp_replay_writeback_vit_delta_order_fu_244_ap_ready == 1'b1)) begin
            grp_replay_writeback_vit_delta_order_fu_244_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start_reg <= 1'b1;
        end else if ((grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_ready == 1'b1)) begin
            grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state20)) begin
            grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start_reg <= 1'b1;
        end else if ((grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_ready == 1'b1)) begin
            grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start_reg <= 1'b0;
    end else begin
        if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (icmp_ln265_1_fu_334_p2 == 1'd1) & (op == 3'd1) & (1'b0 == ap_block_state1_ignore_call1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
            grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start_reg <= 1'b1;
        end else if ((grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_ready == 1'b1)) begin
            grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_writeback_vit_delta_order_fu_255_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state19)) begin
            grp_writeback_vit_delta_order_fu_255_ap_start_reg <= 1'b1;
        end else if ((grp_writeback_vit_delta_order_fu_255_ap_ready == 1'b1)) begin
            grp_writeback_vit_delta_order_fu_255_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((icmp_ln184_fu_349_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3) & (1'b0 == ap_block_state3_on_subcall_done) & (1'd1 == and_ln265_reg_438))) begin
        pass_2_fu_130 <= pass_6_fu_355_p2;
    end else if ((1'b1 == ap_CS_fsm_state6)) begin
        pass_2_fu_130 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state13) & (op == 3'd4) & (1'b0 == ap_block_state13_on_subcall_done) & (icmp_ln196_fu_374_p2 == 1'd0))) begin
        pass_fu_134 <= pass_4_fu_380_p2;
    end else if ((1'b1 == ap_CS_fsm_state14)) begin
        pass_fu_134 <= 2'd0;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
        and_ln265_reg_438 <= and_ln265_fu_340_p2;
        cls_stream_read_reg_442 <= cls_stream_TDATA_int_regslice;
        icmp_ln265_1_reg_434 <= icmp_ln265_1_fu_334_p2;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state10_on_subcall_done)) begin
        ap_ST_fsm_state10_blk = 1'b1;
    end else begin
        ap_ST_fsm_state10_blk = 1'b0;
    end
end

always @ (*) begin
    if ((grp_replay_vit_delta_order_fu_196_ap_done == 1'b0)) begin
        ap_ST_fsm_state11_blk = 1'b1;
    end else begin
        ap_ST_fsm_state11_blk = 1'b0;
    end
end

assign ap_ST_fsm_state12_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state13_on_subcall_done)) begin
        ap_ST_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_fsm_state13_blk = 1'b0;
    end
end

assign ap_ST_fsm_state14_blk = 1'b0;

assign ap_ST_fsm_state15_blk = 1'b0;

always @ (*) begin
    if ((grp_replay_token_major_1_fu_235_ap_done == 1'b0)) begin
        ap_ST_fsm_state16_blk = 1'b1;
    end else begin
        ap_ST_fsm_state16_blk = 1'b0;
    end
end

assign ap_ST_fsm_state17_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state18_on_subcall_done)) begin
        ap_ST_fsm_state18_blk = 1'b1;
    end else begin
        ap_ST_fsm_state18_blk = 1'b0;
    end
end

assign ap_ST_fsm_state19_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state1)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state20_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state21_on_subcall_done)) begin
        ap_ST_fsm_state21_blk = 1'b1;
    end else begin
        ap_ST_fsm_state21_blk = 1'b0;
    end
end

always @ (*) begin
    if ((regslice_both_x_stream_U_apdone_blk == 1'b1)) begin
        ap_ST_fsm_state22_blk = 1'b1;
    end else begin
        ap_ST_fsm_state22_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state3_on_subcall_done)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if ((grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_done == 1'b0)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

assign ap_ST_fsm_state7_blk = 1'b0;

always @ (*) begin
    if ((grp_replay_token_major_fu_215_ap_done == 1'b0)) begin
        ap_ST_fsm_state8_blk = 1'b1;
    end else begin
        ap_ST_fsm_state8_blk = 1'b0;
    end
end

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state22) & (regslice_both_x_stream_U_apdone_blk == 1'b0))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state22) & (regslice_both_x_stream_U_apdone_blk == 1'b0))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_done_reg == 1'b1) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1) & (mode == 1'd0) & (icmp_ln243_fu_272_p2 == 1'd1))) begin
        cls_stream_TDATA_blk_n = cls_stream_TVALID_int_regslice;
    end else begin
        cls_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1) & (ap_predicate_op76_read_state1 == 1'b1))) begin
        cls_stream_TREADY_int_regslice = 1'b1;
    end else begin
        cls_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_ARADDR = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARADDR;
    end else if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state15) | (1'b1 == ap_CS_fsm_state12) | ((1'b1 == ap_CS_fsm_state13) & (op == 3'd0)))) begin
        gmem1_ARADDR = grp_replay_token_major_1_fu_235_m_axi_gmem1_ARADDR;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_ARADDR = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARADDR;
    end else if (((1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state7) | (1'b1 == ap_CS_fsm_state5) | ((1'b1 == ap_CS_fsm_state3) & (op == 3'd0) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_ARADDR = grp_replay_token_major_fu_215_m_axi_gmem1_ARADDR;
    end else if (((1'b1 == ap_CS_fsm_state11) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_ARADDR = grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARADDR;
    end else if (((1'b1 == ap_CS_fsm_state4) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (icmp_ln265_1_fu_334_p2 == 1'd1) & (op == 3'd1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_ARADDR = grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARADDR;
    end else begin
        gmem1_ARADDR = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_ARLEN = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARLEN;
    end else if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state15) | (1'b1 == ap_CS_fsm_state12) | ((1'b1 == ap_CS_fsm_state13) & (op == 3'd0)))) begin
        gmem1_ARLEN = grp_replay_token_major_1_fu_235_m_axi_gmem1_ARLEN;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_ARLEN = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARLEN;
    end else if (((1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state7) | (1'b1 == ap_CS_fsm_state5) | ((1'b1 == ap_CS_fsm_state3) & (op == 3'd0) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_ARLEN = grp_replay_token_major_fu_215_m_axi_gmem1_ARLEN;
    end else if (((1'b1 == ap_CS_fsm_state11) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_ARLEN = grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARLEN;
    end else if (((1'b1 == ap_CS_fsm_state4) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (icmp_ln265_1_fu_334_p2 == 1'd1) & (op == 3'd1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_ARLEN = grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARLEN;
    end else begin
        gmem1_ARLEN = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_ARVALID = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_ARVALID;
    end else if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state15) | (1'b1 == ap_CS_fsm_state12) | ((1'b1 == ap_CS_fsm_state13) & (op == 3'd0)))) begin
        gmem1_ARVALID = grp_replay_token_major_1_fu_235_m_axi_gmem1_ARVALID;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_ARVALID = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_ARVALID;
    end else if (((1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state7) | (1'b1 == ap_CS_fsm_state5) | ((1'b1 == ap_CS_fsm_state3) & (op == 3'd0) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_ARVALID = grp_replay_token_major_fu_215_m_axi_gmem1_ARVALID;
    end else if (((1'b1 == ap_CS_fsm_state11) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_ARVALID = grp_replay_vit_delta_order_fu_196_m_axi_gmem1_ARVALID;
    end else if (((1'b1 == ap_CS_fsm_state4) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (icmp_ln265_1_fu_334_p2 == 1'd1) & (op == 3'd1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_ARVALID = grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_ARVALID;
    end else begin
        gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state20) | ((1'b1 == ap_CS_fsm_state21) & (mode == 1'd0)))) begin
        gmem1_AWADDR = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWADDR;
    end else if (((1'b1 == ap_CS_fsm_state19) | (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13)))) begin
        gmem1_AWADDR = grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWADDR;
    end else if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_AWADDR = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWADDR;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_AWADDR = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWADDR;
    end else if (((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_AWADDR = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWADDR;
    end else begin
        gmem1_AWADDR = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state20) | ((1'b1 == ap_CS_fsm_state21) & (mode == 1'd0)))) begin
        gmem1_AWLEN = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWLEN;
    end else if (((1'b1 == ap_CS_fsm_state19) | (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13)))) begin
        gmem1_AWLEN = grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWLEN;
    end else if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_AWLEN = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWLEN;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_AWLEN = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWLEN;
    end else if (((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_AWLEN = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWLEN;
    end else begin
        gmem1_AWLEN = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state20) | ((1'b1 == ap_CS_fsm_state21) & (mode == 1'd0)))) begin
        gmem1_AWVALID = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_AWVALID;
    end else if (((1'b1 == ap_CS_fsm_state19) | (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13)))) begin
        gmem1_AWVALID = grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_AWVALID;
    end else if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_AWVALID = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_AWVALID;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_AWVALID = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_AWVALID;
    end else if (((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_AWVALID = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_AWVALID;
    end else begin
        gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state20) | ((1'b1 == ap_CS_fsm_state21) & (mode == 1'd0)))) begin
        gmem1_BREADY = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_BREADY;
    end else if (((1'b1 == ap_CS_fsm_state19) | (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13)))) begin
        gmem1_BREADY = grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_BREADY;
    end else if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_BREADY = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_BREADY;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_BREADY = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_BREADY;
    end else if (((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_BREADY = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_BREADY;
    end else begin
        gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_RREADY = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_RREADY;
    end else if (((1'b1 == ap_CS_fsm_state16) | (1'b1 == ap_CS_fsm_state15) | (1'b1 == ap_CS_fsm_state12) | ((1'b1 == ap_CS_fsm_state13) & (op == 3'd0)))) begin
        gmem1_RREADY = grp_replay_token_major_1_fu_235_m_axi_gmem1_RREADY;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_RREADY = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_RREADY;
    end else if (((1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state7) | (1'b1 == ap_CS_fsm_state5) | ((1'b1 == ap_CS_fsm_state3) & (op == 3'd0) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_RREADY = grp_replay_token_major_fu_215_m_axi_gmem1_RREADY;
    end else if (((1'b1 == ap_CS_fsm_state11) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_RREADY = grp_replay_vit_delta_order_fu_196_m_axi_gmem1_RREADY;
    end else if (((1'b1 == ap_CS_fsm_state4) | ((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (icmp_ln265_1_fu_334_p2 == 1'd1) & (op == 3'd1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)))) begin
        gmem1_RREADY = grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_m_axi_gmem1_RREADY;
    end else begin
        gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state20) | ((1'b1 == ap_CS_fsm_state21) & (mode == 1'd0)))) begin
        gmem1_WDATA = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WDATA;
    end else if (((1'b1 == ap_CS_fsm_state19) | (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13)))) begin
        gmem1_WDATA = grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WDATA;
    end else if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_WDATA = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WDATA;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_WDATA = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WDATA;
    end else if (((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_WDATA = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WDATA;
    end else begin
        gmem1_WDATA = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state20) | ((1'b1 == ap_CS_fsm_state21) & (mode == 1'd0)))) begin
        gmem1_WSTRB = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WSTRB;
    end else if (((1'b1 == ap_CS_fsm_state19) | (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13)))) begin
        gmem1_WSTRB = grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WSTRB;
    end else if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_WSTRB = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WSTRB;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_WSTRB = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WSTRB;
    end else if (((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_WSTRB = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WSTRB;
    end else begin
        gmem1_WSTRB = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state20) | ((1'b1 == ap_CS_fsm_state21) & (mode == 1'd0)))) begin
        gmem1_WVALID = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_m_axi_gmem1_WVALID;
    end else if (((1'b1 == ap_CS_fsm_state19) | (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13)))) begin
        gmem1_WVALID = grp_writeback_vit_delta_order_fu_255_m_axi_gmem1_WVALID;
    end else if (((1'b1 == ap_CS_fsm_state18) | (1'b1 == ap_CS_fsm_state17))) begin
        gmem1_WVALID = grp_replay_writeback_vit_delta_order_fu_244_m_axi_gmem1_WVALID;
    end else if (((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9))) begin
        gmem1_WVALID = grp_replay_writeback_llm_delta_order_fu_224_m_axi_gmem1_WVALID;
    end else if (((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438)))) begin
        gmem1_WVALID = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_m_axi_gmem1_WVALID;
    end else begin
        gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state10) & (1'b0 == ap_block_state10_on_subcall_done))) begin
        grp_replay_writeback_llm_delta_order_fu_224_ap_continue = 1'b1;
    end else begin
        grp_replay_writeback_llm_delta_order_fu_224_ap_continue = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state18) & (1'b0 == ap_block_state18_on_subcall_done))) begin
        grp_replay_writeback_vit_delta_order_fu_244_ap_continue = 1'b1;
    end else begin
        grp_replay_writeback_vit_delta_order_fu_244_ap_continue = 1'b0;
    end
end

always @ (*) begin
    if (((grp_replay_writeback_vit_delta_order_fu_244_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state18))) begin
        x_stream_TDATA_int_regslice = grp_replay_writeback_vit_delta_order_fu_244_x_stream_TDATA;
    end else if ((((grp_replay_token_major_1_fu_235_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state16)) | ((grp_replay_token_major_1_fu_235_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state13) & (op == 3'd0)))) begin
        x_stream_TDATA_int_regslice = grp_replay_token_major_1_fu_235_x_stream_TDATA;
    end else if (((grp_replay_writeback_llm_delta_order_fu_224_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        x_stream_TDATA_int_regslice = grp_replay_writeback_llm_delta_order_fu_224_x_stream_TDATA;
    end else if ((((grp_replay_token_major_fu_215_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state8)) | ((grp_replay_token_major_fu_215_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state3) & (op == 3'd0) & (1'd0 == and_ln265_reg_438)))) begin
        x_stream_TDATA_int_regslice = grp_replay_token_major_fu_215_x_stream_TDATA;
    end else if (((grp_replay_vit_delta_order_fu_196_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state11))) begin
        x_stream_TDATA_int_regslice = grp_replay_vit_delta_order_fu_196_x_stream_TDATA;
    end else if (((grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TVALID == 1'b1) & (1'b1 == ap_CS_fsm_state4))) begin
        x_stream_TDATA_int_regslice = grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TDATA;
    end else begin
        x_stream_TDATA_int_regslice = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state18)) begin
        x_stream_TVALID_int_regslice = grp_replay_writeback_vit_delta_order_fu_244_x_stream_TVALID;
    end else if (((1'b1 == ap_CS_fsm_state16) | ((1'b1 == ap_CS_fsm_state13) & (op == 3'd0)))) begin
        x_stream_TVALID_int_regslice = grp_replay_token_major_1_fu_235_x_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        x_stream_TVALID_int_regslice = grp_replay_writeback_llm_delta_order_fu_224_x_stream_TVALID;
    end else if (((1'b1 == ap_CS_fsm_state8) | ((1'b1 == ap_CS_fsm_state3) & (op == 3'd0) & (1'd0 == and_ln265_reg_438)))) begin
        x_stream_TVALID_int_regslice = grp_replay_token_major_fu_215_x_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state11)) begin
        x_stream_TVALID_int_regslice = grp_replay_vit_delta_order_fu_196_x_stream_TVALID;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        x_stream_TVALID_int_regslice = grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TVALID;
    end else begin
        x_stream_TVALID_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if ((~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (1'b1 == ap_CS_fsm_state13))) begin
        y_stream_TREADY_int_regslice = grp_writeback_vit_delta_order_fu_255_y_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state18)) begin
        y_stream_TREADY_int_regslice = grp_replay_writeback_vit_delta_order_fu_244_y_stream_TREADY;
    end else if ((1'b1 == ap_CS_fsm_state10)) begin
        y_stream_TREADY_int_regslice = grp_replay_writeback_llm_delta_order_fu_224_y_stream_TREADY;
    end else if (((1'b1 == ap_CS_fsm_state3) & (icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438))) begin
        y_stream_TREADY_int_regslice = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_y_stream_TREADY;
    end else begin
        y_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1) & (mode_read_read_fu_174_p2 == 1'd1) & (icmp_ln243_fu_272_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state21;
            end else if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1) & (mode == 1'd0) & (icmp_ln243_fu_272_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state20;
            end else if ((~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1) & (icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state19;
            end else if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd4) & (1'b0 == ap_block_state1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state14;
            end else if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd0) & (1'b0 == ap_block_state1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end else if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd1) & (1'b0 == ap_block_state1) & (mode_read_read_fu_174_p2 == 1'd1) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end else if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (icmp_ln265_1_fu_334_p2 == 1'd1) & (op == 3'd1) & (1'b0 == ap_block_state1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (icmp_ln265_1_fu_334_p2 == 1'd1) & (op == 3'd2) & (1'b0 == ap_block_state1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (op == 3'd0) & (1'b0 == ap_block_state1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1) & (((~(op == 3'd0) & ~(op == 3'd2) & ~(op == 3'd1) & (icmp_ln250_fu_278_p2 == 1'd1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0)) | ((icmp_ln250_fu_278_p2 == 1'd1) & (icmp_ln265_1_fu_334_p2 == 1'd0) & (op == 3'd2) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) | ((icmp_ln250_fu_278_p2 == 1'd1) & (icmp_ln265_1_fu_334_p2 == 1'd0) & (op == 3'd1) & (1'd0 == and_ln265_fu_340_p2) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else if (((icmp_ln250_fu_278_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1) & (mode == 1'd0) & (1'd1 == and_ln55_fu_312_p2) & (1'd1 == and_ln265_fu_340_p2) & (icmp_ln243_fu_272_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end else if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1) & (((icmp_ln250_fu_278_p2 == 1'd0) & (icmp_ln243_fu_272_p2 == 1'd0)) | ((1'd0 == and_ln55_fu_312_p2) & (icmp_ln243_fu_272_p2 == 1'd0))))) begin
                ap_NS_fsm = ap_ST_fsm_state22;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (1'b0 == ap_block_state3_on_subcall_done) & ((icmp_ln184_fu_349_p2 == 1'd1) | (1'd0 == and_ln265_reg_438)))) begin
                ap_NS_fsm = ap_ST_fsm_state22;
            end else if (((icmp_ln184_fu_349_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3) & (1'b0 == ap_block_state3_on_subcall_done) & (1'd1 == and_ln265_reg_438))) begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((1'b1 == ap_CS_fsm_state4) & (grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            if (((grp_replay_token_major_fu_215_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state8))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            if (((1'b1 == ap_CS_fsm_state10) & (1'b0 == ap_block_state10_on_subcall_done))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end
        end
        ap_ST_fsm_state11 : begin
            if (((grp_replay_vit_delta_order_fu_196_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state11))) begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end
        end
        ap_ST_fsm_state12 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state13 : begin
            if (((1'b1 == ap_CS_fsm_state13) & (1'b0 == ap_block_state13_on_subcall_done) & ((op == 3'd0) | ((op == 3'd1) | (~(op == 3'd4) | (icmp_ln196_fu_374_p2 == 1'd1)))))) begin
                ap_NS_fsm = ap_ST_fsm_state22;
            end else if (((1'b1 == ap_CS_fsm_state13) & (op == 3'd4) & (1'b0 == ap_block_state13_on_subcall_done) & (icmp_ln196_fu_374_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state15;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state15 : begin
            ap_NS_fsm = ap_ST_fsm_state16;
        end
        ap_ST_fsm_state16 : begin
            if (((grp_replay_token_major_1_fu_235_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state16))) begin
                ap_NS_fsm = ap_ST_fsm_state17;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state16;
            end
        end
        ap_ST_fsm_state17 : begin
            ap_NS_fsm = ap_ST_fsm_state18;
        end
        ap_ST_fsm_state18 : begin
            if (((1'b1 == ap_CS_fsm_state18) & (1'b0 == ap_block_state18_on_subcall_done))) begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state18;
            end
        end
        ap_ST_fsm_state19 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state20 : begin
            ap_NS_fsm = ap_ST_fsm_state21;
        end
        ap_ST_fsm_state21 : begin
            if (((1'b1 == ap_CS_fsm_state21) & (1'b0 == ap_block_state21_on_subcall_done))) begin
                ap_NS_fsm = ap_ST_fsm_state22;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state21;
            end
        end
        ap_ST_fsm_state22 : begin
            if (((1'b1 == ap_CS_fsm_state22) & (regslice_both_x_stream_U_apdone_blk == 1'b0))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state22;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign and_ln265_fu_340_p2 = (icmp_ln265_fu_318_p2 & icmp_ln265_1_fu_334_p2);

assign and_ln55_fu_312_p2 = (xor_ln55_fu_300_p2 & icmp_ln55_fu_306_p2);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd10];

assign ap_CS_fsm_state12 = ap_CS_fsm[32'd11];

assign ap_CS_fsm_state13 = ap_CS_fsm[32'd12];

assign ap_CS_fsm_state14 = ap_CS_fsm[32'd13];

assign ap_CS_fsm_state15 = ap_CS_fsm[32'd14];

assign ap_CS_fsm_state16 = ap_CS_fsm[32'd15];

assign ap_CS_fsm_state17 = ap_CS_fsm[32'd16];

assign ap_CS_fsm_state18 = ap_CS_fsm[32'd17];

assign ap_CS_fsm_state19 = ap_CS_fsm[32'd18];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state20 = ap_CS_fsm[32'd19];

assign ap_CS_fsm_state21 = ap_CS_fsm[32'd20];

assign ap_CS_fsm_state22 = ap_CS_fsm[32'd21];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

always @ (*) begin
    ap_block_state1 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0) | ((ap_predicate_op76_read_state1 == 1'b1) & (cls_stream_TVALID_int_regslice == 1'b0)));
end

always @ (*) begin
    ap_block_state10_on_subcall_done = ((ap_sync_grp_replay_writeback_llm_delta_order_fu_224_ap_ready & ap_sync_grp_replay_writeback_llm_delta_order_fu_224_ap_done) == 1'b0);
end

always @ (*) begin
    ap_block_state13_on_subcall_done = (((grp_writeback_vit_delta_order_fu_255_ap_done == 1'b0) & (ap_predicate_op117_call_state13 == 1'b1)) | ((grp_replay_token_major_1_fu_235_ap_done == 1'b0) & (op == 3'd0)));
end

always @ (*) begin
    ap_block_state18_on_subcall_done = ((ap_sync_grp_replay_writeback_vit_delta_order_fu_244_ap_ready & ap_sync_grp_replay_writeback_vit_delta_order_fu_244_ap_done) == 1'b0);
end

always @ (*) begin
    ap_block_state1_ignore_call0 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0) | ((ap_predicate_op76_read_state1 == 1'b1) & (cls_stream_TVALID_int_regslice == 1'b0)));
end

always @ (*) begin
    ap_block_state1_ignore_call1 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0) | ((ap_predicate_op76_read_state1 == 1'b1) & (cls_stream_TVALID_int_regslice == 1'b0)));
end

always @ (*) begin
    ap_block_state21_on_subcall_done = ((grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_done == 1'b0) & (mode == 1'd0));
end

always @ (*) begin
    ap_block_state3_on_subcall_done = (((ap_predicate_op81_call_state3 == 1'b1) & (grp_replay_token_major_fu_215_ap_done == 1'b0)) | ((ap_predicate_op79_call_state3 == 1'b1) & (grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_done == 1'b0)));
end

always @ (*) begin
    ap_predicate_op117_call_state13 = (~(op == 3'd4) & ~(op == 3'd0) & ~(op == 3'd1));
end

always @ (*) begin
    ap_predicate_op76_read_state1 = ((mode == 1'd0) & (icmp_ln243_fu_272_p2 == 1'd1));
end

always @ (*) begin
    ap_predicate_op79_call_state3 = ((icmp_ln265_1_reg_434 == 1'd1) & (op == 3'd2) & (1'd0 == and_ln265_reg_438));
end

always @ (*) begin
    ap_predicate_op81_call_state3 = ((op == 3'd0) & (1'd0 == and_ln265_reg_438));
end

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign ap_sync_grp_replay_writeback_llm_delta_order_fu_224_ap_done = (grp_replay_writeback_llm_delta_order_fu_224_ap_done | ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_done);

assign ap_sync_grp_replay_writeback_llm_delta_order_fu_224_ap_ready = (grp_replay_writeback_llm_delta_order_fu_224_ap_ready | ap_sync_reg_grp_replay_writeback_llm_delta_order_fu_224_ap_ready);

assign ap_sync_grp_replay_writeback_vit_delta_order_fu_244_ap_done = (grp_replay_writeback_vit_delta_order_fu_244_ap_done | ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_done);

assign ap_sync_grp_replay_writeback_vit_delta_order_fu_244_ap_ready = (grp_replay_writeback_vit_delta_order_fu_244_ap_ready | ap_sync_reg_grp_replay_writeback_vit_delta_order_fu_244_ap_ready);

assign cls_stream_TREADY = regslice_both_cls_stream_U_ack_in;

assign grp_replay_token_major_1_fu_235_ap_start = grp_replay_token_major_1_fu_235_ap_start_reg;

assign grp_replay_token_major_1_fu_235_x_stream_TREADY = ((x_stream_TREADY_int_regslice & ap_CS_fsm_state16) | (x_stream_TREADY_int_regslice & ap_CS_fsm_state13));

assign grp_replay_token_major_fu_215_ap_start = grp_replay_token_major_fu_215_ap_start_reg;

assign grp_replay_token_major_fu_215_x_stream_TREADY = ((x_stream_TREADY_int_regslice & ap_CS_fsm_state8) | (x_stream_TREADY_int_regslice & ap_CS_fsm_state3));

assign grp_replay_vit_delta_order_fu_196_ap_start = grp_replay_vit_delta_order_fu_196_ap_start_reg;

assign grp_replay_vit_delta_order_fu_196_x_stream_TREADY = (x_stream_TREADY_int_regslice & ap_CS_fsm_state11);

assign grp_replay_writeback_llm_delta_order_fu_224_ap_start = grp_replay_writeback_llm_delta_order_fu_224_ap_start_reg;

assign grp_replay_writeback_llm_delta_order_fu_224_x_stream_TREADY = (x_stream_TREADY_int_regslice & ap_CS_fsm_state10);

assign grp_replay_writeback_vit_delta_order_fu_244_ap_start = grp_replay_writeback_vit_delta_order_fu_244_ap_start_reg;

assign grp_replay_writeback_vit_delta_order_fu_244_x_stream_TREADY = (x_stream_TREADY_int_regslice & ap_CS_fsm_state18);

assign grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start = grp_STATE_AXI_Pipeline_VITIS_LOOP_102_1_VITIS_LOOP_103_2_fu_206_ap_start_reg;

assign grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start = grp_STATE_AXI_Pipeline_VITIS_LOOP_208_1_fu_264_ap_start_reg;

assign grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start = grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_ap_start_reg;

assign grp_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2_fu_186_x_stream_TREADY = (x_stream_TREADY_int_regslice & ap_CS_fsm_state4);

assign grp_writeback_vit_delta_order_fu_255_ap_start = grp_writeback_vit_delta_order_fu_255_ap_start_reg;

assign icmp_ln184_fu_349_p2 = ((pass_2_fu_130 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln196_fu_374_p2 = ((pass_fu_134 == 2'd2) ? 1'b1 : 1'b0);

assign icmp_ln243_fu_272_p2 = ((op == 3'd3) ? 1'b1 : 1'b0);

assign icmp_ln250_fu_278_p2 = (($signed(l_begin) < $signed(l_close)) ? 1'b1 : 1'b0);

assign icmp_ln265_1_fu_334_p2 = (($signed(tmp_1_fu_324_p4) < $signed(27'd1)) ? 1'b1 : 1'b0);

assign icmp_ln265_fu_318_p2 = ((op == 3'd4) ? 1'b1 : 1'b0);

assign icmp_ln55_fu_306_p2 = (($signed(select_ln55_fu_284_p3) > $signed(l_begin)) ? 1'b1 : 1'b0);

assign mode_read_read_fu_174_p2 = mode;

assign pass_4_fu_380_p2 = (pass_fu_134 + 2'd1);

assign pass_6_fu_355_p2 = (pass_2_fu_130 + 2'd1);

assign select_ln55_fu_284_p3 = ((mode[0:0] == 1'b1) ? 32'd12 : 32'd33);

assign tmp_1_fu_324_p4 = {{l_begin[31:5]}};

assign tmp_fu_292_p3 = l_begin[32'd31];

assign x_stream_TVALID = regslice_both_x_stream_U_vld_out;

assign xor_ln55_fu_300_p2 = (tmp_fu_292_p3 ^ 1'd1);

assign y_stream_TREADY = regslice_both_y_stream_U_ack_in;

endmodule //STATE_AXI
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_token_major_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state,
        x_stream_TDATA,
        x_stream_TVALID,
        x_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 14'd1;
parameter    ap_ST_fsm_state2 = 14'd2;
parameter    ap_ST_fsm_state3 = 14'd4;
parameter    ap_ST_fsm_state4 = 14'd8;
parameter    ap_ST_fsm_state5 = 14'd16;
parameter    ap_ST_fsm_state6 = 14'd32;
parameter    ap_ST_fsm_state7 = 14'd64;
parameter    ap_ST_fsm_state8 = 14'd128;
parameter    ap_ST_fsm_state9 = 14'd256;
parameter    ap_ST_fsm_state10 = 14'd512;
parameter    ap_ST_fsm_state11 = 14'd1024;
parameter    ap_ST_fsm_state12 = 14'd2048;
parameter    ap_ST_fsm_state13 = 14'd4096;
parameter    ap_ST_fsm_state14 = 14'd8192;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;
input   x_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg x_stream_TVALID;

(* fsm_encoding = "none" *) reg   [13:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    gmem1_blk_n_AR;
reg    gmem1_blk_n_R;
wire    ap_CS_fsm_state11;
wire    ap_CS_fsm_state12;
reg    x_stream_TDATA_blk_n;
wire    ap_CS_fsm_state13;
wire   [10:0] add_ln23_fu_191_p2;
reg   [10:0] add_ln23_reg_250;
wire    ap_CS_fsm_state9;
wire   [6:0] add_ln24_fu_203_p2;
reg   [6:0] add_ln24_reg_258;
wire    ap_CS_fsm_state10;
wire   [24:0] trunc_ln210_fu_213_p1;
reg   [24:0] trunc_ln210_reg_263;
reg   [24:0] tmp_reg_268;
reg   [24:0] tmp_s_reg_273;
reg   [24:0] tmp_8_reg_278;
wire   [24:0] trunc_ln26_fu_217_p1;
reg   [24:0] trunc_ln26_reg_283;
reg   [24:0] tmp_1_reg_288;
reg   [24:0] tmp_2_reg_293;
reg   [24:0] tmp_3_reg_298;
reg   [6:0] ct_reg_115;
wire   [0:0] icmp_ln23_fu_185_p2;
wire  signed [63:0] sext_ln23_fu_166_p1;
reg   [10:0] t_fu_86;
wire   [0:0] icmp_ln24_fu_197_p2;
wire   [59:0] trunc_ln9_fu_156_p4;
wire    ap_CS_fsm_state14;
reg   [13:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
wire    ap_ST_fsm_state8_blk;
wire    ap_ST_fsm_state9_blk;
wire    ap_ST_fsm_state10_blk;
reg    ap_ST_fsm_state11_blk;
reg    ap_ST_fsm_state12_blk;
reg    ap_ST_fsm_state13_blk;
wire    ap_ST_fsm_state14_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 14'd1;
//#0 t_fu_86 = 11'd0;
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state9) & (icmp_ln23_fu_185_p2 == 1'd0))) begin
        ct_reg_115 <= 7'd0;
    end else if (((x_stream_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state13))) begin
        ct_reg_115 <= add_ln24_reg_258;
    end
end

always @ (posedge ap_clk) begin
    if ((~((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        t_fu_86 <= 11'd0;
    end else if (((1'b1 == ap_CS_fsm_state10) & (icmp_ln24_fu_197_p2 == 1'd1))) begin
        t_fu_86 <= add_ln23_reg_250;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        add_ln23_reg_250 <= add_ln23_fu_191_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state10)) begin
        add_ln24_reg_258 <= add_ln24_fu_203_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state12)) begin
        tmp_1_reg_288 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_2_reg_293 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_3_reg_298 <= {{m_axi_gmem1_RDATA[120:96]}};
        trunc_ln26_reg_283 <= trunc_ln26_fu_217_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
        tmp_8_reg_278 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_reg_268 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_s_reg_273 <= {{m_axi_gmem1_RDATA[88:64]}};
        trunc_ln210_reg_263 <= trunc_ln210_fu_213_p1;
    end
end

assign ap_ST_fsm_state10_blk = 1'b0;

always @ (*) begin
    if ((m_axi_gmem1_RVALID == 1'b0)) begin
        ap_ST_fsm_state11_blk = 1'b1;
    end else begin
        ap_ST_fsm_state11_blk = 1'b0;
    end
end

always @ (*) begin
    if ((m_axi_gmem1_RVALID == 1'b0)) begin
        ap_ST_fsm_state12_blk = 1'b1;
    end else begin
        ap_ST_fsm_state12_blk = 1'b0;
    end
end

always @ (*) begin
    if ((x_stream_TREADY == 1'b0)) begin
        ap_ST_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_fsm_state13_blk = 1'b0;
    end
end

assign ap_ST_fsm_state14_blk = 1'b0;

always @ (*) begin
    if (((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

assign ap_ST_fsm_state7_blk = 1'b0;

assign ap_ST_fsm_state8_blk = 1'b0;

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state14) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        gmem1_blk_n_AR = m_axi_gmem1_ARREADY;
    end else begin
        gmem1_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state12) | (1'b1 == ap_CS_fsm_state11))) begin
        gmem1_blk_n_R = m_axi_gmem1_RVALID;
    end else begin
        gmem1_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if ((~((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        m_axi_gmem1_ARVALID = 1'b1;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if ((((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state12)) | ((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state11)))) begin
        m_axi_gmem1_RREADY = 1'b1;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state13)) begin
        x_stream_TDATA_blk_n = x_stream_TREADY;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((x_stream_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state13))) begin
        x_stream_TVALID = 1'b1;
    end else begin
        x_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            if (((1'b1 == ap_CS_fsm_state9) & (icmp_ln23_fu_185_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state14;
            end
        end
        ap_ST_fsm_state10 : begin
            if (((1'b1 == ap_CS_fsm_state10) & (icmp_ln24_fu_197_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end
        end
        ap_ST_fsm_state11 : begin
            if (((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state11))) begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end
        end
        ap_ST_fsm_state12 : begin
            if (((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state12))) begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end
        end
        ap_ST_fsm_state13 : begin
            if (((x_stream_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state13))) begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln23_fu_191_p2 = (t_fu_86 + 11'd1);

assign add_ln24_fu_203_p2 = (ct_reg_115 + 7'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd10];

assign ap_CS_fsm_state12 = ap_CS_fsm[32'd11];

assign ap_CS_fsm_state13 = ap_CS_fsm[32'd12];

assign ap_CS_fsm_state14 = ap_CS_fsm[32'd13];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign icmp_ln23_fu_185_p2 = ((t_fu_86 == 11'd1024) ? 1'b1 : 1'b0);

assign icmp_ln24_fu_197_p2 = ((ct_reg_115 == 7'd96) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = sext_ln23_fu_166_p1;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd196608;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign sext_ln23_fu_166_p1 = $signed(trunc_ln9_fu_156_p4);

assign trunc_ln210_fu_213_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln26_fu_217_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln9_fu_156_p4 = {{memory_state[63:4]}};

assign x_stream_TDATA = {{{{{{{{tmp_3_reg_298}, {tmp_2_reg_293}}, {tmp_1_reg_288}}, {trunc_ln26_reg_283}}, {tmp_8_reg_278}}, {tmp_s_reg_273}}, {tmp_reg_268}}, {trunc_ln210_reg_263}};

endmodule //STATE_AXI_replay_token_major_1
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_writeback_llm_delta_order (
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state,
        x_stream_TDATA,
        y_stream_TDATA,
        ap_clk,
        ap_rst,
        memory_state_ap_vld,
        x_stream_TVALID,
        x_stream_TREADY,
        ap_start,
        ap_done,
        y_stream_TVALID,
        y_stream_TREADY,
        ap_ready,
        ap_idle,
        ap_continue
);


output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
input  [199:0] y_stream_TDATA;
input   ap_clk;
input   ap_rst;
input   memory_state_ap_vld;
output   x_stream_TVALID;
input   x_stream_TREADY;
input   ap_start;
output   ap_done;
input   y_stream_TVALID;
output   y_stream_TREADY;
output   ap_ready;
output   ap_idle;
input   ap_continue;

wire    replay_llm_delta_order_indexed_U0_ap_start;
wire    replay_llm_delta_order_indexed_U0_ap_done;
wire    replay_llm_delta_order_indexed_U0_ap_continue;
wire    replay_llm_delta_order_indexed_U0_ap_idle;
wire    replay_llm_delta_order_indexed_U0_ap_ready;
wire    replay_llm_delta_order_indexed_U0_start_out;
wire    replay_llm_delta_order_indexed_U0_start_write;
wire    replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWVALID;
wire   [63:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWADDR;
wire   [0:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWID;
wire   [31:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWLEN;
wire   [2:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWSIZE;
wire   [1:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWBURST;
wire   [1:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWLOCK;
wire   [3:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWCACHE;
wire   [2:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWPROT;
wire   [3:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWQOS;
wire   [3:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWREGION;
wire   [0:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWUSER;
wire    replay_llm_delta_order_indexed_U0_m_axi_gmem1_WVALID;
wire   [127:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_WDATA;
wire   [15:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_WSTRB;
wire    replay_llm_delta_order_indexed_U0_m_axi_gmem1_WLAST;
wire   [0:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_WID;
wire   [0:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_WUSER;
wire    replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARVALID;
wire   [63:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARADDR;
wire   [0:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARID;
wire   [31:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARLEN;
wire   [2:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARSIZE;
wire   [1:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARBURST;
wire   [1:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARLOCK;
wire   [3:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARCACHE;
wire   [2:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARPROT;
wire   [3:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARQOS;
wire   [3:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARREGION;
wire   [0:0] replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARUSER;
wire    replay_llm_delta_order_indexed_U0_m_axi_gmem1_RREADY;
wire    replay_llm_delta_order_indexed_U0_m_axi_gmem1_BREADY;
wire   [199:0] replay_llm_delta_order_indexed_U0_x_stream_TDATA;
wire    replay_llm_delta_order_indexed_U0_x_stream_TVALID;
wire   [31:0] replay_llm_delta_order_indexed_U0_tile_idx_stream_din;
wire    replay_llm_delta_order_indexed_U0_tile_idx_stream_write;
wire   [63:0] replay_llm_delta_order_indexed_U0_memory_state_c_din;
wire    replay_llm_delta_order_indexed_U0_memory_state_c_write;
wire    ap_sync_continue;
wire    writeback_indexed_order_U0_ap_start;
wire    writeback_indexed_order_U0_ap_done;
wire    writeback_indexed_order_U0_ap_continue;
wire    writeback_indexed_order_U0_ap_idle;
wire    writeback_indexed_order_U0_ap_ready;
wire    writeback_indexed_order_U0_m_axi_gmem1_AWVALID;
wire   [63:0] writeback_indexed_order_U0_m_axi_gmem1_AWADDR;
wire   [0:0] writeback_indexed_order_U0_m_axi_gmem1_AWID;
wire   [31:0] writeback_indexed_order_U0_m_axi_gmem1_AWLEN;
wire   [2:0] writeback_indexed_order_U0_m_axi_gmem1_AWSIZE;
wire   [1:0] writeback_indexed_order_U0_m_axi_gmem1_AWBURST;
wire   [1:0] writeback_indexed_order_U0_m_axi_gmem1_AWLOCK;
wire   [3:0] writeback_indexed_order_U0_m_axi_gmem1_AWCACHE;
wire   [2:0] writeback_indexed_order_U0_m_axi_gmem1_AWPROT;
wire   [3:0] writeback_indexed_order_U0_m_axi_gmem1_AWQOS;
wire   [3:0] writeback_indexed_order_U0_m_axi_gmem1_AWREGION;
wire   [0:0] writeback_indexed_order_U0_m_axi_gmem1_AWUSER;
wire    writeback_indexed_order_U0_m_axi_gmem1_WVALID;
wire   [127:0] writeback_indexed_order_U0_m_axi_gmem1_WDATA;
wire   [15:0] writeback_indexed_order_U0_m_axi_gmem1_WSTRB;
wire    writeback_indexed_order_U0_m_axi_gmem1_WLAST;
wire   [0:0] writeback_indexed_order_U0_m_axi_gmem1_WID;
wire   [0:0] writeback_indexed_order_U0_m_axi_gmem1_WUSER;
wire    writeback_indexed_order_U0_m_axi_gmem1_ARVALID;
wire   [63:0] writeback_indexed_order_U0_m_axi_gmem1_ARADDR;
wire   [0:0] writeback_indexed_order_U0_m_axi_gmem1_ARID;
wire   [31:0] writeback_indexed_order_U0_m_axi_gmem1_ARLEN;
wire   [2:0] writeback_indexed_order_U0_m_axi_gmem1_ARSIZE;
wire   [1:0] writeback_indexed_order_U0_m_axi_gmem1_ARBURST;
wire   [1:0] writeback_indexed_order_U0_m_axi_gmem1_ARLOCK;
wire   [3:0] writeback_indexed_order_U0_m_axi_gmem1_ARCACHE;
wire   [2:0] writeback_indexed_order_U0_m_axi_gmem1_ARPROT;
wire   [3:0] writeback_indexed_order_U0_m_axi_gmem1_ARQOS;
wire   [3:0] writeback_indexed_order_U0_m_axi_gmem1_ARREGION;
wire   [0:0] writeback_indexed_order_U0_m_axi_gmem1_ARUSER;
wire    writeback_indexed_order_U0_m_axi_gmem1_RREADY;
wire    writeback_indexed_order_U0_m_axi_gmem1_BREADY;
wire    writeback_indexed_order_U0_memory_state_read;
wire    writeback_indexed_order_U0_y_stream_TREADY;
wire    writeback_indexed_order_U0_tile_idx_stream_read;
wire    tile_idx_stream_full_n;
wire   [31:0] tile_idx_stream_dout;
wire   [6:0] tile_idx_stream_num_data_valid;
wire   [6:0] tile_idx_stream_fifo_cap;
wire    tile_idx_stream_empty_n;
wire    memory_state_c_full_n;
wire   [63:0] memory_state_c_dout;
wire   [2:0] memory_state_c_num_data_valid;
wire   [2:0] memory_state_c_fifo_cap;
wire    memory_state_c_empty_n;
wire    ap_sync_done;
wire   [0:0] start_for_writeback_indexed_order_U0_din;
wire    start_for_writeback_indexed_order_U0_full_n;
wire   [0:0] start_for_writeback_indexed_order_U0_dout;
wire    start_for_writeback_indexed_order_U0_empty_n;

STATE_AXI_replay_llm_delta_order_indexed replay_llm_delta_order_indexed_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(replay_llm_delta_order_indexed_U0_ap_start),
    .start_full_n(start_for_writeback_indexed_order_U0_full_n),
    .ap_done(replay_llm_delta_order_indexed_U0_ap_done),
    .ap_continue(replay_llm_delta_order_indexed_U0_ap_continue),
    .ap_idle(replay_llm_delta_order_indexed_U0_ap_idle),
    .ap_ready(replay_llm_delta_order_indexed_U0_ap_ready),
    .start_out(replay_llm_delta_order_indexed_U0_start_out),
    .start_write(replay_llm_delta_order_indexed_U0_start_write),
    .m_axi_gmem1_AWVALID(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(replay_llm_delta_order_indexed_U0_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(replay_llm_delta_order_indexed_U0_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(replay_llm_delta_order_indexed_U0_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(replay_llm_delta_order_indexed_U0_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(replay_llm_delta_order_indexed_U0_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(replay_llm_delta_order_indexed_U0_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(replay_llm_delta_order_indexed_U0_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(m_axi_gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(m_axi_gmem1_RVALID),
    .m_axi_gmem1_RREADY(replay_llm_delta_order_indexed_U0_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(m_axi_gmem1_RDATA),
    .m_axi_gmem1_RLAST(m_axi_gmem1_RLAST),
    .m_axi_gmem1_RID(m_axi_gmem1_RID),
    .m_axi_gmem1_RFIFONUM(m_axi_gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(m_axi_gmem1_RUSER),
    .m_axi_gmem1_RRESP(m_axi_gmem1_RRESP),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(replay_llm_delta_order_indexed_U0_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_state(memory_state),
    .x_stream_TDATA(replay_llm_delta_order_indexed_U0_x_stream_TDATA),
    .x_stream_TVALID(replay_llm_delta_order_indexed_U0_x_stream_TVALID),
    .x_stream_TREADY(x_stream_TREADY),
    .tile_idx_stream_din(replay_llm_delta_order_indexed_U0_tile_idx_stream_din),
    .tile_idx_stream_num_data_valid(tile_idx_stream_num_data_valid),
    .tile_idx_stream_fifo_cap(tile_idx_stream_fifo_cap),
    .tile_idx_stream_full_n(tile_idx_stream_full_n),
    .tile_idx_stream_write(replay_llm_delta_order_indexed_U0_tile_idx_stream_write),
    .memory_state_c_din(replay_llm_delta_order_indexed_U0_memory_state_c_din),
    .memory_state_c_num_data_valid(memory_state_c_num_data_valid),
    .memory_state_c_fifo_cap(memory_state_c_fifo_cap),
    .memory_state_c_full_n(memory_state_c_full_n),
    .memory_state_c_write(replay_llm_delta_order_indexed_U0_memory_state_c_write)
);

STATE_AXI_writeback_indexed_order writeback_indexed_order_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(writeback_indexed_order_U0_ap_start),
    .ap_done(writeback_indexed_order_U0_ap_done),
    .ap_continue(writeback_indexed_order_U0_ap_continue),
    .ap_idle(writeback_indexed_order_U0_ap_idle),
    .ap_ready(writeback_indexed_order_U0_ap_ready),
    .m_axi_gmem1_AWVALID(writeback_indexed_order_U0_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(m_axi_gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(writeback_indexed_order_U0_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(writeback_indexed_order_U0_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(writeback_indexed_order_U0_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(writeback_indexed_order_U0_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(writeback_indexed_order_U0_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(writeback_indexed_order_U0_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(writeback_indexed_order_U0_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(writeback_indexed_order_U0_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(writeback_indexed_order_U0_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(writeback_indexed_order_U0_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(writeback_indexed_order_U0_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(writeback_indexed_order_U0_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(m_axi_gmem1_WREADY),
    .m_axi_gmem1_WDATA(writeback_indexed_order_U0_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(writeback_indexed_order_U0_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(writeback_indexed_order_U0_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(writeback_indexed_order_U0_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(writeback_indexed_order_U0_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(writeback_indexed_order_U0_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(writeback_indexed_order_U0_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(writeback_indexed_order_U0_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(writeback_indexed_order_U0_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(writeback_indexed_order_U0_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(writeback_indexed_order_U0_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(writeback_indexed_order_U0_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(writeback_indexed_order_U0_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(writeback_indexed_order_U0_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(writeback_indexed_order_U0_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(writeback_indexed_order_U0_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(writeback_indexed_order_U0_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(writeback_indexed_order_U0_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(9'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(m_axi_gmem1_BVALID),
    .m_axi_gmem1_BREADY(writeback_indexed_order_U0_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(m_axi_gmem1_BRESP),
    .m_axi_gmem1_BID(m_axi_gmem1_BID),
    .m_axi_gmem1_BUSER(m_axi_gmem1_BUSER),
    .memory_state_dout(memory_state_c_dout),
    .memory_state_num_data_valid(memory_state_c_num_data_valid),
    .memory_state_fifo_cap(memory_state_c_fifo_cap),
    .memory_state_empty_n(memory_state_c_empty_n),
    .memory_state_read(writeback_indexed_order_U0_memory_state_read),
    .y_stream_TDATA(y_stream_TDATA),
    .y_stream_TVALID(y_stream_TVALID),
    .y_stream_TREADY(writeback_indexed_order_U0_y_stream_TREADY),
    .tile_idx_stream_dout(tile_idx_stream_dout),
    .tile_idx_stream_num_data_valid(tile_idx_stream_num_data_valid),
    .tile_idx_stream_fifo_cap(tile_idx_stream_fifo_cap),
    .tile_idx_stream_empty_n(tile_idx_stream_empty_n),
    .tile_idx_stream_read(writeback_indexed_order_U0_tile_idx_stream_read)
);

STATE_AXI_fifo_w32_d64_A tile_idx_stream_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(replay_llm_delta_order_indexed_U0_tile_idx_stream_din),
    .if_full_n(tile_idx_stream_full_n),
    .if_write(replay_llm_delta_order_indexed_U0_tile_idx_stream_write),
    .if_dout(tile_idx_stream_dout),
    .if_num_data_valid(tile_idx_stream_num_data_valid),
    .if_fifo_cap(tile_idx_stream_fifo_cap),
    .if_empty_n(tile_idx_stream_empty_n),
    .if_read(writeback_indexed_order_U0_tile_idx_stream_read)
);

STATE_AXI_fifo_w64_d2_S memory_state_c_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(replay_llm_delta_order_indexed_U0_memory_state_c_din),
    .if_full_n(memory_state_c_full_n),
    .if_write(replay_llm_delta_order_indexed_U0_memory_state_c_write),
    .if_dout(memory_state_c_dout),
    .if_num_data_valid(memory_state_c_num_data_valid),
    .if_fifo_cap(memory_state_c_fifo_cap),
    .if_empty_n(memory_state_c_empty_n),
    .if_read(writeback_indexed_order_U0_memory_state_read)
);

STATE_AXI_start_for_writeback_indexed_order_U0 start_for_writeback_indexed_order_U0_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(start_for_writeback_indexed_order_U0_din),
    .if_full_n(start_for_writeback_indexed_order_U0_full_n),
    .if_write(replay_llm_delta_order_indexed_U0_start_write),
    .if_dout(start_for_writeback_indexed_order_U0_dout),
    .if_empty_n(start_for_writeback_indexed_order_U0_empty_n),
    .if_read(writeback_indexed_order_U0_ap_ready)
);

assign ap_done = ap_sync_done;

assign ap_idle = (writeback_indexed_order_U0_ap_idle & replay_llm_delta_order_indexed_U0_ap_idle);

assign ap_ready = replay_llm_delta_order_indexed_U0_ap_ready;

assign ap_sync_continue = (ap_sync_done & ap_continue);

assign ap_sync_done = (writeback_indexed_order_U0_ap_done & replay_llm_delta_order_indexed_U0_ap_done);

assign m_axi_gmem1_ARADDR = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARADDR;

assign m_axi_gmem1_ARBURST = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARBURST;

assign m_axi_gmem1_ARCACHE = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARCACHE;

assign m_axi_gmem1_ARID = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARID;

assign m_axi_gmem1_ARLEN = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARLEN;

assign m_axi_gmem1_ARLOCK = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARLOCK;

assign m_axi_gmem1_ARPROT = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARPROT;

assign m_axi_gmem1_ARQOS = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARQOS;

assign m_axi_gmem1_ARREGION = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARREGION;

assign m_axi_gmem1_ARSIZE = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARSIZE;

assign m_axi_gmem1_ARUSER = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARUSER;

assign m_axi_gmem1_ARVALID = replay_llm_delta_order_indexed_U0_m_axi_gmem1_ARVALID;

assign m_axi_gmem1_AWADDR = writeback_indexed_order_U0_m_axi_gmem1_AWADDR;

assign m_axi_gmem1_AWBURST = writeback_indexed_order_U0_m_axi_gmem1_AWBURST;

assign m_axi_gmem1_AWCACHE = writeback_indexed_order_U0_m_axi_gmem1_AWCACHE;

assign m_axi_gmem1_AWID = writeback_indexed_order_U0_m_axi_gmem1_AWID;

assign m_axi_gmem1_AWLEN = writeback_indexed_order_U0_m_axi_gmem1_AWLEN;

assign m_axi_gmem1_AWLOCK = writeback_indexed_order_U0_m_axi_gmem1_AWLOCK;

assign m_axi_gmem1_AWPROT = writeback_indexed_order_U0_m_axi_gmem1_AWPROT;

assign m_axi_gmem1_AWQOS = writeback_indexed_order_U0_m_axi_gmem1_AWQOS;

assign m_axi_gmem1_AWREGION = writeback_indexed_order_U0_m_axi_gmem1_AWREGION;

assign m_axi_gmem1_AWSIZE = writeback_indexed_order_U0_m_axi_gmem1_AWSIZE;

assign m_axi_gmem1_AWUSER = writeback_indexed_order_U0_m_axi_gmem1_AWUSER;

assign m_axi_gmem1_AWVALID = writeback_indexed_order_U0_m_axi_gmem1_AWVALID;

assign m_axi_gmem1_BREADY = writeback_indexed_order_U0_m_axi_gmem1_BREADY;

assign m_axi_gmem1_RREADY = replay_llm_delta_order_indexed_U0_m_axi_gmem1_RREADY;

assign m_axi_gmem1_WDATA = writeback_indexed_order_U0_m_axi_gmem1_WDATA;

assign m_axi_gmem1_WID = writeback_indexed_order_U0_m_axi_gmem1_WID;

assign m_axi_gmem1_WLAST = writeback_indexed_order_U0_m_axi_gmem1_WLAST;

assign m_axi_gmem1_WSTRB = writeback_indexed_order_U0_m_axi_gmem1_WSTRB;

assign m_axi_gmem1_WUSER = writeback_indexed_order_U0_m_axi_gmem1_WUSER;

assign m_axi_gmem1_WVALID = writeback_indexed_order_U0_m_axi_gmem1_WVALID;

assign replay_llm_delta_order_indexed_U0_ap_continue = ap_sync_continue;

assign replay_llm_delta_order_indexed_U0_ap_start = ap_start;

assign start_for_writeback_indexed_order_U0_din = 1'b1;

assign writeback_indexed_order_U0_ap_continue = ap_sync_continue;

assign writeback_indexed_order_U0_ap_start = start_for_writeback_indexed_order_U0_empty_n;

assign x_stream_TDATA = replay_llm_delta_order_indexed_U0_x_stream_TDATA;

assign x_stream_TVALID = replay_llm_delta_order_indexed_U0_x_stream_TVALID;

assign y_stream_TREADY = writeback_indexed_order_U0_y_stream_TREADY;

endmodule //STATE_AXI_replay_writeback_llm_delta_order
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_writeback_vit_delta_order (
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state,
        x_stream_TDATA,
        y_stream_TDATA,
        ap_clk,
        ap_rst,
        memory_state_ap_vld,
        x_stream_TVALID,
        x_stream_TREADY,
        ap_start,
        ap_done,
        y_stream_TVALID,
        y_stream_TREADY,
        ap_ready,
        ap_idle,
        ap_continue
);


output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
input  [199:0] y_stream_TDATA;
input   ap_clk;
input   ap_rst;
input   memory_state_ap_vld;
output   x_stream_TVALID;
input   x_stream_TREADY;
input   ap_start;
output   ap_done;
input   y_stream_TVALID;
output   y_stream_TREADY;
output   ap_ready;
output   ap_idle;
input   ap_continue;

wire    replay_vit_delta_order_indexed_U0_ap_start;
wire    replay_vit_delta_order_indexed_U0_ap_done;
wire    replay_vit_delta_order_indexed_U0_ap_continue;
wire    replay_vit_delta_order_indexed_U0_ap_idle;
wire    replay_vit_delta_order_indexed_U0_ap_ready;
wire    replay_vit_delta_order_indexed_U0_start_out;
wire    replay_vit_delta_order_indexed_U0_start_write;
wire    replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWVALID;
wire   [63:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWADDR;
wire   [0:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWID;
wire   [31:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWLEN;
wire   [2:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWSIZE;
wire   [1:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWBURST;
wire   [1:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWLOCK;
wire   [3:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWCACHE;
wire   [2:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWPROT;
wire   [3:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWQOS;
wire   [3:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWREGION;
wire   [0:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWUSER;
wire    replay_vit_delta_order_indexed_U0_m_axi_gmem1_WVALID;
wire   [127:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_WDATA;
wire   [15:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_WSTRB;
wire    replay_vit_delta_order_indexed_U0_m_axi_gmem1_WLAST;
wire   [0:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_WID;
wire   [0:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_WUSER;
wire    replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARVALID;
wire   [63:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARADDR;
wire   [0:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARID;
wire   [31:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARLEN;
wire   [2:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARSIZE;
wire   [1:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARBURST;
wire   [1:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARLOCK;
wire   [3:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARCACHE;
wire   [2:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARPROT;
wire   [3:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARQOS;
wire   [3:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARREGION;
wire   [0:0] replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARUSER;
wire    replay_vit_delta_order_indexed_U0_m_axi_gmem1_RREADY;
wire    replay_vit_delta_order_indexed_U0_m_axi_gmem1_BREADY;
wire   [199:0] replay_vit_delta_order_indexed_U0_x_stream_TDATA;
wire    replay_vit_delta_order_indexed_U0_x_stream_TVALID;
wire   [31:0] replay_vit_delta_order_indexed_U0_tile_idx_stream_din;
wire    replay_vit_delta_order_indexed_U0_tile_idx_stream_write;
wire   [63:0] replay_vit_delta_order_indexed_U0_memory_state_c_din;
wire    replay_vit_delta_order_indexed_U0_memory_state_c_write;
wire    ap_sync_continue;
wire    writeback_indexed_order_2_U0_ap_start;
wire    writeback_indexed_order_2_U0_ap_done;
wire    writeback_indexed_order_2_U0_ap_continue;
wire    writeback_indexed_order_2_U0_ap_idle;
wire    writeback_indexed_order_2_U0_ap_ready;
wire    writeback_indexed_order_2_U0_m_axi_gmem1_AWVALID;
wire   [63:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWADDR;
wire   [0:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWID;
wire   [31:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWLEN;
wire   [2:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWSIZE;
wire   [1:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWBURST;
wire   [1:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWLOCK;
wire   [3:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWCACHE;
wire   [2:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWPROT;
wire   [3:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWQOS;
wire   [3:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWREGION;
wire   [0:0] writeback_indexed_order_2_U0_m_axi_gmem1_AWUSER;
wire    writeback_indexed_order_2_U0_m_axi_gmem1_WVALID;
wire   [127:0] writeback_indexed_order_2_U0_m_axi_gmem1_WDATA;
wire   [15:0] writeback_indexed_order_2_U0_m_axi_gmem1_WSTRB;
wire    writeback_indexed_order_2_U0_m_axi_gmem1_WLAST;
wire   [0:0] writeback_indexed_order_2_U0_m_axi_gmem1_WID;
wire   [0:0] writeback_indexed_order_2_U0_m_axi_gmem1_WUSER;
wire    writeback_indexed_order_2_U0_m_axi_gmem1_ARVALID;
wire   [63:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARADDR;
wire   [0:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARID;
wire   [31:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARLEN;
wire   [2:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARSIZE;
wire   [1:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARBURST;
wire   [1:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARLOCK;
wire   [3:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARCACHE;
wire   [2:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARPROT;
wire   [3:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARQOS;
wire   [3:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARREGION;
wire   [0:0] writeback_indexed_order_2_U0_m_axi_gmem1_ARUSER;
wire    writeback_indexed_order_2_U0_m_axi_gmem1_RREADY;
wire    writeback_indexed_order_2_U0_m_axi_gmem1_BREADY;
wire    writeback_indexed_order_2_U0_memory_state_read;
wire    writeback_indexed_order_2_U0_y_stream_TREADY;
wire    writeback_indexed_order_2_U0_tile_idx_stream_read;
wire    tile_idx_stream_full_n;
wire   [31:0] tile_idx_stream_dout;
wire   [6:0] tile_idx_stream_num_data_valid;
wire   [6:0] tile_idx_stream_fifo_cap;
wire    tile_idx_stream_empty_n;
wire    memory_state_c_full_n;
wire   [63:0] memory_state_c_dout;
wire   [2:0] memory_state_c_num_data_valid;
wire   [2:0] memory_state_c_fifo_cap;
wire    memory_state_c_empty_n;
wire    ap_sync_done;
wire   [0:0] start_for_writeback_indexed_order_2_U0_din;
wire    start_for_writeback_indexed_order_2_U0_full_n;
wire   [0:0] start_for_writeback_indexed_order_2_U0_dout;
wire    start_for_writeback_indexed_order_2_U0_empty_n;

STATE_AXI_replay_vit_delta_order_indexed replay_vit_delta_order_indexed_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(replay_vit_delta_order_indexed_U0_ap_start),
    .start_full_n(start_for_writeback_indexed_order_2_U0_full_n),
    .ap_done(replay_vit_delta_order_indexed_U0_ap_done),
    .ap_continue(replay_vit_delta_order_indexed_U0_ap_continue),
    .ap_idle(replay_vit_delta_order_indexed_U0_ap_idle),
    .ap_ready(replay_vit_delta_order_indexed_U0_ap_ready),
    .start_out(replay_vit_delta_order_indexed_U0_start_out),
    .start_write(replay_vit_delta_order_indexed_U0_start_write),
    .m_axi_gmem1_AWVALID(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(replay_vit_delta_order_indexed_U0_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(replay_vit_delta_order_indexed_U0_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(replay_vit_delta_order_indexed_U0_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(replay_vit_delta_order_indexed_U0_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(replay_vit_delta_order_indexed_U0_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(replay_vit_delta_order_indexed_U0_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(replay_vit_delta_order_indexed_U0_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(m_axi_gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(m_axi_gmem1_RVALID),
    .m_axi_gmem1_RREADY(replay_vit_delta_order_indexed_U0_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(m_axi_gmem1_RDATA),
    .m_axi_gmem1_RLAST(m_axi_gmem1_RLAST),
    .m_axi_gmem1_RID(m_axi_gmem1_RID),
    .m_axi_gmem1_RFIFONUM(m_axi_gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(m_axi_gmem1_RUSER),
    .m_axi_gmem1_RRESP(m_axi_gmem1_RRESP),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(replay_vit_delta_order_indexed_U0_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_state(memory_state),
    .x_stream_TDATA(replay_vit_delta_order_indexed_U0_x_stream_TDATA),
    .x_stream_TVALID(replay_vit_delta_order_indexed_U0_x_stream_TVALID),
    .x_stream_TREADY(x_stream_TREADY),
    .tile_idx_stream_din(replay_vit_delta_order_indexed_U0_tile_idx_stream_din),
    .tile_idx_stream_num_data_valid(tile_idx_stream_num_data_valid),
    .tile_idx_stream_fifo_cap(tile_idx_stream_fifo_cap),
    .tile_idx_stream_full_n(tile_idx_stream_full_n),
    .tile_idx_stream_write(replay_vit_delta_order_indexed_U0_tile_idx_stream_write),
    .memory_state_c_din(replay_vit_delta_order_indexed_U0_memory_state_c_din),
    .memory_state_c_num_data_valid(memory_state_c_num_data_valid),
    .memory_state_c_fifo_cap(memory_state_c_fifo_cap),
    .memory_state_c_full_n(memory_state_c_full_n),
    .memory_state_c_write(replay_vit_delta_order_indexed_U0_memory_state_c_write)
);

STATE_AXI_writeback_indexed_order_2 writeback_indexed_order_2_U0(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(writeback_indexed_order_2_U0_ap_start),
    .ap_done(writeback_indexed_order_2_U0_ap_done),
    .ap_continue(writeback_indexed_order_2_U0_ap_continue),
    .ap_idle(writeback_indexed_order_2_U0_ap_idle),
    .ap_ready(writeback_indexed_order_2_U0_ap_ready),
    .m_axi_gmem1_AWVALID(writeback_indexed_order_2_U0_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(m_axi_gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(writeback_indexed_order_2_U0_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(writeback_indexed_order_2_U0_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(writeback_indexed_order_2_U0_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(writeback_indexed_order_2_U0_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(writeback_indexed_order_2_U0_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(writeback_indexed_order_2_U0_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(writeback_indexed_order_2_U0_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(writeback_indexed_order_2_U0_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(writeback_indexed_order_2_U0_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(writeback_indexed_order_2_U0_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(writeback_indexed_order_2_U0_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(writeback_indexed_order_2_U0_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(m_axi_gmem1_WREADY),
    .m_axi_gmem1_WDATA(writeback_indexed_order_2_U0_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(writeback_indexed_order_2_U0_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(writeback_indexed_order_2_U0_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(writeback_indexed_order_2_U0_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(writeback_indexed_order_2_U0_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(writeback_indexed_order_2_U0_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(writeback_indexed_order_2_U0_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(writeback_indexed_order_2_U0_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(writeback_indexed_order_2_U0_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(writeback_indexed_order_2_U0_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(writeback_indexed_order_2_U0_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(writeback_indexed_order_2_U0_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(writeback_indexed_order_2_U0_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(writeback_indexed_order_2_U0_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(writeback_indexed_order_2_U0_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(writeback_indexed_order_2_U0_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(writeback_indexed_order_2_U0_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(writeback_indexed_order_2_U0_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(9'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(m_axi_gmem1_BVALID),
    .m_axi_gmem1_BREADY(writeback_indexed_order_2_U0_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(m_axi_gmem1_BRESP),
    .m_axi_gmem1_BID(m_axi_gmem1_BID),
    .m_axi_gmem1_BUSER(m_axi_gmem1_BUSER),
    .memory_state_dout(memory_state_c_dout),
    .memory_state_num_data_valid(memory_state_c_num_data_valid),
    .memory_state_fifo_cap(memory_state_c_fifo_cap),
    .memory_state_empty_n(memory_state_c_empty_n),
    .memory_state_read(writeback_indexed_order_2_U0_memory_state_read),
    .y_stream_TDATA(y_stream_TDATA),
    .y_stream_TVALID(y_stream_TVALID),
    .y_stream_TREADY(writeback_indexed_order_2_U0_y_stream_TREADY),
    .tile_idx_stream_dout(tile_idx_stream_dout),
    .tile_idx_stream_num_data_valid(tile_idx_stream_num_data_valid),
    .tile_idx_stream_fifo_cap(tile_idx_stream_fifo_cap),
    .tile_idx_stream_empty_n(tile_idx_stream_empty_n),
    .tile_idx_stream_read(writeback_indexed_order_2_U0_tile_idx_stream_read)
);

STATE_AXI_fifo_w32_d64_A_x tile_idx_stream_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(replay_vit_delta_order_indexed_U0_tile_idx_stream_din),
    .if_full_n(tile_idx_stream_full_n),
    .if_write(replay_vit_delta_order_indexed_U0_tile_idx_stream_write),
    .if_dout(tile_idx_stream_dout),
    .if_num_data_valid(tile_idx_stream_num_data_valid),
    .if_fifo_cap(tile_idx_stream_fifo_cap),
    .if_empty_n(tile_idx_stream_empty_n),
    .if_read(writeback_indexed_order_2_U0_tile_idx_stream_read)
);

STATE_AXI_fifo_w64_d2_S_x memory_state_c_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(replay_vit_delta_order_indexed_U0_memory_state_c_din),
    .if_full_n(memory_state_c_full_n),
    .if_write(replay_vit_delta_order_indexed_U0_memory_state_c_write),
    .if_dout(memory_state_c_dout),
    .if_num_data_valid(memory_state_c_num_data_valid),
    .if_fifo_cap(memory_state_c_fifo_cap),
    .if_empty_n(memory_state_c_empty_n),
    .if_read(writeback_indexed_order_2_U0_memory_state_read)
);

STATE_AXI_start_for_writeback_indexed_order_2_U0 start_for_writeback_indexed_order_2_U0_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .if_read_ce(1'b1),
    .if_write_ce(1'b1),
    .if_din(start_for_writeback_indexed_order_2_U0_din),
    .if_full_n(start_for_writeback_indexed_order_2_U0_full_n),
    .if_write(replay_vit_delta_order_indexed_U0_start_write),
    .if_dout(start_for_writeback_indexed_order_2_U0_dout),
    .if_empty_n(start_for_writeback_indexed_order_2_U0_empty_n),
    .if_read(writeback_indexed_order_2_U0_ap_ready)
);

assign ap_done = ap_sync_done;

assign ap_idle = (writeback_indexed_order_2_U0_ap_idle & replay_vit_delta_order_indexed_U0_ap_idle);

assign ap_ready = replay_vit_delta_order_indexed_U0_ap_ready;

assign ap_sync_continue = (ap_sync_done & ap_continue);

assign ap_sync_done = (writeback_indexed_order_2_U0_ap_done & replay_vit_delta_order_indexed_U0_ap_done);

assign m_axi_gmem1_ARADDR = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARADDR;

assign m_axi_gmem1_ARBURST = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARBURST;

assign m_axi_gmem1_ARCACHE = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARCACHE;

assign m_axi_gmem1_ARID = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARID;

assign m_axi_gmem1_ARLEN = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARLEN;

assign m_axi_gmem1_ARLOCK = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARLOCK;

assign m_axi_gmem1_ARPROT = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARPROT;

assign m_axi_gmem1_ARQOS = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARQOS;

assign m_axi_gmem1_ARREGION = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARREGION;

assign m_axi_gmem1_ARSIZE = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARSIZE;

assign m_axi_gmem1_ARUSER = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARUSER;

assign m_axi_gmem1_ARVALID = replay_vit_delta_order_indexed_U0_m_axi_gmem1_ARVALID;

assign m_axi_gmem1_AWADDR = writeback_indexed_order_2_U0_m_axi_gmem1_AWADDR;

assign m_axi_gmem1_AWBURST = writeback_indexed_order_2_U0_m_axi_gmem1_AWBURST;

assign m_axi_gmem1_AWCACHE = writeback_indexed_order_2_U0_m_axi_gmem1_AWCACHE;

assign m_axi_gmem1_AWID = writeback_indexed_order_2_U0_m_axi_gmem1_AWID;

assign m_axi_gmem1_AWLEN = writeback_indexed_order_2_U0_m_axi_gmem1_AWLEN;

assign m_axi_gmem1_AWLOCK = writeback_indexed_order_2_U0_m_axi_gmem1_AWLOCK;

assign m_axi_gmem1_AWPROT = writeback_indexed_order_2_U0_m_axi_gmem1_AWPROT;

assign m_axi_gmem1_AWQOS = writeback_indexed_order_2_U0_m_axi_gmem1_AWQOS;

assign m_axi_gmem1_AWREGION = writeback_indexed_order_2_U0_m_axi_gmem1_AWREGION;

assign m_axi_gmem1_AWSIZE = writeback_indexed_order_2_U0_m_axi_gmem1_AWSIZE;

assign m_axi_gmem1_AWUSER = writeback_indexed_order_2_U0_m_axi_gmem1_AWUSER;

assign m_axi_gmem1_AWVALID = writeback_indexed_order_2_U0_m_axi_gmem1_AWVALID;

assign m_axi_gmem1_BREADY = writeback_indexed_order_2_U0_m_axi_gmem1_BREADY;

assign m_axi_gmem1_RREADY = replay_vit_delta_order_indexed_U0_m_axi_gmem1_RREADY;

assign m_axi_gmem1_WDATA = writeback_indexed_order_2_U0_m_axi_gmem1_WDATA;

assign m_axi_gmem1_WID = writeback_indexed_order_2_U0_m_axi_gmem1_WID;

assign m_axi_gmem1_WLAST = writeback_indexed_order_2_U0_m_axi_gmem1_WLAST;

assign m_axi_gmem1_WSTRB = writeback_indexed_order_2_U0_m_axi_gmem1_WSTRB;

assign m_axi_gmem1_WUSER = writeback_indexed_order_2_U0_m_axi_gmem1_WUSER;

assign m_axi_gmem1_WVALID = writeback_indexed_order_2_U0_m_axi_gmem1_WVALID;

assign replay_vit_delta_order_indexed_U0_ap_continue = ap_sync_continue;

assign replay_vit_delta_order_indexed_U0_ap_start = ap_start;

assign start_for_writeback_indexed_order_2_U0_din = 1'b1;

assign writeback_indexed_order_2_U0_ap_continue = ap_sync_continue;

assign writeback_indexed_order_2_U0_ap_start = start_for_writeback_indexed_order_2_U0_empty_n;

assign x_stream_TDATA = replay_vit_delta_order_indexed_U0_x_stream_TDATA;

assign x_stream_TVALID = replay_vit_delta_order_indexed_U0_x_stream_TVALID;

assign y_stream_TREADY = writeback_indexed_order_2_U0_y_stream_TREADY;

endmodule //STATE_AXI_replay_writeback_vit_delta_order
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_writeback_indexed_order_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state_dout,
        memory_state_num_data_valid,
        memory_state_fifo_cap,
        memory_state_empty_n,
        memory_state_read,
        y_stream_TDATA,
        y_stream_TVALID,
        y_stream_TREADY,
        tile_idx_stream_dout,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_empty_n,
        tile_idx_stream_read
);

parameter    ap_ST_fsm_state1 = 3'd1;
parameter    ap_ST_fsm_state2 = 3'd2;
parameter    ap_ST_fsm_state3 = 3'd4;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state_dout;
input  [2:0] memory_state_num_data_valid;
input  [2:0] memory_state_fifo_cap;
input   memory_state_empty_n;
output   memory_state_read;
input  [199:0] y_stream_TDATA;
input   y_stream_TVALID;
output   y_stream_TREADY;
input  [31:0] tile_idx_stream_dout;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_empty_n;
output   tile_idx_stream_read;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg m_axi_gmem1_AWVALID;
reg m_axi_gmem1_WVALID;
reg m_axi_gmem1_BREADY;
reg memory_state_read;
reg y_stream_TREADY;
reg tile_idx_stream_read;

reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    memory_state_blk_n;
reg   [63:0] memory_state_read_reg_63;
reg    ap_block_state1;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_idle;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_ready;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_tile_idx_stream_read;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWVALID;
wire   [63:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWADDR;
wire   [0:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWID;
wire   [31:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLEN;
wire   [2:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWBURST;
wire   [1:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWPROT;
wire   [3:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWQOS;
wire   [3:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWREGION;
wire   [0:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWUSER;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WVALID;
wire   [127:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WDATA;
wire   [15:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WSTRB;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WLAST;
wire   [0:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WID;
wire   [0:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WUSER;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARVALID;
wire   [63:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARADDR;
wire   [0:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARID;
wire   [31:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLEN;
wire   [2:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARBURST;
wire   [1:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARPROT;
wire   [3:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARQOS;
wire   [3:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARREGION;
wire   [0:0] grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARUSER;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_RREADY;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_BREADY;
wire    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_y_stream_TREADY;
reg    grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg;
wire    ap_CS_fsm_state2;
wire    ap_CS_fsm_state3;
reg   [2:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 3'd1;
//#0 grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg = 1'b0;
end

STATE_AXI_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1 grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start),
    .ap_done(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done),
    .ap_idle(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_idle),
    .ap_ready(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_ready),
    .tile_idx_stream_dout(tile_idx_stream_dout),
    .tile_idx_stream_num_data_valid(7'd0),
    .tile_idx_stream_fifo_cap(7'd0),
    .tile_idx_stream_empty_n(tile_idx_stream_empty_n),
    .tile_idx_stream_read(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_tile_idx_stream_read),
    .y_stream_TVALID(y_stream_TVALID),
    .m_axi_gmem1_AWVALID(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(m_axi_gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(m_axi_gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(9'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(m_axi_gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(m_axi_gmem1_BRESP),
    .m_axi_gmem1_BID(m_axi_gmem1_BID),
    .m_axi_gmem1_BUSER(m_axi_gmem1_BUSER),
    .y_stream_TDATA(y_stream_TDATA),
    .y_stream_TREADY(grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_y_stream_TREADY),
    .memory_state_load(memory_state_read_reg_63)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg <= 1'b1;
        end else if ((grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_ready == 1'b1)) begin
            grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
        memory_state_read_reg_63 <= memory_state_dout;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWVALID = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWVALID;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_BREADY = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_BREADY;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_WVALID = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WVALID;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        memory_state_blk_n = memory_state_empty_n;
    end else begin
        memory_state_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
        memory_state_read = 1'b1;
    end else begin
        memory_state_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        tile_idx_stream_read = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_tile_idx_stream_read;
    end else begin
        tile_idx_stream_read = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        y_stream_TREADY = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_y_stream_TREADY;
    end else begin
        y_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

always @ (*) begin
    ap_block_state1 = ((ap_start == 1'b0) | (memory_state_empty_n == 1'b0) | (ap_done_reg == 1'b1));
end

assign grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_ap_start_reg;

assign m_axi_gmem1_ARADDR = 64'd0;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd0;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_ARVALID = 1'b0;

assign m_axi_gmem1_AWADDR = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWADDR;

assign m_axi_gmem1_AWBURST = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWBURST;

assign m_axi_gmem1_AWCACHE = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWCACHE;

assign m_axi_gmem1_AWID = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWID;

assign m_axi_gmem1_AWLEN = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLEN;

assign m_axi_gmem1_AWLOCK = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWLOCK;

assign m_axi_gmem1_AWPROT = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWPROT;

assign m_axi_gmem1_AWQOS = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWQOS;

assign m_axi_gmem1_AWREGION = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWREGION;

assign m_axi_gmem1_AWSIZE = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWSIZE;

assign m_axi_gmem1_AWUSER = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_AWUSER;

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WDATA = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WDATA;

assign m_axi_gmem1_WID = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WID;

assign m_axi_gmem1_WLAST = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WLAST;

assign m_axi_gmem1_WSTRB = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WSTRB;

assign m_axi_gmem1_WUSER = grp_writeback_indexed_order_2_Pipeline_VITIS_LOOP_137_1_fu_52_m_axi_gmem1_WUSER;

endmodule //STATE_AXI_writeback_indexed_order_2
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1 ns / 1 ps

module STATE_AXI_flow_control_loop_pipe_sequential_init(
        ap_clk,
        ap_rst,
        ap_start,
        ap_ready,
        ap_done,
        ap_start_int,
        ap_ready_int,
        ap_done_int,
        ap_continue_int,
        ap_loop_init,
        ap_loop_exit_ready,
        ap_loop_exit_done
);

input   ap_clk;
input   ap_rst;

//Block level handshake with outside loop
input   ap_start;
output  ap_ready;
output  ap_done;

//Block level handshake with loop body
output  ap_start_int;
input   ap_ready_int;
input   ap_done_int;
output  ap_continue_int;

//Init live in variables
output   ap_loop_init;
wire     ap_loop_init;
reg ap_loop_init_int;
reg ap_done;
reg ap_done_cache;

//Exit signal from loop body
input   ap_loop_exit_ready;
input   ap_loop_exit_done;

// power-on initialization
initial begin
//#0 ap_loop_init_int = 1'b1;
//#0 ap_done_cache = 1'b0;
end

assign ap_start_int = ap_start;

assign ap_continue_int = 1'b1;

assign ap_ready = ap_loop_exit_ready;

//ap_loop_init is valid for the first II
//of the first loop run so as to enable
//the init block ops which are pushed into
//the first state of the pipeline region
always @ (posedge ap_clk)
begin
    if (ap_rst == 1'b1) begin
        ap_loop_init_int <= 1'b1;
    end else if(ap_loop_exit_done == 1'b1) begin
        ap_loop_init_int <= 1'b1;
    end else if(ap_ready_int == 1'b1) begin
        ap_loop_init_int <= 1'b0;
    end
end

assign ap_loop_init = ap_loop_init_int & ap_start;

// if no ap_continue port and current module is not STATE_AXI module, 
// ap_done handshakes with ap_start. Internally, flow control sends out 
// ap_conintue_int = 1'b1 so the ap_done_int is asserted high for 1 clock cycle.
// ap_done_cache is used to record ap_done_int, and de-assert if ap_start_int
// is asserted, so DUT can start the next run
always @(posedge ap_clk)
begin
    if (ap_rst == 1'b1) begin
        ap_done_cache <= 1'b0;
    end else if (ap_done_int == 1'b1) begin
        ap_done_cache <= 1'b1;
    end else if (ap_start_int == 1'b1) begin
        ap_done_cache <= 1'b0;
    end
end

// if no ap_continue port and current module is not STATE_AXI module, ap_done handshakes with ap_start
always @(*)
begin
    if ((ap_done_int == 1'b1) || ((ap_done_cache == 1'b1) && (ap_start_int == 1'b0))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

endmodule
        
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_writeback_vit_delta_order (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        y_stream_TVALID,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state,
        y_stream_TDATA,
        y_stream_TREADY
);

parameter    ap_ST_iter0_fsm_state1 = 4'd1;
parameter    ap_ST_iter0_fsm_state2 = 4'd2;
parameter    ap_ST_iter0_fsm_state3 = 4'd4;
parameter    ap_ST_iter0_fsm_state4 = 4'd8;
parameter    ap_ST_iter1_fsm_state5 = 5'd2;
parameter    ap_ST_iter1_fsm_state6 = 5'd4;
parameter    ap_ST_iter1_fsm_state7 = 5'd8;
parameter    ap_ST_iter1_fsm_state8 = 5'd16;
parameter    ap_ST_iter2_fsm_state9 = 5'd2;
parameter    ap_ST_iter2_fsm_state10 = 5'd4;
parameter    ap_ST_iter2_fsm_state11 = 5'd8;
parameter    ap_ST_iter2_fsm_state12 = 5'd16;
parameter    ap_ST_iter1_fsm_state0 = 5'd1;
parameter    ap_ST_iter2_fsm_state0 = 5'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   y_stream_TVALID;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state;
input  [199:0] y_stream_TDATA;
output   y_stream_TREADY;

reg ap_idle;
reg m_axi_gmem1_AWVALID;
reg m_axi_gmem1_WVALID;
reg[127:0] m_axi_gmem1_WDATA;
reg m_axi_gmem1_BREADY;
reg y_stream_TREADY;

reg   [3:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [4:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [4:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
wire    ap_CS_iter0_fsm_state4;
reg   [0:0] icmp_ln116_reg_655;
reg    ap_block_state4_pp0_stage3_iter0;
reg   [0:0] icmp_ln116_reg_655_pp0_iter1_reg;
reg    ap_block_state12_pp0_stage3_iter2;
wire    ap_CS_iter2_fsm_state12;
reg    ap_block_state5_pp0_stage0_iter1;
reg   [0:0] icmp_ln116_reg_655_pp0_iter0_reg;
reg    ap_block_state5_io;
wire    ap_CS_iter1_fsm_state5;
reg    ap_block_state6_io;
wire    ap_CS_iter1_fsm_state6;
reg    ap_block_state7_io;
wire    ap_CS_iter1_fsm_state7;
wire    ap_CS_iter1_fsm_state8;
reg    ap_condition_exit_pp0_iter0_stage3;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem1_blk_n_AW;
reg    gmem1_blk_n_W;
reg    gmem1_blk_n_B;
reg    y_stream_TDATA_blk_n;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln116_fu_228_p2;
reg   [3:0] tp_load_reg_659;
wire   [0:0] icmp_ln117_fu_249_p2;
reg   [0:0] icmp_ln117_reg_664;
wire   [0:0] and_ln116_fu_267_p2;
reg   [0:0] and_ln116_reg_671;
wire   [7:0] select_ln116_1_fu_279_p3;
reg   [7:0] select_ln116_1_reg_677;
wire   [6:0] empty_fu_287_p1;
reg   [6:0] empty_reg_682;
wire   [10:0] add_ln117_1_fu_291_p2;
reg   [10:0] add_ln117_1_reg_688;
wire   [6:0] select_ln117_1_fu_329_p3;
reg   [6:0] select_ln117_1_reg_693;
wire    ap_CS_iter0_fsm_state2;
wire   [11:0] sub_ln120_fu_392_p2;
reg   [11:0] sub_ln120_reg_699;
reg   [10:0] tmp_reg_704;
wire   [18:0] add_ln120_fu_445_p2;
reg   [18:0] add_ln120_reg_709;
wire    ap_CS_iter0_fsm_state3;
wire   [24:0] trunc_ln121_fu_471_p1;
reg   [24:0] trunc_ln121_reg_714;
reg   [24:0] trunc_ln121_1_reg_719;
reg   [24:0] trunc_ln121_2_reg_724;
reg   [24:0] packet_reg_729;
reg   [24:0] trunc_ln121_4_reg_734;
reg   [24:0] trunc_ln121_5_reg_739;
reg   [24:0] trunc_ln121_6_reg_744;
reg   [24:0] packet_2_reg_749;
reg   [59:0] trunc_ln1_reg_754;
wire  signed [63:0] sext_ln118_fu_555_p1;
wire  signed [127:0] sext_ln225_6_fu_585_p1;
wire  signed [127:0] sext_ln225_7_fu_610_p1;
reg   [3:0] tp_fu_144;
wire   [3:0] add_ln118_fu_408_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_tp_load;
reg   [6:0] ct_fu_148;
reg   [10:0] indvar_flatten_fu_152;
wire   [10:0] select_ln117_2_fu_414_p3;
reg   [10:0] ap_sig_allocacmp_indvar_flatten_load;
reg   [7:0] tt_d_fu_156;
reg   [7:0] ap_sig_allocacmp_tt_d_load;
reg   [16:0] indvar_flatten12_fu_160;
wire   [16:0] add_ln116_fu_234_p2;
reg   [16:0] ap_sig_allocacmp_indvar_flatten12_load;
wire   [0:0] icmp_ln118_fu_261_p2;
wire   [0:0] xor_ln116_fu_255_p2;
wire   [7:0] add_ln116_1_fu_273_p2;
wire   [6:0] select_ln116_fu_305_p3;
wire   [0:0] or_ln117_fu_318_p2;
wire   [6:0] add_ln117_fu_312_p2;
wire   [17:0] p_shl_fu_336_p3;
wire   [15:0] p_shl1_fu_347_p3;
wire   [18:0] p_shl_cast_fu_343_p1;
wire   [18:0] p_shl1_cast_fu_354_p1;
wire   [3:0] select_ln117_fu_322_p3;
wire   [2:0] trunc_ln120_fu_364_p1;
wire   [10:0] shl_ln_fu_368_p3;
wire   [8:0] shl_ln120_1_fu_380_p3;
wire   [11:0] zext_ln120_fu_376_p1;
wire   [11:0] zext_ln120_1_fu_388_p1;
wire   [18:0] empty_31_fu_358_p2;
wire   [18:0] or_ln_fu_437_p4;
wire  signed [18:0] sext_ln120_fu_434_p1;
wire   [22:0] tmp_5_fu_455_p3;
wire  signed [63:0] sext_ln120_1_fu_462_p1;
wire   [63:0] add_ln120_1_fu_466_p2;
wire  signed [31:0] sext_ln225_fu_565_p1;
wire  signed [31:0] sext_ln225_1_fu_568_p1;
wire  signed [31:0] sext_ln225_2_fu_571_p1;
wire   [120:0] packet_1_fu_574_p5;
wire  signed [31:0] sext_ln225_3_fu_590_p1;
wire  signed [31:0] sext_ln225_4_fu_593_p1;
wire  signed [31:0] sext_ln225_5_fu_596_p1;
wire   [120:0] packet_3_fu_599_p5;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg   [3:0] ap_NS_iter0_fsm;
reg   [4:0] ap_NS_iter1_fsm;
reg   [4:0] ap_NS_iter2_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter0_fsm_state3_blk;
reg    ap_ST_iter0_fsm_state4_blk;
reg    ap_ST_iter1_fsm_state5_blk;
reg    ap_ST_iter1_fsm_state6_blk;
reg    ap_ST_iter1_fsm_state7_blk;
wire    ap_ST_iter1_fsm_state8_blk;
wire    ap_ST_iter2_fsm_state9_blk;
wire    ap_ST_iter2_fsm_state10_blk;
wire    ap_ST_iter2_fsm_state11_blk;
reg    ap_ST_iter2_fsm_state12_blk;
wire    ap_start_int;
reg    ap_condition_217;
reg    ap_condition_659;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 4'd1;
//#0 ap_CS_iter1_fsm = 5'd1;
//#0 ap_CS_iter2_fsm = 5'd1;
//#0 tp_fu_144 = 4'd0;
//#0 ct_fu_148 = 7'd0;
//#0 indvar_flatten_fu_152 = 11'd0;
//#0 tt_d_fu_156 = 8'd0;
//#0 indvar_flatten12_fu_160 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage3),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage3_iter2))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((ap_loop_exit_ready_pp0_iter1_reg == 1'b0) & (1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage3_iter2))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter1_fsm_state8))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ct_fu_148 <= 7'd0;
    end else if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (icmp_ln116_reg_655 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state3))) begin
        ct_fu_148 <= select_ln117_1_reg_693;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_217)) begin
        if ((icmp_ln116_fu_228_p2 == 1'd0)) begin
            indvar_flatten12_fu_160 <= add_ln116_fu_234_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten12_fu_160 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        indvar_flatten_fu_152 <= 11'd0;
    end else if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (icmp_ln116_reg_655 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        indvar_flatten_fu_152 <= select_ln117_2_fu_414_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        tp_fu_144 <= 4'd0;
    end else if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (icmp_ln116_reg_655 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        tp_fu_144 <= add_ln118_fu_408_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        tt_d_fu_156 <= 8'd0;
    end else if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (icmp_ln116_reg_655 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        tt_d_fu_156 <= select_ln116_1_reg_677;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        add_ln117_1_reg_688 <= add_ln117_1_fu_291_p2;
        and_ln116_reg_671 <= and_ln116_fu_267_p2;
        empty_reg_682 <= empty_fu_287_p1;
        icmp_ln116_reg_655 <= icmp_ln116_fu_228_p2;
        icmp_ln117_reg_664 <= icmp_ln117_fu_249_p2;
        select_ln116_1_reg_677 <= select_ln116_1_fu_279_p3;
        tp_load_reg_659 <= ap_sig_allocacmp_tp_load;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state3))) begin
        add_ln120_reg_709[18 : 1] <= add_ln120_fu_445_p2[18 : 1];
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_pp0_stage3_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln116_reg_655_pp0_iter0_reg <= icmp_ln116_reg_655;
        packet_2_reg_749 <= {{y_stream_TDATA[199:175]}};
        packet_reg_729 <= {{y_stream_TDATA[99:75]}};
        trunc_ln121_1_reg_719 <= {{y_stream_TDATA[49:25]}};
        trunc_ln121_2_reg_724 <= {{y_stream_TDATA[74:50]}};
        trunc_ln121_4_reg_734 <= {{y_stream_TDATA[124:100]}};
        trunc_ln121_5_reg_739 <= {{y_stream_TDATA[149:125]}};
        trunc_ln121_6_reg_744 <= {{y_stream_TDATA[174:150]}};
        trunc_ln121_reg_714 <= trunc_ln121_fu_471_p1;
        trunc_ln1_reg_754 <= {{add_ln120_1_fu_466_p2[63:4]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter1_fsm_state8))) begin
        icmp_ln116_reg_655_pp0_iter1_reg <= icmp_ln116_reg_655_pp0_iter0_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        select_ln117_1_reg_693 <= select_ln117_1_fu_329_p3;
        sub_ln120_reg_699[11 : 6] <= sub_ln120_fu_392_p2[11 : 6];
        tmp_reg_704 <= {{empty_31_fu_358_p2[18:8]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter0_fsm_state2_blk = 1'b0;

assign ap_ST_iter0_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_pp0_stage3_iter0)) begin
        ap_ST_iter0_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) begin
        ap_ST_iter1_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state6_io)) begin
        ap_ST_iter1_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state6_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state7_io)) begin
        ap_ST_iter1_fsm_state7_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state7_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state8_blk = 1'b0;

assign ap_ST_iter2_fsm_state10_blk = 1'b0;

assign ap_ST_iter2_fsm_state11_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state12_pp0_stage3_iter2)) begin
        ap_ST_iter2_fsm_state12_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state12_blk = 1'b0;
    end
end

assign ap_ST_iter2_fsm_state9_blk = 1'b0;

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_pp0_stage3_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (icmp_ln116_reg_655 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
        ap_condition_exit_pp0_iter0_stage3 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage3 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage3_iter2))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_pp0_stage3_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten12_load = 17'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten12_load = indvar_flatten12_fu_160;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 11'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_152;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_tp_load = 4'd0;
    end else begin
        ap_sig_allocacmp_tp_load = tp_fu_144;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_tt_d_load = 8'd0;
    end else begin
        ap_sig_allocacmp_tt_d_load = tt_d_fu_156;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state5) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0))) begin
        gmem1_blk_n_AW = m_axi_gmem1_AWREADY;
    end else begin
        gmem1_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state12) & (icmp_ln116_reg_655_pp0_iter1_reg == 1'd0))) begin
        gmem1_blk_n_B = m_axi_gmem1_BVALID;
    end else begin
        gmem1_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter1_fsm_state7) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0)))) begin
        gmem1_blk_n_W = m_axi_gmem1_WREADY;
    end else begin
        gmem1_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter1_fsm_state5) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0))) begin
        m_axi_gmem1_AWVALID = 1'b1;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage3_iter2) & (icmp_ln116_reg_655_pp0_iter1_reg == 1'd0))) begin
        m_axi_gmem1_BREADY = 1'b1;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_condition_659)) begin
        if ((1'b1 == ap_CS_iter1_fsm_state7)) begin
            m_axi_gmem1_WDATA = sext_ln225_7_fu_610_p1;
        end else if ((1'b1 == ap_CS_iter1_fsm_state6)) begin
            m_axi_gmem1_WDATA = sext_ln225_6_fu_585_p1;
        end else begin
            m_axi_gmem1_WDATA = 'bx;
        end
    end else begin
        m_axi_gmem1_WDATA = 'bx;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state7_io) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter1_fsm_state7) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0)) | (~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter1_fsm_state6) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0)))) begin
        m_axi_gmem1_WVALID = 1'b1;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln116_reg_655 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
        y_stream_TDATA_blk_n = y_stream_TVALID;
    end else begin
        y_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_pp0_stage3_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (icmp_ln116_reg_655 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
        y_stream_TREADY = 1'b1;
    end else begin
        y_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state3;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        ap_ST_iter0_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state3))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state4;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state3;
            end
        end
        ap_ST_iter0_fsm_state4 : begin
            if (~((1'b1 == ap_block_state4_pp0_stage3_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state4;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state5 : begin
            if ((~((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter1_fsm_state5))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state5;
            end
        end
        ap_ST_iter1_fsm_state6 : begin
            if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter1_fsm_state6))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state7;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end
        end
        ap_ST_iter1_fsm_state7 : begin
            if ((~((1'b1 == ap_block_state7_io) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter1_fsm_state7))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state8;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state7;
            end
        end
        ap_ST_iter1_fsm_state8 : begin
            if ((~((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter0_fsm_state4) & (1'b0 == ap_block_state4_pp0_stage3_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state5;
            end else if ((~((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2)) & ((1'b0 == ap_CS_iter0_fsm_state4) | ((1'b1 == ap_CS_iter0_fsm_state4) & (1'b1 == ap_block_state4_pp0_stage3_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state8;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state4_pp0_stage3_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state5;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state9 : begin
            ap_NS_iter2_fsm = ap_ST_iter2_fsm_state10;
        end
        ap_ST_iter2_fsm_state10 : begin
            ap_NS_iter2_fsm = ap_ST_iter2_fsm_state11;
        end
        ap_ST_iter2_fsm_state11 : begin
            ap_NS_iter2_fsm = ap_ST_iter2_fsm_state12;
        end
        ap_ST_iter2_fsm_state12 : begin
            if (((1'b0 == ap_CS_iter1_fsm_state8) & (1'b0 == ap_block_state12_pp0_stage3_iter2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter1_fsm_state8) & (1'b0 == ap_block_state12_pp0_stage3_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage3_iter2) & (icmp_ln116_reg_655_pp0_iter1_reg == 1'd1)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state9;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state12;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter1_fsm_state8))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state9;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

assign add_ln116_1_fu_273_p2 = (ap_sig_allocacmp_tt_d_load + 8'd1);

assign add_ln116_fu_234_p2 = (ap_sig_allocacmp_indvar_flatten12_load + 17'd1);

assign add_ln117_1_fu_291_p2 = (ap_sig_allocacmp_indvar_flatten_load + 11'd1);

assign add_ln117_fu_312_p2 = (select_ln116_fu_305_p3 + 7'd1);

assign add_ln118_fu_408_p2 = (select_ln117_fu_322_p3 + 4'd1);

assign add_ln120_1_fu_466_p2 = ($signed(sext_ln120_1_fu_462_p1) + $signed(memory_state));

assign add_ln120_fu_445_p2 = ($signed(or_ln_fu_437_p4) + $signed(sext_ln120_fu_434_p1));

assign and_ln116_fu_267_p2 = (xor_ln116_fu_255_p2 & icmp_ln118_fu_261_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter0_fsm_state3 = ap_CS_iter0_fsm[32'd2];

assign ap_CS_iter0_fsm_state4 = ap_CS_iter0_fsm[32'd3];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state5 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state6 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter1_fsm_state7 = ap_CS_iter1_fsm[32'd3];

assign ap_CS_iter1_fsm_state8 = ap_CS_iter1_fsm[32'd4];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state12 = ap_CS_iter2_fsm[32'd4];

always @ (*) begin
    ap_block_state12_pp0_stage3_iter2 = ((icmp_ln116_reg_655_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_BVALID == 1'b0));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state4_pp0_stage3_iter0 = ((icmp_ln116_reg_655 == 1'd0) & (y_stream_TVALID == 1'b0));
end

always @ (*) begin
    ap_block_state5_io = ((m_axi_gmem1_AWREADY == 1'b0) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0));
end

always @ (*) begin
    ap_block_state5_pp0_stage0_iter1 = ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2));
end

always @ (*) begin
    ap_block_state6_io = ((m_axi_gmem1_WREADY == 1'b0) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0));
end

always @ (*) begin
    ap_block_state7_io = ((m_axi_gmem1_WREADY == 1'b0) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0));
end

always @ (*) begin
    ap_condition_217 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state7) & (1'b1 == ap_block_state7_io)) | ((1'b1 == ap_CS_iter1_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter1_fsm_state5) & ((1'b1 == ap_block_state5_io) | (1'b1 == ap_block_state5_pp0_stage0_iter1))) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

always @ (*) begin
    ap_condition_659 = (~((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage3_iter2)) & (icmp_ln116_reg_655_pp0_iter0_reg == 1'd0));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage3;

assign empty_31_fu_358_p2 = (p_shl_cast_fu_343_p1 - p_shl1_cast_fu_354_p1);

assign empty_fu_287_p1 = select_ln116_1_fu_279_p3[6:0];

assign icmp_ln116_fu_228_p2 = ((ap_sig_allocacmp_indvar_flatten12_load == 17'd98304) ? 1'b1 : 1'b0);

assign icmp_ln117_fu_249_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 11'd768) ? 1'b1 : 1'b0);

assign icmp_ln118_fu_261_p2 = ((ap_sig_allocacmp_tp_load == 4'd8) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = 64'd0;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd0;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_ARVALID = 1'b0;

assign m_axi_gmem1_AWADDR = sext_ln118_fu_555_p1;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd2;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd65535;

assign m_axi_gmem1_WUSER = 1'd0;

assign or_ln117_fu_318_p2 = (icmp_ln117_reg_664 | and_ln116_reg_671);

assign or_ln_fu_437_p4 = {{{tmp_reg_704}, {select_ln117_1_reg_693}}, {1'd0}};

assign p_shl1_cast_fu_354_p1 = p_shl1_fu_347_p3;

assign p_shl1_fu_347_p3 = {{empty_reg_682}, {9'd0}};

assign p_shl_cast_fu_343_p1 = p_shl_fu_336_p3;

assign p_shl_fu_336_p3 = {{empty_reg_682}, {11'd0}};

assign packet_1_fu_574_p5 = {{{{packet_reg_729}, {sext_ln225_fu_565_p1}}, {sext_ln225_1_fu_568_p1}}, {sext_ln225_2_fu_571_p1}};

assign packet_3_fu_599_p5 = {{{{packet_2_reg_749}, {sext_ln225_3_fu_590_p1}}, {sext_ln225_4_fu_593_p1}}, {sext_ln225_5_fu_596_p1}};

assign select_ln116_1_fu_279_p3 = ((icmp_ln117_fu_249_p2[0:0] == 1'b1) ? add_ln116_1_fu_273_p2 : ap_sig_allocacmp_tt_d_load);

assign select_ln116_fu_305_p3 = ((icmp_ln117_reg_664[0:0] == 1'b1) ? 7'd0 : ct_fu_148);

assign select_ln117_1_fu_329_p3 = ((and_ln116_reg_671[0:0] == 1'b1) ? add_ln117_fu_312_p2 : select_ln116_fu_305_p3);

assign select_ln117_2_fu_414_p3 = ((icmp_ln117_reg_664[0:0] == 1'b1) ? 11'd1 : add_ln117_1_reg_688);

assign select_ln117_fu_322_p3 = ((or_ln117_fu_318_p2[0:0] == 1'b1) ? 4'd0 : tp_load_reg_659);

assign sext_ln118_fu_555_p1 = $signed(trunc_ln1_reg_754);

assign sext_ln120_1_fu_462_p1 = $signed(tmp_5_fu_455_p3);

assign sext_ln120_fu_434_p1 = $signed(sub_ln120_reg_699);

assign sext_ln225_1_fu_568_p1 = $signed(trunc_ln121_1_reg_719);

assign sext_ln225_2_fu_571_p1 = $signed(trunc_ln121_reg_714);

assign sext_ln225_3_fu_590_p1 = $signed(trunc_ln121_6_reg_744);

assign sext_ln225_4_fu_593_p1 = $signed(trunc_ln121_5_reg_739);

assign sext_ln225_5_fu_596_p1 = $signed(trunc_ln121_4_reg_734);

assign sext_ln225_6_fu_585_p1 = $signed(packet_1_fu_574_p5);

assign sext_ln225_7_fu_610_p1 = $signed(packet_3_fu_599_p5);

assign sext_ln225_fu_565_p1 = $signed(trunc_ln121_2_reg_724);

assign shl_ln120_1_fu_380_p3 = {{trunc_ln120_fu_364_p1}, {6'd0}};

assign shl_ln_fu_368_p3 = {{trunc_ln120_fu_364_p1}, {8'd0}};

assign sub_ln120_fu_392_p2 = (zext_ln120_fu_376_p1 - zext_ln120_1_fu_388_p1);

assign tmp_5_fu_455_p3 = {{add_ln120_reg_709}, {4'd0}};

assign trunc_ln120_fu_364_p1 = select_ln117_fu_322_p3[2:0];

assign trunc_ln121_fu_471_p1 = y_stream_TDATA[24:0];

assign xor_ln116_fu_255_p2 = (icmp_ln117_fu_249_p2 ^ 1'd1);

assign zext_ln120_1_fu_388_p1 = shl_ln120_1_fu_380_p3;

assign zext_ln120_fu_376_p1 = shl_ln_fu_368_p3;

always @ (posedge ap_clk) begin
    sub_ln120_reg_699[5:0] <= 6'b000000;
    add_ln120_reg_709[0] <= 1'b0;
end

endmodule //STATE_AXI_writeback_vit_delta_order
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module STATE_AXI_start_for_writeback_indexed_order_2_U0
#(parameter
    MEM_STYLE    = "shiftReg",
    DATA_WIDTH   = 1,
    ADDR_WIDTH   = 1,
    DEPTH        = 2)
(
    // system signal
    input  wire                  clk,
    input  wire                  reset,

    // write
    output wire                  if_full_n,
    input  wire                  if_write_ce,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,
    
    // read 

    output wire                  if_empty_n,
    input  wire                  if_read_ce,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout
);
//------------------------Parameter----------------------

//------------------------Local signal-------------------
    wire [ADDR_WIDTH-1:0] addr;
    wire                  push;
    wire                  pop;
    reg signed [ADDR_WIDTH:0] mOutPtr;
    reg                   empty_n = 1'b0;
    reg                   full_n = 1'b1; 
    // has num_data_valid?  no 
//------------------------Instantiation------------------
    STATE_AXI_start_for_writeback_indexed_order_2_U0_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_STATE_AXI_start_for_writeback_indexed_order_2_U0_ShiftReg (
        .clk        (clk),
        .we         (push),
        .addr       (addr),
        .din        (if_din),
        .dout       (if_dout)
    );
//------------------------Task and function--------------

//------------------------Body---------------------------
    // num_data_valid 

    // almost full/empty 

    // program full/empty 

    assign if_full_n  = full_n; 
    assign if_empty_n = empty_n;

    assign push       = full_n & if_write_ce & if_write;
    assign pop        = empty_n & if_read_ce & if_read;

    assign addr       = mOutPtr[ADDR_WIDTH] == 1'b0 ? mOutPtr[ADDR_WIDTH-1:0] : {ADDR_WIDTH{1'b0}};

    // mOutPtr
    always @(posedge clk) begin
        if (reset)
            mOutPtr <= {ADDR_WIDTH+1{1'b1}};
        else if (push & ~pop)
            mOutPtr <= mOutPtr + 1'b1;
        else if (~push & pop)
            mOutPtr <= mOutPtr - 1'b1;
    end

    // full_n
    always @(posedge clk) begin
        if (reset)
            full_n <= 1'b1;
        else if ((push & ~pop) && (mOutPtr == DEPTH - 2))
            full_n <= 1'b0;
        else if (~push & pop)
            full_n <= 1'b1;
    end

    // empty_n
    always @(posedge clk) begin
        if (reset)
            empty_n <= 1'b0;
        else if (push & ~pop)
            empty_n <= 1'b1;
        else if ((~push & pop) && (mOutPtr == 0))
            empty_n <= 1'b0;
    end

    // almost_full_n 

    // almost_empty_n 

    // prog_full_n 
 
    // prog_empty_n 

    // num_data_valid 

endmodule  


module STATE_AXI_start_for_writeback_indexed_order_2_U0_ShiftReg
#(parameter
    DATA_WIDTH  = 1,
    ADDR_WIDTH  = 1,
    DEPTH       = 2)
(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] din,
    output wire [DATA_WIDTH-1:0] dout
);

    reg [DATA_WIDTH-1:0] SRL_SIG [0:DEPTH-1];
    integer i;

    always @ (posedge clk) begin
        if (we) begin
            for (i=0; i<DEPTH-1; i=i+1)
                SRL_SIG[i+1] <= SRL_SIG[i];
            SRL_SIG[0] <= din;
        end
    end

    assign dout = SRL_SIG[addr];

endmodule// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_vit_delta_order (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        x_stream_TREADY,
        memory_state,
        x_stream_TDATA,
        x_stream_TVALID
);

parameter    ap_ST_iter0_fsm_state1 = 2'd1;
parameter    ap_ST_iter0_fsm_state2 = 2'd2;
parameter    ap_ST_iter1_fsm_state3 = 3'd2;
parameter    ap_ST_iter1_fsm_state4 = 3'd4;
parameter    ap_ST_iter2_fsm_state5 = 3'd2;
parameter    ap_ST_iter2_fsm_state6 = 3'd4;
parameter    ap_ST_iter3_fsm_state7 = 3'd2;
parameter    ap_ST_iter3_fsm_state8 = 3'd4;
parameter    ap_ST_iter4_fsm_state9 = 3'd2;
parameter    ap_ST_iter4_fsm_state10 = 3'd4;
parameter    ap_ST_iter5_fsm_state11 = 3'd2;
parameter    ap_ST_iter5_fsm_state12 = 3'd4;
parameter    ap_ST_iter6_fsm_state13 = 3'd2;
parameter    ap_ST_iter6_fsm_state14 = 3'd4;
parameter    ap_ST_iter7_fsm_state15 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 3'd1;
parameter    ap_ST_iter2_fsm_state0 = 3'd1;
parameter    ap_ST_iter3_fsm_state0 = 3'd1;
parameter    ap_ST_iter4_fsm_state0 = 3'd1;
parameter    ap_ST_iter5_fsm_state0 = 3'd1;
parameter    ap_ST_iter6_fsm_state0 = 3'd1;
parameter    ap_ST_iter7_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input   x_stream_TREADY;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;

reg ap_idle;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg x_stream_TVALID;

reg   [1:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [2:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [2:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [2:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
reg   [2:0] ap_CS_iter4_fsm;
wire    ap_CS_iter4_fsm_state0;
reg   [2:0] ap_CS_iter5_fsm;
wire    ap_CS_iter5_fsm_state0;
reg   [2:0] ap_CS_iter6_fsm;
wire    ap_CS_iter6_fsm_state0;
reg   [1:0] ap_CS_iter7_fsm;
wire    ap_CS_iter7_fsm_state0;
wire    ap_CS_iter0_fsm_state2;
wire    ap_CS_iter1_fsm_state3;
wire    ap_CS_iter1_fsm_state4;
reg   [0:0] icmp_ln49_reg_551;
reg   [0:0] icmp_ln49_reg_551_pp0_iter1_reg;
reg    ap_block_state5_io;
wire    ap_CS_iter2_fsm_state5;
wire    ap_CS_iter2_fsm_state6;
wire    ap_CS_iter3_fsm_state7;
wire    ap_CS_iter3_fsm_state8;
wire    ap_CS_iter4_fsm_state9;
wire    ap_CS_iter4_fsm_state10;
wire    ap_CS_iter5_fsm_state11;
wire    ap_CS_iter5_fsm_state12;
reg   [0:0] icmp_ln49_reg_551_pp0_iter5_reg;
reg    ap_block_state13_pp0_stage0_iter6;
wire    ap_CS_iter6_fsm_state13;
reg    ap_block_state14_pp0_stage1_iter6;
wire    ap_CS_iter6_fsm_state14;
reg   [0:0] icmp_ln49_reg_551_pp0_iter6_reg;
reg    ap_block_state15_pp0_stage0_iter7;
reg    ap_block_state15_io;
wire    ap_CS_iter7_fsm_state15;
reg    ap_condition_exit_pp0_iter0_stage1;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem1_blk_n_AR;
reg    gmem1_blk_n_R;
reg    x_stream_TDATA_blk_n;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln49_fu_227_p2;
reg   [0:0] icmp_ln49_reg_551_pp0_iter0_reg;
reg   [0:0] icmp_ln49_reg_551_pp0_iter2_reg;
reg   [0:0] icmp_ln49_reg_551_pp0_iter3_reg;
reg   [0:0] icmp_ln49_reg_551_pp0_iter4_reg;
reg   [3:0] tp_load_reg_555;
wire   [0:0] icmp_ln50_fu_248_p2;
reg   [0:0] icmp_ln50_reg_560;
wire   [0:0] and_ln49_fu_266_p2;
reg   [0:0] and_ln49_reg_567;
wire   [7:0] select_ln49_1_fu_278_p3;
reg   [7:0] select_ln49_1_reg_573;
wire   [6:0] empty_fu_286_p1;
reg   [6:0] empty_reg_578;
wire   [10:0] add_ln50_1_fu_290_p2;
reg   [10:0] add_ln50_1_reg_584;
wire   [6:0] select_ln50_1_fu_328_p3;
reg   [6:0] select_ln50_1_reg_589;
wire   [11:0] sub_ln53_fu_391_p2;
reg   [11:0] sub_ln53_reg_595;
reg   [10:0] tmp_8_reg_600;
wire   [18:0] add_ln53_fu_444_p2;
reg   [18:0] add_ln53_reg_605;
reg   [59:0] trunc_ln8_reg_610;
wire   [24:0] trunc_ln210_fu_490_p1;
reg   [24:0] trunc_ln210_reg_621;
reg   [24:0] trunc_ln210_reg_621_pp0_iter6_reg;
reg   [24:0] tmp_reg_626;
reg   [24:0] tmp_reg_626_pp0_iter6_reg;
reg   [24:0] tmp_s_reg_631;
reg   [24:0] tmp_s_reg_631_pp0_iter6_reg;
reg   [24:0] tmp_3_reg_636;
reg   [24:0] tmp_3_reg_636_pp0_iter6_reg;
wire   [24:0] trunc_ln54_fu_494_p1;
reg   [24:0] trunc_ln54_reg_641;
reg   [24:0] tmp_5_reg_646;
reg   [24:0] tmp_6_reg_651;
reg   [24:0] tmp_7_reg_656;
wire  signed [63:0] sext_ln206_fu_480_p1;
reg   [3:0] tp_fu_124;
wire   [3:0] add_ln51_fu_407_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_tp_load;
reg   [6:0] ct_fu_128;
reg   [10:0] indvar_flatten_fu_132;
wire   [10:0] select_ln50_2_fu_413_p3;
reg   [10:0] ap_sig_allocacmp_indvar_flatten_load;
reg   [7:0] tt_d_fu_136;
reg   [7:0] ap_sig_allocacmp_tt_d_load;
reg   [16:0] indvar_flatten12_fu_140;
wire   [16:0] add_ln49_fu_233_p2;
reg   [16:0] ap_sig_allocacmp_indvar_flatten12_load;
wire   [0:0] icmp_ln51_fu_260_p2;
wire   [0:0] xor_ln49_fu_254_p2;
wire   [7:0] add_ln49_1_fu_272_p2;
wire   [6:0] select_ln49_fu_304_p3;
wire   [0:0] or_ln50_fu_317_p2;
wire   [6:0] add_ln50_fu_311_p2;
wire   [17:0] p_shl_fu_335_p3;
wire   [15:0] p_shl1_fu_346_p3;
wire   [18:0] p_shl_cast_fu_342_p1;
wire   [18:0] p_shl1_cast_fu_353_p1;
wire   [3:0] select_ln50_fu_321_p3;
wire   [2:0] trunc_ln53_fu_363_p1;
wire   [10:0] shl_ln_fu_367_p3;
wire   [8:0] shl_ln53_1_fu_379_p3;
wire   [11:0] zext_ln53_fu_375_p1;
wire   [11:0] zext_ln53_1_fu_387_p1;
wire   [18:0] empty_46_fu_357_p2;
wire   [18:0] or_ln_fu_436_p4;
wire  signed [18:0] sext_ln53_fu_433_p1;
wire   [22:0] tmp_9_fu_454_p3;
wire  signed [63:0] sext_ln53_1_fu_461_p1;
wire   [63:0] add_ln53_1_fu_465_p2;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg    ap_loop_exit_ready_pp0_iter6_reg;
reg    ap_loop_exit_ready_pp0_iter7_reg;
reg   [1:0] ap_NS_iter0_fsm;
reg   [2:0] ap_NS_iter1_fsm;
reg   [2:0] ap_NS_iter2_fsm;
reg   [2:0] ap_NS_iter3_fsm;
reg   [2:0] ap_NS_iter4_fsm;
reg   [2:0] ap_NS_iter5_fsm;
reg   [2:0] ap_NS_iter6_fsm;
reg   [1:0] ap_NS_iter7_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter1_fsm_state3_blk;
wire    ap_ST_iter1_fsm_state4_blk;
reg    ap_ST_iter2_fsm_state5_blk;
wire    ap_ST_iter2_fsm_state6_blk;
wire    ap_ST_iter3_fsm_state7_blk;
wire    ap_ST_iter3_fsm_state8_blk;
wire    ap_ST_iter4_fsm_state9_blk;
wire    ap_ST_iter4_fsm_state10_blk;
wire    ap_ST_iter5_fsm_state11_blk;
wire    ap_ST_iter5_fsm_state12_blk;
reg    ap_ST_iter6_fsm_state13_blk;
reg    ap_ST_iter6_fsm_state14_blk;
reg    ap_ST_iter7_fsm_state15_blk;
wire    ap_start_int;
reg    ap_condition_276;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 2'd1;
//#0 ap_CS_iter1_fsm = 3'd1;
//#0 ap_CS_iter2_fsm = 3'd1;
//#0 ap_CS_iter3_fsm = 3'd1;
//#0 ap_CS_iter4_fsm = 3'd1;
//#0 ap_CS_iter5_fsm = 3'd1;
//#0 ap_CS_iter6_fsm = 3'd1;
//#0 ap_CS_iter7_fsm = 2'd1;
//#0 tp_fu_124 = 4'd0;
//#0 ct_fu_128 = 7'd0;
//#0 indvar_flatten_fu_132 = 11'd0;
//#0 tt_d_fu_136 = 8'd0;
//#0 indvar_flatten12_fu_140 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage1),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter3_fsm <= ap_ST_iter3_fsm_state0;
    end else begin
        ap_CS_iter3_fsm <= ap_NS_iter3_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter4_fsm <= ap_ST_iter4_fsm_state0;
    end else begin
        ap_CS_iter4_fsm <= ap_NS_iter4_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter5_fsm <= ap_ST_iter5_fsm_state0;
    end else begin
        ap_CS_iter5_fsm <= ap_NS_iter5_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter6_fsm <= ap_ST_iter6_fsm_state0;
    end else begin
        ap_CS_iter6_fsm <= ap_NS_iter6_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter7_fsm <= ap_ST_iter7_fsm_state0;
    end else begin
        ap_CS_iter7_fsm <= ap_NS_iter7_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (ap_loop_exit_ready_pp0_iter6_reg == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ct_fu_128 <= 7'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3) & (icmp_ln49_reg_551_pp0_iter0_reg == 1'd0))) begin
        ct_fu_128 <= select_ln50_1_reg_589;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_276)) begin
        if ((icmp_ln49_fu_227_p2 == 1'd0)) begin
            indvar_flatten12_fu_140 <= add_ln49_fu_233_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten12_fu_140 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        indvar_flatten_fu_132 <= 11'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln49_reg_551 == 1'd0))) begin
        indvar_flatten_fu_132 <= select_ln50_2_fu_413_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        tp_fu_124 <= 4'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln49_reg_551 == 1'd0))) begin
        tp_fu_124 <= add_ln51_fu_407_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        tt_d_fu_136 <= 8'd0;
    end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln49_reg_551 == 1'd0))) begin
        tt_d_fu_136 <= select_ln49_1_reg_573;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        add_ln50_1_reg_584 <= add_ln50_1_fu_290_p2;
        and_ln49_reg_567 <= and_ln49_fu_266_p2;
        empty_reg_578 <= empty_fu_286_p1;
        icmp_ln49_reg_551 <= icmp_ln49_fu_227_p2;
        icmp_ln50_reg_560 <= icmp_ln50_fu_248_p2;
        select_ln49_1_reg_573 <= select_ln49_1_fu_278_p3;
        tp_load_reg_555 <= ap_sig_allocacmp_tp_load;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
        add_ln53_reg_605[18 : 1] <= add_ln53_fu_444_p2[18 : 1];
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln49_reg_551_pp0_iter0_reg <= icmp_ln49_reg_551;
        select_ln50_1_reg_589 <= select_ln50_1_fu_328_p3;
        sub_ln53_reg_595[11 : 6] <= sub_ln53_fu_391_p2[11 : 6];
        tmp_8_reg_600 <= {{empty_46_fu_357_p2[18:8]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln49_reg_551_pp0_iter1_reg <= icmp_ln49_reg_551_pp0_iter0_reg;
        trunc_ln8_reg_610 <= {{add_ln53_1_fu_465_p2[63:4]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln49_reg_551_pp0_iter2_reg <= icmp_ln49_reg_551_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln49_reg_551_pp0_iter3_reg <= icmp_ln49_reg_551_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        icmp_ln49_reg_551_pp0_iter4_reg <= icmp_ln49_reg_551_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        icmp_ln49_reg_551_pp0_iter5_reg <= icmp_ln49_reg_551_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
        icmp_ln49_reg_551_pp0_iter6_reg <= icmp_ln49_reg_551_pp0_iter5_reg;
        tmp_3_reg_636_pp0_iter6_reg <= tmp_3_reg_636;
        tmp_5_reg_646 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_6_reg_651 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_7_reg_656 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_reg_626_pp0_iter6_reg <= tmp_reg_626;
        tmp_s_reg_631_pp0_iter6_reg <= tmp_s_reg_631;
        trunc_ln210_reg_621_pp0_iter6_reg <= trunc_ln210_reg_621;
        trunc_ln54_reg_641 <= trunc_ln54_fu_494_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state13_pp0_stage0_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state13))) begin
        tmp_3_reg_636 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_reg_626 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_s_reg_631 <= {{m_axi_gmem1_RDATA[88:64]}};
        trunc_ln210_reg_621 <= trunc_ln210_fu_490_p1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter0_fsm_state2_blk = 1'b0;

assign ap_ST_iter1_fsm_state3_blk = 1'b0;

assign ap_ST_iter1_fsm_state4_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state5_io)) begin
        ap_ST_iter2_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state5_blk = 1'b0;
    end
end

assign ap_ST_iter2_fsm_state6_blk = 1'b0;

assign ap_ST_iter3_fsm_state7_blk = 1'b0;

assign ap_ST_iter3_fsm_state8_blk = 1'b0;

assign ap_ST_iter4_fsm_state10_blk = 1'b0;

assign ap_ST_iter4_fsm_state9_blk = 1'b0;

assign ap_ST_iter5_fsm_state11_blk = 1'b0;

assign ap_ST_iter5_fsm_state12_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state13_pp0_stage0_iter6)) begin
        ap_ST_iter6_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_iter6_fsm_state13_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state14_pp0_stage1_iter6)) begin
        ap_ST_iter6_fsm_state14_blk = 1'b1;
    end else begin
        ap_ST_iter6_fsm_state14_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) begin
        ap_ST_iter7_fsm_state15_blk = 1'b1;
    end else begin
        ap_ST_iter7_fsm_state15_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln49_reg_551 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (ap_loop_exit_ready_pp0_iter7_reg == 1'b1))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state0) & (1'b1 == ap_CS_iter6_fsm_state0) & (1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten12_load = 17'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten12_load = indvar_flatten12_fu_140;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 11'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_132;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_tp_load = 4'd0;
    end else begin
        ap_sig_allocacmp_tp_load = tp_fu_124;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_tt_d_load = 8'd0;
    end else begin
        ap_sig_allocacmp_tt_d_load = tt_d_fu_136;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln49_reg_551_pp0_iter1_reg == 1'd0))) begin
        gmem1_blk_n_AR = m_axi_gmem1_ARREADY;
    end else begin
        gmem1_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter6_fsm_state14) & (icmp_ln49_reg_551_pp0_iter5_reg == 1'd0)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (icmp_ln49_reg_551_pp0_iter5_reg == 1'd0)))) begin
        gmem1_blk_n_R = m_axi_gmem1_RVALID;
    end else begin
        gmem1_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln49_reg_551_pp0_iter1_reg == 1'd0))) begin
        m_axi_gmem1_ARVALID = 1'b1;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state13_pp0_stage0_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state13) & (icmp_ln49_reg_551_pp0_iter5_reg == 1'd0)) | (~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14) & (icmp_ln49_reg_551_pp0_iter5_reg == 1'd0)))) begin
        m_axi_gmem1_RREADY = 1'b1;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln49_reg_551_pp0_iter6_reg == 1'd0))) begin
        x_stream_TDATA_blk_n = x_stream_TREADY;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln49_reg_551_pp0_iter6_reg == 1'd0))) begin
        x_stream_TVALID = 1'b1;
    end else begin
        x_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if (~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end
        end
        ap_ST_iter1_fsm_state4 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b0 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state5 : begin
            if ((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state5))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end
        end
        ap_ST_iter2_fsm_state6 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter3_fsm)
        ap_ST_iter3_fsm_state7 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state7))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end
        end
        ap_ST_iter3_fsm_state8 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter3_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter4_fsm)
        ap_ST_iter4_fsm_state9 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state9))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end
        end
        ap_ST_iter4_fsm_state10 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter4_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter5_fsm)
        ap_ST_iter5_fsm_state11 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter5_fsm_state11))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state12;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end
        end
        ap_ST_iter5_fsm_state12 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b0 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state12;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter5_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter6_fsm)
        ap_ST_iter6_fsm_state13 : begin
            if ((~((1'b1 == ap_block_state13_pp0_stage0_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state13))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state14;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end
        end
        ap_ST_iter6_fsm_state14 : begin
            if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end else if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b0 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state14;
            end
        end
        ap_ST_iter6_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6))) & (1'b1 == ap_CS_iter5_fsm_state12))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state13;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter6_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter7_fsm)
        ap_ST_iter7_fsm_state15 : begin
            if ((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & ((1'b0 == ap_CS_iter6_fsm_state14) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6))))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end else if (((~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter7_fsm_state15) & (icmp_ln49_reg_551_pp0_iter6_reg == 1'd1)) | (~((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)) & (1'b1 == ap_CS_iter6_fsm_state14) & (1'b0 == ap_block_state14_pp0_stage1_iter6)))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state15;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state15;
            end
        end
        ap_ST_iter7_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state14_pp0_stage1_iter6) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7)))) & (1'b1 == ap_CS_iter6_fsm_state14))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state15;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter7_fsm = 'bx;
        end
    endcase
end

assign add_ln49_1_fu_272_p2 = (ap_sig_allocacmp_tt_d_load + 8'd1);

assign add_ln49_fu_233_p2 = (ap_sig_allocacmp_indvar_flatten12_load + 17'd1);

assign add_ln50_1_fu_290_p2 = (ap_sig_allocacmp_indvar_flatten_load + 11'd1);

assign add_ln50_fu_311_p2 = (select_ln49_fu_304_p3 + 7'd1);

assign add_ln51_fu_407_p2 = (select_ln50_fu_321_p3 + 4'd1);

assign add_ln53_1_fu_465_p2 = ($signed(sext_ln53_1_fu_461_p1) + $signed(memory_state));

assign add_ln53_fu_444_p2 = ($signed(or_ln_fu_436_p4) + $signed(sext_ln53_fu_433_p1));

assign and_ln49_fu_266_p2 = (xor_ln49_fu_254_p2 & icmp_ln51_fu_260_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state3 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state4 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state5 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter2_fsm_state6 = ap_CS_iter2_fsm[32'd2];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state7 = ap_CS_iter3_fsm[32'd1];

assign ap_CS_iter3_fsm_state8 = ap_CS_iter3_fsm[32'd2];

assign ap_CS_iter4_fsm_state0 = ap_CS_iter4_fsm[32'd0];

assign ap_CS_iter4_fsm_state10 = ap_CS_iter4_fsm[32'd2];

assign ap_CS_iter4_fsm_state9 = ap_CS_iter4_fsm[32'd1];

assign ap_CS_iter5_fsm_state0 = ap_CS_iter5_fsm[32'd0];

assign ap_CS_iter5_fsm_state11 = ap_CS_iter5_fsm[32'd1];

assign ap_CS_iter5_fsm_state12 = ap_CS_iter5_fsm[32'd2];

assign ap_CS_iter6_fsm_state0 = ap_CS_iter6_fsm[32'd0];

assign ap_CS_iter6_fsm_state13 = ap_CS_iter6_fsm[32'd1];

assign ap_CS_iter6_fsm_state14 = ap_CS_iter6_fsm[32'd2];

assign ap_CS_iter7_fsm_state0 = ap_CS_iter7_fsm[32'd0];

assign ap_CS_iter7_fsm_state15 = ap_CS_iter7_fsm[32'd1];

always @ (*) begin
    ap_block_state13_pp0_stage0_iter6 = ((icmp_ln49_reg_551_pp0_iter5_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state14_pp0_stage1_iter6 = ((icmp_ln49_reg_551_pp0_iter5_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state15_io = ((icmp_ln49_reg_551_pp0_iter6_reg == 1'd0) & (x_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state15_pp0_stage0_iter7 = ((icmp_ln49_reg_551_pp0_iter6_reg == 1'd0) & (x_stream_TREADY == 1'b0));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state5_io = ((icmp_ln49_reg_551_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_ARREADY == 1'b0));
end

always @ (*) begin
    ap_condition_276 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter7_fsm_state15) & ((1'b1 == ap_block_state15_io) | (1'b1 == ap_block_state15_pp0_stage0_iter7))) | ((1'b1 == ap_CS_iter6_fsm_state14) & (1'b1 == ap_block_state14_pp0_stage1_iter6)) | ((1'b1 == ap_CS_iter6_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage0_iter6)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage1;

assign empty_46_fu_357_p2 = (p_shl_cast_fu_342_p1 - p_shl1_cast_fu_353_p1);

assign empty_fu_286_p1 = select_ln49_1_fu_278_p3[6:0];

assign icmp_ln49_fu_227_p2 = ((ap_sig_allocacmp_indvar_flatten12_load == 17'd98304) ? 1'b1 : 1'b0);

assign icmp_ln50_fu_248_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 11'd768) ? 1'b1 : 1'b0);

assign icmp_ln51_fu_260_p2 = ((ap_sig_allocacmp_tp_load == 4'd8) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = sext_ln206_fu_480_p1;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd2;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign or_ln50_fu_317_p2 = (icmp_ln50_reg_560 | and_ln49_reg_567);

assign or_ln_fu_436_p4 = {{{tmp_8_reg_600}, {select_ln50_1_reg_589}}, {1'd0}};

assign p_shl1_cast_fu_353_p1 = p_shl1_fu_346_p3;

assign p_shl1_fu_346_p3 = {{empty_reg_578}, {9'd0}};

assign p_shl_cast_fu_342_p1 = p_shl_fu_335_p3;

assign p_shl_fu_335_p3 = {{empty_reg_578}, {11'd0}};

assign select_ln49_1_fu_278_p3 = ((icmp_ln50_fu_248_p2[0:0] == 1'b1) ? add_ln49_1_fu_272_p2 : ap_sig_allocacmp_tt_d_load);

assign select_ln49_fu_304_p3 = ((icmp_ln50_reg_560[0:0] == 1'b1) ? 7'd0 : ct_fu_128);

assign select_ln50_1_fu_328_p3 = ((and_ln49_reg_567[0:0] == 1'b1) ? add_ln50_fu_311_p2 : select_ln49_fu_304_p3);

assign select_ln50_2_fu_413_p3 = ((icmp_ln50_reg_560[0:0] == 1'b1) ? 11'd1 : add_ln50_1_reg_584);

assign select_ln50_fu_321_p3 = ((or_ln50_fu_317_p2[0:0] == 1'b1) ? 4'd0 : tp_load_reg_555);

assign sext_ln206_fu_480_p1 = $signed(trunc_ln8_reg_610);

assign sext_ln53_1_fu_461_p1 = $signed(tmp_9_fu_454_p3);

assign sext_ln53_fu_433_p1 = $signed(sub_ln53_reg_595);

assign shl_ln53_1_fu_379_p3 = {{trunc_ln53_fu_363_p1}, {6'd0}};

assign shl_ln_fu_367_p3 = {{trunc_ln53_fu_363_p1}, {8'd0}};

assign sub_ln53_fu_391_p2 = (zext_ln53_fu_375_p1 - zext_ln53_1_fu_387_p1);

assign tmp_9_fu_454_p3 = {{add_ln53_reg_605}, {4'd0}};

assign trunc_ln210_fu_490_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln53_fu_363_p1 = select_ln50_fu_321_p3[2:0];

assign trunc_ln54_fu_494_p1 = m_axi_gmem1_RDATA[24:0];

assign x_stream_TDATA = {{{{{{{{tmp_7_reg_656}, {tmp_6_reg_651}}, {tmp_5_reg_646}}, {trunc_ln54_reg_641}}, {tmp_3_reg_636_pp0_iter6_reg}}, {tmp_s_reg_631_pp0_iter6_reg}}, {tmp_reg_626_pp0_iter6_reg}}, {trunc_ln210_reg_621_pp0_iter6_reg}};

assign xor_ln49_fu_254_p2 = (icmp_ln50_fu_248_p2 ^ 1'd1);

assign zext_ln53_1_fu_387_p1 = shl_ln53_1_fu_379_p3;

assign zext_ln53_fu_375_p1 = shl_ln_fu_367_p3;

always @ (posedge ap_clk) begin
    sub_ln53_reg_595[5:0] <= 6'b000000;
    add_ln53_reg_605[0] <= 1'b0;
end

endmodule //STATE_AXI_replay_vit_delta_order
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        x_stream_TREADY,
        memory_decoder_state,
        x_stream_TDATA,
        x_stream_TVALID
);

parameter    ap_ST_iter0_fsm_state1 = 5'd1;
parameter    ap_ST_iter0_fsm_state2 = 5'd2;
parameter    ap_ST_iter0_fsm_state3 = 5'd4;
parameter    ap_ST_iter0_fsm_state4 = 5'd8;
parameter    ap_ST_iter0_fsm_state5 = 5'd16;
parameter    ap_ST_iter1_fsm_state6 = 6'd2;
parameter    ap_ST_iter1_fsm_state7 = 6'd4;
parameter    ap_ST_iter1_fsm_state8 = 6'd8;
parameter    ap_ST_iter1_fsm_state9 = 6'd16;
parameter    ap_ST_iter1_fsm_state10 = 6'd32;
parameter    ap_ST_iter2_fsm_state11 = 5'd2;
parameter    ap_ST_iter2_fsm_state12 = 5'd4;
parameter    ap_ST_iter2_fsm_state13 = 5'd8;
parameter    ap_ST_iter2_fsm_state14 = 5'd16;
parameter    ap_ST_iter1_fsm_state0 = 6'd1;
parameter    ap_ST_iter2_fsm_state0 = 5'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input   x_stream_TREADY;
input  [63:0] memory_decoder_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;

reg ap_idle;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg x_stream_TVALID;

reg   [4:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [5:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [4:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
wire    ap_CS_iter0_fsm_state5;
wire    ap_CS_iter1_fsm_state6;
wire    ap_CS_iter1_fsm_state7;
wire    ap_CS_iter1_fsm_state8;
wire    ap_CS_iter1_fsm_state9;
wire    ap_CS_iter1_fsm_state10;
reg   [0:0] icmp_ln36_reg_386;
reg   [0:0] icmp_ln36_reg_386_pp0_iter1_reg;
reg    ap_block_state12_pp0_stage1_iter2;
wire    ap_CS_iter2_fsm_state12;
reg    ap_block_state13_pp0_stage2_iter2;
wire    ap_CS_iter2_fsm_state13;
reg    ap_block_state14_pp0_stage3_iter2;
reg    ap_block_state14_io;
wire    ap_CS_iter2_fsm_state14;
reg    ap_condition_exit_pp0_iter0_stage4;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem1_blk_n_AR;
wire    ap_CS_iter0_fsm_state4;
reg    gmem1_blk_n_R;
reg    x_stream_TDATA_blk_n;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln36_fu_187_p2;
reg   [0:0] icmp_ln36_reg_386_pp0_iter0_reg;
reg   [3:0] t_load_reg_390;
wire   [0:0] icmp_ln37_fu_205_p2;
reg   [0:0] icmp_ln37_reg_395;
wire   [6:0] select_ln36_1_fu_217_p3;
reg   [6:0] select_ln36_1_reg_400;
wire   [11:0] add_ln39_fu_286_p2;
reg   [11:0] add_ln39_reg_405;
wire    ap_CS_iter0_fsm_state2;
reg   [59:0] trunc_ln4_reg_410;
wire    ap_CS_iter0_fsm_state3;
reg    ap_block_state4_io;
wire   [24:0] trunc_ln210_fu_339_p1;
reg   [24:0] trunc_ln210_reg_421;
reg   [24:0] tmp_reg_426;
reg   [24:0] tmp_1_reg_431;
reg   [24:0] tmp_2_reg_436;
wire   [24:0] trunc_ln39_1_fu_343_p1;
reg   [24:0] trunc_ln39_1_reg_441;
reg   [24:0] tmp_4_reg_446;
reg   [24:0] tmp_5_reg_451;
reg   [24:0] tmp_6_reg_456;
wire  signed [63:0] sext_ln206_fu_329_p1;
reg   [3:0] t_fu_102;
wire   [3:0] add_ln37_fu_292_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_t_load;
reg   [6:0] ct_fu_106;
reg   [6:0] ap_sig_allocacmp_ct_load;
reg   [9:0] indvar_flatten_fu_110;
wire   [9:0] add_ln36_fu_193_p2;
reg   [9:0] ap_sig_allocacmp_indvar_flatten_load;
wire   [6:0] add_ln36_1_fu_211_p2;
wire   [3:0] select_ln36_fu_235_p3;
wire   [2:0] trunc_ln39_fu_241_p1;
wire   [10:0] shl_ln1_fu_245_p3;
wire   [6:0] shl_ln39_1_fu_257_p3;
wire   [11:0] zext_ln39_fu_253_p1;
wire   [11:0] zext_ln39_1_fu_265_p1;
wire   [7:0] shl_ln39_2_fu_275_p3;
wire   [11:0] sub_ln39_fu_269_p2;
wire   [11:0] zext_ln39_2_fu_282_p1;
wire   [15:0] tmp_s_fu_303_p3;
wire  signed [63:0] sext_ln39_fu_310_p1;
wire   [63:0] add_ln39_1_fu_314_p2;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg   [4:0] ap_NS_iter0_fsm;
reg   [5:0] ap_NS_iter1_fsm;
reg   [4:0] ap_NS_iter2_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter0_fsm_state3_blk;
reg    ap_ST_iter0_fsm_state4_blk;
wire    ap_ST_iter0_fsm_state5_blk;
wire    ap_ST_iter1_fsm_state6_blk;
wire    ap_ST_iter1_fsm_state7_blk;
wire    ap_ST_iter1_fsm_state8_blk;
wire    ap_ST_iter1_fsm_state9_blk;
wire    ap_ST_iter1_fsm_state10_blk;
wire    ap_ST_iter2_fsm_state11_blk;
reg    ap_ST_iter2_fsm_state12_blk;
reg    ap_ST_iter2_fsm_state13_blk;
reg    ap_ST_iter2_fsm_state14_blk;
wire    ap_start_int;
reg    ap_condition_211;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 5'd1;
//#0 ap_CS_iter1_fsm = 6'd1;
//#0 ap_CS_iter2_fsm = 5'd1;
//#0 t_fu_102 = 4'd0;
//#0 ct_fu_106 = 7'd0;
//#0 indvar_flatten_fu_110 = 10'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage4),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state14))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0) & (1'b1 == ap_CS_iter2_fsm_state14))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_211)) begin
        if ((icmp_ln36_fu_187_p2 == 1'd0)) begin
            ct_fu_106 <= select_ln36_1_fu_217_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            ct_fu_106 <= 7'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_211)) begin
        if ((icmp_ln36_fu_187_p2 == 1'd0)) begin
            indvar_flatten_fu_110 <= add_ln36_fu_193_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_110 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        t_fu_102 <= 4'd0;
    end else if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln36_reg_386 == 1'd0))) begin
        t_fu_102 <= add_ln37_fu_292_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        add_ln39_reg_405[11 : 1] <= add_ln39_fu_286_p2[11 : 1];
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln36_reg_386_pp0_iter0_reg <= icmp_ln36_reg_386;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln36_reg_386 <= icmp_ln36_fu_187_p2;
        icmp_ln37_reg_395 <= icmp_ln37_fu_205_p2;
        select_ln36_1_reg_400 <= select_ln36_1_fu_217_p3;
        t_load_reg_390 <= ap_sig_allocacmp_t_load;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state10))) begin
        icmp_ln36_reg_386_pp0_iter1_reg <= icmp_ln36_reg_386_pp0_iter0_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage1_iter2))) begin
        tmp_1_reg_431 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_2_reg_436 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_reg_426 <= {{m_axi_gmem1_RDATA[56:32]}};
        trunc_ln210_reg_421 <= trunc_ln210_fu_339_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter2_fsm_state13) & (1'b0 == ap_block_state13_pp0_stage2_iter2))) begin
        tmp_4_reg_446 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_5_reg_451 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_6_reg_456 <= {{m_axi_gmem1_RDATA[120:96]}};
        trunc_ln39_1_reg_441 <= trunc_ln39_1_fu_343_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state3))) begin
        trunc_ln4_reg_410 <= {{add_ln39_1_fu_314_p2[63:4]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter0_fsm_state2_blk = 1'b0;

assign ap_ST_iter0_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_io)) begin
        ap_ST_iter0_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state4_blk = 1'b0;
    end
end

assign ap_ST_iter0_fsm_state5_blk = 1'b0;

assign ap_ST_iter1_fsm_state10_blk = 1'b0;

assign ap_ST_iter1_fsm_state6_blk = 1'b0;

assign ap_ST_iter1_fsm_state7_blk = 1'b0;

assign ap_ST_iter1_fsm_state8_blk = 1'b0;

assign ap_ST_iter1_fsm_state9_blk = 1'b0;

assign ap_ST_iter2_fsm_state11_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state12_pp0_stage1_iter2)) begin
        ap_ST_iter2_fsm_state12_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state12_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state13_pp0_stage2_iter2)) begin
        ap_ST_iter2_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state13_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) begin
        ap_ST_iter2_fsm_state14_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state14_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5) & (icmp_ln36_reg_386 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage4 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage4 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state14))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_ct_load = 7'd0;
    end else begin
        ap_sig_allocacmp_ct_load = ct_fu_106;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 10'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_110;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_t_load = 4'd0;
    end else begin
        ap_sig_allocacmp_t_load = t_fu_102;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state4) & (icmp_ln36_reg_386 == 1'd0))) begin
        gmem1_blk_n_AR = m_axi_gmem1_ARREADY;
    end else begin
        gmem1_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter2_fsm_state13) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0)))) begin
        gmem1_blk_n_R = m_axi_gmem1_RVALID;
    end else begin
        gmem1_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state4) & (icmp_ln36_reg_386 == 1'd0))) begin
        m_axi_gmem1_ARVALID = 1'b1;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter2_fsm_state13) & (1'b0 == ap_block_state13_pp0_stage2_iter2) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage1_iter2) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0)))) begin
        m_axi_gmem1_RREADY = 1'b1;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0))) begin
        x_stream_TDATA_blk_n = x_stream_TREADY;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0))) begin
        x_stream_TVALID = 1'b1;
    end else begin
        x_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state3;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        ap_ST_iter0_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state3))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state4;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state3;
            end
        end
        ap_ST_iter0_fsm_state4 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state5;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state4;
            end
        end
        ap_ST_iter0_fsm_state5 : begin
            if (~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state5;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state6 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state6))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state7;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end
        end
        ap_ST_iter1_fsm_state7 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state7))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state8;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state7;
            end
        end
        ap_ST_iter1_fsm_state8 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state8))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state9;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state8;
            end
        end
        ap_ST_iter1_fsm_state9 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state9))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state10;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state9;
            end
        end
        ap_ST_iter1_fsm_state10 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end else if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b0 == ap_CS_iter0_fsm_state5))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state10;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state11 : begin
            ap_NS_iter2_fsm = ap_ST_iter2_fsm_state12;
        end
        ap_ST_iter2_fsm_state12 : begin
            if (((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage1_iter2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state13;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state12;
            end
        end
        ap_ST_iter2_fsm_state13 : begin
            if (((1'b1 == ap_CS_iter2_fsm_state13) & (1'b0 == ap_block_state13_pp0_stage2_iter2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state14;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state13;
            end
        end
        ap_ST_iter2_fsm_state14 : begin
            if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b0 == ap_CS_iter1_fsm_state10))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if (((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter1_fsm_state10)) | (~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd1)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state11;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state14;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state10))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state11;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

assign add_ln36_1_fu_211_p2 = (ap_sig_allocacmp_ct_load + 7'd1);

assign add_ln36_fu_193_p2 = (ap_sig_allocacmp_indvar_flatten_load + 10'd1);

assign add_ln37_fu_292_p2 = (select_ln36_fu_235_p3 + 4'd1);

assign add_ln39_1_fu_314_p2 = ($signed(sext_ln39_fu_310_p1) + $signed(memory_decoder_state));

assign add_ln39_fu_286_p2 = (sub_ln39_fu_269_p2 + zext_ln39_2_fu_282_p1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter0_fsm_state3 = ap_CS_iter0_fsm[32'd2];

assign ap_CS_iter0_fsm_state4 = ap_CS_iter0_fsm[32'd3];

assign ap_CS_iter0_fsm_state5 = ap_CS_iter0_fsm[32'd4];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state10 = ap_CS_iter1_fsm[32'd5];

assign ap_CS_iter1_fsm_state6 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state7 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter1_fsm_state8 = ap_CS_iter1_fsm[32'd3];

assign ap_CS_iter1_fsm_state9 = ap_CS_iter1_fsm[32'd4];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state12 = ap_CS_iter2_fsm[32'd2];

assign ap_CS_iter2_fsm_state13 = ap_CS_iter2_fsm[32'd3];

assign ap_CS_iter2_fsm_state14 = ap_CS_iter2_fsm[32'd4];

always @ (*) begin
    ap_block_state12_pp0_stage1_iter2 = ((icmp_ln36_reg_386_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state13_pp0_stage2_iter2 = ((icmp_ln36_reg_386_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state14_io = ((x_stream_TREADY == 1'b0) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0));
end

always @ (*) begin
    ap_block_state14_pp0_stage3_iter2 = ((x_stream_TREADY == 1'b0) & (icmp_ln36_reg_386_pp0_iter1_reg == 1'd0));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state4_io = ((icmp_ln36_reg_386 == 1'd0) & (m_axi_gmem1_ARREADY == 1'b0));
end

always @ (*) begin
    ap_condition_211 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage4;

assign icmp_ln36_fu_187_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 10'd960) ? 1'b1 : 1'b0);

assign icmp_ln37_fu_205_p2 = ((ap_sig_allocacmp_t_load == 4'd8) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = sext_ln206_fu_329_p1;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd2;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign select_ln36_1_fu_217_p3 = ((icmp_ln37_fu_205_p2[0:0] == 1'b1) ? add_ln36_1_fu_211_p2 : ap_sig_allocacmp_ct_load);

assign select_ln36_fu_235_p3 = ((icmp_ln37_reg_395[0:0] == 1'b1) ? 4'd0 : t_load_reg_390);

assign sext_ln206_fu_329_p1 = $signed(trunc_ln4_reg_410);

assign sext_ln39_fu_310_p1 = $signed(tmp_s_fu_303_p3);

assign shl_ln1_fu_245_p3 = {{trunc_ln39_fu_241_p1}, {8'd0}};

assign shl_ln39_1_fu_257_p3 = {{trunc_ln39_fu_241_p1}, {4'd0}};

assign shl_ln39_2_fu_275_p3 = {{select_ln36_1_reg_400}, {1'd0}};

assign sub_ln39_fu_269_p2 = (zext_ln39_fu_253_p1 - zext_ln39_1_fu_265_p1);

assign tmp_s_fu_303_p3 = {{add_ln39_reg_405}, {4'd0}};

assign trunc_ln210_fu_339_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln39_1_fu_343_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln39_fu_241_p1 = select_ln36_fu_235_p3[2:0];

assign x_stream_TDATA = {{{{{{{{tmp_6_reg_456}, {tmp_5_reg_451}}, {tmp_4_reg_446}}, {trunc_ln39_1_reg_441}}, {tmp_2_reg_436}}, {tmp_1_reg_431}}, {tmp_reg_426}}, {trunc_ln210_reg_421}};

assign zext_ln39_1_fu_265_p1 = shl_ln39_1_fu_257_p3;

assign zext_ln39_2_fu_282_p1 = shl_ln39_2_fu_275_p3;

assign zext_ln39_fu_253_p1 = shl_ln1_fu_245_p3;

always @ (posedge ap_clk) begin
    add_ln39_reg_405[0] <= 1'b0;
end

endmodule //STATE_AXI_STATE_AXI_Pipeline_VITIS_LOOP_36_1_VITIS_LOOP_37_2
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_vit_delta_order_indexed (
        ap_clk,
        ap_rst,
        ap_start,
        start_full_n,
        ap_done,
        ap_continue,
        ap_idle,
        ap_ready,
        start_out,
        start_write,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state,
        x_stream_TDATA,
        x_stream_TVALID,
        x_stream_TREADY,
        tile_idx_stream_din,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_full_n,
        tile_idx_stream_write,
        memory_state_c_din,
        memory_state_c_num_data_valid,
        memory_state_c_fifo_cap,
        memory_state_c_full_n,
        memory_state_c_write
);

parameter    ap_ST_fsm_state1 = 3'd1;
parameter    ap_ST_fsm_state2 = 3'd2;
parameter    ap_ST_fsm_state3 = 3'd4;

input   ap_clk;
input   ap_rst;
input   ap_start;
input   start_full_n;
output   ap_done;
input   ap_continue;
output   ap_idle;
output   ap_ready;
output   start_out;
output   start_write;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;
input   x_stream_TREADY;
output  [31:0] tile_idx_stream_din;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_full_n;
output   tile_idx_stream_write;
output  [63:0] memory_state_c_din;
input  [2:0] memory_state_c_num_data_valid;
input  [2:0] memory_state_c_fifo_cap;
input   memory_state_c_full_n;
output   memory_state_c_write;

reg ap_done;
reg ap_idle;
reg start_write;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg tile_idx_stream_write;
reg memory_state_c_write;

reg    real_start;
reg    start_once_reg;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [2:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    internal_ap_ready;
reg    memory_state_c_blk_n;
reg    ap_block_state1;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_done;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_idle;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_ready;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWVALID;
wire   [63:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWADDR;
wire   [0:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWID;
wire   [31:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWLEN;
wire   [2:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWBURST;
wire   [1:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWPROT;
wire   [3:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWQOS;
wire   [3:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWREGION;
wire   [0:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWUSER;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WVALID;
wire   [127:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WDATA;
wire   [15:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WSTRB;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WLAST;
wire   [0:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WID;
wire   [0:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WUSER;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARVALID;
wire   [63:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARADDR;
wire   [0:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARID;
wire   [31:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARLEN;
wire   [2:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARBURST;
wire   [1:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARPROT;
wire   [3:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARQOS;
wire   [3:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARREGION;
wire   [0:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARUSER;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_RREADY;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_BREADY;
wire   [31:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_tile_idx_stream_din;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_tile_idx_stream_write;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TREADY;
wire   [199:0] grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TDATA;
wire    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TVALID;
reg    grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start_reg;
wire    ap_CS_fsm_state2;
wire    ap_CS_fsm_state3;
reg   [2:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 start_once_reg = 1'b0;
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 3'd1;
//#0 grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start_reg = 1'b0;
end

STATE_AXI_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start),
    .ap_done(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_done),
    .ap_idle(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_idle),
    .ap_ready(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_ready),
    .m_axi_gmem1_AWVALID(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(m_axi_gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(m_axi_gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(m_axi_gmem1_RDATA),
    .m_axi_gmem1_RLAST(m_axi_gmem1_RLAST),
    .m_axi_gmem1_RID(m_axi_gmem1_RID),
    .m_axi_gmem1_RFIFONUM(m_axi_gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(m_axi_gmem1_RUSER),
    .m_axi_gmem1_RRESP(m_axi_gmem1_RRESP),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .tile_idx_stream_din(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_tile_idx_stream_din),
    .tile_idx_stream_num_data_valid(7'd0),
    .tile_idx_stream_fifo_cap(7'd0),
    .tile_idx_stream_full_n(tile_idx_stream_full_n),
    .tile_idx_stream_write(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_tile_idx_stream_write),
    .x_stream_TREADY(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TREADY),
    .memory_state(memory_state),
    .x_stream_TDATA(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TDATA),
    .x_stream_TVALID(grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TVALID)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_done == 1'b1))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start_reg <= 1'b1;
        end else if ((grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_ready == 1'b1)) begin
            grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        start_once_reg <= 1'b0;
    end else begin
        if (((real_start == 1'b1) & (internal_ap_ready == 1'b0))) begin
            start_once_reg <= 1'b1;
        end else if ((internal_ap_ready == 1'b1)) begin
            start_once_reg <= 1'b0;
        end
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_done == 1'b1))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
    end
end

always @ (*) begin
    if (((real_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_done == 1'b1))) begin
        internal_ap_ready = 1'b1;
    end else begin
        internal_ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_ARVALID = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARVALID;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_RREADY = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_RREADY;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if ((~((real_start == 1'b0) | (ap_done_reg == 1'b1)) & (1'b1 == ap_CS_fsm_state1))) begin
        memory_state_c_blk_n = memory_state_c_full_n;
    end else begin
        memory_state_c_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
        memory_state_c_write = 1'b1;
    end else begin
        memory_state_c_write = 1'b0;
    end
end

always @ (*) begin
    if (((start_once_reg == 1'b0) & (start_full_n == 1'b0))) begin
        real_start = 1'b0;
    end else begin
        real_start = ap_start;
    end
end

always @ (*) begin
    if (((real_start == 1'b1) & (start_once_reg == 1'b0))) begin
        start_write = 1'b1;
    end else begin
        start_write = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        tile_idx_stream_write = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_tile_idx_stream_write;
    end else begin
        tile_idx_stream_write = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (1'b0 == ap_block_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

always @ (*) begin
    ap_block_state1 = ((real_start == 1'b0) | (ap_done_reg == 1'b1) | (memory_state_c_full_n == 1'b0));
end

assign ap_ready = internal_ap_ready;

assign grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_ap_start_reg;

assign grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TREADY = (x_stream_TREADY & ap_CS_fsm_state3);

assign m_axi_gmem1_ARADDR = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARADDR;

assign m_axi_gmem1_ARBURST = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARBURST;

assign m_axi_gmem1_ARCACHE = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARCACHE;

assign m_axi_gmem1_ARID = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARID;

assign m_axi_gmem1_ARLEN = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARLEN;

assign m_axi_gmem1_ARLOCK = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARLOCK;

assign m_axi_gmem1_ARPROT = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARPROT;

assign m_axi_gmem1_ARQOS = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARQOS;

assign m_axi_gmem1_ARREGION = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARREGION;

assign m_axi_gmem1_ARSIZE = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARSIZE;

assign m_axi_gmem1_ARUSER = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_m_axi_gmem1_ARUSER;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign memory_state_c_din = memory_state;

assign start_out = real_start;

assign tile_idx_stream_din = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_tile_idx_stream_din;

assign x_stream_TDATA = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TDATA;

assign x_stream_TVALID = grp_replay_vit_delta_order_indexed_Pipeline_VITIS_LOOP_84_1_VITIS_LOOP_85_2_VITIS_LO_fu_64_x_stream_TVALID;

endmodule //STATE_AXI_replay_vit_delta_order_indexed
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
//RAW latency 1 

module STATE_AXI_fifo_w64_d2_S_x
#(parameter
    MEM_STYLE    = "shiftReg",
    DATA_WIDTH   = 64,
    ADDR_WIDTH   = 2,
    DEPTH        = 2)
(
    // system signal
    input  wire                  clk,
    input  wire                  reset,

    // write
    output wire                  if_full_n,
    input  wire                  if_write_ce,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,
    
    // read 
    output wire [ADDR_WIDTH:0]   if_num_data_valid, // for FRP
    output wire [ADDR_WIDTH:0]   if_fifo_cap,       // for FRP

    output wire                  if_empty_n,
    input  wire                  if_read_ce,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout
);
//------------------------Parameter----------------------

//------------------------Local signal-------------------
    wire [ADDR_WIDTH-1:0] addr;
    wire                  push;
    wire                  pop;
    reg signed [ADDR_WIDTH:0] mOutPtr;
    reg                   empty_n = 1'b0;
    reg                   full_n = 1'b1; 
    // has num_data_valid? 
    reg  [ADDR_WIDTH:0]   num_data_valid; //yes 
//------------------------Instantiation------------------
    STATE_AXI_fifo_w64_d2_S_x_ShiftReg 
    #(  .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DEPTH      (DEPTH))
    U_STATE_AXI_fifo_w64_d2_S_x_ShiftReg (
        .clk        (clk),
        .we         (push),
        .addr       (addr),
        .din        (if_din),
        .dout       (if_dout)
    );
//------------------------Task and function--------------

//------------------------Body---------------------------
    // num_data_valid 
    assign if_num_data_valid = num_data_valid;
    assign if_fifo_cap       = DEPTH;

    // almost full/empty 

    // program full/empty 

    assign if_full_n  = full_n; 
    assign if_empty_n = empty_n;

    assign push       = full_n & if_write_ce & if_write;
    assign pop        = empty_n & if_read_ce & if_read;

    assign addr       = mOutPtr[ADDR_WIDTH] == 1'b0 ? mOutPtr[ADDR_WIDTH-1:0] : {ADDR_WIDTH{1'b0}};

    // mOutPtr
    always @(posedge clk) begin
        if (reset)
            mOutPtr <= {ADDR_WIDTH+1{1'b1}};
        else if (push & ~pop)
            mOutPtr <= mOutPtr + 1'b1;
        else if (~push & pop)
            mOutPtr <= mOutPtr - 1'b1;
    end

    // full_n
    always @(posedge clk) begin
        if (reset)
            full_n <= 1'b1;
        else if ((push & ~pop) && (mOutPtr == DEPTH - 2))
            full_n <= 1'b0;
        else if (~push & pop)
            full_n <= 1'b1;
    end

    // empty_n
    always @(posedge clk) begin
        if (reset)
            empty_n <= 1'b0;
        else if (push & ~pop)
            empty_n <= 1'b1;
        else if ((~push & pop) && (mOutPtr == 0))
            empty_n <= 1'b0;
    end

    // almost_full_n 

    // almost_empty_n 

    // prog_full_n 
 
    // prog_empty_n 

    // num_data_valid 
    always @(posedge clk) begin
        if (reset)
            num_data_valid <= {ADDR_WIDTH+1{1'b0}};
        else if ( push & ~pop)
            num_data_valid <= num_data_valid + 1;
        else if (~push & pop)
            num_data_valid <= num_data_valid - 1;
    end // 

endmodule  


module STATE_AXI_fifo_w64_d2_S_x_ShiftReg
#(parameter
    DATA_WIDTH  = 64,
    ADDR_WIDTH  = 2,
    DEPTH       = 2)
(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire [DATA_WIDTH-1:0] din,
    output wire [DATA_WIDTH-1:0] dout
);

    reg [DATA_WIDTH-1:0] SRL_SIG [0:DEPTH-1];
    integer i;

    always @ (posedge clk) begin
        if (we) begin
            for (i=0; i<DEPTH-1; i=i+1)
                SRL_SIG[i+1] <= SRL_SIG[i];
            SRL_SIG[0] <= din;
        end
    end

    assign dout = SRL_SIG[addr];

endmodule// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_token_major (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        memory_state,
        x_stream_TDATA,
        x_stream_TVALID,
        x_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 14'd1;
parameter    ap_ST_fsm_state2 = 14'd2;
parameter    ap_ST_fsm_state3 = 14'd4;
parameter    ap_ST_fsm_state4 = 14'd8;
parameter    ap_ST_fsm_state5 = 14'd16;
parameter    ap_ST_fsm_state6 = 14'd32;
parameter    ap_ST_fsm_state7 = 14'd64;
parameter    ap_ST_fsm_state8 = 14'd128;
parameter    ap_ST_fsm_state9 = 14'd256;
parameter    ap_ST_fsm_state10 = 14'd512;
parameter    ap_ST_fsm_state11 = 14'd1024;
parameter    ap_ST_fsm_state12 = 14'd2048;
parameter    ap_ST_fsm_state13 = 14'd4096;
parameter    ap_ST_fsm_state14 = 14'd8192;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;
input   x_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg x_stream_TVALID;

(* fsm_encoding = "none" *) reg   [13:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    gmem1_blk_n_AR;
reg    gmem1_blk_n_R;
wire    ap_CS_fsm_state11;
wire    ap_CS_fsm_state12;
reg    x_stream_TDATA_blk_n;
wire    ap_CS_fsm_state13;
wire   [3:0] add_ln23_fu_189_p2;
reg   [3:0] add_ln23_reg_248;
wire    ap_CS_fsm_state9;
wire   [6:0] add_ln24_fu_201_p2;
reg   [6:0] add_ln24_reg_256;
wire    ap_CS_fsm_state10;
wire   [24:0] trunc_ln210_fu_211_p1;
reg   [24:0] trunc_ln210_reg_261;
reg   [24:0] tmp_reg_266;
reg   [24:0] tmp_s_reg_271;
reg   [24:0] tmp_4_reg_276;
wire   [24:0] trunc_ln26_fu_215_p1;
reg   [24:0] trunc_ln26_reg_281;
reg   [24:0] tmp_6_reg_286;
reg   [24:0] tmp_7_reg_291;
reg   [24:0] tmp_8_reg_296;
reg   [6:0] ct_reg_113;
wire   [0:0] icmp_ln23_fu_183_p2;
wire  signed [63:0] sext_ln23_fu_164_p1;
reg   [3:0] t_fu_84;
wire   [0:0] icmp_ln24_fu_195_p2;
wire   [59:0] trunc_ln_fu_154_p4;
wire    ap_CS_fsm_state14;
reg   [13:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
wire    ap_ST_fsm_state8_blk;
wire    ap_ST_fsm_state9_blk;
wire    ap_ST_fsm_state10_blk;
reg    ap_ST_fsm_state11_blk;
reg    ap_ST_fsm_state12_blk;
reg    ap_ST_fsm_state13_blk;
wire    ap_ST_fsm_state14_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 14'd1;
//#0 t_fu_84 = 4'd0;
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state9) & (icmp_ln23_fu_183_p2 == 1'd0))) begin
        ct_reg_113 <= 7'd0;
    end else if (((x_stream_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state13))) begin
        ct_reg_113 <= add_ln24_reg_256;
    end
end

always @ (posedge ap_clk) begin
    if ((~((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        t_fu_84 <= 4'd0;
    end else if (((1'b1 == ap_CS_fsm_state10) & (icmp_ln24_fu_195_p2 == 1'd1))) begin
        t_fu_84 <= add_ln23_reg_248;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        add_ln23_reg_248 <= add_ln23_fu_189_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state10)) begin
        add_ln24_reg_256 <= add_ln24_fu_201_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state11)) begin
        tmp_4_reg_276 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_reg_266 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_s_reg_271 <= {{m_axi_gmem1_RDATA[88:64]}};
        trunc_ln210_reg_261 <= trunc_ln210_fu_211_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state12)) begin
        tmp_6_reg_286 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_7_reg_291 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_8_reg_296 <= {{m_axi_gmem1_RDATA[120:96]}};
        trunc_ln26_reg_281 <= trunc_ln26_fu_215_p1;
    end
end

assign ap_ST_fsm_state10_blk = 1'b0;

always @ (*) begin
    if ((m_axi_gmem1_RVALID == 1'b0)) begin
        ap_ST_fsm_state11_blk = 1'b1;
    end else begin
        ap_ST_fsm_state11_blk = 1'b0;
    end
end

always @ (*) begin
    if ((m_axi_gmem1_RVALID == 1'b0)) begin
        ap_ST_fsm_state12_blk = 1'b1;
    end else begin
        ap_ST_fsm_state12_blk = 1'b0;
    end
end

always @ (*) begin
    if ((x_stream_TREADY == 1'b0)) begin
        ap_ST_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_fsm_state13_blk = 1'b0;
    end
end

assign ap_ST_fsm_state14_blk = 1'b0;

always @ (*) begin
    if (((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

assign ap_ST_fsm_state7_blk = 1'b0;

assign ap_ST_fsm_state8_blk = 1'b0;

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state14) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state14)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        gmem1_blk_n_AR = m_axi_gmem1_ARREADY;
    end else begin
        gmem1_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state12) | (1'b1 == ap_CS_fsm_state11))) begin
        gmem1_blk_n_R = m_axi_gmem1_RVALID;
    end else begin
        gmem1_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if ((~((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        m_axi_gmem1_ARVALID = 1'b1;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if ((((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state12)) | ((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state11)))) begin
        m_axi_gmem1_RREADY = 1'b1;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state13)) begin
        x_stream_TDATA_blk_n = x_stream_TREADY;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((x_stream_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state13))) begin
        x_stream_TVALID = 1'b1;
    end else begin
        x_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((m_axi_gmem1_ARREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            if (((1'b1 == ap_CS_fsm_state9) & (icmp_ln23_fu_183_p2 == 1'd0))) begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state14;
            end
        end
        ap_ST_fsm_state10 : begin
            if (((1'b1 == ap_CS_fsm_state10) & (icmp_ln24_fu_195_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end
        end
        ap_ST_fsm_state11 : begin
            if (((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state11))) begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end
        end
        ap_ST_fsm_state12 : begin
            if (((m_axi_gmem1_RVALID == 1'b1) & (1'b1 == ap_CS_fsm_state12))) begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end
        end
        ap_ST_fsm_state13 : begin
            if (((x_stream_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state13))) begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state13;
            end
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln23_fu_189_p2 = (t_fu_84 + 4'd1);

assign add_ln24_fu_201_p2 = (ct_reg_113 + 7'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd10];

assign ap_CS_fsm_state12 = ap_CS_fsm[32'd11];

assign ap_CS_fsm_state13 = ap_CS_fsm[32'd12];

assign ap_CS_fsm_state14 = ap_CS_fsm[32'd13];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign icmp_ln23_fu_183_p2 = ((t_fu_84 == 4'd8) ? 1'b1 : 1'b0);

assign icmp_ln24_fu_195_p2 = ((ct_reg_113 == 7'd120) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = sext_ln23_fu_164_p1;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd1920;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign sext_ln23_fu_164_p1 = $signed(trunc_ln_fu_154_p4);

assign trunc_ln210_fu_211_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln26_fu_215_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln_fu_154_p4 = {{memory_state[63:4]}};

assign x_stream_TDATA = {{{{{{{{tmp_8_reg_296}, {tmp_7_reg_291}}, {tmp_6_reg_286}}, {trunc_ln26_reg_281}}, {tmp_4_reg_276}}, {tmp_s_reg_271}}, {tmp_reg_266}}, {trunc_ln210_reg_261}};

endmodule //STATE_AXI_replay_token_major
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        tile_idx_stream_din,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_full_n,
        tile_idx_stream_write,
        x_stream_TREADY,
        memory_state,
        x_stream_TDATA,
        x_stream_TVALID
);

parameter    ap_ST_iter0_fsm_state1 = 5'd1;
parameter    ap_ST_iter0_fsm_state2 = 5'd2;
parameter    ap_ST_iter0_fsm_state3 = 5'd4;
parameter    ap_ST_iter0_fsm_state4 = 5'd8;
parameter    ap_ST_iter0_fsm_state5 = 5'd16;
parameter    ap_ST_iter1_fsm_state6 = 6'd2;
parameter    ap_ST_iter1_fsm_state7 = 6'd4;
parameter    ap_ST_iter1_fsm_state8 = 6'd8;
parameter    ap_ST_iter1_fsm_state9 = 6'd16;
parameter    ap_ST_iter1_fsm_state10 = 6'd32;
parameter    ap_ST_iter2_fsm_state11 = 5'd2;
parameter    ap_ST_iter2_fsm_state12 = 5'd4;
parameter    ap_ST_iter2_fsm_state13 = 5'd8;
parameter    ap_ST_iter2_fsm_state14 = 5'd16;
parameter    ap_ST_iter1_fsm_state0 = 6'd1;
parameter    ap_ST_iter2_fsm_state0 = 5'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
output  [31:0] tile_idx_stream_din;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_full_n;
output   tile_idx_stream_write;
input   x_stream_TREADY;
input  [63:0] memory_state;
output  [199:0] x_stream_TDATA;
output   x_stream_TVALID;

reg ap_idle;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg tile_idx_stream_write;
reg x_stream_TVALID;

reg   [4:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [5:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [4:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
wire    ap_CS_iter0_fsm_state5;
wire    ap_CS_iter1_fsm_state6;
wire    ap_CS_iter1_fsm_state7;
wire    ap_CS_iter1_fsm_state8;
wire    ap_CS_iter1_fsm_state9;
wire    ap_CS_iter1_fsm_state10;
reg   [0:0] icmp_ln67_reg_444;
reg   [0:0] icmp_ln67_reg_444_pp0_iter1_reg;
reg    ap_block_state12_pp0_stage1_iter2;
wire    ap_CS_iter2_fsm_state12;
reg    ap_block_state13_pp0_stage2_iter2;
wire    ap_CS_iter2_fsm_state13;
reg    ap_block_state14_pp0_stage3_iter2;
reg    ap_block_state14_io;
wire    ap_CS_iter2_fsm_state14;
reg    ap_condition_exit_pp0_iter0_stage4;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    tile_idx_stream_blk_n;
reg    gmem1_blk_n_AR;
wire    ap_CS_iter0_fsm_state4;
reg    gmem1_blk_n_R;
reg    x_stream_TDATA_blk_n;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln67_fu_204_p2;
reg   [0:0] icmp_ln67_reg_444_pp0_iter0_reg;
reg   [3:0] t_load_reg_448;
wire   [0:0] icmp_ln68_fu_222_p2;
reg   [0:0] icmp_ln68_reg_453;
wire   [6:0] select_ln67_1_fu_234_p3;
reg   [6:0] select_ln67_1_reg_458;
reg   [6:0] select_ln67_1_reg_458_pp0_iter0_reg;
reg   [6:0] select_ln67_1_reg_458_pp0_iter1_reg;
wire   [2:0] trunc_ln70_fu_258_p1;
reg   [2:0] trunc_ln70_reg_464;
wire    ap_CS_iter0_fsm_state2;
reg   [2:0] trunc_ln70_reg_464_pp0_iter0_reg;
reg   [2:0] trunc_ln70_reg_464_pp0_iter1_reg;
wire   [11:0] add_ln70_fu_303_p2;
reg   [11:0] add_ln70_reg_470;
reg   [59:0] trunc_ln_reg_475;
wire    ap_CS_iter0_fsm_state3;
reg    ap_block_state4_io;
wire   [24:0] trunc_ln210_fu_356_p1;
reg   [24:0] trunc_ln210_reg_486;
reg   [24:0] tmp_reg_491;
reg   [24:0] tmp_s_reg_496;
reg   [24:0] tmp_9_reg_501;
wire   [10:0] add_ln71_fu_391_p2;
reg   [10:0] add_ln71_reg_506;
wire   [24:0] trunc_ln72_fu_397_p1;
reg   [24:0] trunc_ln72_reg_511;
reg   [24:0] tmp_2_reg_516;
reg   [24:0] tmp_3_reg_521;
reg   [24:0] tmp_4_reg_526;
wire  signed [63:0] sext_ln206_fu_346_p1;
reg   [3:0] t_fu_112;
wire   [3:0] add_ln68_fu_309_p2;
wire    ap_loop_init;
reg   [3:0] ap_sig_allocacmp_t_load;
reg   [6:0] ct_fu_116;
reg   [6:0] ap_sig_allocacmp_ct_load;
reg   [9:0] indvar_flatten_fu_120;
wire   [9:0] add_ln67_fu_210_p2;
reg   [9:0] ap_sig_allocacmp_indvar_flatten_load;
wire   [6:0] add_ln67_1_fu_228_p2;
wire   [3:0] select_ln67_fu_252_p3;
wire   [10:0] shl_ln_fu_262_p3;
wire   [6:0] shl_ln70_1_fu_274_p3;
wire   [11:0] zext_ln70_1_fu_270_p1;
wire   [11:0] zext_ln70_2_fu_282_p1;
wire   [7:0] shl_ln70_2_fu_292_p3;
wire   [11:0] sub_ln70_fu_286_p2;
wire   [11:0] zext_ln70_3_fu_299_p1;
wire   [15:0] tmp_10_fu_320_p3;
wire  signed [63:0] sext_ln70_fu_327_p1;
wire   [63:0] add_ln70_1_fu_331_p2;
wire   [9:0] shl_ln70_4_fu_363_p3;
wire   [5:0] shl_ln70_5_fu_374_p3;
wire   [10:0] zext_ln70_4_fu_370_p1;
wire   [10:0] zext_ln70_5_fu_381_p1;
wire   [10:0] sub_ln70_1_fu_385_p2;
wire   [10:0] zext_ln70_fu_360_p1;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg   [4:0] ap_NS_iter0_fsm;
reg   [5:0] ap_NS_iter1_fsm;
reg   [4:0] ap_NS_iter2_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter0_fsm_state3_blk;
reg    ap_ST_iter0_fsm_state4_blk;
wire    ap_ST_iter0_fsm_state5_blk;
wire    ap_ST_iter1_fsm_state6_blk;
wire    ap_ST_iter1_fsm_state7_blk;
wire    ap_ST_iter1_fsm_state8_blk;
wire    ap_ST_iter1_fsm_state9_blk;
wire    ap_ST_iter1_fsm_state10_blk;
wire    ap_ST_iter2_fsm_state11_blk;
reg    ap_ST_iter2_fsm_state12_blk;
reg    ap_ST_iter2_fsm_state13_blk;
reg    ap_ST_iter2_fsm_state14_blk;
wire    ap_start_int;
reg    ap_condition_220;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 5'd1;
//#0 ap_CS_iter1_fsm = 6'd1;
//#0 ap_CS_iter2_fsm = 5'd1;
//#0 t_fu_112 = 4'd0;
//#0 ct_fu_116 = 7'd0;
//#0 indvar_flatten_fu_120 = 10'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage4),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state14))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b0) & (1'b1 == ap_CS_iter2_fsm_state14))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= 1'b0;
    end else if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_220)) begin
        if ((icmp_ln67_fu_204_p2 == 1'd0)) begin
            ct_fu_116 <= select_ln67_1_fu_234_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            ct_fu_116 <= 7'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_220)) begin
        if ((icmp_ln67_fu_204_p2 == 1'd0)) begin
            indvar_flatten_fu_120 <= add_ln67_fu_210_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten_fu_120 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        t_fu_112 <= 4'd0;
    end else if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2) & (icmp_ln67_reg_444 == 1'd0))) begin
        t_fu_112 <= add_ln68_fu_309_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        add_ln70_reg_470[11 : 1] <= add_ln70_fu_303_p2[11 : 1];
        trunc_ln70_reg_464 <= trunc_ln70_fu_258_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter2_fsm_state13) & (1'b0 == ap_block_state13_pp0_stage2_iter2))) begin
        add_ln71_reg_506 <= add_ln71_fu_391_p2;
        tmp_2_reg_516 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_3_reg_521 <= {{m_axi_gmem1_RDATA[88:64]}};
        tmp_4_reg_526 <= {{m_axi_gmem1_RDATA[120:96]}};
        trunc_ln72_reg_511 <= trunc_ln72_fu_397_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln67_reg_444_pp0_iter0_reg <= icmp_ln67_reg_444;
        select_ln67_1_reg_458_pp0_iter0_reg <= select_ln67_1_reg_458;
        trunc_ln70_reg_464_pp0_iter0_reg <= trunc_ln70_reg_464;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln67_reg_444 <= icmp_ln67_fu_204_p2;
        icmp_ln68_reg_453 <= icmp_ln68_fu_222_p2;
        select_ln67_1_reg_458 <= select_ln67_1_fu_234_p3;
        t_load_reg_448 <= ap_sig_allocacmp_t_load;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state10))) begin
        icmp_ln67_reg_444_pp0_iter1_reg <= icmp_ln67_reg_444_pp0_iter0_reg;
        select_ln67_1_reg_458_pp0_iter1_reg <= select_ln67_1_reg_458_pp0_iter0_reg;
        trunc_ln70_reg_464_pp0_iter1_reg <= trunc_ln70_reg_464_pp0_iter0_reg;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage1_iter2))) begin
        tmp_9_reg_501 <= {{m_axi_gmem1_RDATA[120:96]}};
        tmp_reg_491 <= {{m_axi_gmem1_RDATA[56:32]}};
        tmp_s_reg_496 <= {{m_axi_gmem1_RDATA[88:64]}};
        trunc_ln210_reg_486 <= trunc_ln210_fu_356_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state3))) begin
        trunc_ln_reg_475 <= {{add_ln70_1_fu_331_p2[63:4]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter0_fsm_state2_blk = 1'b0;

assign ap_ST_iter0_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_io)) begin
        ap_ST_iter0_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state4_blk = 1'b0;
    end
end

assign ap_ST_iter0_fsm_state5_blk = 1'b0;

assign ap_ST_iter1_fsm_state10_blk = 1'b0;

assign ap_ST_iter1_fsm_state6_blk = 1'b0;

assign ap_ST_iter1_fsm_state7_blk = 1'b0;

assign ap_ST_iter1_fsm_state8_blk = 1'b0;

assign ap_ST_iter1_fsm_state9_blk = 1'b0;

assign ap_ST_iter2_fsm_state11_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state12_pp0_stage1_iter2)) begin
        ap_ST_iter2_fsm_state12_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state12_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state13_pp0_stage2_iter2)) begin
        ap_ST_iter2_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state13_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) begin
        ap_ST_iter2_fsm_state14_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state14_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5) & (icmp_ln67_reg_444 == 1'd1))) begin
        ap_condition_exit_pp0_iter0_stage4 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage4 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (ap_loop_exit_ready_pp0_iter2_reg == 1'b1) & (1'b1 == ap_CS_iter2_fsm_state14))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_ct_load = 7'd0;
    end else begin
        ap_sig_allocacmp_ct_load = ct_fu_116;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten_load = 10'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten_load = indvar_flatten_fu_120;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_t_load = 4'd0;
    end else begin
        ap_sig_allocacmp_t_load = t_fu_112;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state4) & (icmp_ln67_reg_444 == 1'd0))) begin
        gmem1_blk_n_AR = m_axi_gmem1_ARREADY;
    end else begin
        gmem1_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter2_fsm_state13) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0)))) begin
        gmem1_blk_n_R = m_axi_gmem1_RVALID;
    end else begin
        gmem1_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state4) & (icmp_ln67_reg_444 == 1'd0))) begin
        m_axi_gmem1_ARVALID = 1'b1;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter2_fsm_state13) & (1'b0 == ap_block_state13_pp0_stage2_iter2) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage1_iter2) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0)))) begin
        m_axi_gmem1_RREADY = 1'b1;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0))) begin
        tile_idx_stream_blk_n = tile_idx_stream_full_n;
    end else begin
        tile_idx_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0))) begin
        tile_idx_stream_write = 1'b1;
    end else begin
        tile_idx_stream_write = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0))) begin
        x_stream_TDATA_blk_n = x_stream_TREADY;
    end else begin
        x_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0))) begin
        x_stream_TVALID = 1'b1;
    end else begin
        x_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state3;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        ap_ST_iter0_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state3))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state4;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state3;
            end
        end
        ap_ST_iter0_fsm_state4 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state4))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state5;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state4;
            end
        end
        ap_ST_iter0_fsm_state5 : begin
            if (~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state5;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state6 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state6))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state7;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end
        end
        ap_ST_iter1_fsm_state7 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state7))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state8;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state7;
            end
        end
        ap_ST_iter1_fsm_state8 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state8))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state9;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state8;
            end
        end
        ap_ST_iter1_fsm_state9 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state9))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state10;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state9;
            end
        end
        ap_ST_iter1_fsm_state10 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end else if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b0 == ap_CS_iter0_fsm_state5))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state10;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state5))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state6;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state11 : begin
            ap_NS_iter2_fsm = ap_ST_iter2_fsm_state12;
        end
        ap_ST_iter2_fsm_state12 : begin
            if (((1'b1 == ap_CS_iter2_fsm_state12) & (1'b0 == ap_block_state12_pp0_stage1_iter2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state13;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state12;
            end
        end
        ap_ST_iter2_fsm_state13 : begin
            if (((1'b1 == ap_CS_iter2_fsm_state13) & (1'b0 == ap_block_state13_pp0_stage2_iter2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state14;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state13;
            end
        end
        ap_ST_iter2_fsm_state14 : begin
            if ((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b0 == ap_CS_iter1_fsm_state10))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else if (((~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter1_fsm_state10)) | (~((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2)) & (1'b1 == ap_CS_iter2_fsm_state14) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd1)))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state11;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state14;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter1_fsm_state10))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state11;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

assign add_ln67_1_fu_228_p2 = (ap_sig_allocacmp_ct_load + 7'd1);

assign add_ln67_fu_210_p2 = (ap_sig_allocacmp_indvar_flatten_load + 10'd1);

assign add_ln68_fu_309_p2 = (select_ln67_fu_252_p3 + 4'd1);

assign add_ln70_1_fu_331_p2 = ($signed(sext_ln70_fu_327_p1) + $signed(memory_state));

assign add_ln70_fu_303_p2 = (sub_ln70_fu_286_p2 + zext_ln70_3_fu_299_p1);

assign add_ln71_fu_391_p2 = (sub_ln70_1_fu_385_p2 + zext_ln70_fu_360_p1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter0_fsm_state3 = ap_CS_iter0_fsm[32'd2];

assign ap_CS_iter0_fsm_state4 = ap_CS_iter0_fsm[32'd3];

assign ap_CS_iter0_fsm_state5 = ap_CS_iter0_fsm[32'd4];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state10 = ap_CS_iter1_fsm[32'd5];

assign ap_CS_iter1_fsm_state6 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state7 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter1_fsm_state8 = ap_CS_iter1_fsm[32'd3];

assign ap_CS_iter1_fsm_state9 = ap_CS_iter1_fsm[32'd4];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state12 = ap_CS_iter2_fsm[32'd2];

assign ap_CS_iter2_fsm_state13 = ap_CS_iter2_fsm[32'd3];

assign ap_CS_iter2_fsm_state14 = ap_CS_iter2_fsm[32'd4];

always @ (*) begin
    ap_block_state12_pp0_stage1_iter2 = ((icmp_ln67_reg_444_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state13_pp0_stage2_iter2 = ((icmp_ln67_reg_444_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state14_io = ((x_stream_TREADY == 1'b0) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0));
end

always @ (*) begin
    ap_block_state14_pp0_stage3_iter2 = (((x_stream_TREADY == 1'b0) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0)) | ((tile_idx_stream_full_n == 1'b0) & (icmp_ln67_reg_444_pp0_iter1_reg == 1'd0)));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state4_io = ((icmp_ln67_reg_444 == 1'd0) & (m_axi_gmem1_ARREADY == 1'b0));
end

always @ (*) begin
    ap_condition_220 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter2_fsm_state14) & ((1'b1 == ap_block_state14_io) | (1'b1 == ap_block_state14_pp0_stage3_iter2))) | ((1'b1 == ap_CS_iter2_fsm_state13) & (1'b1 == ap_block_state13_pp0_stage2_iter2)) | ((1'b1 == ap_CS_iter2_fsm_state12) & (1'b1 == ap_block_state12_pp0_stage1_iter2))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage4;

assign icmp_ln67_fu_204_p2 = ((ap_sig_allocacmp_indvar_flatten_load == 10'd960) ? 1'b1 : 1'b0);

assign icmp_ln68_fu_222_p2 = ((ap_sig_allocacmp_t_load == 4'd8) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = sext_ln206_fu_346_p1;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd2;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_AWADDR = 64'd0;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd0;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_AWVALID = 1'b0;

assign m_axi_gmem1_BREADY = 1'b0;

assign m_axi_gmem1_WDATA = 128'd0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd0;

assign m_axi_gmem1_WUSER = 1'd0;

assign m_axi_gmem1_WVALID = 1'b0;

assign select_ln67_1_fu_234_p3 = ((icmp_ln68_fu_222_p2[0:0] == 1'b1) ? add_ln67_1_fu_228_p2 : ap_sig_allocacmp_ct_load);

assign select_ln67_fu_252_p3 = ((icmp_ln68_reg_453[0:0] == 1'b1) ? 4'd0 : t_load_reg_448);

assign sext_ln206_fu_346_p1 = $signed(trunc_ln_reg_475);

assign sext_ln70_fu_327_p1 = $signed(tmp_10_fu_320_p3);

assign shl_ln70_1_fu_274_p3 = {{trunc_ln70_fu_258_p1}, {4'd0}};

assign shl_ln70_2_fu_292_p3 = {{select_ln67_1_reg_458}, {1'd0}};

assign shl_ln70_4_fu_363_p3 = {{trunc_ln70_reg_464_pp0_iter1_reg}, {7'd0}};

assign shl_ln70_5_fu_374_p3 = {{trunc_ln70_reg_464_pp0_iter1_reg}, {3'd0}};

assign shl_ln_fu_262_p3 = {{trunc_ln70_fu_258_p1}, {8'd0}};

assign sub_ln70_1_fu_385_p2 = (zext_ln70_4_fu_370_p1 - zext_ln70_5_fu_381_p1);

assign sub_ln70_fu_286_p2 = (zext_ln70_1_fu_270_p1 - zext_ln70_2_fu_282_p1);

assign tile_idx_stream_din = $signed(add_ln71_reg_506);

assign tmp_10_fu_320_p3 = {{add_ln70_reg_470}, {4'd0}};

assign trunc_ln210_fu_356_p1 = m_axi_gmem1_RDATA[24:0];

assign trunc_ln70_fu_258_p1 = select_ln67_fu_252_p3[2:0];

assign trunc_ln72_fu_397_p1 = m_axi_gmem1_RDATA[24:0];

assign x_stream_TDATA = {{{{{{{{tmp_4_reg_526}, {tmp_3_reg_521}}, {tmp_2_reg_516}}, {trunc_ln72_reg_511}}, {tmp_9_reg_501}}, {tmp_s_reg_496}}, {tmp_reg_491}}, {trunc_ln210_reg_486}};

assign zext_ln70_1_fu_270_p1 = shl_ln_fu_262_p3;

assign zext_ln70_2_fu_282_p1 = shl_ln70_1_fu_274_p3;

assign zext_ln70_3_fu_299_p1 = shl_ln70_2_fu_292_p3;

assign zext_ln70_4_fu_370_p1 = shl_ln70_4_fu_363_p3;

assign zext_ln70_5_fu_381_p1 = shl_ln70_5_fu_374_p3;

assign zext_ln70_fu_360_p1 = select_ln67_1_reg_458_pp0_iter1_reg;

always @ (posedge ap_clk) begin
    add_ln70_reg_470[0] <= 1'b0;
end

endmodule //STATE_AXI_replay_llm_delta_order_indexed_Pipeline_VITIS_LOOP_67_1_VITIS_LOOP_68_2
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module STATE_AXI_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        tile_idx_stream_dout,
        tile_idx_stream_num_data_valid,
        tile_idx_stream_fifo_cap,
        tile_idx_stream_empty_n,
        tile_idx_stream_read,
        y_stream_TVALID,
        m_axi_gmem1_AWVALID,
        m_axi_gmem1_AWREADY,
        m_axi_gmem1_AWADDR,
        m_axi_gmem1_AWID,
        m_axi_gmem1_AWLEN,
        m_axi_gmem1_AWSIZE,
        m_axi_gmem1_AWBURST,
        m_axi_gmem1_AWLOCK,
        m_axi_gmem1_AWCACHE,
        m_axi_gmem1_AWPROT,
        m_axi_gmem1_AWQOS,
        m_axi_gmem1_AWREGION,
        m_axi_gmem1_AWUSER,
        m_axi_gmem1_WVALID,
        m_axi_gmem1_WREADY,
        m_axi_gmem1_WDATA,
        m_axi_gmem1_WSTRB,
        m_axi_gmem1_WLAST,
        m_axi_gmem1_WID,
        m_axi_gmem1_WUSER,
        m_axi_gmem1_ARVALID,
        m_axi_gmem1_ARREADY,
        m_axi_gmem1_ARADDR,
        m_axi_gmem1_ARID,
        m_axi_gmem1_ARLEN,
        m_axi_gmem1_ARSIZE,
        m_axi_gmem1_ARBURST,
        m_axi_gmem1_ARLOCK,
        m_axi_gmem1_ARCACHE,
        m_axi_gmem1_ARPROT,
        m_axi_gmem1_ARQOS,
        m_axi_gmem1_ARREGION,
        m_axi_gmem1_ARUSER,
        m_axi_gmem1_RVALID,
        m_axi_gmem1_RREADY,
        m_axi_gmem1_RDATA,
        m_axi_gmem1_RLAST,
        m_axi_gmem1_RID,
        m_axi_gmem1_RFIFONUM,
        m_axi_gmem1_RUSER,
        m_axi_gmem1_RRESP,
        m_axi_gmem1_BVALID,
        m_axi_gmem1_BREADY,
        m_axi_gmem1_BRESP,
        m_axi_gmem1_BID,
        m_axi_gmem1_BUSER,
        y_stream_TDATA,
        y_stream_TREADY,
        memory_state_1
);

parameter    ap_ST_iter0_fsm_state1 = 2'd1;
parameter    ap_ST_iter0_fsm_state2 = 2'd2;
parameter    ap_ST_iter1_fsm_state3 = 3'd2;
parameter    ap_ST_iter1_fsm_state4 = 3'd4;
parameter    ap_ST_iter2_fsm_state5 = 3'd2;
parameter    ap_ST_iter2_fsm_state6 = 3'd4;
parameter    ap_ST_iter3_fsm_state7 = 3'd2;
parameter    ap_ST_iter3_fsm_state8 = 3'd4;
parameter    ap_ST_iter4_fsm_state9 = 3'd2;
parameter    ap_ST_iter4_fsm_state10 = 3'd4;
parameter    ap_ST_iter5_fsm_state11 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 3'd1;
parameter    ap_ST_iter2_fsm_state0 = 3'd1;
parameter    ap_ST_iter3_fsm_state0 = 3'd1;
parameter    ap_ST_iter4_fsm_state0 = 3'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [31:0] tile_idx_stream_dout;
input  [6:0] tile_idx_stream_num_data_valid;
input  [6:0] tile_idx_stream_fifo_cap;
input   tile_idx_stream_empty_n;
output   tile_idx_stream_read;
input   y_stream_TVALID;
output   m_axi_gmem1_AWVALID;
input   m_axi_gmem1_AWREADY;
output  [63:0] m_axi_gmem1_AWADDR;
output  [0:0] m_axi_gmem1_AWID;
output  [31:0] m_axi_gmem1_AWLEN;
output  [2:0] m_axi_gmem1_AWSIZE;
output  [1:0] m_axi_gmem1_AWBURST;
output  [1:0] m_axi_gmem1_AWLOCK;
output  [3:0] m_axi_gmem1_AWCACHE;
output  [2:0] m_axi_gmem1_AWPROT;
output  [3:0] m_axi_gmem1_AWQOS;
output  [3:0] m_axi_gmem1_AWREGION;
output  [0:0] m_axi_gmem1_AWUSER;
output   m_axi_gmem1_WVALID;
input   m_axi_gmem1_WREADY;
output  [127:0] m_axi_gmem1_WDATA;
output  [15:0] m_axi_gmem1_WSTRB;
output   m_axi_gmem1_WLAST;
output  [0:0] m_axi_gmem1_WID;
output  [0:0] m_axi_gmem1_WUSER;
output   m_axi_gmem1_ARVALID;
input   m_axi_gmem1_ARREADY;
output  [63:0] m_axi_gmem1_ARADDR;
output  [0:0] m_axi_gmem1_ARID;
output  [31:0] m_axi_gmem1_ARLEN;
output  [2:0] m_axi_gmem1_ARSIZE;
output  [1:0] m_axi_gmem1_ARBURST;
output  [1:0] m_axi_gmem1_ARLOCK;
output  [3:0] m_axi_gmem1_ARCACHE;
output  [2:0] m_axi_gmem1_ARPROT;
output  [3:0] m_axi_gmem1_ARQOS;
output  [3:0] m_axi_gmem1_ARREGION;
output  [0:0] m_axi_gmem1_ARUSER;
input   m_axi_gmem1_RVALID;
output   m_axi_gmem1_RREADY;
input  [127:0] m_axi_gmem1_RDATA;
input   m_axi_gmem1_RLAST;
input  [0:0] m_axi_gmem1_RID;
input  [8:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [199:0] y_stream_TDATA;
output   y_stream_TREADY;
input  [63:0] memory_state_1;

reg ap_idle;
reg tile_idx_stream_read;
reg m_axi_gmem1_AWVALID;
reg m_axi_gmem1_WVALID;
reg[127:0] m_axi_gmem1_WDATA;
reg m_axi_gmem1_BREADY;
reg y_stream_TREADY;

reg   [1:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [2:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
reg   [2:0] ap_CS_iter2_fsm;
wire    ap_CS_iter2_fsm_state0;
reg   [2:0] ap_CS_iter3_fsm;
wire    ap_CS_iter3_fsm_state0;
reg   [2:0] ap_CS_iter4_fsm;
wire    ap_CS_iter4_fsm_state0;
reg   [1:0] ap_CS_iter5_fsm;
wire    ap_CS_iter5_fsm_state0;
wire    ap_CS_iter0_fsm_state2;
reg   [0:0] icmp_ln137_reg_355;
reg    ap_block_state2_pp0_stage1_iter0;
wire    ap_CS_iter1_fsm_state3;
reg   [0:0] icmp_ln137_reg_355_pp0_iter0_reg;
reg    ap_block_state4_io;
wire    ap_CS_iter1_fsm_state4;
reg   [0:0] icmp_ln137_reg_355_pp0_iter1_reg;
reg    ap_block_state5_io;
wire    ap_CS_iter2_fsm_state5;
reg    ap_block_state6_io;
wire    ap_CS_iter2_fsm_state6;
wire    ap_CS_iter3_fsm_state7;
wire    ap_CS_iter3_fsm_state8;
wire    ap_CS_iter4_fsm_state9;
wire    ap_CS_iter4_fsm_state10;
reg   [0:0] icmp_ln137_reg_355_pp0_iter4_reg;
reg    ap_block_state11_pp0_stage0_iter5;
wire    ap_CS_iter5_fsm_state11;
reg    ap_condition_exit_pp0_iter0_stage1;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    tile_idx_stream_blk_n;
reg    y_stream_TDATA_blk_n;
reg    gmem1_blk_n_AW;
reg    gmem1_blk_n_W;
reg    gmem1_blk_n_B;
reg    ap_block_state1_pp0_stage0_iter0;
wire   [0:0] icmp_ln137_fu_162_p2;
reg   [0:0] icmp_ln137_reg_355_pp0_iter2_reg;
reg   [0:0] icmp_ln137_reg_355_pp0_iter3_reg;
wire   [24:0] trunc_ln140_fu_179_p1;
reg   [24:0] trunc_ln140_reg_359;
reg   [24:0] trunc_ln140_reg_359_pp0_iter1_reg;
reg   [24:0] trunc_ln140_8_reg_364;
reg   [24:0] trunc_ln140_8_reg_364_pp0_iter1_reg;
reg   [24:0] trunc_ln140_9_reg_369;
reg   [24:0] trunc_ln140_9_reg_369_pp0_iter1_reg;
reg   [24:0] packet_reg_374;
reg   [24:0] packet_reg_374_pp0_iter1_reg;
reg   [24:0] trunc_ln140_s_reg_379;
reg   [24:0] trunc_ln140_s_reg_379_pp0_iter1_reg;
reg   [24:0] trunc_ln140_1_reg_384;
reg   [24:0] trunc_ln140_1_reg_384_pp0_iter1_reg;
reg   [24:0] trunc_ln140_2_reg_389;
reg   [24:0] trunc_ln140_2_reg_389_pp0_iter1_reg;
reg   [24:0] packet_5_reg_394;
reg   [24:0] packet_5_reg_394_pp0_iter1_reg;
wire   [30:0] empty_fu_253_p1;
reg   [30:0] empty_reg_399;
reg   [59:0] trunc_ln3_reg_404;
wire  signed [63:0] sext_ln137_fu_283_p1;
wire  signed [127:0] sext_ln225_10_fu_313_p1;
wire  signed [127:0] sext_ln225_14_fu_338_p1;
reg   [9:0] i_fu_108;
wire   [9:0] i_2_fu_168_p2;
wire    ap_loop_init;
reg   [9:0] ap_sig_allocacmp_i_1;
wire   [35:0] shl_ln_fu_257_p3;
wire  signed [63:0] sext_ln227_fu_264_p1;
wire   [63:0] add_ln227_fu_268_p2;
wire  signed [31:0] sext_ln225_fu_293_p1;
wire  signed [31:0] sext_ln225_8_fu_296_p1;
wire  signed [31:0] sext_ln225_9_fu_299_p1;
wire   [120:0] packet_4_fu_302_p5;
wire  signed [31:0] sext_ln225_11_fu_318_p1;
wire  signed [31:0] sext_ln225_12_fu_321_p1;
wire  signed [31:0] sext_ln225_13_fu_324_p1;
wire   [120:0] packet_6_fu_327_p5;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg    ap_loop_exit_ready_pp0_iter2_reg;
reg    ap_loop_exit_ready_pp0_iter3_reg;
reg    ap_loop_exit_ready_pp0_iter4_reg;
reg    ap_loop_exit_ready_pp0_iter5_reg;
reg   [1:0] ap_NS_iter0_fsm;
reg   [2:0] ap_NS_iter1_fsm;
reg   [2:0] ap_NS_iter2_fsm;
reg   [2:0] ap_NS_iter3_fsm;
reg   [2:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter0_fsm_state2_blk;
wire    ap_ST_iter1_fsm_state3_blk;
reg    ap_ST_iter1_fsm_state4_blk;
reg    ap_ST_iter2_fsm_state5_blk;
reg    ap_ST_iter2_fsm_state6_blk;
wire    ap_ST_iter3_fsm_state7_blk;
wire    ap_ST_iter3_fsm_state8_blk;
wire    ap_ST_iter4_fsm_state9_blk;
wire    ap_ST_iter4_fsm_state10_blk;
reg    ap_ST_iter5_fsm_state11_blk;
wire    ap_start_int;
reg    ap_condition_248;
reg    ap_condition_563;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 2'd1;
//#0 ap_CS_iter1_fsm = 3'd1;
//#0 ap_CS_iter2_fsm = 3'd1;
//#0 ap_CS_iter3_fsm = 3'd1;
//#0 ap_CS_iter4_fsm = 3'd1;
//#0 ap_CS_iter5_fsm = 2'd1;
//#0 i_fu_108 = 10'd0;
//#0 ap_done_reg = 1'b0;
end

STATE_AXI_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage1),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter0_fsm <= ap_ST_iter0_fsm_state1;
    end else begin
        ap_CS_iter0_fsm <= ap_NS_iter0_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter1_fsm <= ap_ST_iter1_fsm_state0;
    end else begin
        ap_CS_iter1_fsm <= ap_NS_iter1_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter2_fsm <= ap_ST_iter2_fsm_state0;
    end else begin
        ap_CS_iter2_fsm <= ap_NS_iter2_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter3_fsm <= ap_ST_iter3_fsm_state0;
    end else begin
        ap_CS_iter3_fsm <= ap_NS_iter3_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter4_fsm <= ap_ST_iter4_fsm_state0;
    end else begin
        ap_CS_iter4_fsm <= ap_NS_iter4_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter5_fsm <= ap_ST_iter5_fsm_state0;
    end else begin
        ap_CS_iter5_fsm <= ap_NS_iter5_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter4_reg == 1'b0) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= 1'b0;
    end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_248)) begin
        if ((icmp_ln137_fu_162_p2 == 1'd0)) begin
            i_fu_108 <= i_2_fu_168_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            i_fu_108 <= 10'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        empty_reg_399 <= empty_fu_253_p1;
        icmp_ln137_reg_355_pp0_iter0_reg <= icmp_ln137_reg_355;
        packet_5_reg_394 <= {{y_stream_TDATA[199:175]}};
        packet_reg_374 <= {{y_stream_TDATA[99:75]}};
        trunc_ln140_1_reg_384 <= {{y_stream_TDATA[149:125]}};
        trunc_ln140_2_reg_389 <= {{y_stream_TDATA[174:150]}};
        trunc_ln140_8_reg_364 <= {{y_stream_TDATA[49:25]}};
        trunc_ln140_9_reg_369 <= {{y_stream_TDATA[74:50]}};
        trunc_ln140_reg_359 <= trunc_ln140_fu_179_p1;
        trunc_ln140_s_reg_379 <= {{y_stream_TDATA[124:100]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        icmp_ln137_reg_355_pp0_iter1_reg <= icmp_ln137_reg_355_pp0_iter0_reg;
        packet_5_reg_394_pp0_iter1_reg <= packet_5_reg_394;
        packet_reg_374_pp0_iter1_reg <= packet_reg_374;
        trunc_ln140_1_reg_384_pp0_iter1_reg <= trunc_ln140_1_reg_384;
        trunc_ln140_2_reg_389_pp0_iter1_reg <= trunc_ln140_2_reg_389;
        trunc_ln140_8_reg_364_pp0_iter1_reg <= trunc_ln140_8_reg_364;
        trunc_ln140_9_reg_369_pp0_iter1_reg <= trunc_ln140_9_reg_369;
        trunc_ln140_reg_359_pp0_iter1_reg <= trunc_ln140_reg_359;
        trunc_ln140_s_reg_379_pp0_iter1_reg <= trunc_ln140_s_reg_379;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln137_reg_355_pp0_iter2_reg <= icmp_ln137_reg_355_pp0_iter1_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln137_reg_355_pp0_iter3_reg <= icmp_ln137_reg_355_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln137_reg_355 <= icmp_ln137_fu_162_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
        icmp_ln137_reg_355_pp0_iter4_reg <= icmp_ln137_reg_355_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
        trunc_ln3_reg_404 <= {{add_ln227_fu_268_p2[63:4]}};
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state2_pp0_stage1_iter0)) begin
        ap_ST_iter0_fsm_state2_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state2_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_io)) begin
        ap_ST_iter1_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter1_fsm_state4_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state5_io)) begin
        ap_ST_iter2_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state6_io)) begin
        ap_ST_iter2_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_iter2_fsm_state6_blk = 1'b0;
    end
end

assign ap_ST_iter3_fsm_state7_blk = 1'b0;

assign ap_ST_iter3_fsm_state8_blk = 1'b0;

assign ap_ST_iter4_fsm_state10_blk = 1'b0;

assign ap_ST_iter4_fsm_state9_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state11_pp0_stage0_iter5)) begin
        ap_ST_iter5_fsm_state11_blk = 1'b1;
    end else begin
        ap_ST_iter5_fsm_state11_blk = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln137_reg_355 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (ap_loop_exit_ready_pp0_iter5_reg == 1'b1) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_i_1 = 10'd0;
    end else begin
        ap_sig_allocacmp_i_1 = i_fu_108;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state4) & (icmp_ln137_reg_355_pp0_iter0_reg == 1'd0))) begin
        gmem1_blk_n_AW = m_axi_gmem1_AWREADY;
    end else begin
        gmem1_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (icmp_ln137_reg_355_pp0_iter4_reg == 1'd0))) begin
        gmem1_blk_n_B = m_axi_gmem1_BVALID;
    end else begin
        gmem1_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_iter2_fsm_state6) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)))) begin
        gmem1_blk_n_W = m_axi_gmem1_WREADY;
    end else begin
        gmem1_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4) & (icmp_ln137_reg_355_pp0_iter0_reg == 1'd0))) begin
        m_axi_gmem1_AWVALID = 1'b1;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter5_fsm_state11) & (1'b0 == ap_block_state11_pp0_stage0_iter5) & (icmp_ln137_reg_355_pp0_iter4_reg == 1'd0))) begin
        m_axi_gmem1_BREADY = 1'b1;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_condition_563)) begin
        if ((1'b1 == ap_CS_iter2_fsm_state6)) begin
            m_axi_gmem1_WDATA = sext_ln225_14_fu_338_p1;
        end else if ((1'b1 == ap_CS_iter2_fsm_state5)) begin
            m_axi_gmem1_WDATA = sext_ln225_10_fu_313_p1;
        end else begin
            m_axi_gmem1_WDATA = 'bx;
        end
    end else begin
        m_axi_gmem1_WDATA = 'bx;
    end
end

always @ (*) begin
    if (((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state5) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)) | (~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0)))) begin
        m_axi_gmem1_WVALID = 1'b1;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        tile_idx_stream_blk_n = tile_idx_stream_empty_n;
    end else begin
        tile_idx_stream_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        tile_idx_stream_read = 1'b1;
    end else begin
        tile_idx_stream_read = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        y_stream_TDATA_blk_n = y_stream_TVALID;
    end else begin
        y_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (icmp_ln137_reg_355 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
        y_stream_TREADY = 1'b1;
    end else begin
        y_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_iter0_fsm)
        ap_ST_iter0_fsm_state1 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end
        end
        ap_ST_iter0_fsm_state2 : begin
            if (~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io)))) begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state1;
            end else begin
                ap_NS_iter0_fsm = ap_ST_iter0_fsm_state2;
            end
        end
        default : begin
            ap_NS_iter0_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter1_fsm)
        ap_ST_iter1_fsm_state3 : begin
            if ((~(((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state3))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end
        end
        ap_ST_iter1_fsm_state4 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter0_fsm_state2) & (1'b0 == ap_block_state2_pp0_stage1_iter0))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & ((1'b0 == ap_CS_iter0_fsm_state2) | ((1'b1 == ap_CS_iter0_fsm_state2) & (1'b1 == ap_block_state2_pp0_stage1_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state4;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state2_pp0_stage1_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state2))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state3;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter1_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter2_fsm)
        ap_ST_iter2_fsm_state5 : begin
            if ((~((1'b1 == ap_block_state5_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state5))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end
        end
        ap_ST_iter2_fsm_state6 : begin
            if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter1_fsm_state4) & (1'b0 == ap_block_state4_io))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & ((1'b0 == ap_CS_iter1_fsm_state4) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state6;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io))) & (1'b1 == ap_CS_iter1_fsm_state4))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state5;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter2_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter3_fsm)
        ap_ST_iter3_fsm_state7 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state7))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end
        end
        ap_ST_iter3_fsm_state8 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter2_fsm_state6) & (1'b0 == ap_block_state6_io))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & ((1'b0 == ap_CS_iter2_fsm_state6) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io))))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state8;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state6_io) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5))) & (1'b1 == ap_CS_iter2_fsm_state6))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state7;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter3_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter4_fsm)
        ap_ST_iter4_fsm_state9 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state9))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end
        end
        ap_ST_iter4_fsm_state10 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b0 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state10;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter3_fsm_state8))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state9;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter4_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter5_fsm)
        ap_ST_iter5_fsm_state11 : begin
            if (((1'b0 == ap_CS_iter4_fsm_state10) & (1'b0 == ap_block_state11_pp0_stage0_iter5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else if ((((1'b1 == ap_CS_iter5_fsm_state11) & (1'b0 == ap_block_state11_pp0_stage0_iter5) & (icmp_ln137_reg_355_pp0_iter4_reg == 1'd1)) | ((1'b1 == ap_CS_iter4_fsm_state10) & (1'b0 == ap_block_state11_pp0_stage0_iter5)))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (1'b1 == ap_CS_iter4_fsm_state10))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state11;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter5_fsm = 'bx;
        end
    endcase
end

assign add_ln227_fu_268_p2 = ($signed(memory_state_1) + $signed(sext_ln227_fu_264_p1));

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter0_fsm_state2 = ap_CS_iter0_fsm[32'd1];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state3 = ap_CS_iter1_fsm[32'd1];

assign ap_CS_iter1_fsm_state4 = ap_CS_iter1_fsm[32'd2];

assign ap_CS_iter2_fsm_state0 = ap_CS_iter2_fsm[32'd0];

assign ap_CS_iter2_fsm_state5 = ap_CS_iter2_fsm[32'd1];

assign ap_CS_iter2_fsm_state6 = ap_CS_iter2_fsm[32'd2];

assign ap_CS_iter3_fsm_state0 = ap_CS_iter3_fsm[32'd0];

assign ap_CS_iter3_fsm_state7 = ap_CS_iter3_fsm[32'd1];

assign ap_CS_iter3_fsm_state8 = ap_CS_iter3_fsm[32'd2];

assign ap_CS_iter4_fsm_state0 = ap_CS_iter4_fsm[32'd0];

assign ap_CS_iter4_fsm_state10 = ap_CS_iter4_fsm[32'd2];

assign ap_CS_iter4_fsm_state9 = ap_CS_iter4_fsm[32'd1];

assign ap_CS_iter5_fsm_state0 = ap_CS_iter5_fsm[32'd0];

assign ap_CS_iter5_fsm_state11 = ap_CS_iter5_fsm[32'd1];

always @ (*) begin
    ap_block_state11_pp0_stage0_iter5 = ((icmp_ln137_reg_355_pp0_iter4_reg == 1'd0) & (m_axi_gmem1_BVALID == 1'b0));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state2_pp0_stage1_iter0 = (((y_stream_TVALID == 1'b0) & (icmp_ln137_reg_355 == 1'd0)) | ((icmp_ln137_reg_355 == 1'd0) & (tile_idx_stream_empty_n == 1'b0)));
end

always @ (*) begin
    ap_block_state4_io = ((icmp_ln137_reg_355_pp0_iter0_reg == 1'd0) & (m_axi_gmem1_AWREADY == 1'b0));
end

always @ (*) begin
    ap_block_state5_io = ((icmp_ln137_reg_355_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_block_state6_io = ((icmp_ln137_reg_355_pp0_iter1_reg == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_condition_248 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) | ((1'b1 == ap_CS_iter2_fsm_state6) & (1'b1 == ap_block_state6_io)) | ((1'b1 == ap_CS_iter2_fsm_state5) & (1'b1 == ap_block_state5_io)) | ((1'b1 == ap_CS_iter1_fsm_state4) & (1'b1 == ap_block_state4_io))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

always @ (*) begin
    ap_condition_563 = (~((1'b1 == ap_CS_iter5_fsm_state11) & (1'b1 == ap_block_state11_pp0_stage0_iter5)) & (icmp_ln137_reg_355_pp0_iter1_reg == 1'd0));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage1;

assign empty_fu_253_p1 = tile_idx_stream_dout[30:0];

assign i_2_fu_168_p2 = (ap_sig_allocacmp_i_1 + 10'd1);

assign icmp_ln137_fu_162_p2 = ((ap_sig_allocacmp_i_1 == 10'd960) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = 64'd0;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd0;

assign m_axi_gmem1_ARLOCK = 2'd0;

assign m_axi_gmem1_ARPROT = 3'd0;

assign m_axi_gmem1_ARQOS = 4'd0;

assign m_axi_gmem1_ARREGION = 4'd0;

assign m_axi_gmem1_ARSIZE = 3'd0;

assign m_axi_gmem1_ARUSER = 1'd0;

assign m_axi_gmem1_ARVALID = 1'b0;

assign m_axi_gmem1_AWADDR = sext_ln137_fu_283_p1;

assign m_axi_gmem1_AWBURST = 2'd0;

assign m_axi_gmem1_AWCACHE = 4'd0;

assign m_axi_gmem1_AWID = 1'd0;

assign m_axi_gmem1_AWLEN = 32'd2;

assign m_axi_gmem1_AWLOCK = 2'd0;

assign m_axi_gmem1_AWPROT = 3'd0;

assign m_axi_gmem1_AWQOS = 4'd0;

assign m_axi_gmem1_AWREGION = 4'd0;

assign m_axi_gmem1_AWSIZE = 3'd0;

assign m_axi_gmem1_AWUSER = 1'd0;

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd65535;

assign m_axi_gmem1_WUSER = 1'd0;

assign packet_4_fu_302_p5 = {{{{packet_reg_374_pp0_iter1_reg}, {sext_ln225_fu_293_p1}}, {sext_ln225_8_fu_296_p1}}, {sext_ln225_9_fu_299_p1}};

assign packet_6_fu_327_p5 = {{{{packet_5_reg_394_pp0_iter1_reg}, {sext_ln225_11_fu_318_p1}}, {sext_ln225_12_fu_321_p1}}, {sext_ln225_13_fu_324_p1}};

assign sext_ln137_fu_283_p1 = $signed(trunc_ln3_reg_404);

assign sext_ln225_10_fu_313_p1 = $signed(packet_4_fu_302_p5);

assign sext_ln225_11_fu_318_p1 = $signed(trunc_ln140_2_reg_389_pp0_iter1_reg);

assign sext_ln225_12_fu_321_p1 = $signed(trunc_ln140_1_reg_384_pp0_iter1_reg);

assign sext_ln225_13_fu_324_p1 = $signed(trunc_ln140_s_reg_379_pp0_iter1_reg);

assign sext_ln225_14_fu_338_p1 = $signed(packet_6_fu_327_p5);

assign sext_ln225_8_fu_296_p1 = $signed(trunc_ln140_8_reg_364_pp0_iter1_reg);

assign sext_ln225_9_fu_299_p1 = $signed(trunc_ln140_reg_359_pp0_iter1_reg);

assign sext_ln225_fu_293_p1 = $signed(trunc_ln140_9_reg_369_pp0_iter1_reg);

assign sext_ln227_fu_264_p1 = $signed(shl_ln_fu_257_p3);

assign shl_ln_fu_257_p3 = {{empty_reg_399}, {5'd0}};

assign trunc_ln140_fu_179_p1 = y_stream_TDATA[24:0];

endmodule //STATE_AXI_writeback_indexed_order_Pipeline_VITIS_LOOP_137_1
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
`default_nettype none

module STATE_AXI_gmem1_m_axi
#(parameter
    CONSERVATIVE            = 0,
    NUM_READ_OUTSTANDING    = 2,
    NUM_WRITE_OUTSTANDING   = 2,
    MAX_READ_BURST_LENGTH   = 16,
    MAX_WRITE_BURST_LENGTH  = 16,
    C_M_AXI_ID_WIDTH        = 1,
    C_M_AXI_ADDR_WIDTH      = 32,
    C_M_AXI_DATA_WIDTH      = 32, // power of 2 & range: 2 to 1024
    C_M_AXI_AWUSER_WIDTH    = 1,
    C_M_AXI_ARUSER_WIDTH    = 1,
    C_M_AXI_WUSER_WIDTH     = 1,
    C_M_AXI_RUSER_WIDTH     = 1,
    C_M_AXI_BUSER_WIDTH     = 1,
    C_TARGET_ADDR           = 32'h00000000,
    C_USER_VALUE            = 1'b0,
    C_PROT_VALUE            = 3'b000,
    C_CACHE_VALUE           = 4'b0011,
    USER_DW                 = 32, // multiple of 8
    USER_AW                 = 32,
    USER_MAXREQS            = 16,
    USER_RFIFONUM_WIDTH     = 6,
    MAXI_BUFFER_IMPL        = "block"
)(
    
    
    // system signal
    input  wire                               ACLK,
    input  wire                               ARESET,
    input  wire                               ACLK_EN,
    // write address channel
    output wire [C_M_AXI_ID_WIDTH-1:0]        AWID,
    output wire [C_M_AXI_ADDR_WIDTH-1:0]      AWADDR,
    output wire [7:0]                         AWLEN,
    output wire [2:0]                         AWSIZE,
    output wire [1:0]                         AWBURST,
    output wire [1:0]                         AWLOCK,
    output wire [3:0]                         AWCACHE,
    output wire [2:0]                         AWPROT,
    output wire [3:0]                         AWQOS,
    output wire [3:0]                         AWREGION,
    output wire [C_M_AXI_AWUSER_WIDTH-1:0]    AWUSER,
    output wire                               AWVALID,
    input  wire                               AWREADY,
    // write data channel
    output wire [C_M_AXI_ID_WIDTH-1:0]        WID,
    output wire [C_M_AXI_DATA_WIDTH-1:0]      WDATA,
    output wire [C_M_AXI_DATA_WIDTH/8-1:0]    WSTRB,
    output wire                               WLAST,
    output wire [C_M_AXI_WUSER_WIDTH-1:0]     WUSER,
    output wire                               WVALID,
    input  wire                               WREADY,
    // write response channel
    input  wire [C_M_AXI_ID_WIDTH-1:0]        BID,
    input  wire [1:0]                         BRESP,
    input  wire [C_M_AXI_BUSER_WIDTH-1:0]     BUSER,
    input  wire                               BVALID,
    output wire                               BREADY,
    // read address channel
    output wire [C_M_AXI_ID_WIDTH-1:0]        ARID,
    output wire [C_M_AXI_ADDR_WIDTH-1:0]      ARADDR,
    output wire [7:0]                         ARLEN,
    output wire [2:0]                         ARSIZE,
    output wire [1:0]                         ARBURST,
    output wire [1:0]                         ARLOCK,
    output wire [3:0]                         ARCACHE,
    output wire [2:0]                         ARPROT,
    output wire [3:0]                         ARQOS,
    output wire [3:0]                         ARREGION,
    output wire [C_M_AXI_ARUSER_WIDTH-1:0]    ARUSER,
    output wire                               ARVALID,
    input  wire                               ARREADY,
    // read data channel
    input  wire [C_M_AXI_ID_WIDTH-1:0]        RID,
    input  wire [C_M_AXI_DATA_WIDTH-1:0]      RDATA,
    input  wire [1:0]                         RRESP,
    input  wire                               RLAST,
    input  wire [C_M_AXI_RUSER_WIDTH-1:0]     RUSER,
    input  wire                               RVALID,
    output wire                               RREADY,

    // internal bus ports
    // write address
    input  wire [USER_AW-1:0]                 I_AWADDR,
    input  wire [31:0]                        I_AWLEN,
    input  wire                               I_AWVALID,
    output wire                               I_AWREADY,
    // write data
    input  wire [USER_DW-1:0]                 I_WDATA,
    input  wire [USER_DW/8-1:0]               I_WSTRB,
    input  wire                               I_WVALID,
    output wire                               I_WREADY,
    // write response
    output wire                               I_BVALID,
    input  wire                               I_BREADY,
    // read address
    input  wire [USER_AW-1:0]                 I_ARADDR,
    input  wire [31:0]                        I_ARLEN,
    input  wire                               I_ARVALID,
    output wire                               I_ARREADY,
    // read data
    output wire [USER_DW-1:0]                 I_RDATA,
    output wire                               I_RVALID,
    input  wire                               I_RREADY,
    output wire [USER_RFIFONUM_WIDTH-1:0]     I_RFIFONUM);
//------------------------Local signal-------------------

    wire [C_M_AXI_ADDR_WIDTH-1:0]   AWADDR_Dummy;
    wire [31:0]                     AWLEN_Dummy;
    wire                            AWVALID_Dummy;
    wire                            AWREADY_Dummy;
    wire [C_M_AXI_DATA_WIDTH-1:0]   WDATA_Dummy;
    wire [C_M_AXI_DATA_WIDTH/8-1:0] WSTRB_Dummy;
    wire                            WVALID_Dummy;
    wire                            WREADY_Dummy;
    wire                            BVALID_Dummy;
    wire                            BREADY_Dummy;
    wire [C_M_AXI_ADDR_WIDTH-1:0]   ARADDR_Dummy;
    wire [31:0]                     ARLEN_Dummy;
    wire                            ARVALID_Dummy;
    wire                            ARREADY_Dummy;
    wire [C_M_AXI_DATA_WIDTH-1:0]   RDATA_Dummy;
    wire [1:0]                      RLAST_Dummy;
    wire                            RVALID_Dummy;
    wire                            RREADY_Dummy;
    wire                            RBURST_READY_Dummy;
    
//------------------------Instantiation------------------
    // STATE_AXI_gmem1_m_axi_store
    STATE_AXI_gmem1_m_axi_store #(
        .C_TARGET_ADDR           ( C_TARGET_ADDR ),
        .NUM_WRITE_OUTSTANDING   ( NUM_WRITE_OUTSTANDING ),
        .MAX_WRITE_BURST_LENGTH  ( MAX_WRITE_BURST_LENGTH ),
        .BUS_ADDR_WIDTH          ( C_M_AXI_ADDR_WIDTH ),
        .BUS_DATA_WIDTH          ( C_M_AXI_DATA_WIDTH ),
        .USER_DW                 ( USER_DW ),
        .USER_AW                 ( USER_AW ),
        .USER_MAXREQS            ( USER_MAXREQS ),
        .BUFFER_IMPL             ( MAXI_BUFFER_IMPL )
    ) store_unit (
        .ACLK                    ( ACLK ),
        .ARESET                  ( ARESET ),
        .ACLK_EN                 ( ACLK_EN ),
        .out_AXI_AWADDR          ( AWADDR_Dummy ),
        .out_AXI_AWLEN           ( AWLEN_Dummy ),
        .out_AXI_AWVALID         ( AWVALID_Dummy ),
        .in_AXI_AWREADY          ( AWREADY_Dummy ),
        .out_AXI_WDATA           ( WDATA_Dummy ),
        .out_AXI_WSTRB           ( WSTRB_Dummy ),
        .out_AXI_WVALID          ( WVALID_Dummy ),
        .in_AXI_WREADY           ( WREADY_Dummy ),
        .in_AXI_BVALID           ( BVALID_Dummy ),
        .out_AXI_BREADY          ( BREADY_Dummy ),
        .in_HLS_AWADDR           ( I_AWADDR ),
        .in_HLS_AWLEN            ( I_AWLEN ),
        .in_HLS_AWVALID          ( I_AWVALID ),
        .out_HLS_AWREADY         ( I_AWREADY ),
        .in_HLS_WDATA            ( I_WDATA ),
        .in_HLS_WSTRB            ( I_WSTRB ),
        .in_HLS_WVALID           ( I_WVALID ),
        .out_HLS_WREADY          ( I_WREADY ),
        .out_HLS_BVALID          ( I_BVALID ),
        .in_HLS_BREADY           ( I_BREADY ));

    // STATE_AXI_gmem1_m_axi_load
    STATE_AXI_gmem1_m_axi_load #(
        .C_TARGET_ADDR           ( C_TARGET_ADDR ),
        .NUM_READ_OUTSTANDING    ( NUM_READ_OUTSTANDING ),
        .MAX_READ_BURST_LENGTH   ( MAX_READ_BURST_LENGTH ),
        .BUS_ADDR_WIDTH          ( C_M_AXI_ADDR_WIDTH ),
        .BUS_DATA_WIDTH          ( C_M_AXI_DATA_WIDTH ),
        .USER_DW                 ( USER_DW ),
        .USER_AW                 ( USER_AW ),
        .USER_MAXREQS            ( USER_MAXREQS ),
        .USER_RFIFONUM_WIDTH     ( USER_RFIFONUM_WIDTH ),
        .BUFFER_IMPL             ( MAXI_BUFFER_IMPL )
    ) load_unit (
        .ACLK                    ( ACLK ),
        .ARESET                  ( ARESET ),
        .ACLK_EN                 ( ACLK_EN ),
        .out_AXI_ARADDR          ( ARADDR_Dummy ),
        .out_AXI_ARLEN           ( ARLEN_Dummy ),
        .out_AXI_ARVALID         ( ARVALID_Dummy ),
        .in_AXI_ARREADY          ( ARREADY_Dummy ),
        .in_AXI_RDATA            ( RDATA_Dummy ),
        .in_AXI_RLAST            ( RLAST_Dummy ),
        .in_AXI_RVALID           ( RVALID_Dummy ),
        .out_AXI_RREADY          ( RREADY_Dummy ),
        .out_AXI_RBURST_READY    ( RBURST_READY_Dummy),
        .in_HLS_ARADDR           ( I_ARADDR ),
        .in_HLS_ARLEN            ( I_ARLEN ),
        .in_HLS_ARVALID          ( I_ARVALID ),
        .out_HLS_ARREADY         ( I_ARREADY ),
        .out_HLS_RDATA           ( I_RDATA ),
        .out_HLS_RVALID          ( I_RVALID ),
        .in_HLS_RREADY           ( I_RREADY ),
        .out_HLS_RFIFONUM        ( I_RFIFONUM ));

    // STATE_AXI_gmem1_m_axi_write
    STATE_AXI_gmem1_m_axi_write #(
        .CONSERVATIVE            ( CONSERVATIVE),
        .C_M_AXI_ID_WIDTH        ( C_M_AXI_ID_WIDTH ),
        .C_M_AXI_AWUSER_WIDTH    ( C_M_AXI_AWUSER_WIDTH ),
        .C_M_AXI_WUSER_WIDTH     ( C_M_AXI_WUSER_WIDTH ),
        .C_M_AXI_BUSER_WIDTH     ( C_M_AXI_BUSER_WIDTH ),
        .C_USER_VALUE            ( C_USER_VALUE ),
        .C_PROT_VALUE            ( C_PROT_VALUE ),
        .C_CACHE_VALUE           ( C_CACHE_VALUE ),
        .BUS_ADDR_WIDTH          ( C_M_AXI_ADDR_WIDTH ),
        .BUS_DATA_WIDTH          ( C_M_AXI_DATA_WIDTH ),
        .NUM_WRITE_OUTSTANDING   ( NUM_WRITE_OUTSTANDING ),
        .MAX_WRITE_BURST_LENGTH  ( MAX_WRITE_BURST_LENGTH )
    ) bus_write (
        .ACLK                    ( ACLK ),
        .ARESET                  ( ARESET ),
        .ACLK_EN                 ( ACLK_EN ),
        .out_BUS_AWID            ( AWID ),
        .out_BUS_AWSIZE          ( AWSIZE ),
        .out_BUS_AWBURST         ( AWBURST ),
        .out_BUS_AWLOCK          ( AWLOCK ),
        .out_BUS_AWCACHE         ( AWCACHE ),
        .out_BUS_AWPROT          ( AWPROT ),
        .out_BUS_AWQOS           ( AWQOS ),
        .out_BUS_AWREGION        ( AWREGION ),
        .out_BUS_AWUSER          ( AWUSER ),
        .out_BUS_AWADDR          ( AWADDR ),
        .out_BUS_AWLEN           ( AWLEN ),
        
        
        .out_BUS_AWVALID         ( AWVALID ),
        .in_BUS_AWREADY          ( AWREADY ),
        .out_BUS_WID             ( WID),
        .out_BUS_WUSER           ( WUSER),
        .out_BUS_WDATA           ( WDATA ),
        .out_BUS_WSTRB           ( WSTRB ),
        .out_BUS_WLAST           ( WLAST ),
        
        
        .out_BUS_WVALID          ( WVALID ),
        .in_BUS_WREADY           ( WREADY ),
        .in_BUS_BID              ( BID ),
        .in_BUS_BRESP            ( BRESP ),
        .in_BUS_BUSER            ( BUSER ),
        .in_BUS_BVALID           ( BVALID ),
        
        
        .out_BUS_BREADY          ( BREADY ),
        .in_HLS_AWVALID          ( AWVALID_Dummy ),
        .out_HLS_AWREADY         ( AWREADY_Dummy ),
        .in_HLS_AWADDR           ( AWADDR_Dummy ),
        .in_HLS_AWLEN            ( AWLEN_Dummy ),
        .in_HLS_WVALID           ( WVALID_Dummy ),
        .out_HLS_WREADY          ( WREADY_Dummy ),
        .in_HLS_WSTRB            ( WSTRB_Dummy ),
        .in_HLS_WDATA            ( WDATA_Dummy ),
        .out_HLS_BVALID          ( BVALID_Dummy ),
        .in_HLS_BREADY           ( BREADY_Dummy ));

    // STATE_AXI_gmem1_m_axi_read
    STATE_AXI_gmem1_m_axi_read #(
        .C_M_AXI_ID_WIDTH         ( C_M_AXI_ID_WIDTH ),
        .C_M_AXI_ARUSER_WIDTH     ( C_M_AXI_ARUSER_WIDTH ),
        .C_M_AXI_RUSER_WIDTH      ( C_M_AXI_RUSER_WIDTH ),
        .C_USER_VALUE             ( C_USER_VALUE ),
        .C_PROT_VALUE             ( C_PROT_VALUE ),
        .C_CACHE_VALUE            ( C_CACHE_VALUE ),
        .BUS_ADDR_WIDTH           ( C_M_AXI_ADDR_WIDTH ),
        .BUS_DATA_WIDTH           ( C_M_AXI_DATA_WIDTH ),
        .NUM_READ_OUTSTANDING     ( NUM_READ_OUTSTANDING ),
        .MAX_READ_BURST_LENGTH    ( MAX_READ_BURST_LENGTH )
    ) bus_read (
        .ACLK                     ( ACLK ),
        .ARESET                   ( ARESET ),
        .ACLK_EN                  ( ACLK_EN ),
        .out_BUS_ARID             ( ARID ),
        .out_BUS_ARADDR           ( ARADDR ),
        .out_BUS_ARLEN            ( ARLEN ),
        .out_BUS_ARSIZE           ( ARSIZE ),
        .out_BUS_ARBURST          ( ARBURST ),
        .out_BUS_ARLOCK           ( ARLOCK ),
        .out_BUS_ARCACHE          ( ARCACHE ),
        .out_BUS_ARPROT           ( ARPROT ),
        .out_BUS_ARQOS            ( ARQOS ),
        .out_BUS_ARREGION         ( ARREGION ),
        .out_BUS_ARUSER           ( ARUSER ),
        
        
        .out_BUS_ARVALID          ( ARVALID ),
        .in_BUS_ARREADY           ( ARREADY ),
        .in_BUS_RID               ( RID ),
        .in_BUS_RDATA             ( RDATA ),
        .in_BUS_RRESP             ( RRESP ),
        .in_BUS_RLAST             ( RLAST ),
        .in_BUS_RUSER             ( RUSER ),
        .in_BUS_RVALID            ( RVALID ),
        
        
        .out_BUS_RREADY           ( RREADY ),
        .in_HLS_ARVALID           ( ARVALID_Dummy ),
        .out_HLS_ARREADY          ( ARREADY_Dummy ),
        .in_HLS_ARADDR            ( ARADDR_Dummy ),
        .in_HLS_ARLEN             ( ARLEN_Dummy ),
        .out_HLS_RVALID           ( RVALID_Dummy ),
        .in_HLS_RREADY            ( RREADY_Dummy ),
        .in_HLS_RBUST_READY       ( RBURST_READY_Dummy),
        .out_HLS_RDATA            ( RDATA_Dummy ),
        .out_HLS_RLAST            ( RLAST_Dummy ));

    
endmodule
`default_nettype wire
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps

module STATE_AXI_gmem1_m_axi_load
#(parameter
    C_TARGET_ADDR                         = 32'h00000000,
    NUM_READ_OUTSTANDING                  = 2,
    MAX_READ_BURST_LENGTH                 = 16,
    BUS_ADDR_WIDTH                        = 32,
    BUS_DATA_WIDTH                        = 32,
    USER_DW                               = 16,
    USER_AW                               = 32,
    USER_MAXREQS                          = 16,
    USER_RFIFONUM_WIDTH                   = 6,
    BUFFER_IMPL                           = "auto"
)(
    // system signal
    input  wire                           ACLK,
    input  wire                           ARESET,
    input  wire                           ACLK_EN,

    // read address channel
    output wire [BUS_ADDR_WIDTH-1:0]      out_AXI_ARADDR,
    output wire [31:0]                    out_AXI_ARLEN,
    output wire                           out_AXI_ARVALID,
    input  wire                           in_AXI_ARREADY,
    // read data channel
    input  wire [BUS_DATA_WIDTH-1:0]      in_AXI_RDATA,
    input  wire [1:0]                     in_AXI_RLAST,
    input  wire                           in_AXI_RVALID,
    output wire                           out_AXI_RREADY,
    output wire                           out_AXI_RBURST_READY,

    // internal bus ports
    // read address
    input  wire [USER_AW-1:0]             in_HLS_ARADDR,
    input  wire [31:0]                    in_HLS_ARLEN,
    input  wire                           in_HLS_ARVALID,
    output wire                           out_HLS_ARREADY,
    // read data
    output wire [USER_DW-1:0]             out_HLS_RDATA,
    output wire                           out_HLS_RVALID,
    input  wire                           in_HLS_RREADY,
    output wire [USER_RFIFONUM_WIDTH-1:0] out_HLS_RFIFONUM
);

//------------------------Parameter----------------------
    localparam
        USER_DATA_WIDTH = calc_data_width(USER_DW),
        USER_DATA_BYTES = USER_DATA_WIDTH / 8,
        USER_ADDR_ALIGN = log2(USER_DATA_BYTES),
        BUS_ADDR_ALIGN  = log2(BUS_DATA_WIDTH/8),
        RBUFF_DEPTH     = NUM_READ_OUTSTANDING * MAX_READ_BURST_LENGTH,
        TARGET_ADDR     = C_TARGET_ADDR & (32'hffffffff << USER_ADDR_ALIGN);

//------------------------Task and function--------------
    function integer calc_data_width;
        input integer x;
        integer y;
    begin
        y = 8;
        while (y < x) y = y * 2;
        calc_data_width = y;
    end
    endfunction

    function integer log2;
        input integer x;
        integer n, m;
    begin
        n = 0;
        m = 1;
        while (m < x) begin
            n = n + 1;
            m = m * 2;
        end
        log2 = n;
    end
    endfunction

//------------------------Local signal-------------------

    wire                           next_rreq;
    wire                           ready_for_rreq;
    wire                           rreq_ready;

    wire [USER_AW-1 : 0]           rreq_addr;
    wire [31:0]                    rreq_len;
    wire                           rreq_valid;

    wire                           valid_length;

    reg  [BUS_ADDR_WIDTH-1 : 0]    tmp_addr;
    reg  [31:0]                    tmp_len;
    reg                            tmp_valid;

    wire                           burst_ready;
    wire                           beat_valid;
    wire                           next_beat;
    wire                           last_beat;
    wire [BUS_DATA_WIDTH-1 : 0]    beat_data;
    wire [log2(RBUFF_DEPTH) : 0]   beat_nvalid;

    reg                            ready_for_outstanding;
    
    // regslice io ?  no 
    
    // enable regslice on R channel  no 

//------------------------Instantiation------------------

    

    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH        (USER_AW + 32),
        .ADDR_WIDTH        (log2(USER_MAXREQS)),
        .DEPTH             (USER_MAXREQS)
    ) fifo_rreq (
        .clk               (ACLK),
        .reset             (ARESET),
        .clk_en            (ACLK_EN),
        .if_full_n         (out_HLS_ARREADY),
        .if_write          (in_HLS_ARVALID),
        .if_din            ({in_HLS_ARLEN, in_HLS_ARADDR}),
        .if_empty_n        (rreq_valid),
        .if_read           (next_rreq),
        .if_dout           ({rreq_len, rreq_addr}),
        .if_num_data_valid ());

    // ===================================================================
    // start of ARADDR PREPROCESSOR
    
    assign next_rreq       = rreq_valid && ready_for_rreq;
    assign ready_for_rreq  = ~tmp_valid || (in_AXI_ARREADY && rreq_ready);
    assign valid_length    = (rreq_len != 32'b0) && !rreq_len[31];

    assign out_AXI_ARLEN   = tmp_len;   // Byte length
    assign out_AXI_ARADDR  = tmp_addr;  // Byte address
    assign out_AXI_ARVALID = tmp_valid && rreq_ready;

    always @(posedge ACLK)
    begin
        if (ARESET) begin
            tmp_len  <= 0;
            tmp_addr <= 0;
        end
        else if (ACLK_EN) begin
            if(next_rreq) begin
                tmp_len  <= (rreq_len << USER_ADDR_ALIGN) - 1;            // byte length
                tmp_addr <= TARGET_ADDR + (rreq_addr << USER_ADDR_ALIGN); // byte address
            end
        end
    end
 
    always @(posedge ACLK) 
    begin
        if (ARESET)
            tmp_valid <= 1'b0;
        else if (ACLK_EN) begin
            if (next_rreq && valid_length)
                tmp_valid <= 1'b1;
            else if (in_AXI_ARREADY && rreq_ready)
                tmp_valid <= 1'b0;
        end
    end

    // end of ARADDR PREPROCESSOR
    // ===================================================================

    

    STATE_AXI_gmem1_m_axi_fifo #(
        .MEM_STYLE         (BUFFER_IMPL),
        .DATA_WIDTH        (BUS_DATA_WIDTH + 2),
        .ADDR_WIDTH        (log2(RBUFF_DEPTH)),
        .DEPTH             (RBUFF_DEPTH)
    ) buff_rdata (
        .clk               (ACLK),
        .reset             (ARESET),
        .clk_en            (ACLK_EN),
        .if_full_n         (out_AXI_RREADY),
        .if_write          (in_AXI_RVALID),
        .if_din            ({in_AXI_RLAST, in_AXI_RDATA}),
        .if_empty_n        (beat_valid),
        .if_read           (next_beat),
        .if_dout           ({burst_ready, last_beat, beat_data}),
        .if_num_data_valid (beat_nvalid));

    assign out_AXI_RBURST_READY = ready_for_outstanding;

    always @(posedge ACLK) 
    begin
        if (ARESET)
            ready_for_outstanding <= 1'b0;
        else if (ACLK_EN) begin
            if (next_beat)
                ready_for_outstanding <= burst_ready;
            else
                ready_for_outstanding <= 1'b0;
        end
    end
    // ===================================================================
    // start of RDATA PREPROCESSOR
    generate
    if (USER_DATA_WIDTH == BUS_DATA_WIDTH) begin : bus_equal_gen

        assign rreq_ready       = 1'b1; 
        // regslice io ?  no
        assign next_beat        = in_HLS_RREADY;
        assign out_HLS_RDATA    = beat_data[USER_DW-1 : 0];
        assign out_HLS_RVALID   = beat_valid;
        assign out_HLS_RFIFONUM = beat_nvalid; // 

    end
    else if (USER_DATA_WIDTH < BUS_DATA_WIDTH) begin : bus_wide_gen
        localparam
            TOTAL_SPLIT  = BUS_DATA_WIDTH / USER_DATA_WIDTH,
            SPLIT_ALIGN  = log2(TOTAL_SPLIT);

        wire [USER_AW - 1:0]        tmp_addr_end;

        wire                        offset_full_n;
        wire                        offset_write;
        wire [SPLIT_ALIGN-1 : 0]    start_offset;
        wire [SPLIT_ALIGN-1 : 0]    end_offset;

        wire                        offset_valid;
        wire                        next_offset;
        wire [SPLIT_ALIGN-1 : 0]    head_offset;
        wire [SPLIT_ALIGN-1 : 0]    tail_offset;

        reg                         first_beat;

        wire                        first_data;
        wire                        last_data;
        wire                        ready_for_data;

        reg  [BUS_DATA_WIDTH-1 : 0] data_buf;
        reg                         data_valid;

        reg  [USER_RFIFONUM_WIDTH-1:0] rdata_nvalid; 
        reg  [SPLIT_ALIGN : 0]      data_nvalid;
        wire [SPLIT_ALIGN : 0]      split_nvalid;
        
        wire [SPLIT_ALIGN-1 : 0]    split_cnt;
        reg  [SPLIT_ALIGN-1 : 0]    split_cnt_buf;

        wire                        first_split;
        wire                        next_split;
        wire                        last_split;

        // Recording the offset of start & end address to extract the expect data from beats when USER_DW < BUS_DW.
        STATE_AXI_gmem1_m_axi_fifo #(
            .DATA_WIDTH         (2*SPLIT_ALIGN),
            .ADDR_WIDTH         (log2(NUM_READ_OUTSTANDING)),
            .DEPTH              (NUM_READ_OUTSTANDING)
        ) rreq_offset (
            .clk                (ACLK),
            .reset              (ARESET),
            .clk_en             (ACLK_EN),
            .if_full_n          (offset_full_n),
            .if_write           (offset_write),
            .if_din             ({start_offset, end_offset}),
            .if_empty_n         (offset_valid),
            .if_read            (next_offset),
            .if_dout            ({head_offset, tail_offset}),
            .if_num_data_valid  ());

        assign rreq_ready       = offset_full_n | ~offset_write;
        assign tmp_addr_end     = tmp_addr + tmp_len;

        assign start_offset     = tmp_addr[BUS_ADDR_ALIGN - 1 : 0] >> USER_ADDR_ALIGN;
        assign end_offset       = tmp_addr_end[BUS_ADDR_ALIGN - 1 : 0] >> USER_ADDR_ALIGN;
        assign offset_write     = tmp_valid & in_AXI_ARREADY;

        assign next_offset      = (last_beat & beat_valid) & last_split;
        assign next_beat        = last_split;

        // regslice io ?  no
        assign out_HLS_RDATA    = data_buf[USER_DW-1 : 0];
        assign out_HLS_RVALID   = data_valid;
        assign out_HLS_RFIFONUM = rdata_nvalid + data_nvalid;
        assign ready_for_data   = ~data_valid | in_HLS_RREADY; // 

        assign first_data       = first_beat && beat_valid && offset_valid;
        assign last_data        = last_beat && beat_valid && offset_valid;

        assign first_split      = (~first_data) ? (split_cnt == 0 && beat_valid && ready_for_data) : ((split_cnt == head_offset) && ready_for_data);
        assign last_split       = (~last_data)  ? (split_cnt == (TOTAL_SPLIT-1) && ready_for_data) : ((split_cnt == tail_offset) && ready_for_data);
        assign next_split       = (~first_data) ? (split_cnt != 0 && ready_for_data)               : ((split_cnt != head_offset) && ready_for_data);

        assign split_cnt        = (first_data && (split_cnt_buf == 0)) ? head_offset : split_cnt_buf;

        assign split_nvalid     = (first_data && last_data)  ? tail_offset - head_offset + 1 :
                                   first_data                ? TOTAL_SPLIT - head_offset     :
                                   last_data                 ? tail_offset + 1               :
                                   TOTAL_SPLIT;
        always @(posedge ACLK)
        begin
            if (ARESET)
                split_cnt_buf <= 0;
            else if (ACLK_EN) begin 
                if (last_split)
                    split_cnt_buf <= 0;
                else if (first_split || next_split)
                    split_cnt_buf <= split_cnt + 1;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                first_beat <= 1'b1;
            else if (ACLK_EN) begin
                if (last_beat && last_split)
                    first_beat <= 1'b1;
                else if (first_beat && last_split)
                    first_beat <= 1'b0;
            end
        end

        always @(posedge ACLK)
        begin
            if (ACLK_EN) begin
                if (first_split & first_data)
                    data_buf <= beat_data >> (head_offset * USER_DATA_WIDTH);
                else if (first_split)
                    data_buf <= beat_data;
                else if (next_split)
                    data_buf <= data_buf >> USER_DATA_WIDTH;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                data_valid <= 1'b0;
            else if (ACLK_EN) begin
                if (first_split)
                    data_valid <= 1'b1;
                else if (~(first_split || next_split) && ready_for_data)
                    data_valid <= 1'b0;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                data_nvalid <= 0;
            else if (ACLK_EN) begin
                if (first_split)
                    data_nvalid <= split_nvalid;
                else if (next_split)
                    data_nvalid <= data_nvalid - 1;
                else if (~(first_split || next_split) && ready_for_data)
                    data_nvalid <= 0;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                rdata_nvalid <= 0;
            else if (ACLK_EN) begin
                if (!beat_valid)
                    rdata_nvalid <= 0;
                else
                    rdata_nvalid <= ((beat_nvalid - 1) << SPLIT_ALIGN);
            end
        end
        
    end
    else begin : bus_narrow_gen
        localparam
            TOTAL_PADS      = USER_DATA_WIDTH / BUS_DATA_WIDTH,
            PAD_ALIGN       = log2(TOTAL_PADS);

        reg [USER_DATA_WIDTH-1 : 0] data_buf;
        reg                         data_valid;
        reg [PAD_ALIGN:0]           data_nvalid;
        wire                        ready_for_data;
        wire [USER_RFIFONUM_WIDTH-1 : 0] rdata_num_vld;

        wire [TOTAL_PADS - 1:0]     pad_oh;
        reg  [TOTAL_PADS - 1:0]     pad_oh_reg;

        reg                         first_pad;
        wire                        last_pad;
        wire                        next_pad;

        assign rreq_ready       = 1'b1; 
        assign next_beat        = next_pad;
        assign rdata_num_vld    = beat_nvalid[log2(RBUFF_DEPTH) : PAD_ALIGN] + (beat_nvalid[PAD_ALIGN-1:0] + data_nvalid) >> PAD_ALIGN;
        
        // regslice io ?  no
        assign out_HLS_RDATA    = data_buf[USER_DW-1 : 0];
        assign out_HLS_RVALID   = data_valid;
        assign out_HLS_RFIFONUM = rdata_num_vld;
        assign ready_for_data   = ~data_valid | in_HLS_RREADY;// 

        assign next_pad         = beat_valid && ready_for_data;
        assign last_pad         = pad_oh[TOTAL_PADS - 1];

        always @(posedge ACLK)
        begin
            if (ARESET)
                first_pad <= 1'b1;
            else if (ACLK_EN) begin
                if (next_pad && ~last_pad)
                    first_pad <= 1'b0;
                else if (next_pad && last_pad)
                    first_pad <= 1'b1;
            end
        end

        assign pad_oh = (beat_valid == 0)  ?  0 :
                        (first_pad)        ?  1 :
                        pad_oh_reg;
 
        always @(posedge ACLK)
        begin
            if (ARESET)
                pad_oh_reg <= 0;
            else if (ACLK_EN) begin
                if (next_pad)
                    pad_oh_reg <= {pad_oh[TOTAL_PADS - 2:0], 1'b0};
            end
        end

        genvar  i;
        for (i = 0; i < TOTAL_PADS; i = i + 1) begin : data_gen
            always @(posedge ACLK)
            begin
                if (ACLK_EN) begin
                    if (pad_oh[i] == 1'b1 && ready_for_data)
                        data_buf[i*BUS_DATA_WIDTH +: BUS_DATA_WIDTH] <= beat_data;
                end
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                data_valid <= 1'b0;
            else if (ACLK_EN) begin
                if (next_beat)
                    data_valid <= 1'b1;
                else if (ready_for_data)
                    data_valid <= 1'b0;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                data_nvalid <= 0;
            else if (ACLK_EN) begin
                if (first_pad)
                    data_nvalid <= 1;
                else if (next_pad)
                    data_nvalid <= data_nvalid + 1;
            end
        end

    end
    endgenerate
    // end of RDATA PREPROCESSOR
    // ===================================================================

endmodule


module STATE_AXI_gmem1_m_axi_store
#(parameter
    C_TARGET_ADDR           = 32'h00000000,
    NUM_WRITE_OUTSTANDING   = 2,
    MAX_WRITE_BURST_LENGTH  = 16,
    BUS_ADDR_WIDTH          = 32,
    BUS_DATA_WIDTH          = 32,
    USER_DW                 = 16,
    USER_AW                 = 32,
    USER_MAXREQS            = 16,
    BUFFER_IMPL             = "auto"
)(
    // system signal
    input  wire                        ACLK,
    input  wire                        ARESET,
    input  wire                        ACLK_EN,
    // write address channel
    output wire [BUS_ADDR_WIDTH-1:0]   out_AXI_AWADDR,
    output wire [31:0]                 out_AXI_AWLEN,
    output wire                        out_AXI_AWVALID,
    input  wire                        in_AXI_AWREADY,
    // write data channel
    output wire [BUS_DATA_WIDTH-1:0]   out_AXI_WDATA,
    output wire [BUS_DATA_WIDTH/8-1:0] out_AXI_WSTRB,
    output wire                        out_AXI_WVALID,
    input  wire                        in_AXI_WREADY,
    // write response channel
    input  wire                        in_AXI_BVALID,
    output wire                        out_AXI_BREADY,

    // internal bus ports
    // write address
    input  wire [USER_AW-1:0]          in_HLS_AWADDR,
    input  wire [31:0]                 in_HLS_AWLEN,
    input  wire                        in_HLS_AWVALID,
    output wire                        out_HLS_AWREADY,
    // write data
    input  wire [USER_DW-1:0]          in_HLS_WDATA,
    input  wire [USER_DW/8-1:0]        in_HLS_WSTRB,
    input  wire                        in_HLS_WVALID,
    output wire                        out_HLS_WREADY,
    // write response
    output wire                        out_HLS_BVALID,
    input  wire                        in_HLS_BREADY
);

//------------------------Parameter----------------------
    localparam
        USER_DATA_WIDTH = calc_data_width(USER_DW),
        USER_DATA_BYTES = USER_DATA_WIDTH / 8,
        USER_ADDR_ALIGN = log2(USER_DATA_BYTES),
        BUS_DATA_BYTES  = BUS_DATA_WIDTH / 8,
        BUS_ADDR_ALIGN  = log2(BUS_DATA_BYTES),
        // write buffer size 
        WBUFF_DEPTH     = max(MAX_WRITE_BURST_LENGTH * BUS_DATA_WIDTH / USER_DATA_WIDTH, 1),  
        TARGET_ADDR     = C_TARGET_ADDR & (32'hffffffff << USER_ADDR_ALIGN); 

//------------------------Task and function--------------

    function integer max;
        input integer x;
        input integer y;
    begin
        max = (x > y) ? x : y;
    end
    endfunction

    function integer calc_data_width;
        input integer x;
        integer y;
    begin
        y = 8;
        while (y < x) y = y * 2;
        calc_data_width = y;
    end
    endfunction

    function integer log2;
        input integer x;
        integer n, m;
    begin
        n = 0;
        m = 1;
        while (m < x) begin
            n = n + 1;
            m = m * 2;
        end
        log2 = n;
    end
    endfunction

//------------------------Local signal-------------------

    wire                                next_wreq;
    wire                                ready_for_wreq;
    wire                                wreq_ready;

    wire [USER_AW-1 : 0]                wreq_addr;
    wire [31:0]                         wreq_len;
    wire                                wreq_valid;

    wire                                valid_length;

    reg  [USER_AW-1 : 0]                tmp_addr;
    reg  [31:0]                         tmp_len;
    reg                                 tmp_valid;

    wire                                next_wdata;
    wire                                wdata_valid;
    wire [USER_DW-1 : 0]                tmp_wdata;
    wire [USER_DW/8-1 : 0]              tmp_wstrb;

    wire                                wrsp_ready;
    wire                                wrsp_valid;
    wire                                wrsp_read;
    wire                                wrsp_type;

    wire                                ursp_ready;
    wire                                ursp_write;

    // regslice io ?  no 

//------------------------Instantiation------------------
    

    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH     (USER_AW + 32),
        .ADDR_WIDTH     (log2(USER_MAXREQS)),
        .DEPTH          (USER_MAXREQS)
    ) fifo_wreq (
        .clk            (ACLK),
        .reset          (ARESET),
        .clk_en         (ACLK_EN),
        .if_full_n      (out_HLS_AWREADY),
        .if_write       (in_HLS_AWVALID),
        .if_din         ({in_HLS_AWLEN, in_HLS_AWADDR}),
        .if_empty_n     (wreq_valid),
        .if_read        (next_wreq),
        .if_dout        ({wreq_len, wreq_addr}),
        .if_num_data_valid());

    assign next_wreq = wreq_valid && ready_for_wreq && wrsp_ready;
    assign ready_for_wreq  = ~tmp_valid || (in_AXI_AWREADY && wreq_ready);

    assign valid_length    = (wreq_len != 32'b0) && !wreq_len[31];

    assign out_AXI_AWLEN   = tmp_len;   // Byte length
    assign out_AXI_AWADDR  = tmp_addr;  // Byte address
    assign out_AXI_AWVALID = tmp_valid && wreq_ready;

    always @(posedge ACLK)
    begin
        if (ARESET) begin
            tmp_len  <= 0;
            tmp_addr <= 0;
        end
        else if (ACLK_EN) begin
            if(next_wreq) begin
                tmp_len  <= (wreq_len << USER_ADDR_ALIGN) - 1;
                tmp_addr <= TARGET_ADDR + (wreq_addr << USER_ADDR_ALIGN);
            end
        end
    end
 
    always @(posedge ACLK) 
    begin
        if (ARESET)
            tmp_valid <= 1'b0;
        else if (next_wreq && valid_length)
            tmp_valid <= 1'b1;
        else if (in_AXI_AWREADY && wreq_ready)
            tmp_valid <= 1'b0;
    end

    // ===================================================================

    

    STATE_AXI_gmem1_m_axi_fifo #(
        .MEM_STYLE         (BUFFER_IMPL),
        .DATA_WIDTH        (USER_DW + USER_DW/8),
        .ADDR_WIDTH        (log2(WBUFF_DEPTH)),
        .DEPTH             (WBUFF_DEPTH)
    ) buff_wdata (
        .clk               (ACLK),
        .reset             (ARESET),
        .clk_en            (ACLK_EN),
        .if_full_n         (out_HLS_WREADY),
        .if_write          (in_HLS_WVALID),
        .if_din            ({in_HLS_WSTRB , in_HLS_WDATA}),
        .if_empty_n        (wdata_valid),
        .if_read           (next_wdata),
        .if_dout           ({tmp_wstrb, tmp_wdata}),
        .if_num_data_valid ());

    generate
    if (USER_DATA_WIDTH == BUS_DATA_WIDTH) begin : bus_equal_gen
        assign next_wdata       = in_AXI_WREADY;
        assign out_AXI_WVALID   = wdata_valid;
        assign out_AXI_WDATA    = tmp_wdata;
        assign out_AXI_WSTRB    = tmp_wstrb;

        assign wreq_ready   = 1'b1;

    end
    else if (USER_DATA_WIDTH < BUS_DATA_WIDTH) begin : bus_wide_gen
        localparam
            TOTAL_PADS      = BUS_DATA_WIDTH / USER_DATA_WIDTH,
            PAD_ALIGN       = log2(TOTAL_PADS),
            BEAT_LEN_WIDTH  = 32 - BUS_ADDR_ALIGN;

        function [TOTAL_PADS-1 : 0] decoder;
            input [PAD_ALIGN-1 : 0] din;
            reg  [TOTAL_PADS-1 : 0] dout;
            integer i;
        begin
            dout = {TOTAL_PADS{1'b0}};
            for (i = 0; i < din; i = i + 1)
                dout[i] = 1'b1;
            decoder = dout;
        end
        endfunction

        wire [USER_AW - 1:0]        tmp_addr_end;

        wire                        offset_full_n;
        wire                        offset_write;
        wire [PAD_ALIGN-1 : 0]      start_offset;
        wire [PAD_ALIGN-1 : 0]      end_offset;
        wire [BEAT_LEN_WIDTH-1 : 0] beat_total;

        wire                        offset_valid;
        wire                        next_offset;
        wire [PAD_ALIGN-1 : 0]      head_offset;
        wire [PAD_ALIGN-1 : 0]      tail_offset;

        wire [BEAT_LEN_WIDTH-1 : 0] beat_len;
        reg  [BEAT_LEN_WIDTH-1:0]   len_cnt;

        wire [TOTAL_PADS - 1:0]     add_head;
        wire [TOTAL_PADS - 1:0]     add_tail;
        wire [TOTAL_PADS - 1:0]     pad_oh;
        reg  [TOTAL_PADS - 1:0]     pad_oh_reg;

        wire [TOTAL_PADS-1 : 0]     head_pad_sel;
        wire [0 : TOTAL_PADS-1]     tail_pad_sel; // reverse
        wire                        ready_for_data;
        wire                        next_pad;
        reg                         first_pad;
        wire                        last_pad;
        wire                        first_beat;
        wire                        last_beat;
        wire                        next_beat;

        reg  [BUS_DATA_WIDTH - 1:0] data_buf;
        reg  [BUS_DATA_BYTES - 1:0] strb_buf;
        reg                         data_valid;

        // Recording the offset of start & end address to align beats from data USER_DW < BUS_DW.
        STATE_AXI_gmem1_m_axi_fifo #(
            .DATA_WIDTH             (2*PAD_ALIGN + BEAT_LEN_WIDTH),
            .ADDR_WIDTH             (log2(NUM_WRITE_OUTSTANDING)),
            .DEPTH                  (NUM_WRITE_OUTSTANDING)
        ) wreq_offset (
            .clk                    (ACLK),
            .reset                  (ARESET),
            .clk_en                 (ACLK_EN),
            .if_full_n              (offset_full_n),
            .if_write               (offset_write),
            .if_din                 ({start_offset, end_offset, beat_total}),
            .if_empty_n             (offset_valid),
            .if_read                (next_offset),
            .if_dout                ({head_offset, tail_offset, beat_len}),
            .if_num_data_valid      ());

        assign wreq_ready   = offset_full_n | ~offset_write;
        assign tmp_addr_end = tmp_addr + tmp_len;

        assign start_offset   = tmp_addr[BUS_ADDR_ALIGN-1 : 0] >> USER_ADDR_ALIGN;
        assign end_offset     = ~tmp_addr_end[BUS_ADDR_ALIGN-1 : 0] >> USER_ADDR_ALIGN;
        assign beat_total     = (tmp_len + tmp_addr[BUS_ADDR_ALIGN-1 : 0]) >> BUS_ADDR_ALIGN;

        assign offset_write   = tmp_valid & in_AXI_AWREADY;

        assign out_AXI_WDATA  = data_buf;
        assign out_AXI_WSTRB  = strb_buf;
        assign out_AXI_WVALID = data_valid;

        assign next_wdata     = next_pad;
        assign next_offset    = last_beat && next_beat;
        assign ready_for_data = ~data_valid || in_AXI_WREADY;

        assign first_beat     = (len_cnt == 0) && offset_valid;
        assign last_beat      = (len_cnt == beat_len) && offset_valid;
        assign next_beat      = offset_valid && last_pad && ready_for_data;

        assign next_pad       = offset_valid && wdata_valid && ready_for_data;
        assign last_pad       = (last_beat) ? pad_oh[TOTAL_PADS-tail_offset-1] : pad_oh[TOTAL_PADS-1];

        assign head_pad_sel   = decoder(head_offset);
        assign tail_pad_sel   = decoder(tail_offset);

        always @(posedge ACLK)
        begin
            if (ARESET)
                len_cnt <= 0;
            else if (ACLK_EN) begin
                if (next_offset)
                    len_cnt <= 0;
                else if (next_beat)
                    len_cnt <= len_cnt + 1;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                first_pad <= 1'b1;
            else if (ACLK_EN) begin
                if (next_pad && ~last_pad)
                    first_pad <= 1'b0;
                else if (next_pad && last_pad)
                    first_pad <= 1'b1;
            end
        end 
        
        assign pad_oh = (~wdata_valid)            ? 0                :
                        (first_pad && first_beat) ? 1 << head_offset :
                        (first_pad)?                1                :
                        pad_oh_reg;

        always @(posedge ACLK)
        begin
            if (ARESET)
                pad_oh_reg <= 0;
            else if (ACLK_EN) begin
                if (next_pad)
                    pad_oh_reg <= {pad_oh[TOTAL_PADS - 2:0], 1'b0};
            end
        end

        genvar  i;
        for (i = 0; i < TOTAL_PADS; i = i + 1) begin : data_gen
            assign add_head[i] = head_pad_sel[i] && first_beat;
            assign add_tail[i] = tail_pad_sel[i] && last_beat;

            always @(posedge ACLK)
            begin
                if (ARESET)
                    data_buf[i*USER_DATA_WIDTH +: USER_DATA_WIDTH] <= {USER_DATA_WIDTH{1'b0}};
                else if (ACLK_EN) begin
                    if ((add_head[i] || add_tail[i]) && ready_for_data)
                        data_buf[i*USER_DATA_WIDTH +: USER_DATA_WIDTH] <= {USER_DATA_WIDTH{1'b0}};
                    else if (pad_oh[i] == 1'b1 && ready_for_data)
                        data_buf[i*USER_DATA_WIDTH +: USER_DATA_WIDTH] <= tmp_wdata;
                end
            end

            always @(posedge ACLK)
            begin
                if (ARESET)
                    strb_buf[i*USER_DATA_BYTES +: USER_DATA_BYTES] <= {USER_DATA_BYTES{1'b0}};
                else if (ACLK_EN) begin
                    if ((add_head[i] || add_tail[i]) && ready_for_data)
                        strb_buf[i*USER_DATA_BYTES +: USER_DATA_BYTES] <= {USER_DATA_BYTES{1'b0}};
                    else if (pad_oh[i] == 1'b1 && ready_for_data)
                        strb_buf[i*USER_DATA_BYTES +: USER_DATA_BYTES] <= tmp_wstrb;
                end
            end

        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                data_valid <= 1'b0;
            else if (ACLK_EN) begin
                if (next_beat)
                    data_valid <= 1'b1;
                else if (ready_for_data)
                    data_valid <= 1'b0;
            end
        end

    end
    else begin : bus_narrow_gen
        localparam
            TOTAL_SPLIT       = USER_DATA_WIDTH / BUS_DATA_WIDTH,
            SPLIT_ALIGN       = log2(TOTAL_SPLIT),
            BEAT_LEN_WIDTH    = 32 - BUS_ADDR_ALIGN;


        wire [USER_AW - 1:0]        tmp_addr_end;

        wire                        offset_full_n;
        wire                        offset_write;
        wire  [BEAT_LEN_WIDTH-1 : 0] beat_total;

        wire                        offset_valid;
        wire                        next_offset;

        wire [BEAT_LEN_WIDTH-1 : 0] beat_len;
        reg  [BEAT_LEN_WIDTH-1 : 0] len_cnt;

        wire                        ready_for_data;
        reg  [BUS_DATA_WIDTH - 1:0] data_buf;
        reg  [BUS_DATA_BYTES - 1:0] strb_buf;
        reg                         data_valid;

        reg [SPLIT_ALIGN-1 : 0]     split_cnt;

        wire                        first_split;
        wire                        next_split;
        wire                        last_split;

        // Recording the offset of start & end address to align beats from data USER_DW < BUS_DW.
        STATE_AXI_gmem1_m_axi_fifo #(
            .DATA_WIDTH        (BEAT_LEN_WIDTH),
            .ADDR_WIDTH        (log2(NUM_WRITE_OUTSTANDING)),
            .DEPTH             (NUM_WRITE_OUTSTANDING)
        ) wreq_offset (
            .clk               (ACLK),
            .reset             (ARESET),
            .clk_en            (ACLK_EN),
            .if_full_n         (offset_full_n),
            .if_write          (offset_write),
            .if_din            (beat_total),
            .if_empty_n        (offset_valid),
            .if_read           (next_offset),
            .if_dout           (beat_len),
            .if_num_data_valid ());

        assign wreq_ready     = offset_full_n | ~offset_write;
        assign beat_total     = (tmp_len + tmp_addr[BUS_ADDR_ALIGN-1 : 0]) >> BUS_ADDR_ALIGN;

        assign offset_write   = tmp_valid & in_AXI_AWREADY;

        assign out_AXI_WDATA  = data_buf[BUS_DATA_WIDTH - 1:0];
        assign out_AXI_WSTRB  = strb_buf[BUS_DATA_BYTES - 1:0];
        assign out_AXI_WVALID = data_valid;

        assign next_wdata     = first_split;
        assign next_offset    = (len_cnt == beat_len) && offset_valid && last_split;
        assign ready_for_data = ~data_valid | in_AXI_WREADY;

        assign first_split    = (split_cnt == 0) && wdata_valid && offset_valid && ready_for_data;
        assign last_split     = (split_cnt == (TOTAL_SPLIT - 1)) && ready_for_data;
        assign next_split     = (split_cnt != 0) && ready_for_data;
        
        always @(posedge ACLK)
        begin
            if (ARESET)
                split_cnt <= 0;
            else if (ACLK_EN) begin
                if (last_split)
                    split_cnt <= 0;
                else if (first_split || next_split)
                    split_cnt <= split_cnt + 1;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                len_cnt <= 0;
            else if (ACLK_EN) begin
                if (next_offset)
                    len_cnt <= 0;
                else if (next_wdata || next_split)
                    len_cnt <= len_cnt + 1;
            end
        end
 
        always @(posedge ACLK)
        begin
            if (ACLK_EN) begin
                if (next_wdata)
                    data_buf <= tmp_wdata;
                else if (next_split)
                    data_buf <= data_buf >> BUS_DATA_WIDTH;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                strb_buf <= 0;
            else if (ACLK_EN) begin
                if (next_wdata)
                    strb_buf <= tmp_wstrb;
                else if (next_split)
                    strb_buf <= strb_buf >> BUS_DATA_BYTES;
            end
        end

        always @(posedge ACLK)
        begin
            if (ARESET)
                data_valid <= 0;
            else if (ACLK_EN) begin
                if (next_wdata)
                    data_valid <= 1;
                else if (~(first_split || next_split) && ready_for_data)
                    data_valid <= 0;
            end
        end
    end
    endgenerate

    // ===================================================================

    // generate response for all request (including request with invalid length)
    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH        (1),
        .ADDR_WIDTH        (log2(NUM_WRITE_OUTSTANDING)),
        .DEPTH             (NUM_WRITE_OUTSTANDING)
    ) fifo_wrsp (
        .clk               (ACLK),
        .reset             (ARESET),
        .clk_en            (ACLK_EN),
        .if_full_n         (wrsp_ready),
        .if_write          (next_wreq),
        .if_din            (valid_length),
        .if_empty_n        (wrsp_valid),
        .if_read           (wrsp_read),
        .if_dout           (wrsp_type), // 1 - valid length request, 0 - invalid length request
        .if_num_data_valid ());

    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH        (1),
        .ADDR_WIDTH        (log2(USER_MAXREQS)),
        .DEPTH             (USER_MAXREQS)
    ) user_resp (
        .clk               (ACLK),
        .reset             (ARESET),
        .clk_en            (ACLK_EN),
        .if_full_n         (ursp_ready),
        .if_write          (ursp_write),
        .if_din            (1'b1),
        .if_empty_n        (out_HLS_BVALID),
        .if_read           (in_HLS_BREADY),
        .if_dout           (),
        .if_num_data_valid ());

    

    assign ursp_write  = wrsp_valid && (!wrsp_type || in_AXI_BVALID);
    assign wrsp_read   = ursp_ready && ursp_write;

    assign out_AXI_BREADY = wrsp_type && ursp_ready;

endmodule
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps

//


module STATE_AXI_gmem1_m_axi_read
#(parameter
    C_M_AXI_ID_WIDTH          = 1,
    C_M_AXI_ARUSER_WIDTH      = 1,
    C_M_AXI_RUSER_WIDTH       = 1,
    C_USER_VALUE              = 1'b0,
    C_PROT_VALUE              = 3'b000,
    C_CACHE_VALUE             = 4'b0011,
    BUS_ADDR_WIDTH            = 32,
    BUS_DATA_WIDTH            = 32,
    NUM_READ_OUTSTANDING      = 2,
    MAX_READ_BURST_LENGTH     = 16
)(
    // system signal
    input  wire                            ACLK,
    input  wire                            ARESET,
    input  wire                            ACLK_EN,
    // read address channel
    output wire [C_M_AXI_ID_WIDTH-1:0]     out_BUS_ARID,
    output wire [BUS_ADDR_WIDTH-1:0]       out_BUS_ARADDR,
    output wire [7:0]                      out_BUS_ARLEN,
    output wire [2:0]                      out_BUS_ARSIZE,
    output wire [1:0]                      out_BUS_ARBURST,
    output wire [1:0]                      out_BUS_ARLOCK,
    output wire [3:0]                      out_BUS_ARCACHE,
    output wire [2:0]                      out_BUS_ARPROT,
    output wire [3:0]                      out_BUS_ARQOS,
    output wire [3:0]                      out_BUS_ARREGION,
    output wire [C_M_AXI_ARUSER_WIDTH-1:0] out_BUS_ARUSER,
    output wire                            out_BUS_ARVALID,
    input  wire                            in_BUS_ARREADY,
    // read data channel
    input  wire [C_M_AXI_ID_WIDTH-1:0]     in_BUS_RID,
    input  wire [BUS_DATA_WIDTH-1:0]       in_BUS_RDATA,
    input  wire [1:0]                      in_BUS_RRESP,
    input  wire                            in_BUS_RLAST,
    input  wire [C_M_AXI_RUSER_WIDTH-1:0]  in_BUS_RUSER,
    input  wire                            in_BUS_RVALID,
    output wire                            out_BUS_RREADY,

    // HLS internal read request channel
    input  wire [BUS_ADDR_WIDTH-1:0]       in_HLS_ARADDR,
    input  wire [31:0]                     in_HLS_ARLEN,
    input  wire                            in_HLS_ARVALID,
    output wire                            out_HLS_ARREADY,
    output wire [BUS_DATA_WIDTH-1:0]       out_HLS_RDATA,
    output wire [1:0]                      out_HLS_RLAST,
    output wire                            out_HLS_RVALID,
    input  wire                            in_HLS_RREADY,
    input  wire                            in_HLS_RBUST_READY);

//------------------------Parameter----------------------
    localparam
        BUS_DATA_BYTES  = BUS_DATA_WIDTH / 8,
        BUS_ADDR_ALIGN  = log2(BUS_DATA_BYTES);

//------------------------Task and function--------------
    function integer log2;
        input integer x;
        integer n, m;
    begin
        n = 0;
        m = 1;
        while (m < x) begin
            n = n + 1;
            m = m * 2;
        end
        log2 = n;
    end
    endfunction

//------------------------Local signal-------------------
    // AR channel
    wire                          ost_ctrl_info;
    wire                          ost_ctrl_valid;
    wire                          ost_ctrl_ready;

    // R channel
    wire [BUS_DATA_WIDTH-1:0]     tmp_data;
    wire                          tmp_last;
    wire                          data_valid;
    wire                          data_ready;
    wire                          next_ctrl;
    wire                          need_rlast;
    wire                          burst_valid;
    wire                          last_burst;
    wire                          fifo_rctl_ready;
    wire                          next_burst;
    wire                          burst_end;

    // regslice io ?  no 

//------------------------AR channel begin---------------
//------------------------Instantiation------------------
    STATE_AXI_gmem1_m_axi_burst_converter #(
        .DATA_WIDTH        (BUS_DATA_WIDTH),
        .ADDR_WIDTH        (BUS_ADDR_WIDTH),
        .MAX_BURST_LEN     (MAX_READ_BURST_LENGTH)
    ) rreq_burst_conv (
        .clk               (ACLK),
        .reset             (ARESET),
        .clk_en            (ACLK_EN),

        .in_REQ_ADDR       (in_HLS_ARADDR),
        .in_REQ_LEN        (in_HLS_ARLEN),
        .in_REQ_VALID      (in_HLS_ARVALID),
        .out_REQ_READY     (out_HLS_ARREADY),
         
        .out_BURST_ADDR    (out_BUS_ARADDR),
        .out_BURST_LEN     (out_BUS_ARLEN),
        .out_BURST_VALID   (out_BUS_ARVALID),
        .in_BURST_READY    (in_BUS_ARREADY),

        .out_CTRL_INFO     (ost_ctrl_info),
        .out_CTRL_LEN      (),
        .out_CTRL_VALID    (ost_ctrl_valid),
        .in_CTRL_READY     (ost_ctrl_ready)
    );
    
    
//------------------------Body---------------------------

    assign out_BUS_ARID     = 0;
    assign out_BUS_ARSIZE   = BUS_ADDR_ALIGN;
    assign out_BUS_ARBURST  = 2'b01;
    assign out_BUS_ARLOCK   = 2'b00;
    assign out_BUS_ARCACHE  = C_CACHE_VALUE;
    assign out_BUS_ARPROT   = C_PROT_VALUE;
    assign out_BUS_ARUSER   = C_USER_VALUE;
    assign out_BUS_ARQOS    = 4'b0000;
    assign out_BUS_ARREGION = 4'b0000;

//------------------------AR channel end-----------------

//------------------------R channel begin----------------
//------------------------Instantiation------------------
    STATE_AXI_gmem1_m_axi_reg_slice #(
        .DATA_WIDTH     (BUS_DATA_WIDTH + 1)
    ) rs_rdata (
        .clk            (ACLK),
        .reset          (ARESET),
        .s_data         ({in_BUS_RLAST, in_BUS_RDATA}),
        .s_valid        (in_BUS_RVALID),
        .s_ready        (out_BUS_RREADY),
        .m_data         ({tmp_last, tmp_data}),
        .m_valid        (data_valid),
        .m_ready        (data_ready));

    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH     (1),
        .ADDR_WIDTH     (log2(NUM_READ_OUTSTANDING)),
        .DEPTH          (NUM_READ_OUTSTANDING)
    ) fifo_rctl (
        .clk            (ACLK),
        .reset          (ARESET),
        .clk_en         (ACLK_EN),
        .if_full_n      (ost_ctrl_ready),
        .if_write       (ost_ctrl_valid),
        .if_din         (ost_ctrl_info),
        .if_empty_n     (need_rlast),
        .if_read        (next_ctrl),
        .if_dout        (),
        .if_num_data_valid());

    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH     (1),
        .ADDR_WIDTH     (log2(NUM_READ_OUTSTANDING)),
        .DEPTH          (NUM_READ_OUTSTANDING)
    ) fifo_burst (
        .clk            (ACLK),
        .reset          (ARESET),
        .clk_en         (ACLK_EN),
        .if_full_n      (),
        .if_write       (ost_ctrl_valid),
        .if_din         (ost_ctrl_info),
        .if_empty_n     (burst_valid),
        .if_read        (next_burst),
        .if_dout        (last_burst),
        .if_num_data_valid());

//------------------------Body---------------------------
    assign next_ctrl      = in_HLS_RBUST_READY && need_rlast;
    assign next_burst     = burst_end && data_valid && data_ready;

    assign burst_end      = tmp_last === 1'b1;
    assign out_HLS_RLAST  = {burst_end, burst_end && last_burst && burst_valid};
    assign out_HLS_RDATA  = tmp_data;
    assign out_HLS_RVALID = data_valid;
    assign data_ready     = in_HLS_RREADY;
//------------------------R channel end------------------
endmodule

module STATE_AXI_gmem1_m_axi_write
#(parameter
    CONSERVATIVE              = 0,
    C_M_AXI_ID_WIDTH          = 1,
    C_M_AXI_AWUSER_WIDTH      = 1,
    C_M_AXI_WUSER_WIDTH       = 1,
    C_M_AXI_BUSER_WIDTH       = 1,
    C_USER_VALUE              = 1'b0,
    C_PROT_VALUE              = 3'b000,
    C_CACHE_VALUE             = 4'b0011,
    BUS_ADDR_WIDTH            = 32,
    BUS_DATA_WIDTH            = 32,
    NUM_WRITE_OUTSTANDING     = 2,
    MAX_WRITE_BURST_LENGTH    = 16
)(
    // system signal
    input  wire                             ACLK,
    input  wire                             ARESET,
    input  wire                             ACLK_EN,
    // write address channel
    output wire [C_M_AXI_ID_WIDTH-1:0]      out_BUS_AWID,
    output wire [2:0]                       out_BUS_AWSIZE,
    output wire [1:0]                       out_BUS_AWBURST,
    output wire [1:0]                       out_BUS_AWLOCK,
    output wire [3:0]                       out_BUS_AWCACHE,
    output wire [2:0]                       out_BUS_AWPROT,
    output wire [3:0]                       out_BUS_AWQOS,
    output wire [3:0]                       out_BUS_AWREGION,
    output wire [C_M_AXI_AWUSER_WIDTH-1:0]  out_BUS_AWUSER,
    output wire [BUS_ADDR_WIDTH-1:0]        out_BUS_AWADDR,
    output wire [7:0]                       out_BUS_AWLEN,
    output wire                             out_BUS_AWVALID,
    input  wire                             in_BUS_AWREADY,
    // write data channel
    output wire [C_M_AXI_ID_WIDTH-1:0]      out_BUS_WID,
    output wire [C_M_AXI_WUSER_WIDTH-1:0]   out_BUS_WUSER,
    output wire [BUS_DATA_WIDTH-1:0]        out_BUS_WDATA,
    output wire [BUS_DATA_WIDTH/8-1:0]      out_BUS_WSTRB,
    output wire                             out_BUS_WLAST,
    output wire                             out_BUS_WVALID,
    input  wire                             in_BUS_WREADY,
    // write response channel
    input  wire [C_M_AXI_ID_WIDTH-1:0]      in_BUS_BID,
    input  wire [1:0]                       in_BUS_BRESP,
    input  wire [C_M_AXI_BUSER_WIDTH-1:0]   in_BUS_BUSER,
    input  wire                             in_BUS_BVALID,
    output wire                             out_BUS_BREADY,
    // write request
    input  wire [BUS_ADDR_WIDTH-1:0]        in_HLS_AWADDR,
    input  wire [31:0]                      in_HLS_AWLEN,
    input  wire                             in_HLS_AWVALID,
    output wire                             out_HLS_AWREADY,

    input  wire [BUS_DATA_WIDTH-1:0]        in_HLS_WDATA,
    input  wire [BUS_DATA_WIDTH/8-1:0]      in_HLS_WSTRB,
    input  wire                             in_HLS_WVALID,
    output wire                             out_HLS_WREADY,
    output wire                             out_HLS_BVALID,
    input  wire                             in_HLS_BREADY);

//------------------------Parameter----------------------
    localparam
        BUS_DATA_BYTES  = BUS_DATA_WIDTH / 8,
        BUS_ADDR_ALIGN  = log2(BUS_DATA_BYTES);

//------------------------Task and function--------------
    function integer log2;
        input integer x;
        integer n, m;
    begin
        n = 0;
        m = 1;
        while (m < x) begin
            n = n + 1;
            m = m * 2;
        end
        log2 = n;
    end
    endfunction

//------------------------Local signal-------------------
    // AW channel
    wire [C_M_AXI_ID_WIDTH-1:0]         AWID_Dummy;
    wire [BUS_ADDR_WIDTH - 1:0]         AWADDR_Dummy;
    wire [7:0]                          AWLEN_Dummy;
    wire                                AWVALID_Dummy;
    wire                                AWREADY_Dummy;
 
    wire                                ost_ctrl_info;
    wire [7:0]                          ost_ctrl_len;
    wire                                ost_ctrl_valid;
    wire                                ost_ctrl_ready;

    // W channel
    wire                                next_data;
    wire                                data_valid;
    wire                                data_ready;
    reg  [BUS_DATA_WIDTH - 1:0]         data_buf;
    reg  [BUS_DATA_BYTES - 1:0]         strb_buf;
    wire                                ready_for_data;

    reg  [7:0]                          len_cnt;
    wire [7:0]                          burst_len;
    wire                                fifo_burst_ready;
    wire                                next_burst;
    wire                                burst_valid;
    reg                                 WVALID_Dummy;
    wire                                WREADY_Dummy;
    reg                                 WLAST_Dummy;
    //B channel
    wire                                next_resp;
    wire                                last_resp;
    wire                                need_wrsp;
    wire                                resp_valid;
    wire                                resp_ready;

    // regslice io ?  no 

//------------------------AW channel begin---------------
//------------------------Instantiation------------------
    STATE_AXI_gmem1_m_axi_burst_converter #(
        .DATA_WIDTH        (BUS_DATA_WIDTH),
        .ADDR_WIDTH        (BUS_ADDR_WIDTH),
        .MAX_BURST_LEN     (MAX_WRITE_BURST_LENGTH)
    ) wreq_burst_conv (
        .clk               (ACLK),
        .reset             (ARESET),
        .clk_en            (ACLK_EN),
        
        .in_REQ_ADDR       (in_HLS_AWADDR),
        .in_REQ_LEN        (in_HLS_AWLEN),
        .in_REQ_VALID      (in_HLS_AWVALID),
        .out_REQ_READY     (out_HLS_AWREADY),

        .out_BURST_ADDR    (AWADDR_Dummy),
        .out_BURST_LEN     (AWLEN_Dummy),
        .out_BURST_VALID   (AWVALID_Dummy),
        .in_BURST_READY    (AWREADY_Dummy),

        .out_CTRL_INFO     (ost_ctrl_info),
        .out_CTRL_LEN      (ost_ctrl_len),
        .out_CTRL_VALID    (ost_ctrl_valid),
        .in_CTRL_READY     (ost_ctrl_ready)
    );

    // burst converter
    assign out_BUS_AWID     = 0;
    assign out_BUS_AWSIZE   = BUS_ADDR_ALIGN;
    assign out_BUS_AWBURST  = 2'b01;
    assign out_BUS_AWLOCK   = 2'b00;
    assign out_BUS_AWCACHE  = C_CACHE_VALUE;
    assign out_BUS_AWPROT   = C_PROT_VALUE;
    assign out_BUS_AWUSER   = C_USER_VALUE;
    assign out_BUS_AWQOS    = 4'b0000;
    assign out_BUS_AWREGION = 4'b0000;

//------------------------AW channel end-----------------

//------------------------W channel begin----------------
//------------------------Instantiation------------------

    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH     (8),
        .ADDR_WIDTH     (log2(NUM_WRITE_OUTSTANDING)),
        .DEPTH          (NUM_WRITE_OUTSTANDING)
    ) fifo_burst (
        .clk            (ACLK),
        .reset          (ARESET),
        .clk_en         (ACLK_EN),
        .if_full_n      (),
        .if_write       (ost_ctrl_valid),
        .if_din         (ost_ctrl_len),
        .if_empty_n     (burst_valid),
        .if_read        (next_burst),
        .if_dout        (burst_len),
        .if_num_data_valid());

//------------------------Body---------------------------

    assign out_BUS_WUSER    = C_USER_VALUE;
    assign out_BUS_WID      = 0;
    assign out_HLS_WREADY   = data_ready;

    assign data_valid       = in_HLS_WVALID;
    assign data_ready       = burst_valid && ready_for_data;
    assign next_data        = data_ready && data_valid;
    assign next_burst       = (len_cnt == burst_len) && next_data;
    assign ready_for_data   = ~WVALID_Dummy || WREADY_Dummy;

    always @(posedge ACLK)
    begin
        if (ARESET) begin
            strb_buf <= 0;
            data_buf <= 0;
        end
        if (ACLK_EN) begin
            if (next_data) begin
                data_buf <= in_HLS_WDATA;
                strb_buf <= in_HLS_WSTRB;
            end
        end
    end

    always @(posedge ACLK)
    begin
        if (ARESET)
            WVALID_Dummy <= 1'b0;
        else if (ACLK_EN) begin
            if (next_data)
                WVALID_Dummy <= 1'b1;
            else if (ready_for_data)
                WVALID_Dummy <= 1'b0;
        end
    end

    always @(posedge ACLK)
    begin
        if (ARESET)
            WLAST_Dummy <= 0;
        else if (ACLK_EN) begin
            if (next_burst)
                WLAST_Dummy <= 1;
            else if (ready_for_data)
                WLAST_Dummy <= 0;
        end
    end

    always @(posedge ACLK)
    begin
        if (ARESET)
            len_cnt <= 0;
        else if (ACLK_EN) begin
            if (next_burst)
                len_cnt <= 0;
            else if (next_data)
                len_cnt <= len_cnt + 1;
        end
    end
//------------------------W channel end------------------

    // Write throttling unit
    STATE_AXI_gmem1_m_axi_throttle #(
        .CONSERVATIVE    (CONSERVATIVE),
        .USED_FIX        (0),
        .ADDR_WIDTH      (BUS_ADDR_WIDTH),
        .DATA_WIDTH      (BUS_DATA_WIDTH),
        .DEPTH           (MAX_WRITE_BURST_LENGTH),
        .MAXREQS         (NUM_WRITE_OUTSTANDING),
        .AVERAGE_MODE    (0)
    ) wreq_throttle (
        .clk             (ACLK),
        .reset           (ARESET),
        .clk_en          (ACLK_EN),
        // internal 
        .in_TOP_AWADDR   (AWADDR_Dummy),
        .in_TOP_AWLEN    (AWLEN_Dummy),
        .in_TOP_AWVALID  (AWVALID_Dummy),
        .out_TOP_AWREADY (AWREADY_Dummy),

        .in_TOP_WDATA    (data_buf),
        .in_TOP_WSTRB    (strb_buf),
        .in_TOP_WLAST    (WLAST_Dummy),
        .in_TOP_WVALID   (WVALID_Dummy),
        .out_TOP_WREADY  (WREADY_Dummy),

        // AXI BUS 
        .out_BUS_AWADDR  (out_BUS_AWADDR),
        .out_BUS_AWLEN   (out_BUS_AWLEN),
        .out_BUS_AWVALID (out_BUS_AWVALID),
        .in_BUS_AWREADY  (in_BUS_AWREADY),

        .out_BUS_WDATA   (out_BUS_WDATA),
        .out_BUS_WSTRB   (out_BUS_WSTRB),
        .out_BUS_WLAST   (out_BUS_WLAST),
        .out_BUS_WVALID  (out_BUS_WVALID),
        .in_BUS_WREADY   (in_BUS_WREADY)
    );

    
    
//------------------------B channel begin----------------
//------------------------Instantiation------------------
    STATE_AXI_gmem1_m_axi_reg_slice #(
        .DATA_WIDTH     (1)
    ) rs_resp (
        .clk            (ACLK),
        .reset          (ARESET),
        .s_data         (1'b1),
        .s_valid        (in_BUS_BVALID),
        .s_ready        (out_BUS_BREADY),
        .m_data         (),
        .m_valid        (resp_valid),
        .m_ready        (resp_ready));

    STATE_AXI_gmem1_m_axi_fifo #(
        .DATA_WIDTH     (1),
        .ADDR_WIDTH     (log2(NUM_WRITE_OUTSTANDING)),
        .DEPTH          (NUM_WRITE_OUTSTANDING)
    ) fifo_resp (
        .clk            (ACLK),
        .reset          (ARESET),
        .clk_en         (ACLK_EN),
        .if_full_n      (ost_ctrl_ready),
        .if_write       (ost_ctrl_valid),
        .if_din         (ost_ctrl_info),
        .if_empty_n     (need_wrsp),
        .if_read        (next_resp),
        .if_dout        (last_resp),
        .if_num_data_valid());
//------------------------Body---------------------------

    assign resp_ready = need_wrsp && (in_HLS_BREADY || (last_resp === 1'b0));
    assign next_resp  = resp_ready && resp_valid;

    assign out_HLS_BVALID = resp_valid && (last_resp === 1'b1 ) ;

//------------------------B channel end------------------
endmodule


module STATE_AXI_gmem1_m_axi_burst_converter
#(parameter
    DATA_WIDTH                   = 32,
    ADDR_WIDTH                   = 32,
    MAX_BURST_LEN                = 16
)(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  clk_en,

    input  wire [ADDR_WIDTH-1:0] in_REQ_ADDR,
    input  wire [31:0]           in_REQ_LEN,
    input  wire                  in_REQ_VALID,
    output wire                  out_REQ_READY,

    output wire [ADDR_WIDTH-1:0] out_BURST_ADDR,
    output wire [7:0]            out_BURST_LEN,
    output wire                  out_BURST_VALID,
    input  wire                  in_BURST_READY,

    output wire                  out_CTRL_INFO,
    output wire [7:0]            out_CTRL_LEN,
    output wire                  out_CTRL_VALID,
    input  wire                  in_CTRL_READY
);
//------------------------Parameter----------------------
    localparam
        DATA_BYTES      = DATA_WIDTH / 8,
        ADDR_ALIGN      = log2(DATA_BYTES),
        BOUNDARY_BEATS  = {12-ADDR_ALIGN{1'b1}},
        NUM_BEAT_WIDTH  = log2(MAX_BURST_LEN);
//------------------------Task and function--------------
    function integer log2;
        input integer x;
        integer n, m;
        begin
            n = 0;
            m = 1;
            while (m < x) begin
                n = n + 1;
                m = m * 2;
            end
            log2 = n;
        end
    endfunction
//------------------------Local signal-------------------
    wire [ADDR_WIDTH-1:0]       tmp_addr;
    wire [31:0]                 tmp_len;

    wire                        req_valid;
    wire                        read_req;
    wire                        next_req;

    reg  [ADDR_WIDTH - 1:0]     start_addr;
    wire [ADDR_WIDTH - 1:0]     sect_addr;
    reg  [ADDR_WIDTH - 1:0]     sect_addr_buf;
    reg                         req_handling;

    reg  [11 - ADDR_ALIGN:0]    start_to_4k;
    reg  [11 - ADDR_ALIGN:0]    end_from_4k;
    wire [11 - ADDR_ALIGN:0]    sect_len;
    reg  [11 - ADDR_ALIGN:0]    sect_len_buf;
    reg  [11 - ADDR_ALIGN:0]    beat_len;
    
    reg  [ADDR_WIDTH - 13:0]    sect_cnt;
    reg  [19:0]                 sect_total;
    reg  [19:0]                 sect_total_buf;
    wire [19:0]                 sect_total_tmp;
    wire                        ready_for_sect;

    wire                        single_sect;
    reg                         first_sect;
    reg                         last_sect;
    wire                        last_sect_tmp;
    reg                         last_sect_buf;
    wire                        next_sect;

    reg                         burst_valid;

    wire                        ost_ctrl_info;
    wire [7:0]                  ost_ctrl_len;
    wire                        ost_ctrl_valid;
//------------------------Instantiation------------------
    STATE_AXI_gmem1_m_axi_reg_slice #(
        .DATA_WIDTH     (ADDR_WIDTH + 32)
    ) rs_req (
        .clk            (clk),
        .reset          (reset),
        .s_data         ({in_REQ_LEN, in_REQ_ADDR}),
        .s_valid        (in_REQ_VALID),
        .s_ready        (out_REQ_READY),
        .m_data         ({tmp_len, tmp_addr}),
        .m_valid        (req_valid),
        .m_ready        (next_req));

//------------------------Body---------------------------
    assign read_req      = last_sect_tmp & next_sect | ~req_handling;
    assign next_req      = req_valid & read_req;

    always @(posedge clk)
    begin
        if (reset) begin
            start_addr  <= 0;
            beat_len    <= 0;
            sect_total  <= 0;
            end_from_4k <= 0;
            start_to_4k <= 0;
        end
        else if (clk_en) begin
            if (next_req) begin
                start_addr  <= {tmp_addr[ADDR_WIDTH-1:ADDR_ALIGN], {ADDR_ALIGN{1'b0}}};
                beat_len    <= (tmp_len[11:0] + tmp_addr[ADDR_ALIGN-1:0]) >> ADDR_ALIGN;
                sect_total  <= (tmp_len + tmp_addr[11:0]) >> 12;
                end_from_4k <= (tmp_addr[11:0] + tmp_len[11:0]) >> ADDR_ALIGN; 
                start_to_4k <= BOUNDARY_BEATS - tmp_addr[11:ADDR_ALIGN];
            end
        end
    end

    always @(posedge clk)
    begin
        if (reset)
            req_handling <= 1'b0;
        else if (clk_en) begin
            if (next_req)
                req_handling <= 1'b1;
            else if (~req_valid && last_sect_tmp & next_sect)
                req_handling <= 1'b0;
        end
    end

    // 4k boundary
    assign last_sect_tmp  = single_sect || last_sect;

    assign sect_total_tmp = first_sect ? sect_total : sect_total_buf;
    
    assign single_sect  = (sect_total == 0);

    assign next_sect  = req_handling && ready_for_sect;

    assign sect_addr  = (first_sect)? start_addr : {sect_cnt, {12{1'b0}}};
    
    assign sect_len   = single_sect              ? beat_len :
                        ( first_sect && ~last_sect)? start_to_4k :
                        (~first_sect &&  last_sect)? end_from_4k :
                                                     BOUNDARY_BEATS;

    always @(posedge clk)
    begin
        if (reset) begin
            first_sect <= 1'b0;
            last_sect <= 1'b0;
            sect_cnt <= 0;
        end
        else if (clk_en) begin
            if (next_req) begin
                first_sect <= 1'b1;
                last_sect <= 1'b0;
                sect_cnt <= tmp_addr[ADDR_WIDTH-1:12];
            end
            else if (next_sect) begin
                first_sect <= 1'b0;
                last_sect <= (sect_total_tmp == 1);
                sect_cnt <= sect_cnt + 1;
            end
        end
    end

    always @(posedge clk)
    begin
        if (reset) begin
            sect_addr_buf  <= 0;
            sect_len_buf   <= 0;
            last_sect_buf  <= 1'b0;
            sect_total_buf <= 0;
        end
        else if (clk_en) begin
            if (next_sect) begin
                sect_addr_buf  <= sect_addr;
                sect_len_buf   <= sect_len;
                last_sect_buf  <= last_sect_tmp;
                sect_total_buf <= sect_total_tmp - 1;
            end
        end
    end

    generate
    if (DATA_BYTES >= 4096/MAX_BURST_LEN) begin : must_one_burst
        assign out_BURST_ADDR  = sect_addr_buf;
        assign out_BURST_LEN   = sect_len_buf;
        assign out_BURST_VALID = burst_valid;

        assign out_CTRL_VALID  = next_sect;
        assign out_CTRL_INFO   = last_sect_tmp;
        assign out_CTRL_LEN    = sect_len;

        assign ready_for_sect = ~(burst_valid && ~in_BURST_READY) && in_CTRL_READY;

        always @(posedge clk)
        begin
            if (reset)
                burst_valid <= 1'b0;
            else if (clk_en) begin
                if (next_sect)
                    burst_valid <= 1'b1;
                else if (in_BURST_READY)
                    burst_valid <= 1'b0;
            end
        end

    end
    else begin : could_multi_bursts
        wire [ADDR_WIDTH - 1:0]                   addr_tmp;
        reg  [ADDR_WIDTH - 1:0]                   addr_buf;
        reg  [ADDR_ALIGN + 8:0]                   addr_step;
        wire [7:0]                                len_tmp;
        reg  [7:0]                                len_buf;
        reg                                       sect_handling;
        reg  [11 - NUM_BEAT_WIDTH - ADDR_ALIGN:0] loop_cnt;
        reg                                       first_loop;
        reg                                       last_loop;
        wire                                      next_loop;
        wire                                      ready_for_loop;

        assign out_BURST_ADDR  = addr_buf;
        assign out_BURST_LEN   = len_buf;
        assign out_BURST_VALID = burst_valid;

        assign out_CTRL_VALID  = next_loop;
        assign out_CTRL_INFO   = last_loop && last_sect_buf;
        assign out_CTRL_LEN    = len_tmp;

        assign next_loop       = sect_handling && ready_for_loop;
        assign ready_for_sect  = ~sect_handling || (last_loop && next_loop);
        assign ready_for_loop  = ~(burst_valid && ~in_BURST_READY) && in_CTRL_READY;

        always @(posedge clk)
        begin
            if (reset)
                burst_valid <= 1'b0;
            else if (clk_en) begin
                if (next_loop)
                    burst_valid <= 1'b1;
                else if (in_BURST_READY)
                    burst_valid <= 1'b0;
            end
        end

        always @(posedge clk)
        begin
            if (reset)
                sect_handling <= 1'b0;
            else if (clk_en) begin
                if (req_handling && ~sect_handling)
                    sect_handling <= 1'b1;
                else if (~req_handling && last_loop && next_loop)
                    sect_handling <= 1'b0;
            end
        end

        always @(posedge clk)
        begin
            if (reset) begin
                first_loop <= 1'b0;
                last_loop <= 1'b0;
                loop_cnt <= 0;
            end
            else if (clk_en) begin
                if (next_sect) begin
                    first_loop <= 1'b1;
                    last_loop <= (sect_len[11 - ADDR_ALIGN : NUM_BEAT_WIDTH] == 0);
                    loop_cnt <= sect_len[11 - ADDR_ALIGN : NUM_BEAT_WIDTH];
                end
                else if (next_loop) begin
                    first_loop <= 1'b0;
                    last_loop <= (loop_cnt == 1);
                    loop_cnt <= loop_cnt - 1;
                end
            end
        end

        assign addr_tmp = first_loop ? sect_addr_buf : (addr_buf + addr_step);
        assign len_tmp  = (NUM_BEAT_WIDTH == 0) ? 0 :
                          last_loop ? sect_len_buf[NUM_BEAT_WIDTH - 1:0] : 
                                      { NUM_BEAT_WIDTH{1'b1} };
        always @(posedge clk)
        begin
            if (reset) begin
                addr_buf  <= 0;
                addr_step <= 0;
                len_buf   <= 0;
            end
            else if (clk_en) begin
                if (next_loop) begin
                    addr_buf  <= addr_tmp;
                    addr_step <= (len_tmp + 1) << ADDR_ALIGN;
                    len_buf   <= len_tmp;
                end
            end
        end

    end
    endgenerate

endmodule

module STATE_AXI_gmem1_m_axi_throttle
#(parameter
    CONSERVATIVE   = 0,
    USED_FIX       = 0,
    FIX_VALUE      = 4,
    ADDR_WIDTH     = 32,
    DATA_WIDTH     = 32,
    DEPTH          = 16,
    MAXREQS        = 16,
    AVERAGE_MODE   = 0 
)(
    input  wire                      clk,
    input  wire                      reset,
    input  wire                      clk_en,

    input  wire [ADDR_WIDTH-1:0]     in_TOP_AWADDR,
    input  wire [7:0]                in_TOP_AWLEN,
    input  wire                      in_TOP_AWVALID,
    output wire                      out_TOP_AWREADY,
    input  wire [DATA_WIDTH-1:0]     in_TOP_WDATA,
    input  wire [DATA_WIDTH/8-1:0]   in_TOP_WSTRB,
    input  wire                      in_TOP_WLAST,
    input  wire                      in_TOP_WVALID,
    output wire                      out_TOP_WREADY,

    output wire [ADDR_WIDTH-1:0]     out_BUS_AWADDR,
    output wire [7:0]                out_BUS_AWLEN,
    output wire                      out_BUS_AWVALID,
    input  wire                      in_BUS_AWREADY,
    output wire [DATA_WIDTH-1:0]     out_BUS_WDATA,
    output wire [DATA_WIDTH/8-1:0]   out_BUS_WSTRB,
    output wire                      out_BUS_WLAST,
    output wire                      out_BUS_WVALID,
    input  wire                      in_BUS_WREADY);

    function integer log2;
        input integer x;
        integer n, m;
    begin
        n = 0;
        m = 1;
        while (m < x) begin
            n = n + 1;
            m = m * 2;
        end
        log2 = n;
    end
    endfunction
// aggressive mode
    generate
    if (CONSERVATIVE == 0) begin
        localparam threshold = (USED_FIX)? FIX_VALUE-1 : 0;

        wire                req_en;
        wire                handshake;
        wire  [7:0]         load_init;
        reg   [8:0]         throttl_cnt;

        // AW Channel
        assign out_BUS_AWADDR = in_TOP_AWADDR;
        assign out_BUS_AWLEN  = in_TOP_AWLEN;

        // W Channel
        assign out_BUS_WDATA  = in_TOP_WDATA;
        assign out_BUS_WSTRB  = in_TOP_WSTRB;
        assign out_BUS_WLAST  = in_TOP_WLAST;
        assign out_BUS_WVALID = in_TOP_WVALID & (throttl_cnt > 0);
        assign out_TOP_WREADY = in_BUS_WREADY & (throttl_cnt > 0);

        if (USED_FIX) begin
            assign load_init = FIX_VALUE-1;
            assign handshake = 1'b1;
        end else if (AVERAGE_MODE) begin
            assign load_init = in_TOP_AWLEN;
            assign handshake = 1'b1;
        end else begin
            assign load_init = in_TOP_AWLEN;
            assign handshake = out_BUS_WVALID & in_BUS_WREADY;
        end

        assign out_BUS_AWVALID = in_TOP_AWVALID & req_en;
        assign out_TOP_AWREADY = in_BUS_AWREADY & req_en;
        assign req_en = (throttl_cnt == 0) | (throttl_cnt == 1 & handshake);

        always @(posedge clk)
        begin
            if (reset)
                throttl_cnt <= 0;
            else if (clk_en) begin
                if (in_TOP_AWLEN >= threshold && req_en && in_TOP_AWVALID && in_BUS_AWREADY)
                    throttl_cnt <= load_init + 1'b1; //load
                else if (throttl_cnt > 0 && handshake)
                    throttl_cnt <= throttl_cnt - 1'b1;
            end
        end

    end
// conservative mode
    else begin
        localparam CNT_WIDTH = ((DEPTH < 4)? 2 : log2(DEPTH)) + 1;

        // Instantiation for reg slice for AW channel
        wire                        rs_req_ready;
        wire                        rs_req_valid;
        wire [ADDR_WIDTH + 7 : 0]   rs_req_in;
        wire [ADDR_WIDTH + 7 : 0]   rs_req_out;

        STATE_AXI_gmem1_m_axi_reg_slice #(
            .DATA_WIDTH     (ADDR_WIDTH + 8)
        ) rs_req (
            .clk            (clk),
            .reset          (reset),
            .s_data         (rs_req_in),
            .s_valid        (rs_req_valid),
            .s_ready        (rs_req_ready),
            .m_data         (rs_req_out),
            .m_valid        (out_BUS_AWVALID),
            .m_ready        (in_BUS_AWREADY));

        wire  [DATA_WIDTH + DATA_WIDTH/8 : 0]   data_in;
        wire  [DATA_WIDTH + DATA_WIDTH/8 : 0]   data_out;
        wire  [ADDR_WIDTH + 7 : 0]              req_in;
        reg                                     req_en;
        wire                                    data_en;
        wire                                    fifo_valid;
        wire                                    read_fifo;
        wire                                    req_fifo_valid;
        wire                                    read_req;
        wire                                    data_push;
        wire                                    data_pop;
        reg                                     flying_req;
        reg   [CNT_WIDTH-1 : 0]                 last_cnt;

        //AW Channel
        assign req_in   = {in_TOP_AWLEN, in_TOP_AWADDR};
        assign out_BUS_AWADDR = rs_req_out[ADDR_WIDTH-1 : 0];
        assign out_BUS_AWLEN  = rs_req_out[ADDR_WIDTH+7 : ADDR_WIDTH];
        assign rs_req_valid = req_fifo_valid & req_en;

        assign read_req      = rs_req_ready & req_en;

        always @(*)
        begin
            if (~flying_req & data_en)
                req_en <= 1;
            else if (flying_req & (out_BUS_WLAST & data_pop) & (last_cnt[CNT_WIDTH-1:1] != 0))
                req_en <= 1;
            else
                req_en <= 0;
        end

        always @(posedge clk)
        begin
            if (reset)
                flying_req <= 0;
            else if (clk_en) begin
                if (rs_req_valid & rs_req_ready)
                    flying_req <= 1;
                else if (out_BUS_WLAST & data_pop)
                    flying_req <= 0;
            end
        end

        STATE_AXI_gmem1_m_axi_fifo #(
            .DATA_WIDTH     (ADDR_WIDTH + 8),
            .ADDR_WIDTH     (log2(MAXREQS)),
            .DEPTH          (MAXREQS)
        ) req_fifo (
            .clk            (clk),
            .reset          (reset),
            .clk_en         (clk_en),
            .if_full_n      (out_TOP_AWREADY),
            .if_write       (in_TOP_AWVALID),
            .if_din         (req_in),
            .if_empty_n     (req_fifo_valid),
            .if_read        (read_req),
            .if_dout        (rs_req_in),
            .if_num_data_valid());

        //W Channel
        assign data_in  = {in_TOP_WLAST, in_TOP_WSTRB, in_TOP_WDATA};
        assign out_BUS_WDATA = data_out[DATA_WIDTH-1 : 0];
        assign out_BUS_WSTRB = data_out[DATA_WIDTH+DATA_WIDTH/8-1 : DATA_WIDTH];
        assign out_BUS_WLAST = data_out[DATA_WIDTH+DATA_WIDTH/8];
        assign out_BUS_WVALID = fifo_valid & data_en & flying_req;

        assign data_en   = last_cnt != 0;
        assign data_push = in_TOP_WVALID & out_TOP_WREADY;
        assign data_pop  = fifo_valid & read_fifo;
        assign read_fifo = in_BUS_WREADY & data_en & flying_req;

        always @(posedge clk)
        begin
            if (reset)
                last_cnt <= 0;
            else if (clk_en) begin
                if ((in_TOP_WLAST & data_push) && ~(out_BUS_WLAST & data_pop))
                    last_cnt <= last_cnt + 1;
                else if (~(in_TOP_WLAST & data_push) && (out_BUS_WLAST & data_pop))
                    last_cnt <= last_cnt - 1;
            end
        end
            
        STATE_AXI_gmem1_m_axi_fifo #(
            .DATA_WIDTH     (DATA_WIDTH + DATA_WIDTH/8 + 1),
            .ADDR_WIDTH     (log2(DEPTH)),
            .DEPTH          (DEPTH)
        ) data_fifo (
            .clk            (clk),
            .reset          (reset),
            .clk_en         (clk_en),
            .if_full_n      (out_TOP_WREADY),
            .if_write       (in_TOP_WVALID),
            .if_din         (data_in),
            .if_empty_n     (fifo_valid),
            .if_read        (read_fifo),
            .if_dout        (data_out),
            .if_num_data_valid());

        end
    endgenerate

endmodule



module STATE_AXI_gmem1_m_axi_reg_slice
#(parameter
    DATA_WIDTH = 8
) (
    // system signals
    input  wire                  clk,
    input  wire                  reset,
    // slave side
    input  wire [DATA_WIDTH-1:0] s_data,
    input  wire                  s_valid,
    output wire                  s_ready,
    // master side
    output wire [DATA_WIDTH-1:0] m_data,
    output wire                  m_valid,
    input  wire                  m_ready);
    //------------------------Parameter----------------------
    // state
    localparam [1:0]
        ZERO = 2'b10,
        ONE  = 2'b11,
        TWO  = 2'b01;
    //------------------------Local signal-------------------
    reg  [DATA_WIDTH-1:0] data_p1;
    reg  [DATA_WIDTH-1:0] data_p2;
    wire         load_p1;
    wire         load_p2;
    wire         load_p1_from_p2;
    reg          s_ready_t;
    reg  [1:0]   state;
    reg  [1:0]   next;
    //------------------------Body---------------------------
    assign s_ready = s_ready_t;
    assign m_data  = data_p1;
    assign m_valid = state[0];

    assign load_p1 = (state == ZERO && s_valid) ||
                    (state == ONE && s_valid && m_ready) ||
                    (state == TWO && m_ready);
    assign load_p2 = s_valid & s_ready;
    assign load_p1_from_p2 = (state == TWO);

    // data_p1
    always @(posedge clk) begin
        if (load_p1) begin
            if (load_p1_from_p2)
                data_p1 <= data_p2;
            else
                data_p1 <= s_data;
        end
    end

    // data_p2
    always @(posedge clk) begin
        if (load_p2) data_p2 <= s_data;
    end

    // s_ready_t
    always @(posedge clk) begin
        if (reset)
            s_ready_t <= 1'b0;
        else if (state == ZERO)
            s_ready_t <= 1'b1;
        else if (state == ONE && next == TWO)
            s_ready_t <= 1'b0;
        else if (state == TWO && next == ONE)
            s_ready_t <= 1'b1;
    end

    // state
    always @(posedge clk) begin
        if (reset)
            state <= ZERO;
        else
            state <= next;
    end

    // next
    always @(*) begin
        case (state)
            ZERO:
                if (s_valid & s_ready)
                    next = ONE;
                else
                    next = ZERO;
            ONE:
                if (~s_valid & m_ready)
                    next = ZERO;
                else if (s_valid & ~m_ready)
                    next = TWO;
                else
                    next = ONE;
            TWO:
                if (m_ready)
                    next = ONE;
                else
                    next = TWO;
            default:
                next = ZERO;
        endcase
    end
endmodule

module STATE_AXI_gmem1_m_axi_fifo
#(parameter
    MEM_STYLE   = "shiftreg",
    DATA_WIDTH = 32,
    ADDR_WIDTH = 5,
    DEPTH      = 32
) (
    // system signal
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  clk_en,

    // write
    output wire                  if_full_n,
    input  wire                  if_write,
    input  wire [DATA_WIDTH-1:0] if_din,

    // read
    output wire                  if_empty_n,
    input  wire                  if_read,
    output wire [DATA_WIDTH-1:0] if_dout,
    output wire [ADDR_WIDTH:0]   if_num_data_valid);

//------------------------Local signal-------------------

    wire                  push;
    wire                  pop;
    reg                   full_n = 1'b1;
    reg                   empty_n = 1'b0;
    reg                   dout_vld = 1'b0;
    reg  [ADDR_WIDTH:0]   mOutPtr = 1'b0;

//------------------------Instantiation------------------
    generate 
    if ((MEM_STYLE == "shiftreg") || (DEPTH == 1)) begin
        reg  [ADDR_WIDTH-1:0] raddr = 1'b0;

        STATE_AXI_gmem1_m_axi_srl
        #(  .DATA_WIDTH     (DATA_WIDTH),
            .ADDR_WIDTH     (ADDR_WIDTH),
            .DEPTH          (DEPTH))
        U_fifo_srl(
            .clk            (clk),
            .reset          (reset),
            .clk_en         (clk_en),
            .we             (push),
            .din            (if_din),
            .raddr          (raddr),
            .re             (pop),
            .dout           (if_dout)
        );

        // raddr
        always @(posedge clk) begin
            if (reset == 1'b1)
                raddr <= 1'b0;
            else if (clk_en) begin
                if (push & ~pop & empty_n)
                    raddr <= raddr + 1'b1;
                else if (~push & pop && raddr != 0)
                    raddr <= raddr - 1'b1;
            end
        end

    end else begin
        reg  [ADDR_WIDTH-1:0] waddr = 1'b0;
        reg  [ADDR_WIDTH-1:0] raddr = 1'b0;
        wire [ADDR_WIDTH-1:0] wnext;
        wire [ADDR_WIDTH-1:0] rnext;

        STATE_AXI_gmem1_m_axi_mem
        #(  .MEM_STYLE      (MEM_STYLE),
            .DATA_WIDTH     (DATA_WIDTH),
            .ADDR_WIDTH     (ADDR_WIDTH),
            .DEPTH          (DEPTH))
        U_fifo_mem(
            .clk            (clk),
            .reset          (reset),
            .clk_en         (clk_en),
            .we             (push),
            .waddr          (waddr),
            .din            (if_din),
            .raddr          (rnext),
            .re             (pop),
            .dout           (if_dout)
        );

        assign wnext =  !push                ? waddr :
                        (waddr == DEPTH - 2) ? 1'b0  :
                        waddr + 1'b1;
        assign rnext =  !pop                 ? raddr :
                        (raddr == DEPTH - 2) ? 1'b0  :
                        raddr + 1'b1;

        // waddr
        always @(posedge clk) begin
            if (reset == 1'b1)
                waddr <= 1'b0;
            else if (clk_en)
                waddr <= wnext;
        end

        // raddr
        always @(posedge clk) begin
            if (reset == 1'b1)
                raddr <= 1'b0;
            else if (clk_en)
                raddr <= rnext;
        end
    end
    endgenerate

//------------------------Body---------------------------
    assign if_num_data_valid = dout_vld ? mOutPtr + 1'b1 : 'b0;

    generate if (DEPTH == 1) begin
        assign if_full_n  = !dout_vld;
        assign if_empty_n = dout_vld;
        assign push = !dout_vld & if_write;
        assign pop  = !dout_vld & if_write;
    
    end else begin

        assign if_full_n  = full_n;
        assign if_empty_n = dout_vld;
        assign push = full_n & if_write;
        assign pop  = empty_n & (if_read | ~dout_vld);

        // mOutPtr
        always @(posedge clk) begin
            if (reset == 1'b1)
                mOutPtr <= 'b0;
            else if (clk_en)
                if (push & ~pop)
                    mOutPtr <= mOutPtr + 1'b1;
                else if (~push & pop)
                    mOutPtr <= mOutPtr - 1'b1;
        end

        // full_n
        always @(posedge clk) begin
            if (reset == 1'b1)
                full_n <= 1'b1;
            else if (clk_en)
                if (push & ~pop)
                    full_n <= (mOutPtr != DEPTH - 2);
                else if (~push & pop)
                    full_n <= 1'b1;
        end

        // empty_n
        always @(posedge clk)
        begin
            if (reset)
                empty_n <= 1'b0;
            else if (clk_en) begin
                if (push & ~pop)
                    empty_n <= 1'b1;
                else if (~push & pop)
                    empty_n <= (mOutPtr != 1'b1);
            end
        end
    end
    endgenerate

    // dout_vld
    always @(posedge clk) begin
        if (reset == 1'b1)
            dout_vld <= 1'b0;
        else if (clk_en)
            if (pop)
                dout_vld <= 1'b1;
            else if (if_read)
                dout_vld <= 1'b0;
    end

endmodule

module STATE_AXI_gmem1_m_axi_srl
#(parameter
        DATA_WIDTH  = 32,
        ADDR_WIDTH  = 6,
        DEPTH       = 63
    )(
        input  wire                  clk,
        input  wire                  reset,
        input  wire                  clk_en,
        input  wire                  we,
        input  wire [DATA_WIDTH-1:0] din,
        input  wire [ADDR_WIDTH-1:0] raddr,
        input  wire                  re,
        output reg  [DATA_WIDTH-1:0] dout
    );

    generate
    if (DEPTH > 1) begin
        reg  [DATA_WIDTH-1:0] mem[0:DEPTH-2];

        integer i;
        always @(posedge clk)
        begin
            if (clk_en & we) begin
                for (i = 0; i < DEPTH - 2; i = i + 1) begin
                    mem[i+1] <= mem[i];
                end
                mem[0] <= din;
            end
        end

        always @(posedge clk)
        begin
            if (reset)
                dout <= 0;
            else if (clk_en & re) begin
                dout <= mem[raddr];
            end
        end
    end
    else begin
        always @(posedge clk)
        begin
            if (reset)
                dout <= 0;
            else if (clk_en & we) begin
                dout <= din;
            end
        end
    end
    endgenerate

endmodule

module STATE_AXI_gmem1_m_axi_mem
#(parameter
    MEM_STYLE   = "auto",
    DATA_WIDTH  = 32,
    ADDR_WIDTH  = 6,
    DEPTH       = 63
)(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  clk_en,
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [DATA_WIDTH-1:0] din,
    input  wire [ADDR_WIDTH-1:0] raddr,
    input  wire                  re,
    output reg  [DATA_WIDTH-1:0] dout);

    (* ram_style = MEM_STYLE, rw_addr_collision = "yes" *)
    reg  [DATA_WIDTH-1:0] mem[0:DEPTH-2];
    reg  [ADDR_WIDTH-1:0] raddr_reg;

    //write to ram
    always @(posedge clk) begin
        if (clk_en & we)
            mem[waddr] <= din;
    end

    //buffer the raddr
    always @(posedge clk) begin
        if (clk_en)
            raddr_reg <= raddr;
    end

    //read from ram
    always @(posedge clk) begin
        if (reset)
            dout <= 0;
        else if (clk_en & re)
            dout <= mem[raddr_reg];
    end
endmodule
