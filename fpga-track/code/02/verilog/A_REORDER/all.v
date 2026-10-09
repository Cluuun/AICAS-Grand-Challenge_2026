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

module A_REORDER_regslice_both
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

module A_REORDER_regslice_both_w1
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

module A_REORDER_run_vit_store (
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
        memory_vit_a,
        rv_aq_stream_TDATA,
        rv_aq_stream_TVALID,
        rv_aq_stream_TREADY,
        rv_as_stream_TDATA,
        rv_as_stream_TVALID,
        rv_as_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 8'd1;
parameter    ap_ST_fsm_state2 = 8'd2;
parameter    ap_ST_fsm_state3 = 8'd4;
parameter    ap_ST_fsm_state4 = 8'd8;
parameter    ap_ST_fsm_state5 = 8'd16;
parameter    ap_ST_fsm_state6 = 8'd32;
parameter    ap_ST_fsm_state7 = 8'd64;
parameter    ap_ST_fsm_state8 = 8'd128;

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
input  [2:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [63:0] memory_vit_a;
input  [63:0] rv_aq_stream_TDATA;
input   rv_aq_stream_TVALID;
output   rv_aq_stream_TREADY;
input  [7:0] rv_as_stream_TDATA;
input   rv_as_stream_TVALID;
output   rv_as_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg m_axi_gmem1_AWVALID;
reg[63:0] m_axi_gmem1_AWADDR;
reg[0:0] m_axi_gmem1_AWID;
reg[31:0] m_axi_gmem1_AWLEN;
reg[2:0] m_axi_gmem1_AWSIZE;
reg[1:0] m_axi_gmem1_AWBURST;
reg[1:0] m_axi_gmem1_AWLOCK;
reg[3:0] m_axi_gmem1_AWCACHE;
reg[2:0] m_axi_gmem1_AWPROT;
reg[3:0] m_axi_gmem1_AWQOS;
reg[3:0] m_axi_gmem1_AWREGION;
reg[0:0] m_axi_gmem1_AWUSER;
reg m_axi_gmem1_WVALID;
reg m_axi_gmem1_BREADY;
reg rv_aq_stream_TREADY;
reg rv_as_stream_TREADY;

(* fsm_encoding = "none" *) reg   [7:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    gmem1_blk_n_AW;
reg    gmem1_blk_n_B;
wire    ap_CS_fsm_state8;
wire  signed [59:0] trunc_ln_fu_71_p4;
reg   [59:0] trunc_ln_reg_92;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_done;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_idle;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_ready;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWVALID;
wire   [63:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWADDR;
wire   [0:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWID;
wire   [31:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWLEN;
wire   [2:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWBURST;
wire   [1:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWPROT;
wire   [3:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWQOS;
wire   [3:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWREGION;
wire   [0:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWUSER;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WVALID;
wire   [127:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WDATA;
wire   [15:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WSTRB;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WLAST;
wire   [0:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WID;
wire   [0:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WUSER;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARVALID;
wire   [63:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARADDR;
wire   [0:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARID;
wire   [31:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARLEN;
wire   [2:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARBURST;
wire   [1:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARPROT;
wire   [3:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARQOS;
wire   [3:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARREGION;
wire   [0:0] grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARUSER;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_RREADY;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_BREADY;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_rv_aq_stream_TREADY;
wire    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_rv_as_stream_TREADY;
reg    grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start_reg;
wire    ap_CS_fsm_state2;
wire    ap_CS_fsm_state3;
wire  signed [63:0] sext_ln64_fu_81_p1;
reg   [7:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
reg    ap_ST_fsm_state8_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_fsm = 8'd1;
//#0 grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start_reg = 1'b0;
end

A_REORDER_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3 grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start),
    .ap_done(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_done),
    .ap_idle(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_idle),
    .ap_ready(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_ready),
    .rv_aq_stream_TVALID(rv_aq_stream_TVALID),
    .rv_as_stream_TVALID(rv_as_stream_TVALID),
    .m_axi_gmem1_AWVALID(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(m_axi_gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(m_axi_gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(3'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(m_axi_gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(m_axi_gmem1_BRESP),
    .m_axi_gmem1_BID(m_axi_gmem1_BID),
    .m_axi_gmem1_BUSER(m_axi_gmem1_BUSER),
    .sext_ln64(trunc_ln_reg_92),
    .rv_aq_stream_TDATA(rv_aq_stream_TDATA),
    .rv_aq_stream_TREADY(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_rv_aq_stream_TREADY),
    .rv_as_stream_TDATA(rv_as_stream_TDATA),
    .rv_as_stream_TREADY(grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_rv_as_stream_TREADY)
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
        grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start_reg <= 1'b1;
        end else if ((grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_ready == 1'b1)) begin
            grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        trunc_ln_reg_92 <= {{memory_vit_a[63:4]}};
    end
end

always @ (*) begin
    if (((m_axi_gmem1_AWREADY == 1'b0) | (ap_start == 1'b0))) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

always @ (*) begin
    if ((grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

assign ap_ST_fsm_state7_blk = 1'b0;

always @ (*) begin
    if ((m_axi_gmem1_BVALID == 1'b0)) begin
        ap_ST_fsm_state8_blk = 1'b1;
    end else begin
        ap_ST_fsm_state8_blk = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)) | ((m_axi_gmem1_BVALID == 1'b1) & (1'b1 == ap_CS_fsm_state8)))) begin
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
    if (((m_axi_gmem1_BVALID == 1'b1) & (1'b1 == ap_CS_fsm_state8))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        gmem1_blk_n_AW = m_axi_gmem1_AWREADY;
    end else begin
        gmem1_blk_n_AW = 1'b1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        gmem1_blk_n_B = m_axi_gmem1_BVALID;
    end else begin
        gmem1_blk_n_B = 1'b1;
    end
end

always @ (*) begin
    if ((~((m_axi_gmem1_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        m_axi_gmem1_AWADDR = sext_ln64_fu_81_p1;
    end else if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWADDR = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWADDR;
    end else begin
        m_axi_gmem1_AWADDR = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWBURST = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWBURST;
    end else begin
        m_axi_gmem1_AWBURST = 2'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWCACHE = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWCACHE;
    end else begin
        m_axi_gmem1_AWCACHE = 4'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWID = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWID;
    end else begin
        m_axi_gmem1_AWID = 1'd0;
    end
end

always @ (*) begin
    if ((~((m_axi_gmem1_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        m_axi_gmem1_AWLEN = 32'd98304;
    end else if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWLEN = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWLEN;
    end else begin
        m_axi_gmem1_AWLEN = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWLOCK = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWLOCK;
    end else begin
        m_axi_gmem1_AWLOCK = 2'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWPROT = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWPROT;
    end else begin
        m_axi_gmem1_AWPROT = 3'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWQOS = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWQOS;
    end else begin
        m_axi_gmem1_AWQOS = 4'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWREGION = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWREGION;
    end else begin
        m_axi_gmem1_AWREGION = 4'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWSIZE = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWSIZE;
    end else begin
        m_axi_gmem1_AWSIZE = 3'd0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWUSER = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWUSER;
    end else begin
        m_axi_gmem1_AWUSER = 1'd0;
    end
end

always @ (*) begin
    if ((~((m_axi_gmem1_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        m_axi_gmem1_AWVALID = 1'b1;
    end else if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_AWVALID = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_AWVALID;
    end else begin
        m_axi_gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((m_axi_gmem1_BVALID == 1'b1) & (1'b1 == ap_CS_fsm_state8))) begin
        m_axi_gmem1_BREADY = 1'b1;
    end else if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_BREADY = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_BREADY;
    end else begin
        m_axi_gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2))) begin
        m_axi_gmem1_WVALID = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WVALID;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        rv_aq_stream_TREADY = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_rv_aq_stream_TREADY;
    end else begin
        rv_aq_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        rv_as_stream_TREADY = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_rv_as_stream_TREADY;
    end else begin
        rv_as_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if ((~((m_axi_gmem1_AWREADY == 1'b0) | (ap_start == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
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
            if (((m_axi_gmem1_BVALID == 1'b1) & (1'b1 == ap_CS_fsm_state8))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
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

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_ap_start_reg;

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

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WDATA = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WDATA;

assign m_axi_gmem1_WID = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WID;

assign m_axi_gmem1_WLAST = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WLAST;

assign m_axi_gmem1_WSTRB = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WSTRB;

assign m_axi_gmem1_WUSER = grp_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3_fu_60_m_axi_gmem1_WUSER;

assign sext_ln64_fu_81_p1 = trunc_ln_fu_71_p4;

assign trunc_ln_fu_71_p4 = {{memory_vit_a[63:4]}};

endmodule //A_REORDER_run_vit_store
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="A_REORDER_A_REORDER,hls_ip_2023_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xczu5ev-fbvb900-1L-i,HLS_INPUT_CLOCK=2.500000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=2.429188,HLS_SYN_LAT=196635,HLS_SYN_TPT=none,HLS_SYN_MEM=16,HLS_SYN_DSP=0,HLS_SYN_FF=1326,HLS_SYN_LUT=1983,HLS_VERSION=2023_2}" *)

module A_REORDER (
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
        memory_vit_a,
        rv_aq_stream_TDATA,
        rv_aq_stream_TVALID,
        rv_aq_stream_TREADY,
        rv_as_stream_TDATA,
        rv_as_stream_TVALID,
        rv_as_stream_TREADY,
        mux_aq_stream_TDATA,
        mux_aq_stream_TVALID,
        mux_aq_stream_TREADY,
        mux_as_stream_TDATA,
        mux_as_stream_TVALID,
        mux_as_stream_TREADY
);

parameter    ap_ST_fsm_state1 = 6'd1;
parameter    ap_ST_fsm_state2 = 6'd2;
parameter    ap_ST_fsm_state3 = 6'd4;
parameter    ap_ST_fsm_state4 = 6'd8;
parameter    ap_ST_fsm_state5 = 6'd16;
parameter    ap_ST_fsm_state6 = 6'd32;
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
input  [63:0] memory_vit_a;
input  [63:0] rv_aq_stream_TDATA;
input   rv_aq_stream_TVALID;
output   rv_aq_stream_TREADY;
input  [7:0] rv_as_stream_TDATA;
input   rv_as_stream_TVALID;
output   rv_as_stream_TREADY;
output  [63:0] mux_aq_stream_TDATA;
output   mux_aq_stream_TVALID;
input   mux_aq_stream_TREADY;
output  [7:0] mux_as_stream_TDATA;
output   mux_as_stream_TVALID;
input   mux_as_stream_TREADY;

reg ap_done;
reg ap_idle;
reg ap_ready;

 reg    ap_rst_n_inv;
reg    ap_done_reg;
(* fsm_encoding = "none" *) reg   [5:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire    ap_CS_fsm_state2;
wire    grp_run_vit_store_fu_60_ap_start;
wire    grp_run_vit_store_fu_60_ap_done;
wire    grp_run_vit_store_fu_60_ap_idle;
wire    grp_run_vit_store_fu_60_ap_ready;
wire    grp_run_vit_store_fu_60_m_axi_gmem1_AWVALID;
wire   [63:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWADDR;
wire   [0:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWID;
wire   [31:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWLEN;
wire   [2:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWBURST;
wire   [1:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWPROT;
wire   [3:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWQOS;
wire   [3:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWREGION;
wire   [0:0] grp_run_vit_store_fu_60_m_axi_gmem1_AWUSER;
wire    grp_run_vit_store_fu_60_m_axi_gmem1_WVALID;
wire   [127:0] grp_run_vit_store_fu_60_m_axi_gmem1_WDATA;
wire   [15:0] grp_run_vit_store_fu_60_m_axi_gmem1_WSTRB;
wire    grp_run_vit_store_fu_60_m_axi_gmem1_WLAST;
wire   [0:0] grp_run_vit_store_fu_60_m_axi_gmem1_WID;
wire   [0:0] grp_run_vit_store_fu_60_m_axi_gmem1_WUSER;
wire    grp_run_vit_store_fu_60_m_axi_gmem1_ARVALID;
wire   [63:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARADDR;
wire   [0:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARID;
wire   [31:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARLEN;
wire   [2:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARBURST;
wire   [1:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARPROT;
wire   [3:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARQOS;
wire   [3:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARREGION;
wire   [0:0] grp_run_vit_store_fu_60_m_axi_gmem1_ARUSER;
wire    grp_run_vit_store_fu_60_m_axi_gmem1_RREADY;
wire    grp_run_vit_store_fu_60_m_axi_gmem1_BREADY;
wire    grp_run_vit_store_fu_60_rv_aq_stream_TREADY;
wire    grp_run_vit_store_fu_60_rv_as_stream_TREADY;
wire    grp_run_vit_replay_fu_72_ap_start;
wire    grp_run_vit_replay_fu_72_ap_done;
wire    grp_run_vit_replay_fu_72_ap_idle;
wire    grp_run_vit_replay_fu_72_ap_ready;
wire    grp_run_vit_replay_fu_72_m_axi_gmem1_AWVALID;
wire   [63:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWADDR;
wire   [0:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWID;
wire   [31:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWLEN;
wire   [2:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWSIZE;
wire   [1:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWBURST;
wire   [1:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWLOCK;
wire   [3:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWCACHE;
wire   [2:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWPROT;
wire   [3:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWQOS;
wire   [3:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWREGION;
wire   [0:0] grp_run_vit_replay_fu_72_m_axi_gmem1_AWUSER;
wire    grp_run_vit_replay_fu_72_m_axi_gmem1_WVALID;
wire   [127:0] grp_run_vit_replay_fu_72_m_axi_gmem1_WDATA;
wire   [15:0] grp_run_vit_replay_fu_72_m_axi_gmem1_WSTRB;
wire    grp_run_vit_replay_fu_72_m_axi_gmem1_WLAST;
wire   [0:0] grp_run_vit_replay_fu_72_m_axi_gmem1_WID;
wire   [0:0] grp_run_vit_replay_fu_72_m_axi_gmem1_WUSER;
wire    grp_run_vit_replay_fu_72_m_axi_gmem1_ARVALID;
wire   [63:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARADDR;
wire   [0:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARID;
wire   [31:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARLEN;
wire   [2:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARSIZE;
wire   [1:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARBURST;
wire   [1:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARLOCK;
wire   [3:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARCACHE;
wire   [2:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARPROT;
wire   [3:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARQOS;
wire   [3:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARREGION;
wire   [0:0] grp_run_vit_replay_fu_72_m_axi_gmem1_ARUSER;
wire    grp_run_vit_replay_fu_72_m_axi_gmem1_RREADY;
wire    grp_run_vit_replay_fu_72_m_axi_gmem1_BREADY;
wire    grp_run_vit_replay_fu_72_mux_aq_stream_TREADY;
wire    grp_run_vit_replay_fu_72_mux_as_stream_TREADY;
wire   [63:0] grp_run_vit_replay_fu_72_mux_aq_stream_TDATA;
wire    grp_run_vit_replay_fu_72_mux_aq_stream_TVALID;
wire   [7:0] grp_run_vit_replay_fu_72_mux_as_stream_TDATA;
wire    grp_run_vit_replay_fu_72_mux_as_stream_TVALID;
reg    gmem1_AWVALID;
wire    gmem1_AWREADY;
reg    gmem1_WVALID;
wire    gmem1_WREADY;
reg    gmem1_ARVALID;
wire    gmem1_ARREADY;
wire    gmem1_RVALID;
reg    gmem1_RREADY;
wire   [127:0] gmem1_RDATA;
wire   [2:0] gmem1_RFIFONUM;
wire    gmem1_BVALID;
reg    gmem1_BREADY;
reg    grp_run_vit_store_fu_60_ap_start_reg;
wire    ap_CS_fsm_state3;
reg    grp_run_vit_replay_fu_72_ap_start_reg;
wire    ap_CS_fsm_state4;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state6;
wire    regslice_both_mux_aq_stream_U_apdone_blk;
wire    regslice_both_mux_as_stream_U_apdone_blk;
reg    ap_block_state6;
reg   [5:0] ap_NS_fsm;
reg    ap_block_state1;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
reg    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
reg    ap_ST_fsm_state5_blk;
reg    ap_ST_fsm_state6_blk;
wire    regslice_both_rv_aq_stream_U_apdone_blk;
wire   [63:0] rv_aq_stream_TDATA_int_regslice;
wire    rv_aq_stream_TVALID_int_regslice;
reg    rv_aq_stream_TREADY_int_regslice;
wire    regslice_both_rv_aq_stream_U_ack_in;
wire    regslice_both_rv_as_stream_U_apdone_blk;
wire   [7:0] rv_as_stream_TDATA_int_regslice;
wire    rv_as_stream_TVALID_int_regslice;
reg    rv_as_stream_TREADY_int_regslice;
wire    regslice_both_rv_as_stream_U_ack_in;
wire    mux_aq_stream_TREADY_int_regslice;
wire    regslice_both_mux_aq_stream_U_vld_out;
wire    mux_as_stream_TREADY_int_regslice;
wire    regslice_both_mux_as_stream_U_vld_out;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_done_reg = 1'b0;
//#0 ap_CS_fsm = 6'd1;
//#0 grp_run_vit_store_fu_60_ap_start_reg = 1'b0;
//#0 grp_run_vit_replay_fu_72_ap_start_reg = 1'b0;
end

A_REORDER_run_vit_store grp_run_vit_store_fu_60(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_run_vit_store_fu_60_ap_start),
    .ap_done(grp_run_vit_store_fu_60_ap_done),
    .ap_idle(grp_run_vit_store_fu_60_ap_idle),
    .ap_ready(grp_run_vit_store_fu_60_ap_ready),
    .m_axi_gmem1_AWVALID(grp_run_vit_store_fu_60_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(gmem1_AWREADY),
    .m_axi_gmem1_AWADDR(grp_run_vit_store_fu_60_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_run_vit_store_fu_60_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_run_vit_store_fu_60_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_run_vit_store_fu_60_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_run_vit_store_fu_60_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_run_vit_store_fu_60_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_run_vit_store_fu_60_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_run_vit_store_fu_60_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_run_vit_store_fu_60_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_run_vit_store_fu_60_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_run_vit_store_fu_60_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_run_vit_store_fu_60_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(gmem1_WREADY),
    .m_axi_gmem1_WDATA(grp_run_vit_store_fu_60_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_run_vit_store_fu_60_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_run_vit_store_fu_60_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_run_vit_store_fu_60_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_run_vit_store_fu_60_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_run_vit_store_fu_60_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(1'b0),
    .m_axi_gmem1_ARADDR(grp_run_vit_store_fu_60_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_run_vit_store_fu_60_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_run_vit_store_fu_60_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_run_vit_store_fu_60_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_run_vit_store_fu_60_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_run_vit_store_fu_60_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_run_vit_store_fu_60_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_run_vit_store_fu_60_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_run_vit_store_fu_60_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_run_vit_store_fu_60_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_run_vit_store_fu_60_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(1'b0),
    .m_axi_gmem1_RREADY(grp_run_vit_store_fu_60_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(128'd0),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(3'd0),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(gmem1_BVALID),
    .m_axi_gmem1_BREADY(grp_run_vit_store_fu_60_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .memory_vit_a(memory_vit_a),
    .rv_aq_stream_TDATA(rv_aq_stream_TDATA_int_regslice),
    .rv_aq_stream_TVALID(rv_aq_stream_TVALID_int_regslice),
    .rv_aq_stream_TREADY(grp_run_vit_store_fu_60_rv_aq_stream_TREADY),
    .rv_as_stream_TDATA(rv_as_stream_TDATA_int_regslice),
    .rv_as_stream_TVALID(rv_as_stream_TVALID_int_regslice),
    .rv_as_stream_TREADY(grp_run_vit_store_fu_60_rv_as_stream_TREADY)
);

A_REORDER_run_vit_replay grp_run_vit_replay_fu_72(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_run_vit_replay_fu_72_ap_start),
    .ap_done(grp_run_vit_replay_fu_72_ap_done),
    .ap_idle(grp_run_vit_replay_fu_72_ap_idle),
    .ap_ready(grp_run_vit_replay_fu_72_ap_ready),
    .m_axi_gmem1_AWVALID(grp_run_vit_replay_fu_72_m_axi_gmem1_AWVALID),
    .m_axi_gmem1_AWREADY(1'b0),
    .m_axi_gmem1_AWADDR(grp_run_vit_replay_fu_72_m_axi_gmem1_AWADDR),
    .m_axi_gmem1_AWID(grp_run_vit_replay_fu_72_m_axi_gmem1_AWID),
    .m_axi_gmem1_AWLEN(grp_run_vit_replay_fu_72_m_axi_gmem1_AWLEN),
    .m_axi_gmem1_AWSIZE(grp_run_vit_replay_fu_72_m_axi_gmem1_AWSIZE),
    .m_axi_gmem1_AWBURST(grp_run_vit_replay_fu_72_m_axi_gmem1_AWBURST),
    .m_axi_gmem1_AWLOCK(grp_run_vit_replay_fu_72_m_axi_gmem1_AWLOCK),
    .m_axi_gmem1_AWCACHE(grp_run_vit_replay_fu_72_m_axi_gmem1_AWCACHE),
    .m_axi_gmem1_AWPROT(grp_run_vit_replay_fu_72_m_axi_gmem1_AWPROT),
    .m_axi_gmem1_AWQOS(grp_run_vit_replay_fu_72_m_axi_gmem1_AWQOS),
    .m_axi_gmem1_AWREGION(grp_run_vit_replay_fu_72_m_axi_gmem1_AWREGION),
    .m_axi_gmem1_AWUSER(grp_run_vit_replay_fu_72_m_axi_gmem1_AWUSER),
    .m_axi_gmem1_WVALID(grp_run_vit_replay_fu_72_m_axi_gmem1_WVALID),
    .m_axi_gmem1_WREADY(1'b0),
    .m_axi_gmem1_WDATA(grp_run_vit_replay_fu_72_m_axi_gmem1_WDATA),
    .m_axi_gmem1_WSTRB(grp_run_vit_replay_fu_72_m_axi_gmem1_WSTRB),
    .m_axi_gmem1_WLAST(grp_run_vit_replay_fu_72_m_axi_gmem1_WLAST),
    .m_axi_gmem1_WID(grp_run_vit_replay_fu_72_m_axi_gmem1_WID),
    .m_axi_gmem1_WUSER(grp_run_vit_replay_fu_72_m_axi_gmem1_WUSER),
    .m_axi_gmem1_ARVALID(grp_run_vit_replay_fu_72_m_axi_gmem1_ARVALID),
    .m_axi_gmem1_ARREADY(gmem1_ARREADY),
    .m_axi_gmem1_ARADDR(grp_run_vit_replay_fu_72_m_axi_gmem1_ARADDR),
    .m_axi_gmem1_ARID(grp_run_vit_replay_fu_72_m_axi_gmem1_ARID),
    .m_axi_gmem1_ARLEN(grp_run_vit_replay_fu_72_m_axi_gmem1_ARLEN),
    .m_axi_gmem1_ARSIZE(grp_run_vit_replay_fu_72_m_axi_gmem1_ARSIZE),
    .m_axi_gmem1_ARBURST(grp_run_vit_replay_fu_72_m_axi_gmem1_ARBURST),
    .m_axi_gmem1_ARLOCK(grp_run_vit_replay_fu_72_m_axi_gmem1_ARLOCK),
    .m_axi_gmem1_ARCACHE(grp_run_vit_replay_fu_72_m_axi_gmem1_ARCACHE),
    .m_axi_gmem1_ARPROT(grp_run_vit_replay_fu_72_m_axi_gmem1_ARPROT),
    .m_axi_gmem1_ARQOS(grp_run_vit_replay_fu_72_m_axi_gmem1_ARQOS),
    .m_axi_gmem1_ARREGION(grp_run_vit_replay_fu_72_m_axi_gmem1_ARREGION),
    .m_axi_gmem1_ARUSER(grp_run_vit_replay_fu_72_m_axi_gmem1_ARUSER),
    .m_axi_gmem1_RVALID(gmem1_RVALID),
    .m_axi_gmem1_RREADY(grp_run_vit_replay_fu_72_m_axi_gmem1_RREADY),
    .m_axi_gmem1_RDATA(gmem1_RDATA),
    .m_axi_gmem1_RLAST(1'b0),
    .m_axi_gmem1_RID(1'd0),
    .m_axi_gmem1_RFIFONUM(gmem1_RFIFONUM),
    .m_axi_gmem1_RUSER(1'd0),
    .m_axi_gmem1_RRESP(2'd0),
    .m_axi_gmem1_BVALID(1'b0),
    .m_axi_gmem1_BREADY(grp_run_vit_replay_fu_72_m_axi_gmem1_BREADY),
    .m_axi_gmem1_BRESP(2'd0),
    .m_axi_gmem1_BID(1'd0),
    .m_axi_gmem1_BUSER(1'd0),
    .mux_aq_stream_TREADY(grp_run_vit_replay_fu_72_mux_aq_stream_TREADY),
    .mux_as_stream_TREADY(grp_run_vit_replay_fu_72_mux_as_stream_TREADY),
    .memory_vit_a(memory_vit_a),
    .mux_aq_stream_TDATA(grp_run_vit_replay_fu_72_mux_aq_stream_TDATA),
    .mux_aq_stream_TVALID(grp_run_vit_replay_fu_72_mux_aq_stream_TVALID),
    .mux_as_stream_TDATA(grp_run_vit_replay_fu_72_mux_as_stream_TDATA),
    .mux_as_stream_TVALID(grp_run_vit_replay_fu_72_mux_as_stream_TVALID)
);

A_REORDER_gmem1_m_axi #(
    .CONSERVATIVE( 1 ),
    .USER_MAXREQS( 7 ),
    .MAX_READ_BURST_LENGTH( 2 ),
    .MAX_WRITE_BURST_LENGTH( 2 ),
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
    .USER_RFIFONUM_WIDTH( 3 ),
    .USER_DW( 128 ),
    .USER_AW( 64 ),
    .NUM_READ_OUTSTANDING( 2 ),
    .NUM_WRITE_OUTSTANDING( 2 ))
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
    .I_ARADDR(grp_run_vit_replay_fu_72_m_axi_gmem1_ARADDR),
    .I_ARLEN(grp_run_vit_replay_fu_72_m_axi_gmem1_ARLEN),
    .I_RVALID(gmem1_RVALID),
    .I_RREADY(gmem1_RREADY),
    .I_RDATA(gmem1_RDATA),
    .I_RFIFONUM(gmem1_RFIFONUM),
    .I_AWVALID(gmem1_AWVALID),
    .I_AWREADY(gmem1_AWREADY),
    .I_AWADDR(grp_run_vit_store_fu_60_m_axi_gmem1_AWADDR),
    .I_AWLEN(grp_run_vit_store_fu_60_m_axi_gmem1_AWLEN),
    .I_WVALID(gmem1_WVALID),
    .I_WREADY(gmem1_WREADY),
    .I_WDATA(grp_run_vit_store_fu_60_m_axi_gmem1_WDATA),
    .I_WSTRB(grp_run_vit_store_fu_60_m_axi_gmem1_WSTRB),
    .I_BVALID(gmem1_BVALID),
    .I_BREADY(gmem1_BREADY)
);

A_REORDER_regslice_both #(
    .DataWidth( 64 ))
regslice_both_rv_aq_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(rv_aq_stream_TDATA),
    .vld_in(rv_aq_stream_TVALID),
    .ack_in(regslice_both_rv_aq_stream_U_ack_in),
    .data_out(rv_aq_stream_TDATA_int_regslice),
    .vld_out(rv_aq_stream_TVALID_int_regslice),
    .ack_out(rv_aq_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_rv_aq_stream_U_apdone_blk)
);

A_REORDER_regslice_both #(
    .DataWidth( 8 ))
regslice_both_rv_as_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(rv_as_stream_TDATA),
    .vld_in(rv_as_stream_TVALID),
    .ack_in(regslice_both_rv_as_stream_U_ack_in),
    .data_out(rv_as_stream_TDATA_int_regslice),
    .vld_out(rv_as_stream_TVALID_int_regslice),
    .ack_out(rv_as_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_rv_as_stream_U_apdone_blk)
);

A_REORDER_regslice_both #(
    .DataWidth( 64 ))
regslice_both_mux_aq_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_run_vit_replay_fu_72_mux_aq_stream_TDATA),
    .vld_in(grp_run_vit_replay_fu_72_mux_aq_stream_TVALID),
    .ack_in(mux_aq_stream_TREADY_int_regslice),
    .data_out(mux_aq_stream_TDATA),
    .vld_out(regslice_both_mux_aq_stream_U_vld_out),
    .ack_out(mux_aq_stream_TREADY),
    .apdone_blk(regslice_both_mux_aq_stream_U_apdone_blk)
);

A_REORDER_regslice_both #(
    .DataWidth( 8 ))
regslice_both_mux_as_stream_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(grp_run_vit_replay_fu_72_mux_as_stream_TDATA),
    .vld_in(grp_run_vit_replay_fu_72_mux_as_stream_TVALID),
    .ack_in(mux_as_stream_TREADY_int_regslice),
    .data_out(mux_as_stream_TDATA),
    .vld_out(regslice_both_mux_as_stream_U_vld_out),
    .ack_out(mux_as_stream_TREADY),
    .apdone_blk(regslice_both_mux_as_stream_U_apdone_blk)
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
        end else if (((1'b0 == ap_block_state6) & (1'b1 == ap_CS_fsm_state6))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_run_vit_replay_fu_72_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state4)) begin
            grp_run_vit_replay_fu_72_ap_start_reg <= 1'b1;
        end else if ((grp_run_vit_replay_fu_72_ap_ready == 1'b1)) begin
            grp_run_vit_replay_fu_72_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_run_vit_store_fu_60_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state2)) begin
            grp_run_vit_store_fu_60_ap_start_reg <= 1'b1;
        end else if ((grp_run_vit_store_fu_60_ap_ready == 1'b1)) begin
            grp_run_vit_store_fu_60_ap_start_reg <= 1'b0;
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
    if ((grp_run_vit_store_fu_60_ap_done == 1'b0)) begin
        ap_ST_fsm_state3_blk = 1'b1;
    end else begin
        ap_ST_fsm_state3_blk = 1'b0;
    end
end

assign ap_ST_fsm_state4_blk = 1'b0;

always @ (*) begin
    if ((grp_run_vit_replay_fu_72_ap_done == 1'b0)) begin
        ap_ST_fsm_state5_blk = 1'b1;
    end else begin
        ap_ST_fsm_state5_blk = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state6)) begin
        ap_ST_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_fsm_state6_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_state6) & (1'b1 == ap_CS_fsm_state6))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = ap_done_reg;
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
    if (((1'b0 == ap_block_state6) & (1'b1 == ap_CS_fsm_state6))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state5) | (1'b1 == ap_CS_fsm_state4))) begin
        gmem1_ARVALID = grp_run_vit_replay_fu_72_m_axi_gmem1_ARVALID;
    end else begin
        gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) | (1'b1 == ap_CS_fsm_state3))) begin
        gmem1_AWVALID = grp_run_vit_store_fu_60_m_axi_gmem1_AWVALID;
    end else begin
        gmem1_AWVALID = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) | (1'b1 == ap_CS_fsm_state3))) begin
        gmem1_BREADY = grp_run_vit_store_fu_60_m_axi_gmem1_BREADY;
    end else begin
        gmem1_BREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state5) | (1'b1 == ap_CS_fsm_state4))) begin
        gmem1_RREADY = grp_run_vit_replay_fu_72_m_axi_gmem1_RREADY;
    end else begin
        gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) | (1'b1 == ap_CS_fsm_state3))) begin
        gmem1_WVALID = grp_run_vit_store_fu_60_m_axi_gmem1_WVALID;
    end else begin
        gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        rv_aq_stream_TREADY_int_regslice = grp_run_vit_store_fu_60_rv_aq_stream_TREADY;
    end else begin
        rv_aq_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        rv_as_stream_TREADY_int_regslice = grp_run_vit_store_fu_60_rv_as_stream_TREADY;
    end else begin
        rv_as_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b0 == ap_block_state1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_CS_fsm_state3) & (grp_run_vit_store_fu_60_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            if (((1'b1 == ap_CS_fsm_state5) & (grp_run_vit_replay_fu_72_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state6 : begin
            if (((1'b0 == ap_block_state6) & (1'b1 == ap_CS_fsm_state6))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
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

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

always @ (*) begin
    ap_block_state1 = ((ap_done_reg == 1'b1) | (ap_start == 1'b0));
end

always @ (*) begin
    ap_block_state6 = ((regslice_both_mux_as_stream_U_apdone_blk == 1'b1) | (regslice_both_mux_aq_stream_U_apdone_blk == 1'b1));
end

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign grp_run_vit_replay_fu_72_ap_start = grp_run_vit_replay_fu_72_ap_start_reg;

assign grp_run_vit_replay_fu_72_mux_aq_stream_TREADY = (mux_aq_stream_TREADY_int_regslice & ap_CS_fsm_state5);

assign grp_run_vit_replay_fu_72_mux_as_stream_TREADY = (mux_as_stream_TREADY_int_regslice & ap_CS_fsm_state5);

assign grp_run_vit_store_fu_60_ap_start = grp_run_vit_store_fu_60_ap_start_reg;

assign mux_aq_stream_TVALID = regslice_both_mux_aq_stream_U_vld_out;

assign mux_as_stream_TVALID = regslice_both_mux_as_stream_U_vld_out;

assign rv_aq_stream_TREADY = regslice_both_rv_aq_stream_U_ack_in;

assign rv_as_stream_TREADY = regslice_both_rv_as_stream_U_ack_in;

endmodule //A_REORDER
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

module A_REORDER_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        rv_aq_stream_TVALID,
        rv_as_stream_TVALID,
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
        sext_ln64,
        rv_aq_stream_TDATA,
        rv_aq_stream_TREADY,
        rv_as_stream_TDATA,
        rv_as_stream_TREADY
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input   rv_aq_stream_TVALID;
input   rv_as_stream_TVALID;
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
input  [2:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input  [59:0] sext_ln64;
input  [63:0] rv_aq_stream_TDATA;
output   rv_aq_stream_TREADY;
input  [7:0] rv_as_stream_TDATA;
output   rv_as_stream_TREADY;

reg ap_idle;
reg m_axi_gmem1_WVALID;
reg rv_aq_stream_TREADY;
reg rv_as_stream_TREADY;

reg   [0:0] ap_CS_iter0_fsm;
wire    ap_CS_iter0_fsm_state1;
reg   [1:0] ap_CS_iter1_fsm;
wire    ap_CS_iter1_fsm_state0;
wire   [0:0] icmp_ln64_fu_102_p2;
reg    ap_block_state1_pp0_stage0_iter0;
reg   [0:0] icmp_ln64_reg_152;
wire   [0:0] icmp_ln64_reg_152_pp0_iter0_reg;
reg    ap_block_state2_io;
wire    ap_CS_iter1_fsm_state2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem1_blk_n_W;
reg    rv_aq_stream_TDATA_blk_n;
reg    rv_as_stream_TDATA_blk_n;
reg   [63:0] rv_aq_stream_read_reg_156;
wire   [3:0] trunc_ln69_fu_114_p1;
reg   [3:0] trunc_ln69_reg_161;
reg   [16:0] indvar_flatten10_fu_60;
wire   [16:0] add_ln64_fu_108_p2;
wire    ap_loop_init;
reg   [16:0] ap_sig_allocacmp_indvar_flatten10_load;
wire   [67:0] pack_fu_129_p3;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg    ap_loop_exit_ready_pp0_iter1_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
reg    ap_ST_iter1_fsm_state2_blk;
wire    ap_start_int;
reg    ap_condition_112;
wire    ap_ce_reg;

// power-on initialization
initial begin
//#0 ap_CS_iter0_fsm = 1'd1;
//#0 ap_CS_iter1_fsm = 2'd1;
//#0 indvar_flatten10_fu_60 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

A_REORDER_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1) & (1'b0 == ap_block_state2_io))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state2_io) & (ap_loop_exit_ready == 1'b0))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_112)) begin
        if ((icmp_ln64_fu_102_p2 == 1'd0)) begin
            indvar_flatten10_fu_60 <= add_ln64_fu_108_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten10_fu_60 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        icmp_ln64_reg_152 <= icmp_ln64_fu_102_p2;
        rv_aq_stream_read_reg_156 <= rv_aq_stream_TDATA;
        trunc_ln69_reg_161 <= trunc_ln69_fu_114_p1;
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
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (icmp_ln64_fu_102_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state2) & (ap_loop_exit_ready_pp0_iter1_reg == 1'b1) & (1'b0 == ap_block_state2_io))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_iter0_fsm_state1) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten10_load = 17'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten10_load = indvar_flatten10_fu_60;
    end
end

always @ (*) begin
    if (((icmp_ln64_reg_152 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        gmem1_blk_n_W = m_axi_gmem1_WREADY;
    end else begin
        gmem1_blk_n_W = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln64_reg_152 == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state2_io))) begin
        m_axi_gmem1_WVALID = 1'b1;
    end else begin
        m_axi_gmem1_WVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln64_fu_102_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        rv_aq_stream_TDATA_blk_n = rv_aq_stream_TVALID;
    end else begin
        rv_aq_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (icmp_ln64_fu_102_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        rv_aq_stream_TREADY = 1'b1;
    end else begin
        rv_aq_stream_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln64_fu_102_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1) & (ap_start_int == 1'b1))) begin
        rv_as_stream_TDATA_blk_n = rv_as_stream_TVALID;
    end else begin
        rv_as_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (icmp_ln64_fu_102_p2 == 1'd0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        rv_as_stream_TREADY = 1'b1;
    end else begin
        rv_as_stream_TREADY = 1'b0;
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
            if (((1'b0 == ap_block_state2_io) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_block_state1_pp0_stage0_iter0))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else if ((((icmp_ln64_reg_152_pp0_iter0_reg == 1'd1) & (1'b1 == ap_CS_iter1_fsm_state2) & (1'b0 == ap_block_state2_io)) | ((1'b1 == ap_CS_iter0_fsm_state1) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b0 == ap_block_state2_io)))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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

assign add_ln64_fu_108_p2 = (ap_sig_allocacmp_indvar_flatten10_load + 17'd1);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter1_fsm_state0 = ap_CS_iter1_fsm[32'd0];

assign ap_CS_iter1_fsm_state2 = ap_CS_iter1_fsm[32'd1];

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = ((ap_start_int == 1'b0) | ((rv_as_stream_TVALID == 1'b0) & (icmp_ln64_fu_102_p2 == 1'd0)) | ((icmp_ln64_fu_102_p2 == 1'd0) & (rv_aq_stream_TVALID == 1'b0)));
end

always @ (*) begin
    ap_block_state2_io = ((icmp_ln64_reg_152 == 1'd0) & (m_axi_gmem1_WREADY == 1'b0));
end

always @ (*) begin
    ap_condition_112 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter1_fsm_state2) & (1'b1 == ap_block_state2_io))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign icmp_ln64_fu_102_p2 = ((ap_sig_allocacmp_indvar_flatten10_load == 17'd98304) ? 1'b1 : 1'b0);

assign icmp_ln64_reg_152_pp0_iter0_reg = icmp_ln64_reg_152;

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

assign m_axi_gmem1_RREADY = 1'b0;

assign m_axi_gmem1_WDATA = pack_fu_129_p3;

assign m_axi_gmem1_WID = 1'd0;

assign m_axi_gmem1_WLAST = 1'b0;

assign m_axi_gmem1_WSTRB = 16'd65535;

assign m_axi_gmem1_WUSER = 1'd0;

assign pack_fu_129_p3 = {{trunc_ln69_reg_161}, {rv_aq_stream_read_reg_156}};

assign trunc_ln69_fu_114_p1 = rv_as_stream_TDATA[3:0];

endmodule //A_REORDER_run_vit_store_Pipeline_VITIS_LOOP_64_1_VITIS_LOOP_65_2_VITIS_LOOP_66_3
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.2 (64-bit)
// Tool Version Limit: 2023.10
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================

`timescale 1 ns / 1 ps

module A_REORDER_flow_control_loop_pipe_sequential_init(
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

// if no ap_continue port and current module is not A_REORDER module, 
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

// if no ap_continue port and current module is not A_REORDER module, ap_done handshakes with ap_start
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

module A_REORDER_run_vit_replay (
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
        mux_aq_stream_TREADY,
        mux_as_stream_TREADY,
        memory_vit_a,
        mux_aq_stream_TDATA,
        mux_aq_stream_TVALID,
        mux_as_stream_TDATA,
        mux_as_stream_TVALID
);

parameter    ap_ST_iter0_fsm_state1 = 1'd1;
parameter    ap_ST_iter1_fsm_state2 = 2'd2;
parameter    ap_ST_iter2_fsm_state3 = 2'd2;
parameter    ap_ST_iter3_fsm_state4 = 2'd2;
parameter    ap_ST_iter4_fsm_state5 = 2'd2;
parameter    ap_ST_iter5_fsm_state6 = 2'd2;
parameter    ap_ST_iter6_fsm_state7 = 2'd2;
parameter    ap_ST_iter7_fsm_state8 = 2'd2;
parameter    ap_ST_iter8_fsm_state9 = 2'd2;
parameter    ap_ST_iter9_fsm_state10 = 2'd2;
parameter    ap_ST_iter10_fsm_state11 = 2'd2;
parameter    ap_ST_iter11_fsm_state12 = 2'd2;
parameter    ap_ST_iter12_fsm_state13 = 2'd2;
parameter    ap_ST_iter1_fsm_state0 = 2'd1;
parameter    ap_ST_iter2_fsm_state0 = 2'd1;
parameter    ap_ST_iter3_fsm_state0 = 2'd1;
parameter    ap_ST_iter4_fsm_state0 = 2'd1;
parameter    ap_ST_iter5_fsm_state0 = 2'd1;
parameter    ap_ST_iter6_fsm_state0 = 2'd1;
parameter    ap_ST_iter7_fsm_state0 = 2'd1;
parameter    ap_ST_iter8_fsm_state0 = 2'd1;
parameter    ap_ST_iter9_fsm_state0 = 2'd1;
parameter    ap_ST_iter10_fsm_state0 = 2'd1;
parameter    ap_ST_iter11_fsm_state0 = 2'd1;
parameter    ap_ST_iter12_fsm_state0 = 2'd1;

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
input  [2:0] m_axi_gmem1_RFIFONUM;
input  [0:0] m_axi_gmem1_RUSER;
input  [1:0] m_axi_gmem1_RRESP;
input   m_axi_gmem1_BVALID;
output   m_axi_gmem1_BREADY;
input  [1:0] m_axi_gmem1_BRESP;
input  [0:0] m_axi_gmem1_BID;
input  [0:0] m_axi_gmem1_BUSER;
input   mux_aq_stream_TREADY;
input   mux_as_stream_TREADY;
input  [63:0] memory_vit_a;
output  [63:0] mux_aq_stream_TDATA;
output   mux_aq_stream_TVALID;
output  [7:0] mux_as_stream_TDATA;
output   mux_as_stream_TVALID;

reg ap_idle;
reg m_axi_gmem1_ARVALID;
reg m_axi_gmem1_RREADY;
reg mux_aq_stream_TVALID;
reg mux_as_stream_TVALID;

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
reg   [1:0] ap_CS_iter8_fsm;
wire    ap_CS_iter8_fsm_state0;
reg   [1:0] ap_CS_iter9_fsm;
wire    ap_CS_iter9_fsm_state0;
reg   [1:0] ap_CS_iter10_fsm;
wire    ap_CS_iter10_fsm_state0;
reg   [1:0] ap_CS_iter11_fsm;
wire    ap_CS_iter11_fsm_state0;
reg   [1:0] ap_CS_iter12_fsm;
wire    ap_CS_iter12_fsm_state0;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_CS_iter1_fsm_state2;
wire    ap_CS_iter2_fsm_state3;
reg   [0:0] icmp_ln82_reg_551;
reg   [0:0] icmp_ln82_reg_551_pp0_iter2_reg;
reg   [0:0] or_ln84_reg_594;
reg   [0:0] and_ln84_reg_598;
reg    ap_predicate_op93_readreq_state4;
reg    ap_block_state4_io;
wire    ap_CS_iter3_fsm_state4;
wire    ap_CS_iter4_fsm_state5;
wire    ap_CS_iter5_fsm_state6;
wire    ap_CS_iter6_fsm_state7;
wire    ap_CS_iter7_fsm_state8;
wire    ap_CS_iter8_fsm_state9;
wire    ap_CS_iter9_fsm_state10;
wire    ap_CS_iter10_fsm_state11;
reg   [0:0] icmp_ln82_reg_551_pp0_iter10_reg;
reg    ap_block_state12_pp0_stage0_iter11;
wire    ap_CS_iter11_fsm_state12;
reg   [0:0] icmp_ln82_reg_551_pp0_iter11_reg;
reg    ap_block_state13_pp0_stage0_iter12;
reg    ap_block_state13_io;
wire    ap_CS_iter12_fsm_state13;
wire   [0:0] icmp_ln82_fu_212_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
reg    gmem1_blk_n_AR;
reg    gmem1_blk_n_R;
reg    mux_aq_stream_TDATA_blk_n;
reg    mux_as_stream_TDATA_blk_n;
reg   [0:0] first_iter_1_reg_152;
reg   [0:0] first_iter_0_reg_163;
wire   [0:0] icmp_ln82_reg_551_pp0_iter0_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter1_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter3_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter4_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter5_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter6_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter7_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter8_reg;
reg   [0:0] icmp_ln82_reg_551_pp0_iter9_reg;
wire   [0:0] icmp_ln83_fu_224_p2;
reg   [0:0] icmp_ln83_reg_555;
reg   [0:0] icmp_ln83_reg_555_pp0_iter1_reg;
wire   [0:0] xor_ln82_fu_270_p2;
reg   [0:0] xor_ln82_reg_567;
wire   [0:0] icmp_ln84_fu_275_p2;
reg   [0:0] icmp_ln84_reg_572;
wire   [0:0] and_ln82_1_fu_281_p2;
reg   [0:0] and_ln82_1_reg_577;
wire   [3:0] select_ln83_fu_306_p3;
reg   [3:0] select_ln83_reg_584;
wire   [6:0] empty_18_fu_314_p1;
reg   [6:0] empty_18_reg_589;
wire   [0:0] or_ln84_fu_402_p2;
wire   [0:0] and_ln84_fu_414_p2;
reg   [59:0] trunc_ln_reg_603;
reg   [3:0] s_bits_reg_614;
wire   [63:0] trunc_ln92_fu_496_p1;
reg   [63:0] trunc_ln92_reg_619;
reg   [0:0] ap_phi_mux_first_iter_1_phi_fu_156_p4;
wire    ap_loop_init;
reg   [0:0] ap_phi_mux_first_iter_0_phi_fu_168_p4;
wire  signed [63:0] sext_ln84_fu_476_p1;
reg   [3:0] hct_fu_96;
wire   [3:0] select_ln85_fu_463_p3;
reg   [7:0] indvar_flatten_fu_100;
wire   [7:0] select_ln84_fu_329_p3;
reg   [3:0] h_fu_104;
reg   [10:0] indvar_flatten12_fu_108;
wire   [10:0] select_ln83_1_fu_236_p3;
reg   [10:0] ap_sig_allocacmp_indvar_flatten12_load;
reg   [7:0] tt_fu_112;
wire   [7:0] select_ln82_1_fu_293_p3;
reg   [16:0] indvar_flatten34_fu_116;
wire   [16:0] add_ln82_fu_218_p2;
reg   [16:0] ap_sig_allocacmp_indvar_flatten34_load;
wire   [10:0] add_ln83_1_fu_230_p2;
wire   [7:0] add_ln82_1_fu_287_p2;
wire   [3:0] select_ln82_fu_263_p3;
wire   [3:0] add_ln83_fu_300_p2;
wire   [0:0] or_ln84_1_fu_324_p2;
wire   [7:0] add_ln84_fu_318_p2;
wire   [0:0] icmp_ln85_fu_365_p2;
wire   [0:0] or_ln82_fu_355_p2;
wire   [0:0] xor_ln83_fu_381_p2;
wire   [0:0] and_ln82_fu_371_p2;
wire   [0:0] or_ln83_2_fu_386_p2;
wire   [0:0] or_ln82_1_fu_360_p2;
wire   [0:0] or_ln83_1_fu_397_p2;
wire   [0:0] and_ln83_fu_391_p2;
wire   [0:0] or_ln83_fu_376_p2;
wire   [0:0] xor_ln84_fu_408_p2;
wire   [20:0] tmp_1_fu_420_p4;
wire   [63:0] p_cast2_fu_428_p1;
wire   [63:0] empty_19_fu_432_p2;
wire   [0:0] or_ln85_fu_453_p2;
wire   [0:0] or_ln85_1_fu_458_p2;
wire   [3:0] add_ln85_fu_447_p2;
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
reg    ap_loop_exit_ready_pp0_iter8_reg;
reg    ap_loop_exit_ready_pp0_iter9_reg;
reg    ap_loop_exit_ready_pp0_iter10_reg;
reg    ap_loop_exit_ready_pp0_iter11_reg;
reg    ap_loop_exit_ready_pp0_iter12_reg;
reg   [0:0] ap_NS_iter0_fsm;
reg   [1:0] ap_NS_iter1_fsm;
reg   [1:0] ap_NS_iter2_fsm;
reg   [1:0] ap_NS_iter3_fsm;
reg   [1:0] ap_NS_iter4_fsm;
reg   [1:0] ap_NS_iter5_fsm;
reg   [1:0] ap_NS_iter6_fsm;
reg   [1:0] ap_NS_iter7_fsm;
reg   [1:0] ap_NS_iter8_fsm;
reg   [1:0] ap_NS_iter9_fsm;
reg   [1:0] ap_NS_iter10_fsm;
reg   [1:0] ap_NS_iter11_fsm;
reg   [1:0] ap_NS_iter12_fsm;
reg    ap_ST_iter0_fsm_state1_blk;
wire    ap_ST_iter1_fsm_state2_blk;
wire    ap_ST_iter2_fsm_state3_blk;
reg    ap_ST_iter3_fsm_state4_blk;
wire    ap_ST_iter4_fsm_state5_blk;
wire    ap_ST_iter5_fsm_state6_blk;
wire    ap_ST_iter6_fsm_state7_blk;
wire    ap_ST_iter7_fsm_state8_blk;
wire    ap_ST_iter8_fsm_state9_blk;
wire    ap_ST_iter9_fsm_state10_blk;
wire    ap_ST_iter10_fsm_state11_blk;
reg    ap_ST_iter11_fsm_state12_blk;
reg    ap_ST_iter12_fsm_state13_blk;
wire    ap_start_int;
reg    ap_condition_282;
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
//#0 ap_CS_iter8_fsm = 2'd1;
//#0 ap_CS_iter9_fsm = 2'd1;
//#0 ap_CS_iter10_fsm = 2'd1;
//#0 ap_CS_iter11_fsm = 2'd1;
//#0 ap_CS_iter12_fsm = 2'd1;
//#0 hct_fu_96 = 4'd0;
//#0 indvar_flatten_fu_100 = 8'd0;
//#0 h_fu_104 = 4'd0;
//#0 indvar_flatten12_fu_108 = 11'd0;
//#0 tt_fu_112 = 8'd0;
//#0 indvar_flatten34_fu_116 = 17'd0;
//#0 ap_done_reg = 1'b0;
end

A_REORDER_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
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
        ap_CS_iter10_fsm <= ap_ST_iter10_fsm_state0;
    end else begin
        ap_CS_iter10_fsm <= ap_NS_iter10_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter11_fsm <= ap_ST_iter11_fsm_state0;
    end else begin
        ap_CS_iter11_fsm <= ap_NS_iter11_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter12_fsm <= ap_ST_iter12_fsm_state0;
    end else begin
        ap_CS_iter12_fsm <= ap_NS_iter12_fsm;
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
        ap_CS_iter8_fsm <= ap_ST_iter8_fsm_state0;
    end else begin
        ap_CS_iter8_fsm <= ap_NS_iter8_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_iter9_fsm <= ap_ST_iter9_fsm_state0;
    end else begin
        ap_CS_iter9_fsm <= ap_NS_iter9_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if ((~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & (ap_loop_exit_ready_pp0_iter12_reg == 1'b1) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & (ap_loop_exit_ready_pp0_iter11_reg == 1'b0) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
        ap_loop_exit_ready_pp0_iter12_reg <= 1'b0;
    end else if ((~((1'b1 == ap_block_state12_pp0_stage0_iter11) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)))) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
        ap_loop_exit_ready_pp0_iter12_reg <= ap_loop_exit_ready_pp0_iter11_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (icmp_ln82_reg_551_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        first_iter_0_reg_163 <= 1'd0;
    end else if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        first_iter_0_reg_163 <= 1'd1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (icmp_ln82_reg_551_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        first_iter_1_reg_152 <= and_ln84_reg_598;
    end else if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        first_iter_1_reg_152 <= 1'd1;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        h_fu_104 <= 4'd0;
    end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (icmp_ln82_reg_551_pp0_iter0_reg == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        h_fu_104 <= select_ln83_fu_306_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        hct_fu_96 <= 4'd0;
    end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (icmp_ln82_reg_551_pp0_iter1_reg == 1'd0) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        hct_fu_96 <= select_ln85_fu_463_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_282)) begin
        if ((icmp_ln82_fu_212_p2 == 1'd0)) begin
            indvar_flatten12_fu_108 <= select_ln83_1_fu_236_p3;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten12_fu_108 <= 11'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_condition_282)) begin
        if ((icmp_ln82_fu_212_p2 == 1'd0)) begin
            indvar_flatten34_fu_116 <= add_ln82_fu_218_p2;
        end else if ((ap_loop_init == 1'b1)) begin
            indvar_flatten34_fu_116 <= 17'd0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        indvar_flatten_fu_100 <= 8'd0;
    end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (icmp_ln82_reg_551_pp0_iter0_reg == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        indvar_flatten_fu_100 <= select_ln84_fu_329_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        tt_fu_112 <= 8'd0;
    end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (icmp_ln82_reg_551_pp0_iter0_reg == 1'd0) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        tt_fu_112 <= select_ln82_1_fu_293_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
        and_ln82_1_reg_577 <= and_ln82_1_fu_281_p2;
        ap_loop_exit_ready_pp0_iter2_reg <= ap_loop_exit_ready_pp0_iter1_reg;
        empty_18_reg_589 <= empty_18_fu_314_p1;
        icmp_ln82_reg_551_pp0_iter1_reg <= icmp_ln82_reg_551;
        icmp_ln83_reg_555_pp0_iter1_reg <= icmp_ln83_reg_555;
        icmp_ln84_reg_572 <= icmp_ln84_fu_275_p2;
        select_ln83_reg_584 <= select_ln83_fu_306_p3;
        xor_ln82_reg_567 <= xor_ln82_fu_270_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
        and_ln84_reg_598 <= and_ln84_fu_414_p2;
        ap_loop_exit_ready_pp0_iter3_reg <= ap_loop_exit_ready_pp0_iter2_reg;
        icmp_ln82_reg_551_pp0_iter2_reg <= icmp_ln82_reg_551_pp0_iter1_reg;
        or_ln84_reg_594 <= or_ln84_fu_402_p2;
        trunc_ln_reg_603 <= {{empty_19_fu_432_p2[63:4]}};
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter9_fsm_state10))) begin
        ap_loop_exit_ready_pp0_iter10_reg <= ap_loop_exit_ready_pp0_iter9_reg;
        icmp_ln82_reg_551_pp0_iter9_reg <= icmp_ln82_reg_551_pp0_iter8_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter10_fsm_state11))) begin
        ap_loop_exit_ready_pp0_iter11_reg <= ap_loop_exit_ready_pp0_iter10_reg;
        icmp_ln82_reg_551_pp0_iter10_reg <= icmp_ln82_reg_551_pp0_iter9_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_loop_exit_ready_pp0_iter1_reg <= ap_loop_exit_ready;
        icmp_ln82_reg_551 <= icmp_ln82_fu_212_p2;
        icmp_ln83_reg_555 <= icmp_ln83_fu_224_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_loop_exit_ready_pp0_iter4_reg <= ap_loop_exit_ready_pp0_iter3_reg;
        icmp_ln82_reg_551_pp0_iter3_reg <= icmp_ln82_reg_551_pp0_iter2_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
        ap_loop_exit_ready_pp0_iter5_reg <= ap_loop_exit_ready_pp0_iter4_reg;
        icmp_ln82_reg_551_pp0_iter4_reg <= icmp_ln82_reg_551_pp0_iter3_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
        ap_loop_exit_ready_pp0_iter6_reg <= ap_loop_exit_ready_pp0_iter5_reg;
        icmp_ln82_reg_551_pp0_iter5_reg <= icmp_ln82_reg_551_pp0_iter4_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
        ap_loop_exit_ready_pp0_iter7_reg <= ap_loop_exit_ready_pp0_iter6_reg;
        icmp_ln82_reg_551_pp0_iter6_reg <= icmp_ln82_reg_551_pp0_iter5_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter7_fsm_state8))) begin
        ap_loop_exit_ready_pp0_iter8_reg <= ap_loop_exit_ready_pp0_iter7_reg;
        icmp_ln82_reg_551_pp0_iter7_reg <= icmp_ln82_reg_551_pp0_iter6_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter8_fsm_state9))) begin
        ap_loop_exit_ready_pp0_iter9_reg <= ap_loop_exit_ready_pp0_iter8_reg;
        icmp_ln82_reg_551_pp0_iter8_reg <= icmp_ln82_reg_551_pp0_iter7_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((~((1'b1 == ap_block_state12_pp0_stage0_iter11) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)))) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
        icmp_ln82_reg_551_pp0_iter11_reg <= icmp_ln82_reg_551_pp0_iter10_reg;
        s_bits_reg_614 <= {{m_axi_gmem1_RDATA[67:64]}};
        trunc_ln92_reg_619 <= trunc_ln92_fu_496_p1;
    end
end

always @ (*) begin
    if ((1'b1 == ap_block_state1_pp0_stage0_iter0)) begin
        ap_ST_iter0_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_iter0_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_iter10_fsm_state11_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state12_pp0_stage0_iter11)) begin
        ap_ST_iter11_fsm_state12_blk = 1'b1;
    end else begin
        ap_ST_iter11_fsm_state12_blk = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) begin
        ap_ST_iter12_fsm_state13_blk = 1'b1;
    end else begin
        ap_ST_iter12_fsm_state13_blk = 1'b0;
    end
end

assign ap_ST_iter1_fsm_state2_blk = 1'b0;

assign ap_ST_iter2_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((1'b1 == ap_block_state4_io)) begin
        ap_ST_iter3_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_iter3_fsm_state4_blk = 1'b0;
    end
end

assign ap_ST_iter4_fsm_state5_blk = 1'b0;

assign ap_ST_iter5_fsm_state6_blk = 1'b0;

assign ap_ST_iter6_fsm_state7_blk = 1'b0;

assign ap_ST_iter7_fsm_state8_blk = 1'b0;

assign ap_ST_iter8_fsm_state9_blk = 1'b0;

assign ap_ST_iter9_fsm_state10_blk = 1'b0;

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (icmp_ln82_fu_212_p2 == 1'd1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & (ap_loop_exit_ready_pp0_iter12_reg == 1'b1) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if (((ap_start_int == 1'b0) & (1'b1 == ap_CS_iter7_fsm_state0) & (1'b1 == ap_CS_iter6_fsm_state0) & (1'b1 == ap_CS_iter5_fsm_state0) & (1'b1 == ap_CS_iter4_fsm_state0) & (1'b1 == ap_CS_iter3_fsm_state0) & (1'b1 == ap_CS_iter2_fsm_state0) & (1'b1 == ap_CS_iter1_fsm_state0) & (1'b1 == ap_CS_iter0_fsm_state1) & (1'b1 == ap_CS_iter12_fsm_state0) & (1'b1 == ap_CS_iter11_fsm_state0) & (1'b1 == ap_CS_iter10_fsm_state0) & (1'b1 == ap_CS_iter9_fsm_state0) & (1'b1 == ap_CS_iter8_fsm_state0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln82_reg_551_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_phi_mux_first_iter_0_phi_fu_168_p4 = 1'd0;
    end else begin
        ap_phi_mux_first_iter_0_phi_fu_168_p4 = first_iter_0_reg_163;
    end
end

always @ (*) begin
    if (((icmp_ln82_reg_551_pp0_iter2_reg == 1'd0) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        ap_phi_mux_first_iter_1_phi_fu_156_p4 = and_ln84_reg_598;
    end else begin
        ap_phi_mux_first_iter_1_phi_fu_156_p4 = first_iter_1_reg_152;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_sig_allocacmp_indvar_flatten12_load = 11'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten12_load = indvar_flatten12_fu_108;
    end
end

always @ (*) begin
    if (((ap_loop_init == 1'b1) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
        ap_sig_allocacmp_indvar_flatten34_load = 17'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten34_load = indvar_flatten34_fu_116;
    end
end

always @ (*) begin
    if (((ap_predicate_op93_readreq_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        gmem1_blk_n_AR = m_axi_gmem1_ARREADY;
    end else begin
        gmem1_blk_n_AR = 1'b1;
    end
end

always @ (*) begin
    if (((icmp_ln82_reg_551_pp0_iter10_reg == 1'd0) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
        gmem1_blk_n_R = m_axi_gmem1_RVALID;
    end else begin
        gmem1_blk_n_R = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (ap_predicate_op93_readreq_state4 == 1'b1) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
        m_axi_gmem1_ARVALID = 1'b1;
    end else begin
        m_axi_gmem1_ARVALID = 1'b0;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state12_pp0_stage0_iter11) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)))) & (icmp_ln82_reg_551_pp0_iter10_reg == 1'd0) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
        m_axi_gmem1_RREADY = 1'b1;
    end else begin
        m_axi_gmem1_RREADY = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln82_reg_551_pp0_iter11_reg == 1'd0) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
        mux_aq_stream_TDATA_blk_n = mux_aq_stream_TREADY;
    end else begin
        mux_aq_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & (icmp_ln82_reg_551_pp0_iter11_reg == 1'd0) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
        mux_aq_stream_TVALID = 1'b1;
    end else begin
        mux_aq_stream_TVALID = 1'b0;
    end
end

always @ (*) begin
    if (((icmp_ln82_reg_551_pp0_iter11_reg == 1'd0) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
        mux_as_stream_TDATA_blk_n = mux_as_stream_TREADY;
    end else begin
        mux_as_stream_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & (icmp_ln82_reg_551_pp0_iter11_reg == 1'd0) & (1'b1 == ap_CS_iter12_fsm_state13))) begin
        mux_as_stream_TVALID = 1'b1;
    end else begin
        mux_as_stream_TVALID = 1'b0;
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
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b0 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & ((1'b0 == ap_CS_iter0_fsm_state1) | ((1'b1 == ap_block_state1_pp0_stage0_iter0) & (1'b1 == ap_CS_iter0_fsm_state1))))) begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state0;
            end else begin
                ap_NS_iter1_fsm = ap_ST_iter1_fsm_state2;
            end
        end
        ap_ST_iter1_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter0_fsm_state1))) begin
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
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b0 == ap_CS_iter1_fsm_state2))) begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state0;
            end else begin
                ap_NS_iter2_fsm = ap_ST_iter2_fsm_state3;
            end
        end
        ap_ST_iter2_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter1_fsm_state2))) begin
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
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end else if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_CS_iter2_fsm_state3))) begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state0;
            end else begin
                ap_NS_iter3_fsm = ap_ST_iter3_fsm_state4;
            end
        end
        ap_ST_iter3_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter2_fsm_state3))) begin
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
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & ((1'b0 == ap_CS_iter3_fsm_state4) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))))) begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state0;
            end else begin
                ap_NS_iter4_fsm = ap_ST_iter4_fsm_state5;
            end
        end
        ap_ST_iter4_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state4_io) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter3_fsm_state4))) begin
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
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_CS_iter4_fsm_state5))) begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state0;
            end else begin
                ap_NS_iter5_fsm = ap_ST_iter5_fsm_state6;
            end
        end
        ap_ST_iter5_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter4_fsm_state5))) begin
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
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_CS_iter5_fsm_state6))) begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state0;
            end else begin
                ap_NS_iter6_fsm = ap_ST_iter6_fsm_state7;
            end
        end
        ap_ST_iter6_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter5_fsm_state6))) begin
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
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_CS_iter6_fsm_state7))) begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state0;
            end else begin
                ap_NS_iter7_fsm = ap_ST_iter7_fsm_state8;
            end
        end
        ap_ST_iter7_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter6_fsm_state7))) begin
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

always @ (*) begin
    case (ap_CS_iter8_fsm)
        ap_ST_iter8_fsm_state9 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter7_fsm_state8))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state9;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_CS_iter7_fsm_state8))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state0;
            end else begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state9;
            end
        end
        ap_ST_iter8_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter7_fsm_state8))) begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state9;
            end else begin
                ap_NS_iter8_fsm = ap_ST_iter8_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter8_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter9_fsm)
        ap_ST_iter9_fsm_state10 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter8_fsm_state9))) begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state10;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_CS_iter8_fsm_state9))) begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state0;
            end else begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state10;
            end
        end
        ap_ST_iter9_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter8_fsm_state9))) begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state10;
            end else begin
                ap_NS_iter9_fsm = ap_ST_iter9_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter9_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter10_fsm)
        ap_ST_iter10_fsm_state11 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter9_fsm_state10))) begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state11;
            end else if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b0 == ap_CS_iter9_fsm_state10))) begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state0;
            end else begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state11;
            end
        end
        ap_ST_iter10_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter9_fsm_state10))) begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state11;
            end else begin
                ap_NS_iter10_fsm = ap_ST_iter10_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter10_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter11_fsm)
        ap_ST_iter11_fsm_state12 : begin
            if ((~((1'b1 == ap_block_state12_pp0_stage0_iter11) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)))) & (1'b1 == ap_CS_iter10_fsm_state11))) begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state12;
            end else if ((~((1'b1 == ap_block_state12_pp0_stage0_iter11) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)))) & (1'b0 == ap_CS_iter10_fsm_state11))) begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state0;
            end else begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state12;
            end
        end
        ap_ST_iter11_fsm_state0 : begin
            if ((~(((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))) & (1'b1 == ap_CS_iter10_fsm_state11))) begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state12;
            end else begin
                ap_NS_iter11_fsm = ap_ST_iter11_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter11_fsm = 'bx;
        end
    endcase
end

always @ (*) begin
    case (ap_CS_iter12_fsm)
        ap_ST_iter12_fsm_state13 : begin
            if ((~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & ((1'b0 == ap_CS_iter11_fsm_state12) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12))))) begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state0;
            end else if (((~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & (1'b0 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | (~((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)) & (icmp_ln82_reg_551_pp0_iter11_reg == 1'd1) & (1'b1 == ap_CS_iter12_fsm_state13)))) begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state13;
            end else begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state13;
            end
        end
        ap_ST_iter12_fsm_state0 : begin
            if ((~((1'b1 == ap_block_state12_pp0_stage0_iter11) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12)))) & (1'b1 == ap_CS_iter11_fsm_state12))) begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state13;
            end else begin
                ap_NS_iter12_fsm = ap_ST_iter12_fsm_state0;
            end
        end
        default : begin
            ap_NS_iter12_fsm = 'bx;
        end
    endcase
end

assign add_ln82_1_fu_287_p2 = (tt_fu_112 + 8'd1);

assign add_ln82_fu_218_p2 = (ap_sig_allocacmp_indvar_flatten34_load + 17'd1);

assign add_ln83_1_fu_230_p2 = (ap_sig_allocacmp_indvar_flatten12_load + 11'd1);

assign add_ln83_fu_300_p2 = (select_ln82_fu_263_p3 + 4'd1);

assign add_ln84_fu_318_p2 = (indvar_flatten_fu_100 + 8'd1);

assign add_ln85_fu_447_p2 = (hct_fu_96 + 4'd1);

assign and_ln82_1_fu_281_p2 = (xor_ln82_fu_270_p2 & icmp_ln84_fu_275_p2);

assign and_ln82_fu_371_p2 = (xor_ln82_reg_567 & icmp_ln85_fu_365_p2);

assign and_ln83_fu_391_p2 = (or_ln83_2_fu_386_p2 & and_ln82_fu_371_p2);

assign and_ln84_fu_414_p2 = (xor_ln84_fu_408_p2 & or_ln83_fu_376_p2);

assign ap_CS_iter0_fsm_state1 = ap_CS_iter0_fsm[32'd0];

assign ap_CS_iter10_fsm_state0 = ap_CS_iter10_fsm[32'd0];

assign ap_CS_iter10_fsm_state11 = ap_CS_iter10_fsm[32'd1];

assign ap_CS_iter11_fsm_state0 = ap_CS_iter11_fsm[32'd0];

assign ap_CS_iter11_fsm_state12 = ap_CS_iter11_fsm[32'd1];

assign ap_CS_iter12_fsm_state0 = ap_CS_iter12_fsm[32'd0];

assign ap_CS_iter12_fsm_state13 = ap_CS_iter12_fsm[32'd1];

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

assign ap_CS_iter8_fsm_state0 = ap_CS_iter8_fsm[32'd0];

assign ap_CS_iter8_fsm_state9 = ap_CS_iter8_fsm[32'd1];

assign ap_CS_iter9_fsm_state0 = ap_CS_iter9_fsm[32'd0];

assign ap_CS_iter9_fsm_state10 = ap_CS_iter9_fsm[32'd1];

always @ (*) begin
    ap_block_state12_pp0_stage0_iter11 = ((icmp_ln82_reg_551_pp0_iter10_reg == 1'd0) & (m_axi_gmem1_RVALID == 1'b0));
end

always @ (*) begin
    ap_block_state13_io = (((mux_as_stream_TREADY == 1'b0) & (icmp_ln82_reg_551_pp0_iter11_reg == 1'd0)) | ((icmp_ln82_reg_551_pp0_iter11_reg == 1'd0) & (mux_aq_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state13_pp0_stage0_iter12 = (((mux_as_stream_TREADY == 1'b0) & (icmp_ln82_reg_551_pp0_iter11_reg == 1'd0)) | ((icmp_ln82_reg_551_pp0_iter11_reg == 1'd0) & (mux_aq_stream_TREADY == 1'b0)));
end

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start_int == 1'b0);
end

always @ (*) begin
    ap_block_state4_io = ((ap_predicate_op93_readreq_state4 == 1'b1) & (m_axi_gmem1_ARREADY == 1'b0));
end

always @ (*) begin
    ap_condition_282 = (~((1'b1 == ap_block_state1_pp0_stage0_iter0) | ((1'b1 == ap_CS_iter12_fsm_state13) & ((1'b1 == ap_block_state13_io) | (1'b1 == ap_block_state13_pp0_stage0_iter12))) | ((1'b1 == ap_block_state12_pp0_stage0_iter11) & (1'b1 == ap_CS_iter11_fsm_state12)) | ((1'b1 == ap_block_state4_io) & (1'b1 == ap_CS_iter3_fsm_state4))) & (1'b1 == ap_CS_iter0_fsm_state1));
end

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

always @ (*) begin
    ap_predicate_op93_readreq_state4 = ((1'd1 == and_ln84_reg_598) & (or_ln84_reg_594 == 1'd1) & (icmp_ln82_reg_551_pp0_iter2_reg == 1'd0));
end

assign empty_18_fu_314_p1 = select_ln82_1_fu_293_p3[6:0];

assign empty_19_fu_432_p2 = (p_cast2_fu_428_p1 + memory_vit_a);

assign icmp_ln82_fu_212_p2 = ((ap_sig_allocacmp_indvar_flatten34_load == 17'd98304) ? 1'b1 : 1'b0);

assign icmp_ln82_reg_551_pp0_iter0_reg = icmp_ln82_reg_551;

assign icmp_ln83_fu_224_p2 = ((ap_sig_allocacmp_indvar_flatten12_load == 11'd768) ? 1'b1 : 1'b0);

assign icmp_ln84_fu_275_p2 = ((indvar_flatten_fu_100 == 8'd64) ? 1'b1 : 1'b0);

assign icmp_ln85_fu_365_p2 = ((hct_fu_96 == 4'd8) ? 1'b1 : 1'b0);

assign m_axi_gmem1_ARADDR = sext_ln84_fu_476_p1;

assign m_axi_gmem1_ARBURST = 2'd0;

assign m_axi_gmem1_ARCACHE = 4'd0;

assign m_axi_gmem1_ARID = 1'd0;

assign m_axi_gmem1_ARLEN = 32'd64;

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

assign mux_aq_stream_TDATA = trunc_ln92_reg_619;

assign mux_as_stream_TDATA = s_bits_reg_614;

assign or_ln82_1_fu_360_p2 = (icmp_ln83_reg_555_pp0_iter1_reg | ap_phi_mux_first_iter_0_phi_fu_168_p4);

assign or_ln82_fu_355_p2 = (icmp_ln83_reg_555_pp0_iter1_reg | ap_phi_mux_first_iter_1_phi_fu_156_p4);

assign or_ln83_1_fu_397_p2 = (or_ln82_1_fu_360_p2 | and_ln82_1_reg_577);

assign or_ln83_2_fu_386_p2 = (xor_ln83_fu_381_p2 | icmp_ln83_reg_555_pp0_iter1_reg);

assign or_ln83_fu_376_p2 = (or_ln82_fu_355_p2 | and_ln82_1_reg_577);

assign or_ln84_1_fu_324_p2 = (icmp_ln83_reg_555 | and_ln82_1_fu_281_p2);

assign or_ln84_fu_402_p2 = (or_ln83_1_fu_397_p2 | and_ln83_fu_391_p2);

assign or_ln85_1_fu_458_p2 = (or_ln85_fu_453_p2 | icmp_ln83_reg_555_pp0_iter1_reg);

assign or_ln85_fu_453_p2 = (and_ln83_fu_391_p2 | and_ln82_1_reg_577);

assign p_cast2_fu_428_p1 = tmp_1_fu_420_p4;

assign select_ln82_1_fu_293_p3 = ((icmp_ln83_reg_555[0:0] == 1'b1) ? add_ln82_1_fu_287_p2 : tt_fu_112);

assign select_ln82_fu_263_p3 = ((icmp_ln83_reg_555[0:0] == 1'b1) ? 4'd0 : h_fu_104);

assign select_ln83_1_fu_236_p3 = ((icmp_ln83_fu_224_p2[0:0] == 1'b1) ? 11'd1 : add_ln83_1_fu_230_p2);

assign select_ln83_fu_306_p3 = ((and_ln82_1_fu_281_p2[0:0] == 1'b1) ? add_ln83_fu_300_p2 : select_ln82_fu_263_p3);

assign select_ln84_fu_329_p3 = ((or_ln84_1_fu_324_p2[0:0] == 1'b1) ? 8'd1 : add_ln84_fu_318_p2);

assign select_ln85_fu_463_p3 = ((or_ln85_1_fu_458_p2[0:0] == 1'b1) ? 4'd1 : add_ln85_fu_447_p2);

assign sext_ln84_fu_476_p1 = $signed(trunc_ln_reg_603);

assign tmp_1_fu_420_p4 = {{{select_ln83_reg_584}, {empty_18_reg_589}}, {10'd0}};

assign trunc_ln92_fu_496_p1 = m_axi_gmem1_RDATA[63:0];

assign xor_ln82_fu_270_p2 = (icmp_ln83_reg_555 ^ 1'd1);

assign xor_ln83_fu_381_p2 = (icmp_ln84_reg_572 ^ 1'd1);

assign xor_ln84_fu_408_p2 = (1'd1 ^ and_ln83_fu_391_p2);

endmodule //A_REORDER_run_vit_replay
// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================
// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

`timescale 1ns/1ps
`default_nettype none

module A_REORDER_gmem1_m_axi
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
    // A_REORDER_gmem1_m_axi_store
    A_REORDER_gmem1_m_axi_store #(
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

    // A_REORDER_gmem1_m_axi_load
    A_REORDER_gmem1_m_axi_load #(
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

    // A_REORDER_gmem1_m_axi_write
    A_REORDER_gmem1_m_axi_write #(
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

    // A_REORDER_gmem1_m_axi_read
    A_REORDER_gmem1_m_axi_read #(
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

module A_REORDER_gmem1_m_axi_load
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

    

    A_REORDER_gmem1_m_axi_fifo #(
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

    

    A_REORDER_gmem1_m_axi_fifo #(
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
        A_REORDER_gmem1_m_axi_fifo #(
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


module A_REORDER_gmem1_m_axi_store
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
    

    A_REORDER_gmem1_m_axi_fifo #(
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

    

    A_REORDER_gmem1_m_axi_fifo #(
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
        A_REORDER_gmem1_m_axi_fifo #(
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
        A_REORDER_gmem1_m_axi_fifo #(
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
    A_REORDER_gmem1_m_axi_fifo #(
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

    A_REORDER_gmem1_m_axi_fifo #(
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


module A_REORDER_gmem1_m_axi_read
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
    A_REORDER_gmem1_m_axi_burst_converter #(
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
    A_REORDER_gmem1_m_axi_reg_slice #(
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

    A_REORDER_gmem1_m_axi_fifo #(
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

    A_REORDER_gmem1_m_axi_fifo #(
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

module A_REORDER_gmem1_m_axi_write
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
    A_REORDER_gmem1_m_axi_burst_converter #(
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

    A_REORDER_gmem1_m_axi_fifo #(
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
    A_REORDER_gmem1_m_axi_throttle #(
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
    A_REORDER_gmem1_m_axi_reg_slice #(
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

    A_REORDER_gmem1_m_axi_fifo #(
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


module A_REORDER_gmem1_m_axi_burst_converter
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
    A_REORDER_gmem1_m_axi_reg_slice #(
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

module A_REORDER_gmem1_m_axi_throttle
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

        A_REORDER_gmem1_m_axi_reg_slice #(
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

        A_REORDER_gmem1_m_axi_fifo #(
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
            
        A_REORDER_gmem1_m_axi_fifo #(
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



module A_REORDER_gmem1_m_axi_reg_slice
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

module A_REORDER_gmem1_m_axi_fifo
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

        A_REORDER_gmem1_m_axi_srl
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

        A_REORDER_gmem1_m_axi_mem
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

module A_REORDER_gmem1_m_axi_srl
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

module A_REORDER_gmem1_m_axi_mem
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
